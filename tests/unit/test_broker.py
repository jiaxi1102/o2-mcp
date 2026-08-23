"""Offline tests for the persistent, dynamically framed O2 command broker.

The integration cases run the exact embedded remote helper as a local Python
child. They therefore exercise one long-lived byte channel, Unix-socket clients,
policy enforcement, command timeouts, and lifecycle cleanup without SSH, O2, or
Duo access.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import plistlib
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path

import pytest

from o2mcp import CommandResult, O2Config, O2Connection, O2UnsafeTransportError
from o2mcp import broker as broker_module
from o2mcp.broker import (
    DEFAULT_REMOTE_RESPONSE_GRACE_SECONDS,
    MAX_LAUNCH_BYTES,
    MAX_REMOTE_RESPONSE_TRANSFER_SECONDS,
    MAX_STATE_COMMAND_PREVIEW_CHARS,
    MAX_UNIX_SOCKET_PATH_BYTES,
    BrokerClient,
    BrokerExecutionResult,
    BrokerServer,
    O2BrokerError,
    O2BrokerUnavailableError,
    _read_launch_fd,
    _read_state,
    _receipt_number,
    _receipt_returncode,
    prepare_broker_directory,
)
from o2mcp.broker_protocol import (
    FRAME_MAGIC,
    MAX_COMMAND_BYTES,
    MAX_OUTPUT_BYTES,
    MAX_REQUEST_ID_BYTES,
    MAX_STDIN_BYTES,
    MAX_TIMEOUT_SECONDS,
    PROTOCOL_VERSION,
    BrokerProtocolError,
    encode_frame,
    read_frame,
    remote_helper_source,
    write_frame,
)
from o2mcp.connection import _held_lock_explanation
from o2mcp.policy import DEFAULT_LOGIN_COOLDOWN_SECONDS, O2PolicyDeniedError, O2PolicyStore


def _reuse_policy(tmp_path) -> O2PolicyStore:
    """Create a valid reuse-only policy through its public state transitions."""

    store = O2PolicyStore(tmp_path / "O2_POLICY.json", client_id="broker-test")
    disabled = store.disable(reason="initialize broker integration test")
    store.enable_reuse(
        expected_revision=disabled["revision"],
        expected_generation=disabled["generation"],
        approval_reference="offline test approval",
    )
    return store


def _pathological_json_payloads() -> list[bytes]:
    """Build JSON that exceeds runtime parser guards without exceeding a frame."""

    payloads = [b'{"nested":' + (b"[" * 10000) + b"0" + (b"]" * 10000) + b"}"]
    get_digit_limit = getattr(sys, "get_int_max_str_digits", None)
    if get_digit_limit is not None:
        digit_limit = get_digit_limit()
        if digit_limit > 0:
            payloads.append(b'{"integer":' + (b"9" * (digit_limit + 1)) + b"}")
    return payloads


@pytest.fixture
def broker_root():
    """Provide a short physical path within macOS's small AF_UNIX limit."""

    root = Path(tempfile.mkdtemp(prefix="o2b-test-", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _start_local_broker(
    tmp_path,
    broker_root,
    *,
    local_request_timeout=5.0,
    remote_response_grace=5.0,
):
    """Start BrokerServer around the production remote helper on a local pipe."""

    policy = _reuse_policy(tmp_path)
    paths = prepare_broker_directory(broker_root)
    server = BrokerServer(
        paths=paths,
        policy_file=policy.path,
        transport_argv=[sys.executable, "-u", "-c", remote_helper_source()],
        alias="offline-o2",
        destination={"hostname": "offline.example", "user": "offline", "port": "22"},
        local_request_timeout=local_request_timeout,
        remote_response_grace=remote_response_grace,
    )
    thread = threading.Thread(target=server.serve_forever, name="offline-o2-broker", daemon=True)
    thread.start()
    client = BrokerClient(paths.root)
    # Production waits for a pipe acknowledgement emitted after SSH spawn. This
    # in-process test has no parent launcher, so wait on the server's own listener
    # publication instead of racing a transient lock observation on a busy CI VM.
    deadline = time.monotonic() + 10
    status = client.local_status()
    while status.get("responsive") is not True and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
        status = client.local_status()
    assert status.get("responsive") is True, status
    return policy, server, thread, client


def _stop_local_broker(thread: threading.Thread, client: BrokerClient) -> None:
    """Stop the integration daemon and prove its local endpoint disappears."""

    client.stop(reason="offline test complete")
    thread.join(timeout=5)
    assert not thread.is_alive()
    with pytest.raises(O2BrokerUnavailableError):
        client.ping()


def test_frame_round_trip_preserves_newlines_and_unicode():
    """Length framing must not confuse command output with protocol boundaries."""

    payload = {"type": "result", "stdout": "line 1\n{not a frame}\n雪\n", "returncode": 0}

    assert read_frame(io.BytesIO(encode_frame(payload))) == payload


def test_frame_writer_handles_partial_binary_writes():
    """Unix SocketIO may accept only a prefix of a large frame per write call."""

    class _PartialWriter:
        def __init__(self):
            self.output = bytearray()

        def write(self, value):
            accepted = min(7, len(value))
            self.output.extend(value[:accepted])
            return accepted

        def flush(self):
            return None

    writer = _PartialWriter()
    payload = {"type": "result", "stdout": "x" * 100}

    write_frame(writer, payload)

    assert read_frame(io.BytesIO(writer.output)) == payload


def test_frame_rejects_oversized_and_truncated_payloads():
    """Invalid peer lengths fail before JSON parsing or unbounded allocation."""

    with pytest.raises(BrokerProtocolError, match="maximum"):
        encode_frame({"value": "too large"}, max_bytes=4)
    with pytest.raises(BrokerProtocolError, match="not JSON serializable"):
        encode_frame({"command": "\ud800"})
    with pytest.raises(BrokerProtocolError, match="expected bytes"):
        read_frame(io.BytesIO(FRAME_MAGIC + struct.pack("!I", 10) + b"{}"))
    for body in _pathological_json_payloads():
        raw = FRAME_MAGIC + struct.pack("!I", len(body)) + body
        with pytest.raises(BrokerProtocolError, match="not valid UTF-8 JSON"):
            read_frame(io.BytesIO(raw))


def test_first_remote_frame_can_resynchronize_after_login_banner():
    """Bounded shell startup text cannot corrupt the remote hello frame."""

    hello = {"type": "hello", "protocol": PROTOCOL_VERSION}
    stream = io.BytesIO(b"Authorized users only\n" + encode_frame(hello))

    assert read_frame(stream, resynchronize=True) == hello


def test_multiple_dynamic_commands_share_one_transport_process(tmp_path, broker_root):
    """Distinct clients and commands reuse one remote helper/SSH process."""

    _policy, server, thread, client = _start_local_broker(tmp_path, broker_root)
    try:
        ssh_pid = server.transport.pid
        first = client.execute("printf 'first\\n'", timeout=5)
        second_client = BrokerClient(client.paths.root)
        second = second_client.execute("printf 'second:%s\\n' \"$((20 + 22))\"", timeout=5)
        third = client.execute("cat", timeout=5, input_text="framed stdin\n")

        assert first.returncode == 0 and first.stdout == "first\n"
        assert second.returncode == 0 and second.stdout == "second:42\n"
        assert third.returncode == 0 and third.stdout == "framed stdin\n"
        assert server.transport.pid == ssh_pid
        assert client.ping()["commands_completed"] == 3
    finally:
        _stop_local_broker(thread, client)


def test_remote_timeout_returns_result_without_restarting_channel(tmp_path, broker_root):
    """A timed-out logical command leaves the persistent helper usable."""

    _policy, server, thread, client = _start_local_broker(tmp_path, broker_root)
    try:
        ssh_pid = server.transport.pid
        timed_out = client.execute("sleep 2", timeout=0.1)
        after = client.execute("printf alive", timeout=5)

        assert timed_out.returncode == 124 and timed_out.timed_out is True
        assert "timed out" in timed_out.stderr
        assert after.stdout == "alive"
        assert server.transport.pid == ssh_pid
    finally:
        _stop_local_broker(thread, client)


def test_none_timeout_preserves_indefinite_execution_contract(tmp_path, broker_root):
    """An explicit absent deadline stays JSON null rather than becoming 24 hours."""

    _policy, server, thread, client = _start_local_broker(tmp_path, broker_root)
    try:
        ssh_pid = server.transport.pid
        result = client.execute("printf no-deadline", timeout=None)

        assert result.stdout == "no-deadline"
        assert result.timed_out is False
        assert server.transport.pid == ssh_pid
    finally:
        _stop_local_broker(thread, client)


def test_remote_output_is_truncated_while_channel_remains_usable(tmp_path, broker_root):
    """Noisy commands are drained but retain only the bounded response prefix."""

    _policy, server, thread, client = _start_local_broker(tmp_path, broker_root)
    try:
        ssh_pid = server.transport.pid
        noisy = client.execute(
            'python3 -c \'import os; os.write(1, b"x" * 2097152); os.write(2, b"y" * 2097152)\'',
            timeout=10,
        )

        assert len(noisy.stdout.encode()) == MAX_OUTPUT_BYTES
        assert len(noisy.stderr.encode()) == MAX_OUTPUT_BYTES
        assert noisy.stdout_truncated is True and noisy.stderr_truncated is True
        assert client.execute("printf after-noise", timeout=5).stdout == "after-noise"
        assert server.transport.pid == ssh_pid
    finally:
        _stop_local_broker(thread, client)


def test_blocked_stdin_feeder_kills_inheriting_descendant(tmp_path, broker_root):
    """A non-reading stdin descendant cannot leak a feeder thread and payload."""

    _policy, server, thread, client = _start_local_broker(tmp_path, broker_root)
    descendant_pid = None
    script = (
        "import subprocess; "
        "child = subprocess.Popen(['/bin/sleep', '30'], stdin=0, "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "print(child.pid, flush=True)"
    )
    try:
        ssh_pid = server.transport.pid
        result = client.execute(
            "python3 -c " + shlex.quote(script),
            timeout=5,
            input_text="x" * (512 * 1024),
        )
        descendant_pid = int(result.stdout.strip())

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(descendant_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            pytest.fail("stdin-inheriting descendant survived feeder cleanup")

        assert result.returncode == 0
        assert client.execute("printf after-stdin-cleanup", timeout=5).stdout == "after-stdin-cleanup"
        assert server.transport.pid == ssh_pid
    finally:
        if descendant_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(descendant_pid, signal.SIGKILL)
        _stop_local_broker(thread, client)


def test_oversized_command_is_rejected_without_losing_persistent_channel(tmp_path, broker_root):
    """An exec-safe command bound must reject locally and preserve one SSH session."""

    _policy, server, thread, client = _start_local_broker(tmp_path, broker_root)
    oversized = "x" * (MAX_COMMAND_BYTES + 1)
    try:
        ssh_pid = server.transport.pid
        with pytest.raises(ValueError, match="byte maximum"):
            client.execute(oversized, timeout=5)

        # A same-user caller may bypass BrokerClient and write directly to the
        # Unix socket. The daemon must reject that frame before acknowledging
        # dispatch, so the remote helper and sole transport remain untouched.
        direct = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        direct.connect(str(client.paths.socket))
        with direct.makefile("rwb", buffering=0) as stream:
            write_frame(
                stream,
                {
                    "type": "exec",
                    "protocol": PROTOCOL_VERSION,
                    "id": "oversized-command",
                    "command": oversized,
                    "timeout_seconds": 5,
                    "stdin": None,
                },
            )
            assert read_frame(stream) == {"type": "error", "error": "invalid_request"}
        direct.close()

        unsafe_text_requests = (
            {
                "type": "exec",
                "protocol": PROTOCOL_VERSION,
                "id": "x" * (MAX_REQUEST_ID_BYTES + 1),
                "command": "printf must-not-run",
                "timeout_seconds": 5,
                "stdin": None,
            },
            {
                "type": "exec",
                "protocol": PROTOCOL_VERSION,
                "id": "invalid-stdin",
                "command": "cat",
                "timeout_seconds": 5,
                "stdin": "\ud800",
            },
            {
                "type": "exec",
                "protocol": PROTOCOL_VERSION,
                "id": "unexpected-structure",
                "command": "printf must-not-run",
                "timeout_seconds": 5,
                "stdin": None,
                "unused_nested_field": {"value": [[[[1]]]]},
            },
        )
        for request in unsafe_text_requests:
            # ensure_ascii deliberately models a hand-crafted peer that can put
            # an unpaired surrogate into otherwise UTF-8 JSON. Production
            # encode_frame rejects it before socket access.
            body = json.dumps(request, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
            unsafe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            unsafe.connect(str(client.paths.socket))
            with unsafe.makefile("rwb", buffering=0) as stream:
                stream.write(FRAME_MAGIC + struct.pack("!I", len(body)) + body)
                assert read_frame(stream) == {"type": "error", "error": "invalid_request"}
            unsafe.close()

        assert client.execute("printf still-alive", timeout=5).stdout == "still-alive"
        assert server.transport.pid == ssh_pid
        assert client.ping()["commands_completed"] == 1
    finally:
        _stop_local_broker(thread, client)


def test_remote_hello_timeout_releases_lifetime_lock(tmp_path, broker_root):
    """A live transport that never speaks protocol cannot wedge all later starts."""

    policy = _reuse_policy(tmp_path)
    paths = prepare_broker_directory(broker_root)
    server = BrokerServer(
        paths=paths,
        policy_file=policy.path,
        transport_argv=[sys.executable, "-u", "-c", "import time; time.sleep(30)"],
        alias="silent-o2",
        destination={"hostname": "silent.example", "user": "offline", "port": "22"},
        startup_timeout=0.1,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)

    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    status = BrokerClient(paths.root).local_status()
    assert status["status"] == "failed"
    assert "no protocol hello" in status["error"]
    assert BrokerClient(paths.root).launch_in_progress() is False
    assert not paths.socket.exists()


def test_listener_publication_failure_preserves_failed_attempt_cooldown(
    tmp_path,
    broker_root,
    monkeypatch,
):
    """A remote hello is not success until the trusted local endpoint exists."""

    policy = _reuse_policy(tmp_path)
    grant = policy.authorize_login(
        expected_revision=policy.snapshot().revision,
        expected_generation=policy.snapshot().generation,
        target="login",
        allow_offvpn=True,
        approval_reference="offline listener-publication failure test",
    )
    launcher_pid = os.getpid()
    policy.consume_login_grant(grant.id, "login")
    paths = prepare_broker_directory(broker_root)
    paths.socket.write_text("same-user stale regular file")
    paths.socket.chmod(0o600)
    server = BrokerServer(
        paths=paths,
        policy_file=policy.path,
        transport_argv=[sys.executable, "-u", "-c", remote_helper_source()],
        alias="offline-o2",
        destination={"hostname": "offline.example", "user": "offline", "port": "22"},
        grant_id=grant.id,
        login_target="login",
        launcher_client_id=grant.client_id,
        launcher_pid=launcher_pid,
    )
    # Production launches the daemon as a direct child. This in-process test
    # models that binding without forking another test harness process.
    monkeypatch.setattr(os, "getppid", lambda: launcher_pid)
    outcomes = []
    thread = threading.Thread(target=lambda: outcomes.append(server.serve_forever()), daemon=True)

    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert outcomes == [1]
    attempt = policy.snapshot().state["login_attempt"]
    assert attempt["outcome"] == "failed"
    assert attempt["blocked_until"] >= attempt["started_at"] + DEFAULT_LOGIN_COOLDOWN_SECONDS
    # Refusing an untrusted path must not then delete it during generic cleanup.
    assert paths.socket.read_text() == "same-user stale regular file"


def test_disconnected_local_caller_does_not_kill_shared_channel(tmp_path, broker_root):
    """An MCP timeout may lose one reply but must not force a new SSH login."""

    _policy, server, thread, client = _start_local_broker(tmp_path, broker_root)
    try:
        ssh_pid = server.transport.pid
        abandoned = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        abandoned.connect(str(client.paths.socket))
        with abandoned.makefile("rwb", buffering=0) as stream:
            write_frame(
                stream,
                {
                    "type": "exec",
                    "protocol": PROTOCOL_VERSION,
                    "id": "abandoned-request",
                    "command": "sleep 0.1; printf abandoned",
                    "timeout_seconds": 5,
                    "stdin": None,
                },
            )
            assert read_frame(stream) == {"type": "dispatched", "id": "abandoned-request"}
        abandoned.close()

        # The daemon must first drain the abandoned command, then it can serve a
        # new client over the same helper process.
        after = client.execute("printf still-alive", timeout=5)
        assert after.stdout == "still-alive"
        assert server.transport.pid == ssh_pid
        assert client.ping()["commands_completed"] == 2
    finally:
        _stop_local_broker(thread, client)


def test_caller_that_disconnects_in_queue_is_never_dispatched(tmp_path, broker_root):
    """A request abandoned before its dispatch acknowledgement is cancelled."""

    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    try:
        first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        first.connect(str(client.paths.socket))
        first_stream = first.makefile("rwb", buffering=0)
        write_frame(
            first_stream,
            {
                "type": "exec",
                "protocol": PROTOCOL_VERSION,
                "id": "first-blocking-request",
                "command": "sleep 0.3; printf first",
                "timeout_seconds": 5,
                "stdin": None,
            },
        )
        assert read_frame(first_stream) == {"type": "dispatched", "id": "first-blocking-request"}

        abandoned = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        abandoned.connect(str(client.paths.socket))
        with abandoned.makefile("rwb", buffering=0) as stream:
            write_frame(
                stream,
                {
                    "type": "exec",
                    "protocol": PROTOCOL_VERSION,
                    "id": "closed-while-queued",
                    "command": "printf must-not-run",
                    "timeout_seconds": 5,
                    "stdin": None,
                },
            )
        abandoned.close()
        first_response = read_frame(first_stream)
        first_stream.close()
        first.close()

        # Give the daemon time to accept the closed socket and reject it at the
        # dispatch acknowledgement boundary.
        time.sleep(0.1)
        assert first_response["stdout"] == "first"
        assert client.ping()["commands_completed"] == 1
        assert client.execute("printf after", timeout=5).stdout == "after"
    finally:
        _stop_local_broker(thread, client)


def test_timed_out_queued_stop_is_cancelled_before_shutdown(tmp_path, broker_root):
    """A stop that reported timeout cannot terminate the shared broker later."""

    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    marker = tmp_path / "long-command-started"
    command_result = []
    command = "python3 -c " + shlex.quote(
        "from pathlib import Path; import time; " f"Path({str(marker)!r}).write_text('started'); time.sleep(0.4)"
    )
    command_thread = threading.Thread(
        target=lambda: command_result.append(client.execute(command, timeout=2)),
        daemon=True,
    )
    try:
        command_thread.start()
        deadline = time.monotonic() + 2
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), "the queue occupant never reached its remote execution boundary"

        with pytest.raises(O2BrokerUnavailableError, match="did not answer locally"):
            client.stop(reason="caller deadline is intentionally shorter than queue", timeout=0.05)

        command_thread.join(timeout=3)
        assert command_result and command_result[0].returncode == 0
        # The abandoned stop frame is accepted only after the long command, but
        # its response write detects the closed caller before mutating lifecycle
        # state. Later users therefore retain the same healthy channel.
        assert client.execute("printf still-running", timeout=2).stdout == "still-running"
    finally:
        command_thread.join(timeout=3)
        _stop_local_broker(thread, client)


def test_command_timeout_starts_only_after_dispatch(tmp_path, broker_root):
    """A short command timeout must not expire while another client owns the queue."""

    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    first_stream = None
    try:
        first.connect(str(client.paths.socket))
        first_stream = first.makefile("rwb", buffering=0)
        write_frame(
            first_stream,
            {
                "type": "exec",
                "protocol": PROTOCOL_VERSION,
                "id": "queue-occupant",
                "command": "sleep 1.5; printf first",
                "timeout_seconds": 5,
                "stdin": None,
            },
        )
        assert read_frame(first_stream) == {"type": "dispatched", "id": "queue-occupant"}

        # The second command's one-second limit is shorter than its 1.5-second
        # queue wait, but comfortably covers its own tiny process after dispatch.
        queued = client.execute("printf second", timeout=1.0)
        first_response = read_frame(first_stream)

        assert first_response["stdout"] == "first"
        assert queued.stdout == "second"
        assert queued.timed_out is False
    finally:
        if first_stream is not None:
            first_stream.close()
        first.close()
        _stop_local_broker(thread, client)


def test_connect_deadline_is_cleared_before_a_large_queued_request_write(broker_root, monkeypatch):
    """The connect-only timeout must not govern queue backpressure on writes."""

    class ObservedDuplex:
        """Minimal framed stream that asserts the socket deadline at first write."""

        def __init__(self, owner):
            self.owner = owner
            self.responses = io.BytesIO()

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def write(self, data):
            # A queued one-MiB frame may block here behind the current command.
            # Any inherited five-second connect deadline would make that valid
            # queue wait fail before the daemon can acknowledge dispatch.
            assert self.owner.timeout is None
            request = read_frame(io.BytesIO(bytes(data)))
            request_id = request["id"]
            self.responses = io.BytesIO(
                encode_frame({"type": "dispatched", "id": request_id})
                + encode_frame(
                    {
                        "type": "result",
                        "id": request_id,
                        "returncode": 0,
                        "stdout": "",
                        "stderr": "",
                        "timed_out": False,
                        "duration_seconds": 0.0,
                        "stdout_truncated": False,
                        "stderr_truncated": False,
                    }
                )
            )
            return len(data)

        def flush(self):
            return None

        def read(self, size):
            return self.responses.read(size)

    class ObservedSocket:
        """Record deadline changes without opening a real broker connection."""

        def __init__(self):
            self.timeout = 5.0
            self.stream = ObservedDuplex(self)

        def settimeout(self, timeout):
            self.timeout = timeout

        def makefile(self, *_args, **_kwargs):
            return self.stream

        def close(self):
            return None

    client = BrokerClient(broker_root)
    observed = ObservedSocket()
    monkeypatch.setattr(client, "_connect", lambda **_kwargs: observed)

    result = client.execute("cat >/dev/null", timeout=5, input_text="x" * MAX_STDIN_BYTES)

    assert result.returncode == 0
    assert observed.timeout == (
        5.0 + DEFAULT_REMOTE_RESPONSE_GRACE_SECONDS + MAX_REMOTE_RESPONSE_TRANSFER_SECONDS + 5.0
    )


def test_malformed_direct_socket_request_is_rejected_without_remote_forward(tmp_path, broker_root):
    """Same-user socket access cannot crash the helper with an invalid timeout."""

    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    try:
        direct = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        direct.connect(str(client.paths.socket))
        with direct.makefile("rwb", buffering=0) as stream:
            write_frame(
                stream,
                {
                    "type": "exec",
                    "protocol": PROTOCOL_VERSION,
                    "id": "invalid-timeout",
                    "command": "printf must-not-run",
                    "timeout_seconds": float("inf"),
                    "stdin": None,
                },
            )
            response = read_frame(stream)
        direct.close()

        # JSON may be syntactically plausible yet exceed Python's safe integer
        # or nesting limits. Feed bytes directly because the normal encoder
        # correctly refuses these values before socket access.
        for body in _pathological_json_payloads():
            pathological = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            pathological.connect(str(client.paths.socket))
            pathological.sendall(FRAME_MAGIC + struct.pack("!I", len(body)) + body)
            pathological.close()

        assert response == {"type": "error", "error": "invalid_request"}
        assert client.execute("printf healthy", timeout=5).stdout == "healthy"
        assert client.ping()["commands_completed"] == 1
    finally:
        _stop_local_broker(thread, client)


@pytest.mark.parametrize(
    "timeout",
    [0, -1, float("nan"), float("inf"), True, 10**1000, MAX_TIMEOUT_SECONDS + 1, 1e20],
)
def test_client_rejects_invalid_timeout_before_socket_access(tmp_path, timeout):
    """Malformed core callers fail locally before inspecting or creating paths."""

    client = BrokerClient(tmp_path / "absent")

    with pytest.raises(ValueError, match="finite positive"):
        client.execute("true", timeout=timeout)

    assert not client.paths.root.exists()


def test_client_rejects_unencodable_stdin_before_socket_access(tmp_path):
    """A Unicode surrogate cannot be dispatched or tear down a ready helper."""

    client = BrokerClient(tmp_path / "absent")

    with pytest.raises(ValueError, match="valid UTF-8"):
        client.execute("cat", timeout=1, input_text="\ud800")

    assert not client.paths.root.exists()


def test_client_rejects_oversized_stdin_before_socket_access(tmp_path):
    """Command stdin stays bounded independently from the response-frame limit."""

    client = BrokerClient(tmp_path / "absent")
    oversized_stdin = "x" * (MAX_STDIN_BYTES + 1)

    with pytest.raises(ValueError, match="stdin must be valid UTF-8"):
        client.execute("cat", timeout=1, input_text=oversized_stdin)

    assert not client.paths.root.exists()


def test_stalled_remote_write_releases_policy_after_stopping_transport(tmp_path, broker_root):
    """A blocked SSH stdin write cannot retain the global policy mutex forever."""

    policy = _reuse_policy(tmp_path)
    server = BrokerServer(
        paths=prepare_broker_directory(broker_root),
        policy_file=policy.path,
        transport_argv=[sys.executable, "-c", "pass"],
        alias="offline-o2",
        destination={"hostname": "offline.example", "user": "offline", "port": "22"},
        remote_write_timeout=0.05,
    )
    server.transport = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    read_fd, write_fd = os.pipe()
    remote_writer = os.fdopen(write_fd, "wb", buffering=0)

    started = time.monotonic()
    try:
        with pytest.raises(O2BrokerError, match="frame write made no progress"), policy.serialize_reuse_launch():
            server._write_remote_frame_with_deadline(
                remote_writer,
                {
                    "type": "exec",
                    "protocol": PROTOCOL_VERSION,
                    "id": "stalled-write",
                    "command": "cat",
                    "timeout_seconds": 5,
                    "stdin": "x" * (256 * 1024),
                },
            )

        assert time.monotonic() - started < 2
        assert server.transport.poll() is not None
        disabled = policy.disable(reason="write deadline released policy mutex")
        assert disabled["mode"] == "disabled"
    finally:
        if server.transport.poll() is None:
            server.transport.kill()
            server.transport.wait(timeout=2)
        remote_writer.close()
        os.close(read_fd)


def test_stalled_remote_response_stops_transport_and_unpublishes_broker(tmp_path, broker_root):
    """An alive SSH process with a silent helper must not remain reusable."""

    _policy, server, thread, client = _start_local_broker(
        tmp_path,
        broker_root,
        remote_response_grace=0.05,
    )
    try:
        # The command suspends its parent, which is the exact embedded remote
        # helper running locally for this test. Its transport process remains
        # alive but cannot emit a result until the daemon-side deadline kills
        # it, reproducing the SSH-alive/helper-silent failure mode offline.
        with pytest.raises(O2BrokerError, match="dispatched but its result was lost"):
            client.execute("kill -STOP $PPID", timeout=0.05)

        thread.join(timeout=3)
        assert not thread.is_alive()
        assert server.transport is not None and server.transport.poll() is not None
        assert not client.paths.socket.exists()
        state = client.local_status()
        assert state["status"] == "failed"
        assert "response made no progress" in state["error"]
    finally:
        if thread.is_alive():
            _stop_local_broker(thread, client)


def test_slow_progressing_remote_response_uses_size_scaled_deadline(tmp_path, broker_root):
    """A legal large response may outlive fixed grace while bytes keep arriving."""

    policy = _reuse_policy(tmp_path)
    server = BrokerServer(
        paths=prepare_broker_directory(broker_root),
        policy_file=policy.path,
        transport_argv=[sys.executable, "-c", "pass"],
        alias="offline-o2",
        destination={"hostname": "offline.example", "user": "offline", "port": "22"},
        remote_response_grace=0.05,
    )
    server.transport = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    read_fd, write_fd = os.pipe()
    remote_reader = os.fdopen(read_fd, "rb", buffering=0)
    response = encode_frame(
        {
            "type": "result",
            "id": "slow-result",
            "returncode": 0,
            "stdout": "x" * (32 * 1024),
            "stderr": "",
        }
    )

    def send_slowly() -> None:
        """Take longer than grace overall while staying continuously active."""

        try:
            for offset in range(0, len(response), 4096):
                os.write(write_fd, response[offset : offset + 4096])
                time.sleep(0.02)
        finally:
            os.close(write_fd)

    writer = threading.Thread(target=send_slowly, daemon=True)
    writer.start()
    try:
        result = server._read_remote_frame_with_deadline(remote_reader, command_timeout=0.05)

        assert result["id"] == "slow-result"
        assert len(result["stdout"]) == 32 * 1024
        assert server.transport.poll() is None
    finally:
        remote_reader.close()
        writer.join(timeout=2)
        if server.transport.poll() is None:
            server.transport.kill()
            server.transport.wait(timeout=2)


def test_slow_progressing_remote_write_uses_size_scaled_deadline(tmp_path, broker_root):
    """Continued pipe progress may use the larger frame-size total budget."""

    policy = _reuse_policy(tmp_path)
    server = BrokerServer(
        paths=prepare_broker_directory(broker_root),
        policy_file=policy.path,
        transport_argv=[sys.executable, "-c", "pass"],
        alias="offline-o2",
        destination={"hostname": "offline.example", "user": "offline", "port": "22"},
        remote_write_timeout=0.05,
    )
    server.transport = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    read_fd, write_fd = os.pipe()
    remote_writer = os.fdopen(write_fd, "wb", buffering=0)
    received = bytearray()

    def drain_slowly() -> None:
        """Read often enough for progress while staying slower than five seconds total."""

        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                return
            received.extend(chunk)
            time.sleep(0.01)

    reader = threading.Thread(target=drain_slowly, daemon=True)
    reader.start()
    try:
        with policy.serialize_reuse_launch():
            server._write_remote_frame_with_deadline(
                remote_writer,
                {
                    "type": "exec",
                    "protocol": PROTOCOL_VERSION,
                    "id": "slow-progress",
                    "command": "cat",
                    "timeout_seconds": 5,
                    "stdin": "x" * (128 * 1024),
                },
            )

        assert received
        assert server.transport.poll() is None
    finally:
        remote_writer.close()
        reader.join(timeout=2)
        os.close(read_fd)
        if server.transport.poll() is None:
            server.transport.kill()
            server.transport.wait(timeout=2)


def test_daemon_rechecks_global_policy_before_forwarding(tmp_path, broker_root):
    """A direct socket client cannot race or bypass workstation disable."""

    policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    try:
        before = client.ping()["commands_completed"]
        policy.disable(reason="offline incident")

        with pytest.raises(O2PolicyDeniedError, match="disabled"):
            client.execute("printf should-not-run", timeout=5)
        assert client.ping()["commands_completed"] == before
    finally:
        _stop_local_broker(thread, client)


def test_disable_is_linearized_with_remote_frame_launch(tmp_path, broker_root):
    """Disable cannot complete inside the check-to-frame-write race window."""

    policy, server, thread, client = _start_local_broker(tmp_path, broker_root)
    entered = threading.Event()
    release = threading.Event()
    command_result = []
    disable_result = []
    real_policy = server.policy

    class _PausingPolicy:
        """Hold the real policy mutex immediately before the frame write."""

        @contextmanager
        def serialize_reuse_launch(self):
            with real_policy.serialize_reuse_launch():
                entered.set()
                release.wait(timeout=5)
                yield

    server.policy = _PausingPolicy()
    command_thread = threading.Thread(
        target=lambda: command_result.append(client.execute("printf launched-before-disable", timeout=5))
    )
    disable_thread = threading.Thread(target=lambda: disable_result.append(policy.disable(reason="race test")))
    try:
        command_thread.start()
        assert entered.wait(timeout=5)
        disable_thread.start()
        time.sleep(0.05)
        assert disable_thread.is_alive(), "disable must wait until the remote frame has been launched"

        release.set()
        command_thread.join(timeout=5)
        disable_thread.join(timeout=5)
        assert command_result[0].stdout == "launched-before-disable"
        assert disable_result[0]["mode"] == "disabled"
        with pytest.raises(O2PolicyDeniedError):
            client.execute("printf too-late", timeout=5)
    finally:
        release.set()
        command_thread.join(timeout=5)
        disable_thread.join(timeout=5)
        _stop_local_broker(thread, client)


def test_local_status_never_starts_or_reconnects_transport(tmp_path):
    """Absent broker status is local evidence, not an implicit probe."""

    client = BrokerClient(tmp_path / "not-started")

    status = client.local_status()

    assert status["status"] == "absent"
    assert status["responsive"] is False
    assert not (tmp_path / "not-started").exists()


def test_overlong_socket_path_fails_before_directory_or_transport_creation(tmp_path):
    """An unbindable macOS socket path is rejected during local preflight."""

    long_root = tmp_path / ("x" * MAX_UNIX_SOCKET_PATH_BYTES)

    with pytest.raises(O2BrokerError, match="shorter private absolute path"):
        prepare_broker_directory(long_root)

    assert not long_root.exists()


def test_broker_state_is_json_serializable(tmp_path, broker_root):
    """Broker receipts remain ordinary operator-readable JSON objects."""

    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    try:
        state = client.local_status()
        # Round-tripping guards against accidentally persisting Path/socket
        # objects when new diagnostics are added.
        assert json.loads(json.dumps(state))["status"] == "ready"
    finally:
        _stop_local_broker(thread, client)


def test_completed_command_receipt_records_channel_occupancy(tmp_path, broker_root):
    """Every finished command must leave measurable evidence of what it cost."""

    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    try:
        assert client.execute("sleep 0.2; printf done", timeout=5).stdout == "done"

        state = client.local_status()
        last = state["last_command"]
        assert state["in_flight"] is None
        assert state["busy"] is False
        assert last["command"]["preview"] == "sleep 0.2; printf done"
        assert last["command"]["preview_truncated"] is False
        assert last["returncode"] == 0
        assert last["timed_out"] is False
        assert last["completed_at"] >= last["dispatched_at"]
        # Broker-observed occupancy is what starves other callers, so it is
        # recorded beside the remote helper's own measurement, not replaced by it.
        assert last["duration_seconds"] >= 0.1
        assert last["remote_duration_seconds"] >= 0.1
    finally:
        _stop_local_broker(thread, client)


def test_busy_receipt_names_the_command_holding_the_serialized_channel(tmp_path, broker_root):
    """An unanswered ping must still identify the command that owns the channel."""

    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    occupant = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    occupant_stream = None
    try:
        occupant.connect(str(client.paths.socket))
        occupant_stream = occupant.makefile("rwb", buffering=0)
        write_frame(
            occupant_stream,
            {
                "type": "exec",
                "protocol": PROTOCOL_VERSION,
                "id": "channel-occupant",
                "command": "sleep 3; printf occupied",
                "timeout_seconds": 10,
                "stdin": None,
            },
        )
        assert read_frame(occupant_stream) == {"type": "dispatched", "id": "channel-occupant"}

        deadline = time.monotonic() + 5
        status = client.local_status()
        while status.get("busy") is not True and time.monotonic() < deadline:
            status = client.local_status()

        # The daemon is healthy but serialized, so it cannot answer a ping while
        # this command runs. Without the receipt that is indistinguishable from a
        # daemon that has stopped serving entirely.
        assert status["busy"] is True
        assert status["responsive"] is False
        assert status["in_flight"]["request_id"] == "channel-occupant"
        assert status["in_flight"]["command"]["preview"] == "sleep 3; printf occupied"
        assert status["busy_for_seconds"] >= 0
        # Busy is not un-ready: the receipt must still authorize queued reuse.
        assert status["status"] == "ready"

        assert read_frame(occupant_stream)["stdout"] == "occupied"
        assert client.local_status()["busy"] is False
    finally:
        if occupant_stream is not None:
            occupant_stream.close()
        occupant.close()
        _stop_local_broker(thread, client)


def test_terminal_receipt_retains_the_command_that_was_in_flight(tmp_path, broker_root):
    """A daemon that dies mid-command must still name what it was running."""

    _policy, _server, thread, client = _start_local_broker(
        tmp_path,
        broker_root,
        remote_response_grace=0.05,
    )
    try:
        with pytest.raises(O2BrokerError, match="dispatched but its result was lost"):
            client.execute("kill -STOP $PPID", timeout=0.05)

        thread.join(timeout=3)
        assert not thread.is_alive()

        state = client.local_status()
        assert state["status"] == "failed"
        assert state["in_flight"]["command"]["preview"] == "kill -STOP $PPID"
        # The forensic record survives, but a terminal receipt is never busy.
        assert state["busy"] is False
    finally:
        if thread.is_alive():
            _stop_local_broker(thread, client)


def test_receipt_bounds_the_command_preview_and_records_a_digest(tmp_path, broker_root):
    """A long command is identified by digest without being copied into the receipt."""

    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    try:
        marker = "x" * (MAX_STATE_COMMAND_PREVIEW_CHARS * 3)
        command = f"printf '%s' {marker} > /dev/null"
        assert client.execute(command, timeout=5).returncode == 0

        fingerprint = client.local_status()["last_command"]["command"]
        assert fingerprint["sha256"] == hashlib.sha256(command.encode("utf-8")).hexdigest()
        assert fingerprint["bytes"] == len(command.encode("utf-8"))
        assert len(fingerprint["preview"]) == MAX_STATE_COMMAND_PREVIEW_CHARS
        assert fingerprint["preview_truncated"] is True
    finally:
        _stop_local_broker(thread, client)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "12", None, True, 10**400])
def test_receipt_rejects_an_unusable_remote_duration(value):
    """A result frame is only as trustworthy as the remote helper that wrote it."""

    assert _receipt_number(value) is None


@pytest.mark.parametrize("value", [True, 2**31, -(2**31) - 1, "0", 1.0, None])
def test_receipt_rejects_an_implausible_remote_returncode(value):
    """Only a plausibly sized integer exit status is retained in the receipt."""

    assert _receipt_returncode(value) is None


def test_receipt_keeps_ordinary_remote_measurements():
    """Sanitizing hostile values must not discard the normal ones."""

    assert _receipt_number(1.5) == 1.5
    assert _receipt_number(2) == 2.0
    assert _receipt_returncode(0) == 0
    assert _receipt_returncode(-1) == -1


def test_occupancy_survives_a_wall_clock_step(tmp_path, broker_root):
    """An NTP or manual clock correction must not corrupt the occupancy metric."""

    stepped = {"offset": 0.0}

    def stepping_clock() -> float:
        return time.time() + stepped["offset"]

    policy = _reuse_policy(tmp_path)
    paths = prepare_broker_directory(broker_root)
    server = BrokerServer(
        paths=paths,
        policy_file=policy.path,
        transport_argv=[sys.executable, "-u", "-c", remote_helper_source()],
        alias="offline-o2",
        destination={"hostname": "offline.example", "user": "offline", "port": "22"},
        clock=stepping_clock,
    )
    thread = threading.Thread(target=server.serve_forever, name="stepped-clock-broker", daemon=True)
    thread.start()
    client = BrokerClient(paths.root)
    try:
        deadline = time.monotonic() + 10
        while client.local_status().get("responsive") is not True and time.monotonic() < deadline:
            time.sleep(0.01)

        # Step the wall clock an hour backwards while the command is running.
        # Subtracting epoch timestamps would clamp the result to 0.0 via ``max``.
        stepper = threading.Timer(0.2, lambda: stepped.__setitem__("offset", -3600.0))
        stepper.start()
        try:
            assert client.execute("sleep 0.5; printf stepped", timeout=10).stdout == "stepped"
        finally:
            stepper.cancel()

        assert client.local_status()["last_command"]["duration_seconds"] >= 0.3
    finally:
        client.stop(reason="stepped clock test complete")
        thread.join(timeout=5)


def test_a_pong_retires_an_in_flight_record_left_by_a_failed_write(tmp_path, broker_root, monkeypatch):
    """A daemon that answers a ping is proof the channel is idle.

    Receipt writes are best effort, so a suppressed completion write can leave a
    finished command recorded as in flight. The record cannot retire itself; the
    pong must, or a responsive broker reports itself busy indefinitely.
    """

    real_write = broker_module._atomic_json_write
    failures = {"armed": False, "raised": 0}

    def flaky_completion_write(path, payload):
        completing = payload.get("last_command") is not None and payload.get("in_flight") is None
        if failures["armed"] and completing:
            failures["armed"] = False
            failures["raised"] += 1
            raise OSError(28, "No space left on device")
        return real_write(path, payload)

    monkeypatch.setattr(broker_module, "_atomic_json_write", flaky_completion_write)
    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    try:
        failures["armed"] = True
        assert client.execute("printf stale", timeout=5).stdout == "stale"
        assert failures["raised"] == 1

        # The durable receipt is genuinely stale: it still names a command that
        # has already finished.
        assert _read_state(client.paths)["in_flight"]["command"]["preview"] == "printf stale"

        status = client.local_status()
        assert status["responsive"] is True
        assert status["busy"] is False
    finally:
        _stop_local_broker(thread, client)


def test_remote_write_failure_after_acknowledgement_is_still_attributed(tmp_path, broker_root):
    """A command that dies between acknowledgement and execution must be named.

    The caller has already been told `dispatched`, so it can only report an
    unknown outcome and is directed at the receipt. The receipt must therefore
    name the command whose outcome is unknown.
    """

    policy = _reuse_policy(tmp_path)
    paths = prepare_broker_directory(broker_root)
    # A helper that completes the protocol hello and then never drains stdin, so
    # the remote frame write stalls only after the caller was acknowledged.
    stalling_helper = (
        "import base64,sys,time;"
        "sys.stdout.buffer.write(base64.b64decode(sys.argv[1]));"
        "sys.stdout.buffer.flush();"
        "time.sleep(30)"
    )
    hello = base64.b64encode(encode_frame({"type": "hello", "protocol": PROTOCOL_VERSION})).decode()
    server = BrokerServer(
        paths=paths,
        policy_file=policy.path,
        transport_argv=[sys.executable, "-u", "-c", stalling_helper, hello],
        alias="offline-o2",
        destination={"hostname": "offline.example", "user": "offline", "port": "22"},
        remote_write_timeout=0.05,
    )
    thread = threading.Thread(target=server.serve_forever, name="stalled-write-broker", daemon=True)
    thread.start()
    client = BrokerClient(paths.root)
    try:
        deadline = time.monotonic() + 10
        while client.local_status().get("responsive") is not True and time.monotonic() < deadline:
            time.sleep(0.01)

        with pytest.raises(O2BrokerError, match="dispatched but its result was lost"):
            client.execute("cat", timeout=5, input_text="x" * (256 * 1024))

        thread.join(timeout=5)
        assert not thread.is_alive()

        state = _read_state(paths)
        assert state["status"] == "failed"
        assert state["in_flight"]["command"]["preview"] == "cat"
    finally:
        if server.transport is not None and server.transport.poll() is None:
            server.transport.kill()
            server.transport.wait(timeout=2)
        thread.join(timeout=5)


def test_receipt_write_failure_does_not_abort_a_running_command(tmp_path, broker_root, monkeypatch):
    """A transient filesystem failure must not tear down the shared channel.

    ``_atomic_json_write`` raises a raw ``OSError``, so guarding only against
    ``O2BrokerError`` would let a full disk kill an already-dispatched command.
    """

    real_write = broker_module._atomic_json_write
    failures = {"armed": False, "raised": 0}

    def flaky_write(path, payload):
        if failures["armed"]:
            failures["armed"] = False
            failures["raised"] += 1
            raise OSError(28, "No space left on device")
        return real_write(path, payload)

    monkeypatch.setattr(broker_module, "_atomic_json_write", flaky_write)
    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    try:
        failures["armed"] = True
        assert client.execute("printf survived", timeout=5).stdout == "survived"
        assert failures["raised"] == 1
        # The channel and its receipt must both still work afterwards.
        assert client.execute("printf again", timeout=5).stdout == "again"
        assert client.local_status()["last_command"]["command"]["preview"] == "printf again"
    finally:
        _stop_local_broker(thread, client)


def test_held_lock_explanation_names_a_busy_command():
    """A held lock plus a silent ping must distinguish busy from wedged."""

    busy = _held_lock_explanation(
        {
            "busy": True,
            "busy_for_seconds": 42.4,
            "in_flight": {"command": {"preview": "du -sb /n/groups/tabin"}},
        }
    )
    assert busy == "It has been serving a command for 42s: 'du -sb /n/groups/tabin'."


@pytest.mark.parametrize(
    "status",
    [
        {"busy": False},
        {},
        {"busy": True, "in_flight": None},
        {"busy": True, "in_flight": {"command": "not-an-object"}},
    ],
)
def test_held_lock_explanation_falls_back_without_a_usable_receipt(status):
    """A missing or malformed in-flight record must not crash the start path."""

    explanation = _held_lock_explanation(status)
    assert explanation.startswith("It ") and explanation.endswith(".")


def test_launch_capability_is_an_inherited_one_shot_descriptor(broker_root):
    """Launch metadata is consumed from a bounded pipe, never a replaceable path."""

    paths = prepare_broker_directory(broker_root)
    payload = {
        "schema_version": 1,
        "broker_dir": str(paths.root),
        "policy_file": str(paths.root / "policy.json"),
        "alias": "offline-o2",
        "destination": {"hostname": "offline.example", "user": "offline", "port": "22"},
        "grant_id": "one-shot-grant",
        "login_target": "login",
        "launcher_client_id": "test-client",
        "launcher_pid": 123,
        "startup_timeout": 5.0,
        "transport_argv": [sys.executable, "-c", "pass"],
    }
    read_fd, write_fd = os.pipe()
    try:
        serialized = json.dumps(payload).encode("utf-8")
        assert len(serialized) < MAX_LAUNCH_BYTES
        os.write(write_fd, serialized)
        os.close(write_fd)
        write_fd = -1

        assert _read_launch_fd(read_fd)["grant_id"] == "one-shot-grant"
        read_fd = -1  # _read_launch_fd owns and closes the inherited descriptor.
        assert not (paths.root / "launch.json").exists()
    finally:
        for fd in (read_fd, write_fd):
            if fd >= 0:
                os.close(fd)


@pytest.mark.parametrize(
    ("receipt", "mode"),
    [
        ('{"schema_version":1,"status":"ready","protocol":1,"alias":"offline-o2"}\n', 0o600),
        (None, None),
        ("{malformed", 0o600),
        (
            '{"schema_version":1,"status":"ready","protocol":2,"alias":"offline-o2",'
            '"destination":{"hostname":"offline.example","user":"offline","port":"22"}}\n',
            0o644,
        ),
        (
            '{"schema_version":1,"status":"starting","protocol":2,"alias":"offline-o2",'
            '"destination":{"hostname":"offline.example","user":"offline","port":"22"}}\n',
            0o600,
        ),
    ],
)
def test_unverified_receipt_fails_before_connecting_to_daemon(broker_root, receipt, mode):
    """A client sends nothing unless a trusted receipt proves protocol compatibility."""

    paths = prepare_broker_directory(broker_root)
    if receipt is not None:
        paths.state.write_text(receipt)
        paths.state.chmod(mode)
    stale_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale_socket.bind(str(paths.socket))
    paths.socket.chmod(0o600)
    try:
        with pytest.raises(O2BrokerUnavailableError, match="no trusted ready receipt"):
            BrokerClient(paths.root).execute("printf unsafe", timeout=1)
    finally:
        stale_socket.close()
        paths.socket.unlink()


def test_new_daemon_rejects_pre_acknowledgement_client_protocol(tmp_path, broker_root):
    """An old client cannot execute a command it would misreport as failed."""

    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    old_client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        old_client.connect(str(client.paths.socket))
        with old_client.makefile("rwb", buffering=0) as stream:
            write_frame(
                stream,
                {
                    "type": "exec",
                    # Protocol 1 did not send this version field and expected a
                    # result immediately instead of a dispatch acknowledgement.
                    "id": "obsolete-client",
                    "command": "printf must-not-run",
                    "timeout_seconds": 5,
                    "stdin": None,
                },
            )
            assert read_frame(stream) == {"type": "error", "error": "invalid_request"}
        assert client.ping()["commands_completed"] == 0
    finally:
        old_client.close()
        _stop_local_broker(thread, client)


def test_incomplete_local_frame_times_out_without_wedging_broker(tmp_path, broker_root):
    """A stalled same-user socket cannot block later commands or local stop."""

    _policy, _server, thread, client = _start_local_broker(
        tmp_path,
        broker_root,
        local_request_timeout=0.1,
    )
    stalled = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        stalled.connect(str(client.paths.socket))
        stalled.sendall(FRAME_MAGIC[:2])
        # Let the server accept this connection and expire its incomplete frame
        # before checking that the next complete client still reaches it.
        time.sleep(0.2)

        assert client.execute("printf recovered", timeout=2).stdout == "recovered"
    finally:
        stalled.close()
        _stop_local_broker(thread, client)


def test_configured_alias_mismatch_blocks_commands_but_allows_local_stop(tmp_path, broker_root):
    """Changing an alias cannot silently reuse the old host or trap its daemon."""

    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    mismatched = BrokerClient(client.paths.root, expected_alias="different-o2")
    try:
        with pytest.raises(O2BrokerUnavailableError, match="targets alias 'offline-o2'"):
            mismatched.execute("printf wrong-host", timeout=2)
        assert client.ping()["commands_completed"] == 0

        stopped = mismatched.stop(reason="configured alias changed")
        assert stopped["type"] == "stopping"
        thread.join(timeout=5)
        assert not thread.is_alive()
    finally:
        if thread.is_alive():
            _stop_local_broker(thread, client)


def test_expanded_destination_change_blocks_stale_broker_reuse(tmp_path, broker_root):
    """HostName, user, or port changes cannot reuse the alias's previous daemon."""

    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    rebound = BrokerClient(
        client.paths.root,
        expected_alias="offline-o2",
        expected_destination={"hostname": "new.example", "user": "offline", "port": "22"},
    )
    try:
        with pytest.raises(O2BrokerUnavailableError, match="expanded destination"):
            rebound.execute("printf wrong-destination", timeout=2)
        assert client.ping()["commands_completed"] == 0
    finally:
        _stop_local_broker(thread, client)


def test_trusted_stale_protocol_receipt_still_allows_local_stop(tmp_path, broker_root):
    """A package upgrade cannot strand the old daemon behind command checks."""

    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    state = json.loads(client.paths.state.read_text())
    state["protocol"] = 1
    client.paths.state.write_text(json.dumps(state))
    client.paths.state.chmod(0o600)
    try:
        with pytest.raises(O2BrokerUnavailableError, match="no trusted ready receipt"):
            client.execute("printf obsolete", timeout=2)

        stopped = client.stop(reason="retire stale protocol")
        assert stopped["type"] == "stopping"
        thread.join(timeout=5)
        assert not thread.is_alive()
    finally:
        if thread.is_alive():
            # Restore the current receipt only for emergency test cleanup.
            state["protocol"] = PROTOCOL_VERSION
            client.paths.state.write_text(json.dumps(state))
            client.paths.state.chmod(0o600)
            _stop_local_broker(thread, client)


def test_commands_do_not_source_login_profiles_per_frame(tmp_path, broker_root, monkeypatch):
    """The helper inherits one session environment without per-command banners."""

    home = tmp_path / "remote-home"
    home.mkdir()
    (home / ".bash_profile").write_text("echo unexpected-login-profile\n")
    (home / ".bashrc").write_text("echo unexpected-bashrc\n")
    bash_env = home / "bash-env"
    bash_env.write_text("echo unexpected-bash-env\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("BASH_ENV", str(bash_env))

    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    try:
        result = client.execute("printf clean", timeout=2)

        assert result.stdout == "clean"
        assert result.stderr == ""
    finally:
        _stop_local_broker(thread, client)


class _FakeBroker:
    """Record production connection routing without a socket or subprocess."""

    def __init__(self, stdout: str = "brokered\n") -> None:
        self.calls = []
        self.stdout = stdout

    def execute(self, command, *, timeout, input_text=None):
        self.calls.append({"command": command, "timeout": timeout, "input_text": input_text})
        return BrokerExecutionResult(0, self.stdout, "", False, 0.01, False, False)


def test_connection_routes_login_commands_to_broker_not_controlmaster(tmp_path, monkeypatch):
    """The production execution path cannot regress to one SSH child per call."""

    policy = _reuse_policy(tmp_path)
    fake = _FakeBroker()
    config = O2Config(policy_file=policy.path, broker_dir=tmp_path / "broker-client")

    def forbidden_runner(_argv, _timeout, _input_text):
        raise AssertionError("an injected runner must not implicitly bypass broker routing")

    connection = O2Connection(config, runner=forbidden_runner, policy=policy, broker_client=fake)
    monkeypatch.setattr(
        connection,
        "master_running",
        lambda _alias=None: (_ for _ in ()).throw(AssertionError("ControlMaster must not be inspected")),
    )

    result = connection.run("hostname", timeout=17, input_text="payload")

    assert result.stdout == "brokered\n"
    assert result.argv == ["o2-broker", config.host_alias, "hostname"]
    assert fake.calls == [{"command": "hostname", "timeout": 17.0, "input_text": "payload"}]


def test_connection_preserves_explicit_none_timeout_for_broker(tmp_path):
    """The public no-deadline sentinel must reach the framed protocol unchanged."""

    policy = _reuse_policy(tmp_path)
    fake = _FakeBroker()
    config = O2Config(policy_file=policy.path, broker_dir=tmp_path / "broker-client")
    connection = O2Connection(config, policy=policy, broker_client=fake)

    result = connection.run("printf indefinite", timeout=None)

    assert result.stdout == "brokered\n"
    assert fake.calls == [{"command": "printf indefinite", "timeout": None, "input_text": None}]


def test_private_legacy_test_transport_rejects_real_subprocess_runner(tmp_path):
    """The offline argv seam cannot accidentally enable real one-off SSH."""

    policy = _reuse_policy(tmp_path)
    config = O2Config(policy_file=policy.path, broker_dir=tmp_path / "broker-client")

    with pytest.raises(ValueError, match="injected non-production runner"):
        O2Connection(config, policy=policy, _legacy_test_transport=True)


def test_production_raw_ssh_is_not_a_broker_fallback(tmp_path):
    """Callers cannot bypass framing by asking run_raw to open a session channel."""

    policy = _reuse_policy(tmp_path)
    config = O2Config(policy_file=policy.path, broker_dir=tmp_path / "broker-client")
    connection = O2Connection(config, policy=policy, broker_client=_FakeBroker())

    with pytest.raises(O2UnsafeTransportError, match="new session channel"):
        connection.run_raw(["ssh", "o2", "hostname"])


def test_broker_transport_is_direct_single_attempt_and_disables_helpers(tmp_path):
    """The sole granted SSH process seals config and cannot fan out via hooks."""

    policy = _reuse_policy(tmp_path)
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Host o2\n  HostName example.invalid\n")
    config = O2Config(
        policy_file=policy.path,
        broker_dir=tmp_path / "broker-client",
        ssh_config_file=ssh_config,
    )

    def resolve_config(argv, _timeout, _input_text):
        """Return representative ``ssh -G`` output without touching a network."""

        assert argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]
        return CommandResult(
            list(argv),
            0,
            "\n".join(
                [
                    "host o2",
                    "hostname example.invalid",
                    "user offline",
                    "port 22",
                    # Repeated identities are ordered, meaningful OpenSSH
                    # inputs and must survive the expanded-config replay.
                    "identityfile /tmp/id_one",
                    "identityfile /tmp/id_two",
                    # These unsafe values model a same-user config that must
                    # not override the broker's hard-coded safety contract.
                    "proxycommand unsafe-helper",
                    "controlpath /tmp/unsafe.sock",
                    "stdinnull yes",
                ]
            )
            + "\n",
            "",
        )

    connection = O2Connection(
        config,
        runner=resolve_config,
        policy=policy,
        broker_client=_FakeBroker(),
    )

    argv = connection._broker_transport_argv(config.host_alias)
    option_values = {argv[index + 1] for index, token in enumerate(argv[:-1]) if token == "-o"}

    assert argv[:3] == [O2Connection.SSH_EXECUTABLE, "-F", "/dev/null"]
    assert argv[argv.index("-S") + 1] == "none"
    assert {
        "HostName=example.invalid",
        "User=offline",
        "Port=22",
        "ControlMaster=no",
        "ControlPath=none",
        "ControlPersist=no",
        "ConnectionAttempts=1",
        "ProxyCommand=none",
        "ProxyJump=none",
        "PermitLocalCommand=no",
        "KnownHostsCommand=none",
        "RemoteCommand=none",
        "ClearAllForwardings=yes",
        "CanonicalizeHostname=no",
        "ForkAfterAuthentication=no",
        "PKCS11Provider=none",
        "StdinNull=no",
        "identityfile=/tmp/id_one",
        "identityfile=/tmp/id_two",
    } <= option_values
    assert "proxycommand=unsafe-helper" not in option_values
    assert "controlpath=/tmp/unsafe.sock" not in option_values
    assert "stdinnull=yes" not in option_values
    assert argv[-2] == config.host_alias
    assert "python3 -u -c" in argv[-1]


def test_transfer_host_commands_use_separate_persistent_broker(tmp_path):
    """Run transitions retain transfer-host execution without raw SSH channels."""

    policy = _reuse_policy(tmp_path)
    login = _FakeBroker(stdout="login\n")
    transfer = _FakeBroker(stdout="transfer\n")
    config = O2Config(
        policy_file=policy.path,
        broker_dir=tmp_path / "login-broker",
        transfer_broker_dir=tmp_path / "transfer-broker",
    )
    connection = O2Connection(
        config,
        policy=policy,
        broker_client=login,
        transfer_broker_client=transfer,
    )

    result = connection.run("nohup transition &", alias=config.transfer_alias, timeout=60)

    assert result.stdout == "transfer\n"
    assert login.calls == []
    assert transfer.calls == [{"command": "nohup transition &", "timeout": 60.0, "input_text": None}]


def test_explicit_role_disambiguates_a_shared_ssh_alias(tmp_path):
    """Role identity, rather than alias text, selects the governed broker."""

    policy = _reuse_policy(tmp_path)
    login = _FakeBroker(stdout="login\n")
    transfer = _FakeBroker(stdout="transfer\n")
    config = O2Config(
        host_alias="shared-o2",
        transfer_alias="shared-o2",
        policy_file=policy.path,
        broker_dir=tmp_path / "login-broker",
        transfer_broker_dir=tmp_path / "transfer-broker",
    )
    connection = O2Connection(
        config,
        policy=policy,
        broker_client=login,
        transfer_broker_client=transfer,
    )

    result = connection.run(
        "printf transfer",
        alias="shared-o2",
        broker_role="transfer",
        timeout=5,
    )

    assert result.stdout == "transfer\n"
    assert login.calls == []
    assert transfer.calls[0]["command"] == "printf transfer"


@pytest.mark.parametrize("transfer", [False, True])
def test_authorized_launcher_starts_one_detached_broker_and_reuses_it(tmp_path, broker_root, monkeypatch, transfer):
    """One role-matched grant launches a daemon that later calls reuse locally."""

    policy = _reuse_policy(tmp_path)
    config = O2Config(
        policy_file=policy.path,
        broker_dir=tmp_path / "unused-login-broker" if transfer else broker_root,
        transfer_broker_dir=broker_root if transfer else tmp_path / "unused-transfer-broker",
        ssh_config_file=tmp_path / "ssh_config",
        broker_start_timeout=5,
        globalprotect_settings_file=tmp_path / "globalprotect-settings.plist",
    )
    config.ssh_config_file.write_text(
        "Host o2\n  HostName example.invalid\nHost o2-transfer\n  HostName transfer.example.invalid\n"
    )
    with config.globalprotect_settings_file.open("wb") as stream:
        plistlib.dump(
            {
                "Palo Alto Networks": {
                    "GlobalProtect": {
                        "PanSetup": {"Portal": "vpn.hms.harvard.edu"},
                        "PanGPS": {"PreferredIP_test": "10.116.16.225"},
                    }
                }
            },
            stream,
        )
    client = BrokerClient(broker_root)
    connection_kwargs = {"transfer_broker_client": client} if transfer else {"broker_client": client}
    connection = O2Connection(config, policy=policy, **connection_kwargs)
    target = "transfer" if transfer else "login"
    grant = policy.authorize_login(
        expected_revision=policy.snapshot().revision,
        expected_generation=policy.snapshot().generation,
        target=target,
        allow_offvpn=True,
        approval_reference=f"offline {target} broker launch",
    )
    monkeypatch.setattr(
        connection,
        "_broker_transport_argv",
        lambda _target: [sys.executable, "-u", "-c", remote_helper_source()],
    )

    try:
        started = connection.start_broker(grant_id=grant.id, transfer=transfer)
        first_pid = started["daemon"]["pid"]
        assert set(started["destination"]) == {"hostname", "user", "port"}
        assert not (client.paths.root / "launch.json").exists(), "launch data must never enter the filesystem"
        alias = config.transfer_alias if transfer else config.host_alias
        assert connection.run("printf launched", alias=alias, timeout=5).stdout == "launched"

        # An already-ready broker is a no-op and must not need or consume a
        # second authorization.
        again = connection.start_broker(transfer=transfer)
        assert again["daemon"]["pid"] == first_pid
        assert policy.snapshot().state["login_attempt"]["outcome"] == "success"
    finally:
        connection.stop_broker(reason="offline launcher test complete", transfer=transfer)
        deadline = time.monotonic() + 5
        while client.launch_in_progress() and time.monotonic() < deadline:
            time.sleep(0.05)


@pytest.mark.parametrize("transfer", [False, True])
def test_on_vpn_launcher_auto_authorizes_exactly_one_broker(tmp_path, broker_root, monkeypatch, transfer):
    """Both host roles may use standing authority only after local VPN proof."""

    policy = _reuse_policy(tmp_path)
    config = O2Config(
        policy_file=policy.path,
        broker_dir=tmp_path / "unused-login-broker" if transfer else broker_root,
        transfer_broker_dir=broker_root if transfer else tmp_path / "unused-transfer-broker",
        ssh_config_file=tmp_path / "ssh_config",
        broker_start_timeout=5,
        globalprotect_settings_file=tmp_path / "globalprotect-settings.plist",
    )
    config.ssh_config_file.write_text(
        "Host o2\n  HostName example.invalid\nHost o2-transfer\n  HostName transfer.example.invalid\n"
    )
    with config.globalprotect_settings_file.open("wb") as stream:
        plistlib.dump(
            {
                "Palo Alto Networks": {
                    "GlobalProtect": {
                        "PanSetup": {"Portal": "vpn.hms.harvard.edu"},
                        "PanGPS": {"PreferredIP_test": "10.116.16.225"},
                    }
                }
            },
            stream,
        )

    def vpn_route_runner(argv, _timeout, _input_text):
        """Resolve the safe SSH config and answer the local VPN route proof."""

        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
            host = "transfer.example.invalid" if argv[-1] == "o2-transfer" else "example.invalid"
            return CommandResult(
                list(argv),
                0,
                f"hostname {host}\nuser offline\nport 22\ncontrolpath /tmp/{argv[-1]}.sock\n",
                "",
            )
        if argv[:2] == [O2Connection.ROUTE_EXECUTABLE, "get"]:
            return CommandResult(list(argv), 0, "interface: utun6\n", "")
        if argv[:1] == [O2Connection.IFCONFIG_EXECUTABLE]:
            return CommandResult(list(argv), 0, "inet 10.116.16.225 netmask 0xffffffff\n", "")
        raise AssertionError(f"unexpected subprocess in offline broker test: {argv}")

    client = BrokerClient(broker_root)
    connection_kwargs = {"transfer_broker_client": client} if transfer else {"broker_client": client}
    connection = O2Connection(config, runner=vpn_route_runner, policy=policy, **connection_kwargs)
    monkeypatch.setattr(
        connection,
        "_broker_transport_argv",
        lambda _target: [sys.executable, "-u", "-c", remote_helper_source()],
    )

    started = None
    try:
        started = connection.start_broker(transfer=transfer, auto_authorize_on_vpn=True)
        assert started["responsive"] is True
        attempt = policy.snapshot().state["login_attempt"]
        assert attempt["target"] == ("transfer" if transfer else "login")
        assert attempt["allow_offvpn"] is False
        assert attempt["outcome"] == "success"
    finally:
        # Do not mask an earlier assertion or setup failure with a second error
        # from trying to stop a broker that never reached its local endpoint.
        if started is not None:
            connection.stop_broker(reason="offline automatic launcher test complete", transfer=transfer)
        deadline = time.monotonic() + 5
        while client.launch_in_progress() and time.monotonic() < deadline:
            time.sleep(0.05)
