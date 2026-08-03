"""Offline tests for the persistent, dynamically framed O2 command broker.

The integration cases run the exact embedded remote helper as a local Python
child. They therefore exercise one long-lived byte channel, Unix-socket clients,
policy enforcement, command timeouts, and lifecycle cleanup without SSH, O2, or
Duo access.
"""

from __future__ import annotations

import io
import json
import shutil
import socket
import struct
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from o2mcp import O2Config, O2Connection, O2UnsafeTransportError
from o2mcp.broker import (
    MAX_UNIX_SOCKET_PATH_BYTES,
    BrokerClient,
    BrokerExecutionResult,
    BrokerServer,
    O2BrokerError,
    O2BrokerUnavailableError,
    prepare_broker_directory,
)
from o2mcp.broker_protocol import (
    FRAME_MAGIC,
    BrokerProtocolError,
    encode_frame,
    read_frame,
    remote_helper_source,
    write_frame,
)
from o2mcp.policy import O2PolicyDeniedError, O2PolicyStore


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


@pytest.fixture
def broker_root():
    """Provide a short physical path within macOS's small AF_UNIX limit."""

    root = Path(tempfile.mkdtemp(prefix="o2b-test-", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _start_local_broker(tmp_path, broker_root):
    """Start BrokerServer around the production remote helper on a local pipe."""

    policy = _reuse_policy(tmp_path)
    paths = prepare_broker_directory(broker_root)
    server = BrokerServer(
        paths=paths,
        policy_file=policy.path,
        transport_argv=[sys.executable, "-u", "-c", remote_helper_source()],
        alias="offline-o2",
    )
    thread = threading.Thread(target=server.serve_forever, name="offline-o2-broker", daemon=True)
    thread.start()
    client = BrokerClient(paths.root)
    # Production waits for a pipe acknowledgement emitted after the daemon has
    # acquired this lock and spawned SSH. The in-process test has no parent
    # launcher, so model that ordering explicitly before calling wait_until_ready.
    deadline = time.monotonic() + 5
    while not client.launch_in_progress() and time.monotonic() < deadline:
        time.sleep(0.01)
    client.wait_until_ready(timeout=5)
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


def test_frame_rejects_oversized_and_truncated_payloads():
    """Invalid peer lengths fail before JSON parsing or unbounded allocation."""

    with pytest.raises(BrokerProtocolError, match="maximum"):
        encode_frame({"value": "too large"}, max_bytes=4)
    with pytest.raises(BrokerProtocolError, match="expected bytes"):
        read_frame(io.BytesIO(FRAME_MAGIC + struct.pack("!I", 10) + b"{}"))


def test_first_remote_frame_can_resynchronize_after_login_banner():
    """Bounded shell startup text cannot corrupt the remote hello frame."""

    hello = {"type": "hello", "protocol": 1}
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
                    "id": "abandoned-request",
                    "command": "sleep 0.1; printf abandoned",
                    "timeout_seconds": 5,
                    "stdin": None,
                },
            )
        abandoned.close()

        # The daemon must first drain the abandoned command, then it can serve a
        # new client over the same helper process.
        after = client.execute("printf still-alive", timeout=5)
        assert after.stdout == "still-alive"
        assert server.transport.pid == ssh_pid
        assert client.ping()["commands_completed"] == 2
    finally:
        _stop_local_broker(thread, client)


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
                    "id": "invalid-timeout",
                    "command": "printf must-not-run",
                    "timeout_seconds": float("inf"),
                    "stdin": None,
                },
            )
            response = read_frame(stream)
        direct.close()

        assert response == {"type": "error", "error": "invalid_request"}
        assert client.execute("printf healthy", timeout=5).stdout == "healthy"
        assert client.ping()["commands_completed"] == 1
    finally:
        _stop_local_broker(thread, client)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_client_rejects_invalid_timeout_before_socket_access(tmp_path, timeout):
    """Malformed core callers fail locally before inspecting or creating paths."""

    client = BrokerClient(tmp_path / "absent")

    with pytest.raises(ValueError, match="finite positive"):
        client.execute("true", timeout=timeout)

    assert not client.paths.root.exists()


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


def test_launch_file_state_is_json_serializable(tmp_path, broker_root):
    """Broker receipts remain ordinary operator-readable JSON objects."""

    _policy, _server, thread, client = _start_local_broker(tmp_path, broker_root)
    try:
        state = client.local_status()
        # Round-tripping guards against accidentally persisting Path/socket
        # objects when new diagnostics are added.
        assert json.loads(json.dumps(state))["status"] == "ready"
    finally:
        _stop_local_broker(thread, client)


class _FakeBroker:
    """Record production connection routing without a socket or subprocess."""

    def __init__(self) -> None:
        self.calls = []

    def execute(self, command, *, timeout, input_text=None):
        self.calls.append({"command": command, "timeout": timeout, "input_text": input_text})
        return BrokerExecutionResult(0, "brokered\n", "", False, 0.01, False, False)


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
    """The sole granted SSH process cannot attach to a mux or fan out via hooks."""

    policy = _reuse_policy(tmp_path)
    config = O2Config(policy_file=policy.path, broker_dir=tmp_path / "broker-client")
    connection = O2Connection(config, policy=policy, broker_client=_FakeBroker())

    argv = connection._broker_transport_argv(config.host_alias, tmp_path / "inspected_config")
    option_values = {argv[index + 1] for index, token in enumerate(argv[:-1]) if token == "-o"}

    assert argv[:3] == [O2Connection.SSH_EXECUTABLE, "-F", str(tmp_path / "inspected_config")]
    assert argv[argv.index("-S") + 1] == "none"
    assert {
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
    } <= option_values
    assert argv[-2] == config.host_alias
    assert "python3 -u -c" in argv[-1]


def test_authorized_launcher_starts_one_detached_broker_and_reuses_it(tmp_path, broker_root, monkeypatch):
    """One grant launches the daemon; later commands and starts stay local-only."""

    policy = _reuse_policy(tmp_path)
    config = O2Config(
        policy_file=policy.path,
        broker_dir=broker_root,
        ssh_config_file=tmp_path / "ssh_config",
        broker_start_timeout=5,
    )
    config.ssh_config_file.write_text("Host o2\n  HostName example.invalid\n")
    client = BrokerClient(broker_root)
    connection = O2Connection(config, policy=policy, broker_client=client)
    grant = policy.authorize_login(
        expected_revision=policy.snapshot().revision,
        expected_generation=policy.snapshot().generation,
        target="login",
        allow_offvpn=True,
        approval_reference="offline broker launch",
    )
    monkeypatch.setattr(
        connection,
        "_broker_transport_argv",
        lambda _target, _config: [sys.executable, "-u", "-c", remote_helper_source()],
    )

    try:
        started = connection.start_broker(grant_id=grant.id)
        first_pid = started["daemon"]["pid"]
        assert connection.run("printf launched", timeout=5).stdout == "launched"

        # An already-ready broker is a no-op and must not need or consume a
        # second authorization.
        again = connection.start_broker()
        assert again["daemon"]["pid"] == first_pid
        assert policy.snapshot().state["login_attempt"]["outcome"] == "success"
    finally:
        connection.stop_broker(reason="offline launcher test complete")
        deadline = time.monotonic() + 5
        while client.launch_in_progress() and time.monotonic() < deadline:
            time.sleep(0.05)
