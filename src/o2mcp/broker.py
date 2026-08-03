"""Workstation-wide broker for one persistent O2 SSH session channel.

An OpenSSH ControlMaster reuses one TCP/authentication transport, but each
``ssh host command`` invocation still opens a new SSH *session channel*. HMS O2
can apply Duo policy at that later boundary. This broker instead owns one SSH
process whose single remote session runs :func:`remote_helper_source` for the
entire broker lifetime. Independent MCP processes send dynamically framed
commands to a mode-0600 Unix socket; the broker serializes them over that one
already-authorized channel.

The daemon never reconnects. If SSH or the remote helper exits, the Unix socket
is removed and later operations fail locally. The durable O2 policy is checked
again inside the daemon before every command, so connecting directly to the
local socket cannot bypass a concurrent workstation-wide disable.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import select
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable

from o2mcp.broker_protocol import (
    PROTOCOL_VERSION,
    BrokerProtocolError,
    read_frame,
    write_frame,
)
from o2mcp.policy import O2PolicyDeniedError, O2PolicyError, O2PolicyStore


class O2BrokerError(RuntimeError):
    """Base class for local persistent-broker failures."""


class O2BrokerUnavailableError(O2BrokerError):
    """Raised when no verified workstation broker is accepting commands."""


class O2BrokerStartupError(O2BrokerError):
    """Raised when the one authorized broker launch cannot become ready."""


class O2BrokerCommandOutcomeUnknownError(O2BrokerError):
    """Raised when a dispatched command loses its result and must not be retried."""


class _O2BrokerTransportError(O2BrokerError):
    """Internal marker for a fatal failure on the sole remote byte stream."""


# macOS exposes a 104-byte ``sockaddr_un.sun_path`` including its terminating
# null. Use a slightly smaller portable ceiling so an overlong custom
# O2_BROKER_DIR fails before an approved SSH attempt is consumed.
MAX_UNIX_SOCKET_PATH_BYTES = 100


@dataclass(frozen=True)
class BrokerPaths:
    """Filesystem endpoints owned by one workstation-wide broker instance."""

    root: Path

    @property
    def socket(self) -> Path:
        """Return the Unix-domain command socket path."""

        return self.root / "command.sock"

    @property
    def state(self) -> Path:
        """Return the durable local status receipt path."""

        return self.root / "state.json"

    @property
    def lock(self) -> Path:
        """Return the lifetime lock held by the sole broker daemon."""

        return self.root / "daemon.lock"

    @property
    def launch(self) -> Path:
        """Return the private one-shot daemon launch specification path."""

        return self.root / "launch.json"

    @property
    def ssh_config(self) -> Path:
        """Return the inspected SSH-config snapshot used for broker startup."""

        return self.root / "ssh_config"

    @property
    def log(self) -> Path:
        """Return the daemon and SSH diagnostic log path."""

        return self.root / "broker.log"


@dataclass(frozen=True)
class BrokerExecutionResult:
    """Result returned by one logical command over the persistent channel."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    stdout_truncated: bool
    stderr_truncated: bool


def _validate_private_directory(path: Path, *, create: bool) -> None:
    """Require a physical, caller-owned mode-0700 broker directory.

    A Unix socket is an authorization boundary only when another local user
    cannot replace or connect to it. Symlinked or permissive directories fail
    closed instead of being silently repaired.
    """

    path = path.expanduser()
    if create:
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise O2BrokerError(f"broker directory is unavailable: {path}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise O2BrokerError(f"broker directory is not a physical directory: {path}")
    if metadata.st_uid != os.getuid():
        raise O2BrokerError(f"broker directory is not owned by uid {os.getuid()}: {path}")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise O2BrokerError(f"broker directory must have mode 0700: {path}")


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one mode-0600 JSON file inside a validated directory."""

    _validate_private_directory(path.parent, create=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def prepare_broker_directory(path: Path) -> BrokerPaths:
    """Create and validate the private broker directory for a launcher."""

    paths = BrokerPaths(path.expanduser())
    socket_bytes = len(os.fsencode(paths.socket)) + 1
    if socket_bytes > MAX_UNIX_SOCKET_PATH_BYTES:
        raise O2BrokerError(
            f"broker socket path needs {socket_bytes} bytes; portable maximum is {MAX_UNIX_SOCKET_PATH_BYTES}. "
            "Set the applicable O2_BROKER_DIR or O2_TRANSFER_BROKER_DIR to a shorter private absolute path."
        )
    _validate_private_directory(paths.root, create=True)
    return paths


def atomic_private_text_write(path: Path, text: str) -> None:
    """Atomically write one mode-0600 UTF-8 launcher artifact.

    Launch specifications and inspected SSH config can contain usernames,
    identity paths, or remote helper source. They share the receipt writer's
    ownership and permission guarantees but are intentionally plain text.
    """

    _validate_private_directory(path.parent, create=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


@contextmanager
def open_private_append(path: Path):
    """Open an owner-only regular log without following a pre-existing symlink."""

    _validate_private_directory(path.parent, create=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise O2BrokerError(f"cannot open private broker log {path}: {exc}") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise O2BrokerError("broker log is not a caller-owned regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise O2BrokerError("broker log must have mode 0600")
        with os.fdopen(fd, "ab", buffering=0) as handle:
            fd = -1
            yield handle
    finally:
        if fd >= 0:
            os.close(fd)


def _open_lifetime_lock(path: Path, *, create: bool) -> BinaryIO:
    """Open the broker lock inode without following aliases or widening access."""

    flags = os.O_RDWR | (os.O_CREAT if create else 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise O2BrokerError("broker lifetime lock is not a caller-owned regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise O2BrokerError("broker lifetime lock must have mode 0600")
        return os.fdopen(fd, "a+b")
    except Exception:
        os.close(fd)
        raise


def _read_state(paths: BrokerPaths) -> dict[str, Any]:
    """Read a locally trustworthy broker receipt or return an absent state."""

    if not paths.root.exists():
        return {"status": "absent"}
    try:
        _validate_private_directory(paths.root, create=False)
        metadata = paths.state.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise O2BrokerError("broker state receipt is not a caller-owned regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise O2BrokerError("broker state receipt must have mode 0600")
        payload = json.loads(paths.state.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise O2BrokerError("broker state receipt must contain a JSON object")
        schema_version = payload.get("schema_version")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != 1:
            raise O2BrokerError("broker state receipt has an unsupported schema version")
        protocol = payload.get("protocol")
        if not isinstance(protocol, int) or isinstance(protocol, bool):
            raise O2BrokerError("broker state receipt has no valid protocol version")
        if not isinstance(payload.get("status"), str):
            raise O2BrokerError("broker state receipt has no valid lifecycle status")
        return payload
    except FileNotFoundError:
        return {"status": "absent"}
    except (json.JSONDecodeError, OSError) as exc:
        return {"status": "invalid", "error": str(exc)}
    except O2BrokerError as exc:
        return {"status": "invalid", "error": str(exc)}


class BrokerClient:
    """Local Unix-socket client shared by independently launched MCP processes."""

    def __init__(self, root: str | Path) -> None:
        self.paths = BrokerPaths(Path(root).expanduser())

    def _connect(self, *, timeout: float | None) -> socket.socket:
        """Connect to the validated private endpoint without sending a frame."""

        _validate_private_directory(self.paths.root, create=False)
        try:
            metadata = self.paths.socket.lstat()
        except FileNotFoundError as exc:
            raise O2BrokerUnavailableError(
                "No persistent O2 command broker is running. Obtain an explicit one-shot login grant, "
                "start the broker once, and then reuse it; do not fall back to raw SSH."
            ) from exc
        if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise O2BrokerUnavailableError("The O2 broker endpoint is not a caller-owned Unix socket.")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise O2BrokerUnavailableError("The O2 broker socket is accessible outside its owner.")
        state = _read_state(self.paths)
        if state.get("status") != "ready" or state.get("protocol") != PROTOCOL_VERSION:
            raise O2BrokerUnavailableError(
                f"The O2 broker has no trusted ready receipt for protocol {PROTOCOL_VERSION} "
                f"(status={state.get('status')!r}, protocol={state.get('protocol')!r}). "
                "Stop any old broker locally before an explicitly authorized restart."
            )

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        try:
            client.connect(str(self.paths.socket))
            return client
        except OSError as exc:
            client.close()
            raise O2BrokerUnavailableError(f"The persistent O2 broker did not answer locally: {exc}") from exc

    def _request(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        """Send one bounded local control request and read its sole response."""

        client = self._connect(timeout=timeout)
        try:
            with client.makefile("rwb", buffering=0) as stream:
                write_frame(stream, payload)
                return read_frame(stream)
        except (OSError, BrokerProtocolError) as exc:
            raise O2BrokerUnavailableError(f"The persistent O2 broker did not answer locally: {exc}") from exc
        finally:
            client.close()

    def execute(
        self,
        command: str,
        *,
        timeout: float,
        input_text: str | None = None,
    ) -> BrokerExecutionResult:
        """Execute one logical command without opening another SSH channel."""

        if not isinstance(command, str) or not command:
            raise ValueError("broker command must be a non-empty string")
        valid_timeout = (
            isinstance(timeout, (int, float))
            and not isinstance(timeout, bool)
            and math.isfinite(timeout)
            and timeout > 0
        )
        if not valid_timeout:
            raise ValueError("broker timeout must be one finite positive number")
        if input_text is not None and not isinstance(input_text, str):
            raise ValueError("broker stdin must be text or None")
        request_id = str(uuid.uuid4())
        request = {
            "type": "exec",
            "protocol": PROTOCOL_VERSION,
            "id": request_id,
            "command": command,
            "timeout_seconds": timeout,
            "stdin": input_text,
        }
        client = self._connect(timeout=5.0)
        dispatched = False
        try:
            with client.makefile("rwb", buffering=0) as stream:
                write_frame(stream, request)
                # Queue time is intentionally not command time. The daemon
                # writes `dispatched` only after this request reaches the front
                # and policy still permits its remote frame. If this caller
                # disconnects first, the daemon cancels the queued request.
                client.settimeout(None)
                response = read_frame(stream)
                if response.get("type") == "error" and response.get("error") == "policy_denied":
                    raise O2PolicyDeniedError(str(response.get("message") or "O2 policy denied broker execution"))
                if response != {"type": "dispatched", "id": request_id}:
                    raise O2BrokerCommandOutcomeUnknownError(
                        f"O2 broker request {request_id} was sent but returned an unexpected dispatch response: "
                        f"{response!r}. Do not retry automatically; inspect broker state first."
                    )
                dispatched = True
                client.settimeout(timeout + 10.0)
                response = read_frame(stream)
        except O2PolicyDeniedError:
            raise
        except (OSError, BrokerProtocolError) as exc:
            if dispatched:
                raise O2BrokerCommandOutcomeUnknownError(
                    f"O2 broker command {request_id} was dispatched but its result was lost: {exc}. "
                    "Do not retry automatically; inspect remote receipts or state first."
                ) from exc
            raise O2BrokerUnavailableError(f"The O2 broker did not dispatch the command: {exc}") from exc
        finally:
            client.close()
        if response.get("type") != "result" or response.get("id") != request_id:
            raise O2BrokerCommandOutcomeUnknownError(
                f"O2 broker command {request_id} was dispatched but returned an unexpected response: "
                f"{response!r}. Do not retry automatically; inspect remote receipts or state first."
            )
        try:
            return BrokerExecutionResult(
                returncode=int(response["returncode"]),
                stdout=str(response.get("stdout", "")),
                stderr=str(response.get("stderr", "")),
                timed_out=bool(response.get("timed_out", False)),
                duration_seconds=float(response.get("duration_seconds", 0.0)),
                stdout_truncated=bool(response.get("stdout_truncated", False)),
                stderr_truncated=bool(response.get("stderr_truncated", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise O2BrokerCommandOutcomeUnknownError(
                f"O2 broker command {request_id} was dispatched but returned a malformed result: "
                f"{response!r}. Do not retry automatically; inspect remote receipts or state first."
            ) from exc

    def ping(self, *, timeout: float = 1.0) -> dict[str, Any]:
        """Verify the local daemon only; this never sends a remote frame."""

        response = self._request({"type": "ping", "id": str(uuid.uuid4())}, timeout=timeout)
        if response.get("type") != "pong" or response.get("protocol") != PROTOCOL_VERSION:
            raise O2BrokerUnavailableError(f"unexpected local broker status response: {response!r}")
        return response

    def local_status(self) -> dict[str, Any]:
        """Return receipt plus local responsiveness without contacting O2."""

        state = _read_state(self.paths)
        try:
            pong = self.ping(timeout=0.25)
        except O2BrokerError as exc:
            return {**state, "responsive": False, "local_error": str(exc)}
        return {**state, "responsive": True, "daemon": pong}

    def launch_in_progress(self) -> bool:
        """Return whether a daemon holds the lifetime lock, without signaling it."""

        try:
            _validate_private_directory(self.paths.root, create=False)
            with _open_lifetime_lock(self.paths.lock, create=False) as handle:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                return False
        except (FileNotFoundError, O2BrokerError):
            return False
        except OSError as exc:
            raise O2BrokerError(f"cannot inspect the broker lifetime lock: {exc}") from exc

    def wait_until_ready(self, *, timeout: float) -> dict[str, Any]:
        """Wait locally for one already-launched broker; never spawn or retry SSH."""

        deadline = time.monotonic() + timeout
        last_error = "broker has not published its local socket"
        while time.monotonic() < deadline:
            try:
                return self.ping(timeout=min(0.5, max(0.05, deadline - time.monotonic())))
            except O2BrokerError as exc:
                last_error = str(exc)
                # If the daemon no longer owns the lock, it cannot later become
                # ready; fail immediately instead of consuming the full Duo wait.
                if not self.launch_in_progress():
                    break
                time.sleep(0.1)
        raise O2BrokerStartupError(
            f"The authorized broker launch did not become locally ready within {timeout:.1f}s: {last_error}. "
            "Do not start another attempt; inspect o2_local_status and the broker log first."
        )

    def stop(self, *, reason: str, timeout: float = 3.0) -> dict[str, Any]:
        """Ask the local daemon to close its one SSH process and exit."""

        response = self._request(
            {"type": "stop", "id": str(uuid.uuid4()), "reason": reason},
            timeout=timeout,
        )
        if response.get("type") != "stopping":
            raise O2BrokerError(f"unexpected broker stop response: {response!r}")
        return response


class _DeadlineSocketReader:
    """Expose ``read`` with one absolute deadline over an accepted socket.

    ``socket.settimeout`` alone is an inactivity timeout and can be extended by a
    peer that trickles bytes. Recomputing the remaining budget before every
    ``recv`` ensures the entire first frame, not merely each chunk, is bounded.
    """

    def __init__(self, client: socket.socket, timeout: float) -> None:
        self.client = client
        self.deadline = time.monotonic() + timeout

    def read(self, size: int) -> bytes:
        """Read at most ``size`` bytes before the shared absolute deadline."""

        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise socket.timeout("local broker request deadline expired")
        self.client.settimeout(remaining)
        return self.client.recv(size)


class BrokerServer:
    """Own one transport process and serialize local clients over its streams."""

    def __init__(
        self,
        *,
        paths: BrokerPaths,
        policy_file: Path,
        transport_argv: list[str],
        alias: str,
        grant_id: str | None = None,
        ack_fd: int | None = None,
        startup_timeout: float = 90.0,
        local_request_timeout: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.paths = paths
        self.policy = O2PolicyStore(policy_file)
        self.transport_argv = list(transport_argv)
        self.alias = alias
        self.grant_id = grant_id
        self.ack_fd = ack_fd
        valid_startup_timeout = (
            isinstance(startup_timeout, (int, float))
            and not isinstance(startup_timeout, bool)
            and math.isfinite(startup_timeout)
            and startup_timeout > 0
        )
        if not valid_startup_timeout:
            raise ValueError("broker startup timeout must be finite and positive")
        self.startup_timeout = startup_timeout
        valid_local_timeout = (
            isinstance(local_request_timeout, (int, float))
            and not isinstance(local_request_timeout, bool)
            and math.isfinite(local_request_timeout)
            and local_request_timeout > 0
        )
        if not valid_local_timeout:
            raise ValueError("broker local request timeout must be finite and positive")
        self.local_request_timeout = local_request_timeout
        self.clock = clock
        self.transport: subprocess.Popen[bytes] | None = None
        self.listener: socket.socket | None = None
        self._stop_requested = False
        self._commands_completed = 0
        self._state: dict[str, Any] = {}
        self._lock_handle: BinaryIO | None = None

    def _write_state(self, status: str, **details: Any) -> None:
        """Persist an operator-readable lifecycle receipt after each transition."""

        self._state.update(
            {
                "schema_version": 1,
                "protocol": PROTOCOL_VERSION,
                "status": status,
                "pid": os.getpid(),
                "alias": self.alias,
                "updated_at": self.clock(),
                "commands_completed": self._commands_completed,
                **details,
            }
        )
        _atomic_json_write(self.paths.state, self._state)

    def _ack(self, message: str) -> None:
        """Notify the grant-consuming parent that SSH did or did not spawn."""

        if self.ack_fd is None:
            return
        try:
            os.write(self.ack_fd, (message.rstrip("\n") + "\n").encode("utf-8"))
        finally:
            os.close(self.ack_fd)
            self.ack_fd = None

    def _acquire_lifetime_lock(self) -> None:
        """Ensure only one daemon can own the workstation broker paths."""

        _validate_private_directory(self.paths.root, create=True)
        handle = _open_lifetime_lock(self.paths.lock, create=True)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise O2BrokerStartupError("another O2 broker daemon already owns the lifetime lock") from exc
        self._lock_handle = handle

    def _drain_transport_stderr(self, stderr: BinaryIO) -> None:
        """Append SSH diagnostics so its bounded pipe can never deadlock."""

        with open_private_append(self.paths.log) as log:
            while True:
                chunk = stderr.read(4096)
                if not chunk:
                    break
                log.write(chunk)

    def _start_transport(self) -> tuple[BinaryIO, BinaryIO]:
        """Spawn the sole SSH process and validate the remote protocol hello."""

        self._write_state("starting", started_at=self.clock())
        try:
            self.transport = subprocess.Popen(
                self.transport_argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except Exception as exc:
            self._ack(f"ERROR transport spawn failed: {exc}")
            raise O2BrokerStartupError(f"persistent transport could not be spawned: {exc}") from exc
        self._ack(f"SPAWNED {self.transport.pid}")
        assert self.transport.stdin is not None
        assert self.transport.stdout is not None
        assert self.transport.stderr is not None
        threading.Thread(
            target=self._drain_transport_stderr,
            args=(self.transport.stderr,),
            name="o2-broker-stderr",
            daemon=True,
        ).start()
        hello_result: dict[str, Any] = {}

        def receive_hello() -> None:
            """Read the first frame in a daemon thread bounded by startup_timeout."""

            try:
                # A remote login shell may emit a bounded site/user banner on
                # stdout before execing Python. The protocol magic permits this
                # one initial resynchronization; command responses stay strict.
                hello_result["payload"] = read_frame(self.transport.stdout, resynchronize=True)
            except Exception as exc:  # pragma: no cover - surfaced by parent thread
                hello_result["error"] = exc

        hello_thread = threading.Thread(target=receive_hello, name="o2-broker-hello", daemon=True)
        hello_thread.start()
        hello_thread.join(timeout=self.startup_timeout)
        if hello_thread.is_alive():
            raise O2BrokerStartupError(
                f"persistent transport emitted no protocol hello within {self.startup_timeout:.1f}s"
            )
        if "error" in hello_result:
            exc = hello_result["error"]
            raise O2BrokerStartupError(f"persistent transport ended before protocol hello: {exc}") from exc
        hello = hello_result.get("payload")
        if hello != {"type": "hello", "protocol": PROTOCOL_VERSION}:
            raise O2BrokerStartupError(f"unexpected remote broker hello: {hello!r}")
        if self.grant_id is not None:
            self.policy.finish_login_attempt(self.grant_id, outcome="success", returncode=0)
        return self.transport.stdin, self.transport.stdout

    def _bind_listener(self) -> None:
        """Publish the local endpoint only after the remote helper is ready."""

        try:
            metadata = self.paths.socket.lstat()
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != os.getuid():
                raise O2BrokerStartupError("refusing to replace an untrusted broker socket path")
            self.paths.socket.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.paths.socket))
        os.chmod(self.paths.socket, 0o600)
        listener.listen(16)
        listener.settimeout(1.0)
        self.listener = listener
        assert self.transport is not None
        self._write_state("ready", ssh_pid=self.transport.pid, ready_at=self.clock())

    def _handle_client(self, client: socket.socket, remote_in: BinaryIO, remote_out: BinaryIO) -> None:
        """Handle one local request, forwarding only validated exec frames."""

        # One same-user process must not be able to monopolize the serialized
        # broker by sending only a frame prefix. The timeout covers receipt of
        # the first and only local request; command execution has its own remote
        # deadline after the explicit dispatch acknowledgement.
        try:
            request = read_frame(_DeadlineSocketReader(client, self.local_request_timeout))
        except (OSError, BrokerProtocolError):
            # A malformed or disconnected same-user local client is not
            # evidence that the remote SSH channel failed. Drop only this
            # request so one expired MCP call cannot force another Duo.
            return

        # Local responses are also bounded so an acknowledged caller that stops
        # reading a large result cannot wedge later clients. This deadline does
        # not govern or interrupt the remote command itself.
        client.settimeout(self.local_request_timeout)
        with client.makefile("wb", buffering=0) as local:
            request_type = request.get("type")
            if request_type == "ping":
                with suppress(OSError, BrokerProtocolError):
                    write_frame(
                        local,
                        {
                            "type": "pong",
                            "protocol": PROTOCOL_VERSION,
                            "pid": os.getpid(),
                            "ssh_pid": self.transport.pid if self.transport else None,
                            "commands_completed": self._commands_completed,
                        },
                    )
                return
            if request_type == "stop":
                self._stop_requested = True
                self._write_state("stopping", stop_reason=str(request.get("reason") or "local request"))
                with suppress(OSError, BrokerProtocolError):
                    write_frame(local, {"type": "stopping", "pid": os.getpid()})
                return
            if request_type != "exec":
                with suppress(OSError, BrokerProtocolError):
                    write_frame(local, {"type": "error", "error": "invalid_request"})
                return
            request_timeout = request.get("timeout_seconds")
            valid_timeout = (
                isinstance(request_timeout, (int, float))
                and not isinstance(request_timeout, bool)
                and math.isfinite(request_timeout)
                and request_timeout > 0
            )
            valid_request = (
                request.get("protocol") == PROTOCOL_VERSION
                and isinstance(request.get("id"), str)
                and bool(request["id"])
                and isinstance(request.get("command"), str)
                and bool(request["command"])
                and valid_timeout
                and (request.get("stdin") is None or isinstance(request.get("stdin"), str))
            )
            if not valid_request:
                with suppress(OSError, BrokerProtocolError):
                    write_frame(local, {"type": "error", "error": "invalid_request"})
                return

            # The client checks policy before connecting, while this second gate
            # blocks hand-crafted local socket requests that try to bypass the
            # MCP layer. Holding the policy mutex through the remote frame write
            # also linearizes a concurrent disable: it either prevents this
            # command or observes a command that was already launched.
            try:
                with self.policy.serialize_reuse_launch():
                    # This acknowledgement is the execution boundary. If a
                    # queued caller has disconnected, do not forward its frame;
                    # otherwise it could perform a mutation after reporting a
                    # local timeout and invite a dangerous retry.
                    readable, _, _ = select.select([client], [], [], 0)
                    if readable:
                        try:
                            pending = client.recv(1, socket.MSG_PEEK)
                        except OSError:
                            return
                        if not pending:
                            return
                    try:
                        write_frame(local, {"type": "dispatched", "id": request["id"]})
                    except (OSError, BrokerProtocolError):
                        return
                    write_frame(remote_in, request)
            except O2PolicyError as exc:
                with suppress(OSError, BrokerProtocolError):
                    write_frame(local, {"type": "error", "error": "policy_denied", "message": str(exc)})
                return
            except (OSError, BrokerProtocolError) as exc:
                raise _O2BrokerTransportError(f"persistent remote stream failed: {exc}") from exc
            try:
                response = read_frame(remote_out)
            except (OSError, BrokerProtocolError) as exc:
                raise _O2BrokerTransportError(f"persistent remote stream failed: {exc}") from exc
            if response.get("id") != request.get("id"):
                raise _O2BrokerTransportError("remote response id does not match the serialized request")
            self._commands_completed += 1
            self._write_state("ready", last_command_at=self.clock())
            # A caller may time out or close while its remote command continues.
            # The result has already been drained, so losing only the local reply
            # must not tear down the healthy shared channel.
            with suppress(OSError, BrokerProtocolError):
                write_frame(local, response)

    def serve_forever(self) -> int:
        """Run until a local stop or transport failure; never reconnect."""

        outcome = "failed"
        failure: str | None = None
        try:
            self._acquire_lifetime_lock()
            remote_in, remote_out = self._start_transport()
            self._bind_listener()
            while not self._stop_requested:
                assert self.listener is not None
                assert self.transport is not None
                if self.transport.poll() is not None:
                    failure = f"persistent SSH process exited with return code {self.transport.returncode}"
                    break
                try:
                    client, _ = self.listener.accept()
                except socket.timeout:
                    continue
                with client:
                    try:
                        self._handle_client(client, remote_in, remote_out)
                    except _O2BrokerTransportError as exc:
                        failure = f"broker protocol failed: {exc}"
                        break
            if self._stop_requested:
                outcome = "stopped"
            elif failure is None:
                failure = "broker loop ended unexpectedly"
        except Exception as exc:
            failure = str(exc)
            if self.grant_id is not None:
                try:
                    returncode = self.transport.returncode if self.transport is not None else None
                    self.policy.finish_login_attempt(self.grant_id, outcome="failed", returncode=returncode)
                except O2PolicyError:
                    # The primary startup failure remains in the broker receipt;
                    # a concurrent repair/disable must not be overwritten.
                    pass
            with suppress(O2BrokerError):
                self._write_state("failed", error=failure)
            return 1
        finally:
            if self.listener is not None:
                self.listener.close()
            with suppress(FileNotFoundError):
                self.paths.socket.unlink()
            if self.transport is not None and self.transport.poll() is None:
                self.transport.terminate()
                try:
                    self.transport.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.transport.kill()
                    self.transport.wait(timeout=5)
            if self._lock_handle is not None:
                self._lock_handle.close()

        self._write_state(outcome, stopped_at=self.clock(), error=failure)
        return 0 if outcome == "stopped" else 1


def _load_launch(path: Path) -> dict[str, Any]:
    """Load and validate the private launch file written by the MCP parent."""

    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise O2BrokerStartupError("broker launch file is not a caller-owned regular file")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise O2BrokerStartupError("broker launch file must have mode 0600")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_strings = ("broker_dir", "policy_file", "alias", "grant_id")
    if not isinstance(payload, dict) or any(not isinstance(payload.get(key), str) for key in required_strings):
        raise O2BrokerStartupError("broker launch file is missing required string fields")
    argv = payload.get("transport_argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item for item in argv):
        raise O2BrokerStartupError("broker launch transport_argv must be a non-empty string list")
    startup_timeout = payload.get("startup_timeout")
    if (
        not isinstance(startup_timeout, (int, float))
        or isinstance(startup_timeout, bool)
        or not math.isfinite(startup_timeout)
        or startup_timeout <= 0
    ):
        raise O2BrokerStartupError("broker launch startup_timeout must be finite and positive")
    return payload


def _install_signal_handlers(server: BrokerServer) -> None:
    """Translate local TERM/INT into the same graceful no-reconnect shutdown."""

    def request_stop(_signum: int, _frame: Any) -> None:
        server._stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)


def main(argv: list[str] | None = None) -> int:
    """Console entry point for the private broker daemon process."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="run the broker daemon")
    parser.add_argument("--launch-file", type=Path, required=True)
    parser.add_argument("--ack-fd", type=int, required=True)
    args = parser.parse_args(argv)
    if not args.serve:
        parser.error("--serve is required")

    try:
        payload = _load_launch(args.launch_file)
        server = BrokerServer(
            paths=BrokerPaths(Path(payload["broker_dir"])),
            policy_file=Path(payload["policy_file"]),
            transport_argv=payload["transport_argv"],
            alias=payload["alias"],
            grant_id=payload["grant_id"],
            ack_fd=args.ack_fd,
            startup_timeout=float(payload["startup_timeout"]),
        )
        _install_signal_handlers(server)
        return server.serve_forever()
    except Exception as exc:
        with suppress(OSError):
            os.write(args.ack_fd, f"ERROR {exc}\n".encode())
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess integration
    raise SystemExit(main())
