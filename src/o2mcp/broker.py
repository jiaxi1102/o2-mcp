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
import hashlib
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
    MAX_COMMAND_BYTES,
    MAX_FRAME_BYTES,
    MAX_REQUEST_ID_BYTES,
    MAX_STDIN_BYTES,
    MAX_TIMEOUT_SECONDS,
    PROTOCOL_VERSION,
    BrokerProtocolError,
    command_within_exec_limit,
    encode_frame,
    read_frame,
    utf8_text_within_limit,
    write_frame,
)
from o2mcp.policy import LoginTarget, O2PolicyDeniedError, O2PolicyError, O2PolicyStore


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
# The private launch payload contains only paths, grant metadata, SSH argv, and
# the embedded helper source. Bound descriptor reads so a malformed parent
# cannot make the daemon accumulate unbounded local data before policy checks.
MAX_LAUNCH_BYTES = 1024 * 1024
# A stalled SSH channel must not retain the workstation policy mutex forever.
# Five seconds is ample for a <=16 MiB local pipe write while still keeping a
# global incident stop responsive when the remote side no longer drains input.
DEFAULT_REMOTE_WRITE_TIMEOUT_SECONDS = 5.0
MIN_REMOTE_WRITE_BYTES_PER_SECOND = 128 * 1024
# The remote helper normally needs only its command deadline plus bounded drain
# cleanup to begin returning a frame. Give it a small inactivity window; the
# client separately includes the daemon's full size-scaled transfer budget.
DEFAULT_REMOTE_RESPONSE_GRACE_SECONDS = 5.0
# Result frames can legally approach the 16-MiB protocol ceiling after JSON
# escaping. Progress resets the short inactivity timer, while this conservative
# throughput floor supplies a separate absolute transfer budget.
MIN_REMOTE_RESPONSE_BYTES_PER_SECOND = 64 * 1024
MAX_REMOTE_RESPONSE_TRANSFER_SECONDS = MAX_FRAME_BYTES / MIN_REMOTE_RESPONSE_BYTES_PER_SECOND
# One caller must not be able to wait forever for a channel another caller is
# holding. Before dispatch nothing has been sent, so ending that wait is safe to
# retry -- unlike any failure afterwards -- and an error naming the occupying
# command is far more useful than an MCP call that never returns. Scale the
# budget with the caller's own deadline: waiting longer than your own command
# would have taken is the point at which the silence, not the queue, is the
# problem. A floor keeps short commands from failing behind ordinary traffic.
# The protocol permits a seven-day deadline, and the daemon's watchdog budget
# follows whatever a caller requests -- so an hour-long timeout also buys an
# hour before a silent transport is torn down. The MCP tool caps itself at 300s,
# but that guard sits on one input model; this ceiling applies to every caller of
# the client, so an in-process one cannot reinstate a multi-hour hold on a
# channel shared by every MCP process on the workstation.
MAX_COMMAND_TIMEOUT_SECONDS = 3600.0
# A forced stop must outlast the contention that made it necessary: its
# connection may sit in the listen backlog behind queued commands before the
# daemon accepts it.
FORCED_STOP_WAIT_SECONDS = 60.0
MIN_QUEUE_WAIT_SECONDS = 60.0
DEFAULT_QUEUE_WAIT_SECONDS = 300.0
# ``socket.settimeout(0)`` selects non-blocking mode rather than an immediate
# deadline, so an exhausted budget must still pass a small positive value and
# let the read deadline report the expiry.
_MINIMUM_SOCKET_TIMEOUT_SECONDS = 0.001
# One bounded command preview is retained in the owner-only receipt so an
# operator can name the command occupying the serialized channel. The bound is
# in characters, not bytes, so truncation cannot split a UTF-8 code point, and
# the full command stays out of the receipt: the digest identifies a repeat
# offender without copying arguments that may embed sensitive paths.
MAX_STATE_COMMAND_PREVIEW_CHARS = 200


def _is_bounded_positive_timeout(value: Any) -> bool:
    """Validate finite deadlines without numeric or socket timeout overflow.

    ``math.isfinite`` converts integers to a C double and can raise
    ``OverflowError`` for a protocol-valid but extremely large integer. Direct
    same-user socket requests must be rejected, not allowed to unwind the
    daemon and terminate its sole authenticated transport. A separate upper
    bound also keeps ``socket.settimeout`` inside portable platform ranges.
    """

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and 0 < value <= MAX_TIMEOUT_SECONDS
    except OverflowError:
        return False


def _command_fingerprint(command: str) -> dict[str, Any]:
    """Summarize one command for the receipt without copying it in full."""

    encoded = command.encode("utf-8")
    preview = command[:MAX_STATE_COMMAND_PREVIEW_CHARS]
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "preview": preview,
        "preview_truncated": len(preview) != len(command),
    }


def _receipt_number(value: Any) -> float | None:
    """Return one finite number for the receipt, or None for anything else.

    Result frames cross the SSH channel, so their fields are only as trustworthy
    as the remote helper. Filtering here keeps a non-finite or non-numeric value
    out of a file that every later reader must still be able to parse.
    """

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


class O2BrokerBusyError(O2BrokerCommandOutcomeUnknownError):
    """Raised when a queued command was not acknowledged before its deadline.

    A subclass, not a sibling, and deliberately so. The broker is healthy and
    merely occupied, which is worth reporting separately -- but the caller
    cannot prove its request was never dispatched. The daemon writes the
    acknowledgement and forwards the command as one step, so a budget expiring
    in that instant leaves a command that is running and a client that never
    saw it acknowledged. Inheriting the uncertain-outcome contract means every
    handler that already refuses to auto-retry keeps refusing here, and a
    mutating command cannot be silently duplicated by a caller that trusted an
    optimistic classification.
    """


def _queue_wait_seconds(timeout: float | None) -> float:
    """Return how long a caller may wait for the channel before giving up."""

    if timeout is None:
        return DEFAULT_QUEUE_WAIT_SECONDS
    return max(MIN_QUEUE_WAIT_SECONDS, float(timeout))


def _interprocess_monotonic() -> float | None:
    """Return a monotonic reading that two processes may validly subtract.

    ``time.monotonic()`` has an implementation-defined reference point. On macOS
    it maps to ``mach_absolute_time()``, which excludes time the machine spent
    asleep and therefore differs from ``CLOCK_MONOTONIC`` on the same host by
    however long that host has slept. A reading the daemon persists is only
    meaningful to a reader sampling the identical source, so name the POSIX
    clock rather than depend on what ``time.monotonic()`` happens to map to for
    a given interpreter and platform. Where that clock is unavailable, publish
    nothing: no elapsed time is better than one another process cannot
    interpret.
    """

    clock_gettime = getattr(time, "clock_gettime", None)
    clock_id = getattr(time, "CLOCK_MONOTONIC", None)
    if clock_gettime is None or clock_id is None:
        return None
    try:
        return float(clock_gettime(clock_id))
    except (OSError, ValueError):
        return None


def _receipt_returncode(value: Any) -> int | None:
    """Return one plausibly sized remote exit status, or None."""

    if type(value) is not int or not -(2**31) <= value < 2**31:
        return None
    return value


@dataclass(frozen=True)
class BrokerPaths:
    """Filesystem endpoints owned by one workstation-wide broker instance."""

    root: Path

    @property
    def socket(self) -> Path:
        """Return the Unix-domain command socket path."""

        return self.root / "command.sock"

    @property
    def control_socket(self) -> Path:
        """Return the Unix-domain control socket path.

        Separate from the command socket because it is serviced by its own
        thread: a daemon blocked on a command's result cannot answer on the
        command socket at all, which is what made stopping a busy broker
        impossible. This endpoint carries no commands -- only stop and ping --
        so widening it later is a deliberate act rather than an accident.
        """

        return self.root / "control.sock"

    @property
    def state(self) -> Path:
        """Return the durable local status receipt path."""

        return self.root / "state.json"

    @property
    def lock(self) -> Path:
        """Return the lifetime lock held by the sole broker daemon."""

        return self.root / "daemon.lock"

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
    socket_bytes = max(
        len(os.fsencode(paths.socket)) + 1,
        len(os.fsencode(paths.control_socket)) + 1,
    )
    if socket_bytes > MAX_UNIX_SOCKET_PATH_BYTES:
        raise O2BrokerError(
            f"broker socket path needs {socket_bytes} bytes; portable maximum is {MAX_UNIX_SOCKET_PATH_BYTES}. "
            "Set the applicable O2_BROKER_DIR or O2_TRANSFER_BROKER_DIR to a shorter private absolute path."
        )
    _validate_private_directory(paths.root, create=True)
    return paths


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
        if not isinstance(payload.get("alias"), str) or not payload["alias"]:
            raise O2BrokerError("broker state receipt has no valid SSH alias")
        destination = payload.get("destination")
        valid_destination = isinstance(destination, dict) and all(
            isinstance(destination.get(key), str) and bool(destination[key]) for key in ("hostname", "user", "port")
        )
        # Protocol 1 predates destination binding. Keep its otherwise trusted
        # receipt readable only so a local stop can retire that daemon; protocol
        # 2 command clients require the full identity below.
        if protocol >= 2 and not valid_destination:
            raise O2BrokerError("broker state receipt has no valid expanded SSH destination")
        return payload
    except FileNotFoundError:
        return {"status": "absent"}
    except (json.JSONDecodeError, OSError) as exc:
        return {"status": "invalid", "error": str(exc)}
    except O2BrokerError as exc:
        return {"status": "invalid", "error": str(exc)}


class BrokerClient:
    """Local Unix-socket client shared by independently launched MCP processes."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_alias: str | None = None,
        expected_destination: dict[str, str] | None = None,
    ) -> None:
        """Bind a client to one broker directory and optional configured target."""

        self.paths = BrokerPaths(Path(root).expanduser())
        self.expected_alias = expected_alias
        self.expected_destination = dict(expected_destination) if expected_destination is not None else None

    def _connect(
        self,
        *,
        timeout: float | None,
        require_expected_alias: bool = True,
        require_current_protocol: bool = True,
    ) -> socket.socket:
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
        if state.get("status") != "ready" or (require_current_protocol and state.get("protocol") != PROTOCOL_VERSION):
            raise O2BrokerUnavailableError(
                f"The O2 broker has no trusted ready receipt for protocol {PROTOCOL_VERSION} "
                f"(status={state.get('status')!r}, protocol={state.get('protocol')!r}). "
                "Stop any old broker locally before an explicitly authorized restart."
            )
        if require_expected_alias and self.expected_alias is not None and state.get("alias") != self.expected_alias:
            raise O2BrokerUnavailableError(
                f"The ready O2 broker targets alias {state.get('alias')!r}, but this client is configured for "
                f"{self.expected_alias!r}. Stop the old broker locally before an explicitly authorized restart."
            )
        if (
            require_expected_alias
            and self.expected_destination is not None
            and state.get("destination") != self.expected_destination
        ):
            raise O2BrokerUnavailableError(
                f"The ready O2 broker targets expanded destination {state.get('destination')!r}, but the current "
                f"SSH configuration resolves to {self.expected_destination!r}. Stop the old broker locally before "
                "an explicitly authorized restart."
            )

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        try:
            client.connect(str(self.paths.socket))
            return client
        except OSError as exc:
            client.close()
            raise O2BrokerUnavailableError(f"The persistent O2 broker did not answer locally: {exc}") from exc

    def _request(
        self,
        payload: dict[str, Any],
        *,
        timeout: float | None,
        require_expected_alias: bool = True,
        require_current_protocol: bool = True,
    ) -> dict[str, Any]:
        """Send one bounded local control request and read its sole response."""

        client = self._connect(
            timeout=timeout,
            require_expected_alias=require_expected_alias,
            require_current_protocol=require_current_protocol,
        )
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
        if not command_within_exec_limit(command):
            raise ValueError(f"broker command exceeds the {MAX_COMMAND_BYTES}-byte maximum")
        if timeout is not None and not _is_bounded_positive_timeout(timeout):
            raise ValueError(
                f"broker timeout must be None or a finite positive number no greater than {MAX_TIMEOUT_SECONDS}"
            )
        if timeout is not None and float(timeout) > MAX_COMMAND_TIMEOUT_SECONDS:
            raise ValueError(
                f"broker command timeout {float(timeout):.0f}s exceeds the {MAX_COMMAND_TIMEOUT_SECONDS:.0f}s "
                "ceiling. One command's deadline is also how long it may hold the shared serialized channel and "
                "how long the daemon will wait before tearing down a silent transport. Submit a job and poll for "
                "it instead of waiting inside one command."
            )
        if input_text is not None and not isinstance(input_text, str):
            raise ValueError("broker stdin must be text or None")
        if input_text is not None and not utf8_text_within_limit(
            input_text,
            MAX_STDIN_BYTES,
            allow_empty=True,
        ):
            raise ValueError(f"broker stdin must be valid UTF-8 within the {MAX_STDIN_BYTES}-byte limit")
        request_id = str(uuid.uuid4())
        request = {
            "type": "exec",
            "protocol": PROTOCOL_VERSION,
            "id": request_id,
            "command": command,
            "timeout_seconds": timeout,
            "stdin": input_text,
        }
        try:
            # Validate the complete escaped JSON body, not only raw stdin. Quotes,
            # backslashes, and control characters can expand substantially and
            # must fail before a broker connection or dispatch acknowledgement.
            encode_frame(request)
        except BrokerProtocolError as exc:
            raise ValueError(f"encoded broker request exceeds the frame contract: {exc}") from exc
        queue_wait = _queue_wait_seconds(timeout)
        queue_deadline = time.monotonic() + queue_wait
        # The queue begins at the connection, not after it. The daemon accepts
        # nothing while it serves a command, so callers pile into the listen
        # backlog; once that is full, connect() itself blocks. A short ceiling
        # there would report a healthy but saturated broker as absent, and
        # absent is the one diagnosis that tells a caller to start another one.
        # A receipt that fails validation still raises before any connection, so
        # a genuinely missing broker keeps its own error.
        try:
            client = self._connect(timeout=queue_wait)
        except O2BrokerUnavailableError as exc:
            if isinstance(exc.__cause__, socket.timeout):
                raise self._undispatched_error(request_id, queue_wait) from exc
            raise
        # The connect deadline guards only the local connect. Queue
        # time is not command time: the daemon acknowledges `dispatched` only
        # once this request reaches the front and policy still permits its
        # remote frame, and a large valid frame can fill the Unix send buffer
        # while an earlier command is being served. Give the whole pre-dispatch
        # phase one absolute budget rather than no budget at all: waiting
        # forever turns a busy channel into an MCP call that never returns,
        # which is the failure this bound exists to replace. Expiry is reported,
        # not treated as proof the request never ran -- see
        # ``_undispatched_error``. Abandoning the socket does usually make the
        # daemon cancel the queued request rather than run it unobserved, but
        # that check is best effort and cannot be relied on for correctness.
        dispatched = False
        try:
            with client.makefile("rwb", buffering=0) as stream:
                client.settimeout(max(_MINIMUM_SOCKET_TIMEOUT_SECONDS, queue_deadline - time.monotonic()))
                write_frame(stream, request)
                # Reading the acknowledgement through the socket rather than the
                # stream is what gives this one frame an absolute deadline instead
                # of a per-recv one. Alternating with the stream below is safe only
                # because ``makefile`` was opened unbuffered: neither side can
                # strand bytes the other then fails to see.
                response = read_frame(_DeadlineSocketReader(client, max(0.0, queue_deadline - time.monotonic())))
                if response.get("type") == "error":
                    if response.get("error") == "policy_denied":
                        raise O2PolicyDeniedError(str(response.get("message") or "O2 policy denied broker execution"))
                    # A refusal is issued before anything is forwarded, so this
                    # is a rejected request, not an uncertain outcome.
                    raise O2BrokerError(
                        f"The O2 broker refused request {request_id}: {response.get('error')!r}. "
                        f"{response.get('message') or ''}".strip()
                    )
                if response != {"type": "dispatched", "id": request_id}:
                    raise O2BrokerCommandOutcomeUnknownError(
                        f"O2 broker request {request_id} was sent but returned an unexpected dispatch response: "
                        f"{response!r}. Do not retry automatically; inspect broker state first."
                    )
                dispatched = True
                # Keep the local caller alive for the daemon's entire legal
                # command-plus-result budget. Otherwise a slow but progressing
                # large response would be preserved by the daemon yet still be
                # misreported locally as an unknown outcome after ten seconds.
                client.settimeout(
                    None
                    if timeout is None
                    else timeout + DEFAULT_REMOTE_RESPONSE_GRACE_SECONDS + MAX_REMOTE_RESPONSE_TRANSFER_SECONDS + 5.0
                )
                response = read_frame(stream)
        except O2PolicyDeniedError:
            raise
        except (OSError, BrokerProtocolError) as exc:
            if dispatched:
                raise O2BrokerCommandOutcomeUnknownError(
                    f"O2 broker command {request_id} was dispatched but its result was lost: {exc}. "
                    "Do not retry automatically; inspect remote receipts or state first."
                ) from exc
            if isinstance(exc, socket.timeout):
                raise self._undispatched_error(request_id, queue_wait) from exc
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

    def _busy_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        """Describe an in-flight command so an unanswered ping stays diagnosable.

        The daemon serializes local clients, so it cannot answer a ping while a
        command occupies the one channel. Any command slower than the ping
        deadline therefore makes a healthy broker look unresponsive. Reporting
        the dispatched command and its elapsed time separates that ordinary busy
        state from a daemon that has actually stopped serving. A terminal
        receipt keeps its in-flight record for forensics but is never busy.

        Only consult this when the ping went unanswered. Because receipt writes
        are best effort, a suppressed completion write can leave a finished
        command recorded as in flight, and the record alone cannot retire it.
        """

        in_flight = state.get("in_flight")
        if state.get("status") != "ready" or not isinstance(in_flight, dict):
            return {"busy": False}
        # A daemon killed before it can run its terminal write -- SIGKILL, a
        # crash, a reboot -- leaves a `ready` receipt naming a command that
        # nothing will ever retire. The lifetime lock is the liveness signal: no
        # holder means no daemon, so the record describes a command that is not
        # running and reporting it as busy would recreate the ambiguity this
        # diagnostic exists to remove. Inspection failures are not evidence of
        # occupancy either, and this call must never raise: a status read stays
        # readable even when the lock cannot be examined.
        try:
            if not self.launch_in_progress():
                return {"busy": False}
        except O2BrokerError:
            return {"busy": False}
        summary: dict[str, Any] = {"busy": True}
        # A held lock implies a live daemon, hence no reboot since it started,
        # so its persisted monotonic reading is comparable with this process's
        # clock. Prefer it: subtracting epochs would let an NTP or manual step
        # inflate this value or clamp it to zero, the same defect already fixed
        # for completed occupancy. Without a usable reading, report no elapsed
        # time rather than a wrong one.
        dispatched = _receipt_number(in_flight.get("dispatched_clock_monotonic"))
        sampled = _interprocess_monotonic()
        if dispatched is not None and sampled is not None:
            summary["busy_for_seconds"] = max(0.0, sampled - dispatched)
        return summary

    def _undispatched_error(self, request_id: str, waited: float) -> O2BrokerCommandOutcomeUnknownError:
        """Classify a request whose acknowledgement never arrived in time.

        The receipt can settle the question in one direction: if this very
        request is the one recorded in flight, it was dispatched and is running,
        whatever this client failed to read. It can never settle it in the other
        direction, because the daemon acknowledges and forwards as one step, so
        silence is consistent with both a queued request and one dispatched in
        the instant the budget expired.
        """

        state = _read_state(self.paths)
        in_flight = state.get("in_flight")
        if isinstance(in_flight, dict) and in_flight.get("request_id") == request_id:
            return O2BrokerCommandOutcomeUnknownError(
                f"O2 broker command {request_id} was not acknowledged within {waited:.0f}s, but the broker "
                "receipt records it as the command now running. It was dispatched; do not retry. Inspect "
                "remote receipts or o2_local_status for its outcome."
            )
        busy = self._busy_summary(state)
        detail = "The broker did not reach this request in time."
        if busy.get("busy"):
            command = in_flight.get("command") if isinstance(in_flight, dict) else None
            preview = command.get("preview") if isinstance(command, dict) else None
            elapsed = busy.get("busy_for_seconds")
            held = f" for {elapsed:.0f}s" if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) else ""
            named = f": {preview!r}" if isinstance(preview, str) and preview else ""
            detail = f"The broker has been serving another command{held}{named}."
        return O2BrokerBusyError(
            f"O2 broker request {request_id} waited {waited:.0f}s without an acknowledgement. {detail} It was "
            "almost certainly never dispatched, but that cannot be proven: the daemon acknowledges and forwards "
            "as one step. Repeat a read-only command freely; verify a mutating one against remote receipts "
            "before repeating it. Long remote waits belong in a submitted job, not in one command."
        )

    def local_status(self) -> dict[str, Any]:
        """Return receipt plus local responsiveness without contacting O2."""

        state = _read_state(self.paths)
        try:
            pong = self.ping(timeout=0.25)
        except O2BrokerError as exc:
            # A command dispatched after the snapshot above publishes its own
            # receipt and then leaves this ping queued behind it, so the
            # pre-ping snapshot can omit the very command that caused the
            # silence. Re-read before summarizing: the fresher receipt is what
            # explains an unanswered ping.
            state = _read_state(self.paths)
            return {**state, **self._busy_summary(state), "responsive": False, "local_error": str(exc)}
        # A serialized daemon can only answer between commands, so a pong is
        # positive proof that nothing occupies the channel. That evidence
        # outranks the receipt and retires an in-flight record that a suppressed
        # best-effort completion write would otherwise leave standing forever.
        return {**state, "busy": False, "responsive": True, "daemon": pong}

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

    def _connect_control(self, *, timeout: float) -> socket.socket:
        """Open the control endpoint without requiring a reusable receipt.

        A stop must work when the receipt is stale, the alias has changed, or
        the protocol has moved on -- the situations where retiring the daemon is
        most necessary. The authority here is the same as for commands: a
        mode-0600 socket inside a mode-0700 directory this user owns.
        """

        _validate_private_directory(self.paths.root, create=False)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        try:
            client.connect(str(self.paths.control_socket))
            return client
        except OSError as exc:
            client.close()
            raise O2BrokerUnavailableError(f"The O2 broker control endpoint did not answer locally: {exc}") from exc

    def _control_request(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        """Send one control frame and read its sole reply."""

        client = self._connect_control(timeout=timeout)
        try:
            with client.makefile("wb", buffering=0) as stream:
                write_frame(stream, payload)
            return read_frame(_DeadlineSocketReader(client, timeout))
        except (OSError, BrokerProtocolError) as exc:
            raise O2BrokerUnavailableError(f"The O2 broker control endpoint did not answer locally: {exc}") from exc
        finally:
            client.close()

    def control_ping(self, *, timeout: float = 2.0) -> dict[str, Any]:
        """Ask the control thread whether the daemon is alive, and whether busy.

        Unlike the command-socket ping this is answerable while a command runs,
        because the thread answering it is not the thread blocked on the result.
        """

        response = self._control_request({"type": "ping", "id": str(uuid.uuid4())}, timeout=timeout)
        if response.get("type") != "pong" or response.get("protocol") != PROTOCOL_VERSION:
            raise O2BrokerUnavailableError(f"unexpected broker control status response: {response!r}")
        return response

    def control_stop(self, *, reason: str, timeout: float = 5.0) -> dict[str, Any]:
        """Retire the daemon through the endpoint that is never blocked.

        The command loop rechecks the stop flag before accepting anything else,
        so requests already queued behind the running command are skipped rather
        than served first. The running command is still allowed to finish:
        abandoning it would convert a stop into an unknown remote outcome.
        """

        response = self._control_request(
            {"type": "stop", "id": str(uuid.uuid4()), "reason": reason},
            timeout=timeout,
        )
        if response.get("type") != "stopping":
            raise O2BrokerError(f"unexpected broker control stop response: {response!r}")
        return response

    def stop(self, *, reason: str, timeout: float | None = None, force: bool = False) -> dict[str, Any]:
        """Ask the local daemon to close its one SSH process and exit.

        A plain stop is abandoned if this caller's short deadline expires before
        the daemon reaches it, which is deliberate: a stop reported as failed
        should not shut the shared broker down minutes later. It also means a
        plain stop cannot retire a busy broker at all, because a busy daemon
        cannot reach any request until its command ends.

        ``force`` marks the request as one to honour regardless, so the operator's
        intent survives their own timeout. An older daemon ignores the flag and
        simply behaves as before.

        It is graceful and it is queued, which bounds how fast it can possibly
        act. The daemon serves one connection at a time from a FIFO accept
        queue, so a forced stop takes effect after the in-flight command *and
        any connections already queued ahead of it* have been served -- a delay
        bounded by queue depth times the per-command ceiling, not by one
        command. Nothing here abandons a running command, and no request jumps
        the queue; a stop that must act immediately still needs a signal sent by
        hand, and a control path that is not behind the command queue is the
        proper fix.
        """

        # A forced stop is an explicit operator action against a broker that is
        # by definition busy, so it waits: its connection may have to enter a
        # backlog behind queued commands, and failing fast there would leave the
        # only working remedy unreachable exactly when it is needed.
        if timeout is None:
            timeout = FORCED_STOP_WAIT_SECONDS if force else 3.0
        response = self._request(
            {"type": "stop", "id": str(uuid.uuid4()), "reason": reason, "force": bool(force)},
            timeout=timeout,
            # A local stop must remain possible after the configured SSH alias
            # or package protocol changes; only command reuse requires the
            # current destination and dispatch semantics. Protocols 1 and 2 use
            # the same framing and local stop request.
            require_expected_alias=False,
            require_current_protocol=False,
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
        destination: dict[str, str],
        login_target: LoginTarget | None = None,
        launcher_client_id: str | None = None,
        launcher_pid: int | None = None,
        grant_id: str | None = None,
        ack_fd: int | None = None,
        startup_timeout: float = 90.0,
        local_request_timeout: float = 5.0,
        remote_write_timeout: float = DEFAULT_REMOTE_WRITE_TIMEOUT_SECONDS,
        remote_response_grace: float = DEFAULT_REMOTE_RESPONSE_GRACE_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.paths = paths
        self.policy = O2PolicyStore(policy_file)
        self.transport_argv = list(transport_argv)
        self.alias = alias
        if not isinstance(destination, dict) or not all(
            isinstance(destination.get(key), str) and bool(destination[key]) for key in ("hostname", "user", "port")
        ):
            raise ValueError("broker destination must contain non-empty hostname, user, and port strings")
        self.destination = {key: destination[key] for key in ("hostname", "user", "port")}
        if grant_id is not None:
            if login_target not in {"login", "transfer"}:
                raise ValueError("authorized broker launch must identify the login or transfer role")
            if not isinstance(launcher_client_id, str) or not launcher_client_id:
                raise ValueError("authorized broker launch must identify its originating client")
            if type(launcher_pid) is not int or launcher_pid <= 0:
                raise ValueError("authorized broker launch must identify its positive launcher PID")
        self.login_target = login_target
        self.launcher_client_id = launcher_client_id
        self.launcher_pid = launcher_pid
        self.grant_id = grant_id
        self.ack_fd = ack_fd
        if not _is_bounded_positive_timeout(startup_timeout):
            raise ValueError("broker startup timeout must be finite and positive")
        self.startup_timeout = startup_timeout
        if not _is_bounded_positive_timeout(local_request_timeout):
            raise ValueError("broker local request timeout must be finite and positive")
        self.local_request_timeout = local_request_timeout
        if not _is_bounded_positive_timeout(remote_write_timeout):
            raise ValueError("broker remote write timeout must be finite and positive")
        self.remote_write_timeout = remote_write_timeout
        if not _is_bounded_positive_timeout(remote_response_grace):
            raise ValueError("broker remote response grace must be finite and positive")
        self.remote_response_grace = remote_response_grace
        self.clock = clock
        self.transport: subprocess.Popen[bytes] | None = None
        self.listener: socket.socket | None = None
        self.control_listener: socket.socket | None = None
        self._control_thread: threading.Thread | None = None
        self._control_stopping = False
        # Two threads now publish receipts. ``_write_state`` reads, mutates and
        # rewrites one dict, so without this a control stop landing during a
        # command completion could interleave and persist a torn record.
        self._state_lock = threading.Lock()
        # Held across check-and-acknowledge, and by a control stop while it
        # sets the flag. Rechecking the flag alone would only narrow the
        # window: a request accepted before the stop has left the queue the
        # outer loop skips, so nothing else prevents it starting after the
        # operator was told the broker was stopping.
        #
        # The section ends at the acknowledgement rather than the remote frame
        # write, because past that point forwarding is obligatory -- the caller
        # has been told it was dispatched, and refusing would leave it believing
        # a command ran that never did. So the guarantee is that no command is
        # *acknowledged* after a stop is, and the cost is one local socket
        # write rather than a frame write that a large stdin can stretch to
        # tens of seconds.
        self._dispatch_lock = threading.Lock()
        self._stop_requested = False
        self._commands_completed = 0
        self._in_flight: dict[str, Any] | None = None
        self._state: dict[str, Any] = {}
        self._lock_handle: BinaryIO | None = None

    def _running_status(self) -> str:
        """Return the status a mid-command receipt may claim.

        A command completing after a stop was acknowledged must not republish
        `ready`: between that write and the terminal one, a client reading the
        receipt would believe the broker reusable and try to use it.
        """

        return "stopping" if self._stop_requested else "ready"

    def _write_state(self, status: str, **details: Any) -> None:
        """Persist an operator-readable lifecycle receipt after each transition."""

        with self._state_lock:
            self._write_state_locked(status, **details)

    def _write_state_locked(self, status: str, **details: Any) -> None:
        """Mutate and persist the receipt; callers must hold the state lock."""

        self._state.update(
            {
                "schema_version": 1,
                "protocol": PROTOCOL_VERSION,
                "status": status,
                "pid": os.getpid(),
                "alias": self.alias,
                "destination": self.destination,
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
            if self.grant_id is None:
                # Offline integration tests may construct a local broker without
                # a login attempt. Production daemon launches always carry a
                # consumed grant and take the guarded branch below.
                self.transport = self._spawn_transport_process()
            else:
                assert self.login_target is not None
                assert self.launcher_client_id is not None
                assert self.launcher_pid is not None
                if os.getppid() != self.launcher_pid:
                    raise O2BrokerStartupError(
                        "broker daemon is not a direct child of the process that consumed the login grant"
                    )
                # Validate the exact active attempt immediately around the only
                # authentication-capable spawn. This is intentionally daemon-
                # side: replaying launch metadata cannot bypass policy merely by
                # invoking the private entry point directly.
                with self.policy.authorize_consumed_broker_launch(
                    self.grant_id,
                    self.login_target,
                    client_id=self.launcher_client_id,
                    launcher_pid=self.launcher_pid,
                ):
                    self.transport = self._spawn_transport_process()
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
        return self.transport.stdin, self.transport.stdout

    def _spawn_transport_process(self) -> subprocess.Popen[bytes]:
        """Create the broker's sole persistent transport subprocess."""

        return subprocess.Popen(
            self.transport_argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

    def _terminate_transport(self, *, timeout: float) -> None:
        """Stop the sole transport locally and wait a bounded time for cleanup."""

        process = self.transport
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)

    def _write_remote_frame_with_deadline(self, remote_in: BinaryIO, request: dict[str, Any]) -> None:
        """Write one request without indefinitely holding the policy mutex.

        Python's buffered pipe writes expose no per-call deadline. Perform the
        bounded write in a helper thread; if SSH stops draining its stdin, kill
        the sole transport so the pipe closes, the writer unblocks, and the
        policy mutex can be released for an incident disable.
        """

        frame = memoryview(encode_frame(request))
        frame_size = len(frame)
        started = time.monotonic()
        inactivity_deadline = started + self.remote_write_timeout
        absolute_deadline = started + self.remote_write_timeout + (frame_size / MIN_REMOTE_WRITE_BYTES_PER_SECOND)
        try:
            descriptor = remote_in.fileno()
            was_blocking = os.get_blocking(descriptor)
            os.set_blocking(descriptor, False)
        except (OSError, ValueError) as exc:
            raise _O2BrokerTransportError(f"persistent remote stream is not writable: {exc}") from exc

        try:
            while frame:
                now = time.monotonic()
                remaining = min(inactivity_deadline, absolute_deadline) - now
                if remaining <= 0:
                    self._terminate_transport(timeout=1.0)
                    raise _O2BrokerTransportError(
                        "persistent remote frame write exceeded its progress/size deadline; transport stopped"
                    )
                _, writable, _ = select.select([], [descriptor], [], remaining)
                if not writable:
                    self._terminate_transport(timeout=1.0)
                    raise _O2BrokerTransportError(
                        "persistent remote frame write made no progress before its deadline; transport stopped"
                    )
                try:
                    written = os.write(descriptor, frame[:65536])
                except BlockingIOError:
                    continue
                if written <= 0:
                    raise O2BrokerError("persistent remote stream accepted no frame bytes")
                frame = frame[written:]
                # Continued progress may use the size-scaled total budget, while
                # a fully stalled channel is still detected within the fixed
                # inactivity window so policy disable remains responsive.
                inactivity_deadline = time.monotonic() + self.remote_write_timeout
        except _O2BrokerTransportError:
            raise
        except (OSError, ValueError, O2BrokerError) as exc:
            raise _O2BrokerTransportError(f"persistent remote stream failed: {exc}") from exc
        finally:
            with suppress(OSError, ValueError):
                os.set_blocking(descriptor, was_blocking)

    def _read_remote_frame_with_deadline(
        self,
        remote_out: BinaryIO,
        command_timeout: float | None,
    ) -> dict[str, Any]:
        """Read one result without leaving a finite command wedged forever.

        The remote helper owns command timeout enforcement, but an alive SSH
        process is not proof that the helper can still reply. For finite
        commands, wait through the command deadline, then require continuing
        byte progress within a short inactivity window and a frame-size-scaled
        absolute budget. A miss is fatal to the channel: after dispatch, no
        automatic retry is safe, and retaining the socket would falsely
        advertise reuse.

        The unbounded branch below is no longer reachable from a request: the
        daemon bounds a deadline-free exec to the workstation ceiling before
        dispatching it, because a helper that never replies to one would block
        the shared channel with nothing to time out. It is kept for direct
        callers and reads synchronously by their explicit choice.
        """

        if command_timeout is None:
            return read_frame(remote_out)

        response_result: dict[str, Any] = {}
        progress = {"first_at": None, "last_at": None, "complete": False}
        progress_changed = threading.Condition()

        class ProgressReader:
            """Expose small reads so completed bytes refresh inactivity state."""

            def read(self, size: int) -> bytes:
                # ``BufferedReader.read(large_size)`` may wait for the complete
                # frame and hide intermediate network progress. Four-KiB reads
                # let the supervising thread distinguish slow transfer from a
                # silent helper without parsing protocol bytes twice.
                chunk = remote_out.read(min(size, 4096))
                if chunk:
                    now = time.monotonic()
                    with progress_changed:
                        if progress["first_at"] is None:
                            progress["first_at"] = now
                        progress["last_at"] = now
                        progress_changed.notify_all()
                return chunk

        def receive_response() -> None:
            """Capture the blocking read so the daemon can enforce its deadline."""

            try:
                response_result["payload"] = read_frame(ProgressReader())
            except Exception as exc:  # pragma: no cover - surfaced by parent thread
                response_result["error"] = exc
            finally:
                with progress_changed:
                    progress["complete"] = True
                    progress_changed.notify_all()

        reader = threading.Thread(target=receive_response, name="o2-broker-response", daemon=True)
        started = time.monotonic()
        command_deadline = started + float(command_timeout)
        inactivity_deadline = command_deadline + self.remote_response_grace
        absolute_deadline = inactivity_deadline + MAX_REMOTE_RESPONSE_TRANSFER_SECONDS
        last_progress_seen: float | None = None
        reader.start()
        with progress_changed:
            while not progress["complete"]:
                latest_progress = progress["last_at"]
                if isinstance(latest_progress, float) and latest_progress != last_progress_seen:
                    inactivity_deadline = latest_progress + self.remote_response_grace
                    last_progress_seen = latest_progress
                first_progress = progress["first_at"]
                if isinstance(first_progress, float):
                    absolute_deadline = (
                        first_progress + self.remote_response_grace + MAX_REMOTE_RESPONSE_TRANSFER_SECONDS
                    )
                remaining = min(inactivity_deadline, absolute_deadline) - time.monotonic()
                if remaining <= 0:
                    break
                progress_changed.wait(timeout=remaining)
            timed_out = not progress["complete"]
        if timed_out:
            # Killing the sole transport closes its stdout pipe and releases the
            # reader. The outer broker loop then removes its socket and lifetime
            # lock, making the unknown outcome visible rather than reusable.
            self._terminate_transport(timeout=1.0)
            reader.join(timeout=1.0)
            raise _O2BrokerTransportError(
                "persistent remote response made no progress or exceeded its size-scaled deadline; " "transport stopped"
            )
        reader.join(timeout=1.0)
        if "error" in response_result:
            exc = response_result["error"]
            raise _O2BrokerTransportError(f"persistent remote stream failed: {exc}") from exc
        response = response_result.get("payload")
        if not isinstance(response, dict):  # Defensive guard for the thread handoff.
            raise _O2BrokerTransportError("persistent remote stream returned no response frame")
        return response

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
        # From this point the daemon owns both the descriptor and pathname, so
        # outer cleanup may remove them. A refusal before bind must leave an
        # untrusted pre-existing object untouched.
        self.listener = listener
        os.chmod(self.paths.socket, 0o600)
        # Callers pile into this backlog while a command is served, and a
        # connection that cannot be queued fails outright -- including a stop.
        # Size it well above the number of MCP processes on a workstation so a
        # control request is not locked out by ordinary command contention.
        listener.listen(128)
        listener.settimeout(1.0)
        assert self.transport is not None
        self._write_state("ready", ssh_pid=self.transport.pid, ready_at=self.clock())

    def _bind_control_listener(self) -> None:
        """Publish the control endpoint alongside the command endpoint."""

        try:
            metadata = self.paths.control_socket.lstat()
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != os.getuid():
                raise O2BrokerStartupError("refusing to replace an untrusted broker control socket path")
            self.paths.control_socket.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.paths.control_socket))
        self.control_listener = listener
        os.chmod(self.paths.control_socket, 0o600)
        # Control traffic is one frame per operator action, never bulk, so a
        # small backlog is ample. The timeout is what lets this thread notice a
        # shutdown rather than blocking in accept until a caller arrives.
        listener.listen(8)
        listener.settimeout(1.0)

    def _serve_control(self) -> None:
        """Answer stop and ping while the command thread is busy.

        This runs on its own thread for one reason: the command loop spends its
        time blocked on a remote result and cannot accept anything until that
        command ends. Everything here is deliberately cheap and local -- no
        remote frame is ever written from this thread, and no command can be
        submitted through this socket -- so servicing it concurrently adds no
        second path to O2.
        """

        while not self._stop_requested and not self._control_stopping:
            listener = self.control_listener
            if listener is None:
                return
            try:
                client, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                # The listener was closed underneath us during shutdown.
                return
            # One malformed or hostile request must not take the control
            # thread down with it: a dead control thread is a silently
            # unstoppable broker, which is the failure this endpoint exists to
            # remove. Drop the request, keep serving.
            with client, suppress(Exception):
                self._handle_control_client(client)

    def _handle_control_client(self, client: socket.socket) -> None:
        """Handle one control request. Never forwards anything to O2."""

        request = read_frame(_DeadlineSocketReader(client, self.local_request_timeout))
        client.settimeout(self.local_request_timeout)
        request_type = request.get("type")
        with client.makefile("wb", buffering=0) as reply:
            if request_type == "ping":
                write_frame(
                    reply,
                    {
                        "type": "pong",
                        "protocol": PROTOCOL_VERSION,
                        "pid": os.getpid(),
                        "commands_completed": self._commands_completed,
                        # Answerable here precisely because this thread is not
                        # the one blocked on the command.
                        "busy": self._in_flight is not None,
                    },
                )
                return
            if request_type == "stop":
                # Setting the flag is the whole mechanism. The command loop
                # rechecks it before accepting anything else, so every request
                # already queued behind the running command is skipped rather
                # than served first -- which is what makes this prompt. The
                # running command itself is still allowed to finish; abandoning
                # it would turn a stop into an unknown remote outcome.
                # Take the dispatch lock so this cannot land between a command
                # thread's own stop check and its acknowledgement. That section
                # is one local socket write, so waiting here is brief.
                with self._dispatch_lock:
                    self._stop_requested = True
                # The state lock also guards the shutdown marker, so a stop
                # arriving while the daemon publishes its terminal receipt
                # cannot overwrite it with `stopping` and leave that standing as
                # the final word.
                finalizing = False
                with self._state_lock:
                    if self._control_stopping:
                        finalizing = True
                    else:
                        with suppress(O2BrokerError, OSError):
                            self._write_state_locked(
                                "stopping", stop_reason=str(request.get("reason") or "control request")
                            )
                if finalizing:
                    write_frame(reply, {"type": "error", "error": "shutting_down"})
                    return
                write_frame(reply, {"type": "stopping", "pid": os.getpid(), "via": "control"})
                return
            # Anything else, including exec, is refused. This socket is not a
            # second way to reach O2 and must never become one.
            write_frame(reply, {"type": "error", "error": "invalid_request"})

    @staticmethod
    def _local_caller_disconnected(client: socket.socket) -> bool:
        """Return whether a queued local caller has closed its endpoint.

        The accepted request may have waited behind a long remote command. A
        zero-time readiness check distinguishes a closed socket (EOF) from a
        still-waiting caller without consuming any unexpected pending bytes.
        """

        readable, _, _ = select.select([client], [], [], 0)
        if not readable:
            return False
        try:
            return not client.recv(1, socket.MSG_PEEK)
        except OSError:
            return True

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
                # A stop can wait behind a long serialized command. By default a
                # caller whose local deadline expires has its request cancelled,
                # so a stop reported as failed does not shut the shared broker
                # down minutes later. A forced stop opts out of exactly that: it
                # is the only way to retire a busy broker, since a busy daemon
                # cannot reach any request until its command ends, and discarding
                # it here is what left o2_stop_broker unable to do so. Forced or
                # not, the in-flight command is never abandoned.
                if bool(request.get("force")):
                    self._stop_requested = True
                    self._write_state("stopping", stop_reason=str(request.get("reason") or "forced local request"))
                    with suppress(OSError, BrokerProtocolError):
                        write_frame(local, {"type": "stopping", "pid": os.getpid()})
                    return
                if self._local_caller_disconnected(client):
                    return
                try:
                    write_frame(local, {"type": "stopping", "pid": os.getpid()})
                except (OSError, BrokerProtocolError):
                    return
                self._stop_requested = True
                self._write_state("stopping", stop_reason=str(request.get("reason") or "local request"))
                return
            if request_type != "exec":
                with suppress(OSError, BrokerProtocolError):
                    write_frame(local, {"type": "error", "error": "invalid_request"})
                return
            request_timeout = request.get("timeout_seconds")
            valid_timeout = request_timeout is None or _is_bounded_positive_timeout(request_timeout)
            request_id = request.get("id")
            valid_request_id = utf8_text_within_limit(request_id, MAX_REQUEST_ID_BYTES)
            stdin_text = request.get("stdin")
            valid_stdin = stdin_text is None or utf8_text_within_limit(
                stdin_text,
                MAX_STDIN_BYTES,
                allow_empty=True,
            )
            expected_keys = {"type", "protocol", "id", "command", "timeout_seconds", "stdin"}
            valid_request = (
                set(request) == expected_keys
                and request.get("protocol") == PROTOCOL_VERSION
                and valid_request_id
                and command_within_exec_limit(request.get("command"))
                and valid_timeout
                and valid_stdin
            )
            if not valid_request:
                with suppress(OSError, BrokerProtocolError):
                    write_frame(local, {"type": "error", "error": "invalid_request"})
                return
            # The client enforces this ceiling too, but a client-side guard binds
            # only the callers that carry it. Every already-running MCP process
            # keeps its previous client until restarted, and the protocol still
            # permits seven days, so the daemon -- the one component every caller
            # shares -- has to be where a workstation-wide bound actually holds.
            #
            # Shorten the deadline rather than refuse. A refusal has no wire form
            # an older client reads correctly: it recognizes only `policy_denied`
            # and classifies anything else as a command that may already have
            # run, so refusing would tell exactly the callers this guard exists
            # for not to retry something that never left the workstation. A
            # clamp serves the request, holds the channel no longer than the
            # ceiling, and is reported in stderr rather than silently.
            clamped_from: float | None = None
            if request_timeout is None:
                # A deadline-free request is the one shape that can wedge this
                # daemon outright: the result read below has no watchdog for it,
                # so a helper that never replies blocks the shared channel with
                # nothing to time out. The protocol still carries JSON null, but
                # the daemon will not honour it as unbounded.
                clamped_from = math.inf
                request_timeout = MAX_COMMAND_TIMEOUT_SECONDS
            elif float(request_timeout) > MAX_COMMAND_TIMEOUT_SECONDS:
                clamped_from = float(request_timeout)
                request_timeout = MAX_COMMAND_TIMEOUT_SECONDS
            # Never forward caller-owned container structure. Rebuilding the
            # exact wire object keeps local/remote validation independent of
            # JSON parser recursion behavior across Python versions.
            remote_request = {key: request[key] for key in expected_keys}
            if clamped_from is not None:
                remote_request["timeout_seconds"] = MAX_COMMAND_TIMEOUT_SECONDS

            # The client checks policy before connecting, while this second gate
            # blocks hand-crafted local socket requests that try to bypass the
            # MCP layer. Holding the policy mutex through the remote frame write
            # also linearizes a concurrent disable: it either prevents this
            # command or observes a command that was already launched.
            try:
                with self.policy.serialize_reuse_launch():
                    with self._dispatch_lock:
                        # This acknowledgement is the execution boundary. If a
                        # queued caller has disconnected, do not forward its
                        # frame; otherwise it could perform a mutation after
                        # reporting a local timeout and invite a dangerous retry.
                        if self._local_caller_disconnected(client):
                            return
                        # A stop acknowledged on the control endpoint must mean
                        # no further work starts. This request was accepted
                        # before that stop and so is not in the queue the outer
                        # loop skips. Returning without a reply is the same
                        # cancellation the disconnect check performs: every
                        # client reads it as "not dispatched", exactly true.
                        if self._stop_requested:
                            return
                        try:
                            write_frame(local, {"type": "dispatched", "id": request["id"]})
                        except (OSError, BrokerProtocolError):
                            return
                        # The acknowledgement is also where this command starts
                        # owning the channel, and the remote frame write below
                        # can take seconds for a large stdin. Record it in
                        # memory here, with no receipt I/O under the policy
                        # mutex, so measured occupancy covers that write and a
                        # write that fails after acknowledgement is still
                        # attributed to this command.
                        # Epoch timestamps are what an operator reads, but they
                        # are not a safe basis for a duration: an NTP or manual
                        # step during a long command would inflate the occupancy
                        # metric or collapse it to zero. Measure elapsed time the
                        # way every other deadline in this daemon does.
                        # Occupancy is computed entirely inside this process, so
                        # it uses the ordinary monotonic clock; the published
                        # reading is subtracted by a different process and must
                        # name its clock, and is absent when it is unavailable.
                        dispatched_at = self.clock()
                        dispatched_monotonic = time.monotonic()
                        dispatched_shared = _interprocess_monotonic()
                        in_flight = {
                            "request_id": request_id,
                            "command": _command_fingerprint(request["command"]),
                            "timeout_seconds": request_timeout,
                            "dispatched_at": dispatched_at,
                        }
                        if dispatched_shared is not None:
                            in_flight["dispatched_clock_monotonic"] = dispatched_shared
                        self._in_flight = in_flight
                    # Past the acknowledgement this command is committed: the
                    # caller has been told it was dispatched, so the frame must
                    # be forwarded. That is why the lock ends above rather than
                    # around this write, which a large stdin can stretch to tens
                    # of seconds and which would otherwise delay every stop.
                    self._write_remote_frame_with_deadline(remote_in, remote_request)
            except O2PolicyError as exc:
                with suppress(OSError, BrokerProtocolError):
                    write_frame(local, {"type": "error", "error": "policy_denied", "message": str(exc)})
                return
            except (OSError, BrokerProtocolError) as exc:
                raise _O2BrokerTransportError(f"persistent remote stream failed: {exc}") from exc
            # Publish the dispatched command. No ping can be answered until it
            # finishes, so this receipt is the only local evidence of what the
            # channel is doing.
            #
            # This lands after the remote frame write, so another process still
            # reads `busy: false` for the duration of that write. Moving it
            # earlier would put an unbounded fsync inside
            # ``serialize_reuse_launch``, whose contract excludes unrelated
            # preparation and whose hold time is deliberately bounded by
            # explicit deadlines so an incident disable stays responsive. The
            # window is bounded by the write deadline, a write that fails inside
            # it is still attributed by the terminal receipt, and the occupancy
            # metric already covers it because timing starts at the
            # acknowledgement above.
            #
            # Publishing is diagnostic, so a transient filesystem failure must
            # not abort a command that is already running remotely; a directory
            # that no longer validates as owner-only is a security condition,
            # not a diagnostic one, and is deliberately left to fail closed.
            with suppress(OSError):
                self._write_state(self._running_status(), in_flight=in_flight)
            response = self._read_remote_frame_with_deadline(remote_out, request_timeout)
            if response.get("id") != request.get("id"):
                raise _O2BrokerTransportError("remote response id does not match the serialized request")
            self._commands_completed += 1
            completed_at = self.clock()
            # ``_write_state`` accumulates keys across writes, so the in-flight
            # record must be cleared explicitly rather than left to expire.
            self._in_flight = None
            with suppress(OSError):
                self._write_state(
                    self._running_status(),
                    last_command_at=completed_at,
                    in_flight=None,
                    last_command={
                        **{key: value for key, value in in_flight.items() if key != "dispatched_clock_monotonic"},
                        "completed_at": completed_at,
                        # Broker-observed occupancy: how long this command held the
                        # sole serialized channel, which is what starves other
                        # callers. The remote helper's own measurement is recorded
                        # beside it rather than in place of it.
                        "duration_seconds": max(0.0, time.monotonic() - dispatched_monotonic),
                        "remote_duration_seconds": _receipt_number(response.get("duration_seconds")),
                        "returncode": _receipt_returncode(response.get("returncode")),
                        "timed_out": bool(response.get("timed_out", False)),
                        "stdout_truncated": bool(response.get("stdout_truncated", False)),
                        "stderr_truncated": bool(response.get("stderr_truncated", False)),
                    },
                )
            if clamped_from is not None:
                # Say it where a caller already looks for notes the broker adds,
                # alongside the truncation notices, so the shortened deadline is
                # visible rather than inferred from an early timeout.
                requested = "no deadline" if math.isinf(clamped_from) else f"{clamped_from:.0f}s"
                note = (
                    f"command deadline reduced from {requested} to "
                    f"{MAX_COMMAND_TIMEOUT_SECONDS:.0f}s by the workstation ceiling on how long one command may "
                    "hold the shared serialized channel"
                )
                existing = response.get("stderr")
                response["stderr"] = f"{existing}\n{note}" if isinstance(existing, str) and existing else note
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
            self._bind_control_listener()
            self._control_thread = threading.Thread(target=self._serve_control, name="o2-broker-control", daemon=True)
            self._control_thread.start()
            # A remote protocol hello alone is not reusable authority. Clear
            # the retry cooldown only after the local socket and trusted ready
            # receipt have both been published successfully.
            if self.grant_id is not None:
                self.policy.finish_login_attempt(self.grant_id, outcome="success", returncode=0)
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
            with suppress(O2BrokerError, OSError):
                self._write_state("failed", error=failure, in_flight=self._in_flight)
            return 1
        finally:
            # Retire the control endpoint first. Closing its listener is what
            # releases the thread from accept; the flag alone would leave it
            # waiting out its timeout, and the socket file must not outlive the
            # daemon that owns it.
            with self._state_lock:
                self._control_stopping = True
            if self.control_listener is not None:
                with suppress(OSError):
                    self.control_listener.close()
                with suppress(FileNotFoundError):
                    self.paths.control_socket.unlink()
            if self._control_thread is not None:
                # An accepted control client may still be inside its own
                # request deadline, so wait past that rather than a shorter
                # fixed spell. The state-lock guard above is what keeps
                # correctness from depending on this completing.
                self._control_thread.join(timeout=self.local_request_timeout + 1.0)
            if self.listener is not None:
                self.listener.close()
                with suppress(FileNotFoundError):
                    self.paths.socket.unlink()
            self._terminate_transport(timeout=5.0)
            if self._lock_handle is not None:
                self._lock_handle.close()

        self._write_state(outcome, stopped_at=self.clock(), error=failure, in_flight=self._in_flight)
        return 0 if outcome == "stopped" else 1


def _validate_launch_payload(payload: Any) -> dict[str, Any]:
    """Validate launch data received only through the inherited pipe."""

    required_strings = (
        "broker_dir",
        "policy_file",
        "alias",
        "grant_id",
        "login_target",
        "launcher_client_id",
    )
    if not isinstance(payload, dict):
        raise O2BrokerStartupError("broker launch payload must be a JSON object")
    if payload.get("schema_version") != 1:
        raise O2BrokerStartupError("broker launch payload has an unsupported schema version")
    if any(not isinstance(payload.get(key), str) or not payload[key] for key in required_strings):
        raise O2BrokerStartupError("broker launch payload is missing required string fields")
    if payload["login_target"] not in {"login", "transfer"}:
        raise O2BrokerStartupError("broker launch login_target must be 'login' or 'transfer'")
    launcher_pid = payload.get("launcher_pid")
    if type(launcher_pid) is not int or launcher_pid <= 0:
        raise O2BrokerStartupError("broker launch launcher_pid must be a positive integer")
    argv = payload.get("transport_argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item for item in argv):
        raise O2BrokerStartupError("broker launch transport_argv must be a non-empty string list")
    destination = payload.get("destination")
    if not isinstance(destination, dict) or not all(
        isinstance(destination.get(key), str) and bool(destination[key]) for key in ("hostname", "user", "port")
    ):
        raise O2BrokerStartupError("broker launch destination must contain hostname, user, and port strings")
    startup_timeout = payload.get("startup_timeout")
    if not _is_bounded_positive_timeout(startup_timeout):
        raise O2BrokerStartupError("broker launch startup_timeout must be finite and positive")
    return payload


def _read_launch_fd(fd: int) -> dict[str, Any]:
    """Read one bounded, one-shot launch payload from an inherited pipe.

    Unlike a mode-0600 path, an anonymous pipe cannot be replaced by another
    same-UID task between parent authorization and child consumption. The
    daemon closes its sole descriptor after EOF, so no durable launch recipe
    remains available for replay.
    """

    if type(fd) is not int or fd < 0:
        raise O2BrokerStartupError("broker launch descriptor must be a non-negative integer")
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = os.read(fd, min(65536, MAX_LAUNCH_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_LAUNCH_BYTES:
                raise O2BrokerStartupError(f"broker launch payload exceeds {MAX_LAUNCH_BYTES} bytes")
    finally:
        with suppress(OSError):
            os.close(fd)
    try:
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise O2BrokerStartupError(f"broker launch payload is not valid UTF-8 JSON: {exc}") from exc
    return _validate_launch_payload(payload)


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
    parser.add_argument("--launch-fd", type=int, required=True)
    parser.add_argument("--ack-fd", type=int, required=True)
    args = parser.parse_args(argv)
    if not args.serve:
        parser.error("--serve is required")

    try:
        payload = _read_launch_fd(args.launch_fd)
        server = BrokerServer(
            paths=BrokerPaths(Path(payload["broker_dir"])),
            policy_file=Path(payload["policy_file"]),
            transport_argv=payload["transport_argv"],
            alias=payload["alias"],
            destination=payload["destination"],
            login_target=payload["login_target"],
            launcher_client_id=payload["launcher_client_id"],
            launcher_pid=payload["launcher_pid"],
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
