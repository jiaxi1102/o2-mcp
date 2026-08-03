"""Workstation-wide policy state for HMS O2 access.

The policy store replaces the historical collection of ``O2_DISABLED`` files,
login mutex files, and retry receipts with one durable JSON state machine.  Every
MCP process reads and mutates the same file, while a separate ``flock`` target is
used only to serialize atomic rewrites; the mutex contains no policy of its own.

Two durable modes are intentionally supported:

``disabled``
    No new remote O2 operation may be initiated.  Local diagnostics and explicit
    local process control remain available, and existing detached transfers are
    not terminated automatically.

``reuse_only``
    Existing, exactly pinned ControlMaster sockets may be reused with all SSH
    authentication methods disabled.  A new authentication is possible only
    after a short-lived, client-bound, one-attempt login grant is atomically
    consumed.

There is deliberately no durable ``normal`` mode.  A persistent state that lets
any later task pass ``allow_new_login=True`` would recreate the cross-task Duo
storm this module exists to prevent.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

PolicyMode = Literal["disabled", "reuse_only"]
LoginTarget = Literal["login", "transfer"]

SCHEMA_VERSION = 1
DEFAULT_GRANT_TTL_SECONDS = 300.0
DEFAULT_LOGIN_COOLDOWN_SECONDS = 300.0
MAX_EVENTS = 64

# All connections created by one MCP server process share an identity.  Binding a
# grant to this value prevents another concurrently running task from consuming a
# user's authorization merely because both tasks can read the global policy file.
_PROCESS_CLIENT_ID = f"{os.getpid()}-{uuid.uuid4()}"


class O2PolicyError(RuntimeError):
    """Base class for policy-state and authorization failures."""


class O2PolicyDeniedError(O2PolicyError):
    """Raised when the current durable mode forbids a remote O2 operation."""


class O2PolicyInvalidError(O2PolicyError):
    """Raised when policy state is missing, unsafe, or malformed."""


class O2PolicyConflictError(O2PolicyError):
    """Raised when a caller tries to mutate a stale policy revision."""


class O2LoginGrantError(O2PolicyError):
    """Raised when a one-shot login authorization cannot be issued or consumed."""


@dataclass(frozen=True)
class PolicySnapshot:
    """Validated policy data plus a local-only diagnostic error, if any.

    Invalid or missing state always has an effective mode of ``disabled``.  The
    original error remains visible to ``o2_local_status`` without allowing a
    remote caller to mistake fail-closed behavior for a valid disabled record.
    """

    path: Path
    valid: bool
    effective_mode: PolicyMode
    state: dict[str, Any] | None
    error: str | None = None

    @property
    def revision(self) -> int:
        """Return the observed revision, using zero before initialization."""

        if self.state is None:
            return 0
        revision = self.state.get("revision", 0)
        return revision if type(revision) is int else 0


@dataclass(frozen=True)
class LoginGrant:
    """A validated one-attempt authorization to authenticate to one O2 host."""

    id: str
    client_id: str
    target: LoginTarget
    allow_offvpn: bool
    created_at: float
    expires_at: float
    remaining_attempts: int
    approval_reference: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LoginGrant:
        """Validate and construct a grant from persisted JSON data."""

        required_strings = ("id", "client_id", "target", "approval_reference")
        if any(not isinstance(payload.get(key), str) or not payload[key] for key in required_strings):
            raise O2PolicyInvalidError("login_grant is missing a required non-empty string field")
        if payload["target"] not in {"login", "transfer"}:
            raise O2PolicyInvalidError("login_grant.target must be 'login' or 'transfer'")
        if type(payload.get("allow_offvpn")) is not bool:
            raise O2PolicyInvalidError("login_grant.allow_offvpn must be a boolean")
        if type(payload.get("remaining_attempts")) is not int or payload["remaining_attempts"] not in {0, 1}:
            raise O2PolicyInvalidError("login_grant.remaining_attempts must be zero or one")
        for key in ("created_at", "expires_at"):
            if type(payload.get(key)) not in {int, float}:
                raise O2PolicyInvalidError(f"login_grant.{key} must be a timestamp")
        return cls(
            id=payload["id"],
            client_id=payload["client_id"],
            target=payload["target"],
            allow_offvpn=payload["allow_offvpn"],
            created_at=float(payload["created_at"]),
            expires_at=float(payload["expires_at"]),
            remaining_attempts=payload["remaining_attempts"],
            approval_reference=payload["approval_reference"],
        )


class O2PolicyStore:
    """Read and atomically mutate the workstation-wide O2 policy file.

    Args:
        path: Authoritative JSON policy path.
        client_id: Stable identity for one MCP process.  Tests may inject a
            deterministic value; production uses a random process-local value.
        clock: Wall-clock provider used for durable expiry timestamps.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        client_id: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path).expanduser()
        self.mutex_path = self.path.with_suffix(".mutex")
        self.client_id = client_id or _PROCESS_CLIENT_ID
        self._clock = clock

    # -- public read/gate API -------------------------------------------------
    def snapshot(self) -> PolicySnapshot:
        """Return validated local state, failing closed without raising.

        This method performs filesystem and JSON inspection only.  It never
        invokes SSH, resolves a remote hostname, or touches a ControlMaster.
        """

        try:
            state = self._read_valid_state()
        except O2PolicyInvalidError as exc:
            return PolicySnapshot(
                path=self.path,
                valid=False,
                effective_mode="disabled",
                state=None,
                error=str(exc),
            )
        return PolicySnapshot(
            path=self.path,
            valid=True,
            effective_mode=state["mode"],
            state=state,
        )

    def require_reuse_allowed(self) -> dict[str, Any]:
        """Return valid state when remote reuse is allowed, otherwise raise."""

        state = self._read_valid_state()
        if state["mode"] != "reuse_only":
            raise O2PolicyDeniedError(
                f"O2 policy mode is '{state['mode']}'. Remote O2 operations are disabled; "
                "local policy, socket, process, receipt, and transfer-log inspection remains available."
            )
        return state

    def preview_login_grant(self, grant_id: str, target: LoginTarget) -> LoginGrant:
        """Validate a grant without consuming it for local route preflight.

        Consumption performs the same checks again while holding the global
        mutex, so a concurrent disable or expiry between preview and use still
        fails closed.
        """

        state = self.require_reuse_allowed()
        return self._require_matching_grant(state, grant_id, target)

    # -- public mutations -----------------------------------------------------
    def disable(self, *, reason: str) -> dict[str, Any]:
        """Materialize ``disabled`` and revoke any unconsumed login grant.

        A safety stop deliberately does not require an expected revision.  It
        must win over stale clients and remain possible when the JSON is missing
        or malformed, provided the filesystem object itself is safe to replace.
        """

        clean_reason = self._clean_reference(reason, field="reason")
        with self._locked():
            state = self._read_for_repair_or_initialize()
            state["mode"] = "disabled"
            state["login_grant"] = None
            self._append_event(state, "policy_disabled", reason=clean_reason)
            return self._write_next_revision(state)

    def enable_reuse(self, *, expected_revision: int, approval_reference: str) -> dict[str, Any]:
        """Transition from ``disabled`` to ``reuse_only`` after explicit approval."""

        reference = self._clean_reference(approval_reference, field="approval_reference")
        with self._locked():
            state = self._read_valid_state()
            self._require_revision(state, expected_revision)
            state["mode"] = "reuse_only"
            self._append_event(state, "policy_reuse_enabled", approval_reference=reference)
            return self._write_next_revision(state)

    def authorize_login(
        self,
        *,
        expected_revision: int,
        target: LoginTarget,
        allow_offvpn: bool,
        approval_reference: str,
        ttl_seconds: float = DEFAULT_GRANT_TTL_SECONDS,
    ) -> LoginGrant:
        """Issue one short-lived login grant bound to this MCP client and host."""

        if target not in {"login", "transfer"}:
            raise O2LoginGrantError("target must be exactly 'login' or 'transfer'")
        if type(allow_offvpn) is not bool:
            raise O2LoginGrantError("allow_offvpn must be a boolean")
        if not 0 < ttl_seconds <= DEFAULT_GRANT_TTL_SECONDS:
            raise O2LoginGrantError(
                f"ttl_seconds must be greater than zero and at most {DEFAULT_GRANT_TTL_SECONDS:.0f}"
            )
        reference = self._clean_reference(approval_reference, field="approval_reference")

        with self._locked():
            state = self._read_valid_state()
            self._require_revision(state, expected_revision)
            if state["mode"] != "reuse_only":
                raise O2PolicyDeniedError("Enable reuse_only mode before authorizing a new O2 login.")

            now = self._clock()
            self._expire_stale_authorization(state, now)
            if state.get("login_grant") is not None:
                raise O2LoginGrantError("Another unexpired workstation-wide login grant already exists.")

            attempt = state.get("login_attempt")
            if isinstance(attempt, dict):
                # ``blocked_until`` governs the global cooldown independently
                # of whether the SSH subprocess is still active or has already
                # reported failure.  Restricting this check to ``active`` would
                # permit an immediate retry after a failed/timed-out Duo push.
                blocked_until = float(attempt.get("blocked_until", 0.0))
                if now < blocked_until:
                    remaining = blocked_until - now
                    raise O2LoginGrantError(
                        f"A prior O2 login attempt is still active or cooling down for {remaining:.1f}s. "
                        "Do not authorize another Duo-pushing attempt yet."
                    )
                if attempt.get("outcome") == "active":
                    attempt["outcome"] = "stale"
                    attempt["finished_at"] = now

            grant = LoginGrant(
                id=str(uuid.uuid4()),
                client_id=self.client_id,
                target=target,
                allow_offvpn=allow_offvpn,
                created_at=now,
                expires_at=now + ttl_seconds,
                remaining_attempts=1,
                approval_reference=reference,
            )
            state["login_grant"] = self._grant_dict(grant)
            self._append_event(
                state,
                "login_authorized",
                grant_id=grant.id,
                target=target,
                allow_offvpn=allow_offvpn,
                approval_reference=reference,
            )
            self._write_next_revision(state)
            return grant

    def consume_login_grant(self, grant_id: str, target: LoginTarget) -> LoginGrant:
        """Consume a matching grant and persist an active attempt before SSH.

        The grant is removed in the same locked rewrite that creates the attempt
        receipt.  A timeout, exception, or process crash therefore cannot leave a
        reusable authorization for another task.
        """

        with self._locked():
            state = self._read_valid_state()
            if state["mode"] != "reuse_only":
                raise O2PolicyDeniedError("O2 became disabled before the login grant could be consumed.")
            grant = self._require_matching_grant(state, grant_id, target)
            now = self._clock()
            state["login_grant"] = None
            state["login_attempt"] = {
                "grant_id": grant.id,
                "client_id": grant.client_id,
                "target": grant.target,
                "allow_offvpn": grant.allow_offvpn,
                "started_at": now,
                "finished_at": None,
                "outcome": "active",
                "returncode": None,
                "blocked_until": now + DEFAULT_LOGIN_COOLDOWN_SECONDS,
            }
            self._append_event(state, "login_grant_consumed", grant_id=grant.id, target=target)
            self._write_next_revision(state)
            return grant

    def finish_login_attempt(self, grant_id: str, *, outcome: str, returncode: int | None) -> None:
        """Record one login attempt's terminal outcome without changing mode.

        A concurrent safety stop may have advanced the revision while SSH was in
        flight.  This method rereads and merges only the matching attempt fields,
        preserving the newer mode and any revoked authorization.
        """

        if outcome not in {"success", "failed", "timed_out", "error"}:
            raise ValueError(f"unsupported login outcome: {outcome}")
        with self._locked():
            state = self._read_valid_state()
            attempt = state.get("login_attempt")
            if not isinstance(attempt, dict) or attempt.get("grant_id") != grant_id:
                # A missing/replaced receipt is a policy-integrity failure, but it
                # must not hide the SSH result or rewrite unrelated current state.
                raise O2PolicyConflictError(
                    f"The active login attempt for grant {grant_id} is no longer present in policy state."
                )
            attempt["finished_at"] = self._clock()
            attempt["outcome"] = outcome
            attempt["returncode"] = returncode
            # Successful socket verification makes reuse authoritative, so no
            # retry cooldown is necessary.  Failures retain blocked_until.
            if outcome == "success":
                attempt["blocked_until"] = attempt["finished_at"]
            self._append_event(
                state,
                "login_attempt_finished",
                grant_id=grant_id,
                target=attempt.get("target"),
                outcome=outcome,
                returncode=returncode,
            )
            self._write_next_revision(state)

    # -- validation and persistence ------------------------------------------
    def _read_valid_state(self) -> dict[str, Any]:
        """Read one safe, regular, owned, mode-0600 policy JSON file."""

        self._validate_parent_directory()
        try:
            metadata = self.path.lstat()
        except FileNotFoundError as exc:
            raise O2PolicyInvalidError(
                f"O2 policy is not initialized at {self.path}; effective mode is disabled."
            ) from exc
        except OSError as exc:
            raise O2PolicyInvalidError(f"Cannot inspect O2 policy at {self.path}: {exc}") from exc
        self._validate_file_metadata(self.path, metadata)
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            raise O2PolicyInvalidError(f"Cannot read valid O2 policy JSON at {self.path}: {exc}") from exc
        return self._validate_state(payload)

    def _read_for_repair_or_initialize(self) -> dict[str, Any]:
        """Return current state or a conservative skeleton for ``disable``.

        Missing or malformed JSON can be safely replaced with disabled state, but
        an unsafe filesystem object (symlink, foreign owner, non-regular file, or
        permissive mode) is never overwritten automatically.
        """

        self._validate_parent_directory()
        try:
            metadata = self.path.lstat()
        except FileNotFoundError:
            return self._initial_state()
        except OSError as exc:
            raise O2PolicyInvalidError(f"Cannot inspect O2 policy at {self.path}: {exc}") from exc
        self._validate_file_metadata(self.path, metadata)
        try:
            return self._validate_state(json.loads(self.path.read_text()))
        except (OSError, ValueError, O2PolicyInvalidError):
            return self._initial_state()

    def _validate_state(self, payload: Any) -> dict[str, Any]:
        """Strictly validate the durable fields needed for safe decisions."""

        if not isinstance(payload, dict):
            raise O2PolicyInvalidError("O2 policy root must be a JSON object")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise O2PolicyInvalidError(
                f"Unsupported O2 policy schema {payload.get('schema_version')!r}; expected {SCHEMA_VERSION}."
            )
        revision = payload.get("revision")
        if type(revision) is not int or revision < 0:
            raise O2PolicyInvalidError("O2 policy revision must be a non-negative integer")
        if payload.get("mode") not in {"disabled", "reuse_only"}:
            raise O2PolicyInvalidError("O2 policy mode must be 'disabled' or 'reuse_only'")
        if payload.get("login_grant") is not None:
            if not isinstance(payload["login_grant"], dict):
                raise O2PolicyInvalidError("login_grant must be null or an object")
            LoginGrant.from_dict(payload["login_grant"])
        if payload.get("login_attempt") is not None and not isinstance(payload["login_attempt"], dict):
            raise O2PolicyInvalidError("login_attempt must be null or an object")
        if not isinstance(payload.get("events", []), list):
            raise O2PolicyInvalidError("events must be an array")
        # Return a detached mutable copy so callers cannot accidentally mutate a
        # structure shared with an input fixture or JSON decoder cache.
        return json.loads(json.dumps(payload))

    @staticmethod
    def _validate_file_metadata(path: Path, metadata: os.stat_result) -> None:
        """Reject policy/mutex paths that could redirect or be edited by others."""

        if not stat.S_ISREG(metadata.st_mode):
            raise O2PolicyInvalidError(f"O2 policy object must be a regular non-symlink file: {path}")
        if metadata.st_uid != os.getuid():
            raise O2PolicyInvalidError(f"O2 policy object is not owned by uid {os.getuid()}: {path}")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise O2PolicyInvalidError(f"O2 policy object must not be accessible by group or other users: {path}")

    def _validate_parent_directory(self) -> None:
        """Reject a policy directory that another local account can replace.

        File mode ``0600`` is insufficient when the containing directory is
        accessible by group or other users: a writable directory permits file
        replacement, and a readable directory discloses policy metadata.  Both
        read and mutation paths call this check so unsafe permissions fail
        closed rather than being silently repaired.
        """

        try:
            metadata = self.path.parent.lstat()
        except FileNotFoundError as exc:
            raise O2PolicyInvalidError(f"O2 policy directory does not exist: {self.path.parent}") from exc
        except OSError as exc:
            raise O2PolicyInvalidError(f"Cannot inspect O2 policy directory {self.path.parent}: {exc}") from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise O2PolicyInvalidError(f"O2 policy directory must be a physical directory: {self.path.parent}")
        if metadata.st_uid != os.getuid():
            raise O2PolicyInvalidError(f"O2 policy directory is not owned by uid {os.getuid()}: {self.path.parent}")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise O2PolicyInvalidError(
                "O2 policy directory must not be accessible by group or other users: " f"{self.path.parent}"
            )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the stable workstation-wide mutex around one JSON mutation."""

        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            # An existing permissive or aliased directory is a policy-integrity
            # error.  Do not chmod it automatically: mutation must fail closed
            # until the human repairs the unsafe filesystem boundary explicitly.
            self._validate_parent_directory()
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(self.mutex_path, flags, 0o600)
        except O2PolicyInvalidError:
            raise
        except OSError as exc:
            raise O2PolicyInvalidError(f"Cannot create O2 policy mutex at {self.mutex_path}: {exc}") from exc

        handle = os.fdopen(fd, "r+")
        try:
            metadata = os.fstat(handle.fileno())
            self._validate_file_metadata(self.mutex_path, metadata)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        except OSError as exc:
            raise O2PolicyInvalidError(f"Cannot lock O2 policy mutex at {self.mutex_path}: {exc}") from exc
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def _write_next_revision(self, state: dict[str, Any]) -> dict[str, Any]:
        """Increment, stamp, atomically replace, and reread one policy state."""

        state["schema_version"] = SCHEMA_VERSION
        state["revision"] = int(state.get("revision", 0)) + 1
        state["updated_at"] = self._clock()
        state["updated_by"] = {"client_id": self.client_id, "pid": os.getpid()}
        serialized = json.dumps(state, indent=2, sort_keys=True) + "\n"

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                os.chmod(temp_path, 0o600)
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
            temp_path = None
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            raise O2PolicyInvalidError(f"Cannot atomically write O2 policy at {self.path}: {exc}") from exc
        finally:
            if temp_path is not None:
                with suppress(FileNotFoundError):
                    temp_path.unlink()
        return self._read_valid_state()

    # -- small state-machine helpers -----------------------------------------
    def _initial_state(self) -> dict[str, Any]:
        """Return conservative state used only for initialization or repair."""

        return {
            "schema_version": SCHEMA_VERSION,
            "revision": 0,
            "mode": "disabled",
            "updated_at": self._clock(),
            "updated_by": {"client_id": self.client_id, "pid": os.getpid()},
            "login_grant": None,
            "login_attempt": None,
            "events": [],
        }

    def _require_revision(self, state: dict[str, Any], expected_revision: int) -> None:
        """Reject a mutation whose approval was based on stale local state."""

        if type(expected_revision) is not int or expected_revision < 0:
            raise O2PolicyConflictError("expected_revision must be a non-negative integer")
        if state["revision"] != expected_revision:
            raise O2PolicyConflictError(
                f"O2 policy changed from expected revision {expected_revision} to {state['revision']}; "
                "reread o2_local_status before requesting another transition."
            )

    def _require_matching_grant(
        self,
        state: dict[str, Any],
        grant_id: str,
        target: LoginTarget,
    ) -> LoginGrant:
        """Return the exact usable grant or raise a scoped authorization error."""

        raw = state.get("login_grant")
        if not isinstance(raw, dict):
            raise O2LoginGrantError("No one-shot O2 login grant is active.")
        grant = LoginGrant.from_dict(raw)
        if grant.id != grant_id:
            raise O2LoginGrantError("The supplied O2 login grant id does not match current policy state.")
        if grant.client_id != self.client_id:
            raise O2LoginGrantError("The O2 login grant belongs to a different MCP task/client.")
        if grant.target != target:
            raise O2LoginGrantError(
                f"The O2 login grant is scoped to '{grant.target}', not requested target '{target}'."
            )
        if grant.remaining_attempts != 1:
            raise O2LoginGrantError("The O2 login grant has already been consumed.")
        if self._clock() >= grant.expires_at:
            raise O2LoginGrantError("The one-shot O2 login grant has expired; request fresh explicit approval.")
        return grant

    def _expire_stale_authorization(self, state: dict[str, Any], now: float) -> None:
        """Clear an expired unconsumed grant before considering a new grant."""

        raw = state.get("login_grant")
        if not isinstance(raw, dict):
            return
        grant = LoginGrant.from_dict(raw)
        if now >= grant.expires_at:
            state["login_grant"] = None
            self._append_event(state, "login_grant_expired", grant_id=grant.id, target=grant.target)

    @staticmethod
    def _grant_dict(grant: LoginGrant) -> dict[str, Any]:
        """Serialize a grant without relying on a non-stdlib dataclass helper."""

        return {
            "id": grant.id,
            "client_id": grant.client_id,
            "target": grant.target,
            "allow_offvpn": grant.allow_offvpn,
            "authorization_method": "explicit_user_approval",
            "approval_reference": grant.approval_reference,
            "created_at": grant.created_at,
            "expires_at": grant.expires_at,
            "remaining_attempts": grant.remaining_attempts,
        }

    def _append_event(self, state: dict[str, Any], event: str, **details: Any) -> None:
        """Append a compact bounded audit event to the authoritative state."""

        events = state.setdefault("events", [])
        events.append({"at": self._clock(), "event": event, "client_id": self.client_id, **details})
        del events[:-MAX_EVENTS]

    @staticmethod
    def _clean_reference(value: str, *, field: str) -> str:
        """Validate short audit metadata without storing an entire chat transcript."""

        if not isinstance(value, str) or not value.strip():
            raise O2PolicyInvalidError(f"{field} must be a non-empty string")
        cleaned = " ".join(value.split())
        if len(cleaned) > 240:
            raise O2PolicyInvalidError(f"{field} must be at most 240 characters")
        return cleaned
