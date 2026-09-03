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
import math
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
LoginAuthorizationMethod = Literal["explicit_user_approval", "standing_on_vpn"]

SCHEMA_VERSION = 2
# Schema 1 predates the durable launch-evidence ledger and is still readable: it
# is migrated in memory on read and persisted as 2 by the next policy write, so
# an existing policy file is never invalidated merely by upgrading this code.
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
DEFAULT_GRANT_TTL_SECONDS = 300.0
DEFAULT_LOGIN_COOLDOWN_SECONDS = 300.0
MAX_EVENTS = 64
# The durable mint ledger is bounded too, but by refusing rather than evicting;
# see _append_launch_evidence_mint for why silently dropping one is the defect.
MAX_LAUNCH_EVIDENCE_MINTS = 256

# Digests recorded in the audit ledger must be literal hex, never a label.
_HEX_DIGITS = "0123456789abcdef"

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

    @property
    def generation(self) -> str | None:
        """Return the durable state generation used to prevent revision ABA."""

        if self.state is None:
            return None
        generation = self.state.get("generation")
        return generation if isinstance(generation, str) else None


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
    authorization_method: LoginAuthorizationMethod = "explicit_user_approval"

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, now: float | None = None) -> LoginGrant:
        """Validate and construct a grant from persisted JSON data.

        Args:
            payload: Serialized one-shot authorization.
            now: Current policy clock for detecting future-dated grants. The
                optional form keeps the dataclass parser usable in isolated
                helpers, while durable state validation always supplies it.
        """

        required_strings = ("id", "client_id", "target", "approval_reference")
        if any(not isinstance(payload.get(key), str) or not payload[key] for key in required_strings):
            raise O2PolicyInvalidError("login_grant is missing a required non-empty string field")
        if payload["target"] not in {"login", "transfer"}:
            raise O2PolicyInvalidError("login_grant.target must be 'login' or 'transfer'")
        # Older schema-v1 grants predate this explicit field and were always
        # issued from a fresh user approval. Preserve that interpretation while
        # validating every newly persisted authorization method.
        authorization_method = payload.get("authorization_method", "explicit_user_approval")
        if authorization_method not in {"explicit_user_approval", "standing_on_vpn"}:
            raise O2PolicyInvalidError("login_grant.authorization_method is unsupported")
        if type(payload.get("allow_offvpn")) is not bool:
            raise O2PolicyInvalidError("login_grant.allow_offvpn must be a boolean")
        if type(payload.get("remaining_attempts")) is not int or payload["remaining_attempts"] not in {0, 1}:
            raise O2PolicyInvalidError("login_grant.remaining_attempts must be zero or one")
        for key in ("created_at", "expires_at"):
            if type(payload.get(key)) not in {int, float} or not math.isfinite(payload[key]):
                raise O2PolicyInvalidError(f"login_grant.{key} must be a timestamp")
        if payload["expires_at"] <= payload["created_at"]:
            raise O2PolicyInvalidError("login_grant.expires_at must be later than created_at")
        if payload["expires_at"] - payload["created_at"] > DEFAULT_GRANT_TTL_SECONDS:
            raise O2PolicyInvalidError("login_grant lifetime exceeds the maximum authorization TTL")
        if now is not None and payload["created_at"] > now:
            # A clock rollback or tampered future timestamp must not stretch a
            # nominal five-minute approval into an authorization valid for
            # months or years before its future expiry is finally reached.
            raise O2PolicyInvalidError("login_grant.created_at cannot be later than the current policy clock")
        return cls(
            id=payload["id"],
            client_id=payload["client_id"],
            target=payload["target"],
            allow_offvpn=payload["allow_offvpn"],
            created_at=float(payload["created_at"]),
            expires_at=float(payload["expires_at"]),
            remaining_attempts=payload["remaining_attempts"],
            approval_reference=payload["approval_reference"],
            authorization_method=authorization_method,
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
        # Append rather than replace the policy suffix. A configured policy
        # already ending in ``.mutex`` must never alias the flock inode: policy
        # rewrites use os.replace(), and replacing the locked inode would let a
        # second process lock the new file inside the first critical section.
        self.mutex_path = self.path.with_name(f"{self.path.name}.mutex")
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

    @contextmanager
    def serialize_reuse_launch(self) -> Iterator[None]:
        """Recheck reuse policy and hold its mutex through one process spawn.

        Callers must enter this context immediately around creation of the
        remote-capable child, not around unrelated preparation or the child's
        full lifetime.  A concurrent :meth:`disable` therefore linearizes
        either before the check (and denies the launch) or after the child
        exists (where it is preserved as an already-running operation).
        """

        with self._locked():
            state = self._read_valid_state()
            if state["mode"] != "reuse_only":
                raise O2PolicyDeniedError(
                    f"O2 policy mode is '{state['mode']}'. Remote O2 operations are disabled; "
                    "local policy, socket, process, receipt, and transfer-log inspection remains available."
                )
            yield

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

    def enable_reuse(
        self,
        *,
        expected_revision: int,
        expected_generation: str,
        approval_reference: str,
    ) -> dict[str, Any]:
        """Transition from ``disabled`` to ``reuse_only`` after explicit approval."""

        reference = self._clean_reference(approval_reference, field="approval_reference")
        with self._locked():
            state = self._read_valid_state()
            self._require_revision(state, expected_revision, expected_generation)
            residual_grant = state.get("login_grant")
            if isinstance(residual_grant, dict):
                self._append_event(
                    state,
                    "login_grant_revoked_on_reuse_enable",
                    grant_id=residual_grant.get("id"),
                )
            # Global reuse approval is not login approval. Clear any residual
            # grant from an externally materialized disabled state so a prior
            # authorization cannot cross this policy transition.
            state["login_grant"] = None
            state["mode"] = "reuse_only"
            self._append_event(state, "policy_reuse_enabled", approval_reference=reference)
            return self._write_next_revision(state)

    def record_launch_evidence_mint(
        self,
        *,
        expected_revision: int,
        expected_generation: str,
        approval_reference: str,
        stage: str,
        job_id: str,
        package: str,
        evidence_sha256: str,
        plan_sha256: str,
    ) -> dict[str, Any]:
        """Record one operator-approved launch-evidence mint in the audit ledger.

        This is what authenticates the record. The caller must quote the current
        generation and revision -- obtainable only from a fresh status snapshot,
        so an approval cannot be replayed across an intervening policy write --
        together with a human approval reference. Minting is otherwise an
        ordinary read of cluster artifacts and would attest nothing.

        ``evidence_sha256`` is the digest of the *complete* evidence record this
        approval covers, and storing it here is what ties the approval to that
        exact content. Recording only the stage, job, and package would leave a
        holder of a legitimate record free to alter its runtime identities or
        package digests, recompute the record's own unkeyed digest, and keep an
        approval object that still agreed with the ledger. With the digest
        stored, that edit no longer matches the entry that approved it.

        The digest goes into ``launch_evidence_mints``, which event eviction
        never touches, as well as into the rolling event log. An attestation
        recorded only in a buffer bounded to MAX_EVENTS would stop being
        verifiable after enough unrelated policy traffic.

        It deliberately does NOT consume a login grant or change mode: attesting
        a finished run is not authority to start another one.
        """

        # The approval reference is free text and only ever lives here, so
        # collapsing its whitespace is right. The other three are also stored
        # verbatim in the evidence record and compared exactly by
        # verify_launch_evidence, so normalizing them here would make a freshly
        # minted record fail its own verification.
        reference = self._clean_reference(approval_reference, field="approval_reference")
        clean_stage = self._clean_literal(stage, field="stage", max_length=240)
        clean_job = self._clean_literal(job_id, field="job_id", max_length=240)
        clean_package = self._clean_literal(package, field="package", max_length=4096)
        evidence_digest = self._clean_digest(evidence_sha256, field="evidence_sha256")
        approved_plan_digest = self._clean_digest(plan_sha256, field="plan_sha256")
        with self._locked():
            state = self._read_valid_state()
            self._require_revision(state, expected_revision, expected_generation)
            # Every field the returned approval carries is built from this one
            # entry, so the record's operator_approval cannot name a reference,
            # client, revision or time the ledger does not also record. Editing
            # any of them in a record in hand then disagrees with the ledger.
            entry = {
                "at": self._clock(),
                "client_id": self.client_id,
                "approval_reference": reference,
                "stage": clean_stage,
                "job_id": clean_job,
                "package": clean_package,
                "evidence_sha256": evidence_digest,
                "plan_sha256": approved_plan_digest,
                # The revision this write is about to produce, so the entry names
                # the policy state the approval belongs to.
                "policy_revision": int(state.get("revision", 0)) + 1,
                "policy_generation": state.get("generation"),
            }
            # The durable ledger first: if it is full this raises before anything
            # is written, so a refused mint leaves no event claiming otherwise.
            self._append_launch_evidence_mint(state, entry)
            # The rolling event stays as the operational log; the ledger above is
            # what a record is verified against.
            self._append_event(
                state,
                "launch_evidence_minted",
                **{key: value for key, value in entry.items() if key not in {"at", "client_id"}},
            )
            self._write_next_revision(state)
        return {
            "approval_reference": entry["approval_reference"],
            "evidence_sha256": entry["evidence_sha256"],
            "plan_sha256": entry["plan_sha256"],
            "policy_revision": entry["policy_revision"],
            "policy_generation": entry["policy_generation"],
            "client_id": entry["client_id"],
            "approved_at": entry["at"],
        }

    def authorize_login(
        self,
        *,
        expected_revision: int,
        expected_generation: str,
        target: LoginTarget,
        allow_offvpn: bool,
        approval_reference: str,
        authorization_method: LoginAuthorizationMethod = "explicit_user_approval",
        ttl_seconds: float = DEFAULT_GRANT_TTL_SECONDS,
    ) -> LoginGrant:
        """Issue one short-lived login grant bound to this MCP client and host."""

        if target not in {"login", "transfer"}:
            raise O2LoginGrantError("target must be exactly 'login' or 'transfer'")
        if type(allow_offvpn) is not bool:
            raise O2LoginGrantError("allow_offvpn must be a boolean")
        if authorization_method not in {"explicit_user_approval", "standing_on_vpn"}:
            raise O2LoginGrantError("authorization_method is unsupported")
        if authorization_method == "standing_on_vpn" and allow_offvpn:
            raise O2LoginGrantError("standing_on_vpn authorization cannot allow off-VPN login")
        if not 0 < ttl_seconds <= DEFAULT_GRANT_TTL_SECONDS:
            raise O2LoginGrantError(
                f"ttl_seconds must be greater than zero and at most {DEFAULT_GRANT_TTL_SECONDS:.0f}"
            )
        reference = self._clean_reference(approval_reference, field="approval_reference")

        with self._locked():
            state = self._read_valid_state()
            self._require_revision(state, expected_revision, expected_generation)
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
                authorization_method=authorization_method,
            )
            state["login_grant"] = self._grant_dict(grant)
            self._append_event(
                state,
                "login_authorized",
                grant_id=grant.id,
                target=target,
                allow_offvpn=allow_offvpn,
                authorization_method=authorization_method,
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
            return self._consume_login_grant_locked(grant_id, target, launcher_pid=os.getpid())

    def revoke_unused_standing_grant(self, grant_id: str, *, reason: str) -> bool:
        """Revoke one unconsumed auto-start grant owned by this MCP process.

        Local preflight can still fail after route proof and grant creation. The
        caller must not strand an undisclosed client-bound grant that blocks the
        workstation for its full TTL. Explicit user grants and grants belonging
        to another process are never revoked through this cleanup path.
        """

        reference = self._clean_reference(reason, field="reason")
        with self._locked():
            state = self._read_valid_state()
            raw = state.get("login_grant")
            if not isinstance(raw, dict):
                return False
            grant = LoginGrant.from_dict(raw, now=self._clock())
            if grant.id != grant_id:
                return False
            if grant.client_id != self.client_id:
                raise O2LoginGrantError("Cannot revoke a standing grant owned by another MCP client.")
            if grant.authorization_method != "standing_on_vpn":
                raise O2LoginGrantError("Automatic cleanup cannot revoke an explicit user login grant.")
            state["login_grant"] = None
            self._append_event(
                state,
                "standing_login_grant_revoked",
                grant_id=grant.id,
                target=grant.target,
                reason=reference,
            )
            self._write_next_revision(state)
            return True

    @contextmanager
    def consume_login_grant_for_launch(self, grant_id: str, target: LoginTarget) -> Iterator[LoginGrant]:
        """Consume a grant and hold the policy mutex through process launch.

        The caller must enter this context immediately around the single
        authentication-capable runner invocation.  Holding the mutex makes the
        launch linearizable with :meth:`disable`: a disable either persists
        before consumption and prevents launch, or waits until the already
        authorized runner call has returned.  It can therefore never report a
        completed safety stop and then have this call spawn SSH afterward.

        The context deliberately covers no route/config preparation.  Those
        local-only operations happen before grant consumption so an unrelated
        local error does not destroy the one-shot authorization.
        """

        with self._locked():
            yield self._consume_login_grant_locked(grant_id, target, launcher_pid=os.getpid())

    @contextmanager
    def authorize_consumed_broker_launch(
        self,
        grant_id: str,
        target: LoginTarget,
        *,
        client_id: str,
        launcher_pid: int,
    ) -> Iterator[None]:
        """Validate one consumed grant while serializing the broker SSH spawn.

        The MCP parent first consumes the one-shot grant and starts a local
        daemon. The daemon then enters this gate immediately around its sole SSH
        ``Popen``. A concurrent disable therefore either wins first and blocks
        SSH, or waits until the already-authorized process exists.

        Binding the attempt to both the originating process identity and its PID
        prevents a retained or copied launch artifact from being replayed by a
        later process while an unrelated attempt happens to be active.
        """

        if type(launcher_pid) is not int or launcher_pid <= 0:
            raise O2LoginGrantError("broker launcher_pid must be a positive integer")
        with self._locked():
            state = self._read_valid_state()
            if state["mode"] != "reuse_only":
                raise O2PolicyDeniedError("O2 became disabled before the authorized broker could spawn SSH.")
            attempt = state.get("login_attempt")
            valid_attempt = (
                isinstance(attempt, dict)
                and attempt.get("outcome") == "active"
                and attempt.get("grant_id") == grant_id
                and attempt.get("target") == target
                and attempt.get("client_id") == client_id
                and attempt.get("launcher_pid") == launcher_pid
            )
            if not valid_attempt:
                raise O2LoginGrantError("The broker launch is not bound to the current active, consumed login attempt.")
            yield

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
            if attempt.get("outcome") != "active":
                # Terminal attempts are immutable evidence. In particular, a
                # later broker cleanup failure must not rewrite success as
                # failure after success already cleared the retry cooldown.
                raise O2PolicyConflictError(f"The login attempt for grant {grant_id} is already terminal.")
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

    def _consume_login_grant_locked(
        self,
        grant_id: str,
        target: LoginTarget,
        *,
        launcher_pid: int,
    ) -> LoginGrant:
        """Persist one active attempt while the caller owns ``self._locked``."""

        state = self._read_valid_state()
        if state["mode"] != "reuse_only":
            raise O2PolicyDeniedError("O2 became disabled before the login grant could be consumed.")
        grant = self._require_matching_grant(state, grant_id, target)
        now = self._clock()
        state["login_grant"] = None
        state["login_attempt"] = {
            "grant_id": grant.id,
            "client_id": grant.client_id,
            # The broker daemon must be a direct child of the process that
            # consumed this grant. This field is also useful incident evidence
            # for identifying which MCP process initiated the attempt.
            "launcher_pid": launcher_pid,
            "target": grant.target,
            "allow_offvpn": grant.allow_offvpn,
            "authorization_method": grant.authorization_method,
            "started_at": now,
            "finished_at": None,
            "outcome": "active",
            "returncode": None,
            "blocked_until": now + DEFAULT_LOGIN_COOLDOWN_SECONDS,
        }
        self._append_event(
            state,
            "login_grant_consumed",
            grant_id=grant.id,
            target=target,
            authorization_method=grant.authorization_method,
        )
        self._write_next_revision(state)
        return grant

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
            payload = json.loads(self.path.read_text())
        except (OSError, ValueError):
            # Nothing parsed, so there is nothing to salvage.
            return self._conservative_repair_state(metadata, salvaged_mints=[])
        try:
            return self._validate_state(payload)
        except O2PolicyInvalidError:
            # The JSON parsed but something in it is invalid. One malformed mint
            # must not cost every other attestation, so keep the entries that
            # stand on their own.
            return self._conservative_repair_state(metadata, salvaged_mints=self._salvage_mints(payload))

    def _salvage_mints(self, payload: Any) -> list[dict[str, Any]]:
        """Keep the mint entries that are individually valid, drop the rest.

        Repair replaces the whole file, so without this a single malformed entry
        would erase every other attestation in the ledger -- and those entries
        are the only durable authenticators their evidence records have. An
        entry that does not validate on its own authenticates nothing and is
        dropped; one that does is worth keeping regardless of what damaged its
        neighbour.
        """

        if not isinstance(payload, dict) or not isinstance(payload.get("launch_evidence_mints"), list):
            return []
        salvaged: list[dict[str, Any]] = []
        for entry in payload["launch_evidence_mints"]:
            if len(salvaged) >= MAX_LAUNCH_EVIDENCE_MINTS:
                break
            try:
                self._validate_launch_evidence_mints([entry])
            except O2PolicyInvalidError:
                continue
            salvaged.append(json.loads(json.dumps(entry)))
        return salvaged

    def _conservative_repair_state(
        self, metadata: os.stat_result, *, salvaged_mints: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Replace malformed JSON while retaining a file-age retry cooldown.

        Truncation destroys the exact login-attempt receipt, so repair cannot
        prove that no Duo-producing SSH process just ran. The policy file's
        modification time is the last remaining durable evidence: treat it as
        the start of an unknown failed attempt, capped at the current clock for
        future-dated metadata. A genuinely old malformed file has already aged
        out; a recent corruption retains the remainder of the global cooldown.
        """

        now = self._clock()
        modification_time = float(metadata.st_mtime)
        if not math.isfinite(modification_time):
            modification_time = now
        started_at = max(0.0, min(modification_time, now))
        state = self._initial_state()
        state["launch_evidence_mints"] = salvaged_mints
        repair_id = f"repair-{uuid.uuid4()}"
        state["login_attempt"] = {
            "grant_id": repair_id,
            "client_id": self.client_id,
            # The cooldown is workstation-global, so either concrete target is
            # sufficient when malformed bytes erased the original target.
            "target": "login",
            "allow_offvpn": False,
            "started_at": started_at,
            "finished_at": now,
            "outcome": "error",
            "returncode": None,
            "blocked_until": started_at + DEFAULT_LOGIN_COOLDOWN_SECONDS,
        }
        self._append_event(
            state,
            "policy_repaired_with_conservative_cooldown",
            repair_id=repair_id,
            source_modified_at=modification_time,
            # Say how much of the attestation ledger survived, so a repair that
            # silently emptied it is distinguishable from one that had nothing
            # to keep.
            launch_evidence_mints_kept=len(salvaged_mints),
        )
        return state

    def _validate_state(self, payload: Any) -> dict[str, Any]:
        """Strictly validate the durable fields needed for safe decisions."""

        if not isinstance(payload, dict):
            raise O2PolicyInvalidError("O2 policy root must be a JSON object")
        version = payload.get("schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise O2PolicyInvalidError(
                f"Unsupported O2 policy schema {version!r}; expected one of {SUPPORTED_SCHEMA_VERSIONS}."
            )
        # Detach before migrating so no caller's structure -- or a JSON decoder
        # cache -- is mutated by reading.
        payload = json.loads(json.dumps(payload))
        if version != SCHEMA_VERSION:
            # Schema 1 is otherwise entirely valid; it simply has no durable mint
            # ledger yet. Adding the empty list upgrades it in memory, and the
            # next policy write persists the new version. Nothing is rewritten
            # merely to read, and no existing state is invalidated.
            payload.setdefault("launch_evidence_mints", [])
            payload["schema_version"] = SCHEMA_VERSION
        revision = payload.get("revision")
        if type(revision) is not int or revision < 0:
            raise O2PolicyInvalidError("O2 policy revision must be a non-negative integer")
        generation = payload.get("generation")
        if not isinstance(generation, str) or not generation:
            raise O2PolicyInvalidError("O2 policy generation must be a non-empty UUID")
        try:
            uuid.UUID(generation)
        except ValueError as exc:
            raise O2PolicyInvalidError("O2 policy generation must be a valid UUID") from exc
        if payload.get("mode") not in {"disabled", "reuse_only"}:
            raise O2PolicyInvalidError("O2 policy mode must be 'disabled' or 'reuse_only'")
        if payload.get("login_grant") is not None:
            if not isinstance(payload["login_grant"], dict):
                raise O2PolicyInvalidError("login_grant must be null or an object")
            LoginGrant.from_dict(payload["login_grant"], now=self._clock())
        if payload.get("login_attempt") is not None:
            if not isinstance(payload["login_attempt"], dict):
                raise O2PolicyInvalidError("login_attempt must be null or an object")
            self._validate_login_attempt(payload["login_attempt"])
        if not isinstance(payload.get("events", []), list):
            raise O2PolicyInvalidError("events must be an array")
        self._validate_launch_evidence_mints(payload.get("launch_evidence_mints", []))
        return payload

    @staticmethod
    def _validate_login_attempt(attempt: dict[str, Any]) -> None:
        """Validate the complete cooldown receipt before any policy decision.

        A partial active receipt must never default its cooldown to zero: doing
        so would turn corrupt state into permission for another Duo-producing
        attempt. Terminal receipts remain authoritative until their persisted
        ``blocked_until`` time passes, so they receive the same strict checks.
        """

        required_strings = ("grant_id", "client_id", "target", "outcome")
        if any(not isinstance(attempt.get(field), str) or not attempt[field] for field in required_strings):
            raise O2PolicyInvalidError("login_attempt is missing a required non-empty string field")
        if attempt["target"] not in {"login", "transfer"}:
            raise O2PolicyInvalidError("login_attempt.target must be 'login' or 'transfer'")
        if attempt["outcome"] not in {
            "active",
            "success",
            "failed",
            "timed_out",
            "error",
            "stale",
        }:
            raise O2PolicyInvalidError("login_attempt.outcome is unsupported")
        if type(attempt.get("allow_offvpn")) is not bool:
            raise O2PolicyInvalidError("login_attempt.allow_offvpn must be a boolean")
        authorization_method = attempt.get("authorization_method", "explicit_user_approval")
        if authorization_method not in {"explicit_user_approval", "standing_on_vpn"}:
            raise O2PolicyInvalidError("login_attempt.authorization_method is unsupported")
        if authorization_method == "standing_on_vpn" and attempt["allow_offvpn"]:
            raise O2PolicyInvalidError("a standing_on_vpn login_attempt cannot allow off-VPN login")
        launcher_pid = attempt.get("launcher_pid")
        # Receipts created before the persistent-broker rollout did not record a
        # launcher PID. They remain valid cooldown evidence, but the new daemon
        # authorization gate never accepts a missing value for an SSH spawn.
        if launcher_pid is not None and (type(launcher_pid) is not int or launcher_pid <= 0):
            raise O2PolicyInvalidError("login_attempt.launcher_pid must be a positive integer when present")
        for field in ("started_at", "blocked_until"):
            value = attempt.get(field)
            if type(value) not in {int, float} or not math.isfinite(value):
                raise O2PolicyInvalidError(f"login_attempt.{field} must be a finite timestamp")
        finished_at = attempt.get("finished_at")
        returncode = attempt.get("returncode")
        if attempt["outcome"] == "active":
            if finished_at is not None or returncode is not None:
                raise O2PolicyInvalidError("an active login_attempt cannot be finished")
        else:
            if type(finished_at) not in {int, float} or not math.isfinite(finished_at):
                raise O2PolicyInvalidError("a terminal login_attempt requires a finite finished_at")
            if finished_at < attempt["started_at"]:
                raise O2PolicyInvalidError("login_attempt.finished_at cannot precede started_at")
        if returncode is not None and type(returncode) is not int:
            raise O2PolicyInvalidError("login_attempt.returncode must be null or an integer")
        if attempt["outcome"] == "success":
            # Successful master verification intentionally clears the retry
            # cooldown, but the receipt must prove that exact terminal state.
            if returncode != 0 or attempt["blocked_until"] != finished_at:
                raise O2PolicyInvalidError(
                    "a successful login_attempt must have returncode 0 and clear its cooldown at finished_at"
                )
        elif attempt["blocked_until"] < (attempt["started_at"] + DEFAULT_LOGIN_COOLDOWN_SECONDS):
            raise O2PolicyInvalidError("a non-success login_attempt must preserve the full retry cooldown")

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
        replacement, and a readable directory discloses policy metadata. Read
        paths call this check and fail closed. Mutation paths first use the
        descriptor-anchored legacy permission migration below, then reach this
        same validation while reading policy state.
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

    def _prepare_policy_directory(self) -> None:
        """Create or safely tighten the owned physical policy directory.

        o2-mcp 0.2 created ``~/.agent_locks`` under the caller's ordinary umask,
        which commonly left an otherwise owned directory at mode ``0755``.
        Mutations migrate that exact legacy shape to ``0700`` through a
        descriptor opened with ``O_NOFOLLOW``. Symlinks, non-directories, and
        foreign-owned objects remain hard failures. Arbitrary absolute policy
        paths are supported, but their parent permissions are never changed:
        automatic migration is limited to ``~/.agent_locks`` so unrelated
        shared/project directories cannot lose intentional access.
        """

        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            flags = os.O_RDONLY
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_DIRECTORY"):
                flags |= os.O_DIRECTORY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            directory_fd = os.open(self.path.parent, flags)
        except OSError as exc:
            raise O2PolicyInvalidError(f"Cannot prepare O2 policy directory {self.path.parent}: {exc}") from exc

        try:
            metadata = os.fstat(directory_fd)
            if not stat.S_ISDIR(metadata.st_mode):
                raise O2PolicyInvalidError(f"O2 policy directory must be a physical directory: {self.path.parent}")
            if metadata.st_uid != os.getuid():
                raise O2PolicyInvalidError(f"O2 policy directory is not owned by uid {os.getuid()}: {self.path.parent}")
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                legacy_directory = Path(os.path.abspath(Path.home() / ".agent_locks"))
                configured_directory = Path(os.path.abspath(self.path.parent))
                if configured_directory != legacy_directory:
                    raise O2PolicyInvalidError(
                        "O2 policy directory is permissive and is not the known "
                        f"legacy directory {legacy_directory}; create a dedicated "
                        "mode-0700 directory instead of changing unrelated contents"
                    )
                os.fchmod(directory_fd, 0o700)
                metadata = os.fstat(directory_fd)
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                raise O2PolicyInvalidError(
                    "O2 policy directory permission migration did not produce mode 0700: " f"{self.path.parent}"
                )
        except O2PolicyInvalidError:
            raise
        except OSError as exc:
            raise O2PolicyInvalidError(f"Cannot secure O2 policy directory {self.path.parent}: {exc}") from exc
        finally:
            os.close(directory_fd)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the stable workstation-wide mutex around one JSON mutation."""

        try:
            self._prepare_policy_directory()
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
        except O2PolicyInvalidError:
            handle.close()
            raise
        except OSError as exc:
            handle.close()
            raise O2PolicyInvalidError(f"Cannot lock O2 policy mutex at {self.mutex_path}: {exc}") from exc

        try:
            # Preserve exceptions from the protected operation. In particular,
            # the launch context may surface a subprocess OSError; mislabeling
            # it as a mutex error would hide the real failure after consuming
            # the one-shot authorization.
            yield
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
            # A fresh generation makes repair distinguishable from every state a
            # client could have observed before malformed JSON was replaced. The
            # numeric revision may restart, but the compare-and-swap pair cannot.
            "generation": str(uuid.uuid4()),
            "revision": 0,
            "mode": "disabled",
            "updated_at": self._clock(),
            "updated_by": {"client_id": self.client_id, "pid": os.getpid()},
            "login_grant": None,
            "login_attempt": None,
            "events": [],
            "launch_evidence_mints": [],
        }

    def _require_revision(
        self,
        state: dict[str, Any],
        expected_revision: int,
        expected_generation: str,
    ) -> None:
        """Reject a mutation whose approval predates any write or repair."""

        if type(expected_revision) is not int or expected_revision < 0:
            raise O2PolicyConflictError("expected_revision must be a non-negative integer")
        if not isinstance(expected_generation, str) or not expected_generation:
            raise O2PolicyConflictError("expected_generation must be a non-empty UUID")
        if state["revision"] != expected_revision or state["generation"] != expected_generation:
            raise O2PolicyConflictError(
                "O2 policy changed from expected generation/revision "
                f"{expected_generation}/{expected_revision} to "
                f"{state['generation']}/{state['revision']}; "
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
            "authorization_method": grant.authorization_method,
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

    def _append_launch_evidence_mint(self, state: dict[str, Any], entry: dict[str, Any]) -> None:
        """Append one mint to the ledger that ordinary event eviction never touches.

        Deliberately not `_append_event`. That list is a rolling buffer capped at
        MAX_EVENTS, and this ledger sees many events per session, so an approval
        recorded only there stops being verifiable once unrelated policy traffic
        pushes it out -- tamper-evidence with a shelf life, which is worse than
        not claiming any.

        This list is bounded too, but by refusing rather than evicting. Dropping
        the oldest attestation to make room is exactly the defect being fixed, so
        a full ledger is a loud and recoverable failure instead of a silent one.
        """

        mints = state.setdefault("launch_evidence_mints", [])
        if len(mints) >= MAX_LAUNCH_EVIDENCE_MINTS:
            raise O2PolicyInvalidError(
                f"the launch-evidence ledger already holds its maximum of {MAX_LAUNCH_EVIDENCE_MINTS} mints. "
                f"Archive and prune {self.path} before minting again: evicting an older attestation to make "
                "room would silently make that record unverifiable."
            )
        mints.append(dict(entry))

    @staticmethod
    def _clean_reference(value: str, *, field: str) -> str:
        """Validate short audit metadata without storing an entire chat transcript."""

        if not isinstance(value, str) or not value.strip():
            raise O2PolicyInvalidError(f"{field} must be a non-empty string")
        cleaned = " ".join(value.split())
        if len(cleaned) > 240:
            raise O2PolicyInvalidError(f"{field} must be at most 240 characters")
        return cleaned

    @staticmethod
    def _validate_launch_evidence_mints(mints: Any) -> None:
        """Validate the durable ledger, not merely its shape.

        These entries are the sole durable authenticators for evidence records:
        `verify_launch_evidence` compares a record field by field against one of
        them. An entry that is a JSON object but missing fields, or carrying a
        malformed digest, would therefore be surfaced through status and
        preserved across writes while authenticating nothing -- and a record
        checked against it could pass on absent-equals-absent. Refusing makes a
        damaged ledger loud rather than quietly useless.
        """

        if not isinstance(mints, list):
            raise O2PolicyInvalidError("launch_evidence_mints must be an array")
        if len(mints) > MAX_LAUNCH_EVIDENCE_MINTS:
            raise O2PolicyInvalidError(
                f"launch_evidence_mints holds {len(mints)} entries, more than the {MAX_LAUNCH_EVIDENCE_MINTS} maximum"
            )
        for index, entry in enumerate(mints):
            if not isinstance(entry, dict):
                raise O2PolicyInvalidError(f"launch_evidence_mints[{index}] must be an object")
            for field in ("approval_reference", "stage", "job_id", "package", "client_id", "policy_generation"):
                value = entry.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise O2PolicyInvalidError(f"launch_evidence_mints[{index}].{field} must be a non-empty string")
            for field in ("evidence_sha256", "plan_sha256"):
                value = entry.get(field)
                if not isinstance(value, str) or len(value) != 64 or value.strip(_HEX_DIGITS):
                    raise O2PolicyInvalidError(
                        f"launch_evidence_mints[{index}].{field} must be a 64-character SHA-256 hex digest"
                    )
            recorded_at = entry.get("at")
            if not isinstance(recorded_at, (int, float)) or isinstance(recorded_at, bool):
                raise O2PolicyInvalidError(f"launch_evidence_mints[{index}].at must be a number")
            if not math.isfinite(float(recorded_at)) or float(recorded_at) < 0:
                raise O2PolicyInvalidError(f"launch_evidence_mints[{index}].at must be a finite, non-negative time")
            revision = entry.get("policy_revision")
            if type(revision) is not int or revision < 0:
                raise O2PolicyInvalidError(
                    f"launch_evidence_mints[{index}].policy_revision must be a non-negative integer"
                )
            try:
                uuid.UUID(entry["policy_generation"])
            except ValueError as exc:
                raise O2PolicyInvalidError(
                    f"launch_evidence_mints[{index}].policy_generation must be a valid UUID"
                ) from exc

    @staticmethod
    def _clean_literal(value: str, *, field: str, max_length: int) -> str:
        """Validate an audited value without normalizing the characters in it.

        `_clean_reference` collapses runs of whitespace, which is right for a
        free-text approval note and wrong for anything the evidence record also
        stores verbatim: the ledger would file the mint under a value the record
        does not carry, and `verify_launch_evidence` compares them exactly, so a
        freshly minted record would fail its own verification. Control
        characters are still refused -- they cannot appear in a value this
        server accepted, and would corrupt an audit line.
        """

        if not isinstance(value, str) or not value.strip():
            raise O2PolicyInvalidError(f"{field} must be a non-empty string")
        cleaned = value.strip()
        if len(cleaned) > max_length:
            raise O2PolicyInvalidError(f"{field} must be at most {max_length} characters")
        if any(ord(character) < 32 or ord(character) == 127 for character in cleaned):
            raise O2PolicyInvalidError(f"{field} must not contain control characters")
        return cleaned

    @staticmethod
    def _clean_digest(value: str, *, field: str) -> str:
        """Require a literal SHA-256 hex digest for a binding the ledger records."""

        cleaned = value.strip().lower() if isinstance(value, str) else ""
        if len(cleaned) != 64 or cleaned.strip(_HEX_DIGITS):
            raise O2PolicyInvalidError(f"{field} must be a 64-character SHA-256 hex digest")
        return cleaned
