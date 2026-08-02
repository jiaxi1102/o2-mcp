"""Safe, reusable SSH command channel for HMS O2.

A Python port of ``scripts/o2_ssh_master.sh`` that preserves its safety contract
exactly, but is testable and composable:

- The user-level ``~/.agent_locks/O2_DISABLED`` lock and legacy project lock are
  hard stops on every operation.
- All SSH uses BatchMode (public key only) — a dead master or missing key fails
  fast instead of triggering a Duo/MFA phone prompt.
- Remote commands run only through an already-established ControlMaster socket;
  opening a NEW login requires an explicit opt-in (one approved MFA verification).

The actual subprocess call is injected (``runner``) so the whole class is unit
tested offline without ever touching the network.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from o2mcp.config import O2Config


class O2LockedError(RuntimeError):
    """Raised when the local O2 safety lock forbids any cluster operation."""


class O2MasterUnavailableError(RuntimeError):
    """Raised when a command needs the ControlMaster but none is running."""


class O2OffVpnError(RuntimeError):
    """Raised when opening a new login would egress off the HMS VPN (→ a Duo push)."""


class O2LoginCoordinationError(RuntimeError):
    """Raised when a new O2 login cannot be serialized safely across processes."""


@dataclass
class CommandResult:
    """The outcome of a single subprocess invocation."""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


# A runner takes (argv, timeout, input_text) and returns a CommandResult.
Runner = Callable[[list[str], Optional[float], Optional[str]], CommandResult]


def default_runner(argv: list[str], timeout: float | None, input_text: str | None) -> CommandResult:
    """Run a command via subprocess without exposing the MCP protocol stream.

    ``o2-mcp`` is a stdio MCP server, so inheriting the parent process's stdin
    would let a child ``ssh``/``rsync`` process consume JSON-RPC messages meant
    for FastMCP. Commands that do not explicitly need input therefore receive
    ``/dev/null``. Callers that provide ``input_text`` still get an isolated
    pipe, which is required when staging scripts and other small remote files.
    """

    # `subprocess.run(input=...)` creates its own PIPE and rejects a simultaneous
    # `stdin=` argument. Build the two modes explicitly so no-input commands are
    # disconnected from the server's protocol stream while payload commands keep
    # their intentional pipe.
    stdin_kwargs = {"input": input_text} if input_text is not None else {"stdin": subprocess.DEVNULL}
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        **stdin_kwargs,
    )
    return CommandResult(argv=list(argv), returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


class O2Connection:
    """Manage and use the persistent O2 SSH ControlMaster connection.

    ``LOGIN_RETRY_COOLDOWN_SECONDS`` is deliberately process-independent and
    non-configurable. Different Codex tasks must agree on the same retry window;
    a per-process environment override would recreate the coordination gap this
    class exists to close.
    """

    LOGIN_RETRY_COOLDOWN_SECONDS = 300.0

    def __init__(self, config: O2Config | None = None, runner: Runner = default_runner) -> None:
        self.config = config or O2Config()
        self._runner = runner

    # -- safety -----------------------------------------------------------------
    def active_lock_file(self) -> Path | None:
        """Return the safety lock currently blocking O2, if any.

        The configured/user-level lock is authoritative for new installations.
        The working-directory lock preserves the pre-0.2 safety contract during
        migration: an upgrade must never silently bypass an already-engaged
        project emergency stop. Duplicate paths are harmless and are collapsed
        to keep the check and any error message deterministic.
        """

        if self.config.ignore_lock:
            return None
        configured = self.config.lock_file
        legacy = Path.cwd() / ".agent_locks" / "O2_DISABLED"
        for candidate in dict.fromkeys((configured, legacy)):
            if candidate.exists():
                return candidate
        return None

    def is_locked(self) -> bool:
        """Whether the local O2 safety lock is engaged."""
        return self.active_lock_file() is not None

    def _require_unlocked(self) -> None:
        active_lock = self.active_lock_file()
        if active_lock is not None:
            raise O2LockedError(
                f"O2 access is locally disabled by {active_lock}. "
                "Refusing every O2 SSH/rsync command to prevent repeated Duo/MFA prompts. "
                "Remove that file (or set O2_IGNORE_LOCAL_LOCK=1) only after confirming O2 access is safe."
            )

    def _master_start_lock_file(self) -> Path:
        """Return the interprocess mutex used for every new O2 master login.

        The mutex is deliberately shared by the login and transfer aliases.
        Those aliases use different SSH sockets, but both can trigger Duo; one
        workstation must never attempt both authentications concurrently. Its
        path is deliberately independent of ``O2_SSH_LOCK_FILE``: upgraded MCP
        registrations may still name different legacy project locks, but every
        process for this user must converge on one login-coordination boundary.
        """

        return Path.home() / ".agent_locks" / "O2_LOGIN_START.lock"

    def _master_start_attempt_file(self) -> Path:
        """Return the shared receipt that suppresses retries after a failed start."""

        return Path.home() / ".agent_locks" / "O2_LOGIN_START_ATTEMPT.json"

    def _record_master_start_attempt(self, target: str, *, returncode: int | None = None) -> None:
        """Persist a start receipt before SSH so queued callers cannot retry it.

        The file is intentionally written directly and fsynced while the login
        mutex is held. If the process crashes mid-write, the recent file's mtime
        still activates the cooldown; a partially written receipt therefore
        fails safe rather than allowing the next queued process to call SSH.
        """

        receipt_path = self._master_start_attempt_file()
        payload = {
            "started_at": time.time(),
            "pid": os.getpid(),
            "target": target,
            "returncode": returncode,
        }
        try:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            with receipt_path.open("w") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise O2LoginCoordinationError(
                f"Cannot persist the O2 login-attempt receipt at {receipt_path}; "
                f"refusing to start SSH because queued callers could repeat Duo. OS error: {exc}"
            ) from exc

    def _clear_master_start_attempt(self) -> None:
        """Remove the retry-suppression receipt after a confirmed successful start."""

        receipt_path = self._master_start_attempt_file()
        # A live master is authoritative and all callers will reuse it. Keep a
        # receipt that cannot be removed as a conservative guard if the master
        # immediately disappears; never turn a good login into a reported
        # failure merely because local cleanup was unavailable.
        with suppress(OSError):
            receipt_path.unlink(missing_ok=True)

    def _require_login_retry_ready(self) -> None:
        """Refuse a fresh login while a recent process-wide attempt is cooling down.

        The receipt is created *before* ``ssh -MNf``. It therefore covers a
        normal nonzero result, a timeout/exception, or a process crash. Corrupt
        receipts use their filesystem mtime, preserving the safety window even
        when the writer died before completing JSON serialization.
        """

        receipt_path = self._master_start_attempt_file()
        try:
            stat = receipt_path.stat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise O2LoginCoordinationError(
                f"Cannot inspect the O2 login-attempt receipt at {receipt_path}; "
                f"refusing an uncoordinated retry. OS error: {exc}"
            ) from exc

        started_at = stat.st_mtime
        try:
            payload = json.loads(receipt_path.read_text())
            candidate = payload.get("started_at") if isinstance(payload, dict) else None
            # ``bool`` is an ``int`` subclass and JSON also permits non-finite
            # floats in Python's permissive decoder. Neither is a trustworthy
            # timestamp; retain the fail-safe mtime for those values.
            if type(candidate) in (int, float) and math.isfinite(float(candidate)):
                started_at = float(candidate)
        except (OSError, ValueError, TypeError):
            # A recent malformed receipt most likely means its owner crashed
            # during the login attempt. The mtime is the safest fallback.
            pass

        age = max(0.0, time.time() - started_at)
        remaining = self.LOGIN_RETRY_COOLDOWN_SECONDS - age
        if remaining > 0:
            raise O2LoginCoordinationError(
                f"A workstation-wide O2 login attempt occurred {age:.1f}s ago; refusing another Duo-pushing "
                f"login for {remaining:.1f}s. Receipt: {receipt_path}"
            )

        try:
            receipt_path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise O2LoginCoordinationError(
                f"Cannot clear the expired O2 login-attempt receipt at {receipt_path}; "
                f"refusing an uncoordinated retry. OS error: {exc}"
            ) from exc

    @contextmanager
    def _serialized_master_start(self) -> Iterator[None]:
        """Hold a workstation-wide mutex while deciding whether to log in.

        ``master_running()`` followed by ``ssh -MNf`` is otherwise a classic
        check-then-act race: two Codex task processes can both observe no socket
        and each initiate a Duo-pushing login. ``flock`` is released by the OS
        if a process exits, so a crashed MCP server cannot leave a stale lock.

        Coordination failures fail closed. Opening an uncoordinated O2 login is
        more harmful than asking the caller to repair local lock permissions.
        """

        lock_path = self._master_start_lock_file()
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_handle = lock_path.open("a+")
        except OSError as exc:
            raise O2LoginCoordinationError(
                f"Cannot create the O2 login coordination lock at {lock_path}; "
                f"refusing to risk concurrent Duo prompts. OS error: {exc}"
            ) from exc

        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            # The file was opened but never locked. Close it directly rather than
            # running the normal unlock path and potentially masking the useful
            # coordination failure with a second OS error.
            lock_handle.close()
            raise O2LoginCoordinationError(
                f"Cannot acquire the O2 login coordination lock at {lock_path}; "
                f"refusing to risk concurrent Duo prompts. OS error: {exc}"
            ) from exc

        try:
            yield
        finally:
            # Closing the descriptor releases flock even if the guarded SSH call
            # raises. The explicit unlock keeps the lifetime obvious in review.
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock_handle.close()

    # -- ControlMaster lifecycle ------------------------------------------------
    def _master_check_argv(self, alias: str) -> list[str]:
        """Build the exact local-only SSH command used to probe a master socket."""

        return ["ssh", *self.config.base_ssh_opts(), "-O", "check", alias]

    def master_running(self, alias: str | None = None) -> bool:
        """Return whether a reusable ControlMaster socket is alive for ``alias``.

        Defaults to the login host alias. Pass the transfer alias to check the
        transfer node's own master — it is a separate host and a separate control
        socket, so a live login master does not imply a live transfer master.
        """
        if self.is_locked():
            return False
        target = alias or self.config.host_alias
        result = self._runner(
            self._master_check_argv(target),
            self.config.connect_timeout + 5,
            None,
        )
        return result.ok

    def _egress_interface(self, alias: str) -> str | None:
        """Local-only: the network interface a NEW connection to ``alias`` would use.

        Resolves the alias's ``HostName`` via ``ssh -G`` (a config dump — NO connection,
        no Duo) and asks the OS routing table (``route get``). Returns the interface name
        (e.g. ``"utun6"``, ``"en0"``) or ``None`` when it can't be determined (e.g. the
        ``route`` tool is unavailable), so the caller can fail OPEN instead of locking out.
        """
        try:
            cfg = self._runner(["ssh", "-G", alias], self.config.connect_timeout, None)
            host = None
            for line in cfg.stdout.splitlines():
                if line[:9].lower() == "hostname ":
                    host = line.split(None, 1)[1].strip()
                    break
            if not host:
                return None
            route = self._runner(["route", "get", host], self.config.connect_timeout, None)
            if not route.ok:
                return None
            for line in route.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("interface:"):
                    return stripped.split(":", 1)[1].strip()
            return None
        except (OSError, subprocess.TimeoutExpired):
            # Best-effort probe: a missing `route`/`ssh` binary (FileNotFoundError) or a probe
            # timeout must fail OPEN (return None) — matching this method's contract — never
            # propagate and block an otherwise-legitimate login.
            return None

    def _require_on_vpn(self, target: str) -> None:
        """Refuse a new login that would leave via a non-VPN (physical) interface.

        O2 autopushes Duo to non-HMS source IPs, so a login egressing via ``en0``
        instead of the HMS VPN tunnel triggers a phone prompt. Only refuse when the
        interface is KNOWN and is not a VPN tunnel; if it can't be determined, proceed
        (fail open) so an unusual setup is never locked out.
        """
        iface = self._egress_interface(target)
        if iface and not iface.startswith(self.config.vpn_iface_prefix):
            raise O2OffVpnError(
                f"O2 ('{target}') currently routes via '{iface}', not the HMS VPN tunnel "
                f"('{self.config.vpn_iface_prefix}*'). A fresh login from a non-HMS IP triggers a Duo "
                "push. Connect the HMS VPN (GlobalProtect) so `route get` shows a VPN interface, then "
                "retry — or pass allow_offvpn=True (or set O2_REQUIRE_VPN=0) to override and accept the push."
            )

    def start_master(
        self, *, allow_new_login: bool = False, alias: str | None = None, allow_offvpn: bool = False
    ) -> CommandResult:
        """Open the persistent ControlMaster for ``alias`` (default the login host).

        O2 autopushes Duo on every new connection, so opening a master costs one
        approved push; every later command reuses that socket for free (~8h). Refused
        unless ``allow_new_login=True`` so it is always a deliberate, once-per-session
        action — never something a loop can do. Pass ``alias=config.transfer_alias``
        to open the transfer node's own master (a separate host/socket) so a
        transfer-node rsync/ssh has a master to reuse instead of opening a fresh
        Duo-pushing login. Unless ``allow_offvpn=True`` (or ``O2_REQUIRE_VPN=0``), a new
        login is refused with :class:`O2OffVpnError` when the route to ``target`` does not
        egress via a VPN tunnel interface — opening from a non-HMS IP would trigger a Duo push.
        """
        self._require_unlocked()
        target = alias or self.config.host_alias
        if self.master_running(target):
            return CommandResult(self._master_check_argv(target), 0, "master already running", "")
        if not allow_new_login:
            raise O2MasterUnavailableError(
                f"No O2 ControlMaster is running for '{target}' and allow_new_login is False. "
                "O2 autopushes Duo on a new connection; call again with allow_new_login=True to perform "
                "exactly one approved login, then reuse it for the rest of the session."
            )
        with self._serialized_master_start():
            # Another Codex task may have opened the requested master while this
            # process waited for the mutex. Recheck both the emergency stop and
            # the socket *inside* the critical section; only the first contender
            # is then allowed to execute the Duo-pushing command.
            self._require_unlocked()
            if self.master_running(target):
                self._clear_master_start_attempt()
                return CommandResult(self._master_check_argv(target), 0, "master already running", "")
            self._require_login_retry_ready()
            if self.config.require_vpn and not allow_offvpn:
                self._require_on_vpn(target)
            # Persist before invoking SSH. A nonzero result, exception, or
            # process crash leaves the receipt in place so every process that was
            # already queued behind this mutex fails closed during the cooldown.
            self._record_master_start_attempt(target)
            result = self._runner(
                ["ssh", *self.config.base_ssh_opts(), "-MNf", target],
                self.config.connect_timeout + 30,
                None,
            )
            if result.ok:
                self._clear_master_start_attempt()
            else:
                # Best-effort enrichment makes local incident diagnosis easier;
                # the pre-SSH receipt already provides the fail-closed guarantee.
                with suppress(O2LoginCoordinationError):
                    self._record_master_start_attempt(target, returncode=result.returncode)
            return result

    def stop_master(self) -> CommandResult:
        """Close the persistent ControlMaster (non-fatal if already closed)."""
        return self._runner(
            ["ssh", *self.config.base_ssh_opts(), "-O", "exit", self.config.host_alias],
            self.config.connect_timeout + 5,
            None,
        )

    # -- remote execution -------------------------------------------------------
    def run(
        self,
        command: str,
        *,
        timeout: float | None = 120.0,
        require_master: bool = True,
        input_text: str | None = None,
    ) -> CommandResult:
        """Run a remote shell command over the existing ControlMaster.

        By default this refuses unless a master is already running, so accidental
        polling loops fail fast rather than spawning new O2 logins. ``input_text``
        is piped to the remote command's stdin (used to stage scripts remotely).
        """
        self._require_unlocked()
        if require_master and not self.master_running():
            raise O2MasterUnavailableError(
                "No O2 ControlMaster is running. Start one first (start_master with allow_new_login=True, "
                "or the local Terminal/tmux bridge) so commands reuse a single authenticated connection."
            )
        return self._runner(
            ["ssh", *self.config.base_ssh_opts(), self.config.host_alias, command],
            timeout,
            input_text,
        )

    def probe(self) -> CommandResult:
        """Lightweight connectivity check: ``hostname; whoami; date`` on O2."""
        return self.run("hostname; whoami; date", timeout=self.config.connect_timeout + 5)

    def _target_alias_from_argv(self, argv: list[str]) -> str | None:
        """Infer which configured host alias a raw rsync/ssh argv connects to.

        rsync targets appear as ``<alias>:<path>`` and raw ssh as a bare ``<alias>``
        token. The transfer alias is checked first so a transfer-node command is
        never validated against the (different) login master. Returns ``None`` when
        no configured alias appears, leaving the login alias as the default.
        """
        for alias in (self.config.transfer_alias, self.config.host_alias):
            if alias and any(token == alias or token.startswith(f"{alias}:") for token in argv):
                return alias
        return None

    def run_raw(
        self,
        argv: list[str],
        *,
        timeout: float | None = 120.0,
        require_master: bool = True,
        master_alias: str | None = None,
    ) -> CommandResult:
        """Run a local command (e.g. rsync) after the safety-lock + master checks.

        rsync opens its own ssh via ``-e`` and is meant to reuse the existing
        ControlMaster socket from the SSH config. By default this refuses unless a
        master is already running, so a transfer can never silently open a fresh
        connection — which on O2 means an out-of-band Duo push (a brand-new MFA
        login) outside the one approved master. The guard verifies the master for
        the alias the command actually targets: ``master_alias`` if given, else the
        alias inferred from ``argv`` (an ``<alias>:path`` rsync target or a bare
        ``<alias>`` ssh host), else the login alias. So a transfer-node transfer
        (``o2-transfer``) is never validated against the login master even when the
        caller forgets to pass ``master_alias``. Like :meth:`run`, the local lock is
        honored first. Pass ``require_master=False`` only for a transport that
        deliberately tolerates a cold connection.
        """
        self._require_unlocked()
        effective_alias = master_alias if master_alias is not None else self._target_alias_from_argv(argv)
        if require_master and not self.master_running(effective_alias):
            raise O2MasterUnavailableError(
                f"No O2 ControlMaster is running for '{effective_alias or self.config.host_alias}'; refusing a raw "
                "transport (rsync/ssh) that would open a fresh Duo-pushing login. Start one first (start_master "
                "with allow_new_login=True, or the local Terminal/tmux bridge) so transfers reuse the single "
                "authenticated connection."
            )
        return self._runner(list(argv), timeout, None)
