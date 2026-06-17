"""Safe, reusable SSH command channel for HMS O2.

A Python port of ``scripts/o2_ssh_master.sh`` that preserves its safety contract
exactly, but is testable and composable:

- The ``.agent_locks/O2_DISABLED`` lock is a hard stop on every operation.
- All SSH uses BatchMode (public key only) — a dead master or missing key fails
  fast instead of triggering a Duo/MFA phone prompt.
- Remote commands run only through an already-established ControlMaster socket;
  opening a NEW login requires an explicit opt-in (one approved MFA verification).

The actual subprocess call is injected (``runner``) so the whole class is unit
tested offline without ever touching the network.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

from o2mcp.config import O2Config


class O2LockedError(RuntimeError):
    """Raised when the local O2 safety lock forbids any cluster operation."""


class O2MasterUnavailableError(RuntimeError):
    """Raised when a command needs the ControlMaster but none is running."""


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
    """Run a command via subprocess, capturing output (the real I/O seam)."""
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input_text,
        check=False,
    )
    return CommandResult(argv=list(argv), returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


class O2Connection:
    """Manage and use the persistent O2 SSH ControlMaster connection."""

    def __init__(self, config: O2Config | None = None, runner: Runner = default_runner) -> None:
        self.config = config or O2Config()
        self._runner = runner

    # -- safety -----------------------------------------------------------------
    def is_locked(self) -> bool:
        """Whether the local O2 safety lock is engaged."""
        return self.config.lock_file.exists() and not self.config.ignore_lock

    def _require_unlocked(self) -> None:
        if self.is_locked():
            raise O2LockedError(
                f"O2 access is locally disabled by {self.config.lock_file}. "
                "Refusing every O2 SSH/rsync command to prevent repeated Duo/MFA prompts. "
                "Remove that file (or set O2_IGNORE_LOCAL_LOCK=1) only after confirming O2 access is safe."
            )

    # -- ControlMaster lifecycle ------------------------------------------------
    def master_running(self, alias: str | None = None) -> bool:
        """Return whether a reusable ControlMaster socket is alive for ``alias``.

        Defaults to the login host alias. Pass the transfer alias to check the
        transfer node's own master — it is a separate host and a separate control
        socket, so a live login master does not imply a live transfer master.
        """
        if self.is_locked():
            return False
        result = self._runner(
            ["ssh", *self.config.base_ssh_opts(), "-O", "check", alias or self.config.host_alias],
            self.config.connect_timeout + 5,
            None,
        )
        return result.ok

    def start_master(self, *, allow_new_login: bool = False, alias: str | None = None) -> CommandResult:
        """Open the persistent ControlMaster for ``alias`` (default the login host).

        O2 autopushes Duo on every new connection, so opening a master costs one
        approved push; every later command reuses that socket for free (~8h). Refused
        unless ``allow_new_login=True`` so it is always a deliberate, once-per-session
        action — never something a loop can do. Pass ``alias=config.transfer_alias``
        to open the transfer node's own master (a separate host/socket) so a
        transfer-node rsync/ssh has a master to reuse instead of opening a fresh
        Duo-pushing login.
        """
        self._require_unlocked()
        target = alias or self.config.host_alias
        if self.master_running(target):
            return CommandResult(["ssh", "-O", "check", target], 0, "master already running", "")
        if not allow_new_login:
            raise O2MasterUnavailableError(
                f"No O2 ControlMaster is running for '{target}' and allow_new_login is False. "
                "O2 autopushes Duo on a new connection; call again with allow_new_login=True to perform "
                "exactly one approved login, then reuse it for the rest of the session."
            )
        return self._runner(
            ["ssh", *self.config.base_ssh_opts(), "-MNf", target],
            self.config.connect_timeout + 30,
            None,
        )

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
