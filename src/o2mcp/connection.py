"""Safe, reusable SSH command channel for HMS O2.

A Python port of ``scripts/o2_ssh_master.sh`` that preserves its safety contract
exactly, but is testable and composable:

- The user-level ``~/.agent_locks/O2_DISABLED`` lock and legacy project lock are
  hard stops on every operation.
- Only ``start_master`` can authenticate. Every other SSH/rsync subprocess has
  all authentication methods disabled and uses an inspected, pinned socket with
  live SSH config disabled, so OpenSSH cannot silently replace a missing
  multiplexed connection with a Duo-triggering standalone login.
- Remote commands run only through an already-established ControlMaster socket;
  opening a NEW login requires an explicit opt-in (one approved MFA verification).

The actual subprocess call is injected (``runner``) so the whole class is unit
tested offline without ever touching the network.
"""

from __future__ import annotations

import fcntl
import glob
import json
import math
import os
import shlex
import subprocess
import tempfile
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


class O2UnsafeTransportError(RuntimeError):
    """Raised when a caller requests a transport that could authenticate anew."""


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

    # Never let PATH lookup or an arbitrary executable whose basename is `ssh`
    # or `rsync` bypass the transport guards. O2 MCP targets macOS and the Linux
    # CI/runtime layout, where the operating-system clients live at these paths.
    # Bare caller spellings are accepted for compatibility but normalized to the
    # absolute binaries before execution; all other paths are rejected.
    SSH_EXECUTABLE = "/usr/bin/ssh"
    RSYNC_EXECUTABLE = "/usr/bin/rsync"

    # Rsync accepts short options in clusters, but these options consume an
    # argument. Once one appears, the remainder of the token (or the following
    # argv element when there is no remainder) is data, not more option letters.
    # This list mirrors rsync's documented short forms: --modify-window (-@),
    # --block-size (-B), --rsh (-e), --filter (-f), --remote-option (-M), and
    # --temp-dir (-T). Treating an ``e`` inside one of those arguments as ``-e``
    # would reject valid transfers such as ``-M--fake-super`` or ``-T/cache``.
    _RSYNC_SHORT_OPTIONS_WITH_ARGUMENTS = frozenset({"@", "B", "e", "f", "M", "T"})

    # Long options can likewise take their argument from the next argv element.
    # Keep the argument-taking names explicit so a value such as the ``-e`` in
    # ``--exclude -e`` is not reinterpreted as an SSH transport. This set follows
    # rsync's client option table (all string/integer arguments, including
    # compatibility aliases); options with an attached ``=value`` do not consume
    # the next element and are handled naturally by the parser below.
    _RSYNC_LONG_OPTIONS_WITH_ARGUMENTS = frozenset(
        {
            "address",
            "backup-dir",
            "block-size",
            "bwlimit",
            "cc",
            "checksum-choice",
            "checksum-seed",
            "chmod",
            "chown",
            "compare-dest",
            "compress-choice",
            "compress-level",
            "compress-threads",
            "config",
            "contimeout",
            "copy-as",
            "copy-dest",
            "debug",
            "dparam",
            "early-input",
            "exclude",
            "exclude-from",
            "files-from",
            "filter",
            "groupmap",
            "iconv",
            "include",
            "include-from",
            "info",
            "link-dest",
            "log-file",
            "log-file-format",
            "log-format",
            "max-alloc",
            "max-delete",
            "max-size",
            "min-size",
            "modify-window",
            "only-write-batch",
            "out-format",
            "outbuf",
            "partial-dir",
            "password-file",
            "port",
            "protocol",
            "read-batch",
            "remote-option",
            "rsh",
            "rsync-path",
            "skip-compress",
            "sockopts",
            "stderr",
            "stop-after",
            "stop-at",
            "suffix",
            "temp-dir",
            "time-limit",
            "timeout",
            "usermap",
            "write-batch",
            "zc",
            "zl",
            "zt",
        }
    )

    # OpenSSH short options that consume either the rest of their token or the
    # following argv element. The sanitizer uses this to keep scanning past
    # benign options such as ``-p 22`` instead of mistaking their argument for
    # the destination and leaving a later ``-S`` override untouched.
    _SSH_SHORT_OPTIONS_WITH_ARGUMENTS = frozenset(
        {"B", "D", "E", "F", "I", "J", "L", "O", "P", "Q", "R", "S", "W", "b", "c", "e", "i", "l", "m", "o", "p", "w"}
    )

    def __init__(self, config: O2Config | None = None, runner: Runner = default_runner) -> None:
        self.config = config or O2Config()
        self._runner = runner
        # A connection may build an rsync argv and then validate/run it. Cache the
        # safely expanded config for each target so those two steps cannot drift
        # and do not repeatedly parse the same local files.
        self._resolved_ssh_configs: dict[str, dict[str, str]] = {}
        self._safe_ssh_config_text_cache: str | None = None

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
        """Build a config-isolated command that probes one exact master socket.

        ``-F /dev/null`` prevents a later probe from evaluating user config (and
        especially ``Match exec``) after the socket has already been resolved by
        the safe parser. The explicit ``-S`` is sufficient for a control command;
        no network connection or authentication is attempted.
        """

        return [
            self.SSH_EXECUTABLE,
            "-F",
            "/dev/null",
            "-S",
            self._resolved_control_path(alias),
            *self.config.base_ssh_opts(),
            "-O",
            "check",
            alias,
        ]

    def master_running(self, alias: str | None = None) -> bool:
        """Return whether a reusable ControlMaster socket is alive for ``alias``.

        Defaults to the login host alias. Pass the transfer alias to check the
        transfer node's own master — it is a separate host and a separate control
        socket, so a live login master does not imply a live transfer master. A
        local config/probe timeout or missing SSH executable is treated as
        unavailable: this boolean guard must fail closed rather than crash its
        callers or allow them to infer that reuse is safe.
        """
        if self.is_locked():
            return False
        target = alias or self.config.host_alias
        try:
            result = self._runner(
                self._master_check_argv(target),
                self.config.connect_timeout + 5,
                None,
            )
        except (OSError, subprocess.TimeoutExpired):
            # Both config expansion and `ssh -O check` are local-only probes.
            # Either failure means the expected socket was not proven reusable.
            return False
        return result.ok

    def _egress_interface(self, alias: str) -> str | None:
        """Local-only: the network interface a NEW connection to ``alias`` would use.

        Resolves the alias's ``HostName`` from the safely flattened SSH config and
        asks the OS routing table (``route get``). Returns the interface name (e.g.
        ``"utun6"``, ``"en0"``) or ``None`` when it can't be determined (e.g. the
        ``route`` tool is unavailable), so the caller can fail OPEN instead of
        locking out. Unsafe SSH config is not an indeterminate route and therefore
        propagates as a hard failure.
        """
        try:
            host = self._resolved_ssh_config(alias).get("hostname")
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
            # Run the one authentication-capable operation against a flattened
            # copy of the user's SSH config that has been inspected for Match
            # directives. The exact resolved socket is pinned explicitly so the
            # post-start check and all reuse-only clients address the same master.
            with self._safe_ssh_config_path() as safe_config:
                result = self._runner(
                    [
                        self.SSH_EXECUTABLE,
                        "-F",
                        safe_config,
                        "-S",
                        self._resolved_control_path(target),
                        *self.config.base_ssh_opts(),
                        "-MNf",
                        target,
                    ],
                    self.config.connect_timeout + 30,
                    None,
                )
            if result.ok:
                # ``ssh -f`` can return zero after authentication and then lose
                # its backgrounded connection immediately. Treating the parent
                # process's exit code as the whole startup contract created a
                # dangerous false positive: callers believed they had a reusable
                # master, the retry receipt was removed, and a later command could
                # initiate another user-approved login attempt. Verify the actual
                # control socket before reporting success or clearing the receipt.
                try:
                    verification = self._runner(
                        self._master_check_argv(target),
                        self.config.connect_timeout + 5,
                        None,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    # The pre-start receipt deliberately remains in place. A
                    # local verification failure is ambiguous, so the safest
                    # behavior is to suppress automatic retries for the normal
                    # cooldown window rather than risk another Duo prompt.
                    with suppress(O2LoginCoordinationError):
                        self._record_master_start_attempt(target, returncode=255)
                    detail = f"{type(exc).__name__}: {exc}"
                    return CommandResult(
                        # Report the step that failed. Keeping the verification
                        # command paired with its synthetic SSH failure code
                        # makes CommandResult internally consistent while the
                        # diagnostic below preserves output from the start step.
                        argv=self._master_check_argv(target),
                        returncode=255,
                        # No verification subprocess completed, so there is no
                        # stdout belonging to the reported command outcome.
                        stdout="",
                        stderr=self._master_verification_error(target, detail, result.stderr),
                    )

                if verification.ok:
                    self._clear_master_start_attempt()
                else:
                    # Record the failed control-socket check rather than the
                    # misleading zero returned by ``ssh -MNf``. This receipt is
                    # the cross-process evidence that a fresh login was already
                    # attempted and must not be retried immediately.
                    with suppress(O2LoginCoordinationError):
                        self._record_master_start_attempt(target, returncode=verification.returncode)
                    detail = verification.stderr.strip() or verification.stdout.strip() or "no SSH diagnostics"
                    return CommandResult(
                        argv=verification.argv,
                        returncode=verification.returncode or 255,
                        stdout=verification.stdout,
                        stderr=self._master_verification_error(target, detail, result.stderr),
                    )
            else:
                # Best-effort enrichment makes local incident diagnosis easier;
                # the pre-SSH receipt already provides the fail-closed guarantee.
                with suppress(O2LoginCoordinationError):
                    self._record_master_start_attempt(target, returncode=result.returncode)
            return result

    def _master_verification_error(self, target: str, detail: str, start_stderr: str) -> str:
        """Describe a zero-exit SSH start whose reusable socket did not survive.

        Preserve any diagnostic text emitted by the original ``ssh -MNf`` call,
        because HMS login banners or disconnect messages can explain why the
        backgrounded master vanished. The message also makes the retained retry
        receipt explicit so an operator knows not to launch another login loop.
        """

        message = (
            f"SSH reported success starting the O2 ControlMaster for '{target}', but the post-start "
            f"control-socket check failed: {detail}. The login-attempt receipt remains active; do not "
            "retry automatically."
        )
        prior = start_stderr.strip()
        return f"{prior}\n{message}" if prior else message

    def stop_master(self) -> CommandResult:
        """Close the persistent ControlMaster (non-fatal if already closed)."""
        target = self.config.host_alias
        return self._runner(
            [
                self.SSH_EXECUTABLE,
                "-F",
                "/dev/null",
                "-S",
                self._resolved_control_path(target),
                *self.config.base_ssh_opts(),
                "-O",
                "exit",
                target,
            ],
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
        alias: str | None = None,
    ) -> CommandResult:
        """Run a remote shell command over an existing ControlMaster only.

        ``alias`` defaults to the login host and may name the separately approved
        transfer master. The historical ``require_master`` parameter remains for
        source compatibility, but passing ``False`` is now rejected: a cold SSH
        connection is never a safe fallback for an O2 MCP operation.

        The socket check is intentionally paired with authentication-disabled SSH
        options. A check alone has a race: the master can die before this second
        subprocess starts, and OpenSSH would normally authenticate a new standalone
        connection. With every authentication method disabled, that race fails
        closed without generating a Duo request. ``input_text`` is piped to the
        remote command's stdin when staging scripts or other small payloads.
        """
        self._require_unlocked()
        if not require_master:
            raise O2UnsafeTransportError(
                "Cold O2 SSH execution is disabled. Start one explicitly authorized ControlMaster, then retry."
            )
        target = alias or self.config.host_alias
        if not self.master_running(target):
            raise O2MasterUnavailableError(
                f"No O2 ControlMaster is running for '{target}'. Start one explicitly, then retry; ordinary "
                "commands cannot fall back to a new Duo-triggering SSH connection."
            )
        return self._runner(
            [*self._reuse_only_ssh_prefix(target), target, command],
            timeout,
            input_text,
        )

    def probe(self) -> CommandResult:
        """Lightweight connectivity check: ``hostname; whoami; date`` on O2."""
        return self.run("hostname; whoami; date", timeout=self.config.connect_timeout + 5)

    def _ssh_destination_from_argv(self, argv: list[str]) -> str | None:
        """Return the destination operand from one raw SSH command.

        Only the first non-option operand is the SSH destination; everything
        after it is the remote command and must not influence which ControlMaster
        socket is selected. Argument-taking short options are skipped using the
        same option table as the transport sanitizer, including attached values
        such as ``-p22`` and separate values such as ``-p 22``.
        """

        index = 1
        while index < len(argv):
            token = argv[index]
            if token == "--":
                return argv[index + 1] if index + 1 < len(argv) else None
            if token == "-" or not token.startswith("-"):
                return token
            if token.startswith("--"):
                # OpenSSH has no long client options. Leave validation of an
                # invalid spelling to SSH, but do not treat it as a destination.
                index += 1
                continue

            cluster = token[1:]
            for option_index, option in enumerate(cluster):
                if option not in self._SSH_SHORT_OPTIONS_WITH_ARGUMENTS:
                    continue
                # If no value is attached to this short option, its following
                # argv element is option data rather than the destination.
                if not cluster[option_index + 1 :]:
                    index += 1
                break
            index += 1
        return None

    def _rsync_operands_from_argv(self, argv: list[str]) -> list[str]:
        """Return rsync path operands while excluding option arguments.

        Alias-like text in an option value (for example an exclude pattern)
        cannot identify the remote endpoint. This parser mirrors the hardening
        parser's treatment of argument-taking short and long options so target
        inference considers only actual source/destination operands.
        """

        operands: list[str] = []
        index = 1
        options_finished = False
        while index < len(argv):
            token = argv[index]
            if options_finished:
                operands.append(token)
                index += 1
                continue
            if token == "--":
                options_finished = True
                index += 1
                continue
            if token == "-" or not token.startswith("-"):
                operands.append(token)
                index += 1
                continue
            if token.startswith("--"):
                option_name, equals, _attached_argument = token[2:].partition("=")
                if not equals and option_name in self._RSYNC_LONG_OPTIONS_WITH_ARGUMENTS:
                    index += 1
                index += 1
                continue

            cluster = token[1:]
            for option_index, option in enumerate(cluster):
                if option not in self._RSYNC_SHORT_OPTIONS_WITH_ARGUMENTS:
                    continue
                if not cluster[option_index + 1 :]:
                    index += 1
                break
            index += 1
        return operands

    def _target_alias_from_argv(self, argv: list[str]) -> str | None:
        """Infer the configured O2 endpoint used by a raw rsync/SSH argv.

        Rsync endpoints appear as ``[user@]<alias>:<path>`` while raw SSH has one
        destination operand. Parse each command's option structure first so an
        alias mentioned in an option value or remote command cannot select the
        wrong socket. Strip the optional user qualifier only for host comparison,
        then return the full ``[user@]alias`` endpoint because ``%r``-based
        ControlPath templates resolve different sockets for different users.
        """

        if not argv:
            return None
        executable = argv[0]
        if executable in {"ssh", self.SSH_EXECUTABLE}:
            destination = self._ssh_destination_from_argv(argv)
            candidates = [destination] if destination is not None else []
        elif executable in {"rsync", self.RSYNC_EXECUTABLE}:
            # A colon distinguishes remote-shell/daemon operands from ordinary
            # local paths, which may coincidentally equal an O2 alias.
            candidates = [operand for operand in self._rsync_operands_from_argv(argv) if ":" in operand]
        else:
            candidates = []

        for alias in (self.config.transfer_alias, self.config.host_alias):
            if not alias:
                continue
            for token in candidates:
                endpoint = token.split(":", 1)[0]
                host = endpoint.rsplit("@", 1)[-1]
                if host == alias:
                    return endpoint
        return None

    def _flatten_ssh_config(self, path: Path, *, stack: tuple[Path, ...] = ()) -> str:
        """Read an SSH config and inline ``Include`` files without executing it.

        OpenSSH evaluates ``Match exec`` predicates even for ``ssh -G``. A
        predicate can run an arbitrary shell command, including another SSH
        process that authenticates and triggers Duo. This reader therefore
        inspects the literal config graph first and rejects every ``Match`` block
        before OpenSSH is allowed to expand host tokens.

        Includes are flattened into a temporary config so the subsequently
        executed ``ssh -G -F <temporary>`` cannot discover an uninspected file.
        OpenSSH resolves relative paths in user-config ``Include`` directives
        beneath ``~/.ssh`` (even when ``-F`` names another file), so this reader
        does the same; absolute paths and ``~`` are also supported. Tokenized
        include paths (for example ``%d``) are rejected because reproducing
        OpenSSH's expansion incorrectly would risk selecting a different socket.
        """

        expanded_path = path.expanduser().resolve()
        if expanded_path in stack:
            chain = " -> ".join(str(item) for item in (*stack, expanded_path))
            raise O2UnsafeTransportError(f"SSH config Include cycle while resolving O2: {chain}")
        try:
            lines = expanded_path.read_text().splitlines(keepends=True)
        except OSError as exc:
            raise O2UnsafeTransportError(f"Cannot read O2 SSH config {expanded_path}: {exc}") from exc

        flattened: list[str] = []
        next_stack = (*stack, expanded_path)
        for line_number, raw_line in enumerate(lines, start=1):
            try:
                fields = shlex.split(raw_line, comments=True, posix=True)
            except ValueError as exc:
                raise O2UnsafeTransportError(
                    f"Cannot safely parse SSH config {expanded_path}:{line_number}: {exc}"
                ) from exc
            if not fields:
                flattened.append(raw_line)
                continue

            # OpenSSH accepts both ``Keyword value`` and ``Keyword=value`` (with
            # optional whitespace around ``=``). Normalize those spellings before
            # classifying the directive; otherwise ``Match=exec`` would evade the
            # literal safety scan and be executed by the later ``ssh -G`` call.
            keyword, attached_separator, attached_argument = fields[0].partition("=")
            directive = keyword.lower()
            arguments = list(fields[1:])
            if attached_separator:
                if attached_argument:
                    arguments.insert(0, attached_argument)
            elif arguments and arguments[0].startswith("="):
                equals_argument = arguments.pop(0)[1:]
                if equals_argument:
                    arguments.insert(0, equals_argument)

            if directive == "match":
                raise O2UnsafeTransportError(
                    f"SSH config {expanded_path}:{line_number} contains Match. "
                    "O2 refuses configs whose Match predicates could execute a fresh SSH/Duo probe; "
                    "move the O2 alias to a Match-free config selected by O2_SSH_CONFIG_FILE."
                )
            if directive != "include":
                flattened.append(raw_line)
                continue
            if not arguments:
                raise O2UnsafeTransportError(f"SSH config {expanded_path}:{line_number} has an empty Include.")

            for pattern_text in arguments:
                if "%" in pattern_text:
                    raise O2UnsafeTransportError(
                        f"SSH config {expanded_path}:{line_number} uses a tokenized Include path; "
                        "set O2_SSH_CONFIG_FILE to a Match-free, explicit O2 config instead."
                    )
                pattern = Path(pattern_text).expanduser()
                if not pattern.is_absolute():
                    pattern = Path.home() / ".ssh" / pattern
                # OpenSSH silently ignores Include globs with no matches. Sort
                # matches so the flattened file preserves deterministic option
                # precedence across filesystems.
                for included_name in sorted(glob.glob(str(pattern))):
                    flattened.append(self._flatten_ssh_config(Path(included_name), stack=next_stack))

        text = "".join(flattened)
        # Prevent an included file whose final line lacks a newline from being
        # concatenated with the caller's next directive in the flattened copy.
        return text if not text or text.endswith("\n") else text + "\n"

    def _safe_ssh_config_text(self) -> str:
        """Return one inspected, Match-free SSH configuration snapshot."""

        if self._safe_ssh_config_text_cache is None:
            self._safe_ssh_config_text_cache = self._flatten_ssh_config(self.config.ssh_config_file)
        return self._safe_ssh_config_text_cache

    @contextmanager
    def _safe_ssh_config_path(self) -> Iterator[str]:
        """Yield a private temporary path containing the inspected SSH config."""

        # NamedTemporaryFile uses mode 0600 by default. Keep it open while SSH
        # reads it; this is safe on the supported Unix/macOS deployment and
        # ensures the file is removed even when the runner raises or times out.
        with tempfile.NamedTemporaryFile(mode="w", prefix="o2-mcp-ssh-", suffix=".config") as handle:
            handle.write(self._safe_ssh_config_text())
            handle.flush()
            yield handle.name

    def _resolved_ssh_config(self, target: str) -> dict[str, str]:
        """Expand one target through an inspected config without ``Match exec``."""

        # Even local config expansion is forbidden under the incident lock. This
        # protects argv-only builders such as async rsync, which resolve their
        # socket before the eventual subprocess launch.
        self._require_unlocked()
        cached = self._resolved_ssh_configs.get(target)
        if cached is not None:
            return cached

        with self._safe_ssh_config_path() as safe_config:
            result = self._runner(
                [self.SSH_EXECUTABLE, "-G", "-F", safe_config, target],
                self.config.connect_timeout,
                None,
            )
        if not result.ok:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
            raise O2UnsafeTransportError(f"Could not safely resolve SSH config for '{target}': {detail}")

        resolved: dict[str, str] = {}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition(" ")
            if separator:
                resolved.setdefault(key.lower(), value.strip())
        self._resolved_ssh_configs[target] = resolved
        return resolved

    def _resolved_control_path(self, target: str) -> str:
        """Return the target's original, safely expanded ControlPath.

        Reuse-only commands disable ProxyJump/ProxyCommand so those helpers cannot
        start independently authenticating SSH subprocesses. ``%C`` ControlPath
        templates include the jump-host identity, however, so letting a guarded
        command recompute the path after disabling its proxy would select a
        different socket. Expand the inspected original config first, then pin
        that exact path with ``-S`` while all later SSH invocations use
        ``-F /dev/null``.
        """

        path = self._resolved_ssh_config(target).get("controlpath")
        if path and path.lower() != "none":
            return path
        raise O2UnsafeTransportError(
            f"SSH target '{target}' has no ControlPath; refusing an authentication-capable fallback."
        )

    def _reuse_only_ssh_prefix(self, alias: str) -> list[str]:
        """Return ``ssh`` plus a pinned socket and fail-closed client options."""

        return [
            self.SSH_EXECUTABLE,
            "-F",
            "/dev/null",
            "-S",
            self._resolved_control_path(alias),
            *self.config.reuse_only_ssh_opts(),
        ]

    def reuse_only_ssh_transport(self, alias: str) -> str:
        """Return the shell-form SSH transport used by rsync ``-e``."""

        return shlex.join(self._reuse_only_ssh_prefix(alias))

    @staticmethod
    def _ssh_o_option_name(option: str) -> str:
        """Return the case-normalized keyword from one ``ssh -o`` argument."""

        normalized = option.lstrip("=").strip()
        return normalized.replace("=", " ", 1).split(None, 1)[0].lower() if normalized else ""

    def _sanitize_caller_ssh_options(self, tokens: list[str], *, stop_at_host: bool) -> list[str]:
        """Remove socket/config overrides and reject endpoint identity changes.

        The guarded prefix owns ``-F`` and ``-S``. OpenSSH gives a later ``-S``
        precedence, so merely prepending the pinned socket is insufficient; every
        caller-supplied config-file, ControlPath, and socket override is removed.
        User, port, and hostname overrides are rejected because changing those
        values after socket resolution can make ``%r``/``%p``/``%h`` ControlPath
        templates stale or silently route the command through a different pinned
        master. Callers can express the user safely as ``user@alias``, which is
        retained during target inference and socket resolution; host and port
        remain owned by the inspected alias configuration.

        For a direct SSH argv, option parsing stops at the destination so a remote
        command argument named ``-S`` remains ordinary data. Rsync's ``-e`` value
        contains only the SSH executable and its options, so its entire token list
        is inspected.
        """

        sanitized: list[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "--":
                sanitized.extend(tokens[index:])
                break
            if token == "-" or not token.startswith("-"):
                if stop_at_host:
                    sanitized.extend(tokens[index:])
                    break
                sanitized.append(token)
                index += 1
                continue
            if token.startswith("--"):
                # SSH has no long client options. Preserve the token so OpenSSH
                # can report the invalid input rather than guessing its meaning.
                sanitized.append(token)
                index += 1
                continue

            cluster = token[1:]
            kept_flags: list[str] = []
            handled_argument_option = False
            for option_index, option in enumerate(cluster):
                if option not in self._SSH_SHORT_OPTIONS_WITH_ARGUMENTS:
                    kept_flags.append(option)
                    continue

                handled_argument_option = True
                attached_argument = cluster[option_index + 1 :]
                if attached_argument:
                    argument = attached_argument
                    consumes_next = False
                else:
                    if index + 1 >= len(tokens):
                        raise O2UnsafeTransportError(f"SSH option -{option} requires an argument.")
                    argument = tokens[index + 1]
                    consumes_next = True

                if option in {"F", "S"}:
                    # Preserve any preceding flag-only cluster (e.g. the ``v``
                    # in ``-vS/path``), but discard the unsafe option and value.
                    if kept_flags:
                        sanitized.append("-" + "".join(kept_flags))
                elif option in {"l", "p"}:
                    identity = "user" if option == "l" else "port"
                    raise O2UnsafeTransportError(
                        f"SSH {identity} options are disabled for guarded O2 transports; "
                        "use user@o2 or user@o2-transfer for user selection and keep host/port in the inspected alias."
                    )
                elif option == "o" and self._ssh_o_option_name(argument) in {
                    "controlpath",
                    "hostname",
                    "port",
                    "user",
                }:
                    option_name = self._ssh_o_option_name(argument)
                    if option_name != "controlpath":
                        raise O2UnsafeTransportError(
                            f"SSH {option_name} options are disabled for guarded O2 transports; "
                            "use user@o2 or user@o2-transfer for user selection and keep host/port in the "
                            "inspected alias."
                        )
                    if kept_flags:
                        sanitized.append("-" + "".join(kept_flags))
                else:
                    # This argument-taking option is unrelated to socket/user
                    # identity. Preserve its exact representation and value.
                    sanitized.append(token)
                    if consumes_next:
                        sanitized.append(argument)

                if consumes_next:
                    index += 1
                break

            if not handled_argument_option:
                sanitized.append(token)
            index += 1

        return sanitized

    def _reuse_only_ssh_command(self, command: str, alias: str) -> str:
        """Harden an rsync ``-e`` transport so it cannot authenticate anew.

        ``rsync`` receives its remote-shell transport as one shell-like string.
        We parse that string, require SSH (rather than an arbitrary executable),
        and prepend the reuse-only options before caller-provided options. OpenSSH
        keeps the first value supplied for an option, so an unsafe later override
        cannot re-enable authentication.
        """
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            raise O2UnsafeTransportError(f"Invalid rsync SSH transport: {exc}") from exc
        if not tokens or tokens[0] not in {"ssh", self.SSH_EXECUTABLE}:
            raise O2UnsafeTransportError(
                f"O2 rsync transports must use the trusted {self.SSH_EXECUTABLE} client through the configured "
                "ControlMaster; arbitrary SSH executable paths are disabled."
            )
        safe = self._reuse_only_ssh_prefix(alias)[1:]
        caller_tokens = tokens[1:]
        if caller_tokens[: len(safe)] == safe:
            caller_tokens = caller_tokens[len(safe) :]
        caller_tokens = self._sanitize_caller_ssh_options(caller_tokens, stop_at_host=False)
        # Normalize a caller's bare `ssh` spelling to the absolute system binary
        # so subprocess/rsync cannot resolve an attacker-controlled PATH entry.
        return shlex.join([self.SSH_EXECUTABLE, *safe, *caller_tokens])

    def _harden_raw_transport_argv(self, argv: list[str], alias: str) -> list[str]:
        """Return an SSH/rsync argv whose fallback path cannot authenticate.

        ``run_raw`` is intentionally limited to the two transport executables the
        package owns. Direct SSH gets the guard options prepended. Rsync receives
        the same options through its ``-e``/``--rsh`` transport, including when a
        library caller supplied an incomplete or permissive transport string.
        """
        if not argv:
            raise O2UnsafeTransportError("Refusing an empty raw O2 transport command.")

        hardened = list(argv)
        executable = hardened[0]
        if executable in {"ssh", self.SSH_EXECUTABLE}:
            safe = self._reuse_only_ssh_prefix(alias)[1:]
            caller_tokens = hardened[1:]
            if caller_tokens[: len(safe)] == safe:
                caller_tokens = caller_tokens[len(safe) :]
            caller_tokens = self._sanitize_caller_ssh_options(caller_tokens, stop_at_host=True)
            return [self.SSH_EXECUTABLE, *safe, *caller_tokens]
        if executable not in {"rsync", self.RSYNC_EXECUTABLE}:
            raise O2UnsafeTransportError(
                f"run_raw accepts only the trusted {self.SSH_EXECUTABLE} or {self.RSYNC_EXECUTABLE} O2 "
                "transports; arbitrary executable paths are disabled."
            )
        # As with SSH, pin rsync itself before it interprets the guarded `-e`
        # transport. A caller-supplied wrapper named rsync could otherwise ignore
        # that transport and open its own authentication-capable connection.
        hardened[0] = self.RSYNC_EXECUTABLE

        # An explicit rsync remote shell overrides RSYNC_RSH and the user's
        # environment. Normalize every supported spelling so detached and
        # synchronous transfers share the same fail-closed transport contract.
        found_transport = False
        index = 1
        while index < len(hardened):
            token = hardened[index]
            if token == "--":
                # Everything after rsync's option terminator is a path, even if
                # it begins with ``-e``; never rewrite user data as transport
                # configuration.
                break
            if token in {"-e", "--rsh"}:
                if index + 1 >= len(hardened):
                    raise O2UnsafeTransportError(f"{token} requires an SSH transport argument.")
                hardened[index + 1] = self._reuse_only_ssh_command(hardened[index + 1], alias)
                found_transport = True
                index += 2
                continue
            if token.startswith("--rsh="):
                transport = token.split("=", 1)[1]
                hardened[index] = "--rsh=" + self._reuse_only_ssh_command(transport, alias)
                found_transport = True
            elif token.startswith("--"):
                # A separate value for an argument-taking long option is data,
                # even when that value is literally ``-e`` or ``--rsh``. Skip it
                # now so the next loop iteration cannot mistake it for transport
                # configuration. ``--name=value`` already carries its argument
                # in this token and therefore does not consume the next element.
                option_name, equals, _attached_argument = token[2:].partition("=")
                if not equals and option_name in self._RSYNC_LONG_OPTIONS_WITH_ARGUMENTS:
                    if index + 1 >= len(hardened):
                        raise O2UnsafeTransportError(f"{token} requires an argument.")
                    index += 1
            elif token.startswith("-") and not token.startswith("--"):
                # Rsync permits clustered short options, but options that take
                # arguments terminate the cluster: their attached remainder (or
                # the next argv element) is data. Walk the cluster in order so an
                # ``e`` inside ``-M--fake-super`` or ``-T/tmp/cache`` is not
                # misidentified as the remote-shell option.
                cluster = token[1:]
                for option_index, option in enumerate(cluster):
                    if option not in self._RSYNC_SHORT_OPTIONS_WITH_ARGUMENTS:
                        continue

                    attached_argument = cluster[option_index + 1 :]
                    if option == "e":
                        # A clustered ``-e`` consumes either the remainder of
                        # this token (``-avzessh`` / ``-avze=ssh``) or the next
                        # token (``-avze ssh``). Normalize either representation;
                        # merely inserting an earlier safe ``-e`` would not help
                        # because rsync honors the later transport override.
                        prefix = token[: option_index + 2]
                        equals = "=" if attached_argument.startswith("=") else ""
                        transport = attached_argument[len(equals) :]
                        if transport:
                            hardened[index] = prefix + equals + self._reuse_only_ssh_command(transport, alias)
                        else:
                            if index + 1 >= len(hardened):
                                raise O2UnsafeTransportError(f"{token} requires an SSH transport argument.")
                            hardened[index + 1] = self._reuse_only_ssh_command(hardened[index + 1], alias)
                            index += 1
                        found_transport = True
                    elif not attached_argument:
                        # The next argv element belongs to this non-transport
                        # option. Skip it even when it resembles ``-e`` so it is
                        # never reinterpreted as a second rsync option.
                        if index + 1 >= len(hardened):
                            raise O2UnsafeTransportError(f"{token} requires an argument.")
                        index += 1

                    # Any argument-taking option ends short-option parsing for
                    # this token because every remaining character is its data.
                    break
            index += 1

        if not found_transport:
            hardened[1:1] = ["-e", self._reuse_only_ssh_command("ssh", alias)]
        return hardened

    def run_raw(
        self,
        argv: list[str],
        *,
        timeout: float | None = 120.0,
        require_master: bool = True,
        master_alias: str | None = None,
    ) -> CommandResult:
        """Run a fail-closed SSH/rsync transport after lock and master checks.

        rsync opens its own ssh via ``-e`` and is meant to reuse the existing
        ControlMaster socket from the SSH config. By default this refuses unless a
        master is already running. The transport is also rewritten with every SSH
        authentication method disabled; this closes the check/use race where a
        dying socket could otherwise make OpenSSH fall back to a brand-new MFA
        login. The guard verifies the master for
        the alias the command actually targets: ``master_alias`` if given, else the
        endpoint inferred from ``argv`` (a ``[user@]<alias>:path`` rsync target or
        a bare ``[user@]<alias>`` SSH host), else the login alias. So a transfer-node
        transfer (``o2-transfer``) is never validated against the login master even
        when the caller forgets to pass ``master_alias``. Like :meth:`run`, the
        local lock is honored first. ``require_master=False`` is retained only to
        give existing callers an actionable error; cold transports are no longer
        supported.
        """
        self._require_unlocked()
        if not require_master:
            raise O2UnsafeTransportError(
                "Cold O2 SSH/rsync execution is disabled. Start one explicitly authorized ControlMaster, then retry."
            )
        effective_target = master_alias if master_alias is not None else self._target_alias_from_argv(argv)
        target = effective_target or self.config.host_alias
        hardened = self._harden_raw_transport_argv(argv, target)
        if not self.master_running(effective_target):
            raise O2MasterUnavailableError(
                f"No O2 ControlMaster is running for '{effective_target or self.config.host_alias}'; refusing a raw "
                "transport (rsync/ssh) that would open a fresh Duo-pushing login. Start one first (start_master "
                "with allow_new_login=True, or the local Terminal/tmux bridge) so transfers reuse the single "
                "authenticated connection."
            )
        return self._runner(hardened, timeout, None)
