"""Safe, reusable SSH command channel for HMS O2.

A Python port of ``scripts/o2_ssh_master.sh`` that preserves its safety contract
exactly, but is testable and composable:

- The workstation-wide ``O2_POLICY.json`` state governs every remote operation.
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

import glob
import shlex
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from o2mcp.config import O2Config
from o2mcp.policy import (
    LoginTarget,
    O2LoginGrantError,
    O2PolicyDeniedError,
    O2PolicyStore,
)

# Compatibility alias for downstream clients during the 0.3 rollout.  The name
# no longer represents a filesystem lock; all behavior is delegated to the
# authoritative policy-state machine.
O2LockedError = O2PolicyDeniedError


class O2MasterUnavailableError(RuntimeError):
    """Raised when a command needs the ControlMaster but none is running."""


class O2OffVpnError(RuntimeError):
    """Raised when opening a new login would egress off the HMS VPN (→ a Duo push)."""


class O2LoginCoordinationError(O2LoginGrantError):
    """Compatibility error for callers that previously handled login mutex failures."""


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
    """Manage persistent O2 ControlMasters under one global policy state."""

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

    # Caller values for these keywords could re-enable authentication, launch a
    # proxy/local helper, or change the endpoint/socket lifecycle. Reject them
    # explicitly instead of depending on OpenSSH's repeated-option precedence.
    _SSH_FORBIDDEN_CALLER_O_OPTIONS = frozenset(
        {
            "batchmode",
            "certificatefile",
            "challengeresponseauthentication",
            "controlmaster",
            "controlpersist",
            "gssapiauthentication",
            "hostbasedauthentication",
            "hostname",
            "identityagent",
            "identityfile",
            "kbdinteractiveauthentication",
            "knownhostscommand",
            "localcommand",
            "numberofpasswordprompts",
            "passwordauthentication",
            "permitlocalcommand",
            "pkcs11provider",
            "port",
            "preferredauthentications",
            "proxycommand",
            "proxyjump",
            "pubkeyauthentication",
            "securitykeyprovider",
            "user",
        }
    )

    def __init__(
        self,
        config: O2Config | None = None,
        runner: Runner = default_runner,
        *,
        policy: O2PolicyStore | None = None,
    ) -> None:
        """Create a connection bound to the global policy and injected runner.

        A caller may inject ``policy`` for deterministic offline tests.  Normal
        MCP processes construct stores from :class:`O2Config`, whose process-wide
        client identity ensures an authorization issued by one tool call can be
        consumed by a later call in the same task but not by another task.
        """

        self.config = config or O2Config()
        self._runner = runner
        self.policy = policy or O2PolicyStore(self.config.policy_file)
        # A connection may build an rsync argv and then validate/run it. Cache the
        # safely expanded config for each target so those two steps cannot drift
        # and do not repeatedly parse the same local files.
        self._resolved_ssh_configs: dict[str, dict[str, str]] = {}
        self._safe_ssh_config_text_cache: str | None = None

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
        ``"utun6"`` or ``"en0"``) or ``None`` when it cannot be determined. The
        caller treats an indeterminate route as off-VPN unless the consumed grant
        explicitly permits off-VPN authentication.
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
            # Route inspection is local-only. Returning None lets the caller emit
            # one actionable authorization error rather than confusing a missing
            # local utility with a remote connection failure.
            return None

    def _require_on_vpn(self, target: str) -> None:
        """Refuse a new login that would leave via a non-VPN (physical) interface.

        O2 autopushes Duo to non-HMS source IPs. Unknown routing now fails closed:
        the user may authorize the same one-shot login with ``allow_offvpn`` when
        VPN routing is intentionally unavailable.
        """
        iface = self._egress_interface(target)
        if iface is None or not iface.startswith(self.config.vpn_iface_prefix):
            route_detail = "could not be determined" if iface is None else f"uses '{iface}'"
            raise O2OffVpnError(
                f"The local route for O2 target '{target}' {route_detail}, not a proven HMS VPN tunnel "
                f"('{self.config.vpn_iface_prefix}*'). Connect GlobalProtect or issue a new one-shot login "
                "grant with allow_offvpn=true after explicit user approval."
            )

    def start_master(self, *, grant_id: str | None = None, alias: str | None = None) -> CommandResult:
        """Open one ControlMaster only after consuming a matching policy grant.

        An already-running exact master is a local no-op and does not consume a
        grant.  Otherwise the grant is route-checked, atomically consumed, and
        converted to an active attempt receipt before the sole authentication-
        capable SSH subprocess is launched.  No failure path retries SSH.
        """

        self.policy.require_reuse_allowed()
        target = alias or self.config.host_alias
        logical_target: LoginTarget = "transfer" if target == self.config.transfer_alias else "login"
        if target not in {self.config.host_alias, self.config.transfer_alias}:
            raise O2LoginGrantError(f"A new master may target only configured O2 aliases, not '{target}'.")
        if self.master_running(target):
            return CommandResult(self._master_check_argv(target), 0, "master already running", "")
        if not grant_id:
            raise O2MasterUnavailableError(
                f"No O2 ControlMaster is running for '{target}'. A short-lived one-shot login grant scoped to "
                f"'{logical_target}' is required; ordinary booleans cannot authorize a Duo-pushing login."
            )

        grant = self.policy.preview_login_grant(grant_id, logical_target)
        if not grant.allow_offvpn:
            self._require_on_vpn(target)
        consumed = None
        try:
            # Consumption and the authentication-capable runner call share the
            # workstation policy mutex. A concurrent disable therefore wins
            # before consumption or completes only after launch; it can never
            # persist disabled in the gap between those two actions. Contexts
            # enter left-to-right so safe config preparation still happens
            # before the one-shot grant is consumed.
            with (
                self._safe_ssh_config_path() as safe_config,
                self.policy.consume_login_grant_for_launch(grant_id, logical_target) as consumed,
            ):
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
        except subprocess.TimeoutExpired:
            if consumed is not None:
                self.policy.finish_login_attempt(consumed.id, outcome="timed_out", returncode=None)
            raise
        except Exception:
            if consumed is not None:
                self.policy.finish_login_attempt(consumed.id, outcome="error", returncode=None)
            raise

        if not result.ok:
            self.policy.finish_login_attempt(consumed.id, outcome="failed", returncode=result.returncode)
            return result

        # ``ssh -f`` can return zero before the backgrounded master disappears.
        # Only an exact socket control check is accepted as the postcondition.
        try:
            verification = self._runner(
                self._master_check_argv(target),
                self.config.connect_timeout + 5,
                None,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.policy.finish_login_attempt(consumed.id, outcome="failed", returncode=255)
            detail = f"{type(exc).__name__}: {exc}"
            return CommandResult(
                argv=self._master_check_argv(target),
                returncode=255,
                stdout="",
                stderr=self._master_verification_error(target, detail, result.stderr),
            )

        if verification.ok:
            self.policy.finish_login_attempt(consumed.id, outcome="success", returncode=0)
            return result

        self.policy.finish_login_attempt(consumed.id, outcome="failed", returncode=verification.returncode)
        detail = verification.stderr.strip() or verification.stdout.strip() or "no SSH diagnostics"
        return CommandResult(
            argv=verification.argv,
            returncode=verification.returncode or 255,
            stdout=verification.stdout,
            stderr=self._master_verification_error(target, detail, result.stderr),
        )

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
        self.policy.require_reuse_allowed()
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

    def _transport_endpoints_from_argv(self, argv: list[str]) -> list[str]:
        """Return every SSH/rsync ``[user@]host`` endpoint present in argv.

        An empty list is distinct from an unrecognized endpoint: local-only rsync
        has no remote host and may safely use the login default, while a typoed or
        ungoverned host must be rejected before a pinned O2 socket overrides it.
        Rsync treats a colon as remote-shell syntax only when it appears before
        any slash, so local paths such as ``./sample:a`` remain local operands.
        """

        if not argv:
            return []
        executable = argv[0]
        if executable in {"ssh", self.SSH_EXECUTABLE}:
            destination = self._ssh_destination_from_argv(argv)
            return [destination] if destination is not None else []
        if executable not in {"rsync", self.RSYNC_EXECUTABLE}:
            return []

        endpoints: list[str] = []
        for operand in self._rsync_operands_from_argv(argv):
            endpoint, separator, _remote_path = operand.partition(":")
            if separator and "/" not in endpoint:
                endpoints.append(endpoint)
        return endpoints

    def _target_alias_from_argv(self, argv: list[str]) -> str | None:
        """Infer the configured O2 endpoint used by a raw rsync/SSH argv.

        Parse each command's option structure first so an alias mentioned in an
        option value or remote command cannot select the wrong socket. Strip the
        optional user qualifier only for host comparison, then return the full
        ``[user@]alias`` endpoint because ``%r``-based ControlPath templates
        resolve different sockets for different users.
        """

        endpoints = self._transport_endpoints_from_argv(argv)
        for alias in (self.config.transfer_alias, self.config.host_alias):
            if not alias:
                continue
            for endpoint in endpoints:
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
                elif option in {"I", "J", "O", "i", "l", "p"}:
                    option_label = {
                        "I": "PKCS11 provider",
                        "J": "ProxyJump",
                        "O": "control command",
                        "i": "identity file",
                        "l": "user",
                        "p": "port",
                    }[option]
                    raise O2UnsafeTransportError(
                        f"SSH {option_label} options are disabled for guarded O2 transports; "
                        "use user@o2 or user@o2-transfer for user selection and keep host/port in the inspected alias."
                    )
                elif option == "o":
                    option_name = self._ssh_o_option_name(argument)
                    if option_name in self._SSH_FORBIDDEN_CALLER_O_OPTIONS:
                        raise O2UnsafeTransportError(
                            f"SSH {option_name} options are disabled for guarded O2 transports; "
                            "authentication, proxy/helper, and endpoint settings are owned by the guarded transport."
                        )
                    if option_name == "controlpath":
                        if kept_flags:
                            sanitized.append("-" + "".join(kept_flags))
                    else:
                        # Benign caller options remain available, but only after
                        # every safety-sensitive keyword has been denied above.
                        sanitized.append(token)
                        if consumes_next:
                            sanitized.append(argument)
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
        """Run a fail-closed SSH/rsync transport after policy and master checks.

        rsync opens its own ssh via ``-e`` and is meant to reuse the existing
        ControlMaster socket from the SSH config. By default this refuses unless a
        master is already running. The transport is also rewritten with every SSH
        authentication method disabled; this closes the check/use race where a
        dying socket could otherwise make OpenSSH fall back to a brand-new MFA
        login. The guard verifies the endpoint inferred from ``argv`` (a
        ``[user@]<alias>:path`` rsync target or bare ``[user@]<alias>`` SSH host),
        falling back to the login alias only when no endpoint is present. An
        explicit ``master_alias`` must exactly match an inferred endpoint because
        the pinned socket determines the actual host and user. Thus neither a
        transfer-node alias nor a user-qualified socket can disagree with the
        command text silently. Like :meth:`run`, global policy is checked first.
        ``require_master=False`` is retained only to give existing callers an
        actionable error; cold transports are no longer supported.
        """
        self.policy.require_reuse_allowed()
        if not require_master:
            raise O2UnsafeTransportError(
                "Cold O2 SSH/rsync execution is disabled. Start one explicitly authorized ControlMaster, then retry."
            )
        endpoints = self._transport_endpoints_from_argv(argv)
        configured_aliases = {alias for alias in (self.config.host_alias, self.config.transfer_alias) if alias}
        unrecognized = [endpoint for endpoint in endpoints if endpoint.rsplit("@", 1)[-1] not in configured_aliases]
        if unrecognized:
            raise O2UnsafeTransportError(
                f"Transport destination '{unrecognized[0]}' is not a configured O2 alias; "
                "refusing to override it with the pinned login socket."
            )
        if len(set(endpoints)) > 1:
            raise O2UnsafeTransportError(
                "One guarded transport cannot name multiple O2 endpoints: " + ", ".join(dict.fromkeys(endpoints))
            )

        inferred_target = self._target_alias_from_argv(argv)
        if master_alias is not None and inferred_target is not None and master_alias != inferred_target:
            # The pinned socket determines the actual multiplexed destination.
            # Letting an explicit alias disagree with argv would silently execute
            # on the socket's host/user rather than the destination the caller
            # wrote, especially when `%r` gives user-qualified sockets.
            raise O2UnsafeTransportError(
                f"Explicit master_alias '{master_alias}' disagrees with transport destination "
                f"'{inferred_target}'; use one identical [user@]alias or omit master_alias."
            )
        effective_target = master_alias if master_alias is not None else inferred_target
        target = effective_target or self.config.host_alias
        hardened = self._harden_raw_transport_argv(argv, target)
        if not self.master_running(effective_target):
            raise O2MasterUnavailableError(
                f"No O2 ControlMaster is running for '{effective_target or self.config.host_alias}'; refusing a raw "
                "transport (rsync/ssh) that would open a fresh Duo-pushing login. Authorize and consume one "
                "host-scoped login grant through o2_start_master, then reuse that exact connection."
            )
        return self._runner(hardened, timeout, None)
