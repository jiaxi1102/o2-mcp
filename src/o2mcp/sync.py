"""File transfer to/from O2 via rsync over the existing SSH connection.

rsync reuses the ControlMaster configured in the SSH config (so no extra login),
and the ``-e`` transport is pinned to BatchMode so a missing key fails fast
instead of prompting for Duo. Large transfers can opt into the dedicated O2
transfer node; by default we reuse the login alias to keep to a single
authenticated connection.

Remote paths are backslash-escaped for the remote shell. rsync hands the
post-colon path to a *remote* shell, which otherwise word-splits a path
containing spaces — ``.../20260329 - 20nm GEM Human Mouse PSM/Human`` was being
truncated at the first space, creating a bogus ``.../20260329`` directory. We
escape whitespace and quoting/command characters but deliberately leave the
expansion/glob metacharacters (``~ $ { } * ? [ ]``) bare, so home/variable
expansion still works (``~/jobs``, ``$SCRATCH/out``, ``${SCRATCH}/out``) — unlike
``shlex.quote``, which would wrap the whole path in single quotes and suppress
that expansion. Escaping ``()`` and the backtick keeps ``$(...)``/`` `...` ``
command substitution from firing, so an expanded ``$`` stays safe.
``--protect-args`` would also fix the splitting, but the
local transport here may be openrsync / rsync 2.6.9 (stock macOS), which does not
support it; escaping is version-agnostic and a no-op for plain paths (so existing
transfers are byte-for-byte unchanged). Only the remote side needs it — the local
path is passed as its own argv token and never sees a shell.
"""

from __future__ import annotations

from o2mcp.connection import CommandResult, O2Connection

_DEFAULT_RSYNC_ARGS = ["-avz", "--partial", "--progress"]

# Characters we backslash-escape in a remote path handed to the shell rsync runs it
# through: whitespace (would word-split a multi-word path) plus quoting / command /
# redirection characters. Escaping ``(`` ``)`` and the backtick means ``$(...)`` and
# ``` `...` ``` command substitution cannot fire, so an expanded ``$`` is safe. Expansion
# and glob metacharacters (``~ $ { } * ? [ ]``) are deliberately NOT escaped, so the remote
# shell still expands ``~``, ``$VAR`` and ``${VAR}`` (and matches globs). A literal path
# containing those metacharacters is unsupported — rare for transfer paths, and the
# tradeoff that keeps home/variable shortcuts working.
_REMOTE_ESCAPE = frozenset(" \t\n\r\\'\"`;|&<>()#!")


def _escape_remote_path(path: str) -> str:
    """Backslash-escape a remote path for the shell rsync runs it through.

    Protects spaces and neutralizes quoting/command characters while leaving ``~``,
    ``$VAR``/``${VAR}`` and globs bare so the remote shell still expands them.
    """
    return "".join("\\" + c if c in _REMOTE_ESCAPE else c for c in path)


class O2Sync:
    """rsync push/pull helpers built on an :class:`O2Connection`."""

    def __init__(self, connection: O2Connection) -> None:
        self.conn = connection

    def _rsync_e_opt(self) -> str:
        """The ``-e`` ssh transport string enforcing batch mode."""
        return "ssh " + " ".join(self.conn.config.base_ssh_opts())

    def _alias(self, transfer: bool) -> str:
        return self.conn.config.transfer_alias if transfer else self.conn.config.host_alias

    def _remote(self, alias: str, path: str) -> str:
        """``<alias>:<path>`` with the remote path escaped for the remote shell."""
        return f"{alias}:{_escape_remote_path(path)}"

    def push_argv(
        self,
        local_path: str,
        remote_path: str,
        *,
        transfer: bool = False,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """The exact rsync argv :meth:`push` would run (built, not executed)."""
        return self._build_rsync(
            source=local_path,
            dest=self._remote(self._alias(transfer), remote_path),
            extra_args=extra_args,
        )

    def pull_argv(
        self,
        remote_path: str,
        local_path: str,
        *,
        transfer: bool = False,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """The exact rsync argv :meth:`pull` would run (built, not executed)."""
        return self._build_rsync(
            source=self._remote(self._alias(transfer), remote_path),
            dest=local_path,
            extra_args=extra_args,
        )

    def push(
        self,
        local_path: str,
        remote_path: str,
        *,
        transfer: bool = False,
        extra_args: list[str] | None = None,
        timeout: float | None = 3600.0,
    ) -> CommandResult:
        """Upload ``local_path`` to ``<alias>:remote_path`` on O2."""
        argv = self.push_argv(local_path, remote_path, transfer=transfer, extra_args=extra_args)
        return self.conn.run_raw(argv, timeout=timeout, master_alias=self._alias(transfer))

    def pull(
        self,
        remote_path: str,
        local_path: str,
        *,
        transfer: bool = False,
        extra_args: list[str] | None = None,
        timeout: float | None = 3600.0,
    ) -> CommandResult:
        """Download ``<alias>:remote_path`` from O2 into ``local_path``."""
        argv = self.pull_argv(remote_path, local_path, transfer=transfer, extra_args=extra_args)
        return self.conn.run_raw(argv, timeout=timeout, master_alias=self._alias(transfer))

    def _build_rsync(self, *, source: str, dest: str, extra_args: list[str] | None) -> list[str]:
        return [
            "rsync",
            *_DEFAULT_RSYNC_ARGS,
            "-e",
            self._rsync_e_opt(),
            *(extra_args or []),
            source,
            dest,
        ]
