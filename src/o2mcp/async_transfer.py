"""Detached (non-blocking) rsync transfers to/from O2.

:meth:`O2Sync.push`/:meth:`~O2Sync.pull` block until rsync finishes — fine for a
script, but it ties up an MCP tool call for the whole transfer (minutes for a
multi-GB upload). :class:`O2AsyncTransfer` instead *launches* rsync as a detached
background process — its own session, stdout/stderr redirected to a log file —
and returns a handle immediately. The transfer keeps running between tool calls,
so the agent can do other work and poll :meth:`status` when it wants.

Completion survives an MCP-server restart. A tiny ``bash`` wrapper records
rsync's real exit code to a ``.rc`` file when it exits, so :meth:`status` reports
done/failed by reading files on disk even though the in-memory ``Popen`` handle
is gone. State (one ``.json`` + ``.log`` + ``.rc`` per transfer) lives under
``~/.cache/o2mcp/transfers`` (override with ``O2_ASYNC_STATE_DIR``).

The same safety contract as the blocking path is enforced *before* launching: the
local ``O2_DISABLED`` lock is honored, and a transfer refuses unless the
ControlMaster for its alias is already up — so a background rsync can never open
a fresh Duo-pushing login. The detached rsync reuses that master through the SSH
config's ControlPath, exactly like :class:`O2Sync`.

The subprocess seam (``spawner``) and clock are injected so the whole class is
unit-tested offline without spawning real processes.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from o2mcp.connection import O2Connection, O2LockedError, O2MasterUnavailableError
from o2mcp.sync import O2Sync


class Process(Protocol):
    """The slice of ``subprocess.Popen`` the manager relies on (so tests can fake it)."""

    pid: int

    def poll(self) -> int | None:  # None while running; the exit code once finished
        ...


# A spawner launches ``argv`` detached, writing output to ``log_path``; returns the process.
Spawner = Callable[[list[str], Path], Process]

# Processes launched in THIS server lifetime, keyed by transfer id. Keeping the
# Popen lets status() reap it via poll() (so no zombies pile up) and read accurate
# liveness; after a server restart the registry is empty and status() falls back to
# the on-disk pid probe + exit-code file.
_LIVE: dict[str, Process] = {}

# Monotonic, process-global transfer-id counter. A per-instance counter would reset on
# every tool call (the MCP server builds a fresh O2AsyncTransfer per call), so two
# same-second transfers in one process could collide on the id and overwrite each
# other's state files; a global counter (itertools.count is atomic) prevents that.
_ID_SEQ = itertools.count(1)

# A generated transfer id: "<push|pull>-YYYYmmdd-HHMMSS-<pid>-<seq>". status()/cancel()
# validate a caller-supplied id against this before building any path, so a malformed or
# relative id (e.g. "../x", "/tmp/x") can never resolve a metadata path outside state_dir.
_ID_RE = re.compile(r"(?:push|pull)-\d{8}-\d{6}-\d+-\d+")


def default_spawner(argv: list[str], log_path: Path) -> Process:
    """Launch ``argv`` detached, stdout+stderr -> ``log_path``; return the Popen.

    ``start_new_session=True`` puts the child in its own session/process group so
    it outlives this process and can be signalled as a group on cancel. stdin is
    closed so rsync can never block waiting on a prompt.
    """
    # The `with` closes the parent's copy of the fd once Popen has dup'd it for the
    # child; the detached child keeps writing through its own inherited copy.
    with open(log_path, "ab", buffering=0) as log:
        return subprocess.Popen(
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


_TOCHK = re.compile(r"to-ch(?:k|eck)=(\d+)/(\d+)")  # rsync's running counter (to-chk / to-check)
# A per-file progress sample: "<bytes> <pct>% <rate> <eta>". rsync writes these with \r,
# which str.splitlines() splits on, so each sample is its own line here.
_PROGRESS_SAMPLE = re.compile(r"^[\d,]+\s+\d+%")
_DONE_SAMPLE = re.compile(r"^[\d,]+\s+100%")


def _parse_progress(text: str) -> dict[str, Any]:
    """Best-effort summary of rsync ``--progress`` output: files done/total + current file.

    Handles two client formats. Real rsync emits a running ``to-chk=remaining/total``
    counter (exact done/total). openrsync (stock macOS) does not, so we fall back to
    counting completed-file (100%) samples; a total isn't reliably knowable there (its
    "Transfer starting: N files" counts the whole tree, not the to-send subset), so it
    stays None.
    """
    files_done = files_total = None
    lines = text.splitlines()
    chk = list(_TOCHK.finditer(text))
    if chk:
        remaining, total = int(chk[-1].group(1)), int(chk[-1].group(2))
        files_total = total
        files_done = total - remaining
    else:
        done = sum(1 for line in lines if _DONE_SAMPLE.match(line.strip()))
        files_done = done or None
    # current/last file = newest line that is neither a progress sample nor a header.
    last_file = None
    for line in reversed(lines):
        s = line.strip()
        if not s or _PROGRESS_SAMPLE.match(s) or s.startswith("Transfer starting"):
            continue
        last_file = s
        break
    return {"files_done": files_done, "files_total": files_total, "last_file": last_file}


@dataclass
class TransferHandle:
    """A launched background transfer (also the on-disk ``.json`` metadata schema)."""

    id: str
    direction: str  # "push" | "pull"
    local: str
    remote: str
    transfer_node: bool
    argv: list[str]
    pid: int
    log_path: str
    rc_path: str
    meta_path: str
    start_time: float


def _default_state_dir() -> Path:
    env = os.environ.get("O2_ASYNC_STATE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "o2mcp" / "transfers"


class O2AsyncTransfer:
    """Launch and monitor detached rsync transfers (non-blocking push/pull)."""

    def __init__(
        self,
        connection: O2Connection | None = None,
        *,
        sync: O2Sync | None = None,
        state_dir: str | Path | None = None,
        spawner: Spawner = default_spawner,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.conn = connection or O2Connection()
        self.sync = sync or O2Sync(self.conn)
        self.state_dir = Path(state_dir) if state_dir is not None else _default_state_dir()
        self._spawn = spawner
        self._clock = clock

    # -- launch -----------------------------------------------------------------
    def push_async(self, local_path: str, remote_path: str, *, transfer: bool = False) -> TransferHandle:
        """Start a background upload of ``local_path`` to ``remote_path`` on O2."""
        return self._launch("push", local=local_path, remote=remote_path, transfer=transfer)

    def pull_async(self, remote_path: str, local_path: str, *, transfer: bool = False) -> TransferHandle:
        """Start a background download of ``remote_path`` into ``local_path``."""
        return self._launch("pull", local=local_path, remote=remote_path, transfer=transfer)

    def _launch(self, direction: str, *, local: str, remote: str, transfer: bool) -> TransferHandle:
        if self.conn.is_locked():
            raise O2LockedError(
                f"O2 access is locally disabled by {self.conn.config.lock_file}. "
                "Refusing to launch a background transfer."
            )
        alias = self.sync._alias(transfer)
        if not self.conn.master_running(alias):
            raise O2MasterUnavailableError(
                f"No O2 ControlMaster is running for '{alias}'; refusing to launch a background rsync "
                "that would open a fresh Duo-pushing login. Start one first (o2_start_master with "
                "allow_new_login=True), then retry."
            )
        if direction == "push":
            argv = self.sync.push_argv(local, remote, transfer=transfer)
        else:
            argv = self.sync.pull_argv(remote, local, transfer=transfer)

        self.state_dir.mkdir(parents=True, exist_ok=True)
        tid = self._new_id(direction)
        log_path = self.state_dir / f"{tid}.log"
        rc_path = self.state_dir / f"{tid}.rc"
        meta_path = self.state_dir / f"{tid}.json"
        # Wrapper records rsync's exit code to <rc_path> and then exits WITH it, so
        # completion is detectable two ways: in-process via the Popen (poll() returns
        # rsync's code, since the wrapper propagates it) and, after a server restart,
        # from the <rc_path> file on disk. Running rsync as a child (not exec) lets
        # the wrapper write the file; "$@" replays argv verbatim — no re-quoting.
        wrapped = [
            "bash",
            "-c",
            'rc="$1"; shift; "$@"; ec=$?; echo "$ec" > "$rc"; exit "$ec"',
            "bash",
            str(rc_path),
            *argv,
        ]
        proc = self._spawn(wrapped, log_path)
        _LIVE[tid] = proc
        handle = TransferHandle(
            id=tid,
            direction=direction,
            local=local,
            remote=remote,
            transfer_node=transfer,
            argv=argv,
            pid=proc.pid,
            log_path=str(log_path),
            rc_path=str(rc_path),
            meta_path=str(meta_path),
            start_time=self._clock(),
        )
        meta_path.write_text(json.dumps(asdict(handle), indent=2))
        return handle

    def _new_id(self, direction: str) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._clock()))
        return f"{direction}-{stamp}-{os.getpid()}-{next(_ID_SEQ):04d}"

    def _meta_path(self, transfer_id: str) -> Path | None:
        """The metadata path for a well-formed transfer id, else ``None``.

        Validating against the generated-id pattern keeps a caller-supplied id with path
        separators (``../x``, ``/tmp/x``) from resolving a metadata path outside state_dir.
        """
        if not _ID_RE.fullmatch(transfer_id):
            return None
        return self.state_dir / f"{transfer_id}.json"

    # -- monitor ----------------------------------------------------------------
    def status(self, transfer_id: str | None = None, *, log_tail: int = 20):
        """One transfer's status (``transfer_id`` given) or a list of all known transfers."""
        if transfer_id is None:
            metas = sorted(self.state_dir.glob("*.json")) if self.state_dir.exists() else []
            return [self._status_from_meta(p, log_tail=log_tail) for p in metas]
        meta_path = self._meta_path(transfer_id)
        if meta_path is None or not meta_path.exists():
            return {"ok": False, "error": "unknown_transfer", "transfer_id": transfer_id}
        return self._status_from_meta(meta_path, log_tail=log_tail)

    def _status_from_meta(self, meta_path: Path, *, log_tail: int) -> dict[str, Any]:
        meta = json.loads(meta_path.read_text())
        rc_path = Path(meta["rc_path"])
        log_path = Path(meta["log_path"])

        # rsync's real exit code, from the file the wrapper writes (authoritative,
        # and the only source after a server restart).
        returncode = None
        if rc_path.exists():
            raw = rc_path.read_text().strip()
            try:
                returncode = int(raw)
            except ValueError:
                returncode = None

        # Liveness. A recorded rc file is authoritative: the wrapper writes it only after
        # the transfer exits, so the transfer is finished regardless of whether the PID is
        # now alive (after a restart the OS may have reused it). Otherwise prefer the
        # in-process Popen (poll() reaps it -> no zombie, accurate even with no rc, e.g. a
        # cancelled transfer), and only as a last resort probe the bare PID.
        proc = _LIVE.get(meta["id"])
        if returncode is not None:
            finished = True
        elif proc is not None:
            code = proc.poll()
            finished = code is not None
            if finished:
                returncode = code  # wrapper propagates rsync's code; covers a missing rc file
        else:
            finished = not _pid_alive(int(meta["pid"]))

        if not finished:
            state = "running"
        elif returncode is None:
            state = "crashed"  # finished but never recorded an exit code (killed before writing rc)
        else:
            state = "done" if returncode == 0 else "failed"
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        tail = "\n".join(log_text.splitlines()[-log_tail:]) if log_tail else ""
        return {
            "ok": True,
            "transfer_id": meta["id"],
            "direction": meta["direction"],
            "state": state,
            "returncode": returncode,
            "pid": meta["pid"],
            "local": meta["local"],
            "remote": meta["remote"],
            "elapsed_s": round(self._clock() - meta["start_time"], 1),
            "progress": _parse_progress(log_text),
            "log_path": meta["log_path"],
            "log_tail": tail,
        }

    # -- control ----------------------------------------------------------------
    def cancel(self, transfer_id: str) -> dict[str, Any]:
        """Stop a running transfer this server launched (SIGTERM to its process group).

        Only transfers launched in THIS process (tracked in ``_LIVE``) can be cancelled:
        their live ``Popen`` identifies the real child, so a possibly-reused PID is never
        signalled. A transfer that already finished (rc file present) is a no-op, and one
        launched by a previous server session cannot be cancelled by PID alone (the OS may
        have recycled it) — it is refused rather than risk killing an unrelated process.
        """
        meta_path = self._meta_path(transfer_id)
        if meta_path is None or not meta_path.exists():
            return {"ok": False, "error": "unknown_transfer", "transfer_id": transfer_id}
        meta = json.loads(meta_path.read_text())
        if Path(meta["rc_path"]).exists():
            return {
                "ok": True,
                "transfer_id": transfer_id,
                "signalled": False,
                "state": "finished",
                "message": "transfer already finished; nothing to cancel",
            }
        proc = _LIVE.get(meta["id"])
        if proc is None:
            return {
                "ok": False,
                "error": "not_cancellable",
                "transfer_id": transfer_id,
                "signalled": False,
                "message": (
                    "transfer was launched by a previous server session; refusing to signal its PID "
                    "(the OS may have reused it). Kill it manually if it is somehow still running."
                ),
            }
        pid = proc.pid
        signalled = False
        if proc.poll() is None:  # our child, still running -> safe to signal its group
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                signalled = True
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGTERM)
                    signalled = True
                except (ProcessLookupError, PermissionError):
                    signalled = False
        return {"ok": True, "transfer_id": transfer_id, "pid": pid, "signalled": signalled}
