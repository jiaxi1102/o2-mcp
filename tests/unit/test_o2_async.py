"""Offline unit tests for o2mcp.async_transfer (detached rsync transfers).

The subprocess spawner and clock are injected, so these never spawn a real
process or touch the network: they assert the wrapped command is built correctly
(incl. remote-path quoting), the safety lock + ControlMaster guards fire before
any launch, and status/cancel report the right state from a faked process +
on-disk exit-code files (incl. the post-restart fallback when the Popen is gone).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from o2mcp import (
    CommandResult,
    O2AsyncTransfer,
    O2Config,
    O2Connection,
    O2LockedError,
    O2MasterUnavailableError,
    O2Sync,
    async_transfer,
)


@pytest.fixture(autouse=True)
def _clear_live_registry():
    """The launched-process registry is module-global; isolate it per test."""
    async_transfer._LIVE.clear()
    yield
    async_transfer._LIVE.clear()


class FakeProc:
    """Minimal stand-in for subprocess.Popen: pid + poll() (None until finished)."""

    def __init__(self, pid: int):
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def finish(self, code: int) -> None:
        self.returncode = code


class FakeSpawner:
    """Records launches and returns a FakeProc the test can later finish()."""

    def __init__(self, *, pid: int = 4321):
        self.calls: list[dict] = []
        self.pid = pid
        self.procs: list[FakeProc] = []

    def __call__(self, argv, log_path) -> FakeProc:
        self.calls.append({"argv": list(argv), "log_path": Path(log_path)})
        proc = FakeProc(self.pid)
        self.procs.append(proc)
        return proc


def _conn(tmp_path: Path, *, master: bool = True, locked: bool = False) -> O2Connection:
    lock = tmp_path / "O2_DISABLED"
    if locked:
        lock.write_text("disabled")
    cfg = O2Config(host_alias="o2", transfer_alias="o2-transfer", connect_timeout=20, lock_file=lock)

    def runner(argv, timeout, input_text) -> CommandResult:
        if "-O" in argv and "check" in argv:
            return CommandResult(list(argv), 0 if master else 255, "", "")
        return CommandResult(list(argv), 0, "", "")

    return O2Connection(cfg, runner=runner)


def _mgr(tmp_path: Path, spawner: FakeSpawner, **conn_kw) -> O2AsyncTransfer:
    return O2AsyncTransfer(
        _conn(tmp_path, **conn_kw),
        state_dir=tmp_path / "state",
        spawner=spawner,
        clock=lambda: 1000.0,
    )


# --- launch ------------------------------------------------------------------
def test_push_async_launches_detached_with_escaped_remote(tmp_path):
    spawner = FakeSpawner()
    mgr = _mgr(tmp_path, spawner)
    remote = "/n/groups/tabin/jzhao/o2_gem_diffusion/data/20260329 - 20nm GEM Human Mouse PSM/Human"
    handle = mgr.push_async("/local/Human", remote)

    assert handle.pid == 4321
    assert handle.direction == "push"
    assert async_transfer._LIVE[handle.id] is spawner.procs[0]  # registered for poll()-based reaping
    assert len(spawner.calls) == 1
    wrapped = spawner.calls[0]["argv"]
    # bash wrapper that records + propagates the exit code, then the real rsync argv verbatim.
    assert wrapped[:2] == ["bash", "-c"]
    assert wrapped[3] == "bash" and wrapped[4] == handle.rc_path
    rsync_argv = wrapped[5:]
    assert rsync_argv == O2Sync(_conn(tmp_path)).push_argv("/local/Human", remote)
    assert rsync_argv[0] == "rsync"
    assert rsync_argv[-1] == "o2:" + remote.replace(" ", "\\ ")  # remote path escaped for the remote shell

    # metadata persisted (the schema status() reads); argv stored is the rsync argv, not the wrapper.
    meta = json.loads(Path(handle.meta_path).read_text())
    assert meta["id"] == handle.id and meta["pid"] == 4321 and meta["argv"] == rsync_argv


def test_push_async_requires_master(tmp_path):
    spawner = FakeSpawner()
    mgr = _mgr(tmp_path, spawner, master=False)
    with pytest.raises(O2MasterUnavailableError):
        mgr.push_async("/local/x", "/remote/x")
    assert spawner.calls == []  # nothing launched without a master


def test_push_async_blocked_by_lock(tmp_path):
    spawner = FakeSpawner()
    mgr = _mgr(tmp_path, spawner, locked=True)
    with pytest.raises(O2LockedError):
        mgr.push_async("/local/x", "/remote/x")
    assert spawner.calls == []


def test_pull_async_builds_pull_argv(tmp_path):
    spawner = FakeSpawner()
    mgr = _mgr(tmp_path, spawner)
    handle = mgr.pull_async("/remote/results", "/local/results", transfer=True)
    rsync_argv = spawner.calls[0]["argv"][5:]
    assert rsync_argv == O2Sync(_conn(tmp_path)).pull_argv("/remote/results", "/local/results", transfer=True)
    assert handle.direction == "pull"


# --- status: in-process (Popen.poll drives liveness) -------------------------
def test_status_running_until_process_finishes(tmp_path):
    spawner = FakeSpawner()
    mgr = _mgr(tmp_path, spawner)
    handle = mgr.push_async("/local/x", "/remote/x")
    assert mgr.status(handle.id)["state"] == "running"  # poll() is None
    spawner.procs[0].finish(0)
    st = mgr.status(handle.id)
    assert st["state"] == "done" and st["returncode"] == 0


def test_status_failed_and_cancelled_exit_codes(tmp_path):
    spawner = FakeSpawner()
    mgr = _mgr(tmp_path, spawner)
    # rsync error exit
    h1 = mgr.push_async("/a", "/ra")
    spawner.procs[0].finish(23)
    assert mgr.status(h1.id)["state"] == "failed"
    # killed by signal (no rc file written): negative code -> still "failed", not stuck "running"
    h2 = mgr.push_async("/b", "/rb")
    spawner.procs[1].finish(-15)
    assert mgr.status(h2.id)["state"] == "failed"


def test_status_prefers_rc_file_for_returncode(tmp_path):
    spawner = FakeSpawner()
    mgr = _mgr(tmp_path, spawner)
    handle = mgr.push_async("/local/x", "/remote/x")
    Path(handle.rc_path).write_text("0\n")  # wrapper recorded success
    spawner.procs[0].finish(0)
    st = mgr.status(handle.id)
    assert st["state"] == "done" and st["returncode"] == 0


# --- status: post-restart fallback (Popen gone, read from disk) --------------
def test_status_after_restart_uses_rc_and_pid(tmp_path, monkeypatch):
    spawner = FakeSpawner()
    mgr = _mgr(tmp_path, spawner)
    handle = mgr.push_async("/local/x", "/remote/x")
    async_transfer._LIVE.clear()  # simulate an MCP-server restart: in-memory handle lost

    monkeypatch.setattr(async_transfer, "_pid_alive", lambda pid: True)
    assert mgr.status(handle.id)["state"] == "running"  # no rc, pid still alive

    Path(handle.rc_path).write_text("0\n")
    monkeypatch.setattr(async_transfer, "_pid_alive", lambda pid: False)
    assert mgr.status(handle.id)["state"] == "done"  # rc recorded success

    Path(handle.rc_path).unlink()
    assert mgr.status(handle.id)["state"] == "crashed"  # pid gone, never recorded a code


def test_status_rc_file_is_authoritative_over_live_pid(tmp_path, monkeypatch):
    # After a restart a finished transfer (rc present) must report done/failed even if the
    # recorded PID is now alive again (OS reused it) — the rc file beats the pid probe.
    mgr = _mgr(tmp_path, FakeSpawner())
    handle = mgr.push_async("/local/x", "/remote/x")
    async_transfer._LIVE.clear()  # server restart: Popen handle lost
    monkeypatch.setattr(async_transfer, "_pid_alive", lambda pid: True)  # PID reused / alive
    Path(handle.rc_path).write_text("0\n")
    assert mgr.status(handle.id)["state"] == "done"  # not "running"
    Path(handle.rc_path).write_text("23\n")
    assert mgr.status(handle.id)["state"] == "failed"


def test_status_unknown_id(tmp_path):
    mgr = _mgr(tmp_path, FakeSpawner())
    res = mgr.status("push-nope-0-001")
    assert res["ok"] is False and res["error"] == "unknown_transfer"


def test_status_and_cancel_reject_malformed_ids(tmp_path):
    # Ids with path separators or the wrong shape are rejected before any path is built,
    # so they cannot read/signal outside the state dir (path-traversal guard).
    mgr = _mgr(tmp_path, FakeSpawner())
    for bad in ["../x", "/tmp/x", "push/../../etc", "foo.json", "push-1-2"]:
        assert mgr.status(bad)["error"] == "unknown_transfer"
        assert mgr.cancel(bad)["error"] == "unknown_transfer"
    # a well-formed but absent id is still unknown, via the normal not-found path
    assert mgr.status("push-20260617-001234-99-1")["error"] == "unknown_transfer"


def test_status_lists_all(tmp_path):
    spawner = FakeSpawner()
    mgr = _mgr(tmp_path, spawner)
    mgr.push_async("/a", "/ra")
    mgr.push_async("/b", "/rb")
    listed = mgr.status()
    assert isinstance(listed, list) and len(listed) == 2
    assert {row["remote"] for row in listed} == {"/ra", "/rb"}


def test_progress_parsing_rsync_tochk():
    # Real rsync: a running to-chk=remaining/total gives exact done/total.
    log = (
        "Human_PSM_400k_14.tif\n"
        "  1,234,567  42%   45.00MB/s    0:00:10\n"
        "Human_PSM_400k_15.tif\n"
        "  9,000,000 100%   50.00MB/s    0:00:27 (xfr#13, to-chk=7/20)\n"
    )
    prog = async_transfer._parse_progress(log)
    assert prog["files_total"] == 20 and prog["files_done"] == 13
    assert prog["last_file"] == "Human_PSM_400k_15.tif"  # progress sample lines skipped


def test_progress_parsing_openrsync():
    # openrsync (stock macOS): no to-chk; samples joined by \r (str.splitlines splits
    # them); filenames can start with a digit ("400k/..."); a done file shows 100%.
    log = (
        "Transfer starting: 41 files\n"
        "400k/\n"
        "400k/Human_PSM_400k_3.tif\n"
        "    6553600   1%  12MB/s  0:01:03\r  801000000 100%  13MB/s  0:00:00\r"
        "400k/Human_PSM_400k_4.tif\n"
        "  120000000  15%  11MB/s  0:00:55\r"
    )
    prog = async_transfer._parse_progress(log)
    assert prog["files_done"] == 1 and prog["files_total"] is None  # 1 file at 100%; total unknowable
    assert prog["last_file"] == "400k/Human_PSM_400k_4.tif"  # digit-leading current file handled


# --- cancel ------------------------------------------------------------------
def test_cancel_unknown_id(tmp_path):
    mgr = _mgr(tmp_path, FakeSpawner())
    res = mgr.cancel("push-nope-0-001")
    assert res["ok"] is False and res["error"] == "unknown_transfer"


def test_cancel_owned_transfer_signals_process_group(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, FakeSpawner(pid=5555))
    handle = mgr.push_async("/local/x", "/remote/x")  # registered in _LIVE, poll() -> None (running)

    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(async_transfer.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(async_transfer.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
    res = mgr.cancel(handle.id)
    assert res["ok"] is True and res["signalled"] is True
    assert killed and killed[0][0] == 5555  # SIGTERM to the transfer's process group


def test_cancel_finished_transfer_is_noop(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, FakeSpawner())
    handle = mgr.push_async("/local/x", "/remote/x")
    Path(handle.rc_path).write_text("0\n")  # wrapper recorded completion
    killed: list = []
    monkeypatch.setattr(async_transfer.os, "killpg", lambda *a: killed.append(a))
    res = mgr.cancel(handle.id)
    assert res["ok"] is True and res["signalled"] is False and res["state"] == "finished"
    assert killed == []  # a finished transfer is never signalled


def test_cancel_post_restart_refuses(tmp_path, monkeypatch):
    # After a restart the Popen handle is gone; the PID may have been reused, so refuse.
    mgr = _mgr(tmp_path, FakeSpawner())
    handle = mgr.push_async("/local/x", "/remote/x")
    async_transfer._LIVE.clear()  # simulate the server restart
    killed: list = []
    monkeypatch.setattr(async_transfer.os, "killpg", lambda *a: killed.append(a))
    res = mgr.cancel(handle.id)
    assert res["ok"] is False and res["error"] == "not_cancellable" and res["signalled"] is False
    assert killed == []  # a possibly-reused PID is never signalled
