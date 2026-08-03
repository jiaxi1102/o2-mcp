"""Offline unit tests for o2mcp.async_transfer (detached rsync transfers).

The subprocess spawner and clock are injected, so these never spawn a real
process or touch the network: they assert the wrapped command is built correctly
(incl. remote-path quoting), the policy + ControlMaster guards fire before
any launch, and status/cancel report the right state from a faked process +
on-disk exit-code files (incl. the post-restart fallback when the Popen is gone).
"""

from __future__ import annotations

import json
import threading
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
    policy_file = tmp_path / "O2_POLICY.json"
    policy_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation": "00000000-0000-4000-8000-000000000001",
                "revision": 1,
                "mode": "disabled" if locked else "reuse_only",
                "login_grant": None,
                "login_attempt": None,
                "events": [],
            }
        )
    )
    policy_file.chmod(0o600)
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text(
        "Host o2 o2-transfer\n"
        "  HostName o2.hms.harvard.edu\n"
        "  User jiz947\n"
        "  ControlPath /tmp/%n-control.sock\n"
    )
    cfg = O2Config(
        host_alias="o2",
        transfer_alias="o2-transfer",
        connect_timeout=20,
        policy_file=policy_file,
        ssh_config_file=ssh_config,
    )

    def runner(argv, timeout, input_text) -> CommandResult:
        if "-O" in argv and "check" in argv:
            return CommandResult(list(argv), 0 if master else 255, "", "")
        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
            return CommandResult(list(argv), 0, f"controlpath /tmp/{argv[-1]}-control.sock\n", "")
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
    assert rsync_argv[0] == O2Connection.RSYNC_EXECUTABLE
    transport = rsync_argv[rsync_argv.index("-e") + 1]
    assert "PreferredAuthentications=none" in transport
    assert "PubkeyAuthentication=no" in transport
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


def test_disable_cannot_complete_between_async_check_and_spawn(tmp_path):
    """A detached rsync spawn shares disable's workstation mutex."""

    class DisableDuringSpawn(FakeSpawner):
        """Start a competing policy disable from inside the spawn seam."""

        def __init__(self):
            super().__init__()
            self.policy = None
            self.disable_started = threading.Event()
            self.disable_finished = threading.Event()
            self.disable_thread = None

        def __call__(self, argv, log_path) -> FakeProc:
            assert self.policy is not None

            def disable() -> None:
                self.disable_started.set()
                self.policy.disable(reason="concurrent detached-transfer stop")
                self.disable_finished.set()

            self.disable_thread = threading.Thread(target=disable)
            self.disable_thread.start()
            assert self.disable_started.wait(timeout=1)
            assert not self.disable_finished.wait(timeout=0.1)
            return super().__call__(argv, log_path)

    spawner = DisableDuringSpawn()
    mgr = _mgr(tmp_path, spawner)
    spawner.policy = mgr.conn.policy

    handle = mgr.push_async("/local/x", "/remote/x")

    assert spawner.disable_thread is not None
    spawner.disable_thread.join(timeout=2)
    assert handle.pid == 4321 and spawner.disable_finished.is_set()
    assert mgr.conn.policy.snapshot().effective_mode == "disabled"


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


def test_status_isolates_corrupt_metadata_when_listing(tmp_path):
    """One truncated receipt must not suppress healthy detached transfers."""

    mgr = _mgr(tmp_path, FakeSpawner())
    healthy = mgr.push_async("/a", "/remote-a")
    corrupt = mgr.state_dir / "push-20260617-001234-99-0001.json"
    corrupt.write_text("{truncated")

    listed = mgr.status()

    assert len(listed) == 2
    assert any(row.get("transfer_id") == healthy.id and row["ok"] for row in listed)
    invalid = next(row for row in listed if row.get("meta_path") == str(corrupt))
    assert invalid["ok"] is False
    assert invalid["error"] == "invalid_transfer_metadata"


def test_status_isolates_non_finite_numeric_metadata(tmp_path):
    """JSON infinity cannot escape the corrupt-receipt diagnostic boundary."""

    mgr = _mgr(tmp_path, FakeSpawner())
    corrupt = mgr.state_dir / "push-20260617-001234-99-0001.json"
    mgr.state_dir.mkdir(parents=True)
    corrupt.write_text('{"id":"push-20260617-001234-99-0001","pid":1e999}')

    result = mgr.status("push-20260617-001234-99-0001")

    assert result["ok"] is False
    assert result["error"] == "invalid_transfer_metadata"


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
