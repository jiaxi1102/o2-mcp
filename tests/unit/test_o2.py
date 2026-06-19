"""Offline unit tests for the o2mcp cluster primitives.

The subprocess seam is injected, so these never touch the network: they assert
the exact ssh/sbatch/squeue/rsync commands are built, the safety lock and
ControlMaster guards fire, and Slurm output is parsed correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from o2mcp import (
    CommandResult,
    O2Config,
    O2Connection,
    O2LockedError,
    O2MasterUnavailableError,
    O2OffVpnError,
    O2Slurm,
    O2Sync,
)
from o2mcp import keepalive as o2keepalive


class RecordingRunner:
    """A fake subprocess runner: records calls, answers via a response function."""

    def __init__(self, *, master: bool = True, responder=None):
        self.calls = []
        self.master = master
        self._responder = responder

    def __call__(self, argv, timeout, input_text) -> CommandResult:
        self.calls.append({"argv": list(argv), "timeout": timeout, "input": input_text})
        if "-O" in argv and "check" in argv:
            return CommandResult(list(argv), 0 if self.master else 255, "", "")
        if "-MNf" in argv:
            return CommandResult(list(argv), 0, "", "")
        if self._responder is not None:
            out, err, rc = self._responder(argv, input_text)
            return CommandResult(list(argv), rc, out, err)
        return CommandResult(list(argv), 0, "", "")

    @property
    def remote_commands(self):
        """The remote command string of every `ssh ... <alias> <cmd>` call."""
        cmds = []
        for call in self.calls:
            argv = call["argv"]
            if argv and argv[0] == "ssh" and "-O" not in argv and "-MNf" not in argv:
                cmds.append(argv[-1])
        return cmds


def _config(tmp_path: Path, *, locked: bool = False) -> O2Config:
    lock = tmp_path / "O2_DISABLED"
    if locked:
        lock.write_text("disabled")
    return O2Config(host_alias="o2", transfer_alias="o2-transfer", connect_timeout=20, lock_file=lock)


# --- safety lock -------------------------------------------------------------
def test_lock_blocks_everything(tmp_path):
    runner = RecordingRunner()
    conn = O2Connection(_config(tmp_path, locked=True), runner=runner)
    assert conn.is_locked() is True
    assert conn.master_running() is False
    with pytest.raises(O2LockedError):
        conn.run("hostname")
    with pytest.raises(O2LockedError):
        conn.run_raw(["rsync", "x", "y"])
    # No ssh was ever attempted under a lock.
    assert runner.calls == []


# --- ControlMaster guards ----------------------------------------------------
def test_run_requires_master(tmp_path):
    runner = RecordingRunner(master=False)
    conn = O2Connection(_config(tmp_path), runner=runner)
    assert conn.master_running() is False
    with pytest.raises(O2MasterUnavailableError):
        conn.run("squeue")


def test_run_raw_requires_master(tmp_path):
    # A raw transport (rsync/ssh) must also refuse without a master, or it would
    # open a fresh connection — a new Duo push — outside the one approved master.
    runner = RecordingRunner(master=False)
    conn = O2Connection(_config(tmp_path), runner=runner)
    assert conn.master_running() is False
    with pytest.raises(O2MasterUnavailableError):
        conn.run_raw(["rsync", "x", "y"])
    # An explicit opt-out is still honored for a deliberately-cold transport.
    conn.run_raw(["rsync", "x", "y"], require_master=False)
    assert runner.calls[-1]["argv"][0] == "rsync"


def test_run_uses_batchmode_and_alias(tmp_path):
    runner = RecordingRunner(master=True)
    conn = O2Connection(_config(tmp_path), runner=runner)
    result = conn.run("hostname; whoami")
    assert result.ok
    last = runner.calls[-1]["argv"]
    assert last[0] == "ssh"
    assert "BatchMode=yes" in last
    assert last[-2] == "o2"
    assert last[-1] == "hostname; whoami"


def test_start_master_requires_opt_in(tmp_path):
    runner = RecordingRunner(master=False)
    conn = O2Connection(_config(tmp_path), runner=runner)
    with pytest.raises(O2MasterUnavailableError):
        conn.start_master(allow_new_login=False)
    # Opt-in opens the master with -MNf.
    result = conn.start_master(allow_new_login=True)
    assert result.ok
    assert "-MNf" in runner.calls[-1]["argv"]


def test_start_master_noop_when_running(tmp_path):
    conn = O2Connection(_config(tmp_path), runner=RecordingRunner(master=True))
    result = conn.start_master(allow_new_login=False)
    assert result.ok and "already running" in result.stdout


def test_start_master_can_open_the_transfer_alias(tmp_path):
    # The transfer node has its own master; start_master(alias=...) must be able to
    # open it so transfer-node rsync/ssh has a socket to reuse (otherwise the new
    # transfer-master guard could never be satisfied).
    runner = RecordingRunner(master=False)
    conn = O2Connection(_config(tmp_path), runner=runner)
    result = conn.start_master(allow_new_login=True, alias=conn.config.transfer_alias)
    assert result.ok
    assert "-MNf" in runner.calls[-1]["argv"]
    assert runner.calls[-1]["argv"][-1] == "o2-transfer"


# --- VPN egress guard (HMS O2 Duos non-HMS source IPs) -----------------------
def _vpn_responder(interface):
    """Responder answering the guard's `ssh -G` (HostName) and `route get` (interface)."""

    def responder(argv, input_text):
        if argv[:2] == ["ssh", "-G"]:
            return ("hostname o2.hms.harvard.edu\n", "", 0)
        if argv[:2] == ["route", "get"]:
            if interface is None:
                return ("", "no route to host", 1)  # undetermined -> caller fails open
            return (f"   route to: o2\n   interface: {interface}\n   gateway: x\n", "", 0)
        return ("", "", 0)

    return responder


def test_start_master_refuses_off_vpn(tmp_path):
    # Route egresses via a physical interface (en0) -> a fresh login would Duo. Refuse,
    # and never open the login (-MNf is never invoked).
    runner = RecordingRunner(master=False, responder=_vpn_responder("en0"))
    conn = O2Connection(_config(tmp_path), runner=runner)
    with pytest.raises(O2OffVpnError):
        conn.start_master(allow_new_login=True)
    assert not any("-MNf" in call["argv"] for call in runner.calls)


def test_start_master_allows_on_vpn(tmp_path):
    # Route egresses via the VPN tunnel (utun*) -> proceed to open the master.
    runner = RecordingRunner(master=False, responder=_vpn_responder("utun6"))
    conn = O2Connection(_config(tmp_path), runner=runner)
    result = conn.start_master(allow_new_login=True)
    assert result.ok and "-MNf" in runner.calls[-1]["argv"]


def test_start_master_offvpn_override(tmp_path):
    # allow_offvpn=True bypasses the guard even on a physical interface.
    runner = RecordingRunner(master=False, responder=_vpn_responder("en0"))
    conn = O2Connection(_config(tmp_path), runner=runner)
    result = conn.start_master(allow_new_login=True, allow_offvpn=True)
    assert result.ok and "-MNf" in runner.calls[-1]["argv"]


def test_start_master_failopen_when_iface_undetermined(tmp_path):
    # If the interface can't be determined (route unavailable), proceed rather than lock out.
    runner = RecordingRunner(master=False, responder=_vpn_responder(None))
    conn = O2Connection(_config(tmp_path), runner=runner)
    result = conn.start_master(allow_new_login=True)
    assert result.ok and "-MNf" in runner.calls[-1]["argv"]


def test_start_master_guard_disabled_via_config(tmp_path):
    # O2_REQUIRE_VPN=0 (config.require_vpn=False) disables the guard entirely.
    config = _config(tmp_path)
    config.require_vpn = False
    runner = RecordingRunner(master=False, responder=_vpn_responder("en0"))
    conn = O2Connection(config, runner=runner)
    result = conn.start_master(allow_new_login=True)
    assert result.ok and "-MNf" in runner.calls[-1]["argv"]
    # Guard disabled -> no egress probing at all.
    assert not any(call["argv"][:2] == ["route", "get"] for call in runner.calls)


def test_o2config_vpn_fields_appended_last():
    # The VPN fields must come AFTER the existing public fields so a positional
    # O2Config(...) caller isn't silently shifted (e.g. default_user -> require_vpn).
    from dataclasses import fields

    names = [f.name for f in fields(O2Config)]
    assert names.index("require_vpn") > names.index("default_log_dir")
    assert names.index("vpn_iface_prefix") > names.index("default_user")


# --- Slurm submit/monitor ----------------------------------------------------
def test_submit_parses_job_id(tmp_path):
    runner = RecordingRunner(master=True, responder=lambda argv, _i: ("Submitted batch job 38874784\n", "", 0))
    slurm = O2Slurm(O2Connection(_config(tmp_path), runner=runner))
    res = slurm.submit("/home/jiz947/jobs/run.sbatch", sbatch_args=["--time=02:00:00"])
    assert res.submitted and res.job_id == "38874784"
    cmd = runner.remote_commands[-1]
    assert cmd.startswith("sbatch ")
    assert "--time=02:00:00" in cmd
    assert "/home/jiz947/jobs/run.sbatch" in cmd


def test_submit_text_stages_then_submits(tmp_path):
    runner = RecordingRunner(master=True, responder=lambda argv, _i: ("Submitted batch job 5\n", "", 0))
    slurm = O2Slurm(O2Connection(_config(tmp_path), runner=runner))
    res = slurm.submit_text("#!/bin/bash\n#SBATCH -t 1:00\nsrun hostname\n", "/scratch/jobs/x.sbatch")
    assert res.submitted and res.job_id == "5"
    # First remote command stages the file (cat > path) with the script on stdin.
    stage_call = [c for c in runner.calls if c["input"] is not None][0]
    assert "cat >" in stage_call["argv"][-1]
    assert stage_call["input"].startswith("#!/bin/bash")
    assert runner.remote_commands[-1].startswith("sbatch ")


def test_submit_reports_failure_when_no_job_id(tmp_path):
    runner = RecordingRunner(master=True, responder=lambda argv, _i: ("", "sbatch: error: invalid partition\n", 1))
    slurm = O2Slurm(O2Connection(_config(tmp_path), runner=runner))
    res = slurm.submit("/home/jiz947/jobs/run.sbatch")
    assert res.submitted is False and res.job_id is None


def test_squeue_parsing(tmp_path):
    out = (
        "38874784|clock_grid|RUNNING|01:23:45|08:00:00|1|node042\n"
        "38874785|clock_nuc|PENDING|0:00|5-00:00:00|1|(Priority)\n"
    )
    runner = RecordingRunner(master=True, responder=lambda argv, _i: (out, "", 0))
    slurm = O2Slurm(O2Connection(_config(tmp_path), runner=runner))
    jobs = slurm.queue("jiz947")
    assert len(jobs) == 2
    assert jobs[0] == {
        "job_id": "38874784",
        "name": "clock_grid",
        "state": "RUNNING",
        "elapsed": "01:23:45",
        "time_limit": "08:00:00",
        "nodes": "1",
        "reason": "node042",
    }
    assert jobs[1]["state"] == "PENDING" and jobs[1]["reason"] == "(Priority)"
    assert "squeue -u jiz947" in runner.remote_commands[-1]


def test_job_status_parsing(tmp_path):
    out = "38874784|clock_grid|COMPLETED|01:23:45|0:0|||2026-06-12T10:00:00|2026-06-12T11:23:45|node042\n"
    runner = RecordingRunner(master=True, responder=lambda argv, _i: (out, "", 0))
    slurm = O2Slurm(O2Connection(_config(tmp_path), runner=runner))
    rows = slurm.job_status("38874784")
    assert rows[0]["state"] == "COMPLETED" and rows[0]["exit_code"] == "0:0"
    assert "sacct -j 38874784" in runner.remote_commands[-1]


def test_tail_and_cancel(tmp_path):
    runner = RecordingRunner(master=True, responder=lambda argv, _i: ("...log tail...\n", "", 0))
    slurm = O2Slurm(O2Connection(_config(tmp_path), runner=runner))
    slurm.tail_log("~/logs/myproject/clock_grid_38874784.out", lines=50)
    assert runner.remote_commands[-1] == "tail -n 50 ~/logs/myproject/clock_grid_38874784.out"
    slurm.cancel("38874784")
    assert runner.remote_commands[-1] == "scancel 38874784"


# --- rsync transfers ---------------------------------------------------------
def test_push_pull_build_rsync(tmp_path):
    runner = RecordingRunner(master=True)
    sync = O2Sync(O2Connection(_config(tmp_path), runner=runner))
    sync.push("./local/run.sbatch", "/scratch/jobs/run.sbatch")
    argv = runner.calls[-1]["argv"]
    assert argv[0] == "rsync"
    assert "-e" in argv
    e_opt = argv[argv.index("-e") + 1]
    assert e_opt.startswith("ssh ") and "BatchMode=yes" in e_opt
    assert argv[-2] == "./local/run.sbatch"
    assert argv[-1] == "o2:/scratch/jobs/run.sbatch"

    sync.pull("/scratch/out/results", "./local/results", transfer=True)
    argv = runner.calls[-1]["argv"]
    assert argv[-2] == "o2-transfer:/scratch/out/results"
    assert argv[-1] == "./local/results"


def test_remote_path_with_spaces_is_escaped(tmp_path):
    # rsync hands the post-colon path to a remote shell; an unescaped space-bearing
    # path (".../20260329 - 20nm GEM Human Mouse PSM/Human") gets word-split there
    # and truncated at the first space. Spaces must be backslash-escaped.
    runner = RecordingRunner(master=True)
    sync = O2Sync(O2Connection(_config(tmp_path), runner=runner))
    remote = "/n/groups/tabin/jzhao/o2_gem_diffusion/data/20260329 - 20nm GEM Human Mouse PSM/Human"
    sync.push("/local/Human", remote)
    argv = runner.calls[-1]["argv"]
    assert argv[-2] == "/local/Human"  # local side is an argv token: never shell-split
    # every space is backslash-escaped (so the remote shell treats it as literal, not a split)
    assert argv[-1] == "o2:" + remote.replace(" ", "\\ ")
    assert " " not in argv[-1].replace("\\ ", "")  # no UNescaped space remains


def test_remote_path_preserves_tilde_and_vars(tmp_path):
    # ~, $VAR and ${VAR} stay bare so the remote shell still expands them.
    runner = RecordingRunner(master=True)
    sync = O2Sync(O2Connection(_config(tmp_path), runner=runner))
    assert sync.push_argv("a", "~/jobs/run.sbatch")[-1] == "o2:~/jobs/run.sbatch"
    assert sync.push_argv("a", "$SCRATCH/out")[-1] == "o2:$SCRATCH/out"
    assert sync.push_argv("a", "${SCRATCH}/out")[-1] == "o2:${SCRATCH}/out"  # braced var preserved
    assert sync.push_argv("a", "$SCRATCH/my out")[-1] == "o2:$SCRATCH/my\\ out"  # spaces still escaped
    assert sync.push_argv("a", "$(whoami)/x")[-1] == "o2:$\\(whoami\\)/x"  # () escaped -> no command substitution


def test_escape_is_noop_for_plain_paths(tmp_path):
    # Space-free paths must be byte-for-byte unchanged (no behavior change, no stray escapes).
    runner = RecordingRunner(master=True)
    sync = O2Sync(O2Connection(_config(tmp_path), runner=runner))
    assert sync.push_argv("a", "/n/groups/tabin/jzhao/runs/foo")[-1] == "o2:/n/groups/tabin/jzhao/runs/foo"
    # push_argv builds exactly what push() runs.
    sync.push("a", "/n/groups/tabin/jzhao/runs/foo")
    assert runner.calls[-1]["argv"] == sync.push_argv("a", "/n/groups/tabin/jzhao/runs/foo")


def test_transfer_uses_the_transfer_alias_master(tmp_path):
    # The login master is up but the transfer-node master is NOT. A normal transfer
    # (login alias) proceeds; a transfer-node transfer must refuse rather than let
    # rsync open a fresh Duo-pushing login to o2-transfer.
    def runner(argv, timeout, input_text):
        if "-O" in argv and "check" in argv:
            return CommandResult(list(argv), 0 if argv[-1] == "o2" else 255, "", "")
        return CommandResult(list(argv), 0, "", "")

    sync = O2Sync(O2Connection(_config(tmp_path), runner=runner))
    sync.push("a", "b")  # login alias master is up -> ok
    with pytest.raises(O2MasterUnavailableError):
        sync.push("a", "b", transfer=True)  # transfer alias master is down -> refuse


def test_run_raw_infers_target_alias_from_argv(tmp_path):
    # Even without master_alias, run_raw must check the alias the command targets
    # (inferred from an <alias>:path rsync target), not always the login alias.
    def runner(argv, timeout, input_text):
        if "-O" in argv and "check" in argv:
            return CommandResult(list(argv), 0 if argv[-1] == "o2" else 255, "", "")
        return CommandResult(list(argv), 0, "", "")

    conn = O2Connection(_config(tmp_path), runner=runner)
    conn.run_raw(["rsync", "-e", "ssh", "x", "o2:/p"])  # login master up -> ok
    with pytest.raises(O2MasterUnavailableError):
        conn.run_raw(["rsync", "-e", "ssh", "x", "o2-transfer:/p"])  # transfer master down -> refuse
    # a bare ssh to the transfer node is inferred too
    with pytest.raises(O2MasterUnavailableError):
        conn.run_raw(["ssh", "o2-transfer", "ls"])


def test_rsync_blocked_by_lock(tmp_path):
    runner = RecordingRunner()
    sync = O2Sync(O2Connection(_config(tmp_path, locked=True), runner=runner))
    with pytest.raises(O2LockedError):
        sync.push("a", "b")
    assert runner.calls == []


def test_o2_core_is_dependency_light():
    """Importing the o2 core must stay stdlib-only (mcp/numpy live only in the server)."""
    import subprocess
    import sys

    code = (
        "import importlib, sys\n"
        "importlib.import_module('o2mcp')\n"
        "bad = [m for m in ('mcp', 'numpy', 'torch', 'pandas') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


# --- keepalive (must never open a new login) ---------------------------------
def _patch_keepalive(monkeypatch, conn):
    monkeypatch.setattr(o2keepalive, "O2Connection", lambda config=None: conn)


def test_keepalive_skips_when_locked(tmp_path, monkeypatch):
    runner = RecordingRunner()
    _patch_keepalive(monkeypatch, O2Connection(_config(tmp_path, locked=True), runner=runner))
    assert o2keepalive.keepalive() == {"action": "skipped", "reason": "locked"}
    assert runner.calls == []  # never touched ssh


def test_keepalive_skips_when_no_master(tmp_path, monkeypatch):
    runner = RecordingRunner(master=False)
    _patch_keepalive(monkeypatch, O2Connection(_config(tmp_path), runner=runner))
    out = o2keepalive.keepalive()
    assert out["action"] == "skipped" and out["reason"] == "no_master"
    # It probed the master socket but NEVER ran a remote command (no new login).
    assert "true" not in runner.remote_commands


def test_keepalive_pings_existing_master(tmp_path, monkeypatch):
    runner = RecordingRunner(master=True)
    _patch_keepalive(monkeypatch, O2Connection(_config(tmp_path), runner=runner))
    out = o2keepalive.keepalive()
    assert out["action"] == "pinged" and out["ok"] is True
    assert runner.remote_commands[-1] == "true"  # harmless no-op resets the idle timer


def test_keepalive_clears_stale_master_on_timeout(tmp_path, monkeypatch):
    """If the ping stalls (stale master), tear it down instead of reconnecting again."""
    import subprocess as sp

    calls = []

    def runner(argv, timeout, input_text):
        calls.append(list(argv))
        if "-O" in argv and "check" in argv:
            return CommandResult(list(argv), 0, "", "")  # local master process "running"
        if "-O" in argv and "exit" in argv:
            return CommandResult(list(argv), 0, "exit sent", "")
        if argv[-1] == "true":
            raise sp.TimeoutExpired(argv, timeout)  # connection dead -> ping stalls
        return CommandResult(list(argv), 0, "", "")

    _patch_keepalive(monkeypatch, O2Connection(_config(tmp_path), runner=runner))
    out = o2keepalive.keepalive()
    assert out["action"] == "stale_master_cleared"
    assert any("-O" in c and "exit" in c for c in calls)  # tore the stale master down
