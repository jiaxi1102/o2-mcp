"""Offline unit tests for the o2mcp cluster primitives.

The subprocess seam is injected, so these never touch the network: they assert
the exact ssh/sbatch/squeue/rsync commands are built, the safety lock and
ControlMaster guards fire, and Slurm output is parsed correctly.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from o2mcp import (
    CommandResult,
    O2Config,
    O2Connection,
    O2LockedError,
    O2LoginCoordinationError,
    O2MasterUnavailableError,
    O2OffVpnError,
    O2Slurm,
    O2Sync,
    O2UnsafeTransportError,
    default_runner,
)
from o2mcp import keepalive as o2keepalive


@pytest.fixture(autouse=True)
def isolate_user_level_o2_files(monkeypatch, tmp_path):
    """Keep global-lock and login-mutex tests out of the developer's real home."""

    monkeypatch.setenv("HOME", str(tmp_path / "test-home"))


class RecordingRunner:
    """A fake subprocess runner: records calls, answers via a response function."""

    def __init__(self, *, master: bool = True, responder=None, start_persists: bool = True):
        self.calls = []
        self.master = master
        self._responder = responder
        self._start_persists = start_persists

    def __call__(self, argv, timeout, input_text) -> CommandResult:
        self.calls.append({"argv": list(argv), "timeout": timeout, "input": input_text})
        if "-O" in argv and "check" in argv:
            return CommandResult(list(argv), 0 if self.master else 255, "", "")
        if "-MNf" in argv:
            # A successful fake startup normally creates a reusable master so
            # the production postcondition is exercised. Tests can disable this
            # transition to reproduce SSH returning zero before its background
            # connection disappears.
            if self._start_persists:
                self.master = True
            return CommandResult(list(argv), 0, "", "")
        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
            # Config expansion is a local-only prerequisite, not the remote
            # command under test. Model both values production code consumes.
            out = f"hostname {argv[-1]}.example\ncontrolpath /tmp/{argv[-1]}-control.sock\n"
            return CommandResult(list(argv), 0, out, "")
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
            if (
                argv
                and argv[0] == O2Connection.SSH_EXECUTABLE
                and "-O" not in argv
                and "-MNf" not in argv
                and "-G" not in argv
            ):
                cmds.append(argv[-1])
        return cmds


class ConcurrentStartRunner:
    """Model two MCP processes racing to create the same SSH master.

    The barrier forces both callers to complete their initial no-master check
    before either can proceed. A correct interprocess guard then lets only one
    caller execute ``ssh -MNf``; the second sees the master on its guarded
    recheck and returns the existing-master result.
    """

    def __init__(self, *, start_returncode: int = 0) -> None:
        self._state_lock = threading.Lock()
        self._initial_check_barrier = threading.Barrier(2)
        self._initial_checks = 0
        self._start_returncode = start_returncode
        self.master = False
        self.start_count = 0

    def __call__(self, argv: list[str], timeout: float | None, input_text: str | None) -> CommandResult:
        """Return deterministic SSH-check/start results for the race test."""

        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
            return CommandResult(list(argv), 0, "controlpath /tmp/o2-control.sock\n", "")

        if "-O" in argv and "check" in argv:
            with self._state_lock:
                synchronize_initial_check = not self.master and self._initial_checks < 2
                if synchronize_initial_check:
                    self._initial_checks += 1
                master = self.master
            if synchronize_initial_check:
                self._initial_check_barrier.wait(timeout=5)
            return CommandResult(list(argv), 0 if master else 255, "", "")

        if "-MNf" in argv:
            with self._state_lock:
                self.start_count += 1
                if self._start_returncode == 0:
                    self.master = True
            stderr = "" if self._start_returncode == 0 else "simulated start failure"
            return CommandResult(list(argv), self._start_returncode, "", stderr)

        return CommandResult(list(argv), 0, "", "")


def _config(tmp_path: Path, *, locked: bool = False) -> O2Config:
    lock = tmp_path / "O2_DISABLED"
    if locked:
        lock.write_text("disabled")
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text(
        "Host o2\n"
        "  HostName o2.hms.harvard.edu\n"
        "  User jiz947\n"
        "  ControlPath /tmp/o2-control.sock\n"
        "Host o2-transfer\n"
        "  HostName transfer.rc.hms.harvard.edu\n"
        "  User jiz947\n"
        "  ControlPath /tmp/o2-transfer-control.sock\n"
    )
    return O2Config(
        host_alias="o2",
        transfer_alias="o2-transfer",
        connect_timeout=20,
        lock_file=lock,
        ssh_config_file=ssh_config,
    )


def test_default_lock_is_workstation_wide(monkeypatch, tmp_path):
    """Unconfigured clients must converge on one user-level emergency stop."""

    monkeypatch.delenv("O2_SSH_LOCK_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert O2Config().lock_file == tmp_path / ".agent_locks" / "O2_DISABLED"


def test_explicit_lock_path_still_overrides_global_default(monkeypatch, tmp_path):
    """Deployments may intentionally provide another shared lock location."""

    explicit = tmp_path / "shared" / "O2_DISABLED"
    monkeypatch.setenv("O2_SSH_LOCK_FILE", str(explicit))

    assert O2Config().lock_file == explicit


def test_reuse_only_options_disable_every_fallback_authentication_method(tmp_path):
    """Ordinary commands must be unable to authenticate if multiplexing fails."""

    opts = _config(tmp_path).reuse_only_ssh_opts()
    option_values = {opts[index + 1] for index, token in enumerate(opts[:-1]) if token == "-o"}

    # BatchMode alone is insufficient on O2 because successful key authentication
    # can itself trigger Duo. Disabling the key and every alternate authentication
    # mechanism ensures a missing socket produces a local SSH failure, not a login.
    assert {
        "ControlMaster=no",
        "ConnectionAttempts=1",
        "ProxyCommand=none",
        "ProxyJump=none",
        "PreferredAuthentications=none",
        "PubkeyAuthentication=no",
        "PasswordAuthentication=no",
        "KbdInteractiveAuthentication=no",
        "GSSAPIAuthentication=no",
        "HostbasedAuthentication=no",
        "NumberOfPasswordPrompts=0",
    } <= option_values


# --- subprocess stdio isolation ---------------------------------------------
def test_default_runner_uses_devnull_without_input(monkeypatch):
    """A child command must never inherit the stdio MCP server's JSON-RPC input."""

    captured = {}

    def fake_run(argv, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr("o2mcp.connection.subprocess.run", fake_run)

    result = default_runner(["ssh", "o2", "hostname"], timeout=5.0, input_text=None)

    assert result.ok
    assert captured["stdin"] is subprocess.DEVNULL
    assert "input" not in captured


def test_default_runner_pipes_explicit_input(monkeypatch):
    """Explicit staging content must use a private pipe rather than `/dev/null`."""

    captured = {}

    def fake_run(argv, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr("o2mcp.connection.subprocess.run", fake_run)

    payload = "#!/bin/bash\necho staged\n"
    result = default_runner(["ssh", "o2", "cat > job.sh"], timeout=5.0, input_text=payload)

    assert result.ok
    assert captured["input"] == payload
    assert "stdin" not in captured


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


def test_legacy_project_lock_remains_a_hard_stop(monkeypatch, tmp_path):
    """Upgrading must not bypass an existing cwd-scoped emergency lock."""

    project = tmp_path / "legacy-project"
    legacy_lock = project / ".agent_locks" / "O2_DISABLED"
    legacy_lock.parent.mkdir(parents=True)
    legacy_lock.write_text("disabled before global-lock migration")
    monkeypatch.chdir(project)

    # The configured global-style lock is intentionally absent. The connection
    # must still find the historical project lock before executing any SSH.
    config = O2Config(lock_file=tmp_path / "user-lock" / "O2_DISABLED")
    runner = RecordingRunner()
    conn = O2Connection(config, runner=runner)

    assert conn.is_locked() is True
    with pytest.raises(O2LockedError, match=str(legacy_lock)):
        conn.run("hostname")
    assert runner.calls == []


# --- ControlMaster guards ----------------------------------------------------
def test_run_requires_master(tmp_path):
    runner = RecordingRunner(master=False)
    conn = O2Connection(_config(tmp_path), runner=runner)
    assert conn.master_running() is False
    with pytest.raises(O2MasterUnavailableError):
        conn.run("squeue")


def test_master_running_treats_probe_timeout_as_unavailable(tmp_path):
    """A hung local socket check must fail closed without escaping as an error."""

    def runner(argv, timeout, input_text):
        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
            return CommandResult(list(argv), 0, "controlpath /tmp/o2-control.sock\n", "")
        raise subprocess.TimeoutExpired(argv, timeout)

    conn = O2Connection(_config(tmp_path), runner=runner)

    assert conn.master_running() is False


def test_run_raw_requires_master(tmp_path):
    # A raw transport (rsync/ssh) must also refuse without a master, or it would
    # open a fresh connection — a new Duo push — outside the one approved master.
    runner = RecordingRunner(master=False)
    conn = O2Connection(_config(tmp_path), runner=runner)
    assert conn.master_running() is False
    with pytest.raises(O2MasterUnavailableError):
        conn.run_raw(["rsync", "x", "y"])
    # The former escape hatch is retained only as an explicit failure so an old
    # caller cannot silently regain cold-connection behavior after upgrading.
    calls_before_opt_out = len(runner.calls)
    with pytest.raises(O2UnsafeTransportError, match="Cold O2 SSH/rsync execution is disabled"):
        conn.run_raw(["rsync", "x", "y"], require_master=False)
    assert len(runner.calls) == calls_before_opt_out


def test_run_uses_reuse_only_authentication_and_alias(tmp_path):
    runner = RecordingRunner(master=True)
    conn = O2Connection(_config(tmp_path), runner=runner)
    result = conn.run("hostname; whoami")
    assert result.ok
    last = runner.calls[-1]["argv"]
    assert last[0] == O2Connection.SSH_EXECUTABLE
    assert last[:5] == [O2Connection.SSH_EXECUTABLE, "-F", "/dev/null", "-S", "/tmp/o2-control.sock"]
    assert last[5 : 5 + len(conn.config.reuse_only_ssh_opts())] == conn.config.reuse_only_ssh_opts()
    assert "PreferredAuthentications=none" in last
    assert "PubkeyAuthentication=no" in last
    assert "PermitLocalCommand=no" in last
    assert "KnownHostsCommand=none" in last
    assert last[-2] == "o2"
    assert last[-1] == "hostname; whoami"


def test_run_pins_proxy_derived_control_path_before_disabling_proxy(tmp_path):
    """ProxyJump-dependent ``%C`` sockets retain their original identity."""

    class ProxyConfigRunner(RecordingRunner):
        def __call__(self, argv, timeout, input_text) -> CommandResult:
            if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
                self.calls.append({"argv": list(argv), "timeout": timeout, "input": input_text})
                return CommandResult(
                    list(argv),
                    0,
                    "proxyjump bastion.example\ncontrolpath /tmp/cm-original-jump-hash\n",
                    "",
                )
            return super().__call__(argv, timeout, input_text)

    runner = ProxyConfigRunner(master=True)
    conn = O2Connection(_config(tmp_path), runner=runner)

    result = conn.run("hostname")

    assert result.ok
    command = runner.calls[-1]["argv"]
    assert command[:5] == [
        O2Connection.SSH_EXECUTABLE,
        "-F",
        "/dev/null",
        "-S",
        "/tmp/cm-original-jump-hash",
    ]
    assert "ProxyJump=none" in command and "ProxyCommand=none" in command


def test_match_ssh_config_is_rejected_before_openssh_runs(tmp_path):
    """Socket expansion must never execute a Match predicate's shell command."""

    config = _config(tmp_path)
    config.ssh_config_file.write_text(
        "Host o2\n"
        "  HostName o2.hms.harvard.edu\n"
        "  User jiz947\n"
        "  ControlPath /tmp/o2-control.sock\n"
        'Match=exec "ssh unsafe-probe.example true"\n'
        "  ServerAliveInterval 30\n"
    )
    runner = RecordingRunner(master=True)
    conn = O2Connection(config, runner=runner)

    with pytest.raises(O2UnsafeTransportError, match="contains Match"):
        conn.run("hostname")

    # The rejection comes from literal file inspection. In particular, not even
    # `ssh -G` is launched, because that command evaluates Match exec itself.
    assert runner.calls == []


def test_match_in_included_ssh_config_is_rejected(tmp_path):
    """An Include cannot hide an executable Match block from the safety scan."""

    config = _config(tmp_path)
    included = tmp_path / "o2-extra.conf"
    included.write_text('Match exec "ssh unsafe-probe.example true"\n  ServerAliveInterval 30\n')
    config.ssh_config_file.write_text(
        f"Include={included}\n"
        "Host o2\n"
        "  HostName o2.hms.harvard.edu\n"
        "  User jiz947\n"
        "  ControlPath /tmp/o2-control.sock\n"
    )
    runner = RecordingRunner(master=True)

    with pytest.raises(O2UnsafeTransportError, match="contains Match"):
        O2Connection(config, runner=runner).run("hostname")

    assert runner.calls == []


def test_run_rejects_cold_connection_opt_out(tmp_path):
    """No library caller may bypass the master-only contract."""

    runner = RecordingRunner(master=True)
    conn = O2Connection(_config(tmp_path), runner=runner)

    with pytest.raises(O2UnsafeTransportError, match="Cold O2 SSH execution is disabled"):
        conn.run("hostname", require_master=False)

    assert runner.calls == []


def test_start_master_requires_opt_in(tmp_path):
    runner = RecordingRunner(master=False)
    conn = O2Connection(_config(tmp_path), runner=runner)
    with pytest.raises(O2MasterUnavailableError):
        conn.start_master(allow_new_login=False)
    # Opt-in opens the master with -MNf.
    result = conn.start_master(allow_new_login=True)
    assert result.ok
    starts = [call["argv"] for call in runner.calls if "-MNf" in call["argv"]]
    assert len(starts) == 1
    # The explicitly authorized start is the only path allowed to authenticate;
    # reuse-only options would make creation of the initial master impossible.
    assert "PubkeyAuthentication=no" not in starts[0]
    assert "PreferredAuthentications=none" not in starts[0]
    assert runner.calls[-1]["argv"] == conn._master_check_argv("o2")


def test_start_master_rejects_zero_exit_when_control_socket_disappears(tmp_path):
    """A zero exit from ``ssh -MNf`` is not success without a live socket."""

    runner = RecordingRunner(master=False, start_persists=False)
    conn = O2Connection(_config(tmp_path), runner=runner)

    result = conn.start_master(allow_new_login=True)

    assert result.ok is False
    assert result.returncode == 255
    assert result.argv == conn._master_check_argv("o2")
    assert result.stdout == ""
    assert "post-start control-socket check failed" in result.stderr
    assert sum("-MNf" in call["argv"] for call in runner.calls) == 1

    # The failed postcondition is retained as a workstation-wide cooldown
    # receipt. A second caller must fail closed before another Duo-pushing SSH
    # process can start.
    receipt = json.loads(conn._master_start_attempt_file().read_text())
    assert receipt["target"] == "o2"
    assert receipt["returncode"] == 255
    with pytest.raises(O2LoginCoordinationError, match="refusing another Duo-pushing login"):
        conn.start_master(allow_new_login=True)
    assert sum("-MNf" in call["argv"] for call in runner.calls) == 1


def test_start_master_records_post_start_verification_timeout(tmp_path):
    """A timed-out socket postcondition retains an actionable cooldown receipt."""

    class VerificationTimeoutRunner(RecordingRunner):
        def __call__(self, argv, timeout, input_text) -> CommandResult:
            if "-O" in argv and "check" in argv and self.master:
                self.calls.append({"argv": list(argv), "timeout": timeout, "input": input_text})
                raise subprocess.TimeoutExpired(argv, timeout)
            return super().__call__(argv, timeout, input_text)

    runner = VerificationTimeoutRunner(master=False)
    conn = O2Connection(_config(tmp_path), runner=runner)

    result = conn.start_master(allow_new_login=True)

    assert result.ok is False and result.returncode == 255
    assert "TimeoutExpired" in result.stderr
    receipt = json.loads(conn._master_start_attempt_file().read_text())
    assert receipt["returncode"] == 255


def test_start_master_noop_when_running(tmp_path):
    conn = O2Connection(_config(tmp_path), runner=RecordingRunner(master=True))
    result = conn.start_master(allow_new_login=False)
    assert result.ok and "already running" in result.stdout
    assert result.argv == [
        O2Connection.SSH_EXECUTABLE,
        "-F",
        "/dev/null",
        "-S",
        "/tmp/o2-control.sock",
        *conn.config.base_ssh_opts(),
        "-O",
        "check",
        "o2",
    ]


def test_start_master_can_open_the_transfer_alias(tmp_path):
    # The transfer node has its own master; start_master(alias=...) must be able to
    # open it so transfer-node rsync/ssh has a socket to reuse (otherwise the new
    # transfer-master guard could never be satisfied).
    runner = RecordingRunner(master=False)
    conn = O2Connection(_config(tmp_path), runner=runner)
    result = conn.start_master(allow_new_login=True, alias=conn.config.transfer_alias)
    assert result.ok
    starts = [call for call in runner.calls if "-MNf" in call["argv"]]
    assert len(starts) == 1 and starts[0]["argv"][-1] == "o2-transfer"
    assert runner.calls[-1]["argv"] == conn._master_check_argv("o2-transfer")


def test_concurrent_master_starts_execute_one_login(tmp_path):
    """Even legacy project configs must share one O2/Duo login mutex."""

    runner = ConcurrentStartRunner()
    ssh_config_file = _config(tmp_path).ssh_config_file
    configs = [
        O2Config(
            host_alias="o2",
            lock_file=tmp_path / project / "O2_DISABLED",
            ssh_config_file=ssh_config_file,
        )
        for project in ("clock-project", "diffusion-project")
    ]
    connections = [O2Connection(config, runner=runner) for config in configs]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda conn: conn.start_master(allow_new_login=True), connections))

    assert runner.start_count == 1
    assert all(result.ok for result in results)
    assert sum("already running" in result.stdout for result in results) == 1
    reused = next(result for result in results if "already running" in result.stdout)
    assert reused.argv == [
        O2Connection.SSH_EXECUTABLE,
        "-F",
        "/dev/null",
        "-S",
        "/tmp/o2-control.sock",
        *configs[0].base_ssh_opts(),
        "-O",
        "check",
        "o2",
    ]

    mutex_paths = {conn._master_start_lock_file() for conn in connections}
    assert mutex_paths == {tmp_path / "test-home" / ".agent_locks" / "O2_LOGIN_START.lock"}
    assert not connections[0]._master_start_attempt_file().exists()


def test_concurrent_callers_do_not_retry_a_failed_login(tmp_path):
    """One failed SSH start must put every queued contender into cooldown."""

    runner = ConcurrentStartRunner(start_returncode=255)
    ssh_config_file = _config(tmp_path).ssh_config_file
    configs = [
        O2Config(
            host_alias="o2",
            lock_file=tmp_path / project / "O2_DISABLED",
            ssh_config_file=ssh_config_file,
        )
        for project in ("clock-project", "diffusion-project")
    ]
    connections = [O2Connection(config, runner=runner) for config in configs]

    def invoke(conn):
        """Return either the first command result or the queued caller's guard error."""

        try:
            return conn.start_master(allow_new_login=True)
        except O2LoginCoordinationError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(invoke, connections))

    command_results = [outcome for outcome in outcomes if isinstance(outcome, CommandResult)]
    guard_errors = [outcome for outcome in outcomes if isinstance(outcome, O2LoginCoordinationError)]
    assert runner.start_count == 1
    assert len(command_results) == 1 and command_results[0].returncode == 255
    assert len(guard_errors) == 1 and "refusing another Duo-pushing login" in str(guard_errors[0])

    receipt = json.loads(connections[0]._master_start_attempt_file().read_text())
    assert receipt["returncode"] == 255


def test_expired_failed_attempt_allows_one_new_login(tmp_path):
    """The shared failure receipt must not block intentional retries forever."""

    runner = RecordingRunner(master=False)
    conn = O2Connection(_config(tmp_path), runner=runner)
    receipt_path = conn._master_start_attempt_file()
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(
            {
                "started_at": time.time() - conn.LOGIN_RETRY_COOLDOWN_SECONDS - 1.0,
                "pid": 1,
                "target": "o2",
                "returncode": 255,
            }
        )
    )

    result = conn.start_master(allow_new_login=True)

    assert result.ok
    assert sum("-MNf" in call["argv"] for call in runner.calls) == 1
    assert not receipt_path.exists()


def test_timed_out_login_leaves_cooldown_for_the_next_caller(tmp_path):
    """An SSH exception must not let the following task generate another Duo call."""

    class TimeoutStartRunner(RecordingRunner):
        """Raise exactly where a real SSH master start can time out."""

        def __call__(self, argv, timeout, input_text) -> CommandResult:
            if "-MNf" in argv:
                self.calls.append({"argv": list(argv), "timeout": timeout, "input": input_text})
                raise subprocess.TimeoutExpired(argv, timeout)
            return super().__call__(argv, timeout, input_text)

    runner = TimeoutStartRunner(master=False)
    conn = O2Connection(_config(tmp_path), runner=runner)

    with pytest.raises(subprocess.TimeoutExpired):
        conn.start_master(allow_new_login=True)
    with pytest.raises(O2LoginCoordinationError, match="refusing another Duo-pushing login"):
        conn.start_master(allow_new_login=True)

    assert sum("-MNf" in call["argv"] for call in runner.calls) == 1
    receipt = json.loads(conn._master_start_attempt_file().read_text())
    assert receipt["returncode"] is None


def test_master_start_fails_closed_when_coordination_lock_cannot_be_created(monkeypatch, tmp_path):
    """A broken mutex location must never fall back to an uncoordinated login."""

    non_directory = tmp_path / "not-a-directory"
    non_directory.write_text("occupied")
    config = _config(tmp_path)
    runner = RecordingRunner(master=False)
    conn = O2Connection(config, runner=runner)
    monkeypatch.setattr(conn, "_master_start_lock_file", lambda: non_directory / "O2_LOGIN_START.lock")

    with pytest.raises(O2LoginCoordinationError) as exc_info:
        conn.start_master(allow_new_login=True)

    assert "OS error:" in str(exc_info.value)
    assert str(non_directory) in str(exc_info.value)
    assert not any("-MNf" in call["argv"] for call in runner.calls)


# --- VPN egress guard (HMS O2 Duos non-HMS source IPs) -----------------------
def _vpn_responder(interface):
    """Responder answering the guard's `ssh -G` (HostName) and `route get` (interface)."""

    def responder(argv, input_text):
        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
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
    assert result.ok and any("-MNf" in call["argv"] for call in runner.calls)


def test_start_master_offvpn_override(tmp_path):
    # allow_offvpn=True bypasses the guard even on a physical interface.
    runner = RecordingRunner(master=False, responder=_vpn_responder("en0"))
    conn = O2Connection(_config(tmp_path), runner=runner)
    result = conn.start_master(allow_new_login=True, allow_offvpn=True)
    assert result.ok and any("-MNf" in call["argv"] for call in runner.calls)


def test_start_master_failopen_when_iface_undetermined(tmp_path):
    # If the interface can't be determined (route unavailable), proceed rather than lock out.
    runner = RecordingRunner(master=False, responder=_vpn_responder(None))
    conn = O2Connection(_config(tmp_path), runner=runner)
    result = conn.start_master(allow_new_login=True)
    assert result.ok and any("-MNf" in call["argv"] for call in runner.calls)


def test_start_master_failopen_when_route_binary_missing(tmp_path):
    # A missing `route` binary makes the runner raise FileNotFoundError (exactly as
    # subprocess.run does) BEFORE any CommandResult.ok check — the probe must swallow it
    # and fail OPEN, not propagate the error and block an otherwise-legitimate login.
    def responder(argv, _input):
        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
            return ("hostname o2.hms.harvard.edu\n", "", 0)
        if argv[:2] == ["route", "get"]:
            raise FileNotFoundError(2, "No such file or directory", "route")
        return ("", "", 0)

    runner = RecordingRunner(master=False, responder=responder)
    conn = O2Connection(_config(tmp_path), runner=runner)
    result = conn.start_master(allow_new_login=True)
    assert result.ok and any("-MNf" in call["argv"] for call in runner.calls)


def test_start_master_guard_disabled_via_config(tmp_path):
    # O2_REQUIRE_VPN=0 (config.require_vpn=False) disables the guard entirely.
    config = _config(tmp_path)
    config.require_vpn = False
    runner = RecordingRunner(master=False, responder=_vpn_responder("en0"))
    conn = O2Connection(config, runner=runner)
    result = conn.start_master(allow_new_login=True)
    assert result.ok and any("-MNf" in call["argv"] for call in runner.calls)
    # Guard disabled -> no egress probing at all.
    assert not any(call["argv"][:2] == ["route", "get"] for call in runner.calls)


def test_o2config_new_fields_are_appended_for_positional_compatibility():
    # Safety fields must come AFTER the original public fields so a positional
    # O2Config(...) caller isn't silently shifted (e.g. default_user -> require_vpn).
    from dataclasses import fields

    names = [f.name for f in fields(O2Config)]
    assert names.index("require_vpn") > names.index("default_log_dir")
    assert names.index("vpn_iface_prefix") > names.index("default_user")
    assert names.index("ssh_config_file") > names.index("vpn_iface_prefix")


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
    assert argv[0] == O2Connection.RSYNC_EXECUTABLE
    assert "-e" in argv
    e_opt = argv[argv.index("-e") + 1]
    assert e_opt.startswith(f"{O2Connection.SSH_EXECUTABLE} ") and "BatchMode=yes" in e_opt
    assert "PreferredAuthentications=none" in e_opt
    assert "PubkeyAuthentication=no" in e_opt
    assert argv[-2] == "./local/run.sbatch"
    assert argv[-1] == "o2:/scratch/jobs/run.sbatch"

    sync.pull("/scratch/out/results", "./local/results", transfer=True)
    argv = runner.calls[-1]["argv"]
    assert argv[-2] == "o2-transfer:/scratch/out/results"
    assert argv[-1] == "./local/results"

    # Programmatic builder callers may use rsync's user-qualified endpoint.
    # Preserve that user when selecting the `%r`-dependent ControlPath.
    argv = sync._build_rsync(source="./local/results", dest="alice@o2-transfer:/scratch/results", extra_args=None)
    transport = argv[argv.index("-e") + 1]
    assert "-S /tmp/alice@o2-transfer-control.sock" in transport


def test_run_raw_hardens_direct_ssh_and_permissive_rsync(tmp_path):
    """Library callers cannot smuggle an authentication-capable raw transport."""

    runner = RecordingRunner(master=True)
    conn = O2Connection(_config(tmp_path), runner=runner)

    conn.run_raw(["ssh", "o2", "hostname"])
    direct = runner.calls[-1]["argv"]
    assert direct[:5] == [O2Connection.SSH_EXECUTABLE, "-F", "/dev/null", "-S", "/tmp/o2-control.sock"]
    assert direct[5 : 5 + len(conn.config.reuse_only_ssh_opts())] == conn.config.reuse_only_ssh_opts()

    # A later raw ProxyJump request cannot win over the earlier command-line
    # `ProxyJump=none`. Otherwise the child proxy SSH could authenticate even
    # though the outer O2 client has every authentication method disabled.
    conn.run_raw(["ssh", "-J", "proxy.example", "o2", "hostname"])
    direct = runner.calls[-1]["argv"]
    assert direct.index("ProxyJump=none") < direct.index("-J")

    # Caller-selected config/socket paths are stripped, including after another
    # option with a separate argument. The guarded /dev/null config and resolved
    # socket remain the only effective values.
    conn.run_raw(
        [
            "ssh",
            "-c",
            "aes128-ctr",
            "-F",
            "/tmp/unsafe-config",
            "-S",
            "/tmp/unsafe-socket",
            "-o",
            "ControlPath=/tmp/also-unsafe",
            "o2",
            "hostname",
        ]
    )
    direct = runner.calls[-1]["argv"]
    assert direct[:5] == [O2Connection.SSH_EXECUTABLE, "-F", "/dev/null", "-S", "/tmp/o2-control.sock"]
    assert "/tmp/unsafe-config" not in direct
    assert "/tmp/unsafe-socket" not in direct
    assert "ControlPath=/tmp/also-unsafe" not in direct

    conn.run_raw(
        [
            "rsync",
            "-e",
            "ssh -o PubkeyAuthentication=yes -o ProxyCommand='ssh proxy.example'",
            "x",
            "o2:/p",
        ]
    )
    rsync = runner.calls[-1]["argv"]
    transport = rsync[rsync.index("-e") + 1]
    # The safe option is prepended, and OpenSSH honors the first command-line
    # value, so the caller's later attempt to re-enable keys is ineffective.
    assert transport.index("PubkeyAuthentication=no") < transport.index("PubkeyAuthentication=yes")
    assert "PreferredAuthentications=none" in transport
    assert transport.index("ProxyCommand=none") < transport.index("ProxyCommand=ssh proxy.example")

    conn.run_raw(
        [
            "rsync",
            "-e",
            "ssh -F /tmp/unsafe-config -S /tmp/unsafe-socket -o ControlPath=/tmp/also-unsafe",
            "x",
            "o2:/p",
        ]
    )
    rsync = runner.calls[-1]["argv"]
    transport = rsync[rsync.index("-e") + 1]
    assert "-F /dev/null" in transport
    assert "-S /tmp/o2-control.sock" in transport
    assert "/tmp/unsafe-config" not in transport
    assert "/tmp/unsafe-socket" not in transport
    assert "/tmp/also-unsafe" not in transport
    assert "PreferredAuthentications=none" in transport

    # Every remote-shell option is normalized, not just the first. This prevents
    # a later compact/long-form override from restoring a cold-login path.
    conn.run_raw(["rsync", "-e", "ssh", "-essh -o PubkeyAuthentication=yes", "x", "o2:/p"])
    rsync = runner.calls[-1]["argv"]
    transports = [
        rsync[rsync.index("-e") + 1],
        next(token[2:] for token in rsync if token.startswith("-e") and token != "-e"),
    ]
    assert all("PubkeyAuthentication=no" in transport for transport in transports)
    assert all("PreferredAuthentications=none" in transport for transport in transports)

    conn.run_raw(["rsync", "-avze", "ssh -o PubkeyAuthentication=yes", "x", "o2:/p"])
    clustered = runner.calls[-1]["argv"]
    transport = clustered[clustered.index("-avze") + 1]
    assert transport.index("PubkeyAuthentication=no") < transport.index("PubkeyAuthentication=yes")

    conn.run_raw(["rsync", "-avzessh -o PubkeyAuthentication=yes", "x", "o2:/p"])
    attached = runner.calls[-1]["argv"]
    transport = next(token[len("-avze") :] for token in attached if token.startswith("-avze"))
    assert transport.index("PubkeyAuthentication=no") < transport.index("PubkeyAuthentication=yes")

    # Rsync options that consume arguments end a short-option cluster. The
    # letters inside those arguments are data, so an ``e`` in --fake-super or a
    # temp path must not be mistaken for another remote-shell option.
    conn.run_raw(["rsync", "-M--fake-super", "-T/tmp/cache", "x", "o2:/p"])
    argument_clusters = runner.calls[-1]["argv"]
    assert "-M--fake-super" in argument_clusters
    assert "-T/tmp/cache" in argument_clusters
    inserted_transport = argument_clusters[argument_clusters.index("-e") + 1]
    assert "PreferredAuthentications=none" in inserted_transport

    # The same rule applies when a non-transport option takes its argument from
    # the next argv element. Even an argument literally named ``-e`` remains an
    # argument to ``-M``; the guard inserts its own separate hardened transport.
    conn.run_raw(["rsync", "-M", "-e", "x", "o2:/p"])
    separated_argument = runner.calls[-1]["argv"]
    assert separated_argument[separated_argument.index("-M") + 1] == "-e"
    assert separated_argument[1] == "-e"
    assert "PreferredAuthentications=none" in separated_argument[2]

    # Long options can also consume the next argv element. A pattern named
    # ``-e`` is valid data for --exclude, not a second remote-shell declaration.
    conn.run_raw(["rsync", "--exclude", "-e", "x", "o2:/p"])
    long_argument = runner.calls[-1]["argv"]
    exclude_index = long_argument.index("--exclude")
    assert long_argument[exclude_index + 1] == "-e"
    assert long_argument[1] == "-e"
    assert "PreferredAuthentications=none" in long_argument[2]

    # An equals-attached long-option value stays inside its own token and does
    # not hide the following real remote-shell option from normalization.
    conn.run_raw(["rsync", "--exclude=-e", "-e", "ssh", "x", "o2:/p"])
    attached_long_argument = runner.calls[-1]["argv"]
    assert "--exclude=-e" in attached_long_argument
    explicit_transport = attached_long_argument[attached_long_argument.index("-e") + 1]
    assert "PreferredAuthentications=none" in explicit_transport

    # Rsync also accepts ``-o=argument``. Preserve the equals sign while
    # hardening the attached remote shell.
    conn.run_raw(["rsync", "-avze=ssh -o PubkeyAuthentication=yes", "x", "o2:/p"])
    equals_cluster = runner.calls[-1]["argv"]
    equals_transport = next(token[len("-avze=") :] for token in equals_cluster if token.startswith("-avze="))
    assert equals_transport.index("PubkeyAuthentication=no") < equals_transport.index("PubkeyAuthentication=yes")


@pytest.mark.parametrize(
    "argv",
    [
        ["ssh", "-l", "alice", "o2", "hostname"],
        ["ssh", "-o", "User=alice", "o2", "hostname"],
        ["ssh", "-p", "2222", "o2", "hostname"],
        ["ssh", "-p2222", "o2", "hostname"],
        ["ssh", "-o", "Port=2222", "o2", "hostname"],
        ["ssh", "-o", "HostName=other.example", "o2", "hostname"],
        ["rsync", "-e", "ssh -l alice", "x", "o2:/p"],
        ["rsync", "-e", "ssh -o User=alice", "x", "o2:/p"],
        ["rsync", "-e", "ssh -p 2222", "x", "o2:/p"],
        ["rsync", "-e", "ssh -o Port=2222", "x", "o2:/p"],
    ],
)
def test_run_raw_rejects_ssh_endpoint_identity_options(tmp_path, argv):
    """Endpoint identity must not change after the exact socket is resolved."""

    runner = RecordingRunner(master=True)
    conn = O2Connection(_config(tmp_path), runner=runner)

    with pytest.raises(O2UnsafeTransportError, match="user options|port options|hostname options"):
        conn.run_raw(argv)

    assert not any(
        call["argv"]
        and call["argv"][0] in {O2Connection.SSH_EXECUTABLE, O2Connection.RSYNC_EXECUTABLE}
        and "-O" not in call["argv"]
        and "-G" not in call["argv"]
        for call in runner.calls
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["rsync", "-e", "custom-rsh", "x", "o2:/p"],
        ["rsync", "-e", "/tmp/ssh", "x", "o2:/p"],
        ["/tmp/ssh", "o2", "hostname"],
        ["/tmp/rsync", "-e", "ssh", "x", "o2:/p"],
    ],
)
def test_run_raw_rejects_untrusted_transport_executables(tmp_path, argv):
    """A wrapper with a trusted-looking basename must not bypass hardening."""

    runner = RecordingRunner(master=True)
    conn = O2Connection(_config(tmp_path), runner=runner)

    with pytest.raises(O2UnsafeTransportError, match="trusted|arbitrary executable paths"):
        conn.run_raw(argv)

    assert runner.calls == []


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
        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
            return CommandResult(list(argv), 0, f"controlpath /tmp/{argv[-1]}-control.sock\n", "")
        return CommandResult(list(argv), 0, "", "")

    sync = O2Sync(O2Connection(_config(tmp_path), runner=runner))
    sync.push("a", "b")  # login alias master is up -> ok
    with pytest.raises(O2MasterUnavailableError):
        sync.push("a", "b", transfer=True)  # transfer alias master is down -> refuse


def test_run_raw_infers_target_alias_from_argv(tmp_path):
    # Even without master_alias, run_raw must check the alias the command targets
    # (inferred from a [user@]<alias>:path rsync target), not always the login alias.
    def runner(argv, timeout, input_text):
        if "-O" in argv and "check" in argv:
            return CommandResult(list(argv), 0 if argv[-1] == "o2" else 255, "", "")
        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
            return CommandResult(list(argv), 0, f"controlpath /tmp/{argv[-1]}-control.sock\n", "")
        return CommandResult(list(argv), 0, "", "")

    conn = O2Connection(_config(tmp_path), runner=runner)
    conn.run_raw(["rsync", "-e", "ssh", "x", "o2:/p"])  # login master up -> ok
    with pytest.raises(O2MasterUnavailableError):
        conn.run_raw(["rsync", "-e", "ssh", "x", "o2-transfer:/p"])  # transfer master down -> refuse
    # a bare ssh to the transfer node is inferred too
    with pytest.raises(O2MasterUnavailableError):
        conn.run_raw(["ssh", "o2-transfer", "ls"])
    # Standard user-qualified rsync and SSH destinations must select the same
    # transfer-node socket rather than falling back to the login alias.
    with pytest.raises(O2MasterUnavailableError):
        conn.run_raw(["rsync", "-e", "ssh", "x", "alice@o2-transfer:/p"])
    with pytest.raises(O2MasterUnavailableError):
        conn.run_raw(["ssh", "alice@o2-transfer", "ls"])

    # When a user-qualified master does exist, both the local socket lookup and
    # the control check retain that user. This matters for ControlPath templates
    # containing %r, which must not silently select the alias's default user.
    recording_runner = RecordingRunner(master=True)
    recording_conn = O2Connection(_config(tmp_path), runner=recording_runner)
    recording_conn.run_raw(["rsync", "-e", "ssh", "x", "alice@o2-transfer:/p"])
    raw_rsync = recording_runner.calls[-1]["argv"]
    transport = shlex.split(raw_rsync[raw_rsync.index("-e") + 1])
    assert transport[transport.index("-S") + 1] == "/tmp/alice@o2-transfer-control.sock"
    assert any(call["argv"][-1] == "alice@o2-transfer" and "-O" in call["argv"] for call in recording_runner.calls)

    # Only SSH's destination operand selects a socket. An alias appearing in the
    # remote command is ordinary data and must not override the actual `o2` host.
    recording_conn.run_raw(["ssh", "o2", "echo", "o2-transfer"])
    raw_ssh = recording_runner.calls[-1]["argv"]
    assert raw_ssh[raw_ssh.index("-S") + 1] == "/tmp/o2-control.sock"

    # Likewise, an rsync option argument can resemble a remote endpoint without
    # being a source or destination. Target inference must ignore that value.
    recording_conn.run_raw(["rsync", "--exclude", "o2-transfer:/ignored", "x", "o2:/p"])
    raw_rsync = recording_runner.calls[-1]["argv"]
    transport = shlex.split(raw_rsync[raw_rsync.index("-e") + 1])
    assert transport[transport.index("-S") + 1] == "/tmp/o2-control.sock"


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
        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
            return CommandResult(list(argv), 0, f"controlpath /tmp/{argv[-1]}-control.sock\n", "")
        if argv[-1] == "true":
            raise sp.TimeoutExpired(argv, timeout)  # connection dead -> ping stalls
        return CommandResult(list(argv), 0, "", "")

    _patch_keepalive(monkeypatch, O2Connection(_config(tmp_path), runner=runner))
    out = o2keepalive.keepalive()
    assert out["action"] == "stale_master_cleared"
    assert any("-O" in c and "exit" in c for c in calls)  # tore the stale master down
