"""Offline unit tests for the o2mcp cluster primitives.

The subprocess seam is injected, so these never touch the network: they assert
the exact ssh/sbatch/squeue/rsync commands are built, the global policy and
ControlMaster guards fire, and Slurm output is parsed correctly.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import threading
from pathlib import Path

import pytest

from o2mcp import (
    CommandResult,
    O2Config,
    O2LockedError,
    O2LoginCoordinationError,
    O2LoginGrantError,
    O2MasterUnavailableError,
    O2OffVpnError,
    O2Slurm,
    O2Sync,
    O2UnsafeTransportError,
    default_runner,
)
from o2mcp import (
    O2Connection as _ProductionO2Connection,
)
from o2mcp import keepalive as o2keepalive
from o2mcp.broker import BrokerExecutionResult, O2BrokerUnavailableError
from o2mcp.broker_protocol import MAX_TIMEOUT_SECONDS


class O2Connection(_ProductionO2Connection):
    """Use the explicit offline legacy transport for argv regression tests."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("_legacy_test_transport", "broker_client" not in kwargs)
        super().__init__(*args, **kwargs)


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


def _config(tmp_path: Path, *, locked: bool = False) -> O2Config:
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
        policy_file=policy_file,
        ssh_config_file=ssh_config,
    )


def test_default_policy_is_workstation_wide(monkeypatch, tmp_path):
    """Unconfigured clients must converge on one user-level policy state."""

    monkeypatch.delenv("O2_POLICY_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert O2Config().policy_file == tmp_path / ".agent_locks" / "O2_POLICY.json"


def test_explicit_policy_path_overrides_global_default(monkeypatch, tmp_path):
    """Isolated deployments may intentionally provide one shared state path."""

    explicit = tmp_path / "shared" / "O2_POLICY.json"
    monkeypatch.setenv("O2_POLICY_FILE", str(explicit))

    assert O2Config().policy_file == explicit


def test_relative_policy_paths_are_rejected(monkeypatch):
    """Working-directory-dependent state cannot coordinate MCP processes."""

    monkeypatch.setenv("O2_POLICY_FILE", "relative/O2_POLICY.json")
    with pytest.raises(ValueError, match="must be absolute"):
        O2Config()
    with pytest.raises(ValueError, match="must be absolute"):
        O2Config(policy_file=Path("another-relative-policy.json"))


def test_role_specific_broker_directories_are_absolute_and_distinct(tmp_path):
    """Separate host roles cannot accidentally contend for one socket authority."""

    with pytest.raises(ValueError, match="transfer broker directory must be.*absolute"):
        O2Config(transfer_broker_dir=Path("relative-transfer-broker"))

    shared = tmp_path / "shared-broker"
    with pytest.raises(ValueError, match="must use different authority directories"):
        O2Config(broker_dir=shared, transfer_broker_dir=shared)

    root = tmp_path / "authority-root"
    root.mkdir()
    with pytest.raises(ValueError, match="must use different authority directories"):
        O2Config(
            broker_dir=root / "login",
            transfer_broker_dir=root / "temporary" / ".." / "login",
        )

    linked_root = tmp_path / "linked-authority-root"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="must use different authority directories"):
        O2Config(
            broker_dir=root / "login",
            transfer_broker_dir=linked_root / "login",
        )


@pytest.mark.parametrize("invalid_timeout", [True, 0, float("inf"), MAX_TIMEOUT_SECONDS + 1])
def test_broker_start_timeout_is_rejected_before_authorized_launch(tmp_path, invalid_timeout):
    """Configuration cannot defer an invalid timeout until after grant use."""

    with pytest.raises(ValueError, match="broker start timeout"):
        O2Config(
            broker_dir=tmp_path / "login-broker",
            transfer_broker_dir=tmp_path / "transfer-broker",
            broker_start_timeout=invalid_timeout,
        )


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


# --- global policy -----------------------------------------------------------
def test_disabled_policy_blocks_every_remote_operation(tmp_path):
    """Disabled mode rejects remote paths before any SSH subprocess is built."""

    runner = RecordingRunner()
    conn = O2Connection(_config(tmp_path, locked=True), runner=runner)
    with pytest.raises(O2LockedError):
        conn.run("hostname")
    with pytest.raises(O2LockedError):
        conn.run_raw(["rsync", "x", "y"])
    # Even local `ssh -G` configuration expansion is skipped while disabled.
    assert runner.calls == []


def test_legacy_project_lock_is_not_a_second_policy_source(monkeypatch, tmp_path):
    """Only O2_POLICY.json is authoritative after the breaking migration."""

    project = tmp_path / "legacy-project"
    legacy_lock = project / ".agent_locks" / "O2_DISABLED"
    legacy_lock.parent.mkdir(parents=True)
    legacy_lock.write_text("obsolete")
    monkeypatch.chdir(project)

    runner = RecordingRunner(master=True)
    conn = O2Connection(_config(tmp_path), runner=runner)

    assert conn.run("hostname").ok
    assert runner.remote_commands == ["hostname"]


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


def test_policy_disable_cannot_complete_between_reuse_check_and_launch(tmp_path):
    """A remote child launch linearizes before or after the global stop."""

    class DisableDuringRemoteRunner(RecordingRunner):
        """Attempt the safety stop while the guarded runner is being invoked."""

        def __init__(self, policy):
            super().__init__(master=True)
            self._policy = policy
            self.disable_started = threading.Event()
            self.disable_finished = threading.Event()
            self.disable_thread = None

        def __call__(self, argv, timeout, input_text) -> CommandResult:
            result = super().__call__(argv, timeout, input_text)
            if not argv or argv[-1] != "hostname":
                return result

            def disable() -> None:
                self.disable_started.set()
                self._policy.disable(reason="concurrent reuse stop")
                self.disable_finished.set()

            self.disable_thread = threading.Thread(target=disable)
            self.disable_thread.start()
            assert self.disable_started.wait(timeout=1)
            # The child-launch seam still owns the policy mutex, so disabled
            # cannot become durable until this launch has linearized.
            assert not self.disable_finished.wait(timeout=0.1)
            return result

    config = _config(tmp_path)
    bootstrap = O2Connection(config, runner=RecordingRunner(master=True))
    runner = DisableDuringRemoteRunner(bootstrap.policy)
    conn = O2Connection(config, runner=runner, policy=bootstrap.policy)

    result = conn.run("hostname")

    assert runner.disable_thread is not None
    runner.disable_thread.join(timeout=2)
    assert result.ok and runner.disable_finished.is_set()
    assert conn.policy.snapshot().effective_mode == "disabled"


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


def _authorize(conn: O2Connection, *, target="transfer", allow_offvpn=False):
    """Issue one deterministic test grant against the currently observed revision."""

    snapshot = conn.policy.snapshot()
    return conn.policy.authorize_login(
        expected_revision=snapshot.revision,
        expected_generation=snapshot.generation,
        target=target,
        allow_offvpn=allow_offvpn,
        approval_reference="explicit test approval",
    )


def _start_transfer_master(conn: O2Connection, *, grant_id=None):
    """Exercise the sole retained ControlMaster start path explicitly."""

    return conn.start_master(
        grant_id=grant_id,
        alias=conn.config.transfer_alias,
        login_target="transfer",
    )


def test_transfer_master_requires_one_shot_grant(tmp_path):
    runner = RecordingRunner(master=False, responder=_vpn_responder("utun6"))
    conn = O2Connection(_config(tmp_path), runner=runner)
    with pytest.raises(O2MasterUnavailableError):
        _start_transfer_master(conn)

    grant = _authorize(conn)
    result = _start_transfer_master(conn, grant_id=grant.id)

    assert result.ok
    starts = [call["argv"] for call in runner.calls if "-MNf" in call["argv"]]
    assert len(starts) == 1
    assert "PubkeyAuthentication=no" not in starts[0]
    assert conn.policy.snapshot().state["login_grant"] is None
    assert conn.policy.snapshot().state["login_attempt"]["outcome"] == "success"


def test_policy_disable_cannot_complete_between_consumption_and_launch(tmp_path):
    """A safety stop is serialized with the sole authentication-capable call."""

    class DisableDuringLaunchRunner(RecordingRunner):
        """Start a competing disable while the launch runner still owns control."""

        def __init__(self, policy, **kwargs):
            super().__init__(**kwargs)
            self._policy = policy
            self.disable_started = threading.Event()
            self.disable_finished = threading.Event()
            self.disable_thread = None

        def __call__(self, argv, timeout, input_text) -> CommandResult:
            result = super().__call__(argv, timeout, input_text)
            if "-MNf" not in argv:
                return result

            def disable() -> None:
                self.disable_started.set()
                self._policy.disable(reason="concurrent incident stop")
                self.disable_finished.set()

            self.disable_thread = threading.Thread(target=disable)
            self.disable_thread.start()
            assert self.disable_started.wait(timeout=1)
            # The launch runner is still executing inside the policy context,
            # so disable must remain blocked on the workstation mutex.
            assert not self.disable_finished.wait(timeout=0.1)
            return result

    config = _config(tmp_path)
    bootstrap = O2Connection(config, runner=RecordingRunner(master=False))
    runner = DisableDuringLaunchRunner(
        bootstrap.policy,
        master=False,
        responder=_vpn_responder("utun6"),
    )
    conn = O2Connection(config, runner=runner, policy=bootstrap.policy)
    grant = _authorize(conn)

    result = _start_transfer_master(conn, grant_id=grant.id)

    assert runner.disable_thread is not None
    runner.disable_thread.join(timeout=2)
    assert not runner.disable_thread.is_alive()
    assert runner.disable_finished.is_set()
    assert result.ok
    assert conn.policy.snapshot().effective_mode == "disabled"


def test_start_master_rejects_zero_exit_when_control_socket_disappears(tmp_path):
    """A zero exit is failure and the consumed grant cannot be retried."""

    runner = RecordingRunner(master=False, start_persists=False, responder=_vpn_responder("utun6"))
    conn = O2Connection(_config(tmp_path), runner=runner)
    grant = _authorize(conn)

    result = _start_transfer_master(conn, grant_id=grant.id)

    assert result.ok is False and result.returncode == 255
    assert "post-start control-socket check failed" in result.stderr
    attempt = conn.policy.snapshot().state["login_attempt"]
    assert attempt["outcome"] == "failed" and attempt["returncode"] == 255
    with pytest.raises(O2LoginGrantError):
        _start_transfer_master(conn, grant_id=grant.id)
    assert sum("-MNf" in call["argv"] for call in runner.calls) == 1


def test_start_master_records_post_start_verification_timeout(tmp_path):
    """A timed-out postcondition is recorded without a second SSH attempt."""

    class VerificationTimeoutRunner(RecordingRunner):
        def __call__(self, argv, timeout, input_text) -> CommandResult:
            if "-O" in argv and "check" in argv and self.master:
                self.calls.append({"argv": list(argv), "timeout": timeout, "input": input_text})
                raise subprocess.TimeoutExpired(argv, timeout)
            return super().__call__(argv, timeout, input_text)

    runner = VerificationTimeoutRunner(master=False, responder=_vpn_responder("utun6"))
    conn = O2Connection(_config(tmp_path), runner=runner)
    grant = _authorize(conn)

    result = _start_transfer_master(conn, grant_id=grant.id)

    assert result.ok is False and result.returncode == 255
    assert "TimeoutExpired" in result.stderr
    assert conn.policy.snapshot().state["login_attempt"]["outcome"] == "failed"


def test_start_master_noop_when_running_does_not_need_grant(tmp_path):
    conn = O2Connection(_config(tmp_path), runner=RecordingRunner(master=True))
    result = _start_transfer_master(conn)
    assert result.ok and "already running" in result.stdout
    assert conn.policy.snapshot().state["login_grant"] is None


def test_start_master_can_open_the_transfer_alias_with_transfer_grant(tmp_path):
    runner = RecordingRunner(master=False, responder=_vpn_responder("utun6"))
    conn = O2Connection(_config(tmp_path), runner=runner)
    grant = _authorize(conn, target="transfer")

    result = _start_transfer_master(conn, grant_id=grant.id)

    assert result.ok
    starts = [call for call in runner.calls if "-MNf" in call["argv"]]
    assert len(starts) == 1 and starts[0]["argv"][-1] == "o2-transfer"


def test_stop_master_targets_transfer_and_legacy_login_roles(tmp_path):
    """The lifecycle API can retire both supported and pre-upgrade masters."""

    runner = RecordingRunner(master=True)
    conn = O2Connection(_config(tmp_path), runner=runner)

    result = conn.stop_master()

    assert result.ok
    assert runner.calls[-1]["argv"][-3:] == ["-O", "exit", "o2-transfer"]
    assert runner.calls[-1]["argv"][-1] == conn.config.transfer_alias
    legacy = conn.stop_master(alias=conn.config.host_alias)
    assert legacy.ok
    assert runner.calls[-1]["argv"][-3:] == ["-O", "exit", "o2"]
    assert "PubkeyAuthentication=no" in runner.calls[-1]["argv"]
    with pytest.raises(O2UnsafeTransportError, match="configured login or transfer ControlMaster"):
        conn.stop_master(alias="unconfigured-o2")


def test_default_master_start_is_retired_even_when_aliases_are_identical(tmp_path):
    """Alias equality cannot turn the unsafe default login role into transfer."""

    config = _config(tmp_path)
    config.transfer_alias = config.host_alias
    runner = RecordingRunner(master=False, responder=_vpn_responder("utun6"))
    conn = O2Connection(config, runner=runner)
    grant = _authorize(conn, target="login")

    with pytest.raises(O2UnsafeTransportError, match="Login ControlMaster startup is retired"):
        conn.start_master(grant_id=grant.id)
    assert not any("-MNf" in call["argv"] for call in runner.calls)


def test_explicit_role_disambiguates_one_shared_alias(tmp_path):
    """A transfer-scoped grant remains usable through a shared master alias."""

    config = _config(tmp_path)
    config.transfer_alias = config.host_alias
    runner = RecordingRunner(master=False, responder=_vpn_responder("utun6"))
    conn = O2Connection(config, runner=runner)
    grant = _authorize(conn, target="transfer")

    result = conn.start_master(
        grant_id=grant.id,
        alias=config.transfer_alias,
        login_target="transfer",
    )

    assert result.ok


def test_compatibility_coordination_error_catches_grant_failures():
    """Legacy handlers must catch every replacement coordination failure."""

    with pytest.raises(O2LoginCoordinationError):
        raise O2LoginGrantError("replacement grant failure")


def test_login_grant_cannot_cross_host_scope(tmp_path):
    runner = RecordingRunner(master=False)
    conn = O2Connection(_config(tmp_path), runner=runner)
    grant = _authorize(conn, target="login", allow_offvpn=True)

    with pytest.raises(O2LoginGrantError, match="scoped to 'login'"):
        _start_transfer_master(conn, grant_id=grant.id)
    assert not any("-MNf" in call["argv"] for call in runner.calls)


def test_timed_out_login_consumes_grant_and_leaves_attempt_receipt(tmp_path):
    """A subprocess timeout cannot leave authorization for a second Duo call."""

    class TimeoutStartRunner(RecordingRunner):
        def __call__(self, argv, timeout, input_text) -> CommandResult:
            if "-MNf" in argv:
                self.calls.append({"argv": list(argv), "timeout": timeout, "input": input_text})
                raise subprocess.TimeoutExpired(argv, timeout)
            return super().__call__(argv, timeout, input_text)

    runner = TimeoutStartRunner(master=False, responder=_vpn_responder("utun6"))
    conn = O2Connection(_config(tmp_path), runner=runner)
    grant = _authorize(conn)

    with pytest.raises(subprocess.TimeoutExpired):
        _start_transfer_master(conn, grant_id=grant.id)
    with pytest.raises(O2LoginGrantError):
        _start_transfer_master(conn, grant_id=grant.id)

    state = conn.policy.snapshot().state
    assert state["login_grant"] is None
    assert state["login_attempt"]["outcome"] == "timed_out"
    assert sum("-MNf" in call["argv"] for call in runner.calls) == 1


# --- VPN egress guard (HMS O2 Duos non-HMS source IPs) -----------------------
def _vpn_responder(interface):
    """Responder answering the guard's `ssh -G` (HostName) and `route get` (interface)."""

    def responder(argv, input_text):
        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
            return ("hostname o2.hms.harvard.edu\n", "", 0)
        if argv[:2] == ["route", "get"]:
            if interface is None:
                return ("", "no route to host", 1)
            return (f"   route to: o2\n   interface: {interface}\n   gateway: x\n", "", 0)
        return ("", "", 0)

    return responder


def test_start_master_refuses_off_vpn(tmp_path):
    # Route egresses via a physical interface (en0) -> a fresh login would Duo. Refuse,
    # and never open the login (-MNf is never invoked).
    runner = RecordingRunner(master=False, responder=_vpn_responder("en0"))
    conn = O2Connection(_config(tmp_path), runner=runner)
    grant = _authorize(conn)
    with pytest.raises(O2OffVpnError):
        _start_transfer_master(conn, grant_id=grant.id)
    assert not any("-MNf" in call["argv"] for call in runner.calls)


def test_start_master_allows_on_vpn(tmp_path):
    # Route egresses via the VPN tunnel (utun*) -> proceed to open the master.
    runner = RecordingRunner(master=False, responder=_vpn_responder("utun6"))
    conn = O2Connection(_config(tmp_path), runner=runner)
    grant = _authorize(conn)
    result = _start_transfer_master(conn, grant_id=grant.id)
    assert result.ok and any("-MNf" in call["argv"] for call in runner.calls)


def test_start_master_auto_authorizes_one_on_vpn_attempt(tmp_path):
    """Standing VPN authority uses the normal one-attempt grant and receipt."""

    runner = RecordingRunner(master=False, responder=_vpn_responder("utun6"))
    conn = O2Connection(_config(tmp_path), runner=runner)

    result = conn.start_master(
        alias=conn.config.transfer_alias,
        login_target="transfer",
        auto_authorize_on_vpn=True,
    )

    assert result.ok
    assert sum("-MNf" in call["argv"] for call in runner.calls) == 1
    state = conn.policy.snapshot().state
    assert state["login_grant"] is None
    assert state["login_attempt"]["allow_offvpn"] is False
    assert state["login_attempt"]["target"] == "transfer"
    assert state["login_attempt"]["outcome"] == "success"
    authorized = [event for event in state["events"] if event["event"] == "login_authorized"]
    assert authorized[-1]["approval_reference"].startswith("standing user authorization")


def test_start_master_auto_authorization_asks_off_vpn_without_policy_mutation(tmp_path):
    """Off-VPN auto-start fails locally before minting a grant or opening SSH."""

    runner = RecordingRunner(master=False, responder=_vpn_responder("en0"))
    conn = O2Connection(_config(tmp_path), runner=runner)
    before = conn.policy.snapshot()

    with pytest.raises(O2OffVpnError):
        conn.start_master(
            alias=conn.config.transfer_alias,
            login_target="transfer",
            auto_authorize_on_vpn=True,
        )

    after = conn.policy.snapshot()
    assert after.revision == before.revision
    assert after.state["login_grant"] is None
    assert after.state["login_attempt"] is None
    assert not any("-MNf" in call["argv"] for call in runner.calls)


def test_start_master_offvpn_override(tmp_path):
    # Off-VPN permission is carried only by the consumed one-shot grant.
    runner = RecordingRunner(master=False, responder=_vpn_responder("en0"))
    conn = O2Connection(_config(tmp_path), runner=runner)
    grant = _authorize(conn, allow_offvpn=True)
    result = _start_transfer_master(conn, grant_id=grant.id)
    assert result.ok and any("-MNf" in call["argv"] for call in runner.calls)


def test_start_master_fails_closed_when_iface_undetermined(tmp_path):
    """Unknown routing needs an explicitly off-VPN-scoped grant."""

    runner = RecordingRunner(master=False, responder=_vpn_responder(None))
    conn = O2Connection(_config(tmp_path), runner=runner)
    grant = _authorize(conn)
    with pytest.raises(O2OffVpnError):
        _start_transfer_master(conn, grant_id=grant.id)
    assert not any("-MNf" in call["argv"] for call in runner.calls)


def test_start_master_route_binary_missing_needs_offvpn_grant(tmp_path):
    """A missing local route utility is indeterminate rather than implicit approval."""

    def responder(argv, _input):
        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
            return ("hostname o2.hms.harvard.edu\n", "", 0)
        if argv[:2] == ["route", "get"]:
            raise FileNotFoundError(2, "No such file or directory", "route")
        return ("", "", 0)

    runner = RecordingRunner(master=False, responder=responder)
    conn = O2Connection(_config(tmp_path), runner=runner)
    grant = _authorize(conn, allow_offvpn=True)
    result = _start_transfer_master(conn, grant_id=grant.id)
    assert result.ok and any("-MNf" in call["argv"] for call in runner.calls)


def test_o2config_new_fields_are_appended_for_positional_compatibility():
    # Policy/VPN fields remain after the original public connection fields so a
    # positional O2Config(...) caller is not silently shifted.
    from dataclasses import fields

    names = [f.name for f in fields(O2Config)]
    assert names.index("policy_file") < names.index("default_user")
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
    assert argv[-1] == "o2-transfer:/scratch/jobs/run.sbatch"

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

    # Every remote-shell option is normalized, not just the first. Use a benign
    # caller option here; safety-sensitive `-o` options are rejected separately.
    conn.run_raw(["rsync", "-e", "ssh", "-essh -o ServerAliveInterval=15", "x", "o2:/p"])
    rsync = runner.calls[-1]["argv"]
    transports = [
        rsync[rsync.index("-e") + 1],
        next(token[2:] for token in rsync if token.startswith("-e") and token != "-e"),
    ]
    assert all("PubkeyAuthentication=no" in transport for transport in transports)
    assert all("PreferredAuthentications=none" in transport for transport in transports)
    assert "ServerAliveInterval=15" in transports[1]

    conn.run_raw(["rsync", "-avze", "ssh -o ServerAliveInterval=15", "x", "o2:/p"])
    clustered = runner.calls[-1]["argv"]
    transport = clustered[clustered.index("-avze") + 1]
    assert "PubkeyAuthentication=no" in transport and "ServerAliveInterval=15" in transport

    conn.run_raw(["rsync", "-avzessh -o ServerAliveInterval=15", "x", "o2:/p"])
    attached = runner.calls[-1]["argv"]
    transport = next(token[len("-avze") :] for token in attached if token.startswith("-avze"))
    assert "PubkeyAuthentication=no" in transport and "ServerAliveInterval=15" in transport

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
    conn.run_raw(["rsync", "-avze=ssh -o ServerAliveInterval=15", "x", "o2:/p"])
    equals_cluster = runner.calls[-1]["argv"]
    equals_transport = next(token[len("-avze=") :] for token in equals_cluster if token.startswith("-avze="))
    assert "PubkeyAuthentication=no" in equals_transport and "ServerAliveInterval=15" in equals_transport


@pytest.mark.parametrize(
    "argv",
    [
        ["ssh", "-l", "alice", "o2", "hostname"],
        ["ssh", "-o", "User=alice", "o2", "hostname"],
        ["ssh", "-p", "2222", "o2", "hostname"],
        ["ssh", "-p2222", "o2", "hostname"],
        ["ssh", "-o", "Port=2222", "o2", "hostname"],
        ["ssh", "-o", "HostName=other.example", "o2", "hostname"],
        ["ssh", "-J", "proxy.example", "o2", "hostname"],
        ["ssh", "-o", "ProxyCommand=ssh proxy.example", "o2", "hostname"],
        ["ssh", "-o", "PubkeyAuthentication=yes", "o2", "hostname"],
        ["ssh", "-i", "/tmp/key", "o2", "hostname"],
        ["rsync", "-e", "ssh -l alice", "x", "o2:/p"],
        ["rsync", "-e", "ssh -o User=alice", "x", "o2:/p"],
        ["rsync", "-e", "ssh -p 2222", "x", "o2:/p"],
        ["rsync", "-e", "ssh -o Port=2222", "x", "o2:/p"],
        ["rsync", "-e", "ssh -o ProxyCommand='ssh proxy.example'", "x", "o2:/p"],
        ["rsync", "-e", "ssh -o PubkeyAuthentication=yes", "x", "o2:/p"],
    ],
)
def test_run_raw_rejects_safety_sensitive_ssh_options(tmp_path, argv):
    """Caller options cannot alter endpoint, proxy, or authentication safety."""

    runner = RecordingRunner(master=True)
    conn = O2Connection(_config(tmp_path), runner=runner)

    with pytest.raises(O2UnsafeTransportError, match="options are disabled"):
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
    assert argv[-1] == "o2-transfer:" + remote.replace(" ", "\\ ")
    assert " " not in argv[-1].replace("\\ ", "")  # no UNescaped space remains


def test_remote_path_preserves_tilde_and_vars(tmp_path):
    # ~, $VAR and ${VAR} stay bare so the remote shell still expands them.
    runner = RecordingRunner(master=True)
    sync = O2Sync(O2Connection(_config(tmp_path), runner=runner))
    assert sync.push_argv("a", "~/jobs/run.sbatch")[-1] == "o2-transfer:~/jobs/run.sbatch"
    assert sync.push_argv("a", "$SCRATCH/out")[-1] == "o2-transfer:$SCRATCH/out"
    assert sync.push_argv("a", "${SCRATCH}/out")[-1] == "o2-transfer:${SCRATCH}/out"  # braced var preserved
    assert sync.push_argv("a", "$SCRATCH/my out")[-1] == "o2-transfer:$SCRATCH/my\\ out"  # spaces still escaped
    assert (
        sync.push_argv("a", "$(whoami)/x")[-1] == "o2-transfer:$\\(whoami\\)/x"
    )  # () escaped -> no command substitution


def test_escape_is_noop_for_plain_paths(tmp_path):
    # Space-free paths must be byte-for-byte unchanged (no behavior change, no stray escapes).
    runner = RecordingRunner(master=True)
    sync = O2Sync(O2Connection(_config(tmp_path), runner=runner))
    assert sync.push_argv("a", "/n/groups/tabin/jzhao/runs/foo")[-1] == "o2-transfer:/n/groups/tabin/jzhao/runs/foo"
    # push_argv builds exactly what push() runs.
    sync.push("a", "/n/groups/tabin/jzhao/runs/foo")
    assert runner.calls[-1]["argv"] == sync.push_argv("a", "/n/groups/tabin/jzhao/runs/foo")


def test_default_rsync_requires_transfer_master_but_legacy_login_reuse_remains(tmp_path):
    # The login master is up but the transfer-node master is NOT. Default rsync
    # must refuse on the transfer alias, while an explicit legacy-login reuse may
    # proceed without creating a new login master.
    def runner(argv, timeout, input_text):
        if "-O" in argv and "check" in argv:
            return CommandResult(list(argv), 0 if argv[-1] == "o2" else 255, "", "")
        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
            return CommandResult(list(argv), 0, f"controlpath /tmp/{argv[-1]}-control.sock\n", "")
        return CommandResult(list(argv), 0, "", "")

    sync = O2Sync(O2Connection(_config(tmp_path), runner=runner))
    with pytest.raises(O2MasterUnavailableError):
        sync.push("a", "b")  # transfer alias master is down -> refuse
    sync.push("a", "b", transfer=False)  # pre-existing login master may still be reused


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


@pytest.mark.parametrize(
    ("argv", "master_alias"),
    [
        (["ssh", "o2", "hostname"], "o2-transfer"),
        (["ssh", "alice@o2", "hostname"], "o2"),
        (["rsync", "x", "o2-transfer:/p"], "o2"),
    ],
)
def test_run_raw_rejects_master_alias_destination_mismatch(tmp_path, argv, master_alias):
    """The explicit socket identity must equal the command's actual endpoint."""

    runner = RecordingRunner(master=True)
    conn = O2Connection(_config(tmp_path), runner=runner)

    with pytest.raises(O2UnsafeTransportError, match="disagrees with transport destination"):
        conn.run_raw(argv, master_alias=master_alias)

    # The mismatch is determined from argv alone, before config expansion or any
    # local master probe has a chance to obscure the caller error.
    assert runner.calls == []


@pytest.mark.parametrize(
    "argv",
    [
        ["ssh", "o22", "hostname"],
        ["rsync", "x", "other:/p"],
        ["rsync", "o2:/p", "other:/q"],
    ],
)
def test_run_raw_rejects_unrecognized_transport_destinations(tmp_path, argv):
    """A typoed host must not be overridden by the default pinned O2 socket."""

    runner = RecordingRunner(master=True)
    conn = O2Connection(_config(tmp_path), runner=runner)

    with pytest.raises(O2UnsafeTransportError, match="not a configured O2 alias"):
        conn.run_raw(argv)

    assert runner.calls == []


def test_run_raw_rejects_multiple_configured_endpoints(tmp_path):
    """One rsync command cannot safely pin sockets for two remote endpoints."""

    runner = RecordingRunner(master=True)
    conn = O2Connection(_config(tmp_path), runner=runner)

    with pytest.raises(O2UnsafeTransportError, match="multiple O2 endpoints"):
        conn.run_raw(["rsync", "o2:/source", "o2-transfer:/dest"])

    assert runner.calls == []


def test_rsync_local_colon_path_is_not_mistaken_for_remote_endpoint(tmp_path):
    """A colon after a slash is a local rsync path, not an unrecognized host."""

    runner = RecordingRunner(master=True)
    conn = O2Connection(_config(tmp_path), runner=runner)

    result = conn.run_raw(["rsync", "./sample:a", "./out"])

    assert result.ok
    assert runner.calls[-1]["argv"][0] == O2Connection.RSYNC_EXECUTABLE


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


# --- keepalive (must never open a new session channel) -----------------------
def _patch_keepalive(monkeypatch, conn):
    monkeypatch.setattr(o2keepalive, "O2Connection", lambda config=None: conn)


class _KeepaliveBroker:
    """Minimal broker double for keepalive routing and failure behavior."""

    def __init__(self, *, responsive=True, fail=False):
        self.responsive = responsive
        self.fail = fail
        self.commands = []

    def local_status(self):
        return {"responsive": self.responsive}

    def execute(self, command, *, timeout, input_text=None):
        self.commands.append(command)
        if self.fail:
            raise O2BrokerUnavailableError("persistent channel ended")
        return BrokerExecutionResult(0, "", "", False, 0.01, False, False)


def test_keepalive_skips_when_policy_disabled(tmp_path, monkeypatch):
    runner = RecordingRunner()
    _patch_keepalive(monkeypatch, O2Connection(_config(tmp_path, locked=True), runner=runner))
    assert o2keepalive.keepalive() == {"action": "skipped", "reason": "O2PolicyDeniedError"}
    assert runner.calls == []  # never touched ssh


def test_keepalive_skips_when_no_broker(tmp_path, monkeypatch):
    broker = _KeepaliveBroker(responsive=False)
    _patch_keepalive(monkeypatch, O2Connection(_config(tmp_path), broker_client=broker))
    out = o2keepalive.keepalive()
    assert out["action"] == "skipped" and out["reason"] == "no_broker"
    assert broker.commands == []


def test_keepalive_pings_existing_broker(tmp_path, monkeypatch):
    broker = _KeepaliveBroker()
    _patch_keepalive(monkeypatch, O2Connection(_config(tmp_path), broker_client=broker))
    out = o2keepalive.keepalive()
    assert out["action"] == "pinged" and out["ok"] is True
    assert broker.commands == ["true"]


def test_keepalive_skips_failed_broker_without_restart(tmp_path, monkeypatch):
    """A dead channel is reported locally; keepalive never starts a replacement."""

    broker = _KeepaliveBroker(fail=True)
    _patch_keepalive(monkeypatch, O2Connection(_config(tmp_path), broker_client=broker))
    out = o2keepalive.keepalive()
    assert out == {"action": "skipped", "reason": "O2BrokerUnavailableError"}
    assert broker.commands == ["true"]
