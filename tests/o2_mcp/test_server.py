"""End-to-end tests for the o2_mcp MCP server, driven through the MCP protocol.

These exercise the full server path (FastMCP argument validation -> async tool ->
o2mcp core -> JSON payload), with the subprocess call injected so no network
is touched. They need the ``mcp`` SDK (Python 3.10+), so they are skipped in the
default 3.9 test environment (install with ``pip install -e ".[dev,o2]"`` on a 3.10+ env).
"""

from __future__ import annotations

import json
import plistlib
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("anyio")

from mcp.server.fastmcp.exceptions import ToolError  # noqa: E402

from o2mcp import (  # noqa: E402
    CommandResult,
    O2BrokerBusyError,  # noqa: E402
    O2BrokerCommandOutcomeUnknownError,  # noqa: E402
    O2Config,
    async_transfer,  # noqa: E402
    transfer_tools,  # noqa: E402
)
from o2mcp import O2AsyncTransfer as _RealAsyncTransfer  # noqa: E402
from o2mcp import (
    O2Connection as _ProductionO2Connection,
)
from o2mcp import billing  # noqa: E402
from o2mcp import server as o2server  # noqa: E402


class O2Connection(_ProductionO2Connection):
    """Select the explicit offline transport for MCP fake-runner tests."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("_legacy_test_transport", True)
        super().__init__(*args, **kwargs)


@pytest.fixture(autouse=True)
def isolate_user_level_o2_files(monkeypatch, tmp_path):
    """Prevent protocol tests from writing login receipts into the real home."""

    monkeypatch.setenv("HOME", str(tmp_path / "test-home"))


class FakeRunner:
    """Deterministic stand-in for the subprocess runner (records calls)."""

    def __init__(self, *, master: bool = True, responder=None, start_persists: bool = True):
        self.calls = []
        self.master = master
        self._responder = responder
        self._start_persists = start_persists

    def __call__(self, argv, timeout, input_text) -> CommandResult:
        self.calls.append({"argv": list(argv), "input": input_text})
        if "-O" in argv and "check" in argv:
            return CommandResult(list(argv), 0 if self.master else 255, "", "")
        if "-MNf" in argv:
            if self._start_persists:
                self.master = True
            return CommandResult(list(argv), 0, "", "")
        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
            return CommandResult(list(argv), 0, f"controlpath /tmp/{argv[-1]}-control.sock\n", "")
        if self._responder is not None:
            out, err, rc = self._responder(argv, input_text)
            return CommandResult(list(argv), rc, out, err)
        return CommandResult(list(argv), 0, "", "")


def _patch_connection(
    monkeypatch, tmp_path, *, master=True, responder=None, locked=False, start_persists=True
) -> FakeRunner:
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
    cfg = O2Config(
        host_alias="o2",
        transfer_alias="o2-transfer",
        connect_timeout=20,
        policy_file=policy_file,
        ssh_config_file=ssh_config,
        globalprotect_settings_file=tmp_path / "globalprotect-settings.plist",
    )
    with cfg.globalprotect_settings_file.open("wb") as stream:
        plistlib.dump(
            {
                "Palo Alto Networks": {
                    "GlobalProtect": {
                        "PanSetup": {"Portal": "vpn.hms.harvard.edu"},
                        "PanGPS": {"PreferredIP_test": "10.116.16.225"},
                    }
                }
            },
            stream,
        )
    runner = FakeRunner(master=master, responder=responder, start_persists=start_persists)
    connection = O2Connection(cfg, runner=runner)
    runner.connection = connection
    monkeypatch.setattr(o2server, "_connection", lambda: connection)
    return runner


async def _call(name, arguments):
    """Invoke a tool via the MCP protocol and return its parsed JSON payload."""
    result = await o2server.mcp.call_tool(name, arguments)
    content = result[0] if isinstance(result, tuple) else result
    text = content[0].text if isinstance(content, list) else content
    return json.loads(text)


# --- registry / annotations --------------------------------------------------
@pytest.mark.anyio
async def test_tool_registry_and_annotations():
    tools = {t.name: t for t in await o2server.mcp.list_tools()}
    assert set(tools) == {
        "o2_status",
        "o2_local_status",
        "o2_probe",
        "o2_policy_disable",
        "o2_policy_enable_reuse",
        "o2_authorize_login",
        "o2_start_master",
        "o2_stop_master",
        "o2_start_broker",
        "o2_stop_broker",
        "o2_exec",
        "o2_submit_job",
        "o2_squeue",
        "o2_job_status",
        "o2_tail_log",
        "o2_cancel_job",
        "o2_push",
        "o2_pull",
        # non-blocking transfers
        "o2_push_async",
        "o2_pull_async",
        "o2_transfer_status",
        "o2_transfer_cancel",
        # workspace-layout tools
        "o2_disk_report",
        "o2_workspace_gc",
        "o2_place",
        # pre-submission pricing
        "o2_price_job",
        "o2_refresh_billing_weights",
    }
    assert tools["o2_status"].annotations.readOnlyHint is True
    assert tools["o2_status"].annotations.openWorldHint is False
    assert tools["o2_local_status"].annotations.openWorldHint is False
    assert tools["o2_probe"].annotations.openWorldHint is True
    assert tools["o2_submit_job"].annotations.readOnlyHint is False
    assert tools["o2_cancel_job"].annotations.destructiveHint is True
    assert tools["o2_stop_master"].annotations.destructiveHint is True
    assert tools["o2_stop_broker"].annotations.destructiveHint is True
    assert tools["o2_workspace_gc"].annotations.destructiveHint is True
    assert tools["o2_disk_report"].annotations.readOnlyHint is True
    assert tools["o2_push_async"].annotations.readOnlyHint is False
    assert tools["o2_transfer_status"].annotations.readOnlyHint is True
    assert tools["o2_transfer_cancel"].annotations.destructiveHint is True
    # Pricing is arithmetic over a cached weight table: it reaches the cluster
    # only when explicitly asked to refresh, so it must never look like a tool
    # that mutates or that requires a connection to answer.
    assert tools["o2_price_job"].annotations.readOnlyHint is True
    # Pricing reads a local cache and nothing else, so it neither writes nor
    # reaches outside itself. Refreshing that cache is a separate tool, because
    # a tool that can write must not claim to be read-only at the point a
    # client decides whether to auto-approve it.
    assert tools["o2_price_job"].annotations.openWorldHint is False
    assert tools["o2_refresh_billing_weights"].annotations.readOnlyHint is False
    assert tools["o2_refresh_billing_weights"].annotations.openWorldHint is True


@pytest.mark.anyio
@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # runpy re-executes the module on purpose
async def test_python_dash_m_startup_registers_transfer_tools(monkeypatch):
    # Under `python -m o2mcp.server` the running module is __main__; the transfer
    # tools must register on the instance main() actually serves. Regression guard: a
    # self-import of o2mcp.server would attach them to a duplicate module's FastMCP
    # that is never served, silently dropping all six transfer tools on that startup path.
    import runpy

    from mcp.server.fastmcp import FastMCP

    served: dict = {}
    monkeypatch.setattr(FastMCP, "run", lambda self, *a, **k: served.__setitem__("mcp", self))
    runpy.run_module("o2mcp.server", run_name="__main__", alter_sys=True)
    names = {t.name for t in await served["mcp"].list_tools()}
    assert {"o2_push", "o2_pull", "o2_push_async", "o2_pull_async", "o2_transfer_status", "o2_transfer_cancel"} <= names


# --- workspace-layout tools ---------------------------------------------------
def _workspace_responder(argv, input_text):
    command = argv[-1]
    if "du -sb" in command:
        return (
            "1932735283\t/home/jiz947/.cache\n"
            "412000000\t/home/jiz947/.o2ctl/legacy_trash\n"
            "8804682956\t/home/jiz947/envs\n",
            "",
            0,
        )
    if command.startswith("cat >"):
        return ("LAUNCHED", "", 0)
    return ("", "", 0)


@pytest.mark.anyio
async def test_disk_report_flags_regenerable_and_redundant(monkeypatch, tmp_path):
    _patch_connection(monkeypatch, tmp_path, responder=_workspace_responder)
    payload = await _call("o2_disk_report", {"params": {}})
    assert payload["ok"] is True
    assert payload["reclaimable_bytes"] == 1932735283 + 412000000  # .cache + legacy_trash, not envs


@pytest.mark.anyio
async def test_place_resolves_canonical_path(monkeypatch, tmp_path):
    _patch_connection(monkeypatch, tmp_path, responder=_workspace_responder)
    payload = await _call("o2_place", {"params": {"kind": "results", "project": "myproject"}})
    assert payload["ok"] is True and payload["path"] == "/n/groups/tabin/jzhao/results/myproject"


@pytest.mark.anyio
async def test_workspace_gc_dry_run_returns_script(monkeypatch, tmp_path):
    _patch_connection(monkeypatch, tmp_path, responder=_workspace_responder)
    payload = await _call("o2_workspace_gc", {"params": {"dry_run": True}})
    assert payload["ok"] is True and payload["submitted"] is False
    assert "/home/jiz947/.cache" in payload["script"] and "/home/jiz947/envs" not in payload["script"]


# --- policy and authentication paths ----------------------------------------
@pytest.mark.anyio
async def test_status_is_local_only_and_does_not_probe(monkeypatch, tmp_path):
    runner = _patch_connection(monkeypatch, tmp_path, master=False)
    payload = await _call("o2_status", {})

    assert payload["ok"] is True and payload["local_only"] is True
    assert payload["policy"]["effective_mode"] == "reuse_only"
    assert runner.calls == []


@pytest.mark.anyio
async def test_local_status_reports_disabled_policy_without_ssh(monkeypatch, tmp_path):
    """Diagnostics name the policy state while making no SSH runner call."""

    runner = _patch_connection(monkeypatch, tmp_path, master=True, locked=True)
    payload = await _call("o2_local_status", {})

    assert payload["ok"] is True
    assert payload["policy"]["effective_mode"] == "disabled"
    assert payload["policy"]["path"] == str(tmp_path / "O2_POLICY.json")
    assert set(payload["command_brokers"]) == {"login", "transfer"}
    assert runner.calls == []


@pytest.mark.anyio
async def test_local_status_preserves_policy_when_transfer_metadata_is_corrupt(monkeypatch, tmp_path):
    """A truncated detached-transfer receipt cannot hide policy recovery data."""

    runner = _patch_connection(monkeypatch, tmp_path, master=False, locked=True)
    transfer_dir = tmp_path / "test-home" / ".cache" / "o2mcp" / "transfers"
    transfer_dir.mkdir(parents=True)
    corrupt = transfer_dir / "push-20260617-001234-99-0001.json"
    corrupt.write_text("{truncated")

    payload = await _call("o2_local_status", {})

    assert payload["ok"] is True
    assert payload["policy"]["effective_mode"] == "disabled"
    assert payload["policy"]["generation"]
    assert payload["transfers"][0]["error"] == "invalid_transfer_metadata"
    assert runner.calls == []


@pytest.mark.anyio
async def test_local_status_preserves_policy_when_socket_directory_is_unreadable(monkeypatch, tmp_path):
    """A socket enumeration race cannot replace local status with one error."""

    runner = _patch_connection(monkeypatch, tmp_path, master=False, locked=True)
    socket_root = tmp_path / "test-home" / ".ssh" / "controlmasters"
    socket_root.mkdir(parents=True)
    original_iterdir = Path.iterdir

    def selective_iterdir(path):
        """Model permission loss only at the ControlMaster directory."""

        if path == socket_root:
            raise PermissionError("socket directory became unreadable")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", selective_iterdir)
    payload = await _call("o2_local_status", {})

    assert payload["ok"] is True
    assert payload["policy"]["effective_mode"] == "disabled"
    assert payload["control_sockets"] == []
    assert runner.calls == []


@pytest.mark.anyio
async def test_run_without_master_is_actionable(monkeypatch, tmp_path):
    _patch_connection(monkeypatch, tmp_path, master=False)
    payload = await _call("o2_exec", {"params": {"command": "squeue"}})
    assert payload["ok"] is False
    assert payload["error"] == "no_master"
    assert "ControlMaster" in payload["message"]


@pytest.mark.anyio
async def test_start_master_refused_without_grant(monkeypatch, tmp_path):
    """Login masters are retired because they still open challenged channels."""

    _patch_connection(monkeypatch, tmp_path, master=False)
    payload = await _call("o2_start_master", {"params": {}})
    assert payload["ok"] is False and payload["error"] == "login_master_retired"
    assert "o2_start_broker" in payload["message"]


@pytest.mark.anyio
async def test_start_master_reports_failed_post_start_verification(monkeypatch, tmp_path):
    """A vanished transfer master must not be reported as successful."""

    _patch_connection(monkeypatch, tmp_path, master=False, start_persists=False)
    status = await _call("o2_local_status", {})
    authorization = await _call(
        "o2_authorize_login",
        {
            "params": {
                "expected_revision": status["policy"]["revision"],
                "expected_generation": status["policy"]["generation"],
                "target": "transfer",
                "allow_offvpn": True,
                "approval_reference": "explicit test approval",
            }
        },
    )

    payload = await _call(
        "o2_start_master",
        {"params": {"grant_id": authorization["grant_id"], "transfer": True}},
    )

    assert payload["ok"] is False
    assert payload["returncode"] == 255
    assert "post-start control-socket check failed" in payload["stderr"]


@pytest.mark.anyio
async def test_stop_master_targets_transfer_and_legacy_login_roles(monkeypatch, tmp_path):
    """The MCP can retire either known master without restoring login startup."""

    runner = _patch_connection(monkeypatch, tmp_path, master=True)

    stopped = await _call("o2_stop_master", {"params": {"transfer": True}})
    legacy = await _call("o2_stop_master", {"params": {"transfer": False}})

    assert stopped["ok"] is True and stopped["alias"] == "o2-transfer"
    assert legacy["ok"] is True and legacy["alias"] == "o2"
    stop_calls = [call for call in runner.calls if "-O" in call["argv"]]
    assert stop_calls[-2]["argv"][-3:] == ["-O", "exit", "o2-transfer"]
    assert stop_calls[-1]["argv"][-3:] == ["-O", "exit", "o2"]


@pytest.mark.anyio
async def test_transfer_broker_tools_preserve_role_selection(monkeypatch):
    """MCP input must carry transfer intent through start, probe, and stop."""

    class _Config:
        host_alias = "o2"
        transfer_alias = "o2-transfer"
        connect_timeout = 20

    class _Connection:
        config = _Config()

        def __init__(self):
            self.calls = []

        def start_broker(self, *, grant_id=None, transfer=False, auto_authorize_on_vpn=False):
            self.calls.append(("start", grant_id, transfer, auto_authorize_on_vpn))
            return {"responsive": True}

        def run(self, command, *, timeout, alias, broker_role=None):
            self.calls.append(("run", command, timeout, alias, broker_role))
            return CommandResult(["broker", alias, command], 0, "transfer-ok\n", "")

        def probe(self, *, alias=None, broker_role=None):
            # Mirrors the real connection, which routes its probe through `run`
            # so a single deadline governs every caller of it.
            return self.run("hostname; whoami; date", timeout=25, alias=alias, broker_role=broker_role)

        def stop_broker(self, *, reason, transfer=False, force=False):
            self.calls.append(("stop", reason, transfer, force))
            return {"type": "stopping"}

    connection = _Connection()
    monkeypatch.setattr(o2server, "_connection", lambda: connection)

    started = await _call("o2_start_broker", {"params": {"grant_id": "grant-1", "transfer": True}})
    probed = await _call("o2_probe", {"params": {"transfer": True}})
    stopped = await _call(
        "o2_stop_broker",
        {"params": {"reason": "offline role-routing test", "transfer": True}},
    )

    assert started["ok"] is True and started["target"] == "transfer"
    assert probed["ok"] is True and probed["alias"] == "o2-transfer"
    assert stopped["ok"] is True and stopped["target"] == "transfer"
    assert connection.calls == [
        ("start", "grant-1", True, True),
        ("run", "hostname; whoami; date", 25, "o2-transfer", "transfer"),
        ("stop", "offline role-routing test", True, False),
    ]


@pytest.mark.anyio
async def test_transfer_master_auto_authorizes_only_on_proven_vpn(monkeypatch, tmp_path):
    """The MCP default starts on VPN without requiring a separate grant call."""

    def on_vpn(argv, _input_text):
        if argv[:2] == [O2Connection.ROUTE_EXECUTABLE, "get"]:
            return ("interface: utun6\n", "", 0)
        if argv[:1] == [O2Connection.IFCONFIG_EXECUTABLE]:
            return ("inet 10.116.16.225 netmask 0xffffffff\n", "", 0)
        return ("", "", 0)

    runner = _patch_connection(monkeypatch, tmp_path, master=False, responder=on_vpn)
    # FakeRunner owns ssh -G so add the HostName that route proof needs.
    original = runner.__call__

    def with_hostname(argv, timeout, input_text):
        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
            runner.calls.append({"argv": list(argv), "input": input_text})
            return CommandResult(
                list(argv),
                0,
                f"hostname transfer.rc.hms.harvard.edu\ncontrolpath /tmp/{argv[-1]}-control.sock\n",
                "",
            )
        return original(argv, timeout, input_text)

    runner.connection._runner = with_hostname

    payload = await _call("o2_start_master", {"params": {"transfer": True}})

    assert payload["ok"] is True and payload["alias"] == "o2-transfer"
    assert sum("-MNf" in call["argv"] for call in runner.calls) == 1
    attempt = runner.connection.policy.snapshot().state["login_attempt"]
    assert attempt["target"] == "transfer"
    assert attempt["allow_offvpn"] is False


@pytest.mark.anyio
async def test_dispatched_result_loss_is_explicitly_not_retry_safe(monkeypatch):
    """The MCP response must discourage duplicating an uncertain remote action."""

    class _Connection:
        def run(self, *_args, **_kwargs):
            raise O2BrokerCommandOutcomeUnknownError("dispatched result was lost")

    monkeypatch.setattr(o2server, "_connection", _Connection)
    payload = await _call("o2_exec", {"params": {"command": "sbatch job.sh"}})

    assert payload == {
        "ok": False,
        "error": "broker_outcome_unknown",
        "message": "dispatched result was lost",
        "retry_safe": False,
    }


@pytest.mark.anyio
async def test_disabled_policy_blocks_tool(monkeypatch, tmp_path):
    _patch_connection(monkeypatch, tmp_path, master=True, locked=True)
    payload = await _call("o2_exec", {"params": {"command": "hostname"}})
    assert payload["ok"] is False and payload["error"] == "policy_disabled"


@pytest.mark.anyio
async def test_policy_disable_and_explicit_global_reenable(monkeypatch, tmp_path):
    _patch_connection(monkeypatch, tmp_path)
    disabled = await _call("o2_policy_disable", {"params": {"reason": "Duo incident"}})
    assert disabled["policy"]["mode"] == "disabled"

    enabled = await _call(
        "o2_policy_enable_reuse",
        {
            "params": {
                "expected_revision": disabled["policy"]["revision"],
                "expected_generation": disabled["policy"]["generation"],
                "approval_reference": "explicit global re-enable",
                "acknowledge_global": True,
            }
        },
    )
    assert enabled["ok"] is True and enabled["policy"]["mode"] == "reuse_only"


# --- submit / monitor (mocked runner) ----------------------------------------
@pytest.mark.anyio
async def test_submit_job_returns_job_id(monkeypatch, tmp_path):
    _patch_connection(
        monkeypatch, tmp_path, master=True, responder=lambda argv, _i: ("Submitted batch job 999\n", "", 0)
    )
    payload = await _call("o2_submit_job", {"params": {"remote_script_path": "~/jobs/run.sbatch"}})
    assert payload["ok"] is True and payload["submitted"] is True and payload["job_id"] == "999"


@pytest.mark.anyio
async def test_submit_text_stages_and_submits(monkeypatch, tmp_path):
    runner = _patch_connection(
        monkeypatch, tmp_path, master=True, responder=lambda argv, _i: ("Submitted batch job 7\n", "", 0)
    )
    payload = await _call(
        "o2_submit_job",
        {"params": {"script_text": "#!/bin/bash\nsrun hostname\n", "remote_path": "~/jobs/x.sbatch"}},
    )
    assert payload["job_id"] == "7"
    staged = [c for c in runner.calls if c["input"] is not None]
    assert staged and staged[0]["input"].startswith("#!/bin/bash")


@pytest.mark.anyio
async def test_submit_bad_input(monkeypatch, tmp_path):
    _patch_connection(monkeypatch, tmp_path, master=True)
    payload = await _call("o2_submit_job", {"params": {}})
    assert payload["ok"] is False and payload["error"] == "bad_input"


@pytest.mark.anyio
async def test_squeue_returns_structured_rows(monkeypatch, tmp_path):
    out = "100|clock_grid|RUNNING|01:00:00|08:00:00|1|node1\n"
    _patch_connection(monkeypatch, tmp_path, master=True, responder=lambda argv, _i: (out, "", 0))
    payload = await _call("o2_squeue", {"params": {"user": "jiz947"}})
    assert payload["ok"] is True
    assert payload["jobs"][0]["state"] == "RUNNING" and payload["jobs"][0]["job_id"] == "100"


@pytest.mark.anyio
async def test_cancel_job(monkeypatch, tmp_path):
    runner = _patch_connection(monkeypatch, tmp_path, master=True)
    payload = await _call("o2_cancel_job", {"params": {"job_id": "100"}})
    assert payload["ok"] is True
    assert runner.calls[-1]["argv"][-1] == "scancel 100"


# --- input validation (Pydantic, via the MCP layer) --------------------------
@pytest.mark.anyio
async def test_a_busy_broker_is_reported_separately_but_still_fails_closed(monkeypatch):
    """Occupied is worth naming, but a queue expiry is not proof nothing ran.

    The daemon acknowledges and forwards as one step, so a budget expiring in
    that instant leaves a command running and a caller that never saw it
    acknowledged. `broker_busy` subclasses the uncertain-outcome error for that
    reason, and must be caught before its base or it would report the wrong
    error string.
    """

    class _Connection:
        def run(self, *_args, **_kwargs):
            raise O2BrokerBusyError("waited 60s without an acknowledgement")

    monkeypatch.setattr(o2server, "_connection", _Connection)
    payload = await _call("o2_exec", {"params": {"command": "squeue -u me"}})

    assert payload["ok"] is False
    assert payload["error"] == "broker_busy"
    # Never advertise a mutating command as safe to duplicate.
    assert payload["retry_safe"] is False


def test_busy_is_an_uncertain_outcome_so_existing_handlers_fail_closed():
    """Inheritance is the guarantee: no caller can opt out of the safe default."""

    assert issubclass(O2BrokerBusyError, O2BrokerCommandOutcomeUnknownError)


@pytest.mark.anyio
async def test_exec_timeout_is_capped_so_one_command_cannot_hold_the_channel_for_an_hour(monkeypatch, tmp_path):
    """The cap is the contract: long remote waits belong in a submitted job."""

    _patch_connection(monkeypatch, tmp_path, master=True)
    with pytest.raises(ToolError):
        await o2server.mcp.call_tool(
            "o2_exec",
            {"params": {"command": "sleep 3000", "timeout_seconds": 3600}},
        )

    at_the_cap = await _call(
        "o2_exec",
        {"params": {"command": "true", "timeout_seconds": o2server.MAX_EXEC_TIMEOUT_SECONDS}},
    )
    assert at_the_cap["ok"] is True


def test_an_omitted_exec_timeout_stays_within_the_cap():
    """Most callers omit the timeout, so the default is the common path.

    A ceiling the default escapes is not a ceiling on the traffic that matters.
    """

    omitted = o2server.RunInput(command="squeue -u me")
    assert omitted.timeout_seconds <= o2server.MAX_EXEC_TIMEOUT_SECONDS


def test_no_input_default_escapes_its_own_constraint():
    """Sweep every input model, not just the field that got this wrong.

    Pydantic does not validate defaults unless asked, so a default outside a
    declared bound passes straight through to the caller it was meant to bound.
    """

    import inspect

    from pydantic import BaseModel

    offenders: list[str] = []
    for _name, model in vars(o2server).items():
        if not (inspect.isclass(model) and issubclass(model, BaseModel) and model is not BaseModel):
            continue
        for field_name, field in model.model_fields.items():
            default = field.default
            if isinstance(default, bool) or not isinstance(default, (int, float)):
                continue
            for constraint in getattr(field, "metadata", []):
                upper = getattr(constraint, "le", None)
                strict_upper = getattr(constraint, "lt", None)
                lower = getattr(constraint, "ge", None)
                strict_lower = getattr(constraint, "gt", None)
                where = f"{model.__name__}.{field_name} default={default}"
                if upper is not None and default > upper:
                    offenders.append(f"{where} exceeds le={upper}")
                if strict_upper is not None and default >= strict_upper:
                    offenders.append(f"{where} not below lt={strict_upper}")
                if lower is not None and default < lower:
                    offenders.append(f"{where} below ge={lower}")
                if strict_lower is not None and default <= strict_lower:
                    offenders.append(f"{where} not above gt={strict_lower}")

    assert offenders == [], offenders


@pytest.mark.anyio
async def test_a_remote_wait_is_not_expressible_through_exec(monkeypatch, tmp_path):
    """The cap is what bans waiting, because detecting a wait cannot work.

    This command is taken verbatim from a live occupancy record: an agent
    waiting on a Slurm job by sleeping remotely under a 290s deadline, holding
    the channel every other session shares for most of five minutes in order to
    do nothing. `sleep`, `python -c "time.sleep(...)"` and a `while` loop are
    one intent in three shapes, so the bound is on occupancy rather than on
    anything read out of the command string.
    """

    _patch_connection(monkeypatch, tmp_path, master=True)
    with pytest.raises(ToolError):
        await o2server.mcp.call_tool(
            "o2_exec",
            {
                "params": {
                    "command": "sleep 280; sacct -j 51202268 --format=State,Elapsed -P -n",
                    "timeout_seconds": 290,
                }
            },
        )

    # Still ample for what the tool is actually for.
    inspection = await _call("o2_exec", {"params": {"command": "squeue -u me", "timeout_seconds": 30}})
    assert inspection["ok"] is True


@pytest.mark.anyio
async def test_invalid_input_is_rejected(monkeypatch, tmp_path):
    _patch_connection(monkeypatch, tmp_path, master=True)
    # Empty command violates min_length=1; the MCP layer must reject it.
    with pytest.raises(ToolError):
        await o2server.mcp.call_tool("o2_exec", {"params": {"command": ""}})


@pytest.mark.anyio
async def test_start_master_can_open_transfer_alias(monkeypatch, tmp_path):
    # The MCP tool must be able to open the transfer-node master, so transfer-node
    # moves (o2_run_promote/archive, o2_push/pull use_transfer_node) have a master
    # to reuse instead of hitting the transfer-master guard with no way to satisfy it.
    runner = _patch_connection(monkeypatch, tmp_path, master=False)
    status = await _call("o2_local_status", {})
    authorization = await _call(
        "o2_authorize_login",
        {
            "params": {
                "expected_revision": status["policy"]["revision"],
                "expected_generation": status["policy"]["generation"],
                "target": "transfer",
                "allow_offvpn": True,
                "approval_reference": "explicit transfer login approval",
            }
        },
    )
    res = await _call(
        "o2_start_master",
        {"params": {"grant_id": authorization["grant_id"], "transfer": True}},
    )
    assert res["ok"] is True and res["alias"] == "o2-transfer"
    mnf = [c for c in runner.calls if "-MNf" in c["argv"]]
    assert mnf and mnf[-1]["argv"][-1] == "o2-transfer"


# --- non-blocking transfers --------------------------------------------------
class _FakeProc:
    """Stand-in for the launched Popen: pid + poll() (None until the test finishes it)."""

    def __init__(self, pid):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode


def _patch_async(monkeypatch, tmp_path, *, master=True):
    """Patch _connection (master state) + O2AsyncTransfer to inject a fake spawner.

    Returns (launched, state_dir, procs): ``launched`` records each spawned argv and
    ``procs`` the FakeProcs returned, so a test asserts the wrapped rsync command and
    can mark a transfer finished — all without spawning a real process.
    """
    async_transfer._LIVE.clear()  # module-global registry: isolate per test
    _patch_connection(monkeypatch, tmp_path, master=master)
    launched: list[list[str]] = []
    procs: list[_FakeProc] = []
    state_dir = tmp_path / "astate"

    def spawner(argv, log_path):
        launched.append(list(argv))
        proc = _FakeProc(4321)
        procs.append(proc)
        return proc

    # the async tools live in transfer_tools and import O2AsyncTransfer there
    monkeypatch.setattr(
        transfer_tools,
        "O2AsyncTransfer",
        lambda conn: _RealAsyncTransfer(conn, state_dir=state_dir, spawner=spawner, clock=lambda: 1000.0),
    )
    return launched, state_dir, procs


@pytest.mark.anyio
async def test_push_async_returns_transfer_id_without_blocking(monkeypatch, tmp_path):
    launched, _, _ = _patch_async(monkeypatch, tmp_path, master=True)
    remote = "/n/groups/tabin/jzhao/o2_gem_diffusion/data/20260329 - 20nm GEM Human Mouse PSM/Human"
    payload = await _call("o2_push_async", {"params": {"local_path": "/local/Human", "remote_path": remote}})
    assert payload["ok"] is True
    assert payload["transfer_id"].startswith("push-")
    assert payload["pid"] == 4321
    # launched the detached bash-wrapped rsync, with the remote path escaped.
    assert launched and launched[0][0] == "bash"
    transport = launched[0][launched[0].index("-e") + 1]
    assert "PreferredAuthentications=none" in transport
    assert "PubkeyAuthentication=no" in transport
    assert launched[0][-1] == "o2-transfer:" + remote.replace(" ", "\\ ")


@pytest.mark.anyio
async def test_push_async_refuses_without_master(monkeypatch, tmp_path):
    launched, _, _ = _patch_async(monkeypatch, tmp_path, master=False)
    payload = await _call("o2_push_async", {"params": {"local_path": "/local/x", "remote_path": "/remote/x"}})
    assert payload["ok"] is False and payload["error"] == "no_master"
    assert launched == []  # nothing launched without an approved master


@pytest.mark.anyio
async def test_transfer_status_reports_done_and_lists(monkeypatch, tmp_path):
    _, _, procs = _patch_async(monkeypatch, tmp_path, master=True)
    started = await _call("o2_push_async", {"params": {"local_path": "/a", "remote_path": "/ra"}})
    tid = started["transfer_id"]
    # while the process is live, it reports running...
    assert (await _call("o2_transfer_status", {"params": {"transfer_id": tid}}))["state"] == "running"
    # ...and once it finishes (poll() returns the exit code), done.
    procs[0].returncode = 0
    one = await _call("o2_transfer_status", {"params": {"transfer_id": tid}})
    assert one["ok"] is True and one["state"] == "done" and one["returncode"] == 0
    # no id -> list of all transfers
    allof = await _call("o2_transfer_status", {"params": {}})
    assert allof["ok"] is True and len(allof["transfers"]) == 1


@pytest.fixture
def anyio_backend():
    return "asyncio"


# --- pre-submission pricing --------------------------------------------------
@pytest.mark.anyio
async def test_price_job_without_a_cache_says_how_to_get_one(monkeypatch, tmp_path):
    # Pricing must be usable before any connection exists, so a missing cache is
    # a normal state with a clear next step -- not an error about SSH.
    monkeypatch.setenv("O2_BILLING_WEIGHTS_CACHE", str(tmp_path / "absent.json"))
    payload = await _call("o2_price_job", {"params": {"partition": "short", "cpus": 4}})
    assert payload["ok"] is False
    assert payload["error"] == "no_weight_cache"
    assert "o2_refresh_billing_weights" in payload["message"]


@pytest.mark.anyio
async def test_price_job_reports_cheaper_partitions_for_the_same_request(monkeypatch, tmp_path):
    from o2mcp import billing

    monkeypatch.setenv("O2_BILLING_WEIGHTS_CACHE", str(tmp_path / "weights.json"))
    billing.save_weight_cache(
        billing.parse_weight_table(
            "PartitionName=gpu_quad TRESBillingWeights=CPU=1.0,Mem=0.0625G,GRES/gpu=5.0"
            " TRES=cpu=400,mem=4000G,node=10,gres/gpu=40\n"
            "PartitionName=gpu_requeue TRESBillingWeights=CPU=0.1,Mem=0.00625G,GRES/gpu=0.1"
            " TRES=cpu=1080,mem=10000G,node=27,gres/gpu=108\n"
        ),
        captured_at=1000.0,
        priority_flags=[],
    )
    payload = await _call(
        "o2_price_job",
        {"params": {"partition": "gpu_quad", "cpus": 4, "mem_gb": 6, "gpus": 1}},
    )
    assert payload["billing_units"] == 9
    assert payload["alternatives"][0]["partition"] == "gpu_requeue"
    assert payload["alternatives"][0]["units"] < 9


@pytest.mark.anyio
async def test_price_job_direct_call_without_memory_uses_the_partition_default(monkeypatch, tmp_path):
    # `cpus=4` with no mem_gb is a request that names no memory, not a request
    # for none: Slurm applies DefMemPerCPU and bills it.
    from o2mcp import billing

    monkeypatch.setenv("O2_BILLING_WEIGHTS_CACHE", str(tmp_path / "w.json"))
    billing.save_weight_cache(
        billing.parse_weight_table("PartitionName=short TRESBillingWeights=CPU=1.0,Mem=0.0625G DefMemPerCPU=4096\n"),
        captured_at=1000.0,
        priority_flags=[],
    )
    payload = await _call("o2_price_job", {"params": {"partition": "short", "cpus": 4}})
    assert payload["ok"] is True
    assert payload["request"]["mem_gb"] == pytest.approx(16.0)
    assert "partition default" in payload["request"]["mem_source"]
    assert payload["billing_units"] == 5


@pytest.mark.anyio
async def test_price_job_refuses_an_unknown_partition(monkeypatch, tmp_path):
    from o2mcp import billing

    monkeypatch.setenv("O2_BILLING_WEIGHTS_CACHE", str(tmp_path / "weights.json"))
    billing.save_weight_cache(
        billing.parse_weight_table("PartitionName=short TRESBillingWeights=CPU=1.0,Mem=0.0625G\n"),
        captured_at=1000.0,
        priority_flags=[],
    )
    payload = await _call("o2_price_job", {"params": {"partition": "made_up", "cpus": 1}})
    assert payload["ok"] is False
    assert payload["error"] == "unpriceable"


@pytest.mark.anyio
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
async def test_price_job_registers_under_python_dash_m(monkeypatch):
    # A @mcp.tool below the `if __name__ == "__main__"` block never executes
    # under `python -m o2mcp.server`: main() blocks in mcp.run() first, so the
    # tool is absent for the server's whole lifetime. Import-based entry points
    # and tests mask it, because they finish importing before calling main.
    import runpy

    from mcp.server.fastmcp import FastMCP

    served: dict = {}

    def fake_run(self, *args, **kwargs):
        served["tools"] = {t.name for t in self._tool_manager.list_tools()}

    monkeypatch.setattr(FastMCP, "run", fake_run, raising=False)
    runpy.run_module("o2mcp.server", run_name="__main__", alter_sys=True)
    assert "o2_price_job" in served.get("tools", set())


@pytest.mark.anyio
async def test_pricing_never_writes_the_weight_cache(tmp_path, monkeypatch):
    """The readOnly claim, checked against behaviour rather than the annotation."""
    cache = tmp_path / "weights.json"
    monkeypatch.setenv("O2_BILLING_WEIGHTS_CACHE", str(cache))
    billing.save_weight_cache(
        {"short": billing.Weights(cpu=1.0, mem_per_gb=0.0625)}, 1.0, str(cache), priority_flags=[]
    )
    before = cache.read_bytes()
    payload = json.loads(await o2server.o2_price_job(o2server.PriceJobInput(partition="short", cpus=2, mem_gb=16)))
    assert payload["ok"] is True
    assert cache.read_bytes() == before


@pytest.mark.anyio
async def test_price_job_says_when_alternatives_were_withheld(tmp_path, monkeypatch):
    """An empty alternatives list is not the same claim as "nothing is cheaper"."""
    monkeypatch.setenv("O2_BILLING_WEIGHTS_CACHE", str(tmp_path / "w.json"))
    billing.save_weight_cache(
        billing.parse_weight_table(
            "PartitionName=short TRESBillingWeights=CPU=1.0,Mem=0.0625G"
            " TRES=cpu=4000,mem=40000G,node=10\n"
            "PartitionName=cheap TRESBillingWeights=CPU=0.1,Mem=0.00625G"
            " TRES=cpu=128,mem=256G,node=2\n"
        ),
        captured_at=1000.0,
        priority_flags=[],
    )
    payload = json.loads(
        await o2server.o2_price_job(o2server.PriceJobInput(partition="short", cpus=64, mem_gb=128, nodes=1))
    )
    assert payload["ok"] is True
    assert payload["alternatives"] == []
    assert "SINGLE node" in payload["alternatives_note"]

    # And with no node count pinned, the comparison happens and no note is set.
    plain = json.loads(await o2server.o2_price_job(o2server.PriceJobInput(partition="short", cpus=64, mem_gb=128)))
    assert plain["alternatives"]
    assert "alternatives_note" not in plain
