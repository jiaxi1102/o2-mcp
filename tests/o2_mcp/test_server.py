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
from pydantic import ValidationError  # noqa: E402

from o2mcp import (  # noqa: E402
    CommandResult,
    O2BrokerBusyError,  # noqa: E402
    O2BrokerCommandOutcomeUnknownError,  # noqa: E402
    O2Config,
    async_transfer,  # noqa: E402
    billing,  # noqa: E402
    transfer_tools,  # noqa: E402
)
from o2mcp import O2AsyncTransfer as _RealAsyncTransfer  # noqa: E402
from o2mcp import (
    O2Connection as _ProductionO2Connection,
)
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
        # governed launch attestation
        "o2_mint_launch_evidence",
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
    # Minting reads cluster artifacts and appends one audit event, so it is not
    # read-only; but it publishes nothing and removes nothing, so a client must
    # not be told it is destructive either.
    assert tools["o2_mint_launch_evidence"].annotations.readOnlyHint is False
    assert tools["o2_mint_launch_evidence"].annotations.destructiveHint is False
    assert tools["o2_mint_launch_evidence"].annotations.openWorldHint is False
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
async def test_price_job_always_labels_alternatives_as_prices(tmp_path, monkeypatch):
    """The rows are a price comparison, and every response has to say so.

    Whether a job may actually run on a partition is Slurm's admission control,
    which the cached partition configuration cannot settle -- so a caller
    reading these as "partitions that can run this" has been told something
    this tool does not know.
    """
    monkeypatch.setenv("O2_BILLING_WEIGHTS_CACHE", str(tmp_path / "w.json"))
    billing.save_weight_cache(
        billing.parse_weight_table(
            "PartitionName=short TRESBillingWeights=CPU=1.0,Mem=0.0625G"
            " TRES=cpu=4000,mem=40000G,node=10\n"
            "PartitionName=cheap TRESBillingWeights=CPU=0.1,Mem=0.00625G"
            " TRES=cpu=4000,mem=40000G,node=10\n"
        ),
        captured_at=1000.0,
        priority_flags=[],
    )
    payload = json.loads(await o2server.o2_price_job(o2server.PriceJobInput(partition="short", cpus=64, mem_gb=128)))
    assert payload["ok"] is True
    assert payload["alternatives"]
    assert "NOT verified" in payload["alternatives_note"]

    # Present even when nothing survived the filters, so an empty list is not
    # read as "nothing cheaper exists" either.
    pinned = json.loads(
        await o2server.o2_price_job(o2server.PriceJobInput(partition="short", cpus=4, mem_gb=16, nodes=1))
    )
    assert "NOT verified" in pinned["alternatives_note"]


@pytest.mark.anyio
async def test_price_job_prices_a_cpu_only_partition_through_the_tool(tmp_path, monkeypatch):
    """Through o2_price_job, not through resolve_request alone.

    The unit tests for this exercised resolve_request directly and passed while
    the tool still returned unpriceable: price() resolves a second time, and
    the mem_unknown sentinel read as an explicit --mem=0 on that pass. Only a
    test at this level sees the whole path.
    """
    monkeypatch.setenv("O2_BILLING_WEIGHTS_CACHE", str(tmp_path / "w.json"))
    billing.save_weight_cache(
        billing.parse_weight_table(
            "PartitionName=plain State=UP AllowGroups=ALL TotalCPUs=400 TotalNodes=10\n"
            "PartitionName=billed TRESBillingWeights=CPU=0.1,Mem=0.00625G State=UP"
            " AllowGroups=ALL TotalCPUs=400 TotalNodes=10\n"
        ),
        captured_at=1000.0,
        priority_flags=[],
    )
    payload = json.loads(await o2server.o2_price_job(o2server.PriceJobInput(partition="plain", cpus=8)))
    assert payload["ok"] is True
    assert payload["billing_units"] == 8
    assert payload["request"]["mem_source"] == "not billed on plain"
    # The comparison is withheld, and the response says which reason applies.
    assert payload["alternatives"] == []
    assert "does not bill memory" in payload["alternatives_note"]


@pytest.mark.anyio
async def test_price_job_still_refuses_an_explicit_zero(tmp_path, monkeypatch):
    """The sentinel must not become a way for a real --mem=0 to slip through."""
    monkeypatch.setenv("O2_BILLING_WEIGHTS_CACHE", str(tmp_path / "w.json"))
    billing.save_weight_cache(
        billing.parse_weight_table(
            "PartitionName=p TRESBillingWeights=CPU=1,Mem=0.0625G State=UP"
            " AllowGroups=ALL TotalCPUs=400 TotalNodes=10"
        ),
        captured_at=1000.0,
        priority_flags=[],
    )
    payload = json.loads(await o2server.o2_price_job(o2server.PriceJobInput(partition="p", cpus=8, mem_gb=0)))
    assert payload["ok"] is False
    assert payload["error"] == "unpriceable"
    assert "all memory on every allocated node" in payload["message"]


@pytest.mark.anyio
async def test_the_price_job_schema_states_what_its_numbers_mean():
    """The schema is what an agent reads at the call site.

    Lives here, not in tests/unit: that suite is dependency-free and runs on a
    Python where the MCP extras are not installed, so importing the server
    there broke the whole lane.
    """
    mem = o2server.PriceJobInput.model_fields["mem_gb"].description
    assert "TOTAL" in mem
    assert "per NODE" in mem
    nodes = o2server.PriceJobInput.model_fields["nodes"].description
    assert "whenever the submission" in nodes
    assert "MaxNodes" in nodes
    cpus = o2server.PriceJobInput.model_fields["cpus"].description
    assert "Total CPUs" in cpus


@pytest.mark.anyio
async def test_price_job_rejects_non_finite_numbers(tmp_path, monkeypatch):
    """A JSON number that overflows to inf satisfied gt/ge and then hit int().

    OverflowError is not BillingError, so it escaped the handler and the tool
    call crashed rather than answering.
    """
    monkeypatch.setenv("O2_BILLING_WEIGHTS_CACHE", str(tmp_path / "w.json"))
    billing.save_weight_cache(
        billing.parse_weight_table(
            "PartitionName=short TRESBillingWeights=CPU=1.0,Mem=0.0625G" " TRES=cpu=400,mem=4000G,node=10"
        ),
        captured_at=1000.0,
        priority_flags=[],
    )
    for field in ("cpus", "mem_gb", "gpus", "nodes"):
        with pytest.raises(ValidationError):
            o2server.PriceJobInput(**{"partition": "short", "cpus": 4, field: float("inf")})
    with pytest.raises(ValidationError):
        o2server.PriceJobInput(partition="short", cpus=float("nan"))
    # A finite request still prices.
    payload = json.loads(await o2server.o2_price_job(o2server.PriceJobInput(partition="short", cpus=4, mem_gb=16)))
    assert payload["ok"] is True


def test_submitting_without_a_price_names_the_flags_it_saw():
    """A warning that fires on every submission is noise, so it must be specific.

    Presence of a resource flag is safe to detect; its VALUE is the sbatch
    parser o2_price_job deliberately does not have.
    """
    params = o2server.SubmitInput(
        script_text="#!/bin/bash\n#SBATCH --mem=32G\n#SBATCH -c 4\n#SBATCH --gres=gpu:1\nsrun x\n",
        remote_path="/n/scratch/x.sh",
    )
    record = o2server._pricing_record(params)
    assert record["priced"] is False
    assert record["resource_flags_seen"] == ["--gres", "--mem", "-c"]
    assert "carries no price" in record["note"]


def test_a_submission_with_no_resource_flags_is_not_nagged():
    params = o2server.SubmitInput(
        script_text="#!/bin/bash\n#SBATCH --job-name=x\nsrun hostname\n",
        remote_path="/n/scratch/x.sh",
    )
    record = o2server._pricing_record(params)
    assert record["resource_flags_seen"] == []
    assert "note" not in record


def test_an_unreadable_remote_script_says_so_rather_than_guessing():
    """A script whose directives could not be read is UNKNOWN, not resourceless.

    Saying it has no resource flags would be a claim the call cannot support.
    """
    record = o2server._pricing_record(
        o2server.SubmitInput(remote_script_path="/n/scratch/x.sh"), remote_directives=None
    )
    assert record["priced"] is False
    assert "could not be read" in record["note"]


def test_a_remote_script_gets_the_same_check_as_an_inlined_one():
    # The directives come back from one cheap grep, so a submission by path is
    # no longer a blind spot.
    record = o2server._pricing_record(
        o2server.SubmitInput(remote_script_path="/n/scratch/x.sh"),
        remote_directives=["#SBATCH --mem=64G", "#SBATCH --gres=gpu:1", "#SBATCH -t 4:00:00"],
    )
    assert record["resource_flags_seen"] == ["--gres", "--mem"]
    assert "carries no price" in record["note"]


def test_a_readable_remote_script_with_no_resource_flags_is_quiet():
    record = o2server._pricing_record(
        o2server.SubmitInput(remote_script_path="/n/scratch/x.sh"),
        remote_directives=["#SBATCH --job-name=x", "#SBATCH -t 1:00:00"],
    )
    assert record["resource_flags_seen"] == []
    assert "note" not in record


# Every sbatch option that changes what a job is allocated, from the sbatch(1)
# option list. Kept as data so the audit is repeatable: a flag added to sbatch,
# or dropped from _RESOURCE_FLAGS, shows up here rather than as a silent gap.
_RESOURCE_FLAGS = o2server._RESOURCE_FLAGS

_ALLOCATION_OPTIONS = (
    "-n",
    "--ntasks",
    "-c",
    "--cpus-per-task",
    "--ntasks-per-node",
    "--ntasks-per-socket",
    "--ntasks-per-core",
    "--ntasks-per-gpu",
    "-N",
    "--nodes",
    "--overcommit",
    "--exclusive",
    "--mem",
    "--mem-per-cpu",
    "--mem-per-gpu",
    "-G",
    "--gpus",
    "--gpus-per-node",
    "--gpus-per-socket",
    "--gpus-per-task",
    "--gres",
    "--cpus-per-gpu",
    "-B",
    "--extra-node-info",
    "--sockets-per-node",
    "--cores-per-socket",
    "--threads-per-core",
    "--mincpus",
    "--tres-per-task",
    "--core-spec",
    "--thread-spec",
    # Not a size, but it picks the weights the size is priced with, so a
    # submission naming one and carrying no price is the same warning.
    "-p",
    "--partition",
    # TRES with no input on o2_price_job at all: a weighted site bills for
    # these and nothing here can ask about them.
    "--licenses",
    "-L",
    "--bb",
    "--bbf",
    # Not a resource: a QoS UsageFactor multiplies the charge itself.
    "--qos",
    "-q",
    # An explicit node list sets the node count when nothing else does.
    "--nodelist",
    "-w",
    "--nodefile",
    "-F",
)


def test_no_allocation_option_goes_unnoticed():
    """Each of these is either a resource flag or an unpriceable one.

    A flag in neither list is a job recorded as requesting nothing -- a MISSED
    warning. These were found nine at a time by auditing the option list, after
    being reported one at a time.
    """
    known = set(_RESOURCE_FLAGS) | set(billing.UNPRICEABLE_OPTIONS) | set(billing.UNPRICEABLE_ALIASES.values())
    assert [opt for opt in _ALLOCATION_OPTIONS if opt not in known] == []


def test_an_allocation_option_is_reported_however_it_is_written():
    # Through the scanner, not just the tuple: each has to survive tokenising.
    for option in _ALLOCATION_OPTIONS:
        # --exclusive takes only user/mcs/topo, and a SCOPED one is
        # deliberately not flagged -- it allocates what was asked for. Only the
        # bare spelling is the whole-node case.
        spellings = (option,) if option == "--exclusive" else (option, f"{option}=1", f"{option} 1")
        for directive in spellings:
            params = o2server.SubmitInput(script_text=f"#!/bin/bash\n#SBATCH {directive}\n", remote_path="/n/x.sh")
            reported = o2server._resource_flags_seen(params) + o2server._unpriceable_options_seen(params)
            assert reported, directive


def test_a_hash_inside_a_directive_value_is_not_a_comment():
    # sbatch reads a directive's arguments directly, so `#` in a value is part
    # of it. Splitting on comments ended the line at the hash and silently
    # dropped every option after it.
    params = o2server.SubmitInput(
        script_text="#!/bin/bash\n#SBATCH --comment=issue#123 --exclusive\n", remote_path="/n/x.sh"
    )
    assert o2server._unpriceable_options_seen(params) == ["--exclusive"]
    params = o2server.SubmitInput(
        script_text="#!/bin/bash\n#SBATCH --comment=issue#123 --mem=64G\n", remote_path="/n/x.sh"
    )
    assert o2server._resource_flags_seen(params) == ["--mem"]


def test_a_disabled_directive_is_not_a_directive():
    """`#SBATCH_DISABLED` is how a directive gets switched off.

    Matching the marker without a boundary read it as live and, after slicing
    seven characters, reported the option it was meant to disable.
    """
    for marker in ("#SBATCH_DISABLED", "#SBATCHFOO", "#SBATCH-OLD"):
        params = o2server.SubmitInput(script_text=f"#!/bin/bash\n{marker} --exclusive\n", remote_path="/n/x.sh")
        assert o2server._unpriceable_options_seen(params) == [], marker
    # The real marker still works, with any leading or extra whitespace.
    for line in ("#SBATCH --exclusive", "  #SBATCH --exclusive", "#SBATCH\t--exclusive"):
        params = o2server.SubmitInput(script_text=f"#!/bin/bash\n{line}\n", remote_path="/n/x.sh")
        assert o2server._unpriceable_options_seen(params) == ["--exclusive"], line


def test_a_colon_separated_hetjob_is_recognised():
    # Slurm separates heterogeneous components with `hetjob` OR a lone `:`,
    # and the wrapper forwards that colon through as its own argument.
    params = o2server.SubmitInput(
        script_text="#!/bin/bash\n", remote_path="/n/x.sh", sbatch_args=["--nodes=1", ":", "--nodes=2"]
    )
    assert o2server._unpriceable_options_seen(params) == ["hetjob"]
    # A colon INSIDE a value is not a separator.
    plain = o2server.SubmitInput(script_text="#!/bin/bash\n#SBATCH --gres=gpu:1\n", remote_path="/n/x.sh")
    assert o2server._unpriceable_options_seen(plain) == []
    assert o2server._resource_flags_seen(plain) == ["--gres"]


def test_an_explicit_node_list_sizes_the_allocation():
    def mk(directive):
        return o2server.SubmitInput(script_text=f"#!/bin/bash\n#SBATCH {directive}\n", remote_path="/n/x.sh")

    assert o2server._resource_flags_seen(mk("--nodelist=node[01-04]")) == ["--nodelist"]
    assert o2server._resource_flags_seen(mk("-w node01")) == ["-w"]
    assert o2server._resource_flags_seen(mk("--nodefile=f.txt")) == ["--nodefile"]
    # --exclude removes candidates without changing how much is allocated.
    assert o2server._resource_flags_seen(mk("--exclude=node09")) == []


def test_oversubscribe_is_not_a_resource_request():
    # It permits sharing a node; it does not grow the allocation. Warning about
    # it would send someone to price a shape that has not changed.
    for directive in ("--oversubscribe", "-s"):
        params = o2server.SubmitInput(script_text=f"#!/bin/bash\n#SBATCH {directive}\n", remote_path="/n/x.sh")
        assert o2server._resource_flags_seen(params) == [], directive
        assert o2server._unpriceable_options_seen(params) == [], directive


def test_the_short_gpu_flag_is_a_resource_request():
    # -G is --gpus. Missing it recorded a GPU job as carrying no resource
    # flags at all -- a missed warning on the most expensive TRES there is.
    def mk(directive):
        return o2server.SubmitInput(script_text=f"#!/bin/bash\n#SBATCH {directive}\n", remote_path="/n/x.sh")

    assert o2server._resource_flags_seen(mk("-G1")) == ["-G"]
    assert o2server._resource_flags_seen(mk("-G 2")) == ["-G"]
    assert o2server._resource_flags_seen(mk("--gpus=1")) == ["--gpus"]
    # Lowercase -g is not an sbatch resource flag.
    assert o2server._resource_flags_seen(mk("-g x")) == []


def test_short_aliases_of_unpriceable_options_are_caught():
    """`-O` is --overcommit and `-B` is --extra-node-info, per sbatch(1).

    Checking only the long spelling let either through with no warning -- a
    MISSED warning, which is the direction that matters.
    """

    def mk(directive):
        return o2server.SubmitInput(script_text=f"#!/bin/bash\n#SBATCH {directive}\n", remote_path="/n/x.sh")

    assert o2server._unpriceable_options_seen(mk("-O")) == ["--overcommit"]
    for spelling in ("-B2:8:2", "-B 2:8:2", "--extra-node-info=2:8:2"):
        assert o2server._unpriceable_options_seen(mk(spelling)) == ["--extra-node-info"], spelling
    # Case matters: -o is --output and -b is --begin, neither unpriceable.
    assert o2server._unpriceable_options_seen(mk("-o out.txt")) == []
    assert o2server._unpriceable_options_seen(mk("-b now+1hour")) == []


def test_every_alias_names_an_option_in_the_table():
    # The alias map sits beside UNPRICEABLE_OPTIONS; a typo there would fail
    # silently, since the scanner only ever looks aliases up by long name.
    for option in billing.UNPRICEABLE_ALIASES:
        assert option in billing.UNPRICEABLE_OPTIONS, option


def test_a_directive_below_the_script_body_is_inert():
    """sbatch stops reading directives at the first line of actual code.

    A #SBATCH sitting under the script's commands is never applied, so warning
    about it describes an option the job does not have.
    """
    below = o2server.SubmitInput(
        script_text="#!/bin/bash\n#SBATCH --mem=4G\n\nsrun ./run.sh\n#SBATCH --exclusive\n",
        remote_path="/n/x.sh",
    )
    assert o2server._unpriceable_options_seen(below) == []
    assert o2server._resource_flags_seen(below) == ["--mem"]
    # Comments and blank lines do not end the block, so this one IS applied.
    above = o2server.SubmitInput(
        script_text="#!/bin/bash\n# a note\n\n#SBATCH --exclusive\nsrun ./run.sh\n",
        remote_path="/n/x.sh",
    )
    assert o2server._unpriceable_options_seen(above) == ["--exclusive"]


def test_a_scoped_exclusive_is_not_warned_about():
    """--exclusive=user does NOT take the whole node, so it prices normally.

    sbatch(1) applies the whole-node rule only "if user/mcs/topo are not
    specified". Warning about the scoped forms would talk a reader out of an
    option that costs them nothing extra.
    """

    def mk(directive):
        return o2server.SubmitInput(script_text=f"#!/bin/bash\n#SBATCH {directive}\n", remote_path="/n/x.sh")

    assert o2server._unpriceable_options_seen(mk("--exclusive")) == ["--exclusive"]
    for scope in ("user", "mcs", "topo"):
        assert o2server._unpriceable_options_seen(mk(f"--exclusive={scope}")) == [], scope


def test_an_option_named_inside_another_options_value_is_not_set():
    # submit() quotes each sbatch_args element into one argument, so this is a
    # --comment whose text mentions --exclusive, not a job taking whole nodes.
    params = o2server.SubmitInput(
        script_text="#!/bin/bash\n", remote_path="/n/x.sh", sbatch_args=["--comment=do not use --exclusive"]
    )
    assert o2server._unpriceable_options_seen(params) == []
    assert o2server._resource_flags_seen(params) == []
    # ...but a real option beside it still registers.
    params = o2server.SubmitInput(
        script_text="#!/bin/bash\n", remote_path="/n/x.sh", sbatch_args=["--comment=see --mem notes", "--mem=64G"]
    )
    assert o2server._resource_flags_seen(params) == ["--mem"]


def test_a_malformed_receipt_cannot_silence_an_unpriceable_option():
    # The valid-receipt path was fixed first; a garbage string reached the same
    # silence through the malformed-receipt branch.
    record = o2server._pricing_record(
        o2server.SubmitInput(
            script_text="#!/bin/bash\n#SBATCH --exclusive\n", remote_path="/n/x.sh", priced="typo-not-a-receipt"
        )
    )
    assert record["unpriceable_options_seen"] == ["--exclusive"]
    assert "cannot price these" in record["note"]
    # And it still says the receipt was unreadable.
    assert "not a recognisable" in record["note"]


def test_a_receipt_cannot_silence_an_unpriceable_option():
    """Passing a valid receipt must not buy silence about --exclusive.

    o2_price_job refuses every option in that table, so a receipt necessarily
    describes some OTHER shape. Returning early on any valid receipt made the
    warning silenceable by supplying an unrelated one.
    """
    table = billing.parse_weight_table(
        "PartitionName=short TRESBillingWeights=CPU=1,Mem=0.0625G"
        " TRES=cpu=400,mem=4000G,node=10 State=UP AllowGroups=ALL"
    )
    receipt = billing.price_receipt(billing.price(billing.Request(cpus=4, mem_gb=16), table, "short"))
    record = o2server._pricing_record(
        o2server.SubmitInput(script_text="#!/bin/bash\n#SBATCH --exclusive\n", remote_path="/n/x.sh", priced=receipt)
    )
    # `priced` answers "was THIS shape priced?" -- and it provably was not,
    # however valid the receipt is. A client gating on the boolean must not
    # read a receipt for some other shape as this job having a price.
    assert record["priced"] is False
    assert record["receipt"] is not None
    assert record["unpriceable_options_seen"] == ["--exclusive"]
    assert "describes a different shape" in record["note"]


def test_a_receipt_does_not_turn_an_unreadable_script_into_a_clean_one():
    """Unknown is not absent -- including on the receipt path.

    A receipt says a price was obtained. It cannot say the script has no
    option that would invalidate it, and here the script could not be read.
    """
    table = billing.parse_weight_table(
        "PartitionName=short TRESBillingWeights=CPU=1,Mem=0.0625G"
        " TRES=cpu=400,mem=4000G,node=10 State=UP AllowGroups=ALL"
    )
    receipt = billing.price_receipt(billing.price(billing.Request(cpus=4, mem_gb=16), table, "short"))
    record = o2server._pricing_record(
        o2server.SubmitInput(remote_script_path="/n/scratch/x.sh", priced=receipt), remote_directives=None
    )
    assert record["priced"] is True
    assert "could not be read" in record["note"]
    # A script that WAS read and sets nothing stays quiet.
    quiet = o2server._pricing_record(
        o2server.SubmitInput(remote_script_path="/n/scratch/x.sh", priced=receipt),
        remote_directives=["#SBATCH -t 1:00:00"],
    )
    assert "note" not in quiet


def test_an_ordinary_receipt_still_records_nothing_extra():
    # The warning above must not fire for a submission that has no such option.
    table = billing.parse_weight_table(
        "PartitionName=short TRESBillingWeights=CPU=1,Mem=0.0625G"
        " TRES=cpu=400,mem=4000G,node=10 State=UP AllowGroups=ALL"
    )
    receipt = billing.price_receipt(billing.price(billing.Request(cpus=4, mem_gb=16), table, "short"))
    record = o2server._pricing_record(
        o2server.SubmitInput(script_text="#!/bin/bash\n#SBATCH --mem=4G\n", remote_path="/n/x.sh", priced=receipt)
    )
    assert record["priced"] is True
    assert "unpriceable_options_seen" not in record
    assert "note" not in record


def test_receipt_counts_read_back_as_whole_numbers():
    # They land in the submission JSON; "4.0 CPUs" invites a wrong question.
    back = billing.parse_price_receipt("o2price/1 partition=short cpus=4 mem_gb=16 gpus=0 units=5")
    assert (back["cpus"], back["gpus"], back["units"]) == (4, 0, 5)
    assert all(isinstance(back[k], int) for k in ("cpus", "gpus", "units"))
    # mem_gb stays float: 0.25 GB is an ordinary 256 MB request.
    assert isinstance(back["mem_gb"], float)


def test_an_exclusive_script_is_not_silent():
    """--exclusive bills the whole node, and set alone it drew no warning.

    The scan looked only for resource flags, so the single most expensive
    directive in sbatch produced an empty list and no note at all.
    """
    params = o2server.SubmitInput(script_text="#!/bin/bash\n#SBATCH --exclusive\n", remote_path="/n/x.sh")
    record = o2server._pricing_record(params)
    assert record["resource_flags_seen"] == []
    assert record["unpriceable_options_seen"] == ["--exclusive"]
    # sbatch(1) is specific: all CPUs and GRES on the node, memory as requested.
    assert "every CPU and GRES on the nodes it lands on" in record["note"]
    assert "memory is still billed as requested" in record["note"]
    # Not the ordinary advice: o2_price_job answers `unpriceable` for these.
    assert "cannot price these" in record["note"]


def test_every_unpriceable_option_is_covered_by_the_scan():
    # Derived from the table, so an option added there needs no edit here.
    for option in billing.UNPRICEABLE_OPTIONS:
        text = "#!/bin/bash\n#SBATCH " + ("hetjob" if option == "hetjob" else f"{option}\n")
        params = o2server.SubmitInput(script_text=text, remote_path="/n/x.sh")
        assert option in o2server._unpriceable_options_seen(params), option


def test_an_ordinary_mem_request_is_not_called_unpriceable():
    # --mem=0 means every byte on every node; --mem=4G is just a request.
    def mk(directive):
        return o2server.SubmitInput(script_text=f"#!/bin/bash\n#SBATCH {directive}\n", remote_path="/n/x.sh")

    assert o2server._unpriceable_options_seen(mk("--mem=4G")) == []
    assert o2server._unpriceable_options_seen(mk("--mem=0")) == ["--mem=0"]
    assert o2server._unpriceable_options_seen(mk("--mem=0G")) == ["--mem=0"]
    # sbatch takes the value attached or separated, and both mean every byte.
    assert o2server._unpriceable_options_seen(mk("--mem 0")) == ["--mem=0"]
    assert o2server._unpriceable_options_seen(mk("--mem 0GB")) == ["--mem=0"]
    assert o2server._unpriceable_options_seen(mk("--mem 4G")) == []
    # And --mem must not match inside --mem-per-cpu either way.
    assert o2server._unpriceable_options_seen(mk("--mem-per-cpu 0")) == []


def test_unpriceable_options_are_found_in_a_remote_script_too():
    record = o2server._pricing_record(
        o2server.SubmitInput(remote_script_path="/n/scratch/x.sh"),
        remote_directives=["#SBATCH --exclusive", "#SBATCH -t 8:00:00"],
    )
    assert record["unpriceable_options_seen"] == ["--exclusive"]


def test_only_the_script_that_will_run_is_inspected():
    """script_text wins over remote_script_path, so the record follows it.

    SubmitInput permits both. The submit path sends the text; a record built
    from the OTHER script's directives would list flags this job does not set.
    """
    record = o2server._pricing_record(
        o2server.SubmitInput(
            script_text="#!/bin/bash\n#SBATCH --job-name=x\n",
            remote_path="/n/x.sh",
            remote_script_path="/n/scratch/other.sh",
        ),
        remote_directives=["#SBATCH --mem=64G", "#SBATCH --gres=gpu:1"],
    )
    assert record["resource_flags_seen"] == []
    assert "note" not in record


def test_an_unreadable_path_is_not_reported_when_text_is_what_ships():
    # The submitted script was read perfectly well -- it is right here. Calling
    # it unknown because a displaced path could not be read is a false alarm.
    record = o2server._pricing_record(
        o2server.SubmitInput(
            script_text="#!/bin/bash\n#SBATCH --job-name=x\n",
            remote_path="/n/x.sh",
            remote_script_path="/n/scratch/gone.sh",
        ),
        remote_directives=None,
    )
    assert "note" not in record


def test_submitted_remote_path_follows_the_submit_precedence():
    # The predicate itself, so the read and the record cannot drift apart.
    assert o2server._submitted_remote_path(o2server.SubmitInput(remote_script_path="/n/a.sh")) == "/n/a.sh"
    both = o2server.SubmitInput(script_text="#!/bin/bash\n", remote_path="/n/x.sh", remote_script_path="/n/a.sh")
    assert o2server._submitted_remote_path(both) is None
    # An empty script_text is still what submit_text() sends, so it still wins.
    empty = o2server.SubmitInput(script_text="", remote_path="/n/x.sh", remote_script_path="/n/a.sh")
    assert o2server._submitted_remote_path(empty) is None


def test_a_receipt_is_recorded_beside_the_job():
    table = billing.parse_weight_table(
        "PartitionName=short TRESBillingWeights=CPU=1,Mem=0.0625G"
        " TRES=cpu=400,mem=4000G,node=10 State=UP AllowGroups=ALL"
    )
    payload = billing.price(billing.Request(cpus=4, mem_gb=16), table, "short")
    params = o2server.SubmitInput(remote_script_path="/n/scratch/x.sh", priced=billing.price_receipt(payload))
    record = o2server._pricing_record(params)
    assert record["priced"] is True
    assert record["receipt"]["partition"] == "short"
    assert record["receipt"]["units"] == payload["billing_units"]


def test_an_unrecognised_priced_value_is_reported_not_swallowed():
    record = o2server._pricing_record(
        o2server.SubmitInput(remote_script_path="/n/x.sh", priced="hand-written nonsense")
    )
    assert record["priced"] is False
    assert "not a recognisable" in record["note"]


def test_short_flags_do_not_match_inside_long_ones():
    # "-n" must not fire on "--nodes", or the warning names flags that are not
    # there and stops being worth reading.
    params = o2server.SubmitInput(script_text="#!/bin/bash\n#SBATCH --nodes=2\n", remote_path="/n/x.sh")
    assert o2server._pricing_record(params)["resource_flags_seen"] == ["--nodes"]


def test_short_flags_with_attached_values_are_detected():
    # "-c4", "-N2" and "-pshort" are ordinary sbatch. Requiring whitespace or
    # "=" after a short flag missed all of them, so a fully specified
    # submission looked resourceless and the warning stayed silent.
    params = o2server.SubmitInput(
        script_text="#!/bin/bash\n#SBATCH -c4\n#SBATCH -N2\n#SBATCH -pshort\n",
        remote_path="/n/x.sh",
    )
    assert o2server._pricing_record(params)["resource_flags_seen"] == ["-N", "-c", "-p"]


def test_long_flags_still_end_at_the_token():
    # The attached-value allowance must not let "--mem" match "--mem-per-cpu".
    params = o2server.SubmitInput(script_text="#!/bin/bash\n#SBATCH --mem-per-cpu=4G\n", remote_path="/n/x.sh")
    assert o2server._pricing_record(params)["resource_flags_seen"] == ["--mem-per-cpu"]


# --- governed launch attestation ---------------------------------------------
_PACKAGE = "/pkg/attempt-002"
_PAYLOADS = {"payloads/frame 001.ims": "1" * 64, "payloads/frame002.ims": "2" * 64}


def _launch_evidence_responder(job_id="52085188", *, package=_PACKAGE, payloads=None, approved_package=None):
    """Serve the artifacts and the payload digests the mint reads off the cluster.

    ``package`` is where the artifacts actually live, ``approved_package`` is what
    the plan and the diagnostic name; they differ only when a test is checking
    that a substituted directory is refused.
    """

    import json as _json

    from o2mcp.launch_evidence import plan_digest

    on_disk = dict(_PAYLOADS if payloads is None else payloads)
    named = package if approved_package is None else approved_package
    plan = {
        "attempt_id": "002",
        "software": {"bundle": {"bundle_sha256": "b" * 64}},
        "runtime_wrapper": {"sha256": "w" * 64},
        "interpreter": {"sha256": "i" * 64, "closure_sha256": "c" * 64, "context_sha256": "x" * 64},
        "destination": {"inode": 42, "mount": "/n/scratch", "expected_package": named},
    }
    diagnostic = {
        "status": "diagnostic_success",
        "continuation_authorized": False,
        "schema_version": 2,
        "plan_sha256": plan_digest(plan),
        "attempt_id": "002",
        "runtime": {
            "source_bundle": {"bundle_sha256": "b" * 64},
            "runtime_wrapper_sha256": "w" * 64,
            "approved_interpreter": {
                "approved_path": "/usr/bin/python3",
                "sha256": "i" * 64,
                "runtime_context_sha256": "x" * 64,
                "dynamic_closure": {"closure_sha256": "c" * 64},
            },
        },
        "destination_binding": {
            "inode": 42,
            "linux_mountinfo": {"mount_point": "/n/scratch", "mount_source": "server:/scratch"},
        },
        "slurm": {
            "scheduler": {"job_id": job_id, "allocated_hostnames": ["compute-b-16-192"]},
            "environment": {"SLURM_JOB_ACCOUNT": "tabin", "SLURM_JOB_PARTITION": "short"},
        },
        "launch": {"loaded_library_closure": {"loaded_closure_sha256": "l" * 64}},
        "output": {
            "package": named,
            "reopened_output_sha256": "r" * 64,
            "verification": {"status": "success", "n_payloads": 9},
        },
    }
    owner = {"plan_sha256": plan_digest(plan), "attempt_id": "002"}
    manifest = "\n".join(f"{digest}  {name}" for name, digest in sorted(_PAYLOADS.items()))
    digests = "\n".join(
        f"{'d' * 64}  {package}/{name}"
        for name in sorted(("PUBLICATION_OWNER.json", "SUCCESS.json", "SHA256SUMS", "conversion_manifest.json"))
    )

    def responder(argv, input_text):
        command = argv[-1]
        if command.startswith("sha256sum"):
            # The second hold on the channel: the payloads the manifest named.
            return "\n".join(f"{digest}  {package}/{name}" for name, digest in sorted(on_disk.items())), "", 0
        payload = "\n".join(
            [
                "===DIAGNOSTIC===",
                _json.dumps(diagnostic),
                "===PLAN===",
                _json.dumps(plan),
                "===OWNER===",
                _json.dumps(owner),
                "===MANIFEST===",
                manifest,
                "===DIGESTS===",
                digests,
            ]
        )
        return payload, "", 0

    return responder


def _mint_params(policy, **overrides):
    params = {
        "diagnostic_path": "/home/u/diag.json",
        "plan_path": "/home/u/plan.json",
        "package_path": _PACKAGE,
        "expected_revision": policy["revision"],
        "expected_generation": policy["generation"],
        "approval_reference": "operator approved canary 002",
    }
    params.update(overrides)
    return {"params": params}


@pytest.mark.anyio
async def test_mint_launch_evidence_binds_the_chain_and_records_the_approval(monkeypatch, tmp_path):
    """The record exists only when every link agrees, and the mint is audited."""

    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder())
    snapshot = await _call("o2_local_status", {})
    policy = snapshot["policy"]

    minted = await _call(
        "o2_mint_launch_evidence",
        {
            "params": {
                "diagnostic_path": "/home/u/diag.json",
                "plan_path": "/home/u/plan.json",
                "package_path": "/pkg/attempt-002",
                "stage": "platform-canary",
                "expected_revision": policy["revision"],
                "expected_generation": policy["generation"],
                "approval_reference": "operator approved canary 002",
            }
        },
    )
    assert minted["ok"] is True
    record = minted["launch_evidence"]
    assert record["binding_check"]["all_links_agree"] is True
    assert record["submission"]["job_id"] == "52085188"
    assert record["operator_approval"]["approval_reference"] == "operator approved canary 002"
    assert len(minted["launch_evidence_sha256"]) == 64
    # st_dev is host-local for NFS, so it must not be part of the binding.
    assert "device" not in record["destination"]

    after = await _call("o2_local_status", {})
    assert any(event.get("event") == "launch_evidence_minted" for event in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_refuses_a_stale_approval(monkeypatch, tmp_path):
    """An approval that predates a policy write must not still mint."""

    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder())
    snapshot = await _call("o2_local_status", {})
    policy = snapshot["policy"]
    refused = await _call(
        "o2_mint_launch_evidence",
        {
            "params": {
                "diagnostic_path": "/home/u/diag.json",
                "plan_path": "/home/u/plan.json",
                "package_path": "/pkg/attempt-002",
                "expected_revision": policy["revision"] + 5,
                "expected_generation": policy["generation"],
                "approval_reference": "stale approval",
            }
        },
    )
    assert refused["ok"] is False
    assert refused["error"] == "policy_conflict"
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_rejects_a_relative_path_before_it_reaches_a_shell(monkeypatch, tmp_path):
    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder())
    snapshot = await _call("o2_local_status", {})
    policy = snapshot["policy"]
    refused = await _call(
        "o2_mint_launch_evidence",
        {
            "params": {
                "diagnostic_path": "../escape/diag.json",
                "plan_path": "/home/u/plan.json",
                "package_path": "/pkg/attempt-002",
                "expected_revision": policy["revision"],
                "expected_generation": policy["generation"],
                "approval_reference": "relative path",
            }
        },
    )
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    assert "absolute normalized" in refused["message"]


@pytest.mark.anyio
async def test_mint_refuses_a_package_the_plan_did_not_approve(monkeypatch, tmp_path):
    """The caller picks the directory that is read; the plan picks which is valid.

    Here the artifacts -- owner marker included, as a copy would be -- sit in a
    directory the plan never approved. Without binding the read-back path, the
    record would name the approved package while its digests described this one.
    """

    _patch_connection(
        monkeypatch,
        tmp_path,
        responder=_launch_evidence_responder(package=_PACKAGE, approved_package="/pkg/attempt-002-real"),
    )
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    assert "package_read_back_path" in refused["message"]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_rehashes_the_payloads_and_refuses_one_that_drifted(monkeypatch, tmp_path):
    """A payload that no longer matches SHA256SUMS ends the mint.

    The run diagnostic still reports its own verification as ``success``: that
    verdict is the executed process vouching for itself, which is exactly what
    this record exists not to rely on.
    """

    drifted = dict(_PAYLOADS)
    drifted["payloads/frame002.ims"] = "0" * 64
    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(payloads=drifted))
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    assert "frame002.ims" in refused["message"]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_refuses_a_payload_the_package_no_longer_holds(monkeypatch, tmp_path):
    missing = dict(_PAYLOADS)
    missing.pop("payloads/frame002.ims")
    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(payloads=missing))
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert "was not hashed on the cluster" in refused["message"]


@pytest.mark.anyio
async def test_mint_records_the_record_digest_in_the_audit_ledger(monkeypatch, tmp_path):
    """The ledger entry must pin the content, not just note that a mint happened.

    Storing only the stage, job, and package would let a holder of a legitimate
    record edit its runtime identities, recompute the record's own unkeyed
    digest, and keep an approval that still agreed with the ledger.
    """

    from o2mcp.launch_evidence import evidence_content_digest

    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder())
    snapshot = await _call("o2_local_status", {})
    minted = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert minted["ok"] is True
    record = minted["launch_evidence"]

    after = await _call("o2_local_status", {})
    events = [e for e in after["policy"]["recent_events"] if e.get("event") == "launch_evidence_minted"]
    assert len(events) == 1
    assert events[0]["evidence_sha256"] == minted["evidence_content_sha256"]
    assert events[0]["evidence_sha256"] == evidence_content_digest(record)
    assert events[0]["plan_sha256"] == record["approved_plan"]["sha256"]
    # The approval the record carries names the same digest the ledger recorded.
    assert record["operator_approval"]["evidence_sha256"] == events[0]["evidence_sha256"]

    # An edited record no longer produces the digest its approval was recorded
    # against, which is what makes the ledger entry authenticate it.
    tampered = json.loads(json.dumps(record))
    tampered["runtime_identities"]["interpreter_sha256"] = "0" * 64
    assert evidence_content_digest(tampered) != events[0]["evidence_sha256"]


@pytest.mark.anyio
async def test_mint_binds_a_package_whose_path_contains_spaces(monkeypatch, tmp_path):
    """``sha256sum`` writes the whole filename, so splitting on spaces loses it.

    Dropping those digests made the mint refuse an otherwise valid package for
    the sole reason that its path was legal.
    """

    package = "/n/scratch/attempt 002/published package"
    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(package=package))
    snapshot = await _call("o2_local_status", {})
    minted = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"], package_path=package))
    assert minted["ok"] is True
    assert minted["launch_evidence"]["destination"]["package"] == package
    assert set(minted["launch_evidence"]["verified_package"]["file_digests"]) == {
        "PUBLICATION_OWNER.json",
        "SUCCESS.json",
        "SHA256SUMS",
        "conversion_manifest.json",
    }
    assert minted["launch_evidence"]["verified_package"]["payload_reverification"]["n_payloads_reverified"] == 2


@pytest.mark.anyio
async def test_mint_refuses_a_drifted_chain_without_recording_an_approval(monkeypatch, tmp_path):
    """A refused mint must leave no audit entry implying the chain agreed."""

    def drifted(argv, input_text):
        payload, err, rc = _launch_evidence_responder()(argv, input_text)
        return payload.replace('"' + "b" * 64 + '"', '"' + "0" * 64 + '"', 1), err, rc

    _patch_connection(monkeypatch, tmp_path, responder=drifted)
    snapshot = await _call("o2_local_status", {})
    policy = snapshot["policy"]
    refused = await _call(
        "o2_mint_launch_evidence",
        {
            "params": {
                "diagnostic_path": "/home/u/diag.json",
                "plan_path": "/home/u/plan.json",
                "package_path": "/pkg/attempt-002",
                "expected_revision": policy["revision"],
                "expected_generation": policy["generation"],
                "approval_reference": "should not be recorded",
            }
        },
    )
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])
