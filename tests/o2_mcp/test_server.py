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
    # It reads O2 files and queries Slurm through the broker, so it is not closed-world.
    assert tools["o2_mint_launch_evidence"].annotations.openWorldHint is True
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
    "--hint",
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


def _short_receipt():
    table = billing.parse_weight_table(
        "PartitionName=short TRESBillingWeights=CPU=1,Mem=0.0625G"
        " TRES=cpu=400,mem=4000G,node=10 State=UP AllowGroups=ALL"
    )
    return billing.price_receipt(billing.price(billing.Request(cpus=4, mem_gb=16), table, "short"))


def test_a_receipt_for_another_partition_is_not_this_job_s_price():
    """Weights differ per partition, so the number does not carry across.

    The one part of binding a receipt to a script that needs no script parser:
    the submission names a partition, the receipt names one, compare them.
    """
    record = o2server._pricing_record(
        o2server.SubmitInput(
            script_text="#!/bin/bash\n#SBATCH --partition=long\n",
            remote_path="/n/x.sh",
            priced=_short_receipt(),
        ),
        env_partition=None,
    )
    assert record["priced"] is False
    assert "priced for partition 'short'" in record["note"]
    # The receipt is still reported -- it is evidence about some shape.
    assert record["receipt"]["partition"] == "short"


def test_a_matching_partition_still_prices():
    record = o2server._pricing_record(
        o2server.SubmitInput(
            script_text="#!/bin/bash\n#SBATCH --partition=short\n",
            remote_path="/n/x.sh",
            priced=_short_receipt(),
        )
    )
    assert record["priced"] is True
    assert "note" not in record


def test_an_argument_beats_a_directive_when_they_disagree():
    # sbatch takes the command line, so the argument decides the comparison --
    # in both directions, or the check would only ever be one of them.
    conflicting = o2server._pricing_record(
        o2server.SubmitInput(
            script_text="#!/bin/bash\n#SBATCH --partition=short\n",
            remote_path="/n/x.sh",
            sbatch_args=["-plong"],
            priced=_short_receipt(),
        )
    )
    assert conflicting["priced"] is False
    agreeing = o2server._pricing_record(
        o2server.SubmitInput(
            script_text="#!/bin/bash\n#SBATCH --partition=long\n",
            remote_path="/n/x.sh",
            sbatch_args=["-pshort"],
            priced=_short_receipt(),
        )
    )
    assert agreeing["priced"] is True


def test_a_partition_list_conflicts_only_when_the_receipt_is_not_in_it():
    # sbatch picks one of the list; which is not knowable here, so a receipt
    # priced for any member stays consistent.
    def rec(value):
        return o2server._pricing_record(
            o2server.SubmitInput(
                script_text=f"#!/bin/bash\n#SBATCH --partition={value}\n",
                remote_path="/n/x.sh",
                priced=_short_receipt(),
            ),
            env_partition=None,
        )

    assert rec("long,short")["priced"] is True
    assert rec("long,gpu")["priced"] is False


def test_a_submission_naming_no_partition_is_not_a_conflict():
    record = o2server._pricing_record(
        o2server.SubmitInput(
            script_text="#!/bin/bash\n#SBATCH --mem=16G\n", remote_path="/n/x.sh", priced=_short_receipt()
        )
    )
    assert record["priced"] is True


def test_the_last_partition_option_wins_whatever_its_spelling():
    """sbatch parses sequentially, so position decides, not spelling.

    Searching "--partition" before "-p" answered `short` for
    ["--partition=short", "-plong"], which sbatch reads as long.
    """

    def mk(args):
        return o2server.SubmitInput(script_text="#!/bin/bash\n", remote_path="/n/x.sh", sbatch_args=args)

    assert o2server._stated_partition(mk(["--partition=short", "-plong"]), env_partition=None) == "long"
    assert o2server._stated_partition(mk(["-pshort", "--partition=long"]), env_partition=None) == "long"
    assert o2server._stated_partition(mk(["-plong"]), env_partition=None) == "long"


def test_the_environment_outranks_a_directive():
    # The broker copies os.environ into every command, so SBATCH_PARTITION
    # reaches sbatch and beats the script -- but not the command line.
    scripted = o2server.SubmitInput(script_text="#!/bin/bash\n#SBATCH --partition=long\n", remote_path="/n/x.sh")
    assert o2server._stated_partition(scripted, env_partition="short") == "short"
    assert o2server._stated_partition(scripted, env_partition=None) == "long"
    with_argument = o2server.SubmitInput(
        script_text="#!/bin/bash\n#SBATCH --partition=long\n",
        remote_path="/n/x.sh",
        sbatch_args=["-pgpu"],
    )
    assert o2server._stated_partition(with_argument, env_partition="short") == "gpu"


def test_an_unreadable_environment_claims_no_conflict():
    """Undecidable is not a conflict.

    The environment outranks the script, so when it cannot be read the winning
    partition is unknown -- and a record that asserts a conflict there is
    claiming something it cannot establish.
    """
    params = o2server.SubmitInput(
        script_text="#!/bin/bash\n#SBATCH --partition=long\n",
        remote_path="/n/x.sh",
        priced=_short_receipt(),
    )
    assert o2server._stated_partition(params) is o2server.UNKNOWN_ENV
    record = o2server._pricing_record(params)
    assert record["priced"] is True
    assert "note" not in record
    # ...but a command-line partition outranks the environment either way, so
    # that conflict IS decidable without reading it.
    from_argument = o2server.SubmitInput(
        script_text="#!/bin/bash\n", remote_path="/n/x.sh", sbatch_args=["-plong"], priced=_short_receipt()
    )
    assert o2server._pricing_record(from_argument)["priced"] is False


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
# Distinguishes "use the default accounting row" from "accounting knows nothing".
_UNSET = object()
_REQUIRED_FILES = ("PUBLICATION_OWNER.json", "SUCCESS.json", "SHA256SUMS", "conversion_manifest.json")
_PAYLOADS = {"payloads/frame 001.ims": "1" * 64, "payloads/frame002.ims": "2" * 64}


def _launch_evidence_responder(
    job_id="52085188",
    *,
    package=_PACKAGE,
    payloads=None,
    manifest_payloads=None,
    approved_package=None,
    manifest_sha256=None,
    owner_sha256=None,
    n_payloads=None,
    resolved_package=None,
    unreadable=None,
    mutated=None,
    package_inode=None,
    observed_inode=None,
    diagnostic_edit=None,
    stage_id="platform-canary",
    grown_by=None,
    root_error=None,
    accounting=_UNSET,
):
    """Serve the artifacts and the payload digests the mint reads off the cluster.

    ``package`` is where the artifacts actually live, ``approved_package`` is what
    the plan and the diagnostic name; they differ only when a test is checking
    that a substituted directory is refused. ``manifest_payloads`` is what
    SHA256SUMS lists and ``payloads`` is what those files actually hash to, so a
    test can make the package drift out from under its own manifest.
    """

    import base64 as _base64
    import hashlib as _hashlib
    import json as _json
    import shlex

    from o2mcp.launch_evidence import plan_digest

    listed = dict(_PAYLOADS if manifest_payloads is None else manifest_payloads)
    inode_reported = (
        (7483730113644355164 if package_inode is None else package_inode) if observed_inode is None else observed_inode
    )
    if accounting is _UNSET:
        accounting = f"{job_id}|COMPLETED|tabin|short|"
    on_disk = dict(listed if payloads is None else payloads)
    named = package if approved_package is None else approved_package
    manifest_bytes = ("\n".join(f"{digest}  {name}" for name, digest in sorted(listed.items())) + "\n").encode()
    plan = {
        "attempt_id": "002",
        "software": {"bundle": {"bundle_sha256": "b" * 64}},
        "runtime_wrapper": {"sha256": "a1" * 32},
        "interpreter": {"sha256": "b2" * 32, "closure_sha256": "c" * 64, "context_sha256": "c3" * 32},
        "stage_id": stage_id,
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
            "runtime_wrapper_sha256": "a1" * 32,
            "approved_interpreter": {
                "approved_path": "/usr/bin/python3",
                "sha256": "b2" * 32,
                "runtime_context_sha256": "c3" * 32,
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
        "launch": {"loaded_library_closure": {"loaded_closure_sha256": "d4" * 32}},
        "output": {
            "package": named,
            "package_inode": 7483730113644355164 if package_inode is None else package_inode,
            "sha256sums_sha256": _hashlib.sha256(manifest_bytes).hexdigest(),
            "conversion_manifest_sha256": "d" * 64,
            "reopened_output_sha256": "e5" * 32,
            "verification": {
                "status": "success",
                "n_payloads": len(listed) if n_payloads is None else n_payloads,
            },
        },
    }
    if diagnostic_edit is not None:
        diagnostic_edit(diagnostic)
    owner = {"plan_sha256": plan_digest(plan), "attempt_id": "002"}
    # The server proves the manifest it parsed is the file it recorded a digest
    # for, so the fixture's SHA256SUMS digest has to be the real one.
    owner_bytes = _json.dumps(owner).encode()

    def _body_for(operand):
        if operand.endswith("diag.json"):
            return _json.dumps(diagnostic).encode()
        if operand.endswith("plan.json"):
            return _json.dumps(plan).encode()
        if operand.endswith("PUBLICATION_OWNER.json"):
            return owner_bytes
        return manifest_bytes if grown_by is None else manifest_bytes + b"x" * grown_by

    def responder(argv, input_text):
        command = argv[-1]
        if command.startswith("/usr/bin/python3 -c "):
            # The no-follow hasher: names arrive on stdin, JSON comes back.
            raw = input_text or ""
            if raw[:2] == "Z:":
                import base64 as _b64
                import zlib as _zlib

                raw = _zlib.decompress(_b64.b64decode(raw[2:])).decode("utf-8")
            asked = [name for name in raw.split("\n") if name]
            known = dict(on_disk)
            known.update({name: "d" * 64 for name in _REQUIRED_FILES})
            # These two are also read for their contents, and those bytes are
            # verified against the digest reported here, so they must agree.
            known["SHA256SUMS"] = (
                _hashlib.sha256(manifest_bytes).hexdigest() if manifest_sha256 is None else manifest_sha256
            )
            known["PUBLICATION_OWNER.json"] = (
                _hashlib.sha256(owner_bytes).hexdigest() if owner_sha256 is None else owner_sha256
            )
            # The stat-only re-check is the same interpreter with a different
            # program; tell them apart by the shape each one declares.
            restat = '"aggregate"' in command
            if root_error is not None:
                empty = {"stats": {}} if restat else {"digests": {}}
                empty.update({"errors": {package: root_error}, "package_inode": 0})
                return _json.dumps(empty), "", 0
            reply = {"digests": {}, "errors": {}, "stats": {}, "package_inode": inode_reported}
            seen = []
            for name in asked:
                if name in (unreadable or {}):
                    reply["errors"][name] = (unreadable or {})[name]
                elif name in known:
                    if not restat:
                        reply["digests"][name] = known[name]
                    # A payload rewritten between hashing and recording keeps its
                    # name and its parent directory, so only its own identity moves.
                    bump = 1 if (restat and name in (mutated or ())) else 0
                    identity = [11, 20, 300 + bump, 400 + bump]
                    reply["stats"][name] = identity
                    seen.append("\0".join([name, *(str(v) for v in identity)]) + "\n")
                else:
                    reply["errors"][name] = "ENOENT"
            if restat:
                # The re-check answers with one constant-size aggregate rather
                # than per-name identities, so a package of any size is read in
                # a single pass with no window between batches.
                seen.sort()
                reply = {
                    "errors": reply["errors"],
                    "package_inode": inode_reported,
                    "named": dict(reply["stats"]),
                    "aggregate": _hashlib.sha256("".join(seen).encode("utf-8")).hexdigest(),
                }
            return _json.dumps(reply), "", 0
        if command.startswith("tail -c +") and command.endswith("| wc -c"):
            # The end-of-file probe: one byte past the size that was stat'd.
            operand = shlex.split(command.split(" -- ")[1].split(" |")[0])[0]
            start = int(command.split("tail -c +")[1].split(" ")[0]) - 1
            body = _body_for(operand)
            return ("1" if len(body) > start else "0") + "\n", "", 0
        if command.startswith("tail -c +"):
            # Every large artifact is read in bounded pieces so it still fits the
            # broker's output cap; serve exactly the slice that was asked for.
            start = int(command.split("tail -c +")[1].split(" ")[0]) - 1
            count = int(command.split("head -c ")[1].split(" ")[0])
            operand = shlex.split(command.split(" -- ")[1].split(" |")[0])[0]
            return _base64.b64encode(_body_for(operand)[start : start + count]).decode(), "", 0
        if command.startswith("sacct "):
            # Slurm accounting, read through the broker rather than taken from
            # the diagnostic. One allocation row, `|`-separated, no header.
            return "" if accounting is None else accounting, "", 0
        payload = "\n".join(
            [
                "===DIAGNOSTICSIZE===",
                str(len(_json.dumps(diagnostic).encode())),
                "===PLANSIZE===",
                str(len(_json.dumps(plan).encode())),
                "===OWNERSIZE===",
                str(len(owner_bytes)),
                "===MANIFESTSIZE===",
                str(len(manifest_bytes)),
                "===RESOLVED===",
                package if resolved_package is None else resolved_package,
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
    assert "could not be read as ordinary files" in refused["message"]
    assert "ENOENT" in refused["message"]


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
async def test_mint_is_recorded_in_the_durable_ledger_not_only_the_event_log(monkeypatch, tmp_path):
    """A record stays verifiable after the rolling event buffer has moved on."""

    from o2mcp.launch_evidence import evidence_content_digest

    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder())
    snapshot = await _call("o2_local_status", {})
    minted = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert minted["ok"] is True
    digest = evidence_content_digest(minted["launch_evidence"])

    after = await _call("o2_local_status", {})
    assert after["policy"]["launch_evidence_mint_count"] == 1
    ledger = after["policy"]["recent_launch_evidence_mints"]
    assert [entry["evidence_sha256"] for entry in ledger] == [digest]
    assert ledger[0]["stage"] == "platform-canary"
    assert ledger[0]["job_id"] == "52085188"


@pytest.mark.anyio
async def test_mint_refuses_a_manifest_that_changed_while_it_was_read(monkeypatch, tmp_path):
    """The bytes that chose the payloads must be the file whose digest is recorded.

    The manifest is read and hashed by two commands in the same line. If it is
    replaced in between, the record would report the digest of a manifest it
    never used to decide anything.
    """

    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(manifest_sha256="0" * 64))
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    assert "changed while it was being read" in refused["message"]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_refuses_an_owner_marker_that_changed_while_it_was_read(monkeypatch, tmp_path):
    """The owner marker ties the package to this plan, so its bytes must be its file.

    Replaced between the read and the hash, an old marker could satisfy the
    plan and attempt checks while `file_digests` recorded the replacement's
    hash -- the record would then authenticate a package whose current owner
    identifies a different launch.
    """

    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(owner_sha256="0" * 64))
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    assert "publication owner changed while it was being read" in refused["message"]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_refuses_a_diagnostic_with_no_job_id(monkeypatch, tmp_path):
    """A record that binds no submitted job is not the thing this tool issues."""

    def drop_job(diagnostic):
        diagnostic["slurm"]["scheduler"]["job_id"] = None

    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(diagnostic_edit=drop_job))
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    assert "job_id" in refused["message"]
    after = await _call("o2_local_status", {})
    # In particular, the ledger must not have recorded the string "None" as a job.
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_refuses_a_manifest_shortened_after_the_run(monkeypatch, tmp_path):
    """Payloads deleted from disk and from SHA256SUMS together are still caught.

    Every remaining entry rehashes correctly, so only the count the run recorded
    reveals the deletion.
    """

    shortened = {"payloads/frame002.ims": "2" * 64}
    responder = _launch_evidence_responder(manifest_payloads=shortened, n_payloads=len(_PAYLOADS))
    _patch_connection(monkeypatch, tmp_path, responder=responder)
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    assert "but the run verified" in refused["message"]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_refuses_a_run_whose_own_outcome_is_not_success(monkeypatch, tmp_path):
    def fail(diagnostic):
        diagnostic["status"] = "diagnostic_failed"

    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(diagnostic_edit=fail))
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert "does not say it finished" in refused["message"]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_refuses_a_package_pathname_that_resolves_elsewhere(monkeypatch, tmp_path):
    """The approved spelling reaching a substituted directory must not mint."""

    responder = _launch_evidence_responder(resolved_package="/pkg/substituted")
    _patch_connection(monkeypatch, tmp_path, responder=responder)
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    assert "resolved_package_path" in refused["message"]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_refuses_a_payload_that_is_a_symlink(monkeypatch, tmp_path):
    """The payload is opened with O_NOFOLLOW, so a link never gets hashed.

    The previous `find`-based scan and the hashing were separate commands, so a
    link created between them was hashed anyway. Opening and hashing are now one
    act on one descriptor.
    """

    responder = _launch_evidence_responder(unreadable={"payloads/frame002.ims": "symlink"})
    _patch_connection(monkeypatch, tmp_path, responder=responder)
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    assert "symlink" in refused["message"]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_refuses_a_package_it_cannot_open_as_ordinary_files(monkeypatch, tmp_path):
    """An unreadable or non-regular entry ends the mint rather than being skipped.

    The old `;`-joined `find` swallowed its own failure, so an execute-only
    package scanned clean; now every entry has to be opened and hashed, and
    anything that cannot be is named.
    """

    responder = _launch_evidence_responder(
        unreadable={"payloads/frame002.ims": "EACCES", "payloads/frame 001.ims": "not a regular file"}
    )
    _patch_connection(monkeypatch, tmp_path, responder=responder)
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert "EACCES" in refused["message"] and "not a regular file" in refused["message"]


@pytest.mark.anyio
async def test_mint_refuses_a_package_renamed_under_the_approved_path(monkeypatch, tmp_path):
    """A different directory moved into place resolves the same and has a new inode.

    `destination.inode` names the parent, so it never caught this; the run
    records the package directory's own inode and the server observes it through
    the descriptor it read the package under.
    """

    responder = _launch_evidence_responder(observed_inode=1234567890)
    _patch_connection(monkeypatch, tmp_path, responder=responder)
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert "package_inode" in refused["message"]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_the_payload_hash_opens_and_reads_in_one_act(monkeypatch, tmp_path):
    """No pathname is resolved separately from the read it justifies.

    The previous design scanned for symlinks and hashed in different commands,
    so a link created between them was hashed anyway. Every payload is now
    opened with O_NOFOLLOW and hashed through that descriptor, and nothing
    shells out to `find` or `sha256sum` for the package at all.
    """

    runner = _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder())
    snapshot = await _call("o2_local_status", {})
    minted = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert minted["ok"] is True

    commands = [call["argv"][-1] for call in runner.calls]
    assert any(command.startswith("/usr/bin/python3 -c ") for command in commands)
    assert not [command for command in commands if "sha256sum" in command]
    assert not [command for command in commands if command.startswith("find ")]
    # The names travel on stdin, so no package path is embedded per payload.
    for call in runner.calls:
        if call["argv"][-1].startswith("/usr/bin/python3 -c "):
            assert call["input"], "the hasher takes its names on stdin"


@pytest.mark.anyio
async def test_a_minted_record_verifies_against_its_own_ledger_entry(monkeypatch, tmp_path):
    """End to end: what the tool returns is what the ledger says was approved."""

    from o2mcp.launch_evidence import verify_launch_evidence

    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder())
    snapshot = await _call("o2_local_status", {})
    minted = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert minted["ok"] is True
    record = minted["launch_evidence"]

    after = await _call("o2_local_status", {})
    entry = after["policy"]["recent_launch_evidence_mints"][-1]
    verify_launch_evidence(record, entry)

    # Rewriting who approved it now disagrees with the ledger, even though the
    # content digest is untouched.
    forged = json.loads(json.dumps(record))
    forged["operator_approval"]["approval_reference"] = "someone else approved this"
    with pytest.raises(o2server.LaunchEvidenceError, match="approval_reference"):
        verify_launch_evidence(forged, entry)


@pytest.mark.anyio
async def test_a_stage_with_whitespace_still_verifies_against_its_ledger_entry(monkeypatch, tmp_path):
    """A record must never fail the verification it was just issued under.

    The stage is stored verbatim in the record and compared exactly against the
    ledger, so collapsing its whitespace on the way into the ledger would make a
    freshly minted record fail its own check.
    """

    from o2mcp.launch_evidence import verify_launch_evidence

    # The plan names the stage, so the plan under test names this one.
    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(stage_id="platform  canary"))
    snapshot = await _call("o2_local_status", {})
    minted = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"], stage="platform  canary"))
    assert minted["ok"] is True, minted
    assert minted["launch_evidence"]["stage"] == "platform  canary"

    after = await _call("o2_local_status", {})
    entry = after["policy"]["recent_launch_evidence_mints"][-1]
    assert entry["stage"] == "platform  canary"
    verify_launch_evidence(minted["launch_evidence"], entry)


@pytest.mark.anyio
async def test_mint_refuses_a_job_slurm_says_failed(monkeypatch, tmp_path):
    responder = _launch_evidence_responder(accounting="52085188|FAILED|tabin|short|")
    _patch_connection(monkeypatch, tmp_path, responder=responder)
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert "did not finish" in refused["message"]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_records_what_the_cluster_resolved(monkeypatch, tmp_path):
    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder())
    snapshot = await _call("o2_local_status", {})
    minted = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert minted["ok"] is True, minted
    assert minted["launch_evidence"]["destination"]["resolved_package_path"] == _PACKAGE


@pytest.mark.anyio
async def test_mint_refuses_a_job_slurm_accounting_does_not_know(monkeypatch, tmp_path):
    """A run too old to attest, or a job that never existed, must not mint.

    Falling back to the diagnostic's own claim when accounting has aged out is
    exactly the quiet degradation this record exists to prevent, so silence from
    `sacct` refuses and names the retention window.
    """

    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(accounting=""))
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    assert "purged after 366 days" in refused["message"]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_refuses_a_job_belonging_to_another_allocation(monkeypatch, tmp_path):
    """A real job id is not enough; it has to be this run's job."""

    responder = _launch_evidence_responder(accounting="52085188|COMPLETED|someone-else|priority|")
    _patch_connection(monkeypatch, tmp_path, responder=responder)
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert "scheduler account" in refused["message"]


@pytest.mark.anyio
async def test_mint_asks_accounting_about_the_claimed_job_and_records_the_answer(monkeypatch, tmp_path):
    runner = _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder())
    snapshot = await _call("o2_local_status", {})
    minted = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert minted["ok"] is True
    assert minted["launch_evidence"]["submission"]["scheduler_accounting"] == {
        "job_id": "52085188",
        "state": "COMPLETED",
        "account": "tabin",
        "partition": "short",
        "comment": "",
    }
    sacct = [call["argv"][-1] for call in runner.calls if call["argv"][-1].startswith("sacct ")]
    assert len(sacct) == 1 and "52085188" in sacct[0]


@pytest.mark.anyio
async def test_a_job_id_that_is_not_one_never_reaches_a_shell(monkeypatch, tmp_path):
    """The id comes from the untrusted diagnostic, so its shape is checked first."""

    def inject(diagnostic):
        diagnostic["slurm"]["scheduler"]["job_id"] = "52085188; touch /tmp/pwned"

    runner = _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(diagnostic_edit=inject))
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    assert "is not a Slurm job id" in refused["message"]
    assert not [call for call in runner.calls if call["argv"][-1].startswith("sacct ")]


@pytest.mark.anyio
async def test_mint_reads_a_manifest_larger_than_the_broker_output_cap(monkeypatch, tmp_path):
    """base64 expansion put a big SHA256SUMS over the 1 MiB cap in one read.

    The payload hashing is already batched for large packages, so the manifest
    must not be the one thing that makes them unmintable. Chunking needs no
    per-chunk check: the reassembled bytes are verified against the digest
    recorded for that filename.
    """

    from o2mcp.broker_protocol import MAX_OUTPUT_BYTES, MAX_STDIN_BYTES

    payloads = {f"payloads/{'segment-' * 12}{index:05d}.ome.tif": f"{index:064d}" for index in range(9000)}
    runner = _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(manifest_payloads=payloads))
    snapshot = await _call("o2_local_status", {})
    minted = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert minted["ok"] is True
    reverified = minted["launch_evidence"]["verified_package"]["payload_reverification"]
    assert reverified["n_payloads_reverified"] == 9000

    # Even at this size the re-check stays a single pass: the name list is
    # compressed rather than split, because batching it would reopen the window
    # between an early stat and the ledger write.
    rechecks = [call for call in runner.calls if '"aggregate"' in call["argv"][-1]]
    assert len(rechecks) == 1
    assert rechecks[0]["input"].startswith("Z:"), "9000 names should not have fitted uncompressed"
    assert len(rechecks[0]["input"].encode("utf-8")) <= MAX_STDIN_BYTES

    reads = [call["argv"][-1] for call in runner.calls if call["argv"][-1].startswith("tail -c +")]
    assert len(reads) > 1, "a manifest this size should have needed more than one read"
    # Whole in one read, base64 would have overrun the broker's output cap.
    manifest_bytes = sum(len(name) + 66 + 1 for name in payloads)
    assert manifest_bytes * 4 / 3 > MAX_OUTPUT_BYTES


@pytest.mark.anyio
async def test_mint_reads_a_large_plan_and_diagnostic_in_pieces(monkeypatch, tmp_path):
    """Neither JSON artifact has a size bound, and they shared one 1 MiB stream.

    A plan with a large dataset list could therefore truncate the artifact read
    and make an otherwise valid package unmintable.
    """

    def inflate(diagnostic):
        diagnostic["padding"] = ["x" * 512] * 3000

    runner = _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(diagnostic_edit=inflate))
    snapshot = await _call("o2_local_status", {})
    minted = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert minted["ok"] is True

    reads = [call["argv"][-1] for call in runner.calls if "diag.json" in call["argv"][-1]]
    chunked = [command for command in reads if command.startswith("tail -c +")]
    assert len(chunked) > 1, "a diagnostic this size should have needed more than one read"


@pytest.mark.anyio
async def test_mint_refuses_a_diagnostic_too_large_to_be_one(monkeypatch, tmp_path):
    """The size driving the loop is untrusted for these artifacts too."""

    def enormous(argv, input_text):
        payload, err, rc = _launch_evidence_responder()(argv, input_text)
        if argv[-1].startswith("tail -c +") and "diag.json" in argv[-1]:
            raise AssertionError("the ceiling should have refused before any chunk was read")
        marker = "===DIAGNOSTICSIZE===\n"
        head, found, tail = payload.partition(marker)
        if not found:
            return payload, err, rc
        return f"{head}{marker}{1024**4}\n{tail.split(chr(10), 1)[1]}", err, rc

    _patch_connection(monkeypatch, tmp_path, responder=enormous)
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert "byte ceiling" in refused["message"]


@pytest.mark.anyio
async def test_mint_refuses_a_manifest_too_large_to_be_a_package(monkeypatch, tmp_path):
    """The size driving the chunk loop is untrusted, so it must be bounded.

    A sparse SHA256SUMS claiming a terabyte would otherwise mean millions of
    sequential broker commands and an attempt to accumulate that much base64 --
    one mint monopolising the shared channel indefinitely.
    """

    def enormous(argv, input_text):
        payload, err, rc = _launch_evidence_responder()(argv, input_text)
        if argv[-1].startswith("tail -c +"):
            raise AssertionError("the size ceiling should have refused before any chunk was read")
        marker = "===MANIFESTSIZE===\n"
        head, found, tail = payload.partition(marker)
        if not found:
            return payload, err, rc
        return f"{head}{marker}{1024**4}\n{tail.split(chr(10), 1)[1]}", err, rc

    runner = _patch_connection(monkeypatch, tmp_path, responder=enormous)
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    assert "byte ceiling" in refused["message"]
    assert not [call for call in runner.calls if call["argv"][-1].startswith("tail -c +")]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_refuses_a_stage_the_plan_did_not_name(monkeypatch, tmp_path):
    """The label arrives with the approval, and the plan names it too."""

    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder())
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"], stage="acquisition"))
    assert refused["ok"] is False
    assert "stage_id" in refused["message"]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_refuses_a_swapped_manifest_of_the_same_size(monkeypatch, tmp_path):
    """The run recorded SHA256SUMS's digest, so a same-size swap is caught."""

    responder = _launch_evidence_responder(manifest_sha256="0" * 64)
    _patch_connection(monkeypatch, tmp_path, responder=responder)
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert "SHA256SUMS" in refused["message"]


@pytest.mark.anyio
async def test_mint_binds_a_comment_naming_the_plan(monkeypatch, tmp_path):
    """Present, it must be the plan digest; absent, the record says it is not bound."""

    unbound = _launch_evidence_responder()
    _patch_connection(monkeypatch, tmp_path, responder=unbound)
    snapshot = await _call("o2_local_status", {})
    minted = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert minted["ok"] is True
    assert minted["launch_evidence"]["binding_check"]["scheduler_comment_bound"] is False

    wrong = _launch_evidence_responder(
        accounting="52085188|COMPLETED|tabin|short|o2plan:v1:" + "0" * 64 + ":platform-canary:a002"
    )
    _patch_connection(monkeypatch, tmp_path, responder=wrong)
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert "names plan" in refused["message"]


@pytest.mark.anyio
async def test_the_artifact_command_asks_for_every_section_it_requires(monkeypatch, tmp_path):
    """The reader must not require a section the command never asks for.

    A responder that answers generously hides this: the command stopped emitting
    the owner artifact while the reader still required it, so every real broker
    response would have refused the mint on a missing section.
    """

    runner = _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder())
    snapshot = await _call("o2_local_status", {})
    minted = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert minted["ok"] is True

    artifact = [call["argv"][-1] for call in runner.calls if "===DIAGNOSTICSIZE===" in call["argv"][-1]][0]
    for marker in o2server._LAUNCH_EVIDENCE_MARKERS:
        assert marker in artifact, f"{marker} is required by the reader but never asked for"
    # And the owner is actually read, not merely sized.
    assert any("PUBLICATION_OWNER.json" in call["argv"][-1] for call in runner.calls)


@pytest.mark.anyio
async def test_mint_refuses_a_manifest_that_grew_after_it_was_sized(monkeypatch, tmp_path):
    """A prefix that hashes correctly is not the file that is on disk now.

    The size is taken once, before the digest and before the chunk reads, so a
    manifest extended afterwards would hand back a prefix matching the recorded
    digest while carrying entries this never sees.
    """

    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(grown_by=4096))
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    assert "grew while it was being read" in refused["message"]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_the_recheck_reads_the_whole_package_in_one_pass(monkeypatch, tmp_path):
    """A batched re-check carries the very defect it exists to close.

    Batching caches an early batch's identities while later batches are still
    being read, so a payload overwritten in between compares equal against the
    stale copy. The reply is a single constant-size aggregate precisely so the
    whole package is read by one process in one pass, however many payloads it
    holds -- note the hashing itself still batches, because it must carry a
    digest per file.
    """

    many = {f"payloads/frame{index:04d}.ims": f"{index:064x}" for index in range(400)}
    runner = _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(manifest_payloads=many))
    snapshot = await _call("o2_local_status", {})
    minted = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert minted["ok"] is True, minted.get("message")

    rechecks = [c for c in runner.calls if '"aggregate"' in c["argv"][-1]]
    assert len(rechecks) == 1
    # And that single pass really did cover every payload, not a prefix of them.
    asked = [n for n in (rechecks[0]["input"] or "").split("\n") if n]
    assert set(many) <= set(asked)

    hashes = [c for c in runner.calls if '"digests"' in c["argv"][-1]]
    assert len(hashes) > 1, "the hashing pass is expected to batch; only the re-check must not"


@pytest.mark.anyio
async def test_the_recheck_is_the_last_thing_done_before_the_ledger_write(monkeypatch, tmp_path):
    """No remote call may follow the re-check.

    Be precise about what this does and does not establish. It does NOT
    distinguish the re-check sitting before or after ``mint({})``: that call is
    pure local computation, so the re-check is the last remote call either way,
    and this test passes with the check in either position. It is a regression
    guard for the thing that WOULD matter -- a future remote read added after the
    re-check, which would put an unverified round trip between the last
    observation of the package and the ledger entry.
    """

    runner = _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder())
    snapshot = await _call("o2_local_status", {})
    minted = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert minted["ok"] is True, minted.get("message")

    assert (
        '"aggregate"' in runner.calls[-1]["argv"][-1]
    ), "the payload re-check must be the final remote call; anything after it reopens the window"


@pytest.mark.anyio
async def test_mint_refuses_a_payload_rewritten_after_it_was_hashed(monkeypatch, tmp_path):
    """Hashing takes several round trips; the bytes must still be there at the end.

    A publisher able to write regular files can overwrite an early payload once
    its digest has been taken. The digest still matches SHA256SUMS, and neither
    package-inode check notices, because overwriting a file does not replace its
    parent directory -- so the record would authenticate bytes that are no
    longer published.
    """

    _patch_connection(
        monkeypatch,
        tmp_path,
        responder=_launch_evidence_responder(mutated={"payloads/frame002.ims"}),
    )
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    assert "changed after they were hashed" in refused["message"]
    assert "payloads/frame002.ims" in refused["message"]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_refuses_a_required_file_rewritten_after_it_was_hashed(monkeypatch, tmp_path):
    """The same protection covers the required files, not only the payloads."""

    _patch_connection(
        monkeypatch,
        tmp_path,
        responder=_launch_evidence_responder(mutated={"conversion_manifest.json"}),
    )
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert "changed after they were hashed" in refused["message"]


@pytest.mark.anyio
async def test_mint_refuses_a_package_path_replaced_by_a_symlink(monkeypatch, tmp_path):
    """`realpath` runs earlier, so the root is opened O_NOFOLLOW as well.

    A package directory renamed away and replaced with a symlink to it after the
    resolution would otherwise be followed, and the target's inode reported as
    though it were the package's.
    """

    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(root_error="symlink"))
    snapshot = await _call("o2_local_status", {})
    refused = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert refused["ok"] is False
    assert refused["error"] == "launch_evidence_refused"
    assert "symlink" in refused["message"] and _PACKAGE in refused["message"]
    after = await _call("o2_local_status", {})
    assert not any(e.get("event") == "launch_evidence_minted" for e in after["policy"]["recent_events"])


@pytest.mark.anyio
async def test_mint_hashes_a_large_package_in_batches_the_broker_accepts(monkeypatch, tmp_path):
    """A package with many payloads must still mint.

    One `sha256sum` naming every payload overruns the broker's command limit,
    and BrokerClient.execute rejects it before it runs -- so an entirely
    legitimate package could never be attested.
    """

    from o2mcp.broker_protocol import MAX_COMMAND_BYTES, MAX_STDIN_BYTES

    payloads = {f"payloads/{'segment-' * 16}{index:04d}.ims": f"{index:064d}" for index in range(3000)}
    runner = _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(manifest_payloads=payloads))
    snapshot = await _call("o2_local_status", {})
    minted = await _call("o2_mint_launch_evidence", _mint_params(snapshot["policy"]))
    assert minted["ok"] is True
    assert minted["launch_evidence"]["verified_package"]["payload_reverification"]["n_payloads_reverified"] == 3000

    hashes = [call for call in runner.calls if call["argv"][-1].startswith("/usr/bin/python3 -c ")]
    # One call for the required metadata files, then two or more payload batches.
    assert len(hashes) > 2, "3000 payloads should have needed more than one batch"
    assert all(len(call["input"].encode("utf-8")) <= MAX_STDIN_BYTES for call in hashes)
    assert all(len(call["argv"][-1].encode("utf-8")) <= MAX_COMMAND_BYTES for call in hashes)


def test_hash_batches_refuses_a_name_it_can_never_send():
    with pytest.raises(o2server.LaunchEvidenceError, match="too long to hash in one call"):
        list(o2server._hash_batches(["x" * 700000]))


def test_hash_batches_keeps_one_call_when_the_reply_fits():
    assert list(o2server._hash_batches(["a", "b", "c"])) == [["a", "b", "c"]]


def test_hash_batches_bounds_the_reply_not_the_command():
    """The names go out on stdin, so the reply is what has to be bounded.

    Each name costs its own length twice -- once on the way out, once echoed in
    the JSON -- plus a digest, and the broker caps captured output at 1 MiB.
    """

    from o2mcp.broker_protocol import MAX_OUTPUT_BYTES, MAX_STDIN_BYTES

    names = [f"payloads/{'segment-' * 16}{index:05d}.ims" for index in range(3000)]
    batches = list(o2server._hash_batches(names))
    assert len(batches) > 1, "3000 names should not fit in one reply"
    assert sum(len(batch) for batch in batches) == len(names)
    assert [name for batch in batches for name in batch] == names
    for batch in batches:
        stdin = "\n".join(batch) + "\n"
        assert len(stdin.encode("utf-8")) <= MAX_STDIN_BYTES
        # The reply carries every name back plus a 64-character digest each.
        reply = sum(len(name.encode("utf-8")) + 70 for name in batch)
        assert reply <= MAX_OUTPUT_BYTES


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

    def drift(diagnostic):
        diagnostic["runtime"]["source_bundle"]["bundle_sha256"] = "0" * 64

    _patch_connection(monkeypatch, tmp_path, responder=_launch_evidence_responder(diagnostic_edit=drift))
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
