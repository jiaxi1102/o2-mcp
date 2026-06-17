"""End-to-end tests for the o2_mcp MCP server, driven through the MCP protocol.

These exercise the full server path (FastMCP argument validation -> async tool ->
o2mcp core -> JSON payload), with the subprocess call injected so no network
is touched. They need the ``mcp`` SDK (Python 3.10+), so they are skipped in the
default 3.9 test environment (install with ``pip install -e ".[dev,o2]"`` on a 3.10+ env).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mcp")
pytest.importorskip("anyio")

from mcp.server.fastmcp.exceptions import ToolError  # noqa: E402

from o2mcp import (  # noqa: E402
    CommandResult,
    O2Config,
    O2Connection,
    async_transfer,  # noqa: E402
    transfer_tools,  # noqa: E402
)
from o2mcp import O2AsyncTransfer as _RealAsyncTransfer  # noqa: E402
from o2mcp import server as o2server  # noqa: E402


class FakeRunner:
    """Deterministic stand-in for the subprocess runner (records calls)."""

    def __init__(self, *, master: bool = True, responder=None):
        self.calls = []
        self.master = master
        self._responder = responder

    def __call__(self, argv, timeout, input_text) -> CommandResult:
        self.calls.append({"argv": list(argv), "input": input_text})
        if "-O" in argv and "check" in argv:
            return CommandResult(list(argv), 0 if self.master else 255, "", "")
        if "-MNf" in argv:
            return CommandResult(list(argv), 0, "", "")
        if self._responder is not None:
            out, err, rc = self._responder(argv, input_text)
            return CommandResult(list(argv), rc, out, err)
        return CommandResult(list(argv), 0, "", "")


def _patch_connection(monkeypatch, tmp_path, *, master=True, responder=None, locked=False) -> FakeRunner:
    cfg = O2Config(
        host_alias="o2",
        transfer_alias="o2-transfer",
        connect_timeout=20,
        lock_file=tmp_path / "O2_DISABLED",
    )
    if locked:
        cfg.lock_file.write_text("disabled")
    runner = FakeRunner(master=master, responder=responder)
    monkeypatch.setattr(o2server, "_connection", lambda: O2Connection(cfg, runner=runner))
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
        "o2_start_master",
        "o2_run",
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
    }
    assert tools["o2_status"].annotations.readOnlyHint is True
    assert tools["o2_submit_job"].annotations.readOnlyHint is False
    assert tools["o2_cancel_job"].annotations.destructiveHint is True
    assert tools["o2_workspace_gc"].annotations.destructiveHint is True
    assert tools["o2_disk_report"].annotations.readOnlyHint is True
    assert tools["o2_push_async"].annotations.readOnlyHint is False
    assert tools["o2_transfer_status"].annotations.readOnlyHint is True
    assert tools["o2_transfer_cancel"].annotations.destructiveHint is True


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


# --- safety paths ------------------------------------------------------------
@pytest.mark.anyio
async def test_status_reports_not_connected(monkeypatch, tmp_path):
    _patch_connection(monkeypatch, tmp_path, master=False)
    payload = await _call("o2_status", {})
    assert payload == {"ok": True, "locked": False, "master_running": False, "probe": None}


@pytest.mark.anyio
async def test_run_without_master_is_actionable(monkeypatch, tmp_path):
    _patch_connection(monkeypatch, tmp_path, master=False)
    payload = await _call("o2_run", {"params": {"command": "squeue"}})
    assert payload["ok"] is False
    assert payload["error"] == "no_master"
    assert "ControlMaster" in payload["message"]


@pytest.mark.anyio
async def test_start_master_refused_without_optin(monkeypatch, tmp_path):
    _patch_connection(monkeypatch, tmp_path, master=False)
    payload = await _call("o2_start_master", {"params": {"allow_new_login": False}})
    assert payload["ok"] is False and payload["error"] == "no_master"


@pytest.mark.anyio
async def test_lock_blocks_tool(monkeypatch, tmp_path):
    _patch_connection(monkeypatch, tmp_path, master=True, locked=True)
    payload = await _call("o2_run", {"params": {"command": "hostname"}})
    assert payload["ok"] is False and payload["error"] == "o2_locked"


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
async def test_invalid_input_is_rejected(monkeypatch, tmp_path):
    _patch_connection(monkeypatch, tmp_path, master=True)
    # Empty command violates min_length=1; the MCP layer must reject it.
    with pytest.raises(ToolError):
        await o2server.mcp.call_tool("o2_run", {"params": {"command": ""}})


@pytest.mark.anyio
async def test_start_master_can_open_transfer_alias(monkeypatch, tmp_path):
    # The MCP tool must be able to open the transfer-node master, so transfer-node
    # moves (o2_run_promote/archive, o2_push/pull use_transfer_node) have a master
    # to reuse instead of hitting the transfer-master guard with no way to satisfy it.
    runner = _patch_connection(monkeypatch, tmp_path, master=False)
    res = await _call("o2_start_master", {"params": {"allow_new_login": True, "transfer": True}})
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
    assert launched[0][-1] == "o2:" + remote.replace(" ", "\\ ")


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
