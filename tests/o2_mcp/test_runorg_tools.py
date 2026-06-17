"""End-to-end: o2mcp.runorg.tools.register wires the run-org tools onto a FastMCP server.

Needs the mcp SDK (3.10+); skipped in the 3.9 unit env. Exercises the register hook with
an injected runs_factory (fake connection) + a real run_tool wrapper — no network.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mcp")
pytest.importorskip("anyio")

import anyio  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

from o2mcp import CommandResult, O2Config, O2Connection  # noqa: E402
from o2mcp.runorg import (  # noqa: E402
    O2Runs,
    RunPolicy,
    tools,  # noqa: E402
)

POLICY = RunPolicy(pipeline_keywords=(("grid", "grid"),), sweep_markers=("wip",))


class _Runner:
    def __call__(self, argv, timeout, input_text) -> CommandResult:
        if "-O" in argv and "check" in argv:  # master is up
            return CommandResult(list(argv), 0, "", "")
        return CommandResult(list(argv), 0, "", "")  # registry read etc. -> empty/ok


async def _run_tool(work):
    return json.dumps(await anyio.to_thread.run_sync(work))


def _build(tmp_path) -> FastMCP:
    cfg = O2Config(host_alias="o2", transfer_alias="o2-transfer", lock_file=tmp_path / "O2_DISABLED")
    factory = lambda: O2Runs(O2Connection(cfg, runner=_Runner()), POLICY)  # noqa: E731
    mcp = FastMCP("probe")
    tools.register(mcp, runs_factory=factory, run_tool=_run_tool)
    return mcp


async def _call(mcp, name, arguments):
    result = await mcp.call_tool(name, arguments)
    content = result[0] if isinstance(result, tuple) else result
    text = content[0].text if isinstance(content, list) else content
    return json.loads(text)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_register_wires_all_runorg_tools(tmp_path):
    mcp = _build(tmp_path)
    tl = {t.name: t for t in await mcp.list_tools()}
    assert set(tl) == {
        "o2_submit_run",
        "o2_run_register",
        "o2_run_list",
        "o2_run_show",
        "o2_run_classify",
        "o2_run_promote",
        "o2_run_archive",
        "o2_run_gc",
    }
    assert tl["o2_run_archive"].annotations.destructiveHint is True
    assert tl["o2_run_list"].annotations.readOnlyHint is True
    assert tl["o2_run_register"].annotations.readOnlyHint is False


@pytest.mark.anyio
async def test_run_list_roundtrips_through_mcp(tmp_path):
    payload = await _call(_build(tmp_path), "o2_run_list", {"params": {}})
    assert payload["ok"] is True and payload["runs"] == []  # empty registry


@pytest.mark.anyio
async def test_run_register_refuses_without_dataset(tmp_path):
    from mcp.server.fastmcp.exceptions import ToolError

    mcp = _build(tmp_path)
    with pytest.raises(ToolError):  # pydantic min_length=1 rejects empty datasets
        await mcp.call_tool("o2_run_register", {"params": {"campaign": "c", "pipeline": "grid", "datasets": []}})
