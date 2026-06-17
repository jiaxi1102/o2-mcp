"""MCP file-transfer tools for O2: blocking push/pull + non-blocking async.

Split out of ``server.py`` so each module stays within the repo file-length budget
and the transfer surface is a single focused unit (it is also the slice that moves
to a standalone o2-mcp package later).

:func:`register` attaches the tools to the *running* server's FastMCP instance,
using its shared helpers (``srv._connection`` etc.) looked up at call time. This
matters two ways: tests that monkeypatch ``server._connection`` keep working, and
the tools attach to the live instance even under ``python -m o2mcp.server``
(where the running module is ``__main__``, not a re-imported ``o2mcp.server`` —
a self-import would register on a *duplicate* FastMCP that is never served).
``O2AsyncTransfer``/``O2Sync`` are module globals (the async tests patch them here).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from o2mcp import O2AsyncTransfer, O2Sync
from o2mcp.async_transfer import TransferHandle


# --- input models ------------------------------------------------------------
class TransferInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    local_path: str = Field(..., description="Local path.", min_length=1)
    remote_path: str = Field(..., description="Remote path on O2.", min_length=1)
    use_transfer_node: bool = Field(
        default=False, description="Use the dedicated O2 transfer node alias (for large transfers)."
    )


class TransferStatusInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    transfer_id: str | None = Field(
        default=None,
        description="A specific transfer id (from o2_push_async/o2_pull_async); omit to list ALL known transfers.",
    )
    log_tail: int = Field(default=20, description="Trailing rsync log lines to include per transfer.", ge=0, le=2000)


class TransferIdInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    transfer_id: str = Field(
        ..., description="The transfer id returned by o2_push_async / o2_pull_async.", min_length=1
    )


def _handle_payload(handle: TransferHandle) -> dict[str, Any]:
    """The agent-facing summary of a just-launched background transfer."""
    return {
        "transfer_id": handle.id,
        "direction": handle.direction,
        "pid": handle.pid,
        "local": handle.local,
        "remote": handle.remote,
        "log_path": handle.log_path,
        "message": (
            "Transfer started in the background and is running detached. Poll o2_transfer_status "
            "with this transfer_id (it keeps running between tool calls); re-run the same command "
            "to resume if it is interrupted (rsync --partial)."
        ),
    }


def register(mcp, srv) -> None:
    """Register the transfer tools on ``mcp``, using helpers from the running ``srv`` module.

    ``mcp`` and ``srv`` are the *live* FastMCP instance and server module (passed by
    server.py as ``sys.modules[__name__]``), so the tools attach to the served instance
    and resolve ``srv._connection``/``_run_tool``/``_command_payload`` at call time.
    """

    @mcp.tool(
        name="o2_push",
        annotations={"title": "Upload files to O2", "readOnlyHint": False, "openWorldHint": True},
    )
    async def o2_push(params: TransferInput) -> str:
        """rsync a local file/directory up to O2 (reuses the ControlMaster, no extra login)."""

        def work() -> dict[str, Any]:
            result = O2Sync(srv._connection()).push(
                params.local_path, params.remote_path, transfer=params.use_transfer_node
            )
            return {"ok": result.ok, **srv._command_payload(result)}

        return await srv._run_tool(work)

    @mcp.tool(
        name="o2_pull",
        annotations={"title": "Download files from O2", "readOnlyHint": False, "openWorldHint": True},
    )
    async def o2_pull(params: TransferInput) -> str:
        """rsync a remote O2 file/directory down to the local machine."""

        def work() -> dict[str, Any]:
            result = O2Sync(srv._connection()).pull(
                params.remote_path, params.local_path, transfer=params.use_transfer_node
            )
            return {"ok": result.ok, **srv._command_payload(result)}

        return await srv._run_tool(work)

    @mcp.tool(
        name="o2_push_async",
        annotations={"title": "Upload files to O2 (non-blocking)", "readOnlyHint": False, "openWorldHint": True},
    )
    async def o2_push_async(params: TransferInput) -> str:
        """Start an rsync UPLOAD in the background and return immediately (does NOT block).

        Prefer this over o2_push for large/slow uploads: it launches a detached rsync
        (reusing the ControlMaster — no extra Duo) and returns a transfer_id right away,
        so you can do other work while it runs and poll o2_transfer_status when you want.
        The transfer keeps going between tool calls and survives an MCP-server restart;
        re-running the same upload resumes it (rsync --partial). Refuses unless the
        ControlMaster is already up (start it once with o2_start_master).
        """

        def work() -> dict[str, Any]:
            handle = O2AsyncTransfer(srv._connection()).push_async(
                params.local_path, params.remote_path, transfer=params.use_transfer_node
            )
            return {"ok": True, **_handle_payload(handle)}

        return await srv._run_tool(work)

    @mcp.tool(
        name="o2_pull_async",
        annotations={"title": "Download files from O2 (non-blocking)", "readOnlyHint": False, "openWorldHint": True},
    )
    async def o2_pull_async(params: TransferInput) -> str:
        """Start an rsync DOWNLOAD in the background and return immediately (does NOT block).

        The non-blocking counterpart of o2_pull (see o2_push_async). Returns a
        transfer_id to poll with o2_transfer_status.
        """

        def work() -> dict[str, Any]:
            handle = O2AsyncTransfer(srv._connection()).pull_async(
                params.remote_path, params.local_path, transfer=params.use_transfer_node
            )
            return {"ok": True, **_handle_payload(handle)}

        return await srv._run_tool(work)

    @mcp.tool(
        name="o2_transfer_status",
        annotations={"title": "Status of async O2 transfers", "readOnlyHint": True, "openWorldHint": False},
    )
    async def o2_transfer_status(params: TransferStatusInput) -> str:
        """Report progress/state of background transfers (o2_push_async / o2_pull_async).

        With a transfer_id: that transfer's state (running | done | failed | crashed),
        rsync exit code, files done/total, last file, elapsed time, and a log tail.
        Without one: a list of every known transfer. Reads only local state files — no
        SSH, no Duo, safe to poll freely.
        """

        def work() -> dict[str, Any]:
            result = O2AsyncTransfer(srv._connection()).status(params.transfer_id, log_tail=params.log_tail)
            if params.transfer_id is None:
                return {"ok": True, "transfers": result}
            return result

        return await srv._run_tool(work)

    @mcp.tool(
        name="o2_transfer_cancel",
        annotations={
            "title": "Cancel an async O2 transfer",
            "readOnlyHint": False,
            "destructiveHint": True,
            "openWorldHint": False,
        },
    )
    async def o2_transfer_cancel(params: TransferIdInput) -> str:
        """Terminate a running background transfer (SIGTERM to its rsync process group)."""

        def work() -> dict[str, Any]:
            return O2AsyncTransfer(srv._connection()).cancel(params.transfer_id)

        return await srv._run_tool(work)
