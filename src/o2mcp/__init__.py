"""Safe, testable HMS O2 cluster access (connection, Slurm, file transfer, workspace).

A generic, project-agnostic toolkit for the HMS O2 cluster: a Duo-frugal SSH
ControlMaster channel, Slurm submit/monitor, rsync transfers (blocking and
non-blocking), and workspace/disk hygiene. The pure pieces here are stdlib-only
and unit-tested offline. The MCP server that exposes them as agent tools lives in
``o2mcp.server`` and is imported separately (it needs the optional ``mcp``
dependency and Python >= 3.10).

Project-specific layers (e.g. run-organization for a particular pipeline) build
*on* this package rather than living in it.
"""

from __future__ import annotations

from o2mcp.async_transfer import O2AsyncTransfer, TransferHandle, default_spawner
from o2mcp.config import O2Config
from o2mcp.connection import (
    CommandResult,
    O2Connection,
    O2LockedError,
    O2MasterUnavailableError,
    default_runner,
)
from o2mcp.slurm import O2Slurm, SubmitResult
from o2mcp.sync import O2Sync
from o2mcp.workspace import WorkspaceLayout, classify_entry, plan_prune, summarize_report
from o2mcp.workspace_exec import GcPlan, O2Workspace

__all__ = [
    "O2Config",
    "O2Connection",
    "CommandResult",
    "O2LockedError",
    "O2MasterUnavailableError",
    "default_runner",
    "O2Slurm",
    "SubmitResult",
    "O2Sync",
    "O2AsyncTransfer",
    "TransferHandle",
    "default_spawner",
    "O2Workspace",
    "GcPlan",
    "WorkspaceLayout",
    "classify_entry",
    "summarize_report",
    "plan_prune",
]
