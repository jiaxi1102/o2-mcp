#!/usr/bin/env python3
"""MCP server for HMS O2 cluster access (``o2_mcp``).

Exposes the safe, tested ``o2mcp`` primitives as MCP tools so an agent can
directly submit Slurm work, debug with arbitrary remote commands, monitor jobs,
tail logs, and move files.

DUO MODEL — read this before using the tools. HMS O2 uses Duo *autopush*: EVERY
new SSH connection fires a Duo push, even key-only/BatchMode (this is not
keyboard-interactive, so it cannot be disabled client-side). The tools are built
to make this cost exactly ONE push per session:

  1. Call ``o2_start_master`` ONCE (one push you approve) to open a persistent
     ControlMaster. It stays up for ~8h (ControlPersist).
  2. Every other tool (run/submit/squeue/status/tail/cancel/push/pull and the
     non-blocking push_async/pull_async) REUSES that master and costs NO additional
     push. (o2_transfer_status/o2_transfer_cancel read local state only — no SSH.)

NEVER open the master in a loop, and never run these tools on a short timer — a
periodic poller that reconnects each cycle is what causes a "Duo call every
minute". To completely avoid Duo, don't poll O2 from here at all (have O2 push
results out via Globus/OnDemand) or ask HMS RC for SSH-certificate access; see
``docs/O2_MCP.md``.

The ``.agent_locks/O2_DISABLED`` lock is honored as a hard stop on every tool.

Run as a local stdio server:

    python -m o2mcp.server        # or the `o2-mcp` console script

Requires the optional ``mcp`` dependency (``pip install -e ".[o2]"``) and Python
>= 3.10. The underlying primitives in ``o2mcp`` work on Python 3.9 and are
unit-tested without it.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

import anyio
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from o2mcp import (
    CommandResult,
    O2Connection,
    O2LockedError,
    O2MasterUnavailableError,
    O2OffVpnError,
    O2Slurm,
    O2Workspace,
    transfer_tools,
)

mcp = FastMCP("o2_mcp")


# --- shared helpers ----------------------------------------------------------
def _connection() -> O2Connection:
    """A fresh connection (config read from env); the ControlMaster persists out-of-process."""
    return O2Connection()


def _command_payload(result: CommandResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


async def _run_tool(fn: Callable[[], dict[str, Any]]) -> str:
    """Run a blocking O2 operation off the event loop and JSON-encode the result.

    Lock and master-availability errors are turned into actionable, non-fatal
    payloads instead of crashing the tool call.
    """
    try:
        payload = await anyio.to_thread.run_sync(fn)
    except O2LockedError as exc:
        payload = {"ok": False, "error": "o2_locked", "message": str(exc)}
    except O2MasterUnavailableError as exc:
        payload = {"ok": False, "error": "no_master", "message": str(exc)}
    except O2OffVpnError as exc:
        payload = {"ok": False, "error": "off_vpn", "message": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive
        payload = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
    return json.dumps(payload, indent=2)


# --- input models ------------------------------------------------------------
class StartMasterInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allow_new_login: bool = Field(
        default=False,
        description=(
            "Must be true to open a NEW O2 login. O2 autopushes Duo on every new "
            "connection, so this costs exactly one push; all later tools reuse the "
            "master for free. Do this once per session — never in a loop."
        ),
    )
    transfer: bool = Field(
        default=False,
        description=(
            "Open the dedicated transfer-node master (o2-transfer) instead of the "
            "login master. The transfer node is a separate host with its own socket, "
            "so a transfer-node move (o2_run_promote / o2_run_archive, or o2_push/pull "
            "with use_transfer_node) needs this opened once (one more approved push) "
            "in addition to the login master."
        ),
    )
    allow_offvpn: bool = Field(
        default=False,
        description=(
            "Override the HMS-VPN egress guard. By default opening a master is refused "
            "when the route to O2 does NOT go through a VPN tunnel interface, because a "
            "login from a non-HMS IP triggers a Duo push. Set true only if you intend to "
            "connect off-VPN and accept that push (or set O2_REQUIRE_VPN=0 to disable)."
        ),
    )


class RunInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    command: str = Field(..., description="Remote shell command to run on an O2 login node.", min_length=1)
    timeout_seconds: float = Field(default=120.0, description="Command timeout in seconds.", gt=0, le=3600)


class SubmitInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    remote_script_path: str | None = Field(
        default=None, description="Path to an sbatch script that already exists on O2."
    )
    script_text: str | None = Field(
        default=None, description="sbatch script contents to stage to O2 before submitting (use with remote_path)."
    )
    remote_path: str | None = Field(
        default=None, description="Where to stage script_text on O2 (required when script_text is given)."
    )
    sbatch_args: list[str] = Field(default_factory=list, description="Extra sbatch flags, e.g. ['--time=02:00:00'].")


class QueueInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    user: str | None = Field(default=None, description="Username for squeue -u (defaults to remote $USER).")


class JobIdInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    job_id: str = Field(..., description="Slurm job id.", min_length=1)


class TailLogInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    remote_path: str = Field(..., description="Path to the remote log file on O2.", min_length=1)
    lines: int = Field(default=100, description="Number of trailing lines to show.", ge=1, le=10000)


# --- workspace-layout inputs (see o2mcp.workspace / docs/WORKSPACE_LAYOUT.md) ---
class WorkspaceReportInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    roots: list[str] | None = Field(
        default=None, description="Roots to scan (default: the home and scratch tier roots)."
    )


class WorkspaceGcInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    dry_run: bool = Field(default=True, description="When true, return the prune script without executing it.")
    roots: list[str] | None = Field(default=None, description="Roots to scan (default: home and scratch tier roots).")


class PlaceInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    kind: str = Field(
        ...,
        description="Artifact kind: results|data|runs_active|runs_kept|registry|work|staging|logs|archive.",
        min_length=1,
    )
    project: str | None = Field(default=None, description="Optional project name to nest under the kind's root.")


# --- tools -------------------------------------------------------------------
@mcp.tool(
    name="o2_status",
    annotations={"title": "O2 connection status", "readOnlyHint": True, "openWorldHint": True},
)
async def o2_status() -> str:
    """Report O2 access state: safety lock, ControlMaster, and a connectivity probe.

    Returns JSON: {"locked": bool, "master_running": bool, "probe": {...}|null}.
    When no master is running, probe is null and you must start one
    (o2_start_master) before running commands or submitting jobs.
    """

    def work() -> dict[str, Any]:
        conn = _connection()
        locked = conn.is_locked()
        master = (not locked) and conn.master_running()
        probe = _command_payload(conn.probe()) if master else None
        return {"ok": True, "locked": locked, "master_running": master, "probe": probe}

    return await _run_tool(work)


@mcp.tool(
    name="o2_start_master",
    annotations={"title": "Start O2 SSH master", "readOnlyHint": False, "openWorldHint": True},
)
async def o2_start_master(params: StartMasterInput) -> str:
    """Open the persistent O2 SSH ControlMaster so later tools reuse one login.

    HMS O2 autopushes Duo on every new connection, so opening the master costs
    exactly ONE push (which you approve); after that every tool reuses the master
    with no further push for ~8h. Refused unless allow_new_login is true. If a
    master is already running this is a no-op (and costs nothing). Call this once
    per session — never on a timer or in a loop.
    """

    def work() -> dict[str, Any]:
        conn = _connection()
        alias = conn.config.transfer_alias if params.transfer else None
        result = conn.start_master(
            allow_new_login=params.allow_new_login, alias=alias, allow_offvpn=params.allow_offvpn
        )
        return {"ok": result.ok, "alias": alias or conn.config.host_alias, **_command_payload(result)}

    return await _run_tool(work)


@mcp.tool(
    name="o2_run",
    annotations={"title": "Run a command on O2", "readOnlyHint": False, "openWorldHint": True},
)
async def o2_run(params: RunInput) -> str:
    """Run an arbitrary shell command on an O2 login node (debugging / inspection).

    Reuses the existing ControlMaster (no extra Duo push) and refuses if none is
    running — start one first with o2_start_master. Returns JSON with
    returncode/stdout/stderr.
    """

    def work() -> dict[str, Any]:
        result = _connection().run(params.command, timeout=params.timeout_seconds)
        return {"ok": result.ok, **_command_payload(result)}

    return await _run_tool(work)


@mcp.tool(
    name="o2_submit_job",
    annotations={"title": "Submit a Slurm job", "readOnlyHint": False, "openWorldHint": True},
)
async def o2_submit_job(params: SubmitInput) -> str:
    """Submit an sbatch job, returning the parsed Slurm job id.

    Provide either remote_script_path (a script already on O2) or script_text +
    remote_path (stage the script to O2, then submit). Returns JSON:
    {"submitted": bool, "job_id": str|null, "returncode": int, "stdout": str, "stderr": str}.
    """

    def work() -> dict[str, Any]:
        slurm = O2Slurm(_connection())
        if params.script_text is not None:
            if not params.remote_path:
                return {"ok": False, "error": "bad_input", "message": "remote_path is required with script_text."}
            res = slurm.submit_text(params.script_text, params.remote_path, sbatch_args=params.sbatch_args)
        elif params.remote_script_path:
            res = slurm.submit(params.remote_script_path, sbatch_args=params.sbatch_args)
        else:
            return {
                "ok": False,
                "error": "bad_input",
                "message": "Provide remote_script_path or script_text+remote_path.",
            }
        return {"ok": res.submitted, "submitted": res.submitted, "job_id": res.job_id, **_command_payload(res.command)}

    return await _run_tool(work)


@mcp.tool(
    name="o2_squeue",
    annotations={"title": "List Slurm jobs", "readOnlyHint": True, "openWorldHint": True},
)
async def o2_squeue(params: QueueInput) -> str:
    """List the user's current Slurm jobs (squeue) as structured rows.

    Returns JSON: {"jobs": [{"job_id","name","state","elapsed","time_limit","nodes","reason"}, ...]}.
    """

    def work() -> dict[str, Any]:
        return {"ok": True, "jobs": O2Slurm(_connection()).queue(params.user)}

    return await _run_tool(work)


@mcp.tool(
    name="o2_job_status",
    annotations={"title": "Slurm job accounting", "readOnlyHint": True, "openWorldHint": True},
)
async def o2_job_status(params: JobIdInput) -> str:
    """Get sacct accounting for one job (state, elapsed, exit code, memory).

    Returns JSON: {"rows": [{"job_id","name","state","elapsed","exit_code","max_rss",...}, ...]}.
    """

    def work() -> dict[str, Any]:
        return {"ok": True, "rows": O2Slurm(_connection()).job_status(params.job_id)}

    return await _run_tool(work)


@mcp.tool(
    name="o2_tail_log",
    annotations={"title": "Tail an O2 log", "readOnlyHint": True, "openWorldHint": True},
)
async def o2_tail_log(params: TailLogInput) -> str:
    """Tail the last N lines of a remote Slurm log file.

    Logs typically live under ~/logs/o2/<job-name>_<jobid>.out|.err.
    Returns JSON with the log text in stdout.
    """

    def work() -> dict[str, Any]:
        result = O2Slurm(_connection()).tail_log(params.remote_path, lines=params.lines)
        return {"ok": result.ok, **_command_payload(result)}

    return await _run_tool(work)


@mcp.tool(
    name="o2_cancel_job",
    annotations={
        "title": "Cancel a Slurm job",
        "readOnlyHint": False,
        "destructiveHint": True,
        "openWorldHint": True,
    },
)
async def o2_cancel_job(params: JobIdInput) -> str:
    """Cancel a running or queued Slurm job (scancel). Destructive: the job is killed."""

    def work() -> dict[str, Any]:
        result = O2Slurm(_connection()).cancel(params.job_id)
        return {"ok": result.ok, **_command_payload(result)}

    return await _run_tool(work)


# --- workspace-layout tools --------------------------------------------------
@mcp.tool(
    name="o2_disk_report",
    annotations={"title": "O2 disk usage + hygiene report", "readOnlyHint": True, "openWorldHint": True},
)
async def o2_disk_report(params: WorkspaceReportInput) -> str:
    """Per-tier disk usage with hygiene flags (regenerable / redundant / misplaced).

    Read-only. Walks the home + scratch tiers (depth 1), classifies each entry, and
    returns totals by disposition plus a reclaimable-bytes estimate. This is the
    repeatable, codified version of a manual disk audit — review it before running
    o2_workspace_gc. See docs/WORKSPACE_LAYOUT.md.
    """

    def work() -> dict[str, Any]:
        report = O2Workspace(_connection()).disk_report(params.roots)
        return {"ok": True, **report}

    return await _run_tool(work)


@mcp.tool(
    name="o2_workspace_gc",
    annotations={
        "title": "Prune regenerable + redundant disk",
        "readOnlyHint": False,
        "destructiveHint": True,
        "openWorldHint": True,
    },
)
async def o2_workspace_gc(params: WorkspaceGcInput) -> str:
    """Prune ONLY the regenerable (caches/builds) and redundant (trash/surplus
    snapshots) findings from o2_disk_report — never relocate/review/keep items.

    Snapshot history is thinned to the newest few; everything else prunable is
    removed. Runs detached and fail-closed. dry_run (default true) returns the
    generated script without executing it.
    """

    def work() -> dict[str, Any]:
        plan = O2Workspace(_connection()).gc(dry_run=params.dry_run, roots=params.roots)
        return {
            "ok": True,
            "dry_run": plan.dry_run,
            "submitted": plan.submitted,
            "pruned_paths": plan.pruned_paths,
            "message": plan.message,
            "script": plan.script,
        }

    return await _run_tool(work)


@mcp.tool(
    name="o2_place",
    annotations={"title": "Resolve canonical output path", "readOnlyHint": True, "openWorldHint": True},
)
async def o2_place(params: PlaceInput) -> str:
    """Resolve the canonical path for an output kind (+ optional project) per the
    workspace tier convention — so outputs land on the right tier, not invented paths.

    e.g. {kind:'results', project:'myproject'} -> /n/groups/tabin/jzhao/results/myproject.
    """

    def work() -> dict[str, Any]:
        return {
            "ok": True,
            "kind": params.kind,
            "project": params.project,
            "path": O2Workspace(_connection()).place(params.kind, params.project),
        }

    return await _run_tool(work)


transfer_tools.register(mcp, sys.modules[__name__])


def main() -> None:
    """Console-script / module entry point: run the stdio MCP server."""
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
