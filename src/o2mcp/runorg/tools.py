"""MCP tool wrappers for the run-organization engine (registered by a consumer).

:func:`register` attaches the run-org tools (``o2_submit_run`` plus
``o2_run_register``/``list``/``show``/``classify``/``promote``/``archive``/``gc``)
to a consumer's FastMCP server. The consumer supplies two things:

- ``runs_factory()`` → a fresh :class:`~o2mcp.runorg.executor.O2Runs` per call (its own
  connection + the project's :class:`~o2mcp.runorg.policy.RunPolicy` baked in), and
- ``run_tool(work)`` → the consumer's async wrapper that runs the blocking ``work``
  off the event loop and JSON-encodes the result, turning lock/master errors into
  non-fatal payloads (e.g. ``o2mcp.server._run_tool``).

Dashboards are deliberately NOT here — they are project-specific (see the o2-mcp#1
design note). This module needs the optional ``mcp``/``pydantic`` deps, so it is NOT
imported by ``o2mcp.__init__`` (kept stdlib-only); import it explicitly to register.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from o2mcp.runorg.executor import O2Runs, TransitionPlan

RunsFactory = Callable[[], O2Runs]
RunToolWrapper = Callable[[Callable[[], dict]], Awaitable[str]]


# --- input models ------------------------------------------------------------
class SubmitRunInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    remote_script_path: str | None = Field(
        default=None, description="Path to an sbatch script already on O2 (may use {RUN_ROOT}/{RUN_ID})."
    )
    script_text: str | None = Field(
        default=None,
        description="sbatch script contents staged into the run dir (may use {RUN_ROOT}/{RUN_ID} placeholders).",
    )
    remote_path: str | None = Field(
        default=None, description="Where to stage script_text (default: <RUN_ROOT>/job.sbatch)."
    )
    sbatch_args: list[str] = Field(default_factory=list, description="Extra sbatch flags, e.g. ['--time=02:00:00'].")
    run_dir: str | None = Field(
        default=None, description="Attach the job to this existing run dir (the path o2_run_register returned)."
    )
    campaign: str | None = Field(
        default=None, description="Register a fresh run under this campaign (requires pipeline + datasets)."
    )
    pipeline: str | None = Field(
        default=None, description="Pipeline name for a new run (one of the project's RunPolicy pipelines)."
    )
    datasets: list[str] = Field(default_factory=list, description="Source dataset names for a new run.")
    variant: str = Field(default="", description="Short variant label distinguishing the new run within its campaign.")
    derived_from: str | None = Field(
        default=None, description="Source data path the new run references (so inputs are not re-copied)."
    )


class RunRegisterInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    campaign: str = Field(..., description="Campaign key grouping related runs.", min_length=1)
    pipeline: str = Field(..., description="Pipeline name (one of the project's RunPolicy pipelines).", min_length=1)
    datasets: list[str] = Field(
        ..., description="Source dataset names (e.g. dated raw experiment folders).", min_length=1
    )
    variant: str = Field(default="", description="Short variant label distinguishing this run within the campaign.")
    derived_from: str | None = Field(
        default=None,
        description="Source data path the run references (recorded so inputs are NOT copied into the run).",
    )


class RunListInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    campaign: str | None = Field(default=None, description="Filter to one campaign.")
    status: str | None = Field(default=None, description="Filter to a lifecycle status (active|kept|archived|purged).")


class RunDirInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    run_dir: str = Field(..., description="Absolute path of the run directory on O2.", min_length=1)


class RunClassifyInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    root: str | None = Field(default=None, description="Runs root to scan (defaults to the scratch runs root).")
    depth_grouped: bool = Field(
        default=False, description="True for the campaign-nested layout; False for a legacy flat RUN_* tree."
    )


class RunTransitionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    run_dir: str = Field(..., description="Absolute path of the run directory to move.", min_length=1)
    dry_run: bool = Field(default=True, description="When true, return the generated script without executing it.")


class RunGcInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    older_than_days: int = Field(
        default=30, description="List active scratch runs untouched for at least this many days.", ge=1
    )


def _plan_payload(plan: TransitionPlan) -> dict[str, Any]:
    return {
        "run_id": plan.run_id,
        "action": plan.action,
        "started": plan.started,
        "pid": plan.pid,
        "log_path": plan.log_path,
        "message": plan.message,
        "script": plan.script,
    }


def register(mcp, runs_factory: RunsFactory, run_tool: RunToolWrapper) -> None:
    """Register the run-org tools on ``mcp``, building a fresh O2Runs via ``runs_factory`` per call."""

    @mcp.tool(
        name="o2_submit_run",
        annotations={"title": "Submit a job into a registered run", "readOnlyHint": False, "openWorldHint": True},
    )
    async def o2_submit_run(params: SubmitRunInput) -> str:
        """Register (or attach to) a run, submit a job into it, and record the job id.

        Every job is tied to a registered run so the durable registry stays the source of
        truth for what ran on which dataset. Provide EITHER run_dir (attach) OR
        campaign+pipeline+datasets (register a fresh run first). The script may reference
        the run via {RUN_ROOT}/{RUN_ID}. Returns {ok, run_id, run_dir, submitted, job_id, …}.
        """

        def work() -> dict[str, Any]:
            return runs_factory().submit_run(
                remote_script_path=params.remote_script_path,
                script_text=params.script_text,
                remote_path=params.remote_path,
                sbatch_args=params.sbatch_args,
                run_dir=params.run_dir,
                campaign=params.campaign,
                pipeline=params.pipeline,
                datasets=params.datasets,
                variant=params.variant,
                derived_from=params.derived_from,
            )

        return await run_tool(work)

    @mcp.tool(
        name="o2_run_register",
        annotations={"title": "Register a new O2 run", "readOnlyHint": False, "openWorldHint": True},
    )
    async def o2_run_register(params: RunRegisterInput) -> str:
        """Allocate a run directory per convention and seed its run.json; returns RUN_ROOT.

        The sanctioned way to start a run: write all outputs under the returned RUN_ROOT.
        Refused unless campaign, pipeline, and ≥1 dataset are given (no unclassified runs).
        """

        def work() -> dict[str, Any]:
            return {
                "ok": True,
                **runs_factory().register(
                    campaign=params.campaign,
                    pipeline=params.pipeline,
                    datasets=params.datasets,
                    variant=params.variant,
                    derived_from=params.derived_from,
                ),
            }

        return await run_tool(work)

    @mcp.tool(
        name="o2_run_list",
        annotations={"title": "List registered O2 runs", "readOnlyHint": True, "openWorldHint": True},
    )
    async def o2_run_list(params: RunListInput) -> str:
        """Query the durable run registry (filter by campaign and/or status)."""

        def work() -> dict[str, Any]:
            rows = runs_factory().load_registry()
            if params.campaign:
                rows = [r for r in rows if r.get("campaign") == params.campaign]
            if params.status:
                rows = [r for r in rows if r.get("status") == params.status]
            return {"ok": True, "runs": rows}

        return await run_tool(work)

    @mcp.tool(
        name="o2_run_show",
        annotations={"title": "Show one O2 run manifest", "readOnlyHint": True, "openWorldHint": True},
    )
    async def o2_run_show(params: RunDirInput) -> str:
        """Return the full run.json for a run dir (synthesized from legacy metadata if absent)."""

        def work() -> dict[str, Any]:
            manifest = runs_factory().read_manifest(params.run_dir)
            if manifest is None:
                return {"ok": False, "error": "not_found", "message": f"no run metadata under {params.run_dir}"}
            return {"ok": True, "manifest": json.loads(manifest.to_json())}

        return await run_tool(work)

    @mcp.tool(
        name="o2_run_classify",
        annotations={"title": "Classify runs keep/sweep", "readOnlyHint": True, "openWorldHint": True},
    )
    async def o2_run_classify(params: RunClassifyInput) -> str:
        """Read every run under a root and advise keep (→ group) vs sweep (→ standby).

        Read-only and heuristic (per the project's RunPolicy markers + latest-COMPLETED
        rule). Review this before promoting/archiving. Returns rows with retention + reason.
        """

        def work() -> dict[str, Any]:
            rows = runs_factory().classify(params.root, depth_grouped=params.depth_grouped)
            keep = sum(1 for r in rows if r["retention"] == "keep")
            return {"ok": True, "total": len(rows), "keep": keep, "sweep": len(rows) - keep, "rows": rows}

        return await run_tool(work)

    @mcp.tool(
        name="o2_run_promote",
        annotations={"title": "Promote a run to group", "readOnlyHint": False, "openWorldHint": True},
    )
    async def o2_run_promote(params: RunTransitionInput) -> str:
        """Promote an active run to durable group storage (active → kept).

        Copies to the group runs root, verifies the copy, flips its manifest to 'kept', then
        frees scratch. Runs detached on the transfer node; returns a pid + log path. dry_run
        (default true) returns the generated script without executing it.
        """

        def work() -> dict[str, Any]:
            return {"ok": True, **_plan_payload(runs_factory().promote(params.run_dir, dry_run=params.dry_run))}

        return await run_tool(work)

    @mcp.tool(
        name="o2_run_archive",
        annotations={
            "title": "Archive a run to standby",
            "readOnlyHint": False,
            "destructiveHint": True,
            "openWorldHint": True,
        },
    )
    async def o2_run_archive(params: RunTransitionInput) -> str:
        """Archive a run cold to standby as a verified tar.zst, then free scratch.

        Destructive once executed: after the tarball is written, checksummed, and
        integrity-tested, the scratch copy is removed. Excludes are taken from the
        project's RunPolicy. Runs detached on the transfer node; returns a pid + log path.
        dry_run (default true) returns the script without executing it.
        """

        def work() -> dict[str, Any]:
            return {"ok": True, **_plan_payload(runs_factory().archive(params.run_dir, dry_run=params.dry_run))}

        return await run_tool(work)

    @mcp.tool(
        name="o2_run_gc",
        annotations={"title": "Audit purgeable runs", "readOnlyHint": True, "openWorldHint": True},
    )
    async def o2_run_gc(params: RunGcInput) -> str:
        """List active scratch runs untouched for ≥ N days (purge-risk audit; advisory)."""

        def work() -> dict[str, Any]:
            return {"ok": True, "candidates": runs_factory().gc_candidates(older_than_days=params.older_than_days)}

        return await run_tool(work)
