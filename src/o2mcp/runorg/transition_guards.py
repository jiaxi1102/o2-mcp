"""Preconditions for destructive execution-run lifecycle transitions."""

from __future__ import annotations

import shlex

from o2mcp.runorg.runs import RunManifest


def require_certified_terminal_execution(manifest: RunManifest, action: str) -> None:
    """Reject deletion-capable transitions before execution is certified terminal.

    Promotion is a release action and therefore requires successful completion.
    Archive may preserve either a completed or a failed run for audit, but both
    manifest surfaces must agree with authenticated execution provenance.
    """

    execution = (manifest.provenance or {}).get("execution", {})
    execution_state = execution.get("state") if isinstance(execution, dict) else None
    result_state = (manifest.result or {}).get("status")
    allowed = {"COMPLETED"} if action == "promote" else {"COMPLETED", "FAILED"}
    if execution_state not in allowed or result_state != execution_state:
        raise ValueError(
            f"{action} requires matching certified terminal execution/result state; "
            f"observed execution={execution_state!r}, result={result_state!r}"
        )


def live_jobs_command(job_ids: list[str]) -> str:
    """Return a fail-closed squeue query for the manifest's exact numeric jobs."""

    if any(not isinstance(job_id, str) or not job_id.isdigit() for job_id in job_ids):
        raise ValueError("transition manifest contains a nonnumeric Slurm job ID")
    if not job_ids:
        return "printf ''"
    joined = ",".join(job_ids)
    return f"squeue -h -j {shlex.quote(joined)} -o '%i|%T'"


__all__ = ["live_jobs_command", "require_certified_terminal_execution"]
