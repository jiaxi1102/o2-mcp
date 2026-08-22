"""Preconditions for destructive execution-run lifecycle transitions."""

from __future__ import annotations

import shlex

from o2mcp.runorg.execution_backend import ExecutionBackend, receipt_matches
from o2mcp.runorg.execution_evidence import current_task_receipts_valid, latest_reconciliation_receipt
from o2mcp.runorg.execution_models import RECONCILE_COMPLETE, RECONCILE_FAILED
from o2mcp.runorg.plans import ExecutionPlan
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


def require_current_terminal_evidence(backend: ExecutionBackend, plan: ExecutionPlan, action: str) -> None:
    """Revalidate terminal receipts from current bytes before source deletion.

    Registry state is only a cached projection.  A completed reconciliation whose
    task or aggregate receipts were later deleted cannot certify promotion.  A
    failed archive remains auditable only when at least one authenticated failed
    reconciliation still exists; every completed stage is revalidated as well.
    """

    receipts = {stage.stage_id: latest_reconciliation_receipt(backend, plan, stage) for stage in plan.stages}
    failed = {
        stage_id
        for stage_id, receipt in receipts.items()
        if receipt is not None and receipt.decision == RECONCILE_FAILED
    }
    if action == "promote" and failed:
        raise ValueError(f"promote refuses failed stages: {', '.join(sorted(failed))}")
    if action == "archive" and not failed and any(receipt is None for receipt in receipts.values()):
        raise ValueError("archive lacks an authenticated failed or complete terminal decision")

    stages = {stage.stage_id: stage for stage in plan.stages}

    def blocked_by_failure(stage_id: str, seen: set[str] | None = None) -> bool:
        """Return whether an unscheduled stage descends from a failed stage."""

        seen = set() if seen is None else seen
        if stage_id in seen:
            return False
        seen.add(stage_id)
        dependencies = stages[stage_id].depends_on
        return any(item in failed or blocked_by_failure(item, seen.copy()) for item in dependencies)

    for stage in plan.stages:
        receipt = receipts[stage.stage_id]
        if receipt is None:
            # Only after-ok descendants are scheduler-suppressed by an upstream
            # failure. After-any work is required to run after that failure and
            # must therefore produce its own authenticated terminal receipt.
            if action == "archive" and stage.dependency_mode == "afterok" and blocked_by_failure(stage.stage_id):
                continue
            raise ValueError(f"{action} stage {stage.stage_id} lacks authenticated terminal reconciliation")
        if receipt.decision == RECONCILE_FAILED:
            continue
        if receipt.decision != RECONCILE_COMPLETE:
            raise ValueError(f"{action} stage {stage.stage_id} is not terminal")
        if not current_task_receipts_valid(backend, plan, stage):
            raise ValueError(f"{action} stage {stage.stage_id} current task receipts are missing or changed")
        observations = tuple(backend.observe_receipt(plan.paths.run_root, spec) for spec in stage.expected_receipts)
        if any(not item.trustworthy for item in observations) or not all(
            receipt_matches(spec, item) for spec, item in zip(stage.expected_receipts, observations)
        ):
            raise ValueError(f"{action} stage {stage.stage_id} current stage receipts are missing or changed")


__all__ = ["live_jobs_command", "require_certified_terminal_execution", "require_current_terminal_evidence"]
