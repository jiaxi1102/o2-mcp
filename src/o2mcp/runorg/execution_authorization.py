"""Signed task/dependency authorization for one scheduler attempt."""

from __future__ import annotations

from o2mcp.runorg.execution_backend import ExecutionBackend
from o2mcp.runorg.execution_evidence import read_reconciliation_receipt, read_strict_json, read_submission_rejection
from o2mcp.runorg.execution_models import RECONCILE_RETRY, SubmissionIdentity
from o2mcp.runorg.execution_paths import reconciler_followup_path, submission_rejection_path
from o2mcp.runorg.execution_rendering import select_tasks
from o2mcp.runorg.plan_stages import StageSpec
from o2mcp.runorg.plans import ExecutionPlan


def derive_attempt_authorization(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
    attempt: int,
    signed_dependency_jobs: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the sole plan-authorized task and dependency set for an attempt."""

    all_tasks = select_tasks(stage, None)
    all_task_ids = tuple(task.task_id for task in all_tasks)
    if attempt == 1:
        return all_task_ids, signed_dependency_jobs

    if stage.dependency_mode == "afterany" and stage.depends_on and not stage.tasks:
        path = reconciler_followup_path(plan, stage.stage_id, attempt)
        followup = read_strict_json(backend, path, "reconciler follow-up authorization")
        allowed = {
            "attempt",
            "dependency_job_ids",
            "plan_sha256",
            "schema_version",
            "stage_id",
            "trigger_job_id",
            "trigger_stage_id",
        }
        dependencies = followup.get("dependency_job_ids")
        if (
            set(followup) != allowed
            or followup.get("schema_version") != 1
            or followup.get("plan_sha256") != plan.plan_sha256
            or followup.get("stage_id") != stage.stage_id
            or followup.get("attempt") != attempt
            or not isinstance(dependencies, list)
            or any(not isinstance(item, str) or not item.isdigit() for item in dependencies)
            or len(dependencies) != len(stage.depends_on)
            or len(set(dependencies)) != len(dependencies)
        ):
            raise ValueError("reconciler follow-up authorization is malformed or foreign")
        return all_task_ids, tuple(dependencies)

    previous = read_reconciliation_receipt(backend, plan, stage, attempt - 1)
    if previous is not None:
        if previous.decision != RECONCILE_RETRY:
            raise ValueError("retry authorization receipt does not authorize a next attempt")
        task_ids = previous.retry_task_ids
    else:
        identity = SubmissionIdentity(plan.plan_sha256, stage.stage_id, attempt - 1)
        rejection = read_submission_rejection(backend, submission_rejection_path(plan, identity))
        if rejection is None or rejection.identity != identity:
            raise ValueError("retry attempt has no preceding reconciliation or rejection authorization")
        tasks = select_tasks(stage, rejection.task_ids)
        indices = tuple(task.array_index for task in tasks if task.array_index is not None)
        if rejection.task_indices != indices:
            raise ValueError("submission rejection task indices do not match the signed stage")
        task_ids = tuple(task.task_id for task in tasks)
    return tuple(task_ids), signed_dependency_jobs


__all__ = ["derive_attempt_authorization"]
