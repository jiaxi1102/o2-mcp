"""Signed task/dependency authorization for one scheduler attempt."""

from __future__ import annotations

from o2mcp.runorg.execution_backend import ExecutionBackend
from o2mcp.runorg.execution_evidence import (
    authenticate_followup_authorization,
    read_reconciliation_receipt,
    read_submission_rejection,
)
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

    # Both dependency modes can require a replacement generation. ``afterany``
    # children rebind as soon as an upstream retry is accepted; ``afterok``
    # children rebind only after that replacement dependency is authenticated
    # as complete. In both cases the immutable authorization fixes the exact
    # latest dependency job IDs and reruns the full signed task set.
    followup_authorized = bool(stage.depends_on) and (
        backend.read_text(reconciler_followup_path(plan, stage.stage_id, attempt)) is not None
    )
    if followup_authorized:
        dependencies = authenticate_followup_authorization(
            backend,
            plan,
            stage,
            attempt,
            expected_dependency_job_ids=signed_dependency_jobs,
        )
        return all_task_ids, dependencies

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
