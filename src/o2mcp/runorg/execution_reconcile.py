"""Pure task/state helpers for files-as-truth execution reconciliation."""

from __future__ import annotations

from collections.abc import Sequence

from o2mcp.runorg.execution_models import TERMINAL_SLURM_STATES, PlannedTask, SlurmTaskState
from o2mcp.runorg.plan_stages import StageSpec
from o2mcp.runorg.plans import ExecutionPlan


def stage_by_id(plan: ExecutionPlan, stage_id: str) -> StageSpec:
    """Return one signed stage without accepting a caller-supplied replacement."""

    for stage in plan.stages:
        if stage.stage_id == stage_id:
            return stage
    raise ValueError(f"unknown stage_id {stage_id!r}")


def signed_attempt_bound(plan: ExecutionPlan, stage: StageSpec) -> int:
    """Return the plan-derived submission bound for one stage.

    An ``afterany`` non-array reconciler needs its own signed retry budget plus
    one new generation for every possible accepted retry of each dependency.
    Deriving the extension solely from immutable stage policies keeps the
    scheduler identity space bounded without requiring adapters to guess a
    fan-in-specific reconciler limit.
    """

    bound = stage.retry_policy.max_attempts
    if stage.dependency_mode == "afterany" and stage.depends_on and not stage.tasks:
        bound += sum(stage_by_id(plan, dependency).retry_policy.max_attempts - 1 for dependency in stage.depends_on)
    return bound


def task_sort_key(task: PlannedTask) -> tuple[int, int, str]:
    """Sort array indices numerically while keeping non-array stages deterministic."""

    return (0, task.array_index, task.task_id) if task.array_index is not None else (1, 0, task.task_id)


def task_state(task: PlannedTask, states: Sequence[SlurmTaskState]) -> SlurmTaskState | None:
    """Select a child state and preserve terminal root cancellation semantics."""

    for state in states:
        if state.array_index == task.array_index:
            return state
    roots = [state for state in states if state.array_index is None]
    if task.array_index is not None and roots and roots[-1].normalized_state() in TERMINAL_SLURM_STATES:
        # A cancelled/failed root is meaningful evidence, not generic absence.
        # Only a completed root missing an expected child is classified MISSING.
        root = roots[-1]
        normalized_root = root.normalized_state()
        if normalized_root == "COMPLETED":
            return SlurmTaskState(task.array_index, "MISSING", None)
        return SlurmTaskState(task.array_index, normalized_root, root.exit_code)
    return None


def is_retryable(stage: StageSpec, state: str, exit_code: int | None, receipts_valid: bool) -> bool:
    """Apply only retry conditions included in the immutable stage policy."""

    policy = stage.retry_policy
    signed_terminal_condition = state in policy.retryable_slurm_states or (
        exit_code is not None and exit_code in policy.retryable_exit_codes
    )
    # Missing-only never reinterprets an explicit terminal scheduler state.
    missing_task_condition = state == "MISSING" and not receipts_valid and policy.retry_missing_receipts
    return signed_terminal_condition or missing_task_condition


__all__ = ["is_retryable", "signed_attempt_bound", "stage_by_id", "task_sort_key", "task_state"]
