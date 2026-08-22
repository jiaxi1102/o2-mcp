"""Canonical internal receipt paths for the shared execution engine.

Pipeline-owned receipts are forbidden from this namespace.  Centralizing these
constructors prevents submission, reconciliation, and registry recovery from
silently disagreeing about where immutable operational evidence belongs.
"""

from __future__ import annotations

import posixpath

from o2mcp.runorg.execution_models import SubmissionIdentity
from o2mcp.runorg.plans import ExecutionPlan


def bound_plan_path(plan: ExecutionPlan) -> str:
    """Return the one-time run-to-plan binding path."""

    return posixpath.join(plan.paths.receipts_root, "execution", "execution-plan.json")


def submission_record_path(plan: ExecutionPlan, identity: SubmissionIdentity) -> str:
    """Return the immutable accepted-submission record path."""

    return posixpath.join(
        plan.paths.receipts_root,
        "execution",
        "submissions",
        identity.stage_id,
        f"attempt-{identity.attempt:03d}.json",
    )


def submission_intent_path(plan: ExecutionPlan, identity: SubmissionIdentity) -> str:
    """Return the atomic pre-sbatch ownership claim path."""

    return posixpath.join(
        plan.paths.receipts_root,
        "execution",
        "submission-intents",
        identity.stage_id,
        f"attempt-{identity.attempt:03d}.json",
    )


def submission_invocation_path(plan: ExecutionPlan, identity: SubmissionIdentity) -> str:
    """Return the no-replace marker for crossing the ``sbatch`` boundary.

    The intent and invocation markers deliberately represent different facts.
    Any process may recover an intent whose original creator died before the
    invocation marker was published.  Once invocation is published, however,
    only scheduler lookup may resolve the attempt: automatically taking over an
    invocation would risk duplicating a job whose accounting row is merely late.
    """

    return posixpath.join(
        plan.paths.receipts_root,
        "execution",
        "submission-invocations",
        identity.stage_id,
        f"attempt-{identity.attempt:03d}.json",
    )


def submission_rejection_path(plan: ExecutionPlan, identity: SubmissionIdentity) -> str:
    """Return the definitive scheduler-rejection receipt path."""

    return posixpath.join(
        plan.paths.receipts_root,
        "execution",
        "submission-rejections",
        identity.stage_id,
        f"attempt-{identity.attempt:03d}.json",
    )


def reconciler_followup_path(plan: ExecutionPlan, stage_id: str, attempt: int) -> str:
    """Return the authorization path for a retry-bound reconciler generation."""

    return posixpath.join(
        plan.paths.receipts_root,
        "execution",
        "reconciler-followups",
        stage_id,
        f"attempt-{attempt:03d}.json",
    )


def task_attempt_path(plan: ExecutionPlan, identity: SubmissionIdentity, task_id: str) -> str:
    """Return one stable task's immutable per-attempt evidence path."""

    return posixpath.join(
        plan.paths.receipts_root,
        "execution",
        "task-attempts",
        identity.stage_id,
        task_id,
        f"attempt-{identity.attempt:03d}.json",
    )


def reconciliation_path(plan: ExecutionPlan, stage_id: str, attempt: int) -> str:
    """Return one stage attempt's immutable files-as-truth decision path."""

    return posixpath.join(
        plan.paths.receipts_root,
        "execution",
        "reconciliations",
        stage_id,
        f"attempt-{attempt:03d}.json",
    )


def pending_registry_path(plan: ExecutionPlan, stage_id: str, attempt: int) -> str:
    """Return one locked, independently clearable registry outbox path."""

    return posixpath.join(
        plan.paths.receipts_root,
        "execution",
        "pending-registry",
        stage_id,
        f"attempt-{attempt:03d}.json",
    )


__all__ = [
    "bound_plan_path",
    "pending_registry_path",
    "reconciler_followup_path",
    "reconciliation_path",
    "submission_intent_path",
    "submission_invocation_path",
    "submission_record_path",
    "submission_rejection_path",
    "task_attempt_path",
]
