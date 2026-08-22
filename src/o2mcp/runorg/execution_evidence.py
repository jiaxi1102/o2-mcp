"""Strict readers for immutable execution-engine evidence.

Operational JSON under ``receipts/execution`` controls retries and downstream
launches.  Reading it through one module prevents permissive ``dict.get`` logic
from accidentally treating malformed, foreign, or incomplete files as proof.
"""

from __future__ import annotations

from o2mcp.runorg.execution_backend import ExecutionBackend, receipt_matches
from o2mcp.runorg.execution_models import (
    RECONCILE_RETRY,
    SUCCESS_SLURM_STATES,
    PlannedTask,
    SubmissionIdentity,
    SubmissionRecord,
    SubmissionRejectionRecord,
    TaskAttemptReceipt,
)
from o2mcp.runorg.execution_paths import (
    reconciler_followup_path,
    reconciliation_path,
    submission_record_path,
    submission_rejection_path,
    task_attempt_path,
)
from o2mcp.runorg.execution_reconcile import is_retryable
from o2mcp.runorg.execution_rendering import select_tasks
from o2mcp.runorg.plan_stages import StageSpec
from o2mcp.runorg.plans import ExecutionPlan
from o2mcp.runorg.reconciliation_receipts import ReconciliationReceipt
from o2mcp.runorg.strict_json import strict_json_object


def authenticate_followup_authorization(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
    attempt: int,
    *,
    expected_dependency_job_ids: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Authenticate an after-any follow-up and its retry trigger relation.

    The follow-up file is not a signature by itself.  Each dependency job must
    be an authenticated submission of the corresponding signed prerequisite,
    and the trigger must be that prerequisite's retry job at the same ordered
    dependency slot.  New submission calls additionally bind the authorization
    to the exact dependency set derived immediately before intent publication.
    """

    path = reconciler_followup_path(plan, stage.stage_id, attempt)
    value = read_strict_json(backend, path, "reconciler follow-up authorization")
    allowed = {
        "attempt",
        "dependency_job_ids",
        "plan_sha256",
        "schema_version",
        "stage_id",
        "trigger_job_id",
        "trigger_stage_id",
    }
    if set(value) != allowed:
        raise ValueError("reconciler follow-up authorization has unsupported fields")
    dependencies = value["dependency_job_ids"]
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["plan_sha256"]) is not str
        or value["plan_sha256"] != plan.plan_sha256
        or type(value["stage_id"]) is not str
        or value["stage_id"] != stage.stage_id
        or type(value["attempt"]) is not int
        or value["attempt"] != attempt
        or type(dependencies) is not list
        or any(type(item) is not str or not item.isdigit() for item in dependencies)
        or len(dependencies) != len(stage.depends_on)
        or len(set(dependencies)) != len(dependencies)
    ):
        raise ValueError("reconciler follow-up authorization is malformed or foreign")
    dependency_ids = tuple(dependencies)
    if expected_dependency_job_ids is not None and dependency_ids != expected_dependency_job_ids:
        raise ValueError("follow-up dependencies differ from the currently authorized signed DAG jobs")

    stages = {candidate.stage_id: candidate for candidate in plan.stages}
    trigger_stage = value["trigger_stage_id"]
    trigger_job = value["trigger_job_id"]
    if type(trigger_stage) is not str or trigger_stage not in stage.depends_on:
        raise ValueError("follow-up trigger stage is not a signed dependency")
    if type(trigger_job) is not str or not trigger_job.isdigit():
        raise ValueError("follow-up trigger job is invalid")
    for dependency, job_id in zip(stage.depends_on, dependency_ids):
        records = read_plan_submission_records(backend, plan, stages[dependency])
        matches = [record for record in records if record.job_id == job_id]
        if len(matches) != 1:
            raise ValueError(f"follow-up dependency job {job_id} is not authenticated for stage {dependency}")
        if dependency == trigger_stage and (job_id != trigger_job or matches[0].identity.attempt <= 1):
            raise ValueError("follow-up trigger must be the authenticated retry job for its dependency")
    return dependency_ids


def read_submission_record(
    backend: ExecutionBackend,
    path: str,
    *,
    expected_identity=None,
    expected_task_ids: tuple[str, ...] | None = None,
    expected_task_indices: tuple[int, ...] | None = None,
    expected_dependency_mode: str | None = None,
    expected_dependency_job_ids: tuple[str, ...] | None = None,
) -> SubmissionRecord | None:
    """Read and, when supplied, authenticate one submission-path contract.

    Parsing a well-formed record is not enough because an attacker or interrupted
    copy can place a valid record at the wrong attempt path.  Callers that use a
    record as scheduler or reconciliation evidence therefore pass every value
    derived independently from the signed plan and its prior authorizations.
    """

    text = backend.read_text(path)
    if text is None:
        return None
    try:
        value = strict_json_object(text, "submission record")
        record = SubmissionRecord.from_dict(value)
        checks = (
            (expected_identity, record.identity, "identity"),
            (expected_task_ids, record.task_ids, "task IDs"),
            (expected_task_indices, record.task_indices, "task indices"),
            (expected_dependency_mode, record.dependency_mode, "dependency mode"),
            (expected_dependency_job_ids, record.dependency_job_ids, "dependency job IDs"),
        )
        for expected, observed, label in checks:
            if expected is not None and observed != expected:
                raise ValueError(f"submission record {label} do not match its authorized path contract")
        return record
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid immutable submission record: {path}") from exc


def read_submission_rejection(backend: ExecutionBackend, path: str) -> SubmissionRejectionRecord | None:
    """Read one definitive scheduler-rejection record with strict validation."""

    text = backend.read_text(path)
    if text is None:
        return None
    try:
        value = strict_json_object(text, "submission rejection")
        return SubmissionRejectionRecord.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid immutable submission rejection: {path}") from exc


def read_strict_json(backend: ExecutionBackend, path: str, label: str) -> dict[str, object]:
    """Read a control-plane JSON object without accepting absence or coercion."""

    text = backend.read_text(path)
    if not text:
        raise ValueError(f"{label} is missing: {path}")
    try:
        return strict_json_object(text, label)
    except ValueError as exc:
        raise ValueError(f"{label} is malformed: {path}") from exc


def read_reconciliation_receipt(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
    attempt: int,
) -> ReconciliationReceipt | None:
    """Decode a reconciliation receipt and verify its task-attempt evidence.

    Structural identity alone is insufficient: a truncated or fabricated
    ``COMPLETED`` object must not release an ``afterok`` stage.  Every classified
    task is therefore tied back to the immutable scheduler and receipt verdict
    written for one accepted submission record.
    """

    path = reconciliation_path(plan, stage.stage_id, attempt)
    text = backend.read_text(path)
    if text is None:
        return None
    try:
        value = strict_json_object(text, "reconciliation receipt")
        receipt = ReconciliationReceipt.from_dict(
            value,
            plan_sha256=plan.plan_sha256,
            stage=stage,
            attempt=attempt,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid immutable reconciliation receipt: {path}") from exc
    _validate_task_evidence(backend, plan, stage, receipt)
    return receipt


def latest_reconciliation_receipt(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
) -> ReconciliationReceipt | None:
    """Return the latest valid bounded reconciliation receipt for ``stage``."""

    for attempt in range(stage.retry_policy.max_attempts, 0, -1):
        receipt = read_reconciliation_receipt(backend, plan, stage, attempt)
        if receipt is not None:
            return receipt
    return None


def read_plan_submission_records(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
    *,
    through_attempt: int | None = None,
) -> tuple[SubmissionRecord, ...]:
    """Return only records authenticated against their signed attempt contract.

    The record pathname is not evidence of its contents.  For each existing
    bounded attempt, this reader independently derives task IDs, stable array
    indices, dependency mode, and dependency job IDs from the plan plus the
    immediately preceding reconciliation/rejection authorization.
    """

    upper = min(through_attempt or stage.retry_policy.max_attempts, stage.retry_policy.max_attempts)
    records: list[SubmissionRecord] = []
    for attempt in range(1, upper + 1):
        identity = SubmissionIdentity(plan.plan_sha256, stage.stage_id, attempt)
        path = submission_record_path(plan, identity)
        if backend.read_text(path) is None:
            continue
        task_ids = _record_task_authorization(backend, plan, stage, attempt)
        tasks = select_tasks(stage, task_ids)
        task_indices = tuple(task.array_index for task in tasks if task.array_index is not None)
        record = read_submission_record(
            backend,
            path,
            expected_identity=identity,
            expected_task_ids=tuple(task.task_id for task in tasks),
            expected_task_indices=task_indices,
            expected_dependency_mode=stage.dependency_mode,
        )
        assert record is not None
        _validate_record_dependencies(backend, plan, stage, record)
        records.append(record)
    return tuple(records)


def _record_task_authorization(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
    attempt: int,
) -> tuple[str, ...]:
    """Derive the task contract for one existing record."""

    all_task_ids = tuple(task.task_id for task in select_tasks(stage, None))
    if attempt > 1 and stage.dependency_mode == "afterany" and stage.depends_on and not stage.tasks:
        dependency_values = authenticate_followup_authorization(backend, plan, stage, attempt)
        task_ids = all_task_ids
        if len(dependency_values) != len(stage.depends_on):
            raise ValueError("follow-up dependency cardinality differs from the signed DAG")
    else:
        if attempt == 1:
            task_ids = all_task_ids
        else:
            previous = read_reconciliation_receipt(backend, plan, stage, attempt - 1)
            if previous is not None and previous.decision == RECONCILE_RETRY:
                task_ids = previous.retry_task_ids
            else:
                previous_identity = SubmissionIdentity(plan.plan_sha256, stage.stage_id, attempt - 1)
                rejection = read_submission_rejection(
                    backend,
                    submission_rejection_path(plan, previous_identity),
                )
                if rejection is None or rejection.identity != previous_identity:
                    raise ValueError("submission record lacks its signed retry authorization")
                task_ids = rejection.task_ids
    return tuple(task_ids)


def _validate_record_dependencies(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
    record: SubmissionRecord,
) -> None:
    """Prove each recorded dependency is a job of its signed prerequisite.

    A later retry can make a different prerequisite job "latest" after this
    record was submitted.  Historical evidence therefore validates membership
    in the declared stage's authenticated jobs rather than incorrectly rewriting
    history to whichever attempt happens to be latest now.
    """

    if len(record.dependency_job_ids) != len(stage.depends_on):
        raise ValueError("submission record dependency cardinality differs from the signed DAG")
    stages = {candidate.stage_id: candidate for candidate in plan.stages}
    for dependency, job_id in zip(stage.depends_on, record.dependency_job_ids):
        dependency_records = read_plan_submission_records(backend, plan, stages[dependency])
        authorized_jobs = {item.job_id for item in dependency_records}
        if job_id not in authorized_jobs:
            raise ValueError(f"submission dependency job {job_id} is not authenticated evidence for stage {dependency}")
    if record.identity.attempt > 1 and stage.dependency_mode == "afterany" and stage.depends_on and not stage.tasks:
        dependency_values = authenticate_followup_authorization(
            backend,
            plan,
            stage,
            record.identity.attempt,
        )
        if dependency_values != record.dependency_job_ids:
            raise ValueError("submission record differs from its exact follow-up dependency authorization")
    if stage.depends_on and not record.dependency_job_ids:
        # Keep this explicit even though cardinality catches it: this is the
        # dangerous dependency-bypass shape the contract exists to prevent.
        raise ValueError("dependent submission record cannot omit authorized dependencies")


def current_task_receipts_status(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
) -> bool | None:
    """Return current task validity, or ``None`` for an untrustworthy read.

    Immutable attempt evidence proves what reconciliation observed, but pipeline
    receipts can still be deleted or replaced afterward.  ``afterok`` gating
    therefore checks both historical verdicts and current files.  A transport
    failure is distinct from proven missing bytes so reconciliation can wait
    rather than sealing a false terminal failure.
    """

    for task in select_tasks(stage, None):
        observations = tuple(
            backend.observe_receipt(plan.paths.run_root, receipt) for receipt in task.expected_receipts
        )
        if any(not observation.trustworthy for observation in observations):
            return None
        if not all(
            receipt_matches(spec, observation) for spec, observation in zip(task.expected_receipts, observations)
        ):
            return False
    return True


def current_task_receipts_valid(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
) -> bool:
    """Fail closed unless current task receipts are positively valid."""

    return current_task_receipts_status(backend, plan, stage) is True


def authenticated_task_verdict(
    stage: StageSpec,
    task: PlannedTask,
    receipt: TaskAttemptReceipt,
) -> tuple[bool, bool]:
    """Recompute success and retryability from signed policy and raw evidence.

    The serialized booleans are redundant audit conveniences, not authority.
    Recomputing them prevents a contradictory receipt such as ``FAILED`` with a
    nonzero exit code but ``successful=true`` from releasing an after-ok gate.
    """

    expected_paths = tuple(spec.path for spec in task.expected_receipts)
    observed_paths = tuple(item.path for item in receipt.receipt_observations)
    if observed_paths != expected_paths:
        raise ValueError("task-attempt receipt observations differ from the signed receipt scope")
    receipts_valid = all(
        receipt_matches(spec, observation)
        for spec, observation in zip(task.expected_receipts, receipt.receipt_observations)
    )
    normalized = receipt.slurm_state.strip().upper().split("+", 1)[0]
    successful = normalized in SUCCESS_SLURM_STATES and receipt.exit_code == 0 and receipts_valid
    retryable = not successful and is_retryable(stage, normalized, receipt.exit_code, receipts_valid)
    if receipt.successful != successful or receipt.retryable != retryable:
        raise ValueError("task-attempt receipt verdict contradicts scheduler, receipts, or signed retry policy")
    return successful, retryable


def _validate_task_evidence(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
    reconciliation: ReconciliationReceipt,
) -> None:
    """Require each task category to agree with its latest durable verdict."""

    tasks = {task.task_id: task for task in select_tasks(stage, None)}
    records = list(read_plan_submission_records(backend, plan, stage, through_attempt=reconciliation.attempt))

    categories = {
        **{task_id: "SUCCESS" for task_id in reconciliation.successful_task_ids},
        **{task_id: "RETRY" for task_id in reconciliation.retry_task_ids},
        **{task_id: "FAILED" for task_id in reconciliation.failed_task_ids if task_id != "__stage_receipts__"},
    }
    for task_id, expected_category in categories.items():
        task = tasks[task_id]
        evidence: list[TaskAttemptReceipt] = []
        for record in records:
            if task_id not in record.task_ids:
                continue
            path = task_attempt_path(plan, record.identity, task_id)
            text = backend.read_text(path)
            if text is None:
                continue
            try:
                value = strict_json_object(text, "task-attempt evidence")
                item = TaskAttemptReceipt.from_dict(value)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid task-attempt evidence: {path}") from exc
            if (
                item.identity != record.identity
                or item.task_id != task_id
                or item.array_index != task.array_index
                or item.job_id != record.job_id
            ):
                raise ValueError(f"task-attempt evidence identity mismatch: {path}")
            try:
                authenticated_task_verdict(stage, task, item)
            except ValueError as exc:
                raise ValueError(f"task-attempt evidence verdict mismatch: {path}") from exc
            evidence.append(item)

        if not evidence:
            raise ValueError(f"reconciliation task {task_id} has no immutable attempt evidence")
        if expected_category == "SUCCESS":
            if not any(item.successful for item in evidence):
                raise ValueError(f"reconciliation success for {task_id} lacks successful evidence")
        else:
            latest = evidence[-1]
            if expected_category == "RETRY" and (latest.successful or not latest.retryable):
                raise ValueError(f"reconciliation retry for {task_id} lacks retryable evidence")
            if expected_category == "FAILED" and (latest.successful or latest.retryable):
                raise ValueError(f"reconciliation failure for {task_id} lacks terminal evidence")


__all__ = [
    "authenticate_followup_authorization",
    "current_task_receipts_valid",
    "current_task_receipts_status",
    "authenticated_task_verdict",
    "latest_reconciliation_receipt",
    "read_reconciliation_receipt",
    "read_plan_submission_records",
    "read_strict_json",
    "read_submission_record",
    "read_submission_rejection",
]
