"""Strict readers for immutable execution-engine evidence.

Operational JSON under ``receipts/execution`` controls retries and downstream
launches.  Reading it through one module prevents permissive ``dict.get`` logic
from accidentally treating malformed, foreign, or incomplete files as proof.
"""

from __future__ import annotations

import hashlib
import re

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
    submission_intent_path,
    submission_invocation_path,
    submission_record_path,
    submission_rejection_path,
    task_attempt_path,
)
from o2mcp.runorg.execution_reconcile import is_retryable, signed_attempt_bound
from o2mcp.runorg.execution_rendering import select_tasks
from o2mcp.runorg.lifecycle_coordination import claim_name
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
    """Authenticate a dependency follow-up and its retry trigger relation.

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


def next_unrejected_attempt(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
    after_attempt: int,
    expected_task_ids: tuple[str, ...],
) -> tuple[int, tuple[SubmissionIdentity, ...]]:
    """Advance across a contiguous chain of authenticated scheduler rejections.

    Definitive rejection consumes a signed attempt without creating a submission
    record. Reconciliation must therefore inspect those immutable receipts when
    selecting the next bounded attempt, while refusing gaps, foreign identities,
    or a task set that differs from the preceding retry authorization.
    """

    rejected: list[SubmissionIdentity] = []
    attempt = after_attempt + 1
    if not expected_task_ids:
        return attempt, ()
    while attempt <= signed_attempt_bound(plan, stage):
        identity = SubmissionIdentity(plan.plan_sha256, stage.stage_id, attempt)
        rejection = read_submission_rejection(
            backend,
            submission_rejection_path(plan, identity),
        )
        if rejection is None:
            break
        authorized, _followup_dependencies = _committed_attempt_contract(backend, plan, stage, attempt)
        if rejection.identity != identity or rejection.task_ids != expected_task_ids or authorized != expected_task_ids:
            raise ValueError("submission rejection differs from its signed retry authorization")
        rejected.append(identity)
        attempt += 1
    return attempt, tuple(rejected)


def read_submission_invocation_claim_id(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    expected_identity: SubmissionIdentity,
) -> str | None:
    """Authenticate an invocation marker and return its exact holder claim.

    The marker is the durable bridge between an uncertain ``sbatch`` call and a
    later scheduler recovery. Carrying its holder identity into the registry
    outbox lets that recovery retire the original caller's claim without ever
    touching a concurrent caller's distinct ownership.
    """

    path = submission_invocation_path(plan, expected_identity)
    text = backend.read_text(path)
    if text is None:
        return None
    value = strict_json_object(text, "submission invocation")
    expected = {
        "attempt",
        "comment",
        "intent_sha256",
        "lifecycle_claim_id",
        "plan_sha256",
        "schema_version",
        "stage_id",
    }
    if set(value) != expected:
        raise ValueError("submission invocation has unsupported fields")
    claim_id = value["lifecycle_claim_id"]
    intent_text = backend.read_text(submission_intent_path(plan, expected_identity))
    expected_operation = (
        f"submit:{expected_identity.plan_sha256}:{expected_identity.stage_id}:{expected_identity.attempt}"
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["attempt"]) is not int
        or value["attempt"] != expected_identity.attempt
        or value["comment"] != expected_identity.comment
        or value["plan_sha256"] != expected_identity.plan_sha256
        or value["stage_id"] != expected_identity.stage_id
        or type(value["intent_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", value["intent_sha256"]) is None
        or intent_text is None
        or value["intent_sha256"] != hashlib.sha256(intent_text.encode()).hexdigest()
        or type(claim_id) is not str
    ):
        raise ValueError("submission invocation is malformed or foreign")
    # Empty claim IDs are retained solely for lightweight test backends that do
    # not implement lifecycle coordination. Production IDs must pass the same
    # filename validator used by acquire/release commands.
    if claim_id:
        claim_name(claim_id)
        if not claim_id.startswith(hashlib.sha256(expected_operation.encode()).hexdigest() + "-"):
            raise ValueError("submission invocation claim belongs to another operation")
    return claim_id


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
    """Return terminal evidence for the latest occupied stage generation.

    A dependency retry can authorize and submit a newer downstream generation
    after an older one completed. Once that authorization exists, the old
    receipt is no longer current evidence: callers must wait for the new exact
    attempt to reconcile (or reject terminal certification if it never can).
    """

    latest_followup = latest_followup_attempt(backend, plan, stage)
    for attempt in range(signed_attempt_bound(plan, stage), 0, -1):
        receipt = read_reconciliation_receipt(backend, plan, stage, attempt)
        if attempt == latest_followup or receipt is not None:
            # Returning None here is intentional: a newer authorized audit or
            # task generation without its own terminal receipt invalidates an
            # older COMPLETED decision.
            return receipt
    return None


def latest_followup_attempt(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
) -> int | None:
    """Return the newest authenticated upstream-triggered generation."""

    for attempt in range(signed_attempt_bound(plan, stage), 1, -1):
        path = reconciler_followup_path(plan, stage.stage_id, attempt)
        if backend.read_text(path) is None:
            continue
        authenticate_followup_authorization(backend, plan, stage, attempt)
        return attempt
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

    attempt_bound = signed_attempt_bound(plan, stage)
    upper = min(through_attempt or attempt_bound, attempt_bound)
    records: list[SubmissionRecord] = []
    for attempt in range(1, upper + 1):
        identity = SubmissionIdentity(plan.plan_sha256, stage.stage_id, attempt)
        path = submission_record_path(plan, identity)
        if backend.read_text(path) is None:
            continue
        task_ids, followup_dependencies = _committed_attempt_contract(backend, plan, stage, attempt)
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
        _validate_record_dependencies(backend, plan, stage, record, followup_dependencies)
        records.append(record)
    return tuple(records)


def _committed_attempt_contract(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
    attempt: int,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    """Return the task contract this attempt's immutable intent committed to.

    Two coordinators can legitimately disagree about one attempt: an ordinary
    missing-only retry may still be deriving its preserved subset while
    dependency propagation publishes a full-task follow-up for the same attempt.
    The pre-submit intent is the single atomic arbitration point, so this reader
    authenticates every authorization the plan allows and then accepts the one
    the winning intent actually committed.  Re-deriving a preference here would
    permanently invalidate the winner's own record.

    The second element is the follow-up dependency tuple when the committed
    contract is a replacement generation, and ``None`` for an ordinary attempt.
    """

    all_task_ids = tuple(task.task_id for task in select_tasks(stage, None))
    followup_dependencies = None
    if attempt > 1 and stage.depends_on and _followup_exists(backend, plan, stage, attempt):
        followup_dependencies = authenticate_followup_authorization(backend, plan, stage, attempt)
        if len(followup_dependencies) != len(stage.depends_on):
            raise ValueError("follow-up dependency cardinality differs from the signed DAG")
    ordinary_task_ids = all_task_ids if attempt == 1 else _ordinary_retry_task_ids(backend, plan, stage, attempt)
    if followup_dependencies is None and ordinary_task_ids is None:
        raise ValueError("submission record lacks its signed retry authorization")

    intent = _intent_contract(backend, plan, stage, attempt)
    if intent is None:
        # Evidence predating its own intent can only be read through the single
        # preferred authorization, exactly as before intents arbitrated.
        if followup_dependencies is not None:
            return all_task_ids, followup_dependencies
        assert ordinary_task_ids is not None
        return ordinary_task_ids, None
    intent_task_ids, intent_dependency_job_ids = intent
    if (
        followup_dependencies is not None
        and intent_task_ids == all_task_ids
        and intent_dependency_job_ids == followup_dependencies
    ):
        return all_task_ids, followup_dependencies
    if ordinary_task_ids is not None and intent_task_ids == ordinary_task_ids:
        return ordinary_task_ids, None
    raise ValueError("submission intent matches no authorization signed for this attempt")


def followup_owns_attempt(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
    attempt: int,
) -> bool:
    """Report whether a follow-up still owns one attempt identity.

    An ordinary retry and a dependency follow-up can both target the same
    attempt; the pre-submit intent arbitrates between them.  A follow-up whose
    authorization lost that race never became a generation, so propagation must
    open the next attempt rather than replay the winner's unrelated submission.
    An attempt with no intent yet is still the follow-up's to submit.
    """

    if _intent_contract(backend, plan, stage, attempt) is None:
        return True
    return _committed_attempt_contract(backend, plan, stage, attempt)[1] is not None


def _followup_exists(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
    attempt: int,
) -> bool:
    """Report whether a dependency follow-up authorization occupies an attempt."""

    return backend.read_text(reconciler_followup_path(plan, stage.stage_id, attempt)) is not None


def _ordinary_retry_task_ids(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
    attempt: int,
) -> tuple[str, ...] | None:
    """Return the preceding retry/rejection subset, or ``None`` when unsigned."""

    previous = read_reconciliation_receipt(backend, plan, stage, attempt - 1)
    if previous is not None and previous.decision == RECONCILE_RETRY:
        return tuple(previous.retry_task_ids)
    previous_identity = SubmissionIdentity(plan.plan_sha256, stage.stage_id, attempt - 1)
    rejection = read_submission_rejection(backend, submission_rejection_path(plan, previous_identity))
    if rejection is None or rejection.identity != previous_identity:
        return None
    return tuple(rejection.task_ids)


def _intent_contract(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
    attempt: int,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Return the exact task and dependency tuple one intent committed."""

    identity = SubmissionIdentity(plan.plan_sha256, stage.stage_id, attempt)
    path = submission_intent_path(plan, identity)
    if backend.read_text(path) is None:
        return None
    value = read_strict_json(backend, path, "submission intent")
    task_ids = value.get("task_ids")
    dependency_job_ids = value.get("dependency_job_ids")
    if (
        value.get("schema_version") != 1
        or value.get("plan_sha256") != plan.plan_sha256
        or value.get("stage_id") != stage.stage_id
        or value.get("attempt") != attempt
        or type(task_ids) is not list
        or any(type(item) is not str for item in task_ids)
        or type(dependency_job_ids) is not list
        or any(type(item) is not str for item in dependency_job_ids)
    ):
        raise ValueError(f"submission intent is malformed or foreign: {path}")
    return tuple(task_ids), tuple(dependency_job_ids)


def _validate_record_dependencies(
    backend: ExecutionBackend,
    plan: ExecutionPlan,
    stage: StageSpec,
    record: SubmissionRecord,
    followup_dependencies: tuple[str, ...] | None = None,
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
    if followup_dependencies is not None and followup_dependencies != record.dependency_job_ids:
        # Only the arbitrated replacement generation is held to the follow-up
        # tuple.  A follow-up that lost the intent race for this attempt governs
        # a later generation and must not invalidate the winner's record.
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
    latest_verdicts: dict[str, tuple[bool, bool]] = {}
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
                verdict = authenticated_task_verdict(stage, task, item)
            except ValueError as exc:
                raise ValueError(f"task-attempt evidence verdict mismatch: {path}") from exc
            evidence.append(item)
            latest_verdicts[task_id] = verdict

        if not evidence:
            raise ValueError(f"reconciliation task {task_id} has no immutable attempt evidence")
        if expected_category == "SUCCESS":
            if not any(item.successful for item in evidence):
                raise ValueError(f"reconciliation success for {task_id} lacks successful evidence")
        else:
            latest = evidence[-1]
            if expected_category == "RETRY" and (latest.successful or not latest.retryable):
                raise ValueError(f"reconciliation retry for {task_id} lacks retryable evidence")
            if expected_category == "FAILED" and latest.successful:
                raise ValueError(f"reconciliation failure for {task_id} lacks terminal evidence")

    if reconciliation.decision == "FAILED":
        failed_tasks = set(reconciliation.failed_task_ids) - {"__stage_receipts__"}
        retryable_failures = {task_id for task_id in failed_tasks if latest_verdicts.get(task_id) == (False, True)}
        terminal_failures = {task_id for task_id in failed_tasks if latest_verdicts.get(task_id) == (False, False)}
        # A retryable task may be classified FAILED only when another task has
        # already made the exact stage irrecoverable, or when every signed
        # attempt has been consumed.  This preserves raw retryability in the
        # TaskAttemptReceipt while authenticating the stage-level budget rule.
        if retryable_failures and not terminal_failures and reconciliation.attempt < signed_attempt_bound(plan, stage):
            raise ValueError("reconciliation failure contains retryable tasks before the signed attempt bound")


__all__ = [
    "authenticate_followup_authorization",
    "current_task_receipts_valid",
    "current_task_receipts_status",
    "authenticated_task_verdict",
    "latest_reconciliation_receipt",
    "latest_followup_attempt",
    "next_unrejected_attempt",
    "read_reconciliation_receipt",
    "read_plan_submission_records",
    "read_strict_json",
    "read_submission_record",
    "read_submission_rejection",
]
