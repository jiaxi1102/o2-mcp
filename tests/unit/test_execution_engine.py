"""Offline fault-injection tests for idempotent ExecutionPlan execution."""

from __future__ import annotations

import hashlib
import json
import posixpath
from dataclasses import dataclass

import pytest

from o2mcp.runorg import (
    CanonicalPaths,
    CommandSpec,
    DatasetIdentity,
    ExecutionEngine,
    ExecutionPlan,
    PreparedRunIdentity,
    ReceiptObservation,
    ReceiptSpec,
    RegistryUpdate,
    ResourceSpec,
    RetryPolicy,
    SlurmJob,
    SlurmTaskState,
    StageSpec,
    SubmissionIdentity,
    SubmissionRequest,
    SubmissionUncertain,
    SubmitOutcome,
    TaskSpec,
)
from o2mcp.runorg.execution_models import (
    ACCEPTED,
    DEFINITELY_NOT_INVOKED,
    INVOKED_OUTCOME_UNKNOWN,
    RECONCILE_COMPLETE,
    RECONCILE_FAILED,
    RECONCILE_RETRY_SUBMITTED,
)
from o2mcp.runorg.execution_paths import pending_registry_path

CAMPAIGN = "execution-canary"
RUN_ID = f"RUN_20260822T010203Z_{CAMPAIGN}__fault-test"
RUN_ROOT = f"/n/scratch/users/test/runs/{CAMPAIGN}/{RUN_ID}"


class FakeExecutionBackend:
    """In-memory Slurm/filesystem boundary with injectable submit failures."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.jobs: dict[str, dict[str, object]] = {}
        self.requests: list[SubmissionRequest] = []
        self.next_job_id = 8000
        self.lose_next_submit_response = False
        self.hide_next_submitted_job = False
        self.intent_present_at_submit: list[bool] = []
        self.lifecycle_claims: set[str] = set()
        self.transition_marked = False

    def find_jobs(self, comment: str):
        return tuple(
            SlurmJob(job_id, comment, str(data.get("state", "PENDING")))
            for job_id, data in self.jobs.items()
            if data["comment"] == comment and bool(data.get("visible", True))
        )

    def prepare_submission(self, request: SubmissionRequest) -> None:
        """The in-memory dispatcher needs no separate filesystem preparation."""

        return None

    def invoke_submission(self, request: SubmissionRequest) -> SubmitOutcome:
        intent_suffix = f"/submission-intents/{request.identity.stage_id}/attempt-{request.identity.attempt:03d}.json"
        self.intent_present_at_submit.append(any(path.endswith(intent_suffix) for path in self.files))
        self.requests.append(request)
        job_id = str(self.next_job_id)
        self.next_job_id += 1
        self.jobs[job_id] = {
            "comment": request.comment,
            "state": "PENDING",
            "task_states": (SlurmTaskState(None, "PENDING", None),),
            "visible": not self.hide_next_submitted_job,
        }
        self.hide_next_submitted_job = False
        if self.lose_next_submit_response:
            self.lose_next_submit_response = False
            # Slurm accepted the job, but the transport dropped its response.
            return SubmitOutcome(INVOKED_OUTCOME_UNKNOWN)
        return SubmitOutcome(ACCEPTED, job_id=job_id, returncode=0)

    def task_states(self, job_id: str):
        return self.jobs[job_id]["task_states"]

    def set_states(self, job_id: str, *states: SlurmTaskState) -> None:
        self.jobs[job_id]["task_states"] = tuple(states)
        root = next((state for state in states if state.array_index is None), None)
        if root is not None:
            self.jobs[job_id]["state"] = root.state

    def reveal_jobs(self) -> None:
        """Simulate delayed squeue/sacct visibility after sbatch acceptance."""

        for job in self.jobs.values():
            job["visible"] = True

    def observe_receipt(self, run_root: str, receipt: ReceiptSpec) -> ReceiptObservation:
        path = posixpath.join(run_root, receipt.path)
        if path not in self.files:
            return ReceiptObservation(receipt.path, False, None)
        digest = hashlib.sha256(self.files[path].encode()).hexdigest()
        return ReceiptObservation(receipt.path, True, digest)

    def put_receipt(self, receipt: ReceiptSpec, text: str = "ok\n") -> str:
        path = posixpath.join(RUN_ROOT, receipt.path)
        self.files[path] = text
        return path

    def read_text(self, path: str) -> str | None:
        return self.files.get(path)

    def write_immutable_text(self, path: str, text: str) -> bool:
        """Publish exact bytes once and report whether this caller created them.

        The production backend uses an atomic hard-link publication.  Returning
        the creator bit is load-bearing: only the process that wins publication
        may call ``sbatch``; identical losers must remain query-only.
        """

        created = path not in self.files
        if not created and self.files[path] != text:
            raise RuntimeError(f"immutable conflict: {path}")
        self.files[path] = text
        return created

    def write_mutable_text(self, path: str, text: str) -> None:
        self.files[path] = text

    def compare_and_swap_text(self, path: str, expected: str | None, replacement: str | None) -> bool:
        """Apply the production backend's exact-byte CAS semantics in memory."""

        if self.files.get(path) != expected or (path not in self.files and expected is not None):
            return False
        if replacement is None:
            self.files.pop(path, None)
        else:
            self.files[path] = replacement
        return True

    def acquire_lifecycle_claim(self, _run_root: str, operation_id: str) -> bool:
        """Model the sibling transition/claim exclusion used in production."""

        if self.transition_marked:
            return False
        self.lifecycle_claims.add(operation_id)
        return True

    def release_lifecycle_claim(self, _run_root: str, operation_id: str) -> None:
        """Release exactly one converged operation claim."""

        self.lifecycle_claims.discard(operation_id)


@dataclass
class FakeRegistry:
    """Registry synchronizer that can fail a bounded number of writes."""

    failures_remaining: int = 0

    def __post_init__(self) -> None:
        self.updates: list[RegistryUpdate] = []

    def validate_plan(self, _plan: ExecutionPlan) -> bool:
        """Represent an already authenticated prepared run in engine unit tests."""

        return True

    def synchronize(self, _plan: ExecutionPlan, update: RegistryUpdate) -> bool:
        if self.failures_remaining:
            self.failures_remaining -= 1
            return False
        self.updates.append(update)
        return True


def _receipt(task_id: str) -> ReceiptSpec:
    return ReceiptSpec(
        receipt_id=f"receipt-{task_id}",
        path=f"receipts/stages/analyze/{task_id}.json",
    )


def _command(task_id: str) -> CommandSpec:
    return CommandSpec(
        argv=("/usr/bin/python3", "-m", "canary.worker", "--task", task_id),
        working_directory=f"{RUN_ROOT}/work",
        runtime_fingerprint_sha256="e" * 64,
        environment=("LC_ALL=C",),
    )


def _resources(parallelism: int = 3) -> ResourceSpec:
    return ResourceSpec(
        partition="short",
        cpus=2,
        memory_mb=4096,
        time_limit="00:10:00",
        array_parallelism=parallelism,
    )


def _array_stage(*, retry_policy: RetryPolicy | None = None) -> StageSpec:
    tasks = tuple(
        TaskSpec(
            task_id=f"movie-{index}",
            array_index=index,
            command=_command(f"movie-{index}"),
            expected_receipts=(_receipt(f"movie-{index}"),),
        )
        for index in range(3)
    )
    return StageSpec(
        stage_id="analyze",
        resources=_resources(),
        expected_receipts=(),
        tasks=tasks,
        retry_policy=retry_policy
        or RetryPolicy(
            max_attempts=2,
            retryable_slurm_states=("NODE_FAIL",),
            retry_missing_receipts=True,
            backoff_seconds=7,
        ),
    )


def _plan(*, stages: tuple[StageSpec, ...] | None = None) -> ExecutionPlan:
    return ExecutionPlan(
        project="execution-tests",
        campaign=CAMPAIGN,
        pipeline="canary",
        run_id=RUN_ID,
        source_commit="a" * 40,
        source_bundle_sha256="b" * 64,
        datasets=(DatasetIdentity("dataset-1", "c" * 64),),
        paths=CanonicalPaths(
            run_root=RUN_ROOT,
            work_root=f"{RUN_ROOT}/work",
            results_root="/n/groups/lab/results/execution-tests/dataset-1",
            receipts_root=f"{RUN_ROOT}/receipts",
            logs_root=f"{RUN_ROOT}/logs",
        ),
        stages=stages or (_array_stage(),),
    )


def test_prepared_identity_must_precede_and_match_plan_sealing() -> None:
    """The allocated run root and dataset scope are fixed before plan hashing."""

    prepared = PreparedRunIdentity(
        project="execution-tests",
        campaign=CAMPAIGN,
        pipeline="canary",
        run_id=RUN_ID,
        run_root=RUN_ROOT,
        created_utc="20260822T010203Z",
        dataset_ids=("dataset-1",),
    )
    template = _plan()
    sealed = prepared.seal_plan(
        source_commit=template.source_commit,
        source_bundle_sha256=template.source_bundle_sha256,
        datasets=template.datasets,
        paths=template.paths,
        stages=template.stages,
    )
    assert sealed == template

    changed_paths = CanonicalPaths(
        run_root=RUN_ROOT.replace("fault-test", "other-run"),
        work_root=f"{RUN_ROOT.replace('fault-test', 'other-run')}/work",
        results_root=template.paths.results_root,
        receipts_root=f"{RUN_ROOT.replace('fault-test', 'other-run')}/receipts",
        logs_root=f"{RUN_ROOT.replace('fault-test', 'other-run')}/logs",
    )
    with pytest.raises(ValueError, match="prepared run"):
        prepared.seal_plan(
            source_commit=template.source_commit,
            source_bundle_sha256=template.source_bundle_sha256,
            datasets=template.datasets,
            paths=changed_paths,
            stages=template.stages,
        )


def test_lost_submit_response_recovers_exact_job_without_duplicate() -> None:
    """An accepted job with a lost response is found by comment, never resubmitted."""

    backend = FakeExecutionBackend()
    backend.lose_next_submit_response = True
    engine = ExecutionEngine(backend)
    plan = _plan()

    first = engine.submit_stage(plan, "analyze")
    second = engine.submit_stage(plan, "analyze")

    assert len(backend.jobs) == 1
    assert len(backend.requests) == 1
    assert backend.intent_present_at_submit == [True]
    assert first.record.recovered is True
    assert first.record.job_id == second.record.job_id
    assert second.submitted is False
    assert SubmissionIdentity.from_comment(first.record.identity.comment) == first.record.identity


def test_lost_response_with_accounting_delay_keeps_replays_query_only() -> None:
    """An immutable intent blocks resubmit until the accepted job becomes visible."""

    backend = FakeExecutionBackend()
    backend.lose_next_submit_response = True
    backend.hide_next_submitted_job = True
    engine = ExecutionEngine(backend)
    plan = _plan()

    with pytest.raises(RuntimeError, match="no matching job is visible"):
        engine.submit_stage(plan, "analyze")
    assert len(backend.jobs) == 1 and len(backend.requests) == 1

    # The second call sees the pre-submit intent and performs only the scheduler
    # query.  It must not call sbatch while accounting visibility is delayed.
    with pytest.raises(RuntimeError, match="query again"):
        engine.submit_stage(plan, "analyze")
    assert len(backend.jobs) == 1 and len(backend.requests) == 1

    backend.reveal_jobs()
    recovered = engine.submit_stage(plan, "analyze")
    assert recovered.record.recovered is True
    assert recovered.record.job_id == "8000"
    assert len(backend.jobs) == 1 and len(backend.requests) == 1


def test_zero_based_array_and_comment_are_rendered_exactly() -> None:
    """Array index zero survives selection, dispatcher rendering, and sbatch argv."""

    backend = FakeExecutionBackend()
    result = ExecutionEngine(backend).submit_stage(_plan(), "analyze")
    request = backend.requests[0]

    assert request.task_indices == (0, 1, 2)
    assert "--array=0,1,2%3" in request.sbatch_args()
    assert f"--comment={result.record.identity.comment}" in request.sbatch_args()
    assert 'case "${SLURM_ARRAY_TASK_ID:?missing array task id}" in' in request.script_text
    assert "  0)" in request.script_text


def test_node_failure_retries_only_missing_and_preserves_successes() -> None:
    """Two successful tasks remain immutable while one lost node task is retried."""

    backend = FakeExecutionBackend()
    engine = ExecutionEngine(backend)
    plan = _plan()
    first = engine.submit_stage(plan, "analyze").record
    backend.set_states(
        first.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    stable_paths = [backend.put_receipt(_receipt("movie-0")), backend.put_receipt(_receipt("movie-2"))]
    stable_bytes = {path: backend.files[path] for path in stable_paths}

    retry = engine.reconcile_stage(plan, "analyze")

    assert retry.decision == RECONCILE_RETRY_SUBMITTED
    assert retry.successful_task_ids == ("movie-0", "movie-2")
    assert retry.retry_task_ids == ("movie-1",)
    assert retry.retry_submission is not None
    assert retry.retry_submission.task_ids == ("movie-1",)
    assert retry.retry_submission.task_indices == (1,)
    assert "--array=1%1" in backend.requests[-1].sbatch_args()
    assert "--begin=now+7seconds" in backend.requests[-1].sbatch_args()
    assert {path: backend.files[path] for path in stable_paths} == stable_bytes

    second_job = retry.retry_submission.job_id
    backend.set_states(
        second_job,
        SlurmTaskState(None, "COMPLETED", 0),
        SlurmTaskState(1, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-1"))
    completed = engine.reconcile_stage(plan, "analyze")

    assert completed.decision == RECONCILE_COMPLETE
    assert completed.successful_task_ids == ("movie-0", "movie-1", "movie-2")
    assert len(backend.jobs) == 2
    assert {path: backend.files[path] for path in stable_paths} == stable_bytes


def test_task_attempt_receipts_are_immutable_and_idempotent() -> None:
    """Reconciliation replays identical evidence but rejects changed terminal facts."""

    backend = FakeExecutionBackend()
    engine = ExecutionEngine(backend)
    plan = _plan()
    record = engine.submit_stage(plan, "analyze").record
    backend.set_states(
        record.job_id,
        SlurmTaskState(None, "COMPLETED", 0),
        *(SlurmTaskState(index, "COMPLETED", 0) for index in range(3)),
    )
    for index in range(3):
        backend.put_receipt(_receipt(f"movie-{index}"))

    assert engine.reconcile_stage(plan, "analyze").decision == RECONCILE_COMPLETE
    snapshot = dict(backend.files)
    assert engine.reconcile_stage(plan, "analyze").decision == RECONCILE_COMPLETE
    assert backend.files == snapshot

    backend.set_states(
        record.job_id,
        SlurmTaskState(None, "FAILED", 1),
        SlurmTaskState(0, "FAILED", 1),
        SlurmTaskState(1, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    # Once terminal evidence is sealed, later scheduler-history drift cannot
    # rewrite it.  The files-as-truth receipt remains authoritative.
    assert engine.reconcile_stage(plan, "analyze").decision == RECONCILE_COMPLETE
    assert backend.files == snapshot


def test_afterany_dependency_is_scheduler_visible() -> None:
    """A reconciler-like stage waits for terminal prerequisites, not success only."""

    preflight_receipt = ReceiptSpec("preflight-done", "receipts/stages/preflight/done.json")
    preflight = StageSpec(
        stage_id="preflight",
        command=_command("preflight"),
        resources=_resources(1),
        expected_receipts=(preflight_receipt,),
    )
    reconcile_receipt = ReceiptSpec("reconcile-done", "receipts/stages/reconcile/done.json")
    reconciler = StageSpec(
        stage_id="reconcile",
        command=_command("reconcile"),
        resources=_resources(1),
        expected_receipts=(reconcile_receipt,),
        depends_on=("preflight",),
        dependency_mode="afterany",
    )
    plan = _plan(stages=(preflight, reconciler))
    backend = FakeExecutionBackend()
    engine = ExecutionEngine(backend)

    prerequisite = engine.submit_stage(plan, "preflight").record
    engine.submit_afterany_reconciler(plan, "reconcile")

    request = backend.requests[-1]
    assert request.dependency_mode == "afterany"
    assert request.dependency_job_ids == (prerequisite.job_id,)
    assert f"--dependency=afterany:{prerequisite.job_id}" in request.sbatch_args()


def test_explicit_cancelled_task_is_not_missing_only_retryable() -> None:
    """User cancellation remains terminal unless the signed policy names it."""

    backend = FakeExecutionBackend()
    engine = ExecutionEngine(backend)
    plan = _plan()
    record = engine.submit_stage(plan, "analyze").record
    backend.set_states(
        record.job_id,
        SlurmTaskState(None, "CANCELLED", 0),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(1, "CANCELLED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))

    result = engine.reconcile_stage(plan, "analyze")

    assert result.decision == RECONCILE_FAILED
    assert result.failed_task_ids == ("movie-1",)
    assert len(backend.jobs) == 1
    with pytest.raises(ValueError, match="authorization"):
        engine.submit_stage(plan, "analyze", attempt=2, task_ids=("movie-1",))


def test_registry_write_failure_reconciles_without_resubmission() -> None:
    """Registry repair is metadata-only and cannot create another Slurm job."""

    backend = FakeExecutionBackend()
    registry = FakeRegistry(failures_remaining=1)
    engine = ExecutionEngine(backend, registry)
    plan = _plan()

    submitted = engine.submit_stage(plan, "analyze")
    assert submitted.registry_synced is False
    assert len(backend.jobs) == 1
    pending_path = pending_registry_path(plan, "analyze", 1)
    assert json.loads(backend.files[pending_path])["execution_status"] == "SUBMITTED"

    assert engine.reconcile_registry(plan) is True
    assert pending_path not in backend.files
    assert len(registry.updates) == 1
    assert len(backend.jobs) == 1


def test_retry_bound_and_nonretryable_failure_fail_closed() -> None:
    """The engine neither exceeds max_attempts nor retries unsigned failures."""

    backend = FakeExecutionBackend()
    engine = ExecutionEngine(backend)
    plan = _plan()
    first = engine.submit_stage(plan, "analyze").record
    backend.set_states(
        first.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        *(SlurmTaskState(index, "NODE_FAIL", 1) for index in range(3)),
    )
    retry = engine.reconcile_stage(plan, "analyze")
    assert retry.decision == RECONCILE_RETRY_SUBMITTED
    assert len(backend.jobs) == 2

    second = retry.retry_submission
    assert second is not None
    backend.set_states(
        second.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        *(SlurmTaskState(index, "NODE_FAIL", 1) for index in range(3)),
    )
    exhausted = engine.reconcile_stage(plan, "analyze")
    assert exhausted.decision == RECONCILE_FAILED
    assert len(backend.jobs) == 2

    no_retry = RetryPolicy(max_attempts=2, retryable_slurm_states=("NODE_FAIL",))
    second_backend = FakeExecutionBackend()
    second_engine = ExecutionEngine(second_backend)
    second_plan = _plan(stages=(_array_stage(retry_policy=no_retry),))
    record = second_engine.submit_stage(second_plan, "analyze").record
    for index in range(3):
        second_backend.put_receipt(_receipt(f"movie-{index}"))
    second_backend.set_states(
        record.job_id,
        SlurmTaskState(None, "FAILED", 2),
        *(SlurmTaskState(index, "FAILED", 2) for index in range(3)),
    )
    failed = second_engine.reconcile_stage(second_plan, "analyze")
    assert failed.decision == RECONCILE_FAILED
    assert len(second_backend.jobs) == 1


def test_duplicate_scheduler_identity_fails_before_receipt_write() -> None:
    """Two jobs with one comment are surfaced rather than arbitrarily selected."""

    backend = FakeExecutionBackend()
    plan = _plan()
    identity = SubmissionIdentity(plan.plan_sha256, "analyze", 1)
    backend.jobs["7001"] = {"comment": identity.comment, "state": "RUNNING", "task_states": ()}
    backend.jobs["7002"] = {"comment": identity.comment, "state": "RUNNING", "task_states": ()}

    with pytest.raises(RuntimeError, match="multiple Slurm jobs"):
        ExecutionEngine(backend).submit_stage(plan, "analyze")


def test_record_replay_reconstructs_registry_update_when_outbox_was_never_written() -> None:
    """A crash after record publication cannot leave run.json unaware of the job."""

    backend = FakeExecutionBackend()
    registry = FakeRegistry(failures_remaining=1)
    plan = _plan()
    engine = ExecutionEngine(backend, registry)
    first = engine.submit_stage(plan, "analyze")
    assert not first.registry_synced
    assert backend.lifecycle_claims
    backend.files.pop(pending_registry_path(plan, "analyze", 1), None)

    replay = engine.submit_stage(plan, "analyze")
    assert not replay.submitted and replay.registry_synced
    assert len(backend.requests) == 1
    assert registry.updates[-1].job_ids == (first.record.job_id,)
    assert registry.updates[-1].stage_status == "SUBMITTED"
    assert not backend.lifecycle_claims


def test_preparation_failure_is_same_attempt_recoverable_without_invocation_marker() -> None:
    """Dispatcher/log preparation failure proves sbatch was not called."""

    backend = FakeExecutionBackend()
    plan = _plan()
    original = backend.prepare_submission
    failures = 1

    def fail_once(request: SubmissionRequest):
        nonlocal failures
        if failures:
            failures -= 1
            return SubmitOutcome(DEFINITELY_NOT_INVOKED, stderr="temporary staging failure")
        return original(request)

    backend.prepare_submission = fail_once  # type: ignore[method-assign]
    engine = ExecutionEngine(backend)
    with pytest.raises(RuntimeError, match="temporary staging failure"):
        engine.submit_stage(plan, "analyze")
    assert not any("submission-invocations" in path for path in backend.files)
    assert not backend.requests
    assert engine.submit_stage(plan, "analyze").record.identity.attempt == 1


def test_verified_pre_sbatch_failure_clears_invocation_ownership_for_same_attempt() -> None:
    """A pinned-dispatcher failure proven before sbatch is not uncertain work."""

    backend = FakeExecutionBackend()
    plan = _plan()
    original = backend.invoke_submission
    failures = 1

    def fail_before_sbatch(request: SubmissionRequest):
        nonlocal failures
        if failures:
            failures -= 1
            return SubmitOutcome(DEFINITELY_NOT_INVOKED, stderr="dispatcher digest mismatch")
        return original(request)

    backend.invoke_submission = fail_before_sbatch  # type: ignore[method-assign]
    engine = ExecutionEngine(backend)
    with pytest.raises(RuntimeError, match="dispatcher digest mismatch"):
        engine.submit_stage(plan, "analyze")
    assert not any("submission-invocations" in path for path in backend.files)
    assert engine.submit_stage(plan, "analyze").record.identity.attempt == 1


def test_uncertain_success_never_authorizes_retry_attempt() -> None:
    """A zero/unparseable or lost success remains uncertain, not rejected."""

    backend = FakeExecutionBackend()
    backend.lose_next_submit_response = True
    backend.hide_next_submitted_job = True
    plan = _plan()
    engine = ExecutionEngine(backend)
    with pytest.raises(SubmissionUncertain):
        engine.submit_stage(plan, "analyze")
    assert not any("submission-rejections" in path for path in backend.files)
    with pytest.raises(ValueError, match="no preceding reconciliation or rejection"):
        engine.submit_stage(plan, "analyze", attempt=2)
    assert len(backend.requests) == 1


def test_deleted_current_receipt_invalidates_completed_dependency_gate() -> None:
    """Historical COMPLETED evidence cannot release work after output deletion."""

    upstream = StageSpec(
        stage_id="upstream",
        command=_command("upstream"),
        resources=_resources(1),
        expected_receipts=(ReceiptSpec("upstream-done", "receipts/stages/upstream/done.json"),),
    )
    downstream = StageSpec(
        stage_id="downstream",
        command=_command("downstream"),
        resources=_resources(1),
        expected_receipts=(ReceiptSpec("downstream-done", "receipts/stages/downstream/done.json"),),
        depends_on=("upstream",),
        dependency_mode="afterok",
    )
    plan = _plan(stages=(upstream, downstream))
    backend = FakeExecutionBackend()
    engine = ExecutionEngine(backend)
    record = engine.submit_stage(plan, "upstream").record
    backend.set_states(record.job_id, SlurmTaskState(None, "COMPLETED", 0))
    receipt_path = posixpath.join(RUN_ROOT, upstream.expected_receipts[0].path)
    backend.files[receipt_path] = "certified\n"
    assert engine.reconcile_stage(plan, "upstream").decision == RECONCILE_COMPLETE

    del backend.files[receipt_path]
    with pytest.raises(ValueError, match="lack authenticated COMPLETED"):
        engine.submit_stage(plan, "downstream")


def test_untrustworthy_final_receipt_recheck_waits_instead_of_sealing_failure() -> None:
    """A transport/read fault at the final gate remains transient evidence."""

    stage = StageSpec(
        stage_id="single",
        command=_command("single"),
        resources=_resources(1),
        expected_receipts=(ReceiptSpec("single-done", "receipts/stages/single/done.json"),),
    )
    plan = _plan(stages=(stage,))
    backend = FakeExecutionBackend()
    record = ExecutionEngine(backend).submit_stage(plan, "single").record
    backend.set_states(record.job_id, SlurmTaskState(None, "COMPLETED", 0))
    backend.put_receipt(stage.expected_receipts[0])
    original = backend.observe_receipt
    observations = 0

    def fail_second_observation(run_root: str, receipt: ReceiptSpec) -> ReceiptObservation:
        nonlocal observations
        observations += 1
        if observations == 2:
            return ReceiptObservation(receipt.path, False, None, trustworthy=False, error="temporary read failure")
        return original(run_root, receipt)

    backend.observe_receipt = fail_second_observation  # type: ignore[method-assign]
    result = ExecutionEngine(backend).reconcile_stage(plan, "single")
    assert result.decision == "WAIT"
    assert result.active_task_ids == ("__current_task_receipts__",)
