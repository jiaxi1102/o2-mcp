"""Adversarial regressions for execution ownership and state reconciliation.

These tests deliberately force interleavings and out-of-order observations that
ordinary happy-path fakes cannot produce.  The assertions encode the safety
properties needed when multiple agents, delayed Slurm accounting, and registry
repair all act on the same immutable execution plan.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import shutil
import subprocess
import threading
from collections import deque

import pytest

from o2mcp.runorg import (
    CanonicalPaths,
    CommandSpec,
    DatasetIdentity,
    ExecutionEngine,
    ExecutionPlan,
    ReceiptObservation,
    ReceiptSpec,
    ResourceSpec,
    RetryPolicy,
    RunManifest,
    SlurmJob,
    SlurmTaskState,
    StageSpec,
    SubmissionRejected,
    SubmissionRequest,
    SubmissionUncertain,
    SubmitOutcome,
    TaskSpec,
)
from o2mcp.runorg.execution_evidence import latest_reconciliation_receipt
from o2mcp.runorg.execution_models import (
    ACCEPTED,
    DEFINITELY_REJECTED,
    INVOKED_OUTCOME_UNKNOWN,
    PlannedTask,
    RegistryUpdate,
    canonical_json,
)
from o2mcp.runorg.execution_paths import (
    pending_registry_path,
    reconciler_followup_path,
    reconciliation_path,
    task_attempt_path,
)
from o2mcp.runorg.execution_reconcile import signed_attempt_bound
from o2mcp.runorg.execution_rendering import render_dispatcher
from o2mcp.runorg.lifecycle_coordination import new_claim_id
from o2mcp.runorg.registry_sync import merge_execution_manifest
from o2mcp.runorg.transition_guards import require_current_terminal_evidence

CAMPAIGN = "adversarial-execution"
RUN_ID = f"RUN_20260822T010203Z_{CAMPAIGN}__race"
RUN_ROOT = f"/n/scratch/users/test/runs/{CAMPAIGN}/{RUN_ID}"


class ConcurrentBackend:
    """Thread-safe backend whose intent barrier reproduces two-caller races.

    The barrier forces prior reads to finish before atomic intent publication.
    """

    def __init__(self, *, intent_barrier: threading.Barrier | None = None) -> None:
        self.files: dict[str, str] = {}
        self.jobs: dict[str, dict[str, object]] = {}
        self.requests: list[SubmissionRequest] = []
        self.outcomes: deque[SubmitOutcome] = deque()
        self.next_job_id = 9000
        self.intent_barrier = intent_barrier
        self.lifecycle_claims: set[str] = set()
        self._lock = threading.RLock()

    def find_jobs(self, comment: str):
        """Return an atomic snapshot of jobs with one exact comment."""
        with self._lock:
            return tuple(
                SlurmJob(job_id, comment, str(data["state"]))
                for job_id, data in self.jobs.items()
                if data["comment"] == comment
            )

    def prepare_submission(self, request: SubmissionRequest) -> None:
        """The in-memory backend always has prepared dispatcher inputs."""
        return None

    def invoke_submission(self, request: SubmissionRequest) -> SubmitOutcome:
        """Record a scheduler call and materialize a job only when accepted."""
        with self._lock:
            self.requests.append(request)
            outcome = self.outcomes.popleft() if self.outcomes else None
            if outcome is not None and outcome.status == DEFINITELY_REJECTED:
                return outcome
            job_id = outcome.job_id if outcome is not None and outcome.status == ACCEPTED else str(self.next_job_id)
            self.next_job_id = max(self.next_job_id + 1, int(job_id) + 1)
            self.jobs[job_id] = {
                "comment": request.comment,
                "state": "PENDING",
                "task_states": (SlurmTaskState(None, "PENDING", None),),
            }
            if outcome is not None and outcome.status == INVOKED_OUTCOME_UNKNOWN:
                return outcome
            return SubmitOutcome(ACCEPTED, job_id=job_id, returncode=0)

    def task_states(self, job_id: str):
        """Return a stable accounting snapshot for one fake job."""
        with self._lock:
            return tuple(self.jobs[job_id]["task_states"])

    def set_states(self, job_id: str, *states: SlurmTaskState) -> None:
        """Replace fake accounting state while preserving the job identity."""
        with self._lock:
            self.jobs[job_id]["task_states"] = tuple(states)
            root = next((state for state in states if state.array_index is None), None)
            if root is not None:
                self.jobs[job_id]["state"] = root.state

    def observe_receipt(self, run_root: str, receipt: ReceiptSpec) -> ReceiptObservation:
        """Observe exact in-memory pipeline bytes under ``run_root``."""
        path = posixpath.join(run_root, receipt.path)
        with self._lock:
            text = self.files.get(path)
        digest = hashlib.sha256(text.encode()).hexdigest() if text is not None else None
        return ReceiptObservation(receipt.path, text is not None, digest)

    def put_receipt(self, receipt: ReceiptSpec, text: str = "ok\n") -> None:
        """Create a pipeline-owned receipt for a reconciliation test."""
        with self._lock:
            self.files[posixpath.join(RUN_ROOT, receipt.path)] = text

    def read_text(self, path: str) -> str | None:
        """Read one file atomically from the fake filesystem."""
        with self._lock:
            return self.files.get(path)

    def write_immutable_text(self, path: str, text: str) -> bool:
        """Atomically publish complete bytes and return exclusive ownership."""
        if self.intent_barrier is not None and "/submission-intents/" in path:
            self.intent_barrier.wait(timeout=5)
        with self._lock:
            if path in self.files:
                if self.files[path] != text:
                    raise RuntimeError(f"immutable conflict: {path}")
                return False
            self.files[path] = text
            return True

    def write_mutable_text(self, path: str, text: str) -> None:
        """Replace non-evidence current state under the same fake lock."""
        with self._lock:
            self.files[path] = text

    def compare_and_swap_text(self, path: str, expected: str | None, replacement: str | None) -> bool:
        """Atomically update one fake outbox item only from exact current bytes."""
        with self._lock:
            current = self.files.get(path)
            if current != expected or (path not in self.files and expected is not None):
                return False
            if replacement is None:
                self.files.pop(path, None)
            else:
                self.files[path] = replacement
            return True

    def acquire_lifecycle_claim(self, _run_root: str, operation_id: str) -> str | None:
        """Return a distinct fake holder for every concurrent caller."""

        claim_id = new_claim_id(operation_id)
        self.lifecycle_claims.add(claim_id)
        return claim_id

    def release_lifecycle_claim(self, _run_root: str, claim_id: str) -> None:
        """Retire only the exact fake holder supplied by the coordinator."""

        self.lifecycle_claims.discard(claim_id)

    def matching_lifecycle_claims(self, _run_root: str, operation_id: str) -> tuple[str, ...]:
        """Return every in-memory holder with the operation-derived prefix."""

        prefix = hashlib.sha256(operation_id.encode()).hexdigest() + "-"
        return tuple(sorted(item for item in self.lifecycle_claims if item.startswith(prefix)))


def _receipt(task_id: str, *, stage: str = "compute") -> ReceiptSpec:
    """Return a pipeline receipt outside the engine-reserved namespace."""
    return ReceiptSpec(f"receipt-{stage}-{task_id}", f"receipts/stages/{stage}/{task_id}.json")


def _command(name: str) -> CommandSpec:
    """Build one deterministic fake scientific command."""
    return CommandSpec(
        argv=("/usr/bin/python3", "-m", "canary.worker", "--task", name),
        working_directory=f"{RUN_ROOT}/work",
        runtime_fingerprint_sha256="e" * 64,
        environment=("LC_ALL=C",),
    )


def _resources(parallelism: int = 3) -> ResourceSpec:
    """Return conservative typed resources for the fake requests."""
    return ResourceSpec(
        partition="short",
        cpus=2,
        memory_mb=4096,
        time_limit="00:10:00",
        array_parallelism=parallelism,
    )


def _compute_stage(
    *,
    max_attempts: int = 2,
    stage_id: str = "compute",
    task_prefix: str = "movie",
) -> StageSpec:
    """Build a three-movie array with stable zero-based indices."""
    tasks = tuple(
        TaskSpec(
            task_id=f"{task_prefix}-{index}",
            array_index=index,
            command=_command(f"{task_prefix}-{index}"),
            expected_receipts=(_receipt(f"{task_prefix}-{index}"),),
        )
        for index in range(3)
    )
    return StageSpec(
        stage_id=stage_id,
        resources=_resources(),
        expected_receipts=(),
        tasks=tasks,
        retry_policy=RetryPolicy(
            max_attempts=max_attempts,
            retryable_slurm_states=("NODE_FAIL",),
            retry_missing_receipts=True,
        ),
    )


def _plan(*, stages: tuple[StageSpec, ...] | None = None, source_commit: str = "a" * 40) -> ExecutionPlan:
    """Build an immutable plan sharing one deliberately fixed registered run."""
    return ExecutionPlan(
        project="execution-tests",
        campaign=CAMPAIGN,
        pipeline="canary",
        run_id=RUN_ID,
        source_commit=source_commit,
        source_bundle_sha256="b" * 64,
        datasets=(DatasetIdentity("dataset-1", "c" * 64),),
        paths=CanonicalPaths(
            run_root=RUN_ROOT,
            work_root=f"{RUN_ROOT}/work",
            results_root="/n/groups/lab/results/execution-tests/dataset-1",
            receipts_root=f"{RUN_ROOT}/receipts",
            logs_root=f"{RUN_ROOT}/logs",
        ),
        stages=stages or (_compute_stage(),),
    )


def test_two_callers_racing_identical_intent_submit_exactly_one_job() -> None:
    """Only the atomic intent creator may submit after a forced two-caller race."""
    backend = ConcurrentBackend(intent_barrier=threading.Barrier(2))
    plan = _plan()
    outcomes: list[object] = []

    def invoke() -> None:
        try:
            outcomes.append(ExecutionEngine(backend).submit_stage(plan, "compute"))
        except SubmissionUncertain as exc:
            # The losing caller may query before the winner has made the new job
            # visible.  Uncertainty is safe; a second scheduler call is not.
            outcomes.append(exc)

    callers = [threading.Thread(target=invoke) for _ in range(2)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=10)
        assert not caller.is_alive()

    assert len(backend.requests) == 1
    assert len(backend.jobs) == 1
    assert len(outcomes) == 2

    # Disable the one-shot test barrier before replay.  The immutable submission
    # record or scheduler identity must now recover the same single job.
    backend.intent_barrier = None
    replay = ExecutionEngine(backend).submit_stage(plan, "compute")
    assert replay.record.job_id == "9000"
    assert len(backend.requests) == 1
    assert not backend.lifecycle_claims


def test_terminal_replay_retires_claim_abandoned_before_invocation() -> None:
    """A pre-marker crash holder is recoverable from its operation identity."""

    backend = ConcurrentBackend()
    plan = _plan()
    operation_id = f"submit:{plan.plan_sha256}:compute:1"
    abandoned = backend.acquire_lifecycle_claim(plan.paths.run_root, operation_id)
    assert abandoned is not None

    result = ExecutionEngine(backend).submit_stage(plan, "compute")

    assert result.record.identity.attempt == 1
    assert not backend.lifecycle_claims


def test_reconciliation_never_creates_a_transition_claim() -> None:
    """Convergent polling delegates every scheduler mutation to submit_stage."""

    backend = ConcurrentBackend()
    plan = _plan()
    engine = ExecutionEngine(backend)
    record = engine.submit_stage(plan, "compute").record
    backend.set_states(
        record.job_id,
        SlurmTaskState(None, "COMPLETED", 0),
        *(SlurmTaskState(index, "COMPLETED", 0) for index in range(3)),
    )
    for index in range(3):
        backend.put_receipt(_receipt(f"movie-{index}"))

    assert engine.reconcile_stage(plan, "compute").decision == "COMPLETED"
    assert not backend.lifecycle_claims


@pytest.mark.parametrize(
    ("stage_id", "attempt"),
    (("compute", 0), ("bad/stage", 1)),
)
def test_invalid_submission_identity_never_acquires_lifecycle_claim(stage_id: str, attempt: int) -> None:
    """Malformed caller input fails before publishing transition blockers."""

    backend = ConcurrentBackend()

    with pytest.raises(ValueError):
        ExecutionEngine(backend).submit_stage(_plan(), stage_id, attempt=attempt)

    assert not backend.lifecycle_claims


def test_owner_crash_before_invocation_is_safely_recoverable() -> None:
    """An intent alone proves sbatch was never crossed and may be recovered."""

    class CrashAfterIntent(ConcurrentBackend):
        crashed = False

        def write_immutable_text(self, path: str, text: str) -> bool:
            created = super().write_immutable_text(path, text)
            if created and "/submission-intents/" in path and not self.crashed:
                self.crashed = True
                raise KeyboardInterrupt("owner died after intent")
            return created

    backend = CrashAfterIntent()
    plan = _plan()
    with pytest.raises(KeyboardInterrupt, match="after intent"):
        ExecutionEngine(backend).submit_stage(plan, "compute")
    assert not backend.requests and not backend.jobs

    recovered = ExecutionEngine(backend).submit_stage(plan, "compute")
    assert recovered.record.job_id == "9000"
    assert len(backend.requests) == 1 and len(backend.jobs) == 1


def test_owner_crash_after_invocation_never_uses_unsafe_takeover() -> None:
    """Crossing the invocation marker remains query-only until Slurm proves a job."""

    class CrashAfterInvocation(ConcurrentBackend):
        crashed = False

        def write_immutable_text(self, path: str, text: str) -> bool:
            created = super().write_immutable_text(path, text)
            if created and "/submission-invocations/" in path and not self.crashed:
                self.crashed = True
                raise KeyboardInterrupt("owner died at invocation boundary")
            return created

    backend = CrashAfterInvocation()
    plan = _plan()
    with pytest.raises(KeyboardInterrupt, match="invocation boundary"):
        ExecutionEngine(backend).submit_stage(plan, "compute")
    with pytest.raises(SubmissionUncertain, match="already owned"):
        ExecutionEngine(backend).submit_stage(plan, "compute")
    assert not backend.requests and not backend.jobs


def test_registered_identity_validation_precedes_all_remote_writes() -> None:
    """A mismatched prepared run cannot be bound or submitted first and rejected later."""

    class RejectingRegistry:
        def validate_plan(self, _plan: ExecutionPlan) -> bool:
            return False

        def synchronize(self, _plan: ExecutionPlan, _update: RegistryUpdate) -> bool:
            raise AssertionError("invalid plans must never reach synchronization")

    backend = ConcurrentBackend()
    with pytest.raises(ValueError, match="registered active run"):
        ExecutionEngine(backend, RejectingRegistry()).submit_stage(_plan(), "compute")
    assert not backend.files and not backend.requests and not backend.jobs


def test_definitive_rejection_is_stable_and_authorizes_only_next_attempt() -> None:
    """A rejected attempt is replayable evidence, not a permanently poisoned intent."""

    backend = ConcurrentBackend()
    backend.outcomes.append(SubmitOutcome(DEFINITELY_REJECTED, returncode=1, stderr="invalid qos"))
    engine = ExecutionEngine(backend)
    plan = _plan()

    with pytest.raises(SubmissionRejected, match="invalid qos"):
        engine.submit_stage(plan, "compute")
    assert len(backend.requests) == 1 and not backend.jobs

    # The same attempt never resubmits.  Its immutable rejection instead permits
    # exactly attempt two with the same task set and signed scheduler arguments.
    with pytest.raises(SubmissionRejected, match="definitively rejected"):
        engine.submit_stage(plan, "compute")
    accepted = engine.submit_stage(plan, "compute", attempt=2)
    assert accepted.record.identity.attempt == 2
    assert len(backend.requests) == 2 and len(backend.jobs) == 1

    with pytest.raises(ValueError, match="exceeds signed max_attempts"):
        engine.submit_stage(plan, "compute", attempt=3)


def test_rejection_replay_retires_original_owner_after_lost_release() -> None:
    """Durable rejection evidence repairs an invocation owner's stale claim."""

    class LoseFirstRelease(ConcurrentBackend):
        """Inject one lost release reply after rejection evidence is durable."""

        lose_release = True

        def release_lifecycle_claim(self, run_root: str, claim_id: str) -> None:
            """Leave the first exact owner in place, then behave idempotently."""

            if self.lose_release:
                self.lose_release = False
                raise RuntimeError("injected lifecycle release failure")
            super().release_lifecycle_claim(run_root, claim_id)

    backend = LoseFirstRelease()
    backend.outcomes.append(SubmitOutcome(DEFINITELY_REJECTED, returncode=1, stderr="invalid qos"))
    engine = ExecutionEngine(backend)
    plan = _plan()

    # The scheduler rejection is immutable, but the invocation owner's claim
    # remains because its first release response was lost.
    with pytest.raises(RuntimeError, match="lifecycle release failure"):
        engine.submit_stage(plan, "compute")
    assert len(backend.lifecycle_claims) == 1

    # Replay authenticates the invocation marker, retires both the stale owner
    # and its own observer, and still reports the stable scheduler rejection.
    with pytest.raises(SubmissionRejected, match="definitively rejected"):
        engine.submit_stage(plan, "compute")
    assert not backend.lifecycle_claims


def test_failed_rejection_publication_recovers_exact_invocation_for_same_attempt() -> None:
    """A pre-create rejection write failure cannot permanently poison the run."""

    class FailFirstRejectionWrite(ConcurrentBackend):
        """Raise before publishing the first definitive-rejection receipt."""

        failed = False

        def write_immutable_text(self, path: str, text: str) -> bool:
            if "/submission-rejections/compute/attempt-001.json" in path and not self.failed:
                self.failed = True
                raise OSError("injected rejection write failure")
            return super().write_immutable_text(path, text)

    backend = FailFirstRejectionWrite()
    backend.outcomes.append(SubmitOutcome(DEFINITELY_REJECTED, returncode=1, stderr="invalid qos"))
    engine = ExecutionEngine(backend)
    plan = _plan()

    with pytest.raises(RuntimeError, match="retry this same attempt"):
        engine.submit_stage(plan, "compute")

    assert not any("submission-invocations/compute/attempt-001" in path for path in backend.files)
    assert not backend.lifecycle_claims

    recovered = engine.submit_stage(plan, "compute")

    assert recovered.record.identity.attempt == 1
    assert recovered.record.job_id == "9000"
    assert len(backend.requests) == 2
    assert len(backend.jobs) == 1
    assert not backend.lifecycle_claims


def test_lost_rejection_write_response_preserves_durable_rejection() -> None:
    """An exception after create must not make a rejected attempt reusable."""

    class LoseFirstRejectionResponse(ConcurrentBackend):
        """Publish exact rejection bytes, then lose the creator's response."""

        failed = False

        def write_immutable_text(self, path: str, text: str) -> bool:
            created = super().write_immutable_text(path, text)
            if "/submission-rejections/compute/attempt-001.json" in path and not self.failed:
                self.failed = True
                raise OSError("injected lost rejection response")
            return created

    backend = LoseFirstRejectionResponse()
    backend.outcomes.append(SubmitOutcome(DEFINITELY_REJECTED, returncode=1, stderr="invalid qos"))
    engine = ExecutionEngine(backend)
    plan = _plan()

    with pytest.raises(SubmissionRejected, match="invalid qos"):
        engine.submit_stage(plan, "compute")
    with pytest.raises(SubmissionRejected, match="definitively rejected"):
        engine.submit_stage(plan, "compute")

    assert len(backend.requests) == 1
    assert not backend.jobs
    assert not backend.lifecycle_claims


def test_run_root_is_immutably_bound_to_one_complete_plan() -> None:
    """Disjoint stage paths cannot let a second plan reuse a registered run."""

    backend = ConcurrentBackend()
    first = _plan()
    ExecutionEngine(backend).submit_stage(first, "compute")

    second = _plan(source_commit="d" * 40)
    with pytest.raises(RuntimeError, match="immutable conflict"):
        ExecutionEngine(backend).submit_stage(second, "compute")
    assert len(backend.requests) == 1


def test_pipeline_receipt_cannot_overlap_engine_evidence_namespace() -> None:
    """A scientific command cannot forge completion using an engine receipt path."""

    malicious = ReceiptSpec("forged", "receipts/execution/submissions/compute/attempt-001.json")
    stage = StageSpec(
        stage_id="compute",
        command=_command("compute"),
        resources=_resources(1),
        expected_receipts=(malicious,),
    )
    with pytest.raises(ValueError, match="reserved execution-engine receipt namespace"):
        _plan(stages=(stage,))


def test_dispatcher_checks_runtime_bytes_before_exec(tmp_path) -> None:
    """The signed runtime fingerprint is enforced on the compute-side script."""

    work = tmp_path / "work"
    work.mkdir()
    worker = tmp_path / "worker.sh"
    worker.write_text("#!/bin/sh\nprintf 'ran' > \"$1\"\n")
    worker.chmod(0o755)
    digest = hashlib.sha256(worker.read_bytes()).hexdigest()
    output = tmp_path / "output.txt"
    command = CommandSpec(
        argv=(str(worker), str(output)),
        working_directory=str(work),
        runtime_fingerprint_sha256=digest,
        runtime_fingerprint_path=str(worker),
    )
    dispatcher = render_dispatcher((PlannedTask("task", None, command, (_receipt("runtime"),)),))

    assert "/usr/bin/sha256sum" in dispatcher
    if not posixpath.exists("/usr/bin/sha256sum"):
        shasum = shutil.which("shasum")
        assert shasum is not None
        dispatcher = dispatcher.replace("/usr/bin/sha256sum", f"{shasum} -a 256")

    good = subprocess.run(["/bin/bash"], input=dispatcher, text=True, capture_output=True, check=False)
    assert good.returncode == 0
    assert output.read_text() == "ran"

    worker.write_text("#!/bin/sh\nprintf 'tampered' > \"$1\"\n")
    bad = subprocess.run(["/bin/bash"], input=dispatcher, text=True, capture_output=True, check=False)
    assert bad.returncode == 70
    assert "runtime fingerprint mismatch" in bad.stderr
    assert output.read_text() == "ran"


def test_runtime_fingerprint_cannot_authenticate_an_unrelated_file(tmp_path) -> None:
    """The executable itself, normally an immutable wrapper, must be hashed."""

    executable = tmp_path / "worker"
    executable.write_text("#!/bin/sh\nexit 0\n")
    unrelated = tmp_path / "environment.lock"
    unrelated.write_text("unchanged\n")
    with pytest.raises(ValueError, match=r"must equal argv\[0\]"):
        CommandSpec(
            argv=(str(executable),),
            working_directory=str(tmp_path),
            runtime_fingerprint_path=str(unrelated),
            runtime_fingerprint_sha256=hashlib.sha256(unrelated.read_bytes()).hexdigest(),
        )


def test_retry_submission_automatically_binds_next_afterany_reconciler() -> None:
    """A missing-only retry creates its own dependency-bound audit generation."""

    reconciler = StageSpec(
        stage_id="audit",
        command=_command("audit"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="audit"),),
        depends_on=("compute",),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(max_attempts=2),
    )
    plan = _plan(stages=(_compute_stage(), reconciler))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)
    compute = engine.submit_stage(plan, "compute").record
    first_audit = engine.submit_afterany_reconciler(plan, "audit").record
    backend.set_states(
        compute.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))

    result = engine.reconcile_stage(plan, "compute")

    assert result.retry_submission is not None
    retry_job = result.retry_submission.job_id
    audit_requests = [request for request in backend.requests if request.identity.stage_id == "audit"]
    assert [request.identity.attempt for request in audit_requests] == [1, 2]
    assert audit_requests[0].dependency_job_ids == (compute.job_id,)
    assert audit_requests[1].dependency_job_ids == (retry_job,)
    assert first_audit.identity.attempt == 1


def test_new_afterany_generation_invalidates_older_completion() -> None:
    """A completed audit cannot certify outputs from a later upstream retry."""

    audit = StageSpec(
        stage_id="audit",
        command=_command("audit"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="audit"),),
        depends_on=("compute",),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(max_attempts=2),
    )
    plan = _plan(stages=(_compute_stage(), audit))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)
    compute = engine.submit_stage(plan, "compute").record
    first_audit = engine.submit_afterany_reconciler(plan, "audit").record
    backend.set_states(first_audit.job_id, SlurmTaskState(None, "COMPLETED", 0))
    backend.put_receipt(_receipt("done", stage="audit"))
    assert engine.reconcile_stage(plan, "audit").decision == "COMPLETED"
    assert latest_reconciliation_receipt(backend, plan, audit).attempt == 1

    backend.set_states(
        compute.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))
    retry = engine.reconcile_stage(plan, "compute").retry_submission
    assert retry is not None

    audit_records = engine._submission_records(plan, audit)
    second_audit = max(audit_records, key=lambda record: record.identity.attempt)
    assert second_audit.identity.attempt == 2
    # The immutable attempt-1 completion still exists, but it is no longer
    # current terminal evidence once generation 2 is authorized.
    assert latest_reconciliation_receipt(backend, plan, audit) is None

    backend.set_states(second_audit.job_id, SlurmTaskState(None, "COMPLETED", 0))
    assert engine.reconcile_stage(plan, "audit").decision == "COMPLETED"
    latest = latest_reconciliation_receipt(backend, plan, audit)
    assert latest is not None and latest.attempt == 2


def test_task_bearing_afterany_stage_rebinds_after_upstream_retry() -> None:
    """Every signed task reruns against a replacement dependency generation."""

    preflight = StageSpec(
        stage_id="preflight",
        command=_command("preflight"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="preflight"),),
        retry_policy=RetryPolicy(max_attempts=2, retryable_slurm_states=("NODE_FAIL",)),
    )
    base_compute = _compute_stage()
    dependent_compute = StageSpec(
        stage_id=base_compute.stage_id,
        resources=base_compute.resources,
        expected_receipts=base_compute.expected_receipts,
        tasks=base_compute.tasks,
        depends_on=("preflight",),
        dependency_mode="afterany",
        retry_policy=base_compute.retry_policy,
    )
    plan = _plan(stages=(preflight, dependent_compute))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)
    first_preflight = engine.submit_stage(plan, "preflight").record
    engine.submit_stage(plan, "compute")
    backend.set_states(first_preflight.job_id, SlurmTaskState(None, "NODE_FAIL", 1))

    retry = engine.reconcile_stage(plan, "preflight").retry_submission

    assert retry is not None and retry.identity.attempt == 2
    compute_requests = [request for request in backend.requests if request.identity.stage_id == "compute"]
    assert [request.identity.attempt for request in compute_requests] == [1, 2]
    assert compute_requests[1].dependency_job_ids == (retry.job_id,)
    assert tuple(task.task_id for task in compute_requests[1].tasks) == tuple(
        task.task_id for task in base_compute.tasks
    )


def test_afterany_retry_generation_propagates_through_nested_chain() -> None:
    """A replacement generation must rebind every transitive afterany child."""

    first = StageSpec(
        stage_id="first",
        command=_command("first"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="first"),),
        retry_policy=RetryPolicy(max_attempts=2, retryable_slurm_states=("NODE_FAIL",)),
    )
    middle = StageSpec(
        stage_id="middle",
        command=_command("middle"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="middle"),),
        depends_on=("first",),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(max_attempts=1),
    )
    last = StageSpec(
        stage_id="last",
        command=_command("last"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="last"),),
        depends_on=("middle",),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(max_attempts=1),
    )
    plan = _plan(stages=(first, middle, last))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)
    first_submission = engine.submit_stage(plan, "first").record
    engine.submit_afterany_reconciler(plan, "middle")
    engine.submit_afterany_reconciler(plan, "last")
    backend.set_states(first_submission.job_id, SlurmTaskState(None, "NODE_FAIL", 1))

    retry = engine.reconcile_stage(plan, "first").retry_submission

    assert retry is not None and retry.identity.attempt == 2
    assert signed_attempt_bound(plan, middle) == 2
    assert signed_attempt_bound(plan, last) == 2
    middle_requests = [request for request in backend.requests if request.identity.stage_id == "middle"]
    last_requests = [request for request in backend.requests if request.identity.stage_id == "last"]
    assert [request.identity.attempt for request in middle_requests] == [1, 2]
    assert [request.identity.attempt for request in last_requests] == [1, 2]
    assert last_requests[1].dependency_job_ids == (
        next(job_id for job_id, job in backend.jobs.items() if job["comment"] == middle_requests[1].comment),
    )


def test_afterany_replacement_rebinds_completed_afterok_child() -> None:
    """A completed afterok child cannot certify an obsolete parent generation."""

    first = StageSpec(
        stage_id="first",
        command=_command("first"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="first"),),
        retry_policy=RetryPolicy(max_attempts=2, retryable_slurm_states=("NODE_FAIL",)),
    )
    middle = StageSpec(
        stage_id="middle",
        command=_command("middle"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="middle"),),
        depends_on=("first",),
        dependency_mode="afterany",
    )
    last = StageSpec(
        stage_id="last",
        command=_command("last"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="last"),),
        depends_on=("middle",),
        dependency_mode="afterok",
    )
    plan = _plan(stages=(first, middle, last))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)

    first_submission = engine.submit_stage(plan, "first").record
    first_middle = engine.submit_afterany_reconciler(plan, "middle").record
    backend.set_states(first_middle.job_id, SlurmTaskState(None, "COMPLETED", 0))
    backend.put_receipt(_receipt("done", stage="middle"))
    assert engine.reconcile_stage(plan, "middle").decision == "COMPLETED"

    first_last = engine.submit_stage(plan, "last").record
    backend.set_states(first_last.job_id, SlurmTaskState(None, "COMPLETED", 0))
    backend.put_receipt(_receipt("done", stage="last"))
    assert engine.reconcile_stage(plan, "last").decision == "COMPLETED"

    # The root retry immediately creates middle generation two. The afterok
    # child remains untouched until that replacement parent is certified.
    backend.set_states(first_submission.job_id, SlurmTaskState(None, "NODE_FAIL", 1))
    first_retry = engine.reconcile_stage(plan, "first").retry_submission
    assert first_retry is not None
    middle_records = engine._submission_records(plan, middle)
    assert [record.identity.attempt for record in middle_records] == [1, 2]
    assert [request.identity.attempt for request in backend.requests if request.identity.stage_id == "last"] == [1]

    replacement_middle = middle_records[-1]
    backend.set_states(replacement_middle.job_id, SlurmTaskState(None, "COMPLETED", 0))
    assert engine.reconcile_stage(plan, "middle").decision == "COMPLETED"

    last_requests = [request for request in backend.requests if request.identity.stage_id == "last"]
    assert [request.identity.attempt for request in last_requests] == [1, 2]
    assert last_requests[-1].dependency_job_ids == (replacement_middle.job_id,)
    assert signed_attempt_bound(plan, last) == 2
    # The old completion is no longer current once generation two is authorized.
    assert latest_reconciliation_receipt(backend, plan, last) is None


def test_task_bearing_afterany_retry_still_rebinds_its_downstream_audit() -> None:
    """Dependency mode must not make a compute array look like its audit stage.

    Arrays can legitimately wait ``afterany`` on preflight work.  If such an
    array retries, its downstream audit still needs a fresh scheduler generation
    bound to the retry job, just like an otherwise independent compute stage.
    """

    preflight = StageSpec(
        stage_id="preflight",
        command=_command("preflight"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="preflight"),),
    )
    base_compute = _compute_stage()
    dependent_compute = StageSpec(
        stage_id=base_compute.stage_id,
        resources=base_compute.resources,
        expected_receipts=base_compute.expected_receipts,
        tasks=base_compute.tasks,
        depends_on=("preflight",),
        dependency_mode="afterany",
        retry_policy=base_compute.retry_policy,
    )
    audit = StageSpec(
        stage_id="audit",
        command=_command("audit"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="audit"),),
        depends_on=("compute",),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(max_attempts=2),
    )
    plan = _plan(stages=(preflight, dependent_compute, audit))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)
    engine.submit_stage(plan, "preflight")
    compute = engine.submit_stage(plan, "compute").record
    engine.submit_afterany_reconciler(plan, "audit")
    backend.set_states(
        compute.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))

    result = engine.reconcile_stage(plan, "compute")

    assert result.retry_submission is not None
    retry_job = result.retry_submission.job_id
    audit_requests = [request for request in backend.requests if request.identity.stage_id == "audit"]
    assert [request.identity.attempt for request in audit_requests] == [1, 2]
    assert audit_requests[-1].dependency_job_ids == (retry_job,)


def test_afterany_reconciler_uses_its_own_retry_policy_without_followup_authorization() -> None:
    """An audit's own failure is distinct from a compute-triggered generation.

    A retry-bound follow-up authorization exists only when an accepted upstream
    retry requires a fresh scheduler dependency.  When the audit job itself
    fails, its signed retry policy and immutable reconciliation receipt authorize
    the next attempt directly; requiring a nonexistent upstream follow-up would
    strand the audit despite its explicit retry budget.
    """

    audit = StageSpec(
        stage_id="audit",
        command=_command("audit"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="audit"),),
        depends_on=("compute",),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(
            max_attempts=2,
            retryable_slurm_states=("NODE_FAIL",),
        ),
    )
    plan = _plan(stages=(_compute_stage(), audit))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)
    engine.submit_stage(plan, "compute")
    first_audit = engine.submit_afterany_reconciler(plan, "audit").record
    backend.set_states(first_audit.job_id, SlurmTaskState(None, "NODE_FAIL", 1))

    result = engine.reconcile_stage(plan, "audit")

    assert result.retry_submission is not None
    assert result.retry_submission.identity.attempt == 2
    assert [request.identity.attempt for request in backend.requests if request.identity.stage_id == "audit"] == [
        1,
        2,
    ]
    assert backend.read_text(reconciler_followup_path(plan, "audit", 2)) is None


def test_accepted_compute_retry_is_repairable_when_final_audit_followup_is_rejected() -> None:
    """Persist accepted retry metadata before a fallible final audit launch.

    The upstream retry has already crossed Slurm and produced immutable accepted
    evidence.  Even if the last plan-authorized audit generation is definitively
    rejected, a registry outbox must retain the accepted job and its lifecycle
    holders so metadata-only reconciliation can converge without resubmission.
    """

    class ToggleRegistry:
        """Authenticate plans while allowing synchronization to be paused."""

        enabled = True

        def __init__(self) -> None:
            self.updates: list[RegistryUpdate] = []

        def validate_plan(self, _plan: ExecutionPlan) -> bool:
            return True

        def synchronize(self, _plan: ExecutionPlan, update: RegistryUpdate) -> bool:
            if not self.enabled:
                return False
            self.updates.append(update)
            return True

    class RejectFinalAudit(ConcurrentBackend):
        """Reject only the compute-triggered second audit generation."""

        def invoke_submission(self, request: SubmissionRequest) -> SubmitOutcome:
            if request.identity.stage_id == "audit" and request.identity.attempt == 2:
                return SubmitOutcome(DEFINITELY_REJECTED, returncode=1, stderr="audit qos rejected")
            return super().invoke_submission(request)

    audit = StageSpec(
        stage_id="audit",
        command=_command("audit"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="audit"),),
        depends_on=("compute",),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(max_attempts=1),
    )
    plan = _plan(stages=(_compute_stage(), audit))
    backend = RejectFinalAudit()
    registry = ToggleRegistry()
    engine = ExecutionEngine(backend, registry)
    compute = engine.submit_stage(plan, "compute").record
    engine.submit_afterany_reconciler(plan, "audit")
    backend.set_states(
        compute.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))
    registry.enabled = False

    with pytest.raises(SubmissionRejected, match="audit qos rejected"):
        engine.reconcile_stage(plan, "compute")

    retry_path = pending_registry_path(plan, "compute", 2)
    pending = json.loads(backend.files[retry_path])
    assert pending["attempt"] == 2
    assert pending["job_ids"][-1] == "9002"
    assert pending["lifecycle_claim_ids"]

    # Replay consumes the rejected audit generation, retires its invocation
    # owner, and still fails closed because the signed fan-in bound is exhausted.
    with pytest.raises(ValueError, match="exhausted its signed attempt bound"):
        engine.submit_stage(plan, "compute", attempt=2, task_ids=("movie-1",))

    registry.enabled = True
    assert engine.reconcile_registry(plan)
    assert retry_path not in backend.files
    assert any(update.stage_id == "compute" and update.attempt == 2 for update in registry.updates)
    assert not backend.lifecycle_claims


def test_each_retrying_dependency_receives_a_distinct_followup_generation() -> None:
    """Fan-in retries cannot exhaust a reconciler's per-stage retry budget."""

    compute_a = _compute_stage(stage_id="compute-a", task_prefix="a")
    compute_b = _compute_stage(stage_id="compute-b", task_prefix="b")
    audit = StageSpec(
        stage_id="audit",
        command=_command("audit"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="audit"),),
        depends_on=("compute-a", "compute-b"),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(max_attempts=2),
    )
    plan = _plan(stages=(compute_a, compute_b, audit))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)
    first_a = engine.submit_stage(plan, "compute-a").record
    first_b = engine.submit_stage(plan, "compute-b").record
    engine.submit_afterany_reconciler(plan, "audit")

    def make_one_task_retryable(job_id: str, prefix: str) -> None:
        """Complete two tasks and leave the middle task for one bounded retry."""

        backend.set_states(
            job_id,
            SlurmTaskState(None, "NODE_FAIL", 1),
            SlurmTaskState(0, "COMPLETED", 0),
            SlurmTaskState(2, "COMPLETED", 0),
        )
        backend.put_receipt(_receipt(f"{prefix}-0"))
        backend.put_receipt(_receipt(f"{prefix}-2"))

    make_one_task_retryable(first_a.job_id, "a")
    retry_a = engine.reconcile_stage(plan, "compute-a").retry_submission
    assert retry_a is not None
    make_one_task_retryable(first_b.job_id, "b")
    retry_b = engine.reconcile_stage(plan, "compute-b").retry_submission
    assert retry_b is not None

    audit_requests = [request for request in backend.requests if request.identity.stage_id == "audit"]
    assert [request.identity.attempt for request in audit_requests] == [1, 2, 3]
    assert audit_requests[-1].dependency_job_ids == (retry_a.job_id, retry_b.job_id)
    assert not backend.lifecycle_claims


def test_reconciliation_advances_past_a_definitively_rejected_retry() -> None:
    """A rejected retry consumes attempt two but does not strand attempt three."""

    plan = _plan(stages=(_compute_stage(max_attempts=3),))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)
    compute = engine.submit_stage(plan, "compute").record
    backend.set_states(
        compute.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))
    backend.outcomes.append(SubmitOutcome(DEFINITELY_REJECTED, returncode=1, stderr="invalid qos"))

    with pytest.raises(SubmissionRejected, match="invalid qos"):
        engine.reconcile_stage(plan, "compute")
    recovered = engine.reconcile_stage(plan, "compute")

    assert recovered.retry_submission is not None
    assert recovered.retry_submission.identity.attempt == 3
    assert [request.identity.attempt for request in backend.requests] == [1, 2, 3]


def test_final_rejected_retry_seals_failure_at_consumed_attempt() -> None:
    """A rejected last attempt cannot overwrite the earlier RETRY decision."""

    plan = _plan(stages=(_compute_stage(max_attempts=2),))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)
    compute = engine.submit_stage(plan, "compute").record
    backend.set_states(
        compute.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))
    backend.outcomes.append(SubmitOutcome(DEFINITELY_REJECTED, returncode=1, stderr="invalid qos"))

    with pytest.raises(SubmissionRejected, match="invalid qos"):
        engine.reconcile_stage(plan, "compute")
    exhausted = engine.reconcile_stage(plan, "compute")

    assert exhausted.decision == "FAILED"
    assert exhausted.attempt == 2
    assert backend.read_text(reconciliation_path(plan, "compute", 1)) is not None
    assert backend.read_text(reconciliation_path(plan, "compute", 2)) is not None
    assert [request.identity.attempt for request in backend.requests] == [1, 2]


def test_tampered_followup_dependency_is_rejected_before_intent_or_sbatch() -> None:
    """After-any authorization authenticates jobs and trigger before mutation."""

    class TamperFollowupBackend(ConcurrentBackend):
        def write_immutable_text(self, path: str, text: str) -> bool:
            if "/reconciler-followups/audit/attempt-002.json" in path:
                value = json.loads(text)
                value["dependency_job_ids"] = ["999999"]
                text = canonical_json(value)
            return super().write_immutable_text(path, text)

    audit = StageSpec(
        stage_id="audit",
        command=_command("audit"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="audit"),),
        depends_on=("compute",),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(max_attempts=2),
    )
    plan = _plan(stages=(_compute_stage(), audit))
    backend = TamperFollowupBackend()
    engine = ExecutionEngine(backend)
    compute = engine.submit_stage(plan, "compute").record
    engine.submit_afterany_reconciler(plan, "audit")
    backend.set_states(
        compute.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))

    with pytest.raises(ValueError, match="dependency job|authorized signed DAG"):
        engine.reconcile_stage(plan, "compute")
    assert not any("submission-intents/audit/attempt-002" in path for path in backend.files)
    assert [request.identity.attempt for request in backend.requests if request.identity.stage_id == "audit"] == [1]


def test_retry_followup_outbox_recovers_crash_after_retry_record() -> None:
    """Replaying an accepted retry repairs exactly one missing audit generation."""

    class CrashAfterRetryRecord(ConcurrentBackend):
        crashed = False

        def write_immutable_text(self, path: str, text: str) -> bool:
            created = super().write_immutable_text(path, text)
            if created and "/submissions/compute/attempt-002.json" in path and not self.crashed:
                self.crashed = True
                raise KeyboardInterrupt("died before follow-up outbox")
            return created

    audit = StageSpec(
        stage_id="audit",
        command=_command("audit"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="audit"),),
        depends_on=("compute",),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(max_attempts=2),
    )
    plan = _plan(stages=(_compute_stage(), audit))
    backend = CrashAfterRetryRecord()
    engine = ExecutionEngine(backend)
    compute = engine.submit_stage(plan, "compute").record
    engine.submit_afterany_reconciler(plan, "audit")
    backend.set_states(
        compute.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))

    with pytest.raises(KeyboardInterrupt, match="before follow-up"):
        engine.reconcile_stage(plan, "compute")
    assert [request.identity.attempt for request in backend.requests if request.identity.stage_id == "audit"] == [1]

    # Recovery replays the accepted compute attempt rather than polling its task
    # state or resubmitting it.  The durable record converges the outbox once.
    recovered = ExecutionEngine(backend).submit_stage(
        plan,
        "compute",
        attempt=2,
        task_ids=("movie-1",),
    )
    assert recovered.record.identity.attempt == 2
    audit_attempts = [request.identity.attempt for request in backend.requests if request.identity.stage_id == "audit"]
    assert audit_attempts == [1, 2]
    ExecutionEngine(backend).submit_stage(
        plan,
        "compute",
        attempt=2,
        task_ids=("movie-1",),
    )
    assert [request.identity.attempt for request in backend.requests if request.identity.stage_id == "audit"] == [1, 2]


def test_rejected_followup_generation_advances_without_replaying_sbatch() -> None:
    """A definitive audit rejection consumes its generation and authorizes the next."""

    class RejectSecondAudit(ConcurrentBackend):
        rejected = False

        def invoke_submission(self, request: SubmissionRequest) -> SubmitOutcome:
            """Reject only the first retry-bound audit scheduler call."""

            if request.identity.stage_id == "audit" and request.identity.attempt == 2 and not self.rejected:
                self.rejected = True
                self.outcomes.appendleft(SubmitOutcome(DEFINITELY_REJECTED, returncode=1, stderr="audit qos rejected"))
            return super().invoke_submission(request)

    audit = StageSpec(
        stage_id="audit",
        command=_command("audit"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="audit"),),
        depends_on=("compute",),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(max_attempts=2),
    )
    plan = _plan(stages=(_compute_stage(), audit))
    backend = RejectSecondAudit()
    engine = ExecutionEngine(backend)
    compute = engine.submit_stage(plan, "compute").record
    engine.submit_afterany_reconciler(plan, "audit")
    backend.set_states(
        compute.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))

    with pytest.raises(SubmissionRejected, match="audit qos rejected"):
        engine.reconcile_stage(plan, "compute")
    # Replaying the accepted compute retry consumes audit attempt two's signed
    # rejection and submits attempt three exactly once.
    ExecutionEngine(backend).submit_stage(plan, "compute", attempt=2, task_ids=("movie-1",))

    audit_attempts = [request.identity.attempt for request in backend.requests if request.identity.stage_id == "audit"]
    assert audit_attempts == [1, 2, 3]
    assert not backend.lifecycle_claims


def test_distinct_dependency_retry_skips_rejected_occupied_followup() -> None:
    """Another dependency cannot overwrite a rejected generation's authorization."""

    class RejectSecondAudit(ConcurrentBackend):
        rejected = False

        def invoke_submission(self, request: SubmissionRequest) -> SubmitOutcome:
            """Reject A's first retry audit, then accept later generations."""

            if request.identity.stage_id == "audit" and request.identity.attempt == 2 and not self.rejected:
                self.rejected = True
                self.outcomes.appendleft(SubmitOutcome(DEFINITELY_REJECTED, returncode=1, stderr="audit qos rejected"))
            return super().invoke_submission(request)

    compute_a = _compute_stage(stage_id="compute-a", task_prefix="a")
    compute_b = _compute_stage(stage_id="compute-b", task_prefix="b")
    audit = StageSpec(
        stage_id="audit",
        command=_command("audit"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="audit"),),
        depends_on=("compute-a", "compute-b"),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(max_attempts=2),
    )
    plan = _plan(stages=(compute_a, compute_b, audit))
    backend = RejectSecondAudit()
    engine = ExecutionEngine(backend)
    first_a = engine.submit_stage(plan, "compute-a").record
    first_b = engine.submit_stage(plan, "compute-b").record
    engine.submit_afterany_reconciler(plan, "audit")

    def make_retryable(job_id: str, prefix: str) -> None:
        """Leave only the middle task eligible for retry."""

        backend.set_states(
            job_id,
            SlurmTaskState(None, "NODE_FAIL", 1),
            SlurmTaskState(0, "COMPLETED", 0),
            SlurmTaskState(2, "COMPLETED", 0),
        )
        backend.put_receipt(_receipt(f"{prefix}-0"))
        backend.put_receipt(_receipt(f"{prefix}-2"))

    make_retryable(first_a.job_id, "a")
    with pytest.raises(SubmissionRejected, match="audit qos rejected"):
        engine.reconcile_stage(plan, "compute-a")

    # B retries before A's rejected follow-up is repaired. It must allocate
    # attempt three rather than conflict with A's immutable attempt-two outbox.
    make_retryable(first_b.job_id, "b")
    retry_b = engine.reconcile_stage(plan, "compute-b").retry_submission
    assert retry_b is not None
    ExecutionEngine(backend).submit_stage(plan, "compute-a", attempt=2, task_ids=("a-1",))

    audit_attempts = [request.identity.attempt for request in backend.requests if request.identity.stage_id == "audit"]
    assert audit_attempts == [1, 2, 3, 4]
    assert not backend.lifecycle_claims


def test_afterok_requires_authenticated_completion_not_scheduler_exit() -> None:
    """A zero Slurm exit cannot release downstream work before receipt validation."""

    downstream = StageSpec(
        stage_id="publish",
        command=_command("publish"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="publish"),),
        depends_on=("compute",),
        dependency_mode="afterok",
    )
    plan = _plan(stages=(_compute_stage(), downstream))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)
    compute = engine.submit_stage(plan, "compute").record
    backend.set_states(
        compute.job_id,
        SlurmTaskState(None, "COMPLETED", 0),
        *(SlurmTaskState(index, "COMPLETED", 0) for index in range(3)),
    )

    with pytest.raises(ValueError, match="authenticated COMPLETED"):
        engine.submit_stage(plan, "publish")
    assert engine.reconcile_stage(plan, "compute").decision == "FAILED"

    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)
    compute = engine.submit_stage(plan, "compute").record
    backend.set_states(
        compute.job_id,
        SlurmTaskState(None, "COMPLETED", 0),
        *(SlurmTaskState(index, "COMPLETED", 0) for index in range(3)),
    )
    for index in range(3):
        backend.put_receipt(_receipt(f"movie-{index}"))
    assert engine.reconcile_stage(plan, "compute").decision == "COMPLETED"
    published = engine.submit_stage(plan, "publish")
    assert published.record.dependency_job_ids == (compute.job_id,)


@pytest.mark.parametrize(("dependency_mode", "allowed"), (("afterok", True), ("afterany", False)))
def test_failed_ancestor_suppresses_only_afterok_descendants(
    dependency_mode: str,
    allowed: bool,
) -> None:
    """Archive still requires terminal evidence from after-any descendants."""

    upstream = _compute_stage()
    downstream = StageSpec(
        stage_id="audit",
        command=_command("audit"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="audit"),),
        depends_on=("compute",),
        dependency_mode=dependency_mode,
    )
    plan = _plan(stages=(upstream, downstream))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)
    record = engine.submit_stage(plan, "compute").record
    backend.set_states(
        record.job_id,
        SlurmTaskState(None, "CANCELLED", 1),
        *(SlurmTaskState(index, "CANCELLED", 1) for index in range(3)),
    )
    assert engine.reconcile_stage(plan, "compute").decision == "FAILED"

    if allowed:
        require_current_terminal_evidence(backend, plan, "archive")
    else:
        with pytest.raises(ValueError, match="audit lacks authenticated terminal"):
            require_current_terminal_evidence(backend, plan, "archive")


def test_completed_afterany_stage_resets_failure_suppression() -> None:
    """Required after-ok work below a successful recovery stage cannot be skipped."""

    failed_compute = _compute_stage()
    recovery = StageSpec(
        stage_id="recovery",
        command=_command("recovery"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="recovery"),),
        depends_on=("compute",),
        dependency_mode="afterany",
    )
    publish = StageSpec(
        stage_id="publish",
        command=_command("publish"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="publish"),),
        depends_on=("recovery",),
        dependency_mode="afterok",
    )
    plan = _plan(stages=(failed_compute, recovery, publish))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)

    compute = engine.submit_stage(plan, "compute").record
    backend.set_states(
        compute.job_id,
        SlurmTaskState(None, "CANCELLED", 1),
        *(SlurmTaskState(index, "CANCELLED", 1) for index in range(3)),
    )
    assert engine.reconcile_stage(plan, "compute").decision == "FAILED"

    recovery_job = engine.submit_stage(plan, "recovery").record
    backend.set_states(recovery_job.job_id, SlurmTaskState(None, "COMPLETED", 0))
    backend.put_receipt(_receipt("done", stage="recovery"))
    assert engine.reconcile_stage(plan, "recovery").decision == "COMPLETED"

    with pytest.raises(ValueError, match="publish lacks authenticated terminal"):
        require_current_terminal_evidence(backend, plan, "archive")


def test_malformed_reconciliation_cannot_release_afterok_stage() -> None:
    """A decision string without plan-bound task evidence is not completion proof."""

    publish = StageSpec(
        stage_id="publish",
        command=_command("publish"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="publish"),),
        depends_on=("compute",),
        dependency_mode="afterok",
    )
    plan = _plan(stages=(_compute_stage(), publish))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)
    engine.submit_stage(plan, "compute")
    backend.files[reconciliation_path(plan, "compute", 1)] = '{"decision":"COMPLETED"}\n'

    with pytest.raises(ValueError, match="invalid immutable reconciliation"):
        engine.submit_stage(plan, "publish")
    assert [request.identity.stage_id for request in backend.requests] == ["compute"]


def test_afterok_rechecks_current_task_receipts_after_reconciliation() -> None:
    """Historical success cannot hide a deleted or replaced files-as-truth receipt."""

    publish = StageSpec(
        stage_id="publish",
        command=_command("publish"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="publish"),),
        depends_on=("compute",),
        dependency_mode="afterok",
    )
    plan = _plan(stages=(_compute_stage(), publish))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)
    compute = engine.submit_stage(plan, "compute").record
    backend.set_states(
        compute.job_id,
        SlurmTaskState(None, "COMPLETED", 0),
        *(SlurmTaskState(index, "COMPLETED", 0) for index in range(3)),
    )
    for index in range(3):
        backend.put_receipt(_receipt(f"movie-{index}"))
    assert engine.reconcile_stage(plan, "compute").decision == "COMPLETED"

    del backend.files[posixpath.join(RUN_ROOT, _receipt("movie-1").path)]
    with pytest.raises(ValueError, match="authenticated COMPLETED"):
        engine.submit_stage(plan, "publish")
    assert [request.identity.stage_id for request in backend.requests] == ["compute"]


def test_untrustworthy_receipt_observation_never_authorizes_retry() -> None:
    """A read failure remains transient instead of becoming missing evidence."""

    class FailingReceiptProbe(ConcurrentBackend):
        def observe_receipt(self, run_root: str, receipt: ReceiptSpec) -> ReceiptObservation:
            if receipt.receipt_id == "receipt-compute-movie-1":
                return ReceiptObservation(
                    receipt.path,
                    False,
                    trustworthy=False,
                    error="simulated transport failure",
                )
            return super().observe_receipt(run_root, receipt)

    backend = FailingReceiptProbe()
    engine = ExecutionEngine(backend)
    plan = _plan()
    record = engine.submit_stage(plan, "compute").record
    backend.set_states(
        record.job_id,
        SlurmTaskState(None, "COMPLETED", 0),
        *(SlurmTaskState(index, "COMPLETED", 0) for index in range(3)),
    )
    for index in range(3):
        backend.put_receipt(_receipt(f"movie-{index}"))

    result = engine.reconcile_stage(plan, "compute")
    assert result.decision == "WAIT"
    assert result.active_task_ids == ("movie-1",)
    assert len(backend.requests) == 1
    assert backend.read_text(task_attempt_path(plan, record.identity, "movie-1")) is None


def test_manifest_merge_is_monotonic_for_attempts_and_terminal_state() -> None:
    """Delayed callbacks cannot regress attempt two or resurrect a failed plan."""

    plan = _plan()
    manifest = RunManifest(
        run_id=RUN_ID,
        campaign=CAMPAIGN,
        pipeline="canary",
        created_utc="20260822T010203Z",
        datasets=["dataset-1"],
        extra={"future": {"preserved": True}},
    )
    retrying = RegistryUpdate(plan.plan_sha256, "compute", "RETRYING", "RETRYING", ("9000", "9001"), 2)
    delayed = RegistryUpdate(plan.plan_sha256, "compute", "SUBMITTED", "SUBMITTED", ("9000",), 1)
    failed = RegistryUpdate(plan.plan_sha256, "compute", "FAILED", "FAILED", ("9000", "9001"), 2)
    late_audit = RegistryUpdate(plan.plan_sha256, "audit", "COMPLETED", "COMPLETED", ("9002",), 1)

    merged = merge_execution_manifest(manifest, plan, retrying)
    merged = merge_execution_manifest(merged, plan, delayed)
    assert merged.provenance["execution"]["stages"]["compute"] == {
        "attempt": 2,
        "status": "RETRYING",
    }
    merged = merge_execution_manifest(merged, plan, failed)
    merged = merge_execution_manifest(merged, plan, late_audit)
    assert merged.provenance["execution"]["state"] == "FAILED"
    assert merged.result["status"] == "FAILED"
    assert merged.extra["future"] == {"preserved": True}


def test_failed_registry_state_dominates_completed_in_both_orders() -> None:
    """Terminal registry meaning is deterministic under callback reordering."""

    plan = _plan()
    manifest = RunManifest(
        run_id=RUN_ID,
        campaign=CAMPAIGN,
        pipeline="canary",
        created_utc="20260822T010203Z",
        datasets=["dataset-1"],
    )
    completed = RegistryUpdate(plan.plan_sha256, "compute", "COMPLETED", "COMPLETED", ("9000",), 1)
    failed = RegistryUpdate(plan.plan_sha256, "compute", "FAILED", "FAILED", ("9000",), 1)

    for first, second in ((completed, failed), (failed, completed)):
        merged = merge_execution_manifest(manifest, plan, first)
        merged = merge_execution_manifest(merged, plan, second)
        execution = merged.provenance["execution"]
        assert execution["state"] == "FAILED"
        assert execution["stages"]["compute"] == {"attempt": 1, "status": "FAILED"}


def test_root_cancellation_with_absent_child_state_is_not_missing_retry() -> None:
    """Root CANCELLED is inherited by absent child rows and remains terminal."""

    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)
    plan = _plan()
    record = engine.submit_stage(plan, "compute").record
    backend.set_states(
        record.job_id,
        SlurmTaskState(None, "CANCELLED", 0),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))

    result = engine.reconcile_stage(plan, "compute")

    assert result.decision == "FAILED"
    assert result.failed_task_ids == ("movie-1",)
    assert len(backend.requests) == 1
