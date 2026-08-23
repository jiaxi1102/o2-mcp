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
import sys
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
from o2mcp.runorg.execution_backend import RECEIPT_PROBE_PROGRAM
from o2mcp.runorg.execution_evidence import (
    latest_reconciliation_receipt,
    read_plan_submission_records,
)
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
from o2mcp.runorg.execution_reconcile import signed_attempt_bound, stage_by_id
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


def _dependent_plan(*, preflight_attempts: int = 2) -> ExecutionPlan:
    """Build one ``afterany`` array behind a single retryable prerequisite."""

    preflight = StageSpec(
        stage_id="preflight",
        command=_command("preflight"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="preflight"),),
        retry_policy=RetryPolicy(max_attempts=preflight_attempts, retryable_slurm_states=("NODE_FAIL",)),
    )
    base_compute = _compute_stage()
    compute = StageSpec(
        stage_id=base_compute.stage_id,
        resources=base_compute.resources,
        expected_receipts=base_compute.expected_receipts,
        tasks=base_compute.tasks,
        depends_on=("preflight",),
        dependency_mode="afterany",
        retry_policy=base_compute.retry_policy,
    )
    return _plan(stages=(preflight, compute))


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


def test_first_successful_parent_retry_does_not_create_child_followup() -> None:
    """A never-launched afterok child keeps ordinary attempt-one authorization."""

    parent = StageSpec(
        stage_id="parent",
        command=_command("parent"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="parent"),),
        retry_policy=RetryPolicy(max_attempts=2, retryable_slurm_states=("NODE_FAIL",)),
    )
    child = StageSpec(
        stage_id="child",
        command=_command("child"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="child"),),
        depends_on=("parent",),
        dependency_mode="afterok",
    )
    plan = _plan(stages=(parent, child))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)

    first = engine.submit_stage(plan, "parent").record
    backend.set_states(first.job_id, SlurmTaskState(None, "NODE_FAIL", 1))
    retry = engine.reconcile_stage(plan, "parent").retry_submission
    assert retry is not None and retry.identity.attempt == 2

    backend.set_states(retry.job_id, SlurmTaskState(None, "COMPLETED", 0))
    backend.put_receipt(_receipt("done", stage="parent"))
    assert engine.reconcile_stage(plan, "parent").decision == "COMPLETED"

    assert engine._submission_records(plan, child) == ()
    assert backend.read_text(reconciler_followup_path(plan, "child", 1)) is None
    assert engine.submit_stage(plan, "child").record.identity.attempt == 1


def test_afterok_followup_waits_for_every_regenerated_prerequisite() -> None:
    """Fan-in authorization is published only after every latest parent completes."""

    roots = tuple(
        StageSpec(
            stage_id=f"root-{name}",
            command=_command(f"root-{name}"),
            resources=_resources(1),
            expected_receipts=(_receipt("done", stage=f"root-{name}"),),
            retry_policy=RetryPolicy(max_attempts=2, retryable_slurm_states=("NODE_FAIL",)),
        )
        for name in ("a", "b")
    )
    middles = tuple(
        StageSpec(
            stage_id=f"middle-{name}",
            command=_command(f"middle-{name}"),
            resources=_resources(1),
            expected_receipts=(_receipt("done", stage=f"middle-{name}"),),
            depends_on=(f"root-{name}",),
            dependency_mode="afterany",
        )
        for name in ("a", "b")
    )
    child = StageSpec(
        stage_id="fan-in",
        command=_command("fan-in"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="fan-in"),),
        depends_on=("middle-a", "middle-b"),
        dependency_mode="afterok",
    )
    plan = _plan(stages=(*roots, *middles, child))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)

    root_records = {stage.stage_id: engine.submit_stage(plan, stage.stage_id).record for stage in roots}
    middle_records = {
        stage.stage_id: engine.submit_afterany_reconciler(plan, stage.stage_id).record for stage in middles
    }
    for stage in middles:
        record = middle_records[stage.stage_id]
        backend.set_states(record.job_id, SlurmTaskState(None, "COMPLETED", 0))
        backend.put_receipt(_receipt("done", stage=stage.stage_id))
        assert engine.reconcile_stage(plan, stage.stage_id).decision == "COMPLETED"

    first_child = engine.submit_stage(plan, "fan-in").record
    backend.set_states(first_child.job_id, SlurmTaskState(None, "COMPLETED", 0))
    backend.put_receipt(_receipt("done", stage="fan-in"))
    assert engine.reconcile_stage(plan, "fan-in").decision == "COMPLETED"

    for stage_id, record in root_records.items():
        backend.set_states(record.job_id, SlurmTaskState(None, "NODE_FAIL", 1))
        assert engine.reconcile_stage(plan, stage_id).retry_submission is not None

    replacements = {stage.stage_id: engine._submission_records(plan, stage)[-1] for stage in middles}
    first_replacement = replacements["middle-a"]
    backend.set_states(first_replacement.job_id, SlurmTaskState(None, "COMPLETED", 0))
    assert engine.reconcile_stage(plan, "middle-a").decision == "COMPLETED"
    assert backend.read_text(reconciler_followup_path(plan, "fan-in", 2)) is None

    second_replacement = replacements["middle-b"]
    backend.set_states(second_replacement.job_id, SlurmTaskState(None, "COMPLETED", 0))
    assert engine.reconcile_stage(plan, "middle-b").decision == "COMPLETED"

    child_requests = [request for request in backend.requests if request.identity.stage_id == "fan-in"]
    assert [request.identity.attempt for request in child_requests] == [1, 2]
    assert child_requests[-1].dependency_job_ids == (
        first_replacement.job_id,
        second_replacement.job_id,
    )


def test_afterok_followup_uses_latest_success_after_replacement_retry() -> None:
    """A failed replacement cannot remain the trigger for its successful retry."""

    root = StageSpec(
        stage_id="root",
        command=_command("root"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="root"),),
        retry_policy=RetryPolicy(max_attempts=2, retryable_slurm_states=("NODE_FAIL",)),
    )
    middle = StageSpec(
        stage_id="middle",
        command=_command("middle"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="middle"),),
        depends_on=("root",),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(max_attempts=2, retryable_slurm_states=("NODE_FAIL",)),
    )
    child = StageSpec(
        stage_id="child",
        command=_command("child"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="child"),),
        depends_on=("middle",),
        dependency_mode="afterok",
    )
    plan = _plan(stages=(root, middle, child))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)

    root_one = engine.submit_stage(plan, "root").record
    middle_one = engine.submit_afterany_reconciler(plan, "middle").record
    backend.set_states(middle_one.job_id, SlurmTaskState(None, "COMPLETED", 0))
    backend.put_receipt(_receipt("done", stage="middle"))
    assert engine.reconcile_stage(plan, "middle").decision == "COMPLETED"
    child_one = engine.submit_stage(plan, "child").record
    backend.set_states(child_one.job_id, SlurmTaskState(None, "COMPLETED", 0))
    backend.put_receipt(_receipt("done", stage="child"))
    assert engine.reconcile_stage(plan, "child").decision == "COMPLETED"

    backend.set_states(root_one.job_id, SlurmTaskState(None, "NODE_FAIL", 1))
    assert engine.reconcile_stage(plan, "root").retry_submission is not None
    middle_two = engine._submission_records(plan, middle)[-1]
    backend.set_states(middle_two.job_id, SlurmTaskState(None, "NODE_FAIL", 1))
    middle_retry = engine.reconcile_stage(plan, "middle").retry_submission
    assert middle_retry is not None and middle_retry.identity.attempt == 3

    backend.set_states(middle_retry.job_id, SlurmTaskState(None, "COMPLETED", 0))
    assert engine.reconcile_stage(plan, "middle").decision == "COMPLETED"

    child_requests = [request for request in backend.requests if request.identity.stage_id == "child"]
    assert [request.identity.attempt for request in child_requests] == [1, 2]
    assert child_requests[-1].dependency_job_ids == (middle_retry.job_id,)
    authorization = json.loads(backend.files[reconciler_followup_path(plan, "child", 2)])
    assert authorization["trigger_job_id"] == middle_retry.job_id


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


def test_missing_only_retry_preserves_dependency_generation_after_coordinator_crash() -> None:
    """A delayed ordinary retry cannot mix successes with a newer prerequisite.

    This models a coordinator dying after the dependent array's reconciliation
    receipt and the prerequisite retry record are durable, but before follow-up
    propagation runs.  Replaying the dependent retry must keep its successful
    tasks and dependency job from generation one together.
    """

    plan = _dependent_plan()
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)

    preflight_one = engine.submit_stage(plan, "preflight").record
    compute_one = engine.submit_stage(plan, "compute").record
    backend.set_states(
        compute_one.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))
    decision = engine.reconcile_stage(plan, "compute", submit_retry=False)
    assert decision.decision == "RETRY_MISSING_ONLY"

    # Suppress only this call's automatic child propagation to reproduce the
    # exact post-acceptance crash window. A fresh coordinator will recover from
    # the immutable records below.
    engine._ensure_reconciler_followups = lambda *_args, **_kwargs: ()  # type: ignore[method-assign]
    backend.set_states(preflight_one.job_id, SlurmTaskState(None, "NODE_FAIL", 1))
    preflight_two = engine.reconcile_stage(plan, "preflight").retry_submission
    assert preflight_two is not None

    compute_two = (
        ExecutionEngine(backend)
        .submit_stage(
            plan,
            "compute",
            attempt=2,
            task_ids=("movie-1",),
        )
        .record
    )

    assert compute_two.task_ids == ("movie-1",)
    assert compute_two.dependency_job_ids == (preflight_one.job_id,)
    assert compute_two.dependency_job_ids != (preflight_two.job_id,)


def test_rejected_dependency_followup_still_binds_its_own_retry_generation() -> None:
    """A rejected follow-up, not a newer prerequisite, authorizes its retry.

    An immutable follow-up authorization opens a replacement generation even when
    the scheduler refuses that exact call.  Retrying the rejected generation must
    reuse the follow-up's dependency tuple; resolving the DAG again would silently
    skip to a prerequisite generation that no authorization covers.
    """

    class RejectFirstComputeFollowup(ConcurrentBackend):
        """Refuse only the first dependency-triggered compute generation."""

        def invoke_submission(self, request: SubmissionRequest) -> SubmitOutcome:
            if request.identity.stage_id == "compute" and request.identity.attempt == 2:
                return SubmitOutcome(DEFINITELY_REJECTED, returncode=1, stderr="compute qos rejected")
            return super().invoke_submission(request)

    plan = _dependent_plan(preflight_attempts=3)
    preflight = stage_by_id(plan, "preflight")
    backend = RejectFirstComputeFollowup()
    engine = ExecutionEngine(backend)

    preflight_one = engine.submit_stage(plan, "preflight").record
    engine.submit_stage(plan, "compute")
    backend.set_states(preflight_one.job_id, SlurmTaskState(None, "NODE_FAIL", 1))
    with pytest.raises(SubmissionRejected, match="compute qos rejected"):
        engine.reconcile_stage(plan, "preflight")
    preflight_two = read_plan_submission_records(backend, plan, preflight)[-1]
    assert preflight_two.identity.attempt == 2

    # Crash before the next propagation so a third prerequisite generation is
    # accepted without ever writing compute's follow-up authorization for it.
    engine._ensure_reconciler_followups = lambda *_args, **_kwargs: ()  # type: ignore[method-assign]
    backend.set_states(preflight_two.job_id, SlurmTaskState(None, "NODE_FAIL", 1))
    preflight_three = engine.reconcile_stage(plan, "preflight").retry_submission
    assert preflight_three is not None

    compute_three = (
        ExecutionEngine(backend)
        .submit_stage(
            plan,
            "compute",
            attempt=3,
            task_ids=("movie-0", "movie-1", "movie-2"),
        )
        .record
    )

    assert compute_three.task_ids == ("movie-0", "movie-1", "movie-2")
    assert compute_three.dependency_job_ids == (preflight_two.job_id,)
    assert compute_three.dependency_job_ids != (preflight_three.job_id,)


def test_rejected_first_dependent_attempt_reruns_against_the_current_generation() -> None:
    """Without preserved successes a full rerun may adopt the latest prerequisite.

    Pinning exists to stop a preserved task from pairing with a newer dependency
    job.  A dependent stage whose only prior call was rejected has no accepted
    outputs at all, so binding its full-task retry to the currently authenticated
    prerequisite mixes nothing.
    """

    class RejectFirstCompute(ConcurrentBackend):
        """Refuse only the dependent stage's initial submission."""

        def invoke_submission(self, request: SubmissionRequest) -> SubmitOutcome:
            if request.identity.stage_id == "compute" and request.identity.attempt == 1:
                return SubmitOutcome(DEFINITELY_REJECTED, returncode=1, stderr="compute qos rejected")
            return super().invoke_submission(request)

    plan = _dependent_plan()
    compute = stage_by_id(plan, "compute")
    backend = RejectFirstCompute()
    engine = ExecutionEngine(backend)

    preflight_one = engine.submit_stage(plan, "preflight").record
    with pytest.raises(SubmissionRejected, match="compute qos rejected"):
        engine.submit_stage(plan, "compute")
    assert not read_plan_submission_records(backend, plan, compute)

    # Suppress propagation so the retry travels the ordinary rejection path
    # rather than a dependency follow-up authorization.
    engine._ensure_reconciler_followups = lambda *_args, **_kwargs: ()  # type: ignore[method-assign]
    backend.set_states(preflight_one.job_id, SlurmTaskState(None, "NODE_FAIL", 1))
    preflight_two = engine.reconcile_stage(plan, "preflight").retry_submission
    assert preflight_two is not None

    compute_two = (
        ExecutionEngine(backend)
        .submit_stage(
            plan,
            "compute",
            attempt=2,
            task_ids=("movie-0", "movie-1", "movie-2"),
        )
        .record
    )

    assert compute_two.task_ids == ("movie-0", "movie-1", "movie-2")
    assert compute_two.dependency_job_ids == (preflight_two.job_id,)


def test_followup_published_during_an_ordinary_retry_does_not_invalidate_it() -> None:
    """The pre-submit intent, not a later authorization, settles one attempt.

    A pending missing-only retry and dependency propagation can target the same
    attempt identity.  If the follow-up authorization lands between the retry's
    authorization read and its intent write, the retry still owns the attempt.
    Its record must stay readable, and the replacement generation must move to
    the next attempt instead of silently vanishing.
    """

    class PublishFollowupAtIntent(ConcurrentBackend):
        """Land a legitimate follow-up exactly before the retry writes intent."""

        pending_authorization: tuple[str, str] | None = None

        def write_immutable_text(self, path: str, text: str) -> bool:
            if self.pending_authorization is not None and "/submission-intents/compute/" in path:
                target, payload = self.pending_authorization
                self.pending_authorization = None
                super().write_immutable_text(target, payload)
            return super().write_immutable_text(path, text)

    plan = _dependent_plan()
    compute = stage_by_id(plan, "compute")
    backend = PublishFollowupAtIntent()
    engine = ExecutionEngine(backend)

    preflight_one = engine.submit_stage(plan, "preflight").record
    compute_one = engine.submit_stage(plan, "compute").record
    backend.set_states(
        compute_one.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))
    assert engine.reconcile_stage(plan, "compute", submit_retry=False).decision == "RETRY_MISSING_ONLY"

    # Accept the prerequisite retry, holding its child propagation so the
    # follow-up can be published at the exact racing instant below.
    engine._ensure_reconciler_followups = lambda *_args, **_kwargs: ()  # type: ignore[method-assign]
    backend.set_states(preflight_one.job_id, SlurmTaskState(None, "NODE_FAIL", 1))
    preflight_two = engine.reconcile_stage(plan, "preflight").retry_submission
    assert preflight_two is not None
    backend.pending_authorization = (
        reconciler_followup_path(plan, "compute", 2),
        canonical_json(
            {
                "attempt": 2,
                "dependency_job_ids": [preflight_two.job_id],
                "plan_sha256": plan.plan_sha256,
                "schema_version": 1,
                "stage_id": "compute",
                "trigger_job_id": preflight_two.job_id,
                "trigger_stage_id": "preflight",
            }
        ),
    )

    winner = (
        ExecutionEngine(backend)
        .submit_stage(
            plan,
            "compute",
            attempt=2,
            task_ids=("movie-1",),
        )
        .record
    )
    assert backend.read_text(reconciler_followup_path(plan, "compute", 2)) is not None

    # The winning ordinary attempt stays authenticated evidence rather than
    # failing strict validation against the authorization that lost the race.
    records = read_plan_submission_records(backend, plan, compute)
    assert [record.identity.attempt for record in records] == [1, 2]
    assert records[-1].task_ids == winner.task_ids == ("movie-1",)
    assert records[-1].dependency_job_ids == (preflight_one.job_id,)

    # The losing authorization must not be reported as the current generation
    # either: filtering records to it would discard the successes this retry
    # deliberately preserved and seal a spurious failure.
    backend.set_states(winner.job_id, SlurmTaskState(None, "COMPLETED", 0), SlurmTaskState(1, "COMPLETED", 0))
    backend.put_receipt(_receipt("movie-1"))
    assert ExecutionEngine(backend).reconcile_stage(plan, "compute").decision == "COMPLETED"

    # The replacement generation is not lost: propagation opens the next attempt
    # with every signed task bound to the new prerequisite.
    replacement = ExecutionEngine(backend)._ensure_reconciler_followups(plan, "preflight", preflight_two)
    assert [record.identity.attempt for record in replacement] == [3]
    assert replacement[0].task_ids == ("movie-0", "movie-1", "movie-2")
    assert replacement[0].dependency_job_ids == (preflight_two.job_id,)


def test_promote_refuses_a_child_completed_against_a_superseded_generation() -> None:
    """Terminal receipts alone cannot prove a dependent run is settled.

    An upstream replacement generation completes before its downstream
    replacement authorization can be published, and a crash can end the run
    permanently inside that window.  A promote that only saw two COMPLETED
    receipts would delete the source while the child still owes a rerun against
    the newer prerequisite, so the guard compares consumed against current
    dependency jobs instead of trusting the decisions.
    """

    preflight = StageSpec(
        stage_id="preflight",
        command=_command("preflight"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="preflight"),),
        retry_policy=RetryPolicy(max_attempts=2, retryable_slurm_states=("NODE_FAIL",)),
    )
    base_compute = _compute_stage()
    compute = StageSpec(
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
        dependency_mode="afterok",
        retry_policy=RetryPolicy(max_attempts=2),
    )
    plan = _plan(stages=(preflight, compute, audit))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)

    def complete_compute(record) -> None:
        backend.set_states(
            record.job_id,
            SlurmTaskState(None, "COMPLETED", 0),
            *(SlurmTaskState(index, "COMPLETED", 0) for index in range(3)),
        )
        for index in range(3):
            backend.put_receipt(_receipt(f"movie-{index}"))

    preflight_one = engine.submit_stage(plan, "preflight").record
    compute_one = engine.submit_stage(plan, "compute").record
    complete_compute(compute_one)
    assert engine.reconcile_stage(plan, "compute").decision == "COMPLETED"

    audit_one = engine.submit_stage(plan, "audit").record
    backend.set_states(audit_one.job_id, SlurmTaskState(None, "COMPLETED", 0))
    backend.put_receipt(_receipt("done", stage="audit"))
    assert engine.reconcile_stage(plan, "audit").decision == "COMPLETED"
    assert audit_one.dependency_job_ids == (compute_one.job_id,)

    # The prerequisite retries and its replacement completes, which rebinds and
    # reruns compute but leaves audit on the superseded generation.
    backend.set_states(preflight_one.job_id, SlurmTaskState(None, "NODE_FAIL", 1))
    preflight_two = engine.reconcile_stage(plan, "preflight").retry_submission
    assert preflight_two is not None
    backend.set_states(preflight_two.job_id, SlurmTaskState(None, "COMPLETED", 0))
    backend.put_receipt(_receipt("done", stage="preflight"))
    assert engine.reconcile_stage(plan, "preflight").decision == "COMPLETED"

    compute_two = read_plan_submission_records(backend, plan, compute)[-1]
    assert compute_two.identity.attempt == 2
    assert compute_two.dependency_job_ids == (preflight_two.job_id,)
    complete_compute(compute_two)

    # Model the window: compute's completion is durable before audit's
    # replacement authorization can be published.
    engine._ensure_reconciler_followups = lambda *_args, **_kwargs: ()  # type: ignore[method-assign]
    assert engine.reconcile_stage(plan, "compute").decision == "COMPLETED"

    with pytest.raises(ValueError, match="superseded dependency generations: compute"):
        require_current_terminal_evidence(backend, plan, "promote")

    # Once the owed rerun lands against the current generation, promote clears.
    replacement = ExecutionEngine(backend)._ensure_reconciler_followups(plan, "compute", compute_two)
    assert [record.identity.attempt for record in replacement] == [2]
    audit_two = replacement[0]
    assert audit_two.dependency_job_ids == (compute_two.job_id,)
    backend.set_states(audit_two.job_id, SlurmTaskState(None, "COMPLETED", 0))
    backend.put_receipt(_receipt("done", stage="audit"))
    assert ExecutionEngine(backend).reconcile_stage(plan, "audit").decision == "COMPLETED"
    require_current_terminal_evidence(backend, plan, "promote")


def test_rejected_ordinary_attempt_does_not_inherit_a_losing_followup(
    tmp_path,
) -> None:
    """A losing authorization is never the newest dependency authority.

    An ordinary missing-only attempt can win an attempt's intent and still be
    definitively rejected by Slurm.  The next rejection retry carries that
    preserved subset, so adopting the losing authorization's dependencies would
    bind successes produced against the old prerequisite to a replacement one.
    """

    class RejectComputeRetry(ConcurrentBackend):
        """Land the authorization at intent time, then refuse the winner."""

        pending_authorization: tuple[str, str] | None = None

        def write_immutable_text(self, path: str, text: str) -> bool:
            if self.pending_authorization is not None and "/submission-intents/compute/" in path:
                target, payload = self.pending_authorization
                self.pending_authorization = None
                super().write_immutable_text(target, payload)
            return super().write_immutable_text(path, text)

        def invoke_submission(self, request: SubmissionRequest) -> SubmitOutcome:
            if request.identity.stage_id == "compute" and request.identity.attempt == 2:
                return SubmitOutcome(DEFINITELY_REJECTED, returncode=1, stderr="compute qos rejected")
            return super().invoke_submission(request)

    plan = _dependent_plan()
    backend = RejectComputeRetry()
    engine = ExecutionEngine(backend)

    preflight_one = engine.submit_stage(plan, "preflight").record
    compute_one = engine.submit_stage(plan, "compute").record
    backend.set_states(
        compute_one.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))
    assert engine.reconcile_stage(plan, "compute", submit_retry=False).decision == "RETRY_MISSING_ONLY"

    engine._ensure_reconciler_followups = lambda *_args, **_kwargs: ()  # type: ignore[method-assign]
    backend.set_states(preflight_one.job_id, SlurmTaskState(None, "NODE_FAIL", 1))
    preflight_two = engine.reconcile_stage(plan, "preflight").retry_submission
    assert preflight_two is not None
    backend.pending_authorization = (
        reconciler_followup_path(plan, "compute", 2),
        canonical_json(
            {
                "attempt": 2,
                "dependency_job_ids": [preflight_two.job_id],
                "plan_sha256": plan.plan_sha256,
                "schema_version": 1,
                "stage_id": "compute",
                "trigger_job_id": preflight_two.job_id,
                "trigger_stage_id": "preflight",
            }
        ),
    )

    # The ordinary retry owns attempt two's intent and is then rejected.
    with pytest.raises(SubmissionRejected, match="compute qos rejected"):
        ExecutionEngine(backend).submit_stage(plan, "compute", attempt=2, task_ids=("movie-1",))

    third = ExecutionEngine(backend).submit_stage(plan, "compute", attempt=3, task_ids=("movie-1",)).record
    assert third.task_ids == ("movie-1",)
    assert third.dependency_job_ids == (preflight_one.job_id,)
    assert third.dependency_job_ids != (preflight_two.job_id,)


def test_receipt_probe_refuses_symlinks_at_any_depth(tmp_path) -> None:
    """Only a real regular file below the run root may certify a receipt.

    ``lstat`` declines to follow just the final name, so a task that replaces an
    intermediate directory with a link to an external tree would still have its
    target hashed.  Promotion and archive preserve the link rather than those
    bytes, so the run would be released as certified against data that can
    change or disappear afterwards.
    """

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "result.json").write_text("planted\n")

    root = tmp_path / "run"
    (root / "receipts" / "task").mkdir(parents=True)
    (root / "receipts" / "task" / "result.json").write_text("ok\n")
    (root / "receipts" / "linked.json").symlink_to(outside / "result.json")
    (root / "swapped").mkdir()
    (root / "swapped" / "task").symlink_to(outside)

    def probe(relative: str) -> dict:
        result = subprocess.run(
            [sys.executable, "-c", RECEIPT_PROBE_PROGRAM, str(root), relative],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    certified = probe("receipts/task/result.json")
    assert certified["exists"] is True
    assert certified["sha256"] == hashlib.sha256(b"ok\n").hexdigest()

    refused = (
        "swapped/task/result.json",  # symlinked ancestor directory
        "receipts/linked.json",  # symlinked leaf
        "receipts/task",  # a directory is not receipt bytes
        "receipts/task/absent.json",  # nothing there
        "receipts/../../escape.json",  # traversal out of the run root
    )
    for relative in refused:
        assert probe(relative) == {"exists": False, "sha256": None}, relative


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


def test_stale_unsubmitted_followup_is_superseded_instead_of_replayed() -> None:
    """A superseded authorization must not wedge every later reconciliation.

    With two dependencies, a coordinator can die after publishing one parent's
    follow-up authorization but before its submission intent.  A retry of the
    other parent then allocates a newer generation bound to both current jobs.
    Replaying the first authorization would bind its stale dependency tuple,
    which submission rejects against the now-current jobs, and reconciliation
    invokes propagation before publishing terminal receipts -- so the stage
    could never converge or archive again.
    """

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

    # A retries, and the coordinator dies after publishing audit's authorization
    # but before its submission intent exists.
    engine._ensure_reconciler_followups = lambda *_args, **_kwargs: ()  # type: ignore[method-assign]
    make_retryable(first_a.job_id, "a")
    retry_a = engine.reconcile_stage(plan, "compute-a").retry_submission
    assert retry_a is not None
    backend.write_immutable_text(
        reconciler_followup_path(plan, "audit", 2),
        canonical_json(
            {
                "attempt": 2,
                "dependency_job_ids": [retry_a.job_id, first_b.job_id],
                "plan_sha256": plan.plan_sha256,
                "schema_version": 1,
                "stage_id": "audit",
                "trigger_job_id": retry_a.job_id,
                "trigger_stage_id": "compute-a",
            }
        ),
    )

    # B retries and allocates a newer generation bound to both current jobs.
    make_retryable(first_b.job_id, "b")
    retry_b = ExecutionEngine(backend).reconcile_stage(plan, "compute-b").retry_submission
    assert retry_b is not None

    # Replaying A's propagation must supersede the stale authorization rather
    # than resubmit it against dependency jobs that have since moved.  B's
    # generation already binds both current jobs, so it satisfies A's trigger
    # too and no further identity may be spent.
    replayed = ExecutionEngine(backend)._ensure_reconciler_followups(plan, "compute-a", retry_a)
    assert replayed == ()
    audit_records = read_plan_submission_records(backend, plan, stage_by_id(plan, "audit"))
    assert [record.identity.attempt for record in audit_records] == [1, 3]
    assert audit_records[-1].dependency_job_ids == (retry_a.job_id, retry_b.job_id)


def test_superseded_replay_does_not_spend_a_signed_attempt(tmp_path) -> None:
    """Supersession must reuse a covering generation, not consume a new one.

    A reconciler's derived bound only funds one replacement per dependency
    retry.  If replaying a superseded trigger allocated an extra identity even
    though a submitted generation already binds every current dependency job,
    a stage with a tight budget would exhaust that bound and could never
    publish its terminal receipt.
    """

    compute_a = _compute_stage(stage_id="compute-a", task_prefix="a")
    compute_b = _compute_stage(stage_id="compute-b", task_prefix="b")
    audit = StageSpec(
        stage_id="audit",
        command=_command("audit"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="audit"),),
        depends_on=("compute-a", "compute-b"),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(max_attempts=1),
    )
    plan = _plan(stages=(compute_a, compute_b, audit))
    assert signed_attempt_bound(plan, stage_by_id(plan, "audit")) == 3

    backend = ConcurrentBackend()
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

    # A retries and the coordinator dies before submitting audit's generation.
    engine._ensure_reconciler_followups = lambda *_args, **_kwargs: ()  # type: ignore[method-assign]
    make_retryable(first_a.job_id, "a")
    retry_a = engine.reconcile_stage(plan, "compute-a").retry_submission
    assert retry_a is not None
    backend.write_immutable_text(
        reconciler_followup_path(plan, "audit", 2),
        canonical_json(
            {
                "attempt": 2,
                "dependency_job_ids": [retry_a.job_id, first_b.job_id],
                "plan_sha256": plan.plan_sha256,
                "schema_version": 1,
                "stage_id": "audit",
                "trigger_job_id": retry_a.job_id,
                "trigger_stage_id": "compute-a",
            }
        ),
    )

    # B retries and validly submits the generation bound to both current jobs.
    make_retryable(first_b.job_id, "b")
    retry_b = ExecutionEngine(backend).reconcile_stage(plan, "compute-b").retry_submission
    assert retry_b is not None

    # Replaying A must recognize that generation rather than exceed the bound.
    replayed = ExecutionEngine(backend)._ensure_reconciler_followups(plan, "compute-a", retry_a)
    assert replayed == ()
    audit_records = read_plan_submission_records(backend, plan, stage_by_id(plan, "audit"))
    assert [record.identity.attempt for record in audit_records] == [1, 3]
    assert audit_records[-1].dependency_job_ids == (retry_a.job_id, retry_b.job_id)


def test_superseded_replay_reuses_an_authorized_but_unsubmitted_generation() -> None:
    """A covering generation counts before it is submitted, not only after.

    A crash between writing the second dependency's authorization and
    publishing its intent leaves a generation that already binds every current
    job but has no record.  Replaying the first trigger must carry that
    generation rather than allocate another identity the bound cannot fund.
    """

    compute_a = _compute_stage(stage_id="compute-a", task_prefix="a")
    compute_b = _compute_stage(stage_id="compute-b", task_prefix="b")
    audit = StageSpec(
        stage_id="audit",
        command=_command("audit"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="audit"),),
        depends_on=("compute-a", "compute-b"),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(max_attempts=1),
    )
    plan = _plan(stages=(compute_a, compute_b, audit))
    assert signed_attempt_bound(plan, stage_by_id(plan, "audit")) == 3

    backend = ConcurrentBackend()
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

    def authorize(attempt: int, dependency_jobs: list, trigger_job: str, trigger_stage: str) -> None:
        """Publish one follow-up authorization without submitting it."""

        backend.write_immutable_text(
            reconciler_followup_path(plan, "audit", attempt),
            canonical_json(
                {
                    "attempt": attempt,
                    "dependency_job_ids": dependency_jobs,
                    "plan_sha256": plan.plan_sha256,
                    "schema_version": 1,
                    "stage_id": "audit",
                    "trigger_job_id": trigger_job,
                    "trigger_stage_id": trigger_stage,
                }
            ),
        )

    engine._ensure_reconciler_followups = lambda *_args, **_kwargs: ()  # type: ignore[method-assign]
    make_retryable(first_a.job_id, "a")
    retry_a = engine.reconcile_stage(plan, "compute-a").retry_submission
    assert retry_a is not None
    authorize(2, [retry_a.job_id, first_b.job_id], retry_a.job_id, "compute-a")

    make_retryable(first_b.job_id, "b")
    retry_b = engine.reconcile_stage(plan, "compute-b").retry_submission
    assert retry_b is not None
    # The coordinator dies here: generation three is authorized but never sent.
    authorize(3, [retry_a.job_id, retry_b.job_id], retry_b.job_id, "compute-b")

    replayed = ExecutionEngine(backend)._ensure_reconciler_followups(plan, "compute-a", retry_a)
    assert [record.identity.attempt for record in replayed] == [3]
    assert replayed[0].dependency_job_ids == (retry_a.job_id, retry_b.job_id)
    assert replayed[0].task_ids == ("audit",)


def test_afterok_certification_binds_to_the_resolved_dependency_job() -> None:
    """Stage-level completion is not enough to release an afterok child.

    A prerequisite can accept a replacement job that has not reconciled yet.
    Depending on that job would let Slurm release this stage on a zero exit
    while its required receipts are absent -- exactly what the gate exists to
    prevent -- so certification is bound to the resolved job, not the stage.
    """

    root = StageSpec(
        stage_id="root",
        command=_command("root"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="root"),),
        retry_policy=RetryPolicy(max_attempts=2, retryable_slurm_states=("NODE_FAIL",)),
    )
    preflight = StageSpec(
        stage_id="preflight",
        command=_command("preflight"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="preflight"),),
        depends_on=("root",),
        dependency_mode="afterany",
        retry_policy=RetryPolicy(max_attempts=2, retryable_slurm_states=("NODE_FAIL",)),
    )
    audit = StageSpec(
        stage_id="audit",
        command=_command("audit"),
        resources=_resources(1),
        expected_receipts=(_receipt("done", stage="audit"),),
        depends_on=("preflight",),
        dependency_mode="afterok",
        retry_policy=RetryPolicy(max_attempts=2),
    )
    plan = _plan(stages=(root, preflight, audit))
    backend = ConcurrentBackend()
    engine = ExecutionEngine(backend)

    root_one = engine.submit_stage(plan, "root").record
    preflight_one = engine.submit_stage(plan, "preflight").record
    backend.set_states(preflight_one.job_id, SlurmTaskState(None, "COMPLETED", 0))
    backend.put_receipt(_receipt("done", stage="preflight"))
    assert engine.reconcile_stage(plan, "preflight").decision == "COMPLETED"

    # The certified generation may release the child.
    engine._validate_afterok_dependency_jobs(plan, stage_by_id(plan, "audit"), (preflight_one.job_id,))

    # And the check is on the submission path, receiving the resolved tuple
    # rather than only the stage, so a replacement accepted after the early
    # gate cannot slip into the published intent.
    checked: list[tuple[str, tuple[str, ...]]] = []

    class SpyEngine(ExecutionEngine):
        """Record the exact dependency tuple certification received."""

        def _validate_afterok_dependency_jobs(self, plan, stage, dependency_jobs):
            checked.append((stage.stage_id, dependency_jobs))
            return super()._validate_afterok_dependency_jobs(plan, stage, dependency_jobs)

    SpyEngine(backend).submit_stage(plan, "audit")
    assert ("audit", (preflight_one.job_id,)) in checked

    # The prerequisite accepts a replacement that has not reconciled.
    backend.set_states(root_one.job_id, SlurmTaskState(None, "NODE_FAIL", 1))
    assert engine.reconcile_stage(plan, "root").retry_submission is not None
    preflight_two = read_plan_submission_records(backend, plan, stage_by_id(plan, "preflight"))[-1]
    assert preflight_two.identity.attempt == 2

    with pytest.raises(ValueError, match="without authenticated COMPLETED reconciliation"):
        engine._validate_afterok_dependency_jobs(
            plan,
            stage_by_id(plan, "audit"),
            (preflight_two.job_id,),
        )


def test_lost_arbitration_reuses_a_covering_authorization() -> None:
    """Losing an attempt must not allocate a generation that already exists.

    An authorization can lose its attempt to an ordinary retry while another
    dependency has already authorized a generation covering the current
    dependency tuple.  Allocating a further identity would leave two full-task
    jobs racing over the same outputs once the other trigger is replayed.
    """

    class PublishFollowupAtIntent(ConcurrentBackend):
        """Land an authorization exactly as the ordinary retry writes intent."""

        pending_authorization: tuple[str, str] | None = None

        def write_immutable_text(self, path: str, text: str) -> bool:
            if self.pending_authorization is not None and "/submission-intents/compute/" in path:
                target, payload = self.pending_authorization
                self.pending_authorization = None
                super().write_immutable_text(target, payload)
            return super().write_immutable_text(path, text)

    compute_a = _compute_stage(stage_id="compute-a", task_prefix="a")
    compute_b = _compute_stage(stage_id="compute-b", task_prefix="b")
    base = _compute_stage()
    compute = StageSpec(
        stage_id=base.stage_id,
        resources=base.resources,
        expected_receipts=base.expected_receipts,
        tasks=base.tasks,
        depends_on=("compute-a", "compute-b"),
        dependency_mode="afterany",
        retry_policy=base.retry_policy,
    )
    plan = _plan(stages=(compute_a, compute_b, compute))
    backend = PublishFollowupAtIntent()
    engine = ExecutionEngine(backend)

    first_a = engine.submit_stage(plan, "compute-a").record
    first_b = engine.submit_stage(plan, "compute-b").record
    compute_one = engine.submit_stage(plan, "compute").record
    backend.set_states(
        compute_one.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))
    assert engine.reconcile_stage(plan, "compute", submit_retry=False).decision == "RETRY_MISSING_ONLY"

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

    def authorization(attempt: int, jobs: list, trigger_job: str, trigger_stage: str) -> tuple[str, str]:
        """Return the exact authorization bytes propagation would publish."""

        return (
            reconciler_followup_path(plan, "compute", attempt),
            canonical_json(
                {
                    "attempt": attempt,
                    "dependency_job_ids": jobs,
                    "plan_sha256": plan.plan_sha256,
                    "schema_version": 1,
                    "stage_id": "compute",
                    "trigger_job_id": trigger_job,
                    "trigger_stage_id": trigger_stage,
                }
            ),
        )

    engine._ensure_reconciler_followups = lambda *_args, **_kwargs: ()  # type: ignore[method-assign]
    make_retryable(first_a.job_id, "a")
    retry_a = engine.reconcile_stage(plan, "compute-a").retry_submission
    assert retry_a is not None

    # A's authorization loses attempt two to compute's own ordinary retry.
    backend.pending_authorization = authorization(2, [retry_a.job_id, first_b.job_id], retry_a.job_id, "compute-a")
    winner = ExecutionEngine(backend).submit_stage(plan, "compute", attempt=2, task_ids=("movie-1",)).record
    assert winner.dependency_job_ids == (first_a.job_id, first_b.job_id)

    # B then authorizes the covering generation but dies before submitting it.
    make_retryable(first_b.job_id, "b")
    retry_b = engine.reconcile_stage(plan, "compute-b").retry_submission
    assert retry_b is not None
    target, payload = authorization(3, [retry_a.job_id, retry_b.job_id], retry_b.job_id, "compute-b")
    backend.write_immutable_text(target, payload)
    assert not [
        record
        for record in read_plan_submission_records(backend, plan, stage_by_id(plan, "compute"))
        if record.identity.attempt == 3
    ]

    replayed = ExecutionEngine(backend)._ensure_reconciler_followups(plan, "compute-a", retry_a)
    assert [record.identity.attempt for record in replayed] == [3]
    assert replayed[0].dependency_job_ids == (retry_a.job_id, retry_b.job_id)


def test_replacement_generation_is_ordered_after_the_prior_one() -> None:
    """Two generations of one stage must never be able to write concurrently.

    A replacement runs the same signed commands in the same working directory
    and writes the same receipt and output paths.  Slurm ANDs comma-separated
    dependency clauses, so the earlier generation gates the replacement without
    entering the signed DAG tuple whose cardinality is authenticated elsewhere.
    """

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

    compute_one = engine.submit_stage(plan, "compute").record
    audit_one = engine.submit_afterany_reconciler(plan, "audit").record

    def audit_request(attempt: int) -> SubmissionRequest:
        """Return the exact scheduler request one audit attempt presented."""

        return next(
            request
            for request in backend.requests
            if request.identity.stage_id == "audit" and request.identity.attempt == attempt
        )

    # The first generation has nothing to wait behind.
    assert audit_request(1).ordering_job_ids == ()
    assert f"--dependency=afterany:{compute_one.job_id}" in audit_request(1).sbatch_args()

    backend.set_states(
        compute_one.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))
    compute_two = engine.reconcile_stage(plan, "compute").retry_submission
    assert compute_two is not None

    replacement = audit_request(2)
    assert replacement.dependency_job_ids == (compute_two.job_id,)
    assert replacement.ordering_job_ids == (audit_one.job_id,)
    # Explicit ordering covers the generation this coordinator can see, and
    # ``singleton`` covers any it cannot: an attempt between sbatch and its
    # record is invisible to every check a coordinator could perform, so the
    # scheduler enforces the exclusion instead.
    assert (
        f"--dependency=afterany:{compute_two.job_id},afterany:{audit_one.job_id},singleton" in replacement.sbatch_args()
    )

    def job_names(request: SubmissionRequest) -> list:
        """Return the rendered job-name options for one request."""

        return [argument for argument in request.sbatch_args() if argument.startswith("--job-name=")]

    # The name is stage-scoped, so singleton spans every generation of it.
    assert job_names(replacement) == [f"--job-name=o2p-{plan.plan_sha256[:10]}-audit"]
    assert job_names(audit_request(1)) == job_names(replacement)
    # A first generation has nothing to serialize against.
    assert not [argument for argument in audit_request(1).sbatch_args() if argument.endswith(",singleton")]


def test_never_launched_afterany_child_keeps_its_ordinary_first_attempt() -> None:
    """Propagation must not authorize attempt one for a child that never ran.

    A dependency follow-up may never occupy attempt one, so allocating one for a
    child with no prior submission raises on the parent's reconciliation and on
    every replay, even though that parent retry is already durably accepted.
    The child's ordinary attempt one resolves the latest dependency job itself.
    """

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

    # The audit is deliberately never submitted before the parent retries.
    compute_one = engine.submit_stage(plan, "compute").record
    backend.set_states(
        compute_one.job_id,
        SlurmTaskState(None, "NODE_FAIL", 1),
        SlurmTaskState(0, "COMPLETED", 0),
        SlurmTaskState(2, "COMPLETED", 0),
    )
    backend.put_receipt(_receipt("movie-0"))
    backend.put_receipt(_receipt("movie-2"))

    compute_two = engine.reconcile_stage(plan, "compute").retry_submission
    assert compute_two is not None
    assert backend.read_text(reconciler_followup_path(plan, "audit", 1)) is None
    assert not read_plan_submission_records(backend, plan, stage_by_id(plan, "audit"))

    # Replay stays convergent rather than raising on an immutable attempt one.
    ExecutionEngine(backend)._ensure_reconciler_followups(plan, "compute", compute_two)

    ordinary = ExecutionEngine(backend).submit_afterany_reconciler(plan, "audit").record
    assert ordinary.identity.attempt == 1
    assert ordinary.dependency_job_ids == (compute_two.job_id,)


def test_allocation_waits_for_an_earlier_attempt_to_publish_its_record() -> None:
    """The ordering fence is built from records, so allocation must wait for them.

    An attempt that crossed ``sbatch`` but has not published its record yet is
    invisible to the fence.  Allocating the next generation past that window
    would omit a live job from its dependency list, leaving two replacements
    running the same signed commands against the same output and receipt paths.
    """

    class UncertainSecondAudit(ConcurrentBackend):
        """Lose audit attempt two's outcome after it owns its invocation.

        No job is materialized, so scheduler lookup cannot resolve the call and
        the attempt stays invocation-owned with no record -- the exact window
        the ordering fence cannot see.
        """

        def invoke_submission(self, request: SubmissionRequest) -> SubmitOutcome:
            if request.identity.stage_id == "audit" and request.identity.attempt == 2:
                with self._lock:
                    self.requests.append(request)
                return SubmitOutcome(INVOKED_OUTCOME_UNKNOWN, returncode=0)
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
    backend = UncertainSecondAudit()
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

    # A's retry launches audit attempt two, which owns its invocation but never
    # publishes a record.
    make_retryable(first_a.job_id, "a")
    with pytest.raises(SubmissionUncertain):
        engine.reconcile_stage(plan, "compute-a")
    audit_records = read_plan_submission_records(backend, plan, stage_by_id(plan, "audit"))
    assert [record.identity.attempt for record in audit_records] == [1]
    assert backend.read_text(reconciler_followup_path(plan, "audit", 2)) is not None

    # B's retry must not allocate past that window.
    make_retryable(first_b.job_id, "b")
    retry_b = ExecutionEngine(backend).reconcile_stage(plan, "compute-b").retry_submission
    assert retry_b is not None
    assert backend.read_text(reconciler_followup_path(plan, "audit", 3)) is None
    assert not [
        request
        for request in backend.requests
        if request.identity.stage_id == "audit" and request.identity.attempt == 3
    ]


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


def test_a_newer_generation_recovers_a_failed_run_without_reviving_stale_ones() -> None:
    """Sticky failure must not outlive the generation that caused it.

    An ``afterany`` descendant can fail against one dependency generation and
    then succeed against an authenticated replacement.  Pinning the plan-level
    state to that superseded failure would block promotion of a recovered run
    forever, while a genuinely stale callback must still be unable to resurrect
    the failure.
    """

    plan = _plan()
    manifest = RunManifest(
        run_id=RUN_ID,
        campaign=CAMPAIGN,
        pipeline="canary",
        created_utc="20260822T010203Z",
        datasets=["dataset-1"],
    )
    failed = RegistryUpdate(plan.plan_sha256, "audit", "FAILED", "FAILED", ("9000",), 1)
    recovered = RegistryUpdate(plan.plan_sha256, "audit", "COMPLETED", "COMPLETED", ("9000", "9002"), 2)
    stale_failure = RegistryUpdate(plan.plan_sha256, "audit", "FAILED", "FAILED", ("9000",), 1)

    merged = merge_execution_manifest(manifest, plan, failed)
    assert merged.provenance["execution"]["state"] == "FAILED"
    assert merged.result["status"] == "FAILED"

    merged = merge_execution_manifest(merged, plan, recovered)
    assert merged.provenance["execution"]["stages"]["audit"] == {"attempt": 2, "status": "COMPLETED"}
    assert merged.provenance["execution"]["state"] == "COMPLETED"
    assert merged.result["status"] == "COMPLETED"

    # The superseded callback is still refused, so recovery is not a hole.
    merged = merge_execution_manifest(merged, plan, stale_failure)
    assert merged.provenance["execution"]["stages"]["audit"] == {"attempt": 2, "status": "COMPLETED"}
    assert merged.provenance["execution"]["state"] == "COMPLETED"
    assert merged.result["status"] == "COMPLETED"


def test_a_replacement_generation_reopens_a_completed_run() -> None:
    """A completed run must not keep advertising completion while work runs.

    Sticky completion exists to stop a delayed callback from reporting a
    terminal run as merely active.  An accepted higher attempt is the opposite:
    it is a replacement generation actually starting, and the registry and
    status consumers have to see that.
    """

    plan = _plan()
    manifest = RunManifest(
        run_id=RUN_ID,
        campaign=CAMPAIGN,
        pipeline="canary",
        created_utc="20260822T010203Z",
        datasets=["dataset-1"],
    )
    completed = RegistryUpdate(plan.plan_sha256, "compute", "COMPLETED", "COMPLETED", ("9000",), 1)
    replacement = RegistryUpdate(plan.plan_sha256, "compute", "RETRYING", "RETRYING", ("9000", "9002"), 2)
    stale_completion = RegistryUpdate(plan.plan_sha256, "compute", "COMPLETED", "COMPLETED", ("9000",), 1)

    merged = merge_execution_manifest(manifest, plan, completed)
    assert merged.provenance["execution"]["state"] == "COMPLETED"

    merged = merge_execution_manifest(merged, plan, replacement)
    assert merged.provenance["execution"]["stages"]["compute"] == {"attempt": 2, "status": "RETRYING"}
    assert merged.provenance["execution"]["state"] == "RETRYING"
    assert merged.result["status"] == "RETRYING"

    # The superseded callback still cannot move the run back to completed.
    merged = merge_execution_manifest(merged, plan, stale_completion)
    assert merged.provenance["execution"]["state"] == "RETRYING"
    assert merged.result["status"] == "RETRYING"


def test_one_recovered_stage_does_not_clear_another_stages_failure() -> None:
    """Recovery is per generation, not a blanket reset of the run."""

    plan = _plan()
    manifest = RunManifest(
        run_id=RUN_ID,
        campaign=CAMPAIGN,
        pipeline="canary",
        created_utc="20260822T010203Z",
        datasets=["dataset-1"],
    )
    compute_failed = RegistryUpdate(plan.plan_sha256, "compute", "FAILED", "FAILED", ("9000",), 1)
    audit_recovered = RegistryUpdate(plan.plan_sha256, "audit", "COMPLETED", "COMPLETED", ("9002",), 2)

    merged = merge_execution_manifest(manifest, plan, compute_failed)
    merged = merge_execution_manifest(merged, plan, audit_recovered)
    assert merged.provenance["execution"]["stages"]["compute"] == {"attempt": 1, "status": "FAILED"}
    assert merged.provenance["execution"]["state"] == "FAILED"
    assert merged.result["status"] == "FAILED"


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
