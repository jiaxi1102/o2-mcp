"""Adversarial regressions for authenticated execution state hardening."""

from __future__ import annotations

import hashlib
import json
import posixpath
import subprocess
from dataclasses import dataclass

import pytest

from o2mcp.runorg import (
    CanonicalPaths,
    CommandSpec,
    DatasetIdentity,
    ExecutionEngine,
    ExecutionPlan,
    ReceiptObservation,
    ReceiptSpec,
    RegistryUpdate,
    ResourceSpec,
    RetryPolicy,
    SlurmJob,
    SlurmTaskState,
    StageSpec,
    SubmissionIdentity,
    SubmissionRecord,
    SubmitOutcome,
)
from o2mcp.runorg.execution_models import ACCEPTED, PlannedTask, SubmissionRequest, TaskAttemptReceipt, canonical_json
from o2mcp.runorg.execution_paths import pending_registry_path, submission_record_path, task_attempt_path
from o2mcp.runorg.execution_rendering import render_dispatcher
from o2mcp.runorg.registry_outbox import merge_registry_updates

RUN_ID = "RUN_20260822T010203Z_hardening__canary"
RUN_ROOT = f"/n/scratch/users/test/runs/hardening/{RUN_ID}"


class Backend:
    """Small lock-free fake; each test drives it from one coordinator thread."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.jobs: dict[str, dict[str, object]] = {}
        self.requests: list[SubmissionRequest] = []
        self.next_job = 7100

    def find_jobs(self, comment: str):
        return tuple(
            SlurmJob(job_id, comment, str(value["state"]))
            for job_id, value in self.jobs.items()
            if value["comment"] == comment
        )

    def prepare_submission(self, request: SubmissionRequest) -> None:
        """The fake has no remote dispatcher preparation step."""

        return None

    def invoke_submission(self, request: SubmissionRequest) -> SubmitOutcome:
        job_id = str(self.next_job)
        self.next_job += 1
        self.requests.append(request)
        self.jobs[job_id] = {
            "comment": request.comment,
            "state": "PENDING",
            "task_states": (SlurmTaskState(None, "PENDING", None),),
        }
        return SubmitOutcome(ACCEPTED, job_id=job_id, returncode=0)

    def task_states(self, job_id: str):
        return self.jobs[job_id]["task_states"]

    def observe_receipt(self, run_root: str, receipt: ReceiptSpec) -> ReceiptObservation:
        text = self.files.get(posixpath.join(run_root, receipt.path))
        digest = hashlib.sha256(text.encode()).hexdigest() if text is not None else None
        return ReceiptObservation(receipt.path, text is not None, digest)

    def read_text(self, path: str) -> str | None:
        return self.files.get(path)

    def write_immutable_text(self, path: str, text: str) -> bool:
        if path in self.files:
            if self.files[path] != text:
                raise RuntimeError(f"immutable conflict: {path}")
            return False
        self.files[path] = text
        return True

    def write_mutable_text(self, path: str, text: str) -> None:
        self.files[path] = text

    def compare_and_swap_text(self, path: str, expected: str | None, replacement: str | None) -> bool:
        current = self.files.get(path)
        if current != expected or (path not in self.files and expected is not None):
            return False
        if replacement is None:
            self.files.pop(path, None)
        else:
            self.files[path] = replacement
        return True


@dataclass
class Registry:
    """Authenticated registry whose writes can be held for outbox assertions."""

    accept: bool = True

    def __post_init__(self) -> None:
        self.updates: list[RegistryUpdate] = []

    def validate_plan(self, _plan: ExecutionPlan) -> bool:
        return True

    def synchronize(self, _plan: ExecutionPlan, update: RegistryUpdate) -> bool:
        if self.accept:
            self.updates.append(update)
        return self.accept


def _command(name: str) -> CommandSpec:
    return CommandSpec(
        argv=("/usr/bin/python3", "-m", "worker", name),
        working_directory=f"{RUN_ROOT}/work",
        runtime_fingerprint_sha256="e" * 64,
    )


def _stage(stage_id: str, *, depends_on: tuple[str, ...] = ()) -> StageSpec:
    receipt = ReceiptSpec(f"receipt-{stage_id}", f"receipts/stages/{stage_id}/done.json")
    return StageSpec(
        stage_id=stage_id,
        command=_command(stage_id),
        resources=ResourceSpec("short", 1, 1024, "00:05:00", 1),
        expected_receipts=(receipt,),
        depends_on=depends_on,
        dependency_mode="afterany" if depends_on else "afterok",
        retry_policy=RetryPolicy(max_attempts=2, retryable_slurm_states=("NODE_FAIL",)),
    )


def _plan(*, stages: tuple[StageSpec, ...] | None = None) -> ExecutionPlan:
    return ExecutionPlan(
        project="hardening",
        campaign="hardening",
        pipeline="canary",
        run_id=RUN_ID,
        source_commit="a" * 40,
        source_bundle_sha256="b" * 64,
        datasets=(DatasetIdentity("dataset", "c" * 64),),
        paths=CanonicalPaths(
            RUN_ROOT,
            f"{RUN_ROOT}/work",
            "/n/groups/lab/results/hardening/dataset",
            f"{RUN_ROOT}/receipts",
            f"{RUN_ROOT}/logs",
        ),
        stages=stages or (_stage("compute"),),
    )


def test_public_submission_cannot_override_signed_dependencies() -> None:
    """A caller cannot submit a dependent stage immediately with an empty list."""

    plan = _plan(stages=(_stage("compute"), _stage("audit", depends_on=("compute",))))
    backend = Backend()
    engine = ExecutionEngine(backend)
    compute = engine.submit_stage(plan, "compute").record
    with pytest.raises(TypeError, match="dependency_job_ids"):
        engine.submit_stage(plan, "audit", dependency_job_ids=())  # type: ignore[call-arg]
    audit = engine.submit_stage(plan, "audit").record
    assert audit.dependency_job_ids == (compute.job_id,)
    assert len(backend.requests) == 2


def test_foreign_submission_record_at_expected_path_is_rejected() -> None:
    """Valid JSON for another stage cannot masquerade as the expected attempt."""

    plan = _plan()
    backend = Backend()
    expected = SubmissionIdentity(plan.plan_sha256, "compute", 1)
    foreign = SubmissionRecord(
        SubmissionIdentity(plan.plan_sha256, "foreign", 1),
        "9999",
        ("foreign",),
        (),
        True,
        "afterany",
        (),
    )
    backend.files[submission_record_path(plan, expected)] = canonical_json(foreign.to_dict())
    with pytest.raises(ValueError, match="invalid immutable submission record"):
        ExecutionEngine(backend).submit_stage(plan, "compute")
    assert not backend.requests


def test_contradictory_task_receipt_cannot_certify_success() -> None:
    """Serialized verdict booleans never override raw scheduler/receipt evidence."""

    plan = _plan()
    backend = Backend()
    engine = ExecutionEngine(backend)
    record = engine.submit_stage(plan, "compute").record
    backend.jobs[record.job_id]["task_states"] = (SlurmTaskState(None, "FAILED", 1),)
    bad = TaskAttemptReceipt(
        record.identity,
        "compute",
        None,
        record.job_id,
        "FAILED",
        1,
        (ReceiptObservation("receipts/stages/compute/done.json", False, None),),
        True,
        False,
    )
    backend.files[task_attempt_path(plan, record.identity, "compute")] = canonical_json(bad.to_dict())
    with pytest.raises(ValueError, match="verdict contradicts"):
        engine.reconcile_stage(plan, "compute")


def test_reconcile_authenticates_active_run_before_writing() -> None:
    """A stale/unregistered run cannot gain reconciliation evidence."""

    class RejectingRegistry(Registry):
        def validate_plan(self, _plan: ExecutionPlan) -> bool:
            return False

    plan = _plan()
    backend = Backend()
    record = ExecutionEngine(backend).submit_stage(plan, "compute").record
    backend.jobs[record.job_id]["task_states"] = (SlurmTaskState(None, "FAILED", 1),)
    before = dict(backend.files)
    with pytest.raises(ValueError, match="registered active run"):
        ExecutionEngine(backend, RejectingRegistry()).reconcile_stage(plan, "compute")
    assert backend.files == before


def test_registry_outbox_is_per_attempt_monotonic_and_independently_cleared() -> None:
    """Delayed/cross-stage callbacks cannot overwrite or erase one another."""

    plan = _plan(stages=(_stage("compute"), _stage("audit", depends_on=("compute",))))
    backend = Backend()
    registry = Registry(accept=False)
    engine = ExecutionEngine(backend, registry)
    compute = engine.submit_stage(plan, "compute").record
    engine.submit_stage(plan, "audit")
    completed = RegistryUpdate(plan.plan_sha256, "compute", "COMPLETED", "COMPLETED", (compute.job_id,), 1)
    delayed = RegistryUpdate(plan.plan_sha256, "compute", "SUBMITTED", "SUBMITTED", (compute.job_id,), 1)
    engine._sync_registry(plan, completed)
    engine._sync_registry(plan, delayed)
    retrying = RegistryUpdate(plan.plan_sha256, "compute", "RETRYING", "RETRYING", (compute.job_id,), 2)
    engine._sync_registry(plan, retrying)
    compute_path = pending_registry_path(plan, "compute", 1)
    retry_path = pending_registry_path(plan, "compute", 2)
    audit_path = pending_registry_path(plan, "audit", 1)
    assert json.loads(backend.files[compute_path])["stage_status"] == "COMPLETED"
    assert json.loads(backend.files[retry_path])["attempt"] == 2
    assert audit_path in backend.files

    registry.accept = True
    assert engine._sync_registry(plan, completed) is True
    assert compute_path not in backend.files
    assert retry_path in backend.files
    assert audit_path in backend.files
    assert engine.reconcile_registry(plan) is True
    assert audit_path not in backend.files


def test_registry_outbox_job_union_is_byte_deterministic() -> None:
    """Equivalent joins must serialize job IDs in stable numeric order."""

    current = RegistryUpdate("a" * 64, "compute", "SUBMITTED", "SUBMITTED", ("10", "2"), 1)
    incoming = RegistryUpdate("a" * 64, "compute", "WAIT", "RUNNING", ("3", "2"), 1)

    merged = merge_registry_updates(current, incoming)

    assert merged.job_ids == ("2", "3", "10")


def test_verified_runtime_copy_is_immune_to_path_replacement(tmp_path) -> None:
    """Replacing argv[0] after hashing cannot change the bytes actually run."""

    worker = tmp_path / "worker.sh"
    output = tmp_path / "output.txt"
    worker.write_text('#!/bin/sh\nprintf original > "$1"\n')
    worker.chmod(0o755)
    digest = hashlib.sha256(worker.read_bytes()).hexdigest()
    command = CommandSpec(
        argv=(str(worker), str(output)),
        working_directory=str(tmp_path),
        runtime_fingerprint_path=str(worker),
        runtime_fingerprint_sha256=digest,
    )
    runtime_receipt = ReceiptSpec("runtime", "receipts/stages/runtime.json")
    script = render_dispatcher((PlannedTask("task", None, command, (runtime_receipt,)),))
    hasher = tmp_path / "hash-and-replace.py"
    hasher.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib, pathlib, sys\n"
        "p = pathlib.Path(sys.argv[-1])\n"
        "print(hashlib.sha256(p.read_bytes()).hexdigest(), p)\n"
        f"pathlib.Path({str(worker)!r}).write_text('#!/bin/sh\\nprintf tampered > \\\"$1\\\"\\n')\n"
    )
    hasher.chmod(0o755)
    script = script.replace("/usr/bin/sha256sum", str(hasher))
    result = subprocess.run(["/bin/bash"], input=script, text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert output.read_text() == "original"
