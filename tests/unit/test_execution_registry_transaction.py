"""Focused adversarial coverage for execution-registry CAS transactions."""

from __future__ import annotations

from types import SimpleNamespace

from o2mcp.runorg import (
    CanonicalPaths,
    CommandSpec,
    DatasetIdentity,
    ExecutionPlan,
    ReceiptSpec,
    ResourceSpec,
    StageSpec,
)
from o2mcp.runorg.execution_models import RegistryUpdate
from o2mcp.runorg.registry_sync import merge_execution_manifest, synchronize_execution_transaction
from o2mcp.runorg.runs import RunManifest

RUN_ID = "RUN_20260822T010203Z_adversarial-execution__race"
RUN_ROOT = f"/n/scratch/users/test/runs/adversarial-execution/{RUN_ID}"


def _registry_plan() -> ExecutionPlan:
    """Build the smallest immutable identity needed by registry merge logic."""

    return ExecutionPlan(
        project="execution-tests",
        campaign="adversarial-execution",
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
        stages=(
            StageSpec(
                stage_id="compute",
                command=CommandSpec(
                    argv=("/bin/true",),
                    working_directory=f"{RUN_ROOT}/work",
                    runtime_fingerprint_sha256="e" * 64,
                ),
                resources=ResourceSpec("short", 1, 1024, "00:05:00", 1),
                expected_receipts=(ReceiptSpec("done", "receipts/stages/compute/done.json"),),
            ),
        ),
    )


def test_registry_transaction_reports_compare_and_swap_conflict() -> None:
    """A stale reader gets a retryable conflict instead of clobbering run.json."""

    class ConflictConnection:
        """Represent another writer winning the remote registry lock."""

        def run(self, _command, *, timeout, input_text):
            """Return the transaction helper's dedicated CAS-conflict status."""

            return SimpleNamespace(ok=False, returncode=43, stdout="", stderr="")

    plan = _registry_plan()
    manifest = RunManifest(
        run_id=RUN_ID,
        campaign="adversarial-execution",
        pipeline="canary",
        created_utc="20260822T010203Z",
        datasets=["dataset-1"],
    )
    update = RegistryUpdate(plan.plan_sha256, "compute", "SUBMITTED", "SUBMITTED", ("9000",), 1)
    merged = merge_execution_manifest(manifest, plan, update)
    result = synchronize_execution_transaction(
        ConflictConnection(),
        run_dir=RUN_ROOT,
        registry_path="/n/groups/lab/registry.jsonl",
        current_manifest_text=manifest.to_json(),
        merged_manifest=merged,
    )
    assert result["ok"] is False
    assert result["error"] == "concurrent_update"
