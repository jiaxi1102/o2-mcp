"""Offline contract tests for immutable cross-pipeline execution plans."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from o2mcp.runorg import (
    CanonicalPaths,
    CommandSpec,
    DatasetIdentity,
    ExecutionPlan,
    ReceiptSpec,
    ResourceSpec,
    RetryPolicy,
    StageSpec,
    TaskSpec,
)
from o2mcp.runorg.runs import RunManifest

CAMPAIGN = "example-campaign"
RUN_ID = f"RUN_20260821T120000Z_{CAMPAIGN}__example-run-001"
RUN_ROOT = f"/n/scratch/users/test/runs/{CAMPAIGN}/{RUN_ID}"


def _receipt(stage: str, name: str = "completed") -> ReceiptSpec:
    """Return one canonical receipt below the test run's receipt tree."""

    return ReceiptSpec(
        receipt_id=f"{stage}-{name}",
        path=f"receipts/stages/{stage}/{name}.json",
    )


def _resources(*, parallelism: int = 1) -> ResourceSpec:
    """Return conservative resources shared by the compact test stages."""

    return ResourceSpec(
        partition="short",
        cpus=2,
        memory_mb=4096,
        time_limit="00:20:00",
        array_parallelism=parallelism,
    )


def _command(*argv: str) -> CommandSpec:
    """Return an exact command bound to the test run and runtime fingerprint."""

    return CommandSpec(
        argv=argv,
        working_directory=f"{RUN_ROOT}/work",
        runtime_fingerprint_sha256="e" * 64,
        environment=("LC_ALL=C",),
    )


def _plan(*, stages: tuple[StageSpec, ...] | None = None) -> ExecutionPlan:
    """Build a representative two-stage plan used by validation tests."""

    if stages is None:
        preflight = StageSpec(
            stage_id="preflight",
            command=_command("/usr/bin/python3", "-m", "example.preflight"),
            resources=_resources(),
            expected_receipts=(_receipt("preflight"),),
        )
        analyze = StageSpec(
            stage_id="analyze",
            resources=_resources(parallelism=2),
            expected_receipts=(_receipt("analyze"),),
            depends_on=("preflight",),
            dependency_mode="afterany",
            tasks=(
                TaskSpec(
                    task_id="movie-0002",
                    array_index=1,
                    command=_command("/usr/bin/python3", "-m", "example.analyze", "--task", "movie-0002"),
                    expected_receipts=(_receipt("analyze", "movie-0002"),),
                ),
                TaskSpec(
                    task_id="movie-0001",
                    array_index=0,
                    command=_command("/usr/bin/python3", "-m", "example.analyze", "--task", "movie-0001"),
                    expected_receipts=(_receipt("analyze", "movie-0001"),),
                ),
            ),
            retry_policy=RetryPolicy(
                max_attempts=2,
                retryable_slurm_states=("NODE_FAIL",),
                retry_missing_receipts=True,
                backoff_seconds=10,
            ),
        )
        # Deliberately reverse the input order.  DAG ordering and canonical JSON
        # must be deterministic rather than relying on renderer insertion order.
        stages = (analyze, preflight)

    return ExecutionPlan(
        project="example-analysis",
        campaign=CAMPAIGN,
        pipeline="example-pipeline",
        run_id=RUN_ID,
        source_commit="a" * 40,
        source_bundle_sha256="b" * 64,
        datasets=(
            DatasetIdentity(
                dataset_id="dataset-b",
                manifest_sha256="d" * 64,
            ),
            DatasetIdentity(
                dataset_id="dataset-a",
                manifest_sha256="c" * 64,
                storage_binding_sha256="f" * 64,
                source_uri="amdata://dataset-a",
            ),
        ),
        paths=CanonicalPaths(
            run_root=RUN_ROOT,
            work_root=f"{RUN_ROOT}/work",
            results_root="/n/groups/lab/results/example-analysis/dataset-a",
            receipts_root=f"{RUN_ROOT}/receipts",
            logs_root=f"{RUN_ROOT}/logs",
            promotion_root="/n/groups/lab/runs/example-analysis",
        ),
        stages=stages,
    )


def test_plan_round_trip_authenticates_canonical_bytes() -> None:
    """A JSON envelope must preserve the immutable plan and verify its digest."""

    plan = _plan()
    restored = ExecutionPlan.from_json(plan.to_json())

    assert restored == plan
    assert restored.plan_sha256 == plan.plan_sha256
    assert restored.topological_stage_ids() == ("preflight", "analyze")
    assert len(plan.plan_sha256) == 64

    manifest = RunManifest(
        run_id=plan.run_id,
        campaign=plan.campaign,
        pipeline=plan.pipeline,
        created_utc="2026-08-21T12:00:00Z",
        datasets=[dataset.dataset_id for dataset in plan.datasets],
    )
    assert manifest.validate(for_register=True) == []


def test_runtime_fingerprint_path_stays_out_of_the_version_one_digest() -> None:
    """Adding an input alias must not rewrite stored plans' canonical bytes.

    ``runtime_fingerprint_path`` is validated to equal ``argv[0]``, so it adds
    nothing the digest does not already cover.  Emitting it would change the
    canonical form of every schema-version-1 plan written before the field
    existed, and those stored digests would stop verifying while the schema
    version still claimed compatibility.
    """

    plan = _plan()
    body = json.loads(plan.to_json())
    commands = [task["command"] for stage in body["execution_plan"]["stages"] for task in stage.get("tasks", [])] + [
        stage["command"] for stage in body["execution_plan"]["stages"] if stage.get("command")
    ]
    assert commands
    assert not any("runtime_fingerprint_path" in command for command in commands)

    # An envelope written before the field existed still verifies.
    assert ExecutionPlan.from_json(json.dumps(body), expected_plan_sha256=plan.plan_sha256) == plan

    # And one that does carry it decodes to the same digest.
    for command in commands:
        command["runtime_fingerprint_path"] = command["argv"][0]
    carried = ExecutionPlan.from_json(json.dumps(body), expected_plan_sha256=plan.plan_sha256)
    assert carried.plan_sha256 == plan.plan_sha256


def test_digest_is_independent_of_set_like_input_order() -> None:
    """Dataset, stage, task, dependency, and receipt ordering must hash stably."""

    original = _plan()
    payload = original.to_dict()
    payload["datasets"].reverse()
    payload["stages"].reverse()
    analyze = next(stage for stage in payload["stages"] if stage["stage_id"] == "analyze")
    analyze["tasks"].reverse()

    reordered = ExecutionPlan.from_dict(payload)
    assert reordered.plan_sha256 == original.plan_sha256


def test_storage_binding_is_covered_by_dataset_identity() -> None:
    """A changed canonical storage binding must create a different execution plan."""

    plan = _plan()
    dataset = next(item for item in plan.datasets if item.dataset_id == "dataset-a")
    changed = replace(dataset, storage_binding_sha256="0" * 64)
    changed_plan = replace(plan, datasets=tuple(changed if item == dataset else item for item in plan.datasets))
    assert changed_plan.plan_sha256 != plan.plan_sha256


def test_envelope_tampering_fails_closed() -> None:
    """Changing any reviewed field without recomputing the digest is rejected."""

    envelope = json.loads(_plan().to_json())
    envelope["execution_plan"]["stages"][0]["resources"]["memory_mb"] += 1

    with pytest.raises(ValueError, match="digest mismatch"):
        ExecutionPlan.from_envelope(envelope)


def test_unknown_signed_fields_fail_closed() -> None:
    """No launcher may smuggle semantics this plan version ignores while hashing."""

    envelope = json.loads(_plan().to_json())
    envelope["execution_plan"]["stages"][0]["shell_fragment"] = "rm -rf /"

    with pytest.raises(ValueError, match="unsupported fields"):
        ExecutionPlan.from_envelope(envelope)


def test_duplicate_json_members_and_untrusted_digest_drift_fail_closed() -> None:
    """Ambiguous JSON and a mismatch with an independently reviewed SHA are rejected."""

    plan = _plan()
    duplicate = plan.to_json().replace(
        '"plan_sha256": ',
        f'"plan_sha256": "{plan.plan_sha256}",\n  "plan_sha256": ',
        1,
    )
    with pytest.raises(ValueError, match="duplicate key"):
        ExecutionPlan.from_json(duplicate)

    with pytest.raises(ValueError, match="independently trusted digest"):
        ExecutionPlan.from_json(plan.to_json(), expected_plan_sha256="0" * 64)


def test_plan_and_nested_contracts_are_frozen() -> None:
    """The validated object graph cannot be mutated after its hash is reviewed."""

    plan = _plan()
    with pytest.raises(FrozenInstanceError):
        plan.pipeline = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.stages[0].depends_on = ()  # type: ignore[misc]

    with pytest.raises(ValueError, match="immutable tuple"):
        RetryPolicy(  # type: ignore[arg-type]
            max_attempts=2,
            retryable_slurm_states=["NODE_FAIL"],
        )

    class DatasetSubclass(DatasetIdentity):
        """A duck-typed subclass that could override signed serialization."""

    with pytest.raises(ValueError, match="DatasetIdentity objects"):
        replace(
            plan,
            datasets=(DatasetSubclass(dataset_id="subclass", manifest_sha256="a" * 64),),
        )

    class PlanSubclass(ExecutionPlan):
        """A subclass could otherwise replace canonical serialization."""

    with pytest.raises(ValueError, match="subclasses are not supported"):
        PlanSubclass(**plan.__dict__)


def test_stage_graph_rejects_unknown_dependencies_and_cycles() -> None:
    """Every dependency must exist and the stage graph must remain acyclic."""

    unknown = StageSpec(
        stage_id="analyze",
        command=_command("/usr/bin/true"),
        resources=_resources(),
        expected_receipts=(_receipt("analyze"),),
        depends_on=("missing",),
    )
    with pytest.raises(ValueError, match="unknown dependencies"):
        _plan(stages=(unknown,))

    first = StageSpec(
        stage_id="first",
        command=_command("/usr/bin/true"),
        resources=_resources(),
        expected_receipts=(_receipt("first"),),
        depends_on=("second",),
    )
    second = StageSpec(
        stage_id="second",
        command=_command("/usr/bin/true"),
        resources=_resources(),
        expected_receipts=(_receipt("second"),),
        depends_on=("first",),
    )
    with pytest.raises(ValueError, match="contains a cycle"):
        _plan(stages=(first, second))


def test_plan_rejects_derived_attempt_bounds_above_identity_limit() -> None:
    """Dense retry DAGs must fail before three-digit attempt identities overflow."""

    stages: list[StageSpec] = []
    for index in range(9):
        stage_id = f"dense-{index}"
        stages.append(
            StageSpec(
                stage_id=stage_id,
                command=_command("/usr/bin/true"),
                resources=_resources(),
                expected_receipts=(_receipt(stage_id),),
                depends_on=tuple(stage.stage_id for stage in stages),
                dependency_mode="afterany" if stages else "afterok",
                retry_policy=RetryPolicy(max_attempts=5),
            )
        )

    with pytest.raises(ValueError, match="derived attempt bound 1025 exceeds.*999"):
        _plan(stages=tuple(stages))


def test_dependency_mode_distinguishes_stage_gates_from_reconcilers() -> None:
    """The signed DAG must preserve after-success versus after-terminal intent."""

    plan = _plan()
    analyze = next(stage for stage in plan.stages if stage.stage_id == "analyze")
    assert analyze.dependency_mode == "afterany"
    assert ExecutionPlan.from_json(plan.to_json()) == plan

    with pytest.raises(ValueError, match="dependency_mode"):
        StageSpec(
            stage_id="unsafe",
            command=_command("/usr/bin/true"),
            resources=_resources(),
            expected_receipts=(_receipt("unsafe"),),
            dependency_mode="after",
        )

    with pytest.raises(ValueError, match="root stage"):
        StageSpec(
            stage_id="root-reconciler",
            command=_command("/usr/bin/true"),
            resources=_resources(),
            expected_receipts=(_receipt("root-reconciler"),),
            dependency_mode="afterany",
        )


def test_array_tasks_bind_indices_and_exact_commands() -> None:
    """Missing-only retries have stable indices and no launcher-specific templates."""

    plan = _plan()
    analyze = next(stage for stage in plan.stages if stage.stage_id == "analyze")
    assert analyze.command is None
    assert [(task.array_index, task.task_id) for task in analyze.tasks] == [
        (0, "movie-0001"),
        (1, "movie-0002"),
    ]
    assert analyze.tasks[0].command.argv[-1] == "movie-0001"

    duplicate_index = replace(analyze.tasks[1], array_index=0)
    with pytest.raises(ValueError, match="array indices"):
        replace(analyze, tasks=(analyze.tasks[0], duplicate_index))


def test_receipts_must_be_unique_and_inside_canonical_receipts_root() -> None:
    """A task cannot point the reconciler at arbitrary or shared run payloads."""

    outside = StageSpec(
        stage_id="outside",
        command=_command("/usr/bin/true"),
        resources=_resources(),
        expected_receipts=(ReceiptSpec(receipt_id="outside", path="outputs/not-a-receipt.json"),),
    )
    with pytest.raises(ValueError, match="inside paths.receipts_root"):
        _plan(stages=(outside,))

    shared_path = "receipts/stages/shared/result.json"
    first = StageSpec(
        stage_id="first",
        command=_command("/usr/bin/true"),
        resources=_resources(),
        expected_receipts=(ReceiptSpec(receipt_id="first", path=shared_path),),
    )
    second = StageSpec(
        stage_id="second",
        command=_command("/usr/bin/true"),
        resources=_resources(),
        expected_receipts=(ReceiptSpec(receipt_id="second", path=shared_path),),
    )
    with pytest.raises(ValueError, match="is shared"):
        _plan(stages=(first, second))

    duplicate_id = StageSpec(
        stage_id="duplicate-id",
        command=_command("/usr/bin/true"),
        resources=_resources(),
        expected_receipts=(
            ReceiptSpec(
                receipt_id="first",
                path="receipts/stages/duplicate-id/completed.json",
            ),
        ),
    )
    with pytest.raises(ValueError, match="receipt ID"):
        _plan(stages=(first, duplicate_id))


def test_every_reconcilable_unit_requires_positive_completion_evidence() -> None:
    """An optional diagnostic alone can never certify a stage or array task."""

    optional = ReceiptSpec(
        receipt_id="optional",
        path="receipts/stages/optional/diagnostic.json",
        required=False,
    )
    with pytest.raises(ValueError, match="at least one required receipt"):
        StageSpec(
            stage_id="optional",
            command=_command("/usr/bin/true"),
            resources=_resources(),
            expected_receipts=(optional,),
        )

    with pytest.raises(ValueError, match="at least one required receipt"):
        TaskSpec(
            task_id="optional-task",
            array_index=1,
            command=_command("/usr/bin/true"),
            expected_receipts=(optional,),
        )


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (RetryPolicy(max_attempts=5), None),
        ("too-many", "max_attempts"),
        ("unknown-state", "unsupported retryable"),
        ("unbounded-default", "require max_attempts"),
    ],
)
def test_retry_policy_is_explicitly_bounded(policy: object, message: str | None) -> None:
    """Only small, declared operational retry envelopes are representable."""

    if policy == "too-many":
        constructor = lambda: RetryPolicy(max_attempts=6)  # noqa: E731
    elif policy == "unknown-state":
        constructor = lambda: RetryPolicy(max_attempts=2, retryable_slurm_states=("FAILED",))  # noqa: E731
    elif policy == "unbounded-default":
        constructor = lambda: RetryPolicy(max_attempts=1, retry_missing_receipts=True)  # noqa: E731
    else:
        assert isinstance(policy, RetryPolicy)
        assert policy.max_attempts == 5
        return

    with pytest.raises(ValueError, match=message):
        constructor()


def test_malformed_paths_and_resources_fail_before_submission() -> None:
    """Operationally dangerous roots and resource values never enter a plan."""

    with pytest.raises(ValueError, match="strictly inside run_root"):
        CanonicalPaths(
            run_root="/n/scratch/run",
            work_root="/tmp/work",
            results_root="/n/groups/results",
            receipts_root="/n/scratch/run/receipts",
            logs_root="/n/scratch/run/logs",
        )

    with pytest.raises(ValueError, match="memory_mb"):
        ResourceSpec(
            partition="short",
            cpus=2,
            memory_mb=0,
            time_limit="00:20:00",
        )

    with pytest.raises(ValueError, match="positive"):
        ResourceSpec(
            partition="short",
            cpus=2,
            memory_mb=4096,
            time_limit="00:00:00",
        )

    with pytest.raises(ValueError, match="between 1 and 255"):
        RetryPolicy(max_attempts=2, retryable_exit_codes=(0,))


def test_specialized_slurm_resources_are_preserved_without_raw_overrides() -> None:
    """GPU model, constraints, exclusions, and licenses remain part of the plan SHA."""

    resources = ResourceSpec(
        partition="gpu",
        cpus=4,
        memory_mb=16384,
        time_limit="01:00:00",
        gpus=1,
        gpu_type="l40s",
        constraint="avx2&localdisk",
        exclude_nodes=("compute-g-02", "compute-g-01"),
        licenses=("gurobi:1",),
    )
    restored = ResourceSpec.from_dict(resources.to_dict())
    assert restored == resources
    assert restored.exclude_nodes == ("compute-g-01", "compute-g-02")
    assert restored.to_dict()["gpu_type"] == "l40s"

    with pytest.raises(ValueError, match="gpu_type requires"):
        replace(resources, gpus=0)


def test_run_identity_and_command_context_are_bound_to_canonical_paths() -> None:
    """Registry identity cannot point reconciliation at another run's payloads."""

    plan = _plan()
    other_root = "/n/scratch/users/test/runs/example-campaign/RUN_20260821T120001Z_other"
    other_paths = CanonicalPaths(
        run_root=other_root,
        work_root=f"{other_root}/work",
        results_root=plan.paths.results_root,
        receipts_root=f"{other_root}/receipts",
        logs_root=f"{other_root}/logs",
        promotion_root=plan.paths.promotion_root,
    )
    with pytest.raises(ValueError, match="basename must equal run_id"):
        replace(plan, paths=other_paths)

    with pytest.raises(ValueError, match="campaign encoded in run_id"):
        replace(plan, campaign="different-campaign")

    outside_command = replace(
        next(stage for stage in plan.stages if stage.stage_id == "preflight").command,
        working_directory="/tmp/work",
    )
    preflight = next(stage for stage in plan.stages if stage.stage_id == "preflight")
    with pytest.raises(ValueError, match="working_directory must be inside"):
        _plan(stages=(replace(preflight, command=outside_command),))

    receipt_command = replace(preflight.command, working_directory=plan.paths.receipts_root)
    with pytest.raises(ValueError, match="paths.work_root"):
        _plan(stages=(replace(preflight, command=receipt_command),))


def test_mutable_run_roots_and_environment_inheritance_are_disallowed() -> None:
    """Work products cannot collide with evidence or inherit ambient launcher state."""

    with pytest.raises(ValueError, match="distinct and non-overlapping"):
        CanonicalPaths(
            run_root=RUN_ROOT,
            work_root=f"{RUN_ROOT}/payload",
            results_root="/n/groups/results",
            receipts_root=f"{RUN_ROOT}/payload/receipts",
            logs_root=f"{RUN_ROOT}/logs",
        )

    with pytest.raises(ValueError, match="distinct and non-overlapping"):
        CanonicalPaths(
            run_root=RUN_ROOT,
            work_root=f"{RUN_ROOT}/work",
            results_root=f"{RUN_ROOT}/work/results",
            receipts_root=f"{RUN_ROOT}/receipts",
            logs_root=f"{RUN_ROOT}/logs",
        )

    with pytest.raises(ValueError, match="environment_mode"):
        CommandSpec(
            argv=("/usr/bin/true",),
            working_directory=f"{RUN_ROOT}/work",
            runtime_fingerprint_sha256="e" * 64,
            environment_mode="inherit",
        )


def test_unpaired_unicode_surrogates_fail_during_contract_validation() -> None:
    """Malformed decoded strings never leak a UnicodeEncodeError from hashing."""

    with pytest.raises(ValueError, match="Unicode"):
        DatasetIdentity(
            dataset_id="surrogate",
            manifest_sha256="a" * 64,
            source_uri="amdata://bad/\ud800",
        )

    with pytest.raises(ValueError, match="single-line"):
        CommandSpec(
            argv=("/usr/bin/true", "\ud800"),
            working_directory=f"{RUN_ROOT}/work",
            runtime_fingerprint_sha256="e" * 64,
        )


def test_direct_construction_rejects_wrong_runtime_types_cleanly() -> None:
    """Adapters get field-specific validation errors rather than incidental TypeErrors."""

    with pytest.raises(ValueError, match="run_root must be a single-line"):
        CanonicalPaths(  # type: ignore[arg-type]
            run_root=1,
            work_root="/n/scratch/run/work",
            results_root="/n/groups/results",
            receipts_root="/n/scratch/run/receipts",
            logs_root="/n/scratch/run/logs",
        )

    with pytest.raises(ValueError, match="command arguments"):
        CommandSpec(  # type: ignore[arg-type]
            argv=(1,),
            working_directory=f"{RUN_ROOT}/work",
            runtime_fingerprint_sha256="e" * 64,
        )
