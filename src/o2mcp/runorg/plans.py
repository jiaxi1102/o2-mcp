"""Immutable, project-neutral execution-plan contract for governed O2 runs.

The run-organization layer previously knew how to allocate a run directory and
record Slurm job IDs, but it had no checksummed contract for *what* the run was
supposed to execute.  :class:`ExecutionPlan` binds the project and pipeline
identity, source code, datasets, canonical paths, stage DAG, resources, expected
receipts, and bounded retry policy without embedding clock, GEM, or other
scientific semantics in :mod:`o2mcp`.

The plan is a frozen object graph.  Its SHA-256 is calculated from canonical JSON
that excludes the digest itself, allowing later submission and reconciliation to
use the digest as an idempotency key without trusting a caller-supplied value.
Pipeline repositories remain responsible for rendering scientific commands and
defining scientific acceptance.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from o2mcp.runorg.plan_components import (
    _GIT_COMMIT_RE,
    DEPENDENCY_MODES,
    EXECUTION_PLAN_SCHEMA_VERSION,
    RETRYABLE_SLURM_STATES,
    CanonicalPaths,
    DatasetIdentity,
    ReceiptSpec,
    ResourceSpec,
    RetryPolicy,
    _canonical_json_bytes,
    _is_within,
    _reject_unknown_keys,
    _require_int,
    _require_mapping,
    _require_sequence,
    _require_str,
    _validate_identifier,
    _validate_sha256,
)
from o2mcp.runorg.plan_stages import CommandSpec, StageSpec, TaskSpec
from o2mcp.runorg.runs import _RUN_ID_RE, campaign_of
from o2mcp.runorg.strict_json import strict_json_value


@dataclass(frozen=True)
class ExecutionPlan:
    """A complete immutable operational contract for one registered O2 run."""

    project: str
    campaign: str
    pipeline: str
    run_id: str
    source_commit: str
    source_bundle_sha256: str
    datasets: tuple[DatasetIdentity, ...]
    paths: CanonicalPaths
    stages: tuple[StageSpec, ...]
    schema_version: int = EXECUTION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate cross-object invariants after every construction path."""

        if type(self) is not ExecutionPlan:
            raise ValueError("ExecutionPlan subclasses are not supported by the immutable wire contract")
        if not isinstance(self.datasets, tuple) or not isinstance(self.stages, tuple):
            raise ValueError("execution plan datasets and stages must be immutable tuples")
        if type(self.paths) is not CanonicalPaths:
            raise ValueError("execution plan paths must be CanonicalPaths")
        if any(type(dataset) is not DatasetIdentity for dataset in self.datasets):
            raise ValueError("execution plan datasets entries must be DatasetIdentity objects")
        if any(type(stage) is not StageSpec for stage in self.stages):
            raise ValueError("execution plan stages entries must be StageSpec objects")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise ValueError("schema_version must be an integer")
        if self.schema_version != EXECUTION_PLAN_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {EXECUTION_PLAN_SCHEMA_VERSION}, got {self.schema_version}")
        for field_name in ("project", "campaign", "pipeline", "run_id"):
            _validate_identifier(getattr(self, field_name), field_name)
        if not _RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError("run_id must match the registered RUN_<UTCtimestamp>Z_<slug> convention")
        if campaign_of(self.run_id) != self.campaign:
            raise ValueError("campaign must match the campaign encoded in run_id")
        if posixpath.basename(self.paths.run_root) != self.run_id:
            raise ValueError("paths.run_root basename must equal run_id")
        if posixpath.basename(posixpath.dirname(self.paths.run_root)) != self.campaign:
            raise ValueError("paths.run_root parent directory must equal campaign")
        if not isinstance(self.source_commit, str) or not _GIT_COMMIT_RE.fullmatch(self.source_commit):
            raise ValueError("source_commit must be a full 40- or 64-character lowercase Git object ID")
        _validate_sha256(self.source_bundle_sha256, "source_bundle_sha256")
        if not self.datasets:
            raise ValueError("execution plan must contain at least one dataset")
        dataset_ids = [dataset.dataset_id for dataset in self.datasets]
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("dataset IDs must be unique")
        if not self.stages:
            raise ValueError("execution plan must contain at least one stage")

        stage_ids = [stage.stage_id for stage in self.stages]
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("stage IDs must be unique")
        stage_id_set = set(stage_ids)
        for stage in self.stages:
            missing = sorted(set(stage.depends_on) - stage_id_set)
            if missing:
                raise ValueError(f"stage {stage.stage_id} has unknown dependencies: {missing}")

        # Computing the order detects cycles and supplies a deterministic order for
        # later launchers without requiring the input JSON to be topologically sorted.
        self.topological_stage_ids()
        for stage_id, bound in self._derived_attempt_bounds().items():
            if bound > 999:
                # SubmissionIdentity encodes attempts as exactly three decimal
                # digits. Reject an otherwise valid dense DAG here rather than
                # letting evidence scans fail only after a run is registered.
                raise ValueError(f"stage {stage_id} derived attempt bound {bound} exceeds the identity limit 999")
        self._validate_receipt_tree()
        self._validate_command_roots()

        # Dataset and stage arrays are identity-keyed collections rather than an
        # execution sequence; DAG dependencies determine launch order.
        object.__setattr__(
            self,
            "datasets",
            tuple(sorted(self.datasets, key=lambda item: item.dataset_id)),
        )
        object.__setattr__(
            self,
            "stages",
            tuple(sorted(self.stages, key=lambda item: item.stage_id)),
        )

    def _validate_receipt_tree(self) -> None:
        """Require every expected receipt to live in the canonical receipt tree."""

        all_receipts: list[tuple[str, ReceiptSpec]] = []
        for stage in self.stages:
            all_receipts.extend((f"stage {stage.stage_id}", receipt) for receipt in stage.expected_receipts)
            for task in stage.tasks:
                all_receipts.extend(
                    (f"stage {stage.stage_id} task {task.task_id}", receipt) for receipt in task.expected_receipts
                )

        seen_paths: dict[str, str] = {}
        seen_ids: dict[str, str] = {}
        for owner, receipt in all_receipts:
            absolute = posixpath.join(self.paths.run_root, receipt.path)
            if not _is_within(absolute, self.paths.receipts_root):
                raise ValueError(f"{owner} receipt {receipt.path!r} must resolve inside paths.receipts_root")
            engine_receipts = posixpath.join(self.paths.receipts_root, "execution")
            if absolute == engine_receipts or _is_within(absolute, engine_receipts):
                raise ValueError(
                    f"{owner} receipt {receipt.path!r} overlaps the reserved execution-engine receipt namespace"
                )
            previous = seen_paths.get(receipt.path)
            if previous is not None:
                raise ValueError(f"receipt path {receipt.path!r} is shared by {previous} and {owner}")
            seen_paths[receipt.path] = owner
            previous_id_owner = seen_ids.get(receipt.receipt_id)
            if previous_id_owner is not None:
                raise ValueError(f"receipt ID {receipt.receipt_id!r} is shared by {previous_id_owner} and {owner}")
            seen_ids[receipt.receipt_id] = owner

    def _validate_command_roots(self) -> None:
        """Keep every task working directory inside the authenticated run tree."""

        for stage in self.stages:
            commands = ([stage.command] if stage.command is not None else []) + [task.command for task in stage.tasks]
            for command in commands:
                if command.working_directory != self.paths.work_root and not _is_within(
                    command.working_directory,
                    self.paths.work_root,
                ):
                    raise ValueError(f"stage {stage.stage_id} command working_directory must be inside paths.work_root")

    def topological_stage_ids(self) -> tuple[str, ...]:
        """Return a stable topological order or raise when the DAG contains a cycle."""

        by_id = {stage.stage_id: stage for stage in self.stages}
        state: dict[str, int] = {}
        ordered: list[str] = []

        def visit(stage_id: str, stack: tuple[str, ...]) -> None:
            status = state.get(stage_id, 0)
            if status == 2:
                return
            if status == 1:
                cycle = " -> ".join(stack + (stage_id,))
                raise ValueError(f"stage dependency graph contains a cycle: {cycle}")
            state[stage_id] = 1
            for dependency in sorted(by_id[stage_id].depends_on):
                visit(dependency, stack + (stage_id,))
            state[stage_id] = 2
            ordered.append(stage_id)

        for stage_id in sorted(by_id):
            visit(stage_id, ())
        return tuple(ordered)

    def _derived_attempt_bounds(self) -> dict[str, int]:
        """Derive complete bounded identity spaces for every signed stage.

        An ``afterany`` stage needs one replacement generation for every
        noninitial generation of each direct dependency. An ``afterok`` stage
        needs replacement generations only when a dependency itself was
        regenerated by an upstream replacement; the dependency's own retries
        happen before that child is eligible to launch. Dependency bounds
        already include upstream-triggered generations, so this recursive
        calculation propagates the finite allowance through mixed-mode DAGs.
        """

        by_id = {stage.stage_id: stage for stage in self.stages}
        derived: dict[str, int] = {}

        def derive(stage_id: str) -> int:
            """Memoize one node after the graph's cycle check has succeeded."""

            if stage_id in derived:
                return derived[stage_id]
            stage = by_id[stage_id]
            bound = stage.retry_policy.max_attempts
            if stage.depends_on:
                if stage.dependency_mode == "afterany":
                    bound += sum(derive(dependency) - 1 for dependency in stage.depends_on)
                else:
                    bound += sum(
                        derive(dependency) - by_id[dependency].retry_policy.max_attempts
                        for dependency in stage.depends_on
                    )
            derived[stage_id] = bound
            return bound

        for stage_id in self.topological_stage_ids():
            derive(stage_id)
        return derived

    def signed_attempt_bound(self, stage_id: str) -> int:
        """Return one stage's plan-validated scheduler identity bound."""

        bounds = self._derived_attempt_bounds()
        try:
            return bounds[stage_id]
        except KeyError as error:
            raise ValueError(f"unknown stage_id {stage_id!r}") from error

    def to_dict(self) -> dict[str, Any]:
        """Return the unhashed canonical plan payload."""

        return {
            "campaign": self.campaign,
            "datasets": [dataset.to_dict() for dataset in sorted(self.datasets, key=lambda item: item.dataset_id)],
            "paths": self.paths.to_dict(),
            "pipeline": self.pipeline,
            "project": self.project,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "source_bundle_sha256": self.source_bundle_sha256,
            "source_commit": self.source_commit,
            "stages": [stage.to_dict() for stage in sorted(self.stages, key=lambda item: item.stage_id)],
        }

    @property
    def plan_sha256(self) -> str:
        """Return the immutable digest derived from canonical plan content."""

        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()

    def to_envelope(self) -> dict[str, Any]:
        """Return the self-checksummed on-disk representation."""

        return {
            "execution_plan": self.to_dict(),
            "plan_sha256": self.plan_sha256,
        }

    def to_json(self) -> str:
        """Serialize the self-checksummed envelope as review-friendly JSON."""

        return json.dumps(self.to_envelope(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExecutionPlan:
        """Build and validate an execution plan from its unhashed payload."""

        data = _require_mapping(value, "execution_plan")
        _reject_unknown_keys(
            data,
            {
                "project",
                "campaign",
                "pipeline",
                "run_id",
                "source_commit",
                "source_bundle_sha256",
                "datasets",
                "paths",
                "stages",
                "schema_version",
            },
            "execution_plan",
        )
        datasets = _require_sequence(data.get("datasets"), "execution_plan.datasets")
        stages = _require_sequence(data.get("stages"), "execution_plan.stages")
        return cls(
            project=_require_str(data.get("project"), "execution_plan.project"),
            campaign=_require_str(data.get("campaign"), "execution_plan.campaign"),
            pipeline=_require_str(data.get("pipeline"), "execution_plan.pipeline"),
            run_id=_require_str(data.get("run_id"), "execution_plan.run_id"),
            source_commit=_require_str(data.get("source_commit"), "execution_plan.source_commit"),
            source_bundle_sha256=_require_str(
                data.get("source_bundle_sha256"),
                "execution_plan.source_bundle_sha256",
            ),
            datasets=tuple(
                DatasetIdentity.from_dict(_require_mapping(item, "execution_plan.datasets[]")) for item in datasets
            ),
            paths=CanonicalPaths.from_dict(_require_mapping(data.get("paths"), "execution_plan.paths")),
            stages=tuple(StageSpec.from_dict(_require_mapping(item, "execution_plan.stages[]")) for item in stages),
            schema_version=_require_int(data.get("schema_version"), "execution_plan.schema_version"),
        )

    @classmethod
    def from_envelope(cls, value: Mapping[str, Any]) -> ExecutionPlan:
        """Load a self-checksummed envelope and fail closed on digest mismatch."""

        envelope = _require_mapping(value, "execution plan envelope")
        _reject_unknown_keys(
            envelope,
            {"execution_plan", "plan_sha256"},
            "execution plan envelope",
        )
        recorded_sha = _require_str(envelope.get("plan_sha256"), "plan_sha256")
        _validate_sha256(recorded_sha, "plan_sha256")
        plan = cls.from_dict(_require_mapping(envelope.get("execution_plan"), "execution_plan"))
        if recorded_sha != plan.plan_sha256:
            raise ValueError("execution plan digest mismatch: the envelope content changed after hashing")
        return plan

    @classmethod
    def from_json(cls, text: str, *, expected_plan_sha256: str | None = None) -> ExecutionPlan:
        """Load an integrity-checked plan and optionally authenticate a trusted SHA.

        The digest stored beside the plan detects accidental corruption but is
        not a signature.  Supplying a separately trusted expected digest is what
        authenticates the exact reviewed plan at a submission boundary.
        """

        decoded = strict_json_value(text, "execution plan")
        plan = cls.from_envelope(_require_mapping(decoded, "execution plan envelope"))
        if expected_plan_sha256 is not None:
            _validate_sha256(expected_plan_sha256, "expected_plan_sha256")
            if plan.plan_sha256 != expected_plan_sha256:
                raise ValueError("execution plan does not match the independently trusted digest")
        return plan


__all__ = [
    "DEPENDENCY_MODES",
    "CanonicalPaths",
    "CommandSpec",
    "DatasetIdentity",
    "EXECUTION_PLAN_SCHEMA_VERSION",
    "ExecutionPlan",
    "RETRYABLE_SLURM_STATES",
    "ReceiptSpec",
    "ResourceSpec",
    "RetryPolicy",
    "StageSpec",
    "TaskSpec",
]
