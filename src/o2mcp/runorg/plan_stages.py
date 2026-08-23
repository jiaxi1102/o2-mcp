"""Exact command, array-task, and stage-DAG contracts for O2 plans."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from o2mcp.runorg.plan_components import (
    DEPENDENCY_MODES,
    ReceiptSpec,
    ResourceSpec,
    RetryPolicy,
    _is_single_line_string,
    _reject_unknown_keys,
    _require_int,
    _require_mapping,
    _require_sequence,
    _require_str,
    _validate_absolute_path,
    _validate_identifier,
    _validate_sha256,
)

_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class CommandSpec:
    """An exact argv and authenticated runtime context for one Slurm payload.

    Shell fragments and implicit module setup are absent. Adapters must use an
    absolute executable, bind the runtime fingerprint, and enumerate non-secret
    environment values that can affect scientific behavior.
    """

    argv: tuple[str, ...]
    working_directory: str
    runtime_fingerprint_sha256: str
    runtime_fingerprint_path: str | None = None
    environment: tuple[str, ...] = ()
    environment_mode: str = "clean"

    def __post_init__(self) -> None:
        """Reject execution context that different launchers could interpret differently."""

        if not isinstance(self.argv, tuple) or not isinstance(self.environment, tuple):
            raise ValueError("command argv and environment must be immutable tuples")
        if not self.argv:
            raise ValueError("command argv cannot be empty")
        for argument in self.argv:
            if not _is_single_line_string(argument):
                raise ValueError("command arguments must be non-empty single-line strings")
        _validate_absolute_path(self.argv[0], "command executable")
        _validate_absolute_path(self.working_directory, "command working_directory")
        _validate_sha256(self.runtime_fingerprint_sha256, "command runtime_fingerprint_sha256")
        runtime_path = self.runtime_fingerprint_path or self.argv[0]
        _validate_absolute_path(runtime_path, "command runtime_fingerprint_path")
        # Hashing an unrelated lock file does not authenticate the executable.
        # Pipelines that need to bind a larger environment must make argv[0] an
        # immutable wrapper whose own bytes are hashed here; that wrapper can then
        # validate a signed environment manifest before launching scientific code.
        if runtime_path != self.argv[0]:
            raise ValueError("command runtime_fingerprint_path must equal argv[0]")
        object.__setattr__(self, "runtime_fingerprint_path", runtime_path)
        if self.environment_mode != "clean":
            raise ValueError("command environment_mode must be 'clean'")

        names: list[str] = []
        for assignment in self.environment:
            if not _is_single_line_string(assignment) or "=" not in assignment:
                raise ValueError("command environment entries must be single-line NAME=VALUE strings")
            name, _value = assignment.split("=", 1)
            if not _ENVIRONMENT_NAME_RE.fullmatch(name):
                raise ValueError(f"invalid command environment variable name: {name!r}")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError("command environment variable names must be unique")

        # Environment order has no semantics; canonicalize it for stable hashes.
        object.__setattr__(self, "environment", tuple(sorted(self.environment)))

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by the plan digest."""

        return {
            "argv": list(self.argv),
            "environment": sorted(self.environment),
            "environment_mode": self.environment_mode,
            "runtime_fingerprint_sha256": self.runtime_fingerprint_sha256,
            # ``runtime_fingerprint_path`` is deliberately absent: it is
            # validated to equal ``argv[0]`` and therefore carries nothing the
            # digest does not already cover.  Emitting it would rewrite the
            # canonical form of every schema-version-1 plan written before the
            # field existed, so their stored digests would no longer verify and
            # their execution evidence could not be replayed.
            "working_directory": self.working_directory,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CommandSpec:
        """Build one exact command from decoded JSON."""

        data = _require_mapping(value, "command")
        _reject_unknown_keys(
            data,
            {
                "argv",
                "working_directory",
                "runtime_fingerprint_sha256",
                "runtime_fingerprint_path",
                "environment",
                "environment_mode",
            },
            "command",
        )
        argv = _require_sequence(data.get("argv"), "command.argv")
        environment = _require_sequence(data.get("environment", []), "command.environment")
        return cls(
            argv=tuple(_require_str(argument, "command.argv[]") for argument in argv),
            working_directory=_require_str(data.get("working_directory"), "command.working_directory"),
            runtime_fingerprint_sha256=_require_str(
                data.get("runtime_fingerprint_sha256"),
                "command.runtime_fingerprint_sha256",
            ),
            runtime_fingerprint_path=(
                _require_str(data["runtime_fingerprint_path"], "command.runtime_fingerprint_path")
                if data.get("runtime_fingerprint_path") is not None
                else None
            ),
            environment=tuple(_require_str(item, "command.environment[]") for item in environment),
            environment_mode=_require_str(data.get("environment_mode", "clean"), "command.environment_mode"),
        )


@dataclass(frozen=True)
class TaskSpec:
    """One stable array task and the receipts that prove its completion."""

    task_id: str
    array_index: int
    command: CommandSpec
    expected_receipts: tuple[ReceiptSpec, ...]

    def __post_init__(self) -> None:
        """Require at least one receipt so task success is externally provable."""

        if not isinstance(self.expected_receipts, tuple):
            raise ValueError("task expected_receipts must be an immutable tuple")
        if any(type(receipt) is not ReceiptSpec for receipt in self.expected_receipts):
            raise ValueError("task expected_receipts entries must be ReceiptSpec objects")
        if isinstance(self.array_index, bool) or not isinstance(self.array_index, int) or self.array_index < 0:
            raise ValueError("task array_index must be a nonnegative integer")
        if type(self.command) is not CommandSpec:
            raise ValueError("task command must be a CommandSpec")
        _validate_identifier(self.task_id, "task_id")
        if not self.expected_receipts:
            raise ValueError("each task must declare at least one expected receipt")
        if not any(receipt.required for receipt in self.expected_receipts):
            raise ValueError("each task must declare at least one required receipt")
        _reject_duplicate_receipts(self.expected_receipts, f"task {self.task_id}")
        object.__setattr__(
            self,
            "expected_receipts",
            tuple(sorted(self.expected_receipts, key=lambda item: item.path)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by the plan digest."""

        return {
            "array_index": self.array_index,
            "command": self.command.to_dict(),
            "expected_receipts": [
                receipt.to_dict() for receipt in sorted(self.expected_receipts, key=lambda item: item.path)
            ],
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskSpec:
        """Build an array task from decoded JSON."""

        data = _require_mapping(value, "task")
        _reject_unknown_keys(data, {"task_id", "array_index", "command", "expected_receipts"}, "task")
        receipts = _require_sequence(data.get("expected_receipts"), "task.expected_receipts")
        return cls(
            task_id=_require_str(data.get("task_id"), "task.task_id"),
            array_index=_require_int(data.get("array_index"), "task.array_index"),
            command=CommandSpec.from_dict(_require_mapping(data.get("command"), "task.command")),
            expected_receipts=tuple(
                ReceiptSpec.from_dict(_require_mapping(item, "task.expected_receipts[]")) for item in receipts
            ),
        )


@dataclass(frozen=True)
class StageSpec:
    """One node in the execution DAG, including optional stable array tasks."""

    stage_id: str
    resources: ResourceSpec
    expected_receipts: tuple[ReceiptSpec, ...]
    command: CommandSpec | None = None
    depends_on: tuple[str, ...] = ()
    dependency_mode: str = "afterok"
    tasks: tuple[TaskSpec, ...] = ()
    retry_policy: RetryPolicy = RetryPolicy()

    def __post_init__(self) -> None:
        """Validate command, dependencies, task identities, and receipts."""

        for field_name in ("depends_on", "expected_receipts", "tasks"):
            if not isinstance(getattr(self, field_name), tuple):
                raise ValueError(f"stage {field_name} must be an immutable tuple")
        if type(self.resources) is not ResourceSpec:
            raise ValueError("stage resources must be a ResourceSpec")
        if type(self.retry_policy) is not RetryPolicy:
            raise ValueError("stage retry_policy must be a RetryPolicy")
        if any(type(receipt) is not ReceiptSpec for receipt in self.expected_receipts):
            raise ValueError("stage expected_receipts entries must be ReceiptSpec objects")
        if any(type(task) is not TaskSpec for task in self.tasks):
            raise ValueError("stage tasks entries must be TaskSpec objects")
        if self.command is not None and type(self.command) is not CommandSpec:
            raise ValueError("stage command must be a CommandSpec when supplied")
        _validate_identifier(self.stage_id, "stage_id")
        if self.tasks and self.command is not None:
            raise ValueError(f"array stage {self.stage_id} must put exact commands on its tasks")
        if not self.tasks and self.command is None:
            raise ValueError(f"non-array stage {self.stage_id} must declare a command")
        if any(not isinstance(dependency, str) for dependency in self.depends_on):
            raise ValueError(f"stage {self.stage_id} dependencies must be strings")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError(f"stage {self.stage_id} dependencies cannot contain duplicates")
        if self.stage_id in self.depends_on:
            raise ValueError(f"stage {self.stage_id} cannot depend on itself")
        if self.dependency_mode not in DEPENDENCY_MODES:
            raise ValueError(f"stage {self.stage_id} dependency_mode must be one of {DEPENDENCY_MODES}")
        if not self.depends_on and self.dependency_mode != "afterok":
            raise ValueError(f"root stage {self.stage_id} cannot use dependency_mode {self.dependency_mode!r}")
        _reject_duplicate_receipts(self.expected_receipts, f"stage {self.stage_id}")

        if not self.expected_receipts and not self.tasks:
            raise ValueError(f"stage {self.stage_id} must declare a stage receipt or receipt-bearing tasks")
        if not self.tasks and not any(receipt.required for receipt in self.expected_receipts):
            raise ValueError(f"non-array stage {self.stage_id} must declare at least one required receipt")

        task_ids = [task.task_id for task in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError(f"stage {self.stage_id} task IDs must be unique")
        task_indices = [task.array_index for task in self.tasks]
        if len(set(task_indices)) != len(task_indices):
            raise ValueError(f"stage {self.stage_id} task array indices must be unique")
        if self.resources.array_parallelism > max(1, len(self.tasks)) and self.tasks:
            raise ValueError(f"stage {self.stage_id} array_parallelism cannot exceed its task count")

        # Canonicalize unordered DAG edges, receipts, and task identities.
        object.__setattr__(self, "depends_on", tuple(sorted(self.depends_on)))
        object.__setattr__(
            self,
            "expected_receipts",
            tuple(sorted(self.expected_receipts, key=lambda item: item.path)),
        )
        object.__setattr__(self, "tasks", tuple(sorted(self.tasks, key=lambda item: item.task_id)))

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by the plan digest."""

        return {
            "command": self.command.to_dict() if self.command is not None else None,
            "dependency_mode": self.dependency_mode,
            "depends_on": sorted(self.depends_on),
            "expected_receipts": [
                receipt.to_dict() for receipt in sorted(self.expected_receipts, key=lambda item: item.path)
            ],
            "resources": self.resources.to_dict(),
            "retry_policy": self.retry_policy.to_dict(),
            "stage_id": self.stage_id,
            "tasks": [task.to_dict() for task in sorted(self.tasks, key=lambda item: item.task_id)],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StageSpec:
        """Build a stage specification from decoded JSON."""

        data = _require_mapping(value, "stage")
        _reject_unknown_keys(
            data,
            {
                "stage_id",
                "command",
                "resources",
                "expected_receipts",
                "depends_on",
                "dependency_mode",
                "tasks",
                "retry_policy",
            },
            "stage",
        )
        dependencies = _require_sequence(data.get("depends_on", []), "stage.depends_on")
        receipts = _require_sequence(data.get("expected_receipts"), "stage.expected_receipts")
        tasks = _require_sequence(data.get("tasks", []), "stage.tasks")
        return cls(
            stage_id=_require_str(data.get("stage_id"), "stage.stage_id"),
            resources=ResourceSpec.from_dict(_require_mapping(data.get("resources"), "stage.resources")),
            expected_receipts=tuple(
                ReceiptSpec.from_dict(_require_mapping(item, "stage.expected_receipts[]")) for item in receipts
            ),
            command=(
                CommandSpec.from_dict(_require_mapping(data["command"], "stage.command"))
                if data.get("command") is not None
                else None
            ),
            depends_on=tuple(_require_str(dependency, "stage.depends_on[]") for dependency in dependencies),
            dependency_mode=_require_str(data.get("dependency_mode", "afterok"), "stage.dependency_mode"),
            tasks=tuple(TaskSpec.from_dict(_require_mapping(item, "stage.tasks[]")) for item in tasks),
            retry_policy=RetryPolicy.from_dict(_require_mapping(data.get("retry_policy", {}), "stage.retry_policy")),
        )


def _reject_duplicate_receipts(receipts: Sequence[ReceiptSpec], owner: str) -> None:
    """Reject duplicate receipt identities or paths inside one contract owner."""

    receipt_ids = [receipt.receipt_id for receipt in receipts]
    paths = [receipt.path for receipt in receipts]
    if len(set(receipt_ids)) != len(receipt_ids):
        raise ValueError(f"{owner} receipt IDs must be unique")
    if len(set(paths)) != len(paths):
        raise ValueError(f"{owner} receipt paths must be unique")


__all__ = [
    "CommandSpec",
    "StageSpec",
    "TaskSpec",
]
