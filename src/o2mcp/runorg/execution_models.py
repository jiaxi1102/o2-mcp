"""Frozen operational records for idempotent Slurm plan execution.

These values form the boundary between the pure coordinator and a concrete O2
transport.  Keeping them project-neutral and immutable lets tests replace Slurm
and the remote filesystem without weakening the production identity contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from o2mcp.runorg.plan_components import ReceiptSpec, ResourceSpec, _validate_identifier, _validate_sha256
from o2mcp.runorg.plan_stages import CommandSpec
from o2mcp.runorg.strict_json import exact_bool, exact_int, exact_list, exact_object, exact_str

EXECUTION_RECEIPT_SCHEMA_VERSION = 1

ACTIVE_SLURM_STATES = frozenset(
    {
        # Slurm can render a nonterminal state flag instead of the underlying
        # base state.  Keep the official flag names here as well as the active
        # base states so callers do not mistake a scheduler transition for a
        # final accounting result.
        "CONFIGURING",
        "COMPLETING",
        "EXPEDITING",
        "LAUNCH_FAILED",
        "PENDING",
        "POWER_UP_NODE",
        "RECONFIG_FAIL",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "RESIZING",
        "RESV_DEL_HOLD",
        "RUNNING",
        "SIGNALING",
        "SPECIAL_EXIT",
        "STAGE_OUT",
        "STOPPED",
        "SUSPENDED",
        "UPDATE_DB",
    }
)
SUCCESS_SLURM_STATES = frozenset({"COMPLETED"})
TERMINAL_SLURM_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)

RECONCILE_WAIT = "WAIT"
RECONCILE_COMPLETE = "COMPLETED"
RECONCILE_RETRY = "RETRY_MISSING_ONLY"
RECONCILE_RETRY_SUBMITTED = "RETRY_SUBMITTED"
RECONCILE_FAILED = "FAILED"

_COMMENT_RE = re.compile(r"^o2plan:v1:(?P<sha>[0-9a-f]{64}):(?P<stage>[A-Za-z0-9._-]+):a(?P<attempt>\d{3})$")


def canonical_json(value: dict[str, Any]) -> str:
    """Return deterministic JSON text for immutable execution receipts."""

    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class SubmissionIdentity:
    """The exact plan/stage/attempt identity placed in Slurm ``--comment``."""

    plan_sha256: str
    stage_id: str
    attempt: int

    def __post_init__(self) -> None:
        """Validate values before they cross the scheduler boundary."""

        _validate_sha256(self.plan_sha256, "plan_sha256")
        _validate_identifier(self.stage_id, "stage_id")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or not 1 <= self.attempt <= 999:
            raise ValueError("attempt must be an integer between 1 and 999")

    @property
    def comment(self) -> str:
        """Return a reversible, collision-resistant Slurm comment value."""

        return f"o2plan:v1:{self.plan_sha256}:{self.stage_id}:a{self.attempt:03d}"

    @classmethod
    def from_comment(cls, comment: str) -> SubmissionIdentity:
        """Parse an engine-owned Slurm comment or fail closed."""

        match = _COMMENT_RE.fullmatch(comment)
        if match is None:
            raise ValueError("not a valid o2-mcp execution comment")
        return cls(match.group("sha"), match.group("stage"), int(match.group("attempt")))


@dataclass(frozen=True)
class PlannedTask:
    """One task selected for a particular Slurm attempt."""

    task_id: str
    array_index: int | None
    command: CommandSpec
    expected_receipts: tuple[ReceiptSpec, ...]

    def __post_init__(self) -> None:
        """Require a stable task identity and exact command/receipt objects."""

        _validate_identifier(self.task_id, "task_id")
        if self.array_index is not None and (
            isinstance(self.array_index, bool) or not isinstance(self.array_index, int) or self.array_index < 0
        ):
            raise ValueError("array_index must be a nonnegative integer or None")
        if type(self.command) is not CommandSpec:
            raise ValueError("planned task command must be CommandSpec")
        if not isinstance(self.expected_receipts, tuple) or not self.expected_receipts:
            raise ValueError("planned task receipts must be a non-empty tuple")
        if any(type(receipt) is not ReceiptSpec for receipt in self.expected_receipts):
            raise ValueError("planned task receipts must contain ReceiptSpec values")


@dataclass(frozen=True)
class SubmissionRequest:
    """A fully rendered scheduler request presented to an execution backend."""

    identity: SubmissionIdentity
    run_id: str
    tasks: tuple[PlannedTask, ...]
    resources: ResourceSpec
    dependency_mode: str
    dependency_job_ids: tuple[str, ...]
    script_path: str
    script_text: str
    stdout_pattern: str
    stderr_pattern: str
    begin_delay_seconds: int = 0
    # Earlier generations of this same stage.  They are ordering-only: a
    # replacement runs the same signed commands in the same working directory
    # and writes the same receipt and output paths, so it must not start while a
    # prior attempt is still queued or running.  They are deliberately NOT part
    # of ``dependency_job_ids``, which is the signed DAG tuple whose cardinality
    # is authenticated against ``depends_on`` throughout the evidence readers.
    ordering_job_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate scheduler-facing values before submission can occur."""

        if not self.tasks:
            raise ValueError("submission request must contain at least one task")
        if any(type(task) is not PlannedTask for task in self.tasks):
            raise ValueError("submission tasks must contain PlannedTask values")
        indices = [task.array_index for task in self.tasks]
        is_array = any(index is not None for index in indices)
        if is_array and any(index is None for index in indices):
            raise ValueError("a submission cannot mix array and non-array tasks")
        if len(set(indices)) != len(indices):
            raise ValueError("submission task array indices must be unique")
        if self.dependency_mode not in {"afterany", "afterok"}:
            raise ValueError("dependency_mode must be afterany or afterok")
        if any(not job_id.isdigit() for job_id in self.dependency_job_ids):
            raise ValueError("dependency job IDs must be numeric strings")
        if len(set(self.dependency_job_ids)) != len(self.dependency_job_ids):
            raise ValueError("dependency job IDs cannot contain duplicates")
        for path_name in ("script_path", "stdout_pattern", "stderr_pattern"):
            path = getattr(self, path_name)
            if not path.startswith("/") or "\n" in path or "\r" in path:
                raise ValueError(f"{path_name} must be a single-line absolute path")
        if (
            isinstance(self.begin_delay_seconds, bool)
            or not isinstance(self.begin_delay_seconds, int)
            or not 0 <= self.begin_delay_seconds <= 3600
        ):
            raise ValueError("begin_delay_seconds must be an integer between 0 and 3600")

    @property
    def comment(self) -> str:
        """Expose the submission identity in the form queried from Slurm."""

        return self.identity.comment

    @property
    def task_indices(self) -> tuple[int, ...]:
        """Return the exact selected array indices, including index zero."""

        return tuple(task.array_index for task in self.tasks if task.array_index is not None)

    def sbatch_args(self) -> tuple[str, ...]:
        """Render only engine-owned, typed scheduler options.

        The execution plan intentionally has no arbitrary ``extra_sbatch_args``;
        every option that can alter execution identity is emitted from a typed
        value here.
        """

        args = [
            f"--comment={self.comment}",
            f"--job-name=o2p-{self.identity.plan_sha256[:10]}-{self.identity.stage_id}-a{self.identity.attempt:03d}",
            f"--partition={self.resources.partition}",
            f"--cpus-per-task={self.resources.cpus}",
            f"--mem={self.resources.memory_mb}M",
            f"--time={self.resources.time_limit}",
            f"--output={self.stdout_pattern}",
            f"--error={self.stderr_pattern}",
        ]
        if self.task_indices:
            indices = ",".join(str(index) for index in self.task_indices)
            args.append(f"--array={indices}%{min(self.resources.array_parallelism, len(self.task_indices))}")
        clauses = []
        if self.dependency_job_ids:
            clauses.append(f"{self.dependency_mode}:{':'.join(self.dependency_job_ids)}")
        if self.ordering_job_ids:
            # Slurm ANDs comma-separated clauses, so a prior generation gates
            # this one without weakening the signed DAG relation above.
            clauses.append(f"afterany:{':'.join(self.ordering_job_ids)}")
        if clauses:
            args.append(f"--dependency={','.join(clauses)}")
        if self.begin_delay_seconds:
            args.append(f"--begin=now+{self.begin_delay_seconds}seconds")
        if self.resources.gpus:
            gpu = f"{self.resources.gpu_type}:" if self.resources.gpu_type else ""
            args.append(f"--gres=gpu:{gpu}{self.resources.gpus}")
        if self.resources.account is not None:
            args.append(f"--account={self.resources.account}")
        if self.resources.qos is not None:
            args.append(f"--qos={self.resources.qos}")
        if self.resources.constraint is not None:
            args.append(f"--constraint={self.resources.constraint}")
        if self.resources.exclude_nodes:
            args.append(f"--exclude={','.join(self.resources.exclude_nodes)}")
        if self.resources.licenses:
            args.append(f"--licenses={','.join(self.resources.licenses)}")
        return tuple(args)


@dataclass(frozen=True)
class SlurmJob:
    """One scheduler job discovered by its authenticated comment."""

    job_id: str
    comment: str
    state: str = ""


DEFINITELY_NOT_INVOKED = "DEFINITELY_NOT_INVOKED"
DEFINITELY_REJECTED = "DEFINITELY_REJECTED"
ACCEPTED = "ACCEPTED"
INVOKED_OUTCOME_UNKNOWN = "INVOKED_OUTCOME_UNKNOWN"


@dataclass(frozen=True)
class SubmitOutcome:
    """Explicit scheduler-boundary outcome with no ambiguous success coercion."""

    status: str
    job_id: str | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""

    def __post_init__(self) -> None:
        """Require evidence consistent with the claimed invocation outcome."""

        allowed = {DEFINITELY_NOT_INVOKED, DEFINITELY_REJECTED, ACCEPTED, INVOKED_OUTCOME_UNKNOWN}
        if self.status not in allowed:
            raise ValueError("submission outcome status is invalid")
        if self.job_id is not None and (type(self.job_id) is not str or not self.job_id.isdigit()):
            raise ValueError("submission outcome job_id must be a numeric string or None")
        if self.returncode is not None and type(self.returncode) is not int:
            raise ValueError("submission outcome returncode must be an integer or None")
        if type(self.stdout) is not str or type(self.stderr) is not str:
            raise ValueError("submission outcome stdout/stderr must be strings")
        if self.status == ACCEPTED and self.job_id is None:
            raise ValueError("ACCEPTED submission outcome requires a parsed job ID")
        if self.status != ACCEPTED and self.job_id is not None:
            raise ValueError("only ACCEPTED submission outcome may carry a job ID")
        if self.status == DEFINITELY_REJECTED and (self.returncode is None or self.returncode == 0):
            raise ValueError("DEFINITELY_REJECTED requires a proven nonzero sbatch return code")
        if self.status == DEFINITELY_NOT_INVOKED and self.returncode is not None:
            raise ValueError("preparation failure is not an sbatch return code")


@dataclass(frozen=True)
class SlurmTaskState:
    """The accounting state for an array element or the root job."""

    array_index: int | None
    state: str
    exit_code: int | None = None

    def normalized_state(self) -> str:
        """Remove Slurm suffixes such as ``CANCELLED by 123`` or ``+``."""

        return self.state.split()[0].rstrip("+").upper()


@dataclass(frozen=True)
class ReceiptObservation:
    """Observed state of one expected pipeline receipt.

    ``trustworthy=False`` means the backend could not distinguish absence from
    an observation failure (for example, a transport or parser error).  Such an
    observation is transient operational state and must never be persisted as
    evidence that a receipt is missing.
    """

    path: str
    exists: bool
    sha256: str | None = None
    trustworthy: bool = True
    error: str | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous combinations before reconciliation consumes them."""

        if not isinstance(self.path, str) or not self.path:
            raise ValueError("receipt observation path must be non-empty text")
        if not isinstance(self.exists, bool) or not isinstance(self.trustworthy, bool):
            raise ValueError("receipt observation flags must be booleans")
        if self.sha256 is not None:
            _validate_sha256(self.sha256, "receipt observation sha256")
        if self.exists and self.sha256 is None:
            raise ValueError("an existing receipt observation must include its SHA-256")
        if self.trustworthy:
            if self.error is not None:
                raise ValueError("a trustworthy receipt observation cannot include an error")
        else:
            if self.exists or self.sha256 is not None:
                raise ValueError("an untrustworthy observation cannot assert receipt bytes")
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError("an untrustworthy observation must explain the observation failure")


@dataclass(frozen=True)
class SubmissionRecord:
    """Immutable record binding one plan attempt to one scheduler-proven job.

    ``recovered`` means the job ID was confirmed by the exact Slurm comment
    query before publication.  It intentionally does not encode whether this
    particular caller received the original ``sbatch`` response, because that
    caller-local observation would make concurrent immutable receipts differ.
    """

    identity: SubmissionIdentity
    job_id: str
    task_ids: tuple[str, ...]
    task_indices: tuple[int, ...]
    recovered: bool
    dependency_mode: str
    dependency_job_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate durable submission evidence independently of its producer."""

        if type(self.identity) is not SubmissionIdentity:
            raise ValueError("submission record identity must be SubmissionIdentity")
        if not self.job_id.isdigit():
            raise ValueError("submission record job_id must be numeric")
        if not isinstance(self.task_ids, tuple) or not self.task_ids:
            raise ValueError("submission record task_ids must be a non-empty tuple")
        for task_id in self.task_ids:
            _validate_identifier(task_id, "submission task_id")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("submission record task_ids cannot contain duplicates")
        if not isinstance(self.task_indices, tuple) or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in self.task_indices
        ):
            raise ValueError("submission record task_indices must be nonnegative integers")
        if self.task_indices and len(self.task_indices) != len(self.task_ids):
            raise ValueError("array submission must bind one index to every task")
        if len(set(self.task_indices)) != len(self.task_indices):
            raise ValueError("submission record task_indices cannot contain duplicates")
        if not isinstance(self.recovered, bool):
            raise ValueError("submission record recovered must be a boolean")
        if self.dependency_mode not in {"afterany", "afterok"}:
            raise ValueError("submission record dependency_mode is invalid")
        if not isinstance(self.dependency_job_ids, tuple) or any(
            not item.isdigit() for item in self.dependency_job_ids
        ):
            raise ValueError("submission record dependency_job_ids must be numeric strings")
        if len(set(self.dependency_job_ids)) != len(self.dependency_job_ids):
            raise ValueError("submission record dependency_job_ids cannot contain duplicates")

    def to_dict(self) -> dict[str, Any]:
        """Return the immutable JSON payload written after submit/recovery."""

        return {
            "attempt": self.identity.attempt,
            "comment": self.identity.comment,
            "dependency_job_ids": list(self.dependency_job_ids),
            "dependency_mode": self.dependency_mode,
            "job_id": self.job_id,
            "plan_sha256": self.identity.plan_sha256,
            "recovered": self.recovered,
            "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
            "stage_id": self.identity.stage_id,
            "task_ids": list(self.task_ids),
            "task_indices": list(self.task_indices),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SubmissionRecord:
        """Load a previously authenticated submission record."""

        allowed = {
            "attempt",
            "comment",
            "dependency_job_ids",
            "dependency_mode",
            "job_id",
            "plan_sha256",
            "recovered",
            "schema_version",
            "stage_id",
            "task_ids",
            "task_indices",
        }
        exact_object(value, allowed, "submission record")
        if exact_int(value["schema_version"], "submission record schema_version") != EXECUTION_RECEIPT_SCHEMA_VERSION:
            raise ValueError("submission record has unsupported fields or schema")
        identity = SubmissionIdentity(
            exact_str(value["plan_sha256"], "submission record plan_sha256"),
            exact_str(value["stage_id"], "submission record stage_id"),
            exact_int(value["attempt"], "submission record attempt"),
        )
        if exact_str(value["comment"], "submission record comment") != identity.comment:
            raise ValueError("submission record comment does not match its identity")
        job_id = exact_str(value["job_id"], "submission record job_id")
        if not job_id.isdigit():
            raise ValueError("submission record job_id must be numeric")
        task_ids = exact_list(value["task_ids"], "submission record task_ids")
        task_indices = exact_list(value["task_indices"], "submission record task_indices")
        dependency_ids = exact_list(value["dependency_job_ids"], "submission record dependency_job_ids")
        return cls(
            identity=identity,
            job_id=job_id,
            task_ids=tuple(exact_str(item, "submission record task_id") for item in task_ids),
            task_indices=tuple(exact_int(item, "submission record task_index") for item in task_indices),
            recovered=exact_bool(value["recovered"], "submission record recovered"),
            dependency_mode=exact_str(value["dependency_mode"], "submission record dependency_mode"),
            dependency_job_ids=tuple(exact_str(item, "submission record dependency job_id") for item in dependency_ids),
        )


@dataclass(frozen=True)
class SubmissionRejectionRecord:
    """Immutable proof that Slurm definitively rejected one exact attempt.

    A rejection is materially different from an uncertain transport outcome:
    Slurm returned a response without a job ID, so no hidden accepted job needs
    to be recovered.  Persisting that distinction prevents an intent from being
    misreported as uncertain forever and can authorize the next bounded attempt.
    """

    identity: SubmissionIdentity
    task_ids: tuple[str, ...]
    task_indices: tuple[int, ...]
    returncode: int
    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        """Validate scheduler rejection evidence before durable publication."""

        if type(self.identity) is not SubmissionIdentity:
            raise ValueError("submission rejection identity must be SubmissionIdentity")
        if not isinstance(self.task_ids, tuple) or not self.task_ids:
            raise ValueError("submission rejection task_ids must be a non-empty tuple")
        for task_id in self.task_ids:
            _validate_identifier(task_id, "submission rejection task_id")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("submission rejection task_ids cannot contain duplicates")
        if not isinstance(self.task_indices, tuple) or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in self.task_indices
        ):
            raise ValueError("submission rejection task_indices must be nonnegative integers")
        if self.task_indices and len(self.task_indices) != len(self.task_ids):
            raise ValueError("array rejection must bind one index to every task")
        if len(set(self.task_indices)) != len(self.task_indices):
            raise ValueError("submission rejection task_indices cannot contain duplicates")
        if type(self.returncode) is not int or self.returncode == 0:
            raise ValueError("submission rejection requires a proven nonzero sbatch return code")
        for field_name in ("stdout", "stderr"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise ValueError(f"submission rejection {field_name} must be text")

    def to_dict(self) -> dict[str, Any]:
        """Return the strict wire representation used for retry authorization."""

        return {
            "attempt": self.identity.attempt,
            "comment": self.identity.comment,
            "decision": "RETRY_SUBMISSION",
            "plan_sha256": self.identity.plan_sha256,
            "returncode": self.returncode,
            "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
            "stage_id": self.identity.stage_id,
            "stderr": self.stderr,
            "stdout": self.stdout,
            "task_ids": list(self.task_ids),
            "task_indices": list(self.task_indices),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SubmissionRejectionRecord:
        """Strictly decode a definitive scheduler-rejection receipt."""

        allowed = {
            "attempt",
            "comment",
            "decision",
            "plan_sha256",
            "returncode",
            "schema_version",
            "stage_id",
            "stderr",
            "stdout",
            "task_ids",
            "task_indices",
        }
        exact_object(value, allowed, "submission rejection")
        schema_version = exact_int(value["schema_version"], "submission rejection schema_version")
        if schema_version != EXECUTION_RECEIPT_SCHEMA_VERSION:
            raise ValueError("submission rejection has unsupported fields or schema")
        if exact_str(value["decision"], "submission rejection decision") != "RETRY_SUBMISSION":
            raise ValueError("submission rejection decision is invalid")
        identity = SubmissionIdentity(
            exact_str(value["plan_sha256"], "submission rejection plan_sha256"),
            exact_str(value["stage_id"], "submission rejection stage_id"),
            exact_int(value["attempt"], "submission rejection attempt"),
        )
        if exact_str(value["comment"], "submission rejection comment") != identity.comment:
            raise ValueError("submission rejection comment does not match its identity")
        task_ids = exact_list(value["task_ids"], "submission rejection task_ids")
        task_indices = exact_list(value["task_indices"], "submission rejection task_indices")
        return cls(
            identity=identity,
            task_ids=tuple(exact_str(item, "submission rejection task_id") for item in task_ids),
            task_indices=tuple(exact_int(item, "submission rejection task_index") for item in task_indices),
            returncode=exact_int(value["returncode"], "submission rejection returncode"),
            stdout=exact_str(value["stdout"], "submission rejection stdout"),
            stderr=exact_str(value["stderr"], "submission rejection stderr"),
        )


@dataclass(frozen=True)
class SubmissionResult:
    """Coordinator result distinguishing new submit from recovered/replayed work."""

    record: SubmissionRecord
    submitted: bool
    registry_synced: bool


@dataclass(frozen=True)
class TaskAttemptReceipt:
    """Immutable files-as-truth verdict for one task in one attempt."""

    identity: SubmissionIdentity
    task_id: str
    array_index: int | None
    job_id: str
    slurm_state: str
    exit_code: int | None
    receipt_observations: tuple[ReceiptObservation, ...]
    successful: bool
    retryable: bool

    def __post_init__(self) -> None:
        """Require one internally consistent terminal task verdict."""

        if type(self.identity) is not SubmissionIdentity:
            raise ValueError("task receipt identity must be SubmissionIdentity")
        _validate_identifier(self.task_id, "task receipt task_id")
        if self.array_index is not None and (
            isinstance(self.array_index, bool) or not isinstance(self.array_index, int) or self.array_index < 0
        ):
            raise ValueError("task receipt array_index must be nonnegative or None")
        if not self.job_id.isdigit():
            raise ValueError("task receipt job_id must be numeric")
        if not self.slurm_state or "\n" in self.slurm_state or "\r" in self.slurm_state:
            raise ValueError("task receipt slurm_state must be a non-empty single-line value")
        if self.exit_code is not None and (
            isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int) or self.exit_code < 0
        ):
            raise ValueError("task receipt exit_code must be nonnegative or None")
        if not isinstance(self.receipt_observations, tuple) or any(
            type(item) is not ReceiptObservation for item in self.receipt_observations
        ):
            raise ValueError("task receipt observations must contain ReceiptObservation values")
        if any(not item.trustworthy for item in self.receipt_observations):
            raise ValueError("task-attempt evidence cannot contain untrustworthy observations")
        if not isinstance(self.successful, bool) or not isinstance(self.retryable, bool):
            raise ValueError("task receipt successful/retryable flags must be booleans")
        if self.successful and self.retryable:
            raise ValueError("a successful task receipt cannot also be retryable")

    def to_dict(self) -> dict[str, Any]:
        """Return stable task-attempt evidence for durable storage."""

        return {
            "array_index": self.array_index,
            "attempt": self.identity.attempt,
            "comment": self.identity.comment,
            "exit_code": self.exit_code,
            "job_id": self.job_id,
            "plan_sha256": self.identity.plan_sha256,
            "receipts": [
                {"exists": item.exists, "path": item.path, "sha256": item.sha256} for item in self.receipt_observations
            ],
            "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
            "slurm_state": self.slurm_state,
            "stage_id": self.identity.stage_id,
            "successful": self.successful,
            "retryable": self.retryable,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskAttemptReceipt:
        """Strictly decode immutable task evidence written by any engine process."""

        allowed = {
            "array_index",
            "attempt",
            "comment",
            "exit_code",
            "job_id",
            "plan_sha256",
            "receipts",
            "retryable",
            "schema_version",
            "slurm_state",
            "stage_id",
            "successful",
            "task_id",
        }
        exact_object(value, allowed, "task-attempt receipt")
        if exact_int(value["schema_version"], "task-attempt schema_version") != EXECUTION_RECEIPT_SCHEMA_VERSION:
            raise ValueError("task-attempt receipt has unsupported fields or schema")
        identity = SubmissionIdentity(
            exact_str(value["plan_sha256"], "task-attempt plan_sha256"),
            exact_str(value["stage_id"], "task-attempt stage_id"),
            exact_int(value["attempt"], "task-attempt attempt"),
        )
        if exact_str(value["comment"], "task-attempt comment") != identity.comment:
            raise ValueError("task-attempt receipt comment does not match its identity")
        receipts = exact_list(value["receipts"], "task-attempt receipts")
        observations: list[ReceiptObservation] = []
        for item in receipts:
            observation = exact_object(item, {"exists", "path", "sha256"}, "task-attempt observation")
            sha256 = observation["sha256"]
            if sha256 is not None:
                sha256 = exact_str(sha256, "task-attempt observation sha256")
            observations.append(
                ReceiptObservation(
                    exact_str(observation["path"], "task-attempt observation path"),
                    exact_bool(observation["exists"], "task-attempt observation exists"),
                    sha256,
                )
            )
        array_index = value["array_index"]
        if array_index is not None:
            array_index = exact_int(array_index, "task-attempt array_index")
        exit_code = value["exit_code"]
        if exit_code is not None:
            exit_code = exact_int(exit_code, "task-attempt exit_code")
        return cls(
            identity=identity,
            task_id=exact_str(value["task_id"], "task-attempt task_id"),
            array_index=array_index,
            job_id=exact_str(value["job_id"], "task-attempt job_id"),
            slurm_state=exact_str(value["slurm_state"], "task-attempt slurm_state"),
            exit_code=exit_code,
            receipt_observations=tuple(observations),
            successful=exact_bool(value["successful"], "task-attempt successful"),
            retryable=exact_bool(value["retryable"], "task-attempt retryable"),
        )


@dataclass(frozen=True)
class RegistryUpdate:
    """A coalescible registry update plus exact claims its outbox can repair."""

    plan_sha256: str
    stage_id: str
    stage_status: str
    execution_status: str
    job_ids: tuple[str, ...]
    attempt: int
    lifecycle_claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate current-state metadata before it can rewrite run.json."""

        _validate_sha256(self.plan_sha256, "registry plan_sha256")
        _validate_identifier(self.stage_id, "registry stage_id")
        allowed_stage_statuses = {
            "SUBMITTED",
            "RETRYING",
            RECONCILE_WAIT,
            RECONCILE_COMPLETE,
            RECONCILE_FAILED,
            RECONCILE_RETRY,
            RECONCILE_RETRY_SUBMITTED,
        }
        if self.stage_status not in allowed_stage_statuses:
            raise ValueError("unsupported registry stage_status")
        if self.execution_status not in {"ACTIVE", "SUBMITTED", "RUNNING", "RETRYING", "COMPLETED", "FAILED"}:
            raise ValueError("unsupported registry execution_status")
        if not isinstance(self.job_ids, tuple) or any(not item.isdigit() for item in self.job_ids):
            raise ValueError("registry job_ids must be an immutable tuple of numeric strings")
        if len(set(self.job_ids)) != len(self.job_ids):
            raise ValueError("registry job_ids cannot contain duplicates")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("registry attempt must be a positive integer")
        if not isinstance(self.lifecycle_claim_ids, tuple) or any(
            re.fullmatch(r"[0-9a-f]{64}-[0-9a-f]{64}", item) is None for item in self.lifecycle_claim_ids
        ):
            raise ValueError("registry lifecycle_claim_ids must contain exact holder identities")
        if len(set(self.lifecycle_claim_ids)) != len(self.lifecycle_claim_ids):
            raise ValueError("registry lifecycle_claim_ids cannot contain duplicates")
        object.__setattr__(self, "job_ids", tuple(sorted(self.job_ids, key=int)))
        object.__setattr__(self, "lifecycle_claim_ids", tuple(sorted(set(self.lifecycle_claim_ids))))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic registry update payload."""

        return {
            "attempt": self.attempt,
            "execution_status": self.execution_status,
            "job_ids": list(self.job_ids),
            "lifecycle_claim_ids": list(self.lifecycle_claim_ids),
            "plan_sha256": self.plan_sha256,
            "stage_id": self.stage_id,
            "stage_status": self.stage_status,
        }

    @property
    def event_id(self) -> str:
        """Return a stable ID useful in warnings and audit logs."""

        payload = json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ReconcileResult:
    """One stage reconciliation decision and any retry it launched."""

    decision: str
    stage_id: str
    attempt: int
    successful_task_ids: tuple[str, ...]
    retry_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]
    active_task_ids: tuple[str, ...]
    retry_submission: SubmissionRecord | None = None
    registry_synced: bool = True


class SubmissionUncertain(RuntimeError):
    """Raised when sbatch may have accepted work but no job can yet be found."""


class SubmissionRejected(RuntimeError):
    """Raised when Slurm definitively rejected an attempt without creating a job."""


class DuplicateSubmissionError(RuntimeError):
    """Raised when more than one Slurm job has the same execution identity."""


__all__ = [
    "ACCEPTED",
    "ACTIVE_SLURM_STATES",
    "DEFINITELY_NOT_INVOKED",
    "DEFINITELY_REJECTED",
    "DuplicateSubmissionError",
    "PlannedTask",
    "ReceiptObservation",
    "ReconcileResult",
    "RegistryUpdate",
    "SlurmJob",
    "SlurmTaskState",
    "SubmissionIdentity",
    "SubmissionRecord",
    "SubmissionRejectionRecord",
    "SubmissionRejected",
    "SubmissionResult",
    "SubmissionRequest",
    "SubmissionUncertain",
    "SubmitOutcome",
    "INVOKED_OUTCOME_UNKNOWN",
    "SUCCESS_SLURM_STATES",
    "TERMINAL_SLURM_STATES",
    "TaskAttemptReceipt",
    "canonical_json",
    "RECONCILE_COMPLETE",
    "RECONCILE_FAILED",
    "RECONCILE_RETRY",
    "RECONCILE_RETRY_SUBMITTED",
    "RECONCILE_WAIT",
]
