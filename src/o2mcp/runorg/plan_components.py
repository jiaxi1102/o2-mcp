"""Frozen execution-plan values and strict JSON decoders for the Python-3.9 core."""

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

EXECUTION_PLAN_SCHEMA_VERSION = 1

# Adapters may choose only terminal states the reconciler understands.
RETRYABLE_SLURM_STATES = (
    "BOOT_FAIL",
    "NODE_FAIL",
    "PREEMPTED",
    "REVOKED",
)

# Dependency modes have materially different failure semantics. A normal stage
# waits for successful prerequisites (``afterok``), whereas a
# files-as-truth reconciler uses ``afterany`` to classify partial arrays safely.
DEPENDENCY_MODES = ("afterany", "afterok")

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_TIME_LIMIT_RE = re.compile(r"^(?:\d+-)?\d{1,2}:\d{2}:\d{2}$")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return stable UTF-8 JSON bytes used for every plan digest.

    Sorted keys and compact separators remove insignificant formatting drift.
    ``allow_nan=False`` also rejects non-standard tokens that another JSON
    implementation might hash differently.
    """

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Return ``value`` as a mapping or raise a field-specific error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    """Fail closed when signed JSON contains fields this version cannot interpret.

    Ignoring an unknown field would be dangerous: another launcher could act on
    it even though this implementation omitted it while recomputing the plan
    digest.  Rejection ensures every accepted byte has one shared meaning.
    """

    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unsupported fields: {unknown}")


def _require_sequence(value: Any, field_name: str) -> Sequence[Any]:
    """Return a JSON-array-like value while rejecting strings and mappings."""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array")
    return value


def _require_str(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    """Return a validated string and keep errors close to the source field."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} cannot be empty")
    return value


def _require_int(value: Any, field_name: str) -> int:
    """Return a real integer; booleans are rejected despite Python subclassing."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _is_single_line_string(value: Any) -> bool:
    """Return whether a value is a non-empty string safe for argv or env use."""

    return isinstance(value, str) and bool(value) and not any(char in value for char in ("\x00", "\n", "\r"))


def _validate_identifier(value: str, field_name: str) -> None:
    """Require one portable identity component with no path semantics."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not _IDENTIFIER_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{field_name} {value!r} must match {_IDENTIFIER_RE.pattern!r} and cannot be '.' or '..'")


def _validate_sha256(value: str, field_name: str) -> None:
    """Require a lowercase, full-length SHA-256 hexadecimal digest."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256 digest")


def _validate_absolute_path(path: str, field_name: str) -> None:
    """Require a normalized absolute POSIX path without traversal components."""

    if not _is_single_line_string(path):
        raise ValueError(f"{field_name} must be a single-line NUL-free path")
    if not path.startswith("/"):
        raise ValueError(f"{field_name} must be an absolute POSIX path")
    normalized = posixpath.normpath(path)
    if normalized != path or path == "/":
        raise ValueError(f"{field_name} must be normalized and cannot be the filesystem root")


def _validate_relative_path(path: str, field_name: str) -> None:
    """Require a portable run-relative payload path with no upward traversal."""

    if not _is_single_line_string(path):
        raise ValueError(f"{field_name} must be a single-line NUL-free path")
    if path.startswith("/") or path in {"", ".", ".."}:
        raise ValueError(f"{field_name} must be a non-empty run-relative path")
    normalized = posixpath.normpath(path)
    if normalized != path or normalized.startswith("../"):
        raise ValueError(f"{field_name} must be normalized and cannot traverse above the run root")


def _is_within(path: str, root: str) -> bool:
    """Return whether normalized absolute ``path`` is strictly inside ``root``."""

    try:
        return posixpath.commonpath((path, root)) == root and path != root
    except ValueError:
        # Defensive for hypothetical mixed-drive/path implementations.  POSIX O2
        # paths should never raise here, but false is safer than accepting drift.
        return False


@dataclass(frozen=True)
class DatasetIdentity:
    """One pipeline input dataset bound to its reviewed manifest bytes."""

    dataset_id: str
    manifest_sha256: str
    storage_binding_sha256: str | None = None
    source_uri: str | None = None

    def __post_init__(self) -> None:
        """Validate portable identity and authenticated manifest provenance."""

        if not isinstance(self.dataset_id, str):
            raise ValueError("dataset_id must be a string")
        if not isinstance(self.manifest_sha256, str):
            raise ValueError("manifest_sha256 must be a string")
        _validate_identifier(self.dataset_id, "dataset_id")
        _validate_sha256(self.manifest_sha256, "manifest_sha256")
        if self.storage_binding_sha256 is not None:
            _validate_sha256(self.storage_binding_sha256, "storage_binding_sha256")
        if self.source_uri is not None and (not isinstance(self.source_uri, str) or not self.source_uri):
            raise ValueError("source_uri must be a non-empty string when supplied")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by the plan digest."""

        data: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "manifest_sha256": self.manifest_sha256,
        }
        if self.storage_binding_sha256 is not None:
            data["storage_binding_sha256"] = self.storage_binding_sha256
        if self.source_uri is not None:
            data["source_uri"] = self.source_uri
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DatasetIdentity:
        """Build a dataset identity from decoded JSON with strict field types."""

        data = _require_mapping(value, "dataset")
        _reject_unknown_keys(
            data,
            {"dataset_id", "manifest_sha256", "storage_binding_sha256", "source_uri"},
            "dataset",
        )
        return cls(
            dataset_id=_require_str(data.get("dataset_id"), "dataset.dataset_id"),
            manifest_sha256=_require_str(data.get("manifest_sha256"), "dataset.manifest_sha256"),
            storage_binding_sha256=(
                _require_str(data["storage_binding_sha256"], "dataset.storage_binding_sha256")
                if data.get("storage_binding_sha256") is not None
                else None
            ),
            source_uri=(
                _require_str(data["source_uri"], "dataset.source_uri") if data.get("source_uri") is not None else None
            ),
        )


@dataclass(frozen=True)
class CanonicalPaths:
    """Absolute paths that define one run's execution and result boundaries."""

    run_root: str
    work_root: str
    results_root: str
    receipts_root: str
    logs_root: str
    promotion_root: str | None = None

    def __post_init__(self) -> None:
        """Require normalized roots and keep ephemeral trees inside the run."""

        for field_name in (
            "run_root",
            "work_root",
            "results_root",
            "receipts_root",
            "logs_root",
        ):
            _validate_absolute_path(getattr(self, field_name), field_name)
        if self.promotion_root is not None:
            _validate_absolute_path(self.promotion_root, "promotion_root")

        # Work, receipt, and logs move with the run. Results may use an external
        # content-addressed root; PR3 applies project-specific storage policy.
        for field_name in ("work_root", "receipts_root", "logs_root"):
            path = getattr(self, field_name)
            if not _is_within(path, self.run_root):
                raise ValueError(f"{field_name} must be strictly inside run_root")

        mutable_roots = {
            "work_root": self.work_root,
            "receipts_root": self.receipts_root,
            "logs_root": self.logs_root,
        }
        for left_name, left_path in mutable_roots.items():
            for right_name, right_path in mutable_roots.items():
                if left_name >= right_name:
                    continue
                if posixpath.commonpath((left_path, right_path)) in {left_path, right_path}:
                    raise ValueError(f"{left_name} and {right_name} must be distinct and non-overlapping")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by the plan digest."""

        data: dict[str, Any] = {
            "logs_root": self.logs_root,
            "receipts_root": self.receipts_root,
            "results_root": self.results_root,
            "run_root": self.run_root,
            "work_root": self.work_root,
        }
        if self.promotion_root is not None:
            data["promotion_root"] = self.promotion_root
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CanonicalPaths:
        """Build canonical paths from decoded JSON with strict field types."""

        data = _require_mapping(value, "paths")
        _reject_unknown_keys(
            data,
            {
                "run_root",
                "work_root",
                "results_root",
                "receipts_root",
                "logs_root",
                "promotion_root",
            },
            "paths",
        )
        return cls(
            run_root=_require_str(data.get("run_root"), "paths.run_root"),
            work_root=_require_str(data.get("work_root"), "paths.work_root"),
            results_root=_require_str(data.get("results_root"), "paths.results_root"),
            receipts_root=_require_str(data.get("receipts_root"), "paths.receipts_root"),
            logs_root=_require_str(data.get("logs_root"), "paths.logs_root"),
            promotion_root=(
                _require_str(data["promotion_root"], "paths.promotion_root")
                if data.get("promotion_root") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ResourceSpec:
    """Conservative, serializable Slurm resources for one execution stage."""

    partition: str
    cpus: int
    memory_mb: int
    time_limit: str
    gpus: int = 0
    array_parallelism: int = 1
    account: str | None = None
    qos: str | None = None

    def __post_init__(self) -> None:
        """Bound resources so malformed plans fail before Slurm submission."""

        for field_name in ("cpus", "memory_mb", "gpus", "array_parallelism"):
            if isinstance(getattr(self, field_name), bool) or not isinstance(getattr(self, field_name), int):
                raise ValueError(f"{field_name} must be an integer")
        _validate_identifier(self.partition, "partition")
        if not isinstance(self.time_limit, str):
            raise ValueError("time_limit must be a string")
        if not 1 <= self.cpus <= 1024:
            raise ValueError("cpus must be between 1 and 1024")
        if not 1 <= self.memory_mb <= 16 * 1024 * 1024:
            raise ValueError("memory_mb must be between 1 and 16777216")
        match = _TIME_LIMIT_RE.fullmatch(self.time_limit)
        if not match:
            raise ValueError("time_limit must use Slurm D-HH:MM:SS or HH:MM:SS syntax")
        clock = self.time_limit.rsplit("-", 1)[-1]
        hours, minutes, seconds = (int(component) for component in clock.split(":"))
        if minutes >= 60 or seconds >= 60:
            raise ValueError("time_limit minutes and seconds must each be below 60")
        days = int(self.time_limit.split("-", 1)[0]) if "-" in self.time_limit else 0
        total_seconds = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
        if not 1 <= total_seconds <= 14 * 24 * 60 * 60:
            raise ValueError("time_limit must be positive and no longer than 14 days")
        if not 0 <= self.gpus <= 64:
            raise ValueError("gpus must be between 0 and 64")
        if not 1 <= self.array_parallelism <= 10000:
            raise ValueError("array_parallelism must be between 1 and 10000")
        for field_name in ("account", "qos"):
            value = getattr(self, field_name)
            if value is not None:
                _validate_identifier(value, field_name)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by the plan digest."""

        data: dict[str, Any] = {
            "array_parallelism": self.array_parallelism,
            "cpus": self.cpus,
            "gpus": self.gpus,
            "memory_mb": self.memory_mb,
            "partition": self.partition,
            "time_limit": self.time_limit,
        }
        if self.account is not None:
            data["account"] = self.account
        if self.qos is not None:
            data["qos"] = self.qos
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResourceSpec:
        """Build a resource specification from decoded JSON."""

        data = _require_mapping(value, "resources")
        _reject_unknown_keys(
            data,
            {
                "partition",
                "cpus",
                "memory_mb",
                "time_limit",
                "gpus",
                "array_parallelism",
                "account",
                "qos",
            },
            "resources",
        )
        return cls(
            partition=_require_str(data.get("partition"), "resources.partition"),
            cpus=_require_int(data.get("cpus"), "resources.cpus"),
            memory_mb=_require_int(data.get("memory_mb"), "resources.memory_mb"),
            time_limit=_require_str(data.get("time_limit"), "resources.time_limit"),
            gpus=_require_int(data.get("gpus", 0), "resources.gpus"),
            array_parallelism=_require_int(data.get("array_parallelism", 1), "resources.array_parallelism"),
            account=(_require_str(data["account"], "resources.account") if data.get("account") is not None else None),
            qos=(_require_str(data["qos"], "resources.qos") if data.get("qos") is not None else None),
        )


@dataclass(frozen=True)
class ReceiptSpec:
    """One expected run-relative receipt, optionally bound to known bytes."""

    receipt_id: str
    path: str
    required: bool = True
    sha256: str | None = None

    def __post_init__(self) -> None:
        """Validate receipt identity, location, and optional byte binding."""

        _validate_identifier(self.receipt_id, "receipt_id")
        _validate_relative_path(self.path, "receipt path")
        if not isinstance(self.required, bool):
            raise ValueError("receipt required must be a boolean")
        if self.sha256 is not None:
            _validate_sha256(self.sha256, "receipt sha256")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by the plan digest."""

        data: dict[str, Any] = {
            "path": self.path,
            "receipt_id": self.receipt_id,
            "required": self.required,
        }
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReceiptSpec:
        """Build an expected receipt from decoded JSON."""

        data = _require_mapping(value, "receipt")
        _reject_unknown_keys(data, {"receipt_id", "path", "required", "sha256"}, "receipt")
        required = data.get("required", True)
        if not isinstance(required, bool):
            raise ValueError("receipt.required must be a boolean")
        return cls(
            receipt_id=_require_str(data.get("receipt_id"), "receipt.receipt_id"),
            path=_require_str(data.get("path"), "receipt.path"),
            required=required,
            sha256=(_require_str(data["sha256"], "receipt.sha256") if data.get("sha256") is not None else None),
        )


@dataclass(frozen=True)
class RetryPolicy:
    """A deliberately bounded policy for exact, unchanged task retries."""

    max_attempts: int = 1
    retryable_slurm_states: tuple[str, ...] = ()
    retryable_exit_codes: tuple[int, ...] = ()
    retry_missing_receipts: bool = False
    backoff_seconds: int = 0

    def __post_init__(self) -> None:
        """Reject unbounded or scientifically ambiguous retry behavior."""

        if not isinstance(self.retryable_slurm_states, tuple):
            raise ValueError("retryable_slurm_states must be an immutable tuple")
        if not isinstance(self.retryable_exit_codes, tuple):
            raise ValueError("retryable_exit_codes must be an immutable tuple")
        if any(not isinstance(state, str) for state in self.retryable_slurm_states):
            raise ValueError("retryable_slurm_states entries must be strings")
        for field_name in ("max_attempts", "backoff_seconds"):
            if isinstance(getattr(self, field_name), bool) or not isinstance(getattr(self, field_name), int):
                raise ValueError(f"{field_name} must be an integer")
        if not isinstance(self.retry_missing_receipts, bool):
            raise ValueError("retry_missing_receipts must be a boolean")
        if any(isinstance(code, bool) or not isinstance(code, int) for code in self.retryable_exit_codes):
            raise ValueError("retryable_exit_codes entries must be integers")
        if not 1 <= self.max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        unknown_states = sorted(set(self.retryable_slurm_states) - set(RETRYABLE_SLURM_STATES))
        if unknown_states:
            raise ValueError(f"unsupported retryable Slurm states: {unknown_states}")
        if len(set(self.retryable_slurm_states)) != len(self.retryable_slurm_states):
            raise ValueError("retryable_slurm_states cannot contain duplicates")
        if len(set(self.retryable_exit_codes)) != len(self.retryable_exit_codes):
            raise ValueError("retryable_exit_codes cannot contain duplicates")
        if any(code < 1 or code > 255 for code in self.retryable_exit_codes):
            raise ValueError("retryable_exit_codes must be between 1 and 255")
        if not 0 <= self.backoff_seconds <= 3600:
            raise ValueError("backoff_seconds must be between 0 and 3600")
        if self.max_attempts == 1 and (
            self.retryable_slurm_states
            or self.retryable_exit_codes
            or self.retry_missing_receipts
            or self.backoff_seconds
        ):
            raise ValueError("retry conditions require max_attempts greater than one")

        # Canonicalize semantic sets so equality and hashing agree.
        object.__setattr__(self, "retryable_slurm_states", tuple(sorted(self.retryable_slurm_states)))
        object.__setattr__(self, "retryable_exit_codes", tuple(sorted(self.retryable_exit_codes)))

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by the plan digest."""

        return {
            "backoff_seconds": self.backoff_seconds,
            "max_attempts": self.max_attempts,
            "retry_missing_receipts": self.retry_missing_receipts,
            "retryable_exit_codes": sorted(self.retryable_exit_codes),
            "retryable_slurm_states": sorted(self.retryable_slurm_states),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RetryPolicy:
        """Build a bounded retry policy from decoded JSON."""

        data = _require_mapping(value, "retry_policy")
        _reject_unknown_keys(
            data,
            {
                "max_attempts",
                "retryable_slurm_states",
                "retryable_exit_codes",
                "retry_missing_receipts",
                "backoff_seconds",
            },
            "retry_policy",
        )
        states = _require_sequence(data.get("retryable_slurm_states", []), "retryable_slurm_states")
        exit_codes = _require_sequence(data.get("retryable_exit_codes", []), "retryable_exit_codes")
        missing = data.get("retry_missing_receipts", False)
        if not isinstance(missing, bool):
            raise ValueError("retry_missing_receipts must be a boolean")
        return cls(
            max_attempts=_require_int(data.get("max_attempts", 1), "max_attempts"),
            retryable_slurm_states=tuple(_require_str(state, "retryable_slurm_states[]") for state in states),
            retryable_exit_codes=tuple(_require_int(code, "retryable_exit_codes[]") for code in exit_codes),
            retry_missing_receipts=missing,
            backoff_seconds=_require_int(data.get("backoff_seconds", 0), "backoff_seconds"),
        )


__all__ = [
    "CanonicalPaths",
    "DEPENDENCY_MODES",
    "DatasetIdentity",
    "EXECUTION_PLAN_SCHEMA_VERSION",
    "RETRYABLE_SLURM_STATES",
    "ReceiptSpec",
    "ResourceSpec",
    "RetryPolicy",
]
