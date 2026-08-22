"""Generic HPC run-organization: register → promote/archive across storage tiers.

A *run* is a self-describing, relocatable unit with an explicit lifecycle mapped
onto three storage tiers (scratch → group → standby). This subpackage provides the
project-agnostic engine — :class:`RunLayout`, :class:`RunManifest`, the keep/sweep
classifier, the JSONL registry, the shell *planners*, and the :class:`O2Runs`
executor — parameterized by a project-supplied :class:`RunPolicy` (taxonomy:
pipelines, markers, view/heavy suffixes, run-skeleton subdirs, archive excludes).

Everything here is stdlib-only and importable on the Python-3.9 core path. The MCP
tool wrappers live in :mod:`o2mcp.runorg.tools` (they need the optional ``mcp``
dependency) and are registered by a consumer onto its own FastMCP server.
"""

from __future__ import annotations

from o2mcp.runorg.execution_backend import ExecutionBackend, O2ExecutionBackend
from o2mcp.runorg.execution_engine import ExecutionEngine, RegistrySynchronizer
from o2mcp.runorg.execution_models import (
    DuplicateSubmissionError,
    ReceiptObservation,
    ReconcileResult,
    RegistryUpdate,
    SlurmJob,
    SlurmTaskState,
    SubmissionIdentity,
    SubmissionRecord,
    SubmissionRejected,
    SubmissionRejectionRecord,
    SubmissionRequest,
    SubmissionResult,
    SubmissionUncertain,
    SubmitOutcome,
    TaskAttemptReceipt,
)
from o2mcp.runorg.executor import O2RunRegistrySynchronizer, O2Runs, RunPreparationError, TransitionPlan
from o2mcp.runorg.plans import (
    DEPENDENCY_MODES,
    EXECUTION_PLAN_SCHEMA_VERSION,
    RETRYABLE_SLURM_STATES,
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
from o2mcp.runorg.policy import (
    DEFAULT_HEAVY_SLUG_MARKERS,
    DEFAULT_HEAVY_THRESHOLD_BYTES,
    DEFAULT_KEEP_MARKERS,
    GENERIC_POLICY,
    RunPolicy,
)
from o2mcp.runorg.prepared import PreparedRunIdentity
from o2mcp.runorg.runs import (
    RETENTION_AUTO,
    RETENTION_KEEP,
    RETENTION_SWEEP,
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
    STATUS_KEPT,
    STATUS_PURGED,
    VALID_STATUSES,
    RunLayout,
    RunManifest,
    campaign_of,
    classify_run,
    is_regenerable_intermediate,
    migration_target,
    parse_registry,
    parse_run_id,
    plan_archive_script,
    plan_gc_candidates_command,
    plan_promote_script,
    plan_register_commands,
    registry_line,
    variant_of,
)

__all__ = [
    "O2Runs",
    "O2RunRegistrySynchronizer",
    "RunPreparationError",
    "TransitionPlan",
    "PreparedRunIdentity",
    "ExecutionEngine",
    "ExecutionBackend",
    "O2ExecutionBackend",
    "RegistrySynchronizer",
    "SubmissionIdentity",
    "SubmissionRequest",
    "SubmissionRecord",
    "SubmissionRejectionRecord",
    "SubmissionRejected",
    "SubmissionResult",
    "SubmitOutcome",
    "SubmissionUncertain",
    "DuplicateSubmissionError",
    "TaskAttemptReceipt",
    "ReceiptObservation",
    "SlurmJob",
    "SlurmTaskState",
    "ReconcileResult",
    "RegistryUpdate",
    "ExecutionPlan",
    "DatasetIdentity",
    "CanonicalPaths",
    "CommandSpec",
    "ResourceSpec",
    "ReceiptSpec",
    "RetryPolicy",
    "TaskSpec",
    "StageSpec",
    "DEPENDENCY_MODES",
    "EXECUTION_PLAN_SCHEMA_VERSION",
    "RETRYABLE_SLURM_STATES",
    "RunPolicy",
    "GENERIC_POLICY",
    "DEFAULT_KEEP_MARKERS",
    "DEFAULT_HEAVY_SLUG_MARKERS",
    "DEFAULT_HEAVY_THRESHOLD_BYTES",
    "RunLayout",
    "RunManifest",
    "campaign_of",
    "variant_of",
    "classify_run",
    "is_regenerable_intermediate",
    "migration_target",
    "parse_run_id",
    "parse_registry",
    "registry_line",
    "plan_register_commands",
    "plan_promote_script",
    "plan_archive_script",
    "plan_gc_candidates_command",
    "STATUS_ACTIVE",
    "STATUS_KEPT",
    "STATUS_ARCHIVED",
    "STATUS_PURGED",
    "VALID_STATUSES",
    "RETENTION_KEEP",
    "RETENTION_SWEEP",
    "RETENTION_AUTO",
]
