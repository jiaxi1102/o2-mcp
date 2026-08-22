"""Monotonic per-stage/attempt registry outbox helpers.

Registry synchronization is deliberately separated from Slurm truth.  These
pure functions merge repeated callbacks for one stage attempt so a delayed
``SUBMITTED`` observation cannot replace ``FAILED`` or discard job IDs while a
compare-and-swap backend provides the cross-process lock boundary.
"""

from __future__ import annotations

from o2mcp.runorg.execution_models import RegistryUpdate
from o2mcp.runorg.strict_json import exact_int, exact_list, exact_object, exact_str, strict_json_object

_STAGE_RANK = {
    "SUBMITTED": 10,
    "WAIT": 20,
    "RETRY_MISSING_ONLY": 30,
    "RETRYING": 40,
    "RETRY_SUBMITTED": 50,
    "COMPLETED": 100,
    "FAILED": 110,
}
_EXECUTION_RANK = {
    "ACTIVE": 10,
    "SUBMITTED": 20,
    "RUNNING": 30,
    "RETRYING": 40,
    "COMPLETED": 100,
    "FAILED": 110,
}


def decode_registry_update(text: str, *, label: str = "pending registry update") -> RegistryUpdate:
    """Strictly decode one outbox payload without permissive type coercion."""

    value = strict_json_object(text, label)
    expected = {
        "attempt",
        "execution_status",
        "job_ids",
        "lifecycle_claim_ids",
        "plan_sha256",
        "stage_id",
        "stage_status",
    }
    # Accept pre-ownership outboxes during a rolling upgrade, but continue to
    # reject every unknown field rather than silently weakening the wire schema.
    if "lifecycle_claim_ids" not in value:
        expected.remove("lifecycle_claim_ids")
    exact_object(value, expected, label)
    jobs = exact_list(value["job_ids"], f"{label} job_ids")
    claims = exact_list(value.get("lifecycle_claim_ids", []), f"{label} lifecycle_claim_ids")
    return RegistryUpdate(
        plan_sha256=exact_str(value["plan_sha256"], f"{label} plan_sha256"),
        stage_id=exact_str(value["stage_id"], f"{label} stage_id"),
        stage_status=exact_str(value["stage_status"], f"{label} stage_status"),
        execution_status=exact_str(value["execution_status"], f"{label} execution_status"),
        job_ids=tuple(exact_str(item, f"{label} job_id") for item in jobs),
        attempt=exact_int(value["attempt"], f"{label} attempt"),
        lifecycle_claim_ids=tuple(exact_str(item, f"{label} lifecycle claim_id") for item in claims),
    )


def merge_registry_updates(current: RegistryUpdate | None, incoming: RegistryUpdate) -> RegistryUpdate:
    """Return the monotonic lattice join for one stage/attempt outbox item."""

    if current is None:
        return incoming
    identity = (incoming.plan_sha256, incoming.stage_id, incoming.attempt)
    if (current.plan_sha256, current.stage_id, current.attempt) != identity:
        raise ValueError("registry outbox path contains a different plan/stage/attempt")
    stage_status = max(
        (current.stage_status, incoming.stage_status),
        key=lambda value: _STAGE_RANK[value],
    )
    execution_status = max(
        (current.execution_status, incoming.execution_status),
        key=lambda value: _EXECUTION_RANK[value],
    )
    return RegistryUpdate(
        plan_sha256=incoming.plan_sha256,
        stage_id=incoming.stage_id,
        stage_status=stage_status,
        execution_status=execution_status,
        # Registry JSON is durable provenance, so the lattice join must be
        # byte-deterministic across processes regardless of Python set order.
        job_ids=tuple(sorted(set(current.job_ids) | set(incoming.job_ids), key=int)),
        attempt=incoming.attempt,
        lifecycle_claim_ids=tuple(sorted(set(current.lifecycle_claim_ids) | set(incoming.lifecycle_claim_ids))),
    )


__all__ = ["decode_registry_update", "merge_registry_updates"]
