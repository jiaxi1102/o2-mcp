"""Monotonic per-stage/attempt registry outbox helpers.

Registry synchronization is deliberately separated from Slurm truth.  These
pure functions merge repeated callbacks for one stage attempt so a delayed
``SUBMITTED`` observation cannot replace ``FAILED`` or discard job IDs while a
compare-and-swap backend provides the cross-process lock boundary.
"""

from __future__ import annotations

import json

from o2mcp.runorg.execution_models import RegistryUpdate

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

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is malformed") from exc
    expected = {
        "attempt",
        "execution_status",
        "job_ids",
        "plan_sha256",
        "stage_id",
        "stage_status",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} has unsupported fields")
    if not isinstance(value["attempt"], int) or isinstance(value["attempt"], bool):
        raise ValueError(f"{label} attempt must be an integer")
    if not isinstance(value["job_ids"], list) or any(not isinstance(item, str) for item in value["job_ids"]):
        raise ValueError(f"{label} job_ids must be an array of strings")
    for field in ("execution_status", "plan_sha256", "stage_id", "stage_status"):
        if not isinstance(value[field], str):
            raise ValueError(f"{label} {field} must be text")
    return RegistryUpdate(
        plan_sha256=value["plan_sha256"],
        stage_id=value["stage_id"],
        stage_status=value["stage_status"],
        execution_status=value["execution_status"],
        job_ids=tuple(value["job_ids"]),
        attempt=value["attempt"],
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
    )


__all__ = ["decode_registry_update", "merge_registry_updates"]
