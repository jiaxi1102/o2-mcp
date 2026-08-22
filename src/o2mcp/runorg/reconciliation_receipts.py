"""Strict wire contract for immutable files-as-truth reconciliation evidence.

Reconciliation receipts authorize both downstream stages and bounded retries, so
they are control-plane inputs rather than informational JSON.  This module keeps
their structural and task-coverage validation independent of scheduler access;
the execution engine additionally verifies the referenced task-attempt receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from o2mcp.runorg.execution_models import (
    RECONCILE_COMPLETE,
    RECONCILE_FAILED,
    RECONCILE_RETRY,
)
from o2mcp.runorg.plan_components import _validate_identifier, _validate_sha256
from o2mcp.runorg.plan_stages import StageSpec

_STAGE_RECEIPTS_SENTINEL = "__stage_receipts__"
_TERMINAL_DECISIONS = frozenset({RECONCILE_COMPLETE, RECONCILE_FAILED, RECONCILE_RETRY})


def _strict_identifier_list(value: object, field_name: str) -> tuple[str, ...]:
    """Decode one sorted unique identifier array without coercing wire types."""

    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"reconciliation {field_name} must be an array of strings")
    identifiers = tuple(value)
    if identifiers != tuple(sorted(identifiers)) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"reconciliation {field_name} must be sorted and unique")
    for identifier in identifiers:
        if identifier == _STAGE_RECEIPTS_SENTINEL and field_name == "failed_task_ids":
            continue
        _validate_identifier(identifier, f"reconciliation {field_name}[]")
    return identifiers


@dataclass(frozen=True)
class ReconciliationReceipt:
    """One terminal, plan-bound reconciliation decision for a stage attempt."""

    plan_sha256: str
    stage_id: str
    attempt: int
    decision: str
    successful_task_ids: tuple[str, ...]
    retry_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]
    active_task_ids: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Validate intrinsic invariants before plan-specific coverage checks."""

        _validate_sha256(self.plan_sha256, "reconciliation plan_sha256")
        _validate_identifier(self.stage_id, "reconciliation stage_id")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("reconciliation attempt must be a positive integer")
        if self.decision not in _TERMINAL_DECISIONS:
            raise ValueError("reconciliation decision must be terminal")
        if self.schema_version != 1:
            raise ValueError("unsupported reconciliation schema version")
        for field_name in (
            "successful_task_ids",
            "retry_task_ids",
            "failed_task_ids",
            "active_task_ids",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                raise ValueError(f"reconciliation {field_name} must be an immutable tuple")
        if self.active_task_ids:
            raise ValueError("immutable reconciliation evidence cannot contain active tasks")
        categorized = self.successful_task_ids + self.retry_task_ids + self.failed_task_ids
        actual = [item for item in categorized if item != _STAGE_RECEIPTS_SENTINEL]
        if len(actual) != len(set(actual)):
            raise ValueError("a task cannot appear in multiple reconciliation categories")

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        plan_sha256: str,
        stage: StageSpec,
        attempt: int,
    ) -> ReconciliationReceipt:
        """Strictly decode and bind a receipt to its expected plan and task scope."""

        if type(value) is not dict:
            raise ValueError("reconciliation receipt must be a JSON object")
        allowed = {
            "active_task_ids",
            "attempt",
            "decision",
            "failed_task_ids",
            "plan_sha256",
            "retry_task_ids",
            "schema_version",
            "stage_id",
            "successful_task_ids",
        }
        if set(value) != allowed:
            raise ValueError("reconciliation receipt has unsupported or missing fields")
        if type(value["plan_sha256"]) is not str or value["plan_sha256"] != plan_sha256:
            raise ValueError("reconciliation receipt plan SHA mismatch")
        if type(value["stage_id"]) is not str or value["stage_id"] != stage.stage_id:
            raise ValueError("reconciliation receipt stage mismatch")
        if type(value["attempt"]) is not int or value["attempt"] != attempt:
            raise ValueError("reconciliation receipt attempt mismatch")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported reconciliation schema version")
        if type(value["decision"]) is not str:
            raise ValueError("reconciliation decision must be text")

        receipt = cls(
            plan_sha256=value["plan_sha256"],
            stage_id=value["stage_id"],
            attempt=value["attempt"],
            decision=value["decision"],
            successful_task_ids=_strict_identifier_list(value["successful_task_ids"], "successful_task_ids"),
            retry_task_ids=_strict_identifier_list(value["retry_task_ids"], "retry_task_ids"),
            failed_task_ids=_strict_identifier_list(value["failed_task_ids"], "failed_task_ids"),
            active_task_ids=_strict_identifier_list(value["active_task_ids"], "active_task_ids"),
            schema_version=value["schema_version"],
        )
        receipt._validate_stage_coverage(stage)
        return receipt

    def _validate_stage_coverage(self, stage: StageSpec) -> None:
        """Require the decision to classify the exact immutable stage task scope."""

        expected = {task.task_id for task in stage.tasks} or {stage.stage_id}
        actual_failed = set(self.failed_task_ids) - {_STAGE_RECEIPTS_SENTINEL}
        classified = set(self.successful_task_ids) | set(self.retry_task_ids) | actual_failed
        if classified != expected:
            raise ValueError("reconciliation receipt does not cover the exact stage task scope")
        stage_receipts_failed = _STAGE_RECEIPTS_SENTINEL in self.failed_task_ids

        if self.decision == RECONCILE_COMPLETE:
            if set(self.successful_task_ids) != expected or self.retry_task_ids or self.failed_task_ids:
                raise ValueError("completed reconciliation must classify every task as successful")
        elif self.decision == RECONCILE_RETRY:
            if not self.retry_task_ids or self.failed_task_ids:
                raise ValueError("retry reconciliation requires retry tasks and no terminal failures")
        elif self.decision == RECONCILE_FAILED:
            if not self.failed_task_ids or self.retry_task_ids:
                raise ValueError("failed reconciliation requires terminal failure evidence")
            if stage_receipts_failed and set(self.successful_task_ids) != expected:
                raise ValueError("stage-receipt failure is valid only after all tasks succeeded")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical mapping written by the execution engine."""

        return {
            "active_task_ids": list(self.active_task_ids),
            "attempt": self.attempt,
            "decision": self.decision,
            "failed_task_ids": list(self.failed_task_ids),
            "plan_sha256": self.plan_sha256,
            "retry_task_ids": list(self.retry_task_ids),
            "schema_version": self.schema_version,
            "stage_id": self.stage_id,
            "successful_task_ids": list(self.successful_task_ids),
        }


__all__ = ["ReconciliationReceipt"]
