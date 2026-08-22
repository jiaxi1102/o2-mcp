"""Retry-bound ``afterany`` follow-up generation for the execution engine.

Compute retries require fresh scheduler-visible audit jobs.  Keeping that fan-in
outbox protocol separate makes its signed attempt accounting and rejection
recovery reviewable without expanding the central submission coordinator.
"""

from __future__ import annotations

from o2mcp.runorg.execution_evidence import (
    authenticate_followup_authorization,
    latest_reconciliation_receipt,
    next_unrejected_attempt,
    read_strict_json,
    read_submission_rejection,
)
from o2mcp.runorg.execution_models import RECONCILE_COMPLETE, SubmissionIdentity, SubmissionRecord, canonical_json
from o2mcp.runorg.execution_paths import reconciler_followup_path, submission_rejection_path
from o2mcp.runorg.execution_reconcile import signed_attempt_bound
from o2mcp.runorg.execution_rendering import select_tasks
from o2mcp.runorg.plans import ExecutionPlan


class ExecutionFollowupMixin:
    """Generate and recover dependency-bound reconciler submissions."""

    def _ensure_reconciler_followups(
        self,
        plan: ExecutionPlan,
        retried_stage_id: str,
        retry_record: SubmissionRecord,
    ) -> tuple[SubmissionRecord, ...]:
        """Rebind downstream stages to an accepted replacement generation.

        ``afterany`` descendants rebind immediately because Slurm can wait on
        the replacement job regardless of its outcome. ``afterok`` descendants
        rebind only after the replacement dependency has a current authenticated
        completion receipt. Task-bearing descendants rerun all signed tasks
        because their earlier outputs observed stale dependency inputs.
        Definitively rejected generations are consumed and advanced within the
        plan-derived aggregate bound.
        """

        submitted: list[SubmissionRecord] = []
        retried_stage = next(stage for stage in plan.stages if stage.stage_id == retried_stage_id)
        for reconciler in plan.stages:
            if retried_stage_id not in reconciler.depends_on:
                continue
            if reconciler.dependency_mode == "afterok":
                completion = latest_reconciliation_receipt(self.backend, plan, retried_stage)
                if (
                    completion is None
                    or completion.decision != RECONCILE_COMPLETE
                    or completion.attempt < retry_record.identity.attempt
                ):
                    # The accepted replacement is not yet scientifically
                    # certified. Reconciliation calls this helper again after
                    # publishing its current completion receipt.
                    continue
            records = self._submission_records(plan, reconciler)
            # The authorization is a durable outbox item. Replay its exact
            # generation after a crash unless immutable rejection consumed it.
            existing_generation = None
            occupied_generations: list[int] = []
            reconciler_bound = signed_attempt_bound(plan, reconciler)
            for candidate in range(1, reconciler_bound + 1):
                path = reconciler_followup_path(plan, reconciler.stage_id, candidate)
                text = self.backend.read_text(path)
                if text is None:
                    continue
                # Every authorization occupies its immutable attempt identity,
                # even when it belongs to another dependency trigger or its
                # scheduler call was rejected without producing a record.
                authenticate_followup_authorization(
                    self.backend,
                    plan,
                    reconciler,
                    candidate,
                )
                occupied_generations.append(candidate)
                authorization = read_strict_json(self.backend, path, "reconciler follow-up authorization")
                if (
                    authorization.get("plan_sha256") == plan.plan_sha256
                    and authorization.get("stage_id") == reconciler.stage_id
                    and authorization.get("attempt") == candidate
                    and authorization.get("trigger_job_id") == retry_record.job_id
                    and authorization.get("trigger_stage_id") == retried_stage_id
                ):
                    existing_generation = candidate
            if existing_generation is not None:
                identity = SubmissionIdentity(plan.plan_sha256, reconciler.stage_id, existing_generation)
                rejection = read_submission_rejection(
                    self.backend,
                    submission_rejection_path(plan, identity),
                )
                if rejection is None:
                    submitted.append(
                        self.submit_dependency_followup(
                            plan,
                            reconciler.stage_id,
                            attempt=existing_generation,
                        ).record
                    )
                    continue
                expected_task_ids = tuple(task.task_id for task in select_tasks(reconciler, None))
                next_attempt, rejected = next_unrejected_attempt(
                    self.backend,
                    plan,
                    reconciler,
                    existing_generation - 1,
                    expected_task_ids,
                )
                for rejected_identity in rejected:
                    self._release_rejected_invocation_owner(plan, rejected_identity, "")
            else:
                next_attempt = 1
            next_attempt = max(
                next_attempt,
                max(
                    [record.identity.attempt for record in records] + occupied_generations,
                    default=0,
                )
                + 1,
            )
            if next_attempt > reconciler_bound:
                raise ValueError(
                    f"afterany reconciler {reconciler.stage_id} exhausted its signed attempt bound "
                    f"while rebinding retry job {retry_record.job_id}"
                )
            dependency_job_ids = self._dependency_jobs(plan, reconciler)
            authorization = {
                "attempt": next_attempt,
                "dependency_job_ids": list(dependency_job_ids),
                "plan_sha256": plan.plan_sha256,
                "schema_version": 1,
                "stage_id": reconciler.stage_id,
                "trigger_job_id": retry_record.job_id,
                "trigger_stage_id": retried_stage_id,
            }
            self._write_immutable_text(
                plan,
                reconciler_followup_path(plan, reconciler.stage_id, next_attempt),
                canonical_json(authorization),
            )
            submitted.append(
                self.submit_dependency_followup(
                    plan,
                    reconciler.stage_id,
                    attempt=next_attempt,
                ).record
            )
        return tuple(submitted)


__all__ = ["ExecutionFollowupMixin"]
