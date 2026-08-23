"""Idempotent submission and files-as-truth reconciliation for O2 plans.

The engine separates immutable plans from the execution backend. Its safety
properties are:

* one plan/stage/attempt maps to one reversible Slurm comment;
* a missing ``sbatch`` response is queried, never blindly resubmitted;
* immutable attempt receipts prove completed tasks and exclude them from retries;
* retries are bounded by the signed stage policy; and
* registry failures are recorded as pending metadata work, not reported as a
  failed submission that could tempt a caller to duplicate live jobs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from o2mcp.runorg.execution_authorization import derive_attempt_authorization
from o2mcp.runorg.execution_backend import ExecutionBackend, O2ExecutionBackend, receipt_matches, text_sha256
from o2mcp.runorg.execution_evidence import (
    authenticated_task_verdict,
    current_task_receipts_status,
    current_task_receipts_valid,
    latest_followup_attempt,
    latest_reconciliation_receipt,
    next_unrejected_attempt,
    read_plan_submission_records,
    read_submission_invocation_claim_id,
    read_submission_rejection,
)
from o2mcp.runorg.execution_followups import ExecutionFollowupMixin
from o2mcp.runorg.execution_models import (
    ACCEPTED,
    DEFINITELY_NOT_INVOKED,
    DEFINITELY_REJECTED,
    INVOKED_OUTCOME_UNKNOWN,
    RECONCILE_COMPLETE,
    RECONCILE_FAILED,
    RECONCILE_RETRY,
    RECONCILE_RETRY_SUBMITTED,
    RECONCILE_WAIT,
    SUCCESS_SLURM_STATES,
    TERMINAL_SLURM_STATES,
    DuplicateSubmissionError,
    PlannedTask,
    ReconcileResult,
    RegistryUpdate,
    SlurmTaskState,
    SubmissionIdentity,
    SubmissionRecord,
    SubmissionRejected,
    SubmissionRejectionRecord,
    SubmissionResult,
    SubmissionUncertain,
    TaskAttemptReceipt,
    canonical_json,
)
from o2mcp.runorg.execution_paths import (
    reconciliation_path,
    submission_intent_path,
    submission_invocation_path,
    submission_record_path,
    submission_rejection_path,
    task_attempt_path,
)
from o2mcp.runorg.execution_reconcile import is_retryable, signed_attempt_bound, stage_by_id, task_state
from o2mcp.runorg.execution_registry import ExecutionRegistryMixin
from o2mcp.runorg.execution_rendering import build_submission_request, select_tasks, submission_intent
from o2mcp.runorg.plan_stages import StageSpec
from o2mcp.runorg.plans import ExecutionPlan
from o2mcp.runorg.reconciliation_receipts import ReconciliationReceipt
from o2mcp.runorg.strict_json import strict_json_object


class RegistrySynchronizer(Protocol):
    """Consumer of current execution state for ``run.json`` and registry JSONL."""

    def validate_plan(self, plan: ExecutionPlan) -> bool:
        """Authenticate the registered run identity before any scheduler write."""

    def synchronize(self, plan: ExecutionPlan, update: RegistryUpdate) -> bool:
        """Persist ``update`` and return whether both registry surfaces agree."""


class ExecutionEngine(ExecutionRegistryMixin, ExecutionFollowupMixin):
    """Coordinate exact plan attempts over a fakeable scheduler boundary."""

    def __init__(self, backend: ExecutionBackend, registry: RegistrySynchronizer | None = None) -> None:
        if isinstance(backend, O2ExecutionBackend) and registry is None:
            raise ValueError("production O2 execution requires a run.json/registry synchronizer")
        self.backend = backend
        self.registry = registry

    def submit_stage(
        self,
        plan: ExecutionPlan,
        stage_id: str,
        *,
        attempt: int = 1,
        task_ids: Sequence[str] | None = None,
    ) -> SubmissionResult:
        """Retain ownership after uncertain calls; release proven safe failures."""

        # Validate the caller-controlled identity before creating a durable
        # lifecycle holder. Invalid operations have no evidence path through
        # which a later replay could safely retire an abandoned claim, and every
        # outstanding claim blocks promote and archive, so an unknown stage or an
        # attempt above the signed bound must be refused before acquisition
        # rather than inside it. Only pure checks are hoisted; binding the plan
        # publishes durable evidence and stays behind the claim.
        stage = stage_by_id(plan, stage_id)
        bound = signed_attempt_bound(plan, stage)
        if attempt > bound:
            raise ValueError(f"stage {stage_id} attempt {attempt} exceeds signed max_attempts bound={bound}")
        identity = SubmissionIdentity(plan.plan_sha256, stage_id, attempt)
        operation_id = f"submit:{identity.plan_sha256}:{identity.stage_id}:{identity.attempt}"
        claim_id = self._acquire_lifecycle(plan, operation_id)
        try:
            result = self._submit_stage_claimed(
                plan,
                stage_id,
                attempt=attempt,
                task_ids=task_ids,
                claim_id=claim_id,
            )
        except SubmissionRejected:
            self._release_lifecycle(plan, claim_id)
            raise
        except SubmissionUncertain:
            raise
        except Exception:
            self._release_if_not_invocation_owner(plan, identity, claim_id)
            raise
        return result

    def _submit_stage_claimed(
        self,
        plan: ExecutionPlan,
        stage_id: str,
        *,
        attempt: int = 1,
        task_ids: Sequence[str] | None = None,
        claim_id: str = "",
    ) -> SubmissionResult:
        """Submit once, or recover by exact comment without marker takeover."""

        self._bind_plan(plan)
        stage = stage_by_id(plan, stage_id)
        attempt_bound = signed_attempt_bound(plan, stage)
        if attempt > attempt_bound:
            raise ValueError(f"stage {stage_id} attempt {attempt} exceeds signed max_attempts bound={attempt_bound}")
        identity = SubmissionIdentity(plan.plan_sha256, stage_id, attempt)
        existing_record = next(
            (record for record in self._submission_records(plan, stage) if record.identity.attempt == attempt),
            None,
        )
        if existing_record is None:
            # Certification applies to creating a generation, not to recovering
            # one.  A recorded attempt is authenticated against its historical
            # intent and dependencies below; re-certifying it against current
            # state would make an interrupted submission unreplayable as soon as
            # a prerequisite advances, stranding its registry outbox and
            # lifecycle claim.
            self._validate_afterok_dependencies(plan, stage)
            authorized_task_ids, authorized_dependencies = derive_attempt_authorization(
                self.backend,
                plan,
                stage,
                attempt,
                self._dependency_jobs(plan, stage),
            )
            # Another coordinator can accept a replacement prerequisite between
            # the stage gate above and this resolution, so certification is
            # bound to the exact jobs about to enter the intent.
            self._validate_afterok_dependency_jobs(plan, stage, authorized_dependencies)
        else:
            # The strict record reader already proved each historical dependency
            # job belongs to the corresponding signed prerequisite.  Preserve
            # those exact historical IDs even if that prerequisite later retried.
            authorized_task_ids = existing_record.task_ids
            authorized_dependencies = existing_record.dependency_job_ids
        requested_task_ids = authorized_task_ids if task_ids is None else tuple(task_ids)
        if requested_task_ids != authorized_task_ids:
            raise ValueError("requested task set differs from the signed attempt authorization")
        selected = select_tasks(stage, authorized_task_ids)
        record_path = submission_record_path(plan, identity)
        expected_task_ids = tuple(task.task_id for task in selected)
        expected_task_indices = tuple(task.array_index for task in selected if task.array_index is not None)
        if existing_record is not None:
            replay_request = build_submission_request(
                plan,
                stage,
                identity,
                selected,
                authorized_dependencies,
                self._prior_generation_jobs(plan, stage, attempt),
            )
            expected_intent = canonical_json(submission_intent(replay_request))
            recorded_intent = self.backend.read_text(submission_intent_path(plan, identity))
            if recorded_intent != expected_intent:
                raise ValueError("submission record is missing its exact immutable pre-submit intent")
            scheduler_job = self._unique_existing_job(identity)
            if scheduler_job is not None and scheduler_job != existing_record.job_id:
                raise DuplicateSubmissionError(
                    f"recorded job {existing_record.job_id} disagrees with scheduler job {scheduler_job}"
                )
            # A crash can occur after the immutable record is published but
            # before an outbox item exists.  Reconstruct the exact update from
            # authenticated evidence; merely draining existing outbox files
            # would leave run.json unaware of an accepted live job forever.
            status = "RETRYING" if attempt > 1 else "SUBMITTED"
            update = RegistryUpdate(
                plan_sha256=plan.plan_sha256,
                stage_id=stage_id,
                stage_status=status,
                execution_status=status,
                job_ids=self._all_recorded_job_ids(plan),
                attempt=attempt,
            )
            invocation_claim_id = read_submission_invocation_claim_id(
                self.backend,
                plan,
                existing_record.identity,
            )
            if invocation_claim_id is not None:
                self._release_preinvocation_losers(plan, existing_record.identity, invocation_claim_id)
            claim_ids = self._submission_claim_ids(
                plan,
                existing_record.identity,
                claim_id,
                invocation_claim_id,
            )
            self._enqueue_registry_update(plan, update, claim_ids)
            # Replay converges the follow-up only after the accepted compute job
            # and its lifecycle holders have a durable registry repair pointer.
            if attempt > 1:
                self._ensure_reconciler_followups(plan, stage_id, existing_record)
            synced = self._sync_registry(
                plan,
                update,
                claim_ids,
            )
            return SubmissionResult(existing_record, submitted=False, registry_synced=synced)

        rejection = read_submission_rejection(self.backend, submission_rejection_path(plan, identity))
        if rejection is not None:
            if rejection.identity != identity or rejection.task_ids != expected_task_ids:
                raise ValueError("submission rejection path contains mismatched immutable evidence")
            # Rejection proves that any prior invocation owner is now repairable.
            self._release_rejected_invocation_owner(plan, identity, claim_id)
            raise SubmissionRejected(
                f"Slurm definitively rejected {identity.comment}; use bounded attempt {attempt + 1} when authorized"
            )

        request = build_submission_request(
            plan,
            stage,
            identity,
            selected,
            authorized_dependencies,
            self._prior_generation_jobs(plan, stage, attempt),
        )

        # Intent records the exact scheduler request but does not itself grant
        # submission ownership.  Separating it from the invocation marker lets a
        # new process safely recover when the intent creator died before crossing
        # the scheduler boundary.
        intent_path = submission_intent_path(plan, identity)
        intent_text = canonical_json(submission_intent(request))
        self._write_immutable_text(plan, intent_path, intent_text)

        found = self._unique_existing_job(identity)
        invocation_path = submission_invocation_path(plan, identity)
        invocation_claim_id = read_submission_invocation_claim_id(self.backend, plan, identity)
        submitted_now = False
        if found is None:
            if invocation_claim_id is not None:
                # Another holder owns the only permissible scheduler call. This
                # caller provably did not invoke, so release only its own claim.
                self._release_lifecycle(plan, claim_id)
                raise SubmissionUncertain(
                    f"sbatch invocation for {identity.comment} is already owned but no matching job is visible; "
                    "query again and do not resubmit"
                )
            preparation = self.backend.prepare_submission(request)
            if preparation is not None:
                if preparation.status != DEFINITELY_NOT_INVOKED:
                    raise ValueError("submission preparation returned an invocation outcome")
                # No durable rejection is written: this exact attempt may be
                # prepared again because sbatch was provably never called.
                raise RuntimeError(preparation.stderr.strip() or "submission preparation failed before sbatch")
            invocation = {
                "attempt": identity.attempt,
                "comment": identity.comment,
                "intent_sha256": text_sha256(intent_text),
                "lifecycle_claim_id": claim_id,
                "plan_sha256": plan.plan_sha256,
                "schema_version": 1,
                "stage_id": stage_id,
            }
            try:
                invocation_created = self._write_immutable_text(plan, invocation_path, canonical_json(invocation))
            except RuntimeError:
                invocation_created = False
            if not invocation_created:
                invocation_claim_id = read_submission_invocation_claim_id(self.backend, plan, identity)
                # A lost create response can leave this caller as the durable
                # invocation owner. Release only a distinct losing observer.
                self._release_if_not_invocation_owner(plan, identity, claim_id)
                raise SubmissionUncertain(
                    f"sbatch invocation for {identity.comment} is already owned but no matching job is visible; "
                    "query again and do not resubmit"
                )
            # The marker fences every other pre-marker holder: each loser must
            # observe this owner and abort before crossing sbatch.
            self._release_preinvocation_losers(plan, identity, claim_id)
            try:
                outcome = self.backend.invoke_submission(request)
            except Exception as exc:
                outcome = None
                invocation_error = exc
            else:
                invocation_error = None
            if outcome is not None and outcome.status == ACCEPTED:
                found = outcome.job_id
                submitted_now = True
            else:
                # Even an explicit wrapper outcome is checked against Slurm by
                # the signed comment before classifying the attempt.
                found = self._unique_existing_job(identity)
                if found is None and outcome is not None and outcome.status == DEFINITELY_NOT_INVOKED:
                    invocation_path = submission_invocation_path(plan, identity)
                    invocation_text = canonical_json(invocation)
                    if not self._compare_and_swap_text(plan, invocation_path, invocation_text, None):
                        raise SubmissionUncertain(
                            f"pre-sbatch failure for {identity.comment} could not clear its exact invocation marker"
                        )
                    raise RuntimeError(outcome.stderr.strip() or "submission failed before sbatch invocation")
                if found is None and outcome is not None and outcome.status == DEFINITELY_REJECTED:
                    if outcome.returncode is None:  # guarded by SubmitOutcome; retained for type narrowing
                        raise ValueError("definitive rejection lacks a nonzero sbatch return code")
                    rejection = SubmissionRejectionRecord(
                        identity=identity,
                        task_ids=expected_task_ids,
                        task_indices=expected_task_indices,
                        returncode=outcome.returncode,
                        stdout=outcome.stdout,
                        stderr=outcome.stderr,
                    )
                    rejection_path = submission_rejection_path(plan, identity)
                    rejection_text = canonical_json(rejection.to_dict())
                    try:
                        self._write_immutable_text(plan, rejection_path, rejection_text)
                    except Exception as rejection_error:
                        # A lost write response may still have published the
                        # exact rejection. Authenticate that case before making
                        # the invocation identity reusable.
                        try:
                            persisted_rejection = self.backend.read_text(rejection_path)
                        except Exception as read_error:
                            raise SubmissionUncertain(
                                f"Slurm rejected {identity.comment}, but durable rejection evidence "
                                "cannot be observed; invocation remains owned"
                            ) from read_error
                        if persisted_rejection == rejection_text:
                            raise SubmissionRejected(
                                outcome.stderr.strip() or outcome.stdout.strip() or "sbatch rejected the request"
                            ) from rejection_error
                        if persisted_rejection is not None:
                            raise SubmissionUncertain(
                                f"Slurm rejected {identity.comment}, but its rejection path contains "
                                "conflicting evidence; invocation remains owned"
                            ) from rejection_error
                        # No rejection bytes and a definitive scheduler response
                        # prove no job exists. Clear only this caller's exact
                        # invocation marker; if compare-clear loses a race, keep
                        # the claim and fail uncertain rather than risk a duplicate.
                        invocation_text = canonical_json(invocation)
                        if not self._compare_and_swap_text(
                            plan,
                            invocation_path,
                            invocation_text,
                            None,
                        ):
                            raise SubmissionUncertain(
                                f"Slurm rejected {identity.comment}, but its exact invocation marker "
                                "could not be recovered"
                            ) from rejection_error
                        raise RuntimeError(
                            f"Slurm rejected {identity.comment}, but rejection evidence publication "
                            "failed before becoming durable; retry this same attempt"
                        ) from rejection_error
                    raise SubmissionRejected(
                        outcome.stderr.strip() or outcome.stdout.strip() or "sbatch rejected the request"
                    )
                if found is None:
                    detail = "sbatch invocation outcome is unknown"
                    if invocation_error is not None:
                        detail = str(invocation_error) or detail
                    elif outcome is not None and outcome.status not in {
                        INVOKED_OUTCOME_UNKNOWN,
                        DEFINITELY_REJECTED,
                    }:
                        raise ValueError("backend returned an invalid post-invocation outcome")
                    raise SubmissionUncertain(
                        f"{detail} for {identity.comment}; no matching job is visible yet and retry is forbidden"
                    ) from invocation_error

        # Re-query after acceptance before publishing canonical evidence.
        scheduler_job = self._unique_existing_job(identity)
        if scheduler_job is not None:
            if found is not None and scheduler_job != found:
                raise DuplicateSubmissionError(f"scheduler identity {identity.comment} resolved to inconsistent jobs")
            found = scheduler_job
        if found is None or type(found) is not str or not found.isdigit():
            raise SubmissionUncertain(f"no numeric Slurm job ID can be proven for {identity.comment}")

        record = SubmissionRecord(
            identity=identity,
            job_id=found,
            task_ids=expected_task_ids,
            task_indices=expected_task_indices,
            # The durable record must be independent of which concurrent caller
            # reaches this line first.  Every accepted identity is re-read from
            # Slurm immediately above, so ``recovered`` records that scheduler
            # proof rather than the caller-local fact of receiving sbatch output.
            # ``SubmissionResult.submitted`` retains that ephemeral distinction.
            recovered=True,
            dependency_mode=stage.dependency_mode,
            dependency_job_ids=authorized_dependencies,
        )
        self._write_immutable_text(plan, record_path, canonical_json(record.to_dict()))
        status = "RETRYING" if attempt > 1 else "SUBMITTED"
        update = RegistryUpdate(
            plan_sha256=plan.plan_sha256,
            stage_id=stage_id,
            stage_status=status,
            execution_status=status,
            job_ids=self._all_recorded_job_ids(plan),
            attempt=attempt,
        )
        claim_ids = self._submission_claim_ids(plan, record.identity, claim_id, invocation_claim_id)
        self._enqueue_registry_update(plan, update, claim_ids)
        if attempt > 1:
            self._ensure_reconciler_followups(plan, stage_id, record)
        registry_synced = self._sync_registry(
            plan,
            update,
            claim_ids,
        )
        return SubmissionResult(record, submitted=submitted_now, registry_synced=registry_synced)

    def submit_afterany_reconciler(
        self,
        plan: ExecutionPlan,
        stage_id: str,
        *,
        attempt: int = 1,
    ) -> SubmissionResult:
        """Submit signed ``afterany`` audit; workstation checks are recovery only."""

        stage = stage_by_id(plan, stage_id)
        if stage.dependency_mode != "afterany" or not stage.depends_on:
            raise ValueError("an explicit reconciler must have dependencies and dependency_mode='afterany'")
        if stage.tasks:
            raise ValueError("an explicit dependency reconciler must be a non-array stage")
        return self.submit_stage(plan, stage_id, attempt=attempt)

    def submit_afterany_followup(
        self,
        plan: ExecutionPlan,
        stage_id: str,
        *,
        attempt: int,
    ) -> SubmissionResult:
        """Submit one authorized generation after an upstream retry.

        Unlike the initial reconciler convenience method, this path permits a
        task-bearing ``afterany`` stage. Its follow-up authorization reruns all
        signed tasks against the newest dependency jobs, preventing stale
        downstream outputs from surviving an upstream replacement attempt.
        """

        stage = stage_by_id(plan, stage_id)
        if stage.dependency_mode != "afterany" or not stage.depends_on:
            raise ValueError("an afterany follow-up must have signed dependencies")
        return self.submit_dependency_followup(plan, stage_id, attempt=attempt)

    def submit_dependency_followup(
        self,
        plan: ExecutionPlan,
        stage_id: str,
        *,
        attempt: int,
    ) -> SubmissionResult:
        """Submit one immutable replacement generation for either dependency mode."""

        stage = stage_by_id(plan, stage_id)
        if not stage.depends_on:
            raise ValueError("a dependency follow-up must have signed dependencies")
        if attempt <= 1:
            raise ValueError("a dependency follow-up generation must use attempt > 1")
        return self.submit_stage(plan, stage_id, attempt=attempt)

    def reconcile_stage(
        self,
        plan: ExecutionPlan,
        stage_id: str,
        *,
        submit_retry: bool = True,
    ) -> ReconcileResult:
        """Coordinate lifecycle exclusion around stage reconciliation writes."""

        # Reconciliation itself is convergent metadata work. Any retry crosses
        # Slurm through submit_stage's separate lifecycle claim, while deletion
        # still requires terminal receipts plus synchronized registry state.
        # Avoiding a long-lived poll claim removes its pre-evidence crash window
        # without bulk-releasing a concurrent caller.
        return self._reconcile_stage_claimed(plan, stage_id, submit_retry=submit_retry)

    def _reconcile_stage_claimed(
        self,
        plan: ExecutionPlan,
        stage_id: str,
        *,
        submit_retry: bool = True,
        claim_id: str = "",
    ) -> ReconcileResult:
        """Reconcile task states and optionally launch a bounded missing-only retry.

        Successful tasks from any previous attempt are final.  They are never
        selected for a later attempt.  A task is successful only when Slurm
        reports a zero-exit ``COMPLETED`` state and every required pipeline
        receipt exists with the expected digest, when one was signed.
        """

        # Reconciliation writes task evidence, decision receipts, retry intents,
        # and registry state.  Authenticate the still-active registered run before
        # any of those mutations, just as submission does before crossing Slurm.
        self._bind_plan(plan)
        stage = stage_by_id(plan, stage_id)
        records = self._submission_records(plan, stage)
        if not records:
            raise ValueError(f"stage {stage_id} has no submitted attempt to reconcile")
        # Recovery is convergent rather than tied to the original caller's stack:
        # any later reconciliation repairs a retry record whose follow-up audit
        # submission was interrupted after the compute job had been accepted.
        latest_record = records[-1]
        if latest_record.identity.attempt > 1:
            # Historical retry generations are stale once a newer accepted
            # attempt exists. Recover only the latest record so an afterok
            # authorization cannot name an older failed trigger while binding
            # its dependency set to the newest successful job.
            self._ensure_reconciler_followups(plan, stage_id, latest_record)
        generation_start = latest_followup_attempt(self.backend, plan, stage)
        if generation_start is not None:
            # Upstream-triggered afterany generations supersede every earlier
            # task verdict, including prior successes. Own retries submitted
            # after this authorization remain in the same generation and can
            # still preserve successes from its earlier attempts.
            records = tuple(record for record in records if record.identity.attempt >= generation_start)
            if not records:
                raise ValueError(
                    f"stage {stage_id} latest afterany generation {generation_start} has no accepted submission"
                )
        tasks = {task.task_id: task for task in select_tasks(stage, None)}
        states_by_job = {record.job_id: tuple(self.backend.task_states(record.job_id)) for record in records}

        successful: list[str] = []
        retryable: list[str] = []
        failed: list[str] = []
        active: list[str] = []
        for task_id, task in sorted(tasks.items()):
            task_records = [record for record in records if task_id in record.task_ids]
            verdict = self._reconcile_task(plan, stage, task, task_records, states_by_job)
            if verdict == "SUCCESS":
                successful.append(task_id)
            elif verdict == "ACTIVE":
                active.append(task_id)
            elif verdict == "RETRY":
                retryable.append(task_id)
            else:
                failed.append(task_id)

        current_attempt = max(record.identity.attempt for record in records)
        next_attempt, rejected_attempts = next_unrejected_attempt(
            self.backend, plan, stage, current_attempt, tuple(sorted(retryable))
        )
        # Rejection consumes an attempt without a SubmissionRecord. Advance the
        # evidence path so terminal failure cannot overwrite an earlier retry.
        evidence_attempt = max([current_attempt, *(item.attempt for item in rejected_attempts)])
        for rejected_identity in rejected_attempts:
            self._release_rejected_invocation_owner(plan, rejected_identity, claim_id)
        retry_submission = None
        if active:
            decision = RECONCILE_WAIT
        elif failed:
            # A non-retryable task means the exact stage cannot ever certify;
            # launching retries for its siblings would spend resources without
            # changing that scientific outcome.
            failed.extend(retryable)
            retryable = []
            decision = RECONCILE_FAILED
        elif retryable:
            if next_attempt > signed_attempt_bound(plan, stage):
                failed.extend(retryable)
                retryable = []
                decision = RECONCILE_FAILED
            else:
                decision = RECONCILE_RETRY
        else:
            # Re-observe the current task outputs immediately before certifying
            # this stage. Historical attempt receipts are not proof that the
            # pipeline files still exist unchanged.
            task_receipts_valid = current_task_receipts_status(self.backend, plan, stage)
            stage_receipts_valid = self._stage_receipts_valid(plan, stage)
            if task_receipts_valid is None:
                decision = RECONCILE_WAIT
                active.append("__current_task_receipts__")
            elif not task_receipts_valid:
                decision = RECONCILE_FAILED
                # The existing sentinel denotes failure of the current
                # files-as-truth surface, whether task- or stage-scoped.
                failed.append("__stage_receipts__")
            elif stage_receipts_valid is None:
                decision = RECONCILE_WAIT
                active.append("__stage_receipts__")
            elif not stage_receipts_valid:
                # Required stage-level receipts belong in a separate finalizer
                # stage. Re-running successful movie tasks cannot safely
                # synthesize an absent aggregate receipt.
                decision = RECONCILE_FAILED
                failed.append("__stage_receipts__")
            else:
                decision = RECONCILE_COMPLETE

        # WAIT is a transient observation and therefore not immutable evidence.
        # Write a reconciliation receipt only once the attempt has a terminal
        # decision; later polls then replay byte-identically.
        if decision != RECONCILE_WAIT:
            reconciliation = ReconciliationReceipt(
                attempt=evidence_attempt,
                decision=decision,
                failed_task_ids=tuple(sorted(failed)),
                plan_sha256=plan.plan_sha256,
                retry_task_ids=tuple(sorted(retryable)),
                stage_id=stage_id,
                successful_task_ids=tuple(sorted(successful)),
            )
            self._write_immutable_text(
                plan,
                reconciliation_path(plan, stage_id, evidence_attempt),
                canonical_json(reconciliation.to_dict()),
            )

        if decision == RECONCILE_RETRY and submit_retry:
            retry_result = self.submit_stage(
                plan,
                stage_id,
                attempt=next_attempt,
                task_ids=tuple(sorted(retryable)),
            )
            retry_submission = retry_result.record
            decision = RECONCILE_RETRY_SUBMITTED

        update = RegistryUpdate(
            plan_sha256=plan.plan_sha256,
            stage_id=stage_id,
            stage_status=decision,
            execution_status=self._execution_status(plan, stage_id, decision),
            job_ids=self._all_recorded_job_ids(plan),
            attempt=(retry_submission.identity.attempt if retry_submission is not None else evidence_attempt),
        )
        registry_synced = self._sync_registry(plan, update)
        if decision == RECONCILE_COMPLETE:
            # A replacement ``afterok`` child cannot be authorized until this
            # exact upstream generation is certified. Trigger it only after the
            # completion receipt and registry repair pointer are durable.
            latest_record = records[-1]
            if latest_record.identity.attempt > 1:
                self._ensure_reconciler_followups(plan, stage_id, latest_record)
        return ReconcileResult(
            decision=decision,
            stage_id=stage_id,
            attempt=evidence_attempt,
            successful_task_ids=tuple(sorted(successful)),
            retry_task_ids=tuple(sorted(retryable)),
            failed_task_ids=tuple(sorted(failed)),
            active_task_ids=tuple(sorted(active)),
            retry_submission=retry_submission,
            registry_synced=registry_synced,
        )

    def _reconcile_task(
        self,
        plan: ExecutionPlan,
        stage: StageSpec,
        task: PlannedTask,
        records: Sequence[SubmissionRecord],
        states_by_job: dict[str, tuple[SlurmTaskState, ...]],
    ) -> str:
        """Return SUCCESS, ACTIVE, RETRY, or FAILED for one stable task."""

        latest_verdict = "FAILED"
        for record in records:
            attempt_path = task_attempt_path(plan, record.identity, task.task_id)
            existing_text = self.backend.read_text(attempt_path)
            if existing_text is not None:
                try:
                    existing = TaskAttemptReceipt.from_dict(strict_json_object(existing_text, "task-attempt receipt"))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"malformed immutable task-attempt receipt: {attempt_path}") from exc
                if (
                    existing.identity != record.identity
                    or existing.task_id != task.task_id
                    or existing.job_id != record.job_id
                ):
                    raise ValueError(f"task-attempt receipt identity mismatch: {attempt_path}")
                successful, retryable = authenticated_task_verdict(stage, task, existing)
                if successful:
                    return "SUCCESS"
                latest_verdict = "RETRY" if retryable else "FAILED"
                continue

            state = task_state(task, states_by_job[record.job_id])
            if state is None:
                latest_verdict = "ACTIVE"
                continue
            normalized = state.normalized_state()
            if normalized not in TERMINAL_SLURM_STATES and normalized != "MISSING":
                # Slurm may report a nonterminal state flag (for example,
                # REQUEUED or RESIZING) instead of the underlying base state.
                # Immutable attempt evidence is safe only for the explicit
                # terminal semantic class. ``MISSING`` is the engine's synthetic
                # verdict for an absent child under a terminal COMPLETED array
                # root, so it must reach the signed missing-receipt retry policy.
                # Unknown future flags fail closed as active instead.
                latest_verdict = "ACTIVE"
                continue

            observations = tuple(
                self.backend.observe_receipt(plan.paths.run_root, receipt) for receipt in task.expected_receipts
            )
            if any(not observation.trustworthy for observation in observations):
                # Scheduler termination is stable, but the filesystem observation
                # is not.  Do not freeze a false missing-receipt verdict or spend a
                # retry because SSH/filesystem inspection temporarily failed.
                latest_verdict = "ACTIVE"
                continue
            receipts_valid = all(
                receipt_matches(spec, observed) for spec, observed in zip(task.expected_receipts, observations)
            )
            successful = normalized in SUCCESS_SLURM_STATES and state.exit_code == 0 and receipts_valid
            retryable = not successful and is_retryable(stage, normalized, state.exit_code, receipts_valid)
            receipt = TaskAttemptReceipt(
                identity=record.identity,
                task_id=task.task_id,
                array_index=task.array_index,
                job_id=record.job_id,
                slurm_state=normalized,
                exit_code=state.exit_code,
                receipt_observations=observations,
                successful=successful,
                retryable=retryable,
            )
            self._write_immutable_text(
                plan,
                task_attempt_path(plan, receipt.identity, receipt.task_id),
                canonical_json(receipt.to_dict()),
            )
            if successful:
                return "SUCCESS"
            latest_verdict = "RETRY" if retryable else "FAILED"
        return latest_verdict

    def _dependency_jobs(self, plan: ExecutionPlan, stage: StageSpec) -> tuple[str, ...]:
        """Resolve each prerequisite to its latest submitted attempt."""

        job_ids: list[str] = []
        for dependency in stage.depends_on:
            dependency_stage = stage_by_id(plan, dependency)
            records = self._submission_records(plan, dependency_stage)
            if not records:
                raise ValueError(f"dependency stage {dependency} has no submitted job")
            job_ids.append(records[-1].job_id)
        return tuple(job_ids)

    def _validate_afterok_dependencies(self, plan: ExecutionPlan, stage: StageSpec) -> None:
        """Require files-as-truth completion before an ``afterok`` stage is queued.

        Slurm's ``afterok`` checks only the prerequisite process exit code.  It
        cannot know whether required pipeline receipts are absent or corrupt.
        Refusing early submission prevents a downstream scientific stage from
        running after a superficially successful but uncertified prerequisite.
        """

        if stage.dependency_mode != "afterok":
            return
        incomplete = [
            dependency
            for dependency in stage.depends_on
            if not self._has_completed_reconciliation(plan, stage_by_id(plan, dependency))
        ]
        if incomplete:
            raise ValueError(
                "afterok dependencies lack authenticated COMPLETED reconciliation receipts: "
                + ", ".join(sorted(incomplete))
            )

    def _prior_generation_jobs(
        self,
        plan: ExecutionPlan,
        stage: StageSpec,
        attempt: int,
    ) -> tuple[str, ...]:
        """Return this stage's earlier generation jobs in attempt order.

        Only strictly earlier attempts are included, which keeps the value
        immutable for a given identity: later attempts cannot appear below one
        that already exists, so a replay renders byte-identical intent.
        """

        return tuple(
            record.job_id for record in self._submission_records(plan, stage) if record.identity.attempt < attempt
        )

    def _validate_afterok_dependency_jobs(
        self,
        plan: ExecutionPlan,
        stage: StageSpec,
        dependency_jobs: tuple[str, ...],
    ) -> None:
        """Require the exact resolved prerequisite jobs to be certified complete.

        Stage-level certification is not enough: a prerequisite can accept a
        replacement job that has not reconciled yet, and ``afterok`` against that
        job would release this stage on a zero exit while its required receipts
        are absent or corrupt -- the condition the gate exists to prevent.
        """

        if stage.dependency_mode != "afterok":
            return
        uncertified = []
        for dependency, job_id in zip(stage.depends_on, dependency_jobs):
            dependency_stage = stage_by_id(plan, dependency)
            receipt = latest_reconciliation_receipt(self.backend, plan, dependency_stage)
            certified = next(
                (record for record in self._submission_records(plan, dependency_stage) if record.job_id == job_id),
                None,
            )
            if (
                receipt is None
                or receipt.decision != RECONCILE_COMPLETE
                or certified is None
                or receipt.attempt != certified.identity.attempt
                or not self._has_completed_reconciliation(plan, dependency_stage)
            ):
                uncertified.append(f"{dependency}:{job_id}")
        if uncertified:
            raise ValueError(
                "afterok dependencies resolve to jobs without authenticated COMPLETED reconciliation: "
                + ", ".join(sorted(uncertified))
            )

    def _unique_existing_job(self, identity: SubmissionIdentity) -> str | None:
        """Return one matching job ID or reject an identity collision."""

        jobs = tuple(self.backend.find_jobs(identity.comment))
        unique = sorted({job.job_id for job in jobs})
        if len(unique) > 1:
            raise DuplicateSubmissionError(
                f"{identity.comment} is attached to multiple Slurm jobs: {', '.join(unique)}"
            )
        return unique[0] if unique else None

    def _submission_records(self, plan: ExecutionPlan, stage: StageSpec) -> tuple[SubmissionRecord, ...]:
        """Return records only after full plan/path authorization checks."""

        return read_plan_submission_records(self.backend, plan, stage)

    def _all_recorded_job_ids(self, plan: ExecutionPlan) -> tuple[str, ...]:
        job_ids: set[str] = set()
        for stage in plan.stages:
            job_ids.update(record.job_id for record in self._submission_records(plan, stage))
        return tuple(sorted(job_ids, key=int))

    def _stage_receipts_valid(self, plan: ExecutionPlan, stage: StageSpec) -> bool | None:
        """Return receipt validity, or ``None`` when observation was not trustworthy."""

        observations = [self.backend.observe_receipt(plan.paths.run_root, item) for item in stage.expected_receipts]
        if any(not observed.trustworthy for observed in observations):
            return None
        return all(receipt_matches(spec, observed) for spec, observed in zip(stage.expected_receipts, observations))

    def _execution_status(self, plan: ExecutionPlan, stage_id: str, decision: str) -> str:
        if decision == RECONCILE_FAILED:
            return "FAILED"
        if decision in {RECONCILE_RETRY, RECONCILE_RETRY_SUBMITTED}:
            return "RETRYING"
        if decision == RECONCILE_WAIT:
            return "RUNNING"
        if decision != RECONCILE_COMPLETE:
            return "ACTIVE"
        # A later audit/reconciler may complete after an upstream scientific
        # stage failed.  Terminal failure is sticky; it must not regress to
        # ACTIVE merely because the most recently observed stage succeeded.
        if any(self._latest_reconciliation_decision(plan, stage) == RECONCILE_FAILED for stage in plan.stages):
            return "FAILED"
        for stage in plan.stages:
            if not self._has_completed_reconciliation(plan, stage):
                return "ACTIVE"
        return "COMPLETED"

    def _latest_reconciliation_decision(self, plan: ExecutionPlan, stage: StageSpec) -> str | None:
        """Return the latest strictly validated decision for one stage, if any."""

        receipt = latest_reconciliation_receipt(self.backend, plan, stage)
        return receipt.decision if receipt is not None else None

    def _has_completed_reconciliation(self, plan: ExecutionPlan, stage: StageSpec) -> bool:
        receipt = latest_reconciliation_receipt(self.backend, plan, stage)
        if receipt is None or receipt.decision != RECONCILE_COMPLETE:
            return False
        # Stage-level aggregate receipts are not represented by task-attempt
        # evidence. Re-observe them at the downstream gate and fail closed on an
        # untrustworthy or changed result.
        return (
            current_task_receipts_valid(self.backend, plan, stage) and self._stage_receipts_valid(plan, stage) is True
        )


__all__ = ["ExecutionEngine", "RegistrySynchronizer"]
