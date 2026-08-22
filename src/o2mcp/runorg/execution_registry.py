"""Registry outbox and lifecycle-claim mixin for the execution engine."""

from __future__ import annotations

from dataclasses import replace

from o2mcp.runorg.execution_evidence import read_submission_invocation_claim_id
from o2mcp.runorg.execution_models import RegistryUpdate, SubmissionIdentity, canonical_json
from o2mcp.runorg.execution_paths import bound_plan_path, pending_registry_path
from o2mcp.runorg.execution_reconcile import signed_attempt_bound
from o2mcp.runorg.plans import ExecutionPlan
from o2mcp.runorg.registry_outbox import decode_registry_update, merge_registry_updates


class ExecutionRegistryMixin:
    """Provide monotonic registry convergence and transition coordination."""

    def reconcile_registry(self, plan: ExecutionPlan) -> bool:
        """Drain every per-stage/attempt outbox item without touching Slurm."""

        if self.registry is None:
            return True
        operation_id = f"registry:{plan.plan_sha256}"
        claim_id = self._acquire_lifecycle(plan, operation_id)
        try:
            self._bind_plan(plan)
            all_synced = True
            for stage in plan.stages:
                for attempt in range(1, signed_attempt_bound(plan, stage) + 1):
                    path = pending_registry_path(plan, stage.stage_id, attempt)
                    while (text := self.backend.read_text(path)) is not None:
                        update = decode_registry_update(text)
                        if (
                            update.plan_sha256 != plan.plan_sha256
                            or update.stage_id != stage.stage_id
                            or update.attempt != attempt
                        ):
                            raise ValueError("pending registry update belongs to a different path identity")
                        try:
                            synced = bool(self.registry.synchronize(plan, update))
                        except Exception:
                            synced = False
                        if not synced:
                            all_synced = False
                            break
                        # A holder recorded in this exact outbox has finished
                        # every scheduler/evidence mutation before enqueueing it.
                        # Retire those exact owners before compare-clear so a
                        # release failure leaves durable repair work to replay.
                        for repaired_claim_id in update.lifecycle_claim_ids:
                            self._release_lifecycle(plan, repaired_claim_id)
                        self.backend.compare_and_swap_text(path, text, None)
            return all_synced
        finally:
            # Registry repair itself never invokes Slurm. Pending outboxes retain
            # their originating mutation claims, so this metadata-only holder
            # need not remain after a failed synchronization attempt.
            self._release_lifecycle(plan, claim_id)

    def _sync_registry(
        self,
        plan: ExecutionPlan,
        update: RegistryUpdate,
        claim_ids: tuple[str, ...] = (),
    ) -> bool:
        """Enqueue one update with every exact repairable lifecycle owner."""

        if self.registry is None:
            for claim_id in claim_ids:
                self._release_lifecycle(plan, claim_id)
            return True
        update = replace(
            update,
            lifecycle_claim_ids=tuple(sorted(set(update.lifecycle_claim_ids) | {item for item in claim_ids if item})),
        )
        pending_path = pending_registry_path(plan, update.stage_id, update.attempt)
        pending_text = self._merge_outbox(pending_path, update)
        try:
            synced = bool(self.registry.synchronize(plan, decode_registry_update(pending_text)))
        except Exception:
            synced = False
        if not synced:
            return False
        # Release before compare-clear: the outbox is the crash-repair pointer
        # for this exact owner. If release fails, preserve it and report pending
        # convergence rather than losing the only evidence that can repair the
        # otherwise transition-blocking claim.
        # ``_merge_outbox`` may have joined holders left by an earlier failed
        # registry write with the holders supplied by this invocation.  Decode
        # the exact bytes that synchronized successfully so every durable repair
        # pointer is retired before those bytes are compare-cleared.
        merged_update = decode_registry_update(pending_text)
        for claim_id in merged_update.lifecycle_claim_ids:
            try:
                self._release_lifecycle(plan, claim_id)
            except Exception:
                return False
        return self.backend.compare_and_swap_text(pending_path, pending_text, None)

    def _merge_outbox(self, path: str, update: RegistryUpdate) -> str:
        """CAS-join ``update`` into one per-attempt outbox under contention."""

        for _ in range(32):
            current_text = self.backend.read_text(path)
            current = decode_registry_update(current_text) if current_text is not None else None
            merged_text = canonical_json(merge_registry_updates(current, update).to_dict())
            if current_text == merged_text:
                return merged_text
            if self.backend.compare_and_swap_text(path, current_text, merged_text):
                return merged_text
        raise RuntimeError(f"registry outbox remained contended: {path}")

    def _bind_plan(self, plan: ExecutionPlan) -> None:
        """Validate the active run and bind it to one immutable plan envelope."""

        if self.registry is not None:
            try:
                valid = bool(self.registry.validate_plan(plan))
            except Exception as exc:
                raise ValueError("registered run identity could not be authenticated before submission") from exc
            if not valid:
                raise ValueError("execution plan does not match the registered active run identity")
        self.backend.write_immutable_text(bound_plan_path(plan), plan.to_json())

    def _acquire_lifecycle(self, plan: ExecutionPlan, operation_id: str) -> str:
        """Acquire and return one distinct transition-boundary holder ID."""

        acquire = getattr(self.backend, "acquire_lifecycle_claim", None)
        if acquire is None:
            return ""
        claim_id = acquire(plan.paths.run_root, operation_id)
        if claim_id is None:
            raise ValueError("run lifecycle transition is in progress; control-plane mutation refused")
        if type(claim_id) is not str:
            raise TypeError("execution backend returned an invalid lifecycle claim identity")
        return claim_id

    def _release_lifecycle(self, plan: ExecutionPlan, claim_id: str) -> None:
        """Release one exact holder claim after durable convergence."""

        release = getattr(self.backend, "release_lifecycle_claim", None)
        if release is not None and claim_id:
            release(plan.paths.run_root, claim_id)

    def _release_if_not_invocation_owner(
        self,
        plan: ExecutionPlan,
        identity: SubmissionIdentity,
        claim_id: str,
    ) -> None:
        """Release only when durable evidence proves this caller did not invoke.

        Malformed or unreadable invocation evidence fails closed by retaining the
        claim. A valid marker owned by another concurrent caller proves this
        holder never crossed ``sbatch`` and is therefore safe to release.
        """

        try:
            owner = read_submission_invocation_claim_id(
                self.backend,
                plan,
                identity,
            )
        except Exception:
            return
        if owner != claim_id:
            self._release_lifecycle(plan, claim_id)

    def _release_rejected_invocation_owner(
        self,
        plan: ExecutionPlan,
        identity: SubmissionIdentity,
        observer_claim_id: str,
    ) -> None:
        """Retire a prior invocation owner after definitive rejection proof.

        The caller's observer claim is released by the ordinary
        :class:`SubmissionRejected` boundary; this helper targets only a stale,
        distinct owner left by a lost release response.
        """

        owner_claim_id = read_submission_invocation_claim_id(self.backend, plan, identity)
        if owner_claim_id is not None and owner_claim_id != observer_claim_id:
            self._release_lifecycle(plan, owner_claim_id)


__all__ = ["ExecutionRegistryMixin"]
