"""Registry outbox and lifecycle-claim mixin for the execution engine."""

from __future__ import annotations

from o2mcp.runorg.execution_models import RegistryUpdate, canonical_json
from o2mcp.runorg.execution_paths import bound_plan_path, pending_registry_path
from o2mcp.runorg.plans import ExecutionPlan
from o2mcp.runorg.registry_outbox import decode_registry_update, merge_registry_updates


class ExecutionRegistryMixin:
    """Provide monotonic registry convergence and transition coordination."""

    def reconcile_registry(self, plan: ExecutionPlan) -> bool:
        """Drain every per-stage/attempt outbox item without touching Slurm."""

        if self.registry is None:
            return True
        operation_id = f"registry:{plan.plan_sha256}"
        self._acquire_lifecycle(plan, operation_id)
        self._bind_plan(plan)
        all_synced = True
        for stage in plan.stages:
            for attempt in range(1, stage.retry_policy.max_attempts + 1):
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
                    # Clear only the bytes just synchronized. A concurrent
                    # lattice advance remains and is processed by the loop.
                    self.backend.compare_and_swap_text(path, text, None)
        if all_synced:
            self._release_lifecycle(plan, operation_id)
        return all_synced

    def _sync_registry(self, plan: ExecutionPlan, update: RegistryUpdate) -> bool:
        """Enqueue, synchronize, then compare-clear one monotonic update."""

        if self.registry is None:
            return True
        pending_path = pending_registry_path(plan, update.stage_id, update.attempt)
        pending_text = self._merge_outbox(pending_path, update)
        try:
            synced = bool(self.registry.synchronize(plan, decode_registry_update(pending_text)))
        except Exception:
            synced = False
        if not synced:
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

    def _acquire_lifecycle(self, plan: ExecutionPlan, operation_id: str) -> None:
        """Acquire the shared transition boundary when supported by a backend."""

        acquire = getattr(self.backend, "acquire_lifecycle_claim", None)
        if acquire is not None and not acquire(plan.paths.run_root, operation_id):
            raise ValueError("run lifecycle transition is in progress; control-plane mutation refused")

    def _release_lifecycle(self, plan: ExecutionPlan, operation_id: str) -> None:
        """Release one completed mutation claim after durable convergence."""

        release = getattr(self.backend, "release_lifecycle_claim", None)
        if release is not None:
            release(plan.paths.run_root, operation_id)


__all__ = ["ExecutionRegistryMixin"]
