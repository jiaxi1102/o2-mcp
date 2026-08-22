"""Detached promote/archive implementation mixed into :class:`O2Runs`.

Separating this lifecycle boundary keeps the general run executor reviewable and
ensures every destructive path uses the same marker, evidence, scheduler, and
rollback protocol.
"""

from __future__ import annotations

import hashlib
import posixpath
import shlex
from dataclasses import dataclass

from o2mcp.runorg.execution_backend import O2ExecutionBackend
from o2mcp.runorg.plans import ExecutionPlan
from o2mcp.runorg.runs import STATUS_ACTIVE, STATUS_KEPT, RunManifest, plan_archive_script, plan_promote_script
from o2mcp.runorg.transition_coordinator import begin_transition, rollback_transition
from o2mcp.runorg.transition_guards import (
    live_jobs_command,
    require_certified_terminal_execution,
    require_current_terminal_evidence,
)


@dataclass
class TransitionPlan:
    """The outcome of a promote/archive request (dry-run or launched)."""

    run_id: str
    action: str
    script: str
    started: bool = False
    pid: str | None = None
    log_path: str | None = None
    message: str = ""


class TransitionExecutorMixin:
    """Provide coordinated destructive lifecycle methods to ``O2Runs``."""

    def promote(self, run_dir: str, *, dry_run: bool = True, run_remote: bool = True) -> TransitionPlan:
        """Build or launch a certified scratch-to-group promotion."""

        manifest = self._validated_transition_source(
            run_dir, action="promote", allowed_statuses=(STATUS_ACTIVE, STATUS_KEPT)
        )
        if manifest.status == STATUS_KEPT:
            return TransitionPlan(
                manifest.run_id,
                "promote",
                "",
                started=False,
                message="already kept at the canonical durable path; no transfer or deletion performed",
            )
        transition_id = self._transition_id(manifest, "promote")
        script = plan_promote_script(self.layout, manifest, source_dir=run_dir, transition_id=transition_id)
        return self._transition(
            manifest.run_id,
            "promote",
            script,
            run_dir=run_dir,
            transition_id=transition_id,
            dry_run=dry_run,
            run_remote=run_remote,
        )

    def archive(self, run_dir: str, *, dry_run: bool = True, run_remote: bool = True) -> TransitionPlan:
        """Build or launch a certified scratch/group-to-standby archive."""

        manifest = self._validated_transition_source(
            run_dir, action="archive", allowed_statuses=(STATUS_ACTIVE, STATUS_KEPT)
        )
        transition_id = self._transition_id(manifest, "archive")
        script = plan_archive_script(
            self.layout,
            manifest,
            source_dir=run_dir,
            archive_excludes=self.policy.archive_excludes,
            transition_id=transition_id,
        )
        return self._transition(
            manifest.run_id,
            "archive",
            script,
            run_dir=run_dir,
            transition_id=transition_id,
            dry_run=dry_run,
            run_remote=run_remote,
        )

    def _validated_transition_source(
        self, run_dir: str, *, action: str, allowed_statuses: tuple[str, ...]
    ) -> RunManifest:
        """Require a strict manifest at its exact canonical tier."""

        manifest = self._read_strict_manifest(run_dir)
        if manifest is None:
            raise ValueError(f"{action} requires an existing strict run.json")
        if manifest.status not in allowed_statuses:
            raise ValueError(f"{action} does not accept run status {manifest.status!r}")
        expected = self.layout.run_dir(manifest.status, manifest.campaign, manifest.run_id)
        if run_dir != expected:
            raise ValueError(f"{action} source {run_dir!r} is not the canonical {manifest.status} path {expected!r}")
        require_certified_terminal_execution(manifest, action)
        # Preview-time feedback remains useful, but actual transitions repeat
        # this query after atomically marking the transition and augment the ID
        # set from submission/outbox evidence.
        live = self._run(live_jobs_command(manifest.slurm_job_ids), timeout=60)
        if not live.ok:
            raise ValueError(f"{action} could not prove that registered Slurm jobs are terminal")
        if live.stdout.strip():
            raise ValueError(f"{action} refuses a source with live Slurm jobs: {live.stdout.strip()}")
        return manifest

    @staticmethod
    def _transition_id(manifest: RunManifest, action: str) -> str:
        """Bind a deterministic transition marker to exact manifest bytes."""

        return hashlib.sha256(f"{action}\0{manifest.run_id}\0{manifest.to_json()}".encode()).hexdigest()

    def _transition(
        self,
        run_id: str,
        action: str,
        script: str,
        *,
        run_dir: str,
        transition_id: str,
        dry_run: bool,
        run_remote: bool,
    ) -> TransitionPlan:
        """Mark, recertify, prove no live jobs, then launch one detached script."""

        if dry_run or not run_remote:
            return TransitionPlan(run_id, action, script, started=False, message="dry_run: script not executed")
        boundary = begin_transition(self.conn, run_dir, transition_id)
        try:
            manifest = self._read_strict_manifest(run_dir)
            if manifest is None:
                raise ValueError(f"{action} source disappeared after transition marking")
            require_certified_terminal_execution(manifest, action)
            expected_source = self.layout.run_dir(manifest.status, manifest.campaign, manifest.run_id)
            if run_dir != expected_source:
                raise ValueError(f"{action} source identity changed after transition marking")
            execution = (manifest.provenance or {}).get("execution", {})
            expected_sha = execution.get("plan_sha256") if isinstance(execution, dict) else None
            if type(expected_sha) is not str:
                raise ValueError(f"{action} manifest lacks an authenticated execution plan digest")
            backend = O2ExecutionBackend(self.conn)
            plan_text = backend.read_text(posixpath.join(run_dir, "receipts", "execution", "execution-plan.json"))
            plan = ExecutionPlan.from_json(plan_text or "", expected_plan_sha256=expected_sha)
            if (
                plan.paths.run_root != run_dir
                or plan.run_id != manifest.run_id
                or plan.campaign != manifest.campaign
                or plan.pipeline != manifest.pipeline
                or sorted(item.dataset_id for item in plan.datasets) != sorted(manifest.datasets)
            ):
                raise ValueError(f"{action} execution plan identity differs from the marked source run")
            require_current_terminal_evidence(backend, plan, action)
            jobs = sorted(set(manifest.slurm_job_ids) | set(boundary.job_ids), key=int)
            live = self._run(live_jobs_command(jobs), timeout=60)
            if not live.ok:
                raise ValueError(f"{action} could not prove that every discovered Slurm job is terminal")
            if live.stdout.strip():
                raise ValueError(f"{action} refuses a source with live Slurm jobs: {live.stdout.strip()}")
        except Exception:
            rollback_transition(self.conn, run_dir, transition_id)
            raise
        return self._stage_and_launch_transition(run_id, action, script, run_dir, transition_id)

    def _stage_and_launch_transition(
        self, run_id: str, action: str, script: str, run_dir: str, transition_id: str
    ) -> TransitionPlan:
        """Stage and detach a marked transition, rolling back pre-launch errors."""

        script_path = posixpath.join(self.layout.scratch_runs_root, ".jobs", f"{action}_{run_id}.sh")
        log_path = script_path + ".log"
        self._run(f"mkdir -p {shlex.quote(posixpath.dirname(script_path))}", timeout=60)
        stage = self.conn.run(f"cat > {shlex.quote(script_path)}", timeout=60, input_text=script)
        if not stage.ok:
            rollback_transition(self.conn, run_dir, transition_id)
            return TransitionPlan(run_id, action, script, message=stage.stderr.strip() or "staging failed")
        launch = f"nohup bash {shlex.quote(script_path)} > {shlex.quote(log_path)} 2>&1 < /dev/null & echo PID $!"
        result = self.conn.run(launch, timeout=60, alias=self.conn.config.transfer_alias, broker_role="transfer")
        pid = next((token for token in result.stdout.split() if token.isdigit()), "")
        if not (result.ok and pid):
            rollback_transition(self.conn, run_dir, transition_id)
        return TransitionPlan(
            run_id,
            action,
            script,
            started=result.ok and bool(pid),
            pid=pid or None,
            log_path=log_path,
            message=f"launched on transfer node (pid {pid}); tail {log_path}" if pid else result.stderr.strip(),
        )


__all__ = ["TransitionExecutorMixin", "TransitionPlan"]
