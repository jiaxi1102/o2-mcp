"""Detached promote/archive implementation mixed into :class:`O2Runs`.

Separating this lifecycle boundary keeps the general run executor reviewable and
ensures every destructive path uses the same marker, evidence, scheduler, and
rollback protocol.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
import shlex
from dataclasses import dataclass
from typing import Any

from o2mcp.runorg.execution_backend import O2ExecutionBackend
from o2mcp.runorg.execution_models import ReceiptObservation
from o2mcp.runorg.plan_components import ReceiptSpec
from o2mcp.runorg.plans import ExecutionPlan
from o2mcp.runorg.runs import STATUS_ACTIVE, STATUS_KEPT, RunManifest, plan_archive_script, plan_promote_script
from o2mcp.runorg.transition_coordinator import (
    TransitionRecovery,
    begin_transition,
    rollback_transition,
)
from o2mcp.runorg.transition_coordinator import (
    recover_transition as recover_marked_transition,
)
from o2mcp.runorg.transition_guards import (
    live_jobs_command,
    require_certified_terminal_execution,
    require_current_terminal_evidence,
)
from o2mcp.runorg.transition_scripts import source_quarantine_path


class _RelocatedEvidenceBackend:
    """Read copied execution evidence through its immutable original paths.

    Promotion copies an execution plan without rewriting it, so a plan in the
    durable kept tree still names the former scratch run root.  Destructive
    transition validation must preserve that signed identity while reading the
    corresponding copied evidence from the canonical kept source.  The wrapper
    intentionally relocates reads only; all other backend operations retain the
    delegate's behavior.
    """

    def __init__(self, delegate: O2ExecutionBackend, original_root: str, current_root: str) -> None:
        self._delegate = delegate
        self._original_root = original_root.rstrip("/")
        self._current_root = current_root.rstrip("/")

    def _current_path(self, path: str) -> str:
        """Map a path inside the signed run root onto the current source tree."""

        if path == self._original_root:
            return self._current_root
        prefix = self._original_root + "/"
        if path.startswith(prefix):
            return self._current_root + path[len(self._original_root) :]
        return path

    def read_text(self, path: str) -> str | None:
        """Read immutable execution metadata from the relocated copy."""

        return self._delegate.read_text(self._current_path(path))

    def observe_receipt(self, run_root: str, receipt: ReceiptSpec) -> ReceiptObservation:
        """Observe pipeline receipts below the relocated run root."""

        return self._delegate.observe_receipt(self._current_path(run_root), receipt)

    def __getattr__(self, name: str) -> Any:
        """Forward unused protocol operations without changing their semantics."""

        return getattr(self._delegate, name)


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

    def recover_transition(
        self,
        run_dir: str,
        action: str,
        *,
        dry_run: bool = True,
    ) -> TransitionRecovery:
        """Inspect or clear a cleanly abandoned promote/archive marker.

        This operation is deliberately explicit and defaults to read-only. It
        does not attempt to repair partial copies or quarantines: those states
        remain fenced for manual review. Applying recovery is allowed only when
        the source is still certified and canonical, no registered job is live,
        no transition process remains, and no staging or destination artifact
        exists.
        """

        if action not in {"archive", "promote"}:
            raise ValueError("transition recovery action must be 'promote' or 'archive'")
        manifest = self._validated_transition_source(
            run_dir,
            action=action,
            allowed_statuses=(STATUS_ACTIVE, STATUS_KEPT),
        )
        transition_id = self._transition_id(manifest, action)
        script_path = self._transition_script_path(action, manifest.run_id)
        quarantine = source_quarantine_path(run_dir, transition_id)

        if action == "promote":
            destination = self.layout.run_dir(STATUS_KEPT, manifest.campaign, manifest.run_id)
            must_be_absent = (quarantine, destination)
            absent_patterns = (
                posixpath.join(
                    posixpath.dirname(destination),
                    f".{posixpath.basename(destination)}.promote.??????",
                ),
            )
        else:
            tarball = self.layout.archive_tarball(manifest.campaign, manifest.run_id)
            must_be_absent = (
                quarantine,
                tarball,
                self.layout.archive_checksum(manifest.campaign, manifest.run_id),
                self.layout.archive_manifest(manifest.campaign, manifest.run_id),
            )
            absent_patterns = (
                posixpath.join(
                    posixpath.dirname(tarball),
                    f".{manifest.run_id}.archive.??????",
                ),
            )

        return recover_marked_transition(
            self.conn,
            run_dir,
            transition_id,
            script_path=script_path,
            must_be_absent=must_be_absent,
            absent_patterns=absent_patterns,
            apply=not dry_run,
            # The detached transition shell is launched on the transfer host.
            # Its PID is node-local evidence, so recovery must inspect /proc on
            # that same host rather than the login broker.
            alias=self.conn.config.transfer_alias,
            broker_role="transfer",
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

    def _require_certified_plan_evidence(
        self,
        action: str,
        manifest: RunManifest,
        run_dir: str,
        backend: O2ExecutionBackend,
        plan_text: str | None,
        expected_sha: object,
    ) -> None:
        """Recertify one execution-plan run against files-as-truth."""

        if type(expected_sha) is not str:
            raise ValueError(f"{action} manifest lacks an authenticated execution plan digest")
        plan = ExecutionPlan.from_json(plan_text or "", expected_plan_sha256=expected_sha)
        # A kept run contains a byte-identical copy of the plan registered in
        # scratch.  Permit that one canonical relocation without weakening
        # any of the plan's signed identity fields or rewriting its digest.
        allowed_plan_roots = {run_dir}
        if manifest.status == STATUS_KEPT:
            allowed_plan_roots.add(self.layout.run_dir(STATUS_ACTIVE, manifest.campaign, manifest.run_id))
        if (
            plan.paths.run_root not in allowed_plan_roots
            or plan.run_id != manifest.run_id
            or plan.campaign != manifest.campaign
            or plan.pipeline != manifest.pipeline
            or sorted(item.dataset_id for item in plan.datasets) != sorted(manifest.datasets)
        ):
            raise ValueError(f"{action} execution plan identity differs from the marked source run")
        evidence_backend = (
            backend
            if plan.paths.run_root == run_dir
            else _RelocatedEvidenceBackend(backend, plan.paths.run_root, run_dir)
        )
        require_current_terminal_evidence(evidence_backend, plan, action)

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
            backend = O2ExecutionBackend(self.conn)
            plan_text = backend.read_text(posixpath.join(run_dir, "receipts", "execution", "execution-plan.json"))
            # A run registered outside the execution engine has neither signal
            # and keeps the shared live-job criteria below rather than losing
            # its lifecycle path to an engine-only gate.  Either signal alone
            # proves an execution-plan run, so a manifest that lost its
            # provenance still fails closed here.
            if plan_text is not None or expected_sha is not None:
                self._require_certified_plan_evidence(
                    action,
                    manifest,
                    run_dir,
                    backend,
                    plan_text,
                    expected_sha,
                )
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

        script_path = self._transition_script_path(action, run_id)
        log_path = script_path + ".log"
        try:
            mkdir = self._run(f"mkdir -p {shlex.quote(posixpath.dirname(script_path))}", timeout=60)
        except Exception:
            rollback_transition(self.conn, run_dir, transition_id)
            raise
        if not mkdir.ok:
            rollback_transition(self.conn, run_dir, transition_id)
            return TransitionPlan(run_id, action, script, message=mkdir.stderr.strip() or "staging directory failed")
        try:
            stage = self.conn.run(f"cat > {shlex.quote(script_path)}", timeout=60, input_text=script)
        except Exception:
            rollback_transition(self.conn, run_dir, transition_id)
            raise
        if not stage.ok:
            rollback_transition(self.conn, run_dir, transition_id)
            return TransitionPlan(run_id, action, script, message=stage.stderr.strip() or "staging failed")
        launch = f"nohup bash {shlex.quote(script_path)} > {shlex.quote(log_path)} 2>&1 < /dev/null & echo PID $!"
        try:
            result = self.conn.run(launch, timeout=60, alias=self.conn.config.transfer_alias, broker_role="transfer")
        except Exception as exc:
            # Raising here would hide the most important fact: the transport may
            # have failed after the remote shell launched the detached process.
            return TransitionPlan(
                run_id,
                action,
                script,
                log_path=log_path,
                message=f"launch outcome ambiguous; transition marker retained for reconciliation: {exc}",
            )
        pid_matches = re.findall(r"(?m)^PID ([0-9]+)\s*$", result.stdout)
        pid = pid_matches[0] if len(pid_matches) == 1 else ""
        launched = result.ok and bool(pid)
        if launched:
            message = f"launched on transfer node (pid {pid}); tail {log_path}"
        else:
            # Once the detached command has been sent, a transport failure or
            # missing PID cannot prove it did not start.  Retaining the marker
            # prevents concurrent submissions from racing a possible transition.
            detail = result.stderr.strip() or result.stdout.strip() or "no PID returned"
            message = f"launch outcome ambiguous; transition marker retained for reconciliation: {detail}"
        return TransitionPlan(
            run_id,
            action,
            script,
            started=launched,
            pid=pid if launched else None,
            log_path=log_path,
            message=message,
        )

    def _transition_script_path(self, action: str, run_id: str) -> str:
        """Return the stable staged script path used for launch inspection."""

        return posixpath.join(self.layout.scratch_runs_root, ".jobs", f"{action}_{run_id}.sh")


__all__ = ["TransitionExecutorMixin", "TransitionPlan"]
