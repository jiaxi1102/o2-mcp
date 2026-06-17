"""Executor for the run-organization lifecycle over an :class:`O2Connection`.

:class:`O2Runs` is the orchestrator the MCP tools call. It keeps light, fast
metadata operations (register / list / show / classify / registry sync / gc audit)
on the login node, and launches the heavy, large-IO tier transitions (promote
rsync, archive tar+zstd) **detached on the O2 transfer node** — standby is writable
only there, and the transfers take hours, so the call returns immediately with a
pid + log path. Every transition has a ``dry_run`` that returns the exact script
without touching anything; the script always verifies before it frees any scratch.

The pure conventions live in :mod:`o2mcp.runorg.runs`; this module only wires them
to the connection. Python 3.9, no third-party deps.
"""

from __future__ import annotations

import posixpath
import shlex
from dataclasses import dataclass
from typing import Any

from o2mcp.connection import CommandResult, O2Connection
from o2mcp.runorg.policy import RunPolicy
from o2mcp.runorg.runs import (
    STATUS_ACTIVE,
    RunLayout,
    RunManifest,
    campaign_of,
    classify_run,
    merge_status_json,
    parse_registry,
    parse_submission_env,
    plan_archive_script,
    plan_gc_candidates_command,
    plan_promote_script,
    plan_register_commands,
    plan_write_manifest_command,
    registry_line,
    sort_job_ids,
    variant_of,
)
from o2mcp.slurm import O2Slurm


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


# Remote gather for the status board: emits one JSON line with the dataset list,
# the registry, each run's latest status JSON, and live squeue/sacct text. Run via
# `python3 -` (stdin) on O2; placeholders are filled with the configured roots
# (plain paths, no shell metachars). Defensive throughout — a missing dir or a bad
# status file degrades to empty, never an exception.


class O2Runs:
    """Run-lifecycle operations over an established O2 connection."""

    def __init__(self, connection: O2Connection, policy: RunPolicy, layout: RunLayout | None = None) -> None:
        self.conn = connection
        self.policy = policy
        self.layout = layout or RunLayout.from_config(connection.config)

    # -- light metadata ops (login node) ---------------------------------------
    def _run(self, command: str, *, timeout: float = 120.0) -> CommandResult:
        return self.conn.run(command, timeout=timeout)

    def utc_now(self) -> str:
        """A UTC run-id timestamp (``YYYYMMDDTHHMMSSZ``) minted on O2."""
        res = self._run("date -u +%Y%m%dT%H%M%SZ", timeout=30)
        return res.stdout.strip() if res.ok else ""

    def list_run_dirs(self, root: str | None = None, *, depth_grouped: bool = True) -> list[str]:
        """Absolute paths of run directories under a runs root.

        ``depth_grouped`` lists ``<root>/<campaign>/RUN_*`` (the new layout); set it
        False to list a legacy flat ``<root>/RUN_*`` tree (e.g. clock_true_ultrack).
        """
        base = root or self.layout.scratch_runs_root
        depth = 2 if depth_grouped else 1
        res = self._run(
            f"find {shlex.quote(base)} -mindepth {depth} -maxdepth {depth} "
            f"-type d -name 'RUN_*' 2>/dev/null | sort",
            timeout=120,
        )
        return [line for line in res.stdout.splitlines() if line.strip()]

    def read_manifest(self, run_dir: str) -> RunManifest | None:
        """Return the run.json manifest, synthesizing one from legacy metadata if absent."""
        cat = self._run(f"cat {shlex.quote(posixpath.join(run_dir, 'run.json'))} 2>/dev/null", timeout=60)
        if cat.ok and cat.stdout.strip():
            try:
                return RunManifest.from_json(cat.stdout)
            except (ValueError, TypeError):
                pass
        return self._synthesize_manifest(run_dir)

    def _synthesize_manifest(self, run_dir: str) -> RunManifest:
        run_id = posixpath.basename(run_dir.rstrip("/"))
        data: dict[str, Any] = {}
        env = self._run(f"cat {shlex.quote(posixpath.join(run_dir, 'submission_summary.env'))} 2>/dev/null", timeout=60)
        if env.ok and env.stdout.strip():
            data.update(parse_submission_env(env.stdout))
        status = self._run(f"cat {shlex.quote(run_dir)}/o2ctl_status/*status*.json 2>/dev/null", timeout=60)
        result_status = None
        if status.ok and status.stdout.strip():
            merged = merge_status_json(status.stdout)
            result_status = merged.pop("result_status", None)
            for key in ("datasets", "experiment_ids", "slurm_job_ids"):
                if merged.get(key):
                    union = set(data.get(key, [])) | set(merged[key])
                    data[key] = sort_job_ids(union) if key == "slurm_job_ids" else sorted(union)
        created = self._stat_mtime(run_dir)
        return RunManifest(
            run_id=run_id,
            campaign=campaign_of(run_id, self.policy.view_suffixes),
            variant=variant_of(run_id, self.policy.view_suffixes),
            pipeline=_infer_pipeline(run_id, self.policy),
            created_utc=created,
            status=STATUS_ACTIVE,
            datasets=data.get("datasets", []),
            experiment_ids=data.get("experiment_ids", []),
            slurm_job_ids=data.get("slurm_job_ids", []),
            storage_roots=data.get("storage_roots", []),
            result={"status": result_status} if result_status else {},
        )

    def _stat_mtime(self, path: str) -> str:
        res = self._run(f"date -u -d @$(stat -c %Y {shlex.quote(path)}) +%Y%m%dT%H%M%SZ 2>/dev/null", timeout=30)
        return res.stdout.strip() if res.ok else ""

    def _dir_exists(self, path: str) -> bool:
        """Whether ``path`` is an existing directory on O2 (``test -d``)."""
        return self._run(f"test -d {shlex.quote(path)}", timeout=30).ok

    # -- register --------------------------------------------------------------
    def register(
        self,
        *,
        campaign: str,
        pipeline: str,
        datasets: list[str],
        variant: str = "",
        derived_from: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Create an active run dir per convention and seed its run.json.

        Returns ``{ok, run_id, run_dir, problems}``. Refuses (ok=False) when the
        required classification fields are missing — there are no unclassified runs.
        """
        result, _manifest = self._register(
            campaign=campaign,
            pipeline=pipeline,
            datasets=datasets,
            variant=variant,
            derived_from=derived_from,
            run_id=run_id,
        )
        return result

    def _register(
        self,
        *,
        campaign: str,
        pipeline: str,
        datasets: list[str],
        variant: str = "",
        derived_from: str | None = None,
        run_id: str | None = None,
    ) -> tuple[dict[str, Any], RunManifest | None]:
        """Register backend that also returns the built manifest object.

        The public :meth:`register` discards the manifest (its result is JSON-encoded
        by the MCP layer); :meth:`submit_run` keeps it so the follow-up job-id record
        needs no read-back of the just-written run.json.
        """
        ts = self.utc_now()
        slug_variant = variant or "run"
        rid = run_id or f"RUN_{ts}_{_safe_slug(campaign)}__{_safe_slug(slug_variant)}"
        manifest = RunManifest(
            run_id=rid,
            campaign=campaign,
            variant=variant,
            pipeline=pipeline,
            created_utc=ts,
            status=STATUS_ACTIVE,
            datasets=list(datasets),
            source_view={"derived_from": derived_from, "materialized": False} if derived_from else {},
        )
        problems = manifest.validate(for_register=True)
        if problems:
            return {"ok": False, "problems": problems, "run_id": rid}, None
        run_dir = self.layout.run_dir(STATUS_ACTIVE, campaign, rid)
        for command in plan_register_commands(self.layout, manifest, self.policy.run_subdirs):
            res = self._run(command, timeout=60)
            if not res.ok:
                return {"ok": False, "problems": [res.stderr.strip() or "register command failed"], "run_id": rid}, None
        self.append_registry(manifest)
        return {"ok": True, "run_id": rid, "run_dir": run_dir, "problems": []}, manifest

    # -- record a submitted job onto an existing run ---------------------------
    def record_job(
        self,
        run_dir: str,
        job_id: str,
        *,
        manifest: RunManifest | None = None,
        result_status: str | None = None,
    ) -> dict[str, Any]:
        """Record a Slurm ``job_id`` on a run's manifest (and refresh its registry row).

        Adds the id to ``slurm_job_ids`` (deduplicated, sorted), rewrites run.json in
        place, and appends an updated registry line so the durable index reflects the
        submission. Pass ``manifest`` to skip the read-back when the caller already
        holds it; otherwise the manifest is read (synthesized from legacy metadata if
        absent). Returns ``{ok, run_id, slurm_job_ids}``.
        """
        manifest = manifest or self.read_manifest(run_dir)
        if manifest is None:
            return {"ok": False, "problems": [f"no run metadata under {run_dir}"]}
        jid = str(job_id).strip()
        if jid and not jid.isdigit():
            return {
                "ok": False,
                "problems": [f"job_id {job_id!r} is not a numeric Slurm id"],
                "run_id": manifest.run_id,
            }
        if jid and jid not in manifest.slurm_job_ids:
            manifest.slurm_job_ids = sort_job_ids(set(manifest.slurm_job_ids) | {jid})
        if result_status:
            manifest.result = {**(manifest.result or {}), "status": result_status}
        write = self._run(plan_write_manifest_command(run_dir, manifest), timeout=60)
        if not write.ok:
            return {
                "ok": False,
                "problems": [write.stderr.strip() or "manifest write failed"],
                "run_id": manifest.run_id,
            }
        self.append_registry(manifest)
        return {"ok": True, "run_id": manifest.run_id, "slurm_job_ids": list(manifest.slurm_job_ids)}

    # -- submit (register-or-attach + sbatch + record) -------------------------
    def submit_run(
        self,
        *,
        remote_script_path: str | None = None,
        script_text: str | None = None,
        remote_path: str | None = None,
        sbatch_args: list[str] | None = None,
        run_dir: str | None = None,
        campaign: str | None = None,
        pipeline: str | None = None,
        datasets: list[str] | None = None,
        variant: str = "",
        derived_from: str | None = None,
    ) -> dict[str, Any]:
        """Register (or attach to) a run, submit a job into it, and record the job id.

        The dogfooding submit path: every job is tied to a registered run so the
        durable registry stays the source of truth for what ran on which dataset.
        Provide EITHER ``run_dir`` (attach to an existing run) OR
        ``campaign`` + ``pipeline`` + ``datasets`` (register a fresh run first). The
        job script may reference the run via the ``{RUN_ROOT}`` / ``{RUN_ID}``
        placeholders, substituted once the run dir is known; ``script_text`` is staged
        into the run dir by default. Returns
        ``{ok, run_id, run_dir, registered, submitted, job_id, record, ...}``.
        """
        # 1. Resolve the run: attach to an existing dir, or register a fresh one.
        registered = False
        if run_dir:
            # read_manifest always SYNTHESIZES a manifest (never None), so guard against
            # attaching to a non-existent / non-run directory explicitly: the dir must
            # exist on O2 and yield a well-formed manifest (valid RUN_<ts>Z_<slug> id).
            # Otherwise a bogus run_dir would fabricate a manifest and orphan the job.
            if not self._dir_exists(run_dir):
                return {"ok": False, "error": "unknown_run", "message": f"run dir does not exist on O2: {run_dir}"}
            manifest = self.read_manifest(run_dir)
            if manifest is None or manifest.validate():
                return {"ok": False, "error": "unknown_run", "message": f"no valid run manifest under {run_dir}"}
            run_id = manifest.run_id
        elif campaign and pipeline and datasets:
            reg, manifest = self._register(
                campaign=campaign,
                pipeline=pipeline,
                datasets=list(datasets),
                variant=variant,
                derived_from=derived_from,
            )
            if not reg.get("ok") or manifest is None:
                return {"ok": False, "error": "register_failed", **reg}
            run_dir, run_id, registered = reg["run_dir"], reg["run_id"], True
        else:
            return {
                "ok": False,
                "error": "bad_input",
                "message": "Provide run_dir to attach, or campaign+pipeline+datasets to register a new run.",
            }

        # 2. Submit, substituting the now-known run paths into the script/paths.
        def _fill(text: str | None) -> str | None:
            if text is None:
                return None
            return text.replace("{RUN_ROOT}", run_dir).replace("{RUN_ID}", run_id)

        slurm = O2Slurm(self.conn)
        if script_text is not None:
            dest = _fill(remote_path) or posixpath.join(run_dir, "job.sbatch")
            submit = slurm.submit_text(_fill(script_text) or "", dest, sbatch_args=sbatch_args)
        elif remote_script_path:
            submit = slurm.submit(_fill(remote_script_path) or remote_script_path, sbatch_args=sbatch_args)
        else:
            return {
                "ok": False,
                "error": "bad_input",
                "message": "Provide remote_script_path or script_text (with optional remote_path).",
                "run_id": run_id,
                "run_dir": run_dir,
                "registered": registered,
            }

        payload: dict[str, Any] = {
            "ok": submit.submitted,
            "run_id": run_id,
            "run_dir": run_dir,
            "registered": registered,
            "submitted": submit.submitted,
            "job_id": submit.job_id,
            "returncode": submit.command.returncode,
            "stdout": submit.command.stdout,
            "stderr": submit.command.stderr,
        }
        # 3. Record the job id onto the run. Non-fatal if it fails: the run is already
        # in the registry (register appended its row before submit) and the job is
        # live, so `ok` stays True (= job submitted, consistent with o2_submit_job) and
        # we surface a warning + record.ok=False rather than risk a double-submit by
        # reporting overall failure. Only the job-id annotation on run.json is missing;
        # the returned job_id still tracks it and it can be re-recorded.
        if submit.submitted and submit.job_id:
            record = self.record_job(run_dir, submit.job_id, manifest=manifest)
            payload["record"] = record
            if not record.get("ok"):
                payload["record_warning"] = (
                    f"job {submit.job_id} is running and the run is registered, but writing the job id onto "
                    f"run.json failed ({'; '.join(record.get('problems', []))}); re-record it later."
                )
        return payload

    # -- classify --------------------------------------------------------------
    def classify(self, root: str | None = None, *, depth_grouped: bool = False) -> list[dict[str, Any]]:
        """Read every run under ``root`` and tag it keep/sweep (advisory)."""
        run_dirs = self.list_run_dirs(root, depth_grouped=depth_grouped)
        manifests: list[tuple[str, RunManifest]] = []
        for run_dir in run_dirs:
            manifest = self.read_manifest(run_dir)
            if manifest is not None:
                manifests.append((run_dir, manifest))
        latest_by_campaign: dict[str, str] = {}
        for _run_dir, manifest in manifests:
            current = latest_by_campaign.get(manifest.campaign)
            if current is None or manifest.run_id > current:
                latest_by_campaign[manifest.campaign] = manifest.run_id
        rows: list[dict[str, Any]] = []
        for run_dir, manifest in manifests:
            verdict = classify_run(
                manifest.run_id,
                self.policy,
                result_status=(manifest.result or {}).get("status"),
                is_latest_in_campaign=(latest_by_campaign.get(manifest.campaign) == manifest.run_id),
            )
            rows.append(
                {
                    "run_id": manifest.run_id,
                    "run_dir": run_dir,
                    "campaign": manifest.campaign,
                    "datasets": manifest.datasets,
                    "result_status": (manifest.result or {}).get("status"),
                    "retention": verdict["retention"],
                    "reason": verdict["reason"],
                }
            )
        return sorted(rows, key=lambda r: (r["campaign"], r["run_id"]))

    # -- tier transitions (run detached on the transfer node) ------------------
    # Standby is writable only from the O2 transfer node (login/compute nodes
    # cannot write it), and a tar+zstd of a large run takes hours, so transitions
    # are launched DETACHED on the transfer node and return immediately. The script
    # verifies (checksum / rsync-itemize) before it frees any scratch source.
    def promote(self, run_dir: str, *, dry_run: bool = True, run_remote: bool = True) -> TransitionPlan:
        manifest = self.read_manifest(run_dir) or self._synthesize_manifest(run_dir)
        script = plan_promote_script(self.layout, manifest, source_dir=run_dir)
        return self._transition(manifest.run_id, "promote", script, dry_run=dry_run, run_remote=run_remote)

    def archive(self, run_dir: str, *, dry_run: bool = True, run_remote: bool = True) -> TransitionPlan:
        manifest = self.read_manifest(run_dir) or self._synthesize_manifest(run_dir)
        script = plan_archive_script(
            self.layout, manifest, source_dir=run_dir, archive_excludes=self.policy.archive_excludes
        )
        return self._transition(manifest.run_id, "archive", script, dry_run=dry_run, run_remote=run_remote)

    def _transition(self, run_id: str, action: str, script: str, *, dry_run: bool, run_remote: bool) -> TransitionPlan:
        if dry_run or not run_remote:
            return TransitionPlan(run_id, action, script, started=False, message="dry_run: script not executed")
        script_path = posixpath.join(self.layout.scratch_runs_root, ".jobs", f"{action}_{run_id}.sh")
        log_path = script_path + ".log"
        self._run(f"mkdir -p {shlex.quote(posixpath.dirname(script_path))}", timeout=60)
        # stage the script body verbatim via stdin (cat writes exactly what it reads)
        stage = self.conn.run(f"cat > {shlex.quote(script_path)}", timeout=60, input_text=script)
        if not stage.ok:
            return TransitionPlan(
                run_id, action, script, started=False, message=stage.stderr.strip() or "staging failed"
            )
        launch = f"nohup bash {shlex.quote(script_path)} > {shlex.quote(log_path)} 2>&1 < /dev/null & echo PID $!"
        argv = ["ssh", *self.conn.config.base_ssh_opts(), self.conn.config.transfer_alias, launch]
        # This raw ssh targets the transfer node, so verify the transfer master
        # (not the login master) — otherwise a down transfer master would let ssh
        # open a fresh Duo-pushing login here.
        res = self.conn.run_raw(argv, timeout=60, master_alias=self.conn.config.transfer_alias)
        pid = ""
        for token in res.stdout.split():
            if token.isdigit():
                pid = token
        return TransitionPlan(
            run_id,
            action,
            script,
            started=res.ok and bool(pid),
            pid=pid or None,
            log_path=log_path,
            message=f"launched on transfer node (pid {pid}); tail {log_path}" if pid else res.stderr.strip(),
        )

    # -- registry --------------------------------------------------------------
    def append_registry(self, manifest: RunManifest) -> CommandResult:
        line = registry_line(manifest)
        path = shlex.quote(self.layout.registry_path)
        return self._run(
            f"mkdir -p {shlex.quote(posixpath.dirname(self.layout.registry_path))} && "
            f"printf '%s\\n' {shlex.quote(line)} >> {path}",
            timeout=60,
        )

    def load_registry(self) -> list[dict[str, Any]]:
        res = self._run(f"cat {shlex.quote(self.layout.registry_path)} 2>/dev/null", timeout=60)
        return parse_registry(res.stdout) if res.ok else []

    # -- gc --------------------------------------------------------------------
    def gc_candidates(self, *, older_than_days: int = 30) -> list[dict[str, str]]:
        res = self._run(plan_gc_candidates_command(self.layout, older_than_days=older_than_days), timeout=120)
        rows: list[dict[str, str]] = []
        for line in res.stdout.splitlines():
            if "\t" in line:
                mtime, path = line.split("\t", 1)
                rows.append({"mtime": mtime, "path": path})
        return rows


# --- module helpers ----------------------------------------------------------
def _safe_slug(text: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9.-]+", "-", str(text).strip().lower()).strip("-") or "x"


def _infer_pipeline(run_id: str, policy: RunPolicy) -> str:
    """Map a run-id to a pipeline via the policy's ordered (substring → pipeline) rules."""
    slug = run_id.lower()
    for needle, pipeline in policy.pipeline_keywords:
        if needle in slug:
            return pipeline
    return policy.fallback_pipeline
