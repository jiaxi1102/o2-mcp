"""Monotonic, cross-process-safe execution registry synchronization.

``run.json`` and the append-only registry cannot be replaced atomically as one
filesystem object.  This module therefore uses a compare-and-swap transaction
under a remote ``flock``: publish the merged manifest atomically, then fsync one
registry row.  A crash between those operations is repaired by replaying the
same update; no older update can overwrite newer run state.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from typing import Any

from o2mcp.connection import O2Connection
from o2mcp.runorg.execution_models import RegistryUpdate
from o2mcp.runorg.lifecycle_coordination import coordination_lock, coordination_root
from o2mcp.runorg.plans import ExecutionPlan
from o2mcp.runorg.runs import RunManifest, registry_line, sort_job_ids

_STAGE_STATUS_RANK = {
    "SUBMITTED": 10,
    "WAIT": 20,
    "RETRY_MISSING_ONLY": 30,
    "RETRYING": 40,
    "RETRY_SUBMITTED": 50,
    "COMPLETED": 100,
    # Integrity failure dominates completion regardless of callback ordering.
    # Equal terminal ranks made the registry's meaning depend on a race.
    "FAILED": 110,
}


def merge_execution_manifest(
    manifest: RunManifest,
    plan: ExecutionPlan,
    update: RegistryUpdate,
) -> RunManifest:
    """Return a lossless monotonic manifest update.

    The function is pure so adversarial ordering tests can exercise it without
    O2.  Attempts may advance but never retreat.  Terminal plan states are sticky
    because a late audit or delayed submission callback must not resurrect a
    failed/completed run as merely active.
    """

    merged = RunManifest.from_dict(manifest.to_dict())
    provenance = dict(merged.provenance or {})
    previous_execution = dict(provenance.get("execution", {}))
    previous_plan_sha = previous_execution.get("plan_sha256")
    if previous_plan_sha not in {None, plan.plan_sha256}:
        raise ValueError("run.json is already bound to a different execution plan")

    stages = dict(previous_execution.get("stages", {}))
    previous_stage = stages.get(update.stage_id)
    accept_stage_update = True
    if isinstance(previous_stage, dict):
        previous_attempt = previous_stage.get("attempt")
        previous_status = previous_stage.get("status")
        if type(previous_attempt) is not int or type(previous_status) is not str:
            raise ValueError("run.json execution stage state has invalid exact types")
        if (
            previous_attempt > update.attempt
            or previous_attempt == update.attempt
            and _STAGE_STATUS_RANK.get(previous_status, 0) > _STAGE_STATUS_RANK.get(update.stage_status, 0)
        ):
            accept_stage_update = False
        elif previous_attempt == update.attempt and {previous_status, update.stage_status} == {
            "COMPLETED",
            "FAILED",
        }:
            # Apply the terminal lattice explicitly rather than relying on which
            # concurrent callback happened to acquire the registry lock last.
            stages[update.stage_id] = {"attempt": update.attempt, "status": "FAILED"}
            accept_stage_update = False
    if accept_stage_update:
        stages[update.stage_id] = {"attempt": update.attempt, "status": update.stage_status}

    previous_state = previous_execution.get("state", "")
    if type(previous_state) is not str:
        raise ValueError("run.json execution state must be a string")
    execution_state = update.execution_status
    if previous_state == "FAILED" or execution_state == "FAILED":
        execution_state = "FAILED"
    elif previous_state == "COMPLETED":
        execution_state = "COMPLETED"

    provenance["execution"] = {
        **previous_execution,
        "plan_sha256": plan.plan_sha256,
        "project": plan.project,
        "source_bundle_sha256": plan.source_bundle_sha256,
        "source_commit": plan.source_commit,
        "state": execution_state,
        "stages": stages,
    }
    merged.provenance = provenance
    merged.slurm_job_ids = sort_job_ids(set(merged.slurm_job_ids) | set(update.job_ids))
    merged.result = {**(merged.result or {}), "status": execution_state}
    return merged


def synchronize_execution_transaction(
    connection: O2Connection,
    *,
    run_dir: str,
    registry_path: str,
    current_manifest_text: str,
    merged_manifest: RunManifest,
) -> dict[str, Any]:
    """CAS-publish ``run.json`` and append its registry row under one remote lock."""

    payload = json.dumps(
        {
            "expected_manifest_sha256": hashlib.sha256(current_manifest_text.encode("utf-8")).hexdigest(),
            "manifest_text": merged_manifest.to_json(),
            "registry_line": registry_line(merged_manifest),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    program = _transaction_program()
    result = connection.run(
        " ".join(
            (
                "python3 -c",
                shlex.quote(program),
                shlex.quote(run_dir),
                shlex.quote(registry_path),
                shlex.quote(coordination_root(run_dir)),
                shlex.quote(coordination_lock(run_dir)),
            )
        ),
        timeout=120,
        input_text=payload,
    )
    if not result.ok:
        return {
            "ok": False,
            "error": (
                "concurrent_update"
                if result.returncode == 43
                else "lifecycle_transition_in_progress" if result.returncode == 44 else "registry_transaction_failed"
            ),
            "problems": [result.stderr.strip() or result.stdout.strip() or "execution registry transaction failed"],
        }
    return {
        "ok": True,
        "run_id": merged_manifest.run_id,
        "execution_status": (merged_manifest.result or {}).get("status"),
        "slurm_job_ids": list(merged_manifest.slurm_job_ids),
    }


def _transaction_program() -> str:
    """Return the dependency-free remote CAS transaction program."""

    return r"""
import fcntl, hashlib, json, os, stat, sys, tempfile
run_dir, registry_path, coordination, lifecycle_lock_path = sys.argv[1:5]
payload = json.load(sys.stdin)
manifest_path = os.path.join(run_dir, 'run.json')
registry_lock_path = os.path.join(run_dir, '.execution-registry.lock')

# Transition marking takes this same sibling lock. The check and every run-root
# or registry mutation therefore happen before a transition can quarantine the
# source, without relying on a persistent reconciliation claim.
os.makedirs(os.path.dirname(coordination), exist_ok=True)
lifecycle_lock = os.open(
    lifecycle_lock_path,
    os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
    0o600,
)
try:
    if not stat.S_ISREG(os.fstat(lifecycle_lock).st_mode):
        raise SystemExit('coordination lock is not regular')
    fcntl.flock(lifecycle_lock, fcntl.LOCK_EX)
    if os.path.lexists(coordination) and (
        os.path.islink(coordination)
        or not stat.S_ISDIR(os.stat(coordination, follow_symlinks=False).st_mode)
    ):
        raise SystemExit('coordination root is not a real directory')
    if os.path.lexists(os.path.join(coordination, 'transition.json')):
        raise SystemExit(44)

    with open(registry_lock_path, 'a+', encoding='utf-8') as registry_lock:
        fcntl.flock(registry_lock.fileno(), fcntl.LOCK_EX)
        with open(manifest_path, encoding='utf-8') as handle:
            current = handle.read()
        if hashlib.sha256(current.encode('utf-8')).hexdigest() != payload['expected_manifest_sha256']:
            raise SystemExit(43)
        fd, temporary = tempfile.mkstemp(prefix='.run-json-', dir=run_dir)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                handle.write(payload['manifest_text'])
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, manifest_path)
            directory_fd = os.open(run_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
                cleanup_directory_fd = os.open(run_dir, os.O_RDONLY)
                try:
                    os.fsync(cleanup_directory_fd)
                finally:
                    os.close(cleanup_directory_fd)
        registry_parent = os.path.dirname(registry_path)
        os.makedirs(registry_parent, exist_ok=True)
        with open(registry_path, 'a', encoding='utf-8') as registry:
            registry.write(payload['registry_line'] + '\n')
            registry.flush()
            os.fsync(registry.fileno())
finally:
    os.close(lifecycle_lock)
print(json.dumps({'ok': True}, sort_keys=True))
""".strip()


__all__ = ["merge_execution_manifest", "synchronize_execution_transaction"]
