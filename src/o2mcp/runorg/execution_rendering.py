"""Deterministic Slurm dispatcher and intent rendering for execution plans.

The coordinator owns state transitions, while this module owns the exact bytes
handed to Slurm.  Keeping shell construction isolated makes it practical to
review quoting, runtime verification, sparse-array dispatch, and scheduler intent
without growing the state machine beyond the repository's file-size budget.
"""

from __future__ import annotations

import hashlib
import posixpath
import shlex
from collections.abc import Sequence

from o2mcp.runorg.execution_models import PlannedTask, SubmissionIdentity, SubmissionRequest
from o2mcp.runorg.execution_reconcile import task_sort_key
from o2mcp.runorg.plan_stages import CommandSpec, StageSpec
from o2mcp.runorg.plans import ExecutionPlan


def select_tasks(stage: StageSpec, task_ids: Sequence[str] | None) -> tuple[PlannedTask, ...]:
    """Resolve an exact task subset without changing stable array indices."""

    if stage.tasks:
        available = {
            task.task_id: PlannedTask(task.task_id, task.array_index, task.command, task.expected_receipts)
            for task in stage.tasks
        }
    else:
        assert stage.command is not None  # guaranteed by StageSpec validation
        available = {
            stage.stage_id: PlannedTask(stage.stage_id, None, stage.command, stage.expected_receipts),
        }
    selected_ids = tuple(sorted(available)) if task_ids is None else tuple(task_ids)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selected task IDs must be a non-empty unique set")
    unknown = sorted(set(selected_ids) - set(available))
    if unknown:
        raise ValueError(f"unknown task IDs for stage {stage.stage_id}: {unknown}")
    return tuple(sorted((available[task_id] for task_id in selected_ids), key=task_sort_key))


def build_submission_request(
    plan: ExecutionPlan,
    stage: StageSpec,
    identity: SubmissionIdentity,
    tasks: tuple[PlannedTask, ...],
    dependency_job_ids: tuple[str, ...],
) -> SubmissionRequest:
    """Render one deterministic dispatcher and its typed scheduler request."""

    attempt_name = f"attempt-{identity.attempt:03d}"
    stage_work = posixpath.join(plan.paths.work_root, "execution", stage.stage_id)
    stage_logs = posixpath.join(plan.paths.logs_root, "execution", stage.stage_id, attempt_name)
    script_path = posixpath.join(stage_work, attempt_name + ".sbatch")
    suffix = "%A_%a" if tasks[0].array_index is not None else "%j"
    return SubmissionRequest(
        identity=identity,
        run_id=plan.run_id,
        tasks=tasks,
        resources=stage.resources,
        dependency_mode=stage.dependency_mode,
        dependency_job_ids=dependency_job_ids,
        script_path=script_path,
        script_text=render_dispatcher(tasks, run_root=plan.paths.run_root),
        stdout_pattern=posixpath.join(stage_logs, suffix + ".out"),
        stderr_pattern=posixpath.join(stage_logs, suffix + ".err"),
        begin_delay_seconds=stage.retry_policy.backoff_seconds if identity.attempt > 1 else 0,
    )


def render_dispatcher(tasks: Sequence[PlannedTask], *, run_root: str | None = None) -> str:
    """Render a deterministic Bash dispatcher without ``eval`` or templates.

    Sparse and zero-based Slurm indices are matched explicitly.  Each selected
    command verifies its signed runtime fingerprint before changing directory or
    executing scientific code; a changed wrapper/interpreter therefore cannot
    run under the reviewed plan SHA.
    """

    lines = ["#!/bin/bash", "set -euo pipefail"]
    if run_root is not None:
        # Scientific tasks hold a shared lifecycle lock.  Promotion/archive takes
        # the exclusive form of the same lock, closing the check-to-delete window
        # after the control plane has verified that Slurm has no live jobs.
        lock_path = shlex.quote(posixpath.join(run_root, ".execution-source.lock"))
        lines.extend([f"exec 8> {lock_path}", "flock -s 8"])
    if tasks[0].array_index is None:
        lines.extend(command_lines(tasks[0].command))
    else:
        lines.extend(['case "${SLURM_ARRAY_TASK_ID:?missing array task id}" in'])
        for task in tasks:
            lines.append(f"  {task.array_index})")
            lines.extend(f"    {line}" for line in command_lines(task.command))
            lines.append("    ;;")
        lines.extend(["  *) echo 'unplanned array task index' >&2; exit 64 ;;", "esac"])
    return "\n".join(lines) + "\n"


def command_lines(command: CommandSpec) -> list[str]:
    """Copy, verify, and execute immutable runtime bytes in a clean environment.

    Hashing a pathname and later executing that pathname has a rename race.  We
    first copy the executable to a private file, hash the copy, and execute that
    same private inode.  A source replacement before/during the copy either
    yields the signed bytes or fails the digest check; a later replacement is
    irrelevant to the already staged runtime.
    """

    runtime_path = shlex.quote(command.runtime_fingerprint_path)
    expected = shlex.quote(command.runtime_fingerprint_sha256)
    environment = " ".join(shlex.quote(item) for item in command.environment)
    argv_tail = " ".join(shlex.quote(item) for item in command.argv[1:])
    env_prefix = f"env -i {environment} " if environment else "env -i "
    return [
        'runtime_copy=$(mktemp "${TMPDIR:-/tmp}/o2mcp-runtime.XXXXXX")',
        f'/bin/cp -p -- {runtime_path} "$runtime_copy"',
        'runtime_sha=$(/usr/bin/sha256sum -- "$runtime_copy")',
        "runtime_sha=${runtime_sha%% *}",
        f'if [[ "$runtime_sha" != {expected} ]]; then rm -f -- "$runtime_copy"; '
        "echo 'runtime fingerprint mismatch' >&2; exit 70; fi",
        f"cd -- {shlex.quote(command.working_directory)}",
        "set +e",
        f'{env_prefix}"$runtime_copy" {argv_tail}'.rstrip(),
        "runtime_status=$?",
        "set -e",
        'rm -f -- "$runtime_copy"',
        'exit "$runtime_status"',
    ]


def submission_intent(request: SubmissionRequest) -> dict[str, object]:
    """Return the immutable pre-sbatch payload used to block duplicate retries."""

    return {
        "attempt": request.identity.attempt,
        "comment": request.comment,
        "dependency_job_ids": list(request.dependency_job_ids),
        "dependency_mode": request.dependency_mode,
        "plan_sha256": request.identity.plan_sha256,
        "sbatch_args": list(request.sbatch_args()),
        "schema_version": 1,
        "script_path": request.script_path,
        "script_sha256": hashlib.sha256(request.script_text.encode("utf-8")).hexdigest(),
        "stage_id": request.identity.stage_id,
        "task_ids": [task.task_id for task in request.tasks],
        "task_indices": [task.array_index for task in request.tasks if task.array_index is not None],
    }


__all__ = ["build_submission_request", "render_dispatcher", "select_tasks", "submission_intent"]
