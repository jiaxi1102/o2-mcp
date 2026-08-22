"""Fakeable Slurm and remote-filesystem boundary for execution plans.

The coordinator in :mod:`o2mcp.runorg.execution_engine` performs no SSH and no
shell parsing.  It depends on :class:`ExecutionBackend`, which can be implemented
by an in-memory test double or by :class:`O2ExecutionBackend` over an already
authenticated :class:`~o2mcp.connection.O2Connection`.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import shlex
from collections.abc import Sequence
from typing import Protocol

from o2mcp.connection import O2Connection
from o2mcp.runorg.execution_models import (
    ReceiptObservation,
    SlurmJob,
    SlurmTaskState,
    SubmissionRequest,
    SubmitOutcome,
)
from o2mcp.runorg.plan_components import ReceiptSpec
from o2mcp.slurm import O2Slurm


class ExecutionBackend(Protocol):
    """Operations required by the project-neutral execution coordinator."""

    def find_jobs(self, comment: str) -> Sequence[SlurmJob]:
        """Return every root Slurm job whose comment exactly matches ``comment``."""

    def submit(self, request: SubmissionRequest) -> SubmitOutcome:
        """Stage and submit one exact request without implementing retries."""

    def task_states(self, job_id: str) -> Sequence[SlurmTaskState]:
        """Return sacct-like root and array-element states for ``job_id``."""

    def observe_receipt(self, run_root: str, receipt: ReceiptSpec) -> ReceiptObservation:
        """Observe existence and SHA-256 for one expected pipeline receipt."""

    def read_text(self, path: str) -> str | None:
        """Read text or return ``None`` when the path does not exist."""

    def write_immutable_text(self, path: str, text: str) -> bool:
        """Publish complete text exactly once.

        Returns ``True`` only to the caller that atomically created ``path`` and
        ``False`` for a byte-identical replay.  Submission ownership depends on
        this distinction: merely observing identical bytes must never authorize
        a second caller to invoke ``sbatch``.
        """

    def write_mutable_text(self, path: str, text: str) -> None:
        """Atomically replace a current-state file that is not an evidence receipt."""

    def compare_and_swap_text(self, path: str, expected: str | None, replacement: str | None) -> bool:
        """Atomically replace/remove ``path`` only when exact current bytes match."""


class O2ExecutionBackend:
    """Production execution boundary using one established O2 connection.

    This class never starts a login and never retries ``sbatch``.  A lost submit
    response is reconciled by the engine through :meth:`find_jobs`, using the
    immutable comment identity that was part of the original request.
    """

    def __init__(self, connection: O2Connection) -> None:
        self.connection = connection
        self.slurm = O2Slurm(connection)

    def find_jobs(self, comment: str) -> Sequence[SlurmJob]:
        """Search live and recent accounting records for an exact Slurm comment."""

        # squeue catches newly accepted jobs before accounting ingestion; sacct
        # catches fast-completing jobs that already left the queue.  The seven-day
        # window is intentionally bounded because recovery happens immediately
        # around submission, not as an unscoped historical search.
        command = "\n".join(
            [
                "set +e",
                "squeue -u \"$USER\" -h -o '%i|%k|%T' 2>&1",
                "squeue_status=$?",
                "sacct -X -S now-7days -n -P --format=JobIDRaw,Comment,State 2>&1",
                "sacct_status=$?",
                'printf \'__O2MCP_QUERY_STATUS__|%s|%s\\n\' "$squeue_status" "$sacct_status"',
            ]
        )
        result = self.connection.run(command, timeout=60)
        if not result.ok:
            raise RuntimeError(f"Slurm identity query failed: {result.stderr.strip() or result.stdout.strip()}")
        lines = result.stdout.splitlines()
        status_lines = [line for line in lines if line.startswith("__O2MCP_QUERY_STATUS__|")]
        if len(status_lines) != 1:
            raise RuntimeError("Slurm identity query did not return its completion sentinel")
        status_parts = status_lines[0].split("|")
        if len(status_parts) != 3 or status_parts[1:] != ["0", "0"]:
            raise RuntimeError(
                "Slurm identity query was incomplete: "
                f"squeue={status_parts[1] if len(status_parts) > 1 else '?'} "
                f"sacct={status_parts[2] if len(status_parts) > 2 else '?'}"
            )
        jobs: dict[str, SlurmJob] = {}
        for line in lines:
            if line.startswith("__O2MCP_QUERY_STATUS__|"):
                continue
            parts = line.strip().split("|", 2)
            if len(parts) != 3 or parts[1] != comment:
                continue
            raw_job_id, _recorded_comment, state = parts
            root_job_id = raw_job_id.split("_", 1)[0].split(".", 1)[0]
            if root_job_id.isdigit():
                jobs[root_job_id] = SlurmJob(root_job_id, comment, state)
        return tuple(sorted(jobs.values(), key=lambda item: int(item.job_id)))

    def submit(self, request: SubmissionRequest) -> SubmitOutcome:
        """Stage the immutable dispatcher and invoke ``sbatch`` exactly once."""

        self.write_immutable_text(request.script_path, request.script_text)
        log_parents = sorted(
            {
                posixpath.dirname(request.stdout_pattern),
                posixpath.dirname(request.stderr_pattern),
            }
        )
        mkdir = self.connection.run(
            "mkdir -p " + " ".join(shlex.quote(path) for path in log_parents),
            timeout=60,
        )
        if not mkdir.ok:
            return SubmitOutcome(None, accepted=False, stderr=mkdir.stderr or "could not create log directory")
        submit = self.slurm.submit(request.script_path, sbatch_args=list(request.sbatch_args()), timeout=60)
        return SubmitOutcome(
            job_id=submit.job_id,
            accepted=submit.submitted,
            response_received=True,
            stdout=submit.command.stdout,
            stderr=submit.command.stderr,
        )

    def task_states(self, job_id: str) -> Sequence[SlurmTaskState]:
        """Read root and array-element states from Slurm accounting."""

        result = self.connection.run(
            f"sacct -j {shlex.quote(job_id)} -X -n -P --format=JobIDRaw,State,ExitCode",
            timeout=60,
        )
        if not result.ok:
            raise RuntimeError(
                f"Slurm accounting query failed for job {job_id}: {result.stderr.strip() or result.stdout.strip()}"
            )
        states: list[SlurmTaskState] = []
        for line in result.stdout.splitlines():
            parts = line.strip().split("|", 2)
            if len(parts) != 3:
                continue
            raw_job_id, state, exit_code_text = parts
            # Job steps (.batch/.extern) are implementation detail.  Root array
            # records use JOBID_INDEX, including the valid zero index.
            if "." in raw_job_id:
                continue
            array_index = None
            if "_" in raw_job_id:
                root, index_text = raw_job_id.split("_", 1)
                if root != job_id or not index_text.isdigit():
                    continue
                array_index = int(index_text)
            elif raw_job_id != job_id:
                continue
            exit_code = None
            first_exit = exit_code_text.split(":", 1)[0]
            if first_exit.isdigit():
                exit_code = int(first_exit)
            states.append(SlurmTaskState(array_index, state, exit_code))
        return tuple(states)

    def observe_receipt(self, run_root: str, receipt: ReceiptSpec) -> ReceiptObservation:
        """Hash an expected receipt without trusting shell formatting."""

        path = posixpath.join(run_root, receipt.path)
        program = (
            "import hashlib,json,os,sys;"
            "p=sys.argv[1];e=os.path.isfile(p);"
            "h=hashlib.sha256(open(p,'rb').read()).hexdigest() if e else None;"
            "print(json.dumps({'exists':e,'sha256':h},sort_keys=True))"
        )
        result = self.connection.run(
            f"python3 -c {shlex.quote(program)} {shlex.quote(path)}",
            timeout=120,
        )
        if not result.ok:
            # A failed read is not proof of absence.  Returning an explicit
            # untrustworthy observation makes reconciliation wait instead of
            # turning a transient SSH/filesystem failure into a scientific retry.
            return ReceiptObservation(
                receipt.path,
                False,
                None,
                trustworthy=False,
                error=result.stderr.strip() or result.stdout.strip() or "receipt observation command failed",
            )
        try:
            observed = json.loads(result.stdout)
            if not isinstance(observed, dict) or set(observed) != {"exists", "sha256"}:
                raise ValueError("unexpected receipt observation fields")
            exists = observed["exists"]
            sha256 = observed["sha256"]
            if not isinstance(exists, bool) or (sha256 is not None and not isinstance(sha256, str)):
                raise ValueError("invalid receipt observation types")
            return ReceiptObservation(receipt.path, exists, sha256)
        except (json.JSONDecodeError, ValueError) as exc:
            return ReceiptObservation(
                receipt.path,
                False,
                None,
                trustworthy=False,
                error=f"invalid receipt observation response: {exc}",
            )

    def read_text(self, path: str) -> str | None:
        """Read one remote UTF-8 file while distinguishing absence from emptiness."""

        result = self.connection.run(f"cat -- {shlex.quote(path)}", timeout=60)
        return result.stdout if result.ok else None

    def write_immutable_text(self, path: str, text: str) -> bool:
        """Atomically publish a complete remote evidence file.

        Writing directly into an ``O_EXCL`` destination leaves a truncated final
        file if the writer dies between ``open`` and ``fsync``.  Instead, write and
        fsync a private temporary file, then use ``link`` as the atomic no-replace
        publication primitive.  A competing caller can observe only complete
        bytes and receives ``False`` rather than submission ownership.
        """

        program = "\n".join(
            [
                "import os, sys, tempfile",
                "path = sys.argv[1]",
                "payload = sys.stdin.read()",
                "parent = os.path.dirname(path)",
                "os.makedirs(parent, exist_ok=True)",
                "fd, temporary = tempfile.mkstemp(prefix='.immutable-', dir=parent)",
                "try:",
                "    with os.fdopen(fd, 'w', encoding='utf-8') as handle:",
                "        handle.write(payload)",
                "        handle.flush()",
                "        os.fsync(handle.fileno())",
                "    try:",
                "        os.link(temporary, path)",
                "    except FileExistsError:",
                "        with open(path, encoding='utf-8') as handle:",
                "            existing = handle.read()",
                "        if existing != payload:",
                "            raise SystemExit(42)",
                "        print('EXISTING')",
                "    else:",
                "        directory_fd = os.open(parent, os.O_RDONLY)",
                "        try:",
                "            os.fsync(directory_fd)",
                "        finally:",
                "            os.close(directory_fd)",
                "        print('CREATED')",
                "finally:",
                "    try:",
                "        os.unlink(temporary)",
                "    except FileNotFoundError:",
                "        pass",
            ]
        )
        result = self.connection.run(
            f"python3 -c {shlex.quote(program)} {shlex.quote(path)}",
            timeout=120,
            input_text=text,
        )
        if not result.ok:
            reason = (
                "immutable receipt already exists with different bytes" if result.returncode == 42 else "write failed"
            )
            raise RuntimeError(f"{reason}: {path}: {result.stderr.strip()}")
        outcome = result.stdout.strip()
        if outcome not in {"CREATED", "EXISTING"}:
            raise RuntimeError(f"immutable write returned an invalid publication result for {path}: {outcome!r}")
        return outcome == "CREATED"

    def write_mutable_text(self, path: str, text: str) -> None:
        """Atomically replace a remote current-state file after fsync."""

        program = "\n".join(
            [
                "import os, sys, tempfile",
                "path = sys.argv[1]",
                "payload = sys.stdin.read()",
                "parent = os.path.dirname(path)",
                "os.makedirs(parent, exist_ok=True)",
                "fd, temporary = tempfile.mkstemp(prefix='.tmp-', dir=parent)",
                "try:",
                "    with os.fdopen(fd, 'w', encoding='utf-8') as handle:",
                "        handle.write(payload)",
                "        handle.flush()",
                "        os.fsync(handle.fileno())",
                "    os.replace(temporary, path)",
                "finally:",
                "    if os.path.exists(temporary):",
                "        os.unlink(temporary)",
            ]
        )
        result = self.connection.run(
            f"python3 -c {shlex.quote(program)} {shlex.quote(path)}",
            timeout=120,
            input_text=text,
        )
        if not result.ok:
            raise RuntimeError(f"atomic state write failed: {path}: {result.stderr.strip()}")

    def compare_and_swap_text(self, path: str, expected: str | None, replacement: str | None) -> bool:
        """Perform a lock-protected exact-byte CAS for mutable outbox state.

        Absence and an empty file are distinct.  The operation fsyncs both the
        replacement and its parent before releasing the sibling lock, allowing
        callers to merge and compare-clear one exact outbox payload safely.
        """

        payload = json.dumps({"expected": expected, "replacement": replacement}, ensure_ascii=False)
        program = "\n".join(
            [
                "import fcntl, json, os, sys, tempfile",
                "path = sys.argv[1]",
                "payload = json.load(sys.stdin)",
                "parent = os.path.dirname(path)",
                "os.makedirs(parent, exist_ok=True)",
                "lock_path = path + '.lock'",
                "with open(lock_path, 'a+', encoding='utf-8') as lock:",
                "    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)",
                "    try:",
                "        with open(path, encoding='utf-8') as handle:",
                "            current = handle.read()",
                "    except FileNotFoundError:",
                "        current = None",
                "    if current != payload['expected']:",
                "        print('CONFLICT')",
                "        raise SystemExit(0)",
                "    replacement = payload['replacement']",
                "    if replacement is None:",
                "        try:",
                "            os.unlink(path)",
                "        except FileNotFoundError:",
                "            pass",
                "    else:",
                "        fd, temporary = tempfile.mkstemp(prefix='.cas-', dir=parent)",
                "        try:",
                "            with os.fdopen(fd, 'w', encoding='utf-8') as handle:",
                "                handle.write(replacement)",
                "                handle.flush()",
                "                os.fsync(handle.fileno())",
                "            os.replace(temporary, path)",
                "        finally:",
                "            if os.path.exists(temporary):",
                "                os.unlink(temporary)",
                "    parent_fd = os.open(parent, os.O_RDONLY)",
                "    try:",
                "        os.fsync(parent_fd)",
                "    finally:",
                "        os.close(parent_fd)",
                "    print('SWAPPED')",
            ]
        )
        result = self.connection.run(
            f"python3 -c {shlex.quote(program)} {shlex.quote(path)}",
            timeout=120,
            input_text=payload,
        )
        if not result.ok:
            raise RuntimeError(f"mutable CAS failed: {path}: {result.stderr.strip()}")
        outcome = result.stdout.strip()
        if outcome not in {"CONFLICT", "SWAPPED"}:
            raise RuntimeError(f"mutable CAS returned invalid result for {path}: {outcome!r}")
        return outcome == "SWAPPED"


def receipt_matches(spec: ReceiptSpec, observation: ReceiptObservation) -> bool:
    """Return whether an observation satisfies one expected receipt contract."""

    if not observation.trustworthy:
        return False
    if not observation.exists:
        return not spec.required
    return spec.sha256 is None or spec.sha256 == observation.sha256


def text_sha256(text: str) -> str:
    """Return a small helper digest used by tests and dispatcher verification."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["ExecutionBackend", "O2ExecutionBackend", "receipt_matches", "text_sha256"]
