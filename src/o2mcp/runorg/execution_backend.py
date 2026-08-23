"""Fakeable Slurm and remote-filesystem boundary for execution plans.

The coordinator in :mod:`o2mcp.runorg.execution_engine` performs no SSH and no
shell parsing.  It depends on :class:`ExecutionBackend`, which can be implemented
by an in-memory test double or by :class:`O2ExecutionBackend` over an already
authenticated :class:`~o2mcp.connection.O2Connection`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import posixpath
import re
import shlex
from collections.abc import Sequence
from typing import Protocol

from o2mcp.connection import O2Connection
from o2mcp.runorg.execution_models import (
    ACCEPTED,
    DEFINITELY_NOT_INVOKED,
    DEFINITELY_REJECTED,
    INVOKED_OUTCOME_UNKNOWN,
    ReceiptObservation,
    SlurmJob,
    SlurmTaskState,
    SubmissionRequest,
    SubmitOutcome,
)
from o2mcp.runorg.execution_remote_fs import remote_fs_command
from o2mcp.runorg.lifecycle_coordination import coordination_command, matching_claims_command, new_claim_id
from o2mcp.runorg.plan_components import ReceiptSpec
from o2mcp.runorg.strict_json import strict_json_object
from o2mcp.slurm import O2Slurm


class ExecutionBackend(Protocol):
    """Operations required by the project-neutral execution coordinator."""

    def find_jobs(self, comment: str) -> Sequence[SlurmJob]:
        """Return every root Slurm job whose comment exactly matches ``comment``."""

    def prepare_submission(self, request: SubmissionRequest) -> SubmitOutcome | None:
        """Prepare immutable inputs; return ``None`` only when invocation is safe."""

    def invoke_submission(self, request: SubmissionRequest) -> SubmitOutcome:
        """Cross the sbatch boundary once and classify the evidence explicitly."""

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

    def write_immutable_text_fenced(self, run_root: str, path: str, text: str) -> bool:
        """Publish immutable bytes while excluding a destructive transition."""

    def compare_and_swap_text_fenced(
        self,
        run_root: str,
        path: str,
        expected: str | None,
        replacement: str | None,
    ) -> bool:
        """Apply one exact CAS while excluding a destructive transition."""

    def acquire_lifecycle_claim(self, run_root: str, operation_id: str) -> str | None:
        """Return an exact holder claim, or ``None`` when transition has won."""

    def release_lifecycle_claim(self, run_root: str, claim_id: str) -> None:
        """Release one exact holder claim after durable convergence."""

    def matching_lifecycle_claims(self, run_root: str, operation_id: str) -> Sequence[str]:
        """Return every durable holder for one exact mutation operation."""


# A receipt is evidence only when every component below the authenticated run
# root is real: ``lstat`` declines to follow just the final name, so a task that
# replaces an intermediate directory with a link to an external tree would still
# have its target hashed.  Promotion and archive preserve the link rather than
# those bytes, releasing a run as certified against data that can change or
# vanish.  The walk therefore refuses a link at any depth below the root, and
# ``O_NOFOLLOW`` closes the window between the check and the read.  Components
# of the run root itself are not walked: that path is the authenticated boundary
# and legitimately traverses site symlinks on the cluster.
RECEIPT_PROBE_PROGRAM = """
import hashlib, json, os, stat, sys

root, relative = sys.argv[1], sys.argv[2]
parts = [part for part in relative.split('/') if part not in ('', '.')]
valid = bool(parts) and '..' not in parts
current = root
if valid:
    for depth, part in enumerate(parts):
        current = os.path.join(current, part)
        try:
            mode = os.lstat(current).st_mode
        except OSError:
            valid = False
            break
        expected = stat.S_ISDIR if depth < len(parts) - 1 else stat.S_ISREG
        if not expected(mode):
            valid = False
            break

digest = None
if valid:
    try:
        handle = os.open(current, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    except OSError:
        valid = False
    else:
        with os.fdopen(handle, 'rb') as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()

print(json.dumps({'exists': valid, 'sha256': digest}, sort_keys=True))
"""


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
        """Search live and lifetime accounting records for an exact Slurm comment."""

        # squeue catches newly accepted jobs before accounting ingestion; sacct
        # catches jobs that already left the queue. Invocation evidence is
        # durable for the lifetime of a run, so recovery must use the same
        # lifetime rather than an arbitrary recent-time window. Restrict the
        # historical query to the current user while searching from the Unix
        # epoch; otherwise an old lost response becomes permanently
        # unrecoverable after the former seven-day cutoff.
        command = "\n".join(
            [
                "set +e",
                "squeue -u \"$USER\" -h -o '%i|%k|%T' 2>&1",
                "squeue_status=$?",
                'sacct -X -S 1970-01-01 -u "$USER" -n -P --format=JobIDRaw,Comment,State 2>&1',
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

    def prepare_submission(self, request: SubmissionRequest) -> SubmitOutcome | None:
        """Stage and verify all scheduler inputs without invoking ``sbatch``.

        Preparation is intentionally recoverable on the same attempt because a
        failure here proves that Slurm was never called.  The engine publishes
        its invocation marker only after this method succeeds.
        """

        try:
            self.write_immutable_text(request.script_path, request.script_text)
        except Exception as exc:
            return SubmitOutcome(DEFINITELY_NOT_INVOKED, stderr=str(exc))
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
            return SubmitOutcome(DEFINITELY_NOT_INVOKED, stderr=mkdir.stderr or "could not create log directory")
        return None

    def invoke_submission(self, request: SubmissionRequest) -> SubmitOutcome:
        """Verify a held dispatcher descriptor and pass that descriptor to sbatch.

        ``sbatch /proc/self/fd/N`` removes the usual hash-to-exec race: replacing
        the pathname after verification cannot change the inode Slurm reads.
        A zero return code without the canonical job-ID line is not rejection;
        it is an uncertain accepted outcome that must be recovered by comment.
        """

        program = r"""
import hashlib, json, os, stat, subprocess, sys
path, expected, args_json = sys.argv[1:4]
try:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode): raise ValueError('dispatcher is not a regular file')
        payload = b''
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            payload += chunk
        if hashlib.sha256(payload).hexdigest() != expected: raise ValueError('dispatcher digest mismatch')
        os.lseek(fd, 0, os.SEEK_SET)
        result = subprocess.run(
            ['sbatch', *json.loads(args_json), f'/proc/self/fd/{fd}'],
            pass_fds=(fd,), text=True, capture_output=True,
        )
    finally:
        os.close(fd)
except Exception as exc:
    response = {'invoked': False, 'returncode': None, 'stdout': '', 'stderr': str(exc)}
else:
    response = {
        'invoked': True, 'returncode': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr,
    }
print(json.dumps(response, sort_keys=True))
"""
        command = " ".join(
            (
                "python3 -c",
                shlex.quote(program),
                shlex.quote(request.script_path),
                shlex.quote(text_sha256(request.script_text)),
                shlex.quote(json.dumps(list(request.sbatch_args()))),
            )
        )
        try:
            result = self.connection.run(command, timeout=60)
        except Exception as exc:
            return SubmitOutcome(INVOKED_OUTCOME_UNKNOWN, stderr=str(exc))
        if not result.ok and not result.stdout.strip():
            # The transport may have failed after the remote sbatch process was
            # started; transport return codes cannot prove scheduler rejection.
            return SubmitOutcome(INVOKED_OUTCOME_UNKNOWN, stdout=result.stdout, stderr=result.stderr)
        try:
            value = strict_json_object(result.stdout, "sbatch wrapper response")
            if set(value) != {"invoked", "returncode", "stderr", "stdout"}:
                raise ValueError("unexpected fields")
            invoked = value["invoked"]
            returncode = value["returncode"]
            stdout = value["stdout"]
            stderr = value["stderr"]
            if type(invoked) is not bool or type(stdout) is not str or type(stderr) is not str:
                raise ValueError("invalid response types")
        except ValueError:
            return SubmitOutcome(INVOKED_OUTCOME_UNKNOWN, stdout=result.stdout, stderr=result.stderr)
        if not invoked:
            if returncode is not None:
                return SubmitOutcome(INVOKED_OUTCOME_UNKNOWN, stdout=stdout, stderr=stderr)
            return SubmitOutcome(DEFINITELY_NOT_INVOKED, stdout=stdout, stderr=stderr)
        if type(returncode) is not int:
            return SubmitOutcome(INVOKED_OUTCOME_UNKNOWN, stdout=stdout, stderr=stderr)
        if returncode != 0:
            return SubmitOutcome(DEFINITELY_REJECTED, returncode=returncode, stdout=stdout, stderr=stderr)
        match = re.search(r"Submitted batch job (\d+)", stdout) or re.search(r"Submitted batch job (\d+)", stderr)
        if match is None:
            return SubmitOutcome(INVOKED_OUTCOME_UNKNOWN, returncode=0, stdout=stdout, stderr=stderr)
        return SubmitOutcome(ACCEPTED, job_id=match.group(1), returncode=0, stdout=stdout, stderr=stderr)

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

        result = self.connection.run(
            " ".join(
                (
                    "python3 -c",
                    shlex.quote(RECEIPT_PROBE_PROGRAM),
                    shlex.quote(run_root),
                    shlex.quote(receipt.path),
                )
            ),
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
            observed = strict_json_object(result.stdout, "receipt observation")
            if not isinstance(observed, dict) or set(observed) != {"exists", "sha256"}:
                raise ValueError("unexpected receipt observation fields")
            exists = observed["exists"]
            sha256 = observed["sha256"]
            if not isinstance(exists, bool) or (sha256 is not None and not isinstance(sha256, str)):
                raise ValueError("invalid receipt observation types")
            return ReceiptObservation(receipt.path, exists, sha256)
        except ValueError as exc:
            return ReceiptObservation(
                receipt.path,
                False,
                None,
                trustworthy=False,
                error=f"invalid receipt observation response: {exc}",
            )

    def read_text(self, path: str) -> str | None:
        """Read one remote UTF-8 file while distinguishing absence from emptiness."""

        result = self.connection.run(remote_fs_command("read", path), timeout=60, input_text="{}")
        if not result.ok:
            raise RuntimeError(f"safe control-file read failed: {path}: {result.stderr.strip()}")
        value = strict_json_object(result.stdout, "safe read response")
        if value.get("state") == "MISSING" and value.get("payload") is None:
            return None
        if set(value) != {"payload", "state"} or value["state"] != "PRESENT" or type(value["payload"]) is not str:
            raise RuntimeError(f"safe control-file read returned invalid data for {path}")
        try:
            return base64.b64decode(value["payload"], validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"safe control-file read returned invalid UTF-8 for {path}") from exc

    def write_immutable_text(self, path: str, text: str) -> bool:
        """Atomically publish a complete remote evidence file.

        Writing directly into an ``O_EXCL`` destination leaves a truncated final
        file if the writer dies between ``open`` and ``fsync``.  Instead, write and
        fsync a private temporary file, then use ``link`` as the atomic no-replace
        publication primitive.  A competing caller can observe only complete
        bytes and receives ``False`` rather than submission ownership.
        """

        return self._write_immutable_text(path, text, run_root=None)

    def write_immutable_text_fenced(self, run_root: str, path: str, text: str) -> bool:
        """Publish immutable evidence under the transition coordination lock."""

        return self._write_immutable_text(path, text, run_root=run_root)

    def _write_immutable_text(self, path: str, text: str, *, run_root: str | None) -> bool:
        """Implement ordinary and lifecycle-fenced immutable publication."""

        payload = json.dumps({"payload": base64.b64encode(text.encode()).decode("ascii")})
        result = self.connection.run(
            remote_fs_command("immutable", path, run_root=run_root),
            timeout=120,
            input_text=payload,
        )
        if not result.ok:
            reason = (
                "immutable receipt already exists with different bytes" if result.returncode == 42 else "write failed"
            )
            raise RuntimeError(f"{reason}: {path}: {result.stderr.strip()}")
        outcome = strict_json_object(result.stdout, "immutable write response").get("state")
        if outcome not in {"CREATED", "EXISTING"}:
            raise RuntimeError(f"immutable write returned an invalid publication result for {path}: {outcome!r}")
        return outcome == "CREATED"

    def write_mutable_text(self, path: str, text: str) -> None:
        """Atomically replace a remote current-state file after fsync."""

        payload = json.dumps({"payload": base64.b64encode(text.encode()).decode("ascii")})
        result = self.connection.run(
            remote_fs_command("mutable", path),
            timeout=120,
            input_text=payload,
        )
        if not result.ok:
            raise RuntimeError(f"atomic state write failed: {path}: {result.stderr.strip()}")

    def compare_and_swap_text(self, path: str, expected: str | None, replacement: str | None) -> bool:
        """Perform a lock-protected exact-byte CAS for mutable outbox state.

        Absence and an empty file are distinct.  The operation fsyncs both the
        replacement and its parent before releasing the sibling lock, allowing
        callers to merge and compare-clear one exact outbox payload safely.
        """

        return self._compare_and_swap_text(path, expected, replacement, run_root=None)

    def compare_and_swap_text_fenced(
        self,
        run_root: str,
        path: str,
        expected: str | None,
        replacement: str | None,
    ) -> bool:
        """Apply one CAS under the transition coordination lock."""

        return self._compare_and_swap_text(path, expected, replacement, run_root=run_root)

    def _compare_and_swap_text(
        self,
        path: str,
        expected: str | None,
        replacement: str | None,
        *,
        run_root: str | None,
    ) -> bool:
        """Implement ordinary and lifecycle-fenced exact-byte CAS."""

        def encode(value: str | None) -> str | None:
            """Encode optional exact bytes without conflating absence and empty text."""

            return None if value is None else base64.b64encode(value.encode()).decode("ascii")

        payload = json.dumps({"expected": encode(expected), "replacement": encode(replacement)})
        result = self.connection.run(
            remote_fs_command("cas", path, run_root=run_root),
            timeout=120,
            input_text=payload,
        )
        if not result.ok:
            raise RuntimeError(f"mutable CAS failed: {path}: {result.stderr.strip()}")
        outcome = strict_json_object(result.stdout, "mutable CAS response").get("state")
        if outcome not in {"CONFLICT", "SWAPPED"}:
            raise RuntimeError(f"mutable CAS returned invalid result for {path}: {outcome!r}")
        return outcome == "SWAPPED"

    def acquire_lifecycle_claim(self, run_root: str, operation_id: str) -> str | None:
        """Atomically create distinct holder ownership unless transition won."""

        claim_id = new_claim_id(operation_id)
        result = self.connection.run(coordination_command("acquire", run_root, claim_id), timeout=60)
        if not result.ok:
            raise RuntimeError(f"lifecycle claim failed: {result.stderr.strip()}")
        outcome = result.stdout.strip()
        if outcome not in {"ACQUIRED", "TRANSITION"}:
            raise RuntimeError(f"invalid lifecycle claim response: {outcome!r}")
        return claim_id if outcome == "ACQUIRED" else None

    def release_lifecycle_claim(self, run_root: str, claim_id: str) -> None:
        """Remove only the exact holder claim named by its ownership token."""

        result = self.connection.run(coordination_command("release", run_root, claim_id), timeout=60)
        if not result.ok or result.stdout.strip() != "RELEASED":
            raise RuntimeError(f"lifecycle claim release failed: {result.stderr.strip()}")

    def matching_lifecycle_claims(self, run_root: str, operation_id: str) -> Sequence[str]:
        """List same-operation holders so terminal replay can retire orphans."""

        result = self.connection.run(matching_claims_command(run_root, operation_id), timeout=60)
        if not result.ok:
            raise RuntimeError(f"lifecycle claim query failed: {result.stderr.strip()}")
        try:
            values = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("lifecycle claim query returned invalid JSON") from exc
        if type(values) is not list or any(type(item) is not str for item in values):
            raise RuntimeError("lifecycle claim query returned invalid holder identities")
        return tuple(values)


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
