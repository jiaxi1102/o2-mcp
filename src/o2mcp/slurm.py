"""Slurm job operations over the O2 connection: submit, status, logs, cancel.

These wrap the exact commands the project already uses by hand (``sbatch``,
``squeue -u``, ``sacct -j``, ``tail``, ``scancel``) and parse their output into
structured records, so an agent can submit work and monitor it without
hand-parsing terminal text.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Any

from o2mcp.connection import CommandResult, O2Connection

_SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")


def _quote_remote_path(path: str) -> str:
    """Shell-quote a remote path while preserving a leading ``~/`` for expansion.

    ``shlex.quote('~/x')`` yields ``'~/x'`` which the remote shell treats as a
    literal tilde. Quoting only the remainder keeps ``~`` expandable while still
    protecting paths that contain spaces or shell metacharacters.
    """
    if path == "~":
        return "~"
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


# Stable, parser-friendly squeue columns (pipe-delimited, no header).
_SQUEUE_FORMAT = "%i|%j|%T|%M|%l|%D|%R"
_SQUEUE_FIELDS = ["job_id", "name", "state", "elapsed", "time_limit", "nodes", "reason"]

_SACCT_FORMAT = "JobID,JobName,State,Elapsed,ExitCode,MaxRSS,ReqMem,Start,End,NodeList"
_SACCT_FIELDS = [
    "job_id",
    "name",
    "state",
    "elapsed",
    "exit_code",
    "max_rss",
    "req_mem",
    "start",
    "end",
    "node_list",
]


@dataclass
class SubmitResult:
    """Outcome of an sbatch submission."""

    job_id: str | None
    submitted: bool
    command: CommandResult


class O2Slurm:
    """Slurm operations built on an :class:`O2Connection`."""

    def __init__(self, connection: O2Connection) -> None:
        self.conn = connection

    def submit(
        self,
        remote_script_path: str,
        *,
        sbatch_args: list[str] | None = None,
        timeout: float = 60.0,
    ) -> SubmitResult:
        """Submit an sbatch script that already exists on O2.

        Returns the parsed Slurm job id (or ``submitted=False`` with the raw
        command result when sbatch did not report one).
        """
        args = " ".join(shlex.quote(a) for a in (sbatch_args or []))
        command = f"sbatch {args} {_quote_remote_path(remote_script_path)}".replace("  ", " ").strip()
        result = self.conn.run(command, timeout=timeout)
        match = _SUBMITTED_RE.search(result.stdout) or _SUBMITTED_RE.search(result.stderr)
        job_id = match.group(1) if match else None
        return SubmitResult(job_id=job_id, submitted=job_id is not None, command=result)

    def submit_text(
        self,
        script_text: str,
        remote_path: str,
        *,
        sbatch_args: list[str] | None = None,
        timeout: float = 60.0,
    ) -> SubmitResult:
        """Stage an sbatch script's TEXT to ``remote_path`` on O2, then submit it.

        The script is written via the existing ControlMaster (no extra login) and
        made executable before submission.
        """
        quoted = _quote_remote_path(remote_path)
        stage = self.conn.run(
            f'mkdir -p "$(dirname {quoted})" && cat > {quoted} && chmod +x {quoted}',
            timeout=timeout,
            input_text=script_text,
        )
        if not stage.ok:
            return SubmitResult(job_id=None, submitted=False, command=stage)
        return self.submit(remote_path, sbatch_args=sbatch_args, timeout=timeout)

    def queue(self, user: str | None = None, *, timeout: float = 30.0) -> list[dict[str, str]]:
        """Return the current Slurm queue for ``user`` as structured rows."""
        user_token = shlex.quote(user) if user else '"$USER"'
        result = self.conn.run(f"squeue -u {user_token} -h -o {shlex.quote(_SQUEUE_FORMAT)}", timeout=timeout)
        return _parse_delimited(result.stdout, _SQUEUE_FIELDS)

    def job_status(self, job_id: str, *, timeout: float = 30.0) -> list[dict[str, str]]:
        """Return sacct accounting rows for one job (the job plus its job steps)."""
        result = self.conn.run(
            f"sacct -j {shlex.quote(str(job_id))} --noheader --parsable2 --format={_SACCT_FORMAT}",
            timeout=timeout,
        )
        return _parse_delimited(result.stdout, _SACCT_FIELDS, delimiter="|")

    def tail_log(self, remote_path: str, *, lines: int = 100, timeout: float = 30.0) -> CommandResult:
        """Tail the last ``lines`` of a remote log file."""
        return self.conn.run(f"tail -n {int(lines)} {_quote_remote_path(remote_path)}", timeout=timeout)

    def cancel(self, job_id: str, *, timeout: float = 30.0) -> CommandResult:
        """Cancel a Slurm job with scancel."""
        return self.conn.run(f"scancel {shlex.quote(str(job_id))}", timeout=timeout)


def _parse_delimited(text: str, fields: list[str], delimiter: str = "|") -> list[dict[str, str]]:
    """Parse pipe-delimited, headerless command output into a list of dicts."""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(delimiter)
        # Tolerate trailing/missing columns rather than dropping the row.
        parts = (parts + [""] * len(fields))[: len(fields)]
        rows.append(dict(zip(fields, parts)))
    return rows
