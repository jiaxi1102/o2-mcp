"""Shared submission/transition coordination outside a deletable run root.

Claims live beside the run directory, so destructive transitions cannot erase
the very mutex that excludes new submissions.  Submission claims deliberately
survive uncertain scheduler outcomes and registry-sync crashes; replay removes
them only after the accepted job is represented durably in registry state.
"""

from __future__ import annotations

import hashlib
import posixpath
import shlex


def coordination_root(run_root: str) -> str:
    """Return the sibling coordination directory for one canonical run root."""

    parent, run_id = posixpath.split(run_root.rstrip("/"))
    return posixpath.join(parent, f".{run_id}.execution-coordination")


def coordination_lock(run_root: str) -> str:
    """Return the sibling lock shared by claims and transition marking."""

    return coordination_root(run_root) + ".lock"


def claim_name(operation: str) -> str:
    """Map a bounded operation identity to a filesystem-safe stable name."""

    digest = hashlib.sha256(operation.encode()).hexdigest()
    return f"claim-{digest}.json"


COORDINATION_PROGRAM = r"""
import fcntl, json, os, stat, sys
op, root, lock_path, name = sys.argv[1:5]
parent = os.path.dirname(root)
if not os.path.isabs(root) or os.path.basename(root) in ('', '.', '..'):
    raise SystemExit('invalid coordination root')
os.makedirs(parent, exist_ok=True)
lock = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
try:
    if not stat.S_ISREG(os.fstat(lock).st_mode):
        raise SystemExit('coordination lock is not regular')
    fcntl.flock(lock, fcntl.LOCK_EX)
    os.makedirs(root, mode=0o700, exist_ok=True)
    if os.path.islink(root) or not stat.S_ISDIR(os.stat(root, follow_symlinks=False).st_mode):
        raise SystemExit('coordination root is not a real directory')
    marker = os.path.join(root, 'transition.json')
    claim = os.path.join(root, name)
    if op == 'acquire':
        if os.path.lexists(marker):
            print('TRANSITION')
        else:
            try:
                fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                fd = os.open(claim, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise SystemExit('claim is not regular')
                os.close(fd)
            else:
                os.write(fd, json.dumps({'operation': name}, sort_keys=True).encode())
                os.fsync(fd); os.close(fd)
            print('ACQUIRED')
    elif op == 'release':
        try: os.unlink(claim)
        except FileNotFoundError: pass
        print('RELEASED')
    else:
        raise SystemExit('invalid coordination operation')
finally:
    os.close(lock)
"""


def coordination_command(operation: str, run_root: str, operation_id: str) -> str:
    """Render the fixed claim program for an acquire or release operation."""

    root = coordination_root(run_root)
    return " ".join(
        (
            "python3 -c",
            shlex.quote(COORDINATION_PROGRAM),
            shlex.quote(operation),
            shlex.quote(root),
            shlex.quote(coordination_lock(run_root)),
            shlex.quote(claim_name(operation_id)),
        )
    )


__all__ = ["claim_name", "coordination_command", "coordination_lock", "coordination_root"]
