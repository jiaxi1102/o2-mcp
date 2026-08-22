"""Shared submission/transition coordination outside a deletable run root.

Claims live beside the run directory, so destructive transitions cannot erase
the very mutex that excludes new submissions.  Submission claims deliberately
survive uncertain scheduler outcomes and registry-sync crashes; replay removes
them only after the accepted job is represented durably in registry state.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
import secrets
import shlex


def coordination_root(run_root: str) -> str:
    """Return the sibling coordination directory for one canonical run root."""

    parent, run_id = posixpath.split(run_root.rstrip("/"))
    return posixpath.join(parent, f".{run_id}.execution-coordination")


def coordination_lock(run_root: str) -> str:
    """Return the sibling lock shared by claims and transition marking."""

    return coordination_root(run_root) + ".lock"


_CLAIM_ID_RE = re.compile(r"^[0-9a-f]{64}-[0-9a-f]{64}$")


def new_claim_id(operation: str) -> str:
    """Return an operation-bound, unguessable identity for one claim holder.

    The operation digest keeps diagnostics groupable without exposing arbitrary
    operation text in a filename.  The random suffix is the ownership token:
    concurrent callers of the same operation must never share a releasable file.
    """

    digest = hashlib.sha256(operation.encode()).hexdigest()
    return f"{digest}-{secrets.token_hex(32)}"


def claim_name(claim_id: str) -> str:
    """Map a validated holder identity to its coordination filename."""

    if _CLAIM_ID_RE.fullmatch(claim_id) is None:
        raise ValueError("lifecycle claim ID is invalid")
    return f"claim-{claim_id}.json"


COORDINATION_PROGRAM = r"""
import fcntl, json, os, stat, sys
op, root, lock_path, name, claim_id = sys.argv[1:6]
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
    def fsync_root():
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    def strict_claim(handle, descriptor):
        if os.fstat(descriptor).st_size > 512: raise SystemExit('claim is oversized')
        def pairs(items):
            value = {}
            for key, item in items:
                if key in value: raise ValueError('duplicate claim key')
                value[key] = item
            return value
        return json.load(
            handle,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    if op == 'acquire':
        if os.path.lexists(marker):
            print('TRANSITION')
        else:
            fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump({'claim_id': claim_id}, handle, sort_keys=True)
                handle.flush(); os.fsync(handle.fileno())
            # The file fsync preserves bytes; the directory fsync preserves the
            # exclusion name itself across a metadata/server crash.
            fsync_root()
            print('ACQUIRED')
    elif op == 'release':
        try:
            fd = os.open(claim, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd); raise SystemExit('claim is not regular')
            with os.fdopen(fd, encoding='utf-8') as handle: value = strict_claim(handle, fd)
            if value != {'claim_id': claim_id}: raise SystemExit('claim ownership mismatch')
            os.unlink(claim)
            # Persist retirement so a reboot cannot revive a claim that was
            # already reported as released to the coordinator.
            fsync_root()
        print('RELEASED')
    else:
        raise SystemExit('invalid coordination operation')
finally:
    os.close(lock)
"""


def coordination_command(operation: str, run_root: str, claim_id: str) -> str:
    """Render the fixed claim program for one exact holder identity."""

    root = coordination_root(run_root)
    name = claim_name(claim_id)
    return " ".join(
        (
            "python3 -c",
            shlex.quote(COORDINATION_PROGRAM),
            shlex.quote(operation),
            shlex.quote(root),
            shlex.quote(coordination_lock(run_root)),
            shlex.quote(name),
            shlex.quote(claim_id),
        )
    )


_MATCHING_CLAIMS_PROGRAM = r"""
import fcntl, glob, json, os, stat, sys
root, lock_path, prefix = sys.argv[1:4]
lock = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
try:
    if not stat.S_ISREG(os.fstat(lock).st_mode): raise SystemExit('coordination lock is not regular')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not os.path.lexists(root):
        print('[]'); raise SystemExit(0)
    if os.path.islink(root) or not stat.S_ISDIR(os.stat(root, follow_symlinks=False).st_mode):
        raise SystemExit('coordination root is not a real directory')
    claims = []
    for path in glob.glob(os.path.join(root, f'claim-{prefix}*.json')):
        name = os.path.basename(path)
        claim_id = name[len('claim-'):-len('.json')]
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd); raise SystemExit('claim is not regular')
        with os.fdopen(fd, encoding='utf-8') as handle: value = json.load(handle)
        if value != {'claim_id': claim_id}: raise SystemExit('claim ownership mismatch')
        claims.append(claim_id)
    print(json.dumps(sorted(claims)))
finally:
    os.close(lock)
"""


def matching_claims_command(run_root: str, operation_id: str) -> str:
    """Render a locked query for every holder of one exact operation."""

    prefix = hashlib.sha256(operation_id.encode()).hexdigest() + "-"
    return " ".join(
        (
            "python3 -c",
            shlex.quote(_MATCHING_CLAIMS_PROGRAM),
            shlex.quote(coordination_root(run_root)),
            shlex.quote(coordination_lock(run_root)),
            shlex.quote(prefix),
        )
    )


__all__ = [
    "claim_name",
    "coordination_command",
    "coordination_lock",
    "coordination_root",
    "matching_claims_command",
    "new_claim_id",
]
