"""Atomic transition marking and scheduler-evidence discovery.

The transition marker and execution claims use one sibling lock.  Establishing
the marker first prevents any later engine mutation or submission from being
accepted while the caller performs its terminal scheduler proof and launches a
detached transfer.  Job IDs are recovered from authenticated-looking control
records as well as run.json, covering a crash after sbatch but before registry
annotation; an unmatched invocation marker fails closed.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from o2mcp.connection import O2Connection
from o2mcp.runorg.lifecycle_coordination import coordination_lock, coordination_root
from o2mcp.runorg.strict_json import strict_json_object


@dataclass(frozen=True)
class TransitionBoundary:
    """Result of atomically marking a transition and scanning job evidence."""

    job_ids: tuple[str, ...]


_BEGIN_PROGRAM = r"""
import fcntl, glob, hashlib, json, os, stat, sys, tempfile
run_root, coordination, lock_path, token = sys.argv[1:5]
os.makedirs(os.path.dirname(coordination), exist_ok=True)
lock = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
created = False
try:
    if not stat.S_ISREG(os.fstat(lock).st_mode): raise SystemExit('coordination lock is not regular')
    fcntl.flock(lock, fcntl.LOCK_EX)
    os.makedirs(coordination, mode=0o700, exist_ok=True)
    if os.path.islink(coordination): raise SystemExit('coordination root is a symlink')
    claims = glob.glob(os.path.join(coordination, 'claim-*.json'))
    if claims: raise SystemExit('execution mutation claims are still active')
    marker = os.path.join(coordination, 'transition.json')
    try:
        marker_fd = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        fd, temporary = tempfile.mkstemp(prefix='.transition-', dir=coordination)
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(token); handle.flush(); os.fsync(handle.fileno())
        try: os.link(temporary, marker)
        finally: os.unlink(temporary)
        created = True
        directory_fd = os.open(coordination, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    else:
        if not stat.S_ISREG(os.fstat(marker_fd).st_mode): raise SystemExit('transition marker is not regular')
        with os.fdopen(marker_fd, encoding='utf-8') as handle: current = handle.read()
        if current != token: raise SystemExit('a different transition is already marked')
        # The deterministic token identifies the reviewed transition, not this
        # caller.  A second identical caller therefore observes, but never owns,
        # the existing marker and must not be allowed to roll it back.
        print(json.dumps({'already_marked': True, 'job_ids': []}, sort_keys=True))
        raise SystemExit(0)

    execution = os.path.join(run_root, 'receipts', 'execution')
    records = {}
    rejections = set()
    invocations = set()
    jobs = set()
    def strict(path):
        def pairs(items):
            out = {}
            for key, value in items:
                if key in out: raise ValueError('duplicate key')
                out[key] = value
            return out
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd); raise ValueError('control evidence is not regular')
        with os.fdopen(fd, encoding='utf-8') as handle:
            return json.load(
                handle,
                object_pairs_hook=pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
    for path in glob.glob(os.path.join(execution, 'submissions', '*', 'attempt-*.json')):
        value = strict(path)
        job = value.get('job_id')
        comment = value.get('comment')
        if type(job) is not str or not job.isdigit() or type(comment) is not str:
            raise SystemExit('invalid submission evidence')
        records[comment] = job; jobs.add(job)
    for path in glob.glob(os.path.join(execution, 'submission-rejections', '*', 'attempt-*.json')):
        value = strict(path); comment = value.get('comment')
        if (
            type(comment) is not str
            or type(value.get('returncode')) is not int
            or value['returncode'] == 0
        ):
            raise SystemExit('invalid rejection evidence')
        rejections.add(comment)
    for path in glob.glob(os.path.join(execution, 'submission-invocations', '*', 'attempt-*.json')):
        value = strict(path); comment = value.get('comment'); claim_id = value.get('lifecycle_claim_id')
        expected = {
            'attempt', 'comment', 'intent_sha256', 'lifecycle_claim_id',
            'plan_sha256', 'schema_version', 'stage_id',
        }
        operation = f"submit:{value.get('plan_sha256')}:{value.get('stage_id')}:{value.get('attempt')}"
        claim_prefix = hashlib.sha256(operation.encode()).hexdigest() + '-'
        expected_comment = (
            f"o2plan:v1:{value.get('plan_sha256')}:{value.get('stage_id')}:"
            f"a{value.get('attempt', 0):03d}"
        )
        if (
            set(value) != expected
            or type(comment) is not str
            or comment != expected_comment
            or type(value.get('schema_version')) is not int or value['schema_version'] != 1
            or type(value.get('attempt')) is not int or value['attempt'] < 1
            or type(value.get('plan_sha256')) is not str or len(value['plan_sha256']) != 64
            or type(value.get('stage_id')) is not str
            or type(value.get('intent_sha256')) is not str or len(value['intent_sha256']) != 64
            or any(char not in '0123456789abcdef' for char in value['plan_sha256'] + value['intent_sha256'])
            or type(claim_id) is not str or len(claim_id) != 129 or not claim_id.startswith(claim_prefix)
            or any(char not in '0123456789abcdef-' for char in claim_id)
        ): raise SystemExit('invalid invocation evidence')
        invocations.add(comment)
    unresolved = invocations - set(records) - rejections
    if unresolved: raise SystemExit('unresolved sbatch invocation blocks transition')
    for path in glob.glob(os.path.join(execution, 'pending-registry', '*', 'attempt-*.json')):
        value = strict(path); values = value.get('job_ids')
        if type(values) is not list or any(type(job) is not str or not job.isdigit() for job in values):
            raise SystemExit('invalid registry outbox evidence')
        jobs.update(values)
    print(json.dumps({'already_marked': False, 'job_ids': sorted(jobs, key=int)}, sort_keys=True))
except BaseException:
    # Only this process knows whether it linked the marker.  Clean up a marker
    # created by a failed evidence scan here, while the coordination lock is
    # still held; callers must never guess ownership from a shared token.
    if created:
        try: os.unlink(marker)
        except FileNotFoundError: pass
        directory_fd = os.open(coordination, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    raise
finally:
    os.close(lock)
"""


_ROLLBACK_PROGRAM = r"""
import fcntl, os, stat, sys
coordination, lock_path, token = sys.argv[1:4]
lock = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
try:
    if not stat.S_ISREG(os.fstat(lock).st_mode): raise SystemExit('coordination lock is not regular')
    fcntl.flock(lock, fcntl.LOCK_EX)
    marker = os.path.join(coordination, 'transition.json')
    try:
        marker_fd = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError: current = None
    else:
        if not stat.S_ISREG(os.fstat(marker_fd).st_mode): raise SystemExit('transition marker is not regular')
        with os.fdopen(marker_fd, encoding='utf-8') as handle: current = handle.read()
    if current == token: os.unlink(marker)
    elif current is not None: raise SystemExit('transition marker changed; refusing rollback')
finally:
    os.close(lock)
"""


def begin_transition(connection: O2Connection, run_root: str, token: str) -> TransitionBoundary:
    """Mark transition atomically and return all discovered scheduler job IDs."""

    command = " ".join(
        (
            "python3 -c",
            shlex.quote(_BEGIN_PROGRAM),
            shlex.quote(run_root),
            shlex.quote(coordination_root(run_root)),
            shlex.quote(coordination_lock(run_root)),
            shlex.quote(token),
        )
    )
    result = connection.run(command, timeout=120)
    if not result.ok:
        # The remote helper tracks marker ownership and rolls back only a marker
        # it created.  A local rollback here could erase another identical
        # caller's transition after a lost or ambiguous response.
        raise ValueError(result.stderr.strip() or result.stdout.strip() or "could not mark lifecycle transition")
    value = strict_json_object(result.stdout, "transition boundary response")
    jobs = value.get("job_ids")
    already_marked = value.get("already_marked")
    if (
        set(value) != {"already_marked", "job_ids"}
        or type(already_marked) is not bool
        or type(jobs) is not list
        or any(type(job) is not str for job in jobs)
    ):
        raise ValueError("transition boundary returned invalid job evidence")
    if already_marked:
        raise ValueError("the reviewed lifecycle transition is already marked by another caller")
    return TransitionBoundary(tuple(jobs))


def rollback_transition(connection: O2Connection, run_root: str, token: str) -> None:
    """Clear only the exact marker created by a failed pre-launch transition."""

    command = " ".join(
        (
            "python3 -c",
            shlex.quote(_ROLLBACK_PROGRAM),
            shlex.quote(coordination_root(run_root)),
            shlex.quote(coordination_lock(run_root)),
            shlex.quote(token),
        )
    )
    result = connection.run(command, timeout=60)
    if not result.ok:
        raise RuntimeError(f"could not roll back transition marker: {result.stderr.strip()}")


__all__ = ["TransitionBoundary", "begin_transition", "rollback_transition"]
