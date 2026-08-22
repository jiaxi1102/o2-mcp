"""Atomic transition marking and scheduler-evidence discovery.

The transition marker and execution claims use one sibling lock.  Establishing
the marker first prevents any later engine mutation or submission from being
accepted while the caller performs its terminal scheduler proof and launches a
detached transfer.  Job IDs are recovered from authenticated-looking control
records as well as run.json, covering a crash after sbatch but before registry
annotation; an unmatched invocation marker fails closed.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass

from o2mcp.connection import O2Connection
from o2mcp.runorg.lifecycle_coordination import coordination_lock, coordination_root
from o2mcp.runorg.strict_json import strict_json_object


@dataclass(frozen=True)
class TransitionBoundary:
    """Result of atomically marking a transition and scanning job evidence."""

    job_ids: tuple[str, ...]


@dataclass(frozen=True)
class TransitionRecovery:
    """State-inspected outcome for one retained lifecycle marker."""

    status: str
    marker_present: bool
    cleared: bool
    active_pids: tuple[str, ...]
    blockers: tuple[str, ...]


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


_RECOVER_PROGRAM = r"""
import fcntl, glob, json, os, stat, sys
source, coordination, lock_path, token, script_path, absent_json, patterns_json, apply = sys.argv[1:9]
absent_paths = json.loads(absent_json)
absent_patterns = json.loads(patterns_json)
lock = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
try:
    if not stat.S_ISREG(os.fstat(lock).st_mode): raise SystemExit('coordination lock is not regular')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if os.path.islink(coordination): raise SystemExit('coordination root is a symlink')
    marker = os.path.join(coordination, 'transition.json')
    try:
        marker_fd = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        print(json.dumps({
            'active_pids': [], 'blockers': [], 'cleared': False,
            'marker_present': False, 'status': 'marker_absent',
        }, sort_keys=True))
        raise SystemExit(0)
    if not stat.S_ISREG(os.fstat(marker_fd).st_mode):
        os.close(marker_fd); raise SystemExit('transition marker is not regular')
    with os.fdopen(marker_fd, encoding='utf-8') as handle: current = handle.read()
    if current != token: raise SystemExit('transition marker does not match the reviewed transition')

    # Search only processes owned by this account. A transition script remains
    # represented by its parent shell while rsync/tar children run, and its
    # exact staged path is a distinct argv element. The recovery process also
    # receives that path, but its executable is Python rather than a shell.
    active = []
    for cmdline in glob.glob('/proc/[0-9]*/cmdline'):
        proc_dir = os.path.dirname(cmdline)
        try:
            if os.stat(proc_dir).st_uid != os.getuid(): continue
            with open(cmdline, 'rb') as handle: raw = handle.read()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        args = [part.decode('utf-8', errors='surrogateescape') for part in raw.split(b'\0') if part]
        try: script_index = args.index(script_path)
        except ValueError: continue
        shells = {'bash', 'dash', 'ksh', 'sh', 'zsh'}
        if any(os.path.basename(arg) in shells for arg in args[:script_index]):
            active.append(os.path.basename(proc_dir))

    blockers = []
    try: source_info = os.lstat(source)
    except FileNotFoundError: blockers.append('source_missing')
    else:
        if not stat.S_ISDIR(source_info.st_mode): blockers.append('source_not_plain_directory')
    for path in absent_paths:
        if os.path.lexists(path): blockers.append('unexpected_path:' + path)
    for pattern in absent_patterns:
        for path in sorted(glob.glob(pattern)):
            blockers.append('unexpected_path:' + path)
    if active: blockers.append('transition_process_active')

    if blockers:
        status = 'active' if active else 'manual_recovery_required'
        cleared = False
    elif apply != '1':
        status = 'recoverable'
        cleared = False
    else:
        os.unlink(marker)
        directory_fd = os.open(coordination, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
        status = 'cleared'
        cleared = True
    print(json.dumps({
        'active_pids': sorted(active, key=int), 'blockers': blockers, 'cleared': cleared,
        'marker_present': True, 'status': status,
    }, sort_keys=True))
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


def recover_transition(
    connection: O2Connection,
    run_root: str,
    token: str,
    *,
    script_path: str,
    must_be_absent: tuple[str, ...],
    absent_patterns: tuple[str, ...],
    apply: bool = False,
    alias: str | None = None,
    broker_role: str | None = None,
) -> TransitionRecovery:
    """Inspect and optionally clear one cleanly abandoned transition marker.

    Recovery is intentionally narrower than rollback. It clears a retained
    marker only while holding the lifecycle lock and only when the original
    source is still an ordinary directory, no transition shell is active, and
    no quarantine, staging, or published destination exists. Any partial state
    remains fenced for manual inspection rather than being guessed away.
    """

    command = " ".join(
        (
            "python3 -c",
            shlex.quote(_RECOVER_PROGRAM),
            shlex.quote(run_root),
            shlex.quote(coordination_root(run_root)),
            shlex.quote(coordination_lock(run_root)),
            shlex.quote(token),
            shlex.quote(script_path),
            shlex.quote(json.dumps(list(must_be_absent), separators=(",", ":"))),
            shlex.quote(json.dumps(list(absent_patterns), separators=(",", ":"))),
            "1" if apply else "0",
        )
    )
    connection_kwargs = {}
    if alias is not None:
        connection_kwargs["alias"] = alias
    if broker_role is not None:
        connection_kwargs["broker_role"] = broker_role
    result = connection.run(command, timeout=120, **connection_kwargs)
    if not result.ok:
        raise ValueError(result.stderr.strip() or result.stdout.strip() or "could not inspect lifecycle transition")
    value = strict_json_object(result.stdout, "transition recovery response")
    required = {"active_pids", "blockers", "cleared", "marker_present", "status"}
    if (
        set(value) != required
        or type(value["active_pids"]) is not list
        or any(type(pid) is not str or not pid.isdigit() for pid in value["active_pids"])
        or type(value["blockers"]) is not list
        or any(type(blocker) is not str for blocker in value["blockers"])
        or type(value["cleared"]) is not bool
        or type(value["marker_present"]) is not bool
        or value["status"] not in {"active", "cleared", "manual_recovery_required", "marker_absent", "recoverable"}
    ):
        raise ValueError("transition recovery returned invalid state evidence")
    return TransitionRecovery(
        status=value["status"],
        marker_present=value["marker_present"],
        cleared=value["cleared"],
        active_pids=tuple(value["active_pids"]),
        blockers=tuple(value["blockers"]),
    )


__all__ = [
    "TransitionBoundary",
    "TransitionRecovery",
    "begin_transition",
    "recover_transition",
    "rollback_transition",
]
