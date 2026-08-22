"""Anchored no-follow remote file operations for execution control records.

The program in this module runs on O2 through the already-authenticated
connection.  Every path component is opened relative to a trusted descriptor,
never followed through a symlink, and every leaf is opened nonblocking before
its regular-file type is checked.  This is deliberately stricter than ordinary
``cat``/``open`` because these files authorize jobs and destructive lifecycle
transitions.
"""

from __future__ import annotations

import shlex

from o2mcp.runorg.lifecycle_coordination import coordination_lock, coordination_root

# Kept as a real multiline program rather than shell fragments so all four
# operations share exactly the same path-walk and post-publication checks.
REMOTE_FS_PROGRAM = r"""
import base64, errno, fcntl, json, os, stat, sys, tempfile

op, path = sys.argv[1:3]
if len(sys.argv) not in (3, 6):
    raise SystemExit('invalid remote filesystem argument count')
run_root = sys.argv[3] if len(sys.argv) == 6 else None
coordination = sys.argv[4] if len(sys.argv) == 6 else None
coordination_lock_path = sys.argv[5] if len(sys.argv) == 6 else None
request = json.loads(sys.stdin.read() or '{}')

def fail(message, code=41):
    print(json.dumps({'error': message}, sort_keys=True))
    raise SystemExit(code)

def open_parent(target, create=False):
    if not target.startswith('/') or target == '/':
        fail('path must be an absolute leaf path')
    pieces = [part for part in target.split('/') if part]
    leaf = pieces.pop()
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in pieces:
            if part in ('.', '..'):
                fail('relative path component rejected')
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o700, dir_fd=fd)
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd, leaf
    except BaseException:
        os.close(fd)
        raise

def read_leaf(parent, leaf):
    try:
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
    except FileNotFoundError:
        return None, None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            fail('control path is not a regular file')
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b''.join(chunks), (info.st_dev, info.st_ino)
    finally:
        os.close(fd)

def decode(name):
    value = request.get(name)
    if value is None:
        return None
    if type(value) is not str:
        fail(name + ' must be base64 text')
    try:
        return base64.b64decode(value, validate=True)
    except Exception:
        fail(name + ' is invalid base64')

def verify_lexical(expected):
    check, check_leaf = open_parent(path)
    try:
        now = os.fstat(check)
        if check_leaf != leaf or (now.st_dev, now.st_ino) != (anchored_parent.st_dev, anchored_parent.st_ino):
            fail('control path ancestor changed during operation')
        final, _ = read_leaf(check, leaf)
        if final != expected:
            fail('control path final bytes changed during verification')
    finally:
        os.close(check)

def publish(parent, leaf, payload, *, replace):
    temporary = '.o2mcp-%d-%s' % (os.getpid(), os.urandom(8).hex())
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
        temp_info = os.fstat(fd)
        if replace:
            os.rename(temporary, leaf, src_dir_fd=parent, dst_dir_fd=parent)
        else:
            os.link(temporary, leaf, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        os.fsync(parent)
        final, inode = read_leaf(parent, leaf)
        if final != payload or inode != (temp_info.st_dev, temp_info.st_ino):
            fail('published leaf changed during verification')
    finally:
        os.close(fd)
        removed = False
        try:
            os.unlink(temporary, dir_fd=parent)
            removed = True
        except FileNotFoundError:
            pass
        if removed:
            # The publication barrier also persisted the temporary hard-link
            # name. Persist its cleanup before reporting a complete operation.
            os.fsync(parent)

lifecycle_lock_fd = None
parent = None
try:
    if run_root is not None:
        # Hold the same sibling lock used to publish transition.json for the
        # entire control-record mutation. This avoids persistent poll claims
        # while making check-and-write indivisible with respect to promotion or
        # archive, even if the run root is renamed immediately afterward.
        if not path.startswith(run_root.rstrip('/') + '/'):
            fail('fenced control path is outside its run root')
        os.makedirs(os.path.dirname(coordination), exist_ok=True)
        lifecycle_lock_fd = os.open(
            coordination_lock_path,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            0o600,
        )
        if not stat.S_ISREG(os.fstat(lifecycle_lock_fd).st_mode):
            fail('coordination lock is not regular')
        fcntl.flock(lifecycle_lock_fd, fcntl.LOCK_EX)
        if os.path.lexists(coordination) and (
            os.path.islink(coordination)
            or not stat.S_ISDIR(os.stat(coordination, follow_symlinks=False).st_mode)
        ):
            fail('coordination root is not a real directory')
        if os.path.lexists(os.path.join(coordination, 'transition.json')):
            fail('run lifecycle transition is in progress', 44)
    parent, leaf = open_parent(path, create=op in ('immutable', 'mutable', 'cas'))
    anchored_parent = os.fstat(parent)
    if op == 'read':
        value, _ = read_leaf(parent, leaf)
        verify_lexical(value)
        response = {
            'state': 'MISSING' if value is None else 'PRESENT',
            'payload': None if value is None else base64.b64encode(value).decode('ascii'),
        }
        print(json.dumps(response, sort_keys=True))
    elif op == 'immutable':
        payload = decode('payload')
        try:
            publish(parent, leaf, payload, replace=False)
            state = 'CREATED'
        except FileExistsError:
            existing, _ = read_leaf(parent, leaf)
            if existing != payload:
                fail('immutable file already exists with different bytes', 42)
            state = 'EXISTING'
        # Re-walk from root to detect an ancestor swap while the anchored fd was held.
        verify_lexical(payload)
        print(json.dumps({'state': state}, sort_keys=True))
    elif op == 'mutable':
        payload = decode('payload')
        publish(parent, leaf, payload, replace=True)
        verify_lexical(payload)
        print(json.dumps({'state': 'WRITTEN'}, sort_keys=True))
    elif op == 'cas':
        expected, replacement = decode('expected'), decode('replacement')
        lock_name = leaf + '.lock'
        flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        lock_fd = os.open(lock_name, flags, 0o600, dir_fd=parent)
        try:
            if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                fail('CAS lock is not a regular file')
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            current, _ = read_leaf(parent, leaf)
            if current != expected:
                print(json.dumps({'state': 'CONFLICT'}, sort_keys=True))
            else:
                if replacement is None:
                    try:
                        os.unlink(leaf, dir_fd=parent)
                    except FileNotFoundError:
                        pass
                    os.fsync(parent)
                    final, _ = read_leaf(parent, leaf)
                    if final is not None:
                        fail('CAS removal was replaced during verification')
                else:
                    publish(parent, leaf, replacement, replace=True)
                verify_lexical(replacement)
                print(json.dumps({'state': 'SWAPPED'}, sort_keys=True))
        finally:
            os.close(lock_fd)
    else:
        fail('unknown operation')
except FileNotFoundError:
    if op == 'read':
        print(json.dumps({'state': 'MISSING', 'payload': None}, sort_keys=True))
    else:
        fail('required ancestor is missing')
finally:
    if parent is not None:
        os.close(parent)
    if lifecycle_lock_fd is not None:
        os.close(lifecycle_lock_fd)
"""


def remote_fs_command(operation: str, path: str, *, run_root: str | None = None) -> str:
    """Render one remote operation, optionally fenced against transitions."""

    arguments = [operation, path]
    if run_root is not None:
        arguments.extend(
            (
                run_root,
                coordination_root(run_root),
                coordination_lock(run_root),
            )
        )
    return " ".join(("python3 -c", shlex.quote(REMOTE_FS_PROGRAM), *(shlex.quote(item) for item in arguments)))


__all__ = ["REMOTE_FS_PROGRAM", "remote_fs_command"]
