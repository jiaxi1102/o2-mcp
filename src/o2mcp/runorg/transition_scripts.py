"""Shell fragments for rollback-safe promote and archive transitions.

The lifecycle planners keep these fragments separate from manifest/layout code
because the failure paths form their own small transaction protocol. A detached
script may clear its transition marker only after it has proven that private
staging and partial publication are gone and that the source is again usable.
"""

from __future__ import annotations

import posixpath
import shlex


def transition_marker_guard(source_dir: str, transition_id: str) -> list[str]:
    """Verify the sibling transition marker before touching the source tree."""

    source = source_dir.rstrip("/")
    parent, run_id = posixpath.split(source)
    coordination = posixpath.join(parent, f".{run_id}.execution-coordination")
    marker = posixpath.join(coordination, "transition.json")
    return [
        f"exec 7> {shlex.quote(coordination + '.lock')}",
        "flock -x 7",
        f'test "$(cat -- {shlex.quote(marker)})" = {shlex.quote(transition_id)}',
        "flock -u 7",
    ]


def source_quarantine_path(source_dir: str, transition_id: str) -> str:
    """Return the private sibling used to freeze a transition source."""

    source = source_dir.rstrip("/")
    parent, run_id = posixpath.split(source)
    return posixpath.join(parent, f".{run_id}.deleting.{transition_id}")


def rollback_uncommitted_transition_lines(source_dir: str, transition_id: str) -> list[str]:
    """Install an EXIT trap that restores source and retires a proven rollback.

    The trap is installed immediately after authenticating the transition marker,
    before destination checks or staging creation. It removes only paths this
    exact script proved it published. The marker is cleared under the shared lock
    only when every rollback postcondition is true; otherwise it remains as an
    explicit manual-recovery blocker.
    """

    source = source_dir.rstrip("/")
    parent, run_id = posixpath.split(source)
    quarantine = source_quarantine_path(source, transition_id)
    coordination = posixpath.join(parent, f".{run_id}.execution-coordination")
    marker = posixpath.join(coordination, "transition.json")
    return [
        'staging=""',
        "source_frozen=0",
        "published_paths=()",
        "cleanup_uncommitted_transition() {",
        "  status=$?",
        "  trap - EXIT",
        # Keep attempting independent recovery steps after one cleanup error.
        # The marker is the final decision: it is retired only if every exact
        # postcondition below proves the transition fully rolled back.
        "  set +e",
        "  cleanup_ok=1",
        '  if test -n "${verify_output:-}"; then',
        '    rm -f -- "$verify_output" || cleanup_ok=0',
        "  fi",
        '  if test -n "${staging:-}"; then',
        '    rm -rf -- "$staging" || cleanup_ok=0',
        '    test ! -e "$staging" || cleanup_ok=0',
        "  fi",
        # Roll publication back in reverse commit order. For archives this
        # removes the manifest marker before its checksum and payload.
        "  for ((published_index=${#published_paths[@]} - 1; published_index >= 0; published_index--)); do",
        '    rm -rf -- "${published_paths[$published_index]}" || cleanup_ok=0',
        '    test ! -e "${published_paths[$published_index]}" || cleanup_ok=0',
        "  done",
        f'  if test "$source_frozen" = 1 && test ! -e {shlex.quote(source)}; then',
        f"    mv -T -- {shlex.quote(quarantine)} {shlex.quote(source)} || cleanup_ok=0",
        "  fi",
        # A writer that recreated the original source while the quarantine
        # still exists leaves two trees requiring inspection; never clear the
        # marker or overwrite either tree in that ambiguous state.
        f"  test -d {shlex.quote(source)} || cleanup_ok=0",
        f"  test ! -e {shlex.quote(quarantine)} || cleanup_ok=0",
        '  if test "$cleanup_ok" = 1; then',
        "    if flock -x 7; then",
        f'      if test "$(cat -- {shlex.quote(marker)} 2>/dev/null)" = {shlex.quote(transition_id)}; then',
        f"        rm -f -- {shlex.quote(marker)} || cleanup_ok=0",
        "      else",
        "        cleanup_ok=0",
        "      fi",
        "      flock -u 7 || cleanup_ok=0",
        "    else",
        "      cleanup_ok=0",
        "    fi",
        "  fi",
        '  if test "$cleanup_ok" != 1; then',
        "    echo 'transition rollback incomplete; marker retained for recovery' >&2",
        "  fi",
        '  exit "$status"',
        "}",
        "trap cleanup_uncommitted_transition EXIT",
    ]


def freeze_source_before_publication_lines(source_dir: str, transition_id: str) -> list[str]:
    """Prove and quarantine the source before publishing a destination."""

    source = source_dir.rstrip("/")
    quarantine = source_quarantine_path(source, transition_id)
    return [
        source_snapshot_assignment(source, "source_final_sha"),
        'if test "$source_final_sha" != "$source_baseline_sha"; then '
        "echo 'source changed during transition; refusing publication' >&2; exit 77; fi",
        f"test ! -e {shlex.quote(quarantine)}",
        f"mv -T -- {shlex.quote(source)} {shlex.quote(quarantine)}",
        "source_frozen=1",
        source_snapshot_assignment(quarantine, "quarantine_sha"),
        'if test "$quarantine_sha" != "$source_baseline_sha"; then '
        "echo 'frozen source changed after rename; refusing publication' >&2; exit 78; fi",
    ]


def delete_frozen_source_lines(source_dir: str, transition_id: str) -> list[str]:
    """Delete a pre-certified quarantine after destination publication commits."""

    source = source_dir.rstrip("/")
    parent, run_id = posixpath.split(source)
    quarantine = source_quarantine_path(source, transition_id)
    coordination = posixpath.join(parent, f".{run_id}.execution-coordination")
    marker = posixpath.join(coordination, "transition.json")
    return [
        f"rm -rf -- {shlex.quote(quarantine)}",
        "flock -x 7",
        f'test "$(cat -- {shlex.quote(marker)})" = {shlex.quote(transition_id)}',
        f"rm -f -- {shlex.quote(marker)}",
        "flock -u 7",
        "echo FREED_SCRATCH",
    ]


def source_snapshot_assignment(source_dir: str, variable: str) -> str:
    """Render a deterministic source-tree digest before destructive cleanup."""

    program = "\n".join(
        [
            "import hashlib, os, stat, sys",
            "root = os.path.abspath(sys.argv[1])",
            "digest = hashlib.sha256()",
            "def visit(path, relative):",
            "    info = os.lstat(path)",
            "    mode = info.st_mode",
            "    digest.update(relative.encode('utf-8') + b'\\0' + str(stat.S_IMODE(mode)).encode() + b'\\0')",
            "    if stat.S_ISDIR(mode):",
            "        digest.update(b'd\\0')",
            "        for name in sorted(os.listdir(path)):",
            "            if relative == '.' and name == '.execution-source.lock':",
            "                continue",
            "            child_relative = name if relative == '.' else relative + '/' + name",
            "            visit(os.path.join(path, name), child_relative)",
            "    elif stat.S_ISREG(mode):",
            "        digest.update(b'f\\0')",
            "        with open(path, 'rb') as handle:",
            "            for chunk in iter(lambda: handle.read(1024 * 1024), b''):",
            "                digest.update(chunk)",
            "    elif stat.S_ISLNK(mode):",
            "        digest.update(b'l\\0' + os.readlink(path).encode('utf-8') + b'\\0')",
            "    else:",
            "        raise SystemExit('unsupported special file in transition source: ' + relative)",
            "visit(root, '.')",
            "print(digest.hexdigest())",
        ]
    )
    return f"{variable}=$(python3 -c {shlex.quote(program)} {shlex.quote(source_dir.rstrip('/'))})"


__all__ = [
    "delete_frozen_source_lines",
    "freeze_source_before_publication_lines",
    "rollback_uncommitted_transition_lines",
    "source_snapshot_assignment",
    "transition_marker_guard",
]
