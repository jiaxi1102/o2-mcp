"""Fault-injection tests for deletion-capable run transitions."""

from __future__ import annotations

import hashlib
import os
import subprocess
from types import SimpleNamespace

import pytest

from o2mcp.runorg.executor import O2Runs
from o2mcp.runorg.lifecycle_coordination import coordination_lock, coordination_root
from o2mcp.runorg.runs import RunLayout, RunManifest, plan_archive_script, plan_promote_script
from o2mcp.runorg.transition_coordinator import _BEGIN_PROGRAM
from o2mcp.runorg.transition_guards import live_jobs_command, require_certified_terminal_execution


def _seed_transition_marker(source: str, manifest: RunManifest, action: str) -> str:
    """Create and return the exact executor-owned sibling marker path."""

    token = hashlib.sha256(f"{action}\0{manifest.run_id}\0{manifest.to_json()}".encode()).hexdigest()
    root = os.path.join(os.path.dirname(source), f".{manifest.run_id}.execution-coordination")
    os.makedirs(root, exist_ok=True)
    marker = os.path.join(root, "transition.json")
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write(token)
    return marker


def _write_tool(path, name: str, body: str) -> None:
    """Install one executable test double ahead of platform utilities."""

    tool = path / name
    tool.write_text(body)
    tool.chmod(0o755)


def test_promotion_detects_source_write_after_copy_verification(tmp_path) -> None:
    """A late writer leaves source intact and no committed kept destination."""

    layout = RunLayout(
        str(tmp_path / "scratch"),
        str(tmp_path / "group"),
        str(tmp_path / "standby"),
        str(tmp_path / "registry.jsonl"),
    )
    manifest = RunManifest(
        run_id="RUN_20260822T010203Z_transition__fault",
        campaign="transition",
        pipeline="canary",
        created_utc="20260822T010203Z",
        datasets=["dataset"],
    )
    source = layout.run_dir("active", manifest.campaign, manifest.run_id)
    destination = layout.run_dir("kept", manifest.campaign, manifest.run_id)
    os.makedirs(source)
    with open(os.path.join(source, "payload.txt"), "w", encoding="utf-8") as handle:
        handle.write("authenticated bytes\n")

    tools = tmp_path / "tools"
    tools.mkdir()
    _write_tool(tools, "flock", "#!/bin/sh\nexit 0\n")
    _write_tool(
        tools,
        "rsync",
        """#!/bin/bash
set -e
dry=0
paths=()
for arg in "$@"; do
  case "$arg" in
    -nric) dry=1 ;;
    --*|-*) ;;
    *) paths+=("$arg") ;;
  esac
done
if test "$dry" = 1; then
  count=0
  if test -e "$INJECT_COUNT"; then count=$(cat "$INJECT_COUNT"); fi
  count=$((count + 1))
  printf '%s\n' "$count" > "$INJECT_COUNT"
  # Mutate only after both dry-run comparisons have passed.  The transition's
  # independent final snapshot must still reject these unverified bytes.
  if test "$count" = 2; then printf 'late bytes\n' > "$INJECT_SOURCE/late.txt"; fi
  exit 0
fi
src=${paths[${#paths[@]}-2]}
dst=${paths[${#paths[@]}-1]}
mkdir -p -- "$dst"
/bin/cp -R -- "${src%/}/." "${dst%/}/"
""",
    )
    _write_tool(
        tools,
        "mv",
        """#!/bin/bash
set -e
paths=()
for arg in "$@"; do
  case "$arg" in --no-clobber|-T|--) ;; *) paths+=("$arg") ;; esac
done
src=${paths[0]}
dst=${paths[1]}
/bin/mv -- "$src" "$dst"
""",
    )
    _write_tool(
        tools,
        "sha256sum",
        """#!/usr/bin/env python3
import hashlib, pathlib, sys
path = pathlib.Path(sys.argv[-1])
print(hashlib.sha256(path.read_bytes()).hexdigest(), path)
""",
    )
    script = plan_promote_script(layout, manifest, source_dir=source)
    marker = _seed_transition_marker(source, manifest, "promote")
    assert ".execution-source.lock" in script
    assert "--exclude=/.execution-source.lock" in script
    assert "--exclude=.execution-source.lock" not in script
    assert script.index("source_final_sha=") < script.index(f'mv --no-clobber -T -- "$staging" {destination}')
    assert script.index("source_final_sha=") < script.rindex("rm -rf")
    result = subprocess.run(
        ["/bin/bash"],
        input=script,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PATH": f"{tools}:{os.environ['PATH']}",
            "INJECT_SOURCE": source,
            "INJECT_COUNT": str(tmp_path / "rsync-dry-run-count"),
        },
        check=False,
    )
    assert result.returncode == 77, result.stderr
    assert os.path.exists(os.path.join(source, "payload.txt"))
    assert os.path.exists(os.path.join(source, "late.txt"))
    assert not os.path.exists(destination)
    assert not os.path.exists(marker)


def test_archive_detects_source_write_before_publication(tmp_path) -> None:
    """Archive exit 77 leaves no tarball, checksum, or manifest commit marker."""

    layout = RunLayout(
        str(tmp_path / "scratch"),
        str(tmp_path / "group"),
        str(tmp_path / "standby"),
        str(tmp_path / "registry.jsonl"),
    )
    manifest = RunManifest(
        run_id="RUN_20260822T010203Z_transition__archive-fault",
        campaign="transition",
        pipeline="canary",
        created_utc="20260822T010203Z",
        datasets=["dataset"],
    )
    source = layout.run_dir("active", manifest.campaign, manifest.run_id)
    tarball = layout.archive_tarball(manifest.campaign, manifest.run_id)
    checksum = layout.archive_checksum(manifest.campaign, manifest.run_id)
    archived_manifest = layout.archive_manifest(manifest.campaign, manifest.run_id)
    os.makedirs(source)
    with open(os.path.join(source, "payload.txt"), "w", encoding="utf-8") as handle:
        handle.write("authenticated bytes\n")

    tools = tmp_path / "archive-tools"
    tools.mkdir()
    _write_tool(tools, "flock", "#!/bin/sh\nexit 0\n")
    _write_tool(
        tools,
        "tar",
        """#!/bin/bash
set -e
while test "$#" -gt 0; do
  if test "$1" = "-cf"; then
    shift
    printf 'synthetic archive bytes\n' > "$1"
    exit 0
  fi
  shift
done
exit 2
""",
    )
    _write_tool(
        tools,
        "zstd",
        """#!/bin/bash
set -e
# tar creation is stubbed separately, so this invocation is the integrity gate.
printf 'late archive bytes\n' > "$INJECT_SOURCE/late.txt"
exit 0
""",
    )
    _write_tool(
        tools,
        "mv",
        """#!/bin/bash
set -e
paths=()
for arg in "$@"; do
  case "$arg" in --no-clobber|-T|--) ;; *) paths+=("$arg") ;; esac
done
/bin/mv -- "${paths[0]}" "${paths[1]}"
""",
    )
    _write_tool(
        tools,
        "sha256sum",
        """#!/usr/bin/env python3
import hashlib, pathlib, sys
path = pathlib.Path(sys.argv[-1])
print(hashlib.sha256(path.read_bytes()).hexdigest(), path)
""",
    )

    script = plan_archive_script(layout, manifest, source_dir=source)
    marker = _seed_transition_marker(source, manifest, "archive")
    assert script.index("source_final_sha=") < script.index('mv --no-clobber -- "$staging/archive.tar.zst"')
    result = subprocess.run(
        ["/bin/bash"],
        input=script,
        text=True,
        capture_output=True,
        env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "INJECT_SOURCE": source},
        check=False,
    )

    assert result.returncode == 77, result.stderr
    assert os.path.exists(os.path.join(source, "payload.txt"))
    assert os.path.exists(os.path.join(source, "late.txt"))
    assert not os.path.exists(tarball)
    assert not os.path.exists(checksum)
    assert not os.path.exists(archived_manifest)
    assert not os.path.exists(marker)


def test_marking_refuses_a_source_that_vanished_before_the_lock(tmp_path) -> None:
    """A marker must never be published for a source that no longer exists.

    Preview validation happens before the coordination lock, so a concurrent
    transition can relocate the source in between.  Marking anyway strands the
    run: the launched script cannot lock a missing source, its rollback trap
    keeps the marker, and recovery reports a manual blocker.
    """

    campaign = tmp_path / "camp"
    campaign.mkdir()
    run_root = str(campaign / "RUN_20260101T000000Z_camp__v1")

    def mark() -> subprocess.CompletedProcess:
        """Run the real marking program exactly as the coordinator does."""

        return subprocess.run(
            [
                "python3",
                "-c",
                _BEGIN_PROGRAM,
                run_root,
                coordination_root(run_root),
                coordination_lock(run_root),
                "a" * 64,
            ],
            capture_output=True,
            text=True,
        )

    vanished = mark()
    assert vanished.returncode != 0
    assert "transition source is missing" in vanished.stderr
    assert not os.path.exists(os.path.join(coordination_root(run_root), "transition.json"))

    # The same call succeeds once the source is a real directory.
    os.makedirs(run_root)
    marked = mark()
    assert marked.returncode == 0, marked.stderr
    assert os.path.exists(os.path.join(coordination_root(run_root), "transition.json"))


def test_queue_query_survives_purged_job_ids(tmp_path) -> None:
    """A historical run must be able to prove nothing of its is still live.

    ``squeue -j`` exits nonzero once every named job has aged out of the
    controller, which is indistinguishable from a failed query, so certified
    historical runs could never leave scratch storage.  Reading the caller's own
    queue separates "nothing matched" from "could not ask".
    """

    def query(squeue_body: str, job_ids=("9000", "9001")) -> subprocess.CompletedProcess:
        """Run the generated query against one fake scheduler."""

        tools = tmp_path / f"tools-{abs(hash(squeue_body))}"
        tools.mkdir()
        _write_tool(tools, "squeue", squeue_body)
        return subprocess.run(
            ["sh", "-c", live_jobs_command(list(job_ids))],
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}"},
        )

    # Nothing of this run is queued any more: proceed rather than fail closed.
    purged = query("#!/bin/sh\nexit 0\n")
    assert purged.returncode == 0 and purged.stdout.strip() == ""

    # Another user's or another run's job must not block this transition.
    unrelated = query("#!/bin/sh\nprintf '9999|RUNNING\\n'\n")
    assert unrelated.returncode == 0 and unrelated.stdout.strip() == ""

    # A live job of this run, including one array element, still refuses it.
    live = query("#!/bin/sh\nprintf '9001|RUNNING\\n9999|PENDING\\n'\n")
    assert live.returncode == 0 and live.stdout.strip() == "9001|RUNNING"
    element = query("#!/bin/sh\nprintf '9000_3|RUNNING\\n'\n")
    assert element.returncode == 0 and element.stdout.strip() == "9000_3|RUNNING"

    # An actual query failure stays fatal.
    unreachable = query("#!/bin/sh\necho 'slurm_load_jobs error' >&2\nexit 1\n")
    assert unreachable.returncode != 0

    # The cases above exercise the generated command against fake schedulers;
    # they cannot reproduce real squeue's "Invalid job id specified" exit, which
    # is the behaviour that made purged runs unpromotable.  Pin the structural
    # contract instead: the query must never name job IDs to squeue.
    generated = live_jobs_command(["9000", "9001"])
    assert " -j " not in generated
    assert "-u " in generated


def test_transition_requires_matching_certified_terminal_state() -> None:
    """Planning cannot delete an inconsistently certified execution run.

    Runs registered outside the execution engine carry no execution provenance
    at all and keep their own release criteria; refusing them here would delete
    the lifecycle path of every ``o2_run_register`` run. A run that does carry
    execution provenance must still be certified terminal and self-consistent.
    """

    manifest = RunManifest(
        run_id="RUN_20260822T010203Z_transition__guard",
        campaign="transition",
        pipeline="canary",
        created_utc="20260822T010203Z",
        datasets=["dataset"],
    )
    require_certified_terminal_execution(manifest, "promote")
    require_certified_terminal_execution(manifest, "archive")

    manifest.provenance = {"execution": {"state": "ACTIVE"}}
    manifest.result = {"status": "ACTIVE"}
    with pytest.raises(ValueError, match="certified terminal"):
        require_certified_terminal_execution(manifest, "promote")

    # Present-but-inconsistent execution provenance still fails closed.
    manifest.provenance = {"execution": {"state": "COMPLETED"}}
    manifest.result = {"status": "FAILED"}
    with pytest.raises(ValueError, match="certified terminal"):
        require_certified_terminal_execution(manifest, "promote")

    manifest.provenance = {"execution": {}}
    manifest.result = {"status": "COMPLETED"}
    with pytest.raises(ValueError, match="certified terminal"):
        require_certified_terminal_execution(manifest, "promote")

    # A run allocated by prepare_execution_run() is engine-owned before its plan
    # is bound, so it must not fall through to the legacy path and be published
    # or deleted with no plan, jobs, or terminal evidence at all.
    manifest.provenance = {"execution_preparation": {"project": "canary"}}
    manifest.result = {}
    for action in ("promote", "archive"):
        with pytest.raises(ValueError, match="certified terminal"):
            require_certified_terminal_execution(manifest, action)

    manifest.provenance = {"execution": {"state": "COMPLETED"}}
    manifest.result = {"status": "COMPLETED"}
    require_certified_terminal_execution(manifest, "promote")


def test_transition_refuses_certified_run_with_live_slurm_job(tmp_path) -> None:
    """Terminal metadata alone cannot race an accounting-visible live writer."""

    layout = RunLayout(
        str(tmp_path / "scratch"),
        str(tmp_path / "group"),
        str(tmp_path / "standby"),
        str(tmp_path / "registry.jsonl"),
    )
    manifest = RunManifest(
        run_id="RUN_20260822T010203Z_transition__live",
        campaign="transition",
        pipeline="canary",
        created_utc="20260822T010203Z",
        datasets=["dataset"],
        slurm_job_ids=["12345"],
        result={"status": "COMPLETED"},
        provenance={"execution": {"state": "COMPLETED"}},
    )

    class Connection:
        def run(self, command, *, timeout, input_text=None, **_kwargs):
            del timeout, input_text
            if command.startswith("cat "):
                return SimpleNamespace(ok=True, stdout=manifest.to_json(), stderr="")
            # The query is now a small shell pipeline rather than a bare
            # squeue call, so match the tool anywhere in the command.
            if "squeue" in command:
                return SimpleNamespace(ok=True, stdout="12345|RUNNING\n", stderr="")
            return SimpleNamespace(ok=True, stdout="", stderr="")

    runs = object.__new__(O2Runs)
    runs.conn = Connection()
    runs.layout = layout
    runs.policy = SimpleNamespace(archive_excludes=())
    run_dir = layout.run_dir("active", manifest.campaign, manifest.run_id)
    with pytest.raises(ValueError, match="live Slurm jobs"):
        runs.promote(run_dir, dry_run=True)


def test_transition_refuses_a_manifest_that_changed_after_review(tmp_path) -> None:
    """The launched script embeds the run.json that was actually reviewed.

    A registry update can land between the preview that rendered the script and
    the lock that marks the transition.  Publishing the reviewed bytes anyway
    would drop the newly certified attempt, job IDs, or terminal state from the
    durable destination, so a changed manifest fails closed instead.
    """

    layout = RunLayout(
        str(tmp_path / "scratch"),
        str(tmp_path / "group"),
        str(tmp_path / "standby"),
        str(tmp_path / "registry.jsonl"),
    )
    reviewed = RunManifest(
        run_id="RUN_20260822T010203Z_transition__drift",
        campaign="transition",
        pipeline="canary",
        created_utc="20260822T010203Z",
        datasets=["dataset"],
        slurm_job_ids=["12345"],
        result={"status": "COMPLETED"},
    )
    # The same run after one more accepted job was recorded.
    updated = RunManifest.from_json(reviewed.to_json())
    updated.slurm_job_ids = ["12345", "12346"]

    class Connection:
        """Serve the reviewed manifest first, then the newer one."""

        def __init__(self) -> None:
            self.reads = 0

        def run(self, command, *, timeout, input_text=None, **_kwargs):
            del timeout
            if command.startswith("cat ") and "run.json" in command:
                self.reads += 1
                body = reviewed if self.reads == 1 else updated
                return SimpleNamespace(ok=True, stdout=body.to_json(), stderr="")
            if input_text == "{}":
                # The safe control-file reader; this run has no execution plan.
                return SimpleNamespace(ok=True, stdout='{"payload": null, "state": "MISSING"}', stderr="")
            if command.startswith("python3 -c"):
                payload = '{"already_marked": false, "job_ids": []}'
                return SimpleNamespace(ok=True, stdout=payload, stderr="")
            return SimpleNamespace(ok=True, stdout="", stderr="")

    runs = object.__new__(O2Runs)
    runs.conn = Connection()
    runs.layout = layout
    runs.policy = SimpleNamespace(archive_excludes=())
    run_dir = layout.run_dir("active", reviewed.campaign, reviewed.run_id)

    with pytest.raises(ValueError, match="changed after review"):
        runs.promote(run_dir, dry_run=False)


def test_post_digest_injection_recreates_source_but_is_not_deleted(tmp_path) -> None:
    """A write after source freeze survives while only quarantine is deleted."""

    layout = RunLayout(
        str(tmp_path / "scratch"),
        str(tmp_path / "group"),
        str(tmp_path / "standby"),
        str(tmp_path / "registry.jsonl"),
    )
    manifest = RunManifest(
        run_id="RUN_20260822T010203Z_transition__post-digest",
        campaign="transition",
        pipeline="canary",
        created_utc="20260822T010203Z",
        datasets=["dataset"],
    )
    source = layout.run_dir("active", manifest.campaign, manifest.run_id)
    destination = layout.run_dir("kept", manifest.campaign, manifest.run_id)
    os.makedirs(source)
    with open(os.path.join(source, "payload.txt"), "w", encoding="utf-8") as handle:
        handle.write("authenticated bytes\n")
    tools = tmp_path / "post-digest-tools"
    tools.mkdir()
    _write_tool(tools, "flock", "#!/bin/sh\nexit 0\n")
    _write_tool(
        tools,
        "rsync",
        """#!/bin/bash
set -e
dry=0; paths=()
for arg in "$@"; do
  case "$arg" in -nric) dry=1 ;; --*|-*) ;; *) paths+=("$arg") ;; esac
done
if test "$dry" = 1; then exit 0; fi
src=${paths[${#paths[@]}-2]}; dst=${paths[${#paths[@]}-1]}
mkdir -p -- "$dst"; /bin/cp -R -- "${src%/}/." "${dst%/}/"
""",
    )
    _write_tool(
        tools,
        "mv",
        """#!/bin/bash
set -e
paths=()
for arg in "$@"; do
  case "$arg" in --no-clobber|-T|--) ;; *) paths+=("$arg") ;; esac
done
src=${paths[0]}; dst=${paths[1]}
/bin/mv -- "$src" "$dst"
if test "$src" = "$INJECT_SOURCE"; then
  mkdir -p -- "$INJECT_SOURCE"
  printf 'post-digest bytes\n' > "$INJECT_SOURCE/late.txt"
fi
""",
    )
    _write_tool(
        tools,
        "sha256sum",
        """#!/usr/bin/env python3
import hashlib, pathlib, sys
path = pathlib.Path(sys.argv[-1])
print(hashlib.sha256(path.read_bytes()).hexdigest(), path)
""",
    )
    script = plan_promote_script(layout, manifest, source_dir=source)
    _seed_transition_marker(source, manifest, "promote")
    result = subprocess.run(
        ["/bin/bash"],
        input=script,
        text=True,
        capture_output=True,
        env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "INJECT_SOURCE": source},
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert os.path.exists(os.path.join(source, "late.txt"))
    assert os.path.exists(destination)
    assert not any(".deleting." in name for name in os.listdir(os.path.dirname(source)))
