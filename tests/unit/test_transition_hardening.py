"""Fault-injection tests for deletion-capable run transitions."""

from __future__ import annotations

import hashlib
import os
import subprocess
from types import SimpleNamespace

import pytest

from o2mcp.runorg.executor import O2Runs
from o2mcp.runorg.runs import RunLayout, RunManifest, plan_promote_script
from o2mcp.runorg.transition_guards import require_certified_terminal_execution


def _seed_transition_marker(source: str, manifest: RunManifest, action: str) -> None:
    """Create the exact executor-owned sibling marker for script fault tests."""

    token = hashlib.sha256(f"{action}\0{manifest.run_id}\0{manifest.to_json()}".encode()).hexdigest()
    root = os.path.join(os.path.dirname(source), f".{manifest.run_id}.execution-coordination")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "transition.json"), "w", encoding="utf-8") as handle:
        handle.write(token)


def _write_tool(path, name: str, body: str) -> None:
    """Install one executable test double ahead of platform utilities."""

    tool = path / name
    tool.write_text(body)
    tool.chmod(0o755)


def test_promotion_detects_source_write_after_copy_verification(tmp_path) -> None:
    """A late writer leaves source intact instead of losing unverified bytes."""

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
if test "$dry" = 1; then exit 0; fi
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
if test "$dst" = "$INJECT_DEST"; then
  printf 'late bytes\n' > "$INJECT_SOURCE/late.txt"
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
    assert ".execution-source.lock" in script
    assert "--exclude=/.execution-source.lock" in script
    assert "--exclude=.execution-source.lock" not in script
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
            "INJECT_DEST": destination,
        },
        check=False,
    )
    assert result.returncode == 77, result.stderr
    assert os.path.exists(os.path.join(source, "payload.txt"))
    assert os.path.exists(os.path.join(source, "late.txt"))


def test_transition_requires_matching_certified_terminal_state() -> None:
    """Planning cannot delete an active or inconsistently certified run."""

    manifest = RunManifest(
        run_id="RUN_20260822T010203Z_transition__guard",
        campaign="transition",
        pipeline="canary",
        created_utc="20260822T010203Z",
        datasets=["dataset"],
    )
    with pytest.raises(ValueError, match="certified terminal"):
        require_certified_terminal_execution(manifest, "promote")
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
            if command.startswith("squeue "):
                return SimpleNamespace(ok=True, stdout="12345|RUNNING\n", stderr="")
            return SimpleNamespace(ok=True, stdout="", stderr="")

    runs = object.__new__(O2Runs)
    runs.conn = Connection()
    runs.layout = layout
    runs.policy = SimpleNamespace(archive_excludes=())
    run_dir = layout.run_dir("active", manifest.campaign, manifest.run_id)
    with pytest.raises(ValueError, match="live Slurm jobs"):
        runs.promote(run_dir, dry_run=True)


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
