"""Offline unit tests for the generic run-organization engine (o2mcp.runorg).

The pure conventions need no third-party deps; the O2Runs executor is exercised with
an injected runner (no network). Everything is parameterized by a synthetic RunPolicy.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from types import SimpleNamespace

from o2mcp import CommandResult, O2Config
from o2mcp import O2Connection as _ProductionO2Connection
from o2mcp.broker import BrokerExecutionResult
from o2mcp.runorg import (
    RETENTION_KEEP,
    RETENTION_SWEEP,
    STATUS_KEPT,
    O2Runs,
    RegistryUpdate,
    RunLayout,
    RunManifest,
    RunPolicy,
    campaign_of,
    classify_run,
    is_regenerable_intermediate,
    migration_target,
    plan_archive_script,
    plan_promote_script,
    plan_register_commands,
    variant_of,
)
from o2mcp.runorg.executor import _infer_pipeline
from o2mcp.runorg.runs import _safe
from o2mcp.runorg.transition_coordinator import TransitionBoundary


def _seed_transition_marker(source: str, manifest: RunManifest, action: str) -> None:
    """Create the marker normally established by the executor before launch."""

    token = hashlib.sha256(f"{action}\0{manifest.run_id}\0{manifest.to_json()}".encode()).hexdigest()
    root = os.path.join(os.path.dirname(source), f".{manifest.run_id}.execution-coordination")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "transition.json"), "w", encoding="utf-8") as handle:
        handle.write(token)


class O2Connection(_ProductionO2Connection):
    """Select the offline fake-runner command path for run-organization tests."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("_legacy_test_transport", True)
        super().__init__(*args, **kwargs)


TEST_POLICY = RunPolicy(
    pipeline_keywords=(("ultrack", "track"), ("grid", "grid")),
    fallback_pipeline="unknown",
    pipelines=("grid", "track"),
    sweep_markers=("_test", "sweep", "wip"),
    view_suffixes=("maps", "qc"),
    heavy_view_suffixes=("maps",),
    run_subdirs=("logs", "views"),
    archive_excludes=("source_views",),
)


class _Runner:
    """Fake subprocess runner: master is up; scriptable responder for remote commands."""

    def __init__(self, responder=None):
        self.calls: list[dict] = []
        self._responder = responder

    def __call__(self, argv, timeout, input_text) -> CommandResult:
        self.calls.append({"argv": list(argv), "input": input_text})
        if "-O" in argv and "check" in argv:
            return CommandResult(list(argv), 0, "", "")
        if argv[:2] == [O2Connection.SSH_EXECUTABLE, "-G"]:
            return CommandResult(list(argv), 0, f"controlpath /tmp/{argv[-1]}-control.sock\n", "")
        if self._responder is not None:
            out, err, rc = self._responder(argv, input_text)
            return CommandResult(list(argv), rc, out, err)
        return CommandResult(list(argv), 0, "", "")


def _cfg(tmp_path):
    policy_file = tmp_path / "O2_POLICY.json"
    policy_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation": "00000000-0000-4000-8000-000000000001",
                "revision": 1,
                "mode": "reuse_only",
                "login_grant": None,
                "login_attempt": None,
                "events": [],
            }
        )
    )
    policy_file.chmod(0o600)
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text(
        "Host o2 o2-transfer\n  HostName o2.hms.harvard.edu\n  User jiz947\n  ControlPath /tmp/%n-control.sock\n"
    )
    return O2Config(
        host_alias="o2",
        transfer_alias="o2-transfer",
        connect_timeout=20,
        policy_file=policy_file,
        ssh_config_file=ssh_config,
    )


def _runs(tmp_path, responder=None, policy=TEST_POLICY) -> O2Runs:
    return O2Runs(O2Connection(_cfg(tmp_path), runner=_Runner(responder)), policy)


# --- pure conventions --------------------------------------------------------
def test_manifest_round_trip_and_validation():
    m = RunManifest(
        run_id="RUN_20260605T110309Z_camp__v1",
        campaign="camp",
        pipeline="grid",
        created_utc="20260605T110309Z",
        datasets=["ds1"],
    )
    assert m.validate(for_register=True) == []
    assert RunManifest.from_json(m.to_json()).run_id == m.run_id


def test_manifest_rewrite_preserves_unknown_future_fields():
    """Execution reconciliation must not truncate fields from a newer schema."""

    payload = {
        "run_id": "RUN_20260605T110309Z_camp__v1",
        "campaign": "camp",
        "pipeline": "grid",
        "created_utc": "20260605T110309Z",
        "datasets": ["ds1"],
        "future_release_binding": {"sha256": "a" * 64, "accepted": True},
    }
    restored = json.loads(RunManifest.from_dict(payload).to_json())
    assert restored["future_release_binding"] == payload["future_release_binding"]


def test_run_id_and_campaign_components_fail_closed():
    """Loose timestamps and dot traversal components cannot identify new runs."""

    loose = RunManifest(
        run_id="RUN_20260605T_camp__v1",
        campaign="camp",
        pipeline="grid",
        created_utc="20260605T000000Z",
        datasets=["ds1"],
    )
    assert any("does not match" in problem for problem in loose.validate(for_register=True))
    assert _safe(".") == "uncategorized"
    assert _safe("..") == "uncategorized"


def test_classify_keep_sweep_via_policy():
    assert classify_run("RUN_20260101T000000Z_x__final", TEST_POLICY)["retention"] == RETENTION_KEEP  # keep-marker
    assert classify_run("RUN_20260101T000000Z_x__wip", TEST_POLICY)["retention"] == RETENTION_SWEEP  # sweep-marker
    keep = classify_run(
        "RUN_20260101T000000Z_x__plain", TEST_POLICY, result_status="COMPLETED", is_latest_in_campaign=True
    )
    assert keep["retention"] == RETENTION_KEEP  # latest COMPLETED
    assert classify_run("RUN_20260101T000000Z_x__plain", TEST_POLICY)["retention"] == RETENTION_SWEEP  # default


def test_infer_pipeline_and_campaign_via_policy():
    assert _infer_pipeline("RUN_x_ultrack_v2", TEST_POLICY) == "track"
    assert _infer_pipeline("RUN_x_mystery", TEST_POLICY) == "unknown"  # fallback
    assert (
        campaign_of("RUN_20260101T000000Z_foo_grid_maps", TEST_POLICY.view_suffixes) == "foo_grid"
    )  # strips view-suffix
    assert campaign_of("RUN_20260101T000000Z_camp__v1", TEST_POLICY.view_suffixes) == "camp"
    assert variant_of("RUN_20260101T000000Z_camp__v1", TEST_POLICY.view_suffixes) == "v1"


def test_regenerable_and_migration_via_policy():
    assert is_regenerable_intermediate("RUN_x_foo_maps", TEST_POLICY) is True  # heavy view suffix
    assert is_regenerable_intermediate("RUN_x_foo_sweep", TEST_POLICY) is True  # heavy slug marker
    assert is_regenerable_intermediate("RUN_x_foo_plain", TEST_POLICY, size_bytes=1024) is False
    assert migration_target("RUN_x_foo__final", TEST_POLICY)["target"] == "promote"
    assert (
        migration_target("RUN_x_foo_final_maps", TEST_POLICY)["target"] == "archive"
    )  # keep-marker but heavy → archive


# --- planners (policy-driven) ------------------------------------------------
def test_plan_register_includes_policy_subdirs(tmp_path):
    layout = RunLayout(
        str(tmp_path / "scratch"),
        str(tmp_path / "group"),
        str(tmp_path / "standby"),
        str(tmp_path / "registry.jsonl"),
    )
    m = RunManifest(
        run_id="RUN_20260101T000000Z_camp__v1",
        campaign="camp",
        pipeline="grid",
        created_utc="20260101T000000Z",
        datasets=["d"],
    )
    cmds = plan_register_commands(layout, m, TEST_POLICY.run_subdirs)
    assert any("logs" in c and "views" in c for c in cmds)  # mkdir creates the policy subdirs
    assert "&& mkdir " in cmds[0]  # run root itself is an exclusive create


def test_plan_archive_uses_policy_excludes(tmp_path):
    layout = RunLayout(
        str(tmp_path / "scratch"),
        str(tmp_path / "group"),
        str(tmp_path / "standby"),
        str(tmp_path / "registry.jsonl"),
    )
    m = RunManifest(
        run_id="RUN_20260101T000000Z_camp__v1",
        campaign="camp",
        pipeline="grid",
        created_utc="20260101T000000Z",
        datasets=["d"],
        status=STATUS_KEPT,
    )
    script = plan_archive_script(
        layout,
        m,
        source_dir="/scratch/runs/camp/RUN_20260101T000000Z_camp__v1",
        archive_excludes=TEST_POLICY.archive_excludes,
    )
    assert "--exclude=source_views" in script and "zstd" in script


def _transition_test_environment(tmp_path):
    """Provide a no-op flock on macOS for pre-publication shell regressions."""

    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    flock = tools / "flock"
    flock.write_text("#!/bin/sh\nexit 0\n")
    flock.chmod(0o755)
    return {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}"}


def test_promotion_refuses_existing_destination_without_deleting_source(tmp_path):
    """A stale durable tree cannot be merged with a clean source then promoted."""

    layout = RunLayout(
        str(tmp_path / "scratch"),
        str(tmp_path / "group"),
        str(tmp_path / "standby"),
        str(tmp_path / "registry.jsonl"),
    )
    manifest = RunManifest(
        run_id="RUN_20260101T000000Z_camp__v1",
        campaign="camp",
        pipeline="grid",
        created_utc="20260101T000000Z",
        datasets=["d"],
    )
    source = layout.run_dir("active", manifest.campaign, manifest.run_id)
    destination = layout.run_dir("kept", manifest.campaign, manifest.run_id)
    os.makedirs(source)
    os.makedirs(destination)
    with open(os.path.join(source, "clean.txt"), "w", encoding="utf-8") as handle:
        handle.write("clean\n")
    with open(os.path.join(destination, "stale.txt"), "w", encoding="utf-8") as handle:
        handle.write("stale\n")

    script = plan_promote_script(layout, manifest, source_dir=source)
    _seed_transition_marker(source, manifest, "promote")
    assert "rsync -nric --delete" in script and "mv --no-clobber -T" in script
    result = subprocess.run(
        ["/bin/bash"],
        input=script,
        text=True,
        capture_output=True,
        env=_transition_test_environment(tmp_path),
        check=False,
    )
    assert result.returncode == 76
    assert os.path.exists(os.path.join(source, "clean.txt"))
    assert os.path.exists(os.path.join(destination, "stale.txt"))


def test_archive_refuses_partial_destination_without_clobbering_source(tmp_path):
    """Interrupted or existing archive artifacts require explicit inspection."""

    layout = RunLayout(
        str(tmp_path / "scratch"),
        str(tmp_path / "group"),
        str(tmp_path / "standby"),
        str(tmp_path / "registry.jsonl"),
    )
    manifest = RunManifest(
        run_id="RUN_20260101T000000Z_camp__v1",
        campaign="camp",
        pipeline="grid",
        created_utc="20260101T000000Z",
        datasets=["d"],
    )
    source = layout.run_dir("active", manifest.campaign, manifest.run_id)
    os.makedirs(source)
    with open(os.path.join(source, "clean.txt"), "w", encoding="utf-8") as handle:
        handle.write("clean\n")
    tarball = layout.archive_tarball(manifest.campaign, manifest.run_id)
    os.makedirs(os.path.dirname(tarball), exist_ok=True)
    with open(tarball, "wb") as handle:
        handle.write(b"existing archive bytes")

    script = plan_archive_script(layout, manifest, source_dir=source)
    _seed_transition_marker(source, manifest, "archive")
    assert script.count("mv --no-clobber") == 3
    assert script.index('archive.sha256"') < script.index('run.json"')
    result = subprocess.run(
        ["/bin/bash"],
        input=script,
        text=True,
        capture_output=True,
        env=_transition_test_environment(tmp_path),
        check=False,
    )
    assert result.returncode == 76
    with open(tarball, "rb") as handle:
        assert handle.read() == b"existing archive bytes"
    assert os.path.exists(os.path.join(source, "clean.txt"))


# --- executor (injected runner) ----------------------------------------------
def test_register_runs_commands_and_appends_registry(tmp_path):
    seen: list[str] = []

    def responder(argv, _inp):
        cmd = argv[-1]
        seen.append(cmd)
        return ("20260101T000000Z", "", 0) if "date -u" in cmd else ("", "", 0)

    res = _runs(tmp_path, responder).register(campaign="camp", pipeline="grid", datasets=["d"], variant="v1")
    assert res["ok"] is True and res["run_id"].startswith("RUN_20260101T000000Z_camp__")
    assert any("registry" in c for c in seen)  # appended a registry row


def test_register_surfaces_registry_append_failure(tmp_path):
    """A prepared directory is not reported as fully registered if JSONL is stale."""

    def responder(argv, _inp):
        cmd = argv[-1]
        if "date -u" in cmd:
            return ("20260101T000000Z", "", 0)
        if "registry" in cmd:
            return ("", "registry is read-only", 1)
        return ("", "", 0)

    result = _runs(tmp_path, responder).register(
        campaign="camp",
        pipeline="grid",
        datasets=["d"],
        variant="v1",
    )
    assert result["ok"] is False
    assert result["error"] == "registry_write_failed"
    assert result["prepared"] is True
    assert "registry is read-only" in result["problems"]


def test_submit_validates_script_before_allocating_run(tmp_path):
    """A malformed submit request cannot orphan a registered run directory."""

    seen: list[str] = []

    def responder(argv, _inp):
        seen.append(argv[-1])
        return ("", "", 0)

    result = _runs(tmp_path, responder).submit_run(campaign="camp", pipeline="grid", datasets=["d"])
    assert result["ok"] is False and result["error"] == "bad_input"
    assert not any("date -u" in command or "mkdir" in command for command in seen)


def test_record_job_surfaces_registry_failure_after_manifest_write(tmp_path):
    """Callers can distinguish a live job from its pending registry annotation."""

    manifest = RunManifest(
        run_id="RUN_20260101T000000Z_camp__v1",
        campaign="camp",
        pipeline="grid",
        created_utc="20260101T000000Z",
        datasets=["d"],
    )

    def responder(argv, _inp):
        command = argv[-1]
        if "registry" in command:
            return ("", "registry unavailable", 1)
        return ("", "", 0)

    result = _runs(tmp_path, responder).record_job(
        "/scratch/runs/camp/RUN_20260101T000000Z_camp__v1",
        "12345",
        manifest=manifest,
    )
    assert result["ok"] is False
    assert result["manifest_written"] is True
    assert result["slurm_job_ids"] == ["12345"]


def test_recover_prepared_execution_run_repairs_registry_without_reallocation(tmp_path):
    """A registry-only preparation failure reuses the exact existing run ID/root."""

    manifest = RunManifest(
        run_id="RUN_20260101T000000Z_camp__v1",
        campaign="camp",
        pipeline="grid",
        created_utc="20260101T000000Z",
        datasets=["d"],
        provenance={"execution_preparation": {"project": "example-project"}},
    )
    calls: list[tuple[str, str | None]] = []

    def responder(argv, _inp):
        command = argv[-1]
        calls.append((command, _inp))
        if command.startswith("cat ") and "run.json" in command:
            return (manifest.to_json(), "", 0)
        if command.startswith("python3 -c") and _inp:
            return ('{"ok": true}\n', "", 0)
        return ("", "", 0)

    runs = _runs(tmp_path, responder)
    run_dir = runs.layout.run_dir("active", manifest.campaign, manifest.run_id)
    identity = runs.recover_prepared_execution_run(project="example-project", run_dir=run_dir)

    assert identity.run_id == manifest.run_id
    assert identity.run_root == run_dir
    assert any("registry" in command for command, _ in calls)
    assert not any("date -u" in command or ("mkdir" in command and "registry" not in command) for command, _ in calls)


def test_execution_sync_updates_manifest_and_registry_from_same_state(tmp_path):
    """Automatic synchronization binds plan provenance, jobs, and registry status."""

    manifest = RunManifest(
        run_id="RUN_20260101T000000Z_camp__v1",
        campaign="camp",
        pipeline="grid",
        created_utc="20260101T000000Z",
        datasets=["d"],
        provenance={"execution_preparation": {"project": "example-project"}},
        extra={"future_field": {"preserve": True}},
    )
    calls: list[tuple[str, str | None]] = []

    def responder(argv, _inp):
        command = argv[-1]
        calls.append((command, _inp))
        if command.startswith("cat ") and "run.json" in command:
            return (manifest.to_json(), "", 0)
        if command.startswith("python3 -c") and _inp:
            return ('{"ok": true}\n', "", 0)
        return ("", "", 0)

    runs = _runs(tmp_path, responder)
    run_dir = runs.layout.run_dir("active", manifest.campaign, manifest.run_id)
    plan = SimpleNamespace(
        plan_sha256="a" * 64,
        paths=SimpleNamespace(run_root=run_dir),
        campaign=manifest.campaign,
        run_id=manifest.run_id,
        pipeline=manifest.pipeline,
        datasets=[SimpleNamespace(dataset_id="d")],
        project="example-project",
        source_bundle_sha256="b" * 64,
        source_commit="c" * 40,
    )
    update = RegistryUpdate(
        plan_sha256=plan.plan_sha256,
        stage_id="analyze",
        stage_status="SUBMITTED",
        execution_status="SUBMITTED",
        job_ids=("12345",),
        attempt=1,
    )

    assert runs.validate_execution_plan(plan)["ok"] is True
    manifest.status = STATUS_KEPT
    assert runs.validate_execution_plan(plan)["ok"] is False
    assert runs.synchronize_execution(plan, update)["ok"] is False
    manifest.status = "active"
    plan.project = "wrong-project"
    assert runs.validate_execution_plan(plan)["ok"] is False
    plan.project = "example-project"
    result = runs.synchronize_execution(plan, update)

    assert result["ok"] is True
    transaction_input = next(
        input_text for command, input_text in calls if command.startswith("python3 -c") and input_text
    )
    transaction = json.loads(transaction_input)
    merged_manifest = json.loads(transaction["manifest_text"])
    registry_row = json.loads(transaction["registry_line"])
    assert merged_manifest["provenance"]["execution"]["plan_sha256"] == plan.plan_sha256
    assert merged_manifest["future_field"] == {"preserve": True}
    assert merged_manifest["slurm_job_ids"] == ["12345"]
    assert registry_row["result_status"] == "SUBMITTED"


def test_promote_archive_dry_run_return_scripts(tmp_path):
    manifest_json = (
        '{"run_id":"RUN_20260101T000000Z_camp__v1","campaign":"camp","pipeline":"grid",'
        '"created_utc":"20260101T000000Z","status":"active","datasets":["d"],'
        '"result":{"status":"COMPLETED"},"provenance":{"execution":{"state":"COMPLETED"}}}'
    )

    def responder(argv, _inp):
        cmd = argv[-1]
        if "run.json" in cmd and cmd.startswith("cat "):
            return (manifest_json, "", 0)
        return ("", "", 0)

    runs = _runs(tmp_path, responder)
    rd = runs.layout.run_dir("active", "camp", "RUN_20260101T000000Z_camp__v1")
    promote = runs.promote(rd, dry_run=True)
    assert promote.started is False and "rsync" in promote.script
    archive = runs.archive(rd, dry_run=True)
    assert archive.started is False and "--exclude=source_views" in archive.script  # policy excludes in script


def test_live_transition_uses_persistent_transfer_broker(tmp_path, monkeypatch):
    """A detached promotion is framed through the role-specific transfer session."""

    manifest_json = (
        '{"run_id":"RUN_20260101T000000Z_camp__v1","campaign":"camp","pipeline":"grid",'
        '"created_utc":"20260101T000000Z","status":"active","datasets":["d"],'
        '"result":{"status":"COMPLETED"},"provenance":{"execution":{"state":"COMPLETED",'
        '"plan_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}}'
    )

    class _Broker:
        def __init__(self, *, transfer=False):
            self.transfer = transfer
            self.calls = []

        def execute(self, command, *, timeout, input_text=None):
            self.calls.append({"command": command, "timeout": timeout, "input": input_text})
            if command.startswith("cat ") and "run.json" in command:
                stdout = manifest_json
            elif self.transfer and command.startswith("nohup bash "):
                stdout = "PID 4321\n"
            else:
                stdout = ""
            return BrokerExecutionResult(0, stdout, "", False, 0.01, False, False)

    login_broker = _Broker()
    transfer_broker = _Broker(transfer=True)
    config = _cfg(tmp_path)
    runs = O2Runs(
        _ProductionO2Connection(
            config,
            broker_client=login_broker,
            transfer_broker_client=transfer_broker,
        ),
        TEST_POLICY,
    )
    run_dir = runs.layout.run_dir("active", "camp", "RUN_20260101T000000Z_camp__v1")
    # This test isolates transfer-broker routing. Dedicated transition tests
    # exercise marker creation and current receipt certification.
    monkeypatch.setattr("o2mcp.runorg.transition_executor.begin_transition", lambda *_args: TransitionBoundary(()))
    monkeypatch.setattr("o2mcp.runorg.transition_executor.O2ExecutionBackend.read_text", lambda *_args: "{}")
    monkeypatch.setattr(
        "o2mcp.runorg.transition_executor.ExecutionPlan.from_json",
        lambda *_args, **_kwargs: SimpleNamespace(
            paths=SimpleNamespace(run_root=run_dir),
            run_id="RUN_20260101T000000Z_camp__v1",
            campaign="camp",
            pipeline="grid",
            datasets=(SimpleNamespace(dataset_id="d"),),
        ),
    )
    monkeypatch.setattr(
        "o2mcp.runorg.transition_executor.require_current_terminal_evidence",
        lambda *_args: None,
    )
    plan = runs.promote(run_dir, dry_run=False)

    assert plan.started is True and plan.pid == "4321"
    assert not any(call["command"].startswith("nohup bash ") for call in login_broker.calls)
    launch = next(call for call in transfer_broker.calls if call["command"].startswith("nohup bash "))
    assert launch["timeout"] == 60.0


def test_read_manifest_consults_policy_legacy_reader(tmp_path):
    sentinel = RunManifest(
        run_id="RUN_20260101T000000Z_camp__v1",
        campaign="camp",
        pipeline="grid",
        created_utc="20260101T000000Z",
        datasets=["d"],
    )
    calls = []

    def reader(run_dir, *, read):
        calls.append(run_dir)
        return sentinel

    policy = RunPolicy(legacy_manifest_reader=reader)
    runs = _runs(tmp_path, responder=lambda argv, inp: ("", "", 0), policy=policy)  # run.json absent
    got = runs.read_manifest("/scratch/runs/camp/RUN_20260101T000000Z_camp__v1")
    assert got is sentinel and calls == ["/scratch/runs/camp/RUN_20260101T000000Z_camp__v1"]


def test_classify_and_list_run_dirs_default_to_grouped_layout():
    # The programmatic defaults must agree (regression: classify once defaulted False
    # while list_run_dirs and the o2_run_classify tool defaulted True → empty scans).
    import inspect

    default = lambda fn: inspect.signature(fn).parameters["depth_grouped"].default  # noqa: E731
    assert default(O2Runs.classify) is True
    assert default(O2Runs.list_run_dirs) is True
