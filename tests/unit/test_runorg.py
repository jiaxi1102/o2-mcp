"""Offline unit tests for the generic run-organization engine (o2mcp.runorg).

The pure conventions need no third-party deps; the O2Runs executor is exercised with
an injected runner (no network). Everything is parameterized by a synthetic RunPolicy.
"""

from __future__ import annotations

from o2mcp import CommandResult, O2Config, O2Connection
from o2mcp.runorg import (
    RETENTION_KEEP,
    RETENTION_SWEEP,
    STATUS_KEPT,
    O2Runs,
    RunLayout,
    RunManifest,
    RunPolicy,
    campaign_of,
    classify_run,
    is_regenerable_intermediate,
    migration_target,
    plan_archive_script,
    plan_register_commands,
    variant_of,
)
from o2mcp.runorg.executor import _infer_pipeline

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
        if self._responder is not None:
            out, err, rc = self._responder(argv, input_text)
            return CommandResult(list(argv), rc, out, err)
        return CommandResult(list(argv), 0, "", "")


def _cfg(tmp_path):
    return O2Config(
        host_alias="o2", transfer_alias="o2-transfer", connect_timeout=20, lock_file=tmp_path / "O2_DISABLED"
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
    layout = RunLayout.from_config(_cfg(tmp_path))
    m = RunManifest(
        run_id="RUN_20260101T000000Z_camp__v1",
        campaign="camp",
        pipeline="grid",
        created_utc="20260101T000000Z",
        datasets=["d"],
    )
    cmds = plan_register_commands(layout, m, TEST_POLICY.run_subdirs)
    assert any("logs" in c and "views" in c for c in cmds)  # mkdir creates the policy subdirs


def test_plan_archive_uses_policy_excludes(tmp_path):
    layout = RunLayout.from_config(_cfg(tmp_path))
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


def test_promote_archive_dry_run_return_scripts(tmp_path):
    manifest_json = (
        '{"run_id":"RUN_20260101T000000Z_camp__v1","campaign":"camp","pipeline":"grid",'
        '"created_utc":"20260101T000000Z","status":"active","datasets":["d"]}'
    )

    def responder(argv, _inp):
        cmd = argv[-1]
        if "run.json" in cmd and cmd.startswith("cat "):
            return (manifest_json, "", 0)
        return ("", "", 0)

    runs = _runs(tmp_path, responder)
    rd = "/scratch/runs/camp/RUN_20260101T000000Z_camp__v1"
    promote = runs.promote(rd, dry_run=True)
    assert promote.started is False and "rsync" in promote.script
    archive = runs.archive(rd, dry_run=True)
    assert archive.started is False and "--exclude=source_views" in archive.script  # policy excludes in script


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
