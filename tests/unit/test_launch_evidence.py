"""Offline tests for the governed launch-evidence record.

These exercise the binding surface only: no SSH, no cluster, no MCP SDK. The
point of the record is that it refuses to exist when the launch chain disagrees,
so most of these assert a refusal.
"""

from __future__ import annotations

import pytest

from o2mcp.launch_evidence import (
    LAUNCH_EVIDENCE_SCHEMA,
    LaunchEvidenceError,
    build_launch_evidence,
    canonical_json,
    launch_evidence_digest,
    parse_json_artifact,
    plan_digest,
    required_package_files,
)

_PLAN = {
    "attempt_id": "002",
    "software": {"bundle": {"bundle_sha256": "b" * 64}},
    "runtime_wrapper": {"sha256": "w" * 64},
    "interpreter": {"sha256": "i" * 64, "closure_sha256": "c" * 64, "context_sha256": "x" * 64},
    "destination": {"inode": 6411787343743799620, "mount": "/n/scratch", "expected_package": "/pkg/attempt-002"},
}


def _diagnostic(plan: dict) -> dict:
    """A run diagnostic that agrees with the plan on every bound field."""

    return {
        "status": "diagnostic_success",
        "continuation_authorized": False,
        "schema_version": 2,
        "plan_sha256": plan_digest(plan),
        "attempt_id": "002",
        "runtime": {
            "source_bundle": {"bundle_sha256": "b" * 64},
            "runtime_wrapper_sha256": "w" * 64,
            "approved_interpreter": {
                "approved_path": "/usr/bin/python3",
                "sha256": "i" * 64,
                "runtime_context_sha256": "x" * 64,
                "dynamic_closure": {"closure_sha256": "c" * 64},
            },
        },
        "destination_binding": {
            "inode": 6411787343743799620,
            "linux_mountinfo": {"mount_point": "/n/scratch", "mount_source": "server:/scratch"},
        },
        "slurm": {
            "scheduler": {"job_id": "52085188", "allocated_hostnames": ["compute-b-16-192"]},
            "environment": {"SLURM_JOB_ACCOUNT": "tabin", "SLURM_JOB_PARTITION": "short"},
        },
        "launch": {"loaded_library_closure": {"loaded_closure_sha256": "l" * 64}},
        "output": {
            "package": "/pkg/attempt-002",
            "reopened_output_sha256": "r" * 64,
            "verification": {"status": "success", "n_payloads": 9},
        },
    }


def _digests() -> dict[str, str]:
    return {name: "d" * 64 for name in required_package_files()}


def _owner(plan: dict) -> dict:
    return {"plan_sha256": plan_digest(plan), "attempt_id": "002"}


def _build(**overrides):
    plan = overrides.pop("plan", _PLAN)
    diagnostic = overrides.pop("diagnostic", _diagnostic(plan))
    return build_launch_evidence(
        diagnostic=diagnostic,
        plan=plan,
        package_digests=overrides.pop("package_digests", _digests()),
        owner=overrides.pop("owner", _owner(plan)),
        approval=overrides.pop("approval", {"approval_reference": "operator approved"}),
        stage=overrides.pop("stage", "platform-canary"),
    )


def test_intact_chain_mints_a_record() -> None:
    record = _build()
    assert record["schema"] == LAUNCH_EVIDENCE_SCHEMA
    assert record["binding_check"]["all_links_agree"] is True
    assert record["submission"]["job_id"] == "52085188"
    assert record["runtime_identities"]["loaded_closure_sha256"] == "l" * 64
    assert record["operator_approval"]["approval_reference"] == "operator approved"


def test_digest_is_reproducible_and_content_dependent() -> None:
    first = launch_evidence_digest(_build())
    assert first == launch_evidence_digest(_build())
    assert first != launch_evidence_digest(_build(stage="acquisition"))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("runtime", "source_bundle", "bundle_sha256"), "0" * 64),
        (("runtime", "runtime_wrapper_sha256"), "0" * 64),
        (("runtime", "approved_interpreter", "sha256"), "0" * 64),
        (("runtime", "approved_interpreter", "dynamic_closure", "closure_sha256"), "0" * 64),
        (("runtime", "approved_interpreter", "runtime_context_sha256"), "0" * 64),
        (("destination_binding", "inode"), 1),
        (("output", "package"), "/pkg/somewhere-else"),
        (("attempt_id",), "003"),
    ],
)
def test_any_drifted_link_refuses_to_mint(path, replacement) -> None:
    """A record must not exist for a run that differs from the approved plan."""

    diagnostic = _diagnostic(_PLAN)
    target = diagnostic
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(LaunchEvidenceError, match="does not agree"):
        _build(diagnostic=diagnostic)


def test_absent_field_is_reported_as_absent_not_mismatched() -> None:
    diagnostic = _diagnostic(_PLAN)
    del diagnostic["runtime"]["approved_interpreter"]["runtime_context_sha256"]
    with pytest.raises(LaunchEvidenceError, match="absent from the run diagnostic"):
        _build(diagnostic=diagnostic)


def test_owner_marker_must_name_the_same_plan() -> None:
    """The owner file is what ties the package on disk to THIS approved plan."""

    with pytest.raises(LaunchEvidenceError, match="owner_plan_sha256"):
        _build(owner={"plan_sha256": "0" * 64, "attempt_id": "002"})


def test_owner_marker_must_name_the_same_attempt() -> None:
    with pytest.raises(LaunchEvidenceError, match="owner_attempt_id"):
        _build(owner={"plan_sha256": plan_digest(_PLAN), "attempt_id": "003"})


def test_unverified_package_refuses_to_mint() -> None:
    diagnostic = _diagnostic(_PLAN)
    diagnostic["output"]["verification"]["status"] = "failed"
    with pytest.raises(LaunchEvidenceError, match="not 'success'"):
        _build(diagnostic=diagnostic)


def test_missing_package_file_refuses_to_mint() -> None:
    partial = _digests()
    partial.pop("SUCCESS.json")
    with pytest.raises(LaunchEvidenceError, match="package files absent"):
        _build(package_digests=partial)


def test_plan_drift_changes_the_expected_plan_digest() -> None:
    """A plan edited after approval no longer matches the diagnostic's binding."""

    edited = dict(_PLAN)
    edited["destination"] = dict(_PLAN["destination"], mount="/other")
    with pytest.raises(LaunchEvidenceError, match="plan_sha256"):
        _build(plan=edited, diagnostic=_diagnostic(_PLAN))


def test_host_local_device_is_not_bound() -> None:
    """st_dev differs per host for NFS; binding it would fail a correct run."""

    record = _build()
    assert "device" not in record["destination"]
    assert "host-local" in record["destination"]["device_note"]


def test_canonical_json_is_stable_and_sorted() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{\n  "a": 2,\n  "b": 1\n}\n'


def test_parse_json_artifact_names_the_bad_artifact() -> None:
    with pytest.raises(LaunchEvidenceError, match="run diagnostic is not valid JSON"):
        parse_json_artifact("{not json", label="run diagnostic")
    with pytest.raises(LaunchEvidenceError, match="must be a JSON object"):
        parse_json_artifact("[1, 2]", label="execution plan")
