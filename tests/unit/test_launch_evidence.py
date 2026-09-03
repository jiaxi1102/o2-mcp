"""Offline tests for the governed launch-evidence record.

These exercise the binding surface only: no SSH, no cluster, no MCP SDK. The
point of the record is that it refuses to exist when the launch chain disagrees,
so most of these assert a refusal.
"""

from __future__ import annotations

import base64
import hashlib
import json

import pytest

from o2mcp.launch_evidence import (
    EXPECTED_DIAGNOSTIC_STATUS,
    LAUNCH_EVIDENCE_SCHEMA,
    SACCT_RETENTION_DAYS,
    SYMLINK_SCAN_OK,
    LaunchEvidenceError,
    build_launch_evidence,
    canonical_json,
    claimed_job_id,
    evidence_content_digest,
    launch_evidence_digest,
    parse_checksum_manifest,
    parse_encoded_checksum_manifest,
    parse_encoded_json_artifact,
    parse_json_artifact,
    parse_scheduler_record,
    parse_sha256_lines,
    plan_digest,
    refuse_package_symlinks,
    required_package_files,
    verify_launch_evidence,
)

_PLAN = {
    "attempt_id": "002",
    "software": {"bundle": {"bundle_sha256": "b" * 64}},
    "runtime_wrapper": {"sha256": "a1" * 32},
    "interpreter": {"sha256": "b2" * 32, "closure_sha256": "c" * 64, "context_sha256": "c3" * 32},
    "destination": {"inode": 6411787343743799620, "mount": "/n/scratch", "expected_package": "/pkg/attempt-002"},
}


_MANIFEST = {"payloads/frame 001.ims": "1" * 64, "payloads/frame002.ims": "2" * 64}


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
            "runtime_wrapper_sha256": "a1" * 32,
            "approved_interpreter": {
                "approved_path": "/usr/bin/python3",
                "sha256": "b2" * 32,
                "runtime_context_sha256": "c3" * 32,
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
        "launch": {"loaded_library_closure": {"loaded_closure_sha256": "d4" * 32}},
        "output": {
            "package": "/pkg/attempt-002",
            "reopened_output_sha256": "e5" * 32,
            "verification": {"status": "success", "n_payloads": len(_MANIFEST)},
        },
    }


def _digests() -> dict[str, str]:
    return {name: "d" * 64 for name in required_package_files()}


def _manifest() -> dict[str, str]:
    """What the package's own SHA256SUMS claims its payloads hash to."""

    return dict(_MANIFEST)


def _scheduler_record(**overrides) -> dict[str, str]:
    """What Slurm accounting reports for the job the diagnostic claims."""

    record = {"job_id": "52085188", "state": "COMPLETED", "account": "tabin", "partition": "short"}
    record.update(overrides)
    return record


def _owner(plan: dict) -> dict:
    return {"plan_sha256": plan_digest(plan), "attempt_id": "002"}


def _build(**overrides):
    plan = overrides.pop("plan", _PLAN)
    diagnostic = overrides.pop("diagnostic", _diagnostic(plan))
    manifest = overrides.pop("checksum_manifest", _manifest())
    return build_launch_evidence(
        diagnostic=diagnostic,
        plan=plan,
        package_digests=overrides.pop("package_digests", _digests()),
        checksum_manifest=manifest,
        payload_digests=overrides.pop("payload_digests", dict(manifest)),
        owner=overrides.pop("owner", _owner(plan)),
        approval=overrides.pop("approval", {"approval_reference": "operator approved"}),
        stage=overrides.pop("stage", "platform-canary"),
        read_back_package_path=overrides.pop("read_back_package_path", "/pkg/attempt-002"),
        resolved_package_path=overrides.pop("resolved_package_path", "/pkg/attempt-002"),
        scheduler_record=overrides.pop("scheduler_record", _scheduler_record()),
    )


def test_intact_chain_mints_a_record() -> None:
    record = _build()
    assert record["schema"] == LAUNCH_EVIDENCE_SCHEMA
    assert record["binding_check"]["all_links_agree"] is True
    # The count is what was actually checked, payload by payload, not a constant.
    assert record["binding_check"]["checked"] > len(_manifest())
    assert record["submission"]["job_id"] == "52085188"
    assert record["unbound_run_reported"]["loaded_closure_sha256"] == "d4" * 32
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
    with pytest.raises(LaunchEvidenceError, match="run diagnostic is malformed JSON"):
        parse_json_artifact("{not json", label="run diagnostic")
    with pytest.raises(LaunchEvidenceError, match="must be a JSON object"):
        parse_json_artifact("[1, 2]", label="execution plan")


def test_a_duplicate_member_name_refuses_rather_than_picking_one() -> None:
    """`json.loads` keeps the last value silently, leaving the bytes ambiguous.

    An artifact carrying `attempt_id` twice would be bound on one reading while
    a reviewer opening the same file could reasonably take the other, and the
    record does not retain the bytes to settle it afterwards.
    """

    with pytest.raises(LaunchEvidenceError, match="duplicate key"):
        parse_json_artifact('{"attempt_id": "002", "attempt_id": "003"}', label="execution plan")
    with pytest.raises(LaunchEvidenceError, match="duplicate key"):
        parse_json_artifact('{"destination": {"expected_package": "/a", "expected_package": "/b"}}', label="plan")


def test_a_non_finite_number_refuses_too() -> None:
    """Digesting a record containing NaN would fail later; refuse at the source."""

    with pytest.raises(LaunchEvidenceError, match="malformed JSON"):
        parse_json_artifact('{"n_payloads": NaN}', label="run diagnostic")


# --- the read-back directory must be the package the plan approved ------------
def test_read_back_directory_must_be_the_approved_package() -> None:
    """A copied owner marker must not let a caller substitute another directory.

    The caller picks which directory is hashed. If that is not the package the
    plan approved, the record would name package A under ``destination`` while
    its file digests described package B.
    """

    with pytest.raises(LaunchEvidenceError, match="package_read_back_path"):
        _build(read_back_package_path="/pkg/attempt-002-copy")


def test_read_back_directory_tolerates_a_cosmetic_trailing_slash() -> None:
    record = _build(read_back_package_path="/pkg/attempt-002/")
    assert record["destination"]["read_back_package_path"] == "/pkg/attempt-002/"


def test_read_back_directory_must_be_a_real_path() -> None:
    with pytest.raises(LaunchEvidenceError, match="package_read_back_path"):
        _build(read_back_package_path="   ")


# --- the package is reverified, not taken on the run's word -------------------
def test_reported_success_does_not_substitute_for_reverifying_payloads() -> None:
    """The run says the package verified; a payload that no longer matches wins.

    This is the whole point of minting outside the executed process: a payload
    corrupted after the diagnostic was written, or a run that reported success
    falsely, must not still produce a record.
    """

    drifted = dict(_manifest())
    drifted["payloads/frame002.ims"] = "0" * 64
    with pytest.raises(LaunchEvidenceError, match="SHA256SUMS=.* on_disk="):
        _build(payload_digests=drifted)


def test_payload_listed_but_never_hashed_refuses_to_mint() -> None:
    partial = dict(_manifest())
    partial.pop("payloads/frame002.ims")
    with pytest.raises(LaunchEvidenceError, match="listed in SHA256SUMS but was not hashed"):
        _build(payload_digests=partial)


def test_empty_manifest_refuses_to_mint() -> None:
    with pytest.raises(LaunchEvidenceError, match="nothing could be reverified"):
        _build(checksum_manifest={}, payload_digests={})


def test_metadata_file_must_hash_the_same_in_both_reads() -> None:
    """A file covered by both reads changing between them ends the mint."""

    manifest = dict(_manifest())
    manifest["SUCCESS.json"] = "e" * 64
    payloads = dict(manifest)
    with pytest.raises(LaunchEvidenceError, match="changed between the two reads"):
        _build(checksum_manifest=manifest, payload_digests=payloads)


def test_reverification_is_recorded_in_the_minted_record() -> None:
    record = _build()
    reverification = record["verified_package"]["payload_reverification"]
    assert reverification["n_payloads_reverified"] == len(_manifest())
    assert reverification["manifest_sha256"] == "d" * 64


# --- sha256sum parsing --------------------------------------------------------
def test_filenames_with_spaces_survive_parsing() -> None:
    """``line.split()`` would drop every one of these and refuse a valid package."""

    text = "{}  /pkg/attempt 002/a payload.ims\n{} */pkg/attempt 002/b.ims\n".format("1" * 64, "2" * 64)
    assert parse_sha256_lines(text, label="output") == {
        "/pkg/attempt 002/a payload.ims": "1" * 64,
        "/pkg/attempt 002/b.ims": "2" * 64,
    }


def test_a_single_separator_manifest_still_parses() -> None:
    """A SHA256SUMS not written by coreutils must not be refused for its spacing."""

    assert parse_checksum_manifest("{} a payload.ims\n".format("1" * 64)) == {"a payload.ims": "1" * 64}


def test_a_gnu_name_that_starts_with_a_space_keeps_it() -> None:
    assert parse_sha256_lines("{}   leading.ims".format("1" * 64), label="output") == {" leading.ims": "1" * 64}


def test_escaped_filename_is_refused_rather_than_guessed_at() -> None:
    with pytest.raises(LaunchEvidenceError, match="escaped filename"):
        parse_sha256_lines("\\{}  /pkg/new\\nline.ims\n".format("1" * 64), label="output")


def test_a_line_that_is_not_a_digest_entry_is_refused_not_skipped() -> None:
    with pytest.raises(LaunchEvidenceError, match="not a sha256 entry"):
        parse_sha256_lines("{}  ok.ims\nsha256sum: nope\n".format("1" * 64), label="output")


def test_a_name_listed_twice_with_different_digests_is_refused() -> None:
    text = "{}  a.ims\n{}  a.ims\n".format("1" * 64, "2" * 64)
    with pytest.raises(LaunchEvidenceError, match="twice with different digests"):
        parse_sha256_lines(text, label="output")


@pytest.mark.parametrize("name", ["/etc/passwd", "../../outside.ims", "sub/../../outside.ims"])
def test_manifest_cannot_steer_the_reverification_outside_the_package(name) -> None:
    with pytest.raises(LaunchEvidenceError, match="not a payload path inside the package"):
        parse_checksum_manifest("{}  {}\n".format("1" * 64, name))


def test_manifest_normalizes_a_leading_dot_slash() -> None:
    assert parse_checksum_manifest("{}  ./a payload.ims\n".format("1" * 64)) == {"a payload.ims": "1" * 64}


# --- the approval is bound to the record it approved ---------------------------
def test_content_digest_ignores_the_approval_that_was_recorded_against_it() -> None:
    """The ledger stores this digest, so it cannot depend on the approval itself."""

    unapproved = _build(approval={})
    digest = evidence_content_digest(unapproved)
    approved = _build(approval={"approval_reference": "operator approved", "evidence_sha256": digest})
    assert evidence_content_digest(approved) == digest
    assert launch_evidence_digest(approved) != launch_evidence_digest(unapproved)


def test_content_digest_moves_when_any_bound_field_moves() -> None:
    baseline = evidence_content_digest(_build())
    assert evidence_content_digest(_build(stage="acquisition")) != baseline
    assert evidence_content_digest(_build(package_digests={**_digests(), "SUCCESS.json": "0" * 64})) != baseline


def test_an_approval_recorded_against_another_record_refuses_to_mint() -> None:
    """The approval object is only worth what the ledger recorded it against."""

    with pytest.raises(LaunchEvidenceError, match="recorded against a different record"):
        _build(approval={"approval_reference": "operator approved", "evidence_sha256": "0" * 64})


def test_an_approval_carrying_the_right_digest_mints() -> None:
    digest = evidence_content_digest(_build(approval={}))
    record = _build(approval={"approval_reference": "operator approved", "evidence_sha256": digest})
    assert record["operator_approval"]["evidence_sha256"] == digest


# --- the manifest parsed must be the manifest whose digest was recorded --------
def _encoded_manifest() -> tuple[str, str]:
    raw = "{}  a payload.ims\n".format("1" * 64).encode()
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()


def test_encoded_manifest_parses_when_its_digest_matches() -> None:
    encoded, digest = _encoded_manifest()
    assert parse_encoded_checksum_manifest(encoded, expected_sha256=digest) == {"a payload.ims": "1" * 64}


def test_a_manifest_replaced_between_the_two_reads_refuses_to_mint() -> None:
    """The bytes that chose the payloads and the digest recorded must be one file."""

    encoded, _ = _encoded_manifest()
    with pytest.raises(LaunchEvidenceError, match="changed while it was being read"):
        parse_encoded_checksum_manifest(encoded, expected_sha256="0" * 64)


def test_a_manifest_with_no_recorded_digest_refuses_to_mint() -> None:
    encoded, _ = _encoded_manifest()
    with pytest.raises(LaunchEvidenceError, match="carried no digest"):
        parse_encoded_checksum_manifest(encoded, expected_sha256=None)


def test_a_manifest_that_is_not_base64_is_named_as_such() -> None:
    with pytest.raises(LaunchEvidenceError, match="valid base64"):
        parse_encoded_checksum_manifest("not base64 !!", expected_sha256="0" * 64)


def test_encoded_json_artifact_must_hash_to_the_digest_recorded_for_it() -> None:
    raw = b'{"plan_sha256": "abc"}'
    encoded = base64.b64encode(raw).decode()
    digest = hashlib.sha256(raw).hexdigest()
    assert parse_encoded_json_artifact(encoded, expected_sha256=digest, label="publication owner") == {
        "plan_sha256": "abc"
    }
    with pytest.raises(LaunchEvidenceError, match="publication owner changed while it was being read"):
        parse_encoded_json_artifact(encoded, expected_sha256="0" * 64, label="publication owner")


# --- the record must name the job it attests ----------------------------------
@pytest.mark.parametrize("job_id", [None, "", "   ", [], {}, True])
def test_a_record_that_binds_no_submitted_job_refuses_to_mint(job_id) -> None:
    diagnostic = _diagnostic(_PLAN)
    diagnostic["slurm"]["scheduler"]["job_id"] = job_id
    with pytest.raises(LaunchEvidenceError, match="would bind no submitted job"):
        _build(diagnostic=diagnostic)


def test_an_integer_job_id_is_accepted() -> None:
    diagnostic = _diagnostic(_PLAN)
    diagnostic["slurm"]["scheduler"]["job_id"] = 52085188
    assert _build(diagnostic=diagnostic)["submission"]["job_id"] == 52085188


# --- the manifest must cover every payload the run counted --------------------
def test_a_manifest_shortened_after_the_run_refuses_to_mint() -> None:
    """Deleting payloads from disk and from SHA256SUMS together is caught here.

    Every remaining entry rehashes correctly, so nothing else in the chain
    notices; only the count the run recorded does. On the real canary package
    SHA256SUMS holds exactly the 9 payloads the diagnostic counts -- it cannot
    list itself, and SUCCESS.json is written after it.
    """

    shortened = {"payloads/frame002.ims": "2" * 64}
    with pytest.raises(LaunchEvidenceError, match="lists 1 payloads but the run verified 2"):
        _build(checksum_manifest=shortened, payload_digests=dict(shortened))


def test_a_manifest_with_extra_entries_refuses_to_mint() -> None:
    padded = {**_manifest(), "payloads/frame003.ims": "3" * 64}
    with pytest.raises(LaunchEvidenceError, match="lists 3 payloads but the run verified 2"):
        _build(checksum_manifest=padded, payload_digests=dict(padded))


@pytest.mark.parametrize("count", [None, "2", 2.0, True])
def test_a_diagnostic_without_a_usable_payload_count_refuses_to_mint(count) -> None:
    diagnostic = _diagnostic(_PLAN)
    diagnostic["output"]["verification"]["n_payloads"] = count
    with pytest.raises(LaunchEvidenceError, match="nothing pins the manifest length"):
        _build(diagnostic=diagnostic)


# --- the run's own outcome must say it finished -------------------------------
def test_a_diagnostic_that_did_not_succeed_refuses_to_mint() -> None:
    """A failed run must not yield a continuation credential.

    The canary signals failure by raising, so a diagnostic only exists for a run
    that reached the end -- but a status that is not the expected one still must
    not be copied into an otherwise successful record.
    """

    diagnostic = _diagnostic(_PLAN)
    diagnostic["status"] = "diagnostic_failed"
    with pytest.raises(LaunchEvidenceError, match="does not say it finished"):
        _build(diagnostic=diagnostic)


def test_an_absent_diagnostic_status_refuses_to_mint() -> None:
    diagnostic = _diagnostic(_PLAN)
    del diagnostic["status"]
    with pytest.raises(LaunchEvidenceError, match="does not say it finished"):
        _build(diagnostic=diagnostic)


def test_an_unauthorized_continuation_is_not_a_failure() -> None:
    """`continuation_authorized: false` is the expected value and must still mint.

    It is the run asserting it cannot authenticate its own launch, which is the
    reason this record exists. Treating it as failure would make the record
    impossible to mint for exactly the runs that need it.
    """

    diagnostic = _diagnostic(_PLAN)
    assert diagnostic["continuation_authorized"] is False
    record = _build(diagnostic=diagnostic)
    assert record["unbound_run_reported"]["continuation_authorized"] is False
    assert record["run_diagnostic"]["status"] == EXPECTED_DIAGNOSTIC_STATUS


# --- links must not carry the read outside the approved package ---------------
def test_a_package_pathname_that_resolves_elsewhere_refuses_to_mint() -> None:
    """Spelling the approved package is not the same as being it.

    `cat` and `sha256sum` follow links, so a pathname that passes the lexical
    comparison can still open a substituted directory holding a copied owner
    marker.
    """

    with pytest.raises(LaunchEvidenceError, match="resolved_package_path"):
        _build(resolved_package_path="/pkg/somewhere-else")


def test_any_symlink_in_the_package_refuses_to_mint() -> None:
    """A published package contains none, so one appearing is grounds to refuse.

    This replaces resolving each payload and checking containment. That was two
    independent pathname resolutions, and a run that toggled a link between them
    could pass the check while the digest came from outside; removing the object
    being raced is simpler and strictly stronger, and it matches an invariant the
    publisher already enforces by hard-linking and by rejecting symlinks in its
    own verifier.
    """

    with pytest.raises(LaunchEvidenceError, match="contains symlinks"):
        refuse_package_symlinks(
            f"/pkg/attempt-002/payloads/frame002.ims\n{SYMLINK_SCAN_OK}\n", package_path="/pkg/attempt-002"
        )


def test_a_package_with_no_symlinks_is_accepted() -> None:
    refuse_package_symlinks(SYMLINK_SCAN_OK, package_path="/pkg/attempt-002")
    refuse_package_symlinks(f"\n  \n{SYMLINK_SCAN_OK}\n", package_path="/pkg/attempt-002")


def test_a_scan_that_did_not_complete_is_not_an_empty_package() -> None:
    """`find` exits non-zero when it cannot enumerate, and says nothing either way.

    A package that is searchable but not readable still lets its known files be
    opened, so a publisher could chmod it to execute-only and hide a payload
    symlink behind a scan that reported nothing because it could not look. The
    scan therefore has to announce its own success.
    """

    with pytest.raises(LaunchEvidenceError, match="did not complete"):
        refuse_package_symlinks("", package_path="/pkg/attempt-002")
    with pytest.raises(LaunchEvidenceError, match="did not complete"):
        refuse_package_symlinks("/pkg/attempt-002/link", package_path="/pkg/attempt-002")


def test_the_refusal_names_the_offending_links_without_dumping_all_of_them() -> None:
    listing = "\n".join([*(f"/pkg/attempt-002/link{index}" for index in range(9)), SYMLINK_SCAN_OK])
    with pytest.raises(LaunchEvidenceError, match="and 4 more") as caught:
        refuse_package_symlinks(listing, package_path="/pkg/attempt-002")
    assert "/pkg/attempt-002/link0" in str(caught.value)
    assert "/pkg/attempt-002/link8" not in str(caught.value)


def test_the_resolved_package_is_recorded() -> None:
    record = _build()
    assert record["destination"]["resolved_package_path"] == "/pkg/attempt-002"


# --- the job identity comes from the scheduler, not from the run --------------
def test_a_fabricated_job_id_refuses_to_mint() -> None:
    """The last field taken on the run's word is now anchored outside it.

    A compromised run can put any plausible id in the diagnostic, so the record
    binds what Slurm accounting reports rather than what the run claims.
    """

    with pytest.raises(LaunchEvidenceError, match="scheduler job_id"):
        _build(scheduler_record=_scheduler_record(job_id="99999999"))


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [("account", "someone-else", "scheduler account"), ("partition", "priority", "scheduler partition")],
)
def test_a_job_belonging_to_another_allocation_refuses_to_mint(field, value, expected) -> None:
    with pytest.raises(LaunchEvidenceError, match=expected):
        _build(scheduler_record=_scheduler_record(**{field: value}))


def test_the_accounting_row_is_recorded(_=None) -> None:
    record = _build()
    assert record["submission"]["scheduler_accounting"] == _scheduler_record()


def test_a_job_state_that_is_not_completed_still_mints() -> None:
    """A mint can legitimately race the job's final accounting transition.

    The state is recorded, but gating on it would refuse correct runs for their
    timing rather than for their content.
    """

    record = _build(scheduler_record=_scheduler_record(state="COMPLETING"))
    assert record["submission"]["scheduler_accounting"]["state"] == "COMPLETING"


def test_an_aged_out_job_refuses_rather_than_falling_back_to_the_diagnostic() -> None:
    """Accounting purges after a year; silence must not become trust."""

    with pytest.raises(LaunchEvidenceError, match=f"purged after {SACCT_RETENTION_DAYS} days"):
        parse_scheduler_record("", job_id="52085188")


def test_ambiguous_accounting_refuses_to_mint() -> None:
    two = "52085188|COMPLETED|tabin|short\n52085188|FAILED|tabin|short\n"
    with pytest.raises(LaunchEvidenceError, match="returned 2 allocations"):
        parse_scheduler_record(two, job_id="52085188")


def test_accounting_answering_about_another_job_refuses_to_mint() -> None:
    with pytest.raises(LaunchEvidenceError, match="answered for job"):
        parse_scheduler_record("99999999|COMPLETED|tabin|short\n", job_id="52085188")


def test_a_malformed_accounting_row_refuses_to_mint() -> None:
    with pytest.raises(LaunchEvidenceError, match="not the four fields"):
        parse_scheduler_record("52085188|COMPLETED\n", job_id="52085188")


def test_a_well_formed_accounting_row_parses() -> None:
    assert parse_scheduler_record("52085188|COMPLETED|tabin|short\n", job_id="52085188") == _scheduler_record()


# --- the claimed job id must be a job id before it reaches a shell ------------
@pytest.mark.parametrize("job_id", ["52085188", "52085188_4", 52085188])
def test_a_plausible_job_id_is_extracted(job_id) -> None:
    diagnostic = _diagnostic(_PLAN)
    diagnostic["slurm"]["scheduler"]["job_id"] = job_id
    assert claimed_job_id(diagnostic) == str(job_id)


@pytest.mark.parametrize("job_id", ["; rm -rf /", "52085188; sacct", "abc", "", "   ", None, True, []])
def test_a_job_id_that_is_not_one_is_refused_before_any_command(job_id) -> None:
    diagnostic = _diagnostic(_PLAN)
    diagnostic["slurm"]["scheduler"]["job_id"] = job_id
    with pytest.raises(LaunchEvidenceError):
        claimed_job_id(diagnostic)


# --- a job the scheduler says failed cannot produce a record ------------------
@pytest.mark.parametrize(
    "state",
    ["FAILED", "CANCELLED", "CANCELLED by 12345", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"],
)
def test_a_terminally_failed_job_refuses_to_mint(state) -> None:
    """A record attests a finished stage, so Slurm saying otherwise ends it.

    The diagnostic can be written before a later failure, or fabricated outright
    under this threat model, so the scheduler's verdict is the one that counts.
    """

    with pytest.raises(LaunchEvidenceError, match="did not finish"):
        parse_scheduler_record(f"52085188|{state}|tabin|short\n", job_id="52085188")


@pytest.mark.parametrize("state", ["COMPLETED", "RUNNING", "COMPLETING"])
def test_a_state_reachable_after_the_job_ran_still_mints(state) -> None:
    """The accounting race the record documents must stay allowed."""

    assert parse_scheduler_record(f"52085188|{state}|tabin|short\n", job_id="52085188")["state"] == state


@pytest.mark.parametrize("state", ["PENDING", "REQUEUED", "REQUEUE_HOLD", "CONFIGURING", "SUSPENDED", "RESIZING"])
def test_a_job_that_has_not_run_refuses_to_mint(state) -> None:
    """The allowance is for accounting settling, not for jobs that never started.

    runorg classes these as active, but a queued job cannot have produced the
    diagnostic being attested -- and allowing them would let a compromised
    diagnostic name any queued job in the same account and partition.
    """

    with pytest.raises(LaunchEvidenceError, match="did not finish"):
        parse_scheduler_record(f"52085188|{state}|tabin|short\n", job_id="52085188")


def test_an_unrecognised_state_refuses_rather_than_being_assumed_benign() -> None:
    with pytest.raises(LaunchEvidenceError, match="did not finish"):
        parse_scheduler_record("52085188|SOME_FUTURE_STATE|tabin|short\n", job_id="52085188")


# --- the approval fields are verified against the ledger, not just the digest --
def _ledger_entry(record, **overrides):
    """The durable mint entry the policy store would have written for a record."""

    approval = record["operator_approval"]
    entry = {
        "at": approval["approved_at"],
        "client_id": approval["client_id"],
        "approval_reference": approval["approval_reference"],
        "stage": record["stage"],
        "job_id": str(record["submission"]["job_id"]),
        "package": record["destination"]["package"],
        "evidence_sha256": approval["evidence_sha256"],
        "plan_sha256": approval["plan_sha256"],
        "policy_revision": approval["policy_revision"],
        "policy_generation": approval["policy_generation"],
    }
    entry.update(overrides)
    return entry


def _approved_record(**overrides):
    """A record carrying a complete approval, as the server builds one."""

    approval = {
        "approval_reference": "operator approved canary 002",
        "plan_sha256": plan_digest(_PLAN),
        "policy_revision": 12,
        "policy_generation": "00000000-0000-4000-8000-000000000001",
        "client_id": "1234-abcd",
        "approved_at": 1000.0,
    }
    approval["evidence_sha256"] = evidence_content_digest(_build(approval={}))
    approval.update(overrides)
    return _build(approval=approval)


def test_an_untouched_record_verifies_against_its_ledger_entry() -> None:
    record = _approved_record()
    verify_launch_evidence(record, _ledger_entry(record))


def test_edited_content_fails_verification() -> None:
    record = _approved_record()
    entry = _ledger_entry(record)
    record["runtime_identities"]["interpreter_sha256"] = "0" * 64
    with pytest.raises(LaunchEvidenceError, match="the ledger entry approved"):
        verify_launch_evidence(record, entry)


@pytest.mark.parametrize(
    "field",
    ["approval_reference", "client_id", "policy_revision", "policy_generation", "approved_at"],
)
def test_a_rewritten_approval_field_fails_verification(field) -> None:
    """The content digest cannot cover these, so the ledger entry must.

    They do not exist when the digest is taken, so a holder of a valid record
    could otherwise rewrite who approved it, under what revision, and when --
    recompute the record's own digest, and still match the ledger.
    """

    record = _approved_record()
    entry = _ledger_entry(record)
    record["operator_approval"][field] = "tampered" if isinstance(record["operator_approval"][field], str) else 999
    with pytest.raises(LaunchEvidenceError, match=field):
        verify_launch_evidence(record, entry)


def test_a_record_reassigned_to_another_stage_or_package_fails_verification() -> None:
    record = _approved_record()
    with pytest.raises(LaunchEvidenceError, match="stage"):
        verify_launch_evidence(record, _ledger_entry(record, stage="acquisition"))
    with pytest.raises(LaunchEvidenceError, match="package"):
        verify_launch_evidence(record, _ledger_entry(record, package="/pkg/other"))


def test_a_record_without_an_approval_cannot_be_verified() -> None:
    record = _build(approval={})
    with pytest.raises(LaunchEvidenceError, match="no operator approval"):
        verify_launch_evidence(record, _ledger_entry(_approved_record()))


# --- unbound values must not sit among the authenticated ones -----------------
def test_run_reported_values_are_separated_from_the_bound_ones() -> None:
    """Nothing checks any of these, so none of them is a claim this record makes.

    A compromised run can put any value in each. Keeping them among the bound
    fields made the record look like it attested a loaded library set, an
    allocation, a filesystem and a reopened output it had never checked.
    """

    record = _build()
    unbound = record["unbound_run_reported"]
    assert set(unbound) == {
        "note",
        "interpreter_path",
        "loaded_closure_sha256",
        "allocated_hostnames",
        "mount_source",
        "reopened_output_sha256",
        "continuation_authorized",
        "diagnostic_schema_version",
    }
    assert "not part of this record's authenticated claims" in unbound["note"]
    assert unbound["loaded_closure_sha256"] == "d4" * 32
    assert unbound["mount_source"] == "server:/scratch"
    assert unbound["reopened_output_sha256"] == "e5" * 32
    assert unbound["allocated_hostnames"] == ["compute-b-16-192"]
    # And none of them is left behind in a section that reads as authenticated.
    assert "mount_source" not in record["destination"]
    assert "reopened_output_sha256" not in record["verified_package"]
    assert "allocated_hostnames" not in record["submission"]
    assert "loaded_closure_sha256" not in record["runtime_identities"]


def test_every_authenticated_field_is_bound_observed_or_the_servers_own() -> None:
    """A field added to an authenticated section without a check must fail here.

    This is the guard that keeps the previous test true. Every leaf outside
    `unbound_run_reported` is listed with what makes it trustworthy; adding one
    without deciding forces this assertion rather than quietly extending what
    the record appears to attest.
    """

    accounted = {
        # bound to the approved plan by _BINDINGS
        ("approved_plan", "attempt_id"),
        ("runtime_identities", "source_bundle_sha256"),
        ("runtime_identities", "runtime_wrapper_sha256"),
        ("runtime_identities", "interpreter_sha256"),
        ("runtime_identities", "interpreter_closure_sha256"),
        ("runtime_identities", "interpreter_replay_context_sha256"),
        ("destination", "package"),
        ("destination", "inode"),
        ("destination", "mount_point"),
        # confirmed against Slurm accounting
        ("submission", "job_id"),
        ("submission", "account"),
        ("submission", "partition"),
        # required to hold a specific value
        ("verified_package", "verification_status"),
        ("run_diagnostic", "status"),
        # bound to the manifest the server read and rehashed
        ("verified_package", "n_payloads"),
        # computed or observed by the server itself
        ("schema",),
        ("stage",),
        ("approved_plan", "sha256"),
        ("destination", "read_back_package_path"),
        ("destination", "resolved_package_path"),
        ("destination", "device_note"),
        ("binding_check", "all_links_agree"),
        ("binding_check", "checked"),
    }
    record = _build()
    leaves = set()
    for key, value in record.items():
        if key in {"unbound_run_reported", "operator_approval"}:
            continue
        # These subtrees are wholly server-computed; their contents are not
        # individually enumerated here.
        if key == "submission":
            leaves.update((key, field) for field in value if field != "scheduler_accounting")
        elif key == "verified_package":
            leaves.update((key, field) for field in value if field not in {"file_digests", "payload_reverification"})
        elif isinstance(value, dict):
            leaves.update((key, field) for field in value)
        else:
            leaves.add((key,))
    assert leaves == accounted


def test_an_extra_approval_field_fails_verification() -> None:
    """The digest blanks this object, so an added claim would ride along free.

    Comparing only the known keys let a holder of a valid record append an
    unauthenticated claim, recompute the whole-record digest, and still verify.
    """

    record = _approved_record()
    entry = _ledger_entry(record)
    record["operator_approval"]["authorized_by"] = "Alice"
    with pytest.raises(LaunchEvidenceError, match="not recorded in the ledger"):
        verify_launch_evidence(record, entry)


def test_a_missing_approval_field_fails_verification() -> None:
    record = _approved_record()
    entry = _ledger_entry(record)
    del record["operator_approval"]["client_id"]
    with pytest.raises(LaunchEvidenceError, match="client_id"):
        verify_launch_evidence(record, entry)


# --- agreeing on nonsense is not a binding ------------------------------------
@pytest.mark.parametrize(
    ("path", "plan_path", "value"),
    [
        (("runtime", "source_bundle", "bundle_sha256"), ("software", "bundle", "bundle_sha256"), ""),
        (("runtime", "runtime_wrapper_sha256"), ("runtime_wrapper", "sha256"), {}),
        (("runtime", "approved_interpreter", "sha256"), ("interpreter", "sha256"), "not a digest"),
        (("runtime", "approved_interpreter", "sha256"), ("interpreter", "sha256"), "Z" * 64),
        (("destination_binding", "inode"), ("destination", "inode"), 0),
        (("destination_binding", "inode"), ("destination", "inode"), "42"),
        (("destination_binding", "inode"), ("destination", "inode"), True),
    ],
)
def test_a_bound_field_agreeing_on_a_malformed_value_refuses(path, plan_path, value) -> None:
    """Both sides matching is not enough if neither is the kind of thing meant.

    A falsified plan and a matching diagnostic could otherwise settle on an
    empty string or an object for an interpreter digest, pass every check, and
    produce a record whose identities identify nothing.
    """

    plan = json.loads(json.dumps(_PLAN))
    target = plan
    for key in plan_path[:-1]:
        target = target[key]
    target[plan_path[-1]] = value
    diagnostic = _diagnostic(plan)
    target = diagnostic
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(LaunchEvidenceError, match="is not a well-formed"):
        _build(plan=plan, diagnostic=diagnostic)


def test_a_mount_point_that_is_not_a_path_refuses() -> None:
    plan = json.loads(json.dumps(_PLAN))
    plan["destination"]["mount"] = "scratch"
    diagnostic = _diagnostic(plan)
    diagnostic["destination_binding"]["linux_mountinfo"]["mount_point"] = "scratch"
    with pytest.raises(LaunchEvidenceError, match="is not a well-formed path"):
        _build(plan=plan, diagnostic=diagnostic)


def test_an_uppercase_digest_is_refused_so_comparisons_stay_exact() -> None:
    plan = json.loads(json.dumps(_PLAN))
    plan["interpreter"]["sha256"] = "B2" * 32
    diagnostic = _diagnostic(plan)
    diagnostic["runtime"]["approved_interpreter"]["sha256"] = "B2" * 32
    with pytest.raises(LaunchEvidenceError, match="is not a well-formed sha256"):
        _build(plan=plan, diagnostic=diagnostic)
