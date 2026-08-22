"""Adversarial tests for strict control files and lifecycle coordination."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from o2mcp.runorg.execution_backend import O2ExecutionBackend
from o2mcp.runorg.execution_models import RegistryUpdate, SubmissionRecord
from o2mcp.runorg.lifecycle_coordination import claim_name, coordination_root
from o2mcp.runorg.registry_outbox import decode_registry_update
from o2mcp.runorg.registry_sync import synchronize_execution_transaction
from o2mcp.runorg.runs import RunManifest
from o2mcp.runorg.strict_json import strict_json_object
from o2mcp.runorg.transition_coordinator import begin_transition, rollback_transition


class LocalConnection:
    """Execute the dependency-free O2 helper programs against a temporary tree."""

    def run(self, command, *, timeout, input_text=None, **_kwargs):
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            input=input_text,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return SimpleNamespace(
            ok=result.returncode == 0,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


def test_strict_json_parity_rejects_duplicates_nonfinite_and_wrong_top_level() -> None:
    """Every control reader starts from the same ambiguity-rejecting decoder."""

    malformed = ('{"a":1,"a":2}', '{"a":NaN}', "[]", "1")
    for text in malformed:
        with pytest.raises(ValueError):
            strict_json_object(text, "parity corpus")

    duplicate_manifest = '{"run_id":"a","run_id":"b"}'
    with pytest.raises(ValueError, match="duplicate key"):
        RunManifest.from_json(duplicate_manifest)
    duplicate_outbox = (
        '{"attempt":1,"attempt":2,"execution_status":"ACTIVE","job_ids":[],'
        '"plan_sha256":"' + "a" * 64 + '","stage_id":"s","stage_status":"SUBMITTED"}'
    )
    with pytest.raises(ValueError, match="duplicate key"):
        decode_registry_update(duplicate_outbox)


def test_control_record_exact_types_reject_bool_float_and_string_coercion() -> None:
    """Wire values cannot change meaning through Python int/str coercion."""

    payload = {
        "attempt": 1,
        "comment": f"o2plan:v1:{'a' * 64}:stage:a001",
        "dependency_job_ids": [],
        "dependency_mode": "afterok",
        "job_id": "123",
        "plan_sha256": "a" * 64,
        "recovered": True,
        "schema_version": 1,
        "stage_id": "stage",
        "task_ids": ["stage"],
        "task_indices": [],
    }
    assert SubmissionRecord.from_dict(payload).job_id == "123"
    for field, invalid in (("attempt", True), ("attempt", 1.0), ("job_id", 123), ("recovered", 1)):
        changed = {**payload, field: invalid}
        with pytest.raises(ValueError):
            SubmissionRecord.from_dict(changed)

    manifest = {
        "run_id": "RUN_20260822T010203Z_strict__types",
        "campaign": "strict",
        "pipeline": "test",
        "created_utc": "20260822T010203Z",
        "datasets": ["d"],
        "schema_version": True,
    }
    with pytest.raises(ValueError, match="schema_version"):
        RunManifest.from_json(json.dumps(manifest))

    claim_id = "a" * 64 + "-" + "b" * 64
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        RegistryUpdate("c" * 64, "stage", "SUBMITTED", "SUBMITTED", ("1",), 1, (claim_id, claim_id))


def test_remote_control_fs_rejects_symlink_ancestors_and_nonregular_leaves(tmp_path) -> None:
    """No-follow/nonblocking checks prevent redirection and FIFO hangs."""

    backend = O2ExecutionBackend(LocalConnection())
    real = tmp_path / "real"
    real.mkdir()
    ancestor = tmp_path / "alias"
    ancestor.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="safe control-file read"):
        backend.read_text(str(ancestor / "record.json"))
    with pytest.raises(RuntimeError, match="write failed"):
        backend.write_immutable_text(str(ancestor / "record.json"), "x")

    fifo = real / "record.fifo"
    os.mkfifo(fifo)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="safe control-file read"):
        backend.read_text(str(fifo))
    assert time.monotonic() - started < 2

    target = real / "target"
    target.write_text("other")
    leaf = real / "leaf"
    leaf.symlink_to(target)
    with pytest.raises(RuntimeError, match="write failed"):
        backend.write_immutable_text(str(leaf), "payload")


def test_remote_control_fs_immutable_and_cas_round_trip(tmp_path) -> None:
    """Anchored helpers preserve exact immutable replay and CAS semantics."""

    backend = O2ExecutionBackend(LocalConnection())
    path = str(tmp_path / "control" / "state.json")
    assert backend.write_immutable_text(path, "one")
    assert not backend.write_immutable_text(path, "one")
    assert backend.read_text(path) == "one"
    assert backend.compare_and_swap_text(path, "one", "two")
    assert not backend.compare_and_swap_text(path, "one", None)
    assert backend.compare_and_swap_text(path, "two", None)
    assert backend.read_text(path) is None


def test_fenced_control_write_cannot_recreate_quarantined_run_root(tmp_path) -> None:
    """A transition marker wins atomically over every later evidence mutation."""

    connection = LocalConnection()
    backend = O2ExecutionBackend(connection)
    run_root = str(tmp_path / "campaign" / "RUN_20260822T010203Z_fenced__write")
    os.makedirs(os.path.join(run_root, "receipts", "execution"))
    target = os.path.join(run_root, "receipts", "execution", "late.json")

    begin_transition(connection, run_root, "a" * 64)
    quarantine = run_root + ".quarantine"
    os.rename(run_root, quarantine)

    with pytest.raises(RuntimeError, match="write failed"):
        backend.write_immutable_text_fenced(run_root, target, "late")

    assert not os.path.exists(run_root)
    assert not os.path.exists(os.path.join(quarantine, "receipts", "execution", "late.json"))


def test_fenced_registry_sync_cannot_recreate_quarantined_run_root(tmp_path) -> None:
    """run.json/registry synchronization shares the same transition fence."""

    connection = LocalConnection()
    run_id = "RUN_20260822T010203Z_fenced__registry"
    run_root = str(tmp_path / "campaign" / run_id)
    os.makedirs(os.path.join(run_root, "receipts", "execution"))
    manifest = RunManifest(
        run_id=run_id,
        campaign="fenced",
        pipeline="canary",
        created_utc="20260822T010203Z",
        datasets=["dataset"],
    )
    with open(os.path.join(run_root, "run.json"), "w", encoding="utf-8") as handle:
        handle.write(manifest.to_json())

    begin_transition(connection, run_root, "b" * 64)
    quarantine = run_root + ".quarantine"
    os.rename(run_root, quarantine)
    result = synchronize_execution_transaction(
        connection,
        run_dir=run_root,
        registry_path=str(tmp_path / "registry.jsonl"),
        current_manifest_text=manifest.to_json(),
        merged_manifest=manifest,
    )

    assert result["ok"] is False
    assert result["error"] == "lifecycle_transition_in_progress"
    assert not os.path.exists(run_root)
    assert not (tmp_path / "registry.jsonl").exists()


def test_submission_claim_and_transition_marker_have_one_atomic_winner(tmp_path) -> None:
    """No scheduler boundary can be accepted after transition marking wins."""

    connection = LocalConnection()
    backend = O2ExecutionBackend(connection)
    run_root = str(tmp_path / "campaign" / "RUN_20260822T010203Z_race__one")
    os.makedirs(os.path.join(run_root, "receipts", "execution"))
    token = "f" * 64

    with ThreadPoolExecutor(max_workers=2) as pool:
        claim_future = pool.submit(backend.acquire_lifecycle_claim, run_root, "submit:stage:1")
        transition_future = pool.submit(begin_transition, connection, run_root, token)
        claim_result = claim_future.result()
        try:
            transition_result = transition_future.result()
        except ValueError:
            transition_result = None

    assert (claim_result is not None, transition_result is not None) in {(True, False), (False, True)}
    if claim_result is not None:
        backend.release_lifecycle_claim(run_root, claim_result)
    else:
        rollback_transition(connection, run_root, token)


def test_identical_operations_hold_distinct_claims_until_each_owner_releases(tmp_path) -> None:
    """One caller cannot expose transition while an identical caller is active."""

    connection = LocalConnection()
    backend = O2ExecutionBackend(connection)
    run_root = str(tmp_path / "campaign" / "RUN_20260822T010203Z_same__operation")
    os.makedirs(os.path.join(run_root, "receipts", "execution"))

    first = backend.acquire_lifecycle_claim(run_root, "reconcile:plan:stage")
    second = backend.acquire_lifecycle_claim(run_root, "reconcile:plan:stage")
    assert first is not None and second is not None and first != second

    backend.release_lifecycle_claim(run_root, first)
    with pytest.raises(ValueError, match="execution mutation claims are still active"):
        begin_transition(connection, run_root, "a" * 64)

    backend.release_lifecycle_claim(run_root, second)
    boundary = begin_transition(connection, run_root, "b" * 64)
    assert boundary.job_ids == ()
    rollback_transition(connection, run_root, "b" * 64)


def test_private_incomplete_claim_staging_does_not_block_transition(tmp_path) -> None:
    """A pre-publication crash leaves no malformed final exclusion claim."""

    connection = LocalConnection()
    run_root = str(tmp_path / "campaign" / "RUN_20260822T010203Z_partial__claim")
    os.makedirs(os.path.join(run_root, "receipts", "execution"))
    coordination = coordination_root(run_root)
    os.makedirs(coordination)
    # The atomic acquire program stages under a dot-prefixed private name.  A
    # crash before hard-link publication may leave these bytes, but they are not
    # an ownership claim and therefore cannot strand the lifecycle forever.
    with open(os.path.join(coordination, ".claim-dead.json.tmp-orphan"), "w", encoding="utf-8") as handle:
        handle.write('{"claim_id":')

    boundary = begin_transition(connection, run_root, "d" * 64)
    assert boundary.job_ids == ()
    rollback_transition(connection, run_root, "d" * 64)


def test_identical_transition_reentry_cannot_rollback_first_callers_marker(tmp_path) -> None:
    """A deterministic transition token does not confer marker ownership."""

    connection = LocalConnection()
    run_root = str(tmp_path / "campaign" / "RUN_20260822T010203Z_same__transition")
    os.makedirs(os.path.join(run_root, "receipts", "execution"))
    token = "c" * 64

    begin_transition(connection, run_root, token)
    marker = os.path.join(coordination_root(run_root), "transition.json")
    with pytest.raises(ValueError, match="already marked by another caller"):
        begin_transition(connection, run_root, token)

    # The losing caller must leave the first caller's boundary intact.
    with open(marker, encoding="utf-8") as handle:
        assert handle.read() == token
    rollback_transition(connection, run_root, token)


def test_claim_release_rejects_duplicate_key_ownership_evidence(tmp_path) -> None:
    """Tampered claim JSON cannot collapse duplicate ownership keys on release."""

    backend = O2ExecutionBackend(LocalConnection())
    run_root = str(tmp_path / "campaign" / "RUN_20260822T010203Z_claim__strict")
    claim_id = backend.acquire_lifecycle_claim(run_root, "submit:plan:stage:1")
    assert claim_id is not None
    path = os.path.join(coordination_root(run_root), claim_name(claim_id))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f'{{"claim_id":"foreign","claim_id":"{claim_id}"}}')

    with pytest.raises(RuntimeError, match="duplicate claim key"):
        backend.release_lifecycle_claim(run_root, claim_id)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"claim_id": claim_id}, handle, sort_keys=True)
    backend.release_lifecycle_claim(run_root, claim_id)


def test_unresolved_invocation_blocks_transition_even_without_registry_job_id(tmp_path) -> None:
    """The sbatch crash window is detected from invocation evidence itself."""

    connection = LocalConnection()
    run_root = str(tmp_path / "campaign" / "RUN_20260822T010203Z_orphan__one")
    invocation_dir = os.path.join(run_root, "receipts", "execution", "submission-invocations", "stage")
    os.makedirs(invocation_dir)
    plan_sha256 = "a" * 64
    operation = f"submit:{plan_sha256}:stage:1"
    claim_id = hashlib.sha256(operation.encode()).hexdigest() + "-" + "b" * 64
    with open(os.path.join(invocation_dir, "attempt-001.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "attempt": 1,
                "comment": f"o2plan:v1:{plan_sha256}:stage:a001",
                "intent_sha256": "c" * 64,
                "lifecycle_claim_id": claim_id,
                "plan_sha256": plan_sha256,
                "schema_version": 1,
                "stage_id": "stage",
            },
            handle,
        )

    with pytest.raises(ValueError, match="unresolved sbatch invocation"):
        begin_transition(connection, run_root, "e" * 64)
    assert not os.path.exists(os.path.join(coordination_root(run_root), "transition.json"))
