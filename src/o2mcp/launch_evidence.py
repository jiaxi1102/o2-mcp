"""Mint the authenticated launch-evidence record for a governed O2 stage.

A repository canary can verify its own scientific path but cannot authenticate
its own launch: code that has already been subverted will happily report that it
has not been. The authority therefore has to come from outside the executed
process, which is what this module provides.

The record binds, in one object, the approved plan, the submitted job, the
runtime identities, the destination, and the verified package -- and it *checks*
every link rather than merely restating it. Any disagreement between what the
plan approved and what the run reported makes minting fail; a record only exists
when the chain is intact end to end.

Two properties are deliberate and should survive future edits:

* **The artifacts are read from the cluster, never accepted from the caller.**
  The server reads them back through the authenticated broker before calling in
  here, so the record attests what is actually on O2 rather than what a caller
  claims. Adding a "pass the JSON directly" path would quietly turn this into
  self-attestation.
* **Minting requires a live operator approval.** The caller must quote the
  current policy generation and revision -- obtainable only from a fresh status
  snapshot -- together with a human approval reference, and the mint is recorded
  in the policy audit ledger. That approval is the authenticating element, in
  the sense of AGENTS.md's "one conditional approval of an exact frozen replay".

This module is pure stdlib on purpose so the binding logic stays testable
without the MCP SDK.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

LAUNCH_EVIDENCE_SCHEMA = "o2-launch-evidence-v1"
REQUIRED_PACKAGE_FILES: tuple[str, ...] = (
    "PUBLICATION_OWNER.json",
    "SUCCESS.json",
    "SHA256SUMS",
    "conversion_manifest.json",
)


class LaunchEvidenceError(RuntimeError):
    """One link in the launch chain is missing, malformed, or disagrees."""


def _dig(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    """Walk a dotted path, returning None rather than raising on a gap."""

    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


# Each entry is (label, path within the run diagnostic, path within the plan).
# Declarative so a reviewer can see the whole binding surface at once, and so a
# new field cannot be added to one side without appearing here.
_BINDINGS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("attempt_id", ("attempt_id",), ("attempt_id",)),
    (
        "source_bundle_sha256",
        ("runtime", "source_bundle", "bundle_sha256"),
        ("software", "bundle", "bundle_sha256"),
    ),
    (
        "runtime_wrapper_sha256",
        ("runtime", "runtime_wrapper_sha256"),
        ("runtime_wrapper", "sha256"),
    ),
    (
        "interpreter_sha256",
        ("runtime", "approved_interpreter", "sha256"),
        ("interpreter", "sha256"),
    ),
    (
        "interpreter_closure_sha256",
        ("runtime", "approved_interpreter", "dynamic_closure", "closure_sha256"),
        ("interpreter", "closure_sha256"),
    ),
    (
        "interpreter_replay_context_sha256",
        ("runtime", "approved_interpreter", "runtime_context_sha256"),
        ("interpreter", "context_sha256"),
    ),
    ("destination_inode", ("destination_binding", "inode"), ("destination", "inode")),
    (
        "destination_mount_point",
        ("destination_binding", "linux_mountinfo", "mount_point"),
        ("destination", "mount"),
    ),
    ("package_path", ("output", "package"), ("destination", "expected_package")),
)


def canonical_json(payload: Any) -> str:
    """Serialize deterministically so a digest is reproducible by a reviewer."""

    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


def plan_digest(plan: Mapping[str, Any]) -> str:
    """Digest the plan exactly as the plan builder that froze it did."""

    return hashlib.sha256(canonical_json(plan).encode("utf-8")).hexdigest()


def build_launch_evidence(
    *,
    diagnostic: Mapping[str, Any],
    plan: Mapping[str, Any],
    package_digests: Mapping[str, str],
    owner: Mapping[str, Any],
    approval: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    """Verify every link and return the record, or raise naming what disagreed.

    ``package_digests`` maps each required package filename to the SHA-256 the
    server computed by reading that file back off the cluster.
    """

    expected_plan_sha256 = plan_digest(plan)
    mismatches: list[str] = []

    def bind(label: str, observed: Any, expected: Any) -> None:
        if observed is None:
            mismatches.append(f"{label}: absent from the run diagnostic")
        elif observed != expected:
            mismatches.append(f"{label}: run={observed!r} plan={expected!r}")

    bind("plan_sha256", diagnostic.get("plan_sha256"), expected_plan_sha256)
    for label, run_path, plan_path in _BINDINGS:
        bind(label, _dig(diagnostic, run_path), _dig(plan, plan_path))

    # The owner marker is written first inside the claimed package, so it is the
    # one artifact proving the package on disk belongs to THIS approved plan.
    bind("owner_plan_sha256", owner.get("plan_sha256"), expected_plan_sha256)
    bind("owner_attempt_id", owner.get("attempt_id"), _dig(plan, ("attempt_id",)))

    missing_files = [name for name in REQUIRED_PACKAGE_FILES if name not in package_digests]
    if missing_files:
        mismatches.append("package files absent: {}".format(", ".join(sorted(missing_files))))

    verification_status = _dig(diagnostic, ("output", "verification", "status"))
    if verification_status != "success":
        mismatches.append(f"package verification status is {verification_status!r}, not 'success'")

    if mismatches:
        raise LaunchEvidenceError(
            "refusing to mint launch evidence; the launch chain does not agree: " + "; ".join(mismatches)
        )

    scheduler = _dig(diagnostic, ("slurm", "scheduler")) or {}
    environment = _dig(diagnostic, ("slurm", "environment")) or {}
    binding = diagnostic.get("destination_binding") or {}
    mountinfo = binding.get("linux_mountinfo") or {}
    interpreter = _dig(diagnostic, ("runtime", "approved_interpreter")) or {}
    closure = interpreter.get("dynamic_closure") or {}

    record = {
        "schema": LAUNCH_EVIDENCE_SCHEMA,
        "stage": stage,
        "approved_plan": {"sha256": expected_plan_sha256, "attempt_id": plan.get("attempt_id")},
        "submission": {
            "job_id": scheduler.get("job_id"),
            "allocated_hostnames": scheduler.get("allocated_hostnames"),
            "account": environment.get("SLURM_JOB_ACCOUNT"),
            "partition": environment.get("SLURM_JOB_PARTITION"),
        },
        "runtime_identities": {
            "source_bundle_sha256": _dig(diagnostic, ("runtime", "source_bundle", "bundle_sha256")),
            "runtime_wrapper_sha256": _dig(diagnostic, ("runtime", "runtime_wrapper_sha256")),
            "interpreter_path": interpreter.get("approved_path"),
            "interpreter_sha256": interpreter.get("sha256"),
            "interpreter_closure_sha256": closure.get("closure_sha256"),
            "interpreter_replay_context_sha256": interpreter.get("runtime_context_sha256"),
            "loaded_closure_sha256": _dig(diagnostic, ("launch", "loaded_library_closure", "loaded_closure_sha256")),
        },
        "destination": {
            "package": _dig(diagnostic, ("output", "package")),
            "inode": binding.get("inode"),
            "mount_point": mountinfo.get("mount_point"),
            "mount_source": mountinfo.get("mount_source"),
            # st_dev is assigned per host for network filesystems -- the same NFS
            # mount reports different numbers on the login and compute nodes --
            # so binding it would fail a correct run. Inode and mount source are
            # server-issued and stable.
            "device_note": "st_dev is host-local for NFS and is deliberately not bound",
        },
        "verified_package": {
            "verification_status": verification_status,
            "n_payloads": _dig(diagnostic, ("output", "verification", "n_payloads")),
            "reopened_output_sha256": _dig(diagnostic, ("output", "reopened_output_sha256")),
            "file_digests": dict(package_digests),
        },
        "run_diagnostic": {
            "status": diagnostic.get("status"),
            "continuation_authorized": diagnostic.get("continuation_authorized"),
            "schema_version": diagnostic.get("schema_version"),
        },
        "operator_approval": dict(approval),
        "binding_check": {"all_links_agree": True, "checked": len(_BINDINGS) + 4},
    }
    return record


def launch_evidence_digest(record: Mapping[str, Any]) -> str:
    """Digest a minted record so it can be quoted and re-verified later."""

    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def parse_json_artifact(text: str, *, label: str) -> dict[str, Any]:
    """Parse one artifact read off the cluster, naming it when it is not JSON."""

    try:
        payload = json.loads(text)
    except ValueError as error:
        raise LaunchEvidenceError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise LaunchEvidenceError(f"{label} must be a JSON object")
    return payload


def required_package_files() -> tuple[str, ...]:
    """Expose the package files a mint must digest, for the server's reader."""

    return REQUIRED_PACKAGE_FILES


__all__ = [
    "LAUNCH_EVIDENCE_SCHEMA",
    "LaunchEvidenceError",
    "build_launch_evidence",
    "canonical_json",
    "launch_evidence_digest",
    "parse_json_artifact",
    "plan_digest",
    "required_package_files",
]
