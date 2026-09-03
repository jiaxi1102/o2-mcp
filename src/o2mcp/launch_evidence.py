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
  in the policy audit ledger *against this record's content digest*. That
  approval is the authenticating element, in the sense of AGENTS.md's "one
  conditional approval of an exact frozen replay", and storing the digest with
  it is what stops an edited copy of a legitimate record from still matching
  the ledger.
* **The package is reverified here, not taken on the run's word.** The run
  diagnostic's own "verification succeeded" is the executed process vouching
  for itself, so every payload SHA256SUMS names is hashed again on the cluster
  and compared in this module.

This module is pure stdlib on purpose so the binding logic stays testable
without the MCP SDK.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

LAUNCH_EVIDENCE_SCHEMA = "o2-launch-evidence-v1"
# The canary emits exactly two status strings -- "success" for package
# verification and this one for the run object -- and signals every failure by
# raising, so a diagnostic file exists only for a run that reached the end.
# Requiring it therefore cannot refuse a correct run.
EXPECTED_DIAGNOSTIC_STATUS = "diagnostic_success"
REQUIRED_PACKAGE_FILES: tuple[str, ...] = (
    "PUBLICATION_OWNER.json",
    "SUCCESS.json",
    "SHA256SUMS",
    "conversion_manifest.json",
)


class LaunchEvidenceError(RuntimeError):
    """One link in the launch chain is missing, malformed, or disagrees."""


# GNU sha256sum writes "<64 hex><space><mode><name>", where <mode> is " " for a
# text read and "*" for a binary one and <name> runs to the end of the line.
# Splitting on whitespace would silently drop every entry whose filename
# contains a space, which is a legal character in an O2 path. The mode is
# optional here only so a manifest written with a single separator by something
# other than coreutils still parses; it cannot mis-read GNU output, because a
# name that itself starts with a space or "*" still keeps its leading character
# once the fixed separator is consumed.
_SHA256_LINE = re.compile(r"^([0-9a-fA-F]{64}) [ *]?(.+)$")


def parse_sha256_lines(text: str, *, label: str) -> dict[str, str]:
    """Parse ``sha256sum`` output -- and a SHA256SUMS manifest, same format.

    Returns ``{name: digest}`` with the name exactly as the tool wrote it. A
    line that is not an entry raises rather than being skipped: a partially
    parsed listing looks indistinguishable from a package that is missing
    files, and this record must never rest on that ambiguity.
    """

    digests: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line.strip():
            continue
        if line.startswith("\\"):
            # GNU sha256sum escapes a name containing a newline or a backslash
            # by prefixing its line with "\". The original name cannot be
            # recovered unambiguously from that, and a record must not bind a
            # filename it had to guess at.
            raise LaunchEvidenceError(f"{label} contains an escaped filename, which cannot be bound unambiguously")
        match = _SHA256_LINE.match(line)
        if match is None:
            raise LaunchEvidenceError(f"{label} has a line that is not a sha256 entry: {line[:120]!r}")
        digest, name = match.group(1).lower(), match.group(2)
        if digests.get(name, digest) != digest:
            raise LaunchEvidenceError(f"{label} lists {name!r} twice with different digests")
        digests[name] = digest
    return digests


def parse_checksum_manifest(text: str, *, label: str = "SHA256SUMS") -> dict[str, str]:
    """Parse the package's own checksum manifest into ``{payload: digest}``.

    Every name must be a path *inside* the package: the reverification hashes
    what this manifest names, so an absolute or traversal-bearing entry would
    let a package steer the read outside itself and have the result counted as
    that package's payload.
    """

    manifest: dict[str, str] = {}
    for name, digest in parse_sha256_lines(text, label=label).items():
        candidate = PurePosixPath(name)
        normalized = str(candidate)
        if candidate.is_absolute() or ".." in candidate.parts or normalized in {".", ""} or not name.strip():
            raise LaunchEvidenceError(f"{label} names {name!r}, which is not a payload path inside the package")
        if manifest.get(normalized, digest) != digest:
            raise LaunchEvidenceError(f"{label} lists {normalized!r} twice with different digests")
        manifest[normalized] = digest
    if not manifest:
        raise LaunchEvidenceError(f"{label} lists no payloads, so there is nothing to reverify")
    return manifest


def decode_verified_artifact(encoded: str, *, expected_sha256: str | None, label: str) -> str:
    """Decode one package file and prove it is the file whose digest was recorded.

    A file inside the package is read by one command and hashed by another, so
    on their own the bytes that were parsed and the digest the record reports
    for that filename are not the same evidence: a replacement in between would
    let a record quote content it never used alongside a digest of something
    else. Hashing the decoded bytes here and requiring them to match closes
    that. The transport is base64 so the bytes survive line-splitting exactly --
    a digest over reconstructed text would prove only that the reconstruction
    was self-consistent.
    """

    try:
        raw = base64.b64decode("".join(encoded.split()), validate=True)
    except (binascii.Error, ValueError) as error:
        raise LaunchEvidenceError(f"{label} did not come back as valid base64: {error}") from error
    if not expected_sha256:
        raise LaunchEvidenceError(f"the package digest output carried no digest for {label}")
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha256:
        raise LaunchEvidenceError(
            f"{label} changed while it was being read: the bytes parsed hash to {observed}, "
            f"but the file digest recorded is {expected_sha256}"
        )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LaunchEvidenceError(f"{label} is not valid UTF-8: {error}") from error


def parse_encoded_checksum_manifest(
    encoded: str, *, expected_sha256: str | None, label: str = "SHA256SUMS"
) -> dict[str, str]:
    """Decode the manifest, prove it is the file whose digest was recorded, parse it."""

    return parse_checksum_manifest(
        decode_verified_artifact(encoded, expected_sha256=expected_sha256, label=label), label=label
    )


def parse_encoded_json_artifact(encoded: str, *, expected_sha256: str | None, label: str) -> dict[str, Any]:
    """Decode a JSON package file, prove it is the file that was hashed, parse it."""

    return parse_json_artifact(
        decode_verified_artifact(encoded, expected_sha256=expected_sha256, label=label), label=label
    )


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


def _same_posix_path(left: Any, right: Any) -> bool:
    """Compare two O2 paths ignoring only cosmetic separator differences."""

    if not isinstance(left, str) or not isinstance(right, str) or not left.strip() or not right.strip():
        return False
    return str(PurePosixPath(left)) == str(PurePosixPath(right))


def refuse_package_symlinks(listing: str, *, package_path: str) -> None:
    """Refuse a package containing any symlink at all.

    Resolving each payload and requiring it to land inside the package was two
    independent pathname resolutions, and a run that toggled a link between them
    could pass the containment check while the digest came from elsewhere.
    There is no way to make two shell commands share a descriptor, so this
    removes the thing being raced instead: a published package contains no
    symlinks, so any link found is grounds to refuse rather than to reason
    about.

    That is not a conservative approximation. The publisher hard-links and
    passes ``follow_symlinks=False`` at every site, and the package verifier
    rejects symlinks outright, so a package containing one is already invalid by
    its own rules.
    """

    offenders = [line.strip() for line in listing.splitlines() if line.strip()]
    if offenders:
        shown = ", ".join(repr(name) for name in offenders[:5])
        more = f" (and {len(offenders) - 5} more)" if len(offenders) > 5 else ""
        raise LaunchEvidenceError(
            f"refusing to mint launch evidence; {package_path!r} contains symlinks, which a published "
            f"package never does and its own verifier rejects: {shown}{more}"
        )


def build_launch_evidence(
    *,
    diagnostic: Mapping[str, Any],
    plan: Mapping[str, Any],
    package_digests: Mapping[str, str],
    checksum_manifest: Mapping[str, str],
    payload_digests: Mapping[str, str],
    owner: Mapping[str, Any],
    approval: Mapping[str, Any],
    stage: str,
    read_back_package_path: str,
    resolved_package_path: str,
) -> dict[str, Any]:
    """Verify every link and return the record, or raise naming what disagreed.

    ``package_digests`` maps each required package filename to the SHA-256 the
    server computed by reading that file back off the cluster.
    ``checksum_manifest`` is the package's own SHA256SUMS as parsed, and
    ``payload_digests`` is what those same payloads actually hash to now --
    hashed by the server, not by the run. ``read_back_package_path`` is the
    directory the server actually read and ``resolved_package_path`` is what it
    resolves to on the cluster; both must be the package the plan approved. A
    manifest entry cannot escape the package because the names are refused if
    they are absolute or traversing and the package is refused if it holds any
    symlink at all; see ``refuse_package_symlinks``.
    """

    expected_plan_sha256 = plan_digest(plan)
    approved_package = _dig(plan, ("destination", "expected_package"))
    mismatches: list[str] = []
    checks = 0

    def bind(label: str, observed: Any, expected: Any) -> None:
        nonlocal checks
        checks += 1
        if observed is None:
            mismatches.append(f"{label}: absent from the run diagnostic")
        elif observed != expected:
            mismatches.append(f"{label}: run={observed!r} plan={expected!r}")

    bind("plan_sha256", diagnostic.get("plan_sha256"), expected_plan_sha256)
    for label, run_path, plan_path in _BINDINGS:
        bind(label, _dig(diagnostic, run_path), _dig(plan, plan_path))

    # The caller chooses which directory is read back, so without this the owner
    # marker and the digests below could describe package B while the record
    # reports package A -- a copied marker is all it would take. Bind the
    # directory that was actually read to the one the plan approved; the plan
    # and the diagnostic are bound to each other just above.
    checks += 1
    if not _same_posix_path(read_back_package_path, approved_package):
        mismatches.append(f"package_read_back_path: read={read_back_package_path!r} plan={approved_package!r}")

    # A lexical comparison is not enough on its own: `cat` and `sha256sum` follow
    # links, so a pathname that spells the approved package can still open a
    # substituted directory. Bind what the cluster actually resolved it to.
    checks += 1
    if not _same_posix_path(resolved_package_path, approved_package):
        mismatches.append(
            f"resolved_package_path: the cluster resolves the package to {resolved_package_path!r}, "
            f"but the plan approved {approved_package!r}"
        )

    # The owner marker is written first inside the claimed package, so it is the
    # one artifact proving the package on disk belongs to THIS approved plan.
    bind("owner_plan_sha256", owner.get("plan_sha256"), expected_plan_sha256)
    bind("owner_attempt_id", owner.get("attempt_id"), _dig(plan, ("attempt_id",)))

    checks += 1
    missing_files = [name for name in REQUIRED_PACKAGE_FILES if name not in package_digests]
    if missing_files:
        mismatches.append("package files absent: {}".format(", ".join(sorted(missing_files))))

    # A record whose whole purpose is to bind an approved plan to a submitted job
    # must name the job. Without this the field is simply null, and the audit
    # ledger records the literal string "None" as the job it approved.
    checks += 1
    job_id = _dig(diagnostic, ("slurm", "scheduler", "job_id"))
    if isinstance(job_id, bool) or not isinstance(job_id, (str, int)) or not str(job_id).strip():
        mismatches.append(f"submission job_id is {job_id!r}, so the record would bind no submitted job")

    checks += 1
    verification_status = _dig(diagnostic, ("output", "verification", "status"))
    if verification_status != "success":
        mismatches.append(f"package verification status is {verification_status!r}, not 'success'")

    # The run's own top-level outcome. Note that `continuation_authorized: false`
    # is NOT a failure and is deliberately not checked: it is the run asserting
    # it cannot authenticate its own launch, which is the reason this record
    # exists at all.
    checks += 1
    diagnostic_status = diagnostic.get("status")
    if diagnostic_status != EXPECTED_DIAGNOSTIC_STATUS:
        mismatches.append(
            f"run diagnostic status is {diagnostic_status!r}, not {EXPECTED_DIAGNOSTIC_STATUS!r}; "
            "the run's own outcome does not say it finished"
        )

    # That status is the executed process vouching for itself, and this record
    # exists precisely because such a process cannot authenticate its own
    # output: a payload deleted or altered after the run reported success, or a
    # run that reported success falsely, would both still read as verified. So
    # recompute the package's own manifest here from digests taken outside it.
    if not checksum_manifest:
        checks += 1
        mismatches.append("SHA256SUMS lists no payloads, so nothing could be reverified")

    # SHA256SUMS covers exactly the payloads the run counted: it cannot list
    # itself, and SUCCESS.json is written after it. So a manifest replaced after
    # the run with a valid but shorter one -- payloads deleted from disk and
    # from the manifest together -- is caught here and nowhere else.
    checks += 1
    n_payloads = _dig(diagnostic, ("output", "verification", "n_payloads"))
    if not isinstance(n_payloads, int) or isinstance(n_payloads, bool):
        mismatches.append(f"the run diagnostic's n_payloads is {n_payloads!r}, so nothing pins the manifest length")
    elif n_payloads != len(checksum_manifest):
        mismatches.append(f"SHA256SUMS lists {len(checksum_manifest)} payloads but the run verified {n_payloads}")
    for name, expected_digest in sorted(checksum_manifest.items()):
        checks += 1
        observed_digest = payload_digests.get(name)
        if observed_digest is None:
            mismatches.append(f"payload {name!r} is listed in SHA256SUMS but was not hashed on the cluster")
        elif observed_digest != expected_digest:
            mismatches.append(f"payload {name!r}: SHA256SUMS={expected_digest} on_disk={observed_digest}")
        # A metadata file the manifest also covers was hashed twice, once per
        # read. Requiring the two to agree closes the window between them.
        elif name in package_digests and package_digests[name] != observed_digest:
            mismatches.append(f"payload {name!r} changed between the two reads of the package")

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
            "read_back_package_path": read_back_package_path,
            "resolved_package_path": resolved_package_path,
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
            "payload_reverification": {
                "manifest": "SHA256SUMS",
                "manifest_sha256": package_digests.get("SHA256SUMS"),
                "n_payloads_reverified": len(checksum_manifest),
                "method": (
                    "every path SHA256SUMS names was hashed on the cluster and compared here, so this "
                    "record does not rest on the run's own verification verdict"
                ),
            },
        },
        "run_diagnostic": {
            "status": diagnostic.get("status"),
            "continuation_authorized": diagnostic.get("continuation_authorized"),
            "schema_version": diagnostic.get("schema_version"),
        },
        "operator_approval": dict(approval),
        "binding_check": {"all_links_agree": True, "checked": checks},
    }

    # The ledger records the approval against this content digest, so a record
    # whose content no longer produces the approved digest is not the record the
    # operator approved, whatever its approval object says.
    recorded_digest = approval.get("evidence_sha256")
    if recorded_digest is not None and recorded_digest != evidence_content_digest(record):
        raise LaunchEvidenceError(
            "refusing to mint launch evidence; the approval in the audit ledger was recorded against a "
            "different record than the one built here"
        )
    return record


def launch_evidence_digest(record: Mapping[str, Any]) -> str:
    """Digest a minted record so it can be quoted and re-verified later."""

    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def evidence_content_digest(record: Mapping[str, Any]) -> str:
    """Digest everything the record asserts, excluding the operator approval.

    This is the digest the policy ledger stores when the mint is approved, and
    it is what makes that ledger entry authenticate the record rather than
    merely note that a mint happened. The approval itself is excluded because
    it does not exist yet when the digest is taken -- and because a holder of a
    legitimate record could otherwise edit its runtime identities or package
    digests, recompute the unkeyed whole-record digest, and keep an approval
    object that still matched the ledger. Recomputing this from a record in hand
    and comparing it with the ledger entry detects exactly that.
    """

    content = dict(record)
    content["operator_approval"] = {}
    return hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()


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
    "EXPECTED_DIAGNOSTIC_STATUS",
    "LAUNCH_EVIDENCE_SCHEMA",
    "LaunchEvidenceError",
    "build_launch_evidence",
    "canonical_json",
    "decode_verified_artifact",
    "evidence_content_digest",
    "launch_evidence_digest",
    "parse_checksum_manifest",
    "parse_encoded_checksum_manifest",
    "parse_encoded_json_artifact",
    "parse_json_artifact",
    "parse_sha256_lines",
    "refuse_package_symlinks",
    "plan_digest",
    "required_package_files",
]
