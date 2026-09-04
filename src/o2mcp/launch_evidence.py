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
without the MCP SDK. The two in-repo imports -- the scheduler's success states
and the duplicate-rejecting JSON decoder -- are themselves stdlib-only, and
exist so this does not restate rules the repository already defines.
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

from o2mcp.runorg.execution_models import SUCCESS_SLURM_STATES, SubmissionIdentity
from o2mcp.runorg.strict_json import strict_json_object

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


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _well_formed(kind: str, value: Any) -> bool:
    """Is this value even the kind of thing the field is supposed to hold?

    Equality alone is not a binding. A falsified plan and a matching diagnostic
    can agree on ``{}`` or ``""`` for an interpreter digest, and every check
    would pass while the record carried an identity that identifies nothing.
    """

    if kind == "sha256":
        return isinstance(value, str) and bool(_SHA256.match(value))
    if kind == "inode":
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if kind == "path":
        return isinstance(value, str) and value.startswith("/") and bool(value.strip())
    return isinstance(value, str) and bool(value.strip())


# Each entry is (label, path within the run diagnostic, path within the plan,
# the shape the value must have). Declarative so a reviewer can see the whole
# binding surface at once, and so a new field cannot be added to one side
# without appearing here.
_BINDINGS: tuple[tuple[str, tuple[str, ...], tuple[str, ...], str], ...] = (
    ("attempt_id", ("attempt_id",), ("attempt_id",), "text"),
    (
        "source_bundle_sha256",
        ("runtime", "source_bundle", "bundle_sha256"),
        ("software", "bundle", "bundle_sha256"),
        "sha256",
    ),
    (
        "runtime_wrapper_sha256",
        ("runtime", "runtime_wrapper_sha256"),
        ("runtime_wrapper", "sha256"),
        "sha256",
    ),
    (
        "interpreter_sha256",
        ("runtime", "approved_interpreter", "sha256"),
        ("interpreter", "sha256"),
        "sha256",
    ),
    (
        "interpreter_closure_sha256",
        ("runtime", "approved_interpreter", "dynamic_closure", "closure_sha256"),
        ("interpreter", "closure_sha256"),
        "sha256",
    ),
    (
        "interpreter_replay_context_sha256",
        ("runtime", "approved_interpreter", "runtime_context_sha256"),
        ("interpreter", "context_sha256"),
        "sha256",
    ),
    ("destination_inode", ("destination_binding", "inode"), ("destination", "inode"), "inode"),
    (
        "destination_mount_point",
        ("destination_binding", "linux_mountinfo", "mount_point"),
        ("destination", "mount"),
        "path",
    ),
    ("package_path", ("output", "package"), ("destination", "expected_package"), "path"),
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


# Slurm job ids are numeric, with an underscore-separated task index for array
# members. Requiring the shape keeps a value read from the untrusted diagnostic
# from reaching a shell as anything but a job id.
_JOB_ID = re.compile(r"^[0-9]+(_[0-9]+)?$")
SACCT_RETENTION_DAYS = 366
# States a finished governed stage can legitimately be reported in. The
# allowance exists for one reason only -- a mint can arrive before accounting
# settles -- so it covers just the states a job passes through *after it has
# run*: RUNNING, because the diagnostic is written before the process exits, and
# COMPLETING, the final transition. The success state itself comes from this
# repo's own scheduler vocabulary rather than being restated here.
#
# PENDING and REQUEUED are excluded even though runorg classes them as active: a
# job that has not started cannot have produced the diagnostic being attested,
# and allowing them would let a compromised diagnostic name any queued job in
# the same account and partition. Everything else -- every terminal failure, and
# any state neither this code nor Slurm's vocabulary recognises -- refuses, so a
# future state cannot quietly become mintable.
MINTABLE_JOB_STATES = SUCCESS_SLURM_STATES | frozenset({"RUNNING", "COMPLETING"})


def claimed_job_id(diagnostic: Mapping[str, Any]) -> str:
    """Return the job the diagnostic claims, or refuse if it names none usably.

    This is what the server must query accounting for, so it is extracted --
    and shape-checked -- before any command is built from it.
    """

    job_id = _dig(diagnostic, ("slurm", "scheduler", "job_id"))
    if isinstance(job_id, bool) or not isinstance(job_id, (str, int)) or not str(job_id).strip():
        raise LaunchEvidenceError(
            f"refusing to mint launch evidence; submission job_id is {job_id!r}, "
            "so the record would bind no submitted job"
        )
    candidate = str(job_id).strip()
    if not _JOB_ID.match(candidate):
        raise LaunchEvidenceError(
            f"refusing to mint launch evidence; submission job_id {candidate!r} is not a Slurm job id"
        )
    return candidate


def parse_scheduler_record(text: str, *, job_id: str) -> dict[str, str]:
    """Parse one ``sacct`` allocation row into the job identity it reports.

    An empty result is refused rather than falling back to the diagnostic's own
    claim: accounting is purged after 366 days, and a mint that quietly
    trusted the run because its job had aged out is exactly the degradation this
    record exists to prevent.
    """

    rows = [line for line in text.splitlines() if line.strip()]
    if not rows:
        raise LaunchEvidenceError(
            f"refusing to mint launch evidence; Slurm accounting has no record of job {job_id}. "
            f"Accounting is purged after {SACCT_RETENTION_DAYS} days, so either the run is too old to "
            "attest or the job never existed; either way the job identity cannot be confirmed here"
        )
    if len(rows) > 1:
        raise LaunchEvidenceError(
            f"refusing to mint launch evidence; Slurm accounting returned {len(rows)} allocations for "
            f"job {job_id}, so its identity is ambiguous"
        )
    fields = rows[0].split("|")
    if len(fields) != 5:
        raise LaunchEvidenceError(f"the Slurm accounting row for job {job_id} is not the five fields requested")
    reported_id, state, account, partition, comment = (field.strip() for field in fields)
    if reported_id != job_id:
        raise LaunchEvidenceError(
            f"refusing to mint launch evidence; Slurm accounting answered for job {reported_id!r} "
            f"when asked about {job_id!r}"
        )
    # sacct decorates some states, e.g. "CANCELLED by 12345", so compare the verb.
    verb = state.split()[0] if state.split() else ""
    if verb not in MINTABLE_JOB_STATES:
        raise LaunchEvidenceError(
            f"refusing to mint launch evidence; Slurm reports job {job_id} as {state!r}. A record attests a "
            "finished governed stage, so a job the scheduler says did not finish cannot produce one"
        )
    return {
        "job_id": reported_id,
        "state": state,
        "account": account,
        "partition": partition,
        "comment": comment,
    }


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
    observed_package_inode: int,
    scheduler_record: Mapping[str, str],
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
    symlink at all; see ``refuse_package_symlinks``. ``scheduler_record`` is what
    Slurm accounting reports for the claimed job, read through the broker rather
    than taken from the diagnostic.
    """

    expected_plan_sha256 = plan_digest(plan)
    approved_package = _dig(plan, ("destination", "expected_package"))
    mismatches: list[str] = []
    checks = 0

    def bind(label: str, observed: Any, expected: Any, kind: str = "text") -> None:
        nonlocal checks
        checks += 1
        if observed is None:
            mismatches.append(f"{label}: absent from the run diagnostic")
        elif observed != expected:
            mismatches.append(f"{label}: run={observed!r} plan={expected!r}")
        # Agreeing on a value that is not the kind of thing the field holds is
        # not a binding: a falsified plan and a matching diagnostic could settle
        # on an empty string or an object for an interpreter digest and pass
        # every check while identifying nothing.
        elif not _well_formed(kind, observed):
            mismatches.append(f"{label}: {observed!r} is not a well-formed {kind}")

    bind("plan_sha256", diagnostic.get("plan_sha256"), expected_plan_sha256, "sha256")
    for label, run_path, plan_path, kind in _BINDINGS:
        bind(label, _dig(diagnostic, run_path), _dig(plan, plan_path), kind)

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
    # The stage names which governed stage this record is for, and it arrives
    # with the operator's approval. The plan names it too, so the two must agree
    # -- otherwise the same artifacts would mint under any label an operator
    # happened to approve.
    bind("stage_id", stage, _dig(plan, ("stage_id",)), "text")

    # Resolving the pathname does not settle identity: a different directory
    # renamed into the approved path resolves the same. The run recorded the
    # package's own inode, and the server observed it through the descriptor it
    # actually read; requiring the two to agree is what closes that.
    # `destination.inode` is NOT this -- it names the parent directory -- so both
    # are bound, separately.
    checks += 1
    recorded_inode = _dig(diagnostic, ("output", "package_inode"))
    if not _well_formed("inode", recorded_inode):
        mismatches.append(f"package_inode: the run recorded {recorded_inode!r}, which is not an inode")
    elif recorded_inode != observed_package_inode:
        mismatches.append(
            f"package_inode: run={recorded_inode} observed={observed_package_inode}; the directory read is "
            "not the one the run wrote"
        )

    checks += 1
    if not _same_posix_path(resolved_package_path, approved_package):
        mismatches.append(
            f"resolved_package_path: the cluster resolves the package to {resolved_package_path!r}, "
            f"but the plan approved {approved_package!r}"
        )

    # The owner marker is written first inside the claimed package, so it is the
    # one artifact proving the package on disk belongs to THIS approved plan.
    bind("owner_plan_sha256", owner.get("plan_sha256"), expected_plan_sha256, "sha256")
    bind("owner_attempt_id", owner.get("attempt_id"), _dig(plan, ("attempt_id",)), "text")

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
    else:
        # The job identity was the last field taken on the run's word, and a
        # compromised run can put any plausible id here. Bind it to what the
        # scheduler itself reports instead. The account and partition come from
        # the job's own environment, so accounting must agree with them.
        checks += 1
        if scheduler_record.get("job_id") != str(job_id).strip():
            mismatches.append(
                f"scheduler job_id: diagnostic={str(job_id).strip()!r} "
                f"accounting={scheduler_record.get('job_id')!r}"
            )
        for label, key, reported in (
            ("account", "SLURM_JOB_ACCOUNT", "account"),
            ("partition", "SLURM_JOB_PARTITION", "partition"),
        ):
            checks += 1
            claimed = _dig(diagnostic, ("slurm", "environment", key))
            if claimed != scheduler_record.get(reported):
                mismatches.append(
                    f"scheduler {label}: diagnostic={claimed!r} accounting={scheduler_record.get(reported)!r}"
                )
        # The Slurm comment is the only field that ties the accounting row to
        # *this* plan rather than to any job of the same account and partition.
        # The execution engine already writes a structured identity there --
        # o2plan:v1:<plan sha>:<stage>:aNNN -- so this parses it with the
        # engine's own reader rather than inventing a second format, and binds
        # both the plan digest and the stage it carries.
        #
        # It is checked when present and recorded as unbound when absent, so the
        # binding is live for engine-submitted jobs without making jobs
        # submitted before it unattestable. That asymmetry is deliberate. A
        # comment that is present but not an engine identity is refused: a
        # non-empty comment this cannot verify is not evidence of anything.
        comment = (scheduler_record.get("comment") or "").strip()
        if comment:
            checks += 1
            try:
                identity = SubmissionIdentity.from_comment(comment)
            except ValueError:
                mismatches.append(
                    f"scheduler comment: accounting={comment!r} is not an o2-mcp submission identity, so "
                    "it cannot confirm which plan that job ran"
                )
            else:
                if identity.plan_sha256 != expected_plan_sha256:
                    mismatches.append(
                        f"scheduler comment: the job names plan {identity.plan_sha256} but this record "
                        f"attests {expected_plan_sha256}"
                    )
                checks += 1
                if identity.stage_id != stage:
                    mismatches.append(
                        f"scheduler comment: the job names stage {identity.stage_id!r} but this record "
                        f"attests {stage!r}"
                    )
                # Retries of the same plan and stage differ only here, so without
                # this a diagnostic could name a completed job from a different
                # attempt and every other scheduler check would still pass. The
                # engine writes the attempt zero-padded to three digits, which is
                # the same form the plan's attempt_id and the package name use.
                checks += 1
                claimed_attempt = _dig(plan, ("attempt_id",))
                if f"{identity.attempt:03d}" != claimed_attempt:
                    mismatches.append(
                        f"scheduler comment: the job names attempt {identity.attempt:03d} but this record "
                        f"attests {claimed_attempt!r}"
                    )

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

    # The run recorded digests of the two files it built the package around, so
    # the bytes the server read can be bound to the bytes the run verified. This
    # is what the payload count could only approximate: an equal-size manifest
    # replacement passes a cardinality check and fails this one.
    for label, name, run_path in (
        ("sha256sums_sha256", "SHA256SUMS", ("output", "sha256sums_sha256")),
        ("conversion_manifest_sha256", "conversion_manifest.json", ("output", "conversion_manifest_sha256")),
    ):
        checks += 1
        recorded = _dig(diagnostic, run_path)
        observed = package_digests.get(name)
        if not _well_formed("sha256", recorded):
            mismatches.append(f"{label}: the run recorded {recorded!r}, which is not a digest")
        elif recorded != observed:
            mismatches.append(f"{label}: run={recorded} read={observed!r}; {name} is not the file the run verified")

    # SHA256SUMS covers exactly the payloads the run counted: it cannot list
    # itself, and SUCCESS.json is written after it. So a manifest replaced after
    # the run with a valid but shorter one -- payloads deleted from disk and
    # from the manifest together -- is caught here as well, on a different
    # failure than the digest binding above.
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
            # job_id, account and partition are all bound to what Slurm
            # accounting reports for the job, read through the broker. The id is
            # stored in the canonical text form the ledger and sacct both use:
            # keeping the raw value would let a diagnostic reporting " 52085188 "
            # mint a record that then failed its own ledger comparison, since
            # the ledger stores the stripped id.
            "job_id": str(scheduler.get("job_id")).strip(),
            "account": environment.get("SLURM_JOB_ACCOUNT"),
            "partition": environment.get("SLURM_JOB_PARTITION"),
            # What accounting itself said. The state is recorded but not gated
            # on beyond refusing a job that did not run: a mint can legitimately
            # race a job's final accounting transition, and refusing on that
            # would fail correct runs for their timing rather than their content.
            "scheduler_accounting": dict(scheduler_record),
        },
        "runtime_identities": {
            # Every digest here is bound to the approved plan by _BINDINGS, so a
            # run that differs from what was approved cannot carry them.
            "source_bundle_sha256": _dig(diagnostic, ("runtime", "source_bundle", "bundle_sha256")),
            "runtime_wrapper_sha256": _dig(diagnostic, ("runtime", "runtime_wrapper_sha256")),
            "interpreter_sha256": interpreter.get("sha256"),
            "interpreter_closure_sha256": closure.get("closure_sha256"),
            "interpreter_replay_context_sha256": interpreter.get("runtime_context_sha256"),
        },
        "destination": {
            # package, inode and mount_point are bound to the plan; the two
            # read-back paths are the server's own, one of them observed on the
            # cluster.
            "package": _dig(diagnostic, ("output", "package")),
            "read_back_package_path": read_back_package_path,
            "resolved_package_path": resolved_package_path,
            # The parent directory's inode, bound plan-to-diagnostic.
            "inode": binding.get("inode"),
            # The package directory's own inode, observed by the server through
            # the descriptor it read the package under and bound to the one the
            # run recorded.
            "package_inode": observed_package_inode,
            "mount_point": mountinfo.get("mount_point"),
            # st_dev is assigned per host for network filesystems -- the same NFS
            # mount reports different numbers on the login and compute nodes --
            # so binding it would fail a correct run. The inode is stable.
            "device_note": "st_dev is host-local for NFS and is deliberately not bound",
        },
        "verified_package": {
            # The status is required to be "success" and the payload count is
            # bound to the manifest length; the digests below are the server's
            # own, computed by reading the package back.
            "verification_status": verification_status,
            "n_payloads": _dig(diagnostic, ("output", "verification", "n_payloads")),
            "file_digests": dict(package_digests),
            "payload_reverification": {
                "manifest": "SHA256SUMS",
                "manifest_sha256": package_digests.get("SHA256SUMS"),
                "n_payloads_reverified": len(checksum_manifest),
                "method": (
                    "every path SHA256SUMS names was hashed on the cluster and compared here, so this "
                    "record does not rest on the run's own verification verdict"
                ),
                # Say what the guarantee is, rather than letting a reader infer a
                # stronger one. Every payload is re-read in a single pass after
                # hashing and before this record is approved, which is what rules
                # out a payload rewritten between its own digest and the ledger
                # entry. It is a point-in-time check, not a lock: nothing here can
                # hold an NFS package still, so bytes rewritten after the final
                # pass are outside what this attests.
                "stability": (
                    "identity of every payload re-read in one pass after hashing and before approval; "
                    "point-in-time, not a lock -- an NFS package cannot be held still, so a rewrite "
                    "after the final pass is outside what this record attests"
                ),
            },
        },
        "run_diagnostic": {
            "status": diagnostic.get("status"),
            # Say how far the binding actually reaches. Every field taken from
            # this file is cross-checked against Slurm accounting, but the FILE
            # is chosen by the caller and its provenance is not established: an
            # actor able to write the approved package location could point at a
            # fabricated diagnostic whose fields were copied from a real job for
            # the same plan, and every check here would agree.
            #
            # No additional content check closes that. Job id, account,
            # partition, comment, node and times are all readable by whoever can
            # write the package, so any of them can be copied. Closing it needs
            # something that actor cannot produce -- a secret the job embeds at
            # submission, or an artifact only the scheduler can write -- which is
            # a change to the submission path, not to minting.
            "binding": (
                "fields cross-checked against Slurm accounting; the diagnostic file itself is "
                "caller-selected and its provenance is not established, so an actor with write "
                "access to the approved package location could substitute one whose fields match"
            ),
        },
        # Everything the executed process reported that nothing here checks. A
        # compromised run can put any value in any of these, so they are kept
        # apart from the authenticated claims rather than mixed in with them: a
        # reader must never have to know which fields of a section were verified.
        # Moving a field out of here means adding a binding or a server-side
        # observation for it, not relabelling it.
        "unbound_run_reported": {
            "note": (
                "reported by the executed process and bound to nothing the plan approved, the "
                "scheduler confirmed, or the server observed; not part of this record's "
                "authenticated claims"
            ),
            # No approved counterpart exists for either of these.
            "interpreter_path": interpreter.get("approved_path"),
            "loaded_closure_sha256": _dig(diagnostic, ("launch", "loaded_library_closure", "loaded_closure_sha256")),
            # The accounting query does not ask for a node list: Slurm returns a
            # compact hostlist expression that would have to be expanded to
            # compare, and expanding it wrongly would refuse correct runs.
            "allocated_hostnames": scheduler.get("allocated_hostnames"),
            # The mount point is bound; the source backing it is not observed.
            "mount_source": mountinfo.get("mount_source"),
            # Empty for jobs submitted before the engine set a structured
            # identity here; non-empty it is parsed and bound, and the attempt
            # index it also carries is recorded but not compared, because the
            # plan's attempt_id format is not this module's to assume.
            "scheduler_comment": (scheduler_record.get("comment") or "").strip() or None,
            # Never recomputed here -- the payloads are rehashed against
            # SHA256SUMS instead, which is what the package's integrity rests on.
            "reopened_output_sha256": _dig(diagnostic, ("output", "reopened_output_sha256")),
            # `false` is the expected value and is deliberately not treated as a
            # failure: it is the run asserting it cannot authenticate its own
            # launch, which is why this record exists.
            "continuation_authorized": diagnostic.get("continuation_authorized"),
            "diagnostic_schema_version": diagnostic.get("schema_version"),
        },
        "operator_approval": dict(approval),
        "binding_check": {
            "all_links_agree": True,
            "checked": checks,
            # Whether the accounting row could be tied to this plan. The plan
            # builder sets the job's Slurm comment to the plan digest; jobs
            # submitted before that carry none, and for those the job identity
            # rests on id, account, partition and state alone. Said here because
            # binding_check is where a reader looks for what was checked.
            "scheduler_comment_bound": bool((scheduler_record.get("comment") or "").strip()),
        },
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


# Every field of a record's ``operator_approval`` and the ledger entry it must
# equal. ``evidence_sha256`` is the digest itself, so it is checked separately.
_APPROVAL_TO_LEDGER: tuple[tuple[str, str], ...] = (
    ("approval_reference", "approval_reference"),
    ("plan_sha256", "plan_sha256"),
    ("policy_revision", "policy_revision"),
    ("policy_generation", "policy_generation"),
    ("client_id", "client_id"),
    ("approved_at", "at"),
)


def verify_launch_evidence(record: Mapping[str, Any], ledger_entry: Mapping[str, Any]) -> None:
    """Check a record in hand against the ledger entry that approved it.

    The content digest deliberately excludes the whole ``operator_approval``
    object, because none of it exists when the digest is taken. That leaves the
    approval fields themselves unprotected by the digest: a holder of a valid
    record could rewrite the reference, the approving client, the revision or
    the time, recompute the record's own digest, and still match the ledger.

    The ledger entry carries all of those fields, so the check is to compare
    them -- which is what this does. It is the supported way to verify a
    persisted record, and it is why the entry records more than the digest.
    """

    observed = evidence_content_digest(record)
    if observed != ledger_entry.get("evidence_sha256"):
        raise LaunchEvidenceError(
            f"this record's content digests to {observed}, but the ledger entry approved "
            f"{ledger_entry.get('evidence_sha256')!r}"
        )
    approval = record.get("operator_approval")
    if not isinstance(approval, Mapping) or not approval:
        # The content digest ignores this object by design, so an unapproved
        # record digests the same as the approved one it was built alongside.
        # Say that plainly rather than reporting every field as disagreeing.
        raise LaunchEvidenceError("this record carries no operator approval to verify")
    # Comparing only the known keys would let an extra one ride along: the
    # content digest blanks this object entirely, so a holder of a valid record
    # could add "authorized_by": "Alice", recompute the whole-record digest, and
    # still verify. The approval must be exactly the ledger-backed schema.
    expected_fields = {field for field, _ in _APPROVAL_TO_LEDGER} | {"evidence_sha256"}
    disagreements = [
        f"{field}: not recorded in the ledger, so it is not part of any approval"
        for field in sorted(set(approval) - expected_fields)
    ]
    disagreements += [
        f"{field}: record={approval.get(field)!r} ledger={ledger_entry.get(key)!r}"
        for field, key in _APPROVAL_TO_LEDGER
        if approval.get(field) != ledger_entry.get(key)
    ]
    if approval.get("evidence_sha256") != ledger_entry.get("evidence_sha256"):
        disagreements.append(
            f"evidence_sha256: record={approval.get('evidence_sha256')!r} "
            f"ledger={ledger_entry.get('evidence_sha256')!r}"
        )
    # The ledger's own identifying fields, all three of them. Comparing only
    # some would let the entry name a different job than the record does while
    # still reporting success -- and it is the sole durable approval, so nothing
    # else would catch it. The job id is stringified because the ledger stores
    # text while a diagnostic may report the id as a number.
    identity = (
        ("stage", record.get("stage"), ledger_entry.get("stage")),
        ("package", _dig(record, ("destination", "package")), ledger_entry.get("package")),
        ("job_id", str(_dig(record, ("submission", "job_id"))), ledger_entry.get("job_id")),
    )
    for field, claimed, recorded in identity:
        if claimed != recorded:
            disagreements.append(f"{field}: record={claimed!r} ledger={recorded!r}")
    if disagreements:
        raise LaunchEvidenceError(
            "this record does not match the approval recorded for it: " + "; ".join(disagreements)
        )


def parse_json_artifact(text: str, *, label: str) -> dict[str, Any]:
    """Parse one artifact read off the cluster, naming it when it is not JSON.

    Parsed strictly: `json.loads` resolves a duplicate member name to the last
    value and says nothing, so an artifact carrying `attempt_id` twice would be
    bound on one interpretation while a reviewer opening the same bytes -- or
    any other parser -- could reasonably read the other. The record does not
    retain the original bytes, so that ambiguity would be unresolvable after the
    fact. The repository already has a decoder that refuses duplicates and
    non-finite numbers; this uses it rather than restating the rule.
    """

    try:
        return strict_json_object(text, label)
    except ValueError as error:
        raise LaunchEvidenceError(str(error)) from error


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
    "verify_launch_evidence",
    "MINTABLE_JOB_STATES",
    "SACCT_RETENTION_DAYS",
    "claimed_job_id",
    "parse_scheduler_record",
    "plan_digest",
    "required_package_files",
]
