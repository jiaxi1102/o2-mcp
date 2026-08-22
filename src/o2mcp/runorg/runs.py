"""Run-organization conventions: layout, manifest, classifier, registry, planners.

A *run* is one self-describing, relocatable unit with an explicit lifecycle that
maps onto the three O2 storage tiers::

    active ───promote──▶ kept ───archive──▶ archived ───gc──▶ purged
   (scratch)            (group)            (standby)         (tombstone)

This module is the pure, dependency-free, Python-3.9 core of that system: the
:class:`RunLayout` path conventions, the canonical :class:`RunManifest`
(``run.json``) schema plus back-compat readers for the legacy metadata formats,
a heuristic :func:`classify_run`, registry (JSONL) helpers, and the command
*planners* that emit the exact shell sequences for promote/archive/gc. Everything
here is testable offline; the :class:`~o2mcp.runorg.executor.O2Runs` executor runs
the planned commands over an :class:`~o2mcp.connection.O2Connection`.

No torch/cellpose/network imports — importable on the CPU-only core path.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from o2mcp.config import O2Config
from o2mcp.runorg.policy import RunPolicy

SCHEMA_VERSION = 1

# Lifecycle states (kept as plain strings so the manifest JSON is trivial).
STATUS_ACTIVE = "active"
STATUS_KEPT = "kept"
STATUS_ARCHIVED = "archived"
STATUS_PURGED = "purged"
VALID_STATUSES = (STATUS_ACTIVE, STATUS_KEPT, STATUS_ARCHIVED, STATUS_PURGED)

# Retention intent (drives promote vs archive vs sweep during migration/gc).
RETENTION_KEEP = "keep"  # promote to durable group storage
RETENTION_SWEEP = "sweep"  # archive cold to standby, then free scratch
RETENTION_AUTO = "auto"  # let gc decide by age

# Fields a caller MUST supply to register a run (no unclassified runs).
REQUIRED_FOR_REGISTER = ("campaign", "pipeline", "datasets")

# Marker word-lists, view-suffixes and the heavy-suffix threshold are project-specific
# and come from a RunPolicy (see o2mcp.runorg.policy); the generic engine holds none.
_RUN_ID_RE = re.compile(r"^RUN_(?P<ts>\d{8}T\d{6}Z)_(?P<slug>.+)$")


# --- layout ------------------------------------------------------------------
@dataclass(frozen=True)
class RunLayout:
    """Resolve canonical run paths for each tier from an :class:`O2Config`."""

    scratch_runs_root: str
    group_runs_root: str
    standby_archive_root: str
    registry_path: str

    @classmethod
    def from_config(cls, config: O2Config) -> RunLayout:
        return cls(
            scratch_runs_root=config.scratch_runs_root,
            group_runs_root=config.group_runs_root,
            standby_archive_root=config.standby_archive_root,
            registry_path=config.registry_path,
        )

    def tier_root(self, status: str) -> str:
        if status == STATUS_KEPT:
            return self.group_runs_root
        if status == STATUS_ARCHIVED:
            return self.standby_archive_root
        return self.scratch_runs_root

    def run_dir(self, status: str, campaign: str, run_id: str) -> str:
        """Directory for a live run (active/kept). Campaign-grouped, never flat."""
        return posixpath.join(self.tier_root(status), _safe(campaign), run_id)

    def archive_tarball(self, campaign: str, run_id: str) -> str:
        return posixpath.join(self.standby_archive_root, _safe(campaign), run_id + ".tar.zst")

    def archive_manifest(self, campaign: str, run_id: str) -> str:
        """The run.json kept UNcompressed beside the tarball so it stays queryable."""
        return posixpath.join(self.standby_archive_root, _safe(campaign), run_id + ".run.json")

    def archive_checksum(self, campaign: str, run_id: str) -> str:
        return posixpath.join(self.standby_archive_root, _safe(campaign), run_id + ".tar.zst.sha256")


def _safe(component: str) -> str:
    """A filesystem-safe path component (campaigns become kebab-ish slugs)."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(component).strip()).strip("-")
    # A literal dot component survives the character filter but still has path
    # traversal semantics.  Treat it as absent rather than letting a campaign
    # escape the configured storage root (issue #3).
    return cleaned if cleaned and cleaned not in {".", ".."} else "uncategorized"


def sort_job_ids(ids: Iterable[str]) -> list[str]:
    """Deduplicate and sort Slurm job ids numerically.

    Slurm job ids are integers rendered as strings, so a plain ``sorted`` orders
    them lexicographically ("100" before "88"). Sort by integer value instead. A
    stray non-numeric id is tolerated and ordered (lexically) after the numeric
    ones rather than raising, so a legacy/synthesized manifest never breaks.
    """
    return sorted(set(ids), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))


# --- manifest ----------------------------------------------------------------
@dataclass
class RunManifest:
    """The canonical ``run.json`` record for one run (supersedes legacy formats)."""

    run_id: str
    campaign: str
    pipeline: str
    created_utc: str
    status: str = STATUS_ACTIVE
    variant: str = ""
    datasets: list[str] = field(default_factory=list)
    experiment_ids: list[str] = field(default_factory=list)
    slurm_job_ids: list[str] = field(default_factory=list)
    source_view: dict[str, Any] = field(default_factory=dict)
    storage_roots: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    retention: str = RETENTION_AUTO
    size_bytes: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    tombstone: dict[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION
    # Forward-compatible fields are preserved verbatim when an older o2-mcp
    # rewrites run.json during execution reconciliation.  They are deliberately
    # excluded from the dataclass's normal field namespace and merged by
    # :meth:`to_dict` so a future schema cannot be silently truncated.
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def validate(self, *, for_register: bool = False) -> list[str]:
        """Return a list of human-readable problems (empty == valid)."""
        problems: list[str] = []
        if not _RUN_ID_RE.match(self.run_id):
            problems.append(f"run_id {self.run_id!r} does not match RUN_<UTCtimestamp>Z_<slug>")
        if self.status not in VALID_STATUSES:
            problems.append(f"status {self.status!r} not in {VALID_STATUSES}")
        if for_register:
            if not self.campaign:
                problems.append("campaign is required")
            if not self.pipeline:
                problems.append("pipeline is required")
            if not self.datasets:
                problems.append("at least one dataset is required")
        return problems

    def to_dict(self) -> dict[str, Any]:
        """Return a lossless manifest mapping, including unknown input fields."""

        data = asdict(self)
        extra = data.pop("extra")
        conflicts = sorted(set(extra) & set(data))
        if conflicts:
            raise ValueError(f"manifest extra fields conflict with known fields: {conflicts}")
        data.update(extra)
        return data

    def to_json(self) -> str:
        """Serialize the lossless manifest in a stable, review-friendly form."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        data = json.loads(text)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunManifest:
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}  # type: ignore[attr-defined]
        values = {k: v for k, v in data.items() if k in known}
        values["extra"] = {k: v for k, v in data.items() if k not in known and k != "extra"}
        return cls(**values)

    def registry_row(self) -> dict[str, Any]:
        """The compact one-line summary stored in the registry JSONL."""
        return {
            "run_id": self.run_id,
            "campaign": self.campaign,
            "variant": self.variant,
            "pipeline": self.pipeline,
            "status": self.status,
            "created_utc": self.created_utc,
            "datasets": list(self.datasets),
            "experiment_ids": list(self.experiment_ids),
            "slurm_job_ids": list(self.slurm_job_ids),
            "retention": self.retention,
            "size_bytes": self.size_bytes,
            "result_status": (self.result or {}).get("status"),
        }


# --- run-id parsing ----------------------------------------------------------
def parse_run_id(run_id: str) -> dict[str, str]:
    """Split a run id into ``timestamp``/``slug`` (best-effort; never raises)."""
    match = _RUN_ID_RE.match(run_id)
    if not match:
        return {"timestamp": "", "slug": run_id}
    return {"timestamp": match.group("ts"), "slug": match.group("slug")}


def campaign_of(run_id: str, view_suffixes: Sequence[str] = ()) -> str:
    """Derive a campaign key from a run id by stripping the timestamp and a trailing
    view-suffix (from ``view_suffixes``).

    New runs use an explicit ``campaign__variant`` slug, so the campaign is the part
    before ``__``. Legacy runs have no separator: strip a trailing project view-suffix
    so e.g. ``RUN_..._gpd0524_grid_singlecell_maps`` and its ``..._period_qc_diagnostics``
    sibling both fold to ``gpd0524_grid_singlecell``.
    """
    slug = parse_run_id(run_id)["slug"]
    if "__" in slug:
        return slug.split("__", 1)[0].strip("_") or "uncategorized"
    for suffix in sorted(view_suffixes, key=len, reverse=True):
        token = "_" + suffix
        if slug.endswith(token):
            slug = slug[: -len(token)]
            break
    return slug.strip("_") or "uncategorized"


def variant_of(run_id: str, view_suffixes: Sequence[str] = ()) -> str:
    """The variant segment of a run id (after ``__``, or after the campaign prefix)."""
    slug = parse_run_id(run_id)["slug"]
    if "__" in slug:
        return slug.split("__", 1)[1].strip("_")
    campaign = campaign_of(run_id, view_suffixes)
    return slug[len(campaign) :].strip("_") if slug.startswith(campaign) else ""


# --- legacy metadata readers -------------------------------------------------
def parse_submission_env(text: str) -> dict[str, Any]:
    """Parse a legacy ``submission_summary.env`` into manifest-relevant fields."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line or " " in line.split("=", 1)[0]:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    out: dict[str, Any] = {}
    storage_roots = sorted({v for k, v in fields.items() if k.endswith("_STORAGE_ROOT")})
    if "DEFAULT_STORAGE_ROOT" in fields:
        storage_roots = sorted(set(storage_roots) | {fields["DEFAULT_STORAGE_ROOT"]})
    if storage_roots:
        out["storage_roots"] = storage_roots
    job_ids: list[str] = []
    for key, value in fields.items():
        if key.endswith("_JOB_IDS"):
            job_ids.extend(part for part in re.split(r"[,\s]+", value) if part)
    if job_ids:
        out["slurm_job_ids"] = sort_job_ids(job_ids)
    return out


def merge_status_json(text: str) -> dict[str, Any]:
    """Pull dataset/experiment/status/jobs out of a legacy ``*_status.json`` blob.

    The status directory may concatenate several JSON objects; we tolerate that by
    scanning for top-level objects and merging their fields.
    """
    out: dict[str, Any] = {"datasets": [], "experiment_ids": [], "slurm_job_ids": [], "result_status": None}
    for obj in _iter_json_objects(text):
        if not isinstance(obj, dict):
            continue
        for src, dst in (
            ("dataset_name", "datasets"),
            ("experiment_id", "experiment_ids"),
            ("slurm_job_id", "slurm_job_ids"),
        ):
            value = obj.get(src)
            if value and str(value) not in out[dst]:
                out[dst].append(str(value))
        if obj.get("status"):
            out["result_status"] = str(obj["status"])
    return out


def _iter_json_objects(text: str):
    """Yield successive top-level JSON values from a possibly-concatenated string."""
    decoder = json.JSONDecoder()
    index, length = 0, len(text)
    while index < length:
        while index < length and text[index] in " \t\r\n":
            index += 1
        if index >= length:
            return
        try:
            obj, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return
        yield obj
        index = end


# --- classifier --------------------------------------------------------------
def classify_run(
    run_id: str, policy: RunPolicy, *, result_status: str | None = None, is_latest_in_campaign: bool = False
) -> dict[str, str]:
    """Heuristically tag a run ``keep`` (→ group) or ``sweep`` (→ standby).

    Precedence (first match wins): a ``policy.keep_markers`` substring in the slug →
    ``keep``; a ``policy.sweep_markers`` debug/iteration marker → ``sweep`` (debug stays
    debug even when it is the only/last run of its line); otherwise the latest COMPLETED
    variant of its campaign → ``keep``; everything else defaults to ``sweep``. Returns
    ``{retention, reason}`` — always advisory; the caller reviews before anything moves.
    """
    slug = parse_run_id(run_id)["slug"].lower()
    for marker in policy.keep_markers:
        if marker in slug:
            return {"retention": RETENTION_KEEP, "reason": f"keep-marker '{marker}' in name"}
    for marker in policy.sweep_markers:
        if marker in slug:
            return {"retention": RETENTION_SWEEP, "reason": f"debug/iteration marker '{marker}' in name"}
    if is_latest_in_campaign and (result_status or "").upper() == "COMPLETED":
        return {"retention": RETENTION_KEEP, "reason": "latest COMPLETED variant of its campaign"}
    if (result_status or "").upper() in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
        return {"retention": RETENTION_SWEEP, "reason": f"non-success result '{result_status}'"}
    return {"retention": RETENTION_SWEEP, "reason": "no keep-marker; not latest-in-campaign (default sweep)"}


def is_regenerable_intermediate(run_id: str, policy: RunPolicy, size_bytes: int | None = None) -> bool:
    """Whether a run is a heavy, regenerable intermediate (heavy view / marker / huge).

    Such a run is archived cold to standby even when its name carries a keep marker —
    the pipeline can recreate it from raw data, so it does not earn durable (backed-up)
    group storage. Driven by ``policy.heavy_view_suffixes`` (slug endswith),
    ``policy.heavy_slug_markers`` (substring), and ``policy.heavy_threshold_bytes`` (size).
    """
    slug = parse_run_id(run_id)["slug"].lower()
    if any(slug.endswith("_" + suffix) for suffix in policy.heavy_view_suffixes):
        return True
    if any(marker in slug for marker in policy.heavy_slug_markers):
        return True
    return size_bytes is not None and size_bytes > policy.heavy_threshold_bytes


def migration_target(
    run_id: str,
    policy: RunPolicy,
    *,
    result_status: str | None = None,
    is_latest_in_campaign: bool = False,
    size_bytes: int | None = None,
) -> dict[str, str]:
    """Decide ``promote`` (→ group) vs ``archive`` (→ standby) for one run.

    A run is promoted only when it is classified ``keep`` AND is not a heavy
    regenerable intermediate; everything else is archived cold. Returns
    ``{target, reason}``.
    """
    verdict = classify_run(run_id, policy, result_status=result_status, is_latest_in_campaign=is_latest_in_campaign)
    if verdict["retention"] == RETENTION_KEEP and not is_regenerable_intermediate(run_id, policy, size_bytes):
        return {"target": "promote", "reason": verdict["reason"]}
    if verdict["retention"] == RETENTION_KEEP:
        return {"target": "archive", "reason": "keep-marked but heavy regenerable intermediate (archive, not group)"}
    return {"target": "archive", "reason": verdict["reason"]}


# --- registry ----------------------------------------------------------------
def registry_line(manifest: RunManifest) -> str:
    """One compact JSON line (no newlines inside) for the append-only registry."""
    return json.dumps(manifest.registry_row(), sort_keys=True)


def parse_registry(text: str) -> list[dict[str, Any]]:
    """Parse a registry JSONL blob; last row per run_id wins (append-only updates)."""
    by_id: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("run_id"):
            by_id[row["run_id"]] = row
    return list(by_id.values())


# --- command planners (pure: build shell, never execute) ---------------------
def plan_register_commands(layout: RunLayout, manifest: RunManifest, run_subdirs: Sequence[str] = ()) -> list[str]:
    """Create one run directory exclusively, then seed its ``run.json``.

    The run root itself uses plain ``mkdir`` rather than ``mkdir -p``.  A run ID
    is an execution identity, so reusing an existing directory must fail instead
    of overwriting its manifest and mixing two submissions.
    """

    run_dir = layout.run_dir(STATUS_ACTIVE, manifest.campaign, manifest.run_id)
    quoted = shlex.quote(run_dir)
    parent = shlex.quote(posixpath.dirname(run_dir))
    subdirs = " ".join(shlex.quote(posixpath.join(run_dir, d)) for d in run_subdirs)
    manifest_path = shlex.quote(posixpath.join(run_dir, "run.json"))
    heredoc = _heredoc(manifest.to_json())
    commands = [f"mkdir -p {parent} && mkdir {quoted}"]
    if subdirs:
        commands.append(f"mkdir -p {subdirs}")
    commands.extend([f"cat > {manifest_path} {heredoc}", f"printf '%s\\n' {quoted}"])
    return commands


def plan_write_manifest_command(run_dir: str, manifest: RunManifest) -> str:
    """A single command that (re)writes ``run.json`` for an existing run dir.

    Used to persist an in-place manifest update (e.g. recording a freshly
    submitted Slurm job id) without touching the rest of the run skeleton.
    """
    manifest_path = shlex.quote(posixpath.join(run_dir, "run.json"))
    return f"cat > {manifest_path} {_heredoc(manifest.to_json())}"


def plan_promote_script(layout: RunLayout, manifest: RunManifest, *, source_dir: str) -> str:
    """A bash script that copies an active run to durable group storage (verified),
    flips its manifest to ``kept``, and then frees the scratch copy.

    ``rsync --remove-source-files`` is intentionally *not* used; we verify the copy
    with a second ``rsync -ni`` (must report no differences) before deleting source.
    """
    dest = layout.run_dir(STATUS_KEPT, manifest.campaign, manifest.run_id)
    if posixpath.normpath(source_dir) == posixpath.normpath(dest):
        raise ValueError("promotion source and durable destination must be different paths")
    kept = _with_status(manifest, STATUS_KEPT)
    return _render_transfer_script(
        title=f"promote {manifest.run_id} -> group (kept)",
        source_dir=source_dir,
        dest_dir=dest,
        manifest_json=kept.to_json(),
        manifest_rel="run.json",
        excludes=[],
        free_source=True,
    )


def plan_archive_script(
    layout: RunLayout,
    manifest: RunManifest,
    *,
    source_dir: str,
    archive_excludes: Sequence[str] = (),
) -> str:
    """A bash script that tars+zstd-compresses a run to standby, checksums it, writes
    the manifest beside it (uncompressed, queryable), verifies, then frees scratch.

    ``archive_excludes`` (from the RunPolicy) lists top-level paths omitted from the
    tarball — e.g. redundant source copies recorded elsewhere in the manifest.
    """
    archived = _with_status(manifest, STATUS_ARCHIVED)
    tarball = layout.archive_tarball(manifest.campaign, manifest.run_id)
    manifest_dest = layout.archive_manifest(manifest.campaign, manifest.run_id)
    checksum = layout.archive_checksum(manifest.campaign, manifest.run_id)
    archive_parent = posixpath.dirname(tarball)
    staging_template = posixpath.join(archive_parent, "." + manifest.run_id + ".archive.XXXXXX")
    parent = posixpath.dirname(source_dir.rstrip("/"))
    base = posixpath.basename(source_dir.rstrip("/"))
    exclude = " ".join(f"--exclude={shlex.quote(e)}" for e in (*archive_excludes, ".execution-source.lock"))
    manifest_json = archived.to_json()
    return "\n".join(
        [
            "#!/bin/bash",
            f"# archive {manifest.run_id} -> standby (cold, tar.zst)",
            "set -euo pipefail",
            f"exec 8> {shlex.quote(posixpath.join(source_dir, '.execution-source.lock'))}",
            "flock -x 8",
            _source_snapshot_assignment(source_dir, "source_baseline_sha"),
            f"mkdir -p {shlex.quote(archive_parent)}",
            f"exec 9> {shlex.quote(posixpath.join(archive_parent, '.' + manifest.run_id + '.archive.lock'))}",
            "flock -x 9",
            # The manifest is the publication commit marker and is moved last.
            # A previous complete or partial archive is never overwritten.  A
            # partial package requires explicit inspection/cleanup, while the
            # source remains intact because deletion happens only after commit.
            f"if test -e {shlex.quote(manifest_dest)} || test -e {shlex.quote(tarball)} || "
            f"test -e {shlex.quote(checksum)}; then echo 'archive destination already exists' >&2; exit 76; fi",
            f"staging=$(mktemp -d {shlex.quote(staging_template)})",
            "trap 'rm -rf -- \"$staging\"' EXIT",
            f"tar {exclude} --use-compress-program='zstd -19 --long=27 -T0' "
            f'-cf "$staging/archive.tar.zst" -C {shlex.quote(parent)} {shlex.quote(base)}',
            "archive_sha=$(sha256sum \"$staging/archive.tar.zst\" | cut -d' ' -f1)",
            f'printf \'%s  %s\\n\' "$archive_sha" {shlex.quote(tarball)} > "$staging/archive.sha256"',
            'test -s "$staging/archive.tar.zst"',
            'zstd -t "$staging/archive.tar.zst"',
            f'cat > "$staging/run.json" {_heredoc(manifest_json)}',
            # Publish data first and the small manifest last.  Consumers must
            # treat the manifest as the commit marker for the three-file package.
            f'mv --no-clobber -- "$staging/archive.tar.zst" {shlex.quote(tarball)}',
            'test ! -e "$staging/archive.tar.zst"',
            f'mv --no-clobber -- "$staging/archive.sha256" {shlex.quote(checksum)}',
            'test ! -e "$staging/archive.sha256"',
            f'mv --no-clobber -- "$staging/run.json" {shlex.quote(manifest_dest)}',
            'test ! -e "$staging/run.json"',
            'rmdir -- "$staging"',
            "trap - EXIT",
            f"test \"$(sha256sum {shlex.quote(manifest_dest)} | cut -d' ' -f1)\" = "
            f"{shlex.quote(hashlib.sha256((manifest_json + chr(10)).encode()).hexdigest())}",
            _source_snapshot_assignment(source_dir, "source_final_sha"),
            'if test "$source_final_sha" != "$source_baseline_sha"; then '
            "echo 'source changed during archive; refusing deletion' >&2; exit 77; fi",
            f"echo ARCHIVED {shlex.quote(tarball)}",
            f"rm -rf {shlex.quote(source_dir.rstrip('/'))}",
            "echo FREED_SCRATCH",
        ]
    )


def plan_gc_candidates_command(layout: RunLayout, *, older_than_days: int) -> str:
    """Command listing active scratch runs whose tree is older than N days (gc audit)."""
    root = shlex.quote(layout.scratch_runs_root)
    return (
        f"find {root} -mindepth 2 -maxdepth 2 -type d -mtime +{int(older_than_days)} "
        f"-printf '%TY-%Tm-%Td\\t%p\\n' 2>/dev/null | sort"
    )


# --- helpers -----------------------------------------------------------------
def _with_status(manifest: RunManifest, status: str) -> RunManifest:
    clone = RunManifest.from_dict(manifest.to_dict())
    clone.status = status
    return clone


def _heredoc(body: str) -> str:
    """A quoted heredoc that writes ``body`` literally (no shell expansion)."""
    return "<<'__RUN_JSON__'\n" + body + "\n__RUN_JSON__"


def _render_transfer_script(
    *,
    title: str,
    source_dir: str,
    dest_dir: str,
    manifest_json: str,
    manifest_rel: str,
    excludes: Sequence[str],
    free_source: bool,
) -> str:
    if posixpath.normpath(source_dir) == posixpath.normpath(dest_dir):
        raise ValueError("transfer source and destination must be different paths")
    src_slash = shlex.quote(source_dir.rstrip("/") + "/")
    dest_parent = posixpath.dirname(dest_dir)
    effective_excludes = tuple(item for item in excludes if item) + (".execution-source.lock",)
    exclude = " ".join(f"--exclude={shlex.quote(e)}" for e in effective_excludes)
    rsync = f'rsync -a {exclude} {src_slash} "$staging/"'.replace("   ", " ")
    verify_forward = f'rsync -nric --delete -a {exclude} {src_slash} "$staging/"'.replace("   ", " ")
    verify_reverse = f'rsync -nric --delete -a {exclude} "$staging/" {src_slash}'.replace("   ", " ")
    # ``_heredoc`` terminates the final JSON line with LF; authenticate the
    # actual published bytes rather than the pre-heredoc Python string.
    manifest_sha = hashlib.sha256((manifest_json + "\n").encode()).hexdigest()
    staging_template = posixpath.join(
        dest_parent,
        "." + posixpath.basename(dest_dir) + ".promote.XXXXXX",
    )
    lines = [
        "#!/bin/bash",
        f"# {title}",
        "set -euo pipefail",
        f"exec 8> {shlex.quote(posixpath.join(source_dir, '.execution-source.lock'))}",
        "flock -x 8",
        _source_snapshot_assignment(source_dir, "source_baseline_sha"),
        f"mkdir -p {shlex.quote(dest_parent)}",
        f"exec 9> {shlex.quote(posixpath.join(dest_parent, '.' + posixpath.basename(dest_dir) + '.promote.lock'))}",
        "flock -x 9",
        # Never merge into an old or partial durable tree.  A sibling staging
        # directory is verified exactly and then renamed into an absent final path.
        f"if test -e {shlex.quote(dest_dir)}; then echo 'durable destination already exists' >&2; exit 76; fi",
        f"staging=$(mktemp -d {shlex.quote(staging_template)})",
        "trap 'rm -rf -- \"$staging\"' EXIT",
        rsync,
        "verify_output=$(mktemp)",
        'trap \'rm -f "$verify_output"; rm -rf -- "$staging"\' EXIT',
        # Verify both directions so destination-only residue is impossible.  The
        # lifecycle manifest is changed only after the copied active tree passes.
        f'if ! {verify_forward} >"$verify_output"; then cat "$verify_output" >&2; exit 74; fi',
        'if test -s "$verify_output"; then cat "$verify_output" >&2; exit 75; fi',
        f'if ! {verify_reverse} >"$verify_output"; then cat "$verify_output" >&2; exit 74; fi',
        'if test -s "$verify_output"; then cat "$verify_output" >&2; exit 75; fi',
        'rm -f "$verify_output"',
        f'cat > "$staging/{manifest_rel}" {_heredoc(manifest_json)}',
        f'test "$(sha256sum "$staging/{manifest_rel}" | cut -d\' \' -f1)" = {shlex.quote(manifest_sha)}',
        f'mv --no-clobber -T -- "$staging" {shlex.quote(dest_dir)}',
        'test ! -e "$staging"',
        "trap - EXIT",
        _source_snapshot_assignment(source_dir, "source_final_sha"),
        'if test "$source_final_sha" != "$source_baseline_sha"; then '
        "echo 'source changed during promotion; refusing deletion' >&2; exit 77; fi",
        f"echo COPIED {shlex.quote(dest_dir)}",
    ]
    if free_source:
        lines += [f"rm -rf {shlex.quote(source_dir.rstrip('/'))}", "echo FREED_SCRATCH"]
    return "\n".join(lines)


def _source_snapshot_assignment(source_dir: str, variable: str) -> str:
    """Render a deterministic source-tree digest used before destructive cleanup.

    The lifecycle lock itself is excluded because opening it is coordination,
    not scientific data.  The dependency-free Python walker hashes relative
    names, modes, types, symlink targets, and regular-file bytes without relying
    on platform-specific GNU tar flags.
    """

    program = "\n".join(
        [
            "import hashlib, os, stat, sys",
            "root = os.path.abspath(sys.argv[1])",
            "digest = hashlib.sha256()",
            "def visit(path, relative):",
            "    info = os.lstat(path)",
            "    mode = info.st_mode",
            "    digest.update(relative.encode('utf-8') + b'\\0' + str(stat.S_IMODE(mode)).encode() + b'\\0')",
            "    if stat.S_ISDIR(mode):",
            "        digest.update(b'd\\0')",
            "        for name in sorted(os.listdir(path)):",
            "            if relative == '.' and name == '.execution-source.lock':",
            "                continue",
            "            child_relative = name if relative == '.' else relative + '/' + name",
            "            visit(os.path.join(path, name), child_relative)",
            "    elif stat.S_ISREG(mode):",
            "        digest.update(b'f\\0')",
            "        with open(path, 'rb') as handle:",
            "            for chunk in iter(lambda: handle.read(1024 * 1024), b''):",
            "                digest.update(chunk)",
            "    elif stat.S_ISLNK(mode):",
            "        digest.update(b'l\\0' + os.readlink(path).encode('utf-8') + b'\\0')",
            "    else:",
            "        raise SystemExit('unsupported special file in transition source: ' + relative)",
            "visit(root, '.')",
            "print(digest.hexdigest())",
        ]
    )
    return f"{variable}=$(python3 -c {shlex.quote(program)} {shlex.quote(source_dir.rstrip('/'))})"
