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
_RUN_ID_RE = re.compile(r"^RUN_(?P<ts>\d{8}T\d{0,6}Z?)_(?P<slug>.+)$")


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
    return cleaned or "uncategorized"


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

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        data = json.loads(text)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunManifest:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

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
    """Commands to create an active run dir (+ optional project subdirs) and drop ``run.json``."""
    run_dir = layout.run_dir(STATUS_ACTIVE, manifest.campaign, manifest.run_id)
    quoted = shlex.quote(run_dir)
    targets = " ".join([quoted, *(shlex.quote(posixpath.join(run_dir, d)) for d in run_subdirs)])
    manifest_path = shlex.quote(posixpath.join(run_dir, "run.json"))
    heredoc = _heredoc(manifest.to_json())
    return [f"mkdir -p {targets}", f"cat > {manifest_path} {heredoc}", f"printf '%s\\n' {quoted}"]


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
    parent = posixpath.dirname(source_dir.rstrip("/"))
    base = posixpath.basename(source_dir.rstrip("/"))
    exclude = " ".join(f"--exclude={shlex.quote(e)}" for e in archive_excludes)
    return "\n".join(
        [
            "#!/bin/bash",
            f"# archive {manifest.run_id} -> standby (cold, tar.zst)",
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(posixpath.dirname(tarball))}",
            f"cat > {shlex.quote(manifest_dest)} {_heredoc(archived.to_json())}",
            f"tar {exclude} --use-compress-program='zstd -19 --long=27 -T0' "
            f"-cf {shlex.quote(tarball)} -C {shlex.quote(parent)} {shlex.quote(base)}",
            f"sha256sum {shlex.quote(tarball)} > {shlex.quote(checksum)}",
            f"test -s {shlex.quote(tarball)}",
            f"zstd -t {shlex.quote(tarball)}",  # integrity-test the archive before deleting source
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
    clone = RunManifest.from_dict(asdict(manifest))
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
    src_slash = shlex.quote(source_dir.rstrip("/") + "/")
    dest_slash = shlex.quote(dest_dir.rstrip("/") + "/")
    exclude = " ".join(f"--exclude={shlex.quote(e)}" for e in excludes if e)
    rsync = f"rsync -a {exclude} {src_slash} {dest_slash}".replace("   ", " ")
    verify = f"rsync -ni -a {exclude} {src_slash} {dest_slash}".replace("   ", " ")
    manifest_path = shlex.quote(posixpath.join(dest_dir, manifest_rel))
    lines = [
        "#!/bin/bash",
        f"# {title}",
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(dest_dir)}",
        rsync,
        f"cat > {manifest_path} {_heredoc(manifest_json)}",
        # verify: a dry-run itemize must show nothing left to transfer
        f"test -z \"$({verify} | grep -v '^$' || true)\"",
        f"echo COPIED {shlex.quote(dest_dir)}",
    ]
    if free_source:
        lines += [f"rm -rf {shlex.quote(source_dir.rstrip('/'))}", "echo FREED_SCRATCH"]
    return "\n".join(lines)
