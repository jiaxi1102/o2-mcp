"""Workspace layout conventions + disk hygiene for the O2 home/group/scratch tiers.

This module
governs the broader workspace — which storage tier each kind of artifact belongs
on, where per-project outputs go, and a classifier that flags regenerable caches,
redundant snapshot history, and misplaced data — so disk hygiene becomes a
repeatable ``o2_disk_report`` → ``o2_workspace_gc`` loop instead of a manual audit.

Tier contract (see ``docs/WORKSPACE_LAYOUT.md``):

- **home**   — code + config ONLY (backed up, small quota).
- **group**  — durable data, results, kept runs, the registry (backed up).
- **scratch**— ephemeral: active runs, staging, work, caches (auto-purged ~30 d).
- **standby**— cold archive (tarballs).

Pure, dependency-free, Python 3.9. The :class:`~o2mcp.workspace_exec.O2Workspace`
executor gathers the on-disk facts and runs the planned prunes over a connection.
No torch/cellpose/network imports — importable on the CPU-only core path.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from o2mcp.config import O2Config

# Storage tiers.
TIER_HOME = "home"
TIER_GROUP = "group"
TIER_SCRATCH = "scratch"
TIER_STANDBY = "standby"

# Disposition of a disk entry once classified.
KEEP = "keep"  # belongs where it is; leave alone
PRUNE_REGENERABLE = "prune_regenerable"  # caches/builds that rebuild on demand
PRUNE_REDUNDANT = "prune_redundant"  # trash / surplus snapshots / stale markers
RELOCATE_GROUP = "relocate_group"  # durable output sitting on the wrong tier (home)
REVIEW = "review"  # large/ambiguous; surface for a human decision

# Artifact kinds and the tier they belong on (the "paved road" for outputs).
# results/runs-kept/data live on durable group; work/active-runs/staging on scratch.
_KIND_TIER = {
    "results": TIER_GROUP,
    "data": TIER_GROUP,
    "runs_kept": TIER_GROUP,
    "registry": TIER_GROUP,
    "runs_active": TIER_SCRATCH,
    "work": TIER_SCRATCH,
    "staging": TIER_SCRATCH,
    "logs": TIER_GROUP,
    "archive": TIER_STANDBY,
}


@dataclass(frozen=True)
class WorkspaceLayout:
    """Resolve canonical workspace paths per tier/kind from an :class:`O2Config`."""

    home_root: str
    group_root: str
    scratch_root: str
    standby_root: str
    snapshot_keep: int = 2

    @classmethod
    def from_config(cls, config: O2Config) -> WorkspaceLayout:
        return cls(
            home_root=config.home_root,
            group_root=config.group_root,
            scratch_root=config.scratch_root,
            standby_root=config.standby_root,
            snapshot_keep=config.snapshot_keep,
        )

    def tier_root(self, tier: str) -> str:
        return {
            TIER_HOME: self.home_root,
            TIER_GROUP: self.group_root,
            TIER_SCRATCH: self.scratch_root,
            TIER_STANDBY: self.standby_root,
        }[tier]

    def place(self, kind: str, project: str | None = None) -> str:
        """Canonical absolute path for an artifact ``kind`` (optionally per project).

        ``place('results', 'myproject')`` →
        ``/n/groups/tabin/jzhao/results/myproject``. Agents and sbatch scripts
        should resolve output locations here instead of inventing paths.
        """
        if kind not in _KIND_TIER:
            raise ValueError(f"unknown artifact kind {kind!r}; known: {sorted(_KIND_TIER)}")
        root = self.tier_root(_KIND_TIER[kind])
        # registry is a file, runs roots are bare; everything else is <root>/<kind>[/<project>].
        if kind == "registry":
            return posixpath.join(root, "runs", "registry.jsonl")
        if kind in ("runs_active", "runs_kept"):
            base = posixpath.join(root, "runs")
        elif kind == "archive":
            base = posixpath.join(root, "runs_archive")
        elif kind == "data":
            base = posixpath.join(root, "data")
        else:
            base = posixpath.join(root, kind)
        return posixpath.join(base, _safe(project)) if project else base


def _safe(component: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(component).strip()).strip("-")
    return cleaned or "uncategorized"


# --- classification rules ----------------------------------------------------
# Each rule: (regex on the path, category label, disposition, reason). First match
# wins; order matters (specific KEEP rules before broad PRUNE rules).
_RULES: tuple[tuple[str, str, str, str], ...] = (
    # Keepers that look prunable but are load-bearing runtime state.
    (r"/\.cellpose/models$", "model-cache", KEEP, "segmentation weights; compute nodes may lack internet to re-fetch"),
    (r"/envs/[^/]+$", "conda-env", KEEP, "interpreter referenced by sbatch absolute paths"),
    (r"/\.ssh$", "ssh-config", KEEP, "credentials"),
    # Redundant: trash and surplus snapshot history.
    (r"(^|/)(legacy_)?trash(/|$)", "trash", PRUNE_REDUNDANT, "explicitly-named trash"),
    (r"/snapshots$", "snapshot-history", PRUNE_REDUNDANT, "timestamped DB/registry snapshots; keep latest few"),
    (r"_adoption_backups$", "db-backup", PRUNE_REDUNDANT, "one-time pre-adoption DB snapshots, superseded"),
    (r"\.pid(\.log)?$", "stale-marker", PRUNE_REDUNDANT, "stale process marker"),
    # Regenerable caches and build trees.
    (r"/\.cache(/|$)", "cache", PRUNE_REGENERABLE, "package/download cache; repopulates on demand"),
    (r"/(build|_build)$", "build-tree", PRUNE_REGENERABLE, "compiled build tree; rebuildable from source"),
    (r"/\.venv(-gpu)?$", "venv", PRUNE_REGENERABLE, "virtualenv; rebuildable from requirements"),
    (r"/(vendor|node_modules)$", "vendored-deps", PRUNE_REGENERABLE, "vendored dependencies; re-fetchable"),
    (r"/(src|opt)/[^/]*(eigen|hoomd)", "downloaded-source", PRUNE_REGENERABLE, "downloaded upstream source"),
    # Misplaced durable output: results materialized in home.
    (
        r"^/home/[^/]+/results(/|$)",
        "results-in-home",
        RELOCATE_GROUP,
        "durable results on backed-up home; belongs on group",
    ),
)


def classify_entry(
    path: str, *, tier: str = "", size_bytes: int = 0, review_threshold_bytes: int = 5 * 1024**3
) -> dict[str, Any]:
    """Classify one disk entry into a category + disposition + reason.

    Pure path-based heuristic (plus a size threshold for the REVIEW fallback). The
    caller supplies the absolute ``path`` and (optionally) its ``size_bytes``.
    Anything unmatched and large is flagged REVIEW; small unmatched entries KEEP.
    """
    p = path.rstrip("/")
    for pattern, category, disposition, reason in _RULES:
        if re.search(pattern, p):
            return {"path": path, "category": category, "disposition": disposition, "reason": reason}
    if size_bytes >= review_threshold_bytes:
        return {
            "path": path,
            "category": "large-unclassified",
            "disposition": REVIEW,
            "reason": f"{_human(size_bytes)} and not a known category; review placement",
        }
    return {"path": path, "category": "other", "disposition": KEEP, "reason": "small / unrecognized; left in place"}


# --- report aggregation ------------------------------------------------------
def summarize_report(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate classified entries: per-disposition totals + reclaimable estimate.

    ``entries`` are dicts with at least ``disposition`` and ``size_bytes``.
    Reclaimable = bytes under PRUNE_* dispositions (safe to delete).
    """
    by_disposition: dict[str, dict[str, Any]] = {}
    reclaimable = 0
    for entry in entries:
        disp = entry.get("disposition", KEEP)
        size = int(entry.get("size_bytes", 0) or 0)
        bucket = by_disposition.setdefault(disp, {"count": 0, "bytes": 0, "items": []})
        bucket["count"] += 1
        bucket["bytes"] += size
        bucket["items"].append({"path": entry.get("path"), "size_bytes": size, "reason": entry.get("reason")})
        if disp in (PRUNE_REGENERABLE, PRUNE_REDUNDANT):
            reclaimable += size
    for bucket in by_disposition.values():
        bucket["human"] = _human(bucket["bytes"])
        bucket["items"].sort(key=lambda i: i["size_bytes"], reverse=True)
    return {
        "by_disposition": by_disposition,
        "reclaimable_bytes": reclaimable,
        "reclaimable_human": _human(reclaimable),
    }


# --- prune planner (pure: build shell, never execute) ------------------------
def plan_prune(entries: Sequence[dict[str, Any]], layout: WorkspaceLayout) -> str:
    """A fail-closed bash script that deletes only the PRUNE_* entries.

    Snapshot-history directories are thinned to the newest ``snapshot_keep`` files
    rather than removed wholesale; everything else under PRUNE_REGENERABLE /
    PRUNE_REDUNDANT is ``rm -rf``'d. RELOCATE/REVIEW/KEEP entries are never touched.
    """
    lines = [
        "#!/bin/bash",
        "# o2 workspace gc — prune regenerable + redundant only",
        "set -uo pipefail",
        'log(){ echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }',
        "freed=0",
    ]
    for entry in entries:
        disp = entry.get("disposition")
        path = entry.get("path", "")
        if disp not in (PRUNE_REGENERABLE, PRUNE_REDUNDANT) or not path:
            continue
        q = shlex.quote(path)
        if entry.get("category") == "snapshot-history":
            keep = max(0, int(layout.snapshot_keep))
            # Keep the newest N entries by mtime; delete the rest. Never touch non-snapshot dirs.
            lines.append(
                f"if [ -d {q} ]; then n=$(ls -1 {q} | wc -l); "
                f'ls -1t {q} | tail -n +{keep + 1} | while IFS= read -r f; do rm -rf {q}/"$f"; done; '
                f'log "PRUNED snapshots in {path} (kept newest {keep} of $n)"; fi'
            )
        else:
            lines.append(
                f"if [ -e {q} ]; then sz=$(du -sb {q} 2>/dev/null | cut -f1); rm -rf {q} && "
                f'{{ freed=$((freed + ${{sz:-0}})); log "PRUNED {path} (${{sz:-0}} B)"; }}; fi'
            )
    lines.append('log "=== gc done; freed ~$((freed / 1024 / 1024)) MiB of standalone entries ==="')
    return "\n".join(lines)


# --- helpers -----------------------------------------------------------------
def _human(num_bytes: int) -> str:
    value = float(max(0, int(num_bytes)))
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}T"
