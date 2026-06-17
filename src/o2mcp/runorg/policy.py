"""Project-specific run taxonomy/policy for the generic run-organization engine.

The run lifecycle (register → promote/archive across scratch→group→standby, the
JSONL registry, keep/sweep classification) is generic; the *taxonomy* — which
pipelines exist, how a run slug maps to a pipeline, which name-markers mean
"keep" vs "sweep", which suffixes are heavy/regenerable, the run-skeleton
subdirs, and what the archive excludes — is project-specific and lives here.

Each consumer (e.g. clock, GEM) defines its own :class:`RunPolicy`; the engine
holds one and threads it through classification, inference, and the planners.
Storage roots/registry path are NOT here — they stay in :class:`o2mcp.O2Config`
(env-driven, set per consumer so projects don't share a registry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Generic keep-markers: a run whose slug contains one of these is durable by intent
# regardless of project. Sensible default; a consumer may override.
DEFAULT_KEEP_MARKERS: tuple[str, ...] = ("official", "production", "recommended", "winner", "final", "release")

# Generic slug substrings that mark a run as a heavy/regenerable intermediate.
DEFAULT_HEAVY_SLUG_MARKERS: tuple[str, ...] = ("sweep",)

DEFAULT_HEAVY_THRESHOLD_BYTES: int = 2 * 1024**3  # 2 GiB


@dataclass(frozen=True)
class RunPolicy:
    """The project-specific run taxonomy threaded through the run-org engine."""

    # Ordered (slug-substring → pipeline) rules; first match wins. Empty = no inference.
    pipeline_keywords: tuple[tuple[str, str], ...] = ()
    # Returned by inference when no keyword matches.
    fallback_pipeline: str = "unknown"
    # Optional explicit catalog of valid pipelines (for help/validation). Empty = accept any.
    pipelines: tuple[str, ...] = ()

    # Classification markers (slug substrings). keep has a generic default; sweep is per-project.
    keep_markers: tuple[str, ...] = DEFAULT_KEEP_MARKERS
    sweep_markers: tuple[str, ...] = ()

    # Regenerable-intermediate detection: slug endswith one of these suffixes, OR contains a
    # heavy marker, OR exceeds the size threshold.
    heavy_view_suffixes: tuple[str, ...] = ()
    heavy_slug_markers: tuple[str, ...] = DEFAULT_HEAVY_SLUG_MARKERS
    heavy_threshold_bytes: int = DEFAULT_HEAVY_THRESHOLD_BYTES

    # Campaign folding: trailing view-suffixes stripped from a slug to recover its campaign.
    view_suffixes: tuple[str, ...] = ()

    # Run skeleton: subdirectories created under a freshly registered run dir.
    run_subdirs: tuple[str, ...] = ()

    # Archive: top-level paths excluded from the cold tarball (e.g. redundant source copies).
    archive_excludes: tuple[str, ...] = ()

    # Optional project hook to synthesize a manifest from a legacy run dir lacking run.json.
    # Signature: (run_dir, *, read) -> RunManifest | None, where `read` runs a remote command.
    legacy_manifest_reader: Callable[..., Any] | None = field(default=None)


# A minimal generic policy (no pipelines/markers/skeleton): enough for register/list/show on a
# project that hasn't defined its taxonomy yet. Consumers should supply a real one.
GENERIC_POLICY = RunPolicy()
