"""Executor for workspace disk hygiene over an :class:`O2Connection`.

:class:`O2Workspace` gathers the on-disk facts (depth-1 ``du`` of the home and
scratch tiers), classifies each entry via the pure rules in
:mod:`o2mcp.workspace`, and reports per-tier usage + misplacement flags
(``o2_disk_report``). ``gc`` turns the PRUNE_* findings into a fail-closed,
detached prune script (``o2_workspace_gc``), with a ``dry_run`` that returns the
script without executing. ``place`` resolves canonical output paths.

Python 3.9, no third-party deps.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any

from o2mcp.connection import O2Connection
from o2mcp.workspace import (
    PRUNE_REDUNDANT,
    PRUNE_REGENERABLE,
    WorkspaceLayout,
    classify_entry,
    plan_prune,
    summarize_report,
)


@dataclass
class GcPlan:
    """Outcome of a workspace gc request (dry-run or launched)."""

    dry_run: bool
    script: str
    pruned_paths: list[str] = field(default_factory=list)
    submitted: bool = False
    message: str = ""


class O2Workspace:
    """Disk-hygiene operations over an established O2 connection."""

    def __init__(self, connection: O2Connection, layout: WorkspaceLayout | None = None) -> None:
        self.conn = connection
        self.layout = layout or WorkspaceLayout.from_config(connection.config)

    def place(self, kind: str, project: str | None = None) -> str:
        """Resolve the canonical absolute path for an artifact kind (+ project)."""
        return self.layout.place(kind, project)

    # -- gather + report -------------------------------------------------------
    def _gather(self, roots: list[str]) -> list[tuple[str, int]]:
        """Byte-accurate depth-1 sizes of each root's children (files + dotdirs)."""
        globs = " ".join(f"{shlex.quote(r)}/* {shlex.quote(r)}/.[!.]*" for r in roots)
        res = self.conn.run(f"du -sb {globs} 2>/dev/null", timeout=300.0)
        out: list[tuple[str, int]] = []
        for line in res.stdout.splitlines():
            if "\t" not in line:
                continue
            size_s, path = line.split("\t", 1)
            try:
                out.append((path.strip(), int(size_s)))
            except ValueError:
                continue
        return out

    def disk_report(self, roots: list[str] | None = None) -> dict[str, Any]:
        """Per-tier usage + classified misplacement/prune flags for home + scratch."""
        roots = roots or [self.layout.home_root, self.layout.scratch_root]
        classified = [
            {**classify_entry(path, size_bytes=size), "size_bytes": size} for path, size in self._gather(roots)
        ]
        return {"roots": roots, "entries": classified, **summarize_report(classified)}

    # -- gc --------------------------------------------------------------------
    def gc(self, *, dry_run: bool = True, roots: list[str] | None = None) -> GcPlan:
        """Prune only the regenerable + redundant findings (detached, fail-closed).

        ``dry_run`` (default) returns the generated script without running it. When
        executed, the script is launched detached on O2 (logging to
        ``~/o2_workspace_gc.log``) so a VPN blip cannot interrupt it.
        """
        report = self.disk_report(roots)
        prunable = [e for e in report["entries"] if e["disposition"] in (PRUNE_REGENERABLE, PRUNE_REDUNDANT)]
        script = plan_prune(prunable, self.layout)
        pruned_paths = [e["path"] for e in prunable]
        if dry_run or not prunable:
            msg = "dry_run: script not executed" if dry_run else "nothing to prune"
            return GcPlan(dry_run=True, script=script, pruned_paths=pruned_paths, submitted=False, message=msg)
        remote = "$HOME/o2_workspace_gc.sh"
        launch = (
            f"cat > {remote} && chmod +x {remote} && "
            f"(nohup bash {remote} > $HOME/o2_workspace_gc.log 2>&1 </dev/null &) && echo LAUNCHED"
        )
        res = self.conn.run(launch, timeout=60.0, input_text=script)
        return GcPlan(
            dry_run=False,
            script=script,
            pruned_paths=pruned_paths,
            submitted=res.ok and "LAUNCHED" in res.stdout,
            message=res.stdout.strip() or res.stderr.strip(),
        )
