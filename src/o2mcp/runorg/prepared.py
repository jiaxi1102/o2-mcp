"""Prepared run identities used to seal immutable execution plans safely.

An execution plan includes its run ID and canonical run root in its digest.  The
run therefore has to be allocated *before* an adapter can construct the final
plan.  :class:`PreparedRunIdentity` is the narrow hand-off between allocation
and plan sealing: it binds the registry identity, dataset scope, and run root,
but intentionally contains no scientific commands.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass

from o2mcp.runorg.plan_components import CanonicalPaths, DatasetIdentity, _validate_identifier
from o2mcp.runorg.plan_stages import StageSpec
from o2mcp.runorg.plans import ExecutionPlan
from o2mcp.runorg.runs import _RUN_ID_RE, campaign_of


@dataclass(frozen=True)
class PreparedRunIdentity:
    """A registered, immutable identity that is ready for plan construction.

    ``dataset_ids`` are deliberately only stable identifiers.  The pipeline
    adapter supplies the richer :class:`DatasetIdentity` records, including
    reviewed manifest and storage-binding hashes, when it seals the plan.
    """

    project: str
    campaign: str
    pipeline: str
    run_id: str
    run_root: str
    created_utc: str
    dataset_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject identities that could point at a different registry record."""

        for field_name in ("project", "campaign", "pipeline"):
            _validate_identifier(getattr(self, field_name), field_name)
        if not _RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError("prepared run_id must match RUN_<UTCtimestamp>Z_<slug>")
        if campaign_of(self.run_id) != self.campaign:
            raise ValueError("prepared campaign must match the campaign encoded in run_id")
        if not self.run_root.startswith("/") or posixpath.normpath(self.run_root) != self.run_root:
            raise ValueError("prepared run_root must be a normalized absolute POSIX path")
        if posixpath.basename(self.run_root) != self.run_id:
            raise ValueError("prepared run_root basename must equal run_id")
        if posixpath.basename(posixpath.dirname(self.run_root)) != self.campaign:
            raise ValueError("prepared run_root parent must equal campaign")
        if not self.created_utc or "\n" in self.created_utc or "\r" in self.created_utc:
            raise ValueError("prepared created_utc must be a non-empty single-line value")
        if not isinstance(self.dataset_ids, tuple) or not self.dataset_ids:
            raise ValueError("prepared dataset_ids must be a non-empty immutable tuple")
        for dataset_id in self.dataset_ids:
            _validate_identifier(dataset_id, "dataset_ids[]")
        if len(set(self.dataset_ids)) != len(self.dataset_ids):
            raise ValueError("prepared dataset_ids cannot contain duplicates")
        object.__setattr__(self, "dataset_ids", tuple(sorted(self.dataset_ids)))

    def seal_plan(
        self,
        *,
        source_commit: str,
        source_bundle_sha256: str,
        datasets: tuple[DatasetIdentity, ...],
        paths: CanonicalPaths,
        stages: tuple[StageSpec, ...],
    ) -> ExecutionPlan:
        """Build an :class:`ExecutionPlan` that exactly matches this allocation.

        The explicit equality checks make it impossible for an adapter to prepare
        one run, then accidentally hash commands against another run root or a
        broadened dataset scope.
        """

        if paths.run_root != self.run_root:
            raise ValueError("execution plan paths.run_root does not match the prepared run")
        planned_dataset_ids = tuple(sorted(dataset.dataset_id for dataset in datasets))
        if planned_dataset_ids != self.dataset_ids:
            raise ValueError("execution plan datasets do not match the prepared dataset scope")
        return ExecutionPlan(
            project=self.project,
            campaign=self.campaign,
            pipeline=self.pipeline,
            run_id=self.run_id,
            source_commit=source_commit,
            source_bundle_sha256=source_bundle_sha256,
            datasets=datasets,
            paths=paths,
            stages=stages,
        )


__all__ = ["PreparedRunIdentity"]
