"""Configuration for the O2 cluster connection.

Defaults mirror the existing shell tooling (``scripts/o2_ssh_master.sh`` and the
``Host o2`` SSH config block) so the MCP server is a drop-in, safer replacement
for the ad-hoc ssh/rsync commands. Everything is overridable via environment
variables (the same names the shell scripts already use) or explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_lock_file() -> Path:
    """The O2 safety lock path (``.agent_locks/O2_DISABLED`` under the repo)."""
    env = os.environ.get("O2_SSH_LOCK_FILE")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / ".agent_locks" / "O2_DISABLED"


@dataclass
class O2Config:
    """Connection settings for HMS O2.

    Attributes:
        host_alias: SSH alias for login/compute commands (the ControlMaster host).
        transfer_alias: SSH alias for bulk rsync transfers (the O2 transfer node).
        connect_timeout: SSH ``ConnectTimeout`` in seconds.
        lock_file: If this path exists, every O2 operation refuses to run (the
            project's hard stop against repeated Duo/MFA prompts).
        ignore_lock: Mirror of ``O2_IGNORE_LOCAL_LOCK=1`` to bypass the lock.
        default_user: Username for ``squeue -u`` etc.; ``None`` resolves to ``$USER`` remotely.
        default_log_dir: Remote directory pattern where Slurm logs land.
    """

    host_alias: str = field(default_factory=lambda: os.environ.get("O2_SSH_HOST_ALIAS", "o2"))
    transfer_alias: str = field(default_factory=lambda: os.environ.get("O2_SSH_TRANSFER_ALIAS", "o2-transfer"))
    connect_timeout: int = field(default_factory=lambda: int(os.environ.get("O2_SSH_CONNECT_TIMEOUT_SECONDS", "20")))
    lock_file: Path = field(default_factory=_default_lock_file)
    ignore_lock: bool = field(default_factory=lambda: os.environ.get("O2_IGNORE_LOCAL_LOCK", "0") == "1")
    default_user: str | None = field(default_factory=lambda: os.environ.get("O2_USER") or None)
    default_log_dir: str = field(default_factory=lambda: os.environ.get("O2_LOG_DIR", "~/logs/o2"))

    # Run-organization storage tiers (consumed by the run-organization layer in the project using this library).
    # active runs live on (purgeable) scratch; promoted keepers move to backed-up group;
    # archived runs become cold tarballs on standby. The registry MUST live on a durable
    # tier (group) so it survives a scratch purge.
    scratch_runs_root: str = field(
        default_factory=lambda: os.environ.get("O2_SCRATCH_RUNS_ROOT", "/n/scratch/users/j/jiz947/runs")
    )
    group_runs_root: str = field(
        default_factory=lambda: os.environ.get("O2_GROUP_RUNS_ROOT", "/n/groups/tabin/jzhao/runs")
    )
    standby_archive_root: str = field(
        default_factory=lambda: os.environ.get(
            "O2_STANDBY_ARCHIVE_ROOT", "/n/standby/hms/genetics/tabin/compute/jzhao/runs_archive"
        )
    )
    registry_path: str = field(
        default_factory=lambda: os.environ.get("O2_RUN_REGISTRY", "/n/groups/tabin/jzhao/runs/registry.jsonl")
    )

    # Workspace layout tiers (see o2mcp.workspace and docs/WORKSPACE_LAYOUT.md).
    # home = code+config only; group = durable data/results; scratch = ephemeral work;
    # standby = cold archive. Per-project outputs resolve under these roots.
    home_root: str = field(default_factory=lambda: os.environ.get("O2_HOME_ROOT", "/home/jiz947"))
    group_root: str = field(default_factory=lambda: os.environ.get("O2_GROUP_ROOT", "/n/groups/tabin/jzhao"))
    scratch_root: str = field(default_factory=lambda: os.environ.get("O2_SCRATCH_ROOT", "/n/scratch/users/j/jiz947"))
    standby_root: str = field(
        default_factory=lambda: os.environ.get("O2_STANDBY_ROOT", "/n/standby/hms/genetics/tabin/compute/jzhao")
    )
    # How many timestamped DB/registry snapshots to retain when pruning snapshot history.
    snapshot_keep: int = field(default_factory=lambda: int(os.environ.get("O2_SNAPSHOT_KEEP", "2")))

    def base_ssh_opts(self) -> list[str]:
        """SSH options enforcing public-key/batch mode (never password or Duo).

        These match the project's SSH-config contract: batch mode, no TTY, no
        keyboard-interactive fallback, so a missing key or dead master fails fast
        instead of triggering an interactive MFA prompt.
        """
        return [
            "-o",
            "BatchMode=yes",
            "-o",
            "RequestTTY=no",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
        ]
