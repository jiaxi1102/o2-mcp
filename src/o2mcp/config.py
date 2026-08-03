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


def default_policy_file() -> Path:
    """Return the single workstation-wide O2 policy-state path.

    ``O2_POLICY_FILE`` supports isolated deployments and offline tests without
    reviving the former project-local lock model. Every cooperating MCP process
    must resolve this setting to the same file.
    """

    env = os.environ.get("O2_POLICY_FILE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".agent_locks" / "O2_POLICY.json"


def _default_ssh_config_file() -> Path:
    """Return the user SSH config that defines the governed O2 aliases."""

    configured = os.environ.get("O2_SSH_CONFIG_FILE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".ssh" / "config"


@dataclass
class O2Config:
    """Connection settings for HMS O2.

    Attributes:
        host_alias: SSH alias for login/compute commands (the ControlMaster host).
        transfer_alias: SSH alias for bulk rsync transfers (the O2 transfer node).
        connect_timeout: SSH ``ConnectTimeout`` in seconds.
        policy_file: Workstation-wide JSON state controlling disabled,
            reuse-only, and one-shot login authorization behavior. There is no
            environment variable that bypasses a valid disabled policy.
        default_user: Username for ``squeue -u`` etc.; ``None`` resolves to ``$USER`` remotely.
        default_log_dir: Remote directory pattern where Slurm logs land.
        ssh_config_file: User SSH config containing the O2 alias definitions.
            The connection layer reads and flattens this file without executing
            ``Match exec`` predicates before asking OpenSSH to expand a socket.
    """

    host_alias: str = field(default_factory=lambda: os.environ.get("O2_SSH_HOST_ALIAS", "o2"))
    transfer_alias: str = field(default_factory=lambda: os.environ.get("O2_SSH_TRANSFER_ALIAS", "o2-transfer"))
    connect_timeout: int = field(default_factory=lambda: int(os.environ.get("O2_SSH_CONNECT_TIMEOUT_SECONDS", "20")))
    policy_file: Path = field(default_factory=default_policy_file)
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

    # VPN egress guard. These settings were appended so they did not shift the
    # positional order of the original public fields (the dataclass is not
    # kw_only because this core stays Python 3.9-compatible). New fields continue
    # to be appended below for the same compatibility reason.
    # Refuse to open a NEW O2 login unless the route to O2 egresses via a VPN tunnel interface
    # (prefix below): O2 autopushes Duo to any non-HMS source IP, so a login leaving via a
    # physical interface (en0) instead of the HMS VPN triggers a phone prompt.
    # Off-VPN access is never enabled durably by configuration; it must be scoped
    # into the one-shot login grant issued after explicit user approval.
    vpn_iface_prefix: str = field(default_factory=lambda: os.environ.get("O2_VPN_IFACE_PREFIX", "utun"))
    ssh_config_file: Path = field(default_factory=_default_ssh_config_file)

    def __post_init__(self) -> None:
        """Normalize and validate process-independent local authority paths.

        The policy file coordinates independently launched MCP processes. A
        relative path would resolve against each process's current directory
        and silently create multiple workstation safety states, so both
        environment-derived and explicitly constructed configurations must use
        one absolute path.
        """

        self.policy_file = Path(self.policy_file).expanduser()
        if not self.policy_file.is_absolute():
            raise ValueError(
                "O2 policy path must be absolute; set O2_POLICY_FILE to one " "workstation-wide absolute path"
            )
        self.ssh_config_file = Path(self.ssh_config_file).expanduser()

    def base_ssh_opts(self) -> list[str]:
        """Return baseline SSH options shared by login and reuse operations.

        These options make subprocesses non-interactive, but they intentionally do
        not disable public-key authentication: :meth:`O2Connection.start_master`
        needs public-key authentication for the one explicitly authorized login.
        Ordinary commands and transfers must use :meth:`reuse_only_ssh_opts`
        instead so a missing ControlMaster cannot fall back to a fresh connection.
        """
        return [
            "-o",
            "BatchMode=yes",
            "-o",
            "RequestTTY=no",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
        ]

    def reuse_only_ssh_opts(self) -> list[str]:
        """Return SSH options that can reuse a master but cannot authenticate anew.

        OpenSSH multiplex clients normally fall back to a standalone connection
        when ``ControlPath`` is missing or stops listening. On HMS O2 that fallback
        is unsafe because even a key-based connection can trigger an automatic Duo
        request. These command-line options leave multiplexing enabled as a client
        (OpenSSH documents ``ControlMaster=no`` clients as able to reuse an existing
        configured socket) while disabling every authentication method that could
        establish a new session. ProxyJump and ProxyCommand are also disabled: an
        SSH proxy is a separate subprocess whose authentication settings would not
        inherit the outer client's fail-closed restrictions. LocalCommand and
        KnownHostsCommand hooks are disabled for the same reason: neither may be
        allowed to spawn an authentication-capable helper from a reuse-only call.

        The options are passed on the command line so they take precedence over a
        permissive user SSH config. If the master disappears between the explicit
        socket check and command execution, SSH therefore fails with authentication
        disabled rather than contacting Duo.
        """
        return [
            *self.base_ssh_opts(),
            "-o",
            "ControlMaster=no",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ProxyCommand=none",
            "-o",
            "ProxyJump=none",
            "-o",
            "PermitLocalCommand=no",
            "-o",
            "KnownHostsCommand=none",
            "-o",
            "PreferredAuthentications=none",
            "-o",
            "PubkeyAuthentication=no",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
            "-o",
            "GSSAPIAuthentication=no",
            "-o",
            "HostbasedAuthentication=no",
            "-o",
            "NumberOfPasswordPrompts=0",
        ]
