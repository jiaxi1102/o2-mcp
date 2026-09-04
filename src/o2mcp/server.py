#!/usr/bin/env python3
"""MCP server for HMS O2 cluster access (``o2_mcp``).

Exposes the safe, tested ``o2mcp`` primitives as MCP tools so an agent can
directly submit Slurm work, debug with arbitrary remote commands, monitor jobs,
tail logs, and move files.

DUO MODEL — read this before using the tools. HMS O2 can challenge both a new
SSH transport and a new session channel inside a ControlMaster. The persistent
global policy therefore defaults fail-closed. Each configured host role uses a
workstation-wide broker that keeps one SSH session channel open; starting either
broker requires a short-lived, client-bound, one-attempt role-matched grant.

NEVER open the master in a loop, and never run these tools on a short timer — a
periodic poller that reconnects each cycle is what causes a "Duo call every
minute". To completely avoid Duo, don't poll O2 from here at all (have O2 push
results out via Globus/OnDemand) or ask HMS RC for SSH-certificate access; see
``docs/O2_MCP.md``.

``~/.agent_locks/O2_POLICY.json`` is the sole authoritative policy state.

Run as a local stdio server:

    python -m o2mcp.server        # or the `o2-mcp` console script

Requires the optional ``mcp`` dependency (``pip install -e ".[o2]"``) and Python
>= 3.10. The underlying primitives in ``o2mcp`` work on Python 3.9 and are
unit-tested without it.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import shlex
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal

import anyio
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from o2mcp import (
    CommandResult,
    O2AsyncTransfer,
    O2BrokerBusyError,
    O2BrokerCommandOutcomeUnknownError,
    O2BrokerError,
    O2BrokerStartupError,
    O2BrokerUnavailableError,
    O2Connection,
    O2LoginGrantError,
    O2MasterUnavailableError,
    O2OffVpnError,
    O2PolicyConflictError,
    O2PolicyDeniedError,
    O2PolicyInvalidError,
    O2Slurm,
    O2UnsafeTransportError,
    O2Workspace,
    billing,
    transfer_tools,
)
from o2mcp.broker_protocol import MAX_COMMAND_BYTES
from o2mcp.connection import BROKER_TRUNCATION_NOTE
from o2mcp.launch_evidence import (
    LaunchEvidenceError,
    build_launch_evidence,
    claimed_job_id,
    evidence_content_digest,
    launch_evidence_digest,
    parse_encoded_checksum_manifest,
    parse_encoded_json_artifact,
    parse_json_artifact,
    parse_scheduler_record,
    required_package_files,
)
from o2mcp.policy import LoginTarget
from o2mcp.slurm import _quote_remote_path

mcp = FastMCP("o2_mcp")


# --- shared helpers ----------------------------------------------------------
def _connection() -> O2Connection:
    """Build a client whose broker daemons and transfer master persist out of process."""
    return O2Connection()


def _command_payload(result: CommandResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


async def _run_tool(fn: Callable[[], dict[str, Any]]) -> str:
    """Run a blocking O2 operation off the event loop and JSON-encode the result.

    Policy and master-availability errors are turned into actionable, non-fatal
    payloads instead of crashing the tool call.
    """
    try:
        payload = await anyio.to_thread.run_sync(fn)
    except O2PolicyDeniedError as exc:
        payload = {"ok": False, "error": "policy_disabled", "message": str(exc)}
    except O2PolicyInvalidError as exc:
        payload = {"ok": False, "error": "policy_invalid", "message": str(exc)}
    except O2PolicyConflictError as exc:
        payload = {"ok": False, "error": "policy_conflict", "message": str(exc)}
    except O2LoginGrantError as exc:
        payload = {"ok": False, "error": "login_grant_invalid", "message": str(exc)}
    except O2MasterUnavailableError as exc:
        payload = {"ok": False, "error": "no_master", "message": str(exc)}
    except O2OffVpnError as exc:
        payload = {"ok": False, "error": "off_vpn", "message": str(exc)}
    except O2UnsafeTransportError as exc:
        payload = {"ok": False, "error": "unsafe_transport", "message": str(exc)}
    except O2BrokerBusyError as exc:
        # A subclass of the outcome-unknown error and must be caught before it.
        # The broker being merely occupied is worth reporting separately, but a
        # budget that expires as the daemon acknowledges leaves a command that
        # may already be running, so the retry instruction stays conservative.
        payload = {"ok": False, "error": "broker_busy", "message": str(exc), "retry_safe": False}
    except O2BrokerCommandOutcomeUnknownError as exc:
        payload = {"ok": False, "error": "broker_outcome_unknown", "message": str(exc), "retry_safe": False}
    except O2BrokerUnavailableError as exc:
        payload = {"ok": False, "error": "no_broker", "message": str(exc)}
    except O2BrokerStartupError as exc:
        payload = {"ok": False, "error": "broker_start_failed", "message": str(exc)}
    except O2BrokerError as exc:
        payload = {"ok": False, "error": "broker_error", "message": str(exc)}
    except LaunchEvidenceError as exc:
        # Refusing to mint is this tool working as intended, so it gets a named
        # code rather than falling into the defensive catch-all below.
        payload = {"ok": False, "error": "launch_evidence_refused", "message": str(exc)}
    except subprocess.TimeoutExpired as exc:
        payload = {"ok": False, "error": "operation_timeout", "message": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive
        payload = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
    return json.dumps(payload, indent=2)


# --- input models ------------------------------------------------------------
class StartMasterInput(BaseModel):
    """Configure one transfer-master start or local reuse check."""

    model_config = ConfigDict(extra="forbid")
    grant_id: str | None = Field(
        default=None,
        description=(
            "One-shot grant returned by o2_authorize_login. It is unnecessary when "
            "the exact requested ControlMaster is already running. When omitted, "
            "auto_authorize_on_vpn controls the standing on-VPN authorization path."
        ),
    )
    transfer: bool = Field(
        default=False,
        description=(
            "Open the dedicated transfer-node master (o2-transfer). Login-node "
            "commands now use o2_start_broker instead of a ControlMaster."
        ),
    )
    auto_authorize_on_vpn: bool = Field(
        default=True,
        description=(
            "When grant_id is omitted, automatically authorize exactly one start only if the local route to the "
            "transfer host is proven to use the HMS VPN. Off-VPN or indeterminate routing fails before SSH and "
            "requires explicit user approval through o2_authorize_login with allow_offvpn=true."
        ),
    )


class StopMasterInput(BaseModel):
    """Select the sole retained ControlMaster role for local shutdown."""

    model_config = ConfigDict(extra="forbid")
    transfer: bool = Field(
        default=True,
        description="Must remain true; login-role masters are retired and command brokers have their own stop tool.",
    )


class StartBrokerInput(BaseModel):
    """Configure one broker start or local reuse check for an O2 host role."""

    model_config = ConfigDict(extra="forbid")
    grant_id: str | None = Field(
        default=None,
        description=(
            "One-shot login grant returned by o2_authorize_login. It is unnecessary "
            "when the workstation broker is already locally responsive. When omitted, "
            "auto_authorize_on_vpn controls the standing on-VPN authorization path."
        ),
    )
    transfer: bool = Field(
        default=False,
        description="Start the separately granted transfer-host broker instead of the login-host broker.",
    )
    auto_authorize_on_vpn: bool = Field(
        default=True,
        description=(
            "When grant_id is omitted, automatically authorize exactly one start only if the selected O2 host "
            "is proven to route through the HMS VPN. Off-VPN or indeterminate routing fails before SSH and "
            "requires explicit user approval through o2_authorize_login with allow_offvpn=true."
        ),
    )


class StopBrokerInput(BaseModel):
    """Local-only reason for closing the persistent broker process."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    reason: str = Field(..., min_length=1, max_length=240)
    transfer: bool = Field(default=False, description="Stop the transfer-host broker instead of the login broker.")
    force: bool = Field(
        default=False,
        description=(
            "Have the daemon honour this stop even after the request times out locally. Needed when the "
            "broker is busy: an ordinary queued stop is cancelled once its caller gives up, so it cannot "
            "retire a broker that is serving a command. A forced stop is graceful and queued, not "
            "prioritized: it takes effect after the in-flight command and any requests already waiting "
            "ahead of it, and abandons none of them."
        ),
    )


class DisablePolicyInput(BaseModel):
    """Local-only reason for engaging the global O2 safety stop."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    reason: str = Field(..., min_length=1, max_length=240)


class EnableReuseInput(BaseModel):
    """Explicit global transition from disabled to reuse-only operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    expected_revision: int = Field(..., ge=0)
    expected_generation: str = Field(
        ...,
        min_length=1,
        description="Generation UUID from the same o2_local_status snapshot as expected_revision.",
    )
    approval_reference: str = Field(..., min_length=1, max_length=240)
    acknowledge_global: bool = Field(
        ...,
        description=(
            "Must be true to confirm the user approved a workstation-global transition, "
            "not merely continuation of one project task."
        ),
    )


class AuthorizeLoginInput(BaseModel):
    """Scope one explicit user authorization into a one-attempt grant."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    expected_revision: int = Field(..., ge=0)
    expected_generation: str = Field(
        ...,
        min_length=1,
        description="Generation UUID from the same o2_local_status snapshot as expected_revision.",
    )
    target: Literal["login", "transfer"]
    allow_offvpn: bool = Field(
        default=False,
        description="Whether this exact one-shot login may proceed without a proven HMS VPN route.",
    )
    approval_reference: str = Field(..., min_length=1, max_length=240)


class MintLaunchEvidenceInput(BaseModel):
    """Bind one finished governed stage into an operator-approved record."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    diagnostic_path: str = Field(
        ...,
        min_length=1,
        description="Absolute O2 path to the stage's run diagnostic JSON (the job's stdout).",
    )
    plan_path: str = Field(
        ...,
        min_length=1,
        description="Absolute O2 path to the frozen execution plan JSON that was approved.",
    )
    package_path: str = Field(
        ...,
        min_length=1,
        description="Absolute O2 path to the published package directory the stage verified.",
    )
    stage: str = Field(default="platform-canary", min_length=1, max_length=120)
    expected_revision: int = Field(..., ge=0)
    expected_generation: str = Field(
        ...,
        min_length=1,
        description="Generation UUID from the same o2_local_status snapshot as expected_revision.",
    )
    approval_reference: str = Field(
        ...,
        min_length=1,
        max_length=240,
        description="The operator's explicit approval of this exact record; it is what authenticates it.",
    )
    timeout_seconds: float = Field(default=60.0, gt=0, le=60.0)


class ProbeInput(BaseModel):
    """Select one existing role-specific broker for an explicit remote probe."""

    model_config = ConfigDict(extra="forbid")
    transfer: bool = Field(
        default=False,
        description="Use the existing transfer-host broker instead of the login-host broker.",
    )


# One command's timeout is one command's hold on a channel shared by every MCP
# process on the workstation -- roughly twenty of them here -- so this number is
# not a per-caller convenience. It is how long one caller may stop all the
# others from doing anything at all.
#
# A minute is generous for what this tool is for: inspecting state. `squeue`,
# `sacct`, `cat`, staging a script all return in well under it. What a minute
# will not accommodate is *waiting* -- a remote `sleep`, a `while ... squeue`
# poll, an `srun` held in the foreground. That is deliberate and is the point of
# the number. Detecting a wait by inspecting the command string cannot work:
# `sleep 280`, `python -c 'time.sleep(280)'` and `while ! test -f done; do :;
# done` are the same intent in three shapes, and any pattern broad enough to
# catch them also catches a legitimate `sleep 5` between two operations.
# Bounding occupancy needs no such judgement, and it treats a slow command and a
# deliberate wait alike -- which is correct, because the callers being starved
# cannot tell them apart either.
#
# Work that genuinely takes longer belongs in a submitted job: o2_submit_job,
# then o2_job_status or o2_squeue polled from here, each of which holds the
# channel for well under a second per check.
MAX_EXEC_TIMEOUT_SECONDS = 60.0


class RunInput(BaseModel):
    # ``validate_default`` because a constraint that skips the default is not a
    # constraint on the common path: most callers omit the timeout, so a default
    # above the ceiling silently grants every one of them more than the ceiling
    # allows. Validating it turns that into an import-time failure instead.
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", validate_default=True)
    command: str = Field(..., description="Remote shell command to run on an O2 login node.", min_length=1)
    timeout_seconds: float = Field(
        default=30.0,
        description=(
            "Command timeout in seconds, default 30, capped at 60. The channel is shared and serialized by every MCP "
            "process on this workstation, so this is how long one command may block every other caller "
            "from doing anything at all. A minute is ample for inspecting state, and deliberately will "
            "not accommodate waiting: do not sleep, poll, or hold a foreground srun inside a command. "
            "Work that takes longer belongs in a submitted job -- o2_submit_job, then o2_job_status or "
            "o2_squeue polled from here, each holding the channel for under a second per check."
        ),
        gt=0,
        le=MAX_EXEC_TIMEOUT_SECONDS,
    )


class SubmitInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    remote_script_path: str | None = Field(
        default=None, description="Path to an sbatch script that already exists on O2."
    )
    script_text: str | None = Field(
        default=None, description="sbatch script contents to stage to O2 before submitting (use with remote_path)."
    )
    remote_path: str | None = Field(
        default=None, description="Where to stage script_text on O2 (required when script_text is given)."
    )
    priced: str | None = Field(
        default=None,
        description=(
            "The `receipt` string from an o2_price_job call for this "
            "submission's resource shape. Optional and never blocking, but "
            "passing it records WHAT was priced alongside the job id, so a "
            "submission that skipped pricing is visible in its own receipt "
            "rather than indistinguishable from one that did not."
        ),
    )
    sbatch_args: list[str] = Field(default_factory=list, description="Extra sbatch flags, e.g. ['--time=02:00:00'].")


class QueueInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    user: str | None = Field(default=None, description="Username for squeue -u (defaults to remote $USER).")


class PriceJobInput(BaseModel):
    # allow_inf_nan=False: a JSON number that overflows to inf satisfies gt/ge
    # and then reaches int(), which raises OverflowError outside the
    # BillingError handler -- the tool call crashes rather than answering.
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", allow_inf_nan=False)
    partition: str = Field(..., description="Partition the job would run on.", min_length=1)
    cpus: float = Field(
        ...,
        description=(
            "Total CPUs the submission REQUESTS, across all tasks: --ntasks "
            "multiplied by --cpus-per-task. Give what the directives ask for, "
            "not what you expect Slurm to allocate -- where mem_per_cpu_gb "
            "exceeds a partition's MaxMemPerCPU this raises the count itself, "
            "and passing an already-raised number would raise it again. The "
            "allocated count comes back in the response."
        ),
        gt=0,
    )
    mem_gb: float | None = Field(
        default=None,
        description=(
            "TOTAL memory in GB across the whole allocation. sbatch's --mem is "
            "per NODE, so a two-node job written --mem=32G holds 64 here; "
            "--mem-per-cpu multiplies out the same way. Passing the directive "
            "unchanged halves the charge and moves every boundary. Omit only to "
            "price the partition's configured default; omitting it does not "
            "mean a request for no memory, and 0 is not accepted because sbatch "
            "reads --mem=0 as all memory on every node."
        ),
        ge=0,
    )
    mem_per_cpu_gb: float | None = Field(
        default=None,
        description=(
            "The --mem-per-cpu value in GB, when the submission used one. Give "
            "it IN ADDITION to mem_gb: MaxMemPerCPU acts on the per-CPU figure "
            "and not on an absolute --mem, so without it a partition that caps "
            "per-CPU memory cannot be priced correctly -- Slurm lowers the "
            "per-CPU value and adds CPUs to keep the total, which raises the "
            "bill. Omit it for an absolute --mem."
        ),
        gt=0,
    )
    ntasks: float | None = Field(
        default=None,
        description=(
            "Tasks the allocation runs (--ntasks), when the submission states "
            "one. Slurm raises --cpus-per-task, so a MaxMemPerCPU adjustment "
            "is rounded up for EACH task and the grouping changes the total: "
            "two tasks of one CPU is not the same allocation as one task of "
            "two. Defaults to Slurm's own default of one, for which the "
            "arithmetic is the same either way."
        ),
        gt=0,
    )
    gpus: float | None = Field(default=None, description="GPUs the allocation will hold.", ge=0)
    gpu_model: str | None = Field(
        default=None,
        description=(
            "GPU model, when the partition prices models separately " "(TRESBillingWeights GRES/gpu:<model>)."
        ),
    )
    nodes: float | None = Field(
        default=None,
        description=(
            "Nodes the allocation will hold. Give it whenever the submission "
            "pins a node count: it is what lets MinNodes, MaxNodes and the "
            "per-node CPU and memory caps be checked, and what bounds the "
            "headroom this reports. Required when the partition defaults "
            "memory per node and mem_gb is omitted, since that default is per "
            "node and cannot be multiplied out without it."
        ),
        gt=0,
    )
    # No refresh flag: refreshing WRITES the weight cache, and a tool that can
    # write must not advertise readOnlyHint. o2_refresh_billing_weights owns
    # that operation so this one's hint stays true.


class JobIdInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    job_id: str = Field(..., description="Slurm job id.", min_length=1)


class TailLogInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    remote_path: str = Field(..., description="Path to the remote log file on O2.", min_length=1)
    lines: int = Field(default=100, description="Number of trailing lines to show.", ge=1, le=10000)


# --- workspace-layout inputs (see o2mcp.workspace / docs/WORKSPACE_LAYOUT.md) ---
class WorkspaceReportInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    roots: list[str] | None = Field(
        default=None, description="Roots to scan (default: the home and scratch tier roots)."
    )


class WorkspaceGcInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    dry_run: bool = Field(default=True, description="When true, return the prune script without executing it.")
    roots: list[str] | None = Field(default=None, description="Roots to scan (default: home and scratch tier roots).")


class PlaceInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    kind: str = Field(
        ...,
        description="Artifact kind: results|data|runs_active|runs_kept|registry|work|staging|logs|archive.",
        min_length=1,
    )
    project: str | None = Field(default=None, description="Optional project name to nest under the kind's root.")


# --- tools -------------------------------------------------------------------
def _local_socket_inventory() -> list[dict[str, Any]]:
    """Inspect conventional ControlMaster sockets without invoking SSH.

    ``o2_local_status`` must remain visibly different from a remote probe.  A
    filesystem inventory can report socket presence but intentionally does not
    claim the socket is remotely responsive; only ``o2_probe`` establishes that.
    """

    sockets: list[dict[str, Any]] = []
    roots = [Path.home() / ".ssh" / "controlmasters"]
    for root in roots:
        if not root.is_dir():
            continue
        try:
            candidates = sorted(root.iterdir())
        except OSError:
            # Socket inventory is only one local diagnostic facet. A directory
            # permission race or removal must not hide the independently useful
            # policy generation/revision, process list, and transfer receipts.
            continue
        for candidate in candidates:
            try:
                metadata = candidate.lstat()
            except OSError:
                continue
            if stat.S_ISSOCK(metadata.st_mode):
                sockets.append(
                    {
                        "path": str(candidate),
                        "owner_uid": metadata.st_uid,
                        "modified_at": metadata.st_mtime,
                    }
                )
    return sockets


def _local_o2_processes() -> list[dict[str, Any]]:
    """Return O2-related local process rows using ``ps`` only.

    This is diagnostic text matching, not an authorization decision.  The
    connection layer still proves exact socket reuse before every remote call.
    """

    try:
        proc = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid, ppid, command = fields
        lowered = command.lower()
        if not any(
            marker in lowered
            for marker in ("o2-mcp", "o2mcp.broker", "clock-runs-mcp", "o2.hms", "o2.rc.hms", "o2-transfer")
        ):
            continue
        rows.append({"pid": int(pid), "ppid": int(ppid), "command": command})
    return rows


def _local_status_payload() -> dict[str, Any]:
    """Build the shared local-only payload for canonical and compatibility tools."""

    conn = _connection()
    snapshot = conn.policy.snapshot()
    state = snapshot.state or {}
    grant = state.get("login_grant") if isinstance(state, dict) else None
    if isinstance(grant, dict):
        # The id is safe to display because consumption is also bound to the
        # issuing process's client_id.  Exposing it helps the authorized task
        # recover from an MCP response-display failure without enabling theft.
        grant = {**grant, "owned_by_this_client": grant.get("client_id") == conn.policy.client_id}
    try:
        transfers = O2AsyncTransfer(conn).status()
    except (OSError, ValueError, OverflowError, KeyError, TypeError) as exc:
        # Policy recovery must remain visible even when the transfer directory
        # itself is unreadable. Individual malformed receipts are normally
        # isolated by O2AsyncTransfer.status; this is the directory-level guard.
        transfers = {
            "ok": False,
            "error": "transfer_status_unavailable",
            "message": str(exc),
        }
    login_broker = conn.broker_local_status()
    transfer_broker = conn.broker_local_status(transfer=True)
    return {
        "ok": True,
        "local_only": True,
        "policy": {
            "path": str(snapshot.path),
            "valid": snapshot.valid,
            "effective_mode": snapshot.effective_mode,
            "generation": snapshot.generation,
            "revision": snapshot.revision,
            "error": snapshot.error,
            "login_grant": grant,
            "login_attempt": state.get("login_attempt") if isinstance(state, dict) else None,
            "recent_events": state.get("events", [])[-10:] if isinstance(state, dict) else [],
            # The durable attestation ledger. Unlike recent_events this is never
            # evicted, so a record minted long ago is still verifiable against
            # it; only the newest few are surfaced here, the rest live in the file.
            "launch_evidence_mint_count": (
                len(state.get("launch_evidence_mints", [])) if isinstance(state, dict) else 0
            ),
            "recent_launch_evidence_mints": (
                state.get("launch_evidence_mints", [])[-10:] if isinstance(state, dict) else []
            ),
        },
        "control_sockets": _local_socket_inventory(),
        # Keep the singular login-broker key for clients written against the
        # initial 0.4 preview while exposing both role-specific endpoints.
        "command_broker": login_broker,
        "command_brokers": {"login": login_broker, "transfer": transfer_broker},
        "processes": _local_o2_processes(),
        "transfers": transfers,
    }


@mcp.tool(
    name="o2_local_status",
    annotations={"title": "Local O2 policy and process status", "readOnlyHint": True, "openWorldHint": False},
)
async def o2_local_status() -> str:
    """Inspect policy, sockets, processes, receipts, and transfer logs locally.

    This tool never invokes SSH or a remote command and never retries, starts, or
    stops a ControlMaster.
    """

    return await _run_tool(_local_status_payload)


@mcp.tool(
    name="o2_status",
    annotations={"title": "Deprecated local O2 status", "readOnlyHint": True, "openWorldHint": False},
)
async def o2_status() -> str:
    """Compatibility alias for :func:`o2_local_status`; never probes remotely."""

    return await _run_tool(_local_status_payload)


@mcp.tool(
    name="o2_policy_disable",
    annotations={"title": "Disable all new O2 remote operations", "readOnlyHint": False, "openWorldHint": False},
)
async def o2_policy_disable(params: DisablePolicyInput) -> str:
    """Engage the workstation-global safety stop without terminating processes."""

    def work() -> dict[str, Any]:
        state = _connection().policy.disable(reason=params.reason)
        return {"ok": True, "policy": state}

    return await _run_tool(work)


@mcp.tool(
    name="o2_policy_enable_reuse",
    annotations={"title": "Enable reuse-only O2 access", "readOnlyHint": False, "openWorldHint": False},
)
async def o2_policy_enable_reuse(params: EnableReuseInput) -> str:
    """Enable existing-master reuse after explicit global user authorization."""

    def work() -> dict[str, Any]:
        if not params.acknowledge_global:
            return {
                "ok": False,
                "error": "global_acknowledgement_required",
                "message": "acknowledge_global must be true for a workstation-wide O2 transition.",
            }
        state = _connection().policy.enable_reuse(
            expected_revision=params.expected_revision,
            expected_generation=params.expected_generation,
            approval_reference=params.approval_reference,
        )
        return {"ok": True, "policy": state}

    return await _run_tool(work)


@mcp.tool(
    name="o2_authorize_login",
    annotations={"title": "Issue one O2 login grant", "readOnlyHint": False, "openWorldHint": False},
)
async def o2_authorize_login(params: AuthorizeLoginInput) -> str:
    """Translate explicit user approval into one short-lived host-scoped grant."""

    def work() -> dict[str, Any]:
        grant = _connection().policy.authorize_login(
            expected_revision=params.expected_revision,
            expected_generation=params.expected_generation,
            target=params.target,
            allow_offvpn=params.allow_offvpn,
            approval_reference=params.approval_reference,
        )
        return {
            "ok": True,
            "grant_id": grant.id,
            "target": grant.target,
            "allow_offvpn": grant.allow_offvpn,
            "expires_at": grant.expires_at,
            "remaining_attempts": grant.remaining_attempts,
        }

    return await _run_tool(work)


@mcp.tool(
    name="o2_probe",
    annotations={"title": "Explicit reuse-only O2 probe", "readOnlyHint": True, "openWorldHint": True},
)
async def o2_probe(params: ProbeInput) -> str:
    """Run one fixed remote probe through the existing persistent broker."""

    def work() -> dict[str, Any]:
        conn = _connection()
        alias = conn.config.transfer_alias if params.transfer else conn.config.host_alias
        # Call the connection's own probe rather than restating it here: the
        # duplicate deadline is what let the command ceiling bind one probe and
        # not the other.
        result = conn.probe(alias=alias, broker_role="transfer" if params.transfer else "login")
        return {"ok": result.ok, "alias": alias, **_command_payload(result)}

    return await _run_tool(work)


@mcp.tool(
    name="o2_start_master",
    annotations={"title": "Start O2 SSH master", "readOnlyHint": False, "openWorldHint": True},
)
async def o2_start_master(params: StartMasterInput) -> str:
    """Open only the legacy transfer ControlMaster after one matching grant.

    Login-master requests fail locally because a ControlMaster does not prevent
    per-session Duo. Transfer requests retain the prior one-shot grant behavior
    during the MVP transition.
    """

    def work() -> dict[str, Any]:
        if not params.transfer:
            return {
                "ok": False,
                "error": "login_master_retired",
                "message": (
                    "A login ControlMaster does not prevent O2 from challenging each new session channel. "
                    "Use o2_start_broker with a login-scoped grant instead."
                ),
            }
        conn = _connection()
        alias = conn.config.transfer_alias if params.transfer else None
        # Pass the requested role separately from its configured alias.  Some
        # installations deliberately use one alias/socket for both roles, so
        # comparing alias strings cannot reliably recover grant scope.
        login_target: LoginTarget = "transfer" if params.transfer else "login"
        result = conn.start_master(
            grant_id=params.grant_id,
            alias=alias,
            login_target=login_target,
            auto_authorize_on_vpn=params.auto_authorize_on_vpn,
        )
        return {"ok": result.ok, "alias": alias or conn.config.host_alias, **_command_payload(result)}

    return await _run_tool(work)


@mcp.tool(
    name="o2_stop_master",
    annotations={
        "title": "Stop O2 SSH master",
        "readOnlyHint": False,
        "destructiveHint": True,
        "openWorldHint": False,
    },
)
async def o2_stop_master(params: StopMasterInput) -> str:
    """Close a transfer or legacy login ControlMaster locally, even if disabled."""

    def work() -> dict[str, Any]:
        conn = _connection()
        alias = conn.config.transfer_alias if params.transfer else conn.config.host_alias
        result = conn.stop_master(alias=alias)
        return {"ok": result.ok, "alias": alias, **_command_payload(result)}

    return await _run_tool(work)


@mcp.tool(
    name="o2_start_broker",
    annotations={"title": "Start persistent O2 command broker", "readOnlyHint": False, "openWorldHint": True},
)
async def o2_start_broker(params: StartBrokerInput) -> str:
    """Start one dynamic command channel after consuming a role-matched grant.

    The daemon never reconnects. An already-responsive broker is a local no-op;
    a daemon that is still starting blocks another attempt.
    """

    def work() -> dict[str, Any]:
        status = _connection().start_broker(
            grant_id=params.grant_id,
            transfer=params.transfer,
            auto_authorize_on_vpn=params.auto_authorize_on_vpn,
        )
        return {
            "ok": status.get("responsive") is True,
            "target": "transfer" if params.transfer else "login",
            "broker": status,
        }

    return await _run_tool(work)


@mcp.tool(
    name="o2_stop_broker",
    annotations={
        "title": "Stop persistent O2 command broker",
        "readOnlyHint": False,
        "destructiveHint": True,
        "openWorldHint": False,
    },
)
async def o2_stop_broker(params: StopBrokerInput) -> str:
    """Close the local broker and its SSH process without changing policy.

    The default socket request retires an idle broker cleanly but cannot retire a
    busy one, because a queued stop is cancelled once its caller times out. When
    that happens the error names the command holding the channel and points at
    `force`, which the daemon honours even after the caller gives up waiting.

    A forced stop is graceful and queued rather than prioritized: the daemon
    serves one request at a time in arrival order, so it takes effect after the
    in-flight command and anything already waiting ahead of it, and abandons
    none of them.
    """

    def work() -> dict[str, Any]:
        return {
            "ok": True,
            "target": "transfer" if params.transfer else "login",
            "broker": _connection().stop_broker(reason=params.reason, transfer=params.transfer, force=params.force),
        }

    return await _run_tool(work)


@mcp.tool(
    name="o2_exec",
    annotations={"title": "Run a command on O2", "readOnlyHint": False, "openWorldHint": True},
)
async def o2_exec(params: RunInput) -> str:
    """Run an arbitrary shell command on an O2 login node (debugging / inspection).

    Reuses the broker's existing SSH session channel and refuses if none is
    running — start one first with o2_start_broker. Returns JSON with
    returncode/stdout/stderr.

    That channel is serialized and shared by every MCP process on this
    workstation, so one command occupies it for all of them, and the broker
    cannot answer a status ping while it runs. Do not wait inside a command: a
    remote sleep, a `while ... squeue` poll, or any other blocking wait holds
    the shared channel for its full duration and needs a long timeout_seconds
    that starves every other caller. Poll from here instead — submit with
    o2_submit_job, then call o2_job_status or o2_squeue repeatedly, which
    releases the channel between checks. When a caller is blocked behind a
    long command, o2_local_status names that command and its elapsed time.
    """

    def work() -> dict[str, Any]:
        result = _connection().run(params.command, timeout=params.timeout_seconds)
        return {"ok": result.ok, **_command_payload(result)}

    return await _run_tool(work)


# Flags that carry a resource shape. Their PRESENCE is what is detected here,
# never their values: spotting that "--mem" occurs is trivial and safe, while
# deciding it means 32 GB total is the sbatch parser o2_price_job deliberately
# does not have. A warning that names what it saw is useful; one that fires on
# every submission is noise, and noise is ignored.
# Every sbatch option that states or shapes the resources a job is allocated,
# audited against the sbatch(1) option list in one pass rather than added one
# at a time as each is noticed missing. A flag absent here is a job recorded as
# carrying no resource request -- a MISSED warning, the direction that costs
# someone real fair share.
#
# Not here on purpose:
#   --oversubscribe   permits sharing a node, does not grow the allocation
#   --exclusive and the socket/core topology options live in
#     billing.UNPRICEABLE_OPTIONS instead, because they cannot be priced at all
#
# Short forms are included only where sbatch(1) documents one: -c, -n, -N, -p,
# -G and -S. The remaining long names have none.
_RESOURCE_FLAGS = (
    "--mem",
    "--mem-per-cpu",
    "--mem-per-gpu",
    "--mincpus",
    "--cpus-per-task",
    "-c",
    "--cpus-per-gpu",
    "--ntasks",
    "-n",
    "--ntasks-per-node",
    "--ntasks-per-core",
    "--ntasks-per-gpu",
    "--nodes",
    "-N",
    "--gres",
    "--gpus",
    "-G",
    "--gpus-per-task",
    "--gpus-per-node",
    "--tres-per-task",
    "--core-spec",
    "-S",
    "--thread-spec",
    # An explicit node list sets the node count when nothing else does --
    # --nodelist=node[01-04] is four nodes -- so it sizes the allocation. -x /
    # --exclude is not here: it removes candidates without changing the size.
    "--nodelist",
    "-w",
    "--nodefile",
    "-F",
    "--partition",
    "-p",
)


def _submitted_remote_path(params: SubmitInput) -> str | None:
    """The remote script this call will actually submit, if it submits one.

    SubmitInput permits script_text and remote_script_path together, and the
    submit path prefers script_text. Both the directive read and the record it
    feeds have to follow that same precedence, or a submission carrying both
    reports the flags of a script that never runs -- or calls one unreadable
    while submitting a script read perfectly well. One predicate, so the two
    paths cannot answer this differently.
    """
    if params.script_text is not None:
        return None
    return params.remote_script_path


def _remote_directives(path: str) -> list[str] | None:
    """The #SBATCH lines of a script already on O2, or None if unreadable.

    One cheap read, so a remote submission gets the same check as an inlined
    one. Only the directive lines come back, never the script body: this looks
    for which flags are SET, and the rest of the file is the user's code with
    no reason to be pulled here.

    None on any failure. An unreadable script must not block a submission --
    the check exists to enrich a record, and a submission that would otherwise
    succeed cannot be made to fail by it.
    """
    try:
        result = _connection().run(
            # Redirected, not passed as an argument: awk has no `--` end-of-
            # options marker, so a path is safer arriving through the shell,
            # which also keeps a leading dash from reading as an option.
            "awk '/^[[:space:]]*#SBATCH([[:space:]]|$)/{print; next} "
            "/^[[:space:]]*($|#)/{next} {exit}' "
            f"< {_quote_remote_path(path)}",
            timeout=15.0,
        )
    except Exception:
        return None
    # awk rather than grep, because sbatch stops reading directives at the first
    # line that is neither blank nor a comment: a #SBATCH sitting below the
    # script's code is inert, and reporting it warns about an option the job
    # does not have. This stops where sbatch stops.
    #
    # awk exits 0 whether or not it printed anything -- "no directives" is an
    # answer -- and nonzero when it could not read the file, which is not. An
    # earlier `|| true` collapsed the two, so a missing script came back as a
    # script with no resource flags: the precise claim this function exists to
    # avoid making.
    if result.returncode != 0:
        return None
    return [line for line in (result.stdout or "").splitlines() if line.strip()]


# `#SBATCH` must end at a boundary. Without it `#SBATCH_DISABLED --exclusive`
# -- an ordinary way to switch a directive off -- reads as a live directive,
# and slicing seven characters leaves the option behind to be reported.
_SBATCH_LINE = re.compile(r"^\s*#SBATCH(?=\s|$)")


def _leading_directives(script: str) -> list[str]:
    """The #SBATCH lines sbatch will actually read.

    sbatch stops processing directives at the first line that is neither blank
    nor a comment, so a #SBATCH below the script's code is inert. Collecting it
    warns about an option the job does not have.
    """
    lines = []
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if _SBATCH_LINE.match(line):
                lines.append(line)
            continue
        break
    return lines


def _option_tokens(params: SubmitInput, remote_directives: list[str] | None = None) -> list[str]:
    """The sbatch options this call can see, split the way sbatch sees them.

    Tokenised, not scanned as text. submit() shell-quotes each sbatch_args
    element into one argument, so an option name sitting inside another
    option's VALUE -- `--comment=do not use --exclusive` -- is not an option
    being set, and a substring scan of the joined string called it one.

    Built once and shared, so a second scan cannot re-derive the script_text /
    remote_script_path precedence and get it wrong. `is not None`, not
    truthiness -- an empty script_text is still what submit_text() sends.
    """
    tokens = list(params.sbatch_args or [])
    if params.script_text is not None:
        lines = _leading_directives(params.script_text)
    elif remote_directives:
        lines = list(remote_directives)
    else:
        lines = []
    for line in lines:
        match = _SBATCH_LINE.match(line)
        if not match:
            continue
        body = line[match.end() :]
        try:
            # comments=False: sbatch reads a directive's arguments directly, so
            # a `#` inside a value is part of it. Splitting on comments turned
            # `--comment=issue#123 --exclusive` into a line ending at the hash
            # and dropped the option after it.
            tokens += shlex.split(body)
        except ValueError:
            # An unbalanced quote is not ours to resolve; take the words as
            # they fall rather than dropping the line and reporting silence.
            tokens += body.split()
    return tokens


def _value_after(flag: str, tokens: list[str]) -> str | None:
    """The value given to `flag`, attached or separated, or None if unset."""
    for index, token in enumerate(tokens):
        if token == flag:
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            return "" if following.startswith("-") else following
        if token.startswith(flag + "="):
            return token[len(flag) + 1 :]
    return None


def _flag_is_set(flag: str, tokens: list[str]) -> bool:
    """Whether `flag` appears as an option, not inside another one's value."""
    for token in tokens:
        if token == flag or token.startswith(flag + "="):
            return True
        # Short flags may carry the value attached -- -c4, -N2, -pshort are all
        # ordinary sbatch. Long ones may not, so "--mem" never matches
        # "--mem-per-cpu".
        if not flag.startswith("--") and len(token) > len(flag) and token.startswith(flag):
            return True
    return False


_ZERO_MEM = re.compile(r"^0[KMGT]?B?$", re.IGNORECASE)


def _unpriceable_options_seen(params: SubmitInput, remote_directives: list[str] | None = None) -> list[str]:
    """Which options appear that o2_price_job refuses to price at all.

    Derived from billing.UNPRICEABLE_OPTIONS rather than a second hand-kept
    list, so an option added there is caught here without a matching edit.

    These matter more than the priceable ones, not less: a script whose only
    directive is --exclusive is among the most expensive things submittable and
    drew no warning at all while this scan looked only for resource flags. They
    need their own note too -- o2_price_job answers `unpriceable` for these, so
    telling the reader to go and price the shape sends them nowhere.
    """
    tokens = _option_tokens(params, remote_directives)
    seen = set()
    for option in billing.UNPRICEABLE_OPTIONS:
        if option == "hetjob":
            # Slurm separates heterogeneous components either with the `hetjob`
            # directive or with a lone `:` between argument groups, and the
            # wrapper forwards that colon through as its own argument. A value
            # containing a colon (--gres=gpu:1) is never a lone token.
            found = "hetjob" in tokens or ":" in tokens
        elif option == "--mem=0":
            value = _value_after("--mem", tokens)
            found = value is not None and bool(_ZERO_MEM.match(value))
        elif option == "--exclusive":
            # sbatch(1) applies the whole-node rule only when no scope is
            # given: with --exclusive=user or =mcs the job is allocated what it
            # asked for, so it prices normally and warning about it would talk
            # the reader out of an option that costs them nothing extra.
            found = "--exclusive" in tokens
        else:
            # A short alias is the same option: `-O` is --overcommit and
            # `-B2:8:2` is --extra-node-info, and checking only the long
            # spelling let either through with no warning at all.
            found = _flag_is_set(option, tokens)
            alias = billing.UNPRICEABLE_ALIASES.get(option)
            if alias and _flag_is_set(alias, tokens):
                found = True
        if found:
            seen.add(option)
    return sorted(seen)


def _resource_flags_seen(params: SubmitInput, remote_directives: list[str] | None = None) -> list[str]:
    """Which resource-bearing flags appear in what this call can actually see."""
    tokens = _option_tokens(params, remote_directives)
    return sorted(flag for flag in _RESOURCE_FLAGS if _flag_is_set(flag, tokens))


def _pricing_record(params: SubmitInput, remote_directives: list[str] | None = None) -> dict[str, Any]:
    """What this submission can say about having been priced.

    Advisory only -- it never blocks. A receipt proves a price was obtained and
    what for; it cannot prove the price describes THIS script, which would need
    the parser that was kept out of o2_price_job on purpose. Recording it puts
    the shape beside the job id so a skipped price is visible afterwards rather
    than silent.
    """
    receipt = billing.parse_price_receipt(params.priced or "")
    unpriceable = _unpriceable_options_seen(params, remote_directives)
    if receipt:
        # `priced` answers "was THIS submission's shape priced?", so an
        # unpriceable option makes it false however valid the receipt is: the
        # branch below proves o2_price_job cannot have priced this allocation.
        # The receipt is still reported -- it is evidence about some shape, and
        # discarding it would lose that -- but a client gating on the boolean
        # must not read it as this job having a price.
        record: dict[str, Any] = {"priced": not unpriceable, "receipt": receipt}
        if unpriceable:
            # A receipt cannot describe THIS allocation: price() refuses every
            # option in that table, so whatever was priced, it was a different
            # shape. Returning early on any valid receipt made the warning
            # silenceable by passing an unrelated one -- the reverse of what a
            # receipt is for.
            record["unpriceable_options_seen"] = unpriceable
            record["note"] = (
                "A receipt was supplied, but this submission also sets "
                + ", ".join(unpriceable)
                + " -- and o2_price_job refuses to price those, so the receipt describes a "
                "different shape than the one being submitted. Read the price as a floor, "
                "not as this job's cost: "
                + "; ".join(f"{opt} {billing.UNPRICEABLE_OPTIONS[opt]}" for opt in unpriceable)
                + "."
            )
        elif _submitted_remote_path(params) and remote_directives is None:
            # Unknown is not absent -- the principle this whole record is built
            # on, and the receipt branch was quietly breaking it. A valid
            # receipt says a price was obtained; it cannot say the script has
            # no option that would invalidate it, and here the script could not
            # be read to check.
            record["note"] = (
                "A receipt was supplied, but the script's #SBATCH lines could not be read "
                "on O2, so whether it sets an option that cannot be priced -- --exclusive "
                "and the like -- is unknown here. The receipt is recorded as given."
            )
        return record
    seen = _resource_flags_seen(params, remote_directives)
    record = {"priced": False, "resource_flags_seen": seen}
    if unpriceable:
        record["unpriceable_options_seen"] = unpriceable
    if unpriceable:
        # Ahead of the ordinary warning AND ahead of the malformed-receipt one:
        # these are the costly options, and the advice differs. o2_price_job
        # answers `unpriceable` for them, so "price the shape" would send the
        # reader to a refusal. Gating this on `not params.priced` let an
        # unreadable `priced` string suppress it -- the same silencing a valid
        # receipt used to buy, reached through the other branch.
        record["note"] = (
            "This submission sets "
            + ", ".join(f"{opt} ({billing.UNPRICEABLE_OPTIONS[opt]})" for opt in unpriceable)
            + ". o2_price_job cannot price these from the directives alone, so no receipt "
            "is possible and none is expected. Their cost depends on the nodes Slurm "
            "picks; if that is not what was intended, drop the option and price the "
            "shape instead."
        )
        if params.priced:
            record["note"] += (
                " The `priced` value supplied is also not a recognisable o2_price_job "
                "receipt, so nothing about a price is recorded here either."
            )
    elif params.priced:
        record["note"] = (
            "A `priced` value was given but is not a recognisable o2_price_job "
            "receipt, so nothing about the price is recorded here."
        )
    elif seen:
        record["note"] = (
            "This submission sets " + ", ".join(seen) + " and carries no price. "
            "Fair share is charged on the ALLOCATION, so an ordinary-looking "
            "--mem can overcharge by a whole billing unit with nothing in the "
            "job's output to reveal it. Price the shape with o2_price_job and "
            "pass its `receipt` to record what was priced."
        )
    elif _submitted_remote_path(params) and remote_directives is None:
        record["note"] = (
            "The script lives on O2 and its #SBATCH lines could not be read, so "
            "whether it requests resources is unknown here and no price "
            "accompanied it. Price the shape with o2_price_job if it does."
        )
    return record


def _absolute_remote_path(value: str, *, label: str) -> str:
    """Require an absolute, traversal-free remote path before it reaches a shell."""

    cleaned = value.strip()
    if not cleaned.startswith("/") or ".." in PurePosixPath(cleaned).parts:
        raise LaunchEvidenceError(f"{label} must be an absolute normalized O2 path")
    return cleaned


_MARKER_DIAGNOSTIC_SIZE = "===DIAGNOSTICSIZE==="
_MARKER_PLAN_SIZE = "===PLANSIZE==="
_MARKER_OWNER_SIZE = "===OWNERSIZE==="
_MARKER_MANIFEST_SIZE = "===MANIFESTSIZE==="
_MARKER_RESOLVED = "===RESOLVED==="
_LAUNCH_EVIDENCE_MARKERS = (
    _MARKER_DIAGNOSTIC_SIZE,
    _MARKER_PLAN_SIZE,
    _MARKER_OWNER_SIZE,
    _MARKER_MANIFEST_SIZE,
    _MARKER_RESOLVED,
)


# Opening each payload with O_NOFOLLOW and hashing that descriptor is the only
# way to make identity and content one act. `sha256sum` has no no-follow mode and
# a shell cannot hold a descriptor across two commands, so this runs a small
# program on the cluster instead -- a command like any other, staging nothing.
# It walks each relative path component by component with openat(O_NOFOLLOW), so
# a symlinked ancestor is refused as well as a symlinked leaf, and reports the
# package directory's inode from the very descriptor it read through. The root
# is opened O_NOFOLLOW too: `realpath` runs earlier, so a package path replaced
# with a symlink after that would otherwise be followed here and report the
# target's inode as though it were the package's.
#
# /usr/bin/python3 on O2 is 3.9, so this stays 3.9 syntax.
_NOFOLLOW_HASHER = """
import errno, hashlib, json, os, stat, sys
root = sys.argv[1]
names = [n for n in sys.stdin.read().split(chr(10)) if n]
out = {"digests": {}, "errors": {}, "package_inode": 0}
O_SEARCH = getattr(os, "O_PATH", 0)
def open_root(path):
    # O_NOFOLLOW on the absolute path guards only the FINAL component, so an
    # ancestor the publisher controls could be renamed and replaced with a
    # symlink after realpath ran; the open would follow it and still reach the
    # approved inode. Walk down from "/" instead, refusing a link at every
    # step, exactly as the payload paths below are walked. "/" itself cannot
    # be a symlink, so it is the one component opened without O_NOFOLLOW.
    #
    # Ancestors are only traversed, never read, so ask for a search-only
    # descriptor. Opening them O_RDONLY would demand read permission that plain
    # pathname resolution never needs, and an execute-only ancestor -- mode 0711
    # is ordinary for shared parents -- would make a perfectly valid package
    # unmintable. O_PATH is Linux-only and degrades to a normal open elsewhere,
    # which only affects local development.
    parts = [p for p in path.split("/") if p]
    fd = os.open("/", os.O_DIRECTORY | (O_SEARCH or os.O_RDONLY))
    try:
        for part in parts[:-1]:
            nxt = os.open(part, os.O_DIRECTORY | os.O_NOFOLLOW | (O_SEARCH or os.O_RDONLY), dir_fd=fd)
            os.close(fd)
            fd = nxt
        if parts:
            # The package directory itself is the one being attested, and its
            # inode is reported from this very descriptor.
            last = os.open(parts[-1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = last
    except BaseException:
        os.close(fd)
        raise
    return fd
try:
    rfd = open_root(root)
except OSError as exc:
    out["errors"][root] = (
        "symlink" if exc.errno == errno.ELOOP else errno.errorcode.get(exc.errno, str(exc.errno))
    )
    sys.stdout.write(json.dumps(out))
    raise SystemExit(0)
try:
    out["package_inode"] = os.fstat(rfd).st_ino
    for name in names:
        parts = name.split("/")
        opened = []
        try:
            cur = rfd
            for part in parts[:-1]:
                cur = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur)
                opened.append(cur)
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=cur)
            opened.append(fd)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                out["errors"][name] = "not a regular file"
                continue
            h = hashlib.sha256()
            while True:
                chunk = os.read(fd, 1048576)
                if not chunk:
                    break
                h.update(chunk)
            out["digests"][name] = h.hexdigest()
        except OSError as exc:
            out["errors"][name] = (
                "symlink" if exc.errno == errno.ELOOP else errno.errorcode.get(exc.errno, str(exc.errno))
            )
        finally:
            for f in opened:
                os.close(f)
finally:
    os.close(rfd)
sys.stdout.write(json.dumps(out))
"""

# The interpreter is named absolutely so no PATH or module environment can
# substitute another one; it is present on O2 without sourcing anything.
_NOFOLLOW_PYTHON = "/usr/bin/python3"


def _hash_without_following(*, package_path: str, names: list[str], timeout: float) -> tuple[dict[str, str], int]:
    """Hash each name through an O_NOFOLLOW descriptor, and report the package inode.

    Returns the digests and the inode of the directory the hashes were read
    through, so the caller can bind that identity to the one the run recorded.
    """

    program = shlex.quote(_NOFOLLOW_HASHER)
    command = f"{_NOFOLLOW_PYTHON} -c {program} {shlex.quote(package_path)}"
    digests: dict[str, str] = {}
    inodes: set[int] = set()
    for batch in _hash_batches(names):
        result = _connection().run(command, timeout=timeout, input_text="\n".join(batch) + "\n")
        if not result.ok:
            raise LaunchEvidenceError(
                "could not hash the package on O2 without following links: {}".format(
                    (result.stderr or "").strip()[:400]
                )
            )
        _refuse_truncated_read(result, label="the package hash read")
        payload = parse_json_artifact(result.stdout, label="the no-follow hash output")
        errors = payload.get("errors") or {}
        if errors:
            shown = "; ".join(f"{name}: {reason}" for name, reason in sorted(errors.items())[:5])
            more = f" (and {len(errors) - 5} more)" if len(errors) > 5 else ""
            raise LaunchEvidenceError(
                "refusing to mint launch evidence; these package entries could not be read as ordinary "
                f"files through a no-follow open, so the package is not one this can attest: {shown}{more}"
            )
        inode = payload.get("package_inode")
        if not isinstance(inode, int) or isinstance(inode, bool) or inode <= 0:
            raise LaunchEvidenceError("the package directory reported no usable inode")
        inodes.add(inode)
        for name, digest in (payload.get("digests") or {}).items():
            if name not in batch:
                raise LaunchEvidenceError(f"the hash output named {name!r}, which was not requested")
            digests[name] = digest
    missing = [name for name in names if name not in digests]
    if missing:
        raise LaunchEvidenceError("the package hash returned nothing for: {}".format(", ".join(sorted(missing)[:5])))
    if len(inodes) != 1:
        raise LaunchEvidenceError("the package directory changed identity while it was being read")
    return digests, inodes.pop()


def _refuse_truncated_read(result: CommandResult, *, label: str) -> None:
    """Refuse an artifact the broker cut short.

    The broker caps captured output and reports the cut only as a note on
    stderr, leaving the return code 0. A partial digest listing is
    indistinguishable from a package missing files, so a truncated read must
    end the mint rather than quietly shrink what the record covers.
    """

    if BROKER_TRUNCATION_NOTE in (result.stderr or ""):
        raise LaunchEvidenceError(f"{label} was truncated by the broker's output cap, so it cannot be bound")


def _read_launch_artifacts(
    *, diagnostic_path: str, plan_path: str, package_path: str, timeout: float
) -> dict[str, Any]:
    """Read every artifact back off the cluster through the authenticated broker.

    Reading them here rather than accepting caller-supplied content is the point:
    the record must attest what is actually on O2. One command keeps this to a
    single hold on the shared channel; hashing the payloads the manifest names
    needs a second one, because that list only exists once this has returned.
    """

    manifest_path = str(PurePosixPath(package_path) / "SHA256SUMS")
    owner_path = str(PurePosixPath(package_path) / "PUBLICATION_OWNER.json")
    command = "; ".join(
        [
            f"printf '%s\\n' {shlex.quote(_MARKER_DIAGNOSTIC_SIZE)}",
            f"stat -c %s -- {shlex.quote(diagnostic_path)}",
            f"printf '\\n%s\\n' {shlex.quote(_MARKER_PLAN_SIZE)}",
            f"stat -c %s -- {shlex.quote(plan_path)}",
            f"printf '\\n%s\\n' {shlex.quote(_MARKER_MANIFEST_SIZE)}",
            f"stat -c %s -- {shlex.quote(manifest_path)}",
            f"printf '\\n%s\\n' {shlex.quote(_MARKER_OWNER_SIZE)}",
            f"stat -c %s -- {shlex.quote(owner_path)}",
            f"printf '\\n%s\\n' {shlex.quote(_MARKER_RESOLVED)}",
            f"realpath -- {shlex.quote(package_path)}",
        ]
    )
    # This one is not batched, so a package path long enough to overrun the
    # broker's limit would be rejected as an opaque ValueError. Name it instead.
    if len(command.encode("utf-8")) > MAX_COMMAND_BYTES:
        raise LaunchEvidenceError(f"the artifact read for {package_path!r} does not fit in one broker command")
    result = _connection().run(command, timeout=timeout)
    if not result.ok:
        raise LaunchEvidenceError(
            "could not read the launch artifacts from O2: {}".format((result.stderr or "").strip()[:400])
        )
    _refuse_truncated_read(result, label="the launch artifact read")

    sections: dict[str, str] = {}
    current: str | None = None
    collected: list[str] = []
    for line in result.stdout.splitlines():
        if line.strip() in _LAUNCH_EVIDENCE_MARKERS:
            if current is not None:
                sections[current] = "\n".join(collected)
            current = line.strip()
            collected = []
            continue
        collected.append(line)
    if current is not None:
        sections[current] = "\n".join(collected)
    missing = [marker for marker in _LAUNCH_EVIDENCE_MARKERS if marker not in sections]
    if missing:
        raise LaunchEvidenceError("artifact read returned no {} section".format(", ".join(missing)))

    # The required package files are hashed through no-follow descriptors, which
    # is also where the package directory's own inode is observed.
    package_digests, package_inode = _hash_without_following(
        package_path=package_path, names=list(required_package_files()), timeout=timeout
    )
    # The manifest is read and hashed by different commands, so the bytes that
    # choose the payloads must be proven to be the file whose digest the record
    # reports. That check lives in parse_encoded_checksum_manifest.
    manifest = parse_encoded_checksum_manifest(
        _read_artifact_base64(
            path=manifest_path,
            size_section=sections[_MARKER_MANIFEST_SIZE],
            timeout=timeout,
            label="SHA256SUMS",
            ceiling=_MAX_MANIFEST_BYTES,
        ),
        expected_sha256=package_digests.get("SHA256SUMS"),
    )
    # The pathname the caller spelled can reach a different directory than it
    # names, so the record says what the cluster resolved it to.
    resolved = [line.strip() for line in sections[_MARKER_RESOLVED].splitlines() if line.strip()]
    if len(resolved) != 1:
        raise LaunchEvidenceError("the package directory did not resolve to exactly one path on the cluster")
    # The diagnostic and the plan are read the same way. Neither has a size
    # bound in its own schema, and together they shared the artifact read's one
    # 1 MiB stream, so a large but legitimate plan made a package unmintable.
    diagnostic_text = _decoded_artifact(
        path=diagnostic_path,
        size_section=sections[_MARKER_DIAGNOSTIC_SIZE],
        timeout=timeout,
        label="run diagnostic",
    )
    plan_text = _decoded_artifact(
        path=plan_path, size_section=sections[_MARKER_PLAN_SIZE], timeout=timeout, label="execution plan"
    )
    return {
        "resolved_package_path": resolved[0],
        "package_inode": package_inode,
        "diagnostic": parse_json_artifact(diagnostic_text, label="run diagnostic"),
        "plan": parse_json_artifact(plan_text, label="execution plan"),
        # The owner marker is the artifact tying the package to this plan, so
        # like SHA256SUMS the bytes parsed must be the file that was hashed.
        "owner": parse_encoded_json_artifact(
            _read_artifact_base64(
                path=owner_path,
                size_section=sections[_MARKER_OWNER_SIZE],
                timeout=timeout,
                label="publication owner",
                ceiling=_MAX_JSON_ARTIFACT_BYTES,
            ),
            expected_sha256=package_digests.get("PUBLICATION_OWNER.json"),
            label="publication owner",
        ),
        "checksum_manifest": manifest,
        "package_digests": package_digests,
    }


def _hash_package_payloads(
    *, package_path: str, manifest: dict[str, str], timeout: float
) -> tuple[dict[str, str], int]:
    """Hash every payload SHA256SUMS names through a no-follow descriptor.

    This is a second pass because it has to be: the payload list only exists
    once the manifest has been read. Running ``sha256sum -c`` remotely would
    avoid it at the cost of putting the comparison back inside a process whose
    verdict this record exists to check, so the digests come back raw and
    ``build_launch_evidence`` decides.

    The package inode comes back with them, observed through the same descriptor
    the payloads were read under, so the caller can confirm the directory did not
    change identity between the two passes.
    """

    return _hash_without_following(package_path=package_path, names=sorted(manifest), timeout=timeout)


# base64 expands by 4/3, and the broker caps captured output at 1 MiB, so a file
# read whole overruns it somewhere above ~750 KiB. Packages with several thousand
# files reach that for SHA256SUMS, and a plan with a large dataset list reaches it
# too, so both are chunked rather than being the thing that makes a large but
# legitimate package unmintable.
_ARTIFACT_CHUNK_BYTES = 384 * 1024
# The size that drives the read loop comes from `stat` on a file the package
# controls, so it is untrusted input to a resource decision. A sparse manifest
# claiming a terabyte would otherwise mean millions of sequential broker commands
# and an attempt to accumulate that much base64 in memory. 8 MiB is about 30,000
# manifest entries at typical path lengths, far beyond any real package.
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
# A run diagnostic or a frozen plan is JSON describing one stage; 4 MiB of it is
# already far past anything a stage legitimately produces.
_MAX_JSON_ARTIFACT_BYTES = 4 * 1024 * 1024


def _read_artifact_base64(*, path: str, size_section: str, timeout: float, label: str, ceiling: int) -> str:
    """Read one artifact in bounded pieces and return its base64.

    Every artifact read this way is larger than the broker's 1 MiB output cap
    for some legitimate input -- a package with thousands of payloads, a plan
    with a large dataset list -- and reading one whole turns that into an
    unmintable package rather than a slower read. The size comes from `stat` on
    a file the package controls, so it is bounded before it drives the loop.

    No per-chunk check is needed: SHA256SUMS is verified against the digest
    recorded for that filename and the plan against `diagnostic.plan_sha256`, so
    anything that changed mid-read fails where a whole-file read would have.
    """

    reported = [line.strip() for line in size_section.splitlines() if line.strip()]
    if len(reported) != 1 or not reported[0].isdigit():
        raise LaunchEvidenceError(f"the size of {label} could not be read from the cluster")
    size = int(reported[0])
    if size == 0:
        raise LaunchEvidenceError(f"{label} is empty")
    if size > ceiling:
        raise LaunchEvidenceError(
            f"{label} is {size} bytes, over the {ceiling}-byte ceiling this reads. Something that large is "
            "not part of a package this tool can attest, and reading it would hold the shared channel for "
            "thousands of commands"
        )
    encoded: list[str] = []
    for start in range(0, size, _ARTIFACT_CHUNK_BYTES):
        count = min(_ARTIFACT_CHUNK_BYTES, size - start)
        command = f"tail -c +{start + 1} -- {shlex.quote(path)} | head -c {count} | base64"
        result = _connection().run(command, timeout=timeout)
        if not result.ok:
            raise LaunchEvidenceError(
                "could not read {} from O2: {}".format(label, (result.stderr or "").strip()[:400])
            )
        _refuse_truncated_read(result, label=f"the {label} read")
        encoded.append("".join(result.stdout.split()))
    # The size was taken once, before the digest was computed and before these
    # reads. A file that grew afterwards would still hand back a prefix that
    # hashes to the recorded digest, so the manifest could be extended with
    # entries this never sees. Probing one byte past the end refuses that.
    probe = f"tail -c +{size + 1} -- {shlex.quote(path)} | head -c 1 | wc -c"
    result = _connection().run(probe, timeout=timeout)
    if not result.ok:
        raise LaunchEvidenceError(
            "could not confirm the end of {} on O2: {}".format(label, (result.stderr or "").strip()[:400])
        )
    if result.stdout.strip() != "0":
        raise LaunchEvidenceError(
            f"{label} grew while it was being read, so the bytes hashed are only a prefix of the file now " "on disk"
        )
    return "".join(encoded)


def _decoded_artifact(*, path: str, size_section: str, timeout: float, label: str) -> str:
    """Read one JSON artifact in bounded pieces and return its text."""

    encoded = _read_artifact_base64(
        path=path,
        size_section=size_section,
        timeout=timeout,
        label=label,
        ceiling=_MAX_JSON_ARTIFACT_BYTES,
    )
    try:
        return base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, ValueError) as error:
        raise LaunchEvidenceError(f"{label} did not come back as decodable UTF-8: {error}") from error


# One call per payload would be correct but would take a hold on the shared
# channel per file. Batching bounds what comes back instead: each name costs its
# own length plus a 64-character digest and a little JSON, and the broker caps
# captured output at 1 MiB. The names go out on stdin, which has its own 1 MiB
# cap and is the smaller side, so bounding the reply bounds both.
_HASH_REPLY_BUDGET = 600 * 1024


def _hash_batches(names: list[str]) -> Iterator[list[str]]:
    """Group payload names into calls whose replies the broker will carry.

    A package with thousands of files overruns the output cap in one call and
    the whole reply is truncated, so an entirely legitimate package could never
    be attested.
    """

    batch: list[str] = []
    used = 0
    for name in names:
        cost = len(name.encode("utf-8")) * 2 + 80
        if cost > _HASH_REPLY_BUDGET:
            raise LaunchEvidenceError(f"payload name is too long to hash in one call: {name!r}")
        if batch and used + cost > _HASH_REPLY_BUDGET:
            yield batch
            batch, used = [], 0
        batch.append(name)
        used += cost
    if batch:
        yield batch


def _read_scheduler_record(*, job_id: str, timeout: float) -> dict[str, str]:
    """Ask Slurm accounting who actually ran the job the diagnostic claims.

    Every other field in the record is anchored outside the executed process --
    the plan by digest, the package by rehashing, the directory by cluster-side
    resolution -- and this is what stops the job identity from being the one
    exception. ``-X`` returns the allocation rather than its steps, so a normal
    job is exactly one row.

    ``Comment`` is requested because the plan builder sets it to the plan digest.
    It is not required: jobs submitted before that lands have none, and
    ``build_launch_evidence`` treats an empty comment as an explicit unbound
    field rather than as agreement. A non-empty one must match.
    """

    # Comment carries a 64-character digest plus stage and attempt, so pin its
    # width rather than trusting the display default. Measured on O2 (Slurm
    # 25.11.7) the suffix is inert under -P: default, %20 and %512 all returned
    # the same full 44-character JobName, while the non-parsable form truncated
    # it to "gem_segme+". It is kept anyway as portability insurance, since a
    # build that did apply widths would silently truncate the digest.
    fields = "JobID,State,Account,Partition,Comment%256"
    command = f"sacct -j {shlex.quote(job_id)} -X -n -P -o {shlex.quote(fields)}"
    result = _connection().run(command, timeout=timeout)
    if not result.ok:
        raise LaunchEvidenceError(
            "could not read Slurm accounting for job {}: {}".format(job_id, (result.stderr or "").strip()[:400])
        )
    _refuse_truncated_read(result, label="the Slurm accounting read")
    return parse_scheduler_record(result.stdout, job_id=job_id)


@mcp.tool(
    name="o2_mint_launch_evidence",
    annotations={
        "title": "Mint authenticated O2 launch evidence",
        "readOnlyHint": False,
        "destructiveHint": False,
        # This reads files from O2 and queries Slurm through the broker, so it
        # does interact with entities outside this process, whatever the local
        # policy write alongside it might suggest.
        "openWorldHint": True,
    },
)
async def o2_mint_launch_evidence(params: MintLaunchEvidenceInput) -> str:
    """Bind a finished governed stage into one operator-approved evidence record.

    A repository canary can verify its own scientific path but cannot
    authenticate its own launch, so the authority has to come from outside the
    executed process. This reads the run diagnostic, the frozen plan, the
    publication owner marker, the package's SHA256SUMS, and the package digests
    back off the cluster through the authenticated broker, rehashes every
    payload the manifest names rather than trusting the run's own "verified"
    verdict, confirms the claimed job against Slurm accounting, checks that every
    link agrees -- including that the directory read
    is the package the plan approved -- and records the mint, with the digest of
    the record it approves, in the policy audit ledger.

    Minting attests a finished run. It is deliberately NOT authority to start
    another one: it consumes no login grant and changes no policy mode.
    """

    def work() -> dict[str, Any]:
        diagnostic_path = _absolute_remote_path(params.diagnostic_path, label="diagnostic_path")
        plan_path = _absolute_remote_path(params.plan_path, label="plan_path")
        package_path = _absolute_remote_path(params.package_path, label="package_path")
        artifacts = _read_launch_artifacts(
            diagnostic_path=diagnostic_path,
            plan_path=plan_path,
            package_path=package_path,
            timeout=params.timeout_seconds,
        )
        payload_digests, payload_pass_inode = _hash_package_payloads(
            package_path=package_path,
            manifest=artifacts["checksum_manifest"],
            timeout=params.timeout_seconds,
        )
        if payload_pass_inode != artifacts["package_inode"]:
            raise LaunchEvidenceError(
                "refusing to mint launch evidence; the package directory changed identity between the "
                f"metadata read (inode {artifacts['package_inode']}) and the payload read "
                f"(inode {payload_pass_inode})"
            )
        scheduler_record = _read_scheduler_record(
            job_id=claimed_job_id(artifacts["diagnostic"]), timeout=params.timeout_seconds
        )

        def mint(approval: dict[str, Any]) -> dict[str, Any]:
            return build_launch_evidence(
                diagnostic=artifacts["diagnostic"],
                plan=artifacts["plan"],
                package_digests=artifacts["package_digests"],
                checksum_manifest=artifacts["checksum_manifest"],
                payload_digests=payload_digests,
                owner=artifacts["owner"],
                approval=approval,
                stage=params.stage,
                read_back_package_path=package_path,
                resolved_package_path=artifacts["resolved_package_path"],
                observed_package_inode=artifacts["package_inode"],
                scheduler_record=scheduler_record,
            )

        # Verify BEFORE recording an approval: a chain that does not agree must
        # not leave an audit entry implying it did.
        preliminary = mint({})
        approval = _connection().policy.record_launch_evidence_mint(
            expected_revision=params.expected_revision,
            expected_generation=params.expected_generation,
            approval_reference=params.approval_reference,
            stage=params.stage,
            job_id=str(preliminary["submission"]["job_id"]),
            package=str(preliminary["destination"]["package"]),
            # The ledger entry records the digest of the record it approves, so
            # an edited copy no longer matches the approval it carries.
            evidence_sha256=evidence_content_digest(preliminary),
            plan_sha256=str(preliminary["approved_plan"]["sha256"]),
        )
        record = mint(approval)
        return {
            "ok": True,
            "launch_evidence": record,
            "launch_evidence_sha256": launch_evidence_digest(record),
            "evidence_content_sha256": evidence_content_digest(record),
        }

    return await _run_tool(work)


@mcp.tool(
    name="o2_submit_job",
    annotations={"title": "Submit a Slurm job", "readOnlyHint": False, "openWorldHint": True},
)
async def o2_submit_job(params: SubmitInput) -> str:
    """Submit an sbatch job, returning the parsed Slurm job id.

    Provide either remote_script_path (a script already on O2) or script_text +
    remote_path (stage the script to O2, then submit). Returns JSON:
    {"submitted": bool, "job_id": str|null, "returncode": int, "stdout": str, "stderr": str}.
    """

    def work() -> dict[str, Any]:
        slurm = O2Slurm(_connection())
        # Read the remote script's directives BEFORE submitting, so an
        # unreadable script surfaces as "unknown" rather than as a submission
        # that already happened.
        directives = None
        submitted_path = _submitted_remote_path(params)
        if submitted_path:
            # Read them even when a receipt was supplied. A receipt does not
            # rule out an unpriceable option -- it cannot describe one -- so
            # skipping the read here would have left that warning reachable for
            # an inline script and silent for a remote one.
            directives = _remote_directives(submitted_path)
        if params.script_text is not None:
            if not params.remote_path:
                return {"ok": False, "error": "bad_input", "message": "remote_path is required with script_text."}
            res = slurm.submit_text(params.script_text, params.remote_path, sbatch_args=params.sbatch_args)
        elif params.remote_script_path:
            res = slurm.submit(params.remote_script_path, sbatch_args=params.sbatch_args)
        else:
            return {
                "ok": False,
                "error": "bad_input",
                "message": "Provide remote_script_path or script_text+remote_path.",
            }
        return {
            "ok": res.submitted,
            "submitted": res.submitted,
            "job_id": res.job_id,
            "pricing": _pricing_record(params, directives),
            **_command_payload(res.command),
        }

    return await _run_tool(work)


@mcp.tool(
    name="o2_squeue",
    annotations={"title": "List Slurm jobs", "readOnlyHint": True, "openWorldHint": True},
)
async def o2_squeue(params: QueueInput) -> str:
    """List the user's current Slurm jobs (squeue) as structured rows.

    Returns JSON: {"jobs": [{"job_id","name","state","elapsed","time_limit","nodes","reason"}, ...]}.
    """

    def work() -> dict[str, Any]:
        return {"ok": True, "jobs": O2Slurm(_connection()).queue(params.user)}

    return await _run_tool(work)


@mcp.tool(
    name="o2_job_status",
    annotations={"title": "Slurm job accounting", "readOnlyHint": True, "openWorldHint": True},
)
async def o2_job_status(params: JobIdInput) -> str:
    """Get sacct accounting for one job (state, elapsed, exit code, memory).

    Returns JSON: {"rows": [{"job_id","name","state","elapsed","exit_code","max_rss",...}, ...]}.
    """

    def work() -> dict[str, Any]:
        return {"ok": True, "rows": O2Slurm(_connection()).job_status(params.job_id)}

    return await _run_tool(work)


@mcp.tool(
    name="o2_tail_log",
    annotations={"title": "Tail an O2 log", "readOnlyHint": True, "openWorldHint": True},
)
async def o2_tail_log(params: TailLogInput) -> str:
    """Tail the last N lines of a remote Slurm log file.

    Logs typically live under ~/logs/o2/<job-name>_<jobid>.out|.err.
    Returns JSON with the log text in stdout.
    """

    def work() -> dict[str, Any]:
        result = O2Slurm(_connection()).tail_log(params.remote_path, lines=params.lines)
        return {"ok": result.ok, **_command_payload(result)}

    return await _run_tool(work)


@mcp.tool(
    name="o2_cancel_job",
    annotations={
        "title": "Cancel a Slurm job",
        "readOnlyHint": False,
        "destructiveHint": True,
        "openWorldHint": True,
    },
)
async def o2_cancel_job(params: JobIdInput) -> str:
    """Cancel a running or queued Slurm job (scancel). Destructive: the job is killed."""

    def work() -> dict[str, Any]:
        result = O2Slurm(_connection()).cancel(params.job_id)
        return {"ok": result.ok, **_command_payload(result)}

    return await _run_tool(work)


# --- workspace-layout tools --------------------------------------------------
@mcp.tool(
    name="o2_disk_report",
    annotations={"title": "O2 disk usage + hygiene report", "readOnlyHint": True, "openWorldHint": True},
)
async def o2_disk_report(params: WorkspaceReportInput) -> str:
    """Per-tier disk usage with hygiene flags (regenerable / redundant / misplaced).

    Read-only. Walks the home + scratch tiers (depth 1), classifies each entry, and
    returns totals by disposition plus a reclaimable-bytes estimate. This is the
    repeatable, codified version of a manual disk audit — review it before running
    o2_workspace_gc. See docs/WORKSPACE_LAYOUT.md.
    """

    def work() -> dict[str, Any]:
        report = O2Workspace(_connection()).disk_report(params.roots)
        return {"ok": True, **report}

    return await _run_tool(work)


@mcp.tool(
    name="o2_workspace_gc",
    annotations={
        "title": "Prune regenerable + redundant disk",
        "readOnlyHint": False,
        "destructiveHint": True,
        "openWorldHint": True,
    },
)
async def o2_workspace_gc(params: WorkspaceGcInput) -> str:
    """Prune ONLY the regenerable (caches/builds) and redundant (trash/surplus
    snapshots) findings from o2_disk_report — never relocate/review/keep items.

    Snapshot history is thinned to the newest few; everything else prunable is
    removed. Runs detached and fail-closed. dry_run (default true) returns the
    generated script without executing it.
    """

    def work() -> dict[str, Any]:
        plan = O2Workspace(_connection()).gc(dry_run=params.dry_run, roots=params.roots)
        return {
            "ok": True,
            "dry_run": plan.dry_run,
            "submitted": plan.submitted,
            "pruned_paths": plan.pruned_paths,
            "message": plan.message,
            "script": plan.script,
        }

    return await _run_tool(work)


@mcp.tool(
    name="o2_place",
    annotations={"title": "Resolve canonical output path", "readOnlyHint": True, "openWorldHint": True},
)
async def o2_place(params: PlaceInput) -> str:
    """Resolve the canonical path for an output kind (+ optional project) per the
    workspace tier convention — so outputs land on the right tier, not invented paths.

    e.g. {kind:'results', project:'myproject'} -> /n/groups/tabin/jzhao/results/myproject.
    """

    def work() -> dict[str, Any]:
        return {
            "ok": True,
            "kind": params.kind,
            "project": params.project,
            "path": O2Workspace(_connection()).place(params.kind, params.project),
        }

    return await _run_tool(work)


transfer_tools.register(mcp, sys.modules[__name__])


def main() -> None:
    """Console-script / module entry point: run the stdio MCP server."""
    mcp.run()


@mcp.tool(
    name="o2_price_job",
    annotations={
        "title": "Price an allocation before submitting",
        # True in both senses now: this reads a local cache and writes nothing.
        # Refreshing that cache lives in o2_refresh_billing_weights, because a
        # tool that can write must not claim to be read-only at the point a
        # client decides whether to auto-approve it.
        "readOnlyHint": True,
        "openWorldHint": False,
    },
)
async def o2_price_job(params: PriceJobInput) -> str:
    """Compute what an allocation will cost in Slurm billing units, before it runs.

    Fair share is bought with ALLOCATED resources, and the weighted TRES sum is
    floored -- so memory is sold in whole blocks and a request sitting exactly
    on a block edge pays for a full block while forfeiting the headroom inside
    it. Nothing in a job's own output reveals that, so the moment to check is
    while the request is being written.

    Takes a resource SHAPE, not a script. Reading a script means reimplementing
    sbatch's option semantics -- per-task and per-GPU forms, GRES grammar,
    attached short options, --mem=0 meaning "everything", partition caps that
    silently raise the CPU count -- and any subtle error there produces a
    confident wrong number with nothing to catch it. Read the directives
    yourself, state the shape you found in the plan the user approves, and price
    that: your reading is then visible and can be challenged.

    Pure arithmetic over a cached weight table: no SSH and no Slurm call at all,
    so this answers while the O2 policy is disabled and before any broker
    exists. Populate the cache once with o2_refresh_billing_weights. Returns the price, a breakdown that reconciles to
    the floored units, the nearest memory boundary, and cheaper partitions for
    the identical allocation.

    Advisory only. It never submits, never edits a request, and does not
    recommend holding less memory than a job already has -- an OOM kill bills
    its full elapsed time AND forces a rerun, so a request trimmed too close is
    a net loss, not a smaller win.
    """

    def work() -> dict[str, Any]:
        table: dict[str, billing.Weights] = {}
        captured_at: float | None = None

        cached = billing.load_weight_cache()
        if not cached:
            return {
                "ok": False,
                "error": "no_weight_cache",
                "message": (
                    "No cached TRESBillingWeights. Run o2_refresh_billing_weights "
                    "once while a login broker is ready; afterwards pricing needs "
                    "no connection at all."
                ),
            }
        unsupported = billing.unsupported_billing_model(cached)
        if unsupported:
            # Refused before any number is produced: a price computed under the
            # wrong model is not a smaller error than no price, it is a
            # confident one that reaches an approval.
            return {
                "ok": False,
                "error": "unsupported_billing_model",
                "message": unsupported,
            }
        table = billing.cache_to_table(cached)
        captured_at = cached.get("captured_at")

        unresolved = billing.Request(
            cpus=params.cpus,
            mem_gb=params.mem_gb or 0.0,
            gpus=params.gpus or 0.0,
            # Omitting mem_gb means the shape names no memory, which is not the
            # same as requesting none: the partition default applies.
            mem_specified=params.mem_gb is not None,
            nodes=params.nodes or 1.0,
            nodes_stated=params.nodes is not None,
            gpu_model=params.gpu_model,
            mem_per_cpu_gb=params.mem_per_cpu_gb,
            ntasks=params.ntasks or 1.0,
        )

        try:
            request = billing.resolve_request(unresolved, table, params.partition)
            payload = billing.price(request, table, params.partition, captured_at)
        except billing.BillingError as exc:
            return {"ok": False, "error": "unpriceable", "message": str(exc)}
        # Compare the SAME concrete allocation elsewhere; using an unresolved
        # request priced every alternative with zero memory.
        # `original` is the request as the caller stated it: resolving a
        # candidate from the shape THIS partition produced cannot undo a CPU
        # count its own cap forced.
        payload["alternatives"] = billing.alternatives(request, table, params.partition, original=unresolved)
        # Always, not conditionally: the rows are a price comparison, and a
        # caller who reads them as "partitions that can run this" has been told
        # something this cache cannot know.
        payload["receipt"] = billing.price_receipt(payload)
        payload["alternatives_note"] = billing.alternatives_caveat()
        if request.mem_unknown:
            # An empty list here has a specific cause worth naming: the price
            # stands, the comparison cannot.
            payload["alternatives_note"] = (
                "Not compared: this partition does not bill memory, so the size "
                "was never established -- but any partition that DOES bill it "
                "would be priced as holding none. The price above is exact; "
                "state mem_gb to compare partitions. "
            ) + payload["alternatives_note"]
        payload["ok"] = True
        return payload

    return await _run_tool(work)


class RefreshWeightsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


@mcp.tool(
    name="o2_refresh_billing_weights",
    annotations={
        "title": "Refresh the cached Slurm billing weights",
        # This REPLACES the local weight cache, so it is not read-only. Saying
        # otherwise would let a client auto-approve a write.
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def o2_refresh_billing_weights(params: RefreshWeightsInput) -> str:
    """Re-read TRESBillingWeights from the cluster and replace the local cache.

    Split out from o2_price_job so that tool can honestly claim readOnlyHint:
    pricing reads the cache and writes nothing, while this reads scontrol
    through the broker and replaces what pricing reads. Weights change rarely,
    so this is run once and then not again for a long time.

    Needs a ready login broker. It queries only partition configuration -- no
    job data, and nothing about the cluster is modified.
    """

    def work() -> dict[str, Any]:
        result = _connection().run("scontrol show partition -o", timeout=30.0)
        if not result.ok:
            return {
                "ok": False,
                "error": "weights_unavailable",
                "message": "Could not read partition weights: {}".format((result.stderr or "").strip()[:200]),
            }
        table = billing.parse_weight_table(result.stdout)
        if not table:
            return {
                "ok": False,
                "error": "weights_unavailable",
                "message": (
                    "scontrol returned no partitions with billing weights; the " "existing cache was left untouched."
                ),
            }
        # PriorityFlags is cluster-global and decides whether Billing is the SUM
        # of weighted TRES or their MAX. Weights alone cannot reveal it, and the
        # two imply opposite advice about memory, so the flags are captured with
        # the weights or the cache is not written at all.
        flags_result = _connection().run("scontrol show config", timeout=30.0)
        if not flags_result.ok:
            return {
                "ok": False,
                "error": "weights_unavailable",
                "message": (
                    "Read the partition weights but not PriorityFlags ({}), which "
                    "decides whether billing sums or maximises the weighted TRES. "
                    "The existing cache was left untouched.".format((flags_result.stderr or "").strip()[:120])
                ),
            }
        priority_flags = billing.parse_priority_flags(flags_result.stdout)
        captured_at = time.time()
        billing.save_weight_cache(table, captured_at, priority_flags=priority_flags)
        return {
            "ok": True,
            "captured_at": captured_at,
            "priority_flags": priority_flags,
            # Same predicate the refusal uses: an exact-match copy here once
            # reported "sum" for a cluster the very next price call refused.
            "billing_model": "max" if billing.max_based_flags(priority_flags) else "sum",
            "partitions": sorted(table),
            "unpriceable": {name: w.unpriceable_tres for name, w in sorted(table.items()) if w.unpriceable_tres},
        }

    return await _run_tool(work)


if __name__ == "__main__":  # pragma: no cover
    main()
