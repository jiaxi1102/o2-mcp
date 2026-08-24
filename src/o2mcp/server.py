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

import json
import stat
import subprocess
import sys
from pathlib import Path
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
    transfer_tools,
)
from o2mcp.policy import LoginTarget

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
            "Only affects the fallback path. Stops now go to the broker's control endpoint, which a "
            "separate daemon thread answers immediately even while a command holds the channel, and which "
            "skips work already queued behind it. This flag is used only when no control endpoint exists "
            "-- a daemon started before it -- where a stop is otherwise cancelled once its caller gives up. "
            "No stop abandons a running command."
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


class ProbeInput(BaseModel):
    """Select one existing role-specific broker for an explicit remote probe."""

    model_config = ConfigDict(extra="forbid")
    transfer: bool = Field(
        default=False,
        description="Use the existing transfer-host broker instead of the login-host broker.",
    )


# One command's timeout is one command's hold on a channel shared by every MCP
# process on the workstation, and it is also the watchdog budget the daemon uses
# before it will tear down a silent transport. An hour of either is far past the
# point where waiting remotely should have been a submitted job, so bound both
# with the same number.
MAX_EXEC_TIMEOUT_SECONDS = 300.0


class RunInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    command: str = Field(..., description="Remote shell command to run on an O2 login node.", min_length=1)
    timeout_seconds: float = Field(
        default=120.0,
        description=(
            "Command timeout in seconds, capped at 300. The channel is shared and serialized, so this "
            "is how long one command may block every other caller. Work that needs longer belongs in a "
            "submitted job polled from here, not in a single command."
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
    sbatch_args: list[str] = Field(default_factory=list, description="Extra sbatch flags, e.g. ['--time=02:00:00'].")


class QueueInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    user: str | None = Field(default=None, description="Username for squeue -u (defaults to remote $USER).")


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
        result = conn.run(
            "hostname; whoami; date",
            timeout=conn.config.connect_timeout + 5,
            alias=alias,
            broker_role="transfer" if params.transfer else "login",
        )
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

    A stop is answered on the broker's control endpoint, so it is prompt even
    while the broker is busy. It lets the in-flight command finish and then
    exits: commands that were queued behind it are DISCARDED, not run. A caller
    whose request is discarded is told it was not dispatched, so nothing it
    submitted has run on O2 -- but do not assume queued work completes.
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
        return {"ok": res.submitted, "submitted": res.submitted, "job_id": res.job_id, **_command_payload(res.command)}

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


if __name__ == "__main__":  # pragma: no cover
    main()
