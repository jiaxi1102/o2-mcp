# o2-mcp

Generic, project-agnostic access to the **HMS O2 cluster**, exposed both as a Python
library (`o2mcp`) and as an **MCP server** (`o2-mcp`) so an agent can submit Slurm work,
run remote commands, monitor jobs, move files, and keep disk tidy — without triggering a
Duo push on every action.

Extracted from `clock-oscillation-analysis` so the cluster tooling is shared
infrastructure (used by multiple analysis projects) rather than living inside one of them.
Project-specific layers (e.g. run-organization for a particular pipeline) build *on* this
package rather than living in it.

## Duo model (read this first)

HMS O2 uses Duo **autopush**: every *new* SSH connection can fire a Duo push, even
key-only / BatchMode. The workstation-wide `~/.agent_locks/O2_POLICY.json` state
therefore defaults fail-closed. Ordinary work may reuse an existing persistent
ControlMaster; a new authentication requires a short-lived, client-bound,
host-scoped, one-attempt grant returned by `o2_authorize_login`.

Ordinary commands and transfers are authentication-disabled on purpose. OpenSSH normally
falls back to a standalone connection when a configured control socket disappears; on O2,
that key-based fallback can generate an unexpected Duo request. The MCP therefore passes
`ControlMaster=no`, `PreferredAuthentications=none`, and disables public-key, password,
keyboard-interactive, GSSAPI, and host-based authentication for every non-start operation.
It also disables `ProxyJump` and `ProxyCommand`, because proxy SSH subprocesses would not
inherit the outer client's authentication restrictions, and disables local/known-host command
hooks that could launch another process. Those options still permit reuse of an
already-authenticated master: the MCP first resolves the alias's original `ControlPath` from
an inspected, flattened SSH config and pins that expanded socket with `-S`, preserving `%C`
paths whose hash includes a jump host. Every reuse command then uses `-F /dev/null`; caller
`-F`, `-S`, and `ControlPath` overrides are removed. User overrides must use the unambiguous
`user@alias` form, while port and hostname overrides are rejected because they would no
longer match the pinned socket. SSH and rsync are normalized to the operating system's
`/usr/bin` clients; executable paths or wrappers supplied by callers are rejected. A missing
or failed socket terminates locally instead of opening a replacement login.

OpenSSH evaluates `Match exec` shell predicates even for its nominally local `ssh -G` config
dump. The MCP therefore reads the configured SSH file and recursively flattens `Include`
directives itself, rejects any `Match` block before launching OpenSSH, and runs `ssh -G` only
against that private inspected snapshot. If your normal SSH config contains `Match`, put the
O2 `Host` blocks in a separate Match-free file and set `O2_SSH_CONFIG_FILE` to it.

The optional `o2-transfer` alias is a different host and therefore needs its own
separately approved grant and ControlMaster. A login grant cannot be used for the
transfer host or consumed by another MCP task.

Never open the master in a loop or run authentication tools on a timer. The policy
file has two durable modes: `disabled` blocks new remote operations, while
`reuse_only` permits only existing exact sockets. There is deliberately no durable
`normal` mode. The one-shot grant is atomically removed and converted to an active
attempt receipt before SSH, so a failure, timeout, or crash cannot leave reusable
authorization for a queued task. Policy generation/revision pairs and a stable
internal mutex serialize every transition across MCP processes. Callers pass
both values from one `o2_local_status` snapshot; repair creates a new generation,
so a repeated numeric revision cannot revive stale approval.

**Be on the HMS VPN.** O2 only *skips* Duo for connections from HMS-trusted source IPs — i.e.
when your SSH egresses through the HMS VPN (GlobalProtect), not your normal internet
interface. If the VPN is down (or split-tunnel isn't routing O2's subnet), even the one
`o2_start_master` login comes from a non-HMS IP and Duo-pushes, and so does every reconnect
after the master drops. To make this failure impossible, `o2_start_master` **refuses to open a
new login unless the route to O2 egresses via a VPN tunnel interface** (it checks
`route get` locally — no connection, no Duo). An explicitly approved grant may
scope `allow_offvpn: true` to that one attempt. There is no durable environment
bypass. Tune the expected interface prefix with `O2_VPN_IFACE_PREFIX` (default
`utun`).

## Install

```bash
# The core (config/connection/sync/slurm/async_transfer/keepalive/workspace) is pure-stdlib
# and runs on Python 3.9. The MCP server needs the mcp SDK (Python >= 3.10):
pip install -e ".[o2]"     # on a 3.10+ env
```

## MCP server config

```jsonc
{
  "mcpServers": {
    "o2": {
      "type": "stdio",
      "command": "/path/to/venv/bin/o2-mcp",
      "env": {
        "O2_SSH_HOST_ALIAS": "o2",
        "O2_SSH_TRANSFER_ALIAS": "o2-transfer",
        "O2_SSH_CONFIG_FILE": "/Users/you/.ssh/config",
        "O2_POLICY_FILE": "/Users/you/.agent_locks/O2_POLICY.json"
      }
    }
  }
}
```

Requires `Host o2` (and optionally `Host o2-transfer`) blocks in `~/.ssh/config` with
`ControlMaster auto` + a `ControlPath` socket.

## Tools

| Tool | Purpose | Hint |
|------|---------|------|
| `o2_local_status` | Local policy, sockets, processes, receipts, and transfer logs; never SSH | read-only/local |
| `o2_status` | Deprecated local-only compatibility alias | read-only/local |
| `o2_policy_disable` | Block every new remote O2 operation without killing existing processes | write/local |
| `o2_policy_enable_reuse` | Explicitly enable existing-master reuse at an observed generation/revision | write/local |
| `o2_authorize_login` | Issue one short-lived login or transfer-host grant | write/local |
| `o2_start_master` | Consume one matching grant and attempt exactly one master start | write |
| `o2_probe` | One explicit fixed remote probe through an existing master; never retried | read-only |
| `o2_exec` | Run an arbitrary command on a login node | write |
| `o2_submit_job` | `sbatch` a script (existing path or staged `script_text`); returns the job id | write |
| `o2_squeue` | `squeue -u <user>` as structured rows | read-only |
| `o2_job_status` | `sacct -j <id>` accounting (state, elapsed, exit code, MaxRSS) | read-only |
| `o2_tail_log` | Tail a remote log file | read-only |
| `o2_cancel_job` | `scancel <id>` | **destructive** |
| `o2_push` / `o2_pull` | rsync up/down (reuses the master; `use_transfer_node` for big moves) | write |
| `o2_push_async` / `o2_pull_async` | Non-blocking rsync: launch detached, return a `transfer_id` immediately | write |
| `o2_transfer_status` | Progress/state of async transfers (`running`/`done`/`failed`/`crashed`); omit id to list all | read-only |
| `o2_transfer_cancel` | SIGTERM a running async transfer's process group | **destructive** |
| `o2_disk_report` | Per-tier usage + hygiene flags (regenerable/redundant/misplaced) | read-only |
| `o2_workspace_gc` | Prune regenerable + redundant disk (detached, dry-run default) | **destructive** |
| `o2_place` | Resolve the canonical output path for a kind (+project) per tier | read-only |

### Non-blocking transfers

`o2_push_async` / `o2_pull_async` launch a detached rsync and return a `transfer_id` right
away, so the agent can keep working and poll `o2_transfer_status` instead of blocking a tool
call for a multi-GB transfer. The transfer keeps running between tool calls and survives an
MCP-server restart (a wrapper records rsync's exit code to disk); re-running the same command
resumes it (`rsync --partial`). Remote paths are escaped so spaces transfer intact while
`~`/`$VAR`/`${VAR}` still expand. State lives under `~/.cache/clock_o2_mcp/transfers`
(`O2_ASYNC_STATE_DIR` to override).

## Safety contract

- `O2_POLICY.json` is the sole policy state. Missing, malformed, symlinked,
  wrong-owner, or permissively readable state is effectively `disabled`; no
  project/ancestor lock files or bypass environment variables are consulted.
- Only `o2_start_master` may authenticate, and only after atomically consuming a
  matching client/host/off-VPN-scoped grant. Remote commands, transfers, and
  lifecycle launches disable every SSH authentication method, so OpenSSH's normal
  missing-socket fallback cannot generate a new Duo request.
- The library retains its historical `require_master` parameters for source compatibility,
  but rejects `require_master=False`; callers cannot opt back into cold SSH/rsync behavior.
- `o2_local_status` and the deprecated `o2_status` alias never invoke SSH.
  `o2_probe` is the only status-like remote operation and runs exactly once.
- The policy JSON contains the login-attempt receipt and five-minute cooldown.
  A zero exit from `ssh -MNf` is successful only when an immediate exact-socket
  control check confirms that the background master survived.
- `disabled` does not automatically stop a master or detached transfer. Local
  inspection and separately approved local transfer cancellation remain available.
- Destructive/transfer-node operations default to dry-run where applicable and verify before
  freeing scratch.

## Development

```bash
pip install -e ".[dev,o2]"
ruff check src tests && black --check src tests && pytest -m "not o2" -q
```

The core stays import-light (stdlib only); `mcp`/`pydantic`/`anyio` are needed only by the
server. Tests inject the subprocess seam, so they run fully offline (no cluster, no network).
