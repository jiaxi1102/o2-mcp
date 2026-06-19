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

HMS O2 uses Duo **autopush**: every *new* SSH connection fires a Duo push, even key-only /
BatchMode. The tools are built so this costs exactly **one push per session**:

1. Call `o2_start_master` **once** (one push you approve) to open a persistent SSH
   **ControlMaster** (stays up ~8h via `ControlPersist`).
2. Every other tool reuses that master and costs **no** additional push.

Never open the master in a loop or run these tools on a short timer — a periodic reconnect
is what causes "a Duo call every minute". The `.agent_locks/O2_DISABLED` lock file is a hard
stop honored by every operation.

**Be on the HMS VPN.** O2 only *skips* Duo for connections from HMS-trusted source IPs — i.e.
when your SSH egresses through the HMS VPN (GlobalProtect), not your normal internet
interface. If the VPN is down (or split-tunnel isn't routing O2's subnet), even the one
`o2_start_master` login comes from a non-HMS IP and Duo-pushes, and so does every reconnect
after the master drops. To make this failure impossible, `o2_start_master` **refuses to open a
new login unless the route to O2 egresses via a VPN tunnel interface** (it checks `route get`
locally — no connection, no Duo). Override with `allow_offvpn: true` on the tool, or disable the
guard with `O2_REQUIRE_VPN=0`; tune the expected interface prefix with `O2_VPN_IFACE_PREFIX`
(default `utun`).

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
        "O2_SSH_LOCK_FILE": "/path/to/.agent_locks/O2_DISABLED"
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
| `o2_status` | Lock state, ControlMaster state, `hostname; whoami; date` probe | read-only |
| `o2_start_master` | Open the persistent SSH master (needs `allow_new_login`) | write |
| `o2_run` | Run an arbitrary command on a login node | write |
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

- All SSH uses `BatchMode=yes` (public key only) — a dead master or missing key fails fast
  instead of triggering an interactive MFA prompt.
- Remote commands run only through an already-established ControlMaster; opening a new login
  requires explicit opt-in (`allow_new_login`).
- The `O2_DISABLED` lock hard-stops every operation.
- Destructive/transfer-node operations default to dry-run where applicable and verify before
  freeing scratch.

## Development

```bash
pip install -e ".[dev,o2]"
ruff check src tests && black --check src tests && pytest -m "not o2" -q
```

The core stays import-light (stdlib only); `mcp`/`pydantic`/`anyio` are needed only by the
server. Tests inject the subprocess seam, so they run fully offline (no cluster, no network).
