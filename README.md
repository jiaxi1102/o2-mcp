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

Our August 2026 incident testing showed that HMS O2 can issue Duo challenges not
only for a new SSH transport, but also when a new **session channel** is opened
inside an existing OpenSSH ControlMaster. A normal `ssh o2 command` invocation
therefore is not Duo-safe merely because it reaches the expected mux socket.

Version 0.4 replaces that command pattern with one workstation-wide broker per
configured host role. Login-node work normally needs only the login broker; a
separate transfer broker exists for commands that must execute on the transfer
host:

1. `o2_start_broker` consumes a short-lived, client-bound, one-attempt
   role-matched grant and starts exactly one SSH process for that broker.
2. That SSH process opens exactly one remote session running a small embedded
   Python helper.
3. Every later `o2_exec`, Slurm, workspace, keepalive, and run-organization
   command is length-prefixed JSON sent through the same session's stdin/stdout.
4. Independently launched MCP tasks share the applicable broker through a
   mode-0600 Unix socket. Commands are serialized within each role in the MVP.
5. The daemon never reconnects. If the channel dies, its socket disappears and
   later commands fail locally; another login requires another explicit grant.

The workstation-wide `~/.agent_locks/O2_POLICY.json` still defaults fail-closed.
Both the MCP client and the broker daemon check it before every logical command,
so a concurrent `disabled` transition wins even against a hand-crafted local
socket request. Disabling does not terminate a command already in progress.

The broker uses an `O2B1` magic marker, four-byte network-order lengths, and
UTF-8 JSON rather than newline framing. Remote stdout/stderr is drained while
retaining at most 1 MiB per stream, so noisy commands cannot grow broker memory
without bound and newlines or JSON-looking output cannot corrupt the next
command. Frames are limited to 16 MiB. One command timeout returns code 124
without reconnecting or replacing the persistent channel. Logical commands use
a non-login Bash that inherits the one session environment, avoiding repeated
profile banners and startup latency.

ControlMaster hardening remains in the library for the transfer compatibility
layer and offline regression tests. It is **not** the login command boundary:
both the MCP wrapper and public `O2Connection.start_master()` API reject login
master starts and raw SSH, directing callers to the broker instead. See
[the broker design](docs/PERSISTENT_COMMAND_BROKER.md).

OpenSSH evaluates `Match exec` shell predicates even for its nominally local `ssh -G` config
dump. The MCP therefore reads the configured SSH file and recursively flattens `Include`
directives itself, rejects any `Match` block before launching OpenSSH, and runs `ssh -G` only
against that private inspected snapshot. If your normal SSH config contains `Match`, put the
O2 `Host` blocks in a separate Match-free file and set `O2_SSH_CONFIG_FILE` to it.

The optional `o2-transfer` alias remains a different host with its own grant,
command broker, and rsync-compatibility ControlMaster. Run transitions and
explicit transfer-host probes use the framed transfer broker. Existing detached
rsync transfers remain on their pinned ControlMaster path and are preserved;
that compatibility path must not be described as Duo-free merely because its
ControlMaster exists.

Never open the broker/master in a loop or run authentication tools on a timer. The policy
file has two durable modes: `disabled` blocks new remote operations, while
`reuse_only` permits only existing exact sockets. There is deliberately no durable
`normal` mode. The one-shot grant is atomically removed and converted to an active
attempt receipt before SSH, so a failure, timeout, or crash cannot leave reusable
authorization for a queued task. Policy generation/revision pairs and a stable
internal mutex serialize every transition across MCP processes. Callers pass
both values from one `o2_local_status` snapshot; repair creates a new generation,
so a repeated numeric revision cannot revive stale approval.

If repeated Duo prompts begin while the native MCP tools are unavailable, use
the package's local-only emergency command. It performs the same atomic policy
transition as `o2_policy_disable` and never invokes SSH or any remote probe:

```bash
o2-mcp-policy-disable --reason "repeated Duo prompts"
```

This command can only reduce authority. Re-enabling reuse and issuing a login
grant remain MCP-only operations with explicit global or target-scoped approval.

**Be on the HMS VPN.** O2 normally skips Duo for connections from HMS-trusted source IPs — i.e.
when your SSH egresses through the HMS VPN (GlobalProtect), not your normal internet
interface. If the VPN is down (or split-tunnel isn't routing O2's subnet), the one
`o2_start_broker` login or transfer-host startup may Duo-push. To make accidental off-VPN launch impossible,
`o2_start_broker` **refuses to open a new login unless the route to O2 egresses via a VPN
tunnel interface** (it checks
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
        "O2_POLICY_FILE": "/Users/you/.agent_locks/O2_POLICY.json",
        "O2_BROKER_DIR": "/Users/you/.agent_locks/o2-broker",
        "O2_TRANSFER_BROKER_DIR": "/Users/you/.agent_locks/o2-transfer-broker"
      }
    }
  }
}
```

Requires a `Host o2` block (and optionally `Host o2-transfer`) in the inspected
SSH config. The broker snapshots that Match-free config before consuming the
grant and explicitly disables mux reuse, proxies, and local command hooks for
its one transport. `O2_BROKER_DIR` must be absolute, private, and short enough
for a macOS Unix socket; the default satisfies those constraints.
`O2_TRANSFER_BROKER_DIR` has the same requirements and must be distinct from
`O2_BROKER_DIR`.

## Tools

| Tool | Purpose | Hint |
|------|---------|------|
| `o2_local_status` | Local policy, sockets, processes, receipts, and transfer logs; never SSH | read-only/local |
| `o2_status` | Deprecated local-only compatibility alias | read-only/local |
| `o2_policy_disable` | Block every new remote O2 operation without killing existing processes | write/local |
| `o2_policy_enable_reuse` | Explicitly enable existing broker/transport reuse at an observed generation/revision | write/local |
| `o2_authorize_login` | Issue one short-lived login or transfer-host grant | write/local |
| `o2_start_broker` | Consume one role-matched grant and open that host role's persistent command channel (`transfer=true` for the transfer host) | write |
| `o2_stop_broker` | Locally close one role-specific broker and its SSH process; no policy change | **destructive/local** |
| `o2_start_master` | Transfer compatibility only; login-master starts are rejected | write |
| `o2_probe` | One explicit fixed command through the existing broker; never retried | read-only |
| `o2_exec` | Run an arbitrary command on a login node | write |
| `o2_submit_job` | `sbatch` a script (existing path or staged `script_text`); returns the job id | write |
| `o2_squeue` | `squeue -u <user>` as structured rows | read-only |
| `o2_job_status` | `sacct -j <id>` accounting (state, elapsed, exit code, MaxRSS) | read-only |
| `o2_tail_log` | Tail a remote log file | read-only |
| `o2_cancel_job` | `scancel <id>` | **destructive** |
| `o2_push` / `o2_pull` | rsync up/down through the existing transfer master by default; `use_transfer_node=false` is legacy reuse only | write |
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
  `O2_POLICY_FILE` must be absolute. The first explicit policy mutation safely
  tightens an owned physical legacy `~/.agent_locks` directory from `0755` to
  `0700`; aliased or foreign-owned directories remain invalid.
- Only `o2_start_broker` (role-specific command transport) or the transfer rsync
  compatibility start can authenticate, and only after atomically consuming a
  matching client/host/off-VPN-scoped grant. The parent starts only a local
  daemon under the policy mutex; that daemon revalidates the exact active
  attempt and holds the mutex around its sole SSH spawn. Its canonical
  `launch.json` is atomically claimed and erased before SSH, so it cannot be
  replayed as a durable authentication recipe.
- Login and transfer-host commands never invoke SSH directly and never open a
  new ControlMaster session channel. Raw SSH through `run_raw` is rejected in
  production. Rsync remains the explicitly isolated compatibility boundary.
- Each broker socket directory is physical, caller-owned, and mode 0700; its
  socket and files are mode 0600. One lifetime `flock` per role prevents
  overlapping daemons. An unbindably long socket path fails before grant
  consumption. Clients require a trusted `ready` receipt for the exact protocol
  version, role alias, and expanded `HostName`/`User`/`Port`; incomplete local
  frames expire on a finite absolute deadline. Destination changes block command
  reuse but not local broker stop, so the stale daemon can be retired without
  contacting O2.
- The library retains its historical `require_master` parameters for source compatibility,
  but rejects `require_master=False`; callers cannot opt back into cold SSH/rsync behavior.
- Blocking and detached rsync default to the transfer alias, whose master can be
  started with a transfer-scoped one-shot grant. Login-alias rsync remains only
  for reusing an already-existing legacy login master; the MCP cannot create one.
- `o2_local_status` and the deprecated `o2_status` alias inspect broker receipts,
  Unix sockets, processes, policy, and transfer logs locally; neither invokes SSH.
  `o2_probe` is the only status-like remote operation and runs exactly once
  through the selected role-specific broker.
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
