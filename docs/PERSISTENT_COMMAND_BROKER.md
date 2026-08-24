# Persistent O2 command broker MVP

## Problem

OpenSSH multiplexing reuses a TCP connection and authentication context, but a
subsequent `ssh host command` still opens a new SSH session channel. Observed O2
behavior in August 2026 showed a Duo call during that operation even though the
child process opened only the expected local mux socket and no new TCP socket.
ControlMaster socket pinning remains necessary to prevent authentication
fallback, but it is not sufficient to suppress per-session challenges.

## Authentication and command boundaries

```mermaid
flowchart LR
    A["MCP task A"] --> L["login broker socket"]
    B["MCP task B"] --> L
    C["MCP task C"] --> T["transfer broker socket"]
    L --> LD["login broker daemon"]
    T --> TD["transfer broker daemon"]
    LD -->|"framed JSON"| LS["one login-host SSH session"]
    TD -->|"framed JSON"| TS["one transfer-host SSH session"]
    LS --> LR["embedded remote helper"]
    TS --> TR["embedded remote helper"]
    LR --> LX["serialized bash -c commands"]
    TR --> TX["serialized bash -c commands"]
```

Starting either role-specific broker is an authentication boundary. It consumes
the matching login- or transfer-scoped one-shot grant, including the off-VPN
scope, and launches exactly one direct SSH transport. It deliberately sets
`-S none`, `ControlMaster=no`, and `ControlPath=none`: binding broker lifetime to
an older mux master would reintroduce the disappearance and channel-lifetime
problem the broker is meant to solve.

The grant-consuming parent holds the global policy mutex while it starts only
the local detached daemon, then releases it so the daemon can perform its own
authorization gate. Immediately around SSH creation, the daemon reacquires the
mutex and verifies that the grant id, role, originating client, and parent PID
still match the active consumed attempt. A policy disable therefore either wins
the handoff and prevents SSH or follows an already-started operation. Launch
data is sent only through a bounded anonymous pipe inherited by that child; no
same-UID-replaceable path or durable recipe exists. The daemon records the login
attempt as successful only after the remote helper sends the expected protocol
hello and the owner-only local socket plus trusted ready receipt are published.

There is no automatic retry or reconnect. The remote protocol hello has the same
finite startup deadline as the launcher, so a silent child cannot retain the
lifetime lock indefinitely. A startup timeout leaves one attempt receipt for
inspection. A later task must not infer that it may start another channel from
the absence of a ready socket.

## Protocol

Every frame is:

```text
4-byte ASCII magic: O2B1
4-byte unsigned big-endian payload length
UTF-8 JSON object of exactly that length
```

Logical command text is capped at 64 KiB because it becomes a remote
`bash -c` argument; this keeps a valid frame below the host's `execve` argument
limit. Larger scripts must be transferred as files and invoked with a short
command. Process-spawn failures return an ordinary command result and do not
terminate the persistent helper. The client validates the complete JSON-escaped
request before connecting, and stdin is capped at 1 MiB, so expansion cannot
fail after dispatch and bulk payloads remain on the transfer path.

The daemon may scan through at most 64 KiB of unframed output before the first
remote hello, accommodating a login-shell banner. Every later frame is strict.

The remote helper first emits:

```json
{"type":"hello","protocol":2}
```

A logical command request contains protocol version 2, a random request id,
command, timeout, and optional stdin text. A JSON `null` timeout preserves the
public API's explicit no-deadline behavior; finite deadlines must be positive
and no longer than seven days so local socket timeouts stay platform-safe.
Explicit versioning prevents either side of an in-place client/daemon upgrade
from misinterpreting acknowledgement semantics and executing an apparently
failed request. The helper executes
`/bin/bash --noprofile --norc -c <command>` in the environment inherited from
the one SSH session, so per-command login profiles cannot add banners or consume
timeouts. It concurrently drains both output streams while retaining bounded
prefixes, and returns the same id, return code, duration, timeout flag, and
truncation flags. Commands are serialized within each broker so responses cannot
be reordered.

The frame limit is 16 MiB. Stdout and stderr retain at most 1 MiB each; later
bytes are drained and discarded rather than accumulated in memory. A remote
timeout returns code 124 and the helper remains available for the next frame.
For finite commands, the daemon separately bounds its result-frame wait with an
inactivity timeout and a frame-size-scaled absolute budget. Slow response
progress refreshes the inactivity deadline; if SSH remains alive but the helper
stops responding, the daemon terminates and unpublishes the sole transport. The
dispatched outcome remains unknown and is never retried automatically.

The local client first waits for a `dispatched` acknowledgement, which the daemon
writes only when the request reaches the serialized execution boundary and the
policy still permits it. Queue delay does not consume the command's remote
timeout. A caller that disconnects before acknowledgement is cancelled locally;
if the result stream is lost after acknowledgement, the MCP reports
`broker_outcome_unknown` with `retry_safe=false` instead of inviting an unsafe
automatic retry.

## Local authority and failure behavior

- `~/.agent_locks/o2-broker` and
  `~/.agent_locks/o2-transfer-broker` default to distinct physical owner-only
  mode-0700 directories. `O2_BROKER_DIR` and `O2_TRANSFER_BROKER_DIR` may
  override them only with distinct absolute filesystem authorities; paths are
  canonicalized and existing identities compared so `..` or symlinked
  ancestors cannot make both roles share one socket/lock directory.
- `command.sock`, state, lock, and log are owner-only.
  Symlinked or permissive authority files fail closed. Authentication-capable
  launch data never enters the filesystem; it is consumed once from a bounded
  inherited descriptor before any SSH spawn. The inspected `ssh -G` expansion
  is sealed directly into that launch argv, while SSH reads `-F /dev/null`, so
  there is no replaceable config pathname between authorization and spawn.
- A client connects only when a physical mode-0600 state receipt positively
  reports `ready` under the exact protocol version; missing, malformed, or
  stale-version receipts are not treated as compatible defaults.
- Command reuse also requires both the SSH alias and its expanded
  `HostName`/`User`/`Port` identity to match that role's current inspected
  configuration. A local stop remains available after a destination or protocol
  change so the stale daemon can be retired safely without sending a remote
  command.
- One lifetime `flock` per role prevents two daemons from owning an endpoint.
- Both `O2Connection.run` and the daemon re-read `O2_POLICY.json` before a
  command. A global disable cannot be bypassed with a direct socket client.
- The remote frame write has a five-second inactivity deadline plus a
  frame-size-scaled total deadline while the policy mutex is held. Continued
  progress is allowed at slow-link rates; a stalled SSH stdin terminates that
  broken transport and releases global policy authority instead of blocking
  incident disable.
- Disabling policy does not kill an already-running command or broker. It blocks
  the next frame. `o2_stop_broker` is an explicit local process-control action.
- If SSH exits or framing fails, the daemon removes its socket, writes a failed
  receipt, terminates the child if necessary, and exits. It never reconnects.
- The first local frame has an absolute five-second deadline, so a same-user
  process cannot wedge the serialized broker by sending an incomplete or
  byte-trickled request. Local response writes are bounded by the same deadline.
- `o2_local_status` may read the receipt and ping the Unix socket, but it never
  sends a frame to O2.

### Bounding one caller's cost

- A single `o2_exec` may request at most 300 seconds. That number is both how
  long one command can block every other caller on the shared channel and the
  watchdog budget before a silent transport is torn down. Work needing longer
  belongs in a submitted job polled from the caller, which releases the channel
  between checks.
- The wait for a dispatch acknowledgement is bounded too, scaled to the caller's
  own deadline (`max(60s, timeout_seconds)`, or 300s when no deadline was
  given). That budget starts at the connection, not after it: the daemon accepts
  nothing while serving a command, so callers fill the listen backlog and
  eventually `connect` itself blocks. A short ceiling there would report a
  healthy but saturated broker as *absent*, which is the one diagnosis that
  points a caller at starting another one. A receipt that fails validation still
  raises before any connection, so a genuinely missing broker keeps its own
  error. Without this bound a starved caller simply never returns, which is
  indistinguishable from a hang; `o2_exec` reports it as `broker_busy` and the
  message names the command occupying the channel from the in-flight receipt.
- A budget expiry is **not** proof the request was never dispatched. The daemon
  writes the acknowledgement and forwards the command as one step, so a budget
  expiring in that instant leaves a command running and a caller that never read
  the acknowledgement. `O2BrokerBusyError` therefore subclasses
  `O2BrokerCommandOutcomeUnknownError` and reports `retry_safe: false`, so a
  mutating command cannot be duplicated by a caller trusting an optimistic
  classification. The receipt settles the one direction it can: when the request
  recorded in flight is the caller's own, the command is known to be running and
  the error says so outright.
- Abandoning the socket is also how the daemon learns to cancel a queued
  request. It checks for a disconnected caller before forwarding, so a timed-out
  request is dropped rather than run unobserved.

### Stopping a broker that cannot answer

- `o2_stop_broker` asks over the socket. That retires an idle broker cleanly and
  **cannot** retire a busy one: the request queues behind the command holding the
  channel, and the daemon cancels a queued stop whose caller has already timed
  out. That cancellation is deliberate — a stop reported as failed should not
  shut the shared broker down minutes later — but it also means the documented
  remedy did nothing in exactly the situation that prompts reaching for it.
- `force: true` marks the request as one the daemon honours regardless of
  whether its caller is still waiting. It is the only way to retire a busy
  broker through the socket, and it stays inside the socket's own authority: no
  pid is read and no signal is sent. An older daemon ignores the flag and
  behaves as before.
- **A forced stop is queued, not prioritized, and that bounds how fast it can
  act.** The daemon serves one connection at a time from a FIFO accept queue, so
  a forced stop takes effect after the in-flight command *and any connections
  already queued ahead of it* — a delay bounded by queue depth times the
  per-command ceiling, not by a single command. Through `o2_exec` that is queue
  depth × 300s; a library caller at the 3600s ceiling makes it far worse. It
  abandons nothing and jumps nothing.
- A stop that must act *now* still means killing the daemon by hand. The proper
  fix is a control path that does not sit behind the command queue — a second
  listener the daemon services independently — which is deliberately **not** in
  this change: it is a new authority boundary and wants its own review.
- Signalling the daemon by pid was tried and abandoned. A pid can only be taken
  from the receipt or from the lock file, and neither survives the gap to the
  `kill`: a receipt outlives its daemon, and a lock read proves only who held
  the lock during that read. Without a `pidfd` — unavailable on macOS — there is
  no way to bind a pid to a process across that window, so the socket, whose
  authority is the connection itself, is the correct channel.
- Abandoning a running command outright remains a manual `kill -9`, which leaves
  an orphaned receipt that `busy` correctly ignores because the lock is free.
- Force is recommended only when the refreshed receipt confirms a busy daemon. A
  missing broker or an unreadable receipt raises the same error type, and force
  resolves neither, so those keep their own diagnosis.

### Bounding one caller's cost, part two

- The 300s cap on `o2_exec` guards one input model. `BrokerClient.execute`
  enforces a 3600s ceiling for every caller of the client, **and the daemon
  enforces it again** for every caller of the socket. Only the daemon can hold a
  workstation-wide bound: a client-side guard binds only the processes carrying
  it, and every already-running MCP process keeps its previous client until
  restarted, while the protocol still permits seven days. The daemon
  **shortens** an over-ceiling deadline rather than refusing it, and reports the
  reduction in `stderr` alongside the truncation notices. Refusing looks
  cleaner but has no wire form an older client reads correctly: it recognizes
  only `policy_denied` and classifies anything else as a command that may
  already have run — so refusing would tell exactly the stale callers this
  guard exists for not to retry something that never left the workstation. A
  clamp serves the request, bounds the channel, and says so.
- **A deadline-free request is bounded too.** `timeout_seconds: null` is the one
  shape that can wedge the daemon outright: its result read has no watchdog, so
  a helper that never replies holds the shared channel with nothing to time out.
  The protocol still carries JSON null, but the daemon bounds it to the ceiling
  before dispatch and reports that in `stderr` like any other reduction. The
  unbounded read remains only for direct in-process callers who choose it.
- Any refusal frame the daemon does send is reported by the client as a
  pre-dispatch rejection rather than an uncertain outcome, since nothing is
  forwarded before one is written.
- A forced stop must stay reachable under the contention that made it necessary,
  so its connection waits (60s) rather than failing fast, and the listen backlog
  is sized well above the number of MCP processes on a workstation. A connection
  that cannot be queued fails outright, and a stop locked out by ordinary command
  contention is no remedy at all.
- An explicit `timeout=None` is still accepted on the wire, but the daemon does
  **not** honour it as unbounded: it is clamped to the ceiling before dispatch
  like any over-ceiling deadline, and the reduction is reported in `stderr` as
  "no deadline". A library caller must not design a long-running command around
  an unbounded contract.
- The earlier argument for leaving it unbounded — that a dead helper, a dead SSH
  process, and a black-holed network all close the transport's stdout, so the
  read ends in every failure mode — is true as far as it goes, and it is why
  every *failure* mode self-heals. What it misses is the case it dismissed: a
  command that genuinely never finishes is not just "what the caller asked for"
  when the thing being asked for is exclusive use of a channel twenty sessions
  share. That is precisely the shape an agent polling inside a command creates,
  and it is the one request the daemon has no watchdog for.

### Command observability

The daemon serializes local clients over one channel, so it cannot answer a
ping while a command is running. Any command slower than the ping deadline
therefore makes a healthy broker report `responsive: false`, which on its own
is indistinguishable from a daemon that has stopped serving. The receipt
carries the missing half of that picture:

- `in_flight` names the command currently holding the channel (`request_id`,
  bounded command fingerprint, requested `timeout_seconds`, `dispatched_at`)
  and is `null` whenever the channel is idle. It is written before the daemon
  waits for a result and is retained by the terminal receipt, so a daemon that
  dies mid-command still records what it was running.
- `last_command` records the completed command plus `duration_seconds` — the
  broker-observed channel occupancy, which is what starves other callers —
  alongside the helper's own `remote_duration_seconds`, `returncode`,
  `timed_out`, and truncation flags. Remote-reported numbers are filtered to
  finite, plausibly sized values before they reach a file later readers must
  still parse.
- `local_status` consults that record only when the ping went unanswered, and
  then reports `busy` and `busy_for_seconds` so the caller learns which command
  is responsible instead of only that no answer arrived. A successful pong
  reports `busy: false` regardless of the record: a serialized daemon can only
  answer between commands, so a pong is positive proof that nothing occupies
  the channel, and it retires an `in_flight` entry that a suppressed
  best-effort write would otherwise leave standing. A terminal receipt keeps
  its `in_flight` record for forensics but is never reported as busy.
- `duration_seconds` is measured with `time.monotonic()`, like every other
  deadline in the daemon, so an NTP or manual wall-clock step during a long
  command cannot inflate the metric or collapse it to zero. The epoch
  timestamps beside it are for operator reading, not arithmetic.
- The in-flight receipt is published after the remote frame write, so another
  process reads `busy: false` for that write's duration. Publishing earlier
  would place an unbounded fsync inside the policy mutex, whose hold time is
  deliberately bounded so an incident disable stays responsive. The window is
  bounded by the write deadline, a write that fails inside it is still named by
  the terminal receipt, and occupancy already covers it because timing starts at
  the acknowledgement.
- `busy` also requires that a daemon still holds the lifetime lock. A daemon
  killed before its terminal write -- SIGKILL, a crash, a reboot -- leaves a
  `ready` receipt naming a command nothing will retire, and without that check
  it would report busy forever. `busy_for_seconds` is derived from a persisted
  monotonic reading rather than the epoch, which is sound precisely because a
  held lock implies a live daemon and therefore the same boot. That reading
  names its clock explicitly (`CLOCK_MONOTONIC`) instead of relying on
  `time.monotonic()`, whose reference point is implementation-defined: on macOS
  it maps to `mach_absolute_time()`, which excludes time the host spent asleep
  and so differs from `CLOCK_MONOTONIC` on the same machine by however long it
  has slept. Publishing one and subtracting the other would be silently wrong
  by that amount. Where the named clock is unavailable the field is omitted
  rather than guessed.
- The record is best effort, not a ledger. Consecutive suppressed writes can
  leave it naming an earlier command than the one actually running; it is a
  diagnostic aid, and the pong remains the authority on whether the channel is
  free.
- Busy is not un-ready. `status` stays `ready` while a command runs, because a
  busy broker is still a reusable broker: later clients queue behind the
  in-flight command rather than being refused for lacking a ready receipt.
- The command fingerprint is a SHA-256 digest, a UTF-8 byte length, and a
  preview bounded to the first 200 characters. The full command is deliberately
  not copied into the receipt, so a repeat offender stays identifiable without
  retaining arguments that may embed sensitive paths.
- The dispatch-time receipt write is best effort. It is diagnostic, so a
  transient failure must not abort a command the channel is ready to run.

## MVP boundaries

`o2_exec`, Slurm tools, workspace tools, and keepalive reach the login broker
through `O2Connection.run`. Run-organization reads normally use that broker;
promotion and archive launches that must run on the transfer host use its
separate persistent broker. Raw SSH commands are rejected for both roles.

Existing detached rsync transfers are not terminated or migrated. Transfer
data still uses the separately governed ControlMaster compatibility path and
should not be described as Duo-free. The transfer command broker eliminates raw
SSH only for commands; replacing rsync's distinct session boundary remains
future work. New rsync operations default to the dedicated transfer alias because
that role retains an explicit grant-gated master startup. Login-alias rsync can
reuse a legacy master but cannot create one through either the MCP wrapper or
the public connection API. `o2_stop_master` and
`O2Connection.stop_master()` close either the supported transfer master or a
pre-upgrade login master locally through its exact socket with authentication
disabled. This shutdown compatibility does not restore login-master startup.

## Offline validation

`tests/unit/test_broker.py` runs the exact embedded remote helper as a local
Python child. It proves that multiple clients and dynamically different commands
share one process, command stdin and multiline output survive framing, noisy
output stays bounded, queue time is separated from command time, an abandoned
queued request is not dispatched, protocol-hello timeout releases the lock,
policy disable blocks forwarding, and detached role-specific launchers consume
one matching grant then reuse one daemon. Run-organization tests prove transfer
commands select the transfer broker. No test in this PR contacts O2 or starts
SSH.
