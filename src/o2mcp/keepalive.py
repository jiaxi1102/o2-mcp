"""Keep an existing O2 ControlMaster warm so it does not idle out.

A no-op ``true`` is run through the master, which resets ``ControlPersist``'s idle
timer (a plain socket ``-O check`` does not).

IMPORTANT — only useful ON the HMS network / VPN. ``ssh -O check`` reports the
local master *process*, not whether O2 is still reachable through it. If the
underlying connection has been dropped (which happens within minutes from
outside the HMS network), the ping stalls trying to reconnect — and because O2
autopushes Duo on every new connection, that reconnect triggers a Duo push. To
contain that, on a stalled ping this tears the stale master down (``-O exit``) so
the NEXT cycle sees "no master" and skips instead of reconnecting again. Net
effect: at most one push per stale event, not one per cycle. Still: do not run
this off-network. On-network (VPN), connections are stable and Duo-free, so it is
safe and useful there.

Safety invariants (enforced via ``O2Connection``): honors the
``.agent_locks/O2_DISABLED`` lock, and never opens a master itself — a dead
master means "skip", not "reconnect".
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from typing import Any

from o2mcp.config import O2Config
from o2mcp.connection import O2Connection, O2LockedError, O2MasterUnavailableError


def keepalive(config: O2Config | None = None) -> dict[str, Any]:
    """Ping an already-open O2 master; skip/clean up rather than reconnect."""
    conn = O2Connection(config)
    try:
        if conn.is_locked():
            return {"action": "skipped", "reason": "locked"}
        if not conn.master_running():
            return {"action": "skipped", "reason": "no_master"}
        result = conn.run("true", timeout=8.0)
        return {"action": "pinged", "ok": result.ok, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        # The local master socket exists but O2 is no longer reachable through it,
        # so the ping stalled (likely re-authenticating). Tear the stale master
        # down so the next cycle skips instead of reconnecting again.
        with contextlib.suppress(Exception):
            conn.stop_master()
        return {"action": "stale_master_cleared", "reason": "ping_timed_out"}
    except (O2LockedError, O2MasterUnavailableError) as exc:
        return {"action": "skipped", "reason": type(exc).__name__}


def main() -> None:
    """Console entry point: print the keepalive outcome as JSON."""
    print(json.dumps(keepalive()))


if __name__ == "__main__":  # pragma: no cover
    main()
