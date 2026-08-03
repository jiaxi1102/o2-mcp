"""Send a no-op through an existing persistent O2 command broker.

A no-op ``true`` is dynamically framed over the broker's one SSH session. It
does not create another SSH process or session channel.

The broker never reconnects, so a dead channel means "skip", not a new Duo
attempt. OpenSSH's own ServerAlive messages monitor the transport without
opening command channels; this command exists only for deployments that want a
remote application-level liveness marker.

Safety invariants (enforced twice by ``O2Connection`` and the daemon): the global
policy must permit reuse, and this command never starts a broker itself.
"""

from __future__ import annotations

import json
from typing import Any

from o2mcp.broker import O2BrokerError
from o2mcp.config import O2Config
from o2mcp.connection import O2Connection
from o2mcp.policy import O2PolicyDeniedError, O2PolicyInvalidError


def keepalive(config: O2Config | None = None) -> dict[str, Any]:
    """Ping an already-open broker and skip rather than reconnect."""

    conn = O2Connection(config)
    try:
        conn.policy.require_reuse_allowed()
        if conn.broker_local_status().get("responsive") is not True:
            return {"action": "skipped", "reason": "no_broker"}
        result = conn.run("true", timeout=8.0)
        return {"action": "pinged", "ok": result.ok, "returncode": result.returncode}
    except (O2PolicyDeniedError, O2PolicyInvalidError, O2BrokerError) as exc:
        return {"action": "skipped", "reason": type(exc).__name__}


def main() -> None:
    """Console entry point: print the keepalive outcome as JSON."""
    print(json.dumps(keepalive()))


if __name__ == "__main__":  # pragma: no cover
    main()
