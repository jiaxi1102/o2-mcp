"""Local emergency commands for the workstation-wide O2 safety policy.

This module intentionally exposes only the fail-closed ``disable`` transition.
Enabling reuse or authorizing a login remains an MCP operation because those
changes expand authority and must be bound to an observed policy snapshot and
explicit human approval.  The local command exists so a repeated-Duo incident
can still be stopped when the active client did not load the MCP namespace.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from o2mcp.config import default_policy_file
from o2mcp.policy import O2PolicyError, O2PolicyStore


def _policy_path() -> Path:
    """Resolve the same single policy path used by the MCP configuration.

    The emergency path deliberately reads only ``O2_POLICY_FILE`` rather than
    constructing :class:`O2Config`: malformed unrelated SSH settings must not
    prevent an incident stop from reaching the policy store.
    """

    path = default_policy_file()
    if not path.is_absolute():
        raise ValueError("O2_POLICY_FILE must resolve to one absolute workstation-wide path")
    return path


def _parser() -> argparse.ArgumentParser:
    """Build the deliberately narrow command-line interface."""

    parser = argparse.ArgumentParser(
        prog="o2-mcp-policy-disable",
        description="Disable all new O2 remote operations without contacting O2.",
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="Human-readable incident reason stored in the local policy event log.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Apply the local fail-closed transition and print the resulting identity.

    No socket, SSH client, hostname resolver, or remote command is consulted.
    Policy-store validation and atomic replacement remain authoritative, so the
    CLI cannot bypass unsafe path or ownership checks.
    """

    args = _parser().parse_args(argv)
    try:
        store = O2PolicyStore(_policy_path())
        state = store.disable(reason=args.reason)
    except (O2PolicyError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "ok": True,
                "mode": state["mode"],
                "generation": state["generation"],
                "revision": state["revision"],
                "policy_file": str(store.path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the console entry point.
    raise SystemExit(main())
