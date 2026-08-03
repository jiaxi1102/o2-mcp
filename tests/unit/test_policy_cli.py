"""Offline tests for the fail-closed O2 policy command-line fallback."""

from __future__ import annotations

import json

from o2mcp.policy import O2PolicyStore
from o2mcp.policy_cli import main


def test_policy_disable_cli_initializes_default_state(monkeypatch, tmp_path, capsys):
    """The fallback reaches the standard global path without remote activity."""

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("O2_POLICY_FILE", raising=False)

    assert main(["--reason", "repeated Duo prompts"]) == 0

    payload = json.loads(capsys.readouterr().out)
    policy_path = tmp_path / ".agent_locks" / "O2_POLICY.json"
    assert payload["ok"] is True
    assert payload["mode"] == "disabled"
    assert payload["policy_file"] == str(policy_path)
    assert O2PolicyStore(policy_path).snapshot().effective_mode == "disabled"


def test_policy_disable_cli_revokes_existing_grant(monkeypatch, tmp_path, capsys):
    """An incident stop wins even when another task holds login authority."""

    policy_path = tmp_path / "policy" / "O2_POLICY.json"
    monkeypatch.setenv("O2_POLICY_FILE", str(policy_path))
    store = O2PolicyStore(policy_path, client_id="other-task", clock=lambda: 1000.0)
    disabled = store.disable(reason="initialize")
    reuse = store.enable_reuse(
        expected_revision=disabled["revision"],
        expected_generation=disabled["generation"],
        approval_reference="explicit global test approval",
    )
    store.authorize_login(
        expected_revision=reuse["revision"],
        expected_generation=reuse["generation"],
        target="login",
        allow_offvpn=False,
        approval_reference="explicit target test approval",
    )

    assert main(["--reason", "stop all new activity"]) == 0

    capsys.readouterr()
    snapshot = store.snapshot()
    assert snapshot.state is not None
    assert snapshot.state["mode"] == "disabled"
    assert snapshot.state["login_grant"] is None


def test_policy_disable_cli_rejects_relative_policy_path(monkeypatch, capsys):
    """A process-local relative path cannot masquerade as the global state."""

    monkeypatch.setenv("O2_POLICY_FILE", "relative/O2_POLICY.json")

    assert main(["--reason", "incident"]) == 2

    payload = json.loads(capsys.readouterr().err)
    assert payload["ok"] is False
    assert "absolute workstation-wide path" in payload["error"]
