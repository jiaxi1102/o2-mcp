"""Offline tests for the workstation-wide O2 policy state machine.

These tests exercise filesystem validation, atomic revision transitions, scoped
one-shot grants, and cross-thread serialization.  They never construct an SSH
command or touch the network.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from o2mcp.policy import (
    O2LoginGrantError,
    O2PolicyConflictError,
    O2PolicyDeniedError,
    O2PolicyInvalidError,
    O2PolicyStore,
)


def _reuse_store(tmp_path, *, client_id="client-a", clock=lambda: 1000.0) -> O2PolicyStore:
    """Create valid reuse-only state through the same public transitions as production."""

    store = O2PolicyStore(tmp_path / "O2_POLICY.json", client_id=client_id, clock=clock)
    disabled = store.disable(reason="initialize test policy")
    store.enable_reuse(expected_revision=disabled["revision"], approval_reference="explicit test approval")
    return store


def test_missing_policy_is_effectively_disabled(tmp_path):
    """Absence is diagnostic state, never an implicit reuse authorization."""

    store = O2PolicyStore(tmp_path / "O2_POLICY.json", client_id="client-a")
    snapshot = store.snapshot()

    assert snapshot.valid is False
    assert snapshot.effective_mode == "disabled"
    assert snapshot.revision == 0
    with pytest.raises(O2PolicyInvalidError):
        store.require_reuse_allowed()


def test_disable_initializes_secure_atomic_state(tmp_path):
    store = O2PolicyStore(tmp_path / "policy" / "O2_POLICY.json", client_id="client-a", clock=lambda: 12.5)

    state = store.disable(reason="Duo incident")

    assert state["mode"] == "disabled" and state["revision"] == 1
    assert state["events"][-1]["event"] == "policy_disabled"
    assert os.stat(store.path).st_mode & 0o777 == 0o600
    assert os.stat(store.path.parent).st_mode & 0o777 == 0o700


def test_reuse_enable_uses_compare_and_swap_revision(tmp_path):
    store = O2PolicyStore(tmp_path / "O2_POLICY.json", client_id="client-a")
    state = store.disable(reason="initialize")

    with pytest.raises(O2PolicyConflictError):
        store.enable_reuse(expected_revision=state["revision"] - 1, approval_reference="stale approval")

    enabled = store.enable_reuse(
        expected_revision=state["revision"],
        approval_reference="explicit global re-enable",
    )
    assert enabled["mode"] == "reuse_only"
    assert enabled["revision"] == state["revision"] + 1


@pytest.mark.parametrize("unsafe_kind", ["symlink", "permissions", "malformed"])
def test_unsafe_or_malformed_policy_fails_closed(tmp_path, unsafe_kind):
    policy = tmp_path / "O2_POLICY.json"
    if unsafe_kind == "symlink":
        target = tmp_path / "target.json"
        target.write_text("{}")
        target.chmod(0o600)
        policy.symlink_to(target)
    elif unsafe_kind == "permissions":
        policy.write_text("{}")
        policy.chmod(0o644)
    else:
        policy.write_text("not-json")
        policy.chmod(0o600)

    store = O2PolicyStore(policy, client_id="client-a")

    assert store.snapshot().effective_mode == "disabled"
    with pytest.raises(O2PolicyInvalidError):
        store.require_reuse_allowed()


@pytest.mark.parametrize("unsafe_kind", ["permissions", "symlink"])
def test_unsafe_policy_directory_fails_closed_for_reads_and_mutations(tmp_path, unsafe_kind):
    """A secure file cannot compensate for an replaceable parent directory."""

    policy_directory = tmp_path / "policy"
    if unsafe_kind == "permissions":
        policy_directory.mkdir(mode=0o755)
        policy_directory.chmod(0o755)
    else:
        real_directory = tmp_path / "real-policy"
        real_directory.mkdir(mode=0o700)
        policy_directory.symlink_to(real_directory, target_is_directory=True)
    store = O2PolicyStore(policy_directory / "O2_POLICY.json", client_id="client-a")

    assert store.snapshot().effective_mode == "disabled"
    with pytest.raises(O2PolicyInvalidError):
        store.disable(reason="must not repair an unsafe parent")


def test_login_grant_is_client_target_and_time_scoped(tmp_path):
    now = [1000.0]
    owner = _reuse_store(tmp_path, client_id="owner", clock=lambda: now[0])
    revision = owner.snapshot().revision
    grant = owner.authorize_login(
        expected_revision=revision,
        target="login",
        allow_offvpn=True,
        approval_reference="start login master off VPN",
        ttl_seconds=10,
    )

    other = O2PolicyStore(owner.path, client_id="other", clock=lambda: now[0])
    with pytest.raises(O2LoginGrantError, match="different MCP task"):
        other.preview_login_grant(grant.id, "login")
    with pytest.raises(O2LoginGrantError, match="scoped to 'login'"):
        owner.preview_login_grant(grant.id, "transfer")

    now[0] = 1011.0
    with pytest.raises(O2LoginGrantError, match="expired"):
        owner.preview_login_grant(grant.id, "login")


def test_grant_consumption_is_atomic_across_contenders(tmp_path):
    """Two callers sharing one grant can persist exactly one active attempt."""

    store = _reuse_store(tmp_path, client_id="same-client")
    grant = store.authorize_login(
        expected_revision=store.snapshot().revision,
        target="login",
        allow_offvpn=False,
        approval_reference="one approved attempt",
    )
    contenders = [O2PolicyStore(store.path, client_id="same-client", clock=lambda: 1000.0) for _ in range(2)]

    def consume(contender):
        try:
            return contender.consume_login_grant(grant.id, "login")
        except O2LoginGrantError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(consume, contenders))

    assert sum(not isinstance(outcome, Exception) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, O2LoginGrantError) for outcome in outcomes) == 1
    state = store.snapshot().state
    assert state["login_grant"] is None
    assert state["login_attempt"]["grant_id"] == grant.id
    assert state["login_attempt"]["outcome"] == "active"


def test_active_attempt_blocks_new_authorization_until_cooldown(tmp_path):
    now = [1000.0]
    store = _reuse_store(tmp_path, client_id="client-a", clock=lambda: now[0])
    grant = store.authorize_login(
        expected_revision=store.snapshot().revision,
        target="login",
        allow_offvpn=False,
        approval_reference="first attempt",
    )
    store.consume_login_grant(grant.id, "login")

    with pytest.raises(O2LoginGrantError, match="cooling down"):
        store.authorize_login(
            expected_revision=store.snapshot().revision,
            target="login",
            allow_offvpn=False,
            approval_reference="unsafe immediate retry",
        )

    now[0] += 301.0
    replacement = store.authorize_login(
        expected_revision=store.snapshot().revision,
        target="login",
        allow_offvpn=False,
        approval_reference="fresh explicit approval after cooldown",
    )
    assert replacement.id != grant.id


@pytest.mark.parametrize("outcome", ["failed", "timed_out", "error"])
def test_failed_attempt_remains_blocked_until_global_cooldown(tmp_path, outcome):
    """A terminal SSH failure must not allow another immediate Duo attempt."""

    now = [1000.0]
    store = _reuse_store(tmp_path, client_id="client-a", clock=lambda: now[0])
    grant = store.authorize_login(
        expected_revision=store.snapshot().revision,
        target="login",
        allow_offvpn=False,
        approval_reference="first attempt",
    )
    store.consume_login_grant(grant.id, "login")
    store.finish_login_attempt(grant.id, outcome=outcome, returncode=255)

    with pytest.raises(O2LoginGrantError, match="cooling down"):
        store.authorize_login(
            expected_revision=store.snapshot().revision,
            target="login",
            allow_offvpn=False,
            approval_reference="unsafe immediate retry",
        )

    now[0] += 301.0
    replacement = store.authorize_login(
        expected_revision=store.snapshot().revision,
        target="login",
        allow_offvpn=False,
        approval_reference="fresh approval after cooldown",
    )
    assert replacement.id != grant.id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("blocked_until", None),
        ("started_at", "not-a-timestamp"),
        ("outcome", "unknown"),
        ("grant_id", ""),
        ("allow_offvpn", "yes"),
    ],
)
def test_malformed_login_attempt_receipt_fails_closed(tmp_path, field, value):
    """Corrupt cooldown data cannot be interpreted as authorization to retry."""

    store = _reuse_store(tmp_path, client_id="client-a")
    grant = store.authorize_login(
        expected_revision=store.snapshot().revision,
        target="login",
        allow_offvpn=False,
        approval_reference="first attempt",
    )
    store.consume_login_grant(grant.id, "login")
    payload = json.loads(store.path.read_text())
    if value is None:
        payload["login_attempt"].pop(field)
    else:
        payload["login_attempt"][field] = value
    store.path.write_text(json.dumps(payload))
    store.path.chmod(0o600)

    snapshot = store.snapshot()
    assert snapshot.valid is False
    assert snapshot.effective_mode == "disabled"
    with pytest.raises(O2PolicyInvalidError):
        store.authorize_login(
            expected_revision=payload["revision"],
            target="login",
            allow_offvpn=False,
            approval_reference="must fail closed",
        )


def test_truncated_non_success_cooldown_receipt_fails_closed(tmp_path):
    """A plausible timestamp cannot shorten the workstation-wide retry delay."""

    store = _reuse_store(tmp_path, client_id="client-a")
    grant = store.authorize_login(
        expected_revision=store.snapshot().revision,
        target="login",
        allow_offvpn=False,
        approval_reference="first attempt",
    )
    store.consume_login_grant(grant.id, "login")
    payload = json.loads(store.path.read_text())
    # Keep the receipt internally ordered so this regression specifically proves
    # enforcement of the complete cooldown, not only the older monotonic check.
    payload["login_attempt"]["blocked_until"] = payload["login_attempt"]["started_at"] + 1.0
    store.path.write_text(json.dumps(payload))
    store.path.chmod(0o600)

    snapshot = store.snapshot()
    assert snapshot.valid is False
    assert snapshot.effective_mode == "disabled"


def test_concurrent_disable_revokes_grant_and_preserves_attempt_result(tmp_path):
    store = _reuse_store(tmp_path, client_id="client-a")
    grant = store.authorize_login(
        expected_revision=store.snapshot().revision,
        target="login",
        allow_offvpn=True,
        approval_reference="approved attempt",
    )
    store.consume_login_grant(grant.id, "login")

    disabled = O2PolicyStore(store.path, client_id="incident-task").disable(reason="new Duo incident")
    assert disabled["mode"] == "disabled"

    # The SSH-owning process may finish after another task disables O2. Recording
    # that outcome must not accidentally re-enable policy.
    store.finish_login_attempt(grant.id, outcome="failed", returncode=255)
    final = store.snapshot().state
    assert final["mode"] == "disabled"
    assert final["login_attempt"]["outcome"] == "failed"
    with pytest.raises(O2PolicyDeniedError):
        store.require_reuse_allowed()


def test_disable_repairs_owned_malformed_json_but_not_symlink(tmp_path):
    policy = tmp_path / "O2_POLICY.json"
    policy.write_text("{truncated")
    policy.chmod(0o600)
    store = O2PolicyStore(policy, client_id="client-a")

    repaired = store.disable(reason="repair to safest state")
    assert repaired["mode"] == "disabled"
    assert json.loads(policy.read_text())["schema_version"] == 1
