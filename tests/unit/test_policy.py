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
    DEFAULT_GRANT_TTL_SECONDS,
    MAX_LAUNCH_EVIDENCE_MINTS,
    SCHEMA_VERSION,
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
    store.enable_reuse(
        expected_revision=disabled["revision"],
        expected_generation=disabled["generation"],
        approval_reference="explicit test approval",
    )
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


def test_policy_path_cannot_alias_appended_mutex(tmp_path):
    """A policy named ``*.mutex`` still locks a separate stable inode."""

    store = O2PolicyStore(tmp_path / "custom-policy.mutex", client_id="client-a")

    state = store.disable(reason="initialize unusual absolute policy name")

    assert state["mode"] == "disabled"
    assert store.mutex_path == tmp_path / "custom-policy.mutex.mutex"
    assert store.mutex_path != store.path
    assert store.path.is_file() and store.mutex_path.is_file()


def test_reuse_enable_uses_compare_and_swap_revision(tmp_path):
    store = O2PolicyStore(tmp_path / "O2_POLICY.json", client_id="client-a")
    state = store.disable(reason="initialize")

    with pytest.raises(O2PolicyConflictError):
        store.enable_reuse(
            expected_revision=state["revision"] - 1,
            expected_generation=state["generation"],
            approval_reference="stale approval",
        )

    enabled = store.enable_reuse(
        expected_revision=state["revision"],
        expected_generation=state["generation"],
        approval_reference="explicit global re-enable",
    )
    assert enabled["mode"] == "reuse_only"
    assert enabled["revision"] == state["revision"] + 1


def test_repair_changes_generation_even_when_revision_repeats(tmp_path):
    """Malformed-state repair cannot create an ABA match for stale approval."""

    store = O2PolicyStore(tmp_path / "O2_POLICY.json", client_id="client-a")
    original = store.disable(reason="initial disabled state")
    store.path.write_text("{truncated")
    store.path.chmod(0o600)

    repaired = store.disable(reason="repair malformed policy")

    assert repaired["revision"] == original["revision"]
    assert repaired["generation"] != original["generation"]
    with pytest.raises(O2PolicyConflictError, match="generation/revision"):
        store.enable_reuse(
            expected_revision=original["revision"],
            expected_generation=original["generation"],
            approval_reference="stale pre-repair approval",
        )


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


def test_unsafe_policy_directory_fails_closed_for_reads_and_mutations(tmp_path):
    """A secure file cannot compensate for an replaceable parent directory."""

    policy_directory = tmp_path / "policy"
    real_directory = tmp_path / "real-policy"
    real_directory.mkdir(mode=0o700)
    policy_directory.symlink_to(real_directory, target_is_directory=True)
    store = O2PolicyStore(policy_directory / "O2_POLICY.json", client_id="client-a")

    assert store.snapshot().effective_mode == "disabled"
    with pytest.raises(O2PolicyInvalidError):
        store.disable(reason="must not repair an unsafe parent")


def test_owned_legacy_policy_directory_is_tightened_on_disable(monkeypatch, tmp_path):
    """An upgrade can initialize policy in the 0755 directory made by 0.2."""

    monkeypatch.setenv("HOME", str(tmp_path))
    policy_directory = tmp_path / ".agent_locks"
    policy_directory.mkdir(mode=0o755)
    policy_directory.chmod(0o755)
    store = O2PolicyStore(policy_directory / "O2_POLICY.json", client_id="client-a")

    # Reads remain fail-closed until an explicit mutation performs migration.
    assert store.snapshot().effective_mode == "disabled"
    state = store.disable(reason="initialize upgraded global policy")

    assert state["mode"] == "disabled"
    assert os.stat(policy_directory).st_mode & 0o777 == 0o700


def test_permissive_nonlegacy_policy_directory_is_not_modified(tmp_path):
    """A custom policy path cannot chmod an unrelated shared directory."""

    shared_directory = tmp_path / "shared-project"
    shared_directory.mkdir(mode=0o755)
    shared_directory.chmod(0o755)
    store = O2PolicyStore(shared_directory / "O2_POLICY.json", client_id="client-a")

    with pytest.raises(O2PolicyInvalidError, match="dedicated mode-0700"):
        store.disable(reason="must not change shared directory access")

    assert os.stat(shared_directory).st_mode & 0o777 == 0o755


def test_login_grant_is_client_target_and_time_scoped(tmp_path):
    now = [1000.0]
    owner = _reuse_store(tmp_path, client_id="owner", clock=lambda: now[0])
    revision = owner.snapshot().revision
    grant = owner.authorize_login(
        expected_revision=revision,
        expected_generation=owner.snapshot().generation,
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


def test_persisted_login_grant_cannot_exceed_maximum_ttl(tmp_path):
    """A corrupted receipt cannot extend one approval beyond five minutes."""

    store = _reuse_store(tmp_path, client_id="client-a")
    grant = store.authorize_login(
        expected_revision=store.snapshot().revision,
        expected_generation=store.snapshot().generation,
        target="login",
        allow_offvpn=False,
        approval_reference="bounded approval",
    )
    payload = json.loads(store.path.read_text())
    payload["login_grant"]["expires_at"] = grant.created_at + DEFAULT_GRANT_TTL_SECONDS + 1.0
    store.path.write_text(json.dumps(payload))
    store.path.chmod(0o600)

    snapshot = store.snapshot()
    assert snapshot.valid is False
    assert snapshot.effective_mode == "disabled"


def test_future_dated_login_grant_fails_closed(tmp_path):
    """Clock rollback cannot extend a five-minute grant until a future date."""

    now = [1000.0]
    store = _reuse_store(tmp_path, client_id="client-a", clock=lambda: now[0])
    store.authorize_login(
        expected_revision=store.snapshot().revision,
        expected_generation=store.snapshot().generation,
        target="login",
        allow_offvpn=False,
        approval_reference="bounded approval",
    )
    payload = json.loads(store.path.read_text())
    payload["login_grant"]["created_at"] = 2000.0
    payload["login_grant"]["expires_at"] = 2300.0
    store.path.write_text(json.dumps(payload))
    store.path.chmod(0o600)

    snapshot = store.snapshot()
    assert snapshot.valid is False
    assert snapshot.effective_mode == "disabled"


def test_reuse_enable_revokes_residual_disabled_grant(tmp_path):
    """Global reuse approval never revives an older login authorization."""

    store = _reuse_store(tmp_path, client_id="client-a")
    grant = store.authorize_login(
        expected_revision=store.snapshot().revision,
        expected_generation=store.snapshot().generation,
        target="login",
        allow_offvpn=False,
        approval_reference="older login approval",
    )
    payload = json.loads(store.path.read_text())
    payload["mode"] = "disabled"
    store.path.write_text(json.dumps(payload))
    store.path.chmod(0o600)
    observed = store.snapshot()

    enabled = store.enable_reuse(
        expected_revision=observed.revision,
        expected_generation=observed.generation,
        approval_reference="global reuse only",
    )

    assert enabled["mode"] == "reuse_only"
    assert enabled["login_grant"] is None
    assert any(
        event["event"] == "login_grant_revoked_on_reuse_enable" and event["grant_id"] == grant.id
        for event in enabled["events"]
    )


def test_grant_consumption_is_atomic_across_contenders(tmp_path):
    """Two callers sharing one grant can persist exactly one active attempt."""

    store = _reuse_store(tmp_path, client_id="same-client")
    grant = store.authorize_login(
        expected_revision=store.snapshot().revision,
        expected_generation=store.snapshot().generation,
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
    assert state["login_attempt"]["launcher_pid"] == os.getpid()


def test_broker_spawn_requires_exact_active_consumed_attempt(tmp_path):
    """A daemon cannot turn stale launch metadata into another SSH attempt."""

    store = _reuse_store(tmp_path, client_id="origin-client")
    grant = store.authorize_login(
        expected_revision=store.snapshot().revision,
        expected_generation=store.snapshot().generation,
        target="login",
        allow_offvpn=True,
        approval_reference="authorized broker launch",
    )
    store.consume_login_grant(grant.id, "login")

    # The exact active attempt is the only shape allowed immediately around the
    # daemon's SSH Popen.
    with store.authorize_consumed_broker_launch(
        grant.id,
        "login",
        client_id=grant.client_id,
        launcher_pid=os.getpid(),
    ):
        pass

    with pytest.raises(O2LoginGrantError, match="not bound"):  # noqa: SIM117 - Python 3.9 floor
        with store.authorize_consumed_broker_launch(
            grant.id,
            "transfer",
            client_id=grant.client_id,
            launcher_pid=os.getpid(),
        ):
            pass

    store.finish_login_attempt(grant.id, outcome="success", returncode=0)
    with pytest.raises(O2LoginGrantError, match="not bound"):  # noqa: SIM117 - Python 3.9 floor
        with store.authorize_consumed_broker_launch(
            grant.id,
            "login",
            client_id=grant.client_id,
            launcher_pid=os.getpid(),
        ):
            pass


def test_terminal_login_attempt_cannot_be_rewritten(tmp_path):
    """Cleanup errors must not corrupt immutable success/cooldown evidence."""

    store = _reuse_store(tmp_path)
    grant = store.authorize_login(
        expected_revision=store.snapshot().revision,
        expected_generation=store.snapshot().generation,
        target="login",
        allow_offvpn=True,
        approval_reference="terminal evidence test",
    )
    store.consume_login_grant(grant.id, "login")
    store.finish_login_attempt(grant.id, outcome="success", returncode=0)

    with pytest.raises(O2PolicyConflictError, match="already terminal"):
        store.finish_login_attempt(grant.id, outcome="failed", returncode=255)

    attempt = store.snapshot().state["login_attempt"]
    assert attempt["outcome"] == "success"
    assert attempt["blocked_until"] == attempt["finished_at"]


def test_disabled_policy_wins_before_broker_transport_spawn(tmp_path):
    """A consumed grant is not enough once the global mode becomes disabled."""

    store = _reuse_store(tmp_path, client_id="origin-client")
    grant = store.authorize_login(
        expected_revision=store.snapshot().revision,
        expected_generation=store.snapshot().generation,
        target="login",
        allow_offvpn=True,
        approval_reference="authorized broker launch",
    )
    store.consume_login_grant(grant.id, "login")
    store.disable(reason="incident arrived during local daemon handoff")

    with pytest.raises(O2PolicyDeniedError, match="disabled"):  # noqa: SIM117 - Python 3.9 floor
        with store.authorize_consumed_broker_launch(
            grant.id,
            "login",
            client_id=grant.client_id,
            launcher_pid=os.getpid(),
        ):
            pass


def test_active_attempt_blocks_new_authorization_until_cooldown(tmp_path):
    now = [1000.0]
    store = _reuse_store(tmp_path, client_id="client-a", clock=lambda: now[0])
    grant = store.authorize_login(
        expected_revision=store.snapshot().revision,
        expected_generation=store.snapshot().generation,
        target="login",
        allow_offvpn=False,
        approval_reference="first attempt",
    )
    store.consume_login_grant(grant.id, "login")

    with pytest.raises(O2LoginGrantError, match="cooling down"):
        store.authorize_login(
            expected_revision=store.snapshot().revision,
            expected_generation=store.snapshot().generation,
            target="login",
            allow_offvpn=False,
            approval_reference="unsafe immediate retry",
        )

    now[0] += 301.0
    replacement = store.authorize_login(
        expected_revision=store.snapshot().revision,
        expected_generation=store.snapshot().generation,
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
        expected_generation=store.snapshot().generation,
        target="login",
        allow_offvpn=False,
        approval_reference="first attempt",
    )
    store.consume_login_grant(grant.id, "login")
    store.finish_login_attempt(grant.id, outcome=outcome, returncode=255)

    with pytest.raises(O2LoginGrantError, match="cooling down"):
        store.authorize_login(
            expected_revision=store.snapshot().revision,
            expected_generation=store.snapshot().generation,
            target="login",
            allow_offvpn=False,
            approval_reference="unsafe immediate retry",
        )

    now[0] += 301.0
    replacement = store.authorize_login(
        expected_revision=store.snapshot().revision,
        expected_generation=store.snapshot().generation,
        target="login",
        allow_offvpn=False,
        approval_reference="fresh approval after cooldown",
    )
    assert replacement.id != grant.id


def test_standing_vpn_grant_is_audited_distinctly_and_cannot_allow_offvpn(tmp_path):
    """Standing route authority must not masquerade as fresh user approval."""

    store = _reuse_store(tmp_path, client_id="client-a")
    snapshot = store.snapshot()
    grant = store.authorize_login(
        expected_revision=snapshot.revision,
        expected_generation=snapshot.generation,
        target="login",
        allow_offvpn=False,
        approval_reference="standing on-VPN test authority",
        authorization_method="standing_on_vpn",
    )

    assert grant.authorization_method == "standing_on_vpn"
    state = store.snapshot().state
    assert state["login_grant"]["authorization_method"] == "standing_on_vpn"
    assert state["events"][-1]["authorization_method"] == "standing_on_vpn"
    assert store.revoke_unused_standing_grant(grant.id, reason="offline preflight failed") is True
    revoked = store.snapshot().state
    assert revoked["login_grant"] is None
    assert revoked["events"][-1]["event"] == "standing_login_grant_revoked"

    # A standing route rule can never be repurposed into the explicit off-VPN
    # exception, even if a caller tries to combine the two parameters directly.
    other = _reuse_store(tmp_path / "other", client_id="client-b")
    other_snapshot = other.snapshot()
    with pytest.raises(O2LoginGrantError, match="cannot allow off-VPN"):
        other.authorize_login(
            expected_revision=other_snapshot.revision,
            expected_generation=other_snapshot.generation,
            target="login",
            allow_offvpn=True,
            approval_reference="invalid standing authority",
            authorization_method="standing_on_vpn",
        )

    explicit = _reuse_store(tmp_path / "explicit", client_id="client-c")
    explicit_snapshot = explicit.snapshot()
    explicit_grant = explicit.authorize_login(
        expected_revision=explicit_snapshot.revision,
        expected_generation=explicit_snapshot.generation,
        target="login",
        allow_offvpn=False,
        approval_reference="fresh explicit approval",
    )
    with pytest.raises(O2LoginGrantError, match="cannot revoke an explicit"):
        explicit.revoke_unused_standing_grant(explicit_grant.id, reason="must preserve explicit authority")
    assert explicit.snapshot().state["login_grant"]["id"] == explicit_grant.id


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
        expected_generation=store.snapshot().generation,
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
            expected_generation=payload["generation"],
            target="login",
            allow_offvpn=False,
            approval_reference="must fail closed",
        )


def test_truncated_non_success_cooldown_receipt_fails_closed(tmp_path):
    """A plausible timestamp cannot shorten the workstation-wide retry delay."""

    store = _reuse_store(tmp_path, client_id="client-a")
    grant = store.authorize_login(
        expected_revision=store.snapshot().revision,
        expected_generation=store.snapshot().generation,
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
        expected_generation=store.snapshot().generation,
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
    assert json.loads(policy.read_text())["schema_version"] == SCHEMA_VERSION


def test_malformed_policy_repair_preserves_recent_retry_cooldown(tmp_path):
    """Truncation cannot erase evidence of a just-started login attempt."""

    now = [1000.0]
    policy = tmp_path / "O2_POLICY.json"
    policy.write_text("{truncated")
    policy.chmod(0o600)
    os.utime(policy, (900.0, 900.0))
    store = O2PolicyStore(policy, client_id="repair-client", clock=lambda: now[0])

    repaired = store.disable(reason="repair malformed state")
    enabled = store.enable_reuse(
        expected_revision=repaired["revision"],
        expected_generation=repaired["generation"],
        approval_reference="reuse only after repair",
    )

    attempt = enabled["login_attempt"]
    assert attempt["outcome"] == "error"
    assert attempt["started_at"] == 900.0
    assert attempt["blocked_until"] == 1200.0
    with pytest.raises(O2LoginGrantError, match="cooling down for 200.0s"):
        store.authorize_login(
            expected_revision=enabled["revision"],
            expected_generation=enabled["generation"],
            target="login",
            allow_offvpn=False,
            approval_reference="must remain blocked",
        )


# --- the durable launch-evidence ledger ---------------------------------------
def _mint(store, snapshot, *, evidence="a" * 64, reference="operator approved"):
    return store.record_launch_evidence_mint(
        expected_revision=snapshot["revision"],
        expected_generation=snapshot["generation"],
        approval_reference=reference,
        stage="platform-canary",
        job_id="52085188",
        package="/pkg/attempt-002",
        evidence_sha256=evidence,
        plan_sha256="b" * 64,
    )


def test_a_mint_survives_the_event_buffer_rolling_over(tmp_path):
    """The digest that authenticates a record must not age out of the ledger.

    `events` is a rolling buffer bounded to MAX_EVENTS and this file sees many
    events per session, so an approval recorded only there stops being
    verifiable after ordinary unrelated policy traffic -- tamper-evidence with a
    shelf life.
    """

    store = _reuse_store(tmp_path)
    _mint(store, store.snapshot().state)
    for index in range(200):
        state = store.snapshot().state
        store.disable(reason=f"churn {index}")
        store.enable_reuse(
            expected_revision=store.snapshot().state["revision"],
            expected_generation=state["generation"],
            approval_reference=f"churn {index}",
        )

    final = store.snapshot().state
    assert not any(event.get("event") == "launch_evidence_minted" for event in final["events"])
    assert [mint["evidence_sha256"] for mint in final["launch_evidence_mints"]] == ["a" * 64]
    assert final["launch_evidence_mints"][0]["plan_sha256"] == "b" * 64
    assert final["launch_evidence_mints"][0]["package"] == "/pkg/attempt-002"


def test_a_full_ledger_refuses_rather_than_evicting_an_attestation(tmp_path):
    """Dropping the oldest attestation to make room is the defect being fixed."""

    store = _reuse_store(tmp_path)
    for index in range(MAX_LAUNCH_EVIDENCE_MINTS):
        _mint(store, store.snapshot().state, evidence=f"{index:064x}")
    before = store.snapshot().state
    assert len(before["launch_evidence_mints"]) == MAX_LAUNCH_EVIDENCE_MINTS

    with pytest.raises(O2PolicyInvalidError, match="maximum of"):
        _mint(store, before, evidence="f" * 64)

    after = store.snapshot().state
    # Refused before anything was written: no revision bump, no partial event.
    assert after["revision"] == before["revision"]
    assert after["launch_evidence_mints"][0]["evidence_sha256"] == f"{0:064x}"
    assert not any(mint["evidence_sha256"] == "f" * 64 for mint in after["launch_evidence_mints"])


def test_schema_1_state_is_migrated_rather_than_invalidated(tmp_path):
    """Upgrading this code must not brick an existing policy file."""

    policy = tmp_path / "O2_POLICY.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation": "00000000-0000-4000-8000-000000000001",
                "revision": 7,
                "mode": "reuse_only",
                "login_grant": None,
                "login_attempt": None,
                "events": [{"at": 1.0, "event": "policy_reuse_enabled"}],
            }
        )
    )
    policy.chmod(0o600)
    store = O2PolicyStore(policy, client_id="client-a", clock=lambda: 1000.0)

    snapshot = store.snapshot()
    assert snapshot.valid is True
    assert snapshot.effective_mode == "reuse_only"
    assert snapshot.state["launch_evidence_mints"] == []
    assert snapshot.state["revision"] == 7
    # Reading migrates in memory only; the file is untouched until a write.
    assert json.loads(policy.read_text())["schema_version"] == 1

    _mint(store, snapshot.state)
    persisted = json.loads(policy.read_text())
    assert persisted["schema_version"] == SCHEMA_VERSION
    assert [mint["evidence_sha256"] for mint in persisted["launch_evidence_mints"]] == ["a" * 64]


def test_an_unknown_schema_is_still_refused(tmp_path):
    policy = tmp_path / "O2_POLICY.json"
    policy.write_text(json.dumps({"schema_version": 99, "generation": "x", "revision": 0, "mode": "disabled"}))
    policy.chmod(0o600)
    store = O2PolicyStore(policy, client_id="client-a")
    snapshot = store.snapshot()
    assert snapshot.valid is False
    assert "Unsupported O2 policy schema" in (snapshot.error or "")


def test_a_malformed_mint_ledger_is_refused(tmp_path):
    policy = tmp_path / "O2_POLICY.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "generation": "00000000-0000-4000-8000-000000000001",
                "revision": 1,
                "mode": "reuse_only",
                "login_grant": None,
                "login_attempt": None,
                "events": [],
                "launch_evidence_mints": ["not an object"],
            }
        )
    )
    policy.chmod(0o600)
    snapshot = O2PolicyStore(policy, client_id="client-a").snapshot()
    assert snapshot.valid is False
    assert "launch_evidence_mints" in (snapshot.error or "")
