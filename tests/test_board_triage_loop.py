"""Tests for the weft.board L1 feedback engine — SHADOW mode.

Covers weft-board-epic Task 7 / PRD §Compounding Loops "Triage Correction
Ratchet":
- The rule registry ships exactly 2 data-described rules (repeat-snooze,
  repeat-dismiss).
- `board_feedback_mode` config defaults to "shadow".
- `record_triage_event` / `act` append rows to `board_triage_events` (the
  loop's SIGNAL), and `act` rejects tools outside its write allowlist.
- The SHADOW NO-WRITE GATE (PRD Validation V8): a shadow pass over signals
  that WOULD trigger both rules writes proposals ONLY — tracker rows and
  alert counts are byte-for-byte unchanged, and `_apply_proposal` (the only
  mutation call site) is never invoked in shadow mode.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.board import (
    ACT_ALLOWLIST,
    DEFAULT_BOARD_FEEDBACK_MODE,
    DISMISS_DISTINCT_ITEM_THRESHOLD,
    RULE_REGISTRY,
    SNOOZE_REPEAT_THRESHOLD,
    TriageRule,
    act,
    get_board_feedback_mode,
    record_triage_event,
    run_feedback_pass,
)
from weft.models import (
    AlertChannel,
    AlertCreate,
    AlertType,
    NudgeMode,
    TrackerCreate,
    TrackerKind,
)


@pytest.fixture
def now():
    return datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Rule registry — data-described, exactly 2 rules
# ---------------------------------------------------------------------------


class TestRuleRegistry:
    """{name, signal_predicate, proposed_action} — 2 rules, no more."""

    def test_registry_has_exactly_two_rules(self):
        assert len(RULE_REGISTRY) == 2

    def test_registry_rule_names(self):
        names = {rule.name for rule in RULE_REGISTRY}
        assert names == {"repeat-snooze", "repeat-dismiss"}

    def test_registry_entries_are_triage_rule_dataclass_instances(self):
        for rule in RULE_REGISTRY:
            assert isinstance(rule, TriageRule)
            assert callable(rule.signal_predicate)
            assert callable(rule.proposed_action)

    def test_repeat_snooze_proposed_action_shape(self):
        rule = next(r for r in RULE_REGISTRY if r.name == "repeat-snooze")
        target = {"item_id": "tr-1", "source": "tracker", "kind": "task", "snooze_count": 3}
        change = rule.proposed_action(target)
        assert change["field"] == "nudge_interval"
        assert change["item_id"] == "tr-1"

    def test_repeat_dismiss_proposed_action_matches_by_kind_not_substring(self):
        """Ratified decision (weft-64c14697): hidden_kinds matches by
        KIND/ID equality, never name-substring — the proposed change must
        carry the exact kind identifier, not a name fragment."""
        rule = next(r for r in RULE_REGISTRY if r.name == "repeat-dismiss")
        target = {"source": "trigger", "kind": "time", "distinct_dismissed": 3}
        change = rule.proposed_action(target)
        assert change["add_to"] == "hidden_kinds"
        assert change["kind"] == "time"
        assert change["source"] == "trigger"
        # No name/title field anywhere in the shape — this is a kind/id
        # match, not a substring match against a title.
        assert "name" not in change
        assert "title" not in change


# ---------------------------------------------------------------------------
# board_feedback_mode config — defaults to shadow
# ---------------------------------------------------------------------------


class TestBoardFeedbackModeConfig:
    def test_default_constant_is_shadow(self):
        assert DEFAULT_BOARD_FEEDBACK_MODE == "shadow"

    def test_get_mode_defaults_to_shadow_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("WEFT_BOARD_FEEDBACK_MODE", raising=False)
        assert get_board_feedback_mode() == "shadow"

    def test_get_mode_respects_env_override(self, monkeypatch):
        monkeypatch.setenv("WEFT_BOARD_FEEDBACK_MODE", "active")
        assert get_board_feedback_mode() == "active"

        monkeypatch.setenv("WEFT_BOARD_FEEDBACK_MODE", "off")
        assert get_board_feedback_mode() == "off"

    def test_get_mode_falls_back_to_shadow_on_invalid_value(self, monkeypatch):
        monkeypatch.setenv("WEFT_BOARD_FEEDBACK_MODE", "not-a-real-mode")
        assert get_board_feedback_mode() == "shadow"


# ---------------------------------------------------------------------------
# record_triage_event — the SIGNAL append
# ---------------------------------------------------------------------------


class TestRecordTriageEvent:
    async def test_appends_a_row_with_expected_fields(self, pool, now):
        await record_triage_event(
            pool,
            item_id="tr-signal-1",
            source="tracker",
            kind="task",
            urgency_at_surface="overdue",
            age_days_at_surface=4.5,
            verb="snooze",
            snooze_duration_days=2.0,
        )

        row = await pool.fetchrow(
            "SELECT * FROM board_triage_events WHERE item_id = $1", "tr-signal-1",
        )
        assert row is not None
        assert row["source"] == "tracker"
        assert row["kind"] == "task"
        assert row["urgency_at_surface"] == "overdue"
        assert abs(row["age_days_at_surface"] - 4.5) < 0.01
        assert row["verb"] == "snooze"
        assert abs(row["snooze_duration_days"] - 2.0) < 0.01

    async def test_snooze_duration_days_defaults_to_null(self, pool):
        await record_triage_event(
            pool,
            item_id="tr-signal-2",
            source="tracker",
            kind="task",
            urgency_at_surface="due_soon",
            age_days_at_surface=1.0,
            verb="close",
        )
        row = await pool.fetchrow(
            "SELECT * FROM board_triage_events WHERE item_id = $1", "tr-signal-2",
        )
        assert row["snooze_duration_days"] is None


# ---------------------------------------------------------------------------
# act() — fires an existing write tool + appends the triage event
# ---------------------------------------------------------------------------


class TestAct:
    async def test_act_dismisses_tracker_and_appends_triage_event(self, pool, now):
        from weft.trackers import create_tracker, get_tracker

        tracker = await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.task,
                title="Act dismiss target",
                nudge_mode=NudgeMode.once,
                nudge_after=now - timedelta(hours=1),
            ),
        )

        result = await act(
            pool,
            tool="weft_tracker_dismiss",
            args={"tracker_id": tracker.id},
            item_id=tracker.id,
            source="tracker",
            kind="task",
            urgency_at_surface="overdue",
            age_days_at_surface=0.5,
            verb="dismiss",
        )

        assert result["id"] == tracker.id

        # The underlying write actually happened (once-mode tracker's
        # nudge_mode flips to none on dismiss).
        updated = await get_tracker(pool, tracker.id)
        assert updated.nudge_mode == NudgeMode.none

        # The triage event was appended.
        row = await pool.fetchrow(
            "SELECT * FROM board_triage_events WHERE item_id = $1", tracker.id,
        )
        assert row is not None
        assert row["verb"] == "dismiss"
        assert row["urgency_at_surface"] == "overdue"

    async def test_act_rejects_tool_outside_allowlist_no_write_no_event(self, pool):
        events_before = await pool.fetchval("SELECT count(*) FROM board_triage_events")

        with pytest.raises(ValueError):
            await act(
                pool,
                tool="weft_forget",  # not a triage write tool
                args={"id": "whatever"},
                item_id="whatever",
                source="tracker",
                kind="task",
                urgency_at_surface="overdue",
                age_days_at_surface=1.0,
                verb="close",
            )

        events_after = await pool.fetchval("SELECT count(*) FROM board_triage_events")
        assert events_after == events_before

    def test_allowlist_contains_only_named_triage_write_tools(self):
        """PRD §Ground Truth names these exact existing write tools."""
        assert set(ACT_ALLOWLIST) == {
            "weft_tracker_close",
            "weft_tracker_snooze",
            "weft_tracker_dismiss",
            "weft_tracker_update",
            "weft_alert_dismiss",
            "weft_trigger_delete",
            "weft_trigger_fire",
        }


# ---------------------------------------------------------------------------
# Rule firing against real seeded board_triage_events
# ---------------------------------------------------------------------------


class TestRuleFiring:
    async def test_repeat_snooze_fires_at_threshold(self, pool):
        for _ in range(SNOOZE_REPEAT_THRESHOLD):
            await record_triage_event(
                pool, item_id="tr-repeat", source="tracker", kind="task",
                urgency_at_surface="overdue", age_days_at_surface=1.0,
                verb="snooze", snooze_duration_days=1.0,
            )

        proposals = await run_feedback_pass(pool, mode="shadow")
        snooze_proposals = [p for p in proposals if p["rule"] == "repeat-snooze"]
        assert len(snooze_proposals) == 1
        assert snooze_proposals[0]["target_id"] == "tr-repeat"

    async def test_repeat_snooze_does_not_fire_below_threshold(self, pool):
        for _ in range(SNOOZE_REPEAT_THRESHOLD - 1):
            await record_triage_event(
                pool, item_id="tr-not-enough", source="tracker", kind="task",
                urgency_at_surface="overdue", age_days_at_surface=1.0,
                verb="snooze", snooze_duration_days=1.0,
            )

        proposals = await run_feedback_pass(pool, mode="shadow")
        assert [p for p in proposals if p["rule"] == "repeat-snooze"] == []

    async def test_repeat_dismiss_fires_across_distinct_items(self, pool):
        for i in range(DISMISS_DISTINCT_ITEM_THRESHOLD):
            await record_triage_event(
                pool, item_id=f"trg-{i}", source="trigger", kind="time",
                urgency_at_surface="pending", age_days_at_surface=2.0,
                verb="dismiss",
            )

        proposals = await run_feedback_pass(pool, mode="shadow")
        dismiss_proposals = [p for p in proposals if p["rule"] == "repeat-dismiss"]
        assert len(dismiss_proposals) == 1
        assert dismiss_proposals[0]["target_id"] == "trigger:time"

    async def test_repeat_dismiss_does_not_fire_with_fewer_distinct_items(self, pool):
        for i in range(DISMISS_DISTINCT_ITEM_THRESHOLD - 1):
            await record_triage_event(
                pool, item_id=f"trg-few-{i}", source="trigger", kind="time",
                urgency_at_surface="pending", age_days_at_surface=2.0,
                verb="dismiss",
            )

        proposals = await run_feedback_pass(pool, mode="shadow")
        assert [p for p in proposals if p["rule"] == "repeat-dismiss"] == []

    async def test_repeat_dismiss_does_not_fire_when_same_item_dismissed_repeatedly(self, pool):
        """Distinct-item counting, not raw event counting: the same item_id
        dismissed 3x is NOT 3 distinct items."""
        for _ in range(DISMISS_DISTINCT_ITEM_THRESHOLD):
            await record_triage_event(
                pool, item_id="trg-same", source="trigger", kind="time",
                urgency_at_surface="pending", age_days_at_surface=2.0,
                verb="dismiss",
            )

        proposals = await run_feedback_pass(pool, mode="shadow")
        assert [p for p in proposals if p["rule"] == "repeat-dismiss"] == []

    async def test_off_mode_returns_no_proposals_and_writes_nothing(self, pool):
        for _ in range(SNOOZE_REPEAT_THRESHOLD):
            await record_triage_event(
                pool, item_id="tr-off-mode", source="tracker", kind="task",
                urgency_at_surface="overdue", age_days_at_surface=1.0,
                verb="snooze", snooze_duration_days=1.0,
            )

        proposals = await run_feedback_pass(pool, mode="off")
        assert proposals == []

        count = await pool.fetchval("SELECT count(*) FROM board_feedback_proposals")
        assert count == 0


# ---------------------------------------------------------------------------
# THE SHADOW NO-WRITE GATE — PRD Validation V8 (the load-bearing property)
# ---------------------------------------------------------------------------


class TestShadowNoWriteGate:
    """With events seeded so BOTH rules fire, a shadow pass must write
    exactly 2 proposals and mutate NO tracker/alert row. This is the hard
    gate from PRD §Validation V8 and the Epic's Critical Implementation
    Note — not a soft convention."""

    async def _seed_both_rules_plus_noise(self, pool, tracker_id, now):
        """~12 board_triage_events: one tracker snoozed 3x (repeat-snooze),
        one (source, kind) dismissed across 3 distinct items
        (repeat-dismiss), plus noise events that must NOT push any
        non-qualifying signal over threshold."""
        # Rule 1 target: same tracker snoozed 3x.
        for _ in range(SNOOZE_REPEAT_THRESHOLD):
            await record_triage_event(
                pool, item_id=tracker_id, source="tracker", kind="task",
                urgency_at_surface="overdue", age_days_at_surface=5.0,
                verb="snooze", snooze_duration_days=1.0,
            )

        # Rule 2 target: 3 distinct trigger items of kind "time" dismissed.
        for i in range(DISMISS_DISTINCT_ITEM_THRESHOLD):
            await record_triage_event(
                pool, item_id=f"trg-dismiss-{i}", source="trigger", kind="time",
                urgency_at_surface="pending", age_days_at_surface=3.0,
                verb="dismiss",
            )

        # Noise: a different tracker snoozed only once (below threshold).
        await record_triage_event(
            pool, item_id="tr-noise-snooze", source="tracker", kind="task",
            urgency_at_surface="due_soon", age_days_at_surface=0.5,
            verb="snooze", snooze_duration_days=1.0,
        )

        # Noise: a dismiss of a different kind (doesn't add to the "time"
        # distinct-item count).
        await record_triage_event(
            pool, item_id="trg-noise-kind", source="trigger", kind="threshold",
            urgency_at_surface="no_date", age_days_at_surface=6.0,
            verb="dismiss",
        )

        # Noise: unrelated verb (close) on 3 different items — proves
        # non-snooze/dismiss verbs don't leak into either rule.
        for i in range(3):
            await record_triage_event(
                pool, item_id=f"tr-closed-{i}", source="tracker", kind="task",
                urgency_at_surface="overdue", age_days_at_surface=2.0,
                verb="close",
            )

        # Total: 3 + 3 + 1 + 1 + 3 = 11; one more noise row to round out
        # ~12 seeded events (Epic done_when).
        await record_triage_event(
            pool, item_id="tr-noise-extra", source="tracker", kind="follow_up",
            urgency_at_surface="pending", age_days_at_surface=10.0,
            verb="snooze", snooze_duration_days=1.0,
        )

    async def test_shadow_pass_writes_two_proposals_tracker_and_alerts_unchanged(
        self, pool, now,
    ):
        from weft.alerts import create_alert
        from weft.trackers import create_tracker

        tracker = await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.task,
                title="Repeatedly snoozed tracker",
                nudge_mode=NudgeMode.recur,
                nudge_after=now - timedelta(hours=2),
                nudge_interval=timedelta(days=1),
            ),
        )
        alert = await create_alert(
            pool,
            AlertCreate(
                alert_type=AlertType.due_task,
                title="Unrelated alert",
                trigger_at=now + timedelta(days=1),
                channel=AlertChannel.log,
            ),
        )

        await self._seed_both_rules_plus_noise(pool, tracker.id, now)

        seeded_event_count = await pool.fetchval(
            "SELECT count(*) FROM board_triage_events",
        )
        assert seeded_event_count == 12

        tracker_row_before = dict(
            await pool.fetchrow("SELECT * FROM trackers WHERE id = $1", tracker.id)
        )
        alert_row_before = dict(
            await pool.fetchrow("SELECT * FROM alerts WHERE id = $1", alert.id)
        )
        tracker_count_before = await pool.fetchval("SELECT count(*) FROM trackers")
        alert_count_before = await pool.fetchval("SELECT count(*) FROM alerts")

        proposals = await run_feedback_pass(pool, mode="shadow")

        # (a) Exactly 2 proposals written, one per rule.
        assert len(proposals) == 2
        assert {p["rule"] for p in proposals} == {"repeat-snooze", "repeat-dismiss"}
        for p in proposals:
            assert p["mode"] == "shadow"

        proposal_count = await pool.fetchval(
            "SELECT count(*) FROM board_feedback_proposals",
        )
        assert proposal_count == 2

        # (b) Tracker row is BYTE-FOR-BYTE unchanged.
        tracker_row_after = dict(
            await pool.fetchrow("SELECT * FROM trackers WHERE id = $1", tracker.id)
        )
        assert tracker_row_before == tracker_row_after

        # (c) Alert row is BYTE-FOR-BYTE unchanged, and no new alert fired.
        alert_row_after = dict(
            await pool.fetchrow("SELECT * FROM alerts WHERE id = $1", alert.id)
        )
        assert alert_row_before == alert_row_after

        tracker_count_after = await pool.fetchval("SELECT count(*) FROM trackers")
        alert_count_after = await pool.fetchval("SELECT count(*) FROM alerts")
        assert tracker_count_before == tracker_count_after
        assert alert_count_before == alert_count_after

    async def test_shadow_mode_never_calls_apply_proposal(self, pool, now, monkeypatch):
        """Direct proof the mode gate is load-bearing: patch the ONLY
        mutation call site with a counter. Shadow mode must call it zero
        times; active mode (same seeded data) must call it once per
        proposal. If the `mode == "active"` gate in `run_feedback_pass`
        were removed, this assertion would fail."""
        import weft.board as board_module
        from weft.trackers import create_tracker

        tracker = await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.task,
                title="Gate-proof tracker",
                nudge_mode=NudgeMode.once,
                nudge_after=now - timedelta(hours=1),
            ),
        )
        await self._seed_both_rules_plus_noise(pool, tracker.id, now)

        calls: list[tuple[str, dict]] = []

        async def _fake_apply_proposal(pool_arg, rule_name, proposal):
            calls.append((rule_name, proposal))

        monkeypatch.setattr(board_module, "_apply_proposal", _fake_apply_proposal)

        shadow_proposals = await run_feedback_pass(pool, mode="shadow")
        assert len(shadow_proposals) == 2
        assert calls == [], (
            "shadow mode must never call _apply_proposal — the SHADOW "
            "NO-WRITE GATE (PRD V8) is broken if this fires"
        )

        # Re-seed (proposals accumulate independently of events) and run
        # active mode against the SAME signal — proves the gate is a real
        # branch, not a predicate that never matches.
        active_proposals = await run_feedback_pass(pool, mode="active")
        assert len(active_proposals) == 2
        assert len(calls) == 2, (
            "active mode should call _apply_proposal once per proposal — "
            "if this is 0, the gate is unreachable and the test above is "
            "vacuous"
        )
