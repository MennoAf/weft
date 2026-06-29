"""Pydantic models for Weft memory records and relationships."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    preference = "preference"
    fact = "fact"
    pattern = "pattern"
    relationship = "relationship"
    solution = "solution"
    architecture = "architecture"
    user_model = "user_model"
    handoff = "handoff"
    issue = "issue"
    decision = "decision"
    milestone = "milestone"
    anti_pattern = "anti_pattern"


# Literal union of all MemoryType values — used in MCP tool signatures so the
# JSON Schema explicitly enumerates valid types.  This prevents LLM clients
# from pre-validating against an inferred (potentially stale) enum.
# Keep in sync with MemoryType above.
MemoryTypeLiteral = Literal[
    "preference", "fact", "pattern", "relationship", "solution",
    "architecture", "user_model", "handoff", "issue", "decision",
    "milestone", "anti_pattern",
]


class MemorySource(str, Enum):
    conversation = "conversation"
    code = "code"
    documentation = "documentation"
    inference = "inference"
    seed = "seed"
    ingest = "ingest"


# Literal union of all MemorySource values — used in MCP tool signatures so the
# JSON Schema explicitly enumerates valid sources.
# Keep in sync with MemorySource above.
MemorySourceLiteral = Literal[
    "conversation", "code", "documentation", "inference", "seed", "ingest",
]


class MemoryStatus(str, Enum):
    active = "active"
    archived = "archived"
    decayed = "decayed"


class RelationType(str, Enum):
    supersedes = "supersedes"
    related_to = "related_to"
    contradicts = "contradicts"
    derived_from = "derived_from"
    # Links a cross-project merge candidate (review_status='pending_review',
    # 0.6<=sim<0.85) to the existing belief it would merge into. Edge points
    # candidate(source) -> existing(target). Read by the quarantine merge action.
    merge_candidate = "merge_candidate"


def _weft_id() -> str:
    return f"weft-{uuid.uuid4().hex[:8]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Memory(BaseModel):
    """A single memory record."""

    id: str = Field(default_factory=_weft_id)
    type: MemoryType
    topic: list[str] = Field(default_factory=list)
    content: str
    source: MemorySource = MemorySource.conversation
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    token_count: int = 0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    accessed_at: datetime = Field(default_factory=_now)
    access_count: int = 0
    project_id: str | None = None
    agent_id: str | None = None
    workspace_id: str | None = None
    status: MemoryStatus = MemoryStatus.active
    pinned: bool = False
    usefulness_score: float = Field(default=0.7, ge=0.0, le=1.0)
    usefulness_count: int = 0
    last_boosted_at: datetime | None = None
    review_after: datetime | None = None
    write_provenance: str = "supervisor"
    review_status: str = "active"
    project_facets: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        d = self.model_dump(mode="json")
        d["type"] = self.type.value
        d["source"] = self.source.value
        d["status"] = self.status.value
        d["usefulness_score"] = self.usefulness_score
        d["usefulness_count"] = self.usefulness_count
        return d


class MemoryRelationship(BaseModel):
    """A typed relationship between two memories."""

    source_id: str
    target_id: str
    relation: RelationType
    created_at: datetime = Field(default_factory=_now)


class MemoryCreate(BaseModel):
    """Input model for creating a memory (used by MCP tools)."""

    type: MemoryType
    content: str
    topic: list[str] = Field(default_factory=list)
    source: MemorySource = MemorySource.conversation
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    project_id: str | None = None
    agent_id: str | None = None
    workspace_id: str | None = None
    pinned: bool = False
    review_after: datetime | None = None
    project_facets: list[str] = Field(default_factory=list)


class Workspace(BaseModel):
    """A shared-brain primitive: a named bucket of memories that multiple
    user_ids can read. The owner (created_by) plus members listed in
    workspace_members can SELECT memories with matching workspace_id."""

    id: str
    name: str
    description: str | None = None
    created_by: str
    install_pubkey: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class WorkspaceMember(BaseModel):
    """A membership row. member_identity carries kind+id so federation can
    later add remote-install members without a schema change."""

    workspace_id: str
    member_identity: dict[str, Any]
    role: str = "member"
    added_by: str
    added_at: datetime = Field(default_factory=_now)

    @property
    def user_id(self) -> str | None:
        if self.member_identity.get("kind") == "local_user":
            return self.member_identity.get("user_id")
        return None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class TrackerKind(str, Enum):
    outreach = "outreach"
    task = "task"
    follow_up = "follow_up"
    meal_plan = "meal_plan"
    shopping_list = "shopping_list"
    pantry = "pantry"
    watch = "watch"
    list = "list"
    trace = "trace"


TrackerKindLiteral = Literal[
    "outreach", "task", "follow_up", "meal_plan",
    "shopping_list", "pantry", "watch", "list", "trace",
]


class TrackerState(str, Enum):
    in_progress = "in_progress"
    awaiting_reply = "awaiting_reply"
    blocked = "blocked"
    done = "done"
    abandoned = "abandoned"

    @classmethod
    def open_states(cls) -> set[str]:
        return {"in_progress", "awaiting_reply", "blocked"}

    @classmethod
    def terminal_states(cls) -> set[str]:
        return {"done", "abandoned"}


TrackerStateLiteral = Literal[
    "in_progress", "awaiting_reply", "blocked", "done", "abandoned",
]


class NudgeMode(str, Enum):
    none = "none"
    once = "once"
    recur = "recur"


NudgeModeLiteral = Literal["none", "once", "recur"]


def _tracker_id() -> str:
    return f"tr-{uuid.uuid4().hex[:10]}"


class Tracker(BaseModel):
    """A lifecycle-aware open-loop primitive. State changes over time;
    nudges fire on schedule; can be snoozed, dismissed, or closed.

    Wick Phase 3 — see weft_v2_spec.md §3 for the full design."""

    id: str = Field(default_factory=_tracker_id)
    user_id: str | None = None
    project_id: str | None = None
    entity_id: str | None = None
    kind: TrackerKind
    title: str
    state: TrackerState = TrackerState.in_progress
    state_history: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    last_touch: datetime = Field(default_factory=_now)
    nudge_mode: NudgeMode = NudgeMode.none
    nudge_after: datetime | None = None
    nudge_interval: timedelta | None = None
    snooze_until: datetime | None = None
    trigger_ids: list[str] = Field(default_factory=list)
    provenance: str = "supervisor"
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def is_open(self) -> bool:
        return self.state.value in TrackerState.open_states()

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["kind"] = self.kind.value
        d["state"] = self.state.value
        d["nudge_mode"] = self.nudge_mode.value
        if self.nudge_interval is not None:
            d["nudge_interval_seconds"] = int(self.nudge_interval.total_seconds())
        return d


class TrackerCreate(BaseModel):
    """Input model for creating a tracker."""

    kind: TrackerKind
    title: str
    project_id: str | None = None
    entity_id: str | None = None
    state: TrackerState = TrackerState.in_progress
    context: dict[str, Any] = Field(default_factory=dict)
    nudge_mode: NudgeMode = NudgeMode.none
    nudge_after: datetime | None = None
    nudge_interval: timedelta | None = None
    provenance: str = "supervisor"


class BehaviorScope(str, Enum):
    global_ = "global"
    project = "project"
    agent = "agent"


class Behavior(BaseModel):
    """A persistent behavioral rule — tells agents how to act, not what happened."""

    id: str = Field(default_factory=_weft_id)
    trigger_pattern: str
    action: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    scope: BehaviorScope = BehaviorScope.global_
    project_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    priority: int = Field(default=0, description="Higher = stronger override")
    enabled: bool = True
    access_count: int = 0
    token_count: int = 0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    status: str = "active"
    write_provenance: str = "supervisor"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        d = self.model_dump(mode="json")
        d["scope"] = self.scope.value
        return d


class BehaviorCreate(BaseModel):
    """Input model for creating a behavior."""

    trigger_pattern: str
    action: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    scope: BehaviorScope = BehaviorScope.global_
    project_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    priority: int = 0
    enabled: bool = True


class BehaviorMatch(BaseModel):
    """Result from a behavior search operation."""

    behavior: Behavior
    similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = self.behavior.to_dict()
        d["similarity"] = round(self.similarity, 4)
        return d


class EntityType(str, Enum):
    person = "person"
    project = "project"
    company = "company"
    tool = "tool"
    concept = "concept"


class Entity(BaseModel):
    """A first-class entity (person, project, company, tool, concept)."""

    id: str = Field(default_factory=_weft_id)
    name: str
    entity_type: EntityType = EntityType.concept
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    project_id: str | None = None
    agent_id: str | None = None
    status: str = "active"
    mention_count: int = 0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        d = self.model_dump(mode="json")
        d["entity_type"] = self.entity_type.value
        return d


class EntityCreate(BaseModel):
    """Input model for creating an entity."""

    name: str
    entity_type: EntityType = EntityType.concept
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    project_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None


class EpisodeStatus(str, Enum):
    open = "open"
    closed = "closed"
    expired = "expired"
    graduated = "graduated"


class Episode(BaseModel):
    """A time-bounded grouping of memories into a causal sequence."""

    id: str = Field(default_factory=_weft_id)
    title: str
    summary: str | None = None
    project_id: str | None = None
    agent_id: str | None = None
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    expires_at: datetime | None = None
    graduated_memory_id: str | None = None
    status: EpisodeStatus = EpisodeStatus.open
    token_count: int = 0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @property
    def is_expired(self) -> bool:
        """Check if this episode has passed its TTL."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        d = self.model_dump(mode="json")
        d["status"] = self.status.value
        return d


class EpisodeCreate(BaseModel):
    """Input model for creating an episode."""

    title: str
    summary: str | None = None
    project_id: str | None = None
    agent_id: str | None = None
    ttl_hours: float | None = None


class EpisodeWithMemories(BaseModel):
    """An episode with its linked memories in order."""

    episode: Episode
    memories: list[Memory] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.episode.to_dict()
        d["memories"] = [m.to_dict() for m in self.memories]
        d["memory_count"] = len(self.memories)
        return d


class TurnRole(str, Enum):
    user = "user"
    assistant = "assistant"
    tool = "tool"
    system = "system"


TurnRoleLiteral = Literal["user", "assistant", "tool", "system"]


def _turn_id() -> str:
    return f"et-{uuid.uuid4().hex[:10]}"


class EpisodeTurn(BaseModel):
    """A single conversational turn inside an episode — raw dialogue trace.

    Lives alongside Episode + EpisodeMemory, never replacing them. Each turn
    carries its own embedding for retrieval, its own occurred_at for temporal
    queries, and an optional trace_id mapping to Wick's run_id. The
    importance_score is the Face hook for retention gating; populated async
    post-hoc, gates behavior at graduation rather than ingest.
    """

    id: str = Field(default_factory=_turn_id)
    episode_id: str
    turn_index: int
    role: TurnRole
    content: str
    occurred_at: datetime = Field(default_factory=_now)
    trace_id: str | None = None
    importance_score: float | None = None
    token_count: int = 0
    user_id: str | None = None
    created_at: datetime = Field(default_factory=_now)
    # Boost-loop signals (v46). Mirror the belief-tier scoring fields so
    # turn-tier recall can rerank by usefulness × recency the same way
    # ``score_memory`` does. ``last_boosted_at`` is None until the first
    # session-end boost lands.
    usefulness_score: float = Field(default=0.7, ge=0.0, le=1.0)
    usefulness_count: int = 0
    last_boosted_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["role"] = self.role.value
        return d


class EpisodeTurnCreate(BaseModel):
    """Input model for appending a turn to an episode."""

    episode_id: str
    role: TurnRole
    content: str
    occurred_at: datetime | None = None
    trace_id: str | None = None


class ModeWeights(BaseModel):
    """Retrieval weight overrides for a named mode/persona.

    vector_weight/bm25_weight: balance between semantic and keyword search [0.0–1.0].
    recency_bias: preference for recent memories [0.0–1.0], 0 = no preference.
    entity_boost/behavior_boost: multiplicative factors [0.0–10.0], 1.0 = neutral.
    """

    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    recency_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    entity_boost: float = Field(default=1.0, ge=0.0, le=10.0)
    behavior_boost: float = Field(default=1.0, ge=0.0, le=10.0)


class ModeCreate(BaseModel):
    """Input model for creating a mode."""

    name: str
    description: str | None = None
    weights: ModeWeights = Field(default_factory=ModeWeights)
    project_id: str | None = None
    agent_id: str | None = None


class Mode(BaseModel):
    """A named retrieval persona with weight overrides, scoped per user."""

    id: str = Field(default_factory=_weft_id)
    user_id: str | None = None
    name: str
    description: str | None = None
    weights: ModeWeights = Field(default_factory=ModeWeights)
    project_id: str | None = None
    agent_id: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        return self.model_dump(mode="json")


class AlertType(str, Enum):
    due_task = "due_task"
    stale_decision = "stale_decision"
    follow_up = "follow_up"
    custom = "custom"
    daily_brief = "daily_brief"
    check_in_low_mood = "check_in_low_mood"
    check_in_low_sleep = "check_in_low_sleep"
    check_in_declining_trend = "check_in_declining_trend"
    loom_stale_claim = "loom_stale_claim"
    loom_epic_ready = "loom_epic_ready"
    loom_blocked_pile_up = "loom_blocked_pile_up"
    memory_consolidation_overdue = "memory_consolidation_overdue"
    memory_count_threshold = "memory_count_threshold"
    memory_contradiction = "memory_contradiction"


class AlertChannel(str, Enum):
    log = "log"
    slack = "slack"


class AlertStatus(str, Enum):
    pending = "pending"
    fired = "fired"
    dismissed = "dismissed"


class AlertCreate(BaseModel):
    """Input model for creating an alert.

    trigger_at must be a timezone-aware UTC datetime.
    Use datetime.now(timezone.utc) for immediate delivery or a future timestamp.
    """

    alert_type: AlertType
    title: str
    body: str | None = None
    trigger_at: datetime
    channel: AlertChannel = AlertChannel.log
    channel_target: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    agent_id: str | None = None

    @classmethod
    def _validate_trigger_at(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("trigger_at must be timezone-aware")
        return v

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._validate_trigger_at(self.trigger_at)


class Alert(BaseModel):
    """A proactive alert — scheduled notification that fires at trigger_at."""

    id: str = Field(default_factory=_weft_id)
    user_id: str | None = None
    alert_type: AlertType
    title: str
    body: str | None = None
    trigger_at: datetime
    status: AlertStatus = AlertStatus.pending
    channel: AlertChannel = AlertChannel.log
    channel_target: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    agent_id: str | None = None
    fired_at: datetime | None = None
    created_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        d = self.model_dump(mode="json")
        d["alert_type"] = self.alert_type.value
        d["status"] = self.status.value
        d["channel"] = self.channel.value
        return d


class TriggerConditionType(str, Enum):
    """Type of condition that activates a trigger."""

    time = "time"          # Fires at/after a specific time or on a cron schedule
    threshold = "threshold"  # Fires when a metric exceeds a value
    event = "event"        # Fires when a named event occurs
    absence = "absence"    # Fires when something hasn't happened for N hours


class TriggerStatus(str, Enum):
    enabled = "enabled"
    disabled = "disabled"
    fired = "fired"        # One-shot trigger that has already fired


class TriggerCreate(BaseModel):
    """Input model for creating a proactive trigger.

    Condition payloads are validated per condition_type:
      time:      requires 'trigger_at' (ISO datetime string)
      threshold: requires 'metric' (str) and 'threshold' (number)
      event:     requires 'event_name' (str)
      absence:   requires 'absence_hours' (positive number)
    """

    name: str
    condition_type: TriggerConditionType
    condition: dict[str, Any] = Field(default_factory=dict)
    action: str  # Description of what should happen when triggered
    cooldown_hours: float | None = None  # Minimum hours between firings
    max_fires: int | None = None  # None = unlimited
    project_id: str | None = None
    agent_id: str | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._validate_condition()

    def _validate_condition(self) -> None:
        ct = self.condition_type
        c = self.condition

        if ct == TriggerConditionType.time:
            if "trigger_at" not in c:
                raise ValueError("time trigger requires 'trigger_at' in condition")
            try:
                datetime.fromisoformat(c["trigger_at"])
            except (ValueError, TypeError) as e:
                raise ValueError(f"trigger_at must be a valid ISO datetime: {e}") from e

        elif ct == TriggerConditionType.threshold:
            if "metric" not in c:
                raise ValueError("threshold trigger requires 'metric' in condition")
            if "threshold" not in c:
                raise ValueError("threshold trigger requires 'threshold' in condition")
            if not isinstance(c["threshold"], (int, float)):
                raise ValueError("threshold must be a number")

        elif ct == TriggerConditionType.event:
            if "event_name" not in c:
                raise ValueError("event trigger requires 'event_name' in condition")

        elif ct == TriggerConditionType.absence:
            if "absence_hours" not in c:
                raise ValueError("absence trigger requires 'absence_hours' in condition")
            if not isinstance(c["absence_hours"], (int, float)) or c["absence_hours"] <= 0:
                raise ValueError("absence_hours must be a positive number")


class Trigger(BaseModel):
    """A proactive trigger — condition-driven rule that fires actions."""

    id: str = Field(default_factory=_weft_id)
    name: str
    condition_type: TriggerConditionType
    condition: dict[str, Any] = Field(default_factory=dict)
    action: str
    status: TriggerStatus = TriggerStatus.enabled
    cooldown_hours: float | None = None
    max_fires: int | None = None
    fire_count: int = 0
    last_fired_at: datetime | None = None
    project_id: str | None = None
    agent_id: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    write_provenance: str = "supervisor"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        d = self.model_dump(mode="json")
        d["condition_type"] = self.condition_type.value
        d["status"] = self.status.value
        return d


class CheckInCreate(BaseModel):
    """Input model for logging a mood/sleep/energy check-in."""

    mood: int | None = None  # 1-5 scale
    sleep_hours: float | None = None
    energy: int | None = None  # 1-5 scale
    notes: str | None = None
    logged_at: datetime | None = None  # defaults to now() in DB

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.mood is not None and not 1 <= self.mood <= 5:
            raise ValueError("mood must be 1-5")
        if self.energy is not None and not 1 <= self.energy <= 5:
            raise ValueError("energy must be 1-5")
        if self.sleep_hours is not None and not 0 <= self.sleep_hours <= 24:
            raise ValueError("sleep_hours must be 0-24")


class CheckIn(BaseModel):
    """A mood/sleep/energy check-in record."""

    id: str = Field(default_factory=_weft_id)
    user_id: str | None = None
    mood: int | None = None
    sleep_hours: float | None = None
    energy: int | None = None
    notes: str | None = None
    logged_at: datetime = Field(default_factory=_now)
    created_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        return self.model_dump(mode="json")


class CalibrationOutcome(str, Enum):
    approved = "approved"
    rejected = "rejected"
    modified = "modified"


class CalibrationCreate(BaseModel):
    """Input model for recording an agent action calibration."""

    action_category: str
    action_description: str
    outcome: CalibrationOutcome
    agent_id: str | None = None
    project_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class CalibrationRecord(BaseModel):
    """A record of an agent action outcome (approved/rejected/modified)."""

    id: str = Field(default_factory=_weft_id)
    action_category: str
    action_description: str
    outcome: CalibrationOutcome
    agent_id: str | None = None
    project_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    user_id: str | None = None
    # Trust tier of the caller that attested this outcome (weft/auth.py caller
    # mode): 'supervisor' (trusted) vs 'agent' (untrusted). Only trusted-origin
    # approvals drive auto-promotion. See calibration.TRUSTED_CALIBRATION_ORIGINS.
    origin: str = "supervisor"
    created_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        d = self.model_dump(mode="json")
        d["outcome"] = self.outcome.value
        return d


class DegradationTriggerType(str, Enum):
    """Type of condition that triggers a degradation policy."""

    low_confidence = "low_confidence"      # Memory confidence drops below threshold
    api_error = "api_error"                # Repeated API/embedding failures
    context_decay = "context_decay"        # Context window saturation or staleness
    budget_breach = "budget_breach"        # Token budget exceeded
    cost_breach = "cost_breach"            # Cost (USD) budget % exceeded


DegradationTriggerTypeLiteral = Literal[
    "low_confidence", "api_error", "context_decay", "budget_breach", "cost_breach",
]


class DegradationAction(str, Enum):
    """Action to take when a degradation policy fires."""

    pause = "pause"          # Suspend the operation
    escalate = "escalate"    # Escalate to human/supervisor
    restart = "restart"      # Restart the operation from scratch
    restrict = "restrict"    # Limit scope or capabilities


DegradationActionLiteral = Literal[
    "pause", "escalate", "restart", "restrict",
]


class DegradationPolicyStatus(str, Enum):
    active = "active"
    disabled = "disabled"
    fired = "fired"


class DegradationPolicyCreate(BaseModel):
    """Input model for creating a degradation policy.

    Condition payloads are validated per trigger_type:
      low_confidence: requires 'threshold' (float 0.0-1.0)
      api_error:      requires 'max_errors' (positive int) and 'window_minutes' (positive number)
      context_decay:  requires 'max_age_hours' (positive number)
      budget_breach:  requires 'max_tokens' (positive int)
    """

    name: str
    trigger_type: DegradationTriggerType
    condition: dict[str, Any] = Field(default_factory=dict)
    action: DegradationAction
    description: str | None = None
    cooldown_minutes: float | None = None  # Minimum minutes between firings
    max_fires: int | None = None           # None = unlimited
    project_id: str | None = None
    agent_id: str | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._validate_condition()

    def _validate_condition(self) -> None:
        tt = self.trigger_type
        c = self.condition

        if tt == DegradationTriggerType.low_confidence:
            if "threshold" not in c:
                raise ValueError(
                    "low_confidence trigger requires 'threshold' in condition"
                )
            if not isinstance(c["threshold"], (int, float)):
                raise ValueError("threshold must be a number")
            if not 0.0 <= c["threshold"] <= 1.0:
                raise ValueError("threshold must be between 0.0 and 1.0")

        elif tt == DegradationTriggerType.api_error:
            if "max_errors" not in c:
                raise ValueError(
                    "api_error trigger requires 'max_errors' in condition"
                )
            if not isinstance(c["max_errors"], int) or c["max_errors"] <= 0:
                raise ValueError("max_errors must be a positive integer")
            if "window_minutes" not in c:
                raise ValueError(
                    "api_error trigger requires 'window_minutes' in condition"
                )
            if (
                not isinstance(c["window_minutes"], (int, float))
                or c["window_minutes"] <= 0
            ):
                raise ValueError("window_minutes must be a positive number")

        elif tt == DegradationTriggerType.context_decay:
            if "max_age_hours" not in c:
                raise ValueError(
                    "context_decay trigger requires 'max_age_hours' in condition"
                )
            if (
                not isinstance(c["max_age_hours"], (int, float))
                or c["max_age_hours"] <= 0
            ):
                raise ValueError("max_age_hours must be a positive number")

        elif tt == DegradationTriggerType.budget_breach:
            if "max_tokens" not in c:
                raise ValueError(
                    "budget_breach trigger requires 'max_tokens' in condition"
                )
            if not isinstance(c["max_tokens"], int) or c["max_tokens"] <= 0:
                raise ValueError("max_tokens must be a positive integer")

        elif tt == DegradationTriggerType.cost_breach:
            if "pct_used" not in c:
                raise ValueError(
                    "cost_breach trigger requires 'pct_used' in condition"
                )
            if (
                not isinstance(c["pct_used"], (int, float))
                or c["pct_used"] <= 0
            ):
                raise ValueError("pct_used must be a positive number")


class DegradationPolicy(BaseModel):
    """A degradation policy — condition-driven rule for handling system degradation."""

    id: str = Field(default_factory=_weft_id)
    name: str
    trigger_type: DegradationTriggerType
    condition: dict[str, Any] = Field(default_factory=dict)
    action: DegradationAction
    description: str | None = None
    status: DegradationPolicyStatus = DegradationPolicyStatus.active
    cooldown_minutes: float | None = None
    max_fires: int | None = None
    fire_count: int = 0
    last_fired_at: datetime | None = None
    project_id: str | None = None
    agent_id: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        d = self.model_dump(mode="json")
        d["trigger_type"] = self.trigger_type.value
        d["action"] = self.action.value
        d["status"] = self.status.value
        return d


class ContradictionWarning(BaseModel):
    """A warning that a new memory may contradict an existing one."""

    type: str = "contradiction"
    memory_id: str
    content_preview: str
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_text(self) -> str:
        return f'Warning: This may contradict memory {self.memory_id}: "{self.content_preview}"'


class MemoryRecall(BaseModel):
    """Result from a recall/search operation."""

    memory: Memory
    similarity: float = 0.0

    @property
    def relevance_score(self) -> float:
        """Composite score combining similarity, confidence, and usefulness.

        Weighted: 50% similarity, 30% confidence, 20% usefulness.
        """
        return (
            0.5 * self.similarity
            + 0.3 * self.memory.confidence
            + 0.2 * self.memory.usefulness_score
        )

    def to_dict(self) -> dict[str, Any]:
        d = self.memory.to_dict()
        d["similarity"] = round(self.similarity, 4)
        d["relevance_score"] = round(self.relevance_score, 4)
        return d
