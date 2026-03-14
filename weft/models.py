"""Pydantic models for Weft memory records and relationships."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

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


class MemorySource(str, Enum):
    conversation = "conversation"
    code = "code"
    documentation = "documentation"
    inference = "inference"
    seed = "seed"
    ingest = "ingest"


class MemoryStatus(str, Enum):
    active = "active"
    archived = "archived"
    decayed = "decayed"


class RelationType(str, Enum):
    supersedes = "supersedes"
    related_to = "related_to"
    contradicts = "contradicts"
    derived_from = "derived_from"


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
    status: MemoryStatus = MemoryStatus.active
    pinned: bool = False
    usefulness_score: float = Field(default=0.7, ge=0.0, le=1.0)
    usefulness_count: int = 0
    last_boosted_at: datetime | None = None
    review_after: datetime | None = None

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
    pinned: bool = False
    review_after: datetime | None = None


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


class EpisodeStatus(str, Enum):
    open = "open"
    closed = "closed"


class Episode(BaseModel):
    """A time-bounded grouping of memories into a causal sequence."""

    id: str = Field(default_factory=_weft_id)
    title: str
    summary: str | None = None
    project_id: str | None = None
    agent_id: str | None = None
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    status: EpisodeStatus = EpisodeStatus.open
    token_count: int = 0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

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


class EpisodeWithMemories(BaseModel):
    """An episode with its linked memories in order."""

    episode: Episode
    memories: list[Memory] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.episode.to_dict()
        d["memories"] = [m.to_dict() for m in self.memories]
        d["memory_count"] = len(self.memories)
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
