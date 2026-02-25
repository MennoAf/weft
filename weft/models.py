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


class MemorySource(str, Enum):
    conversation = "conversation"
    code = "code"
    documentation = "documentation"
    inference = "inference"


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        d = self.model_dump(mode="json")
        d["type"] = self.type.value
        d["source"] = self.source.value
        d["status"] = self.status.value
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


class MemoryRecall(BaseModel):
    """Result from a recall/search operation."""

    memory: Memory
    similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = self.memory.to_dict()
        d["similarity"] = round(self.similarity, 4)
        return d
