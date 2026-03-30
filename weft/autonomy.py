"""Autonomy policy models for governing agent action permissions.

The three-tier authority model:
- NEVER: Hard stops — actions the agent must never take autonomously.
- EARNED: Actions that start restricted but can be promoted via calibration.
- ALWAYS: Safe-zone actions the agent can always perform without asking.

Philosophy: conservative defaults, iterate from usage. Everything defaults to
the most restricted tier and migrates upward based on real calibration data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from weft.models import _now, _weft_id


class AutonomyTier(str, Enum):
    never = "never"
    earned = "earned"
    always = "always"


class ActionPolicy(BaseModel):
    """A policy governing whether an agent may perform a specific action."""

    id: str = Field(default_factory=_weft_id)
    action: str  # e.g. "send_slack_message", "create_pr", "deploy"
    tier: AutonomyTier = AutonomyTier.never
    description: str | None = None
    conditions: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    enabled: bool = True
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        d = self.model_dump(mode="json")
        d["tier"] = self.tier.value
        return d


class ActionPolicyCreate(BaseModel):
    """Input model for creating an action policy."""

    action: str
    tier: AutonomyTier = AutonomyTier.never
    description: str | None = None
    conditions: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    enabled: bool = True


class PolicyCalibrationEvent(BaseModel):
    """A record of a tier change or approval/rejection used for calibration."""

    id: str = Field(default_factory=_weft_id)
    policy_id: str
    previous_tier: AutonomyTier
    new_tier: AutonomyTier
    reason: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    created_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        d = self.model_dump(mode="json")
        d["previous_tier"] = self.previous_tier.value
        d["new_tier"] = self.new_tier.value
        return d


class PolicyCalibrationEventCreate(BaseModel):
    """Input model for recording a calibration event."""

    policy_id: str
    previous_tier: AutonomyTier
    new_tier: AutonomyTier
    reason: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
