"""Four-layer configuration: defaults → TOML config file → project YAML → environment variables."""

from __future__ import annotations

import logging
import os
import re
import tomllib
from enum import Enum
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file (no-op if missing). Must happen before any os.environ reads.
# Search order: cwd .env (default), then ~/.weft/.env as fallback.
# This ensures the CLI works when run from outside the Weft source tree.
if not os.environ.get("WEFT_TESTING"):
    load_dotenv()
    load_dotenv(Path.home() / ".weft" / ".env")

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".weft" / "config.toml"

# Mapping from flat TOML keys to nested WeftConfig paths.
# Keys listed here are the recognised "dotted" TOML keys that
# ``weft config set`` accepts (e.g. "database.url").
_KEY_MAP: dict[str, tuple[str, str]] = {
    "env": ("", "env"),
    "project_name": ("", "project_name"),
    "api_key": ("", "api_key"),
    "log_level": ("", "log_level"),
    "database.url": ("database", "url"),
    "database.ca_cert_file": ("database", "ca_cert_file"),
    "database.pool_min_size": ("database", "pool_min_size"),
    "database.pool_max_size": ("database", "pool_max_size"),
    "database.statement_cache_size": ("database", "statement_cache_size"),
    "database.command_timeout": ("database", "command_timeout"),
    "database.acquire_timeout": ("database", "acquire_timeout"),
    "database.prime_timeout": ("database", "prime_timeout"),
    "redis.url": ("redis", "url"),
    "embedding.provider": ("embedding", "provider"),
    "embedding.model": ("embedding", "model"),
    "embedding.dimensions": ("embedding", "dimensions"),
    "embedding.batch_size": ("embedding", "batch_size"),
    "text_generation.provider": ("text_generation", "provider"),
    "text_generation.models": ("text_generation", "models"),
    "retrieval.default_top_k": ("retrieval", "default_top_k"),
    "retrieval.similarity_threshold": ("retrieval", "similarity_threshold"),
    "retrieval.context_budget_tokens": ("retrieval", "context_budget_tokens"),
    "retrieval.contradiction_check_on_write": ("retrieval", "contradiction_check_on_write"),
    "retrieval.recovery_mode": ("retrieval", "recovery_mode"),
    "retrieval.recovery_planner_enabled": ("retrieval", "recovery_planner_enabled"),
    "retrieval.facet_boost": ("retrieval", "facet_boost"),
    "retrieval.facet_auto_merge_threshold": ("retrieval", "facet_auto_merge_threshold"),
    "retrieval.facet_candidate_threshold": ("retrieval", "facet_candidate_threshold"),
    "decay.enabled": ("decay", "enabled"),
    "decay.half_life_days": ("decay", "half_life_days"),
    "decay.floor_score": ("decay", "floor_score"),
    "alert.poll_interval": ("alert", "poll_interval"),
    "alert.batch_size": ("alert", "batch_size"),
    "slack_sync.interval": ("slack_sync", "interval"),
    "slack_sync.smart_ingest_max_per_sync": ("slack_sync", "smart_ingest_max_per_sync"),
    "daily_brief.time": ("daily_brief", "time"),
    "daily_brief.timezone": ("daily_brief", "timezone"),
    "daily_brief.channel": ("daily_brief", "channel"),
    "daily_brief.calendar_id": ("daily_brief", "calendar_id"),
    "primer.disabled_sections": ("primer", "disabled_sections"),
}


class WeftEnv(str, Enum):
    local = "local"
    production = "production"


class MigrationMode(str, Enum):
    """Startup handling for database schema migrations."""

    apply = "apply"
    verify = "verify"


class DatabaseConfig(BaseModel):
    url: str = "postgresql://weft:weft_local@localhost:5433/weft"
    # Optional PEM-encoded CA certificate for providers whose root is not in
    # the runtime image trust store (for example Supabase Root 2021 CA).
    # When unset, create_pool uses the operating system trust store.
    ca_cert: str | None = None
    # Preferred deployment-safe form: path to a PEM bundle, avoiding certificate
    # contents in shell history and process listings.
    ca_cert_file: Path | None = None
    pool_min_size: int = 2
    pool_max_size: int = 20
    statement_cache_size: int | None = None  # Set to 0 for pgbouncer/Supabase pooler
    # Per-query ceiling passed to asyncpg. A stuck query (lock, seq-scan, pooler
    # latency) is cancelled instead of hanging forever and holding its connection.
    command_timeout: float | None = 30.0
    # Max wait for a free pooled connection in acquire(). When the pool is
    # drained, callers fail fast with a clear error instead of blocking
    # indefinitely — the difference between a visible error and a silent hang.
    acquire_timeout: float | None = 10.0
    # Wall-clock budget for the complete parallel session-primer build. Per-query
    # timeouts alone do not bound the aggregate fan-out latency.
    prime_timeout: float = 60.0


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6380"


class EmbeddingConfig(BaseModel):
    """Embedding profile; local FastEmbed is the zero-key default."""

    provider: str = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"
    # Keep the existing pgvector schema contract; FastEmbed's native 384-vector
    # output is padded to this width by the provider.
    dimensions: int = 768
    batch_size: int = 64


class RetrievalConfig(BaseModel):
    default_top_k: int = 10
    similarity_threshold: float = 0.5
    context_budget_tokens: int = 4000
    contradiction_check_on_write: bool = True
    recovery_mode: Literal["off", "deterministic", "model"] = "off"
    recovery_planner_enabled: bool = False
    cross_project_search: bool = True
    cross_project_limit: int = 3
    # Facet-based recall (Memory v2 Phase 1). Single source of truth for the
    # three facet magic numbers; the module-level constants in store.py /
    # consolidation.py derive from these defaults so behavior is identical.
    facet_boost: float = 1.15  # post-query rank multiplier for facet-overlap beliefs
    facet_auto_merge_threshold: float = 0.85  # cross-project sim>=: append a facet
    facet_candidate_threshold: float = 0.6  # cross-project sim in [this, auto): review


class DecayConfig(BaseModel):
    enabled: bool = True
    half_life_days: int = 30
    floor_score: float = 0.1


class AlertConfig(BaseModel):
    poll_interval: int = 60  # seconds between scheduler polls
    batch_size: int = 50  # max alerts per poll cycle


class SlackSyncConfig(BaseModel):
    interval: int = 1800  # seconds between auto-sync cycles (default 30min)
    smart_ingest: bool = True  # enable LLM-powered smart ingestion pipeline
    smart_ingest_max_per_sync: int = 50  # max messages to smart-ingest per sync cycle
    react_on_ingest: bool = True  # add emoji reaction after successful ingest
    ingest_reaction_emoji: str = "brain"  # emoji name (without colons)
    skip_prefixes: list[str] = ["/checkin"]  # skip messages starting with these (already captured elsewhere)


class DailyBriefConfig(BaseModel):
    time: str = "08:00"  # HH:MM wall-clock time for delivery
    timezone: str = "America/New_York"  # IANA timezone for brief schedule
    channel: str = ""  # Slack channel ID or name; empty = skip Slack delivery
    channel_type: str = "slack"  # Outbound connector: slack, discord, or none
    calendar_id: str = "primary"  # Google Calendar ID to query
    # Map project_id (the slug used in handoff memories) to an absolute repo
    # path. Used by the "active projects" section to count git commits per
    # project in the activity window. Projects not in this map still surface
    # via handoff memories but contribute zero commit signal.
    project_repos: dict[str, str] = Field(default_factory=dict)
    # Top-N active projects to surface. Activity = commits + handoffs in window.
    active_projects_top_n: int = 3
    active_projects_window_hours: int = 24


class PrimerConfig(BaseModel):
    disabled_sections: list[str] = Field(
        default_factory=list,
        description="Section names to suppress from the primer output "
        "(e.g. ['triggers', 'cost']). Data is preserved; only display is affected.",
    )


class DiscordConfig(BaseModel):
    """Owner-mapping config for Discord slash commands.

    Single-user mode: set WEFT_DISCORD_OWNER_DISCORD_ID to the Discord
    snowflake (user.id) of the owner and WEFT_DISCORD_OWNER_WEFT_USER_ID
    to the corresponding Weft user UUID.

    When a slash command fires, the handler resolves the interaction's
    user.id against owner_discord_id. On match, owner_weft_user_id is
    injected into the RLS context. On mismatch, an ephemeral error is
    returned.

    TODO(multi-user): replace the two scalar fields with a dict[str, str]
    mapping (discord_snowflake → weft_uuid) loaded from env or config
    file, so multiple Discord accounts can be bound to different Weft
    users. The resolver below in commands.py has a single lookup point
    that must change at that migration.
    """

    owner_discord_id: str | None = None
    """Discord snowflake (user.id string) of the configured owner. None = unconfigured."""

    owner_weft_user_id: str | None = None
    """Weft user UUID to inject into RLS when the owner fires a command. None = unconfigured."""


class QuarantineReviewConfig(BaseModel):
    """Layer 3.5 — periodic LLM review of agent-provenance writes."""

    enabled: bool = True
    interval: int = 21600  # 6h between cycles by default
    limit: int = 100  # max memories per cycle
    concurrency: int = 4  # bounded concurrent Anthropic calls
    model: str = "claude-haiku-4-5-20251001"


class TextGenerationConfig(BaseModel):
    """Provider and logical-role model selection for text generation.

    The role map is deliberately abstract: feature code asks for a role such
    as ``ingest_classifier`` rather than assuming a provider-specific model.
    """

    provider: str = "anthropic"
    models: dict[str, str] = Field(default_factory=dict)


# Cost-enforcement config types live here (not in weft.cost_enforcement)
# because weft.db.connection imports WeftConfig at module-load time, which
# is loaded by every cost_enforcement dependency — a circular cycle if
# the types lived alongside their consumer module. Keeping the dataclasses
# co-located with the rest of the config also matches the pattern for
# AlertConfig, SlackSyncConfig, etc.


class CostThresholdLiteral(str, Enum):
    """String enum for autonomy tier values — duplicated locally to avoid
    importing weft.autonomy at config-load time. Values must match
    AutonomyTier exactly. Kept in sync via the test
    test_cost_threshold_tier_enum_matches_autonomy_tier."""

    never = "never"
    earned = "earned"
    always = "always"


class CostThreshold(BaseModel):
    """One band on the cost-enforcement ladder. See weft.cost_enforcement."""

    pct_used: float
    demote_actions: list[str] = Field(default_factory=list)
    demote_to: CostThresholdLiteral = CostThresholdLiteral.earned
    feed_degradation: bool = True
    notify: bool = True


class CostEnforcementConfig(BaseModel):
    """Runtime config for the cost enforcement loop."""

    enabled: bool = False
    interval_seconds: int = 300
    daily_limit_usd: float = 50.0
    thresholds: list[CostThreshold] = Field(
        default_factory=lambda: [
            CostThreshold(pct_used=75.0, notify=True),
            CostThreshold(
                pct_used=90.0,
                demote_actions=["*"],
                demote_to=CostThresholdLiteral.earned,
            ),
            CostThreshold(
                pct_used=100.0,
                demote_actions=["*"],
                demote_to=CostThresholdLiteral.never,
            ),
        ]
    )


class AlertCooldownConfig(BaseModel):
    """Per-AlertType cooldown windows in minutes.

    Replaces the V1 single ``dedup_hours=24`` constant. Producers call
    ``alert_dedup.should_fire(alert_type, dedup_key, cooldown_minutes=...)``
    where the cooldown comes from this config. Sensible defaults: short
    for things that matter immediately (contradictions), long for things
    that recur on weekly cadence (consolidation, count thresholds).

    Use ``minutes_for(alert_type)`` to look up a cooldown with the
    ``default_minutes`` fallback so unrecognized types still have a
    sane window.
    """

    default_minutes: float = 1440.0  # 24h fallback for unrecognized types
    by_type: dict[str, float] = Field(
        default_factory=lambda: {
            # Loom awareness — per-task/project/epic dedup means we can be
            # more aggressive than the old single-bucket 24h.
            "loom_stale_claim": 720.0,         # 12h per task
            "loom_blocked_pile_up": 240.0,     # 4h per project
            "loom_epic_ready": 720.0,          # 12h per epic
            # Memory hygiene — these recur on weekly+ cadence.
            "stale_decision": 10080.0,         # 1 week per memory
            "memory_consolidation_overdue": 1440.0,  # 24h (singleton)
            "memory_count_threshold": 4320.0,  # 3 days (singleton)
            # Contradictions matter immediately — short cooldown.
            "memory_contradiction": 60.0,      # 1h per memory
            # Check-in derived alerts.
            "check_in_low_mood": 1440.0,
            "check_in_low_sleep": 1440.0,
            "check_in_declining_trend": 4320.0,
        }
    )

    def minutes_for(self, alert_type: object) -> float:
        """Look up cooldown for an AlertType (or its .value), with fallback."""
        key = getattr(alert_type, "value", alert_type)
        return self.by_type.get(str(key), self.default_minutes)


DEFAULT_PROJECT_NAME = "default"


def configured_project_name(value: object) -> str | None:
    """Return a usable configured scope name, excluding the default sentinel."""
    if not isinstance(value, str):
        return None
    name = value.strip()
    if not name or name.casefold() == DEFAULT_PROJECT_NAME:
        return None
    return name


class WeftConfig(BaseModel):
    env: WeftEnv = WeftEnv.local
    migration_mode: MigrationMode = MigrationMode.apply
    project_name: str = DEFAULT_PROJECT_NAME
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)
    alert: AlertConfig = Field(default_factory=AlertConfig)
    slack_sync: SlackSyncConfig = Field(default_factory=SlackSyncConfig)
    daily_brief: DailyBriefConfig = Field(default_factory=DailyBriefConfig)
    quarantine_review: QuarantineReviewConfig = Field(
        default_factory=QuarantineReviewConfig
    )
    text_generation: TextGenerationConfig = Field(default_factory=TextGenerationConfig)
    cost_enforcement: CostEnforcementConfig = Field(
        default_factory=CostEnforcementConfig
    )
    alert_cooldowns: AlertCooldownConfig = Field(
        default_factory=AlertCooldownConfig
    )
    primer: PrimerConfig = Field(default_factory=PrimerConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    api_key: str | None = None
    supabase_url: str | None = None
    # Supabase anon (publishable) key — required for the consent page to
    # boot the Supabase JS SDK and call ``supabase.auth.oauth.*``. Public
    # by design; safe to embed in HTML.
    supabase_anon_key: str | None = None
    # Supabase Management API token (account-scoped) used ONLY to auto-restore a
    # paused project on a failed connection (see weft.db.connection.create_pool).
    # SECURITY: this is an ACCOUNT-WIDE token — it can restore/modify/delete any
    # project on the account, not just this one. Safe for single-user self-host
    # (you own the account); do NOT set it in a hosted multi-tenant deployment.
    # When unset, auto-restore is disabled and a paused project yields a clear
    # actionable error instead of a raw socket failure.
    supabase_access_token: str | None = None
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # OAuth 2.1 authorization-server config (Phase 1 scaffold).
    # All fields below are dormant unless ``oauth_enabled`` is True; when
    # disabled the server behaves byte-identically to the known-good
    # ``c965d49`` API-key-only snapshot.
    # ------------------------------------------------------------------
    oauth_enabled: bool = False
    oauth_issuer: str = ""
    oauth_jwt_private_key_pem: str | None = None
    # Phase 4c — optional overlap key during rotation. When set, both
    # keys are exposed in JWKS and the verifier accepts either. The
    # primary PEM still signs new tokens until rotation is complete.
    oauth_jwt_private_key_pem_next: str | None = None
    oauth_jwt_kid: str = ""  # derived at load time from the public key
    oauth_session_secret: str = ""
    oauth_sole_user_sub: str | None = None
    oauth_access_ttl_s: int = 3600
    oauth_refresh_ttl_s: int = 2_592_000
    # Phase 3: Supabase provider to pass to ``/auth/v1/authorize?provider=``.
    # Default ``email`` uses magic-link auth which is available on any
    # Supabase project with email auth enabled. For projects configured
    # with a third-party provider (github, google, …), override via
    # ``WEFT_SUPABASE_AUTH_PROVIDER``.
    supabase_auth_provider: str = "email"

    @property
    def is_production(self) -> bool:
        return self.env == WeftEnv.production


# --- TOML config file helpers ---


def load_config_file(path: Path | None = None) -> dict:
    """Read the TOML config file and return its contents as a dict.

    Returns an empty dict if the file does not exist or is invalid.
    """
    p = path or CONFIG_PATH
    if not p.exists():
        return {}
    try:
        with open(p, "rb") as f:
            return tomllib.load(f)
    except Exception:
        logger.warning("Failed to parse TOML config at %s", p)
        return {}


def _toml_escape(value: str) -> str:
    """Escape a string value for TOML output."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_toml_value(value: object) -> str:
    """Format a Python value as a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    # Default: treat as string
    return f'"{_toml_escape(str(value))}"'


def _write_toml(data: dict, path: Path) -> None:
    """Write a (potentially nested) dict as TOML to *path*.

    Only supports one level of nesting (tables with scalar values),
    which is all that WeftConfig needs.
    """
    lines: list[str] = []

    # Top-level scalar keys first
    for key, value in data.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {_format_toml_value(value)}")

    # Then table sections
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"\n[{key}]")
            for sub_key, sub_value in value.items():
                lines.append(f"{sub_key} = {_format_toml_value(sub_value)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def save_config_value(key: str, value: str, path: Path | None = None) -> None:
    """Read existing TOML, update a single key, and write back.

    *key* can be a dotted path like ``database.url``.
    The *value* string is coerced to the appropriate Python type based on the
    WeftConfig field definition.
    """
    p = path or CONFIG_PATH
    data = load_config_file(p)
    data = config_data_with_updates(data, {key: value})
    _write_toml(data, p)


def config_data_with_updates(data: dict, updates: dict[str, str]) -> dict:
    """Return *data* with validated/coerced dotted-key updates applied.

    This is deliberately side-effect free so callers can produce a plan before
    writing.  Unknown keys are rejected instead of being silently persisted.
    """
    result = dict(data)
    for key, value in updates.items():
        if key not in _KEY_MAP and not key.startswith("text_generation.models."):
            raise ValueError(f"Unknown config key: {key}")
        coerced: object = _coerce_value(key, value)
        parts = key.split(".", 1)
        if len(parts) == 2:
            section, field = parts
            section_data = dict(result.get(section, {}))
            section_data[field] = coerced
            result[section] = section_data
        else:
            result[key] = coerced
    return result


def initialize_config(path: Path | None = None) -> bool:
    """Create the minimal local config once; return whether it was created.

    Existing files are never opened for writing.  Credentials, infrastructure,
    migrations, and provider setup are intentionally outside this helper.
    """
    p = path or CONFIG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with p.open("x", encoding="utf-8") as config_file:
            config_file.write(
                '# Weft configuration; use ``weft config set`` for explicit changes.\n'
                'project_name = "default"\n'
            )
    except FileExistsError:
        return False
    return True


def write_config_data(data: dict, path: Path | None = None) -> None:
    """Write already validated config data to *path* (the only write seam)."""
    _write_toml(data, path or CONFIG_PATH)


def _coerce_value(key: str, value: str) -> object:
    """Coerce a string value to the type expected by WeftConfig for *key*."""
    # Build a type lookup from the config models
    type_map: dict[str, type] = {}
    for section_name, section_model in [
        ("database", DatabaseConfig),
        ("redis", RedisConfig),
        ("embedding", EmbeddingConfig),
        ("text_generation", TextGenerationConfig),
        ("retrieval", RetrievalConfig),
        ("decay", DecayConfig),
        ("alert", AlertConfig),
        ("slack_sync", SlackSyncConfig),
        ("daily_brief", DailyBriefConfig),
        ("quarantine_review", QuarantineReviewConfig),
        ("cost_enforcement", CostEnforcementConfig),
        ("alert_cooldowns", AlertCooldownConfig),
        ("primer", PrimerConfig),
    ]:
        for field_name, field_info in section_model.model_fields.items():
            ftype = field_info.annotation
            type_map[f"{section_name}.{field_name}"] = ftype  # type: ignore[assignment]

    # Top-level fields
    for field_name, field_info in WeftConfig.model_fields.items():
        if field_info.annotation in (str, int, float, bool):
            type_map[field_name] = field_info.annotation  # type: ignore[assignment]

    expected = type_map.get(key, str)
    if expected is bool:
        return value.lower() in ("true", "1", "yes")
    if expected is int:
        return int(value)
    if expected is float:
        return float(value)
    return value


def _apply_toml_to_config(data: dict, config: WeftConfig) -> None:
    """Apply parsed TOML data onto an existing WeftConfig instance."""
    if "project_name" in data:
        config.project_name = str(data["project_name"])
    if "log_level" in data:
        config.log_level = str(data["log_level"])
    if "database" in data and isinstance(data["database"], dict):
        for k, v in data["database"].items():
            if hasattr(config.database, k):
                setattr(config.database, k, v)
    if "redis" in data and isinstance(data["redis"], dict):
        for k, v in data["redis"].items():
            if hasattr(config.redis, k):
                setattr(config.redis, k, v)
    if "embedding" in data and isinstance(data["embedding"], dict):
        for k, v in data["embedding"].items():
            if hasattr(config.embedding, k):
                setattr(config.embedding, k, v)
    if "retrieval" in data and isinstance(data["retrieval"], dict):
        for k, v in data["retrieval"].items():
            if hasattr(config.retrieval, k):
                setattr(config.retrieval, k, v)
    if "decay" in data and isinstance(data["decay"], dict):
        for k, v in data["decay"].items():
            if hasattr(config.decay, k):
                setattr(config.decay, k, v)
    if "alert" in data and isinstance(data["alert"], dict):
        for k, v in data["alert"].items():
            if hasattr(config.alert, k):
                setattr(config.alert, k, v)
    if "slack_sync" in data and isinstance(data["slack_sync"], dict):
        for k, v in data["slack_sync"].items():
            if hasattr(config.slack_sync, k):
                setattr(config.slack_sync, k, v)
    if "daily_brief" in data and isinstance(data["daily_brief"], dict):
        for k, v in data["daily_brief"].items():
            if hasattr(config.daily_brief, k):
                setattr(config.daily_brief, k, v)
    if "quarantine_review" in data and isinstance(data["quarantine_review"], dict):
        for k, v in data["quarantine_review"].items():
            if hasattr(config.quarantine_review, k):
                setattr(config.quarantine_review, k, v)
    if "text_generation" in data and isinstance(data["text_generation"], dict):
        for k, v in data["text_generation"].items():
            if hasattr(config.text_generation, k):
                setattr(config.text_generation, k, v)
    if "cost_enforcement" in data and isinstance(data["cost_enforcement"], dict):
        # ``thresholds`` is a list of structured dicts; everything else is scalar.
        for k, v in data["cost_enforcement"].items():
            if k == "thresholds" and isinstance(v, list):
                config.cost_enforcement.thresholds = [
                    CostThreshold(**t) if isinstance(t, dict) else t for t in v
                ]
            elif hasattr(config.cost_enforcement, k):
                setattr(config.cost_enforcement, k, v)
    if "alert_cooldowns" in data and isinstance(data["alert_cooldowns"], dict):
        for k, v in data["alert_cooldowns"].items():
            if k == "by_type" and isinstance(v, dict):
                # Merge user overrides into defaults rather than replacing — TOML
                # users almost always want to tune one type, not redeclare all.
                merged = dict(config.alert_cooldowns.by_type)
                merged.update({str(kk): float(vv) for kk, vv in v.items()})
                config.alert_cooldowns.by_type = merged
            elif hasattr(config.alert_cooldowns, k):
                setattr(config.alert_cooldowns, k, v)
    if "primer" in data and isinstance(data["primer"], dict):
        for k, v in data["primer"].items():
            if hasattr(config.primer, k):
                setattr(config.primer, k, v)


# --- YAML helpers (for project-level config) ---


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _flatten_yaml(data: dict) -> dict:
    """Convert nested YAML structure to flat WeftConfig fields."""
    flat: dict = {}
    if "weft" in data:
        weft = data["weft"]
        if "project_name" in weft:
            flat["project_name"] = weft["project_name"]
    # Keep nested YAML sections partial. The final config merge supplies
    # omitted fields from lower-precedence layers and model defaults.
    for section in (
        "database", "redis", "embedding", "retrieval", "decay", "alert",
        "slack_sync", "daily_brief", "quarantine_review", "primer",
    ):
        value = data.get(section)
        if isinstance(value, dict):
            flat[section] = value
    if isinstance(data.get("cost_enforcement"), dict):
        flat["cost_enforcement"] = data["cost_enforcement"]
    if isinstance(data.get("alert_cooldowns"), dict):
        flat["alert_cooldowns"] = data["alert_cooldowns"]
    if "logging" in data and "level" in data["logging"]:
        flat["log_level"] = data["logging"]["level"]
    return flat


# --- DSN helpers ---


def _encode_dsn_password(dsn: str) -> str:
    """URL-encode the password portion of a PostgreSQL DSN.

    Handles passwords with special characters (/, :, ;, etc.) that break
    standard URL parsing. Only modifies the password; leaves the rest intact.
    """
    m = re.match(r"^(REDACTED]+:)(.+)(@.+)$", dsn)
    if not m:
        return dsn
    prefix, password, suffix = m.groups()
    # Already encoded if it contains %XX sequences
    if "%" in password:
        return dsn
    encoded = quote(password, safe="")
    return f"{prefix}{encoded}{suffix}"


# --- Main loader ---


def _merge_config_data(config: WeftConfig, data: dict) -> WeftConfig:
    """Apply one partial config layer without discarding nested defaults."""
    def merge_mapping(base: dict, override: dict) -> dict:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, BaseModel):
                value = value.model_dump(mode="python")
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge_mapping(merged[key], value)
            else:
                merged[key] = value
        return merged

    return WeftConfig(**merge_mapping(config.model_dump(mode="python"), data))


def load_config(project_dir: str | Path | None = None) -> WeftConfig:
    """Load config merging: defaults → TOML config file → project YAML → env vars.

    Precedence (highest wins): env vars > project YAML > TOML file > defaults.
    """
    # Layer 1: defaults, then user-wide TOML.
    config = WeftConfig()
    toml_data = load_config_file()
    if toml_data:
        _apply_toml_to_config(toml_data, config)

    # Layer 2: global YAML defaults (legacy), then project YAML overrides.
    global_path = Path.home() / ".weft" / "config.yaml"
    global_data = _flatten_yaml(_load_yaml(global_path))
    if global_data:
        config = _merge_config_data(config, global_data)

    project_data: dict = {}
    if project_dir:
        project_path = Path(project_dir) / ".weft" / "config.yaml"
        project_data = _flatten_yaml(_load_yaml(project_path))
    if project_data:
        config = _merge_config_data(config, project_data)

    # Layer 3: env var overrides (highest precedence)
    # DATABASE_URL is the standard convention (Fly.io, Supabase, etc.)
    if url := os.environ.get("WEFT_DATABASE_URL") or os.environ.get("DATABASE_URL"):
        config.database.url = _encode_dsn_password(url)
    if ca_cert := os.environ.get("WEFT_DATABASE_CA_CERT"):
        config.database.ca_cert = ca_cert
    if ca_cert_file := os.environ.get("WEFT_DATABASE_CA_CERT_FILE"):
        config.database.ca_cert_file = Path(ca_cert_file).expanduser()
    # Pool sizing and timeouts are env-overridable so prod can be tuned without a
    # redeploy (e.g. shrink pool_max_size below the Supabase pooler's ceiling).
    if pool_max := os.environ.get("WEFT_DB_POOL_MAX_SIZE"):
        config.database.pool_max_size = int(pool_max)
    if pool_min := os.environ.get("WEFT_DB_POOL_MIN_SIZE"):
        config.database.pool_min_size = int(pool_min)
    if cmd_timeout := os.environ.get("WEFT_DB_COMMAND_TIMEOUT"):
        config.database.command_timeout = float(cmd_timeout)
    if acq_timeout := os.environ.get("WEFT_DB_ACQUIRE_TIMEOUT"):
        config.database.acquire_timeout = float(acq_timeout)
    if prime_timeout := os.environ.get("WEFT_DB_PRIME_TIMEOUT"):
        config.database.prime_timeout = float(prime_timeout)
    if "WEFT_REDIS_URL" in os.environ:
        config.redis.url = os.environ["WEFT_REDIS_URL"]
    if provider := os.environ.get("WEFT_EMBEDDING_PROVIDER"):
        config.embedding.provider = provider
    if model := os.environ.get("WEFT_EMBEDDING_MODEL"):
        config.embedding.model = model
    if provider := os.environ.get("WEFT_TEXT_PROVIDER"):
        config.text_generation.provider = provider
    for role in (
        "ingest_classifier",
        "codebase_summary",
        "codebase_architecture",
        "quarantine_review",
        "belief_detector",
        "topic_synthesis",
        "replay_aggregate",
    ):
        if model := os.environ.get(f"WEFT_TEXT_MODEL_{role.upper()}"):
            config.text_generation.models[role] = model
    if recovery_mode := os.environ.get("WEFT_RETRIEVAL_RECOVERY_MODE"):
        config.retrieval = RetrievalConfig.model_validate(
            {
                **config.retrieval.model_dump(mode="python"),
                "recovery_mode": recovery_mode.strip(),
            }
        )
    if recovery_planner := os.environ.get("WEFT_RETRIEVAL_RECOVERY_PLANNER_ENABLED"):
        config.retrieval.recovery_planner_enabled = recovery_planner.lower() in ("1", "true", "yes")
    if level := os.environ.get("WEFT_LOG_LEVEL"):
        config.log_level = level
    if env := os.environ.get("WEFT_ENV"):
        config.env = WeftEnv(env)
    migration_mode = os.environ.get("WEFT_MIGRATION_MODE")
    if migration_mode:
        config.migration_mode = MigrationMode(migration_mode)
    if api_key := os.environ.get("WEFT_API_KEY"):
        config.api_key = api_key
    if supabase_url := os.environ.get("SUPABASE_URL"):
        config.supabase_url = supabase_url.rstrip("/")
    if supabase_anon_key := os.environ.get("SUPABASE_ANON_KEY"):
        config.supabase_anon_key = supabase_anon_key.strip()
    if supabase_token := os.environ.get("SUPABASE_ACCESS_TOKEN"):
        config.supabase_access_token = supabase_token.strip()
    if poll_interval := os.environ.get("WEFT_ALERT_POLL_INTERVAL"):
        config.alert.poll_interval = int(poll_interval)
    if batch_size := os.environ.get("WEFT_ALERT_BATCH_SIZE"):
        config.alert.batch_size = int(batch_size)
    if sync_interval := os.environ.get("WEFT_SLACK_SYNC_INTERVAL"):
        config.slack_sync.interval = int(sync_interval)
    if quarantine_enabled := os.environ.get("WEFT_QUARANTINE_REVIEW_ENABLED"):
        config.quarantine_review.enabled = quarantine_enabled.lower() in ("1", "true", "yes")
    if brief_time := os.environ.get("WEFT_DAILY_BRIEF_TIME"):
        config.daily_brief.time = brief_time.strip()
    if brief_tz := os.environ.get("WEFT_DAILY_BRIEF_TZ"):
        config.daily_brief.timezone = brief_tz.strip()
    if brief_channel := os.environ.get("WEFT_DAILY_BRIEF_CHANNEL"):
        config.daily_brief.channel = brief_channel.strip()
    if brief_channel_type := os.environ.get("WEFT_DAILY_BRIEF_CHANNEL_TYPE"):
        config.daily_brief.channel_type = brief_channel_type.strip()

    if config.env is WeftEnv.production and not migration_mode:
        raise ValueError(
            "WEFT_MIGRATION_MODE must be explicitly set in production; "
            "use 'verify' for the restricted runtime or 'apply' only for a "
            "deliberate owner-managed migration process"
        )

    # OAuth 2.1 authz-server env vars (optional unless WEFT_OAUTH_ENABLED=1).
    # When disabled, every field below stays at its default and the OAuth
    # module is never instantiated.
    if oauth_flag := os.environ.get("WEFT_OAUTH_ENABLED"):
        config.oauth_enabled = oauth_flag.lower() in ("1", "true", "yes")
    if oauth_issuer := os.environ.get("OAUTH_ISSUER"):
        config.oauth_issuer = oauth_issuer.rstrip("/")
    if oauth_pem := os.environ.get("OAUTH_JWT_PRIVATE_KEY_PEM"):
        config.oauth_jwt_private_key_pem = oauth_pem
    if oauth_pem_next := os.environ.get("OAUTH_JWT_PRIVATE_KEY_PEM_NEXT"):
        config.oauth_jwt_private_key_pem_next = oauth_pem_next
    if oauth_session := os.environ.get("OAUTH_SESSION_SECRET"):
        config.oauth_session_secret = oauth_session
    if oauth_sole := os.environ.get("OAUTH_SOLE_USER_SUB"):
        config.oauth_sole_user_sub = oauth_sole
    if oauth_access_ttl := os.environ.get("OAUTH_ACCESS_TTL_S"):
        try:
            config.oauth_access_ttl_s = int(oauth_access_ttl)
        except ValueError:
            logger.warning("Invalid OAUTH_ACCESS_TTL_S=%r; using default", oauth_access_ttl)
    if oauth_refresh_ttl := os.environ.get("OAUTH_REFRESH_TTL_S"):
        try:
            config.oauth_refresh_ttl_s = int(oauth_refresh_ttl)
        except ValueError:
            logger.warning("Invalid OAUTH_REFRESH_TTL_S=%r; using default", oauth_refresh_ttl)
    if provider := os.environ.get("WEFT_SUPABASE_AUTH_PROVIDER"):
        config.supabase_auth_provider = provider.strip()

    # Discord owner mapping (single-user mode).
    if discord_owner_id := os.environ.get("WEFT_DISCORD_OWNER_DISCORD_ID"):
        config.discord.owner_discord_id = discord_owner_id.strip()
    if discord_owner_weft_id := os.environ.get("WEFT_DISCORD_OWNER_WEFT_USER_ID"):
        config.discord.owner_weft_user_id = discord_owner_weft_id.strip()

    # The legacy ``oauth_jwt_*`` fields exist for backwards compatibility
    # with the prior Weft-as-OAuth-server deployment. In the new
    # Supabase-as-OAuth-server architecture nothing reads them.
    return config
