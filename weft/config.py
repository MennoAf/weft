"""Four-layer configuration: defaults → TOML config file → project YAML → environment variables."""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".weft" / "config.toml"

# Mapping from flat TOML keys to nested WeftConfig paths.
# Keys listed here are the recognised "dotted" TOML keys that
# ``weft config set`` accepts (e.g. "database.url").
_KEY_MAP: dict[str, tuple[str, str]] = {
    "project_name": ("", "project_name"),
    "log_level": ("", "log_level"),
    "database.url": ("database", "url"),
    "database.pool_min_size": ("database", "pool_min_size"),
    "database.pool_max_size": ("database", "pool_max_size"),
    "redis.url": ("redis", "url"),
    "embedding.provider": ("embedding", "provider"),
    "embedding.model": ("embedding", "model"),
    "embedding.dimensions": ("embedding", "dimensions"),
    "embedding.batch_size": ("embedding", "batch_size"),
    "retrieval.default_top_k": ("retrieval", "default_top_k"),
    "retrieval.similarity_threshold": ("retrieval", "similarity_threshold"),
    "retrieval.context_budget_tokens": ("retrieval", "context_budget_tokens"),
    "decay.enabled": ("decay", "enabled"),
    "decay.half_life_days": ("decay", "half_life_days"),
    "decay.floor_score": ("decay", "floor_score"),
}


class DatabaseConfig(BaseModel):
    url: str = "postgresql://weft:weft_local@localhost:5433/weft"
    pool_min_size: int = 2
    pool_max_size: int = 10


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6380"


class EmbeddingConfig(BaseModel):
    provider: str = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"
    dimensions: int = 384
    batch_size: int = 64


class RetrievalConfig(BaseModel):
    default_top_k: int = 10
    similarity_threshold: float = 0.5
    context_budget_tokens: int = 4000


class DecayConfig(BaseModel):
    enabled: bool = True
    half_life_days: int = 30
    floor_score: float = 0.1


class WeftConfig(BaseModel):
    project_name: str = "default"
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)
    log_level: str = "INFO"


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

    # Coerce value to the right type
    coerced: object = _coerce_value(key, value)

    parts = key.split(".", 1)
    if len(parts) == 2:
        section, field = parts
        data.setdefault(section, {})[field] = coerced
    else:
        data[key] = coerced

    _write_toml(data, p)


def _coerce_value(key: str, value: str) -> object:
    """Coerce a string value to the type expected by WeftConfig for *key*."""
    # Build a type lookup from the config models
    type_map: dict[str, type] = {}
    for section_name, section_model in [
        ("database", DatabaseConfig),
        ("redis", RedisConfig),
        ("embedding", EmbeddingConfig),
        ("retrieval", RetrievalConfig),
        ("decay", DecayConfig),
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
    if "database" in data:
        flat["database"] = DatabaseConfig(**data["database"])
    if "redis" in data:
        flat["redis"] = RedisConfig(**data["redis"])
    if "embedding" in data:
        flat["embedding"] = EmbeddingConfig(**data["embedding"])
    if "retrieval" in data:
        flat["retrieval"] = RetrievalConfig(**data["retrieval"])
    if "decay" in data:
        flat["decay"] = DecayConfig(**data["decay"])
    if "logging" in data and "level" in data["logging"]:
        flat["log_level"] = data["logging"]["level"]
    return flat


# --- Main loader ---


def load_config(project_dir: str | Path | None = None) -> WeftConfig:
    """Load config merging: defaults → TOML config file → project YAML → env vars.

    Precedence (highest wins): env vars > project YAML > TOML file > defaults.
    """
    # Layer 1: global YAML defaults (legacy)
    global_path = Path.home() / ".weft" / "config.yaml"
    global_data = _flatten_yaml(_load_yaml(global_path))

    # Layer 2: project YAML overrides
    project_data: dict = {}
    if project_dir:
        project_path = Path(project_dir) / ".weft" / "config.yaml"
        project_data = _flatten_yaml(_load_yaml(project_path))

    # Merge YAML layers: project overrides global
    merged = {**global_data, **project_data}
    config = WeftConfig(**merged)

    # Layer 3: TOML config file (~/.weft/config.toml)
    toml_data = load_config_file()
    if toml_data:
        _apply_toml_to_config(toml_data, config)

    # Layer 4: env var overrides (highest precedence)
    if url := os.environ.get("WEFT_DATABASE_URL"):
        config.database.url = url
    if url := os.environ.get("WEFT_REDIS_URL"):
        config.redis.url = url
    if provider := os.environ.get("WEFT_EMBEDDING_PROVIDER"):
        config.embedding.provider = provider
    if model := os.environ.get("WEFT_EMBEDDING_MODEL"):
        config.embedding.model = model
    if level := os.environ.get("WEFT_LOG_LEVEL"):
        config.log_level = level

    return config
