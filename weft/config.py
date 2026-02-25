"""Three-layer configuration: global → project → environment variables."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


def load_config(project_dir: str | Path | None = None) -> WeftConfig:
    """Load config merging global → project → env vars."""
    # Layer 1: global defaults
    global_path = Path.home() / ".weft" / "config.yaml"
    global_data = _flatten_yaml(_load_yaml(global_path))

    # Layer 2: project overrides
    project_data: dict = {}
    if project_dir:
        project_path = Path(project_dir) / ".weft" / "config.yaml"
        project_data = _flatten_yaml(_load_yaml(project_path))

    # Merge: project overrides global
    merged = {**global_data, **project_data}
    config = WeftConfig(**merged)

    # Layer 3: env var overrides
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
