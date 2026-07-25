from __future__ import annotations

import pytest
from pydantic import ValidationError

from weft.models import MemoryCreate, MemoryType, PreferenceMetadata


def test_preference_metadata_valid_variants() -> None:
    for polarity in ("positive", "negative", "constraint", "avoidance"):
        metadata = PreferenceMetadata(
            polarity=polarity,
            strength="hard" if polarity in {"constraint", "avoidance"} else "soft",
            subject="commute",
            value="audio-only",
            context=["workday"],
        )
        assert metadata.polarity == polarity
        assert metadata.schema_version == 1


def test_preference_metadata_rejects_unknown_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PreferenceMetadata(polarity="unknown", strength="soft")
    with pytest.raises(ValidationError):
        PreferenceMetadata(polarity="positive", strength="soft", invented=True)
    with pytest.raises(ValidationError):
        PreferenceMetadata(polarity="positive", strength="soft", subject=" ")


def test_non_preference_rejects_preference_metadata() -> None:
    metadata = PreferenceMetadata(polarity="positive", strength="soft")
    with pytest.raises(ValidationError, match="requires type='preference'"):
        MemoryCreate(
            type=MemoryType.fact,
            content="This is a fact.",
            preference_metadata=metadata,
        )


def test_preference_metadata_does_not_change_embed_text() -> None:
    from weft.store import embed_text_for_memory

    assert embed_text_for_memory("likes tea", ["drinks"]) == "likes tea drinks"
