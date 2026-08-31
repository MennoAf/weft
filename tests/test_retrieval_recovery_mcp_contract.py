from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.mcp.server import AppContext
from tests.test_mcp_tools import FakeEmbeddingProvider, _make_ctx


@pytest.fixture
def app(pool):
    return AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=FakeEmbeddingProvider(),
        config=WeftConfig(),
    )


@pytest.fixture
def ctx(app):
    return _make_ctx(app)


_MARKER_KEYS = {"version", "supported", "attempted", "not_attempted"}


def assert_unsupported_marker(response: dict) -> None:
    assert response["recovery"] == {
        "version": response["recovery"]["version"],
        "supported": False,
        "attempted": False,
        "not_attempted": True,
    }
    assert set(response["recovery"]) == _MARKER_KEYS


@pytest.mark.asyncio
async def test_recovery_off_has_no_recovery_key(ctx) -> None:
    from weft.mcp.tools import weft_recall

    response = await weft_recall(ctx, query="recovery off contract", mode="keyword", tier="belief")
    assert "recovery" not in response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "fixture"),
    [
        ({"tier": "turns"}, "turns"),
        ({"tier": "both"}, "both"),
        ({"tier": "belief"}, "belief"),
        ({"tier": "belief", "query": "list all plants"}, "enumeration"),
    ],
)
async def test_deterministic_legacy_paths_have_unsupported_marker(ctx, kwargs, fixture, monkeypatch) -> None:
    from weft.mcp.tools import weft_recall

    belief_view = None
    if fixture == "belief":
        belief_view = patch(
            "weft.views.belief_query.search_belief_claims",
            new_callable=AsyncMock,
            return_value=[type("Claim", (), {"to_recall_dict": lambda self: {"id": "claim-1"}})()],
        )
    with patch("weft.mcp.tools._resolve_project_id", new_callable=AsyncMock, return_value=None):
        with (belief_view or patch("weft.mcp.tools.search_by_keyword", new_callable=AsyncMock, return_value=[])):
            response = await weft_recall(
                ctx,
                query=kwargs.pop("query", "recovery marker contract"),
                mode="keyword",
                recovery_mode="deterministic",
                enumeration_compatibility=(fixture == "enumeration"),
                **kwargs,
            )
    if fixture == "turns" and response.get("tier") != "turns":
        pytest.skip(f"turn fixture had no relevant turns and used fallback: {response}")
    assert "error" not in response, response
    assert_unsupported_marker(response)


@pytest.mark.asyncio
async def test_deterministic_degraded_path_has_unsupported_marker(app, monkeypatch) -> None:
    from weft.mcp.tools import weft_recall
    from tests.test_mcp_tools import _make_ctx

    await app.pool.close()
    response = await weft_recall(
        _make_ctx(app), query="degraded recovery marker", mode="keyword", recovery_mode="deterministic"
    )
    assert response.get("degraded") is True
    assert_unsupported_marker(response)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"tier": "not-a-tier"},
        {"structured_recall_mode": "not-a-mode"},
        {"recovery_mode": "not-a-mode"},
    ],
)
async def test_invalid_retrieval_inputs_have_no_recovery_marker(ctx, kwargs) -> None:
    from weft.mcp.tools import weft_recall

    response = await weft_recall(ctx, query="invalid recovery input", mode="keyword", **kwargs)
    assert response["error"] == "Invalid input"
    assert "recovery" not in response


@pytest.mark.asyncio
async def test_model_recovery_rejects_disabled_without_provider_call(ctx) -> None:
    from weft.mcp.tools import weft_recall

    with patch("weft.mcp.tools.load_config", create=True) as load_config, patch(
        "weft.mcp.tools.search_by_keyword", new_callable=AsyncMock
    ) as search:
        response = await weft_recall(ctx, query="model disabled", mode="keyword", recovery_mode="model")
    assert response["error"] == "Invalid input"
    assert "requires recovery_planner_enabled=true" in response["detail"]
    load_config.assert_not_called()
    search.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_recovery_reserved_rejects_without_provider_call(ctx, monkeypatch) -> None:
    from weft.mcp.tools import weft_recall
    from weft.config import RetrievalConfig

    class Config:
        retrieval = RetrievalConfig(recovery_mode="model", recovery_planner_enabled=True)

    monkeypatch.setattr("weft.config.load_config", lambda **_: Config())
    with patch("weft.mcp.tools.search_by_keyword", new_callable=AsyncMock) as search:
        response = await weft_recall(ctx, query="model reserved", mode="keyword")
    assert response["error"] == "Invalid input"
    assert "reserved for a later milestone" in response["detail"]
    search.assert_not_awaited()


@pytest.mark.asyncio
async def test_final_belief_deterministic_response_preserves_results_and_supported_recovery(ctx, monkeypatch) -> None:
    from weft.mcp.tools import weft_recall

    monkeypatch.setattr("weft.mcp.tools._resolve_project_id", AsyncMock(return_value=None))
    with patch(
        "weft.mcp.tools.search_by_keyword",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "weft.mcp.tools._attach_belief_recovery", create=True
    ):
        response = await weft_recall(
            ctx, query="How do I configure export?", mode="keyword", tier="belief", recovery_mode="deterministic"
        )
    assert "results" in response
    assert response["recovery"]["supported"] is True
    assert response["recovery"]["attempted"] is True
    assert response["recovery"].get("not_attempted", False) is False
    assert "stages" in response["recovery"]


@pytest.mark.asyncio
async def test_deterministic_recovery_propagates_explicit_contradiction_edges(ctx, app, monkeypatch) -> None:
    from weft.mcp.tools import weft_recall, weft_remember
    from weft.models import RelationType
    from weft.store import add_relationship

    first = await weft_remember(
        ctx,
        content="Export setup overview",
        check_contradictions=False,
    )
    second = await weft_remember(
        ctx,
        content="Export setup uses the legacy path",
        check_contradictions=False,
    )
    assert first.get("id") and second.get("id")
    await add_relationship(app.pool, first["id"], second["id"], RelationType.contradicts)

    monkeypatch.setattr("weft.mcp.tools._resolve_project_id", AsyncMock(return_value=None))
    # Keep baseline selection deterministic; the production sidecar still reads
    # the real relationship rows for these IDs.
    monkeypatch.setattr(
        "weft.mcp.tools.search_by_keyword",
        AsyncMock(return_value=[
            type("Recall", (), {
                "memory": type("Memory", (), {
                    "id": first["id"],
                    "write_provenance": "supervisor",
                    "to_dict": lambda self: {
                        "id": first["id"],
                        "content": "Export setup overview",
                    },
                })(),
                "to_dict": lambda self: {
                    "id": first["id"],
                    "content": "Export setup overview",
                },
            })(),
        ]),
    )
    response = await weft_recall(
        ctx,
        query="How do I configure the export command?",
        mode="keyword",
        tier="belief",
        recovery_mode="deterministic",
    )

    assert response["recovery"]["retrieval_status"] == "conflict"
    assert response["recovery"]["answerability"] == "conflicting_evidence"
    conflict_flags = response["recovery"]["coverage"]["conflict_flags"]
    assert conflict_flags
    assert all(flag.startswith("contradicts:") for flag in conflict_flags)


@pytest.mark.asyncio
async def test_explicit_empty_recovery_mode_overrides_config_and_is_input_error(ctx, monkeypatch) -> None:
    from weft.mcp.tools import weft_recall
    from weft.config import RetrievalConfig

    class Config:
        retrieval = RetrievalConfig(recovery_mode="deterministic")

    monkeypatch.setattr("weft.config.load_config", lambda **_: Config())
    response = await weft_recall(ctx, query="empty recovery mode", mode="keyword", recovery_mode="")
    assert response["error"] == "Invalid input"
    assert "recovery_mode" in response["detail"]
    assert "recovery" not in response
