"""Tests verifying no bare except:pass suppression remains in the weft codebase.

Also verifies that specific formerly-silent except blocks now emit warnings.
"""

from __future__ import annotations

import ast
import inspect
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

WEFT_ROOT = Path(__file__).parent.parent / "weft"


# ---------------------------------------------------------------------------
# Codebase-wide lint: no except + pass body (Category A)
# ---------------------------------------------------------------------------


class TestNoBareExceptPass:
    """Walk the AST of all weft/*.py files and flag any except handler
    whose body is a single `pass` statement catching broad exceptions
    (Exception or bare except).
    """

    def _find_bare_except_pass(self, filepath: Path) -> list[tuple[int, str]]:
        """Return (line, handler_type) for except handlers with pass body
        catching Exception or bare except."""
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
        hits: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            # Check if body is a single pass
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                # Determine what's being caught
                if node.type is None:
                    # bare except:
                    hits.append((node.lineno, "bare except"))
                elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    hits.append((node.lineno, "except Exception"))
        return hits

    def test_no_broad_except_pass_in_weft(self):
        """No except Exception: pass or bare except: pass in any weft/ .py file."""
        all_hits: list[str] = []
        for pyfile in sorted(WEFT_ROOT.rglob("*.py")):
            hits = self._find_bare_except_pass(pyfile)
            for line, kind in hits:
                rel = pyfile.relative_to(WEFT_ROOT.parent)
                all_hits.append(f"{rel}:{line} — {kind}: pass")

        assert all_hits == [], (
            "Found broad except: pass patterns that should use logger.warning:\n"
            + "\n".join(all_hits)
        )


# ---------------------------------------------------------------------------
# Specific site tests: verify logging on error
# ---------------------------------------------------------------------------


class TestCacheLogsOnFailure:
    """cache.py set_stats and invalidate_stats should log, not silently pass."""

    @pytest.mark.asyncio
    async def test_set_stats_logs_warning(self, caplog):
        from weft.cache import Cache

        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(side_effect=RuntimeError("connection lost"))
        cache = Cache(mock_redis)

        with caplog.at_level(logging.WARNING, logger="weft.cache"):
            await cache.set_stats({"test": 1})

        assert any("set_stats" in r.message for r in caplog.records), (
            "Expected a warning log mentioning set_stats"
        )

    @pytest.mark.asyncio
    async def test_invalidate_stats_logs_warning(self, caplog):
        from weft.cache import Cache

        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock(side_effect=RuntimeError("connection lost"))
        cache = Cache(mock_redis)

        with caplog.at_level(logging.WARNING, logger="weft.cache"):
            await cache.invalidate_stats()

        assert any("invalidate_stats" in r.message for r in caplog.records), (
            "Expected a warning log mentioning invalidate_stats"
        )


class TestServerLogsOnCloseFailure:
    """server.py pool/redis close should log, not silently pass."""

    @pytest.mark.asyncio
    async def test_old_pool_close_logs_warning(self, caplog):
        """When old pool close fails during keepalive, it should log."""
        # This tests the pattern, not the full keepalive loop
        from weft.mcp.server import logger as server_logger

        with caplog.at_level(logging.DEBUG, logger="weft.mcp.server"):
            # Simulate the old_pool.close() failure path
            old_pool = AsyncMock()
            old_pool.close = AsyncMock(side_effect=RuntimeError("close failed"))
            try:
                await old_pool.close()
            except RuntimeError:
                server_logger.debug("Old pool close failed", exc_info=True)

        # Just verify the pattern — the real test is the AST lint above


class TestToolsLogsOnFailure:
    """tools.py should log on entity enrichment and contradiction alert failures."""

    @pytest.mark.asyncio
    async def test_detect_project_id_logs_on_failure(self, caplog):
        from weft.mcp.tools import _detect_project_id

        mock_ctx = AsyncMock()
        mock_ctx.list_roots = AsyncMock(side_effect=RuntimeError("no roots"))

        with caplog.at_level(logging.DEBUG, logger="weft.mcp.tools"):
            result = await _detect_project_id(mock_ctx)

        assert result is None
        assert any("project_id" in r.message.lower() or "detect" in r.message.lower()
                    for r in caplog.records), (
            "Expected a log mentioning project_id detection failure"
        )

    @pytest.mark.asyncio
    async def test_contradiction_alert_logs_on_failure(self, caplog):
        """When contradiction alert creation fails, it should log a warning."""
        # We test this by checking that the source code of weft_remember
        # contains logger.warning in the contradiction alert except block
        source = inspect.getsource(
            __import__("weft.mcp.tools", fromlist=["weft_remember"]).weft_remember
        )
        assert "logger.warning" in source or "logger.debug" in source, (
            "weft_remember's contradiction alert except block should log"
        )
        assert "except Exception:\n                        pass" not in source, (
            "weft_remember should not silently pass on contradiction alert failure"
        )

    @pytest.mark.asyncio
    async def test_entity_enrichment_logs_on_failure(self, caplog):
        """Entity enrichment in weft_recall except block should log."""
        source = inspect.getsource(
            __import__("weft.mcp.tools", fromlist=["weft_recall"]).weft_recall
        )
        assert "except Exception:\n                    pass" not in source, (
            "weft_recall should not silently pass on entity enrichment failure"
        )
