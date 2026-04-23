"""Tests for user identity module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from weft.config.user_identity import describe_user_id, get_user_id, set_user_id


class TestGetUserIdIdempotent:
    """Test that get_user_id returns the same value on multiple calls."""

    def test_get_user_id_idempotent(self):
        """Call twice, assert both return equal non-empty strs."""
        first_call = get_user_id()
        second_call = get_user_id()

        assert first_call == second_call
        assert isinstance(first_call, str)
        assert len(first_call) > 0


class TestGetUserIdPersists:
    """Test that get_user_id persists the UUID to a file."""

    def test_get_user_id_persists(self):
        """Use tempfile; patch home(); assert file exists with valid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                first_result = get_user_id()

                # Assert file exists and contains valid JSON
                user_id_file = Path(temp_dir) / ".weft" / "user_id.json"
                assert user_id_file.exists()
                assert user_id_file.is_file()

                file_content = json.loads(user_id_file.read_text())
                assert "user_id" in file_content
                assert file_content["user_id"] == first_result

                # Call again; assert returned value equals stored result
                second_result = get_user_id()
                assert second_result == first_result


class TestGetUserIdFormat:
    """Test that get_user_id returns a valid UUID hex string."""

    def test_get_user_id_format(self):
        """Assert return matches UUID hex format (32 hex chars)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                # Clear env var so fallback path runs.
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("WEFT_USER_ID", None)
                    result = get_user_id()

                # Assert length is 32
                assert len(result) == 32

                # Assert matches regex for hex chars
                import re

                assert re.match(r"^[a-f0-9]{32}$", result)


# ---------------------------------------------------------------------------
# Precedence chain: env var > config file > random UUID fallback
# ---------------------------------------------------------------------------


class TestPrecedenceChain:
    """WEFT_USER_ID env > ~/.weft/user_id.json explicit > random UUID."""

    def test_env_var_wins_over_config_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                # Seed the config file with one value.
                set_user_id("from-config")
                # Set env to a different value.
                with patch.dict(os.environ, {"WEFT_USER_ID": "from-env"}):
                    assert get_user_id() == "from-env"
                # Without env, config file value returns.
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("WEFT_USER_ID", None)
                    assert get_user_id() == "from-config"

    def test_config_file_wins_over_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("WEFT_USER_ID", None)
                    set_user_id("deliberate-id")
                    assert get_user_id() == "deliberate-id"

    def test_fallback_generates_and_persists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("WEFT_USER_ID", None)
                    first = get_user_id()
                    # File exists with same value
                    path = Path(temp_dir) / ".weft" / "user_id.json"
                    assert json.loads(path.read_text())["user_id"] == first
                    # Second call returns same value (no regeneration)
                    assert get_user_id() == first

    def test_env_var_does_not_persist(self):
        """Env var overrides at read time but never writes to the config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                with patch.dict(os.environ, {"WEFT_USER_ID": "ephemeral"}):
                    assert get_user_id() == "ephemeral"
                    # Config file was never created because env path returned first.
                    assert not (Path(temp_dir) / ".weft" / "user_id.json").exists()


# ---------------------------------------------------------------------------
# set_user_id: explicit persistence
# ---------------------------------------------------------------------------


class TestSetUserId:
    def test_set_persists_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                set_user_id("my-canonical-id")
                path = Path(temp_dir) / ".weft" / "user_id.json"
                assert json.loads(path.read_text())["user_id"] == "my-canonical-id"

    def test_set_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                set_user_id("first")
                set_user_id("second")
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("WEFT_USER_ID", None)
                    assert get_user_id() == "second"

    def test_set_rejects_empty(self):
        with pytest.raises(ValueError):
            set_user_id("")

    def test_set_rejects_non_string(self):
        with pytest.raises(ValueError):
            set_user_id(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# describe_user_id: diagnostics without side effects
# ---------------------------------------------------------------------------


class TestDescribeUserId:
    def test_describe_env_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                with patch.dict(os.environ, {"WEFT_USER_ID": "env-val"}):
                    info = describe_user_id()
                    assert info["user_id"] == "env-val"
                    assert info["source"] == "env"

    def test_describe_config_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("WEFT_USER_ID", None)
                    set_user_id("persisted")
                    info = describe_user_id()
                    assert info["user_id"] == "persisted"
                    assert info["source"] == "config"

    def test_describe_unset_does_not_generate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("WEFT_USER_ID", None)
                    info = describe_user_id()
                    assert info["user_id"] is None
                    assert info["source"] == "unset"
                    # Critical: describe_user_id must NOT create the file.
                    assert not (Path(temp_dir) / ".weft" / "user_id.json").exists()
