"""Tests for user identity module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from weft.config.user_identity import get_user_id


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
                result = get_user_id()

                # Assert length is 32
                assert len(result) == 32

                # Assert matches regex for hex chars
                import re

                assert re.match(r"^[a-f0-9]{32}$", result)
