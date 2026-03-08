"""Tests for weft.auth — JWT user identity extraction and contextvar."""

from __future__ import annotations

import time
from unittest.mock import patch

import jwt as pyjwt
import pytest

from weft.auth import (
    current_user_id,
    extract_user_id,
    extract_user_id_from_header,
)

# Shared test secret
_SECRET = "test-supabase-jwt-secret-32chars!"


def _make_jwt(
    sub: str | None = "test-user-uuid",
    exp: int | None = None,
    secret: str = _SECRET,
    algorithm: str = "HS256",
    **extra_claims,
) -> str:
    """Build a signed JWT for testing."""
    payload: dict = {}
    if sub is not None:
        payload["sub"] = sub
    if exp is not None:
        payload["exp"] = exp
    else:
        payload["exp"] = int(time.time()) + 3600  # 1 hour from now
    payload.update(extra_claims)
    return pyjwt.encode(payload, secret, algorithm=algorithm)


@pytest.fixture(autouse=True)
def _set_jwt_secret():
    """Ensure SUPABASE_JWT_SECRET is set for all tests."""
    with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": _SECRET}):
        yield


# --- extract_user_id ---


class TestExtractUserId:
    def test_valid_jwt(self):
        token = _make_jwt(sub="user-abc-123")
        assert extract_user_id(token) == "user-abc-123"

    def test_expired_jwt(self):
        token = _make_jwt(exp=int(time.time()) - 100)
        assert extract_user_id(token) is None

    def test_wrong_secret(self):
        token = _make_jwt(secret="wrong-secret-entirely!!!!!")
        assert extract_user_id(token) is None

    def test_malformed_not_jwt(self):
        assert extract_user_id("not-a-jwt-at-all") is None

    def test_malformed_partial_jwt(self):
        assert extract_user_id("abc.def") is None

    def test_empty_string(self):
        assert extract_user_id("") is None

    def test_missing_sub_claim(self):
        payload = {"exp": int(time.time()) + 3600, "role": "anon"}
        token = pyjwt.encode(payload, _SECRET, algorithm="HS256")
        assert extract_user_id(token) is None

    def test_empty_sub_claim(self):
        token = _make_jwt(sub="")
        assert extract_user_id(token) is None

    def test_numeric_sub_rejected(self):
        payload = {"sub": 12345, "exp": int(time.time()) + 3600}
        token = pyjwt.encode(payload, _SECRET, algorithm="HS256")
        assert extract_user_id(token) is None

    def test_no_secret_configured(self):
        with patch.dict("os.environ", {}, clear=True):
            token = _make_jwt()
            assert extract_user_id(token) is None

    def test_extra_claims_ignored(self):
        token = _make_jwt(sub="user-xyz", role="authenticated", aud="weft")
        assert extract_user_id(token) == "user-xyz"


# --- extract_user_id_from_header ---


class TestExtractUserIdFromHeader:
    def test_valid_bearer(self):
        token = _make_jwt(sub="user-from-header")
        result = extract_user_id_from_header(f"Bearer {token}")
        assert result == "user-from-header"

    def test_none_header(self):
        assert extract_user_id_from_header(None) is None

    def test_empty_header(self):
        assert extract_user_id_from_header("") is None

    def test_wrong_scheme(self):
        token = _make_jwt()
        assert extract_user_id_from_header(f"Basic {token}") is None

    def test_bearer_no_token(self):
        assert extract_user_id_from_header("Bearer ") is None

    def test_bearer_only(self):
        assert extract_user_id_from_header("Bearer") is None

    def test_case_sensitive_bearer(self):
        token = _make_jwt()
        assert extract_user_id_from_header(f"bearer {token}") is None


# --- current_user_id contextvar ---


class TestContextVar:
    def test_default_is_none(self):
        assert current_user_id.get() is None

    def test_set_and_get(self):
        tok = current_user_id.set("user-ctx-test")
        try:
            assert current_user_id.get() == "user-ctx-test"
        finally:
            current_user_id.reset(tok)

    def test_reset_restores_default(self):
        tok = current_user_id.set("temporary")
        current_user_id.reset(tok)
        assert current_user_id.get() is None
