"""TDD — Tests for GBMAuth (RED phase).

GBM usa autenticación basada en token:
  1. Login: POST /auth/login con api_key + api_secret → recibe JWT
  2. Requests subsecuentes: header ``Authorization: Bearer <token>``
  3. El token expira; se debe refrescar cuando sea necesario

Esta clase maneja el estado del token. La llamada HTTP real la delega
al conector REST (DIP: GBMAuth no conoce httpx).
"""
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from trading_system.connectors.gbm.auth import GBMAuth


class TestGBMAuthLogin:
    async def test_login_returns_token(self):
        auth = GBMAuth(api_key="my-key", api_secret="my-secret")
        login_fn = AsyncMock(return_value=("jwt-token-abc", 3600))
        token = await auth.login(login_fn)
        assert token == "jwt-token-abc"
        login_fn.assert_awaited_once_with("my-key", "my-secret")

    async def test_login_stores_token_and_expiry(self):
        auth = GBMAuth(api_key="my-key", api_secret="my-secret")
        login_fn = AsyncMock(return_value=("jwt-token-xyz", 3600))
        await auth.login(login_fn)
        assert auth.token == "jwt-token-xyz"
        assert auth.is_token_valid() is True

    async def test_empty_credentials_raise_value_error(self):
        with pytest.raises(ValueError, match="API key"):
            GBMAuth(api_key="", api_secret="secret")

        with pytest.raises(ValueError, match="API secret"):
            GBMAuth(api_key="key", api_secret="")


class TestTokenValidity:
    async def test_token_valid_before_expiry(self):
        auth = GBMAuth(api_key="k", api_secret="s")
        auth._token = "abc"
        auth._token_expires_at = datetime.now(UTC) + timedelta(seconds=300)
        assert auth.is_token_valid() is True

    async def test_token_expired_invalid(self):
        auth = GBMAuth(api_key="k", api_secret="s")
        auth._token = "abc"
        auth._token_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        assert auth.is_token_valid() is False

    async def test_no_token_invalid(self):
        auth = GBMAuth(api_key="k", api_secret="s")
        assert auth.is_token_valid() is False

    async def test_token_valid_with_small_buffer(self):
        auth = GBMAuth(api_key="k", api_secret="s")
        auth._token = "abc"
        # Expira en 15 segundos — dentro del buffer de 60s, debe ser inválido
        auth._token_expires_at = datetime.now(UTC) + timedelta(seconds=15)
        assert auth.is_token_valid() is False


class TestBuildHeaders:
    async def test_headers_contain_bearer_token(self):
        auth = GBMAuth(api_key="k", api_secret="s")
        auth._token = "my-jwt"
        auth._token_expires_at = datetime.now(UTC) + timedelta(hours=1)
        headers = auth.build_headers()
        assert headers["Authorization"] == "Bearer my-jwt"

    async def test_headers_without_token_raise(self):
        auth = GBMAuth(api_key="k", api_secret="s")
        with pytest.raises(RuntimeError, match="No hay token"):
            auth.build_headers()


class TestEnsureValidToken:
    async def test_ensures_valid_calls_login_when_expired(self):
        auth = GBMAuth(api_key="k", api_secret="s")
        login_fn = AsyncMock(return_value=("fresh-token", 3600))

        await auth.ensure_valid_token(login_fn)
        login_fn.assert_awaited_once()

    async def test_ensures_valid_skips_login_when_valid(self):
        auth = GBMAuth(api_key="k", api_secret="s")
        auth._token = "existing"
        auth._token_expires_at = datetime.now(UTC) + timedelta(hours=1)
        login_fn = AsyncMock()

        await auth.ensure_valid_token(login_fn)
        login_fn.assert_not_awaited()