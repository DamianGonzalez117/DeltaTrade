"""TDD — Tests for retry module (RED→GREEN).

Se prueba que:
  1. TransientError se lanza en 5xx/429/timeouts.
  2. 4xx no se reintentan.
  3. El backoff reintena y eventualmente tiene éxito.
"""
from __future__ import annotations

import httpx
import pytest

from trading_system.core.retry import (
    TransientError,
    retry_async,
    to_transient_if_needed,
)


class TestToTransientIfNeeded:
    def test_500_becomes_transient(self):
        response = httpx.Response(500, request=httpx.Request("GET", "https://x"))
        exc = httpx.HTTPStatusError("server error", request=response.request, response=response)
        result = to_transient_if_needed(exc)
        assert isinstance(result, TransientError)

    def test_429_becomes_transient(self):
        response = httpx.Response(429, request=httpx.Request("GET", "https://x"))
        exc = httpx.HTTPStatusError("rate limit", request=response.request, response=response)
        result = to_transient_if_needed(exc)
        assert isinstance(result, TransientError)

    def test_400_stays_as_is(self):
        response = httpx.Response(400, request=httpx.Request("GET", "https://x"))
        exc = httpx.HTTPStatusError("bad request", request=response.request, response=response)
        result = to_transient_if_needed(exc)
        assert not isinstance(result, TransientError)
        assert isinstance(result, httpx.HTTPStatusError)

    def test_timeout_becomes_transient(self):
        exc = httpx.TimeoutException("timed out")
        result = to_transient_if_needed(exc)
        assert isinstance(result, TransientError)

    def test_network_error_becomes_transient(self):
        exc = httpx.NetworkError("connection reset")
        result = to_transient_if_needed(exc)
        assert isinstance(result, TransientError)

    def test_unrelated_error_passes_through(self):
        exc = ValueError("not transient")
        result = to_transient_if_needed(exc)
        assert isinstance(result, ValueError)
        assert not isinstance(result, TransientError)


class TestRetryAsync:
    async def test_succeeds_first_try(self):
        calls = 0

        async def ok_fn():
            nonlocal calls
            calls += 1
            return {"ok": True}

        result = await retry_async(ok_fn)
        assert result == {"ok": True}
        assert calls == 1

    async def test_retries_on_transient_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("trading_system.core.retry.MIN_WAIT_S", 0)
        monkeypatch.setattr("trading_system.core.retry.MAX_WAIT_S", 0)

        calls = 0

        async def flaky_fn():
            nonlocal calls
            calls += 1
            if calls < 2:
                raise TransientError("transient")
            return {"ok": True}

        result = await retry_async(flaky_fn)
        assert result == {"ok": True}
        assert calls == 2

    async def test_raises_after_max_attempts(self, monkeypatch):
        monkeypatch.setattr("trading_system.core.retry.MIN_WAIT_S", 0)
        monkeypatch.setattr("trading_system.core.retry.MAX_WAIT_S", 0)
        monkeypatch.setattr("trading_system.core.retry.MAX_ATTEMPTS", 2)

        calls = 0

        async def always_fail():
            nonlocal calls
            calls += 1
            raise TransientError("always fails")

        with pytest.raises(TransientError, match="always fails"):
            await retry_async(always_fail)
        assert calls == 2

    async def test_does_not_retry_non_transient(self):
        calls = 0

        async def bad_request():
            nonlocal calls
            calls += 1
            raise ValueError("bad request, not transient")

        with pytest.raises(ValueError, match="bad request"):
            await retry_async(bad_request)
        assert calls == 1
