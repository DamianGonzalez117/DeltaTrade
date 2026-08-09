"""TDD — Tests for GBMPollingConnector (RED phase).

Implementa ``PollingMarketDataConnector``. Consulta el ticker de un
símbolo vía REST y reporta si el mercado está abierto usando
``BMVCalendar``.

El ``httpx.AsyncClient`` se inyecta para tests con ``MockTransport``.
El ``BMVCalendar`` también se inyecta para poder simular horas arbitrarias.
"""
from datetime import datetime
from decimal import Decimal

import httpx
import pytest

from trading_system.connectors.base import Ticker
from trading_system.connectors.gbm.calendar import BMVCalendar
from trading_system.connectors.gbm.polling import GBMPollingConnector


def _ticker_response(symbol: str = "NAFTRAC", last: str = "540.50",
                     bid: str = "540.40", ask: str = "540.60",
                     ts: str = "1700000000000") -> dict:
    return {
        "success": True,
        "payload": {
            "symbol": symbol,
            "lastPrice": last,
            "bid": bid,
            "ask": ask,
            "timestamp": ts,
        },
    }


def _make_connector(handler, calendar: BMVCalendar | None = None) -> GBMPollingConnector:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://api.gbm.com")
    cal = calendar or BMVCalendar()
    return GBMPollingConnector(base_url="https://api.gbm.com", client=client, calendar=cal)


class TestGetTicker:
    async def test_returns_ticker_with_correct_fields(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/market-data/ticker/NAFTRAC"
            return httpx.Response(200, json=_ticker_response())

        connector = _make_connector(handler)
        await connector.connect()
        ticker = await connector.get_ticker("NAFTRAC")
        await connector.disconnect()

        assert isinstance(ticker, Ticker)
        assert ticker.symbol == "NAFTRAC"
        assert ticker.last == Decimal("540.50")
        assert ticker.bid == Decimal("540.40")
        assert ticker.ask == Decimal("540.60")
        assert ticker.timestamp_ms == 1700000000000

    async def test_get_ticker_sends_get_request(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["path"] = request.url.path
            return httpx.Response(200, json=_ticker_response())

        connector = _make_connector(handler)
        await connector.connect()
        await connector.get_ticker("LIVEPOLI-1")
        await connector.disconnect()

        assert seen["method"] == "GET"
        assert "LIVEPOLI-1" in seen["path"]

    async def test_http_error_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"success": False, "error": "not found"})

        connector = _make_connector(handler)
        await connector.connect()
        with pytest.raises(httpx.HTTPStatusError):
            await connector.get_ticker("UNKNOWN")
        await connector.disconnect()


class TestIsMarketOpen:
    async def test_returns_true_when_calendar_says_open(self):
        class FakeCalendar(BMVCalendar):
            def is_market_open(self, dt: datetime) -> bool:
                return True

        connector = _make_connector(lambda r: httpx.Response(200), calendar=FakeCalendar())
        await connector.connect()
        assert await connector.is_market_open() is True
        await connector.disconnect()

    async def test_returns_false_when_calendar_says_closed(self):
        class FakeCalendar(BMVCalendar):
            def is_market_open(self, dt: datetime) -> bool:
                return False

        connector = _make_connector(lambda r: httpx.Response(200), calendar=FakeCalendar())
        await connector.connect()
        assert await connector.is_market_open() is False
        await connector.disconnect()

    async def test_uses_current_utc_time(self):
        class CaptureCalendar(BMVCalendar):
            captured_dt: datetime | None = None

            def is_market_open(self, dt: datetime) -> bool:
                self.captured_dt = dt
                return True

        cal = CaptureCalendar()
        connector = _make_connector(lambda r: httpx.Response(200), calendar=cal)
        await connector.connect()
        await connector.is_market_open()
        await connector.disconnect()

        assert cal.captured_dt is not None
        assert cal.captured_dt.tzinfo is not None


class TestConnectDisconnect:
    async def test_connect_then_disconnect(self):
        connector = _make_connector(lambda r: httpx.Response(200, json=_ticker_response()))
        await connector.connect()
        await connector.disconnect()
        assert True