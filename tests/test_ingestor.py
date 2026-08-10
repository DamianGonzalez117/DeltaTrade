"""TDD — Tests for BitsoDataIngestor (RED→GREEN).

El cliente HTTP se mockea con MockTransport. La agregación OHLCV y el
mapeo se prueban sin tocar la red ni la DB.
"""

import httpx
import pandas as pd

from trading_system.core.enums import TimeFrame
from trading_system.services.ingestor import BitsoDataIngestor


def _trades_payload(n: int = 5, base_ts: int = 1700000000000) -> dict:
    trades = []
    for i in range(n):
        trades.append({
            "tid": str(1000 + i),
            "price": str(500000 + i * 100),
            "amount": str(0.01 * (i + 1)),
            "created_at": base_ts + i * 60_000,  # 1 min apart
            "book": "btc_mxn",
            "side": "buy" if i % 2 == 0 else "sell",
        })
    return {"success": True, "payload": trades}


class TestFetchTrades:
    async def test_fetch_trades_returns_list(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/v3/trades/" in request.url.path
            assert "btc_mxn" in str(request.url)
            return httpx.Response(200, json=_trades_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport, base_url="https://api.bitso.com")
        ingestor = BitsoDataIngestor(client=client)

        trades = await ingestor.fetch_trades("btc_mxn", limit=5)
        await ingestor.aclose()

        assert len(trades) == 5
        assert "price" in trades[0]

    async def test_fetch_trades_with_marker(self):
        seen_params = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_params["marker"] = request.url.params.get("marker")
            return httpx.Response(200, json=_trades_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport, base_url="https://api.bitso.com")
        ingestor = BitsoDataIngestor(client=client)

        await ingestor.fetch_trades("btc_mxn", limit=5, marker="1005")
        await ingestor.aclose()

        assert seen_params["marker"] == "1005"


class TestFetchTradesBatch:
    async def test_fetch_batch_returns_dataframe(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_trades_payload(10))

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport, base_url="https://api.bitso.com")
        ingestor = BitsoDataIngestor(client=client)

        df = await ingestor.fetch_trades_batch("btc_mxn", total=20)
        await ingestor.aclose()

        assert isinstance(df, pd.DataFrame)
        assert "timestamp" in df.columns
        assert "price" in df.columns
        assert len(df) > 0

    async def test_fetch_batch_empty_when_no_trades(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"success": True, "payload": []})

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport, base_url="https://api.bitso.com")
        ingestor = BitsoDataIngestor(client=client)

        df = await ingestor.fetch_trades_batch("btc_mxn", total=10)
        await ingestor.aclose()

        assert df.empty


class TestAggregateOHLCV:
    def test_aggregate_to_1min_candles(self):
        trades = pd.DataFrame({
            "timestamp": pd.to_datetime(
                [1700000000000, 1700000030000, 1700000060000], unit="ms", utc=True
            ),
            "price": [500000.0, 501000.0, 502000.0],
            "amount": [0.5, 0.3, 0.2],
        })
        ohlcv = BitsoDataIngestor.aggregate_ohlcv(trades, TimeFrame.M1)
        assert len(ohlcv) > 0
        assert "open" in ohlcv.columns
        assert "close" in ohlcv.columns
        assert ohlcv.iloc[0]["open"] == 500000.0

    def test_aggregate_empty_trades(self):
        trades = pd.DataFrame({"timestamp": [], "price": [], "amount": []})
        ohlcv = BitsoDataIngestor.aggregate_ohlcv(trades, TimeFrame.H1)
        assert ohlcv.empty

    def test_aggregate_1h_candles(self):
        # 3 trades: two in the same hour, one in the next
        ts_ms = [1700000000000, 1700001800000, 1700003600000]
        trades = pd.DataFrame({
            "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True),
            "price": [500000.0, 501000.0, 502000.0],
            "amount": [0.5, 0.3, 0.2],
        })
        ohlcv = BitsoDataIngestor.aggregate_ohlcv(trades, TimeFrame.H1)
        assert len(ohlcv) == 2
        first = ohlcv.iloc[0]
        assert first["high"] == 501000.0
        assert first["low"] == 500000.0
        assert first["volume"] == 0.8
        second = ohlcv.iloc[1]
        assert second["open"] == 502000.0
        assert second["close"] == 502000.0


class TestParseSymbol:
    def test_parses_underscore_symbol(self):
        base, quote = BitsoDataIngestor.parse_symbol("btc_mxn")
        assert base == "btc"
        assert quote == "mxn"

    def test_parses_no_underscore_defaults_to_mxn(self):
        base, quote = BitsoDataIngestor.parse_symbol("CUSTOM")
        assert base == "CUSTOM"
        assert quote == "mxn"