"""TDD — Tests for BitsoWebSocketConnector (RED→GREEN phase).

El conector WS implementa ``StreamingMarketDataConnector``.
Se suscribe a los canales ``trades`` y ``diff-orders`` de Bitso, mantiene
un estado interno de mejor bid/ask/last, y emite un ``Ticker`` al callback.

Estrategia de test:
  - Suscripción: se verifica que se envíen los mensajes de subscribe
    correctos al WebSocket (usando FakeWebSocket).
  - Parseo de mensajes: se llama ``_handle_message`` directamente con
    mensajes JSON crudos, sin depender del loop asíncrono.
  - Integración: ``run()`` procesa toda la cola del FakeWebSocket.
"""
import json
from decimal import Decimal

from trading_system.connectors.base import Ticker
from trading_system.connectors.bitso.websocket import BitsoWebSocketConnector


def _trade_msg(book: str, price: str, ts: int = 1700000000000) -> str:
    return json.dumps({
        "type": "trades",
        "book": book,
        "payload": [{"r": price, "a": "0.01", "t": ts}],
    })


def _diff_orders_msg(book: str, bids: list[dict], asks: list[dict]) -> str:
    return json.dumps({
        "type": "diff-orders",
        "book": book,
        "payload": bids + asks,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _make_connector_with_subscription(
    messages: list[str] | None = None,
    make_fake_ws_factory=None,
    symbol: str = "btc_mxn",
):
    ws, factory = make_fake_ws_factory(messages)
    connector = BitsoWebSocketConnector(ws_url="wss://ws.bitso.com", ws_factory=factory)
    received: list[Ticker] = []

    async def callback(ticker: Ticker) -> None:
        received.append(ticker)

    await connector.subscribe_ticker(symbol, callback)
    return connector, ws, received


# ---------------------------------------------------------------------------
# Subscription
# ---------------------------------------------------------------------------

class TestSubscribeTicker:
    async def test_subscribe_sends_both_channels(self, make_fake_ws_factory):
        ws, factory = make_fake_ws_factory()
        connector = BitsoWebSocketConnector(ws_url="wss://ws.bitso.com", ws_factory=factory)

        async def callback(t: Ticker) -> None:
            pass

        await connector.subscribe_ticker("btc_mxn", callback)

        subscribed = [m for m in ws.sent_jsons if m.get("action") == "subscribe"]
        assert len(subscribed) == 2
        types = {m["type"] for m in subscribed}
        assert types == {"trades", "diff-orders"}
        assert all(m["book"] == "btc_mxn" for m in subscribed)

    async def test_unsubscribe_sends_unsubscribe_message(self, make_fake_ws_factory):
        ws, factory = make_fake_ws_factory()
        connector = BitsoWebSocketConnector(ws_url="wss://ws.bitso.com", ws_factory=factory)

        async def callback(t: Ticker) -> None:
            pass

        await connector.subscribe_ticker("btc_mxn", callback)
        await connector.unsubscribe("btc_mxn")

        unsubs = [m for m in ws.sent_jsons if m.get("action") == "unsubscribe"]
        assert len(unsubs) == 2


# ---------------------------------------------------------------------------
# Message handling via _handle_message (deterministic, no async loop)
# ---------------------------------------------------------------------------

class TestHandleMessageTrades:
    async def test_trade_message_updates_last_and_emits_ticker(self, make_fake_ws_factory):
        connector, _ws, received = await _make_connector_with_subscription(
            make_fake_ws_factory=make_fake_ws_factory,
        )

        await connector._handle_message(_trade_msg("btc_mxn", "500000", ts=1700000000123))

        assert len(received) == 1
        t = received[0]
        assert isinstance(t, Ticker)
        assert t.symbol == "btc_mxn"
        assert t.last == Decimal(500000)
        assert t.timestamp_ms == 1700000000123

    async def test_multiple_trades_update_last_price(self, make_fake_ws_factory):
        connector, _ws, received = await _make_connector_with_subscription(
            make_fake_ws_factory=make_fake_ws_factory,
        )

        await connector._handle_message(_trade_msg("btc_mxn", "500000"))
        await connector._handle_message(_trade_msg("btc_mxn", "501000"))

        assert len(received) == 2
        assert received[0].last == Decimal(500000)
        assert received[1].last == Decimal(501000)


class TestHandleMessageDiffOrders:
    async def test_diff_orders_update_bid_and_ask(self, make_fake_ws_factory):
        connector, _ws, received = await _make_connector_with_subscription(
            make_fake_ws_factory=make_fake_ws_factory,
        )

        await connector._handle_message(
            _diff_orders_msg("btc_mxn", [{"r": "499000", "s": "0", "d": "0.5"}], [])
        )
        await connector._handle_message(
            _diff_orders_msg("btc_mxn", [], [{"r": "501000", "s": "1", "d": "0.3"}])
        )
        await connector._handle_message(_trade_msg("btc_mxn", "500000"))

        assert len(received) == 1
        trade_ticker = received[-1]
        assert trade_ticker.bid == Decimal(499000)
        assert trade_ticker.ask == Decimal(501000)
        assert trade_ticker.last == Decimal(500000)


# ---------------------------------------------------------------------------
# Multiple symbols
# ---------------------------------------------------------------------------

class TestMultipleSymbols:
    async def test_dispatch_to_correct_callback(self, make_fake_ws_factory):
        _ws, factory = make_fake_ws_factory()
        connector = BitsoWebSocketConnector(ws_url="wss://ws.bitso.com", ws_factory=factory)

        btc_received: list[Ticker] = []
        eth_received: list[Ticker] = []

        async def cb_btc(t: Ticker) -> None:
            btc_received.append(t)

        async def cb_eth(t: Ticker) -> None:
            eth_received.append(t)

        await connector.subscribe_ticker("btc_mxn", cb_btc)
        await connector.subscribe_ticker("eth_mxn", cb_eth)

        await connector._handle_message(_trade_msg("btc_mxn", "500000"))
        await connector._handle_message(_trade_msg("eth_mxn", "30000"))

        assert len(btc_received) == 1
        assert btc_received[0].symbol == "btc_mxn"
        assert len(eth_received) == 1
        assert eth_received[0].symbol == "eth_mxn"
        assert eth_received[0].last == Decimal(30000)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestInvalidMessages:
    async def test_malformed_json_does_not_crash(self, make_fake_ws_factory):
        connector, _ws, received = await _make_connector_with_subscription(
            make_fake_ws_factory=make_fake_ws_factory,
        )

        await connector._handle_message("not-json")
        await connector._handle_message(_trade_msg("btc_mxn", "500000"))

        assert len(received) == 1

    async def test_missing_type_does_not_crash(self, make_fake_ws_factory):
        connector, _ws, received = await _make_connector_with_subscription(
            make_fake_ws_factory=make_fake_ws_factory,
        )

        await connector._handle_message(json.dumps({"book": "btc_mxn", "payload": []}))

        assert len(received) == 0

    async def test_diff_orders_without_trade_does_not_emit(self, make_fake_ws_factory):
        connector, _ws, received = await _make_connector_with_subscription(
            make_fake_ws_factory=make_fake_ws_factory,
        )

        await connector._handle_message(
            _diff_orders_msg("btc_mxn", [{"r": "499000", "s": "0", "d": "0.5"}], [])
        )

        assert len(received) == 0


# ---------------------------------------------------------------------------
# Integration: run() loop with FakeWebSocket
# ---------------------------------------------------------------------------

class TestRunLoopIntegration:
    async def test_run_processes_all_messages_then_exits(self, make_fake_ws_factory):
        messages = [
            _trade_msg("btc_mxn", "500000"),
            _trade_msg("btc_mxn", "501000"),
            _trade_msg("btc_mxn", "502000"),
        ]
        connector, ws, received = await _make_connector_with_subscription(
            messages=messages,
            make_fake_ws_factory=make_fake_ws_factory,
        )

        await connector.run()

        assert len(received) == 3
        assert received[-1].last == Decimal(502000)
        assert ws.is_closed