"""
Conector WebSocket de Bitso.

Implementa ``StreamingMarketDataConnector``. Se suscribe a los canales
``trades`` y ``diff-orders`` de Bitso, mantiene un estado interno con el
mejor bid/ask y último precio, y emite un ``Ticker`` normalizado al
callback en cada actualización.

Diseño:
  - ``subscribe_ticker``: registra callback + envía mensajes de
    suscripción. No bloquea (non-blocking).
  - ``run``: inicia el loop de recepción (bloquea hasta desconexión).
    En producción se lanza como task; en tests se llama directamente o
    se prueba ``_handle_message`` sin el loop.
  - ``_handle_message``: parsea y despacha un mensaje crudo. Punto
    clave para TDD — testeable sin loop asíncrono.

El ``ws_factory`` se inyecta por constructor (DIP) para que los tests
pasen un ``FakeWebSocket`` sin tocar la red.
"""
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import structlog

from trading_system.connectors.base import StreamingMarketDataConnector, Ticker

logger = structlog.get_logger(__name__)

_TRADES = "trades"
_DIFF_ORDERS = "diff-orders"


class _WSConnection(Protocol):
    """Structural type for anything the connector can talk to (real or fake)."""

    async def send(self, message: str) -> None: ...
    async def recv(self) -> str: ...
    async def close(self) -> None: ...


@dataclass
class _TickerState:
    """Estado mutable para un book: mejor bid/ask y último precio.

    SRP: solo guarda y combina datos; no sabe nada de sockets.
    """
    symbol: str
    bid: Decimal | None = None
    ask: Decimal | None = None
    last: Decimal | None = None
    last_ts: int = 0

    def to_ticker(self) -> Ticker:
        return Ticker(
            symbol=self.symbol,
            bid=self.bid or Decimal(0),
            ask=self.ask or Decimal(0),
            last=self.last or Decimal(0),
            timestamp_ms=self.last_ts,
        )


class BitsoWebSocketConnector(StreamingMarketDataConnector):
    """Streaming de datos de mercado de Bitso vía WebSocket."""

    def __init__(
        self,
        *,
        ws_url: str = "wss://ws.bitso.com",
        ws_factory: Callable[[str], Awaitable[Any]] | None = None,
    ) -> None:
        self._ws_url = ws_url
        self._ws_factory = ws_factory
        self._ws: _WSConnection | None = None
        self._callbacks: dict[str, Callable[[Ticker], Awaitable[None]]] = {}
        self._states: dict[str, _TickerState] = {}
        self._running = False

    async def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[Ticker], Awaitable[None]],
    ) -> None:
        self._callbacks[symbol] = callback
        self._states[symbol] = _TickerState(symbol=symbol)

        if self._ws is None:
            await self._connect_ws()

        await self._send_subscribe(symbol, _TRADES)
        await self._send_subscribe(symbol, _DIFF_ORDERS)

    async def unsubscribe(self, symbol: str) -> None:
        await self._send_unsubscribe(symbol, _TRADES)
        await self._send_unsubscribe(symbol, _DIFF_ORDERS)
        self._callbacks.pop(symbol, None)
        self._states.pop(symbol, None)

    async def run(self) -> None:
        """Inicia el loop de recepción. Bloquea hasta que se cierra la conexión."""
        self._running = True
        while self._running and self._ws is not None:
            try:
                raw = await self._ws.recv()
            except (ConnectionError, OSError, RuntimeError):
                logger.warning("bitso_ws.disconnected")
                self._running = False
                break
            if isinstance(raw, str):
                await self._handle_message(raw)

    async def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        logger.info("bitso_ws.stopped")

    async def _connect_ws(self) -> None:
        if self._ws_factory is not None:
            self._ws = await self._ws_factory(self._ws_url)
        else:
            import websockets

            self._ws = await websockets.connect(self._ws_url)  # type: ignore[assignment]
        logger.info("bitso_ws.connected", url=self._ws_url)

    async def _send_subscribe(self, symbol: str, channel: str) -> None:
        if self._ws is None:
            return
        msg = json.dumps({"action": "subscribe", "book": symbol, "type": channel})
        await self._ws.send(msg)

    async def _send_unsubscribe(self, symbol: str, channel: str) -> None:
        if self._ws is None:
            return
        msg = json.dumps({"action": "unsubscribe", "book": symbol, "type": channel})
        await self._ws.send(msg)

    async def _handle_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("bitso_ws.malformed_message", raw=raw[:200])
            return
        if not isinstance(data, dict) or "type" not in data or "book" not in data:
            return

        book = data["book"]
        if book not in self._callbacks:
            return
        state = self._states[book]

        msg_type = data["type"]
        if msg_type == _TRADES:
            self._apply_trades(state, data.get("payload", []))
        elif msg_type == _DIFF_ORDERS:
            self._apply_diff_orders(state, data.get("payload", []))

        if state.last is not None:
            ticker = state.to_ticker()
            callback = self._callbacks[book]
            await callback(ticker)

    def _apply_trades(self, state: _TickerState, payload: list[dict]) -> None:
        for trade in payload:
            if "r" in trade and "t" in trade:
                state.last = Decimal(str(trade["r"]))
                state.last_ts = int(trade["t"])

    def _apply_diff_orders(self, state: _TickerState, payload: list[dict]) -> None:
        for order in payload:
            if "r" not in order or "s" not in order:
                continue
            rate = Decimal(str(order["r"]))
            side = str(order["s"])
            if side == "0":
                state.bid = rate
            elif side == "1":
                state.ask = rate