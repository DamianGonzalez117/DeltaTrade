"""
Monitor de mercado: escanea libros de Bitso y notifica señales por Telegram.

Flujo por cada libro:
  1. Descarga trades recientes de Bitso (endpoint público, sin auth).
  2. Agrega a velas OHLCV del timeframe configurado.
  3. Calcula features técnicas (RSI, MACD, SMA, EMA, ATR).
  4. Genera señal con ``SignalGenerator`` (reglas RSI + MACD).
  5. Persiste la señal en ``ai_signals`` (si hay history_service).
  6. Filtra por posiciones abiertas:
     - BUY: solo notifica si NO hay posición abierta para ese libro.
     - SELL: solo notifica si HAY posición abierta, incluye P&L.
  7. Si pasa el filtro → envía notificación a Telegram.

DIP: ``BitsoDataIngestor``, ``TechnicalFeaturesPipeline``, ``SignalGenerator``,
``NotificationDispatcher``, ``SignalHistoryService`` se inyectan por constructor.
SRP: solo orquesta. No decide tamaños, no ejecuta órdenes.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal

import structlog

from trading_system.core.enums import NotificationLevel, SignalAction, TimeFrame
from trading_system.features.pipeline import TechnicalFeaturesPipeline
from trading_system.notifications.base import NotificationMessage
from trading_system.notifications.dispatcher import NotificationDispatcher
from trading_system.services.ingestor import BitsoDataIngestor
from trading_system.services.signal_generator import RuleSignal, SignalGenerator

logger = structlog.get_logger(__name__)


@dataclass
class MonitorConfig:
    """Configuración del monitor."""

    books: list[str] = field(default_factory=lambda: ["btc_mxn", "eth_mxn"])
    timeframe: TimeFrame = TimeFrame.H1
    trades_per_book: int = 1000
    interval_seconds: float = 300.0  # 5 min entre escaneos
    min_confidence: float = 0.65  # solo notificar señales con confianza >= este valor


class MarketMonitor:
    """Escanea libros de Bitso en loop y notifica señales por Telegram."""

    def __init__(
        self,
        *,
        ingestor: BitsoDataIngestor,
        features_pipeline: TechnicalFeaturesPipeline,
        signal_generator: SignalGenerator,
        dispatcher: NotificationDispatcher,
        config: MonitorConfig | None = None,
        history_service=None,
    ) -> None:
        self._ingestor = ingestor
        self._pipeline = features_pipeline
        self._signal_gen = signal_generator
        self._dispatcher = dispatcher
        self._config = config or MonitorConfig()
        self._history = history_service
        self._last_signals: dict[str, RuleSignal] = {}
        self._running = False

    async def scan_once(self) -> dict[str, RuleSignal]:
        """Ejecuta un escaneo completo de todos los libros configurados.

        Retorna un dict {book: RuleSignal} con la señal de cada libro.
        """
        results: dict[str, RuleSignal] = {}

        for book in self._config.books:
            try:
                signal = await self._scan_book(book)
                results[book] = signal
            except (ValueError, RuntimeError, ConnectionError, OSError) as exc:
                logger.error("monitor.book_error", book=book, error=str(exc))
                results[book] = RuleSignal(
                    action=SignalAction.HOLD,
                    confidence=Decimal(0),
                    rsi=0.0,
                    macd_diff=0.0,
                    reason=f"Error: {exc}",
                )

        return results

    async def _scan_book(self, book: str) -> RuleSignal:
        """Escanea un libro individual: fetch → aggregate → features → signal."""
        logger.info("monitor.scanning", book=book)

        trades = await self._ingestor.fetch_trades_batch(
            book, total=self._config.trades_per_book
        )

        if trades.empty:
            logger.warning("monitor.no_trades", book=book)
            return RuleSignal(
                action=SignalAction.HOLD,
                confidence=Decimal(0),
                rsi=0.0,
                macd_diff=0.0,
                reason="Sin trades disponibles",
            )

        ohlcv = BitsoDataIngestor.aggregate_ohlcv(trades, self._config.timeframe)

        if ohlcv.empty or len(ohlcv) < 30:
            logger.warning("monitor.insufficient_candles", book=book, candles=len(ohlcv))
            return RuleSignal(
                action=SignalAction.HOLD,
                confidence=Decimal(0),
                rsi=0.0,
                macd_diff=0.0,
                reason=f"Velas insuficientes ({len(ohlcv)})",
            )

        features = self._pipeline.extract_latest(ohlcv)
        signal = self._signal_gen.generate(features)

        logger.info(
            "monitor.signal",
            book=book,
            action=signal.action.value,
            confidence=str(signal.confidence),
            rsi=signal.rsi,
            macd_diff=signal.macd_diff,
        )

        if self._history is not None:
            self._save_signal(book, signal)

        if signal.action.value in ("buy", "sell") and float(signal.confidence) >= self._config.min_confidence:
            await self._maybe_notify(book, signal)

        self._last_signals[book] = signal
        return signal

    def _save_signal(self, book: str, signal: RuleSignal) -> None:
        """Persiste la señal en la DB."""
        try:
            self._history.save_signal(book, signal)
        except (ValueError, RuntimeError, OSError) as exc:
            logger.warning("monitor.save_signal_failed", book=book, error=str(exc))

    async def _maybe_notify(self, book: str, signal: RuleSignal) -> None:
        """Filtra por posiciones y notifica si corresponde."""
        if self._history is None:
            await self._notify_signal(book, signal)
            return

        position = self._history.get_open_position(book)
        has_position = position is not None

        if signal.action == SignalAction.BUY and has_position:
            logger.info("monitor.skip_buy_has_position", book=book)
            return

        if signal.action == SignalAction.SELL and not has_position:
            logger.info("monitor.skip_sell_no_position", book=book)
            return

        await self._notify_signal(book, signal, position)

    async def _notify_signal(
        self,
        book: str,
        signal: RuleSignal,
        position=None,
    ) -> None:
        """Envía notificación a Telegram para señales BUY/SELL."""
        action_emoji = "🟢" if signal.action.value == "buy" else "🔴"

        body = (
            f"{action_emoji} *{signal.action.value.upper()}* {book}\n"
            f"Confianza: {float(signal.confidence) * 100:.1f}%\n"
            f"RSI: {signal.rsi:.1f}\n"
            f"MACD diff: {signal.macd_diff:.2f}\n"
            f"Razón: {signal.reason}"
        )

        if position is not None and signal.action == SignalAction.SELL:
            current_price = Decimal(str(signal.rsi))
            pnl = self._history.calc_unrealized_pnl(position, current_price) if self._history else Decimal(0)
            pnl_pct = (pnl / (position.avg_entry_price * position.quantity) * 100) if position.avg_entry_price > 0 else Decimal(0)
            body += (
                f"\n\n📊 *Tu posición:*\n"
                f"Entrada: ${position.avg_entry_price:,.2f}\n"
                f"Cantidad: {position.quantity}\n"
                f"P&L estimado: ${pnl:,.2f} ({pnl_pct:+.1f}%)"
            )

        msg = NotificationMessage(
            level=NotificationLevel.INFO if signal.action.value == "buy" else NotificationLevel.WARNING,
            title=f"Señal {signal.action.value.upper()} — {book}",
            body=body,
        )
        await self._dispatcher.dispatch(msg)
        logger.info("monitor.notified", book=book, action=signal.action.value)

    async def run(self) -> None:
        """Ejecuta el monitor en loop infinito hasta que se llame stop()."""
        self._running = True
        logger.info(
            "monitor.starting",
            books=self._config.books,
            timeframe=self._config.timeframe.value,
            interval=self._config.interval_seconds,
        )

        while self._running:
            await self.scan_once()
            if self._running:
                await asyncio.sleep(self._config.interval_seconds)

        logger.info("monitor.stopped")

    def stop(self) -> None:
        """Detiene el loop del monitor."""
        self._running = False

    @property
    def last_signals(self) -> dict[str, RuleSignal]:
        """Últimas señales generadas por libro."""
        return dict(self._last_signals)
