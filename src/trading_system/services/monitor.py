"""
Monitor de mercado: escanea libros de Bitso y notifica señales por Telegram.

Flujo por cada libro:
  1. Descarga trades recientes de Bitso (endpoint público, sin auth).
  2. Agrega a velas OHLCV del timeframe configurado.
  3. Calcula features técnicas (RSI, MACD, SMA, EMA, ATR, volume ratio).
  4. Genera señal con ``SignalGenerator`` (reglas RSI+MACD y EMA+volumen).
  5. Persiste la señal en ``ai_signals`` (si hay history_service).
  6. Monitorea la posición abierta del usuario contra umbrales:
     - Ganancia >= profit_target_pct → "VENDER, objetivo alcanzado".
     - Pérdida <= stop_loss_pct → "STOP LOSS".
     (independiente de la señal técnica; anti-spam por cruce de umbral)
  7. Filtra señales por posiciones abiertas:
     - BUY: solo notifica si NO hay posición abierta para ese libro.
     - SELL: solo notifica si HAY posición abierta, incluye P&L con precio real.
  8. Si pasa el filtro → envía notificación a Telegram (BUY enriquecida
     con precio actual, objetivo de venta sugerido y stats del usuario).

DIP: ``BitsoDataIngestor``, ``TechnicalFeaturesPipeline``, ``SignalGenerator``,
``NotificationDispatcher``, ``SignalHistoryService``, ``TradeAnalyzer``
se inyectan por constructor.
SRP: solo orquesta. No decide tamaños, no ejecuta órdenes.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal

import httpx
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
    profit_target_pct: float = 0.05  # aviso de VENDER cuando ganancia >= 5%
    stop_loss_pct: float = 0.02  # aviso de STOP LOSS cuando pérdida >= 2%


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
        trade_analyzer=None,
    ) -> None:
        self._ingestor = ingestor
        self._pipeline = features_pipeline
        self._signal_gen = signal_generator
        self._dispatcher = dispatcher
        self._config = config or MonitorConfig()
        self._history = history_service
        self._analyzer = trade_analyzer
        self._last_signals: dict[str, RuleSignal] = {}
        self._last_prices: dict[str, float] = {}
        self._threshold_state: dict[str, str | None] = {}
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
            except (ValueError, RuntimeError, ConnectionError, OSError, httpx.HTTPError, httpx.TimeoutException) as exc:
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
        last_price = float(ohlcv["close"].iloc[-1])
        self._last_prices[book] = last_price

        logger.info(
            "monitor.signal",
            book=book,
            action=signal.action.value,
            confidence=str(signal.confidence),
            rsi=signal.rsi,
            macd_diff=signal.macd_diff,
            last_price=last_price,
        )

        if self._history is not None:
            self._save_signal(book, signal)

        await self._check_position_thresholds(book, last_price)

        if signal.action.value in ("buy", "sell") and float(signal.confidence) >= self._config.min_confidence:
            await self._maybe_notify(book, signal, last_price)

        self._last_signals[book] = signal
        return signal

    def _save_signal(self, book: str, signal: RuleSignal) -> None:
        """Persiste la señal en la DB."""
        try:
            self._history.save_signal(book, signal)
        except (ValueError, RuntimeError, OSError) as exc:
            logger.warning("monitor.save_signal_failed", book=book, error=str(exc))

    async def _check_position_thresholds(self, book: str, last_price: float) -> None:
        """Monitorea la posición abierta contra objetivos de ganancia y stop loss.

        Anti-spam: 1 notificación por cruce de umbral. El estado se rearma
        al volver a zona neutral o al cerrar la posición.
        """
        if self._history is None:
            return

        position = self._history.get_open_position(book)
        if position is None:
            self._threshold_state[book] = None
            return

        entry = float(position.avg_entry_price)
        if entry <= 0:
            return

        pnl_pct = (last_price - entry) / entry

        if pnl_pct >= self._config.profit_target_pct:
            if self._threshold_state.get(book) != "profit":
                self._threshold_state[book] = "profit"
                await self._notify_profit_target(book, position, last_price, pnl_pct)
        elif pnl_pct <= -self._config.stop_loss_pct:
            if self._threshold_state.get(book) != "loss":
                self._threshold_state[book] = "loss"
                await self._notify_stop_loss(book, position, last_price, pnl_pct)
        else:
            self._threshold_state[book] = None

    async def _notify_profit_target(self, book: str, position, last_price: float, pnl_pct: float) -> None:
        """Aviso: objetivo de ganancia alcanzado, momento de vender."""
        pnl = self._calc_pnl(position, last_price)
        body = (
            f"🎯 *VENDER — Objetivo alcanzado*\n"
            f"{book}\n\n"
            f"Entrada: ${float(position.avg_entry_price):,.2f}\n"
            f"Actual: ${last_price:,.2f} ({pnl_pct * 100:+.1f}%)\n"
            f"P&L estimado: ${pnl:,.2f}\n\n"
            f"Confirma con: `vendí {book} {last_price:,.2f}`"
        )
        msg = NotificationMessage(
            level=NotificationLevel.INFO,
            title=f"Objetivo alcanzado — {book}",
            body=body,
        )
        await self._dispatcher.dispatch(msg)
        logger.info("monitor.profit_target_notified", book=book, pnl_pct=round(pnl_pct, 4))

    async def _notify_stop_loss(self, book: str, position, last_price: float, pnl_pct: float) -> None:
        """Aviso: precio cayó por debajo del stop loss."""
        pnl = self._calc_pnl(position, last_price)
        body = (
            f"⚠️ *STOP LOSS — Pérdida en posición*\n"
            f"{book}\n\n"
            f"Entrada: ${float(position.avg_entry_price):,.2f}\n"
            f"Actual: ${last_price:,.2f} ({pnl_pct * 100:+.1f}%)\n"
            f"P&L estimado: ${pnl:,.2f}\n\n"
            f"Evalúa salir: `vendí {book} {last_price:,.2f}`"
        )
        msg = NotificationMessage(
            level=NotificationLevel.WARNING,
            title=f"Stop Loss — {book}",
            body=body,
        )
        await self._dispatcher.dispatch(msg)
        logger.info("monitor.stop_loss_notified", book=book, pnl_pct=round(pnl_pct, 4))

    def _calc_pnl(self, position, last_price: float) -> Decimal:
        """P&L no realizado: usa el history_service si está, si no calcula directo."""
        if self._history is not None:
            try:
                return self._history.calc_unrealized_pnl(position, Decimal(str(last_price)))
            except (ValueError, RuntimeError, OSError):
                pass
        return (Decimal(str(last_price)) - position.avg_entry_price) * position.quantity

    async def _maybe_notify(self, book: str, signal: RuleSignal, last_price: float) -> None:
        """Filtra por posiciones y notifica si corresponde."""
        if self._history is None:
            await self._notify_signal(book, signal, last_price=last_price)
            return

        position = self._history.get_open_position(book)
        has_position = position is not None

        if signal.action == SignalAction.BUY and has_position:
            logger.info("monitor.skip_buy_has_position", book=book)
            return

        if signal.action == SignalAction.SELL and not has_position:
            logger.info("monitor.skip_sell_no_position", book=book)
            return

        await self._notify_signal(book, signal, position, last_price)

    async def _notify_signal(
        self,
        book: str,
        signal: RuleSignal,
        position=None,
        last_price: float | None = None,
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

        if signal.action == SignalAction.BUY:
            body = self._enrich_buy_body(body, book, last_price)

        if position is not None and signal.action == SignalAction.SELL and last_price is not None:
            current_price = Decimal(str(last_price))
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

    def _enrich_buy_body(self, body: str, book: str, last_price: float | None) -> str:
        """Agrega precio actual, objetivo de venta sugerido y stats al BUY."""
        if last_price is None:
            return body

        body += f"\n\n💰 *Precio actual:* ${last_price:,.2f}"

        if self._analyzer is None:
            return body

        try:
            target_pct = self._analyzer.suggest_profit_target_pct(
                book, default_pct=self._config.profit_target_pct * 100
            )
            target_price = last_price * (1 + target_pct / 100)
            body += f"\n🎯 *Objetivo de venta sugerido:* ${target_price:,.2f} (+{target_pct:.1f}%)"

            stats = self._analyzer.get_closed_stats(book)
            if stats is not None:
                body += (
                    f"\n📈 *Tu historial:* {stats.n_trades} trades, "
                    f"{stats.win_rate * 100:.0f}% win-rate, {stats.avg_profit_pct:+.1f}% promedio"
                )
        except (ValueError, RuntimeError, OSError):
            logger.warning("monitor.analyzer_failed", book=book)

        return body

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
            try:
                await self.scan_once()
            except (httpx.HTTPError, httpx.TimeoutException, ValueError, RuntimeError, ConnectionError, OSError) as exc:
                logger.error("monitor.scan_failed", error=str(exc), error_type=type(exc).__name__)
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

    @property
    def last_prices(self) -> dict[str, float]:
        """Último precio real (último close) por libro."""
        return dict(self._last_prices)
