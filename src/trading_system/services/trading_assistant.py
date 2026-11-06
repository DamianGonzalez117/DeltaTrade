"""
Asistente de trading conversacional (Telegram + Bitso).

Flujo completo que implementa:

  COMPRA:
    1. El monitor detecta una señal BUY (sin posición abierta ni propuesta
       pendiente) → mensaje con criptomoneda, % de éxito (confianza),
       tiempo estimado hasta la venta, objetivo y saldo disponible.
    2. Pregunta "¿cuánto quieres comprar?" (monto en MXN).
    3. El usuario responde el monto (o ``todo`` / ``cancelar``) → orden de
       mercado REAL en Bitso por ese monto (``minor``) → posición abierta
       en la DB con cantidad y precio reales de los fills.

  VENTA (objetivo de ganancia, stop loss o señal técnica con posición):
    1. Mensaje pidiendo confirmación: "¿vender? sí/no".
    2. "sí" → orden de mercado REAL de venta (toda la posición) → mensaje
       con la ganancia neta y el tiempo desde la compra.
    3. "no" → estadística de qué pasaría al esperar un poco más
       (``WaitAnalyzer``: % de casos que subieron/bajaron, ganancia y
       pérdida promedio, mejor/peor caso, tendencia y recomendación) →
       se pregunta de nuevo "¿vender o esperar?".
    4. "esperar" → se respeta la decisión y se activa un cooldown para no
       molestar con ese libro hasta que la situación cambie.

  ANTI-SPAM (regla: "si ya compré una cripto, no más mensajes de esa
  cripto hasta vender"):
    - Nunca se propone comprar un libro con posición abierta.
    - Solo una propuesta de compra global a la vez (expira sola).
    - Cooldown por libro tras comprar/cancelar/expirar una propuesta.
    - Una confirmación de venta pendiente por libro; re-ask bloqueado por
      cooldown tras "esperar".

DIP: ``sender`` (TelegramCommandBot), ``SignalHistoryService``,
``BitsoTrader`` (None = modo papel), ``TradeAnalyzer``, ``MarketMonitor``
y ``WaitAnalyzer`` se inyectan.
SRP: orquesta la conversación. La estadística la calcula WaitAnalyzer y
las órdenes las coloca BitsoTrader.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

import httpx
import structlog

from trading_system.services.trader import FillDetail
from trading_system.services.wait_analyzer import WaitAnalyzer

if TYPE_CHECKING:
    from trading_system.services.monitor import MarketMonitor
    from trading_system.services.signal_generator import RuleSignal
    from trading_system.services.signal_history import SignalHistoryService
    from trading_system.services.trade_analyzer import TradeAnalyzer
    from trading_system.services.trader import BitsoTrader

logger = structlog.get_logger(__name__)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

_MIN_PAPER_AMOUNT_MXN = Decimal(25)

_YES_WORDS = {"si", "yes", "vender", "vende", "vendo", "confirmo", "ok", "ya"}
_NO_WORDS = {
    "no",
    "esperar",
    "espera",
    "espero",
    "wait",
    "mas",
    "luego",
    "despues",
    "todavia",
}
_ALL_WORDS = {"todo", "toda", "all"}
_CANCEL_WORDS = {"cancelar", "cancel", "cancela", "olvidalo"}

_COMMAND_WORDS = {
    "compré", "compre", "compro", "buy", "c",
    "vendí", "vendi", "sell", "v",
    "posiciones", "positions", "p",
    "historial", "history", "h",
    "estado", "status", "s",
    "ayuda", "help", "?",
}


@dataclass(frozen=True, slots=True)
class AssistantConfig:
    """Tiempos y parámetros del ciclo de conversación."""

    proposal_expiry_minutes: float = 10.0  # la propuesta de compra expira si no responde
    buy_cooldown_minutes: float = 30.0  # tras comprar/cancelar/expirar, no re-proponer
    sell_reask_cooldown_minutes: float = 20.0  # tras "esperar", no volver a preguntar
    sell_pending_expiry_minutes: float = 30.0  # confirmación de venta sin responder
    horizon_candles: int = 6  # velas mínimas de la estadística de espera (adaptativa hasta 48)


@dataclass
class _PendingBuy:
    """Propuesta de compra activa (una sola globalmente)."""

    book: str
    price: float
    confidence: Decimal
    reason: str
    est_hold_minutes: float | None
    expires_at: datetime


@dataclass
class _PendingSell:
    """Confirmación de venta pendiente para un libro."""

    book: str
    trigger: str  # "profit" | "loss" | "signal"
    deferred: bool = False  # True tras responder "no" (ya se enviaron stats)
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class TradingAssistant:
    """Conversación de compra/venida con confirmación humana vía Telegram."""

    def __init__(
        self,
        *,
        sender,
        history_service: SignalHistoryService,
        trader: BitsoTrader | None = None,
        trade_analyzer: TradeAnalyzer | None = None,
        monitor: MarketMonitor | None = None,
        wait_analyzer: WaitAnalyzer | None = None,
        config: AssistantConfig | None = None,
    ) -> None:
        self._sender = sender  # objeto con `async def send_message(text)`
        self._history = history_service
        self._trader = trader  # None → modo papel (sin órdenes reales)
        self._analyzer = trade_analyzer
        self._monitor = monitor
        self._config = config or AssistantConfig()
        self._wait = wait_analyzer or WaitAnalyzer(
            horizon_candles=self._config.horizon_candles
        )
        self._pending_buy: _PendingBuy | None = None
        self._pending_sells: dict[str, _PendingSell] = {}
        self._buy_cooldown_until: dict[str, datetime] = {}
        self._sell_cooldown_until: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def live_trading(self) -> bool:
        """True cuando hay trader real inyectado (órdenes en Bitso)."""
        return self._trader is not None

    @property
    def pending_buy_book(self) -> str | None:
        return self._pending_buy.book if self._pending_buy else None

    @property
    def pending_sell_books(self) -> list[str]:
        return list(self._pending_sells)

    # ------------------------------------------------------------------
    # API para el monitor (eventos de señal/umbral)
    # ------------------------------------------------------------------

    async def on_buy_opportunity(
        self,
        book: str,
        signal: RuleSignal,
        last_price: float,
    ) -> None:
        """Señal BUY detectada: propone la compra si todos los filtros pasan.

        Filtros anti-spam: posición abierta en el libro, propuesta pendiente
        global, cooldown del libro, caducidad de propuestas previas.
        """
        self.purge_expired()

        if self._history.get_open_position(book) is not None:
            logger.info("assistant.skip_buy_has_position", book=book)
            return
        if self._pending_buy is not None:
            logger.info("assistant.skip_buy_pending_exists", book=book)
            return
        if self._now() < self._buy_cooldown_until.get(book, _EPOCH):
            logger.info("assistant.skip_buy_cooldown", book=book)
            return

        base = book.split("_")[0].upper()
        target_pct = self._suggest_target_pct(book)
        target_price = last_price * (1 + target_pct / 100)
        est_hold = self._estimate_hold_minutes(book, target_pct)

        body = (
            f"🟢 *OPORTUNIDAD DE COMPRA — {base} ({book})*\n"
            f"Probabilidad de éxito: {float(signal.confidence) * 100:.1f}%\n"
            f"Precio actual: ${last_price:,.2f}\n"
            f"Razón: {signal.reason}\n"
            f"🎯 Objetivo: ${target_price:,.2f} (+{target_pct:.1f}%)"
        )
        if est_hold is not None:
            body += f"\n⏱ Tiempo estimado hasta la venta: ~{_format_duration(est_hold)}"
        body += "\n\n" + await self._funds_line(book, last_price)
        body += (
            "\n\n¿Cuánto quieres comprar (MXN)?\n"
            "Responde con el monto (ej: `500`), `todo` para usar todo tu saldo "
            "o `cancelar`."
        )

        self._pending_buy = _PendingBuy(
            book=book,
            price=last_price,
            confidence=signal.confidence,
            reason=signal.reason,
            est_hold_minutes=est_hold,
            expires_at=self._now() + timedelta(minutes=self._config.proposal_expiry_minutes),
        )
        await self._send(body)
        logger.info(
            "assistant.buy_proposed",
            book=book,
            confidence=str(signal.confidence),
            est_hold_minutes=est_hold,
        )

    async def on_sell_recommendation(
        self,
        book: str,
        trigger: str,
        last_price: float,
        pnl_pct: float,
    ) -> None:
        """Momento de vender (objetivo/stop/señal): pide confirmación."""
        self.purge_expired()

        position = self._history.get_open_position(book)
        if position is None:
            return
        if book in self._pending_sells:
            logger.info("assistant.skip_sell_pending", book=book)
            return
        if self._now() < self._sell_cooldown_until.get(book, _EPOCH):
            logger.info("assistant.skip_sell_cooldown", book=book)
            return

        entry = float(position.avg_entry_price)
        pnl = float(
            self._history.calc_unrealized_pnl(position, Decimal(str(last_price)))
        )

        if trigger == "profit":
            header = f"🎯 *MOMENTO DE VENDER — {book}*\nObjetivo de ganancia alcanzado"
        elif trigger == "loss":
            header = f"⚠️ *STOP LOSS — {book}*\nPérdida por debajo del límite"
        else:
            header = f"🔴 *SEÑAL DE VENTA — {book}*\nIndicadores técnicos bajistas"

        emoji = "🟢" if pnl >= 0 else "🔴"
        body = (
            f"{header}\n\n"
            f"Tu posición: {position.quantity} @ ${entry:,.2f}\n"
            f"Precio actual: ${last_price:,.2f} ({pnl_pct * 100:+.1f}%)\n"
            f"P&L estimado: ${pnl:,.2f} {emoji}\n\n"
            f"¿Quieres vender? Responde *sí* o *no*."
        )

        self._pending_sells[book] = _PendingSell(
            book=book,
            trigger=trigger,
            expires_at=self._now()
            + timedelta(minutes=self._config.sell_pending_expiry_minutes),
        )
        await self._send(body)
        logger.info("assistant.sell_confirmation_asked", book=book, trigger=trigger)

    def purge_expired(self) -> None:
        """Descarta propuestas/confirmaciones vencidas y activa cooldowns."""
        now = self._now()
        if self._pending_buy is not None and now >= self._pending_buy.expires_at:
            book = self._pending_buy.book
            self._buy_cooldown_until[book] = now + timedelta(
                minutes=self._config.buy_cooldown_minutes
            )
            self._pending_buy = None
            logger.info("assistant.buy_proposal_expired", book=book)

        for book, pending in list(self._pending_sells.items()):
            if now >= pending.expires_at:
                self._pending_sells.pop(book, None)
                self._sell_cooldown_until[book] = now + timedelta(
                    minutes=self._config.sell_reask_cooldown_minutes
                )
                logger.info("assistant.sell_pending_expired", book=book)

    # ------------------------------------------------------------------
    # API para el bot (respuestas del usuario)
    # ------------------------------------------------------------------

    async def try_handle_reply(self, text: str) -> bool:
        """Procesa ``text`` como respuesta a una conversación pendiente.

        Retorna True si el mensaje fue consumido (era una respuesta válida
        para una propuesta/confirmación pendiente) y no debe tratarse como
        comando. Los comandos clásicos nunca se consumen.
        """
        self.purge_expired()
        normalized = _normalize(text)

        if not normalized or normalized.split()[0] in _COMMAND_WORDS:
            return False

        tokens = normalized.split()

        # "cancelar" cancela cualquier conversación pendiente.
        if tokens[0] in _CANCEL_WORDS:
            if self._pending_buy is not None or self._pending_sells:
                await self._handle_cancel_all()
                return True
            return False

        # Las respuestas de venta (sí/no/vender/esperar) tienen prioridad
        # cuando hay confirmaciones pendientes.
        if self._pending_sells and _sell_intent(normalized) is not None:
            return await self._handle_sell_answer(normalized)

        # Propuesta de compra pendiente: cualquier texto que no sea comando
        # se interpreta como respuesta al "¿cuánto quieres comprar?".
        if self._pending_buy is not None:
            if _looks_like_amount(normalized):
                await self._handle_buy_amount(normalized)
                return True
            if _sell_intent(normalized) == "esperar":
                # "no" a la propuesta → cancelar la compra
                book = self._pending_buy.book
                self._pending_buy = None
                self._set_buy_cooldown(book)
                await self._send(
                    f"👌 Entiendo que no quieres comprar {book} ahora. "
                    f"Te aviso cuando haya otra oportunidad."
                )
                return True
            await self._send(
                "No entendí el monto. Respóndeme solo con el número en MXN "
                "(ej: `500` o `1500.50`), `todo`, o `cancelar`."
            )
            return True

        return False

    async def _handle_cancel_all(self) -> None:
        """Cancela la propuesta de compra y las confirmaciones pendientes."""
        cancelled: list[str] = []
        if self._pending_buy is not None:
            book = self._pending_buy.book
            self._pending_buy = None
            self._set_buy_cooldown(book)
            cancelled.append(f"compra de {book}")
        for book in list(self._pending_sells):
            self._pending_sells.pop(book, None)
            self._set_sell_cooldown(book)
            cancelled.append(f"venta de {book}")
        await self._send(
            f"👌 Cancelado: {', '.join(cancelled)}.\n"
            f"Te aviso cuando haya algo nuevo."
        )

    async def _handle_buy_amount(self, normalized: str) -> None:
        """El usuario respondió cuánto comprar (o todo)."""
        pending = self._pending_buy
        if pending is None:
            return

        amount = await self._resolve_amount(normalized, pending.book)
        if amount is None:
            return  # ya se envió el mensaje de error/uso; la propuesta sigue viva

        await self._execute_buy(pending, amount)

    async def _execute_buy(self, pending: _PendingBuy, amount_mxn: Decimal) -> None:
        """Coloca la orden de compra real (o simulada) y registra la posición."""
        book = pending.book
        base = book.split("_")[0].upper()

        if self._history.get_open_position(book) is not None:
            self._pending_buy = None
            await self._send(
                f"⚠️ Ya tienes una posición abierta en {book}; no compres más "
                f"hasta venderla."
            )
            return

        price_hint = self._last_price(book) or pending.price
        try:
            if self._trader is not None:
                fill = await self._trader.buy_market(book, amount_mxn, price_hint)
            else:
                fill = _paper_buy(amount_mxn, price_hint)
        except ValueError as exc:
            await self._send(f"❌ {exc}\n¿Otro monto? (o `cancelar`)")
            return
        except (httpx.HTTPError, httpx.TimeoutException, RuntimeError, OSError) as exc:
            logger.error("assistant.buy_failed", book=book, error=str(exc))
            await self._send(
                f"❌ Error al colocar la orden en Bitso: {exc}\n"
                f"Puedes reintentar con otro monto o `cancelar`."
            )
            return

        try:
            position = self._history.confirm_buy(book, fill.quantity, fill.average_price)
        except ValueError as exc:
            self._pending_buy = None
            await self._send(
                f"⚠️ La orden se ejecutó en Bitso pero no pude registrar la "
                f"posición: {exc}\nRevisa con `posiciones`."
            )
            return

        self._pending_buy = None
        self._set_buy_cooldown(book)

        mode = "" if self.live_trading else "🧪 *MODO PAPEL* (sin dinero real)\n"
        body = (
            f"✅ *COMPRA EJECUTADA — {book}*\n{mode}"
            f"Compré: {fill.quantity} {base} @ ${fill.average_price:,.2f}\n"
            f"Gastado: ${fill.total_minor:,.2f} MXN"
        )
        if fill.fee_amount > 0:
            body += f"\nComisión: {fill.fee_amount} {fill.fee_currency.upper()}"
        if not self.live_trading or fill.used_fallback:
            body += "\n_(precio estimado, sin confirmación de fills)_"
        body += f"\nID posición: {position.id}"
        body += self._target_line(book, float(fill.average_price))
        body += (
            f"\n\n⏱ Te avisaré cuando sea momento de vender (estimado: "
            f"{_format_duration(pending.est_hold_minutes) or 'por definir'})."
        )
        await self._send(body)
        logger.info(
            "assistant.buy_executed",
            book=book,
            qty=str(fill.quantity),
            avg_price=str(fill.average_price),
            order_id=fill.order_id,
        )

    async def _handle_sell_answer(self, normalized: str) -> bool:
        """Respuesta a una confirmación de venta. True si fue consumida."""
        tokens = normalized.split()
        intent = _sell_intent(normalized)
        book_arg = tokens[1] if len(tokens) > 1 else None

        if book_arg is not None:
            if book_arg in self._pending_sells:
                book = book_arg
            else:
                await self._send(
                    f"No tengo confirmación de venta pendiente para {book_arg}. "
                    f"Pendientes: {', '.join(self._pending_sells)}"
                )
                return True
        elif len(self._pending_sells) == 1:
            book = next(iter(self._pending_sells))
        else:
            await self._send(
                "Tengo varias confirmaciones pendientes. Especifica el libro, "
                f"ej: `{tokens[0]} {' '.join(self._pending_sells)}`."
            )
            return True

        if intent == "vender":
            await self._execute_sell(book)
        else:
            await self._defer_sell(book)
        return True

    async def _execute_sell(self, book: str) -> None:
        """Vende la posición (orden real o simulada) y reporta el resultado."""
        pending = self._pending_sells.get(book)
        position = self._history.get_open_position(book)
        if position is None:
            if pending is not None:
                self._pending_sells.pop(book, None)
                self._set_sell_cooldown(book)
            await self._send(f"📭 Ya no hay posición abierta en {book}.")
            return
        if pending is None:
            # defensive: expiró justo entre el mensaje y la respuesta
            pending = _PendingSell(book=book, trigger="manual")
            self._pending_sells[book] = pending

        base = book.split("_")[0].upper()
        entry = float(position.avg_entry_price)
        price_hint = self._last_price(book) or entry

        try:
            if self._trader is not None:
                fill = await self._trader.sell_market(book, position.quantity, price_hint)
            else:
                fill = _paper_sell(position.quantity, price_hint)
        except (httpx.HTTPError, httpx.TimeoutException, RuntimeError, OSError, ValueError) as exc:
            # devolvemos la confirmación para que pueda reintentar con "vender"
            self._pending_sells[book] = pending
            logger.error("assistant.sell_failed", book=book, error=str(exc))
            await self._send(
                f"❌ Error al vender en Bitso: {exc}\n"
                f"Reintenta respondiendo *vender* en un momento."
            )
            return

        try:
            closed = self._history.close_position(book, fill.average_price)
        except ValueError as exc:
            await self._send(f"❌ {exc}")
            return

        self._pending_sells.pop(book, None)
        self._set_sell_cooldown(book)

        pnl = closed.realized_pnl
        cost = closed.avg_entry_price * closed.quantity
        pnl_pct = float(pnl / cost * 100) if cost > 0 else 0.0
        held_minutes = (self._now() - closed.opened_at).total_seconds() / 60.0

        mode = "" if self.live_trading else "🧪 *MODO PAPEL* (sin dinero real)\n"
        emoji = "🟢" if pnl >= 0 else "🔴"
        result_word = "Ganancia" if pnl >= 0 else "Resultado"
        body = (
            f"✅ *VENTA EJECUTADA — {book}*\n{mode}"
            f"Vendí: {closed.quantity} {base} @ ${fill.average_price:,.2f}\n"
            f"Recibido: ${fill.total_minor:,.2f} MXN\n\n"
            f"💰 *{result_word}: {'+' if pnl >= 0 else '-'}${abs(pnl):,.2f} "
            f"({pnl_pct:+.1f}%) {emoji}*\n"
            f"⏱ Tiempo desde la compra: {_format_duration(held_minutes)}"
        )
        if not self.live_trading or fill.used_fallback:
            body += "\n_(precio estimado, sin confirmación de fills)_"
        body += self._stats_line(book)
        await self._send(body)
        logger.info(
            "assistant.sell_executed",
            book=book,
            pnl=str(pnl),
            held_minutes=round(held_minutes, 1),
            order_id=fill.order_id,
        )

    async def _defer_sell(self, book: str) -> None:
        """Respondió "no": envía estadística de espera y pregunta de nuevo.

        Si ya se habían enviado las estadísticas (segundo "no"), se respeta
        la decisión de esperar y se activa el cooldown del libro.
        """
        pending = self._pending_sells.get(book)
        if pending is None:
            await self._send(f"No tengo confirmación de venta pendiente para {book}.")
            return

        if pending.deferred:
            self._pending_sells.pop(book, None)
            self._set_sell_cooldown(book)
            await self._send(
                f"👌 Ok, esperaré con {book}. No te volveré a preguntar por un "
                f"rato; te aviso cuando la situación cambie (objetivo, stop o "
                f"señal)."
            )
            return

        body = await self._wait_stats_body(book)
        await self._send(body)
        pending.deferred = True
        pending.expires_at = self._now() + timedelta(
            minutes=self._config.sell_pending_expiry_minutes
        )
        logger.info("assistant.sell_deferred", book=book)

    # ------------------------------------------------------------------
    # Estadística de espera
    # ------------------------------------------------------------------

    async def _wait_stats_body(self, book: str) -> str:
        """Cuerpo del mensaje "¿conviene esperar?" con la estadística."""
        position = self._history.get_open_position(book)
        ohlcv = self._monitor.get_last_ohlcv(book) if self._monitor else None
        timeframe_minutes = self._monitor.timeframe_minutes if self._monitor else 60
        current_rsi = None
        if self._monitor is not None:
            signal = self._monitor.last_signals.get(book)
            if signal is not None:
                current_rsi = signal.rsi

        stats = self._wait.analyze(
            ohlcv,
            current_rsi=current_rsi,
            timeframe_minutes=timeframe_minutes,
        )

        header = f"📊 *¿CONVIENE ESPERAR? — {book}*"
        if position is not None:
            entry = float(position.avg_entry_price)
            price = self._last_price(book) or entry
            pnl_pct = (price - entry) / entry * 100 if entry > 0 else 0.0
            header += (
                f"\nPosición: {position.quantity} @ ${entry:,.2f} "
                f"({pnl_pct:+.1f}% ahora)"
            )

        if stats is None:
            return (
                f"{header}\n\n"
                f"No tengo suficiente historial para darte una estadística "
                f"fiable de qué pasaría al esperar.\n"
                f"Lo que sí sé: la señal que disparó este aviso sigue "
                f"activa, y mientras más esperas más expuesto estás a un "
                f"retroceso.\n\n"
                f"¿Vender o esperar? (*vender* / *esperar*)"
            )

        if stats.recommendation == "esperar":
            advice = "💡 La estadística favorece *esperar* un poco más."
        elif stats.recommendation == "vender":
            advice = "💡 La estadística favorece *vender* ahora."
        else:
            advice = "💡 La estadística es mixta: ni esperar ni vender claramente."

        return (
            f"{header}\n\n"
            f"Si esperas ~{_format_duration(stats.horizon_minutes)} "
            f"({stats.horizon_candles} velas de {timeframe_minutes}m), en "
            f"{stats.n_cases} casos históricos con RSI similar:\n\n"
            f"• Subió más: {stats.rise_share * 100:.0f}% de los casos "
            f"({stats.avg_gain_pct:+.1f}% adicional en promedio)\n"
            f"• Bajó: {stats.fall_share * 100:.0f}% de los casos "
            f"({stats.avg_loss_pct:+.1f}% en promedio)\n"
            f"• Máximo histórico alcanzado: {stats.best_case_pct:+.1f}% | "
            f"Caída máxima: {stats.worst_case_pct:+.1f}%\n\n"
            f"Tendencia actual: {stats.trend}\n"
            f"{advice}\n\n"
            f"¿Vender o esperar? (*vender* / *esperar*)"
        )

    # ------------------------------------------------------------------
    # Utilidades de montos y tiempos
    # ------------------------------------------------------------------

    async def _resolve_amount(self, normalized: str, book: str) -> Decimal | None:
        """Convierte la respuesta en un monto MXN; None si es inválido."""
        if normalized in _ALL_WORDS:
            if self._trader is None:
                await self._send(
                    "🧪 En modo papel no puedo consultar tu saldo; respóndeme "
                    "con un monto fijo (ej: `500`)."
                )
                return None
            try:
                available = await self._trader.get_available("mxn")
            except (httpx.HTTPError, httpx.TimeoutException, RuntimeError, OSError) as exc:
                await self._send(f"❌ No pude consultar tu saldo: {exc}\n¿Otro monto?")
                return None
            if available < _MIN_PAPER_AMOUNT_MXN:
                await self._send(
                    f"❌ Solo tienes ${available:,.2f} MXN disponibles; no alcanza "
                    f"para el mínimo."
                )
                return None
            return available

        amount = _parse_amount_mxn(normalized)
        if amount is None or amount <= 0:
            await self._send(
                "No entendí el monto. Respóndeme solo con el número en MXN "
                "(ej: `500` o `1500.50`), `todo`, o `cancelar`."
            )
            return None
        minimum = (
            self._trader.min_amount_mxn if self._trader is not None else _MIN_PAPER_AMOUNT_MXN
        )
        if amount < minimum:
            await self._send(
                f"❌ El monto mínimo es ${minimum:,.0f} MXN. ¿Otro monto? (o `cancelar`)"
            )
            return None
        return amount

    def _suggest_target_pct(self, book: str) -> float:
        """Objetivo de venta % sugerido (analyzer o el del monitor)."""
        default = 5.0
        if self._monitor is not None:
            default = self._monitor.config_profit_target_pct * 100
        if self._analyzer is None:
            return default
        try:
            return self._analyzer.suggest_profit_target_pct(book, default_pct=default)
        except (ValueError, RuntimeError, OSError):
            return default

    def _estimate_hold_minutes(self, book: str, target_pct: float) -> float | None:
        """Tiempo estimado hasta la venta.

        1. Historial propio del usuario en ese libro (tiempo promedio de
           tenencia de trades cerrados).
        2. Heurística: objetivo % / movimiento promedio por vela × minutos
           por vela del timeframe del monitor.
        """
        if self._analyzer is not None:
            try:
                stats = self._analyzer.get_closed_stats(book)
                if stats is not None and stats.n_trades >= 2:
                    return stats.avg_hold_minutes
            except (ValueError, RuntimeError, OSError):
                pass

        ohlcv = self._monitor.get_last_ohlcv(book) if self._monitor else None
        if ohlcv is None or ohlcv.empty:
            return None
        returns = ohlcv["close"].pct_change().dropna().abs().tail(60)
        if returns.empty:
            return None
        avg_move = float(returns.mean())
        if avg_move <= 0:
            return None
        timeframe_minutes = self._monitor.timeframe_minutes if self._monitor else 60
        candles = max(1, round((target_pct / 100) / avg_move))
        return candles * timeframe_minutes

    def _target_line(self, book: str, entry_price: float) -> str:
        if self._analyzer is None:
            return ""
        try:
            target_pct = self._analyzer.suggest_profit_target_pct(book, default_pct=5.0)
            target_price = entry_price * (1 + target_pct / 100)
            return f"\n🎯 Objetivo de venta: ${target_price:,.2f} (+{target_pct:.1f}%)"
        except (ValueError, RuntimeError, OSError):
            return ""

    def _stats_line(self, book: str) -> str:
        if self._analyzer is None:
            return ""
        try:
            stats = self._analyzer.get_closed_stats(book)
            if stats is None:
                return ""
            return (
                f"\n📈 Tu historial: {stats.n_trades} trades, "
                f"{stats.win_rate * 100:.0f}% win-rate, "
                f"{stats.avg_profit_pct:+.1f}% promedio"
            )
        except (ValueError, RuntimeError, OSError):
            return ""

    async def _funds_line(self, book: str, price: float) -> str:
        """Línea de saldo disponible (o aviso de modo papel)."""
        if self._trader is None:
            return "🧪 *Modo papel*: no hay llaves de Bitso, las compras solo se simulan."
        try:
            available = await self._trader.get_available("mxn")
        except (httpx.HTTPError, httpx.TimeoutException, RuntimeError, OSError):
            return "💰 No pude consultar tu saldo en este momento."
        base = book.split("_")[0].upper()
        approx_qty = available / Decimal(str(price)) if price > 0 else Decimal(0)
        return (
            f"💰 Saldo disponible: ${available:,.2f} MXN "
            f"(≈ {approx_qty:.6f} {base})"
        )

    def _last_price(self, book: str) -> float | None:
        if self._monitor is not None:
            return self._monitor.last_prices.get(book)
        return None

    def _set_buy_cooldown(self, book: str) -> None:
        self._buy_cooldown_until[book] = self._now() + timedelta(
            minutes=self._config.buy_cooldown_minutes
        )

    def _set_sell_cooldown(self, book: str) -> None:
        self._sell_cooldown_until[book] = self._now() + timedelta(
            minutes=self._config.sell_reask_cooldown_minutes
        )

    def clear_for_book(self, book: str) -> None:
        """Limpia conversaciones pendientes de un libro (venta manual, etc.)."""
        if self._pending_buy is not None and self._pending_buy.book == book:
            self._pending_buy = None
        self._pending_sells.pop(book, None)

    async def _send(self, text: str) -> None:
        await self._sender.send_message(text)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)


# ----------------------------------------------------------------------
# Helpers de parsing/formato
# ----------------------------------------------------------------------


def _normalize(text: str) -> str:
    """minúsculas + sin acentos, para que "Sí"/"mas"/"después" coincidan."""
    stripped = text.strip().lower()
    replacements = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u"}
    return "".join(replacements.get(ch, ch) for ch in stripped)


_AMOUNT_RE = re.compile(r"\d+(?:\.\d+)?")


def _parse_amount_mxn(normalized: str) -> Decimal | None:
    """Parsea "500", "$1,500.50 mxn" → Decimal; None si no es un monto."""
    text = normalized.replace(" ", "")
    text = text.removesuffix("mxn")
    text = text.replace(",", "").replace("$", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?", text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _looks_like_amount(normalized: str) -> bool:
    if normalized in _ALL_WORDS or normalized in _CANCEL_WORDS:
        return True
    return _parse_amount_mxn(normalized) is not None


def _sell_intent(normalized: str) -> str | None:
    """Clasifica la respuesta como intención de vender o esperar."""
    first = normalized.split()[0] if normalized else ""
    if first in _YES_WORDS:
        return "vender"
    if first in _NO_WORDS:
        return "esperar"
    return None


def _format_duration(minutes: float | None) -> str | None:
    """272 min → "4h 32m"; 3000 min → "2d 2h"; 45 min → "45m"."""
    if minutes is None or minutes <= 0:
        return None
    total = round(minutes)
    days, rest = divmod(total, 1440)
    hours, mins = divmod(rest, 60)
    if days:
        return f"{days}d {hours}h" if hours else f"{days}d"
    if hours:
        return f"{hours}h {mins:02d}m" if mins else f"{hours}h"
    return f"{mins}m"


def _paper_buy(amount_mxn: Decimal, price_hint: float) -> FillDetail:
    avg = Decimal(str(price_hint))
    qty = (amount_mxn / avg).quantize(Decimal("0.00000001"))
    return FillDetail(
        order_id="paper",
        quantity=qty,
        average_price=avg,
        total_minor=amount_mxn,
        fee_amount=Decimal(0),
        fee_currency="",
        used_fallback=True,
    )


def _paper_sell(quantity: Decimal, price_hint: float) -> FillDetail:
    avg = Decimal(str(price_hint))
    return FillDetail(
        order_id="paper",
        quantity=quantity,
        average_price=avg,
        total_minor=(avg * quantity).quantize(Decimal("0.00000001")),
        fee_amount=Decimal(0),
        fee_currency="",
        used_fallback=True,
    )
