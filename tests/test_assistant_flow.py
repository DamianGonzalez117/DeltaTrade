"""TDD — Tests del flujo conversacional del TradingAssistant.

Cubre los requisitos del asistente:
  1. Propuesta de compra: cripto, % de éxito, tiempo estimado, saldo,
     y pregunta del monto.
  2. El usuario dice el monto → compra (real o papel) → posición abierta.
  3. Confirmación de venta sí → venta + ganancia + tiempo desde compra.
  4. Confirmación "no" → estadística de espera + re-pregunta; "esperar"
     final → cooldown.
  5. Anti-spam: nada de propuestas para un libro con posición abierta,
     una propuesta global a la vez, cooldowns.

Se mockean sender/history/trader/analyzer/monitor; nada toca red ni DB.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pandas as pd

from trading_system.core.enums import SignalAction
from trading_system.services.signal_generator import RuleSignal
from trading_system.services.trader import FillDetail
from trading_system.services.trading_assistant import (
    AssistantConfig,
    TradingAssistant,
    _format_duration,
    _parse_amount_mxn,
)

BOOK = "btc_mxn"


def _signal(confidence: str = "0.85", action: SignalAction = SignalAction.BUY) -> RuleSignal:
    return RuleSignal(
        action=action,
        confidence=Decimal(confidence),
        rsi=25.0,
        macd_diff=120.0,
        reason="RSI sobreventa (25.0) + MACD cruce alcista",
    )


def _position(
    qty: str = "0.002",
    entry: str = "500000",
    opened_minutes_ago: float = 272.0,
) -> MagicMock:
    pos = MagicMock()
    pos.quantity = Decimal(qty)
    pos.avg_entry_price = Decimal(entry)
    pos.realized_pnl = Decimal(440)
    pos.opened_at = datetime.now(UTC) - timedelta(minutes=opened_minutes_ago)
    return pos


def _uptrend_ohlcv(n: int = 150) -> pd.DataFrame:
    closes = [100.0 + i for i in range(n)]
    return pd.DataFrame({
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [1.0] * n,
    })


class FakeSender:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_message(self, text: str) -> None:
        self.sent.append(text)

    @property
    def last(self) -> str:
        return self.sent[-1]


def _make_assistant(
    *,
    sender: FakeSender | None = None,
    history: MagicMock | None = None,
    trader: MagicMock | None = None,
    analyzer: MagicMock | None = None,
    monitor: MagicMock | None = None,
    config: AssistantConfig | None = None,
) -> tuple[TradingAssistant, FakeSender, MagicMock]:
    sender = sender or FakeSender()
    history = history or MagicMock()
    history.get_open_position = MagicMock(return_value=None)
    history.get_all_open_positions = MagicMock(return_value=[])
    monitor = monitor or MagicMock()
    monitor.last_prices = {BOOK: 500_000.0}
    monitor.timeframe_minutes = 60
    monitor.config_profit_target_pct = 0.05
    monitor.get_last_ohlcv = MagicMock(return_value=_uptrend_ohlcv())
    monitor.last_signals = {BOOK: _signal()}
    assistant = TradingAssistant(
        sender=sender,
        history_service=history,
        trader=trader,
        trade_analyzer=analyzer,
        monitor=monitor,
        config=config or AssistantConfig(),
    )
    return assistant, sender, history


# ----------------------------------------------------------------------
# 1. Propuesta de compra
# ----------------------------------------------------------------------


class TestBuyProposal:
    async def test_proposes_with_crypto_confidence_time_and_balance(self):
        trader = MagicMock()
        trader.get_available = AsyncMock(return_value=Decimal(5000))
        analyzer = MagicMock()
        analyzer.suggest_profit_target_pct = MagicMock(return_value=4.5)
        analyzer.get_closed_stats = MagicMock(return_value=None)

        assistant, sender, _ = _make_assistant(trader=trader, analyzer=analyzer)

        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)

        body = sender.last
        assert BOOK in body
        assert "85.0%" in body  # probabilidad de éxito
        assert "$500,000.00" in body  # precio actual
        assert "Tiempo estimado hasta la venta" in body
        assert "Saldo disponible" in body
        assert "5,000.00" in body
        assert "¿Cuánto quieres comprar" in body
        assert assistant.pending_buy_book == BOOK

    async def test_paper_mode_when_no_trader(self):
        analyzer = MagicMock()
        analyzer.suggest_profit_target_pct = MagicMock(return_value=5.0)
        analyzer.get_closed_stats = MagicMock(return_value=None)

        assistant, sender, _ = _make_assistant(analyzer=analyzer)

        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)

        assert "Modo papel" in sender.last
        assert assistant.pending_buy_book == BOOK

    async def test_no_proposal_when_position_open(self):
        """Regla: si ya compré una cripto, no más mensajes hasta vender."""
        assistant, sender, history = _make_assistant()
        history.get_open_position = MagicMock(return_value=_position())

        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)

        assert sender.sent == []
        assert assistant.pending_buy_book is None

    async def test_no_second_proposal_while_one_pending(self):
        assistant, sender, _ = _make_assistant()

        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)
        await assistant.on_buy_opportunity("eth_mxn", _signal(), 30_000.0)

        assert len(sender.sent) == 1
        assert assistant.pending_buy_book == BOOK

    async def test_no_proposal_within_cooldown_after_buy(self):
        assistant, sender, history = _make_assistant()
        history.confirm_buy = MagicMock(
            side_effect=lambda book, qty, price: _position(qty=str(qty), entry=str(price))
        )

        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)
        assert await assistant.try_handle_reply("500")
        assert len(sender.sent) == 2  # propuesta + compra ejecutada

        # vendemos manualmente para liberar la regla de "posición abierta"
        history.get_open_position = MagicMock(return_value=None)
        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)
        assert len(sender.sent) == 2  # bloqueado por cooldown

    async def test_estimated_time_uses_user_history_when_available(self):
        from trading_system.services.trade_analyzer import TradeStats

        analyzer = MagicMock()
        analyzer.get_closed_stats = MagicMock(return_value=TradeStats(
            n_trades=4, win_rate=0.5, avg_profit_pct=2.0, avg_hold_minutes=272.0
        ))
        analyzer.suggest_profit_target_pct = MagicMock(return_value=4.5)

        assistant, sender, _ = _make_assistant(analyzer=analyzer)

        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)

        assert "~4h 32m" in sender.last


# ----------------------------------------------------------------------
# 2. Respuesta del monto → compra
# ----------------------------------------------------------------------


class TestBuyExecution:
    async def test_amount_answer_executes_real_buy(self):
        trader = MagicMock()
        trader.get_available = AsyncMock(return_value=Decimal(5000))
        trader.min_amount_mxn = Decimal(25)
        fill = FillDetail(
            order_id="ord-1",
            quantity=Decimal("0.00099800"),
            average_price=Decimal("501000.00"),
            total_minor=Decimal("500.00"),
            fee_amount=Decimal("0.00000200"),
            fee_currency="btc",
        )
        trader.buy_market = AsyncMock(return_value=fill)
        analyzer = MagicMock()
        analyzer.suggest_profit_target_pct = MagicMock(return_value=4.5)
        analyzer.get_closed_stats = MagicMock(return_value=None)

        assistant, sender, history = _make_assistant(trader=trader, analyzer=analyzer)
        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)

        consumed = await assistant.try_handle_reply("$1,500.50 mxn")
        assert consumed is True

        trader.buy_market.assert_awaited_once_with(
            BOOK, Decimal("1500.50"), 500_000.0
        )
        history.confirm_buy.assert_called_once_with(
            BOOK, Decimal("0.00099800"), Decimal("501000.00")
        )
        body = sender.last
        assert "COMPRA EJECUTADA" in body
        assert "0.00099800 BTC" in body
        assert "ord" not in body.lower() or fill.order_id not in body  # sin ruido
        assert assistant.pending_buy_book is None

    async def test_amount_below_minimum_keeps_proposal_alive(self):
        assistant, sender, _ = _make_assistant()

        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)
        consumed = await assistant.try_handle_reply("10")

        assert consumed is True
        assert "mínimo" in sender.last
        assert assistant.pending_buy_book == BOOK  # sigue pendiente

    async def test_invalid_amount_asks_again(self):
        assistant, sender, _ = _make_assistant()

        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)
        consumed = await assistant.try_handle_reply("hola mundo")

        assert consumed is True
        assert "No entendí el monto" in sender.last
        assert assistant.pending_buy_book == BOOK

    async def test_no_to_buy_proposal_cancels_it(self):
        assistant, sender, _ = _make_assistant()

        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)
        consumed = await assistant.try_handle_reply("no")

        assert consumed is True
        assert "no quieres comprar" in sender.last
        assert assistant.pending_buy_book is None

    async def test_cancelar_cancels_proposal(self):
        assistant, sender, _ = _make_assistant()

        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)
        consumed = await assistant.try_handle_reply("cancelar")

        assert consumed is True
        assert "Cancelado" in sender.last
        assert assistant.pending_buy_book is None

    async def test_expired_proposal_is_dropped(self):
        config = AssistantConfig(proposal_expiry_minutes=0.0)
        assistant, _sender, _ = _make_assistant(config=config)

        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)
        # el expiry de 0 minutos vence en el siguiente purge
        consumed = await assistant.try_handle_reply("500")

        # venció: no consume como monto (no hay propuesta) → False
        assert consumed is False
        assert assistant.pending_buy_book is None

    async def test_paper_buy_registers_position_with_hint_price(self):
        assistant, sender, history = _make_assistant()
        history.confirm_buy = MagicMock(
            side_effect=lambda book, qty, price: _position(qty=str(qty), entry=str(price))
        )

        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)
        await assistant.try_handle_reply("500")

        qty = history.confirm_buy.call_args[0][1]
        assert qty == Decimal("0.00100000")
        assert "MODO PAPEL" in sender.last

    async def test_insufficient_balance_keeps_proposal_alive(self):
        trader = MagicMock()
        trader.get_available = AsyncMock(return_value=Decimal(5000))
        trader.min_amount_mxn = Decimal(25)
        trader.buy_market = AsyncMock(
            side_effect=ValueError("Saldo insuficiente: tienes $100.00 MXN disponibles")
        )
        assistant, sender, _ = _make_assistant(trader=trader)

        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)
        await assistant.try_handle_reply("500")

        assert "Saldo insuficiente" in sender.last
        assert assistant.pending_buy_book == BOOK


# ----------------------------------------------------------------------
# 3/4. Confirmación de venta: sí / no / esperar
# ----------------------------------------------------------------------


class TestSellConfirmation:
    async def test_ask_sell_confirmation_on_profit_target(self):
        assistant, sender, history = _make_assistant()
        history.get_open_position = MagicMock(return_value=_position())
        history.calc_unrealized_pnl = MagicMock(return_value=Decimal(440))

        await assistant.on_sell_recommendation(BOOK, "profit", 522_500.0, 0.045)

        body = sender.last
        assert "MOMENTO DE VENDER" in body
        assert "¿Quieres vender?" in body
        assert assistant.pending_sell_books == [BOOK]

    async def test_no_duplicate_ask_while_pending(self):
        assistant, sender, history = _make_assistant()
        history.get_open_position = MagicMock(return_value=_position())

        await assistant.on_sell_recommendation(BOOK, "profit", 522_500.0, 0.045)
        await assistant.on_sell_recommendation(BOOK, "loss", 480_000.0, -0.04)

        assert len(sender.sent) == 1

    async def test_si_executes_sell_and_reports_profit_and_time(self):
        assistant, sender, history = _make_assistant()
        position = _position(opened_minutes_ago=272.0)  # 4h 32m
        history.get_open_position = MagicMock(return_value=position)
        closed = _position(opened_minutes_ago=272.0)
        history.close_position = MagicMock(return_value=closed)

        await assistant.on_sell_recommendation(BOOK, "profit", 522_500.0, 0.045)
        consumed = await assistant.try_handle_reply("sí")

        assert consumed is True
        history.close_position.assert_called_once()
        body = sender.last
        assert "VENTA EJECUTADA" in body
        assert "Ganancia" in body
        assert "+$440.00" in body
        assert "4h 32m" in body
        assert assistant.pending_sell_books == []

    async def test_no_sends_stats_and_asks_again(self):
        assistant, sender, history = _make_assistant()
        history.get_open_position = MagicMock(return_value=_position())
        history.close_position = MagicMock(
            return_value=_position(opened_minutes_ago=100.0)
        )
        # sin señal previa: el RSI actual se toma de la serie (tendencia
        # alcista sintética → estadística determinista a favor de esperar)
        assistant._monitor.last_signals = {}

        await assistant.on_sell_recommendation(BOOK, "profit", 522_500.0, 0.045)
        consumed = await assistant.try_handle_reply("no")

        assert consumed is True
        body = sender.last
        assert "CONVIENE ESPERAR" in body
        assert "Subió más" in body
        assert "Bajó" in body
        assert "estadística favorece *esperar*" in body
        assert "¿Vender o esperar?" in body
        # sigue pendiente para la re-pregunta
        assert assistant.pending_sell_books == [BOOK]
        history.close_position.assert_not_called()

    async def test_vender_after_stats_executes_sell(self):
        assistant, sender, history = _make_assistant()
        history.get_open_position = MagicMock(return_value=_position())
        history.close_position = MagicMock(return_value=_position())

        await assistant.on_sell_recommendation(BOOK, "profit", 522_500.0, 0.045)
        await assistant.try_handle_reply("no")
        consumed = await assistant.try_handle_reply("vender")

        assert consumed is True
        assert "VENTA EJECUTADA" in sender.last
        assert assistant.pending_sell_books == []

    async def test_esperar_after_stats_respects_and_cooldowns(self):
        assistant, sender, history = _make_assistant()
        history.get_open_position = MagicMock(return_value=_position())

        await assistant.on_sell_recommendation(BOOK, "profit", 522_500.0, 0.045)
        await assistant.try_handle_reply("no")
        consumed = await assistant.try_handle_reply("esperar")

        assert consumed is True
        assert "esperaré" in sender.last
        assert assistant.pending_sell_books == []

        # dentro del cooldown no vuelve a preguntar
        await assistant.on_sell_recommendation(BOOK, "profit", 522_500.0, 0.045)
        assert len(sender.sent) == 3  # ask + stats + ok-esperar

    async def test_stats_fallback_without_ohlcv(self):
        assistant, sender, history = _make_assistant()
        history.get_open_position = MagicMock(return_value=_position())
        assistant._monitor.get_last_ohlcv = MagicMock(return_value=None)

        await assistant.on_sell_recommendation(BOOK, "profit", 522_500.0, 0.045)
        await assistant.try_handle_reply("no")

        body = sender.last
        assert "suficiente historial" in body
        assert "¿Vender o esperar?" in body

    async def test_sell_api_error_keeps_pending_for_retry(self):
        trader = MagicMock()
        trader.sell_market = AsyncMock(side_effect=RuntimeError("bitso down"))
        assistant, sender, history = _make_assistant(trader=trader)
        history.get_open_position = MagicMock(return_value=_position())

        await assistant.on_sell_recommendation(BOOK, "profit", 522_500.0, 0.045)
        await assistant.try_handle_reply("sí")

        assert "Error al vender" in sender.last
        assert assistant.pending_sell_books == [BOOK]

    async def test_multiple_pendings_require_book(self):
        assistant, sender, history = _make_assistant()
        history.get_open_position = MagicMock(side_effect=lambda book: _position())
        history.calc_unrealized_pnl = MagicMock(return_value=Decimal(10))
        history.close_position = MagicMock(return_value=_position())

        await assistant.on_sell_recommendation("btc_mxn", "profit", 522_500.0, 0.045)
        assistant._monitor.last_prices["eth_mxn"] = 30_000.0
        await assistant.on_sell_recommendation("eth_mxn", "profit", 31_000.0, 0.033)

        consumed = await assistant.try_handle_reply("vender")
        assert consumed is True
        assert "Especifica el libro" in sender.last

        consumed = await assistant.try_handle_reply("vender eth_mxn")
        assert consumed is True
        assert "VENTA EJECUTADA" in sender.last
        assert "eth_mxn" in sender.last


# ----------------------------------------------------------------------
# 5. Enrutamiento de respuestas vs comandos
# ----------------------------------------------------------------------


class TestReplyRouting:
    async def test_commands_never_consumed_as_replies(self):
        assistant, _sender, _ = _make_assistant()
        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)

        assert await assistant.try_handle_reply("posiciones") is False
        assert await assistant.try_handle_reply("estado") is False
        assert await assistant.try_handle_reply("ayuda") is False

    async def test_amount_not_consumed_without_pending_buy(self):
        assistant, _, _ = _make_assistant()

        assert await assistant.try_handle_reply("500") is False

    async def test_cancelar_without_pendings_not_consumed(self):
        assistant, _, _ = _make_assistant()

        assert await assistant.try_handle_reply("cancelar") is False

    async def test_normalized_replies_with_accents(self):
        assistant, sender, history = _make_assistant()
        history.get_open_position = MagicMock(return_value=_position())
        history.close_position = MagicMock(return_value=_position())

        await assistant.on_sell_recommendation(BOOK, "profit", 522_500.0, 0.045)
        consumed = await assistant.try_handle_reply("SÍ")  # mayúscula + acento

        assert consumed is True
        assert "VENTA EJECUTADA" in sender.last

    async def test_clear_for_book_used_by_manual_commands(self):
        assistant, _, _ = _make_assistant()
        await assistant.on_buy_opportunity(BOOK, _signal(), 500_000.0)

        assistant.clear_for_book(BOOK)

        assert assistant.pending_buy_book is None


# ----------------------------------------------------------------------
# Helpers de formato/parsing
# ----------------------------------------------------------------------


class TestHelpers:
    def test_parse_amount_variants(self):
        assert _parse_amount_mxn("500") == Decimal(500)
        assert _parse_amount_mxn("1500.50 mxn") == Decimal("1500.50")
        assert _parse_amount_mxn("1,500") == Decimal(1500)
        assert _parse_amount_mxn("todo") is None
        assert _parse_amount_mxn("-100") is None
        assert _parse_amount_mxn("abc") is None

    def test_format_duration(self):
        assert _format_duration(272) == "4h 32m"
        assert _format_duration(60) == "1h"
        assert _format_duration(45) == "45m"
        assert _format_duration(3000) == "2d 2h"
        assert _format_duration(1440) == "1d"
        assert _format_duration(0) is None
        assert _format_duration(None) is None
