"""TDD - Tests for TelegramCommandBot (RED->GREEN).

Se mockea el cliente HTTP y el SignalHistoryService.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from trading_system.core.enums import SignalAction
from trading_system.services.signal_generator import RuleSignal
from trading_system.services.telegram_bot import TelegramCommandBot


def _mock_handler(responses: list[httpx.Request]):
    """Crea un MockTransport que captura mensajes y retorna OK."""
    def handler(request: httpx.Request) -> httpx.Response:
        responses.append(request)
        return httpx.Response(200, json={"ok": True, "result": []})
    return httpx.MockTransport(handler)


@pytest.fixture
def mock_history():
    history = MagicMock()
    history.confirm_buy = MagicMock()
    history.close_position = MagicMock()
    history.get_all_open_positions = MagicMock(return_value=[])
    history.get_signal_history = MagicMock(return_value=[])
    history.get_open_position = MagicMock(return_value=None)
    history.calc_unrealized_pnl = MagicMock(return_value=Decimal(0))
    return history


@pytest.fixture
def mock_dispatcher():
    d = MagicMock()
    d.dispatch = AsyncMock(return_value={"telegram": True})
    return d


@pytest.fixture
def sent_messages():
    return []


@pytest.fixture
def bot(mock_history, mock_dispatcher, sent_messages):
    transport = _mock_handler(sent_messages)
    client = httpx.AsyncClient(transport=transport)
    return TelegramCommandBot(
        bot_token="test-token",
        chat_id="123456",
        history_service=mock_history,
        dispatcher=mock_dispatcher,
        client=client,
    )


def _last_body(sent_messages: list[httpx.Request]) -> str:
    """Extrae el cuerpo del ultimo mensaje enviado."""
    return sent_messages[-1].content.decode()


class TestCommandParsing:
    async def test_help_command(self, bot, sent_messages):
        await bot._dispatch_command("ayuda")
        assert len(sent_messages) >= 1
        assert "Comandos" in _last_body(sent_messages) or "comandos" in _last_body(sent_messages)

    async def test_unknown_command(self, bot, sent_messages):
        await bot._dispatch_command("blah blah")
        assert "no reconocido" in _last_body(sent_messages).lower()

    async def test_buy_command_calls_confirm(self, bot, mock_history, sent_messages):
        mock_position = MagicMock()
        mock_position.id = 1
        mock_position.quantity = Decimal("0.01")
        mock_position.avg_entry_price = Decimal(505000)
        mock_history.confirm_buy.return_value = mock_position

        await bot._dispatch_command("compré btc_mxn 0.01 505000")

        mock_history.confirm_buy.assert_called_once_with(
            "btc_mxn", Decimal("0.01"), Decimal(505000)
        )
        assert "compra" in _last_body(sent_messages).lower()

    async def test_buy_command_with_buy_alias(self, bot, mock_history):
        mock_position = MagicMock()
        mock_position.id = 1
        mock_position.quantity = Decimal("0.01")
        mock_position.avg_entry_price = Decimal(505000)
        mock_history.confirm_buy.return_value = mock_position

        await bot._dispatch_command("buy btc_mxn 0.01 505000")
        mock_history.confirm_buy.assert_called_once()

    async def test_buy_invalid_format(self, bot, sent_messages):
        await bot._dispatch_command("compré btc_mxn")
        assert "Formato" in _last_body(sent_messages)

    async def test_buy_value_error_from_service(self, bot, mock_history, sent_messages):
        mock_history.confirm_buy.side_effect = ValueError("Ya hay una posicion")

        await bot._dispatch_command("compré btc_mxn 0.01 505000")
        assert "Ya hay" in _last_body(sent_messages)

    async def test_sell_command_calls_close(self, bot, mock_history, sent_messages):
        mock_position = MagicMock()
        mock_position.id = 1
        mock_position.realized_pnl = Decimal(150)
        mock_history.close_position.return_value = mock_position

        await bot._dispatch_command("vendí btc_mxn 520000")

        mock_history.close_position.assert_called_once_with(
            "btc_mxn", Decimal(520000)
        )
        assert "venta" in _last_body(sent_messages).lower()

    async def test_sell_with_v_alias(self, bot, mock_history):
        mock_position = MagicMock()
        mock_position.id = 1
        mock_position.realized_pnl = Decimal(150)
        mock_history.close_position.return_value = mock_position

        await bot._dispatch_command("v btc_mxn 520000")
        mock_history.close_position.assert_called_once()

    async def test_sell_invalid_format(self, bot, sent_messages):
        await bot._dispatch_command("vendí btc_mxn")
        assert "Formato" in _last_body(sent_messages)

    async def test_sell_value_error(self, bot, mock_history, sent_messages):
        mock_history.close_position.side_effect = ValueError("No hay posicion")

        await bot._dispatch_command("vendí btc_mxn 520000")
        assert "No hay" in _last_body(sent_messages)

    async def test_positions_empty(self, bot, sent_messages):
        await bot._cmd_positions()
        assert "No tienes" in _last_body(sent_messages)

    async def test_positions_with_data(self, bot, mock_history, sent_messages):
        mock_pos = MagicMock()
        mock_pos.asset.symbol = "btc_mxn"
        mock_pos.quantity = Decimal("0.01")
        mock_pos.avg_entry_price = Decimal(505000)
        mock_pos.opened_at.strftime.return_value = "2024-01-01 10:00"
        mock_history.get_all_open_positions.return_value = [mock_pos]

        await bot._cmd_positions()
        body = _last_body(sent_messages)
        assert "btc_mxn" in body
        assert "Posiciones" in body

    async def test_history_empty(self, bot, sent_messages):
        await bot._cmd_history([])
        assert "registradas" in _last_body(sent_messages)

    async def test_history_with_data(self, bot, mock_history, sent_messages):
        mock_sig = MagicMock()
        mock_sig.generated_at.strftime.return_value = "01-01 10:00"
        mock_sig.asset.symbol = "btc_mxn"
        mock_sig.action = SignalAction.BUY
        mock_sig.confidence = Decimal("0.85")
        mock_history.get_signal_history.return_value = [mock_sig]

        await bot._cmd_history(["historial"])
        body = _last_body(sent_messages)
        assert "btc_mxn" in body
        assert "Historial" in body

    async def test_history_with_book_filter(self, bot, mock_history):
        await bot._cmd_history(["historial", "btc_mxn", "5"])
        mock_history.get_signal_history.assert_called_once_with(
            book="btc_mxn", limit=5
        )

    async def test_status_no_monitor(self, bot, sent_messages):
        await bot._cmd_status()
        assert "no ha escaneado" in _last_body(sent_messages)

    async def test_status_with_monitor(self, bot, sent_messages, mock_history):
        mock_monitor = MagicMock()
        mock_monitor.last_signals = {
            "btc_mxn": RuleSignal(
                action=SignalAction.BUY,
                confidence=Decimal("0.85"),
                rsi=25.0,
                macd_diff=100.0,
                reason="test",
            )
        }
        bot._monitor = mock_monitor

        await bot._cmd_status()
        body = _last_body(sent_messages)
        assert "btc_mxn" in body
        assert "BUY" in body


class TestHandleUpdate:
    async def test_ignores_non_message_updates(self, bot, mock_history):
        await bot._handle_update({"update_id": 1})
        mock_history.get_all_open_positions.assert_not_called()

    async def test_ignores_wrong_chat_id(self, bot, mock_history):
        await bot._handle_update({
            "update_id": 1,
            "message": {
                "chat": {"id": 999},
                "text": "ayuda",
            },
        })
        mock_history.confirm_buy.assert_not_called()

    async def test_processes_correct_chat_id(self, bot, sent_messages):
        await bot._handle_update({
            "update_id": 1,
            "message": {
                "chat": {"id": 123456},
                "text": "ayuda",
            },
        })
        assert len(sent_messages) >= 1

    async def test_ignores_empty_text(self, bot, mock_history):
        await bot._handle_update({
            "update_id": 1,
            "message": {
                "chat": {"id": 123456},
                "text": "",
            },
        })
        mock_history.confirm_buy.assert_not_called()


class TestSendMessageFormat:
    """_send_message usa HTML (fix: Markdown rompe con btc_mxn → 400)."""

    async def test_send_uses_html_parse_mode(self, bot, sent_messages):
        await bot._send_message("✅ *Compra registrada*\nLibro: btc_mxn")
        import json
        payload = json.loads(sent_messages[-1].content)
        assert payload["parse_mode"] == "HTML"
        assert "<b>Compra registrada</b>" in payload["text"]
        assert "btc_mxn" in payload["text"]

    async def test_send_help_with_italic_and_books(self, bot, sent_messages):
        await bot._dispatch_command("ayuda")
        import json
        payload = json.loads(sent_messages[-1].content)
        assert payload["parse_mode"] == "HTML"
        # btc_mxn dentro de <code> no debe romper el parseo
        assert "btc_mxn" in payload["text"]
        assert "<i>ayuda</i>" in payload["text"] or "ayuda" in payload["text"]


class TestTradeAnalyzerIntegration:
    """vendí/compré incluyen stats del TradeAnalyzer cuando está inyectado."""

    @pytest.fixture
    def mock_analyzer(self):
        from trading_system.services.trade_analyzer import TradeStats

        analyzer = MagicMock()
        analyzer.suggest_profit_target_pct = MagicMock(return_value=4.5)
        analyzer.get_closed_stats = MagicMock(return_value=TradeStats(
            n_trades=3, win_rate=2 / 3, avg_profit_pct=3.7, avg_hold_minutes=95.0
        ))
        return analyzer

    @pytest.fixture
    def bot_with_analyzer(self, mock_history, mock_dispatcher, sent_messages, mock_analyzer):
        transport = _mock_handler(sent_messages)
        client = httpx.AsyncClient(transport=transport)
        return TelegramCommandBot(
            bot_token="test-token",
            chat_id="123456",
            history_service=mock_history,
            dispatcher=mock_dispatcher,
            client=client,
            trade_analyzer=mock_analyzer,
        )

    async def test_sell_includes_stats_and_next_target(
        self, bot_with_analyzer, mock_history, sent_messages
    ):
        mock_position = MagicMock()
        mock_position.id = 1
        mock_position.realized_pnl = Decimal(150)
        mock_history.close_position.return_value = mock_position

        await bot_with_analyzer._dispatch_command("vendí btc_mxn 520000")

        body = _last_body(sent_messages)
        assert "win-rate" in body.lower()
        assert "67%" in body
        assert "objetivo sugerido" in body.lower()
        assert "+4.5%" in body

    async def test_buy_includes_suggested_target(
        self, bot_with_analyzer, mock_history, sent_messages
    ):
        mock_position = MagicMock()
        mock_position.id = 1
        mock_position.quantity = Decimal("0.01")
        mock_position.avg_entry_price = Decimal(505000)
        mock_history.confirm_buy.return_value = mock_position

        await bot_with_analyzer._dispatch_command("compré btc_mxn 0.01 505000")

        body = _last_body(sent_messages)
        assert "objetivo" in body.lower()
        assert "+4.5%" in body
        target_price = 505000 * 1.045
        assert f"${target_price:,.2f}" in body

    async def test_sell_without_analyzer_still_works(self, bot, mock_history, sent_messages):
        mock_position = MagicMock()
        mock_position.id = 1
        mock_position.realized_pnl = Decimal(150)
        mock_history.close_position.return_value = mock_position

        await bot._dispatch_command("vendí btc_mxn 520000")

        body = _last_body(sent_messages)
        assert "venta" in body.lower()
        assert "P&amp;L" in body or "P&L" in body

    async def test_sell_analyzer_error_does_not_break(
        self, bot_with_analyzer, mock_history, sent_messages, mock_analyzer
    ):
        mock_position = MagicMock()
        mock_position.id = 1
        mock_position.realized_pnl = Decimal(150)
        mock_history.close_position.return_value = mock_position
        mock_analyzer.get_closed_stats.side_effect = RuntimeError("db down")

        await bot_with_analyzer._dispatch_command("vendí btc_mxn 520000")

        body = _last_body(sent_messages)
        assert "venta" in body.lower()
