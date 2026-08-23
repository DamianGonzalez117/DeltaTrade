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
