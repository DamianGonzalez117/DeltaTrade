"""TDD — Tests for OrderExecutor (RED phase).

``OrderExecutor`` toma un ``RiskOutput`` con decision=APPROVED y:
  1. Construye los parámetros de la orden (symbol, side, type, quantity, price).
  2. Llama al ``ExchangeConnector`` para colocar la orden.
  3. Notifica el resultado via ``NotificationDispatcher``.
  4. Retorna un ``ExecutionResult``.

Si la decisión es REJECTED, no coloca orden — solo notifica.
Si la decisión es APPROVED pero no hay position_size, no ordena.

DIP: ``ExchangeConnector`` y ``NotificationDispatcher`` se inyectan.
"""
from decimal import Decimal
from unittest.mock import AsyncMock

from trading_system.connectors.base import ExecutionResult
from trading_system.core.enums import (
    MarketVenue,
    OrderSide,
    OrderType,
    RiskDecision,
    SignalAction,
)
from trading_system.execution.executor import OrderExecutor
from trading_system.notifications.base import NotificationMessage
from trading_system.notifications.dispatcher import NotificationDispatcher
from trading_system.risk.types import RiskOutput


def _mock_connector() -> AsyncMock:
    connector = AsyncMock()
    connector.place_order = AsyncMock(
        return_value=ExecutionResult(
            external_order_id="ex-123",
            status="open",
            filled_quantity=Decimal(0),
            average_price=None,
            raw_response={"success": True},
        )
    )
    return connector


def _mock_dispatcher() -> AsyncMock:
    dispatcher = AsyncMock(spec=NotificationDispatcher)
    dispatcher.dispatch = AsyncMock(return_value={"log": True})
    return dispatcher


def _approved_output(**overrides) -> RiskOutput:
    defaults = {
        "decision": RiskDecision.APPROVED,
        "reason": "All rules passed",
        "position_size": Decimal("0.01"),
        "stop_loss_price": Decimal(485000),
        "take_profit_price": Decimal(515000),
        "atr_at_evaluation": Decimal(15000),
        "max_capital_pct": Decimal("0.05"),
    }
    defaults.update(overrides)
    return RiskOutput(**defaults)


class TestExecuteApproved:
    async def test_places_market_order_when_approved(self):
        connector = _mock_connector()
        dispatcher = _mock_dispatcher()
        executor = OrderExecutor(connector=connector, dispatcher=dispatcher)

        result = await executor.execute(
            risk_output=_approved_output(),
            symbol="btc_mxn",
            signal_action=SignalAction.BUY,
            venue=MarketVenue.BITSO,
        )

        connector.place_order.assert_awaited_once()
        kwargs = connector.place_order.call_args.kwargs
        assert kwargs["symbol"] == "btc_mxn"
        assert kwargs["side"] == OrderSide.BUY
        assert kwargs["order_type"] == OrderType.MARKET
        assert kwargs["quantity"] == Decimal("0.01")
        assert isinstance(result, ExecutionResult)
        assert result.external_order_id == "ex-123"

    async def test_sell_signal_sends_sell_side(self):
        connector = _mock_connector()
        dispatcher = _mock_dispatcher()
        executor = OrderExecutor(connector=connector, dispatcher=dispatcher)

        await executor.execute(
            risk_output=_approved_output(),
            symbol="btc_mxn",
            signal_action=SignalAction.SELL,
            venue=MarketVenue.BITSO,
        )

        kwargs = connector.place_order.call_args.kwargs
        assert kwargs["side"] == OrderSide.SELL

    async def test_notifies_on_success(self):
        connector = _mock_connector()
        dispatcher = _mock_dispatcher()
        executor = OrderExecutor(connector=connector, dispatcher=dispatcher)

        await executor.execute(
            risk_output=_approved_output(),
            symbol="btc_mxn",
            signal_action=SignalAction.BUY,
            venue=MarketVenue.BITSO,
        )

        dispatcher.dispatch.assert_awaited_once()
        msg = dispatcher.dispatch.call_args.args[0]
        assert isinstance(msg, NotificationMessage)
        assert "btc_mxn" in msg.body


class TestExecuteRejected:
    async def test_does_not_place_order_when_rejected(self):
        connector = _mock_connector()
        dispatcher = _mock_dispatcher()
        executor = OrderExecutor(connector=connector, dispatcher=dispatcher)

        result = await executor.execute(
            risk_output=RiskOutput(
                decision=RiskDecision.REJECTED,
                reason="Drawdown exceeded",
                position_size=None,
                stop_loss_price=None,
                take_profit_price=None,
                atr_at_evaluation=Decimal(15000),
                max_capital_pct=None,
            ),
            symbol="btc_mxn",
            signal_action=SignalAction.BUY,
            venue=MarketVenue.BITSO,
        )

        connector.place_order.assert_not_awaited()
        assert result is None

    async def test_notifies_critical_when_rejected(self):
        connector = _mock_connector()
        dispatcher = _mock_dispatcher()
        executor = OrderExecutor(connector=connector, dispatcher=dispatcher)

        await executor.execute(
            risk_output=RiskOutput(
                decision=RiskDecision.REJECTED,
                reason="Drawdown exceeded",
                position_size=None,
                stop_loss_price=None,
                take_profit_price=None,
                atr_at_evaluation=Decimal(15000),
                max_capital_pct=None,
            ),
            symbol="btc_mxn",
            signal_action=SignalAction.BUY,
            venue=MarketVenue.BITSO,
        )

        dispatcher.dispatch.assert_awaited_once()


class TestExecuteHold:
    async def test_does_not_place_order_on_hold(self):
        connector = _mock_connector()
        dispatcher = _mock_dispatcher()
        executor = OrderExecutor(connector=connector, dispatcher=dispatcher)

        result = await executor.execute(
            risk_output=_approved_output(position_size=None),
            symbol="btc_mxn",
            signal_action=SignalAction.HOLD,
            venue=MarketVenue.BITSO,
        )

        connector.place_order.assert_not_awaited()
        assert result is None


class TestExecuteNoPositionSize:
    async def test_approved_but_no_size_does_not_order(self):
        connector = _mock_connector()
        dispatcher = _mock_dispatcher()
        executor = OrderExecutor(connector=connector, dispatcher=dispatcher)

        result = await executor.execute(
            risk_output=_approved_output(position_size=None),
            symbol="btc_mxn",
            signal_action=SignalAction.BUY,
            venue=MarketVenue.BITSO,
        )

        connector.place_order.assert_not_awaited()
        assert result is None