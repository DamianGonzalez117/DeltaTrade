"""
Ejecutor de órdenes: toma un ``RiskOutput`` aprobado y lo ejecuta.

Flujo:
  1. Si decision != APPROVED o no hay position_size → no ordena, solo notifica.
  2. Construye los parámetros (symbol, side, type=MARKET, quantity).
  3. Llama al ``ExchangeConnector`` para colocar la orden.
  4. Notifica el resultado via ``NotificationDispatcher``.

DIP: ``ExchangeConnector`` y ``NotificationDispatcher`` se inyectan.
SRP: solo ejecuta y notifica. No decide (eso hace el motor de riesgo).
"""
from decimal import Decimal

import structlog

from trading_system.connectors.base import ExecutionResult
from trading_system.core.enums import (
    NotificationLevel,
    OrderSide,
    OrderType,
    RiskDecision,
    SignalAction,
)
from trading_system.notifications.base import NotificationMessage
from trading_system.notifications.dispatcher import NotificationDispatcher
from trading_system.risk.types import RiskOutput

logger = structlog.get_logger(__name__)

_ACTION_TO_SIDE = {
    SignalAction.BUY: OrderSide.BUY,
    SignalAction.SELL: OrderSide.SELL,
}


class OrderExecutor:
    """Ejecuta órdenes aprobadas por el motor de riesgo."""

    def __init__(
        self,
        *,
        connector,
        dispatcher: NotificationDispatcher,
    ) -> None:
        self._connector = connector
        self._dispatcher = dispatcher

    async def execute(
        self,
        *,
        risk_output: RiskOutput,
        symbol: str,
        signal_action: SignalAction,
        venue,
    ) -> ExecutionResult | None:
        if risk_output.decision == RiskDecision.REJECTED:
            await self._notify_rejected(risk_output, symbol)
            return None

        if signal_action == SignalAction.HOLD or risk_output.position_size is None:
            logger.info("executor.skipped", symbol=symbol, reason="HOLD or no position size")
            return None

        side = _ACTION_TO_SIDE[signal_action]
        quantity = risk_output.position_size

        logger.info(
            "executor.placing_order",
            symbol=symbol,
            side=side.value,
            quantity=str(quantity),
        )

        result = await self._connector.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )

        await self._notify_success(result, symbol, side, quantity)

        return result

    async def _notify_success(
        self,
        result: ExecutionResult,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
    ) -> None:
        msg = NotificationMessage(
            level=NotificationLevel.INFO,
            title="Order Placed",
            body=(
                f"{side.value.upper()} {quantity} {symbol} "
                f"→ ext_id={result.external_order_id} status={result.status}"
            ),
        )
        await self._dispatcher.dispatch(msg)

    async def _notify_rejected(self, risk_output: RiskOutput, symbol: str) -> None:
        msg = NotificationMessage(
            level=NotificationLevel.WARNING,
            title="Order Rejected by Risk Engine",
            body=f"{symbol}: {risk_output.reason}",
        )
        await self._dispatcher.dispatch(msg)