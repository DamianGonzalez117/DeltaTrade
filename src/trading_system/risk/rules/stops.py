"""Regla: stop-loss y take-profit basados en ATR.

Esta regla complementa al sizer: si el sizer ya calculó los niveles,
esta regla los valida. Si no, los calcula independientemente.

SRP: solo valida/calcula niveles de SL/TP. DIP: implementa ``RiskRule``.
"""
from decimal import Decimal

from trading_system.core.enums import SignalAction
from trading_system.risk.rules.base import RiskRule, RuleResult
from trading_system.risk.types import RiskInput


class StopLossTakeProfitRule(RiskRule):
    """Calcula o valida stop-loss y take-profit basados en ATR."""

    def __init__(
        self,
        sl_atr_mult: Decimal = Decimal("2.0"),
        tp_atr_mult: Decimal = Decimal("3.0"),
    ) -> None:
        self._sl_mult = sl_atr_mult
        self._tp_mult = tp_atr_mult

    @property
    def name(self) -> str:
        return "stop_loss_take_profit"

    def evaluate(self, inp: RiskInput) -> RuleResult:
        if inp.signal_action == SignalAction.HOLD:
            return RuleResult(passed=True, reason="HOLD signal, no SL/TP needed")

        if inp.atr is None or inp.atr == 0:
            return RuleResult(
                passed=False,
                reason="Cannot calculate SL/TP: ATR is missing",
            )

        sl_distance = self._sl_mult * inp.atr
        tp_distance = self._tp_mult * inp.atr

        if inp.signal_action == SignalAction.BUY:
            sl_price = inp.current_price - sl_distance
            tp_price = inp.current_price + tp_distance
        else:
            sl_price = inp.current_price + sl_distance
            tp_price = inp.current_price - tp_distance

        return RuleResult(
            passed=True,
            reason=f"SL={sl_price}, TP={tp_price}",
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
        )