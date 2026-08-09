"""Regla: position sizing basado en volatilidad (ATR) + stop-loss/take-profit.

Calcula el tamaño de posición como una fracción del equity, inversamente
proporcional a la volatilidad (ATR). También calcula stop-loss y
take-profit basados en múltiplos del ATR.

Fórmula:
  risk_per_unit = sl_atr_mult * ATR  (distancia al stop)
  position_value = max_capital_pct * equity
  position_size_pct = position_value / (price * 1)  → capped at max_capital_pct
  qty = position_value / price

SRP: solo calcula tamaño y niveles. DIP: implementa ``RiskRule``.
"""
from decimal import Decimal

from trading_system.core.enums import SignalAction
from trading_system.risk.rules.base import RiskRule, RuleResult
from trading_system.risk.types import RiskInput


class VolatilityPositionSizer(RiskRule):
    """Position sizing por volatilidad ATR con SL/TP automáticos."""

    def __init__(
        self,
        atr_multiplier: Decimal = Decimal("1.5"),
        sl_atr_mult: Decimal = Decimal("2.0"),
        tp_atr_mult: Decimal = Decimal("3.0"),
    ) -> None:
        self._atr_mult = atr_multiplier
        self._sl_mult = sl_atr_mult
        self._tp_mult = tp_atr_mult

    @property
    def name(self) -> str:
        return "volatility_sizer"

    def evaluate(self, inp: RiskInput) -> RuleResult:
        if inp.signal_action == SignalAction.HOLD:
            return RuleResult(passed=True, reason="HOLD signal, no position needed")

        if inp.atr is None or inp.atr == 0:
            return RuleResult(
                passed=True,
                reason="No ATR available, sizing deferred",
                position_size_pct=None,
            )

        sl_distance = self._sl_mult * inp.atr
        tp_distance = self._tp_mult * inp.atr

        if inp.signal_action == SignalAction.BUY:
            sl_price = inp.current_price - sl_distance
            tp_price = inp.current_price + tp_distance
        else:
            sl_price = inp.current_price + sl_distance
            tp_price = inp.current_price - tp_distance

        max_position_value = inp.max_capital_pct * inp.portfolio_equity
        position_size_pct = min(
            inp.max_capital_pct,
            max_position_value / inp.current_price,
        )

        return RuleResult(
            passed=True,
            reason=f"Position sized via ATR={inp.atr}",
            position_size_pct=position_size_pct,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            max_capital_pct=inp.max_capital_pct,
        )