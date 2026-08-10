"""Regla: control de exposición máxima del portafolio.

Rechaza si el valor total de posiciones abiertas excede el porcentaje
máximo permitido del equity. Las señales HOLD siempre pasan.
"""
from decimal import Decimal

from trading_system.core.enums import SignalAction
from trading_system.risk.rules.base import RiskRule, RuleResult
from trading_system.risk.types import RiskInput


class MaxExposureRule(RiskRule):
    """Verifica que la exposición total no exceda el límite."""

    def __init__(self, max_total_exposure_pct: Decimal = Decimal("0.50")) -> None:
        self._max_pct = max_total_exposure_pct

    @property
    def name(self) -> str:
        return "max_exposure"

    def evaluate(self, inp: RiskInput) -> RuleResult:
        if inp.signal_action == SignalAction.HOLD:
            return RuleResult(passed=True, reason="HOLD signal bypasses exposure check")
        if inp.portfolio_equity == 0:
            return RuleResult(passed=False, reason="Portfolio equity is zero")
        exposure_pct = inp.current_exposure / inp.portfolio_equity
        if exposure_pct > self._max_pct:
            return RuleResult(
                passed=False,
                reason=(
                    f"Total exposure {exposure_pct:.1%} exceeds "
                    f"max {self._max_pct:.1%}"
                ),
            )
        return RuleResult(passed=True, reason="Exposure within limits")