"""Regla: control de drawdown máximo del portafolio.

Rechaza si el drawdown actual excede el límite máximo configurado.
Las señales HOLD siempre pasan (no abren nueva posición).
"""
from trading_system.core.enums import SignalAction
from trading_system.risk.rules.base import RiskRule, RuleResult
from trading_system.risk.types import RiskInput


class DrawdownRule(RiskRule):
    """Verifica que el drawdown no exceda el máximo permitido."""

    @property
    def name(self) -> str:
        return "drawdown"

    def evaluate(self, inp: RiskInput) -> RuleResult:
        if inp.signal_action == SignalAction.HOLD:
            return RuleResult(passed=True, reason="HOLD signal bypasses drawdown check")
        if inp.current_drawdown >= inp.max_drawdown_pct:
            return RuleResult(
                passed=False,
                reason=(
                    f"Current drawdown {inp.current_drawdown:.1%} >= "
                    f"max {inp.max_drawdown_pct:.1%}"
                ),
            )
        return RuleResult(passed=True, reason="Drawdown within limits")