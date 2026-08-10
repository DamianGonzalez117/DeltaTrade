"""Regla: verificación de horario de mercado.

Para venues con horario restringido (ej. GBM/BMV), rechaza operaciones
fuera del horario bursátil. Bitso (24/7) pasa siempre (is_market_open=None).

Las señales HOLD siempre pasan (no requieren ejecución inmediata).
"""
from trading_system.core.enums import SignalAction
from trading_system.risk.rules.base import RiskRule, RuleResult
from trading_system.risk.types import RiskInput


class MarketHoursRule(RiskRule):
    """Verifica que el mercado esté abierto antes de operar."""

    @property
    def name(self) -> str:
        return "market_hours"

    def evaluate(self, inp: RiskInput) -> RuleResult:
        if inp.signal_action == SignalAction.HOLD:
            return RuleResult(passed=True, reason="HOLD signal bypasses hours check")

        if inp.is_market_open is None:
            return RuleResult(passed=True, reason="Market hours not checked (24/7 venue)")

        if not inp.is_market_open:
            return RuleResult(
                passed=False,
                reason="Market is closed, cannot execute trades",
            )

        return RuleResult(passed=True, reason="Market is open")