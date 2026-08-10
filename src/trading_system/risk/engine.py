"""
Motor de riesgo determinista.

Ejecuta una lista de ``RiskRule`` sobre un ``RiskInput`` y combina los
resultados en un ``RiskOutput``:

  1. Si cualquier regla "gate" (drawdown, exposure, hours) falla → REJECTED.
  2. Si todas pasan → el sizer calcula tamaño/SL/TP → APPROVED.
  3. La razón concatena todas las reglas que fallaron.

DIP: depende de ``RiskRule`` (ABC). OCP: nueva regla = inyectar por constructor.
"""
from __future__ import annotations

from decimal import Decimal

import structlog

from trading_system.core.enums import RiskDecision, SignalAction
from trading_system.risk.rules.base import RiskRule, RuleResult
from trading_system.risk.rules.drawdown import DrawdownRule
from trading_system.risk.rules.exposure import MaxExposureRule
from trading_system.risk.rules.hours import MarketHoursRule
from trading_system.risk.rules.sizing import VolatilityPositionSizer
from trading_system.risk.rules.stops import StopLossTakeProfitRule
from trading_system.risk.types import RiskInput, RiskOutput

logger = structlog.get_logger(__name__)


class RiskEngine:
    """Motor determinista que evalúa señales contra reglas de riesgo."""

    def __init__(self, rules: list[RiskRule]) -> None:
        self._rules = rules

    @property
    def rules(self) -> list[RiskRule]:
        return list(self._rules)

    def evaluate(self, inp: RiskInput) -> RiskOutput:
        results: list[tuple[RiskRule, RuleResult]] = []
        failed_reasons: list[str] = []

        for rule in self._rules:
            result = rule.evaluate(inp)
            results.append((rule, result))
            if not result.passed:
                failed_reasons.append(f"[{rule.name}] {result.reason}")

        if failed_reasons:
            reason = "; ".join(failed_reasons)
            logger.info(
                "risk_engine.rejected",
                reasons=failed_reasons,
                symbol=inp.asset_symbol,
            )
            return RiskOutput(
                decision=RiskDecision.REJECTED,
                reason=reason,
                position_size=None,
                stop_loss_price=None,
                take_profit_price=None,
                atr_at_evaluation=inp.atr,
                max_capital_pct=None,
            )

        # All rules passed — collect sizing and SL/TP from results
        position_size_pct: Decimal | None = None
        stop_loss_price: Decimal | None = None
        take_profit_price: Decimal | None = None
        max_capital_pct: Decimal | None = None

        for _rule, result in results:
            if result.position_size_pct is not None:
                position_size_pct = result.position_size_pct
            if result.stop_loss_price is not None:
                stop_loss_price = result.stop_loss_price
            if result.take_profit_price is not None:
                take_profit_price = result.take_profit_price
            if result.max_capital_pct is not None:
                max_capital_pct = result.max_capital_pct

        if inp.signal_action == SignalAction.HOLD:
            return RiskOutput(
                decision=RiskDecision.APPROVED,
                reason="HOLD signal approved, no position needed",
                position_size=None,
                stop_loss_price=None,
                take_profit_price=None,
                atr_at_evaluation=inp.atr,
                max_capital_pct=inp.max_capital_pct,
            )

        # Calculate actual quantity from percentage
        actual_size: Decimal | None = None
        if position_size_pct is not None and inp.current_price > 0:
            actual_size = position_size_pct * inp.portfolio_equity / inp.current_price

        logger.info(
            "risk_engine.approved",
            symbol=inp.asset_symbol,
            position_size=actual_size,
            sl=stop_loss_price,
            tp=take_profit_price,
        )

        return RiskOutput(
            decision=RiskDecision.APPROVED,
            reason="All risk rules passed",
            position_size=actual_size,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            atr_at_evaluation=inp.atr,
            max_capital_pct=max_capital_pct or inp.max_capital_pct,
        )

    @classmethod
    def create_default(cls) -> RiskEngine:
        """Factory con el set estándar de reglas del sistema."""
        return cls(
            rules=[
                MarketHoursRule(),
                DrawdownRule(),
                MaxExposureRule(),
                VolatilityPositionSizer(),
                StopLossTakeProfitRule(),
            ]
        )