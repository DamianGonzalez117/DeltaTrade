"""Motor de riesgo determinista del sistema de trading.

Exporta:
  - ``RiskEngine``: orquestador de reglas.
  - ``RiskInput`` / ``RiskOutput``: tipos de entrada/salida.
  - ``RiskRule`` / ``RuleResult``: interfaz y resultado de reglas.
  - Reglas concretas: exposure, drawdown, sizing, stops, hours.
"""
from trading_system.risk.engine import RiskEngine
from trading_system.risk.rules import (
    DrawdownRule,
    MarketHoursRule,
    MaxExposureRule,
    RiskRule,
    RuleResult,
    StopLossTakeProfitRule,
    VolatilityPositionSizer,
)
from trading_system.risk.types import RiskInput, RiskOutput

__all__ = [
    "DrawdownRule",
    "MarketHoursRule",
    "MaxExposureRule",
    "RiskEngine",
    "RiskInput",
    "RiskOutput",
    "RiskRule",
    "RuleResult",
    "StopLossTakeProfitRule",
    "VolatilityPositionSizer",
]