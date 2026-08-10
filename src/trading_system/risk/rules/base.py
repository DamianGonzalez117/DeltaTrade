"""
Contrato base para las reglas del motor de riesgo.

Cada regla implementa ``RiskRule.evaluate(input) -> RuleResult``.
El motor itera las reglas y combina los resultados en un ``RiskOutput``.

OCP: nueva regla = nueva clase que implementa ``RiskRule``.
DIP: ``RiskEngine`` depende de ``RiskRule`` (ABC), no de clases concretas.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from trading_system.risk.types import RiskInput


@dataclass(frozen=True, slots=True)
class RuleResult:
    """Resultado de una regla individual."""

    passed: bool
    reason: str
    position_size_pct: Decimal | None = None
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    max_capital_pct: Decimal | None = None


class RiskRule(ABC):
    """Interfaz común para todas las reglas de riesgo."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre identificador de la regla."""
        ...

    @abstractmethod
    def evaluate(self, inp: RiskInput) -> RuleResult:
        """Evalúa la regla sobre el input de riesgo."""
        ...