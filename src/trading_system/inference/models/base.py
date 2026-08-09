"""Contrato base para modelos técnicos de inferencia."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from trading_system.core.enums import SignalAction


@dataclass(frozen=True, slots=True)
class TechnicalResult:
    """Resultado del modelo técnico."""

    action: SignalAction
    confidence: Decimal  # 0.0 a 1.0


class TechnicalModel(ABC):
    """Interfaz para modelos cuantitativos (XGBoost, LSTM, etc.)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def predict(self, features: dict[str, float]) -> TechnicalResult:
        ...