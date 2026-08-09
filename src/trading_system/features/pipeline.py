"""
Pipeline de features técnicas.

Orquesta múltiples ``FeatureCalculator`` sobre el mismo DataFrame OHLCV:
  1. ``calculate_all``: ejecuta cada calculador, combina columnas.
  2. ``extract_latest``: devuelve la última fila como dict (para IA/riesgo).

DIP: depende de ``FeatureCalculator`` (ABC), no de clases concretas.
OCP: añadir un indicador nuevo = pasar otra instancia por constructor.
"""
from __future__ import annotations

import pandas as pd

from trading_system.features.base import FeatureCalculator
from trading_system.features.indicators.atr import ATRCalculator
from trading_system.features.indicators.macd import MACDCalculator
from trading_system.features.indicators.moving_averages import (
    EMACalculator,
    SMACalculator,
)
from trading_system.features.indicators.rsi import RSICalculator


class TechnicalFeaturesPipeline:
    """Combina múltiples indicadores técnicos en un solo DataFrame."""

    def __init__(self, calculators: list[FeatureCalculator]) -> None:
        self._calculators = calculators

    @property
    def calculators(self) -> list[FeatureCalculator]:
        return list(self._calculators)

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._calculators:
            return pd.DataFrame(index=df.index)
        parts = [calc.calculate(df) for calc in self._calculators]
        return pd.concat(parts, axis=1)

    def extract_latest(self, df: pd.DataFrame) -> dict[str, float]:
        if not self._calculators:
            return {}
        combined = self.calculate_all(df)
        last_row = combined.iloc[-1]
        return {col: float(val) for col, val in last_row.items() if pd.notna(val)}

    @classmethod
    def create_default(cls) -> TechnicalFeaturesPipeline:
        """Factory con el set estándar de indicadores del sistema."""
        return cls(
            calculators=[
                RSICalculator(period=14),
                MACDCalculator(fast=12, slow=26, signal=9),
                SMACalculator(period=20),
                EMACalculator(period=12),
                EMACalculator(period=26),
                ATRCalculator(period=14),
            ]
        )