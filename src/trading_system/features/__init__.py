"""Pipeline de features técnicas del sistema de trading.

Exporta:
  - ``FeatureCalculator``: interfaz base para indicadores.
  - ``TechnicalFeaturesPipeline``: orquestador de múltiples indicadores.
  - Indicadores: RSI, MACD, SMA, EMA, ATR.
"""
from trading_system.features.base import FeatureCalculator
from trading_system.features.indicators.atr import ATRCalculator
from trading_system.features.indicators.macd import MACDCalculator
from trading_system.features.indicators.moving_averages import (
    EMACalculator,
    SMACalculator,
)
from trading_system.features.indicators.rsi import RSICalculator
from trading_system.features.pipeline import TechnicalFeaturesPipeline

__all__ = [
    "ATRCalculator",
    "EMACalculator",
    "FeatureCalculator",
    "MACDCalculator",
    "RSICalculator",
    "SMACalculator",
    "TechnicalFeaturesPipeline",
]