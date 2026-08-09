"""Medias móviles: SMA (Simple) y EMA (Exponential).

SRP: cada clase calcula un tipo de media. DIP: implementan ``FeatureCalculator``.
"""
import pandas as pd
from ta.trend import EMAIndicator, SMAIndicator

from trading_system.features.base import FeatureCalculator


class SMACalculator(FeatureCalculator):
    """Simple Moving Average con periodo configurable."""

    def __init__(self, period: int = 20) -> None:
        if period <= 0:
            raise ValueError("period debe ser positivo")
        self._period = period

    @property
    def period(self) -> int:
        return self._period

    @property
    def name(self) -> str:
        return f"sma_{self._period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        sma = SMAIndicator(close=df["close"], window=self._period)
        return pd.DataFrame(
            {f"sma_{self._period}": sma.sma_indicator()},
            index=df.index,
        )


class EMACalculator(FeatureCalculator):
    """Exponential Moving Average con periodo configurable."""

    def __init__(self, period: int = 12) -> None:
        if period <= 0:
            raise ValueError("period debe ser positivo")
        self._period = period

    @property
    def period(self) -> int:
        return self._period

    @property
    def name(self) -> str:
        return f"ema_{self._period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        ema = EMAIndicator(close=df["close"], window=self._period)
        return pd.DataFrame(
            {f"ema_{self._period}": ema.ema_indicator()},
            index=df.index,
        )