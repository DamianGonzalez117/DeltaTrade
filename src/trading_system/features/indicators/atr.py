"""ATR (Average True Range) — calculador de features de volatilidad.

SRP: solo calcula ATR. DIP: implementa ``FeatureCalculator``.
"""
import pandas as pd
from ta.volatility import AverageTrueRange

from trading_system.features.base import FeatureCalculator


class ATRCalculator(FeatureCalculator):
    """ATR con periodo configurable."""

    def __init__(self, period: int = 14) -> None:
        if period <= 0:
            raise ValueError("period debe ser positivo")
        self._period = period

    @property
    def period(self) -> int:
        return self._period

    @property
    def name(self) -> str:
        return f"atr_{self._period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        atr = AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self._period,
        )
        return pd.DataFrame(
            {f"atr_{self._period}": atr.average_true_range()},
            index=df.index,
        )