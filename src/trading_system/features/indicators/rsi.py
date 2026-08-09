"""RSI (Relative Strength Index) — calculador de features técnicos.

SRP: solo calcula RSI. DIP: implementa ``FeatureCalculator``.
"""
import pandas as pd
from ta.momentum import RSIIndicator

from trading_system.features.base import FeatureCalculator


class RSICalculator(FeatureCalculator):
    """RSI con periodo configurable."""

    def __init__(self, period: int = 14) -> None:
        if period <= 0:
            raise ValueError("period debe ser positivo")
        self._period = period

    @property
    def name(self) -> str:
        return f"rsi_{self._period}"

    @property
    def period(self) -> int:
        return self._period

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi = RSIIndicator(close=df["close"], window=self._period)
        return pd.DataFrame(
            {f"rsi_{self._period}": rsi.rsi()},
            index=df.index,
        )