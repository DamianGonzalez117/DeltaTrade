"""MACD (Moving Average Convergence Divergence) — calculador de features.

Produce 3 columnas: macd, macd_signal, macd_diff.
SRP: solo calcula MACD. DIP: implementa ``FeatureCalculator``.
"""
import pandas as pd
from ta.trend import MACD

from trading_system.features.base import FeatureCalculator


class MACDCalculator(FeatureCalculator):
    """MACD con parámetros configurables."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if fast <= 0:
            raise ValueError("fast debe ser positivo")
        if slow <= fast:
            raise ValueError("slow debe ser mayor que fast")
        if signal <= 0:
            raise ValueError("signal debe ser positivo")
        self._fast = fast
        self._slow = slow
        self._signal = signal

    @property
    def name(self) -> str:
        return f"macd_{self._fast}_{self._slow}_{self._signal}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        macd = MACD(
            close=df["close"],
            window_slow=self._slow,
            window_fast=self._fast,
            window_sign=self._signal,
        )
        return pd.DataFrame(
            {
                "macd": macd.macd(),
                "macd_signal": macd.macd_signal(),
                "macd_diff": macd.macd_diff(),
            },
            index=df.index,
        )