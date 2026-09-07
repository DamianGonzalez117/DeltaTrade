"""Volume ratio: volumen actual / SMA(volumen, period).

Detecta picos de actividad. ratio > 1.5 → volumen inusualmente alto
(confirma movimiento de precio con participación real del mercado).

SRP: solo calcula el ratio. DIP: implementa ``FeatureCalculator``.
"""
import pandas as pd

from trading_system.features.base import FeatureCalculator


class VolumeRatioCalculator(FeatureCalculator):
    """Ratio de volumen actual contra su media móvil (period configurable)."""

    def __init__(self, period: int = 20) -> None:
        if period <= 0:
            raise ValueError("period debe ser positivo")
        self._period = period

    @property
    def period(self) -> int:
        return self._period

    @property
    def name(self) -> str:
        return f"volume_ratio_{self._period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        vol_sma = df["volume"].rolling(window=self._period).mean()
        ratio = df["volume"] / vol_sma
        return pd.DataFrame(
            {f"volume_ratio_{self._period}": ratio},
            index=df.index,
        )
