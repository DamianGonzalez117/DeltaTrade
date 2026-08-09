"""
Contrato base para todos los calculadores de indicadores técnicos.

Cada indicador implementa ``FeatureCalculator``: dado un DataFrame
OHLCV, devuelve un DataFrame con la(s) columna(s) del indicador.

OCP: añadir un indicador nuevo = implementar esta interfaz, sin tocar
el pipeline ni otros indicadores.
"""
from abc import ABC, abstractmethod

import pandas as pd


class FeatureCalculator(ABC):
    """Interfaz común para todos los indicadores técnicos."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre identificador del indicador (ej. ``rsi_14``)."""
        ...

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula el indicador sobre un DataFrame OHLCV.

        Args:
            df: DataFrame con columnas open, high, low, close, volume.

        Returns:
            DataFrame con la(s) columna(s) del indicador, mismo índice.
        """
        ...