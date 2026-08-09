"""TDD — Tests for FeatureCalculator base interface and RSICalculator (RED phase).

``FeatureCalculator`` es el contrato común: dado un DataFrame OHLCV,
devuelve un DataFrame con la(s) columna(s) del indicador.

``RSICalculator`` usa la librería ``ta`` internamente pero normaliza
la salida a una sola columna ``rsi_<period>``.
"""
import numpy as np
import pandas as pd
import pytest

from trading_system.features.base import FeatureCalculator
from trading_system.features.indicators.rsi import RSICalculator


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """50 velas sintéticas con suficiente variación para RSI."""
    rng = np.random.default_rng(seed=42)
    n = 50
    close = 100 + np.cumsum(rng.normal(0, 1.5, n))
    high = close + rng.uniform(0.1, 1, n)
    low = close - rng.uniform(0.1, 1, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(100, 1000, n)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# FeatureCalculator ABC
# ---------------------------------------------------------------------------

class TestFeatureCalculatorABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            FeatureCalculator()  # type: ignore[abstract]

    def test_subclass_must_implement_name_and_calculate(self):
        class Incomplete(FeatureCalculator):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# RSICalculator
# ---------------------------------------------------------------------------

class TestRSICalculator:
    def test_name_property_includes_period(self):
        rsi = RSICalculator(period=14)
        assert "rsi" in rsi.name
        assert "14" in rsi.name

    def test_calculate_returns_dataframe(self, sample_ohlcv):
        rsi = RSICalculator(period=14)
        result = rsi.calculate(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_result_has_rsi_column(self, sample_ohlcv):
        rsi = RSICalculator(period=14)
        result = rsi.calculate(sample_ohlcv)
        assert "rsi_14" in result.columns

    def test_rsi_values_in_valid_range(self, sample_ohlcv):
        rsi = RSICalculator(period=14)
        result = rsi.calculate(sample_ohlcv)
        rsi_values = result["rsi_14"].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()

    def test_first_rows_are_nan_until_warmup(self, sample_ohlcv):
        rsi = RSICalculator(period=14)
        result = rsi.calculate(sample_ohlcv)
        assert result["rsi_14"].iloc[:13].isna().all()
        assert pd.notna(result["rsi_14"].iloc[13])

    def test_last_row_has_valid_rsi(self, sample_ohlcv):
        rsi = RSICalculator(period=14)
        result = rsi.calculate(sample_ohlcv)
        last = result["rsi_14"].iloc[-1]
        assert pd.notna(last)
        assert 0 <= last <= 100

    def test_different_period_changes_column_name(self, sample_ohlcv):
        rsi = RSICalculator(period=21)
        result = rsi.calculate(sample_ohlcv)
        assert "rsi_21" in result.columns

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError, match="period"):
            RSICalculator(period=0)

        with pytest.raises(ValueError, match="period"):
            RSICalculator(period=-5)