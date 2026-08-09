"""TDD — Tests for ATRCalculator (RED phase).

ATR (Average True Range) mide volatilidad. Usa high, low, close.
"""
import numpy as np
import pandas as pd
import pytest

from trading_system.features.indicators.atr import ATRCalculator


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(seed=77)
    n = 50
    close = 500 + np.cumsum(rng.normal(0, 3, n))
    high = close + rng.uniform(0.5, 3, n)
    low = close - rng.uniform(0.5, 3, n)
    open_ = close + rng.normal(0, 0.5, n)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100},
        index=idx,
    )


class TestATRCalculator:
    def test_name_includes_period(self):
        atr = ATRCalculator(period=14)
        assert "atr" in atr.name
        assert "14" in atr.name

    def test_calculate_returns_atr_column(self, sample_ohlcv):
        atr = ATRCalculator(period=14)
        result = atr.calculate(sample_ohlcv)
        assert f"atr_{atr.period}" in result.columns

    def test_atr_values_are_non_negative(self, sample_ohlcv):
        atr = ATRCalculator(period=14)
        result = atr.calculate(sample_ohlcv)
        vals = result[f"atr_{atr.period}"].dropna()
        assert (vals >= 0).all()

    def test_warmup_rows_not_meaningful(self, sample_ohlcv):
        atr = ATRCalculator(period=14)
        result = atr.calculate(sample_ohlcv)
        # ta llena con 0.0 durante warmup; a partir del period hay valores reales
        assert pd.notna(result[f"atr_{atr.period}"].iloc[-1])
        assert (result[f"atr_{atr.period}"].iloc[:13] == 0).all()

    def test_last_row_valid(self, sample_ohlcv):
        atr = ATRCalculator(period=14)
        result = atr.calculate(sample_ohlcv)
        assert pd.notna(result[f"atr_{atr.period}"].iloc[-1])

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError, match="period"):
            ATRCalculator(period=0)