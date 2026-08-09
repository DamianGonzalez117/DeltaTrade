"""TDD — Tests for SMACalculator and EMACalculator (RED phase)."""
import numpy as np
import pandas as pd
import pytest

from trading_system.features.indicators.moving_averages import (
    EMACalculator,
    SMACalculator,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(seed=99)
    n = 50
    close = 200 + np.cumsum(rng.normal(0, 2, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100},
        index=idx,
    )


class TestSMACalculator:
    def test_name_includes_period(self):
        sma = SMACalculator(period=20)
        assert "sma" in sma.name
        assert "20" in sma.name

    def test_calculate_returns_sma_column(self, sample_ohlcv):
        sma = SMACalculator(period=20)
        result = sma.calculate(sample_ohlcv)
        assert f"sma_{sma.period}" in result.columns

    def test_warmup_rows_are_nan(self, sample_ohlcv):
        sma = SMACalculator(period=20)
        result = sma.calculate(sample_ohlcv)
        assert result[f"sma_{sma.period}"].iloc[:19].isna().all()
        assert pd.notna(result[f"sma_{sma.period}"].iloc[19])

    def test_last_row_valid(self, sample_ohlcv):
        sma = SMACalculator(period=10)
        result = sma.calculate(sample_ohlcv)
        assert pd.notna(result[f"sma_{sma.period}"].iloc[-1])

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError, match="period"):
            SMACalculator(period=0)


class TestEMACalculator:
    def test_name_includes_period(self):
        ema = EMACalculator(period=12)
        assert "ema" in ema.name
        assert "12" in ema.name

    def test_calculate_returns_ema_column(self, sample_ohlcv):
        ema = EMACalculator(period=12)
        result = ema.calculate(sample_ohlcv)
        assert f"ema_{ema.period}" in result.columns

    def test_warmup_then_valid_values(self, sample_ohlcv):
        ema = EMACalculator(period=12)
        result = ema.calculate(sample_ohlcv)
        assert pd.notna(result[f"ema_{ema.period}"].iloc[-1])

    def test_last_row_valid(self, sample_ohlcv):
        ema = EMACalculator(period=26)
        result = ema.calculate(sample_ohlcv)
        assert pd.notna(result[f"ema_{ema.period}"].iloc[-1])

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError, match="period"):
            EMACalculator(period=-1)