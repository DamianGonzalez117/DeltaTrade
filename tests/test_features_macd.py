"""TDD — Tests for MACDCalculator (RED phase).

MACD produce 3 columnas: macd, macd_signal, macd_diff.
"""
import numpy as np
import pandas as pd
import pytest

from trading_system.features.indicators.macd import MACDCalculator


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    n = 60
    close = 100 + np.cumsum(rng.normal(0, 1.5, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100},
        index=idx,
    )


class TestMACDCalculator:
    def test_name_includes_params(self):
        macd = MACDCalculator(fast=12, slow=26, signal=9)
        assert "macd" in macd.name

    def test_calculate_returns_three_columns(self, sample_ohlcv):
        macd = MACDCalculator()
        result = macd.calculate(sample_ohlcv)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_diff" in result.columns

    def test_warmup_rows_are_nan(self, sample_ohlcv):
        macd = MACDCalculator(fast=12, slow=26, signal=9)
        result = macd.calculate(sample_ohlcv)
        assert result["macd"].iloc[:25].isna().all()

    def test_last_row_has_values(self, sample_ohlcv):
        macd = MACDCalculator()
        result = macd.calculate(sample_ohlcv)
        assert pd.notna(result["macd"].iloc[-1])
        assert pd.notna(result["macd_signal"].iloc[-1])
        assert pd.notna(result["macd_diff"].iloc[-1])

    def test_invalid_fast_period_raises(self):
        with pytest.raises(ValueError, match="fast"):
            MACDCalculator(fast=0)

    def test_slow_must_be_greater_than_fast(self):
        with pytest.raises(ValueError, match="slow.*mayor"):
            MACDCalculator(fast=26, slow=12)