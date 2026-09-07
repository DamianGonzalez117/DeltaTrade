"""TDD — Tests for VolumeRatioCalculator (RED phase).

Volume ratio = volumen actual / SMA(volumen, period).
Detecta picos de actividad: ratio > 1.5 = volumen inusualmente alto.
"""
import numpy as np
import pandas as pd
import pytest

from trading_system.features.indicators.volume import VolumeRatioCalculator


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    n = 50
    rng = np.random.default_rng(seed=99)
    close = 500 + np.cumsum(rng.normal(0, 2, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="min")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": rng.uniform(80, 120, n),
        },
        index=idx,
    )


class TestVolumeRatioCalculator:
    def test_name_includes_period(self):
        calc = VolumeRatioCalculator(period=20)
        assert calc.name == "volume_ratio_20"

    def test_calculate_returns_column(self, sample_ohlcv):
        calc = VolumeRatioCalculator(period=20)
        result = calc.calculate(sample_ohlcv)
        assert "volume_ratio_20" in result.columns

    def test_constant_volume_gives_ratio_one(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df["volume"] = 100.0
        calc = VolumeRatioCalculator(period=20)
        result = calc.calculate(df)
        ratio = result["volume_ratio_20"].dropna()
        assert np.allclose(ratio, 1.0)

    def test_volume_spike_gives_ratio_above_one(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df.iloc[-1, df.columns.get_loc("volume")] = 1_000.0  # spike final
        calc = VolumeRatioCalculator(period=20)
        result = calc.calculate(df)
        assert result["volume_ratio_20"].iloc[-1] > 5.0

    def test_low_volume_gives_ratio_below_one(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df.iloc[-1, df.columns.get_loc("volume")] = 1.0  # caída final
        calc = VolumeRatioCalculator(period=20)
        result = calc.calculate(df)
        assert 0 < result["volume_ratio_20"].iloc[-1] < 0.05

    def test_warmup_rows_are_nan(self, sample_ohlcv):
        calc = VolumeRatioCalculator(period=20)
        result = calc.calculate(sample_ohlcv)
        assert result["volume_ratio_20"].iloc[:18].isna().all()
        assert pd.notna(result["volume_ratio_20"].iloc[-1])

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError, match="period"):
            VolumeRatioCalculator(period=0)
