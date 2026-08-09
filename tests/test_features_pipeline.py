"""TDD — Tests for TechnicalFeaturesPipeline (RED phase).

El pipeline:
  1. Recibe una lista de ``FeatureCalculator`` inyectables (DIP/OCP).
  2. Ejecuta cada uno sobre el mismo DataFrame OHLCV.
  3. Combina todas las columnas en un solo DataFrame.
  4. ``extract_latest()`` devuelve la última fila como dict (para IA/riesgo).
"""
import numpy as np
import pandas as pd
import pytest

from trading_system.features.base import FeatureCalculator
from trading_system.features.indicators.atr import ATRCalculator
from trading_system.features.indicators.macd import MACDCalculator
from trading_system.features.indicators.moving_averages import (
    EMACalculator,
    SMACalculator,
)
from trading_system.features.indicators.rsi import RSICalculator
from trading_system.features.pipeline import TechnicalFeaturesPipeline


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    n = 60
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


class TestPipelineConstruction:
    def test_empty_pipeline_creates_empty_features(self, sample_ohlcv):
        pipeline = TechnicalFeaturesPipeline(calculators=[])
        result = pipeline.calculate_all(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv)

    def test_pipeline_with_indicators_combines_columns(self, sample_ohlcv):
        calculators = [
            RSICalculator(period=14),
            MACDCalculator(),
            ATRCalculator(period=14),
        ]
        pipeline = TechnicalFeaturesPipeline(calculators=calculators)
        result = pipeline.calculate_all(sample_ohlcv)
        assert "rsi_14" in result.columns
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_diff" in result.columns
        assert "atr_14" in result.columns

    def test_pipeline_preserves_index(self, sample_ohlcv):
        pipeline = TechnicalFeaturesPipeline(
            calculators=[RSICalculator(14), SMACalculator(20)]
        )
        result = pipeline.calculate_all(sample_ohlcv)
        assert result.index.equals(sample_ohlcv.index)


class TestExtractLatest:
    def test_extract_latest_returns_dict(self, sample_ohlcv):
        pipeline = TechnicalFeaturesPipeline(
            calculators=[RSICalculator(14), ATRCalculator(14)]
        )
        features = pipeline.extract_latest(sample_ohlcv)
        assert isinstance(features, dict)

    def test_extract_latest_has_all_indicator_keys(self, sample_ohlcv):
        pipeline = TechnicalFeaturesPipeline(
            calculators=[RSICalculator(14), EMACalculator(12), ATRCalculator(14)]
        )
        features = pipeline.extract_latest(sample_ohlcv)
        assert "rsi_14" in features
        assert "ema_12" in features
        assert "atr_14" in features

    def test_extract_latest_values_are_not_nan(self, sample_ohlcv):
        pipeline = TechnicalFeaturesPipeline(
            calculators=[RSICalculator(14), ATRCalculator(14), MACDCalculator()]
        )
        features = pipeline.extract_latest(sample_ohlcv)
        for key, value in features.items():
            assert not np.isnan(value), f"{key} is NaN"

    def test_extract_latest_with_empty_pipeline(self, sample_ohlcv):
        pipeline = TechnicalFeaturesPipeline(calculators=[])
        features = pipeline.extract_latest(sample_ohlcv)
        assert features == {}


class TestPipelineOCP:
    def test_adding_new_calculator_does_not_modify_pipeline(self, sample_ohlcv):
        """Un FeatureCalculator custom se integra sin tocar el pipeline."""
        class CustomIndicator(FeatureCalculator):
            @property
            def name(self) -> str:
                return "custom_1"

            def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(
                    {"custom_1": df["close"].pct_change()},
                    index=df.index,
                )

        pipeline = TechnicalFeaturesPipeline(
            calculators=[RSICalculator(14), CustomIndicator()]
        )
        result = pipeline.calculate_all(sample_ohlcv)
        assert "rsi_14" in result.columns
        assert "custom_1" in result.columns

    def test_default_pipeline_has_standard_indicators(self):
        pipeline = TechnicalFeaturesPipeline.create_default()
        names = [c.name for c in pipeline.calculators]
        assert any("rsi" in n for n in names)
        assert any("macd" in n for n in names)
        assert any("atr" in n for n in names)