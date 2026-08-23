"""TDD — Tests for TrainingPipeline (RED→GREEN).

Se usa un clasificador mock que cumple el protocolo ``XGBClassifierLike``.
No se entrena un XGBoost real (lento y requiere dependencia pesada).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_system.features.pipeline import TechnicalFeaturesPipeline
from trading_system.services.training import TrainingPipeline


class MockClassifier:
    """Mock classifier que cumple el protocolo XGBClassifierLike."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._X = X
        self._y = y

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.array([[0.7, 0.2, 0.1]] * n)

    def save_model(self, path: str) -> None:
        Path(path).write_text(json.dumps({"mock": True, "kwargs": self.kwargs}))

    def get_booster(self) -> Any:
        return self


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    """Genera OHLCV sintético con tendencia para tener labels variados."""
    np.random.seed(42)
    close = 500_000 + np.cumsum(np.random.randn(n) * 1000)
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open": close - 100,
        "high": close + 200,
        "low": close - 200,
        "close": close,
        "volume": np.random.uniform(0.1, 1.0, n),
    })


@pytest.fixture
def pipeline() -> TrainingPipeline:
    return TrainingPipeline(
        features_pipeline=TechnicalFeaturesPipeline.create_default(),
        classifier_factory=MockClassifier,
        forward_window=5,
        buy_threshold=0.01,
        sell_threshold=-0.01,
    )


class TestLabelForwardReturns:
    def test_creates_label_column(self, pipeline: TrainingPipeline):
        df = _make_ohlcv(50)
        labeled = pipeline.label_forward_returns(df)
        assert "label" in labeled.columns
        assert "forward_return" in labeled.columns

    def test_labels_are_valid_integers(self, pipeline: TrainingPipeline):
        df = _make_ohlcv(50)
        labeled = pipeline.label_forward_returns(df)
        assert labeled["label"].isin([0, 1, 2]).all()

    def test_drops_nan_forward_returns(self, pipeline: TrainingPipeline):
        df = _make_ohlcv(50)
        labeled = pipeline.label_forward_returns(df)
        assert labeled["forward_return"].notna().all()
        # last forward_window rows should be dropped
        assert len(labeled) == 50 - 5


class TestPrepareTrainingData:
    def test_returns_X_y_arrays(self, pipeline: TrainingPipeline):
        ohlcv = _make_ohlcv(100)
        X, y = pipeline.prepare_training_data(ohlcv)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] > 0

    def test_feature_names_populated(self, pipeline: TrainingPipeline):
        ohlcv = _make_ohlcv(100)
        pipeline.prepare_training_data(ohlcv)
        assert len(pipeline.feature_names) > 0
        assert "label" not in pipeline.feature_names

    def test_raises_on_empty_data(self, pipeline: TrainingPipeline):
        ohlcv = pd.DataFrame({
            "ts": [], "open": [], "high": [], "low": [], "close": [], "volume": [],
        })
        with pytest.raises(ValueError, match="No hay filas"):
            pipeline.prepare_training_data(ohlcv)


class TestTrain:
    def test_train_returns_classifier(self, pipeline: TrainingPipeline):
        ohlcv = _make_ohlcv(100)
        clf = pipeline.train(ohlcv)
        assert isinstance(clf, MockClassifier)
        assert clf._X is not None
        assert clf._y is not None

    def test_train_raises_on_insufficient_samples(self, pipeline: TrainingPipeline):
        # 40 rows: MACD warmup=26, forward_window=5 → ~9 samples < 10
        ohlcv = _make_ohlcv(40)
        with pytest.raises(ValueError, match="Insuficientes"):
            pipeline.train(ohlcv)


class TestSaveModel:
    def test_save_model_writes_files(
        self, pipeline: TrainingPipeline, tmp_path: Path
    ):
        ohlcv = _make_ohlcv(100)
        pipeline.train(ohlcv)
        model_path = tmp_path / "model.json"
        pipeline.save_model(model_path)
        assert model_path.exists()

        meta_path = tmp_path / "model.features.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "feature_names" in meta
        assert "forward_window" in meta
        assert meta["forward_window"] == 5

    def test_save_raises_if_not_trained(self, pipeline: TrainingPipeline, tmp_path: Path):
        with pytest.raises(RuntimeError, match="no entrenado"):
            pipeline.save_model(tmp_path / "model.json")


class TestLoadOHLVFromDF:
    def test_normalizes_dataframe(self):
        df = pd.DataFrame({
            "open": [1.0], "high": [2.0], "low": [0.5],
            "close": [1.5], "volume": [100.0],
        })
        result = TrainingPipeline.load_ohlcv_from_df(df)
        assert "ts" in result.columns
        assert "close" in result.columns

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"open": [1.0], "close": [1.5]})
        with pytest.raises(ValueError, match="faltantes"):
            TrainingPipeline.load_ohlcv_from_df(df)
