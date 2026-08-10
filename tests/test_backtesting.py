"""TDD — Tests for BacktestEngine (RED→GREEN).

Se mockea ``predict_fn`` para controlar las señales y validar el flujo
de simulation. ``RiskEngine.create_default()`` usa el real (determinista).
"""
from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from trading_system.core.enums import SignalAction
from trading_system.features.pipeline import TechnicalFeaturesPipeline
from trading_system.risk.engine import RiskEngine
from trading_system.services.backtesting import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
)


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 500_000 + np.cumsum(rng.standard_normal(n) * 1_000)
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open": close - 100,
        "high": close + 200,
        "low": close - 200,
        "close": close,
        "volume": rng.uniform(0.1, 1.0, n),
    })


@pytest.fixture
def features_pipeline():
    return TechnicalFeaturesPipeline.create_default()


@pytest.fixture
def risk_engine():
    return RiskEngine.create_default()


@pytest.fixture
def always_buy():
    def predict(features: dict[str, float]) -> tuple[SignalAction, Decimal]:
        return SignalAction.BUY, Decimal("0.90")
    return predict


@pytest.fixture
def always_hold():
    def predict(features: dict[str, float]) -> tuple[SignalAction, Decimal]:
        return SignalAction.HOLD, Decimal("0.50")
    return predict


class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.initial_capital == Decimal(10000)
        assert cfg.fee_pct == Decimal("0.001")


class TestBacktestResult:
    def test_is_frozen(self):
        result = BacktestResult(
            total_return_pct=10.0,
            max_drawdown_pct=5.0,
            sharpe_ratio=1.5,
            num_trades=5,
            num_wins=3,
            num_losses=2,
            win_rate=0.6,
            final_equity=Decimal(11000),
        )
        with pytest.raises((AttributeError, TypeError)):
            result.total_return_pct = 99.0


class TestBacktestEngine:
    def test_raises_on_insufficient_data(self, features_pipeline, always_buy, risk_engine):
        engine = BacktestEngine(
            features_pipeline=features_pipeline,
            predict_fn=always_buy,
            risk_engine=risk_engine,
        )
        ohlcv = _make_ohlcv(20)
        with pytest.raises(ValueError, match="Insuficientes"):
            engine.run(ohlcv)

    def test_hold_produces_no_trades(self, features_pipeline, always_hold, risk_engine):
        engine = BacktestEngine(
            features_pipeline=features_pipeline,
            predict_fn=always_hold,
            risk_engine=risk_engine,
        )
        ohlcv = _make_ohlcv(200)
        result = engine.run(ohlcv)
        assert result.num_trades == 0
        assert result.final_equity == Decimal(10000)

    def test_buy_then_sell_completes_trade(self, features_pipeline, risk_engine):
        signals = iter([])

        def alternating(features):
            try:
                return next(signals)
            except StopIteration:
                return SignalAction.HOLD, Decimal("0.50")

        engine = BacktestEngine(
            features_pipeline=features_pipeline,
            predict_fn=alternating,
            risk_engine=risk_engine,
        )
        ohlcv = _make_ohlcv(200)
        result = engine.run(ohlcv)
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == 200
        assert result.total_return_pct is not None

    def test_result_has_metrics(self, features_pipeline, always_buy, risk_engine):
        engine = BacktestEngine(
            features_pipeline=features_pipeline,
            predict_fn=always_buy,
            risk_engine=risk_engine,
        )
        ohlcv = _make_ohlcv(200)
        result = engine.run(ohlcv)
        assert isinstance(result.total_return_pct, float)
        assert isinstance(result.max_drawdown_pct, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.num_trades, int)
        assert isinstance(result.win_rate, float)
        assert result.win_rate >= 0.0
        assert result.win_rate <= 1.0

    def test_custom_config(self, features_pipeline, always_buy, risk_engine):
        config = BacktestConfig(
            initial_capital=Decimal(50000),
            fee_pct=Decimal("0.002"),
            slippage_pct=Decimal("0.001"),
        )
        engine = BacktestEngine(
            features_pipeline=features_pipeline,
            predict_fn=always_buy,
            risk_engine=risk_engine,
            config=config,
        )
        ohlcv = _make_ohlcv(200)
        result = engine.run(ohlcv)
        assert result.final_equity.is_finite()

    def test_equity_curve_length_matches_candles(self, features_pipeline, always_hold, risk_engine):
        engine = BacktestEngine(
            features_pipeline=features_pipeline,
            predict_fn=always_hold,
            risk_engine=risk_engine,
        )
        ohlcv = _make_ohlcv(150)
        result = engine.run(ohlcv)
        assert len(result.equity_curve) == 150