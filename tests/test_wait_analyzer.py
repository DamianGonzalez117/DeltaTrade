"""TDD — Tests para WaitAnalyzer (estadística de "¿qué pasaría si espero?").

Datos sintéticos deterministas: series estrictamente crecientes/decrecientes
producen RSI extremo constante, de modo que los casos similares y los
resultados forward son 100% conocidos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.services.wait_analyzer import WaitAnalyzer


def _ohlcv_from_closes(closes: list[float]) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    return pd.DataFrame({
        "open": closes,
        "high": closes * 1.005,
        "low": closes * 0.995,
        "close": closes,
        "volume": [1.0] * n,
    })


def _uptrend_ohlcv(n: int = 120, step: float = 1.0) -> pd.DataFrame:
    closes = [100.0 + step * i for i in range(n)]
    return _ohlcv_from_closes(closes)


def _downtrend_ohlcv(n: int = 120, step: float = 1.0) -> pd.DataFrame:
    closes = [300.0 - step * i for i in range(n)]
    return _ohlcv_from_closes(closes)


class TestAnalyzeBasic:
    def test_returns_none_without_data(self):
        analyzer = WaitAnalyzer()
        assert analyzer.analyze(None, current_rsi=50.0, timeframe_minutes=60) is None

    def test_returns_none_with_short_history(self):
        analyzer = WaitAnalyzer()
        short = _uptrend_ohlcv(n=20)
        assert analyzer.analyze(short, current_rsi=99.0, timeframe_minutes=60) is None

    def test_uptrend_recommends_wait(self):
        analyzer = WaitAnalyzer(horizon_candles=5, max_horizon_candles=5)
        stats = analyzer.analyze(
            _uptrend_ohlcv(), current_rsi=100.0, timeframe_minutes=60
        )

        assert stats is not None
        assert stats.rise_share == 1.0
        assert stats.fall_share == 0.0
        assert stats.avg_gain_pct > 0
        assert stats.recommendation == "esperar"
        assert stats.trend == "alcista"
        assert stats.horizon_candles == 5
        assert stats.horizon_minutes == 5 * 60
        assert stats.n_cases > 0

    def test_downtrend_recommends_sell(self):
        analyzer = WaitAnalyzer(horizon_candles=5, max_horizon_candles=5)
        stats = analyzer.analyze(
            _downtrend_ohlcv(), current_rsi=0.0, timeframe_minutes=60
        )

        assert stats is not None
        assert stats.fall_share == 1.0
        assert stats.rise_share == 0.0
        assert stats.avg_loss_pct < 0
        assert stats.recommendation == "vender"
        assert stats.trend == "bajista"

    def test_current_rsi_defaults_to_last_of_series(self):
        analyzer = WaitAnalyzer(horizon_candles=5, max_horizon_candles=5)
        # sin current_rsi: usa el RSI final (100 en la tendencia alcista)
        stats = analyzer.analyze(_uptrend_ohlcv(), current_rsi=None, timeframe_minutes=60)

        assert stats is not None
        assert stats.rise_share == 1.0


class TestAdaptiveHorizon:
    """El horizonte crece con el historial disponible, acotado al rango."""

    def test_horizon_grows_with_history(self):
        # 400 velas // 8 = 50 → acotado al máximo de 48
        analyzer = WaitAnalyzer()
        stats = analyzer.analyze(_uptrend_ohlcv(n=400), current_rsi=100.0, timeframe_minutes=1)

        assert stats is not None
        assert stats.horizon_candles == 48
        assert stats.horizon_minutes == 48  # 48 velas de 1m

    def test_horizon_floor_with_little_history(self):
        # 120 velas // 8 = 15 (dentro del rango 6..48)
        analyzer = WaitAnalyzer()
        stats = analyzer.analyze(_uptrend_ohlcv(n=120), current_rsi=100.0, timeframe_minutes=1)

        assert stats is not None
        assert stats.horizon_candles == 15

    def test_horizon_minimum_with_tiny_history(self):
        # 48 velas // 8 = 6 → piso de 6
        analyzer = WaitAnalyzer()
        stats = analyzer.analyze(_uptrend_ohlcv(n=48), current_rsi=100.0, timeframe_minutes=5)

        assert stats is not None
        assert stats.horizon_candles == 6
        assert stats.horizon_minutes == 30  # 6 velas de 5m

    def test_custom_divisor_and_bounds(self):
        analyzer = WaitAnalyzer(
            horizon_candles=10, max_horizon_candles=20, history_divisor=4
        )
        # 120 velas // 4 = 30 → acotado a 20
        stats = analyzer.analyze(_uptrend_ohlcv(n=120), current_rsi=100.0, timeframe_minutes=60)

        assert stats is not None
        assert stats.horizon_candles == 20

    def test_invalid_bounds_raise(self):
        import pytest

        with pytest.raises(ValueError, match="max_horizon_candles"):
            WaitAnalyzer(horizon_candles=10, max_horizon_candles=5)
        with pytest.raises(ValueError, match="history_divisor"):
            WaitAnalyzer(history_divisor=0)


class TestAnalyzeStatistics:
    def test_shares_sum_one_and_best_worst_consistent(self):
        rng = np.random.default_rng(7)
        n = 300
        closes = 100 + np.cumsum(rng.standard_normal(n) * 0.8)
        ohlcv = _ohlcv_from_closes(list(closes))

        analyzer = WaitAnalyzer(horizon_candles=6, min_cases=3)
        stats = analyzer.analyze(ohlcv, current_rsi=50.0, timeframe_minutes=60)

        assert stats is not None
        assert abs(stats.rise_share + stats.fall_share - 1.0) < 1e-9
        # el máximo alcanzado nunca puede ser menor que el retorno final promedio
        if stats.rise_share > 0:
            assert stats.best_case_pct >= stats.avg_gain_pct
        if stats.fall_share > 0:
            assert stats.worst_case_pct <= stats.avg_loss_pct

    def test_band_widening_finds_more_cases(self):
        """RSI 50 con banda estrecha puede no juntar casos en serie sintética
        con RSIs extremos; el ensanchado debe recuperar casos."""
        # serie que alterna subidas y bajadas → RSI oscilante
        closes = []
        price = 100.0
        for i in range(200):
            price *= 1.03 if i % 2 == 0 else 0.97
            closes.append(price)
        ohlcv = _ohlcv_from_closes(closes)

        analyzer = WaitAnalyzer(horizon_candles=4, max_horizon_candles=4, min_cases=10, rsi_band=1.0)
        stats = analyzer.analyze(ohlcv, current_rsi=50.0, timeframe_minutes=15)

        assert stats is not None
        assert stats.n_cases >= 10
        assert stats.horizon_minutes == stats.horizon_candles * 15

    def test_trend_neutral_on_gentle_oscillation(self):
        import math

        closes = [100.0 + 0.05 * math.sin(i / 8) for i in range(200)]
        ohlcv = _ohlcv_from_closes(closes)

        analyzer = WaitAnalyzer(horizon_candles=5, max_horizon_candles=5)
        stats = analyzer.analyze(ohlcv, current_rsi=50.0, timeframe_minutes=60)

        assert stats is not None
        # oscilación suave: EMA9 ≈ EMA21 → tendencia neutral
        assert stats.trend == "neutral"
        # con precios casi planos, las recomendaciones no explotan
        assert stats.recommendation in {"esperar", "vender", "neutro"}
