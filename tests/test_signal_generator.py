"""TDD — Tests for SignalGenerator (RED→GREEN)."""
from __future__ import annotations

from decimal import Decimal

import pytest

from trading_system.core.enums import SignalAction
from trading_system.services.signal_generator import SignalGenerator


@pytest.fixture
def generator() -> SignalGenerator:
    return SignalGenerator(
        rsi_oversold=30.0,
        rsi_overbought=70.0,
        rsi_extreme_low=20.0,
        rsi_extreme_high=80.0,
        min_confidence=0.60,
        max_confidence=0.95,
    )


class TestGenerate:
    def test_buy_when_oversold_and_macd_bullish(self, generator: SignalGenerator):
        features = {"rsi_14": 25.0, "macd_diff": 120.5}
        signal = generator.generate(features)
        assert signal.action == SignalAction.BUY
        assert signal.confidence > Decimal("0.60")
        assert "sobreventa" in signal.reason.lower()

    def test_sell_when_overbought_and_macd_bearish(self, generator: SignalGenerator):
        features = {"rsi_14": 75.0, "macd_diff": -80.3}
        signal = generator.generate(features)
        assert signal.action == SignalAction.SELL
        assert signal.confidence > Decimal("0.60")
        assert "sobrecompra" in signal.reason.lower()

    def test_hold_when_rsi_oversold_but_macd_not_bullish(self, generator: SignalGenerator):
        features = {"rsi_14": 25.0, "macd_diff": -50.0}
        signal = generator.generate(features)
        assert signal.action == SignalAction.HOLD

    def test_hold_when_rsi_overbought_but_macd_not_bearish(self, generator: SignalGenerator):
        features = {"rsi_14": 75.0, "macd_diff": 50.0}
        signal = generator.generate(features)
        assert signal.action == SignalAction.HOLD

    def test_hold_when_rsi_neutral(self, generator: SignalGenerator):
        features = {"rsi_14": 50.0, "macd_diff": 10.0}
        signal = generator.generate(features)
        assert signal.action == SignalAction.HOLD
        assert signal.confidence == Decimal("0.50")

    def test_hold_when_missing_features(self, generator: SignalGenerator):
        signal = generator.generate({})
        assert signal.action == SignalAction.HOLD

    def test_extreme_oversold_gives_max_confidence(self, generator: SignalGenerator):
        features = {"rsi_14": 15.0, "macd_diff": 200.0}
        signal = generator.generate(features)
        assert signal.action == SignalAction.BUY
        assert signal.confidence == Decimal("0.95")

    def test_extreme_overbought_gives_max_confidence(self, generator: SignalGenerator):
        features = {"rsi_14": 85.0, "macd_diff": -200.0}
        signal = generator.generate(features)
        assert signal.action == SignalAction.SELL
        assert signal.confidence == Decimal("0.95")

    def test_threshold_oversold_gives_min_confidence(self, generator: SignalGenerator):
        features = {"rsi_14": 29.9, "macd_diff": 0.01}
        signal = generator.generate(features)
        assert signal.action == SignalAction.BUY
        assert signal.confidence >= Decimal("0.60")
        assert signal.confidence < Decimal("0.70")

    def test_signal_is_frozen(self, generator: SignalGenerator):
        signal = generator.generate({"rsi_14": 50.0})
        with pytest.raises((AttributeError, TypeError)):
            signal.action = SignalAction.BUY

    def test_signal_has_rsi_and_macd_fields(self, generator: SignalGenerator):
        signal = generator.generate({"rsi_14": 25.0, "macd_diff": 100.0})
        assert signal.rsi == 25.0
        assert signal.macd_diff == 100.0


class TestEmaVolumeRule:
    """Regla alternativa de crecimiento: EMA9>EMA21 + MACD alcista + volumen alto."""

    @pytest.fixture
    def growth_features(self) -> dict[str, float]:
        return {
            "rsi_14": 55.0,  # neutral: la regla RSI no dispara
            "macd_diff": 80.0,
            "ema_9": 510_000.0,
            "ema_21": 505_000.0,
            "volume_ratio_20": 2.0,
        }

    def test_buy_when_ema_bullish_and_volume_spike(self, generator, growth_features):
        signal = generator.generate(growth_features)
        assert signal.action == SignalAction.BUY
        assert signal.confidence >= Decimal("0.60")
        assert "ema" in signal.reason.lower()

    def test_hold_when_ema_bearish(self, generator, growth_features):
        features = {**growth_features, "ema_9": 500_000.0}
        signal = generator.generate(features)
        assert signal.action == SignalAction.HOLD

    def test_hold_when_macd_not_bullish(self, generator, growth_features):
        features = {**growth_features, "macd_diff": -10.0}
        signal = generator.generate(features)
        assert signal.action == SignalAction.HOLD

    def test_hold_when_volume_below_threshold(self, generator, growth_features):
        features = {**growth_features, "volume_ratio_20": 1.2}
        signal = generator.generate(features)
        assert signal.action == SignalAction.HOLD

    def test_higher_volume_gives_higher_confidence(self, generator, growth_features):
        low = generator.generate({**growth_features, "volume_ratio_20": 1.6})
        high = generator.generate({**growth_features, "volume_ratio_20": 3.5})
        assert high.confidence > low.confidence

    def test_extreme_volume_caps_at_max_confidence(self, generator, growth_features):
        signal = generator.generate({**growth_features, "volume_ratio_20": 10.0})
        assert signal.confidence == Decimal("0.95")

    def test_hold_when_ema_features_missing(self, generator, growth_features):
        features = {k: v for k, v in growth_features.items() if not k.startswith("ema")}
        signal = generator.generate(features)
        assert signal.action == SignalAction.HOLD

    def test_rsi_rule_still_fires_without_ema_features(self, generator):
        features = {"rsi_14": 25.0, "macd_diff": 120.0}
        signal = generator.generate(features)
        assert signal.action == SignalAction.BUY
        assert "sobreventa" in signal.reason.lower()
