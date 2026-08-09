"""TDD — Tests for inference types: InferenceInput and InferenceOutput (RED phase).

``InferenceInput`` agrupa features técnicas + headlines/noticias.
``InferenceOutput`` trae la señal híbrida: action, confidence, y optionally
the breakdown of technical and sentiment contributions.
"""
from decimal import Decimal

import pytest

from trading_system.core.enums import AssetClass, MarketVenue, SignalAction
from trading_system.inference.types import InferenceInput, InferenceOutput


class TestInferenceInput:
    def test_creation_with_features_and_headlines(self):
        inp = InferenceInput(
            asset_symbol="btc_mxn",
            asset_venue=MarketVenue.BITSO,
            asset_class=AssetClass.CRYPTO,
            technical_features={"rsi_14": 65.2, "atr_14": 15000.0, "macd": 120.5},
            headlines=["Bitcoin rally continues as ETF inflows surge"],
        )
        assert inp.asset_symbol == "btc_mxn"
        assert inp.technical_features["rsi_14"] == 65.2
        assert len(inp.headlines) == 1

    def test_headlines_default_empty(self):
        inp = InferenceInput(
            asset_symbol="eth_mxn",
            asset_venue=MarketVenue.BITSO,
            asset_class=AssetClass.CRYPTO,
            technical_features={"rsi_14": 30.0},
        )
        assert inp.headlines is None

    def test_is_frozen(self):
        inp = InferenceInput(
            asset_symbol="btc_mxn",
            asset_venue=MarketVenue.BITSO,
            asset_class=AssetClass.CRYPTO,
            technical_features={},
        )
        with pytest.raises(AttributeError):
            inp.asset_symbol = "eth_mxn"  # type: ignore[misc]


class TestInferenceOutput:
    def test_buy_signal_with_high_confidence(self):
        out = InferenceOutput(
            action=SignalAction.BUY,
            confidence=Decimal("0.87"),
            model_name="hybrid_v1",
            model_version="0.1.0",
            technical_score=Decimal("0.75"),
            sentiment_score=Decimal("0.80"),
            technical_weight=Decimal("0.6"),
            sentiment_weight=Decimal("0.4"),
        )
        assert out.action == SignalAction.BUY
        assert out.confidence == Decimal("0.87")
        assert out.technical_score == Decimal("0.75")
        assert out.sentiment_score == Decimal("0.80")

    def test_hold_signal_with_low_confidence(self):
        out = InferenceOutput(
            action=SignalAction.HOLD,
            confidence=Decimal("0.42"),
            model_name="hybrid_v1",
            model_version="0.1.0",
            technical_score=Decimal("0.50"),
            sentiment_score=Decimal("0.30"),
            technical_weight=Decimal("0.6"),
            sentiment_weight=Decimal("0.4"),
        )
        assert out.action == SignalAction.HOLD
        assert out.confidence == Decimal("0.42")

    def test_confidence_in_valid_range(self):
        out = InferenceOutput(
            action=SignalAction.SELL,
            confidence=Decimal("0.65"),
            model_name="hybrid_v1",
            model_version="0.1.0",
            technical_score=Decimal("0.20"),
            sentiment_score=Decimal("0.10"),
            technical_weight=Decimal("0.6"),
            sentiment_weight=Decimal("0.4"),
        )
        assert Decimal(0) <= out.confidence <= Decimal(1)

    def test_is_frozen(self):
        out = InferenceOutput(
            action=SignalAction.BUY,
            confidence=Decimal("0.8"),
            model_name="hybrid_v1",
            model_version="0.1.0",
            technical_score=Decimal("0.7"),
            sentiment_score=Decimal("0.6"),
            technical_weight=Decimal("0.6"),
            sentiment_weight=Decimal("0.4"),
        )
        with pytest.raises(AttributeError):
            out.action = SignalAction.SELL  # type: ignore[misc]