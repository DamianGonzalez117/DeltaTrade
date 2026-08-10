"""TDD — Tests for HybridInferenceEngine (RED phase).

El motor híbrido combina:
  1. TechnicalModel → score técnico (action + confidence)
  2. SentimentAnalyzer → score de sentimiento [-1, 1]
  3. Combina con pesos configurables → InferenceOutput

Regla de combinación:
  - score técnico normalizado a [-1, 1] (buy=+1, sell=-1, hold=0) * confidence
  - hybrid_score = technical_weight * tech_score + sentiment_weight * sentiment_score
  - action = BUY if hybrid > threshold, SELL if < -threshold, else HOLD
  - confidence = |hybrid_score|
"""
from decimal import Decimal
from unittest.mock import MagicMock

from trading_system.core.enums import AssetClass, MarketVenue, SignalAction
from trading_system.inference.engine import HybridInferenceEngine
from trading_system.inference.models.base import TechnicalModel, TechnicalResult
from trading_system.inference.sentiment.base import SentimentAnalyzer, SentimentResult
from trading_system.inference.types import InferenceInput, InferenceOutput


def _make_input(**overrides) -> InferenceInput:
    defaults = {
        "asset_symbol": "btc_mxn",
        "asset_venue": MarketVenue.BITSO,
        "asset_class": AssetClass.CRYPTO,
        "technical_features": {"rsi_14": 65.0, "atr_14": 15000.0},
        "headlines": ["Bitcoin rally continues"],
    }
    defaults.update(overrides)
    return InferenceInput(**defaults)


def _mock_tech_model(action=SignalAction.BUY, confidence=0.8) -> MagicMock:
    model = MagicMock(spec=TechnicalModel)
    model.predict.return_value = TechnicalResult(
        action=action,
        confidence=Decimal(str(confidence)),
    )
    model.model_name = "test_xgboost"
    return model


def _mock_sentiment_analyzer(score=0.5, label="positive") -> MagicMock:
    analyzer = MagicMock(spec=SentimentAnalyzer)
    analyzer.analyze.return_value = SentimentResult(
        score=Decimal(str(score)),
        label=label,
    )
    analyzer.model_name = "test_finbert"
    return analyzer


class TestHybridEngineConstruction:
    def test_default_weights_sixty_forty(self):
        engine = HybridInferenceEngine(
            technical_model=_mock_tech_model(),
            sentiment_analyzer=_mock_sentiment_analyzer(),
        )
        assert engine.technical_weight == Decimal("0.6")
        assert engine.sentiment_weight == Decimal("0.4")


class TestHybridEngineBuy:
    def test_buy_signal_when_both_positive(self):
        engine = HybridInferenceEngine(
            technical_model=_mock_tech_model(action=SignalAction.BUY, confidence=0.8),
            sentiment_analyzer=_mock_sentiment_analyzer(score=0.6, label="positive"),
        )
        result = engine.infer(_make_input())

        assert isinstance(result, InferenceOutput)
        assert result.action == SignalAction.BUY
        assert result.confidence > 0
        assert result.technical_score > 0
        assert result.sentiment_score > 0

    def test_buy_signal_when_technical_positive_sentiment_neutral(self):
        engine = HybridInferenceEngine(
            technical_model=_mock_tech_model(action=SignalAction.BUY, confidence=0.9),
            sentiment_analyzer=_mock_sentiment_analyzer(score=0.0, label="neutral"),
        )
        result = engine.infer(_make_input())
        assert result.action == SignalAction.BUY
        assert result.confidence > 0


class TestHybridEngineSell:
    def test_sell_signal_when_both_negative(self):
        engine = HybridInferenceEngine(
            technical_model=_mock_tech_model(action=SignalAction.SELL, confidence=0.8),
            sentiment_analyzer=_mock_sentiment_analyzer(score=-0.7, label="negative"),
        )
        result = engine.infer(_make_input())

        assert result.action == SignalAction.SELL
        assert result.confidence > 0
        assert result.technical_score < 0
        assert result.sentiment_score < 0


class TestHybridEngineHold:
    def test_hold_signal_when_scores_near_zero(self):
        engine = HybridInferenceEngine(
            technical_model=_mock_tech_model(action=SignalAction.HOLD, confidence=0.5),
            sentiment_analyzer=_mock_sentiment_analyzer(score=0.0, label="neutral"),
        )
        result = engine.infer(_make_input())

        assert result.action == SignalAction.HOLD

    def test_hold_when_tech_buy_sentiment_sell_cancel_out(self):
        engine = HybridInferenceEngine(
            technical_model=_mock_tech_model(action=SignalAction.BUY, confidence=0.5),
            sentiment_analyzer=_mock_sentiment_analyzer(score=-0.8, label="negative"),
        )
        result = engine.infer(_make_input())

        assert result.action == SignalAction.HOLD

    def test_no_headlines_still_produces_signal(self):
        engine = HybridInferenceEngine(
            technical_model=_mock_tech_model(action=SignalAction.BUY, confidence=0.8),
            sentiment_analyzer=_mock_sentiment_analyzer(score=0.0, label="neutral"),
        )
        result = engine.infer(_make_input(headlines=None))
        assert result.action == SignalAction.BUY


class TestHybridEngineOutput:
    def test_output_contains_model_name_and_version(self):
        engine = HybridInferenceEngine(
            technical_model=_mock_tech_model(),
            sentiment_analyzer=_mock_sentiment_analyzer(),
        )
        result = engine.infer(_make_input())
        assert "hybrid" in result.model_name.lower()
        assert result.model_version

    def test_output_weights_match_engine_config(self):
        engine = HybridInferenceEngine(
            technical_model=_mock_tech_model(),
            sentiment_analyzer=_mock_sentiment_analyzer(),
            technical_weight=Decimal("0.7"),
            sentiment_weight=Decimal("0.3"),
        )
        result = engine.infer(_make_input())
        assert result.technical_weight == Decimal("0.7")
        assert result.sentiment_weight == Decimal("0.3")


class TestThresholdConfigurable:
    def test_custom_threshold_changes_hold_boundary(self):
        """With a higher threshold, signals need stronger conviction."""
        engine = HybridInferenceEngine(
            technical_model=_mock_tech_model(action=SignalAction.HOLD, confidence=0.5),
            sentiment_analyzer=_mock_sentiment_analyzer(score=0.2, label="positive"),
            decision_threshold=Decimal("0.8"),
        )
        result = engine.infer(_make_input())
        # hybrid_score = 0.6*0 + 0.4*0.2 = 0.08 < 0.8 → HOLD
        assert result.action == SignalAction.HOLD