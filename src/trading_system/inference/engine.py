"""
Motor de inferencia híbrido: combina señales técnicas + sentimiento.

Regla de combinación:
  1. El modelo técnico produce (action, confidence).
  2. Se normaliza a [-1, 1]: buy → +confidence, sell → -confidence, hold → 0.
  3. El analizador de sentimiento produce score [-1, 1].
  4. hybrid_score = technical_weight * tech_score + sentiment_weight * sentiment_score
  5. action = BUY if hybrid > threshold, SELL if < -threshold, else HOLD.
  6. confidence = |hybrid_score|.

DIP: depende de ``TechnicalModel`` y ``SentimentAnalyzer`` (ABCs).
OCP: nuevo modelo o analyzer = inyectar por constructor.
"""
from __future__ import annotations

from decimal import Decimal

import structlog

from trading_system.core.enums import SignalAction
from trading_system.inference.models.base import TechnicalModel
from trading_system.inference.sentiment.base import SentimentAnalyzer
from trading_system.inference.types import InferenceInput, InferenceOutput

logger = structlog.get_logger(__name__)

_ACTION_TO_SCORE = {
    SignalAction.BUY: Decimal(1),
    SignalAction.HOLD: Decimal(0),
    SignalAction.SELL: Decimal(-1),
}


class HybridInferenceEngine:
    """Combina modelo técnico + analizador de sentimiento en una señal híbrida."""

    def __init__(
        self,
        *,
        technical_model: TechnicalModel,
        sentiment_analyzer: SentimentAnalyzer,
        technical_weight: Decimal = Decimal("0.6"),
        sentiment_weight: Decimal = Decimal("0.4"),
        decision_threshold: Decimal = Decimal("0.15"),
        model_version: str = "0.1.0",
    ) -> None:
        self._tech_model = technical_model
        self._sentiment = sentiment_analyzer
        self._tech_weight = technical_weight
        self._sentiment_weight = sentiment_weight
        self._threshold = decision_threshold
        self._model_version = model_version

    @property
    def technical_weight(self) -> Decimal:
        return self._tech_weight

    @property
    def sentiment_weight(self) -> Decimal:
        return self._sentiment_weight

    def infer(self, inp: InferenceInput) -> InferenceOutput:
        tech_result = self._tech_model.predict(inp.technical_features)

        headlines = inp.headlines if inp.headlines is not None else []
        sent_result = self._sentiment.analyze(headlines)

        tech_score = _ACTION_TO_SCORE[tech_result.action] * tech_result.confidence
        sent_score = sent_result.score

        hybrid_score = (
            self._tech_weight * tech_score
            + self._sentiment_weight * sent_score
        )

        if hybrid_score > self._threshold:
            action = SignalAction.BUY
        elif hybrid_score < -self._threshold:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD

        confidence = abs(hybrid_score)
        if confidence > 1:
            confidence = Decimal(1)

        logger.info(
            "hybrid_inference.result",
            symbol=inp.asset_symbol,
            action=action.value,
            confidence=str(confidence),
            tech_score=str(tech_score),
            sent_score=str(sent_score),
            hybrid=str(hybrid_score),
        )

        return InferenceOutput(
            action=action,
            confidence=confidence,
            model_name=f"hybrid:{self._tech_model.model_name}+{self._sentiment.model_name}",
            model_version=self._model_version,
            technical_score=tech_score,
            sentiment_score=sent_score,
            technical_weight=self._tech_weight,
            sentiment_weight=self._sentiment_weight,
        )