"""Analizador de sentimiento con FinBERT (HuggingFace transformers).

DIP: el pipeline de transformers se inyecta por constructor para que
los tests pasen un mock sin descargar el modelo.
"""
from collections.abc import Callable
from decimal import Decimal

from trading_system.inference.sentiment.base import SentimentAnalyzer, SentimentResult

_LABEL_MAP = {"positive": 1, "neutral": 0, "negative": -1}


class FinBERTSentimentAnalyzer(SentimentAnalyzer):
    """Analizador de sentimiento basado en ProsusAI/finbert."""

    def __init__(
        self,
        pipeline: Callable[[list[str]], list],
        model_name: str = "ProsusAI/finbert",
    ) -> None:
        self._pipeline = pipeline
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return f"finbert:{self._model_name}"

    def analyze(self, headlines: list[str]) -> SentimentResult:
        if not headlines:
            return SentimentResult(score=Decimal(0), label="neutral")

        raw_results = self._pipeline(headlines)
        total_score = 0.0
        label_counts: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}

        for result_set in raw_results:
            if isinstance(result_set, list):
                best = max(result_set, key=lambda r: r.get("score", 0))
            else:
                best = result_set
            label = best["label"].lower()
            score = best["score"]
            total_score += score * _LABEL_MAP.get(label, 0)
            if label in label_counts:
                label_counts[label] += 1

        avg_score = total_score / len(headlines)
        dominant_label = max(label_counts, key=lambda k: label_counts[k])

        return SentimentResult(
            score=Decimal(str(avg_score)),
            label=dominant_label,
        )