"""Contrato base para analizadores de sentimiento."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class SentimentResult:
    """Resultado del análisis de sentimiento."""

    score: Decimal  # -1.0 (muy negativo) a 1.0 (muy positivo)
    label: str       # "positive", "negative", "neutral"


class SentimentAnalyzer(ABC):
    """Interfaz para analizadores de sentimiento NLP."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def analyze(self, headlines: list[str]) -> SentimentResult:
        ...