"""
Tipos de datos del motor de inferencia híbrido.

``InferenceInput`` agrupa features técnicas + headlines para sentimiento.
``InferenceOutput`` trae la señal híbrida combinada: action, confidence,
y el desglose de contribuciones técnica y de sentimiento.
"""
from dataclasses import dataclass
from decimal import Decimal

from trading_system.core.enums import AssetClass, MarketVenue, SignalAction


@dataclass(frozen=True, slots=True)
class InferenceInput:
    """Entrada del motor de inferencia híbrido."""

    asset_symbol: str
    asset_venue: MarketVenue
    asset_class: AssetClass
    technical_features: dict[str, float]
    headlines: list[str] | None = None


@dataclass(frozen=True, slots=True)
class InferenceOutput:
    """Salida del motor de inferencia híbrido."""

    action: SignalAction
    confidence: Decimal
    model_name: str
    model_version: str
    technical_score: Decimal
    sentiment_score: Decimal
    technical_weight: Decimal
    sentiment_weight: Decimal