"""Motor de inferencia híbrido del sistema de trading.

Exporta:
  - ``HybridInferenceEngine``: combina técnico + sentimiento.
  - ``InferenceInput`` / ``InferenceOutput``: tipos de datos.
  - ``SentimentAnalyzer`` / ``SentimentResult``: interfaz y resultado de sentimiento.
  - ``TechnicalModel`` / ``TechnicalResult``: interfaz y resultado técnico.
  - ``FinBERTSentimentAnalyzer``: implementación con FinBERT.
  - ``XGBoostTechnicalModel``: implementación con XGBoost.
"""
from trading_system.inference.engine import HybridInferenceEngine
from trading_system.inference.models import (
    TechnicalModel,
    TechnicalResult,
    XGBoostTechnicalModel,
)
from trading_system.inference.sentiment import (
    FinBERTSentimentAnalyzer,
    SentimentAnalyzer,
    SentimentResult,
)
from trading_system.inference.types import InferenceInput, InferenceOutput

__all__ = [
    "FinBERTSentimentAnalyzer",
    "HybridInferenceEngine",
    "InferenceInput",
    "InferenceOutput",
    "SentimentAnalyzer",
    "SentimentResult",
    "TechnicalModel",
    "TechnicalResult",
    "XGBoostTechnicalModel",
]