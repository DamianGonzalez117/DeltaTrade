"""
Modelos de la Capa de Inteligencia (IA): puntuaciones de sentimiento (NLP)
y señales generadas por los modelos cuantitativos/híbridos.
"""
from datetime import datetime
from decimal import Decimal

from sqlalchemy import String, DateTime, Numeric, ForeignKey, Text, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from trading_system.core.enums import SignalAction
from trading_system.db.base import Base, TimestampMixin


class SentimentScore(Base):
    """
    Resultado de aplicar FinBERT/LLM a una noticia o titular. `asset_id` es
    opcional: una noticia puede referirse a un activo específico o al
    mercado en general (ej. política monetaria de Banxico).
    """
    __tablename__ = "sentiment_scores"
    __table_args__ = (
        Index("ix_sentiment_asset_published", "asset_id", "published_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int | None] = mapped_column(
        ForeignKey("assets.id", ondelete="SET NULL"), nullable=True
    )
    source: Mapped[str] = mapped_column(String(64), nullable=False)      # p.ej. "reuters_rss", "gpt_summary"
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    sentiment_score: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)  # -1.0000 a 1.0000
    sentiment_label: Mapped[str] = mapped_column(String(16), nullable=False)          # positive/negative/neutral
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)               # p.ej. "ProsusAI/finbert"

    raw_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AISignal(Base, TimestampMixin):
    """
    Salida del motor de inferencia híbrido (técnico + sentimiento) para un
    activo en un momento dado. Es el input crudo que recibe el Motor de
    Riesgo antes de decidir si se ejecuta algo.
    """
    __tablename__ = "ai_signals"
    __table_args__ = (
        Index("ix_signals_asset_generated", "asset_id", "generated_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    model_name: Mapped[str] = mapped_column(String(64), nullable=False)     # p.ej. "xgboost_v1", "tft_v1"
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)

    action: Mapped[SignalAction] = mapped_column(String(8), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)  # 0.0000 a 1.0000

    # Snapshot de los inputs que originaron la señal, para trazabilidad y reentrenamiento.
    technical_features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    sentiment_score_id: Mapped[int | None] = mapped_column(
        ForeignKey("sentiment_scores.id", ondelete="SET NULL"), nullable=True
    )

    asset: Mapped["Asset"] = relationship()
    sentiment_score: Mapped["SentimentScore | None"] = relationship()
    risk_evaluation: Mapped["RiskEvaluation | None"] = relationship(
        back_populates="signal", uselist=False
    )
