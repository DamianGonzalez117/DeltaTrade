"""
Modelo de la Capa de Riesgo: registra la decisión determinista tomada sobre
cada señal de IA, ANTES de que exista cualquier orden. Ninguna orden debe
poder crearse sin una fila aquí con decision='approved' o 'adjusted'.
"""
from datetime import datetime
from decimal import Decimal

from sqlalchemy import String, DateTime, Numeric, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from trading_system.core.enums import RiskDecision
from trading_system.db.base import Base


class RiskEvaluation(Base):
    __tablename__ = "risk_evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)
    signal_id: Mapped[int] = mapped_column(
        ForeignKey("ai_signals.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    decision: Mapped[RiskDecision] = mapped_column(String(16), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)  # regla(s) que gatillaron la decisión

    # Parámetros calculados por el motor determinista (nulos si fue rechazada).
    position_size: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)
    max_capital_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    stop_loss_price: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)
    take_profit_price: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)
    atr_at_evaluation: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)

    signal: Mapped["AISignal"] = relationship(back_populates="risk_evaluation")
    order: Mapped["Order | None"] = relationship(back_populates="risk_evaluation", uselist=False)
