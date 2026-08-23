"""
Modelos de la Capa de Ejecución: toda orden real enviada a Bitso/GBM debe
colgar de un RiskEvaluation aprobado. Aquí también vive el estado agregado
del portafolio (posiciones abiertas y snapshots históricos de equity).
"""
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from trading_system.core.enums import (
    MarketVenue,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionStatus,
)
from trading_system.db.base import Base, TimestampMixin


class Order(Base, TimestampMixin):
    __tablename__ = "orders"
    __table_args__ = (
        Index("ix_orders_asset_status", "asset_id", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="RESTRICT"), nullable=False)
    risk_evaluation_id: Mapped[int | None] = mapped_column(
        ForeignKey("risk_evaluations.id", ondelete="SET NULL"), unique=True, nullable=True
    )

    venue: Mapped[MarketVenue] = mapped_column(String(16), nullable=False)
    side: Mapped[OrderSide] = mapped_column(String(8), nullable=False)
    order_type: Mapped[OrderType] = mapped_column(String(16), nullable=False)
    status: Mapped[OrderStatus] = mapped_column(String(20), default=OrderStatus.PENDING, nullable=False)

    quantity: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    price: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)  # nulo en órdenes de mercado

    external_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)  # id devuelto por el exchange
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    asset: Mapped["Asset"] = relationship()
    risk_evaluation: Mapped["RiskEvaluation | None"] = relationship(back_populates="order")
    fills: Mapped[list["Fill"]] = relationship(back_populates="order")


class Fill(Base):
    """Ejecución parcial o total de una orden (una orden puede tener 1..N fills)."""
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), nullable=False)

    filled_quantity: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    filled_price: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    fee: Mapped[Decimal] = mapped_column(Numeric(24, 8), default=0, nullable=False)
    fee_currency: Mapped[str] = mapped_column(String(8), nullable=False)
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    order: Mapped["Order"] = relationship(back_populates="fills")


class Position(Base, TimestampMixin):
    __tablename__ = "positions"
    __table_args__ = (
        Index("ix_positions_asset_status", "asset_id", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="RESTRICT"), nullable=False)

    status: Mapped[PositionStatus] = mapped_column(String(8), default=PositionStatus.OPEN, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    avg_entry_price: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(24, 8), default=0, nullable=False)

    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    asset: Mapped["Asset"] = relationship()


class PortfolioSnapshot(Base):
    """Foto periódica del estado global del portafolio (para drawdown/reportes)."""
    __tablename__ = "portfolio_snapshots"
    __table_args__ = (
        Index("ix_portfolio_snapshots_ts", "ts"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    total_equity_mxn: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    cash_balance_mxn: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    positions_value_mxn: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    drawdown_pct: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False)
