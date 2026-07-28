"""
Modelos de la Capa de Datos: catálogo de activos y series de precios OHLCV.
"""
from datetime import datetime
from decimal import Decimal

from sqlalchemy import String, Boolean, DateTime, Numeric, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from trading_system.core.enums import MarketVenue, AssetClass, TimeFrame
from trading_system.db.base import Base, TimestampMixin


class Asset(Base, TimestampMixin):
    """
    Catálogo único de instrumentos operables, sin importar el venue.

    Ej: BTC en Bitso -> symbol='btc_mxn', venue='bitso', asset_class='crypto'
        Un ETF en GBM -> symbol='NAFTRAC', venue='gbm', asset_class='etf'
    """
    __tablename__ = "assets"
    __table_args__ = (
        UniqueConstraint("symbol", "venue", name="uq_assets_symbol_venue"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    venue: Mapped[MarketVenue] = mapped_column(String(16), nullable=False)
    asset_class: Mapped[AssetClass] = mapped_column(String(16), nullable=False)
    base_currency: Mapped[str] = mapped_column(String(8), nullable=False)
    quote_currency: Mapped[str] = mapped_column(String(8), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    ohlcv_bars: Mapped[list["MarketDataOHLCV"]] = relationship(back_populates="asset")

    def __repr__(self) -> str:
        return f"<Asset {self.symbol}@{self.venue}>"


class MarketDataOHLCV(Base):
    """
    Velas de precio por activo/timeframe. Es la tabla de mayor volumen del
    sistema: se indexa por (asset_id, timeframe, ts) para acceso eficiente
    tanto de features técnicos como de backtesting.
    """
    __tablename__ = "market_data_ohlcv"
    __table_args__ = (
        UniqueConstraint("asset_id", "timeframe", "ts", name="uq_ohlcv_asset_tf_ts"),
        Index("ix_ohlcv_asset_tf_ts", "asset_id", "timeframe", "ts"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), nullable=False)
    timeframe: Mapped[TimeFrame] = mapped_column(String(4), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    open: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)

    asset: Mapped["Asset"] = relationship(back_populates="ohlcv_bars")
