"""
Servicio de historial de señales y gestión de posiciones.

Responsabilidades:
  1. Persistir cada señal generada por el monitor en ``ai_signals``.
  2. Gestionar posiciones abiertas por el usuario en ``positions``:
     - ``confirm_buy``: abre una posición cuando el usuario confirma compra.
     - ``close_position``: cierra la posición y calcula P&L.
     - ``get_open_position``: verifica si ya hay posición abierta para un libro.
  3. Recuperar historial de señales para consulta.

DIP: usa ``Session`` de SQLAlchemy inyectable.
SRP: solo persiste y consulta. No genera señales ni notifica.
"""
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from trading_system.core.enums import (
    MarketVenue,
    PositionStatus,
)
from trading_system.db.models import AISignal, Asset, Position
from trading_system.services.signal_generator import RuleSignal

logger = structlog.get_logger(__name__)


class SignalHistoryService:
    """Persiste señales y gestiona posiciones del usuario."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_or_create_asset(self, book: str) -> Asset:
        """Busca o crea el Asset para el libro de Bitso."""
        existing = self._session.execute(
            select(Asset).where(
                Asset.symbol == book,
                Asset.venue == MarketVenue.BITSO,
            )
        ).scalar_one_or_none()

        if existing is not None:
            return existing

        parts = book.split("_")
        base = parts[0] if len(parts) == 2 else book
        quote = parts[1] if len(parts) == 2 else "mxn"

        asset = Asset(
            symbol=book,
            name=f"{base.upper()} / {quote.upper()}",
            venue=MarketVenue.BITSO,
            asset_class="crypto",  # type: ignore[arg-type]
            base_currency=base.lower(),
            quote_currency=quote.lower(),
        )
        self._session.add(asset)
        self._session.flush()
        return asset

    def save_signal(self, book: str, signal: RuleSignal) -> AISignal:
        """Persiste una señal del monitor en ``ai_signals``."""
        asset = self.get_or_create_asset(book)
        record = AISignal(
            asset_id=asset.id,
            generated_at=datetime.now(UTC),
            model_name="rule_rsi_macd",
            model_version="0.1.0",
            action=signal.action,
            confidence=signal.confidence,
            technical_features={
                "rsi_14": signal.rsi,
                "macd_diff": signal.macd_diff,
                "reason": signal.reason,
            },
        )
        self._session.add(record)
        self._session.flush()
        self._session.commit()
        logger.info(
            "history.signal_saved",
            signal_id=record.id,
            book=book,
            action=signal.action.value,
        )
        return record

    def get_signal_history(
        self,
        book: str | None = None,
        limit: int = 50,
    ) -> list[AISignal]:
        """Recupera el historial de señales, opcionalmente filtrado por libro."""
        stmt = select(AISignal).order_by(AISignal.generated_at.desc()).limit(limit)

        if book is not None:
            asset = self._session.execute(
                select(Asset).where(
                    Asset.symbol == book,
                    Asset.venue == MarketVenue.BITSO,
                )
            ).scalar_one_or_none()
            if asset is None:
                return []
            stmt = (
                select(AISignal)
                .where(AISignal.asset_id == asset.id)
                .order_by(AISignal.generated_at.desc())
                .limit(limit)
            )

        return list(self._session.execute(stmt).scalars().all())

    def confirm_buy(
        self,
        book: str,
        quantity: Decimal,
        entry_price: Decimal,
    ) -> Position:
        """Registra una compra confirmada por el usuario. Abre posición."""
        existing = self.get_open_position(book)
        if existing is not None:
            raise ValueError(
                f"Ya hay una posición abierta para {book} "
                f"(id={existing.id}, qty={existing.quantity}, "
                f"entry={existing.avg_entry_price}). Ciérrala primero."
            )

        asset = self.get_or_create_asset(book)
        position = Position(
            asset_id=asset.id,
            status=PositionStatus.OPEN,
            quantity=quantity,
            avg_entry_price=entry_price,
            realized_pnl=Decimal(0),
            opened_at=datetime.now(UTC),
        )
        self._session.add(position)
        self._session.flush()
        self._session.commit()
        logger.info(
            "history.buy_confirmed",
            position_id=position.id,
            book=book,
            qty=str(quantity),
            entry_price=str(entry_price),
        )
        return position

    def close_position(
        self,
        book: str,
        exit_price: Decimal,
    ) -> Position:
        """Cierra la posición abierta para ``book`` y calcula P&L."""
        position = self.get_open_position(book)
        if position is None:
            raise ValueError(f"No hay posición abierta para {book}")

        pnl = (exit_price - position.avg_entry_price) * position.quantity
        position.realized_pnl = pnl
        position.status = PositionStatus.CLOSED
        position.closed_at = datetime.now(UTC)
        self._session.flush()
        self._session.commit()

        logger.info(
            "history.position_closed",
            position_id=position.id,
            book=book,
            pnl=str(pnl),
            exit_price=str(exit_price),
        )
        return position

    def get_open_position(self, book: str) -> Position | None:
        """Retorna la posición abierta para ``book``, o None si no hay."""
        asset = self._session.execute(
            select(Asset).where(
                Asset.symbol == book,
                Asset.venue == MarketVenue.BITSO,
            )
        ).scalar_one_or_none()

        if asset is None:
            return None

        return self._session.execute(
            select(Position).where(
                Position.asset_id == asset.id,
                Position.status == PositionStatus.OPEN,
            )
        ).scalar_one_or_none()

    def get_all_open_positions(self) -> list[Position]:
        """Retorna todas las posiciones abiertas."""
        return list(
            self._session.execute(
                select(Position).where(Position.status == PositionStatus.OPEN)
            ).scalars().all()
        )

    def calc_unrealized_pnl(
        self,
        position: Position,
        current_price: Decimal,
    ) -> Decimal:
        """Calcula el P&L no realizado de una posición abierta."""
        return (current_price - position.avg_entry_price) * position.quantity
