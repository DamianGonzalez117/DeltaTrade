"""
Análisis de trades cerrados para sugerir objetivos de venta.

Responsabilidades:
  1. ``get_closed_stats``: estadísticas por libro (n, win-rate,
     ganancia % promedio, tiempo de tenencia promedio).
  2. ``suggest_profit_target_pct``: objetivo de venta % basado en el
     promedio de los trades ganadores del usuario (conservado al 90%),
     con fallback al default cuando no hay historial ganador.

DIP: usa ``Session`` de SQLAlchemy inyectable.
SRP: solo analiza historial. No genera señales ni notifica.
"""
from __future__ import annotations

from dataclasses import dataclass

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from trading_system.core.enums import MarketVenue, PositionStatus
from trading_system.db.models import Asset, Position

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TradeStats:
    """Estadísticas de trades cerrados para un libro."""

    n_trades: int
    win_rate: float
    avg_profit_pct: float
    avg_hold_minutes: float


class TradeAnalyzer:
    """Analiza posiciones cerradas y sugiere objetivos de venta."""

    def __init__(
        self,
        session: Session,
        *,
        conservative_factor: float = 0.9,
        min_target_pct: float = 1.0,
    ) -> None:
        self._session = session
        self._factor = conservative_factor
        self._min_target = min_target_pct

    def get_closed_stats(self, book: str) -> TradeStats | None:
        """Estadísticas de trades cerrados para ``book``; None si no hay."""
        positions = self._get_closed_positions(book)
        if not positions:
            self._session.commit()  # libera la transacción de solo lectura
            return None

        profits = [self._profit_pct(p) for p in positions]
        wins = [p for p in profits if p > 0]
        holds = [self._hold_minutes(p) for p in positions]

        stats = TradeStats(
            n_trades=len(positions),
            win_rate=len(wins) / len(positions),
            avg_profit_pct=sum(profits) / len(profits),
            avg_hold_minutes=sum(holds) / len(holds),
        )
        self._session.commit()  # libera la transacción de solo lectura
        logger.info(
            "analyzer.stats",
            book=book,
            n=stats.n_trades,
            win_rate=round(stats.win_rate, 3),
        )
        return stats

    def suggest_profit_target_pct(self, book: str, *, default_pct: float = 5.0) -> float:
        """Objetivo de venta % para la próxima compra de ``book``.

        Basado en el promedio de ganancia de los trades ganadores del
        usuario, conservado al 90%. Si no hay ganadores, usa ``default_pct``.
        """
        positions = self._get_closed_positions(book)
        win_pcts = [self._profit_pct(p) for p in positions if self._profit_pct(p) > 0]
        self._session.commit()  # libera la transacción de solo lectura

        if not win_pcts:
            logger.info("analyzer.default_target", book=book, target=default_pct)
            return default_pct

        target = (sum(win_pcts) / len(win_pcts)) * self._factor
        target = max(target, self._min_target)
        logger.info("analyzer.suggested_target", book=book, target=round(target, 2))
        return target

    def _get_closed_positions(self, book: str) -> list[Position]:
        asset = self._session.execute(
            select(Asset).where(
                Asset.symbol == book,
                Asset.venue == MarketVenue.BITSO,
            )
        ).scalar_one_or_none()
        if asset is None:
            return []

        return list(
            self._session.execute(
                select(Position).where(
                    Position.asset_id == asset.id,
                    Position.status == PositionStatus.CLOSED,
                ).order_by(Position.closed_at.desc())
            ).scalars().all()
        )

    @staticmethod
    def _profit_pct(pos: Position) -> float:
        if pos.avg_entry_price == 0:
            return 0.0
        return float(
            (pos.realized_pnl / (pos.avg_entry_price * pos.quantity)) * 100
        )

    @staticmethod
    def _hold_minutes(pos: Position) -> float:
        if pos.closed_at is None or pos.opened_at is None:
            return 0.0
        delta = pos.closed_at - pos.opened_at
        return delta.total_seconds() / 60.0
