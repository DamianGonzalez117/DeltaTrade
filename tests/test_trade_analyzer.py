"""TDD — Tests for TradeAnalyzer (RED→GREEN).

Analiza posiciones cerradas para sugerir objetivos de venta.
Se usa la DB de Docker (Postgres real) con limpieza por transacción.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trading_system.config import settings
from trading_system.core.enums import PositionStatus
from trading_system.db.base import Base
from trading_system.db.models import Asset, Position
from trading_system.services.trade_analyzer import TradeAnalyzer, TradeStats


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(settings.database_url, future=True)
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture
def session(engine):
    SessionLocal = sessionmaker(bind=engine, autoflush=False, future=True)
    sess = SessionLocal()
    yield sess
    sess.rollback()
    sess.close()


@pytest.fixture(autouse=True)
def cleanup_tables(engine):
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM positions"))
        conn.execute(text("DELETE FROM assets"))
        conn.commit()
    yield


@pytest.fixture
def analyzer(session):
    return TradeAnalyzer(session)


def _make_asset(session, symbol: str = "btc_mxn") -> Asset:
    asset = Asset(
        symbol=symbol,
        name=symbol.upper(),
        venue="bitso",
        asset_class="crypto",
        base_currency=symbol.split("_")[0],
        quote_currency=symbol.split("_")[1],
    )
    session.add(asset)
    session.flush()
    return asset


def _closed_position(
    session,
    asset_id: int,
    qty: Decimal,
    entry: Decimal,
    exit_: Decimal,
    hold_minutes: float = 120.0,
) -> Position:
    opened = datetime.now(UTC) - timedelta(minutes=hold_minutes)
    pnl = (exit_ - entry) * qty
    pos = Position(
        asset_id=asset_id,
        status=PositionStatus.CLOSED,
        quantity=qty,
        avg_entry_price=entry,
        realized_pnl=pnl,
        opened_at=opened,
        closed_at=opened + timedelta(minutes=hold_minutes),
    )
    session.add(pos)
    session.flush()
    return pos


class TestGetClosedStats:
    def test_returns_none_when_no_history(self, analyzer, session):
        asset = _make_asset(session)
        assert analyzer.get_closed_stats(asset.symbol) is None

    def test_stats_with_mixed_trades(self, analyzer, session):
        asset = _make_asset(session)
        # 2 ganadoras (+10%, +5%) y 1 perdedora (-4%) sobre entrada 1000, qty 1
        _closed_position(session, asset.id, Decimal(1), Decimal(1000), Decimal(1100))
        _closed_position(session, asset.id, Decimal(1), Decimal(1000), Decimal(1050))
        _closed_position(session, asset.id, Decimal(1), Decimal(1000), Decimal(960))

        stats = analyzer.get_closed_stats(asset.symbol)
        assert isinstance(stats, TradeStats)
        assert stats.n_trades == 3
        assert stats.win_rate == pytest.approx(2 / 3)
        assert stats.avg_profit_pct == pytest.approx((10.0 + 5.0 - 4.0) / 3)
        assert stats.avg_hold_minutes == pytest.approx(120.0)

    def test_stats_only_for_requested_book(self, analyzer, session):
        btc = _make_asset(session, "btc_mxn")
        eth = _make_asset(session, "eth_mxn")
        _closed_position(session, btc.id, Decimal(1), Decimal(1000), Decimal(1100))
        _closed_position(session, eth.id, Decimal(1), Decimal(1000), Decimal(900))

        stats = analyzer.get_closed_stats("btc_mxn")
        assert stats is not None
        assert stats.n_trades == 1
        assert stats.win_rate == 1.0

    def test_ignores_open_positions(self, analyzer, session):
        asset = _make_asset(session)
        _closed_position(session, asset.id, Decimal(1), Decimal(1000), Decimal(1100))
        session.add(Position(
            asset_id=asset.id,
            status=PositionStatus.OPEN,
            quantity=Decimal(1),
            avg_entry_price=Decimal(1000),
            realized_pnl=Decimal(0),
            opened_at=datetime.now(UTC),
        ))
        session.flush()

        stats = analyzer.get_closed_stats(asset.symbol)
        assert stats.n_trades == 1


class TestSuggestProfitTarget:
    def test_default_when_no_history(self, analyzer, session):
        asset = _make_asset(session)
        target = analyzer.suggest_profit_target_pct(asset.symbol, default_pct=5.0)
        assert target == pytest.approx(5.0)

    def test_suggestion_from_winning_trades(self, analyzer, session):
        asset = _make_asset(session)
        # ganadoras: +10% y +8% → promedio 9% → sugerencia 8.1% (90%)
        _closed_position(session, asset.id, Decimal(1), Decimal(1000), Decimal(1100))
        _closed_position(session, asset.id, Decimal(1), Decimal(1000), Decimal(1080))

        target = analyzer.suggest_profit_target_pct(asset.symbol, default_pct=5.0)
        assert target == pytest.approx(9.0 * 0.9)

    def test_falls_back_to_default_when_only_losers(self, analyzer, session):
        asset = _make_asset(session)
        _closed_position(session, asset.id, Decimal(1), Decimal(1000), Decimal(950))

        target = analyzer.suggest_profit_target_pct(asset.symbol, default_pct=5.0)
        assert target == pytest.approx(5.0)

    def test_suggestion_at_least_one_percent(self, analyzer, session):
        asset = _make_asset(session)
        # ganadora marginal: +1.5% → 90% = 1.35% → piso 1.0%
        _closed_position(session, asset.id, Decimal(1), Decimal(1000), Decimal(1015))

        target = analyzer.suggest_profit_target_pct(asset.symbol, default_pct=5.0)
        assert target == pytest.approx(1.35)


class TestTransactionRelease:
    """Las lecturas deben cerrar su transacción (no dejar locks en apps 24/7)."""

    def test_get_closed_stats_releases(self, analyzer, session):
        asset = _make_asset(session)
        _closed_position(session, asset.id, Decimal(1), Decimal(1000), Decimal(1100))

        analyzer.get_closed_stats(asset.symbol)
        assert not session.in_transaction()

    def test_get_closed_stats_no_history_releases(self, analyzer, session):
        asset = _make_asset(session)

        analyzer.get_closed_stats(asset.symbol)
        assert not session.in_transaction()

    def test_suggest_target_releases(self, analyzer, session):
        asset = _make_asset(session)

        analyzer.suggest_profit_target_pct(asset.symbol, default_pct=5.0)
        assert not session.in_transaction()
