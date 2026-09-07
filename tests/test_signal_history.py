"""TDD — Tests for SignalHistoryService (RED→GREEN).

Se usa la DB de Docker (Postgres real) con limpieza por transacción.
"""
from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trading_system.config import settings
from trading_system.core.enums import PositionStatus, SignalAction
from trading_system.db.base import Base
from trading_system.db.models import AISignal
from trading_system.services.signal_generator import RuleSignal
from trading_system.services.signal_history import SignalHistoryService


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
    """Limpia las tablas antes de cada test para evitar conflictos de PK."""
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM positions"))
        conn.execute(text("DELETE FROM ai_signals"))
        conn.execute(text("DELETE FROM assets"))
        conn.commit()
    yield


@pytest.fixture
def service(session):
    return SignalHistoryService(session)


def _buy_signal() -> RuleSignal:
    return RuleSignal(
        action=SignalAction.BUY,
        confidence=Decimal("0.85"),
        rsi=22.0,
        macd_diff=150.0,
        reason="RSI sobreventa + MACD cruce alcista",
    )


def _sell_signal() -> RuleSignal:
    return RuleSignal(
        action=SignalAction.SELL,
        confidence=Decimal("0.80"),
        rsi=78.0,
        macd_diff=-200.0,
        reason="RSI sobrecompra + MACD cruce bajista",
    )


def _hold_signal() -> RuleSignal:
    return RuleSignal(
        action=SignalAction.HOLD,
        confidence=Decimal("0.50"),
        rsi=50.0,
        macd_diff=10.0,
        reason="RSI neutral",
    )


class TestSaveSignal:
    def test_persists_buy_signal(self, service, session):
        signal = _buy_signal()
        record = service.save_signal("btc_mxn", signal)
        assert record.id is not None
        assert record.action == SignalAction.BUY
        assert record.confidence == Decimal("0.85")
        assert record.model_name == "rule_rsi_macd"
        assert record.technical_features["rsi_14"] == 22.0

    def test_persists_sell_signal(self, service, session):
        record = service.save_signal("eth_mxn", _sell_signal())
        assert record.action == SignalAction.SELL

    def test_persists_hold_signal(self, service, session):
        record = service.save_signal("sol_mxn", _hold_signal())
        assert record.action == SignalAction.HOLD


class TestGetSignalHistory:
    def test_returns_recent_signals(self, service):
        service.save_signal("btc_mxn", _buy_signal())
        service.save_signal("btc_mxn", _hold_signal())
        history = service.get_signal_history("btc_mxn", limit=10)
        assert len(history) >= 2
        assert all(isinstance(s, AISignal) for s in history)

    def test_filters_by_book(self, service):
        service.save_signal("btc_mxn", _buy_signal())
        service.save_signal("eth_mxn", _sell_signal())
        btc_history = service.get_signal_history("btc_mxn", limit=10)
        for s in btc_history:
            assert s.asset.symbol == "btc_mxn"

    def test_returns_empty_for_unknown_book(self, service):
        history = service.get_signal_history("nonexistent_mxn", limit=10)
        assert history == []

    def test_returns_all_books_when_no_filter(self, service):
        service.save_signal("btc_mxn", _buy_signal())
        service.save_signal("eth_mxn", _sell_signal())
        history = service.get_signal_history(limit=50)
        assert len(history) >= 2


class TestConfirmBuy:
    def test_creates_open_position(self, service):
        pos = service.confirm_buy("btc_mxn", Decimal("0.01"), Decimal(500000))
        assert pos.id is not None
        assert pos.status == PositionStatus.OPEN
        assert pos.quantity == Decimal("0.01")
        assert pos.avg_entry_price == Decimal(500000)
        assert pos.opened_at is not None
        assert pos.closed_at is None

    def test_raises_if_position_already_open(self, service):
        service.confirm_buy("btc_mxn", Decimal("0.01"), Decimal(500000))
        with pytest.raises(ValueError, match="Ya hay una posición"):
            service.confirm_buy("btc_mxn", Decimal("0.02"), Decimal(510000))


class TestClosePosition:
    def test_closes_position_with_pnl(self, service):
        service.confirm_buy("btc_mxn", Decimal("0.01"), Decimal(500000))
        pos = service.close_position("btc_mxn", Decimal(510000))
        assert pos.status == PositionStatus.CLOSED
        assert pos.closed_at is not None
        assert pos.realized_pnl == Decimal(100)  # (510000 - 500000) * 0.01

    def test_raises_if_no_open_position(self, service):
        with pytest.raises(ValueError, match="No hay posición"):
            service.close_position("xrp_mxn", Decimal(10))

    def test_negative_pnl_on_loss(self, service):
        service.confirm_buy("eth_mxn", Decimal("0.5"), Decimal(30000))
        pos = service.close_position("eth_mxn", Decimal(28000))
        assert pos.realized_pnl < 0
        assert pos.realized_pnl == Decimal("-1000.00")  # (28000 - 30000) * 0.5


class TestGetOpenPosition:
    def test_returns_none_when_no_position(self, service):
        assert service.get_open_position("sol_mxn") is None

    def test_returns_position_after_confirm(self, service):
        service.confirm_buy("btc_mxn", Decimal("0.01"), Decimal(500000))
        pos = service.get_open_position("btc_mxn")
        assert pos is not None
        assert pos.status == PositionStatus.OPEN

    def test_returns_none_after_close(self, service):
        service.confirm_buy("btc_mxn", Decimal("0.01"), Decimal(500000))
        service.close_position("btc_mxn", Decimal(510000))
        assert service.get_open_position("btc_mxn") is None


class TestGetAllOpenPositions:
    def test_returns_all_open(self, service):
        service.confirm_buy("btc_mxn", Decimal("0.01"), Decimal(500000))
        service.confirm_buy("eth_mxn", Decimal("0.5"), Decimal(30000))
        positions = service.get_all_open_positions()
        assert len(positions) >= 2
        assert all(p.status == PositionStatus.OPEN for p in positions)


class TestCalcUnrealizedPnl:
    def test_positive_pnl(self, service):
        service.confirm_buy("btc_mxn", Decimal("0.01"), Decimal(500000))
        pos = service.get_open_position("btc_mxn")
        pnl = service.calc_unrealized_pnl(pos, Decimal(520000))
        assert pnl == Decimal(200)  # (520000 - 500000) * 0.01

    def test_negative_pnl(self, service):
        service.confirm_buy("btc_mxn", Decimal("0.01"), Decimal(500000))
        pos = service.get_open_position("btc_mxn")
        pnl = service.calc_unrealized_pnl(pos, Decimal(490000))
        assert pnl == Decimal(-100)  # (490000 - 500000) * 0.01


class TestTransactionRelease:
    """Las lecturas deben cerrar su transacción (no dejar locks en apps 24/7)."""

    def test_get_open_position_releases_transaction(self, service, session):
        service.confirm_buy("btc_mxn", Decimal("0.01"), Decimal(500000))
        # SELECT abre transacción; el servicio debe cerrarla con commit
        service.get_open_position("btc_mxn")
        assert not session.in_transaction()

    def test_get_open_position_unknown_book_releases(self, service, session):
        service.get_open_position("no_existe_mxn")
        assert not session.in_transaction()

    def test_get_all_open_positions_releases(self, service, session):
        service.get_all_open_positions()
        assert not session.in_transaction()

    def test_get_signal_history_releases(self, service, session):
        service.save_signal("btc_mxn", _buy_signal())
        service.get_signal_history(book="btc_mxn")
        assert not session.in_transaction()
