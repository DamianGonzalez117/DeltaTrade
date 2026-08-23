"""TDD — Tests for PersistenceService (RED→GREEN).

Se usa la DB de Docker (Postgres real) con limpieza por transacción.
Cada test hace rollback al final para no dejar basura.
"""
from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trading_system.config import settings
from trading_system.connectors.base import ExecutionResult
from trading_system.core.enums import (
    MarketVenue,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskDecision,
    SignalAction,
)
from trading_system.db.base import Base
from trading_system.inference.types import InferenceOutput
from trading_system.risk.types import RiskOutput
from trading_system.services.persistence import PersistenceService, _map_status


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


@pytest.fixture
def service(session):
    return PersistenceService(session)


@pytest.fixture
def asset(service):
    return service.get_or_create_asset(
        symbol="btc_mxn",
        venue=MarketVenue.BITSO,
        asset_class="crypto",
    )


class TestGetOrCreateAsset:
    def test_creates_new_asset(self, service, session):
        asset = service.get_or_create_asset(
            symbol="eth_mxn",
            venue=MarketVenue.BITSO,
            asset_class="crypto",
        )
        assert asset.id is not None
        assert asset.symbol == "eth_mxn"
        assert asset.base_currency == "eth"
        assert asset.quote_currency == "mxn"

    def test_returns_existing_asset(self, service, asset):
        again = service.get_or_create_asset(
            symbol="btc_mxn",
            venue=MarketVenue.BITSO,
        )
        assert again.id == asset.id


class TestSaveSignal:
    def test_persists_signal(self, service, asset):
        inference = InferenceOutput(
            action=SignalAction.BUY,
            confidence=Decimal("0.8750"),
            model_name="xgboost_v1",
            model_version="0.1.0",
            technical_score=Decimal("0.90"),
            sentiment_score=Decimal("0.70"),
            technical_weight=Decimal("0.60"),
            sentiment_weight=Decimal("0.40"),
        )
        signal = service.save_signal(
            asset_id=asset.id,
            inference_output=inference,
            technical_features={"rsi_14": 55.3},
        )
        assert signal.id is not None
        assert signal.action == SignalAction.BUY
        assert signal.confidence == Decimal("0.8750")
        assert signal.technical_features == {"rsi_14": 55.3}


class TestSaveRiskEvaluation:
    def test_persists_risk_eval(self, service, asset):
        inference = InferenceOutput(
            action=SignalAction.BUY,
            confidence=Decimal("0.80"),
            model_name="xgboost_v1",
            model_version="0.1.0",
            technical_score=Decimal("0.85"),
            sentiment_score=Decimal("0.60"),
            technical_weight=Decimal("0.60"),
            sentiment_weight=Decimal("0.40"),
        )
        signal = service.save_signal(asset_id=asset.id, inference_output=inference)

        risk_output = RiskOutput(
            decision=RiskDecision.APPROVED,
            reason="Dentro de límites",
            position_size=Decimal("0.01"),
            stop_loss_price=Decimal(950000),
            take_profit_price=Decimal(1050000),
            atr_at_evaluation=Decimal(12000),
            max_capital_pct=Decimal("0.05"),
        )
        risk_eval = service.save_risk_evaluation(
            signal_id=signal.id,
            risk_output=risk_output,
        )
        assert risk_eval.id is not None
        assert risk_eval.decision == RiskDecision.APPROVED
        assert risk_eval.position_size == Decimal("0.01")


class TestSaveOrder:
    def test_persists_order(self, service, asset):
        inference = InferenceOutput(
            action=SignalAction.BUY,
            confidence=Decimal("0.80"),
            model_name="xgboost_v1",
            model_version="0.1.0",
            technical_score=Decimal("0.85"),
            sentiment_score=Decimal("0.60"),
            technical_weight=Decimal("0.60"),
            sentiment_weight=Decimal("0.40"),
        )
        signal = service.save_signal(asset_id=asset.id, inference_output=inference)
        risk_output = RiskOutput(
            decision=RiskDecision.APPROVED,
            reason="ok",
            position_size=Decimal("0.01"),
            stop_loss_price=Decimal(950000),
            take_profit_price=Decimal(1050000),
            atr_at_evaluation=Decimal(12000),
            max_capital_pct=Decimal("0.05"),
        )
        risk_eval = service.save_risk_evaluation(
            signal_id=signal.id,
            risk_output=risk_output,
        )

        order = service.save_order(
            asset_id=asset.id,
            risk_evaluation_id=risk_eval.id,
            venue=MarketVenue.BITSO,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            external_order_id="ext-123",
            status=OrderStatus.SENT,
        )
        assert order.id is not None
        assert order.risk_evaluation_id == risk_eval.id
        assert order.external_order_id == "ext-123"


class TestUpdateOrderFromResult:
    def test_updates_order_and_creates_fill(self, service, asset):
        inference = InferenceOutput(
            action=SignalAction.BUY,
            confidence=Decimal("0.80"),
            model_name="xgboost_v1",
            model_version="0.1.0",
            technical_score=Decimal("0.85"),
            sentiment_score=Decimal("0.60"),
            technical_weight=Decimal("0.60"),
            sentiment_weight=Decimal("0.40"),
        )
        signal = service.save_signal(asset_id=asset.id, inference_output=inference)
        risk_output = RiskOutput(
            decision=RiskDecision.APPROVED,
            reason="ok",
            position_size=Decimal("0.01"),
            stop_loss_price=Decimal(950000),
            take_profit_price=Decimal(1050000),
            atr_at_evaluation=Decimal(12000),
            max_capital_pct=Decimal("0.05"),
        )
        risk_eval = service.save_risk_evaluation(
            signal_id=signal.id, risk_output=risk_output
        )
        order = service.save_order(
            asset_id=asset.id,
            risk_evaluation_id=risk_eval.id,
            venue=MarketVenue.BITSO,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
        )

        result = ExecutionResult(
            external_order_id="ext-456",
            status="filled",
            filled_quantity=Decimal("0.01"),
            average_price=Decimal(501000),
            raw_response={"id": "ext-456"},
        )
        updated = service.update_order_from_result(order, result)

        assert updated.external_order_id == "ext-456"
        assert updated.status == OrderStatus.FILLED
        assert updated.closed_at is not None

        fills = list(updated.fills)
        assert len(fills) == 1
        assert fills[0].filled_quantity == Decimal("0.01")
        assert fills[0].filled_price == Decimal(501000)


class TestMapStatus:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("filled", OrderStatus.FILLED),
            ("OPEN", OrderStatus.OPEN),
            ("partially_filled", OrderStatus.PARTIALLY_FILLED),
            ("cancelled", OrderStatus.CANCELLED),
            ("canceled", OrderStatus.CANCELLED),
            ("rejected", OrderStatus.REJECTED),
            ("unknown_status", OrderStatus.OPEN),
        ],
    )
    def test_maps_known_statuses(self, raw, expected):
        assert _map_status(raw) == expected


class TestFullChain:
    def test_signal_risk_order_fill_chain(self, service, asset):
        """Verifica que la cadena completa se persiste con FKs correctas."""
        inference = InferenceOutput(
            action=SignalAction.BUY,
            confidence=Decimal("0.90"),
            model_name="xgboost_v1",
            model_version="0.1.0",
            technical_score=Decimal("0.92"),
            sentiment_score=Decimal("0.80"),
            technical_weight=Decimal("0.60"),
            sentiment_weight=Decimal("0.40"),
        )
        signal = service.save_signal(
            asset_id=asset.id,
            inference_output=inference,
            technical_features={"rsi_14": 45.0, "macd": 120.5},
        )

        risk_output = RiskOutput(
            decision=RiskDecision.APPROVED,
            reason="Dentro de límites de exposición",
            position_size=Decimal("0.02"),
            stop_loss_price=Decimal(945000),
            take_profit_price=Decimal(1055000),
            atr_at_evaluation=Decimal(12500),
            max_capital_pct=Decimal("0.05"),
        )
        risk_eval = service.save_risk_evaluation(
            signal_id=signal.id,
            risk_output=risk_output,
        )

        order = service.save_order(
            asset_id=asset.id,
            risk_evaluation_id=risk_eval.id,
            venue=MarketVenue.BITSO,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.02"),
            external_order_id="ext-chain-1",
            status=OrderStatus.SENT,
        )

        result = ExecutionResult(
            external_order_id="ext-chain-1",
            status="filled",
            filled_quantity=Decimal("0.02"),
            average_price=Decimal(500500),
            raw_response={},
        )
        service.update_order_from_result(order, result)

        # Verify FK chain
        assert order.risk_evaluation_id == risk_eval.id
        assert risk_eval.signal_id == signal.id
        assert signal.asset_id == asset.id
        assert len(order.fills) == 1
