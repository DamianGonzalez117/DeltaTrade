"""
Test de humo del esquema: levanta todas las tablas contra la Postgres de
docker-compose e inserta un registro mínimo por tabla clave, validando las
relaciones (Asset -> AISignal -> RiskEvaluation -> Order).

Requiere:
    docker compose up -d postgres
    pip install -e ".[dev]"
    pytest
"""
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trading_system.config import settings
from trading_system.core.enums import (
    AssetClass, MarketVenue, SignalAction, RiskDecision,
    OrderSide, OrderType,
)
from trading_system.db.base import Base
from trading_system.db.models import Asset, AISignal, RiskEvaluation, Order


@pytest.fixture(scope="module")
def db_session():
    engine = create_engine(settings.database_url, future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, future=True)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_full_pipeline_smoke(db_session):
    now = datetime.now(timezone.utc)

    asset = Asset(
        symbol="btc_mxn",
        name="Bitcoin / Peso Mexicano",
        venue=MarketVenue.BITSO,
        asset_class=AssetClass.CRYPTO,
        base_currency="BTC",
        quote_currency="MXN",
    )
    db_session.add(asset)
    db_session.flush()

    signal = AISignal(
        asset_id=asset.id,
        generated_at=now,
        model_name="xgboost_v1",
        model_version="0.1.0",
        action=SignalAction.BUY,
        confidence=Decimal("0.8750"),
    )
    db_session.add(signal)
    db_session.flush()

    risk_eval = RiskEvaluation(
        signal_id=signal.id,
        evaluated_at=now,
        decision=RiskDecision.APPROVED,
        reason="Dentro de límites de exposición y volatilidad (ATR).",
        position_size=Decimal("0.01"),
        max_capital_pct=Decimal("0.05"),
        stop_loss_price=Decimal("950000.00"),
        take_profit_price=Decimal("1050000.00"),
    )
    db_session.add(risk_eval)
    db_session.flush()

    order = Order(
        asset_id=asset.id,
        risk_evaluation_id=risk_eval.id,
        venue=MarketVenue.BITSO,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.01"),
    )
    db_session.add(order)
    db_session.flush()

    assert order.risk_evaluation.signal.asset.symbol == "btc_mxn"
