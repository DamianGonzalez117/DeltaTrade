"""
Servicio de persistencia: guarda el pipeline completo en la DB.

Cadena obligatoria:
    AISignal → RiskEvaluation → Order → Fill

DIP: usa ``Session`` de SQLAlchemy inyectable para tests.
SRP: solo persiste. No decide, no ejecuta, no infiere.
OCP: añadir nuevas entidades = añadir métodos, no modificar existentes.
"""
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from trading_system.connectors.base import ExecutionResult
from trading_system.core.enums import (
    MarketVenue,
    OrderSide,
    OrderStatus,
    OrderType,
)
from trading_system.db.models import (
    AISignal,
    Asset,
    Fill,
    Order,
    RiskEvaluation,
)
from trading_system.inference.types import InferenceOutput
from trading_system.risk.types import RiskOutput

logger = structlog.get_logger(__name__)


class PersistenceService:
    """Persiste señales, evaluaciones de riesgo, órdenes y fills en la DB."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_or_create_asset(
        self,
        symbol: str,
        venue: MarketVenue,
        name: str = "",
        asset_class: str = "crypto",
        base_currency: str = "",
        quote_currency: str = "",
    ) -> Asset:
        """Busca o crea un Asset en la sesión actual."""
        existing = self._session.execute(
            select(Asset).where(
                Asset.symbol == symbol,
                Asset.venue == venue,
            )
        ).scalar_one_or_none()

        if existing is not None:
            return existing

        if not base_currency or not quote_currency:
            parts = symbol.split("_")
            base_currency = parts[0] if len(parts) == 2 else symbol
            quote_currency = parts[1] if len(parts) == 2 else "mxn"

        asset = Asset(
            symbol=symbol,
            name=name or f"{base_currency.upper()} / {quote_currency.upper()}",
            venue=venue,
            asset_class=asset_class,  # type: ignore[arg-type]
            base_currency=base_currency.lower(),
            quote_currency=quote_currency.lower(),
        )
        self._session.add(asset)
        self._session.flush()
        return asset

    def save_signal(
        self,
        asset_id: int,
        inference_output: InferenceOutput,
        technical_features: dict[str, float] | None = None,
        sentiment_score_id: int | None = None,
    ) -> AISignal:
        """Persiste una señal de IA generada por el motor de inferencia."""
        signal = AISignal(
            asset_id=asset_id,
            generated_at=datetime.now(UTC),
            model_name=inference_output.model_name,
            model_version=inference_output.model_version,
            action=inference_output.action,
            confidence=inference_output.confidence,
            technical_features=technical_features,
            sentiment_score_id=sentiment_score_id,
        )
        self._session.add(signal)
        self._session.flush()
        logger.info(
            "persist.signal_saved",
            signal_id=signal.id,
            asset_id=asset_id,
            action=inference_output.action.value,
        )
        return signal

    def save_risk_evaluation(
        self,
        signal_id: int,
        risk_output: RiskOutput,
    ) -> RiskEvaluation:
        """Persiste la evaluación de riesgo para una señal dada."""
        risk_eval = RiskEvaluation(
            signal_id=signal_id,
            evaluated_at=datetime.now(UTC),
            decision=risk_output.decision,
            reason=risk_output.reason,
            position_size=risk_output.position_size,
            max_capital_pct=risk_output.max_capital_pct,
            stop_loss_price=risk_output.stop_loss_price,
            take_profit_price=risk_output.take_profit_price,
            atr_at_evaluation=risk_output.atr_at_evaluation,
        )
        self._session.add(risk_eval)
        self._session.flush()
        logger.info(
            "persist.risk_saved",
            risk_eval_id=risk_eval.id,
            signal_id=signal_id,
            decision=risk_output.decision.value,
        )
        return risk_eval

    def save_order(
        self,
        asset_id: int,
        risk_evaluation_id: int,
        venue: MarketVenue,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None = None,
        external_order_id: str | None = None,
        status: OrderStatus = OrderStatus.PENDING,
    ) -> Order:
        """Persiste una orden enviada al exchange."""
        order = Order(
            asset_id=asset_id,
            risk_evaluation_id=risk_evaluation_id,
            venue=venue,
            side=side,
            order_type=order_type,
            status=status,
            quantity=quantity,
            price=price,
            external_order_id=external_order_id,
            submitted_at=datetime.now(UTC) if external_order_id else None,
        )
        self._session.add(order)
        self._session.flush()
        logger.info(
            "persist.order_saved",
            order_id=order.id,
            asset_id=asset_id,
            side=side.value,
            status=status.value,
        )
        return order

    def update_order_from_result(
        self,
        order: Order,
        result: ExecutionResult,
    ) -> Order:
        """Actualiza una orden existente con el resultado del exchange."""
        order.external_order_id = result.external_order_id
        order.status = _map_status(result.status)
        if order.submitted_at is None:
            order.submitted_at = datetime.now(UTC)
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            order.closed_at = datetime.now(UTC)
        self._session.flush()

        if result.filled_quantity > 0 and result.average_price is not None:
            self.save_fill(
                order_id=order.id,
                filled_quantity=result.filled_quantity,
                filled_price=result.average_price,
                fee=Decimal(0),
                fee_currency="mxn",
            )

        logger.info(
            "persist.order_updated",
            order_id=order.id,
            external_id=result.external_order_id,
            status=order.status.value if order.status else "unknown",
        )
        return order

    def save_fill(
        self,
        order_id: int,
        filled_quantity: Decimal,
        filled_price: Decimal,
        fee: Decimal,
        fee_currency: str,
        executed_at: datetime | None = None,
    ) -> Fill:
        """Persiste un fill (ejecución parcial o total de una orden)."""
        fill = Fill(
            order_id=order_id,
            filled_quantity=filled_quantity,
            filled_price=filled_price,
            fee=fee,
            fee_currency=fee_currency,
            executed_at=executed_at or datetime.now(UTC),
        )
        self._session.add(fill)
        self._session.flush()
        logger.info(
            "persist.fill_saved",
            fill_id=fill.id,
            order_id=order_id,
            qty=str(filled_quantity),
            price=str(filled_price),
        )
        return fill

    def commit(self) -> None:
        """Hace commit de la transacción actual."""
        self._session.commit()

    def rollback(self) -> None:
        """Hace rollback de la transacción actual."""
        self._session.rollback()


def _map_status(raw: str) -> OrderStatus:
    """Mapea el status string del exchange a nuestro enum."""
    raw_lower = raw.lower()
    mapping: dict[str, OrderStatus] = {
        "filled": OrderStatus.FILLED,
        "open": OrderStatus.OPEN,
        "partially_filled": OrderStatus.PARTIALLY_FILLED,
        "cancelled": OrderStatus.CANCELLED,
        "canceled": OrderStatus.CANCELLED,
        "rejected": OrderStatus.REJECTED,
        "pending": OrderStatus.PENDING,
        "sent": OrderStatus.SENT,
        "error": OrderStatus.ERROR,
    }
    return mapping.get(raw_lower, OrderStatus.OPEN)
