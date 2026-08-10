"""
Tipos de datos del motor de riesgo.

``RiskInput`` agrupa toda la información que las reglas necesitan para
evaluar una señal de IA. ``RiskOutput`` es el resultado determinista.

Ambos son frozen dataclasses (inmutables) para garantizar que el motor
no tenga side-effects: una vez evaluado, el resultado no cambia.
"""
from dataclasses import dataclass
from decimal import Decimal

from trading_system.core.enums import MarketVenue, RiskDecision, SignalAction


@dataclass(frozen=True, slots=True)
class RiskInput:
    """Entrada del motor de riesgo: todo lo necesario para decidir."""

    signal_action: SignalAction
    confidence: Decimal
    asset_symbol: str
    asset_venue: MarketVenue
    current_price: Decimal
    atr: Decimal | None
    portfolio_equity: Decimal
    current_exposure: Decimal
    max_capital_pct: Decimal
    max_drawdown_pct: Decimal
    current_drawdown: Decimal
    is_market_open: bool | None = None


@dataclass(frozen=True, slots=True)
class RiskOutput:
    """Salida determinista del motor de riesgo."""

    decision: RiskDecision
    reason: str
    position_size: Decimal | None
    stop_loss_price: Decimal | None
    take_profit_price: Decimal | None
    atr_at_evaluation: Decimal | None
    max_capital_pct: Decimal | None