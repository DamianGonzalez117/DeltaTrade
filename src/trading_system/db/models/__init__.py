"""
Importa todos los modelos para que queden registrados en Base.metadata.

Cualquier módulo que necesite `Base.metadata` completo (Alembic, tests que
crean el esquema en SQLite/Postgres efímero, etc.) debe importar este
paquete, no un submódulo individual.
"""
from trading_system.db.models.market import Asset, MarketDataOHLCV
from trading_system.db.models.signals import SentimentScore, AISignal
from trading_system.db.models.risk import RiskEvaluation
from trading_system.db.models.trading import Order, Fill, Position, PortfolioSnapshot
from trading_system.db.models.notifications import NotificationLog

__all__ = [
    "Asset",
    "MarketDataOHLCV",
    "SentimentScore",
    "AISignal",
    "RiskEvaluation",
    "Order",
    "Fill",
    "Position",
    "PortfolioSnapshot",
    "NotificationLog",
]
