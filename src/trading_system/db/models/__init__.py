"""
Importa todos los modelos para que queden registrados en Base.metadata.

Cualquier módulo que necesite `Base.metadata` completo (Alembic, tests que
crean el esquema en SQLite/Postgres efímero, etc.) debe importar este
paquete, no un submódulo individual.
"""
from trading_system.db.models.market import Asset, MarketDataOHLCV
from trading_system.db.models.notifications import NotificationLog
from trading_system.db.models.risk import RiskEvaluation
from trading_system.db.models.signals import AISignal, SentimentScore
from trading_system.db.models.trading import Fill, Order, PortfolioSnapshot, Position

__all__ = [
    "AISignal",
    "Asset",
    "Fill",
    "MarketDataOHLCV",
    "NotificationLog",
    "Order",
    "PortfolioSnapshot",
    "Position",
    "RiskEvaluation",
    "SentimentScore",
]
