"""
Enumeraciones de dominio.

Se definen una sola vez aquí y se reutilizan tanto en los modelos de base
de datos (capa de persistencia) como en los DTOs de Pydantic (capa de
servicios/IA), evitando duplicación y strings "mágicos" sueltos en el código.
"""
from enum import StrEnum


class MarketVenue(StrEnum):
    """Plataforma de ejecución/origen del activo."""
    BITSO = "bitso"
    GBM = "gbm"


class AssetClass(StrEnum):
    CRYPTO = "crypto"
    EQUITY = "equity"
    ETF = "etf"


class TimeFrame(StrEnum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class SignalAction(StrEnum):
    """Salida del motor de inferencia de IA."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(StrEnum):
    PENDING = "pending"          # creada localmente, aún no enviada
    SENT = "sent"                # enviada al exchange, esperando confirmación
    OPEN = "open"                # activa en el libro del exchange
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


class PositionStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class RiskDecision(StrEnum):
    """Resultado del motor de riesgo determinista sobre una señal de IA."""
    APPROVED = "approved"
    REJECTED = "rejected"
    ADJUSTED = "adjusted"        # aprobada pero con tamaño/niveles modificados


class NotificationChannel(StrEnum):
    TELEGRAM = "telegram"
    DISCORD = "discord"
    LOG = "log"


class NotificationLevel(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
