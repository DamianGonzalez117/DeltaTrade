"""
Contrato que debe cumplir cualquier conector de plataforma de trading.

La capa de negocio (motor de riesgo, motor de ejecución) programa contra
estas interfaces, nunca contra un SDK concreto de Bitso o GBM. Así, añadir
un nuevo broker es implementar esta interfaz, sin tocar el resto del
sistema (objetivo de "Arquitectura Escalable y Decoplada").

Se separan dos interfaces porque el patrón de acceso a datos de mercado es
distinto por naturaleza:
  - Bitso: mercado 24/7, datos en tiempo real vía WebSocket (streaming).
  - GBM: mercado con horario bursátil, se consulta en polling (REST).
Ambas comparten la interfaz de ejecución de órdenes (`ExchangeConnector`).
"""
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal

from trading_system.core.enums import OrderSide, OrderType


@dataclass(frozen=True, slots=True)
class Ticker:
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class Balance:
    currency: str
    total: Decimal
    available: Decimal


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Respuesta normalizada de cualquier exchange al enviar/cancelar una orden."""
    external_order_id: str
    status: str
    filled_quantity: Decimal
    average_price: Decimal | None
    raw_response: dict


class ExchangeConnector(ABC):
    """Operaciones comunes de cuenta y ejecución, sin importar el venue."""

    @abstractmethod
    async def connect(self) -> None:
        """Inicializa la conexión (auth, sesión HTTP, etc.)."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Libera recursos de forma ordenada (cierra sockets/clientes HTTP)."""

    @abstractmethod
    async def get_balances(self) -> list[Balance]:
        """Balances disponibles por moneda/activo en la cuenta."""

    @abstractmethod
    async def place_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None = None,
    ) -> ExecutionResult:
        """Envía una orden. `price` es obligatorio salvo para órdenes de mercado."""

    @abstractmethod
    async def cancel_order(self, external_order_id: str) -> ExecutionResult:
        """Cancela una orden abierta por su id en el exchange."""

    @abstractmethod
    async def get_order_status(self, external_order_id: str) -> ExecutionResult:
        """Consulta el estado actual de una orden ya enviada."""


class StreamingMarketDataConnector(ABC):
    """
    Fuente de datos de mercado en tiempo real vía WebSocket (ej. Bitso).
    El callback se invoca por cada tick/actualización recibida.
    """

    @abstractmethod
    async def subscribe_ticker(
        self, symbol: str, callback: Callable[[Ticker], Awaitable[None]]
    ) -> None:
        """Se suscribe al canal de ticker de `symbol` y despacha cada update al callback."""

    @abstractmethod
    async def unsubscribe(self, symbol: str) -> None:
        """Cancela la suscripción activa para `symbol`."""


class PollingMarketDataConnector(ABC):
    """
    Fuente de datos de mercado por consulta periódica (ej. GBM durante
    horario bursátil). No mantiene conexión persistente.
    """

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Obtiene el precio actual de `symbol` mediante una llamada REST puntual."""

    @abstractmethod
    async def is_market_open(self) -> bool:
        """Indica si la sesión bursátil (BMV/SIC) está actualmente abierta."""
