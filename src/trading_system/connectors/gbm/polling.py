"""
Conector de polling de datos de mercado de GBM.

Implementa ``PollingMarketDataConnector``. Consulta el precio de un
símbolo vía REST y reporta si el mercado está abierto usando
``BMVCalendar``.

DIP: ``httpx.AsyncClient`` y ``BMVCalendar`` se inyectan.
SRP: solo mapea respuestas HTTP → ``Ticker`` y delega el horario al calendario.
"""
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import structlog

from trading_system.connectors.base import PollingMarketDataConnector, Ticker
from trading_system.connectors.gbm.calendar import BMVCalendar
from trading_system.core.retry import retry_async

logger = structlog.get_logger(__name__)

_TICKER_PATH = "/api/v1/market-data/ticker/"


class GBMPollingConnector(PollingMarketDataConnector):
    """Polling de datos de mercado de GBM vía REST."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.gbm.com",
        client: httpx.AsyncClient | None = None,
        calendar: BMVCalendar | None = None,
    ) -> None:
        self._base_url = base_url
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(base_url=base_url)
            self._owns_client = True
        self._calendar = calendar or BMVCalendar()
        self._connected = False

    async def connect(self) -> None:
        self._connected = True
        logger.info("gbm_polling.connected", base_url=self._base_url)

    async def disconnect(self) -> None:
        self._connected = False
        if self._owns_client:
            await self._client.aclose()
        logger.info("gbm_polling.disconnected")

    async def get_ticker(self, symbol: str) -> Ticker:
        return await retry_async(self._fetch_ticker, symbol)

    async def _fetch_ticker(self, symbol: str) -> Ticker:
        response = await self._client.get(f"{_TICKER_PATH}{symbol}")
        response.raise_for_status()
        data = response.json()
        payload = data["payload"]
        return Ticker(
            symbol=payload["symbol"],
            bid=Decimal(str(payload["bid"])),
            ask=Decimal(str(payload["ask"])),
            last=Decimal(str(payload["lastPrice"])),
            timestamp_ms=int(payload["timestamp"]),
        )

    async def is_market_open(self) -> bool:
        return self._calendar.is_market_open(datetime.now(UTC))