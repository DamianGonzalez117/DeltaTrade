"""
Conector REST de Bitso (API v3).

Implementa ``ExchangeConnector`` usando ``httpx.AsyncClient``. El cliente
HTTP se inyecta por constructor (DIP) para que los tests pasen un
``MockTransport`` sin tocar la red.

Endpoints usados:
    GET    /v3/balance/
    POST   /v3/orders/
    DELETE /v3/orders/{oid}
    GET    /v3/orders/{oid}

La autenticación se delega a ``BitsoAuth`` (SRP): este conector solo se
preocupa de construir el request y mapear la respuesta a DTOs.
"""
import json
from decimal import Decimal

import httpx
import structlog

from trading_system.connectors.base import Balance, ExecutionResult
from trading_system.connectors.bitso.auth import BitsoAuth
from trading_system.core.enums import OrderSide, OrderType

logger = structlog.get_logger(__name__)

_BASE_PATH = "/v3"
_BALANCE_PATH = f"{_BASE_PATH}/balance/"
_ORDERS_PATH = f"{_BASE_PATH}/orders/"


class BitsoRESTConnector:
    """Implementación de ``ExchangeConnector`` para Bitso vía REST v3."""

    def __init__(
        self,
        *,
        auth: BitsoAuth,
        base_url: str = "https://api.bitso.com",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._auth = auth
        self._base_url = base_url
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(base_url=base_url)
            self._owns_client = True
        self._connected = False

    async def connect(self) -> None:
        self._connected = True
        logger.info("bitso_rest.connected", base_url=self._base_url)

    async def disconnect(self) -> None:
        self._connected = False
        if self._owns_client:
            await self._client.aclose()
        logger.info("bitso_rest.disconnected")

    async def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
    ) -> dict:
        body_str = json.dumps(body) if body else ""
        headers = self._auth.build_headers(method, path, body_str)
        response = await self._client.request(
            method=method,
            url=path,
            content=body_str if body_str else b"",
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def get_balances(self) -> list[Balance]:
        data = await self._request("GET", _BALANCE_PATH)
        balances = data["payload"]["balances"]
        return [
            Balance(
                currency=b["currency"],
                total=Decimal(str(b["total"])),
                available=Decimal(str(b["available"])),
            )
            for b in balances
        ]

    async def place_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None = None,
    ) -> ExecutionResult:
        order_body: dict[str, str] = {
            "book": symbol,
            "side": str(side.value),
            "type": str(order_type.value),
            "major": str(quantity),
        }
        if order_type == OrderType.LIMIT:
            if price is None:
                raise ValueError("price es obligatorio para órdenes limit")
            order_body["price"] = str(price)
        elif price is not None:
            order_body["price"] = str(price)

        data = await self._request("POST", _ORDERS_PATH, body=order_body)
        payload = data["payload"]
        return ExecutionResult(
            external_order_id=payload["oid"],
            status=payload["status"],
            filled_quantity=Decimal(0),
            average_price=None,
            raw_response=data,
        )

    async def cancel_order(self, external_order_id: str) -> ExecutionResult:
        path = f"{_ORDERS_PATH}{external_order_id}"
        data = await self._request("DELETE", path)
        return ExecutionResult(
            external_order_id=external_order_id,
            status="cancelled",
            filled_quantity=Decimal(0),
            average_price=None,
            raw_response=data,
        )

    async def get_order_status(self, external_order_id: str) -> ExecutionResult:
        path = f"{_ORDERS_PATH}{external_order_id}"
        data = await self._request("GET", path)
        payload = data["payload"]
        return ExecutionResult(
            external_order_id=payload["oid"],
            status=payload["status"],
            filled_quantity=Decimal(0),
            average_price=None,
            raw_response=data,
        )