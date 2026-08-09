"""
Conector REST de GBM.

Implementa ``ExchangeConnector`` usando ``httpx.AsyncClient``.

Diferencias con Bitso:
  - Auth stateful: login JWT → Bearer en cada request (no HMAC por request)
  - Endpoints bajo /api/v1/
  - Login se hace en ``connect()``, token se refresca automáticamente

DIP: ``httpx.AsyncClient`` se inyecta para tests con ``MockTransport``.
SRP: delega auth a ``GBMAuth``, calendario a ``BMVCalendar``.
"""
import json
from decimal import Decimal

import httpx
import structlog

from trading_system.connectors.base import Balance, ExecutionResult
from trading_system.connectors.gbm.auth import GBMAuth
from trading_system.core.enums import OrderSide, OrderType

logger = structlog.get_logger(__name__)

_LOGIN_PATH = "/api/v1/auth/login"
_BALANCE_PATH = "/api/v1/account/balances"
_ORDERS_PATH = "/api/v1/orders"


class GBMRESTConnector:
    """Implementación de ``ExchangeConnector`` para GBM vía REST."""

    def __init__(
        self,
        *,
        auth: GBMAuth,
        base_url: str = "https://api.gbm.com",
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
        await self._auth.ensure_valid_token(self._login_fn)
        self._connected = True
        logger.info("gbm_rest.connected", base_url=self._base_url)

    async def disconnect(self) -> None:
        self._connected = False
        if self._owns_client:
            await self._client.aclose()
        logger.info("gbm_rest.disconnected")

    async def _login_fn(self, api_key: str, api_secret: str) -> tuple[str, int]:
        body = json.dumps({"apiKey": api_key, "apiSecret": api_secret})
        response = await self._client.post(
            _LOGIN_PATH,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        payload = data["payload"]
        return payload["token"], int(payload["expiresIn"])

    async def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
    ) -> dict:
        await self._auth.ensure_valid_token(self._login_fn)
        headers = self._auth.build_headers()
        body_str = json.dumps(body) if body else ""
        if body_str:
            headers["Content-Type"] = "application/json"
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
            "symbol": symbol,
            "side": str(side.value),
            "type": str(order_type.value),
            "quantity": str(quantity),
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
            external_order_id=payload["orderId"],
            status=payload["status"],
            filled_quantity=Decimal(0),
            average_price=None,
            raw_response=data,
        )

    async def cancel_order(self, external_order_id: str) -> ExecutionResult:
        path = f"{_ORDERS_PATH}/{external_order_id}"
        data = await self._request("DELETE", path)
        payload = data["payload"]
        return ExecutionResult(
            external_order_id=payload["orderId"],
            status="cancelled",
            filled_quantity=Decimal(0),
            average_price=None,
            raw_response=data,
        )

    async def get_order_status(self, external_order_id: str) -> ExecutionResult:
        path = f"{_ORDERS_PATH}/{external_order_id}"
        data = await self._request("GET", path)
        payload = data["payload"]
        return ExecutionResult(
            external_order_id=payload["orderId"],
            status=payload["status"],
            filled_quantity=Decimal(0),
            average_price=None,
            raw_response=data,
        )