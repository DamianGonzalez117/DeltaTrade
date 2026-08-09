"""TDD — Tests for GBMRESTConnector (RED phase).

Implementa ``ExchangeConnector`` para GBM.

Diferencias con Bitso:
  - Auth stateful: login → JWT, luego Bearer en cada request
  - Endpoints distintos (/api/v1/...)
  - Respuestas con estructura diferente

El ``httpx.AsyncClient`` se inyecta con ``MockTransport`` para tests.
"""
from decimal import Decimal

import httpx
import pytest

from trading_system.connectors.base import Balance, ExecutionResult
from trading_system.connectors.gbm.auth import GBMAuth
from trading_system.connectors.gbm.rest import GBMRESTConnector
from trading_system.core.enums import OrderSide, OrderType

# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------

def _login_payload(token: str = "jwt-token", ttl: int = 3600) -> dict:
    return {"success": True, "payload": {"token": token, "expiresIn": ttl}}


def _balance_payload() -> dict:
    return {
        "success": True,
        "payload": {
            "balances": [
                {"currency": "MXN", "total": "100000.00", "available": "80000.00"},
                {"currency": "USD", "total": "5000.00", "available": "4000.00"},
            ]
        },
    }


def _place_order_payload() -> dict:
    return {
        "success": True,
        "payload": {
            "orderId": "gbm-order-123",
            "status": "queued",
        },
    }


def _order_status_payload(status: str = "filled", oid: str = "gbm-order-123") -> dict:
    return {
        "success": True,
        "payload": {
            "orderId": oid,
            "status": status,
        },
    }


def _cancel_payload() -> dict:
    return {"success": True, "payload": {"orderId": "gbm-order-123", "status": "cancelled"}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_connector(handler, api_key: str = "gbm-key", api_secret: str = "gbm-secret") -> GBMRESTConnector:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://api.gbm.com")
    auth = GBMAuth(api_key=api_key, api_secret=api_secret)
    return GBMRESTConnector(auth=auth, base_url="https://api.gbm.com", client=client)


# ---------------------------------------------------------------------------
# connect (login)
# ---------------------------------------------------------------------------

class TestConnect:
    async def test_connect_performs_login(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                seen["login"] = True
                return httpx.Response(200, json=_login_payload())
            return httpx.Response(200, json={})

        connector = _make_connector(handler)
        await connector.connect()
        assert seen.get("login") is True
        assert connector._auth.token == "jwt-token"
        await connector.disconnect()

    async def test_connect_sends_credentials_in_body(self):
        seen_body = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            seen_body["json"] = json.loads(request.content) if request.content else {}
            return httpx.Response(200, json=_login_payload())

        connector = _make_connector(handler, api_key="MYKEY", api_secret="MYSECRET")
        await connector.connect()
        assert seen_body["json"]["apiKey"] == "MYKEY"
        assert seen_body["json"]["apiSecret"] == "MYSECRET"
        await connector.disconnect()


# ---------------------------------------------------------------------------
# get_balances
# ---------------------------------------------------------------------------

class TestGetBalances:
    async def test_returns_balance_dtos(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                return httpx.Response(200, json=_login_payload())
            return httpx.Response(200, json=_balance_payload())

        connector = _make_connector(handler)
        await connector.connect()
        balances = await connector.get_balances()
        await connector.disconnect()

        assert len(balances) == 2
        assert all(isinstance(b, Balance) for b in balances)
        mxn = next(b for b in balances if b.currency == "MXN")
        assert mxn.total == Decimal("100000.00")
        assert mxn.available == Decimal("80000.00")

    async def test_injects_bearer_token(self):
        seen_headers = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                return httpx.Response(200, json=_login_payload())
            seen_headers.update(dict(request.headers))
            return httpx.Response(200, json=_balance_payload())

        connector = _make_connector(handler)
        await connector.connect()
        await connector.get_balances()
        await connector.disconnect()

        assert seen_headers["authorization"] == "Bearer jwt-token"

    async def test_refreshes_token_when_expired(self):
        call_count = {"login": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                call_count["login"] += 1
                return httpx.Response(200, json=_login_payload(token="refreshed"))
            return httpx.Response(200, json=_balance_payload())

        connector = _make_connector(handler)
        await connector.connect()
        # Forzar expiración
        connector._auth._token_expires_at = None
        await connector.get_balances()
        await connector.disconnect()

        assert call_count["login"] == 2


# ---------------------------------------------------------------------------
# place_order
# ---------------------------------------------------------------------------

class TestPlaceOrder:
    async def test_market_buy_sends_correct_body(self):
        seen_body = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                return httpx.Response(200, json=_login_payload())
            import json
            seen_body["json"] = json.loads(request.content)
            return httpx.Response(200, json=_place_order_payload())

        connector = _make_connector(handler)
        await connector.connect()
        await connector.place_order(
            symbol="NAFTRAC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal(10),
        )
        await connector.disconnect()

        assert seen_body["json"]["symbol"] == "NAFTRAC"
        assert seen_body["json"]["side"] == "buy"
        assert seen_body["json"]["type"] == "market"
        assert "quantity" in seen_body["json"]

    async def test_limit_order_includes_price(self):
        seen_body = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                return httpx.Response(200, json=_login_payload())
            import json
            seen_body["json"] = json.loads(request.content)
            return httpx.Response(200, json=_place_order_payload())

        connector = _make_connector(handler)
        await connector.connect()
        await connector.place_order(
            symbol="LIVEPOLI-1",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal(100),
            price=Decimal("550.00"),
        )
        await connector.disconnect()

        assert seen_body["json"]["type"] == "limit"
        assert seen_body["json"]["price"] == "550.00"

    async def test_limit_without_price_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                return httpx.Response(200, json=_login_payload())
            return httpx.Response(200, json=_place_order_payload())

        connector = _make_connector(handler)
        await connector.connect()
        with pytest.raises(ValueError, match="price"):
            await connector.place_order(
                symbol="NAFTRAC",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal(10),
            )
        await connector.disconnect()

    async def test_returns_execution_result(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                return httpx.Response(200, json=_login_payload())
            return httpx.Response(200, json=_place_order_payload())

        connector = _make_connector(handler)
        await connector.connect()
        result = await connector.place_order(
            symbol="NAFTRAC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal(10),
        )
        await connector.disconnect()

        assert isinstance(result, ExecutionResult)
        assert result.external_order_id == "gbm-order-123"
        assert result.status == "queued"


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------

class TestCancelOrder:
    async def test_cancel_sends_delete(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                return httpx.Response(200, json=_login_payload())
            seen["method"] = request.method
            seen["path"] = request.url.path
            return httpx.Response(200, json=_cancel_payload())

        connector = _make_connector(handler)
        await connector.connect()
        result = await connector.cancel_order("gbm-order-123")
        await connector.disconnect()

        assert seen["method"] == "DELETE"
        assert "gbm-order-123" in seen["path"]
        assert result.status == "cancelled"

    async def test_cancel_unknown_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                return httpx.Response(200, json=_login_payload())
            return httpx.Response(404, json={"success": False})

        connector = _make_connector(handler)
        await connector.connect()
        with pytest.raises(httpx.HTTPStatusError):
            await connector.cancel_order("nonexistent")
        await connector.disconnect()


# ---------------------------------------------------------------------------
# get_order_status
# ---------------------------------------------------------------------------

class TestGetOrderStatus:
    async def test_returns_current_status(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                return httpx.Response(200, json=_login_payload())
            return httpx.Response(200, json=_order_status_payload(status="filled"))

        connector = _make_connector(handler)
        await connector.connect()
        result = await connector.get_order_status("gbm-order-123")
        await connector.disconnect()

        assert result.external_order_id == "gbm-order-123"
        assert result.status == "filled"

    async def test_get_status_sends_get(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                return httpx.Response(200, json=_login_payload())
            seen["method"] = request.method
            seen["path"] = request.url.path
            return httpx.Response(200, json=_order_status_payload())

        connector = _make_connector(handler)
        await connector.connect()
        await connector.get_order_status("gbm-order-xyz")
        await connector.disconnect()

        assert seen["method"] == "GET"
        assert "gbm-order-xyz" in seen["path"]


# ---------------------------------------------------------------------------
# disconnect
# ---------------------------------------------------------------------------

class TestDisconnect:
    async def test_disconnect_after_connect(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                return httpx.Response(200, json=_login_payload())
            return httpx.Response(200, json=_balance_payload())

        connector = _make_connector(handler)
        await connector.connect()
        await connector.disconnect()
        assert True