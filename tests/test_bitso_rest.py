"""TDD — Tests for BitsoRESTConnector (RED phase).

Cada endpoint de Bitso v3 se mockea con ``httpx.MockTransport`` para que
los tests no toquen la red. El conector debe:
  1. Inyectar los headers de auth correctos en cada request.
  2. Mapear las respuestas JSON de Bitso a los DTOs del dominio.
  3. Manejar errores HTTP y de formato de manera predecible.
"""
from decimal import Decimal

import httpx
import pytest

from trading_system.connectors.base import Balance, ExecutionResult
from trading_system.connectors.bitso.auth import BitsoAuth
from trading_system.connectors.bitso.rest import BitsoRESTConnector
from trading_system.core.enums import OrderSide, OrderType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _balance_payload() -> dict:
    return {
        "success": True,
        "payload": {
            "balances": [
                {"currency": "btc", "total": "1.50000000", "available": "1.00000000"},
                {"currency": "mxn", "total": "50000.00000000", "available": "45000.00000000"},
            ]
        },
    }


def _place_order_payload() -> dict:
    return {
        "success": True,
        "payload": {
            "oid": "order-abc-123",
            "status": "open",
        },
    }


def _order_status_payload(status: str = "filled", oid: str = "order-abc-123") -> dict:
    return {
        "success": True,
        "payload": {
            "oid": oid,
            "status": status,
        },
    }


def _cancel_payload() -> dict:
    return {
        "success": True,
        "payload": ["order-abc-123"],
    }


def _make_connector(
    handler,
    api_key: str = "test-key",
    api_secret: str = "test-secret",
) -> BitsoRESTConnector:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://api.bitso.com")
    auth = BitsoAuth(api_key, api_secret)
    return BitsoRESTConnector(auth=auth, base_url="https://api.bitso.com", client=client)


# ---------------------------------------------------------------------------
# get_balances
# ---------------------------------------------------------------------------

class TestGetBalances:
    async def test_returns_list_of_balance_dtos(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v3/balance/"
            assert request.method == "GET"
            return httpx.Response(200, json=_balance_payload())

        connector = _make_connector(handler)
        await connector.connect()
        balances = await connector.get_balances()
        await connector.disconnect()

        assert len(balances) == 2
        assert all(isinstance(b, Balance) for b in balances)
        btc = next(b for b in balances if b.currency == "btc")
        assert btc.total == Decimal("1.5")
        assert btc.available == Decimal("1.0")
        mxn = next(b for b in balances if b.currency == "mxn")
        assert mxn.total == Decimal("50000.0")

    async def test_injects_auth_headers(self):
        seen_headers = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_headers.update(dict(request.headers))
            return httpx.Response(200, json=_balance_payload())

        connector = _make_connector(handler, api_key="MYKEY", api_secret="MYSECRET")
        await connector.connect()
        await connector.get_balances()
        await connector.disconnect()

        # un solo header: Bitso <key>:<nonce>:<signature>
        assert seen_headers["authorization"].startswith("Bitso MYKEY:")
        assert seen_headers["authorization"].count(":") == 2
        # el esquema viejo de headers separados ya no se envía
        assert "x-request-signature" not in seen_headers
        assert "x-request-timestamp" not in seen_headers

    async def test_http_400_raises_http_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"success": False, "error": {"code": "bad"}})

        connector = _make_connector(handler)
        await connector.connect()
        with pytest.raises(httpx.HTTPStatusError):
            await connector.get_balances()
        await connector.disconnect()


# ---------------------------------------------------------------------------
# place_order
# ---------------------------------------------------------------------------

class TestPlaceOrder:
    async def test_market_buy_sends_correct_body(self):
        seen_body = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_body["json"] = __import__("json").loads(request.content)
            return httpx.Response(200, json=_place_order_payload())

        connector = _make_connector(handler)
        await connector.connect()
        await connector.place_order(
            symbol="btc_mxn",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
        )
        await connector.disconnect()

        assert seen_body["json"]["book"] == "btc_mxn"
        assert seen_body["json"]["side"] == "buy"
        assert seen_body["json"]["type"] == "market"
        assert "major" in seen_body["json"]
        assert "price" not in seen_body["json"]

    async def test_market_buy_with_minor_sends_minor_not_major(self):
        """Compra por monto en MXN: market + minor (para el asistente)."""
        seen_body = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_body["json"] = __import__("json").loads(request.content)
            return httpx.Response(200, json=_place_order_payload())

        connector = _make_connector(handler)
        await connector.connect()
        await connector.place_order(
            symbol="btc_mxn",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            minor=Decimal("1500.50"),
        )
        await connector.disconnect()

        assert seen_body["json"]["minor"] == "1500.50"
        assert "major" not in seen_body["json"]
        assert seen_body["json"]["type"] == "market"

    async def test_minor_rejected_for_limit_orders(self):
        connector = _make_connector(lambda r: httpx.Response(200, json=_place_order_payload()))
        await connector.connect()
        with pytest.raises(ValueError, match="minor"):
            await connector.place_order(
                symbol="btc_mxn",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                minor=Decimal("1500.50"),
            )
        await connector.disconnect()

    async def test_major_and_minor_together_rejected(self):
        connector = _make_connector(lambda r: httpx.Response(200, json=_place_order_payload()))
        await connector.connect()
        with pytest.raises(ValueError, match="no ambos"):
            await connector.place_order(
                symbol="btc_mxn",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.01"),
                minor=Decimal("1500.50"),
            )
        await connector.disconnect()

    async def test_neither_major_nor_minor_rejected(self):
        connector = _make_connector(lambda r: httpx.Response(200, json=_place_order_payload()))
        await connector.connect()
        with pytest.raises(ValueError, match="Especifica"):
            await connector.place_order(
                symbol="btc_mxn",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
            )
        await connector.disconnect()

    async def test_limit_order_includes_price(self):
        seen_body = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_body["json"] = __import__("json").loads(request.content)
            return httpx.Response(200, json=_place_order_payload())

        connector = _make_connector(handler)
        await connector.connect()
        await connector.place_order(
            symbol="btc_mxn",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.5"),
            price=Decimal("1000000.00"),
        )
        await connector.disconnect()

        assert seen_body["json"]["type"] == "limit"
        assert seen_body["json"]["price"] == "1000000.00"

    async def test_limit_order_without_price_raises(self):
        connector = _make_connector(lambda r: httpx.Response(200, json=_place_order_payload()))
        await connector.connect()
        with pytest.raises(ValueError, match="price"):
            await connector.place_order(
                symbol="btc_mxn",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.01"),
            )
        await connector.disconnect()

    async def test_returns_execution_result(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_place_order_payload())

        connector = _make_connector(handler)
        await connector.connect()
        result = await connector.place_order(
            symbol="btc_mxn",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
        )
        await connector.disconnect()

        assert isinstance(result, ExecutionResult)
        assert result.external_order_id == "order-abc-123"
        assert result.status == "open"
        assert result.filled_quantity == Decimal(0)
        assert result.average_price is None
        assert result.raw_response["success"] is True


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------

class TestCancelOrder:
    async def test_cancel_sends_delete_to_correct_path(self):
        seen_path = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_path["method"] = request.method
            seen_path["path"] = request.url.path
            return httpx.Response(200, json=_cancel_payload())

        connector = _make_connector(handler)
        await connector.connect()
        result = await connector.cancel_order("order-abc-123")
        await connector.disconnect()

        assert seen_path["method"] == "DELETE"
        assert seen_path["path"] == "/v3/orders/order-abc-123"
        assert isinstance(result, ExecutionResult)
        assert result.external_order_id == "order-abc-123"

    async def test_cancel_unknown_order_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"success": False, "error": {"code": "not_found"}})

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
            return httpx.Response(200, json=_order_status_payload(status="filled"))

        connector = _make_connector(handler)
        await connector.connect()
        result = await connector.get_order_status("order-abc-123")
        await connector.disconnect()

        assert result.external_order_id == "order-abc-123"
        assert result.status == "filled"

    async def test_get_status_sends_get_request(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["path"] = request.url.path
            return httpx.Response(200, json=_order_status_payload())

        connector = _make_connector(handler)
        await connector.connect()
        await connector.get_order_status("order-xyz")
        await connector.disconnect()

        assert seen["method"] == "GET"
        assert seen["path"] == "/v3/orders/order-xyz"


class TestGetOrderTrades:
    async def test_returns_fills_list_from_correct_path(self):
        seen = {}
        payload = {
            "success": True,
            "payload": [
                {
                    "book": "btc_mxn",
                    "major": "0.00200000",
                    "minor": "-1000.00",
                    "price": "500000.00",
                    "fees_amount": "-0.00000200",
                    "fees_currency": "btc",
                    "side": "buy",
                    "tid": "1",
                    "oid": "order-abc-123",
                }
            ],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["path"] = request.url.path
            return httpx.Response(200, json=payload)

        connector = _make_connector(handler)
        await connector.connect()
        trades = await connector.get_order_trades("order-abc-123")
        await connector.disconnect()

        assert seen["method"] == "GET"
        assert seen["path"] == "/v3/order_trades/order-abc-123"
        assert len(trades) == 1
        assert trades[0]["oid"] == "order-abc-123"

    async def test_non_list_payload_returns_empty(self):
        payload = {"success": True, "payload": {"unexpected": "dict"}}

        connector = _make_connector(lambda r: httpx.Response(200, json=payload))
        await connector.connect()
        trades = await connector.get_order_trades("order-abc-123")
        await connector.disconnect()

        assert trades == []


# ---------------------------------------------------------------------------
# connect / disconnect lifecycle
# ---------------------------------------------------------------------------

class TestConnectDisconnect:
    async def test_connect_then_disconnect_does_not_raise(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_balance_payload())

        connector = _make_connector(handler)
        await connector.connect()
        await connector.disconnect()
        assert True