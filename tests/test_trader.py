"""TDD — Tests para BitsoTrader (órdenes reales + agregación de fills).

Se mockea el conector REST; nada toca la red.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from trading_system.core.enums import OrderSide, OrderType
from trading_system.services.trader import BitsoTrader, FillDetail


def _order_trades_payload(oid: str = "ord-123") -> dict:
    """Compra de 0.002 BTC por ~1000 MXN con comisión en BTC."""
    return {
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
                "oid": oid,
            },
        ],
    }


def _mock_connector(*, available_mxn: str = "5000.00") -> MagicMock:
    connector = MagicMock()
    connector.get_balances = AsyncMock(return_value=[
        MagicMock(currency="btc", total=Decimal("0.01"), available=Decimal("0.01")),
        MagicMock(currency="mxn", total=Decimal(available_mxn), available=Decimal(available_mxn)),
    ])
    return connector


class TestGetAvailable:
    async def test_returns_available_for_currency(self):
        connector = _mock_connector()
        trader = BitsoTrader(connector=connector)

        assert await trader.get_available("mxn") == Decimal("5000.00")

    async def test_returns_zero_for_missing_currency(self):
        connector = MagicMock()
        connector.get_balances = AsyncMock(return_value=[])
        trader = BitsoTrader(connector=connector)

        assert await trader.get_available("mxn") == Decimal(0)


class TestBuyMarket:
    async def test_sends_market_order_with_minor_amount(self):
        connector = _mock_connector()
        connector.place_order = AsyncMock(return_value=MagicMock(external_order_id="ord-123"))
        connector.get_order_trades = AsyncMock(return_value=_order_trades_payload()["payload"])
        trader = BitsoTrader(connector=connector)

        fill = await trader.buy_market("btc_mxn", Decimal(1000), 500_000.0)

        connector.place_order.assert_awaited_once_with(
            symbol="btc_mxn",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            minor=Decimal(1000),
        )
        assert fill.order_id == "ord-123"

    async def test_aggregates_fees_charged_in_major(self):
        connector = _mock_connector()
        connector.place_order = AsyncMock(return_value=MagicMock(external_order_id="ord-1"))
        connector.get_order_trades = AsyncMock(return_value=_order_trades_payload()["payload"])
        trader = BitsoTrader(connector=connector)

        fill = await trader.buy_market("btc_mxn", Decimal(1000), 500_000.0)

        # net = gross(0.002) - fee(0.000002) = 0.001998
        assert fill.quantity == Decimal("0.00199800")
        # precio efectivo = gastado / neto = 1000 / 0.001998 = 500500.50...
        assert fill.average_price == Decimal("500500.50050050")
        assert fill.total_minor == Decimal("1000.00")
        assert fill.fee_amount == Decimal("0.00000200")
        assert fill.fee_currency == "btc"
        assert not fill.used_fallback

    async def test_rejects_amount_below_minimum(self):
        trader = BitsoTrader(connector=MagicMock(), min_amount_mxn=Decimal(100))
        with pytest.raises(ValueError, match="mínimo"):
            await trader.buy_market("btc_mxn", Decimal(50), 500_000.0)

    async def test_rejects_insufficient_balance(self):
        connector = MagicMock()
        connector.get_balances = AsyncMock(return_value=[
            MagicMock(currency="mxn", total=Decimal(100), available=Decimal(100)),
        ])
        trader = BitsoTrader(connector=connector)

        with pytest.raises(ValueError, match="Saldo insuficiente"):
            await trader.buy_market("btc_mxn", Decimal(500), 500_000.0)

    async def test_falls_back_to_price_hint_without_fills(self):
        connector = _mock_connector()
        connector.place_order = AsyncMock(return_value=MagicMock(external_order_id="ord-9"))
        connector.get_order_trades = AsyncMock(side_effect=httpx.ConnectError("boom"))
        trader = BitsoTrader(connector=connector)

        fill = await trader.buy_market("btc_mxn", Decimal(500), 500_000.0)

        assert fill.used_fallback is True
        assert fill.quantity == Decimal("0.00100000")
        assert fill.average_price == Decimal("500000.0")


class TestSellMarket:
    async def test_sells_with_major_amount(self):
        connector = MagicMock()
        connector.place_order = AsyncMock(return_value=MagicMock(external_order_id="ord-2"))
        trader = BitsoTrader(connector=connector)
        connector.get_order_trades = AsyncMock(return_value=[
            {
                "book": "btc_mxn",
                "major": "-0.00199800",
                "minor": "1028.96700",
                "price": "515300.00",
                "fees_amount": "-10.28967",
                "fees_currency": "mxn",
                "side": "sell",
                "tid": "2",
                "oid": "ord-2",
            },
        ])

        fill = await trader.sell_market("btc_mxn", Decimal("0.00199800"), 515_000.0)

        connector.place_order.assert_awaited_once_with(
            symbol="btc_mxn",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.00199800"),
        )
        # recibido neto = 1028.967 - 10.28967 MXN de comisión
        assert fill.total_minor == Decimal("1018.67733")
        # precio de salida neto = recibido / qty = 1018.67733 / 0.001998
        assert fill.average_price == Decimal("509848.51351351")
        assert fill.fee_currency == "mxn"

    async def test_rejects_zero_quantity(self):
        trader = BitsoTrader(connector=MagicMock())
        with pytest.raises(ValueError, match="mayor a 0"):
            await trader.sell_market("btc_mxn", Decimal(0), 500_000.0)

    async def test_fallback_without_fills(self):
        connector = MagicMock()
        connector.place_order = AsyncMock(return_value=MagicMock(external_order_id="ord-3"))
        connector.get_order_trades = AsyncMock(return_value=[])
        trader = BitsoTrader(connector=connector)

        fill = await trader.sell_market("btc_mxn", Decimal("0.01"), 600_000.0)

        assert fill.used_fallback is True
        assert fill.average_price == Decimal("600000.0")
        assert fill.total_minor == Decimal("6000.00000000")


class TestFillDetailDefaults:
    def test_is_frozen_dataclass(self):
        fill = FillDetail(
            order_id="x",
            quantity=Decimal(1),
            average_price=Decimal(1),
            total_minor=Decimal(1),
            fee_amount=Decimal(0),
            fee_currency="",
        )
        assert fill.used_fallback is False
