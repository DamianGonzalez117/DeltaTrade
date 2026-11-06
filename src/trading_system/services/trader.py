"""
Ejecución de compras y ventas reales en Bitso para el asistente interactivo.

Responsabilidades:
  1. ``buy_market``: compra por monto en MXN (orden de mercado con
     ``minor``) tras validar saldo disponible; agrega los fills reales
     (``GET /v3/order_trades/``) para reportar cantidad neta y precio
     promedio (comisiones descontadas cuando se cobran en cripto).
  2. ``sell_market``: vende toda la cantidad de la posición (orden de
     mercado con ``major``) y reporta el monto neto recibido en MXN.
  3. ``get_available``: saldo disponible de una moneda (para validar
     cuánto se puede comprar antes de preguntar el monto).

DIP: ``BitsoRESTConnector`` se inyecta (tests con MockTransport).
SRP: solo ejecuta y agrega fills. No decide, no conversa, no persiste.

Si los fills no se pueden consultar (error de red), se usa ``price_hint``
(último precio del monitor) como estimación y ``used_fallback=True``.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING

import httpx
import structlog

from trading_system.core.enums import OrderSide, OrderType

if TYPE_CHECKING:
    from trading_system.connectors.bitso.rest import BitsoRESTConnector

logger = structlog.get_logger(__name__)

_QTY_DP = Decimal("0.00000001")  # 8 decimales como máximo en Bitso


@dataclass(frozen=True, slots=True)
class FillDetail:
    """Resultado de una orden de mercado ejecutada (real o estimado)."""

    order_id: str
    quantity: Decimal  # major neto (comisión descontada si se cobró en major)
    average_price: Decimal  # precio promedio efectivo (incluye efecto de comisión)
    total_minor: Decimal  # monto en la moneda quote (MXN): gastado o recibido
    fee_amount: Decimal
    fee_currency: str
    used_fallback: bool = False


class BitsoTrader:
    """Coloca órdenes de mercado en Bitso y agrega los llenados reales."""

    def __init__(
        self,
        *,
        connector: BitsoRESTConnector,
        min_amount_mxn: Decimal = Decimal(25),
    ) -> None:
        self._connector = connector
        self._min_amount = min_amount_mxn

    @property
    def min_amount_mxn(self) -> Decimal:
        """Monto mínimo de compra en MXN."""
        return self._min_amount

    async def get_available(self, currency: str = "mxn") -> Decimal:
        """Saldo disponible de ``currency`` (0 si no existe la moneda)."""
        balances = await self._connector.get_balances()
        for balance in balances:
            if balance.currency == currency:
                return balance.available
        return Decimal(0)

    async def buy_market(
        self,
        book: str,
        amount_mxn: Decimal,
        price_hint: float,
    ) -> FillDetail:
        """Compra por monto en MXN (market con ``minor``).

        Valida el saldo disponible antes de ordenar. Agrega los fills para
        obtener la cantidad neta de cripto (si la comisión se cobra en
        cripto, se descuenta) y el precio promedio efectivo.
        """
        if amount_mxn <= 0:
            raise ValueError("El monto debe ser mayor a 0")
        if amount_mxn < self._min_amount:
            raise ValueError(f"El monto mínimo es ${self._min_amount:,.0f} MXN")

        available = await self.get_available("mxn")
        if amount_mxn > available:
            raise ValueError(
                f"Saldo insuficiente: tienes ${available:,.2f} MXN disponibles"
            )

        base, quote = _split_book(book)
        result = await self._connector.place_order(
            symbol=book,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            minor=amount_mxn,
        )
        logger.info(
            "trader.buy_order_placed",
            book=book,
            order_id=result.external_order_id,
            amount_mxn=str(amount_mxn),
        )

        fills = await self._safe_order_trades(result.external_order_id)
        if fills:
            return self._aggregate_buy(result.external_order_id, fills, base, quote)

        # Sin fills consultables → estimación con el último precio conocido.
        avg = Decimal(str(price_hint))
        qty = _quantize(amount_mxn / avg)
        logger.warning("trader.buy_fallback_price", book=book, order_id=result.external_order_id)
        return FillDetail(
            order_id=result.external_order_id,
            quantity=qty,
            average_price=avg,
            total_minor=amount_mxn,
            fee_amount=Decimal(0),
            fee_currency="",
            used_fallback=True,
        )

    async def sell_market(
        self,
        book: str,
        quantity: Decimal,
        price_hint: float,
    ) -> FillDetail:
        """Vende ``quantity`` (market con ``major``) y reporta lo recibido.

        ``average_price`` es el precio de salida neto (comisión en MXN
        descontada del monto recibido), de modo que
        ``(exit − entry) × qty`` coincide con la ganancia neta.
        """
        if quantity <= 0:
            raise ValueError("La cantidad a vender debe ser mayor a 0")

        base, quote = _split_book(book)
        result = await self._connector.place_order(
            symbol=book,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )
        logger.info(
            "trader.sell_order_placed",
            book=book,
            order_id=result.external_order_id,
            quantity=str(quantity),
        )

        fills = await self._safe_order_trades(result.external_order_id)
        if fills:
            return self._aggregate_sell(result.external_order_id, fills, base, quote)

        avg = Decimal(str(price_hint))
        logger.warning("trader.sell_fallback_price", book=book, order_id=result.external_order_id)
        return FillDetail(
            order_id=result.external_order_id,
            quantity=quantity,
            average_price=avg,
            total_minor=_quantize(avg * quantity),
            fee_amount=Decimal(0),
            fee_currency="",
            used_fallback=True,
        )

    async def _safe_order_trades(self, order_id: str) -> list[dict]:
        """Consulta los fills; ante cualquier error devuelve [] (fallback)."""
        try:
            fills = await self._connector.get_order_trades(order_id)
            return [f for f in fills if f.get("major") not in (None, "0")]
        except (httpx.HTTPError, httpx.TimeoutException, KeyError, ValueError, RuntimeError, OSError) as exc:
            logger.warning("trader.order_trades_failed", order_id=order_id, error=str(exc))
            return []

    def _aggregate_buy(
        self,
        order_id: str,
        fills: list[dict],
        base: str,
        quote: str,
    ) -> FillDetail:
        """Agrega los fills de una compra: qty neta + precio efectivo.

        Compra típica: comisión cobrada en la cripto recibida → la cantidad
        neta es gross − fee. Precio efectivo = gastado / qty neta, para que
        ``entry × qty`` coincida con el MXN realmente gastado.
        """
        gross = _abs_sum(f["major"] for f in fills)
        total_minor = _abs_sum(f["minor"] for f in fills)
        fee_amount, fee_currency = _sum_fees(fills)

        net = gross
        spent = total_minor
        if fee_currency == base and fee_amount > 0:
            net = gross - fee_amount
        elif fee_currency == quote and fee_amount > 0:
            spent = total_minor + fee_amount

        net = _quantize(net)
        average_price = _quantize(spent / net) if net > 0 else Decimal(0)
        return FillDetail(
            order_id=order_id,
            quantity=net,
            average_price=average_price,
            total_minor=spent,
            fee_amount=fee_amount,
            fee_currency=fee_currency,
        )

    def _aggregate_sell(
        self,
        order_id: str,
        fills: list[dict],
        base: str,
        quote: str,
    ) -> FillDetail:
        """Agrega los fills de una venta: recibido neto + precio de salida.

        Venta típica: comisión cobrada en MXN → recibido = total − fee.
        Precio de salida efectivo = recibido / qty vendida.
        """
        gross = _abs_sum(f["major"] for f in fills)
        total_minor = _abs_sum(f["minor"] for f in fills)
        fee_amount, fee_currency = _sum_fees(fills)

        proceeds = total_minor
        if fee_currency == quote and fee_amount > 0:
            proceeds = total_minor - fee_amount

        average_price = _quantize(proceeds / gross) if gross > 0 else Decimal(0)
        return FillDetail(
            order_id=order_id,
            quantity=_quantize(gross),
            average_price=average_price,
            total_minor=proceeds,
            fee_amount=fee_amount,
            fee_currency=fee_currency,
        )

    async def aclose(self) -> None:
        await self._connector.disconnect()


def _split_book(book: str) -> tuple[str, str]:
    parts = book.split("_")
    if len(parts) != 2:
        raise ValueError(f"Book inválido: {book}")
    return parts[0].lower(), parts[1].lower()


def _dec(value) -> Decimal:
    return Decimal(str(value))


def _abs_sum(values) -> Decimal:
    """Suma valores absolutos empezando en Decimal(0) (tipado estable)."""
    total = Decimal(0)
    for value in values:
        total += abs(_dec(value))
    return total


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(_QTY_DP, rounding=ROUND_DOWN)


def _sum_fees(fills: list[dict]) -> tuple[Decimal, str]:
    """Suma las comisiones de una sola moneda (Bitso cobra en major o minor)."""
    by_currency: dict[str, Decimal] = {}
    for fill in fills:
        currency = str(fill.get("fees_currency") or "")
        amount = abs(_dec(fill.get("fees_amount") or 0))
        if currency and amount > 0:
            by_currency[currency] = by_currency.get(currency, Decimal(0)) + amount
    if not by_currency:
        return Decimal(0), ""
    currency = next(iter(by_currency))
    return by_currency[currency], currency
