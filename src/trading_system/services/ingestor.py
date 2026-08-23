"""
Ingesta de datos históricos OHLCV desde Bitso a la base de datos.

Bitso API v3 endpoint público (sin auth):
    GET /v3/trades/book/  → trades individuales
    GET /v3/ticker/       → ticker actual
    GET /v3/available_books/ → catálogo de books

Para velas OHLCV, Bitso no expone un endpoint directo de candles,
pero podemos usar el endpoint público de trades y agregar, o usar
el endpoint alternativo:
    GET https://api.bitso.com/v3/trades/?book=btc_mxn&limit=100&sort=desc

Para simplicidad del entrenamiento, este módulo:
  1. Descarga trades de Bitso (público, sin auth).
  2. Los agrega a velas OHLCV del timeframe deseado usando pandas.
  3. Los persiste en ``market_data_ohlcv``.

DIP: ``httpx.AsyncClient`` se inyecta para tests.
SRP: solo descarga y persiste. No decide ni ejecuta.
"""
from __future__ import annotations

from decimal import Decimal

import httpx
import pandas as pd
import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from trading_system.core.enums import AssetClass, MarketVenue, TimeFrame
from trading_system.db.models import Asset, MarketDataOHLCV
from trading_system.db.session import get_session

logger = structlog.get_logger(__name__)

_TRADES_PATH = "/v3/trades/"
_DEFAULT_LIMIT = 100


class BitsoDataIngestor:
    """Descarga trades de Bitso y los persiste como velas OHLCV."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.bitso.com",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(base_url=base_url)
            self._owns_client = True

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch_trades(
        self,
        book: str,
        limit: int = 100,
        marker: str | None = None,
        sort: str = "desc",
    ) -> list[dict]:
        """Descarga trades de Bitso (endpoint público, sin auth)."""
        params: dict[str, str] = {
            "book": book,
            "limit": str(limit),
            "sort": sort,
        }
        if marker:
            params["marker"] = marker
        response = await self._client.get(_TRADES_PATH, params=params)
        response.raise_for_status()
        data = response.json()
        return data["payload"]

    async def fetch_trades_batch(
        self,
        book: str,
        total: int = 1000,
    ) -> pd.DataFrame:
        """Descarga múltiples lotes de trades y los devuelve como DataFrame."""
        all_trades: list[dict] = []
        marker: str | None = None
        remaining = total

        while remaining > 0:
            batch_limit = min(_DEFAULT_LIMIT, remaining)
            trades = await self.fetch_trades(
                book=book, limit=batch_limit, marker=marker, sort="desc"
            )
            if not trades:
                break
            all_trades.extend(trades)
            remaining -= len(trades)
            marker = trades[-1].get("tid")
            if len(trades) < batch_limit:
                break

        if not all_trades:
            return pd.DataFrame()

        df = pd.DataFrame(all_trades)
        df["timestamp"] = pd.to_datetime(df["created_at"], unit="ms", utc=True)
        df["price"] = df["price"].astype(float)
        df["amount"] = df["amount"].astype(float)
        return df[["timestamp", "price", "amount"]]

    @staticmethod
    def aggregate_ohlcv(
        trades: pd.DataFrame,
        timeframe: TimeFrame,
    ) -> pd.DataFrame:
        """Agrega trades en velas OHLCV del timeframe indicado."""
        if trades.empty:
            return pd.DataFrame(
                columns=["ts", "open", "high", "low", "close", "volume"]
            )

        tf_map: dict[TimeFrame, str] = {
            TimeFrame.M1: "1min",
            TimeFrame.M5: "5min",
            TimeFrame.M15: "15min",
            TimeFrame.H1: "1h",
            TimeFrame.H4: "4h",
            TimeFrame.D1: "1D",
        }
        rule = tf_map.get(timeframe, "1h")

        trades = trades.set_index("timestamp").sort_index()
        aggregated = trades.resample(rule).agg(
            {
                "price": ["first", "max", "min", "last"],
                "amount": "sum",
            }
        )
        aggregated.columns = ["open", "high", "low", "close", "volume"]
        aggregated = aggregated.dropna(subset=["open"])

        result = aggregated.reset_index().rename(columns={"timestamp": "ts"})
        return result

    def persist_ohlcv(
        self,
        ohlcv: pd.DataFrame,
        asset: Asset,
        timeframe: TimeFrame,
    ) -> int:
        """Persiste velas OHLCV en la DB (upsert). Retorna filas insertadas."""
        if ohlcv.empty:
            return 0

        rows: list[dict] = []
        for _, row in ohlcv.iterrows():
            rows.append({
                "asset_id": asset.id,
                "timeframe": str(timeframe.value),
                "ts": row["ts"].to_pydatetime(),
                "open": Decimal(str(row["open"])),
                "high": Decimal(str(row["high"])),
                "low": Decimal(str(row["low"])),
                "close": Decimal(str(row["close"])),
                "volume": Decimal(str(row["volume"])),
            })

        with get_session() as session:
            stmt = pg_insert(MarketDataOHLCV).values(rows)
            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_ohlcv_asset_tf_ts"
            )
            result = session.execute(stmt)
            return int(result.rowcount or 0)  # type: ignore[attr-defined]

    @staticmethod
    def parse_symbol(symbol: str) -> tuple[str, str]:
        """Separa un símbolo 'btc_mxn' en ('btc', 'mxn'). Lógica pura sin DB."""
        parts = symbol.split("_")
        if len(parts) == 2:
            return parts[0], parts[1]
        return symbol, "mxn"

    @staticmethod
    def get_or_create_asset(
        symbol: str,
        venue: MarketVenue = MarketVenue.BITSO,
        asset_class: AssetClass = AssetClass.CRYPTO,
        base_currency: str = "",
        quote_currency: str = "",
    ) -> tuple[str, str]:
        """Asegura que el activo existe en la DB. Retorna (base, quote)."""
        if not base_currency or not quote_currency:
            base_currency, quote_currency = BitsoDataIngestor.parse_symbol(symbol)

        with get_session() as session:
            existing = session.execute(
                select(Asset).where(
                    Asset.symbol == symbol,
                    Asset.venue == venue,
                )
            ).scalar_one_or_none()

            if existing is None:
                asset = Asset(
                    symbol=symbol,
                    name=f"{base_currency.upper()} / {quote_currency.upper()}",
                    venue=venue,
                    asset_class=asset_class,
                    base_currency=base_currency.lower(),
                    quote_currency=quote_currency.lower(),
                )
                session.add(asset)
                session.flush()
                session.commit()
                logger.info(
                    "ingestor.asset_created",
                    symbol=symbol,
                    venue=venue.value,
                    asset_id=asset.id,
                )

        return base_currency, quote_currency