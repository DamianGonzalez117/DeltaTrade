"""
CLI entry point for DeltaTrade.

Comandos:
    python -m trading_system.main ingest   --book btc_mxn --timeframe 1h --total 1000
    python -m trading_system.main train    --asset-id 1 --timeframe 1h --output models/xgb_v1.json
    python -m trading_system.main migrate  [up|down|stamp]
    python -m trading_system.main run      --symbol btc_mxn --venue bitso

Usa argparse (stdlib) para evitar dependencias extra.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import structlog

from trading_system.config import settings

logger = structlog.get_logger(__name__)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingesta datos históricos de Bitso y los persiste como OHLCV."""
    from trading_system.core.enums import AssetClass, MarketVenue, TimeFrame
    from trading_system.services.ingestor import BitsoDataIngestor

    async def _run():
        ingestor = BitsoDataIngestor(base_url=settings.bitso_rest_url)
        try:
            base, quote = BitsoDataIngestor.parse_symbol(args.book)
            BitsoDataIngestor.get_or_create_asset(
                symbol=args.book,
                venue=MarketVenue.BITSO,
                asset_class=AssetClass.CRYPTO,
                base_currency=base,
                quote_currency=quote,
            )

            logger.info("ingest.start", book=args.book, total=args.total)
            trades = await ingestor.fetch_trades_batch(args.book, total=args.total)

            if trades.empty:
                logger.warning("ingest.no_trades", book=args.book)
                return

            timeframe = TimeFrame(args.timeframe)
            ohlcv = BitsoDataIngestor.aggregate_ohlcv(trades, timeframe)
            logger.info("ingest.aggregated", candles=len(ohlcv))

            from sqlalchemy import select

            from trading_system.db.models import Asset
            from trading_system.db.session import get_session

            with get_session() as session:
                asset = session.execute(
                    select(Asset).where(
                        Asset.symbol == args.book,
                        Asset.venue == MarketVenue.BITSO,
                    )
                ).scalar_one()
                inserted = ingestor.persist_ohlcv(ohlcv, asset, timeframe)

            logger.info("ingest.done", inserted=inserted, book=args.book)
        finally:
            await ingestor.aclose()

    asyncio.run(_run())


def cmd_train(args: argparse.Namespace) -> None:
    """Entrena el modelo XGBoost desde datos históricos en la DB."""
    from trading_system.core.enums import TimeFrame
    from trading_system.db.session import get_session
    from trading_system.features.pipeline import TechnicalFeaturesPipeline
    from trading_system.services.training import TrainingPipeline

    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.error("train.requires_xgboost", hint="pip install xgboost")
        sys.exit(1)

    timeframe = TimeFrame(args.timeframe)

    with get_session() as session:
        ohlcv = TrainingPipeline.load_ohlcv_from_db(
            session, args.asset_id, timeframe
        )

    if ohlcv.empty:
        logger.error("train.no_data", asset_id=args.asset_id, timeframe=timeframe.value)
        sys.exit(1)

    logger.info("train.start", rows=len(ohlcv), asset_id=args.asset_id)

    pipeline = TrainingPipeline(
        features_pipeline=TechnicalFeaturesPipeline.create_default(),
        classifier_factory=XGBClassifier,
        forward_window=args.forward_window,
        buy_threshold=args.buy_threshold,
        sell_threshold=-abs(args.buy_threshold),
    )

    pipeline.train(ohlcv)

    output_path = Path(args.output)
    pipeline.save_model(output_path)
    logger.info("train.done", output=str(output_path), features=pipeline.feature_names)


def cmd_migrate(args: argparse.Namespace) -> int:
    """Ejecuta migraciones de Alembic. Retorna exit code."""
    import subprocess

    direction = args.direction
    cmd = ["alembic"]

    if direction == "up":
        cmd.append("upgrade")
        cmd.append("head")
    elif direction == "down":
        cmd.append("downgrade")
        cmd.append("-1")
    elif direction == "stamp":
        cmd.append("stamp")
        cmd.append("head")
    elif direction == "base":
        cmd.append("stamp")
        cmd.append("base")
    else:
        logger.error("migrate.unknown_direction", direction=direction)
        return 1

    logger.info("migrate.running", cmd=" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("migrate.failed", stderr=result.stderr)
        return result.returncode
    print(result.stdout)
    logger.info("migrate.done")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Ejecuta el pipeline completo en modo live (inference -> risk -> execution)."""

    logger.info(
        "run.start",
        symbol=args.symbol,
        venue=args.venue,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        logger.info("run.dry_run", msg="Pipeline no ejecutado (dry-run). Verifica config y DB.")
        return 0

    logger.info(
        "run.live",
        msg="Modo live aún no implementado. Requiere wiring completo de conectores.",
    )
    return 0


def _ensure_db() -> None:
    """Verifica que las tablas existan; si no, las crea con Base.metadata."""
    from sqlalchemy import inspect

    from trading_system.db.base import Base
    from trading_system.db.models import (  # noqa: F401 (register models)
        AISignal,
        Asset,
        Fill,
        MarketDataOHLCV,
        NotificationLog,
        Order,
        PortfolioSnapshot,
        Position,
        RiskEvaluation,
        SentimentScore,
    )
    from trading_system.db.session import engine

    inspector = inspect(engine)
    if "assets" not in inspector.get_table_names():
        logger.info("db.missing_tables", hint="Creando esquema con Base.metadata")
        Base.metadata.create_all(engine)
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM alembic_version"))
            conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('23b72d85a866')"))
            conn.commit()
        logger.info("db.schema_created")


def cmd_scan(args: argparse.Namespace) -> int:
    """Escanea libros de Bitso y notifica señales por Telegram."""
    from trading_system.core.enums import TimeFrame
    from trading_system.features.pipeline import TechnicalFeaturesPipeline
    from trading_system.notifications.dispatcher import NotificationDispatcher
    from trading_system.notifications.log_notifier import LogNotifier
    from trading_system.notifications.telegram import TelegramNotifier
    from trading_system.services.ingestor import BitsoDataIngestor
    from trading_system.services.monitor import MarketMonitor, MonitorConfig
    from trading_system.services.signal_generator import SignalGenerator
    from trading_system.services.signal_history import SignalHistoryService

    _ensure_db()
    timeframe = TimeFrame(args.timeframe)

    notifiers: list = [LogNotifier()]

    if settings.telegram_bot_token and settings.telegram_chat_id:
        tg = TelegramNotifier(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )
        notifiers.append(tg)
        logger.info("scan.telegram_enabled", chat_id=settings.telegram_chat_id)
    else:
        logger.warning("scan.telegram_not_configured", hint="Configura TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en .env")

    dispatcher = NotificationDispatcher(notifiers)

    ingestor = BitsoDataIngestor(base_url=settings.bitso_rest_url)

    history_service = None
    if not args.no_history:
        from trading_system.db.session import SessionLocal
        history_service = SignalHistoryService(SessionLocal())

    monitor = MarketMonitor(
        ingestor=ingestor,
        features_pipeline=TechnicalFeaturesPipeline.create_default(),
        signal_generator=SignalGenerator(),
        dispatcher=dispatcher,
        config=MonitorConfig(
            books=args.books,
            timeframe=timeframe,
            trades_per_book=args.trades,
            interval_seconds=args.interval,
            min_confidence=args.min_confidence,
        ),
        history_service=history_service,
    )

    async def _run() -> None:
        try:
            if args.once:
                results = await monitor.scan_once()
                for book, signal in results.items():
                    pos = history_service.get_open_position(book) if history_service else None
                    pos_str = f" [POSITION: qty={pos.quantity}, entry={pos.avg_entry_price}]" if pos else ""
                    print(f"  {book}: {signal.action.value.upper()} (conf={signal.confidence}, rsi={signal.rsi:.1f}){pos_str}")
            elif args.interactive and history_service is not None:
                import asyncio as _aio

                from trading_system.services.telegram_bot import TelegramCommandBot

                tg_bot = TelegramCommandBot(
                    bot_token=settings.telegram_bot_token,
                    chat_id=settings.telegram_chat_id,
                    history_service=history_service,
                    dispatcher=dispatcher,
                    monitor=monitor,
                )

                scan_task = _aio.create_task(monitor.run())
                bot_task = _aio.create_task(tg_bot.run())

                try:
                    await _aio.gather(scan_task, bot_task)
                except KeyboardInterrupt:
                    monitor.stop()
                    tg_bot.stop()
                    scan_task.cancel()
                    bot_task.cancel()
            else:
                await monitor.run()
        finally:
            await ingestor.aclose()
            for n in notifiers:
                if hasattr(n, "aclose"):
                    await n.aclose()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("scan.interrupted")
        monitor.stop()

    return 0


def cmd_confirm_buy(args: argparse.Namespace) -> int:
    """Confirma una compra y abre una posición en la DB."""
    from trading_system.db.session import SessionLocal
    from trading_system.services.signal_history import SignalHistoryService

    service = SignalHistoryService(SessionLocal())
    try:
        position = service.confirm_buy(
            book=args.book,
            quantity=Decimal(args.qty),
            entry_price=Decimal(args.price),
        )
        print(f"✅ Posición abierta: id={position.id} book={args.book} qty={position.quantity} entry={position.avg_entry_price}")
        logger.info("confirm_buy.success", book=args.book, position_id=position.id)
        return 0
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1


def cmd_close_position(args: argparse.Namespace) -> int:
    """Cierra una posición abierta y calcula P&L."""
    from trading_system.db.session import SessionLocal
    from trading_system.services.signal_history import SignalHistoryService

    service = SignalHistoryService(SessionLocal())
    try:
        position = service.close_position(
            book=args.book,
            exit_price=Decimal(args.price),
        )
        pnl = position.realized_pnl
        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        print(f"✅ Posición cerrada: id={position.id} book={args.book} exit={Decimal(args.price):,} P&L={pnl_str}")
        logger.info("close_position.success", book=args.book, position_id=position.id, pnl=str(pnl))
        return 0
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1


def cmd_positions(args: argparse.Namespace) -> int:
    """Lista posiciones abiertas."""
    from trading_system.db.session import SessionLocal
    from trading_system.services.signal_history import SignalHistoryService

    service = SignalHistoryService(SessionLocal())
    positions = service.get_all_open_positions()

    if not positions:
        print("  No hay posiciones abiertas.")
        return 0

    print(f"  Posiciones abiertas ({len(positions)}):")
    for pos in positions:
        print(
            f"    id={pos.id}  {pos.asset.symbol:12s}  "
            f"qty={pos.quantity:<14}  entry=${pos.avg_entry_price:,.2f}  "
            f"opened={pos.opened_at.strftime('%Y-%m-%d %H:%M')}"
        )
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """Muestra el historial de señales."""
    from trading_system.db.session import SessionLocal
    from trading_system.services.signal_history import SignalHistoryService

    service = SignalHistoryService(SessionLocal())
    signals = service.get_signal_history(
        book=args.book,
        limit=args.limit,
    )

    if not signals:
        print("  No hay señales registradas.")
        return 0

    print(f"  Historial de señales ({len(signals)}):")
    for sig in signals:
        action_str = sig.action.value if hasattr(sig.action, "value") else str(sig.action)
        print(
            f"    {sig.generated_at.strftime('%Y-%m-%d %H:%M')}  "
            f"{sig.asset.symbol:12s}  {action_str.upper():4s}  "
            f"conf={sig.confidence:.4f}  model={sig.model_name}"
        )
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Ejecuta backtest sobre datos históricos de la DB."""
    import json

    from trading_system.core.enums import SignalAction, TimeFrame
    from trading_system.db.session import get_session
    from trading_system.features.pipeline import TechnicalFeaturesPipeline
    from trading_system.risk.engine import RiskEngine
    from trading_system.services.backtesting import BacktestConfig, BacktestEngine
    from trading_system.services.training import TrainingPipeline

    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.error("backtest.requires_xgboost", hint="pip install xgboost")
        return 1

    timeframe = TimeFrame(args.timeframe)

    with get_session() as session:
        ohlcv = TrainingPipeline.load_ohlcv_from_db(session, args.asset_id, timeframe)

    if ohlcv.empty:
        logger.error("backtest.no_data", asset_id=args.asset_id)
        return 1

    logger.info("backtest.start", rows=len(ohlcv), asset_id=args.asset_id)

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("backtest.model_not_found", path=str(model_path))
        return 1

    meta_path = model_path.with_suffix(".features.json")
    with open(meta_path) as f:
        meta = json.load(f)
    feature_names: list[str] = meta["feature_names"]

    classifier = XGBClassifier()
    classifier.load_model(str(model_path))

    def predict(features: dict[str, float]) -> tuple[SignalAction, Decimal]:
        ordered = [features.get(name, 0.0) for name in feature_names]
        probas = classifier.predict_proba([ordered])[0]
        best_idx = int(np.argmax(probas))
        actions = [SignalAction.BUY, SignalAction.HOLD, SignalAction.SELL]
        return actions[best_idx], Decimal(str(float(probas[best_idx])))

    config = BacktestConfig(
        initial_capital=Decimal(args.capital),
    )
    engine = BacktestEngine(
        features_pipeline=TechnicalFeaturesPipeline.create_default(),
        predict_fn=predict,
        risk_engine=RiskEngine.create_default(),
        config=config,
    )

    result = engine.run(ohlcv)
    logger.info(
        "backtest.result",
        return_pct=result.total_return_pct,
        max_drawdown_pct=result.max_drawdown_pct,
        sharpe=result.sharpe_ratio,
        trades=result.num_trades,
        win_rate=result.win_rate,
        final_equity=str(result.final_equity),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-system",
        description="DeltaTrade — sistema multi-mercado de trading algorítmico.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingesta datos históricos de Bitso")
    ingest.add_argument("--book", default="btc_mxn", help="Par Bitso (ej. btc_mxn)")
    ingest.add_argument("--timeframe", default="1h", help="Timeframe (1m,5m,15m,1h,4h,1d)")
    ingest.add_argument("--total", type=int, default=1000, help="Número de trades a descargar")
    ingest.set_defaults(func=cmd_ingest)

    train = sub.add_parser("train", help="Entrena modelo XGBoost desde datos en DB")
    train.add_argument("--asset-id", type=int, required=True, help="ID del activo en la DB")
    train.add_argument("--timeframe", default="1h", help="Timeframe")
    train.add_argument("--output", default="models/xgb_v1.json", help="Ruta de salida del modelo")
    train.add_argument("--forward-window", type=int, default=5, help="Ventana forward-return")
    train.add_argument("--buy-threshold", type=float, default=0.01, help="Threshold BUY")
    train.set_defaults(func=cmd_train)

    migrate = sub.add_parser("migrate", help="Ejecuta migraciones Alembic")
    migrate.add_argument(
        "direction",
        choices=["up", "down", "stamp", "base"],
        help="up=upgrade head, down=rollback 1, stamp=stamp head, base=stamp base",
    )
    migrate.set_defaults(func=cmd_migrate)

    run = sub.add_parser("run", help="Ejecuta el pipeline en modo live")
    run.add_argument("--symbol", default="btc_mxn", help="Símbolo a operar")
    run.add_argument("--venue", default="bitso", choices=["bitso", "gbm"], help="Venue")
    run.add_argument("--dry-run", action="store_true", help="No enviar órdenes reales")
    run.set_defaults(func=cmd_run)

    scan = sub.add_parser("scan", help="Escanea libros de Bitso y notifica señales por Telegram")
    scan.add_argument(
        "--books",
        nargs="+",
        default=["btc_mxn", "eth_mxn"],
        help="Libros de Bitso a escanear (ej. btc_mxn eth_mxn)",
    )
    scan.add_argument("--timeframe", default="1h", help="Timeframe (1m,5m,15m,1h,4h,1d)")
    scan.add_argument("--trades", type=int, default=1000, help="Trades a descargar por libro")
    scan.add_argument("--interval", type=float, default=300.0, help="Segundos entre escaneos")
    scan.add_argument("--min-confidence", type=float, default=0.65, help="Confianza mínima para notificar")
    scan.add_argument("--once", action="store_true", help="Solo un escaneo, sin loop")
    scan.add_argument("--interactive", action="store_true", help="Bot interactivo de Telegram (responde compré/vendí/posiciones)")
    scan.add_argument("--no-history", action="store_true", help="No persistir señales ni filtrar por posiciones")
    scan.set_defaults(func=cmd_scan)

    confirm_buy = sub.add_parser("confirm-buy", help="Confirma una compra y abre posición")
    confirm_buy.add_argument("--book", required=True, help="Libro (ej. btc_mxn)")
    confirm_buy.add_argument("--qty", required=True, help="Cantidad comprada")
    confirm_buy.add_argument("--price", required=True, help="Precio de entrada")
    confirm_buy.set_defaults(func=cmd_confirm_buy)

    close_pos = sub.add_parser("close-position", help="Cierra una posición y calcula P&L")
    close_pos.add_argument("--book", required=True, help="Libro (ej. btc_mxn)")
    close_pos.add_argument("--price", required=True, help="Precio de salida")
    close_pos.set_defaults(func=cmd_close_position)

    positions = sub.add_parser("positions", help="Lista posiciones abiertas")
    positions.set_defaults(func=cmd_positions)

    history = sub.add_parser("history", help="Muestra historial de señales")
    history.add_argument("--book", default=None, help="Filtrar por libro")
    history.add_argument("--limit", type=int, default=20, help="Número máximo de señales")
    history.set_defaults(func=cmd_history)

    backtest = sub.add_parser("backtest", help="Backtest sobre datos históricos")
    backtest.add_argument("--asset-id", type=int, required=True, help="ID del activo")
    backtest.add_argument("--timeframe", default="1h", help="Timeframe")
    backtest.add_argument("--model", default="models/xgb_v1.json", help="Modelo entrenado")
    backtest.add_argument("--capital", default="10000", help="Capital inicial")
    backtest.set_defaults(func=cmd_backtest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())