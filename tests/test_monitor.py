"""TDD — Tests for MarketMonitor (RED→GREEN).

Se mockea BitsoDataIngestor y NotificationDispatcher.
No toca la red ni la DB.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from trading_system.core.enums import SignalAction, TimeFrame
from trading_system.features.pipeline import TechnicalFeaturesPipeline
from trading_system.notifications.dispatcher import NotificationDispatcher
from trading_system.services.ingestor import BitsoDataIngestor
from trading_system.services.monitor import MarketMonitor, MonitorConfig
from trading_system.services.signal_generator import RuleSignal, SignalGenerator


def _make_trades_df(n: int = 2000, base_price: float = 500_000.0, seed: int = 42) -> pd.DataFrame:
    """Genera trades sintéticos con suficiente variación para RSI/MACD.

    n=2000 a 1min de distancia = ~33 velas de 1h (suficiente para MACD-26 + RSI-14).
    """
    rng = np.random.default_rng(seed)
    prices = base_price + np.cumsum(rng.standard_normal(n) * 2000)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "amount": rng.uniform(0.01, 0.5, n),
    })


@pytest.fixture
def mock_ingestor():
    ingestor = MagicMock(spec=BitsoDataIngestor)
    ingestor.fetch_trades_batch = AsyncMock()
    return ingestor


@pytest.fixture
def features_pipeline():
    return TechnicalFeaturesPipeline.create_default()


@pytest.fixture
def signal_generator():
    return SignalGenerator()


@pytest.fixture
def mock_dispatcher():
    dispatcher = MagicMock(spec=NotificationDispatcher)
    dispatcher.dispatch = AsyncMock(return_value={"telegram": True})
    return dispatcher


@pytest.fixture
def monitor(mock_ingestor, features_pipeline, signal_generator, mock_dispatcher):
    return MarketMonitor(
        ingestor=mock_ingestor,
        features_pipeline=features_pipeline,
        signal_generator=signal_generator,
        dispatcher=mock_dispatcher,
        config=MonitorConfig(
            books=["btc_mxn", "eth_mxn"],
            timeframe=TimeFrame.H1,
            trades_per_book=200,
            interval_seconds=1.0,
            min_confidence=0.60,
        ),
    )


class TestScanOnce:
    async def test_returns_signal_per_book(self, monitor, mock_ingestor):
        mock_ingestor.fetch_trades_batch.return_value = _make_trades_df(200)
        results = await monitor.scan_once()

        assert "btc_mxn" in results
        assert "eth_mxn" in results
        assert all(isinstance(s, RuleSignal) for s in results.values())

    async def test_handles_empty_trades(self, monitor, mock_ingestor):
        mock_ingestor.fetch_trades_batch.return_value = pd.DataFrame()
        results = await monitor.scan_once()

        assert results["btc_mxn"].action == SignalAction.HOLD
        assert "Sin trades" in results["btc_mxn"].reason

    async def test_handles_insufficient_candles(self, monitor, mock_ingestor):
        mock_ingestor.fetch_trades_batch.return_value = _make_trades_df(10, seed=42)
        results = await monitor.scan_once()

        assert results["btc_mxn"].action == SignalAction.HOLD
        assert "insuficientes" in results["btc_mxn"].reason.lower()

    async def test_handles_exception_gracefully(self, monitor, mock_ingestor):
        mock_ingestor.fetch_trades_batch.side_effect = RuntimeError("Network error")
        results = await monitor.scan_once()

        assert results["btc_mxn"].action == SignalAction.HOLD
        assert "Error" in results["btc_mxn"].reason

    async def test_stores_last_signals(self, monitor, mock_ingestor):
        mock_ingestor.fetch_trades_batch.return_value = _make_trades_df(2000)
        await monitor.scan_once()

        assert "btc_mxn" in monitor.last_signals
        assert "eth_mxn" in monitor.last_signals


class TestNotificationDispatch:
    async def test_buy_signal_triggers_notification(self, mock_ingestor, features_pipeline, mock_dispatcher):
        # Mock signal_generator to always return BUY
        mock_signal_gen = MagicMock(spec=SignalGenerator)
        mock_signal_gen.generate.return_value = RuleSignal(
            action=SignalAction.BUY,
            confidence=Decimal("0.85"),
            rsi=22.0,
            macd_diff=150.0,
            reason="RSI sobreventa (22.0) + MACD cruce alcista",
        )

        monitor = MarketMonitor(
            ingestor=mock_ingestor,
            features_pipeline=features_pipeline,
            signal_generator=mock_signal_gen,
            dispatcher=mock_dispatcher,
            config=MonitorConfig(books=["btc_mxn"], min_confidence=0.60),
        )
        mock_ingestor.fetch_trades_batch.return_value = _make_trades_df(2000)

        await monitor.scan_once()

        assert mock_dispatcher.dispatch.called
        call_args = mock_dispatcher.dispatch.call_args[0][0]
        assert "BUY" in call_args.body
        assert "btc_mxn" in call_args.body

    async def test_sell_signal_triggers_notification(self, mock_ingestor, features_pipeline, mock_dispatcher):
        mock_signal_gen = MagicMock(spec=SignalGenerator)
        mock_signal_gen.generate.return_value = RuleSignal(
            action=SignalAction.SELL,
            confidence=Decimal("0.80"),
            rsi=78.0,
            macd_diff=-200.0,
            reason="RSI sobrecompra (78.0) + MACD cruce bajista",
        )

        monitor = MarketMonitor(
            ingestor=mock_ingestor,
            features_pipeline=features_pipeline,
            signal_generator=mock_signal_gen,
            dispatcher=mock_dispatcher,
            config=MonitorConfig(books=["btc_mxn"], min_confidence=0.60),
        )
        mock_ingestor.fetch_trades_batch.return_value = _make_trades_df(2000)

        await monitor.scan_once()

        assert mock_dispatcher.dispatch.called
        call_args = mock_dispatcher.dispatch.call_args[0][0]
        assert "SELL" in call_args.body

    async def test_hold_does_not_trigger_notification(self, monitor, mock_ingestor, mock_dispatcher):
        # Generate stable prices that should produce HOLD
        n = 2000
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        trades_df = pd.DataFrame({
            "timestamp": timestamps,
            "price": [500_000.0 + i * 10 for i in range(n)],
            "amount": [0.1] * n,
        })
        mock_ingestor.fetch_trades_batch.return_value = trades_df

        await monitor.scan_once()

        # No notification should be sent for HOLD signals
        mock_dispatcher.dispatch.assert_not_called()

    async def test_low_confidence_does_not_trigger_notification(self, mock_ingestor, features_pipeline, signal_generator, mock_dispatcher):
        monitor = MarketMonitor(
            ingestor=mock_ingestor,
            features_pipeline=features_pipeline,
            signal_generator=signal_generator,
            dispatcher=mock_dispatcher,
            config=MonitorConfig(
                books=["btc_mxn"],
                min_confidence=0.99,  # impossibly high
            ),
        )
        mock_ingestor.fetch_trades_batch.return_value = _make_trades_df(2000, seed=99)
        await monitor.scan_once()

        mock_dispatcher.dispatch.assert_not_called()


class TestPositionFiltering:
    """Verifica que el monitor filtra señales según posiciones abiertas."""

    def _make_buy_monitor(self, mock_ingestor, features_pipeline, mock_dispatcher, history_service):
        mock_signal_gen = MagicMock(spec=SignalGenerator)
        mock_signal_gen.generate.return_value = RuleSignal(
            action=SignalAction.BUY,
            confidence=Decimal("0.85"),
            rsi=22.0,
            macd_diff=150.0,
            reason="RSI sobreventa + MACD cruce alcista",
        )
        monitor = MarketMonitor(
            ingestor=mock_ingestor,
            features_pipeline=features_pipeline,
            signal_generator=mock_signal_gen,
            dispatcher=mock_dispatcher,
            config=MonitorConfig(books=["btc_mxn"], min_confidence=0.60),
            history_service=history_service,
        )
        mock_ingestor.fetch_trades_batch.return_value = _make_trades_df(2000)
        return monitor

    def _make_sell_monitor(self, mock_ingestor, features_pipeline, mock_dispatcher, history_service):
        mock_signal_gen = MagicMock(spec=SignalGenerator)
        mock_signal_gen.generate.return_value = RuleSignal(
            action=SignalAction.SELL,
            confidence=Decimal("0.80"),
            rsi=78.0,
            macd_diff=-200.0,
            reason="RSI sobrecompra + MACD cruce bajista",
        )
        monitor = MarketMonitor(
            ingestor=mock_ingestor,
            features_pipeline=features_pipeline,
            signal_generator=mock_signal_gen,
            dispatcher=mock_dispatcher,
            config=MonitorConfig(books=["btc_mxn"], min_confidence=0.60),
            history_service=history_service,
        )
        mock_ingestor.fetch_trades_batch.return_value = _make_trades_df(2000)
        return monitor

    async def test_buy_notified_when_no_position(
        self, mock_ingestor, features_pipeline, mock_dispatcher
    ):
        mock_history = MagicMock()
        mock_history.get_open_position.return_value = None
        mock_history.save_signal = MagicMock()

        monitor = self._make_buy_monitor(mock_ingestor, features_pipeline, mock_dispatcher, mock_history)
        await monitor.scan_once()

        assert mock_dispatcher.dispatch.called

    async def test_buy_skipped_when_position_open(
        self, mock_ingestor, features_pipeline, mock_dispatcher
    ):
        mock_position = MagicMock()
        mock_history = MagicMock()
        mock_history.get_open_position.return_value = mock_position
        mock_history.save_signal = MagicMock()

        monitor = self._make_buy_monitor(mock_ingestor, features_pipeline, mock_dispatcher, mock_history)
        await monitor.scan_once()

        mock_dispatcher.dispatch.assert_not_called()

    async def test_sell_notified_when_position_open(
        self, mock_ingestor, features_pipeline, mock_dispatcher
    ):
        mock_position = MagicMock()
        mock_position.avg_entry_price = Decimal(500000)
        mock_position.quantity = Decimal("0.01")
        mock_history = MagicMock()
        mock_history.get_open_position.return_value = mock_position
        mock_history.calc_unrealized_pnl.return_value = Decimal(200)
        mock_history.save_signal = MagicMock()

        monitor = self._make_sell_monitor(mock_ingestor, features_pipeline, mock_dispatcher, mock_history)
        await monitor.scan_once()

        assert mock_dispatcher.dispatch.called
        call_args = mock_dispatcher.dispatch.call_args[0][0]
        assert "SELL" in call_args.body
        assert "posición" in call_args.body.lower() or "Entrada" in call_args.body

    async def test_sell_skipped_when_no_position(
        self, mock_ingestor, features_pipeline, mock_dispatcher
    ):
        mock_history = MagicMock()
        mock_history.get_open_position.return_value = None
        mock_history.save_signal = MagicMock()

        monitor = self._make_sell_monitor(mock_ingestor, features_pipeline, mock_dispatcher, mock_history)
        await monitor.scan_once()

        mock_dispatcher.dispatch.assert_not_called()

    async def test_signal_saved_to_history(
        self, mock_ingestor, features_pipeline, mock_dispatcher
    ):
        mock_history = MagicMock()
        mock_history.get_open_position.return_value = None
        mock_history.save_signal = MagicMock()

        monitor = self._make_buy_monitor(mock_ingestor, features_pipeline, mock_dispatcher, mock_history)
        await monitor.scan_once()

        mock_history.save_signal.assert_called_once()
        saved_signal = mock_history.save_signal.call_args[0][1]
        assert saved_signal.action == SignalAction.BUY


class TestRunLoop:
    async def test_run_stops_with_stop(self, monitor, mock_ingestor):
        mock_ingestor.fetch_trades_batch.return_value = _make_trades_df(200)

        # Start run in background, then stop after a short delay
        import asyncio

        task = asyncio.create_task(monitor.run())
        await asyncio.sleep(0.1)
        monitor.stop()
        await asyncio.sleep(0.2)  # let it exit
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert not monitor._running
