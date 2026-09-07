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


def _make_ramp_trades(n: int = 2000, start: float = 100.0, end: float = 106.0) -> pd.DataFrame:
    """Rampa lineal con volumen constante: precio final determinista, señal HOLD."""
    prices = np.linspace(start, end, n)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "amount": [1.0] * n,
    })


def _make_position(entry: float, qty: float = 1.0):
    pos = MagicMock()
    pos.avg_entry_price = Decimal(str(entry))
    pos.quantity = Decimal(str(qty))
    return pos


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
        mock_position.avg_entry_price = Decimal(500_000)
        mock_position.quantity = Decimal("0.01")
        mock_history = MagicMock()
        mock_history.get_open_position.return_value = mock_position
        mock_history.save_signal = MagicMock()

        monitor = self._make_buy_monitor(mock_ingestor, features_pipeline, mock_dispatcher, mock_history)
        # Umbrales inalcanzables: este test verifica solo el filtrado de señal BUY
        monitor._config.profit_target_pct = 999.0
        monitor._config.stop_loss_pct = 999.0
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


class TestProfitThresholds:
    """Monitoreo de objetivos de ganancia y stop loss sobre posiciones abiertas."""

    def _make_monitor(
        self,
        mock_ingestor,
        features_pipeline,
        mock_dispatcher,
        history_service,
        trades_df,
    ):
        monitor = MarketMonitor(
            ingestor=mock_ingestor,
            features_pipeline=features_pipeline,
            signal_generator=SignalGenerator(),
            dispatcher=mock_dispatcher,
            config=MonitorConfig(
                books=["btc_mxn"],
                min_confidence=0.60,
                profit_target_pct=0.05,
                stop_loss_pct=0.02,
            ),
            history_service=history_service,
        )
        mock_ingestor.fetch_trades_batch.return_value = trades_df
        return monitor

    def _mock_history_with_position(self, entry: float):
        history = MagicMock()
        history.get_open_position.return_value = _make_position(entry)
        history.calc_unrealized_pnl.side_effect = lambda pos, price: (
            (price - pos.avg_entry_price) * pos.quantity
        )
        history.save_signal = MagicMock()
        return history

    async def test_profit_target_notified(
        self, mock_ingestor, features_pipeline, mock_dispatcher
    ):
        # Entrada 100, actual 106 → +6% ≥ 5% → VENDER
        history = self._mock_history_with_position(100.0)
        monitor = self._make_monitor(
            mock_ingestor, features_pipeline, mock_dispatcher, history,
            _make_ramp_trades(end=106.0),
        )
        await monitor.scan_once()

        assert mock_dispatcher.dispatch.called
        body = mock_dispatcher.dispatch.call_args[0][0].body
        assert "objetivo" in body.lower()
        assert "VENDER" in body

    async def test_stop_loss_notified(
        self, mock_ingestor, features_pipeline, mock_dispatcher
    ):
        # Entrada 112, actual 106 → -5.4% ≤ -2% → STOP LOSS
        history = self._mock_history_with_position(112.0)
        monitor = self._make_monitor(
            mock_ingestor, features_pipeline, mock_dispatcher, history,
            _make_ramp_trades(end=106.0),
        )
        await monitor.scan_once()

        assert mock_dispatcher.dispatch.called
        body = mock_dispatcher.dispatch.call_args[0][0].body
        assert "stop loss" in body.lower()

    async def test_no_notification_in_neutral_zone(
        self, mock_ingestor, features_pipeline, mock_dispatcher
    ):
        # Entrada 103, actual 106 → +2.9% (entre -2% y +5%) → nada
        history = self._mock_history_with_position(103.0)
        monitor = self._make_monitor(
            mock_ingestor, features_pipeline, mock_dispatcher, history,
            _make_ramp_trades(end=106.0),
        )
        await monitor.scan_once()

        mock_dispatcher.dispatch.assert_not_called()

    async def test_anti_spam_no_duplicate(
        self, mock_ingestor, features_pipeline, mock_dispatcher
    ):
        history = self._mock_history_with_position(100.0)
        monitor = self._make_monitor(
            mock_ingestor, features_pipeline, mock_dispatcher, history,
            _make_ramp_trades(end=106.0),
        )
        await monitor.scan_once()
        await monitor.scan_once()

        assert mock_dispatcher.dispatch.call_count == 1

    async def test_rearms_after_returning_to_neutral(
        self, mock_ingestor, features_pipeline, mock_dispatcher
    ):
        history = self._mock_history_with_position(100.0)
        monitor = self._make_monitor(
            mock_ingestor, features_pipeline, mock_dispatcher, history,
            _make_ramp_trades(end=106.0),
        )
        await monitor.scan_once()  # +6% → notifica

        mock_ingestor.fetch_trades_batch.return_value = _make_ramp_trades(end=102.0)
        await monitor.scan_once()  # +2% → neutral, rearma

        mock_ingestor.fetch_trades_batch.return_value = _make_ramp_trades(end=106.0)
        await monitor.scan_once()  # +6% → notifica de nuevo

        assert mock_dispatcher.dispatch.call_count == 2

    async def test_rearms_when_position_closed(
        self, mock_ingestor, features_pipeline, mock_dispatcher
    ):
        history = self._mock_history_with_position(100.0)
        monitor = self._make_monitor(
            mock_ingestor, features_pipeline, mock_dispatcher, history,
            _make_ramp_trades(end=106.0),
        )
        await monitor.scan_once()  # notifica objetivo

        history.get_open_position.return_value = None
        await monitor.scan_once()  # posición cerrada → rearma

        history.get_open_position.return_value = _make_position(100.0)
        await monitor.scan_once()  # nueva posición en ganancia → notifica

        assert mock_dispatcher.dispatch.call_count == 2

    async def test_no_threshold_check_without_history(
        self, mock_ingestor, features_pipeline, mock_dispatcher
    ):
        monitor = MarketMonitor(
            ingestor=mock_ingestor,
            features_pipeline=features_pipeline,
            signal_generator=SignalGenerator(),
            dispatcher=mock_dispatcher,
            config=MonitorConfig(books=["btc_mxn"], min_confidence=0.60),
        )
        mock_ingestor.fetch_trades_batch.return_value = _make_ramp_trades(end=106.0)
        await monitor.scan_once()

        mock_dispatcher.dispatch.assert_not_called()


class TestBuyNotificationEnrichment:
    """BUY incluye precio actual, objetivo de venta sugerido y stats."""

    def _make_enriched_buy_monitor(
        self, mock_ingestor, features_pipeline, mock_dispatcher, trade_analyzer
    ):
        mock_signal_gen = MagicMock(spec=SignalGenerator)
        mock_signal_gen.generate.return_value = RuleSignal(
            action=SignalAction.BUY,
            confidence=Decimal("0.85"),
            rsi=22.0,
            macd_diff=150.0,
            reason="Crecimiento: EMA9>EMA21 + volumen 2.0x",
        )
        mock_history = MagicMock()
        mock_history.get_open_position.return_value = None
        mock_history.save_signal = MagicMock()

        monitor = MarketMonitor(
            ingestor=mock_ingestor,
            features_pipeline=features_pipeline,
            signal_generator=mock_signal_gen,
            dispatcher=mock_dispatcher,
            config=MonitorConfig(books=["btc_mxn"], min_confidence=0.60, profit_target_pct=0.05),
            history_service=mock_history,
            trade_analyzer=trade_analyzer,
        )
        mock_ingestor.fetch_trades_batch.return_value = _make_ramp_trades(end=106.0)
        return monitor

    async def test_buy_includes_current_price_and_target(
        self, mock_ingestor, features_pipeline, mock_dispatcher
    ):
        analyzer = MagicMock()
        analyzer.suggest_profit_target_pct.return_value = 4.5
        analyzer.get_closed_stats.return_value = None

        monitor = self._make_enriched_buy_monitor(
            mock_ingestor, features_pipeline, mock_dispatcher, analyzer
        )
        await monitor.scan_once()

        body = mock_dispatcher.dispatch.call_args[0][0].body
        assert "106.00" in body  # precio actual (último close)
        assert "Objetivo de venta sugerido" in body
        assert "+4.5%" in body

    async def test_buy_includes_historical_stats(
        self, mock_ingestor, features_pipeline, mock_dispatcher
    ):
        from trading_system.services.trade_analyzer import TradeStats

        analyzer = MagicMock()
        analyzer.suggest_profit_target_pct.return_value = 4.5
        analyzer.get_closed_stats.return_value = TradeStats(
            n_trades=3, win_rate=2 / 3, avg_profit_pct=3.7, avg_hold_minutes=95.0
        )

        monitor = self._make_enriched_buy_monitor(
            mock_ingestor, features_pipeline, mock_dispatcher, analyzer
        )
        await monitor.scan_once()

        body = mock_dispatcher.dispatch.call_args[0][0].body
        assert "3 trades" in body
        assert "67%" in body  # win-rate
        assert "3.7%" in body  # avg profit


class TestSellUsesRealPrice:
    """La notificación SELL usa el precio real del último close, no el RSI."""

    async def test_sell_pnl_uses_last_close(
        self, mock_ingestor, features_pipeline, mock_dispatcher
    ):
        mock_signal_gen = MagicMock(spec=SignalGenerator)
        mock_signal_gen.generate.return_value = RuleSignal(
            action=SignalAction.SELL,
            confidence=Decimal("0.80"),
            rsi=78.0,
            macd_diff=-200.0,
            reason="RSI sobrecompra + MACD cruce bajista",
        )
        history = MagicMock()
        history.get_open_position.return_value = _make_position(100.0)
        history.calc_unrealized_pnl.side_effect = lambda pos, price: (
            (price - pos.avg_entry_price) * pos.quantity
        )
        history.save_signal = MagicMock()

        monitor = MarketMonitor(
            ingestor=mock_ingestor,
            features_pipeline=features_pipeline,
            signal_generator=mock_signal_gen,
            dispatcher=mock_dispatcher,
            # Umbrales inalcanzables: este test verifica solo el P&L del SELL
            config=MonitorConfig(
                books=["btc_mxn"],
                min_confidence=0.60,
                profit_target_pct=999.0,
                stop_loss_pct=999.0,
            ),
            history_service=history,
        )
        mock_ingestor.fetch_trades_batch.return_value = _make_ramp_trades(end=106.0)
        await monitor.scan_once()

        history.calc_unrealized_pnl.assert_called_once()
        used_price = history.calc_unrealized_pnl.call_args[0][1]
        assert float(used_price) == pytest.approx(106.0)  # NO 78 (RSI)


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
