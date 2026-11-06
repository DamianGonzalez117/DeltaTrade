"""TDD — Tests de la delegación MarketMonitor → TradingAssistant.

El monitor con asistente inyectado le cede las decisiones de compra/venta
en vez de notificar plano. Se mockean ingestor/dispatcher/assistant.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd

from trading_system.core.enums import SignalAction, TimeFrame
from trading_system.features.pipeline import TechnicalFeaturesPipeline
from trading_system.services.monitor import MarketMonitor, MonitorConfig
from trading_system.services.signal_generator import RuleSignal, SignalGenerator


def _signal(action: SignalAction, confidence: str = "0.80") -> RuleSignal:
    return RuleSignal(
        action=action,
        confidence=Decimal(confidence),
        rsi=25.0 if action == SignalAction.BUY else 75.0,
        macd_diff=10.0 if action == SignalAction.BUY else -10.0,
        reason="test",
    )


def _position(entry: float = 500_000.0, qty: str = "0.01"):
    pos = MagicMock()
    pos.avg_entry_price = Decimal(str(entry))
    pos.quantity = Decimal(qty)
    return pos


def _make_monitor(assistant: MagicMock, history: MagicMock) -> MarketMonitor:
    ingestor = MagicMock()
    ingestor.fetch_trades_batch = AsyncMock()
    dispatcher = MagicMock()
    dispatcher.dispatch = AsyncMock()
    return MarketMonitor(
        ingestor=ingestor,
        features_pipeline=TechnicalFeaturesPipeline.create_default(),
        signal_generator=SignalGenerator(),
        dispatcher=dispatcher,
        config=MonitorConfig(
            books=["btc_mxn"],
            timeframe=TimeFrame.H1,
            min_confidence=0.60,
            profit_target_pct=0.05,
            stop_loss_pct=0.02,
        ),
        history_service=history,
        assistant=assistant,
    )


def _make_assistant_mock() -> MagicMock:
    assistant = MagicMock()
    assistant.on_buy_opportunity = AsyncMock()
    assistant.on_sell_recommendation = AsyncMock()
    return assistant


class TestSignalDelegation:
    async def test_buy_signal_goes_to_assistant_not_dispatcher(self):
        assistant = _make_assistant_mock()
        history = MagicMock()
        history.get_open_position = MagicMock(return_value=None)
        monitor = _make_monitor(assistant, history)

        await monitor._maybe_notify("btc_mxn", _signal(SignalAction.BUY), 500_000.0)

        assistant.on_buy_opportunity.assert_awaited_once_with(
            "btc_mxn", _signal(SignalAction.BUY), 500_000.0
        )
        monitor._dispatcher.dispatch.assert_not_awaited()

    async def test_sell_signal_with_position_asks_assistant(self):
        assistant = _make_assistant_mock()
        history = MagicMock()
        history.get_open_position = MagicMock(return_value=_position())
        monitor = _make_monitor(assistant, history)

        await monitor._maybe_notify("btc_mxn", _signal(SignalAction.SELL), 520_000.0)

        assistant.on_sell_recommendation.assert_awaited_once()
        call = assistant.on_sell_recommendation.await_args
        assert call.args[0] == "btc_mxn"
        assert call.args[1] == "signal"
        assert call.args[2] == 520_000.0
        # pnl_pct = (520000 - 500000) / 500000 = 0.04
        assert call.args[3] == 0.04

    async def test_sell_signal_without_position_skipped(self):
        assistant = _make_assistant_mock()
        history = MagicMock()
        history.get_open_position = MagicMock(return_value=None)
        monitor = _make_monitor(assistant, history)

        await monitor._maybe_notify("btc_mxn", _signal(SignalAction.SELL), 520_000.0)

        assistant.on_sell_recommendation.assert_not_awaited()

    async def test_without_assistant_falls_back_to_dispatcher(self):
        assistant = None
        history = MagicMock()
        history.get_open_position = MagicMock(return_value=None)
        monitor = _make_monitor(assistant, history)

        await monitor._maybe_notify("btc_mxn", _signal(SignalAction.BUY), 500_000.0)

        monitor._dispatcher.dispatch.assert_awaited_once()


class TestThresholdDelegation:
    async def test_profit_target_cross_asks_assistant_once(self):
        assistant = _make_assistant_mock()
        history = MagicMock()
        history.get_open_position = MagicMock(return_value=_position(entry=500_000.0))
        history.calc_unrealized_pnl = MagicMock(return_value=Decimal(250))
        monitor = _make_monitor(assistant, history)

        # +5.2% sobre entrada de 500k → cruza objetivo
        await monitor._check_position_thresholds("btc_mxn", 526_000.0)
        await monitor._check_position_thresholds("btc_mxn", 527_000.0)  # re-armo anti-spam

        assistant.on_sell_recommendation.assert_awaited_once_with(
            "btc_mxn", "profit", 526_000.0, 0.052
        )
        monitor._dispatcher.dispatch.assert_not_awaited()

    async def test_stop_loss_cross_asks_assistant(self):
        assistant = _make_assistant_mock()
        history = MagicMock()
        history.get_open_position = MagicMock(return_value=_position(entry=500_000.0))
        history.calc_unrealized_pnl = MagicMock(return_value=Decimal(-150))
        monitor = _make_monitor(assistant, history)

        # −3% sobre entrada → cruza stop loss de 2%
        await monitor._check_position_thresholds("btc_mxn", 485_000.0)

        assistant.on_sell_recommendation.assert_awaited_once_with(
            "btc_mxn", "loss", 485_000.0, -0.03
        )

    async def test_neutral_zone_does_not_ask(self):
        assistant = _make_assistant_mock()
        history = MagicMock()
        history.get_open_position = MagicMock(return_value=_position(entry=500_000.0))
        monitor = _make_monitor(assistant, history)

        await monitor._check_position_thresholds("btc_mxn", 505_000.0)

        assistant.on_sell_recommendation.assert_not_awaited()


class TestOhlcvExposure:
    async def test_scan_stores_last_ohlcv_and_prices(self):
        assistant = _make_assistant_mock()
        history = MagicMock()
        history.get_open_position = MagicMock(return_value=None)
        history.save_signal = MagicMock()
        monitor = _make_monitor(assistant, history)

        rng = np.random.default_rng(42)
        n = 2000
        prices = 500_000 + np.cumsum(rng.standard_normal(n) * 2000)
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        trades = pd.DataFrame({
            "timestamp": timestamps,
            "price": prices,
            "amount": rng.uniform(0.01, 0.5, n),
        })
        monitor._ingestor.fetch_trades_batch = AsyncMock(return_value=trades)

        await monitor._scan_book("btc_mxn")

        ohlcv = monitor.get_last_ohlcv("btc_mxn")
        assert ohlcv is not None and not ohlcv.empty
        assert "close" in ohlcv.columns
        assert monitor.last_prices["btc_mxn"] == float(ohlcv["close"].iloc[-1])

    def test_timeframe_minutes(self):
        assistant = _make_assistant_mock()
        monitor = _make_monitor(assistant, MagicMock())
        assert monitor.timeframe_minutes == 60  # H1

    def test_config_profit_target_pct(self):
        assistant = _make_assistant_mock()
        monitor = _make_monitor(assistant, MagicMock())
        assert monitor.config_profit_target_pct == 0.05
