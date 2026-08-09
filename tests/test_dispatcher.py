"""TDD — Tests for NotificationDispatcher (RED phase)."""
from unittest.mock import AsyncMock

from trading_system.core.enums import NotificationChannel, NotificationLevel
from trading_system.notifications.base import NotificationMessage
from trading_system.notifications.dispatcher import NotificationDispatcher
from trading_system.notifications.log_notifier import LogNotifier


def _make_message(level=NotificationLevel.INFO) -> NotificationMessage:
    return NotificationMessage(
        level=level,
        title="Test",
        body="Test body",
        channel=NotificationChannel.LOG,
    )


class TestDispatcherDispatch:
    async def test_dispatch_to_all_notifiers(self):
        notifier1 = AsyncMock()
        notifier1.channel = NotificationChannel.LOG
        notifier1.send = AsyncMock(return_value=True)
        notifier2 = AsyncMock()
        notifier2.channel = NotificationChannel.TELEGRAM
        notifier2.send = AsyncMock(return_value=True)

        dispatcher = NotificationDispatcher(notifiers=[notifier1, notifier2])
        results = await dispatcher.dispatch(_make_message())

        assert results["log"] is True
        assert results["telegram"] is True
        notifier1.send.assert_awaited_once()
        notifier2.send.assert_awaited_once()

    async def test_dispatch_collects_failures(self):
        notifier1 = AsyncMock()
        notifier1.channel = NotificationChannel.LOG
        notifier1.send = AsyncMock(return_value=True)
        notifier2 = AsyncMock()
        notifier2.channel = NotificationChannel.TELEGRAM
        notifier2.send = AsyncMock(return_value=False)

        dispatcher = NotificationDispatcher(notifiers=[notifier1, notifier2])
        results = await dispatcher.dispatch(_make_message())

        assert results["log"] is True
        assert results["telegram"] is False

    async def test_dispatch_handles_exception_gracefully(self):
        notifier1 = AsyncMock()
        notifier1.channel = NotificationChannel.TELEGRAM
        notifier1.send = AsyncMock(side_effect=RuntimeError("timeout"))

        dispatcher = NotificationDispatcher(notifiers=[notifier1])
        results = await dispatcher.dispatch(_make_message())

        assert results["telegram"] is False

    async def test_empty_dispatcher_returns_empty(self):
        dispatcher = NotificationDispatcher()
        results = await dispatcher.dispatch(_make_message())
        assert results == {}


class TestAddNotifier:
    async def test_add_notifier_appends(self):
        dispatcher = NotificationDispatcher()
        assert len(dispatcher.notifiers) == 0

        notifier = AsyncMock()
        notifier.channel = NotificationChannel.LOG
        dispatcher.add_notifier(notifier)

        assert len(dispatcher.notifiers) == 1


class TestDispatchCriticalOnly:
    async def test_info_message_skipped(self):
        notifier = AsyncMock()
        notifier.channel = NotificationChannel.TELEGRAM
        notifier.send = AsyncMock(return_value=True)

        dispatcher = NotificationDispatcher(notifiers=[notifier])
        results = await dispatcher.dispatch_critical_only(
            _make_message(level=NotificationLevel.INFO)
        )

        assert results == {}
        notifier.send.assert_not_awaited()

    async def test_critical_message_sent(self):
        notifier = AsyncMock()
        notifier.channel = NotificationChannel.TELEGRAM
        notifier.send = AsyncMock(return_value=True)

        dispatcher = NotificationDispatcher(notifiers=[notifier])
        results = await dispatcher.dispatch_critical_only(
            _make_message(level=NotificationLevel.CRITICAL)
        )

        assert results["telegram"] is True
        notifier.send.assert_awaited_once()

    async def test_warning_message_sent(self):
        notifier = AsyncMock()
        notifier.channel = NotificationChannel.TELEGRAM
        notifier.send = AsyncMock(return_value=True)

        dispatcher = NotificationDispatcher(notifiers=[notifier])
        results = await dispatcher.dispatch_critical_only(
            _make_message(level=NotificationLevel.WARNING)
        )

        assert results["telegram"] is True


class TestRealLogNotifierIntegration:
    async def test_dispatcher_with_log_notifier(self):
        dispatcher = NotificationDispatcher(notifiers=[LogNotifier()])
        results = await dispatcher.dispatch(_make_message())
        assert results["log"] is True