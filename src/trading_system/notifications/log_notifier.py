"""Notifier que escribe a structlog (canal de log interno).

SRP: solo loggea. No toca HTTP ni DB.
"""
import structlog

from trading_system.core.enums import NotificationChannel
from trading_system.notifications.base import NotificationMessage, Notifier

logger = structlog.get_logger(__name__)

_LEVEL_MAP = {
    "info": "info",
    "warning": "warning",
    "critical": "critical",
}


class LogNotifier(Notifier):
    """Notifier que envía mensajes a structlog."""

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.LOG

    async def send(self, message: NotificationMessage) -> bool:
        log_fn = getattr(logger, _LEVEL_MAP.get(message.level.value, "info"))
        log_fn(
            "notification.sent",
            title=message.title,
            body=message.body,
            level=message.level.value,
            entity_type=message.related_entity_type,
            entity_id=message.related_entity_id,
        )
        return True