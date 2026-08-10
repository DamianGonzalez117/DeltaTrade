"""
Dispatcher de notificaciones: enruta mensajes a múltiples notifiers.

Puede filtrar por nivel (ej. solo CRITICAL a Telegram) o enviar a todos.
DIP: depende de ``Notifier`` (ABC), no de implementaciones concretas.
"""
import asyncio

import structlog

from trading_system.core.enums import NotificationLevel
from trading_system.notifications.base import NotificationMessage, Notifier

logger = structlog.get_logger(__name__)


class NotificationDispatcher:
    """Envía mensajes a todos los notifiers registrados."""

    def __init__(self, notifiers: list[Notifier] | None = None) -> None:
        self._notifiers: list[Notifier] = list(notifiers or [])

    @property
    def notifiers(self) -> list[Notifier]:
        return list(self._notifiers)

    def add_notifier(self, notifier: Notifier) -> None:
        self._notifiers.append(notifier)

    async def dispatch(self, message: NotificationMessage) -> dict[str, bool]:
        """Envía el mensaje a todos los notifiers y retorna los resultados."""
        results: dict[str, bool] = {}
        tasks = []
        for n in self._notifiers:
            tasks.append(n.send(message))
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for notifier, response in zip(self._notifiers, responses):
            if isinstance(response, Exception):
                logger.error(
                    "dispatcher.notifier_failed",
                    channel=notifier.channel.value,
                    error=str(response),
                )
                results[notifier.channel.value] = False
            else:
                results[notifier.channel.value] = bool(response)
        return results

    async def dispatch_critical_only(self, message: NotificationMessage) -> dict[str, bool]:
        """Solo envía si el nivel es CRITICAL o WARNING."""
        if message.level not in (NotificationLevel.CRITICAL, NotificationLevel.WARNING):
            return {}
        return await self.dispatch(message)