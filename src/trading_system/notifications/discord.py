"""Notifier para Discord Webhooks.

DIP: ``httpx.AsyncClient`` se inyecta para tests con MockTransport.
"""
import json

import httpx
import structlog

from trading_system.core.enums import NotificationChannel, NotificationLevel
from trading_system.notifications.base import NotificationMessage, Notifier

logger = structlog.get_logger(__name__)

_LEVEL_COLOR = {
    NotificationLevel.INFO: 0x3498DB,
    NotificationLevel.WARNING: 0xF39C12,
    NotificationLevel.CRITICAL: 0xE74C3C,
}


class DiscordNotifier(Notifier):
    """Envía notificaciones vía webhook de Discord."""

    def __init__(
        self,
        *,
        webhook_url: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._webhook_url = webhook_url
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient()
            self._owns_client = True

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.DISCORD

    async def send(self, message: NotificationMessage) -> bool:
        color = _LEVEL_COLOR.get(message.level, 0x3498DB)
        payload = json.dumps({
            "embeds": [
                {
                    "title": message.title,
                    "description": message.body,
                    "color": color,
                }
            ]
        })
        try:
            response = await self._client.post(
                self._webhook_url,
                content=payload,
                headers={"Content-Type": "application/json"},
            )
            return response.status_code in (200, 204)
        except httpx.HTTPError as exc:
            logger.error("discord.send_failed", error=str(exc))
            return False

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()