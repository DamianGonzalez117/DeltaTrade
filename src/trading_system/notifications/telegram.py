"""Notifier para Telegram Bot API.

DIP: ``httpx.AsyncClient`` se inyecta para tests con MockTransport.
"""
import json

import httpx
import structlog

from trading_system.core.enums import NotificationChannel
from trading_system.notifications.base import NotificationMessage, Notifier

logger = structlog.get_logger(__name__)

_LEVEL_EMOJI = {
    "info": "ℹ️",
    "warning": "⚠️",
    "critical": "🚨",
}


class TelegramNotifier(Notifier):
    """Envía notificaciones vía Telegram Bot API."""

    def __init__(
        self,
        *,
        bot_token: str,
        chat_id: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient()
            self._owns_client = True

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.TELEGRAM

    async def send(self, message: NotificationMessage) -> bool:
        emoji = _LEVEL_EMOJI.get(message.level.value, "ℹ️")
        text = f"{emoji} *{message.title}*\n{message.body}"
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = json.dumps({
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
        })
        try:
            response = await self._client.post(
                url,
                content=payload,
                headers={"Content-Type": "application/json"},
            )
            return response.status_code == 200
        except httpx.HTTPError as exc:
            logger.error("telegram.send_failed", error=str(exc))
            return False

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()