"""Capa de notificaciones del sistema de trading.

Exporta:
  - ``Notifier`` / ``NotificationMessage``: interfaz y DTO.
  - ``LogNotifier``: notifier a structlog.
  - ``TelegramNotifier``: notifier a Telegram Bot API.
  - ``DiscordNotifier``: notifier a Discord webhook.
  - ``NotificationDispatcher``: enruta a múltiples notifiers.
"""
from trading_system.notifications.base import NotificationMessage, Notifier
from trading_system.notifications.discord import DiscordNotifier
from trading_system.notifications.dispatcher import NotificationDispatcher
from trading_system.notifications.log_notifier import LogNotifier
from trading_system.notifications.telegram import TelegramNotifier

__all__ = [
    "DiscordNotifier",
    "LogNotifier",
    "NotificationDispatcher",
    "NotificationMessage",
    "Notifier",
    "TelegramNotifier",
]