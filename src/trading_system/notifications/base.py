"""Contrato base para notificadores.

Cada notifier implementa ``Notifier.send(message) -> bool``.
``NotificationMessage`` es el DTO compartido.

OCP: nuevo canal = nueva clase que implementa ``Notifier``.
DIP: ``NotificationDispatcher`` depende de ``Notifier`` (ABC).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from trading_system.core.enums import NotificationChannel, NotificationLevel


@dataclass(frozen=True, slots=True)
class NotificationMessage:
    """Mensaje de notificación normalizado."""

    level: NotificationLevel
    title: str
    body: str
    channel: NotificationChannel = NotificationChannel.LOG
    related_entity_type: str | None = None
    related_entity_id: int | None = None


class Notifier(ABC):
    """Interfaz para enviar notificaciones a un canal específico."""

    @property
    @abstractmethod
    def channel(self) -> NotificationChannel:
        ...

    @abstractmethod
    async def send(self, message: NotificationMessage) -> bool:
        """Envía el mensaje. Retorna True si tuvo éxito, False si falló."""
        ...