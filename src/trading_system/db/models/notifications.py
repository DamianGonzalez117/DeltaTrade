"""Modelo de auditoría de alertas emitidas por el sistema."""
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from trading_system.core.enums import NotificationChannel, NotificationLevel
from trading_system.db.base import Base


class NotificationLog(Base):
    __tablename__ = "notifications_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    channel: Mapped[NotificationChannel] = mapped_column(String(16), nullable=False)
    level: Mapped[NotificationLevel] = mapped_column(String(16), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)

    related_entity_type: Mapped[str | None] = mapped_column(String(32), nullable=True)  # "order", "signal", etc.
    related_entity_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
