"""
Base declarativa compartida por todos los modelos ORM.

Se define una convención de nombres explícita para constraints/índices para
que Alembic genere migraciones consistentes y para poder referenciar
constraints por nombre en migraciones futuras (ALTER/DROP) sin adivinar.
"""
from datetime import datetime, timezone

from sqlalchemy import MetaData, DateTime
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    """Mixin con columnas de auditoría temporal estándar (UTC)."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )
