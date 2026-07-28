"""
Gestión del engine y las sesiones de base de datos.

Uso típico en un servicio:

    from trading_system.db.session import get_session

    with get_session() as session:
        session.add(algo)
        session.commit()
"""
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from trading_system.config import settings

engine = create_engine(
    settings.database_url,
    echo=settings.db_echo,
    pool_pre_ping=True,   # evita conexiones "muertas" en procesos de larga duración (bots 24/7)
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@contextmanager
def get_session() -> Iterator[Session]:
    """Context manager que garantiza commit/rollback/close correctos."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
