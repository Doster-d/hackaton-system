"""Инициализация базы данных и вспомогательные функции работы с сессиями."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from hackaton_system.config import get_settings
from hackaton_system.db.models import Base

# Настройки читаем один раз, чтобы не открывать .env в каждом модуле.
settings = get_settings()

# Engine и фабрика сессий создаются при импорте — приложение использует их повторно.
engine: Engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Создаёт таблицы базы данных, если они ещё не созданы."""

    if settings.database_url.startswith("sqlite:///"):
        Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    # DeclarativeBase знает про все модели, поэтому create_all создаёт таблицы целиком.
    Base.metadata.create_all(bind=engine)
    _ensure_person_optional_columns()


def _ensure_person_optional_columns() -> None:
    """Добавляет новые опциональные колонки в таблицу persons для SQLite."""

    if engine.dialect.name != "sqlite":
        return
    with engine.connect() as conn:
        rows = conn.exec_driver_sql("PRAGMA table_info(persons)").fetchall()
        column_names = {row[1] for row in rows}
        if "person_type_conf" not in column_names:
            conn.exec_driver_sql(
                "ALTER TABLE persons ADD COLUMN person_type_conf FLOAT DEFAULT 0.0"
            )
        if "reid_descriptor" not in column_names:
            conn.exec_driver_sql(
                "ALTER TABLE persons ADD COLUMN reid_descriptor TEXT"
            )


@contextmanager
def get_session() -> Iterator[Session]:
    """Контекстный менеджер, обеспечивающий транзакционную сессию."""

    session = SessionLocal()
    try:
        # Возвращаем сессию вызывающему коду; commit/rollback управляются здесь.
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
