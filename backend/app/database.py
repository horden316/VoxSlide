from collections.abc import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from .config import get_settings


settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from . import models

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def _add_missing_columns() -> None:
    # create_all does not add new columns to tables that already exist.
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "projects" in inspector.get_table_names():
        columns = {column["name"] for column in inspector.get_columns("projects")}
        if "glossary" not in columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE projects ADD COLUMN glossary TEXT NOT NULL DEFAULT '[]'"))
