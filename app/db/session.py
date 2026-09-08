"""
MACRO MILESTONE 6 — engine/session construction.

Deliberately NOT a single module-level global engine: every helper here
takes `database_url` explicitly, so the API app, the CLI/tests, and the
test suite's in-memory SQLite database can each build their own
engine/session pair without monkeypatching or import-order tricks.
`get_engine` is cached per URL so repeated calls (e.g. once per request
dependency) reuse the same connection pool instead of opening a new one
every time.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base


@lru_cache(maxsize=8)
def get_engine(database_url: str) -> Engine:
    connect_args: dict[str, object] = {}
    if database_url.startswith("sqlite"):
        # Needed because FastAPI can hand the same connection to different
        # threads across requests; SQLite otherwise refuses cross-thread use.
        connect_args["check_same_thread"] = False
    return create_engine(database_url, connect_args=connect_args, future=True)


def get_sessionmaker(database_url: str) -> sessionmaker[Session]:
    engine = get_engine(database_url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db(database_url: str) -> None:
    """Creates all tables if they don't already exist. Safe to call on
    every app startup — idempotent, and works identically against a fresh
    SQLite file or an already-provisioned PostgreSQL database."""
    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)


def check_database_connection(database_url: str) -> bool:
    """Used by GET /health. Returns True if a trivial query succeeds,
    False on any database-layer failure — never raises, and never
    includes the DATABASE_URL (which may contain credentials) in any
    caller-visible output."""
    try:
        engine = get_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False
