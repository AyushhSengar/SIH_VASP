"""
MACRO MILESTONE 6 tests — database-unavailable behavior.

Points app.state.session_factory at a database that cannot be opened
(a nonexistent directory) to simulate the database being unreachable,
without needing a real PostgreSQL server to take down.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

WALLET = "0x1111111111111111111111111111111111111111"
_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


@pytest.fixture()
def broken_client(tmp_path, monkeypatch):
    db_path = tmp_path / "ok.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    from app.api.main import app

    with TestClient(app) as test_client:
        # After startup (tables created against a working DB), swap the
        # session factory for one pointed at a database file whose parent
        # directory does not exist — every connection attempt will fail,
        # simulating "database unavailable" without touching a real server.
        broken_engine = create_engine(
            "sqlite:////this/directory/does/not/exist/broken.db",
            connect_args={"check_same_thread": False},
        )
        app.state.session_factory = sessionmaker(
            bind=broken_engine, autoflush=False, autocommit=False, future=True
        )
        app.state.database_url = "sqlite:////this/directory/does/not/exist/broken.db"

        yield test_client

    app.dependency_overrides.clear()


def test_health_with_unavailable_database_returns_503(broken_client: TestClient):
    resp = broken_client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["database"] == "unavailable"


def test_get_investigation_with_unavailable_database_returns_503(broken_client: TestClient):
    resp = broken_client.get("/investigations/some-id")
    assert resp.status_code == 503
    assert resp.json()["error"] == "DATABASE_UNAVAILABLE"
