"""GET /health — never requires auth, never touches blockchain logic."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.db.session import check_database_connection

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> JSONResponse:
    settings = get_settings()
    database_url = getattr(request.app.state, "database_url", settings.database_url)

    if check_database_connection(database_url):
        return JSONResponse(status_code=200, content={"status": "ok", "database": "connected"})

    return JSONResponse(
        status_code=503,
        content={"status": "degraded", "database": "unavailable"},
    )
