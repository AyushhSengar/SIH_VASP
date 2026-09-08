"""
MACRO MILESTONE 6 — FastAPI application.

Run locally with:

    python -m uvicorn app.api.main:app --reload

Then check:
    GET http://127.0.0.1:8000/health
    GET http://127.0.0.1:8000/docs

Two investigation surfaces exist and they are deliberately separate:

    POST /investigations   — persists a record; its ML summary is the
                             Milestone-5 SYNTHETIC_DEMO classifier
    POST /analysis/brief   — the real-data pipeline `investigate.py` runs,
                             returning the same brief `--brief` prints

Neither is a newer version of the other. `/investigations` is the persistence
surface, `/analysis` is the analysis surface, and the ML behind them differs —
which is why the endpoint you call tells you which model answered.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.routes import analysis, health, investigations
from app.core.config import get_settings
from app.db.session import get_sessionmaker, init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    init_db(settings.database_url)
    app.state.database_url = settings.database_url
    app.state.session_factory = get_sessionmaker(settings.database_url)
    yield


app = FastAPI(
    title="Blockchain Intelligence API",
    description=(
        "Macro Milestone 6: persistence + investigation service on top of "
        "the M1-M5 blockchain-intelligence pipeline. M5 ML predictions under "
        "/investigations are always trained on SYNTHETIC_DEMO data and are "
        "never a substitute for the evidence-based M4 attribution result; "
        "/analysis/brief runs the real-data pipeline instead."
    ),
    version="6.0.0",
    lifespan=lifespan,
)

register_exception_handlers(app)

# The browser frontend is a separate origin in development (Vite on 5173), so
# without this every request from it fails preflight. The allowed origins are
# an explicit list from configuration, not "*": this API is unauthenticated and
# a wildcard would let any page the operator has open drive investigations
# against this deployment. Credentials are not allowed for the same reason —
# there is no session to send, and permitting them would be inviting one.
_origins = get_settings().cors_allow_origins
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

app.include_router(health.router)
app.include_router(investigations.router)
app.include_router(analysis.router)
