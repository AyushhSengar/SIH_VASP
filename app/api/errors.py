"""
MACRO MILESTONE 6 — error handling.

Maps app.investigation.errors exceptions to HTTP responses. This is the
ONLY place that translates a service-layer error into an HTTP status
code — routes just raise the plain exception and let this handle it.

Status codes (per the M6 spec):
    422 -> invalid wallet / unsupported chain
    404 -> investigation not found
    400 -> graph unavailable (GRAPH_NOT_FOUND)
    400 -> the investigation stopped before producing a result
            (INVESTIGATION_STOPPED)
    503 -> database unavailable
    503 -> live acquisition is not configured (PROVIDER_NOT_CONFIGURED)
    500 -> internal service failure / anything unanticipated

Every handler returns a small, fixed JSON body. None of them ever
include exception args from arbitrary/unexpected exceptions, a stack
trace, or environment variables — see the catch-all handler below.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.investigation.errors import (
    DatabaseUnavailableError,
    GraphNotFoundError,
    InternalServiceFailure,
    InvalidWalletError,
    InvestigationNotFoundError,
    UnsupportedChainError,
)
from app.investigation.pipeline import PipelineError
from app.investigation.runner import MissingProviderCredentialError

logger = logging.getLogger("app.api")


def _error_body(error: str, detail: str) -> dict[str, str]:
    return {"error": error, "detail": detail}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(InvalidWalletError)
    async def _invalid_wallet(request: Request, exc: InvalidWalletError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_body("INVALID_WALLET", str(exc)),
        )

    @app.exception_handler(UnsupportedChainError)
    async def _unsupported_chain(request: Request, exc: UnsupportedChainError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_body("UNSUPPORTED_CHAIN", str(exc)),
        )

    @app.exception_handler(InvestigationNotFoundError)
    async def _not_found(request: Request, exc: InvestigationNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=_error_body("INVESTIGATION_NOT_FOUND", str(exc)),
        )

    @app.exception_handler(GraphNotFoundError)
    async def _graph_not_found(request: Request, exc: GraphNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=_error_body("GRAPH_NOT_FOUND", str(exc)),
        )

    @app.exception_handler(MissingProviderCredentialError)
    async def _missing_credential(
        request: Request, exc: MissingProviderCredentialError
    ) -> JSONResponse:
        # The one PipelineError whose message must not travel: it names the
        # deployment's configuration. Starlette resolves a handler by walking
        # the exception's MRO, so this subclass handler wins over the
        # PipelineError one below regardless of registration order. The operator
        # needs the detail, so it goes to the server log; the client gets a
        # fixed string and, crucially, still gets an error rather than a result
        # computed from substituted data.
        logger.error("Live acquisition is not configured: %s", exc)
        return JSONResponse(
            status_code=503,
            content=_error_body(
                "PROVIDER_NOT_CONFIGURED",
                "Live blockchain acquisition is not available on this "
                "deployment, and no demo or synthetic data is substituted for "
                "it. Retry with a wallet whose real data is already cached, or "
                "ask the operator to configure a provider credential.",
            ),
        )

    @app.exception_handler(PipelineError)
    async def _pipeline_stopped(request: Request, exc: PipelineError) -> JSONResponse:
        # PipelineError messages are written by this project, are deterministic,
        # and state a data or input outcome ("the provider returned no activity
        # for X", "that is not a valid address"). Forwarding them is the point:
        # the alternative is a client that knows only that *something* failed
        # and cannot tell an empty address from a broken deployment. The one
        # message that would disclose configuration is handled above.
        logger.info("Investigation stopped for %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=400,
            content=_error_body("INVESTIGATION_STOPPED", str(exc)),
        )

    @app.exception_handler(DatabaseUnavailableError)
    async def _db_unavailable(request: Request, exc: DatabaseUnavailableError) -> JSONResponse:
        logger.error("Database unavailable: %s", exc)
        return JSONResponse(
            status_code=503,
            content=_error_body(
                "DATABASE_UNAVAILABLE", "The database is currently unavailable."
            ),
        )

    @app.exception_handler(InternalServiceFailure)
    async def _internal_failure(request: Request, exc: InternalServiceFailure) -> JSONResponse:
        logger.error("Internal service failure: %s", exc)
        return JSONResponse(
            status_code=500,
            content=_error_body(
                "INTERNAL_SERVICE_FAILURE", "An internal error occurred."
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Never leak the exception message, args, or a traceback to the
        # client — only ever a fixed, safe string. Full details go to the
        # server-side log only.
        logger.exception("Unhandled exception while serving %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=_error_body(
                "INTERNAL_SERVICE_FAILURE", "An unexpected internal error occurred."
            ),
        )
