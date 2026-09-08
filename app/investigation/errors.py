"""
MACRO MILESTONE 6 — service-layer error types.

These are deliberately plain Python exceptions with no FastAPI/HTTP
knowledge — app/investigation/ and app/db/ stay usable outside a web
context (e.g. from a future CLI). app/api/errors.py is the only place
that maps these to HTTP status codes.
"""

from __future__ import annotations


class InvestigationServiceError(Exception):
    """Base class for all investigation-service failures."""


class InvalidWalletError(InvestigationServiceError):
    """The wallet address failed provider-level address validation."""


class UnsupportedChainError(InvestigationServiceError):
    """The requested chain isn't one this deployment supports."""


class GraphNotFoundError(InvestigationServiceError):
    """No graph could be obtained/loaded for this wallet — e.g. the
    provider returned no transaction activity at all, so M2's
    build_graph() has nothing to build from, or an explicitly requested
    cached graph file does not exist on disk."""


class InvestigationNotFoundError(InvestigationServiceError):
    """No investigation exists with the given id."""


class DatabaseUnavailableError(InvestigationServiceError):
    """The database could not be reached or a query failed at the
    database layer. Never carries the original driver exception's
    message forward to the API response (that could leak connection
    details) — callers should present a generic, safe message."""


class InternalServiceFailure(InvestigationServiceError):
    """Catch-all for genuine internal errors (e.g. malformed seed data)
    that are not the caller's fault and are not a database problem."""
