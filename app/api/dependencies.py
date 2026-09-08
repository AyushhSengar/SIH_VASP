"""
MACRO MILESTONE 6 — FastAPI dependencies.

`get_provider_factory` is the seam tests use to avoid live Etherscan
calls: override it with `app.dependency_overrides[get_provider_factory]`
to inject a fake BlockchainProvider, per the M6 test requirements.
FastAPI resolves overrides transitively, so overriding it here also
takes effect inside get_investigation_service below without any extra
wiring.
"""

from __future__ import annotations

from typing import Iterator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.repository import InvestigationRepository
from app.investigation.service import (
    InvestigationService,
    ProviderFactory,
    _default_provider_factory,
)


def get_app_settings() -> Settings:
    return get_settings()


def get_db_session(request: Request) -> Iterator[Session]:
    session_factory = request.app.state.session_factory
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def get_repository(
    db: Session = Depends(get_db_session),
) -> InvestigationRepository:
    return InvestigationRepository(db)


def get_provider_factory() -> ProviderFactory:
    """Default provider factory: real EtherscanProvider. Override this in
    tests (`app.dependency_overrides[get_provider_factory] = ...`) to
    inject a fake/mock provider so tests never touch the network."""
    return _default_provider_factory


def get_investigation_service(
    settings: Settings = Depends(get_app_settings),
    repository: InvestigationRepository = Depends(get_repository),
    provider_factory: ProviderFactory = Depends(get_provider_factory),
) -> InvestigationService:
    return InvestigationService(
        settings=settings, repository=repository, provider_factory=provider_factory
    )
