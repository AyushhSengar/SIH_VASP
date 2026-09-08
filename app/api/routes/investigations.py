"""
MACRO MILESTONE 6 — investigation routes.

Every route is a thin adapter: parse request -> call
InvestigationService or InvestigationRepository -> shape the response.
No blockchain-intelligence logic lives here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_investigation_service, get_repository
from app.api.schemas import (
    InvestigationCreateRequest,
    InvestigationCreateResponse,
    InvestigationSummaryResponse,
    M4EvidenceSummary,
    M5MLSummary,
)
from app.attribution.models import AttributionResult
from app.db.repository import InvestigationRepository
from app.investigation.errors import InvestigationNotFoundError
from app.investigation.service import InvestigationService
from app.ml.models import MLPrediction

router = APIRouter(prefix="/investigations", tags=["investigations"])


@router.post("", status_code=201, response_model=InvestigationCreateResponse)
async def create_investigation(
    body: InvestigationCreateRequest,
    service: InvestigationService = Depends(get_investigation_service),
) -> InvestigationCreateResponse:
    result = await service.run_investigation(
        wallet=body.wallet,
        chain=body.chain,
        max_hops=body.max_hops,
        max_paths=body.max_paths,
        use_demo_seed=bool(body.use_demo_seed),
        ml_seed=body.ml_seed,
    )

    return InvestigationCreateResponse(
        investigation_id=result.investigation_id,
        wallet=result.wallet,
        chain=result.chain,
        created_at=result.record.created_at,
        m4_evidence=M4EvidenceSummary(
            status=result.attribution.status.value,
            candidate_count=len(result.attribution.candidates),
            search_truncated=result.attribution.search_truncated,
        ),
        m5_ml_prediction=M5MLSummary(
            predicted_label=result.ml_prediction.predicted_label.value,
            training_data_type=result.ml_prediction.training_data_type,
            model_name=result.ml_prediction.model_name,
            model_version=result.ml_prediction.model_version,
        ),
        training_data_type=result.ml_prediction.training_data_type,
    )


@router.get("/{investigation_id}", response_model=InvestigationSummaryResponse)
def get_investigation(
    investigation_id: str,
    repository: InvestigationRepository = Depends(get_repository),
) -> InvestigationSummaryResponse:
    record = repository.get_investigation(investigation_id)
    if record is None:
        raise InvestigationNotFoundError(
            f"No investigation found with id '{investigation_id}'."
        )

    return InvestigationSummaryResponse(
        investigation_id=record.id,
        wallet=record.wallet,
        chain=record.chain,
        graph_path=record.graph_path,
        max_hops=record.max_hops,
        search_truncated=record.search_truncated,
        attribution_status=record.attribution_status,
        ml_predicted_label=record.ml_predicted_label,
        created_at=record.created_at,
    )


@router.get("/{investigation_id}/attribution", response_model=AttributionResult)
def get_attribution(
    investigation_id: str,
    repository: InvestigationRepository = Depends(get_repository),
) -> AttributionResult:
    investigation = repository.get_investigation(investigation_id)
    if investigation is None:
        raise InvestigationNotFoundError(
            f"No investigation found with id '{investigation_id}'."
        )

    attribution_record = repository.get_attribution(investigation_id)
    if attribution_record is None:
        raise InvestigationNotFoundError(
            f"No M4 attribution result found for investigation '{investigation_id}'."
        )

    # The COMPLETE, original AttributionResult — reconstructed verbatim
    # from what M4 produced, not re-derived or merged with ML fields.
    return AttributionResult.model_validate_json(attribution_record.attribution_json)


@router.get("/{investigation_id}/ml", response_model=MLPrediction)
def get_ml_prediction(
    investigation_id: str,
    repository: InvestigationRepository = Depends(get_repository),
) -> MLPrediction:
    investigation = repository.get_investigation(investigation_id)
    if investigation is None:
        raise InvestigationNotFoundError(
            f"No investigation found with id '{investigation_id}'."
        )

    ml_record = repository.get_ml_prediction(investigation_id)
    if ml_record is None:
        raise InvestigationNotFoundError(
            f"No M5 ML prediction found for investigation '{investigation_id}'."
        )

    # The COMPLETE, original MLPrediction — including training_data_type
    # "SYNTHETIC_DEMO" and the disclaimer, never stripped or summarized.
    return MLPrediction.model_validate_json(ml_record.ml_json)
