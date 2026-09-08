"""
MACRO MILESTONE 6 — repository layer.

The ONLY place, besides app/db/models.py and app/db/session.py, allowed
to import sqlalchemy. API routes and the investigation service talk to
this class, never to a Session directly, so raw queries don't spread
across the codebase.

Every method wraps sqlalchemy errors in DatabaseUnavailableError (see
app.investigation.errors) rather than letting sqlalchemy.exc types leak
into the API layer — the API layer should never need to know this
repository is backed by SQLAlchemy at all.
"""

from __future__ import annotations

import uuid

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.attribution.models import AttributionResult
from app.db.models import AttributionRecord, InvestigationRecord, MLPredictionRecord
from app.investigation.errors import DatabaseUnavailableError
from app.ml.models import MLPrediction


class InvestigationRepository:
    def __init__(self, db: Session):
        self._db = db

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------

    def create_investigation(
        self,
        *,
        investigation_id: str,
        wallet: str,
        chain: str,
        max_hops: int,
        max_paths: int,
        graph_path: str | None,
        search_truncated: bool,
        attribution_status: str,
        ml_predicted_label: str,
        training_data_type: str,
    ) -> InvestigationRecord:
        record = InvestigationRecord(
            id=investigation_id,
            wallet=wallet,
            chain=chain,
            max_hops=max_hops,
            max_paths=max_paths,
            graph_path=graph_path,
            search_truncated=search_truncated,
            attribution_status=attribution_status,
            ml_predicted_label=ml_predicted_label,
            training_data_type=training_data_type,
        )
        try:
            self._db.add(record)
            self._db.commit()
            self._db.refresh(record)
        except SQLAlchemyError as exc:
            self._db.rollback()
            raise DatabaseUnavailableError(
                "Database unavailable while persisting the investigation."
            ) from exc
        return record

    def create_attribution(
        self, *, investigation_id: str, attribution_result: AttributionResult
    ) -> AttributionRecord:
        record = AttributionRecord(
            id=uuid.uuid4().hex,
            investigation_id=investigation_id,
            wallet=attribution_result.wallet,
            status=attribution_result.status.value,
            attribution_json=attribution_result.model_dump_json(),
        )
        try:
            self._db.add(record)
            self._db.commit()
            self._db.refresh(record)
        except SQLAlchemyError as exc:
            self._db.rollback()
            raise DatabaseUnavailableError(
                "Database unavailable while persisting the M4 attribution result."
            ) from exc
        return record

    def create_ml_prediction(
        self, *, investigation_id: str, ml_prediction: MLPrediction
    ) -> MLPredictionRecord:
        record = MLPredictionRecord(
            id=uuid.uuid4().hex,
            investigation_id=investigation_id,
            wallet=ml_prediction.wallet,
            predicted_label=ml_prediction.predicted_label.value,
            training_data_type=ml_prediction.training_data_type,
            ml_json=ml_prediction.model_dump_json(),
        )
        try:
            self._db.add(record)
            self._db.commit()
            self._db.refresh(record)
        except SQLAlchemyError as exc:
            self._db.rollback()
            raise DatabaseUnavailableError(
                "Database unavailable while persisting the M5 ML prediction."
            ) from exc
        return record

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    def get_investigation(self, investigation_id: str) -> InvestigationRecord | None:
        try:
            return self._db.get(InvestigationRecord, investigation_id)
        except SQLAlchemyError as exc:
            raise DatabaseUnavailableError(
                "Database unavailable while reading the investigation."
            ) from exc

    def get_attribution(self, investigation_id: str) -> AttributionRecord | None:
        try:
            return (
                self._db.query(AttributionRecord)
                .filter(AttributionRecord.investigation_id == investigation_id)
                .one_or_none()
            )
        except SQLAlchemyError as exc:
            raise DatabaseUnavailableError(
                "Database unavailable while reading the M4 attribution result."
            ) from exc

    def get_ml_prediction(self, investigation_id: str) -> MLPredictionRecord | None:
        try:
            return (
                self._db.query(MLPredictionRecord)
                .filter(MLPredictionRecord.investigation_id == investigation_id)
                .one_or_none()
            )
        except SQLAlchemyError as exc:
            raise DatabaseUnavailableError(
                "Database unavailable while reading the M5 ML prediction."
            ) from exc
