"""
MACRO MILESTONE 6 — API request/response schemas.

`GET /investigations/{id}/attribution` and `GET /investigations/{id}/ml`
deliberately return app.attribution.models.AttributionResult and
app.ml.models.MLPrediction directly rather than a duplicated schema here
— that's what "return the COMPLETE M4 AttributionResult / M5
MLPrediction" means, and it guarantees the API can never drift from
what M4/M5 actually produced. Everything in this file is either a
request body or a deliberately-summarized response (POST create summary,
GET investigation summary, health) that must NOT contain the full
nested objects.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class InvestigationCreateRequest(BaseModel):
    wallet: str
    chain: str = "ethereum"
    max_hops: Optional[int] = Field(default=None, ge=1)
    max_paths: Optional[int] = Field(default=None, ge=1)
    use_demo_seed: Optional[bool] = False
    ml_seed: Optional[int] = None


class M4EvidenceSummary(BaseModel):
    """Summary only — see GET /investigations/{id}/attribution for the
    complete AttributionResult, including every candidate."""

    status: str
    candidate_count: int
    search_truncated: bool


class M5MLSummary(BaseModel):
    """Summary only — see GET /investigations/{id}/ml for the complete
    MLPrediction, including the full feature snapshot and disclaimer."""

    # model_name/model_version are ML-model metadata, unrelated to
    # pydantic's own BaseModel machinery — silences the protected-namespace
    # warning (same fix already applied in app/ml/models.py::MLPrediction).
    model_config = ConfigDict(protected_namespaces=())

    predicted_label: str
    training_data_type: str
    model_name: str
    model_version: str


class InvestigationCreateResponse(BaseModel):
    investigation_id: str
    wallet: str
    chain: str
    created_at: dt.datetime

    m4_evidence: M4EvidenceSummary
    m5_ml_prediction: M5MLSummary
    training_data_type: str


class InvestigationSummaryResponse(BaseModel):
    investigation_id: str
    wallet: str
    chain: str
    graph_path: Optional[str]
    max_hops: int
    search_truncated: bool
    attribution_status: str
    ml_predicted_label: str
    created_at: dt.datetime


class HealthResponse(BaseModel):
    status: str
    database: str


class BriefRequest(BaseModel):
    """Body for `POST /analysis/brief`.

    Every field changes behaviour; there are no cosmetic options. The wallet
    and chain are validated by the pipeline itself rather than by a regex here,
    so the HTTP surface and the CLI accept exactly the same inputs and reject
    the same ones for the same stated reason.
    """

    wallet: str
    chain: str = "ethereum"
    #: True reuses a real artefact already on disk for this wallet when one
    #: exists (answered as CACHED REAL DATA), which is what makes an
    #: interactive request return in seconds. False forces a live fetch, which
    #: expands hop by hop and can take minutes.
    prefer_cached: bool = True
    max_hops: Optional[int] = Field(default=None, ge=1)
    max_paths: Optional[int] = Field(default=None, ge=1)
    #: The ML stage is advisory and never changes the attribution evidence, so
    #: it can be skipped without affecting any finding.
    enable_ml: bool = True


class ErrorResponse(BaseModel):
    """Uniform error body. `detail` is always a safe, human-readable
    message — never a stack trace, secret, or raw driver exception."""

    error: str
    detail: str
