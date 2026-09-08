"""
MACRO MILESTONE 6 — ORM models.

Three tables, linked by `investigation_id`:
  - investigations       (one row per investigation run)
  - attribution_results  (the M4 AttributionResult for that run)
  - ml_predictions        (the M5 MLPrediction for that run)

Design choice — full-object JSON columns instead of flattening every
nested field (paths, candidates, hops, feature vectors) into relational
columns: AttributionResult and MLPrediction are already the canonical,
versioned pydantic output models from M4/M5. Storing their exact
`model_dump_json()` means the API layer can return the COMPLETE object
byte-for-equivalent to what M4/M5 produced, with zero risk of a
hand-maintained relational mapping silently dropping or renaming a
field. A handful of denormalized scalar columns (status, predicted
label, etc.) are kept alongside for cheap filtering/summaries without
deserializing the JSON blob.

These models are intentionally the ONLY place that knows about SQL.
Nothing outside app/db/ should import sqlalchemy.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class InvestigationRecord(Base):
    __tablename__ = "investigations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    wallet: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    chain: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    max_hops: Mapped[int] = mapped_column(Integer, nullable=False)
    max_paths: Mapped[int] = mapped_column(Integer, nullable=False)

    graph_path: Mapped[str] = mapped_column(String(512), nullable=True)
    search_truncated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Denormalized read models of the M4/M5 outputs — the authoritative,
    # complete records live in AttributionRecord/MLPredictionRecord below.
    attribution_status: Mapped[str] = mapped_column(String(32), nullable=False)
    ml_predicted_label: Mapped[str] = mapped_column(String(64), nullable=False)
    training_data_type: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    attribution: Mapped["AttributionRecord"] = relationship(
        back_populates="investigation",
        uselist=False,
        cascade="all, delete-orphan",
    )
    ml_prediction: Mapped["MLPredictionRecord"] = relationship(
        back_populates="investigation",
        uselist=False,
        cascade="all, delete-orphan",
    )


class AttributionRecord(Base):
    """Persists the complete Macro Milestone 4 AttributionResult, verbatim,
    for one investigation. Never contains ML fields — see
    app.attribution.models.AttributionResult's module docstring for why
    that separation is structurally guaranteed upstream, before this
    table ever sees the object."""

    __tablename__ = "attribution_results"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    investigation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("investigations.id"), nullable=False, unique=True, index=True
    )

    wallet: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)

    # Complete app.attribution.models.AttributionResult, serialized via
    # AttributionResult.model_dump_json(). This is the source of truth
    # returned by GET /investigations/{id}/attribution.
    attribution_json: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    investigation: Mapped["InvestigationRecord"] = relationship(back_populates="attribution")


class MLPredictionRecord(Base):
    """Persists the complete Macro Milestone 5 MLPrediction, verbatim, for
    one investigation. Always SYNTHETIC_DEMO training data — never contains
    or implies a VASP attribution field. See app.ml.models.MLPrediction's
    module docstring for the structural separation this relies on."""

    __tablename__ = "ml_predictions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    investigation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("investigations.id"), nullable=False, unique=True, index=True
    )

    wallet: Mapped[str] = mapped_column(String(128), nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(64), nullable=False)
    training_data_type: Mapped[str] = mapped_column(String(32), nullable=False)

    # Complete app.ml.models.MLPrediction, serialized via
    # MLPrediction.model_dump_json(). This is the source of truth returned
    # by GET /investigations/{id}/ml — including feature_snapshot and
    # disclaimer, so the synthetic/demo caveat can never be dropped.
    ml_json: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    investigation: Mapped["InvestigationRecord"] = relationship(back_populates="ml_prediction")
