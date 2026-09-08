"""
MACRO MILESTONE 6 tests — repository layer.

Uses an isolated in-memory SQLite database per test (StaticPool keeps
the same in-memory connection alive across the session used here).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.attribution.models import AttributionResult, AttributionStatus
from app.db.models import Base
from app.db.repository import InvestigationRepository
from app.ml.models import MLLabel, MLPrediction, WalletFeatures


@pytest.fixture()
def repository():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    db = session_factory()
    try:
        yield InvestigationRepository(db)
    finally:
        db.close()


def _wallet_features(wallet: str) -> WalletFeatures:
    return WalletFeatures(
        wallet=wallet,
        in_degree=1,
        out_degree=1,
        unique_in_counterparties=1,
        unique_out_counterparties=1,
        total_edge_count=2,
        path_count=1,
        max_hop_count=1,
        avg_hop_count=1.0,
        max_path_duration_seconds=10.0,
        avg_path_duration_seconds=10.0,
        paths_with_unknown_duration=0,
        has_split_pattern=False,
        has_consolidation_pattern=False,
        has_rapid_hopping=False,
        has_high_frequency_counterparty=False,
        has_repeated_forwarding=False,
        has_temporal_burst=False,
        behavior_pattern_count=0,
        attribution_status=AttributionStatus.NONE.value,
        has_direct_evidence=False,
        has_indirect_evidence=False,
        candidate_count=0,
    )


def _attribution_result(wallet: str) -> AttributionResult:
    return AttributionResult(
        wallet=wallet,
        status=AttributionStatus.NONE,
        candidates=[],
        max_hops=4,
        search_truncated=False,
        notes=["no known VASP reached"],
    )


def _ml_prediction(wallet: str) -> MLPrediction:
    return MLPrediction(
        wallet=wallet,
        predicted_label=MLLabel.LIKELY_NOT_VASP_CONNECTED,
        training_data_type="SYNTHETIC_DEMO",
        model_name="DecisionTreeClassifier",
        model_version="m5-synthetic-v1",
        random_seed=42,
        feature_snapshot=_wallet_features(wallet),
        disclaimer="Synthetic/demo model — not real-world attribution.",
    )


def test_investigation_create_get_round_trip(repository: InvestigationRepository):
    wallet = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    investigation_id = uuid.uuid4().hex

    created = repository.create_investigation(
        investigation_id=investigation_id,
        wallet=wallet,
        chain="ethereum",
        max_hops=4,
        max_paths=500,
        graph_path="data/graphs/example.gpickle",
        search_truncated=False,
        attribution_status=AttributionStatus.NONE.value,
        ml_predicted_label=MLLabel.LIKELY_NOT_VASP_CONNECTED.value,
        training_data_type="SYNTHETIC_DEMO",
    )
    assert created.id == investigation_id

    fetched = repository.get_investigation(investigation_id)
    assert fetched is not None
    assert fetched.wallet == wallet
    assert fetched.chain == "ethereum"
    assert fetched.graph_path == "data/graphs/example.gpickle"


def test_get_investigation_returns_none_when_missing(repository: InvestigationRepository):
    assert repository.get_investigation(uuid.uuid4().hex) is None


def test_attribution_stored_and_read_back_separately(repository: InvestigationRepository):
    wallet = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    investigation_id = uuid.uuid4().hex
    repository.create_investigation(
        investigation_id=investigation_id,
        wallet=wallet,
        chain="ethereum",
        max_hops=4,
        max_paths=500,
        graph_path=None,
        search_truncated=False,
        attribution_status=AttributionStatus.NONE.value,
        ml_predicted_label=MLLabel.LIKELY_NOT_VASP_CONNECTED.value,
        training_data_type="SYNTHETIC_DEMO",
    )

    attribution = _attribution_result(wallet)
    repository.create_attribution(investigation_id=investigation_id, attribution_result=attribution)

    stored = repository.get_attribution(investigation_id)
    assert stored is not None
    round_tripped = AttributionResult.model_validate_json(stored.attribution_json)
    assert round_tripped == attribution

    # M4 attribution must never contain ML fields.
    assert not hasattr(round_tripped, "predicted_label")
    assert not hasattr(round_tripped, "training_data_type")


def test_ml_prediction_stored_and_read_back_separately(repository: InvestigationRepository):
    wallet = "0xcccccccccccccccccccccccccccccccccccccccc"
    investigation_id = uuid.uuid4().hex
    repository.create_investigation(
        investigation_id=investigation_id,
        wallet=wallet,
        chain="ethereum",
        max_hops=4,
        max_paths=500,
        graph_path=None,
        search_truncated=False,
        attribution_status=AttributionStatus.NONE.value,
        ml_predicted_label=MLLabel.LIKELY_NOT_VASP_CONNECTED.value,
        training_data_type="SYNTHETIC_DEMO",
    )

    prediction = _ml_prediction(wallet)
    repository.create_ml_prediction(investigation_id=investigation_id, ml_prediction=prediction)

    stored = repository.get_ml_prediction(investigation_id)
    assert stored is not None
    round_tripped = MLPrediction.model_validate_json(stored.ml_json)
    assert round_tripped == prediction
    assert round_tripped.training_data_type == "SYNTHETIC_DEMO"

    # M5 prediction must never contain VASP attribution fields.
    assert not hasattr(round_tripped, "vasp_name")
    assert not hasattr(round_tripped, "matched_address")
    assert not hasattr(round_tripped, "evidence_tier")


def test_attribution_and_ml_are_independent_records(repository: InvestigationRepository):
    """Persisting one must not create or affect the other."""
    wallet = "0xdddddddddddddddddddddddddddddddddddddddd"
    investigation_id = uuid.uuid4().hex
    repository.create_investigation(
        investigation_id=investigation_id,
        wallet=wallet,
        chain="ethereum",
        max_hops=4,
        max_paths=500,
        graph_path=None,
        search_truncated=False,
        attribution_status=AttributionStatus.NONE.value,
        ml_predicted_label=MLLabel.LIKELY_NOT_VASP_CONNECTED.value,
        training_data_type="SYNTHETIC_DEMO",
    )
    repository.create_attribution(
        investigation_id=investigation_id, attribution_result=_attribution_result(wallet)
    )

    assert repository.get_attribution(investigation_id) is not None
    assert repository.get_ml_prediction(investigation_id) is None
