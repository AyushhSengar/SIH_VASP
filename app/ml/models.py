"""
MACRO MILESTONE 5 — ML models.

--------------------------------------------------------------------------
HARD SEPARATION FROM ATTRIBUTION (do not remove): WalletFeatures and
MLPrediction share NO fields with VASPCandidate/AttributionResult
(app/attribution/models.py) — no `vasp_name`, no `matched_address`, no
`evidence_tier`. This is intentional and structurally enforced (see
tests/test_ml_predictor.py::test_ml_prediction_has_no_attribution_fields).
An MLPrediction is a behavioral-pattern classification, never a VASP
identity claim. Nothing in app/ml/ may construct, mutate, or return a
VASPCandidate or AttributionResult.

NO FAKE CONFIDENCE (do not remove): MLPrediction deliberately has no
numeric confidence/probability field. The underlying scikit-learn model
can produce predict_proba() output, but surfacing it here would present
a hand-crafted synthetic-data classifier's internal score as if it were
a calibrated, real-world probability. It is not. `disclaimer` and
`training_data_type` exist specifically so this can never be reported
without the caveat attached, on every single prediction object — not
just once in a CLI banner.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class MLLabel(str, Enum):
    LIKELY_VASP_CONNECTED = "LIKELY_VASP_CONNECTED"
    LIKELY_NOT_VASP_CONNECTED = "LIKELY_NOT_VASP_CONNECTED"
    # Returned (by the predictor, not the model itself — see
    # app/ml/predictor.py) whenever a wallet has no graph presence and no
    # traced paths at all: there is nothing for a classifier to honestly
    # say anything about, so it does not guess.
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class WalletFeatures(BaseModel):
    """Feature vector for one wallet, built ONLY from data already produced
    by M1-M4 (graph structure, M3 tracing, M3 behavior detection, M4
    attribution). No feature here is invented or fetched independently.
    """

    wallet: str

    # --- graph-structural (Milestone 2) ---
    in_degree: int
    out_degree: int
    unique_in_counterparties: int
    unique_out_counterparties: int
    total_edge_count: int

    # --- fund-flow tracing (Milestone 3 / Phase A) ---
    path_count: int
    max_hop_count: int
    avg_hop_count: float
    max_path_duration_seconds: Optional[float] = None
    avg_path_duration_seconds: Optional[float] = None
    # Paths whose duration could not be computed (missing/invalid hop
    # timestamps) — counted explicitly rather than silently imputed as 0
    # in this field; the feature *vector* does impute 0 for the classifier
    # (see WalletFeatures.to_feature_vector docstring) but that imputation
    # is visible here, not hidden.
    paths_with_unknown_duration: int

    # --- behavioral pattern flags (Milestone 3 / Phase B) ---
    has_split_pattern: bool
    has_consolidation_pattern: bool
    has_rapid_hopping: bool
    has_high_frequency_counterparty: bool
    has_repeated_forwarding: bool
    has_temporal_burst: bool
    behavior_pattern_count: int

    # --- existing attribution evidence flags (Milestone 4) ---
    # Read-only reflections of an AttributionResult already computed
    # elsewhere. Nothing in app/ml/ writes to AttributionResult.
    attribution_status: str  # AttributionStatus.value, kept as plain str so
    # this module never needs to import app.attribution's enum types
    has_direct_evidence: bool
    has_indirect_evidence: bool
    candidate_count: int

    @staticmethod
    def feature_names() -> list[str]:
        """Fixed, ordered feature names matching to_feature_vector()
        exactly. Changing either without changing the other breaks the
        classifier silently — keep them next to each other on purpose."""
        return [
            "in_degree",
            "out_degree",
            "unique_in_counterparties",
            "unique_out_counterparties",
            "total_edge_count",
            "path_count",
            "max_hop_count",
            "avg_hop_count",
            "max_path_duration_seconds",
            "avg_path_duration_seconds",
            "paths_with_unknown_duration",
            "has_split_pattern",
            "has_consolidation_pattern",
            "has_rapid_hopping",
            "has_high_frequency_counterparty",
            "has_repeated_forwarding",
            "has_temporal_burst",
            "behavior_pattern_count",
            "has_direct_evidence",
            "has_indirect_evidence",
            "candidate_count",
        ]

    def to_feature_vector(self) -> list[float]:
        """Deterministic, fixed-order numeric vector for the classifier.

        Missing durations are imputed as 0.0 here ONLY for the numeric
        vector fed to scikit-learn (which cannot take None) — the
        original None values remain visible on the model fields above,
        and `paths_with_unknown_duration` exists precisely so this
        imputation is never silently hidden from anyone inspecting the
        WalletFeatures object directly.
        """
        return [
            float(self.in_degree),
            float(self.out_degree),
            float(self.unique_in_counterparties),
            float(self.unique_out_counterparties),
            float(self.total_edge_count),
            float(self.path_count),
            float(self.max_hop_count),
            float(self.avg_hop_count),
            float(self.max_path_duration_seconds or 0.0),
            float(self.avg_path_duration_seconds or 0.0),
            float(self.paths_with_unknown_duration),
            float(self.has_split_pattern),
            float(self.has_consolidation_pattern),
            float(self.has_rapid_hopping),
            float(self.has_high_frequency_counterparty),
            float(self.has_repeated_forwarding),
            float(self.has_temporal_burst),
            float(self.behavior_pattern_count),
            float(self.has_direct_evidence),
            float(self.has_indirect_evidence),
            float(self.candidate_count),
        ]


class MLPrediction(BaseModel):
    """Output of the M5 classifier for one wallet.

    Deliberately does NOT subclass or embed VASPCandidate/AttributionResult
    and shares no field names with them (see module docstring). This is a
    behavioral-pattern label, not a VASP identity claim.
    """

    # model_name/model_version are ML-model metadata field names, not
    # related to pydantic's own BaseModel machinery — silence the
    # protected-namespace warning that would otherwise fire on every import.
    model_config = ConfigDict(protected_namespaces=())

    wallet: str
    predicted_label: MLLabel

    training_data_type: Literal["SYNTHETIC_DEMO"] = "SYNTHETIC_DEMO"
    model_name: str
    model_version: str
    random_seed: int

    feature_snapshot: WalletFeatures
    disclaimer: str
