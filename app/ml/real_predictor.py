"""
REAL PREDICTOR — loading a real-data model and explaining one prediction.

The rule this module exists to enforce: a prediction may only be produced by an
artifact that was actually trained and persisted by
`app/ml/real_training.py`. When no such artifact exists, `predict` returns a
refusal carrying the reason — never a default class, never a 0.5 "unknown"
probability, never a value from the synthetic demo model in
`app/ml/predictor.py`.

WHAT AN EXPLANATION HERE IS, AND IS NOT
--------------------------------------------------------------------------
The explanation names the features that pushed this address's score, with the
address's actual value beside the training-population median, so a reader can
see *why* the model scored it and check each number against the graph. It is a
description of a statistical association learned from graph behaviour.

It is NOT evidence. `PredictionResult.evidence_class` is fixed at SUPPORTING
for exactly that reason: a prediction may corroborate an address-level finding
and may never outrank one, and a report must not be able to present it as
though it could.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict

from app.core.config import Settings, get_settings
from app.ml.real_features import (
    FEATURE_SCHEMA_VERSION,
    extract_address_features,
)


class FeatureContribution(BaseModel):
    """One feature's role in this prediction, with checkable numbers."""

    feature: str
    value: float
    # Median of the same feature across the training samples, so "high" is a
    # measured comparison rather than an adjective.
    training_median: Optional[float] = None
    importance: float
    method: str


class PredictionResult(BaseModel):
    """A single prediction, or a refusal, in one shape.

    `available=False` is a normal outcome and callers must handle it: it is
    what "no trained real model exists" looks like, and it is preferable to
    any invented number.
    """

    model_config = ConfigDict(protected_namespaces=())

    address: str
    task: str
    available: bool

    predicted_class: Optional[str] = None
    probability: Optional[float] = None
    decision_threshold: Optional[float] = None

    model_name: Optional[str] = None
    model_version: Optional[str] = None
    dataset_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
    label_schema_version: Optional[str] = None
    trained_at_utc: Optional[str] = None
    random_seed: Optional[int] = None
    training_sample_count: Optional[int] = None
    training_class_counts: dict[str, int] = {}
    # The held-out metrics of the model making this prediction, carried on the
    # prediction itself so a reader always sees how good the model is at the
    # moment they see what it claims.
    test_metrics: dict[str, Any] = {}

    contributions: list[FeatureContribution] = []
    #: Fixed: an ML prediction is never DIRECT or INDIRECT evidence.
    evidence_class: str = "SUPPORTING"
    interpretation: list[str] = []
    unavailable_reason: Optional[str] = None
    temporal_data_missing: bool = False


def find_artifact(task: str, settings: Optional[Settings] = None) -> Optional[Path]:
    """Locates the newest persisted artifact for a task, or None.

    "Newest" is by modification time so that retraining supersedes an older
    model without needing a registry file. Returning None is a supported
    result, not an error condition.
    """
    settings = settings or get_settings()
    directory = Path(settings.ml_artifact_dir)
    if not directory.is_dir():
        return None
    candidates = sorted(
        directory.glob(f"{task}-*.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load(path: Path) -> tuple[dict, dict]:
    import joblib

    bundle = joblib.load(path)
    sidecar_path = path.with_suffix(".json")
    metadata: dict = {}
    if sidecar_path.is_file():
        try:
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A corrupt sidecar costs us provenance detail but must not stop a
            # prediction the model itself can still make; the missing fields
            # simply stay None in the result rather than being guessed.
            metadata = {}
    return bundle, metadata


def _training_medians(metadata: dict) -> dict[str, float]:
    return {
        item["feature"]: item["median"]
        for item in metadata.get("training_feature_medians", [])
        if "feature" in item and "median" in item
    }


def _importance_index(metadata: dict) -> tuple[dict[str, float], str]:
    importances = metadata.get("feature_importances", [])
    index = {
        item["feature"]: float(item.get("importance", 0.0))
        for item in importances
        if "feature" in item
    }
    method = importances[0].get("method", "unknown") if importances else "unknown"
    return index, method


def predict(
    graph: nx.MultiDiGraph,
    address: str,
    task: str = "account_type",
    settings: Optional[Settings] = None,
    top_features: int = 8,
) -> PredictionResult:
    """Predicts for one address using a persisted real-data model.

    Every failure mode returns `available=False` with a specific reason: no
    artifact, a feature-schema mismatch, or a load failure. A schema mismatch
    in particular must never be papered over — a vector built by a different
    feature version is positionally meaningless to the model, and predicting
    from it would produce a confident-looking number with no basis.
    """
    settings = settings or get_settings()
    address = address.lower()

    path = find_artifact(task, settings)
    if path is None:
        return PredictionResult(
            address=address,
            task=task,
            available=False,
            unavailable_reason=(
                f"No trained real-data model exists for task '{task}' in "
                f"{settings.ml_artifact_dir}. Training was refused because the "
                "available real labels did not meet the per-class minimum; see "
                "the labelling outcome for the exact shortfall. No prediction "
                "is reported rather than substituting a synthetic model."
            ),
        )

    try:
        bundle, metadata = _load(path)
    except Exception as exc:
        return PredictionResult(
            address=address,
            task=task,
            available=False,
            unavailable_reason=(
                f"Model artifact {path.name} could not be loaded "
                f"({type(exc).__name__}). No prediction is reported."
            ),
        )

    artifact_schema = bundle.get("feature_schema_version")
    if artifact_schema != FEATURE_SCHEMA_VERSION:
        return PredictionResult(
            address=address,
            task=task,
            available=False,
            model_version=bundle.get("model_version"),
            unavailable_reason=(
                f"Feature schema mismatch: the artifact was trained on "
                f"{artifact_schema}, this build computes {FEATURE_SCHEMA_VERSION}. "
                "Feature vectors are positional, so the two are not "
                "interchangeable. Retrain before predicting."
            ),
        )

    model = bundle["model"]
    classes: list[str] = bundle["classes"]
    threshold = float(bundle.get("decision_threshold", 0.5))

    features = extract_address_features(graph, address)
    vector = np.asarray([features.to_vector(task)], dtype=float)

    probability: Optional[float] = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(vector)[0][1])
        predicted_index = 1 if probability >= threshold else 0
    else:
        predicted_index = int(model.predict(vector)[0])

    medians = _training_medians(metadata)
    importance_index, method = _importance_index(metadata)
    ranked = sorted(
        features.values.items(),
        key=lambda kv: (-importance_index.get(kv[0], 0.0), kv[0]),
    )
    contributions = [
        FeatureContribution(
            feature=name,
            value=round(float(value), 6),
            training_median=medians.get(name),
            importance=round(importance_index.get(name, 0.0), 6),
            method=method,
        )
        for name, value in ranked[:top_features]
    ]

    provenance = metadata.get("provenance") or {}
    test_metrics = metadata.get("test_metrics") or {}

    interpretation = [
        f"The model assigns class {classes[predicted_index]} at a decision "
        f"threshold of {threshold:.4f}, chosen on the validation split.",
        "This is a statistical association learned from label-blind graph "
        "behaviour. It is SUPPORTING evidence only: it can corroborate an "
        "address-level finding and can never override one.",
    ]
    if test_metrics:
        interpretation.append(
            "Held-out performance of this model: "
            f"accuracy {test_metrics.get('accuracy')}, "
            f"precision {test_metrics.get('precision')}, "
            f"recall {test_metrics.get('recall')}, "
            f"F1 {test_metrics.get('f1')} on "
            f"{test_metrics.get('sample_count')} test sample(s)."
        )
    if features.temporal_data_missing:
        interpretation.append(
            "This address has no timestamped activity in the loaded graph, so "
            "its temporal features are structurally absent and were scored as "
            "zero. Treat the prediction as weaker than the metrics suggest."
        )

    return PredictionResult(
        address=address,
        task=task,
        available=True,
        predicted_class=classes[predicted_index],
        probability=round(probability, 6) if probability is not None else None,
        decision_threshold=round(threshold, 6),
        model_name=bundle.get("model_name"),
        model_version=bundle.get("model_version"),
        dataset_version=provenance.get("dataset_version"),
        feature_schema_version=artifact_schema,
        label_schema_version=provenance.get("label_schema_version"),
        trained_at_utc=metadata.get("trained_at_utc"),
        random_seed=metadata.get("random_seed"),
        training_sample_count=provenance.get("sample_count"),
        training_class_counts=provenance.get("class_counts") or {},
        test_metrics=test_metrics,
        contributions=contributions,
        interpretation=interpretation,
        temporal_data_missing=features.temporal_data_missing,
    )
