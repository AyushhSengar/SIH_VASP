"""
MACRO MILESTONE 5 — training and prediction.

A single, simple, deterministic scikit-learn DecisionTreeClassifier
trained on the synthetic dataset in app/ml/training_data.py. No deep
learning, no embeddings, no LLM involvement.

--------------------------------------------------------------------------
NEVER TOUCHES ATTRIBUTION (do not remove): predict() takes a WalletFeatures
snapshot and returns an MLPrediction. It has no access to, and cannot
construct or mutate, a VASPCandidate or AttributionResult. See
app/ml/models.py's module docstring for the full separation rationale.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from sklearn.tree import DecisionTreeClassifier

from app.ml.models import MLLabel, MLPrediction, WalletFeatures
from app.ml.training_data import SYNTHETIC_DATA_DISCLAIMER, SYNTHETIC_TRAINING_DATA

MODEL_NAME = "DecisionTreeClassifier"
MODEL_VERSION = "m5-synthetic-v1"
DEFAULT_SEED = 42

# A wallet with zero graph presence and zero traced paths has nothing for
# any classifier to honestly say something about — predict() short-circuits
# to INSUFFICIENT_DATA before ever calling the model in that case.
_MIN_EDGES_OR_PATHS_FOR_PREDICTION = 1


@dataclass(frozen=True)
class TrainedModel:
    classifier: DecisionTreeClassifier
    seed: int
    feature_names: list[str]


def train_model(seed: int = DEFAULT_SEED) -> TrainedModel:
    """Trains on the synthetic dataset only. Fully deterministic for a
    fixed seed: the same seed always produces the same row order (fed to
    the classifier) and the same DecisionTreeClassifier(random_state=seed)
    internal tie-breaking, so predictions are reproducible across runs
    and across processes.
    """
    rows = list(SYNTHETIC_TRAINING_DATA)
    rng = random.Random(seed)
    rng.shuffle(rows)  # order shouldn't matter to a decision tree's split
    # choice, but shuffling deterministically (rather than relying on
    # dataset insertion order) makes that assumption explicit and testable
    # rather than accidental.

    x = [features.to_feature_vector() for features, _label in rows]
    y = [label for _features, label in rows]

    classifier = DecisionTreeClassifier(random_state=seed, max_depth=4)
    classifier.fit(x, y)

    return TrainedModel(
        classifier=classifier, seed=seed, feature_names=WalletFeatures.feature_names()
    )


def predict(model: TrainedModel, features: WalletFeatures) -> MLPrediction:
    has_any_signal = (
        features.total_edge_count >= _MIN_EDGES_OR_PATHS_FOR_PREDICTION
        or features.path_count >= _MIN_EDGES_OR_PATHS_FOR_PREDICTION
    )

    if not has_any_signal:
        label = MLLabel.INSUFFICIENT_DATA
    else:
        raw_label = model.classifier.predict([features.to_feature_vector()])[0]
        label = MLLabel(raw_label)

    return MLPrediction(
        wallet=features.wallet,
        predicted_label=label,
        training_data_type="SYNTHETIC_DEMO",
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        random_seed=model.seed,
        feature_snapshot=features,
        disclaimer=SYNTHETIC_DATA_DISCLAIMER,
    )
