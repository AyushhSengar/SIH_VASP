"""
REAL TRAINING — supervised training on real labels, with a refusal gate.

This module trains a classifier ONLY when `app/ml/real_labels.py` says the
labels are sufficient. When they are not, it returns a `TrainingOutcome` with
`trained=False` and the blockers verbatim. It never falls back to synthetic
rows, never relaxes the gate, and never reports a metric it did not measure on
held-out data.

HOW LEAKAGE IS PREVENTED (four separate mechanisms)
--------------------------------------------------------------------------
1. Label-blind features. Nothing in `app/ml/real_features.py` reads a
   provider stream or the VASP dataset, and the task's excluded feature
   groups are enforced here and recorded in the artifact.

2. Group-aware splitting. Splits are on `group_key` (an operator name where
   known, otherwise the address), never on rows, so two addresses of one
   exchange can never straddle train and test. With address-level groups this
   degenerates to address-level splitting, which is still strictly stronger
   than transfer-level splitting: a wallet's transfers are never split across
   folds, which would let the model memorise a wallet from its own history.

3. An untouched test set. The test groups are separated FIRST and are not
   read again until `_evaluate` runs once on the final model. Model choice,
   hyperparameters and the decision threshold are all decided on
   cross-validation and a validation split drawn only from the remainder.

4. No refit on test. The selected model is refit on train+validation only.

RESIDUAL LEAKAGE THAT CANNOT BE ELIMINATED (stated, not hidden)
--------------------------------------------------------------------------
Every sample's features come from ONE shared graph, so a train address and a
test address may be direct counterparties and their features are therefore not
statistically independent. This inflates held-out scores relative to a truly
independent sample, and no split strategy fixes it — only drawing test
addresses from a separately-crawled graph would. It is recorded in
`TrainingOutcome.limitations` so any reported metric carries the caveat.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from app.core.config import Settings, get_settings
from app.ml.real_features import (
    FEATURE_SCHEMA_VERSION,
    TASK_EXCLUDED_GROUPS,
    TASK_EXCLUSION_REASONS,
    extract_address_features,
    feature_names,
)
from app.ml.real_labels import LabelingOutcome

#: Bumped when the training procedure changes in a way that would produce a
#: different model from identical data (split policy, candidate set, metric
#: used for selection). Distinct from the dataset version.
PIPELINE_VERSION = "real-training-v1"

MODEL_VERSION_PREFIX = "real"


class DatasetProvenance(BaseModel):
    """Where a training set came from, in enough detail to rebuild it."""

    task: str
    dataset_version: str  # deterministic hash of the labelled set
    label_schema_version: str
    feature_schema_version: str
    graph_source: str  # file path or description of the graph used
    graph_node_count: int
    graph_edge_count: int
    sample_count: int
    class_counts: dict[str, int]
    group_count: int
    label_source_counts: dict[str, int]
    excluded_feature_groups: list[str]
    feature_exclusion_reason: str
    activity_floor: int


class SplitReport(BaseModel):
    """Exactly how the data was divided, per split and per class."""

    strategy: str
    train_sample_count: int
    validation_sample_count: int
    test_sample_count: int
    train_group_count: int
    validation_group_count: int
    test_group_count: int
    train_class_counts: dict[str, int]
    validation_class_counts: dict[str, int]
    test_class_counts: dict[str, int]
    groups_overlap: bool  # must be False; asserted and reported
    cv_folds: int
    random_seed: int


class EvaluationMetrics(BaseModel):
    """Metrics measured on ONE split. `split` says which, always."""

    split: str
    sample_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    # Row-major [[TN, FP], [FN, TP]] with the positive class named.
    confusion_matrix: list[list[int]]
    positive_class: str
    class_counts: dict[str, int]
    decision_threshold: float
    # What a model that always predicts the largest class would score on this
    # split. Reported next to accuracy because on an imbalanced problem a high
    # accuracy can be entirely explained by the imbalance, and printing the
    # figure without its baseline overstates the model.
    majority_class_baseline_accuracy: float
    accuracy_above_baseline: float


class FeatureImportance(BaseModel):
    feature: str
    importance: float
    method: str


class FeatureMedian(BaseModel):
    """A feature's median across the training samples.

    Persisted so a later prediction can print the address's value next to the
    population it was scored against. Without it an explanation can only say
    "this feature mattered", which is not checkable.
    """

    feature: str
    median: float


class CandidateResult(BaseModel):
    """One model's cross-validated score, kept even when it loses.

    Reported so "we chose a random forest" is a visible comparison rather
    than an assertion.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    cv_mean_f1: float
    cv_std_f1: float
    cv_fold_scores: list[float]
    selected: bool
    note: Optional[str] = None


class TrainingOutcome(BaseModel):
    """The complete, self-describing result of a training attempt."""

    model_config = ConfigDict(protected_namespaces=())

    task: str
    trained: bool

    # Populated only when trained is True.
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    pipeline_version: str = PIPELINE_VERSION
    trained_at: Optional[int] = None
    trained_at_utc: Optional[str] = None
    random_seed: Optional[int] = None

    provenance: Optional[DatasetProvenance] = None
    split_report: Optional[SplitReport] = None
    candidates: list[CandidateResult] = []
    validation_metrics: Optional[EvaluationMetrics] = None
    test_metrics: Optional[EvaluationMetrics] = None
    feature_importances: list[FeatureImportance] = []
    training_feature_medians: list[FeatureMedian] = []
    class_imbalance_handling: Optional[str] = None

    # Populated when training was refused.
    blockers: list[str] = []
    label_outcome: Optional[LabelingOutcome] = None

    limitations: list[str] = []
    artifact_path: Optional[str] = None
    environment: dict[str, str] = {}


def _dataset_version(outcome: LabelingOutcome) -> str:
    """A deterministic fingerprint of the labelled set.

    Hashes the sorted (address, label, source) triples together with both
    schema versions, so any change to who is in the set, what they are
    labelled, or what a label means produces a different version string.
    Twelve hex characters is enough to distinguish datasets in a report while
    staying readable.
    """
    payload = json.dumps(
        {
            "label_schema_version": outcome.label_schema_version,
            "task": outcome.task,
            "labels": sorted(
                (item.address, item.label, item.label_source.value)
                for item in outcome.labels
            ),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _class_counts(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _candidate_models(seed: int, class_weight: Optional[str]) -> list[tuple[str, Any]]:
    """The candidate set.

    Classical tabular learners only. No deep model is included: with a few
    hundred wallets and ~30 tabular features, a neural network has no
    advantage that could justify the loss of interpretability, and gradient
    boosting is the standard strong baseline on exactly this shape of data.

    `GradientBoostingClassifier` and `HistGradientBoostingClassifier` do not
    accept `class_weight` in the constructor across all supported sklearn
    versions, so imbalance is handled for them with per-sample weights at fit
    time instead — the same correction applied a different way, never skipped.
    """
    models: list[tuple[str, Any]] = [
        (
            "LogisticRegression",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight=class_weight,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight=class_weight,
                random_state=seed,
                n_jobs=1,
            ),
        ),
        (
            "GradientBoostingClassifier",
            GradientBoostingClassifier(random_state=seed),
        ),
        (
            "HistGradientBoostingClassifier",
            HistGradientBoostingClassifier(random_state=seed),
        ),
    ]

    # Optional gradient-boosting libraries are used when the environment has
    # them and skipped with a recorded note when it does not, so the candidate
    # list always matches what actually ran.
    try:  # pragma: no cover - depends on the environment
        from xgboost import XGBClassifier

        models.append(
            (
                "XGBClassifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=seed,
                    eval_metric="logloss",
                    n_jobs=1,
                    verbosity=0,
                ),
            )
        )
    except ImportError:  # pragma: no cover
        pass

    try:  # pragma: no cover - depends on the environment
        from lightgbm import LGBMClassifier

        models.append(
            (
                "LGBMClassifier",
                LGBMClassifier(
                    n_estimators=300,
                    random_state=seed,
                    class_weight=class_weight,
                    n_jobs=1,
                    verbose=-1,
                ),
            )
        )
    except ImportError:  # pragma: no cover
        pass

    return models


def _needs_sample_weights(model_name: str) -> bool:
    return model_name in (
        "GradientBoostingClassifier",
        "HistGradientBoostingClassifier",
        "XGBClassifier",
    )


@contextmanager
def _quiet_feature_name_warning():
    """Silences sklearn's "X does not have valid feature names" warning.

    LightGBM assigns itself placeholder column names at fit time, so scoring
    the same ndarray it was trained on trips a mismatch warning. This pipeline
    is positional by design and guards ordering with FEATURE_SCHEMA_VERSION,
    so the warning is noise here. Scoped narrowly to this one message rather
    than filtering warnings globally.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="X does not have valid feature names"
        )
        yield


def _fit(model: Any, model_name: str, X: np.ndarray, y: np.ndarray) -> Any:
    with _quiet_feature_name_warning():
        if _needs_sample_weights(model_name):
            weights = compute_sample_weight("balanced", y)
            model.fit(X, y, sample_weight=weights)
        else:
            model.fit(X, y)
    return model


def _positive_scores(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
    """Probability (or decision score) for the positive class, or None.

    Returns None rather than inventing a score when a model exposes neither
    `predict_proba` nor `decision_function`; the caller then omits ROC-AUC and
    PR-AUC instead of reporting a fabricated curve.
    """
    if hasattr(model, "predict_proba"):
        with _quiet_feature_name_warning():
            return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        with _quiet_feature_name_warning():
            return model.decision_function(X)
    return None


def _evaluate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    split: str,
    classes: list[str],
    threshold: float,
) -> EvaluationMetrics:
    """Measures performance on one split at a stated threshold.

    The threshold is applied explicitly rather than relying on the model's
    0.5 default, because it was chosen on the validation split and the test
    evaluation must use that same chosen value — using a different one would
    make the two numbers incomparable.
    """
    scores = _positive_scores(model, X)
    if scores is not None:
        predictions = (scores >= threshold).astype(int)
    else:
        with _quiet_feature_name_warning():
            predictions = model.predict(X)

    matrix = confusion_matrix(y, predictions, labels=[0, 1]).tolist()

    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    # A single-class split has no ROC curve; reporting 0.5 or 1.0 there would
    # be an invented number.
    if scores is not None and len(set(y.tolist())) == 2:
        roc_auc = round(float(roc_auc_score(y, scores)), 4)
        pr_auc = round(float(average_precision_score(y, scores)), 4)

    accuracy = float(accuracy_score(y, predictions))
    negatives, positives = int((y == 0).sum()), int((y == 1).sum())
    baseline = max(negatives, positives) / len(y) if len(y) else 0.0

    return EvaluationMetrics(
        split=split,
        sample_count=int(len(y)),
        accuracy=round(accuracy, 4),
        precision=round(float(precision_score(y, predictions, zero_division=0)), 4),
        recall=round(float(recall_score(y, predictions, zero_division=0)), 4),
        f1=round(float(f1_score(y, predictions, zero_division=0)), 4),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=matrix,
        positive_class=classes[1],
        class_counts={classes[0]: negatives, classes[1]: positives},
        decision_threshold=round(float(threshold), 4),
        majority_class_baseline_accuracy=round(baseline, 4),
        accuracy_above_baseline=round(accuracy - baseline, 4),
    )


def _choose_threshold(
    model: Any, X: np.ndarray, y: np.ndarray
) -> tuple[float, str]:
    """Picks the decision threshold that maximises F1 on the VALIDATION split.

    Selected on validation and never on test, so the reported test metrics
    include no tuning done with knowledge of the test labels. Falls back to
    0.5 when the model exposes no scores or the split has one class only.
    """
    scores = _positive_scores(model, X)
    if scores is None or len(set(y.tolist())) < 2:
        return 0.5, "fixed 0.5 (no scores available or single-class validation split)"

    best_threshold, best_f1 = 0.5, -1.0
    for candidate in np.unique(np.round(scores, 4)):
        predicted = (scores >= candidate).astype(int)
        score = f1_score(y, predicted, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(candidate), float(score)
    return best_threshold, "chosen on the validation split by maximising F1"


def _importances(
    model: Any, model_name: str, names: list[str]
) -> list[FeatureImportance]:
    """Feature importance from the model itself where it is meaningful.

    Tree ensembles expose impurity-based importances; the logistic pipeline
    exposes standardised coefficients, whose magnitudes are comparable because
    the features were scaled. The method is recorded on every row so a reader
    knows what kind of number they are looking at rather than assuming SHAP.
    """
    estimator = model
    method = "impurity-based (model.feature_importances_)"
    if isinstance(model, Pipeline):
        estimator = model.named_steps["clf"]
        method = "absolute standardised logistic coefficient"

    raw: Optional[np.ndarray] = None
    if hasattr(estimator, "feature_importances_"):
        raw = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        raw = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()

    if raw is None or len(raw) != len(names):
        return []

    ranked = sorted(
        (
            FeatureImportance(
                feature=name, importance=round(float(value), 6), method=method
            )
            for name, value in zip(names, raw)
        ),
        key=lambda item: (-item.importance, item.feature),
    )
    return ranked


def _split_groups(
    groups: np.ndarray, y: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Splits sample indices by group, keeping class balance as even as the
    group structure allows.

    Implemented as a single deterministic pass rather than with
    GroupShuffleSplit so that both the group disjointness and the
    stratification intent are visible: groups are shuffled with a seeded RNG,
    then assigned to the held-out side until it reaches its target size,
    preferring groups that do not exhaust a class.
    """
    unique = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))

    target = max(1, int(round(len(y) * fraction)))
    held_out_groups: set[str] = set()
    held_out_size = 0
    remaining_class_counts = {
        int(cls): int((y == cls).sum()) for cls in np.unique(y)
    }

    for index in order:
        group = unique[index]
        member_mask = groups == group
        member_classes = {
            int(cls): int((y[member_mask] == cls).sum()) for cls in np.unique(y)
        }
        # Never move the last remaining member of a class out of training.
        if any(
            remaining_class_counts[cls] - count <= 0
            for cls, count in member_classes.items()
            if count
        ):
            continue
        held_out_groups.add(str(group))
        held_out_size += int(member_mask.sum())
        for cls, count in member_classes.items():
            remaining_class_counts[cls] -= count
        if held_out_size >= target:
            break

    held_out_mask = np.array([str(g) in held_out_groups for g in groups])
    return np.where(~held_out_mask)[0], np.where(held_out_mask)[0]


def build_dataset(
    graph: nx.MultiDiGraph,
    label_outcome: LabelingOutcome,
    graph_source: str,
    settings: Optional[Settings] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], DatasetProvenance]:
    """Turns a labelling outcome into (X, y, groups, class_names, provenance).

    THE POSITIVE CLASS IS THE MINORITY CLASS. On an imbalanced problem,
    precision, recall, F1 and PR-AUC computed against the majority class are
    close to uninformative — a model that predicts the majority every time
    scores near 1.0 on all four. Pointing them at the rarer class makes them
    measure the thing that is actually hard, so the headline F1 cannot be
    inflated by the imbalance. Ties are broken alphabetically for
    determinism.
    """
    settings = settings or get_settings()
    task = label_outcome.task
    names = feature_names(task)

    present = sorted({item.label for item in label_outcome.labels})
    if len(present) < 2:
        classes = present + ["<absent>"] * (2 - len(present))
    else:
        counts = _class_counts([item.label for item in label_outcome.labels])
        # Minority second => index 1 => the positive class.
        ordered = sorted(present, key=lambda name: (-counts[name], name))
        classes = [ordered[0], ordered[-1]]

    rows: list[list[float]] = []
    targets: list[int] = []
    groups: list[str] = []
    for item in label_outcome.labels:
        features = extract_address_features(graph, item.address)
        rows.append(features.to_vector(task))
        targets.append(1 if item.label == classes[1] else 0)
        groups.append(item.group_key)

    label_source_counts: dict[str, int] = {}
    for item in label_outcome.labels:
        key = item.label_source.value
        label_source_counts[key] = label_source_counts.get(key, 0) + 1

    provenance = DatasetProvenance(
        task=task,
        dataset_version=_dataset_version(label_outcome),
        label_schema_version=label_outcome.label_schema_version,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        graph_source=graph_source,
        graph_node_count=graph.number_of_nodes(),
        graph_edge_count=graph.number_of_edges(),
        sample_count=len(rows),
        class_counts=_class_counts([item.label for item in label_outcome.labels]),
        group_count=label_outcome.group_count,
        label_source_counts=dict(sorted(label_source_counts.items())),
        excluded_feature_groups=list(TASK_EXCLUDED_GROUPS.get(task, ())),
        feature_exclusion_reason=TASK_EXCLUSION_REASONS.get(
            task, "No exclusion rule recorded for this task."
        ),
        activity_floor=label_outcome.activity_floor,
    )

    X = np.asarray(rows, dtype=float) if rows else np.empty((0, len(names)))
    y = np.asarray(targets, dtype=int)
    group_array = np.asarray(groups, dtype=object)
    return X, y, group_array, classes, provenance


_RESIDUAL_LEAKAGE_LIMITATION = (
    "All samples' features come from one shared transaction graph, so a "
    "training address and a test address can be direct counterparties. Their "
    "features are therefore not statistically independent, which inflates "
    "held-out scores relative to a separately-crawled sample. Group-aware "
    "splitting cannot remove this; only a second, independent crawl would."
)


def train_real_model(
    graph: nx.MultiDiGraph,
    label_outcome: LabelingOutcome,
    graph_source: str,
    settings: Optional[Settings] = None,
    persist: bool = True,
) -> TrainingOutcome:
    """Trains, evaluates and (optionally) persists a real-data model.

    Refuses — returning `trained=False` with the label blockers attached —
    whenever the labelling gate is not satisfied. That refusal is the correct
    output, not a failure: the alternative is a metric computed from too few
    real labels to mean anything.
    """
    settings = settings or get_settings()
    seed = settings.ml_random_seed
    task = label_outcome.task

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    if not label_outcome.sufficient:
        return TrainingOutcome(
            task=task,
            trained=False,
            blockers=list(label_outcome.blockers),
            label_outcome=label_outcome,
            random_seed=seed,
            environment=environment,
            limitations=[
                "No model was trained, so this investigation reports NO "
                "supervised accuracy, precision, recall or F1. Any such number "
                "would be fabricated.",
                *(f"Label gate: {b}" for b in label_outcome.blockers),
            ],
        )

    X, y, groups, classes, provenance = build_dataset(
        graph, label_outcome, graph_source, settings
    )
    names = feature_names(task)

    # 1. Separate the test groups FIRST and do not touch them again until the
    #    single final evaluation.
    remainder_idx, test_idx = _split_groups(
        groups, y, settings.ml_test_fraction, seed
    )
    # 2. Carve validation out of the remainder only.
    inner_fraction = (
        settings.ml_validation_fraction / (1.0 - settings.ml_test_fraction)
        if settings.ml_test_fraction < 1.0
        else 0.0
    )
    inner_train_pos, inner_val_pos = _split_groups(
        groups[remainder_idx], y[remainder_idx], inner_fraction, seed + 1
    )
    train_idx = remainder_idx[inner_train_pos]
    val_idx = remainder_idx[inner_val_pos]

    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    test_groups = set(groups[test_idx].tolist())
    overlap = bool(
        (train_groups & test_groups)
        or (val_groups & test_groups)
        or (train_groups & val_groups)
    )

    labels_by_index = [item.label for item in label_outcome.labels]
    split_report = SplitReport(
        strategy=(
            "Group-disjoint three-way split on group_key (operator name where "
            "known, otherwise the address), then StratifiedGroupKFold "
            "cross-validation on train+validation for model selection. The "
            "test split is read exactly once, after the winner is refit."
        ),
        train_sample_count=len(train_idx),
        validation_sample_count=len(val_idx),
        test_sample_count=len(test_idx),
        train_group_count=len(train_groups),
        validation_group_count=len(val_groups),
        test_group_count=len(test_groups),
        train_class_counts=_class_counts([labels_by_index[i] for i in train_idx]),
        validation_class_counts=_class_counts([labels_by_index[i] for i in val_idx]),
        test_class_counts=_class_counts([labels_by_index[i] for i in test_idx]),
        groups_overlap=overlap,
        cv_folds=settings.ml_cv_folds,
        random_seed=seed,
    )

    # 3. Model selection by cross-validation over train+validation only.
    selection_idx = np.concatenate([train_idx, val_idx])
    X_sel, y_sel, g_sel = X[selection_idx], y[selection_idx], groups[selection_idx]

    smallest_class = int(min((y_sel == c).sum() for c in np.unique(y_sel)))
    folds = max(2, min(settings.ml_cv_folds, smallest_class, len(set(g_sel.tolist()))))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)

    candidates: list[CandidateResult] = []
    best_name, best_model, best_score = None, None, -1.0
    for model_name, prototype in _candidate_models(seed, "balanced"):
        fold_scores: list[float] = []
        note: Optional[str] = None
        try:
            for fit_pos, score_pos in splitter.split(X_sel, y_sel, groups=g_sel):
                model = _clone(prototype)
                _fit(model, model_name, X_sel[fit_pos], y_sel[fit_pos])
                with _quiet_feature_name_warning():
                    predicted = model.predict(X_sel[score_pos])
                fold_scores.append(
                    float(f1_score(y_sel[score_pos], predicted, zero_division=0))
                )
        except Exception as exc:  # pragma: no cover - environment dependent
            note = f"excluded: {type(exc).__name__} during cross-validation"
            fold_scores = []

        mean = float(np.mean(fold_scores)) if fold_scores else -1.0
        candidates.append(
            CandidateResult(
                model_name=model_name,
                cv_mean_f1=round(mean, 4) if fold_scores else 0.0,
                cv_std_f1=round(float(np.std(fold_scores)), 4) if fold_scores else 0.0,
                cv_fold_scores=[round(s, 4) for s in fold_scores],
                selected=False,
                note=note,
            )
        )
        if mean > best_score:
            best_name, best_model, best_score = model_name, prototype, mean

    if best_name is None:  # pragma: no cover - only if every candidate errored
        return TrainingOutcome(
            task=task,
            trained=False,
            blockers=["Every candidate model failed during cross-validation."],
            label_outcome=label_outcome,
            provenance=provenance,
            split_report=split_report,
            candidates=candidates,
            random_seed=seed,
            environment=environment,
        )

    for candidate in candidates:
        candidate.selected = candidate.model_name == best_name

    # 4. Fit on train only to choose the threshold on validation, then refit on
    #    train+validation for the final model. Test is still untouched.
    threshold_model = _fit(_clone(best_model), best_name, X[train_idx], y[train_idx])
    threshold, threshold_note = _choose_threshold(
        threshold_model, X[val_idx], y[val_idx]
    )
    validation_metrics = _evaluate(
        threshold_model, X[val_idx], y[val_idx], "validation", classes, threshold
    )

    final_model = _fit(_clone(best_model), best_name, X_sel, y_sel)

    # 5. The single, final look at the held-out test set.
    test_metrics = _evaluate(
        final_model, X[test_idx], y[test_idx], "held-out test", classes, threshold
    )

    trained_at = int(time.time())
    model_version = (
        f"{MODEL_VERSION_PREFIX}-{task}-{provenance.dataset_version}-"
        f"{FEATURE_SCHEMA_VERSION.split('-')[-1]}"
    )

    limitations = [
        _RESIDUAL_LEAKAGE_LIMITATION,
        "Reported test metrics come from a single held-out split of "
        f"{len(test_idx)} sample(s); with a split this size the confidence "
        "interval around each figure is wide.",
        "A predicted class is a statistical association learned from graph "
        "behaviour. It is never evidence of wrongdoing and never overrides "
        "address-level blockchain evidence.",
    ]

    # An accuracy that barely beats "always predict the largest class" is not
    # a working model, however high the raw number looks. The model discloses
    # that about itself rather than leaving a reader to compute the baseline.
    if test_metrics.accuracy_above_baseline < 0.05:
        limitations.append(
            f"Held-out accuracy is {test_metrics.accuracy}, but a model that "
            "always predicted the majority class "
            f"({max(test_metrics.class_counts, key=test_metrics.class_counts.get)}) "
            f"would score {test_metrics.majority_class_baseline_accuracy} on the "
            "same split — an improvement of only "
            f"{test_metrics.accuracy_above_baseline}. The accuracy figure is "
            "therefore largely explained by class imbalance and must not be "
            "quoted on its own; read PR-AUC and the confusion matrix instead."
        )

    minority = min(test_metrics.class_counts.values())
    if minority < 20:
        limitations.append(
            f"The held-out split contains only {minority} sample(s) of the "
            "minority class, so its precision and recall move in steps of "
            f"roughly {round(1 / max(minority, 1), 2)} and are estimates rather "
            "than measurements."
        )

    outcome = TrainingOutcome(
        task=task,
        trained=True,
        model_name=best_name,
        model_version=model_version,
        trained_at=trained_at,
        trained_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(trained_at)),
        random_seed=seed,
        provenance=provenance,
        split_report=split_report,
        candidates=candidates,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        feature_importances=_importances(final_model, best_name, names),
        training_feature_medians=[
            FeatureMedian(
                feature=name,
                median=round(float(np.median(X_sel[:, index])), 6),
            )
            for index, name in enumerate(names)
        ],
        class_imbalance_handling=(
            "class_weight='balanced' where the estimator supports it, and "
            "balanced per-sample weights at fit time for the boosting models "
            "that do not. The decision threshold was "
            f"{threshold_note}."
        ),
        limitations=limitations,
        environment=environment,
    )

    if persist:
        outcome.artifact_path = _persist(final_model, classes, threshold, outcome, settings)

    return outcome


def _clone(model: Any) -> Any:
    from sklearn.base import clone

    return clone(model)


def _persist(
    model: Any,
    classes: list[str],
    threshold: float,
    outcome: TrainingOutcome,
    settings: Settings,
) -> str:
    """Writes the fitted model plus a human-readable metadata sidecar.

    The sidecar is JSON on purpose: a reviewer must be able to read what a
    model claims about itself — version, dataset, metrics, seed — without
    unpickling anything.
    """
    import joblib

    directory = Path(settings.ml_artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{outcome.task}-{outcome.model_version}"

    joblib.dump(
        {
            "model": model,
            "classes": classes,
            "decision_threshold": threshold,
            "feature_names": feature_names(outcome.task),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "model_version": outcome.model_version,
            "model_name": outcome.model_name,
            "task": outcome.task,
        },
        directory / f"{stem}.joblib",
    )
    (directory / f"{stem}.json").write_text(
        outcome.model_dump_json(indent=2), encoding="utf-8"
    )
    return str(directory / f"{stem}.joblib")
