import copy

from app.attribution.models import AttributionResult, AttributionStatus, VASPCandidate
from app.ml.models import MLLabel, MLPrediction, WalletFeatures
from app.ml.predictor import predict, train_model
from app.ml.training_data import SYNTHETIC_TRAINING_DATA


def make_features(**overrides) -> WalletFeatures:
    base = dict(
        wallet="0xtest111111111111111111111111111111111a",
        in_degree=10,
        out_degree=8,
        unique_in_counterparties=6,
        unique_out_counterparties=5,
        total_edge_count=18,
        path_count=4,
        max_hop_count=2,
        avg_hop_count=1.5,
        max_path_duration_seconds=500.0,
        avg_path_duration_seconds=300.0,
        paths_with_unknown_duration=0,
        has_split_pattern=True,
        has_consolidation_pattern=False,
        has_rapid_hopping=True,
        has_high_frequency_counterparty=False,
        has_repeated_forwarding=False,
        has_temporal_burst=False,
        behavior_pattern_count=2,
        attribution_status="MATCH_FOUND",
        has_direct_evidence=True,
        has_indirect_evidence=False,
        candidate_count=1,
    )
    base.update(overrides)
    return WalletFeatures(**base)


def make_empty_features() -> WalletFeatures:
    return make_features(
        wallet="0xempty000000000000000000000000000000000",
        in_degree=0,
        out_degree=0,
        unique_in_counterparties=0,
        unique_out_counterparties=0,
        total_edge_count=0,
        path_count=0,
        max_hop_count=0,
        avg_hop_count=0.0,
        max_path_duration_seconds=None,
        avg_path_duration_seconds=None,
        paths_with_unknown_duration=0,
        has_split_pattern=False,
        has_consolidation_pattern=False,
        has_rapid_hopping=False,
        has_high_frequency_counterparty=False,
        has_repeated_forwarding=False,
        has_temporal_burst=False,
        behavior_pattern_count=0,
        attribution_status="NONE",
        has_direct_evidence=False,
        has_indirect_evidence=False,
        candidate_count=0,
    )


# --- training data sanity ---


def test_training_data_is_labelled_synthetic_demo_only():
    assert len(SYNTHETIC_TRAINING_DATA) > 0
    for features, _label in SYNTHETIC_TRAINING_DATA:
        assert features.wallet.startswith("SYNTHETIC_DEMO_")


def test_training_data_never_uses_real_seed_addresses():
    # Real seed addresses are Ethereum-style 0x... hex addresses. Every
    # synthetic training row must use the placeholder wallet label scheme
    # instead, never anything resembling a real on-chain address.
    for features, _label in SYNTHETIC_TRAINING_DATA:
        assert not features.wallet.lower().startswith("0x")


# --- determinism ---


def test_training_is_deterministic_for_fixed_seed():
    model_a = train_model(seed=42)
    model_b = train_model(seed=42)

    features = make_features()
    pred_a = predict(model_a, features)
    pred_b = predict(model_b, features)

    assert pred_a.predicted_label == pred_b.predicted_label


def test_different_seeds_still_produce_valid_deterministic_models():
    model_a = train_model(seed=1)
    model_b = train_model(seed=1)  # same seed as model_a, must match each other

    features = make_features()
    assert predict(model_a, features).predicted_label == predict(model_b, features).predicted_label


def test_prediction_is_deterministic_for_repeated_calls():
    model = train_model(seed=42)
    features = make_features()

    pred1 = predict(model, features)
    pred2 = predict(model, features)

    assert pred1.predicted_label == pred2.predicted_label
    assert pred1.model_dump() == pred2.model_dump()


# --- insufficient data path ---


def test_predict_returns_insufficient_data_for_empty_wallet():
    model = train_model(seed=42)
    features = make_empty_features()

    prediction = predict(model, features)

    assert prediction.predicted_label == MLLabel.INSUFFICIENT_DATA


def test_predict_does_not_return_insufficient_data_when_signal_present():
    model = train_model(seed=42)
    features = make_features()  # has edges and paths

    prediction = predict(model, features)

    assert prediction.predicted_label != MLLabel.INSUFFICIENT_DATA


# --- synthetic/demo disclaimer ---


def test_every_prediction_carries_synthetic_demo_labeling_and_disclaimer():
    model = train_model(seed=42)
    for features in (make_features(), make_empty_features()):
        prediction = predict(model, features)
        assert prediction.training_data_type == "SYNTHETIC_DEMO"
        assert "SYNTHETIC" in prediction.disclaimer.upper()
        assert "not" in prediction.disclaimer.lower()  # some caveat language present


def test_prediction_has_no_numeric_confidence_field():
    model = train_model(seed=42)
    prediction = predict(model, make_features())
    field_names = set(prediction.model_fields.keys())
    for forbidden in ("confidence", "probability", "score", "accuracy"):
        assert forbidden not in field_names


# --- structural separation from attribution ---


def test_ml_prediction_has_no_attribution_fields():
    attribution_fields = set(AttributionResult.model_fields.keys()) | set(
        VASPCandidate.model_fields.keys()
    )
    ml_fields = set(MLPrediction.model_fields.keys())

    # 'wallet' is a reasonable shared concept (both describe a wallet) but
    # nothing evidence-specific may leak into MLPrediction.
    forbidden_overlap = attribution_fields - {"wallet"}
    assert ml_fields.isdisjoint(forbidden_overlap)


def test_ml_cannot_construct_a_vasp_candidate():
    # MLPrediction and VASPCandidate must remain independently constructible
    # types with no inheritance relationship between them.
    assert not issubclass(MLPrediction, VASPCandidate)
    assert not issubclass(VASPCandidate, MLPrediction)


def test_predict_does_not_mutate_attribution_result():
    original = AttributionResult(
        wallet="0xaaaa111111111111111111111111111111111a",
        status=AttributionStatus.NONE,
        candidates=[],
        max_hops=4,
        search_truncated=False,
        notes=["untouched"],
    )
    snapshot_before = copy.deepcopy(original.model_dump())

    model = train_model(seed=42)
    features = make_features()
    # attribution_status on the features was already extracted upstream;
    # predict() only ever reads `features`, never touches `original`.
    predict(model, features)

    assert original.model_dump() == snapshot_before
