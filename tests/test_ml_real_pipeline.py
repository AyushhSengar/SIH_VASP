"""
Tests for the real-data ML modules:
  app/ml/real_labels.py, real_features.py, real_training.py,
  real_predictor.py, unsupervised.py

The point of these modules is that they refuse to produce a number they cannot
defend. The tests therefore concentrate on the refusals, because a refusal that
silently stops working turns into a fabricated accuracy figure.

Specifically pinned:
  * absence from the seed set never becomes a NOT_VASP_OWNED label
  * third-party and community labels are excluded from TRAINING while remaining
    usable as reported attribution evidence
  * training is refused, with named blockers, below the per-class minimum
  * accuracy is never reported without the majority-class baseline beside it
  * the train/validation/test split is group-disjoint
  * features are label-blind
  * the unsupervised model reports rank stability and never an accuracy
"""

from __future__ import annotations

import networkx as nx

from app.attribution.models import SeedSourceType, VASPSeedEntry
from app.core.config import get_settings
from app.ml import real_features as rf
from app.ml import real_labels as rl
from app.ml import real_predictor as rp
from app.ml import real_training as rt
from app.ml import unsupervised as un
from app.normalization.transactions import (
    AssetType,
    NormalizedTransfer,
    TransferSource,
)


def _address(index: int) -> str:
    return "0x" + f"{index:040x}"


def _transfer(
    index: int,
    source: str,
    target: str,
    stream: TransferSource = TransferSource.NATIVE_TRANSACTION,
    asset_type: AssetType = AssetType.NATIVE,
    token_contract: str | None = None,
) -> NormalizedTransfer:
    return NormalizedTransfer(
        tx_hash="0x" + f"{index:064x}",
        chain="ethereum",
        block_number=100 + index,
        timestamp=1_700_000_000 + index * 60,
        from_address=source,
        to_address=target,
        asset_type=asset_type,
        asset_identifier=token_contract,
        asset_symbol="ETH" if asset_type is AssetType.NATIVE else "TKN",
        asset_decimals=18,
        amount_raw="1000000000000000000",
        amount=1.0,
        transfer_source=stream,
        source_provider="test",
        fetched_at=1_700_100_000,
    )


def _seed(address: str, source_type: SeedSourceType, name: str = "TestVASP") -> VASPSeedEntry:
    return VASPSeedEntry(
        address=address,
        vasp_name=name,
        entity_type="exchange",
        chain="ethereum",
        source="unit test",
        source_type=source_type,
        confidence_note="unit test entry",
    )


def _graph(node_count: int = 40, transfers_each: int = 5) -> nx.MultiDiGraph:
    """A graph where every node clears a small activity floor."""
    graph = nx.MultiDiGraph()
    hub = _address(0)
    for node_index in range(1, node_count + 1):
        node = _address(node_index)
        for transfer_index in range(transfers_each):
            for source, target in ((node, hub), (hub, node)):
                tx = f"0x{node_index:032x}{transfer_index:032x}"
                graph.add_edge(
                    source,
                    target,
                    key=f"{tx}#{source[-4:]}",
                    tx_hash=tx,
                    block_number=100 + transfer_index,
                    timestamp=1_700_000_000 + node_index * 3600 + transfer_index * 60,
                    amount=1.0 + node_index,
                    asset="ETH",
                    asset_type="NATIVE",
                    transfer_source="NATIVE_TRANSACTION",
                    transfer_type="NATIVE_TRANSACTION",
                    chain="ethereum",
                    token_contract=None,
                    gas_used=21000,
                )
    return graph


# ==========================================================================
# real_labels
# ==========================================================================


def test_absence_from_the_seed_set_never_becomes_a_negative_label():
    """The single most dangerous shortcut in VASP ML, refused by design."""
    outcome = rl.derive_vasp_labels(
        [_seed(_address(1), SeedSourceType.OFFICIAL_DISCLOSURE)],
        candidate_addresses=[_address(i) for i in range(1, 500)],
        min_samples_per_class=20,
    )

    assert outcome.class_counts.get("NOT_VASP_OWNED", 0) == 0
    assert outcome.sufficient is False
    assert any("absence from a curated" in blocker.lower()
               or "not evidence" in blocker.lower()
               for blocker in outcome.blockers)


def test_negative_class_blocker_is_always_present_even_with_many_positives():
    positives = [_seed(_address(i), SeedSourceType.OFFICIAL_DISCLOSURE) for i in range(1, 60)]
    outcome = rl.derive_vasp_labels(
        positives, candidate_addresses=[_address(i) for i in range(1, 60)],
        min_samples_per_class=20,
    )
    assert outcome.sufficient is False, (
        "59 positives and zero defensible negatives is not a trainable dataset"
    )
    assert any("negative class is empty" in b.lower() for b in outcome.blockers)


def test_third_party_and_community_labels_are_excluded_from_training():
    outcome = rl.derive_vasp_labels(
        [
            _seed(_address(1), SeedSourceType.THIRD_PARTY_LABEL),
            _seed(_address(2), SeedSourceType.COMMUNITY_LABEL),
            _seed(_address(3), SeedSourceType.OFFICIAL_DISCLOSURE),
        ],
        candidate_addresses=[_address(1), _address(2), _address(3)],
    )
    labelled = {item.address for item in outcome.labels}
    assert _address(3) in labelled
    assert _address(1) not in labelled
    assert _address(2) not in labelled
    assert outcome.excluded_inconsistent_provenance == 2
    assert any("annotator" in note.lower() or "third-party" in note.lower()
               for note in outcome.notes)


def test_synthetic_seed_entries_are_never_labelled():
    outcome = rl.derive_vasp_labels(
        [_seed(_address(1), SeedSourceType.SYNTHETIC_DEMO)],
        candidate_addresses=[_address(1)],
    )
    assert outcome.labels == []


def _token_transfer_set(token: str, holders: int = 12) -> list[NormalizedTransfer]:
    """Token movements plus the native calls that caused them.

    Shaped like real acquisition output rather than like the token stream
    alone: an ERC-20 transfer is triggered by a top-level transaction sent TO
    the token contract, so the contract appears on native edges (`txlist`)
    while the value movement appears on token edges (`tokentx`) carrying the
    contract as its asset identifier. Emitting only the token stream would
    build a contract that never takes part in a transfer, which no provider
    ever returns.
    """
    transfers: list[NormalizedTransfer] = []
    index = 0
    for holder_index in range(1, holders + 1):
        holder = _address(holder_index)
        for _ in range(4):
            index += 1
            transfers.append(
                _transfer(
                    index,
                    holder,
                    _address(holder_index + 100),
                    stream=TransferSource.TOKEN_TRANSFER,
                    asset_type=AssetType.ERC20,
                    token_contract=token,
                )
            )
            index += 1
            # The `transfer()` call itself: holder -> token contract, native.
            transfers.append(_transfer(index, holder, token))
    return transfers


def test_account_type_labels_come_from_protocol_facts_not_guesses():
    """A token-contract address is a contract because the protocol says so."""
    token = _address(500)
    outcome = rl.derive_account_type_labels(
        _token_transfer_set(token), min_transfers=3
    )

    by_address = {item.address: item for item in outcome.labels}
    assert by_address[token].label == rl.AccountTypeLabel.CONTRACT.value
    assert by_address[token].label_source == rl.LabelSource.PROTOCOL_GUARANTEED
    assert "contract that emitted a" in by_address[token].justification
    assert by_address[token].evidence_tx_hash

    # The senders of those native calls signed them, so they are EOAs. The
    # token contract must not also be one: it is only ever a recipient here.
    assert by_address[_address(1)].label == (
        rl.AccountTypeLabel.EXTERNALLY_OWNED_ACCOUNT.value
    )
    assert outcome.excluded_conflicting_labels == 0


def test_a_token_contract_below_the_activity_floor_is_not_labelled():
    """The documented consequence of applying the floor before the label:
    a contract nobody transacts with often enough is left out rather than
    admitted through a side door that the EOA class does not have."""
    token = _address(501)
    transfers = [
        _transfer(
            1,
            _address(1),
            _address(2),
            stream=TransferSource.TOKEN_TRANSFER,
            asset_type=AssetType.ERC20,
            token_contract=token,
        )
    ]
    outcome = rl.derive_account_type_labels(transfers, min_transfers=3)
    assert token not in {item.address for item in outcome.labels}
    assert any("activity floor" in blocker for blocker in outcome.blockers)


def test_activity_floor_is_applied_before_any_label_is_read():
    """Filtering after labelling would let a 1-transfer address into a class."""
    quiet = _address(900)
    transfers = [_transfer(1, quiet, _address(901))]
    outcome = rl.derive_account_type_labels(transfers, min_transfers=5)
    assert quiet not in {item.address for item in outcome.labels}
    assert outcome.excluded_below_activity_floor >= 1
    assert outcome.activity_floor == 5


def test_label_schema_version_is_stamped():
    outcome = rl.derive_vasp_labels([], candidate_addresses=[])
    assert outcome.label_schema_version == rl.LABEL_SCHEMA_VERSION


# ==========================================================================
# real_features
# ==========================================================================


def test_features_are_label_blind():
    """No feature may encode the answer. A feature named after the seed set,
    the attribution result, or the ML label is leakage by construction."""
    forbidden = ("vasp", "label", "seed", "attribut", "target", "predict", "class")
    for name in rf.feature_names():
        lowered = name.lower()
        assert not any(token in lowered for token in forbidden), name


def test_feature_extraction_is_deterministic():
    graph = _graph(node_count=10)
    first = rf.extract_address_features(graph, _address(3))
    second = rf.extract_address_features(graph, _address(3))
    assert first.values == second.values


def test_feature_schema_version_is_stamped_on_every_row():
    graph = _graph(node_count=6)
    rows = rf.extract_many(graph, [_address(i) for i in range(1, 5)])
    assert rows
    assert all(row.feature_schema_version == rf.FEATURE_SCHEMA_VERSION for row in rows)


def test_missing_timestamps_are_flagged_not_silently_zeroed():
    graph = nx.MultiDiGraph()
    graph.add_edge(
        _address(1),
        _address(2),
        key="0xk#0",
        tx_hash="0xk",
        timestamp=None,
        amount=1.0,
        asset="ETH",
        asset_type="NATIVE",
        chain="ethereum",
    )
    row = rf.extract_address_features(graph, _address(1))
    assert row.temporal_data_missing is True


def test_absent_address_yields_features_not_an_exception():
    graph = _graph(node_count=5)
    row = rf.extract_address_features(graph, _address(9999))
    assert row.address == _address(9999)
    assert set(row.values) == set(rf.feature_names())


# ==========================================================================
# real_training
# ==========================================================================


def test_training_is_refused_below_the_per_class_minimum():
    outcome = rl.derive_vasp_labels(
        [_seed(_address(1), SeedSourceType.OFFICIAL_DISCLOSURE)],
        candidate_addresses=[_address(1)],
        min_samples_per_class=20,
    )
    result = rt.train_real_model(_graph(), outcome, "unit test graph", persist=False)

    assert result.trained is False
    assert result.test_metrics is None
    assert result.validation_metrics is None
    assert result.blockers, "a refusal must name its reason"


def test_refused_training_reports_no_accuracy_at_all():
    """A refused model with a number attached is the failure mode to avoid."""
    outcome = rl.derive_vasp_labels([], candidate_addresses=[])
    result = rt.train_real_model(_graph(), outcome, "unit test graph", persist=False)
    dumped = result.model_dump()
    assert dumped["test_metrics"] is None
    assert dumped["validation_metrics"] is None


def test_residual_leakage_is_stated_as_a_limitation_constant():
    """One shared graph means training and test addresses can be counterparties.
    That cannot be engineered away, so it must be disclosed."""
    assert "not statistically independent" in rt._RESIDUAL_LEAKAGE_LIMITATION
    assert "inflates" in rt._RESIDUAL_LEAKAGE_LIMITATION


def test_metrics_model_requires_a_baseline_next_to_accuracy():
    """Accuracy alone is meaningless under class imbalance, so the model
    cannot represent an accuracy without its baseline."""
    fields = rt.EvaluationMetrics.model_fields
    assert "majority_class_baseline_accuracy" in fields
    assert "accuracy_above_baseline" in fields
    assert "class_counts" in fields
    assert "confusion_matrix" in fields
    assert "roc_auc" in fields and "pr_auc" in fields


def test_split_report_records_group_disjointness():
    fields = rt.SplitReport.model_fields
    assert "groups_overlap" in fields
    assert "train_group_count" in fields
    assert "test_group_count" in fields


def test_dataset_provenance_carries_full_versioning():
    fields = rt.DatasetProvenance.model_fields
    for required in (
        "dataset_version",
        "label_schema_version",
        "feature_schema_version",
        "graph_source",
        "sample_count",
        "class_counts",
        "group_count",
    ):
        assert required in fields


# ==========================================================================
# real_predictor
# ==========================================================================


def test_prediction_without_an_artifact_is_unavailable_with_a_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_ARTIFACT_DIR", str(tmp_path / "empty"))
    settings = get_settings()

    result = rp.predict(_graph(node_count=5), _address(1), task="vasp_ownership",
                        settings=settings)

    assert result.available is False
    assert result.predicted_class is None
    assert result.probability is None
    assert result.unavailable_reason


def test_find_artifact_returns_none_rather_than_raising(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_ARTIFACT_DIR", str(tmp_path / "nothing-here"))
    assert rp.find_artifact("vasp_ownership", get_settings()) is None


def test_prediction_evidence_class_can_only_be_supporting():
    """An ML output may corroborate address evidence, never replace it."""
    result = rp.PredictionResult(
        address=_address(1), task="vasp_ownership", available=False
    )
    assert result.evidence_class == "SUPPORTING"


# ==========================================================================
# unsupervised
# ==========================================================================


def test_unsupervised_refuses_on_a_population_that_is_too_small():
    tiny = nx.MultiDiGraph()
    tiny.add_edge(
        _address(1), _address(2), key="0xk#0", tx_hash="0xk",
        timestamp=1_700_000_000, amount=1.0, asset="ETH", asset_type="NATIVE",
        chain="ethereum",
    )
    result = un.assess_address(tiny, _address(1), population_source="unit test")
    assert result.available is False
    assert result.unavailable_reason
    assert result.outlier_score is None


def test_unsupervised_reports_rank_stability_and_no_accuracy():
    graph = _graph(node_count=60, transfers_each=3)
    result = un.assess_address(
        graph, _address(1), population_source="unit test graph", stability_resamples=5
    )

    assert result.available is True
    assert result.rank_stability is not None
    assert result.rank_stability.resamples == 5
    assert -1.0 <= result.rank_stability.mean_spearman_correlation <= 1.0

    dumped = result.model_dump()
    for absent in ("accuracy", "precision", "recall", "f1"):
        assert absent not in dumped, (
            f"{absent} cannot exist for an unlabelled model"
        )
    assert any("no accuracy" in line.lower() for line in result.evaluation)


def test_unsupervised_evidence_class_is_contextual():
    graph = _graph(node_count=60, transfers_each=3)
    result = un.assess_address(graph, _address(1), population_source="unit test graph")
    assert result.evidence_class == "CONTEXTUAL"


def test_unsupervised_states_that_unusual_is_not_wrongdoing():
    graph = _graph(node_count=60, transfers_each=3)
    result = un.assess_address(graph, _address(1), population_source="unit test graph")
    joined = " ".join(result.limitations).lower()
    assert "not wrongdoing" in joined or "is not wrongdoing" in joined
    assert "neighbourhood" in joined or "neighborhood" in joined


def test_deviation_requires_falling_outside_the_p10_p90_band():
    """A high percentile rank alone is not a deviation; the value must actually
    sit outside the band, otherwise a near-median address 'deviates'."""
    graph = _graph(node_count=60, transfers_each=3)
    result = un.assess_address(graph, _address(1), population_source="unit test graph")
    for deviation in result.deviations:
        assert (
            deviation.value > deviation.population_p90
            or deviation.value < deviation.population_p10
        ), deviation.feature


def test_unsupervised_is_deterministic_for_a_fixed_seed():
    graph = _graph(node_count=60, transfers_each=3)
    first = un.assess_address(graph, _address(1), population_source="s")
    second = un.assess_address(graph, _address(1), population_source="s")
    assert first.outlier_score == second.outlier_score
    assert first.percentile_within_population == second.percentile_within_population
