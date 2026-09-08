import networkx as nx

from app.attribution.models import (
    AttributionResult,
    AttributionStatus,
    EvidenceTier,
    SeedSourceType,
    VASPCandidate,
)
from app.behavior.models import BehaviorPattern, PatternType
from app.ml.features import extract_wallet_features
from app.tracing.models import FundFlowHop, FundFlowPath, TraceResult

WALLET = "0xaaaa111111111111111111111111111111111a"
COUNTERPARTY_1 = "0xbbbb222222222222222222222222222222222b"
COUNTERPARTY_2 = "0xcccc333333333333333333333333333333333c"
COUNTERPARTY_3 = "0xdddd444444444444444444444444444444444d"


def make_graph_with_edges() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_edge(COUNTERPARTY_1, WALLET, key="0x1#0", timestamp=100, tx_hash="0x1")
    graph.add_edge(COUNTERPARTY_2, WALLET, key="0x2#0", timestamp=200, tx_hash="0x2")
    graph.add_edge(WALLET, COUNTERPARTY_3, key="0x3#0", timestamp=300, tx_hash="0x3")
    return graph


def empty_trace_result(source: str) -> TraceResult:
    return TraceResult(
        source=source,
        max_hops=4,
        max_paths=500,
        paths=[],
        edges_explored=0,
        paths_truncated=False,
        edges_limit_hit=False,
        notes=[],
    )


def empty_attribution_result(wallet: str) -> AttributionResult:
    return AttributionResult(
        wallet=wallet,
        status=AttributionStatus.NONE,
        candidates=[],
        max_hops=4,
        search_truncated=False,
        notes=[],
    )


def make_hop(hop_index, from_addr, to_addr, ts, tx_hash="0xhop"):
    return FundFlowHop(
        hop_index=hop_index,
        from_address=from_addr,
        to_address=to_addr,
        tx_hash=tx_hash,
        edge_key=f"{tx_hash}#{hop_index}",
        timestamp=ts,
        amount=1.0,
        asset="ETH",
        asset_type="NATIVE",
        transfer_type="TRANSFER",
        chain="ethereum",
    )


def make_candidate(tier: EvidenceTier) -> VASPCandidate:
    return VASPCandidate(
        vasp_name="DEMO_VASP",
        matched_address=COUNTERPARTY_3,
        entity_type="demo_exchange",
        chain="ethereum",
        source_type=SeedSourceType.SYNTHETIC_DEMO,
        seed_source="synthetic_demo_fixture",
        seed_confidence_note="test fixture",
        evidence_tier=tier,
        hop_distance=1 if tier == EvidenceTier.DIRECT else 2,
        path_addresses=[WALLET, COUNTERPARTY_3],
        tx_hashes=["0x1"],
        hop_timestamps=[100],
    )


def test_wallet_not_in_graph_returns_zeroed_features_without_crashing():
    graph = nx.MultiDiGraph()  # empty graph, wallet absent
    trace_result = empty_trace_result(WALLET)
    attribution = empty_attribution_result(WALLET)

    features = extract_wallet_features(graph, WALLET, trace_result, [], attribution)

    assert features.wallet == WALLET.lower()
    assert features.in_degree == 0
    assert features.out_degree == 0
    assert features.unique_in_counterparties == 0
    assert features.unique_out_counterparties == 0
    assert features.total_edge_count == 0
    assert features.path_count == 0
    assert features.max_hop_count == 0
    assert features.avg_hop_count == 0.0
    assert features.max_path_duration_seconds is None
    assert features.avg_path_duration_seconds is None
    assert features.candidate_count == 0


def test_degree_and_counterparty_counts_from_graph():
    graph = make_graph_with_edges()
    trace_result = empty_trace_result(WALLET)
    attribution = empty_attribution_result(WALLET)

    features = extract_wallet_features(graph, WALLET, trace_result, [], attribution)

    assert features.in_degree == 2  # from COUNTERPARTY_1, COUNTERPARTY_2
    assert features.out_degree == 1  # to COUNTERPARTY_3
    assert features.unique_in_counterparties == 2
    assert features.unique_out_counterparties == 1
    assert features.total_edge_count == 3


def test_hop_and_duration_stats_from_trace_result():
    graph = make_graph_with_edges()
    path1 = FundFlowPath(
        source=WALLET,
        terminal_node=COUNTERPARTY_3,
        hops=[make_hop(0, WALLET, COUNTERPARTY_3, 100, "0xa")],
    )
    path2 = FundFlowPath(
        source=WALLET,
        terminal_node=COUNTERPARTY_2,
        hops=[
            make_hop(0, WALLET, COUNTERPARTY_1, 100, "0xb"),
            make_hop(1, COUNTERPARTY_1, COUNTERPARTY_2, 250, "0xc"),
        ],
    )
    trace_result = TraceResult(
        source=WALLET,
        max_hops=4,
        max_paths=500,
        paths=[path1, path2],
        edges_explored=3,
        paths_truncated=False,
        edges_limit_hit=False,
        notes=[],
    )
    attribution = empty_attribution_result(WALLET)

    features = extract_wallet_features(graph, WALLET, trace_result, [], attribution)

    assert features.path_count == 2
    assert features.max_hop_count == 2  # path2 has 2 hops
    assert features.avg_hop_count == 1.5  # (1 + 2) / 2
    # path1 duration = 100-100 = 0; path2 duration = 250-100 = 150
    assert features.max_path_duration_seconds == 150.0
    assert features.avg_path_duration_seconds == 75.0
    assert features.paths_with_unknown_duration == 0


def test_missing_timestamp_produces_unknown_duration_not_zero_imputed():
    graph = make_graph_with_edges()
    hop_missing_ts = FundFlowHop(
        hop_index=0,
        from_address=WALLET,
        to_address=COUNTERPARTY_3,
        tx_hash="0xnoTS",
        edge_key="0xnoTS#0",
        timestamp=None,  # missing/invalid timestamp
        amount=1.0,
        asset="ETH",
        asset_type="NATIVE",
        transfer_type="TRANSFER",
        chain="ethereum",
    )
    path_with_unknown = FundFlowPath(
        source=WALLET, terminal_node=COUNTERPARTY_3, hops=[hop_missing_ts]
    )
    path_known = FundFlowPath(
        source=WALLET,
        terminal_node=COUNTERPARTY_1,
        hops=[make_hop(0, WALLET, COUNTERPARTY_1, 100, "0xok")],
    )
    trace_result = TraceResult(
        source=WALLET,
        max_hops=4,
        max_paths=500,
        paths=[path_with_unknown, path_known],
        edges_explored=2,
        paths_truncated=False,
        edges_limit_hit=False,
        notes=[],
    )
    attribution = empty_attribution_result(WALLET)

    features = extract_wallet_features(graph, WALLET, trace_result, [], attribution)

    assert features.path_count == 2
    assert features.paths_with_unknown_duration == 1
    # the one path with a known (zero) duration is what informs the stat,
    # NOT the missing one silently treated as zero
    assert features.max_path_duration_seconds == 0.0
    assert features.avg_path_duration_seconds == 0.0


def test_behavior_pattern_flags_reflect_present_patterns_only():
    graph = make_graph_with_edges()
    trace_result = empty_trace_result(WALLET)
    attribution = empty_attribution_result(WALLET)

    patterns = [
        BehaviorPattern(
            pattern_type=PatternType.SPLIT_PATTERN,
            wallet=WALLET,
            evidence=["evidence line"],
            metrics={"count": 5},
            related_addresses=[COUNTERPARTY_1],
        ),
        BehaviorPattern(
            pattern_type=PatternType.TEMPORAL_BURST,
            wallet=WALLET,
            evidence=["evidence line"],
            metrics={"count": 5},
            related_addresses=[COUNTERPARTY_2],
        ),
    ]

    features = extract_wallet_features(graph, WALLET, trace_result, patterns, attribution)

    assert features.has_split_pattern is True
    assert features.has_temporal_burst is True
    assert features.has_consolidation_pattern is False
    assert features.has_rapid_hopping is False
    assert features.has_high_frequency_counterparty is False
    assert features.has_repeated_forwarding is False
    assert features.behavior_pattern_count == 2


def test_attribution_flags_reflect_evidence_tiers_and_status():
    graph = make_graph_with_edges()
    trace_result = empty_trace_result(WALLET)

    attribution_direct = AttributionResult(
        wallet=WALLET,
        status=AttributionStatus.MATCH_FOUND,
        candidates=[make_candidate(EvidenceTier.DIRECT)],
        max_hops=4,
        search_truncated=False,
        notes=[],
    )
    features = extract_wallet_features(graph, WALLET, trace_result, [], attribution_direct)
    assert features.attribution_status == "MATCH_FOUND"
    assert features.has_direct_evidence is True
    assert features.has_indirect_evidence is False
    assert features.candidate_count == 1

    attribution_indirect = AttributionResult(
        wallet=WALLET,
        status=AttributionStatus.MATCH_FOUND,
        candidates=[make_candidate(EvidenceTier.INDIRECT)],
        max_hops=4,
        search_truncated=False,
        notes=[],
    )
    features2 = extract_wallet_features(graph, WALLET, trace_result, [], attribution_indirect)
    assert features2.has_direct_evidence is False
    assert features2.has_indirect_evidence is True

    attribution_none = empty_attribution_result(WALLET)
    features3 = extract_wallet_features(graph, WALLET, trace_result, [], attribution_none)
    assert features3.attribution_status == "NONE"
    assert features3.candidate_count == 0


def test_extraction_is_deterministic_for_identical_inputs():
    graph = make_graph_with_edges()
    trace_result = empty_trace_result(WALLET)
    attribution = empty_attribution_result(WALLET)

    f1 = extract_wallet_features(graph, WALLET, trace_result, [], attribution)
    f2 = extract_wallet_features(graph, WALLET, trace_result, [], attribution)

    assert f1.model_dump() == f2.model_dump()
