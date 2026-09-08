"""
MACRO MILESTONE 3 / PHASE B tests — behavioral pattern detection.

Synthetic graphs only, mirroring app/graph/builder.py's edge schema.
Fully offline.
"""

from __future__ import annotations

import networkx as nx

from app.behavior.detectors import (
    analyze_wallet_behavior,
    detect_fan_in,
    detect_fan_out,
    detect_high_frequency_counterparties,
    detect_rapid_hopping,
    detect_repeated_forwarding,
    detect_temporal_burst,
)
from app.behavior.models import PatternType
from app.core.config import Settings
from app.tracing.tracer import trace_fund_flow

WALLET_A = "0xaaaa111111111111111111111111111111111a"
WALLET_B = "0xbbbb222222222222222222222222222222222b"
WALLET_C = "0xcccc333333333333333333333333333333333c"
WALLET_D = "0xdddd444444444444444444444444444444444d"


def add_edge(graph, u, v, tx_hash, occurrence=0, amount=1.0, ts=100, asset="ETH", asset_type="NATIVE"):
    key = f"{tx_hash}#{occurrence}"
    graph.add_edge(
        u,
        v,
        key=key,
        amount=amount,
        asset=asset,
        asset_type=asset_type,
        timestamp=ts,
        tx_hash=tx_hash,
        transfer_index=occurrence,
        chain="ethereum",
        transfer_type="TRANSFER" if asset_type == "NATIVE" else "TOKEN_TRANSFER",
        status="SUCCESS",
    )
    return key


def low_threshold_settings(**overrides) -> Settings:
    """Small thresholds so tiny synthetic graphs can trigger patterns."""
    base = dict(
        etherscan_api_key="x",
        etherscan_base_url="https://example.invalid",
        etherscan_chain_id=1,
        max_transactions_per_investigation=100,
        default_lookback_days=90,
        http_timeout_seconds=5,
        http_max_retries=1,
        behavior_min_fanout_counterparties=3,
        behavior_min_fanin_counterparties=3,
        behavior_high_frequency_min_transfers=3,
        behavior_rapid_hop_max_seconds=300,
        behavior_burst_window_seconds=3600,
        behavior_burst_min_transfers=3,
        behavior_forwarding_window_seconds=3600,
        behavior_min_forwarding_events=2,
    )
    base.update(overrides)
    return Settings(**base)


# --- fan-out ------------------------------------------------------------


def test_fan_out_detected_above_threshold():
    g = nx.MultiDiGraph()
    for i, dest in enumerate([WALLET_B, WALLET_C, WALLET_D]):
        add_edge(g, WALLET_A, dest, f"0x{i}", ts=100 + i)
    settings = low_threshold_settings()
    pattern = detect_fan_out(g, WALLET_A, settings)
    assert pattern is not None
    assert pattern.pattern_type == PatternType.SPLIT_PATTERN
    assert pattern.metrics["unique_outgoing_counterparties"] == 3
    assert set(pattern.related_addresses) == {WALLET_B, WALLET_C, WALLET_D}


def test_fan_out_not_flagged_when_same_counterparty_repeated():
    """Many outgoing TXs to the SAME counterparty must NOT be a fan-out."""
    g = nx.MultiDiGraph()
    for i in range(5):
        add_edge(g, WALLET_A, WALLET_B, f"0x{i}", ts=100 + i)
    settings = low_threshold_settings()
    pattern = detect_fan_out(g, WALLET_A, settings)
    assert pattern is None


def test_fan_out_below_threshold_not_flagged():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=100)
    settings = low_threshold_settings()
    assert detect_fan_out(g, WALLET_A, settings) is None


# --- fan-in ---------------------------------------------------------------


def test_fan_in_detected_above_threshold():
    g = nx.MultiDiGraph()
    for i, src in enumerate([WALLET_B, WALLET_C, WALLET_D]):
        add_edge(g, src, WALLET_A, f"0x{i}", ts=100 + i)
    settings = low_threshold_settings()
    pattern = detect_fan_in(g, WALLET_A, settings)
    assert pattern is not None
    assert pattern.pattern_type == PatternType.CONSOLIDATION_PATTERN
    assert pattern.metrics["unique_incoming_counterparties"] == 3


# --- high-frequency counterparty -------------------------------------------


def test_high_frequency_counterparty_detected():
    g = nx.MultiDiGraph()
    for i in range(4):
        add_edge(g, WALLET_A, WALLET_B, f"0x{i}", ts=100 + i)
    settings = low_threshold_settings()
    patterns = detect_high_frequency_counterparties(g, WALLET_A, settings)
    assert len(patterns) == 1
    assert patterns[0].pattern_type == PatternType.HIGH_FREQUENCY_COUNTERPARTY
    assert patterns[0].related_addresses == [WALLET_B]
    assert patterns[0].metrics["transfer_count"] == 4


def test_high_frequency_counterparty_counts_both_directions():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=100)
    add_edge(g, WALLET_B, WALLET_A, "0x2", ts=200)
    add_edge(g, WALLET_A, WALLET_B, "0x3", ts=300)
    settings = low_threshold_settings()
    patterns = detect_high_frequency_counterparties(g, WALLET_A, settings)
    assert len(patterns) == 1
    assert patterns[0].metrics["transfer_count"] == 3


# --- temporal burst ---------------------------------------------------------


def test_temporal_burst_detected():
    g = nx.MultiDiGraph()
    for i in range(4):
        add_edge(g, WALLET_A, WALLET_B, f"0x{i}", ts=1000 + i * 10)  # tight cluster
    settings = low_threshold_settings(behavior_burst_window_seconds=100)
    pattern = detect_temporal_burst(g, WALLET_A, settings)
    assert pattern is not None
    assert pattern.pattern_type == PatternType.TEMPORAL_BURST
    assert pattern.metrics["burst_transfer_count"] == 4


def test_temporal_burst_not_flagged_when_spread_out():
    g = nx.MultiDiGraph()
    for i in range(4):
        add_edge(g, WALLET_A, WALLET_B, f"0x{i}", ts=1000 + i * 100000)  # spread far apart
    settings = low_threshold_settings(behavior_burst_window_seconds=60)
    pattern = detect_temporal_burst(g, WALLET_A, settings)
    assert pattern is None


# --- repeated forwarding -----------------------------------------------------


def test_repeated_forwarding_detected():
    g = nx.MultiDiGraph()
    # Receive from B, forward to C and D shortly after (twice)
    add_edge(g, WALLET_B, WALLET_A, "0xin1", ts=1000)
    add_edge(g, WALLET_A, WALLET_C, "0xout1", ts=1010)
    add_edge(g, WALLET_B, WALLET_A, "0xin2", ts=2000)
    add_edge(g, WALLET_A, WALLET_D, "0xout2", ts=2010)
    settings = low_threshold_settings()
    pattern = detect_repeated_forwarding(g, WALLET_A, settings)
    assert pattern is not None
    assert pattern.pattern_type == PatternType.REPEATED_FORWARDING
    assert pattern.metrics["forwarding_event_count"] >= 2


def test_repeated_forwarding_minimal_case_not_flagged():
    """One incoming + one outgoing transfer is NOT enough evidence."""
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_B, WALLET_A, "0xin1", ts=1000)
    add_edge(g, WALLET_A, WALLET_C, "0xout1", ts=1010)
    settings = low_threshold_settings()
    pattern = detect_repeated_forwarding(g, WALLET_A, settings)
    assert pattern is None


def test_repeated_forwarding_ignores_forwarding_to_same_counterparty():
    """Sending money back to the SAME wallet that sent it is not forwarding."""
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_B, WALLET_A, "0xin1", ts=1000)
    add_edge(g, WALLET_A, WALLET_B, "0xout1", ts=1010)
    add_edge(g, WALLET_B, WALLET_A, "0xin2", ts=2000)
    add_edge(g, WALLET_A, WALLET_B, "0xout2", ts=2010)
    settings = low_threshold_settings()
    pattern = detect_repeated_forwarding(g, WALLET_A, settings)
    assert pattern is None


# --- rapid hopping (path-based) ---------------------------------------------


def test_rapid_hopping_detected_from_trace_path():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=1000)
    add_edge(g, WALLET_B, WALLET_C, "0x2", ts=1050)
    result = trace_fund_flow(g, WALLET_A, max_hops=2)
    two_hop = [p for p in result.paths if p.hop_count == 2][0]
    settings = low_threshold_settings(behavior_rapid_hop_max_seconds=300)
    pattern = detect_rapid_hopping(two_hop, settings)
    assert pattern is not None
    assert pattern.pattern_type == PatternType.RAPID_HOPPING
    assert pattern.metrics["hop_count"] == 2


def test_rapid_hopping_not_flagged_when_gap_too_large():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=1000)
    add_edge(g, WALLET_B, WALLET_C, "0x2", ts=100000)  # huge gap
    result = trace_fund_flow(g, WALLET_A, max_hops=2)
    two_hop = [p for p in result.paths if p.hop_count == 2][0]
    settings = low_threshold_settings(behavior_rapid_hop_max_seconds=300)
    pattern = detect_rapid_hopping(two_hop, settings)
    assert pattern is None


def test_rapid_hopping_not_flagged_with_missing_timestamp():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=None)
    add_edge(g, WALLET_B, WALLET_C, "0x2", ts=None)
    result = trace_fund_flow(g, WALLET_A, max_hops=2)
    two_hop = [p for p in result.paths if p.hop_count == 2]
    if two_hop:
        settings = low_threshold_settings()
        assert detect_rapid_hopping(two_hop[0], settings) is None


def test_rapid_hopping_requires_at_least_two_hops():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=1000)
    result = trace_fund_flow(g, WALLET_A, max_hops=1)
    one_hop = result.paths[0]
    settings = low_threshold_settings()
    assert detect_rapid_hopping(one_hop, settings) is None


# --- parallel edges don't break detectors -----------------------------------


def test_parallel_edges_counted_correctly_in_fan_out():
    g = nx.MultiDiGraph()
    # 3 transfers to the SAME destination (parallel edges) should count as
    # 1 unique counterparty, not 3 -> should NOT trigger fan-out at threshold 3
    add_edge(g, WALLET_A, WALLET_B, "0x1", occurrence=0, ts=100)
    add_edge(g, WALLET_A, WALLET_B, "0x1", occurrence=1, ts=100)
    add_edge(g, WALLET_A, WALLET_B, "0x1", occurrence=2, ts=100)
    settings = low_threshold_settings(behavior_min_fanout_counterparties=3)
    assert detect_fan_out(g, WALLET_A, settings) is None
    # but IS high-frequency-counterparty
    patterns = detect_high_frequency_counterparties(g, WALLET_A, settings)
    assert len(patterns) == 1
    assert patterns[0].metrics["transfer_count"] == 3


# --- empty graph / missing wallet -------------------------------------------


def test_empty_graph_all_detectors_return_none_or_empty():
    g = nx.MultiDiGraph()
    settings = low_threshold_settings()
    assert detect_fan_out(g, WALLET_A, settings) is None
    assert detect_fan_in(g, WALLET_A, settings) is None
    assert detect_high_frequency_counterparties(g, WALLET_A, settings) == []
    assert detect_temporal_burst(g, WALLET_A, settings) is None
    assert detect_repeated_forwarding(g, WALLET_A, settings) is None


def test_wallet_not_in_graph_does_not_crash():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_B, WALLET_C, "0x1", ts=100)
    settings = low_threshold_settings()
    assert detect_fan_out(g, WALLET_A, settings) is None
    assert detect_fan_in(g, WALLET_A, settings) is None


# --- deterministic output ----------------------------------------------------


def test_analyze_wallet_behavior_deterministic():
    g = nx.MultiDiGraph()
    for i, dest in enumerate([WALLET_B, WALLET_C, WALLET_D]):
        add_edge(g, WALLET_A, dest, f"0x{i}", ts=100 + i)
    settings = low_threshold_settings()

    patterns1 = analyze_wallet_behavior(g, WALLET_A, settings)
    patterns2 = analyze_wallet_behavior(g, WALLET_A, settings)

    types1 = sorted(p.pattern_type for p in patterns1)
    types2 = sorted(p.pattern_type for p in patterns2)
    assert types1 == types2


def test_analyze_wallet_behavior_includes_rapid_hopping_when_paths_given():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=1000)
    add_edge(g, WALLET_B, WALLET_C, "0x2", ts=1050)
    settings = low_threshold_settings()
    result = trace_fund_flow(g, WALLET_A, max_hops=2, settings=settings)
    patterns = analyze_wallet_behavior(g, WALLET_A, settings, paths=result.paths)
    assert any(p.pattern_type == PatternType.RAPID_HOPPING for p in patterns)


# --- no false certainty: labels never appear -------------------------------


def test_no_forbidden_labels_anywhere_in_output():
    g = nx.MultiDiGraph()
    for i, dest in enumerate([WALLET_B, WALLET_C, WALLET_D]):
        add_edge(g, WALLET_A, dest, f"0x{i}", ts=100 + i)
    settings = low_threshold_settings()
    patterns = analyze_wallet_behavior(g, WALLET_A, settings)
    forbidden = ["criminal", "fraud", "launder", "sanction", "binance", "coinbase"]
    for pattern in patterns:
        text = " ".join(pattern.evidence).lower() + str(pattern.metrics).lower()
        for word in forbidden:
            assert word not in text
