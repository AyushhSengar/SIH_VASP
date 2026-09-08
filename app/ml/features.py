"""
MACRO MILESTONE 5 — feature extraction.

Pure, deterministic, offline: extract_wallet_features() only reads from
objects the caller already has (a Milestone-2 graph, a Milestone-3
TraceResult, a list of Milestone-3 BehaviorPatterns, a Milestone-4
AttributionResult). It never fetches data, never mutates any of its
inputs, and never invents a feature not already derivable from M1-M4.

Every field on WalletFeatures traces to a specific milestone — see
app/ml/models.py's per-field comments for exactly which one.
"""

from __future__ import annotations

import networkx as nx

from app.attribution.models import AttributionResult, EvidenceTier
from app.behavior.models import BehaviorPattern, PatternType
from app.ml.models import WalletFeatures
from app.tracing.models import TraceResult


def extract_wallet_features(
    graph: nx.MultiDiGraph,
    wallet: str,
    trace_result: TraceResult,
    behavior_patterns: list[BehaviorPattern],
    attribution_result: AttributionResult,
) -> WalletFeatures:
    wallet = wallet.lower()

    # --- graph-structural ---
    if wallet in graph:
        out_edges = list(graph.out_edges(wallet, keys=True, data=True))
        in_edges = list(graph.in_edges(wallet, keys=True, data=True))
    else:
        out_edges, in_edges = [], []

    out_degree = len(out_edges)
    in_degree = len(in_edges)
    unique_out_counterparties = len({v for (_u, v, _k, _d) in out_edges})
    unique_in_counterparties = len({u for (u, _v, _k, _d) in in_edges})
    total_edge_count = out_degree + in_degree

    # --- fund-flow tracing ---
    paths = trace_result.paths
    hop_counts = [p.hop_count for p in paths]
    durations = [
        p.path_duration_seconds for p in paths if p.path_duration_seconds is not None
    ]
    paths_with_unknown_duration = sum(
        1 for p in paths if p.path_duration_seconds is None
    )

    path_count = len(paths)
    max_hop_count = max(hop_counts) if hop_counts else 0
    avg_hop_count = (sum(hop_counts) / len(hop_counts)) if hop_counts else 0.0
    max_path_duration_seconds = float(max(durations)) if durations else None
    avg_path_duration_seconds = (
        float(sum(durations) / len(durations)) if durations else None
    )

    # --- behavioral patterns ---
    pattern_types_present = {p.pattern_type for p in behavior_patterns}

    # --- attribution evidence (read-only) ---
    has_direct_evidence = any(
        c.evidence_tier == EvidenceTier.DIRECT for c in attribution_result.candidates
    )
    has_indirect_evidence = any(
        c.evidence_tier == EvidenceTier.INDIRECT for c in attribution_result.candidates
    )

    return WalletFeatures(
        wallet=wallet,
        in_degree=in_degree,
        out_degree=out_degree,
        unique_in_counterparties=unique_in_counterparties,
        unique_out_counterparties=unique_out_counterparties,
        total_edge_count=total_edge_count,
        path_count=path_count,
        max_hop_count=max_hop_count,
        avg_hop_count=avg_hop_count,
        max_path_duration_seconds=max_path_duration_seconds,
        avg_path_duration_seconds=avg_path_duration_seconds,
        paths_with_unknown_duration=paths_with_unknown_duration,
        has_split_pattern=PatternType.SPLIT_PATTERN in pattern_types_present,
        has_consolidation_pattern=PatternType.CONSOLIDATION_PATTERN
        in pattern_types_present,
        has_rapid_hopping=PatternType.RAPID_HOPPING in pattern_types_present,
        has_high_frequency_counterparty=PatternType.HIGH_FREQUENCY_COUNTERPARTY
        in pattern_types_present,
        has_repeated_forwarding=PatternType.REPEATED_FORWARDING
        in pattern_types_present,
        has_temporal_burst=PatternType.TEMPORAL_BURST in pattern_types_present,
        behavior_pattern_count=len(behavior_patterns),
        attribution_status=attribution_result.status.value,
        has_direct_evidence=has_direct_evidence,
        has_indirect_evidence=has_indirect_evidence,
        candidate_count=len(attribution_result.candidates),
    )
