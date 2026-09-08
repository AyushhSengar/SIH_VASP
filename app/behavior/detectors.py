"""
MACRO MILESTONE 3 / PHASE B — behavioral pattern detection (Part 14 of the
spec, "original Milestone 4").

Deterministic, explainable graph/transaction analysis only. No ML, no
neural networks, no clustering, no scikit-learn/TensorFlow/PyTorch/LLMs —
every detector here is a plain, auditable rule over graph structure and
edge timestamps, with its threshold read from centralized configuration
(app/core/config.py).

--------------------------------------------------------------------------
NO FALSE CERTAINTY (do not remove): a BehaviorPattern is an indicator, not
proof of anything. This module must NEVER assign labels such as
"criminal", "fraudulent", "money launderer", "sanctioned", or attribute a
wallet to a specific VASP ("belongs to Binance"). Those require evidence
this milestone does not have (attribution/clustering/sanctions data are
later milestones — explicitly out of scope here). Every detector below
returns pattern_type + evidence + metrics + related_addresses, and
deliberately nothing stronger.
--------------------------------------------------------------------------

Patterns implemented (15). The first six are the original conservative set
(see README for why a separate "peeling" detector was NOT added); the rest
are additional threshold-driven indicators, every one of which reports the
Settings field its threshold came from alongside the value actually
observed, so a reader never has to open this file to know why it fired:
  - SPLIT_PATTERN            (fan-out / splitting)
  - CONSOLIDATION_PATTERN    (fan-in / consolidation)
  - HIGH_FREQUENCY_COUNTERPARTY
  - TEMPORAL_BURST
  - REPEATED_FORWARDING      (conservative stand-in for "possible peeling" —
                              see detect_repeated_forwarding docstring)
  - RAPID_HOPPING            (operates on a Phase-A FundFlowPath, tying
                              Phase B directly to the tracing layer)
  - FAST_INBOUND_OUTBOUND    (rapid pass-through / short holding time)
  - HIGH_COUNTERPARTY_CONCENTRATION
  - ASSET_DIVERSITY
  - REPEATED_AMOUNT_PATTERN
  - DORMANT_THEN_ACTIVE
  - LARGE_VALUE_TRANSFER
  - UNUSUAL_TIMING           (CONTEXTUAL only — an address has no timezone)
  - IN_OUT_IMBALANCE
  - HIGH_ACTIVITY_DENSITY

All detectors are scoped to a single wallet (the wallet under
investigation), matching how the CLI and the eventual attribution pipeline
use them — nothing here does an all-nodes graph scan, which would be
unbounded on a dense graph.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional

import networkx as nx

from app.core.config import Settings, get_settings
from app.behavior.models import BehaviorPattern, IndicatorClass, PatternType
from app.tracing.models import FundFlowPath

# Safety cap on how many incoming/outgoing edges detect_repeated_forwarding
# will pairwise-compare per wallet. Comparing every in-edge against every
# out-edge is O(in * out); on a very high-degree wallet (e.g. thousands of
# edges) that's too expensive to run unbounded, so only the most recent
# edges (by timestamp) are considered. This is a performance bound, not a
# configurable investigative threshold, so it's a module constant rather
# than centralized config.
_MAX_FORWARDING_EDGES_PER_SIDE = 300


def _out_edges(graph: nx.MultiDiGraph, wallet: str) -> list[tuple]:
    if wallet not in graph:
        return []
    return list(graph.out_edges(wallet, keys=True, data=True))


def _in_edges(graph: nx.MultiDiGraph, wallet: str) -> list[tuple]:
    if wallet not in graph:
        return []
    return list(graph.in_edges(wallet, keys=True, data=True))


# How many concrete transaction references a single indicator carries. An
# indicator that fired on 4,000 edges does not become more credible by
# listing all 4,000 hashes, and a report has to stay readable — so a bounded
# sample is attached and the full count always stays in `metrics`. This is a
# presentation bound, not an investigative threshold, hence a module constant.
_MAX_TX_REFERENCES = 10


def _tx_hashes(edge_data: list[dict[str, Any]]) -> list[str]:
    """A deterministic, bounded sample of the transaction hashes behind an
    indicator. Sorted (not insertion-ordered) so two runs over the same graph
    always cite the same transactions."""
    hashes = sorted({d.get("tx_hash") for d in edge_data if d.get("tx_hash")})
    return hashes[:_MAX_TX_REFERENCES]


def _touching_edges(
    graph: nx.MultiDiGraph, wallet: str
) -> list[tuple[str, str, str, dict, str]]:
    """Every edge incident on the wallet as (u, v, key, data, direction).

    A self-loop appears in both in_edges and out_edges, so results are
    de-duplicated on (u, v, key) to stop one transfer being counted twice.
    Its direction is reported as "SELF".
    """
    seen: set[tuple[str, str, str]] = set()
    edges: list[tuple[str, str, str, dict, str]] = []
    for u, v, k, d in _out_edges(graph, wallet):
        identity = (u, v, k)
        if identity in seen:
            continue
        seen.add(identity)
        edges.append((u, v, k, d, "SELF" if u == v else "OUT"))
    for u, v, k, d in _in_edges(graph, wallet):
        identity = (u, v, k)
        if identity in seen:
            continue
        seen.add(identity)
        edges.append((u, v, k, d, "SELF" if u == v else "IN"))
    # Deterministic order regardless of NetworkX's internal adjacency order.
    edges.sort(key=lambda e: (e[3].get("timestamp") or 0, e[2], e[0], e[1]))
    return edges


def _amount_of(data: dict[str, Any]) -> float:
    try:
        return float(data.get("amount") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def detect_fan_out(
    graph: nx.MultiDiGraph, wallet: str, settings: Optional[Settings] = None
) -> Optional[BehaviorPattern]:
    """SPLIT_PATTERN: one wallet distributing funds to many destinations.

    Flags only on unique outgoing counterparty count crossing the
    configured threshold — NOT merely "many outgoing transactions" (a
    wallet sending 50 times to the same counterparty is not a split).
    """
    settings = settings or get_settings()
    out_edges = _out_edges(graph, wallet)
    if not out_edges:
        return None

    counterparties = sorted({v for _, v, _, _ in out_edges})
    if len(counterparties) < settings.behavior_min_fanout_counterparties:
        return None

    timestamps = [d.get("timestamp") for _, _, _, d in out_edges if d.get("timestamp") is not None]
    assets = sorted({d.get("asset", "UNKNOWN") for _, _, _, d in out_edges})

    evidence = [
        f"{len(counterparties)} unique outgoing counterparties across "
        f"{len(out_edges)} outgoing transfer(s)",
        f"Assets involved: {', '.join(assets)}",
    ]

    return BehaviorPattern(
        pattern_type=PatternType.SPLIT_PATTERN,
        wallet=wallet,
        evidence=evidence,
        metrics={
            "unique_outgoing_counterparties": len(counterparties),
            "total_outgoing_transfers": len(out_edges),
        },
        related_addresses=counterparties,
        first_seen=min(timestamps) if timestamps else None,
        last_seen=max(timestamps) if timestamps else None,
        observed_metric="unique_outgoing_counterparties",
        observed_value=len(counterparties),
        threshold=settings.behavior_min_fanout_counterparties,
        threshold_setting="behavior_min_fanout_counterparties",
        relevant_tx_hashes=_tx_hashes([d for _, _, _, d in out_edges]),
    )


def detect_fan_in(
    graph: nx.MultiDiGraph, wallet: str, settings: Optional[Settings] = None
) -> Optional[BehaviorPattern]:
    """CONSOLIDATION_PATTERN: many wallets sending toward one wallet."""
    settings = settings or get_settings()
    in_edges = _in_edges(graph, wallet)
    if not in_edges:
        return None

    counterparties = sorted({u for u, _, _, _ in in_edges})
    if len(counterparties) < settings.behavior_min_fanin_counterparties:
        return None

    timestamps = [d.get("timestamp") for _, _, _, d in in_edges if d.get("timestamp") is not None]
    assets = sorted({d.get("asset", "UNKNOWN") for _, _, _, d in in_edges})

    evidence = [
        f"{len(counterparties)} unique incoming counterparties across "
        f"{len(in_edges)} incoming transfer(s)",
        f"Assets involved: {', '.join(assets)}",
    ]

    return BehaviorPattern(
        pattern_type=PatternType.CONSOLIDATION_PATTERN,
        wallet=wallet,
        evidence=evidence,
        metrics={
            "unique_incoming_counterparties": len(counterparties),
            "total_incoming_transfers": len(in_edges),
        },
        related_addresses=counterparties,
        first_seen=min(timestamps) if timestamps else None,
        last_seen=max(timestamps) if timestamps else None,
        observed_metric="unique_incoming_counterparties",
        observed_value=len(counterparties),
        threshold=settings.behavior_min_fanin_counterparties,
        threshold_setting="behavior_min_fanin_counterparties",
        relevant_tx_hashes=_tx_hashes([d for _, _, _, d in in_edges]),
    )


def detect_high_frequency_counterparties(
    graph: nx.MultiDiGraph, wallet: str, settings: Optional[Settings] = None
) -> list[BehaviorPattern]:
    """HIGH_FREQUENCY_COUNTERPARTY: one pattern per counterparty the wallet
    has exchanged at least `behavior_high_frequency_min_transfers` transfers
    with (either direction, summed)."""
    settings = settings or get_settings()
    counts: dict[str, list[dict]] = {}

    for _, v, _, d in _out_edges(graph, wallet):
        counts.setdefault(v, []).append(d)
    for u, _, _, d in _in_edges(graph, wallet):
        counts.setdefault(u, []).append(d)

    patterns: list[BehaviorPattern] = []
    for counterparty in sorted(counts):
        edge_data = counts[counterparty]
        if len(edge_data) < settings.behavior_high_frequency_min_transfers:
            continue
        valid_ts = [d["timestamp"] for d in edge_data if d.get("timestamp") is not None]
        patterns.append(
            BehaviorPattern(
                pattern_type=PatternType.HIGH_FREQUENCY_COUNTERPARTY,
                wallet=wallet,
                evidence=[
                    f"{len(edge_data)} transfer(s) exchanged with {counterparty} "
                    "(incoming + outgoing combined)"
                ],
                metrics={"transfer_count": len(edge_data)},
                related_addresses=[counterparty],
                first_seen=min(valid_ts) if valid_ts else None,
                last_seen=max(valid_ts) if valid_ts else None,
                observed_metric="transfer_count",
                observed_value=len(edge_data),
                threshold=settings.behavior_high_frequency_min_transfers,
                threshold_setting="behavior_high_frequency_min_transfers",
                relevant_tx_hashes=_tx_hashes(edge_data),
            )
        )
    return patterns


def detect_temporal_burst(
    graph: nx.MultiDiGraph, wallet: str, settings: Optional[Settings] = None
) -> Optional[BehaviorPattern]:
    """TEMPORAL_BURST: unusually dense activity (in+out combined) inside a
    configurable sliding time window. Two-pointer sliding-window scan over
    sorted timestamps — O(n), safe on high-degree wallets."""
    settings = settings or get_settings()

    timestamps: list[int] = []
    for _, _, _, d in _out_edges(graph, wallet):
        if d.get("timestamp") is not None:
            timestamps.append(d["timestamp"])
    for _, _, _, d in _in_edges(graph, wallet):
        if d.get("timestamp") is not None:
            timestamps.append(d["timestamp"])

    if len(timestamps) < settings.behavior_burst_min_transfers:
        return None

    timestamps.sort()
    window = settings.behavior_burst_window_seconds

    left = 0
    best_count = 0
    best_window: Optional[tuple[int, int]] = None
    for right in range(len(timestamps)):
        while timestamps[right] - timestamps[left] > window:
            left += 1
        count = right - left + 1
        if count > best_count:
            best_count = count
            best_window = (timestamps[left], timestamps[right])

    if best_count < settings.behavior_burst_min_transfers or best_window is None:
        return None

    return BehaviorPattern(
        pattern_type=PatternType.TEMPORAL_BURST,
        wallet=wallet,
        evidence=[
            f"{best_count} transfer(s) (incoming + outgoing) fell within a "
            f"{window}-second window"
        ],
        metrics={"burst_transfer_count": best_count, "window_seconds": window},
        related_addresses=[],
        first_seen=best_window[0],
        last_seen=best_window[1],
        observed_metric="burst_transfer_count",
        observed_value=best_count,
        threshold=settings.behavior_burst_min_transfers,
        threshold_setting="behavior_burst_min_transfers",
    )


def detect_repeated_forwarding(
    graph: nx.MultiDiGraph, wallet: str, settings: Optional[Settings] = None
) -> Optional[BehaviorPattern]:
    """REPEATED_FORWARDING: the wallet repeatedly receives funds and sends
    to a *different* counterparty shortly afterward.

    Deliberately conservative (per the "no false certainty" / "be
    conservative on peeling" requirement): this is NOT a peel-chain
    detector and does not claim one. It only counts discrete
    receive-then-forward *events* — an incoming edge followed by an
    outgoing edge to a different address within `behavior_forwarding_
    window_seconds` — and requires at least `behavior_min_forwarding_
    events` such events before flagging anything. A wallet with one
    incoming and one outgoing transfer does not qualify.
    """
    settings = settings or get_settings()

    in_edges = [
        (u, d["timestamp"])
        for u, _, _, d in _in_edges(graph, wallet)
        if d.get("timestamp") is not None
    ]
    out_edges = [
        (v, d["timestamp"])
        for _, v, _, d in _out_edges(graph, wallet)
        if d.get("timestamp") is not None
    ]
    if not in_edges or not out_edges:
        return None

    # Bound pairwise comparison cost on very high-degree wallets — keep the
    # most recent edges on each side (most relevant for a live investigation).
    in_edges = sorted(in_edges, key=lambda e: e[1])[-_MAX_FORWARDING_EDGES_PER_SIDE:]
    out_edges = sorted(out_edges, key=lambda e: e[1])[-_MAX_FORWARDING_EDGES_PER_SIDE:]

    window = settings.behavior_forwarding_window_seconds
    events: list[tuple[str, int, str, int]] = []
    related: set[str] = set()

    for in_counterparty, in_ts in in_edges:
        for out_counterparty, out_ts in out_edges:
            if out_counterparty == in_counterparty:
                continue  # must forward to a DIFFERENT counterparty
            if out_ts >= in_ts and (out_ts - in_ts) <= window:
                events.append((in_counterparty, in_ts, out_counterparty, out_ts))
                related.add(in_counterparty)
                related.add(out_counterparty)

    if len(events) < settings.behavior_min_forwarding_events:
        return None

    evidence = [
        f"Received from {ic} at t={its}, forwarded to {oc} at t={ots} "
        f"({ots - its}s later)"
        for ic, its, oc, ots in events[:5]
    ]
    if len(events) > 5:
        evidence.append(f"...and {len(events) - 5} more receive-then-forward event(s)")

    all_ts = [e[1] for e in events] + [e[3] for e in events]

    return BehaviorPattern(
        pattern_type=PatternType.REPEATED_FORWARDING,
        wallet=wallet,
        evidence=evidence,
        metrics={
            "forwarding_event_count": len(events),
            "window_seconds": window,
        },
        related_addresses=sorted(related),
        first_seen=min(all_ts),
        last_seen=max(all_ts),
        observed_metric="forwarding_event_count",
        observed_value=len(events),
        threshold=settings.behavior_min_forwarding_events,
        threshold_setting="behavior_min_forwarding_events",
    )


def detect_rapid_hopping(
    path: FundFlowPath, settings: Optional[Settings] = None
) -> Optional[BehaviorPattern]:
    """RAPID_HOPPING: a Phase-A fund-flow path whose every consecutive hop
    gap is <= behavior_rapid_hop_max_seconds.

    Requires every hop in the path to have a timestamp — if any is missing,
    the gap can't be computed without fabricating a value, so this
    conservatively returns None rather than guessing.
    """
    settings = settings or get_settings()

    if path.hop_count < 2:
        return None  # "rapid hopping" requires at least 2 hops to have a gap

    timestamps = [h.timestamp for h in path.hops]
    if any(t is None for t in timestamps):
        return None

    gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    if any(g < 0 for g in gaps):
        return None  # should be unreachable given tracer's chronological guard

    if not all(g <= settings.behavior_rapid_hop_max_seconds for g in gaps):
        return None

    duration = path.path_duration_seconds
    evidence = [
        f"{path.hop_count}-hop path completed in {duration} second(s)",
        f"Hop gaps (seconds): {gaps} — all <= "
        f"{settings.behavior_rapid_hop_max_seconds}s threshold",
    ]

    return BehaviorPattern(
        pattern_type=PatternType.RAPID_HOPPING,
        wallet=path.source,
        evidence=evidence,
        metrics={
            "hop_count": path.hop_count,
            "total_duration_seconds": duration if duration is not None else -1,
            "average_hop_interval_seconds": round(sum(gaps) / len(gaps), 2),
        },
        related_addresses=path.addresses[1:],
        first_seen=path.start_timestamp,
        last_seen=path.end_timestamp,
        observed_metric="max_hop_interval_seconds",
        observed_value=max(gaps),
        threshold=settings.behavior_rapid_hop_max_seconds,
        threshold_setting="behavior_rapid_hop_max_seconds",
        relevant_tx_hashes=[h.tx_hash for h in path.hops if h.tx_hash],
    )


def detect_fast_passthrough(
    graph: nx.MultiDiGraph, wallet: str, settings: Optional[Settings] = None
) -> Optional[BehaviorPattern]:
    """FAST_INBOUND_OUTBOUND: value arrived and left again almost immediately.

    Distinct from REPEATED_FORWARDING, which is about *routing* (received
    from X, sent onward to a different Y, repeatedly). This one is purely
    about *holding time* — the same counterparty on both sides still counts,
    because a wallet that never holds value for more than a few minutes is
    behaving like a conduit regardless of who it forwards to.

    Implemented with a binary search for each inbound transfer's next
    outbound transfer, so the cost is O(n log n) rather than the O(in * out)
    of a pairwise scan — this detector runs unbounded on high-degree wallets
    where detect_repeated_forwarding has to cap its comparison set.
    """
    settings = settings or get_settings()

    # Sorted on the tuple's comparable prefix only — two transfers can share
    # a timestamp AND a counterparty, and Python would then try to order the
    # raw edge dicts, which is a TypeError.
    inbound = sorted(
        (
            (d["timestamp"], u, d)
            for u, _, _, d in _in_edges(graph, wallet)
            if d.get("timestamp") is not None
        ),
        key=lambda item: (item[0], item[1]),
    )
    outbound = sorted(
        (
            (d["timestamp"], v, d)
            for _, v, _, d in _out_edges(graph, wallet)
            if d.get("timestamp") is not None
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not inbound or not outbound:
        return None

    out_times = [t for t, _, _ in outbound]
    threshold = settings.behavior_fast_passthrough_max_seconds

    events: list[tuple[int, str, str, int]] = []  # (gap, from, to, in_ts)
    edge_data: list[dict] = []
    related: set[str] = set()
    for in_ts, sender, in_data in inbound:
        index = bisect_left(out_times, in_ts)
        if index >= len(out_times):
            continue
        out_ts, recipient, out_data = outbound[index]
        gap = out_ts - in_ts
        if gap > threshold:
            continue
        events.append((gap, sender, recipient, in_ts))
        edge_data.extend((in_data, out_data))
        related.update((sender, recipient))

    if not events:
        return None

    fastest = min(events, key=lambda e: (e[0], e[3], e[1], e[2]))
    evidence = [
        f"{len(events)} inbound transfer(s) were followed by an outbound "
        f"transfer within {threshold}s",
        f"Fastest turnaround: {fastest[0]}s (in from {fastest[1]} at "
        f"t={fastest[3]}, out to {fastest[2]})",
    ]

    return BehaviorPattern(
        pattern_type=PatternType.FAST_INBOUND_OUTBOUND,
        wallet=wallet,
        evidence=evidence,
        metrics={
            "fast_passthrough_event_count": len(events),
            "fastest_turnaround_seconds": fastest[0],
            "median_turnaround_seconds": sorted(e[0] for e in events)[
                len(events) // 2
            ],
        },
        related_addresses=sorted(related),
        first_seen=min(e[3] for e in events),
        last_seen=max(e[3] for e in events),
        observed_metric="fastest_turnaround_seconds",
        observed_value=fastest[0],
        threshold=threshold,
        threshold_setting="behavior_fast_passthrough_max_seconds",
        relevant_tx_hashes=_tx_hashes(edge_data),
    )


def detect_counterparty_concentration(
    graph: nx.MultiDiGraph, wallet: str, settings: Optional[Settings] = None
) -> Optional[BehaviorPattern]:
    """HIGH_COUNTERPARTY_CONCENTRATION: one counterparty dominates activity.

    Concentration is measured on transfer COUNT, not value: value shares are
    distorted by a single large transfer and by token amounts that are not
    comparable across assets (1 unit of one token is not 1 unit of another),
    whereas counting transfers needs no cross-asset conversion and so states
    something the data actually supports.
    """
    settings = settings or get_settings()
    edges = _touching_edges(graph, wallet)
    if len(edges) < settings.behavior_concentration_min_transfers:
        return None

    counts: Counter = Counter()
    per_counterparty: dict[str, list[dict]] = {}
    for u, v, _, d, direction in edges:
        counterparty = v if direction == "OUT" else u
        counts[counterparty] += 1
        per_counterparty.setdefault(counterparty, []).append(d)

    if not counts:
        return None

    # Deterministic tie-break: highest count, then lowest address.
    top_address, top_count = min(
        counts.items(), key=lambda item: (-item[1], item[0])
    )
    share = top_count / len(edges)
    if share < settings.behavior_counterparty_concentration_min_share:
        return None

    timestamps = [
        d.get("timestamp")
        for d in per_counterparty[top_address]
        if d.get("timestamp") is not None
    ]

    return BehaviorPattern(
        pattern_type=PatternType.HIGH_COUNTERPARTY_CONCENTRATION,
        wallet=wallet,
        evidence=[
            f"{top_address} accounts for {top_count} of {len(edges)} transfer(s) "
            f"touching this wallet ({share:.1%})",
            f"{len(counts)} distinct counterparty address(es) in total",
        ],
        metrics={
            "top_counterparty_share": round(share, 4),
            "top_counterparty_transfer_count": top_count,
            "total_transfers": len(edges),
            "distinct_counterparties": len(counts),
        },
        related_addresses=[top_address],
        first_seen=min(timestamps) if timestamps else None,
        last_seen=max(timestamps) if timestamps else None,
        observed_metric="top_counterparty_share",
        observed_value=round(share, 4),
        threshold=settings.behavior_counterparty_concentration_min_share,
        threshold_setting="behavior_counterparty_concentration_min_share",
        relevant_tx_hashes=_tx_hashes(per_counterparty[top_address]),
        # A dominant counterparty is often an entirely ordinary relationship
        # (an exchange deposit address, a single trading venue), so on its own
        # it colours an investigation rather than driving one.
        classification=IndicatorClass.SUPPORTING_EVIDENCE,
    )


def detect_asset_diversity(
    graph: nx.MultiDiGraph, wallet: str, settings: Optional[Settings] = None
) -> Optional[BehaviorPattern]:
    """ASSET_DIVERSITY: the wallet handled many distinct assets.

    Assets are counted by token contract where one is known and by symbol
    otherwise, because symbols are not unique on-chain — several unrelated
    contracts can all call themselves "USDC", and counting by symbol alone
    would merge them and understate diversity.
    """
    settings = settings or get_settings()
    edges = _touching_edges(graph, wallet)
    if not edges:
        return None

    identities: dict[str, str] = {}  # identity -> display label
    per_asset: dict[str, list[dict]] = {}
    for _, _, _, d, _ in edges:
        contract = d.get("token_contract")
        symbol = d.get("asset") or "UNKNOWN"
        identity = str(contract) if contract else f"symbol:{symbol}"
        identities[identity] = str(symbol)
        per_asset.setdefault(identity, []).append(d)

    if len(identities) < settings.behavior_min_asset_diversity:
        return None

    labels = sorted(set(identities.values()))
    timestamps = [
        d.get("timestamp") for _, _, _, d, _ in edges if d.get("timestamp") is not None
    ]

    return BehaviorPattern(
        pattern_type=PatternType.ASSET_DIVERSITY,
        wallet=wallet,
        evidence=[
            f"{len(identities)} distinct asset(s) moved through this wallet "
            f"across {len(edges)} transfer(s)",
            "Assets: " + ", ".join(labels[:15])
            + (f" ...and {len(labels) - 15} more" if len(labels) > 15 else ""),
        ],
        metrics={
            "distinct_asset_count": len(identities),
            "distinct_symbol_count": len(labels),
            "total_transfers": len(edges),
        },
        related_addresses=[],
        first_seen=min(timestamps) if timestamps else None,
        last_seen=max(timestamps) if timestamps else None,
        observed_metric="distinct_asset_count",
        observed_value=len(identities),
        threshold=settings.behavior_min_asset_diversity,
        threshold_setting="behavior_min_asset_diversity",
        classification=IndicatorClass.CONTEXTUAL,
    )


def detect_repeated_amounts(
    graph: nx.MultiDiGraph, wallet: str, settings: Optional[Settings] = None
) -> Optional[BehaviorPattern]:
    """REPEATED_AMOUNT_PATTERN: the same (asset, amount) recurs.

    Reported as REQUIRES_FURTHER_VERIFICATION rather than as a structuring
    finding: identical repeated amounts are equally characteristic of
    automated payouts, subscription payments, bot activity, and airdrops.
    Distinguishing those from deliberate structuring needs information that
    is not on-chain.
    """
    settings = settings or get_settings()
    edges = _touching_edges(graph, wallet)
    if not edges:
        return None

    decimals = settings.behavior_repeated_amount_decimals
    groups: dict[tuple[str, float], list[dict]] = {}
    for _, _, _, d, _ in edges:
        amount = _amount_of(d)
        if amount <= 0:
            continue  # a zero-value transfer repeating says nothing about amount
        asset = str(d.get("asset") or "UNKNOWN")
        groups.setdefault((asset, round(amount, decimals)), []).append(d)

    qualifying = {
        key: value
        for key, value in groups.items()
        if len(value) >= settings.behavior_repeated_amount_min_occurrences
    }
    if not qualifying:
        return None

    ranked = sorted(qualifying.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    top_key, top_edges = ranked[0]
    evidence = [
        f"{len(top_edges)} transfer(s) of exactly {top_key[1]} {top_key[0]}"
    ] + [
        f"{len(value)} transfer(s) of exactly {key[1]} {key[0]}"
        for key, value in ranked[1:4]
    ]

    all_edges = [d for _, value in ranked for d in value]
    timestamps = [
        d["timestamp"] for d in all_edges if d.get("timestamp") is not None
    ]

    return BehaviorPattern(
        pattern_type=PatternType.REPEATED_AMOUNT_PATTERN,
        wallet=wallet,
        evidence=evidence,
        metrics={
            "repeated_amount_group_count": len(qualifying),
            "max_repeat_count": len(top_edges),
            "most_repeated_amount": top_key[1],
            "most_repeated_asset": top_key[0],
            "rounding_decimals": decimals,
        },
        related_addresses=[],
        first_seen=min(timestamps) if timestamps else None,
        last_seen=max(timestamps) if timestamps else None,
        observed_metric="max_repeat_count",
        observed_value=len(top_edges),
        threshold=settings.behavior_repeated_amount_min_occurrences,
        threshold_setting="behavior_repeated_amount_min_occurrences",
        relevant_tx_hashes=_tx_hashes(top_edges),
        classification=IndicatorClass.REQUIRES_FURTHER_VERIFICATION,
    )


def detect_dormant_then_active(
    graph: nx.MultiDiGraph, wallet: str, settings: Optional[Settings] = None
) -> Optional[BehaviorPattern]:
    """DORMANT_THEN_ACTIVE: a long silence followed by concentrated activity.

    Both halves are required. A long gap alone is unremarkable (most wallets
    have them); a gap followed immediately by a burst is what distinguishes a
    reactivated wallet from one that is simply used infrequently.
    """
    settings = settings or get_settings()
    edges = _touching_edges(graph, wallet)
    timed = sorted(
        (
            (d["timestamp"], d)
            for _, _, _, d, _ in edges
            if d.get("timestamp") is not None
        ),
        # Key on the timestamp only: transfers sharing a timestamp are not
        # orderable by their edge dicts, and their relative order does not
        # matter to a dormancy gap.
        key=lambda item: item[0],
    )
    if len(timed) < 2:
        return None

    times = [t for t, _ in timed]
    best: Optional[tuple[int, int, int, int]] = None  # (gap, resume_ts, count, idx)
    for index in range(1, len(times)):
        gap = times[index] - times[index - 1]
        if gap < settings.behavior_dormancy_min_seconds:
            continue
        resume_ts = times[index]
        window_end = resume_ts + settings.behavior_reactivation_window_seconds
        count = 0
        for later in times[index:]:
            if later > window_end:
                break
            count += 1
        if count < settings.behavior_reactivation_min_transfers:
            continue
        if best is None or gap > best[0]:
            best = (gap, resume_ts, count, index)

    if best is None:
        return None

    gap, resume_ts, count, index = best
    burst_edges = [
        d
        for t, d in timed[index:]
        if t <= resume_ts + settings.behavior_reactivation_window_seconds
    ]
    related = sorted(
        {
            (v if direction == "OUT" else u)
            for u, v, _, d, direction in edges
            if d.get("timestamp") is not None
            and resume_ts
            <= d["timestamp"]
            <= resume_ts + settings.behavior_reactivation_window_seconds
        }
    )

    return BehaviorPattern(
        pattern_type=PatternType.DORMANT_THEN_ACTIVE,
        wallet=wallet,
        evidence=[
            f"No observed activity for {gap} second(s) "
            f"({gap / 86400:.1f} day(s)) ending at t={resume_ts}",
            f"{count} transfer(s) then occurred within "
            f"{settings.behavior_reactivation_window_seconds}s of reactivation",
        ],
        metrics={
            "dormancy_seconds": gap,
            "dormancy_days": round(gap / 86400, 2),
            "reactivation_transfer_count": count,
            "reactivation_timestamp": resume_ts,
        },
        related_addresses=related,
        first_seen=times[index - 1],
        last_seen=max(
            t for t, _ in timed
            if t <= resume_ts + settings.behavior_reactivation_window_seconds
        ),
        observed_metric="dormancy_seconds",
        observed_value=gap,
        threshold=settings.behavior_dormancy_min_seconds,
        threshold_setting="behavior_dormancy_min_seconds",
        relevant_tx_hashes=_tx_hashes(burst_edges),
    )


def detect_large_value_transfers(
    graph: nx.MultiDiGraph, wallet: str, settings: Optional[Settings] = None
) -> Optional[BehaviorPattern]:
    """LARGE_VALUE_TRANSFER: notable native-asset transfers.

    Restricted to the NATIVE asset on purpose. Token amounts are not
    comparable to each other or to ETH without price data this project does
    not have, so applying one numeric threshold across all assets would
    produce a meaningless comparison (1,000 units of a valueless token would
    outrank 10 ETH).
    """
    settings = settings or get_settings()
    edges = _touching_edges(graph, wallet)
    threshold = settings.behavior_large_value_native_amount

    qualifying: list[tuple[float, str, str, dict]] = []
    for u, v, _, d, direction in edges:
        if str(d.get("asset_type") or "").upper() != "NATIVE":
            continue
        amount = _amount_of(d)
        if amount < threshold:
            continue
        qualifying.append((amount, direction, v if direction == "OUT" else u, d))

    if not qualifying:
        return None

    qualifying.sort(key=lambda q: (-q[0], q[2]))
    largest = qualifying[0]
    timestamps = [
        q[3]["timestamp"] for q in qualifying if q[3].get("timestamp") is not None
    ]

    return BehaviorPattern(
        pattern_type=PatternType.LARGE_VALUE_TRANSFER,
        wallet=wallet,
        evidence=[
            f"{len(qualifying)} native-asset transfer(s) at or above "
            f"{threshold}",
            f"Largest: {largest[0]} {largest[3].get('asset', 'NATIVE')} "
            f"{largest[1]} {largest[2]} (tx {largest[3].get('tx_hash')})",
        ],
        metrics={
            "large_transfer_count": len(qualifying),
            "largest_native_amount": largest[0],
            "total_native_in_large_transfers": round(
                sum(q[0] for q in qualifying), 8
            ),
        },
        related_addresses=sorted({q[2] for q in qualifying}),
        first_seen=min(timestamps) if timestamps else None,
        last_seen=max(timestamps) if timestamps else None,
        observed_metric="largest_native_amount",
        observed_value=largest[0],
        threshold=threshold,
        threshold_setting="behavior_large_value_native_amount",
        relevant_tx_hashes=_tx_hashes([q[3] for q in qualifying]),
    )


def detect_unusual_timing(
    graph: nx.MultiDiGraph, wallet: str, settings: Optional[Settings] = None
) -> Optional[BehaviorPattern]:
    """UNUSUAL_TIMING: activity concentrated in a low-activity UTC hour band.

    CONTEXTUAL, and it can never be anything more. An address has no
    timezone: 02:00 UTC is the middle of the night in London and mid-morning
    in Tokyo, so "unusual hours" is a statement about the observer's
    assumption, not about the wallet. Included because it is a stated
    requirement and because the measured share is a genuine fact — but it is
    classified so that no conclusion can rest on it.

    The configured band is inclusive of both endpoints and wraps around
    midnight when start > end (e.g. 22 -> 3 covers 22,23,0,1,2,3).
    """
    settings = settings or get_settings()
    edges = _touching_edges(graph, wallet)
    timed = [d for _, _, _, d, _ in edges if d.get("timestamp") is not None]
    if len(timed) < settings.behavior_unusual_timing_min_transfers:
        return None

    start = settings.behavior_unusual_hour_start_utc
    end = settings.behavior_unusual_hour_end_utc

    def in_band(hour: int) -> bool:
        if start <= end:
            return start <= hour <= end
        return hour >= start or hour <= end  # wraps midnight

    in_band_edges = [
        d
        for d in timed
        if in_band(datetime.fromtimestamp(d["timestamp"], tz=timezone.utc).hour)
    ]
    share = len(in_band_edges) / len(timed)
    if share < settings.behavior_unusual_timing_min_share:
        return None

    return BehaviorPattern(
        pattern_type=PatternType.UNUSUAL_TIMING,
        wallet=wallet,
        evidence=[
            f"{len(in_band_edges)} of {len(timed)} timestamped transfer(s) "
            f"({share:.1%}) fall in the {start:02d}:00-{end:02d}:59 UTC band",
            "UTC only — the operator's actual timezone is unknown, so this "
            "is context, not an indicator of concealment",
        ],
        metrics={
            "unusual_hour_share": round(share, 4),
            "unusual_hour_transfer_count": len(in_band_edges),
            "timestamped_transfer_count": len(timed),
            "band_start_utc": start,
            "band_end_utc": end,
        },
        related_addresses=[],
        first_seen=min(d["timestamp"] for d in in_band_edges),
        last_seen=max(d["timestamp"] for d in in_band_edges),
        observed_metric="unusual_hour_share",
        observed_value=round(share, 4),
        threshold=settings.behavior_unusual_timing_min_share,
        threshold_setting="behavior_unusual_timing_min_share",
        relevant_tx_hashes=_tx_hashes(in_band_edges),
        classification=IndicatorClass.CONTEXTUAL,
    )


def detect_in_out_imbalance(
    graph: nx.MultiDiGraph, wallet: str, settings: Optional[Settings] = None
) -> Optional[BehaviorPattern]:
    """IN_OUT_IMBALANCE: transfer counts are strongly one-directional.

    Measured on counts rather than value for the same cross-asset
    comparability reason as HIGH_COUNTERPARTY_CONCENTRATION. Both extremes
    are reported (receive-mostly and send-mostly), because which one is
    interesting depends entirely on what the investigation is about.
    """
    settings = settings or get_settings()
    edges = _touching_edges(graph, wallet)
    inbound = [e for e in edges if e[4] == "IN"]
    outbound = [e for e in edges if e[4] == "OUT"]
    total = len(inbound) + len(outbound)
    if total < settings.behavior_in_out_imbalance_min_transfers:
        return None

    ratio = abs(len(inbound) - len(outbound)) / total
    if ratio < settings.behavior_in_out_imbalance_min_ratio:
        return None

    direction = "receive-mostly" if len(inbound) > len(outbound) else "send-mostly"
    dominant = inbound if len(inbound) > len(outbound) else outbound
    timestamps = [
        e[3]["timestamp"] for e in dominant if e[3].get("timestamp") is not None
    ]

    return BehaviorPattern(
        pattern_type=PatternType.IN_OUT_IMBALANCE,
        wallet=wallet,
        evidence=[
            f"{len(inbound)} incoming vs {len(outbound)} outgoing transfer(s) "
            f"— imbalance ratio {ratio:.2f} ({direction})",
        ],
        metrics={
            "incoming_transfer_count": len(inbound),
            "outgoing_transfer_count": len(outbound),
            "imbalance_ratio": round(ratio, 4),
            "direction": direction,
        },
        related_addresses=[],
        first_seen=min(timestamps) if timestamps else None,
        last_seen=max(timestamps) if timestamps else None,
        observed_metric="imbalance_ratio",
        observed_value=round(ratio, 4),
        threshold=settings.behavior_in_out_imbalance_min_ratio,
        threshold_setting="behavior_in_out_imbalance_min_ratio",
        classification=IndicatorClass.CONTEXTUAL,
    )


def detect_activity_density(
    graph: nx.MultiDiGraph, wallet: str, settings: Optional[Settings] = None
) -> Optional[BehaviorPattern]:
    """HIGH_ACTIVITY_DENSITY: many transfers per *active* day.

    Density is transfers divided by the number of distinct UTC days on which
    the wallet actually transacted — not by its whole lifespan. Dividing by
    lifespan would let a single busy week look quiet just because the wallet
    is old, which is the opposite of what this indicator is for.
    """
    settings = settings or get_settings()
    edges = _touching_edges(graph, wallet)
    timed = [d for _, _, _, d, _ in edges if d.get("timestamp") is not None]
    if len(timed) < settings.behavior_activity_density_min_transfers:
        return None

    active_days = {d["timestamp"] // 86400 for d in timed}
    density = len(timed) / len(active_days)
    if density < settings.behavior_activity_density_min_per_day:
        return None

    return BehaviorPattern(
        pattern_type=PatternType.HIGH_ACTIVITY_DENSITY,
        wallet=wallet,
        evidence=[
            f"{len(timed)} transfer(s) across {len(active_days)} active UTC "
            f"day(s) = {density:.1f} transfer(s) per active day",
        ],
        metrics={
            "transfers_per_active_day": round(density, 2),
            "active_day_count": len(active_days),
            "timestamped_transfer_count": len(timed),
        },
        related_addresses=[],
        first_seen=min(d["timestamp"] for d in timed),
        last_seen=max(d["timestamp"] for d in timed),
        observed_metric="transfers_per_active_day",
        observed_value=round(density, 2),
        threshold=settings.behavior_activity_density_min_per_day,
        threshold_setting="behavior_activity_density_min_per_day",
        classification=IndicatorClass.CONTEXTUAL,
    )


def analyze_wallet_behavior(
    graph: nx.MultiDiGraph,
    wallet: str,
    settings: Optional[Settings] = None,
    paths: Optional[list[FundFlowPath]] = None,
) -> list[BehaviorPattern]:
    """Runs every Phase-B detector for a single wallet.

    `paths` (typically the output of app.tracing.tracer.trace_fund_flow for
    this wallet) is optional — without it, RAPID_HOPPING is skipped, since
    it depends on the Phase-A tracing layer rather than raw graph structure.

    Order is fixed (not dependent on dict/set iteration) so two runs over
    the same graph produce byte-identical output.
    """
    settings = settings or get_settings()
    patterns: list[BehaviorPattern] = []

    single_result_detectors = (
        detect_fan_out,
        detect_fan_in,
        detect_temporal_burst,
        detect_repeated_forwarding,
        detect_fast_passthrough,
        detect_counterparty_concentration,
        detect_asset_diversity,
        detect_repeated_amounts,
        detect_dormant_then_active,
        detect_large_value_transfers,
        detect_unusual_timing,
        detect_in_out_imbalance,
        detect_activity_density,
    )

    fan_out = detect_fan_out(graph, wallet, settings)
    if fan_out:
        patterns.append(fan_out)

    fan_in = detect_fan_in(graph, wallet, settings)
    if fan_in:
        patterns.append(fan_in)

    patterns.extend(detect_high_frequency_counterparties(graph, wallet, settings))

    # Everything after the two fan detectors, which are emitted first above to
    # preserve the pattern order this function has always produced.
    for detector in single_result_detectors[2:]:
        found = detector(graph, wallet, settings)
        if found:
            patterns.append(found)

    if paths:
        for path in paths:
            rapid = detect_rapid_hopping(path, settings)
            if rapid:
                patterns.append(rapid)

    return patterns
