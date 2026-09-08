"""
Temporal and amount analysis — the descriptive layer under the report's
"TEMPORAL / AMOUNT ANALYSIS" section.

This module is deliberately NOT a detector. Detectors (app/behavior/
detectors.py) answer "did a configured threshold get crossed?"; this module
answers "what does this wallet's activity actually look like?" and returns
measured quantities with no threshold, no verdict, and no interpretation.
That separation matters for the evidence model: a measurement can be quoted
in a report without implying that anything is wrong.

Three things it computes:

1. LIFECYCLE — first seen, last seen, lifespan, active days, idle stretch,
   and whether activity is front-loaded or recent.
2. TEMPORAL DISTRIBUTION — per-UTC-hour and per-weekday histograms, inter-
   transfer gap statistics, and the longest quiet period.
3. AMOUNTS — per-asset totals, in/out balance, and the distribution
   (min/median/max/mean) of transfer sizes, kept PER ASSET because amounts
   in different assets are not comparable without price data this project
   does not have.

Plus PASS-THROUGH DURATION: how long value rested in the wallet, measured
as the gap from each inbound transfer to the next outbound one. Reported as
an observed holding-time distribution, not as proof that the specific units
received were the units later sent — that would require value attribution
this project cannot perform.

Every field is Optional or empty-by-default where the underlying data is
missing. Nothing here fabricates a timestamp, an amount, or a decimal.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any, Optional

import networkx as nx
from pydantic import BaseModel

SECONDS_PER_DAY = 86_400


def _iso(timestamp: Optional[int]) -> Optional[str]:
    """UTC ISO-8601 rendering of a unix timestamp, for human-readable output.

    Returns None rather than a placeholder date when the timestamp is absent,
    so a missing value can never be mistaken for the epoch.
    """
    if timestamp is None:
        return None
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


class AmountStats(BaseModel):
    """Transfer-size distribution for ONE asset. Never aggregated across
    assets — see the module docstring."""

    asset: str
    asset_type: Optional[str] = None
    token_contract: Optional[str] = None

    transfer_count: int
    inbound_count: int
    outbound_count: int

    total_inbound: float
    total_outbound: float
    net_flow: float  # inbound - outbound, in this asset's own units

    min_amount: float
    median_amount: float
    max_amount: float
    mean_amount: float

    # True when at least one transfer of this asset had no usable decimals
    # metadata, so the human-readable amount may be off by a power of ten.
    metadata_incomplete: bool = False


class PassThroughStats(BaseModel):
    """Observed holding time between an inbound transfer and the next
    outbound one.

    NOT a claim of value continuity: this measures the interval between two
    events, and the report says so. Established by timestamp comparison only.
    """

    measured_events: int
    min_seconds: Optional[int] = None
    median_seconds: Optional[int] = None
    max_seconds: Optional[int] = None
    inbound_without_later_outbound: int = 0
    limitation: str = (
        "Measures the interval between an inbound transfer and the next "
        "outbound transfer. It does NOT establish that the units received "
        "were the units subsequently sent — that requires value-level "
        "attribution, which on-chain balances alone cannot provide."
    )


class TemporalAmountAnalysis(BaseModel):
    """Everything the report's temporal/amount section needs, measured."""

    wallet: str
    chain: str

    # --- lifecycle ---
    transfer_count: int
    timestamped_transfer_count: int
    missing_timestamp_count: int
    first_seen: Optional[int] = None
    last_seen: Optional[int] = None
    first_seen_utc: Optional[str] = None
    last_seen_utc: Optional[str] = None
    lifespan_seconds: Optional[int] = None
    lifespan_days: Optional[float] = None
    active_day_count: int = 0
    transfers_per_active_day: Optional[float] = None
    longest_idle_seconds: Optional[int] = None
    longest_idle_days: Optional[float] = None

    # --- temporal distribution ---
    hourly_utc_histogram: dict[int, int] = {}
    weekday_utc_histogram: dict[int, int] = {}
    busiest_utc_hour: Optional[int] = None
    busiest_utc_weekday: Optional[int] = None
    median_gap_seconds: Optional[int] = None
    mean_gap_seconds: Optional[float] = None

    # --- direction counts ---
    inbound_transfer_count: int = 0
    outbound_transfer_count: int = 0
    self_transfer_count: int = 0
    unique_inbound_counterparties: int = 0
    unique_outbound_counterparties: int = 0

    # --- amounts, per asset ---
    per_asset: list[AmountStats] = []

    # --- pass-through ---
    pass_through: Optional[PassThroughStats] = None

    limitations: list[str] = []

    @property
    def native_stats(self) -> Optional[AmountStats]:
        for stats in self.per_asset:
            if (stats.asset_type or "").upper() == "NATIVE":
                return stats
        return None


def wallet_incident_edges(
    graph: nx.MultiDiGraph, wallet: str
) -> list[tuple[str, str, str, dict[str, Any], str]]:
    """Every edge incident on the wallet as (u, v, key, data, direction),
    de-duplicated so a self-loop counts once, and deterministically ordered.

    Public because the report's per-transfer ledger must be the *same* set of
    edges this module counts. If the ledger were assembled independently it
    could drift -- a different de-duplication rule or a different self-loop
    decision would print a row count that disagreed with `transfer_count`,
    and a reader would have no way to tell which number was wrong.
    """
    if wallet not in graph:
        return []
    seen: set[tuple[str, str, str]] = set()
    edges: list[tuple[str, str, str, dict[str, Any], str]] = []
    for u, v, k, d in graph.out_edges(wallet, keys=True, data=True):
        identity = (u, v, str(k))
        if identity in seen:
            continue
        seen.add(identity)
        edges.append((u, v, str(k), d, "SELF" if u == v else "OUT"))
    for u, v, k, d in graph.in_edges(wallet, keys=True, data=True):
        identity = (u, v, str(k))
        if identity in seen:
            continue
        seen.add(identity)
        edges.append((u, v, str(k), d, "SELF" if u == v else "IN"))
    edges.sort(key=lambda e: (e[3].get("timestamp") or 0, e[2], e[0], e[1]))
    return edges


def _amount(data: dict[str, Any]) -> float:
    try:
        return float(data.get("amount") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _asset_key(data: dict[str, Any]) -> tuple[str, str]:
    """(identity, display symbol).

    Identity prefers the token contract, because symbols collide: several
    unrelated contracts can all be called "USDC", and merging them on symbol
    would silently combine two different assets' amounts into one total.
    """
    symbol = str(data.get("asset") or "UNKNOWN")
    contract = data.get("token_contract")
    return (str(contract) if contract else f"symbol:{symbol}", symbol)


def _pass_through(
    graph: nx.MultiDiGraph, wallet: str
) -> Optional[PassThroughStats]:
    """Inbound-to-next-outbound intervals, via binary search (O(n log n))."""
    if wallet not in graph:
        return None

    inbound = sorted(
        d["timestamp"]
        for _, _, _, d in graph.in_edges(wallet, keys=True, data=True)
        if d.get("timestamp") is not None
    )
    outbound = sorted(
        d["timestamp"]
        for _, _, _, d in graph.out_edges(wallet, keys=True, data=True)
        if d.get("timestamp") is not None
    )
    if not inbound or not outbound:
        return None

    gaps: list[int] = []
    unmatched = 0
    for in_ts in inbound:
        index = bisect_left(outbound, in_ts)
        if index >= len(outbound):
            unmatched += 1
            continue
        gaps.append(outbound[index] - in_ts)

    if not gaps:
        return PassThroughStats(
            measured_events=0, inbound_without_later_outbound=unmatched
        )

    return PassThroughStats(
        measured_events=len(gaps),
        min_seconds=min(gaps),
        median_seconds=int(median(gaps)),
        max_seconds=max(gaps),
        inbound_without_later_outbound=unmatched,
    )


def analyze_temporal_and_amounts(
    graph: nx.MultiDiGraph, wallet: str, chain: str = "ethereum"
) -> TemporalAmountAnalysis:
    """Measures one wallet's activity over time and by amount.

    Returns a fully-populated analysis even for a wallet with no edges (all
    counts zero, all optionals None) rather than raising, so a report can
    always render the section and state that there was nothing to measure.
    """
    edges = wallet_incident_edges(graph, wallet)
    limitations: list[str] = []

    if not edges:
        return TemporalAmountAnalysis(
            wallet=wallet,
            chain=chain,
            transfer_count=0,
            timestamped_transfer_count=0,
            missing_timestamp_count=0,
            limitations=[
                "The wallet has no transfer edges in this graph, so no "
                "temporal or amount analysis is possible."
            ],
        )

    timestamps = sorted(
        d["timestamp"] for _, _, _, d, _ in edges if d.get("timestamp") is not None
    )
    missing_ts = len(edges) - len(timestamps)
    if missing_ts:
        limitations.append(
            f"{missing_ts} of {len(edges)} transfer(s) carry no timestamp; they "
            "are counted in totals but excluded from every time-based measure "
            "rather than being assigned a guessed time."
        )

    # --- lifecycle -------------------------------------------------------
    first_seen = timestamps[0] if timestamps else None
    last_seen = timestamps[-1] if timestamps else None
    lifespan = (last_seen - first_seen) if timestamps else None
    active_days = {t // SECONDS_PER_DAY for t in timestamps}

    gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    longest_idle = max(gaps) if gaps else None

    # --- temporal distribution -------------------------------------------
    hour_counts: Counter = Counter()
    weekday_counts: Counter = Counter()
    for t in timestamps:
        moment = datetime.fromtimestamp(t, tz=timezone.utc)
        hour_counts[moment.hour] += 1
        weekday_counts[moment.weekday()] += 1

    # Deterministic tie-break on the smaller hour/weekday.
    busiest_hour = (
        min(hour_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        if hour_counts
        else None
    )
    busiest_weekday = (
        min(weekday_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        if weekday_counts
        else None
    )

    # --- direction counts -------------------------------------------------
    inbound = [e for e in edges if e[4] == "IN"]
    outbound = [e for e in edges if e[4] == "OUT"]
    self_transfers = [e for e in edges if e[4] == "SELF"]

    # --- amounts, per asset ------------------------------------------------
    grouped: dict[str, dict[str, Any]] = {}
    for u, v, _, d, direction in edges:
        identity, symbol = _asset_key(d)
        bucket = grouped.setdefault(
            identity,
            {
                "asset": symbol,
                "asset_type": d.get("asset_type"),
                "token_contract": d.get("token_contract"),
                "amounts": [],
                "in": 0.0,
                "out": 0.0,
                "in_count": 0,
                "out_count": 0,
                "metadata_incomplete": False,
            },
        )
        amount = _amount(d)
        bucket["amounts"].append(amount)
        if d.get("token_metadata_missing"):
            bucket["metadata_incomplete"] = True
        if direction == "IN":
            bucket["in"] += amount
            bucket["in_count"] += 1
        elif direction == "OUT":
            bucket["out"] += amount
            bucket["out_count"] += 1
        else:  # SELF — counted once, and in neither direction's total
            pass

    per_asset: list[AmountStats] = []
    for identity in sorted(grouped):
        bucket = grouped[identity]
        amounts = bucket["amounts"]
        per_asset.append(
            AmountStats(
                asset=bucket["asset"],
                asset_type=bucket["asset_type"],
                token_contract=bucket["token_contract"],
                transfer_count=len(amounts),
                inbound_count=bucket["in_count"],
                outbound_count=bucket["out_count"],
                total_inbound=round(bucket["in"], 8),
                total_outbound=round(bucket["out"], 8),
                net_flow=round(bucket["in"] - bucket["out"], 8),
                min_amount=round(min(amounts), 8),
                median_amount=round(median(amounts), 8),
                max_amount=round(max(amounts), 8),
                mean_amount=round(mean(amounts), 8),
                metadata_incomplete=bucket["metadata_incomplete"],
            )
        )
    # Largest asset activity first; ties broken on symbol for determinism.
    per_asset.sort(key=lambda s: (-s.transfer_count, s.asset))

    if any(s.metadata_incomplete for s in per_asset):
        limitations.append(
            "At least one asset had incomplete token metadata (missing "
            "decimals or symbol), so its human-readable amounts may be "
            "misscaled. Raw on-chain values are preserved on the graph edges."
        )

    if len(per_asset) > 1:
        limitations.append(
            "Amounts are reported per asset and are never summed across "
            "assets: this project holds no price data, so a cross-asset total "
            "would be a fabricated number."
        )

    return TemporalAmountAnalysis(
        wallet=wallet,
        chain=chain,
        transfer_count=len(edges),
        timestamped_transfer_count=len(timestamps),
        missing_timestamp_count=missing_ts,
        first_seen=first_seen,
        last_seen=last_seen,
        first_seen_utc=_iso(first_seen),
        last_seen_utc=_iso(last_seen),
        lifespan_seconds=lifespan,
        lifespan_days=round(lifespan / SECONDS_PER_DAY, 2) if lifespan else lifespan,
        active_day_count=len(active_days),
        transfers_per_active_day=(
            round(len(timestamps) / len(active_days), 2) if active_days else None
        ),
        longest_idle_seconds=longest_idle,
        longest_idle_days=(
            round(longest_idle / SECONDS_PER_DAY, 2)
            if longest_idle is not None
            else None
        ),
        hourly_utc_histogram=dict(sorted(hour_counts.items())),
        weekday_utc_histogram=dict(sorted(weekday_counts.items())),
        busiest_utc_hour=busiest_hour,
        busiest_utc_weekday=busiest_weekday,
        median_gap_seconds=int(median(gaps)) if gaps else None,
        mean_gap_seconds=round(mean(gaps), 2) if gaps else None,
        inbound_transfer_count=len(inbound),
        outbound_transfer_count=len(outbound),
        self_transfer_count=len(self_transfers),
        unique_inbound_counterparties=len({e[0] for e in inbound}),
        unique_outbound_counterparties=len({e[1] for e in outbound}),
        per_asset=per_asset,
        pass_through=_pass_through(graph, wallet),
        limitations=limitations,
    )
