"""
FUND-FLOW CANDIDATE PLAUSIBILITY GRADING.

Why this module exists
----------------------
A traceable path is not automatically a plausible fund flow. Validating the
real demonstration wallet produced a path that is *technically* perfect —
every edge exists on-chain, timestamps run strictly forward — and yet is
almost certainly meaningless as a fund flow:

    Kraken -> 0x04ed92... : 0.169 ETH   (2019-07-20)
    0x04ed92... -> 0x00000000000444...4c5dc75cb358380d2e3de08a90 : 3.06 MKR (2025-01-30)
    0x0000...a90 -> investigated wallet : 30707.99 GIVE (2026-08-04)

Three hops, three different assets, seven years end to end, and the middle
address is the single highest-degree node in the graph (total degree 5099
against a median of 1) — a shared DEX/router contract that thousands of
unrelated parties touch. Reporting that as "funds flowed from Kraken to this
wallet" would be exactly the fabrication the specification forbids.

So the grade is derived from observable evidence only, and every downgrade
carries the concrete measurement that caused it. Nothing here is a
probability or a score: the output is a discrete grade plus the reasons.

--------------------------------------------------------------------------
Even the strongest grade this module can assign is NOT proof of fund
continuity. PLAUSIBLE means "nothing observable argues against it", which is
a much weaker claim than "the same funds moved". Only a single direct
transfer (DIRECT_TRANSFER) needs no continuity inference at all, because
there is no intermediary to assume anything about.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import networkx as nx
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.tracing.models import FundFlowPath


class PlausibilityGrade(str, Enum):
    """How much weight a traced path can bear as fund-flow evidence."""

    # Exactly one hop: the two addresses transacted directly. No continuity
    # assumption is involved, so this is the only grade that is not an
    # inference about intermediaries.
    DIRECT_TRANSFER = "DIRECT_TRANSFER"
    # Multi-hop, and nothing observable argues against continuity. Still not
    # proof — the intermediaries' other balances are unknown.
    PLAUSIBLE = "PLAUSIBLE"
    # One observable factor argues against continuity.
    WEAK = "WEAK"
    # Two or more factors argue against continuity. The addresses are
    # connected, but this path should not be presented as a fund flow.
    IMPLAUSIBLE = "IMPLAUSIBLE"


class PlausibilityConcern(str, Enum):
    ASSET_CHANGED = "ASSET_CHANGED"
    LONG_TIME_GAP = "LONG_TIME_GAP"
    HIGH_THROUGHPUT_INTERMEDIARY = "HIGH_THROUGHPUT_INTERMEDIARY"
    AMOUNT_INCREASED = "AMOUNT_INCREASED"
    UNVERIFIABLE_CHRONOLOGY = "UNVERIFIABLE_CHRONOLOGY"


class ConcernDetail(BaseModel):
    """One reason a path was downgraded, with the measurement behind it.

    `observed` and `threshold` are always populated where a threshold
    exists, so a report can state "gap was 217 days, threshold is 30 days"
    rather than an unexplained verdict.
    """

    concern: PlausibilityConcern
    hop_index: Optional[int] = None
    address: Optional[str] = None
    observed: str
    threshold: Optional[str] = None
    explanation: str


class HubIntermediary(BaseModel):
    address: str
    hop_index: int
    in_degree: int
    out_degree: int
    total_degree: int


class PathPlausibility(BaseModel):
    grade: PlausibilityGrade
    concerns: list[ConcernDetail] = []

    hop_count: int
    assets_in_order: list[str] = []
    single_asset_throughout: bool = True

    total_duration_seconds: Optional[int] = None
    max_hop_gap_seconds: Optional[int] = None
    hop_gaps_seconds: list[Optional[int]] = []

    intermediaries: list[str] = []
    hub_intermediaries: list[HubIntermediary] = []
    contract_intermediary_count: int = 0

    # Always present, always the same warning: no grade here proves that the
    # same funds moved from end to end.
    interpretation: str = ""

    @property
    def concern_types(self) -> list[str]:
        return [c.concern.value for c in self.concerns]

    @property
    def supports_fund_flow_narrative(self) -> bool:
        """True only for grades that can carry a fund-flow reading at all.
        Attribution uses this to decide what language a report may use — it
        is never a claim of proof."""
        return self.grade in (
            PlausibilityGrade.DIRECT_TRANSFER,
            PlausibilityGrade.PLAUSIBLE,
        )


_INTERPRETATION = {
    PlausibilityGrade.DIRECT_TRANSFER: (
        "Single direct transfer between the two addresses. No intermediary "
        "is involved, so no fund-continuity assumption is required — this is "
        "the strongest form of address-level evidence available."
    ),
    PlausibilityGrade.PLAUSIBLE: (
        "Multi-hop path with no observable factor arguing against fund "
        "continuity. This does NOT prove the same funds moved end to end: "
        "each intermediary may hold unrelated balances. Treat as a fund-flow "
        "CANDIDATE requiring further verification."
    ),
    PlausibilityGrade.WEAK: (
        "Multi-hop path with one observable factor arguing against fund "
        "continuity. The transaction path is real, but it should be reported "
        "as a connection between addresses rather than as a fund flow."
    ),
    PlausibilityGrade.IMPLAUSIBLE: (
        "Multi-hop path with several observable factors arguing against fund "
        "continuity. The addresses are connected in the transaction graph, "
        "but presenting this as a movement of funds would not be supportable "
        "by the evidence. Report the connection, not a flow."
    ),
}


def _degree(graph: Optional[nx.MultiDiGraph], address: str) -> tuple[int, int]:
    if graph is None or address not in graph:
        return 0, 0
    return int(graph.in_degree(address)), int(graph.out_degree(address))


def assess_path(
    path: FundFlowPath,
    graph: Optional[nx.MultiDiGraph] = None,
    settings: Optional[Settings] = None,
) -> PathPlausibility:
    """Grades one traced path on observable evidence only.

    `graph` is optional: without it, hub detection is skipped (and its
    absence is not silently treated as "no hubs found" — the concern simply
    cannot be evaluated, and hub_intermediaries stays empty).
    """
    settings = settings or get_settings()
    hub_threshold = settings.path_quality_hub_degree_threshold
    gap_threshold = settings.path_quality_max_hop_gap_seconds

    hops = path.hops
    assets = [h.asset for h in hops]
    addresses = path.addresses
    intermediaries = addresses[1:-1] if len(addresses) > 2 else []

    result = PathPlausibility(
        grade=PlausibilityGrade.DIRECT_TRANSFER,
        hop_count=len(hops),
        assets_in_order=assets,
        intermediaries=intermediaries,
        total_duration_seconds=path.path_duration_seconds,
    )

    if not hops:
        result.grade = PlausibilityGrade.IMPLAUSIBLE
        result.interpretation = "Empty path — no evidence at all."
        return result

    concerns: list[ConcernDetail] = []

    # --- Asset continuity ---
    distinct_assets = []
    for asset in assets:
        if asset not in distinct_assets:
            distinct_assets.append(asset)
    result.single_asset_throughout = len(distinct_assets) <= 1
    if not result.single_asset_throughout:
        for i in range(1, len(assets)):
            if assets[i] != assets[i - 1]:
                concerns.append(
                    ConcernDetail(
                        concern=PlausibilityConcern.ASSET_CHANGED,
                        hop_index=i,
                        address=hops[i].from_address,
                        observed=f"asset changes from {assets[i - 1]} to {assets[i]}",
                        explanation=(
                            "The asset received by this intermediary is not the "
                            "asset it sent on. Without independent swap or "
                            "bridge evidence tying the two, the outgoing "
                            "transfer cannot be treated as the same value "
                            "continuing to move."
                        ),
                    )
                )
                break  # one ASSET_CHANGED concern is enough; details recorded

    # --- Timing ---
    gaps: list[Optional[int]] = []
    worst_gap: Optional[int] = None
    unverifiable = False
    for i in range(1, len(hops)):
        prev_ts, this_ts = hops[i - 1].timestamp, hops[i].timestamp
        if prev_ts is None or this_ts is None or prev_ts == 0 or this_ts == 0:
            gaps.append(None)
            unverifiable = True
            continue
        gap = this_ts - prev_ts
        gaps.append(gap)
        if worst_gap is None or gap > worst_gap:
            worst_gap = gap
    result.hop_gaps_seconds = gaps
    result.max_hop_gap_seconds = worst_gap

    if worst_gap is not None and worst_gap > gap_threshold:
        worst_index = gaps.index(worst_gap) + 1
        concerns.append(
            ConcernDetail(
                concern=PlausibilityConcern.LONG_TIME_GAP,
                hop_index=worst_index,
                address=hops[worst_index].from_address,
                observed=f"{worst_gap} seconds ({worst_gap // 86_400} days) between hops",
                threshold=f"{gap_threshold} seconds ({gap_threshold // 86_400} days)",
                explanation=(
                    "The intermediary held value for far longer than a "
                    "forwarding pattern would imply. Over that period its "
                    "balance was almost certainly commingled with unrelated "
                    "funds, so the outgoing transfer cannot be tied to the "
                    "incoming one."
                ),
            )
        )

    if unverifiable and len(hops) > 1:
        concerns.append(
            ConcernDetail(
                concern=PlausibilityConcern.UNVERIFIABLE_CHRONOLOGY,
                observed="one or more hops carry no usable timestamp",
                explanation=(
                    "Chronological ordering could not be verified across the "
                    "whole path, so the sequence is unconfirmed rather than "
                    "wrong. Recorded instead of assuming an order."
                ),
            )
        )

    # --- Hub intermediaries ---
    if graph is not None:
        for offset, address in enumerate(intermediaries):
            hop_index = offset + 1
            in_deg, out_deg = _degree(graph, address)
            total = in_deg + out_deg
            if total >= hub_threshold:
                result.hub_intermediaries.append(
                    HubIntermediary(
                        address=address,
                        hop_index=hop_index,
                        in_degree=in_deg,
                        out_degree=out_deg,
                        total_degree=total,
                    )
                )
        if result.hub_intermediaries:
            worst = max(result.hub_intermediaries, key=lambda h: h.total_degree)
            concerns.append(
                ConcernDetail(
                    concern=PlausibilityConcern.HIGH_THROUGHPUT_INTERMEDIARY,
                    hop_index=worst.hop_index,
                    address=worst.address,
                    observed=(
                        f"intermediary has total degree {worst.total_degree} "
                        f"({worst.in_degree} in / {worst.out_degree} out)"
                    ),
                    threshold=f"{hub_threshold}",
                    explanation=(
                        "This is a high-throughput address — a router, pool, "
                        "bridge, or large custodial wallet that thousands of "
                        "unrelated parties transact with. A path passing "
                        "through it carries almost no information about a "
                        "relationship between the path's endpoints."
                    ),
                )
            )

    result.contract_intermediary_count = sum(
        1 for h in hops if h.is_contract_interaction
    )

    # --- Amount consistency, only comparable within the same asset ---
    for i in range(1, len(hops)):
        if assets[i] != assets[i - 1]:
            continue  # already covered by ASSET_CHANGED; amounts incomparable
        incoming, outgoing = hops[i - 1].amount, hops[i].amount
        if incoming > 0 and outgoing > incoming:
            concerns.append(
                ConcernDetail(
                    concern=PlausibilityConcern.AMOUNT_INCREASED,
                    hop_index=i,
                    address=hops[i].from_address,
                    observed=(
                        f"forwarded {outgoing} {assets[i]} after receiving "
                        f"{incoming} {assets[i - 1]}"
                    ),
                    explanation=(
                        "More value left the intermediary than arrived on the "
                        "previous hop, so the outgoing transfer cannot consist "
                        "solely of the incoming funds. At most a part of it "
                        "could be related."
                    ),
                )
            )
            break

    result.concerns = concerns

    if len(hops) == 1:
        # A single direct transfer is graded on its own terms. Concerns that
        # only make sense for intermediaries cannot apply, and a missing
        # timestamp on a lone hop does not weaken the fact of the transfer.
        result.grade = PlausibilityGrade.DIRECT_TRANSFER
        result.concerns = [
            c
            for c in concerns
            if c.concern
            not in (
                PlausibilityConcern.HIGH_THROUGHPUT_INTERMEDIARY,
                PlausibilityConcern.UNVERIFIABLE_CHRONOLOGY,
            )
        ]
    elif not concerns:
        result.grade = PlausibilityGrade.PLAUSIBLE
    elif len(concerns) == 1:
        result.grade = PlausibilityGrade.WEAK
    else:
        result.grade = PlausibilityGrade.IMPLAUSIBLE

    result.interpretation = _INTERPRETATION[result.grade]
    return result


def best_graded_path(
    paths: list[FundFlowPath],
    graph: Optional[nx.MultiDiGraph] = None,
    settings: Optional[Settings] = None,
) -> Optional[tuple[FundFlowPath, PathPlausibility]]:
    """Picks the strongest-evidence path from a set, preferring plausibility
    over brevity.

    A DIRECT_TRANSFER always wins. Otherwise a PLAUSIBLE 3-hop path beats a
    WEAK 2-hop one: hop count is not what makes a path trustworthy, so
    brevity only breaks ties *within* a grade. Ties break finally on the
    concatenated edge keys, so the same evidence is chosen every run.
    """
    if not paths:
        return None

    grade_rank = {
        PlausibilityGrade.DIRECT_TRANSFER: 0,
        PlausibilityGrade.PLAUSIBLE: 1,
        PlausibilityGrade.WEAK: 2,
        PlausibilityGrade.IMPLAUSIBLE: 3,
    }

    graded = [(p, assess_path(p, graph=graph, settings=settings)) for p in paths]
    graded.sort(
        key=lambda item: (
            grade_rank[item[1].grade],
            len(item[1].concerns),
            item[0].hop_count,
            "|".join(h.edge_key for h in item[0].hops),
        )
    )
    return graded[0]
