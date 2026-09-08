"""
BIDIRECTIONAL VASP ATTRIBUTION.

Answers two separate questions across the whole known-VASP dataset:
  A) wallet -> ... -> VASP   (outbound)
  B) VASP   -> ... -> wallet (inbound)

Both are answered by app.tracing.targeted.trace_targeted — one traversal
per direction, for the entire seed set at once, rather than one full
exploratory trace per seed address. That change is what makes attribution
tractable on a real wallet graph: the previous implementation ran a
complete `trace_fund_flow` outward from every seed address, so its cost
grew with the size of the VASP dataset and each of those traces exhausted
its budget on a graph of order 10^4 nodes / 10^5 edges long before
reaching anything, forcing an INCONCLUSIVE that was an artefact of the
algorithm rather than of the evidence.

--------------------------------------------------------------------------
DO NOT CONFUSE DIRECTION (do not remove): "wallet -> VASP" and
"VASP -> wallet" are different, independently-evidenced claims. A path
found in one direction is NEVER reported as evidence for the other.
Only genuinely bidirectional evidence (both searches independently
succeed) is labeled ConnectionDirection.BIDIRECTIONAL. An inbound-only
result is a real, reportable finding in its own right — a deposit from an
exchange is meaningful even if the wallet never sent anything back.

DO NOT CONFUSE GRAPH CONNECTIVITY WITH EVIDENCE (do not remove): the
optional undirected-connectivity check exists purely to surface
"these two addresses are related somewhere in the graph" for
investigator awareness. It NEVER becomes a BidirectionalCandidate and
NEVER affects AttributionStatus — see UndirectedRelation's docstring.

A PATH IS NOT A FUND FLOW (do not remove): every candidate carries a
PathPlausibility grade from app.tracing.quality and an explicit
`limitations` list. A multi-hop candidate whose grade does not support a
fund-flow reading must be reported as a connection between addresses, not
as a movement of value.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Optional

import networkx as nx

from app.attribution.bidirectional_models import (
    BidirectionalAttributionResult,
    BidirectionalCandidate,
    ConnectedButNoValidPath,
    ConnectionDirection,
    DirectionalEvidence,
    ExactIdentityMatch,
    SearchAccounting,
    UndirectedRelation,
)
from app.attribution.candidate_generator import _supporting_pattern_types
from app.attribution.models import (
    AttributionStatus,
    EvidenceTier,
    SeedSourceType,
    VASPSeedEntry,
)
from app.behavior.models import BehaviorPattern
from app.core.config import Settings, get_settings
from app.tracing.models import FundFlowPath
from app.tracing.quality import PlausibilityGrade, best_graded_path
from app.tracing.targeted import (
    SearchDirection,
    SearchStatus,
    TargetedTraceResult,
    trace_targeted,
)

# Seed provenance strong enough that the label itself needs no caveat beyond
# the path evidence. Everything else gets an explicit limitation line.
_STRONG_PROVENANCE = {
    SeedSourceType.OFFICIAL_DISCLOSURE,
    SeedSourceType.DIRECTLY_VERIFIED,
}


def _paths_for(
    trace: TargetedTraceResult, target: str
) -> list[FundFlowPath]:
    """Every path this trace found between the wallet and one target, in the
    trace's own (already chronological) orientation."""
    if trace.direction == SearchDirection.OUTBOUND:
        return [p for p in trace.paths if p.terminal_node == target]
    return [p for p in trace.paths if p.source == target]


def _build_evidence(
    paths: list[FundFlowPath],
    graph: nx.MultiDiGraph,
    settings: Settings,
) -> Optional[DirectionalEvidence]:
    """Turns one direction's traced paths into a single evidence record.

    The reported path is the strongest-graded one (see
    app.tracing.quality.best_graded_path) — not merely the shortest, since
    hop count is not what makes a path trustworthy. `alternative_path_count`
    records how many others existed so the report can say "one of N routes"
    instead of implying it is the only connection.
    """
    if not paths:
        return None

    chosen, plausibility = best_graded_path(paths, graph=graph, settings=settings)
    hops = chosen.hops

    return DirectionalEvidence(
        hop_distance=chosen.hop_count,
        path_addresses=chosen.addresses,
        tx_hashes=[h.tx_hash for h in hops],
        hop_timestamps=[h.timestamp for h in hops],
        evidence_tier=(
            EvidenceTier.DIRECT if chosen.hop_count == 1 else EvidenceTier.INDIRECT
        ),
        amounts=[h.amount for h in hops],
        assets=[h.asset for h in hops],
        block_numbers=[h.block_number for h in hops],
        edge_keys=[h.edge_key for h in hops],
        path_duration_seconds=chosen.path_duration_seconds,
        plausibility=plausibility,
        alternative_path_count=max(0, len(paths) - 1),
    )


def _limitations(
    seed_entry: VASPSeedEntry,
    outbound_evidence: Optional[DirectionalEvidence],
    inbound_evidence: Optional[DirectionalEvidence],
) -> list[str]:
    """States, per candidate, exactly what this evidence does NOT establish.

    Written from the evidence actually present — never a generic disclaimer
    block — so that a strong candidate is not padded with irrelevant caveats
    and a weak one cannot be presented without its own.
    """
    limits: list[str] = []

    for label, evidence in (("Outbound", outbound_evidence), ("Inbound", inbound_evidence)):
        if evidence is None:
            continue
        if evidence.hop_distance > 1:
            limits.append(
                f"{label} evidence is a {evidence.hop_distance}-hop transaction "
                "path, not a direct transfer. A traceable path does NOT prove "
                "the same funds moved end to end — each intermediary may hold "
                "unrelated balances."
            )
        if evidence.plausibility is not None:
            grade = evidence.plausibility.grade
            if grade in (PlausibilityGrade.WEAK, PlausibilityGrade.IMPLAUSIBLE):
                reasons = ", ".join(evidence.plausibility.concern_types)
                limits.append(
                    f"{label} path is graded {grade.value} as fund-flow "
                    f"evidence ({reasons}); report it as a connection between "
                    "addresses, not as a movement of value."
                )
        if evidence.alternative_path_count:
            limits.append(
                f"{label} direction has {evidence.alternative_path_count} "
                "additional traced route(s); the reported path is one example "
                "of the connection, not the only one."
            )

    if outbound_evidence is None:
        limits.append(
            "No outbound path (wallet -> VASP) was found within the search "
            "scope; this candidate rests on inbound evidence only."
        )
    if inbound_evidence is None:
        limits.append(
            "No inbound path (VASP -> wallet) was found within the search "
            "scope; this candidate rests on outbound evidence only."
        )

    if seed_entry.source_type == SeedSourceType.SYNTHETIC_DEMO:
        limits.append(
            "SYNTHETIC DEMO seed entry — this address is not a real VASP "
            "address and this candidate carries no real-world meaning."
        )
    elif seed_entry.source_type not in _STRONG_PROVENANCE:
        limits.append(
            f"Address ownership rests on a '{seed_entry.source_type.value}' "
            "dataset label, not on an official disclosure by the VASP. A "
            "third-party or community label is not proof of ownership."
        )

    if seed_entry.verification_status and seed_entry.verification_status.lower() in (
        "unverified",
        "third_party_labeled",
    ):
        limits.append(
            f"Dataset verification status for this address is "
            f"'{seed_entry.verification_status}'."
        )

    return limits


def _accounting(trace: TargetedTraceResult) -> SearchAccounting:
    if trace.status == SearchStatus.COMPLETE:
        reason = None
    elif trace.edges_limit_hit:
        reason = "exploration budget exhausted -- re-run with a larger budget"
    elif trace.limited_by_observation_depth:
        reason = (
            f"data observed to {trace.observation_depth} hop(s) only -- "
            "deeper hops were never acquired, so acquisition must be widened"
        )
    else:
        reason = "search did not complete"
    return SearchAccounting(
        direction=trace.direction.value,
        graph_node_count=trace.graph_node_count,
        graph_edge_count=trace.graph_edge_count,
        reachable_node_count=trace.reachable_node_count,
        viable_node_count=trace.viable_node_count,
        edges_explored=trace.edges_explored,
        complete=trace.status == SearchStatus.COMPLETE,
        incomplete_reason=reason,
        observation_depth=trace.observation_depth,
        targets_searched=len(trace.target_outcomes),
        targets_reachable=sum(
            1 for o in trace.target_outcomes.values() if o.reachable_within_max_hops
        ),
        time_window_start=trace.time_window_start,
        edges_excluded_by_time_window=trace.edges_excluded_by_time_window,
        notes=list(trace.notes),
    )


def generate_bidirectional_candidates(
    graph: nx.MultiDiGraph,
    wallet: str,
    seed_index: dict[str, VASPSeedEntry],
    settings: Optional[Settings] = None,
    behavior_patterns: Optional[list[BehaviorPattern]] = None,
    max_hops: Optional[int] = None,
    max_paths: Optional[int] = None,
    check_undirected_relations: bool = True,
    max_edges_explored: Optional[int] = None,
    time_window_days: Optional[int] = None,
    observation_depth: Optional[int] = None,
) -> BidirectionalAttributionResult:
    """Attributes one wallet against the whole known-VASP dataset, in both
    directions, using two destination-aware traversals.

    `max_paths` bounds the example paths kept PER TARGET. Hitting that quota
    is a deliberate early stop, not an incomplete search — attribution needs
    concrete evidence, not an exhaustive census of every route to the same
    address. Only exhausting the edge budget, or a data horizon shallower
    than `max_hops`, makes a search incomplete, and only those can turn a
    "not found" into INCONCLUSIVE.

    `observation_depth` is the hop radius the graph's data actually covers.
    See `trace_targeted`.
    """
    wallet = wallet.lower()
    settings = settings or get_settings()
    behavior_patterns = behavior_patterns or []

    targets = list(seed_index.keys())

    common = dict(
        settings=settings,
        max_hops=max_hops,
        max_paths_per_target=max_paths,
        max_edges_explored=max_edges_explored,
        time_window_days=time_window_days,
        observation_depth=observation_depth,
    )
    outbound = trace_targeted(
        graph, wallet, targets, direction=SearchDirection.OUTBOUND, **common
    )
    inbound = trace_targeted(
        graph, wallet, targets, direction=SearchDirection.INBOUND, **common
    )

    # --- Merge the two directions into per-address candidates. ---
    matched = sorted(set(outbound.targets_reached) | set(inbound.targets_reached))
    candidates: list[BidirectionalCandidate] = []

    for addr in matched:
        seed_entry = seed_index[addr]
        outbound_evidence = _build_evidence(
            _paths_for(outbound, addr), graph, settings
        )
        inbound_evidence = _build_evidence(_paths_for(inbound, addr), graph, settings)

        if outbound_evidence and inbound_evidence:
            direction = ConnectionDirection.BIDIRECTIONAL
        elif outbound_evidence:
            direction = (
                ConnectionDirection.DIRECT_OUTBOUND
                if outbound_evidence.hop_distance == 1
                else ConnectionDirection.INDIRECT_OUTBOUND
            )
        else:
            direction = (
                ConnectionDirection.DIRECT_INBOUND
                if inbound_evidence.hop_distance == 1
                else ConnectionDirection.INDIRECT_INBOUND
            )

        supporting: set[str] = set()
        for evidence in (outbound_evidence, inbound_evidence):
            if evidence is not None:
                supporting.update(
                    _supporting_pattern_types(
                        evidence.path_addresses, behavior_patterns
                    )
                )

        candidates.append(
            BidirectionalCandidate(
                vasp_name=seed_entry.vasp_name,
                matched_address=seed_entry.address,
                entity_type=seed_entry.entity_type,
                chain=seed_entry.chain,
                source_type=seed_entry.source_type,
                seed_source=seed_entry.source,
                seed_source_url=seed_entry.source_url,
                seed_confidence_note=seed_entry.confidence_note,
                wallet_role=seed_entry.wallet_role,
                source_evidence_type=seed_entry.source_evidence_type,
                verification_status=seed_entry.verification_status,
                direction=direction,
                outbound_evidence=outbound_evidence,
                inbound_evidence=inbound_evidence,
                supporting_behavioral_patterns=sorted(supporting),
                limitations=_limitations(
                    seed_entry, outbound_evidence, inbound_evidence
                ),
            )
        )

    def _sort_key(c: BidirectionalCandidate):
        return (c.strongest_hop_distance, c.matched_address)

    candidates.sort(key=_sort_key)

    # --- Connected, but with no chronologically-consistent route. Real
    # findings, deliberately kept out of `candidates`. ---
    blocked: list[ConnectedButNoValidPath] = []
    for trace in (outbound, inbound):
        for addr, outcome in sorted(trace.target_outcomes.items()):
            if outcome.chronologically_blocked and addr not in matched:
                entry = seed_index[addr]
                blocked.append(
                    ConnectedButNoValidPath(
                        vasp_name=entry.vasp_name,
                        vasp_address=entry.address,
                        direction_attempted=trace.direction.value,
                        graph_distance=outcome.graph_distance or 0,
                    )
                )

    # --- Exact identity: the wallet is itself a known VASP address. ---
    identity: Optional[ExactIdentityMatch] = None
    if wallet in seed_index:
        entry = seed_index[wallet]
        identity = ExactIdentityMatch(
            vasp_name=entry.vasp_name,
            matched_address=entry.address,
            entity_type=entry.entity_type,
            chain=entry.chain,
            source_type=entry.source_type,
            seed_source=entry.source,
            seed_source_url=entry.source_url,
            seed_confidence_note=entry.confidence_note,
            wallet_role=entry.wallet_role,
            source_evidence_type=entry.source_evidence_type,
            verification_status=entry.verification_status,
        )

    # --- Optional: undirected relationship discovery for addresses with NO
    # directed evidence in either direction. Never evidence. ---
    related_undirected: list[UndirectedRelation] = []
    if check_undirected_relations and wallet in graph:
        remaining = {
            addr: entry
            for addr, entry in seed_index.items()
            if addr not in matched and addr != wallet and addr in graph
        }
        if remaining:
            undirected_view = graph.to_undirected(as_view=True)
            try:
                distances = nx.single_source_shortest_path_length(
                    undirected_view, wallet, cutoff=outbound.max_hops
                )
            except nx.NetworkXError:
                distances = {}
            for addr, entry in remaining.items():
                d = distances.get(addr)
                if d is not None and d > 0:
                    related_undirected.append(
                        UndirectedRelation(
                            vasp_name=entry.vasp_name,
                            vasp_address=entry.address,
                            undirected_distance=d,
                        )
                    )
            related_undirected.sort(
                key=lambda r: (r.undirected_distance, r.vasp_address)
            )

    # --- Status. A budget stop or a data horizon shorter than MAX_HOPS are the
    # only things that can turn "not found" into INCONCLUSIVE; exhausting
    # MAX_HOPS on fully-observed data is the configured scope of the
    # investigation, and reachability within it was determined exactly. ---
    outbound_truncated = outbound.status == SearchStatus.INCOMPLETE
    inbound_truncated = inbound.status == SearchStatus.INCOMPLETE
    horizon_limited = (
        outbound.limited_by_observation_depth
        or inbound.limited_by_observation_depth
    )
    observed_depth = (
        outbound.observation_depth
        if outbound.observation_depth is not None
        else inbound.observation_depth
    )
    budget_limited = outbound.edges_limit_hit or inbound.edges_limit_hit
    notes: list[str] = []

    if candidates or identity is not None:
        status = AttributionStatus.MATCH_FOUND
    elif outbound_truncated or inbound_truncated:
        status = AttributionStatus.INCONCLUSIVE
        if horizon_limited:
            # `observed_depth` counts the LEADING hops that were acquired in
            # full, so the first hop with a gap in it is hop `observed_depth`
            # itself: some of its addresses were never fetched, and the edges
            # leading onward from them are therefore absent from the data
            # rather than searched and found empty.
            gap_hop = observed_depth or 0
            notes.append(
                "No known VASP address was connected in either direction "
                f"within the {observed_depth} hop(s) this dataset observes. "
                f"Not every address at hop {gap_hop} had its own transactions "
                "acquired, so the graph holds no edges leading onward from "
                f"there and a route of {gap_hop + 1}+ hops could not be seen "
                "at all — it was not searched and found absent. Reachability "
                f"to {observed_depth} hop(s) was exhaustive, so this is a "
                f"confirmed negative at {observed_depth} hop(s) and "
                f"INCONCLUSIVE for the requested MAX_HOPS="
                f"{outbound.max_hops}. Acquiring the remaining addresses at "
                f"hop {gap_hop} would be required to answer the deeper "
                "question."
            )
        if budget_limited:
            notes.append(
                "No known VASP address was connected in either direction, but the "
                "path enumeration was cut short by the edge budget "
                "(TARGETED_TRACE_MAX_EDGES_EXPLORED) before every route within "
                f"MAX_HOPS={outbound.max_hops} was examined. This is "
                "INCONCLUSIVE, not a confirmed negative."
            )
    else:
        status = AttributionStatus.NONE
        notes.append(
            f"No known VASP seed address was reached from or to '{wallet}' "
            f"within MAX_HOPS={outbound.max_hops}, in either direction. "
            "Breadth-first reachability was exhaustive and subject to no "
            "budget, so this is a COMPLETE negative within that depth — not "
            "merely an absence of results."
        )

    if status == AttributionStatus.MATCH_FOUND and (
        outbound_truncated or inbound_truncated
    ):
        if budget_limited:
            notes.append(
                "One or more searches hit the edge budget, so the candidate list "
                "below is a lower bound: further connections may exist that were "
                "not enumerated."
            )
        if horizon_limited:
            notes.append(
                "The candidate list below is a lower bound: this dataset "
                f"observes only {observed_depth} hop(s) from the wallet, so "
                "connections that would require a deeper route are absent "
                "from the data rather than ruled out."
            )

    if identity is not None:
        notes.append(
            f"The investigated address is itself in the known-VASP dataset "
            f"('{identity.vasp_name}'). That is an exact address identity, "
            "reported separately from any traced path."
        )

    incomplete_targets = sorted(
        {
            addr
            for trace in (outbound, inbound)
            for addr, outcome in trace.target_outcomes.items()
            if outcome.search_incomplete
        }
    )
    if incomplete_targets:
        notes.append(
            f"{len(incomplete_targets)} known VASP address(es) are reachable "
            "in the graph but their routes could not be fully enumerated "
            "within the edge budget: no conclusion is drawn about them in "
            "either direction."
        )

    if blocked:
        notes.append(
            f"{len(blocked)} known VASP address(es) are connected to the "
            "wallet by directed edges within the hop limit, but every route "
            "was rejected because its transfer timestamps run backwards — "
            "funds cannot be forwarded before they arrive. Reported in "
            "connected_but_no_valid_path, NOT as candidates."
        )

    if related_undirected:
        notes.append(
            f"{len(related_undirected)} known VASP address(es) are reachable "
            "only through the graph's undirected projection (no directed "
            "evidence in either direction) — see "
            "related_by_undirected_graph_only. This is NOT attribution "
            "evidence and does not change the status above."
        )

    unsupported = [
        c
        for c in candidates
        if not any(
            e.plausibility is not None and e.plausibility.supports_fund_flow_narrative
            for e in (c.outbound_evidence, c.inbound_evidence)
            if e is not None
        )
    ]
    if unsupported:
        notes.append(
            f"{len(unsupported)} of {len(candidates)} candidate(s) have no "
            "path graded as supporting a fund-flow reading. Those are "
            "transaction-graph connections between the addresses; presenting "
            "them as movements of funds would not be supportable by the "
            "evidence. See each candidate's limitations."
        )

    seed_contains_synthetic = any(
        entry.source_type == SeedSourceType.SYNTHETIC_DEMO
        for entry in seed_index.values()
    )
    if seed_contains_synthetic:
        notes.append(
            "WARNING: the loaded VASP seed dataset contains SYNTHETIC_DEMO "
            "entries. Results from this run are not real-world attribution."
        )

    return BidirectionalAttributionResult(
        wallet=wallet,
        status=status,
        candidates=candidates,
        related_by_undirected_graph_only=related_undirected,
        connected_but_no_valid_path=blocked,
        exact_identity_match=identity,
        max_hops=outbound.max_hops,
        outbound_search_truncated=outbound_truncated,
        inbound_searches_truncated=inbound_truncated,
        outbound_accounting=_accounting(outbound),
        inbound_accounting=_accounting(inbound),
        seed_address_count=len(seed_index),
        seed_contains_synthetic=seed_contains_synthetic,
        notes=notes,
    )
