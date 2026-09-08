"""
MACRO MILESTONE 6, PHASE 1 tests — bidirectional VASP attribution.

Fully offline, synthetic addresses only (source_type=SYNTHETIC_DEMO) —
never the real production seed set. Mirrors tests/test_attribution.py's
fixture-building conventions.
"""

from __future__ import annotations

import networkx as nx

from app.attribution.bidirectional import generate_bidirectional_candidates
from app.attribution.bidirectional_models import ConnectionDirection
from app.attribution.candidate_generator import generate_candidates
from app.attribution.matcher import build_seed_index
from app.attribution.models import (
    AttributionStatus,
    EvidenceTier,
    SeedSourceType,
    VASPSeedEntry,
)
from app.behavior.models import BehaviorPattern, PatternType
from app.tracing.quality import PlausibilityGrade
from app.tracing.tracer import trace_fund_flow

SOURCE = "0xaaaa111111111111111111111111111111111a"
NODE_B = "0xbbbb111111111111111111111111111111111b"
NODE_C = "0xcccc111111111111111111111111111111111c"
NODE_X = "0xffff111111111111111111111111111111111f"
VASP_1 = "0xdddd111111111111111111111111111111111d"
VASP_2 = "0xeeee111111111111111111111111111111111e"


def add_edge(
    graph,
    u,
    v,
    tx_hash,
    occurrence=0,
    amount=1.0,
    ts=1_700_000_000,
    asset="ETH",
    block=None,
):
    key = f"{tx_hash}#{occurrence}"
    graph.add_edge(
        u,
        v,
        key=key,
        amount=amount,
        asset=asset,
        asset_type="NATIVE" if asset == "ETH" else "ERC20",
        timestamp=ts,
        block_number=block,
        tx_hash=tx_hash,
        transfer_index=occurrence,
        chain="ethereum",
        transfer_type="TRANSFER",
        status="SUCCESS",
    )
    return key


def demo_seed_entry(address, vasp_name) -> VASPSeedEntry:
    return VASPSeedEntry(
        address=address,
        vasp_name=vasp_name,
        entity_type="demo_exchange",
        chain="ethereum",
        source="synthetic_demo_fixture",
        source_type=SeedSourceType.SYNTHETIC_DEMO,
        confidence_note="Synthetic address used only to validate bidirectional attribution. Not a real VASP.",
    )


def seed_index_with(*entries):
    return build_seed_index(list(entries))


# --- Outbound (mirrors M4, via the unmodified generate_candidates path) ----


def test_direct_outbound_only():
    g = nx.MultiDiGraph()
    add_edge(g, SOURCE, VASP_1, "0xtx1", ts=100)

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(VASP_1, "VASP_1")), max_hops=3
    )

    assert result.status == AttributionStatus.MATCH_FOUND
    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert c.direction == ConnectionDirection.DIRECT_OUTBOUND
    assert c.outbound_evidence is not None
    assert c.outbound_evidence.hop_distance == 1
    assert c.inbound_evidence is None


def test_indirect_outbound_only():
    g = nx.MultiDiGraph()
    add_edge(g, SOURCE, NODE_B, "0xtx1", ts=100)
    add_edge(g, NODE_B, VASP_1, "0xtx2", ts=200)

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(VASP_1, "VASP_1")), max_hops=3
    )

    assert result.status == AttributionStatus.MATCH_FOUND
    c = result.candidates[0]
    assert c.direction == ConnectionDirection.INDIRECT_OUTBOUND
    assert c.outbound_evidence.hop_distance == 2
    assert c.outbound_evidence.path_addresses == [SOURCE, NODE_B, VASP_1]
    assert c.inbound_evidence is None


# --- Inbound (new in Phase 1) -----------------------------------------------


def test_direct_inbound_only():
    g = nx.MultiDiGraph()
    add_edge(g, VASP_1, SOURCE, "0xtx1", ts=100)  # VASP -> SOURCE, no reverse edge

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(VASP_1, "VASP_1")), max_hops=3
    )

    assert result.status == AttributionStatus.MATCH_FOUND
    c = result.candidates[0]
    assert c.direction == ConnectionDirection.DIRECT_INBOUND
    assert c.outbound_evidence is None
    assert c.inbound_evidence is not None
    assert c.inbound_evidence.hop_distance == 1
    assert c.inbound_evidence.path_addresses == [VASP_1, SOURCE]


def test_indirect_inbound_only_matches_real_world_reverse_scenario():
    """Mirrors the reported real-world shape: a known VASP address has a
    directed multi-hop path INTO the investigated wallet, but the wallet
    has no outbound path to the VASP at all. Must be discovered from the
    graph structure alone — nothing here is hardcoded to this address
    pair; the same generate_bidirectional_candidates() call handles any
    wallet/VASP pair.
    """
    g = nx.MultiDiGraph()
    add_edge(g, VASP_1, NODE_B, "0xtx1", ts=100)
    add_edge(g, NODE_B, NODE_C, "0xtx2", ts=200)
    add_edge(g, NODE_C, SOURCE, "0xtx3", ts=300)
    # A decoy outbound edge that does NOT reach any known VASP, to make sure
    # the outbound search genuinely finds nothing rather than the test
    # accidentally never exercising it.
    add_edge(g, SOURCE, NODE_X, "0xdecoy", ts=400)

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(VASP_1, "VASP_1")), max_hops=4
    )

    assert result.status == AttributionStatus.MATCH_FOUND
    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert c.direction == ConnectionDirection.INDIRECT_INBOUND
    assert c.outbound_evidence is None
    assert c.inbound_evidence.hop_distance == 3
    assert c.inbound_evidence.path_addresses == [VASP_1, NODE_B, NODE_C, SOURCE]
    assert c.inbound_evidence.tx_hashes == ["0xtx1", "0xtx2", "0xtx3"]


def test_inbound_search_respects_max_hops_like_outbound():
    g = nx.MultiDiGraph()
    add_edge(g, VASP_1, NODE_B, "0xtx1", ts=100)
    add_edge(g, NODE_B, NODE_C, "0xtx2", ts=200)
    add_edge(g, NODE_C, SOURCE, "0xtx3", ts=300)  # 3 hops from VASP_1 to SOURCE

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(VASP_1, "VASP_1")), max_hops=2
    )

    # Beyond the configured search depth -> a complete negative, not a match.
    assert result.status == AttributionStatus.NONE
    assert result.candidates == []


# --- Bidirectional -----------------------------------------------------------


def test_bidirectional_when_both_directions_have_evidence():
    g = nx.MultiDiGraph()
    add_edge(g, SOURCE, VASP_1, "0xout", ts=100)
    add_edge(g, VASP_1, SOURCE, "0xin", ts=200)

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(VASP_1, "VASP_1")), max_hops=3
    )

    assert result.status == AttributionStatus.MATCH_FOUND
    c = result.candidates[0]
    assert c.direction == ConnectionDirection.BIDIRECTIONAL
    assert c.outbound_evidence is not None
    assert c.inbound_evidence is not None
    assert c.outbound_evidence.tx_hashes == ["0xout"]
    assert c.inbound_evidence.tx_hashes == ["0xin"]


# --- No match / inconclusive -------------------------------------------------


def test_none_when_fully_searched_and_nothing_found():
    g = nx.MultiDiGraph()
    add_edge(g, SOURCE, NODE_X, "0xtx1", ts=100)  # goes nowhere near a VASP

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(VASP_1, "VASP_1")), max_hops=3
    )

    assert result.status == AttributionStatus.NONE
    assert result.candidates == []
    assert result.outbound_search_truncated is False
    assert result.inbound_searches_truncated is False


def test_inconclusive_when_outbound_search_truncated():
    """A search stopped by the EDGE BUDGET must be reported as INCONCLUSIVE,
    never as NONE.

    The target here is genuinely reachable (SOURCE -> mid_i -> VASP_1) and the
    budget is starved so enumeration cannot reach it. This is the only
    condition that can make a search incomplete: reachability itself is
    determined by an unbudgeted breadth-first sweep, so a target that is
    absent from the graph is a complete negative no matter how small the
    budget is (see test_tiny_budget_does_not_make_an_unreachable_target_...).
    """
    g = nx.MultiDiGraph()
    for i in range(20):
        mid = f"0xdest{i:03d}" + "0" * 10
        add_edge(g, SOURCE, mid, f"0xtx{i}", ts=1000 + i)
        add_edge(g, mid, VASP_1, f"0xty{i}", ts=2000 + i)

    result = generate_bidirectional_candidates(
        g,
        SOURCE,
        seed_index_with(demo_seed_entry(VASP_1, "VASP_1")),
        max_hops=2,
        max_edges_explored=1,
    )

    assert result.status == AttributionStatus.INCONCLUSIVE
    assert result.candidates == []
    assert result.outbound_search_truncated is True
    assert result.outbound_accounting.complete is False
    assert any("inconclusive" in n.lower() for n in result.notes)
    # The reachable-but-unenumerated target must be named as unresolved, not
    # quietly omitted.
    assert any("could not be fully enumerated" in n for n in result.notes)


def test_inconclusive_never_reported_as_none_even_with_inbound_search():
    """A budget truncation on the INBOUND (VASP -> wallet) side alone must
    still produce INCONCLUSIVE, not NONE — same rule as outbound."""
    g = nx.MultiDiGraph()
    for i in range(20):
        mid = f"0xdest{i:03d}" + "0" * 10
        add_edge(g, VASP_1, mid, f"0xtx{i}", ts=1000 + i)
        add_edge(g, mid, SOURCE, f"0xty{i}", ts=2000 + i)

    result = generate_bidirectional_candidates(
        g,
        SOURCE,
        seed_index_with(demo_seed_entry(VASP_1, "VASP_1")),
        max_hops=2,
        max_edges_explored=1,
    )

    assert result.status == AttributionStatus.INCONCLUSIVE
    assert result.inbound_searches_truncated is True
    assert result.outbound_search_truncated is False  # nothing to search that way
    assert result.inbound_accounting.complete is False


# --- A complete negative must not be softened into INCONCLUSIVE -------------


def test_tiny_budget_does_not_make_an_unreachable_target_inconclusive():
    """The semantic improvement over the previous implementation: because
    breadth-first reachability is exhaustive and unbudgeted, a target that
    cannot be reached within MAX_HOPS is a COMPLETE negative regardless of
    how small the enumeration budget is. Reporting INCONCLUSIVE here would
    understate a result the search genuinely established."""
    g = nx.MultiDiGraph()
    for i in range(20):
        add_edge(g, SOURCE, f"0xdest{i:03d}" + "0" * 10, f"0xtx{i}", ts=1000 + i)

    result = generate_bidirectional_candidates(
        g,
        SOURCE,
        seed_index_with(demo_seed_entry(VASP_1, "VASP_1")),
        max_hops=1,
        max_edges_explored=1,
    )

    assert result.status == AttributionStatus.NONE
    assert result.outbound_search_truncated is False
    assert result.inbound_searches_truncated is False
    assert any("COMPLETE negative" in n for n in result.notes)


def test_large_irrelevant_fanout_still_yields_a_complete_negative():
    """The wallet has a big neighbourhood, none of which leads to a known
    VASP. Destination-aware pruning means that fan-out is never enumerated,
    so the negative stays complete instead of being budgeted away."""
    g = nx.MultiDiGraph()
    for i in range(300):
        noise = f"0xnoise{i:04d}" + "0" * 8
        add_edge(g, SOURCE, noise, f"0xn{i}", ts=1000 + i)
        add_edge(g, noise, f"0xleaf{i:04d}" + "0" * 9, f"0xm{i}", ts=2000 + i)

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(VASP_1, "VASP_1")), max_hops=4
    )

    assert result.status == AttributionStatus.NONE
    assert result.outbound_accounting.complete is True
    assert result.outbound_accounting.reachable_node_count > 300
    # Nothing can lie on a route to the target, so nothing was walked.
    assert result.outbound_accounting.viable_node_count == 0
    assert result.outbound_accounting.edges_explored == 0


def test_path_quota_is_an_early_stop_not_an_incomplete_search():
    """Keeping only N example paths per target is a deliberate early
    termination — attribution needs evidence, not a census of every route. It
    must not be reported as a truncated (INCONCLUSIVE-producing) search."""
    g = nx.MultiDiGraph()
    for i in range(20):
        mid = f"0xmid{i:03d}" + "0" * 11
        add_edge(g, SOURCE, mid, f"0xa{i}", ts=1000)
        add_edge(g, mid, VASP_1, f"0xb{i}", ts=2000)

    result = generate_bidirectional_candidates(
        g,
        SOURCE,
        seed_index_with(demo_seed_entry(VASP_1, "VASP_1")),
        max_hops=2,
        max_paths=3,
    )

    assert result.status == AttributionStatus.MATCH_FOUND
    assert result.outbound_search_truncated is False
    assert result.outbound_accounting.complete is True
    c = result.candidates[0]
    # 3 paths kept: 1 reported + 2 recorded as alternatives.
    assert c.outbound_evidence.alternative_path_count == 2


# --- Connected, but no chronologically-consistent route ---------------------


def test_chronologically_impossible_route_is_reported_but_is_not_a_candidate():
    """SOURCE -> A at t=500, A -> VASP_1 at t=100. The addresses ARE connected,
    but funds cannot leave A before they arrive. That is a real finding and
    must be surfaced — as a connection, never as a fund-flow candidate."""
    g = nx.MultiDiGraph()
    add_edge(g, SOURCE, NODE_B, "0xtx1", ts=500)
    add_edge(g, NODE_B, VASP_1, "0xtx2", ts=100)

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(VASP_1, "VASP_1")), max_hops=3
    )

    assert result.candidates == []
    assert result.status == AttributionStatus.NONE
    assert len(result.connected_but_no_valid_path) == 1
    blocked = result.connected_but_no_valid_path[0]
    assert blocked.vasp_address == VASP_1
    assert blocked.direction_attempted == "OUTBOUND"
    assert blocked.graph_distance == 2
    assert "NOT as a fund-flow candidate" in blocked.note


# --- The wallet is itself a known VASP address ------------------------------


def test_wallet_that_is_itself_a_seed_address_is_an_exact_identity_match():
    g = nx.MultiDiGraph()
    add_edge(g, SOURCE, NODE_X, "0xtx1", ts=100)

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(SOURCE, "VASP_SELF")), max_hops=3
    )

    assert result.status == AttributionStatus.MATCH_FOUND
    assert result.exact_identity_match is not None
    assert result.exact_identity_match.vasp_name == "VASP_SELF"
    # An identity is not a traced path, so it never becomes a candidate...
    assert result.candidates == []
    # ...and never appears as a "related" address either.
    assert result.related_by_undirected_graph_only == []
    assert any("exact address identity" in n for n in result.notes)


# --- Plausibility grading and per-candidate limitations ---------------------


def test_multi_hop_candidate_carries_a_plausibility_grade_and_limitations():
    g = nx.MultiDiGraph()
    add_edge(g, SOURCE, NODE_B, "0xtx1", ts=100)
    add_edge(g, NODE_B, VASP_1, "0xtx2", ts=700)

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(VASP_1, "VASP_1")), max_hops=3
    )

    c = result.candidates[0]
    assert c.outbound_evidence.plausibility is not None
    assert c.outbound_evidence.plausibility.grade is PlausibilityGrade.PLAUSIBLE
    assert c.outbound_evidence.evidence_tier is EvidenceTier.INDIRECT
    # A 2-hop path must never be presented without the continuity caveat.
    assert any("does NOT prove" in limit for limit in c.limitations)
    # Synthetic seed data must say so, in the candidate itself.
    assert any("SYNTHETIC DEMO" in limit for limit in c.limitations)


def test_direct_transfer_candidate_is_graded_direct_and_carries_full_evidence():
    g = nx.MultiDiGraph()
    add_edge(g, SOURCE, VASP_1, "0xtx1", ts=100, amount=2.5, block=18_000_001)

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(VASP_1, "VASP_1")), max_hops=3
    )

    e = result.candidates[0].outbound_evidence
    assert e.plausibility.grade is PlausibilityGrade.DIRECT_TRANSFER
    assert e.evidence_tier is EvidenceTier.DIRECT
    assert e.amounts == [2.5]
    assert e.assets == ["ETH"]
    assert e.block_numbers == [18_000_001]
    assert e.edge_keys == ["0xtx1#0"]
    assert e.alternative_path_count == 0


def test_stronger_evidence_is_preferred_over_a_shorter_weaker_path():
    """Two routes to the same VASP: a 2-hop one whose intermediary held value
    for over a year, and a clean 3-hop one. The reported evidence must be the
    3-hop path — hop count is not what makes a path trustworthy."""
    g = nx.MultiDiGraph()
    t = 1_600_000_000
    add_edge(g, SOURCE, NODE_B, "0xslow1", ts=t)
    add_edge(g, NODE_B, VASP_1, "0xslow2", ts=t + 400 * 86_400)
    add_edge(g, SOURCE, NODE_C, "0xfast1", ts=t)
    add_edge(g, NODE_C, NODE_X, "0xfast2", ts=t + 60)
    add_edge(g, NODE_X, VASP_1, "0xfast3", ts=t + 120)

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(VASP_1, "VASP_1")), max_hops=3
    )

    e = result.candidates[0].outbound_evidence
    assert e.plausibility.grade is PlausibilityGrade.PLAUSIBLE
    assert e.hop_distance == 3
    assert e.tx_hashes == ["0xfast1", "0xfast2", "0xfast3"]
    assert e.alternative_path_count == 1
    assert any("additional traced route" in limit for limit in result.candidates[0].limitations)


def test_implausible_path_is_reported_as_a_connection_not_a_flow():
    """The shape observed on real data: asset changes at every hop, years
    apart. Still a MATCH (the addresses are connected and the path is real),
    but the report must be told not to call it a movement of funds."""
    g = nx.MultiDiGraph()
    add_edge(g, VASP_1, NODE_B, "0xh1", ts=1_563_621_950, asset="ETH")
    add_edge(g, NODE_B, NODE_C, "0xh2", ts=1_738_231_967, asset="MKR")
    add_edge(g, NODE_C, SOURCE, "0xh3", ts=1_785_059_027, asset="GIVE")

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(VASP_1, "VASP_1")), max_hops=3
    )

    assert result.status == AttributionStatus.MATCH_FOUND
    c = result.candidates[0]
    assert c.direction == ConnectionDirection.INDIRECT_INBOUND
    plausibility = c.inbound_evidence.plausibility
    assert plausibility.grade is PlausibilityGrade.IMPLAUSIBLE
    assert plausibility.supports_fund_flow_narrative is False
    assert any("not as a movement of value" in limit for limit in c.limitations)
    assert any("no path graded as supporting a fund-flow reading" in n for n in result.notes)


# --- Seed provenance is surfaced, and synthetic data announces itself ------


def test_synthetic_seed_dataset_is_flagged_on_the_result():
    g = nx.MultiDiGraph()
    add_edge(g, SOURCE, VASP_1, "0xtx1", ts=100)

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(VASP_1, "VASP_1")), max_hops=3
    )

    assert result.seed_contains_synthetic is True
    assert result.seed_address_count == 1
    assert any("SYNTHETIC_DEMO" in n for n in result.notes)


def test_seed_provenance_fields_are_passed_through_verbatim():
    """Provenance metadata is reported as the dataset recorded it. Absent
    metadata stays absent — it is never filled in with a plausible guess."""
    entry = VASPSeedEntry(
        address=VASP_1,
        vasp_name="VASP_1",
        entity_type="demo_exchange",
        chain="ethereum",
        source="synthetic_demo_fixture",
        source_type=SeedSourceType.SYNTHETIC_DEMO,
        confidence_note="Synthetic.",
        wallet_role="hot_wallet",
        source_evidence_type="community_label",
        verification_status="third_party_labeled",
    )
    g = nx.MultiDiGraph()
    add_edge(g, SOURCE, VASP_1, "0xtx1", ts=100)

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(entry), max_hops=3
    )

    c = result.candidates[0]
    assert c.wallet_role == "hot_wallet"
    assert c.source_evidence_type == "community_label"
    assert c.verification_status == "third_party_labeled"
    assert any("third_party_labeled" in limit for limit in c.limitations)


# --- One traversal per direction, regardless of dataset size ---------------


def test_many_seed_addresses_are_searched_in_a_single_traversal_per_direction():
    """Cost must not grow with the size of the VASP dataset: both directions
    search every seed address in one sweep each."""
    g = nx.MultiDiGraph()
    add_edge(g, SOURCE, VASP_1, "0xtx1", ts=100)
    add_edge(g, VASP_2, SOURCE, "0xtx2", ts=200)

    extra = [
        demo_seed_entry(f"0x{i:040x}", f"UNRELATED_{i}") for i in range(1, 60)
    ]
    seed_index = seed_index_with(
        demo_seed_entry(VASP_1, "VASP_1"), demo_seed_entry(VASP_2, "VASP_2"), *extra
    )

    result = generate_bidirectional_candidates(g, SOURCE, seed_index, max_hops=3)

    assert result.seed_address_count == 61
    assert result.outbound_accounting.targets_searched == 61
    assert result.inbound_accounting.targets_searched == 61
    # Only the two genuinely connected addresses become candidates.
    assert {c.matched_address for c in result.candidates} == {VASP_1, VASP_2}
    # The whole graph is 2 edges; the 59 absent addresses cost no traversal.
    assert result.outbound_accounting.edges_explored <= 2
    assert result.inbound_accounting.edges_explored <= 2


# --- Undirected connectivity is NEVER evidence -------------------------------


def test_undirected_only_relation_is_not_a_candidate_and_does_not_change_status():
    g = nx.MultiDiGraph()
    # SOURCE -> NODE_X and VASP_2 -> NODE_X: undirected-connected via NODE_X,
    # but NEITHER wallet reaches the other directionally.
    add_edge(g, SOURCE, NODE_X, "0xtx1", ts=100)
    add_edge(g, VASP_2, NODE_X, "0xtx2", ts=200)

    result = generate_bidirectional_candidates(
        g, SOURCE, seed_index_with(demo_seed_entry(VASP_2, "VASP_2")), max_hops=3
    )

    assert result.status == AttributionStatus.NONE
    assert result.candidates == []
    assert len(result.related_by_undirected_graph_only) == 1
    relation = result.related_by_undirected_graph_only[0]
    assert relation.vasp_address == VASP_2
    assert relation.undirected_distance == 2
    assert "not attribution evidence" in relation.note.lower()


def test_undirected_check_can_be_disabled():
    g = nx.MultiDiGraph()
    add_edge(g, SOURCE, NODE_X, "0xtx1", ts=100)
    add_edge(g, VASP_2, NODE_X, "0xtx2", ts=200)

    result = generate_bidirectional_candidates(
        g,
        SOURCE,
        seed_index_with(demo_seed_entry(VASP_2, "VASP_2")),
        max_hops=3,
        check_undirected_relations=False,
    )

    assert result.related_by_undirected_graph_only == []


# --- Behavioral patterns still only annotate, never create a candidate -----


def test_behavioral_pattern_annotates_inbound_only_candidate():
    g = nx.MultiDiGraph()
    add_edge(g, VASP_1, SOURCE, "0xtx1", ts=100)

    pattern = BehaviorPattern(
        pattern_type=PatternType.REPEATED_FORWARDING,
        wallet=SOURCE,
        evidence=["synthetic evidence line"],
        metrics={"count": 3},
        related_addresses=[VASP_1],
    )

    result = generate_bidirectional_candidates(
        g,
        SOURCE,
        seed_index_with(demo_seed_entry(VASP_1, "VASP_1")),
        behavior_patterns=[pattern],
        max_hops=3,
    )

    c = result.candidates[0]
    assert "REPEATED_FORWARDING" in c.supporting_behavioral_patterns


def test_behavior_alone_without_any_path_cannot_create_a_candidate():
    g = nx.MultiDiGraph()
    add_edge(g, SOURCE, NODE_X, "0xtx1", ts=100)  # no path to VASP_1 at all

    pattern = BehaviorPattern(
        pattern_type=PatternType.SPLIT_PATTERN,
        wallet=SOURCE,
        evidence=["synthetic evidence line"],
        metrics={"count": 3},
        related_addresses=[VASP_1],
    )

    result = generate_bidirectional_candidates(
        g,
        SOURCE,
        seed_index_with(demo_seed_entry(VASP_1, "VASP_1")),
        behavior_patterns=[pattern],
        max_hops=3,
    )

    assert result.candidates == []
    assert result.status == AttributionStatus.NONE


# --- Reuse guarantee: outbound half matches plain generate_candidates() ----


def test_outbound_half_matches_plain_generate_candidates_exactly():
    """Confirms Phase 1 does not reimplement or diverge from M4's outbound
    logic — same trace, same candidate, same evidence_tier concept."""
    g = nx.MultiDiGraph()
    add_edge(g, SOURCE, NODE_B, "0xtx1", ts=100)
    add_edge(g, NODE_B, VASP_1, "0xtx2", ts=200)

    seed_index = seed_index_with(demo_seed_entry(VASP_1, "VASP_1"))

    plain_trace = trace_fund_flow(g, SOURCE, max_hops=3)
    plain_result = generate_candidates(plain_trace, seed_index)

    bidi_result = generate_bidirectional_candidates(g, SOURCE, seed_index, max_hops=3)

    assert plain_result.status == AttributionStatus.MATCH_FOUND
    assert bidi_result.status == AttributionStatus.MATCH_FOUND
    assert plain_result.candidates[0].hop_distance == bidi_result.candidates[0].outbound_evidence.hop_distance
    assert plain_result.candidates[0].path_addresses == bidi_result.candidates[0].outbound_evidence.path_addresses
    assert plain_result.candidates[0].tx_hashes == bidi_result.candidates[0].outbound_evidence.tx_hashes
