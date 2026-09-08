"""
Tests for the targeted, destination-aware fund-flow search
(app/tracing/targeted.py).

These lock in the properties that make the search trustworthy rather than
merely fast:
  - a "not found" is only reported as COMPLETE when reachability was
    actually exhaustive;
  - a budget stop is surfaced per target, never hidden;
  - graph-reachable-but-chronologically-impossible is its own outcome and
    is not silently collapsed into either "found" or "not found";
  - pruning genuinely happens (an irrelevant fan-out is not walked);
  - results are deterministic run to run.

All graphs are built in-process, so these run fully offline.
"""

from __future__ import annotations

import networkx as nx

from app.tracing.targeted import (
    SearchDirection,
    SearchStatus,
    trace_targeted,
)

W = "0x1111111111111111111111111111111111111111"  # investigated wallet
A = "0xaaaa000000000000000000000000000000000001"
B = "0xbbbb000000000000000000000000000000000002"
C = "0xcccc000000000000000000000000000000000003"
VASP = "0x9999999999999999999999999999999999999999"
VASP2 = "0x8888888888888888888888888888888888888888"


def add_edge(graph, u, v, tx_hash, occurrence=0, ts=1_600_000_000, amount=1.0, asset="ETH"):
    key = f"{tx_hash}#{occurrence}"
    graph.add_edge(
        u,
        v,
        key=key,
        amount=amount,
        asset=asset,
        asset_type="NATIVE",
        transfer_type="TRANSFER",
        transfer_source="NATIVE_TRANSACTION",
        timestamp=ts,
        block_number=1000 + occurrence,
        tx_hash=tx_hash,
        transfer_index=occurrence,
        chain="ethereum",
        status="SUCCESS",
    )
    return key


# --- Outbound: found at 1 and 2 hops ----------------------------------------


def test_direct_outbound_one_hop():
    g = nx.MultiDiGraph()
    add_edge(g, W, VASP, "0x1", ts=100_000_000_0)
    r = trace_targeted(g, W, [VASP], direction=SearchDirection.OUTBOUND, max_hops=3)
    assert r.targets_reached == [VASP]
    outcome = r.target_outcomes[VASP]
    assert outcome.graph_distance == 1
    assert outcome.reachable_within_max_hops is True
    assert r.status is SearchStatus.COMPLETE
    best = r.best_path_for(VASP)
    assert best.source == W and best.terminal_node == VASP
    assert best.hop_count == 1


def test_indirect_outbound_two_hops_path_is_ordered_and_carries_evidence():
    g = nx.MultiDiGraph()
    add_edge(g, W, A, "0x1", ts=1_600_000_000)
    add_edge(g, A, VASP, "0x2", ts=1_600_000_500)
    r = trace_targeted(g, W, [VASP], direction=SearchDirection.OUTBOUND, max_hops=3)
    best = r.best_path_for(VASP)
    assert best.hop_count == 2
    assert best.addresses == [W, A, VASP]
    assert [h.tx_hash for h in best.hops] == ["0x1", "0x2"]
    assert [h.hop_index for h in best.hops] == [0, 1]
    # Full evidence must survive the targeted search too, not just the
    # exploratory tracer.
    assert best.hops[0].block_number == 1000
    assert best.hops[0].chain == "ethereum"


# --- Inbound: reverse search, one traversal ---------------------------------


def test_inbound_path_is_reported_in_true_chronological_orientation():
    """VASP -> A -> wallet must read VASP-first even though the search walks
    backwards from the wallet."""
    g = nx.MultiDiGraph()
    add_edge(g, VASP, A, "0x1", ts=1_600_000_000)
    add_edge(g, A, W, "0x2", ts=1_600_000_500)
    r = trace_targeted(g, W, [VASP], direction=SearchDirection.INBOUND, max_hops=3)
    assert r.targets_reached == [VASP]
    best = r.best_path_for(VASP)
    assert best.source == VASP
    assert best.terminal_node == W
    assert best.addresses == [VASP, A, W]
    assert [h.tx_hash for h in best.hops] == ["0x1", "0x2"]
    assert [h.hop_index for h in best.hops] == [0, 1]
    assert best.start_timestamp == 1_600_000_000
    assert best.end_timestamp == 1_600_000_500


def test_outbound_absent_but_inbound_present_are_independent_answers():
    """The whole point of bidirectional search: VASP -> wallet is meaningful
    on its own, and must not be reported just because wallet -> VASP failed."""
    g = nx.MultiDiGraph()
    add_edge(g, VASP, W, "0x1", ts=1_600_000_000)
    out = trace_targeted(g, W, [VASP], direction=SearchDirection.OUTBOUND, max_hops=3)
    inb = trace_targeted(g, W, [VASP], direction=SearchDirection.INBOUND, max_hops=3)
    assert out.targets_reached == []
    assert out.status is SearchStatus.COMPLETE  # a complete negative
    assert inb.targets_reached == [VASP]


def test_inbound_finds_several_targets_in_one_traversal():
    g = nx.MultiDiGraph()
    add_edge(g, VASP, A, "0x1", ts=1_600_000_000)
    add_edge(g, VASP2, A, "0x2", ts=1_600_000_100)
    add_edge(g, A, W, "0x3", ts=1_600_000_500)
    r = trace_targeted(g, W, [VASP, VASP2], direction=SearchDirection.INBOUND, max_hops=3)
    assert r.targets_reached == sorted([VASP, VASP2])


# --- Complete negatives -----------------------------------------------------


def test_no_match_is_a_complete_negative_not_inconclusive():
    g = nx.MultiDiGraph()
    add_edge(g, W, A, "0x1")
    add_edge(g, A, B, "0x2")
    r = trace_targeted(g, W, [VASP], direction=SearchDirection.OUTBOUND, max_hops=4)
    assert r.targets_reached == []
    assert r.status is SearchStatus.COMPLETE
    assert r.target_outcomes[VASP].reachable_within_max_hops is False
    assert r.target_outcomes[VASP].search_incomplete is False
    assert any("COMPLETE negative" in n for n in r.notes)


def test_target_beyond_max_hops_is_unreachable_within_scope():
    g = nx.MultiDiGraph()
    add_edge(g, W, A, "0x1", ts=1)
    add_edge(g, A, B, "0x2", ts=2)
    add_edge(g, B, C, "0x3", ts=3)
    add_edge(g, C, VASP, "0x4", ts=4)  # 4 hops away
    near = trace_targeted(g, W, [VASP], max_hops=2)
    assert near.targets_reached == []
    assert near.target_outcomes[VASP].reachable_within_max_hops is False
    assert near.status is SearchStatus.COMPLETE
    far = trace_targeted(g, W, [VASP], max_hops=4)
    assert far.targets_reached == [VASP]
    assert far.target_outcomes[VASP].graph_distance == 4


def test_target_absent_from_graph_entirely():
    g = nx.MultiDiGraph()
    add_edge(g, W, A, "0x1")
    r = trace_targeted(g, W, [VASP], max_hops=3)
    assert r.target_outcomes[VASP].graph_distance is None
    assert r.target_outcomes[VASP].reachable_within_max_hops is False
    assert r.status is SearchStatus.COMPLETE


# --- Chronology is its own outcome ------------------------------------------


def test_reachable_but_chronologically_impossible_is_reported_distinctly():
    """W -> A at t=500, A -> VASP at t=100. The addresses are connected, but
    funds cannot leave A before they arrive. Neither a match nor a plain
    'not reachable'."""
    g = nx.MultiDiGraph()
    add_edge(g, W, A, "0x1", ts=500)
    add_edge(g, A, VASP, "0x2", ts=100)
    r = trace_targeted(g, W, [VASP], direction=SearchDirection.OUTBOUND, max_hops=3)
    outcome = r.target_outcomes[VASP]
    assert outcome.reachable_within_max_hops is True  # graph-reachable
    assert outcome.graph_distance == 2
    assert outcome.paths_found == 0  # but no valid fund-flow candidate
    assert outcome.chronologically_blocked is True
    assert outcome.search_incomplete is False
    assert r.status is SearchStatus.COMPLETE
    assert any("timestamps run backwards" in n for n in r.notes)


def test_inbound_chronology_is_enforced_in_reverse():
    """For VASP -> A -> W, the VASP->A transfer must not postdate A->W."""
    g = nx.MultiDiGraph()
    add_edge(g, VASP, A, "0x1", ts=900)  # too late
    add_edge(g, A, W, "0x2", ts=100)
    r = trace_targeted(g, W, [VASP], direction=SearchDirection.INBOUND, max_hops=3)
    assert r.target_outcomes[VASP].paths_found == 0
    assert r.target_outcomes[VASP].chronologically_blocked is True


# --- Budgets must never masquerade as a negative ----------------------------


def test_budget_stop_marks_target_incomplete_not_absent():
    g = nx.MultiDiGraph()
    # Wide fan of parallel routes to the target, then starve the budget.
    for i in range(40):
        mid = f"0xdddd0000000000000000000000000000000{i:05d}"
        add_edge(g, W, mid, f"0xa{i}", ts=100)
        add_edge(g, mid, VASP, f"0xb{i}", ts=200)
    r = trace_targeted(
        g, W, [VASP], max_hops=3, max_paths_per_target=1000, max_edges_explored=3
    )
    assert r.status is SearchStatus.INCOMPLETE
    assert r.edges_limit_hit is True
    assert r.paths_truncated is True
    assert any("budget reached" in n.lower() for n in r.notes)
    # Reachability is still exact even when enumeration was cut short.
    assert r.target_outcomes[VASP].reachable_within_max_hops is True


def test_budgeted_target_with_zero_paths_is_incomplete_not_chronologically_blocked():
    g = nx.MultiDiGraph()
    for i in range(30):
        mid = f"0xdddd0000000000000000000000000000000{i:05d}"
        add_edge(g, W, mid, f"0xa{i}", ts=100)
        add_edge(g, mid, VASP, f"0xb{i}", ts=200)
    r = trace_targeted(g, W, [VASP], max_hops=3, max_edges_explored=1)
    outcome = r.target_outcomes[VASP]
    if outcome.paths_found == 0:
        assert outcome.search_incomplete is True
        assert outcome.chronologically_blocked is False


def test_per_target_path_quota_is_respected():
    g = nx.MultiDiGraph()
    for i in range(20):
        mid = f"0xdddd0000000000000000000000000000000{i:05d}"
        add_edge(g, W, mid, f"0xa{i}", ts=100)
        add_edge(g, mid, VASP, f"0xb{i}", ts=200)
    r = trace_targeted(g, W, [VASP], max_hops=3, max_paths_per_target=5)
    assert r.target_outcomes[VASP].paths_found == 5
    assert r.target_outcomes[VASP].path_quota_reached is True
    # Early termination: it stopped rather than walking all 20 routes.
    assert r.status is SearchStatus.COMPLETE


# --- Pruning actually prunes -------------------------------------------------


def test_irrelevant_fanout_is_not_explored():
    """A large subgraph that cannot lead to the target must be excluded from
    the viable set and never walked. This is the property that replaces
    'raise the budget'."""
    g = nx.MultiDiGraph()
    add_edge(g, W, A, "0x1", ts=100)
    add_edge(g, A, VASP, "0x2", ts=200)
    # 500 dead-end wallets hanging off the wallet, none leading anywhere.
    for i in range(500):
        noise = f"0xeeee0000000000000000000000000000000{i:05d}"
        add_edge(g, W, noise, f"0xn{i}", ts=100)
        add_edge(g, noise, f"0xffff0000000000000000000000000000000{i:05d}", f"0xm{i}", ts=150)
    r = trace_targeted(g, W, [VASP], max_hops=4)
    assert r.targets_reached == [VASP]
    assert r.graph_edge_count > 1000
    # Only W, A and VASP can lie on a route to the target.
    assert r.viable_node_count == 3
    assert r.reachable_node_count > 500  # the noise IS reachable...
    assert r.edges_explored < 20  # ...but was never enumerated
    assert any("Destination-aware pruning" in n for n in r.notes)


def test_parallel_transactions_remain_separately_traceable():
    g = nx.MultiDiGraph()
    add_edge(g, W, VASP, "0xsame", occurrence=0, ts=100, amount=1.0)
    add_edge(g, W, VASP, "0xsame", occurrence=1, ts=100, amount=2.0)
    r = trace_targeted(g, W, [VASP], max_hops=1, max_paths_per_target=10)
    assert r.target_outcomes[VASP].paths_found == 2
    keys = sorted(p.hops[0].edge_key for p in r.paths)
    assert keys == ["0xsame#0", "0xsame#1"]


def test_self_loop_does_not_hang_and_is_not_traversed():
    g = nx.MultiDiGraph()
    add_edge(g, W, W, "0xself", ts=100)
    add_edge(g, W, VASP, "0x1", ts=200)
    r = trace_targeted(g, W, [VASP], max_hops=3)
    assert r.targets_reached == [VASP]


def test_cycle_does_not_hang():
    g = nx.MultiDiGraph()
    add_edge(g, W, A, "0x1", ts=100)
    add_edge(g, A, B, "0x2", ts=200)
    add_edge(g, B, A, "0x3", ts=300)  # cycle
    add_edge(g, B, VASP, "0x4", ts=400)
    r = trace_targeted(g, W, [VASP], max_hops=4)
    assert r.targets_reached == [VASP]


# --- Determinism -------------------------------------------------------------


def test_repeated_runs_produce_identical_evidence():
    g = nx.MultiDiGraph()
    for i in range(6):
        mid = f"0xdddd0000000000000000000000000000000{i:05d}"
        add_edge(g, W, mid, f"0xa{i}", ts=100 + i)
        add_edge(g, mid, VASP, f"0xb{i}", ts=300 + i)
    first = trace_targeted(g, W, [VASP], max_hops=3, max_paths_per_target=3)
    second = trace_targeted(g, W, [VASP], max_hops=3, max_paths_per_target=3)
    signature = lambda r: [  # noqa: E731
        [h.edge_key for h in p.hops] for p in r.paths
    ]
    assert signature(first) == signature(second)
    assert first.edges_explored == second.edges_explored


# --- Degenerate inputs -------------------------------------------------------


def test_empty_graph():
    r = trace_targeted(nx.MultiDiGraph(), W, [VASP], max_hops=3)
    assert r.paths == []
    assert any("empty" in n.lower() for n in r.notes)
    assert r.status is SearchStatus.COMPLETE


def test_wallet_missing_from_graph():
    g = nx.MultiDiGraph()
    add_edge(g, A, VASP, "0x1")
    r = trace_targeted(g, W, [VASP], max_hops=3)
    assert r.paths == []
    assert r.wallet_in_graph is False
    assert any("does not appear in the graph" in n for n in r.notes)


def test_no_targets_supplied():
    g = nx.MultiDiGraph()
    add_edge(g, W, A, "0x1")
    r = trace_targeted(g, W, [], max_hops=3)
    assert r.paths == []
    assert any("No target addresses" in n for n in r.notes)


def test_wallet_is_itself_a_target_is_flagged_not_traced():
    g = nx.MultiDiGraph()
    add_edge(g, W, A, "0x1")
    r = trace_targeted(g, W, [W], max_hops=3)
    assert r.wallet_is_target is True
    assert any("ITSELF one of the known" in n for n in r.notes)
    # Distance 0 is an identity, never a fund-flow path.
    assert r.paths == []


def test_address_matching_is_case_insensitive_and_exact():
    g = nx.MultiDiGraph()
    add_edge(g, W, VASP, "0x1")
    r = trace_targeted(g, W.upper(), [VASP.upper()], max_hops=2)
    assert r.targets_reached == [VASP]
    # A prefix of a real address must never match.
    r2 = trace_targeted(g, W, [VASP[:20]], max_hops=2)
    assert r2.targets_reached == []


# --- Time window -------------------------------------------------------------


def test_time_window_excludes_older_transfers():
    g = nx.MultiDiGraph()
    old = 1_600_000_000
    recent = old + 400 * 86_400
    add_edge(g, W, A, "0xold", ts=old)
    add_edge(g, A, VASP, "0xold2", ts=old + 100)
    add_edge(g, W, B, "0xrecent", ts=recent)
    # Without a window, the old route is found.
    assert trace_targeted(g, W, [VASP], max_hops=3).targets_reached == [VASP]
    # A 30-day window anchored on the most recent activity excludes it.
    windowed = trace_targeted(g, W, [VASP], max_hops=3, time_window_days=30)
    assert windowed.targets_reached == []
    assert windowed.edges_excluded_by_time_window == 2
    assert any("Time window applied" in n for n in windowed.notes)


def test_time_window_keeps_transfers_with_no_timestamp():
    """A transfer with no timestamp cannot be proven outside the window, so
    it must not be silently discarded as evidence."""
    g = nx.MultiDiGraph()
    recent = 1_600_000_000
    add_edge(g, W, A, "0xrecent", ts=recent)
    add_edge(g, A, VASP, "0xnots", ts=0)
    r = trace_targeted(g, W, [VASP], max_hops=3, time_window_days=30)
    assert r.targets_reached == [VASP]


# --- Data horizon: the graph's radius vs the traversal's hop limit -----------
#
# A single-wallet acquisition fetches only the target's own transactions, so
# every edge touches the wallet and the graph holds exactly ONE hop of chain
# history. Traversing that with MAX_HOPS=4 is a complete traversal of a
# dataset that cannot contain a 2-hop route. These lock in that the report
# never presents the second fact as the first.


def test_horizon_shallower_than_hop_limit_is_not_a_complete_negative():
    """The exact bug this guards: exhaustive BFS over a 1-hop star reported
    as a COMPLETE negative at 4 hops."""
    g = nx.MultiDiGraph()
    add_edge(g, W, A, "0xa")
    add_edge(g, B, W, "0xb")
    r = trace_targeted(g, W, [VASP], max_hops=4, observation_depth=1)

    assert r.targets_reached == []
    assert r.status == SearchStatus.INCOMPLETE
    assert r.limited_by_observation_depth is True
    assert r.observation_depth == 1
    joined = " ".join(r.notes)
    assert "DATA HORIZON" in joined
    # It must still credit what WAS established, and blame the data rather
    # than a budget that was never touched.
    assert "COMPLETE negative at 1 hop(s)" in joined
    assert r.edges_limit_hit is False


def test_horizon_at_or_beyond_hop_limit_still_gives_a_complete_negative():
    """A horizon that covers the requested depth must not weaken the result:
    over-reporting INCONCLUSIVE would make NONE unreachable and useless."""
    g = nx.MultiDiGraph()
    add_edge(g, W, A, "0xa")
    add_edge(g, A, B, "0xb", ts=1_600_000_100)
    r = trace_targeted(g, W, [VASP], max_hops=2, observation_depth=2)

    assert r.status == SearchStatus.COMPLETE
    assert r.limited_by_observation_depth is False
    joined = " ".join(r.notes)
    assert "COMPLETE negative" in joined
    assert "DATA HORIZON" not in joined


def test_no_horizon_asserted_preserves_the_original_behaviour():
    g = nx.MultiDiGraph()
    add_edge(g, W, A, "0xa")
    r = trace_targeted(g, W, [VASP], max_hops=4)

    assert r.observation_depth is None
    assert r.limited_by_observation_depth is False
    assert r.status == SearchStatus.COMPLETE


def test_horizon_does_not_suppress_a_route_found_within_it():
    """A 1-hop match is fully observed, so the horizon must not hide it --
    only annotate that deeper routes were not visible."""
    g = nx.MultiDiGraph()
    add_edge(g, W, VASP, "0xdirect")
    r = trace_targeted(g, W, [VASP], max_hops=4, observation_depth=1)

    assert r.targets_reached == [VASP]
    assert r.limited_by_observation_depth is True


def test_horizon_applies_to_the_inbound_direction_too():
    g = nx.MultiDiGraph()
    add_edge(g, A, W, "0xa")
    r = trace_targeted(
        g, W, [VASP], direction=SearchDirection.INBOUND, max_hops=4, observation_depth=1
    )
    assert r.status == SearchStatus.INCOMPLETE
    assert r.limited_by_observation_depth is True
