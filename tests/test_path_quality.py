"""
Tests for fund-flow candidate plausibility grading
(app/tracing/quality.py).

The grader exists so that a technically-valid path is not reported as a
fund flow when the observable evidence argues against continuity. These
tests pin the grades, and — more importantly — pin the requirement that
every downgrade carries the measurement that caused it, so a report can
never show an unexplained verdict.
"""

from __future__ import annotations

import networkx as nx

from app.tracing.models import FundFlowHop, FundFlowPath
from app.tracing.quality import (
    PlausibilityConcern,
    PlausibilityGrade,
    assess_path,
    best_graded_path,
)

W = "0x1111111111111111111111111111111111111111"
A = "0xaaaa000000000000000000000000000000000001"
B = "0xbbbb000000000000000000000000000000000002"
VASP = "0x9999999999999999999999999999999999999999"

DAY = 86_400


def hop(index, frm, to, ts, amount=1.0, asset="ETH", tx=None, contract=False):
    return FundFlowHop(
        hop_index=index,
        from_address=frm,
        to_address=to,
        tx_hash=tx or f"0x{index}",
        edge_key=f"{tx or f'0x{index}'}#0",
        timestamp=ts,
        amount=amount,
        asset=asset,
        asset_type="NATIVE",
        transfer_type="TRANSFER",
        chain="ethereum",
        block_number=1000 + index,
        is_contract_interaction=contract,
    )


def path(*hops):
    return FundFlowPath(
        source=hops[0].from_address, terminal_node=hops[-1].to_address, hops=list(hops)
    )


# --- Direct transfer ---------------------------------------------------------


def test_single_hop_is_direct_transfer_with_no_continuity_assumption():
    p = path(hop(0, W, VASP, 1_600_000_000, amount=5.0))
    q = assess_path(p)
    assert q.grade is PlausibilityGrade.DIRECT_TRANSFER
    assert q.concerns == []
    assert q.supports_fund_flow_narrative is True
    assert "No intermediary" in q.interpretation


def test_single_hop_without_timestamp_is_still_a_direct_transfer():
    """A missing timestamp does not weaken the fact that the transfer
    happened; there is no ordering to verify on a one-hop path."""
    p = path(hop(0, W, VASP, None))
    q = assess_path(p)
    assert q.grade is PlausibilityGrade.DIRECT_TRANSFER
    assert PlausibilityConcern.UNVERIFIABLE_CHRONOLOGY not in [
        c.concern for c in q.concerns
    ]


# --- Plausible multi-hop -----------------------------------------------------


def test_multi_hop_same_asset_tight_timing_is_plausible():
    t = 1_600_000_000
    p = path(
        hop(0, W, A, t, amount=10.0),
        hop(1, A, VASP, t + 600, amount=9.9),
    )
    q = assess_path(p)
    assert q.grade is PlausibilityGrade.PLAUSIBLE
    assert q.concerns == []
    assert q.single_asset_throughout is True
    assert q.max_hop_gap_seconds == 600
    assert q.intermediaries == [A]
    # Even the best multi-hop grade must not claim proof.
    assert "does NOT prove" in q.interpretation


# --- Single concern -> WEAK --------------------------------------------------


def test_asset_change_alone_downgrades_to_weak_with_the_measurement():
    t = 1_600_000_000
    p = path(
        hop(0, W, A, t, amount=10.0, asset="ETH"),
        hop(1, A, VASP, t + 600, amount=500.0, asset="USDC"),
    )
    q = assess_path(p)
    assert q.grade is PlausibilityGrade.WEAK
    assert q.single_asset_throughout is False
    assert q.assets_in_order == ["ETH", "USDC"]
    concern = next(c for c in q.concerns if c.concern is PlausibilityConcern.ASSET_CHANGED)
    assert concern.hop_index == 1
    assert "ETH" in concern.observed and "USDC" in concern.observed
    assert q.supports_fund_flow_narrative is False


def test_long_gap_alone_downgrades_to_weak_and_states_both_numbers():
    t = 1_600_000_000
    p = path(
        hop(0, W, A, t, amount=10.0),
        hop(1, A, VASP, t + 200 * DAY, amount=10.0),
    )
    q = assess_path(p)
    assert q.grade is PlausibilityGrade.WEAK
    concern = next(c for c in q.concerns if c.concern is PlausibilityConcern.LONG_TIME_GAP)
    assert "200 days" in concern.observed
    assert "30 days" in concern.threshold  # default threshold, stated explicitly
    assert q.max_hop_gap_seconds == 200 * DAY


def test_amount_increase_is_flagged_only_within_the_same_asset():
    t = 1_600_000_000
    p = path(
        hop(0, W, A, t, amount=1.0, asset="ETH"),
        hop(1, A, VASP, t + 60, amount=50.0, asset="ETH"),
    )
    q = assess_path(p)
    concern = next(
        c for c in q.concerns if c.concern is PlausibilityConcern.AMOUNT_INCREASED
    )
    assert "50.0" in concern.observed and "1.0" in concern.observed
    assert q.grade is PlausibilityGrade.WEAK


def test_amount_comparison_is_skipped_across_different_assets():
    """10 ETH in, 20000 USDC out is not an 'increase' — the units differ, so
    claiming one would be nonsense."""
    t = 1_600_000_000
    p = path(
        hop(0, W, A, t, amount=10.0, asset="ETH"),
        hop(1, A, VASP, t + 60, amount=20_000.0, asset="USDC"),
    )
    q = assess_path(p)
    assert PlausibilityConcern.AMOUNT_INCREASED not in [c.concern for c in q.concerns]
    assert q.grade is PlausibilityGrade.WEAK  # ASSET_CHANGED only


def test_missing_timestamp_on_multi_hop_is_recorded_not_assumed():
    p = path(
        hop(0, W, A, 1_600_000_000),
        hop(1, A, VASP, None),
    )
    q = assess_path(p)
    concern = next(
        c for c in q.concerns if c.concern is PlausibilityConcern.UNVERIFIABLE_CHRONOLOGY
    )
    assert "no usable timestamp" in concern.observed
    assert q.hop_gaps_seconds == [None]
    assert q.max_hop_gap_seconds is None


# --- Multiple concerns -> IMPLAUSIBLE ---------------------------------------


def test_two_or_more_concerns_make_the_path_implausible():
    t = 1_600_000_000
    p = path(
        hop(0, W, A, t, amount=1.0, asset="ETH"),
        hop(1, A, VASP, t + 400 * DAY, amount=99.0, asset="MKR"),
    )
    q = assess_path(p)
    assert q.grade is PlausibilityGrade.IMPLAUSIBLE
    assert PlausibilityConcern.ASSET_CHANGED in [c.concern for c in q.concerns]
    assert PlausibilityConcern.LONG_TIME_GAP in [c.concern for c in q.concerns]
    assert q.supports_fund_flow_narrative is False
    assert "Report the connection, not a flow" in q.interpretation


# --- Hub intermediaries ------------------------------------------------------


def _graph_with_hub(hub: str, degree: int) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for i in range(degree):
        g.add_edge(f"0xsender{i:034d}", hub, key=f"0xin{i}")
    return g


def test_high_throughput_intermediary_is_detected_and_quantified():
    t = 1_600_000_000
    p = path(
        hop(0, W, A, t, amount=1.0),
        hop(1, A, VASP, t + 60, amount=1.0),
    )
    g = _graph_with_hub(A, 400)
    q = assess_path(p, graph=g)
    hub_concern = next(
        c
        for c in q.concerns
        if c.concern is PlausibilityConcern.HIGH_THROUGHPUT_INTERMEDIARY
    )
    assert hub_concern.address == A
    assert "400" in hub_concern.observed
    assert q.hub_intermediaries[0].total_degree == 400
    assert q.hub_intermediaries[0].in_degree == 400


def test_ordinary_intermediary_is_not_flagged_as_a_hub():
    t = 1_600_000_000
    p = path(hop(0, W, A, t), hop(1, A, VASP, t + 60))
    g = _graph_with_hub(A, 5)
    q = assess_path(p, graph=g)
    assert q.hub_intermediaries == []
    assert q.grade is PlausibilityGrade.PLAUSIBLE


def test_hub_detection_is_skipped_rather_than_assumed_when_no_graph_given():
    t = 1_600_000_000
    p = path(hop(0, W, A, t), hop(1, A, VASP, t + 60))
    q = assess_path(p, graph=None)
    assert q.hub_intermediaries == []
    assert PlausibilityConcern.HIGH_THROUGHPUT_INTERMEDIARY not in [
        c.concern for c in q.concerns
    ]


def test_endpoints_are_never_treated_as_intermediaries():
    """A VASP endpoint is expected to be high-degree; that says nothing bad
    about the path, so only genuine middle addresses are examined."""
    t = 1_600_000_000
    p = path(hop(0, W, VASP, t))
    g = _graph_with_hub(VASP, 5000)
    q = assess_path(p, graph=g)
    assert q.intermediaries == []
    assert q.hub_intermediaries == []
    assert q.grade is PlausibilityGrade.DIRECT_TRANSFER


# --- Choosing between candidate paths ---------------------------------------


def test_best_graded_path_prefers_plausibility_over_brevity():
    t = 1_600_000_000
    weak_short = path(
        hop(0, W, A, t, amount=1.0, asset="ETH", tx="0xweak1"),
        hop(1, A, VASP, t + 500 * DAY, amount=1.0, asset="ETH", tx="0xweak2"),
    )
    plausible_long = path(
        hop(0, W, A, t, amount=5.0, tx="0xok1"),
        hop(1, A, B, t + 60, amount=4.9, tx="0xok2"),
        hop(2, B, VASP, t + 120, amount=4.8, tx="0xok3"),
    )
    chosen, q = best_graded_path([weak_short, plausible_long])
    assert q.grade is PlausibilityGrade.PLAUSIBLE
    assert chosen is plausible_long


def test_best_graded_path_prefers_a_direct_transfer_above_all():
    t = 1_600_000_000
    direct = path(hop(0, W, VASP, t, tx="0xdirect"))
    longer = path(hop(0, W, A, t, tx="0xa"), hop(1, A, VASP, t + 60, tx="0xb"))
    chosen, q = best_graded_path([longer, direct])
    assert q.grade is PlausibilityGrade.DIRECT_TRANSFER
    assert chosen is direct


def test_best_graded_path_is_deterministic():
    t = 1_600_000_000
    paths = [
        path(hop(0, W, A, t, tx=f"0x{i}"), hop(1, A, VASP, t + 60, tx=f"0xy{i}"))
        for i in range(5)
    ]
    first = best_graded_path(paths)
    second = best_graded_path(list(reversed(paths)))
    assert [h.edge_key for h in first[0].hops] == [h.edge_key for h in second[0].hops]


def test_best_graded_path_on_empty_input():
    assert best_graded_path([]) is None


# --- The real demonstration case, reduced to a unit test --------------------


def test_real_world_shape_kraken_via_dex_hub_is_implausible():
    """Mirrors the actual observed cached-real-data path: Kraken -> wallet ->
    high-degree DEX contract -> investigated wallet, three different assets,
    years apart. Technically traceable, and correctly NOT a fund flow."""
    kraken = "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0"
    mid = "0x04ed920e911b3b5d9b79788f83790d55f51dcfca"
    dex = "0x000000000004444c5dc75cb358380d2e3de08a90"
    p = path(
        hop(0, kraken, mid, 1_563_621_950, amount=0.16957, asset="ETH", tx="0x3b03"),
        hop(1, mid, dex, 1_738_231_967, amount=3.0648, asset="MKR", tx="0x4eed"),
        hop(2, dex, W, 1_785_059_027, amount=30_707.99, asset="GIVE", tx="0x01cf"),
    )
    g = _graph_with_hub(dex, 5099)
    q = assess_path(p, graph=g)
    assert q.grade is PlausibilityGrade.IMPLAUSIBLE
    kinds = {c.concern for c in q.concerns}
    assert PlausibilityConcern.ASSET_CHANGED in kinds
    assert PlausibilityConcern.LONG_TIME_GAP in kinds
    assert PlausibilityConcern.HIGH_THROUGHPUT_INTERMEDIARY in kinds
    assert q.supports_fund_flow_narrative is False
