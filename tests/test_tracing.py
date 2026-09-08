"""
MACRO MILESTONE 3 / PHASE A tests — multi-hop fund-flow tracing.

All tests build synthetic NetworkX graphs directly (mirroring exactly what
app/graph/builder.py produces) so they run fully offline, with no
dependency on the live Etherscan fixture.
"""

from __future__ import annotations

import networkx as nx

from app.core.config import get_settings
from app.tracing.tracer import trace_fund_flow

WALLET_A = "0xaaaa111111111111111111111111111111111a"
WALLET_B = "0xbbbb222222222222222222222222222222222b"
WALLET_C = "0xcccc333333333333333333333333333333333c"
WALLET_D = "0xdddd444444444444444444444444444444444d"
WALLET_E = "0xeeee555555555555555555555555555555555e"


def add_edge(
    graph,
    u,
    v,
    tx_hash,
    occurrence=0,
    amount=1.0,
    ts=100,
    asset="ETH",
    asset_type="NATIVE",
    transfer_type="TRANSFER",
    chain="ethereum",
    block=None,
    token_contract=None,
    transfer_source="NATIVE_TRANSACTION",
):
    """Mirrors app/graph/builder.build_graph()'s edge schema.

    `block` / `token_contract` default to None so a test can deliberately
    build an edge that carries no block number (proving the tracer reports
    None rather than fabricating one) as well as one that does.
    """
    key = f"{tx_hash}#{occurrence}"
    graph.add_edge(
        u,
        v,
        key=key,
        amount=amount,
        asset=asset,
        asset_type=asset_type,
        timestamp=ts,
        block_number=block,
        tx_hash=tx_hash,
        transfer_index=occurrence,
        token_contract=token_contract,
        transfer_source=transfer_source,
        chain=chain,
        transfer_type=transfer_type,
        status="SUCCESS",
    )
    return key


SETTINGS = get_settings()


# --- 1/2/3. one/two/three-hop paths ---------------------------------------


def test_one_hop_path():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=100)
    result = trace_fund_flow(g, WALLET_A, max_hops=3)
    assert len(result.paths) == 1
    assert result.paths[0].hop_count == 1
    assert result.paths[0].terminal_node == WALLET_B


def test_two_hop_path():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=100)
    add_edge(g, WALLET_B, WALLET_C, "0x2", ts=200)
    result = trace_fund_flow(g, WALLET_A, max_hops=3)
    hop_counts = sorted(p.hop_count for p in result.paths)
    assert hop_counts == [1, 2]
    two_hop = [p for p in result.paths if p.hop_count == 2][0]
    assert two_hop.terminal_node == WALLET_C
    assert two_hop.addresses == [WALLET_A, WALLET_B, WALLET_C]


def test_three_hop_path():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=100)
    add_edge(g, WALLET_B, WALLET_C, "0x2", ts=200)
    add_edge(g, WALLET_C, WALLET_D, "0x3", ts=300)
    result = trace_fund_flow(g, WALLET_A, max_hops=5)
    three_hop = [p for p in result.paths if p.hop_count == 3]
    assert len(three_hop) == 1
    assert three_hop[0].terminal_node == WALLET_D
    assert three_hop[0].addresses == [WALLET_A, WALLET_B, WALLET_C, WALLET_D]


# --- 4/5. MAX_HOPS boundary -------------------------------------------------


def test_max_hops_boundary_included():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=100)
    add_edge(g, WALLET_B, WALLET_C, "0x2", ts=200)
    result = trace_fund_flow(g, WALLET_A, max_hops=2)
    assert any(p.hop_count == 2 for p in result.paths)


def test_path_never_exceeds_max_hops():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=100)
    add_edge(g, WALLET_B, WALLET_C, "0x2", ts=200)
    add_edge(g, WALLET_C, WALLET_D, "0x3", ts=300)
    add_edge(g, WALLET_D, WALLET_E, "0x4", ts=400)
    result = trace_fund_flow(g, WALLET_A, max_hops=2)
    assert all(p.hop_count <= 2 for p in result.paths)
    assert max(p.hop_count for p in result.paths) == 2


# --- 6/7. parallel edges / same tx_hash multiple events ---------------------


def test_multiple_parallel_edges_all_traceable():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0xtx1", occurrence=0, amount=1.0, ts=100)
    add_edge(g, WALLET_A, WALLET_B, "0xtx2", occurrence=0, amount=2.0, ts=110)
    add_edge(g, WALLET_A, WALLET_B, "0xtx3", occurrence=0, amount=3.0, ts=120)
    result = trace_fund_flow(g, WALLET_A, max_hops=1)
    assert len(result.paths) == 3
    tx_hashes = {p.hops[0].tx_hash for p in result.paths}
    assert tx_hashes == {"0xtx1", "0xtx2", "0xtx3"}


def test_same_tx_hash_multiple_transfer_events_all_traceable():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0xSAME", occurrence=0, amount=1.0, ts=100)
    add_edge(g, WALLET_A, WALLET_B, "0xSAME", occurrence=1, amount=2.0, ts=100)
    add_edge(g, WALLET_A, WALLET_B, "0xSAME", occurrence=2, amount=3.0, ts=100)
    result = trace_fund_flow(g, WALLET_A, max_hops=1)
    assert len(result.paths) == 3
    edge_keys = {p.hops[0].edge_key for p in result.paths}
    assert edge_keys == {"0xSAME#0", "0xSAME#1", "0xSAME#2"}
    amounts = sorted(p.hops[0].amount for p in result.paths)
    assert amounts == [1.0, 2.0, 3.0]


# --- 8. self-loop ------------------------------------------------------------


def test_self_loop_recorded_but_not_extended():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_A, "0xloop", ts=100)
    result = trace_fund_flow(g, WALLET_A, max_hops=5)
    assert len(result.paths) == 1
    assert result.paths[0].terminal_node == WALLET_A
    assert result.paths[0].hop_count == 1


# --- 9. cycle ------------------------------------------------------------


def test_cycle_does_not_infinite_loop_and_is_bounded():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=100)
    add_edge(g, WALLET_B, WALLET_C, "0x2", ts=200)
    add_edge(g, WALLET_C, WALLET_A, "0x3", ts=300)  # cycle back to source
    result = trace_fund_flow(g, WALLET_A, max_hops=10)
    # Must terminate (test itself would hang otherwise) and never revisit A.
    assert all(WALLET_A not in p.addresses[1:] for p in result.paths)
    assert max(p.hop_count for p in result.paths) == 2  # A->B->C, cycle edge dropped


# --- 10/11/12. timestamps: missing, ordering, duration --------------------


def test_missing_timestamp_does_not_block_traversal():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=None)
    result = trace_fund_flow(g, WALLET_A, max_hops=1)
    assert len(result.paths) == 1
    assert result.paths[0].hops[0].timestamp is None
    assert result.paths[0].path_duration_seconds is None


def test_chronological_ordering_enforced_when_timestamps_present():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=500)
    add_edge(g, WALLET_B, WALLET_C, "0x2", ts=100)  # earlier than hop 1 -> invalid
    result = trace_fund_flow(g, WALLET_A, max_hops=3)
    # The 2-hop path through the out-of-order edge must not appear.
    assert all(p.hop_count == 1 for p in result.paths)


def test_path_duration_calculation():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=100)
    add_edge(g, WALLET_B, WALLET_C, "0x2", ts=142)
    result = trace_fund_flow(g, WALLET_A, max_hops=2)
    two_hop = [p for p in result.paths if p.hop_count == 2][0]
    assert two_hop.path_duration_seconds == 42


# --- 13/14/15/16. field preservation ---------------------------------------


def test_amount_asset_txhash_edgekey_preserved():
    g = nx.MultiDiGraph()
    add_edge(
        g,
        WALLET_A,
        WALLET_B,
        "0xabc",
        occurrence=0,
        amount=7.5,
        ts=100,
        asset="USDC",
        asset_type="ERC20",
        transfer_type="TOKEN_TRANSFER",
        block=18_000_123,
        token_contract="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        transfer_source="TOKEN_TRANSFER",
    )
    result = trace_fund_flow(g, WALLET_A, max_hops=1)
    hop = result.paths[0].hops[0]
    assert hop.amount == 7.5
    assert hop.asset == "USDC"
    assert hop.asset_type == "ERC20"
    assert hop.transfer_type == "TOKEN_TRANSFER"
    assert hop.tx_hash == "0xabc"
    assert hop.edge_key == "0xabc#0"
    # Full blockchain evidence must survive the trace, not just amount/asset:
    # a report that can't cite the block or the token contract isn't evidence.
    assert hop.block_number == 18_000_123
    assert hop.token_contract == "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    assert hop.transfer_source == "TOKEN_TRANSFER"
    assert hop.chain == "ethereum"
    assert hop.status == "SUCCESS"


def test_block_number_is_none_when_edge_genuinely_lacks_one():
    """Never fabricated: an edge with no block number yields None, not 0."""
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0xabc", ts=100)  # block defaults to None
    result = trace_fund_flow(g, WALLET_A, max_hops=1)
    assert result.paths[0].hops[0].block_number is None


# --- 17/18. empty graph / missing source ------------------------------------


def test_empty_graph_returns_no_paths_without_crashing():
    g = nx.MultiDiGraph()
    result = trace_fund_flow(g, WALLET_A, max_hops=3)
    assert result.paths == []
    assert any("empty" in n.lower() for n in result.notes)


def test_source_wallet_missing_from_graph():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_B, WALLET_C, "0x1", ts=100)
    result = trace_fund_flow(g, WALLET_A, max_hops=3)
    assert result.paths == []
    assert any("not found" in n.lower() for n in result.notes)


# --- 19. MAX_PATHS / result limiting ---------------------------------------


def test_max_paths_limits_result_count_and_flags_truncation():
    g = nx.MultiDiGraph()
    # fan-out of 10 destinations from A
    for i in range(10):
        add_edge(g, WALLET_A, f"0xdest{i:03d}", f"0xtx{i}", ts=100 + i)
    result = trace_fund_flow(g, WALLET_A, max_hops=1, max_paths=3)
    assert len(result.paths) == 3
    assert result.paths_truncated is True
    assert any("result limit" in n.lower() for n in result.notes)


def test_max_edges_explored_limits_work_and_flags():
    g = nx.MultiDiGraph()
    for i in range(50):
        add_edge(g, WALLET_A, f"0xdest{i:03d}", f"0xtx{i}", ts=100 + i)
    result = trace_fund_flow(g, WALLET_A, max_hops=1, max_paths=1000, max_edges_explored=5)
    assert result.edges_explored <= 5
    assert result.edges_limit_hit is True


# --- 20. deterministic output ------------------------------------------------


def test_deterministic_output_across_runs():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=100)
    add_edge(g, WALLET_A, WALLET_C, "0x2", ts=150)
    add_edge(g, WALLET_B, WALLET_D, "0x3", ts=200)

    result1 = trace_fund_flow(g, WALLET_A, max_hops=3)
    result2 = trace_fund_flow(g, WALLET_A, max_hops=3)

    keys1 = sorted((p.terminal_node, p.hop_count, tuple(h.edge_key for h in p.hops)) for p in result1.paths)
    keys2 = sorted((p.terminal_node, p.hop_count, tuple(h.edge_key for h in p.hops)) for p in result2.paths)
    assert keys1 == keys2


# --- extra: config default sanity ------------------------------------------


def test_default_max_hops_used_when_not_overridden():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET_A, WALLET_B, "0x1", ts=100)
    result = trace_fund_flow(g, WALLET_A)
    assert result.max_hops == SETTINGS.fund_trace_max_hops
