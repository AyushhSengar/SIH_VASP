from app.graph.builder import build_graph, summarize_graph
from app.models import AssetType, NormalizedTransfer, TransferStatus

WALLET_A = "0xaaaa111111111111111111111111111111111a"
WALLET_B = "0xbbbb222222222222222222222222222222222b"
WALLET_C = "0xcccc333333333333333333333333333333333c"


def make_transfer(
    tx_hash, from_addr, to_addr, amount, ts, asset_type=AssetType.NATIVE
):
    return NormalizedTransfer(
        tx_hash=tx_hash,
        chain="ethereum",
        block_number=1,
        timestamp=ts,
        from_address=from_addr,
        to_address=to_addr,
        asset_type=asset_type,
        asset_symbol="ETH" if asset_type == AssetType.NATIVE else "USDC",
        amount_raw=str(int(amount * 1e18)),
        amount=amount,
        status=TransferStatus.SUCCESS,
        source_provider="etherscan",
        fetched_at=0,
    )


def test_build_graph_basic_nodes_and_edges():
    transfers = [
        make_transfer("0x1", WALLET_A, WALLET_B, 1.0, 100),
        make_transfer("0x2", WALLET_B, WALLET_C, 0.5, 200),
    ]
    graph, stats = build_graph(transfers)

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert graph.has_edge(WALLET_A, WALLET_B)
    assert graph.has_edge(WALLET_B, WALLET_C)
    assert stats.notes == []


def test_build_graph_preserves_multi_edges_between_same_pair_different_hashes():
    transfers = [
        make_transfer("0x1", WALLET_A, WALLET_B, 1.0, 100),
        make_transfer("0x2", WALLET_A, WALLET_B, 2.0, 200),
    ]
    graph, stats = build_graph(transfers)

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 2
    assert stats.edges_created == 2
    edge_data = graph.get_edge_data(WALLET_A, WALLET_B)
    amounts = {d["amount"] for d in edge_data.values()}
    assert amounts == {1.0, 2.0}


def test_build_graph_same_tx_hash_multiple_events_same_pair_not_overwritten():
    """The exact bug that was found: a single tx_hash producing multiple
    transfer events between the same (from, to) pair must NOT collapse
    into one edge via key overwrite."""
    transfers = [
        make_transfer("0xSAME", WALLET_A, WALLET_B, 1.0, 100, asset_type=AssetType.ERC20),
        make_transfer("0xSAME", WALLET_A, WALLET_B, 2.0, 100, asset_type=AssetType.ERC20),
        make_transfer("0xSAME", WALLET_A, WALLET_B, 3.0, 100, asset_type=AssetType.ERC20),
    ]
    graph, stats = build_graph(transfers)

    assert graph.number_of_edges() == 3  # all three survive
    assert stats.edges_created == 3
    edge_data = graph.get_edge_data(WALLET_A, WALLET_B)
    assert len(edge_data) == 3
    amounts = sorted(d["amount"] for d in edge_data.values())
    assert amounts == [1.0, 2.0, 3.0]
    # provenance preserved on every one
    assert all(d["tx_hash"] == "0xSAME" for d in edge_data.values())
    # each has a distinct, deterministic occurrence-based key
    keys = set(edge_data.keys())
    assert keys == {"0xSAME#0", "0xSAME#1", "0xSAME#2"}
    transfer_indices = sorted(d["transfer_index"] for d in edge_data.values())
    assert transfer_indices == [0, 1, 2]


def test_build_graph_identical_looking_duplicate_transfers_remain_separate():
    """Two normalized records that are identical in every visible field
    (same hash, same from/to, same amount) still represent two distinct
    on-chain events and must both appear as separate edges — never
    deduplicated away."""
    transfers = [
        make_transfer("0xDUP", WALLET_A, WALLET_B, 1.0, 100),
        make_transfer("0xDUP", WALLET_A, WALLET_B, 1.0, 100),
    ]
    graph, stats = build_graph(transfers)

    assert graph.number_of_edges() == 2
    assert stats.edges_created == 2
    edge_data = graph.get_edge_data(WALLET_A, WALLET_B)
    assert len(edge_data) == 2


def test_build_graph_deterministic_keys_across_runs():
    transfers = [
        make_transfer("0xSAME", WALLET_A, WALLET_B, 1.0, 100),
        make_transfer("0xSAME", WALLET_A, WALLET_B, 2.0, 100),
    ]
    graph1, _ = build_graph(transfers)
    graph2, _ = build_graph(transfers)
    assert set(graph1.get_edge_data(WALLET_A, WALLET_B).keys()) == set(
        graph2.get_edge_data(WALLET_A, WALLET_B).keys()
    )


def test_build_graph_skips_contract_creation_but_keeps_source_node():
    transfers = [
        make_transfer("0x1", WALLET_A, None, 1.0, 100),
    ]
    graph, stats = build_graph(transfers)

    assert WALLET_A in graph.nodes
    assert graph.number_of_edges() == 0
    assert stats.contract_creation_skipped == 1
    assert any("contract-creation" in n for n in stats.notes)


def test_build_graph_contract_creation_count_is_exact_not_capped():
    """Regression test for the original bug: the reported count must equal
    the real number of skipped events, not be capped at 0/1 by string
    matching over notes."""
    transfers = [make_transfer(f"0x{i}", WALLET_A, None, 1.0, 100) for i in range(354)]
    transfers += [make_transfer("0xreal", WALLET_A, WALLET_B, 1.0, 100)]
    graph, stats = build_graph(transfers)

    assert stats.contract_creation_skipped == 354
    assert stats.edges_created == 1
    summary = summarize_graph(graph, stats)
    assert summary.contract_creation_skipped == 354


def test_build_graph_edge_attributes_present():
    transfers = [make_transfer("0xhash1", WALLET_A, WALLET_B, 1.5, 500)]
    graph, _ = build_graph(transfers)

    edge_data = graph.get_edge_data(WALLET_A, WALLET_B)
    data = next(iter(edge_data.values()))
    assert data["amount"] == 1.5
    assert data["tx_hash"] == "0xhash1"
    assert data["timestamp"] == 500
    assert data["transfer_type"] == "TRANSFER"
    assert data["chain"] == "ethereum"


def test_build_graph_token_transfer_type():
    transfers = [
        make_transfer("0x1", WALLET_A, WALLET_B, 10.0, 100, asset_type=AssetType.ERC20)
    ]
    graph, _ = build_graph(transfers)
    edge_data = graph.get_edge_data(WALLET_A, WALLET_B)
    data = next(iter(edge_data.values()))
    assert data["transfer_type"] == "TOKEN_TRANSFER"


def test_summarize_graph_basic_counts():
    transfers = [
        make_transfer("0x1", WALLET_A, WALLET_B, 1.0, 100),
        make_transfer("0x2", WALLET_B, WALLET_C, 0.5, 200, asset_type=AssetType.ERC20),
    ]
    graph, stats = build_graph(transfers)
    summary = summarize_graph(graph, stats)

    assert summary.node_count == 3
    assert summary.edge_count == 2
    assert summary.native_edge_count == 1
    assert summary.token_edge_count == 1
    assert summary.earliest_timestamp == 100
    assert summary.latest_timestamp == 200


def test_summarize_graph_empty_graph_does_not_crash():
    graph, stats = build_graph([])
    summary = summarize_graph(graph, stats)
    assert summary.node_count == 0
    assert summary.edge_count == 0
    assert summary.input_transfer_count == 0
    assert summary.reconciled is True


def test_summarize_graph_detects_self_loop():
    transfers = [make_transfer("0x1", WALLET_A, WALLET_A, 1.0, 100)]
    graph, stats = build_graph(transfers)
    summary = summarize_graph(graph, stats)
    assert summary.self_loop_edges == 1
    assert any("self-loop" in n.lower() for n in summary.notes)


# --- Reconciliation: input_transfers = edges + all skipped categories ---


def test_reconciliation_holds_for_mixed_batch():
    transfers = (
        [make_transfer(f"0xcc{i}", WALLET_A, None, 1.0, 100) for i in range(5)]  # contract creation
        + [make_transfer("0xnormal1", WALLET_A, WALLET_B, 1.0, 100)]
        + [make_transfer("0xsame", WALLET_A, WALLET_C, 1.0, 100)] * 3  # same-hash collisions
    )
    graph, stats = build_graph(transfers)
    summary = summarize_graph(graph, stats)

    assert summary.input_transfer_count == 9
    assert summary.edges_created == 4  # 1 normal + 3 same-hash
    assert summary.contract_creation_skipped == 5
    assert summary.accounted_for == 9
    assert summary.reconciled is True
    assert summary.edge_count == summary.edges_created  # structural cross-check matches


def test_reconciliation_matches_original_bug_report_scale():
    """4000 input transfers, 354 contract-creation, rest are normal or
    same-hash duplicates -> must reconcile to exactly 4000 either way."""
    transfers = [make_transfer(f"0xcc{i}", WALLET_A, None, 1.0, 100) for i in range(354)]
    transfers += [make_transfer(f"0xnorm{i}", WALLET_A, WALLET_B, 1.0, 100) for i in range(3587)]
    transfers += [make_transfer("0xdup", WALLET_A, WALLET_C, 1.0, 100) for _ in range(59)]

    graph, stats = build_graph(transfers)
    summary = summarize_graph(graph, stats)

    assert summary.input_transfer_count == 4000
    assert summary.contract_creation_skipped == 354
    assert summary.edges_created == 3587 + 59  # nothing lost this time
    assert summary.accounted_for == 4000
    assert summary.reconciled is True
