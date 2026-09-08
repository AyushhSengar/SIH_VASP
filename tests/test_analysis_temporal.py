"""
Tests for app/analysis/temporal.py — the descriptive temporal/amount layer.

Synthetic graphs only, mirroring app/graph/builder.py's edge schema. Fully
offline. These tests are about MEASUREMENT correctness: the module has no
thresholds and no verdicts, so what matters is that every number it prints
is the number actually present in the graph, and that missing data stays
missing rather than being filled in.
"""

from __future__ import annotations

import networkx as nx

from app.analysis.temporal import (
    SECONDS_PER_DAY,
    analyze_temporal_and_amounts,
)

WALLET = "0xaaaa111111111111111111111111111111111a"
OTHER = "0xbbbb222222222222222222222222222222222b"
THIRD = "0xcccc333333333333333333333333333333333c"


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
    token_contract=None,
    token_metadata_missing=False,
):
    key = f"{tx_hash}#{occurrence}"
    graph.add_edge(
        u,
        v,
        key=key,
        amount=amount,
        asset=asset,
        asset_type=asset_type,
        token_contract=token_contract,
        token_metadata_missing=token_metadata_missing,
        timestamp=ts,
        tx_hash=tx_hash,
        transfer_index=occurrence,
        chain="ethereum",
        transfer_type="TRANSFER" if asset_type == "NATIVE" else "TOKEN_TRANSFER",
        status="SUCCESS",
    )
    return key


def test_empty_graph_returns_zeroed_analysis_not_an_error():
    """A wallet with no edges must still render a section, and must say why
    there is nothing in it."""
    result = analyze_temporal_and_amounts(nx.MultiDiGraph(), WALLET)

    assert result.transfer_count == 0
    assert result.first_seen is None
    assert result.last_seen is None
    assert result.per_asset == []
    assert result.pass_through is None
    assert any("no transfer edges" in line for line in result.limitations)


def test_wallet_absent_from_populated_graph_is_also_empty():
    g = nx.MultiDiGraph()
    add_edge(g, OTHER, THIRD, "0xaa")

    result = analyze_temporal_and_amounts(g, WALLET)

    assert result.transfer_count == 0
    assert result.limitations


def test_lifecycle_measures_real_span_and_active_days():
    g = nx.MultiDiGraph()
    day = SECONDS_PER_DAY
    add_edge(g, WALLET, OTHER, "0xa1", ts=0)
    add_edge(g, WALLET, OTHER, "0xa2", ts=10)  # same active day
    add_edge(g, OTHER, WALLET, "0xa3", ts=3 * day)

    result = analyze_temporal_and_amounts(g, WALLET)

    assert result.transfer_count == 3
    assert result.first_seen == 0
    assert result.last_seen == 3 * day
    assert result.lifespan_seconds == 3 * day
    assert result.lifespan_days == 3.0
    # Two distinct UTC days touched, not four calendar days of existence.
    assert result.active_day_count == 2
    assert result.transfers_per_active_day == 1.5
    assert result.longest_idle_seconds == 3 * day - 10


def test_missing_timestamps_are_counted_not_guessed():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET, OTHER, "0xa1", ts=1000)
    add_edge(g, WALLET, OTHER, "0xa2", ts=None)

    result = analyze_temporal_and_amounts(g, WALLET)

    assert result.transfer_count == 2  # counted in totals
    assert result.timestamped_transfer_count == 1
    assert result.missing_timestamp_count == 1
    # The un-timestamped transfer must not be dated to the epoch.
    assert result.first_seen == 1000
    assert result.last_seen == 1000
    assert any("no timestamp" in line for line in result.limitations)


def test_direction_counts_and_unique_counterparties():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET, OTHER, "0xa1", ts=1)
    add_edge(g, WALLET, THIRD, "0xa2", ts=2)
    add_edge(g, OTHER, WALLET, "0xa3", ts=3)
    add_edge(g, WALLET, WALLET, "0xa4", ts=4)  # self-transfer

    result = analyze_temporal_and_amounts(g, WALLET)

    assert result.outbound_transfer_count == 2
    assert result.inbound_transfer_count == 1
    assert result.self_transfer_count == 1
    assert result.unique_outbound_counterparties == 2
    assert result.unique_inbound_counterparties == 1
    # The self-loop is counted exactly once overall.
    assert result.transfer_count == 4


def test_amounts_are_grouped_per_asset_and_never_summed_across_them():
    g = nx.MultiDiGraph()
    add_edge(g, OTHER, WALLET, "0xa1", amount=10.0, ts=1, asset="ETH")
    add_edge(g, WALLET, OTHER, "0xa2", amount=4.0, ts=2, asset="ETH")
    add_edge(
        g,
        OTHER,
        WALLET,
        "0xa3",
        amount=500.0,
        ts=3,
        asset="USDC",
        asset_type="ERC20",
        token_contract="0xdead",
    )

    result = analyze_temporal_and_amounts(g, WALLET)

    by_asset = {s.asset: s for s in result.per_asset}
    assert set(by_asset) == {"ETH", "USDC"}

    eth = by_asset["ETH"]
    assert eth.transfer_count == 2
    assert eth.total_inbound == 10.0
    assert eth.total_outbound == 4.0
    assert eth.net_flow == 6.0
    assert eth.min_amount == 4.0
    assert eth.max_amount == 10.0
    assert eth.mean_amount == 7.0

    usdc = by_asset["USDC"]
    assert usdc.token_contract == "0xdead"
    assert usdc.net_flow == 500.0

    assert result.native_stats is eth or result.native_stats == eth
    assert any("never summed across" in line for line in result.limitations)


def test_same_symbol_different_contracts_are_kept_apart():
    """Two unrelated contracts both called USDC must not be merged: doing so
    would invent a total that describes no real asset."""
    g = nx.MultiDiGraph()
    add_edge(
        g, OTHER, WALLET, "0xa1", amount=1.0, ts=1, asset="USDC",
        asset_type="ERC20", token_contract="0x1111",
    )
    add_edge(
        g, OTHER, WALLET, "0xa2", amount=2.0, ts=2, asset="USDC",
        asset_type="ERC20", token_contract="0x2222",
    )

    result = analyze_temporal_and_amounts(g, WALLET)

    assert len(result.per_asset) == 2
    assert {s.token_contract for s in result.per_asset} == {"0x1111", "0x2222"}
    assert all(s.transfer_count == 1 for s in result.per_asset)


def test_incomplete_token_metadata_is_disclosed():
    g = nx.MultiDiGraph()
    add_edge(
        g, OTHER, WALLET, "0xa1", amount=1.0, ts=1, asset="???",
        asset_type="ERC20", token_contract="0x1111", token_metadata_missing=True,
    )

    result = analyze_temporal_and_amounts(g, WALLET)

    assert result.per_asset[0].metadata_incomplete is True
    assert any("token metadata" in line for line in result.limitations)


def test_hourly_and_weekday_histograms_use_utc():
    g = nx.MultiDiGraph()
    # 1970-01-01T00:00:00Z is a Thursday (weekday 3).
    add_edge(g, WALLET, OTHER, "0xa1", ts=0)
    add_edge(g, WALLET, OTHER, "0xa2", ts=3600)  # 01:00Z
    add_edge(g, WALLET, OTHER, "0xa3", ts=3601)  # 01:00Z again

    result = analyze_temporal_and_amounts(g, WALLET)

    assert result.hourly_utc_histogram == {0: 1, 1: 2}
    assert result.weekday_utc_histogram == {3: 3}
    assert result.busiest_utc_hour == 1
    assert result.busiest_utc_weekday == 3


def test_gap_statistics_and_iso_rendering():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET, OTHER, "0xa1", ts=0)
    add_edge(g, WALLET, OTHER, "0xa2", ts=100)
    add_edge(g, WALLET, OTHER, "0xa3", ts=400)

    result = analyze_temporal_and_amounts(g, WALLET)

    assert result.median_gap_seconds == 200  # gaps are 100 and 300
    assert result.mean_gap_seconds == 200.0
    assert result.first_seen_utc == "1970-01-01T00:00:00Z"
    assert result.last_seen_utc == "1970-01-01T00:06:40Z"


def test_pass_through_measures_inbound_to_next_outbound():
    g = nx.MultiDiGraph()
    add_edge(g, OTHER, WALLET, "0xin1", ts=100)
    add_edge(g, WALLET, THIRD, "0xout1", ts=160)
    add_edge(g, OTHER, WALLET, "0xin2", ts=1000)  # no later outbound

    result = analyze_temporal_and_amounts(g, WALLET)

    assert result.pass_through is not None
    assert result.pass_through.measured_events == 1
    assert result.pass_through.min_seconds == 60
    assert result.pass_through.max_seconds == 60
    assert result.pass_through.inbound_without_later_outbound == 1
    # The non-continuity caveat must travel with the number.
    assert "does NOT establish" in result.pass_through.limitation


def test_pass_through_absent_when_one_direction_missing():
    g = nx.MultiDiGraph()
    add_edge(g, OTHER, WALLET, "0xin1", ts=100)

    result = analyze_temporal_and_amounts(g, WALLET)

    assert result.pass_through is None


def test_analysis_is_deterministic_across_runs():
    g = nx.MultiDiGraph()
    for i in range(8):
        add_edge(g, WALLET, OTHER, f"0x{i:02x}", ts=100 + i, amount=float(i))
        add_edge(g, THIRD, WALLET, f"0xb{i:02x}", ts=200 + i, amount=float(i) * 2)

    first = analyze_temporal_and_amounts(g, WALLET)
    second = analyze_temporal_and_amounts(g, WALLET)

    assert first.model_dump() == second.model_dump()
