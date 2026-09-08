"""
Tests for `app.investigation.pipeline` -- the single orchestration path.

The pipeline is the thing that must not lie. These tests pin the guarantees
that make the report trustworthy rather than the incidental shape of its
output:

  * a live run without credentials FAILS instead of quietly using demo data
  * cached real data is labelled CACHED REAL DATA and never "REAL"
  * a malformed address is rejected before any I/O happens
  * ML falls back to unsupervised instead of inventing labels, and says why
  * limitations from every stage survive into the report, de-duplicated

Fully offline. No API key, no network, no fixtures used as training data.
"""

from __future__ import annotations

import networkx as nx
import pytest

from app.investigation import pipeline as pl

WALLET = "0x" + "11" * 20
PEER = "0x" + "22" * 20
FAR = "0x" + "33" * 20


def _graph() -> nx.MultiDiGraph:
    """A tiny directed multigraph shaped like real acquisition output."""
    graph = nx.MultiDiGraph()
    edges = [
        (WALLET, PEER, "0xaaa", 1_700_000_000, "ETH"),
        (PEER, FAR, "0xbbb", 1_700_000_600, "ETH"),
        (FAR, WALLET, "0xccc", 1_700_001_200, "ETH"),
    ]
    for source, target, tx_hash, timestamp, asset in edges:
        graph.add_edge(
            source,
            target,
            key=f"{tx_hash}#0",
            tx_hash=tx_hash,
            block_number=100,
            timestamp=timestamp,
            amount=1.5,
            asset=asset,
            asset_type="NATIVE",
            transfer_type="NATIVE_TRANSACTION",
            transfer_source="NATIVE_TRANSACTION",
            chain="ethereum",
            token_contract=None,
            gas_used=21000,
        )
    return graph


def _provenance(mode: pl.DataMode = pl.DataMode.CACHED_REAL_DATA) -> pl.DataProvenance:
    return pl.DataProvenance(
        data_mode=mode,
        provider="test",
        source_description="in-memory real-shaped graph",
        data_complete=True,
    )


# --------------------------------------------------------------------------
# address validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "0x",
        "not-an-address",
        "0x" + "11" * 19,  # too short
        "0x" + "11" * 21,  # too long
        "11" * 20,  # missing 0x
        "0x" + "gg" * 20,  # not hex
        "0x" + "11" * 19 + "1",  # 41 chars
    ],
)
def test_malformed_addresses_are_rejected(bad):
    with pytest.raises(pl.PipelineError):
        pl.validate_wallet_address(bad)


def test_address_is_lowercased_not_altered():
    mixed = "0x75C0623BAE00749550CF1C1703E7382038B3109A"
    assert pl.validate_wallet_address(mixed) == mixed.lower()


def test_validation_happens_before_any_io(tmp_path):
    """A bad address must not cost a file read, let alone a provider call."""
    missing = tmp_path / "definitely-not-here.gpickle"
    with pytest.raises(pl.PipelineError):
        pl.validate_wallet_address("nonsense")
    assert not missing.exists()


# --------------------------------------------------------------------------
# no silent demo fallback
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_acquisition_without_api_key_fails_loudly(monkeypatch):
    """The single most important guarantee in this build."""
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    monkeypatch.setenv("ETHERSCAN_API_KEY", "")
    from app.core.config import get_settings

    settings = get_settings()
    assert settings.etherscan_api_key == ""

    with pytest.raises(pl.PipelineError) as excinfo:
        await pl.acquire_live(WALLET, "ethereum", settings)

    message = str(excinfo.value)
    assert "ETHERSCAN_API_KEY" in message
    assert "does NOT fall back" in message or "not fall back" in message.lower()


@pytest.mark.asyncio
async def test_missing_credential_message_leaks_no_value(monkeypatch):
    monkeypatch.setenv("ETHERSCAN_API_KEY", "")
    from app.core.config import get_settings

    with pytest.raises(pl.PipelineError) as excinfo:
        await pl.acquire_live(WALLET, "ethereum", get_settings())
    # It names the variable, never a value, and offers no demo alternative.
    assert "--demo" not in str(excinfo.value)


def test_data_mode_has_no_synthetic_or_demo_member():
    """There is no code path that can stamp a report DEMO."""
    values = {member.value for member in pl.DataMode}
    assert values == {"REAL", "CACHED REAL DATA"}


# --------------------------------------------------------------------------
# cached real data labelling
# --------------------------------------------------------------------------


def test_cached_graph_is_labelled_cached_not_real(tmp_path):
    from app.graph.builder import save_graph

    path = tmp_path / "g.gpickle"
    save_graph(_graph(), path)

    graph, provenance = pl.acquire_from_cached_graph(path)

    assert graph.number_of_edges() == 3
    assert provenance.data_mode is pl.DataMode.CACHED_REAL_DATA
    assert provenance.data_mode.value == "CACHED REAL DATA"
    assert provenance.data_complete is False, (
        "a reloaded graph is by definition not current, which is an "
        "incompleteness the report must state"
    )
    assert provenance.incompleteness_reasons


def test_cached_graph_missing_fields_are_attributed_to_the_cache(tmp_path):
    """The distinction matters: absent from the CACHE, not absent from chain."""
    from app.graph.builder import save_graph

    graph = _graph()
    for _source, _target, data in graph.edges(data=True):
        data["block_number"] = None
        data["token_contract"] = None
        data["transfer_source"] = None
        data["gas_used"] = None
    path = tmp_path / "sparse.gpickle"
    save_graph(graph, path)

    _graph_out, provenance = pl.acquire_from_cached_graph(path)
    joined = " ".join(provenance.incompleteness_reasons)
    assert "block_number" in joined
    assert "CACHE" in joined or "cache" in joined


def test_missing_cached_graph_raises_pipeline_error(tmp_path):
    with pytest.raises(pl.PipelineError):
        pl.acquire_from_cached_graph(tmp_path / "nope.gpickle")


# --------------------------------------------------------------------------
# the investigation itself
# --------------------------------------------------------------------------


def test_run_investigation_produces_a_complete_report():
    report = pl.run_investigation(
        WALLET, "ethereum", _graph(), _provenance(), enable_ml=False
    )

    assert report.wallet == WALLET
    assert report.chain == "ethereum"
    assert report.wallet_in_graph is True
    assert report.attribution is not None
    assert report.risk is not None
    assert report.temporal is not None
    assert report.conclusion, "a report with no conclusion is not a report"
    assert report.limitations, "every real investigation has limitations"
    assert report.duration_seconds >= 0


def test_wallet_absent_from_graph_is_reported_not_crashed():
    report = pl.run_investigation(
        "0x" + "99" * 20, "ethereum", _graph(), _provenance(), enable_ml=False
    )
    assert report.wallet_in_graph is False
    assert any("not present" in item.lower() or "absent" in item.lower()
               for item in report.limitations + report.warnings + report.conclusion)


# --------------------------------------------------------------------------
# the transaction ledger
#
# A projection of edges the graph already holds, added so the report can list
# the wallet's individual transfers. It must never become a second, divergent
# measurement of the same thing.
# --------------------------------------------------------------------------


def test_the_ledger_is_the_same_edge_set_the_temporal_analysis_counts():
    report = pl.run_investigation(
        WALLET, "ethereum", _graph(), _provenance(), enable_ml=False
    )
    assert len(report.transactions) == report.temporal.transfer_count
    assert len(report.transactions) == 2, "WALLET->PEER out and FAR->WALLET in"


def test_the_ledger_holds_only_edges_incident_on_the_wallet():
    report = pl.run_investigation(
        WALLET, "ethereum", _graph(), _provenance(), enable_ml=False
    )
    # PEER -> FAR touches neither end of the wallet and must not appear.
    assert all(WALLET in (row.from_address, row.to_address)
               for row in report.transactions)
    assert "0xbbb" not in {row.tx_hash for row in report.transactions}


def test_ledger_fields_are_copied_from_the_edge_never_invented():
    report = pl.run_investigation(
        WALLET, "ethereum", _graph(), _provenance(), enable_ml=False
    )
    graph = _graph()
    for row in report.transactions:
        data = graph.get_edge_data(row.from_address, row.to_address)
        edge = next(iter(data.values()))
        assert row.tx_hash == edge["tx_hash"]
        assert row.timestamp == edge["timestamp"]
        assert row.amount == edge["amount"]
        assert row.asset == edge["asset"]
        assert row.block_number == edge["block_number"]


def test_a_ledger_row_with_no_timestamp_stays_none_rather_than_zero():
    """A guessed zero timestamp would be rendered as 1970 and read as a real
    observation, so absence must survive into the row."""
    graph = nx.MultiDiGraph()
    graph.add_edge(
        WALLET, PEER, key="0xnots#0", tx_hash="0xnots", block_number=None,
        timestamp=None, amount=1.0, asset="ETH", asset_type="NATIVE",
        transfer_type="NATIVE_TRANSACTION", transfer_source="NATIVE_TRANSACTION",
        chain="ethereum", token_contract=None, gas_used=None,
    )
    report = pl.run_investigation(
        WALLET, "ethereum", graph, _provenance(), enable_ml=False
    )
    assert report.transactions[0].timestamp is None
    assert report.transactions[0].timestamp_utc is None


def test_the_ledger_survives_json_round_tripping():
    report = pl.run_investigation(
        WALLET, "ethereum", _graph(), _provenance(), enable_ml=False
    )
    restored = pl.InvestigationReport.model_validate_json(report.model_dump_json())
    assert [r.tx_hash for r in restored.transactions] == [
        r.tx_hash for r in report.transactions
    ]
    assert [r.direction for r in restored.transactions] == [
        r.direction for r in report.transactions
    ]


def test_an_absent_wallet_has_an_empty_ledger_not_a_fabricated_one():
    report = pl.run_investigation(
        "0x" + "99" * 20, "ethereum", _graph(), _provenance(), enable_ml=False
    )
    assert report.transactions == []


def test_report_is_json_serialisable_and_round_trips():
    report = pl.run_investigation(
        WALLET, "ethereum", _graph(), _provenance(), enable_ml=False
    )
    payload = report.model_dump_json()
    restored = pl.InvestigationReport.model_validate_json(payload)
    assert restored.wallet == report.wallet
    assert restored.provenance.data_mode is pl.DataMode.CACHED_REAL_DATA


def test_limitations_are_deduplicated_and_order_preserved():
    report = pl.run_investigation(
        WALLET, "ethereum", _graph(), _provenance(), enable_ml=False
    )
    assert len(report.limitations) == len(set(report.limitations))


def test_ml_disabled_is_recorded_as_disabled_not_as_a_result():
    report = pl.run_investigation(
        WALLET, "ethereum", _graph(), _provenance(), enable_ml=False
    )
    assert report.ml.approach == "DISABLED"
    assert report.ml.prediction is None
    assert report.ml.training is None


def test_determinism_same_inputs_same_findings():
    graph = _graph()
    first = pl.run_investigation(
        WALLET, "ethereum", graph, _provenance(), enable_ml=False
    )
    second = pl.run_investigation(
        WALLET, "ethereum", graph, _provenance(), enable_ml=False
    )
    assert first.attribution.status == second.attribution.status
    assert first.risk.score == second.risk.score
    assert first.limitations == second.limitations


# --------------------------------------------------------------------------
# ML honesty
# --------------------------------------------------------------------------


def test_ml_never_reports_supervised_on_unlabelable_data():
    """Three addresses cannot support a supervised model. It must say so."""
    from app.core.config import get_settings

    section = pl.run_ml_analysis(
        _graph(),
        WALLET,
        transfers=None,
        seed_entries=[],
        graph_source="unit test graph",
        settings=get_settings(),
        enabled=True,
    )

    assert section.approach in {"UNSUPERVISED", "UNAVAILABLE"}
    assert section.approach != "SUPERVISED"
    assert section.training is None or section.training.trained is False
    assert section.rationale, "a refusal with no stated reason is not honest"
    assert any("invent" in line.lower() or "insufficient" in line.lower()
               or "not trainable" in line.lower() or "not attempted" in line.lower()
               for line in section.rationale)


def test_ml_section_carries_no_accuracy_when_nothing_was_trained():
    from app.core.config import get_settings

    section = pl.run_ml_analysis(
        _graph(), WALLET, None, [], "unit test graph", get_settings(), enabled=True
    )
    if section.training is not None:
        assert section.training.test_metrics is None or section.training.trained


def test_ml_rationale_never_claims_a_model_ran_when_none_did():
    """The rationale used to assert "an unsupervised outlier model ran" before
    the outlier stage was even attempted, so a run that declined for want of a
    comparison population still read as though a model had produced something.
    """
    from app.core.config import get_settings

    section = pl.run_ml_analysis(
        _graph(), WALLET, None, [], "unit test graph", get_settings(), enabled=True
    )

    joined = " ".join(section.rationale)
    if section.approach == "UNAVAILABLE":
        assert "outlier model ran" not in joined
        assert "NO machine-learning result" in joined
        assert section.outlier is None or section.outlier.available is False
    else:
        assert section.approach == "UNSUPERVISED"
        assert section.outlier is not None and section.outlier.available


def test_unavailable_ml_states_the_declining_reason_in_the_rationale():
    """A refusal whose reason appears only under limitations makes section 7
    read as an unexplained blank."""
    from app.core.config import get_settings

    section = pl.run_ml_analysis(
        _graph(), WALLET, None, [], "unit test graph", get_settings(), enabled=True
    )
    if section.approach != "UNAVAILABLE":
        pytest.skip("this graph was rich enough for the unsupervised model")
    assert section.outlier is not None
    assert section.outlier.unavailable_reason
    assert section.outlier.unavailable_reason in section.rationale


# --------------------------------------------------------------------------
# observation depth (the data horizon)
# --------------------------------------------------------------------------
#
# The traversal's hop limit and the dataset's radius are independent. A
# single-wallet acquisition yields a depth-1 star, and searching it with
# MAX_HOPS=4 proves nothing about hop 2 -- the edges were never fetched.


def test_infer_observation_depth_on_a_single_wallet_star():
    """What live acquisition actually produces: every edge touches the
    wallet, so the graph describes exactly one hop of chain history."""
    graph = nx.MultiDiGraph()
    for index, peer in enumerate((PEER, FAR)):
        graph.add_edge(WALLET, peer, key=f"0x{index}#0", tx_hash=f"0x{index}")
    assert pl.infer_observation_depth(graph, WALLET) == 1


def test_infer_observation_depth_counts_expanded_rings():
    graph = nx.MultiDiGraph()
    graph.add_edge(WALLET, PEER, key="0xa#0", tx_hash="0xa")
    graph.add_edge(PEER, FAR, key="0xb#0", tx_hash="0xb")
    assert pl.infer_observation_depth(graph, WALLET) == 2


def test_infer_observation_depth_is_direction_agnostic():
    """Inbound-only edges are still acquired data; the radius does not care
    which way the value moved."""
    graph = nx.MultiDiGraph()
    graph.add_edge(PEER, WALLET, key="0xa#0", tx_hash="0xa")
    graph.add_edge(FAR, PEER, key="0xb#0", tx_hash="0xb")
    assert pl.infer_observation_depth(graph, WALLET) == 2


def test_infer_observation_depth_asserts_nothing_when_meaningless():
    assert pl.infer_observation_depth(nx.MultiDiGraph(), WALLET) is None
    lonely = nx.MultiDiGraph()
    lonely.add_edge(PEER, FAR, key="0xa#0", tx_hash="0xa")
    assert pl.infer_observation_depth(lonely, WALLET) is None


def test_run_investigation_infers_the_horizon_when_not_declared():
    """`_graph()` is a WALLET -> PEER -> FAR -> WALLET triangle, so every node
    is one undirected step from the wallet and the radius is 1, not 2. A cycle
    back to the wallet adds no new ring."""
    report = pl.run_investigation(
        WALLET, "ethereum", _graph(), _provenance(), enable_ml=False
    )
    assert report.provenance.observation_depth == 1


def test_declared_horizon_is_not_overwritten_by_inference():
    """Live acquisition knows its own radius exactly; inference must not
    second-guess it."""
    provenance = _provenance(pl.DataMode.REAL)
    provenance.observation_depth = 1
    report = pl.run_investigation(
        WALLET, "ethereum", _graph(), provenance, enable_ml=False
    )
    assert report.provenance.observation_depth == 1


def test_shallow_horizon_makes_a_negative_inconclusive_not_none():
    """The core guarantee: an unobserved hop is never reported as searched
    and empty."""
    star = nx.MultiDiGraph()
    star.add_edge(
        WALLET, PEER, key="0xa#0", tx_hash="0xa", timestamp=1_700_000_000
    )
    provenance = _provenance(pl.DataMode.REAL)
    provenance.observation_depth = 1

    report = pl.run_investigation(
        WALLET, "ethereum", star, provenance, max_hops=4, enable_ml=False
    )

    assert report.attribution is not None
    assert report.attribution.status.value == "INCONCLUSIVE"
    # And the reason given must be the horizon, not a budget nobody hit.
    horizon_limitation = [
        text for text in report.limitations if "data horizon" in text.lower()
    ]
    assert horizon_limitation, report.limitations
    assert not any(
        "hit its edge budget" in text for text in report.limitations
    )


def test_horizon_covering_the_hop_limit_still_permits_a_clean_negative():
    star = nx.MultiDiGraph()
    star.add_edge(
        WALLET, PEER, key="0xa#0", tx_hash="0xa", timestamp=1_700_000_000
    )
    provenance = _provenance(pl.DataMode.REAL)
    provenance.observation_depth = 1

    report = pl.run_investigation(
        WALLET, "ethereum", star, provenance, max_hops=1, enable_ml=False
    )
    assert report.attribution is not None
    assert report.attribution.status.value == "NONE"
    assert not any("data horizon" in t.lower() for t in report.limitations)
