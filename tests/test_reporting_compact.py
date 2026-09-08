"""
Tests for `app.reporting.compact` -- the default ten-block report.

The compact report exists to be read in ten seconds, which makes it exactly the
kind of output that gets "tidied" into something dishonest. These tests pin the
properties that stop that happening:

  * IT COMPUTES NOTHING. Every number it prints must equal the number the
    analysis layer produced. A compact report that disagrees with `--json` is
    worse than no compact report at all, so the values are compared field by
    field against the same `InvestigationReport`.
  * ABSENT IS NOT ZERO. A value the analysis did not produce prints `N/A`; a
    value the analysis measured as zero prints `0`. Collapsing the two would
    turn "not measured" into "measured, and the answer was none".
  * EVIDENCE IS NEVER ABBREVIATED. Addresses and transaction hashes appear in
    full, because a truncated hash cannot be looked up.
  * TRUNCATION IS ANNOUNCED. A capped list states the total and how many rows
    it withheld, and names `--json` as the complete record.
  * ENCODING. Pure ASCII, for the same cp1252 reason as the full renderer.

Fully offline. No API key, no network.
"""

from __future__ import annotations

import networkx as nx
import pytest

from app.behavior.models import BehaviorPattern, PatternType
from app.investigation import pipeline as pl
from app.reporting import compact as cp

WALLET = "0x" + "11" * 20
PEER = "0x" + "22" * 20
FAR = "0x" + "33" * 20


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _edge(graph, source, target, tx_hash, timestamp, asset="ETH", amount=1.5, **extra):
    graph.add_edge(
        source,
        target,
        key=f"{tx_hash}#0",
        tx_hash=tx_hash,
        block_number=100,
        timestamp=timestamp,
        amount=amount,
        asset=asset,
        asset_type="NATIVE" if asset == "ETH" else "ERC20",
        transfer_type="NATIVE_TRANSACTION",
        transfer_source="NATIVE_TRANSACTION",
        status="SUCCESS",
        chain="ethereum",
        token_contract=None if asset == "ETH" else "0x" + "ab" * 20,
        gas_used=21000,
        **extra,
    )


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    _edge(graph, WALLET, PEER, "0x" + "aa" * 32, 1_700_000_000)
    _edge(graph, PEER, FAR, "0x" + "bb" * 32, 1_700_000_600)
    _edge(graph, FAR, WALLET, "0x" + "cc" * 32, 1_700_001_200)
    return graph


def _provenance(**overrides) -> pl.DataProvenance:
    fields = dict(
        data_mode=pl.DataMode.CACHED_REAL_DATA,
        provider="test",
        source_description="in-memory real-shaped graph",
        data_complete=True,
    )
    fields.update(overrides)
    return pl.DataProvenance(**fields)


@pytest.fixture(scope="module")
def report() -> pl.InvestigationReport:
    return pl.run_investigation(
        WALLET, "ethereum", _graph(), _provenance(), enable_ml=False
    )


@pytest.fixture(scope="module")
def rendered(report) -> str:
    return cp.render_compact_report(report)


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------


def test_all_ten_blocks_are_present_in_order(rendered):
    titles = [
        "[1] WALLET",
        "[2] TRANSACTION SUMMARY",
        "[3] COUNTERPARTIES",
        "[4] TRANSACTION ACTIVITY",
        "[5] TIMING ANALYSIS",
        "[6] ASSET SUMMARY",
        "[7] INVESTIGATION FINDINGS",
        "[8] VASP / ENTITY MATCH",
        "[9] RISK",
        "[10] DATA STATUS",
    ]
    positions = [rendered.find(title) for title in titles]
    assert all(p >= 0 for p in positions), "a block is missing entirely"
    assert positions == sorted(positions), "blocks are out of order"


def test_the_compact_report_is_not_the_nine_section_report(rendered):
    """The two must be distinguishable from their text alone -- a reader
    scrolling back through a terminal has to be able to tell which one they
    are looking at, and so does a test."""
    assert "SECTION 1" not in rendered
    assert "SECTION 9" not in rendered
    assert "COMPACT REPORT" in rendered


def test_the_report_is_pure_ascii(rendered):
    assert rendered.isascii(), "cp1252 consoles would mangle anything else"


def test_the_completion_footer_survives(rendered, report):
    assert "INVESTIGATION COMPLETE" in rendered
    assert report.wallet in rendered
    assert f"DATA MODE: {report.provenance.data_mode.value}" in rendered


def test_the_footer_names_the_flags_for_the_other_two_outputs(rendered):
    """A compact report that does not say where the detail went would read as
    if the detail no longer existed."""
    assert "--full-report" in rendered
    assert "--json" in rendered


# --------------------------------------------------------------------------
# no prose
# --------------------------------------------------------------------------


def test_no_methodology_or_evidence_taxonomy_prose(rendered):
    """The phrases that make the full report long must not appear here."""
    forbidden = [
        "does not prove",
        "is not proof",
        "WHAT THIS MEANS",
        "HOW TO READ",
        "METHODOLOGY",
        "LIMITATIONS",
        "EVIDENCE CLASSES",
    ]
    upper = rendered.upper()
    for phrase in forbidden:
        assert phrase.upper() not in upper, f"prose leaked into compact: {phrase}"


def test_no_line_is_a_paragraph(rendered):
    """Tables, not prose. One long ML reason line is allowed through because
    it is a verbatim analysis field, but nothing else may run long."""
    long_lines = [
        line
        for line in rendered.splitlines()
        if len(line) > 140 and "REASON:" not in line
    ]
    assert not long_lines, f"paragraph-shaped output: {long_lines[:2]}"


# --------------------------------------------------------------------------
# the numbers must match the analysis, not be recomputed
# --------------------------------------------------------------------------


def test_summary_counts_equal_the_temporal_analysis(rendered, report):
    t = report.temporal
    assert f"TOTAL TRANSFERS:       {t.transfer_count}" in rendered
    assert f"INCOMING:              {t.inbound_transfer_count}" in rendered
    assert f"OUTGOING:              {t.outbound_transfer_count}" in rendered


def test_counterparty_rows_equal_the_counterparty_analysis(rendered, report):
    assert f"TOTAL UNIQUE:          {len(report.counterparties)}" in rendered
    for cp_item in report.counterparties:
        assert cp_item.address in rendered, "a counterparty was dropped"


def test_risk_score_and_band_are_the_assessed_ones(rendered, report):
    assert f"SCORE:                 {report.risk.score}" in rendered
    assert f"BAND:                  {report.risk.band.value}" in rendered


def test_every_triggered_risk_indicator_appears(rendered, report):
    for component in report.risk.components:
        assert cp._v(component.indicator) in rendered


def test_data_status_reports_both_hop_depths(rendered, report):
    """The observed depth and the requested depth are different facts and the
    compact report must not collapse them -- that conflation is exactly what
    turns an unsearched hop into a searched-and-empty one."""
    observed = report.provenance.observation_depth
    requested = report.parameters["max_hops"]
    assert f"OBSERVED HOP DEPTH:    {observed}" in rendered
    assert f"REQUESTED HOP DEPTH:   {requested}" in rendered


# --------------------------------------------------------------------------
# the transaction ledger
# --------------------------------------------------------------------------


def test_every_wallet_transfer_is_listed_with_full_hash_and_addresses(
    rendered, report
):
    assert report.transactions, "the ledger must be populated"
    for tx in report.transactions:
        assert tx.tx_hash in rendered, "a transaction hash was abbreviated or dropped"
        assert tx.from_address in rendered
        assert tx.to_address in rendered


def test_direction_is_stated_not_inferred(rendered, report):
    directions = {tx.direction for tx in report.transactions}
    assert directions <= {"IN", "OUT", "SELF"}
    for direction in directions:
        assert direction in rendered


def test_the_ledger_counts_the_same_transfers_the_temporal_analysis_does(report):
    """If these two ever disagreed, the report would print a row count that
    contradicted its own summary and a reader could not tell which was right."""
    assert len(report.transactions) == report.temporal.transfer_count


def test_the_ledger_only_contains_edges_touching_the_wallet(report):
    for tx in report.transactions:
        assert WALLET in (tx.from_address, tx.to_address)


def test_the_ledger_names_the_other_end_as_the_counterparty(report):
    for tx in report.transactions:
        if tx.direction == "OUT":
            assert tx.counterparty == tx.to_address
        elif tx.direction == "IN":
            assert tx.counterparty == tx.from_address
        else:
            assert tx.counterparty is None


def test_the_ledger_is_oldest_first(report):
    stamps = [tx.timestamp for tx in report.transactions if tx.timestamp is not None]
    assert stamps == sorted(stamps)


# --------------------------------------------------------------------------
# N/A discipline
# --------------------------------------------------------------------------


def test_absent_values_print_na_not_a_substitute():
    assert cp._na(None) == "N/A"
    assert cp._na("") == "N/A"
    assert cp._num(None) == "N/A"
    assert cp._amt(None) == "N/A"
    assert cp._time(None) == "N/A"


def test_a_measured_zero_prints_zero_not_na():
    """`if not value` would collapse these into N/A, turning a real finding of
    zero into "we did not look"."""
    assert cp._na(0) == "0"
    assert cp._num(0) == "0"
    assert cp._amt(0) == "0"
    assert cp._na(False) == "False"


def test_a_missing_normalization_report_prints_na_not_zero():
    report = pl.run_investigation(
        WALLET, "ethereum", _graph(), _provenance(), enable_ml=False
    )
    assert report.normalization is None
    rendered = cp.render_compact_report(report)
    status = rendered[rendered.find("[10] DATA STATUS") :]
    assert "RECORDS FETCHED:       N/A" in status
    assert "RECORDS RETAINED:      N/A" in status


def test_a_wallet_with_no_activity_says_so_rather_than_printing_zeros():
    absent = "0x" + "99" * 20
    report = pl.run_investigation(
        absent, "ethereum", _graph(), _provenance(), enable_ml=False
    )
    rendered = cp.render_compact_report(report)
    assert "no transfers for this address in the dataset" in rendered
    assert "INVESTIGATION COMPLETE" in rendered
    assert rendered.isascii()


# --------------------------------------------------------------------------
# ML is one or two lines, never an essay
# --------------------------------------------------------------------------


def test_disabled_ml_is_stated_in_one_line(rendered):
    assert "ML:                    DISABLED" in rendered


def test_unavailable_ml_prints_the_approach_and_the_reason_only(report):
    """The exact shape the brief asked for: `ML: UNAVAILABLE` plus a reason,
    and none of the labelling/training/explainability record."""
    stated = report.model_copy(deep=True)
    stated.ml = pl.MLSection(
        approach="UNAVAILABLE",
        rationale=["a long paragraph that must not be printed here"],
        limitations=["another long paragraph that must not be printed here"],
    )
    rendered = cp.render_compact_report(stated)
    assert "ML:                    UNAVAILABLE" in rendered
    assert "REASON:" in rendered
    assert "must not be printed here" not in rendered


def test_ml_rationale_and_limitations_never_reach_the_compact_report(report):
    stated = report.model_copy(deep=True)
    stated.ml = pl.MLSection(approach="DISABLED", rationale=["RATIONALE_MARKER"])
    stated.limitations = ["LIMITATION_MARKER"]
    stated.conclusion = ["CONCLUSION_MARKER"]
    rendered = cp.render_compact_report(stated)
    assert "RATIONALE_MARKER" not in rendered
    assert "LIMITATION_MARKER" not in rendered
    assert "CONCLUSION_MARKER" not in rendered


# --------------------------------------------------------------------------
# table layout
# --------------------------------------------------------------------------


def test_a_cell_that_exactly_fills_its_column_still_has_a_separator():
    """Without a guaranteed space, `unique_outgoing_counterparties` (30 chars
    in a 30-wide column) fuses with the next cell and the metric name and its
    value read as one token."""
    row = cp._row([("unique_outgoing_counterparties", 30), ("6", 6)])
    assert "unique_outgoing_counterparties 6" in row
    assert "counterparties6" not in row


def test_an_oversized_cell_pushes_the_row_right_instead_of_being_truncated():
    address = "0x" + "ab" * 20
    row = cp._row([(address, 10), ("tail", 6)])
    assert address in row, "an address must never be shortened"
    assert row.endswith("tail")


def test_findings_name_at_most_two_addresses_inline_and_count_the_rest(report):
    """Two full addresses fit the width; three do not, and addresses are never
    abbreviated, so the remainder is counted."""
    stated = report.model_copy(deep=True)
    stated.behavior_patterns = [
        BehaviorPattern(
            pattern_type=PatternType.SPLIT_PATTERN,
            wallet=WALLET,
            evidence=["five peers"],
            metrics={"unique_outgoing_counterparties": 5},
            related_addresses=["0x" + f"{i:040x}" for i in range(5)],
            observed_metric="unique_outgoing_counterparties",
            observed_value=5,
            threshold=4,
            threshold_setting="behavior_split_min_counterparties",
        )
    ]
    rendered = cp.render_compact_report(stated)
    line = next(
        line for line in rendered.splitlines() if line.strip().startswith("ADDRESSES:")
    )
    assert "(+3 more)" in line
    assert len(line) <= 145


def test_the_risk_table_does_not_repeat_the_threshold_from_the_findings_table(
    rendered,
):
    """Block 7 already states every behavioural threshold. Reprinting it in
    block 9 is exactly the duplication the compact report exists to remove."""
    risk = rendered[rendered.find("[9] RISK") : rendered.find("[10] DATA STATUS")]
    header = next(line for line in risk.splitlines() if "INDICATOR" in line)
    assert "THRESHOLD" not in header
    assert "VALUE" in header, "the value must stay -- it disambiguates same-named rows"


def test_data_status_discloses_provider_cache_use(report):
    """A REAL run whose requests were served from the response cache is still
    REAL, but the reader is entitled to know what actually hit the network."""
    stated = report.model_copy(deep=True)
    stated.provenance.cache_stats = {"hits": 3, "misses": 0, "writes": 0}
    rendered = cp.render_compact_report(stated)
    assert "PROVIDER CACHE" in rendered
    assert "hits=3" in rendered


# --------------------------------------------------------------------------
# display caps
# --------------------------------------------------------------------------


def _wide_graph(counterparties: int = 60) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    base = 1_700_000_000
    for index in range(counterparties):
        peer = "0x" + f"{index:040x}"
        for occurrence in range(4):
            tx_hash = "0x" + f"{index:032x}{occurrence:032x}"
            source, target = (
                (peer, WALLET) if index % 2 else (WALLET, peer)
            )
            _edge(
                graph,
                source,
                target,
                tx_hash,
                base + index * 3600 + occurrence * 60,
                asset="ETH" if index % 5 == 0 else f"TKN{index % 25}",
            )
    return graph


@pytest.fixture(scope="module")
def wide_rendered() -> str:
    report = pl.run_investigation(
        WALLET, "ethereum", _wide_graph(), _provenance(), enable_ml=False
    )
    return cp.render_compact_report(report)


def test_a_wide_wallet_stays_readable(wide_rendered):
    assert len(wide_rendered.splitlines()) < 700, "the compact report grew a tail"
    assert wide_rendered.isascii()


def test_every_cap_that_bites_states_what_it_withheld(wide_rendered):
    """A cap is only defensible if it is honest about being a cap."""
    assert "not shown" in wide_rendered
    for line in wide_rendered.splitlines():
        if "not shown" in line:
            assert "in total" in line, "a cap did not state the total"
            assert "--json" in line, "a cap did not name the complete record"


def test_counterparties_are_capped_but_the_true_total_is_printed(wide_rendered):
    assert "TOTAL UNIQUE:          60" in wide_rendered
    assert f"more counterparties not shown" in wide_rendered


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------


def test_print_compact_report_writes_to_the_given_stream(report, capsys):
    cp.print_compact_report(report)
    out = capsys.readouterr().out
    assert "[1] WALLET" in out
    assert "INVESTIGATION COMPLETE" in out
