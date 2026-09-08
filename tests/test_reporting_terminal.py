"""
Tests for `app.reporting.terminal`.

Three classes of guarantee are pinned here, all of which have already been
broken once during development:

  * ENCODING -- the rendered report must be pure ASCII. A Windows console on
    code page 1252 turns UTF-8 em dashes into mojibake, and reconfiguring the
    stream does not change how the console interprets the bytes.
  * EVIDENCE CORRECTNESS -- route hops are labelled by comparing addresses,
    never by position. An inbound route runs VASP -> ... -> wallet, so
    positional labelling names the wrong address as the investigated wallet
    exactly half the time.
  * READABILITY OF STRUCTURED FIELDS -- list and dict fields are rendered as
    bullets and key=value text, never as a Python repr. `evidence: ['a', 'b']`
    in an investigation report is a bug, not a formatting preference.
"""

from __future__ import annotations

import networkx as nx
import pytest

from app.investigation import pipeline as pl
from app.reporting import terminal as tr

WALLET = "0x" + "11" * 20
PEER = "0x" + "22" * 20
VASP = "0x" + "33" * 20


# --------------------------------------------------------------------------
# to_ascii
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, expected",
    [
        ("a—b", "a - b"),
        ("a–b", "a-b"),
        ("‘q’", "'q'"),
        ("“q”", '"q"'),
        ("wait…", "wait..."),
        ("a→b", "a->b"),
        ("a←b", "a<-b"),
        ("x≥y", "x>=y"),
        ("x≤y", "x<=y"),
        ("a b", "a b"),
        ("5±3", "5+/-3"),
        ("2×3", "2x3"),
        ("• item", "* item"),
    ],
)
def test_known_typography_folds_to_ascii(source, expected):
    assert tr.to_ascii(source) == expected


def test_unknown_non_ascii_is_substituted_visibly_not_dropped():
    out = tr.to_ascii("中文")
    assert out.isascii()
    assert "?" in out, "silent deletion would hide that anything was lost"


def test_accented_letters_decompose_to_their_base_letter():
    assert tr.to_ascii("café") == "cafe"


def test_to_ascii_is_idempotent():
    once = tr.to_ascii("a — b → c")
    assert tr.to_ascii(once) == once


# --------------------------------------------------------------------------
# small formatters
# --------------------------------------------------------------------------


def test_amount_never_reformats_precision():
    assert tr._amount("1.500000") == "1.5"
    assert tr._amount("0.000000000000000001") == "0.000000000000000001"
    assert tr._amount("1000") == "1000"
    assert tr._amount(None) == "(amount unavailable)"


def test_timestamp_absence_is_stated_not_shown_as_epoch():
    assert tr._ts(None) == "(no timestamp)"
    assert tr._ts(0) == "(no timestamp)"
    assert tr._ts(1_700_000_000) == "2023-11-14 22:13:20Z"


def test_counts_renders_key_value_pairs_not_a_dict_repr():
    assert tr._counts({"B": 2, "A": 1}) == "A=1, B=2"
    assert tr._counts({}) == "(none)"
    assert "{" not in tr._counts({"A": 1})


def test_weekday_is_named_so_the_numbering_is_unambiguous():
    assert tr._weekday(0) == "0 (Mon)"
    assert tr._weekday(6) == "6 (Sun)"
    assert tr._weekday(None) == "(not available)"


def test_wrap_never_splits_a_transaction_hash():
    tx = "0x" + "ab" * 32
    lines = tr._wrap(f"the transaction {tx} is the evidence", indent="  ", width=40)
    joined = "\n".join(lines)
    assert tx in joined, "a hash split across lines cannot be copied or searched"


def test_short_is_used_for_diagrams_only_and_marks_absence():
    assert tr._short(None) == "(unknown)"
    assert "..." in tr._short(WALLET)


# --------------------------------------------------------------------------
# full report rendering
# --------------------------------------------------------------------------


def _graph_with_inbound_vasp() -> nx.MultiDiGraph:
    """VASP -> PEER -> WALLET, i.e. the direction that broke hop labelling."""
    graph = nx.MultiDiGraph()
    hops = [
        (VASP, PEER, "0x" + "a" * 64, 1_700_000_000),
        (PEER, WALLET, "0x" + "b" * 64, 1_700_000_600),
        (WALLET, PEER, "0x" + "c" * 64, 1_700_001_200),
    ]
    for source, target, tx_hash, timestamp in hops:
        graph.add_edge(
            source,
            target,
            key=f"{tx_hash}#0",
            tx_hash=tx_hash,
            block_number=100,
            timestamp=timestamp,
            amount=2.0,
            asset="ETH",
            asset_type="NATIVE",
            transfer_type="NATIVE_TRANSACTION",
            transfer_source="NATIVE_TRANSACTION",
            chain="ethereum",
            token_contract=None,
            gas_used=21000,
        )
    return graph


@pytest.fixture()
def rendered() -> str:
    report = pl.run_investigation(
        WALLET,
        "ethereum",
        _graph_with_inbound_vasp(),
        pl.DataProvenance(
            data_mode=pl.DataMode.CACHED_REAL_DATA,
            provider="test",
            source_description="unit test graph",
            data_complete=True,
        ),
        enable_ml=False,
    )
    return tr.render_report(report)


def test_rendered_report_is_pure_ascii(rendered):
    assert rendered.isascii(), [c for c in rendered if not c.isascii()][:10]


def test_report_contains_all_nine_sections(rendered):
    for number, fragment in [
        (1, "BLOCKCHAIN DATA SUMMARY"),
        (2, "FUND-FLOW ANALYSIS"),
        (3, "VASP ATTRIBUTION"),
        (4, "BIDIRECTIONAL"),
        (5, "BEHAVIORAL INTELLIGENCE"),
        (6, "TEMPORAL AND AMOUNT ANALYSIS"),
        (7, "MACHINE LEARNING ANALYSIS"),
        (8, "EVIDENCE SUMMARY"),
        (9, "FINAL INVESTIGATION CONCLUSION"),
    ]:
        assert f"SECTION {number}" in rendered
        assert fragment in rendered


def test_report_states_the_data_mode_and_never_claims_live(rendered):
    assert "DATA MODE:" in rendered
    assert "CACHED REAL DATA" in rendered


def test_report_opens_with_wallet_and_chain_and_closes_with_completion(rendered):
    assert WALLET in rendered
    assert "ethereum" in rendered
    assert "INVESTIGATION COMPLETE" in rendered


def test_report_contains_no_python_reprs(rendered):
    """A list or dict repr in a report is a rendering bug."""
    offenders = [
        line
        for line in rendered.splitlines()
        if "['" in line or "', '" in line or line.strip().startswith("[{")
    ]
    assert offenders == [], offenders[:5]


def test_a_path_is_never_called_proof_of_fund_movement(rendered):
    assert "TRANSACTION PATH" in rendered
    lowered = rendered.lower()
    assert "proof that the same" in lowered or "not proof" in lowered


def test_behavioural_findings_are_never_called_criminal(rendered):
    lowered = rendered.lower()
    for forbidden in ("is criminal", "criminal activity detected", "is a criminal",
                      "confirmed fraud", "proven money laundering"):
        assert forbidden not in lowered


def test_ml_disabled_says_so_rather_than_printing_an_empty_section(rendered):
    index = rendered.find("SECTION 7")
    section = rendered[index : rendered.find("SECTION 8")]
    assert "DISABLED" in section.upper()


def test_render_is_deterministic():
    report = pl.run_investigation(
        WALLET,
        "ethereum",
        _graph_with_inbound_vasp(),
        pl.DataProvenance(
            data_mode=pl.DataMode.CACHED_REAL_DATA,
            provider="test",
            source_description="unit test graph",
            data_complete=True,
        ),
        enable_ml=False,
    )
    first = tr.render_report(report)
    second = tr.render_report(report)
    assert first == second


# --------------------------------------------------------------------------
# hop labelling -- the reversed-label bug
# --------------------------------------------------------------------------


class _FakeEvidence:
    """Just enough of a `DirectionalEvidence` for `_render_route`.

    Field names mirror the real model exactly; a fake that drifts from the
    model it stands in for would let a renderer bug pass.
    """

    def __init__(self, addresses):
        self.path_addresses = list(addresses)
        self.tx_hashes = ["0x" + "a" * 64, "0x" + "b" * 64]
        self.hop_timestamps = [1_700_000_000, 1_700_000_600]
        self.amounts = [1.0, 1.0]
        self.assets = ["ETH", "ETH"]
        self.block_numbers = [100, 101]
        self.edge_keys = ["k0", "k1"]
        self.hop_distance = 2
        self.path_duration_seconds = 600
        self.plausibility = None
        self.alternative_path_count = 0
        self.evidence_tier = None


def test_inbound_route_labels_the_wallet_as_the_wallet():
    """The bug: hop 0 was hardcoded as the investigated wallet, so on an
    inbound route the VASP was announced as the wallet under investigation."""
    lines = tr._render_route(_FakeEvidence([VASP, PEER, WALLET]), WALLET, VASP)
    text = "\n".join(lines)

    wallet_line = next(line for line in lines if WALLET in line)
    vasp_line = next(line for line in lines if VASP in line)

    assert "investigated wallet" in wallet_line
    assert "VASP dataset address" in vasp_line
    assert "investigated wallet" not in vasp_line
    assert PEER in text and "intermediary" in text


def test_outbound_route_labels_are_equally_correct():
    lines = tr._render_route(_FakeEvidence([WALLET, PEER, VASP]), WALLET, VASP)
    wallet_line = next(line for line in lines if WALLET in line)
    vasp_line = next(line for line in lines if VASP in line)
    assert "investigated wallet" in wallet_line
    assert "VASP dataset address" in vasp_line


def test_hop_labels_are_case_insensitive():
    lines = tr._render_route(
        _FakeEvidence([VASP.upper(), PEER, WALLET.upper()]), WALLET, VASP
    )
    assert any("investigated wallet" in line for line in lines)
    assert any("VASP dataset address" in line for line in lines)


def test_route_states_when_a_hop_field_is_missing():
    evidence = _FakeEvidence([WALLET, PEER, VASP])
    evidence.tx_hashes = []
    evidence.hop_timestamps = []
    evidence.block_numbers = []
    text = "\n".join(tr._render_route(evidence, WALLET, VASP))
    assert "unavailable" in text or "no timestamp" in text or "not recorded" in text


# --------------------------------------------------------------------------
# stream configuration
# --------------------------------------------------------------------------


def test_configure_stdout_tolerates_a_stream_without_reconfigure():
    class _Plain:
        def write(self, _text):
            return None

    tr.configure_stdout(_Plain())  # must not raise


def test_print_report_writes_ascii_to_the_given_stream():
    import io

    report = pl.run_investigation(
        WALLET,
        "ethereum",
        _graph_with_inbound_vasp(),
        pl.DataProvenance(
            data_mode=pl.DataMode.CACHED_REAL_DATA,
            provider="test",
            source_description="unit test graph",
            data_complete=True,
        ),
        enable_ml=False,
    )
    buffer = io.StringIO()
    tr.print_report(report, stream=buffer)
    written = buffer.getvalue()
    assert written.isascii()
    assert "INVESTIGATION COMPLETE" in written


# --------------------------------------------------------------------------
# display caps -- keeping a real mainnet wallet readable without ever
# silently dropping evidence
#
# Uncapped, a real eight-year-old mainnet wallet rendered 23,357 lines, of
# which 16,530 were a list of 2,302 counterparties. That pushed the
# attribution, ML and conclusion sections thousands of lines below the fold.
# A cap is only acceptable if it says so, so every property below is about the
# HONESTY of the cap rather than about the cap itself.
# --------------------------------------------------------------------------


def _wide_graph(counterparties: int = 60, assets: int = 30) -> nx.MultiDiGraph:
    """A wallet with many counterparties and many assets, like a real one."""
    graph = nx.MultiDiGraph()
    base_ts = 1_700_000_000
    for index in range(counterparties):
        peer = "0x" + f"{index:040x}"
        asset = "ETH" if index % assets == 0 else f"TKN{index % assets}"
        # Several transfers each, so the frequency indicators trigger the way
        # they do on real data and the cap has something to act on.
        for occurrence in range(6):
            tx_hash = "0x" + f"{index:032x}{occurrence:032x}"
            graph.add_edge(
                peer if index % 2 else WALLET,
                WALLET if index % 2 else peer,
                key=f"{tx_hash}#0",
                tx_hash=tx_hash,
                block_number=1000 + index,
                timestamp=base_ts + index * 3600 + occurrence * 60,
                amount=1.5,
                asset=asset,
                asset_type="NATIVE" if asset == "ETH" else "ERC20",
                transfer_type="NATIVE_TRANSACTION",
                transfer_source="NATIVE_TRANSACTION",
                chain="ethereum",
                token_contract=None if asset == "ETH" else "0x" + f"{index:040x}",
                gas_used=21000,
            )
    return graph


@pytest.fixture(scope="module")
def wide_report():
    return pl.run_investigation(
        WALLET,
        "ethereum",
        _wide_graph(),
        pl.DataProvenance(
            data_mode=pl.DataMode.CACHED_REAL_DATA,
            provider="test",
            source_description="wide unit test graph",
            data_complete=True,
        ),
        enable_ml=False,
    )


@pytest.fixture(scope="module")
def wide_rendered(wide_report) -> str:
    return tr.render_report(wide_report)


def test_a_wide_wallet_still_renders_a_readable_report(wide_rendered):
    """The regression these caps exist for: an unbounded report is unusable.

    This fixture -- 60 counterparties, 30 assets, 67 indicators, 67 risk
    contributions -- rendered 1,520 lines once the per-list caps were in place
    and 1,112 once repeated findings stopped getting a full evidence block
    each. The bound below is that measured figure plus headroom; it is a
    regression guard, so if a change pushes the report past it the right
    response is to look at what grew, not to raise the number.
    """
    assert len(wide_rendered.splitlines()) < 1200
    assert wide_rendered.isascii()
    assert "SECTION 9" in wide_rendered, "the conclusion must not be pushed off"


def test_every_capped_list_states_what_it_withheld(wide_rendered):
    """A truncated list with no note is indistinguishable from a short one."""
    assert "not shown here" in wide_rendered
    assert "nothing has been discarded" in wide_rendered
    assert "--json" in wide_rendered, "the complete record must be pointed at"


def test_the_omission_note_is_absent_when_nothing_was_withheld(rendered):
    assert "not shown here" not in rendered


def test_omitted_reports_the_true_total_and_remainder():
    text = " ".join(tr._omitted(shown=25, total=2302, noun="counterparties"))
    assert "2277" in text, "2302 - 25"
    assert "2302" in text
    assert tr._omitted(shown=25, total=25, noun="x") == []
    assert tr._omitted(shown=25, total=3, noun="x") == []


def test_a_dataset_identified_counterparty_is_never_the_row_that_is_dropped(
    wide_report,
):
    """Capping on transfer count alone would hide the one row that matters.

    A named VASP counterparty with a single transfer is more consequential than
    an unnamed one with two hundred, so identified entities are ordered first
    and the cap falls behind them.
    """
    quiet_named = wide_report.counterparties[-1].model_copy(
        update={
            "entity_name": "Example Exchange",
            "entity_type": "centralized_exchange",
            "transfer_count": 1,
            "inbound_count": 1,
            "outbound_count": 0,
        }
    )
    report = wide_report.model_copy(
        update={"counterparties": [*wide_report.counterparties[:-1], quiet_named]}
    )
    text = tr.render_report(report)
    assert quiet_named.address in text
    assert "Example Exchange" in text


def test_the_behaviour_census_accounts_for_every_finding(wide_report, wide_rendered):
    """Detail may be capped; the count may not be."""
    total = len(wide_report.behavior_patterns)
    assert total > tr.MAX_BEHAVIOR_INDICATORS_SHOWN, "fixture does not exercise the cap"
    assert f"INDICATORS OBSERVED: {total}" in wide_rendered


def test_every_distinct_indicator_type_survives_the_cap(wide_report, wide_rendered):
    """Twenty copies of one indicator and none of the other nine would be a
    strictly worse report than a shorter, more varied one."""
    section = wide_rendered[
        wide_rendered.find("SECTION 5") : wide_rendered.find("SECTION 6")
    ]
    for pattern in wide_report.behavior_patterns:
        name = tr._v(pattern.pattern_type)
        assert name in section, f"indicator type {name} is invisible in section 5"


def test_behaviour_display_order_puts_first_of_each_type_ahead_of_repeats():
    class _P:
        def __init__(self, kind, tag):
            self.pattern_type = kind
            self.tag = tag

    patterns = [_P("A", 1), _P("A", 2), _P("B", 3), _P("A", 4), _P("C", 5)]
    ordered = tr._behavior_display_order(patterns)
    assert [p.tag for p in ordered] == [1, 3, 5, 2, 4]


def test_the_risk_arithmetic_still_reconciles_when_contributions_are_capped(
    wide_report, wide_rendered
):
    """The strongest claim section 8 makes is that its itemisation sums to the
    score. A cap that broke that sum would have to remove the claim."""
    assert wide_report.risk is not None
    assert len(wide_report.risk.components) > tr.MAX_RISK_CONTRIBUTIONS_SHOWN
    assert "arithmetic verified" in wide_rendered
    assert "DOES NOT MATCH REPORTED SCORE" not in wide_rendered
    assert "not itemised above" in wide_rendered


def test_every_printed_contribution_plus_the_withheld_subtotal_equals_the_score(
    wide_report, wide_rendered
):
    """The arithmetic claim must survive BOTH devices section 8 uses to stay
    short: full itemisation for the first contribution of each indicator, one
    compact line carrying its own value for each repeat, and a single subtotal
    for whatever is past the per-indicator limit. Add up everything printed
    between the CONTRIBUTIONS heading and the TOTAL line and it must come to
    the score exactly -- otherwise 'nothing hidden' is not true.
    """
    import re

    start = wide_rendered.find("CONTRIBUTIONS (")
    end = wide_rendered.find("TOTAL (", start)
    assert start != -1 and end > start
    block = wide_rendered[start:end]

    printed = 0.0
    itemised_in_full = 0
    compact_rows = 0
    subtotals = 0
    for value, rest in re.findall(r"^\s+\+ (\d+\.?\d*)\s+(.*)$", block, re.M):
        printed += float(value)
        if rest.startswith("from "):
            subtotals += 1
        elif rest.startswith("["):
            itemised_in_full += 1
        else:
            compact_rows += 1

    assert itemised_in_full > 0, "no contribution was itemised in full"
    assert compact_rows > 0, "the fixture does not exercise the compact rows"
    assert subtotals == 1, "the remainder must be one explicit subtotal line"
    assert abs(printed - wide_report.risk.score) < 1e-6, (
        f"printed {printed} but the score is {wide_report.risk.score}"
    )


def test_a_repeated_indicator_is_itemised_once_and_then_listed_compactly(
    wide_report, wide_rendered
):
    """Sixty copies of one twenty-five-line evidence block convey one fact and
    sixty addresses. The first copy earns the block; the rest earn a line."""
    counts: dict[str, int] = {}
    for component in wide_report.risk.components:
        name = tr._v(component.indicator)
        counts[name] = counts.get(name, 0) + 1
    repeated = max(counts, key=lambda n: counts[n])
    assert counts[repeated] > 1, "fixture must contain a repeated indicator"

    assert "FURTHER CONTRIBUTIONS, GROUPED BY INDICATOR" in wide_rendered
    assert f"{repeated}: {counts[repeated] - 1} further contribution(s)" in wide_rendered
    # The reason paragraph is identical across repeats; it must be printed for
    # the itemised one and not sixty times over.
    assert wide_rendered.count("crossed its configured threshold") <= len(counts)


def test_repeated_behaviour_findings_are_collapsed_but_still_counted(
    wide_report, wide_rendered
):
    counts: dict[str, int] = {}
    for pattern in wide_report.behavior_patterns:
        name = tr._v(pattern.pattern_type)
        counts[name] = counts.get(name, 0) + 1
    repeated = max(counts, key=lambda n: counts[n])
    assert counts[repeated] > tr.MAX_BEHAVIOR_REPEATS_PER_TYPE

    section = wide_rendered[
        wide_rendered.find("SECTION 5") : wide_rendered.find("SECTION 6")
    ]
    assert "FURTHER FINDINGS, GROUPED BY INDICATOR TYPE" in section
    assert f"{repeated}: {counts[repeated] - 1} further finding(s)" in section
    # The census above still accounts for all of them.
    assert f"{repeated:<34} {counts[repeated]}" in section


def test_a_compact_repeat_row_prints_the_whole_address(wide_report, wide_rendered):
    """A shortened address in an evidence row cannot be re-checked on-chain."""
    section = wide_rendered[
        wide_rendered.find("FURTHER FINDINGS, GROUPED BY INDICATOR TYPE") :
        wide_rendered.find("SECTION 6")
    ]
    rows = [line for line in section.splitlines() if "0x" in line and " = " in line]
    assert rows, "no compact rows were rendered"
    for row in rows:
        assert "..." not in row, row
        address = row.strip().split()[0]
        assert len(address) == 42, row


def test_the_native_asset_is_shown_even_when_it_is_not_the_busiest(wide_report):
    """A wallet spammed with airdropped tokens can push ETH below a cap on
    transfer count; hiding the native-value breakdown would be indefensible."""
    assert wide_report.temporal is not None
    natives = [
        stats
        for stats in wide_report.temporal.per_asset
        if tr._v(stats.asset_type).upper() == "NATIVE"
    ]
    assert natives, "fixture must contain a native asset"
    quiet_native = natives[0].model_copy(update={"transfer_count": 1})
    others = [s for s in wide_report.temporal.per_asset if s is not natives[0]]
    temporal = wide_report.temporal.model_copy(
        update={"per_asset": [*others, quiet_native]}
    )
    report = wide_report.model_copy(update={"temporal": temporal})
    text = tr.render_report(report)
    index = text.find("PER-ASSET AMOUNTS")
    assert index != -1
    per_asset_block = text[index : text.find("SECTION 7")]
    assert f"{quiet_native.asset} (" in per_asset_block


def test_asset_list_names_a_few_and_counts_the_rest():
    assert tr._asset_list([]) == "(none)"
    assert tr._asset_list(["ETH", "DAI"]) == "ETH, DAI"
    many = [f"T{i}" for i in range(40)]
    text = tr._asset_list(many)
    assert "and 32 more" in text
    assert "40 distinct assets" in text
    assert len(text) < 200, "a summary line must not become a paragraph"
