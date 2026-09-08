"""
Tests for `app.reporting.brief` -- the at-a-glance projection.

The brief is the shortest output this project produces, which makes it the most
dangerous one: the fewer words there are, the less likely any single one is to
be questioned, and the greater the temptation to round an awkward finding into a
clean one. These tests pin the properties that stop that:

  * THREE STATES STAY THREE. `matched` is True / False / None, and
    INCONCLUSIVE maps to None -- never to False. A bounded search that found
    no route is not a wallet with no route.
  * IT COMPUTES NOTHING. Every field must equal the value the pipeline already
    produced, so the brief, the compact report and `--json` can never disagree.
  * ABSENT IS N/A. A value the analysis did not produce is not defaulted,
    zeroed, or omitted.
  * THE ML DISCLAIMER IS UNCONDITIONAL. It appears next to a boring verdict as
    well as an interesting one; a disclaimer that only accompanies bad news
    teaches a reader to read its absence as a clean bill of health.
  * A REFUSAL IS A VERDICT. "No model was trainable, because X" occupies the
    line where a reader expects a prediction, rather than leaving it blank.
  * IT IS SHORT. Five blocks, no methodology prose, and no wall of text.

Fully offline. No API key, no network.
"""

from __future__ import annotations

import networkx as nx
import pytest

from app.attribution.bidirectional_models import (
    BidirectionalAttributionResult,
    BidirectionalCandidate,
    ConnectionDirection,
    DirectionalEvidence,
    ExactIdentityMatch,
    SearchAccounting,
)
from app.attribution.models import AttributionStatus, EvidenceTier, SeedSourceType
from app.behavior.models import BehaviorPattern, PatternType
from app.investigation import pipeline as pl
from app.ml.real_predictor import PredictionResult
from app.ml.real_training import EvaluationMetrics, TrainingOutcome
from app.ml.unsupervised import OutlierAssessment
from app.reporting import brief as br

WALLET = "0x" + "11" * 20
PEER = "0x" + "22" * 20
FAR = "0x" + "33" * 20
VASP = "0x" + "44" * 20

TX_A = "0x" + "aa" * 32
TX_B = "0x" + "bb" * 32
TX_C = "0x" + "cc" * 32


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _edge(graph, source, target, tx_hash, timestamp):
    graph.add_edge(
        source,
        target,
        key=f"{tx_hash}#0",
        tx_hash=tx_hash,
        block_number=100,
        timestamp=timestamp,
        amount=1.5,
        asset="ETH",
        asset_type="NATIVE",
        transfer_type="NATIVE_TRANSACTION",
        transfer_source="NATIVE_TRANSACTION",
        status="SUCCESS",
        chain="ethereum",
        token_contract=None,
        gas_used=21000,
    )


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    _edge(graph, WALLET, PEER, TX_A, 1_700_000_000)
    _edge(graph, PEER, FAR, TX_B, 1_700_000_600)
    _edge(graph, FAR, WALLET, TX_C, 1_700_001_200)
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


def _evidence(*addresses, hop_distance=None, tx_hashes=None) -> DirectionalEvidence:
    path = list(addresses)
    return DirectionalEvidence(
        hop_distance=hop_distance if hop_distance is not None else len(path) - 1,
        path_addresses=path,
        tx_hashes=list(tx_hashes) if tx_hashes is not None else [TX_A, TX_B][: len(path) - 1],
        hop_timestamps=[1_700_000_000] * max(len(path) - 1, 0),
        evidence_tier=EvidenceTier.DIRECT if len(path) == 2 else EvidenceTier.INDIRECT,
    )


def _candidate(direction=ConnectionDirection.INDIRECT_OUTBOUND, **overrides):
    fields = dict(
        vasp_name="Example Exchange",
        matched_address=VASP,
        entity_type="exchange",
        chain="ethereum",
        source_type=SeedSourceType.OFFICIAL_DISCLOSURE,
        seed_source="official disclosure",
        seed_confidence_note="published by the operator",
        direction=direction,
        outbound_evidence=_evidence(WALLET, PEER, VASP),
    )
    fields.update(overrides)
    return BidirectionalCandidate(**fields)


def _attribution(status=AttributionStatus.MATCH_FOUND, **overrides):
    fields = dict(
        wallet=WALLET,
        status=status,
        candidates=[_candidate()] if status is AttributionStatus.MATCH_FOUND else [],
        max_hops=3,
        outbound_search_truncated=False,
        inbound_searches_truncated=False,
        seed_address_count=6,
    )
    fields.update(overrides)
    return BidirectionalAttributionResult(**fields)


def _with(report, **fields) -> pl.InvestigationReport:
    """A deep copy of the report with some sections replaced.

    Mutating the module-scoped fixture would leak state between tests, and the
    point of every test here is what the projection does with one exact input.
    """
    stated = report.model_copy(deep=True)
    for name, value in fields.items():
        setattr(stated, name, value)
    return stated


# --------------------------------------------------------------------------
# the three-state answer
# --------------------------------------------------------------------------


def test_a_match_is_yes(report):
    brief = br.build_brief(_with(report, attribution=_attribution()))
    assert brief.vasp_match.status == "MATCH_FOUND"
    assert brief.vasp_match.matched is True
    assert "YES" in br.render_brief(brief)


def test_a_complete_negative_is_no(report):
    brief = br.build_brief(
        _with(report, attribution=_attribution(status=AttributionStatus.NONE))
    )
    assert brief.vasp_match.matched is False
    rendered = br.render_brief(brief)
    assert "VASP MATCH: NO" in rendered
    # A negative is against this dataset, not against all VASPs, and the
    # brief has to say which -- otherwise "NO" reads as a clearance.
    assert "6 dataset address(es)" in brief.vasp_match.note
    assert "not against all VASPs" in brief.vasp_match.note


def test_an_inconclusive_search_is_never_rounded_to_no(report):
    """The single most important assertion in this file."""
    brief = br.build_brief(
        _with(report, attribution=_attribution(status=AttributionStatus.INCONCLUSIVE))
    )
    assert brief.vasp_match.matched is None, "INCONCLUSIVE must not become False"
    assert brief.vasp_match.status == "INCONCLUSIVE"
    rendered = br.render_brief(brief)
    assert "VASP MATCH: INCONCLUSIVE" in rendered
    assert "VASP MATCH: NO" not in rendered


def test_inconclusive_names_which_limit_bit(report):
    """A budget stop and a data horizon call for opposite remedies -- re-run
    wider versus acquire more data -- so the brief names which one happened."""
    attribution = _attribution(
        status=AttributionStatus.INCONCLUSIVE,
        outbound_accounting=SearchAccounting(
            direction="OUTBOUND",
            complete=False,
            incomplete_reason="data observed to 2 hop(s) only -- deeper hops were never acquired",
        ),
    )
    brief = br.build_brief(_with(report, attribution=attribution))
    assert brief.vasp_match.note.startswith("SEARCH INCOMPLETE - ")
    assert "deeper hops were never acquired" in brief.vasp_match.note


def test_inconclusive_without_accounting_still_explains_itself(report):
    brief = br.build_brief(
        _with(report, attribution=_attribution(status=AttributionStatus.INCONCLUSIVE))
    )
    assert "absence of a match is not evidence of absence" in brief.vasp_match.note


def test_an_inconclusive_status_overrides_a_path_it_happens_to_carry(report):
    """A truncated search can still have found one path. The status is the
    answer to "did we finish looking", and the path does not upgrade it."""
    attribution = _attribution(
        status=AttributionStatus.INCONCLUSIVE, candidates=[_candidate()]
    )
    brief = br.build_brief(_with(report, attribution=attribution))
    assert brief.vasp_match.matched is None
    assert brief.vasp_match.evidence_path, "the evidence must still be reported"
    assert "SEARCH INCOMPLETE" in brief.vasp_match.note or "cut short" in brief.vasp_match.note


def test_attribution_that_did_not_run_is_not_run_not_no(report):
    brief = br.build_brief(_with(report, attribution=None))
    assert brief.vasp_match.status == "NOT_RUN"
    assert brief.vasp_match.matched is None
    assert "did not run" in brief.vasp_match.note


# --------------------------------------------------------------------------
# the evidence path
# --------------------------------------------------------------------------


def test_the_path_and_its_hashes_come_through_in_full(report):
    brief = br.build_brief(_with(report, attribution=_attribution()))
    match = brief.vasp_match
    assert match.evidence_path == [WALLET, PEER, VASP]
    assert match.evidence_tx_hashes == [TX_A, TX_B]
    rendered = br.render_brief(brief)
    for value in (WALLET, PEER, VASP, TX_A, TX_B):
        assert value in rendered, "evidence may never be abbreviated"


def test_there_is_one_hash_per_hop_not_one_per_address(report):
    """`evidence_tx_hashes[i]` is the transfer out of `evidence_path[i]`, so it
    is exactly one shorter than the path. An off-by-one here would attribute a
    hop to the wrong transaction."""
    brief = br.build_brief(_with(report, attribution=_attribution()))
    match = brief.vasp_match
    assert len(match.evidence_tx_hashes) == len(match.evidence_path) - 1


def test_a_hop_with_no_recorded_hash_is_none_not_the_next_hops_hash(report):
    """Shifting the next hash up would silently relabel a hop with a
    transaction that does not describe it."""
    candidate = _candidate(
        outbound_evidence=_evidence(WALLET, PEER, VASP, tx_hashes=[TX_A])
    )
    brief = br.build_brief(_with(report, attribution=_attribution(candidates=[candidate])))
    assert brief.vasp_match.evidence_tx_hashes == [TX_A, None]
    assert "N/A" in br.render_brief(brief)


def test_the_shortest_path_is_the_one_reported(report):
    """Fewer intermediaries means fewer unproven links, so the shortest traced
    path is the strongest evidence available."""
    long_candidate = _candidate(
        vasp_name="Far Exchange",
        matched_address="0x" + "55" * 20,
        outbound_evidence=_evidence(WALLET, PEER, FAR, "0x" + "55" * 20),
    )
    short_candidate = _candidate(
        vasp_name="Near Exchange",
        direction=ConnectionDirection.DIRECT_OUTBOUND,
        outbound_evidence=_evidence(WALLET, VASP),
    )
    brief = br.build_brief(
        _with(report, attribution=_attribution(candidates=[long_candidate, short_candidate]))
    )
    assert brief.vasp_match.vasp_name == "Near Exchange"
    assert brief.vasp_match.hop_distance == 1


def test_a_multi_hop_path_states_it_is_not_proof_of_fund_continuity(report):
    brief = br.build_brief(_with(report, attribution=_attribution()))
    assert brief.vasp_match.hop_distance == 2
    assert "not proof that the same funds moved end to end" in brief.vasp_match.note


def test_a_direct_transfer_says_direct_and_claims_nothing_more(report):
    candidate = _candidate(
        direction=ConnectionDirection.DIRECT_OUTBOUND,
        outbound_evidence=_evidence(WALLET, VASP),
    )
    brief = br.build_brief(_with(report, attribution=_attribution(candidates=[candidate])))
    assert brief.vasp_match.note == "Direct transfer (WALLET -> VASP)."


def test_an_inbound_only_candidate_is_reported_on_its_own(report):
    """VASP -> wallet is meaningful without wallet -> VASP; a brief that only
    understood the outbound direction would silently drop it."""
    candidate = _candidate(
        direction=ConnectionDirection.DIRECT_INBOUND,
        outbound_evidence=None,
        inbound_evidence=_evidence(VASP, WALLET),
    )
    brief = br.build_brief(_with(report, attribution=_attribution(candidates=[candidate])))
    assert brief.vasp_match.direction == "DIRECT_INBOUND"
    assert brief.vasp_match.evidence_path == [VASP, WALLET]
    assert "VASP -> WALLET" in brief.vasp_match.note


def test_a_match_with_no_traced_path_says_so_instead_of_printing_an_empty_one(report):
    candidate = _candidate(
        direction=ConnectionDirection.DIRECT_OUTBOUND, outbound_evidence=None
    )
    brief = br.build_brief(_with(report, attribution=_attribution(candidates=[candidate])))
    assert brief.vasp_match.matched is True
    assert brief.vasp_match.evidence_path == []
    assert brief.vasp_match.note == "Matched without a traced multi-hop path."


def test_the_wallet_being_a_dataset_address_is_an_identity_not_a_flow(report):
    identity = ExactIdentityMatch(
        vasp_name="Example Exchange",
        matched_address=WALLET,
        entity_type="exchange",
        chain="ethereum",
        source_type=SeedSourceType.OFFICIAL_DISCLOSURE,
        seed_source="official disclosure",
        seed_confidence_note="published by the operator",
    )
    brief = br.build_brief(
        _with(report, attribution=_attribution(exact_identity_match=identity))
    )
    assert brief.vasp_match.wallet_is_known_vasp == "Example Exchange"
    assert "THE WALLET ITSELF IS A DATASET ADDRESS" in br.render_brief(brief)


# --------------------------------------------------------------------------
# risk and behaviour: two or three lines
# --------------------------------------------------------------------------


def test_the_risk_line_reports_the_assessed_score_and_band(report):
    brief = br.build_brief(report)
    risk = report.risk
    line = brief.risk_summary[0]
    assert line.startswith(f"RISK: {risk.band.value}")
    assert f"score {risk.score}" in line
    assert f"MEDIUM at {risk.band_medium_threshold}" in line
    assert f"HIGH at {risk.band_high_threshold}" in line


def test_the_risk_line_never_reads_as_a_score_out_of_a_maximum(report):
    """"X of a possible Y" reads as a maxed-out risk, which it is not: the
    pair measures how much discounting happened, not progress to a ceiling."""
    line = br.build_brief(report).risk_summary[0]
    assert "of a possible" not in line
    assert f"{report.risk.score}/" not in line


def test_the_indicator_counts_are_triggered_of_checked(report):
    brief = br.build_brief(report)
    triggered = len(report.risk.components)
    checked = triggered + len(report.risk.components_not_triggered)
    assert f"from {triggered} triggered indicator(s) of {checked} checked." in brief.risk_summary[0]


def test_risk_scoring_that_did_not_run_prints_na(report):
    brief = br.build_brief(_with(report, risk=None))
    assert brief.risk_summary[0] == "RISK: N/A (risk scoring did not run)."


def test_behaviour_names_three_patterns_and_counts_the_rest(report):
    patterns = [
        BehaviorPattern(
            pattern_type=pattern,
            wallet=WALLET,
            evidence=["evidence"],
            metrics={},
            related_addresses=[],
            observed_metric="metric",
            observed_value=5,
            threshold=4,
            threshold_setting="setting",
        )
        for pattern in (
            PatternType.SPLIT_PATTERN,
            PatternType.CONSOLIDATION_PATTERN,
            PatternType.HIGH_FREQUENCY_COUNTERPARTY,
            PatternType.RAPID_HOPPING,
            PatternType.TEMPORAL_BURST,
        )
    ]
    brief = br.build_brief(_with(report, behavior_patterns=patterns))
    line = next(line for line in brief.risk_summary if line.startswith("BEHAVIOUR:"))
    assert "5 indicator(s)" in line
    assert "SPLIT_PATTERN, CONSOLIDATION_PATTERN, HIGH_FREQUENCY_COUNTERPARTY" in line
    assert "(+2 more)" in line


def test_behaviour_is_labelled_as_an_indicator_not_a_finding_of_wrongdoing(report):
    patterns = [
        BehaviorPattern(
            pattern_type=PatternType.SPLIT_PATTERN,
            wallet=WALLET,
            evidence=["evidence"],
            metrics={},
            related_addresses=[],
            observed_metric="metric",
            observed_value=5,
            threshold=4,
            threshold_setting="setting",
        )
    ]
    brief = br.build_brief(_with(report, behavior_patterns=patterns))
    line = next(line for line in brief.risk_summary if line.startswith("BEHAVIOUR:"))
    assert "requiring further verification" in line
    assert "not findings of wrongdoing" in line


def test_no_behaviour_says_nothing_crossed_a_threshold(report):
    brief = br.build_brief(_with(report, behavior_patterns=[]))
    assert any(
        "no behavioural indicator crossed its threshold" in line
        for line in brief.risk_summary
    )


def test_an_incomplete_dataset_adds_a_third_line_naming_the_reason(report):
    provenance = _provenance(
        data_complete=False,
        incompleteness_reasons=["the cache predates recent activity"],
    )
    brief = br.build_brief(_with(report, provenance=provenance))
    assert brief.risk_summary[-1] == "DATA: INCOMPLETE - the cache predates recent activity"


def test_a_complete_dataset_adds_no_data_line(report):
    brief = br.build_brief(report)
    assert not any(line.startswith("DATA:") for line in brief.risk_summary)


def test_the_risk_summary_stays_two_or_three_lines(report):
    assert 2 <= len(br.build_brief(report).risk_summary) <= 3


# --------------------------------------------------------------------------
# ML: one line, always with its disclaimer
# --------------------------------------------------------------------------


def _prediction(**overrides) -> PredictionResult:
    fields = dict(
        address=WALLET,
        task="account_type",
        available=True,
        predicted_class="CONTRACT",
        probability=0.83,
        model_name="HistGradientBoosting",
        model_version="real-training-v1",
    )
    fields.update(overrides)
    return PredictionResult(**fields)


def _metrics(**overrides) -> EvaluationMetrics:
    fields = dict(
        split="test",
        sample_count=190,
        accuracy=0.8421,
        precision=0.6882,
        recall=0.9846,
        f1=0.8101,
        confusion_matrix=[[95, 29], [1, 65]],
        positive_class="CONTRACT",
        class_counts={"CONTRACT": 66, "EXTERNALLY_OWNED_ACCOUNT": 124},
        decision_threshold=0.5,
        majority_class_baseline_accuracy=0.6526,
        accuracy_above_baseline=0.1895,
    )
    fields.update(overrides)
    return EvaluationMetrics(**fields)


def test_a_supervised_verdict_is_one_line_naming_the_class_and_the_model(report):
    ml = pl.MLSection(
        approach="SUPERVISED",
        prediction=_prediction(),
        training=TrainingOutcome(
            task="account_type", trained=True, test_metrics=_metrics()
        ),
    )
    brief = br.build_brief(_with(report, ml=ml))
    assert brief.ml.verdict == (
        "account_type = CONTRACT (p=0.83), from a HistGradientBoosting trained "
        "on real labels."
    )
    assert brief.ml.verdict.count("\n") == 0


def test_held_out_metrics_are_the_test_split_ones_verbatim(report):
    """A brief that quoted a training or validation metric as the model's
    accuracy would overstate it, which is the exact failure mode this
    project's ML rules exist to prevent."""
    metrics = _metrics()
    ml = pl.MLSection(
        approach="SUPERVISED",
        prediction=_prediction(),
        training=TrainingOutcome(
            task="account_type",
            trained=True,
            validation_metrics=_metrics(split="validation", accuracy=0.99, f1=0.99),
            test_metrics=metrics,
        ),
    )
    brief = br.build_brief(_with(report, ml=ml))
    assert brief.ml.held_out_accuracy == metrics.accuracy
    assert brief.ml.held_out_f1 == metrics.f1
    assert brief.ml.held_out_accuracy != 0.99


def test_a_prediction_with_no_training_record_reports_no_metrics_not_zero(report):
    ml = pl.MLSection(approach="SUPERVISED", prediction=_prediction())
    brief = br.build_brief(_with(report, ml=ml))
    assert brief.ml.held_out_f1 is None
    assert brief.ml.held_out_accuracy is None
    assert "accuracy=N/A" in br.render_brief(brief)


def test_an_unsupervised_verdict_states_the_population_it_compared_against(report):
    ml = pl.MLSection(
        approach="UNSUPERVISED",
        outlier=OutlierAssessment(
            address=WALLET,
            available=True,
            flagged_as_outlier=True,
            percentile_within_population=96.27,
            population_size=2388,
        ),
    )
    brief = br.build_brief(_with(report, ml=ml))
    assert "unusual" in brief.ml.verdict
    assert "96.27 percentile of 2388 address(es)" in brief.ml.verdict
    assert "No labelled model was trainable" in brief.ml.verdict
    # An outlier score is not a class prediction, and must not read like one.
    assert "VASP" not in brief.ml.verdict


def test_an_unremarkable_wallet_is_stated_as_unremarkable_not_omitted(report):
    ml = pl.MLSection(
        approach="UNSUPERVISED",
        outlier=OutlierAssessment(
            address=WALLET,
            available=True,
            flagged_as_outlier=False,
            percentile_within_population=12.0,
            population_size=2388,
        ),
    )
    brief = br.build_brief(_with(report, ml=ml))
    assert "unremarkable" in brief.ml.verdict


def test_an_unavailable_model_reports_the_reason_as_the_verdict(report):
    ml = pl.MLSection(
        approach="UNAVAILABLE",
        prediction=PredictionResult(
            address=WALLET,
            task="vasp_ownership",
            available=False,
            unavailable_reason="no trained artifact exists for vasp_ownership",
        ),
    )
    brief = br.build_brief(_with(report, ml=ml))
    assert brief.ml.verdict == (
        "No ML verdict: no trained artifact exists for vasp_ownership"
    )
    assert brief.ml.model_name is None


def test_an_unavailable_model_never_reports_a_number(report):
    ml = pl.MLSection(
        approach="UNAVAILABLE",
        rationale=["labels were insufficient in both tasks"],
    )
    brief = br.build_brief(_with(report, ml=ml))
    rendered = br.render_brief(brief)
    ml_block = rendered[rendered.find("ML:") :]
    assert "labels were insufficient in both tasks" in brief.ml.verdict
    assert "MODEL:" not in ml_block, "no model line without a model"
    assert "F1=" not in ml_block, "no metric may be invented for a refusal"


def test_disabled_ml_says_the_evidence_is_unaffected(report):
    brief = br.build_brief(_with(report, ml=pl.MLSection(approach="DISABLED")))
    assert "disabled for this run" in brief.ml.verdict
    assert "evidence above is unaffected" in brief.ml.verdict


def test_ml_that_did_not_run_at_all_is_unavailable_not_absent(report):
    brief = br.build_brief(_with(report, ml=None))
    assert brief.ml.approach == "UNAVAILABLE"
    assert brief.ml.verdict == "ML did not run for this investigation."


@pytest.mark.parametrize(
    "ml",
    [
        None,
        pl.MLSection(approach="DISABLED"),
        pl.MLSection(approach="UNAVAILABLE", rationale=["nothing was trainable"]),
        pl.MLSection(
            approach="UNSUPERVISED",
            outlier=OutlierAssessment(
                address=WALLET,
                available=True,
                flagged_as_outlier=False,
                percentile_within_population=1.0,
                population_size=10,
            ),
        ),
    ],
)
def test_the_disclaimer_is_present_in_every_ml_state(report, ml):
    """Unconditional by design: a disclaimer that only appears next to bad
    news teaches a reader to read its absence as a clean bill of health."""
    brief = br.build_brief(_with(report, ml=ml))
    assert brief.ml.disclaimer == br.ML_DISCLAIMER
    assert "ADVISORY ONLY" in br.render_brief(brief)
    assert "never overrides the VASP attribution evidence" in brief.ml.disclaimer
    assert "never indicates wrongdoing" in brief.ml.disclaimer


# --------------------------------------------------------------------------
# the projection computes nothing
# --------------------------------------------------------------------------


def test_every_header_field_is_the_reports_own_value(report):
    brief = br.build_brief(report)
    assert brief.wallet == report.wallet
    assert brief.chain == report.chain
    assert brief.investigation_id == report.investigation_id
    assert brief.data_mode == report.provenance.data_mode.value
    assert brief.duration_seconds == report.duration_seconds


def test_the_data_mode_is_always_shown(report):
    """Without it a brief looks identical whether the data was fetched live or
    reloaded from a file written weeks ago."""
    rendered = br.render_brief(br.build_brief(report))
    assert f"DATA MODE: {report.provenance.data_mode.value}" in rendered


def test_warnings_are_carried_verbatim(report):
    brief = br.build_brief(_with(report, warnings=["WARNING_MARKER"]))
    assert brief.warnings == ["WARNING_MARKER"]
    assert "WARNING: WARNING_MARKER" in br.render_brief(brief)


def test_the_brief_agrees_with_the_full_report_on_the_attribution_status(report):
    """Two renderers of one model may never disagree; a reader who sees NO in
    the brief and INCONCLUSIVE in --json cannot tell which is the finding."""
    for status in AttributionStatus:
        stated = _with(report, attribution=_attribution(status=status))
        brief = br.build_brief(stated)
        assert brief.vasp_match.status == stated.attribution.status.value


# --------------------------------------------------------------------------
# it is actually brief
# --------------------------------------------------------------------------


def test_the_rendered_brief_is_short(report):
    rendered = br.render_brief(br.build_brief(_with(report, attribution=_attribution())))
    assert len(rendered.splitlines()) < 45, "the brief grew a tail"


def test_the_brief_is_shorter_than_the_compact_report(report):
    from app.reporting import compact as cp

    stated = _with(report, attribution=_attribution())
    brief_lines = len(br.render_brief(br.build_brief(stated)).splitlines())
    compact_lines = len(cp.render_compact_report(stated).splitlines())
    assert brief_lines < compact_lines


def test_no_methodology_prose_reaches_the_brief(report):
    rendered = br.render_brief(br.build_brief(_with(report, attribution=_attribution())))
    upper = rendered.upper()
    for phrase in ("HOW TO READ", "METHODOLOGY", "EVIDENCE CLASSES", "WHAT THIS MEANS"):
        assert phrase not in upper


def test_the_brief_is_pure_ascii(report):
    """cp1252 consoles fold anything else to '?', including in the place a
    reader needs to see direction."""
    rendered = br.render_brief(br.build_brief(_with(report, attribution=_attribution())))
    assert rendered.isascii()


def test_no_line_exceeds_the_rule_width_except_evidence(report):
    """Prose wraps at the rule; an address or hash is never split, because a
    chopped hash cannot be copied or searched for."""
    rendered = br.render_brief(br.build_brief(_with(report, attribution=_attribution())))
    for line in rendered.splitlines():
        if len(line) > br.WIDTH:
            assert any(
                token in line for token in (WALLET, PEER, VASP, TX_A, TX_B)
            ), f"prose overran the rule width: {line}"


def test_the_footer_names_the_other_two_outputs(report):
    """A brief that did not say where the detail went would read as if the
    detail no longer existed."""
    rendered = br.render_brief(br.build_brief(report))
    assert "--full-report" in rendered
    assert "--json" in rendered


def test_print_brief_report_writes_the_brief_to_stdout(report, capsys):
    br.print_brief_report(report)
    out = capsys.readouterr().out
    assert "WALLET INVESTIGATION -- BRIEF" in out
    assert report.wallet in out
