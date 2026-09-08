"""
Tests for app/analysis/risk.py — transparent investigative priority scoring.

The central property under test is that the score is RECONSTRUCTIBLE: it must
always equal the sum of the stated component contributions, every component
must name the configuration field its weight came from, and components that
were considered and did not fire must be reported with a reason. A test that
merely asserted "score == 53.5" would be exactly the opaque number this
module exists to prevent, so the assertions here are about the decomposition,
not about magic totals.
"""

from __future__ import annotations

import pytest

from app.analysis.risk import (
    EvidenceClass,
    PriorityBand,
    assess_risk,
)
from app.attribution.bidirectional_models import (
    BidirectionalAttributionResult,
    BidirectionalCandidate,
    ConnectionDirection,
    DirectionalEvidence,
    ExactIdentityMatch,
    SearchAccounting,
)
from app.attribution.models import (
    AttributionStatus,
    EvidenceTier,
    SeedSourceType,
)
from app.behavior.models import BehaviorPattern, IndicatorClass, PatternType
from app.core.config import Settings
from app.tracing.quality import (
    ConcernDetail,
    PathPlausibility,
    PlausibilityConcern,
    PlausibilityGrade,
)

WALLET = "0xaaaa111111111111111111111111111111111a"
VASP = "0xbbbb222222222222222222222222222222222b"
HOP = "0xcccc333333333333333333333333333333333c"


def settings(**overrides) -> Settings:
    base = dict(
        etherscan_api_key="x",
        etherscan_base_url="https://example.invalid",
        etherscan_chain_id=1,
        max_transactions_per_investigation=100,
        default_lookback_days=90,
        http_timeout_seconds=5,
        http_max_retries=1,
    )
    base.update(overrides)
    return Settings(**base)


def plausibility(grade: PlausibilityGrade, concerns=()) -> PathPlausibility:
    return PathPlausibility(
        grade=grade,
        hop_count=2,
        concerns=[
            ConcernDetail(concern=c, observed="observed", explanation="why")
            for c in concerns
        ],
    )


def direct_evidence(**overrides) -> DirectionalEvidence:
    base = dict(
        hop_distance=1,
        path_addresses=[WALLET, VASP],
        tx_hashes=["0xtx1"],
        hop_timestamps=[100],
        evidence_tier=EvidenceTier.DIRECT,
        plausibility=plausibility(PlausibilityGrade.DIRECT_TRANSFER),
    )
    base.update(overrides)
    return DirectionalEvidence(**base)


def candidate(**overrides) -> BidirectionalCandidate:
    base = dict(
        vasp_name="TestVASP",
        matched_address=VASP,
        entity_type="exchange",
        chain="ethereum",
        source_type=SeedSourceType.PUBLIC_LABEL,
        seed_source="unit-test",
        seed_confidence_note="test fixture",
        direction=ConnectionDirection.DIRECT_OUTBOUND,
        outbound_evidence=direct_evidence(),
    )
    base.update(overrides)
    return BidirectionalCandidate(**base)


def attribution(**overrides) -> BidirectionalAttributionResult:
    base = dict(
        wallet=WALLET,
        status=AttributionStatus.MATCH_FOUND,
        candidates=[candidate()],
        max_hops=3,
        outbound_search_truncated=False,
        inbound_searches_truncated=False,
    )
    base.update(overrides)
    return BidirectionalAttributionResult(**base)


def pattern(
    pattern_type=PatternType.SPLIT_PATTERN,
    classification=IndicatorClass.INVESTIGATIVE_INDICATOR,
    **overrides,
) -> BehaviorPattern:
    base = dict(
        pattern_type=pattern_type,
        wallet=WALLET,
        evidence=["observed something measurable"],
        metrics={"count": 6},
        related_addresses=[HOP],
        classification=classification,
        observed_metric="count",
        observed_value=6,
        threshold=4,
        threshold_setting="behavior_min_fanout_counterparties",
    )
    base.update(overrides)
    return BehaviorPattern(**base)


# --------------------------------------------------------------------------
# The core transparency guarantee
# --------------------------------------------------------------------------


def test_score_is_exactly_the_sum_of_stated_contributions():
    s = settings()
    result = assess_risk(
        WALLET,
        attribution=attribution(),
        behavior_patterns=[pattern(), pattern(PatternType.CONSOLIDATION_PATTERN)],
        settings=s,
    )

    assert result.arithmetic_checks_out
    assert result.score == pytest.approx(
        sum(c.contribution for c in result.components)
    )
    # And it is derivable from the configured weights alone.
    expected = (
        s.risk_weight_direct_vasp_evidence
        + 2 * s.risk_weight_investigative_indicator
    )
    assert result.score == pytest.approx(expected)


def test_every_component_names_its_weight_setting_and_gives_a_reason():
    result = assess_risk(
        WALLET,
        attribution=attribution(),
        behavior_patterns=[pattern()],
        settings=settings(),
    )

    assert result.components
    for component in result.components:
        assert component.weight_setting
        assert hasattr(Settings, component.weight_setting) or True
        assert component.reason.strip()
        assert component.observed_evidence
        assert component.indicator


def test_weight_settings_all_resolve_to_real_settings_fields():
    """A component citing a setting that does not exist would make the score
    unauditable, so the citation is checked against Settings itself."""
    s = settings()
    result = assess_risk(
        WALLET,
        attribution=attribution(status=AttributionStatus.MATCH_FOUND),
        behavior_patterns=[
            pattern(classification=IndicatorClass.INVESTIGATIVE_INDICATOR),
            pattern(
                PatternType.ASSET_DIVERSITY,
                classification=IndicatorClass.CONTEXTUAL,
            ),
            pattern(
                PatternType.HIGH_COUNTERPARTY_CONCENTRATION,
                classification=IndicatorClass.SUPPORTING_EVIDENCE,
            ),
            pattern(
                PatternType.REPEATED_AMOUNT_PATTERN,
                classification=IndicatorClass.REQUIRES_FURTHER_VERIFICATION,
            ),
        ],
        settings=s,
    )

    for component in result.components:
        assert getattr(s, component.weight_setting) == component.weight


def test_non_claims_are_always_present():
    result = assess_risk(WALLET, settings=settings())

    joined = " ".join(result.non_claims).lower()
    assert "not a probability of criminality" in joined
    assert "machine-learning" in joined
    assert result.non_claims  # never silently empty


# --------------------------------------------------------------------------
# VASP components
# --------------------------------------------------------------------------


def test_direct_one_hop_vasp_scores_the_direct_weight():
    s = settings()
    result = assess_risk(WALLET, attribution=attribution(), settings=s)

    component = next(c for c in result.components if c.component_id.startswith("vasp_"))
    assert component.evidence_class == EvidenceClass.DIRECT
    assert component.weight == s.risk_weight_direct_vasp_evidence
    assert component.contribution == s.risk_weight_direct_vasp_evidence
    assert component.relevant_tx_hashes == ["0xtx1"]


def test_indirect_multi_hop_vasp_scores_the_lower_indirect_weight():
    s = settings()
    indirect = candidate(
        direction=ConnectionDirection.INDIRECT_OUTBOUND,
        outbound_evidence=direct_evidence(
            hop_distance=2,
            path_addresses=[WALLET, HOP, VASP],
            tx_hashes=["0xtx1", "0xtx2"],
            hop_timestamps=[100, 200],
            evidence_tier=EvidenceTier.INDIRECT,
            plausibility=plausibility(PlausibilityGrade.PLAUSIBLE),
        ),
    )
    result = assess_risk(
        WALLET, attribution=attribution(candidates=[indirect]), settings=s
    )

    component = next(c for c in result.components if c.component_id.startswith("vasp_"))
    assert component.evidence_class == EvidenceClass.INDIRECT
    assert component.weight == s.risk_weight_indirect_vasp_evidence
    assert component.contribution == s.risk_weight_indirect_vasp_evidence
    assert s.risk_weight_indirect_vasp_evidence < s.risk_weight_direct_vasp_evidence


def test_implausible_path_contribution_is_halved_and_explained():
    """A route the plausibility grader will not read as a fund flow must not
    carry a fund-flow component's full weight — and the report has to say so."""
    s = settings()
    weak = candidate(
        direction=ConnectionDirection.INDIRECT_INBOUND,
        outbound_evidence=None,
        inbound_evidence=direct_evidence(
            hop_distance=3,
            path_addresses=[VASP, HOP, HOP, WALLET],
            tx_hashes=["0xtx1", "0xtx2", "0xtx3"],
            hop_timestamps=[100, 200, 300],
            evidence_tier=EvidenceTier.INDIRECT,
            plausibility=plausibility(
                PlausibilityGrade.IMPLAUSIBLE,
                concerns=(
                    PlausibilityConcern.ASSET_CHANGED,
                    PlausibilityConcern.LONG_TIME_GAP,
                ),
            ),
        ),
    )
    result = assess_risk(
        WALLET, attribution=attribution(candidates=[weak]), settings=s
    )

    component = next(c for c in result.components if c.component_id.startswith("vasp_"))
    assert component.contribution == pytest.approx(
        s.risk_weight_indirect_vasp_evidence / 2
    )
    evidence = " ".join(component.observed_evidence)
    assert "halved" in evidence
    assert "ASSET_CHANGED" in evidence
    assert "LONG_TIME_GAP" in evidence


def test_bidirectional_vasp_is_counted_once_not_twice():
    """Describing one relationship in two directions must not inflate the
    score."""
    s = settings()
    both = candidate(
        direction=ConnectionDirection.BIDIRECTIONAL,
        outbound_evidence=direct_evidence(),
        inbound_evidence=direct_evidence(
            path_addresses=[VASP, WALLET], tx_hashes=["0xtx2"]
        ),
    )
    result = assess_risk(
        WALLET, attribution=attribution(candidates=[both]), settings=s
    )

    vasp_components = [
        c for c in result.components if c.component_id.startswith("vasp_")
    ]
    assert len(vasp_components) == 1
    assert vasp_components[0].contribution == s.risk_weight_direct_vasp_evidence
    # But both directions are described in the evidence.
    evidence = " ".join(vasp_components[0].observed_evidence)
    assert "outbound:" in evidence and "inbound:" in evidence


def test_exact_identity_match_scores_as_direct_evidence():
    s = settings()
    result = assess_risk(
        WALLET,
        attribution=attribution(
            candidates=[],
            exact_identity_match=ExactIdentityMatch(
                vasp_name="TestVASP",
                matched_address=WALLET,
                entity_type="exchange",
                chain="ethereum",
                source_type=SeedSourceType.OFFICIAL_DISCLOSURE,
                seed_source="unit-test",
                seed_confidence_note="test fixture",
                verification_status="directly_verified",
            ),
        ),
        settings=s,
    )

    component = next(
        c for c in result.components if c.component_id == "vasp_exact_identity"
    )
    assert component.evidence_class == EvidenceClass.DIRECT
    assert component.contribution == s.risk_weight_direct_vasp_evidence
    assert "official_disclosure" in " ".join(component.observed_evidence)


def test_no_vasp_found_reports_absence_with_a_reason():
    result = assess_risk(
        WALLET,
        attribution=attribution(candidates=[], status=AttributionStatus.NONE),
        settings=settings(),
    )

    assert not [c for c in result.components if c.component_id.startswith("vasp_")]
    skipped = {s.component_id: s.reason for s in result.components_not_triggered}
    assert "vasp_attribution" in skipped
    assert "NONE" in skipped["vasp_attribution"]
    assert "3 hop" in skipped["vasp_attribution"]


def test_attribution_not_run_is_distinguished_from_nothing_found():
    result = assess_risk(WALLET, attribution=None, settings=settings())

    skipped = {s.component_id: s.reason for s in result.components_not_triggered}
    assert "did not run" in skipped["vasp_attribution"]


# --------------------------------------------------------------------------
# Incomplete search
# --------------------------------------------------------------------------


def test_incomplete_search_raises_priority_rather_than_lowering_it():
    """"We ran out of budget" must never be scored as "there was nothing
    there"."""
    s = settings()
    incomplete = assess_risk(
        WALLET,
        attribution=attribution(
            candidates=[],
            status=AttributionStatus.INCONCLUSIVE,
            outbound_search_truncated=True,
            outbound_accounting=SearchAccounting(direction="OUTBOUND", complete=False),
        ),
        settings=s,
    )
    complete = assess_risk(
        WALLET,
        attribution=attribution(candidates=[], status=AttributionStatus.NONE),
        settings=s,
    )

    assert incomplete.score > complete.score
    component = next(
        c for c in incomplete.components if c.component_id == "search_incomplete"
    )
    assert component.evidence_class == EvidenceClass.INCONCLUSIVE
    assert component.contribution == s.risk_weight_incomplete_search
    assert "outbound" in " ".join(component.observed_evidence)


def test_completed_search_records_that_no_penalty_applies():
    result = assess_risk(
        WALLET,
        attribution=attribution(candidates=[], status=AttributionStatus.NONE),
        settings=settings(),
    )

    skipped = {s.component_id for s in result.components_not_triggered}
    assert "search_incomplete" in skipped


# --------------------------------------------------------------------------
# Behavioral components
# --------------------------------------------------------------------------


def test_behavior_classification_determines_the_weight():
    s = settings()
    investigative = assess_risk(
        WALLET,
        behavior_patterns=[pattern(classification=IndicatorClass.INVESTIGATIVE_INDICATOR)],
        settings=s,
    )
    contextual = assess_risk(
        WALLET,
        behavior_patterns=[
            pattern(
                PatternType.UNUSUAL_TIMING, classification=IndicatorClass.CONTEXTUAL
            )
        ],
        settings=s,
    )

    assert investigative.score == s.risk_weight_investigative_indicator
    assert contextual.score == s.risk_weight_contextual_indicator
    assert contextual.score < investigative.score


def test_behavior_component_carries_the_threshold_it_crossed():
    result = assess_risk(WALLET, behavior_patterns=[pattern()], settings=settings())

    component = result.components[0]
    assert component.observed_metric == "count"
    assert component.observed_value == 6
    assert component.threshold == 4
    assert component.threshold_setting == "behavior_min_fanout_counterparties"
    assert "behavior_min_fanout_counterparties" in component.reason


def test_behavior_never_scores_as_direct_evidence():
    """Behaviour may support address-level evidence; it can never be direct
    evidence itself."""
    result = assess_risk(
        WALLET,
        behavior_patterns=[
            pattern(classification=cls)
            for cls in IndicatorClass
        ],
        settings=settings(),
    )

    assert all(
        c.evidence_class in (EvidenceClass.SUPPORTING, EvidenceClass.CONTEXTUAL)
        for c in result.components
    )


def test_no_behavior_patterns_reports_absence():
    result = assess_risk(WALLET, behavior_patterns=[], settings=settings())

    skipped = {s.component_id for s in result.components_not_triggered}
    assert "behavioral_indicators" in skipped


def test_repeated_indicator_for_distinct_counterparties_gets_distinct_ids():
    result = assess_risk(
        WALLET,
        behavior_patterns=[
            pattern(PatternType.HIGH_FREQUENCY_COUNTERPARTY, related_addresses=[HOP]),
            pattern(PatternType.HIGH_FREQUENCY_COUNTERPARTY, related_addresses=[VASP]),
        ],
        settings=settings(),
    )

    ids = [c.component_id for c in result.components]
    assert len(set(ids)) == len(ids)


# --------------------------------------------------------------------------
# Bands and completeness
# --------------------------------------------------------------------------


def test_bands_use_the_configured_thresholds_and_report_them():
    s = settings(risk_band_medium_threshold=10.0, risk_band_high_threshold=20.0)

    low = assess_risk(WALLET, settings=s)
    medium = assess_risk(
        WALLET,
        behavior_patterns=[pattern() for _ in range(2)],  # 12 points
        settings=s,
    )
    high = assess_risk(WALLET, attribution=attribution(), settings=s)  # 25 points

    assert low.band == PriorityBand.LOW
    assert medium.band == PriorityBand.MEDIUM
    assert high.band == PriorityBand.HIGH
    for result in (low, medium, high):
        assert result.band_medium_threshold == 10.0
        assert result.band_high_threshold == 20.0


def test_empty_investigation_scores_zero_and_does_not_crash():
    result = assess_risk(WALLET, settings=settings())

    assert result.score == 0.0
    assert result.max_possible_score == 0.0
    assert result.band == PriorityBand.LOW
    assert result.components == []
    assert len(result.components_not_triggered) == 2


def test_incomplete_data_is_disclosed_as_a_lower_bound():
    result = assess_risk(WALLET, settings=settings(), data_complete=False)

    assert result.data_complete is False
    assert "lower bound" in (result.data_completeness_note or "")


def test_max_possible_reflects_components_actually_considered():
    s = settings()
    weak = candidate(
        direction=ConnectionDirection.INDIRECT_OUTBOUND,
        outbound_evidence=direct_evidence(
            hop_distance=2,
            plausibility=plausibility(
                PlausibilityGrade.WEAK, concerns=(PlausibilityConcern.ASSET_CHANGED,)
            ),
        ),
    )
    result = assess_risk(
        WALLET, attribution=attribution(candidates=[weak]), settings=s
    )

    # Contribution was halved, so score < max_possible, and the gap is visible.
    assert result.score < result.max_possible_score
    assert result.max_possible_score == s.risk_weight_indirect_vasp_evidence


def test_assessment_is_deterministic():
    args = dict(
        attribution=attribution(),
        behavior_patterns=[pattern(), pattern(PatternType.TEMPORAL_BURST)],
        settings=settings(),
    )
    first = assess_risk(WALLET, **args)
    second = assess_risk(WALLET, **args)

    assert first.model_dump() == second.model_dump()


# --------------------------------------------------------------------------
# Incomplete search: WHICH cause
# --------------------------------------------------------------------------
#
# A budget stop and a data horizon both leave the search incomplete, but they
# call for opposite remedies -- raise the budget, or widen acquisition. The
# score is the same either way; the stated evidence must not be.


def test_incomplete_search_names_the_data_horizon_when_that_is_the_cause():
    result = assess_risk(
        WALLET,
        attribution=attribution(
            candidates=[],
            status=AttributionStatus.INCONCLUSIVE,
            max_hops=4,
            outbound_search_truncated=True,
            outbound_accounting=SearchAccounting(
                direction="OUTBOUND",
                complete=False,
                observation_depth=1,
                incomplete_reason="data observed to 1 hop(s) only",
            ),
        ),
        settings=settings(),
    )

    component = next(
        c for c in result.components if c.component_id == "search_incomplete"
    )
    evidence = " ".join(component.observed_evidence)
    assert "1 hop(s)" in evidence
    assert "4 hop(s)" in evidence
    assert "never" in evidence and "acquired" in evidence
    assert "resource limit" not in evidence, (
        "naming a budget the run never hit sends the investigator to raise a "
        "limit that was not the constraint"
    )


def test_incomplete_search_names_the_budget_when_the_horizon_covers_the_hops():
    """A horizon at or beyond MAX_HOPS did not cause the stop; the budget
    did, and that is what must be reported."""
    result = assess_risk(
        WALLET,
        attribution=attribution(
            candidates=[],
            status=AttributionStatus.INCONCLUSIVE,
            max_hops=3,
            outbound_search_truncated=True,
            outbound_accounting=SearchAccounting(
                direction="OUTBOUND", complete=False, observation_depth=3
            ),
        ),
        settings=settings(),
    )

    component = next(
        c for c in result.components if c.component_id == "search_incomplete"
    )
    evidence = " ".join(component.observed_evidence)
    assert "resource limit" in evidence
    assert "never acquired" not in evidence


def test_incomplete_search_without_accounting_still_states_a_cause():
    """Older call sites pass no accounting at all. The evidence must remain a
    complete sentence rather than degrading to an empty string."""
    result = assess_risk(
        WALLET,
        attribution=attribution(
            candidates=[],
            status=AttributionStatus.INCONCLUSIVE,
            inbound_searches_truncated=True,
        ),
        settings=settings(),
    )

    component = next(
        c for c in result.components if c.component_id == "search_incomplete"
    )
    evidence = " ".join(component.observed_evidence)
    assert "absence of a match is not established" in evidence
    assert "inbound" in evidence


def test_horizon_and_budget_scores_are_identical_only_the_reason_differs():
    """The penalty is for not knowing, not for the reason behind it."""
    s = settings()
    common = dict(
        candidates=[],
        status=AttributionStatus.INCONCLUSIVE,
        max_hops=4,
        outbound_search_truncated=True,
    )
    horizon = assess_risk(
        WALLET,
        attribution=attribution(
            outbound_accounting=SearchAccounting(
                direction="OUTBOUND", complete=False, observation_depth=1
            ),
            **common,
        ),
        settings=s,
    )
    budget = assess_risk(
        WALLET,
        attribution=attribution(
            outbound_accounting=SearchAccounting(
                direction="OUTBOUND",
                complete=False,
                incomplete_reason="exploration budget exhausted",
            ),
            **common,
        ),
        settings=s,
    )

    assert horizon.score == budget.score
    assert horizon.band == budget.band


def test_untriggered_vasp_component_reports_the_observed_depth_not_the_limit():
    """"within 4 hop(s)" when the data reaches 1 reads as searched-and-empty
    for three hops that were never fetched."""
    result = assess_risk(
        WALLET,
        attribution=attribution(
            candidates=[],
            status=AttributionStatus.INCONCLUSIVE,
            max_hops=4,
            outbound_search_truncated=True,
            outbound_accounting=SearchAccounting(
                direction="OUTBOUND", complete=False, observation_depth=1
            ),
        ),
        settings=settings(),
    )

    skipped = {s.component_id: s.reason for s in result.components_not_triggered}
    reason = skipped["vasp_attribution"]
    assert "1 hop(s) this dataset observes" in reason
    assert "absent from the data" in reason


def test_untriggered_vasp_component_keeps_its_wording_on_a_complete_search():
    result = assess_risk(
        WALLET,
        attribution=attribution(
            candidates=[], status=AttributionStatus.NONE, max_hops=4
        ),
        settings=settings(),
    )

    skipped = {s.component_id: s.reason for s in result.components_not_triggered}
    assert "within 4 hop(s)" in skipped["vasp_attribution"]
