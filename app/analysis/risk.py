"""
Investigative priority scoring — transparent by construction.

WHAT THIS IS NOT (do not remove)
--------------------------------------------------------------------------
This is NOT a probability of criminality, a suspicion score, or a
compliance rating. It is an *investigative priority* signal: how much
follow-up material this wallet has produced. A wallet can score highly for
entirely lawful reasons — sending to an exchange, being busy, holding many
assets. Nothing in this module may be reported as evidence of wrongdoing,
and `RiskAssessment.non_claims` states that in the output itself so the
disclaimer travels with the number.

WHY THERE IS A NUMBER AT ALL
--------------------------------------------------------------------------
The requirement is explicit that a bare "Risk = 87" is unacceptable. The
answer here is not to omit the number but to make it *reconstructible*:

  score == sum(component.contribution for component in components)

exactly, with every component naming the indicator that produced it, the
observed evidence, the configuration field its weight came from, and why it
raises priority. A reader can add the components up by hand and get the
same total. There is no hidden term, no normalisation, no curve, and no
model — deriving the score is arithmetic over stated facts.

Components that were CONSIDERED AND DID NOT FIRE are also reported
(`components_not_triggered`), because a score is only auditable if you can
see what was weighed and found absent, not just what was found.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel

from app.attribution.bidirectional_models import (
    BidirectionalAttributionResult,
    ConnectionDirection,
)
from app.attribution.models import AttributionStatus
from app.behavior.models import BehaviorPattern, IndicatorClass
from app.core.config import Settings, get_settings


class EvidenceClass(str, Enum):
    """How directly a component's evidence bears on the conclusion.

    Mirrors the project-wide evidence vocabulary so one word means one thing
    everywhere: DIRECT (a one-hop, address-level observation), INDIRECT (a
    multi-hop traced route), SUPPORTING (corroborates something else but
    cannot stand alone), CONTEXTUAL (background with known confounders),
    INCONCLUSIVE (the search could not settle the question).
    """

    DIRECT = "DIRECT"
    INDIRECT = "INDIRECT"
    SUPPORTING = "SUPPORTING"
    CONTEXTUAL = "CONTEXTUAL"
    INCONCLUSIVE = "INCONCLUSIVE"


class PriorityBand(str, Enum):
    """Coarse bucket for the score. Boundaries are configuration values and
    are always printed next to the band, so the band is never a bare label."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskComponent(BaseModel):
    """One fully-explained contribution to the score.

    Every field is required to be meaningful: a component with no
    `observed_evidence` or no `reason` would be exactly the opaque number
    this module exists to prevent.
    """

    component_id: str  # stable machine identifier
    indicator: str  # the finding's own name (enum value, direction, etc.)
    evidence_class: EvidenceClass
    observed_evidence: list[str]  # concrete, quotable facts
    weight: float  # the configured maximum for this kind of component
    contribution: float  # points actually added to the score
    weight_setting: str  # the Settings field `weight` was read from
    reason: str  # why this raises investigative priority

    observed_metric: Optional[str] = None
    observed_value: Optional[float | int | str] = None
    threshold: Optional[float | int | str] = None
    threshold_setting: Optional[str] = None

    related_addresses: list[str] = []
    relevant_tx_hashes: list[str] = []


class SkippedComponent(BaseModel):
    """A component that was evaluated and contributed nothing, plus why."""

    component_id: str
    reason: str


class RiskAssessment(BaseModel):
    wallet: str

    score: float
    # Summed WEIGHT of the components that fired -- NOT a theoretical maximum
    # and NOT a denominator for a percentage. It equals `score` unless some
    # component was discounted (a weak-plausibility path contributes half its
    # weight), so the pair exposes how much discounting happened and nothing
    # more. There is no finite theoretical maximum to report: per-counterparty
    # indicators mean the component count is unbounded, so any fixed ceiling
    # would be an invented number. Never render this as "X of a possible Y" --
    # a reader takes that as a maxed-out risk, which it is not.
    max_possible_score: float
    band: PriorityBand
    band_medium_threshold: float
    band_high_threshold: float

    components: list[RiskComponent]
    components_not_triggered: list[SkippedComponent] = []

    data_complete: bool = True
    data_completeness_note: Optional[str] = None

    non_claims: list[str] = []

    @property
    def arithmetic_checks_out(self) -> bool:
        """The score IS the sum of the contributions. Exposed so a test — and
        a reader — can assert it rather than trust it."""
        return abs(self.score - sum(c.contribution for c in self.components)) < 1e-9


_INDICATOR_WEIGHT_FIELD = {
    IndicatorClass.INVESTIGATIVE_INDICATOR: (
        "risk_weight_investigative_indicator",
        EvidenceClass.SUPPORTING,
    ),
    IndicatorClass.SUPPORTING_EVIDENCE: (
        "risk_weight_supporting_indicator",
        EvidenceClass.SUPPORTING,
    ),
    IndicatorClass.CONTEXTUAL: (
        "risk_weight_contextual_indicator",
        EvidenceClass.CONTEXTUAL,
    ),
    IndicatorClass.REQUIRES_FURTHER_VERIFICATION: (
        "risk_weight_supporting_indicator",
        EvidenceClass.SUPPORTING,
    ),
}

_NON_CLAIMS = [
    "This score measures how much follow-up material the investigation "
    "produced. It is NOT a probability of criminality, fraud, or sanctions "
    "exposure, and it must not be reported as one.",
    "Reaching a known VASP is normal, lawful behaviour for the overwhelming "
    "majority of wallets. It contributes here because it gives an "
    "investigator a real-world counterparty to serve a request on — not "
    "because it is suspicious.",
    "Behavioural indicators are observations against configured thresholds. "
    "They describe activity; they do not establish intent.",
    "A traced graph path is a chain of transaction relationships, not proof "
    "that the same units of value moved end to end.",
    "No component of this score comes from a machine-learning prediction. "
    "The score is arithmetic over observed on-chain facts only.",
]


def _incomplete_search_evidence(
    attribution: BidirectionalAttributionResult, truncated: list[str]
) -> str:
    """States WHY the search was incomplete, in the search's own terms.

    Two causes are possible and they point the reader in opposite
    directions: a budget stop means re-run with a larger budget, whereas a
    data horizon means the deeper hops were never acquired and no budget
    will produce them. Naming the wrong one wastes an investigator's time.
    """
    directions = " and ".join(truncated) or "traversal"
    depth = next(
        (
            acc.observation_depth
            for acc in (
                attribution.outbound_accounting,
                attribution.inbound_accounting,
            )
            if acc is not None
            and acc.observation_depth is not None
            and acc.observation_depth < attribution.max_hops
        ),
        None,
    )
    if depth is not None:
        return (
            f"The attribution search covered the {depth} hop(s) this dataset "
            f"observes, but was configured for {attribution.max_hops} hop(s) "
            f"in the {directions} direction. Hops beyond {depth} were never "
            "acquired, so the absence of a match is not established for them."
        )
    return (
        "The attribution search hit a configured resource limit "
        f"in the {directions} "
        "direction, so the absence of a match is not established."
    )


def _vasp_components(
    attribution: BidirectionalAttributionResult, settings: Settings
) -> tuple[list[RiskComponent], list[SkippedComponent]]:
    components: list[RiskComponent] = []
    skipped: list[SkippedComponent] = []

    if attribution.exact_identity_match is not None:
        match = attribution.exact_identity_match
        weight = settings.risk_weight_direct_vasp_evidence
        components.append(
            RiskComponent(
                component_id="vasp_exact_identity",
                indicator=f"EXACT_ADDRESS_MATCH:{match.vasp_name}",
                evidence_class=EvidenceClass.DIRECT,
                observed_evidence=[
                    f"The investigated address {match.matched_address} is itself "
                    f"present in the known-VASP dataset as {match.vasp_name} "
                    f"({match.entity_type}).",
                    f"Dataset provenance: {match.source_type.value}"
                    + (
                        f", verification status {match.verification_status}"
                        if match.verification_status
                        else ""
                    ),
                ],
                weight=weight,
                contribution=weight,
                weight_setting="risk_weight_direct_vasp_evidence",
                reason=(
                    "An exact address match is the strongest address-level "
                    "evidence available and identifies the operator directly, "
                    "subject only to the dataset entry's own provenance."
                ),
                related_addresses=[match.matched_address],
            )
        )

    # One component per distinct VASP, graded by its strongest direction. A
    # VASP reached both ways is not counted twice — the double connection is
    # reported inside the component's evidence instead, so the score cannot be
    # inflated by describing one relationship in two directions.
    direct_directions = {
        ConnectionDirection.DIRECT_OUTBOUND,
        ConnectionDirection.DIRECT_INBOUND,
        ConnectionDirection.BIDIRECTIONAL,
    }
    for candidate in attribution.candidates:
        is_direct = (
            candidate.direction in direct_directions
            and candidate.strongest_hop_distance == 1
        )
        weight_setting = (
            "risk_weight_direct_vasp_evidence"
            if is_direct
            else "risk_weight_indirect_vasp_evidence"
        )
        weight = getattr(settings, weight_setting)
        evidence_class = EvidenceClass.DIRECT if is_direct else EvidenceClass.INDIRECT

        evidence: list[str] = [
            f"{candidate.vasp_name} ({candidate.matched_address}) is connected "
            f"to the wallet as {candidate.direction.value} at "
            f"{candidate.strongest_hop_distance} hop(s).",
            f"Dataset provenance: {candidate.source_type.value}"
            + (
                f", verification status {candidate.verification_status}"
                if candidate.verification_status
                else ""
            ),
        ]
        tx_hashes: list[str] = []
        contribution = weight
        for label, directional in (
            ("outbound", candidate.outbound_evidence),
            ("inbound", candidate.inbound_evidence),
        ):
            if directional is None:
                continue
            tx_hashes.extend(directional.tx_hashes)
            grade = (
                directional.plausibility.grade.value
                if directional.plausibility is not None
                else "UNGRADED"
            )
            evidence.append(
                f"{label}: {directional.hop_distance} hop(s), path plausibility "
                f"{grade}, {directional.alternative_path_count} alternative "
                f"route(s) also existed."
            )
            # A route the plausibility grader will not read as a fund flow
            # cannot carry a fund-flow component's full weight. Halved rather
            # than dropped: the addresses ARE transactionally connected, which
            # is itself a lead worth an investigator's time.
            if (
                directional.plausibility is not None
                and not directional.plausibility.supports_fund_flow_narrative
            ):
                contribution = min(contribution, weight / 2)
                evidence.append(
                    f"{label} contribution halved: the observable evidence does "
                    f"not support reading this route as a movement of funds "
                    f"({', '.join(directional.plausibility.concern_types) or 'see grade'})."
                )

        components.append(
            RiskComponent(
                component_id=f"vasp_{candidate.matched_address}",
                indicator=f"{candidate.direction.value}:{candidate.vasp_name}",
                evidence_class=evidence_class,
                observed_evidence=evidence,
                weight=weight,
                contribution=round(contribution, 4),
                weight_setting=weight_setting,
                reason=(
                    "A known VASP address is reachable from the wallet by "
                    "directed, timestamped transfers, which gives the "
                    "investigation a regulated counterparty that can be asked "
                    "for off-chain account records."
                ),
                observed_metric="hop_distance",
                observed_value=candidate.strongest_hop_distance,
                related_addresses=[candidate.matched_address],
                relevant_tx_hashes=sorted(set(tx_hashes))[:10],
            )
        )

    if not attribution.candidates and attribution.exact_identity_match is None:
        # State the depth actually covered, not the depth requested. Saying
        # "within 4 hop(s)" when the data only reaches 1 reads as a searched
        # -and-empty result for hops that were never fetched.
        observed = next(
            (
                acc.observation_depth
                for acc in (
                    attribution.outbound_accounting,
                    attribution.inbound_accounting,
                )
                if acc is not None
                and acc.observation_depth is not None
                and acc.observation_depth < attribution.max_hops
            ),
            None,
        )
        if observed is not None:
            reason = (
                "No known VASP address was reached by a directed path within "
                f"the {observed} hop(s) this dataset observes (attribution "
                f"status: {attribution.status.value}). The configured limit "
                f"was {attribution.max_hops} hop(s), but hops beyond "
                f"{observed} are absent from the data, so nothing was "
                "concluded about them either way."
            )
        else:
            reason = (
                "No known VASP address was reached by a directed path "
                f"within {attribution.max_hops} hop(s) "
                f"(attribution status: {attribution.status.value})."
            )
        skipped.append(
            SkippedComponent(component_id="vasp_attribution", reason=reason)
        )

    # An unfinished search raises priority. Treating "we could not finish
    # looking" as "nothing to find" is the specific mistake this component
    # exists to prevent.
    if attribution.status == AttributionStatus.INCONCLUSIVE:
        weight = settings.risk_weight_incomplete_search
        truncated = [
            name
            for name, hit in (
                ("outbound", attribution.outbound_search_truncated),
                ("inbound", attribution.inbound_searches_truncated),
            )
            if hit
        ]
        components.append(
            RiskComponent(
                component_id="search_incomplete",
                indicator="SEARCH_INCOMPLETE",
                evidence_class=EvidenceClass.INCONCLUSIVE,
                observed_evidence=[
                    _incomplete_search_evidence(attribution, truncated),
                ],
                weight=weight,
                contribution=weight,
                weight_setting="risk_weight_incomplete_search",
                reason=(
                    "An incomplete search leaves the question open. Priority "
                    "goes up rather than down, because unexamined search space "
                    "is not the same as examined-and-clear."
                ),
            )
        )
    else:
        skipped.append(
            SkippedComponent(
                component_id="search_incomplete",
                reason=(
                    "The attribution search completed over the data available "
                    "to it, within its configured budgets, so no "
                    "incompleteness penalty applies."
                ),
            )
        )

    return components, skipped


def assess_risk(
    wallet: str,
    attribution: Optional[BidirectionalAttributionResult] = None,
    behavior_patterns: Optional[list[BehaviorPattern]] = None,
    settings: Optional[Settings] = None,
    data_complete: bool = True,
    data_completeness_note: Optional[str] = None,
) -> RiskAssessment:
    """Builds a fully-decomposed investigative priority assessment.

    Every argument is optional so the function can also describe a wallet for
    which a stage was skipped (e.g. `--no-ml`, or attribution not run) without
    inventing inputs. A missing stage becomes a SkippedComponent with a stated
    reason, never a silent zero.
    """
    settings = settings or get_settings()
    components: list[RiskComponent] = []
    skipped: list[SkippedComponent] = []

    if attribution is not None:
        vasp_components, vasp_skipped = _vasp_components(attribution, settings)
        components.extend(vasp_components)
        skipped.extend(vasp_skipped)
    else:
        skipped.append(
            SkippedComponent(
                component_id="vasp_attribution",
                reason="VASP attribution did not run for this investigation.",
            )
        )

    patterns = behavior_patterns or []
    for pattern in patterns:
        weight_setting, evidence_class = _INDICATOR_WEIGHT_FIELD[
            pattern.classification
        ]
        weight = getattr(settings, weight_setting)
        components.append(
            RiskComponent(
                component_id=f"behavior_{pattern.pattern_type.value.lower()}"
                + (
                    f"_{pattern.related_addresses[0]}"
                    if len(pattern.related_addresses) == 1
                    else ""
                ),
                indicator=pattern.indicator_name,
                evidence_class=evidence_class,
                observed_evidence=list(pattern.evidence),
                weight=weight,
                contribution=weight,
                weight_setting=weight_setting,
                reason=(
                    f"{pattern.indicator_name} crossed its configured threshold "
                    f"({pattern.threshold_setting or 'see detector'}). It is "
                    f"classified {pattern.classification.value}, which is what "
                    "determines how much it may contribute."
                ),
                observed_metric=pattern.observed_metric,
                observed_value=pattern.observed_value,
                threshold=pattern.threshold,
                threshold_setting=pattern.threshold_setting,
                related_addresses=list(pattern.related_addresses)[:10],
                relevant_tx_hashes=list(pattern.relevant_tx_hashes),
            )
        )

    if not patterns:
        skipped.append(
            SkippedComponent(
                component_id="behavioral_indicators",
                reason=(
                    "No behavioural indicator crossed its configured threshold "
                    "for this wallet."
                ),
            )
        )

    score = round(sum(c.contribution for c in components), 4)
    max_possible = round(sum(c.weight for c in components), 4)

    if score >= settings.risk_band_high_threshold:
        band = PriorityBand.HIGH
    elif score >= settings.risk_band_medium_threshold:
        band = PriorityBand.MEDIUM
    else:
        band = PriorityBand.LOW

    note = data_completeness_note
    if not data_complete and note is None:
        note = (
            "The underlying transaction dataset is incomplete, so this score "
            "is a lower bound — findings may exist in the unread history."
        )

    return RiskAssessment(
        wallet=wallet,
        score=score,
        max_possible_score=max_possible,
        band=band,
        band_medium_threshold=settings.risk_band_medium_threshold,
        band_high_threshold=settings.risk_band_high_threshold,
        components=components,
        components_not_triggered=skipped,
        data_complete=data_complete,
        data_completeness_note=note,
        non_claims=list(_NON_CLAIMS),
    )
