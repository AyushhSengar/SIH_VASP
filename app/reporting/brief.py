"""
THE BRIEF — the at-a-glance projection of an investigation.

The compact report answers "show me everything that matters" in ten blocks.
This one answers a narrower question: *what did we find, and can I read it in
five seconds?* Five things, nothing else:

    1. which wallet, on which chain
    2. the VASP match: yes / no / inconclusive, with the operator, the
       direction and the hop distance
    3. the single strongest evidence path — addresses and transaction hashes,
       no amounts, no timestamps, no plausibility prose
    4. a two-to-three line risk and behaviour summary
    5. the ML verdict on one line, with its disclaimer on the next

WHAT THIS MODULE MAY NOT DO
--------------------------------------------------------------------------
Compute anything. Every field is read from `InvestigationReport`, which the
pipeline already produced; `build_brief` selects and shortens, and that is
all. If a value is absent it prints `N/A` — a brief is the most tempting
place to fill a gap with a plausible-looking default, and the shortest
output is exactly where a fabricated number would be least likely to be
questioned.

It also may not upgrade a finding. `INCONCLUSIVE` stays inconclusive: the
brevity of this view is not a licence to round a bounded search up to a
clean negative. `matched` is deliberately a THREE-state field
(True/False/None) for that reason — a boolean would have forced
inconclusive into one of the two answers it is not.

`InvestigationBrief` is a pydantic model rather than a string so the HTTP API
and the terminal share one projection: the frontend renders the same fields
`--brief` prints, and neither can drift into showing something the other
does not.
"""

from __future__ import annotations

import sys
from typing import Optional, TextIO

from pydantic import BaseModel, ConfigDict

from app.investigation.pipeline import InvestigationReport
from app.reporting.compact import (
    NA,
    _index_or_none,
    _na,
    _num,
    _strongest_evidence,
)
from app.reporting.terminal import _v, _wrap as _wrap_text, configure_stdout, to_ascii

WIDTH = 78
_RULE = "=" * WIDTH
_THIN = "-" * WIDTH

#: How many behavioural findings the one-line summary names before the count
#: carries the rest. Three pattern names fit a 78-character line.
_FINDINGS_INLINE = 3

#: The fixed disclaimer attached to every ML verdict, in every state. Not
#: conditional on the verdict being interesting: an advisory signal is
#: advisory when it says "outlier" and equally advisory when it says
#: "unremarkable", and a disclaimer that appears only next to bad news
#: teaches a reader to treat its absence as a clean bill of health.
ML_DISCLAIMER = (
    "ADVISORY ONLY - a statistical signal, not evidence. It never overrides "
    "the VASP attribution evidence above and never indicates wrongdoing."
)


class BriefVASPMatch(BaseModel):
    """The attribution answer, reduced to what a first read needs."""

    #: MATCH_FOUND | NONE | INCONCLUSIVE, verbatim from the attribution.
    status: str
    #: True on a match, False on a *complete* negative, None when the search
    #: was bounded before it could answer. Never collapsed to a boolean.
    matched: Optional[bool] = None
    vasp_name: Optional[str] = None
    direction: Optional[str] = None
    hop_distance: Optional[int] = None
    matched_address: Optional[str] = None
    source_type: Optional[str] = None
    #: The strongest path's addresses in order, and the hash of the edge
    #: leaving each one. `evidence_tx_hashes[i]` is the transfer from
    #: `evidence_path[i]` to `evidence_path[i + 1]`, so it is one shorter than
    #: the path; a hop whose hash the evidence did not carry is None rather
    #: than shifting the next hop's hash into its place.
    evidence_path: list[str] = []
    evidence_tx_hashes: list[Optional[str]] = []
    #: Why this is inconclusive, or what a match does and does not mean.
    note: Optional[str] = None
    #: Set when the investigated wallet IS a dataset address itself, which is
    #: an identity match rather than a traced connection.
    wallet_is_known_vasp: Optional[str] = None


class BriefML(BaseModel):
    """The ML verdict on one line, plus the disclaimer that always follows."""

    model_config = ConfigDict(protected_namespaces=())

    #: SUPERVISED | UNSUPERVISED | UNAVAILABLE | DISABLED
    approach: str
    #: The whole verdict, already written as one human sentence.
    verdict: str
    disclaimer: str = ML_DISCLAIMER
    #: Present only when a real model actually scored this wallet, so a
    #: reader can tell an advisory backed by held-out metrics from one that
    #: is only an outlier percentile.
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    held_out_f1: Optional[float] = None
    held_out_accuracy: Optional[float] = None


class InvestigationBrief(BaseModel):
    """The complete brief. Renderer-agnostic, same as the full report."""

    wallet: str
    chain: str
    investigation_id: str
    #: REAL | CACHED_REAL_DATA | ... — carried because a brief that omitted
    #: it would look identical whether the data was live or from a file.
    data_mode: str
    duration_seconds: float

    vasp_match: BriefVASPMatch
    #: Two or three lines. A list rather than one blob so a UI can lay them
    #: out without parsing.
    risk_summary: list[str] = []
    ml: BriefML
    #: Only the warnings that change how the result should be read. The full
    #: set stays in the report and in --json.
    warnings: list[str] = []


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------


def _vasp_match(report: InvestigationReport) -> BriefVASPMatch:
    attribution = report.attribution
    if attribution is None:
        return BriefVASPMatch(
            status="NOT_RUN",
            note="Attribution did not run, so there is no match to report.",
        )

    status = _v(attribution.status)
    brief = BriefVASPMatch(
        status=status,
        matched=True if status == "MATCH_FOUND" else (False if status == "NONE" else None),
    )

    if attribution.exact_identity_match is not None:
        brief.wallet_is_known_vasp = _na(attribution.exact_identity_match.vasp_name)

    candidate, evidence, label = _strongest_evidence(attribution)
    if candidate is not None and evidence is not None:
        brief.vasp_name = candidate.vasp_name
        brief.direction = _v(candidate.direction)
        brief.hop_distance = evidence.hop_distance
        brief.matched_address = candidate.matched_address
        brief.source_type = _v(candidate.source_type)
        brief.evidence_path = list(evidence.path_addresses)
        brief.evidence_tx_hashes = [
            _index_or_none(evidence.tx_hashes, index)
            for index in range(max(len(evidence.path_addresses) - 1, 0))
        ]
        if evidence.hop_distance > 1:
            brief.note = (
                f"{evidence.hop_distance} hops via {label}. Consecutive "
                "transfers are SUPPORTING EVIDENCE of a route, not proof that "
                "the same funds moved end to end."
            )
        else:
            brief.note = f"Direct transfer ({label})."
    elif status == "MATCH_FOUND" and attribution.candidates:
        # A candidate exists but carries no traced path — an identity match,
        # or evidence recorded without a route. Say so instead of printing an
        # empty path as though the search returned nothing.
        first = attribution.candidates[0]
        brief.vasp_name = first.vasp_name
        brief.direction = _v(first.direction)
        brief.matched_address = first.matched_address
        brief.source_type = _v(first.source_type)
        brief.note = "Matched without a traced multi-hop path."

    if status == "INCONCLUSIVE":
        brief.note = _inconclusive_reason(attribution)
    elif status == "NONE":
        brief.note = (
            f"No route to any of the {attribution.seed_address_count} dataset "
            f"address(es) within {attribution.max_hops} hop(s). A negative "
            "against this dataset, not against all VASPs."
        )

    return brief


def _inconclusive_reason(attribution) -> str:
    """The reason a bounded search could not answer, in one line.

    Two causes exist and they call for opposite remedies — a budget stop
    means re-run with a larger budget, a data horizon means acquire more data
    — so the brief names which one bit rather than saying "incomplete".
    """
    reasons = [
        accounting.incomplete_reason
        for accounting in (
            attribution.outbound_accounting,
            attribution.inbound_accounting,
        )
        if accounting is not None and accounting.incomplete_reason
    ]
    if reasons:
        return f"SEARCH INCOMPLETE - {reasons[0]}"
    return (
        "The search was cut short by a resource limit before the configured "
        "depth was covered, so absence of a match is not evidence of absence."
    )


def _risk_summary(report: InvestigationReport) -> list[str]:
    """Two or three lines: the score, the behaviour, the data caveat.

    Deliberately not the component breakdown — that is the full report's job.
    What a first read needs is the band, whether anything was actually
    observed, and whether the dataset behind both was complete.
    """
    lines: list[str] = []

    risk = report.risk
    if risk is None:
        lines.append(f"RISK: {NA} (risk scoring did not run).")
    else:
        # Deliberately NOT "score / max_possible_score": that pair measures how
        # much discounting happened, not progress towards a ceiling, and a
        # reader takes "X of Y" as a maxed-out risk. There is no finite maximum
        # to quote, so none is quoted.
        lines.append(
            f"RISK: {_v(risk.band)} (score {_na(risk.score)}, "
            f"MEDIUM at {_na(risk.band_medium_threshold)} / "
            f"HIGH at {_na(risk.band_high_threshold)}) from "
            f"{len(risk.components)} triggered indicator(s) of "
            f"{len(risk.components) + len(risk.components_not_triggered)} checked."
        )

    patterns = report.behavior_patterns
    if not patterns:
        lines.append("BEHAVIOUR: no behavioural indicator crossed its threshold.")
    else:
        names = [_v(p.pattern_type) for p in patterns[:_FINDINGS_INLINE]]
        more = len(patterns) - len(names)
        listed = ", ".join(names) + (f" (+{more} more)" if more > 0 else "")
        lines.append(
            f"BEHAVIOUR: {len(patterns)} indicator(s) - {listed}. "
            "Investigative indicators requiring further verification, not "
            "findings of wrongdoing."
        )

    if not report.provenance.data_complete:
        reason = (
            report.provenance.incompleteness_reasons[0]
            if report.provenance.incompleteness_reasons
            else "the dataset is incomplete"
        )
        lines.append(f"DATA: INCOMPLETE - {reason}")

    return lines


def _ml_brief(report: InvestigationReport) -> BriefML:
    """The ML verdict as one sentence, in whichever of the four states it is.

    A refusal is a verdict here, not an omission: "no model was trained,
    because X" is the honest one-liner, and it is what goes on the line where
    a reader expects a prediction. The alternative — leaving the line blank or
    printing a placeholder number — is the exact failure this project refuses.
    """
    ml = report.ml
    if ml is None:
        return BriefML(
            approach="UNAVAILABLE",
            verdict="ML did not run for this investigation.",
        )

    approach = _v(ml.approach)

    if approach == "DISABLED":
        return BriefML(
            approach=approach,
            verdict="ML was disabled for this run (--no-ml). The blockchain "
            "evidence above is unaffected.",
        )

    if approach == "SUPERVISED" and ml.prediction is not None and ml.prediction.available:
        p = ml.prediction
        metrics = ml.training.test_metrics if ml.training is not None else None
        probability = (
            f" (p={p.probability})" if p.probability is not None else ""
        )
        return BriefML(
            approach=approach,
            verdict=(
                f"{_na(p.task)} = {_na(p.predicted_class)}{probability}, from a "
                f"{_na(p.model_name)} trained on real labels."
            ),
            model_name=p.model_name,
            model_version=p.model_version,
            held_out_f1=metrics.f1 if metrics is not None else None,
            held_out_accuracy=metrics.accuracy if metrics is not None else None,
        )

    if approach == "UNSUPERVISED" and ml.outlier is not None and ml.outlier.available:
        o = ml.outlier
        verdict = "unusual" if o.flagged_as_outlier else "unremarkable"
        return BriefML(
            approach=approach,
            verdict=(
                f"Behaviour is {verdict} for this graph - "
                f"{_na(o.percentile_within_population)} percentile of "
                f"{_num(o.population_size)} address(es) by {_na(o.method)}. No "
                "labelled model was trainable, so no class was predicted."
            ),
        )

    reason = None
    if ml.prediction is not None and ml.prediction.unavailable_reason:
        reason = ml.prediction.unavailable_reason
    elif ml.outlier is not None and ml.outlier.unavailable_reason:
        reason = ml.outlier.unavailable_reason
    elif ml.rationale:
        reason = ml.rationale[0]
    return BriefML(
        approach=approach,
        verdict=(
            "No ML verdict: "
            + (reason or "neither a supervised nor an unsupervised model was possible.")
        ),
    )


def build_brief(report: InvestigationReport) -> InvestigationBrief:
    """Projects a full report onto the brief. Computes nothing."""
    return InvestigationBrief(
        wallet=report.wallet,
        chain=report.chain,
        investigation_id=report.investigation_id,
        data_mode=report.provenance.data_mode.value,
        duration_seconds=report.duration_seconds,
        vasp_match=_vasp_match(report),
        risk_summary=_risk_summary(report),
        ml=_ml_brief(report),
        warnings=list(report.warnings),
    )


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _wrap(text: str, indent: str = "  ") -> list[str]:
    """Word-wraps one line of prose at the brief's rule width.

    Delegates to the full renderer's wrapper so all three reports break lines
    the same way — including its rule that a long word is never split, because
    an address or hash chopped across two lines cannot be copied or searched
    for, and this view is mostly addresses and hashes.
    """
    return _wrap_text(text, indent=indent, width=WIDTH)


def render_brief(brief: InvestigationBrief) -> str:
    """Renders the brief as one ASCII-safe string.

    ASCII-folded like every other renderer here: the Windows console falls
    back to cp1252 and would print a Unicode arrow as `?` in exactly the
    place a reader needs to see direction.
    """
    lines: list[str] = [
        _RULE,
        "  WALLET INVESTIGATION -- BRIEF",
        _RULE,
        f"  WALLET:    {brief.wallet}",
        f"  CHAIN:     {brief.chain}    DATA MODE: {brief.data_mode}",
        _THIN,
    ]

    match = brief.vasp_match
    answer = {True: "YES", False: "NO", None: "INCONCLUSIVE"}[match.matched]
    lines.append(f"  VASP MATCH: {answer}  ({match.status})")
    if match.wallet_is_known_vasp:
        lines.append(
            f"  THE WALLET ITSELF IS A DATASET ADDRESS: {match.wallet_is_known_vasp}"
        )
    if match.vasp_name:
        lines.append(f"  ENTITY:    {_na(match.vasp_name)}")
        lines.append(
            f"  DIRECTION: {_na(match.direction)}    HOPS: {_num(match.hop_distance)}"
            f"    SOURCE: {_na(match.source_type)}"
        )
    if match.note:
        lines.extend(_wrap(match.note))

    if match.evidence_path:
        lines.append("")
        lines.append("  EVIDENCE PATH")
        for index, address in enumerate(match.evidence_path):
            lines.append(f"    {index}. {address}")
            tx_hash = _index_or_none(match.evidence_tx_hashes, index)
            if index < len(match.evidence_path) - 1:
                lines.append(f"       tx {tx_hash or NA}")

    lines.append(_THIN)
    for line in brief.risk_summary:
        lines.extend(_wrap(line))

    lines.append(_THIN)
    lines.extend(_wrap(f"ML: {brief.ml.verdict}"))
    if brief.ml.model_name:
        lines.append(
            f"  MODEL: {_na(brief.ml.model_name)} {_na(brief.ml.model_version)}"
            f"  held-out F1={_na(brief.ml.held_out_f1)} "
            f"accuracy={_na(brief.ml.held_out_accuracy)}"
        )
    lines.extend(_wrap(brief.ml.disclaimer))

    if brief.warnings:
        lines.append(_THIN)
        for warning in brief.warnings:
            lines.extend(_wrap(f"WARNING: {warning}"))

    lines.append(_RULE)
    lines.append(
        f"  Full detail: drop --brief   Everything: --full-report   JSON: --json"
    )
    lines.append(_RULE)
    return to_ascii("\n".join(lines))


def print_brief_report(
    report: InvestigationReport, stream: Optional[TextIO] = None
) -> None:
    """Builds and prints the brief. Renders to a stream, forcing UTF-8 first."""
    target = stream if stream is not None else sys.stdout
    configure_stdout(target)
    print(render_brief(build_brief(report)), file=target)


__all__ = [
    "BriefML",
    "BriefVASPMatch",
    "InvestigationBrief",
    "ML_DISCLAIMER",
    "build_brief",
    "print_brief_report",
    "render_brief",
]
