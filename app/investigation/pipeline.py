"""
INVESTIGATION PIPELINE — the single orchestration path for a wallet
investigation.

`investigate.py` is a thin CLI over this module, and any future API endpoint
should call `run_investigation` too. Nothing here prints: the function returns
a fully-populated `InvestigationReport`, and rendering lives in
`app/reporting/`. That separation is what makes the backend independent of any
frontend and lets the whole pipeline be asserted on in tests without capturing
stdout.

DATA MODE IS PART OF THE RESULT
--------------------------------------------------------------------------
Every report states where its data came from:

    REAL              a live provider fetch in this run
    CACHED REAL DATA  real chain data acquired earlier and reloaded from disk

There is no demo mode. If credentials are missing, acquisition raises instead
of substituting anything, and the caller reports the blocker. Cached real data
is never labelled REAL — it is real, but it is not live, and an investigator
reading a report needs to know which they have.

PIPELINE ORDER
--------------------------------------------------------------------------
    acquire (recursive: the wallet's 3 streams, then each discovered
    counterparty's, hop by hop to the requested depth)
    -> normalize -> validate -> build graph
    -> counterparties -> bidirectional VASP attribution -> entity resolution
    -> behavioural indicators -> temporal/amount analysis -> risk
    -> ML (supervised if labels permit, else unsupervised) -> conclusion
"""

from __future__ import annotations

import platform
import time
from collections import deque
from enum import Enum
# Aliased: bare `chain` means "blockchain" everywhere else in this module.
from itertools import chain as iter_chain
from pathlib import Path
from typing import Any, Callable, Optional

import networkx as nx
from pydantic import BaseModel, ConfigDict

from app.analysis.risk import RiskAssessment, assess_risk
from app.analysis.temporal import (
    TemporalAmountAnalysis,
    analyze_temporal_and_amounts,
    wallet_incident_edges,
)
from app.attribution.bidirectional import generate_bidirectional_candidates
from app.attribution.bidirectional_models import (
    AttributionStatus,
    BidirectionalAttributionResult,
)
from app.attribution.entities import (
    AttributedEntity,
    Counterparty,
    build_entity_registry,
    identify_counterparties,
    resolve_candidate_entities,
)
from app.attribution.matcher import build_seed_index
from app.attribution.models import SeedSourceType, VASPSeedEntry
from app.attribution.seed_loader import SeedDataError, load_vasp_seed
from app.behavior.models import BehaviorPattern
from app.behavior.detectors import analyze_wallet_behavior
from app.blockchain.base import (
    BlockchainProvider,
    InvalidAddressError,
    UnsupportedChainError,
)
from app.blockchain.chains import (
    CHAIN_VALIDATED_LIVE,
    SUPPORTED_CHAINS,
    normalize_chain_name,
)
from app.blockchain.ingest import (
    AcquisitionResult,
    IngestionError,
    acquire_wallet_transactions,
)
from app.core.config import Settings, get_settings
from app.investigation.acquisition import (
    STOP_ADDRESS_BUDGET,
    STOP_DEPTH_REACHED,
    STOP_NO_NEW_COUNTERPARTIES,
    acquire_multi_hop,
    stop_reason_text,
)
from app.graph.builder import (
    GraphSummary,
    build_graph,
    load_graph,
    load_transfers_from_fixture,
    save_graph,
    summarize_graph,
)
from app.ml.negative_labels import (
    NegativeReferenceError,
    NonVASPEntry,
    load_non_vasp_reference,
)
from app.ml.real_labels import LabelingOutcome, derive_account_type_labels, derive_vasp_labels
from app.ml.real_predictor import PredictionResult, predict
from app.ml.real_training import TrainingOutcome, train_real_model
from app.ml.unsupervised import OutlierAssessment, assess_address
from app.models import NormalizedTransfer
from app.normalization.transactions import NormalizationReport, normalize_all, validate_transfers

ADDRESS_LENGTH = 42


class DataMode(str, Enum):
    """Where this report's blockchain data came from. Never DEMO."""

    REAL = "REAL"
    CACHED_REAL_DATA = "CACHED REAL DATA"


class PipelineError(Exception):
    """A condition that stops the investigation, with a caller-safe message.

    Raised instead of returning a partial report when there is genuinely
    nothing to analyse — a malformed address, a missing API key, an empty
    dataset. The message is written for an investigator, not a developer, and
    never contains a credential.
    """


class MLSection(BaseModel):
    """The ML part of the report, in every possible state.

    `approach` is the load-bearing field: SUPERVISED means a real trained model
    scored this wallet; UNSUPERVISED means the labels were insufficient and an
    outlier model ran instead; UNAVAILABLE means neither was possible. A reader
    must never have to infer which of the three they are looking at.
    """

    model_config = ConfigDict(protected_namespaces=())

    approach: str  # SUPERVISED | UNSUPERVISED | UNAVAILABLE | DISABLED
    rationale: list[str] = []

    account_type_labels: Optional[LabelingOutcome] = None
    vasp_labels: Optional[LabelingOutcome] = None
    training: Optional[TrainingOutcome] = None
    prediction: Optional[PredictionResult] = None
    outlier: Optional[OutlierAssessment] = None

    limitations: list[str] = []


class DataProvenance(BaseModel):
    """Exactly how this run got its data, down to the file or the streams."""

    data_mode: DataMode
    provider: Optional[str] = None
    source_description: str
    graph_path: Optional[str] = None
    transfers_path: Optional[str] = None
    streams: list[str] = []
    cache_stats: dict[str, int] = {}
    data_complete: bool = True
    incompleteness_reasons: list[str] = []
    acquired_at_utc: Optional[str] = None
    # Hop radius around the wallet for which this dataset's edges are complete.
    # See `infer_observation_depth`.
    observation_depth: Optional[int] = None

    # --- Recursive acquisition facts -------------------------------------
    # All optional and defaulted, so the paths that do not acquire live
    # (cached graph, transfers file) leave them None and the renderer prints
    # N/A rather than a zero that would read as "nothing was fetched".
    #
    # These exist because `observation_depth` alone cannot answer the
    # question an investigator actually asks -- "did it really look four hops
    # out, or did it just say four?" -- and a depth number that came from a
    # requested parameter rather than from fetched data would be exactly the
    # kind of claim this build refuses to make.
    #: Addresses whose own transaction streams were actually retrieved.
    addresses_fetched: Optional[int] = None
    #: Addresses acquisition became aware of, expanded or not.
    addresses_discovered: Optional[int] = None
    #: Hop levels that had at least one address fetched.
    hops_expanded: Optional[int] = None
    #: Greatest hop distance at which any address was discovered.
    max_hop_reached: Optional[int] = None
    #: Hop levels the run was asked to expand.
    requested_expansion_hops: Optional[int] = None
    #: DEPTH_REACHED | NO_NEW_COUNTERPARTIES | ADDRESS_BUDGET_REACHED
    expansion_stop_reason: Optional[str] = None
    #: Deliberate, bounded choices -- not incompleteness. Kept separate from
    #: `incompleteness_reasons` so a normal run does not read as degraded.
    expansion_notes: list[str] = []


class TransferRow(BaseModel):
    """One transfer touching the investigated wallet, as a flat row.

    The report already carried aggregate counts, per-asset totals and a
    counterparty roll-up, but never the individual transfers -- which meant a
    reader could see "31 outgoing" without being able to see *which* 31. This
    model closes that gap for both the compact terminal report and `--json`.

    It is a pure projection of graph edge attributes: nothing here is computed,
    inferred or defaulted. A field the edge does not carry stays None so the
    renderer can print N/A rather than a fabricated zero.
    """

    tx_hash: str
    timestamp: Optional[int] = None
    timestamp_utc: Optional[str] = None
    block_number: Optional[int] = None
    direction: str  # IN | OUT | SELF
    from_address: str
    to_address: str
    counterparty: Optional[str] = None
    asset: Optional[str] = None
    asset_type: Optional[str] = None
    token_contract: Optional[str] = None
    amount: Optional[float] = None
    transfer_source: Optional[str] = None
    status: Optional[str] = None


class InvestigationReport(BaseModel):
    """The complete result of one investigation. Renderer-agnostic."""

    model_config = ConfigDict(protected_namespaces=())

    # --- header ---
    wallet: str
    chain: str
    investigation_id: str
    started_at_utc: str
    duration_seconds: float
    provenance: DataProvenance
    parameters: dict[str, Any] = {}
    environment: dict[str, str] = {}

    # --- section 1 ---
    normalization: Optional[NormalizationReport] = None
    graph_summary: Optional[GraphSummary] = None
    transfer_count: int = 0
    wallet_in_graph: bool = False

    # --- the wallet's own transfers, oldest first ---
    transactions: list[TransferRow] = []

    # --- sections 2-4 ---
    attribution: Optional[BidirectionalAttributionResult] = None
    entities: list[AttributedEntity] = []
    counterparties: list[Counterparty] = []

    # --- section 5 ---
    behavior_patterns: list[BehaviorPattern] = []

    # --- section 6 ---
    temporal: Optional[TemporalAmountAnalysis] = None

    # --- section 7 ---
    ml: Optional[MLSection] = None

    # --- sections 8-9 ---
    risk: Optional[RiskAssessment] = None
    seed_dataset_path: str = ""
    seed_entry_count: int = 0
    seed_provenance_counts: dict[str, int] = {}
    conclusion: list[str] = []
    limitations: list[str] = []
    warnings: list[str] = []


def validate_wallet_address(wallet: str) -> str:
    """Validates and lower-cases an EVM address.

    Kept here rather than only on the provider so that a cached-graph run —
    which never constructs a provider — still rejects a malformed address at
    the same gate with the same message.
    """
    if not wallet or not isinstance(wallet, str):
        raise PipelineError("No wallet address was supplied.")
    candidate = wallet.strip()
    if not candidate.startswith("0x") or len(candidate) != ADDRESS_LENGTH:
        raise PipelineError(
            f"'{wallet}' is not a valid EVM address: expected '0x' followed by "
            f"40 hexadecimal characters ({ADDRESS_LENGTH} characters total), got "
            f"{len(candidate)}."
        )
    body = candidate[2:]
    if any(character not in "0123456789abcdefABCDEF" for character in body):
        raise PipelineError(
            f"'{wallet}' is not a valid EVM address: it contains non-hexadecimal "
            "characters after the '0x' prefix."
        )
    return candidate.lower()


def validate_chain_name(chain: str) -> str:
    """Resolves a chain name to its canonical form, or stops the run.

    Kept next to `validate_wallet_address` and for the same reason: the two
    cached modes never construct a provider, so provider-side resolution alone
    would let `--cached-graph X --chain dogecoin` produce a report whose header,
    every transfer and every attribution line said "dogecoin" about Ethereum
    data. A chain name is a claim about where the evidence came from, so it is
    checked in every mode, before any I/O.

    Returns the canonical name, which the caller must use from then on, so the
    report never echoes the operator's capitalisation back as if it were the
    chain's identity.
    """
    canonical = normalize_chain_name(chain)
    if canonical not in SUPPORTED_CHAINS:
        raise PipelineError(
            f"Unsupported chain '{chain}'. This build can resolve: "
            f"{', '.join(sorted(SUPPORTED_CHAINS))} - of which "
            f"{', '.join(sorted(CHAIN_VALIDATED_LIVE))} has been exercised "
            "against live provider data here. A chain name is not accepted "
            "unless its chain id is known, because querying one chain and "
            "labelling the results as another would produce a report that "
            "states a chain the data did not come from."
        )
    return canonical


async def acquire_real_data(
    provider: BlockchainProvider,
    wallet: str,
    settings: Settings,
    use_cache: bool = True,
) -> AcquisitionResult:
    """Fetches all three streams live. The only path that produces DataMode.REAL."""
    try:
        return await acquire_wallet_transactions(
            provider,
            wallet,
            max_records_per_stream=settings.max_transactions_per_investigation,
            use_cache=use_cache,
        )
    except InvalidAddressError as exc:
        raise PipelineError(f"The provider rejected this address: {exc}") from exc


def _seed_provenance_counts(entries: list[VASPSeedEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        key = entry.source_type.value
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _build_transaction_ledger(
    graph: nx.MultiDiGraph, wallet: str
) -> list[TransferRow]:
    """Projects the wallet's own edges into flat rows for the report.

    Uses the SAME edge set the temporal analysis counts (`wallet_incident_edges`)
    so the row count can never disagree with `transfer_count` for the wallet.
    Every field is read straight off the edge; nothing is inferred. The
    counterparty is the *other* end of the edge -- the destination for an
    outgoing transfer, the source for an incoming one -- and is None for a
    self-transfer, where there is no counterparty to name.
    """
    rows: list[TransferRow] = []
    for u, v, _key, data, direction in wallet_incident_edges(graph, wallet):
        timestamp = data.get("timestamp")
        if direction == "OUT":
            counterparty: Optional[str] = v
        elif direction == "IN":
            counterparty = u
        else:  # SELF
            counterparty = None
        rows.append(
            TransferRow(
                tx_hash=data.get("tx_hash", ""),
                timestamp=timestamp,
                timestamp_utc=(
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(timestamp)))
                    if timestamp
                    else None
                ),
                block_number=data.get("block_number"),
                direction=direction,
                from_address=u,
                to_address=v,
                counterparty=counterparty,
                asset=data.get("asset"),
                asset_type=data.get("asset_type"),
                token_contract=data.get("token_contract"),
                amount=data.get("amount"),
                transfer_source=data.get("transfer_source"),
                status=data.get("status"),
            )
        )
    return rows


def _load_seed(settings: Settings) -> list[VASPSeedEntry]:
    try:
        return load_vasp_seed(settings.vasp_seed_dataset_path)
    except SeedDataError as exc:
        raise PipelineError(
            "The known-VASP dataset could not be loaded from "
            f"{settings.vasp_seed_dataset_path}: {exc}. Attribution cannot run "
            "without it, and no substitute dataset is used."
        ) from exc


def load_non_vasp_negatives(
    settings: Settings, section: MLSection
) -> list[NonVASPEntry]:
    """Loads the curated negative reference, or explains its absence.

    Unlike the known-VASP dataset this is NOT fatal when missing: the blockchain
    evidence does not depend on it, and the only consequence is that the
    supervised VASP task keeps a positive class only and reports itself
    untrainable. A malformed file is reported the same way — never silently
    treated as an empty negative set, which would look identical in the output
    to a file that genuinely documents nothing.
    """
    try:
        return load_non_vasp_reference(settings.non_vasp_reference_path)
    except NegativeReferenceError as exc:
        section.rationale.append(
            "The curated non-VASP reference could not be loaded from "
            f"{settings.non_vasp_reference_path}: {exc}. The supervised VASP "
            "task therefore has no negative class. No address was labelled "
            "NOT_VASP_OWNED to compensate."
        )
        return []


def run_ml_analysis(
    graph: nx.MultiDiGraph,
    wallet: str,
    transfers: Optional[list[NormalizedTransfer]],
    seed_entries: list[VASPSeedEntry],
    graph_source: str,
    settings: Settings,
    enabled: bool = True,
) -> MLSection:
    """Runs the strongest ML the available real labels support, and says which.

    Order of preference, with the reason recorded either way:
      1. A supervised model, if `real_labels` reports sufficient labels. Trained
         with a group-disjoint split and evaluated once on held-out data.
      2. An unsupervised outlier model, which needs no labels and answers the
         weaker question "how unusual is this behaviour in this graph".
      3. Nothing, with the blockers stated.

    Labels are never invented to reach step 1.
    """
    if not enabled:
        return MLSection(
            approach="DISABLED",
            rationale=["ML analysis was disabled for this run (--no-ml)."],
        )

    section = MLSection(approach="UNAVAILABLE")

    # --- attempt the supervised tasks -------------------------------------
    if transfers:
        section.account_type_labels = derive_account_type_labels(
            transfers,
            min_transfers=settings.ml_min_transfers_per_sample,
            min_samples_per_class=settings.ml_min_samples_per_class,
        )
    else:
        section.rationale.append(
            "The account-type task needs the normalized transfer stream to read "
            "protocol-guaranteed labels (which provider stream each record came "
            "from). This run loaded a prebuilt graph, whose edges do not carry "
            "that provenance, so the task was not attempted."
        )

    section.vasp_labels = derive_vasp_labels(
        seed_entries,
        list(graph.nodes()),
        min_samples_per_class=settings.ml_min_samples_per_class,
        negative_reference=load_non_vasp_negatives(settings, section),
    )

    trainable = [
        outcome
        for outcome in (section.account_type_labels, section.vasp_labels)
        if outcome is not None and outcome.sufficient
    ]

    if trainable:
        # Prefer the VASP task when it is trainable: it answers the
        # investigative question directly, where account type is only context.
        chosen = next(
            (o for o in trainable if o.task == "vasp_ownership"), trainable[0]
        )
        section.training = train_real_model(
            graph, chosen, graph_source=graph_source, settings=settings
        )
        if section.training.trained:
            section.approach = "SUPERVISED"
            section.prediction = predict(
                graph, wallet, task=chosen.task, settings=settings
            )
            section.rationale.append(
                f"The '{chosen.task}' task had sufficient real labels "
                f"({chosen.class_counts}), so a supervised model was trained "
                "with a group-disjoint split and evaluated once on held-out "
                "data."
            )
            section.limitations.extend(section.training.limitations)
            return section
        section.rationale.append(
            "Training was attempted and did not complete; see the training "
            "blockers."
        )

    # --- no trainable task: state why, then run the honest alternative ----
    for outcome in (section.account_type_labels, section.vasp_labels):
        if outcome is None or outcome.sufficient:
            continue
        counts = ", ".join(
            f"{label}={count}" for label, count in sorted(outcome.class_counts.items())
        )
        section.rationale.append(
            f"Task '{outcome.task}' is not trainable from this data: "
            f"{counts or 'no labels at all'}, minimum "
            f"{outcome.min_required_per_class} per class."
        )
        section.limitations.extend(outcome.blockers)

    # The unlabelled fallback is attempted BEFORE anything is claimed about
    # it. Stating "an unsupervised model ran" up front was wrong whenever the
    # outlier stage itself declined for want of a comparison population: the
    # rationale asserted a run that section 7 then reported as absent.
    section.outlier = assess_address(
        graph, wallet, population_source=graph_source, settings=settings
    )
    if section.outlier.available:
        section.approach = "UNSUPERVISED"
        section.limitations.extend(section.outlier.limitations)
        section.rationale.append(
            "No labels were invented to make a supervised model possible. "
            "Instead an unsupervised outlier model ran, which requires no "
            "labels and answers a weaker but genuinely measurable question."
        )
    else:
        section.approach = "UNAVAILABLE"
        section.rationale.append(
            "No labels were invented to make a supervised model possible. The "
            "unlabelled alternative -- an unsupervised outlier model -- was "
            "attempted and also declined to report, so this run produced NO "
            "machine-learning result of any kind."
        )
        if section.outlier.unavailable_reason:
            section.rationale.append(section.outlier.unavailable_reason)
            section.limitations.append(section.outlier.unavailable_reason)

    return section


def _build_conclusion(report: InvestigationReport) -> list[str]:
    """Writes the section-9 conclusion strictly from what the run established.

    Every line is a restatement of a field already in the report. Nothing is
    inferred here that is not already evidenced above it, and an incomplete
    search is never summarised as "no connection found".
    """
    lines: list[str] = []
    attribution = report.attribution

    if attribution is None:
        lines.append(
            "NO ATTRIBUTION ASSESSMENT: the investigation did not reach the "
            "attribution stage."
        )
        return lines

    if attribution.status == AttributionStatus.MATCH_FOUND:
        directions = sorted({c.direction.value for c in attribution.candidates})
        names = sorted({c.vasp_name for c in attribution.candidates})
        lines.append(
            f"VASP ATTRIBUTION: {len(attribution.candidates)} candidate(s) "
            f"across {len(names)} operator(s) — {', '.join(names)}."
        )
        lines.append(f"CONNECTION DIRECTIONS OBSERVED: {', '.join(directions)}.")
        lines.append(
            "Each candidate rests on an exact, case-insensitive address match "
            "against the known-VASP dataset. The strength of that claim is the "
            "provenance of the dataset entry, which is stated per candidate in "
            "section 3."
        )
    elif attribution.status == AttributionStatus.INCONCLUSIVE:
        depth = report.provenance.observation_depth
        requested = report.parameters.get("max_hops")
        if depth is not None and requested is not None and depth < requested:
            reason = report.provenance.expansion_stop_reason
            if reason == STOP_NO_NEW_COUNTERPARTIES:
                explanation = (
                    "recursive acquisition ran out of expandable counterparties "
                    f"after {depth} hop(s) -- the value-bearing graph reachable "
                    f"from this wallet does not extend to {requested} hop(s), so "
                    "there was nothing deeper left to fetch"
                )
            elif reason == STOP_DEPTH_REACHED:
                explanation = (
                    "recursive acquisition did expand every requested hop, but "
                    "at least one address along the way could not be read in "
                    "full, so that address's onward edges are unobserved (the "
                    "specific gaps are listed under data limitations)"
                )
            elif reason == STOP_ADDRESS_BUDGET:
                explanation = (
                    "recursive acquisition reached its address budget before "
                    f"expanding out to {requested} hop(s), so the deeper edges "
                    "were never fetched"
                )
            else:
                explanation = (
                    "this dataset was not acquired to that depth, so the deeper "
                    "edges are absent from it"
                )
            lines.append(
                "VASP ATTRIBUTION: INCONCLUSIVE. No known VASP address was "
                f"connected within the {depth} hop(s) this dataset observes, "
                f"and that finding IS complete at {depth} hop(s). It is "
                f"inconclusive for the configured {requested} hop(s) because "
                f"{explanation}. Raising the hop limit alone cannot change "
                "that: the missing edges are not in the data."
            )
        else:
            lines.append(
                "VASP ATTRIBUTION: INCONCLUSIVE. The search did not complete within "
                "its configured budget, so the absence of a match is NOT evidence "
                "that no connection exists. Re-run with a larger budget or a "
                "narrower time window before drawing any conclusion."
            )
    else:
        lines.append(
            "VASP ATTRIBUTION: NONE. No address in the known-VASP dataset was "
            "reached from or to this wallet within the searched depth, and the "
            "search completed. Note this is bounded by the dataset's coverage: "
            f"{report.seed_entry_count} address(es) are in it, so 'no known VASP' "
            "means 'none of those', not 'no exchange'."
        )

    if report.risk is not None:
        lines.append(
            f"INVESTIGATIVE PRIORITY: {report.risk.score} points, band "
            f"{report.risk.band.value} (medium at "
            f"{report.risk.band_medium_threshold}, high at "
            f"{report.risk.band_high_threshold}). The score is the sum of the "
            "itemised contributions in section 8 and nothing else. It is an "
            "open-ended tally of how much follow-up material was found, not a "
            "percentage of some maximum, and not a probability of wrongdoing."
        )

    investigative = [
        p
        for p in report.behavior_patterns
        if getattr(p.classification, "value", str(p.classification))
        == "INVESTIGATIVE_INDICATOR"
    ]
    lines.append(
        f"BEHAVIOURAL INDICATORS: {len(report.behavior_patterns)} observed, of "
        f"which {len(investigative)} are investigative indicators. None of these "
        "is an allegation of criminal conduct; each is an observation with a "
        "stated threshold that requires further verification."
    )

    if report.ml is not None:
        if report.ml.approach == "SUPERVISED" and report.ml.prediction:
            lines.append(
                f"ML: supervised prediction {report.ml.prediction.predicted_class} "
                "as SUPPORTING evidence only — it does not override the "
                "address-level findings above."
            )
        elif report.ml.approach == "UNSUPERVISED" and report.ml.outlier:
            lines.append(
                "ML: no supervised model was trained because the available real "
                "labels were insufficient. An unsupervised outlier model placed "
                "this wallet at the "
                f"{report.ml.outlier.percentile_within_population}th percentile "
                f"of {report.ml.outlier.population_size} addresses. This is "
                "CONTEXTUAL only and is not a VASP determination."
            )
        elif report.ml.approach == "DISABLED":
            lines.append("ML: disabled for this run.")
        else:
            lines.append(
                "ML: no model could be applied. No accuracy or confidence figure "
                "is reported, because none was measured."
            )

    if not report.provenance.data_complete:
        lines.append(
            "DATA COMPLETENESS: the underlying dataset is INCOMPLETE "
            f"({'; '.join(report.provenance.incompleteness_reasons)}). Every "
            "negative finding in this report is provisional."
        )

    depth = report.provenance.observation_depth
    requested_hops = report.parameters.get("max_hops")
    if depth is not None and requested_hops is not None and depth < requested_hops:
        stop = report.provenance.expansion_stop_reason
        lines.append(
            f"SEARCH DEPTH: this dataset's edges are complete to {depth} hop(s) "
            f"around the wallet, but the search was configured for "
            f"{requested_hops} hop(s)"
            + (f", and recursive acquisition stopped because {stop_reason_text(stop)}"
               if stop else "")
            + f". A route of {depth + 1}+ hops is therefore absent from the data "
            f"rather than ruled out. Findings at up to {depth} hop(s) are "
            "complete; anything deeper is unobserved, which is why the path "
            "search reports INCONCLUSIVE rather than NONE."
        )

    return lines


def run_investigation(
    wallet: str,
    chain: str,
    graph: nx.MultiDiGraph,
    provenance: DataProvenance,
    settings: Optional[Settings] = None,
    transfers: Optional[list[NormalizedTransfer]] = None,
    normalization: Optional[NormalizationReport] = None,
    graph_summary: Optional[GraphSummary] = None,
    max_hops: Optional[int] = None,
    max_paths: Optional[int] = None,
    time_window_days: Optional[int] = None,
    enable_ml: bool = True,
    investigation_id: Optional[str] = None,
) -> InvestigationReport:
    """Runs every analysis stage over an already-acquired graph.

    Takes the graph rather than fetching it, so the identical code path serves
    a live fetch, a reloaded real graph, and a test fixture. Acquisition is the
    caller's job precisely because that is where REAL and CACHED REAL DATA
    differ, and the difference must be explicit.
    """
    settings = settings or get_settings()
    started = time.time()
    wallet = validate_wallet_address(wallet)

    # Acquisition paths that measured their own radius declare it (live
    # acquisition counts the hop levels it actually expanded in full). A graph
    # reloaded from disk carries the radius its own run measured, if that run
    # recorded one; only when neither is available is the radius inferred from
    # the graph's shape — and an inferred horizon is still far better than
    # assuming the data covers however many hops the caller asked for.
    if provenance.observation_depth is None:
        provenance.observation_depth = recorded_observation_depth(graph, wallet)
    if provenance.observation_depth is None:
        provenance.observation_depth = infer_observation_depth(graph, wallet)

    report = InvestigationReport(
        wallet=wallet,
        chain=chain,
        investigation_id=investigation_id or f"{int(started)}-{wallet[-8:]}",
        started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        duration_seconds=0.0,
        provenance=provenance,
        normalization=normalization,
        graph_summary=graph_summary,
        transfer_count=graph.number_of_edges(),
        wallet_in_graph=wallet in graph,
        parameters={
            "max_hops": max_hops if max_hops is not None else settings.fund_trace_max_hops,
            "max_paths": (
                max_paths
                if max_paths is not None
                else settings.targeted_trace_max_paths_per_target
            ),
            "time_window_days": (
                time_window_days
                if time_window_days is not None
                else settings.fund_trace_time_window_days
            ),
            "max_edges_explored": settings.targeted_trace_max_edges_explored,
            "ml_enabled": enable_ml,
        },
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    )

    if not report.wallet_in_graph:
        report.warnings.append(
            f"Address {wallet} does not appear on any edge of the loaded graph. "
            "Every downstream section is therefore empty by construction, not "
            "by a finding. If the address is correct, the graph does not cover "
            "its activity."
        )

    # --- VASP dataset -----------------------------------------------------
    seed_entries = _load_seed(settings)
    seed_index = build_seed_index(seed_entries)
    registry = build_entity_registry(seed_index)
    report.seed_dataset_path = settings.vasp_seed_dataset_path
    report.seed_entry_count = len(seed_entries)
    report.seed_provenance_counts = _seed_provenance_counts(seed_entries)

    if any(e.source_type == SeedSourceType.SYNTHETIC_DEMO for e in seed_entries):
        report.warnings.append(
            "The loaded VASP dataset contains SYNTHETIC_DEMO entries. Synthetic "
            "addresses must not appear in a production dataset; any candidate "
            "derived from one is flagged in section 3."
        )

    # --- counterparties ---------------------------------------------------
    report.counterparties = identify_counterparties(graph, wallet, registry)

    # --- the wallet's own transfers, as rows -------------------------------
    # A projection of edges already in the graph, not a new measurement: no
    # analysis result depends on it, and nothing downstream reads it.
    report.transactions = _build_transaction_ledger(graph, wallet)

    # --- behaviour (before attribution: attribution consumes the patterns) -
    report.behavior_patterns = analyze_wallet_behavior(graph, wallet, settings)

    # --- bidirectional attribution ---------------------------------------
    report.attribution = generate_bidirectional_candidates(
        graph,
        wallet,
        seed_index,
        settings=settings,
        behavior_patterns=report.behavior_patterns,
        max_hops=max_hops,
        max_paths=max_paths,
        time_window_days=time_window_days,
        observation_depth=provenance.observation_depth,
    )
    report.entities = resolve_candidate_entities(
        report.attribution.candidates, registry
    )

    # --- temporal / amount ------------------------------------------------
    report.temporal = analyze_temporal_and_amounts(graph, wallet, chain)

    # --- risk -------------------------------------------------------------
    report.risk = assess_risk(
        wallet,
        attribution=report.attribution,
        behavior_patterns=report.behavior_patterns,
        settings=settings,
        data_complete=provenance.data_complete,
        data_completeness_note=(
            "; ".join(provenance.incompleteness_reasons)
            if provenance.incompleteness_reasons
            else None
        ),
    )

    # --- ML ---------------------------------------------------------------
    report.ml = run_ml_analysis(
        graph,
        wallet,
        transfers,
        seed_entries,
        graph_source=f"{provenance.data_mode.value}: {provenance.source_description}",
        settings=settings,
        enabled=enable_ml,
    )

    # --- limitations and conclusion ---------------------------------------
    report.limitations = _collect_limitations(report)
    report.conclusion = _build_conclusion(report)
    report.duration_seconds = round(time.time() - started, 3)
    return report


def _collect_limitations(report: InvestigationReport) -> list[str]:
    """Gathers every stated limitation into one place for section 8.

    Deduplicated while preserving order so the report never pads its own
    limitations list by repeating one finding from three modules.
    """
    limitations: list[str] = [
        "A traceable path through the graph is a TRANSACTION PATH, not proof "
        "that the same funds moved end to end. An intermediary can commingle, "
        "swap, or hold value, so A->B->C does not establish continuity.",
        "A dataset label is an assertion by its source, not proof of ownership. "
        "Only OFFICIAL_DISCLOSURE and DIRECTLY_VERIFIED entries are first-party.",
        "A behavioural indicator is an observation against a stated threshold. "
        "It is not an allegation of criminal conduct.",
        f"Attribution coverage is bounded by the dataset: {report.seed_entry_count} "
        "address(es) are known. An unmatched counterparty is NOT_IN_DATASET, "
        "which is not the same as 'not a VASP'.",
    ]

    if report.provenance.data_mode == DataMode.CACHED_REAL_DATA:
        limitations.append(
            "This report was produced from CACHED REAL DATA: genuine chain data "
            "acquired in an earlier run and reloaded from disk. It is not live. "
            "Any activity after the cache was written is absent."
        )
    if report.attribution is not None and (
        report.attribution.outbound_search_truncated
        or report.attribution.inbound_searches_truncated
    ):
        # Name the actual cause: a horizon limit is not a budget limit, and
        # the remedies differ (acquire more data vs. raise the budget).
        horizon = report.provenance.observation_depth
        if horizon is not None and horizon < report.attribution.max_hops:
            limitations.append(
                f"At least one path search was limited by the data horizon "
                f"({horizon} hop(s) observed, {report.attribution.max_hops} "
                "requested), so its result is INCONCLUSIVE rather than "
                "negative beyond that depth."
            )
        else:
            limitations.append(
                "At least one path search hit its edge budget, so its result is "
                "INCONCLUSIVE rather than negative."
            )
    if report.temporal is not None:
        limitations.extend(report.temporal.limitations)
    if report.ml is not None:
        limitations.extend(report.ml.limitations)

    seen: set[str] = set()
    unique: list[str] = []
    for item in limitations:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


# --------------------------------------------------------------------------
# Acquisition entry points — one per DataMode, so the label can never be wrong
# --------------------------------------------------------------------------


def recorded_observation_depth(
    graph: nx.MultiDiGraph, wallet: str
) -> Optional[int]:
    """The horizon a previous run MEASURED and stamped onto this graph.

    `acquire_live` writes the radius it actually expanded in full, together
    with the wallet it was measured around. Reading it back beats re-deriving
    it from the graph's shape, because a recursive acquisition that stopped on
    a budget leaves unexpanded addresses at an inner hop, which
    `infer_observation_depth` cannot see.

    Returns None unless a radius is present AND was measured around this same
    wallet: a graph saved while investigating address X says nothing about the
    completeness radius around address Y, and applying it there would be an
    unfounded claim rather than a conservative one.
    """
    if graph.graph.get("observation_wallet") != wallet.lower():
        return None
    depth = graph.graph.get("observation_depth")
    return depth if isinstance(depth, int) else None


def infer_observation_depth(
    graph: nx.MultiDiGraph, wallet: str
) -> Optional[int]:
    """Hop radius, around `wallet`, for which this graph's edges are complete.

    A FALLBACK, used only for a graph that carries no measured radius of its
    own (see `recorded_observation_depth`): an older cached graph, or one
    rebuilt from a transfer file.

    This is a property of the ACQUISITION, not of the traversal, and the two
    must never be conflated. An acquisition that stops at the investigated
    wallet fetches only that wallet's own transactions, so every edge touches
    it and the graph describes exactly one hop of chain history. Traversing it
    with MAX_HOPS=4 is still a complete traversal — of a dataset that cannot
    contain a 2-hop route. Calling that a complete negative at 4 hops would
    claim the chain was searched to a depth the data never described.

    Measured as the greatest undirected distance from the wallet to any node.
    The reasoning: a node's onward edges are only present if its own
    transactions were acquired, so every node strictly inside the outermost
    ring was evidently expanded, while the outermost ring is the unexpanded
    frontier.

    Deliberately conservative in one direction: a counterparty that genuinely
    never transacted with anyone else is indistinguishable from one that was
    never expanded, so this can understate the radius and report
    INCONCLUSIVE where NONE would have been defensible. Understating
    certainty is the safe error here; overstating it is the one the report
    must never make.

    The assumption behind it — one unexpanded ring, on the outside — is
    exactly what a budget-stopped recursive acquisition violates: unexpanded
    addresses sit at an inner hop while expanded branches reach further out,
    and this measure would then OVERSTATE the radius. That is why every
    recursive run records its measured radius on the graph and this function
    is consulted only when none was recorded.

    Returns None when the question is meaningless (empty graph, or a wallet
    absent from it), which asserts no horizon at all.
    """
    if wallet not in graph or graph.number_of_nodes() <= 1:
        return None
    seen = {wallet: 0}
    frontier = deque([wallet])
    while frontier:
        node = frontier.popleft()
        depth = seen[node]
        for neighbour in iter_chain(graph.predecessors(node), graph.successors(node)):
            if neighbour not in seen:
                seen[neighbour] = depth + 1
                frontier.append(neighbour)
    return max(seen.values())


async def acquire_live(
    wallet: str,
    chain: str,
    settings: Settings,
    use_cache: bool = True,
    save_to: Optional[Path] = None,
    refresh: bool = False,
    max_hops: Optional[int] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[nx.MultiDiGraph, list[NormalizedTransfer], NormalizationReport, GraphSummary, DataProvenance]:
    """Live acquisition: DataMode.REAL. Fails clearly without credentials.

    Requires the API key up front rather than discovering the problem after a
    failed request, so the error names the missing configuration instead of
    surfacing an HTTP status.

    `use_cache=False` bypasses the response cache in both directions;
    `refresh=True` bypasses only the read, so stale entries are replaced
    rather than left behind. They are separate flags because they answer
    different questions ("touch the cache at all?" and "is what is in it
    current enough?").

    `max_hops` is the same number the path search uses, and it is deliberately
    the SAME number: searching four hops through a dataset that only contains
    one hop's edges cannot find a four-hop route, so acquisition expands to
    the depth the search will look. Defaults to `FUND_TRACE_MAX_HOPS`.

    `progress` is called with short status lines. A recursive run issues
    several provider requests per address and is sequential by design; without
    this the command sits silent for minutes with nothing to show that it is
    working rather than hung.
    """
    from app.blockchain.etherscan import EtherscanProvider

    if not settings.etherscan_api_key:
        raise PipelineError(
            "ETHERSCAN_API_KEY is not set, so no real blockchain data can be "
            "acquired. Copy .env.example to .env and add a key. This build does "
            "NOT fall back to demo or synthetic data: an investigation without "
            "real data would be worthless, so it fails here instead."
        )

    hops = max_hops if max_hops is not None else settings.fund_trace_max_hops

    # Known-VASP addresses are acquisition ENDPOINTS, not routes onward: the
    # question is whether funds reached one, and expanding an exchange hot
    # wallet would spend the whole budget reading transactions belonging to
    # unrelated customers without making the match to it any stronger. Loaded
    # from the same seed dataset attribution uses, so the two cannot disagree
    # about which addresses are known.
    seed_index = build_seed_index(_load_seed(settings))

    try:
        provider = EtherscanProvider(settings, chain_name=chain, refresh=refresh)
    except UnsupportedChainError as exc:
        # The name already passed validate_chain_name, so reaching here means
        # ETHERSCAN_CHAIN_ID is pinned to a different chain than the one asked
        # for. Stopping is the only honest outcome: the provider would answer,
        # but with another chain's transactions under this chain's name.
        raise PipelineError(str(exc)) from exc

    try:
        if not provider.validate_address(wallet):
            raise PipelineError(f"'{wallet}' is not a valid address for {chain}.")
        try:
            acquisition = await acquire_multi_hop(
                provider,
                wallet,
                chain,
                max_hops=hops,
                max_records_wallet=settings.max_transactions_per_investigation,
                max_records_expanded=settings.expansion_max_records_per_stream,
                max_addresses=settings.expansion_max_addresses,
                max_addresses_per_hop=settings.expansion_max_addresses_per_hop,
                terminal_addresses=seed_index.keys(),
                use_cache=use_cache,
                progress=progress,
            )
        except InvalidAddressError as exc:
            raise PipelineError(
                f"The provider rejected this address: {exc}"
            ) from exc
        except IngestionError as exc:
            # Only the investigated wallet's own failure reaches here; a
            # counterparty that could not be read is recorded on its hop and
            # does not abort the run.
            raise PipelineError(
                f"No transaction data could be retrieved for {wallet}: {exc}"
            ) from exc
    finally:
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            await aclose()

    if acquisition.total_records == 0:
        raise PipelineError(
            f"The provider returned no activity for {wallet} on {chain}. There "
            "is no data to analyse. This is a data outcome, not an error: the "
            "address may be unused, or may be correct but outside the queried "
            "range."
        )

    valid, normalization = validate_transfers(acquisition.transfers, chain)
    graph, stats = build_graph(valid)
    summary = summarize_graph(graph, stats)

    # Record the MEASURED horizon on the graph itself, keyed to the wallet it
    # was measured around, so that a later `--cached-graph` run reads back the
    # radius this run actually achieved instead of inferring one from the
    # graph's shape. Inference assumes the outermost ring is the only
    # unexpanded one, which a budget-stopped recursive acquisition breaks:
    # unexpanded addresses sit at hop 2 while expanded branches reach hop 4, so
    # an inferred radius would OVERSTATE what was observed. The wallet is
    # stored with it because a radius is only meaningful relative to the
    # address it was measured from.
    graph.graph["observation_depth"] = acquisition.observation_depth
    graph.graph["observation_wallet"] = wallet.lower()

    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_graph(graph, save_to)

    provenance = DataProvenance(
        data_mode=DataMode.REAL,
        provider=acquisition.provider,
        source_description=(
            f"live {acquisition.provider} recursive fetch of "
            f"{acquisition.total_records} record(s) from "
            f"{acquisition.addresses_fetched} address(es) across "
            f"{acquisition.hops_expanded} expanded hop level(s) of a requested "
            f"{acquisition.requested_hops}"
        ),
        graph_path=str(save_to) if save_to else None,
        streams=acquisition.stream_lines,
        cache_stats=acquisition.cache_stats,
        data_complete=acquisition.complete,
        incompleteness_reasons=acquisition.incompleteness_reasons,
        acquired_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Measured, not requested: the number of LEADING hop levels that were
        # expanded in full. A deferred or truncated address at hop 1 holds this
        # at 1 however deep the run was asked to go, because a route through
        # that address would be absent from the data rather than ruled out.
        observation_depth=acquisition.observation_depth,
        addresses_fetched=acquisition.addresses_fetched,
        addresses_discovered=acquisition.addresses_discovered,
        hops_expanded=acquisition.hops_expanded,
        max_hop_reached=acquisition.max_hop_reached,
        requested_expansion_hops=acquisition.requested_hops,
        expansion_stop_reason=acquisition.stop_reason,
        expansion_notes=acquisition.notes,
    )
    return graph, valid, normalization, summary, provenance


def acquire_from_cached_graph(path: Path) -> tuple[nx.MultiDiGraph, DataProvenance]:
    """Reloads a real graph saved by an earlier run: DataMode.CACHED_REAL_DATA.

    The provenance records that edges from an older builder may lack fields the
    current builder writes, because a report must not imply that a missing
    `block_number` means the chain did not have one.
    """
    if not path.is_file():
        raise PipelineError(
            f"No cached graph at {path}. Nothing was substituted; supply a "
            "valid path or run a live investigation."
        )
    graph = load_graph(path)
    if graph.number_of_edges() == 0:
        raise PipelineError(
            f"The cached graph at {path} has no edges, so there is nothing to "
            "analyse."
        )

    sample = next(iter(graph.edges(data=True)))[2]
    missing = [
        field
        for field in ("block_number", "token_contract", "transfer_source", "gas_used")
        if sample.get(field) is None
    ]

    reasons = [
        "Data is CACHED REAL DATA reloaded from disk, not a live fetch, so "
        "activity after the cache was written is absent."
    ]
    if missing:
        reasons.append(
            "This graph was written by an earlier builder and its edges lack: "
            f"{', '.join(missing)}. Those fields are absent from the CACHE, not "
            "from the chain; rebuild the graph to populate them."
        )

    return graph, DataProvenance(
        data_mode=DataMode.CACHED_REAL_DATA,
        source_description=f"cached real graph {path.name}",
        graph_path=str(path),
        streams=[f"reloaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"],
        data_complete=False,
        incompleteness_reasons=reasons,
    )


def acquire_from_transfers_file(
    path: Path, chain: str
) -> tuple[nx.MultiDiGraph, list[NormalizedTransfer], NormalizationReport, GraphSummary, DataProvenance]:
    """Rebuilds a graph from normalized real transfers on disk.

    Also DataMode.CACHED_REAL_DATA — the records are real, but they were
    acquired earlier. Unlike a cached graph this path yields the transfer
    stream, so the supervised label rules can run.
    """
    if not path.is_file():
        raise PipelineError(f"No transfers file at {path}.")
    transfers = load_transfers_from_fixture(path)
    if not transfers:
        raise PipelineError(f"The transfers file {path} contains no records.")

    valid, normalization = validate_transfers(transfers, chain)
    graph, stats = build_graph(valid)
    summary = summarize_graph(graph, stats)

    return (
        graph,
        valid,
        normalization,
        summary,
        DataProvenance(
            data_mode=DataMode.CACHED_REAL_DATA,
            source_description=f"normalized real transfers from {path.name}",
            transfers_path=str(path),
            streams=[f"{len(transfers)} normalized transfer record(s) on disk"],
            data_complete=False,
            incompleteness_reasons=[
                "Data is CACHED REAL DATA reloaded from a normalized transfer "
                "file, not a live fetch."
            ],
        ),
    )
