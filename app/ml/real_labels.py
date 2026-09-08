"""
REAL LABELS for the production ML pipeline — where ground truth comes from.

This module is the reason the production pipeline is allowed to exist. The
demo pipeline in `app/ml/training_data.py` invents 36 rows; this one derives
labels from facts that are true by protocol or asserted by a sourced dataset,
and REFUSES to produce a model when there are not enough of them.

THE TWO PERMITTED LABEL SOURCES
--------------------------------------------------------------------------
1. PROTOCOL_GUARANTEED — a consequence of how EVM chains work, not an
   opinion:

     * An address that appears as the SENDER of a top-level transaction
       (Etherscan `txlist`, i.e. `TransferSource.NATIVE_TRANSACTION`) signed
       that transaction. Only a key-controlled account can do that, so the
       address is an externally-owned account.

     * An address that appears as the SENDER of an internal transfer
       (`txlistinternal`, i.e. `TransferSource.INTERNAL_TRANSACTION`) moved
       native value from inside contract execution. Only deployed code can
       do that, so the address is a contract.

     * The `asset_identifier` of an ERC-20 transfer is the token contract
       that emitted the Transfer event, so it is a contract.

2. DATASET_PROVENANCE — a curated file that records the address together
   with the evidence for what it is:

     * POSITIVE: an address the known-VASP dataset records with first-party
       provenance (`OFFICIAL_DISCLOSURE` or `DIRECTLY_VERIFIED`). A
       third-party explorer label is NOT promoted to a training label: it is
       someone's assertion, and training on it would teach the model to
       reproduce that annotator rather than a fact about the chain.

     * NEGATIVE: an address the non-VASP reference file records with a
       documented reason why it cannot be a custodial account — no derivable
       key, a public self-identification as self-custody, or verified code
       with no operator-controlled withdrawal path. See
       `app/ml/negative_labels.py`.

WHAT IS EXPLICITLY NOT A LABEL (do not weaken)
--------------------------------------------------------------------------
Absence from the seed set. The seed set is a small curated sample, so
"not in it" carries essentially no information — a wallet missing from six
curated addresses is overwhelmingly likely to be a wallet nobody has
labelled yet, not a wallet that is known not to be a VASP. Labelling those
addresses NON_VASP would manufacture a majority class out of ignorance and
produce a model whose headline accuracy measures nothing.
`LabelingOutcome` therefore reports *insufficient labels* instead, naming
exactly what is missing.

The one honest route to a negative class is a curated file where each
address carries its OWN documented reason for not being a custodial
account — see `app/ml/negative_labels.py`, which `derive_vasp_labels`
accepts through its `negative_reference` argument. Nothing else fills that
class.

A label is also only usable if the address appears in the investigated
graph: features come from observed edges, so a label with no activity
behind it is an all-zero row, and one class being absent more often than
the other would teach the model that missing data means that class. Both
classes are therefore intersected with the dataset, and the drop is
counted in `excluded_absent_from_dataset`.

PROVENANCE CONSISTENCY CHECK
--------------------------------------------------------------------------
`NormalizedTransfer.transfer_source` carries a default
(`NATIVE_TRANSACTION`) so that older call sites keep working. That default
means a record can *claim* a stream it did not come from — and the EOA rule
and the internal-transfer rule both rest on the stream being real. Records
claiming a native stream while describing a token transfer are therefore
counted and excluded from stream-based labelling rather than trusted
(`_native_stream_is_trustworthy`). The token-contract rule needs no such
check: it reads `asset_type`, which is self-evidencing.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Iterable, Optional

from pydantic import BaseModel

from app.attribution.models import SeedSourceType, VASPSeedEntry
from app.ml.negative_labels import NonVASPEntry
from app.models import AssetType, NormalizedTransfer, TransferSource

# Bumped whenever the meaning of a label or the rules above change, so a
# trained artifact can never be silently reinterpreted under new rules.
LABEL_SCHEMA_VERSION = "real-labels-v1"


class LabelSource(str, Enum):
    """Where a label came from. Recorded per sample, never aggregated away."""

    PROTOCOL_GUARANTEED = "PROTOCOL_GUARANTEED"
    DATASET_PROVENANCE = "DATASET_PROVENANCE"


class AccountTypeLabel(str, Enum):
    """The account-type task: is this address key-controlled or code?

    Investigatively this is the distinction between a router, pool, or bridge
    sitting in the middle of a traced route and an account a person or desk
    actually operates. It is the difference between "these funds passed
    through a smart contract" and "someone forwarded these funds", which
    changes how every path in section 2 of the report should be read.
    """

    EXTERNALLY_OWNED_ACCOUNT = "EXTERNALLY_OWNED_ACCOUNT"
    CONTRACT = "CONTRACT"


class VASPLabel(str, Enum):
    """The VASP task. Only the positive class has a defensible definition
    from the data this project has; see the module docstring."""

    VASP_OWNED = "VASP_OWNED"
    NOT_VASP_OWNED = "NOT_VASP_OWNED"


class LabeledAddress(BaseModel):
    """One training sample's label, with the evidence that produced it."""

    address: str
    label: str
    label_source: LabelSource
    # Free text a reviewer can check against the chain or the dataset. Not
    # decoration: it is the only thing that makes "real labels" verifiable.
    justification: str
    # A supporting transaction hash where the guarantee is observable.
    evidence_tx_hash: Optional[str] = None
    # Grouping key for leakage-free splitting: an operator name when known,
    # otherwise the address itself.
    group_key: str


class LabelingOutcome(BaseModel):
    """The result of trying to build a labelled set.

    `sufficient` is the gate the training pipeline obeys. When it is False,
    no model is trained and no metric is reported — the report prints this
    object instead, which is the honest answer to "what is your accuracy?"
    when the data cannot support one.
    """

    task: str
    label_schema_version: str = LABEL_SCHEMA_VERSION
    labels: list[LabeledAddress] = []

    class_counts: dict[str, int] = {}
    sufficient: bool = False
    min_required_per_class: int = 0

    # Populated when the sampling frame or the provenance check removed
    # candidates, so a small dataset is always explained rather than just
    # small.
    excluded_below_activity_floor: int = 0
    excluded_inconsistent_provenance: int = 0
    excluded_conflicting_labels: int = 0
    # Curated labels whose address never appears in the investigated graph.
    # Training them would mean fitting an all-zero feature row, so they are
    # dropped — and counted, because a dataset that shrank has to say so.
    excluded_absent_from_dataset: int = 0
    activity_floor: int = 0

    blockers: list[str] = []
    notes: list[str] = []

    @property
    def group_count(self) -> int:
        return len({item.group_key for item in self.labels})


def _native_stream_is_trustworthy(transfer: NormalizedTransfer) -> bool:
    """Cross-checks a record that claims a NATIVE provider stream.

    The EOA rule and the internal-transfer rule both depend on *which* native
    stream a record came from, and `transfer_source` carries a default, so a
    record can claim `NATIVE_TRANSACTION` without having come from `txlist`.
    A native stream that describes a token transfer cannot have both fields
    right, so it is not trusted for labelling. It remains perfectly valid
    graph evidence — this check governs LABELS only.

    The token-contract rule is deliberately NOT routed through here: that a
    record's `asset_identifier` is a contract follows from `asset_type` being
    a token standard, which is self-evidencing and independent of the stream
    field.
    """
    return transfer.asset_type == AssetType.NATIVE


def _activity_counts(transfers: Iterable[NormalizedTransfer]) -> dict[str, int]:
    """How many transfers each address takes part in, counting every stream
    identically.

    Deliberately label-blind: it must not be possible for the activity floor
    to admit one class more readily than another, or "made it into the
    dataset" would itself become a feature of the label.
    """
    counts: dict[str, int] = defaultdict(int)
    for t in transfers:
        counts[t.from_address.lower()] += 1
        if t.to_address:
            counts[t.to_address.lower()] += 1
    return dict(counts)


def derive_account_type_labels(
    transfers: list[NormalizedTransfer],
    min_transfers: int = 3,
    min_samples_per_class: int = 20,
) -> LabelingOutcome:
    """Derives EOA/CONTRACT labels from protocol guarantees only.

    Both classes are drawn from the same sampling frame — addresses with at
    least `min_transfers` transfers — and the frame is computed before any
    label is read, so neither class is filtered more aggressively than the
    other. An address the data guarantees to be BOTH (possible for a
    delegated account under EIP-7702, and possible from malformed provider
    data) is dropped as ambiguous and counted, never assigned to whichever
    rule ran last.
    """
    outcome = LabelingOutcome(
        task="account_type",
        min_required_per_class=min_samples_per_class,
        activity_floor=min_transfers,
    )

    activity = _activity_counts(transfers)
    eligible = {a for a, n in activity.items() if n >= min_transfers}
    outcome.excluded_below_activity_floor = len(activity) - len(eligible)

    eoa_evidence: dict[str, str] = {}
    contract_evidence: dict[str, tuple[str, str]] = {}

    for t in transfers:
        sender = t.from_address.lower()

        # --- native-stream rules -------------------------------------------
        # These two depend on WHICH stream the record came from, so they are
        # the only rules gated on the provenance cross-check. The gate must
        # not skip the rest of the loop body: the token rule below is
        # independent and applies to exactly the records this gate rejects.
        if t.transfer_source == TransferSource.NATIVE_TRANSACTION:
            if _native_stream_is_trustworthy(t):
                # Signed a top-level transaction => key-controlled account.
                if sender in eligible and sender not in eoa_evidence:
                    eoa_evidence[sender] = t.tx_hash
            else:
                outcome.excluded_inconsistent_provenance += 1
        elif t.transfer_source == TransferSource.INTERNAL_TRANSACTION:
            if _native_stream_is_trustworthy(t):
                # Moved native value from inside contract execution => code.
                if sender in eligible and sender not in contract_evidence:
                    contract_evidence[sender] = (
                        t.tx_hash,
                        "sent native value in an internal transfer, which only "
                        "executing contract code can do",
                    )
            else:
                outcome.excluded_inconsistent_provenance += 1

        # --- token-contract rule -------------------------------------------
        # Evaluated for EVERY record regardless of the stream field: an
        # ERC-20/721/1155 record's asset identifier is by definition the
        # contract that emitted the event, which is self-evidencing.
        if t.asset_type != AssetType.NATIVE and t.asset_identifier:
            token = t.asset_identifier.lower()
            if token in eligible and token not in contract_evidence:
                contract_evidence[token] = (
                    t.tx_hash,
                    f"is the {t.asset_type.value} contract that emitted a "
                    "Transfer event",
                )

    conflicting = set(eoa_evidence) & set(contract_evidence)
    outcome.excluded_conflicting_labels = len(conflicting)
    if conflicting:
        outcome.notes.append(
            f"{len(conflicting)} address(es) satisfied both the EOA and the "
            "contract guarantee and were dropped as ambiguous rather than "
            "assigned a class. A delegated account (EIP-7702) can legitimately "
            "be both."
        )

    labels: list[LabeledAddress] = []
    for address in sorted(set(eoa_evidence) - conflicting):
        labels.append(
            LabeledAddress(
                address=address,
                label=AccountTypeLabel.EXTERNALLY_OWNED_ACCOUNT.value,
                label_source=LabelSource.PROTOCOL_GUARANTEED,
                justification=(
                    "Appears as the sender of a top-level transaction, which "
                    "requires a signature from a private key."
                ),
                evidence_tx_hash=eoa_evidence[address],
                group_key=address,
            )
        )
    for address in sorted(set(contract_evidence) - conflicting):
        tx_hash, why = contract_evidence[address]
        labels.append(
            LabeledAddress(
                address=address,
                label=AccountTypeLabel.CONTRACT.value,
                label_source=LabelSource.PROTOCOL_GUARANTEED,
                justification=f"Address {why}.",
                evidence_tx_hash=tx_hash,
                group_key=address,
            )
        )

    outcome.labels = labels
    outcome.class_counts = {
        AccountTypeLabel.EXTERNALLY_OWNED_ACCOUNT.value: sum(
            1 for item in labels if item.label == AccountTypeLabel.EXTERNALLY_OWNED_ACCOUNT.value
        ),
        AccountTypeLabel.CONTRACT.value: sum(
            1 for item in labels if item.label == AccountTypeLabel.CONTRACT.value
        ),
    }
    _apply_sufficiency_gate(outcome)

    if outcome.excluded_inconsistent_provenance:
        outcome.notes.append(
            f"{outcome.excluded_inconsistent_provenance} record(s) claimed a "
            "native provider stream while describing a token transfer and were "
            "excluded from stream-based labelling (they remain valid graph "
            "evidence). Records loaded from an older normalized fixture take "
            "the default `transfer_source`, which is exactly this case."
        )
    if outcome.class_counts.get(AccountTypeLabel.CONTRACT.value, 0) < outcome.min_required_per_class:
        outcome.blockers.append(
            "Contract labels are scarce in this transfer set. The primary "
            "source for this class is internal transactions "
            "(`txlistinternal`), which contribute a contract label for every "
            "address that moved native value from code; token contracts only "
            "contribute when they also appear on a transfer edge often enough "
            f"to clear the activity floor of {outcome.activity_floor}. "
            "Ingesting internal transactions is what makes the account-type "
            "task trainable."
        )

    return outcome


def derive_vasp_labels(
    seed_entries: Iterable[VASPSeedEntry],
    candidate_addresses: Iterable[str],
    min_samples_per_class: int = 20,
    negative_reference: Optional[Iterable[NonVASPEntry]] = None,
) -> LabelingOutcome:
    """Derives VASP ownership labels from two curated, documented sources.

    The positive class is real: an address the dataset records with
    first-party provenance is a defensible VASP_OWNED sample.

    The negative class has exactly one honest source — `negative_reference`,
    the curated non-VASP file described in `app/ml/negative_labels.py`, where
    every address carries its own documented reason for not being a custodial
    account. When no reference is supplied the negative class stays EMPTY:
    the remaining candidates are addresses nobody has labelled, and
    "unlabelled" is not "not a VASP". The outcome is then insufficient, and
    that is the correct output rather than a failure — the alternative is a
    model trained on a fabricated majority class.

    BOTH classes are restricted to addresses that actually appear in
    `candidate_addresses`, i.e. in the investigated graph. A label for an
    absent address would be trained on an all-zero feature vector, and if one
    class were absent more often than the other the model would learn
    "no data => that class" — a leak that would show up as a high accuracy
    measuring nothing. Addresses dropped for this reason are counted in
    `excluded_absent_from_dataset`, never silently.
    """
    outcome = LabelingOutcome(
        task="vasp_ownership",
        min_required_per_class=min_samples_per_class,
    )

    in_dataset = {a.lower() for a in candidate_addresses}

    first_party = {
        SeedSourceType.OFFICIAL_DISCLOSURE,
        SeedSourceType.DIRECTLY_VERIFIED,
    }
    labels: list[LabeledAddress] = []
    weaker_provenance = 0
    synthetic = 0
    positives_absent = 0

    for entry in seed_entries:
        if entry.source_type.is_synthetic:
            synthetic += 1
            continue
        if entry.source_type not in first_party:
            weaker_provenance += 1
            continue
        address = entry.address.lower()
        if address not in in_dataset:
            positives_absent += 1
            continue
        labels.append(
            LabeledAddress(
                address=address,
                label=VASPLabel.VASP_OWNED.value,
                label_source=LabelSource.DATASET_PROVENANCE,
                justification=(
                    f"Recorded in the known-VASP dataset as {entry.vasp_name} "
                    f"with {entry.source_type.value} provenance"
                    + (f" ({entry.source_url})" if entry.source_url else "")
                    + "."
                ),
                group_key=" ".join(entry.vasp_name.split()).lower(),
            )
        )

    positives = {item.address for item in labels}

    negative_entries = list(negative_reference or [])
    negatives_absent = 0
    evidence_census: dict[str, int] = {}

    for entry in negative_entries:
        address = entry.address.lower()
        if address not in in_dataset:
            negatives_absent += 1
            continue
        if address in positives:
            # The two curated files contradict each other about this address.
            # Dropped and counted rather than resolved by whichever file was
            # read last — a silent winner here would silently flip a label.
            outcome.excluded_conflicting_labels += 1
            continue
        labels.append(
            LabeledAddress(
                address=address,
                label=VASPLabel.NOT_VASP_OWNED.value,
                label_source=LabelSource.DATASET_PROVENANCE,
                justification=(
                    f"Documented in the non-VASP reference as {entry.entity_name} "
                    f"[{entry.reason_class.value}, evidence "
                    f"{entry.evidence_type.value}]: {entry.reason}"
                    + (f" ({entry.source_url})" if entry.source_url else "")
                ),
                group_key=entry.group_key,
            )
        )
        evidence_census[entry.evidence_type.value] = (
            evidence_census.get(entry.evidence_type.value, 0) + 1
        )

    unlabelled = len(in_dataset) - len({item.address for item in labels})

    outcome.labels = labels
    outcome.class_counts = {
        VASPLabel.VASP_OWNED.value: sum(
            1 for item in labels if item.label == VASPLabel.VASP_OWNED.value
        ),
        VASPLabel.NOT_VASP_OWNED.value: sum(
            1 for item in labels if item.label == VASPLabel.NOT_VASP_OWNED.value
        ),
    }
    # Both exclusions are provenance-based, so both belong in the structured
    # counter as well as the prose notes below. A JSON consumer reading
    # excluded_inconsistent_provenance == 0 while a note said "5 excluded"
    # would reasonably conclude nothing had been dropped.
    outcome.excluded_inconsistent_provenance = weaker_provenance + synthetic
    outcome.excluded_absent_from_dataset = positives_absent + negatives_absent
    _apply_sufficiency_gate(outcome)

    if not negative_entries:
        outcome.blockers.append(
            f"The negative class is empty by design. {max(unlabelled, 0)} address(es) "
            "in this investigation carry no dataset entry, but absence from a "
            "curated seed set is not evidence that an address is not "
            "VASP-controlled, so none of them may be labelled NOT_VASP_OWNED. "
            "Filling this class requires negatives with a documented reason — for "
            "example addresses independently attested as self-custody, or "
            "contracts whose code proves they are not a custodial deposit account."
        )
    else:
        labelled_negatives = outcome.class_counts[VASPLabel.NOT_VASP_OWNED.value]
        outcome.notes.append(
            f"{labelled_negatives} NOT_VASP_OWNED label(s) came from the curated "
            f"non-VASP reference, which documents {len(negative_entries)} "
            f"address(es) with a stated reason; {negatives_absent} of them do not "
            "appear in this investigation and were left unlabelled rather than "
            "trained on an all-zero feature vector. Evidence behind the labelled "
            "negatives: "
            + (
                ", ".join(f"{k}={v}" for k, v in sorted(evidence_census.items()))
                or "none"
            )
            + f". A further {max(unlabelled, 0)} address(es) here appear in neither "
            "curated file and remain UNLABELLED — absence from a curated set is "
            "not evidence that an address is not VASP-controlled."
        )
        if labelled_negatives == 0:
            outcome.blockers.append(
                "The negative class is empty: none of the "
                f"{len(negative_entries)} documented non-VASP addresses appear in "
                "this investigation's graph. Labelling them anyway would train on "
                "all-zero feature vectors, and labelling the unlabelled addresses "
                "instead is refused because absence from a curated set is not "
                "evidence that an address is not VASP-controlled."
            )
    if positives_absent:
        outcome.notes.append(
            f"{positives_absent} first-party dataset address(es) were excluded "
            "because they do not appear in this investigation's graph. A label "
            "with no observed activity behind it contributes an all-zero feature "
            "row, which teaches the model that missing data means that class."
        )
    if outcome.excluded_conflicting_labels:
        outcome.notes.append(
            f"{outcome.excluded_conflicting_labels} address(es) are claimed by "
            "both the known-VASP dataset and the non-VASP reference and were "
            "dropped as contradictory rather than assigned a class. This is a "
            "curation error in one of the two files and should be resolved there."
        )
    if weaker_provenance:
        outcome.notes.append(
            f"{weaker_provenance} dataset address(es) were excluded from labels "
            "because their provenance is a third-party or community label. Such "
            "an entry is good enough to report as attribution evidence with its "
            "source attached, but training on it would teach the model to "
            "reproduce an annotator rather than a fact."
        )
    if synthetic:
        outcome.notes.append(
            f"{synthetic} SYNTHETIC_DEMO dataset address(es) were excluded: "
            "synthetic entries may never enter a production training set."
        )

    return outcome


def _apply_sufficiency_gate(outcome: LabelingOutcome) -> None:
    """Sets `sufficient` and records a blocker for every class that is short.

    The gate is deliberately per-class rather than on the total: 600 samples
    of one class and 3 of the other is not a dataset, it is a majority-class
    predictor waiting to report a high accuracy.
    """
    shortfalls = [
        (name, count)
        for name, count in sorted(outcome.class_counts.items())
        if count < outcome.min_required_per_class
    ]
    outcome.sufficient = not shortfalls and len(outcome.class_counts) >= 2

    for name, count in shortfalls:
        outcome.blockers.append(
            f"Class {name} has {count} labelled address(es); "
            f"{outcome.min_required_per_class} are required "
            "(ML_MIN_SAMPLES_PER_CLASS). Training is refused rather than "
            "reporting a metric estimated from too few samples to mean "
            "anything."
        )
