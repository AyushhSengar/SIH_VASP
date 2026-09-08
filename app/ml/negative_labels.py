"""
REAL NEGATIVE LABELS — the documented other half of the VASP training set.

`app/ml/real_labels.py` refuses to label an address NOT_VASP_OWNED just
because the curated seed set does not mention it, and that refusal is
correct: "nobody has labelled this" is not "this is not a VASP". The
consequence is a positive-only dataset, which is untrainable.

This module supplies the only honest way out. A negative label needs its own
documented reason, exactly like a positive one, so negatives are read from a
curated file (`data/seed/non_vasp_reference.json`) where every entry names:

  * a reason class — why this address cannot be a custodial account, and
  * an evidence type — what kind of fact that reason rests on.

Two rules keep this from degenerating back into fabrication:

1.  A reference entry is NOT a training sample. It becomes one only if the
    address actually appears in the investigated graph (see
    `derive_vasp_labels`), so the model never learns from an all-zero feature
    vector for an address nobody transacted with.

2.  Nothing is added to the file to reach a sample count. If the documented
    negatives present in a given investigation fall short of
    ML_MIN_SAMPLES_PER_CLASS, the labelling outcome stays insufficient and
    says so with exact counts. A short honest dataset is the reportable
    result; a padded one is not.

Fully offline: this reads a local file and nothing else.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ValidationError

# Bumped whenever the reference file's meaning changes, so a trained artifact
# can never be silently reinterpreted against a different negative set.
NON_VASP_REFERENCE_VERSION = "non-vasp-reference-v1"

_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


class NonVASPReasonClass(str, Enum):
    """Why an address is documented as not VASP-controlled.

    Being a contract is deliberately not a reason on its own — custodial
    exchanges deploy contracts too. The reason has to be about custody.
    """

    PROTOCOL_EVIDENT_NON_ACCOUNT = "PROTOCOL_EVIDENT_NON_ACCOUNT"
    PUBLIC_BURN_CONVENTION = "PUBLIC_BURN_CONVENTION"
    PUBLIC_SELF_IDENTIFICATION = "PUBLIC_SELF_IDENTIFICATION"
    NON_CUSTODIAL_PROTOCOL_CONTRACT = "NON_CUSTODIAL_PROTOCOL_CONTRACT"


class NonVASPEvidenceType(str, Enum):
    """What kind of fact the reason rests on. Reported alongside every label
    so a reviewer can see how strong the negative class actually is."""

    PROTOCOL_RULE = "PROTOCOL_RULE"
    PUBLISHED_SOURCE_CODE = "PUBLISHED_SOURCE_CODE"
    PUBLIC_STATEMENT = "PUBLIC_STATEMENT"
    ECOSYSTEM_CONVENTION = "ECOSYSTEM_CONVENTION"


class NonVASPEntry(BaseModel):
    """One documented non-VASP address."""

    address: str
    entity_name: str
    reason_class: NonVASPReasonClass
    reason: str
    evidence_type: NonVASPEvidenceType
    source: str
    source_url: str | None = None
    # Optional splitting group, for the same reason positives group on the
    # operator name: several addresses of one protocol (a factory, its router,
    # its later router) are highly correlated, so allowing them to straddle
    # train and test would let the model memorise the protocol from its own
    # siblings. Defaults to the entity name when a file omits it.
    group: str | None = None

    @property
    def group_key(self) -> str:
        return " ".join((self.group or self.entity_name).split()).lower()


class NegativeReferenceError(Exception):
    """Raised whenever the negative reference file cannot be trusted as-is.

    Strict for the same reason `SeedDataError` is: a silently-partial negative
    set would change what a trained model means without changing anything
    visible in the report.
    """


def load_non_vasp_reference(path: str | Path) -> list[NonVASPEntry]:
    """Loads and validates the curated non-VASP reference file.

    Accepts either a bare JSON array or an object with a
    `non_vasp_addresses` array, mirroring `load_vasp_seed`. Rejects a
    malformed entry, a malformed address, and any duplicate address — a
    duplicate means the file states a reason twice and possibly
    inconsistently, which is not something to guess about.
    """
    path = Path(path)

    if not path.exists():
        raise NegativeReferenceError(f"Non-VASP reference file not found at {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise NegativeReferenceError(
            f"Could not read non-VASP reference file at {path}: {exc}"
        ) from exc

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise NegativeReferenceError(
            f"Non-VASP reference file at {path} is not valid JSON: {exc}"
        ) from exc

    if isinstance(raw, list):
        entries_raw = raw
    elif isinstance(raw, dict) and isinstance(raw.get("non_vasp_addresses"), list):
        entries_raw = raw["non_vasp_addresses"]
    else:
        raise NegativeReferenceError(
            f"Non-VASP reference file at {path} must be a JSON array of entries, "
            "or an object with a 'non_vasp_addresses' array."
        )

    entries: list[NonVASPEntry] = []
    seen: dict[str, NonVASPEntry] = {}

    for i, item in enumerate(entries_raw):
        try:
            entry = NonVASPEntry.model_validate(item)
        except ValidationError as exc:
            raise NegativeReferenceError(
                f"Non-VASP reference entry #{i} in {path} is malformed: {exc}"
            ) from exc

        if not _ADDRESS_RE.match(entry.address):
            raise NegativeReferenceError(
                f"Non-VASP reference entry #{i} in {path} has address "
                f"'{entry.address}', which is not a valid EVM address: expected "
                "'0x' followed by 40 hexadecimal characters."
            )

        key = entry.address.lower()
        prior = seen.get(key)
        if prior is not None:
            raise NegativeReferenceError(
                f"Duplicate non-VASP reference address {entry.address} in {path}: "
                f"'{prior.entity_name}' ({prior.reason_class.value}) vs "
                f"'{entry.entity_name}' ({entry.reason_class.value}). "
                "Refusing to guess which reason applies."
            )

        seen[key] = entry
        entries.append(entry)

    return entries
