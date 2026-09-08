"""
Tests for `app.ml.negative_labels` and the negative half of
`derive_vasp_labels`.

The negative class is the single easiest place in this project to fabricate
data, because the tempting shortcut -- "it is not in the seed file, so call it
NOT_VASP_OWNED" -- produces a large, plausible-looking dataset out of nothing
but ignorance, and the resulting model reports a high accuracy that measures
nothing. These tests pin the rules that stop that:

  * A NEGATIVE NEEDS ITS OWN DOCUMENTED REASON. Labels come from a curated
    file where every entry names a reason class and an evidence type, and a
    file that cannot be trusted as-is is rejected rather than partially read.
  * A REFERENCE ENTRY IS NOT A TRAINING SAMPLE. It becomes one only if the
    address actually appears in the investigated graph, so no row is ever fit
    from an all-zero feature vector.
  * A SHORT DATASET STAYS SHORT. Nothing is padded to reach a sample count;
    the outcome reports `sufficient=False` with exact counts instead.
  * WHAT WAS DROPPED IS COUNTED. A dataset that shrank says so, in a
    structured field and not only in prose.

Fully offline: reads local files and nothing else.
"""

from __future__ import annotations

import json

import pytest

from app.attribution.models import SeedSourceType, VASPSeedEntry
from app.ml.negative_labels import (
    NON_VASP_REFERENCE_VERSION,
    NegativeReferenceError,
    NonVASPEntry,
    load_non_vasp_reference,
)
from app.ml.real_labels import VASPLabel, derive_vasp_labels

SHIPPED_REFERENCE = "data/seed/non_vasp_reference.json"

ADDR_A = "0x" + "a1" * 20
ADDR_B = "0x" + "b2" * 20
ADDR_C = "0x" + "c3" * 20


def _entry(address: str, **overrides) -> dict:
    fields = {
        "address": address,
        "entity_name": "Test Protocol",
        "reason_class": "NON_CUSTODIAL_PROTOCOL_CONTRACT",
        "reason": "verified code with no operator-controlled withdrawal path",
        "evidence_type": "PUBLISHED_SOURCE_CODE",
        "source": "test",
    }
    fields.update(overrides)
    return fields


def _write(tmp_path, payload, name: str = "non_vasp.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _seed(address: str, name: str = "Example Exchange", **overrides) -> VASPSeedEntry:
    fields = {
        "address": address,
        "vasp_name": name,
        "entity_type": "exchange",
        "chain": "ethereum",
        "source": "test",
        "source_type": SeedSourceType.OFFICIAL_DISCLOSURE,
        "confidence_note": "test entry",
    }
    fields.update(overrides)
    return VASPSeedEntry(**fields)


# --------------------------------------------------------------------------
# loading: the file is either trustworthy or rejected
# --------------------------------------------------------------------------


def test_a_bare_array_and_a_wrapped_object_load_identically(tmp_path):
    """Mirrors `load_vasp_seed`, so the two curated files can be written in
    whichever shape reads better without a code change."""
    payload = [_entry(ADDR_A)]
    bare = load_non_vasp_reference(_write(tmp_path, payload, "bare.json"))
    wrapped = load_non_vasp_reference(
        _write(tmp_path, {"non_vasp_addresses": payload}, "wrapped.json")
    )
    assert [e.address for e in bare] == [e.address for e in wrapped] == [ADDR_A]


def test_a_missing_file_is_an_error_not_an_empty_negative_class(tmp_path):
    """Silently returning [] would produce a positive-only dataset that looks
    like a curation decision rather than a missing file."""
    with pytest.raises(NegativeReferenceError, match="not found"):
        load_non_vasp_reference(tmp_path / "does_not_exist.json")


def test_malformed_json_is_rejected(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(NegativeReferenceError, match="not valid JSON"):
        load_non_vasp_reference(path)


def test_an_unexpected_top_level_shape_is_rejected(tmp_path):
    with pytest.raises(NegativeReferenceError, match="must be a JSON array"):
        load_non_vasp_reference(_write(tmp_path, {"addresses": []}))


def test_a_missing_reason_is_rejected_rather_than_defaulted(tmp_path):
    """A negative with no stated reason is exactly the fabrication this file
    exists to prevent, so it may not be admitted with a blank reason."""
    broken = _entry(ADDR_A)
    del broken["reason"]
    with pytest.raises(NegativeReferenceError, match="malformed"):
        load_non_vasp_reference(_write(tmp_path, [broken]))


def test_an_unknown_reason_class_is_rejected(tmp_path):
    with pytest.raises(NegativeReferenceError, match="malformed"):
        load_non_vasp_reference(
            _write(tmp_path, [_entry(ADDR_A, reason_class="SEEMS_FINE")])
        )


def test_a_malformed_address_is_rejected(tmp_path):
    with pytest.raises(NegativeReferenceError, match="not a valid EVM address"):
        load_non_vasp_reference(_write(tmp_path, [_entry("0xshort")]))


def test_a_duplicate_address_is_rejected_rather_than_resolved(tmp_path):
    """Two reasons for one address means the file contradicts itself; picking
    whichever was read last would silently choose one."""
    payload = [
        _entry(ADDR_A, entity_name="First"),
        _entry("0x" + ADDR_A[2:].upper(), entity_name="Second"),
    ]
    with pytest.raises(NegativeReferenceError, match="Duplicate"):
        load_non_vasp_reference(_write(tmp_path, payload))


def test_the_group_key_falls_back_to_the_entity_name():
    """Splitting groups keep a protocol's sibling contracts on one side of the
    split; a file that omits the group must not silently get one group per
    address, which would let a router leak its own factory."""
    grouped = NonVASPEntry.model_validate(_entry(ADDR_A, group="uniswap"))
    ungrouped = NonVASPEntry.model_validate(_entry(ADDR_B, entity_name="Some  DEX"))
    assert grouped.group_key == "uniswap"
    assert ungrouped.group_key == "some dex"


# --------------------------------------------------------------------------
# the shipped file
# --------------------------------------------------------------------------


def test_the_shipped_reference_file_loads_and_carries_full_provenance():
    entries = load_non_vasp_reference(SHIPPED_REFERENCE)
    assert entries, "the shipped negative reference must not be empty"
    for entry in entries:
        assert entry.reason.strip(), f"{entry.address} has no stated reason"
        assert entry.source.strip(), f"{entry.address} has no source"
        assert entry.entity_name.strip()


def test_the_shipped_reference_shares_no_address_with_the_known_vasp_seed():
    """The two curated files disagreeing about an address is a curation error.
    `derive_vasp_labels` drops such an address, so this failing would silently
    shrink the dataset rather than break anything visible."""
    from app.attribution.seed_loader import load_vasp_seed
    from app.core.config import get_settings

    negatives = {e.address.lower() for e in load_non_vasp_reference(SHIPPED_REFERENCE)}
    positives = {
        e.address.lower()
        for e in load_vasp_seed(get_settings().vasp_seed_dataset_path)
    }
    assert not (negatives & positives)


def test_the_reference_version_is_pinned():
    """A trained artifact records this string; if the file's meaning changes
    without the version changing, an old model is reinterpreted in silence."""
    assert NON_VASP_REFERENCE_VERSION == "non-vasp-reference-v1"


# --------------------------------------------------------------------------
# derive_vasp_labels: negatives only from the reference, only if present
# --------------------------------------------------------------------------


def test_without_a_reference_the_negative_class_stays_empty():
    """The unlabelled addresses in a graph are addresses nobody has labelled.
    Turning them into a majority class is the one thing this must never do."""
    outcome = derive_vasp_labels(
        [_seed(ADDR_A)],
        candidate_addresses=[ADDR_A, ADDR_B, ADDR_C],
        min_samples_per_class=1,
    )
    assert outcome.class_counts[VASPLabel.NOT_VASP_OWNED.value] == 0
    assert outcome.sufficient is False
    assert any("empty by design" in b for b in outcome.blockers)
    assert any("not evidence" in b for b in outcome.blockers)


def test_a_documented_negative_present_in_the_graph_becomes_a_label():
    reference = [NonVASPEntry.model_validate(_entry(ADDR_B))]
    outcome = derive_vasp_labels(
        [_seed(ADDR_A)],
        candidate_addresses=[ADDR_A, ADDR_B],
        min_samples_per_class=1,
        negative_reference=reference,
    )
    assert outcome.class_counts == {
        VASPLabel.VASP_OWNED.value: 1,
        VASPLabel.NOT_VASP_OWNED.value: 1,
    }
    assert outcome.sufficient is True
    negative = next(
        item for item in outcome.labels if item.label == VASPLabel.NOT_VASP_OWNED.value
    )
    # The justification is the only thing that makes the label checkable.
    assert "NON_CUSTODIAL_PROTOCOL_CONTRACT" in negative.justification
    assert "PUBLISHED_SOURCE_CODE" in negative.justification
    assert "withdrawal path" in negative.justification


def test_a_documented_negative_absent_from_the_graph_is_dropped_and_counted():
    """An absent address would be fit as an all-zero row; if one class were
    absent more often the model would learn "no data means that class"."""
    reference = [
        NonVASPEntry.model_validate(_entry(ADDR_B)),
        NonVASPEntry.model_validate(_entry(ADDR_C)),
    ]
    outcome = derive_vasp_labels(
        [_seed(ADDR_A)],
        candidate_addresses=[ADDR_A, ADDR_B],
        min_samples_per_class=1,
        negative_reference=reference,
    )
    assert outcome.class_counts[VASPLabel.NOT_VASP_OWNED.value] == 1
    assert outcome.excluded_absent_from_dataset == 1
    assert any("do not appear in this investigation" in n for n in outcome.notes)


def test_a_positive_absent_from_the_graph_is_dropped_and_counted():
    outcome = derive_vasp_labels(
        [_seed(ADDR_A), _seed(ADDR_C, name="Absent Exchange")],
        candidate_addresses=[ADDR_A],
        min_samples_per_class=1,
        negative_reference=[NonVASPEntry.model_validate(_entry(ADDR_A))],
    )
    # ADDR_A is claimed by both files, so it is dropped from both classes;
    # ADDR_C is absent from the graph. Nothing is left, and nothing is faked.
    assert outcome.excluded_absent_from_dataset == 1
    assert outcome.excluded_conflicting_labels == 1
    assert outcome.class_counts[VASPLabel.VASP_OWNED.value] == 1
    assert outcome.class_counts[VASPLabel.NOT_VASP_OWNED.value] == 0
    assert outcome.sufficient is False


def test_an_address_in_both_curated_files_is_dropped_not_resolved():
    outcome = derive_vasp_labels(
        [_seed(ADDR_A)],
        candidate_addresses=[ADDR_A, ADDR_B],
        min_samples_per_class=1,
        negative_reference=[
            NonVASPEntry.model_validate(_entry(ADDR_A)),
            NonVASPEntry.model_validate(_entry(ADDR_B)),
        ],
    )
    assert outcome.excluded_conflicting_labels == 1
    labels_for_a = [item for item in outcome.labels if item.address == ADDR_A.lower()]
    assert len(labels_for_a) == 1
    assert labels_for_a[0].label == VASPLabel.VASP_OWNED.value
    assert any("contradictory" in n for n in outcome.notes)


def test_negatives_are_matched_case_insensitively_but_exactly():
    """Same rule as the address matcher: case is irrelevant, everything else
    about the address is not."""
    reference = [NonVASPEntry.model_validate(_entry(ADDR_B.upper()))]
    matched = derive_vasp_labels(
        [_seed(ADDR_A)],
        candidate_addresses=[ADDR_A, ADDR_B.lower()],
        min_samples_per_class=1,
        negative_reference=reference,
    )
    assert matched.class_counts[VASPLabel.NOT_VASP_OWNED.value] == 1

    near_miss = ADDR_B[:-1] + ("0" if ADDR_B[-1] != "0" else "1")
    unmatched = derive_vasp_labels(
        [_seed(ADDR_A)],
        candidate_addresses=[ADDR_A, near_miss],
        min_samples_per_class=1,
        negative_reference=reference,
    )
    assert unmatched.class_counts[VASPLabel.NOT_VASP_OWNED.value] == 0


def test_a_reference_present_but_matching_nothing_says_so_explicitly():
    """Distinct from "no reference was supplied": here the file exists and was
    read, and the blocker has to name that so an operator looks in the right
    place."""
    outcome = derive_vasp_labels(
        [_seed(ADDR_A)],
        candidate_addresses=[ADDR_A],
        min_samples_per_class=1,
        negative_reference=[NonVASPEntry.model_validate(_entry(ADDR_B))],
    )
    assert outcome.class_counts[VASPLabel.NOT_VASP_OWNED.value] == 0
    assert outcome.sufficient is False
    assert any("none of the" in b and "documented non-VASP" in b for b in outcome.blockers)
    assert not any("empty by design" in b for b in outcome.blockers)


def test_a_short_negative_class_is_reported_short_never_padded():
    reference = [NonVASPEntry.model_validate(_entry(ADDR_B))]
    outcome = derive_vasp_labels(
        [_seed(ADDR_A)],
        candidate_addresses=[ADDR_A, ADDR_B],
        min_samples_per_class=20,
        negative_reference=reference,
    )
    assert outcome.sufficient is False
    assert len(outcome.labels) == 2, "no sample may be invented to reach the floor"
    for name in (VASPLabel.VASP_OWNED.value, VASPLabel.NOT_VASP_OWNED.value):
        assert any(
            name in b and "20 are required" in b for b in outcome.blockers
        ), f"the shortfall for {name} was not reported with its counts"


def test_synthetic_and_weak_provenance_positives_never_become_labels():
    outcome = derive_vasp_labels(
        [
            _seed(ADDR_A, source_type=SeedSourceType.SYNTHETIC_DEMO),
            _seed(ADDR_B, name="Explorer Label", source_type=SeedSourceType.COMMUNITY_LABEL),
        ],
        candidate_addresses=[ADDR_A, ADDR_B],
        min_samples_per_class=1,
        negative_reference=[NonVASPEntry.model_validate(_entry(ADDR_C))],
    )
    assert outcome.class_counts[VASPLabel.VASP_OWNED.value] == 0
    assert outcome.excluded_inconsistent_provenance == 2
    assert any("SYNTHETIC_DEMO" in n for n in outcome.notes)
    assert any("third-party or community label" in n for n in outcome.notes)


def test_negative_group_keys_reach_the_label_for_leakage_free_splitting():
    reference = [
        NonVASPEntry.model_validate(_entry(ADDR_B, entity_name="Router", group="uniswap")),
        NonVASPEntry.model_validate(_entry(ADDR_C, entity_name="Factory", group="uniswap")),
    ]
    outcome = derive_vasp_labels(
        [_seed(ADDR_A)],
        candidate_addresses=[ADDR_A, ADDR_B, ADDR_C],
        min_samples_per_class=1,
        negative_reference=reference,
    )
    negatives = [
        item for item in outcome.labels if item.label == VASPLabel.NOT_VASP_OWNED.value
    ]
    assert {item.group_key for item in negatives} == {"uniswap"}, (
        "sibling contracts of one protocol must share a group, or the split "
        "lets a model memorise the protocol from its own siblings"
    )


def test_the_evidence_census_reports_how_strong_the_negative_class_is():
    reference = [
        NonVASPEntry.model_validate(_entry(ADDR_B, evidence_type="PROTOCOL_RULE")),
        NonVASPEntry.model_validate(_entry(ADDR_C, evidence_type="PUBLIC_STATEMENT")),
    ]
    outcome = derive_vasp_labels(
        [_seed(ADDR_A)],
        candidate_addresses=[ADDR_A, ADDR_B, ADDR_C],
        min_samples_per_class=1,
        negative_reference=reference,
    )
    note = next(n for n in outcome.notes if "Evidence behind" in n)
    assert "PROTOCOL_RULE=1" in note
    assert "PUBLIC_STATEMENT=1" in note
