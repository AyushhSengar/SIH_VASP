"""
Tests for app/attribution/entities.py — entity resolution and counterparty
identification.

The load-bearing properties: grouping happens ONLY on the dataset's own
operator name, the exact matched address is never collapsed into a count,
similar-looking names are never merged, and an address absent from the seed
set is reported as NOT_IN_DATASET rather than as "not a VASP".
"""

from __future__ import annotations

import networkx as nx

from app.attribution.bidirectional_models import (
    BidirectionalCandidate,
    ConnectionDirection,
    DirectionalEvidence,
)
from app.attribution.entities import (
    build_entity_registry,
    identify_counterparties,
    resolve_candidate_entities,
)
from app.attribution.matcher import build_seed_index
from app.attribution.models import SeedSourceType, VASPSeedEntry

WALLET = "0xaaaa111111111111111111111111111111111a"
ADDR_1 = "0xbbbb222222222222222222222222222222222b"
ADDR_2 = "0xcccc333333333333333333333333333333333c"
ADDR_3 = "0xdddd444444444444444444444444444444444d"
UNKNOWN = "0xeeee555555555555555555555555555555555e"


def seed(
    address,
    vasp_name="TestVASP",
    entity_type="exchange",
    source_type=SeedSourceType.COMMUNITY_LABEL,
    wallet_role="hot_wallet",
    chain="ethereum",
) -> VASPSeedEntry:
    return VASPSeedEntry(
        address=address,
        vasp_name=vasp_name,
        entity_type=entity_type,
        chain=chain,
        source="unit-test",
        source_type=source_type,
        confidence_note="test fixture",
        wallet_role=wallet_role,
        verification_status="third_party_labeled",
    )


def add_edge(graph, u, v, tx_hash, occurrence=0, amount=1.0, ts=100, asset="ETH"):
    graph.add_edge(
        u,
        v,
        key=f"{tx_hash}#{occurrence}",
        amount=amount,
        asset=asset,
        asset_type="NATIVE",
        timestamp=ts,
        tx_hash=tx_hash,
        transfer_index=occurrence,
        chain="ethereum",
        transfer_type="TRANSFER",
        status="SUCCESS",
    )


def evidence(hop_distance=1, addresses=None, hashes=None) -> DirectionalEvidence:
    return DirectionalEvidence(
        hop_distance=hop_distance,
        path_addresses=addresses or [WALLET, ADDR_1],
        tx_hashes=hashes or ["0xtx1"],
        hop_timestamps=[100] * hop_distance,
    )


def candidate(address, vasp_name="TestVASP", direction=ConnectionDirection.DIRECT_OUTBOUND, hop=1):
    return BidirectionalCandidate(
        vasp_name=vasp_name,
        matched_address=address,
        entity_type="exchange",
        chain="ethereum",
        source_type=SeedSourceType.COMMUNITY_LABEL,
        seed_source="unit-test",
        seed_confidence_note="test fixture",
        direction=direction,
        outbound_evidence=evidence(hop_distance=hop),
    )


# --------------------------------------------------------------------------
# Registry construction
# --------------------------------------------------------------------------


def test_addresses_sharing_a_dataset_name_are_grouped():
    registry = build_entity_registry(
        build_seed_index([seed(ADDR_1, "Binance"), seed(ADDR_2, "Binance")])
    )

    assert len(registry.entities) == 1
    entity = registry.entities[0]
    assert entity.entity_name == "Binance"
    assert entity.address_count == 2
    assert [a.address for a in entity.addresses] == sorted([ADDR_1, ADDR_2])


def test_grouping_basis_states_it_is_a_dataset_assertion():
    registry = build_entity_registry(
        build_seed_index([seed(ADDR_1, "Binance"), seed(ADDR_2, "Binance")])
    )

    basis = registry.entities[0].grouping_basis
    assert "dataset assertion" in basis
    assert "not an" in basis and "inference" in basis


def test_similar_but_distinct_operator_names_are_never_merged():
    """"Binance" and "Binance US" are different regulated entities. Merging
    them would attach one operator's evidence to the other."""
    registry = build_entity_registry(
        build_seed_index([seed(ADDR_1, "Binance"), seed(ADDR_2, "Binance US")])
    )

    assert len(registry.entities) == 2
    assert {e.entity_name for e in registry.entities} == {"Binance", "Binance US"}


def test_grouping_key_normalises_case_and_whitespace_only():
    registry = build_entity_registry(
        build_seed_index([seed(ADDR_1, "Kraken"), seed(ADDR_2, "  kraken ")])
    )

    assert len(registry.entities) == 1
    assert registry.entities[0].address_count == 2


def test_address_lookup_is_exact_and_case_insensitive():
    registry = build_entity_registry(build_seed_index([seed(ADDR_1, "Kraken")]))

    assert registry.entity_for_address(ADDR_1) is not None
    assert registry.entity_for_address(ADDR_1.upper()) is not None
    # No prefix / substring matching.
    assert registry.entity_for_address(ADDR_1[:20]) is None
    assert registry.entity_for_address(UNKNOWN) is None


def test_per_address_provenance_is_preserved_not_averaged():
    registry = build_entity_registry(
        build_seed_index(
            [
                seed(ADDR_1, "OKX", source_type=SeedSourceType.OFFICIAL_DISCLOSURE),
                seed(ADDR_2, "OKX", source_type=SeedSourceType.COMMUNITY_LABEL),
            ]
        )
    )

    entity = registry.entities[0]
    by_address = {a.address: a.source_type for a in entity.addresses}
    assert by_address[ADDR_1] == SeedSourceType.OFFICIAL_DISCLOSURE
    assert by_address[ADDR_2] == SeedSourceType.COMMUNITY_LABEL
    # The entity-level summary is the strongest, and it does NOT overwrite the
    # weaker per-address value above.
    assert entity.strongest_source_type == SeedSourceType.OFFICIAL_DISCLOSURE


def test_entity_type_comes_from_the_strongest_provenance_entry():
    registry = build_entity_registry(
        build_seed_index(
            [
                seed(
                    ADDR_1,
                    "Acme",
                    entity_type="custodian",
                    source_type=SeedSourceType.OFFICIAL_DISCLOSURE,
                ),
                seed(
                    ADDR_2,
                    "Acme",
                    entity_type="exchange",
                    source_type=SeedSourceType.COMMUNITY_LABEL,
                ),
                seed(
                    ADDR_3,
                    "Acme",
                    entity_type="exchange",
                    source_type=SeedSourceType.COMMUNITY_LABEL,
                ),
            ]
        )
    )

    # Two crowd-sourced "exchange" labels do not outvote one official
    # "custodian" disclosure.
    assert registry.entities[0].entity_type == "custodian"


def test_synthetic_entries_are_flagged_at_the_entity_level():
    registry = build_entity_registry(
        build_seed_index(
            [seed(ADDR_1, "DemoVASP", source_type=SeedSourceType.SYNTHETIC_DEMO)]
        )
    )

    assert registry.entities[0].contains_synthetic is True


def test_registry_supports_non_exchange_entity_types_without_special_casing():
    registry = build_entity_registry(
        build_seed_index(
            [
                seed(ADDR_1, "CustodyCo", entity_type="custodian"),
                seed(ADDR_2, "PayCo", entity_type="payment_processor"),
                seed(ADDR_3, "BrokerCo", entity_type="broker"),
            ]
        )
    )

    assert {e.entity_type for e in registry.entities} == {
        "custodian",
        "payment_processor",
        "broker",
    }


def test_empty_seed_index_yields_empty_registry():
    registry = build_entity_registry({})

    assert registry.entities == []
    assert registry.address_index == {}
    assert registry.entity_for_address(ADDR_1) is None


def test_registry_construction_is_deterministic():
    entries = [seed(ADDR_3, "Zeta"), seed(ADDR_1, "Alpha"), seed(ADDR_2, "Alpha")]
    first = build_entity_registry(build_seed_index(entries))
    second = build_entity_registry(build_seed_index(list(reversed(entries))))

    assert first.model_dump() == second.model_dump()


# --------------------------------------------------------------------------
# Candidate resolution
# --------------------------------------------------------------------------


def test_candidates_group_under_one_entity_but_keep_exact_addresses():
    registry = build_entity_registry(
        build_seed_index([seed(ADDR_1, "Binance"), seed(ADDR_2, "Binance")])
    )
    resolved = resolve_candidate_entities(
        [candidate(ADDR_1, "Binance"), candidate(ADDR_2, "Binance")], registry
    )

    assert len(resolved) == 1
    assert resolved[0].entity_name == "Binance"
    # The whole point: addresses are listed, not counted away.
    assert resolved[0].matched_addresses == sorted([ADDR_1, ADDR_2])


def test_unmatched_dataset_addresses_for_the_entity_are_disclosed():
    registry = build_entity_registry(
        build_seed_index(
            [seed(ADDR_1, "Kraken"), seed(ADDR_2, "Kraken"), seed(ADDR_3, "Kraken")]
        )
    )
    resolved = resolve_candidate_entities([candidate(ADDR_1, "Kraken")], registry)

    assert resolved[0].matched_addresses == [ADDR_1]
    assert resolved[0].dataset_addresses_not_matched == sorted([ADDR_2, ADDR_3])
    assert any("NOT reached" in line for line in resolved[0].limitations)


def test_resolution_never_claims_to_identify_an_account_holder():
    registry = build_entity_registry(build_seed_index([seed(ADDR_1, "Kraken")]))
    resolved = resolve_candidate_entities([candidate(ADDR_1, "Kraken")], registry)

    joined = " ".join(resolved[0].limitations)
    assert "does not establish which account" in joined


def test_candidate_absent_from_registry_is_reported_not_dropped():
    registry = build_entity_registry(build_seed_index([seed(ADDR_1, "Kraken")]))
    resolved = resolve_candidate_entities([candidate(UNKNOWN, "MysteryVASP")], registry)

    assert len(resolved) == 1
    assert resolved[0].matched_addresses == [UNKNOWN]
    assert "No grouping inference was made" in resolved[0].grouping_basis


def test_multiple_directions_to_one_entity_are_collected():
    registry = build_entity_registry(
        build_seed_index([seed(ADDR_1, "Kraken"), seed(ADDR_2, "Kraken")])
    )
    resolved = resolve_candidate_entities(
        [
            candidate(ADDR_1, "Kraken", ConnectionDirection.DIRECT_OUTBOUND, hop=1),
            candidate(ADDR_2, "Kraken", ConnectionDirection.INDIRECT_INBOUND, hop=3),
        ],
        registry,
    )

    assert len(resolved) == 1
    assert set(resolved[0].directions) == {
        ConnectionDirection.DIRECT_OUTBOUND,
        ConnectionDirection.INDIRECT_INBOUND,
    }
    # Strongest (closest) evidence wins for the headline hop distance.
    assert resolved[0].strongest_hop_distance == 1


def test_resolved_entities_are_ordered_by_evidence_strength():
    registry = build_entity_registry(
        build_seed_index([seed(ADDR_1, "Far"), seed(ADDR_2, "Near")])
    )
    resolved = resolve_candidate_entities(
        [
            candidate(ADDR_1, "Far", ConnectionDirection.INDIRECT_OUTBOUND, hop=3),
            candidate(ADDR_2, "Near", ConnectionDirection.DIRECT_OUTBOUND, hop=1),
        ],
        registry,
    )

    assert [e.entity_name for e in resolved] == ["Near", "Far"]


def test_no_candidates_resolves_to_nothing():
    registry = build_entity_registry(build_seed_index([seed(ADDR_1, "Kraken")]))

    assert resolve_candidate_entities([], registry) == []


# --------------------------------------------------------------------------
# Counterparty identification
# --------------------------------------------------------------------------


def test_counterparties_are_counted_by_direction_with_tx_references():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET, ADDR_1, "0xa1", ts=100)
    add_edge(g, WALLET, ADDR_1, "0xa2", ts=200)
    add_edge(g, ADDR_1, WALLET, "0xa3", ts=300)
    add_edge(g, ADDR_2, WALLET, "0xa4", ts=400, asset="USDC")

    result = {c.address: c for c in identify_counterparties(g, WALLET)}

    assert result[ADDR_1].transfer_count == 3
    assert result[ADDR_1].outbound_count == 2
    assert result[ADDR_1].inbound_count == 1
    assert result[ADDR_1].first_seen == 100
    assert result[ADDR_1].last_seen == 300
    assert result[ADDR_1].tx_hashes == ["0xa1", "0xa2", "0xa3"]
    assert result[ADDR_2].assets == ["USDC"]


def test_dataset_counterparty_is_named_and_others_are_not_in_dataset():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET, ADDR_1, "0xa1")
    add_edge(g, WALLET, UNKNOWN, "0xa2")
    registry = build_entity_registry(build_seed_index([seed(ADDR_1, "Kraken")]))

    result = {c.address: c for c in identify_counterparties(g, WALLET, registry)}

    assert result[ADDR_1].entity_name == "Kraken"
    assert result[ADDR_1].wallet_role == "hot_wallet"
    assert result[ADDR_1].entity_source_type == SeedSourceType.COMMUNITY_LABEL
    assert result[ADDR_1].identification_status == "IDENTIFIED_FROM_DATASET"

    # Absence from a small curated seed set says nothing about what the
    # address is — so the status must not read as a negative identification.
    assert result[UNKNOWN].entity_name is None
    assert result[UNKNOWN].identification_status == "NOT_IN_DATASET"


def test_self_transfers_are_not_counterparties():
    g = nx.MultiDiGraph()
    add_edge(g, WALLET, WALLET, "0xa1")
    add_edge(g, WALLET, ADDR_1, "0xa2")

    result = identify_counterparties(g, WALLET)

    assert [c.address for c in result] == [ADDR_1]


def test_counterparties_sorted_busiest_first_and_limit_respected():
    g = nx.MultiDiGraph()
    for i in range(5):
        add_edge(g, WALLET, ADDR_1, f"0xa{i}", ts=100 + i)
    add_edge(g, WALLET, ADDR_2, "0xb0", ts=200)
    add_edge(g, WALLET, ADDR_3, "0xc0", ts=300)

    result = identify_counterparties(g, WALLET, limit=2)

    assert len(result) == 2
    assert result[0].address == ADDR_1
    assert result[0].transfer_count == 5


def test_wallet_absent_from_graph_has_no_counterparties():
    g = nx.MultiDiGraph()
    add_edge(g, ADDR_1, ADDR_2, "0xa1")

    assert identify_counterparties(g, WALLET) == []


def test_counterparty_identification_is_deterministic():
    g = nx.MultiDiGraph()
    for i in range(6):
        add_edge(g, WALLET, ADDR_1, f"0xa{i}", ts=100 + i)
        add_edge(g, ADDR_2, WALLET, f"0xb{i}", ts=200 + i)

    first = identify_counterparties(g, WALLET)
    second = identify_counterparties(g, WALLET)

    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]
