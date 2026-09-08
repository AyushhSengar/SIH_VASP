"""
ENTITY RESOLUTION — grouping addresses under the operator that runs them.

What this module is allowed to do
--------------------------------------------------------------------------
Group two addresses under one entity ONLY because the known-VASP dataset
says both belong to the same named operator. That is an explicit, sourced
assertion in `data/seed/known_vasps.json`, with its own provenance per
address, and it is the only grouping evidence used here.

What this module must never do (do not remove)
--------------------------------------------------------------------------
Infer common ownership from behaviour. Two addresses transacting with the
same counterparty, moving similar amounts, sharing a funding source, being
active at the same hours, or sitting in the same connected component are
NOT evidence that one operator controls both. Address clustering by
behaviour is a real technique, but it produces *hypotheses*, and this
project reports evidence — so no such inference exists in this file, and
`EntityGroup.grouping_basis` records the basis for every group so a reader
can see it was a dataset assertion rather than an analytic guess.

The exact matched address is ALWAYS preserved
--------------------------------------------------------------------------
Grouping is presentational: "Binance (2 addresses)" is easier to read than
two hex strings. But an evidence report must state the exact address that
was actually matched, because that is what a reviewer re-checks on-chain.
Every model here therefore carries the concrete address alongside the
entity name, and `EntityGroup.matched_addresses` never collapses to a
count.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from app.attribution.bidirectional_models import (
    BidirectionalCandidate,
    ConnectionDirection,
)
from app.attribution.models import SeedSourceType, VASPSeedEntry


class EntityAddress(BaseModel):
    """One address belonging to an entity, with its own provenance.

    Provenance is per-address, not per-entity: an operator can have one
    officially-disclosed address and one crowd-sourced label, and averaging
    those into a single entity-level confidence would overstate the weaker
    one and understate the stronger.
    """

    address: str
    chain: str
    wallet_role: Optional[str] = None
    source: str
    source_type: SeedSourceType
    source_url: Optional[str] = None
    verification_status: Optional[str] = None
    confidence_note: str


class EntityGroup(BaseModel):
    """One real-world operator and every dataset address attributed to it."""

    entity_key: str  # normalised grouping key (lowercased name)
    entity_name: str  # as recorded in the dataset
    entity_type: str  # exchange / custodian / broker / payment_processor / ...
    chains: list[str]

    addresses: list[EntityAddress]

    # The basis on which these addresses were grouped. Constant by design —
    # there is exactly one permitted basis, and stating it in the output makes
    # that auditable rather than a claim in a docstring.
    grouping_basis: str = (
        "Both addresses are recorded under the same operator name in the "
        "known-VASP dataset. Grouping is a dataset assertion, not an "
        "analytic inference from on-chain behaviour."
    )

    @property
    def address_count(self) -> int:
        return len(self.addresses)

    @property
    def strongest_source_type(self) -> SeedSourceType:
        """Best provenance among this entity's addresses.

        Reported alongside, never instead of, the per-address provenance: one
        officially-disclosed address does not upgrade a crowd-sourced label
        elsewhere in the same group.
        """
        return min(
            (a.source_type for a in self.addresses),
            key=lambda st: st.strength_rank,
        )

    @property
    def contains_synthetic(self) -> bool:
        return any(a.source_type.is_synthetic for a in self.addresses)

    def address_entry(self, address: str) -> Optional[EntityAddress]:
        """Exact, case-insensitive lookup — the only matching this project
        performs on addresses anywhere."""
        target = address.lower()
        for entry in self.addresses:
            if entry.address.lower() == target:
                return entry
        return None


class EntityRegistry(BaseModel):
    """All entities in a seed dataset, plus an exact address index."""

    entities: list[EntityGroup]
    # address (lowercased) -> entity_key. Exact matching only.
    address_index: dict[str, str] = {}

    def entity_for_address(self, address: str) -> Optional[EntityGroup]:
        key = self.address_index.get(address.lower())
        if key is None:
            return None
        for entity in self.entities:
            if entity.entity_key == key:
                return entity
        return None

    @property
    def multi_address_entities(self) -> list[EntityGroup]:
        return [e for e in self.entities if e.address_count > 1]


def _entity_key(entry: VASPSeedEntry) -> str:
    """The grouping key.

    Deliberately just the dataset's own operator name, case-normalised and
    whitespace-collapsed. No aliasing table, no "Binance" == "Binance.US"
    merging, no substring matching: "Binance" and "Binance US" are different
    regulated entities in different jurisdictions, and silently merging them
    would attach one operator's evidence to another. If two dataset entries
    should be one entity, they get the same `vasp_name` in the JSON — a data
    fix, visible in review, rather than a hidden rule in code.
    """
    return " ".join(entry.vasp_name.split()).lower()


def build_entity_registry(seed_index: dict[str, VASPSeedEntry]) -> EntityRegistry:
    """Groups a seed index into entities.

    Input is the same address->entry index attribution already uses, so the
    registry can never disagree with what was matched.
    """
    grouped: dict[str, list[VASPSeedEntry]] = {}
    for entry in seed_index.values():
        grouped.setdefault(_entity_key(entry), []).append(entry)

    entities: list[EntityGroup] = []
    address_index: dict[str, str] = {}

    for key in sorted(grouped):
        entries = sorted(grouped[key], key=lambda e: e.address.lower())
        addresses = [
            EntityAddress(
                address=e.address.lower(),
                chain=e.chain,
                wallet_role=e.wallet_role,
                source=e.source,
                source_type=e.source_type,
                source_url=e.source_url,
                verification_status=e.verification_status,
                confidence_note=e.confidence_note,
            )
            for e in entries
        ]
        # Entity type is taken from the strongest-provenance entry rather than
        # a majority vote, so a well-sourced "custodian" is not overridden by
        # two crowd-sourced "exchange" labels.
        primary = min(entries, key=lambda e: (e.source_type.strength_rank, e.address.lower()))
        entity = EntityGroup(
            entity_key=key,
            entity_name=primary.vasp_name,
            entity_type=primary.entity_type,
            chains=sorted({e.chain for e in entries}),
            addresses=addresses,
        )
        entities.append(entity)
        for address in addresses:
            address_index[address.address] = key

    return EntityRegistry(entities=entities, address_index=address_index)


class AttributedEntity(BaseModel):
    """One entity that this investigation actually connected to, with the
    exact addresses and directions that produced the connection."""

    entity_key: str
    entity_name: str
    entity_type: str
    strongest_source_type: SeedSourceType

    # The exact addresses matched BY THIS INVESTIGATION — a subset of the
    # entity's dataset addresses. Never replaced by a count.
    matched_addresses: list[str]
    directions: list[ConnectionDirection]
    strongest_hop_distance: int

    # Dataset addresses for this entity that were NOT reached. Reported so a
    # reader is not left to assume the whole entity was examined and matched.
    dataset_addresses_not_matched: list[str] = []

    grouping_basis: str
    limitations: list[str] = []


def resolve_candidate_entities(
    candidates: list[BidirectionalCandidate],
    registry: EntityRegistry,
) -> list[AttributedEntity]:
    """Groups attribution candidates by operator.

    Purely a view over candidates that already exist from address-level
    evidence — it cannot create, strengthen, or merge evidence. A candidate
    whose address is absent from the registry (possible if a caller passes a
    registry built from a different seed file) is grouped under its own
    reported VASP name rather than being dropped, because discarding evidence
    is worse than reporting it without a group.
    """
    buckets: dict[str, list[BidirectionalCandidate]] = {}
    for candidate in candidates:
        entity = registry.entity_for_address(candidate.matched_address)
        key = entity.entity_key if entity else " ".join(candidate.vasp_name.split()).lower()
        buckets.setdefault(key, []).append(candidate)

    resolved: list[AttributedEntity] = []
    for key in sorted(buckets):
        group = buckets[key]
        entity = registry.entity_for_address(group[0].matched_address)
        matched = sorted({c.matched_address.lower() for c in group})

        if entity is not None:
            all_dataset = [a.address for a in entity.addresses]
            not_matched = sorted(set(all_dataset) - set(matched))
            entity_name = entity.entity_name
            entity_type = entity.entity_type
            strongest = entity.strongest_source_type
            basis = entity.grouping_basis
        else:
            not_matched = []
            entity_name = group[0].vasp_name
            entity_type = group[0].entity_type
            strongest = group[0].source_type
            basis = (
                "This address is not present in the entity registry that was "
                "built for this run, so the group contains only the address "
                "actually matched. No grouping inference was made."
            )

        limitations = [
            "Entity grouping identifies the operator named in the dataset. It "
            "does not establish which account, customer, or person at that "
            "operator controls the matched address.",
        ]
        if not_matched:
            limitations.append(
                f"{len(not_matched)} further dataset address(es) for this "
                "entity were NOT reached by this investigation; they are "
                "listed so the match is not read as covering the whole entity."
            )
        if entity is not None and entity.contains_synthetic:
            limitations.append(
                "At least one address grouped under this entity is a "
                "SYNTHETIC_DEMO dataset entry and must not be reported as a "
                "real-world finding."
            )

        resolved.append(
            AttributedEntity(
                entity_key=key,
                entity_name=entity_name,
                entity_type=entity_type,
                strongest_source_type=strongest,
                matched_addresses=matched,
                directions=sorted(
                    {c.direction for c in group}, key=lambda d: d.value
                ),
                strongest_hop_distance=min(
                    c.strongest_hop_distance for c in group
                ),
                dataset_addresses_not_matched=not_matched,
                grouping_basis=basis,
                limitations=limitations,
            )
        )

    # Closest evidence first; ties on name for determinism.
    resolved.sort(key=lambda e: (e.strongest_hop_distance, e.entity_name.lower()))
    return resolved


class Counterparty(BaseModel):
    """A direct (1-hop) counterparty of the investigated wallet.

    Direct counterparties are reported separately from traced candidates
    because they need no path inference at all: the wallet and the address
    transacted with each other, once or many times.
    """

    address: str
    transfer_count: int
    inbound_count: int
    outbound_count: int
    assets: list[str]
    first_seen: Optional[int] = None
    last_seen: Optional[int] = None
    tx_hashes: list[str] = []

    # Populated only from an exact dataset match. `None` means "not in the
    # dataset", which is NOT the same as "not a VASP" — see the note below.
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    entity_source_type: Optional[SeedSourceType] = None
    wallet_role: Optional[str] = None
    is_contract: bool = False

    @property
    def identification_status(self) -> str:
        if self.entity_name is not None:
            return "IDENTIFIED_FROM_DATASET"
        return "NOT_IN_DATASET"


_MAX_COUNTERPARTY_TX_REFERENCES = 5


def identify_counterparties(
    graph,
    wallet: str,
    registry: Optional[EntityRegistry] = None,
    limit: Optional[int] = None,
) -> list[Counterparty]:
    """Summarises the wallet's direct counterparties, naming the ones the
    dataset can identify.

    A counterparty with no dataset entry is reported as NOT_IN_DATASET, never
    as "not a VASP": this project's seed set is a small curated sample, so
    absence from it carries essentially no information about what an address
    is. Making that distinction explicit here is the same rule the ML module
    must follow when choosing negatives.
    """
    wallet_normalized = wallet.lower()
    if wallet_normalized not in graph:
        return []

    stats: dict[str, dict] = {}

    def touch(counterparty: str, data: dict, direction: str) -> None:
        bucket = stats.setdefault(
            counterparty,
            {
                "in": 0,
                "out": 0,
                "assets": set(),
                "first": None,
                "last": None,
                "hashes": set(),
                "contract": False,
            },
        )
        bucket[direction] += 1
        asset = data.get("asset")
        if asset:
            bucket["assets"].add(str(asset))
        ts = data.get("timestamp")
        if ts is not None:
            bucket["first"] = ts if bucket["first"] is None else min(bucket["first"], ts)
            bucket["last"] = ts if bucket["last"] is None else max(bucket["last"], ts)
        tx_hash = data.get("tx_hash")
        if tx_hash:
            bucket["hashes"].add(str(tx_hash))
        if data.get("is_contract_interaction"):
            bucket["contract"] = True

    seen_edges: set[tuple[str, str, str]] = set()
    for u, v, k, d in graph.out_edges(wallet_normalized, keys=True, data=True):
        identity = (u, v, str(k))
        if identity in seen_edges:
            continue
        seen_edges.add(identity)
        if v == wallet_normalized:
            continue  # self-transfer: not a counterparty
        touch(v, d, "out")
    for u, v, k, d in graph.in_edges(wallet_normalized, keys=True, data=True):
        identity = (u, v, str(k))
        if identity in seen_edges:
            continue
        seen_edges.add(identity)
        if u == wallet_normalized:
            continue
        touch(u, d, "in")

    counterparties: list[Counterparty] = []
    for address in sorted(stats):
        bucket = stats[address]
        entity = registry.entity_for_address(address) if registry else None
        entry = entity.address_entry(address) if entity else None
        counterparties.append(
            Counterparty(
                address=address,
                transfer_count=bucket["in"] + bucket["out"],
                inbound_count=bucket["in"],
                outbound_count=bucket["out"],
                assets=sorted(bucket["assets"]),
                first_seen=bucket["first"],
                last_seen=bucket["last"],
                tx_hashes=sorted(bucket["hashes"])[:_MAX_COUNTERPARTY_TX_REFERENCES],
                entity_name=entity.entity_name if entity else None,
                entity_type=entity.entity_type if entity else None,
                entity_source_type=entry.source_type if entry else None,
                wallet_role=entry.wallet_role if entry else None,
                is_contract=bucket["contract"],
            )
        )

    # Busiest first; ties on address so the order is stable across runs.
    counterparties.sort(key=lambda c: (-c.transfer_count, c.address))
    if limit is not None:
        return counterparties[:limit]
    return counterparties
