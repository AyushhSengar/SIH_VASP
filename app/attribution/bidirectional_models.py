"""
MACRO MILESTONE 6, PHASE 1 — bidirectional attribution models.

--------------------------------------------------------------------------
ADDITIVE, NOT A REPLACEMENT (do not remove): AttributionResult, VASPCandidate,
EvidenceTier, and generate_candidates() (app/attribution/models.py,
app/attribution/candidate_generator.py) are UNCHANGED by Phase 1 — every
Macro Milestone 4 test and consumer keeps working exactly as before,
because nothing here modifies them. BidirectionalAttributionResult is a
new, separate output produced by a new function
(generate_bidirectional_candidates, app/attribution/bidirectional.py)
that internally reuses generate_candidates() for the outbound half of
its work rather than reimplementing it.

GRAPH CONNECTIVITY IS NOT EVIDENCE (do not remove): a wallet and a known
VASP address being reachable from each other in the graph's UNDIRECTED
projection is a relationship-discovery signal only. It says nothing
about the direction, or even the existence, of an actual transfer
between them (the undirected projection erases direction entirely).
UndirectedRelation entries are therefore NEVER promoted to
BidirectionalCandidate and never contribute to AttributionStatus —
see generate_bidirectional_candidates()'s docstring.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel

from app.attribution.models import AttributionStatus, EvidenceTier, SeedSourceType
from app.tracing.quality import PathPlausibility


class ConnectionDirection(str, Enum):
    """How a known VASP address relates to the investigated wallet,
    directionally, within the configured search depth."""

    DIRECT_OUTBOUND = "DIRECT_OUTBOUND"  # wallet -> VASP, 1 hop
    INDIRECT_OUTBOUND = "INDIRECT_OUTBOUND"  # wallet -> ... -> VASP, >1 hop
    DIRECT_INBOUND = "DIRECT_INBOUND"  # VASP -> wallet, 1 hop
    INDIRECT_INBOUND = "INDIRECT_INBOUND"  # VASP -> ... -> wallet, >1 hop
    BIDIRECTIONAL = "BIDIRECTIONAL"  # both a directed outbound AND inbound path exist


class DirectionalEvidence(BaseModel):
    """One direction's concrete, traceable path — never present unless a
    real directed graph path was found by the targeted search.

    Everything needed to re-verify the claim against the chain is carried
    here (hashes, blocks, timestamps, amounts, assets, edge keys), so a
    reviewer never has to take the conclusion on trust. `plausibility`
    states whether the observable evidence actually supports reading this
    path as a movement of funds — see app/tracing/quality.py.
    """

    hop_distance: int
    path_addresses: list[str]
    tx_hashes: list[str]
    hop_timestamps: list[Optional[int]]

    # --- Additive: full per-hop evidence and honest grading. ---
    # Defaulted so any existing construction site keeps working.
    evidence_tier: Optional[EvidenceTier] = None
    amounts: list[float] = []
    assets: list[str] = []
    block_numbers: list[Optional[int]] = []
    edge_keys: list[str] = []
    path_duration_seconds: Optional[int] = None
    plausibility: Optional[PathPlausibility] = None
    # How many other traced paths existed in this direction. >1 means the
    # reported path is one example, not the only connection.
    alternative_path_count: int = 0


class BidirectionalCandidate(BaseModel):
    vasp_name: str
    matched_address: str
    entity_type: str
    chain: str

    source_type: SeedSourceType
    seed_source: str
    seed_source_url: Optional[str] = None
    seed_confidence_note: str

    # --- Additive seed provenance, surfaced verbatim from the seed entry so
    # a report can state what the label actually rests on. Never fabricated:
    # absent in the dataset means None here, not an invented value. ---
    wallet_role: Optional[str] = None
    source_evidence_type: Optional[str] = None
    verification_status: Optional[str] = None

    direction: ConnectionDirection
    # At least one of these is always populated; both are populated only
    # when direction == BIDIRECTIONAL.
    outbound_evidence: Optional[DirectionalEvidence] = None  # wallet -> ... -> VASP
    inbound_evidence: Optional[DirectionalEvidence] = None  # VASP -> ... -> wallet

    supporting_behavioral_patterns: list[str] = []  # annotation only, see M4 rule

    # Explicit, per-candidate statement of what this evidence does NOT show.
    # Populated by the generator; never empty for a multi-hop candidate.
    limitations: list[str] = []

    evidence_status: Literal["TRACEABLE"] = "TRACEABLE"

    @property
    def strongest_hop_distance(self) -> int:
        distances = [
            e.hop_distance
            for e in (self.outbound_evidence, self.inbound_evidence)
            if e is not None
        ]
        return min(distances) if distances else 0


class ConnectedButNoValidPath(BaseModel):
    """A known VASP address that IS connected to the wallet by directed
    edges within the hop limit, but where no chronologically-consistent
    path exists — funds cannot leave an intermediary before arriving.

    Kept out of `candidates` because it is not a fund-flow candidate, and
    kept out of silence because it is a real, reportable finding: the
    addresses ARE transactionally connected.
    """

    vasp_name: str
    vasp_address: str
    direction_attempted: str  # "OUTBOUND" or "INBOUND"
    graph_distance: int
    note: str = (
        "Directed edges connect these addresses within the hop limit, but "
        "every route was rejected because its transfer timestamps run "
        "backwards. Reported as a transactional connection, NOT as a "
        "fund-flow candidate."
    )


class ExactIdentityMatch(BaseModel):
    """The investigated wallet IS itself a known VASP address.

    This is an exact address identity, not a traced path, so it has no
    direction and no hops — and it is the strongest form of address-level
    evidence there is. Kept in its own field rather than squeezed into
    `candidates` (which model directional connections) so a report can never
    describe it as a fund flow.
    """

    vasp_name: str
    matched_address: str
    entity_type: str
    chain: str
    source_type: SeedSourceType
    seed_source: str
    seed_source_url: Optional[str] = None
    seed_confidence_note: str
    wallet_role: Optional[str] = None
    source_evidence_type: Optional[str] = None
    verification_status: Optional[str] = None
    note: str = (
        "The investigated address is itself present in the known-VASP "
        "dataset. This is an exact address match, not a traced path. Its "
        "strength is entirely the strength of the dataset entry's own "
        "provenance — see source_type and verification_status."
    )


class UndirectedRelation(BaseModel):
    """A known VASP address reachable from the wallet ONLY in the graph's
    undirected projection — no directed transactional evidence exists in
    either direction. This is a relationship-discovery signal only and is
    deliberately kept out of `candidates`/AttributionStatus."""

    vasp_name: str
    vasp_address: str
    undirected_distance: int
    note: str = (
        "Connected only in the undirected graph projection - no directed "
        "transactional evidence exists in either direction between this "
        "wallet and this address. This is a relationship-discovery signal "
        "only, NOT attribution evidence, and does not affect the "
        "attribution status above."
    )


class SearchAccounting(BaseModel):
    """What the search actually did, per direction. Present so a report can
    substantiate a negative result instead of merely asserting it."""

    direction: str
    graph_node_count: int = 0
    graph_edge_count: int = 0
    reachable_node_count: int = 0
    viable_node_count: int = 0
    edges_explored: int = 0
    complete: bool = True
    # Why the search was not complete, when it was not. Two causes are
    # possible and they call for opposite remedies: a budget stop means
    # re-run with a larger budget, a data horizon means acquire more data.
    # Reporting one as the other sends the reader the wrong way.
    incomplete_reason: Optional[str] = None
    observation_depth: Optional[int] = None
    targets_searched: int = 0
    targets_reachable: int = 0
    time_window_start: Optional[int] = None
    edges_excluded_by_time_window: int = 0
    notes: list[str] = []


class BidirectionalAttributionResult(BaseModel):
    wallet: str
    status: AttributionStatus  # same MATCH_FOUND/NONE/INCONCLUSIVE semantics as M4

    candidates: list[BidirectionalCandidate]
    related_by_undirected_graph_only: list[UndirectedRelation] = []
    connected_but_no_valid_path: list[ConnectedButNoValidPath] = []
    # Populated only when the investigated wallet is itself a seed address.
    exact_identity_match: Optional[ExactIdentityMatch] = None

    max_hops: int
    outbound_search_truncated: bool
    inbound_searches_truncated: bool  # True if the reverse search hit a resource limit

    # --- Additive accounting + dataset provenance. ---
    outbound_accounting: Optional[SearchAccounting] = None
    inbound_accounting: Optional[SearchAccounting] = None
    seed_address_count: int = 0
    # True when the seed set contains any SYNTHETIC_DEMO entry, so a report
    # can never present demo data as a real-world finding by accident.
    seed_contains_synthetic: bool = False

    notes: list[str] = []
