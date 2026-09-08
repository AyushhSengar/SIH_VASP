"""
Output models for Macro Milestone 4 (VASP intelligence + explainable
attribution).

--------------------------------------------------------------------------
NO FALSE CERTAINTY (do not remove): a VASPCandidate is evidence that a
known VASP address was reached from the investigated wallet through the
graph — never a claim of ownership. Behavioral patterns may only appear
as `supporting_behavioral_patterns` annotations on a candidate that
already exists from address-level evidence; nothing in this module may
create a candidate from behavior alone. There is deliberately no numeric
confidence score anywhere here (e.g. "87%") — only the four discrete,
evidence-backed statuses defined in AttributionStatus.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel


class SeedSourceType(str, Enum):
    """Provenance of a known-VASP dataset entry — what the ownership claim
    actually rests on.

    This distinction is load-bearing, not decorative: a third-party block
    explorer label is NOT an official statement of ownership by the VASP,
    and an evidence report must never present the two as equivalent. The
    order below is strongest-first, and that order is exposed via
    `strength_rank` so reporting code never has to hardcode its own ranking.

    SYNTHETIC_DEMO is separated at the type level (not merely by convention
    in the `source` string) so downstream code can never accidentally treat
    demo data as real.
    """

    # The VASP itself published this address (proof-of-reserves disclosure,
    # official documentation, signed attestation).
    OFFICIAL_DISCLOSURE = "official_disclosure"
    # Independently verified against a primary source or an on-chain proof
    # (e.g. a signed message from the address) during this project.
    DIRECTLY_VERIFIED = "directly_verified"
    # Published by a reputable, editorially-controlled public source — not
    # the VASP itself, and not crowd-sourced.
    PUBLIC_LABEL = "public_label"
    # A commercial analytics provider's or block explorer's own label.
    THIRD_PARTY_LABEL = "third_party_label"
    # Crowd-sourced / community-submitted tagging (e.g. Etherscan's public
    # address labels). Can be stale or wrong; an investigative lead only.
    COMMUNITY_LABEL = "community_label"
    # Derived by analysis (clustering, behaviour) rather than labeled by
    # anyone. Never sufficient for an ownership claim on its own.
    INFERRED = "inferred"
    # Present in the dataset with no usable provenance recorded.
    UNVERIFIED = "unverified"

    # Synthetic addresses for tests and explicit demo runs. NEVER production.
    SYNTHETIC_DEMO = "synthetic_demo"

    # Retained so seed files written before the taxonomy above still load
    # unchanged. Semantically equivalent to COMMUNITY_LABEL/PUBLIC_LABEL but
    # too coarse to distinguish them, so it ranks with the weaker of the two.
    SOURCED_PUBLIC_LABEL = "sourced_public_label"

    @property
    def strength_rank(self) -> int:
        """Lower is stronger evidence of ownership. Synthetic entries are
        deliberately ranked last so they can never outrank real data."""
        return _SEED_SOURCE_STRENGTH[self]

    @property
    def is_synthetic(self) -> bool:
        return self is SeedSourceType.SYNTHETIC_DEMO


_SEED_SOURCE_STRENGTH = {
    SeedSourceType.OFFICIAL_DISCLOSURE: 0,
    SeedSourceType.DIRECTLY_VERIFIED: 1,
    SeedSourceType.PUBLIC_LABEL: 2,
    SeedSourceType.THIRD_PARTY_LABEL: 3,
    SeedSourceType.SOURCED_PUBLIC_LABEL: 4,
    SeedSourceType.COMMUNITY_LABEL: 4,
    SeedSourceType.INFERRED: 5,
    SeedSourceType.UNVERIFIED: 6,
    SeedSourceType.SYNTHETIC_DEMO: 99,
}


class VASPSeedEntry(BaseModel):
    address: str
    vasp_name: str
    entity_type: str  # e.g. "exchange", "demo_exchange"
    chain: str
    source: str
    source_type: SeedSourceType
    confidence_note: str
    source_url: Optional[str] = None  # present for sourced_public_label entries

    # --- Phase 1 (Macro Milestone 6) additive metadata ---
    # All optional, all default None/absent so every existing seed JSON file
    # (production and demo) continues to validate unchanged — adding this
    # metadata to an entry is a pure JSON edit, never a code change, per
    # "adding a new VASP address should NOT require changing Python logic."
    wallet_role: Optional[str] = None  # e.g. "hot_wallet", "cold_wallet", "deposit_address"
    source_evidence_type: Optional[str] = None  # e.g. "community_label", "official_por_disclosure"
    verification_status: Optional[str] = None  # e.g. "unverified", "third_party_labeled" — never fabricated when unknown


class EvidenceTier(str, Enum):
    DIRECT = "DIRECT"  # known VASP reached at exactly 1 hop
    INDIRECT = "INDIRECT"  # known VASP reached at >1 hop, within MAX_HOPS


class AttributionStatus(str, Enum):
    """Overall result for one wallet's attribution run.

    MATCH_FOUND: at least one VASPCandidate was generated.
    NONE: the trace fully explored the configured MAX_HOPS search depth
        (no MAX_PATHS/MAX_EDGES_EXPLORED truncation) and found no known
        VASP address. A complete negative result.
    INCONCLUSIVE: no candidate was found, but the trace was cut short by
        a resource safety limit (MAX_PATHS or MAX_EDGES_EXPLORED) before
        the configured MAX_HOPS depth could be fully examined. This is
        NOT the same as NONE — the search space was not fully covered,
        so absence of evidence is not evidence of absence here.

    Reaching MAX_HOPS itself (the intentional, configured search-depth
    boundary) does NOT make a result INCONCLUSIVE — a VASP beyond
    MAX_HOPS is outside the investigation's defined scope by design, so
    that case is NONE, not INCONCLUSIVE. Only the two *resource* limits
    (MAX_PATHS, MAX_EDGES_EXPLORED) can produce INCONCLUSIVE.
    """

    MATCH_FOUND = "MATCH_FOUND"
    NONE = "NONE"
    INCONCLUSIVE = "INCONCLUSIVE"


class VASPCandidate(BaseModel):
    vasp_name: str
    matched_address: str
    entity_type: str
    chain: str

    source_type: SeedSourceType
    seed_source: str
    seed_source_url: Optional[str] = None
    seed_confidence_note: str

    evidence_tier: EvidenceTier
    hop_distance: int

    path_addresses: list[str]
    tx_hashes: list[str]
    hop_timestamps: list[Optional[int]]

    supporting_behavioral_patterns: list[str] = []  # pattern_type values only

    # Always TRACEABLE by construction — generate_candidates() never emits
    # a candidate without a concrete path/tx_hash trail (see its docstring).
    evidence_status: Literal["TRACEABLE"] = "TRACEABLE"


class AttributionResult(BaseModel):
    wallet: str
    status: AttributionStatus
    candidates: list[VASPCandidate]
    max_hops: int
    search_truncated: bool  # True if MAX_PATHS or MAX_EDGES_EXPLORED was hit
    notes: list[str] = []
