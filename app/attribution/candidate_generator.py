"""
MACRO MILESTONE 4 — evidence-based VASP candidate generation.

--------------------------------------------------------------------------
Consumes app.tracing's FundFlowPath objects (Phase A / M3) directly —
this module does not re-walk the graph itself, does not introduce a
second hop-limit, and cannot disagree with what M3 actually traced.

Behavioral patterns (M3 Phase B) may only ANNOTATE a candidate that
already exists from address-level path evidence; see
`_supporting_pattern_types`. Nothing here creates a candidate from
behavior alone.
--------------------------------------------------------------------------

DIRECT / INDIRECT / NONE / INCONCLUSIVE semantics (see
app/attribution/models.py::AttributionStatus for the full docstring):
  - A path's terminal node matching a known VASP address at hop 1 is
    DIRECT; at >1 hop (still within FUND_TRACE_MAX_HOPS, since M3 never
    produces paths beyond it) is INDIRECT.
  - If no candidates are found AND the trace was NOT cut short by a
    resource limit (MAX_PATHS / MAX_EDGES_EXPLORED), the result is NONE
    — a complete negative within the configured search depth.
  - If no candidates are found AND the trace WAS cut short by a resource
    limit, the result is INCONCLUSIVE — the search space was not fully
    covered, so this must not be reported as a clean negative.
"""

from __future__ import annotations

from app.attribution.matcher import match_address
from app.attribution.models import (
    AttributionResult,
    AttributionStatus,
    EvidenceTier,
    VASPCandidate,
    VASPSeedEntry,
)
from app.behavior.models import BehaviorPattern
from app.tracing.models import FundFlowPath, TraceResult


def _supporting_pattern_types(
    path_addresses: list[str], behavior_patterns: list[BehaviorPattern]
) -> list[str]:
    """Behavioral patterns whose wallet or related_addresses overlap this
    candidate's already-established path — annotation only, never a
    trigger for candidate creation."""
    path_set = set(path_addresses)
    matched = {
        p.pattern_type.value
        for p in behavior_patterns
        if p.wallet in path_set or (path_set & set(p.related_addresses))
    }
    return sorted(matched)


def generate_candidates(
    trace_result: TraceResult,
    seed_index: dict[str, VASPSeedEntry],
    behavior_patterns: list[BehaviorPattern] | None = None,
) -> AttributionResult:
    behavior_patterns = behavior_patterns or []

    # Keep only the shortest (best-evidence) path per distinct matched
    # VASP address — a 1-hop DIRECT hit is stronger evidence than a
    # longer INDIRECT path to the same address, and M3's tracer already
    # returns every hop-prefix, so the same terminal address can appear
    # at multiple hop counts.
    best_by_address: dict[str, FundFlowPath] = {}
    seed_entry_by_address: dict[str, VASPSeedEntry] = {}

    for path in trace_result.paths:
        seed_entry = match_address(path.terminal_node, seed_index)
        if seed_entry is None:
            continue
        key = seed_entry.address.lower()
        existing = best_by_address.get(key)
        if existing is None or path.hop_count < existing.hop_count:
            best_by_address[key] = path
            seed_entry_by_address[key] = seed_entry

    candidates: list[VASPCandidate] = []
    for key, path in best_by_address.items():
        seed_entry = seed_entry_by_address[key]

        path_addresses = path.addresses
        tx_hashes = [h.tx_hash for h in path.hops]
        hop_timestamps = [h.timestamp for h in path.hops]

        # Guard per spec section 7: never emit a candidate without a
        # concrete, traceable path/tx_hash trail.
        if not path_addresses or not tx_hashes:
            continue

        tier = EvidenceTier.DIRECT if path.hop_count == 1 else EvidenceTier.INDIRECT

        candidates.append(
            VASPCandidate(
                vasp_name=seed_entry.vasp_name,
                matched_address=seed_entry.address,
                entity_type=seed_entry.entity_type,
                chain=seed_entry.chain,
                source_type=seed_entry.source_type,
                seed_source=seed_entry.source,
                seed_source_url=seed_entry.source_url,
                seed_confidence_note=seed_entry.confidence_note,
                evidence_tier=tier,
                hop_distance=path.hop_count,
                path_addresses=path_addresses,
                tx_hashes=tx_hashes,
                hop_timestamps=hop_timestamps,
                supporting_behavioral_patterns=_supporting_pattern_types(
                    path_addresses, behavior_patterns
                ),
            )
        )

    # Deterministic ordering: strongest evidence (shortest hop distance)
    # first, tie-broken by matched address.
    candidates.sort(key=lambda c: (c.hop_distance, c.matched_address))

    search_truncated = trace_result.paths_truncated or trace_result.edges_limit_hit

    notes: list[str] = []
    if candidates:
        status = AttributionStatus.MATCH_FOUND
    elif search_truncated:
        status = AttributionStatus.INCONCLUSIVE
        notes.append(
            "No known VASP address was found, but the fund-flow search was "
            "cut short by a resource limit (MAX_PATHS or MAX_EDGES_EXPLORED) "
            f"before the configured MAX_HOPS={trace_result.max_hops} depth "
            "could be fully examined. This is inconclusive, not a confirmed "
            "negative — widen the limits and re-run for a complete answer."
        )
    else:
        status = AttributionStatus.NONE
        notes.append(
            f"No known VASP seed address was reached from "
            f"'{trace_result.source}' within the fully-examined "
            f"MAX_HOPS={trace_result.max_hops} search depth."
        )

    return AttributionResult(
        wallet=trace_result.source,
        status=status,
        candidates=candidates,
        max_hops=trace_result.max_hops,
        search_truncated=search_truncated,
        notes=notes,
    )
