"""
MACRO MILESTONE 4 tests — VASP intelligence + explainable attribution.

Fully offline. Uses inline synthetic seed entries (source_type =
SYNTHETIC_DEMO) rather than the production data/seed/known_vasps.json,
so these tests never depend on — or risk being confused with — real
VASP data. The required end-to-end synthetic demonstration scenario
(UNKNOWN -> A -> B -> DEMO_VASP) is also covered here using the exact
addresses specified for the CLI demo.
"""

from __future__ import annotations

import json

import networkx as nx
import pytest

from app.attribution.candidate_generator import generate_candidates
from app.attribution.matcher import build_seed_index, match_address
from app.attribution.models import (
    AttributionStatus,
    EvidenceTier,
    SeedSourceType,
    VASPSeedEntry,
)
from app.attribution.seed_loader import SeedDataError, load_vasp_seed
from app.behavior.models import BehaviorPattern, PatternType
from app.tracing.tracer import trace_fund_flow

# Exact synthetic addresses specified for the required end-to-end demo.
UNKNOWN_WALLET = "0x1111111111111111111111111111111111111111"
INTERMEDIATE_A = "0x2222222222222222222222222222222222222222"
INTERMEDIATE_B = "0x3333333333333333333333333333333333333333"
DEMO_VASP = "0x4444444444444444444444444444444444444444"
DEMO_VASP_BETA = "0x5555555555555555555555555555555555555555"
NON_VASP_WALLET = "0x6666666666666666666666666666666666666666"


def add_edge(graph, u, v, tx_hash, occurrence=0, amount=1.0, ts=1_700_000_000, asset="ETH", asset_type="NATIVE"):
    key = f"{tx_hash}#{occurrence}"
    graph.add_edge(
        u,
        v,
        key=key,
        amount=amount,
        asset=asset,
        asset_type=asset_type,
        timestamp=ts,
        tx_hash=tx_hash,
        transfer_index=occurrence,
        chain="ethereum",
        transfer_type="TRANSFER" if asset_type == "NATIVE" else "TOKEN_TRANSFER",
        status="SUCCESS",
    )
    return key


def demo_seed_entry(address=DEMO_VASP, vasp_name="DEMO_VASP") -> VASPSeedEntry:
    return VASPSeedEntry(
        address=address,
        vasp_name=vasp_name,
        entity_type="demo_exchange",
        chain="ethereum",
        source="synthetic_demo_fixture",
        source_type=SeedSourceType.SYNTHETIC_DEMO,
        confidence_note="Synthetic address used only to validate the attribution pipeline. Not a real VASP.",
    )


# --- 1. Direct VASP: UNKNOWN -> DEMO_VASP => DIRECT -------------------------


def test_direct_one_hop_match_is_direct_tier():
    g = nx.MultiDiGraph()
    add_edge(g, UNKNOWN_WALLET, DEMO_VASP, "0xdirect1", ts=1000)

    trace = trace_fund_flow(g, UNKNOWN_WALLET, max_hops=3)
    seed_index = build_seed_index([demo_seed_entry()])
    result = generate_candidates(trace, seed_index)

    assert result.status == AttributionStatus.MATCH_FOUND
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.evidence_tier == EvidenceTier.DIRECT
    assert candidate.hop_distance == 1
    assert candidate.matched_address == DEMO_VASP
    assert candidate.vasp_name == "DEMO_VASP"
    assert candidate.source_type == SeedSourceType.SYNTHETIC_DEMO


# --- 2. Indirect VASP: UNKNOWN -> A -> B -> DEMO_VASP => INDIRECT, 3 hops ---


def test_indirect_three_hop_match_required_demo_scenario():
    g = nx.MultiDiGraph()
    add_edge(g, UNKNOWN_WALLET, INTERMEDIATE_A, "0xhop1", ts=1000)
    add_edge(g, INTERMEDIATE_A, INTERMEDIATE_B, "0xhop2", ts=1010)
    add_edge(g, INTERMEDIATE_B, DEMO_VASP, "0xhop3", ts=1020)

    trace = trace_fund_flow(g, UNKNOWN_WALLET, max_hops=4)
    seed_index = build_seed_index([demo_seed_entry()])
    result = generate_candidates(trace, seed_index)

    assert result.status == AttributionStatus.MATCH_FOUND
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.evidence_tier == EvidenceTier.INDIRECT
    assert candidate.hop_distance == 3
    assert candidate.path_addresses == [UNKNOWN_WALLET, INTERMEDIATE_A, INTERMEDIATE_B, DEMO_VASP]
    assert candidate.tx_hashes == ["0xhop1", "0xhop2", "0xhop3"]


# --- 3. Beyond MAX_HOPS: no candidate, status NONE (not INCONCLUSIVE) ------


def test_vasp_beyond_max_hops_is_none_not_inconclusive():
    g = nx.MultiDiGraph()
    add_edge(g, UNKNOWN_WALLET, INTERMEDIATE_A, "0xhop1", ts=1000)
    add_edge(g, INTERMEDIATE_A, INTERMEDIATE_B, "0xhop2", ts=1010)
    add_edge(g, INTERMEDIATE_B, DEMO_VASP, "0xhop3", ts=1020)  # 3 hops away

    trace = trace_fund_flow(g, UNKNOWN_WALLET, max_hops=2)  # search stops at 2
    seed_index = build_seed_index([demo_seed_entry()])
    result = generate_candidates(trace, seed_index)

    assert result.candidates == []
    assert result.status == AttributionStatus.NONE  # complete search within configured depth
    assert trace.paths_truncated is False
    assert trace.edges_limit_hit is False


# --- 4. No VASP reachable at all => NONE ------------------------------------


def test_no_vasp_reachable_is_none():
    g = nx.MultiDiGraph()
    add_edge(g, UNKNOWN_WALLET, NON_VASP_WALLET, "0xnope", ts=1000)

    trace = trace_fund_flow(g, UNKNOWN_WALLET, max_hops=3)
    seed_index = build_seed_index([demo_seed_entry()])
    result = generate_candidates(trace, seed_index)

    assert result.candidates == []
    assert result.status == AttributionStatus.NONE
    assert any("no known vasp" in n.lower() for n in result.notes)


# --- 5. Search limit reached => INCONCLUSIVE --------------------------------


def test_search_truncated_by_max_paths_is_inconclusive():
    g = nx.MultiDiGraph()
    # Fan-out of many non-VASP destinations so max_paths gets hit before
    # (if it ever would) reaching a VASP; DEMO_VASP is deliberately NOT
    # connected here, so the only way to get MATCH_FOUND would be luck —
    # this graph has no VASP edge at all, isolating the truncation signal.
    for i in range(20):
        add_edge(g, UNKNOWN_WALLET, f"0xdest{i:03d}" + "0" * 10, f"0xtx{i}", ts=1000 + i)

    trace = trace_fund_flow(g, UNKNOWN_WALLET, max_hops=1, max_paths=3)
    assert trace.paths_truncated is True

    seed_index = build_seed_index([demo_seed_entry()])
    result = generate_candidates(trace, seed_index)

    assert result.candidates == []
    assert result.status == AttributionStatus.INCONCLUSIVE
    assert result.search_truncated is True
    assert any("inconclusive" in n.lower() or "resource limit" in n.lower() for n in result.notes)


def test_search_truncated_by_max_edges_is_inconclusive():
    g = nx.MultiDiGraph()
    for i in range(20):
        add_edge(g, UNKNOWN_WALLET, f"0xdest{i:03d}" + "0" * 10, f"0xtx{i}", ts=1000 + i)

    trace = trace_fund_flow(g, UNKNOWN_WALLET, max_hops=1, max_paths=1000, max_edges_explored=3)
    assert trace.edges_limit_hit is True

    seed_index = build_seed_index([demo_seed_entry()])
    result = generate_candidates(trace, seed_index)

    assert result.status == AttributionStatus.INCONCLUSIVE


# --- 6. Deterministic candidate ordering ------------------------------------


def test_two_different_vasps_deterministic_ordering():
    g = nx.MultiDiGraph()
    add_edge(g, UNKNOWN_WALLET, DEMO_VASP_BETA, "0xtoBeta", ts=1000)  # 1 hop
    add_edge(g, UNKNOWN_WALLET, INTERMEDIATE_A, "0xhop1", ts=1000)
    add_edge(g, INTERMEDIATE_A, DEMO_VASP, "0xhop2", ts=1010)  # 2 hops

    trace = trace_fund_flow(g, UNKNOWN_WALLET, max_hops=3)
    seed_index = build_seed_index([demo_seed_entry(), demo_seed_entry(DEMO_VASP_BETA, "DEMO_VASP_BETA")])

    result1 = generate_candidates(trace, seed_index)
    result2 = generate_candidates(trace, seed_index)

    assert len(result1.candidates) == 2
    # Shortest hop distance first (DIRECT before INDIRECT), deterministic.
    assert result1.candidates[0].hop_distance == 1
    assert result1.candidates[0].matched_address == DEMO_VASP_BETA
    assert result1.candidates[1].hop_distance == 2
    assert [c.matched_address for c in result1.candidates] == [c.matched_address for c in result2.candidates]


# --- 7/8. evidence path + tx hash preservation ------------------------------


def test_evidence_path_and_tx_hashes_preserved():
    g = nx.MultiDiGraph()
    add_edge(g, UNKNOWN_WALLET, INTERMEDIATE_A, "0xaaa", ts=500)
    add_edge(g, INTERMEDIATE_A, DEMO_VASP, "0xbbb", ts=600)

    trace = trace_fund_flow(g, UNKNOWN_WALLET, max_hops=3)
    seed_index = build_seed_index([demo_seed_entry()])
    result = generate_candidates(trace, seed_index)

    candidate = result.candidates[0]
    assert candidate.path_addresses == [UNKNOWN_WALLET, INTERMEDIATE_A, DEMO_VASP]
    assert candidate.tx_hashes == ["0xaaa", "0xbbb"]
    assert candidate.hop_timestamps == [500, 600]
    assert candidate.evidence_status == "TRACEABLE"


# --- 9. Parallel MultiDiGraph edges do not collapse -------------------------


def test_parallel_edges_to_same_vasp_do_not_collapse_and_shortest_wins():
    g = nx.MultiDiGraph()
    # Two separate direct transfers to the VASP (parallel edges, distinct tx)
    add_edge(g, UNKNOWN_WALLET, DEMO_VASP, "0xp1", occurrence=0, ts=1000)
    add_edge(g, UNKNOWN_WALLET, DEMO_VASP, "0xp2", occurrence=0, ts=1010)

    trace = trace_fund_flow(g, UNKNOWN_WALLET, max_hops=2)
    # Both parallel 1-hop paths must be present in the trace itself.
    one_hop_paths = [p for p in trace.paths if p.hop_count == 1]
    assert len(one_hop_paths) == 2
    assert {p.hops[0].tx_hash for p in one_hop_paths} == {"0xp1", "0xp2"}

    seed_index = build_seed_index([demo_seed_entry()])
    result = generate_candidates(trace, seed_index)
    # Candidate generation dedupes to ONE candidate per distinct VASP
    # address (best evidence = shortest hop), not one per parallel edge.
    assert len(result.candidates) == 1
    assert result.candidates[0].hop_distance == 1


# --- 10. Synthetic dataset separation ---------------------------------------


def test_synthetic_seed_marked_and_never_mixed_with_production_by_default():
    entry = demo_seed_entry()
    assert entry.source_type == SeedSourceType.SYNTHETIC_DEMO
    assert entry.source == "synthetic_demo_fixture"

    # The production seed file must not contain the demo address.
    with open("data/seed/known_vasps.json") as f:
        production = json.load(f)
    production_addresses = {e["address"].lower() for e in production["seed_addresses"]}
    assert DEMO_VASP.lower() not in production_addresses
    # Every production entry must declare a real provenance from the
    # taxonomy, and none may be synthetic. Asserted as "not synthetic + is a
    # recognised provenance" rather than one literal value, so entries can be
    # graded honestly (community label vs official disclosure) without the
    # test forcing them all to look equally strong.
    for entry in production["seed_addresses"]:
        source_type = SeedSourceType(entry["source_type"])
        assert not source_type.is_synthetic
        assert source_type is not SeedSourceType.UNVERIFIED

    # The demo seed file must contain only synthetic entries.
    with open("data/seed/demo_known_vasps.json") as f:
        demo = json.load(f)
    assert all(e["source_type"] == "synthetic_demo" for e in demo["seed_addresses"])


# --- 11. Malformed seed data fails loudly -----------------------------------


def test_malformed_seed_json_raises(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json")
    with pytest.raises(SeedDataError):
        load_vasp_seed(bad_file)


def test_seed_entry_missing_required_field_raises(tmp_path):
    bad_file = tmp_path / "bad_entry.json"
    bad_file.write_text(json.dumps({"seed_addresses": [{"address": DEMO_VASP}]}))
    with pytest.raises(SeedDataError):
        load_vasp_seed(bad_file)


def test_seed_file_not_found_raises(tmp_path):
    with pytest.raises(SeedDataError):
        load_vasp_seed(tmp_path / "does_not_exist.json")


def test_ambiguous_duplicate_address_raises(tmp_path):
    dup_file = tmp_path / "dup.json"
    dup_file.write_text(
        json.dumps(
            {
                "seed_addresses": [
                    {
                        "address": DEMO_VASP,
                        "vasp_name": "DEMO_VASP",
                        "entity_type": "demo_exchange",
                        "chain": "ethereum",
                        "source": "synthetic_demo_fixture",
                        "source_type": "synthetic_demo",
                        "confidence_note": "x",
                    },
                    {
                        "address": DEMO_VASP,
                        "vasp_name": "DIFFERENT_NAME",  # ambiguous!
                        "entity_type": "demo_exchange",
                        "chain": "ethereum",
                        "source": "synthetic_demo_fixture",
                        "source_type": "synthetic_demo",
                        "confidence_note": "x",
                    },
                ]
            }
        )
    )
    with pytest.raises(SeedDataError):
        load_vasp_seed(dup_file)


def test_exact_duplicate_seed_entry_handled_safely(tmp_path):
    dup_file = tmp_path / "exact_dup.json"
    entry = {
        "address": DEMO_VASP,
        "vasp_name": "DEMO_VASP",
        "entity_type": "demo_exchange",
        "chain": "ethereum",
        "source": "synthetic_demo_fixture",
        "source_type": "synthetic_demo",
        "confidence_note": "x",
    }
    dup_file.write_text(json.dumps({"seed_addresses": [entry, dict(entry)]}))
    entries = load_vasp_seed(dup_file)
    assert len(entries) == 1  # exact duplicate silently deduped, no error


# --- production seed loads correctly -----------------------------------


def test_production_seed_file_loads_without_error():
    entries = load_vasp_seed("data/seed/known_vasps.json")
    assert len(entries) >= 1
    # Real provenance on every entry: nothing synthetic, nothing unsourced,
    # and a citable URL for each so a reviewer can check the label itself.
    assert not any(e.source_type.is_synthetic for e in entries)
    assert all(e.source_type is not SeedSourceType.UNVERIFIED for e in entries)
    assert all(e.source_url for e in entries)
    assert all(e.confidence_note.strip() for e in entries)
    # The taxonomy must actually be used to differentiate strength — an
    # operator's own disclosure and a crowd-sourced label must not be
    # flattened into one undifferentiated "sourced" bucket.
    assert any(
        e.source_type is SeedSourceType.OFFICIAL_DISCLOSURE for e in entries
    )
    assert len({e.source_type for e in entries}) > 1


def test_demo_seed_file_loads_without_error():
    entries = load_vasp_seed("data/seed/demo_known_vasps.json")
    assert len(entries) >= 1
    assert all(e.source_type == SeedSourceType.SYNTHETIC_DEMO for e in entries)


# --- behavioral patterns cannot create a candidate by themselves -----------


def test_behavioral_pattern_alone_cannot_create_candidate():
    g = nx.MultiDiGraph()
    # Wallet with a strong SPLIT_PATTERN-style fan-out, but NONE of the
    # destinations are a known VASP.
    for i in range(6):
        add_edge(g, UNKNOWN_WALLET, f"0xdest{i:03d}" + "0" * 10, f"0xtx{i}", ts=1000 + i)

    trace = trace_fund_flow(g, UNKNOWN_WALLET, max_hops=1)
    fake_pattern = BehaviorPattern(
        pattern_type=PatternType.SPLIT_PATTERN,
        wallet=UNKNOWN_WALLET,
        evidence=["6 unique outgoing counterparties"],
        metrics={"unique_outgoing_counterparties": 6},
        related_addresses=[f"0xdest{i:03d}" + "0" * 10 for i in range(6)],
    )
    seed_index = build_seed_index([demo_seed_entry()])  # DEMO_VASP not touched by any edge
    result = generate_candidates(trace, seed_index, behavior_patterns=[fake_pattern])

    assert result.candidates == []
    assert result.status == AttributionStatus.NONE


def test_behavioral_pattern_annotates_existing_candidate_only():
    g = nx.MultiDiGraph()
    add_edge(g, UNKNOWN_WALLET, DEMO_VASP, "0xdirect", ts=1000)

    trace = trace_fund_flow(g, UNKNOWN_WALLET, max_hops=1)
    overlapping_pattern = BehaviorPattern(
        pattern_type=PatternType.REPEATED_FORWARDING,
        wallet=UNKNOWN_WALLET,
        evidence=["some evidence"],
        metrics={"forwarding_event_count": 2},
        related_addresses=[DEMO_VASP],
    )
    seed_index = build_seed_index([demo_seed_entry()])
    result = generate_candidates(trace, seed_index, behavior_patterns=[overlapping_pattern])

    assert len(result.candidates) == 1
    assert "REPEATED_FORWARDING" in result.candidates[0].supporting_behavioral_patterns


# --- matcher exactness -------------------------------------------------


def test_matcher_is_exact_and_case_insensitive():
    seed_index = build_seed_index([demo_seed_entry()])
    assert match_address(DEMO_VASP.upper(), seed_index) is not None
    assert match_address(DEMO_VASP, seed_index) is not None
    assert match_address(NON_VASP_WALLET, seed_index) is None


# --- 12. existing M3 tests remain passing: verified by running full suite --
