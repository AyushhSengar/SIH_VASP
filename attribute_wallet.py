"""
MACRO MILESTONE 4 CLI — evidence-based VASP candidate attribution.

Usage:
    python attribute_wallet.py <wallet_address> [--max-hops 3] [--chain ethereum]
    python attribute_wallet.py <wallet_address> --graph data/graphs/<file>.gpickle
    python attribute_wallet.py <wallet_address> --demo   # use the synthetic seed set
    python attribute_wallet.py <wallet_address> --seed <path>  # explicit seed override

Reuses app.tracing.trace_fund_flow and app.behavior.analyze_wallet_behavior
(Macro Milestone 3) completely unmodified — this script only adds the
VASP-matching and candidate-generation layer on top. Never fetches
anything from Etherscan; operates entirely on an already-cached
.gpickle graph (see build_graph.py).

IMPORTANT: --demo loads data/seed/demo_known_vasps.json, a synthetic
dataset that exists only to prove the attribution pipeline works end to
end. It is NEVER loaded by default, and every synthetic result printed
by this script is clearly banner-marked as such.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.attribution.candidate_generator import generate_candidates
from app.attribution.matcher import build_seed_index
from app.attribution.models import AttributionStatus, SeedSourceType, VASPCandidate
from app.attribution.seed_loader import SeedDataError, load_vasp_seed
from app.behavior.detectors import analyze_wallet_behavior
from app.core.config import get_settings
from app.graph.builder import load_graph
from app.tracing.tracer import trace_fund_flow
from app.reporting.terminal import configure_stdout

def graphs_dir() -> Path:
    """The graph cache directory, from GRAPH_CACHE_DIR rather than a constant.

    A hardcoded "data/graphs" here meant this script and investigate.py could
    write to and read from different places once GRAPH_CACHE_DIR was set, so a
    graph built by one was invisible to the other. Read at call time, not
    import time, so an env change inside one process still takes effect.
    """
    return Path(get_settings().graph_cache_dir)


def resolve_graph_path(wallet: str | None, chain: str, graph_arg: str | None) -> Path:
    if graph_arg:
        return Path(graph_arg)
    if not wallet:
        raise ValueError("Provide either a wallet address or --graph <path>.")
    return graphs_dir() / f"{wallet.lower()}_{chain}.gpickle"


def print_candidate(candidate: VASPCandidate) -> None:
    print("\n" + "-" * 50)
    if candidate.source_type == SeedSourceType.SYNTHETIC_DEMO:
        print("*** SYNTHETIC DEMONSTRATION — NOT A REAL VASP ***")
    print(f"\nCandidate VASP:      {candidate.vasp_name} ({candidate.entity_type})")
    print(f"Matched address:     {candidate.matched_address}")
    print(f"Evidence tier:       {candidate.evidence_tier.value}")
    print(f"Hop distance:        {candidate.hop_distance}")
    print("\nPath:")
    print("\n   |\n   v\n".join(candidate.path_addresses))
    print(f"\nTransactions:        {', '.join(candidate.tx_hashes)}")
    print(f"Hop timestamps:      {candidate.hop_timestamps}")
    print(f"\nSeed source:         {candidate.seed_source}")
    if candidate.seed_source_url:
        print(f"Seed source URL:     {candidate.seed_source_url}")
    print(f"Seed confidence note: {candidate.seed_confidence_note}")
    if candidate.supporting_behavioral_patterns:
        print(
            "\nSupporting behavioral patterns: "
            f"{', '.join(candidate.supporting_behavioral_patterns)}"
        )
        print(
            "(Supporting evidence only — behavioral patterns did not "
            "create this candidate; the address-level path match did.)"
        )
    print(f"\nEvidence status:     {candidate.evidence_status}")


def main() -> None:
    # These scripts print typographic dashes, which become mojibake on a
    # cp1252 Windows console. Reuse the production renderer's stream setup
    # rather than restating the strings.
    configure_stdout()
    parser = argparse.ArgumentParser(
        description="Macro Milestone 4: evidence-based VASP candidate attribution"
    )
    parser.add_argument("wallet", nargs="?", help="Wallet address matching an existing cached graph")
    parser.add_argument("--chain", default="ethereum")
    parser.add_argument("--graph", help="Explicit path to a cached .gpickle graph")
    parser.add_argument("--max-hops", type=int, default=None, help="Override FUND_TRACE_MAX_HOPS for this run")
    parser.add_argument("--max-paths", type=int, default=None, help="Override FUND_TRACE_MAX_PATHS for this run")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the synthetic demo VASP seed set instead of the production one",
    )
    parser.add_argument("--seed", help="Explicit path to a VASP seed JSON file (overrides --demo)")
    args = parser.parse_args()

    try:
        graph_path = resolve_graph_path(args.wallet, args.chain, args.graph)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not args.wallet:
        print("ERROR: a wallet address is required (used as the trace source), even with --graph.")
        sys.exit(1)

    if not graph_path.exists():
        print(
            f"ERROR: cached graph not found at {graph_path}. "
            "Run build_graph.py for this wallet first (Milestone 2)."
        )
        sys.exit(1)

    wallet = args.wallet.lower()
    settings = get_settings()

    seed_path = args.seed or (
        settings.vasp_demo_seed_dataset_path if args.demo else settings.vasp_seed_dataset_path
    )

    try:
        seed_entries = load_vasp_seed(seed_path)
    except SeedDataError as exc:
        print(f"ERROR loading VASP seed data: {exc}")
        sys.exit(1)

    is_synthetic_run = any(
        e.source_type == SeedSourceType.SYNTHETIC_DEMO for e in seed_entries
    )

    print("=" * 50)
    print("WALLET ATTRIBUTION ANALYSIS")
    print("=" * 50)
    if is_synthetic_run:
        print(
            "\n*** THIS RUN USES A SYNTHETIC DEMO SEED SET ***\n"
            "*** Results below are NOT real-world VASP attribution. ***\n"
            f"*** Seed file: {seed_path} ***"
        )
    print(f"\nGraph:   {graph_path}")
    print(f"Wallet:  {wallet}")
    print(f"Seed:    {seed_path} ({len(seed_entries)} known address(es) loaded)")

    graph = load_graph(graph_path)

    trace_result = trace_fund_flow(
        graph, wallet, settings=settings, max_hops=args.max_hops, max_paths=args.max_paths
    )
    behavior_patterns = analyze_wallet_behavior(graph, wallet, settings, paths=trace_result.paths)

    seed_index = build_seed_index(seed_entries)
    attribution = generate_candidates(trace_result, seed_index, behavior_patterns)

    print(f"\nMAX HOPS:            {attribution.max_hops}")
    print(f"Search truncated:     {attribution.search_truncated}")
    print(f"Status:               {attribution.status.value}")

    if attribution.status == AttributionStatus.MATCH_FOUND:
        for candidate in attribution.candidates:
            print_candidate(candidate)
    elif attribution.status == AttributionStatus.INCONCLUSIVE:
        print("\nINCONCLUSIVE")
        print(
            "The search was cut short by a resource limit (MAX_PATHS or "
            "MAX_EDGES_EXPLORED) before the configured MAX_HOPS depth could "
            "be fully examined. This is NOT the same as NONE — widen the "
            "limits and re-run for a complete answer."
        )
    else:
        print("\nNO KNOWN VASP CANDIDATE FOUND")
        print(
            "The investigated wallet did not reach any known VASP seed "
            f"address within the fully-examined MAX_HOPS={attribution.max_hops} search depth."
        )

    for note in attribution.notes:
        print(f"  - {note}")

    print("\n" + "-" * 50)
    print("Behavioral patterns detected (supporting evidence only):")
    print("-" * 50)
    if not behavior_patterns:
        print("None crossed the configured thresholds for this wallet.")
    else:
        for pattern in behavior_patterns:
            print(f"  - {pattern.pattern_type.value}: {'; '.join(pattern.evidence)}")
    print(
        "\nIMPORTANT: behavioral patterns alone do not establish VASP "
        "attribution. They only support a candidate already generated "
        "from address-level path evidence above."
    )

    print(
        "\n(VASP intelligence + explainable attribution = Macro Milestone 4. "
        "No ML, clustering, embeddings, or numeric confidence scores — "
        "those remain future milestones.)"
    )


if __name__ == "__main__":
    main()
