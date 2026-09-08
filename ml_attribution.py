"""
MACRO MILESTONE 5 CLI — evidence-based attribution (M4, unmodified) plus a
separate, clearly-labeled ML prediction (M5).

Usage:
    python ml_attribution.py <wallet_address> [--max-hops 3] [--chain ethereum]
    python ml_attribution.py <wallet_address> --graph data/graphs/<file>.gpickle
    python ml_attribution.py <wallet_address> --demo   # synthetic VASP seed set
    python ml_attribution.py <wallet_address> --seed <path>  # explicit seed override
    python ml_attribution.py <wallet_address> --ml-seed 42   # ML training seed override

This script reuses the exact same graph -> trace_fund_flow ->
analyze_wallet_behavior -> generate_candidates pipeline as
attribute_wallet.py (Macro Milestone 4), unmodified, then adds ONE new
section: feature extraction -> ML training -> ML prediction (Macro
Milestone 5).

The two sections are printed under clearly separated headers —
"M4 EVIDENCE" and "M5 ML PREDICTION (SYNTHETIC/DEMO)" — and the ML
section can never edit or override anything printed in the M4 section
above it. See app/ml/models.py for why.
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
from app.ml.features import extract_wallet_features
from app.ml.predictor import DEFAULT_SEED, predict, train_model
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
        description="Macro Milestone 5: M4 evidence-based attribution + M5 ML prediction (synthetic/demo)"
    )
    parser.add_argument("wallet", nargs="?", help="Wallet address matching an existing cached graph")
    parser.add_argument("--chain", default="ethereum")
    parser.add_argument("--graph", help="Explicit path to a cached .gpickle graph")
    parser.add_argument("--max-hops", type=int, default=None, help="Override FUND_TRACE_MAX_HOPS for this run")
    parser.add_argument("--max-paths", type=int, default=None, help="Override FUND_TRACE_MAX_PATHS for this run")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the synthetic demo VASP seed set instead of the production one (M4 section only)",
    )
    parser.add_argument("--seed", help="Explicit path to a VASP seed JSON file (overrides --demo)")
    parser.add_argument(
        "--ml-seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for M5 ML model training (default: deterministic default seed)",
    )
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
    print("WALLET ATTRIBUTION + ML ANALYSIS")
    print("=" * 50)
    if is_synthetic_run:
        print(
            "\n*** THIS M4 RUN USES A SYNTHETIC DEMO SEED SET ***\n"
            "*** M4 evidence results below are NOT real-world VASP attribution. ***\n"
            f"*** Seed file: {seed_path} ***"
        )
    print(f"\nGraph:   {graph_path}")
    print(f"Wallet:  {wallet}")
    print(f"Seed:    {seed_path} ({len(seed_entries)} known address(es) loaded)")

    # ---- identical M1-M4 pipeline wiring to attribute_wallet.py ----
    graph = load_graph(graph_path)

    trace_result = trace_fund_flow(
        graph, wallet, settings=settings, max_hops=args.max_hops, max_paths=args.max_paths
    )
    behavior_patterns = analyze_wallet_behavior(graph, wallet, settings, paths=trace_result.paths)

    seed_index = build_seed_index(seed_entries)
    attribution = generate_candidates(trace_result, seed_index, behavior_patterns)

    # =========================== M4 EVIDENCE ===========================
    print("\n" + "=" * 50)
    print("M4 EVIDENCE — deterministic, address/path-based attribution")
    print("=" * 50)

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
    print("Behavioral patterns detected (M4 supporting evidence only):")
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

    # ======================= M5 ML PREDICTION ==========================
    # Everything below is additive and read-only with respect to the M4
    # AttributionResult computed above — it cannot change anything already
    # printed, and nothing in app/ml/ writes back into `attribution`.
    features = extract_wallet_features(
        graph, wallet, trace_result, behavior_patterns, attribution
    )
    model = train_model(seed=args.ml_seed)
    ml_prediction = predict(model, features)

    print("\n" + "=" * 50)
    print("M5 ML PREDICTION (SYNTHETIC/DEMO — NOT REAL-WORLD ATTRIBUTION)")
    print("=" * 50)
    print(f"\nModel:               {ml_prediction.model_name} ({ml_prediction.model_version})")
    print(f"Training data:        {ml_prediction.training_data_type}")
    print(f"Random seed:          {ml_prediction.random_seed}")
    print(f"Predicted label:      {ml_prediction.predicted_label.value}")
    print("\nFeatures used for this prediction:")
    for name, value in zip(
        ml_prediction.feature_snapshot.feature_names(),
        ml_prediction.feature_snapshot.to_feature_vector(),
    ):
        print(f"    {name:35s} {value}")
    print(f"\n{ml_prediction.disclaimer}")
    print(
        "\nThis ML prediction is a SEPARATE, ADDITIONAL signal. It did NOT "
        "create, modify, or override the M4 attribution result printed "
        "above, and it is not itself evidence of VASP ownership."
    )


if __name__ == "__main__":
    main()
