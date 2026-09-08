"""
MILESTONE 2 CLI.

Usage:
    python build_graph.py <wallet_address> [--chain ethereum]
    python build_graph.py --fixture data/fixtures/0xabc..._ethereum.json

Deliberately a separate script from investigate.py: Milestone 1's CLI and
provider/normalization layer are untouched. This reads an already-saved
fixture, builds the NetworkX graph, prints a structural summary, and
caches the graph to disk for the next milestone to reuse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.config import get_settings
from app.graph.builder import build_graph, load_transfers_from_fixture, save_graph, summarize_graph
from app.reporting.terminal import configure_stdout

FIXTURES_DIR = Path("data/fixtures")


def graphs_dir() -> Path:
    """The graph cache directory, from GRAPH_CACHE_DIR rather than a constant.

    A hardcoded "data/graphs" here meant this script and investigate.py could
    write to different places once GRAPH_CACHE_DIR was set, so a graph built by
    one was invisible to the other. Read at call time, not import time, so an
    env change inside one process still takes effect.
    """
    return Path(get_settings().graph_cache_dir)


def resolve_fixture_path(wallet: str | None, chain: str, fixture: str | None) -> Path:
    if fixture:
        return Path(fixture)
    if not wallet:
        raise ValueError("Provide either a wallet address or --fixture <path>.")
    return FIXTURES_DIR / f"{wallet.lower()}_{chain}.json"


def main() -> None:
    # These scripts print typographic dashes, which become mojibake on a
    # cp1252 Windows console. Reuse the production renderer's stream setup
    # rather than restating the strings.
    configure_stdout()
    parser = argparse.ArgumentParser(description="Milestone 2: build NetworkX graph from a normalized fixture")
    parser.add_argument("wallet", nargs="?", help="Wallet address matching an existing fixture")
    parser.add_argument("--chain", default="ethereum", help="Chain the fixture was built for")
    parser.add_argument("--fixture", help="Explicit path to a fixture JSON file (overrides wallet/chain lookup)")
    args = parser.parse_args()

    try:
        fixture_path = resolve_fixture_path(args.wallet, args.chain, args.fixture)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not fixture_path.exists():
        print(
            f"ERROR: fixture not found at {fixture_path}. "
            "Run investigate.py for this wallet first (Milestone 1)."
        )
        sys.exit(1)

    print("=" * 50)
    print("GRAPH CONSTRUCTION — MILESTONE 2")
    print("=" * 50)
    print(f"\nFixture: {fixture_path}\n")

    transfers = load_transfers_from_fixture(fixture_path)
    print(f"Loaded {len(transfers)} normalized transfers.")

    graph, stats = build_graph(transfers)
    summary = summarize_graph(graph, stats)

    print("-" * 50)
    print("INPUT/OUTPUT ACCOUNTING")
    print("-" * 50)
    print(f"Input transfers:              {summary.input_transfer_count}")
    print(f"  -> edges created:           {summary.edges_created}")
    print(f"  -> contract-creation skipped: {summary.contract_creation_skipped}")
    print(f"  -> other skipped:           {summary.other_skipped}")
    print(f"  -> accounted for:           {summary.accounted_for}")
    print(f"  -> RECONCILED:              {'YES' if summary.reconciled else 'NO -- SEE NOTES'}")

    print("-" * 50)
    print("GRAPH SUMMARY")
    print("-" * 50)
    print(f"Nodes (wallets/entities):     {summary.node_count}")
    print(f"Edges (graph.number_of_edges): {summary.edge_count}")
    print(f"  native TRANSFER edges:      {summary.native_edge_count}")
    print(f"  TOKEN_TRANSFER edges:       {summary.token_edge_count}")
    print(f"Self-loop edges:              {summary.self_loop_edges}")
    print(f"Average out-degree:           {summary.average_out_degree}")
    print(f"Average in-degree:            {summary.average_in_degree}")
    print(f"Graph density:                {summary.density}")
    print(f"Earliest edge timestamp:      {summary.earliest_timestamp}")
    print(f"Latest edge timestamp:        {summary.latest_timestamp}")

    if summary.top_out_degree_nodes:
        print("\nTop out-degree nodes (most outgoing counterparties):")
        for entry in summary.top_out_degree_nodes:
            print(f"  {entry.address}  (out-degree {entry.degree})")

    if summary.top_in_degree_nodes:
        print("\nTop in-degree nodes (most incoming counterparties):")
        for entry in summary.top_in_degree_nodes:
            print(f"  {entry.address}  (in-degree {entry.degree})")

    if summary.notes:
        print("\nNotes:")
        for n in summary.notes:
            print(f"  - {n}")

    cache_dir = graphs_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_path = cache_dir / f"{fixture_path.stem}.gpickle"
    save_graph(graph, graph_path)
    print(f"\nGraph cached to: {graph_path}")
    print(
        "\nThis is structural graph construction only. No path discovery, "
        "behavioral detection, clustering, or ML has run yet — those are "
        "Milestones 3+."
    )


if __name__ == "__main__":
    main()
