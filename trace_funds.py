"""
MACRO MILESTONE 3 CLI — multi-hop fund-flow tracing + behavioral analysis.

Usage:
    python trace_funds.py <wallet_address> [--max-hops 3] [--chain ethereum]
    python trace_funds.py --graph data/graphs/<file>.gpickle <wallet_address>

Loads a Milestone-2 cached graph, traces fund-flow candidates from the
given wallet (Phase A), runs behavioral pattern detection over the traced
wallet and paths (Phase B), and prints a concise investigation summary.

Does NOT fetch anything from Etherscan — this operates entirely on the
already-cached .gpickle graph (build_graph.py must have been run first).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.behavior.detectors import analyze_wallet_behavior
from app.behavior.models import BehaviorPattern
from app.core.config import get_settings
from app.graph.builder import load_graph
from app.tracing.models import TraceResult
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

# How many discovered paths to print in full detail before summarizing the
# rest — an output-size control per the "avoid dumping enormous amounts of
# raw data" requirement. Not an investigative threshold, so kept as a CLI
# display constant rather than centralized config.
MAX_PATHS_PRINTED = 10


def resolve_graph_path(wallet: str | None, chain: str, graph_arg: str | None) -> Path:
    if graph_arg:
        return Path(graph_arg)
    if not wallet:
        raise ValueError("Provide either a wallet address or --graph <path>.")
    return graphs_dir() / f"{wallet.lower()}_{chain}.gpickle"


def print_path(index: int, path) -> None:
    print(f"\nPATH #{index}")
    arrow = "\n   |\n   v\n"
    print(arrow.join(path.addresses))
    duration = path.path_duration_seconds
    print(f"Hops:          {path.hop_count}")
    print(f"Duration:      {duration if duration is not None else 'unknown (missing timestamp)'} second(s)")
    print(f"Assets:        {', '.join(path.assets_involved)}")
    print(f"Transactions:  {', '.join(h.tx_hash for h in path.hops)}")


def print_pattern(pattern: BehaviorPattern) -> None:
    print(f"\nPattern: {pattern.pattern_type.value}")
    for line in pattern.evidence:
        print(f"  Evidence: {line}")
    for key, value in pattern.metrics.items():
        print(f"  {key}: {value}")
    if pattern.related_addresses:
        shown = pattern.related_addresses[:5]
        more = len(pattern.related_addresses) - len(shown)
        suffix = f" (+{more} more)" if more > 0 else ""
        print(f"  Related addresses: {', '.join(shown)}{suffix}")


def main() -> None:
    # These scripts print typographic dashes, which become mojibake on a
    # cp1252 Windows console. Reuse the production renderer's stream setup
    # rather than restating the strings.
    configure_stdout()
    parser = argparse.ArgumentParser(
        description="Macro Milestone 3: multi-hop fund-flow tracing + behavioral pattern detection"
    )
    parser.add_argument("wallet", nargs="?", help="Wallet address matching an existing cached graph")
    parser.add_argument("--chain", default="ethereum", help="Chain the graph was built for")
    parser.add_argument("--graph", help="Explicit path to a cached .gpickle graph (overrides wallet/chain lookup)")
    parser.add_argument("--max-hops", type=int, default=None, help="Override FUND_TRACE_MAX_HOPS for this run")
    parser.add_argument("--max-paths", type=int, default=None, help="Override FUND_TRACE_MAX_PATHS for this run")
    args = parser.parse_args()

    try:
        graph_path = resolve_graph_path(args.wallet, args.chain, args.graph)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not graph_path.exists():
        print(
            f"ERROR: cached graph not found at {graph_path}. "
            "Run build_graph.py for this wallet first (Milestone 2)."
        )
        sys.exit(1)

    if not args.wallet:
        print("ERROR: a wallet address is required (used as the trace source), even with --graph.")
        sys.exit(1)

    wallet = args.wallet.lower()
    settings = get_settings()

    print("=" * 50)
    print("FUND FLOW ANALYSIS")
    print("=" * 50)
    print(f"\nGraph:   {graph_path}")
    print(f"Source:  {wallet}")

    graph = load_graph(graph_path)

    trace_result: TraceResult = trace_fund_flow(
        graph,
        wallet,
        settings=settings,
        max_hops=args.max_hops,
        max_paths=args.max_paths,
    )

    print(f"MAX HOPS:            {trace_result.max_hops}")
    print(f"Paths discovered:     {len(trace_result.paths)}")
    print(f"Edges explored:       {trace_result.edges_explored}")
    if trace_result.paths_truncated:
        print("NOTE: result was truncated by FUND_TRACE_MAX_PATHS — more paths may exist.")
    if trace_result.edges_limit_hit:
        print("NOTE: exploration was stopped by FUND_TRACE_MAX_EDGES_EXPLORED — traversal may be incomplete.")
    for note in trace_result.notes:
        print(f"  - {note}")

    # Only the longest path per distinct terminal node is printed in detail,
    # to avoid drowning the user in every hop-1/hop-2/hop-3 prefix of the
    # same route (per "avoid dumping enormous amounts of raw data").
    longest_per_terminal: dict[str, object] = {}
    for path in trace_result.paths:
        existing = longest_per_terminal.get(path.terminal_node)
        if existing is None or path.hop_count > existing.hop_count:
            longest_per_terminal[path.terminal_node] = path
    display_paths = list(longest_per_terminal.values())[:MAX_PATHS_PRINTED]

    for i, path in enumerate(display_paths, 1):
        print_path(i, path)
    remaining = len(longest_per_terminal) - len(display_paths)
    if remaining > 0:
        print(f"\n...and {remaining} more distinct path(s) not shown (see full TraceResult for all).")

    print("\n" + "-" * 50)
    print("BEHAVIORAL ANALYSIS")
    print("-" * 50)
    print(
        "NOTE: these are structural/timing indicators, not proof of "
        "criminal activity, fraud, or VASP identity — see evidence for each."
    )

    patterns = analyze_wallet_behavior(graph, wallet, settings, paths=trace_result.paths)
    if not patterns:
        print("\nNo behavioral patterns crossed the configured thresholds for this wallet.")
    for pattern in patterns:
        print_pattern(pattern)

    print(
        "\n(Multi-hop fund-flow tracing = original Milestone 3, behavioral "
        "pattern detection = original Milestone 4. No VASP attribution, "
        "clustering, or ML — those are later milestones.)"
    )


if __name__ == "__main__":
    main()
