from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.blockchain.chains import CHAIN_VALIDATED_LIVE, SUPPORTED_CHAINS
from app.core.config import ConfigurationError, get_settings
from app.investigation.pipeline import (PipelineError,
                                        acquire_from_cached_graph,
                                        acquire_from_transfers_file,
                                        acquire_live, run_investigation,
                                        validate_chain_name,
                                        validate_wallet_address)
from app.reporting.brief import print_brief_report
from app.reporting.compact import print_compact_report
from app.reporting.terminal import configure_stdout, print_report


def build_parser() -> argparse.ArgumentParser:
    """Every flag here changes behaviour. No cosmetic or placeholder options."""
    parser = argparse.ArgumentParser(
        prog="investigate.py",
        description=(
            "Investigate a blockchain wallet: real data acquisition, graph "
            "construction, bidirectional VASP attribution, behavioural and "
            "temporal analysis, transparent risk scoring and real-data ML."
        ),
        epilog=(
            "Live acquisition requires ETHERSCAN_API_KEY in the environment or "
            "a .env file. No demo or synthetic data is ever substituted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("wallet", help="wallet address to investigate (0x + 40 hex)")
    parser.add_argument(
        "--chain",
        default="ethereum",
        help=(
            "chain to investigate (default: ethereum). Resolvable: "
            + ", ".join(sorted(SUPPORTED_CHAINS))
            + ". Only "
            + ", ".join(sorted(CHAIN_VALIDATED_LIVE))
            + " has been exercised against live provider data in this build; "
            "an unresolvable name is rejected rather than queried under a "
            "default chain id"
        ),
    )

    parser.add_argument(
        "--max-hops",
        type=int,
        default=None,
        help=(
            "hop depth for both acquisition and path search (default: from config, "
            "currently FUND_TRACE_MAX_HOPS=4). Acquisition expands this many hop "
            "levels outward from the investigated wallet before running the search, "
            "so the graph actually contains edges at the depth being searched."
        ),
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=None,
        help="maximum paths to retain per target (default: from config)",
    )
    parser.add_argument(
        "--time-window",
        type=int,
        default=None,
        metavar="DAYS",
        help=(
            "only consider transfers within this many days of the wallet's most "
            "recent activity; 0 disables the window"
        ),
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "re-query the provider instead of reading cached responses; the "
            "fresh responses still replace the cached ones"
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="do not read or write the provider response cache at all",
    )

    parser.add_argument(
        "--cached-graph",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "analyse a real transaction graph saved by an earlier run instead of "
            "fetching; output is labelled CACHED REAL DATA"
        ),
    )
    parser.add_argument(
        "--transfers-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "rebuild the graph from a normalized real transfer file instead of "
            "fetching; output is labelled CACHED REAL DATA. Unlike --cached-graph "
            "this preserves per-record provenance, so the supervised ML labelling "
            "rules can run"
        ),
    )

    parser.add_argument(
        "--json",
        dest="json_path",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help=(
            "emit the complete machine-readable report; with no PATH it goes to "
            "stdout instead of the human report, otherwise to the given file"
        ),
    )

    ml_group = parser.add_mutually_exclusive_group()
    ml_group.add_argument(
        "--ml",
        dest="ml",
        action="store_true",
        default=True,
        help="run the machine-learning stage (default)",
    )
    ml_group.add_argument(
        "--no-ml",
        dest="ml",
        action="store_false",
        help="skip the machine-learning stage entirely",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "print stage-by-stage progress to stderr, and print the full "
            "nine-section report instead of the compact one"
        ),
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--brief",
        action="store_true",
        help=(
            "print the at-a-glance brief instead of the compact report: the "
            "VASP match, the strongest evidence path, a three-line risk "
            "summary and the one-line ML verdict. Analyses exactly the same "
            "way -- this chooses only what is printed"
        ),
    )
    output_group.add_argument(
        "--full-report",
        dest="full_report",
        action="store_true",
        help=(
            "print the complete nine-section report with all evidence, "
            "methodology and limitations (default output is the compact report)"
        ),
    )
    return parser


def _progress(enabled: bool):
    """Progress goes to stderr so that `--json` on stdout stays parseable."""

    def emit(message: str) -> None:
        if enabled:
            print(f"[investigate] {message}", file=sys.stderr, flush=True)

    return emit


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    say = _progress(args.verbose)

    if args.cached_graph is not None and args.transfers_file is not None:
        print(
            "ERROR: --cached-graph and --transfers-file both supply the graph; "
            "choose one.",
            file=sys.stderr,
        )
        return 1


    wallet = validate_wallet_address(args.wallet)
    say(f"address validated: {wallet}")
    chain = validate_chain_name(args.chain)
    say(f"chain resolved: {chain}")

    transfers = None
    normalization = None
    graph_summary = None

    if args.cached_graph is not None:
        say(f"loading cached real graph from {args.cached_graph}")
        graph, provenance = acquire_from_cached_graph(args.cached_graph)
    elif args.transfers_file is not None:
        say(f"loading normalized real transfers from {args.transfers_file}")
        (
            graph,
            transfers,
            normalization,
            graph_summary,
            provenance,
        ) = acquire_from_transfers_file(args.transfers_file, chain)
    else:
        hops = args.max_hops if args.max_hops is not None else settings.fund_trace_max_hops
        say(
            f"acquiring live blockchain data recursively to {hops} hop(s) "
            "(native + internal + token streams per address)"
        )
        graph_path = Path(settings.graph_cache_dir) / f"{wallet}_{chain}_live.gpickle"
        (
            graph,
            transfers,
            normalization,
            graph_summary,
            provenance,
        ) = await acquire_live(
            wallet,
            chain,
            settings,
            use_cache=not args.no_cache,
            save_to=graph_path,
            refresh=args.refresh,
            max_hops=args.max_hops,
            progress=say,
        )

    say(
        f"graph ready: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges [{provenance.data_mode.value}]"
    )
    say("running analysis stages")

    report = run_investigation(
        wallet,
        chain,
        graph,
        provenance,
        settings=settings,
        transfers=transfers,
        normalization=normalization,
        graph_summary=graph_summary,
        max_hops=args.max_hops,
        max_paths=args.max_paths,
        time_window_days=args.time_window,
        enable_ml=args.ml,
    )
    say(f"analysis complete in {report.duration_seconds}s")

    if args.json_path == "-":
        configure_stdout()
        print(report.model_dump_json(indent=2))
        return 0
    if args.full_report or args.verbose:
        print_report(report)
    elif args.brief:
        print_brief_report(report)
    else:
        print_compact_report(report)

    if args.json_path:
        destination = Path(args.json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nMachine-readable report written to {destination}")

    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_stdout()

    try:
        exit_code = asyncio.run(run(args))
    except ConfigurationError as exc:
        print(f"\nCONFIGURATION ERROR: {exc}", file=sys.stderr)
        exit_code = 1
    except PipelineError as exc:
        print(f"\nINVESTIGATION STOPPED: {exc}", file=sys.stderr)
        exit_code = 1
    except KeyboardInterrupt: 
        print("\nInterrupted before the investigation completed.", file=sys.stderr)
        exit_code = 2

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
