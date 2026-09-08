"""
ACQUISITION-MODE SELECTION — one entry point that any caller can use.

`investigate.py` decides where its data comes from from explicit flags
(`--cached-graph`, `--transfers-file`, or neither meaning live). A caller that
has no flags to read — the HTTP API — still has to make that same decision,
and it must make it the same way. This module is that decision, written once.

THE ORDER, AND WHY IT IS THIS ORDER
--------------------------------------------------------------------------
For a wallet with real artefacts already on disk:

    1. `{transfers_cache_dir}/{wallet}_{chain}.json`  — normalized transfers
    2. `{graph_cache_dir}/{wallet}_{chain}_live.gpickle`  — saved real graph
    3. `{graph_cache_dir}/{wallet}_{chain}.gpickle`
    4. live acquisition

Transfers come first because they carry per-record provenance, which a pickled
graph has already thrown away, and provenance is what the supervised labelling
rules need. A graph is preferred over a live fetch only when the caller asked
to reuse cached data.

NOTHING HERE IS A FALLBACK FROM A FAILURE. Selection happens *before* any
acquisition is attempted, purely from what exists on disk, and the chosen mode
is reported. If live acquisition is chosen and then fails, the investigation
stops — it does not quietly drop to an artefact and answer from stale data.
That distinction is the whole point: a report labelled REAL must have come
from a live fetch, and one labelled CACHED REAL DATA must say so.

DISCOVERY IS BY EXACT FILENAME
--------------------------------------------------------------------------
The wallet is lowercased and the name is built by template. No globbing, no
prefix matching, no "closest match" — the same exact-match rule the address
matcher follows, for the same reason: a near-miss here would analyse one
wallet's data and label it with another wallet's address.

WHAT THIS MODULE DOES NOT DO
--------------------------------------------------------------------------
Analysis. It selects a source, calls the existing acquisition function, hands
the graph to `run_investigation`, and returns that report unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from fastapi.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.investigation.errors import InvalidWalletError, UnsupportedChainError
from app.investigation.pipeline import (
    InvestigationReport,
    PipelineError,
    acquire_from_cached_graph,
    acquire_from_transfers_file,
    acquire_live,
    run_investigation,
    validate_chain_name,
    validate_wallet_address,
)


class MissingProviderCredentialError(PipelineError):
    """No provider credential is configured, so live acquisition is impossible.

    A subclass of PipelineError so a caller that only knows about
    PipelineError still stops, but distinguishable so that a transport layer
    can answer with a fixed string instead of forwarding a message that names
    the deployment's configuration.
    """


class AcquisitionMode(str, Enum):
    """Where an investigation's data came from. Reported, never inferred."""

    TRANSFERS_FILE = "TRANSFERS_FILE"
    CACHED_GRAPH = "CACHED_GRAPH"
    LIVE = "LIVE"


@dataclass(frozen=True)
class AcquisitionChoice:
    mode: AcquisitionMode
    #: The artefact that will be read, or None for a live fetch.
    path: Optional[Path]
    #: Why this mode was chosen, in one sentence, for the operator log.
    reason: str


def _candidate_paths(wallet: str, chain: str, settings: Settings) -> list[tuple[AcquisitionMode, Path]]:
    """The exact filenames this wallet's artefacts would have, in preference
    order. Built from settings, never from a hardcoded directory."""
    stem = f"{wallet.lower()}_{chain}"
    return [
        (
            AcquisitionMode.TRANSFERS_FILE,
            Path(settings.transfers_cache_dir) / f"{stem}.json",
        ),
        (
            AcquisitionMode.CACHED_GRAPH,
            Path(settings.graph_cache_dir) / f"{stem}_live.gpickle",
        ),
        (
            AcquisitionMode.CACHED_GRAPH,
            Path(settings.graph_cache_dir) / f"{stem}.gpickle",
        ),
    ]


def select_acquisition(
    wallet: str,
    chain: str,
    settings: Settings,
    *,
    prefer_cached: bool = True,
) -> AcquisitionChoice:
    """Chooses the data source from what exists on disk. Reads nothing.

    `prefer_cached=False` forces a live fetch even when artefacts exist, which
    is what a caller wants when the question is "what is true *now*" rather
    than "what did we observe". It never causes the reverse: cached data is
    never substituted for a live fetch that was asked for and failed.
    """
    if not prefer_cached:
        return AcquisitionChoice(
            mode=AcquisitionMode.LIVE,
            path=None,
            reason="a live fetch was requested, so cached artefacts were not considered",
        )

    for mode, path in _candidate_paths(wallet, chain, settings):
        if path.is_file():
            return AcquisitionChoice(
                mode=mode,
                path=path,
                reason=f"reusing the real {mode.value.lower().replace('_', ' ')} already on disk",
            )

    return AcquisitionChoice(
        mode=AcquisitionMode.LIVE,
        path=None,
        reason="no real artefact for this wallet and chain exists on disk",
    )


async def run_wallet_investigation(
    wallet: str,
    chain: str = "ethereum",
    settings: Optional[Settings] = None,
    *,
    prefer_cached: bool = True,
    max_hops: Optional[int] = None,
    max_paths: Optional[int] = None,
    time_window_days: Optional[int] = None,
    enable_ml: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[InvestigationReport, AcquisitionChoice]:
    """Acquires real data for one wallet and runs every analysis stage.

    Returns the report and the acquisition choice that produced it, so a
    caller can state where the data came from without guessing from the data
    mode alone.

    The analysis stages are CPU-bound and take seconds on a large graph, and
    loading a pickled graph blocks on disk. Both are handed to a worker thread
    so an async server keeps serving while one investigation runs; nothing
    about the analysis itself changes.
    """
    settings = settings or get_settings()
    say = progress or (lambda _message: None)

    # Same validators the CLI uses, so both surfaces accept and reject exactly
    # the same inputs with the same wording. Re-raised as the typed input errors
    # only so that a bad address answers 422 rather than being lumped in with
    # "the investigation stopped" — the message itself is passed through
    # unchanged, because it already says precisely what was wrong with it.
    try:
        wallet = validate_wallet_address(wallet)
    except PipelineError as exc:
        raise InvalidWalletError(str(exc)) from exc
    try:
        chain = validate_chain_name(chain)
    except PipelineError as exc:
        raise UnsupportedChainError(str(exc)) from exc

    choice = select_acquisition(wallet, chain, settings, prefer_cached=prefer_cached)
    say(f"acquisition mode {choice.mode.value}: {choice.reason}")

    transfers = None
    normalization = None
    graph_summary = None

    if choice.mode is AcquisitionMode.TRANSFERS_FILE:
        assert choice.path is not None  # set together with the mode
        (
            graph,
            transfers,
            normalization,
            graph_summary,
            provenance,
        ) = await run_in_threadpool(acquire_from_transfers_file, choice.path, chain)
    elif choice.mode is AcquisitionMode.CACHED_GRAPH:
        assert choice.path is not None
        graph, provenance = await run_in_threadpool(
            acquire_from_cached_graph, choice.path
        )
    else:
        # Checked here rather than inside acquire_live so that the caller can
        # distinguish "this deployment has no credentials" (a configuration
        # fault, and one whose detail must not travel to an HTTP client) from
        # "the chain had nothing for this address" (a data outcome the caller
        # needs to see).
        if not settings.etherscan_api_key:
            raise MissingProviderCredentialError(
                "ETHERSCAN_API_KEY is not set, so no real blockchain data can "
                "be acquired. This build does not substitute demo or synthetic "
                "data."
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
            save_to=graph_path,
            max_hops=max_hops,
            progress=say,
        )

    say(
        f"graph ready: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges [{provenance.data_mode.value}]"
    )

    report = await run_in_threadpool(
        run_investigation,
        wallet,
        chain,
        graph,
        provenance,
        settings=settings,
        transfers=transfers,
        normalization=normalization,
        graph_summary=graph_summary,
        max_hops=max_hops,
        max_paths=max_paths,
        time_window_days=time_window_days,
        enable_ml=enable_ml,
    )
    say(f"analysis complete in {report.duration_seconds}s")
    return report, choice


__all__ = [
    "AcquisitionChoice",
    "AcquisitionMode",
    "MissingProviderCredentialError",
    "run_wallet_investigation",
    "select_acquisition",
]
