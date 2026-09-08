"""
Tests for `app.investigation.runner` -- acquisition-mode selection.

The CLI decides where its data comes from from explicit flags. The HTTP API has
no flags to read, so it asks this module, and both surfaces must answer the same
way. What is pinned here is not the plumbing but the honesty properties:

  * SELECTION IS NOT A FALLBACK. The mode is chosen from what exists on disk
    BEFORE any acquisition is attempted. A live fetch that was asked for and
    then failed never quietly becomes a cached answer -- the investigation
    stops instead, because a report labelled REAL must have come from a live
    fetch.
  * DISCOVERY IS BY EXACT FILENAME. Case-insensitive on the address, exact on
    everything else. A near-miss here would analyse one wallet's data and
    label it with another wallet's address, which is the same failure the
    address matcher's exact-match rule exists to prevent.
  * A MISSING CREDENTIAL IS A REFUSAL, NOT A DEMO. With no provider key and
    nothing on disk, the run raises rather than substituting synthetic data.
  * THE CHOSEN MODE IS REPORTED. The caller is told which source was used and
    why, so it never has to infer it from the data.

Fully offline. No API key, no network.
"""

from __future__ import annotations

import json

import networkx as nx
import pytest

from app.core.config import get_settings
from app.graph.builder import save_graph
from app.investigation.errors import InvalidWalletError, UnsupportedChainError
from app.investigation.pipeline import DataMode, PipelineError
from app.investigation.runner import (
    AcquisitionMode,
    MissingProviderCredentialError,
    run_wallet_investigation,
    select_acquisition,
)

WALLET = "0x" + "11" * 20
PEER = "0x" + "22" * 20
FAR = "0x" + "33" * 20


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    edges = [
        (WALLET, PEER, "0x" + "aa" * 32, 1_700_000_000),
        (PEER, FAR, "0x" + "bb" * 32, 1_700_000_600),
        (FAR, WALLET, "0x" + "cc" * 32, 1_700_001_200),
    ]
    for source, target, tx_hash, timestamp in edges:
        graph.add_edge(
            source,
            target,
            key=f"{tx_hash}#0",
            tx_hash=tx_hash,
            block_number=100,
            timestamp=timestamp,
            amount=1.5,
            asset="ETH",
            asset_type="NATIVE",
            transfer_type="NATIVE_TRANSACTION",
            transfer_source="NATIVE_TRANSACTION",
            status="SUCCESS",
            chain="ethereum",
            token_contract=None,
            gas_used=21000,
        )
    return graph


def _transfer(from_address: str, to_address: str, tx_hash: str, timestamp: int) -> dict:
    return {
        "tx_hash": tx_hash,
        "chain": "ethereum",
        "block_number": 100,
        "timestamp": timestamp,
        "from_address": from_address,
        "to_address": to_address,
        "asset_type": "NATIVE",
        "asset_identifier": None,
        "asset_symbol": "ETH",
        "asset_decimals": 18,
        "amount_raw": "1500000000000000000",
        "amount": 1.5,
        "usd_value": None,
        "gas_used": 21000,
        "gas_fee_native": 0.00105,
        "status": "SUCCESS",
        "is_contract_creation": False,
        "method_id": "0x",
        "source_provider": "etherscan",
        "fetched_at": 1_700_002_000,
    }


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """An isolated pair of artefact directories and no provider credential.

    No credential is the default because it makes an accidental live fetch
    impossible: any test that reaches the network would raise instead.
    """
    transfers_dir = tmp_path / "fixtures"
    graphs_dir = tmp_path / "graphs"
    transfers_dir.mkdir()
    graphs_dir.mkdir()
    monkeypatch.setenv("TRANSFERS_CACHE_DIR", str(transfers_dir))
    monkeypatch.setenv("GRAPH_CACHE_DIR", str(graphs_dir))
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    return transfers_dir, graphs_dir


def _write_transfers(directory, wallet=WALLET, chain="ethereum", records=None):
    path = directory / f"{wallet.lower()}_{chain}.json"
    path.write_text(
        json.dumps(
            records
            if records is not None
            else [
                _transfer(WALLET, PEER, "0x" + "aa" * 32, 1_700_000_000),
                _transfer(PEER, FAR, "0x" + "bb" * 32, 1_700_000_600),
                _transfer(FAR, WALLET, "0x" + "cc" * 32, 1_700_001_200),
            ]
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------
# preference order
# --------------------------------------------------------------------------


def test_with_nothing_on_disk_the_choice_is_live(env):
    choice = select_acquisition(WALLET, "ethereum", get_settings())
    assert choice.mode is AcquisitionMode.LIVE
    assert choice.path is None
    assert "no real artefact" in choice.reason


def test_a_transfers_file_wins_over_every_graph(env):
    """Transfers carry per-record provenance; a pickled graph has already
    thrown it away, and provenance is what the labelling rules need."""
    transfers_dir, graphs_dir = env
    transfers = _write_transfers(transfers_dir)
    save_graph(_graph(), graphs_dir / f"{WALLET}_ethereum_live.gpickle")
    save_graph(_graph(), graphs_dir / f"{WALLET}_ethereum.gpickle")

    choice = select_acquisition(WALLET, "ethereum", get_settings())
    assert choice.mode is AcquisitionMode.TRANSFERS_FILE
    assert choice.path == transfers


def test_the_live_graph_wins_over_the_plain_graph(env):
    _transfers_dir, graphs_dir = env
    live = graphs_dir / f"{WALLET}_ethereum_live.gpickle"
    save_graph(_graph(), live)
    save_graph(_graph(), graphs_dir / f"{WALLET}_ethereum.gpickle")

    choice = select_acquisition(WALLET, "ethereum", get_settings())
    assert choice.mode is AcquisitionMode.CACHED_GRAPH
    assert choice.path == live


def test_a_plain_graph_is_used_when_it_is_all_there_is(env):
    _transfers_dir, graphs_dir = env
    plain = graphs_dir / f"{WALLET}_ethereum.gpickle"
    save_graph(_graph(), plain)

    choice = select_acquisition(WALLET, "ethereum", get_settings())
    assert choice.mode is AcquisitionMode.CACHED_GRAPH
    assert choice.path == plain


def test_prefer_cached_false_forces_live_even_with_artefacts_present(env):
    """"What is true now" is a different question from "what did we observe",
    and a caller asking the first must not be answered from the second."""
    transfers_dir, graphs_dir = env
    _write_transfers(transfers_dir)
    save_graph(_graph(), graphs_dir / f"{WALLET}_ethereum_live.gpickle")

    choice = select_acquisition(
        WALLET, "ethereum", get_settings(), prefer_cached=False
    )
    assert choice.mode is AcquisitionMode.LIVE
    assert choice.path is None
    assert "live fetch was requested" in choice.reason


def test_every_choice_states_its_reason(env):
    transfers_dir, _graphs_dir = env
    _write_transfers(transfers_dir)
    for prefer_cached in (True, False):
        choice = select_acquisition(
            WALLET, "ethereum", get_settings(), prefer_cached=prefer_cached
        )
        assert choice.reason.strip(), "an operator log with no reason is not a log"


# --------------------------------------------------------------------------
# exact-filename discovery
# --------------------------------------------------------------------------


def test_the_address_is_matched_case_insensitively(env):
    """Same rule as the address matcher: case is irrelevant, and nothing else
    about the address is."""
    transfers_dir, _graphs_dir = env
    path = _write_transfers(transfers_dir)
    choice = select_acquisition(WALLET.upper(), "ethereum", get_settings())
    assert choice.mode is AcquisitionMode.TRANSFERS_FILE
    assert choice.path == path


def test_a_prefix_of_the_wallet_is_not_a_match(env):
    """Prefix matching here would analyse a different wallet's data and label
    it with this wallet's address."""
    transfers_dir, _graphs_dir = env
    _write_transfers(transfers_dir, wallet=WALLET[:-2] + "99")
    choice = select_acquisition(WALLET, "ethereum", get_settings())
    assert choice.mode is AcquisitionMode.LIVE


def test_a_filename_with_extra_suffix_is_not_a_match(env):
    transfers_dir, graphs_dir = env
    (transfers_dir / f"{WALLET}_ethereum_backup.json").write_text("[]", encoding="utf-8")
    save_graph(_graph(), graphs_dir / f"{WALLET}_ethereum_old.gpickle")
    choice = select_acquisition(WALLET, "ethereum", get_settings())
    assert choice.mode is AcquisitionMode.LIVE


def test_another_chains_artefact_is_not_used_for_this_chain(env):
    """The stem carries the chain, so a polygon graph can never answer an
    ethereum question."""
    transfers_dir, _graphs_dir = env
    _write_transfers(transfers_dir, chain="polygon")
    assert select_acquisition(WALLET, "ethereum", get_settings()).mode is (
        AcquisitionMode.LIVE
    )
    assert select_acquisition(WALLET, "polygon", get_settings()).mode is (
        AcquisitionMode.TRANSFERS_FILE
    )


def test_a_directory_that_happens_to_have_the_right_name_is_not_a_file(env):
    transfers_dir, _graphs_dir = env
    (transfers_dir / f"{WALLET}_ethereum.json").mkdir()
    assert select_acquisition(WALLET, "ethereum", get_settings()).mode is (
        AcquisitionMode.LIVE
    )


def test_the_artefact_directories_come_from_settings_not_a_hardcoded_path(
    tmp_path, monkeypatch
):
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    monkeypatch.setenv("TRANSFERS_CACHE_DIR", str(elsewhere))
    monkeypatch.setenv("GRAPH_CACHE_DIR", str(tmp_path / "no-graphs"))
    path = _write_transfers(elsewhere)
    assert select_acquisition(WALLET, "ethereum", get_settings()).path == path


# --------------------------------------------------------------------------
# input validation: the same wording as the CLI, typed for the transport
# --------------------------------------------------------------------------


async def test_a_malformed_wallet_is_an_invalid_wallet_error(env):
    with pytest.raises(InvalidWalletError) as raised:
        await run_wallet_investigation("0xnope", "ethereum", get_settings())
    # The validator's own message is passed through: it already says exactly
    # what was wrong, and rewording it here would make the two surfaces differ.
    assert "not a valid EVM address" in str(raised.value)


async def test_an_unsupported_chain_is_an_unsupported_chain_error(env):
    with pytest.raises(UnsupportedChainError):
        await run_wallet_investigation(WALLET, "not-a-real-chain", get_settings())


# --------------------------------------------------------------------------
# no credential means refusal, never demo data
# --------------------------------------------------------------------------


async def test_live_without_a_credential_refuses_rather_than_substituting_data(env):
    with pytest.raises(MissingProviderCredentialError) as raised:
        await run_wallet_investigation(WALLET, "ethereum", get_settings())
    message = str(raised.value)
    assert "ETHERSCAN_API_KEY is not set" in message
    assert "does not substitute demo or synthetic data" in message


async def test_a_requested_live_fetch_never_falls_back_to_a_cached_artefact(env):
    """The point of the whole module: with artefacts on disk and no credential,
    a live request fails. It does not answer from stale data under a REAL
    label, and it does not answer from stale data under any other label
    either -- the caller asked for live."""
    transfers_dir, graphs_dir = env
    _write_transfers(transfers_dir)
    save_graph(_graph(), graphs_dir / f"{WALLET}_ethereum_live.gpickle")

    with pytest.raises(MissingProviderCredentialError):
        await run_wallet_investigation(
            WALLET, "ethereum", get_settings(), prefer_cached=False
        )


def test_the_credential_refusal_is_a_pipeline_error():
    """A caller that only knows about PipelineError must still stop; the
    subclass exists only so a transport layer can answer with a fixed string
    instead of forwarding a message that names the deployment's config."""
    assert issubclass(MissingProviderCredentialError, PipelineError)


# --------------------------------------------------------------------------
# running against real artefacts
# --------------------------------------------------------------------------


async def test_a_cached_graph_run_returns_a_report_labelled_cached(env):
    _transfers_dir, graphs_dir = env
    save_graph(_graph(), graphs_dir / f"{WALLET}_ethereum_live.gpickle")

    report, choice = await run_wallet_investigation(
        WALLET, "ethereum", get_settings(), enable_ml=False
    )
    assert choice.mode is AcquisitionMode.CACHED_GRAPH
    assert report.wallet == WALLET
    assert report.chain == "ethereum"
    assert report.provenance.data_mode is DataMode.CACHED_REAL_DATA
    assert report.provenance.data_mode is not DataMode.REAL, (
        "only a live fetch may be labelled REAL"
    )


async def test_a_transfers_file_run_yields_the_transfer_stream(env):
    """The reason transfers outrank a graph: this mode can populate the
    normalization record, which a pickled graph cannot."""
    transfers_dir, _graphs_dir = env
    _write_transfers(transfers_dir)

    report, choice = await run_wallet_investigation(
        WALLET, "ethereum", get_settings(), enable_ml=False
    )
    assert choice.mode is AcquisitionMode.TRANSFERS_FILE
    assert report.normalization is not None
    assert report.transactions, "the wallet's own transfers must be in the report"


async def test_the_wallet_is_normalized_before_it_reaches_the_report(env):
    _transfers_dir, graphs_dir = env
    save_graph(_graph(), graphs_dir / f"{WALLET}_ethereum_live.gpickle")

    report, _choice = await run_wallet_investigation(
        WALLET.upper(), "ethereum", get_settings(), enable_ml=False
    )
    assert report.wallet == WALLET.lower()


async def test_progress_messages_name_the_chosen_mode(env):
    _transfers_dir, graphs_dir = env
    save_graph(_graph(), graphs_dir / f"{WALLET}_ethereum_live.gpickle")

    messages: list[str] = []
    await run_wallet_investigation(
        WALLET,
        "ethereum",
        get_settings(),
        enable_ml=False,
        progress=messages.append,
    )
    assert any("acquisition mode CACHED_GRAPH" in m for m in messages)
    assert any("graph ready" in m for m in messages)


async def test_a_missing_transfers_file_is_not_silently_skipped(env, monkeypatch):
    """If the artefact vanishes between selection and read, the run stops. It
    does not slide down the preference list to a different source, which would
    mean the reported mode no longer described the data."""
    transfers_dir, _graphs_dir = env
    path = _write_transfers(transfers_dir)
    settings = get_settings()

    original = select_acquisition(WALLET, "ethereum", settings)
    assert original.mode is AcquisitionMode.TRANSFERS_FILE
    path.unlink()

    with pytest.raises(PipelineError, match="No transfers file"):
        await run_wallet_investigation(WALLET, "ethereum", settings, enable_ml=False)
