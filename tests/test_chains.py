"""
Tests for the chain registry and for every gate that depends on it.

The defect these exist for was silent and total: `--chain dogecoin` was
accepted, `EtherscanProvider` used the name only as a label while sending
whatever `ETHERSCAN_CHAIN_ID` said, and the run therefore returned real
ETHEREUM transactions with "dogecoin" printed in the report header, on every
normalized transfer, and beside every attribution. Nothing looked broken. That
is the failure mode worth pinning: a report that states a chain its data did
not come from is not a degraded answer, it is a false one.

So the properties asserted here are:

  * one name -> one chain id, resolved in one place
  * an unresolvable name is REJECTED, never defaulted
  * the id actually sent to the provider comes from the resolved name
  * an ETHERSCAN_CHAIN_ID that contradicts the name is a hard error, not a
    preference honoured over the name
  * the native asset symbol follows the chain rather than being hardcoded
  * all three CLI acquisition modes validate the chain, including the two
    cached ones that never build a provider

Fully offline. No API key, no network.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout

import pytest

import investigate as cli
from app.blockchain.base import UnsupportedChainError
from app.blockchain.chains import (
    CHAIN_VALIDATED_LIVE,
    CHAINS,
    SUPPORTED_CHAINS,
    ChainSpec,
    native_symbol,
    normalize_chain_name,
    resolve_chain,
)
from app.blockchain.etherscan import EtherscanProvider
from app.core.config import Settings
from app.investigation.pipeline import PipelineError, validate_chain_name

WALLET = "0x" + "11" * 20


def _settings(**overrides) -> Settings:
    base = dict(
        etherscan_api_key="test-key-not-real",
        etherscan_base_url="https://example.invalid",
        max_transactions_per_investigation=100,
        default_lookback_days=90,
        http_timeout_seconds=5,
        http_max_retries=1,
    )
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------
# the registry itself
# --------------------------------------------------------------------------


def test_every_entry_is_internally_consistent():
    """A ChainSpec whose `name` differs from its dict key would make
    `resolve_chain(x).name != x`, and the name is what gets written onto every
    transfer. Cheap to assert, expensive to discover later."""
    for key, spec in CHAINS.items():
        assert isinstance(spec, ChainSpec)
        assert spec.name == key
        assert key == key.lower().strip(), "keys are the canonical lowercase form"
        assert spec.chain_id > 0
        assert spec.native_symbol and spec.native_symbol.isupper()
        assert spec.native_decimals == 18


def test_chain_ids_are_unique():
    """Two names sharing an id would mean two report headers for one chain's
    data, which is the same mislabelling this module exists to prevent."""
    ids = [spec.chain_id for spec in CHAINS.values()]
    assert len(ids) == len(set(ids))


def test_supported_chains_is_exactly_the_registry():
    assert SUPPORTED_CHAINS == frozenset(CHAINS)


def test_validated_live_is_a_subset_and_is_not_a_claim_about_everything():
    """CHAIN_VALIDATED_LIVE is an honesty marker: it must name only chains that
    exist in the registry, and it must NOT quietly grow to cover chains this
    build has never been run against."""
    assert CHAIN_VALIDATED_LIVE <= SUPPORTED_CHAINS
    assert CHAIN_VALIDATED_LIVE == frozenset({"ethereum"})


def test_ethereum_resolves_to_chain_id_one():
    assert resolve_chain("ethereum").chain_id == 1
    assert resolve_chain("ethereum").native_symbol == "ETH"


@pytest.mark.parametrize("written", ["Ethereum", "ETHEREUM", "  ethereum  ", "eThErEuM"])
def test_capitalisation_and_surrounding_whitespace_are_forgiven(written):
    """A chain name is the operator's typing, not an on-chain identifier."""
    assert resolve_chain(written).name == "ethereum"
    assert normalize_chain_name(written) == "ethereum"


@pytest.mark.parametrize(
    "written",
    ["eth", "ethereum-mainnet", "ether", "ethereu", "mainnet", "ethereumm"],
)
def test_nothing_beyond_case_and_whitespace_is_forgiven(written):
    """No aliasing, no prefix matching, no "did you mean". A near-miss that
    silently resolves is exactly how one chain's data ends up under another
    chain's name."""
    with pytest.raises(UnsupportedChainError):
        resolve_chain(written)


@pytest.mark.parametrize("written", ["dogecoin", "bitcoin", "", "   ", "not-a-chain"])
def test_an_unresolvable_name_raises_instead_of_defaulting(written):
    with pytest.raises(UnsupportedChainError) as excinfo:
        resolve_chain(written)
    # The message must say what IS resolvable, or the operator is left guessing.
    assert "ethereum" in str(excinfo.value)


def test_none_is_rejected_rather_than_treated_as_the_default_chain():
    with pytest.raises(UnsupportedChainError):
        resolve_chain(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# native symbol -- reports, not raises
# --------------------------------------------------------------------------


def test_native_symbol_follows_the_chain():
    """This replaced a hardcoded `"ETH" if chain == "ethereum" else None`, which
    gave every other chain's native transfers no asset symbol at all."""
    assert native_symbol("ethereum") == "ETH"
    assert native_symbol("polygon") == "POL"
    assert native_symbol("bsc") == "BNB"
    assert native_symbol("avalanche") == "AVAX"


def test_native_symbol_returns_none_for_an_unknown_chain_rather_than_raising():
    """The normalizer's contract is to normalize every record and let the
    separate validation pass reject it with a reason; raising here would abort
    the whole run over one bad record and lose the audit trail."""
    assert native_symbol("dogecoin") is None
    assert native_symbol("") is None


def test_the_normalizer_uses_the_registry_symbol():
    from app.normalization.transactions import normalize_native_transaction

    raw = {
        "hash": "0x" + "a" * 64,
        "blockNumber": "1",
        "timeStamp": "1700000000",
        "from": WALLET,
        "to": "0x" + "22" * 20,
        "value": "1000000000000000000",
    }
    assert normalize_native_transaction(raw, "ethereum", "etherscan").asset_symbol == "ETH"
    assert normalize_native_transaction(raw, "polygon", "etherscan").asset_symbol == "POL"


# --------------------------------------------------------------------------
# the provider: the id sent must come from the resolved name
# --------------------------------------------------------------------------


def test_the_provider_sends_the_chain_id_of_the_named_chain():
    """The original bug in one assertion: the id on the wire used to come from
    config while the name came from the caller, so the two could disagree."""
    for name, spec in CHAINS.items():
        provider = EtherscanProvider(_settings(), chain_name=name)
        assert provider._base_params()["chainid"] == spec.chain_id
        assert provider.chain == name


def test_the_provider_canonicalises_the_name_it_reports():
    provider = EtherscanProvider(_settings(), chain_name="  Ethereum ")
    assert provider.chain == "ethereum"


def test_the_provider_refuses_an_unresolvable_chain_before_any_request():
    with pytest.raises(UnsupportedChainError):
        EtherscanProvider(_settings(), chain_name="dogecoin")


def test_a_configured_chain_id_that_matches_the_name_is_accepted():
    provider = EtherscanProvider(
        _settings(etherscan_chain_id=1), chain_name="ethereum"
    )
    assert provider._base_params()["chainid"] == 1


def test_a_configured_chain_id_that_contradicts_the_name_is_a_hard_error():
    """Honouring the id over the name would query Ethereum and label the result
    Polygon; honouring the name over the id would silently ignore an explicit
    setting. Neither is defensible, so one of them has to be wrong out loud."""
    with pytest.raises(UnsupportedChainError) as excinfo:
        EtherscanProvider(_settings(etherscan_chain_id=137), chain_name="ethereum")
    message = str(excinfo.value)
    assert "137" in message and "ethereum" in message


def test_an_unset_chain_id_is_the_normal_case():
    """`etherscan_chain_id` is optional precisely so the name is the single
    source of truth; a build that required it would reintroduce the two
    independent inputs that could disagree."""
    assert _settings().etherscan_chain_id is None
    provider = EtherscanProvider(_settings(), chain_name="polygon")
    assert provider._base_params()["chainid"] == 137


def test_the_cache_identity_separates_two_chains():
    """Two chains sharing a cache entry would serve one chain's payload for the
    other's query -- the same mislabelling by a different route."""
    eth = EtherscanProvider(_settings(), chain_name="ethereum")
    poly = EtherscanProvider(_settings(), chain_name="polygon")
    params = {"module": "account", "action": "txlist", "address": WALLET}
    assert eth._identity(params) != poly._identity(params)


def test_the_cache_identity_never_carries_the_credential():
    provider = EtherscanProvider(_settings(), chain_name="ethereum")
    identity = provider._identity(provider._base_params())
    assert "apikey" not in identity
    assert "test-key-not-real" not in str(identity)


# --------------------------------------------------------------------------
# pipeline-level validation
# --------------------------------------------------------------------------


def test_validate_chain_name_returns_the_canonical_name():
    assert validate_chain_name("  ETHEREUM ") == "ethereum"


def test_validate_chain_name_stops_the_run_with_an_investigator_readable_message():
    with pytest.raises(PipelineError) as excinfo:
        validate_chain_name("dogecoin")
    message = str(excinfo.value)
    assert "dogecoin" in message
    assert "ethereum" in message
    # It must also say which chains are merely mapped vs actually exercised,
    # so a demonstrator does not read the list as seven validated chains.
    assert "live provider data" in message


def test_the_pipeline_message_is_pure_ascii():
    """It is printed to stderr on a Windows console, where anything outside
    cp1252 becomes mojibake or a UnicodeEncodeError."""
    with pytest.raises(PipelineError) as excinfo:
        validate_chain_name("dogecoin")
    str(excinfo.value).encode("ascii")


# --------------------------------------------------------------------------
# the CLI: all three acquisition modes, including the two cached ones
# --------------------------------------------------------------------------


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    argv_backup = sys.argv[:]
    sys.argv = ["investigate.py", *argv]
    try:
        with redirect_stdout(out), redirect_stderr(err):
            try:
                cli.main()
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
            else:  # pragma: no cover - main always exits
                code = 0
    finally:
        sys.argv = argv_backup
    return code, out.getvalue(), err.getvalue()


def test_the_cli_rejects_an_unresolvable_chain_in_live_mode(monkeypatch):
    """And rejects it as a CHAIN problem: before this fix the run got as far as
    the credential check, so the operator was told to configure an API key for a
    chain that could never have worked."""
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    code, _out, err = _run([WALLET, "--chain", "dogecoin"])
    assert code == 1
    assert "Unsupported chain 'dogecoin'" in err
    assert "ETHERSCAN_API_KEY" not in err


def test_the_cli_rejects_an_unresolvable_chain_with_a_cached_graph(tmp_path):
    """The mode that made this worst: --cached-graph never constructs a
    provider, so provider-side resolution alone would have let the whole report
    be rendered with "dogecoin" in its header over Ethereum data."""
    import networkx as nx

    from app.graph.builder import save_graph

    graph = nx.MultiDiGraph()
    graph.add_edge(
        WALLET,
        "0x" + "22" * 20,
        key="0x" + "a" * 64 + "#0",
        tx_hash="0x" + "a" * 64,
        timestamp=1_700_000_000,
        amount=1.0,
        asset="ETH",
        chain="ethereum",
    )
    path = tmp_path / "real.gpickle"
    save_graph(graph, path)

    code, _out, err = _run(
        [WALLET, "--chain", "dogecoin", "--cached-graph", str(path)]
    )
    assert code == 1
    assert "Unsupported chain 'dogecoin'" in err


def test_the_cli_rejects_an_unresolvable_chain_with_a_transfers_file(tmp_path):
    missing = tmp_path / "transfers.json"  # never read: the chain fails first
    code, _out, err = _run(
        [WALLET, "--chain", "dogecoin", "--transfers-file", str(missing)]
    )
    assert code == 1
    assert "Unsupported chain 'dogecoin'" in err


def test_the_cli_help_does_not_claim_seven_validated_chains():
    """Listing the resolvable chains is documentation; letting a reader infer
    that all seven have been tested here is a claim this build cannot support."""
    # argparse hard-wraps help text, so the phrase is checked against the text
    # with runs of whitespace collapsed rather than as printed.
    help_text = " ".join(cli.build_parser().format_help().split())
    assert "ethereum" in help_text
    assert "exercised against live provider data" in help_text
