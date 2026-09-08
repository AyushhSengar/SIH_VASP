"""
Tests for `investigate.py` -- the production entry point.

The CLI is the only part of this system a demonstrator actually touches, so
the properties pinned here are the ones whose failure would be visible on
stage or would mislead a reader:

  * every documented flag changes behaviour, and two flags documented as
    different are not secretly the same (--refresh vs --no-cache)
  * a bad address exits 1 with an explanation and no traceback
  * --json emits parseable JSON as the WHOLE of stdout, so it can be piped
  * --json PATH writes the file and still prints the human report
  * progress output goes to stderr, never into the JSON on stdout
  * mutually contradictory sources are rejected rather than silently ranked

Runs fully offline against a small real-shaped graph. No API key, no network.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout

import networkx as nx
import pytest

import investigate as cli
from app.graph.builder import save_graph

WALLET = "0x" + "11" * 20
PEER = "0x" + "22" * 20
FAR = "0x" + "33" * 20


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    edges = [
        (WALLET, PEER, "0x" + "a" * 64, 1_700_000_000),
        (PEER, FAR, "0x" + "b" * 64, 1_700_000_600),
        (FAR, WALLET, "0x" + "c" * 64, 1_700_001_200),
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
            chain="ethereum",
            token_contract=None,
            gas_used=21000,
        )
    return graph


@pytest.fixture()
def cached_graph(tmp_path):
    path = tmp_path / "real.gpickle"
    save_graph(_graph(), path)
    return path


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Runs main() the way a shell does, capturing both streams."""
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


# --------------------------------------------------------------------------
# parser wiring
# --------------------------------------------------------------------------


def test_every_documented_flag_is_parsed():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            WALLET,
            "--chain", "ethereum",
            "--max-hops", "4",
            "--max-paths", "50",
            "--time-window", "30",
            "--refresh",
            "--no-cache",
            "--verbose",
            "--full-report",
        ]
    )
    assert args.wallet == WALLET
    assert args.chain == "ethereum"
    assert args.max_hops == 4
    assert args.max_paths == 50
    assert args.time_window == 30
    assert args.refresh is True
    assert args.no_cache is True
    assert args.verbose is True
    assert args.full_report is True
    assert args.ml is True, "ML is on unless --no-ml is given"


def test_full_report_defaults_off_so_the_compact_report_is_the_default():
    args = cli.build_parser().parse_args([WALLET])
    assert args.full_report is False
    assert args.verbose is False


def test_ml_flags_are_mutually_exclusive():
    parser = cli.build_parser()
    assert parser.parse_args([WALLET, "--no-ml"]).ml is False
    assert parser.parse_args([WALLET, "--ml"]).ml is True
    with pytest.raises(SystemExit):
        parser.parse_args([WALLET, "--ml", "--no-ml"])


def test_json_flag_defaults_to_stdout_when_given_no_path():
    parser = cli.build_parser()
    assert parser.parse_args([WALLET]).json_path is None
    assert parser.parse_args([WALLET, "--json"]).json_path == "-"
    assert parser.parse_args([WALLET, "--json", "out.json"]).json_path == "out.json"


def test_wallet_argument_is_required():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


# --------------------------------------------------------------------------
# refresh vs no-cache -- documented as different, so verified as different
# --------------------------------------------------------------------------


def _provider(tmp_path, monkeypatch, refresh: bool):
    from app.blockchain.cache import ProviderResponseCache
    from app.blockchain.etherscan import EtherscanProvider
    from app.core.config import get_settings

    monkeypatch.setenv("ETHERSCAN_API_KEY", "test-key-not-real")
    monkeypatch.setenv("PROVIDER_CACHE_ENABLED", "true")
    monkeypatch.setenv("PROVIDER_CACHE_DIR", str(tmp_path / "cache"))
    settings = get_settings()
    cache = ProviderResponseCache(
        directory=settings.provider_cache_dir,
        ttl_seconds=settings.provider_cache_ttl_seconds,
        enabled=True,
    )
    return EtherscanProvider(settings, cache=cache, refresh=refresh), cache


@pytest.mark.asyncio
async def test_refresh_skips_the_cache_read_but_still_writes(tmp_path, monkeypatch):
    """A refresh that also suppressed the write would leave the cache empty,
    making every subsequent run pay full price for no reason."""
    provider, cache = _provider(tmp_path, monkeypatch, refresh=True)

    calls = {"n": 0}

    async def fake_get(_params):
        calls["n"] += 1
        return {"status": "1", "result": [{"hash": "0xdeadbeef"}]}

    monkeypatch.setattr(provider, "_get", fake_get)

    first = await provider.get_normal_transactions(WALLET)
    second = await provider.get_normal_transactions(WALLET)

    assert first == second == [{"hash": "0xdeadbeef"}]
    assert calls["n"] == 2, "refresh must re-query, not serve the entry it wrote"
    # The write happened: a provider WITHOUT refresh now finds the entry.
    plain, _ = _provider(tmp_path, monkeypatch, refresh=False)
    monkeypatch.setattr(plain, "_get", fake_get)
    await plain.get_normal_transactions(WALLET)
    assert calls["n"] == 2, "the refreshed response should have been cached"


@pytest.mark.asyncio
async def test_no_cache_neither_reads_nor_writes(tmp_path, monkeypatch):
    provider, cache = _provider(tmp_path, monkeypatch, refresh=False)

    calls = {"n": 0}

    async def fake_get(_params):
        calls["n"] += 1
        return {"status": "1", "result": [{"hash": "0xfeed"}]}

    monkeypatch.setattr(provider, "_get", fake_get)

    await provider.get_normal_transactions(WALLET, use_cache=False)
    await provider.get_normal_transactions(WALLET, use_cache=False)
    assert calls["n"] == 2

    # Nothing was stored, so a cache-enabled call still has to fetch.
    await provider.get_normal_transactions(WALLET)
    assert calls["n"] == 3


# --------------------------------------------------------------------------
# end-to-end CLI behaviour
# --------------------------------------------------------------------------


def test_bad_address_exits_1_with_an_explanation_and_no_traceback():
    code, out, err = _run(["not-a-wallet", "--cached-graph", "unused.gpickle"])
    assert code == 1
    assert "INVESTIGATION STOPPED" in err
    assert "Traceback" not in err and "Traceback" not in out


def test_missing_cached_graph_exits_1_naming_the_path(tmp_path):
    missing = tmp_path / "absent.gpickle"
    code, _out, err = _run([WALLET, "--cached-graph", str(missing)])
    assert code == 1
    assert "absent.gpickle" in err


def test_both_graph_sources_at_once_is_rejected(cached_graph, tmp_path):
    code, _out, err = _run(
        [
            WALLET,
            "--cached-graph", str(cached_graph),
            "--transfers-file", str(tmp_path / "t.json"),
        ]
    )
    assert code == 1
    assert "choose one" in err


def test_default_run_prints_the_compact_report(cached_graph):
    """The DEFAULT output is the ten-block compact report, not the nine
    sections. It must be distinguishable from the full report by its text."""
    code, out, _err = _run([WALLET, "--cached-graph", str(cached_graph), "--no-ml"])

    assert code == 0
    assert out.isascii(), "a Windows console on cp1252 would mangle anything else"
    assert "COMPACT REPORT" in out
    for number in range(1, 11):
        assert f"[{number}]" in out, f"compact block {number} is missing"
    assert "SECTION 1" not in out, "the default must not be the nine-section report"
    assert "CACHED REAL DATA" in out
    assert "INVESTIGATION COMPLETE" in out
    assert WALLET in out


def test_full_report_flag_prints_the_complete_report(cached_graph):
    code, out, _err = _run(
        [WALLET, "--cached-graph", str(cached_graph), "--no-ml", "--full-report"]
    )

    assert code == 0
    assert out.isascii(), "a Windows console on cp1252 would mangle anything else"
    for number in range(1, 10):
        assert f"SECTION {number}" in out
    assert "CACHED REAL DATA" in out
    assert "INVESTIGATION COMPLETE" in out
    assert WALLET in out


def test_verbose_also_prints_the_complete_report(cached_graph):
    """--verbose keeps its stderr-progress behaviour AND selects the full
    report, so a demonstrator who wants everything can pass one flag."""
    code, out, _err = _run(
        [WALLET, "--cached-graph", str(cached_graph), "--no-ml", "--verbose"]
    )

    assert code == 0
    for number in range(1, 10):
        assert f"SECTION {number}" in out


def test_json_to_stdout_is_the_only_thing_on_stdout(cached_graph):
    code, out, _err = _run(
        [WALLET, "--cached-graph", str(cached_graph), "--no-ml", "--json"]
    )
    assert code == 0

    payload = json.loads(out)  # must parse with no stripping or trimming
    assert payload["wallet"] == WALLET
    assert payload["chain"] == "ethereum"
    assert payload["provenance"]["data_mode"] == "CACHED REAL DATA"
    assert "SECTION" not in out, "the human report must not be mixed into the JSON"


def test_json_payload_carries_every_analysis_section(cached_graph):
    """One key per report section, so a section silently dropped from the
    machine-readable output fails here rather than in a consumer."""
    _code, out, _err = _run(
        [WALLET, "--cached-graph", str(cached_graph), "--no-ml", "--json"]
    )
    payload = json.loads(out)
    for key in (
        # header
        "wallet", "chain", "investigation_id", "started_at_utc",
        "duration_seconds", "provenance", "parameters",
        # section 1
        "normalization", "graph_summary", "transfer_count", "wallet_in_graph",
        # sections 2-4
        "attribution", "entities", "counterparties",
        # section 5
        "behavior_patterns",
        # section 6
        "temporal",
        # section 7
        "ml",
        # sections 8-9
        "risk", "seed_dataset_path", "seed_entry_count",
        "seed_provenance_counts", "conclusion", "limitations", "warnings",
    ):
        assert key in payload, key
    assert payload["ml"]["approach"] == "DISABLED"
    assert payload["attribution"] is not None
    assert payload["risk"] is not None
    assert payload["temporal"] is not None
    assert payload["limitations"], "a real investigation always has limitations"


def test_json_to_a_file_writes_it_and_still_prints_the_report(cached_graph, tmp_path):
    destination = tmp_path / "nested" / "report.json"
    code, out, _err = _run(
        [
            WALLET,
            "--cached-graph", str(cached_graph),
            "--no-ml",
            "--json", str(destination),
        ]
    )
    assert code == 0
    assert destination.exists(), "the parent directory must be created"
    assert "[1] WALLET" in out, "a file destination must not suppress the report"
    assert json.loads(destination.read_text(encoding="utf-8"))["wallet"] == WALLET


def test_verbose_progress_goes_to_stderr_only(cached_graph):
    _code, out, err = _run(
        [WALLET, "--cached-graph", str(cached_graph), "--no-ml", "--verbose", "--json"]
    )
    assert "[investigate]" in err
    assert "[investigate]" not in out
    json.loads(out)  # stdout is still pure JSON


def test_wallet_absent_from_the_graph_still_completes(cached_graph):
    code, out, _err = _run(
        ["0x" + "99" * 20, "--cached-graph", str(cached_graph), "--no-ml"]
    )
    assert code == 0, "an address with no activity is a finding, not a failure"
    assert "INVESTIGATION COMPLETE" in out


def test_no_ml_is_stated_in_both_reports(cached_graph):
    """A skipped ML stage must be visible, not merely absent. The compact
    report says so in one line; the full report keeps the whole section."""
    _code, compact, _err = _run([WALLET, "--cached-graph", str(cached_graph), "--no-ml"])
    assert "ML:" in compact
    assert "DISABLED" in compact.upper()

    _code, full, _err = _run(
        [WALLET, "--cached-graph", str(cached_graph), "--no-ml", "--full-report"]
    )
    section = full[full.find("SECTION 7") : full.find("SECTION 8")]
    assert "DISABLED" in section.upper()


def test_live_run_without_credentials_exits_1_and_offers_no_demo(monkeypatch):
    monkeypatch.setenv("ETHERSCAN_API_KEY", "")
    code, _out, err = _run([WALLET, "--no-ml"])
    assert code == 1
    assert "ETHERSCAN_API_KEY" in err
    assert "--demo" not in err
    assert "Traceback" not in err
