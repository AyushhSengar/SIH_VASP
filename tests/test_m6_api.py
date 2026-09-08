"""
MACRO MILESTONE 6 tests — FastAPI service.

Fully offline: a FakeProvider stands in for EtherscanProvider via the
get_provider_factory dependency override, so nothing here ever touches
the network. Each test gets an isolated SQLite database file under a
pytest tmp_path, so tests never share state.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.blockchain.base import BlockchainProvider
from app.core.config import Settings

WALLET_WITH_DATA = "0x1111111111111111111111111111111111111111"
WALLET_TO_DEMO_VASP = "0x7777777777777777777777777777777777777777"
DEMO_VASP_ADDRESS = "0x4444444444444444444444444444444444444444"
WALLET_NO_DATA = "0x9999999999999999999999999999999999999999"
INVALID_WALLET = "not-a-wallet"

_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")

# address (lowercased) -> (native_raw, token_raw)
_CANNED_DATA: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {
    WALLET_WITH_DATA: (
        [
            {
                "hash": "0xtx1",
                "blockNumber": "100",
                "timeStamp": "1700000000",
                "from": WALLET_WITH_DATA,
                "to": "0x2222222222222222222222222222222222222222",
                "value": "1000000000000000000",
                "isError": "0",
                "gasUsed": "21000",
                "gasPrice": "1000000000",
            }
        ],
        [],
    ),
    WALLET_TO_DEMO_VASP: (
        [
            {
                "hash": "0xtx2",
                "blockNumber": "101",
                "timeStamp": "1700000100",
                "from": WALLET_TO_DEMO_VASP,
                "to": DEMO_VASP_ADDRESS,
                "value": "2000000000000000000",
                "isError": "0",
                "gasUsed": "21000",
                "gasPrice": "1000000000",
            }
        ],
        [],
    ),
}


class FakeProvider(BlockchainProvider):
    """Stands in for EtherscanProvider. Returns pre-canned raw
    transactions for known addresses and nothing for anyone else —
    fully deterministic, fully offline."""

    def __init__(self, settings: Settings, chain_name: str = "ethereum"):
        self._settings = settings
        self._chain_name = chain_name

    @property
    def name(self) -> str:
        return "fake"

    @property
    def chain(self) -> str:
        return self._chain_name

    def validate_address(self, address: str) -> bool:
        return bool(_ADDRESS_RE.match(address or ""))

    async def get_normal_transactions(
        self, address, start_block=0, end_block=99_999_999, page=1, offset=1000
    ):
        if page != 1:
            return []
        native, _token = _CANNED_DATA.get(address.lower(), ([], []))
        return native

    async def get_token_transfers(
        self, address, start_block=0, end_block=99_999_999, page=1, offset=1000
    ):
        if page != 1:
            return []
        _native, token = _CANNED_DATA.get(address.lower(), ([], []))
        return token

    async def aclose(self) -> None:
        return None


def _fake_provider_factory(settings: Settings, chain: str) -> BlockchainProvider:
    return FakeProvider(settings, chain_name=chain)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "m6_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    # Keep every graph this test writes inside the pytest tmp_path. Without
    # this the suite would persist throwaway graphs into the production
    # graph cache (data/graphs), where a later real investigation could
    # pick them up — see the NOTE in app/investigation/service.py.
    monkeypatch.setenv("GRAPH_CACHE_DIR", str(tmp_path / "graphs"))

    # Import after DATABASE_URL is set so app.state ends up pointed at the
    # isolated per-test database when the lifespan runs.
    from app.api.dependencies import get_provider_factory
    from app.api.main import app

    app.dependency_overrides[get_provider_factory] = lambda: _fake_provider_factory

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------
# health
# ---------------------------------------------------------------------


def test_health_with_working_database(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"status": "ok", "database": "connected"}


def test_health_with_unavailable_database(client: TestClient, monkeypatch):
    import app.api.routes.health as health_module

    monkeypatch.setattr(
        health_module, "check_database_connection", lambda database_url: False
    )
    resp = client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["database"] == "unavailable"


# ---------------------------------------------------------------------
# POST /investigations
# ---------------------------------------------------------------------


def test_create_investigation_valid_request_returns_201(client: TestClient):
    resp = client.post(
        "/investigations",
        json={"wallet": WALLET_WITH_DATA, "chain": "ethereum"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["wallet"] == WALLET_WITH_DATA
    assert body["chain"] == "ethereum"
    assert "investigation_id" in body and body["investigation_id"]
    assert "created_at" in body
    assert body["training_data_type"] == "SYNTHETIC_DEMO"
    assert set(body["m4_evidence"].keys()) == {"status", "candidate_count", "search_truncated"}
    assert set(body["m5_ml_prediction"].keys()) == {
        "predicted_label",
        "training_data_type",
        "model_name",
        "model_version",
    }


def test_create_investigation_invalid_wallet_returns_422(client: TestClient):
    resp = client.post(
        "/investigations", json={"wallet": INVALID_WALLET, "chain": "ethereum"}
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "INVALID_WALLET"


def test_create_investigation_unsupported_chain_returns_422(client: TestClient):
    resp = client.post(
        "/investigations", json={"wallet": WALLET_WITH_DATA, "chain": "not-a-real-chain"}
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "UNSUPPORTED_CHAIN"


def test_create_investigation_missing_graph_returns_400(client: TestClient):
    resp = client.post(
        "/investigations", json={"wallet": WALLET_NO_DATA, "chain": "ethereum"}
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "GRAPH_NOT_FOUND"


def test_create_investigation_ml_stays_synthetic_demo(client: TestClient):
    resp = client.post(
        "/investigations",
        json={"wallet": WALLET_WITH_DATA, "chain": "ethereum", "ml_seed": 42},
    )
    assert resp.status_code == 201
    assert resp.json()["m5_ml_prediction"]["training_data_type"] == "SYNTHETIC_DEMO"


# ---------------------------------------------------------------------
# GET /investigations/{id}
# ---------------------------------------------------------------------


def test_get_investigation_summary(client: TestClient):
    create_resp = client.post(
        "/investigations", json={"wallet": WALLET_WITH_DATA, "chain": "ethereum"}
    )
    investigation_id = create_resp.json()["investigation_id"]

    resp = client.get(f"/investigations/{investigation_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["investigation_id"] == investigation_id
    assert body["wallet"] == WALLET_WITH_DATA
    assert body["chain"] == "ethereum"
    assert "graph_path" in body
    assert "max_hops" in body
    assert "search_truncated" in body
    assert "attribution_status" in body
    assert "ml_predicted_label" in body
    assert "created_at" in body


def test_get_unknown_investigation_returns_404(client: TestClient):
    resp = client.get("/investigations/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"] == "INVESTIGATION_NOT_FOUND"


# ---------------------------------------------------------------------
# GET /investigations/{id}/attribution and /ml
# ---------------------------------------------------------------------


def test_get_attribution_returns_complete_m4_result(client: TestClient):
    create_resp = client.post(
        "/investigations",
        json={"wallet": WALLET_TO_DEMO_VASP, "chain": "ethereum", "use_demo_seed": True},
    )
    investigation_id = create_resp.json()["investigation_id"]

    resp = client.get(f"/investigations/{investigation_id}/attribution")
    assert resp.status_code == 200
    body = resp.json()

    # Complete AttributionResult shape.
    assert set(body.keys()) == {
        "wallet",
        "status",
        "candidates",
        "max_hops",
        "search_truncated",
        "notes",
    }
    assert body["wallet"] == WALLET_TO_DEMO_VASP
    assert body["status"] == "MATCH_FOUND"
    assert len(body["candidates"]) >= 1
    candidate = body["candidates"][0]
    assert candidate["matched_address"] == DEMO_VASP_ADDRESS

    # M4 attribution must contain no ML fields whatsoever.
    forbidden_ml_keys = {"predicted_label", "training_data_type", "model_name", "disclaimer"}
    assert forbidden_ml_keys.isdisjoint(body.keys())
    for c in body["candidates"]:
        assert forbidden_ml_keys.isdisjoint(c.keys())


def test_get_ml_prediction_returns_complete_m5_result(client: TestClient):
    create_resp = client.post(
        "/investigations", json={"wallet": WALLET_WITH_DATA, "chain": "ethereum"}
    )
    investigation_id = create_resp.json()["investigation_id"]

    resp = client.get(f"/investigations/{investigation_id}/ml")
    assert resp.status_code == 200
    body = resp.json()

    assert set(body.keys()) == {
        "wallet",
        "predicted_label",
        "training_data_type",
        "model_name",
        "model_version",
        "random_seed",
        "feature_snapshot",
        "disclaimer",
    }
    assert body["training_data_type"] == "SYNTHETIC_DEMO"
    assert body["disclaimer"]  # non-empty synthetic/demo disclaimer preserved

    # M5 prediction must contain no VASP attribution fields whatsoever.
    forbidden_attribution_keys = {"vasp_name", "matched_address", "evidence_tier", "candidates"}
    assert forbidden_attribution_keys.isdisjoint(body.keys())


def test_attribution_and_ml_endpoints_never_merge_fields(client: TestClient):
    create_resp = client.post(
        "/investigations",
        json={"wallet": WALLET_TO_DEMO_VASP, "chain": "ethereum", "use_demo_seed": True},
    )
    investigation_id = create_resp.json()["investigation_id"]

    attribution_body = client.get(f"/investigations/{investigation_id}/attribution").json()
    ml_body = client.get(f"/investigations/{investigation_id}/ml").json()

    # The two response bodies must never merge fields — 'wallet' is the one
    # reasonable shared concept (both describe the same investigated
    # wallet), matching the same rule already enforced at the model level
    # (see tests/test_ml_predictor.py::test_ml_prediction_has_no_attribution_fields).
    assert set(attribution_body.keys()).isdisjoint(set(ml_body.keys()) - {"wallet"})


# ---------------------------------------------------------------------
# secrets never leak
# ---------------------------------------------------------------------


def test_responses_never_contain_secrets(client: TestClient, monkeypatch):
    monkeypatch.setenv("ETHERSCAN_API_KEY", "super-secret-key-should-never-leak")

    create_resp = client.post(
        "/investigations", json={"wallet": WALLET_WITH_DATA, "chain": "ethereum"}
    )
    investigation_id = create_resp.json()["investigation_id"]

    responses = [
        create_resp,
        client.get("/health"),
        client.get(f"/investigations/{investigation_id}"),
        client.get(f"/investigations/{investigation_id}/attribution"),
        client.get(f"/investigations/{investigation_id}/ml"),
        client.get("/investigations/nonexistent"),
        client.post("/investigations", json={"wallet": INVALID_WALLET}),
    ]
    for resp in responses:
        text = resp.text
        assert "super-secret-key-should-never-leak" not in text
        assert "ETHERSCAN_API_KEY" not in text
        assert "DATABASE_URL" not in text
        assert "sqlite:///" not in text
        assert "Traceback" not in text
