"""
Tests for the InvestigationService acquisition path.

These exist because of two defects that were live in this service and are
invisible from the outside once they happen:

1. It never requested internal transactions, so value moved by contract calls
   was absent from every graph the HTTP API built.
2. It caught RateLimitError and ProviderAPIError from the middle of pagination
   and simply stopped. A wallet with 4000 transfers whose page 2 was rate
   limited produced a 1000-transfer graph that the API then reported as the
   wallet's complete history -- so "no VASP connection found" was
   indistinguishable from a real negative.

Fully offline: every provider here is a fake. No network, no API key.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.blockchain.base import (
    BlockchainProvider,
    ProviderAPIError,
    ProviderUnavailableError,
    RateLimitError,
)
from app.core.config import Settings
from app.db.repository import InvestigationRepository
from app.investigation.errors import GraphNotFoundError, InternalServiceFailure
from app.investigation.service import InvestigationService

WALLET = "0x1111111111111111111111111111111111111111"
OTHER = "0x2222222222222222222222222222222222222222"
INTERNAL_PEER = "0x3333333333333333333333333333333333333333"


def _native(index: int, to_address: str = OTHER) -> dict[str, Any]:
    return {
        "hash": f"0x{index:064x}",
        "blockNumber": str(100 + index),
        "timeStamp": str(1_700_000_000 + index),
        "from": WALLET,
        "to": to_address,
        "value": "1000000000000000000",
        "isError": "0",
        "gasUsed": "21000",
        "gasPrice": "1000000000",
    }


class _BaseFake(BlockchainProvider):
    """Minimal provider. Records which streams were asked for."""

    def __init__(self, settings: Settings, chain_name: str = "ethereum"):
        self._settings = settings
        self._chain_name = chain_name
        self.streams_called: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def chain(self) -> str:
        return self._chain_name

    def validate_address(self, address: str) -> bool:
        return bool(address) and address.startswith("0x") and len(address) == 42

    async def get_normal_transactions(
        self, address, start_block=0, end_block=99_999_999, page=1, offset=1000
    ):
        self.streams_called.append("native")
        return [_native(1)] if page == 1 else []

    async def get_token_transfers(
        self, address, start_block=0, end_block=99_999_999, page=1, offset=1000
    ):
        self.streams_called.append("token")
        return []

    async def aclose(self) -> None:
        return None


class InternalCapableFake(_BaseFake):
    """Advertises internal-transaction support and returns one."""

    def supports_internal_transactions(self) -> bool:
        return True

    async def get_internal_transactions(
        self, address, start_block=0, end_block=99_999_999, page=1, offset=1000
    ):
        self.streams_called.append("internal")
        if page != 1:
            return []
        return [
            {
                "hash": f"0x{99:064x}",
                "blockNumber": "150",
                "timeStamp": "1700000500",
                "from": WALLET,
                "to": INTERNAL_PEER,
                "value": "500000000000000000",
                "isError": "0",
            }
        ]


class RateLimitedPageTwoFake(_BaseFake):
    """Returns a full first page, then rate-limits forever.

    A full page means "there is more"; the second page never arrives. The old
    code returned the first page and declared the dataset complete.
    """

    async def get_normal_transactions(
        self, address, start_block=0, end_block=99_999_999, page=1, offset=1000
    ):
        self.streams_called.append(f"native:p{page}")
        if page == 1:
            return [_native(i) for i in range(offset)]
        raise RateLimitError("rate limit")


class TotallyFailingFake(_BaseFake):
    """Fails on the very first request, so nothing at all is known."""

    async def get_normal_transactions(
        self, address, start_block=0, end_block=99_999_999, page=1, offset=1000
    ):
        raise ProviderAPIError("provider down")


class EmptyFake(_BaseFake):
    async def get_normal_transactions(
        self, address, start_block=0, end_block=99_999_999, page=1, offset=1000
    ):
        return []


class _NullRepository(InvestigationRepository):
    """Captures what would be persisted without needing a database."""

    def __init__(self) -> None:  # noqa: D107 - deliberately no super().__init__
        self.created: list[dict[str, Any]] = []

    def create_investigation(self, **kwargs):
        self.created.append(kwargs)

        class _Record:
            pass

        record = _Record()
        for key, value in kwargs.items():
            setattr(record, key, value)
        record.created_at = None
        return record

    def create_attribution(self, **kwargs):
        return None

    def create_ml_prediction(self, **kwargs):
        return None


def _service(settings: Settings, provider: BlockchainProvider) -> InvestigationService:
    return InvestigationService(
        settings,
        _NullRepository(),
        provider_factory=lambda _settings, _chain: provider,
    )


@pytest.fixture()
def settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("GRAPH_CACHE_DIR", str(tmp_path / "graphs"))
    monkeypatch.setenv("PROVIDER_CACHE_ENABLED", "false")
    from app.core.config import get_settings

    return get_settings()


@pytest.mark.asyncio
async def test_internal_transaction_stream_is_requested(settings):
    """The stream the old implementation never asked for."""
    provider = InternalCapableFake(settings)
    result = await _service(settings, provider).run_investigation(wallet=WALLET)

    assert "internal" in provider.streams_called
    assert result.data_complete is True
    assert result.incompleteness_reasons == []


@pytest.mark.asyncio
async def test_internal_transfers_reach_the_graph(settings, tmp_path):
    """Requesting the stream is useless if the records are then dropped."""
    provider = InternalCapableFake(settings)
    result = await _service(settings, provider).run_investigation(wallet=WALLET)

    from app.graph.builder import load_graph

    graph = load_graph(result.graph_path)
    assert graph.has_edge(WALLET, INTERNAL_PEER), (
        "the internal transfer is absent from the graph, so contract-moved "
        "value would be invisible to tracing"
    )


@pytest.mark.asyncio
async def test_rate_limit_mid_pagination_is_reported_not_swallowed(settings):
    """The core regression: a truncated dataset must never look complete."""
    provider = RateLimitedPageTwoFake(settings)
    result = await _service(settings, provider).run_investigation(wallet=WALLET)

    assert result.data_complete is False
    assert result.incompleteness_reasons, (
        "the dataset was cut short by a rate limit and the result claims no "
        "reason for incompleteness"
    )
    assert any("rate limit" in reason.lower() for reason in result.incompleteness_reasons)


@pytest.mark.asyncio
async def test_total_provider_failure_is_a_service_failure_not_an_empty_result(settings):
    """Zero records because the provider was down is not 'this wallet is idle'."""
    provider = TotallyFailingFake(settings)
    with pytest.raises(InternalServiceFailure):
        await _service(settings, provider).run_investigation(wallet=WALLET)


@pytest.mark.asyncio
async def test_genuinely_empty_wallet_raises_graph_not_found(settings):
    provider = EmptyFake(settings)
    with pytest.raises(GraphNotFoundError):
        await _service(settings, provider).run_investigation(wallet=WALLET)


@pytest.mark.asyncio
async def test_service_failure_message_names_no_credential(settings):
    """An API error body must never carry configuration or key material."""
    provider = TotallyFailingFake(settings)
    with pytest.raises(InternalServiceFailure) as excinfo:
        await _service(settings, provider).run_investigation(wallet=WALLET)

    message = str(excinfo.value).lower()
    for forbidden in ("apikey", "api_key", "etherscan_api_key", "traceback", "token="):
        assert forbidden not in message


@pytest.mark.asyncio
async def test_missing_credential_is_a_configuration_fault_not_unreachability(
    settings,
):
    """The factory raising ProviderUnavailableError must not be reported as a
    network problem -- that sends an operator to debug the wrong layer."""

    def refusing_factory(_settings: Settings, _chain: str) -> BlockchainProvider:
        raise ProviderUnavailableError(
            "ETHERSCAN_API_KEY is not configured, so this provider cannot "
            "retrieve real blockchain data."
        )

    service = InvestigationService(
        settings=settings,
        repository=_NullRepository(),
        provider_factory=refusing_factory,
    )

    with pytest.raises(InternalServiceFailure) as excinfo:
        await service.run_investigation(wallet=WALLET)

    message = str(excinfo.value)
    assert "not configured" in message
    assert "could not be reached" not in message
    assert "ETHERSCAN_API_KEY" in message, "the operator needs to know which setting"
    # Naming the variable is required; carrying a value never is.
    assert "=" not in message.split("ETHERSCAN_API_KEY")[1][:2]


def test_real_provider_refuses_to_construct_without_a_key(monkeypatch):
    """Defence in depth: the pipeline checks the key before constructing, but
    any other caller that forgets must fail here rather than issue requests
    the provider will reject."""
    from app.blockchain.etherscan import EtherscanProvider

    monkeypatch.setenv("ETHERSCAN_API_KEY", "")
    from app.core.config import get_settings

    with pytest.raises(ProviderUnavailableError) as excinfo:
        EtherscanProvider(get_settings())

    message = str(excinfo.value)
    assert "ETHERSCAN_API_KEY" in message
    assert "synthetic" in message.lower() or "demonstration" in message.lower()
