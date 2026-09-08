"""
Etherscan implementation of BlockchainProvider.

Endpoint: Etherscan's V2 unified API — one host for every supported chain,
selected by the `chainid` query parameter (see ETHERSCAN_BASE_URL /
ETHERSCAN_CHAIN_ID in .env.example). Adding another EVM chain is a config
change, not a code change; adding a non-EVM chain means a new provider
class implementing the same interface. All vendor-specific request/response
handling is isolated to this file.

Reliability behaviour (production ingestion)
--------------------------------------------
* Retries are attempted `HTTP_MAX_RETRIES` times (from central config, not
  a hardcoded constant) with exponential backoff, and cover BOTH rate
  limiting AND transient transport failures (timeouts, connection resets) —
  a timeout is retried rather than immediately failing the investigation.
* Etherscan reports application-level errors inside an HTTP 200 body via
  `status == "0"`. "No transactions found" / "No records found" are genuine
  empty results, not errors; anything else is raised.
* A malformed body (non-JSON, or JSON that isn't an object, or a `result`
  that isn't a list where a list is required) raises ProviderAPIError
  rather than being silently coerced to an empty list — an empty list and
  "the provider answered with something we could not parse" must never be
  reported as the same thing.
* Raw payloads are cached through app.blockchain.cache under a
  deterministic key that never contains the API key.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.blockchain.base import (
    BlockchainProvider,
    InvalidAddressError,
    ProviderAPIError,
    ProviderUnavailableError,
    RateLimitError,
    UnsupportedChainError,
)
from app.blockchain.cache import ProviderResponseCache
from app.blockchain.chains import resolve_chain
from app.core.config import Settings

_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")

# Etherscan's own maximum page size. Requesting more is silently clamped by
# the API, which would make pagination think a short page meant "last page".
MAX_PAGE_SIZE = 1000

_EMPTY_RESULT_MESSAGES = frozenset(
    {"No transactions found", "No records found", "No internal transactions found"}
)


class RetryableTransportError(ProviderAPIError):
    """A transport-level failure that is worth retrying (timeout, connection
    reset, DNS blip). Subclasses ProviderAPIError so existing callers that
    only know about ProviderAPIError keep working unchanged once retries
    are exhausted."""


class EtherscanProvider(BlockchainProvider):
    def __init__(
        self,
        settings: Settings,
        chain_name: str = "ethereum",
        cache: Optional[ProviderResponseCache] = None,
        refresh: bool = False,
    ):
        # Refuse to exist without a credential rather than issuing requests
        # that the provider will reject. Without this, a caller that forgot to
        # check reports "the provider could not be reached" for what is really
        # a missing setting, and the operator debugs the network instead of
        # the configuration. The message names the variable, never its value.
        if not settings.etherscan_api_key:
            raise ProviderUnavailableError(
                "ETHERSCAN_API_KEY is not configured, so this provider cannot "
                "retrieve real blockchain data. No demonstration or synthetic "
                "data is substituted."
            )

        self._settings = settings
        # Resolve the chain NAME to a chain id here rather than trusting
        # ETHERSCAN_CHAIN_ID to agree with it. The two used to be independent:
        # `chain_name` was a label carried into every NormalizedTransfer and
        # the report header, while the id actually queried came from config, so
        # `--chain dogecoin` with the default id returned Ethereum transactions
        # that the whole report called Dogecoin. An unresolvable name now fails
        # before the first request.
        self._chain = resolve_chain(chain_name)
        self._chain_name = self._chain.name
        # An explicit ETHERSCAN_CHAIN_ID that contradicts the named chain is a
        # configuration fault, not a preference to honour silently: one of the
        # two is wrong and the operator has to say which.
        configured = getattr(settings, "etherscan_chain_id", None)
        if configured is not None and int(configured) != self._chain.chain_id:
            raise UnsupportedChainError(
                f"ETHERSCAN_CHAIN_ID={configured} does not match chain "
                f"'{self._chain_name}', whose Etherscan chain id is "
                f"{self._chain.chain_id}. Querying one chain while labelling "
                "the results as another would produce a report that states a "
                "chain the data did not come from. Fix ETHERSCAN_CHAIN_ID in "
                ".env or pass the chain that matches it."
            )
        # `refresh` is a whole-run decision, not a per-request one, so it lives
        # on the provider rather than in every call signature. It suppresses
        # cache READS while leaving WRITES intact, which is what "re-fetch"
        # means: the fresh response replaces the stale entry. Suppressing the
        # write too would make a refresh permanently expensive, because the
        # next run would find the cache still empty.
        self._refresh = refresh
        self._client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
        if cache is not None:
            self._cache = cache
        else:
            self._cache = ProviderResponseCache(
                directory=settings.provider_cache_dir,
                ttl_seconds=settings.provider_cache_ttl_seconds,
                enabled=settings.provider_cache_enabled,
            )
        # tenacity needs the attempt count at decoration time, but the count
        # comes from config, so the retry wrapper is built per instance.
        attempts = max(1, int(settings.http_max_retries))
        self._get_with_retry = retry(
            reraise=True,
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(
                (RateLimitError, RetryableTransportError)
            ),
        )(self._get_once)

    @property
    def name(self) -> str:
        return "etherscan"

    @property
    def chain(self) -> str:
        return self._chain_name

    @property
    def cache(self) -> ProviderResponseCache:
        return self._cache

    def validate_address(self, address: str) -> bool:
        return bool(_ADDRESS_RE.match(address or ""))

    def supports_internal_transactions(self) -> bool:
        return True

    async def aclose(self) -> None:
        await self._client.aclose()

    def _base_params(self) -> dict[str, Any]:
        # The chain id comes from the resolved chain, not straight from config,
        # so the id queried and the name written onto every transfer can never
        # disagree. The constructor has already rejected a configured id that
        # contradicts the name.
        return {
            "chainid": self._chain.chain_id,
            "apikey": self._settings.etherscan_api_key,
        }

    async def _get_once(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._client.get(
                self._settings.etherscan_base_url, params=params
            )
        except httpx.TimeoutException as exc:
            raise RetryableTransportError(
                f"Etherscan request timed out: {exc}"
            ) from exc
        except httpx.TransportError as exc:
            # Connection reset / DNS / TLS handshake failures are transient
            # far more often than they are permanent.
            raise RetryableTransportError(
                f"Etherscan transport error: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderAPIError(f"Etherscan HTTP error: {exc}") from exc

        if resp.status_code == 429:
            raise RateLimitError("Etherscan rate limit hit")
        if 500 <= resp.status_code < 600:
            raise RetryableTransportError(
                f"Etherscan returned server error {resp.status_code}",
                status_code=resp.status_code,
            )
        if resp.status_code != 200:
            raise ProviderAPIError(
                f"Etherscan returned status {resp.status_code}",
                status_code=resp.status_code,
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderAPIError(
                "Etherscan returned a body that is not valid JSON"
            ) from exc

        if not isinstance(data, dict):
            raise ProviderAPIError(
                f"Etherscan returned a {type(data).__name__} where a JSON "
                "object was expected"
            )

        # Etherscan signals errors inside a 200 response via status/message fields.
        if data.get("status") == "0" and data.get("message") not in _EMPTY_RESULT_MESSAGES:
            msg = str(data.get("result") or data.get("message") or "unknown error")
            lowered = msg.lower()
            if "rate limit" in lowered or "max calls per sec" in lowered:
                raise RateLimitError(msg)
            raise ProviderAPIError(f"Etherscan API error: {msg}")

        return data

    async def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self._get_with_retry(params)

    def _identity(self, params: dict[str, Any]) -> dict[str, Any]:
        """The cache identity for a request — the query minus credentials,
        plus the provider/chain so two chains never share an entry."""
        identity = {
            k: v
            for k, v in params.items()
            if k not in ("apikey", "api_key", "key", "token")
        }
        identity["_provider"] = self.name
        identity["_chain"] = self._chain_name
        return identity

    async def _fetch_list(
        self,
        params: dict[str, Any],
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """Runs one provider query (through the cache when enabled) and
        returns its `result` list. Raises rather than returning [] when the
        payload's shape is wrong.

        `use_cache=False` disables the cache entirely (--no-cache). The
        provider-level `refresh` flag (--refresh) is narrower: it skips the
        read so the provider is really queried, then still stores the answer.
        """
        identity = self._identity(params)

        if use_cache and not self._refresh:
            cached = self._cache.get(identity)
            if cached is not None:
                payload = cached.get("payload")
                if isinstance(payload, list):
                    return payload

        data = await self._get(params)
        result = data.get("result")

        if result is None and data.get("status") == "0":
            # A legitimate "nothing found" response.
            result = []
        if isinstance(result, str):
            # Etherscan puts error text in `result` for some failures; that
            # was already handled above, so a bare string here is malformed.
            raise ProviderAPIError(
                "Etherscan returned a string result where a list of records "
                "was expected"
            )
        if not isinstance(result, list):
            raise ProviderAPIError(
                f"Etherscan returned a {type(result).__name__} result where a "
                "list of records was expected"
            )

        records = [r for r in result if isinstance(r, dict)]

        if use_cache:
            self._cache.set(identity, records)

        return records

    async def get_normal_transactions(
        self,
        address: str,
        start_block: int = 0,
        end_block: int = 99_999_999,
        page: int = 1,
        offset: int = 1000,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        return await self._account_query(
            "txlist", address, start_block, end_block, page, offset, use_cache
        )

    async def get_token_transfers(
        self,
        address: str,
        start_block: int = 0,
        end_block: int = 99_999_999,
        page: int = 1,
        offset: int = 1000,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        return await self._account_query(
            "tokentx", address, start_block, end_block, page, offset, use_cache
        )

    async def get_internal_transactions(
        self,
        address: str,
        start_block: int = 0,
        end_block: int = 99_999_999,
        page: int = 1,
        offset: int = 1000,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """Internal (contract-executed) native-value transfers. These carry
        real value movement that `txlist` does not show at all — omitting
        them would make any wallet that interacts with contracts look like
        it received or sent nothing."""
        return await self._account_query(
            "txlistinternal", address, start_block, end_block, page, offset, use_cache
        )

    async def _account_query(
        self,
        action: str,
        address: str,
        start_block: int,
        end_block: int,
        page: int,
        offset: int,
        use_cache: bool,
    ) -> list[dict[str, Any]]:
        if not self.validate_address(address):
            raise InvalidAddressError(f"Invalid Ethereum address: {address}")

        params = {
            **self._base_params(),
            "module": "account",
            "action": action,
            "address": address,
            "startblock": start_block,
            "endblock": end_block,
            "page": page,
            "offset": min(int(offset), MAX_PAGE_SIZE),
            "sort": "asc",
        }
        return await self._fetch_list(params, use_cache=use_cache)
