"""
Production transaction acquisition — pagination, budgets, and honest
completeness accounting.

This is the only place in the codebase that decides how many provider
requests an investigation makes and what happens when one of them fails.

Rules it enforces (each maps to a stated requirement)
-----------------------------------------------------
1. FULL PAGINATION, not page 1. Pages are walked until a short page is
   returned, the per-investigation record budget is reached, or the
   provider's result window is exhausted.
2. RESULT-WINDOW WALKING. Etherscan (like most explorers) refuses to serve
   results past a fixed offset (page * offset > 10000). Rather than
   pretending 10,000 records is the whole history, the fetcher advances
   `startblock` to the highest block it has already seen and restarts
   paging from page 1 — the standard, provider-documented way to read past
   the window. Overlapping records from the boundary block are removed by
   occurrence-aware de-duplication, so nothing is double counted and no
   genuinely distinct event is discarded.
3. NO SILENT PARTIAL DATA. A provider failure part-way through never
   `break`s away leaving the caller to believe the dataset is complete.
   `FetchOutcome.complete` is False and `FetchOutcome.error` carries the
   exact reason; if the very first request fails, an IngestionError is
   raised because there is no data to reason about at all. Downstream, an
   incomplete dataset forces attribution to INCONCLUSIVE rather than NONE.
4. NO CREDENTIAL IN OUTPUT. Error text is built from the provider's own
   exception message; provider implementations never put the API key in
   one, and nothing here echoes request parameters.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from inspect import signature
from typing import Any, Awaitable, Callable, Optional

from app.blockchain.base import (
    BlockchainProvider,
    ProviderError,
    RateLimitError,
)

# Provider-imposed ceiling on `page * offset`. Reading beyond it requires
# advancing the block range, which is what _advance_window does.
RESULT_WINDOW_LIMIT = 10_000

FetchFn = Callable[..., Awaitable[list[dict[str, Any]]]]


class IngestionError(RuntimeError):
    """Acquisition could not produce any usable data for a stream."""


@dataclass
class FetchOutcome:
    """One transaction stream's acquisition result and its provenance."""

    stream: str
    records: list[dict[str, Any]] = field(default_factory=list)
    pages_fetched: int = 0
    windows_used: int = 1
    duplicates_dropped: int = 0
    malformed_dropped: int = 0
    complete: bool = True
    budget_reached: bool = False
    supported: bool = True
    error: Optional[str] = None

    @property
    def count(self) -> int:
        return len(self.records)

    def describe(self) -> str:
        if not self.supported:
            return f"{self.stream}: not supported by this provider"
        parts = [f"{self.count} records", f"{self.pages_fetched} page(s)"]
        if self.windows_used > 1:
            parts.append(f"{self.windows_used} block window(s)")
        if self.duplicates_dropped:
            parts.append(f"{self.duplicates_dropped} overlap duplicate(s) removed")
        if self.malformed_dropped:
            parts.append(f"{self.malformed_dropped} malformed record(s) skipped")
        if self.budget_reached:
            parts.append("record budget reached")
        if not self.complete:
            parts.append(f"INCOMPLETE: {self.error}")
        return f"{self.stream}: " + ", ".join(parts)


def _record_identity(record: dict[str, Any]) -> tuple:
    """Byte-identity of a raw provider record.

    Deliberately the FULL record, not just its hash: one transaction hash
    legitimately produces many token-transfer events, and collapsing them
    on hash alone would destroy real evidence. Two records are only treated
    as the same event when every field the provider returned is identical,
    which is exactly what an overlapping re-fetch produces.
    """
    return tuple(sorted((str(k), str(v)) for k, v in record.items()))


def _block_of(record: dict[str, Any]) -> Optional[int]:
    raw = record.get("blockNumber")
    if raw is None:
        return None
    try:
        return int(str(raw), 0) if str(raw).startswith("0x") else int(raw)
    except (TypeError, ValueError):
        return None


def _accepts_use_cache(fetch_fn: FetchFn) -> bool:
    """Whether a provider method exposes the cache-bypass parameter.

    Probed once from the signature rather than by catching TypeError at call
    time — catching TypeError would also swallow a genuine TypeError raised
    from inside the provider and silently re-issue the request.
    """
    try:
        return "use_cache" in signature(fetch_fn).parameters
    except (TypeError, ValueError):
        return False


async def fetch_stream(
    fetch_fn: FetchFn,
    address: str,
    stream: str,
    max_records: int,
    page_size: int = 1000,
    start_block: int = 0,
    end_block: int = 99_999_999,
    use_cache: bool = True,
) -> FetchOutcome:
    """Walks every page (and every block window) of one transaction stream.

    Raises IngestionError only when the FIRST request fails outright, i.e.
    when zero records were obtained and the failure is therefore
    indistinguishable from "we never asked". Any later failure is recorded
    on the outcome as an explicit incompleteness.
    """
    outcome = FetchOutcome(stream=stream)
    accepted = Counter()
    window_start = start_block
    max_pages_per_window = max(1, RESULT_WINDOW_LIMIT // page_size)
    supports_cache_flag = _accepts_use_cache(fetch_fn)

    while True:
        window_counter: Counter = Counter()
        highest_block_this_window: Optional[int] = None
        page = 1
        window_exhausted_by_provider = False

        while page <= max_pages_per_window:
            if len(outcome.records) >= max_records:
                outcome.budget_reached = True
                return outcome

            kwargs: dict[str, Any] = {
                "start_block": window_start,
                "end_block": end_block,
                "page": page,
                "offset": page_size,
            }
            if supports_cache_flag:
                kwargs["use_cache"] = use_cache

            try:
                batch = await fetch_fn(address, **kwargs)
            except RateLimitError as exc:
                message = (
                    f"provider rate limit not cleared after configured retries "
                    f"({exc})"
                )
                if not outcome.records:
                    raise IngestionError(
                        f"Could not retrieve any {stream} records for {address}: "
                        f"{message}"
                    ) from exc
                outcome.complete = False
                outcome.error = message
                return outcome
            except ProviderError as exc:
                message = f"provider error after configured retries ({exc})"
                if not outcome.records:
                    raise IngestionError(
                        f"Could not retrieve any {stream} records for {address}: "
                        f"{message}"
                    ) from exc
                outcome.complete = False
                outcome.error = message
                return outcome

            outcome.pages_fetched += 1

            if not isinstance(batch, list):
                outcome.complete = False
                outcome.error = (
                    "provider returned a non-list page; refusing to guess its shape"
                )
                return outcome

            new_this_page = 0
            for record in batch:
                if not isinstance(record, dict):
                    outcome.malformed_dropped += 1
                    continue
                identity = _record_identity(record)
                window_counter[identity] += 1
                if window_counter[identity] <= accepted[identity]:
                    # Already accepted this exact occurrence from a previous,
                    # overlapping block window.
                    outcome.duplicates_dropped += 1
                    continue
                accepted[identity] = window_counter[identity]
                outcome.records.append(record)
                new_this_page += 1

                block = _block_of(record)
                if block is not None and (
                    highest_block_this_window is None
                    or block > highest_block_this_window
                ):
                    highest_block_this_window = block

                if len(outcome.records) >= max_records:
                    outcome.budget_reached = True
                    return outcome

            if len(batch) < page_size:
                # Short page == the provider has nothing further in this
                # window. This is the only clean "we are done" signal.
                return outcome

            if page == max_pages_per_window:
                window_exhausted_by_provider = True

            page += 1

        if not window_exhausted_by_provider:
            return outcome

        # The provider's result window is full. Advance the block range and
        # keep reading rather than reporting a truncated history as complete.
        if highest_block_this_window is None:
            outcome.complete = False
            outcome.error = (
                "provider result window exhausted and no block numbers were "
                "present in the records, so the block range could not be "
                "advanced"
            )
            return outcome
        if highest_block_this_window <= window_start and outcome.windows_used > 1:
            outcome.complete = False
            outcome.error = (
                f"a single block ({highest_block_this_window}) contains more "
                f"records than the provider's {RESULT_WINDOW_LIMIT}-result "
                "window can return; the remainder of that block cannot be read"
            )
            return outcome

        window_start = highest_block_this_window
        outcome.windows_used += 1


@dataclass
class AcquisitionResult:
    """Everything the acquisition stage produced for one wallet, including
    an explicit, aggregate statement of whether the dataset is complete."""

    address: str
    chain: str
    provider: str
    native: FetchOutcome
    token: FetchOutcome
    internal: FetchOutcome
    cache_stats: dict[str, int] = field(default_factory=dict)

    @property
    def outcomes(self) -> list[FetchOutcome]:
        return [self.native, self.token, self.internal]

    @property
    def total_records(self) -> int:
        return sum(o.count for o in self.outcomes)

    @property
    def complete(self) -> bool:
        """False if ANY stream was cut short — by a provider failure or by
        the per-investigation record budget. Callers must treat an
        incomplete dataset as grounds for INCONCLUSIVE, never NONE."""
        return all(
            o.complete and not o.budget_reached
            for o in self.outcomes
            if o.supported
        )

    @property
    def incompleteness_reasons(self) -> list[str]:
        reasons: list[str] = []
        for o in self.outcomes:
            if not o.supported:
                reasons.append(
                    f"{o.stream}: not available from provider '{self.provider}' — "
                    "value moved by internal contract calls is therefore not "
                    "represented in this dataset"
                )
                continue
            if not o.complete and o.error:
                reasons.append(f"{o.stream}: {o.error}")
            if o.budget_reached:
                reasons.append(
                    f"{o.stream}: stopped at the configured "
                    "MAX_TRANSACTIONS_PER_INVESTIGATION budget, so older "
                    "history was not read"
                )
        return reasons


async def acquire_wallet_transactions(
    provider: BlockchainProvider,
    address: str,
    max_records_per_stream: int,
    page_size: int = 1000,
    use_cache: bool = True,
) -> AcquisitionResult:
    """Acquires all three transaction streams for one wallet.

    Internal transfers are requested only from providers that advertise
    support; on providers that do not, the stream is marked unsupported so
    the gap appears as a stated limitation instead of an empty result that
    reads like "this wallet had none".
    """
    native = await fetch_stream(
        provider.get_normal_transactions,
        address,
        "native_transactions",
        max_records=max_records_per_stream,
        page_size=page_size,
        use_cache=use_cache,
    )
    token = await fetch_stream(
        provider.get_token_transfers,
        address,
        "token_transfers",
        max_records=max_records_per_stream,
        page_size=page_size,
        use_cache=use_cache,
    )

    if provider.supports_internal_transactions():
        internal = await fetch_stream(
            provider.get_internal_transactions,
            address,
            "internal_transactions",
            max_records=max_records_per_stream,
            page_size=page_size,
            use_cache=use_cache,
        )
    else:
        internal = FetchOutcome(
            stream="internal_transactions", supported=False, pages_fetched=0
        )

    cache_stats: dict[str, int] = {}
    cache = getattr(provider, "cache", None)
    if cache is not None and hasattr(cache, "stats"):
        cache_stats = cache.stats()

    return AcquisitionResult(
        address=address,
        chain=provider.chain,
        provider=provider.name,
        native=native,
        token=token,
        internal=internal,
        cache_stats=cache_stats,
    )
