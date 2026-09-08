"""
Tests for recursive multi-hop acquisition (app/investigation/acquisition.py).

WHAT THESE PROVE
--------------------------------------------------------------------------
The claim this module makes is that `--max-hops N` causes N hop levels of
real provider data to be FETCHED, not that a report prints a larger number.
That claim is only worth anything if it is tested at the level of "which
addresses did the provider actually get asked about", so almost every test
here asserts on `provider.fetched` -- the list of addresses a fake provider
was queried for, in order.

Fully offline. Every provider is a fake serving raw Etherscan-shaped dicts,
so nothing here depends on a live Etherscan, an API key, or a network. The
chain layout each fixture serves is written out in the test that uses it.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.blockchain.base import (
    BlockchainProvider,
    ProviderAPIError,
    RateLimitError,
)
from app.blockchain.ingest import IngestionError
from app.investigation.acquisition import (
    STOP_ADDRESS_BUDGET,
    STOP_DEPTH_REACHED,
    STOP_NO_NEW_COUNTERPARTIES,
    acquire_multi_hop,
)

# Repeated-nibble addresses, so a failure message names a readable address and
# lexicographic order is obvious at a glance (a < b < c < d < e < f).
W = "0x" + "a" * 40  # hop 0 -- the investigated wallet
B = "0x" + "b" * 40
C = "0x" + "c" * 40
D = "0x" + "d" * 40
E = "0x" + "e" * 40
VASP = "0x" + "f" * 40  # a known-VASP seed address
ZERO_PEER = "0x" + "9" * 40  # only ever reached by a zero-value transfer

CHAIN = "ethereum"


def _tx(
    index: int,
    frm: str,
    to: str,
    *,
    value: str = "1000000000000000000",
    is_error: str = "0",
) -> dict[str, Any]:
    """One raw Etherscan `txlist` record.

    `index` drives the hash, block and timestamp together, so a chain built
    with increasing indices is chronologically consistent -- which the
    targeted path search requires and which a hand-written fixture gets
    wrong easily.
    """
    return {
        "hash": f"0x{index:064x}",
        "blockNumber": str(100 + index),
        "timeStamp": str(1_700_000_000 + index * 60),
        "from": frm,
        "to": to,
        "value": value,
        "isError": is_error,
        "gasUsed": "21000",
        "gasPrice": "1000000000",
    }


class ChainFake(BlockchainProvider):
    """Serves a fixed address -> native-records map and records every fetch.

    An address absent from the map returns an empty list, which is what a
    real provider does for an address with no activity -- so a test that
    walks off the end of its own fixture terminates rather than looping.

    Internal transfers are advertised as supported (Etherscan serves them)
    and return nothing, so a clean run carries no unsupported-stream
    limitation and `complete` means what the tests below say it means.
    """

    def __init__(self, ledger: dict[str, list[dict[str, Any]]]):
        self.ledger = {k.lower(): v for k, v in ledger.items()}
        self.fetched: list[str] = []
        self.calls: list[tuple[str, str, int]] = []

    @property
    def name(self) -> str:
        return "chainfake"

    @property
    def chain(self) -> str:
        return CHAIN

    def validate_address(self, address: str) -> bool:
        return bool(address) and address.startswith("0x") and len(address) == 42

    async def get_normal_transactions(
        self, address, start_block=0, end_block=99_999_999, page=1, offset=1000
    ):
        self.calls.append((address.lower(), "native", page))
        if page == 1:
            self.fetched.append(address.lower())
            return list(self.ledger.get(address.lower(), []))
        return []

    async def get_token_transfers(
        self, address, start_block=0, end_block=99_999_999, page=1, offset=1000
    ):
        self.calls.append((address.lower(), "token", page))
        return []

    def supports_internal_transactions(self) -> bool:
        return True

    async def get_internal_transactions(
        self, address, start_block=0, end_block=99_999_999, page=1, offset=1000
    ):
        self.calls.append((address.lower(), "internal", page))
        return []

    async def aclose(self) -> None:
        return None


async def _run(fake: BlockchainProvider, **overrides):
    """acquire_multi_hop with small, explicit test budgets.

    Defaults are generous enough not to interfere; a test that is about a
    budget sets that budget itself, so no test depends on a production
    default that could later change.
    """
    kwargs: dict[str, Any] = dict(
        max_hops=4,
        max_records_wallet=1000,
        max_records_expanded=1000,
        max_addresses=50,
        max_addresses_per_hop=50,
    )
    kwargs.update(overrides)
    return await acquire_multi_hop(fake, W, CHAIN, **kwargs)


def _hop(result, number: int):
    """The HopReport for one hop level. Raises if the hop never happened,
    which is itself the assertion most callers want."""
    return next(h for h in result.hops if h.hop == number)


# --------------------------------------------------------------------------
# 1. RECURSIVE DISCOVERY -- the core claim
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_counterparties_are_recursively_fetched():
    """W -> B -> C -> D. All four addresses must be queried, in hop order.

    This is the whole point of the module: before it existed only W was ever
    fetched, so the edges B->C and C->D could not be in the dataset at all.
    """
    fake = ChainFake(
        {
            W: [_tx(1, W, B)],
            B: [_tx(1, W, B), _tx(2, B, C)],
            C: [_tx(2, B, C), _tx(3, C, D)],
            D: [_tx(3, C, D)],
        }
    )
    result = await _run(fake)

    assert fake.fetched == [W, B, C, D]
    assert result.addresses_fetched == 4
    assert result.hops_expanded == 4
    # The edges that only a recursive fetch can produce are really present.
    edges = {(t.from_address, t.to_address) for t in result.transfers}
    assert (B, C) in edges
    assert (C, D) in edges


@pytest.mark.asyncio
async def test_hop_numbering_matches_actual_distance():
    """Each address is recorded at the hop it was really discovered at."""
    fake = ChainFake(
        {
            W: [_tx(1, W, B)],
            B: [_tx(1, W, B), _tx(2, B, C)],
            C: [_tx(2, B, C), _tx(3, C, D)],
            D: [_tx(3, C, D)],
        }
    )
    result = await _run(fake)

    hop_of = {a.address: a.hop for a in result.addresses}
    assert hop_of == {W: 0, B: 1, C: 2, D: 3}


@pytest.mark.asyncio
async def test_transfers_are_ordered_exactly_as_one_normalize_all_call():
    """Merging per-address batches must not change canonical ordering.

    Acquisition normalizes one address at a time (it has to read from/to to
    decide what to fetch next), so a merge is unavoidable. If the merge
    reordered anything, graph construction and tracing would stop being
    deterministic, and two runs of the same investigation could disagree.
    """
    fake = ChainFake(
        {
            W: [_tx(5, W, B), _tx(1, W, C)],
            B: [_tx(5, W, B), _tx(9, B, D)],
            C: [_tx(1, W, C), _tx(3, C, D)],
        }
    )
    result = await _run(fake)

    keys = [
        (t.timestamp, t.block_number, t.tx_hash, t.event_index or "",
         t.transfer_source.value)
        for t in result.transfers
    ]
    assert keys == sorted(keys)


# --------------------------------------------------------------------------
# 2. DEDUPLICATION AND CYCLES
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_address_is_fetched_once_even_when_many_peers_name_it():
    """C is a counterparty of both B and D, and is fetched exactly once.

    Deduplication is a request-budget question, not a cosmetic one: without
    it a hub address would be re-fetched once per neighbour.
    """
    fake = ChainFake(
        {
            W: [_tx(1, W, B), _tx(2, W, D)],
            B: [_tx(1, W, B), _tx(3, B, C)],
            D: [_tx(2, W, D), _tx(4, D, C)],
            C: [_tx(3, B, C), _tx(4, D, C)],
        }
    )
    result = await _run(fake)

    assert fake.fetched.count(C) == 1
    assert sorted(fake.fetched) == sorted([W, B, C, D])
    assert result.addresses_fetched == 4


@pytest.mark.asyncio
async def test_cycle_does_not_cause_refetch_or_hang():
    """W -> B -> C -> W. The cycle back to W must not re-fetch W."""
    fake = ChainFake(
        {
            W: [_tx(1, W, B), _tx(3, C, W)],
            B: [_tx(1, W, B), _tx(2, B, C)],
            C: [_tx(2, B, C), _tx(3, C, W)],
        }
    )
    result = await _run(fake)

    assert fake.fetched == [W, B, C]
    assert fake.fetched.count(W) == 1
    assert result.stop_reason == STOP_NO_NEW_COUNTERPARTIES


# --------------------------------------------------------------------------
# 3. MAX-HOP ENFORCEMENT -- the flag has to actually bound the fetching
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_hops_one_reproduces_the_single_address_fetch():
    """--max-hops 1 must fetch only the wallet.

    This is the pre-existing behaviour and it has to stay reachable: a
    one-hop investigation should not silently start crawling counterparties.
    """
    fake = ChainFake({W: [_tx(1, W, B)], B: [_tx(2, B, C)]})
    result = await _run(fake, max_hops=1)

    assert fake.fetched == [W]
    assert result.observation_depth == 1
    assert result.hops_expanded == 1
    assert result.stop_reason == STOP_DEPTH_REACHED


@pytest.mark.parametrize(
    "hops,expected",
    [
        (1, [W]),
        (2, [W, B]),
        (3, [W, B, C]),
        (4, [W, B, C, D]),
    ],
)
@pytest.mark.asyncio
async def test_each_requested_depth_fetches_exactly_that_many_hop_levels(
    hops, expected
):
    """The fetched address set grows with --max-hops, one hop level at a time."""
    fake = ChainFake(
        {
            W: [_tx(1, W, B)],
            B: [_tx(1, W, B), _tx(2, B, C)],
            C: [_tx(2, B, C), _tx(3, C, D)],
            D: [_tx(3, C, D), _tx(4, D, E)],
            E: [_tx(4, D, E)],
        }
    )
    result = await _run(fake, max_hops=hops)

    assert fake.fetched == expected
    assert result.observation_depth == hops
    assert result.requested_hops == hops
    assert result.stop_reason == STOP_DEPTH_REACHED


@pytest.mark.asyncio
async def test_addresses_at_the_requested_depth_are_recorded_not_expanded():
    """The frontier AT the requested depth is reported, not silently dropped.

    Those addresses are real graph nodes, so they count as discovered; they
    are not expanded, and that is the request being honoured rather than an
    incompleteness.
    """
    fake = ChainFake(
        {
            W: [_tx(1, W, B)],
            B: [_tx(1, W, B), _tx(2, B, C)],
            C: [_tx(2, B, C)],
        }
    )
    result = await _run(fake, max_hops=2)

    assert fake.fetched == [W, B]
    assert C not in fake.fetched
    assert _hop(result, 2).beyond_depth == [C]
    assert result.max_hop_reached == 2
    # Honouring the requested depth is not an incompleteness.
    assert result.complete is True
    assert result.incompleteness_reasons == []


@pytest.mark.asyncio
async def test_observed_depth_is_never_the_requested_depth_when_data_runs_out():
    """A wallet with one dead-end counterparty cannot report 4 observed hops."""
    fake = ChainFake({W: [_tx(1, W, B)], B: [_tx(1, W, B)]})
    result = await _run(fake, max_hops=4)

    assert fake.fetched == [W, B]
    assert result.requested_hops == 4
    assert result.hops_expanded == 2
    assert result.observation_depth == 2
    assert result.max_hop_reached == 1
    assert result.stop_reason == STOP_NO_NEW_COUNTERPARTIES


# --------------------------------------------------------------------------
# 4. VASP ENDPOINTS
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_known_vasp_reached_at_hop_three_is_recorded_with_its_distance():
    """W -> B -> C -> VASP: the seed address is found at its real hop.

    The whole investigative point of recursion. The hop distance is measured
    from the fetched data, not assumed from the requested depth.
    """
    fake = ChainFake(
        {
            W: [_tx(1, W, B)],
            B: [_tx(1, W, B), _tx(2, B, C)],
            C: [_tx(2, B, C), _tx(3, C, VASP)],
        }
    )
    result = await _run(fake, max_hops=4, terminal_addresses=[VASP])

    assert _hop(result, 3).terminal == [VASP]
    # Every edge of the route is present, so W -> B -> C -> VASP is
    # reconstructible from real fetched transactions rather than asserted.
    edges = {(t.from_address, t.to_address) for t in result.transfers}
    assert {(W, B), (B, C), (C, VASP)} <= edges


@pytest.mark.asyncio
async def test_known_vasp_is_not_expanded():
    """A seed address is an endpoint, not a route onward.

    Expanding an exchange hot wallet would spend the whole budget on
    transactions belonging to unrelated customers without strengthening the
    match, so it must never be fetched.
    """
    fake = ChainFake(
        {
            W: [_tx(1, W, VASP)],
            VASP: [_tx(1, W, VASP)] + [_tx(i, VASP, C) for i in range(2, 20)],
        }
    )
    result = await _run(fake, max_hops=4, terminal_addresses=[VASP])

    assert fake.fetched == [W]
    assert _hop(result, 1).terminal == [VASP]
    # Declining to expand an endpoint is a deliberate bound, not a data gap.
    assert result.complete is True
    assert any("known-VASP" in note for note in result.notes)


@pytest.mark.asyncio
async def test_terminal_matching_is_case_insensitive_and_exact():
    """Seed addresses match on exact value, case-folded -- never by prefix."""
    fake = ChainFake({W: [_tx(1, W, VASP)]})
    result = await _run(fake, max_hops=3, terminal_addresses=["0x" + "F" * 40])

    assert _hop(result, 1).terminal == [VASP]
    assert fake.fetched == [W]


@pytest.mark.asyncio
async def test_a_near_miss_address_is_not_treated_as_a_vasp():
    """An address sharing a 39-nibble prefix with a seed entry is NOT a match."""
    near_miss = VASP[:-1] + "0"
    fake = ChainFake(
        {W: [_tx(1, W, near_miss)], near_miss: [_tx(1, W, near_miss)]}
    )
    result = await _run(fake, max_hops=2, terminal_addresses=[VASP])

    assert _hop(result, 1).terminal == []
    # Not a terminal, so it is expanded like any other counterparty.
    assert near_miss in fake.fetched


# --------------------------------------------------------------------------
# 5. BUDGETS -- bounded, and never silently
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_address_budget_stops_expansion_and_says_so():
    """The budget bounds the crawl AND is reported, not silently applied."""
    fake = ChainFake(
        {
            W: [_tx(1, W, B), _tx(2, W, C)],
            B: [_tx(1, W, B), _tx(3, B, D)],
            C: [_tx(2, W, C), _tx(4, C, E)],
            D: [_tx(3, B, D)],
            E: [_tx(4, C, E)],
        }
    )
    result = await _run(fake, max_hops=4, max_addresses=2)

    assert len(fake.fetched) == 2
    assert result.stop_reason == STOP_ADDRESS_BUDGET
    assert result.complete is False
    assert any("budget" in reason for reason in result.incompleteness_reasons)
    # The addresses it declined to fetch are named, not dropped.
    assert [a for h in result.hops for a in h.deferred]


@pytest.mark.asyncio
async def test_the_investigated_wallet_is_fetched_even_under_a_zero_budget():
    """A misconfigured budget must not turn a run into "no activity".

    Budgets bound counterparty expansion. The wallet under investigation is
    the subject of the report, and skipping it would produce an empty dataset
    whose failure message blamed the chain rather than the configuration.
    """
    fake = ChainFake({W: [_tx(1, W, B)], B: [_tx(1, W, B)]})
    result = await _run(fake, max_hops=4, max_addresses=0, max_addresses_per_hop=0)

    assert fake.fetched == [W]
    assert result.total_records == 1
    assert result.stop_reason == STOP_ADDRESS_BUDGET


@pytest.mark.asyncio
async def test_per_hop_cap_defers_within_a_hop_but_still_goes_deeper():
    """A busy hop must not starve the deeper hops.

    With a per-hop cap of 1, hop 1 fetches one of its two counterparties and
    defers the other -- and hop 2 is still attempted, which is the whole
    reason the per-hop cap is separate from the total address budget.
    """
    fake = ChainFake(
        {
            W: [_tx(1, W, B), _tx(2, W, C), _tx(3, W, C)],
            B: [_tx(1, W, B)],
            C: [_tx(2, W, C), _tx(3, W, C), _tx(4, C, D)],
            D: [_tx(4, C, D)],
        }
    )
    result = await _run(fake, max_hops=3, max_addresses_per_hop=1)

    # C is fetched before B: it has two value-bearing edges to W, B has one.
    assert fake.fetched == [W, C, D]
    assert _hop(result, 1).fetched == [C]
    assert _hop(result, 1).deferred == [B]
    # Hop 2 still happened...
    assert result.hops_expanded == 3
    # ...but the unread hop-1 branch holds the claimed radius at 1.
    assert result.observation_depth == 1
    assert result.complete is False


@pytest.mark.asyncio
async def test_frontier_order_is_deterministic():
    """Two runs must fetch the same addresses in the same order.

    Determinism is what makes a budget-limited investigation reproducible; a
    set-iteration order here would let two runs of the same case disagree
    about which counterparties were examined.
    """
    ledger = {
        W: [_tx(1, W, B), _tx(2, W, C), _tx(3, W, D)],
        B: [_tx(1, W, B)],
        C: [_tx(2, W, C)],
        D: [_tx(3, W, D)],
    }
    first = ChainFake(ledger)
    second = ChainFake(ledger)
    await _run(first, max_hops=2, max_addresses=3)
    await _run(second, max_hops=2, max_addresses=3)

    assert first.fetched == second.fetched


@pytest.mark.asyncio
async def test_expanded_addresses_get_the_lower_record_budget():
    """A counterparty is fetched to find routes, not investigated in full.

    The wallet keeps MAX_TRANSACTIONS_PER_INVESTIGATION; an expanded address
    gets the smaller expansion budget, and hitting it is reported instead of
    leaving a truncated history looking complete.
    """
    fake = ChainFake(
        {
            W: [_tx(1, W, B)],
            B: [_tx(1, W, B)] + [_tx(i, B, C) for i in range(2, 12)],
        }
    )
    result = await _run(fake, max_hops=3, max_records_expanded=3)

    b_record = next(a for a in result.addresses if a.address == B)
    assert b_record.result is not None
    assert b_record.result.native.budget_reached is True
    assert b_record.complete is False
    assert result.complete is False
    assert any("cut short" in reason for reason in result.incompleteness_reasons)
    # The wallet's own history was NOT held to the expansion budget.
    assert next(a for a in result.addresses if a.address == W).complete is True


# --------------------------------------------------------------------------
# 6. PROVIDER FAILURE -- degrade, report, never abort or lie
# --------------------------------------------------------------------------


class FailingPeerFake(ChainFake):
    """Serves everything except one address, which always rate-limits."""

    def __init__(self, ledger, broken: str):
        super().__init__(ledger)
        self.broken = broken.lower()

    async def get_normal_transactions(
        self, address, start_block=0, end_block=99_999_999, page=1, offset=1000
    ):
        if address.lower() == self.broken:
            raise RateLimitError("rate limit not cleared")
        return await super().get_normal_transactions(
            address, start_block, end_block, page, offset
        )


@pytest.mark.asyncio
async def test_unreadable_counterparty_is_reported_without_aborting_the_run():
    """One bad counterparty must not discard the hops that did succeed.

    It must also not be quietly skipped: a route through that address is
    absent from the dataset rather than ruled out, and the report has to say
    so or a negative finding would be unearned.
    """
    fake = FailingPeerFake(
        {
            W: [_tx(1, W, B), _tx(2, W, C)],
            B: [_tx(1, W, B), _tx(3, B, D)],
            C: [_tx(2, W, C)],
            D: [_tx(3, B, D)],
        },
        broken=C,
    )
    result = await _run(fake, max_hops=3)

    # The healthy branch was still walked, all the way to hop 2.
    assert B in fake.fetched
    assert D in fake.fetched
    assert C in _hop(result, 1).failed
    assert result.complete is False
    assert any(
        "could not be read" in reason for reason in result.incompleteness_reasons
    )
    # A failure at hop 1 holds the observed radius at 1 however deep it walked.
    assert result.observation_depth == 1


@pytest.mark.asyncio
async def test_unreadable_investigated_wallet_still_raises():
    """The one address whose absence leaves nothing to analyse."""
    fake = FailingPeerFake({W: [_tx(1, W, B)]}, broken=W)

    with pytest.raises(IngestionError):
        await _run(fake, max_hops=3)


@pytest.mark.asyncio
async def test_provider_error_on_a_counterparty_is_also_survivable():
    """Not just rate limits: any ProviderError on a peer is recorded."""

    class ApiErrorPeerFake(ChainFake):
        async def get_normal_transactions(
            self, address, start_block=0, end_block=99_999_999, page=1, offset=1000
        ):
            if address.lower() == B:
                raise ProviderAPIError("upstream 500")
            return await super().get_normal_transactions(
                address, start_block, end_block, page, offset
            )

    fake = ApiErrorPeerFake({W: [_tx(1, W, B)], B: [_tx(1, W, B)]})
    result = await _run(fake, max_hops=3)

    assert result.addresses_fetched == 1
    assert B in _hop(result, 1).failed
    assert result.complete is False


# --------------------------------------------------------------------------
# 7. WHAT IS NOT A FUND-FLOW HOP
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_value_only_counterparty_is_not_expanded_but_is_kept():
    """An approval moves no funds, so it cannot be a hop of a fund flow.

    The transfer itself stays in the dataset -- it is real evidence -- but
    recursion does not follow it, and the declined addresses are counted.
    """
    fake = ChainFake(
        {
            W: [_tx(1, W, B), _tx(2, W, ZERO_PEER, value="0")],
            B: [_tx(1, W, B)],
            ZERO_PEER: [_tx(9, ZERO_PEER, C)],
        }
    )
    result = await _run(fake, max_hops=3)

    assert ZERO_PEER not in fake.fetched
    assert B in fake.fetched
    # The zero-value transfer is still in the dataset.
    assert any(t.to_address == ZERO_PEER for t in result.transfers)
    assert _hop(result, 1).no_value == [ZERO_PEER]
    # A zero-value edge carries no funds, so declining it is not a data gap.
    assert result.complete is True
    assert any("zero-value" in note for note in result.notes)


@pytest.mark.asyncio
async def test_failed_transaction_does_not_open_a_hop():
    """A reverted send moves nothing, whatever its value field says."""
    fake = ChainFake(
        {
            W: [_tx(1, W, B, is_error="1")],
            B: [_tx(1, W, B, is_error="1"), _tx(2, B, C)],
        }
    )
    result = await _run(fake, max_hops=3)

    assert fake.fetched == [W]
    assert _hop(result, 1).no_value == [B]
    # The failed transaction is still recorded as attempted activity.
    assert any(t.to_address == B for t in result.transfers)


@pytest.mark.asyncio
async def test_a_counterparty_with_both_kinds_of_edge_is_expanded():
    """One real transfer is enough; a zero-value edge does not veto it."""
    fake = ChainFake(
        {
            W: [_tx(1, W, B, value="0"), _tx(2, W, B)],
            B: [_tx(1, W, B, value="0"), _tx(2, W, B)],
        }
    )
    result = await _run(fake, max_hops=2)

    assert B in fake.fetched
    assert _hop(result, 1).fetched == [B]
    assert _hop(result, 1).no_value == []


# --------------------------------------------------------------------------
# 8. CACHE PASS-THROUGH
# --------------------------------------------------------------------------


class CacheAwareFake(ChainFake):
    """Records the `use_cache` flag it was called with, and exposes stats."""

    class _Cache:
        def stats(self):
            return {"hits": 7, "misses": 3, "writes": 3}

    def __init__(self, ledger):
        super().__init__(ledger)
        self.cache = self._Cache()
        self.use_cache_flags: list[bool] = []

    async def get_normal_transactions(
        self,
        address,
        start_block=0,
        end_block=99_999_999,
        page=1,
        offset=1000,
        use_cache=True,
    ):
        self.use_cache_flags.append(use_cache)
        return await super().get_normal_transactions(
            address, start_block, end_block, page, offset
        )

    async def get_token_transfers(
        self,
        address,
        start_block=0,
        end_block=99_999_999,
        page=1,
        offset=1000,
        use_cache=True,
    ):
        self.use_cache_flags.append(use_cache)
        return await super().get_token_transfers(
            address, start_block, end_block, page, offset
        )


@pytest.mark.asyncio
async def test_use_cache_flag_reaches_every_expanded_address():
    """Cache semantics must not silently change at hop 2.

    `--no-cache` has to mean "no cache" for the counterparties too, or a run
    would report a cache mode that only applied to the first address.
    """
    ledger = {W: [_tx(1, W, B)], B: [_tx(1, W, B), _tx(2, B, C)], C: [_tx(2, B, C)]}

    off = CacheAwareFake(ledger)
    await _run(off, max_hops=3, use_cache=False)
    assert off.fetched == [W, B, C]
    assert all(flag is False for flag in off.use_cache_flags)

    on = CacheAwareFake(ledger)
    result = await _run(on, max_hops=3, use_cache=True)
    assert all(flag is True for flag in on.use_cache_flags)
    # Cache accounting is surfaced so a REAL run can still disclose how many
    # of its requests never went to the network.
    assert result.cache_stats == {"hits": 7, "misses": 3, "writes": 3}


@pytest.mark.asyncio
async def test_a_cached_address_is_still_not_fetched_twice():
    """Deduplication is by address, so the cache never has to rescue it."""
    fake = CacheAwareFake(
        {
            W: [_tx(1, W, B), _tx(2, W, C)],
            B: [_tx(1, W, B), _tx(3, B, C)],
            C: [_tx(2, W, C), _tx(3, B, C)],
        }
    )
    await _run(fake, max_hops=4)

    # No (address, stream, page) triple is ever requested twice, so no
    # request depends on a cache hit to avoid being a duplicate.
    assert len(fake.calls) == len(set(fake.calls))


# --------------------------------------------------------------------------
# 9. HONEST ACCOUNTING
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_discovered_address_lands_in_exactly_one_bucket():
    """The hop buckets partition the addresses acquisition became aware of.

    If they did not, a report could show a total that no listed category
    accounted for -- which is how an address goes missing unnoticed. This
    fixture puts something in four different buckets at once.
    """
    fake = ChainFake(
        {
            W: [_tx(1, W, B), _tx(2, W, ZERO_PEER, value="0"), _tx(3, W, VASP)],
            B: [_tx(1, W, B), _tx(4, B, C)],
            C: [_tx(4, B, C)],
        }
    )
    result = await _run(fake, max_hops=3, terminal_addresses=[VASP])

    bucketed: list[str] = []
    for report in result.hops:
        bucketed.extend(report.fetched)
        bucketed.extend(report.deferred)
        bucketed.extend(report.terminal)
        bucketed.extend(report.no_value)
        bucketed.extend(report.failed)
        bucketed.extend(report.beyond_depth)

    assert len(bucketed) == len(set(bucketed))  # no address in two buckets
    assert set(bucketed) == {W, B, C, VASP, ZERO_PEER}
    assert result.addresses_discovered == len(bucketed)
    assert result.addresses_fetched == len(fake.fetched)


@pytest.mark.asyncio
async def test_stream_lines_aggregate_instead_of_repeating_per_address():
    """One line per stream, not three lines per fetched address."""
    fake = ChainFake(
        {
            W: [_tx(1, W, B)],
            B: [_tx(1, W, B), _tx(2, B, C)],
            C: [_tx(2, B, C)],
        }
    )
    result = await _run(fake, max_hops=3)

    stream_lines = [
        line
        for line in result.stream_lines
        if line.startswith(("native", "token", "internal"))
    ]
    assert len(stream_lines) == 3
    assert any("4 records across 3 address(es)" in line for line in stream_lines)
    # Plus one line per hop level, so the shape of the walk stays visible.
    assert sum(1 for line in result.stream_lines if line.startswith("hop ")) == 3


@pytest.mark.asyncio
async def test_progress_callback_reports_each_address():
    """A sequential 40-address run must not sit silent."""
    messages: list[str] = []
    fake = ChainFake({W: [_tx(1, W, B)], B: [_tx(1, W, B)]})
    await _run(fake, max_hops=2, progress=messages.append)

    assert any(W in m for m in messages)
    assert any(B in m for m in messages)
    assert any("complete" in m for m in messages)


@pytest.mark.asyncio
async def test_no_transfer_is_invented_for_an_address_with_no_activity():
    """An empty provider response produces no edges, not a placeholder."""
    fake = ChainFake({W: [_tx(1, W, B)], B: []})
    result = await _run(fake, max_hops=3)

    assert fake.fetched == [W, B]
    # Only the one real transaction, seen once.
    assert len(result.transfers) == 1
    assert result.transfers[0].tx_hash == f"0x{1:064x}"


# --------------------------------------------------------------------------
# (10) THE HORIZON THE LIVE PATH RECORDS
# --------------------------------------------------------------------------
#
# `observation_depth` is the number that stops a shallow search being
# reported as a complete negative, so where it comes from matters as much as
# what it says. The live path MEASURES it and stamps it on the graph, and a
# later `--cached-graph` run must read that back instead of re-deriving it
# from the graph's shape: shape-based inference assumes every unexpanded
# address is in the outermost ring, which is precisely what a budget-stopped
# recursive run violates.


def _budget_stopped_ledger() -> dict[str, list[dict[str, Any]]]:
    """W -> {B, E} at hop 1, {C, D} at hop 2, VASP at hop 3.

    Under a four-address budget hops 0 and 1 are expanded in full, C is
    fetched and D is deferred -- so the measured radius is 2 while the
    graph's greatest undirected distance is 3.
    """
    return {
        W: [_tx(1, W, B), _tx(2, W, E)],
        B: [_tx(1, W, B), _tx(3, B, C)],
        E: [_tx(2, W, E), _tx(4, E, D)],
        C: [_tx(3, B, C), _tx(5, C, VASP)],
        D: [_tx(4, E, D)],
    }


@pytest.mark.asyncio
async def test_a_budget_stop_holds_the_measured_depth_below_the_walk():
    """Hop 2 was reached and partly read, which is not the same as observed."""
    fake = ChainFake(_budget_stopped_ledger())
    result = await _run(fake, max_hops=4, max_addresses=4)

    assert fake.fetched == [W, B, E, C]
    assert _hop(result, 2).fetched == [C]
    assert _hop(result, 2).deferred == [D]
    assert result.hops_expanded == 3
    assert result.max_hop_reached == 2
    # Hops 0 and 1 were read in full; hop 2 was not, and one gap there caps
    # the radius however far the walk got.
    assert result.observation_depth == 2
    assert result.stop_reason == STOP_ADDRESS_BUDGET


async def _acquire_live_over_the_fake(monkeypatch, save_to):
    """The real `acquire_live`, wired to `ChainFake` instead of Etherscan.

    Nothing here touches the network: the provider class the pipeline imports
    is replaced, so the key below is never sent anywhere and is only present
    because acquire_live refuses to run without one.
    """
    from app.core.config import get_settings
    from app.investigation import pipeline as pl

    fake = ChainFake(_budget_stopped_ledger())
    monkeypatch.setenv("ETHERSCAN_API_KEY", "unit-test-key-never-sent")
    monkeypatch.setenv("EXPANSION_MAX_ADDRESSES", "4")
    monkeypatch.setattr(
        "app.blockchain.etherscan.EtherscanProvider",
        lambda settings, chain_name=None, refresh=False: fake,
    )

    graph, transfers, _normalization, _summary, provenance = await pl.acquire_live(
        W, CHAIN, get_settings(), use_cache=False, save_to=save_to, max_hops=4
    )
    return fake, graph, transfers, provenance


@pytest.mark.asyncio
async def test_acquire_live_stamps_the_measured_horizon_onto_the_graph(
    monkeypatch, tmp_path
):
    from app.graph.builder import load_graph
    from app.investigation import pipeline as pl

    saved = tmp_path / "live.gpickle"
    fake, graph, _transfers, provenance = await _acquire_live_over_the_fake(
        monkeypatch, saved
    )

    # The budget really bit: EXPANSION_MAX_ADDRESSES came from the environment.
    assert fake.fetched == [W, B, E, C]
    assert provenance.observation_depth == 2
    assert provenance.expansion_stop_reason == STOP_ADDRESS_BUDGET

    assert graph.graph["observation_depth"] == 2
    assert graph.graph["observation_wallet"] == W

    reloaded = load_graph(saved)
    assert pl.recorded_observation_depth(reloaded, W) == 2
    # The reason the stamp exists: shape-based inference would claim a hop
    # this dataset never covered, because D at hop 2 was never expanded while
    # C's branch reached hop 3.
    assert pl.infer_observation_depth(reloaded, W) == 3


@pytest.mark.asyncio
async def test_a_cached_graph_is_analysed_at_the_depth_it_was_acquired_with(
    monkeypatch, tmp_path
):
    """`--cached-graph` must not inherit a deeper horizon than the fetch had."""
    from app.investigation import pipeline as pl

    saved = tmp_path / "live.gpickle"
    await _acquire_live_over_the_fake(monkeypatch, saved)

    graph, provenance = pl.acquire_from_cached_graph(saved)
    assert provenance.observation_depth is None  # nothing declared yet

    report = pl.run_investigation(
        W, CHAIN, graph, provenance, max_hops=4, enable_ml=False
    )
    assert report.provenance.observation_depth == 2
    assert report.provenance.data_mode.value == "CACHED REAL DATA"


@pytest.mark.asyncio
async def test_a_recorded_horizon_is_not_applied_to_a_different_wallet(
    monkeypatch, tmp_path
):
    """A radius is measured around one address and means nothing around
    another, so it is ignored rather than reused."""
    from app.investigation import pipeline as pl

    saved = tmp_path / "live.gpickle"
    _fake, graph, _transfers, _provenance = await _acquire_live_over_the_fake(
        monkeypatch, saved
    )

    assert pl.recorded_observation_depth(graph, B) is None
    assert pl.recorded_observation_depth(graph, W.upper()) == 2
