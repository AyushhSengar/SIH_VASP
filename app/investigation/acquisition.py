"""
RECURSIVE MULTI-HOP ACQUISITION — walking outward from the investigated
wallet, hop by hop, over real provider data.

WHY THIS MODULE EXISTS
--------------------------------------------------------------------------
`app.blockchain.ingest.acquire_wallet_transactions` fetches the three
transaction streams of ONE address, completely and with honest completeness
accounting. That is all it does, and all it should do.

A single-address fetch produces a depth-1 star: every edge touches the
investigated wallet, so the dataset can express "the wallet paid X" and can
never express "X then paid a known exchange". Every question of the form
"where did the money go after it left this wallet" was therefore
unanswerable from the data, no matter how deep the path search was
configured to look. Raising MAX_HOPS on that dataset does not find deeper
routes; it only searches a graph that cannot contain them.

This module closes that gap. It expands counterparties recursively:

    hop 0   the investigated wallet          -> fetch its streams
    hop 1   its counterparties              -> fetch each of their streams
    hop 2   their counterparties            -> fetch each of their streams
    ...     until the requested depth, the frontier, or a budget runs out

Everything it returns is real provider data. Nothing is synthesised, and an
address is never added to the graph unless a real transaction naming it was
retrieved from the provider.

WHAT BOUNDS IT
--------------------------------------------------------------------------
Ethereum is a single connected component; a naive recursive crawl is a crawl
of the whole chain. Four independent bounds apply, and whichever one bites is
NAMED IN THE RESULT rather than silently applied:

1. DEPTH        `max_hops` expansion rounds. `--max-hops 4` expands hops
                0..3, which is what makes nodes at hop 4 reachable.
2. ADDRESSES    a total ceiling on how many addresses are fetched at all.
3. PER-HOP      a ceiling per hop, so one high-degree counterparty at hop 1
                cannot consume the whole budget and starve hop 2.
4. RECORDS      each expanded address has its own per-stream record budget,
                lower than the investigated wallet's.

Two further rules cut work without hiding anything:

TERMINAL ADDRESSES. A discovered address that is itself in the known-VASP
dataset is the ENDPOINT of the question being asked, so it is recorded and
not expanded. Expanding an exchange hot wallet would cost the entire budget
reading transactions belonging to millions of unrelated customers, and would
not make the match to it any stronger. Not expanding it cannot hide a VASP:
it has already been found.

VALUE-BEARING EDGES ONLY. An address is expanded only if a retrieved
transfer of non-zero value connects it to the observed set. A zero-value
edge (an approval, a failed send, a bare contract call) moves no funds, so
it cannot be a hop of a fund flow. The zero-value transfer itself is still
kept, still normalized, and still on the graph — only the recursive
expansion through it is declined, and the count of addresses declined this
way is reported.

DEDUPLICATION AND CYCLES
--------------------------------------------------------------------------
An address is fetched at most once per investigation, at the shallowest hop
it was discovered at. That is simultaneously the request-deduplication rule
and the cycle-prevention rule: a cycle A -> B -> A cannot cause a re-fetch,
because A is already in `hop_of`.

WHAT "OBSERVED DEPTH" MEANS HERE
--------------------------------------------------------------------------
`observation_depth` is the hop radius around the wallet for which this
dataset's edges are COMPLETE. It counts leading hops that were expanded in
full: every address discovered at that hop was fetched, and no fetch was cut
short by an error or a record budget. One deferred or truncated address at
hop 1 stops the count at 1, because the hop-2 edges of that address were
never read and a route through it would be absent from the data rather than
ruled out.

That is deliberately the conservative reading. It makes attribution report
INCONCLUSIVE where a laxer definition would let it report NONE, and
"we did not look far enough" must never be printed as "there is nothing
there".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from app.blockchain.base import BlockchainProvider, InvalidAddressError, ProviderError
from app.blockchain.ingest import (
    AcquisitionResult,
    IngestionError,
    acquire_wallet_transactions,
)
from app.models import NormalizedTransfer, TransferStatus
from app.normalization.transactions import normalize_all, sort_transfers

# --- Why expansion stopped. Exactly one of these describes any run. -------
#: Every requested hop was expanded.
STOP_DEPTH_REACHED = "DEPTH_REACHED"
#: No unexpanded, value-bearing, non-terminal counterparty was left to fetch.
STOP_NO_NEW_COUNTERPARTIES = "NO_NEW_COUNTERPARTIES"
#: The total address ceiling was reached before the requested depth.
STOP_ADDRESS_BUDGET = "ADDRESS_BUDGET_REACHED"

_STOP_TEXT = {
    STOP_DEPTH_REACHED: "the requested hop depth was reached",
    STOP_NO_NEW_COUNTERPARTIES: (
        "no new value-bearing counterparty was left to expand"
    ),
    STOP_ADDRESS_BUDGET: (
        "the acquisition address budget was reached before the requested depth"
    ),
}


def stop_reason_text(reason: Optional[str]) -> str:
    """Human-readable form of a stop reason, for a report line."""
    if reason is None:
        return "not recorded"
    return _STOP_TEXT.get(reason, reason)


ProgressFn = Callable[[str], None]


@dataclass
class AddressAcquisition:
    """One address's fetch, and the hop it was fetched at."""

    address: str
    hop: int
    result: Optional[AcquisitionResult] = None
    #: Set when the provider could not serve this address at all. The
    #: investigation continues -- one unreachable counterparty must not
    #: discard the hops that were successfully read -- but the gap is
    #: reported, never absorbed.
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.result is not None

    @property
    def complete(self) -> bool:
        return self.result is not None and self.result.complete


@dataclass
class HopReport:
    """What happened at one hop level. Every discovered address ends up in
    exactly one of the lists below, so the totals always reconcile."""

    hop: int
    fetched: list[str] = field(default_factory=list)
    #: Discovered here, expandable, but not fetched because a budget bit.
    deferred: list[str] = field(default_factory=list)
    #: Discovered here and in the known-VASP dataset: an endpoint, not a
    #: route onward.
    terminal: list[str] = field(default_factory=list)
    #: Discovered here but connected only by zero-value transfers.
    no_value: list[str] = field(default_factory=list)
    #: Discovered here but the provider could not serve it: address -> reason.
    failed: dict[str, str] = field(default_factory=dict)
    #: Fetched here but at least one stream was cut short.
    truncated: list[str] = field(default_factory=list)
    #: Discovered here, where "here" IS the requested depth. Not expanded
    #: because expanding it would exceed what was asked for. Distinct from
    #: `deferred`, which is a budget cutting the request short: this is the
    #: request being honoured exactly, so it is not incompleteness.
    beyond_depth: list[str] = field(default_factory=list)

    @property
    def discovered(self) -> int:
        return (
            len(self.fetched)
            + len(self.deferred)
            + len(self.terminal)
            + len(self.no_value)
            + len(self.failed)
            + len(self.beyond_depth)
        )

    @property
    def all_discovered_fetched(self) -> bool:
        """Every address this hop discovered was actually fetched.

        This is what governs `observation_depth`: an address that was never
        fetched leaves an entire branch of the graph unread, so the radius
        claim cannot extend past it.

        Deliberately does NOT consider a fetched-but-truncated address.
        That address's edges ARE observed, just not all of them, and the gap
        is already carried by `MultiHopAcquisition.complete` and
        `incompleteness_reasons` -- which force attribution to INCONCLUSIVE on
        their own. Counting it here as well would report a smaller hop radius
        than acquisition actually walked, and understating the radius is not
        the same thing as being careful about it: it would send a reader
        looking for more data when the real problem is one address's history
        being longer than the record budget.

        `terminal` and `no_value` do not spoil it either: neither can carry a
        fund flow onward that this dataset would need to represent (a known
        VASP is the endpoint of the question; a zero-value edge moves
        nothing).
        """
        return not self.deferred and not self.failed


@dataclass
class MultiHopAcquisition:
    """Everything recursive acquisition produced, plus an explicit account of
    what it did not reach and why."""

    wallet: str
    chain: str
    provider: str
    requested_hops: int
    transfers: list[NormalizedTransfer]
    addresses: list[AddressAcquisition]
    hops: list[HopReport]
    stop_reason: str
    cache_stats: dict[str, int] = field(default_factory=dict)

    # ---- counts -------------------------------------------------------
    @property
    def addresses_fetched(self) -> int:
        return sum(1 for a in self.addresses if a.ok)

    @property
    def addresses_discovered(self) -> int:
        """Every distinct address acquisition became aware of, whether or
        not it was expanded."""
        return sum(h.discovered for h in self.hops)

    @property
    def provider_requests(self) -> int:
        """Pages actually requested. Cache hits are requests too -- they were
        issued by the fetcher, just served locally -- so this is the honest
        count of how much work the acquisition asked for."""
        return sum(
            o.pages_fetched
            for a in self.addresses
            if a.result is not None
            for o in a.result.outcomes
        )

    @property
    def total_records(self) -> int:
        return sum(a.result.total_records for a in self.addresses if a.ok)

    @property
    def hops_expanded(self) -> int:
        """How many hop levels had at least one address fetched."""
        return sum(1 for h in self.hops if h.fetched)

    @property
    def max_hop_reached(self) -> int:
        """Greatest hop distance at which any address was discovered.

        Normally one deeper than the deepest expanded hop, because expanding
        hop N is what makes hop N+1 addresses appear -- but only if that
        expansion actually returned a new counterparty, which is why this is
        measured rather than derived from the hop count.
        """
        return max((h.hop for h in self.hops if h.discovered), default=0)

    @property
    def observation_depth(self) -> int:
        """Hop radius for which this dataset's edges are complete.

        Counts LEADING hops in which every discovered address was fetched: a
        gap at hop 1 is not repaired by hop 2 happening to be clean, because
        routes through the hop-1 address that was missed are absent either
        way.
        """
        depth = 0
        for report in sorted(self.hops, key=lambda h: h.hop):
            if not report.fetched or not report.all_discovered_fetched:
                break
            depth += 1
        return depth

    # ---- honesty ------------------------------------------------------
    @property
    def complete(self) -> bool:
        """False if anything the requested depth called for was not read.

        Two independent causes, both of which must be reported: an address
        that was never fetched (deferred by a budget, or unreadable), and an
        address that was fetched but whose streams were cut short.

        Reaching the requested depth is NOT incompleteness -- the frontier
        beyond it was never in scope, which is why `beyond_depth` addresses do
        not count against this.
        """
        if any(h.deferred or h.failed for h in self.hops):
            return False
        return all(a.complete for a in self.addresses if a.ok)

    @property
    def incompleteness_reasons(self) -> list[str]:
        """Aggregated, not one line per address.

        Hop 0 -- the investigated wallet -- is reported verbatim, because its
        own history is the subject of the investigation and must never be
        summarised away. Expanded counterparties are aggregated by cause with
        counts and examples, so forty addresses cannot bury the wallet's own
        limitations under forty near-identical lines. The complete
        per-address record is in `--json`.
        """
        reasons: list[str] = []

        root = next((a for a in self.addresses if a.hop == 0 and a.ok), None)
        if root is not None and root.result is not None:
            reasons.extend(root.result.incompleteness_reasons)
        root_failure = next(
            (a for a in self.addresses if a.hop == 0 and not a.ok), None
        )
        if root_failure is not None:
            reasons.append(f"investigated wallet: {root_failure.error}")

        truncated = [
            a.address for a in self.addresses if a.hop > 0 and a.ok and not a.complete
        ]
        if truncated:
            reasons.append(
                f"{len(truncated)} expanded address(es) had at least one stream "
                "cut short by a record budget or a provider error, so their "
                "onward edges are only partially observed: "
                f"{_examples(truncated)}"
            )

        failed = [a.address for a in self.addresses if a.hop > 0 and not a.ok]
        if failed:
            reasons.append(
                f"{len(failed)} discovered address(es) could not be read from the "
                f"provider at all, so any route through them is absent from this "
                f"dataset rather than ruled out: {_examples(failed)}"
            )

        for report in sorted(self.hops, key=lambda h: h.hop):
            if report.deferred:
                reasons.append(
                    f"hop {report.hop}: {len(report.deferred)} discovered "
                    "address(es) were not expanded because an acquisition budget "
                    f"was reached, so their onward edges were never read: "
                    f"{_examples(report.deferred)}"
                )
        return reasons

    @property
    def notes(self) -> list[str]:
        """Statements about deliberate, bounded choices -- not incompleteness.

        Kept apart from `incompleteness_reasons` because these do not make the
        dataset incomplete for the question asked, and folding them in would
        make every run read as degraded.
        """
        notes: list[str] = [
            f"Recursive acquisition expanded {self.hops_expanded} hop level(s) "
            f"of a requested {self.requested_hops}, fetching "
            f"{self.addresses_fetched} address(es) in "
            f"{self.provider_requests} provider request(s). Stopped because "
            f"{stop_reason_text(self.stop_reason)}."
        ]
        terminal = sorted({a for h in self.hops for a in h.terminal})
        if terminal:
            notes.append(
                f"{len(terminal)} discovered address(es) are in the known-VASP "
                "dataset and were recorded as endpoints rather than expanded "
                f"further: {_examples(terminal)}"
            )
        no_value = sum(len(h.no_value) for h in self.hops)
        if no_value:
            notes.append(
                f"{no_value} discovered address(es) were connected to the "
                "observed set only by zero-value transfers (approvals, failed "
                "sends, bare contract calls). Those transfers are on the graph; "
                "the addresses were not expanded, because a zero-value edge "
                "moves no funds and cannot be a hop of a fund flow."
            )
        return notes

    @property
    def stream_lines(self) -> list[str]:
        """One line per stream, aggregated across every fetched address.

        The single-address version of this listed three lines. Listing three
        lines per address would bury the report in near-identical text, so
        the per-stream totals are summed and the addresses counted.
        """
        lines: list[str] = []
        for stream in (
            "native_transactions",
            "token_transfers",
            "internal_transactions",
        ):
            outcomes = [
                o
                for a in self.addresses
                if a.result is not None
                for o in a.result.outcomes
                if o.stream == stream
            ]
            supported = [o for o in outcomes if o.supported]
            if not supported:
                lines.append(f"{stream}: not supported by this provider")
                continue
            records = sum(o.count for o in supported)
            pages = sum(o.pages_fetched for o in supported)
            incomplete = sum(
                1 for o in supported if not o.complete or o.budget_reached
            )
            text = (
                f"{stream}: {records} records across {len(supported)} address(es), "
                f"{pages} page(s)"
            )
            if incomplete:
                text += f", {incomplete} address(es) INCOMPLETE"
            lines.append(text)
        for report in sorted(self.hops, key=lambda h: h.hop):
            lines.append(
                f"hop {report.hop}: {report.discovered} address(es) discovered, "
                f"{len(report.fetched)} fetched, {len(report.deferred)} deferred, "
                f"{len(report.terminal)} known-VASP endpoint(s), "
                f"{len(report.no_value)} zero-value-only, "
                f"{len(report.failed)} unreadable"
            )
        return lines


def _examples(addresses: list[str], limit: int = 3) -> str:
    """Names a few addresses and states how many were withheld."""
    shown = addresses[:limit]
    text = ", ".join(shown)
    if len(addresses) > limit:
        text += f" (+{len(addresses) - limit} more)"
    return text


def _counterparty_weights(
    transfers: Iterable[NormalizedTransfer], of_address: str
) -> tuple[dict[str, int], set[str]]:
    """Splits an address's counterparties into value-bearing and zero-value.

    Returns (weights, zero_value_only) where `weights` counts the SUCCESSFUL,
    non-zero transfers linking each counterparty to `of_address`, and
    `zero_value_only` holds the counterparties that never appeared on one.

    A failed transaction counts as zero-value however large its `value` field
    is: the transfer reverted, so no funds moved and it cannot be a hop of a
    fund flow. The record itself is still kept and still goes on the graph --
    an attempted transfer is real evidence of intent -- but recursive
    expansion does not follow it.

    Only transfers actually incident to `of_address` are considered. A
    contract-creation transfer has no recipient and so contributes no
    counterparty.
    """
    weights: dict[str, int] = {}
    seen_zero: set[str] = set()
    for transfer in transfers:
        if transfer.from_address == of_address:
            other = transfer.to_address
        elif transfer.to_address == of_address:
            other = transfer.from_address
        else:
            continue
        if not other or other == of_address:
            continue
        moved_value = (
            transfer.status == TransferStatus.SUCCESS
            and transfer.amount is not None
            and transfer.amount > 0
        )
        if moved_value:
            weights[other] = weights.get(other, 0) + 1
        else:
            seen_zero.add(other)
    return weights, {a for a in seen_zero if a not in weights}


async def acquire_multi_hop(
    provider: BlockchainProvider,
    wallet: str,
    chain: str,
    max_hops: int,
    max_records_wallet: int,
    max_records_expanded: int,
    max_addresses: int,
    max_addresses_per_hop: int,
    terminal_addresses: Iterable[str] = (),
    use_cache: bool = True,
    page_size: int = 1000,
    progress: Optional[ProgressFn] = None,
) -> MultiHopAcquisition:
    """Fetches the wallet, then recursively fetches its counterparties.

    `max_hops` is the number of expansion ROUNDS, which is also the deepest
    hop an address can be discovered at. `--max-hops 4` expands hops 0, 1, 2
    and 3, so an address can be discovered at hop 4 -- exactly the depth the
    path search is then configured to look for.

    Addresses are fetched one at a time, deliberately. Concurrent fetching
    would multiply the request rate against a provider whose free tier
    enforces a per-second limit, and the retry logic in the provider is built
    for a sequential caller.

    Raises IngestionError only when the INVESTIGATED WALLET itself cannot be
    read; that is the one address whose absence leaves nothing to analyse. A
    counterparty that cannot be read is recorded on its hop and the run
    continues.
    """
    wallet = wallet.lower()
    terminal = {a.lower() for a in terminal_addresses}
    rounds = max(1, int(max_hops))

    hop_of: dict[str, int] = {wallet: 0}
    addresses: list[AddressAcquisition] = []
    hops: list[HopReport] = []
    batches: list[list[NormalizedTransfer]] = []
    stop_reason = STOP_DEPTH_REACHED

    # (address, expandable) for the current hop. Hop 0 is the wallet, which is
    # always fetched regardless of the rules that govern counterparties.
    frontier: list[str] = [wallet]
    frontier_terminal: list[str] = []
    frontier_no_value: list[str] = []

    for hop in range(rounds):
        report = HopReport(
            hop=hop, terminal=frontier_terminal, no_value=frontier_no_value
        )

        if not frontier:
            # Nothing expandable left. Record the hop so its terminal /
            # zero-value discoveries are still visible, then stop.
            if report.discovered:
                hops.append(report)
            stop_reason = STOP_NO_NEW_COUNTERPARTIES
            break

        if hop > 0 and len(addresses) >= max_addresses:
            report.deferred = list(frontier)
            hops.append(report)
            stop_reason = STOP_ADDRESS_BUDGET
            break

        # This hop's normalized transfers, per address. Held here rather than
        # looked up out of the merged list afterwards, because discovery has
        # to ask "who did THIS address transact with" and the merged list can
        # no longer answer that without re-scanning every batch.
        hop_batches: list[tuple[str, list[NormalizedTransfer]]] = []
        budget_stop = False

        for index, address in enumerate(frontier):
            # Budgets bound COUNTERPARTY expansion. They never apply to hop 0:
            # the investigated wallet is the subject of the investigation, and
            # a misconfigured budget must not silently turn the run into one
            # that fetched nothing and then reports "no activity".
            if hop > 0 and len(addresses) >= max_addresses:
                report.deferred = list(frontier[index:])
                budget_stop = True
                break
            if hop > 0 and len(report.fetched) >= max_addresses_per_hop:
                # A per-hop ceiling defers the rest of THIS hop only; the run
                # continues so deeper hops are still attempted rather than
                # starved by one busy level.
                report.deferred = list(frontier[index:])
                break

            if progress is not None:
                progress(
                    f"hop {hop}: fetching {address} "
                    f"({index + 1}/{len(frontier)}, "
                    f"{len(addresses) + 1}/{max_addresses} addresses)"
                )

            record = AddressAcquisition(address=address, hop=hop)
            try:
                record.result = await acquire_wallet_transactions(
                    provider,
                    address,
                    max_records_per_stream=(
                        max_records_wallet if hop == 0 else max_records_expanded
                    ),
                    page_size=page_size,
                    use_cache=use_cache,
                )
            except (IngestionError, InvalidAddressError, ProviderError) as exc:
                if hop == 0:
                    # The investigated wallet. Nothing to analyse without it.
                    raise
                record.error = str(exc)
                report.failed[address] = str(exc)
                addresses.append(record)
                continue

            addresses.append(record)
            report.fetched.append(address)
            if not record.complete:
                report.truncated.append(address)

            batch = normalize_all(
                record.result.native.records,
                record.result.token.records,
                chain,
                record.result.provider,
                internal_raw=record.result.internal.records,
            )
            batches.append(batch)
            hop_batches.append((address, batch))

        hops.append(report)

        if budget_stop:
            stop_reason = STOP_ADDRESS_BUDGET
            break

        # --- discover the next hop from what this hop actually returned ---
        next_weights: dict[str, int] = {}
        next_zero: set[str] = set()
        for address, batch in hop_batches:
            weights, zero_only = _counterparty_weights(batch, address)
            for other, count in weights.items():
                next_weights[other] = next_weights.get(other, 0) + count
            next_zero |= zero_only

        frontier = []
        frontier_terminal = []
        frontier_no_value = []
        # Deterministic order: strongest observed relationship first, then
        # lexicographic. A budget that bites must cut the same addresses on
        # every run, or two runs of the same investigation would disagree.
        for other in sorted(next_weights, key=lambda a: (-next_weights[a], a)):
            if other in hop_of:
                continue  # already discovered -- dedup and cycle prevention
            hop_of[other] = hop + 1
            if other in terminal:
                frontier_terminal.append(other)
            else:
                frontier.append(other)
        for other in sorted(next_zero):
            if other in hop_of:
                continue
            hop_of[other] = hop + 1
            if other in terminal:
                frontier_terminal.append(other)
            else:
                frontier_no_value.append(other)
    else:
        # The loop ran every requested round. Anything discovered by the last
        # round sits AT the requested depth: it is a real node of the graph and
        # is recorded as such, but expanding it would go past what was asked
        # for, so it is not counted against completeness.
        if frontier or frontier_terminal or frontier_no_value:
            hops.append(
                HopReport(
                    hop=rounds,
                    terminal=frontier_terminal,
                    no_value=frontier_no_value,
                    beyond_depth=frontier,
                )
            )

    if progress is not None:
        progress(
            f"acquisition complete: {sum(1 for a in addresses if a.ok)} address(es) "
            f"fetched, stopped because {stop_reason_text(stop_reason)}"
        )

    cache_stats: dict[str, int] = {}
    cache = getattr(provider, "cache", None)
    if cache is not None and hasattr(cache, "stats"):
        cache_stats = cache.stats()

    merged: list[NormalizedTransfer] = []
    for batch in batches:
        merged.extend(batch)

    return MultiHopAcquisition(
        wallet=wallet,
        chain=chain,
        provider=provider.name,
        requested_hops=rounds,
        transfers=sort_transfers(merged),
        addresses=addresses,
        hops=hops,
        stop_reason=stop_reason,
        cache_stats=cache_stats,
    )


__all__ = [
    "AddressAcquisition",
    "HopReport",
    "MultiHopAcquisition",
    "acquire_multi_hop",
    "stop_reason_text",
    "STOP_ADDRESS_BUDGET",
    "STOP_DEPTH_REACHED",
    "STOP_NO_NEW_COUNTERPARTIES",
]
