"""
Converts raw provider-native dicts into NormalizedTransfer.

This is the ONLY place that should know what an Etherscan `txlist`,
`txlistinternal` or `tokentx` record looks like. Everything downstream —
graph building, tracing, attribution, feature extraction, ML — must never
touch raw provider fields.

Validation policy (do not weaken)
---------------------------------
Normalizing a single record NEVER rejects it: a record with a malformed
address or a missing timestamp is still real evidence that the provider
returned something, and dropping it inside the per-record function would
hide that. Validation is therefore a SEPARATE, explicit pass
(`validate_transfers`) that returns kept records, rejected records, and the
reason for each rejection, so an investigation can state exactly what it
discarded and why instead of silently losing data.

Address handling is exact and case-insensitive only — every address is
lowercased once, here, and compared by equality from then on. No fuzzy,
prefix, substring, or "similar-looking" matching exists anywhere in this
codebase.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from app.blockchain.chains import native_symbol
from app.models import (
    AssetType,
    NormalizedTransfer,
    TransferSource,
    TransferStatus,
)

_ADDRESS_RE = re.compile(r"^0x[a-f0-9]{40}$")
_TX_HASH_RE = re.compile(r"^0x[a-f0-9]{64}$")

# Sanity bounds for a unix timestamp on an EVM chain. 1438269973 is the
# Ethereum genesis block's timestamp — nothing on Ethereum can predate it.
_MIN_PLAUSIBLE_TIMESTAMP = 1_438_269_973
# Guards against a provider returning a millisecond timestamp or garbage;
# ~year 2100.
_MAX_PLAUSIBLE_TIMESTAMP = 4_102_444_800


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _apply_decimals(raw_amount: str, decimals: int) -> float:
    try:
        return int(raw_amount) / (10**decimals)
    except (TypeError, ValueError):
        return 0.0


def _lower(value: Any) -> str:
    return str(value or "").lower()


def _is_contract_interaction(raw: dict[str, Any]) -> bool:
    """True when the transaction carried calldata beyond a bare value send.

    Derived only from what the provider actually returned: a non-trivial
    `input`, a non-trivial `methodId`, or an explicit `functionName`.
    """
    method_id = str(raw.get("methodId") or "")
    if method_id and method_id not in ("0x", "0x00000000"):
        return True
    if str(raw.get("functionName") or "").strip():
        return True
    data = str(raw.get("input") or "")
    return bool(data) and data not in ("0x", "")


def normalize_native_transaction(
    raw: dict[str, Any], chain: str, source_provider: str
) -> NormalizedTransfer:
    """Normalize one record from Etherscan's `txlist` (native ETH transfers)."""

    is_error = raw.get("isError", "0")
    status = TransferStatus.FAILED if is_error == "1" else TransferStatus.SUCCESS

    to_address = raw.get("to") or None
    is_contract_creation = to_address is None or to_address == ""

    amount_raw = raw.get("value", "0")
    amount = _apply_decimals(amount_raw, 18)

    gas_used = raw.get("gasUsed")
    gas_price = raw.get("gasPrice")
    gas_fee_native = None
    if gas_used is not None and gas_price is not None:
        try:
            gas_fee_native = (int(gas_used) * int(gas_price)) / (10**18)
        except (TypeError, ValueError):
            gas_fee_native = None

    return NormalizedTransfer(
        tx_hash=_lower(raw.get("hash", "")),
        chain=chain,
        block_number=_safe_int(raw.get("blockNumber")),
        timestamp=_safe_int(raw.get("timeStamp")),
        from_address=_lower(raw.get("from")),
        to_address=to_address.lower() if to_address else None,
        asset_type=AssetType.NATIVE,
        asset_identifier=None,
        asset_symbol=native_symbol(chain),
        asset_decimals=18,
        amount_raw=str(amount_raw),
        amount=amount,
        usd_value=None,  # not available from free Etherscan tier
        gas_used=_safe_int(gas_used) if gas_used is not None else None,
        gas_fee_native=gas_fee_native,
        gas_price_wei=str(gas_price) if gas_price is not None else None,
        status=status,
        is_contract_creation=is_contract_creation,
        method_id=raw.get("methodId") or None,
        transfer_source=TransferSource.NATIVE_TRANSACTION,
        is_contract_interaction=_is_contract_interaction(raw),
        event_index=None,
        source_provider=source_provider,
        fetched_at=int(time.time()),
    )


def normalize_internal_transaction(
    raw: dict[str, Any], chain: str, source_provider: str
) -> NormalizedTransfer:
    """Normalize one record from Etherscan's `txlistinternal`.

    Internal transfers are native-value movements performed by contract
    execution. They share a `hash` with their parent transaction, so
    `event_index` carries the provider's `traceId` to keep several internal
    transfers of one transaction independently identifiable.

    Note the shape differences from `txlist`: there is no `gasPrice` (the
    parent transaction paid the fee, and attributing it to one internal
    call would be fabrication), and failure is signalled by `isError`.
    """
    is_error = raw.get("isError", "0")
    status = TransferStatus.FAILED if is_error == "1" else TransferStatus.SUCCESS

    to_address = raw.get("to") or None
    call_type = str(raw.get("type") or "").lower()
    # An internal `create`/`create2` produced a contract rather than paying
    # an existing address.
    is_contract_creation = call_type.startswith("create") or not to_address

    amount_raw = raw.get("value", "0")

    return NormalizedTransfer(
        tx_hash=_lower(raw.get("hash", "")),
        chain=chain,
        block_number=_safe_int(raw.get("blockNumber")),
        timestamp=_safe_int(raw.get("timeStamp")),
        from_address=_lower(raw.get("from")),
        to_address=to_address.lower() if to_address else None,
        asset_type=AssetType.NATIVE,
        asset_identifier=None,
        asset_symbol=native_symbol(chain),
        asset_decimals=18,
        amount_raw=str(amount_raw),
        amount=_apply_decimals(amount_raw, 18),
        usd_value=None,
        gas_used=_safe_int(raw.get("gasUsed")) if raw.get("gasUsed") else None,
        gas_fee_native=None,  # paid by the parent transaction, not this call
        gas_price_wei=None,
        status=status,
        is_contract_creation=is_contract_creation,
        method_id=None,
        transfer_source=TransferSource.INTERNAL_TRANSACTION,
        # An internal call is by definition contract execution.
        is_contract_interaction=True,
        event_index=str(raw.get("traceId")) if raw.get("traceId") not in (None, "") else None,
        source_provider=source_provider,
        fetched_at=int(time.time()),
    )


def normalize_token_transfer(
    raw: dict[str, Any], chain: str, source_provider: str
) -> NormalizedTransfer:
    """Normalize one record from Etherscan's `tokentx` (ERC-20 transfers)."""

    raw_decimals = raw.get("tokenDecimal")
    decimals = _safe_int(raw_decimals, default=18)
    amount_raw = raw.get("value", "0")
    symbol = raw.get("tokenSymbol") or None
    # Missing decimals are the dangerous case: assuming 18 when the token
    # actually uses 6 misstates the amount by 10^12. Flagged, not hidden.
    metadata_missing = raw_decimals in (None, "") or not symbol

    return NormalizedTransfer(
        tx_hash=_lower(raw.get("hash", "")),
        chain=chain,
        block_number=_safe_int(raw.get("blockNumber")),
        timestamp=_safe_int(raw.get("timeStamp")),
        from_address=_lower(raw.get("from")),
        to_address=_lower(raw.get("to")) or None,
        asset_type=AssetType.ERC20,
        asset_identifier=_lower(raw.get("contractAddress")) or None,
        asset_symbol=symbol,
        asset_decimals=decimals,
        amount_raw=str(amount_raw),
        amount=_apply_decimals(amount_raw, decimals),
        usd_value=None,
        gas_used=_safe_int(raw.get("gasUsed")) if raw.get("gasUsed") else None,
        gas_fee_native=None,
        gas_price_wei=str(raw.get("gasPrice")) if raw.get("gasPrice") else None,
        status=TransferStatus.SUCCESS,  # tokentx only lists successful transfers
        is_contract_creation=False,
        method_id=None,
        transfer_source=TransferSource.TOKEN_TRANSFER,
        is_contract_interaction=True,  # an ERC-20 transfer is a contract call
        event_index=(
            str(raw.get("logIndex")) if raw.get("logIndex") not in (None, "") else None
        ),
        token_metadata_missing=metadata_missing,
        source_provider=source_provider,
        fetched_at=int(time.time()),
    )


def sort_transfers(transfers: list[NormalizedTransfer]) -> list[NormalizedTransfer]:
    """Canonical chronological ordering for a set of normalized transfers.

    The sort is on (timestamp, block_number, tx_hash, event_index,
    transfer_source) rather than timestamp alone: many transfers share a
    timestamp within a block, and a stable, fully-specified ordering is what
    makes graph construction and tracing deterministic run to run.

    Exposed separately from `normalize_all` because recursive acquisition
    normalizes one address at a time (it has to read the from/to fields to
    decide what to fetch next) and then merges the batches. Merging must
    produce exactly the ordering a single normalize_all call would have, and
    the only way to guarantee that permanently is for both to use this one
    function.
    """
    return sorted(
        transfers,
        key=lambda t: (
            t.timestamp,
            t.block_number,
            t.tx_hash,
            t.event_index or "",
            t.transfer_source.value,
        ),
    )


def normalize_all(
    native_raw: Iterable[dict[str, Any]],
    token_raw: Iterable[dict[str, Any]],
    chain: str,
    source_provider: str,
    internal_raw: Optional[Iterable[dict[str, Any]]] = None,
) -> list[NormalizedTransfer]:
    """Normalizes and merges every stream into one chronologically ordered
    list.

    `internal_raw` is optional and defaults to none so that every existing
    two-stream call site keeps working unchanged.
    """
    normalized = [
        normalize_native_transaction(r, chain, source_provider) for r in native_raw
    ]
    if internal_raw is not None:
        normalized += [
            normalize_internal_transaction(r, chain, source_provider)
            for r in internal_raw
        ]
    normalized += [
        normalize_token_transfer(r, chain, source_provider) for r in token_raw
    ]
    return sort_transfers(normalized)


# --- Explicit validation pass -------------------------------------------------


@dataclass
class RejectedTransfer:
    """One record that could not be trusted, and exactly why."""

    reason: str
    tx_hash: str
    transfer_source: str


@dataclass
class NormalizationReport:
    """Auditable account of what validation did. Every input record ends up
    in exactly one of `kept` / `rejected` / `deduplicated`, so the totals
    always reconcile and nothing can go missing unnoticed."""

    input_count: int = 0
    kept_count: int = 0
    rejected: list[RejectedTransfer] = field(default_factory=list)
    duplicates_removed: int = 0
    missing_timestamp_count: int = 0
    missing_token_metadata_count: int = 0
    unsupported_chain_count: int = 0

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def reconciled(self) -> bool:
        return (
            self.kept_count + self.rejected_count + self.duplicates_removed
            == self.input_count
        )

    @property
    def reason_counts(self) -> dict[str, int]:
        """Rejections grouped by reason, for the report's data-summary section.

        A property, like `rejected_count` and `reconciled` above: as a plain
        method it read as truthy even when nothing was rejected, so a caller
        guarding on `if report.reason_counts:` printed an empty "rejection
        reasons" heading, and one that forgot the parentheses got an
        AttributeError instead of a dict.
        """
        counts: dict[str, int] = {}
        for item in self.rejected:
            counts[item.reason] = counts.get(item.reason, 0) + 1
        return counts


def _transfer_identity(t: NormalizedTransfer) -> tuple:
    """Identity used for de-duplication.

    Includes `event_index` and `transfer_source`, so two genuinely distinct
    events that share a tx_hash are NEVER collapsed. Only records identical
    in every one of these fields — which is what an overlapping provider
    re-fetch produces — are treated as the same event.
    """
    return (
        t.tx_hash,
        t.transfer_source.value,
        t.event_index,
        t.from_address,
        t.to_address,
        t.asset_identifier,
        t.amount_raw,
        t.block_number,
    )


def validate_transfers(
    transfers: Iterable[NormalizedTransfer],
    chain: str,
) -> tuple[list[NormalizedTransfer], NormalizationReport]:
    """Validates addresses, hashes, timestamps, block numbers, amounts,
    chain, and token metadata; de-duplicates exact repeats.

    A record is rejected only when using it would be actively misleading:
    an unusable address (nothing to put on a graph edge), an implausible
    timestamp (would corrupt chronological ordering), a negative amount, or
    a record from a different chain. Missing-but-recoverable problems
    (absent token symbol, absent timestamp on an otherwise valid record)
    are COUNTED and kept, never dropped.
    """
    report = NormalizationReport()
    kept: list[NormalizedTransfer] = []
    seen: set[tuple] = set()

    for t in transfers:
        report.input_count += 1

        if t.chain != chain:
            report.unsupported_chain_count += 1
            report.rejected.append(
                RejectedTransfer(
                    reason=f"record belongs to chain '{t.chain}', not '{chain}'",
                    tx_hash=t.tx_hash,
                    transfer_source=t.transfer_source.value,
                )
            )
            continue

        if not _TX_HASH_RE.match(t.tx_hash):
            report.rejected.append(
                RejectedTransfer(
                    reason="transaction hash is not a 32-byte 0x-prefixed hex value",
                    tx_hash=t.tx_hash,
                    transfer_source=t.transfer_source.value,
                )
            )
            continue

        if not _ADDRESS_RE.match(t.from_address):
            report.rejected.append(
                RejectedTransfer(
                    reason="sender address is not a valid 20-byte hex address",
                    tx_hash=t.tx_hash,
                    transfer_source=t.transfer_source.value,
                )
            )
            continue

        if t.to_address is not None and not _ADDRESS_RE.match(t.to_address):
            report.rejected.append(
                RejectedTransfer(
                    reason="recipient address is not a valid 20-byte hex address",
                    tx_hash=t.tx_hash,
                    transfer_source=t.transfer_source.value,
                )
            )
            continue

        if t.to_address is None and not t.is_contract_creation:
            report.rejected.append(
                RejectedTransfer(
                    reason="no recipient and not flagged as contract creation",
                    tx_hash=t.tx_hash,
                    transfer_source=t.transfer_source.value,
                )
            )
            continue

        if t.block_number < 0:
            report.rejected.append(
                RejectedTransfer(
                    reason="negative block number",
                    tx_hash=t.tx_hash,
                    transfer_source=t.transfer_source.value,
                )
            )
            continue

        if t.amount < 0:
            report.rejected.append(
                RejectedTransfer(
                    reason="negative transfer amount",
                    tx_hash=t.tx_hash,
                    transfer_source=t.transfer_source.value,
                )
            )
            continue

        if t.timestamp == 0:
            # Kept: the transfer itself is real. Counted so the report can
            # say chronological checks could not be applied to it.
            report.missing_timestamp_count += 1
        elif not (
            _MIN_PLAUSIBLE_TIMESTAMP <= t.timestamp <= _MAX_PLAUSIBLE_TIMESTAMP
        ):
            report.rejected.append(
                RejectedTransfer(
                    reason=(
                        "timestamp outside the plausible range for this chain "
                        "(would corrupt chronological ordering)"
                    ),
                    tx_hash=t.tx_hash,
                    transfer_source=t.transfer_source.value,
                )
            )
            continue

        if t.token_metadata_missing:
            report.missing_token_metadata_count += 1

        identity = _transfer_identity(t)
        if identity in seen:
            report.duplicates_removed += 1
            continue
        seen.add(identity)

        kept.append(t)

    report.kept_count = len(kept)
    return kept, report
