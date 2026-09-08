"""
Canonical internal data models.

Per Part 10 of the spec: every provider returns data in its own shape.
Everything past the provider layer — normalization, graph, ML, evidence —
must consume ONLY NormalizedTransfer. No provider-specific field name
should ever leak past app/normalization/.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class AssetType(str, Enum):
    NATIVE = "NATIVE"
    ERC20 = "ERC20"
    ERC721 = "ERC721"
    ERC1155 = "ERC1155"
    UNKNOWN = "UNKNOWN"


class TransferStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class TransferSource(str, Enum):
    """Which provider stream a transfer was read from.

    Kept on every record because the three streams have genuinely different
    evidential properties: `txlist` shows externally-originated value moves,
    `txlistinternal` shows value moved by contract execution (invisible in
    txlist), and `tokentx` shows ERC-20 log events (which carry no native
    value at all). Collapsing them would make it impossible to say why a
    transfer is present or what its absence would mean.
    """

    NATIVE_TRANSACTION = "NATIVE_TRANSACTION"
    INTERNAL_TRANSACTION = "INTERNAL_TRANSACTION"
    TOKEN_TRANSFER = "TOKEN_TRANSFER"


class NormalizedTransfer(BaseModel):
    tx_hash: str
    chain: str
    block_number: int
    timestamp: int  # unix epoch seconds, UTC

    from_address: str
    to_address: Optional[str] = None  # null for contract-creation txs

    asset_type: AssetType
    asset_identifier: Optional[str] = None  # token contract address, if any
    asset_symbol: Optional[str] = None
    asset_decimals: Optional[int] = None

    amount_raw: str  # raw integer amount as string (avoid float precision loss)
    amount: float  # human-readable amount after decimals applied

    usd_value: Optional[float] = None  # not available from free Etherscan tier

    gas_used: Optional[int] = None
    gas_fee_native: Optional[float] = None
    gas_price_wei: Optional[str] = None  # string: wei exceeds float precision

    status: TransferStatus = TransferStatus.UNKNOWN
    is_contract_creation: bool = False
    method_id: Optional[str] = None

    # --- Provenance / disambiguation (added for production ingestion) ---
    # Defaulted so every pre-existing construction site keeps working.
    transfer_source: TransferSource = TransferSource.NATIVE_TRANSACTION
    # True when the transaction carried calldata beyond a plain value send,
    # i.e. the counterparty is being *called*, not merely paid. Derived from
    # the provider's own fields, never guessed from the amount.
    is_contract_interaction: bool = False
    # Distinguishes several transfers that share one tx_hash: the ERC-20
    # log index, or the internal-call trace id. None when the provider did
    # not supply one — never fabricated.
    event_index: Optional[str] = None
    # Set when the provider gave no token metadata, so downstream code can
    # say "symbol unknown" instead of silently showing a contract address
    # as if it were a ticker.
    token_metadata_missing: bool = False

    source_provider: str
    fetched_at: int  # unix epoch seconds when this record was pulled


class WalletSummary(BaseModel):
    address: str
    chain: str

    transaction_count: int
    total_inflow_native: float
    total_outflow_native: float

    unique_senders: int
    unique_receivers: int

    first_seen: Optional[int] = None
    last_seen: Optional[int] = None

    data_completeness_note: Optional[str] = None
