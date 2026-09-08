"""
Provider abstraction (Part 38 of the spec).

The intelligence engine must depend only on this interface, never on a
specific vendor's request/response shape. Swapping Etherscan for
Alchemy/Bitquery/etc later means writing one new class here — nothing
in normalization, graph, or attribution code should change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ProviderError(Exception):
    """Base class for provider-layer failures."""


class InvalidAddressError(ProviderError):
    pass


class RateLimitError(ProviderError):
    pass


class ProviderAPIError(ProviderError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ProviderUnavailableError(ProviderError):
    """The provider could not be used at all for this investigation — e.g.
    no credentials configured. Deliberately distinct from ProviderAPIError
    (a request that reached the provider and failed) so callers can refuse
    to start a production run rather than silently degrading to partial or
    demonstration data."""


class UnsupportedChainError(ProviderError):
    """The chain name is not one this build can retrieve data for.

    Raised before any request, because the alternative is worse than failing:
    a provider that accepts an unknown chain name still has to send *some*
    chain id, so it would return one chain's transactions labelled as
    another's. Every downstream section — attribution, the graph, the report
    header — would then state a chain the data did not come from. There is no
    partial-credit outcome here, so the name is rejected at the boundary.
    """


class BlockchainProvider(ABC):
    """Every method returns raw, provider-native dicts/lists.
    Normalization into NormalizedTransfer happens one layer up,
    in app/normalization/, never inside a provider implementation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def chain(self) -> str:
        ...

    @abstractmethod
    def validate_address(self, address: str) -> bool:
        ...

    @abstractmethod
    async def get_normal_transactions(
        self,
        address: str,
        start_block: int = 0,
        end_block: int = 99_999_999,
        page: int = 1,
        offset: int = 1000,
    ) -> list[dict[str, Any]]:
        """Native-asset (ETH) transactions sent to/from this address."""

    @abstractmethod
    async def get_token_transfers(
        self,
        address: str,
        start_block: int = 0,
        end_block: int = 99_999_999,
        page: int = 1,
        offset: int = 1000,
    ) -> list[dict[str, Any]]:
        """ERC-20 token transfer events involving this address."""

    # --- Optional capabilities -------------------------------------------
    # Deliberately NOT @abstractmethod: internal (contract-to-contract)
    # transfers are not offered by every provider, and making this abstract
    # would break every existing third-party/ test implementation of this
    # interface. A provider that cannot serve them advertises that through
    # supports_internal_transactions() rather than raising, so callers can
    # record the gap as a stated limitation instead of mistaking an empty
    # list for "this wallet had no internal transfers".

    def supports_internal_transactions(self) -> bool:
        return False

    async def get_internal_transactions(
        self,
        address: str,
        start_block: int = 0,
        end_block: int = 99_999_999,
        page: int = 1,
        offset: int = 1000,
    ) -> list[dict[str, Any]]:
        """Internal (contract-executed) value transfers involving this
        address. Returns [] on providers that do not support them — check
        supports_internal_transactions() to tell "unsupported" apart from
        "genuinely none"."""
        return []
