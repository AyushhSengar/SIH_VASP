"""
CHAIN REGISTRY — the one place that knows what a chain NAME means.

Why this exists
--------------------------------------------------------------------------
A chain name reaches the system as free text: `--chain ethereum` on the CLI,
`"chain": "..."` in an API request. Three different things then need to agree
about it — which chain id to query, what the native asset is called, and what
the report prints in its header. Before this module they disagreed:

  * the Etherscan provider took `chain_name` as a label and sent whatever
    `ETHERSCAN_CHAIN_ID` happened to be, so `--chain dogecoin` returned
    ETHEREUM transactions and every section of the report called them
    Dogecoin;
  * the normalizer hardcoded `"ETH" if chain == "ethereum" else None`, so any
    other chain's native transfers came out with no asset symbol at all;
  * the HTTP service kept its own `SUPPORTED_CHAINS` frozenset with a comment
    claiming the CLI enforced the same restriction, which it no longer did.

Mislabelled data is worse than absent data — it is evidence that says
something false — so the name is resolved once, here, and rejected at the
boundary if it cannot be.

Adding a chain
--------------------------------------------------------------------------
One entry below, and its provider must actually serve it. The entries here
are Etherscan V2 chain ids, which is what `chainid` takes on the unified
endpoint. Only `ethereum` has been exercised against live data in this
build; the others are listed because their ids and native symbols are
matters of public record, not because this project has validated them, and
`CHAIN_VALIDATED_LIVE` says which is which so the report can be honest about
it.
"""

from __future__ import annotations

from typing import NamedTuple

from app.blockchain.base import UnsupportedChainError


class ChainSpec(NamedTuple):
    """Everything the system needs to resolve a chain name."""

    name: str
    #: Etherscan V2 `chainid` query parameter.
    chain_id: int
    #: Symbol of the chain's native asset, used when a native transfer carries
    #: no token metadata of its own (it never does — the asset is implied).
    native_symbol: str
    #: Decimals of the native asset. 18 on every EVM chain listed here.
    native_decimals: int = 18


CHAINS: dict[str, ChainSpec] = {
    "ethereum": ChainSpec("ethereum", 1, "ETH"),
    "polygon": ChainSpec("polygon", 137, "POL"),
    "bsc": ChainSpec("bsc", 56, "BNB"),
    "base": ChainSpec("base", 8453, "ETH"),
    "arbitrum": ChainSpec("arbitrum", 42161, "ETH"),
    "optimism": ChainSpec("optimism", 10, "ETH"),
    "avalanche": ChainSpec("avalanche", 43114, "AVAX"),
}

#: Chains this build has actually been run against with live provider data.
#: Everything else in CHAINS is a correct-by-public-record mapping that has
#: not been exercised here, and saying so is the difference between
#: documenting a feature and claiming one.
CHAIN_VALIDATED_LIVE = frozenset({"ethereum"})

SUPPORTED_CHAINS = frozenset(CHAINS)


def normalize_chain_name(chain: str) -> str:
    """Case- and whitespace-insensitive, because a chain name is an operator's
    typing rather than an on-chain identifier. Nothing else is forgiven: no
    aliasing, no prefix matching."""
    return (chain or "").strip().lower()


def resolve_chain(chain: str) -> ChainSpec:
    """The only way to turn a chain name into a chain id.

    Raises rather than guessing. A guess here does not produce a weaker
    answer, it produces a confidently wrong one: the request still goes to
    *some* chain and its transactions still get labelled with the name that
    was asked for.
    """
    key = normalize_chain_name(chain)
    spec = CHAINS.get(key)
    if spec is None:
        raise UnsupportedChainError(
            f"Unsupported chain '{chain}'. This build can resolve: "
            f"{', '.join(sorted(CHAINS))}. A chain name is not accepted "
            "unless its chain id is known, because querying the wrong chain "
            "would return real transactions labelled with the wrong chain."
        )
    return spec


def native_symbol(chain: str) -> str | None:
    """Native asset symbol, or None for a chain this build cannot resolve.

    Returns None instead of raising because the normalizer's job is to reject
    the individual record and carry on, not to abort the run: a transfer whose
    chain does not match the investigated chain is already counted and
    reported as rejected.
    """
    key = normalize_chain_name(chain)
    spec = CHAINS.get(key)
    return spec.native_symbol if spec else None
