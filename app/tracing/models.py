"""
Output models for Macro Milestone 3 / Phase A (multi-hop fund-flow tracing).

Kept separate from app/graph/models.py (structural Milestone-2 summaries)
since these are trace-request outputs, not graph-build bookkeeping.

IMPORTANT SEMANTIC NOTE (see README / task spec):
A graph path is evidence of a *chain of transaction relationships*, not
proof that the exact same funds moved continuously from hop to hop. These
models and everything downstream deliberately use "fund-flow candidate" /
"flow path" language rather than asserting fund continuity.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class FundFlowHop(BaseModel):
    """One traced edge (one blockchain transfer event) within a flow path."""

    hop_index: int  # 0-based position within the path

    from_address: str
    to_address: str

    tx_hash: str
    edge_key: str  # graph MultiDiGraph edge key ("<tx_hash>#<occurrence>")

    timestamp: Optional[int] = None  # unix epoch seconds; None if unavailable

    amount: float
    asset: str
    asset_type: str
    transfer_type: str

    chain: str

    # Copied from the graph edge, which copies it from NormalizedTransfer.
    # None only when the underlying edge genuinely carries no block number
    # (e.g. a hand-built test graph) — never fabricated.
    block_number: Optional[int] = None

    # Token contract address for token transfers; None for native transfers.
    token_contract: Optional[str] = None
    # Which provider stream evidenced this hop (native / internal / token).
    transfer_source: Optional[str] = None
    # Provider's own event index (ERC-20 log index or internal trace id).
    event_index: Optional[str] = None
    # Gas metadata, for distinguishing a plain payment from a contract call.
    gas_used: Optional[int] = None
    gas_fee_native: Optional[float] = None
    is_contract_interaction: Optional[bool] = None
    status: Optional[str] = None


def hop_from_edge(
    hop_index: int,
    from_address: str,
    to_address: str,
    edge_key: object,
    data: dict,
) -> "FundFlowHop":
    """Builds a FundFlowHop from one graph edge's attribute dict.

    Single place that maps edge attributes onto hop fields, so the
    exploratory tracer and the targeted tracer can never drift into
    reporting different evidence for the same edge.
    """
    return FundFlowHop(
        hop_index=hop_index,
        from_address=from_address,
        to_address=to_address,
        tx_hash=data.get("tx_hash", ""),
        edge_key=str(edge_key),
        timestamp=data.get("timestamp"),
        amount=data.get("amount", 0.0),
        asset=data.get("asset", "UNKNOWN"),
        asset_type=data.get("asset_type", "UNKNOWN"),
        transfer_type=data.get("transfer_type", "UNKNOWN"),
        chain=data.get("chain", ""),
        block_number=data.get("block_number"),
        token_contract=data.get("token_contract"),
        transfer_source=data.get("transfer_source"),
        event_index=data.get("event_index"),
        gas_used=data.get("gas_used"),
        gas_fee_native=data.get("gas_fee_native"),
        is_contract_interaction=data.get("is_contract_interaction"),
        status=data.get("status"),
    )


class FundFlowPath(BaseModel):
    """A single traced fund-flow candidate: source -> ... -> terminal node."""

    source: str
    terminal_node: str
    hops: list[FundFlowHop]

    @property
    def hop_count(self) -> int:
        return len(self.hops)

    @property
    def start_timestamp(self) -> Optional[int]:
        if not self.hops:
            return None
        return self.hops[0].timestamp

    @property
    def end_timestamp(self) -> Optional[int]:
        if not self.hops:
            return None
        return self.hops[-1].timestamp

    @property
    def path_duration_seconds(self) -> Optional[int]:
        """Tn - T1 across the path's hops. None (never fabricated) if either
        endpoint timestamp is missing."""
        start, end = self.start_timestamp, self.end_timestamp
        if start is None or end is None:
            return None
        return end - start

    @property
    def assets_involved(self) -> list[str]:
        seen: list[str] = []
        for hop in self.hops:
            if hop.asset not in seen:
                seen.append(hop.asset)
        return seen

    @property
    def addresses(self) -> list[str]:
        """Ordered list of every wallet visited, source first."""
        if not self.hops:
            return [self.source]
        addrs = [self.hops[0].from_address]
        addrs.extend(h.to_address for h in self.hops)
        return addrs


class TraceResult(BaseModel):
    """Full result of tracing fund-flow candidates from one source wallet."""

    source: str
    max_hops: int
    max_paths: int

    paths: list[FundFlowPath]

    edges_explored: int
    paths_truncated: bool  # True if max_paths was hit before search completed
    edges_limit_hit: bool  # True if max_edges_explored was hit

    notes: list[str] = []
