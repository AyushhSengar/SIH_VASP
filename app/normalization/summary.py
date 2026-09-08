"""Computes basic wallet-level summary stats from NormalizedTransfer records.

Milestone 1 scope only: counts, native inflow/outflow, counterparties,
first/last seen. Multi-hop, clustering, and behavioral detection are
later milestones (Part 4 of the spec) — deliberately not here yet.
"""

from __future__ import annotations

from app.models import AssetType, NormalizedTransfer, WalletSummary


def summarize_wallet(
    address: str, chain: str, transfers: list[NormalizedTransfer]
) -> WalletSummary:
    address = address.lower()

    inflow = 0.0
    outflow = 0.0
    senders: set[str] = set()
    receivers: set[str] = set()
    timestamps: list[int] = []

    for t in transfers:
        if t.asset_type != AssetType.NATIVE:
            continue  # Milestone 1: native-ETH flow totals only
        timestamps.append(t.timestamp)
        if t.to_address == address and t.from_address != address:
            inflow += t.amount
            senders.add(t.from_address)
        if t.from_address == address and t.to_address:
            outflow += t.amount
            receivers.add(t.to_address)

    note = None
    if not transfers:
        note = "No transactions returned by provider for this address/window."

    return WalletSummary(
        address=address,
        chain=chain,
        transaction_count=len(transfers),
        total_inflow_native=round(inflow, 8),
        total_outflow_native=round(outflow, 8),
        unique_senders=len(senders),
        unique_receivers=len(receivers),
        first_seen=min(timestamps) if timestamps else None,
        last_seen=max(timestamps) if timestamps else None,
        data_completeness_note=note,
    )
