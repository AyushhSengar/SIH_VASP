"""
MILESTONE 2 — Graph construction (Part 11 of the spec).

Wallets/entities become nodes. Transfers become directed edges. This
module does structural graph-building ONLY:
  - no hop tracing / path discovery      (Milestone 3)
  - no fund-splitting/consolidation etc. (Milestone 4)
  - no clustering                        (Milestone 5)
  - no feature engineering / ML          (Milestones 6-7)

A MultiDiGraph is used deliberately: two wallets can transact many times,
and each individual transfer must remain its own traceable edge — collapsing
them into one edge would lose evidence (Part 11: "every important graph
edge should be traceable to blockchain evidence").

This module only ever consumes NormalizedTransfer (app/models.py). It does
not import from app/blockchain/ or app/normalization/transactions.py,
keeping the graph layer decoupled from provider/normalization concerns.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx

from app.models import AssetType, NormalizedTransfer
from app.graph.models import DegreeEntry, GraphBuildStats, GraphSummary

TRANSFER_TYPE_NATIVE = "TRANSFER"
TRANSFER_TYPE_TOKEN = "TOKEN_TRANSFER"
TRANSFER_TYPE_UNKNOWN = "UNKNOWN"


def _transfer_type_for(asset_type: AssetType) -> str:
    if asset_type == AssetType.NATIVE:
        return TRANSFER_TYPE_NATIVE
    if asset_type in (AssetType.ERC20, AssetType.ERC721, AssetType.ERC1155):
        return TRANSFER_TYPE_TOKEN
    return TRANSFER_TYPE_UNKNOWN


def build_graph(transfers: list[NormalizedTransfer]) -> tuple[nx.MultiDiGraph, GraphBuildStats]:
    """Builds a directed multigraph from normalized transfers.

    Every transfer becomes its own edge — never collapsed or overwritten,
    even when multiple transfers share the same tx_hash and (from, to) pair
    (e.g. a single transaction emitting several ERC-20 Transfer events
    between the same two addresses).

    Edge key design: a deterministic, input-order-based occurrence index is
    synthesized per tx_hash: "<tx_hash>#<occurrence>". This guarantees a
    unique (u, v, key) triple for every transfer regardless of how many
    transfers share a tx_hash, while remaining fully deterministic given the
    same input list. The original tx_hash is preserved as a plain edge
    attribute for provenance — it is never used as the sole edge key. The
    provider's own event index (ERC-20 log index / internal-call trace id),
    when available, is carried alongside as `event_index` so an edge can be
    tied back to a specific log entry, not just a transaction.

    Every field an investigator or a downstream evidence report needs is
    copied onto the edge — tx_hash, block_number, timestamp, from, to,
    amount, asset symbol, token contract, asset type, transfer type, chain,
    stream provenance, status, and gas metadata. Nothing downstream should
    ever have to go back to the transfer list to explain an edge.

    Returns (graph, stats), where `stats` is the authoritative accounting
    of what happened to every input transfer — never re-derived from
    free-text notes.
    """
    graph = nx.MultiDiGraph()
    notes: list[str] = []
    contract_creation_skipped = 0
    edges_created = 0
    tx_hash_occurrence_count: dict[str, int] = {}

    for t in transfers:
        # Node added regardless of whether an edge can be drawn, so the
        # wallet's presence in the dataset is never lost.
        graph.add_node(t.from_address)

        if t.to_address is None:
            contract_creation_skipped += 1
            continue  # no destination -> no edge can be evidenced (Part 11)

        graph.add_node(t.to_address)

        occurrence = tx_hash_occurrence_count.get(t.tx_hash, 0)
        tx_hash_occurrence_count[t.tx_hash] = occurrence + 1
        edge_key = f"{t.tx_hash}#{occurrence}"

        graph.add_edge(
            t.from_address,
            t.to_address,
            key=edge_key,
            amount=t.amount,
            amount_raw=t.amount_raw,
            asset=t.asset_symbol or t.asset_identifier or "UNKNOWN",
            asset_type=t.asset_type.value,
            asset_decimals=t.asset_decimals,
            token_contract=t.asset_identifier,
            token_metadata_missing=t.token_metadata_missing,
            timestamp=t.timestamp,
            block_number=t.block_number,
            tx_hash=t.tx_hash,  # provenance only — never the edge key
            transfer_index=occurrence,
            event_index=t.event_index,
            chain=t.chain,
            transfer_type=_transfer_type_for(t.asset_type),
            transfer_source=t.transfer_source.value,
            status=t.status.value,
            is_contract_interaction=t.is_contract_interaction,
            gas_used=t.gas_used,
            gas_fee_native=t.gas_fee_native,
            gas_price_wei=t.gas_price_wei,
            method_id=t.method_id,
        )
        edges_created += 1

    if contract_creation_skipped:
        notes.append(
            f"{contract_creation_skipped} contract-creation transfer(s) had no "
            "destination address and were excluded from edges (source node "
            "was still recorded)."
        )

    stats = GraphBuildStats(
        input_transfer_count=len(transfers),
        edges_created=edges_created,
        contract_creation_skipped=contract_creation_skipped,
        other_skipped=0,
        notes=notes,
    )

    if not stats.reconciled:
        # Should be unreachable given the logic above, but this is exactly
        # the class of bug that caused the original discrepancy — never
        # let it fail silently again.
        stats.notes.append(
            f"RECONCILIATION FAILURE: {stats.input_transfer_count} input transfers "
            f"but only {stats.accounted_for} accounted for "
            f"({stats.edges_created} edges + {stats.contract_creation_skipped} "
            f"contract-creation + {stats.other_skipped} other). "
            "This indicates a bug in build_graph() — investigate before trusting this graph."
        )

    return graph, stats


def summarize_graph(graph: nx.MultiDiGraph, stats: GraphBuildStats) -> GraphSummary:
    notes = list(stats.notes)

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()  # independent structural count

    native_edge_count = 0
    token_edge_count = 0
    self_loop_edges = 0
    timestamps: list[int] = []

    for u, v, data in graph.edges(data=True):
        if data.get("transfer_type") == TRANSFER_TYPE_NATIVE:
            native_edge_count += 1
        elif data.get("transfer_type") == TRANSFER_TYPE_TOKEN:
            token_edge_count += 1
        if u == v:
            self_loop_edges += 1
        ts = data.get("timestamp")
        if ts:
            timestamps.append(ts)

    # Cross-check: the graph's own edge count (structural) must match
    # build_graph()'s bookkeeping (accounting) independently of each other.
    # A mismatch here would mean a NEW bug distinct from the one just fixed.
    if edge_count != stats.edges_created:
        notes.append(
            f"CROSS-CHECK MISMATCH: graph.number_of_edges()={edge_count} but "
            f"build_graph() reported edges_created={stats.edges_created}. "
            "These are computed independently and must match."
        )

    if stats.reconciled:
        notes.append(
            f"Reconciled: {stats.input_transfer_count} input transfers = "
            f"{stats.edges_created} edges + {stats.contract_creation_skipped} "
            f"contract-creation skipped + {stats.other_skipped} other skipped."
        )

    if node_count == 0:
        notes.append("Graph has no nodes — no valid transfers were available to build from.")
        return GraphSummary(
            input_transfer_count=stats.input_transfer_count,
            edges_created=stats.edges_created,
            edge_count=edge_count,
            node_count=0,
            native_edge_count=0,
            token_edge_count=0,
            contract_creation_skipped=stats.contract_creation_skipped,
            other_skipped=stats.other_skipped,
            self_loop_edges=0,
            accounted_for=stats.accounted_for,
            reconciled=stats.reconciled,
            average_out_degree=0.0,
            average_in_degree=0.0,
            top_out_degree_nodes=[],
            top_in_degree_nodes=[],
            density=0.0,
            notes=notes,
        )

    out_degrees = dict(graph.out_degree())
    in_degrees = dict(graph.in_degree())

    avg_out = sum(out_degrees.values()) / node_count
    avg_in = sum(in_degrees.values()) / node_count

    top_out = sorted(out_degrees.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_in = sorted(in_degrees.items(), key=lambda kv: kv[1], reverse=True)[:5]

    if self_loop_edges:
        notes.append(
            f"{self_loop_edges} self-loop edge(s) found (wallet transacted with itself) — "
            "kept in the graph as-is, not filtered."
        )

    return GraphSummary(
        input_transfer_count=stats.input_transfer_count,
        edges_created=stats.edges_created,
        edge_count=edge_count,
        node_count=node_count,
        native_edge_count=native_edge_count,
        token_edge_count=token_edge_count,
        contract_creation_skipped=stats.contract_creation_skipped,
        other_skipped=stats.other_skipped,
        self_loop_edges=self_loop_edges,
        accounted_for=stats.accounted_for,
        reconciled=stats.reconciled,
        average_out_degree=round(avg_out, 4),
        average_in_degree=round(avg_in, 4),
        top_out_degree_nodes=[DegreeEntry(address=a, degree=d) for a, d in top_out],
        top_in_degree_nodes=[DegreeEntry(address=a, degree=d) for a, d in top_in],
        density=round(nx.density(graph), 8),
        earliest_timestamp=min(timestamps) if timestamps else None,
        latest_timestamp=max(timestamps) if timestamps else None,
        notes=notes,
    )


def load_transfers_from_fixture(path: Path) -> list[NormalizedTransfer]:
    """Loads a Milestone-1-produced fixture back into NormalizedTransfer objects.

    Intentionally reuses the exact same pydantic model as Milestone 1 rather
    than re-parsing raw dicts, so any schema drift is caught immediately by
    pydantic validation instead of silently producing a malformed graph.
    """
    raw = json.loads(Path(path).read_text())
    return [NormalizedTransfer.model_validate(item) for item in raw]


def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    """Pickles the graph for reuse. NetworkX graphs aren't a persistent
    database (Part 6) — this is a convenience cache for the prototype,
    not the eventual PostgreSQL-backed storage from later milestones.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def load_graph(path: Path) -> nx.MultiDiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)
