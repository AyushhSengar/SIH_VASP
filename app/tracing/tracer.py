"""
MACRO MILESTONE 3 / PHASE A — multi-hop fund-flow tracing (Part 12/13 of
the spec, "original Milestone 3").

Traces transaction *relationships* reachable from a source wallet through
the existing Milestone-2 MultiDiGraph, up to a configurable number of hops.

--------------------------------------------------------------------------
SEMANTIC WARNING (do not remove): a graph path is NOT proof that the exact
same funds moved continuously hop to hop. A -> B (10 ETH) followed by
B -> C (9 ETH) is a *fund-flow candidate*, not confirmed fund continuity —
B could hold many other balances. Every path produced here is deliberately
called a "flow path" / "fund-flow candidate", never a confirmed transfer.
--------------------------------------------------------------------------

Algorithm: bounded depth-first search over the MultiDiGraph.
  - Every reachable *prefix* (1-hop, 2-hop, ... up to MAX_HOPS) is recorded
    as its own FundFlowPath, since the question this module answers is
    "what can be reached within N hops", not "give me only the longest
    paths".
  - Multi-edges are never collapsed: every (u, v, key) triple is a
    separate branch of the search, so parallel transfers between the same
    two wallets (including same-tx_hash duplicates from Milestone 2) all
    remain independently traceable.
  - Cycle handling: a node may not be revisited within a single path
    (simple-path constraint) — this alone bounds cycles independent of
    MAX_HOPS. The one deliberate exception is a genuine self-loop edge
    (u == v): it is recorded as a valid terminal hop (it's real evidence),
    but the search never continues past it, so it can't create an
    infinite loop.
  - Chronological ordering: where both the previous hop's timestamp and a
    candidate edge's timestamp are known, the candidate is only followed
    if its timestamp is >= the previous hop's timestamp (funds can't be
    forwarded before they arrive). When either timestamp is missing, the
    check is skipped rather than fabricating an ordering — this can let
    through paths that are not chronologically verified, which is exactly
    why path_duration_seconds is None whenever an endpoint timestamp is
    missing (see app/tracing/models.py).
  - Determinism: outgoing edges at each node are explored in a fixed sort
    order (timestamp, then edge key), so repeated calls against the same
    graph and limits produce identical results.
  - Safety limits: MAX_HOPS, MAX_PATHS, and MAX_EDGES_EXPLORED are all
    read from centralized configuration (app/core/config.py) and are hard
    stops, guarding against combinatorial explosion on a dense graph.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx

from app.core.config import Settings, get_settings
from app.tracing.models import FundFlowHop, FundFlowPath, TraceResult, hop_from_edge


def _sort_key(edge: tuple) -> tuple:
    _, _, key, data = edge
    ts = data.get("timestamp")
    return (ts if ts is not None else float("inf"), key)


def trace_fund_flow(
    graph: nx.MultiDiGraph,
    source: str,
    settings: Optional[Settings] = None,
    max_hops: Optional[int] = None,
    max_paths: Optional[int] = None,
    max_edges_explored: Optional[int] = None,
) -> TraceResult:
    """Traces fund-flow candidates outward from `source`, up to `max_hops`.

    All three limits fall back to centralized configuration
    (FUND_TRACE_MAX_HOPS / FUND_TRACE_MAX_PATHS /
    FUND_TRACE_MAX_EDGES_EXPLORED) when not explicitly overridden — never
    hardcoded here.
    """
    settings = settings or get_settings()
    resolved_max_hops = max_hops if max_hops is not None else settings.fund_trace_max_hops
    resolved_max_paths = max_paths if max_paths is not None else settings.fund_trace_max_paths
    resolved_max_edges = (
        max_edges_explored
        if max_edges_explored is not None
        else settings.fund_trace_max_edges_explored
    )

    notes: list[str] = []

    if graph.number_of_nodes() == 0:
        notes.append("Graph is empty — nothing to trace.")
        return TraceResult(
            source=source,
            max_hops=resolved_max_hops,
            max_paths=resolved_max_paths,
            paths=[],
            edges_explored=0,
            paths_truncated=False,
            edges_limit_hit=False,
            notes=notes,
        )

    if source not in graph:
        notes.append(f"Source wallet '{source}' was not found in the graph.")
        return TraceResult(
            source=source,
            max_hops=resolved_max_hops,
            max_paths=resolved_max_paths,
            paths=[],
            edges_explored=0,
            paths_truncated=False,
            edges_limit_hit=False,
            notes=notes,
        )

    state = {
        "paths": [],
        "edges_explored": 0,
        "paths_truncated": False,
        "edges_limit_hit": False,
    }

    def _dfs(node: str, hops: list[FundFlowHop], visited: set[str]) -> None:
        if len(state["paths"]) >= resolved_max_paths:
            state["paths_truncated"] = True
            return
        if state["edges_explored"] >= resolved_max_edges:
            state["edges_limit_hit"] = True
            return
        if len(hops) >= resolved_max_hops:
            return

        out_edges = sorted(graph.out_edges(node, keys=True, data=True), key=_sort_key)

        for u, v, key, data in out_edges:
            if len(state["paths"]) >= resolved_max_paths:
                state["paths_truncated"] = True
                return
            if state["edges_explored"] >= resolved_max_edges:
                state["edges_limit_hit"] = True
                return

            state["edges_explored"] += 1

            last_ts = hops[-1].timestamp if hops else None
            this_ts = data.get("timestamp")
            if last_ts is not None and this_ts is not None and this_ts < last_ts:
                # Would violate chronological fund-flow ordering — funds
                # can't be forwarded before they arrived. Skip this branch.
                continue

            is_self_loop = v == node
            if v in visited and not is_self_loop:
                # Revisiting a node (other than a same-step self-loop)
                # would create a cycle — bounded out regardless of MAX_HOPS.
                continue

            new_hop = hop_from_edge(
                hop_index=len(hops),
                from_address=u,
                to_address=v,
                edge_key=key,
                data=data,
            )
            new_hops = hops + [new_hop]
            state["paths"].append(
                FundFlowPath(source=source, terminal_node=v, hops=new_hops)
            )

            if len(state["paths"]) >= resolved_max_paths:
                state["paths_truncated"] = True
                return

            if is_self_loop:
                # Recorded as a valid terminal hop, but never traversed
                # further (would recurse forever otherwise).
                continue

            _dfs(v, new_hops, visited | {v})

    _dfs(source, [], {source})

    if state["paths_truncated"]:
        notes.append(
            f"Result limit reached: stopped after {resolved_max_paths} paths "
            "(FUND_TRACE_MAX_PATHS). More paths may exist."
        )
    if state["edges_limit_hit"]:
        notes.append(
            f"Exploration limit reached: stopped after exploring "
            f"{resolved_max_edges} edges (FUND_TRACE_MAX_EDGES_EXPLORED). "
            "Traversal may be incomplete."
        )
    if not state["paths"]:
        notes.append(f"No outgoing fund-flow candidates found from '{source}'.")

    return TraceResult(
        source=source,
        max_hops=resolved_max_hops,
        max_paths=resolved_max_paths,
        paths=state["paths"],
        edges_explored=state["edges_explored"],
        paths_truncated=state["paths_truncated"],
        edges_limit_hit=state["edges_limit_hit"],
        notes=notes,
    )
