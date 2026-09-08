"""
TARGETED FUND-FLOW SEARCH — destination-aware traversal that scales.

Why this module exists
----------------------
`app.tracing.tracer.trace_fund_flow` answers "what is reachable from this
wallet?" by enumerating every path prefix. That is the right answer to that
question, and it is deliberately left unchanged — but it is the wrong tool
for attribution, where the question is much narrower: "does a directed,
chronologically-consistent path exist between this wallet and one of a
small set of known addresses?" Enumerating all prefixes to answer that
wastes essentially all of its work, and on a real wallet graph (order 10^4
nodes / 10^5 edges) it exhausts its budget long before it reaches anything
interesting — which then has to be reported as INCONCLUSIVE.

The fix is NOT a bigger budget. Raising MAX_PATHS / MAX_EDGES_EXPLORED does
not change the asymptotics; it just spends longer doing the same wasteful
enumeration. What this module changes is the search itself:

1. BIDIRECTIONAL LEVEL PRUNING. Two breadth-first sweeps — forward from the
   wallet, backward from the target set — give each node its exact distance
   from the source (`f`) and to the nearest target (`b`). A node can only
   lie on a path of length <= MAX_HOPS if `f[n] + b[n] <= MAX_HOPS`. Both
   sweeps are linear in the graph, and the surviving "viable subgraph" is
   typically orders of magnitude smaller than the whole graph.
2. EXACT NEGATIVES FOR FREE. Because BFS reachability is complete (never
   budgeted), a target absent from the viable set is a *definitive*
   negative within MAX_HOPS — no budget was involved, so it is reported as
   a complete negative rather than as INCONCLUSIVE. This is what makes
   "NONE" trustworthy. The one thing that qualifies it is the DATA
   HORIZON: exhaustive traversal of a graph that only ever acquired one
   hop of chain history proves nothing about hop two. See
   `observation_depth`, which downgrades such a negative to INCOMPLETE
   instead of letting the traversal's completeness stand in for the
   data's.
3. PER-STEP ADMISSIBILITY. Inside the enumeration, a step to node `n` after
   `h` hops is only taken if `h + b[n] <= MAX_HOPS`. Every step is
   therefore on a route that can still complete; the search does not walk
   into dead ends.
4. EARLY TERMINATION. Enumeration stops as soon as every reachable target
   has its configured quota of example paths. Attribution needs concrete
   evidence, not an exhaustive census of every way to get there.
5. REVERSE SEARCH, ONCE. Inbound (VASP -> wallet) is one traversal
   backwards from the wallet — not one forward traversal per known VASP
   address. Cost is independent of how large the VASP dataset grows.
6. TIME WINDOWING. An optional window drops edges outside the period of
   interest before any traversal happens, which is both an investigative
   scope control and real pruning.

--------------------------------------------------------------------------
SEMANTIC WARNING (do not remove): a path found here is a FUND-FLOW
CANDIDATE / TRACEABLE TRANSACTION PATH. A -> B -> C is NOT proof that the
same funds moved from A to C; B may hold unrelated balances. Chronological
consistency is enforced where timestamps exist, which makes a path
*possible*, never *proven*.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Callable, Iterable, Iterator, Optional

import networkx as nx
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.tracing.models import FundFlowHop, FundFlowPath, hop_from_edge


class SearchDirection(str, Enum):
    OUTBOUND = "OUTBOUND"  # wallet -> ... -> target
    INBOUND = "INBOUND"  # target -> ... -> wallet


class SearchStatus(str, Enum):
    """Completeness of the search, kept strictly separate from its result.

    COMPLETE means every route within MAX_HOPS was examined and the answer
    can be trusted in both directions (found or not found). INCOMPLETE
    means a budget stopped the enumeration, so a "not found" for the
    affected targets must be reported as INCONCLUSIVE, never as NONE.
    """

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


class TargetOutcome(BaseModel):
    """Per-target verdict. The distinction between these four cases is the
    whole point of the module: three of them are complete answers and only
    one is inconclusive."""

    target: str
    # Exact BFS distance ignoring chronology; None when unreachable.
    graph_distance: Optional[int] = None
    reachable_within_max_hops: bool = False
    paths_found: int = 0
    # True when the target is graph-reachable but every route was rejected
    # because timestamps ran backwards. A real, reportable finding: the
    # addresses ARE connected, but not in an order funds could have flowed.
    chronologically_blocked: bool = False
    # True when a budget stopped enumeration before this target's routes
    # were exhausted.
    search_incomplete: bool = False
    path_quota_reached: bool = False


class TargetedTraceResult(BaseModel):
    wallet: str
    direction: SearchDirection
    max_hops: int

    paths: list[FundFlowPath] = []
    target_outcomes: dict[str, TargetOutcome] = {}

    # Traversal accounting — proves the pruning actually happened and lets a
    # report state how much of the graph was even considered.
    graph_node_count: int = 0
    graph_edge_count: int = 0
    reachable_node_count: int = 0
    viable_node_count: int = 0
    edges_explored: int = 0

    paths_truncated: bool = False
    edges_limit_hit: bool = False
    status: SearchStatus = SearchStatus.COMPLETE

    # How many hops out from the wallet the graph's edge set is actually
    # complete (the DATA HORIZON), as opposed to how many hops the traversal
    # was willing to walk. None means no horizon was asserted by the caller.
    #
    # These two are independent and conflating them is a correctness bug: an
    # acquisition that stops at the investigated wallet fetches only that
    # wallet's own transactions, so every edge touches it and the graph holds
    # exactly ONE hop of chain history. Walking such a graph with MAX_HOPS=4
    # still cannot see a 2-hop route, because those edges were never acquired.
    # Reporting that as a complete negative would claim the chain was searched
    # to depth 4 when the data only ever described depth 1. A recursive
    # acquisition moves the horizon outward but does not remove it: it is
    # complete only to the hops it expanded in full.
    observation_depth: Optional[int] = None
    limited_by_observation_depth: bool = False

    time_window_start: Optional[int] = None
    time_window_end: Optional[int] = None
    edges_excluded_by_time_window: int = 0

    wallet_in_graph: bool = True
    wallet_is_target: bool = False

    notes: list[str] = []

    @property
    def targets_reached(self) -> list[str]:
        return sorted(t for t, o in self.target_outcomes.items() if o.paths_found)

    def best_path_for(self, target: str) -> Optional[FundFlowPath]:
        """Shortest path to/from one target; ties broken deterministically by
        the concatenated edge keys so repeated runs pick the same evidence."""
        candidates = [
            p
            for p in self.paths
            if (
                p.terminal_node == target
                if self.direction == SearchDirection.OUTBOUND
                else p.source == target
            )
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda p: (p.hop_count, "|".join(h.edge_key for h in p.hops)),
        )


# --- Edge iteration helpers ---------------------------------------------------

EdgeFilter = Callable[[dict], bool]


def _out_edges(graph: nx.MultiDiGraph, node: str) -> Iterator[tuple]:
    for _u, v, key, data in graph.out_edges(node, keys=True, data=True):
        yield v, key, data


def _in_edges(graph: nx.MultiDiGraph, node: str) -> Iterator[tuple]:
    for u, _v, key, data in graph.in_edges(node, keys=True, data=True):
        yield u, key, data


def _edge_sort_key(item: tuple) -> tuple:
    neighbour, key, data = item
    ts = data.get("timestamp")
    return (ts if ts is not None else float("inf"), str(key), str(neighbour))


def resolve_time_window(
    graph: nx.MultiDiGraph,
    window_days: int,
) -> tuple[Optional[int], Optional[int]]:
    """Derives an absolute [start, end] epoch window from a day count.

    Anchored on the LATEST timestamp present in the graph, not on wall-clock
    "now": anchoring on now would make the same cached dataset produce
    different results on different days, which would break reproducibility
    of an evidence report.
    """
    if not window_days or window_days <= 0:
        return None, None
    latest: Optional[int] = None
    for _u, _v, data in graph.edges(data=True):
        ts = data.get("timestamp")
        if ts:
            if latest is None or ts > latest:
                latest = ts
    if latest is None:
        return None, None
    return latest - int(window_days) * 86_400, latest


def _make_edge_filter(window_start: Optional[int]) -> Optional[EdgeFilter]:
    if window_start is None:
        return None

    def _within(data: dict) -> bool:
        ts = data.get("timestamp")
        # A transfer with no timestamp cannot be proven outside the window,
        # so it is kept rather than silently discarded as evidence.
        if not ts:
            return True
        return ts >= window_start

    return _within


# --- Breadth-first level maps -------------------------------------------------


def _bfs_levels(
    graph: nx.MultiDiGraph,
    roots: Iterable[str],
    max_hops: int,
    edge_iter: Callable[[nx.MultiDiGraph, str], Iterator[tuple]],
    edge_filter: Optional[EdgeFilter],
) -> dict[str, int]:
    """Minimum hop distance from any root, capped at max_hops.

    Complete and linear: this is what makes a "not reachable" answer a
    definitive negative rather than a budgeted guess.
    """
    levels: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()
    for root in roots:
        if root in graph and root not in levels:
            levels[root] = 0
            queue.append((root, 0))

    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbour, _key, data in edge_iter(graph, node):
            if edge_filter is not None and not edge_filter(data):
                continue
            if neighbour in levels:
                continue
            levels[neighbour] = depth + 1
            queue.append((neighbour, depth + 1))
    return levels


def trace_targeted(
    graph: nx.MultiDiGraph,
    wallet: str,
    targets: Iterable[str],
    direction: SearchDirection = SearchDirection.OUTBOUND,
    settings: Optional[Settings] = None,
    max_hops: Optional[int] = None,
    max_paths_per_target: Optional[int] = None,
    max_edges_explored: Optional[int] = None,
    time_window_days: Optional[int] = None,
    observation_depth: Optional[int] = None,
) -> TargetedTraceResult:
    """Finds directed, chronologically-consistent paths between `wallet` and
    any address in `targets`.

    OUTBOUND searches wallet -> ... -> target. INBOUND searches
    target -> ... -> wallet, as a single reverse traversal from the wallet
    (never one forward traversal per target).

    Addresses are matched by exact, case-insensitive equality after
    lowercasing. There is no fuzzy, prefix, or substring matching.

    `observation_depth` is the number of hops out from the wallet for which
    the graph's edges are actually complete — the data horizon. Pass it
    whenever it is known (the production pipeline always does). When
    `max_hops` exceeds it, a "not found" is reported as INCOMPLETE rather
    than as a complete negative, because the missing routes were never
    acquired and so were never searched. None means the caller asserts no
    horizon, which leaves the traversal's own completeness as the only
    limit.
    """
    settings = settings or get_settings()
    wallet = wallet.lower()
    target_set = {t.lower() for t in targets if t}

    hops_limit = max_hops if max_hops is not None else settings.fund_trace_max_hops
    quota = (
        max_paths_per_target
        if max_paths_per_target is not None
        else settings.targeted_trace_max_paths_per_target
    )
    edge_budget = (
        max_edges_explored
        if max_edges_explored is not None
        else settings.targeted_trace_max_edges_explored
    )
    window_days = (
        time_window_days
        if time_window_days is not None
        else settings.fund_trace_time_window_days
    )

    window_start, window_end = resolve_time_window(graph, window_days)
    edge_filter = _make_edge_filter(window_start)

    result = TargetedTraceResult(
        wallet=wallet,
        direction=direction,
        max_hops=hops_limit,
        graph_node_count=graph.number_of_nodes(),
        graph_edge_count=graph.number_of_edges(),
        time_window_start=window_start,
        time_window_end=window_end,
        wallet_in_graph=wallet in graph,
        wallet_is_target=wallet in target_set,
        observation_depth=observation_depth,
    )

    # A horizon shorter than the hop limit means the deeper hops are unsearched
    # for want of data, not searched-and-empty. Recorded once, here, so every
    # exit path below inherits it.
    horizon_limited = (
        observation_depth is not None and observation_depth < hops_limit
    )
    result.limited_by_observation_depth = horizon_limited
    if horizon_limited:
        # The search as SPECIFIED (to hops_limit) cannot complete on this
        # data, whatever else happens below, so the status is settled here and
        # every return path inherits it.
        result.status = SearchStatus.INCOMPLETE
        result.notes.append(
            f"DATA HORIZON: the graph contains complete edges only to "
            f"{observation_depth} hop(s) from the wallet, but MAX_HOPS is "
            f"{hops_limit}. Routes longer than {observation_depth} hop(s) "
            "cannot be observed in this dataset because the intervening "
            "transactions were never acquired -- they were not searched and "
            "found absent. Any negative below is therefore limited to "
            f"{observation_depth} hop(s)."
        )

    if edge_filter is not None:
        excluded = 0
        for _u, _v, data in graph.edges(data=True):
            if not edge_filter(data):
                excluded += 1
        result.edges_excluded_by_time_window = excluded
        result.notes.append(
            f"Time window applied: only transfers at or after epoch "
            f"{window_start} ({window_days} day(s) before the most recent "
            f"observed activity) were traversed; {excluded} edge(s) were "
            "outside it."
        )

    # Every target gets an outcome record, including unreachable ones — a
    # target that produced nothing must still appear, with the reason.
    for target in sorted(target_set):
        result.target_outcomes[target] = TargetOutcome(target=target)

    if graph.number_of_nodes() == 0:
        result.notes.append("Graph is empty — nothing to trace.")
        return result
    if wallet not in graph:
        result.notes.append(
            f"Wallet '{wallet}' does not appear in the graph, so no path in "
            "either direction can exist within it."
        )
        return result
    if not target_set:
        result.notes.append("No target addresses were supplied to search for.")
        return result
    if result.wallet_is_target:
        result.notes.append(
            f"The investigated wallet '{wallet}' is ITSELF one of the known "
            "addresses being searched for. That is an exact address identity, "
            "not a traced path — it is reported separately and never as a "
            "multi-hop fund-flow candidate."
        )

    # Direction determines which way each sweep runs. Forward = away from the
    # wallet along the direction funds would travel for this question.
    if direction == SearchDirection.OUTBOUND:
        forward_iter, backward_iter = _out_edges, _in_edges
    else:
        # Searching target -> ... -> wallet means walking backwards from the
        # wallet, so "forward" for the search is the graph's in-edges.
        forward_iter, backward_iter = _in_edges, _out_edges

    # --- Step 1/2: two complete, linear BFS sweeps. ---
    from_wallet = _bfs_levels(graph, [wallet], hops_limit, forward_iter, edge_filter)
    to_target = _bfs_levels(
        graph,
        [t for t in target_set if t in graph],
        hops_limit,
        backward_iter,
        edge_filter,
    )
    result.reachable_node_count = len(from_wallet)

    # --- Viable subgraph: nodes that can lie on a <= MAX_HOPS route. ---
    viable: dict[str, int] = {}
    for node, f in from_wallet.items():
        b = to_target.get(node)
        if b is not None and f + b <= hops_limit:
            viable[node] = b
    result.viable_node_count = len(viable)

    reachable_targets: set[str] = set()
    for target in sorted(target_set):
        outcome = result.target_outcomes[target]
        if target not in graph:
            outcome.reachable_within_max_hops = False
            continue
        distance = from_wallet.get(target)
        outcome.graph_distance = distance
        if distance is not None and 0 < distance <= hops_limit:
            outcome.reachable_within_max_hops = True
            reachable_targets.add(target)

    result.notes.append(
        f"Destination-aware pruning: {len(from_wallet)} node(s) reachable "
        f"from the wallet within {hops_limit} hop(s); {len(viable)} of those "
        "can lie on a route to a searched address and were the only nodes "
        f"enumerated (graph has {graph.number_of_nodes()} nodes / "
        f"{graph.number_of_edges()} edges)."
    )

    if not reachable_targets:
        if horizon_limited:
            # Exhaustive over the acquired data, but the acquired data stops
            # short of the requested depth. Not a complete negative.
            result.notes.append(
                "No searched address is reachable from the wallet within the "
                f"{observation_depth} hop(s) this dataset actually observes. "
                "Breadth-first reachability was exhaustive over the acquired "
                "edges and hit no budget, so this is a COMPLETE negative at "
                f"{observation_depth} hop(s) -- but it is INCONCLUSIVE for the "
                f"requested {hops_limit} hop(s), because hops "
                f"{observation_depth + 1}-{hops_limit} are absent from the "
                "data rather than searched and empty."
            )
        else:
            result.notes.append(
                "No searched address is reachable from the wallet within "
                f"{hops_limit} hop(s). This is a COMPLETE negative: breadth-first "
                "reachability is exhaustive and was not subject to any budget."
            )
        return result

    # --- Bounded enumeration over the viable subgraph only. ---
    state = {"edges_explored": 0, "budget_hit": False, "quota_hit": False}
    per_target_paths: dict[str, int] = {t: 0 for t in reachable_targets}
    remaining = set(reachable_targets)
    chronologically_reached: set[str] = set()

    def _record(target: str, walk: list[tuple]) -> None:
        """Turns an accepted walk into a FundFlowPath in true chronological
        (source -> terminal) orientation, regardless of search direction."""
        if direction == SearchDirection.OUTBOUND:
            ordered = walk
            path_source, terminal = wallet, target
        else:
            # The walk was built backwards from the wallet; reverse it so the
            # reported evidence reads target -> ... -> wallet, in the order
            # funds would actually have moved.
            ordered = list(reversed(walk))
            path_source, terminal = target, wallet

        hops: list[FundFlowHop] = []
        for index, (u, v, key, data) in enumerate(ordered):
            hops.append(
                hop_from_edge(
                    hop_index=index,
                    from_address=u,
                    to_address=v,
                    edge_key=key,
                    data=data,
                )
            )
        result.paths.append(
            FundFlowPath(source=path_source, terminal_node=terminal, hops=hops)
        )
        per_target_paths[target] += 1
        chronologically_reached.add(target)
        if per_target_paths[target] >= quota:
            remaining.discard(target)
            result.target_outcomes[target].path_quota_reached = True
            state["quota_hit"] = True

    def _dfs(node: str, depth: int, walk: list[tuple], visited: set[str]) -> None:
        if not remaining:
            return  # early termination: every target already has its quota
        if state["edges_explored"] >= edge_budget:
            state["budget_hit"] = True
            return
        if depth >= hops_limit:
            return

        candidates = sorted(forward_iter(graph, node), key=_edge_sort_key)

        for neighbour, key, data in candidates:
            if not remaining:
                return
            if state["edges_explored"] >= edge_budget:
                state["budget_hit"] = True
                return
            if edge_filter is not None and not edge_filter(data):
                continue

            # Admissibility: only step onto nodes that can still complete a
            # route inside the hop budget.
            b = viable.get(neighbour)
            if b is None or depth + 1 + b > hops_limit:
                continue

            state["edges_explored"] += 1

            this_ts = data.get("timestamp")
            last_ts = walk[-1][3].get("timestamp") if walk else None
            if last_ts and this_ts:
                # Chronological ordering. Outbound: each successive hop must
                # not precede the one before it. Inbound: we are walking
                # backwards, so each successive (earlier) hop must not
                # postdate the one it feeds.
                if direction == SearchDirection.OUTBOUND:
                    if this_ts < last_ts:
                        continue
                else:
                    if this_ts > last_ts:
                        continue

            is_self_loop = neighbour == node
            if neighbour in visited and not is_self_loop:
                continue

            if direction == SearchDirection.OUTBOUND:
                step = (node, neighbour, key, data)
            else:
                step = (neighbour, node, key, data)
            new_walk = walk + [step]

            if neighbour in remaining:
                _record(neighbour, new_walk)

            if is_self_loop:
                continue
            _dfs(neighbour, depth + 1, new_walk, visited | {neighbour})

    _dfs(wallet, 0, [], {wallet})

    result.edges_explored = state["edges_explored"]
    result.edges_limit_hit = bool(state["budget_hit"])

    for target in sorted(reachable_targets):
        outcome = result.target_outcomes[target]
        outcome.paths_found = per_target_paths.get(target, 0)
        if outcome.paths_found == 0:
            if state["budget_hit"]:
                outcome.search_incomplete = True
            else:
                # Enumeration finished without a budget stop, so the only
                # thing that can have blocked this reachable target is the
                # chronological constraint.
                outcome.chronologically_blocked = True

    unresolved_due_to_budget = [
        t
        for t in reachable_targets
        if result.target_outcomes[t].search_incomplete
    ]
    if state["budget_hit"]:
        result.paths_truncated = True
        result.status = SearchStatus.INCOMPLETE
        result.notes.append(
            f"Enumeration budget reached after exploring {result.edges_explored} "
            f"edge(s) of the viable subgraph (TARGETED_TRACE_MAX_EDGES_EXPLORED="
            f"{edge_budget}). Reachability above is still exact, but path "
            "enumeration is incomplete for "
            f"{len(unresolved_due_to_budget)} reachable target(s)."
        )

    blocked = sorted(
        t for t in reachable_targets
        if result.target_outcomes[t].chronologically_blocked
    )
    if blocked:
        result.notes.append(
            f"{len(blocked)} address(es) are connected to the wallet in the "
            "graph within the hop limit, but every route was rejected because "
            "its transfer timestamps run backwards — funds cannot be "
            "forwarded before they arrive. Reported as a connection without a "
            "chronologically-consistent fund-flow candidate, not as a match."
        )

    if not chronologically_reached:
        result.notes.append(
            "No chronologically-consistent fund-flow candidate was found to "
            "any searched address."
        )

    return result
