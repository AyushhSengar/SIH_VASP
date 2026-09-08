"""
REAL FEATURES for the production ML pipeline — label-blind by construction.

Every feature here is computed from the transfer graph while treating every
edge identically. Nothing is read from the provider stream a transfer came
from, and nothing is read from the known-VASP dataset. That is not a stylistic
choice: both are label sources (see `app/ml/real_labels.py`), so a feature
derived from either would let a model recover the answer instead of learning
from behaviour, and the held-out score would measure nothing.

FEATURE GROUPS AND WHY THEY ARE SEPARABLE
--------------------------------------------------------------------------
`structure`, `value` and `temporal` describe how an address transacts and are
admissible for every task. `asset_mix` describes *which* assets it moves,
which for the account-type task is a proxy for the provider stream that
defines the label (a token contract's activity is ERC-20 by definition), so
that task excludes the group outright. A task declares the groups it excludes
and the exclusion is recorded in the trained artifact, so a reviewer can see
which features the model was allowed to see rather than taking it on trust.

WHAT IS DELIBERATELY NOT COMPUTED (do not add without re-reading this)
--------------------------------------------------------------------------
  * `transfer_source` / `transfer_type` counts — these ARE the account-type
    label.
  * `is_contract_interaction` — an internal transfer and an ERC-20 transfer
    both set it unconditionally, so it reproduces the label almost exactly.
  * `gas_used`, `gas_fee_native`, `gas_price_wei` — only a top-level
    transaction carries its own gas fields, so a non-null gas value is a
    direct signal that the sender signed a transaction, i.e. is an EOA.
  * `method_id`, `is_contract_creation` — same leak, via calldata presence.
  * `token_contract` membership — being a token contract is a label rule.

Combined in/out degree IS kept. An address's degree is not part of any label
rule; contracts and EOAs both range from one transfer to tens of thousands,
and degree being predictive is the model finding real structure rather than
reading the answer.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import networkx as nx
from pydantic import BaseModel

# Bumped on any change to the feature list, its order, or a definition.
# Recorded in every trained artifact and every prediction so a model can
# never be fed a vector it was not trained on.
FEATURE_SCHEMA_VERSION = "real-features-v1"

SECONDS_PER_DAY = 86_400

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "structure": (
        "out_transfer_count",
        "in_transfer_count",
        "total_transfer_count",
        "unique_out_counterparties",
        "unique_in_counterparties",
        "unique_counterparties",
        "reciprocal_counterparty_count",
        "reciprocity_ratio",
        "out_share_of_transfers",
        "transfers_per_counterparty",
        "top_counterparty_share",
        "counterparty_concentration_hhi",
        "self_loop_count",
    ),
    "value": (
        "log_total_out_amount",
        "log_total_in_amount",
        "log_mean_out_amount",
        "log_mean_in_amount",
        "log_max_amount",
        "log_amount_stdev",
        "distinct_amount_ratio",
        "zero_amount_share",
    ),
    "temporal": (
        "lifespan_days",
        "active_day_count",
        "transfers_per_active_day",
        "log_median_gap_seconds",
        "log_min_gap_seconds",
        "max_transfers_in_one_hour",
        "night_hour_share",
        "hour_entropy",
        "weekday_entropy",
        "timestamped_share",
    ),
    "asset_mix": (
        "distinct_asset_count",
        "native_transfer_share",
    ),
}

#: Feature groups a task may not see, and the reason, recorded in artifacts.
TASK_EXCLUDED_GROUPS: dict[str, tuple[str, ...]] = {
    "account_type": ("asset_mix",),
    "vasp_ownership": (),
}

TASK_EXCLUSION_REASONS: dict[str, str] = {
    "account_type": (
        "asset_mix excluded: an address's native/token activity mix is a proxy "
        "for the provider stream (txlist vs tokentx) that defines the "
        "account-type label, so these features could leak the answer."
    ),
    "vasp_ownership": (
        "No group excluded: VASP labels come from dataset provenance, which no "
        "feature in this module reads."
    ),
}


def feature_names(task: Optional[str] = None) -> list[str]:
    """The exact ordered feature list for a task.

    Order is part of the schema: a vector is positional, so a reordering is a
    breaking change and must bump FEATURE_SCHEMA_VERSION.
    """
    excluded = set(TASK_EXCLUDED_GROUPS.get(task or "", ()))
    names: list[str] = []
    for group in ("structure", "value", "temporal", "asset_mix"):
        if group in excluded:
            continue
        names.extend(FEATURE_GROUPS[group])
    return names


class AddressFeatures(BaseModel):
    """Features for one address, keyed by name rather than positional.

    Named storage is what lets an explanation print "out_transfer_count = 44"
    next to an importance value. `to_vector` is the only place a positional
    ordering is imposed, and it always goes through `feature_names`.
    """

    address: str
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    values: dict[str, float]
    # Set when the address had no timestamped transfers, so temporal features
    # are structurally absent rather than genuinely zero.
    temporal_data_missing: bool = False

    def to_vector(self, task: Optional[str] = None) -> list[float]:
        return [float(self.values.get(name, 0.0)) for name in feature_names(task)]


def _entropy(counts: list[int]) -> float:
    """Shannon entropy in bits over a count histogram, 0.0 when degenerate."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _log1p(value: float) -> float:
    """log1p on a non-negative magnitude.

    Amounts on a chain span roughly twenty orders of magnitude, and a raw
    scale would make every linear model a function of the single largest
    transfer. Negative inputs cannot occur (validation rejects them) but are
    clamped rather than raising, so a malformed cached graph degrades a
    feature instead of aborting an investigation.
    """
    return math.log1p(max(value, 0.0))


def _edge_records(graph: nx.MultiDiGraph, address: str) -> tuple[list[dict], list[dict], int]:
    """Collects the address's outbound and inbound edge data exactly once each.

    A self-loop appears in both `out_edges` and `in_edges`; it is counted as
    outbound only and reported separately, so `total_transfer_count` matches
    the number of distinct edges the address actually appears on.
    """
    outbound: list[dict] = []
    inbound: list[dict] = []
    self_loops = 0

    for _, target, key, data in graph.out_edges(address, keys=True, data=True):
        if target == address:
            self_loops += 1
        record = dict(data)
        record["_counterparty"] = target
        outbound.append(record)

    for source, _, key, data in graph.in_edges(address, keys=True, data=True):
        if source == address:
            continue  # already counted as outbound
        record = dict(data)
        record["_counterparty"] = source
        inbound.append(record)

    return outbound, inbound, self_loops


def extract_address_features(
    graph: nx.MultiDiGraph, address: str
) -> AddressFeatures:
    """Computes the full label-blind feature vector for one address.

    An address absent from the graph yields an all-zero vector rather than an
    error: the ML layer must never be the reason an investigation cannot
    finish, and a zero vector is honestly what "no observed activity" looks
    like. `temporal_data_missing` distinguishes that from a genuine zero.
    """
    address = address.lower()
    values: dict[str, float] = {name: 0.0 for name in feature_names()}

    if address not in graph:
        return AddressFeatures(
            address=address, values=values, temporal_data_missing=True
        )

    outbound, inbound, self_loops = _edge_records(graph, address)
    all_edges = outbound + inbound
    out_count = len(outbound)
    in_count = len(inbound)
    total = out_count + in_count

    if total == 0:
        return AddressFeatures(
            address=address, values=values, temporal_data_missing=True
        )

    # --- structure ---------------------------------------------------------
    out_parties = {e["_counterparty"] for e in outbound} - {address}
    in_parties = {e["_counterparty"] for e in inbound} - {address}
    all_parties = out_parties | in_parties
    reciprocal = out_parties & in_parties

    per_party: dict[str, int] = defaultdict(int)
    for edge in all_edges:
        counterparty = edge["_counterparty"]
        if counterparty != address:
            per_party[counterparty] += 1

    values["out_transfer_count"] = float(out_count)
    values["in_transfer_count"] = float(in_count)
    values["total_transfer_count"] = float(total)
    values["unique_out_counterparties"] = float(len(out_parties))
    values["unique_in_counterparties"] = float(len(in_parties))
    values["unique_counterparties"] = float(len(all_parties))
    values["reciprocal_counterparty_count"] = float(len(reciprocal))
    values["reciprocity_ratio"] = (
        len(reciprocal) / len(all_parties) if all_parties else 0.0
    )
    values["out_share_of_transfers"] = out_count / total
    values["transfers_per_counterparty"] = (
        sum(per_party.values()) / len(per_party) if per_party else 0.0
    )
    party_total = sum(per_party.values())
    if party_total:
        values["top_counterparty_share"] = max(per_party.values()) / party_total
        values["counterparty_concentration_hhi"] = sum(
            (n / party_total) ** 2 for n in per_party.values()
        )
    values["self_loop_count"] = float(self_loops)

    # --- value -------------------------------------------------------------
    out_amounts = [float(e.get("amount") or 0.0) for e in outbound]
    in_amounts = [float(e.get("amount") or 0.0) for e in inbound]
    amounts = out_amounts + in_amounts

    values["log_total_out_amount"] = _log1p(sum(out_amounts))
    values["log_total_in_amount"] = _log1p(sum(in_amounts))
    values["log_mean_out_amount"] = _log1p(
        sum(out_amounts) / len(out_amounts) if out_amounts else 0.0
    )
    values["log_mean_in_amount"] = _log1p(
        sum(in_amounts) / len(in_amounts) if in_amounts else 0.0
    )
    values["log_max_amount"] = _log1p(max(amounts) if amounts else 0.0)
    values["log_amount_stdev"] = _log1p(
        statistics.pstdev(amounts) if len(amounts) > 1 else 0.0
    )
    # Repeated amounts: a low ratio means the same figure recurs, which is
    # what automated forwarding and structuring both look like.
    values["distinct_amount_ratio"] = (
        len({round(a, 12) for a in amounts}) / len(amounts) if amounts else 0.0
    )
    values["zero_amount_share"] = (
        sum(1 for a in amounts if a == 0.0) / len(amounts) if amounts else 0.0
    )

    # --- temporal ----------------------------------------------------------
    timestamps = sorted(
        int(e["timestamp"]) for e in all_edges if e.get("timestamp")
    )
    values["timestamped_share"] = len(timestamps) / total
    if not timestamps:
        return AddressFeatures(
            address=address, values=values, temporal_data_missing=True
        )

    span = timestamps[-1] - timestamps[0]
    values["lifespan_days"] = span / SECONDS_PER_DAY

    days = {ts // SECONDS_PER_DAY for ts in timestamps}
    values["active_day_count"] = float(len(days))
    values["transfers_per_active_day"] = len(timestamps) / len(days)

    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    if gaps:
        values["log_median_gap_seconds"] = _log1p(statistics.median(gaps))
        values["log_min_gap_seconds"] = _log1p(min(gaps))

    hours: dict[int, int] = defaultdict(int)
    weekdays: dict[int, int] = defaultdict(int)
    for ts in timestamps:
        moment = datetime.fromtimestamp(ts, tz=timezone.utc)
        hours[moment.hour] += 1
        weekdays[moment.weekday()] += 1
    values["night_hour_share"] = (
        sum(count for hour, count in hours.items() if 0 <= hour <= 5)
        / len(timestamps)
    )
    values["hour_entropy"] = _entropy(list(hours.values()))
    values["weekday_entropy"] = _entropy(list(weekdays.values()))

    # Densest one-hour window, by a sliding count over the sorted stamps —
    # a real burst measure rather than a per-calendar-hour bucket count.
    window_start = 0
    busiest = 0
    for index, ts in enumerate(timestamps):
        while ts - timestamps[window_start] > 3600:
            window_start += 1
        busiest = max(busiest, index - window_start + 1)
    values["max_transfers_in_one_hour"] = float(busiest)

    # --- asset mix ---------------------------------------------------------
    assets = {str(e.get("asset") or "UNKNOWN") for e in all_edges}
    values["distinct_asset_count"] = float(len(assets))
    values["native_transfer_share"] = (
        sum(1 for e in all_edges if e.get("asset_type") == "NATIVE") / total
    )

    return AddressFeatures(address=address, values=values)


def extract_many(
    graph: nx.MultiDiGraph, addresses: list[str]
) -> list[AddressFeatures]:
    """Feature extraction for a list of addresses, in the order given.

    Order is preserved rather than sorted so a caller's pairing of addresses
    to labels cannot be silently permuted.
    """
    return [extract_address_features(graph, address) for address in addresses]
