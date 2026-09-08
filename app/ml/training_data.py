"""
MACRO MILESTONE 5 — synthetic training data.

--------------------------------------------------------------------------
SYNTHETIC / DEMO ONLY (do not remove): every row here is a hand-constructed
WalletFeatures object with a placeholder wallet label
("SYNTHETIC_DEMO_POS_NN" / "SYNTHETIC_DEMO_NEG_NN") — never a real address,
and specifically NEVER one of the addresses in data/seed/known_vasps.json
or data/seed/demo_known_vasps.json (Macro Milestone 4's seed sets). Using
real seed addresses as ML training labels would let the ML layer
effectively re-derive M4's ground truth and dress it up as an independent
"prediction", which is exactly the kind of false-independence this
project must not produce.

This dataset exists ONLY to prove the M5 feature-extraction -> training ->
prediction pipeline runs end to end. It has no claim to represent real
VASP-connected wallet behavior, and every consumer of this module (see
app/ml/predictor.py, ml_attribution.py) must keep the SYNTHETIC_DEMO
labeling attached to anything derived from it.

The two classes are built from simple, deterministic arithmetic on the
row index — no randomness at all is needed to construct the dataset
itself (only model training, in predictor.py, takes a random seed).
--------------------------------------------------------------------------
"""

from __future__ import annotations

from app.ml.models import WalletFeatures

SYNTHETIC_DATA_DISCLAIMER = (
    "This prediction was produced by a model trained ONLY on a small, "
    "hand-constructed SYNTHETIC/DEMO dataset (app/ml/training_data.py). "
    "It does not reflect real-world VASP-connected wallet behavior, "
    "carries no statistically meaningful accuracy claim, and must never "
    "be treated as evidence of VASP attribution on its own. See "
    "app/attribution/ (Macro Milestone 4) for the actual evidence-based "
    "attribution pipeline this prediction does not replace or override."
)

_LABEL_CONNECTED = "LIKELY_VASP_CONNECTED"
_LABEL_NOT_CONNECTED = "LIKELY_NOT_VASP_CONNECTED"

_ROWS_PER_CLASS = 18


def _make_connected_row(i: int) -> tuple[WalletFeatures, str]:
    """A synthetic wallet shaped like one with real fund-flow reach and
    existing M4 evidence: moderate-to-high degree, multiple traced paths,
    at least one behavioral flag, and DIRECT or INDIRECT attribution
    evidence already present."""
    in_degree = 8 + i
    out_degree = 6 + i
    unique_in = min(in_degree, 5 + (i % 4))
    unique_out = min(out_degree, 4 + (i % 3))
    path_count = 2 + (i % 5)
    max_hop = 1 + (i % 3)
    avg_hop = round((max_hop + 1) / 2, 2)
    max_dur = 300.0 + i * 50
    avg_dur = round(max_dur / 2, 2)
    unknown_dur = i % 2

    split = i % 3 == 0
    consolidation = i % 4 == 0
    rapid = i % 2 == 0
    high_frequency = i % 5 == 0
    forwarding = i % 3 == 1
    burst = i % 4 == 1

    has_direct = i % 2 == 0
    has_indirect = not has_direct

    features = WalletFeatures(
        wallet=f"SYNTHETIC_DEMO_POS_{i:02d}",
        in_degree=in_degree,
        out_degree=out_degree,
        unique_in_counterparties=unique_in,
        unique_out_counterparties=unique_out,
        total_edge_count=in_degree + out_degree,
        path_count=path_count,
        max_hop_count=max_hop,
        avg_hop_count=avg_hop,
        max_path_duration_seconds=max_dur,
        avg_path_duration_seconds=avg_dur,
        paths_with_unknown_duration=unknown_dur,
        has_split_pattern=split,
        has_consolidation_pattern=consolidation,
        has_rapid_hopping=rapid,
        has_high_frequency_counterparty=high_frequency,
        has_repeated_forwarding=forwarding,
        has_temporal_burst=burst,
        behavior_pattern_count=sum(
            [split, consolidation, rapid, high_frequency, forwarding, burst]
        ),
        attribution_status="MATCH_FOUND",
        has_direct_evidence=has_direct,
        has_indirect_evidence=has_indirect,
        candidate_count=1 + (i % 2),
    )
    return features, _LABEL_CONNECTED


def _make_unconnected_row(i: int) -> tuple[WalletFeatures, str]:
    """A synthetic wallet shaped like one with little graph presence, no
    meaningful traced flow, no behavioral flags, and no M4 attribution
    evidence at all."""
    in_degree = i % 3
    out_degree = i % 2
    unique_in = min(in_degree, 1)
    unique_out = min(out_degree, 1)
    path_count = i % 2  # 0 or 1 short/uninteresting paths
    max_hop = path_count
    avg_hop = float(max_hop)
    # Sparse wallets rarely have a clean multi-hop duration to report.
    unknown_dur = path_count

    features = WalletFeatures(
        wallet=f"SYNTHETIC_DEMO_NEG_{i:02d}",
        in_degree=in_degree,
        out_degree=out_degree,
        unique_in_counterparties=unique_in,
        unique_out_counterparties=unique_out,
        total_edge_count=in_degree + out_degree,
        path_count=path_count,
        max_hop_count=max_hop,
        avg_hop_count=avg_hop,
        max_path_duration_seconds=None,
        avg_path_duration_seconds=None,
        paths_with_unknown_duration=unknown_dur,
        has_split_pattern=False,
        has_consolidation_pattern=False,
        has_rapid_hopping=False,
        has_high_frequency_counterparty=False,
        has_repeated_forwarding=False,
        has_temporal_burst=False,
        behavior_pattern_count=0,
        attribution_status="NONE",
        has_direct_evidence=False,
        has_indirect_evidence=False,
        candidate_count=0,
    )
    return features, _LABEL_NOT_CONNECTED


SYNTHETIC_TRAINING_DATA: list[tuple[WalletFeatures, str]] = [
    _make_connected_row(i) for i in range(_ROWS_PER_CLASS)
] + [_make_unconnected_row(i) for i in range(_ROWS_PER_CLASS)]
