"""
UNSUPERVISED BEHAVIOURAL OUTLIER ANALYSIS — the strongest honest model the
available real data actually supports.

WHY THIS MODULE EXISTS
--------------------------------------------------------------------------
`app/ml/real_labels.py` refuses to produce a labelled set when the real,
verifiable labels are too few (which, for the VASP task, they always are: a
curated seed set has no honest negative class). The wrong response to that is
to invent labels until a supervised model reports 75-80%. The right response
is to use a method that needs no labels at all, and to be explicit that it
answers a different, weaker question.

Supervised model:    "is this address a VASP deposit address?"  (needs labels)
This module:         "how unusual is this address's behaviour compared with
                      the population in this graph?"            (needs none)

The second question is genuinely useful in an investigation — an address whose
structure, value profile and timing all sit in the tail of its own graph is
worth an analyst's attention — but it is NOT the first question, and nothing
here should be read as answering it.

WHAT AN ANOMALY SCORE IS NOT
--------------------------------------------------------------------------
  * Not a probability of anything. IsolationForest returns a relative measure
    of how few splits it takes to isolate a point; it has no calibrated
    probabilistic meaning and is never reported as one.
  * Not accuracy. There are no labels, so there is nothing to be accurate
    against. `OutlierAssessment.evaluation` says so in those words, and the
    only quantitative quality figure reported is rank stability (below).
  * Not suspicion. Exchange hot wallets, routers and bridges are the most
    behaviourally extreme addresses on any chain and are entirely legitimate.

THE ONE HONEST METRIC AVAILABLE: RANK STABILITY
--------------------------------------------------------------------------
Without labels we cannot ask "is the score right", but we can ask "is the
score reproducible", which is a real and falsifiable question. The model is
refit on bootstrap resamples of the population and the Spearman rank
correlation between each resample's ordering and the full-population ordering
is reported. High correlation means the ranking reflects the population's
structure rather than the particular sample; low correlation means the ranking
is unstable and must not be relied on. This is reported whatever it turns out
to be.
"""

from __future__ import annotations

import platform
import time
from typing import Optional

import networkx as nx
import numpy as np
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.core.config import Settings, get_settings
from app.ml.real_features import (
    FEATURE_SCHEMA_VERSION,
    extract_address_features,
    feature_names,
)

METHOD_VERSION = "real-unsupervised-v1"
METHOD_NAME = "IsolationForest"

#: Percentile band a feature must fall OUTSIDE of to be reported as a
#: deviation. Two-sided: unusually low is as informative as unusually high (an
#: address that only ever receives is as notable as one that only sends).
HIGH_PERCENTILE = 90.0
LOW_PERCENTILE = 10.0


class FeatureDeviation(BaseModel):
    """One feature placing the target in the population's tail.

    Every field is checkable against the graph by hand — that is the point.
    An explanation an analyst cannot verify is not an explanation.
    """

    feature: str
    value: float
    population_median: float
    population_p10: float
    population_p90: float
    percentile_rank: float
    direction: str  # "ABOVE_POPULATION" | "BELOW_POPULATION"


class RankStability(BaseModel):
    """Bootstrap reproducibility of the outlier ranking."""

    resamples: int
    mean_spearman_correlation: float
    min_spearman_correlation: float
    interpretation: str


class OutlierAssessment(BaseModel):
    """The complete unsupervised assessment for one address."""

    address: str
    available: bool

    method: str = METHOD_NAME
    method_version: str = METHOD_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    random_seed: Optional[int] = None
    assessed_at_utc: Optional[str] = None

    # The population the address was compared against, described precisely
    # enough to reproduce: a score is meaningless without it.
    population_size: int = 0
    population_source: Optional[str] = None
    population_activity_floor: int = 0
    graph_node_count: int = 0
    graph_edge_count: int = 0
    addresses_below_activity_floor: int = 0

    # Higher = more unusual. Deliberately NOT called a probability.
    outlier_score: Optional[float] = None
    percentile_within_population: Optional[float] = None
    flagged_as_outlier: Optional[bool] = None
    contamination_setting: Optional[float] = None
    target_in_population: bool = False

    deviations: list[FeatureDeviation] = []
    rank_stability: Optional[RankStability] = None

    #: Fixed: behavioural unusualness is context, never proof.
    evidence_class: str = "CONTEXTUAL"
    evaluation: list[str] = []
    limitations: list[str] = []
    unavailable_reason: Optional[str] = None
    environment: dict[str, str] = {}


def _eligible_population(
    graph: nx.MultiDiGraph, activity_floor: int
) -> tuple[list[str], int]:
    """Addresses with at least `activity_floor` transfer edges.

    The floor exists because features like `hour_entropy` or
    `distinct_amount_ratio` are meaningless for an address with one transfer,
    and a population padded with such addresses would make almost every real
    wallet look like an outlier by comparison. Excluded addresses are counted
    and reported rather than silently dropped.
    """
    eligible: list[str] = []
    excluded = 0
    for node in graph.nodes():
        degree = graph.in_degree(node) + graph.out_degree(node)
        if degree >= activity_floor:
            eligible.append(str(node))
        else:
            excluded += 1
    # Sorted so the population order — and therefore the fit — is deterministic.
    return sorted(eligible), excluded


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, computed directly to avoid a scipy dependency.

    Pearson correlation of the ranks, which is the definition. Ties are
    handled by `argsort`-of-`argsort` ordinal ranking, which is adequate here
    because anomaly scores are continuous and exact ties are rare.
    """
    if len(a) < 2:
        return 0.0
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    if rank_a.std() == 0 or rank_b.std() == 0:
        return 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def _rank_stability(
    matrix: np.ndarray,
    baseline_scores: np.ndarray,
    seed: int,
    resamples: int = 10,
) -> RankStability:
    """Refits on bootstrap resamples and compares orderings.

    Each resample refits the forest on a bootstrap draw of the population but
    scores the FULL population, so the two orderings are over the same points
    and are directly comparable.
    """
    rng = np.random.default_rng(seed)
    correlations: list[float] = []
    n = matrix.shape[0]

    for index in range(resamples):
        draw = rng.integers(0, n, size=n)
        model = IsolationForest(
            n_estimators=200,
            random_state=seed + index + 1,
            contamination="auto",
        )
        try:
            model.fit(matrix[draw])
            correlations.append(
                _spearman(baseline_scores, -model.score_samples(matrix))
            )
        except Exception:  # pragma: no cover - degenerate resample
            continue

    if not correlations:  # pragma: no cover
        return RankStability(
            resamples=0,
            mean_spearman_correlation=0.0,
            min_spearman_correlation=0.0,
            interpretation=(
                "Rank stability could not be measured; treat the ranking as "
                "unvalidated."
            ),
        )

    mean = float(np.mean(correlations))
    if mean >= 0.9:
        verdict = (
            "The ordering is highly reproducible across resamples, so it "
            "reflects the population's structure rather than the sample drawn."
        )
    elif mean >= 0.7:
        verdict = (
            "The ordering is moderately reproducible. Treat the broad ranking "
            "as meaningful and individual positions as approximate."
        )
    else:
        verdict = (
            "The ordering is NOT reproducible across resamples. It must not be "
            "relied on for prioritisation; the population is likely too small "
            "or too heterogeneous for a stable ranking."
        )

    return RankStability(
        resamples=len(correlations),
        mean_spearman_correlation=round(mean, 4),
        min_spearman_correlation=round(float(min(correlations)), 4),
        interpretation=verdict,
    )


def assess_address(
    graph: nx.MultiDiGraph,
    address: str,
    population_source: str,
    settings: Optional[Settings] = None,
    max_deviations: int = 8,
    stability_resamples: int = 10,
) -> OutlierAssessment:
    """Scores one address's behavioural unusualness against its own graph.

    Returns `available=False` — with a stated reason — when the graph cannot
    support the comparison at all: too few eligible addresses for a population
    to mean anything, or a target with no observed activity. Both are honest
    outcomes; a score computed from a five-address population would not be.
    """
    settings = settings or get_settings()
    seed = settings.ml_random_seed
    floor = settings.ml_min_transfers_per_sample
    address = address.lower()

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    base = dict(
        address=address,
        random_seed=seed,
        assessed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        population_source=population_source,
        population_activity_floor=floor,
        graph_node_count=graph.number_of_nodes(),
        graph_edge_count=graph.number_of_edges(),
        environment=environment,
    )

    population, excluded = _eligible_population(graph, floor)

    # A tail cannot be defined without a distribution. 30 is the smallest
    # population where a 90th-percentile statement is worth making at all.
    MIN_POPULATION = 30
    if len(population) < MIN_POPULATION:
        return OutlierAssessment(
            available=False,
            population_size=len(population),
            addresses_below_activity_floor=excluded,
            unavailable_reason=(
                f"Only {len(population)} address(es) in this graph have at least "
                f"{floor} transfers, below the minimum population of "
                f"{MIN_POPULATION} needed for a percentile comparison to mean "
                "anything. No anomaly score is reported."
            ),
            **base,
        )

    names = feature_names()
    target_features = extract_address_features(graph, address)
    if address not in graph or target_features.values.get(
        "total_transfer_count", 0.0
    ) == 0.0:
        return OutlierAssessment(
            available=False,
            population_size=len(population),
            addresses_below_activity_floor=excluded,
            unavailable_reason=(
                f"Address {address} has no transfer activity in the loaded "
                "graph, so there is no behaviour to compare against the "
                "population. No anomaly score is reported."
            ),
            **base,
        )

    rows = [
        extract_address_features(graph, node).to_vector() for node in population
    ]
    matrix = np.asarray(rows, dtype=float)

    target_in_population = address in set(population)
    target_vector = np.asarray([target_features.to_vector()], dtype=float)

    # Standardising first keeps a feature measured in wei-scale logs from
    # dominating the split geometry purely because of its units.
    scaler = StandardScaler().fit(matrix)
    scaled_population = scaler.transform(matrix)
    scaled_target = scaler.transform(target_vector)

    model = IsolationForest(
        n_estimators=300,
        random_state=seed,
        contamination="auto",
    ).fit(scaled_population)

    # Negated so that a larger number means more unusual, which is the
    # direction a reader will assume.
    population_scores = -model.score_samples(scaled_population)
    target_score = float(-model.score_samples(scaled_target)[0])
    percentile = float((population_scores <= target_score).mean() * 100.0)
    flagged = bool(model.predict(scaled_target)[0] == -1)

    deviations: list[FeatureDeviation] = []
    for index, name in enumerate(names):
        column = matrix[:, index]
        value = float(target_features.values.get(name, 0.0))
        p10 = float(np.percentile(column, LOW_PERCENTILE))
        p90 = float(np.percentile(column, HIGH_PERCENTILE))
        rank = float((column <= value).mean() * 100.0)

        # The percentile rank alone is not enough to call something a
        # deviation. On a near-constant column (every address has
        # timestamped_share 1.0, most have self_loop_count 0.0) the rank of a
        # perfectly ordinary value is 100, which would report the population
        # median as an extreme. Requiring the value to actually fall outside
        # the p10-p90 band makes "ABOVE_POPULATION" mean what it says.
        if value > p90:
            direction = "ABOVE_POPULATION"
        elif value < p10:
            direction = "BELOW_POPULATION"
        else:
            continue

        deviations.append(
            FeatureDeviation(
                feature=name,
                value=round(value, 6),
                population_median=round(float(np.median(column)), 6),
                population_p10=round(p10, 6),
                population_p90=round(p90, 6),
                percentile_rank=round(rank, 2),
                direction=direction,
            )
        )
    # Most extreme first, so a truncated list keeps the most informative rows.
    deviations.sort(key=lambda d: -abs(d.percentile_rank - 50.0))

    stability = _rank_stability(
        scaled_population, population_scores, seed, stability_resamples
    )

    return OutlierAssessment(
        available=True,
        population_size=len(population),
        addresses_below_activity_floor=excluded,
        outlier_score=round(target_score, 6),
        percentile_within_population=round(percentile, 2),
        flagged_as_outlier=flagged,
        contamination_setting=None,  # "auto": set by the estimator, not by us
        target_in_population=target_in_population,
        deviations=deviations[:max_deviations],
        rank_stability=stability,
        evaluation=[
            "No accuracy, precision, recall or F1 is reported for this model, "
            "and none can be: unsupervised outlier detection has no labels to "
            "be scored against. Any such figure would be fabricated.",
            "The one quantitative quality measure available is rank stability "
            f"across {stability.resamples} bootstrap resamples: mean Spearman "
            f"correlation {stability.mean_spearman_correlation}, minimum "
            f"{stability.min_spearman_correlation}. {stability.interpretation}",
            f"The score places this address at the "
            f"{round(percentile, 2)}th percentile of "
            f"{len(population)} addresses drawn from {population_source}.",
        ],
        limitations=[
            "The population is the addresses in THIS graph, which was built "
            "outward from the investigated wallet. It is therefore a "
            "neighbourhood sample, not a sample of the chain, and 'unusual "
            "here' does not mean 'unusual on Ethereum'.",
            "An IsolationForest score is a relative isolation measure with no "
            "calibrated probabilistic meaning. It is not a probability and not "
            "a likelihood of any behaviour.",
            "Behavioural unusualness is not wrongdoing. Exchange hot wallets, "
            "routers, bridges and market makers are the most extreme addresses "
            "on any chain and are entirely legitimate.",
            "This model answers 'how unusual is this behaviour', not 'is this "
            "a VASP'. It is not a substitute for the supervised task and is "
            "reported as CONTEXTUAL evidence only.",
        ]
        + (
            []
            if target_in_population
            else [
                "The target address did not clear the population activity "
                f"floor of {floor} transfers, so it was scored against the "
                "population without being part of it. Its percentile is still "
                "well defined, but it is a comparison to busier addresses."
            ]
        ),
        **base,
    )
