"""
Output models for Macro Milestone 3 / Phase B (behavioral pattern
detection, "original Milestone 4").

IMPORTANT PRINCIPLE (do not remove): patterns are indicators, not proof.
BehaviorPattern never carries a criminal/VASP-identity label — only
pattern_type, evidence, metrics, and the addresses involved. See
app/behavior/detectors.py module docstring for the full list of labels
this module must never assign.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class PatternType(str, Enum):
    SPLIT_PATTERN = "SPLIT_PATTERN"  # fan-out / splitting
    CONSOLIDATION_PATTERN = "CONSOLIDATION_PATTERN"  # fan-in / consolidation
    RAPID_HOPPING = "RAPID_HOPPING"
    HIGH_FREQUENCY_COUNTERPARTY = "HIGH_FREQUENCY_COUNTERPARTY"
    REPEATED_FORWARDING = "REPEATED_FORWARDING"
    TEMPORAL_BURST = "TEMPORAL_BURST"

    # --- Additional indicators (additive; every one is threshold-driven from
    # app/core/config.py and reports its own threshold + observed value). ---
    # Value arrived and left again almost immediately (rapid pass-through).
    FAST_INBOUND_OUTBOUND = "FAST_INBOUND_OUTBOUND"
    # A single counterparty dominates the wallet's activity.
    HIGH_COUNTERPARTY_CONCENTRATION = "HIGH_COUNTERPARTY_CONCENTRATION"
    # The wallet handled an unusually wide range of distinct assets.
    ASSET_DIVERSITY = "ASSET_DIVERSITY"
    # The same (asset, amount) pair recurs — structuring-like, or automation.
    REPEATED_AMOUNT_PATTERN = "REPEATED_AMOUNT_PATTERN"
    # Long silence followed by a concentrated burst of activity.
    DORMANT_THEN_ACTIVE = "DORMANT_THEN_ACTIVE"
    # Native-asset transfers at or above the configured notable size.
    LARGE_VALUE_TRANSFER = "LARGE_VALUE_TRANSFER"
    # Activity concentrated in a low-business-activity UTC hour band.
    UNUSUAL_TIMING = "UNUSUAL_TIMING"
    # Transfer counts strongly one-directional (mostly in, or mostly out).
    IN_OUT_IMBALANCE = "IN_OUT_IMBALANCE"
    # Many transfers per active day.
    HIGH_ACTIVITY_DENSITY = "HIGH_ACTIVITY_DENSITY"


class IndicatorClass(str, Enum):
    """How much weight a finding may be given — deliberately NOT a severity
    score, and deliberately never a criminality label.

    INVESTIGATIVE_INDICATOR: an observation worth investigating. Stands on
        its own as a described, measured fact; implies nothing about intent.
    SUPPORTING_EVIDENCE: only meaningful alongside address-level evidence
        (e.g. it corroborates an existing VASP candidate). Never a basis for
        attribution by itself.
    CONTEXTUAL: background colour. Has known confounders that make it
        unreliable in isolation (e.g. UTC hour, with the wallet's real
        timezone unknown).
    REQUIRES_FURTHER_VERIFICATION: the observation is real but the dataset
        available here cannot establish what it means — off-chain records or
        another data source are needed.
    """

    INVESTIGATIVE_INDICATOR = "INVESTIGATIVE_INDICATOR"
    SUPPORTING_EVIDENCE = "SUPPORTING_EVIDENCE"
    CONTEXTUAL = "CONTEXTUAL"
    REQUIRES_FURTHER_VERIFICATION = "REQUIRES_FURTHER_VERIFICATION"


class BehaviorPattern(BaseModel):
    pattern_type: PatternType
    wallet: str

    evidence: list[str]  # short, specific, human-readable evidence lines
    metrics: dict[str, float | int | str]

    related_addresses: list[str]

    first_seen: Optional[int] = None
    last_seen: Optional[int] = None

    # Deliberately no numeric confidence/severity score: with only
    # structural + timing evidence and no VASP/labeling data yet, any
    # single number here would imply more certainty than the evidence
    # supports. Add only if a later milestone's data actually justifies it.
    confidence: Optional[str] = None

    # --- Additive transparency fields ------------------------------------
    # All optional with defaults, so every detector and test written before
    # they existed keeps working unchanged. A report must be able to state
    # "indicator X fired because <observed_metric> = <observed_value>, and
    # the configured threshold is <threshold>" without the reader having to
    # read the detector's source.
    observed_metric: Optional[str] = None  # which metric was compared
    observed_value: Optional[float | int | str] = None  # its actual value
    threshold: Optional[float | int | str] = None  # the configured trigger
    threshold_setting: Optional[str] = None  # the Settings field it came from
    relevant_tx_hashes: list[str] = []  # concrete on-chain references
    classification: IndicatorClass = IndicatorClass.INVESTIGATIVE_INDICATOR

    @property
    def indicator_name(self) -> str:
        """Stable, human-facing name — the enum value, exposed as a property
        so reports never build their own label from the pattern type."""
        return self.pattern_type.value
