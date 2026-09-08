"""
Centralized configuration for the blockchain intelligence engine.

Every threshold, limit, and provider setting must be read from here —
never hardcoded inline in business logic. This file is the single
source of truth and is loaded once at process start.

A malformed value RAISES rather than falling back to the default. Falling
back would be worse than failing: the report prints the threshold each
finding was measured against, so quietly substituting a different number
would make the report state a threshold the operator did not choose. The
error names the variable and what it expected, so a typo in `.env` is a
one-line fix rather than a traceback to interpret.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class ConfigurationError(Exception):
    """A setting is present but unusable. Raised at configuration-read time,
    before any investigation starts, so a typo costs nothing.

    Only ever reports the value of a NUMERIC or BOOLEAN threshold, never of a
    credential: the credential settings are read with `os.getenv` directly and
    never pass through these helpers.
    """


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigurationError(
            f"{name}={raw.strip()!r} is not a whole number. Fix it in .env or "
            f"remove the line to use the default ({default})."
        ) from exc


def _get_optional_int(name: str) -> Optional[int]:
    """Same strictness as `_get_int`, but "unset" is a meaningful value.

    Used where a default would be a guess rather than a sensible fallback:
    ETHERSCAN_CHAIN_ID has no safe default because the safe behaviour is to
    derive the id from the chain name, and a default of 1 would silently
    contradict every chain except Ethereum.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigurationError(
            f"{name}={raw.strip()!r} is not a whole number. Fix it in .env or "
            "remove the line to leave it unset."
        ) from exc


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ConfigurationError(
            f"{name}={raw.strip()!r} is not a number. Fix it in .env or remove "
            f"the line to use the default ({default})."
        ) from exc


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    # Previously anything unrecognised silently meant False, so
    # PROVIDER_CACHE_ENABLED=ture disabled the cache without saying so.
    raise ConfigurationError(
        f"{name}={raw.strip()!r} is not a boolean. Use one of "
        f"{sorted(_TRUE)} or {sorted(_FALSE)}, or remove the line to use the "
        f"default ({str(default).lower()})."
    )


@dataclass(frozen=True)
class Settings:
    etherscan_api_key: str
    etherscan_base_url: str

    max_transactions_per_investigation: int
    default_lookback_days: int
    http_timeout_seconds: int
    http_max_retries: int

    #: Optional override for the Etherscan V2 `chainid` parameter. Normally
    #: unset, which is why it defaults to None: the chain id is derived from the
    #: chain NAME by app.blockchain.chains, so the id queried and the name
    #: written onto every transfer cannot disagree. Set it to pin an expected
    #: id, and the provider will refuse to run if it contradicts the named
    #: chain. It sits here rather than beside the other Etherscan settings only
    #: because a dataclass requires defaulted fields to follow required ones.
    etherscan_chain_id: Optional[int] = None

    # --- Macro Milestone 3 / Phase A: multi-hop fund-flow tracing ---
    # MAX_HOPS: how many edges deep a trace will follow from the source
    # wallet. Kept small by default (4) because hop count grows the search
    # space combinatorially on a dense graph (Milestone 2's verified fixture
    # has 3646 edges over 2303 nodes).
    # Defaults are set on these new fields (rather than left required) so
    # that pre-existing direct `Settings(...)` construction (see
    # tests/test_validation.py, written before this milestone) keeps working
    # unmodified — get_settings() below still always passes explicit values
    # sourced from config/env, these defaults only matter for that call site.
    fund_trace_max_hops: int = 4
    # MAX_PATHS: hard cap on the number of completed paths returned, so a
    # highly-connected source wallet can't produce an unbounded result set.
    fund_trace_max_paths: int = 500
    # MAX_EDGES_EXPLORED: hard cap on total DFS edge expansions (not just
    # completed paths) — the actual combinatorial-explosion guard, since a
    # search can do enormous amounts of work before ever completing (or
    # failing to complete) MAX_PATHS paths.
    fund_trace_max_edges_explored: int = 20000

    # --- Macro Milestone 3 / Phase B: behavioral pattern detection ---
    # Minimum unique outgoing/incoming counterparties for a wallet to be
    # flagged as a fan-out (splitting) / fan-in (consolidation) candidate.
    behavior_min_fanout_counterparties: int = 4
    behavior_min_fanin_counterparties: int = 4
    # A wallet pair is flagged HIGH_FREQUENCY_COUNTERPARTY once at least
    # this many transfer edges exist between the same two addresses
    # (either direction).
    behavior_high_frequency_min_transfers: int = 5
    # RAPID_HOPPING: every consecutive hop gap in a fund-flow path must be
    # <= this many seconds for the whole path to qualify.
    behavior_rapid_hop_max_seconds: int = 300
    # TEMPORAL_BURST: a wallet is flagged if at least
    # behavior_burst_min_transfers edges (in+out) fall inside any sliding
    # window of this many seconds.
    behavior_burst_window_seconds: int = 3600
    behavior_burst_min_transfers: int = 5
    # REPEATED_FORWARDING: an incoming edge followed by an outgoing edge to
    # a *different* counterparty within this many seconds counts as one
    # "receive-then-forward" event; a wallet is flagged once it has at
    # least behavior_min_forwarding_events such events.
    behavior_forwarding_window_seconds: int = 3600
    behavior_min_forwarding_events: int = 2

    # --- Macro Milestone 4: VASP intelligence + explainable attribution ---
    # Paths to the two seed datasets, kept structurally separate per the
    # "production vs synthetic must never mix" requirement:
    #   - vasp_seed_dataset_path: sourced, publicly-documented VASP
    #     addresses only. This is what attribute_wallet.py loads by default.
    #   - vasp_demo_seed_dataset_path: synthetic/demo addresses used only
    #     for tests and explicit --demo CLI runs. Never loaded by default.
    # No separate attribution hop-limit setting is introduced: candidate
    # generation walks the exact same FundFlowPath objects Phase A (M3)
    # already produces under FUND_TRACE_MAX_HOPS, so a second limit would
    # be redundant and could silently disagree with what was actually
    # traced.
    vasp_seed_dataset_path: str = "data/seed/known_vasps.json"
    vasp_demo_seed_dataset_path: str = "data/seed/demo_known_vasps.json"

    # The curated NEGATIVE reference for the supervised vasp_ownership task:
    # real addresses that are documented as NOT custodial accounts, each with
    # its own stated reason. Without it that task has a positive class only,
    # which is untrainable — and the one thing that must never happen instead
    # is labelling unlabelled addresses NOT_VASP_OWNED. Point this at a
    # missing path and the task simply stays untrainable and says so; the
    # investigation itself is unaffected.
    non_vasp_reference_path: str = "data/seed/non_vasp_reference.json"

    # --- Macro Milestone 6: persistence + investigation service ---
    # SQLite by default so local dev/tests need no external database running.
    # PostgreSQL is supported by simply pointing DATABASE_URL at a postgres://
    # (or postgresql+psycopg2://) URL — SQLAlchemy picks the right dialect
    # from the URL scheme; nothing else in this codebase needs to know which
    # backend is in use. This field has a default (like the M3 fields above)
    # so pre-existing direct `Settings(...)` construction in tests written
    # before M6 keeps working unmodified.
    database_url: str = "sqlite:///./data/blockchain_intel.db"

    # --- Provider response caching (production ingestion) ---
    # Raw provider responses are cached on disk under a DETERMINISTIC key
    # derived from (provider, chain, module, action, address, block range,
    # page, offset, sort) — never from wall-clock time — so the same query
    # always maps to the same cache file and a cached investigation is
    # byte-for-byte reproducible. The API key is NEVER part of the key and
    # is never written to the cache.
    provider_cache_enabled: bool = True
    provider_cache_dir: str = "data/cache/provider"
    # 0 == never expire. Cached blockchain history for a closed block range
    # does not change, so a long TTL is safe; the tail of an address's
    # history does grow, which is what --refresh is for.
    provider_cache_ttl_seconds: int = 86_400

    graph_cache_dir: str = "data/graphs"

    #: Where normalized real transfer files live. A run that reads one of
    #: these is still CACHED REAL DATA, never REAL — but unlike a reloaded
    #: graph it keeps per-record provenance, so the supervised labelling rules
    #: can run. Discovery is by exact filename (`{wallet}_{chain}.json`), never
    #: by scanning for something that looks close enough.
    transfers_cache_dir: str = "data/fixtures"

    #: Browser origins allowed to call the HTTP API, comma-separated. The React
    #: frontend in /frontend is served by Vite on 5173 in development, so that
    #: origin is the default. Deliberately NOT "*": the API is unauthenticated,
    #: and a wildcard would let any page the operator happens to have open drive
    #: investigations against this deployment. Set it to the real origin when
    #: the frontend is deployed anywhere else; set it to empty to refuse browser
    #: callers entirely. Read `cors_allow_origins` for the parsed form.
    api_cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Recursive multi-hop acquisition (live counterparty expansion) ---
    # Live acquisition walks outward from the investigated wallet hop by hop,
    # fetching each newly discovered counterparty's own transaction streams.
    # These three numbers are the only thing standing between one wallet and
    # the whole Ethereum graph, so they are hard ceilings, not hints, and
    # whichever one bites is named in the report rather than silently applied.
    #
    # Total addresses whose streams are fetched in one investigation,
    # INCLUDING the investigated wallet. At roughly three to nine provider
    # requests per address, 40 keeps a full run inside a couple of hundred
    # requests, which the free Etherscan tier serves without tripping its
    # rate limit.
    expansion_max_addresses: int = 40
    # Per-hop ceiling. Without it a single high-degree counterparty at hop 1
    # would consume the entire address budget at hop 2 and no deeper hop
    # would ever be attempted.
    expansion_max_addresses_per_hop: int = 30
    # Record budget for an EXPANDED address's streams, deliberately lower
    # than max_transactions_per_investigation (which stays reserved for the
    # investigated wallet's own history). A counterparty is fetched to find
    # onward routes, not to be investigated in its own right; when this
    # budget bites, that address's onward edges are only partially observed
    # and the report says so.
    expansion_max_records_per_stream: int = 1000

    # --- Fund-flow tracing: time window + targeted-search budgets ---
    # 0 == no time window (consider the wallet's full available history).
    # A window narrows traversal to edges within N days of the most recent
    # observed activity, which is both an investigative scope control and a
    # genuine pruning mechanism.
    fund_trace_time_window_days: int = 0
    # Targeted (destination-aware) search budgets. These are deliberately
    # SEPARATE from the exploratory fund_trace_* limits above: targeted
    # search prunes to the viable subgraph first, so it does far less work
    # per useful path and does not need the exploratory search's tight cap.
    # Raising these is not how combinatorial explosion is avoided — the
    # bidirectional level-pruning in app/tracing/targeted.py is.
    targeted_trace_max_paths_per_target: int = 25
    targeted_trace_max_edges_explored: int = 400_000

    # --- Fund-flow candidate plausibility grading ---
    # A traceable path is not automatically a plausible fund flow. These
    # thresholds drive app/tracing/quality.py, which downgrades a path when
    # the observable evidence argues against fund continuity.
    #
    # An intermediary whose total degree reaches this threshold is a
    # high-throughput address (router, DEX pool, bridge, mixer, large
    # exchange wallet). Thousands of unrelated parties transact with it, so
    # "the path went through it" carries essentially no information about a
    # relationship between the endpoints. Default 100 sits well above the
    # ~57 99th-percentile degree measured on a real single-wallet graph, so
    # it flags genuine hubs rather than ordinary busy wallets.
    path_quality_hub_degree_threshold: int = 100
    # A gap between consecutive hops longer than this makes continuity
    # implausible: an intermediary that held value for months has almost
    # certainly commingled it. Default 30 days.
    path_quality_max_hop_gap_seconds: int = 2_592_000

    # --- Behavioral intelligence (additional indicators) ---
    # FAST_INBOUND_OUTBOUND / rapid pass-through: an inbound transfer
    # followed by an outbound transfer within this many seconds.
    behavior_fast_passthrough_max_seconds: int = 600
    # HIGH_COUNTERPARTY_CONCENTRATION: flagged when a single counterparty
    # accounts for at least this share (0-1) of the wallet's transfer count.
    behavior_counterparty_concentration_min_share: float = 0.5
    behavior_concentration_min_transfers: int = 5
    # ASSET_DIVERSITY: flagged at or above this many distinct assets.
    behavior_min_asset_diversity: int = 4
    # REPEATED_AMOUNT_PATTERN: the same (asset, amount) pair seen at least
    # this many times. Amounts are compared after rounding to
    # behavior_repeated_amount_decimals places.
    behavior_repeated_amount_min_occurrences: int = 3
    behavior_repeated_amount_decimals: int = 8
    # DORMANT_THEN_ACTIVE: a gap of at least this long with no activity,
    # followed by at least behavior_reactivation_min_transfers transfers
    # inside behavior_reactivation_window_seconds.
    behavior_dormancy_min_seconds: int = 2_592_000  # 30 days
    behavior_reactivation_window_seconds: int = 86_400
    behavior_reactivation_min_transfers: int = 3
    # LARGE_VALUE_TRANSFER: native-asset transfers at or above this amount.
    # Denominated in the chain's native unit (ETH on Ethereum).
    behavior_large_value_native_amount: float = 100.0
    # UNUSUAL_TIMING: UTC hour window generally associated with low
    # legitimate-business activity. Reported as CONTEXTUAL only — timezone
    # is unknown for an on-chain address, so this can never stand alone.
    behavior_unusual_hour_start_utc: int = 0
    behavior_unusual_hour_end_utc: int = 5
    behavior_unusual_timing_min_share: float = 0.5
    behavior_unusual_timing_min_transfers: int = 5
    # IN_OUT_IMBALANCE: |in-out| / (in+out) at or above this ratio.
    behavior_in_out_imbalance_min_ratio: float = 0.8
    behavior_in_out_imbalance_min_transfers: int = 5
    # HIGH_ACTIVITY_DENSITY: transfers per active day at or above this.
    behavior_activity_density_min_per_day: float = 20.0
    behavior_activity_density_min_transfers: int = 20

    # --- Production machine learning (real-data pipeline) ---
    ml_artifact_dir: str = "data/ml"
    ml_random_seed: int = 42
    # Fractions of the *entity-level* population held out. The test split is
    # untouched during model selection; validation drives model choice.
    ml_test_fraction: float = 0.2
    ml_validation_fraction: float = 0.2
    ml_cv_folds: int = 5
    # Minimum labelled wallets per class before training is attempted at
    # all. Below this the pipeline refuses to train rather than reporting a
    # meaningless metric from a handful of rows.
    ml_min_samples_per_class: int = 20
    # Minimum transfers an address must have in the graph to enter the
    # labelled population AT ALL. Applied identically to every class before
    # any label is read, so it defines a comparable sampling frame rather
    # than filtering one class more aggressively than another. Without it,
    # "has enough graph presence to compute features" would itself correlate
    # with the label and leak.
    ml_min_transfers_per_sample: int = 3

    # --- Investigative priority ("risk") scoring ---
    # These are the ONLY numbers that turn findings into a score, and each
    # one is named in the output next to the points it contributed, so a
    # reader can reconstruct the total by hand. There is deliberately no
    # opaque aggregate: the score is defined as the sum of the stated
    # contributions and nothing else.
    #
    # A word on what the score is NOT: it is an investigative *priority*
    # signal, not a probability of wrongdoing. Reaching an exchange is
    # ordinary behaviour for a legitimate wallet; it scores because it gives
    # an investigator a real-world counterparty to follow up, not because it
    # is suspicious.
    risk_weight_direct_vasp_evidence: float = 25.0
    risk_weight_indirect_vasp_evidence: float = 15.0
    risk_weight_investigative_indicator: float = 6.0
    risk_weight_supporting_indicator: float = 3.0
    risk_weight_contextual_indicator: float = 1.0
    # An incomplete search raises priority rather than lowering it: "we could
    # not finish looking" is not the same as "there was nothing there".
    risk_weight_incomplete_search: float = 5.0
    # Band boundaries, in points. Stated in the report alongside the score.
    risk_band_medium_threshold: float = 20.0
    risk_band_high_threshold: float = 45.0

    @property
    def cors_allow_origins(self) -> tuple[str, ...]:
        """`api_cors_allow_origins` split into origins, blanks dropped.

        Stored as one string because that is what an environment variable is,
        and parsed here so every caller splits it the same way. An empty value
        yields an empty tuple, which the API reads as "no browser origin is
        allowed" — a deployment turning the frontend off, not a request for the
        default.
        """
        return tuple(
            origin.strip()
            for origin in self.api_cors_allow_origins.split(",")
            if origin.strip()
        )


def get_settings() -> Settings:
    return Settings(
        etherscan_api_key=os.getenv("ETHERSCAN_API_KEY", ""),
        etherscan_base_url=os.getenv(
            "ETHERSCAN_BASE_URL", "https://api.etherscan.io/v2/api"
        ),
        etherscan_chain_id=_get_optional_int("ETHERSCAN_CHAIN_ID"),
        max_transactions_per_investigation=_get_int(
            "MAX_TRANSACTIONS_PER_INVESTIGATION", 2000
        ),
        default_lookback_days=_get_int("DEFAULT_LOOKBACK_DAYS", 90),
        http_timeout_seconds=_get_int("HTTP_TIMEOUT_SECONDS", 15),
        http_max_retries=_get_int("HTTP_MAX_RETRIES", 3),
        fund_trace_max_hops=_get_int("FUND_TRACE_MAX_HOPS", 4),
        fund_trace_max_paths=_get_int("FUND_TRACE_MAX_PATHS", 500),
        fund_trace_max_edges_explored=_get_int(
            "FUND_TRACE_MAX_EDGES_EXPLORED", 20000
        ),
        behavior_min_fanout_counterparties=_get_int(
            "BEHAVIOR_MIN_FANOUT_COUNTERPARTIES", 4
        ),
        behavior_min_fanin_counterparties=_get_int(
            "BEHAVIOR_MIN_FANIN_COUNTERPARTIES", 4
        ),
        behavior_high_frequency_min_transfers=_get_int(
            "BEHAVIOR_HIGH_FREQUENCY_MIN_TRANSFERS", 5
        ),
        behavior_rapid_hop_max_seconds=_get_int(
            "BEHAVIOR_RAPID_HOP_MAX_SECONDS", 300
        ),
        behavior_burst_window_seconds=_get_int(
            "BEHAVIOR_BURST_WINDOW_SECONDS", 3600
        ),
        behavior_burst_min_transfers=_get_int("BEHAVIOR_BURST_MIN_TRANSFERS", 5),
        behavior_forwarding_window_seconds=_get_int(
            "BEHAVIOR_FORWARDING_WINDOW_SECONDS", 3600
        ),
        behavior_min_forwarding_events=_get_int(
            "BEHAVIOR_MIN_FORWARDING_EVENTS", 2
        ),
        vasp_seed_dataset_path=os.getenv(
            "VASP_SEED_DATASET_PATH", "data/seed/known_vasps.json"
        ),
        vasp_demo_seed_dataset_path=os.getenv(
            "VASP_DEMO_SEED_DATASET_PATH", "data/seed/demo_known_vasps.json"
        ),
        non_vasp_reference_path=os.getenv(
            "NON_VASP_REFERENCE_PATH", "data/seed/non_vasp_reference.json"
        ),
        database_url=os.getenv(
            "DATABASE_URL", "sqlite:///./data/blockchain_intel.db"
        ),
        provider_cache_enabled=_get_bool("PROVIDER_CACHE_ENABLED", True),
        provider_cache_dir=os.getenv("PROVIDER_CACHE_DIR", "data/cache/provider"),
        provider_cache_ttl_seconds=_get_int("PROVIDER_CACHE_TTL_SECONDS", 86_400),
        graph_cache_dir=os.getenv("GRAPH_CACHE_DIR", "data/graphs"),
        transfers_cache_dir=os.getenv("TRANSFERS_CACHE_DIR", "data/fixtures"),
        api_cors_allow_origins=os.getenv(
            "CORS_ALLOW_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ),
        expansion_max_addresses=_get_int("EXPANSION_MAX_ADDRESSES", 40),
        expansion_max_addresses_per_hop=_get_int(
            "EXPANSION_MAX_ADDRESSES_PER_HOP", 30
        ),
        expansion_max_records_per_stream=_get_int(
            "EXPANSION_MAX_RECORDS_PER_STREAM", 1000
        ),
        fund_trace_time_window_days=_get_int("FUND_TRACE_TIME_WINDOW_DAYS", 0),
        targeted_trace_max_paths_per_target=_get_int(
            "TARGETED_TRACE_MAX_PATHS_PER_TARGET", 25
        ),
        targeted_trace_max_edges_explored=_get_int(
            "TARGETED_TRACE_MAX_EDGES_EXPLORED", 400_000
        ),
        path_quality_hub_degree_threshold=_get_int(
            "PATH_QUALITY_HUB_DEGREE_THRESHOLD", 100
        ),
        path_quality_max_hop_gap_seconds=_get_int(
            "PATH_QUALITY_MAX_HOP_GAP_SECONDS", 2_592_000
        ),
        behavior_fast_passthrough_max_seconds=_get_int(
            "BEHAVIOR_FAST_PASSTHROUGH_MAX_SECONDS", 600
        ),
        behavior_counterparty_concentration_min_share=_get_float(
            "BEHAVIOR_COUNTERPARTY_CONCENTRATION_MIN_SHARE", 0.5
        ),
        behavior_concentration_min_transfers=_get_int(
            "BEHAVIOR_CONCENTRATION_MIN_TRANSFERS", 5
        ),
        behavior_min_asset_diversity=_get_int("BEHAVIOR_MIN_ASSET_DIVERSITY", 4),
        behavior_repeated_amount_min_occurrences=_get_int(
            "BEHAVIOR_REPEATED_AMOUNT_MIN_OCCURRENCES", 3
        ),
        behavior_repeated_amount_decimals=_get_int(
            "BEHAVIOR_REPEATED_AMOUNT_DECIMALS", 8
        ),
        behavior_dormancy_min_seconds=_get_int(
            "BEHAVIOR_DORMANCY_MIN_SECONDS", 2_592_000
        ),
        behavior_reactivation_window_seconds=_get_int(
            "BEHAVIOR_REACTIVATION_WINDOW_SECONDS", 86_400
        ),
        behavior_reactivation_min_transfers=_get_int(
            "BEHAVIOR_REACTIVATION_MIN_TRANSFERS", 3
        ),
        behavior_large_value_native_amount=_get_float(
            "BEHAVIOR_LARGE_VALUE_NATIVE_AMOUNT", 100.0
        ),
        behavior_unusual_hour_start_utc=_get_int(
            "BEHAVIOR_UNUSUAL_HOUR_START_UTC", 0
        ),
        behavior_unusual_hour_end_utc=_get_int("BEHAVIOR_UNUSUAL_HOUR_END_UTC", 5),
        behavior_unusual_timing_min_share=_get_float(
            "BEHAVIOR_UNUSUAL_TIMING_MIN_SHARE", 0.5
        ),
        behavior_unusual_timing_min_transfers=_get_int(
            "BEHAVIOR_UNUSUAL_TIMING_MIN_TRANSFERS", 5
        ),
        behavior_in_out_imbalance_min_ratio=_get_float(
            "BEHAVIOR_IN_OUT_IMBALANCE_MIN_RATIO", 0.8
        ),
        behavior_in_out_imbalance_min_transfers=_get_int(
            "BEHAVIOR_IN_OUT_IMBALANCE_MIN_TRANSFERS", 5
        ),
        behavior_activity_density_min_per_day=_get_float(
            "BEHAVIOR_ACTIVITY_DENSITY_MIN_PER_DAY", 20.0
        ),
        behavior_activity_density_min_transfers=_get_int(
            "BEHAVIOR_ACTIVITY_DENSITY_MIN_TRANSFERS", 20
        ),
        ml_artifact_dir=os.getenv("ML_ARTIFACT_DIR", "data/ml"),
        ml_random_seed=_get_int("ML_RANDOM_SEED", 42),
        ml_test_fraction=_get_float("ML_TEST_FRACTION", 0.2),
        ml_validation_fraction=_get_float("ML_VALIDATION_FRACTION", 0.2),
        ml_cv_folds=_get_int("ML_CV_FOLDS", 5),
        ml_min_samples_per_class=_get_int("ML_MIN_SAMPLES_PER_CLASS", 20),
        ml_min_transfers_per_sample=_get_int("ML_MIN_TRANSFERS_PER_SAMPLE", 3),
        risk_weight_direct_vasp_evidence=_get_float(
            "RISK_WEIGHT_DIRECT_VASP_EVIDENCE", 25.0
        ),
        risk_weight_indirect_vasp_evidence=_get_float(
            "RISK_WEIGHT_INDIRECT_VASP_EVIDENCE", 15.0
        ),
        risk_weight_investigative_indicator=_get_float(
            "RISK_WEIGHT_INVESTIGATIVE_INDICATOR", 6.0
        ),
        risk_weight_supporting_indicator=_get_float(
            "RISK_WEIGHT_SUPPORTING_INDICATOR", 3.0
        ),
        risk_weight_contextual_indicator=_get_float(
            "RISK_WEIGHT_CONTEXTUAL_INDICATOR", 1.0
        ),
        risk_weight_incomplete_search=_get_float(
            "RISK_WEIGHT_INCOMPLETE_SEARCH", 5.0
        ),
        risk_band_medium_threshold=_get_float("RISK_BAND_MEDIUM_THRESHOLD", 20.0),
        risk_band_high_threshold=_get_float("RISK_BAND_HIGH_THRESHOLD", 45.0),
    )
