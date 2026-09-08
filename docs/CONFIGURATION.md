# Configuration

Every setting is read by `app/core/config.py::get_settings()` from the
environment (via `.env`, loaded at import). **Nothing in this project has a
threshold hardcoded in its logic** — anything tunable is here.

`.env.example` is the annotated master copy. Start by copying it:

```bash
copy .env.example .env
```

63 settings. Exactly one is required.

## Behaviour of the loader

**`ETHERSCAN_API_KEY` is the only required setting.** Every other variable has a
default that the code uses when it is unset, so a `.env` containing only a key is
a valid configuration.

**Malformed values fail loudly.** `_get_int`, `_get_float`, `_get_bool` and
`_get_optional_int` raise `ConfigurationError` naming the variable. A
`HTTP_TIMEOUT_SECONDS=fifteen` stops the run; it does not silently become the
default. A silent fallback there would mean the operator believes one thing is
configured while another is running — and the report would be built on a setting
nobody chose.

**`get_settings()` is deliberately not cached.** It reads `os.getenv` on every
call, so a test or a long-running process that changes the environment sees the
change. An `lru_cache` here would pin the first process's environment for the
lifetime of the interpreter and make per-test configuration impossible.

**No secret ever leaves the process.** The API key is never written into a cache
key, a log line, a report, a JSON artefact or an API error body. `DATABASE_URL`
may contain credentials and is excluded from all caller-visible output. Never
commit a filled-in `.env`.

---

## Provider credentials

| Variable | Default | Notes |
|---|---|---|
| `ETHERSCAN_API_KEY` | *(empty — required)* | Without it, live acquisition **fails** naming this variable. It never falls back to demo data. The two offline modes do not need it. Free key: <https://etherscan.io/myapikey> |
| `ETHERSCAN_BASE_URL` | `https://api.etherscan.io/v2/api` | V2 unified API: one host for every chain, selected by `chainid`. Verify the current URL and auth scheme at <https://docs.etherscan.io> before relying on this. |
| `ETHERSCAN_CHAIN_ID` | *(unset)* | **Optional override — normally leave unset.** |

### On `ETHERSCAN_CHAIN_ID`

The chain id is derived from the chain **name** by `app/blockchain/chains.py`,
which is the single place that maps a name to an id. That is what guarantees the
id queried and the name printed cannot disagree.

Setting this pins the id independently of the name. A value that **contradicts**
the requested chain is rejected rather than honoured, because both alternatives
are indefensible: honouring the id would query chain 1 while every line of the
report says "polygon", and honouring the name would silently ignore an explicit
setting. One of the two inputs is wrong and the operator has to say which.

Its only real use is as a **deployment assertion**: pin the id you expect, and any
run whose `--chain` resolves to a different one fails loudly instead of quietly
investigating the wrong chain.

---

## Acquisition bounds

| Variable | Default | Notes |
|---|---|---|
| `MAX_TRANSACTIONS_PER_INVESTIGATION` | `2000` | Per-stream ceiling; native, internal and token count separately. Reaching it marks the dataset **INCOMPLETE** rather than silently truncating, so a negative finding on a busy wallet is never reported as proven absence. |
| `DEFAULT_LOOKBACK_DAYS` | `90` | |
| `HTTP_TIMEOUT_SECONDS` | `15` | |
| `HTTP_MAX_RETRIES` | `3` | Covers rate limiting **and** transient transport faults (timeouts, connection resets), with exponential backoff. |

Raising the transaction ceiling is the honest way to handle a truncated
investigation. Leaving it low and reading `NONE` from a truncated search is not —
which is why the code reports `INCONCLUSIVE` instead.

## Recursive multi-hop acquisition

| Variable | Default | Notes |
|---|---|---|
| `EXPANSION_MAX_ADDRESSES` | `40` | Total addresses fetched per investigation, **including** the wallet itself. At roughly three to nine provider requests per address this keeps a full run inside a couple of hundred requests, which the free Etherscan tier serves without tripping its rate limit. |
| `EXPANSION_MAX_ADDRESSES_PER_HOP` | `30` | Per-hop ceiling. Without it a single high-degree counterparty at hop 1 consumes the whole address budget at hop 2 and no deeper hop is ever attempted. |
| `EXPANSION_MAX_RECORDS_PER_STREAM` | `1000` | Per-stream ceiling for an **expanded** address, deliberately lower than `MAX_TRANSACTIONS_PER_INVESTIGATION`, which stays reserved for the investigated wallet's own history. A counterparty is fetched to find onward routes, not to be investigated in its own right. |

`--max-hops N` drives acquisition as well as the search: live acquisition walks
outward from the wallet hop by hop, fetching each newly discovered
counterparty's own streams, so a four-hop search runs against a graph that
actually contains four hops of edges. Ethereum is one connected component, so
these three numbers are all that stands between one wallet and the whole chain.

They are hard ceilings, and **whichever one bites is named in the report** —
`ACQUISITION STOPPED: ADDRESS_BUDGET_REACHED`, alongside `OBSERVED HOP DEPTH`
(the leading hop levels in which *every* discovered address was fetched) and
`REQUESTED HOP DEPTH`. A run that stopped early reports a shallower observed
depth and downgrades a deeper negative to `INCONCLUSIVE`; it never reports the
requested depth as if it had been reached.

## Caches

| Variable | Default | Notes |
|---|---|---|
| `PROVIDER_CACHE_ENABLED` | `true` | |
| `PROVIDER_CACHE_DIR` | `data/cache/provider` | Raw provider responses, keyed deterministically. **The key never contains the API key.** |
| `PROVIDER_CACHE_TTL_SECONDS` | `86400` | |
| `GRAPH_CACHE_DIR` | `data/graphs` | Built graphs (`.gpickle`), reloadable with `--cached-graph`. Read by `investigate.py` **and** all four legacy scripts, so a graph built by one is visible to the others. |

`--no-cache` bypasses reads and writes for one run. `--refresh` bypasses only
reads, so stale entries are *replaced* rather than left behind — a refresh that
also suppressed the write would leave the cache permanently empty.

Tests point `GRAPH_CACHE_DIR` at a temporary directory, so a test run can never
leave a graph behind that a later real investigation might load.

## Known-VASP dataset

| Variable | Default | Notes |
|---|---|---|
| `VASP_SEED_DATASET_PATH` | `data/seed/known_vasps.json` | Production dataset. Every entry is a real, publicly documented address carrying its own provenance. |
| `VASP_DEMO_SEED_DATASET_PATH` | `data/seed/demo_known_vasps.json` | Synthetic. Used **only** by the legacy scripts under an explicit `--demo` flag and by tests. `investigate.py` never loads it. |

Load either through `app/attribution/seed_loader.py`, not raw `json.load` — the
file is an object with `_comment`, `_provenance` and `seed_addresses`, not a bare
list.

Adding a VASP address is a pure JSON edit; no Python change is needed. Addresses
are deliberately **not** added to inflate the count: an unsourced address weakens
every conclusion drawn from the file.

## Persistence (HTTP API only)

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/blockchain_intel.db` | The CLI does not use a database. Point at `postgresql+psycopg2://...` for Postgres; SQLAlchemy picks the driver from the scheme. **May contain credentials, so it is never included in caller-visible output.** |

## Fund-flow tracing

| Variable | Default | Notes |
|---|---|---|
| `FUND_TRACE_MAX_HOPS` | `4` | Raising this grows the search super-linearly. |
| `FUND_TRACE_MAX_PATHS` | `500` | |
| `FUND_TRACE_MAX_EDGES_EXPLORED` | `20000` | |
| `FUND_TRACE_TIME_WINDOW_DAYS` | `0` | `0` = no window. `--time-window` overrides per run. |
| `TARGETED_TRACE_MAX_PATHS_PER_TARGET` | `25` | |
| `TARGETED_TRACE_MAX_EDGES_EXPLORED` | `400000` | |

**Prefer the targeted search to raising the budgets.** Instead of enumerating
everything reachable, the targeted search looks *toward the known VASP addresses
only*, in both directions. That is what makes a 50k-edge graph tractable, and why
the budgets above do not need raising. Exhausting a budget yields
`SEARCH STATUS: INCONCLUSIVE`, never `NONE`.

| Variable | Default | Notes |
|---|---|---|
| `PATH_QUALITY_HUB_DEGREE_THRESHOLD` | `100` | Above this, a hop is a high-throughput intermediary. |
| `PATH_QUALITY_MAX_HOP_GAP_SECONDS` | `2592000` (30d) | |

A route through a 10k-degree exchange hot wallet is a graph path, not evidence of
fund continuity; a 30-day gap between hops is weaker still. Both are reported as
**caveats on the path**, not used to hide it.

## Behavioural indicators

Each threshold is printed next to the value actually observed, so a finding can
always be checked. These are **investigative indicators** — none is evidence of
wrongdoing, and none is labelled criminal.

| Variable | Default |
|---|---|
| `BEHAVIOR_MIN_FANOUT_COUNTERPARTIES` | `4` |
| `BEHAVIOR_MIN_FANIN_COUNTERPARTIES` | `4` |
| `BEHAVIOR_HIGH_FREQUENCY_MIN_TRANSFERS` | `5` |
| `BEHAVIOR_BURST_WINDOW_SECONDS` | `3600` |
| `BEHAVIOR_BURST_MIN_TRANSFERS` | `5` |
| `BEHAVIOR_ACTIVITY_DENSITY_MIN_PER_DAY` | `20.0` |
| `BEHAVIOR_ACTIVITY_DENSITY_MIN_TRANSFERS` | `20` |
| `BEHAVIOR_RAPID_HOP_MAX_SECONDS` | `300` |
| `BEHAVIOR_FAST_PASSTHROUGH_MAX_SECONDS` | `600` |
| `BEHAVIOR_FORWARDING_WINDOW_SECONDS` | `3600` |
| `BEHAVIOR_MIN_FORWARDING_EVENTS` | `2` |
| `BEHAVIOR_COUNTERPARTY_CONCENTRATION_MIN_SHARE` | `0.5` |
| `BEHAVIOR_CONCENTRATION_MIN_TRANSFERS` | `5` |
| `BEHAVIOR_MIN_ASSET_DIVERSITY` | `4` |
| `BEHAVIOR_REPEATED_AMOUNT_MIN_OCCURRENCES` | `3` |
| `BEHAVIOR_REPEATED_AMOUNT_DECIMALS` | `8` |
| `BEHAVIOR_LARGE_VALUE_NATIVE_AMOUNT` | `100.0` |
| `BEHAVIOR_DORMANCY_MIN_SECONDS` | `2592000` (30d) |
| `BEHAVIOR_REACTIVATION_WINDOW_SECONDS` | `86400` |
| `BEHAVIOR_REACTIVATION_MIN_TRANSFERS` | `3` |
| `BEHAVIOR_UNUSUAL_HOUR_START_UTC` | `0` |
| `BEHAVIOR_UNUSUAL_HOUR_END_UTC` | `5` |
| `BEHAVIOR_UNUSUAL_TIMING_MIN_SHARE` | `0.5` |
| `BEHAVIOR_UNUSUAL_TIMING_MIN_TRANSFERS` | `5` |
| `BEHAVIOR_IN_OUT_IMBALANCE_MIN_RATIO` | `0.8` |
| `BEHAVIOR_IN_OUT_IMBALANCE_MIN_TRANSFERS` | `5` |

Two notes on specific defaults:

* `BEHAVIOR_REPEATED_AMOUNT_DECIMALS=8` is deliberately generous. Rounding to 2dp
  would call `1.001` and `1.009` the same amount and manufacture a pattern.
* The unusual-hour window is **UTC**. A wallet is not "nocturnal" without knowing
  where its operator is, so this is reported as a UTC observation, not an
  inference about a person.

## Risk scoring

The score is a sum of named, individually printed components — there is no opaque
"risk = 87" anywhere. Change a weight and the report's arithmetic changes with it,
visibly.

| Variable | Default |
|---|---|
| `RISK_WEIGHT_DIRECT_VASP_EVIDENCE` | `25.0` |
| `RISK_WEIGHT_INDIRECT_VASP_EVIDENCE` | `15.0` |
| `RISK_WEIGHT_INVESTIGATIVE_INDICATOR` | `6.0` |
| `RISK_WEIGHT_SUPPORTING_INDICATOR` | `3.0` |
| `RISK_WEIGHT_CONTEXTUAL_INDICATOR` | `1.0` |
| `RISK_WEIGHT_INCOMPLETE_SEARCH` | `5.0` |
| `RISK_BAND_MEDIUM_THRESHOLD` | `20.0` |
| `RISK_BAND_HIGH_THRESHOLD` | `45.0` |

`RISK_WEIGHT_INCOMPLETE_SEARCH` is added when a search hit a budget, because an
incomplete search is itself a reason to treat a negative finding with caution.

## Machine learning

| Variable | Default | Notes |
|---|---|---|
| `ML_ARTIFACT_DIR` | `data/ml` | Trained artifacts, schema versions, metrics. |
| `ML_RANDOM_SEED` | `42` | Fixed for reproducibility; recorded on every prediction. |
| `ML_TEST_FRACTION` | `0.2` | Untouched test split. |
| `ML_VALIDATION_FRACTION` | `0.2` | |
| `ML_CV_FOLDS` | `5` | Cross-validation on the training portion only. |
| `ML_MIN_SAMPLES_PER_CLASS` | `20` | **The refusal threshold.** |
| `ML_MIN_TRANSFERS_PER_SAMPLE` | `3` | Activity floor, applied **before any label is read** so it cannot admit one class more readily than another. |

`ML_MIN_SAMPLES_PER_CLASS` is the most consequential ML setting here. Below this
many labelled samples **per class**, training is refused and no metric is
reported — the report prints the blockers instead.

Labels are never invented to get past it. In particular an address absent from the
seed set is never labelled `NOT_VASP_OWNED`, so with the current six-address
dataset the supervised VASP task is legitimately untrainable, and the pipeline
falls back to a labelled-data-free unsupervised outlier analysis and says so. Full
detail in [ML.md](ML.md).

**Lowering this threshold to force a supervised result is not a configuration
choice, it is a misreport.** A model trained on 3 positives has no meaningful
held-out performance regardless of what number it prints.
