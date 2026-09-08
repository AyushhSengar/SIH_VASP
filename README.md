# Blockchain Wallet Investigation Engine

SIH problem statement: **Automated Attribution of Unknown Cryptocurrency Wallets
to Nearest VASPs through Blockchain Intelligence APIs.**

Given a wallet address, this backend acquires that wallet's real on-chain
history, builds a transaction graph, searches it in both directions for
addresses belonging to known Virtual Asset Service Providers, measures
behavioural indicators, scores risk from itemised contributions, runs the
strongest machine-learning analysis its real labels honestly support, and prints
a complete evidence-based report to the terminal.

```bash
python investigate.py 0x75c0623bae00749550cf1c1703e7382038b3109a --chain ethereum
```

That one command is the whole product. Everything below documents what it
actually does, and — just as importantly — what it refuses to do.

---

## The rules this codebase is built around

These are not aspirations. They are enforced in code, and the tests that pin
them are named after them.

| Rule | Where it lives |
|---|---|
| **Real data only.** No synthetic transactions, no fabricated VASP addresses, no invented paths. If credentials are missing the run fails and says so; it never substitutes demo data. | `app/investigation/pipeline.py::acquire_live` |
| **Every report states its data mode.** `REAL` (fetched live this run) or `CACHED REAL DATA` (real chain data reloaded from disk). Cached data is never presented as live. | `DataMode`, report header |
| **Address matching is exact and case-insensitive only.** No fuzzy, prefix, substring or "similar-looking" matching exists anywhere. A near-miss on an address is not weak evidence — it is a different address. | `app/attribution/matcher.py`, `app/normalization/transactions.py` |
| **A graph path is not proof of fund continuity.** A→B→C is reported as a *fund-flow candidate* / *transaction path*, and every candidate carries a plausibility grade and its concerns. | `app/tracing/tracer.py`, `app/tracing/quality.py` |
| **Behaviour is never a crime.** Every finding is an `INVESTIGATIVE_INDICATOR` / `SUPPORTING_EVIDENCE` / `CONTEXTUAL` / `REQUIRES_FURTHER_VERIFICATION`, with its metric, threshold, measured value and the transactions behind it. | `app/behavior/models.py` |
| **A third-party label is not ownership.** Dataset entries carry provenance, and a community label is reported as an investigative lead, never as a confirmed disclosure. | `app/attribution/models.py::SeedSourceType` |
| **ML never overrides blockchain evidence,** and never reports a metric the data cannot support. Insufficient real labels produce a documented refusal plus an unsupervised fallback, not a fabricated accuracy. | `app/ml/real_labels.py`, `app/ml/real_training.py` |
| **An incomplete search is `INCONCLUSIVE`, never `NONE`.** Absence of evidence in a search that hit a resource limit is not evidence of absence. | `app/tracing/tracer.py`, `app/analysis/risk.py` |
| **No secrets in output.** No API key, database URL, credential or traceback reaches a report, a JSON file, or an HTTP response body. | `app/api/errors.py`, `app/reporting/terminal.py` |

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
```

Then put an Etherscan API key in `.env`:

```
ETHERSCAN_API_KEY=your_key_here
```

Free key: <https://etherscan.io/myapikey>. Verify the current base URL and auth
scheme at <https://docs.etherscan.io> before relying on the defaults — Etherscan
migrated to a unified V2 endpoint and may change it again.

Python 3.13 is what this was developed and tested on. `xgboost` and `lightgbm`
are optional: the ML stage adds them as candidate models when importable and
carries on without them when not.

---

## The command line

```
python investigate.py <WALLET_ADDRESS> [options]
```

Every flag changes behaviour. There are no cosmetic or placeholder options.

| Flag | Effect |
|---|---|
| `--chain NAME` | Chain to investigate (default `ethereum`). Resolved through the chain registry; an unresolvable name is rejected before any I/O. |
| `--max-hops N` | Hop depth for **both** acquisition and path search. Acquisition expands this many hop levels outward from the wallet before the search runs, so the graph actually holds edges at the depth being searched. Default from `FUND_TRACE_MAX_HOPS`. |
| `--max-paths N` | Maximum paths retained per target. Default from `FUND_TRACE_MAX_PATHS`. |
| `--time-window DAYS` | Only consider transfers within this many days of the wallet's most recent activity. `0` disables the window. |
| `--refresh` | Re-query the provider instead of reading cached responses. The fresh responses still *replace* the cached ones. |
| `--no-cache` | Do not read **or** write the provider response cache at all. |
| `--cached-graph PATH` | Analyse a real graph saved by an earlier run. Output is labelled `CACHED REAL DATA`. |
| `--transfers-file PATH` | Rebuild the graph from a normalized real transfer file. Also `CACHED REAL DATA`, but unlike `--cached-graph` this preserves per-record provenance, so the supervised labelling rules can run. |
| `--json [PATH]` | With no PATH, the complete machine-readable report is the whole of stdout (pipeable). With a PATH, it is written there and the human report still prints. |
| `--ml` / `--no-ml` | Run or skip the machine-learning stage. `--ml` is the default; the two are mutually exclusive. |
| `--verbose` | Stage-by-stage progress to **stderr**, so `--json` on stdout stays parseable. Also selects the full nine-section report. |
| `--full-report` | Print the complete nine-section report — all evidence, methodology, ML explanation and limitations — instead of the compact default. |

`--refresh` and `--no-cache` are deliberately different, and a test asserts they
are not secretly the same. `--refresh` suppresses only the cache *read*: a
refresh that suppressed the write too would leave the cache permanently empty
and make every later run pay full price for nothing.

Exit codes: `0` completed · `1` could not run (bad address, unresolvable chain,
missing credentials, no data, malformed configuration) · `2` interrupted.

### Data modes

| Invocation | Mode |
|---|---|
| default | `REAL` — live provider fetch in this run |
| `--cached-graph PATH` | `CACHED REAL DATA` |
| `--transfers-file PATH` | `CACHED REAL DATA` |

There is no demo mode and no synthetic fallback. The two cached modes exist
because they are the only way to run the pipeline offline against real data, and
they always label themselves so a reader is never shown stale data as live.

---

## The report

There are two terminal reports over one `InvestigationReport`. Choosing between
them changes what is **displayed** and nothing that was **measured** — both
renderers read the same model, compute nothing, and cannot disagree with each
other or with `--json`.

### Compact report — the default

Ten blocks of facts: numbers, addresses, timestamps and tables. No paragraphs,
no methodology, no evidence taxonomy, no limitations digest.

| # | Block | Contents |
|---|---|---|
| 1 | `WALLET` | Address, chain, data source, analysis duration. |
| 2 | `TRANSACTION SUMMARY` | Total transfers, incoming, outgoing, first and last transfer, active period, active days. |
| 3 | `COUNTERPARTIES` | Total unique, then per address: transfer count, incoming, outgoing, first seen, last seen. |
| 4 | `TRANSACTION ACTIVITY` | Every transfer touching the wallet, oldest first: timestamp, direction, amount, asset, full from/to addresses, full tx hash. |
| 5 | `TIMING ANALYSIS` | Turnarounds measured, fastest / median / longest inbound→outbound turnaround, fast pass-through count, inbound not forwarded, first/last activity, active days, median gap, longest idle. |
| 6 | `ASSET SUMMARY` | Per asset: transfer count, incoming, outgoing, total in, total out, net flow. |
| 7 | `INVESTIGATION FINDINGS` | Per finding: type, observed metric, value, threshold, classification, time window, related addresses. |
| 8 | `VASP / ENTITY MATCH` | Match status, and per candidate the entity, exact matched address, direction and provenance. |
| 9 | `RISK` | Score, band, band thresholds, data completeness, one ML line, then the triggered indicators with value, weight, contribution and evidence class. |
| 10 | `DATA STATUS` | Data mode, provider, records fetched / retained / rejected / duplicates, graph edges, provider-cache counters, observed **and** requested hop depth, data completeness. |

Two rules govern what it prints. A value the analysis did not produce prints
`N/A`, never `0` — "not measured" and "measured, and the answer was none" are
different findings. And when ML produced no result the block states the
approach and the reason in two lines and stops:

```
  ML:                    UNAVAILABLE
  REASON:                Only 4 address(es) in this graph have at least 3 transfers, ...
```

### Full report — `--full-report` or `--verbose`

Nine sections plus a header and an `INVESTIGATION COMPLETE` footer.

| § | Section | Contents |
|---|---|---|
| 1 | `BLOCKCHAIN DATA SUMMARY` | Streams acquired, normalization accounting (in / kept / rejected / duplicates / missing metadata, and whether the totals reconcile), graph shape. |
| 2 | `FUND-FLOW ANALYSIS (TRANSACTION PATHS)` | Candidate paths as ASCII routes with hop count, tx hashes, timestamps, assets and amounts; plausibility grade and concerns per candidate; search accounting. |
| 3 | `VASP ATTRIBUTION` | Dataset used, entry count, provenance breakdown, then each candidate with its exact matched address, direction, and the full provenance of the dataset entry. |
| 4 | `BIDIRECTIONAL ANALYSIS` | Each operator's wallet→VASP and VASP→wallet evidence, stated separately, and a count of operators with evidence in *both* directions. |
| 5 | `BEHAVIORAL INTELLIGENCE` | Each indicator with its observed metric, threshold, measured value, evidence bullets, related addresses and relevant transactions. |
| 6 | `TEMPORAL AND AMOUNT ANALYSIS` | Activity over time, per-asset amount statistics, timing distributions. |
| 7 | `MACHINE LEARNING ANALYSIS` | Approach actually taken (`SUPERVISED` / `UNSUPERVISED` / `UNAVAILABLE` / `DISABLED`), label census, model and dataset versions, honest metrics or an explicit refusal. |
| 8 | `EVIDENCE SUMMARY AND RISK ANALYSIS` | Every risk contribution itemised with weight, threshold, evidence class and reason; the arithmetic reconciles to the reported score line by line. |
| 9 | `FINAL INVESTIGATION CONCLUSION` | What was and was not established, data completeness, and limitations. |

### Terminal output constraints

Both renderers share the same discipline. Output is **pure ASCII**. Windows
consoles default to cp1252, where a typographic dash becomes mojibake or a
`UnicodeEncodeError`, so the renderer folds typography to ASCII and forces UTF-8
on the stream where the platform allows it.

Long lists are **capped and repeats are collapsed**, because an unbounded report
is unusable: a real wallet produced 67 behavioural findings of which 60 were the
same indicator measured against a different counterparty. The renderer itemises
the first finding of each distinct kind in full and lists further findings of an
already-itemised kind on one line each — carrying the address and the measured
value, which are the only fields that differ. Nothing is silently dropped:

* the census always states how many findings exist and how many are itemised;
* the risk arithmetic still reconciles, with anything past the per-indicator
  limit shown as one explicit subtotal;
* **everything withheld from the terminal is present in full in `--json`.**

Addresses in compact rows are printed in full, never shortened — a truncated
address cannot be re-checked on-chain.

---

## Architecture

```
investigate.py                  thin CLI: argv -> pipeline -> renderer
app/
  investigation/pipeline.py     THE orchestration path. Returns InvestigationReport. Prints nothing.
  reporting/terminal.py         Rendering only. Computes nothing, decides nothing.
  reporting/compact.py          The default ten-block report. Same rule: renders, never computes.
  blockchain/
    base.py                     Provider interface + error taxonomy
    chains.py                   Chain registry: the one place a chain NAME becomes a chain id
    etherscan.py                Etherscan V2 provider (retries, rate limits, malformed bodies)
    ingest.py                   Paginated multi-stream acquisition with per-stream completeness
    cache.py                    Deterministic response cache, never keyed on the API key
  normalization/transactions.py Raw provider dicts -> NormalizedTransfer, plus an explicit validation pass
  graph/builder.py              NetworkX MultiDiGraph construction, save/load, structural summary
  tracing/
    tracer.py                   Bounded, deterministic traversal
    targeted.py                 Destination-aware and reverse search
    quality.py                  Route plausibility grading
  attribution/
    matcher.py                  Exact case-insensitive address index
    bidirectional.py            Independent outbound and inbound searches
    entities.py                 Counterparty identification and operator-level grouping
    seed_loader.py              Provenance-aware dataset loading
  behavior/detectors.py         15 threshold-based indicators, each fully explained
  analysis/
    temporal.py                 Timing and amount analysis
    risk.py                     Transparent, itemised risk scoring
  ml/
    real_labels.py              Where ground truth comes from, and when to refuse
    real_features.py            Leakage-checked feature extraction
    real_training.py            Model selection, grouped splits, cross-validation
    real_predictor.py           Versioned prediction + explanation
    unsupervised.py             IsolationForest fallback with rank-stability reporting
  api/                          HTTP surface (see the disclosure below)
  db/                           SQLAlchemy persistence for the HTTP surface
  core/config.py                Every setting, read from the environment, validated loudly
```

`app/investigation/pipeline.py` returns a fully-populated `InvestigationReport`
and prints nothing; the two renderers in `app/reporting/` print and decide
nothing. That separation is what makes the backend independent of any frontend
and lets the whole pipeline be asserted on in tests without capturing stdout.

Depth, per topic, in `docs/`:

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the pipeline stage by stage
* [`docs/EVIDENCE.md`](docs/EVIDENCE.md) — what each evidence class means and does not mean
* [`docs/ML.md`](docs/ML.md) — labels, splitting, metrics, and the refusal path
* [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) — every environment variable
* [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — running, verifying, troubleshooting

---

## Chain support

`app/blockchain/chains.py` is the single place that maps a chain name to a chain
id. It exists because three things previously disagreed about what a chain name
meant: the provider treated it as a label and sent whatever `ETHERSCAN_CHAIN_ID`
said, the normalizer hardcoded `"ETH" if chain == "ethereum" else None`, and the
HTTP service kept its own list. The result was that `--chain dogecoin` returned
real *Ethereum* transactions with "dogecoin" printed in the header, on every
transfer, and beside every attribution.

Resolvable names: `ethereum` (1), `polygon` (137), `bsc` (56), `base` (8453),
`arbitrum` (42161), `optimism` (10), `avalanche` (43114).

**Only `ethereum` has been exercised against live provider data in this build.**
The others are listed because their chain ids and native symbols are matters of
public record, not because this project has validated them; `CHAIN_VALIDATED_LIVE`
records the distinction so the report and the `--help` text can be honest about
it. An unresolvable name is rejected in all three acquisition modes, before any
I/O — including the cached ones, which never construct a provider.

`ETHERSCAN_CHAIN_ID` is an **optional** override, normally unset. Setting it to a
value that contradicts the requested chain is a hard error, not a preference
honoured silently: one of the two inputs is wrong and the operator has to say
which.

---

## VASP dataset and provenance

`data/seed/known_vasps.json` — the production dataset. Load it through
`app/attribution/seed_loader.py`, never as raw JSON (the file is an object with
`_comment`, `_provenance` and `seed_addresses`, not a bare list).

Every entry carries a `source_type` from a provenance ladder:

`official_disclosure` › `directly_verified` › `public_label` ›
`third_party_label` › `community_label` › `inferred` › `unverified`

The distinction is load-bearing. Only `official_disclosure` and
`directly_verified` are first-party enough to become an ML training label; a
community label is an investigative lead and is reported as one, with its source
URL, so a reader can check it rather than trust it.

**The dataset is deliberately small — six addresses.** It has not been padded to
inflate a count. A larger dataset is a data-acquisition problem, not a code
problem, and inventing addresses to fill it would poison every attribution the
system produces.

`data/seed/demo_known_vasps.json` holds synthetic addresses used only to prove
the pipeline works end to end. The production CLI **cannot** load it: entries
marked `synthetic_demo` are rejected on the production path, and only the legacy
`attribute_wallet.py --demo` / the HTTP service's `use_demo_seed` flag reach it.

---

## Machine learning

The production engine is `app/ml/real_*.py`. It derives labels from two sources
only:

1. **`PROTOCOL_GUARANTEED`** — facts about how EVM chains work. An address that
   *sent* a top-level transaction signed it, so it is an externally-owned
   account. An address that sent an internal transfer moved value from inside
   contract execution, so it is a contract. An ERC-20 transfer's
   `asset_identifier` is the contract that emitted the event.
2. **`DATASET_PROVENANCE`** — an address the dataset records with first-party
   provenance. Third-party labels are excluded: training on them teaches the
   model to reproduce the annotator, not a fact about the chain.

**Absence from the seed set is explicitly not a label.** A wallet missing from
six curated addresses is overwhelmingly likely to be unlabelled, not known-not-a-VASP.
Labelling those `NOT_VASP_OWNED` would manufacture a majority class out of
ignorance and produce a headline accuracy that measures nothing.

When labels are insufficient, `LabelingOutcome` says so with exact counts and
names what is missing, and the pipeline falls back to an unsupervised
`IsolationForest` with bootstrap rank stability — reported as rank stability, not
as accuracy, because there is no ground truth to be accurate against.

Candidate supervised models: `LogisticRegression`, `RandomForestClassifier`,
`GradientBoostingClassifier`, `HistGradientBoostingClassifier`, plus
`XGBClassifier` and `LGBMClassifier` when installed. Classical tabular models
only — no deep learning, because nothing about this feature set justifies it.

Splitting is **wallet/entity-grouped** so the same address cannot appear on both
sides of a split, with a train/validation/untouched-test division, stratification
and cross-validation. Every production prediction carries the model name, model
version, dataset version, feature schema version, training timestamp, sample and
per-class counts, metrics and random seed.

Reported metrics are ROC-AUC, PR-AUC, a confusion matrix and the class
distribution. **The number reported is the number measured.** If genuine held-out
performance is 76%, the report says 76%; if it is 68%, the report says 68% and
explains why. See [`docs/ML.md`](docs/ML.md) for the full contract and for what
the current dataset does and does not support.

---

## JSON output

`--json` emits the complete report as machine-readable JSON: the same object the
renderer consumed, with nothing summarised away. This is what makes the
terminal's display caps honest — every counterparty, indicator, risk component
and per-asset statistic withheld from the terminal is present here in full.

The JSON is UTF-8 and contains no credentials: no API key, no database URL, no
password, no connection string. Its `environment` block carries only the Python
version and platform.

The backend does not depend on a frontend. `--json` and the HTTP API are both
built on the same `InvestigationReport`.

---

## HTTP API

```bash
python -m uvicorn app.api.main:app --reload
# http://127.0.0.1:8000/docs
```

```
POST /investigations                     -> 201, investigation_id + attribution/ML summaries
GET  /investigations/{id}                -> summary
GET  /investigations/{id}/attribution    -> the complete AttributionResult
GET  /investigations/{id}/ml             -> the complete MLPrediction
GET  /health                             -> {"status": "ok", "database": "connected"} or 503
```

`422` invalid wallet / unsupported chain · `404` not found · `400`
`GRAPH_NOT_FOUND` · `503` database unavailable · `500` internal failure. Every
error body is a fixed `{"error", "detail"}` shape — never a traceback, never a
credential.

Persistence is three tables linked by `investigation_id`. The attribution and ML
rows store the complete pydantic objects as JSON plus a few denormalized scalars
for filtering, so nothing can be silently dropped by a hand-maintained mapping.
`DATABASE_URL` selects the backend (SQLite by default; point it at
`postgresql+psycopg2://` for Postgres).

### Disclosed gap: the HTTP surface is behind the CLI

`app/investigation/service.py` still calls the **unidirectional**
`generate_candidates` and the **Milestone-5 synthetic-demo classifier**. The
production bidirectional attribution and real-data ML engine are used by
`investigate.py` only.

This is a known, deliberate scope boundary rather than an oversight. The demo
classifier is type-locked — `MLPrediction.training_data_type` is
`Literal["SYNTHETIC_DEMO"]` — and ships a per-prediction disclaimer, so no
caller can mistake it for a real-data result. **Do not use the HTTP `/ml`
endpoint as evidence of anything.** The CLI is the production path.

---

## Legacy milestone CLIs

`build_graph.py`, `trace_funds.py`, `attribute_wallet.py` and
`ml_attribution.py` are the original per-milestone scripts. They still work and
are kept because they exercise individual stages in isolation, which is useful
when debugging one. They read `GRAPH_CACHE_DIR` like everything else, so a graph
built by one is visible to the others and to `investigate.py`.

They are **not** the production path. `ml_attribution.py` in particular runs the
synthetic demo classifier, and `attribute_wallet.py --demo` loads the synthetic
seed set. Use `investigate.py`.

---

## Tests

```bash
python -m pytest -q
```

**550 tests, all passing, fully offline** — no API key, no network. The suite
covers normalization and validation, graph construction, recursive multi-hop
acquisition, tracing and targeted
search, plausibility grading, bidirectional attribution, entity resolution,
behavioural detectors, temporal and risk analysis, the real ML pipeline, the
chain registry, configuration strictness, both renderers, the CLI, the service,
the repository and the API.

Synthetic and demo data exist **only** in tests and fixtures. Nothing on the
production execution path imports from `tests/`, `conftest`, or
`data/fixtures/` — except via the explicit `--transfers-file` flag, which
labels its output `CACHED REAL DATA` and only accepts real normalized records.

The tests are written to fail when the implementation is wrong, not to be
adjusted when it is. Assertions state the property and, where the property was
learned from a real defect, name the defect.

---

## Known limitations

Stated plainly, because a limitation a demonstrator discovers on stage is worse
than one they read here first.

**Data acquisition**

* Live acquisition **has been exercised end to end against Etherscan** with a key
  present in `.env`: `0x75c0623bae00749550cf1c1703e7382038b3109a` on `ethereum`
  with `--max-hops 4` fetched 91,614 real records from 40 addresses in 120
  provider requests, producing a 7,674-node / 89,259-edge graph stamped
  `DATA MODE: REAL`. Without a key the CLI fails with a
  message naming `ETHERSCAN_API_KEY`; it never substitutes demo data.
* **Acquisition expands recursively, but its horizon is bounded by a budget, not
  by the chain.** It fetches the wallet's own transactions, extracts the
  counterparties, fetches those, and repeats to `--max-hops`. Ethereum is one
  connected component, so `EXPANSION_MAX_ADDRESSES` (40) and
  `EXPANSION_MAX_ADDRESSES_PER_HOP` (30) are hard ceilings, and on a busy wallet
  one of them bites before the requested depth is reached. The report then states
  the depth actually **observed** — the leading hop levels in which every
  discovered address was fetched — separately from the depth requested, names the
  budget that stopped it, and downgrades a deeper negative from `NONE` to
  `INCONCLUSIVE`. On the wallet above the observed depth was 2 of a requested 4.
* **An expanded counterparty is read from its oldest transactions forward.**
  `EXPANSION_MAX_RECORDS_PER_STREAM` (1,000) is applied while walking blocks
  ascending, so for a very busy counterparty the acquired slice is its *earliest*
  history, which may predate the investigated wallet entirely. Those records are
  real and the edges are real, but they are unlikely to carry a chronologically
  consistent route onward from the wallet. Windowing expanded fetches around the
  wallet's own activity is the obvious next improvement and is **not**
  implemented.
* Only `ethereum` has been exercised against live data. The other six chains are
  correct-by-public-record mappings, not validated integrations.
* No USD valuation — the free Etherscan tier does not provide historical pricing,
  and `usd_value` is `None` rather than estimated.
* Contract-creation transfers have no destination, so they produce no edge. The
  source address is still recorded as a node and the count of skipped events is
  reported explicitly.
* Graphs written by an older builder lack `block_number`, `token_contract`,
  `transfer_source` and `gas_used` on their edges. Loading one via
  `--cached-graph` therefore shows those fields as absent rather than fabricated.
  `--transfers-file` and live acquisition carry the full metadata.

**Attribution**

* Six dataset entries: five `community_label`, one `official_disclosure`. That is
  an illustrative sample, not a VASP database. A negative result on this dataset
  means "not connected to these six addresses", nothing more.
* Exact address matching only. No proxy-contract resolution, no cross-chain
  identity resolution, no clustering-based ownership inference.

**Machine learning**

* On the largest real dataset available here, the supervised path **refuses to
  train**, and correctly: `account_type` had 7 `CONTRACT` and 57
  `EXTERNALLY_OWNED_ACCOUNT` samples against a 20-per-class minimum, and
  `vasp_ownership` had 1 `VASP_OWNED` against 0 `NOT_VASP_OWNED` — a negative
  class that is empty by design, because absence from the seed set is not a
  label. The report states these counts and falls back to the unsupervised
  model. **No accuracy figure is reported, because none can honestly be.**
* The `CONTRACT` shortfall is partly an artefact of the offline fixture: 1969 of
  its records claim a native stream while describing a token transfer (they
  predate `transfer_source`), so they are excluded from stream-based labelling.
  A live crawl carries intact stream provenance and would yield more contract
  labels. 5 `community_label` addresses are excluded from training on provenance
  grounds.
* The unsupervised fallback reports rank stability (mean Spearman 0.9871, minimum
  0.9822 across 10 bootstraps), not accuracy, precision, recall or F1. Those
  metrics require ground truth that does not exist for this task.
* Reaching a genuine 75–80% supervised result is a data-acquisition problem: a
  live crawl for `account_type`, more first-party-provenance entries for
  `vasp_ownership`.

**Scope**

* The HTTP API is behind the CLI, as disclosed above.
* No authentication or authorization layer.
* No `GET /investigations` list endpoint, so no pagination.
* Frontend is out of scope. The backend is frontend-independent by construction.
#   S I H _ P r o j e c t  
 