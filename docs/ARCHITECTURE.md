# Architecture

How one wallet address becomes one report. Read
[the README](../README.md) first for what the system is; this file is what it
does, stage by stage, and why each stage is shaped the way it is.

## The two-layer rule

```
investigate.py                 argv -> pipeline -> renderer.  No analysis.
app/investigation/pipeline.py  ALL orchestration. Returns InvestigationReport. Prints NOTHING.
app/reporting/compact.py       Default ten-block report. Computes nothing, decides nothing.
app/reporting/terminal.py      Full nine-section report. Computes nothing, decides nothing.
```

This is the single most important structural decision in the codebase, and every
other property depends on it:

* **The backend does not need a frontend.** `--json` and the HTTP API render the
  same `InvestigationReport` the terminal does. A future UI is another renderer,
  not a rewrite.
* **Every stage is assertable without capturing stdout.** Tests call
  `run_investigation(...)` and inspect a pydantic object. A test that has to
  parse printed text is a test that will pass while the analysis is wrong.
* **The renderer cannot invent a finding.** If a number appears in the terminal
  it exists as a field on the report. When the renderer needs to summarise, it
  summarises *from* fields, states that it did, and leaves the full data in the
  JSON.

`run_investigation` takes an already-built graph rather than fetching one. That
is deliberate: acquisition is where `REAL` and `CACHED REAL DATA` differ, and the
difference has to be explicit at the call site instead of buried inside the
analysis.

## Stage 1 — Validation

`validate_wallet_address` and `validate_chain_name`, both in the pipeline, both
running before any I/O in **all three** acquisition modes.

The chain check is in the pipeline rather than only in the provider because the
two cached modes never construct a provider. Provider-side resolution alone let
`--cached-graph real.gpickle --chain dogecoin` render a full report whose header,
every transfer and every attribution line said "dogecoin" about Ethereum data.
A chain name is a claim about where evidence came from, so it is checked
everywhere, and the *canonical* name is used from then on — the report never
echoes the operator's capitalisation back as if it were the chain's identity.

## Stage 2 — Acquisition

`app/blockchain/etherscan.py` + `app/blockchain/ingest.py`.

Three streams per wallet, fetched independently:

| Stream | Etherscan action | What it is |
|---|---|---|
| native | `txlist` | top-level transactions the wallet signed or received |
| internal | `txlistinternal` | value moved from inside contract execution |
| token | `tokentx` | ERC-20 `Transfer` events |

All three are needed. A wallet whose funds arrive through a router contract has
an empty `txlist` for that flow, so a native-only investigation reports no
inbound activity at all — and would then call a VASP attribution `NONE` when the
deposit was sitting in the internal stream the whole time.

**Per-stream completeness is tracked separately.** `FetchOutcome` records whether
each stream was exhausted or stopped at its record ceiling
(`MAX_TRANSACTIONS_PER_INVESTIGATION` for the investigated wallet,
`EXPANSION_MAX_RECORDS_PER_STREAM` for an expanded counterparty), and
a truncated stream marks the dataset incomplete. That flag is what later turns a
"no path found" into `INCONCLUSIVE` instead of `NONE`.

Reliability, all in the provider:

* real cursor pagination, not a fixed page count
* rate-limit and transient-transport retries with exponential backoff
  (`tenacity`), bounded by `HTTP_MAX_RETRIES`
* explicit timeouts (`HTTP_TIMEOUT_SECONDS`)
* malformed bodies, `NOTOK` envelopes and non-list results raise a typed
  provider error rather than being coerced into an empty result — an empty
  result and a broken provider must never look the same
* duplicate suppression by `_record_identity`, since overlapping pages legitimately
  return the same record twice
* deterministic cache keys that **never contain the API key**

The provider interface is `app/blockchain/base.py`. Adding a second provider or a
non-EVM chain means implementing that interface; nothing downstream of
acquisition knows what a provider is.

### Recursive expansion

`app/investigation/acquisition.py`.

`ingest.py` acquires **one** address. `acquisition.py` is the layer above it that
decides *which* addresses, and it is what makes `--max-hops` mean something to
the data rather than only to the traversal. It fetches the wallet's three
streams, extracts the counterparties from the normalized transfers, fetches
those, and repeats — breadth-first, to the requested depth.

The rules that keep it honest and bounded:

* **Deduplicated by address.** A set of already-fetched addresses is checked
  before every fetch, so a counterparty named by ten peers is fetched once and a
  cycle terminates instead of looping.
* **Deterministic frontier.** Each hop's candidates are ordered by how many
  value-bearing edges reached them, then by address, so two runs over the same
  data fetch the same addresses in the same order.
* **Value-bearing edges only open a hop.** A zero-value or failed transfer is
  kept as an edge but does not justify spending budget expanding its
  counterparty — it moves nothing that could continue onward.
* **Known-VASP addresses are endpoints, not routes.** A seed address is recorded
  with the hop distance at which it was reached and is never expanded: the
  question is whether funds reached one, and reading an exchange hot wallet's
  history would spend the entire budget on unrelated customers.
* **Three hard ceilings** (`EXPANSION_MAX_ADDRESSES`,
  `EXPANSION_MAX_ADDRESSES_PER_HOP`, `EXPANSION_MAX_RECORDS_PER_STREAM`), because
  Ethereum is one connected component and an unbounded walk is the whole chain.
* **A counterparty that cannot be read does not abort the run.** It is recorded
  as unreadable on its hop; only the investigated wallet's own failure is fatal,
  because that leaves nothing to analyse.

What it reports is as important as what it fetches. `observation_depth` is
**measured, not requested**: the number of leading hop levels in which every
discovered address was actually fetched. One address deferred by a budget at
hop 2 holds the observed depth at 2 however deep the walk went, because a route
through that address would be *absent from the data* rather than ruled out. The
stop reason (`DEPTH_REACHED`, `NO_NEW_COUNTERPARTIES`, `ADDRESS_BUDGET_REACHED`),
the addresses fetched versus discovered, and the deepest hop reached all travel
on the provenance into the report. Live acquisition also stamps the measured
radius onto the graph it saves, so a later `--cached-graph` run reads back the
depth that run achieved instead of inferring a deeper one from the graph's shape.

## Stage 3 — Normalization

`app/normalization/transactions.py`. Provider dicts become `NormalizedTransfer`
records: one shape for all three streams, with `asset_type`, `transfer_type`,
`asset_symbol`, `asset_identifier`, decimals applied, `chain`, and
`transfer_source` recording which stream it came from.

Normalization and **validation are separate passes**, and that separation is
load-bearing. Normalization normalizes everything; `validate_transfers` then
rejects records with a stated reason, producing a `NormalizationReport` that
accounts for every input:

```
in -> kept + rejected + duplicates,  and RECONCILED: YES/NO
```

If those totals do not reconcile, the report says so rather than quietly
reporting a smaller number. That is the whole point: **evidence is never silently
discarded.** A rejected record has a reason attached and is counted, so a reader
can tell "this wallet has 3969 transfers" from "this wallet has 4000 transfers of
which 31 were duplicates".

The native asset symbol comes from the chain registry, not from
`"ETH" if chain == "ethereum" else None`, which is what it used to be — and which
gave every non-Ethereum native transfer no asset symbol at all.

## Stage 4 — Graph construction

`app/graph/builder.py`. A `networkx.MultiDiGraph`.

* **Directed**, because direction is the evidence. "This wallet sent to Binance"
  and "Binance sent to this wallet" are different claims with different
  investigative meaning, and an undirected graph cannot tell them apart.
* **Multi**, because two wallets transact repeatedly and each transaction is
  separate evidence. Collapsing them to one weighted edge destroys the tx hashes.
  Edge keys are `"<tx_hash>#<occurrence>"` — the occurrence suffix exists because
  one transaction can legitimately emit several transfers between the same pair.
* **Every edge carries** `tx_hash`, `block_number`, `timestamp`, `from`, `to`,
  `amount`, `asset`, `token_contract`, `asset_type`, `transfer_type`, `chain`,
  `transfer_source` and gas metadata. Anything reported later can be traced back
  to a transaction a reader can look up.
* **Deterministic.** Same transfers in, same graph out, same iteration order.

Contract-creation transfers have no destination and therefore produce no edge.
The source is still recorded as a node and the skipped count is reported
explicitly, because a silently-dropped record is indistinguishable from one that
never existed.

## Stage 5 — Tracing

`app/tracing/tracer.py`, `app/tracing/targeted.py`, `app/tracing/quality.py`.

The naive approach — enumerate every path up to N hops — explodes. The wrong fix
is raising `MAX_PATHS` and `MAX_EDGES_EXPLORED` until it stops complaining; that
converts a correctness problem into a slow correctness problem.

What is actually done:

* **Targeted, destination-aware search.** Search *toward the known VASP
  addresses*, not outward into everything reachable. The set of interesting
  destinations is small and known in advance, which is the entire reason the
  budgets do not need raising.
* **Reverse search** from the VASP side for inbound evidence, run independently
  of the forward search.
* **Pruning and early termination** on nodes that cannot reach a target within
  the remaining hop budget.
* **Time windows** (`--time-window`) to bound the search to the relevant period.
* **Explicit budgets** with accounting: paths found, edges explored, whether a
  budget was exhausted.
* **Deterministic traversal order,** so two runs on one graph agree.

Every candidate is graded by `app/tracing/quality.py`:

`DIRECT_TRANSFER` › `PLAUSIBLE` › `WEAK` › `IMPLAUSIBLE`

with concerns named: `ASSET_CHANGED`, `LONG_TIME_GAP`,
`HIGH_THROUGHPUT_INTERMEDIARY`, `AMOUNT_INCREASED`, `UNVERIFIABLE_CHRONOLOGY`.
A route through a 10,000-degree exchange hot wallet is a graph path, not evidence
that these particular funds continued through it. Grades are reported *alongside*
the path, never used to hide it.

## Stage 6 — Attribution

`app/attribution/`.

Matching is `matcher.py`: an exact, case-insensitive index. There is no fuzzy,
prefix, substring or "similar-looking" matching anywhere in the codebase, because
a near-miss on a 20-byte address is not weak evidence of the same address — it is
a different address, and treating it as a lead is how an innocent party gets
attributed to an exchange they never touched.

`bidirectional.py` runs **two independent searches**:

* `wallet -> VASP` — the wallet sent to a VASP-controlled address (a deposit)
* `VASP -> wallet` — a VASP-controlled address sent to the wallet (a withdrawal)

Both are reported separately, and either alone is meaningful. A withdrawal from
an exchange to this wallet identifies the exchange as a lead even when the wallet
never deposited back — an earlier unidirectional design missed exactly that case,
which is the most common one for a receiving wallet.

Directions: `DIRECT_OUTBOUND`, `INDIRECT_OUTBOUND`, `DIRECT_INBOUND`,
`INDIRECT_INBOUND`, `BIDIRECTIONAL`. `BIDIRECTIONAL` requires evidence in *both*
directions and is counted separately in the report.

**What never produces an attribution:**

* membership in the same connected component
* undirected connectivity
* behavioural similarity
* an ML prediction

Those are recorded as `UndirectedRelation` or `ConnectedButNoValidPath` — visible
findings that explicitly are *not* attributions. Address-level evidence comes
first; behaviour and ML may only **support** a conclusion address evidence
already reached.

`entities.py` groups addresses under an operator while always printing the exact
matched address. Common ownership is never inferred from behaviour: two addresses
belong to one operator here only when the dataset says so.

## Stage 7 — Behaviour

`app/behavior/detectors.py`. Fifteen indicators (`PatternType`), each threshold
configurable and each finding carrying:

indicator name · observed metric · threshold · actual value · evidence bullets ·
related addresses · relevant transaction hashes

Classified as `INVESTIGATIVE_INDICATOR` / `SUPPORTING_EVIDENCE` / `CONTEXTUAL` /
`REQUIRES_FURTHER_VERIFICATION`. **Nothing is labelled criminal.** High-frequency
transfers describe a bot, an arbitrageur, a market maker, a payment processor and
a mixer equally well; the report says what was measured and leaves the inference
to the investigator, who has context the chain does not contain.

## Stage 8 — Temporal, amount and risk

`app/analysis/temporal.py` — activity over time, per-asset amount statistics,
hour and weekday distributions. Hours are stated as UTC observations, not as
inferences about a person's schedule.

`app/analysis/risk.py` — a sum of named `RiskComponent`s, each with its weight,
threshold, evidence class and reason. The printed arithmetic reconciles to the
score line by line. There is no opaque "Risk = 87" anywhere; change a weight in
`.env` and the report's arithmetic visibly changes with it.

`SkippedComponent` records contributions that were *considered and not applied*,
with the reason — so a low score is as auditable as a high one.

An incomplete search adds `RISK_WEIGHT_INCOMPLETE_SEARCH`, because not knowing is
itself a reason for caution.

## Stage 9 — ML

`app/ml/real_*.py`, and only when labels honestly support it. Full treatment in
[ML.md](ML.md). The two rules that matter architecturally: ML output can never
override direct blockchain evidence, and a refusal to train is a reported result,
not a failure to route around.

## Rendering

Two renderers over one model. `app/reporting/compact.py` prints the default
ten-block report; `app/reporting/terminal.py` prints the nine-section report
selected by `--full-report` or `--verbose`. Neither computes anything, so the
two can never disagree with each other or with `--json`. See the README for
both block tables and the display-cap policy.

The terminal is **ASCII-only** — Windows consoles default to cp1252 and a
typographic dash there is mojibake or a `UnicodeEncodeError`, so
`configure_stdout()` forces UTF-8 where the platform allows and `to_ascii()`
folds typography regardless. The `--json` artefact is UTF-8 and must be read with
`encoding="utf-8"`; Python on Windows defaults to cp1252 and will otherwise fail
on the first non-ASCII byte.

## Where to add things

| Adding | Touch | Do not touch |
|---|---|---|
| a chain | `app/blockchain/chains.py` | anything else — it is the single name→id authority |
| a provider | implement `app/blockchain/base.py` | normalization onward |
| a VASP address | `data/seed/known_vasps.json` (with provenance) | any Python |
| a behavioural indicator | `app/behavior/detectors.py` + a threshold in config | the renderer |
| a report field | `InvestigationReport` + the renderer | the analysis modules |
| a threshold | `.env` | source |
