# Evidence semantics

Every claim this system prints carries a class, and each class is a promise about
what the claim will and will not support. This file is the contract.

It exists because the failure mode that matters in attribution work is not a
crash. It is a confident sentence that a reader takes to mean more than the data
justifies.

## Evidence classes

`app/analysis/risk.py::EvidenceClass`

| Class | Means | Does **not** mean |
|---|---|---|
| `DIRECT` | The wallet transacted with a dataset address in a single hop. A tx hash proves it. | That the dataset address really belongs to the named operator — that depends on the entry's provenance. |
| `INDIRECT` | A multi-hop path exists between the wallet and a dataset address. | That funds flowed along it. See "path vs continuity" below. |
| `SUPPORTING` | Consistent with a conclusion reached by other evidence. Behavioural indicators and ML land here. | That the conclusion holds. Supporting evidence cannot establish anything on its own. |
| `CONTEXTUAL` | Background that helps a reader interpret the rest. | Anything about this wallet specifically. |
| `INCONCLUSIVE` | The question was asked and the data could not answer it. | A negative answer. |

`INCONCLUSIVE` is the one most systems omit, and the omission is the bug. If a
search hits a budget and finds nothing, "nothing found" is false — the search did
not finish. The report says `SEARCH STATUS: INCONCLUSIVE` and gives the
accounting (paths found, edges explored, which budget was exhausted). **A limit
hit is never reported as `NONE`.**

`NONE` is reserved for a search that ran to completion over complete data and
found nothing. It is a real finding, and it is only available when it is true.

## The data horizon

A budget stop is not the only way a search can be incomplete, and it is not the
common one. The other is the **data horizon**: the hop radius for which the graph
actually holds complete edges, which is a property of *acquisition*, not of the
traversal.

Live acquisition expands recursively — the wallet's own transactions, then its
counterparties', hop by hop to `--max-hops` — but it does so under hard address
budgets, so the radius it *completes* is usually shallower than the radius it
requested. `observation_depth` is the number of leading hop levels in which
**every** discovered address was actually fetched. One address deferred by a
budget at hop 2 holds it at 2 however far the walk got, because a route through
that address would be absent from the data rather than ruled out. Walking such a
graph with `MAX_HOPS=4` proves nothing whatsoever about hop 3: those transactions
were never fetched, so a 3-hop route could not appear *even if it exists
on-chain*.

Reporting that as `NONE` would be the exact failure this document exists to
prevent — an unobserved hop presented as searched and empty. So the pipeline
records `observation_depth` on the provenance (measured by live acquisition,
read back from a graph that recorded its own radius, and otherwise inferred as
the undirected eccentricity of the wallet), passes it into
the traversal, and when it is shallower than `MAX_HOPS`:

* `SearchStatus` becomes `INCOMPLETE` and attribution becomes `INCONCLUSIVE`;
* the report prints `DATA OBSERVED TO: N hop(s)  <-- shallower than MAX HOPS`;
* the negative is stated precisely — complete at N hops, inconclusive beyond;
* the remedy named is **widening acquisition**, not raising a budget.

That last point is why the two causes are distinguished rather than merged into
one "incomplete". They call for opposite actions, and `SearchAccounting`
carries `incomplete_reason` and `observation_depth` so the report can name the
one that actually applies. A run whose horizon reaches `MAX_HOPS` still yields a
clean `NONE` — the horizon narrows claims, it does not suppress them.

## Path is not continuity

The most abusable structure in blockchain analysis:

```
WALLET --0.5 ETH--> 0xINTERMEDIARY --0.5 ETH--> 0xEXCHANGE
```

Read carelessly this says "the wallet sent 0.5 ETH to the exchange". It does not.
It says two transfers happened involving a common address. The intermediary may
have thousands of counterparties; the outbound transfer may predate the inbound
one; the amounts may coincide.

So the vocabulary is fixed:

* **FUND-FLOW CANDIDATE** — a route worth checking
* **TRANSACTION PATH** — a sequence of real transactions sharing addresses
* **TRACEABLE PATH** — a path whose hops are individually verifiable

and never "the funds went to X" for anything but a direct transfer.

Each candidate carries a `PlausibilityGrade` (`app/tracing/quality.py`):

| Grade | Meaning |
|---|---|
| `DIRECT_TRANSFER` | One hop. The tx hash *is* the evidence. |
| `PLAUSIBLE` | Chronology holds, asset consistent, no hub, gaps reasonable. |
| `WEAK` | Something undermines continuity — see concerns. |
| `IMPLAUSIBLE` | The route exists in the graph but continuity is not credible. |

Concerns, named individually:

| Concern | Why it weakens the route |
|---|---|
| `ASSET_CHANGED` | Different asset in and out. Not impossible (a swap), but no longer the same funds. |
| `LONG_TIME_GAP` | A month between hops. Everything in between is unaccounted for. |
| `HIGH_THROUGHPUT_INTERMEDIARY` | A hop with thousands of counterparties connects almost everything to almost everything. |
| `AMOUNT_INCREASED` | More left than arrived, so the excess came from elsewhere. |
| `UNVERIFIABLE_CHRONOLOGY` | Timestamps missing or out of order; the sequence cannot be checked. |

Grades never hide a path. A concern is a caveat printed next to the evidence, not
a filter applied before the reader sees it.

## Address matching

Exact and case-insensitive. Nothing else. No fuzzy matching, no prefix or
substring matching, no edit distance, no "similar-looking address" heuristic.

Case-insensitivity is correct because EVM addresses are hex and mixed case is
EIP-55 checksumming — `0xABC...` and `0xabc...` are the same address. Anything
beyond that is not. Two addresses sharing eight leading characters are unrelated;
`0x` prefix collisions are trivially cheap to manufacture, so a prefix match is
not weak evidence, it is an attack surface.

This is enforced, not conventional: no fuzzy matching function exists in the
codebase to call by mistake.

## Dataset provenance

`app/attribution/models.py::SeedSourceType`

| Provenance | What it is | ML label? |
|---|---|---|
| `official_disclosure` | The operator published this address themselves | yes |
| `directly_verified` | Confirmed by direct interaction or an on-chain proof | yes |
| `public_label` | Widely published, no first-party confirmation | no |
| `third_party_label` | A block explorer or analytics vendor's label | no |
| `community_label` | Crowd-sourced | no |
| `inferred` | Derived by heuristic | no |
| `unverified` | Recorded, unchecked | no |
| `synthetic_demo` | Invented for tests. **Rejected on the production path.** | never |

`SYNTHETIC_DEMO` is separated at the **type level**, not by convention, and
guarded in three independent places. If a dataset containing synthetic entries is
ever loaded — by pointing `VASP_SEED_DATASET_PATH` at the demo file, say — the
report carries a top-level warning, **every affected candidate** carries the
limitation "SYNTHETIC DEMO seed entry — this address is not a real VASP address
and this candidate carries no real-world meaning", and every affected entity
carries its own. There is no configuration under which a synthetic address can be
reported as a real-world finding without saying so on the same line.

**A third-party label is not ownership.** When an explorer labels an address
"Binance 14", it means someone at that explorer concluded so, by a method they
did not publish, at a time they did not record. It is a good lead. It is not a
Binance disclosure, and the report never presents it as one — every candidate
prints its entry's provenance and source URL so a reader can check the claim
rather than inherit it.

The consequence, stated in every report: an attribution built on a
`community_label` entry is an **investigative lead**, not a confirmed
identification.

## Behaviour is not crime

`app/behavior/models.py::IndicatorClass`

| Class | Means |
|---|---|
| `INVESTIGATIVE_INDICATOR` | A measurable pattern worth a human look |
| `SUPPORTING_EVIDENCE` | Consistent with a conclusion other evidence reached |
| `CONTEXTUAL` | Background |
| `REQUIRES_FURTHER_VERIFICATION` | Detected, but the data is not sufficient to rely on |

There is no `SUSPICIOUS`, no `MALICIOUS`, no `LAUNDERING` class, and no risk band
that names a crime.

Every one of the fifteen indicators describes legitimate activity as readily as
illegitimate. `RAPID_HOPPING` is a bot or an arbitrageur. `SPLIT_PATTERN` is a
payroll run. `CONSOLIDATION_PATTERN` is a merchant sweeping deposits.
`HIGH_ACTIVITY_DENSITY` is a market maker. `DORMANT_THEN_ACTIVE` is someone who
found an old key.

So each finding reports its **observed metric, the threshold it crossed, the
value actually measured, the evidence, the related addresses and the relevant
transaction hashes** — and stops. The report gives an investigator something
checkable; it does not give them a conclusion they did not reach.

## Attribution status

`app/attribution/models.py::AttributionStatus`

| Status | Means |
|---|---|
| `MATCH_FOUND` | At least one candidate with address-level evidence |
| `NONE` | Complete search over complete data, no match |
| `INCONCLUSIVE` | Search or data incomplete; a match may exist unseen. Two distinct causes: a budget was exhausted (raise it and re-run), or the data horizon is shallower than `MAX_HOPS` (widen acquisition — no budget will help). The report names which. |

And `EvidenceTier` — `DIRECT` (one hop) or `INDIRECT` (multi-hop) — recorded per
candidate, because the two justify very different statements.

## What ML may and may not do

ML is `SUPPORTING` evidence, permanently.

* It may **not** create an attribution. A model output is not a transaction.
* It may **not** override direct blockchain evidence. When a tx hash and a
  prediction disagree, the tx hash is right; a model that contradicts a
  verifiable transaction is reporting its own error.
* It may **not** be reported without its provenance — model name, model version,
  dataset version, feature schema version, training timestamp, sample and
  per-class counts, metrics and random seed accompany every production
  prediction.
* It may **not** report a metric its labels do not support. See [ML.md](ML.md).

What it may do: rank, flag, and surface a pattern in the feature space that a
human then verifies against the chain.

## Data mode

`app/investigation/pipeline.py::DataMode`

| Mode | Means |
|---|---|
| `REAL` | Fetched live from the provider during this run |
| `CACHED REAL DATA` | Real chain data reloaded from disk, acquired earlier |

Both are real. Only one is current, and neither says how *wide* it is: a live
graph observes as many hops as its address budget allowed acquisition to complete,
and a cached graph observes whatever the run that wrote it did. That is
`observation_depth`, reported separately — see "the
data horizon" above. The distinction between the two modes is printed in the
header,
because a stale graph can miss exactly the transaction that changes the
conclusion, and a reader who thinks they are looking at live data has no way to
know that.

There is no third mode. Production **never** silently falls back to demonstration
data: with no credentials, the run fails and names the missing variable. A report
built from invented transactions is worse than no report, because it looks like a
report.

## Reading a conclusion

Section 9 states what was established, what was not, and what remains
inconclusive. It is written to be quoted, so:

* it never uses "confirmed" for anything short of `DIRECT` evidence on a
  first-party-provenance dataset entry;
* it never converts an indicator into an allegation;
* it says which questions the data could not answer;
* it repeats the data mode, so a quoted conclusion carries its own caveat.

If the honest conclusion is "this wallet has no observable relationship to any of
the six dataset addresses, and the search was incomplete", that is what it says.
