# Operations

Running the system, verifying it, and diagnosing it when it stops.

## The primary command

```bash
python investigate.py 0x75c0623bae00749550cf1c1703e7382038b3109a --chain ethereum
```

That performs the entire investigation: validate, acquire real data, normalize,
build the graph, analyse inbound and outbound activity, trace fund-flow
candidates, attribute in both directions, resolve entities, measure behavioural
indicators, score risk transparently, run the ML analysis its labels support, and
print the compact ten-block report.

Requires `ETHERSCAN_API_KEY` in `.env`. Without it the run **stops** — see
[the failure table](#when-it-stops).

## Choosing the output

Everything below is analysed identically; the flags choose only what is printed.

```bash
python investigate.py <WALLET> --chain ethereum
```

Compact ten-block report — the default, and the one to use for a demonstration.

```bash
python investigate.py <WALLET> --chain ethereum --full-report
```

The complete nine-section report: every route, every evidence classification,
the full ML record, every stated limitation. `--verbose` selects it too, and
additionally streams stage-by-stage progress to stderr.

```bash
python investigate.py <WALLET> --chain ethereum --json
```

The complete machine-readable record. Unaffected by the two flags above — it
always carries everything, including the per-transfer ledger that block 4 of the
compact report displays and any row a display cap withheld.

## Common invocations

Machine-readable output only, safe to pipe (progress goes to stderr):

```bash
python investigate.py 0x75c0623bae00749550cf1c1703e7382038b3109a --json --verbose
```

Both at once — the human report on stdout, JSON to a file:

```bash
python investigate.py 0x75c0623bae00749550cf1c1703e7382038b3109a --json report.json
```

The report on stdout is the compact one unless `--full-report` (or `--verbose`)
is also given.

A deep, time-bounded trace on a busy wallet:

```bash
python investigate.py 0x75c0623bae00749550cf1c1703e7382038b3109a --max-hops 5 --time-window 180
```

Force fresh provider data, replacing the cached responses:

```bash
python investigate.py 0x75c0623bae00749550cf1c1703e7382038b3109a --refresh
```

Skip the ML stage (the blockchain evidence is unaffected):

```bash
python investigate.py 0x75c0623bae00749550cf1c1703e7382038b3109a --no-ml
```

## Offline runs against real data

Both modes label their output `CACHED REAL DATA`, in the header and in the
conclusion. Neither is ever described as live.

**From a saved graph** — fastest, no credentials needed:

```bash
python investigate.py 0x75c0623bae00749550cf1c1703e7382038b3109a --chain ethereum --cached-graph data/graphs/0x75c0623bae00749550cf1c1703e7382038b3109a_ethereum_726c240208074a5599b60ba8e99cede8.gpickle
```

**From a normalized transfer file** — slower, but rebuilds the graph from records
that still carry per-transfer provenance, so the normalization accounting prints
and the supervised labelling rules can actually run:

```bash
python investigate.py 0xd8da6bf26964af9d7eed9e03e53415d37aa96045 --chain ethereum --transfers-file data/fixtures/0xd8da6bf26964af9d7eed9e03e53415d37aa96045_ethereum.json
```

Use `--transfers-file` when you care about ML labelling or the ingestion
accounting; use `--cached-graph` when you only need the graph analysis.

Note that graphs written by an older builder lack `block_number`,
`token_contract`, `transfer_source` and `gas_used` on their edges. Loading one
shows those fields as **absent**, not fabricated. Live acquisition and
`--transfers-file` carry the full metadata.

## When it stops

Exit `0` completed · `1` could not run · `2` interrupted. A stop prints
`INVESTIGATION STOPPED: <reason>` — or `CONFIGURATION ERROR: <reason>` for a
malformed setting — to **stderr**, so `--json` on stdout stays parseable.

| Message | Cause | Fix |
|---|---|---|
| `ETHERSCAN_API_KEY is not set, so no real blockchain data can be acquired.` (and: `This build does NOT fall back to demo or synthetic data`) | No credential. | Put a key in `.env`, or use `--cached-graph` / `--transfers-file`. |
| `Unsupported chain 'X'. This build can resolve: ...` | Name not in the chain registry. | Use a resolvable name. The message lists them and says which have been exercised live. |
| `'X' is not a valid EVM address: expected '0x' followed by 40 hexadecimal characters...` | Malformed address. | Check for a truncated paste — the message reports the length it got. |
| `CONFIGURATION ERROR: HTTP_TIMEOUT_SECONDS='fifteen' is not a whole number. Fix it in .env or remove the line to use the default (15).` | Malformed `.env` value. **Deliberately fatal** rather than silently defaulted. | Fix the value or delete the line. |
| `ETHERSCAN_CHAIN_ID=137 does not match chain 'ethereum', whose Etherscan chain id is 1.` | The pinned id contradicts `--chain`. | Unset `ETHERSCAN_CHAIN_ID` (the normal case) or pass the chain that matches it. |
| `does not appear on any edge of the loaded graph` | Warning, not a stop. The graph is real but for a different wallet. | Check the `--cached-graph` path. |
| `DATA HORIZON: the graph contains complete edges only to N hop(s)...` | Not a stop. Acquisition *completed* fewer hop levels than `--max-hops` requested — normally because an address budget bit — so the deeper result is `INCONCLUSIVE` rather than `NONE`. | Either accept the finding as complete at N hops, or widen acquisition: raise `EXPANSION_MAX_ADDRESSES` / `EXPANSION_MAX_ADDRESSES_PER_HOP` and re-run with `--refresh`. Raising `--max-hops` or the *edge* budget will not help — the edges were never fetched. Check `ACQUISITION STOPPED` for which budget bit. |

Two of these are worth understanding rather than just fixing. The missing-key stop
is the no-synthetic-fallback rule in action: a report built from invented
transactions looks exactly like a real report, which makes it worse than no report
at all. And the configuration error is fatal on purpose: a silently-defaulted
setting means the operator believes one thing is configured while another is
running.

## Reading the report

Nine sections. Where to look first:

* **§1 normalization accounting** — does `reconciled` say `YES`, and does
  `edge accounting` say `RECONCILED`? If not, evidence was lost and the report
  says so. Check the `INCOMPLETE` markers too: a truncated stream is why a later
  section may say `INCONCLUSIVE`. In `--cached-graph` mode this block is absent
  and the report states why — normalization ran when the graph was built, not
  now — so use `--transfers-file` or a live run when you need the accounting.
* **§3 provenance breakdown** — how much of the dataset is
  `official_disclosure` versus `community_label`. This bounds how strongly any
  attribution can be stated.
* **§4 bidirectional count** — how many operators have evidence in **both**
  directions. One-directional evidence is still meaningful; the count is not a
  score.
* **§2 search accounting** — did the search finish? `search completed: NO` names
  the cause, and there are two with opposite remedies. *Budget exhausted* — raise
  `--max-paths` / the edge budget, or narrow `--time-window`, and re-run.
  *Data horizon* — `DATA OBSERVED TO: N hop(s)` is shallower than `MAX HOPS`,
  meaning hops beyond N were never fully acquired; raising a path or edge budget
  cannot help, the address budget has to be raised so acquisition reaches the
  rest of hop N. Either way a negative is
  `INCONCLUSIVE`, not `NONE`. Compare it against `ADDRESSES FETCHED` and
  `ACQUISITION STOPPED` in the data-status block, which name the budget that bit.
* **§7 approach** — `SUPERVISED`, `UNSUPERVISED`, `UNAVAILABLE` or `DISABLED`,
  plus the label census explaining which you got.
* **§8 arithmetic** — the components sum to the score. If a number looks wrong,
  the line that produced it is printed.

Anything the terminal caps for readability is present in full in `--json`.

## The `--json` artefact

Read it as **UTF-8**. On Windows, Python defaults to cp1252 and will fail on the
first non-ASCII byte:

```python
import json
with open("report.json", encoding="utf-8") as fh:   # encoding is required
    report = json.load(fh)
```

Top-level keys mirror `InvestigationReport`: `wallet`, `chain`,
`investigation_id`, `provenance`, `parameters`, `environment`, `normalization`,
`graph_summary`, `attribution`, `entities`, `counterparties`,
`behavior_patterns`, `temporal`, `ml`, `risk`, `seed_*`, `conclusion`,
`limitations`, `warnings`.

The field is `behavior_patterns`, not `behavior`.

It contains no credentials — no API key, no `DATABASE_URL`, no password, no
connection string. The `environment` block carries only the Python version and
platform.

## Tests

```bash
python -m pytest -q
```

550 tests, fully offline. No key, no network. Targeted runs while working on one
area:

```bash
python -m pytest tests/test_chains.py tests/test_real_ml.py -q
```

Verbose, with the first failure's full context:

```bash
python -m pytest -x -vv
```

The suite must end with **0 failed**. If it does not, fix the implementation — a
test adjusted to make a failure disappear removes the only thing that was
reporting a defect.

## Verifying a build honestly

The checks worth running before demonstrating anything:

1. `python -m pytest -q` ends with 0 failed.
2. A complete run against **real** data (live, or a cached real graph — never
   synthetic) exits 0.
3. The header states the correct `DATA MODE`, and it matches how the run was
   actually invoked.
4. Output is pure ASCII on a Windows console — no `?` boxes, no mojibake:
   ```bash
   python investigate.py <WALLET> --cached-graph <PATH> > out.txt 2> err.txt
   ```
   `err.txt` should be empty, and `out.txt` should contain all nine section
   headers plus `INVESTIGATION COMPLETE`.
5. The ML section reports the approach it actually took, with the label census —
   and if it refused to train, that the refusal and its counts are printed rather
   than an accuracy figure.
6. No secret appears in the JSON:
   ```bash
   python -c "import json;d=open('report.json',encoding='utf-8').read();print([k for k in ('apikey','password','postgresql://','sqlite:///') if k in d.lower()])"
   ```
   That should print `[]`.
7. Nothing on the production path reads from `tests/` or a synthetic seed file.

Do not stop at the first successful run. A smoke test that exits 0 tells you the
process did not crash; it does not tell you the report is true.

## Current environment status

**`ETHERSCAN_API_KEY` is now configured in `.env` and live recursive acquisition
has been verified end to end.** A live run of `0x75c0...109a` on `ethereum` with
`--refresh --max-hops 4` fetched **40 addresses in 120 provider requests**,
returning **91,614 real records** across the native, internal and token streams;
normalization kept 89,292 after removing 2,322 duplicates with 0 rejects and
`RECONCILED: YES`, and the graph came out at **7,674 nodes / 89,259 edges**. All
nine sections rendered stamped `DATA MODE: REAL`, exit 0 with empty stderr. Of
those edges, 89,181 touch neither end of the investigated wallet — they were
acquired by expanding counterparties, which is what distinguishes a recursive run
from the radius-1 star the same wallet produced before.

The key is read from `.env` by `load_dotenv()` at import of `app/core/config.py`
and is never printed: it is stripped from cache identities, absent from graph
filenames, absent from the JSON artefact and absent from every log line. Confirm
without revealing it:

```bash
python -c "from app.core.config import get_settings; print('key present:', bool(get_settings().etherscan_api_key))"
```

`.env` is git-ignored. `.env.example` must carry an **empty** `ETHERSCAN_API_KEY=`
— a real value was found there once and removed; check it stays empty.

What a live run does **not** give you: the requested depth as an accomplished
fact. That same run discovered **6,352** addresses and could afford to fetch
**40**, so it stopped with `ACQUISITION STOPPED: ADDRESS_BUDGET_REACHED` —
`REQUESTED HOP DEPTH: 4`, deepest hop actually reached 3, and
`OBSERVED HOP DEPTH: 2`, because addresses at hop 2 were left unexpanded and no
edge leads onward from them. Attribution beyond 2 hops is therefore
`INCONCLUSIVE`, not `NONE`, and raising `--max-hops` or a path/edge budget will
not change it — the remedy is `EXPANSION_MAX_ADDRESSES` /
`EXPANSION_MAX_ADDRESSES_PER_HOP` and a re-run with `--refresh`.

Two further honest limits on that graph:

* **An expanded counterparty is read from its oldest transactions forward.**
  Every stream is walked ascending from block 0 and cut at
  `EXPANSION_MAX_RECORDS_PER_STREAM`, so a busy counterparty contributes its
  *earliest* history, which may predate the investigated wallet entirely (this
  graph spans 2021 to now). Chronologically ordered multi-edge chains through
  such a hop can therefore be absent from the data even where they exist
  on-chain. Windowing expanded fetches around the wallet's own activity is the
  obvious next improvement and is **not** implemented.
* **This wallet produced no multi-hop VASP route.** All nine direct
  counterparties are `[NOT_IN_DATASET]`, and the only seed address anywhere in
  the graph — Binance `0xf977...acec` — sits at undirected distance 4 and is
  reported under `RELATED BY UNDIRECTED CONNECTIVITY ONLY (NOT attributed)`. That
  is the correct output for this wallet against this six-address dataset, not a
  failure of the traversal.

Available real artefacts:

| Path | Wallet | Contents |
|---|---|---|
| `data/graphs/0x75c0...109a_ethereum_726c2402...gpickle` | `0x75c0...109a` | ~13 MB real graph |
| `data/fixtures/0xd8da...6045_ethereum.json` | `0xd8da...6045` | ~3 MB normalized real transfers |

Production seed dataset — six entries, deliberately not padded:

| Operator | Provenance |
|---|---|
| Binance (2 addresses) | `community_label` |
| Kraken (2 addresses) | `community_label` |
| Bitstamp | `community_label` |
| OKX | `official_disclosure` |

A negative attribution against this dataset means "not connected to these six
addresses", nothing broader.

## The HTTP API

```bash
python -m uvicorn app.api.main:app --reload
# http://127.0.0.1:8000/docs
```

Health check, which reports database reachability:

```bash
curl http://127.0.0.1:8000/health
```

**The HTTP surface is behind the CLI**: it still uses unidirectional attribution
and the Milestone-5 synthetic-demo classifier (type-locked to
`training_data_type: SYNTHETIC_DEMO`, with a per-prediction disclaimer). Use
`investigate.py` for anything you intend to rely on, and do not cite the HTTP
`/ml` endpoint as evidence.

## Legacy per-stage scripts

Useful for debugging one stage in isolation. All read `GRAPH_CACHE_DIR`, so a
graph built by one is visible to the others and to `investigate.py`.

```bash
python build_graph.py --fixture data/fixtures/0xd8da6bf26964af9d7eed9e03e53415d37aa96045_ethereum.json
python trace_funds.py 0xd8da6bf26964af9d7eed9e03e53415d37aa96045
python attribute_wallet.py 0xd8da6bf26964af9d7eed9e03e53415d37aa96045
```

`ml_attribution.py` runs the **synthetic demo** classifier, and
`attribute_wallet.py --demo` loads the **synthetic** seed set. Neither is the
production path.

## Housekeeping

| Path | Contents | Safe to delete |
|---|---|---|
| `data/cache/provider/` | Raw provider responses | yes — next run re-fetches |
| `data/graphs/` | Built graphs | yes, but a cached real graph may be the only offline real data available |
| `data/ml/` | Trained artifacts and metrics | yes — retrained on demand |
| `data/blockchain_intel.db` | HTTP API persistence | yes if you do not need past investigations |

Never delete `data/seed/known_vasps.json` — it is curated, provenance-carrying
data, not a cache.
