# Machine learning

The honest version. This file documents what the ML stage does, where its labels
come from, how it is evaluated, and — at length, because it is the most important
part — the conditions under which it **refuses to produce a number**.

Production engine: `app/ml/real_labels.py`, `real_features.py`,
`real_training.py`, `real_predictor.py`, `unsupervised.py`.

## The problem with the obvious approach

The task is "is this wallet controlled by a VASP". The obvious pipeline is: take
the known-VASP seed list as positives, take everything else as negatives, train a
classifier, report accuracy.

That pipeline produces a high number and measures nothing. With six seed
addresses and thousands of graph addresses, "everything else is a negative"
manufactures a 99.9% majority class out of *ignorance*, and a model that always
answers "not a VASP" scores 99.9%. The number is real arithmetic on a fabricated
premise.

So the rule, enforced in `real_labels.py` and stated in its docstring:

> **Absence from the seed set is not a label.**

A wallet missing from six curated addresses is overwhelmingly likely to be
unlabelled. It is not known-not-a-VASP. Calling it one is fabricating ground
truth, and every metric downstream inherits the fabrication.

## Where labels legitimately come from

Two sources, both in `app/ml/real_labels.py` (`LabelSource`):

### `PROTOCOL_GUARANTEED`

Facts about how EVM chains work, not annotations anyone could get wrong:

* An address that **sent a top-level transaction** signed it. Only a private key
  holder can, so it is an `EXTERNALLY_OWNED_ACCOUNT`.
* An address that **sent an internal transfer** moved value from inside contract
  execution, which only code does, so it is a `CONTRACT`.
* An ERC-20 transfer's **`asset_identifier`** is the contract that emitted the
  event, so it is a `CONTRACT`.

These are as close to free ground truth as this domain offers. They support the
`account_type` task (`AccountTypeLabel`: `EXTERNALLY_OWNED_ACCOUNT` / `CONTRACT`).

`_native_stream_is_trustworthy` guards them: a record claiming a native stream
while describing a token transfer is excluded rather than trusted, because the
label's whole validity rests on the stream being what it says it is.

### `DATASET_PROVENANCE`

An address the seed dataset records with **first-party** provenance —
`official_disclosure` or `directly_verified` only. This supports the
`vasp_ownership` task (`VASPLabel`: `VASP_OWNED` / `NOT_VASP_OWNED`).

`public_label`, `third_party_label`, `community_label`, `inferred` and
`unverified` are **excluded from training** even though they are used for
attribution reporting. Training on a vendor's label teaches the model to
reproduce that vendor's heuristic, not a fact about the chain — and then the
model's agreement with the vendor gets reported as accuracy, which is circular.

Schema version: `LABEL_SCHEMA_VERSION = "real-labels-v1"`. Changing what counts
as a label bumps it.

## Features

`app/ml/real_features.py`. 33 named features in four groups, schema version
`real-features-v1`.

| Group | Examples |
|---|---|
| structure | `out_transfer_count`, `unique_counterparties`, `reciprocity_ratio`, `top_counterparty_share`, `counterparty_concentration_hhi`, `self_loop_count` |
| value | `log_total_out_amount`, `log_max_amount`, `distinct_amount_ratio`, `zero_amount_share` |
| temporal | `lifespan_days`, `active_day_count`, `log_median_gap_seconds`, `max_transfers_in_one_hour`, `night_hour_share`, `hour_entropy`, `weekday_entropy`, `timestamped_share` |
| asset_mix | `distinct_asset_count`, `native_transfer_share` |

Features are stored **by name**, and `to_vector()` is the only place a positional
ordering is imposed — always via `feature_names(task)`. Order is part of the
schema, so a reordering is a breaking change that must bump the version. Named
storage is also what lets an explanation print `out_transfer_count = 44` next to
an importance value instead of "feature 7 mattered".

### Per-task feature exclusion (leakage control)

`TASK_EXCLUDED_GROUPS` removes `asset_mix` from the **`account_type`** task, and
records why in the artifact:

> `asset_mix` excluded: an address's native/token activity mix is a proxy for the
> provider stream (`txlist` vs `tokentx`) that defines the account-type label, so
> these features could leak the answer.

That is the subtle leak worth understanding. The `account_type` label is derived
*from which stream a record appeared in*; a feature measuring the native/token
mix is measuring the label's own source. The model would score well by
rediscovering the labelling rule. So `account_type` sees 31 features, not 33.

`vasp_ownership` excludes nothing, and the reason is recorded too: its labels come
from dataset provenance, which no feature in the module reads.

`temporal_data_missing` is a flag rather than zeros, so structurally-absent
temporal features are distinguishable from genuine zeros.

## Splitting

`_split_groups` in `real_training.py`. **Group-aware, by wallet/entity**, so one
address cannot appear on both sides of a split.

Implemented as an explicit seeded pass rather than `GroupShuffleSplit` so both
properties are visible in the code: groups are shuffled with a seeded RNG, then
assigned to the held-out side until it reaches its target size, **skipping any
group whose removal would take the last remaining member of a class out of
training**.

Division: train / validation / **untouched test**. The test split is scored once,
at the end. Cross-validation (`ML_CV_FOLDS`, default 5) runs on the training
portion for model selection, so candidate comparison never touches held-out data.

### The residual leakage this cannot fix

Recorded on every training outcome as a limitation, verbatim:

> All samples' features come from one shared transaction graph, so a training
> address and a test address can be direct counterparties. Their features are
> therefore not statistically independent, which inflates held-out scores
> relative to a separately-crawled sample. Group-aware splitting cannot remove
> this; only a second, independent crawl would.

This is disclosed rather than hidden because it is real and it is unfixable
without more data acquisition. A held-out score from a single-graph dataset is an
optimistic estimate, and a reader deserves to know by how much and why.

## Models

Candidates in `_candidate_models`:

* `LogisticRegression` (in a scaling pipeline)
* `RandomForestClassifier`
* `GradientBoostingClassifier`
* `HistGradientBoostingClassifier`
* `XGBClassifier` — when `xgboost` is importable
* `LGBMClassifier` — when `lightgbm` is importable

Classical tabular models only. **No deep learning**, because nothing about 33
tabular features on a few dozen samples justifies it: a neural network here would
add parameters, non-determinism and opacity while losing to a random forest.

All candidates are cross-validated and **every candidate's score is kept in the
artifact**, including the losers (`CandidateResult`). "We chose a random forest"
is then a visible comparison rather than an assertion.

Class imbalance is handled by `class_weight` / sample weights, with the choice
recorded in `class_imbalance_handling` — never by resampling that duplicates
samples, which inflates metrics by scoring the same address twice.

## Metrics

`EvaluationMetrics`, per split, with the split always named:

accuracy · precision · recall · F1 · **ROC-AUC** · **PR-AUC** · confusion matrix
(row-major `[[TN, FP], [FN, TP]]`, positive class named) · class counts ·
decision threshold · **majority-class baseline accuracy** ·
**accuracy above baseline**

The last two are the ones that keep the rest honest. On an imbalanced problem a
high accuracy can be entirely explained by the imbalance, so the baseline is
printed next to it: "84% accuracy" beside "82% majority baseline" reads correctly
as +2, which is what it is.

PR-AUC is reported alongside ROC-AUC because ROC-AUC is optimistic under
imbalance.

### The number reported is the number measured

> If real held-out performance is 76%, the report says 76%. If it is 68%, the
> report says 68% and explains why.

No threshold tuning against the test split. No re-splitting until a seed produces
a better figure. No reporting a validation score as a test score. No reporting
training accuracy at all. The target of ~75–80% is a target for the *data*, not a
number to be reached by other means.

## Versioning

Every production prediction carries: model name, model version, `PIPELINE_VERSION`
(`real-training-v1`), dataset version, feature schema version
(`real-features-v1`), label schema version (`real-labels-v1`), training timestamp
(epoch and UTC string), sample count, per-class counts, all metrics, random seed
(`ML_RANDOM_SEED`, default 42), and the environment.

Dataset version is a deterministic hash of the sorted `(address, label, source)`
triples plus both schema versions — so a changed dataset produces a changed
version automatically, and a stale artifact is detectable rather than assumed
current.

## The refusal path

`ML_MIN_SAMPLES_PER_CLASS` (default **20**) is the most important ML setting in
the project. Below that many labelled samples **per class**, training is refused:
`TrainingOutcome.trained = False`, `blockers` lists what is missing with exact
counts, and no metric is reported.

`ML_MIN_TRANSFERS_PER_SAMPLE` (default 3) is an activity floor applied to the
sampling frame **before any label is read**, so it cannot admit one class more
readily than another.

### What actually happens on the real data available here

On the largest real dataset in this environment, the supervised path refuses, and
correctly:

| Task | Class counts | Verdict |
|---|---|---|
| `account_type` | 7 `CONTRACT`, 57 `EXTERNALLY_OWNED_ACCOUNT` | refused — 7 < 20 |
| `vasp_ownership` | 1 `VASP_OWNED`, 0 `NOT_VASP_OWNED` | refused — negative class **empty by design** |

The empty negative class is not a bug to be worked around. It is the direct,
correct consequence of the rule at the top of this file: nothing in a 2000-record
graph can be labelled `NOT_VASP_OWNED`, because absence from a six-address
dataset is not evidence of not being a VASP. 5 `community_label` addresses were
additionally excluded from training on provenance grounds.

The `account_type` shortfall has a more mundane cause, and the report names it:
1969 records were excluded from stream-based labelling because they **claimed a
native provider stream while describing a token transfer**. That is what happens
when records are loaded from a normalized fixture written before
`transfer_source` existed — they take the default, and the labelling rule refuses
to trust a stream tag it can see is wrong. Those records remain valid graph
evidence; they just cannot be a label's source.

So the contract class is scarce here for a fixable reason. `CONTRACT` labels come
mainly from **internal transactions** (`txlistinternal`), which contribute a
label for every address that moved native value from code. A live acquisition
carries the correct `transfer_source` on every record, which is what would make
this task trainable. The report says exactly this rather than leaving the reader
to conclude the task is impossible in principle.

**No accuracy is reported for either task, and none can be** on this data.
Reaching a genuine 75–80% supervised result requires a live crawl with intact
stream provenance for `account_type`, and more first-party-provenance dataset
entries for `vasp_ownership`. Both are data-acquisition problems, and they are
stated as such rather than solved with synthetic labels.

## The unsupervised fallback

When supervised training is refused, `app/ml/unsupervised.py` runs instead —
`IsolationForest`, method version `real-unsupervised-v1`, on the same real
features. It needs no labels, which is exactly why it is available when labels
are not.

It reports an outlier score, the features that deviate most from the population
(with the address's value beside the population median, so the deviation is
checkable), and **bootstrap rank stability**: mean and minimum Spearman
correlation of the ranking across resamples.

Measured here: mean Spearman **0.9871**, minimum **0.9822** — the ranking is
highly reproducible.

That is a statement about **stability, not correctness**. The report says so
explicitly:

> No accuracy, precision, recall or F1 is reported for this model, and none can
> be — there is no ground truth to be accurate against. Any such figure would be
> fabricated.

A stable ranking of unusual addresses is genuinely useful for triage. It is not a
VASP classifier, and it is never presented as one.

## Approach reporting

`MLSection.approach` states which path actually ran:

| Value | Means |
|---|---|
| `SUPERVISED` | Trained on real labels, held-out metrics reported |
| `UNSUPERVISED` | Supervised refused; outlier analysis with rank stability |
| `UNAVAILABLE` | Neither path was possible — blockers stated |
| `DISABLED` | `--no-ml`; the stage did not run |

Section 7 always prints the label census and the blockers, whichever path ran, so
a reader can see *why* they got the analysis they got.

## What is deliberately not here

* **No synthetic training data on the production path.** The Milestone-5 demo
  classifier still exists, is type-locked to
  `MLPrediction.training_data_type: Literal["SYNTHETIC_DEMO"]`, ships a
  disclaimer, and is reachable **only** through the HTTP API and
  `ml_attribution.py`. `investigate.py` cannot load it.
* **No test data used as training data.** The sampling frame is the investigation
  graph; `tests/` is never read by the ML pipeline.
* **No deep learning.** Unjustified at this scale.
* **No USD-denominated features.** The free provider tier has no historical
  pricing, and an estimated price would put a fabricated number in the feature
  vector.
