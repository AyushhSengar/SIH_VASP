"""
COMPACT RENDERER -- the default investigation output.

WHAT THIS IS FOR
--------------------------------------------------------------------------
`app.reporting.terminal` prints the complete nine-section report: every piece
of evidence, every methodology note, every stated limitation. That report is
the artefact you read when you are examining a case. It is not the artefact you
read when you want to know, in ten seconds, what this wallet did.

This module is the second one. It prints ten fixed blocks of facts -- numbers,
addresses, timestamps, tables -- and nothing else. No paragraphs, no
methodology, no "why this matters", no evidence taxonomy, no limitations
digest. The full report remains one flag away (`--full-report` / `--verbose`),
and `--json` is unchanged and still complete.

THE RULES THIS RENDERER ENFORCES
--------------------------------------------------------------------------
1. It computes nothing. Exactly like the full renderer, every value printed
   here is a field that an analysis module already committed to. The compact
   report cannot disagree with the full report or with `--json`, because all
   three read the same `InvestigationReport`.

2. A missing value prints `N/A`. Never 0, never "-", never an inferred
   substitute. `N/A` means the analysis did not produce the value; `0` means
   the analysis produced zero. Those are different findings and the reader
   must be able to tell them apart at a glance.

3. Truncation is announced. Long lists are capped so the report stays
   readable, and every cap prints the total and the number withheld. Nothing
   is dropped silently.

4. Evidence values are never abbreviated. Addresses and transaction hashes
   print in full even when that overruns the rule width, because a truncated
   hash cannot be looked up and a truncated address cannot be verified. Rows
   are laid out over two lines rather than shortening the evidence.

ENCODING
--------------------------------------------------------------------------
Same constraint as the full renderer: Windows consoles default to a legacy
code page, so everything is folded to ASCII before printing. The helpers are
reused from `terminal` rather than duplicated, so the two renderers can never
drift apart on formatting a timestamp or an amount.
"""

from __future__ import annotations

import sys
from typing import Any, Optional, TextIO

from app.attribution.bidirectional_models import BidirectionalAttributionResult
from app.investigation.pipeline import InvestigationReport
from app.reporting.terminal import (
    _amount,
    _ts,
    _v,
    configure_stdout,
    to_ascii,
)

#: The compact report's rule width. Wider than the full report's 78 because
#: these blocks are tables of addresses and timestamps rather than wrapped
#: prose, and a 42-character address plus two timestamps does not fit in 78.
WIDTH = 100
_RULE = "=" * WIDTH
_THIN = "-" * WIDTH

#: Display caps. Same philosophy as the full renderer: a cap bounds the
#: printed report only, `--json` still carries everything, and every cap that
#: bites prints how much it withheld.
MAX_COUNTERPARTIES = 30
MAX_TRANSACTIONS = 100
MAX_ASSETS = 20
MAX_FINDINGS = 20
MAX_RISK_ROWS = 20
MAX_ENTITY_ROWS = 15

#: What a field prints when the analysis did not produce a value.
NA = "N/A"

#: How many related addresses a finding names inline before the count carries
#: the rest. Two 42-character addresses plus the label fit the rule width; a
#: third does not, and addresses are never abbreviated.
_RELATED_INLINE = 2


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _na(value: Any) -> str:
    """`N/A` for absent, the value for everything else -- including 0.

    `if not value` would collapse a genuine zero into `N/A`, which would turn
    "measured, and the answer was none" into "not measured". Only None and the
    empty string count as absent.
    """
    if value is None or value == "":
        return NA
    return _v(value)


def _num(value: Any) -> str:
    """An integer count, or N/A. Zero prints as 0."""
    if value is None:
        return NA
    return str(value)


def _amt(value: Any) -> str:
    """An amount, or N/A. Reuses the full renderer's precision rule."""
    if value is None:
        return NA
    return _amount(value)


def _time(value: Any) -> str:
    """A UTC timestamp, or N/A. `_ts` says "(no timestamp)"; the compact
    report says N/A everywhere for consistency."""
    if value in (None, 0, ""):
        return NA
    return _ts(value)


def _duration(seconds: Any) -> str:
    """Seconds as the largest sensible unit, so a reader does not have to
    divide 5443200 by 86400 in their head."""
    if seconds is None:
        return NA
    try:
        total = int(seconds)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return _v(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total}s ({total / 60:.1f}m)"
    if total < 86400:
        return f"{total}s ({total / 3600:.1f}h)"
    return f"{total}s ({total / 86400:.1f}d)"


def _block(number: int, title: str) -> list[str]:
    """A block heading.

    Deliberately `[n] TITLE`, not `SECTION n: TITLE`. The two reports must be
    distinguishable from their text alone -- for a reader scrolling back
    through a terminal, and for a test asserting which one it got.
    """
    return ["", _RULE, f"  [{number}] {title}", _RULE]


def _kv(label: str, value: Any, pad: int = 22) -> str:
    return f"  {label + ':':<{pad}} {value}"


def _capped(items: list[Any], cap: int, noun: str) -> tuple[list[Any], list[str]]:
    """Applies a display cap and returns the note that makes it honest."""
    if len(items) <= cap:
        return items, []
    withheld = len(items) - cap
    return items[:cap], [
        f"  ... {withheld} more {noun} not shown ({len(items)} in total); "
        f"the complete set is in --json."
    ]


def _row(cells: list[tuple[str, int]]) -> str:
    """Lays out one table row.

    A cell wider than its column is NOT truncated -- it pushes the rest of the
    row right. Addresses and hashes live in these tables and a shortened one is
    useless.

    Every cell is followed by at least one space. Without that, a value that
    exactly fills its column butts against the next one and the two read as a
    single field -- `unique_outgoing_counterparties6` instead of a metric name
    and the number 6.
    """
    out = "  "
    for text, width in cells:
        padded = f"{text:<{width}}"
        out += padded if len(padded) > len(text) else text + " "
    return out.rstrip()


# --------------------------------------------------------------------------
# [1] WALLET
# --------------------------------------------------------------------------


def _wallet_block(report: InvestigationReport) -> list[str]:
    prov = report.provenance
    source = _v(prov.data_mode.value)
    if prov.provider:
        source += f" via {prov.provider}"
    lines = _block(1, "WALLET")
    lines.append(_kv("ADDRESS", report.wallet))
    lines.append(_kv("CHAIN", report.chain))
    lines.append(_kv("DATA SOURCE", source))
    lines.append(_kv("DURATION", f"{report.duration_seconds}s"))
    if not report.wallet_in_graph:
        lines.append(
            _kv("IN DATASET", "NO -- this address is on no edge of the dataset")
        )
    return lines


# --------------------------------------------------------------------------
# [2] TRANSACTION SUMMARY
# --------------------------------------------------------------------------


def _summary_block(report: InvestigationReport) -> list[str]:
    lines = _block(2, "TRANSACTION SUMMARY")
    t = report.temporal
    if t is None:
        lines.append(_kv("TOTAL TRANSFERS", _num(report.transfer_count)))
        lines.append(_kv("STATUS", "no temporal analysis available"))
        return lines
    lines.append(_kv("TOTAL TRANSFERS", _num(t.transfer_count)))
    lines.append(_kv("INCOMING", _num(t.inbound_transfer_count)))
    lines.append(_kv("OUTGOING", _num(t.outbound_transfer_count)))
    if t.self_transfer_count:
        lines.append(_kv("SELF-TRANSFERS", _num(t.self_transfer_count)))
    lines.append(_kv("FIRST TRANSFER", _time(t.first_seen)))
    lines.append(_kv("LAST TRANSFER", _time(t.last_seen)))
    lines.append(
        _kv(
            "ACTIVE PERIOD",
            NA if t.lifespan_days is None else f"{t.lifespan_days} days",
        )
    )
    lines.append(_kv("ACTIVE DAYS", _num(t.active_day_count)))
    if t.missing_timestamp_count:
        lines.append(_kv("WITHOUT TIMESTAMP", _num(t.missing_timestamp_count)))
    return lines


# --------------------------------------------------------------------------
# [3] COUNTERPARTIES
# --------------------------------------------------------------------------


def _counterparties_block(report: InvestigationReport) -> list[str]:
    lines = _block(3, "COUNTERPARTIES")
    lines.append(_kv("TOTAL UNIQUE", _num(len(report.counterparties))))
    if not report.counterparties:
        lines.append("  (none)")
        return lines
    shown, note = _capped(report.counterparties, MAX_COUNTERPARTIES, "counterparties")
    lines.append("")
    lines.append(
        _row(
            [
                ("ADDRESS", 44),
                ("TXS", 6),
                ("IN", 6),
                ("OUT", 6),
                ("FIRST SEEN", 22),
                ("LAST SEEN", 22),
            ]
        )
    )
    lines.append("  " + _THIN[:96])
    for cp in shown:
        lines.append(
            _row(
                [
                    (cp.address, 44),
                    (_num(cp.transfer_count), 6),
                    (_num(cp.inbound_count), 6),
                    (_num(cp.outbound_count), 6),
                    (_time(cp.first_seen), 22),
                    (_time(cp.last_seen), 22),
                ]
            )
        )
        if cp.entity_name:
            lines.append(f"      ENTITY: {cp.entity_name} ({_na(cp.entity_type)})")
    lines.extend(note)
    return lines


# --------------------------------------------------------------------------
# [4] TRANSACTION ACTIVITY
# --------------------------------------------------------------------------


def _activity_block(report: InvestigationReport) -> list[str]:
    """Every transfer touching the wallet, oldest first.

    Two lines per transfer. One line cannot hold a timestamp, two full
    42-character addresses, an amount and a 66-character hash, and shortening
    any of those would make the row unusable as evidence -- so the row wraps
    instead. Direction is stated explicitly rather than left to be inferred
    from which side the wallet appears on.
    """
    lines = _block(4, "TRANSACTION ACTIVITY")
    if not report.transactions:
        lines.append("  (no transfers for this address in the dataset)")
        return lines
    shown, note = _capped(report.transactions, MAX_TRANSACTIONS, "transfers")
    lines.append(
        _row([("#", 6), ("TIMESTAMP (UTC)", 22), ("DIR", 5), ("AMOUNT / ASSET", 30)])
    )
    lines.append("  " + _THIN[:96])
    for index, tx in enumerate(shown, start=1):
        asset = _na(tx.asset)
        lines.append(
            _row(
                [
                    (str(index), 6),
                    (_time(tx.timestamp), 22),
                    (tx.direction, 5),
                    (f"{_amt(tx.amount)} {asset}", 30),
                ]
            )
        )
        lines.append(f"        FROM {tx.from_address}    TO {tx.to_address}")
        lines.append(f"        TX   {_na(tx.tx_hash)}")
    lines.extend(note)
    return lines


# --------------------------------------------------------------------------
# [5] TIMING ANALYSIS
# --------------------------------------------------------------------------


def _fast_passthrough_count(report: InvestigationReport) -> Optional[int]:
    """How many inbound->outbound turnarounds fell under the fast threshold.

    Read off the FAST_INBOUND_OUTBOUND finding when it fired. When it did not
    fire, the count is genuinely zero -- the detector emits the pattern for any
    number of qualifying events greater than zero -- but only if turnarounds
    were measurable at all. That precondition is checked by the caller against
    `pass_through`, so this returns None only when the metric is absent from a
    pattern that did fire.
    """
    for pattern in report.behavior_patterns:
        if _v(pattern.pattern_type) == "FAST_INBOUND_OUTBOUND":
            value = pattern.metrics.get("fast_passthrough_event_count")
            return int(value) if value is not None else None
    return 0


def _timing_block(report: InvestigationReport) -> list[str]:
    lines = _block(5, "TIMING ANALYSIS")
    t = report.temporal
    if t is None:
        lines.append("  (no temporal analysis available)")
        return lines
    pt = t.pass_through
    if pt is None:
        lines.append(_kv("TURNAROUND", f"{NA} (no inbound->outbound pair measurable)"))
    else:
        lines.append(_kv("TURNAROUNDS MEASURED", _num(pt.measured_events)))
        lines.append(_kv("FASTEST TURNAROUND", _duration(pt.min_seconds)))
        lines.append(_kv("MEDIAN TURNAROUND", _duration(pt.median_seconds)))
        lines.append(_kv("LONGEST TURNAROUND", _duration(pt.max_seconds)))
        lines.append(
            _kv("FAST PASS-THROUGHS", _num(_fast_passthrough_count(report)))
        )
        lines.append(
            _kv("INBOUND NOT FORWARDED", _num(pt.inbound_without_later_outbound))
        )
    lines.append(_kv("FIRST ACTIVITY", _time(t.first_seen)))
    lines.append(_kv("LAST ACTIVITY", _time(t.last_seen)))
    lines.append(_kv("ACTIVE DAYS", _num(t.active_day_count)))
    lines.append(_kv("MEDIAN GAP", _duration(t.median_gap_seconds)))
    lines.append(_kv("LONGEST IDLE", _duration(t.longest_idle_seconds)))
    return lines


# --------------------------------------------------------------------------
# [6] ASSET SUMMARY
# --------------------------------------------------------------------------


def _asset_block(report: InvestigationReport) -> list[str]:
    lines = _block(6, "ASSET SUMMARY")
    t = report.temporal
    per_asset = list(t.per_asset) if t is not None else []
    if not per_asset:
        lines.append("  (no assets observed)")
        return lines
    shown, note = _capped(per_asset, MAX_ASSETS, "assets")
    lines.append(
        _row(
            [
                ("ASSET", 14),
                ("TXS", 6),
                ("IN", 6),
                ("OUT", 6),
                ("TOTAL IN", 22),
                ("TOTAL OUT", 22),
                ("NET FLOW", 22),
            ]
        )
    )
    lines.append("  " + _THIN[:96])
    for a in shown:
        lines.append(
            _row(
                [
                    (_na(a.asset), 14),
                    (_num(a.transfer_count), 6),
                    (_num(a.inbound_count), 6),
                    (_num(a.outbound_count), 6),
                    (_amt(a.total_inbound), 22),
                    (_amt(a.total_outbound), 22),
                    (_amt(a.net_flow), 22),
                ]
            )
        )
    lines.extend(note)
    return lines


# --------------------------------------------------------------------------
# [7] INVESTIGATION FINDINGS
# --------------------------------------------------------------------------


def _findings_block(report: InvestigationReport) -> list[str]:
    lines = _block(7, "INVESTIGATION FINDINGS")
    patterns = report.behavior_patterns
    lines.append(_kv("TOTAL FINDINGS", _num(len(patterns))))
    if not patterns:
        lines.append("  (no behavioural indicator crossed its threshold)")
        return lines
    shown, note = _capped(patterns, MAX_FINDINGS, "findings")
    lines.append("")
    lines.append(
        _row(
            [
                ("FINDING", 34),
                ("METRIC", 30),
                ("VALUE", 14),
                ("THRESHOLD", 12),
                ("CLASS", 30),
            ]
        )
    )
    lines.append("  " + _THIN[:96])
    for p in shown:
        lines.append(
            _row(
                [
                    (_v(p.pattern_type), 34),
                    (_na(p.observed_metric), 30),
                    (_na(p.observed_value), 14),
                    (_na(p.threshold), 12),
                    (_v(p.classification), 30),
                ]
            )
        )
        window = f"{_time(p.first_seen)} .. {_time(p.last_seen)}"
        lines.append(f"      WINDOW: {window}")
        if p.related_addresses:
            # Two full addresses fit the rule width; a third pushes the line
            # past 145 characters. Addresses are never abbreviated, so the
            # count carries the rest and --json carries all of them.
            addresses = list(p.related_addresses)
            head = ", ".join(addresses[:_RELATED_INLINE])
            if len(addresses) > _RELATED_INLINE:
                head += f" (+{len(addresses) - _RELATED_INLINE} more)"
            lines.append(f"      ADDRESSES: {head}")
    lines.extend(note)
    return lines


# --------------------------------------------------------------------------
# [8] VASP / ENTITY MATCH
# --------------------------------------------------------------------------


def _vasp_block(report: InvestigationReport) -> list[str]:
    lines = _block(8, "VASP / ENTITY MATCH")
    attribution = report.attribution
    if attribution is None:
        lines.append(_kv("MATCH STATUS", NA))
        return lines
    lines.append(_kv("MATCH STATUS", _v(attribution.status)))
    if attribution.exact_identity_match is not None:
        lines.append(
            _kv("WALLET IS A KNOWN VASP", _v(attribution.exact_identity_match))
        )
    if not attribution.candidates:
        lines.append("  (no known VASP address matched)")
        return lines
    shown, note = _capped(attribution.candidates, MAX_ENTITY_ROWS, "candidates")
    lines.append("")
    lines.append(
        _row([("ENTITY", 26), ("MATCHED ADDRESS", 44), ("DIRECTION", 22), ("SOURCE", 22)])
    )
    lines.append("  " + _THIN[:96])
    for c in shown:
        lines.append(
            _row(
                [
                    (_na(c.vasp_name), 26),
                    (_na(c.matched_address), 44),
                    (_v(c.direction), 22),
                    (_v(c.source_type), 22),
                ]
            )
        )
    lines.extend(note)
    lines.extend(_fund_flow_lines(attribution))
    return lines


def _strongest_evidence(
    attribution: BidirectionalAttributionResult,
) -> tuple[Optional[Any], Optional[Any], str]:
    """The candidate whose traced path is the shortest, and that path.

    Shortest wins because a shorter chain has fewer unproven links: every
    additional intermediary is another address whose onward transfer is
    consistent with, but not proof of, fund continuity. Ties break on the
    matched address so two runs of the same investigation print the same path.
    """
    best: tuple[Optional[Any], Optional[Any], str] = (None, None, "")
    best_key: Optional[tuple[int, str]] = None
    for candidate in attribution.candidates:
        for evidence, label in (
            (candidate.outbound_evidence, "WALLET -> VASP"),
            (candidate.inbound_evidence, "VASP -> WALLET"),
        ):
            if evidence is None or not evidence.path_addresses:
                continue
            key = (evidence.hop_distance, candidate.matched_address)
            if best_key is None or key < best_key:
                best_key = key
                best = (candidate, evidence, label)
    return best


def _fund_flow_lines(attribution: BidirectionalAttributionResult) -> list[str]:
    """The hop-by-hop chain of the shortest traced path, with its evidence.

    Prints only when the search actually returned a path: the addresses come
    from `DirectionalEvidence.path_addresses`, which the targeted search only
    populates for a real directed, chronologically-consistent route through
    edges that were actually fetched. There is no code path here that can draw
    a chain from graph proximity, from an undirected relation, or from a
    requested hop count.

    ASCII arrows, not `->` glyphs: the Windows console folds anything outside
    cp1252 to `?`, so a Unicode down-arrow would print as a question mark in
    the exact place the reader needs to see direction.
    """
    candidate, evidence, label = _strongest_evidence(attribution)
    if candidate is None or evidence is None:
        return []

    path = evidence.path_addresses
    lines = ["", f"  FUND FLOW PATH  ({label}, {evidence.hop_distance} hop(s))"]

    for index, address in enumerate(path):
        marker = ""
        if index == 0:
            marker = "  <- investigated wallet" if label.startswith("WALLET") else "  <- VASP dataset address"
        elif index == len(path) - 1:
            marker = (
                f"  <- {candidate.vasp_name} (VASP dataset address)"
                if label.startswith("WALLET")
                else "  <- investigated wallet"
            )
        else:
            marker = "  (intermediary)"
        lines.append(f"  Hop {index}  {address}{marker}")
        if index >= len(path) - 1:
            continue
        # The edge FROM this address TO the next one. Indexed per hop, and
        # each list is read defensively: an older evidence record may carry
        # fewer entries than the path has hops, and a missing value must print
        # as absent rather than shift another hop's evidence onto this one.
        detail = []
        tx_hash = _index_or_none(evidence.tx_hashes, index)
        if tx_hash:
            detail.append(f"tx {tx_hash}")
        amount = _index_or_none(evidence.amounts, index)
        asset = _index_or_none(evidence.assets, index)
        if amount is not None:
            detail.append(f"{_amt(amount)} {asset or ''}".strip())
        stamp = _index_or_none(evidence.hop_timestamps, index)
        if stamp:
            detail.append(_time(stamp))
        lines.append("     |")
        if detail:
            lines.append(f"     |  {'  '.join(detail)}")
        lines.append("     v")

    if evidence.plausibility is not None:
        lines.append(_kv("PATH PLAUSIBILITY", _v(evidence.plausibility)))
    if evidence.alternative_path_count:
        lines.append(
            _kv(
                "ALTERNATIVE PATHS",
                f"{evidence.alternative_path_count} other route(s) traced in this "
                "direction; the chain above is one example, not the only "
                "connection",
            )
        )
    if evidence.hop_distance > 1:
        lines.append(
            "  NOTE: consecutive transfers along a chain are SUPPORTING EVIDENCE "
            "of a route, not proof that the same funds moved end to end."
        )
    return lines


def _index_or_none(values: list, index: int):
    """Element at `index`, or None when the list does not reach that far."""
    if index < 0 or index >= len(values):
        return None
    return values[index]


# --------------------------------------------------------------------------
# [9] RISK
# --------------------------------------------------------------------------


def _ml_lines(report: InvestigationReport) -> list[str]:
    """One or two lines of ML, never more.

    When ML produced no result the compact report states that and the reason,
    and stops. The full report keeps the complete labelling, training and
    explainability record.
    """
    ml = report.ml
    if ml is None:
        return [_kv("ML", NA)]
    approach = _v(ml.approach)
    if approach == "SUPERVISED" and ml.prediction is not None:
        p = ml.prediction
        out = [
            _kv(
                "ML",
                f"SUPERVISED  {_na(p.task)} = {_na(p.predicted_class)}",
            )
        ]
        if p.probability is not None:
            out.append(_kv("ML CONFIDENCE", f"{p.probability}"))
        out.append(
            _kv("ML MODEL", f"{_na(p.model_name)} {_na(p.model_version)}")
        )
        return out
    if approach == "UNSUPERVISED" and ml.outlier is not None:
        o = ml.outlier
        return [
            _kv("ML", f"UNSUPERVISED ({_na(o.method)})"),
            _kv(
                "ML OUTLIER",
                f"flagged={_na(o.flagged_as_outlier)} "
                f"percentile={_na(o.percentile_within_population)}",
            ),
        ]
    reason = None
    if ml.prediction is not None and ml.prediction.unavailable_reason:
        reason = ml.prediction.unavailable_reason
    elif ml.outlier is not None and ml.outlier.unavailable_reason:
        reason = ml.outlier.unavailable_reason
    return [_kv("ML", approach), _kv("REASON", _na(reason))]


def _risk_block(report: InvestigationReport) -> list[str]:
    lines = _block(9, "RISK")
    risk = report.risk
    if risk is None:
        lines.append(_kv("SCORE", NA))
        lines.extend(_ml_lines(report))
        return lines
    lines.append(_kv("SCORE", _na(risk.score)))
    lines.append(_kv("BAND", _v(risk.band)))
    lines.append(
        _kv(
            "BAND THRESHOLDS",
            f"MEDIUM >= {_na(risk.band_medium_threshold)}, "
            f"HIGH >= {_na(risk.band_high_threshold)}",
        )
    )
    lines.append(_kv("DATA COMPLETE", _v(risk.data_complete)))
    lines.extend(_ml_lines(report))
    if not risk.components:
        lines.append("")
        lines.append("  (no risk indicator triggered)")
        return lines
    shown, note = _capped(risk.components, MAX_RISK_ROWS, "indicators")
    lines.append("")
    # No THRESHOLD column: block 7 already prints the threshold for every
    # behavioural finding, and repeating it here would be the duplication the
    # compact report exists to avoid. VALUE stays, because two indicators of
    # the same name (two busy counterparties, say) are only distinguishable by
    # the value they were triggered on.
    lines.append(
        _row(
            [
                ("INDICATOR", 34),
                ("VALUE", 14),
                ("WEIGHT", 9),
                ("CONTRIB", 9),
                ("EVIDENCE", 20),
            ]
        )
    )
    lines.append("  " + _THIN[:96])
    for c in shown:
        lines.append(
            _row(
                [
                    (_v(c.indicator), 34),
                    (_na(c.observed_value), 14),
                    (_na(c.weight), 9),
                    (_na(c.contribution), 9),
                    (_v(c.evidence_class), 20),
                ]
            )
        )
    lines.extend(note)
    return lines


# --------------------------------------------------------------------------
# [10] DATA STATUS
# --------------------------------------------------------------------------


def _data_status_block(report: InvestigationReport) -> list[str]:
    lines = _block(10, "DATA STATUS")
    prov = report.provenance
    lines.append(_kv("DATA MODE", _v(prov.data_mode.value)))
    lines.append(_kv("PROVIDER", _na(prov.provider)))
    norm = report.normalization
    if norm is None:
        lines.append(_kv("RECORDS FETCHED", NA))
        lines.append(_kv("RECORDS RETAINED", NA))
        lines.append(_kv("REJECTED", NA))
        lines.append(_kv("DUPLICATES REMOVED", NA))
    else:
        lines.append(_kv("RECORDS FETCHED", _num(norm.input_count)))
        lines.append(_kv("RECORDS RETAINED", _num(norm.kept_count)))
        lines.append(_kv("REJECTED", _num(len(norm.rejected))))
        lines.append(_kv("DUPLICATES REMOVED", _num(norm.duplicates_removed)))
    lines.append(_kv("GRAPH EDGES", _num(report.transfer_count)))
    # Provider responses can be served from the local response cache even on a
    # REAL run. That is not the same thing as CACHED REAL DATA (a graph
    # reloaded from disk), but a reader is entitled to know which requests
    # actually went to the network on this run.
    stats = prov.cache_stats or {}
    if stats:
        lines.append(
            _kv(
                "PROVIDER CACHE",
                ", ".join(f"{k}={v}" for k, v in sorted(stats.items())),
            )
        )
    lines.append(_kv("OBSERVED HOP DEPTH", _num(prov.observation_depth)))
    lines.append(_kv("REQUESTED HOP DEPTH", _num(report.parameters.get("max_hops"))))
    # Acquisition facts, so "observed 4 hops" can be checked against the work
    # that was actually done rather than taken on trust. Absent on the cached
    # paths, which did not acquire anything, and printed as N/A there.
    if prov.addresses_fetched is not None:
        lines.append(
            _kv(
                "ADDRESSES FETCHED",
                f"{_num(prov.addresses_fetched)} of "
                f"{_num(prov.addresses_discovered)} discovered",
            )
        )
        lines.append(
            _kv(
                "HOP LEVELS EXPANDED",
                f"{_num(prov.hops_expanded)} of "
                f"{_num(prov.requested_expansion_hops)} requested "
                f"(deepest address discovered at hop {_num(prov.max_hop_reached)})",
            )
        )
        lines.append(_kv("ACQUISITION STOPPED", _na(prov.expansion_stop_reason)))
    lines.append(_kv("DATA COMPLETE", _v(prov.data_complete)))
    return lines


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def render_compact_report(report: InvestigationReport) -> str:
    """Renders the ten-block compact report as one ASCII-safe string."""
    lines: list[str] = [
        _RULE,
        "  WALLET INVESTIGATION -- COMPACT REPORT",
        _RULE,
    ]
    lines.extend(_wallet_block(report))
    lines.extend(_summary_block(report))
    lines.extend(_counterparties_block(report))
    lines.extend(_activity_block(report))
    lines.extend(_timing_block(report))
    lines.extend(_asset_block(report))
    lines.extend(_findings_block(report))
    lines.extend(_vasp_block(report))
    lines.extend(_risk_block(report))
    lines.extend(_data_status_block(report))
    lines.append("")
    lines.append(_RULE)
    lines.append(f"  INVESTIGATION COMPLETE -- {report.wallet} on {report.chain}")
    lines.append(
        f"  DATA MODE: {report.provenance.data_mode.value}   "
        f"DURATION: {report.duration_seconds}s"
    )
    lines.append("  Full report: --full-report   Machine-readable: --json")
    lines.append(_RULE)
    return to_ascii("\n".join(lines))


def print_compact_report(
    report: InvestigationReport, stream: Optional[TextIO] = None
) -> None:
    """Renders to a stream, forcing UTF-8 first."""
    target = stream if stream is not None else sys.stdout
    configure_stdout(target)
    print(render_compact_report(report), file=target)


__all__ = [
    "render_compact_report",
    "print_compact_report",
    "WIDTH",
    "MAX_COUNTERPARTIES",
    "MAX_TRANSACTIONS",
    "MAX_ASSETS",
    "MAX_FINDINGS",
    "MAX_RISK_ROWS",
]
