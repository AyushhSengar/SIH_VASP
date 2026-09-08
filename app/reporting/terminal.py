"""
TERMINAL RENDERER — the nine-section investigation report.

Rendering is deliberately separate from analysis: this module reads an
`InvestigationReport` and writes text. It computes nothing, decides nothing, and
never reaches back into the graph. Anything printed here is a field some
analysis module already committed to, which is what makes the printed report and
the `--json` output guaranteed to agree.

TWO RULES THIS RENDERER ENFORCES IN ITS WORDING
--------------------------------------------------------------------------
1. A path is a TRANSACTION PATH, never a proof that the same funds moved end to
   end. Section 2 says so on every route it draws.
2. An empty section is labelled with the reason it is empty. "No candidates
   found because the search completed and matched nothing" and "no candidates
   listed because the search ran out of budget" are different findings, and a
   reader must never have to guess which one they are looking at.

ENCODING
--------------------------------------------------------------------------
Windows consoles default to a legacy code page (cp1252 here), and the analysis
modules' explanatory text contains em-dashes and arrows. Printing those through
cp1252 produces mojibake at best and a UnicodeEncodeError at worst — which would
kill a report mid-render. `configure_stdout` reconfigures the stream to UTF-8
with `errors="replace"` so that a glyph problem can degrade a character but can
never abort an investigation. All box drawing here is plain ASCII regardless.
"""

from __future__ import annotations

import sys
import unicodedata
from enum import Enum
from typing import Any, Optional, TextIO

from app.investigation.pipeline import DataMode, InvestigationReport

WIDTH = 78
_RULE = "=" * WIDTH
_THIN = "-" * WIDTH

# --------------------------------------------------------------------------
# DISPLAY CAPS -- how many items of a repeating list reach the terminal.
#
# These bound the PRINTED report only. The analysis layer stays complete and
# `--json` still carries every item, because discarding evidence to fit a
# screen would defeat the point of the tool. What a cap changes is only how
# much of a long list a human is shown at once.
#
# The caps exist because real wallets are not small. On a mainnet wallet with
# eight years of history the uncapped report was 23,357 lines, of which 16,530
# were a counterparty list -- unreadable, and it buried the attribution and
# ML sections thousands of lines below the fold. A capped list is only
# defensible if it is honest about being capped, so every one of them prints
# the total and the number withheld, and names --json as the complete record.
# Nothing is ever dropped without a line saying so.
#
# A cap on the NUMBER of items is only half the problem. The other half is
# repetition: most of a real wallet's findings are the same finding measured
# against a different counterparty, and printing the full twenty-five-line
# evidence block for each of sixty HIGH_FREQUENCY_COUNTERPARTY findings spends
# fifteen hundred lines to convey one fact plus sixty addresses. So the
# repeating sections itemise the FIRST finding of each distinct kind in full
# and list further findings of an already-itemised kind compactly -- their
# metric, threshold, weight and classification are identical by construction,
# so the address and the measured value are the only new information in them.
#: Direct counterparties (section 3). Ordered busiest-first by the analysis
#: layer, so the cap keeps the ones an investigator would look at first.
MAX_COUNTERPARTIES_SHOWN = 25
#: Behavioural indicator findings given a full evidence block (section 5).
#: Spent on the first finding of each distinct indicator type, so a wallet with
#: fewer than this many types has every one of them itemised in full.
MAX_BEHAVIOR_INDICATORS_SHOWN = 20
#: Compact one-line rows per repeated indicator type (section 5).
MAX_BEHAVIOR_REPEATS_PER_TYPE = 10
#: Per-asset amount breakdowns (section 6). A wallet that has been airdropped
#: spam tokens for years holds hundreds of assets, nearly all with one
#: transfer; the busiest handful is the part worth reading.
MAX_ASSETS_SHOWN = 15
#: Risk contributions given a full itemisation (section 8). Ordered
#: largest-first, and the printed subtotal always reconciles against the full
#: score.
MAX_RISK_CONTRIBUTIONS_SHOWN = 15
#: Compact one-line rows per repeated risk indicator (section 8). Each row
#: still carries its own contribution, so the arithmetic reconciles.
MAX_RISK_REPEATS_PER_INDICATOR = 10


def _omitted(shown: int, total: int, noun: str, indent: str = "    ") -> list[str]:
    """The line that makes a display cap honest.

    Returns nothing when nothing was withheld, so a short list reads cleanly.
    """
    if total <= shown:
        return []
    return _wrap(
        f"... and {total - shown} more {noun} not shown here ({total} in total). "
        f"The complete set is in the --json output; nothing has been discarded.",
        indent=indent,
    )


#: Asset symbols named inline on a summary line before the rest are counted.
_ASSETS_INLINE = 8


def _asset_list(assets: Any) -> str:
    """Names a few assets and counts the rest.

    A wallet that has collected airdropped spam tokens for years holds
    hundreds of symbols. Printing them all turned a one-line summary into a
    twelve-line block on every row of a list, so the count carries the fact
    and the names carry the flavour. The full list is in --json.
    """
    if not assets:
        return "(none)"
    items = list(assets)
    if len(items) <= _ASSETS_INLINE:
        return ", ".join(items)
    return (
        ", ".join(items[:_ASSETS_INLINE])
        + f", and {len(items) - _ASSETS_INLINE} more ({len(items)} distinct assets)"
    )


#: Typographic characters that appear in the analysis modules' explanatory text,
#: mapped to ASCII. The modules are written with proper typography in their
#: docstrings and messages; the terminal is the wrong place for it.
_ASCII_FOLD = {
    0x2014: " - ",  # em dash
    0x2013: "-",  # en dash
    0x2018: "'",
    0x2019: "'",
    0x201C: '"',
    0x201D: '"',
    0x2026: "...",
    0x2192: "->",
    0x2190: "<-",
    0x2265: ">=",
    0x2264: "<=",
    0x00A0: " ",  # non-breaking space
    0x00B1: "+/-",
    0x00D7: "x",
    0x2022: "*",
}


def to_ascii(text: str) -> str:
    """Folds a report to pure ASCII so no console code page can mangle it.

    Writing UTF-8 bytes to a Windows console still running code page 1252 -- the
    default here -- renders an em dash as three garbage characters. Forcing the
    stream to UTF-8 fixes the encoding but not the console's interpretation of
    it, so the only way to guarantee a readable report on an unknown terminal is
    to emit characters every code page agrees on.

    Known typography is mapped deliberately. Anything else is decomposed with
    NFKD, which splits an accented letter into its base letter plus a combining
    mark; the marks are then dropped so 'cafe' survives rather than becoming
    'cafe?'. Whatever still has no ASCII form becomes '?', which is visibly a
    substitution rather than silent corruption.
    """
    folded = text.translate(_ASCII_FOLD)
    if folded.isascii():
        return folded
    decomposed = unicodedata.normalize("NFKD", folded)
    without_marks = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return without_marks.encode("ascii", "replace").decode("ascii")


def configure_stdout(stream: Optional[TextIO] = None) -> None:
    """Forces UTF-8 on the output stream where the platform allows it.

    Called by the CLI before rendering. Wrapped because `reconfigure` exists
    only on `io.TextIOWrapper` — under pytest's captured stdout or a pipe it may
    be absent, and a renderer must not fail because it could not improve the
    encoding.
    """
    target = stream if stream is not None else sys.stdout
    reconfigure = getattr(target, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):  # pragma: no cover - detached/closed stream
        pass


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------


def _v(value: Any) -> str:
    """Enum-safe stringification, so `.value` is never printed as `Enum.X`."""
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _short(address: Optional[str], keep: int = 6) -> str:
    """Abbreviates an address for a diagram, never for an evidence field.

    Evidence lines always print the full address; only the ASCII route diagram
    abbreviates, because a 42-character string per hop makes the shape of the
    route unreadable, which is the diagram's entire purpose.
    """
    if not address:
        return "(unknown)"
    if len(address) <= keep * 2 + 2:
        return address
    return f"{address[: keep + 2]}...{address[-keep:]}"


def _amount(value: Any) -> str:
    """Renders an amount without lying about precision.

    Blockchain amounts are exact decimals; formatting one to a fixed number of
    places would silently invent or destroy precision, so the decimal string is
    printed as-is and only trailing zeros are trimmed.
    """
    if value is None:
        return "(amount unavailable)"
    text = str(value)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _ts(value: Any) -> str:
    """Formats a unix timestamp as UTC, marking absence explicitly."""
    if value in (None, 0):
        return "(no timestamp)"
    try:
        import time

        return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(int(value)))
    except (ValueError, OSError, OverflowError):  # pragma: no cover
        return f"(unparseable: {value})"


def _counts(mapping: Any) -> str:
    """Renders a label->count mapping as `A=3, B=7`.

    A raw dict repr in a report reads as debug output, and the braces and
    quotes make the numbers harder to find than the punctuation.
    """
    if not mapping:
        return "(none)"
    if not isinstance(mapping, dict):  # pragma: no cover - defensive
        return str(mapping)
    return ", ".join(f"{key}={value}" for key, value in sorted(mapping.items()))


_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _weekday(value: Any) -> str:
    """Names a `datetime.weekday()` index. "busiest weekday: 5" is unreadable
    and silently ambiguous (Monday-first or Sunday-first?); "5 (Sat)" is not.
    """
    if value is None:
        return "(not available)"
    try:
        return f"{int(value)} ({_WEEKDAYS[int(value)]})"
    except (ValueError, TypeError, IndexError):  # pragma: no cover - defensive
        return str(value)


def _wrap(text: str, indent: str = "  ", width: int = WIDTH) -> list[str]:
    """Word-wraps a paragraph. The analysis modules emit long explanations and
    an unwrapped 400-character line is functionally unreadable in a terminal.

    Folds to ASCII *before* wrapping, because an em dash becomes three
    characters and folding afterwards would push wrapped lines past `width`.

    Long words are never split. A transaction hash, an address or a file path
    chopped across two lines cannot be copied or searched for, and these
    paragraphs carry exactly those. Such a line overruns `width` instead --
    a deliberate trade of tidiness for usable evidence.
    """
    import textwrap

    folded = to_ascii(str(text))
    return textwrap.wrap(
        folded,
        width=width,
        initial_indent=indent,
        subsequent_indent=indent + "  ",
        break_long_words=False,
        break_on_hyphens=False,
    ) or [indent + folded]


def _bullets(items: list[Any], indent: str = "  ", marker: str = "- ") -> list[str]:
    out: list[str] = []
    for item in items:
        wrapped = _wrap(str(item), indent=indent + " " * len(marker))
        first = wrapped[0]
        out.append(indent + marker + first[len(indent) + len(marker) :])
        out.extend(wrapped[1:])
    return out


def _section(number: int, title: str) -> list[str]:
    return ["", _RULE, f"  SECTION {number}: {title}", _RULE]


def _kv(label: str, value: Any, indent: str = "  ", pad: int = 30) -> str:
    return f"{indent}{label + ':':<{pad}} {value}"


# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------


def _render_header(report: InvestigationReport) -> list[str]:
    """Header. DATA MODE is printed here, before any finding, deliberately.

    A reader must know whether they are looking at live or cached real data
    before they read a single conclusion drawn from it.
    """
    mode = report.provenance.data_mode
    lines = [
        _RULE,
        "  BLOCKCHAIN WALLET INVESTIGATION REPORT",
        _RULE,
        _kv("WALLET", report.wallet),
        _kv("CHAIN", report.chain),
        _kv("DATA MODE", mode.value),
    ]
    # Prose, not an identifier: wrapped rather than run past the right margin.
    # File paths below stay on one line deliberately -- a wrapped path cannot
    # be copied back into a shell.
    lines.extend(
        _wrap(f"DATA SOURCE: {report.provenance.source_description}", indent="  ")
    )
    if report.provenance.provider:
        lines.append(_kv("PROVIDER", report.provenance.provider))
    if report.provenance.graph_path:
        lines.append(_kv("GRAPH FILE", report.provenance.graph_path))
    if report.provenance.transfers_path:
        lines.append(_kv("TRANSFERS FILE", report.provenance.transfers_path))
    lines += [
        _kv("INVESTIGATION ID", report.investigation_id),
        _kv("STARTED (UTC)", report.started_at_utc),
        _kv("DURATION", f"{report.duration_seconds}s"),
        _kv(
            "DATA COMPLETE",
            "YES" if report.provenance.data_complete else "NO (see limitations)",
        ),
    ]

    if mode == DataMode.CACHED_REAL_DATA:
        lines.append("")
        lines.extend(
            _wrap(
                "NOTE: this report was produced from CACHED REAL DATA. The "
                "transactions are genuine on-chain records acquired in an "
                "earlier run and reloaded from disk. They are NOT live: any "
                "activity after the cache was written is absent from every "
                "section below.",
                indent="  ",
            )
        )

    params = ", ".join(f"{k}={v}" for k, v in sorted(report.parameters.items()))
    lines.append("")
    lines.extend(_wrap(f"PARAMETERS: {params}", indent="  "))

    if report.warnings:
        lines.append("")
        lines.append("  WARNINGS")
        lines.extend(_bullets(report.warnings, indent="  ", marker="! "))
    return lines


# --------------------------------------------------------------------------
# section 1 -- blockchain data summary
# --------------------------------------------------------------------------


def _render_data_summary(report: InvestigationReport) -> list[str]:
    lines = _section(1, "BLOCKCHAIN DATA SUMMARY")

    if report.provenance.streams:
        lines.append("  ACQUISITION STREAMS")
        lines.extend(_bullets(report.provenance.streams))
        lines.append("")

    if report.provenance.cache_stats:
        stats = ", ".join(
            f"{k}={v}" for k, v in sorted(report.provenance.cache_stats.items())
        )
        lines.append(_kv("PROVIDER CACHE", stats))

    norm = report.normalization
    if norm is not None:
        lines.append("  NORMALIZATION")
        lines.append(_kv("records in", norm.input_count, indent="    "))
        lines.append(_kv("records kept", norm.kept_count, indent="    "))
        lines.append(_kv("duplicates removed", norm.duplicates_removed, indent="    "))
        lines.append(_kv("rejected", norm.rejected_count, indent="    "))
        lines.append(
            _kv("missing timestamps", norm.missing_timestamp_count, indent="    ")
        )
        lines.append(
            _kv(
                "missing token metadata",
                norm.missing_token_metadata_count,
                indent="    ",
            )
        )
        lines.append(
            _kv(
                "reconciled",
                "YES" if norm.reconciled else "NO -- counts do not balance",
                indent="    ",
            )
        )
        if norm.reason_counts:
            lines.append("    rejection reasons:")
            lines.extend(
                _bullets(
                    [f"{k}: {v}" for k, v in sorted(norm.reason_counts.items())],
                    indent="      ",
                )
            )
        lines.append("")
    else:
        lines.extend(
            _wrap(
                "NORMALIZATION: not reported. This run loaded a prebuilt graph "
                "rather than raw provider records, so the normalization stage "
                "ran when that graph was originally built, not now.",
                indent="  ",
            )
        )
        lines.append("")

    summary = report.graph_summary
    lines.append("  TRANSACTION GRAPH")
    if summary is not None:
        lines.append(_kv("nodes (addresses)", summary.node_count, indent="    "))
        lines.append(_kv("edges (transfers)", summary.edge_count, indent="    "))
        lines.append(_kv("native edges", summary.native_edge_count, indent="    "))
        lines.append(_kv("token edges", summary.token_edge_count, indent="    "))
        lines.append(_kv("self-loop edges", summary.self_loop_edges, indent="    "))
        lines.append(
            _kv(
                "contract creations skipped",
                summary.contract_creation_skipped,
                indent="    ",
            )
        )
        lines.append(_kv("density", round(summary.density, 8), indent="    "))
        lines.append(
            _kv("earliest transfer", _ts(summary.earliest_timestamp), indent="    ")
        )
        lines.append(
            _kv("latest transfer", _ts(summary.latest_timestamp), indent="    ")
        )
        lines.append(
            _kv(
                "edge accounting",
                "RECONCILED" if summary.reconciled else "MISMATCH -- see notes",
                indent="    ",
            )
        )
        if summary.notes:
            lines.extend(_bullets(summary.notes, indent="    "))
    else:
        lines.append(_kv("edges (transfers)", report.transfer_count, indent="    "))
        lines.extend(
            _wrap(
                "Full build statistics are unavailable because this graph was "
                "reloaded rather than built in this run.",
                indent="    ",
            )
        )

    lines.append("")
    lines.append(
        _kv(
            "WALLET PRESENT IN GRAPH",
            "YES" if report.wallet_in_graph else "NO",
        )
    )
    if not report.wallet_in_graph:
        lines.extend(
            _wrap(
                "The investigated address does not appear on any edge of this "
                "graph. Every section below is therefore empty because there is "
                "no data, NOT because the address was examined and cleared.",
                indent="  ",
            )
        )
    return lines


# --------------------------------------------------------------------------
# section 2 -- fund-flow analysis
# --------------------------------------------------------------------------


def _render_route(
    evidence: Any,
    wallet: str,
    matched_address: str,
    indent: str = "    ",
) -> list[str]:
    """Draws one route as an ASCII chain with per-hop evidence.

    Hop roles are decided by comparing each address against the investigated
    wallet and the matched dataset address, NOT by position. An inbound route
    runs VASP -> wallet and an outbound route runs wallet -> VASP, so labelling
    hop 0 as "the wallet" would be wrong exactly half the time.

    The parallel arrays on `DirectionalEvidence` are indexed defensively: an
    older cached graph can carry fewer block numbers than hops, and a renderer
    that raised IndexError on real data would be useless. A missing value is
    printed as missing.
    """
    lines: list[str] = []
    addresses = list(evidence.path_addresses or [])
    hashes = list(evidence.tx_hashes or [])
    stamps = list(evidence.hop_timestamps or [])
    amounts = list(evidence.amounts or [])
    assets = list(evidence.assets or [])
    blocks = list(evidence.block_numbers or [])

    wallet = (wallet or "").lower()
    matched = (matched_address or "").lower()

    def at(seq: list, index: int) -> Any:
        return seq[index] if index < len(seq) else None

    for index, address in enumerate(addresses):
        lowered = str(address).lower()
        if lowered == wallet:
            role = "  <- investigated wallet"
        elif lowered == matched:
            role = "  <- VASP dataset address"
        else:
            role = "  (intermediary)"
        lines.append(f"{indent}[HOP {index}] {address}{role}")

        if index >= len(addresses) - 1:
            break

        tx = at(hashes, index)
        lines.append(f"{indent}    |")
        lines.append(
            f"{indent}    |  tx    {tx if tx else '(tx hash unavailable)'}"
        )
        lines.append(f"{indent}    |  time  {_ts(at(stamps, index))}")
        amount_text = _amount(at(amounts, index))
        asset_text = at(assets, index) or "(asset unavailable)"
        lines.append(f"{indent}    |  value {amount_text} {asset_text}")
        block = at(blocks, index)
        lines.append(
            f"{indent}    |  block {block if block is not None else '(not recorded in this graph)'}"
        )
        lines.append(f"{indent}    v")

    return lines


def _render_evidence_block(
    evidence: Any,
    label: str,
    wallet: str,
    matched_address: str,
    indent: str = "  ",
) -> list[str]:
    lines = [f"{indent}{label}"]
    lines.append(
        _kv(
            "hops",
            f"{evidence.hop_distance} "
            f"({'DIRECT' if evidence.hop_distance == 1 else 'INDIRECT'})",
            indent=indent + "  ",
            pad=26,
        )
    )
    lines.append(
        _kv("evidence tier", _v(evidence.evidence_tier), indent=indent + "  ", pad=26)
    )
    if evidence.path_duration_seconds is not None:
        lines.append(
            _kv(
                "elapsed across route",
                f"{evidence.path_duration_seconds}s",
                indent=indent + "  ",
                pad=26,
            )
        )
    if evidence.alternative_path_count:
        lines.append(
            _kv(
                "alternative routes",
                evidence.alternative_path_count,
                indent=indent + "  ",
                pad=26,
            )
        )
    lines.append("")
    lines.extend(
        _render_route(evidence, wallet, matched_address, indent=indent + "  ")
    )

    plausibility = evidence.plausibility
    if plausibility is not None:
        lines.append("")
        lines.append(
            _kv(
                "ROUTE PLAUSIBILITY",
                _v(plausibility.grade),
                indent=indent + "  ",
                pad=26,
            )
        )
        lines.extend(_wrap(plausibility.interpretation, indent=indent + "    "))
        if plausibility.concerns:
            lines.append(f"{indent}    concerns that weaken this route:")
            for concern in plausibility.concerns:
                lines.extend(
                    _wrap(
                        f"{_v(concern.concern)} at hop {concern.hop_index} "
                        f"({concern.address}): observed {concern.observed}, "
                        f"threshold {concern.threshold}. {concern.explanation}",
                        indent=indent + "      ",
                    )
                )
        if plausibility.hub_intermediaries:
            for hub in plausibility.hub_intermediaries:
                lines.extend(
                    _wrap(
                        f"hub intermediary {hub.address} at hop {hub.hop_index} "
                        f"has in-degree {hub.in_degree} and out-degree "
                        f"{hub.out_degree}; value entering it cannot be followed "
                        "to any particular exit.",
                        indent=indent + "      ",
                    )
                )
    return lines


def _render_fund_flow(report: InvestigationReport) -> list[str]:
    lines = _section(2, "FUND-FLOW ANALYSIS (TRANSACTION PATHS)")
    lines.extend(
        _wrap(
            "IMPORTANT: each route below is a TRANSACTION PATH -- a sequence of "
            "real transfers that connects two addresses through the graph. It is "
            "NOT proof that the same units of value travelled the whole route. "
            "An intermediary can commingle, swap, hold, or split funds, so "
            "A -> B -> C does not establish that C received A's money.",
            indent="  ",
        )
    )
    lines.append("")

    attribution = report.attribution
    if attribution is None:
        lines.append("  No path search was performed.")
        return lines

    lines.append(_kv("SEARCH STATUS", _v(attribution.status)))
    lines.append(_kv("MAX HOPS SEARCHED", attribution.max_hops))
    depth = report.provenance.observation_depth
    if depth is not None:
        # Printed next to MAX HOPS on purpose: the pair is only meaningful
        # together. A hop limit above the data's radius is a scope the dataset
        # cannot honour, and the reader has to see both numbers to know that.
        note = "" if depth >= attribution.max_hops else "  <-- shallower than MAX HOPS"
        lines.append(_kv("DATA OBSERVED TO", f"{depth} hop(s){note}"))

    for accounting in (attribution.outbound_accounting, attribution.inbound_accounting):
        if accounting is None:
            continue
        lines.append("")
        lines.append(f"  {_v(accounting.direction)} SEARCH")
        lines.append(
            _kv("targets searched", accounting.targets_searched, indent="    ", pad=26)
        )
        lines.append(
            _kv("targets reachable", accounting.targets_reachable, indent="    ", pad=26)
        )
        lines.append(
            _kv("edges explored", accounting.edges_explored, indent="    ", pad=26)
        )
        lines.append(
            _kv(
                "reachable / viable nodes",
                f"{accounting.reachable_node_count} / {accounting.viable_node_count}",
                indent="    ",
                pad=26,
            )
        )
        lines.append(
            _kv(
                "search completed",
                "YES"
                if accounting.complete
                else f"NO -- {accounting.incomplete_reason or 'reason not recorded'}",
                indent="    ",
                pad=26,
            )
        )
        if accounting.time_window_start:
            lines.append(
                _kv(
                    "time window from",
                    _ts(accounting.time_window_start),
                    indent="    ",
                    pad=26,
                )
            )
            lines.append(
                _kv(
                    "edges outside window",
                    accounting.edges_excluded_by_time_window,
                    indent="    ",
                    pad=26,
                )
            )
        if accounting.notes:
            lines.extend(_bullets(accounting.notes, indent="    "))

    routes = 0
    for candidate in attribution.candidates:
        for evidence, label in (
            (candidate.outbound_evidence, "WALLET -> VASP (outbound)"),
            (candidate.inbound_evidence, "VASP -> WALLET (inbound)"),
        ):
            if evidence is None:
                continue
            routes += 1
            lines.append("")
            lines.append(_THIN)
            lines.append(
                f"  ROUTE {routes}: {candidate.vasp_name} "
                f"[{_v(candidate.direction)}]"
            )
            lines.append(_THIN)
            lines.extend(
                _render_evidence_block(
                    evidence, label, report.wallet, candidate.matched_address
                )
            )

    if routes == 0:
        lines.append("")
        depth = report.provenance.observation_depth
        horizon_short = depth is not None and depth < attribution.max_hops
        if _v(attribution.status) == "INCONCLUSIVE" and horizon_short:
            lines.extend(
                _wrap(
                    "NO ROUTES LISTED. Within the "
                    f"{depth} hop(s) this dataset observes, no route to a "
                    "known-VASP address exists, and that much IS a complete "
                    "finding. Beyond it nothing can be said: not every address "
                    f"at hop {depth} had its own transactions acquired, so no "
                    "edges lead onward from there and a "
                    f"{depth + 1}-hop route could not appear here even if it "
                    "exists on-chain. Raising --max-hops alone will not help; "
                    "acquisition has to reach the remaining addresses at that "
                    "hop (EXPANSION_MAX_ADDRESSES / "
                    "EXPANSION_MAX_ADDRESSES_PER_HOP).",
                    indent="  ",
                )
            )
        elif _v(attribution.status) == "INCONCLUSIVE":
            lines.extend(
                _wrap(
                    "NO ROUTES LISTED, AND THIS IS NOT A NEGATIVE FINDING. The "
                    "search hit its exploration budget before completing, so "
                    "whether a route exists is unknown. Re-run with a larger "
                    "--max-paths / a narrower --time-window before concluding "
                    "anything.",
                    indent="  ",
                )
            )
        else:
            lines.extend(
                _wrap(
                    "No transaction path was found between this wallet and any "
                    "address in the known-VASP dataset within the searched "
                    "depth, and the search completed. This is bounded by the "
                    f"dataset's coverage ({report.seed_entry_count} known "
                    "address(es)): it means 'none of those', not 'no exchange'.",
                    indent="  ",
                )
            )
    return lines


# --------------------------------------------------------------------------
# section 3 -- VASP attribution
# --------------------------------------------------------------------------


def _render_attribution(report: InvestigationReport) -> list[str]:
    lines = _section(3, "VASP ATTRIBUTION")
    attribution = report.attribution

    lines.append(_kv("KNOWN-VASP DATASET", report.seed_dataset_path))
    lines.append(_kv("DATASET ENTRIES", report.seed_entry_count))
    if report.seed_provenance_counts:
        lines.append("  DATASET PROVENANCE BREAKDOWN")
        lines.extend(
            _bullets(
                [f"{k}: {v}" for k, v in report.seed_provenance_counts.items()],
                indent="    ",
            )
        )
    lines.append("")
    lines.extend(
        _wrap(
            "Attribution is by EXACT, case-insensitive address match only. No "
            "fuzzy, prefix, substring or 'similar-looking' matching is performed "
            "anywhere in this system, because a near-miss on an address is not "
            "weak evidence -- it is a different address.",
            indent="  ",
        )
    )
    lines.append("")

    if attribution is None:
        lines.append("  No attribution was performed.")
        return lines

    lines.append(_kv("ATTRIBUTION STATUS", _v(attribution.status)))
    lines.append(_kv("CANDIDATES", len(attribution.candidates)))

    if attribution.seed_contains_synthetic:
        lines.extend(
            _wrap(
                "! The loaded dataset contains SYNTHETIC_DEMO entries. Any "
                "candidate below derived from one is not real-world evidence.",
                indent="  ",
            )
        )

    identity = attribution.exact_identity_match
    if identity is not None:
        lines.append("")
        lines.extend(
            _wrap(
                f"EXACT IDENTITY MATCH: the investigated address IS a dataset "
                f"address for {identity.vasp_name} "
                f"({_v(identity.source_type)}). {identity.note}",
                indent="  ",
            )
        )

    for number, candidate in enumerate(attribution.candidates, start=1):
        lines.append("")
        lines.append(_THIN)
        lines.append(f"  CANDIDATE {number}: {candidate.vasp_name}")
        lines.append(_THIN)
        lines.append(_kv("matched address", candidate.matched_address, indent="    ", pad=26))
        lines.append(_kv("entity type", _v(candidate.entity_type), indent="    ", pad=26))
        lines.append(_kv("chain", candidate.chain, indent="    ", pad=26))
        lines.append(_kv("direction", _v(candidate.direction), indent="    ", pad=26))
        lines.append(_kv("wallet role", _v(candidate.wallet_role), indent="    ", pad=26))
        lines.append(
            _kv("evidence status", _v(candidate.evidence_status), indent="    ", pad=26)
        )
        lines.append("")
        lines.append("    PROVENANCE OF THE DATASET ENTRY")
        lines.append(_kv("source type", _v(candidate.source_type), indent="      ", pad=24))
        lines.append(_kv("source", candidate.seed_source, indent="      ", pad=24))
        if candidate.seed_source_url:
            lines.append(
                _kv("source URL", candidate.seed_source_url, indent="      ", pad=24)
            )
        lines.append(
            _kv(
                "verification",
                _v(candidate.verification_status),
                indent="      ",
                pad=24,
            )
        )
        lines.append(
            _kv(
                "evidence class",
                _v(candidate.source_evidence_type),
                indent="      ",
                pad=24,
            )
        )
        if candidate.seed_confidence_note:
            lines.extend(_wrap(candidate.seed_confidence_note, indent="      "))

        if candidate.supporting_behavioral_patterns:
            lines.append("")
            lines.append("    SUPPORTING BEHAVIOURAL OBSERVATIONS (not attribution)")
            lines.extend(
                _bullets(
                    [_v(p) for p in candidate.supporting_behavioral_patterns],
                    indent="      ",
                )
            )
        if candidate.limitations:
            lines.append("")
            lines.append("    LIMITATIONS OF THIS CANDIDATE")
            lines.extend(_bullets(candidate.limitations, indent="      "))

    if attribution.connected_but_no_valid_path:
        lines.append("")
        lines.append("  CONNECTED BUT NO VALID DIRECTED PATH (NOT attributed)")
        for item in attribution.connected_but_no_valid_path:
            lines.extend(
                _wrap(
                    f"{item.vasp_name} ({item.vasp_address}) is "
                    f"{item.graph_distance} hop(s) away when "
                    f"{_v(item.direction_attempted)} was attempted. {item.note}",
                    indent="    ",
                )
            )

    if attribution.related_by_undirected_graph_only:
        lines.append("")
        lines.append("  RELATED BY UNDIRECTED CONNECTIVITY ONLY (NOT attributed)")
        lines.extend(
            _wrap(
                "These addresses share a connected component with the wallet but "
                "no directed transfer path exists. Shared component membership is "
                "NOT evidence of a relationship and is listed only so it is not "
                "mistaken for a finding later.",
                indent="    ",
            )
        )
        for item in attribution.related_by_undirected_graph_only:
            lines.append(
                f"      {item.vasp_name} ({item.vasp_address}) at undirected "
                f"distance {item.undirected_distance}"
            )

    if attribution.notes:
        lines.append("")
        lines.append("  SEARCH NOTES")
        lines.extend(_bullets(attribution.notes, indent="    "))

    # entity resolution
    lines.append("")
    lines.append("  ENTITY RESOLUTION")
    if not report.entities:
        lines.append("    No entities resolved (no candidates to group).")
    for entity in report.entities:
        lines.append("")
        lines.append(f"    {entity.entity_name} [{_v(entity.entity_type)}]")
        lines.append(
            _kv(
                "strongest provenance",
                _v(entity.strongest_source_type),
                indent="      ",
                pad=28,
            )
        )
        lines.append(
            _kv(
                "closest hop distance",
                entity.strongest_hop_distance,
                indent="      ",
                pad=28,
            )
        )
        lines.append(
            _kv("directions", ", ".join(_v(d) for d in entity.directions), indent="      ", pad=28)
        )
        lines.append("      matched addresses (exact):")
        for address in entity.matched_addresses:
            lines.append(f"        {address}")
        if entity.dataset_addresses_not_matched:
            lines.append(
                "      other addresses this operator discloses, NOT reached here:"
            )
            for address in entity.dataset_addresses_not_matched:
                lines.append(f"        {address}")
        lines.extend(_wrap(f"grouping basis: {entity.grouping_basis}", indent="      "))
        if entity.limitations:
            lines.extend(_bullets(entity.limitations, indent="      "))

    # counterparties
    lines.append("")
    lines.append("  DIRECT COUNTERPARTIES (1 hop)")
    if not report.counterparties:
        lines.append("    None: the wallet has no direct counterparties in this graph.")
    else:
        lines.append(
            _kv("total", str(len(report.counterparties)), indent="    ", pad=24)
        )
        if len(report.counterparties) > MAX_COUNTERPARTIES_SHOWN:
            lines.extend(
                _wrap(
                    f"showing the {MAX_COUNTERPARTIES_SHOWN} busiest by transfer "
                    "count; identified dataset entities are listed first so a "
                    "named counterparty can never be the one that falls off.",
                    indent="    ",
                )
            )
    # An address the seed dataset can name is the most consequential row in this
    # section, so it is never the row a cap removes. Ordering is otherwise the
    # analysis layer's (busiest first, address as tie-break) and stays stable.
    ordered = sorted(
        report.counterparties,
        key=lambda c: (c.entity_name is None, -c.transfer_count, c.address),
    )
    for counterparty in ordered[:MAX_COUNTERPARTIES_SHOWN]:
        label = counterparty.entity_name or "NOT_IN_DATASET"
        lines.append("")
        lines.append(f"    {counterparty.address}  [{label}]")
        lines.append(
            _kv(
                "transfers",
                f"{counterparty.transfer_count} "
                f"({counterparty.inbound_count} in / {counterparty.outbound_count} out)",
                indent="      ",
                pad=24,
            )
        )
        lines.extend(
            _wrap(
                "assets: " + _asset_list(counterparty.assets),
                indent="      ",
            )
        )
        lines.append(
            _kv(
                "first / last seen",
                f"{_ts(counterparty.first_seen)} .. {_ts(counterparty.last_seen)}",
                indent="      ",
                pad=24,
            )
        )
        if counterparty.entity_source_type:
            lines.append(
                _kv(
                    "dataset provenance",
                    _v(counterparty.entity_source_type),
                    indent="      ",
                    pad=24,
                )
            )
        if counterparty.tx_hashes:
            lines.append("      sample tx hashes:")
            for tx in counterparty.tx_hashes[:3]:
                lines.append(f"        {tx}")
    lines.append("")
    lines.extend(
        _omitted(
            MAX_COUNTERPARTIES_SHOWN, len(report.counterparties), "counterparties"
        )
    )
    return lines


# --------------------------------------------------------------------------
# section 4 -- bidirectional analysis
# --------------------------------------------------------------------------


def _render_bidirectional(report: InvestigationReport) -> list[str]:
    lines = _section(4, "BIDIRECTIONAL ANALYSIS")
    lines.extend(
        _wrap(
            "Direction matters and is never collapsed. A VASP sending to this "
            "wallet (a withdrawal) is evidence in its own right even when the "
            "wallet never sends to that VASP, and the reverse is a deposit. Both "
            "directions are searched independently.",
            indent="  ",
        )
    )
    lines.append("")

    attribution = report.attribution
    if attribution is None or not attribution.candidates:
        lines.append("  No directional evidence to report.")
        if attribution is not None and _v(attribution.status) == "INCONCLUSIVE":
            lines.extend(
                _wrap(
                    "The search was INCOMPLETE, so absence of directional "
                    "evidence here is not evidence of absence.",
                    indent="  ",
                )
            )
        return lines

    buckets: dict[str, list[Any]] = {}
    for candidate in attribution.candidates:
        buckets.setdefault(_v(candidate.direction), []).append(candidate)

    for direction in sorted(buckets):
        lines.append(f"  {direction}")
        for candidate in buckets[direction]:
            outbound = candidate.outbound_evidence
            inbound = candidate.inbound_evidence
            lines.append(
                f"    {candidate.vasp_name} via {candidate.matched_address}"
            )
            if outbound is not None:
                lines.append(
                    f"      wallet -> VASP  : {outbound.hop_distance} hop(s), "
                    f"tier {_v(outbound.evidence_tier)}, "
                    f"route {' -> '.join(_short(a) for a in outbound.path_addresses)}"
                )
            else:
                lines.append(
                    "      wallet -> VASP  : none found (no deposit path observed)"
                )
            if inbound is not None:
                lines.append(
                    f"      VASP -> wallet  : {inbound.hop_distance} hop(s), "
                    f"tier {_v(inbound.evidence_tier)}, "
                    f"route {' -> '.join(_short(a) for a in inbound.path_addresses)}"
                )
            else:
                lines.append(
                    "      VASP -> wallet  : none found (no withdrawal path observed)"
                )
        lines.append("")

    both = [
        c
        for c in attribution.candidates
        if c.outbound_evidence is not None and c.inbound_evidence is not None
    ]
    lines.append(
        _kv(
            "BIDIRECTIONAL RELATIONSHIPS",
            f"{len(both)} operator(s) with evidence in BOTH directions",
        )
    )
    if attribution.outbound_search_truncated:
        lines.extend(
            _wrap(
                "! The outbound search was truncated by its edge budget. Missing "
                "outbound evidence above is INCONCLUSIVE, not negative.",
                indent="  ",
            )
        )
    if attribution.inbound_searches_truncated:
        lines.extend(
            _wrap(
                f"! {attribution.inbound_searches_truncated} inbound search(es) "
                "were truncated by their edge budget. Missing inbound evidence is "
                "INCONCLUSIVE, not negative.",
                indent="  ",
            )
        )
    return lines


# --------------------------------------------------------------------------
# section 5 -- behavioural intelligence
# --------------------------------------------------------------------------


def _split_first_of_each_kind(items: list, kind) -> tuple[list, list]:
    """Partitions a list into (first of each kind, later repeats of a kind).

    Both halves keep the input order, so the caller's own ordering -- largest
    contribution first, busiest counterparty first -- survives inside each half.
    """
    firsts: list = []
    repeats: list = []
    seen: set[str] = set()
    for item in items:
        key = kind(item)
        if key in seen:
            repeats.append(item)
        else:
            seen.add(key)
            firsts.append(item)
    return firsts, repeats


def _behavior_display_order(patterns: list) -> list:
    """Orders indicators so a display cap can never hide an indicator TYPE.

    Real wallets produce the same indicator many times over: a busy mainnet
    address yielded 87 findings of which 78 were HIGH_FREQUENCY_COUNTERPARTY,
    one per counterparty. Truncating that list in its natural order would have
    printed twenty near-identical rows and hidden the nine distinct findings
    behind them -- the opposite of useful.

    So the first occurrence of each distinct indicator type comes first, in the
    analysis layer's order, and the repeats follow. The cap then falls on
    repeats, and the summary table above it still accounts for every finding.
    """
    firsts, repeats = _split_first_of_each_kind(
        patterns, lambda p: _v(p.pattern_type)
    )
    return firsts + repeats


def _behavior_detail(number: int, pattern: Any) -> list[str]:
    """The full evidence block for one indicator finding."""
    lines = [_THIN, f"  INDICATOR {number}: {_v(pattern.pattern_type)}", _THIN]
    lines.append(
        _kv("classification", _v(pattern.classification), indent="    ", pad=26)
    )
    lines.append(_kv("observed metric", pattern.observed_metric, indent="    ", pad=26))
    lines.append(_kv("observed value", pattern.observed_value, indent="    ", pad=26))
    lines.append(
        _kv(
            "threshold",
            f"{pattern.threshold} ({pattern.threshold_setting})",
            indent="    ",
            pad=26,
        )
    )
    if pattern.confidence:
        lines.append(_kv("confidence", pattern.confidence, indent="    ", pad=26))
    if pattern.first_seen or pattern.last_seen:
        lines.append(
            _kv(
                "window",
                f"{_ts(pattern.first_seen)} .. {_ts(pattern.last_seen)}",
                indent="    ",
                pad=26,
            )
        )
    lines.append("    evidence:")
    lines.extend(_bullets(list(pattern.evidence), indent="      "))
    if pattern.metrics:
        lines.append("    metrics:")
        lines.extend(
            _bullets(
                [f"{k} = {v}" for k, v in sorted(pattern.metrics.items())],
                indent="      ",
            )
        )
    if pattern.related_addresses:
        lines.append(f"    related addresses ({len(pattern.related_addresses)}):")
        for address in pattern.related_addresses[:5]:
            lines.append(f"      {address}")
        if len(pattern.related_addresses) > 5:
            lines.append(f"      ... and {len(pattern.related_addresses) - 5} more")
    if pattern.relevant_tx_hashes:
        lines.append(f"    relevant transactions ({len(pattern.relevant_tx_hashes)}):")
        for tx in pattern.relevant_tx_hashes[:5]:
            lines.append(f"      {tx}")
        if len(pattern.relevant_tx_hashes) > 5:
            lines.append(f"      ... and {len(pattern.relevant_tx_hashes) - 5} more")
    lines.append("")
    return lines


def _behavior_repeat_row(pattern: Any) -> str:
    """One compact line for a finding whose type was already itemised in full.

    Carries the two fields that actually differ between repeats -- the address
    the indicator was measured on and the value measured -- and the number of
    transactions behind it. The address is printed in full, never shortened:
    a truncated address cannot be re-checked on-chain.
    """
    addresses = list(pattern.related_addresses or [])
    if len(addresses) == 1:
        subject = addresses[0]
    elif addresses:
        subject = f"({len(addresses)} related addresses)"
    else:
        subject = "(no related address recorded)"
    tx_count = len(pattern.relevant_tx_hashes or [])
    value = _v(pattern.observed_value)
    return f"      {subject:<44} = {value:<8} {tx_count} tx"


def _render_behavior(report: InvestigationReport) -> list[str]:
    lines = _section(5, "BEHAVIORAL INTELLIGENCE")
    lines.extend(
        _wrap(
            "Every item below is an INVESTIGATIVE INDICATOR: an observation "
            "measured against a configured threshold. None of them is an "
            "allegation of criminal conduct, and none is sufficient on its own. "
            "Each states its metric, its threshold and the transactions it was "
            "computed from so it can be independently checked.",
            indent="  ",
        )
    )
    lines.append("")

    if not report.behavior_patterns:
        lines.append(
            "  No behavioural indicator crossed its threshold on this data."
        )
        return lines

    patterns = list(report.behavior_patterns)

    # The census comes before the detail, so a capped list below can never make
    # a finding invisible: every indicator is counted here even when its
    # individual evidence block is only in --json.
    by_type: dict[str, int] = {}
    for pattern in patterns:
        key = _v(pattern.pattern_type)
        by_type[key] = by_type.get(key, 0) + 1
    lines.append(f"  INDICATORS OBSERVED: {len(patterns)} across {len(by_type)} type(s)")
    for name, count in sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"    {name:<34} {count}")
    lines.append("")

    firsts, repeats = _split_first_of_each_kind(
        patterns, lambda p: _v(p.pattern_type)
    )
    detailed = firsts[:MAX_BEHAVIOR_INDICATORS_SHOWN]
    detailed_types = {_v(p.pattern_type) for p in detailed}
    # Identity, not equality: two findings can compare equal field-for-field
    # (the same indicator on two addresses with identical counts) and dropping
    # one of them from the compact list would silently lose a finding.
    detailed_ids = {id(p) for p in detailed}
    # A type that lost its full-detail slot to the cap still has to appear, so
    # its findings join the compact list rather than vanishing from the section.
    compact = [p for p in patterns if id(p) not in detailed_ids]

    if compact:
        lines.extend(
            _wrap(
                f"{len(detailed)} finding(s) are itemised in full below -- the "
                "first of each distinct indicator type. The remaining "
                f"{len(compact)} are repeats of a type already itemised, whose "
                "metric, threshold and classification are identical by "
                "construction, so they are listed compactly afterwards with "
                "the address and value that differ.",
                indent="  ",
            )
        )
        lines.append("")

    for number, pattern in enumerate(detailed, start=1):
        lines.extend(_behavior_detail(number, pattern))

    if compact:
        lines.append(_THIN)
        lines.append("  FURTHER FINDINGS, GROUPED BY INDICATOR TYPE")
        lines.append(_THIN)
        grouped: dict[str, list] = {}
        for pattern in compact:
            grouped.setdefault(_v(pattern.pattern_type), []).append(pattern)
        for name in sorted(grouped, key=lambda n: (-len(grouped[n]), n)):
            group = grouped[name]
            example = group[0]
            lines.append("")
            suffix = (
                "" if name in detailed_types
                else "  (no full evidence block above: too many distinct types)"
            )
            lines.append(f"    {name}: {len(group)} further finding(s){suffix}")
            lines.extend(
                _wrap(
                    f"metric {example.observed_metric}, threshold "
                    f"{example.threshold} ({example.threshold_setting}), "
                    f"classification {_v(example.classification)}",
                    indent="      ",
                )
            )
            for pattern in group[:MAX_BEHAVIOR_REPEATS_PER_TYPE]:
                lines.append(_behavior_repeat_row(pattern))
            lines.extend(
                _omitted(
                    MAX_BEHAVIOR_REPEATS_PER_TYPE,
                    len(group),
                    f"{name} finding(s)",
                    indent="      ",
                )
            )
        lines.append("")
    return lines


# --------------------------------------------------------------------------
# section 6 -- temporal and amount analysis
# --------------------------------------------------------------------------


def _render_temporal(report: InvestigationReport) -> list[str]:
    lines = _section(6, "TEMPORAL AND AMOUNT ANALYSIS")
    analysis = report.temporal
    if analysis is None:
        lines.append("  Not performed.")
        return lines

    lines.append("  ACTIVITY WINDOW")
    lines.append(_kv("transfers analysed", analysis.transfer_count, indent="    ", pad=28))
    lines.append(
        _kv(
            "with timestamps",
            f"{analysis.timestamped_transfer_count} "
            f"({analysis.missing_timestamp_count} missing)",
            indent="    ",
            pad=28,
        )
    )
    lines.append(_kv("first seen (UTC)", analysis.first_seen_utc or "(none)", indent="    ", pad=28))
    lines.append(_kv("last seen (UTC)", analysis.last_seen_utc or "(none)", indent="    ", pad=28))
    lines.append(_kv("lifespan (days)", analysis.lifespan_days, indent="    ", pad=28))
    lines.append(_kv("active days", analysis.active_day_count, indent="    ", pad=28))
    lines.append(
        _kv("transfers / active day", analysis.transfers_per_active_day, indent="    ", pad=28)
    )
    lines.append(
        _kv("longest idle gap (days)", analysis.longest_idle_days, indent="    ", pad=28)
    )
    lines.append(
        _kv("median gap (s)", analysis.median_gap_seconds, indent="    ", pad=28)
    )
    lines.append(_kv("mean gap (s)", analysis.mean_gap_seconds, indent="    ", pad=28))

    lines.append("")
    lines.append("  DIRECTIONAL COUNTS")
    lines.append(_kv("inbound transfers", analysis.inbound_transfer_count, indent="    ", pad=28))
    lines.append(_kv("outbound transfers", analysis.outbound_transfer_count, indent="    ", pad=28))
    lines.append(_kv("self transfers", analysis.self_transfer_count, indent="    ", pad=28))
    lines.append(
        _kv("unique senders", analysis.unique_inbound_counterparties, indent="    ", pad=28)
    )
    lines.append(
        _kv("unique receivers", analysis.unique_outbound_counterparties, indent="    ", pad=28)
    )

    lines.append("")
    lines.append("  TIMING PROFILE (UTC)")
    lines.append(_kv("busiest hour", analysis.busiest_utc_hour, indent="    ", pad=28))
    lines.append(
        _kv("busiest weekday", _weekday(analysis.busiest_utc_weekday), indent="    ", pad=28)
    )
    # Both histograms are dict[bucket] -> count and are sparse: a bucket with no
    # activity is absent, not zero. Every bucket is printed anyway, because
    # "no transfers ever happened at 03:00 UTC" is itself a finding, and the
    # bar is scaled by the largest COUNT (not the largest key).
    if analysis.hourly_utc_histogram:
        peak = max(analysis.hourly_utc_histogram.values()) or 1
        lines.append("    hourly distribution (24 UTC hours):")
        for hour in range(24):
            count = analysis.hourly_utc_histogram.get(hour, 0)
            bar = "#" * int(round(count / peak * 40))
            lines.append(f"      {hour:02d}h |{bar:<40}| {count}")
    if analysis.weekday_utc_histogram:
        peak = max(analysis.weekday_utc_histogram.values()) or 1
        lines.append("    weekday distribution:")
        for weekday in range(7):
            count = analysis.weekday_utc_histogram.get(weekday, 0)
            bar = "#" * int(round(count / peak * 40))
            lines.append(f"      {_WEEKDAYS[weekday]} |{bar:<40}| {count}")

    lines.append("")
    lines.append("  PER-ASSET AMOUNTS")
    # Busiest first (the analysis layer's order), except that the chain's native
    # asset is always shown: on a wallet spammed with airdropped tokens ETH can
    # rank below a cap on transfer count alone, and hiding the native-value
    # breakdown to make room for a worthless token would be indefensible.
    assets_ordered = sorted(
        analysis.per_asset,
        key=lambda s: (_v(s.asset_type).upper() != "NATIVE", -s.transfer_count, s.asset),
    )
    if not analysis.per_asset:
        lines.append("    No asset activity recorded.")
    elif len(assets_ordered) > MAX_ASSETS_SHOWN:
        lines.extend(
            _wrap(
                f"{len(assets_ordered)} distinct assets moved through this "
                f"wallet; the native asset plus the busiest by transfer count "
                f"are broken down below ({MAX_ASSETS_SHOWN} shown).",
                indent="    ",
            )
        )
    for stats in assets_ordered[:MAX_ASSETS_SHOWN]:
        flag = "  [METADATA INCOMPLETE]" if stats.metadata_incomplete else ""
        lines.append("")
        lines.append(f"    {stats.asset} ({_v(stats.asset_type)}){flag}")
        if stats.token_contract:
            lines.append(_kv("contract", stats.token_contract, indent="      ", pad=22))
        lines.append(
            _kv(
                "transfers",
                f"{stats.transfer_count} "
                f"({stats.inbound_count} in / {stats.outbound_count} out)",
                indent="      ",
                pad=22,
            )
        )
        lines.append(_kv("total inbound", _amount(stats.total_inbound), indent="      ", pad=22))
        lines.append(_kv("total outbound", _amount(stats.total_outbound), indent="      ", pad=22))
        lines.append(_kv("net flow", _amount(stats.net_flow), indent="      ", pad=22))
        lines.append(
            _kv(
                "min / median / max",
                f"{_amount(stats.min_amount)} / {_amount(stats.median_amount)} / "
                f"{_amount(stats.max_amount)}",
                indent="      ",
                pad=22,
            )
        )
        lines.append(_kv("mean", _amount(stats.mean_amount), indent="      ", pad=22))

    lines.extend(
        _omitted(MAX_ASSETS_SHOWN, len(analysis.per_asset), "assets", indent="    ")
    )

    pass_through = analysis.pass_through
    if pass_through is not None:
        lines.append("")
        lines.append("  PASS-THROUGH TIMING (inbound followed by outbound)")
        lines.append(
            _kv("measured events", pass_through.measured_events, indent="    ", pad=32)
        )
        lines.append(
            _kv(
                "min / median / max seconds",
                f"{pass_through.min_seconds} / {pass_through.median_seconds} / "
                f"{pass_through.max_seconds}",
                indent="    ",
                pad=32,
            )
        )
        lines.append(
            _kv(
                "inbound with no later outbound",
                pass_through.inbound_without_later_outbound,
                indent="    ",
                pad=32,
            )
        )
        if pass_through.limitation:
            lines.extend(_wrap(pass_through.limitation, indent="    "))

    if analysis.limitations:
        lines.append("")
        lines.append("  LIMITATIONS OF THIS SECTION")
        lines.extend(_bullets(analysis.limitations, indent="    "))
    return lines


# --------------------------------------------------------------------------
# section 7 -- ML analysis
# --------------------------------------------------------------------------


def _render_metrics(metrics: Any, indent: str = "      ") -> list[str]:
    """Prints held-out metrics with the baseline immediately beside accuracy.

    Accuracy is never printed alone. On an imbalanced split a trivial
    majority-class predictor can score very high, so the comparison is the only
    thing that makes the number interpretable, and separating them would invite
    exactly the misreading this project is required to avoid.
    """
    lines = [f"{indent}split: {metrics.split}  (n={metrics.sample_count})"]
    lines.append(
        f"{indent}accuracy               : {metrics.accuracy}"
    )
    lines.append(
        f"{indent}majority-class baseline: {metrics.majority_class_baseline_accuracy}"
        f"   <-- a model that always guesses the biggest class"
    )
    lines.append(
        f"{indent}improvement over it    : {metrics.accuracy_above_baseline}"
    )
    lines.append(f"{indent}positive class         : {metrics.positive_class}")
    lines.append(f"{indent}precision              : {metrics.precision}")
    lines.append(f"{indent}recall                 : {metrics.recall}")
    lines.append(f"{indent}F1                     : {metrics.f1}")
    lines.append(f"{indent}ROC-AUC                : {metrics.roc_auc}")
    lines.append(f"{indent}PR-AUC                 : {metrics.pr_auc}")
    lines.append(f"{indent}decision threshold     : {metrics.decision_threshold}")
    lines.append(f"{indent}class counts           : {_counts(metrics.class_counts)}")
    matrix = metrics.confusion_matrix or []
    if matrix:
        lines.append(f"{indent}confusion matrix (rows = actual, cols = predicted):")
        for row in matrix:
            lines.append(f"{indent}    {row}")
    return lines


def _render_ml(report: InvestigationReport) -> list[str]:
    lines = _section(7, "MACHINE LEARNING ANALYSIS")
    ml = report.ml
    if ml is None:
        lines.append("  Not performed.")
        return lines

    lines.append(_kv("APPROACH", ml.approach))
    lines.append("")
    lines.extend(
        _wrap(
            "This system will not train a supervised model on labels it cannot "
            "defend. Where the real, verifiable labels are too few, it says so "
            "and falls back to a method that needs none -- it does not invent "
            "labels to produce an accuracy figure.",
            indent="  ",
        )
    )
    lines.append("")

    if ml.rationale:
        lines.append("  WHY THIS APPROACH")
        lines.extend(_bullets(ml.rationale, indent="    "))
        lines.append("")

    for outcome, title in (
        (ml.account_type_labels, "ACCOUNT-TYPE LABELS (protocol-guaranteed)"),
        (ml.vasp_labels, "VASP-OWNERSHIP LABELS (dataset provenance)"),
    ):
        if outcome is None:
            continue
        lines.append(f"  {title}")
        lines.append(_kv("label schema", outcome.label_schema_version, indent="    ", pad=30))
        lines.append(_kv("class counts", _counts(outcome.class_counts), indent="    ", pad=30))
        lines.append(
            _kv("minimum per class", outcome.min_required_per_class, indent="    ", pad=30)
        )
        lines.append(
            _kv(
                "sufficient to train",
                "YES" if outcome.sufficient else "NO",
                indent="    ",
                pad=30,
            )
        )
        lines.append(_kv("distinct groups", outcome.group_count, indent="    ", pad=30))
        if outcome.activity_floor:
            lines.append(
                _kv("activity floor", outcome.activity_floor, indent="    ", pad=30)
            )
            lines.append(
                _kv(
                    "excluded below floor",
                    outcome.excluded_below_activity_floor,
                    indent="    ",
                    pad=30,
                )
            )
        if outcome.excluded_inconsistent_provenance:
            lines.append(
                _kv(
                    "excluded (provenance)",
                    outcome.excluded_inconsistent_provenance,
                    indent="    ",
                    pad=30,
                )
            )
        if outcome.excluded_conflicting_labels:
            lines.append(
                _kv(
                    "excluded (ambiguous)",
                    outcome.excluded_conflicting_labels,
                    indent="    ",
                    pad=30,
                )
            )
        if outcome.blockers:
            lines.append("    BLOCKERS")
            lines.extend(_bullets(outcome.blockers, indent="      "))
        if outcome.notes:
            lines.append("    NOTES")
            lines.extend(_bullets(outcome.notes, indent="      "))
        lines.append("")

    training = ml.training
    if training is not None and training.trained:
        lines.append("  TRAINED MODEL")
        lines.append(_kv("task", training.task, indent="    ", pad=30))
        lines.append(_kv("model", training.model_name, indent="    ", pad=30))
        lines.append(_kv("model version", training.model_version, indent="    ", pad=30))
        lines.append(_kv("pipeline version", training.pipeline_version, indent="    ", pad=30))
        lines.append(_kv("trained at (UTC)", training.trained_at_utc, indent="    ", pad=30))
        lines.append(_kv("random seed", training.random_seed, indent="    ", pad=30))
        lines.append(_kv("artifact", training.artifact_path, indent="    ", pad=30))
        lines.append(
            _kv("imbalance handling", training.class_imbalance_handling, indent="    ", pad=30)
        )

        provenance = training.provenance
        if provenance is not None:
            lines.append("")
            lines.append("    DATASET PROVENANCE")
            lines.append(_kv("dataset version", provenance.dataset_version, indent="      ", pad=28))
            lines.append(
                _kv("feature schema", provenance.feature_schema_version, indent="      ", pad=28)
            )
            lines.append(_kv("label schema", provenance.label_schema_version, indent="      ", pad=28))
            lines.append(_kv("graph source", provenance.graph_source, indent="      ", pad=28))
            lines.append(_kv("samples", provenance.sample_count, indent="      ", pad=28))
            lines.append(_kv("class counts", _counts(provenance.class_counts), indent="      ", pad=28))
            lines.append(_kv("groups", provenance.group_count, indent="      ", pad=28))
            lines.append(
                _kv("label sources", provenance.label_source_counts, indent="      ", pad=28)
            )
            lines.append(
                _kv(
                    "excluded feature groups",
                    provenance.excluded_feature_groups or "(none)",
                    indent="      ",
                    pad=28,
                )
            )
            if provenance.feature_exclusion_reason:
                lines.extend(_wrap(provenance.feature_exclusion_reason, indent="        "))

        split = training.split_report
        if split is not None:
            lines.append("")
            lines.append("    SPLIT (leakage control)")
            lines.append(_kv("strategy", split.strategy, indent="      ", pad=28))
            lines.append(
                _kv(
                    "train / val / test n",
                    f"{split.train_sample_count} / {split.validation_sample_count} / "
                    f"{split.test_sample_count}",
                    indent="      ",
                    pad=28,
                )
            )
            lines.append(
                _kv(
                    "train / val / test groups",
                    f"{split.train_group_count} / {split.validation_group_count} / "
                    f"{split.test_group_count}",
                    indent="      ",
                    pad=28,
                )
            )
            lines.append(
                _kv(
                    "group overlap",
                    "NONE (disjoint)" if not split.groups_overlap else "! OVERLAP DETECTED",
                    indent="      ",
                    pad=28,
                )
            )
            lines.append(_kv("CV folds", split.cv_folds, indent="      ", pad=28))
            lines.append(_kv("test class counts", _counts(split.test_class_counts), indent="      ", pad=28))

        if training.candidates:
            lines.append("")
            lines.append("    MODEL SELECTION (cross-validated on TRAIN only)")
            for candidate in training.candidates:
                marker = "*" if candidate.selected else " "
                lines.append(
                    f"      {marker} {candidate.model_name:<28} "
                    f"CV F1 {candidate.cv_mean_f1} +/- {candidate.cv_std_f1}"
                )
                if candidate.note:
                    lines.extend(_wrap(candidate.note, indent="          "))

        if training.validation_metrics is not None:
            lines.append("")
            lines.append("    VALIDATION METRICS (threshold selected here)")
            lines.extend(_render_metrics(training.validation_metrics))
        if training.test_metrics is not None:
            lines.append("")
            lines.append("    HELD-OUT TEST METRICS (untouched until now)")
            lines.extend(_render_metrics(training.test_metrics))

        if training.feature_importances:
            lines.append("")
            lines.append("    TOP FEATURES")
            for importance in training.feature_importances[:10]:
                lines.append(
                    f"      {importance.feature:<32} {importance.importance}  "
                    f"({importance.method})"
                )

        if training.limitations:
            lines.append("")
            lines.append("    LIMITATIONS OF THIS MODEL")
            lines.extend(_bullets(training.limitations, indent="      "))
    elif training is not None:
        lines.append("  TRAINING WAS ATTEMPTED AND REFUSED")
        lines.extend(_bullets(training.blockers, indent="    "))
        lines.append("")

    prediction = ml.prediction
    if prediction is not None:
        lines.append("")
        lines.append("  PREDICTION FOR THIS WALLET")
        if not prediction.available:
            lines.extend(_wrap(str(prediction.unavailable_reason), indent="    "))
        else:
            lines.append(_kv("predicted class", prediction.predicted_class, indent="    ", pad=28))
            lines.append(_kv("probability", prediction.probability, indent="    ", pad=28))
            lines.append(
                _kv("decision threshold", prediction.decision_threshold, indent="    ", pad=28)
            )
            lines.append(_kv("evidence class", prediction.evidence_class, indent="    ", pad=28))
            lines.append(_kv("model", f"{prediction.model_name} {prediction.model_version}", indent="    ", pad=28))
            lines.append(_kv("dataset version", prediction.dataset_version, indent="    ", pad=28))
            lines.append(
                _kv("feature schema", prediction.feature_schema_version, indent="    ", pad=28)
            )
            lines.append(_kv("trained at (UTC)", prediction.trained_at_utc, indent="    ", pad=28))
            lines.append(
                _kv("training samples", prediction.training_sample_count, indent="    ", pad=28)
            )
            lines.append(
                _kv("training classes", prediction.training_class_counts, indent="    ", pad=28)
            )
            if prediction.contributions:
                lines.append("    WHY (feature value vs training median)")
                for contribution in prediction.contributions:
                    median = (
                        contribution.training_median
                        if contribution.training_median is not None
                        else "n/a"
                    )
                    lines.append(
                        f"      {contribution.feature:<30} value {contribution.value:<14} "
                        f"median {median:<14} importance {contribution.importance}"
                    )
            lines.extend(_bullets(prediction.interpretation, indent="    "))

    outlier = ml.outlier
    if outlier is not None:
        lines.append("")
        lines.append("  UNSUPERVISED BEHAVIOURAL OUTLIER ANALYSIS")
        if not outlier.available:
            lines.extend(_wrap(str(outlier.unavailable_reason), indent="    "))
        else:
            lines.append(_kv("method", f"{outlier.method} ({outlier.method_version})", indent="    ", pad=30))
            lines.append(_kv("evidence class", outlier.evidence_class, indent="    ", pad=30))
            lines.append(_kv("population size", outlier.population_size, indent="    ", pad=30))
            lines.extend(
                _wrap(f"population source: {outlier.population_source}", indent="    ")
            )
            lines.append(
                _kv(
                    "activity floor",
                    f"{outlier.population_activity_floor} transfers "
                    f"({outlier.addresses_below_activity_floor} addresses excluded)",
                    indent="    ",
                    pad=30,
                )
            )
            lines.append(_kv("outlier score", outlier.outlier_score, indent="    ", pad=30))
            lines.append(
                _kv(
                    "percentile in population",
                    outlier.percentile_within_population,
                    indent="    ",
                    pad=30,
                )
            )
            lines.append(
                _kv(
                    "flagged as outlier",
                    "YES" if outlier.flagged_as_outlier else "NO",
                    indent="    ",
                    pad=30,
                )
            )
            lines.append(
                _kv(
                    "target inside population",
                    "YES" if outlier.target_in_population else "NO",
                    indent="    ",
                    pad=30,
                )
            )
            stability = outlier.rank_stability
            if stability is not None:
                lines.append("")
                lines.append("    RANK STABILITY (the only honest metric available here)")
                lines.append(
                    _kv("bootstrap resamples", stability.resamples, indent="      ", pad=28)
                )
                lines.append(
                    _kv(
                        "mean Spearman rho",
                        stability.mean_spearman_correlation,
                        indent="      ",
                        pad=28,
                    )
                )
                lines.append(
                    _kv(
                        "minimum Spearman rho",
                        stability.min_spearman_correlation,
                        indent="      ",
                        pad=28,
                    )
                )
                lines.extend(_wrap(stability.interpretation, indent="      "))
            if outlier.deviations:
                lines.append("")
                lines.append("    WHY (features outside the population's p10-p90 band)")
                # Two rows per feature rather than one 111-character row: the
                # single-row form ran well past the report's 78-column margin
                # and wrapped unpredictably on a narrow console, which is worse
                # than an explicit second line.
                lines.append(
                    "      feature                        value        percentile"
                )
                lines.append(
                    "        population median / p10 / p90                direction"
                )
                for deviation in outlier.deviations:
                    lines.append(
                        f"      {deviation.feature:<30} "
                        f"{deviation.value:<12} {deviation.percentile_rank}"
                    )
                    lines.append(
                        f"        {deviation.population_median} / "
                        f"{deviation.population_p10} / {deviation.population_p90}"
                        f"   {deviation.direction}"
                    )
            if outlier.evaluation:
                lines.append("")
                lines.append("    EVALUATION")
                lines.extend(_bullets(outlier.evaluation, indent="      "))
    return lines


# --------------------------------------------------------------------------
# section 8 -- evidence summary
# --------------------------------------------------------------------------


def _render_evidence_summary(report: InvestigationReport) -> list[str]:
    lines = _section(8, "EVIDENCE SUMMARY AND RISK ANALYSIS")
    lines.append("  EVIDENCE CLASS DEFINITIONS USED THROUGHOUT THIS REPORT")
    for name, meaning in (
        ("DIRECT", "a single transfer between the wallet and a dataset address"),
        ("INDIRECT", "a multi-hop transaction path; continuity is NOT established"),
        ("SUPPORTING", "corroborates a finding; can never establish one alone"),
        ("CONTEXTUAL", "background such as behavioural unusualness"),
        ("UNVERIFIED", "asserted by a source that has not been independently checked"),
        ("INCONCLUSIVE", "the search did not complete; nothing may be concluded"),
    ):
        lines.extend(_wrap(f"{name}: {meaning}", indent="    ", ))
    lines.append("")

    risk = report.risk
    if risk is None:
        lines.append("  No risk assessment was produced.")
    else:
        lines.append("  INVESTIGATIVE PRIORITY SCORE (fully itemised)")
        lines.append(
            _kv(
                "score",
                f"{risk.score} points",
                indent="    ",
                pad=30,
            )
        )
        # The denominator is NOT a maximum possible risk -- it is the summed
        # weight of the components that actually fired, which differs from the
        # score only where a component was discounted (a weak-plausibility
        # path contributes half its weight). Printing "51 of a possible 51"
        # invites the reader to see a maxed-out 100% risk, which is a
        # different and much stronger claim than the arithmetic supports.
        lines.append(
            _kv(
                "weight of fired components",
                f"{risk.max_possible_score} "
                f"({len(risk.components)} component(s) fired; the score is "
                "lower only where one was discounted)",
                indent="    ",
                pad=30,
            )
        )
        lines.append(_kv("band", _v(risk.band), indent="    ", pad=30))
        lines.append(
            _kv(
                "band thresholds",
                f"medium >= {risk.band_medium_threshold}, "
                f"high >= {risk.band_high_threshold}",
                indent="    ",
                pad=30,
            )
        )
        lines.append(
            _kv(
                "data complete",
                "YES" if risk.data_complete else "NO",
                indent="    ",
                pad=30,
            )
        )
        if risk.data_completeness_note:
            lines.extend(_wrap(risk.data_completeness_note, indent="      "))

        lines.append("")
        lines.append("    CONTRIBUTIONS (these sum to the score above -- nothing hidden)")
        # The total is computed over EVERY component, not just the printed ones,
        # so the arithmetic check below still means what it says when the detail
        # list is capped. Nothing is quietly left out of the sum: a component is
        # either itemised in full, listed compactly with its own contribution,
        # or included in one explicit withheld subtotal.
        total = sum(float(component.contribution) for component in risk.components)
        # Largest contribution first: if a cap has to fall somewhere, it should
        # fall on the items that moved the score least.
        components_ordered = sorted(
            risk.components,
            key=lambda c: (-float(c.contribution), c.component_id),
        )
        # Same repetition problem as section 5: on a real wallet 60 of 67
        # contributions were HIGH_FREQUENCY_COUNTERPARTY, one per counterparty,
        # each with an identical weight, threshold and reason paragraph. The
        # first of each indicator earns a full itemisation; the rest are one
        # line each, which is where all their new information fits anyway.
        firsts, repeats = _split_first_of_each_kind(
            components_ordered, lambda c: _v(c.indicator)
        )
        detailed = firsts[:MAX_RISK_CONTRIBUTIONS_SHOWN]
        detailed_ids = {id(c) for c in detailed}
        remaining = [c for c in components_ordered if id(c) not in detailed_ids]
        if remaining:
            lines.extend(
                _wrap(
                    f"{len(components_ordered)} components contributed. The "
                    f"{len(detailed)} below are itemised in full -- the largest "
                    "contribution of each distinct indicator. The remaining "
                    f"{len(remaining)} repeat an indicator already itemised, so "
                    "they are listed one line each with their own contribution, "
                    "and anything beyond that limit is subtotalled. The "
                    "arithmetic reconciles line by line either way.",
                    indent="    ",
                )
            )
        for component in detailed:
            lines.append("")
            lines.append(
                f"      + {component.contribution:<6} [{_v(component.evidence_class)}] "
                f"{component.indicator}"
            )
            lines.append(
                _kv("component id", component.component_id, indent="          ", pad=24)
            )
            lines.append(
                _kv(
                    "weight",
                    f"{component.weight} ({component.weight_setting})",
                    indent="          ",
                    pad=24,
                )
            )
            if component.observed_metric:
                lines.append(
                    _kv(
                        "observed",
                        f"{component.observed_metric} = {component.observed_value}",
                        indent="          ",
                        pad=24,
                    )
                )
            if component.threshold is not None:
                lines.append(
                    _kv(
                        "threshold",
                        f"{component.threshold} ({component.threshold_setting})",
                        indent="          ",
                        pad=24,
                    )
                )
            if component.observed_evidence:
                # A list of strings, one observation each: printed as bullets,
                # never as a Python repr with brackets and quotes around it.
                lines.append("          evidence:")
                lines.extend(_bullets(list(component.observed_evidence), indent="            "))
            if component.reason:
                lines.extend(_wrap(f"reason: {component.reason}", indent="          "))
            if component.related_addresses:
                lines.append("          related addresses:")
                for address in component.related_addresses[:4]:
                    lines.append(f"            {address}")
                if len(component.related_addresses) > 4:
                    lines.append(
                        f"            ... and {len(component.related_addresses) - 4} more"
                    )
            if component.relevant_tx_hashes:
                lines.append("          transaction evidence:")
                for tx in component.relevant_tx_hashes[:4]:
                    lines.append(f"            {tx}")
                if len(component.relevant_tx_hashes) > 4:
                    lines.append(
                        f"            ... and {len(component.relevant_tx_hashes) - 4} more"
                    )

        lines.append("")

        # Compact rows, grouped by indicator so the shared metric and threshold
        # are stated once. Each row still carries its own contribution, so these
        # are part of the itemisation rather than part of the withheld subtotal.
        withheld: list = []
        if remaining:
            grouped: dict[str, list] = {}
            for component in remaining:
                grouped.setdefault(_v(component.indicator), []).append(component)
            lines.append("    FURTHER CONTRIBUTIONS, GROUPED BY INDICATOR")
            for name in sorted(grouped, key=lambda n: (-len(grouped[n]), n)):
                group = grouped[name]
                example = group[0]
                lines.append("")
                lines.append(f"      {name}: {len(group)} further contribution(s)")
                detail = (
                    f"weight {example.weight} ({example.weight_setting}), "
                    f"class {_v(example.evidence_class)}"
                )
                if example.threshold is not None:
                    detail += (
                        f", {example.observed_metric} vs threshold "
                        f"{example.threshold} ({example.threshold_setting})"
                    )
                lines.extend(_wrap(detail, indent="        "))
                shown_rows = group[:MAX_RISK_REPEATS_PER_INDICATOR]
                for component in shown_rows:
                    addresses = list(component.related_addresses or [])
                    if len(addresses) == 1:
                        subject = addresses[0]
                    elif addresses:
                        subject = f"({len(addresses)} related addresses)"
                    else:
                        subject = component.component_id
                    lines.append(
                        f"        + {float(component.contribution):<7} {subject}"
                    )
                withheld.extend(group[MAX_RISK_REPEATS_PER_INDICATOR:])
            lines.append("")

        if withheld:
            withheld_total = sum(float(c.contribution) for c in withheld)
            lines.extend(
                _wrap(
                    f"+ {round(withheld_total, 4)}   from {len(withheld)} further "
                    "contribution(s) not itemised above. Each is itemised in full "
                    "in the --json output; they are included in the total below.",
                    indent="      ",
                )
            )
            lines.append("")
        lines.append(
            f"      = {round(total, 4)} TOTAL "
            f"({'arithmetic verified' if abs(total - risk.score) < 1e-6 else '! DOES NOT MATCH REPORTED SCORE'})"
        )

        if risk.components_not_triggered:
            lines.append("")
            lines.append("    INDICATORS CHECKED AND NOT TRIGGERED")
            for skipped in risk.components_not_triggered:
                lines.extend(
                    _wrap(f"{skipped.component_id}: {skipped.reason}", indent="      ")
                )
        if risk.non_claims:
            lines.append("")
            lines.append("    WHAT THIS SCORE IS NOT")
            lines.extend(_bullets(risk.non_claims, indent="      "))

    lines.append("")
    lines.append("  CONSOLIDATED LIMITATIONS")
    lines.extend(_bullets(report.limitations, indent="    "))
    return lines


# --------------------------------------------------------------------------
# section 9 -- conclusion
# --------------------------------------------------------------------------


def _render_conclusion(report: InvestigationReport) -> list[str]:
    lines = _section(9, "FINAL INVESTIGATION CONCLUSION")
    for item in report.conclusion:
        lines.extend(_wrap(item, indent="  ", ))
        lines.append("")
    lines.extend(
        _wrap(
            "This report is an investigative aid. It identifies addresses, "
            "transaction paths and measured behavioural indicators, each with "
            "the evidence needed to check it. It does not establish criminal "
            "conduct, and no finding here should be acted on without "
            "independent verification of the underlying transactions.",
            indent="  ",
        )
    )
    return lines


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------


def render_report(report: InvestigationReport) -> str:
    """Renders the full nine-section report as one ASCII-safe string."""
    lines: list[str] = []
    lines.extend(_render_header(report))
    lines.extend(_render_data_summary(report))
    lines.extend(_render_fund_flow(report))
    lines.extend(_render_attribution(report))
    lines.extend(_render_bidirectional(report))
    lines.extend(_render_behavior(report))
    lines.extend(_render_temporal(report))
    lines.extend(_render_ml(report))
    lines.extend(_render_evidence_summary(report))
    lines.extend(_render_conclusion(report))
    lines.append("")
    lines.append(_RULE)
    lines.append(f"  INVESTIGATION COMPLETE -- {report.wallet} on {report.chain}")
    lines.append(
        f"  DATA MODE: {report.provenance.data_mode.value}   "
        f"DURATION: {report.duration_seconds}s"
    )
    lines.append(_RULE)
    # `_wrap` already folded the prose; this catches anything that reached the
    # output through an f-string interpolation of a model field.
    return to_ascii("\n".join(lines))


def print_report(report: InvestigationReport, stream: Optional[TextIO] = None) -> None:
    """Renders to a stream, forcing UTF-8 first."""
    target = stream if stream is not None else sys.stdout
    configure_stdout(target)
    print(render_report(report), file=target)
