"""
MACRO MILESTONE 4 — known-VASP address matching.

Exact, case-insensitive address matching only. No fuzzy matching, no
prefix matching, no heuristic "looks similar to" logic — an address
either is or isn't in the seed set. This keeps the matching step itself
free of any judgment call that could later be mistaken for evidence.
"""

from __future__ import annotations

from app.attribution.models import VASPSeedEntry


def build_seed_index(entries: list[VASPSeedEntry]) -> dict[str, VASPSeedEntry]:
    """Address (lowercased) -> VASPSeedEntry. Assumes `entries` already
    passed through seed_loader.load_vasp_seed, which guarantees no
    ambiguous duplicates."""
    return {entry.address.lower(): entry for entry in entries}


def match_address(address: str, seed_index: dict[str, VASPSeedEntry]) -> VASPSeedEntry | None:
    return seed_index.get(address.lower())
