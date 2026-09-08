"""
MACRO MILESTONE 4 — VASP seed dataset loading.

Loads and validates a JSON seed file into a list of VASPSeedEntry
objects. Deliberately strict: any structural problem (missing file,
invalid JSON, a malformed entry, or an ambiguous duplicate address)
raises SeedDataError rather than silently producing a partial or empty
seed set — attribution evidence integrity depends on the seed data being
exactly what it claims to be.

Fully offline — this never fetches anything over the network.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from app.attribution.models import VASPSeedEntry


class SeedDataError(Exception):
    """Raised whenever the seed dataset cannot be trusted as-is."""


def load_vasp_seed(path: str | Path) -> list[VASPSeedEntry]:
    path = Path(path)

    if not path.exists():
        raise SeedDataError(f"VASP seed file not found at {path}")

    try:
        raw_text = path.read_text()
    except OSError as exc:
        raise SeedDataError(f"Could not read VASP seed file at {path}: {exc}") from exc

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise SeedDataError(f"VASP seed file at {path} is not valid JSON: {exc}") from exc

    # Accept either a bare JSON array, or an object with a
    # "seed_addresses" array — both are reasonable authoring styles.
    if isinstance(raw, list):
        entries_raw = raw
    elif isinstance(raw, dict) and isinstance(raw.get("seed_addresses"), list):
        entries_raw = raw["seed_addresses"]
    else:
        raise SeedDataError(
            f"VASP seed file at {path} must be a JSON array of entries, "
            "or an object with a 'seed_addresses' array."
        )

    entries: list[VASPSeedEntry] = []
    seen_by_address: dict[str, VASPSeedEntry] = {}

    for i, item in enumerate(entries_raw):
        try:
            entry = VASPSeedEntry.model_validate(item)
        except ValidationError as exc:
            raise SeedDataError(
                f"VASP seed entry #{i} in {path} is malformed: {exc}"
            ) from exc

        key = entry.address.lower()
        prior = seen_by_address.get(key)
        if prior is not None:
            if prior.vasp_name != entry.vasp_name or prior.source_type != entry.source_type:
                raise SeedDataError(
                    f"Ambiguous duplicate seed address {entry.address} in {path}: "
                    f"'{prior.vasp_name}' ({prior.source_type.value}) vs "
                    f"'{entry.vasp_name}' ({entry.source_type.value}). "
                    "Refusing to guess which is correct."
                )
            continue  # exact duplicate entry — safe to ignore, not an error

        seen_by_address[key] = entry
        entries.append(entry)

    return entries
