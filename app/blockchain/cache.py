"""
On-disk cache for raw blockchain-provider responses.

Why this exists
---------------
A single production investigation of an active wallet issues dozens of
paginated provider requests. Re-issuing them on every run is slow, burns
rate limit, and — worse for an evidence-producing tool — makes a run
non-reproducible: two runs minutes apart can see different data with no
record of which one a report came from. Caching the *raw* provider payload
(before normalization) fixes all three and lets an investigation be
re-derived byte-for-byte from what the provider actually said.

Determinism rules (do not weaken)
---------------------------------
* The cache key is a SHA-256 over a canonical, sorted JSON encoding of the
  query identity: provider name, chain, chain id, module, action, address
  (lowercased), block range, page, offset and sort. Nothing time-derived
  and nothing random ever enters the key, so the same logical query always
  maps to the same file.
* The API KEY IS NEVER part of the key and is NEVER written to disk. The
  writer explicitly drops any parameter named `apikey`/`api_key`/`token`
  before persisting, so a cache directory can be shared or inspected
  without leaking a credential.
* A cache entry records the provider, the query identity, and the epoch it
  was fetched at, so a report can state exactly how old its data is
  instead of implying it is live.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

_SECRET_PARAM_NAMES = frozenset({"apikey", "api_key", "apiKey", "key", "token"})


def _canonical_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Strips credentials and normalizes value types so the encoding is
    stable across runs (ints vs. str ints must not produce two keys)."""
    cleaned: dict[str, Any] = {}
    for name, value in identity.items():
        if name in _SECRET_PARAM_NAMES:
            continue
        if isinstance(value, bool):
            cleaned[name] = value
        elif isinstance(value, (int, float)):
            cleaned[name] = str(value)
        elif value is None:
            cleaned[name] = None
        else:
            text = str(value)
            cleaned[name] = text.lower() if name in ("address", "chain") else text
    return cleaned


def cache_key(identity: dict[str, Any]) -> str:
    """Deterministic cache key for one provider query."""
    canonical = _canonical_identity(identity)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ProviderResponseCache:
    """Filesystem cache of raw provider payloads.

    Never raises on cache problems: a corrupt or unreadable entry is a
    cache miss, and a failed write is ignored. A cache is an optimization,
    so a cache fault must never change an investigation's outcome — it may
    only change how long it takes.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        ttl_seconds: int = 0,
        enabled: bool = True,
    ):
        self._dir = Path(directory)
        self._ttl = max(0, int(ttl_seconds))
        self._enabled = enabled
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def directory(self) -> Path:
        return self._dir

    def path_for(self, key: str) -> Path:
        # Two-level fan-out keeps directory sizes sane for large sweeps.
        return self._dir / key[:2] / f"{key}.json"

    def get(self, identity: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Returns the cached envelope {"fetched_at", "identity", "payload"}
        or None on any miss/expiry/corruption."""
        if not self._enabled:
            return None
        path = self.path_for(cache_key(identity))
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            self.misses += 1
            return None
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError:
            self.misses += 1
            return None
        if not isinstance(envelope, dict) or "payload" not in envelope:
            self.misses += 1
            return None
        fetched_at = envelope.get("fetched_at")
        if self._ttl and isinstance(fetched_at, int):
            if time.time() - fetched_at > self._ttl:
                self.misses += 1
                return None
        self.hits += 1
        return envelope

    def set(
        self,
        identity: dict[str, Any],
        payload: Any,
        fetched_at: Optional[int] = None,
    ) -> None:
        if not self._enabled:
            return
        path = self.path_for(cache_key(identity))
        envelope = {
            "fetched_at": int(fetched_at if fetched_at is not None else time.time()),
            "identity": _canonical_identity(identity),
            "payload": payload,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic replace so a crash mid-write can never leave a
            # half-written entry that a later run would treat as real data.
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=path.name,
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(envelope, handle, separators=(",", ":"))
                tmp_name = handle.name
            os.replace(tmp_name, path)
            self.writes += 1
        except OSError:
            return

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}
