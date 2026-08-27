"""In-process, per-process cache for retrieval answers.

Non-answers (timeout, backend error, cancel, unfilled-placeholder empty) are absent from
``logs`` and therefore never offered to ``store``. Empty answers default to uncached
(``empty_ttl_seconds=0``). Every hit carries ``age_seconds``; all consumers that show a
source's status must show it.
"""

import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Seconds a non-empty answer stays usable (1 h).
DEFAULT_TTL_SECONDS = 3600

#: Seconds an empty answer stays usable. 0 = never stored by default.
DEFAULT_EMPTY_TTL_SECONDS = 0

#: Entries kept before the least-recently-used is evicted.
DEFAULT_MAX_ENTRIES = 64

#: Total rows kept across all entries, independent of entry count.
DEFAULT_MAX_ROWS = 200_000


def _as_int(value, default: int) -> int:
    """``value`` as int, or ``default``; a typo must not silently become 0."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def cache_key(query, row_cap=None, guidance: str = "") -> str:
    """Stable key for one retrieval question.

    ``row_cap`` is in the key because a truncated result is a floor. ``guidance`` is in
    the key because a corrected ask must not hit the uncorrected answer. The backend query
    text is excluded: it does not exist until the retriever runs.
    """
    entities = sorted(
        (
            str(getattr(e, "type", "") or ""),
            str(getattr(e, "value", "") or ""),
            str(getattr(e, "value_form", "") or ""),
        )
        for e in (getattr(query, "entities", None) or [])
    )
    material = {
        "source": str(getattr(query, "target_log_source", "") or ""),
        "question": str(getattr(query, "natural_language_query", "") or ""),
        "scope_id": str(getattr(query, "scope_id", "") or ""),
        "actor_id": str(getattr(query, "actor_id", "") or ""),
        "date_from": str(getattr(query, "date_from", "") or ""),
        "date_to": str(getattr(query, "date_to", "") or ""),
        "entities": entities,
        "row_cap": _as_int(row_cap, 0),
        "guidance": guidance or "",
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CacheHit:
    """One answer from the cache.

    ``query`` and ``key_enforced`` are replayed, not recomputed: a hit runs no retriever,
    and ``last_key_enforced`` on a shared retriever would reflect whichever job ran last.
    """

    __slots__ = ("rows", "query", "key_enforced", "age_seconds", "stored_at")

    def __init__(
        self,
        rows: List[Any],
        query: str,
        key_enforced: bool,
        age_seconds: int,
        stored_at: float,
    ):
        self.rows = rows
        self.query = query
        self.key_enforced = key_enforced
        self.age_seconds = age_seconds
        self.stored_at = stored_at

    def describe(self) -> str:
        """Human phrasing for a status line, e.g. ``cached 4m ago``."""
        secs = max(0, int(self.age_seconds))
        if secs < 60:
            return f"cached {secs}s ago"
        if secs < 3600:
            return f"cached {secs // 60}m ago"
        return f"cached {secs // 3600}h ago"


class RetrievalCache:
    """Bounded, TTL'd, thread-safe in-process store of retrieval answers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.ttl_seconds = max(0, _as_int(cfg.get("ttl_seconds"), DEFAULT_TTL_SECONDS))
        self.empty_ttl_seconds = max(
            0, _as_int(cfg.get("empty_ttl_seconds"), DEFAULT_EMPTY_TTL_SECONDS)
        )
        self.max_entries = max(0, _as_int(cfg.get("max_entries"), DEFAULT_MAX_ENTRIES))
        self.max_rows = max(0, _as_int(cfg.get("max_rows"), DEFAULT_MAX_ROWS))
        # Insertion-ordered dict: oldest key is first; `get` re-inserts it at the end (LRU).
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._rows = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0
        self._expiries = 0

    def get(self, key: str) -> Optional[CacheHit]:
        """The answer stored under ``key``, or ``None``; discards undateable entries."""
        if not self.enabled or not key:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            stored_at = entry.get("stored_at")
            if not isinstance(stored_at, (int, float)) or stored_at <= 0:
                self._drop(key)
                self._misses += 1
                logger.warning(
                    "Discarded a retrieval cache entry with no usable stored_at; it "
                    "could never have expired."
                )
                return None
            age = time.time() - float(stored_at)
            ttl = self.empty_ttl_seconds if not entry["rows"] else self.ttl_seconds
            if ttl <= 0 or age > ttl:
                self._drop(key)
                self._expiries += 1
                self._misses += 1
                return None
            self._entries.pop(key)  # re-insert at end (LRU touch)
            self._entries[key] = entry
            self._hits += 1
            return CacheHit(
                rows=list(entry["rows"]),
                query=entry.get("query", ""),
                key_enforced=bool(entry.get("key_enforced", False)),
                age_seconds=int(age),
                stored_at=float(stored_at),
            )

    def store(
        self,
        key: str,
        rows: List[Any],
        query: str = "",
        key_enforced: bool = False,
    ) -> bool:
        """Keep one answer; returns whether it was kept.

        Refused when the answer's TTL is 0 (default for empty results) or when the
        answer alone exceeds the row budget.
        """
        if not self.enabled or not key:
            return False
        rows = list(rows or [])
        ttl = self.empty_ttl_seconds if not rows else self.ttl_seconds
        if ttl <= 0:
            return False
        if self.max_rows and len(rows) > self.max_rows:
            logger.debug(
                "Not caching %d rows: one answer over the %d-row budget would evict "
                "every other entry.",
                len(rows),
                self.max_rows,
            )
            return False
        with self._lock:
            self._drop(key)
            self._entries[key] = {
                "rows": rows,
                "query": query or "",
                "key_enforced": bool(key_enforced),
                "stored_at": time.time(),
            }
            self._rows += len(rows)
            self._stores += 1
            self._evict()
        return True

    def invalidate(self, key: str = "") -> int:
        """Drop one key, or the whole cache when ``key`` is empty. Returns entries dropped."""
        with self._lock:
            if key:
                dropped = 1 if key in self._entries else 0
                self._drop(key)
                return dropped
            dropped = len(self._entries)
            self._entries.clear()
            self._rows = 0
            return dropped

    def stats(self) -> Dict[str, Any]:
        """Counters for ``/health?deep=1``."""
        with self._lock:
            looked_up = self._hits + self._misses
            return {
                "enabled": self.enabled,
                "entries": len(self._entries),
                "rows": self._rows,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / looked_up, 3) if looked_up else 0.0,
                "stores": self._stores,
                "evictions": self._evictions,
                "expiries": self._expiries,
                "ttl_seconds": self.ttl_seconds,
                "empty_ttl_seconds": self.empty_ttl_seconds,
                "max_entries": self.max_entries,
                "max_rows": self.max_rows,
            }

    def _drop(self, key: str) -> None:
        """Remove one key and its rows from the totals. Caller holds the lock."""
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._rows = max(0, self._rows - len(entry["rows"]))

    def _evict(self) -> None:
        """Evict least-recently-used entries until both bounds hold. Caller holds the lock."""
        while self._entries and (
            (self.max_entries and len(self._entries) > self.max_entries)
            or (self.max_rows and self._rows > self.max_rows)
        ):
            oldest = next(iter(self._entries))
            self._drop(oldest)
            self._evictions += 1


def build_retrieval_cache(config: Optional[Dict[str, Any]] = None) -> RetrievalCache:
    """Build from ``log_sources.cache``; a missing block gives a disabled (no-op) cache."""
    cache = RetrievalCache((config or {}).get("cache") or {})
    if cache.enabled:
        logger.info(
            "Retrieval cache ON: ttl=%ss, empty_ttl=%ss, max_entries=%s, max_rows=%s",
            cache.ttl_seconds,
            cache.empty_ttl_seconds,
            cache.max_entries,
            cache.max_rows,
        )
    return cache


def cache_stats(engine) -> Tuple[bool, Dict[str, Any]]:
    """``(wired, stats)`` for any engine, including a mock or ``None``; never raises."""
    cache = getattr(engine, "cache", None)
    try:
        stats = cache.stats()
    except Exception:  # noqa: BLE001 — a reporter must not be the failure it reports
        return False, {}
    if not isinstance(stats, dict):
        return False, {}
    return True, stats
