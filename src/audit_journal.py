"""Who called, and what they changed: one append-only journal of attributable events.

Every other durable record AFIR keeps is about a RUN — a job document, an export, a pack
snapshot. That leaves a whole half of "usage" invisible: a caller who browses the UI, reads
someone else's report, is refused at the door, or edits the shared configuration leaves no
trace anywhere, and neither does the fact that anybody was here at all. The HTTP layer is
the only place that sees those, so it records them here.

Three properties this file exists to hold:

- **Durable, because the obvious sink is not.** Log lines go to stdout, which on a cluster
  driver is a file on ephemeral local disk. An audit trail that a restart erases is not one,
  so entries are appended to the storage backend, beside the feedback log.
- **Batched, because the backend charges per write.** A UC Volume / DBFS append is a
  read-modify-write of the whole object serialised on one thread, so an append per request
  would be quadratic over a day and would put an investigation's writes behind a queue of
  page loads. Entries accumulate in memory and land every ``flush_seconds`` or every
  ``max_buffer`` entries, whichever comes first, and the window that costs is stated in the
  config template rather than hidden here.
- **Best-effort, in one direction only.** Nothing in this module may fail, slow or reorder a
  request: every method swallows its own errors, a full buffer drops its OLDEST entries and
  says how many, and a sink that is refusing writes is reported once rather than per entry.

Rotated daily (``audit/access-YYYY-MM-DD.jsonl``) so retention is a file delete and no
single object grows without bound.
"""

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from src.storage import StorageBackend
from src.utils.deployment import resolve_mode, running_on_databricks_driver

logger = logging.getLogger(__name__)

#: Key prefix inside the store. One segment, so retention is a listing plus a delete.
JOURNAL_PREFIX = "audit"

#: Paths that carry no information about usage and would drown everything that does. The
#: platform's own liveness probe hits `/health` every few seconds forever.
DEFAULT_EXCLUDED_PATHS = ("/health",)

DEFAULT_FLUSH_SECONDS = 30.0
DEFAULT_MAX_BUFFER = 200
#: Hard ceiling while the sink is failing. Past this the OLDEST entries go: a journal that
#: grows until the process dies takes the investigation with it.
DEFAULT_MAX_PENDING = 5000
DEFAULT_RETENTION_DAYS = 90


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value, default: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def day_key(when: Optional[datetime] = None) -> str:
    """The journal key for one UTC day."""
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    return f"{JOURNAL_PREFIX}/access-{stamp}.jsonl"


class AuditJournal:
    """Buffered append-only journal. Disabled instances accept every call and do nothing."""

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        enabled: bool = True,
        flush_seconds: float = DEFAULT_FLUSH_SECONDS,
        max_buffer: int = DEFAULT_MAX_BUFFER,
        max_pending: int = DEFAULT_MAX_PENDING,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        excluded_paths=DEFAULT_EXCLUDED_PATHS,
    ):
        self.storage = storage
        # No store is not a soft failure: there is nowhere durable to put an entry, and
        # buffering into a process that is about to be restarted is worse than being honest.
        self.enabled = bool(enabled and storage is not None)
        self.flush_seconds = max(1.0, _as_float(flush_seconds, DEFAULT_FLUSH_SECONDS))
        self.max_buffer = max(1, _as_int(max_buffer, DEFAULT_MAX_BUFFER))
        self.max_pending = max(
            self.max_buffer, _as_int(max_pending, DEFAULT_MAX_PENDING)
        )
        self.retention_days = _as_int(retention_days, DEFAULT_RETENTION_DAYS)
        self.excluded_paths = frozenset(excluded_paths or ())
        self._pending: deque = deque()
        self._dropped = 0
        self._write_failures = 0
        self._reported_failure = False
        self._task: Optional[asyncio.Task] = None
        self._last_flush = time.time()

    # -- write -------------------------------------------------------------

    def record(self, kind: str, identity=None, **fields) -> None:
        """Buffer one entry. Cheap, synchronous, and never raises."""
        if not self.enabled:
            return
        try:
            entry = {"at": _now(), "kind": str(kind)}
            if identity is not None:
                entry.update(
                    {
                        "user": getattr(identity, "user_name", ""),
                        "user_id": getattr(identity, "user_id", ""),
                        "role": getattr(identity, "role", ""),
                        # How the identity was established, so a `local` entry is never read
                        # as an authenticated one.
                        "auth": getattr(identity, "source", ""),
                    }
                )
            else:
                # Said rather than left blank. No identity was established — a refusal at the
                # door, or a path that reads no header — and an entry with no `auth` at all
                # reads as one whose writer forgot, which is a different claim.
                entry["auth"] = "unresolved"
            entry.update({k: v for k, v in fields.items() if v is not None})
            self._pending.append(entry)
        except Exception as exc:  # noqa: BLE001 — recording must not fail a request
            logger.debug("Audit entry could not be buffered: %s", exc)
            return
        while len(self._pending) > self.max_pending:
            self._pending.popleft()
            self._dropped += 1

    def record_request(self, identity, method, path, status, duration_ms) -> None:
        """One completed HTTP request, unless its path is excluded."""
        if not self.enabled or str(path) in self.excluded_paths:
            return
        self.record(
            "request",
            identity,
            method=str(method),
            path=str(path),
            status=_as_int(status, 0),
            ms=round(_as_float(duration_ms, 0.0), 1),
        )

    @property
    def pending(self) -> int:
        return len(self._pending)

    @property
    def dropped(self) -> int:
        """Entries discarded because the buffer filled while the sink was failing."""
        return self._dropped

    def due(self) -> bool:
        return bool(self._pending) and (
            len(self._pending) >= self.max_buffer
            or (time.time() - self._last_flush) >= self.flush_seconds
        )

    def flush_now(self) -> int:
        """Append everything buffered. Blocking; call via a thread. Returns entries written."""
        if not self.enabled or not self._pending:
            return 0
        # Snapshot first: an entry recorded while this write is in flight belongs to the next
        # batch, and must not be lost by clearing the buffer afterwards.
        batch = [self._pending.popleft() for _ in range(len(self._pending))]
        dropped, self._dropped = self._dropped, 0
        if dropped:
            batch.append(
                {
                    "at": _now(),
                    "kind": "journal_overflow",
                    "dropped": dropped,
                    "detail": "entries discarded; the journal buffer filled",
                }
            )
        lines = "".join(
            json.dumps(entry, ensure_ascii=False, default=str) + "\n" for entry in batch
        )
        try:
            written = bool(self.storage.append_text(day_key(), lines))
        except Exception as exc:  # noqa: BLE001
            written = False
            logger.debug("Audit journal append raised: %s", exc)
        self._last_flush = time.time()
        if not written:
            self._write_failures += 1
            if not self._reported_failure:
                self._reported_failure = True
                logger.error(
                    "The audit journal cannot write to %s storage. Access and "
                    "configuration-change entries are being lost; runs are unaffected.",
                    getattr(self.storage, "kind", "unknown"),
                )
            return 0
        self._reported_failure = False
        return len(batch)

    async def flush(self) -> int:
        """Flush off the event loop."""
        if not self.enabled or not self._pending:
            return 0
        return await asyncio.to_thread(self.flush_now)

    # -- lifecycle ---------------------------------------------------------

    async def _run(self) -> None:
        """Flush on a timer, so an idle period cannot hold the last entries hostage."""
        while True:
            await asyncio.sleep(self.flush_seconds)
            try:
                await self.flush()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("Audit journal flush failed: %s", exc)

    def start(self) -> None:
        """Begin the periodic flush. Requires a running loop; idempotent."""
        if not self.enabled or self._task is not None:
            return
        try:
            self._task = asyncio.get_running_loop().create_task(self._run())
        except RuntimeError:
            # No loop: a synchronous caller (a test, a script). Entries still flush on
            # `flush_now`, so this is a degradation and not a failure.
            logger.debug("Audit journal has no running loop; periodic flush is off.")

    async def stop(self, timeout: float = 10.0) -> int:
        """Cancel the timer and write what is buffered. The last thing a shutdown does with it."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if not self.enabled:
            return 0
        try:
            return await asyncio.wait_for(self.flush(), timeout=timeout)
        except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
            logger.warning("Audit journal did not drain within %.0fs: %s", timeout, exc)
            return 0

    # -- read --------------------------------------------------------------

    def tail(self, limit: int = 200, since: str = "", user: str = "") -> List[dict]:
        """The most recent entries, newest first.

        Reads whole days newest-first and stops as soon as `limit` is satisfied, so the
        common "what happened today" costs one object read however long the journal is.
        """
        if self.storage is None:
            return []
        limit = max(1, _as_int(limit, 200))
        want_user = str(user or "").strip().lower()
        out: List[dict] = []
        for key in self._day_keys():
            try:
                blob = self.storage.get_text(key)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Audit journal %s unreadable: %s", key, exc)
                continue
            for entry in reversed(_parse_lines(blob)):
                if since and str(entry.get("at") or "") < since:
                    # Days are read newest-first and each day is ordered, so this bound is
                    # reached once and everything past it is older.
                    return out
                if want_user and str(entry.get("user") or "").lower() != want_user:
                    continue
                out.append(entry)
                if len(out) >= limit:
                    return out
        return out

    def _day_keys(self) -> List[str]:
        """Every journal object, newest day first. Sorted by NAME: the date is in it."""
        if self.storage is None:
            return []
        try:
            objects = self.storage.list_keys(JOURNAL_PREFIX)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Audit journal listing failed: %s", exc)
            return []
        keys = [obj.key for obj in objects if obj.key.endswith(".jsonl")]
        return sorted(keys, reverse=True)

    def stats(self) -> Dict[str, object]:
        """What the journal itself is doing, for the health surface."""
        return {
            "enabled": self.enabled,
            "pending": self.pending,
            "dropped": self._dropped,
            "write_failures": self._write_failures,
            "flush_seconds": self.flush_seconds,
            "retention_days": self.retention_days,
            "days_held": len(self._day_keys()) if self.enabled else 0,
        }

    # -- housekeeping ------------------------------------------------------

    def prune(self) -> int:
        """Delete journal days past the retention window. Returns the count."""
        if not self.enabled or self.retention_days <= 0:
            return 0
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        ).strftime("%Y-%m-%d")
        removed = 0
        for key in self._day_keys():
            # Compared as text on the key's own date, not on an mtime: an appended file's
            # mtime is the last WRITE, which for the current day is now.
            stamp = key.rsplit("access-", 1)[-1][: len("YYYY-MM-DD")]
            if len(stamp) != 10 or stamp >= cutoff:
                continue
            try:
                if self.storage.delete(key):
                    removed += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not delete audit journal %s: %s", key, exc)
        if removed:
            logger.info(
                "Pruned %d audit journal day(s) older than %d days.",
                removed,
                self.retention_days,
            )
        return removed


def _parse_lines(blob: Optional[str]) -> List[dict]:
    """Parse one JSONL object; a truncated or corrupt line is skipped, never fatal."""
    entries = []
    for line in str(blob or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            loaded = json.loads(line)
        except ValueError:
            continue
        if isinstance(loaded, dict):
            entries.append(loaded)
    return entries


def build_audit_journal(
    config: Optional[dict] = None, storage: Optional[StorageBackend] = None
) -> AuditJournal:
    """Build from the ``audit`` config section.

    ``enabled: auto`` follows the identity layer: on a driver proxy there are several
    callers to tell apart, and on a laptop every caller is the same local operator, so a
    journal there records one person visiting their own machine.
    """
    cfg = (config or {}).get("audit") or {}
    enabled = resolve_mode(
        cfg.get("enabled", "auto"), platform_default=running_on_databricks_driver()
    )
    excluded = cfg.get("exclude_paths", None)
    if isinstance(excluded, str):
        excluded = [p.strip() for p in excluded.split(",") if p.strip()]
    return AuditJournal(
        storage=storage,
        enabled=enabled,
        flush_seconds=cfg.get("flush_seconds", DEFAULT_FLUSH_SECONDS),
        max_buffer=cfg.get("max_buffer", DEFAULT_MAX_BUFFER),
        max_pending=cfg.get("max_pending", DEFAULT_MAX_PENDING),
        retention_days=cfg.get("retention_days", DEFAULT_RETENTION_DAYS),
        excluded_paths=(
            tuple(excluded) if excluded is not None else DEFAULT_EXCLUDED_PATHS
        ),
    )
