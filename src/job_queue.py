"""Admission arithmetic for submitted runs: who starts now, who waits, what is refused.

Three rules: FIFO always; a full queue is refused (a job id for a run that may never start
reads like one that is merely slow); admission is a proposal — the caller re-checks each
released id because the job may have been cancelled while waiting. The width is not a global
throttle: retries, gate-resumes and link-lane children start outside it.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Two: a run spends most of its wall clock on I/O; past two the LLM semaphore orders the work.
DEFAULT_MAX_CONCURRENT = 2

#: Hard ceiling regardless of config; a YAML number cannot authorise unbounded fan-out.
MAX_CONCURRENT_CEILING = 8

#: Submissions that may wait. Past this the queue refuses.
DEFAULT_MAX_QUEUED = 256

#: Engine ceiling on the backlog.
MAX_QUEUED_CEILING = 4096


class QueueFull(RuntimeError):
    """Raised when a submission cannot be queued; carries depth and limit for a client-actionable 429."""

    def __init__(self, depth: int, limit: int):
        self.depth = int(depth)
        self.limit = int(limit)
        super().__init__(
            f"the run queue is full: {self.depth} submission(s) waiting against a "
            f"limit of {self.limit}"
        )


def _bounded(value, default: int, ceiling: int, floor: int = 1) -> int:
    """``value`` clamped to ``[floor, ceiling]``, or ``default`` if unreadable (a typo must not become the most permissive setting)."""
    if value is None or isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(floor, min(ceiling, number))


class JobQueue:
    """FIFO admission control. Reads bounds on every call so a live config change takes effect immediately."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config is not None else {}
        self._waiting: List[str] = []
        self._running: List[str] = []
        self._lock = threading.Lock()
        self._admitted = 0
        self._queued = 0
        self._refused = 0
        self._withdrawn = 0

    def width(self) -> int:
        jobs = self.config.get("jobs") or {}
        return _bounded(
            jobs.get("max_concurrent_jobs"),
            DEFAULT_MAX_CONCURRENT,
            MAX_CONCURRENT_CEILING,
        )

    def max_queued(self) -> int:
        jobs = self.config.get("jobs") or {}
        return _bounded(
            jobs.get("max_queued_jobs"), DEFAULT_MAX_QUEUED, MAX_QUEUED_CEILING
        )

    def admit(self, job_id: str) -> bool:
        """Register a submission: ``True`` = start now, ``False`` = queued. Raises :class:`QueueFull` at the limit; known jobs are not double-counted."""
        with self._lock:
            if job_id in self._running:
                return True
            if job_id in self._waiting:
                return False
            if len(self._running) < self.width():
                self._running.append(job_id)
                self._admitted += 1
                return True
            limit = self.max_queued()
            if len(self._waiting) >= limit:
                self._refused += 1
                raise QueueFull(len(self._waiting), limit)
            self._waiting.append(job_id)
            self._queued += 1
            return False

    def release(self, job_id: str) -> List[str]:
        """Give up a slot and return the ids that may now start. Idempotent; releasing an id that holds no slot frees nothing."""
        with self._lock:
            if job_id in self._running:
                self._running.remove(job_id)
            elif job_id not in self._waiting:
                return []
            else:
                # Finished without being admitted (started via a non-queued route).
                self._waiting.remove(job_id)
            starting = []
            while self._waiting and len(self._running) < self.width():
                nxt = self._waiting.pop(0)
                self._running.append(nxt)
                self._admitted += 1
                starting.append(nxt)
            return starting

    def withdraw(self, job_id: str) -> bool:
        """Remove a queued job from the backlog. Does not free a running slot (a running cancel goes through :meth:`release`)."""
        with self._lock:
            if job_id in self._waiting:
                self._waiting.remove(job_id)
                self._withdrawn += 1
                return True
            return False

    def position(self, job_id: str) -> int:
        """1-based backlog position, or ``0`` for running or unknown (``0`` not ``None``: this is a list field)."""
        with self._lock:
            if job_id in self._waiting:
                return self._waiting.index(job_id) + 1
            return 0

    def waiting(self) -> List[str]:
        """The backlog in the order it will drain."""
        with self._lock:
            return list(self._waiting)

    def running(self) -> List[str]:
        """The ids currently holding a slot."""
        with self._lock:
            return list(self._running)

    def stats(self) -> Dict[str, Any]:
        """Counters and the two live bounds, for ``/api/v1/batches`` and the job list."""
        with self._lock:
            return {
                "width": self.width(),
                "max_queued": self.max_queued(),
                "running": len(self._running),
                "queued": len(self._waiting),
                "admitted": self._admitted,
                "queued_total": self._queued,
                "refused": self._refused,
                "withdrawn": self._withdrawn,
            }
