"""Durable job persistence so a run parked on an approval gate survives a restart.

Evidence goes in a sidecar (``<job_id>.evidence.json``), rewritten only when changed, and
capped — a rehydrated job that is missing its evidence is better than one that looks complete
while having no data. Every method is best-effort and returns bool; losing durability must not
lose the investigation.
"""

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.identity import USER_PREFIX, owner_of, owner_prefix
from src.storage import (LocalStorage, PrefixedStorage, StorageBackend,
                         json_verifier)
from src.utils.paths import data_dir

logger = logging.getLogger(__name__)

# Output keys held in the sidecar instead of the main doc. These are the bulk
# evidence payloads: big, and unchanged for most of a job's life.
EVIDENCE_KEYS = ("logs",)

# Default cap on the serialized evidence sidecar. 64 MB is generous for a normal
# incident and still far below anything that would stall a restart.
DEFAULT_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024

# Default retention for job files on disk. Longer than the in-memory TTL on purpose:
# the point of persisting is to outlive the process, and an approval can sit for days.
DEFAULT_RETENTION_DAYS = 14


class JobStore:
    """Reads and writes job docs as one JSON document per job. Never raises."""

    def __init__(
        self,
        base_dir=None,
        max_evidence_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
        retention_days: float = DEFAULT_RETENTION_DAYS,
        enabled: bool = True,
        storage: Optional[StorageBackend] = None,
    ):
        self.enabled = enabled
        self.max_evidence_bytes = int(max_evidence_bytes)
        self.retention_days = float(retention_days)
        self._configured_dir = Path(base_dir) if base_dir else None
        # An explicit backend wins. Absent one, the store builds a local backend over
        # its own directory on first use — so a caller that passes only base_dir (every
        # existing caller, and every test) is unaffected by this indirection.
        # Explicit backend wins; absent one, a local backend is built on first use from base_dir.
        self._storage = storage
        self._resolved: Optional[StorageBackend] = None
        self._fallback_announced = False
        self._using_fallback = False

    # -- location ----------------------------------------------------------

    def _primary_dir(self) -> Path:
        return self._configured_dir or (data_dir() / "jobs")

    @property
    def storage(self) -> Optional[StorageBackend]:
        """The backend in use, probed on first need (a location existing does not prove it is writable). None if unusable."""
        if not self.enabled:
            return None
        if self._resolved is not None:
            return self._resolved

        primary = self._storage or LocalStorage(root=self._primary_dir())
        where = _describe(primary, self._primary_dir())
        if _probe_writable(primary):
            self._resolved = primary
            self._using_fallback = False
            return self._resolved

        fallback = LocalStorage(root=Path(tempfile.gettempdir()) / "afir_jobs")
        if not _probe_writable(fallback):
            if not self._fallback_announced:
                logger.error(
                    "Job persistence DISABLED: neither %s nor the local fallback %s is "
                    "writable. Jobs will be lost on restart, including any awaiting "
                    "human approval.",
                    where,
                    fallback.root,
                )
                self._fallback_announced = True
            self.enabled = False
            return None

        if not self._fallback_announced:
            logger.error(
                "Job persistence falling back to LOCAL DISK %s because %s is not "
                "writable. This does not survive a container restart — pending "
                "approvals may be lost. Check the storage config / AFIR_DATA_DIR.",
                fallback.root,
                where,
            )
            self._fallback_announced = True
        self._resolved = fallback
        self._using_fallback = True
        return self._resolved

    @property
    def directory(self) -> Optional[Path]:
        """The local root in use, or ``None`` for a backend that has no directory."""
        backend = self.storage
        return getattr(backend, "root", None) if backend is not None else None

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    def _keys(self, job_id: str, owner: str = ""):
        """The (doc, evidence) keys for a job, or (None, None) if unusable.

        An owned job lives under its owner's segment. An unowned one keeps the flat key it
        has always had, so every job written before this existed still loads.
        """
        if self.storage is None:
            return None, None
        safe = _safe_id(job_id)
        base = f"{owner_prefix(owner)}/{safe}" if owner else safe
        return f"{base}.json", f"{base}.evidence.json"

    def _locate(self, job_id: str) -> Optional[str]:
        """Find a job's doc key when the caller has no doc to read the owner from.

        A search rather than an argument: `delete` is reached from prune and from the job
        manager, neither of which holds the incident.
        """
        backend = self.storage
        if backend is None:
            return None
        safe = _safe_id(job_id)
        flat = f"{safe}.json"
        if backend.exists(flat):
            return flat
        suffix = f"/{flat}"
        for obj in backend.list_keys(USER_PREFIX):
            if obj.key.endswith(suffix):
                return obj.key
        return None

    # -- write -------------------------------------------------------------

    def save(self, doc: dict, evidence_changed: bool = False) -> bool:
        """Persist one job doc; ``evidence_changed`` also rewrites the sidecar. Most saves are status updates with unchanged evidence."""
        if not self.enabled:
            return False
        job_id = doc.get("job_id")
        if not job_id:
            logger.warning("Refusing to persist a job doc with no job_id.")
            return False
        doc_key, evidence_key = self._keys(job_id, owner_of(doc.get("incident")))
        if doc_key is None:
            return False

        outputs = dict(doc.get("outputs") or {})
        evidence = {k: outputs.pop(k) for k in EVIDENCE_KEYS if k in outputs}
        slim = dict(doc)
        slim["outputs"] = outputs

        if evidence_changed and evidence:
            written, reason = self._write_evidence(evidence_key, evidence, job_id)
            # Recorded in the doc: a reloaded job must say "evidence gone" not look complete with empty logs.
            slim["evidence_omitted"] = None if written else reason
        elif evidence_changed:
            slim["evidence_omitted"] = None

        try:
            blob = json.dumps(slim, default=str, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Job doc for %s is not serializable: %s", job_id, exc)
            return False
        return self.storage.put_text(doc_key, blob, verify=json_verifier)

    def flush(self, timeout: float = 30.0) -> bool:
        """Flush queued writes; no-op for synchronous backends.

        Gate saves flush because a container dying with the save still queued loses the
        pending approval. ``False`` is a real answer, not an error.
        """
        backend = self.storage
        flush = getattr(backend, "flush", None) if backend is not None else None
        if flush is None:
            return True
        try:
            return bool(flush(timeout=timeout))
        except Exception as exc:  # noqa: BLE001 — best-effort, like every method here
            logger.warning("Storage flush failed: %s", exc)
            return False

    def _write_evidence(self, key: str, evidence: dict, job_id: str):
        """Write the sidecar unless it exceeds the cap. Returns (written, reason)."""
        try:
            blob = json.dumps(evidence, default=str)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Job %s evidence is not serializable: %s", job_id, exc)
            return False, f"not serializable: {exc}"
        size = len(blob.encode("utf-8"))
        if size > self.max_evidence_bytes:
            logger.warning(
                "Job %s evidence is %.1f MB, over the %.1f MB cap; NOT persisted. A "
                "reloaded copy of this job cannot re-run the stages that consume it.",
                job_id,
                size / 1e6,
                self.max_evidence_bytes / 1e6,
            )
            # Remove a stale sidecar so the doc and sidecar cannot disagree.
            self.storage.delete(key)
            return False, (
                f"evidence {size / 1e6:.1f} MB exceeded the "
                f"{self.max_evidence_bytes / 1e6:.1f} MB cap"
            )
        return self.storage.put_text(key, blob, verify=json_verifier), "write failed"

    # -- read --------------------------------------------------------------

    def load_all(self) -> List[dict]:
        """Every persisted job doc, oldest first, evidence merged back in.

        A document that will not parse is skipped with a warning rather than aborting
        the load: one corrupt job must not cost the operator the whole queue.
        """
        backend = self.storage
        if backend is None:
            return []
        docs = []
        for key in sorted(self._job_doc_keys()):
            doc = _parse(backend.get_text(key)) or _parse(
                backend.get_previous_text(key)
            )
            if not isinstance(doc, dict) or not doc.get("job_id"):
                logger.warning("Skipping unreadable job document %s", key)
                continue
            _, evidence_key = self._keys(
                doc["job_id"], owner_of(doc.get("incident"))
            )
            evidence = _parse(backend.get_text(evidence_key)) if evidence_key else None
            if isinstance(evidence, dict):
                doc.setdefault("outputs", {}).update(evidence)
            elif doc.get("evidence_omitted"):
                logger.warning(
                    "Job %s was persisted WITHOUT its evidence (%s); stages that "
                    "consume it cannot be re-run on this copy.",
                    doc["job_id"],
                    doc["evidence_omitted"],
                )
            docs.append(doc)
        docs.sort(key=lambda x: x.get("created_at") or "")
        return docs

    def load_one(self, job_id: str) -> Optional[dict]:
        """One persisted job doc by id, evidence merged back in, or ``None``.

        The read behind rehydrating a run the in-memory TTL evicted: `load_all` would read
        every document to answer for one.
        """
        backend = self.storage
        if backend is None:
            return None
        doc_key = self._locate(job_id)
        if doc_key is None:
            return None
        doc = _parse(backend.get_text(doc_key)) or _parse(
            backend.get_previous_text(doc_key)
        )
        if not isinstance(doc, dict) or not doc.get("job_id"):
            logger.warning("Stored job document %s is unreadable.", doc_key)
            return None
        evidence = _parse(
            backend.get_text(doc_key[: -len(".json")] + ".evidence.json")
        )
        if isinstance(evidence, dict):
            doc.setdefault("outputs", {}).update(evidence)
        elif doc.get("evidence_omitted"):
            logger.warning(
                "Job %s was persisted WITHOUT its evidence (%s); stages that consume it "
                "cannot be re-run on this copy.",
                doc["job_id"],
                doc["evidence_omitted"],
            )
        return doc

    def _job_doc_keys(self) -> List[str]:
        """Every main-document key, excluding the evidence sidecars."""
        backend = self.storage
        if backend is None:
            return []
        return [
            obj.key
            for obj in backend.list_keys("")
            if obj.key.endswith(".json") and not obj.key.endswith(".evidence.json")
        ]

    # -- housekeeping ------------------------------------------------------

    def delete(self, job_id: str) -> bool:
        doc_key = self._locate(job_id)
        if doc_key is None:
            return False
        ok = self.storage.delete(doc_key)
        self.storage.delete(doc_key[: -len(".json")] + ".evidence.json")
        return ok

    def prune(self) -> int:
        """Delete job documents older than the retention window. Returns the count."""
        backend = self.storage
        if backend is None or self.retention_days <= 0:
            return 0
        cutoff = time.time() - (self.retention_days * 86400)
        removed = 0
        for obj in backend.list_keys(""):
            if not obj.key.endswith(".json") or obj.key.endswith(".evidence.json"):
                continue
            try:
                # mtime 0.0 means unknown; treating as epoch would delete the whole queue.
                if not obj.mtime or obj.mtime >= cutoff:
                    continue
                doc = _parse(backend.get_text(obj.key))
                job_id = (doc or {}).get("job_id")
                if job_id:
                    self.delete(job_id)
                else:
                    backend.delete(obj.key)
                removed += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("Prune skipped %s: %s", obj.key, exc)
        if removed:
            logger.info(
                "Pruned %d persisted job(s) older than %.0f days.",
                removed,
                self.retention_days,
            )
        return removed


# -- helpers ---------------------------------------------------------------


def _safe_id(job_id: str) -> str:
    """Restrict a job id to filename-safe characters; imported ids may carry ``/`` or ``..``."""
    return "".join(c for c in str(job_id) if c.isalnum() or c in "-_")[:128] or "job"


def _parse(blob: Optional[str]) -> Optional[dict]:
    """Parse a stored JSON document; ``None`` for absent or unparseable content."""
    if not blob:
        return None
    try:
        loaded = json.loads(blob)
        return loaded if isinstance(loaded, dict) else None
    except (ValueError, TypeError) as exc:
        logger.warning("Stored job document does not parse: %s", exc)
        return None


def _probe_writable(backend: StorageBackend) -> bool:
    """Write and remove a throwaway key. See ``JobStore.storage`` for why."""
    probe = "afir_write_probe.json"
    try:
        if not backend.put_text(probe, "{}"):
            return False
        backend.delete(probe)
        return True
    except Exception as exc:  # noqa: BLE001 — a probe must never raise past here
        logger.debug("Storage write probe failed: %s", exc)
        return False


def _describe(backend: StorageBackend, fallback_dir: Path) -> str:
    """A location an operator can act on, for the degradation message."""
    root = getattr(backend, "root", None)
    return str(root) if root is not None else f"{backend.kind} storage ({fallback_dir})"


def build_job_store(
    config: Optional[Dict] = None, storage: Optional[StorageBackend] = None
) -> JobStore:
    """Build from the ``jobs`` config section. ``storage`` overrides the backend (written under a ``jobs/`` prefix)."""
    cfg = (config or {}).get("jobs", {}) or {}
    mb = cfg.get("max_evidence_mb")
    backend = PrefixedStorage(storage, "jobs") if storage is not None else None
    return JobStore(
        base_dir=cfg.get("dir") or None,
        max_evidence_bytes=(
            int(float(mb) * 1024 * 1024)
            if mb is not None
            else DEFAULT_MAX_EVIDENCE_BYTES
        ),
        retention_days=cfg.get("retention_days", DEFAULT_RETENTION_DAYS),
        enabled=cfg.get("persist", True),
        storage=backend,
    )
