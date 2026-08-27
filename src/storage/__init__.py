"""Durable storage for job state, evidence, exports and the feedback log.

:func:`build_storage` is the single decision point. ``storage.backend`` choices: ``local``
(default, any mountable path), ``databricks`` (Unity Catalog Volume via the Files API),
or ``sql`` (SQLite / PostgreSQL; the only backend safe across multiple replicas). An
unrecognised value falls back to ``local``.
"""

import logging
from typing import Optional

from src.storage.base import StorageBackend, StoredObject, key_prefix, safe_key
from src.storage.local import LocalStorage, json_verifier
from src.storage.prefixed import PrefixedStorage

logger = logging.getLogger(__name__)

__all__ = [
    "StorageBackend",
    "StoredObject",
    "LocalStorage",
    "PrefixedStorage",
    "build_storage",
    "json_verifier",
    "safe_key",
    "key_prefix",
]


def build_storage(config: Optional[dict] = None) -> StorageBackend:
    """Build the configured backend. Always returns something usable."""
    cfg = ((config or {}).get("storage") or {}) if isinstance(config, dict) else {}
    backend = str(cfg.get("backend") or "local").strip().lower()

    if backend in ("", "local", "disk", "filesystem"):
        return LocalStorage(root=cfg.get("root") or None)

    if backend == "databricks":
        try:
            from src.storage.databricks import DatabricksStorage
        except Exception as exc:  # noqa: BLE001 — SDK absent, import error, anything
            logger.error(
                "storage.backend is 'databricks' but that backend could not be "
                "loaded (%s). Falling back to LOCAL DISK, which does NOT survive a "
                "container restart — pending approvals may be lost.",
                exc,
            )
            return LocalStorage(root=cfg.get("root") or None)
        return DatabricksStorage(cfg.get("databricks") or {})

    if backend in ("sql", "database", "db", "sqlite", "postgresql", "postgres"):
        try:
            from src.storage.sql import SqlStorage
        except Exception as exc:  # noqa: BLE001 — driver absent, import error, anything
            logger.error(
                "storage.backend is %r but the SQL backend could not be loaded (%s). "
                "Falling back to LOCAL DISK, which does NOT survive a container "
                "restart — pending approvals may be lost.",
                backend,
                exc,
            )
            return LocalStorage(root=cfg.get("root") or None)
        sql_cfg = dict(cfg.get("sql") or {})
        # Dialect-name alias; an explicit sql.dialect still wins.
        if backend in ("sqlite", "postgresql", "postgres") and not sql_cfg.get(
            "dialect"
        ):
            sql_cfg["dialect"] = backend
        return SqlStorage(sql_cfg)

    logger.error(
        "Unknown storage.backend %r; falling back to LOCAL DISK. Valid values are "
        "'local' (optionally with storage.root pointing at a mounted or external "
        "volume), 'databricks' (a managed OR external Unity Catalog Volume) and 'sql'.",
        backend,
    )
    return LocalStorage(root=cfg.get("root") or None)
