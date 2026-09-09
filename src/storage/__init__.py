"""Durable storage for job state, evidence, exports and the feedback log.

:func:`build_storage` is the single decision point. ``storage.backend`` choices: ``local``
(default, any mountable path), ``databricks`` (Unity Catalog Volume via the Files API),
``dbfs`` (the same machinery over the DBFS API, which needs no UC grant), or ``sql``
(SQLite / PostgreSQL; the only backend safe across multiple replicas). An unrecognised
value falls back to ``local``.

The two Databricks destinations are **alternatives, not a migration**: a UC Volume is the
one to prefer where the app holds ``USE CATALOG`` (governed, auditable, and the namespace
the rest of the platform reads), and DBFS is what a workspace that cannot grant it uses
instead. Which one is in force is a single configured value, so the answer can change
without a code change — see :func:`_workspace_connection` for why it is a single value and
not a second set of credentials.
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


#: What it takes to REACH a workspace, as opposed to where in it a blob lands. Identical
#: for every Databricks destination, so it is declared once under `storage.databricks`.
_CONNECTION_KEYS = ("host", "token_env", "verify_ssl")


def _workspace_connection(cfg: dict, block: str) -> dict:
    """``storage.<block>`` over the connection keys of ``storage.databricks``.

    Switching Databricks destinations must be a one-value edit. A UC Volume and a DBFS
    path differ only in *where* they are — same workspace, same token, same TLS posture —
    so re-declaring the connection under each block would leave a deployment able to
    change `backend` and land on credentials it stopped maintaining. The narrower block
    still wins on every key, including these three, for the run that genuinely reaches a
    second workspace.
    """
    shared = cfg.get("databricks") or {}
    merged = {k: shared[k] for k in _CONNECTION_KEYS if k in shared}
    merged.update(cfg.get(block) or {})
    return merged


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

    if backend in ("dbfs", "databricks_dbfs"):
        try:
            from src.storage.dbfs import DbfsStorage
        except Exception as exc:  # noqa: BLE001 — import error, anything
            logger.error(
                "storage.backend is 'dbfs' but that backend could not be loaded (%s). "
                "Falling back to LOCAL DISK, which does NOT survive a container "
                "restart — pending approvals may be lost.",
                exc,
            )
            return LocalStorage(root=cfg.get("root") or None)
        # `storage.dbfs`, never `storage.root`: that one names a FILESYSTEM path for the
        # local backend, and a DBFS path is not one — reading it here would send a
        # deployment that set both to the wrong place without saying so.
        return DbfsStorage(_workspace_connection(cfg, "dbfs"))

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
        "volume), 'databricks' (a managed OR external Unity Catalog Volume), 'dbfs' "
        "(workspace storage, no UC grant needed) and 'sql'.",
        backend,
    )
    return LocalStorage(root=cfg.get("root") or None)
