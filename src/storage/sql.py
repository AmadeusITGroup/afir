"""Durable storage in a transactional SQL database: one row per named blob.

Dialects: ``sqlite`` and ``postgresql`` (warehouses refused; see ``docs/architecture/storage.md``).
Writes are synchronous so multiple replicas are safe; atomic upsert and one-statement append
are what the Volume backend cannot offer.

Key invariants: ``verify`` runs inside the transaction; ``prev_body`` is a column so ``delete``
takes it; ``mtime`` is stamped in Python, never ``0.0`` for a present row (``prune`` skips
``0.0`` as unknown; the epoch value would delete the queue). One connection per thread (sqlite
forbids sharing; Postgres would interleave transactions). DSN from ``storage.sql.dsn_env``; a
libpq URL carries its password, so there is no ``dsn`` form field.
"""

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from src.storage.base import StorageBackend, StoredObject, key_prefix, safe_key

logger = logging.getLogger(__name__)

#: SQL identifier pattern. Validated, not escaped: it is interpolated into DDL/DML.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Default table name, prefixed to avoid collisions in a shared database.
DEFAULT_TABLE = "afir_blobs"


@dataclass(frozen=True)
class _Dialect:
    """The four things that differ between dialects; every statement is otherwise identical."""

    name: str
    placeholder: str  # parameter marker
    blob_type: str  # column type for the payload
    byte_length: str  # SQL function giving a blob's length in bytes


_DIALECTS = {
    "sqlite": _Dialect("sqlite", "?", "BLOB", "length"),
    # `length()` on a Postgres bytea counts bytes too, but octet_length says so.
    "postgresql": _Dialect("postgresql", "%s", "BYTEA", "octet_length"),
}

#: Accepted spellings so ``postgres`` / ``sqlite3`` work rather than falling back to local disk.
_DIALECT_ALIASES = {
    "sqlite": "sqlite",
    "sqlite3": "sqlite",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "psql": "postgresql",
    "psycopg": "postgresql",
}


def _identifier(value: str, what: str) -> str:
    text = str(value or "").strip()
    parts = text.split(".") if text else []
    if not parts or not all(_IDENTIFIER.fullmatch(p) for p in parts):
        raise ValueError(
            f"storage.sql.{what} must be a SQL identifier (optionally schema-qualified), "
            f"got {value!r}"
        )
    return ".".join(parts)


class SqlStorage(StorageBackend):
    """Named blobs as rows in one table. Never raises except on a bad key."""

    kind = "sql"

    def __init__(self, config: Optional[dict] = None):
        cfg = dict(config or {})
        raw_dialect = str(cfg.get("dialect") or "sqlite").strip().lower()
        self._dialect_name = _DIALECT_ALIASES.get(raw_dialect, raw_dialect)
        # Config tab exposes dsn_env (the var name); an explicit dsn: still wins, redacted on read.
        self._dsn = str(cfg.get("dsn") or "").strip()
        self._dsn_env = str(cfg.get("dsn_env") or "AFIR_SQL_DSN").strip()
        if not self._dsn and self._dsn_env:
            self._dsn = str(os.environ.get(self._dsn_env) or "").strip()
        self._table = DEFAULT_TABLE
        self._fatal: Optional[str] = None
        self._last_error: Optional[str] = None
        self._local = threading.local()
        self._connections: List[object] = []
        self._conn_lock = threading.Lock()
        # Serialises prev_body read-modify-write; Postgres would otherwise read the same prior body concurrently.
        self._write_lock = threading.Lock()

        self._dialect = _DIALECTS.get(self._dialect_name)
        if self._dialect is None:
            self._fatal = (
                f"storage.sql.dialect {raw_dialect!r} is not supported; valid values are "
                f"{sorted(set(_DIALECT_ALIASES))}. See src/storage/sql.py for why a "
                "warehouse (Snowflake, Databricks SQL) is not among them."
            )
        else:
            try:
                self._table = _identifier(cfg.get("table") or DEFAULT_TABLE, "table")
            except ValueError as exc:
                self._fatal = str(exc)

        if self._fatal is None and not self._dsn:
            self._fatal = (
                f"no connection string: storage.sql.dsn is unset and ${self._dsn_env} is "
                "empty, so there is no database to connect to (sqlite: a file path; "
                "postgresql: a libpq URL)"
            )

        if self._fatal is None:
            try:
                self._connect()  # fail at boot, not on the first job save
                self._ensure_table()
            except Exception as exc:  # noqa: BLE001 — any driver error, absent or live
                self._fatal = f"cannot open the configured database: {exc}"

        if self._fatal:
            # Not raised: build_storage catches and falls back; the operator needs a log line.
            logger.error("SQL storage is unusable: %s", self._fatal)
        else:
            logger.info(
                "SQL storage: %s table %s (writes are synchronous and atomic; "
                "multi-replica safe)",
                self._dialect_name,
                self._table,
            )

    # -- connection --------------------------------------------------------

    def _connect(self):
        """This thread's connection, opened on first use.
        Per-thread: sqlite forbids sharing; Postgres would interleave transactions.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        if self._dialect_name == "sqlite":
            import sqlite3
            from pathlib import Path

            path = Path(self._dsn).expanduser()
            if path.parent and str(path.parent) not in ("", "."):
                path.parent.mkdir(parents=True, exist_ok=True)
            # isolation_level=None: explicit transactions so parse-back rollback is ours.
            conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
            # WAL lets a reader run while the writer holds the lock; the default journal blocks it.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
        else:
            try:
                import psycopg  # psycopg 3
            except ImportError:
                try:
                    import psycopg2 as psycopg  # noqa: N813 — same DB-API surface
                except ImportError as exc:
                    raise RuntimeError(
                        "the postgresql dialect needs a driver: "
                        "pip install 'psycopg[binary]'"
                    ) from exc
            conn = psycopg.connect(self._dsn)
            conn.autocommit = False
        with self._conn_lock:
            self._connections.append(conn)
        self._local.conn = conn
        return conn

    def _ensure_table(self) -> None:
        """Create the table if absent; idempotent. ``key`` as PK makes the upsert a single statement."""
        d = self._dialect
        conn = self._connect()
        cur = conn.cursor()
        try:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                f"  storage_key TEXT PRIMARY KEY,"
                f"  body {d.blob_type} NOT NULL,"
                f"  prev_body {d.blob_type},"
                f"  byte_size BIGINT NOT NULL,"
                f"  modified_at DOUBLE PRECISION NOT NULL"
                f")"
                if self._dialect_name != "sqlite"
                else f"CREATE TABLE IF NOT EXISTS {self._table} ("
                f"  storage_key TEXT PRIMARY KEY,"
                f"  body BLOB NOT NULL,"
                f"  prev_body BLOB,"
                f"  byte_size INTEGER NOT NULL,"
                f"  modified_at REAL NOT NULL"
                f")"
            )
            self._commit(conn)
        finally:
            cur.close()

    def _commit(self, conn) -> None:
        if self._dialect_name == "sqlite":
            # isolation_level=None: bare execute is autocommit; an explicit BEGIN is committed below.
            return
        conn.commit()

    def _fail(self, what: str, exc: Exception) -> None:
        self._last_error = f"{what}: {exc}"
        logger.warning("SQL storage %s", self._last_error)

    # -- write -------------------------------------------------------------

    def put_text(
        self, key: str, text: str, verify: Optional[Callable[[str], object]] = None
    ) -> bool:
        return self._put(key, str(text).encode("utf-8"), verify=verify)

    def put_bytes(self, key: str, blob: bytes) -> bool:
        """Store raw bytes without parse-back; a PDF has no cheap validity check."""
        return self._put(key, bytes(blob), verify=None)

    def _put(
        self, key: str, payload: bytes, verify: Optional[Callable[[str], object]]
    ) -> bool:
        validated = safe_key(key)
        if self._fatal:
            return False
        p = self._dialect.placeholder
        now = time.time()
        try:
            with self._write_lock:
                conn = self._connect()
                cur = conn.cursor()
                try:
                    if self._dialect_name == "sqlite":
                        cur.execute("BEGIN IMMEDIATE")
                    # prev_body updated in the same statement: no crash window where they are equal.
                    cur.execute(
                        f"INSERT INTO {self._table} "
                        f"  (storage_key, body, prev_body, byte_size, modified_at) "
                        f"VALUES ({p}, {p}, NULL, {p}, {p}) "
                        f"ON CONFLICT (storage_key) DO UPDATE SET "
                        f"  prev_body = {self._table}.body,"
                        f"  body = excluded.body,"
                        f"  byte_size = excluded.byte_size,"
                        f"  modified_at = excluded.modified_at",
                        (validated, payload, len(payload), now),
                    )
                    if verify is not None:
                        # Read back what the database now holds inside the transaction,
                        # so a truncated payload is rejected rather than committed.
                        cur.execute(
                            f"SELECT body FROM {self._table} WHERE storage_key = {p}",
                            (validated,),
                        )
                        row = cur.fetchone()
                        if row is None:
                            raise RuntimeError(
                                "row vanished between write and read-back"
                            )
                        verify(_as_bytes(row[0]).decode("utf-8"))
                    if self._dialect_name == "sqlite":
                        cur.execute("COMMIT")
                    else:
                        conn.commit()
                    return True
                except Exception:
                    try:
                        if self._dialect_name == "sqlite":
                            cur.execute("ROLLBACK")
                        else:
                            conn.rollback()
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Rollback after a failed put also failed: %s", exc)
                    raise
                finally:
                    cur.close()
        except Exception as exc:  # noqa: BLE001 — never fail a run over storage
            self._fail(f"failed writing {validated}", exc)
            return False

    def append_text(self, key: str, text: str) -> bool:
        """Append in a single atomic statement (``body || excluded.body``), safe across replicas.
        ``prev_body`` is not refreshed: for an append-only log a prior version is not a recovery.
        """
        validated = safe_key(key)
        if self._fatal:
            return False
        payload = str(text).encode("utf-8")
        p = self._dialect.placeholder
        blen = self._dialect.byte_length
        try:
            conn = self._connect()
            cur = conn.cursor()
            try:
                if self._dialect_name == "sqlite":
                    cur.execute("BEGIN IMMEDIATE")
                cur.execute(
                    f"INSERT INTO {self._table} "
                    f"  (storage_key, body, prev_body, byte_size, modified_at) "
                    f"VALUES ({p}, {p}, NULL, {p}, {p}) "
                    f"ON CONFLICT (storage_key) DO UPDATE SET "
                    f"  body = {self._table}.body || excluded.body,"
                    f"  byte_size = {blen}({self._table}.body || excluded.body),"
                    f"  modified_at = excluded.modified_at",
                    (validated, payload, len(payload), time.time()),
                )
                if self._dialect_name == "sqlite":
                    cur.execute("COMMIT")
                else:
                    conn.commit()
                return True
            except Exception:
                try:
                    if self._dialect_name == "sqlite":
                        cur.execute("ROLLBACK")
                    else:
                        conn.rollback()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Rollback after a failed append also failed: %s", exc)
                raise
            finally:
                cur.close()
        except Exception as exc:  # noqa: BLE001
            self._fail(f"failed appending to {validated}", exc)
            return False

    # -- read --------------------------------------------------------------

    def get_text(self, key: str) -> Optional[str]:
        blob = self.get_bytes(key)
        if blob is None:
            return None
        try:
            return blob.decode("utf-8")
        except UnicodeDecodeError as exc:
            self._fail(f"cannot decode {key} as text", exc)
            return None

    def get_bytes(self, key: str) -> Optional[bytes]:
        return self._column(key, "body")

    def get_previous_text(self, key: str) -> Optional[str]:
        blob = self._column(key, "prev_body")
        if blob is None:
            return None
        try:
            return blob.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def _column(self, key: str, column: str) -> Optional[bytes]:
        validated = safe_key(key)
        if self._fatal:
            return None
        p = self._dialect.placeholder
        try:
            cur = self._connect().cursor()
            try:
                cur.execute(
                    f"SELECT {column} FROM {self._table} WHERE storage_key = {p}",
                    (validated,),
                )
                row = cur.fetchone()
            finally:
                cur.close()
        except Exception as exc:  # noqa: BLE001
            self._fail(f"cannot read {validated}", exc)
            return None
        if row is None or row[0] is None:
            return None
        return _as_bytes(row[0])

    def exists(self, key: str) -> bool:
        validated = safe_key(key)
        if self._fatal:
            return False
        p = self._dialect.placeholder
        try:
            cur = self._connect().cursor()
            try:
                cur.execute(
                    f"SELECT 1 FROM {self._table} WHERE storage_key = {p}", (validated,)
                )
                return cur.fetchone() is not None
            finally:
                cur.close()
        except Exception as exc:  # noqa: BLE001
            self._fail(f"cannot check {validated}", exc)
            return False

    def list_keys(self, prefix: str) -> List[StoredObject]:
        """Everything under ``prefix``. Empty on failure. LIKE anchored with ``/`` so
        ``jobs`` cannot match ``jobs_archive/x.json``; ``%``/``_`` are excluded by ``safe_key``.
        """
        validated = key_prefix(prefix)
        if self._fatal:
            return []
        p = self._dialect.placeholder
        try:
            cur = self._connect().cursor()
            try:
                if validated:
                    cur.execute(
                        f"SELECT storage_key, byte_size, modified_at FROM {self._table} "
                        f"WHERE storage_key = {p} OR storage_key LIKE {p}",
                        (validated, f"{validated}/%"),
                    )
                else:
                    cur.execute(
                        f"SELECT storage_key, byte_size, modified_at FROM {self._table}"
                    )
                rows = cur.fetchall()
            finally:
                cur.close()
        except Exception as exc:  # noqa: BLE001
            self._fail(f"cannot list {prefix!r}", exc)
            return []
        return [
            StoredObject(key=str(r[0]), size=int(r[1] or 0), mtime=float(r[2] or 0.0))
            for r in rows
        ]

    # -- housekeeping ------------------------------------------------------

    def delete(self, key: str) -> bool:
        """Remove the row, taking ``prev_body`` with it so ``load_all``'s fallback cannot resurrect a deleted job."""
        validated = safe_key(key)
        if self._fatal:
            return False
        p = self._dialect.placeholder
        try:
            conn = self._connect()
            cur = conn.cursor()
            try:
                cur.execute(
                    f"DELETE FROM {self._table} WHERE storage_key = {p}", (validated,)
                )
                removed = (cur.rowcount or 0) > 0
            finally:
                cur.close()
            self._commit(conn)
            return removed
        except Exception as exc:  # noqa: BLE001
            self._fail(f"cannot delete {validated}", exc)
            return False

    # -- health ------------------------------------------------------------

    @property
    def degradation(self) -> Optional[str]:
        """Why durability is compromised, or ``None``. Fatal outranks transient: misconfiguration means nothing was ever stored."""
        if self._fatal:
            return (
                f"{self._fatal}. Job state, exports and the feedback log are NOT being "
                "stored in the database."
            )
        if self._last_error:
            return f"the last database operation failed ({self._last_error})"
        return None

    def flush(self, timeout: float = 30.0) -> bool:
        """No-op: every write is already committed when it returns. Present for the gate-durability contract."""
        return not self._fatal

    def close(self, timeout: float = 10.0) -> None:
        """Close every thread's connection. Must not raise."""
        with self._conn_lock:
            connections, self._connections = self._connections, []
        for conn in connections:
            try:
                conn.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not close a database connection: %s", exc)
        self._local = threading.local()


def _as_bytes(value) -> bytes:
    """Normalise the driver's blob type: sqlite gives ``bytes``, psycopg2 gives a ``memoryview``."""
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)
