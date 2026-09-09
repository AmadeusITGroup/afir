"""Durable storage on a Unity Catalog Volume, for the Databricks App deployment.

Writes queue through one background thread; reads consult the queue first so ``put``
followed by ``get`` is not a coin flip. Single-writer: ``flush`` must return before
"saved" means "durable", and only one replica is safe.
"""

import json
import logging
import os
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.storage.base import StorageBackend, StoredObject, key_prefix, safe_key

logger = logging.getLogger(__name__)

#: Same role as ``LocalStorage``'s ``.prev``; never reported as a stored object.
_PREV_SUFFIX = ".prev"

#: Distinct keys the queue holds before a write is refused. Reaching this limit means
#: the remote is failing, not that the pipeline is busy.
_MAX_PENDING_KEYS = 512

#: How long a caller waits for queue room before its write is refused (refused, not dropped:
#: ``False`` reaches :attr:`degradation`; a silent drop would not).
_ENQUEUE_WAIT_SECONDS = 30.0

#: Per-call timeout for the Files API; a slow call costs latency, not correctness.
_HTTP_TIMEOUT = 120


def _normalise_host(value: str) -> str:
    """A URL ``requests`` can actually use.

    A host with no scheme raises ``MissingSchema`` inside the blanket ``except`` every call
    below is wrapped in, which reads as a transport error — the same ``HTTP -1`` an absent
    token produces, on a store that is otherwise correctly configured. The platform's
    injected ``DATABRICKS_HOST`` is not guaranteed to carry one. An explicit ``http://`` is
    left alone: the contract suite's fake servers are plain HTTP on localhost, and silently
    upgrading a scheme somebody wrote is a different decision from supplying a missing one.
    """
    host = (value or "").strip().rstrip("/")
    if not host or "://" in host:
        return host
    return f"https://{host}"


@dataclass
class _Pending:
    """One queued write. ``text`` and ``blob`` are mutually exclusive."""

    key: str
    text: Optional[str] = None
    blob: Optional[bytes] = None
    verify: Optional[Callable[[str], object]] = None

    @property
    def payload(self) -> bytes:
        if self.blob is not None:
            return self.blob
        return (self.text or "").encode("utf-8")


@dataclass
class _Stats:
    """What the writer thread has done, for :attr:`degradation` and the logs."""

    written: int = 0
    failed: int = 0
    refused: int = 0
    dropped: int = 0
    verify_failures: int = 0
    last_error: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class DatabricksStorage(StorageBackend):
    """Named blobs as files on a UC Volume, written through one background thread."""

    kind = "databricks"

    def __init__(self, config: Optional[dict] = None):
        cfg = dict(config or {})
        self._catalog = str(cfg.get("catalog") or "").strip()
        self._schema = str(cfg.get("schema") or "afir").strip()
        self._volume = str(cfg.get("volume") or "state").strip()
        self._token_env = str(cfg.get("token_env") or "DATABRICKS_TOKEN").strip()
        # Defaults to True; App containers see the public certificate chain, unlike log-source backends.
        self._verify_ssl = bool(cfg.get("verify_ssl", True))
        self._auth = cfg.get("auth")
        #: What to call `self._auth` in a log line. Recorded where the object is adopted, not
        #: where it is used: `_sdk_auth` assigns to `self._auth`, so from `_token`'s first
        #: branch a token the SDK resolved is indistinguishable from a configured one.
        self._auth_channel = "configured auth" if self._auth is not None else ""
        #: Set once `try_build_auth` has been attempted, so a workspace with no SDK pays for
        #: the import and the probe once rather than on every blob.
        self._sdk_auth_tried = False
        #: Which channel last answered `_token()`, reported once when it changes. Not probed
        #: at construction: `_sdk_auth` would reach the machine's own credentials from inside
        #: every test that builds a backend, and the boot line below says so rather than guessing.
        self._token_channel: Optional[str] = None
        configured_host = str(cfg.get("host") or "").strip().rstrip("/")
        # Normalised at the one place every channel converges on, configured or resolved.
        self._host = _normalise_host(configured_host or self._resolve_host())

        self._root = self._resolve_root(cfg)

        #: Queued writes, newest payload per key; insertion-ordered so the writer drains oldest-first.
        self._pending: Dict[str, _Pending] = {}
        #: Write currently in flight; reads merge it so ``put`` then ``get`` is not a coin flip.
        self._inflight: Dict[str, _Pending] = {}
        self._cond = threading.Condition()
        self._stopped = False
        self._stats = _Stats()
        #: Last payload this process wrote per key; refreshing ``.prev`` then costs one upload, not two.
        self._last_good: Dict[str, bytes] = {}
        self._local = threading.local()
        self._sessions: List[object] = []
        self._session_lock = threading.Lock()
        self._fatal: Optional[str] = None

        if not self._root:
            self._fatal = self._no_root_reason()
        elif not self._host:
            self._fatal = (
                "no Databricks host could be resolved (set storage.databricks.host or "
                "DATABRICKS_HOST)"
            )

        if self._fatal:
            # Not raised: build_storage catches and falls back; the operator needs a log line.
            logger.error("Databricks storage is unusable: %s", self._fatal)

        self._writer = threading.Thread(
            target=self._run_writer, name="afir-storage-writer", daemon=True
        )
        self._writer.start()
        logger.info(
            "%s storage: %s on %s (verify_ssl=%s, token from %s, writer thread started)",
            self.kind,
            self._root or "<unconfigured>",
            self._host or "<no host resolved>",
            self._verify_ssl,
            self._token_channels(),
        )

    def _token_channels(self) -> str:
        """Which bearer-token channels could answer, without asking any of them.

        The boot line named only the root, so a store that writes nothing logged identically
        to a working one and the two causes of ``HTTP -1`` — an unresolvable host and an
        unobtainable token — were indistinguishable from the log. This says which channels
        exist; :meth:`_note_token_channel` then says which one actually answered.
        """
        channels = []
        if self._auth is not None:
            channels.append(self._auth_channel or "configured auth")
        if os.environ.get(self._token_env):
            channels.append(self._token_env)
        # Unknowable without building it, and an App's only channel, so it is always a
        # candidate rather than a claim.
        channels.append("SDK auth (per call)")
        return ", ".join(channels)

    def _note_token_channel(self, channel: str) -> None:
        """Report the channel that answered, once, and again only if it changes.

        A channel that changes mid-run is worth a line of its own: it means a configured
        token stopped working and the SDK took over, or the reverse.
        """
        if channel == self._token_channel:
            return
        self._token_channel = channel
        logger.info("%s storage: bearer token from %s", self.kind, channel)

    # -- configuration -----------------------------------------------------

    def _resolve_root(self, cfg: dict) -> str:
        """The remote directory every key hangs off, or ``""`` when unconfigured.

        A seam rather than an expression because :class:`~src.storage.dbfs.DbfsStorage`
        reuses everything below it and differs in this and four HTTP calls.
        """
        if not self._catalog:
            return ""
        return f"/Volumes/{self._catalog}/{self._schema}/{self._volume}"

    def _no_root_reason(self) -> str:
        """Why :meth:`_resolve_root` answered nothing, for the operator's log line."""
        return (
            "storage.databricks.catalog is not set, so there is no Volume path to write to"
        )

    def _resolve_host(self) -> str:
        """The workspace URL, preferring SDK resolution so an App needs no configured host.

        A PAT is workspace-scoped; a static host in YAML fails as a 403 on the wrong workspace.
        """
        if self._auth is not None:
            try:
                return str(self._auth.host).rstrip("/")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Configured auth could not report a host: %s", exc)
        for var in ("DATABRICKS_HOST", "AFIR_DATABRICKS_HOST"):
            value = os.environ.get(var)
            if value:
                return value.strip().rstrip("/")
        auth = self._sdk_auth()
        if auth is not None:
            try:
                return str(auth.host).rstrip("/")
            except Exception as exc:  # noqa: BLE001
                logger.debug("SDK auth could not report a host: %s", exc)
        return ""

    def _sdk_auth(self):
        """The unified-auth object, built at most once. ``None`` when the SDK is absent."""
        if self._auth is None and not self._sdk_auth_tried:
            self._sdk_auth_tried = True
            try:
                from src.utils.databricks_auth import try_build_auth

                self._auth = try_build_auth()
                if self._auth is not None:
                    self._auth_channel = "SDK auth"
            except Exception as exc:  # noqa: BLE001
                logger.debug("SDK auth unavailable for storage: %s", exc)
        return self._auth

    def _token(self) -> Optional[str]:
        """A currently-valid bearer token, resolved per call so a short-lived OAuth token never expires cached."""
        if self._auth is not None:
            try:
                token = self._auth.token()
                self._note_token_channel(self._auth_channel or "configured auth")
                return token
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "SDK token refresh failed; trying the static token: %s", exc
                )
        static = os.environ.get(self._token_env)
        if static:
            self._note_token_channel(self._token_env)
            return static
        # AN APP HAS NO PAT, and reaching here does not mean it is misconfigured. `host` is
        # answered by the platform's injected `DATABRICKS_HOST` (or a configured value), so
        # `_resolve_host` returns before it ever builds auth and `self._auth` stays None —
        # leaving the static env var as the only candidate, which an App must never set
        # (`DATABRICKS_TOKEN` beside the injected OAuth vars makes the SDK refuse both).
        # Without this the backend answers -1 to every call while the app logs storage as
        # configured, and the loss shows up one restart later as vanished approvals.
        auth = self._sdk_auth()
        if auth is None:
            return None
        try:
            token = auth.token()
            self._note_token_channel(self._auth_channel or "SDK auth")
            return token
        except Exception as exc:  # noqa: BLE001
            logger.debug("SDK token refresh failed and no static token is set: %s", exc)
            return None

    def _no_token_reason(self) -> str:
        """Why there is no bearer token, naming both channels rather than only the env var.

        An App fails here with `DATABRICKS_TOKEN` correctly unset, so a message about that
        variable sends the reader to change the one thing that must not change.
        """
        return (
            f"no bearer token: {self._token_env} is unset and SDK auth is unavailable "
            "(in a Databricks App the injected OAuth credentials should supply it)"
        )

    @property
    def root(self) -> Optional[str]:
        """The Volume path, for the operator-facing "where does state live" answer."""
        return self._root or None

    # -- HTTP --------------------------------------------------------------

    def _http(self, method: str, path: str, *, body=None, params=None):
        """One Files API call. Returns ``(status, bytes)``; a negative status is a transport error.
        Never raises.
        """
        if self._fatal:
            return -1, b""
        token = self._token()
        if not token:
            self._note_error(self._no_token_reason())
            return -1, b""
        session = self._get_session()
        # Quoted because a key may carry characters legal in safe_key and special in a
        # URL; safe="/" keeps the path structure the API needs.
        url = f"{self._host}/api/2.0/fs{urllib.parse.quote(path, safe='/')}"
        headers = {"Authorization": f"Bearer {token}"}
        if body is not None:
            headers["Content-Type"] = "application/octet-stream"
        try:
            resp = session.request(
                method,
                url,
                data=body,
                headers=headers,
                params=params,
                timeout=_HTTP_TIMEOUT,
                verify=self._verify_ssl,
            )
            return resp.status_code, resp.content
        except Exception as exc:  # noqa: BLE001 — a transport failure is an outcome
            logger.debug("%s %s failed: %s", method, path, exc)
            self._note_error(f"{type(exc).__name__}: {exc}")
            return -1, b""

    def _get_session(self):
        """A ``requests.Session`` per thread. A shared session's connection pool is not
        thread-safe; interleaved responses on one socket look like a verification failure.
        """
        session = getattr(self._local, "session", None)
        if session is None:
            import requests

            session = requests.Session()
            self._local.session = session
            with self._session_lock:
                self._sessions.append(session)
        return session

    def _note_error(self, message: str) -> None:
        with self._stats.lock:
            self._stats.last_error = message[:300]

    def _vol(self, key: str) -> str:
        return f"{self._root}/{key}"

    # -- remote primitives -------------------------------------------------

    def _upload(self, key: str, blob: bytes) -> bool:
        """PUT with ``overwrite=true``. A Volume has no ``os.replace``, so the flag is not optional."""
        status, _ = self._http(
            "PUT", f"/files{self._vol(key)}", body=blob, params={"overwrite": "true"}
        )
        if status in (200, 204):
            return True
        logger.warning("Upload of %s failed with HTTP %s", key, status)
        self._note_error(f"upload {key}: HTTP {status}")
        return False

    def _download(self, key: str) -> Optional[bytes]:
        status, body = self._http("GET", f"/files{self._vol(key)}")
        if status == 200:
            return body
        if status != 404:
            # 404 is absence; a 403 read as "not there" turns a permission problem into an empty queue.
            logger.warning(
                "Read of %s returned HTTP %s — treating it as absent, which it may not "
                "be.",
                key,
                status,
            )
            self._note_error(f"read {key}: HTTP {status}")
        return None

    def _remove(self, key: str) -> bool:
        status, _ = self._http("DELETE", f"/files{self._vol(key)}")
        return status in (200, 204)

    def _listdir(self, path: str) -> List[dict]:
        """One directory level. Measured: 404 for an absent directory, never an error."""
        status, body = self._http("GET", f"/directories{path}")
        if status != 200:
            return []
        try:
            return (json.loads(body or b"{}") or {}).get("contents") or []
        except ValueError:
            logger.debug("Directory listing of %s did not parse", path)
            return []

    # -- the writer thread -------------------------------------------------

    def _run_writer(self) -> None:
        """Drain the queue forever. Single-thread serialisation is what makes ``append_text``'s
        read-modify-write safe and the single-replica constraint load-bearing.
        """
        while True:
            with self._cond:
                while not self._pending and not self._stopped:
                    self._cond.wait(timeout=1.0)
                if self._stopped and not self._pending:
                    return
                key = next(iter(self._pending))
                item = self._pending.pop(key)
                # Visible to readers for the whole upload, invisible to nobody.
                self._inflight[key] = item
                self._cond.notify_all()
            try:
                self._write_now(item)
            except Exception as exc:  # noqa: BLE001 — this thread must never die
                logger.warning("Storage writer failed on %s: %s", item.key, exc)
                with self._stats.lock:
                    self._stats.failed += 1
                    self._stats.last_error = f"{item.key}: {exc}"[:300]
            finally:
                with self._cond:
                    self._inflight.pop(key, None)
                    # Notified even on failure: a doomed write must not block a gate announcement.
                    self._cond.notify_all()

    def _write_now(self, item: _Pending) -> bool:
        """Upload one entry, keeping ``.prev`` and verifying the read-back."""
        key = item.key
        payload = item.payload

        if item.verify is not None:
            self._refresh_prev(key)

        if not self._upload(key, payload):
            with self._stats.lock:
                self._stats.failed += 1
            return False

        if item.verify is not None and not self._verify_readback(key, item):
            return False

        self._last_good[key] = payload
        with self._stats.lock:
            self._stats.written += 1
        return True

    def _refresh_prev(self, key: str) -> None:
        """Copy current content to ``<key>.prev`` before overwriting. Uses the in-process
        cache when available; falls back to a download on the first write after a restart.
        """
        blob = self._last_good.get(key)
        if blob is None:
            blob = self._download(key)
        if blob:
            self._upload(f"{key}{_PREV_SUFFIX}", blob)

    def _verify_readback(self, key: str, item: _Pending) -> bool:
        """Read the uploaded bytes back and verify. Reverts to the last known-good copy on failure.
        The read-back catches a truncated upload that the in-hand bytes cannot detect.
        """
        got = self._download(key)
        try:
            if got is None:
                raise ValueError("the object could not be read back")
            item.verify(got.decode("utf-8"))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Stored object %s did not verify after upload (%s); restoring the "
                "previous content. This job's newest state is NOT persisted.",
                key,
                exc,
            )
            with self._stats.lock:
                self._stats.verify_failures += 1
                self._stats.last_error = f"{key} failed verify: {exc}"[:300]
            previous = self._last_good.get(key) or self._download(
                f"{key}{_PREV_SUFFIX}"
            )
            # The restore is also verified: an unverified restore is a silent corruption.
            if previous and self._upload(key, previous) and self._readback_ok(
                key, item, previous
            ):
                return False
            # Nothing to restore: removing is less bad than leaving an unparseable document.
            logger.error(
                "Could not restore %s to a parseable state; REMOVING it. That job will "
                "not come back after a restart.",
                key,
            )
            self._remove(key)
            return False

    def _readback_ok(self, key: str, item: _Pending, expected: bytes) -> bool:
        """Does the Volume now hold ``expected``, and does it verify? Never raises."""
        got = self._download(key)
        if got != expected:
            return False
        try:
            item.verify(got.decode("utf-8"))
            return True
        except Exception:  # noqa: BLE001 — the answer is False, not an exception
            return False

    # -- queueing ----------------------------------------------------------

    def _enqueue(self, item: _Pending) -> bool:
        """Add or replace a pending write. ``False`` only when nothing was accepted."""
        if self._fatal:
            return False
        deadline = time.monotonic() + _ENQUEUE_WAIT_SECONDS
        with self._cond:
            while (
                not self._stopped
                and item.key not in self._pending
                and len(self._pending) >= _MAX_PENDING_KEYS
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=min(remaining, 1.0))
            if self._stopped:
                logger.warning(
                    "Storage is closed; refusing to queue %s rather than accepting a "
                    "write nothing will perform.",
                    item.key,
                )
                with self._stats.lock:
                    self._stats.refused += 1
                return False
            if (
                item.key not in self._pending
                and len(self._pending) >= _MAX_PENDING_KEYS
            ):
                # Refused, not dropped: False reaches degradation; inline write breaks single-writer.
                logger.error(
                    "Storage queue is still at %d keys after %.0fs; REFUSING the write "
                    "of %s. %s is not keeping up and this state is not persisted.",
                    _MAX_PENDING_KEYS,
                    _ENQUEUE_WAIT_SECONDS,
                    item.key,
                    self._root,
                )
                with self._stats.lock:
                    self._stats.refused += 1
                return False
            # Replace in place: every job save is a full snapshot, so older payloads for the same key have no value.
            self._pending[item.key] = item
            self._cond.notify_all()
            return True

    def _peek(self, key: str) -> Optional[_Pending]:
        """The newest payload this process holds for ``key``, queued or in flight."""
        with self._cond:
            return self._pending.get(key) or self._inflight.get(key)

    # -- write -------------------------------------------------------------

    def put_text(
        self, key: str, text: str, verify: Optional[Callable[[str], object]] = None
    ) -> bool:
        validated = safe_key(key)
        if verify is not None:
            # Synchronous pre-check: a bad payload never enters the queue. Read-back is the writer's job.
            try:
                verify(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Refusing to store %s: it does not verify (%s)", key, exc
                )
                return False
        return self._enqueue(_Pending(key=validated, text=text, verify=verify))

    def put_bytes(self, key: str, blob: bytes) -> bool:
        return self._enqueue(_Pending(key=safe_key(key), blob=bytes(blob)))

    def append_text(self, key: str, text: str) -> bool:
        """Append emulated as read-modify-write (no native append on a Volume).
        Safe only because all writes go through one thread; that is the single-replica constraint.
        """
        validated = safe_key(key)
        with self._cond:
            queued = self._pending.get(validated)
            if queued is not None and queued.blob is None:
                queued.text = (queued.text or "") + text
                self._cond.notify_all()
                return True
        # Not queued but possibly in flight; get_text merges inflight, missing it drops a record.
        existing = self.get_text(validated)
        return self._enqueue(_Pending(key=validated, text=(existing or "") + text))

    # -- read --------------------------------------------------------------

    def get_text(self, key: str) -> Optional[str]:
        validated = safe_key(key)
        held = self._peek(validated)
        if held is not None:
            try:
                return held.payload.decode("utf-8")
            except UnicodeDecodeError:
                return None
        blob = self._download(validated)
        if blob is None:
            return None
        try:
            return blob.decode("utf-8")
        except UnicodeDecodeError as exc:
            logger.warning("Stored object %s is not valid UTF-8: %s", key, exc)
            return None

    def get_bytes(self, key: str) -> Optional[bytes]:
        validated = safe_key(key)
        held = self._peek(validated)
        if held is not None:
            return held.payload
        return self._download(validated)

    def get_previous_text(self, key: str) -> Optional[str]:
        """The ``.prev`` copy. Read straight from the Volume: its caller (``JobStore.load_all``)
        runs at startup before anything is queued.
        """
        blob = self._download(f"{safe_key(key)}{_PREV_SUFFIX}")
        if blob is None:
            return None
        try:
            return blob.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def exists(self, key: str) -> bool:
        validated = safe_key(key)
        if self._peek(validated) is not None:
            return True
        return self._download(validated) is not None

    def list_keys(self, prefix: str) -> List[StoredObject]:
        """Walk the Volume under ``prefix``, then merge in-process queued writes.
        One API call per directory level (no recursive listing API).
        """
        validated = key_prefix(prefix)
        if self._fatal:
            return []
        base = f"{self._root}/{validated}" if validated else self._root
        found: Dict[str, StoredObject] = {}
        queue = [base]
        while queue:
            current = queue.pop()
            for entry in self._listdir(current):
                path = str(entry.get("path") or "")
                if entry.get("is_directory"):
                    queue.append(path.rstrip("/"))
                    continue
                rel = path[len(self._root) + 1 :] if path.startswith(self._root) else ""
                if not rel or _is_internal(rel):
                    continue
                found[rel] = StoredObject(
                    key=rel,
                    size=int(entry.get("file_size") or 0),
                    mtime=_seconds(entry.get("last_modified")),
                )
        with self._cond:
            held = list(self._pending.values()) + list(self._inflight.values())
        for item in held:
            if _is_internal(item.key):
                continue
            if validated and not item.key.startswith(f"{validated}/"):
                continue
            existing = found.get(item.key)
            found[item.key] = StoredObject(
                key=item.key,
                size=len(item.payload),
                # Queue mtime is stale or absent; time.time() signals "newer than anything on the Volume".
                mtime=max(time.time(), existing.mtime if existing else 0.0),
            )
        return list(found.values())

    # -- housekeeping ------------------------------------------------------

    def delete(self, key: str) -> bool:
        """Remove ``key`` and its ``.prev``, cancelling any queued write first.
        A surviving queued write would resurrect the key seconds after the delete.
        """
        validated = safe_key(key)
        deadline = time.monotonic() + _HTTP_TIMEOUT
        with self._cond:
            cancelled = self._pending.pop(validated, None) is not None
            while validated in self._inflight and time.monotonic() < deadline:
                self._cond.wait(timeout=1.0)
            self._cond.notify_all()
        removed = self._remove(validated)
        self._remove(f"{validated}{_PREV_SUFFIX}")
        self._last_good.pop(validated, None)
        return removed or cancelled

    # -- durability --------------------------------------------------------

    def flush(self, timeout: float = 30.0) -> bool:
        """Block until nothing is queued or in flight. True when drained in time.
        Gate-opening saves call this; a pending approval that never reached the Volume is the failure to prevent.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while self._pending or self._inflight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error(
                        "Storage flush timed out with %d write(s) outstanding. State "
                        "meant to survive a restart may not have reached %s.",
                        len(self._pending) + len(self._inflight),
                        self._root,
                    )
                    return False
                self._cond.wait(timeout=min(remaining, 1.0))
        return True

    @property
    def degradation(self) -> Optional[str]:
        if self._fatal:
            return f"Databricks storage is not usable: {self._fatal}"
        with self._stats.lock:
            failed = self._stats.failed
            refused = self._stats.refused
            dropped = self._stats.dropped
            verify_failures = self._stats.verify_failures
            last = self._stats.last_error
        with self._cond:
            outstanding = len(self._pending) + len(self._inflight)
        problems = []
        if failed:
            problems.append(f"{failed} write(s) failed")
        if verify_failures:
            problems.append(f"{verify_failures} did not verify after upload")
        if refused:
            problems.append(f"{refused} were refused because the queue was full")
        if dropped:
            problems.append(f"{dropped} were still queued at shutdown and are lost")
        if not problems:
            return None
        detail = f" Last error: {last}" if last else ""
        return (
            f"Durable storage on {self._root} is degraded: "
            + ", ".join(problems)
            + f". {outstanding} write(s) outstanding now.{detail}"
        )

    def close(self, timeout: float = 10.0) -> None:
        """Drain under a bound, then stop the writer. Must not raise.
        Bounded: a short window between ``SIGTERM`` and ``SIGKILL``.
        """
        drained = self.flush(timeout=timeout)
        with self._cond:
            self._stopped = True
            remaining = len(self._pending) + len(self._inflight)
            if remaining:
                with self._stats.lock:
                    self._stats.dropped += remaining
                self._pending.clear()
            self._cond.notify_all()
        if not drained and remaining:
            logger.error(
                "Storage closed with %d write(s) unwritten after %.1fs. Those are lost, "
                "including any job state they carried.",
                remaining,
                timeout,
            )
        try:
            self._writer.join(timeout=min(timeout, 5.0))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Writer thread join failed: %s", exc)
        with self._session_lock:
            sessions, self._sessions = self._sessions, []
        for session in sessions:
            try:
                session.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Closing an HTTP session failed: %s", exc)


def _is_internal(key: str) -> bool:
    """A ``.prev`` is this backend's own mechanic, never a stored object."""
    return key.endswith(_PREV_SUFFIX)


def _seconds(last_modified) -> float:
    """``last_modified`` (milliseconds) as a POSIX second; ``0.0`` for unknown.
    Raw milliseconds would place every object far in the future and ``prune`` would expire nothing.
    """
    try:
        value = float(last_modified)
    except (TypeError, ValueError):
        return 0.0
    if value <= 0:
        return 0.0
    # A millisecond epoch for any plausible date exceeds 10^12; a second epoch is ~1.7e9.
    return value / 1000.0 if value > 10_000_000_000 else value
