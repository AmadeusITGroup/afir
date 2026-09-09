"""Durable storage on DBFS, for a workspace where the app holds no Unity Catalog grant.

Same machinery as the Volume backend — one writer thread, ``.prev``, verify-on-readback,
the bounded ``flush`` a gate waits on — over ``/api/2.0/dbfs/*``, which is authorised by
the workspace token alone. So this SUBCLASSES rather than repeats: a second copy of the
queue would be a second reading of durability, free to disagree with the first, and only
four calls actually differ.

The DBFS API is JSON-in/JSON-out where the Files API is path-and-body, and its measured
semantics differ in four ways that each fail silently if assumed away
(``tests/fake_dbfs_api.py`` encodes them):

* **1 MiB per call, in both directions.** ``put`` refuses ``contents`` above 1,048,576 raw
  bytes and ``read`` refuses that ``length``, both as HTTP 400. An evidence sidecar reaches
  2.07 MB, so ``create`` → ``add-block`` × N → ``close`` is the normal write path here and
  a download is always a loop.
* **A read past the end is an error, not an empty answer.** ``offset == file_size`` answers
  ``bytes_read: 0``; ``offset > file_size`` is a 400. The loop therefore stops on a short
  read and never probes for the end.
* **``delete`` of an absent path answers 200**, alone among these calls, so ``_remove``
  asks first — ``LocalStorage`` is the contract's oracle and it reports ``False`` for
  deleting nothing.
* **An empty directory lists as ``{}``** with no ``files`` key at all, while an absent one
  is a 404. Both are "no entries" to the walk above, but only one of them is absence.

Both write paths create their parent directories, so nothing here calls ``mkdirs``.
"""

import base64
import logging
from typing import List, Optional

from src.storage.databricks import _HTTP_TIMEOUT, DatabricksStorage

logger = logging.getLogger(__name__)

#: The API's own ceiling on one ``contents`` / ``data`` / ``length`` value, in RAW bytes —
#: the base64 expansion on the wire is not counted against it.
_MAX_BLOCK = 1024 * 1024

#: Where a deployment that names no root lands. Under ``/FileStore`` because that is the
#: one DBFS area a workspace user can also browse, which matters for a store an operator
#: may need to inspect by hand after a restart lost something.
DEFAULT_ROOT = "/FileStore/afir/state"


class DbfsStorage(DatabricksStorage):
    """Named blobs as DBFS files, written through the inherited single writer thread."""

    kind = "dbfs"

    def __init__(self, config: Optional[dict] = None):
        cfg = dict(config or {})
        # Read before `super().__init__`, which calls `_resolve_root` on its way through.
        self._configured_root = str(cfg.get("root") or "").strip().rstrip("/")
        super().__init__(cfg)

    # -- configuration -----------------------------------------------------

    def _resolve_root(self, cfg: dict) -> str:
        """Always a path, unlike the Volume backend: DBFS needs no catalog to exist."""
        root = self._configured_root or DEFAULT_ROOT
        return root if root.startswith("/") else f"/{root}"

    def _no_root_reason(self) -> str:  # pragma: no cover — `_resolve_root` never answers ""
        return "storage.dbfs.root resolved to nothing"

    # -- HTTP --------------------------------------------------------------

    def _dbfs(self, op: str, payload: dict, *, post: bool = True):
        """One DBFS API call. Returns ``(status, parsed)``; a negative status is a transport
        error. Never raises, and a body that will not parse is ``{}`` rather than an
        exception — the caller's next question is always about the status.
        """
        if self._fatal:
            return -1, {}
        token = self._token()
        if not token:
            self._note_error(self._no_token_reason())
            return -1, {}
        session = self._get_session()
        try:
            resp = session.request(
                "POST" if post else "GET",
                f"{self._host}/api/2.0/dbfs/{op}",
                json=payload if post else None,
                params=None if post else payload,
                headers={"Authorization": f"Bearer {token}"},
                timeout=_HTTP_TIMEOUT,
                verify=self._verify_ssl,
            )
            try:
                parsed = resp.json() if resp.content else {}
            except ValueError:
                parsed = {}
            return resp.status_code, parsed if isinstance(parsed, dict) else {}
        except Exception as exc:  # noqa: BLE001 — a transport failure is an outcome
            logger.debug("dbfs/%s failed: %s", op, exc)
            self._note_error(f"{type(exc).__name__}: {exc}")
            return -1, {}

    # -- remote primitives -------------------------------------------------

    def _upload(self, key: str, blob: bytes) -> bool:
        """One-shot ``put`` under the block ceiling, a stream over it."""
        path = self._vol(key)
        if len(blob) > _MAX_BLOCK:
            return self._upload_streaming(key, path, blob)
        status, body = self._dbfs(
            "put",
            {
                "path": path,
                "contents": base64.b64encode(blob).decode("ascii"),
                "overwrite": True,
            },
        )
        if status == 200:
            return True
        self._report_failure("upload", key, status, body)
        return False

    def _upload_streaming(self, key: str, path: str, blob: bytes) -> bool:
        """``create`` → ``add-block`` × N → ``close``, for a payload past the block ceiling.

        A failure part-way leaves the handle open and the object incomplete, and there is no
        abort call to fix that with — which is why the failure is reported loudly instead of
        quietly retried: for a verified write the read-back restores the previous content,
        and for an artifact the operator needs to know the file is not whole.
        """
        status, body = self._dbfs("create", {"path": path, "overwrite": True})
        handle = body.get("handle")
        if status != 200 or handle is None:
            self._report_failure("open a stream for", key, status, body)
            return False
        for start in range(0, len(blob), _MAX_BLOCK):
            chunk = blob[start : start + _MAX_BLOCK]
            status, body = self._dbfs(
                "add-block",
                {
                    "handle": handle,
                    "data": base64.b64encode(chunk).decode("ascii"),
                },
            )
            if status != 200:
                self._report_failure(
                    f"write bytes {start}-{start + len(chunk)} of", key, status, body
                )
                return False
        status, body = self._dbfs("close", {"handle": handle})
        if status != 200:
            # Every block landed and the object was never committed, which is the one
            # failure here that leaves nothing at all where a partial file would be.
            self._report_failure("close the stream for", key, status, body)
            return False
        return True

    def _download(self, key: str) -> Optional[bytes]:
        """Read the whole object, one block at a time. ``None`` for absent or unreadable."""
        path = self._vol(key)
        chunks: List[bytes] = []
        offset = 0
        while True:
            status, body = self._dbfs(
                "read",
                {"path": path, "offset": offset, "length": _MAX_BLOCK},
                post=False,
            )
            if status == 404:
                if offset:
                    # Absence at offset 0 is "there is nothing"; absence part-way means the
                    # object was removed underneath the read, and returning the fragment
                    # would hand a caller a truncated document as if it were whole.
                    logger.warning(
                        "%s disappeared %d byte(s) into a read; discarding the partial "
                        "content rather than returning it.",
                        key,
                        offset,
                    )
                    self._note_error(f"read {key}: vanished at offset {offset}")
                return None
            if status != 200:
                # A 403 read as "not there" turns a permission problem into an empty queue.
                logger.warning(
                    "Read of %s returned HTTP %s (%s) — treating it as absent, which it "
                    "may not be.",
                    key,
                    status,
                    _message(body),
                )
                self._note_error(f"read {key}: HTTP {status}")
                return None
            data = str(body.get("data") or "")
            try:
                chunk = base64.b64decode(data) if data else b""
            except Exception as exc:  # noqa: BLE001
                logger.warning("Read of %s returned undecodable base64: %s", key, exc)
                self._note_error(f"read {key}: undecodable payload")
                return None
            # The API's own count, not `len(chunk)`: a transfer that lost bytes would
            # otherwise read as a short block, end the loop, and return a whole-looking file.
            read = int(body.get("bytes_read") or 0)
            if read != len(chunk):
                logger.warning(
                    "Read of %s claimed %d byte(s) and carried %d; discarding it.",
                    key,
                    read,
                    len(chunk),
                )
                self._note_error(f"read {key}: {read} claimed, {len(chunk)} carried")
                return None
            chunks.append(chunk)
            offset += read
            if read < _MAX_BLOCK:
                # A short block is the end. Asking again would be `offset > file_size`,
                # which is a 400 and would discard everything read so far.
                break
        return b"".join(chunks)

    def _remove(self, key: str) -> bool:
        """``True`` only when something was there. ``delete`` alone answers 200 for an absent
        path, so the existence question is asked separately rather than inferred.
        """
        path = self._vol(key)
        status, _ = self._dbfs("get-status", {"path": path}, post=False)
        if status == 404:
            return False
        status, body = self._dbfs("delete", {"path": path, "recursive": False})
        if status == 200:
            return True
        self._report_failure("delete", key, status, body)
        return False

    def _listdir(self, path: str) -> List[dict]:
        """One directory level, in the shape the inherited walk reads.

        Measured: 404 for an absent directory and ``{}`` — no ``files`` key — for an empty
        one, so both arrive here as no entries.
        """
        status, body = self._dbfs("list", {"path": path}, post=False)
        if status != 200:
            return []
        entries = body.get("files") or []
        if not isinstance(entries, list):
            logger.debug("Directory listing of %s was not a list", path)
            return []
        out: List[dict] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            out.append(
                {
                    "path": str(entry.get("path") or ""),
                    "is_directory": bool(entry.get("is_dir")),
                    "file_size": entry.get("file_size") or 0,
                    # Milliseconds, which `_seconds` above narrows; raw, `prune` expires nothing.
                    "last_modified": entry.get("modification_time"),
                }
            )
        return out

    # -- reporting ---------------------------------------------------------

    def _report_failure(self, what: str, key: str, status: int, body: dict) -> None:
        """Log and record one failed call. The API's own message is carried through: the
        difference between a missing grant and an exceeded block size is in that string.
        """
        detail = _message(body)
        logger.warning("Failed to %s %s: HTTP %s (%s)", what, key, status, detail)
        self._note_error(f"{what} {key}: HTTP {status} {detail}")


def _message(body: dict) -> str:
    """The API's error text, or ``""``. ``error_code`` alone is enough to act on."""
    if not isinstance(body, dict):
        return ""
    code = str(body.get("error_code") or "")
    text = str(body.get("message") or "")
    return f"{code}: {text}"[:200] if code or text else ""
