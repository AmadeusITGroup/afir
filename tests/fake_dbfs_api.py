"""A local HTTP server speaking the seven DBFS calls ``DbfsStorage`` makes.

Sibling of ``fake_files_api.py`` and a server for the same reason: patching ``_dbfs`` would
leave untested exactly the parts that were hard to get right — base64 in both directions,
the 1 MiB ceiling that forces a streamed write and a looping read, ``bytes_read`` as the
API's own count, a read past the end being an ERROR rather than an empty answer, and
``delete`` answering 200 for a path that never existed.

Every shape below is measured against a real workspace (2026-08-31):

    POST /api/2.0/dbfs/put        {path, contents:b64, overwrite}  -> 200 {}
                                  contents > 1048576 raw bytes     -> 400 MAX_BLOCK_SIZE_EXCEEDED
                                  exists and not overwrite         -> 400 RESOURCE_ALREADY_EXISTS
    POST /api/2.0/dbfs/create     {path, overwrite}                -> 200 {"handle": N}
    POST /api/2.0/dbfs/add-block  {handle, data:b64}               -> 200 {}
                                  data > 1048576 raw bytes         -> 400 MAX_BLOCK_SIZE_EXCEEDED
    POST /api/2.0/dbfs/close      {handle}                         -> 200 {}, or 404 if not open
    GET  /api/2.0/dbfs/read       ?path&offset&length              -> 200 {bytes_read, data:b64}
                                  offset == file_size              -> 200 {"bytes_read":0,"data":""}
                                  offset >  file_size              -> 400 INVALID_PARAMETER_VALUE
                                  length > 1048576                 -> 400 MAX_READ_SIZE_EXCEEDED
    GET  /api/2.0/dbfs/get-status ?path   -> 200 {path,is_dir,file_size,modification_time}  # ms
    GET  /api/2.0/dbfs/list       ?path   -> 200 {"files":[...]}, {} when the dir is EMPTY
    POST /api/2.0/dbfs/delete     {path, recursive}  -> 200 {} whether or not it existed

    absence on read / get-status / list  -> 404 RESOURCE_DOES_NOT_EXIST

Both write paths create parent directories, which is why nothing here needs ``mkdirs``.

The same gap as its sibling, stated rather than papered over: this proves the backend
correct against the semantics a probe recorded. If the real API diverges the suite passes
and the deployment fails, which is why the shapes are written out above.
"""

import base64
import itertools
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_PREFIX = "/api/2.0/dbfs/"

#: The API's own per-call ceiling, in raw bytes. Enforced here, or the backend's whole
#: reason for streaming and looping goes unexercised.
MAX_BLOCK = 1024 * 1024


class FakeDbfs:
    """The stored state: a flat ``path -> (bytes, mtime_ms)`` map, as DBFS presents itself.

    Flat for the same reason the Volume fake is: a listing is *derived* from the paths, so
    "one level per call" is real rather than simulated, and a directory exists exactly when
    something lives under it.
    """

    def __init__(self):
        self.files = {}
        self.lock = threading.Lock()
        #: Every request as ``(op, payload)``, in order, so a test can assert on the call
        #: pattern — that a small write took one `put` and not three, that a delete asked
        #: `get-status` first, that a read did not keep going past a short block.
        self.calls = []
        #: Seconds to stall inside a committing write (`put` and `close`). The only way to
        #: hold a write *in flight* long enough to assert on that window.
        self.put_delay = 0.0
        #: Open streams from `create`, ``handle -> (path, [chunks])``.
        self.streams = {}
        self._handles = itertools.count(1000)

    # -- state -------------------------------------------------------------

    def put(self, path, blob):
        with self.lock:
            # Milliseconds, monotonically increasing per write, so an ordering assertion
            # cannot pass by accident where two writes share a millisecond.
            now = int(time.time() * 1000)
            previous = self.files.get(path)
            if previous is not None and now <= previous[1]:
                now = previous[1] + 1
            self.files[path] = (bytes(blob), now)

    def get(self, path):
        with self.lock:
            entry = self.files.get(path)
            return entry[0] if entry else None

    def stat(self, path):
        with self.lock:
            entry = self.files.get(path)
            if entry is not None:
                return {
                    "path": path,
                    "is_dir": False,
                    "file_size": len(entry[0]),
                    "modification_time": entry[1],
                }
            prefix = path.rstrip("/") + "/"
            if any(p.startswith(prefix) for p in self.files):
                return {
                    "path": path,
                    "is_dir": True,
                    "file_size": 0,
                    "modification_time": 0,
                }
        return None

    def delete(self, path):
        with self.lock:
            return self.files.pop(path, None) is not None

    def listdir(self, path):
        """One level under ``path``, or ``None`` when nothing lives under it at all."""
        prefix = path.rstrip("/") + "/"
        with self.lock:
            stats = dict(self.files)
        under = [p for p in stats if p.startswith(prefix)]
        if not under:
            return None
        entries = {}
        for full in under:
            head, _, tail = full[len(prefix) :].partition("/")
            child = prefix + head
            if tail:
                entries[child] = {
                    "path": child,
                    "is_dir": True,
                    "file_size": 0,
                    "modification_time": stats[full][1],
                }
            else:
                blob, mtime = stats[full]
                entries[child] = {
                    "path": child,
                    "is_dir": False,
                    "file_size": len(blob),
                    "modification_time": mtime,
                }
        return list(entries.values())

    def open_stream(self, path):
        with self.lock:
            handle = next(self._handles)
            self.streams[handle] = (path, [])
            return handle


class _Handler(BaseHTTPRequestHandler):
    dbfs: FakeDbfs = None  # set by serve_fake_dbfs

    # -- plumbing ----------------------------------------------------------

    def log_message(self, *args):
        """Silent: pytest captures stderr and a request log per call is noise."""

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, code, message=""):
        return self._json(status, {"error_code": code, "message": message or code})

    def _absent(self, path):
        return self._error(
            404,
            "RESOURCE_DOES_NOT_EXIST",
            f"No file or directory exists on path {path}.",
        )

    def _authorized(self):
        """A bearer token must be present. Absent one, 401 — as the real API answers.

        Not cosmetic: ``_token()`` returning ``None`` is a real deployment state (an unset
        ``token_env``), and a fake that served the request anyway would let the backend look
        healthy in exactly the case an operator most needs it to complain.
        """
        header = self.headers.get("Authorization") or ""
        if header.startswith("Bearer ") and header[7:].strip():
            return True
        self._error(401, "UNAUTHENTICATED")
        return False

    def _op(self):
        parsed = urllib.parse.urlparse(self.path)
        if not parsed.path.startswith(_PREFIX):
            return None, {}
        query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        return parsed.path[len(_PREFIX) :], query

    # -- the calls ---------------------------------------------------------

    def do_GET(self):
        op, query = self._op()
        self.dbfs.calls.append((op, query))
        if not self._authorized():
            return
        if op == "read":
            return self._read(query)
        if op == "get-status":
            entry = self.dbfs.stat(query.get("path") or "")
            return self._json(200, entry) if entry else self._absent(query.get("path"))
        if op == "list":
            entries = self.dbfs.listdir(query.get("path") or "")
            if entries is None:
                return self._absent(query.get("path"))
            # An EMPTY directory answers `{}` with no `files` key — measured, and the one
            # shape a `body["files"]` would crash on.
            return self._json(200, {"files": entries} if entries else {})
        return self._error(404, "ENDPOINT_NOT_FOUND", op or "")

    def do_POST(self):
        op, _ = self._op()
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            payload = {}
        self.dbfs.calls.append((op, payload))
        if not self._authorized():
            return
        if op == "put":
            return self._put(payload)
        if op == "create":
            return self._json(
                200, {"handle": self.dbfs.open_stream(str(payload.get("path") or ""))}
            )
        if op == "add-block":
            return self._add_block(payload)
        if op == "close":
            return self._close(payload)
        if op == "delete":
            # 200 whether or not it was there. The backend asks `get-status` first for
            # exactly this reason; a fake that 404'd here would hide that it has to.
            self.dbfs.delete(str(payload.get("path") or ""))
            return self._json(200, {})
        if op == "mkdirs":
            return self._json(200, {})
        return self._error(404, "ENDPOINT_NOT_FOUND", op or "")

    # -- call bodies -------------------------------------------------------

    def _read(self, query):
        path = query.get("path") or ""
        blob = self.dbfs.get(path)
        if blob is None:
            return self._absent(path)
        try:
            offset = int(query.get("offset") or 0)
            length = int(query.get("length") or 0)
        except ValueError:
            return self._error(400, "INVALID_PARAMETER_VALUE", "offset/length")
        if length > MAX_BLOCK:
            return self._error(
                400,
                "MAX_READ_SIZE_EXCEEDED",
                f"Cannot read more than {MAX_BLOCK} bytes in one read, please paginate "
                f"your request. Request: {length} bytes.",
            )
        if offset > len(blob):
            # An error, not an empty answer: the download loop must stop on a short block
            # rather than probing for the end, and this is what punishes it if it does not.
            return self._error(
                400,
                "INVALID_PARAMETER_VALUE",
                f"Cannot read when offset is greater than file size. Found offset = {offset}",
            )
        chunk = blob[offset : offset + length]
        return self._json(
            200,
            {
                "bytes_read": len(chunk),
                "data": base64.b64encode(chunk).decode("ascii"),
            },
        )

    def _put(self, payload):
        path = str(payload.get("path") or "")
        try:
            blob = base64.b64decode(payload.get("contents") or "")
        except Exception:  # noqa: BLE001
            return self._error(400, "INVALID_PARAMETER_VALUE", "contents")
        if len(blob) > MAX_BLOCK:
            return self._error(
                400,
                "MAX_BLOCK_SIZE_EXCEEDED",
                f"The 'contents' data cannot exceed {MAX_BLOCK} bytes. Found: "
                f"{len(blob)} bytes. You might want to use streaming upload instead.",
            )
        if self.dbfs.get(path) is not None and not payload.get("overwrite"):
            # The real API refuses this, and `put_text`'s contract is *replace*: an upload
            # that silently declined would make every save after the first a no-op
            # reporting success.
            return self._error(
                400,
                "RESOURCE_ALREADY_EXISTS",
                f"A file or directory already exists at the input path {path}.",
            )
        if self.dbfs.put_delay:
            time.sleep(self.dbfs.put_delay)
        self.dbfs.put(path, blob)
        return self._json(200, {})

    def _add_block(self, payload):
        handle = payload.get("handle")
        with self.dbfs.lock:
            stream = self.dbfs.streams.get(handle)
        if stream is None:
            return self._error(
                404, "RESOURCE_DOES_NOT_EXIST", f"No such stream with handle: {handle}."
            )
        try:
            chunk = base64.b64decode(payload.get("data") or "")
        except Exception:  # noqa: BLE001
            return self._error(400, "INVALID_PARAMETER_VALUE", "data")
        if len(chunk) > MAX_BLOCK:
            return self._error(
                400,
                "MAX_BLOCK_SIZE_EXCEEDED",
                f"The data cannot exceed {MAX_BLOCK} bytes. Found: {len(chunk)} bytes.",
            )
        stream[1].append(chunk)
        return self._json(200, {})

    def _close(self, payload):
        handle = payload.get("handle")
        with self.dbfs.lock:
            stream = self.dbfs.streams.pop(handle, None)
        if stream is None:
            # A re-close is a 404, measured — the handle is gone, not idempotently fine.
            return self._error(
                404, "RESOURCE_DOES_NOT_EXIST", f"No such stream with handle: {handle}."
            )
        if self.dbfs.put_delay:
            time.sleep(self.dbfs.put_delay)
        # The object appears at close, so a stream that failed part-way leaves nothing
        # rather than a partial file. That is what makes `close` the committing call.
        self.dbfs.put(stream[0], b"".join(stream[1]))
        return self._json(200, {})


def serve_fake_dbfs():
    """Start the server on an ephemeral port. Returns ``(base_url, dbfs, shutdown)``."""
    dbfs = FakeDbfs()
    handler = type("BoundHandler", (_Handler,), {"dbfs": dbfs})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def shutdown():
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    host, port = server.server_address[:2]
    return f"http://{host}:{port}", dbfs, shutdown
