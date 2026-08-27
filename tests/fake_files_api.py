"""A local HTTP server speaking the four Files API calls ``DatabricksStorage`` makes.

A server rather than a monkeypatched ``_http``, because patching the transport leaves untested the
things that were hard to get right: the URL shape, ``overwrite`` as a query parameter, a directory
listing being one level and flagging its subdirectories, ``last_modified`` in milliseconds, and
absence as a 404 rather than an exception. A double answering whatever the code asked for would
confirm the code's assumptions back to it.

Every response shape here is measured: ``scripts/probe_volume_semantics.py`` ran against a real
Unity Catalog Volume and this file mirrors what it observed.

    PUT  /api/2.0/fs/files/<path>?overwrite=true   -> 204, byte-exact on read-back
    GET  /api/2.0/fs/files/<path>                  -> 200 + body, or 404
    DELETE /api/2.0/fs/files/<path>                -> 204, or 404 when absent
    GET  /api/2.0/fs/directories/<path>            -> 200 {"contents": [...]}, or 404
      each entry: {path, name, is_directory, file_size, last_modified}   # ms

One gap, stated rather than papered over: this proves the backend correct against the semantics the
probe recorded. If the real API diverges, the suite passes and the deployment fails, which is why
the four shapes are written out above rather than left to the probe — it is a one-off instrument
pointed at a live Volume and is not in the repository (see CONTRIBUTING.md).
"""

import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_FILES = "/api/2.0/fs/files"
_DIRECTORIES = "/api/2.0/fs/directories"


class FakeVolume:
    """The stored state: a flat ``path -> (bytes, mtime_ms)`` map, as a Volume is.

    Flat on purpose. A Volume has no directories of its own — they are implied by the
    paths of the files in it — so a listing is derived here the same way the real one is,
    which is what makes the "one level per call" behaviour real rather than simulated.
    """

    def __init__(self):
        self.files = {}
        self.lock = threading.Lock()
        #: Every request, in order, so a test can assert on the call pattern (an upload
        #: without ``overwrite=true``, a listing that should not have recursed).
        self.calls = []
        #: Seconds to stall inside an upload. The only way to hold a write *in flight*
        #: long enough to assert on that window — which is where a read that consults
        #: only the queue answers with the previous snapshot, and where a flush that
        #: waits only for the queue returns before the bytes have landed.
        self.put_delay = 0.0

    def put(self, path, blob):
        with self.lock:
            # Milliseconds, and monotonically increasing per write, so an ordering
            # assertion cannot pass by accident on a fast machine where two writes share
            # a millisecond.
            now = int(time.time() * 1000)
            previous = self.files.get(path)
            if previous is not None and now <= previous[1]:
                now = previous[1] + 1
            self.files[path] = (bytes(blob), now)

    def get(self, path):
        with self.lock:
            entry = self.files.get(path)
            return entry[0] if entry else None

    def delete(self, path):
        with self.lock:
            return self.files.pop(path, None) is not None

    def listdir(self, path):
        """One level under ``path``, or ``None`` when nothing lives under it at all."""
        prefix = path.rstrip("/") + "/"
        with self.lock:
            paths = list(self.files)
            stats = dict(self.files)
        under = [p for p in paths if p.startswith(prefix)]
        if not under:
            return None
        contents = {}
        for full in under:
            rest = full[len(prefix) :]
            head, _, tail = rest.partition("/")
            child = prefix + head
            if tail:
                contents[child] = {
                    "path": child,
                    "name": head,
                    "is_directory": True,
                }
            else:
                blob, mtime = stats[full]
                contents[child] = {
                    "path": child,
                    "name": head,
                    "is_directory": False,
                    "file_size": len(blob),
                    "last_modified": mtime,
                }
        return list(contents.values())


class _Handler(BaseHTTPRequestHandler):
    volume: FakeVolume = None  # set by serve_fake_volume

    # -- plumbing ----------------------------------------------------------

    def log_message(self, *args):
        """Silent: pytest captures stderr and a request log per call is noise."""

    def _split(self):
        parsed = urllib.parse.urlparse(self.path)
        return urllib.parse.unquote(parsed.path), urllib.parse.parse_qs(parsed.query)

    def _respond(self, status, body=b"", content_type="application/octet-stream"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _authorized(self):
        """A bearer token must be present. Absent one, 401 — as the real API answers.

        Not cosmetic: ``_token()`` returning ``None`` is a real deployment state (an
        unset ``token_env``), and a fake that served the request anyway would let the
        backend look healthy in exactly the case an operator most needs it to complain.
        """
        header = self.headers.get("Authorization") or ""
        if header.startswith("Bearer ") and header[7:].strip():
            return True
        self._respond(401, b'{"error_code":"UNAUTHENTICATED"}')
        return False

    # -- the four calls ----------------------------------------------------

    def do_PUT(self):
        path, query = self._split()
        self.volume.calls.append(("PUT", path, query))
        if not self._authorized():
            return
        if not path.startswith(_FILES):
            return self._respond(404)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        target = path[len(_FILES) :]
        exists = self.volume.get(target) is not None
        overwrite = (query.get("overwrite") or ["false"])[0].lower() == "true"
        if exists and not overwrite:
            # The real API refuses this. Enforced here because put_text's contract is
            # *replace* and a Volume has no rename to fall back on, so an upload that
            # silently declined would make every save after the first a no-op that
            # reported success.
            return self._respond(409, b'{"error_code":"ALREADY_EXISTS"}')
        if self.volume.put_delay:
            time.sleep(self.volume.put_delay)
        self.volume.put(target, body)
        self._respond(204)

    def do_GET(self):
        path, query = self._split()
        self.volume.calls.append(("GET", path, query))
        if not self._authorized():
            return
        if path.startswith(_FILES):
            blob = self.volume.get(path[len(_FILES) :])
            if blob is None:
                return self._respond(404, b'{"error_code":"NOT_FOUND"}')
            return self._respond(200, blob)
        if path.startswith(_DIRECTORIES):
            contents = self.volume.listdir(path[len(_DIRECTORIES) :])
            if contents is None:
                return self._respond(404, b'{"error_code":"NOT_FOUND"}')
            payload = json.dumps({"contents": contents}).encode()
            return self._respond(200, payload, "application/json")
        self._respond(404)

    def do_DELETE(self):
        path, query = self._split()
        self.volume.calls.append(("DELETE", path, query))
        if not self._authorized():
            return
        if not path.startswith(_FILES):
            return self._respond(404)
        if not self.volume.delete(path[len(_FILES) :]):
            return self._respond(404, b'{"error_code":"NOT_FOUND"}')
        self._respond(204)


def serve_fake_volume():
    """Start the server on an ephemeral port. Returns ``(base_url, volume, shutdown)``."""
    volume = FakeVolume()
    handler = type("BoundHandler", (_Handler,), {"volume": volume})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def shutdown():
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    host, port = server.server_address[:2]
    return f"http://{host}:{port}", volume, shutdown
