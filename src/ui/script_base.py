"""
Base-path shim — the first thing in the page's one script, and a no-op everywhere but
one deployment shape.

Every URL the page asks for is root-relative (``fetch("/api/v1/jobs")``), which is right
for ``python app.py`` and right for a Databricks App: both serve the server at ``/``.
A cluster driver-proxy ingress does not — the page is served under
``/driver-proxy/o/<org>/<cluster>/<port>/`` and the proxy strips that prefix before
forwarding, so the server needs no change and the *browser* does: a root-relative URL
leaves the prefix behind and lands on the workspace, not on AFIR. There is no
``X-Forwarded-Prefix`` header to read, so the prefix is derived from
``location.pathname``.

Wrapping the three natives rather than editing ~57 call sites is the whole point: a
second spelling of "how a URL is built" is a second answer to it, and the call sites keep
their literal paths, which is what ``tests/test_webui.py`` matches against the router.

``afirBasePath`` returns ``""`` for every other shape, and nothing is installed when it
does — local and Apps behaviour stay byte-identical.
"""

SCRIPT_BASE_JS = r"""
/* The prefix, or "" — matched by SHAPE and not by a substring: `/driver-proxy(-api)?/o/`
   then an org, a cluster and a numeric port. A page served at `/` has too few segments
   and returns early. */
function afirBasePath(){
  const parts = location.pathname.split("/");
  if (parts.length < 6) return "";
  if (parts[1] !== "driver-proxy" && parts[1] !== "driver-proxy-api") return "";
  if (parts[2] !== "o") return "";
  if (!parts[3] || !parts[4]) return "";
  if (!/^\d+$/.test(parts[5])) return "";
  return "/" + parts.slice(1, 6).join("/");
}

const AFIR_BASE = afirBasePath();

/* Root-relative only. A blob:, data: or absolute URL is left exactly as given, and an
   already-prefixed one is not prefixed twice — a re-entrant wrap is how a base path
   turns into a 404 that looks like a missing route. */
function afirUrl(u){
  if (!AFIR_BASE) return u;
  if (typeof u !== "string") return u;
  if (u.charAt(0) !== "/") return u;
  if (u.indexOf(AFIR_BASE + "/") === 0) return u;
  return AFIR_BASE + u;
}

const afirNativeFetch = window.fetch.bind(window);
const AfirNativeEventSource = window.EventSource;
const afirNativeOpen = window.open.bind(window);

function afirFetch(u, opts){ return afirNativeFetch(afirUrl(u), opts); }

/* A constructor that returns an object yields that object, so callers get a REAL
   EventSource — `close()`, `addEventListener` and `onmessage` are the native ones. */
function AfirEventSource(u, cfg){ return new AfirNativeEventSource(afirUrl(u), cfg); }

function afirOpen(u, target, features){ return afirNativeOpen(afirUrl(u), target, features); }

if (AFIR_BASE) {
  window.fetch = afirFetch;
  window.EventSource = AfirEventSource;
  window.open = afirOpen;
}
"""

__all__ = ["SCRIPT_BASE_JS"]
