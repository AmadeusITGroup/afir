"""Serves the OpenAPI description and a human-readable rendering of it.

``docs/openapi.yaml`` is hand-authored; ``/openapi.json`` hands it to a client
generator and ``/docs`` renders it for a person. Neither may fall back to
introspecting the router: generated descriptions cannot state what a route is for.

Rendering is server-side and asset-free because the App must work with no egress;
that rules out any CDN-hosted viewer.

The spec is re-read when its mtime changes, so a checkout edit needs no restart.
An absent file is answered 503 with a reason, not a traceback; ``docs/`` ships in
the bundle today but that is a property of the exclude list, not a guarantee.
"""

import html
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.utils.paths import docs_dir

logger = logging.getLogger(__name__)

SPEC_FILENAME = "openapi.yaml"

# (mtime, parsed spec); re-read on mtime change rather than cached for the process lifetime.
_CACHE: Optional[Tuple[float, Dict[str, Any]]] = None
_UNAVAILABLE: Optional[str] = None

_METHODS = ("get", "put", "post", "delete", "patch", "head", "options", "trace")


def spec_path():
    return docs_dir() / SPEC_FILENAME


def unavailable_reason() -> Optional[str]:
    """Why the last :func:`load_spec` returned ``None``, for the 503 body."""
    return _UNAVAILABLE


def load_spec() -> Optional[Dict[str, Any]]:
    global _CACHE, _UNAVAILABLE
    path = spec_path()
    try:
        mtime = path.stat().st_mtime
    except OSError as exc:
        _UNAVAILABLE = f"{path.name} is not present ({exc.strerror})"
        return None
    if _CACHE is not None and _CACHE[0] == mtime:
        return _CACHE[1]
    try:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        _UNAVAILABLE = f"{path.name} could not be read: {exc}"
        logger.warning("OpenAPI specification unreadable: %s", exc)
        return None
    if not isinstance(spec, dict) or "paths" not in spec:
        _UNAVAILABLE = f"{path.name} is not an OpenAPI document"
        return None
    _CACHE = (mtime, spec)
    _UNAVAILABLE = None
    return spec


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def _type_of(schema: Optional[Dict[str, Any]]) -> str:
    """A one-line type label. Follows ``$ref`` by name only; the referenced schema
    is rendered once in its own section rather than inlined at every use."""
    if not isinstance(schema, dict):
        return ""
    if "$ref" in schema:
        return _ref_name(str(schema["$ref"]))
    for key in ("oneOf", "anyOf", "allOf"):
        if key in schema and isinstance(schema[key], list):
            return " | ".join(_type_of(s) for s in schema[key] if isinstance(s, dict))
    kind = schema.get("type")
    if kind == "array":
        inner = _type_of(schema.get("items")) or "any"
        return f"array of {inner}"
    if isinstance(kind, list):
        return " | ".join(str(k) for k in kind)
    if schema.get("enum"):
        return " | ".join(json.dumps(v) for v in schema["enum"])
    return str(kind or "object")


def _operations(spec: Dict[str, Any]) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """Flatten paths into (tag, method, path, operation), tag-major.

    An untagged operation lands under "Other" rather than being dropped, so a spec
    that forgets a tag still renders every route it declares.
    """
    out = []
    for path, item in (spec.get("paths") or {}).items():
        if not isinstance(item, dict):
            continue
        shared = item.get("parameters") or []
        for method in _METHODS:
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            merged = dict(op)
            if shared:
                merged["parameters"] = list(shared) + list(op.get("parameters") or [])
            tags = op.get("tags") or ["Other"]
            out.append((str(tags[0]), method.upper(), str(path), merged))
    order = [str(t.get("name")) for t in (spec.get("tags") or []) if isinstance(t, dict)]

    def sort_key(row):
        tag = row[0]
        return (order.index(tag) if tag in order else len(order), tag, row[2], row[1])

    return sorted(out, key=sort_key)


_CSS = """
:root{--bg:#ffffff;--fg:#1b1f24;--muted:#5b6570;--line:#d8dee4;--panel:#f6f8fa;
--get:#1f6feb;--post:#1a7f37;--delete:#cf222e;--other:#6e40c9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,
BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:60rem;margin:0 auto;padding:2rem 1.25rem 5rem}
h1{font-size:1.65rem;margin:0 0 .35rem}
h2{font-size:1.1rem;margin:2.5rem 0 .75rem;padding-bottom:.35rem;
border-bottom:1px solid var(--line)}
h3{font-size:.95rem;margin:0}
p{margin:.4rem 0}
code,pre,.path{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
font-size:.86em}
.lede{color:var(--muted);margin-bottom:1.25rem}
.meta{color:var(--muted);font-size:.86rem}
.op{border:1px solid var(--line);border-radius:6px;margin:.6rem 0;overflow:hidden}
.op>summary{cursor:pointer;padding:.6rem .8rem;background:var(--panel);
display:flex;gap:.6rem;align-items:baseline;flex-wrap:wrap}
.op>summary::-webkit-details-marker{display:none}
.verb{font-weight:600;font-size:.74rem;letter-spacing:.04em;color:#fff;
padding:.15rem .45rem;border-radius:3px;background:var(--other)}
.verb.GET{background:var(--get)}.verb.POST{background:var(--post)}
.verb.DELETE{background:var(--delete)}
.path{font-weight:600}
.sum{color:var(--muted);flex:1 1 100%}
.body{padding:.8rem}
.dep{color:var(--delete);font-weight:600;font-size:.78rem}
table{border-collapse:collapse;width:100%;margin:.5rem 0 1rem}
th,td{border:1px solid var(--line);padding:.35rem .5rem;text-align:left;
vertical-align:top}
th{background:var(--panel);font-size:.8rem;font-weight:600}
h4{font-size:.82rem;text-transform:uppercase;letter-spacing:.05em;
color:var(--muted);margin:1rem 0 .25rem}
.nav{background:var(--panel);border:1px solid var(--line);border-radius:6px;
padding:.75rem 1rem;margin-bottom:1rem}
.nav a{color:var(--get);text-decoration:none;margin-right:1rem;
white-space:nowrap;line-height:1.9}
.nav a:hover{text-decoration:underline}
.err{border-left:3px solid var(--delete);padding-left:1rem}
@media (prefers-color-scheme:dark){
:root{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#30363d;--panel:#161b22;
--get:#4493f8;--post:#3fb950;--delete:#f85149;--other:#a371f7}
.verb{color:#0d1117}
}
"""


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head>"
        f"<body><main>{body}</main></body></html>\n"
    )


def render_unavailable(reason: Optional[str]) -> str:
    detail = _esc(reason or "the specification could not be loaded")
    return _page(
        "API reference unavailable",
        "<h1>API reference unavailable</h1><div class=\"err\"><p>"
        f"{detail}.</p><p class=\"meta\">The description is read from "
        f"<code>docs/{SPEC_FILENAME}</code> at request time. The API itself is "
        "unaffected; only this reference page and <code>/openapi.json</code> "
        "depend on that file.</p></div>",
    )


def _params_table(params: List[Dict[str, Any]]) -> str:
    rows = []
    for p in params:
        if not isinstance(p, dict):
            continue
        req = "yes" if p.get("required") else "no"
        rows.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td></tr>".format(
                _esc(p.get("name")),
                _esc(p.get("in")),
                _esc(_type_of(p.get("schema"))),
                req,
                _esc(p.get("description")),
            )
        )
    if not rows:
        return ""
    return (
        "<h4>Parameters</h4><table><tr><th>Name</th><th>In</th><th>Type</th>"
        "<th>Required</th><th>Description</th></tr>" + "".join(rows) + "</table>"
    )


def _content_types(container: Dict[str, Any]) -> str:
    content = container.get("content")
    if not isinstance(content, dict):
        return ""
    parts = []
    for media, entry in content.items():
        schema = entry.get("schema") if isinstance(entry, dict) else None
        label = _type_of(schema)
        parts.append(f"<code>{_esc(media)}</code>" + (f" &rarr; {_esc(label)}" if label else ""))
    return ", ".join(parts)


def _request_body(op: Dict[str, Any]) -> str:
    body = op.get("requestBody")
    if not isinstance(body, dict):
        return ""
    req = " (required)" if body.get("required") else ""
    desc = _esc(body.get("description") or "")
    types = _content_types(body)
    return (
        f"<h4>Request body{req}</h4><p>{types}</p>"
        + (f"<p class=\"meta\">{desc}</p>" if desc else "")
    )


def _responses(op: Dict[str, Any]) -> str:
    responses = op.get("responses")
    if not isinstance(responses, dict):
        return ""
    rows = []
    for code, entry in sorted(responses.items(), key=lambda kv: str(kv[0])):
        if not isinstance(entry, dict):
            continue
        rows.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                _esc(code), _esc(entry.get("description")), _content_types(entry)
            )
        )
    if not rows:
        return ""
    return (
        "<h4>Responses</h4><table><tr><th>Status</th><th>Meaning</th>"
        "<th>Body</th></tr>" + "".join(rows) + "</table>"
    )


def _schemas(spec: Dict[str, Any]) -> str:
    schemas = ((spec.get("components") or {}).get("schemas")) or {}
    if not isinstance(schemas, dict) or not schemas:
        return ""
    out = ["<h2 id=\"schemas\">Schemas</h2>"]
    for name, schema in sorted(schemas.items()):
        if not isinstance(schema, dict):
            continue
        required = set(schema.get("required") or [])
        props = schema.get("properties")
        rows = []
        if isinstance(props, dict):
            for field, sub in props.items():
                sub = sub if isinstance(sub, dict) else {}
                rows.append(
                    "<tr><td><code>{}</code></td><td>{}</td><td>{}</td>"
                    "<td>{}</td></tr>".format(
                        _esc(field),
                        _esc(_type_of(sub)),
                        "yes" if field in required else "no",
                        _esc(sub.get("description")),
                    )
                )
        table = (
            "<table><tr><th>Field</th><th>Type</th><th>Required</th>"
            "<th>Description</th></tr>" + "".join(rows) + "</table>"
            if rows
            else "<p class=\"meta\">No declared properties.</p>"
        )
        desc = _esc(schema.get("description") or "")
        out.append(
            f"<details class=\"op\"><summary><h3>{_esc(name)}</h3></summary>"
            f"<div class=\"body\">"
            + (f"<p>{desc}</p>" if desc else "")
            + table
            + "</div></details>"
        )
    return "".join(out)


def render_html(spec: Dict[str, Any]) -> str:
    info = spec.get("info") or {}
    title = str(info.get("title") or "API reference")
    version = _esc(info.get("version") or "")
    head = [f"<h1>{_esc(title)}</h1>"]
    lede = []
    if version:
        lede.append(f"Version {version}")
    if spec.get("openapi"):
        lede.append(f"OpenAPI {_esc(spec['openapi'])}")
    lede.append(
        "machine-readable description at <a href=\"/openapi.json\">/openapi.json</a>"
    )
    head.append(f"<p class=\"lede\">{' &middot; '.join(lede)}</p>")
    if info.get("description"):
        head.append(f"<p>{_esc(info['description'])}</p>")

    servers = spec.get("servers") or []
    if servers:
        items = [
            "<code>{}</code>{}".format(
                _esc(s.get("url")),
                f" &mdash; {_esc(s.get('description'))}" if s.get("description") else "",
            )
            for s in servers
            if isinstance(s, dict)
        ]
        head.append("<h4>Servers</h4><p>" + "<br>".join(items) + "</p>")

    ops = _operations(spec)
    tags = []
    for tag, _m, _p, _o in ops:
        if tag not in tags:
            tags.append(tag)
    nav = "".join(f"<a href=\"#{_esc(t)}\">{_esc(t)}</a>" for t in tags)
    if (spec.get("components") or {}).get("schemas"):
        nav += "<a href=\"#schemas\">Schemas</a>"
    head.append(f"<nav class=\"nav\">{nav}</nav>")

    descriptions = {
        str(t.get("name")): t.get("description")
        for t in (spec.get("tags") or [])
        if isinstance(t, dict)
    }

    body = []
    current = None
    for tag, method, path, op in ops:
        if tag != current:
            current = tag
            body.append(f"<h2 id=\"{_esc(tag)}\">{_esc(tag)}</h2>")
            if descriptions.get(tag):
                body.append(f"<p class=\"meta\">{_esc(descriptions[tag])}</p>")
        dep = "<span class=\"dep\">deprecated</span>" if op.get("deprecated") else ""
        summary = _esc(op.get("summary") or "")
        body.append(
            f"<details class=\"op\"><summary>"
            f"<span class=\"verb {method}\">{method}</span>"
            f"<span class=\"path\">{_esc(path)}</span>{dep}"
            + (f"<span class=\"sum\">{summary}</span>" if summary else "")
            + "</summary><div class=\"body\">"
        )
        if op.get("description"):
            body.append(f"<p>{_esc(op['description'])}</p>")
        if op.get("operationId"):
            body.append(
                f"<p class=\"meta\">Handler: <code>{_esc(op['operationId'])}</code></p>"
            )
        body.append(_params_table(op.get("parameters") or []))
        body.append(_request_body(op))
        body.append(_responses(op))
        body.append("</div></details>")

    body.append(_schemas(spec))
    return _page(title, "".join(head) + "".join(body))
