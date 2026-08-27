"""
Serve the finished investigation artifacts: the report (Markdown / PDF / JSON / HTML)
and the evidence (raw + transformed).

Every read goes through a ``StorageBackend``, defaulting to local disk under
``exports_dir()``. The source of truth is ``fraud_report_<id>.md`` and
``fraud_report_<id>.pdf``, written unconditionally by ``_write_readable_reports``
regardless of ``output_format``, so every UI format derives from an artifact that
exists whichever way the app is configured and for any past job.

HTML is rendered server-side by :func:`markdown_to_html`; the app must work with no
egress. :func:`_safe_id` rejects rather than sanitises: a path traversal resolving to
a real readable object would answer 200 with content it was never meant to serve.
"""

import html
import json
import logging
import os
import re
from typing import Optional, Tuple

from src.storage import LocalStorage, StorageBackend
from src.utils.paths import exports_dir

logger = logging.getLogger(__name__)

#: The backend artifacts are read from. ``None`` means local disk under
#: ``exports_dir()``, resolved per call, which keeps the ``AFIR_DATA_DIR`` override and
#: the ``monkeypatch.setattr(report_delivery, "exports_dir", ...)`` seam tests use.
_STORAGE: Optional[StorageBackend] = None


def set_storage(storage: Optional[StorageBackend]) -> None:
    """Point artifact reads at ``storage``; ``None`` restores local disk.

    Module-level rather than per-function because these are plain functions the HTTP
    layer calls directly; one module-level default is simpler than eight extra params.
    """
    global _STORAGE
    _STORAGE = storage


def _backend() -> StorageBackend:
    return _STORAGE if _STORAGE is not None else LocalStorage(root=exports_dir())


#: What ``?format=`` accepts on the report endpoint. ``view`` is the UI's format: the
#: rendered body as an HTML fragment plus a table of contents, as JSON. The page needs
#: both; heading nav computed server-side avoids a second, divergent parser in the client.
REPORT_FORMATS = ("view", "html", "md", "pdf", "json")

#: What ``?kind=`` accepts on the evidence endpoint.
EVIDENCE_KINDS = ("raw", "transformed")

_CONTENT_TYPES = {
    "view": "application/json",
    "html": "text/html",
    "md": "text/markdown",
    "pdf": "application/pdf",
    "json": "application/json",
}


def _safe_id(incident_id: str) -> str:
    """Reject anything that could escape the exports directory.

    Incident ids reach here straight off a URL path. They are normally a UUID or a
    case reference, but ``../../etc/passwd`` is also a string, and these functions
    build filenames by interpolation. Whitelisting the characters an id can actually
    contain is the check that means the interpolation is safe everywhere below.
    """
    text = str(incident_id or "").strip()
    if not text or not re.fullmatch(r"[A-Za-z0-9._\-]{1,128}", text) or ".." in text:
        raise ValueError(f"invalid incident id {incident_id!r}")
    return text


def report_key(incident_id: str, suffix: str) -> str:
    return f"fraud_report_{_safe_id(incident_id)}.{suffix}"


def evidence_key(incident_id: str, kind: str) -> str:
    if kind not in EVIDENCE_KINDS:
        raise ValueError(f"unknown evidence kind '{kind}'")
    return f"evidence_{kind}_{_safe_id(incident_id)}.json"


def report_path(incident_id: str, suffix: str) -> str:
    """The local path a report artifact occupies.

    Used by ``report_generation`` (the PDF renderer takes a filename) and shown on the
    console. Reads go through :func:`_backend` instead.
    """
    return os.path.join(str(exports_dir()), report_key(incident_id, suffix))


def evidence_path(incident_id: str, kind: str) -> str:
    return os.path.join(str(exports_dir()), evidence_key(incident_id, kind))


def artifact_inventory(incident_id: str) -> dict:
    """What actually exists on disk for this incident, with sizes.

    The UI checks this before drawing download buttons: a greyed-out button with a
    byte count is more accurate than a link that fails, because ``_write_readable_reports``
    is best-effort by design.
    """
    ident = _safe_id(incident_id)
    items = {
        "report_md": report_key(ident, "md"),
        "report_pdf": report_key(ident, "pdf"),
        "evidence_raw": evidence_key(ident, "raw"),
        "evidence_transformed": evidence_key(ident, "transformed"),
        "export_json": f"incident_{ident}.json",
        "export_csv": f"incident_{ident}.csv",
    }
    backend = _backend()
    # One listing, not six existence probes: on a remote backend each probe is a round trip.
    sizes = {obj.name: obj.size for obj in backend.list_keys("")}
    out = {}
    for name, key in items.items():
        exists = key in sizes
        out[name] = {
            "exists": exists,
            "bytes": sizes.get(key, 0) if exists else 0,
            "filename": key,
        }
    return {"incident_id": ident, "artifacts": out}


def list_incidents(limit: int = 200) -> list:
    """Incidents with an artifact on disk, newest first.

    Every id recovered from a filename goes back through :func:`_safe_id`; anything that
    fails is dropped, because a path-traversal filename is otherwise handed to a caller
    that interpolates it. Per-artifact sizes are :func:`artifact_inventory`'s job; doing
    them here would mean six stat calls per row for a list whose only purpose is to be
    clicked.
    """
    patterns = (
        (re.compile(r"^fraud_report_(.+)\.md$"), "report"),
        (re.compile(r"^fraud_report_(.+)\.pdf$"), "pdf"),
        (re.compile(r"^incident_(.+)\.json$"), "export"),
    )
    found: dict = {}
    listing = _backend().list_keys("")
    for obj in listing:
        name = obj.name
        for pattern, kind in patterns:
            match = pattern.match(name)
            if not match:
                continue
            try:
                ident = _safe_id(match.group(1))
            except ValueError:
                logger.warning("ignoring export with an unusable incident id: %s", name)
                break
            entry = found.setdefault(
                ident,
                {
                    "incident_id": ident,
                    "mtime": 0.0,
                    "has_report": False,
                    "has_pdf": False,
                },
            )
            if kind == "report":
                entry["has_report"] = True
            elif kind == "pdf":
                entry["has_pdf"] = True
            entry["mtime"] = max(entry["mtime"], obj.mtime)
            break
    rows = sorted(found.values(), key=lambda r: r["mtime"], reverse=True)
    return rows[: max(0, int(limit))]


def read_markdown(incident_id: str) -> Optional[str]:
    return _backend().get_text(report_key(incident_id, "md"))


def read_pdf(incident_id: str) -> Optional[bytes]:
    return _backend().get_bytes(report_key(incident_id, "pdf"))


def sections_from_report(report) -> Optional[list]:
    """Recover the section list from a job's ``report`` output, if it is recoverable.

    ``txt`` mode hands back a JSON string of sections; import/export hands back a dict.
    PDF bytes are terminal; ``None`` then, and the caller falls back to the Markdown
    file, which is why that file being unconditional matters.
    """
    if report is None:
        return None
    if isinstance(report, (bytes, bytearray)):
        return None
    if isinstance(report, list):
        return report
    if isinstance(report, dict):
        for key in ("sections", "report_sections"):
            if isinstance(report.get(key), list):
                return report[key]
        return None
    if isinstance(report, str):
        try:
            parsed = json.loads(report)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, list) else sections_from_report(parsed)
    if hasattr(report, "model_dump"):
        return sections_from_report(report.model_dump())
    return None


def resolve_report(incident_id: str, fmt: str, report=None) -> Tuple[bytes, str, str]:
    """Return ``(body, content_type, filename)`` for one report format.

    Raises :class:`FileNotFoundError` when the artifact does not exist, so the endpoint
    answers 404 rather than serving an empty file that looks like a corrupt download.
    """
    if fmt not in REPORT_FORMATS:
        raise ValueError(
            f"unknown format '{fmt}'; expected one of " + ", ".join(REPORT_FORMATS)
        )
    ident = _safe_id(incident_id)
    if fmt == "pdf":
        data = read_pdf(ident)
        if data is None:
            raise FileNotFoundError("no PDF report for this incident")
        return data, _CONTENT_TYPES["pdf"], f"fraud_report_{ident}.pdf"
    if fmt == "json":
        sections = sections_from_report(report)
        if sections is None:
            # Fall back to the Markdown so `json` is never simply unavailable: a
            # single-section document carrying the rendered text is a worse answer than
            # real sections, but a much better one than a 404 for a report that exists.
            md = read_markdown(ident)
            if md is None:
                raise FileNotFoundError("no report for this incident")
            sections = [{"section_title": "Report", "content": md}]
        body = json.dumps(
            {"incident_id": ident, "sections": sections}, indent=2, default=str
        ).encode("utf-8")
        return body, _CONTENT_TYPES["json"], f"fraud_report_{ident}.json"
    md = read_markdown(ident)
    if md is None:
        raise FileNotFoundError("no Markdown report for this incident")
    if fmt == "md":
        return md.encode("utf-8"), _CONTENT_TYPES["md"], f"fraud_report_{ident}.md"
    if fmt == "view":
        body, toc = markdown_to_html(md)
        payload = json.dumps(
            {"incident_id": ident, "html": body, "toc": toc, "chars": len(md)}
        ).encode("utf-8")
        return payload, _CONTENT_TYPES["view"], f"fraud_report_{ident}.json"
    return (
        html_document(md, ident).encode("utf-8"),
        _CONTENT_TYPES["html"],
        f"fraud_report_{ident}.html",
    )


def resolve_evidence(incident_id: str, kind: str) -> Tuple[bytes, str, str]:
    """Return ``(body, content_type, filename)`` for one evidence artifact."""
    ident = _safe_id(incident_id)
    data = _backend().get_bytes(evidence_key(ident, kind))
    if data is None:
        raise FileNotFoundError(f"no {kind} evidence for this incident")
    return data, "application/json", f"evidence_{kind}_{ident}.json"


def evidence_outline(incident_id: str, kind: str, max_rows: int = 25) -> dict:
    """A bounded, browsable preview of an evidence artifact for the UI.

    A full evidence download can be tens of megabytes. The outline carries per-source
    counts plus the first ``max_rows`` rows; the download button beside it serves the
    full artifact. Nothing is hidden: the preview is bounded, not the artifact.
    """
    ident = _safe_id(incident_id)
    blob = _backend().get_text(evidence_key(ident, kind))
    if blob is None:
        raise FileNotFoundError(f"no {kind} evidence for this incident")
    size = len(blob.encode("utf-8"))
    doc = json.loads(blob)

    out = {"incident_id": ident, "kind": kind, "bytes": size, "groups": []}
    if kind == "raw" and isinstance(doc, dict):
        for source, rows in doc.items():
            rows = rows if isinstance(rows, list) else []
            out["groups"].append(
                {
                    "name": source,
                    "count": len(rows),
                    "truncated": len(rows) > max_rows,
                    "rows": rows[:max_rows],
                }
            )
        out["groups"].sort(key=lambda g: g["count"], reverse=True)
        out["total_rows"] = sum(g["count"] for g in out["groups"])
        return out

    # Transformed: a single object of named blocks. Surface each by name so the UI can
    # render one collapsible card per block without knowing the schema.
    if isinstance(doc, dict):
        for name, value in doc.items():
            if isinstance(value, list):
                out["groups"].append(
                    {
                        "name": name,
                        "count": len(value),
                        "truncated": len(value) > max_rows,
                        "rows": value[:max_rows],
                    }
                )
            else:
                out["groups"].append(
                    {"name": name, "count": 1, "truncated": False, "value": value}
                )
    out["total_rows"] = sum(g.get("count", 0) for g in out["groups"])
    return out


# Markdown -> HTML

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\*\w])\*(?!\s)([^\*]+?)(?<!\s)\*(?!\*)")
_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


def _inline(text: str) -> str:
    """Escape, then apply inline Markdown. Order is the security invariant.

    Escaping first means report content cannot inject markup: ``<script>`` becomes text.
    The inline patterns then insert only tags this function itself produces. Reversing
    the order would escape those tags visibly; skipping escape entirely would make every
    LLM-authored string an injection vector into the operator's browser.
    """
    out = html.escape(text, quote=False)
    out = _CODE.sub(lambda m: f"<code>{m.group(1)}</code>", out)
    out = _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)
    out = _ITALIC.sub(lambda m: f"<em>{m.group(1)}</em>", out)
    out = _LINK.sub(
        lambda m: f'<a href="{m.group(2)}" target="_blank" rel="noopener">'
        f"{m.group(1)}</a>",
        out,
    )
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def markdown_to_html(md: str) -> Tuple[str, list]:
    """Render the report Markdown to HTML. Returns ``(html_body, toc)``.

    Not a general Markdown implementation: it handles exactly what
    ``ReportGenerationModule.render_markdown`` emits: atx headings, ``-`` bullets,
    ordered lists, fenced code, pipe tables, blockquotes, horizontal rules and
    paragraphs. Both sides live in the same repo, so they are kept consistent by a
    test rather than a version pin.

    Ordered lists are preserved because LLM prose can arrive as numbered steps;
    collapsing that into one paragraph loses the enumeration.

    ``toc`` is ``[{level, title, id}]`` for the heading nav.
    """
    lines = (md or "").replace("\r\n", "\n").split("\n")
    out: list = []
    toc: list = []
    index = 0
    list_tag = ""  # currently open list: "ul", "ol", or ""

    def close_list():
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = ""

    def open_list(tag: str):
        """Open ``tag``, closing a different list first so ul/ol never interleave."""
        nonlocal list_tag
        if list_tag != tag:
            close_list()
            out.append(f"<{tag}>")
            list_tag = tag

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            close_list()
            index += 1
            continue

        # Fenced code.
        if stripped.startswith("```"):
            close_list()
            index += 1
            block = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(html.escape(lines[index], quote=False))
                index += 1
            index += 1  # closing fence
            out.append("<pre><code>" + "\n".join(block) + "</code></pre>")
            continue

        # Headings.
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            close_list()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            anchor = f"sec-{len(toc)}-{_slug(title)}"
            toc.append({"level": level, "title": title, "id": anchor})
            out.append(f'<h{level} id="{anchor}">{_inline(title)}</h{level}>')
            index += 1
            continue

        # Horizontal rule.
        if re.fullmatch(r"(\*\s*){3,}|(-\s*){3,}|(_\s*){3,}", stripped):
            close_list()
            out.append("<hr>")
            index += 1
            continue

        # Pipe table; needs the separator row on line 2 to avoid treating a prose
        # line containing a pipe as a one-column table.
        if (
            stripped.startswith("|")
            and index + 1 < len(lines)
            and re.fullmatch(r"\|[\s:|-]+\|?", lines[index + 1].strip())
        ):
            close_list()
            header = _table_cells(stripped)
            index += 2
            body = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                body.append(_table_cells(lines[index].strip()))
                index += 1
            head_html = "".join(f"<th>{_inline(c)}</th>" for c in header)
            rows_html = "".join(
                "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>"
                for row in body
            )
            out.append(
                f"<table><thead><tr>{head_html}</tr></thead>"
                f"<tbody>{rows_html}</tbody></table>"
            )
            continue

        # Bullets.
        bullet = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if bullet:
            open_list("ul")
            out.append(f"<li>{_inline(bullet.group(1).strip())}</li>")
            index += 1
            continue

        # Ordered list. `1.` / `1)`: the numbers themselves are dropped, since the
        # browser renders them and keeping both would print "1. 1.".
        ordered = re.match(r"^\s*\d{1,3}[.)]\s+(.*)$", line)
        if ordered:
            open_list("ol")
            out.append(f"<li>{_inline(ordered.group(1).strip())}</li>")
            index += 1
            continue

        # Blockquote.
        if stripped.startswith(">"):
            close_list()
            out.append(
                f"<blockquote>{_inline(stripped.lstrip('> ').strip())}" "</blockquote>"
            )
            index += 1
            continue

        # Paragraph: consume until a blank line or the start of another block.
        close_list()
        para = [stripped]
        index += 1
        while index < len(lines):
            nxt = lines[index].strip()
            if not nxt or re.match(r"^(#{1,6}\s|[-*+]\s|\d{1,3}[.)]\s|>|\||```)", nxt):
                break
            para.append(nxt)
            index += 1
        out.append(f"<p>{_inline(' '.join(para))}</p>")

    close_list()
    return "\n".join(out), toc


def _table_cells(row: str) -> list:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


#: Standalone-download styling. Kept separate from the console UI's stylesheet: this
#: HTML is also what an analyst saves and mails to someone who has no access to AFIR,
#: so it has to be legible with no external CSS and print sanely.
_STANDALONE_CSS = """
:root { color-scheme: light; }
body { font: 15px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 46rem; margin: 3rem auto; padding: 0 1.5rem; color: #1a1f2b; }
h1 { font-size: 1.75rem; border-bottom: 2px solid #e2e6ef; padding-bottom: .5rem; }
h2 { font-size: 1.3rem; margin-top: 2.2rem; color: #0b3d6b; }
h3 { font-size: 1.08rem; margin-top: 1.5rem; color: #33415c; }
code { background: #f2f4f8; padding: .1rem .3rem; border-radius: 3px; font-size: .9em; }
pre { background: #f2f4f8; padding: 1rem; overflow-x: auto; border-radius: 6px; }
blockquote { border-left: 3px solid #c8d2e3; margin: 1rem 0; padding: .2rem 1rem;
             color: #4a5468; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .92em; }
th, td { border: 1px solid #dde3ee; padding: .45rem .6rem; text-align: left; }
th { background: #f2f4f8; }
ul, ol { padding-left: 1.4rem; }
hr { border: none; border-top: 1px solid #e2e6ef; margin: 2rem 0; }
@media print { body { margin: 0; max-width: none; } h2 { page-break-after: avoid; } }
"""


def html_document(md: str, incident_id: str = "") -> str:
    """A complete, self-contained HTML file for the rendered report."""
    body, _ = markdown_to_html(md)
    title = f"Fraud Investigation Report {incident_id}".strip()
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        f"<style>{_STANDALONE_CSS}</style></head><body>\n{body}\n</body></html>\n"
    )
