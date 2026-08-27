"""Decode pack-declared encoded fields into real columns, right after retrieval.

Three invariants govern how ``SourceDef.encoded_fields`` declarations are applied:

* Decodes are written as nested children, not flat dotted keys. A flat key resolves to
  no scalar leaves, so ``resolve_path`` returns nothing while the value looks present.
* A separator is a candidate list; the first that occurs wins. A wrong separator yields
  one malformed row for the whole table.
* An absent part stays absent, not ``""``. A missing key differs from an empty string.

Idempotent and best-effort per field (on error: logs and leaves the encoded value).
"""

import base64
import binascii
import csv
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Record separators tried when the pack declares none. The escaped form leads: a payload
# with both ``\n`` and a real newline is ambiguous, and preferring the escaped form
# avoids reading a whole table as one malformed row.
_DEFAULT_RECORD_SEPARATORS = ("\\n", "\n", "\r\n")
_DEFAULT_MAX_RECORDS = 10000
# Appended to a field's own name when the pack declares no `into`. A suffix rather than a
# sibling key, so the decode sits next to its source value under the same parent and
# `<parent>.<field>_decoded.<column>` is a readable path for a condition.
_DEFAULT_INTO_SUFFIX = "_decoded"


def decode_logs(
    logs: Dict[str, List[Dict[str, Any]]],
    knowledge_pack: Any,
) -> Dict[str, int]:
    """Decode every declared encoded field in place across ``logs``.

    Returns ``{source: records_decoded}`` for the sources that produced any, which is the
    caller's only way to tell that the table is now readable, or by its absence that a declared
    field was not in the rows. ``logs`` is mutated because it is the object every downstream
    consumer already shares: the evidence pack, the exports and the job store read the same
    dict, so a decode written here reaches all of them with no plumbing.

    No pack, or a pack declaring nothing, is a no-op returning ``{}``.
    """
    if not logs or knowledge_pack is None:
        return {}
    out: Dict[str, int] = {}
    for source, rows in logs.items():
        specs = _specs_for(knowledge_pack, source)
        if not specs:
            continue
        count = 0
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            for spec in specs:
                try:
                    count += _decode_one(row, spec, source)
                except Exception as exc:  # noqa: BLE001 — one bad payload, not the run
                    logger.warning(
                        "Could not decode encoded field '%s' on source '%s': %s",
                        spec.get("field", "?"),
                        source,
                        exc,
                    )
        if count:
            out[source] = count
            logger.info(
                "Decoded %d record(s) from %d encoded field declaration(s) on '%s'.",
                count,
                len(specs),
                source,
            )
        else:
            logger.info(
                "Source '%s' declares %d encoded field(s); none of the retrieved rows "
                "carried a decodable value.",
                source,
                len(specs),
            )
    return out


def decoded_tables(knowledge_pack: Any) -> Dict[str, List[str]]:
    """``{source: [dotted path of each decoded record table]}`` the pack declares.

    The paths a decode will have written, derived from the same declarations
    :func:`decode_logs` reads, so the shape of the declaration is known in one module only.
    Consumers use it to treat those tables as a table of records nested in a row rather than as
    more leaves of the row: the evidence chronology expands them into one event per record, and
    the trimmer keeps their columns instead of dropping the ones that happen to be constant.

    ``{}`` for a pack that declares nothing.
    """
    out: Dict[str, List[str]] = {}
    sources = getattr(knowledge_pack, "sources", None) or []
    for src in sources:
        name = str(getattr(src, "name", "") or "")
        if not name:
            continue
        paths = [
            str(s.get("into") or (str(s.get("field", "")) + _DEFAULT_INTO_SUFFIX))
            for s in _specs_for(knowledge_pack, name)
        ]
        paths = [p for p in paths if p.strip()]
        if paths:
            out[name] = paths
    return out


def records_at(row: Any, path: str) -> List[Dict[str, Any]]:
    """The decoded record list at ``path``, or ``[]``.

    A separate reader from `correlation.resolve_path` for the reason `_read_path` exists: that
    one returns scalar leaves and a list of dicts has none, so a caller asking it for the table
    gets an empty answer while the table is plainly there.
    """
    node = _read_path(row, path)
    if not isinstance(node, list):
        return []
    return [r for r in node if isinstance(r, dict)]


def _specs_for(knowledge_pack: Any, source: str) -> List[Dict[str, Any]]:
    """The source's `encoded_fields` declarations, or ``[]`` for anything unexpected."""
    try:
        src = knowledge_pack.source(source)
    except Exception:  # noqa: BLE001 — a malformed pack must not break retrieval
        return []
    if src is None:
        return []
    specs = getattr(src, "encoded_fields", None) or []
    return [s for s in specs if isinstance(s, dict) and str(s.get("field", "")).strip()]


def _decode_one(row: Dict[str, Any], spec: Dict[str, Any], source: str) -> int:
    """Decode one declared field on one row. Returns the number of records written."""
    field = str(spec.get("field", "")).strip()
    into = str(spec.get("into", "") or "").strip() or (field + _DEFAULT_INTO_SUFFIX)
    if _read_path(row, into) is not None:
        return 0  # already decoded (idempotent: a re-run must not double the table)
    raw = _read_path(row, field)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return 0
    text = _to_text(raw, spec)
    if text is None:
        return 0
    if str(spec.get("format", "delimited")).strip().lower() == "text":
        _write_into(row, into, text)
        return 1
    records = _parse_delimited(text, spec, source)
    if not records:
        return 0
    records = [_derive(r, spec) for r in records]
    _write_into(row, into, records)
    return len(records)


def _to_text(raw: Any, spec: Dict[str, Any]) -> Optional[str]:
    """The encoded value as text, or ``None`` when it cannot be decoded."""
    encoding = str(spec.get("encoding", "base64") or "").strip().lower()
    charset = str(spec.get("charset", "utf-8") or "utf-8")
    if encoding in ("", "none", "plain", "text"):
        return raw if isinstance(raw, str) else str(raw)
    if encoding != "base64":
        logger.warning("Unknown encoding '%s'; field left encoded.", encoding)
        return None
    payload = raw if isinstance(raw, (str, bytes)) else str(raw)
    try:
        # `validate=False` (the default) ignores whitespace a producer may have wrapped the
        # payload with; a genuinely non-base64 value still raises below.
        data = base64.b64decode(payload)
    except (binascii.Error, ValueError) as exc:
        logger.warning("Value is not decodable base64 (%s); left encoded.", exc)
        return None
    return data.decode(charset, errors="replace")


def _separators(spec: Dict[str, Any]) -> Tuple[str, ...]:
    declared = spec.get("record_separator")
    if declared is None:
        return _DEFAULT_RECORD_SEPARATORS
    if isinstance(declared, str):
        return (declared,) if declared else _DEFAULT_RECORD_SEPARATORS
    cands = tuple(str(s) for s in declared if str(s))
    return cands or _DEFAULT_RECORD_SEPARATORS


def _parse_delimited(
    text: str, spec: Dict[str, Any], source: str
) -> List[Dict[str, Any]]:
    """Split ``text`` into records, then into named columns."""
    # The first declared separator that actually occurs. One that does not occur cannot be
    # the right one, and picking it anyway yields exactly one record: a whole table read as
    # a single malformed row.
    sep = next((s for s in _separators(spec) if s in text), None)
    lines = [ln for ln in (text.split(sep) if sep else [text]) if ln.strip()]
    if not lines:
        return []
    field_sep = str(spec.get("field_separator", ",") or ",")
    header = bool(spec.get("header", True))
    columns = [str(c) for c in (spec.get("columns") or []) if str(c).strip()]
    if header:
        columns = _split_record(lines[0], field_sep)
        lines = lines[1:]
    max_records = int(spec.get("max_records") or _DEFAULT_MAX_RECORDS)
    if len(lines) > max_records:
        logger.warning(
            "Encoded field '%s' on '%s' holds %d records; keeping the first %d "
            "(max_records). The decoded table is a LOWER BOUND.",
            spec.get("field", "?"),
            source,
            len(lines),
            max_records,
        )
        lines = lines[:max_records]
    out: List[Dict[str, Any]] = []
    for line in lines:
        parts = _split_record(line, field_sep)
        if not columns:
            # No header and no declared names, so position is the only name available. Say
            # so, rather than inventing one that reads like a real column.
            rec = {f"column_{i + 1}": p for i, p in enumerate(parts)}
        else:
            rec = {c: parts[i] for i, c in enumerate(columns) if i < len(parts)}
        if rec:
            out.append(rec)
    return out


def _split_record(line: str, field_sep: str) -> List[str]:
    """One record's columns. Single-char separators go through csv (quoting survives)."""
    if len(field_sep) == 1:
        try:
            return [
                p.strip()
                for p in next(csv.reader(io.StringIO(line), delimiter=field_sep))
            ]
        except StopIteration:
            return []
        except csv.Error:
            pass
    return [p.strip() for p in line.split(field_sep)]


def _derive(record: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Split declared columns into named parts, adding them to the record.

    A part the value does not carry is omitted, never written as ``""``. An absent part and an
    empty one are different findings, and a condition reading the empty string reports on a
    value that was never there.
    """
    for rule in spec.get("derive", []) or []:
        if not isinstance(rule, dict):
            continue
        src_col = str(rule.get("from", "") or "").strip()
        names = [str(n) for n in (rule.get("into") or []) if str(n).strip()]
        sep = str(rule.get("separator", "-") or "-")
        if not src_col or not names or src_col not in record:
            continue
        value = str(record.get(src_col, "") or "")
        if not value:
            continue
        parts = value.split(sep)
        for i, name in enumerate(names):
            if i < len(parts) and str(parts[i]).strip():
                record[name] = str(parts[i]).strip()
        remainder = str(rule.get("remainder", "") or "").strip()
        if remainder and len(parts) > len(names):
            tail = sep.join(parts[len(names) :]).strip()
            if tail:
                record[remainder] = tail
    return record


def _read_path(row: Any, path: str) -> Any:
    """The raw node at a dotted ``path``, or ``None``.

    Deliberately not `correlation.resolve_path`, which returns scalar leaves (so a decoded list
    of dicts reads as absent, defeating the idempotence check) and searches aliased spellings,
    right for a pack-declared read and wrong for a write target. Both the exact flat key and the
    nested walk are accepted, since a retriever may deliver either.
    """
    if not isinstance(row, dict):
        return None
    if path in row:
        return row[path]
    node: Any = row
    for seg in path.split("."):
        if not isinstance(node, dict) or seg not in node:
            return None
        node = node[seg]
    return node


def _write_into(row: Dict[str, Any], path: str, value: Any) -> None:
    """Write ``value`` at a dotted ``path``, creating intermediate dicts.

    The nesting is the point. A list of dicts stored under a literal flat dotted key is
    invisible to every reader: `resolve_path` sees the exact key, asks for its terminal leaves,
    and a list of dicts has none, so the decode succeeds and reaches nobody. A conflicting
    non-dict intermediate is left alone and the value goes to the flat key instead, which is at
    least visible in the exports rather than lost.
    """
    segments = path.split(".")
    if len(segments) == 1:
        row[path] = value
        return
    node: Dict[str, Any] = row
    for seg in segments[:-1]:
        child = node.get(seg)
        if not isinstance(child, dict):
            if child is not None:
                logger.warning(
                    "Cannot nest decoded records under '%s' (segment '%s' is not a "
                    "mapping); writing to the flat key instead.",
                    path,
                    seg,
                )
                row[path] = value
                return
            child = {}
            node[seg] = child
        node = child
    node[segments[-1]] = value
