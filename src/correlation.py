"""
Correlation and aggregation stage between retrieval and anomaly detection.

Produces deterministic aggregates (record counts, entity occurrences, cross-source
overlaps) and, when a playbook matches, executes a declarative ``TransformPlan`` over
in-memory rows. LLM steps degrade gracefully; deterministic aggregates always flow
downstream. Output is ``CorrelationResult``.
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from human_guidance import guidance_message
from models.pydantic_models import (STUB_OBSERVED, ConditionCheck,
                                    ConditionGroup, CorrelationKey,
                                    CorrelationResult, SubjectVerdict,
                                    TransformPlan, TransformResult,
                                    TransformStep, ValidationVerdict)
from utils.error_handling import async_retry_with_backoff

logger = logging.getLogger(__name__)

# Shared with the anomaly stage; see _MAX_LLM_INPUT_CHARS in anomaly_detection.py.
_MAX_LLM_INPUT_CHARS = 15000
# Entity types that scope a whole population: matching an actor on one would flag every
# member as a subject. These may still be valid join keys; this set governs subject
# identification only. A pack's own broad types are declared `role: scope` and added
# via `_scope_entity_types`.
#: Engine-owned entity type for the incident window. A pack must not rename it.
_TIME_ENTITY = "time_window"
_BROAD_ENTITY_TYPES = {
    "organization",
    "org",
    "tenant",
    "alert_type",
    "alert_id",
    "incident_ir",
    _TIME_ENTITY,
    "type",
}
# The discovery layer rejects timestamp fields; declared and derived layers bind by name
# through pack field priors and bypass that test. This frozenset enforces the rule for all
# layers: a time-window join key holds of every row pair and is a tautology.
_NON_JOINABLE_ENTITY_TYPES = frozenset({_TIME_ENTITY})
# Cap rows scanned per source so a huge result set can't blow up the aggregation.
_MAX_ROWS_PER_SOURCE = 5000
# Cap rows emitted per transform result so a wide group-by can't blow up the prompt.
_MAX_RESULT_ROWS = 200
# Depth cap in dotted path segments; list traversal does not count toward depth.
_MAX_NEST_DEPTH = 5
# Cap on distinct leaf paths surfaced per source, so a wide/deep schema can't blow up
# the planner prompt.
_MAX_SCHEMA_LEAVES = 200
# Cap on value combinations a single row's multi-valued group_by keys can emit (guards
# against a cartesian blow-up when several list fields each carry many elements).
_MAX_ROW_COMBOS = 64

# --- data-driven join-key discovery caps (fallback layer) -------------------
_MAX_JK_SOURCES = 8  # sources scanned for join keys
_MAX_JK_LEAVES = 40  # candidate leaf fields per source
_MAX_JK_LEAVES_WITH_SUBJECTS = 120  # ceiling past it, subject-valued fields only
_MAX_JK_VALUES = 500  # distinct values sampled per field (set-intersection cost)
_MAX_JK_PAIRS = 2000  # hard cap on total field-pair comparisons
_MIN_JK_DISTINCT = 5  # min distinct values for a field to be a join candidate
_MIN_JK_CONTAINMENT = 0.35  # min max-containment overlap to call a pair a join key
_MIN_JK_SHARED = 2  # min shared distinct values (1 is too fragile)
_TOP_N_AUTO_SYNTH = 3  # top-N discovered keys to auto-correlate deterministically

# Field-name tokens (split on . and _) that suggest an entity type, used by layer-3
# discovery. Pack declarations in `entity_glossary.yaml` `field_name_hints:` take
# precedence; this is the generic fallback for tokens common across security deployments.
_ENTITY_HINT_TOKENS = {
    "user": "user",
    "uid": "user",
    "userid": "user",
    "actor": "user",
    "session": "session",
    "sessionid": "session",
    "token": "session",
    "ip": "ip",
    "ipaddress": "ip",
    "device": "device",
    "fingerprint": "device",
    "email": "email",
    "phone": "phone",
    "mobile": "phone",
    "account": "account",
    "acct": "account",
    "card": "card",
    "bin": "card",
    "transaction": "transaction",
    "txn": "transaction",
}
# Field-name tokens that mark a timestamp column (never a join key).
_TIME_TOKENS = {
    "time",
    "ts",
    "timestamp",
    "date",
    "datetime",
    "created",
    "updated",
    "raised",
    "at",
    "when",
    "inserted",
    "modified",
    "epoch",
    "start",
    "end",
}
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_DIGIT_RE = re.compile(r"\d")


def _scalar(v: Any) -> bool:
    """True for a scalar value we treat as a leaf (str/int/float, but not bool)."""
    return isinstance(v, (str, int, float)) and not isinstance(v, bool)


def _maybe_json(value: Any) -> Any:
    """Parse ``value`` if it is a JSON-string object/array, otherwise return it unchanged.

    Some backends return nested data as a JSON string (e.g. Databricks SQL STRUCT/ARRAY
    columns). Only strings starting with ``{`` or ``[`` are attempted.
    """
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s or s[0] not in "{[":
        return value
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return value


def _dict_get(d: dict, key: str):
    """Look up ``key`` in ``d`` with a case-insensitive fallback.

    Returns ``(found, value)`` to disambiguate a real ``None`` value.
    Exact match wins; covers backends that normalize identifier case (e.g. Snowflake).
    """
    if key in d:
        return True, d[key]
    lower = key.lower()
    for k, v in d.items():
        if isinstance(k, str) and k.lower() == lower:
            return True, v
    return False, None


def _dict_key(d: dict, key: str) -> str:
    """The spelling ``d`` uses for ``key`` (case-insensitive). Returns ``key`` if not found.

    Use before writing so the written key matches the existing one; writing the caller's
    spelling beside a differently-cased existing key creates a duplicate.
    """
    if key in d:
        return key
    lower = key.lower()
    for k in d:
        if isinstance(k, str) and k.lower() == lower:
            return k
    return key


def resolve_path(row: Any, path: str) -> List[Any]:
    """Resolve a dotted ``path`` against a row, returning all matching scalar leaves.

    Handles flat-dotted keys (ES/QL, Databricks), nested dicts, JSON-string nested
    values, and underscore-flattened aliases (``creator.code.value`` ->
    ``creator_code_value``). Lists at any segment are traversed and results flattened.
    """
    # Fast path: the exact flat key exists (`ES|QL` / Databricks dotted-string columns).
    if isinstance(row, dict) and path in row:
        return _terminal_leaves(row[path])
    # Underscore-flattened alias (Databricks struct-leaf aliasing convention). Try the
    # full path, then progressively shorter prefixes, descending into the remainder; so
    # both a scalar leaf (`creator.code.value`→`creator_code_value`) and an aliased struct/
    # JSON column with a nested tail (`contents.items.origin` where the row has
    # `contents_items` holding a JSON struct) resolve.
    if isinstance(row, dict) and "." in path:
        segments = path.split(".")
        for cut in range(len(segments), 0, -1):
            alias = "_".join(segments[:cut])
            found, val = _dict_get(row, alias)
            if not found:
                continue
            rest = segments[cut:]
            if not rest:
                leaves = _terminal_leaves(val)
                if leaves:
                    return leaves
                continue
            # Descend the remaining segments into the aliased value (may be a JSON string
            # struct, a live dict, or a list; `_resolve_segments` handles all three).
            resolved = _resolve_segments(val, rest)
            if resolved:
                return resolved
    return _resolve_segments(row, path.split("."))


def resolve_containers(row: Any, path: str) -> List[Any]:
    """Every node a dotted path lands on, containers included, one per array element.

    Unlike :func:`resolve_path`, which returns only scalars, this returns dicts/objects
    so callers can read paired fields from the same sub-record. An empty path returns the
    row itself. Lists are flattened to their elements.
    """
    if not str(path or "").strip():
        return [row]
    if isinstance(row, dict):
        found, value = _dict_get(row, path)
        if found:
            return _as_containers(value)
    return _resolve_container_segments(row, str(path).split("."))


def _as_containers(value: Any) -> List[Any]:
    """A resolved node as a list of containers; a list becomes its elements."""
    value = _maybe_json(value)
    if isinstance(value, list):
        out: List[Any] = []
        for item in value:
            out.extend(_as_containers(item))
        return out
    return [] if value is None else [value]


def _resolve_container_segments(value: Any, segments: List[str]) -> List[Any]:
    """:func:`_resolve_segments` for containers: descend, then stop without flattening."""
    if not segments:
        return _as_containers(value)
    value = _maybe_json(value)
    head, rest = segments[0], segments[1:]
    if isinstance(value, list):
        out: List[Any] = []
        for item in value:
            out.extend(_resolve_container_segments(item, segments))
        return out
    if isinstance(value, dict):
        remaining = ".".join(segments)
        found, node = _dict_get(value, remaining)
        if found:
            return _as_containers(node)
        found, child = _dict_get(value, head)
        if found:
            return _resolve_container_segments(child, rest)
    return []


def _terminal_leaves(value: Any) -> List[Any]:
    """Scalar leaves of a fully-consumed path: a scalar or an array of scalars.

    A path may legitimately end on a repeated field; a list of scalars yields all
    its elements rather than being treated as absent.
    """
    value = _maybe_json(value)
    if _scalar(value):
        return [value]
    if isinstance(value, list):
        out: List[Any] = []
        for item in value:
            out.extend(_terminal_leaves(item))
        return out
    return []


def _resolve_segments(value: Any, segments: List[str]) -> List[Any]:
    if not segments:
        return _terminal_leaves(value)
    value = _maybe_json(value)  # descend into JSON-string nested data
    head, rest = segments[0], segments[1:]
    if isinstance(value, list):
        out: List[Any] = []
        for item in value:
            out.extend(_resolve_segments(item, segments))
        return out
    if isinstance(value, dict):
        # Support a partially-flattened key that still contains dots (e.g. the dict
        # has the literal remaining path as one key).
        remaining = ".".join(segments)
        found, leaf = _dict_get(value, remaining)
        if found:
            return _terminal_leaves(leaf)
        found, child = _dict_get(value, head)
        if found:
            return _resolve_segments(child, rest)
    return []


def flatten_leaves(row: Any, prefix: str = "", depth: int = 0) -> Dict[str, List[Any]]:
    """Union all dotted leaf paths of a nested row -> their scalar values.

    Descends dicts and lists-of-dicts (depth-capped at ``_MAX_NEST_DEPTH``); a list of
    scalars is a terminal leaf whose values are all collected under one path. Powers
    schema derivation and value scanning. Flat-dotted rows (``ES|QL``/Databricks) already
    have dotted keys, so they surface verbatim.
    """
    out: Dict[str, List[Any]] = defaultdict(list)

    def _walk(value: Any, path: str, d: int) -> None:
        value = _maybe_json(value)  # descend into JSON-string nested data
        if _scalar(value):
            if path:
                out[path].append(value)
            return
        if d >= _MAX_NEST_DEPTH:
            return
        if isinstance(value, dict):
            for k, v in value.items():
                child = f"{path}.{k}" if path else k
                _walk(v, child, d + 1)
        elif isinstance(value, list):
            # Map the same path over each element (list index is not part of the path),
            # so list-of-dicts leaves and scalar-list leaves collect under one path.
            for item in value:
                _walk(item, path, d)

    if isinstance(row, dict):
        _walk(row, prefix, depth)
    return dict(out)


def _row_values(row: Dict[str, Any]) -> List[str]:
    """Flatten a row's scalar leaves (nested + flat) to strings for matching."""
    out: List[str] = []
    for values in flatten_leaves(row).values():
        out.extend(str(v) for v in values)
    return out


def aggregate(logs: Dict[str, List[Dict]], entity_values: List[str]) -> Dict[str, Any]:
    """Deterministic, LLM-free aggregation over retrieved rows.

    ``entity_values`` are the incident's extracted entity identifiers; we count how
    often each appears in each source and which span multiple sources.
    """
    record_counts: Dict[str, int] = {}
    # entity value -> source -> count
    occurrences: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    wanted = [v for v in entity_values if v and v != "*"]

    for source, rows in (logs or {}).items():
        rows = rows or []
        record_counts[source] = len(rows)
        for row in rows[:_MAX_ROWS_PER_SOURCE]:
            if not isinstance(row, dict):
                continue
            haystack = _row_values(row)
            for value in wanted:
                if any(value == cell or value in cell for cell in haystack):
                    occurrences[value][source] += 1

    # Cross-source overlap: entity values seen in more than one source.
    cross_source = {
        value: dict(by_source)
        for value, by_source in occurrences.items()
        if len(by_source) > 1
    }

    return {
        "record_counts": record_counts,
        "total_records": sum(record_counts.values()),
        "entity_occurrences": {v: dict(s) for v, s in occurrences.items()},
        "cross_source_overlap": cross_source,
    }


def derive_schema(logs: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
    """Union the actual dotted leaf paths of each source's rows -> the columns present.

    For nested (Kibana ``_source``) rows this yields dotted leaf paths
    (``actor.unitId``, ``notice.entries.origin.unit``); for flat-dotted
    (``ES|QL``/Databricks) rows the dotted keys surface verbatim. The transform planner
    picks fields only from these, so it can never invent a column that isn't actually
    in the retrieved data.
    """
    schema: Dict[str, List[str]] = {}
    for source, rows in (logs or {}).items():
        cols: Dict[str, None] = {}
        for row in (rows or [])[:_MAX_ROWS_PER_SOURCE]:
            if isinstance(row, dict):
                for leaf in flatten_leaves(row).keys():
                    cols[leaf] = None
                    if len(cols) >= _MAX_SCHEMA_LEAVES:
                        break
        schema[source] = list(cols.keys())
    return schema


# --- data-driven join-key discovery (deterministic fallback layer) ----------


def _name_tokens(field_name: str) -> List[str]:
    """Lowercase tokens of a dotted or underscored field name.

    Splits on ``.`` and ``_`` only, because those are the two separators a column name
    uses. It is deliberately not a prose splitter: a column name contains no spaces,
    so run over a sentence this returns the whole sentence as one token, which can then
    never match anything. Free text goes through :func:`_prose_tokens`.
    """
    return [t for t in re.split(r"[._]", field_name.lower()) if t]


def _prose_tokens(text: str) -> List[str]:
    """Lowercase word tokens of free text; spaces, punctuation and dashes all split.

    The counterpart to :func:`_name_tokens`, and separate from it on purpose: reusing the
    field-name splitter on a human-written title is how a title came to be scored as a
    single unsplittable token (see :meth:`CorrelationModule._playbook_correlation_spec`).
    ``\\W`` is Unicode-aware, so a non-``ASCII`` title keeps its letters instead of being
    shredded into initials; ``_`` is added because it is a word character to ``re``.
    """
    return [t for t in re.split(r"[\W_]+", text.lower()) if t]


def _is_timestamp_field(field_name: str, values: set) -> bool:
    """True if the field looks like a timestamp (by name or by value format)."""
    if any(t in _TIME_TOKENS for t in _name_tokens(field_name)):
        return True
    if not values:
        return False
    iso = sum(1 for v in values if _ISO_DATE_RE.match(str(v)))
    return iso >= 0.5 * len(values)


def _subject_value_set(analysis) -> Set[str]:
    """The incident's own extracted entity values, lowercased, for the floor exemptions.

    Every type is included, broad ones too: this answers "did the incident name this value",
    not "is this value narrow enough to identify a person-of-interest" (see
    ``_scope_entity_types`` for the latter).
    """
    out: Set[str] = set()
    for e in getattr(analysis, "extracted_entities", []) or []:
        v = getattr(e, "value", None)
        if isinstance(v, str) and v.strip():
            out.add(v.strip().lower())
    return out


def _all_values_are_subjects(values: set, subject_values: Optional[Set[str]]) -> bool:
    """True if every value in ``values`` is one the incident itself named.

    Compared case-insensitively on the stripped string, because a column may hold the
    same identifier the alert did in a different case (`ttt1u20np` vs `TTT1U20NP`) and a
    case-sensitive miss here reads as "not a subject field", i.e. as the default.
    """
    if not values or not subject_values:
        return False
    lowered = {str(v).strip().lower() for v in values if str(v).strip()}
    return bool(lowered) and lowered <= subject_values


def _is_candidate_field(
    field_name: str,
    values: set,
    key_filter: str = "strict",
    subject_values: Optional[Set[str]] = None,
) -> bool:
    """True if this field is a plausible join candidate.

    ``key_filter``: ``strict`` requires identifier shape (avg len >= 3, majority
    digit-bearing); ``cardinality`` keeps any high-cardinality field; ``both`` is
    the union. Timestamps are always rejected.

    ``subject_values`` exempts a field from the cardinality floor when every value it
    holds is one the incident itself named: a targeted query returns exactly one value
    per alerted entity, which would otherwise be rejected as low-cardinality.
    """
    if len(values) < _MIN_JK_DISTINCT and not _all_values_are_subjects(
        values, subject_values
    ):
        return False
    if _is_timestamp_field(field_name, values):
        return False
    strvals = [str(v) for v in values]
    avg_len = sum(len(s) for s in strvals) / len(strvals)
    if avg_len < 3:
        return False
    if key_filter in ("cardinality", "both"):
        return True
    # strict: require majority of values to carry a digit (identifier shape).
    with_digit = sum(1 for s in strvals if _DIGIT_RE.search(s))
    if with_digit < 0.5 * len(strvals):
        return False
    # Reject small purely-numeric ranges (sequential counts / small codes).
    if all(s.isdigit() for s in strvals) and len(values) < 20:
        return False
    return True


def _build_field_value_index(
    logs: Dict[str, List[Dict]],
    key_filter: str = "strict",
    subject_values: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, set]]:
    """Per source, {field -> set of scalar string values}, filtered to join candidates.

    Reuses ``flatten_leaves`` so every row shape (nested / flat-dotted / JSON-string /
    case-variant) is handled. Bounded by the ``_MAX_JK_*`` caps; adds no I/O."""
    result: Dict[str, Dict[str, set]] = {}
    for source in list((logs or {}).keys())[:_MAX_JK_SOURCES]:
        field_vals: Dict[str, set] = defaultdict(set)
        for row in (logs.get(source) or [])[:_MAX_ROWS_PER_SOURCE]:
            if not isinstance(row, dict):
                continue
            for leaf, vals in flatten_leaves(row).items():
                if leaf not in field_vals and len(field_vals) >= _MAX_JK_LEAVES:
                    # Subject fields are admitted up to the higher ceiling so the cardinality
                    # exemption in _is_candidate_field can actually reach them.
                    if len(
                        field_vals
                    ) >= _MAX_JK_LEAVES_WITH_SUBJECTS or not _all_values_are_subjects(
                        set(vals), subject_values
                    ):
                        continue
                bucket = field_vals[leaf]
                for v in vals:
                    if len(bucket) >= _MAX_JK_VALUES:
                        break
                    bucket.add(str(v))
        result[source] = {
            k: v
            for k, v in field_vals.items()
            if _is_candidate_field(k, v, key_filter, subject_values)
        }
    return result


def _field_values(rows: List[Dict], field: str) -> Set[str]:
    """Distinct non-empty string values of ``field`` over ``rows``, via ``resolve_path``."""
    out: Set[str] = set()
    for row in (rows or [])[:_MAX_ROWS_PER_SOURCE]:
        if not isinstance(row, dict):
            continue
        for v in resolve_path(row, field):
            if v is None:
                continue
            text = str(v).strip()
            if text:
                out.add(text)
        if len(out) >= _MAX_JK_VALUES:
            break
    return out


def _infer_entity_hint(
    *field_names: str, hints: Optional[Dict[str, str]] = None
) -> str:
    """Map field-name tokens to an entity type; ``'unknown'`` if none match.

    Pack ``hints`` win token-by-token over the engine fallback, so domain-specific
    identifiers are labelled with their domain's entity type.
    """
    hints = hints or {}
    for name in field_names:
        for tok in _name_tokens(name):
            if tok in hints:
                return hints[tok]
            if tok in _ENTITY_HINT_TOKENS:
                return _ENTITY_HINT_TOKENS[tok]
    return "unknown"


class _UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def discover_join_keys(
    logs: Dict[str, List[Dict]],
    schema: Dict[str, List[str]],
    key_filter: str = "strict",
    entity_hints: Optional[Dict[str, str]] = None,
    subject_values: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Find shared identifier fields across sources by comparing value sets.

    Deterministic, LLM-free. ``entity_hints`` labels discovered keys; detection is pure
    value-set arithmetic. Uses max-containment overlap (``|A∩B| / min(|A|,|B|)``).
    Returns ranked ``[{entity_hint, sources, overlap_score, shared_value_count,
    sample_values}]``, one entry per transitively-grouped key. Returns ``[]`` for <2
    sources or when nothing clears the thresholds.
    """
    fv = _build_field_value_index(logs, key_filter, subject_values)
    sources = list(fv.keys())
    if len(sources) < 2:
        return []

    raw_pairs: List[Dict[str, Any]] = []
    comparisons = 0
    stop = False
    for i, sa in enumerate(sources):
        if stop:
            break
        for sb in sources[i + 1 :]:
            if stop:
                break
            for fa, va in fv[sa].items():
                for fb, vb in fv[sb].items():
                    if comparisons >= _MAX_JK_PAIRS:
                        stop = True
                        break
                    comparisons += 1
                    shared = va & vb
                    if len(shared) < _MIN_JK_SHARED and not _all_values_are_subjects(
                        shared, subject_values
                    ):
                        # A single shared value is a coincidence risk unless the incident
                        # named it: a query filtered on it, so the match is not accidental.
                        continue
                    containment = len(shared) / min(len(va), len(vb))
                    if containment < _MIN_JK_CONTAINMENT:
                        continue
                    raw_pairs.append(
                        {
                            "sa": sa,
                            "fa": fa,
                            "sb": sb,
                            "fb": fb,
                            "shared": len(shared),
                            "containment": round(containment, 3),
                            "sample": sorted(shared)[:5],
                        }
                    )
    if not raw_pairs:
        return []

    raw_pairs.sort(key=lambda p: (-p["containment"], -p["shared"]))

    # Group transitively so fieldX~fieldY~fieldZ collapse to one logical key.
    uf = _UnionFind()
    for p in raw_pairs:
        uf.union(f"{p['sa']}.{p['fa']}", f"{p['sb']}.{p['fb']}")

    groups: Dict[str, Dict[str, str]] = defaultdict(dict)
    best: Dict[str, Dict[str, Any]] = {}
    for p in raw_pairs:
        root = uf.find(f"{p['sa']}.{p['fa']}")
        groups[root][p["sa"]] = p["fa"]
        groups[root][p["sb"]] = p["fb"]
        if root not in best:
            best[root] = p

    result: List[Dict[str, Any]] = []
    seen: set = set()
    for p in raw_pairs:
        root = uf.find(f"{p['sa']}.{p['fa']}")
        if root in seen:
            continue
        seen.add(root)
        b = best[root]
        result.append(
            {
                "entity_hint": _infer_entity_hint(b["fa"], b["fb"], hints=entity_hints),
                "sources": dict(groups[root]),
                "overlap_score": b["containment"],
                "shared_value_count": b["shared"],
                "sample_values": b["sample"],
            }
        )
    return result


# --- transform executor (pure Python, deterministic, defensive) -------------


def _coerce_num(v: Any) -> Optional[float]:
    try:
        if isinstance(v, bool):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# Floor below which a number is not treated as an epoch timestamp (1973-03-03).
# Without this, small integers (ids, counters) parse as 1970 dates.
_MIN_EPOCH_SECONDS = 1e8

# Normalise fractional seconds to 6 digits for `datetime.fromisoformat` (pre-3.11).
_FRACTION_RE = re.compile(r"\.(\d{1,9})(?=$|[+\-])")

# Normalise offset spellings `fromisoformat` rejects before 3.11 (`+0000`, `+00`).
# Anchored after a time-of-day segment so a bare date ending in `-20` is not rewritten.
_OFFSET_RE = re.compile(r"^(.*\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)([+-]\d{2})(\d{2})?$")


def _parse_ts(v: Any) -> Optional[datetime]:
    """Best-effort timestamp parse; returns None on anything unparseable.

    Handles epoch numbers (int/float or all-digit string): 13 digits = milliseconds,
    10 digits = seconds; many backends return event times as epoch ints.
    Also handles ISO-8601 (``Z`` normalized) and a few string formats.

    A number below ``_MIN_EPOCH_SECONDS`` is not a timestamp: an id or a counter would
    otherwise become a 1970 date and be indistinguishable from a real one.
    """
    # Epoch numbers (int/float or a purely-numeric string).
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip().isdigit()):
        try:
            n = float(v)
        except (TypeError, ValueError):
            return None
        # Distinguish ms from s by magnitude: >= 1e11 (~year 5138 in seconds) means ms.
        if n >= 1e11:
            n /= 1000.0
        if n < _MIN_EPOCH_SECONDS:
            return None
        try:
            return datetime.fromtimestamp(n, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None
    if not isinstance(v, str) or not v:
        return None
    text = v.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    # Normalise fractional seconds before retry: `fromisoformat` accepts exactly 3 or 6
    # fractional digits before Python 3.11.
    padded = _FRACTION_RE.sub(lambda m: "." + m.group(1)[:6].ljust(6, "0"), text, count=1)
    if padded != text:
        try:
            return datetime.fromisoformat(padded)
        except ValueError:
            pass
    # Normalise offset without colon (e.g. Spark's `±HHMM` default rendering).
    normalized = _OFFSET_RE.sub(
        lambda m: m.group(1) + m.group(2) + ":" + (m.group(3) or "00"), padded, count=1
    )
    if normalized != padded:
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(v.strip(), fmt)
        except ValueError:
            continue
    return None


# Rows sampled per candidate when measuring time-column granularity.
_TIME_FIELD_SAMPLE = 200

# Name tokens that imply sub-day resolution; `date` is absent because it does not.
_SUBDAY_TIME_TOKENS = {"time", "ts", "timestamp", "datetime", "at", "when", "epoch"}


def _time_field_granularity(rows: List[Dict], leaf: str) -> tuple:
    """Measured resolution of ``leaf`` as an event-time column, as a sort key.

    Returns ``(subday, parsed)``, both higher-is-better:

    * ``subday``: 1 if any parsed value carries a non-zero time-of-day (an instant, not
      a calendar day).
    * ``parsed``: count of values that parsed; a rarely-populated column loses to a full one.

    A leaf whose values never parse scores ``(0, 0)`` and is rejected by the caller.
    """
    parsed = 0
    subday = 0
    for row in rows[:_TIME_FIELD_SAMPLE]:
        if not isinstance(row, dict):
            continue
        for v in resolve_path(row, leaf):
            dt = _parse_ts(v)
            if dt is None:
                continue
            parsed += 1
            if dt.hour or dt.minute or dt.second or dt.microsecond:
                subday = 1
    return (subday, parsed)


def pick_time_field(leaves: List[str], rows: List[Dict]) -> str:
    """Best event-time column among ``leaves``, scored over ``rows``; ``""`` if none.

    Ranks by sub-day resolution: a column with a time-of-day beats one with only a
    date. Declaration order breaks ties. Candidates whose values never parse are excluded.
    """
    cands = [
        leaf for leaf in leaves if any(t in _TIME_TOKENS for t in _name_tokens(leaf))
    ]
    if not cands:
        return ""
    scored = []
    for i, leaf in enumerate(cands):
        subday, parsed = _time_field_granularity(rows, leaf)
        if not parsed:
            continue
        name_subday = 1 if set(_name_tokens(leaf)) & _SUBDAY_TIME_TOKENS else 0
        scored.append((subday, name_subday, -i, leaf))
    if not scored:
        # Nothing parsed: keep the name-order answer so an unseen source is unchanged.
        return cands[0]
    scored.sort(reverse=True)
    return scored[0][-1]


def _bucket_key(dt: datetime, bucket: str) -> str:
    """Floor a datetime to a bucket like '1h' or '1d'. Defaults to day."""
    b = (bucket or "1d").lower()
    if b.endswith("h"):
        return dt.strftime("%Y-%m-%d %H:00")
    if b.endswith("m"):
        return dt.strftime("%Y-%m-%d %H:%M")
    return dt.strftime("%Y-%m-%d")


_OPERATORS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
}


def _key_combos(row: Dict, keys: List[str]) -> List[tuple]:
    """Cartesian product of each key's resolved values for one row.

    A key resolving to a list field (e.g. ``records.locator``) contributes every value;
    a missing key contributes ``""`` so the row still groups. Capped at
    ``_MAX_ROW_COMBOS`` to prevent a multi-list cartesian blow-up.
    """
    per_key: List[List[str]] = []
    for k in keys:
        vals = [str(v) for v in resolve_path(row, k)] or [""]
        per_key.append(vals)
    combos: List[tuple] = [()]
    for vals in per_key:
        combos = [c + (v,) for c in combos for v in vals]
        if len(combos) > _MAX_ROW_COMBOS:
            combos = combos[:_MAX_ROW_COMBOS]
            break
    return combos


def _exec_group_by(step: TransformStep, logs: Dict[str, List[Dict]]) -> List[Dict]:
    rows = logs.get(step.source) or []
    groups: Dict[tuple, int] = defaultdict(int)
    distinct_vals: Dict[tuple, set] = defaultdict(set)
    for row in rows[:_MAX_ROWS_PER_SOURCE]:
        if not isinstance(row, dict):
            continue
        for key in _key_combos(row, step.keys):
            if step.agg == "distinct":
                # Register the group even if the target field is absent for this row
                # (defaultdict access materializes the key).
                distinct_vals[key].update(str(v) for v in resolve_path(row, step.field))
            else:
                groups[key] += 1
    out = []
    src = distinct_vals if step.agg == "distinct" else groups
    for key in src:
        rec = {k: key[i] for i, k in enumerate(step.keys)}
        rec["metric"] = (
            len(distinct_vals[key]) if step.agg == "distinct" else groups[key]
        )
        out.append(rec)
    out.sort(key=lambda r: r["metric"], reverse=True)
    return out


def _exec_distinct(step: TransformStep, logs: Dict[str, List[Dict]]) -> List[Dict]:
    rows = logs.get(step.source) or []
    values: set = set()
    for r in rows[:_MAX_ROWS_PER_SOURCE]:
        if isinstance(r, dict):
            values.update(str(v) for v in resolve_path(r, step.field))
    return [{"field": step.field, "distinct_count": len(values)}]


def _exec_time_bucket(step: TransformStep, logs: Dict[str, List[Dict]]) -> List[Dict]:
    rows = logs.get(step.source) or []
    counts: Dict[str, int] = defaultdict(int)
    for row in rows[:_MAX_ROWS_PER_SOURCE]:
        if not isinstance(row, dict):
            continue
        for raw in resolve_path(row, step.field):
            dt = _parse_ts(raw)
            if dt is not None:
                counts[_bucket_key(dt, step.bucket)] += 1
    out = [{"bucket": k, "metric": v} for k, v in counts.items()]
    out.sort(key=lambda r: r["bucket"])
    return out


def _exec_threshold(step: TransformStep, prior: Dict[str, List[Dict]]) -> List[Dict]:
    """Filter a prior group_by/time_bucket result (by its label) on metric vs value."""
    source_rows = prior.get(step.over, [])
    cmp = _OPERATORS.get(step.operator, _OPERATORS[">"])
    out = []
    for row in source_rows:
        metric = _coerce_num(row.get("metric"))
        if metric is not None and cmp(metric, step.value):
            out.append(row)
    return out


def _window_delta(time_window: str) -> Optional[timedelta]:
    """Parse a time_window spec into a max allowed gap. None = no bound (any time).

    Accepts: 'same_day' (calendar day), 'within:<N>h'/'within:<N>d'/'within:<N>m',
    'sequence' (treated as unbounded ordering, presence-only). Unknown/empty: None.
    """
    w = (time_window or "").strip().lower()
    if not w or w == "sequence":
        return None
    if w == "same_day":
        return timedelta(days=1)  # handled specially (calendar day) below
    m = re.match(r"within:\s*(\d+)\s*([hdm])", w)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        return {
            "h": timedelta(hours=n),
            "d": timedelta(days=n),
            "m": timedelta(minutes=n),
        }[unit]
    return None


def _norm_dt(dt: datetime) -> datetime:
    """Make a datetime tz-aware (assume UTC when naive) so mixed-source subtraction
    (epoch → tz-aware UTC vs ISO-without-offset → naive) never raises."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _within_window(
    times_by_source: Dict[str, List[datetime]], time_window: str
) -> bool:
    """True if timestamps across the sources co-occur within the window.

    ``same_day`` requires a shared calendar day; ``within:Nx`` requires the chosen
    instants to sit within N of each other. Only sources that actually have parsed
    timestamps participate in the check; a source with no parseable time contributes
    presence only and does not veto the join. Fewer than two timed sources: return True.
    """
    w = (time_window or "").strip().lower()
    # Keep only sources that have at least one parsed timestamp.
    times = [[_norm_dt(t) for t in ts if t] for ts in times_by_source.values()]
    times = [ts for ts in times if ts]
    if len(times) < 2:
        return True  # not enough timed sources to gate on; presence is enough
    if w == "same_day":
        day_sets = [{t.date() for t in ts} for ts in times]
        common = set.intersection(*day_sets) if day_sets else set()
        return bool(common)
    delta = _window_delta(time_window)
    if delta is None:
        return True  # unbounded (e.g. sequence); presence is enough
    # Ask whether some window of width `delta` contains at least one event from every
    # source: sweep the merged timeline, for each event as the left edge count how many
    # distinct sources fall in [t, t+delta].
    tagged = sorted(
        (t, i) for i, ts in enumerate(times) for t in ts
    )  # O(n log n) on the merged stream
    need = len(times)
    left = 0
    counts: Dict[int, int] = defaultdict(int)
    distinct = 0
    for right in range(len(tagged)):
        counts[tagged[right][1]] += 1
        if counts[tagged[right][1]] == 1:
            distinct += 1
        # Shrink from the left while the window is wider than delta.
        while tagged[right][0] - tagged[left][0] > delta:
            counts[tagged[left][1]] -= 1
            if counts[tagged[left][1]] == 0:
                distinct -= 1
            left += 1
        if distinct == need:
            return True
    return False


def _exec_cross_source_overlap(
    step: TransformStep,
    logs: Dict[str, List[Dict]],
    entity_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict]:
    """Find entity values appearing across >1 source (optionally time-gated).

    ``step.entity`` may be an entity type in ``entity_map`` (each source resolved via its
    own mapped field) or a literal field path (resolved uniformly across all sources).

    When ``step.time_window`` is set, a shared value counts only if the sources' events
    for that value co-occur within the window (using ``step.time_fields`` per source).
    """
    entity_map = entity_map or {}
    sources = step.sources or list(logs.keys())
    entity_type = step.entity or step.field
    time_window = step.time_window or ""
    time_fields = step.time_fields or {}
    # value -> source -> list of event datetimes (for time-gating)
    seen: Dict[str, Dict[str, List[datetime]]] = defaultdict(lambda: defaultdict(list))
    for source in sources:
        # Prefer the per-source mapped field for this entity type; else the literal path.
        field = (entity_map.get(source) or {}).get(entity_type) or entity_type
        tfield = time_fields.get(source, "")
        for row in (logs.get(source) or [])[:_MAX_ROWS_PER_SOURCE]:
            if not isinstance(row, dict):
                continue
            vals = resolve_path(row, field)
            if not vals:
                continue
            row_times = []
            if time_window and tfield:
                row_times = [
                    t for t in (_parse_ts(x) for x in resolve_path(row, tfield)) if t
                ]
            for val in vals:
                text = str(val)
                if text:
                    seen[text][source].extend(row_times)
    # Fold a value seen in only one source into a matching multi-source group so that
    # the same identifier with a suffix variant does not produce an empty join.
    # Uses `_identifiers_match` (the same predicate the verdict engine uses), and only
    # values that are alone in their source can be folded.
    singles = [v for v, by in seen.items() if len(by) == 1]
    if singles:
        for value in singles:
            for other, by_other in seen.items():
                if other == value or value not in seen:
                    continue
                if not _identifiers_match(value, other):
                    continue
                # Fold into `other`; the value reported stays a real spelling from the
                # data so an operator can search for it.
                for src, times in seen[value].items():
                    by_other[src].extend(times)
                del seen[value]
                break

    out = []
    for value, by_source in seen.items():
        if len(by_source) <= 1:
            continue
        if time_window and not _within_window(dict(by_source), time_window):
            continue
        rec = {"value": value, "sources": sorted(by_source.keys())}
        if time_window:
            rec["time_window"] = time_window
        out.append(rec)
    return out


def execute_plan(
    plan: TransformPlan,
    logs: Dict[str, List[Dict]],
    entity_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[TransformResult]:
    """Run a TransformPlan over the retrieved rows. Defensive: a bad step is logged
    and skipped (recorded as an empty result with a note), never raised.

    ``entity_map`` ({source: {entity_type: field}}) lets cross_source_overlap line up
    an entity type across sources that name its field differently."""
    results: List[TransformResult] = []
    by_label: Dict[str, List[Dict]] = {}
    for step in plan.steps or []:
        try:
            if step.op == "group_by":
                rows = _exec_group_by(step, logs)
            elif step.op == "distinct":
                rows = _exec_distinct(step, logs)
            elif step.op == "time_bucket":
                rows = _exec_time_bucket(step, logs)
            elif step.op == "threshold":
                rows = _exec_threshold(step, by_label)
            elif step.op == "cross_source_overlap":
                rows = _exec_cross_source_overlap(step, logs, entity_map)
            else:
                logger.info("Unknown transform op '%s'; skipping.", step.op)
                continue
            # Truncate here (not inside each executor) so the note is always written.
            # A count that reads as "N rows" must not silently be a cap.
            note = ""
            if len(rows) > _MAX_RESULT_ROWS:
                note = (
                    f"TRUNCATED: {len(rows)} rows computed, the top {_MAX_RESULT_ROWS} "
                    f"kept — this count is a cap, not a total"
                )
                logger.info(
                    "Transform step '%s' (%s) truncated: %d rows -> %d.",
                    step.label,
                    step.op,
                    len(rows),
                    _MAX_RESULT_ROWS,
                )
                rows = rows[:_MAX_RESULT_ROWS]
            by_label[step.label] = rows
            results.append(
                TransformResult(label=step.label, op=step.op, rows=rows, note=note)
            )
        except Exception as e:  # one bad step must not break correlation
            logger.warning(
                "Transform step '%s' (%s) failed: %s", step.label, step.op, e
            )
            results.append(
                TransformResult(
                    label=step.label, op=step.op, rows=[], note=f"failed: {e}"
                )
            )
    return results


_PLAN_SYSTEM_PROMPT = (
    "You are a fraud-investigation analyst. Given an incident, the matched fraud "
    "playbook, the REAL columns available per retrieved log source, and deterministic "
    "aggregates, plan the transformations that surface this fraud pattern.\n"
    "Emit a TransformPlan of steps using ONLY these ops:\n"
    "- group_by: count rows (or distinct `field`) per `keys` in a `source`.\n"
    "- distinct: count distinct values of `field` in a `source`.\n"
    "- time_bucket: count rows per time `bucket` ('1h'/'1d') of a timestamp `field`.\n"
    "- threshold: keep rows of a prior step (referenced by its `label` via `over`) "
    "where metric `operator` `value` (flags outliers).\n"
    "- cross_source_overlap: values of `entity` appearing across `sources`. Set "
    "`entity` to an ENTITY TYPE from the resolved keys / normalized entity map (e.g. "
    "'record', 'org_unit') when given — each source is resolved via ITS OWN mapped field, so "
    "the same entity correlates even when sources name the field differently. Prefer "
    "entity types present in >=2 sources. Optionally set `time_window` ('same_day' or "
    "'within:24h') with `time_fields` ({source: timestamp field}) to require the events "
    "to co-occur. Only fall back to a literal column path if no entity type fits.\n"
    "Choose `source`/`keys`/`field` ONLY from the provided columns — never invent a "
    "column. Give each step a short, unique `label`. Plan only what the playbook and "
    "incident justify; prefer a few targeted steps over many."
)

_NARRATE_SYSTEM_PROMPT = (
    "You are a fraud-investigation analyst correlating retrieved log data. Given "
    "deterministic aggregates, the results of playbook-driven transforms, and playbook "
    "guidance, surface the cross-entity and cross-source patterns relevant to the "
    "incident. Return concise findings and a short summary."
)


# --- Pack-driven validation verdict engine (domain-agnostic, pure, no LLM/IO) ---
#
# Evaluates a subject against a knowledge-pack ruleset (``ruleset_spec()``).
# All conditions are dispatched by ``kind``; no procedure-specific logic lives here.
# Missing data makes a check ``unknown``; the subject verdict degrades to "insufficient".

# Forbidden-record match & boolean-flag truthy tokens.
_TRUTHY = {"true", "1", "yes", "y", "t"}


def _norm_identifier(v: Any) -> str:
    """Normalize an actor identifier for comparison: strip non-alphanumerics, uppercase."""
    return re.sub(r"[^A-Za-z0-9]", "", str(v or "")).upper()


def _identifiers_match(a: str, b: str) -> bool:
    """Two identifiers match if equal or one is a prefix of the other (>=4 chars shared).

    Covers a stored identifier being compared against a padded or suffixed form of itself
    (``AB12CD`` vs ``AB12CDXY``) while still distinguishing unrelated ones. A login name
    A login name won't match a coded identifier; the coded field is compared separately.
    """
    na, nb = _norm_identifier(a), _norm_identifier(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    return len(shorter) >= 4 and longer.startswith(shorter)


def _matches_any_identifier(value: Any, normalized: Set[str]) -> bool:
    """True if ``value`` matches any identifier in a pre-normalised set.

    Caller must normalise the set once with :func:`_norm_identifier`; this function is
    called per row and normalisation inside the loop would be redundant.
    """
    if not normalized:
        return False
    n = _norm_identifier(value)
    if not n:
        return False
    if n in normalized:
        return True
    return any(
        len(short) >= 4 and long.startswith(short)
        for short, long in ((n, k) if len(n) <= len(k) else (k, n) for k in normalized)
    )


# --- Pack-declared equivalence forms ----------------------------------------------------
#
# A pattern can be "several textual values that are not equal but are the same thing for the
# purpose being adjudicated". The engine may not own that relation: what counts as the same
# thing is a deployment's own rule, it differs between deployments of one domain, and it is
# the judgement the pack exists to hold. So what follows is a vocabulary of text OPERATIONS
# and no relation — a pack names a pipeline in `shared/equivalence_forms.yaml`, and every kind
# that counts or compares text can be pointed at it.
#
# The three normalisation modes this replaces were the opposite shape: `_values_match`'s
# `id_suffix` is exactly {project: [{case: upper}, {keep: alnum}], compare: {contains: true},
# min_length: 6} — a pack's decision taken in Python, where no pack could restate the 6.

#: Value -> key. Two values are equivalent when their keys are equal, so a projection-only
#: form is transitive by construction and costs one pass over the rows.
_PROJECTION_OPS = (
    "case",
    "keep",
    "strip",
    "prefix",
    "suffix",
    "tokens",
    "collapse_repeats",
    "map",
)

#: Two values -> bool. NOT transitive, which is why a form declaring one must also declare
#: how the question is shaped (an `anchor:` on the condition, or `linkage:` on the form), and
#: why it costs a pairwise pass the engine reports the size of.
_COMPARISON_OPS = (
    "exact",
    "contains",
    "shared_prefix",
    "shared_suffix",
    "edit_distance",
    "min_overlap",
)

#: What each operation REQUIRES, in the author's words. The engine defaults none of them: an
#: absent parameter reads `unknown` and never equality, because a silent fallback to `exact`
#: is the one wrong answer that still looks like a working check. `pack_validate` derives the
#: required-parameter rule from these keys rather than keeping a second copy of the list.
_OP_PARAMS = {
    "case": "upper, lower or fold",
    "keep": "alnum, alpha or digits",
    "strip": "the characters to remove from both ends",
    "prefix": "how many leading characters to keep",
    "suffix": "how many trailing characters to keep",
    "tokens": "a mapping declaring `split`, and optionally `order` and `take`",
    "collapse_repeats": "true",
    "map": "a mapping of value to canonical value",
    "exact": "true",
    "contains": "true",
    "shared_prefix": "how many leading characters must agree",
    "shared_suffix": "how many trailing characters must agree",
    "edit_distance": "the largest number of edits still counted as the same",
    "min_overlap": "the shortest run of shared characters counted as the same",
}

#: How a pairwise question is shaped when the condition declares no anchor. Single and
#: complete linkage produce different classes over identical rows and therefore different
#: verdicts, so the engine picking one would be deciding the rule.
_LINKAGES = ("single", "complete")

#: Token orderings. `as_written` is the identity, which is why it is the one sub-key that may
#: be absent: sorting COLLAPSES more values together, so leaving it out is the narrow reading.
_TOKEN_ORDERS = ("as_written", "sorted")


class FormError(Exception):
    """A form the engine cannot apply, carrying the sentence the condition reports.

    Raised rather than returned because a malformed form and an unresolvable value are
    different answers: the first is an authoring error that makes every reading under it
    wrong, the second is one value the form could not read.
    """


def _op_of(entry: Any) -> Tuple[str, Any]:
    """One pipeline step -> ``(op, parameter)``.

    A step is a single-key mapping. Two operations in one step is an order nobody authored,
    so it is refused rather than resolved by dict order.
    """
    if not isinstance(entry, dict) or len(entry) != 1:
        got = sorted(entry) if isinstance(entry, dict) else type(entry).__name__
        raise FormError(f"each step declares exactly one operation, got {got}")
    op, param = next(iter(entry.items()))
    return str(op), param


def _op_int(op: str, param: Any) -> int:
    """A step's positive-integer parameter. A bool is not an integer here: ``prefix: true``
    is a declaration that lost its number, and ``int(True)`` would silently make it 1.
    """
    if isinstance(param, bool) or not isinstance(param, int) or param < 1:
        raise FormError(f"`{op}` takes {_OP_PARAMS.get(op, 'a positive integer')}")
    return param


def _apply_tokens(value: str, param: Any) -> str:
    """Split, optionally reorder, optionally take a fixed count, and rejoin."""
    if not isinstance(param, dict):
        raise FormError(f"`tokens` takes {_OP_PARAMS['tokens']}")
    sep = param.get("split")
    if not isinstance(sep, str) or not sep:
        raise FormError("`tokens` requires a non-empty `split`")
    parts = [p for p in value.split(sep) if p]
    order = str(param.get("order", "") or "").strip().lower() or "as_written"
    if order not in _TOKEN_ORDERS:
        raise FormError(f"`tokens` order takes {' or '.join(_TOKEN_ORDERS)}")
    if order == "sorted":
        parts = sorted(parts)
    if param.get("take") is not None:
        parts = parts[: _op_int("take", param.get("take"))]
    return sep.join(parts)


def _apply_projection(value: str, op: str, param: Any) -> str:
    """One projection step. Substitutions in `map` are matched against the value AS PROJECTED
    SO FAR, so a pack controls their case by ordering `case:` before them rather than by a
    flag here — which is what composing the pipeline is for."""
    if op == "case":
        mode = str(param or "").strip().lower()
        if mode == "upper":
            return value.upper()
        if mode == "lower":
            return value.lower()
        if mode == "fold":
            return value.casefold()
        raise FormError(f"`case` takes {_OP_PARAMS['case']}")
    if op == "keep":
        cls = str(param or "").strip().lower()
        if cls == "alnum":
            return "".join(c for c in value if c.isalnum())
        if cls == "alpha":
            return "".join(c for c in value if c.isalpha())
        if cls == "digits":
            return "".join(c for c in value if c.isdigit())
        raise FormError(f"`keep` takes {_OP_PARAMS['keep']}")
    if op == "strip":
        if not isinstance(param, str) or not param:
            raise FormError(f"`strip` takes {_OP_PARAMS['strip']}")
        return value.strip(param)
    if op == "prefix":
        return value[: _op_int(op, param)]
    if op == "suffix":
        return value[-_op_int(op, param) :]
    if op == "collapse_repeats":
        if param is not True:
            raise FormError("`collapse_repeats` takes true")
        return "".join(c for i, c in enumerate(value) if i == 0 or c != value[i - 1])
    if op == "tokens":
        return _apply_tokens(value, param)
    if op == "map":
        if not isinstance(param, dict) or not param:
            raise FormError(f"`map` takes {_OP_PARAMS['map']}")
        for k, v in param.items():
            if str(k) == value:
                return str(v)
        return value
    raise FormError(f"`{op}` is not one of the projection operations")


def _canonical_key(value: Any, form: Optional[Dict[str, Any]]) -> Optional[str]:
    """``value`` reduced to the key its form compares on; ``None`` when UNRESOLVABLE.

    No form is the incumbent reading (:func:`_norm_identifier`), so every pack that declares
    none behaves exactly as it did. Unresolvable is not the empty string: a form that reduces
    a value to nothing would otherwise make it equivalent to every other value it could not
    read, fabricating a class out of the form's own failures. Same rule as an absent
    `encoded_fields` part being omitted rather than `""`.
    """
    key = str(value if value is not None else "")
    if not form:
        return _norm_identifier(key) or None
    steps = form.get("project") or []
    if not isinstance(steps, list):
        raise FormError("`project` is an ordered list of single-operation steps")
    for entry in steps:
        op, param = _op_of(entry)
        if op not in _PROJECTION_OPS:
            raise FormError(f"`{op}` is not one of the projection operations")
        key = _apply_projection(key, op, param)
    floor = form.get("min_length")
    if floor is not None and len(key) < _op_int("min_length", floor):
        return None
    return key or None


def _edit_distance(a: str, b: str, cap: int) -> int:
    """Levenshtein distance, abandoned once it exceeds ``cap``.

    The answer is only ever compared against a declared bound, so a full table over two long
    values buys nothing; ``cap + 1`` means "further than the bound" and nothing more.
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _longest_shared_run(a: str, b: str) -> int:
    """The longest run of characters the two values share, in order and contiguously."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else 0)
            best = max(best, cur[j])
        prev = cur
    return best


def _pairwise_match(a: str, b: str, compare: Any) -> bool:
    """Whether two already-projected keys satisfy a form's comparison operation."""
    op, param = _op_of(compare)
    if op not in _COMPARISON_OPS:
        raise FormError(f"`{op}` is not one of the comparison operations")
    if op == "exact":
        if param is not True:
            raise FormError("`exact` takes true")
        return a == b
    if op == "contains":
        if param is not True:
            raise FormError("`contains` takes true")
        return a in b or b in a
    n = _op_int(op, param)
    if op == "shared_prefix":
        # Both values must HAVE n characters, or a 2-character value satisfies a 3-character
        # prefix against everything that starts the same way — the over-collapse this
        # vocabulary's `min_length` exists to make visible, arriving through the comparison.
        return len(a) >= n and len(b) >= n and a[:n] == b[:n]
    if op == "shared_suffix":
        return len(a) >= n and len(b) >= n and a[-n:] == b[-n:]
    if op == "edit_distance":
        return _edit_distance(a, b, n) <= n
    return _longest_shared_run(a, b) >= n


def _form_compare(form: Optional[Dict[str, Any]]) -> Optional[Any]:
    """A form's comparison step, or ``None`` for a projection-only form."""
    if not form:
        return None
    compare = form.get("compare")
    return compare if compare else None


def _equivalence_classes(
    values: List[Any], form: Optional[Dict[str, Any]]
) -> Tuple[List[Tuple[str, List[str]]], List[str], int]:
    """Group ``values`` under ``form`` -> ``(classes, unresolvable, comparisons)``.

    ``classes`` is ``(key, member values)`` ordered by size descending then key ascending;
    ``unresolvable`` holds the raw values the form could not read, which are excluded and
    reported rather than collapsed together. ``comparisons`` is the pairwise work performed —
    reported rather than silently capped, because a cap that truncates a class is the
    fabricated-finding shape one layer down.

    A projection-only form groups by key. A pairwise form clusters under the form's declared
    linkage, over the projected keys and in sorted key order, so the classes are the same on
    two evaluations of one row set.
    """
    keyed: List[Tuple[str, str]] = []
    unresolvable: List[str] = []
    for v in values:
        raw = str(v if v is not None else "").strip()
        if not raw:
            continue
        key = _canonical_key(raw, form)
        if key is None:
            unresolvable.append(raw)
        else:
            keyed.append((key, raw))
    compare = _form_compare(form)
    if compare is None:
        groups: Dict[str, List[str]] = {}
        for key, raw in keyed:
            members = groups.setdefault(key, [])
            if raw not in members:
                members.append(raw)
        classes = [(k, sorted(v)) for k, v in groups.items()]
        return _rank_classes(classes), unresolvable, 0
    linkage = str((form or {}).get("linkage", "") or "").strip().lower()
    if linkage not in _LINKAGES:
        raise FormError(
            "a form declaring `compare` needs `linkage` "
            f"({' or '.join(_LINKAGES)}) to group values without an anchor"
        )
    items = sorted(keyed)
    clusters: List[List[Tuple[str, str]]] = []
    comparisons = 0
    if linkage == "single":
        # Connected components of the match graph: the definition of single linkage, and the
        # reason it is not the greedy version below — a value matching members of two
        # clusters MERGES them, and a greedy pass would report the largest class one short.
        parent = list(range(len(items)))

        def _root(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                comparisons += 1
                if _pairwise_match(items[i][0], items[j][0], compare):
                    parent[_root(i)] = _root(j)
        merged: Dict[int, List[Tuple[str, str]]] = {}
        for i, item in enumerate(items):
            merged.setdefault(_root(i), []).append(item)
        clusters = list(merged.values())
    else:
        # Complete linkage over a match relation is a clique cover, and a maximum one is not
        # computable at this cost — so the rule is stated rather than approximated silently:
        # in sorted key order, each value joins the FIRST existing class every member of which
        # it matches, and otherwise starts its own. Deterministic, which is what the verdict
        # needs; not minimal, which nothing here claims.
        for item in items:
            for cluster in clusters:
                fits = True
                for member in cluster:
                    comparisons += 1
                    if not _pairwise_match(item[0], member[0], compare):
                        fits = False
                        break
                if fits:
                    cluster.append(item)
                    break
            else:
                clusters.append([item])
    classes = []
    for cluster in clusters:
        members = sorted({raw for _, raw in cluster})
        classes.append((min(k for k, _ in cluster), members))
    return _rank_classes(classes), unresolvable, comparisons


def _resolve_form(cond: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    """The equivalence form a condition names at ``key``, or ``None`` for the incumbent
    reading (:func:`_norm_identifier`).

    A name the pack does not declare RAISES rather than falling back, because an unknown form
    is not "no form": comparing on the incumbent key would answer a different question under
    the authority of a declaration nobody wrote.
    """
    name = str(cond.get(key, "") or "").strip()
    if not name:
        return None
    forms = cond.get("_equivalence_forms") or {}
    form = forms.get(name) if isinstance(forms, dict) else None
    if not isinstance(form, dict):
        raise FormError(f"no pack file declares the form {name!r}")
    return form


def _projection_form(cond: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    """As :func:`_resolve_form`, but for the seams that dedup on a KEY.

    A projection is transitive, so "how many distinct values" has one answer. A `compare`
    relation is not, so the same question becomes "how many classes under which linkage" — a
    different question with a different answer, and applying only the form's projection half
    would answer it while reporting the form's name. Refused here and named as an ERROR by
    `pack_validate`, since `value_equivalence` is the kind that asks it.
    """
    form = _resolve_form(cond, key)
    if form and _form_compare(form):
        raise FormError(
            f"the {str(cond.get(key, '')).strip()!r} form declares `compare`, which is not "
            "transitive and so has no distinct count of its own — ask it through a "
            "`value_equivalence` condition, which declares the linkage"
        )
    return form


def _anchored_matches(
    values: List[Any], anchor_keys: List[str], form: Optional[Dict[str, Any]]
) -> Tuple[List[str], List[str], int]:
    """Values equivalent to any anchor key -> ``(matches, unresolvable, comparisons)``.

    An anchored question needs no ``linkage``: there is nothing to cluster, so the relation
    not being transitive costs nothing and every value is compared against a fixed side.
    """
    compare = _form_compare(form)
    matches: List[str] = []
    unresolvable: List[str] = []
    comparisons = 0
    for v in values:
        raw = str(v if v is not None else "").strip()
        if not raw:
            continue
        key = _canonical_key(raw, form)
        if key is None:
            unresolvable.append(raw)
            continue
        for ak in anchor_keys:
            if compare is None:
                hit = key == ak
            else:
                comparisons += 1
                hit = _pairwise_match(key, ak, compare)
            if hit:
                if raw not in matches:
                    matches.append(raw)
                break
    return matches, unresolvable, comparisons


def _rank_classes(classes: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
    """Largest class first, ties on the key ascending — the winner's identity is the finding,
    so it may not depend on dict or row order."""
    return sorted(classes, key=lambda kv: (-len(kv[1]), kv[0]))


def _form_consistent_key(
    key: Any,
    pack: Any,
    logs: Dict[str, List[Dict]],
    real_field: Any,
    field_values: Any,
    log: Any,
) -> Dict[str, str]:
    """Re-bind a correlation key so all sources are compared on the same surface form.

    Returns the corrected ``{source: field}`` map, or ``{}`` to leave the key unchanged.
    The form is chosen from the rows: scored as ``(sources bound, sources sharing a value)``
    and accepted only when it beats the incumbent. Entities with fewer than two declared
    forms are skipped. Independent of ``co_identity.prefer``, which governs display only.
    """
    etype = str(getattr(key, "entity_hint", "") or "")
    forms = [str(f.name) for f in (pack.value_forms_for(etype) or []) if str(f.name)]
    if len(forms) < 2:
        return {}
    incumbent = dict(getattr(key, "sources", {}) or {})

    def score(bound: Dict[str, str]) -> tuple:
        """(sources bound, sources sharing a value with another source).

        Joinability uses ``_identifiers_match``, not raw set intersection. Scored on raw
        equality, the correct form ties with the mixed binding; a tie keeps the incumbent.
        """
        vals = {s: field_values(logs.get(s) or [], f) for s, f in bound.items()}

        def relates(a: Set[str], b: Set[str]) -> bool:
            return any(_identifiers_match(x, y) for x in a for y in b)

        joined = sum(
            1
            for s, v in vals.items()
            if v and any(o != s and relates(v, ov) for o, ov in vals.items())
        )
        return (len(bound), joined)

    best_form, best_bound, best_score = "", incumbent, score(incumbent)
    for form in forms:
        bound = {}
        for src in incumbent:
            f = real_field(src, etype, "", form)
            if f:
                bound[src] = f
        if len(bound) < 2:
            continue
        s = score(bound)
        # Strictly better on (breadth, connectedness); a tie keeps the incumbent.
        if s > best_score:
            best_form, best_bound, best_score = form, bound, s
    if not best_form or best_bound == incumbent:
        return {}
    log.info(
        "Re-bound correlation key '%s' onto the single surface form '%s' (%s): the "
        "resolved binding mixed forms, so the join compared unlike vocabularies. "
        "%d source(s) bound, %d sharing values (was %d/%d).",
        etype,
        best_form,
        ", ".join(f"{s}={f}" for s, f in sorted(best_bound.items())),
        best_score[0],
        best_score[1],
        score(incumbent)[0],
        score(incumbent)[1],
    )
    return best_bound


def _merge_co_identified_subjects(
    subjects: List[str],
    forms: Dict[str, str],
    co_identity: Dict[str, Any],
    logs: Dict[str, List[Dict]],
    entity_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Collapse subject values that the evidence shows are one identity.

    Returns ``(subjects, notes, merged)``. ``merged`` maps each surviving value to the
    values absorbed into it; the caller must keep selecting the dropped values' rows.

    A merge is licensed only by a retrieved row binding both values (and agreeing on each
    ``via`` type where declared). ``forms`` maps subject value to form name; values of
    undeclared forms are never folded.
    """
    named = [str(f) for f in (co_identity.get("forms") or []) if str(f).strip()]
    if len(named) < 2 or len(subjects) < 2:
        return subjects, [], {}
    via = [str(v) for v in (co_identity.get("via") or []) if str(v).strip()]
    prefer = str(co_identity.get("prefer", "") or "")

    # Candidates: declared forms only. An unclassified value cannot participate.
    by_form: Dict[str, List[str]] = {}
    for sv in subjects:
        form = forms.get(sv, "")
        if form in named:
            by_form.setdefault(form, []).append(sv)
    if len(by_form) < 2:
        return subjects, [], {}

    # The entity map is per source; a row is checked against its own source's mapping.
    def _agrees(row: Dict, source: str) -> Tuple[bool, List[str]]:
        """Does ``row`` carry an agreeing value for every declared ``via`` type?"""
        if not via:
            return True, []
        mapping = (entity_map or {}).get(source, {}) or {}
        seen: List[str] = []
        for etype in via:
            field = mapping.get(etype, "")
            vals = (
                [str(v).strip() for v in resolve_path(row, field) if str(v).strip()]
                if field
                else []
            )
            if not vals:
                return False, []
            # Exactly one value per type; two different values would make the row ambiguous.
            if len(set(vals)) != 1:
                return False, []
            seen.append(f"{etype}={vals[0]}")
        return True, seen

    # Pair up across forms, greedily and deterministically: the first form's values in the
    # order the alert gave them, each matched against the other forms' remaining values.
    order = [f for f in named if f in by_form]
    survivors = list(subjects)
    notes: List[str] = []
    merged: Dict[str, List[str]] = {}
    for base_form in order:
        for base in list(by_form.get(base_form, [])):
            if base not in survivors:
                continue
            for other_form in order:
                if other_form == base_form:
                    continue
                for other in list(by_form.get(other_form, [])):
                    if other not in survivors or other == base:
                        continue
                    licence = ""
                    for source, rows in (logs or {}).items():
                        for row in rows or []:
                            if not isinstance(row, dict):
                                continue
                            vals = _row_values(row)
                            # Exact equality, not prefix matching: merging uses the row as
                            # proof that two values name one identity, so prefix tolerance
                            # would merge distinct identifiers sharing a short prefix.
                            if not (
                                any(base == v for v in vals)
                                and any(other == v for v in vals)
                            ):
                                continue
                            ok, seen = _agrees(row, source)
                            if ok:
                                licence = f"{source}" + (
                                    f" ({', '.join(seen)})" if seen else ""
                                )
                                break
                        if licence:
                            break
                    if not licence:
                        continue
                    # Prefer the declared form's value so the report heading is stable.
                    keep, drop = base, other
                    if prefer and forms.get(other, "") == prefer:
                        keep, drop = other, base
                    survivors = [s for s in survivors if s != drop]
                    merged.setdefault(keep, []).append(drop)
                    # A dropped value passes its absorbed aliases forward.
                    for inherited in merged.pop(drop, []):
                        if inherited not in merged[keep]:
                            merged[keep].append(inherited)
                    notes.append(
                        f"co_identity={keep} and {drop} are two declared forms of the same "
                        f"identity, adjudicated as one subject — evidenced by a row in "
                        f"'{licence}' binding both"
                    )
                    if keep != base:
                        break
    if merged:
        logger.info(
            "Co-identity: merged %d subject value(s) into %d — %s",
            sum(len(v) for v in merged.values()),
            len(survivors),
            "; ".join(f"{k} <- {', '.join(v)}" for k, v in merged.items()),
        )
    return survivors, notes, {k: list(v) for k, v in merged.items()}


# Acting-combination cap per subject for per-identity conditions. Truncating would let the engine
# claim unanimity over a subset; instead the condition is left unresolved above this count.
_MAX_ACTING_TUPLES = 25


def _first_resolving_values(node: Any, candidates: List[str]) -> List[str]:
    """Values of the first candidate path that resolves against ``node``.

    A candidate list handles one logical field under several spellings; taking the first
    that resolves avoids double-counting when two spellings both exist on a node.
    Uses :func:`resolve_path` so underscore-flattened aliases are honoured.
    """
    for candidate in candidates:
        vals = [
            str(v).strip()
            for v in resolve_path(node, candidate)
            if str(v or "").strip()
        ]
        if vals:
            return list(dict.fromkeys(vals))
    return []


def _subject_companion_index(
    decl: Dict[str, Any],
    rows_by_logical: Dict[str, List[Dict]],
    subject_entity: str,
) -> Tuple[List[str], Dict[str, Dict[str, List[str]]], Dict[str, List[Dict[str, str]]], str]:
    """The pure walk behind ``subject_discovery``: which subject values the rows carry, and
    which other entity types sit beside each of them. See ``_named_subject_companions``.

    Returns ``(subjects, per_subject_entities, per_subject_tuples, logical_source)`` with no
    cap, notes or logging; those belong to the caller.

    ``per_subject_tuples`` groups companions as observed together, so a condition can ask
    about each actual combination rather than the cross-product of per-type unions.
    """
    if not isinstance(decl, dict) or not decl:
        return [], {}, {}, ""
    logical = str(decl.get("source", "") or "")
    rows = rows_by_logical.get(logical) or []
    subject_paths = [
        str(p).strip() for p in (decl.get("subject") or []) if str(p or "").strip()
    ]
    if not rows or not subject_paths:
        return [], {}, {}, logical
    node_segments = [s for s in str(decl.get("path", "") or "").split(".") if s]
    entity_paths = {
        str(k).strip(): [str(p).strip() for p in (v or []) if str(p or "").strip()]
        for k, v in (decl.get("entities") or {}).items()
        if str(k or "").strip()
    }

    subjects: List[str] = []
    per_subject: Dict[str, Dict[str, List[str]]] = {}
    per_subject_tuples: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        nodes = _resolve_nodes(row, node_segments) if node_segments else [row]
        # Expand each node before reading companions so relative paths resolve per element,
        # not across the whole list at once (which would union all siblings' co-located values).
        elements: List[Any] = []
        for node in nodes:
            node = _maybe_json(node)
            if isinstance(node, list):
                elements.extend(node)
            else:
                elements.append(node)
        for node in elements:
            for value in _first_resolving_values(node, subject_paths):
                if value not in per_subject:
                    subjects.append(value)
                    per_subject[value] = {}
                    per_subject_tuples[value] = []
                observed: Dict[str, str] = {}
                for ent_type, candidates in entity_paths.items():
                    if str(ent_type).lower() == str(subject_entity).lower():
                        # The subject's own type is filled from the subject value itself by
                        # the per-subject loop; a second binding here would compete with it.
                        continue
                    found = _first_resolving_values(node, candidates)
                    if not found:
                        continue
                    kept = per_subject[value].setdefault(ent_type, [])
                    # The same subject value can be reached via two elements; union, deduped.
                    for v in found:
                        if v not in kept:
                            kept.append(v)
                    # A type carried twice by the same element is a union; it cannot
                    # state which value paired with the others and is omitted from the tuple.
                    if len(found) == 1:
                        observed[ent_type] = found[0]
                if observed:
                    seen = per_subject_tuples.setdefault(value, [])
                    if observed not in seen:
                        seen.append(observed)
    return subjects, per_subject, per_subject_tuples, logical


def _named_subject_companions(
    decl: Dict[str, Any],
    rows_by_logical: Dict[str, List[Dict]],
    subject_entity: str,
    subjects: List[str],
    aliases: Dict[str, List[str]],
    named_types: Optional[Set[str]] = None,
) -> Tuple[
    Dict[str, Dict[str, List[str]]], Dict[str, List[Dict[str, str]]], List[str]
]:
    """Co-located entity values for a subject the incident named, filled only by unanimous rows.

    A type fills only when every row of the named subject carries the same value for it.
    Disagreement is reported in a note rather than silently omitted. Observed per-element
    combinations ride back as ``per_subject_tuples`` for conditions that quantify per identity.
    ``named_types`` are skipped so extracted values are not overwritten by the rows.
    Returns ``(per_subject_entities, per_subject_tuples, notes)``.
    """
    if not subjects or not any(str(s).strip() for s in subjects):
        return {}, {}, []
    skip = {str(t).strip().lower() for t in (named_types or set()) if str(t).strip()}
    found, per_subject, per_element, logical = _subject_companion_index(
        decl, rows_by_logical, subject_entity
    )
    if not found:
        return {}, {}, []
    # Named value -> the row value it matches. Exact first, then stripped/casefolded: the
    # incident's spelling of an identifier and the column's are routinely not byte-equal, and a
    # lookup that misses reads exactly like a subject the rows do not carry.
    index: Dict[str, str] = {}
    for value in found:
        index.setdefault(str(value), str(value))
        index.setdefault(str(value).strip().casefold(), str(value))
    out: Dict[str, Dict[str, List[str]]] = {}
    tuples: Dict[str, List[Dict[str, str]]] = {}
    notes: List[str] = []
    for sv in subjects:
        sv = str(sv or "")
        if not sv.strip():
            continue
        # Aliases ride along for the reason they ride along everywhere else: a co-identified
        # subject's rows may carry only the folded-away form, and dropping it goes silent on
        # exactly the evidence the merge proved belongs to this actor.
        candidates = [sv] + [str(a) for a in (aliases.get(sv) or []) if str(a).strip()]
        companions: Dict[str, List[str]] = {}
        observed: List[Dict[str, str]] = []
        for cand in candidates:
            hit = index.get(cand) or index.get(cand.strip().casefold())
            if not hit:
                continue
            for ent_type, values in (per_subject.get(hit) or {}).items():
                if str(ent_type).strip().lower() in skip:
                    continue
                kept = companions.setdefault(ent_type, [])
                for v in values:
                    if v not in kept:
                        kept.append(v)
            for combo in per_element.get(hit) or []:
                shaped = {
                    t: v
                    for t, v in combo.items()
                    if str(t).strip().lower() not in skip
                }
                if shaped and shaped not in observed:
                    observed.append(shaped)
        if observed and len(observed) <= _MAX_ACTING_TUPLES:
            tuples[sv] = observed
        elif observed:
            # Bounded by withholding, not by truncating: a truncated list lets the condition
            # claim unanimity over a subset. Nothing back leaves it unresolved.
            logger.warning(
                "subject_companions: %d acting combination(s) on one subject exceeds the "
                "bound of %d — no per-identity condition can be answered for it.",
                len(observed),
                _MAX_ACTING_TUPLES,
            )
        if not companions:
            continue
        agreed = {t: v for t, v in companions.items() if len(v) == 1}
        split = {t: len(v) for t, v in companions.items() if len(v) > 1}
        if agreed:
            out[sv] = {t: list(v) for t, v in agreed.items()}
            notes.append(
                f"subject_companions={sv} the incident named this subject but not "
                + ", ".join(f"{t}={v[0]}" for t, v in sorted(agreed.items()))
                + f"; each was read from '{logical}' because every row of this subject's "
                f"agrees on it"
            )
        if split:
            note = (
                f"subject_companions={sv} this subject's rows in '{logical}' do NOT agree on "
                + ", ".join(f"{t} ({n} distinct values)" for t, n in sorted(split.items()))
                + ", so no single value can answer a question about THE acting one and any "
                "condition needing it is left unresolved rather than answered from a union"
            )
            if len(observed) > 1:
                note += (
                    f"; {len(observed)} distinct combination(s) were observed together on a "
                    f"row, which a condition declaring `per_acting_identity` asks about one "
                    f"at a time"
                )
            if len(observed) > _MAX_ACTING_TUPLES:
                note += (
                    f" — more than the bound of {_MAX_ACTING_TUPLES}, so such a condition is "
                    f"left unresolved rather than answered over a subset of them"
                )
            notes.append(note)
    return out, tuples, notes


def _discovered_subjects(
    decl: Dict[str, Any],
    rows_by_logical: Dict[str, List[Dict]],
    subject_entity: str,
) -> Tuple[
    List[str],
    Dict[str, Dict[str, List[str]]],
    Dict[str, List[Dict[str, str]]],
    List[str],
]:
    """Subjects read from retrieved rows, where the pack declares where they live.

    Consulted only when extraction produced no subject. ``per_subject_entities`` maps each
    subject value to co-located entity values found in its own element (not a union across
    all elements). Returns ``(subjects, per_subject_entities, per_subject_tuples, notes)``.

    Declaration shape::

        subject_discovery:
          source: <logical source name>
          path: <dotted path to the repeated node>   # optional; omitted = the row itself
          subject: [<path relative to path>, ...]    # candidate list, first that resolves
          entities: {<entity type>: [<relative path>, ...]}   # optional, co-located
          max_subjects: <int>                        # optional bound
    """
    subjects, per_subject, per_subject_tuples, logical = _subject_companion_index(
        decl, rows_by_logical, subject_entity
    )
    if not subjects:
        return [], {}, {}, []

    notes: List[str] = []
    cap = decl.get("max_subjects")
    try:
        cap_n = int(cap) if cap is not None else 0
    except (TypeError, ValueError):
        cap_n = 0
    if cap_n > 0 and len(subjects) > cap_n:
        dropped = len(subjects) - cap_n
        logger.warning(
            "subject_discovery: '%s' yielded %d subject(s), capped at %d — %d not adjudicated.",
            logical,
            len(subjects),
            cap_n,
            dropped,
        )
        notes.append(
            f"subject_cap={cap_n} of {len(subjects)} identities found in '{logical}' are "
            f"adjudicated; {dropped} more were retrieved and are NOT covered by this verdict"
        )
        subjects = subjects[:cap_n]
        per_subject = {k: v for k, v in per_subject.items() if k in set(subjects)}
        per_subject_tuples = {
            k: v for k, v in per_subject_tuples.items() if k in set(subjects)
        }

    if subjects:
        logger.info(
            "subject_discovery: %d subject(s) read from '%s' (%s) — the incident named none.",
            len(subjects),
            logical,
            ", ".join(subjects[:10]),
        )
        for value in subjects:
            companions = per_subject.get(value) or {}
            detail = (
                "; ".join(
                    f"{t}={', '.join(vs)}" for t, vs in sorted(companions.items())
                )
                or "no co-located values"
            )
            notes.append(
                f"subject_discovered={value} was not named by the incident and was read from "
                f"'{logical}'; the values adjudicated with it come from its own record "
                f"({detail})"
            )
    return subjects, per_subject, per_subject_tuples, notes


def _render_acting_identity(combo: Dict[str, str]) -> str:
    """One observed acting identity, as an operator can look it up in the evidence render."""
    return "+".join(f"{k}={v}" for k, v in sorted((combo or {}).items()))


def _quantify_over_members(counts: Dict[str, int]) -> str:
    """``pass`` | ``fail`` | ``unknown`` over a condition asked of each set member.

    Any pass wins (existential). All-fail yields fail (universal). Otherwise unknown, because
    an unresolved member might have passed. Conservative in both polarities: never strengthens
    a verdict. Used for both acting-identity rollups and per-record rollups.
    """
    if counts.get("pass"):
        return "pass"
    if counts.get("fail") and not counts.get("unknown"):
        return "fail"
    return "unknown"


def _pair_units(
    cond: Dict[str, Any], src_rows: Dict[str, List[Dict]]
) -> Tuple[str, List[Dict], str]:
    """The records a two-sided condition is compared within, or the reason it cannot be.

    Returns ``(logical_source, units, refusal)``. A non-empty ``refusal`` means the declaration
    cannot be honoured on these sides; the caller falls back to the pooled reading and reports
    the reason, because a declaration that silently does nothing is the defect it was written to prevent.

    Honoured only when both sides agree on source, ``records:`` path and ``where:`` clauses.
    Sides selecting different records cannot be paired; they measure different things.
    """
    a = cond.get("left") or cond.get("start") or {}
    b = cond.get("right") or cond.get("end") or {}
    if not isinstance(a, dict) or not isinstance(b, dict):
        return "", [], "a side is not a mapping"
    src = str(a.get("source") or "").strip()
    if not src or src != str(b.get("source") or "").strip():
        return "", [], "the two sides name different sources, so no one record carries both"
    if str(a.get("records") or "").strip() != str(b.get("records") or "").strip():
        return src, [], "the two sides declare different `records:` paths"
    if (a.get("where") or []) != (b.get("where") or []):
        return src, [], "the two sides select different records through `where:`"
    # Either side computes the same unit list, `records:` and `where:` already applied.
    return src, _side_rows(src_rows, a), ""


def _rollup_paired_records(
    per: List[Tuple[int, ConditionCheck]],
    cond_id: str,
) -> ConditionCheck:
    """Roll up a two-sided condition asked within each record via ``_quantify_over_members``.

    The per-record value sets are subsets of the pooled ones, so a per-record pass was
    always a pooled pass. Only pooled passes that rested on a cross-record pair can change.
    """
    per = [(i, c) for i, c in (per or []) if c is not None]
    if not per:
        raise ValueError(f"condition '{cond_id}': no record to pair the two sides within")
    buckets: Dict[str, List[Tuple[int, ConditionCheck]]] = {
        "pass": [],
        "fail": [],
        "unknown": [],
    }
    for idx, chk in per:
        buckets.setdefault(str(chk.result), []).append((idx, chk))
    winner = _quantify_over_members({k: len(v) for k, v in buckets.items()})
    why = {
        "pass": "a PASS on one of them holds for the set",
        "fail": "every one of them did",
    }.get(
        winner,
        f"{len(buckets['unknown'])} of them carry only one side and a FAIL here would have "
        f"to hold for all {len(per)}",
    )
    shown = ", ".join(
        f"{name.upper()} on {len(items)}" for name, items in buckets.items() if items
    )
    rep = buckets[winner][0][1]
    note = (
        f"compared WITHIN each of the {len(per)} record(s) that could carry both sides, never "
        f"across them — {shown} — and reported as {winner.upper()} because {why}"
    )
    detail = f"{rep.detail}; {note}" if rep.detail else note
    return rep.model_copy(update={"detail": detail})


def _rollup_acting_identities(
    per: List[Tuple[Dict[str, str], ConditionCheck]],
    cond_id: str,
) -> ConditionCheck:
    """Roll up a condition asked of each acting identity via ``_quantify_over_members``."""
    per = [(t, c) for t, c in (per or []) if c is not None]
    if not per:
        raise ValueError(f"condition '{cond_id}': no acting identity to roll up")
    buckets: Dict[str, List[Tuple[Dict[str, str], ConditionCheck]]] = {
        "pass": [],
        "fail": [],
        "unknown": [],
    }
    for combo, chk in per:
        buckets.setdefault(str(chk.result), []).append((combo, chk))
    winner = _quantify_over_members({k: len(v) for k, v in buckets.items()})
    why = {
        "pass": "a PASS on one of them holds for the set",
        "fail": "every one of them did",
    }.get(
        winner,
        f"{len(buckets['unknown'])} of them could not be resolved and a FAIL here would "
        f"have to hold for all {len(per)}",
    )
    rep = buckets[winner][0][1]
    shown = "; ".join(
        f"{name.upper()} for "
        + ", ".join(_render_acting_identity(t) for t, _ in items[:6])
        + (f" and {len(items) - 6} more" if len(items) > 6 else "")
        for name, items in buckets.items()
        if items
    )
    note = (
        f"asked of each of the {len(per)} identities that acted on this subject — {shown} — "
        f"and reported as {winner.upper()} because {why}"
    )
    detail = f"{rep.detail}; {note}" if rep.detail else note
    return rep.model_copy(update={"detail": detail})


def _rows_for_subject(
    rows: List[Dict],
    subject_value: str,
    aliases: Optional[List[str]] = None,
) -> List[Dict]:
    """Rows whose scalar leaves contain the subject identifier.

    An empty ``subject_value`` returns all rows. ``aliases`` (from
    ``_merge_co_identified_subjects``) are included so a source keyed on the folded-away
    form still contributes its rows.
    """
    if not subject_value:
        return [r for r in rows if isinstance(r, dict)]
    wanted = [str(subject_value)] + [str(a) for a in (aliases or []) if str(a).strip()]
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        cells = _row_values(r)
        if any(sv == c or sv in c for sv in wanted for c in cells):
            out.append(r)
    return out


def _narrow_elements(value: Any, segments: List[str], keep) -> Any:
    """A copy of ``value`` with the repeated node at ``segments`` cut to elements ``keep``
    accepts. Copy-on-write down the path only; every other branch is the same object.

    Mirrors ``_resolve_nodes``' walk exactly so a narrowing that walks differently narrows a
    node nobody reads. Writes back under the key the row spells (``_dict_key``) to avoid
    leaving an un-narrowed sibling key in front of the narrowed one.

    A path naming one node rather than a repeated one is still narrowed: it survives or it becomes
    ``[]``, since a single element either is the wanted one or is somebody else's.
    """
    if not segments:
        node = _maybe_json(value)
        if isinstance(node, list):
            return [e for e in node if keep(e)]
        return node if keep(node) else []
    if isinstance(value, dict):
        for cut in range(len(segments), 0, -1):
            alias = "_".join(segments[:cut])
            found, child = _dict_get(value, alias)
            if found:
                out = dict(value)
                out[_dict_key(value, alias)] = _narrow_elements(
                    _maybe_json(child), segments[cut:], keep
                )
                return out
    node = _maybe_json(value)
    head, rest = segments[0], segments[1:]
    if isinstance(node, list):
        return [_narrow_elements(item, segments, keep) for item in node]
    if isinstance(node, dict):
        found, child = _dict_get(node, head)
        if found:
            out = dict(node)
            out[_dict_key(node, head)] = _narrow_elements(child, rest, keep)
            return out
    return value


def _subject_elements_only(
    rows: List[Dict],
    decl: Dict[str, Any],
    subject_values: List[str],
) -> Tuple[List[Dict], int, int]:
    """Rows with the ``subject_discovery`` node narrowed to this subject's own elements.

    Returns ``(rows, kept, total)``. ``(-1, -1)`` means the declaration could not be applied
    and rows are returned untouched; the caller reports the reason.
    """
    segments = [s for s in str((decl or {}).get("path") or "").split(".") if s]
    subject_paths = [
        str(p).strip() for p in ((decl or {}).get("subject") or []) if str(p or "").strip()
    ]
    wanted = {
        _norm_identifier(v) for v in (subject_values or []) if str(v or "").strip()
    }
    live = [r for r in rows if isinstance(r, dict)]
    if not (segments and subject_paths and wanted):
        return live, -1, -1
    counts = {"total": 0, "kept": 0}

    def _keep(element: Any) -> bool:
        counts["total"] += 1
        hit = any(
            _matches_any_identifier(v, wanted)
            for v in _first_resolving_values(element, subject_paths)
        )
        counts["kept"] += 1 if hit else 0
        return hit

    out = [_narrow_elements(r, segments, _keep) for r in live]
    return out, counts["kept"], counts["total"]


def _rows_matching(
    rows: List[Dict], match_spec: List[Dict[str, Any]]
) -> Tuple[List[Dict], List[str]]:
    """Rows matching every clause in ``match_spec`` (conjunction of per-field value sets).

    Each clause is ``{fields, values, normalize: identifier|exact}``. Clauses with empty
    ``values`` are reported as unresolved rather than dropped.
    Returns ``(matching_rows, unresolved_clause_labels)``.
    """
    matched = [r for r in rows if isinstance(r, dict)]
    unresolved: List[str] = []
    for clause in match_spec or []:
        if not isinstance(clause, dict):
            continue
        fields = [f for f in (clause.get("fields") or []) if f]
        values = [v for v in (clause.get("values") or []) if str(v).strip()]
        if not fields:
            continue
        if not values:
            unresolved.append(clause.get("label") or ",".join(fields))
            continue
        norm = str(clause.get("normalize", "exact")).lower()
        keep = []
        for r in matched:
            found = [v for f in fields for v in resolve_path(r, f)]
            if norm == "identifier":
                hit = any(_identifiers_match(a, b) for a in found for b in values)
            else:
                want = {str(v).strip().upper() for v in values}
                hit = any(str(a).strip().upper() in want for a in found)
            if hit:
                keep.append(r)
        matched = keep
    return matched, unresolved


def _collect(rows: List[Dict], field: str) -> List[Any]:
    """All scalar leaf values of ``field`` across ``rows`` (deduped-preserving order)."""
    out: List[Any] = []
    for r in rows:
        if isinstance(r, dict):
            out.extend(resolve_path(r, field))
    return out


def apply_where(rows: List[Dict], clauses: Any) -> List[Dict]:
    """Rows narrowed by ``[{field, any_of, match}]`` clauses (all applied).

    A clause with no field or no values is skipped rather than emptying the set.
    Public so the follow-up harvest can use the same vocabulary as condition ``where``.
    """
    out = [r for r in rows if isinstance(r, dict)]
    for clause in clauses or []:
        if not isinstance(clause, dict):
            continue
        field = str(clause.get("field") or "").strip()
        want = [
            str(v).strip().upper()
            for v in (clause.get("any_of") or [])
            if str(v).strip()
        ]
        if not field or not want:
            continue
        # Single-letter status codes demand equality; a substring test on 'V' hits every
        # value containing a V. Verbose record types use substring.
        exact = str(clause.get("match", "exact")).lower() == "exact"
        kept = []
        for r in out:
            found = [str(v).strip().upper() for v in resolve_path(r, field)]
            if any((v in want) if exact else any(w in v for w in want) for v in found):
                kept.append(r)
        out = kept
    return out


def _side_rows(src_rows: Dict[str, List[Dict]], side: Dict[str, Any]) -> List[Dict]:
    """Rows of a condition side's source, narrowed by its ``where`` clauses.

    ``records`` (optional) moves the unit of selection inside the row. Three shapes are
    handled: entries under the path (whole array), parallel leaf arrays (zipped by
    position), and one-row-per-entry (exploded). Shape 3 is admitted only for columns
    named after the records path to avoid falling back silently to the row-level reading.
    """
    rows = [
        r for r in src_rows.get(str(side.get("source", "")), []) if isinstance(r, dict)
    ]
    rec_path = str(side.get("records") or "").strip()
    if rec_path:
        # Databricks projections alias struct leaves with their full path to avoid last-segment
        # collisions. Strip that prefix so the ruleset names the leaf inside the record.
        prefix = rec_path.replace(".", "_") + "_"

        def _unprefixed(key: str) -> str:
            return key[len(prefix) :] if key.startswith(prefix) else key

        entries = []
        for r in rows:
            got = False
            for node in _resolve_nodes(r, rec_path.split(".")):
                node = _maybe_json(node)
                # A struct entry is the unit; a scalar leaf under the path is not a record
                # and is dropped rather than guessed at.
                if isinstance(node, dict):
                    entries.append(node)
                    got = True
                elif isinstance(node, list):
                    kids = [e for e in node if isinstance(e, dict)]
                    entries.extend(kids)
                    got = got or bool(kids)
            if got:
                continue
            # ---- shape 2: parallel leaf arrays, zipped by position. ----------------
            # Only when all arrays have the same length; ragged arrays cannot be aligned.
            cols = {}
            for k, v in r.items():
                v = _maybe_json(v)
                if (
                    isinstance(v, list)
                    and v
                    and not any(isinstance(e, (dict, list)) for e in v)
                ):
                    cols[_unprefixed(str(k))] = v
            if len(cols) >= 2 and len({len(v) for v in cols.values()}) == 1:
                n = len(next(iter(cols.values())))
                entries.extend({k: v[i] for k, v in cols.items()} for i in range(n))
                continue
            # Shape 3: the row is one entry, already exploded. Restricted to columns
            # the records path names, so unrelated sources yield no entries.
            flat = {
                _unprefixed(str(k)): _maybe_json(v)
                for k, v in r.items()
                if str(k).startswith(prefix)
            }
            if flat:
                entries.append(flat)
        rows = entries
    return apply_where(rows, side.get("where"))


def _resolve_nodes(value: Any, segments: List[str]) -> List[Any]:
    """Like ``_resolve_segments`` but returns the raw nodes (dicts/lists/scalars) at the
    path (needed to read key/value structures, e.g. a ``{key, value}`` pair list, where
    the pairing matters and a scalar-leaf collect would lose it). Descends JSON-strings
    and maps over lists like the scalar resolver.

    Mirrors ``resolve_path``'s **underscore-flattened alias** handling at the top level: a
    Databricks projection aliases a struct sub-path like ``pay.method`` to a flat column
    ``pay_method``, so when ``value`` is a row dict we try progressively-shorter underscore
    prefixes (``pay_method`` → then descend ``detail.data_map`` into it) before the plain
    segment walk. Without this a key/value scan silently returned nothing on aliased rows.
    """
    if not segments:
        return [value]
    # Top-level alias fallback (row dict with dotted-path collapsed to an underscore key).
    if isinstance(value, dict):
        for cut in range(len(segments), 0, -1):
            alias = "_".join(segments[:cut])
            found, child = _dict_get(value, alias)
            if found:
                return _resolve_nodes(_maybe_json(child), segments[cut:])
    value = _maybe_json(value)
    head, rest = segments[0], segments[1:]
    if isinstance(value, list):
        out: List[Any] = []
        for item in value:
            out.extend(_resolve_nodes(item, segments))
        return out
    if isinstance(value, dict):
        found, child = _dict_get(value, head)
        if found:
            return _resolve_nodes(child, rest)
    return []


def _collect_kv(
    rows: List[Dict],
    list_field: str,
    match_key: str,
    key_name: str = "key",
    value_name: str = "value",
) -> List[Any]:
    """Values from a list-of-``{key, value}`` structure where ``key == match_key``.

    Scans ``list_field`` (e.g. ``pay.method.detail.data_map``) across ``rows`` and returns
    each entry's ``value`` whose ``key`` matches (case-insensitive). Used by an indicator that
    reads a coded attribute out of such a list: ``{key:"<attr>", value:"<code>"}``."""
    want = str(match_key).strip().upper()
    out: List[Any] = []

    def _entries(node: Any) -> List[Any]:
        # A resolved node may be the {key,value} dict itself, or a containing list or
        # JSON-string array; normalize to a flat list of dict entries.
        node = _maybe_json(node)
        if isinstance(node, dict):
            return [node]
        if isinstance(node, list):
            flat: List[Any] = []
            for item in node:
                flat.extend(_entries(item))
            return flat
        return []

    for r in rows:
        if not isinstance(r, dict):
            continue
        for raw in _resolve_nodes(r, list_field.split(".")):
            for node in _entries(raw):
                _, k = _dict_get(node, key_name)
                if str(k).strip().upper() == want:
                    found, v = _dict_get(node, value_name)
                    if found and v is not None:
                        out.append(v)
    return out


def _first_present(rows: List[Dict], fields: List[str]) -> (str, List[Any]):
    """First field (of ``fields``) that resolves to a non-empty value across ``rows``.

    Blank strings are excluded: a SQL backend may return ``''`` for a missing column,
    which would shadow a populated fallback and make an empty result non-empty.
    """
    for f in fields or []:
        vals = [v for v in _collect(rows, f) if str(v).strip()]
        if vals:
            return f, vals
    return "", []


def _flag_leaves(row: Dict, path: str) -> List[Any]:
    """Leaves at ``path`` keeping booleans as well as scalars.

    ``resolve_path`` drops bools (``_scalar`` excludes them to avoid join-key confusion);
    this variant keeps them. Built on
    ``_resolve_nodes``, so it descends nested dicts, lists, JSON-string structs and the
    underscore-flattened alias the same way every other pack path does."""
    out: List[Any] = []
    for node in _resolve_nodes(row, path.split(".")):
        for item in node if isinstance(node, list) else [node]:
            if isinstance(item, bool) or _scalar(item):
                out.append(item)
    return out


def _collect_any(rows: List[Dict], field: str) -> List[Any]:
    """``_collect`` plus boolean leaves at the same path.

    Required when a check's vocabulary includes ``true``/``false``; ``resolve_path``
    drops booleans so a flag column would otherwise be unmatchable.
    """
    out = _collect(rows, field)
    for r in rows:
        if not isinstance(r, dict):
            continue
        for key in (field, field.replace(".", "_")):
            if key in r and isinstance(r[key], bool):
                out.append(r[key])
        out.extend(v for v in _flag_leaves(r, field) if isinstance(v, bool))
    return out


def _first_present_flag(rows: List[Dict], fields: List[str]) -> (str, List[Any]):
    """Like ``_first_present`` but includes boolean leaf values.

    ``resolve_path`` drops booleans; a flag field needs a resolver that keeps them.
    """
    for f in fields or []:
        found: List[Any] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            # Flat key or underscore alias (Databricks aliases dotted leaves that way).
            for key in (f, f.replace(".", "_")):
                if key in r and isinstance(r[key], bool):
                    found.append(r[key])
            # Non-bool leaves via the normal resolver (strings like "true"/"automated").
            found.extend(resolve_path(r, f))
            # Nested booleans: `resolve_path` drops them, so nested flag paths need
            # `_flag_leaves`. Booleans only, so string leaves already collected are not doubled.
            found.extend(v for v in _flag_leaves(r, f) if isinstance(v, bool))
        if found:
            return f, found

    # Alias-tolerant fallback: the generated SQL aliases a dotted struct leaf to a
    # flat name that varies run-to-run, so match any boolean column containing the
    # last path segment as a token.
    tokens = {
        seg.split(".")[-1].strip().lower()
        for seg in (fields or [])
        if seg.split(".")[-1].strip()
    }
    if tokens:
        for r in rows:
            if not isinstance(r, dict):
                continue
            for key, val in r.items():
                if not isinstance(val, bool):
                    continue
                kl = str(key).lower()
                if any(tok in kl for tok in tokens):
                    return key, [val]  # first matching boolean column wins
    return "", []


def _declared_bound(cond: Dict[str, Any], key: str = "max") -> Optional[int]:
    """The condition's declared numeric bound; ``None`` when absent or unreadable.

    An absent bound yields ``unknown`` at evaluation time, not a fabricated default.
    A non-integer value is treated as absent rather than raising.
    """
    raw = cond.get(key, None)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _declared_number(cond: Dict[str, Any], key: str) -> Optional[float]:
    """The condition's declared bound as a real number; ``None`` when absent or unreadable.

    Separate from ``_declared_bound`` rather than a widening of it: that one coerces with
    ``int()``, so a fractional bound becomes ``0`` — a bound every count satisfies. Its
    callers are counting and interval kinds whose bound is a whole number by construction,
    and a non-integer there is refused at authoring time (``pack_validate._BOUNDED_KINDS``).
    """
    raw = cond.get(key, None)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _fmt_num(v: float) -> str:
    """A number rendered for a report line: no trailing ``.0`` on a whole value."""
    return str(int(v)) if float(v).is_integer() else f"{v:.4g}"


#: Comparison operators a `numeric_compare` may declare, over the same set as `TransformStep`.
#: Split by which way the predicate's truth survives a change in the value, because that is
#: what decides whether a comparison against a TRUNCATED read still holds (see `_holds_when`).
_INCREASING_OPS = (">", ">=")  # true stays true as the value rises
_DECREASING_OPS = ("<", "<=")  # true stays true as the value falls
_COMPARE_OPS = _INCREASING_OPS + _DECREASING_OPS + ("==",)


def _compare(value: float, op: str, bound: float) -> Optional[bool]:
    """``value <op> bound``; ``None`` for an operator the engine does not implement."""
    if op == ">":
        return value > bound
    if op == ">=":
        return value >= bound
    if op == "<":
        return value < bound
    if op == "<=":
        return value <= bound
    if op == "==":
        return value == bound
    return None


def _holds_when(op: str, ok: bool, direction: str) -> bool:
    """Whether a comparison's outcome survives the rows a truncated read never returned.

    ``direction`` is what more rows can do to the aggregate: ``up`` only rises, ``down``
    only falls, ``""`` either. A truncated read is a BOUND on the real value, so the
    outcome stands only where moving the value that way cannot flip the predicate — a
    count already past its ceiling stays past it, one below it does not stay below.
    ``==`` never survives: any movement breaks equality.
    """
    if direction == "up":
        return (op in _INCREASING_OPS and ok) or (op in _DECREASING_OPS and not ok)
    if direction == "down":
        return (op in _DECREASING_OPS and ok) or (op in _INCREASING_OPS and not ok)
    return False


#: Ordering relations an `event_order` may declare, as the required position of the ``end``
#: side relative to ``start`` -> ``(strict, end must be the later one)``. The two inclusive
#: forms exist because simultaneity is a real reading and a day-granular column produces it.
_ORDER_RELATIONS = {
    "after": (True, True),
    "not_before": (False, True),
    "before": (True, False),
    "not_after": (False, False),
}

#: Quantifiers over the ``(start, end)`` timestamp pairs. No default: a side may carry many
#: timestamps, and "every pair" and "some pair" answer different questions over one row set.
_ORDER_QUANTIFIERS = ("every", "any")

#: Aggregates a `numeric_compare` may declare.
_AGGREGATES = (
    "count",
    "distinct",
    "sum",
    "min",
    "max",
    "avg",
    "median",
    "ratio",
    "mode",
    "mode_share",
)

#: Aggregates that read their field as TEXT. The rest coerce to numbers, so a concentration
#: question could only be asked about a value the pack named in advance (`where` + `ratio`).
_TEXTUAL_AGGREGATES = ("count", "distinct", "mode", "mode_share")

#: Aggregates whose answer is a value's frequency, so the WINNER is part of the finding.
_MODAL_AGGREGATES = ("mode", "mode_share")

#: Aggregates a `normalize:` form actually changes. `count` reads text but counts rows, so a
#: form leaves its answer untouched — and a note claiming otherwise would attribute the number
#: to a projection that never ran. `pack_validate` warns on the inert declaration.
_FORM_AGGREGATES = ("distinct",) + _MODAL_AGGREGATES

#: Aggregates for which an empty selection is a real zero rather than a gap: the selector
#: ran and matched nothing, which is an answer. Nothing has no minimum, maximum or mean —
#: and no most frequent value either, so neither modal aggregate joins them.
_ZERO_ON_EMPTY = ("count", "distinct", "sum")


def _median(nums: List[float]) -> float:
    """The middle value, or the mean of the two middles at even length."""
    ordered = sorted(float(n) for n in nums)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _modal_counts(
    rows: List[Dict], field: str, form: Optional[Dict[str, Any]] = None
) -> List[Tuple[str, int]]:
    """Every normalised value at ``field`` with its frequency, most frequent first.

    Ties resolve on the value ascending, because the winner's IDENTITY is the finding: a
    dict-iteration winner would name a different value on two evaluations of one row set,
    and a verdict that is not reproducible is not a verdict.

    ``form`` is a pack-declared equivalence form; absent, values are counted on the incumbent
    identifier key, so a pack declaring none counts exactly what it always counted.
    """
    counts: Dict[str, int] = defaultdict(int)
    for v in _collect(rows, field):
        key = _canonical_key(v, form) or ""
        if key:
            counts[key] += 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _modal_note(
    rows: List[Dict], field: str, aggregate: str, form: Optional[Dict[str, Any]] = None
) -> str:
    """Which value won a modal aggregate, or ``""`` for every other aggregate.

    A report reading ``mode = 34 (>= 20)`` states a concentration and names nothing, which is
    the same defect as a condition label printing its requirement instead of its finding — so
    the winner rides on ``observed`` beside the number it produced.
    """
    if aggregate not in _MODAL_AGGREGATES:
        return ""
    ranked = _modal_counts(rows, field, form)
    if not ranked:
        return ""
    winner, hits = ranked[0]
    note = f" [most frequent: {winner} on {hits} of {len(rows)} row(s)"
    if len(ranked) > 1:
        note += f", {len(ranked)} distinct value(s)"
    return note + "]"


def _aggregate_value(
    rows: List[Dict],
    field: str,
    aggregate: str,
    form: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], str]:
    """One aggregate over ``rows`` -> ``(value, direction)``; ``(None, "")`` when unreadable.

    ``direction`` is what more rows can do to the value (see ``_holds_when``). Rows present
    with no readable value at ``field`` is a projection gap and stays unreadable — reading it
    as a zero would turn a column that never arrived into a finding.

    ``form`` applies to the aggregates that read their field as TEXT: `distinct` counts
    equivalence classes rather than spellings, and the modal pair ranks them. The numeric
    aggregates ignore it — a form projects text, and a number has no surface form.
    """
    if aggregate in _TEXTUAL_AGGREGATES:
        if field and rows and not _path_retrieved(rows, [field]):
            return None, ""
        if aggregate == "count":
            # No field: the rows themselves are the unit being counted.
            return float(len(rows) if not field else len(_collect(rows, field))), "up"
        if aggregate == "distinct":
            keys = {_canonical_key(v, form) or "" for v in _collect(rows, field)}
            keys.discard("")
            return float(len(keys)), "up"
        ranked = _modal_counts(rows, field, form)
        if not ranked:
            return None, ""
        top = float(ranked[0][1])
        if aggregate == "mode":
            # The top frequency only rises with more rows, so a `>` conclusion survives a
            # truncated read. The winner's identity does NOT — rows that never came back can
            # carry a different value entirely — which is why it is reported as provisional
            # rather than compared.
            return top, "up"
        if not rows:
            return None, ""
        # A share is non-monotone in both terms, exactly like `ratio`: more rows raise the
        # numerator and the denominator by amounts neither of which is bounded by the other.
        return top / float(len(rows)), ""
    if not rows:
        return (0.0, "up") if aggregate == "sum" else (None, "")
    nums = [n for n in (_coerce_num(v) for v in _collect(rows, field)) if n is not None]
    if not nums:
        return None, ""
    if aggregate == "sum":
        # Monotone only while nothing is negative; a column carrying credits is not.
        return float(sum(nums)), "up" if min(nums) >= 0 else ""
    if aggregate == "min":
        return float(min(nums)), "down"
    if aggregate == "max":
        return float(max(nums)), "up"
    if aggregate == "avg":
        return float(sum(nums)) / float(len(nums)), ""
    if aggregate == "median":
        return _median(nums), ""
    return None, ""


def _grouped_aggregate(
    rows: List[Dict],
    field: str,
    aggregate: str,
    group_by: str,
    form: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], str, str, int, List[Dict]]:
    """The largest per-group aggregate -> ``(value, direction, group, n_groups, group_rows)``.

    The largest group decides either reading of an upper bound — ``>``/``>=`` asks whether
    ANY group exceeds it, ``<``/``<=`` whether EVERY group is within it — and the maximum
    answers both. Mirrors ``velocity_count``, which reads its busiest actor.

    Only an aggregate that rises with more rows keeps its direction here: the largest of
    several minima can move either way as rows arrive, so it is treated as non-monotone.

    The winning group's rows come back because a modal aggregate's answer is a value, and
    naming it needs the rows it was drawn from rather than the whole selection.
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        if not isinstance(r, dict):
            continue
        for v in _collect([r], group_by):
            key = _norm_identifier(v)
            if key:
                groups[key].append(r)
    best: Optional[float] = None
    best_key, best_dir = "", ""
    best_rows: List[Dict] = []
    for key in sorted(groups):
        value, direction = _aggregate_value(groups[key], field, aggregate, form)
        if value is None:
            continue
        if best is None or value > best:
            best, best_key, best_dir = value, key, direction
            best_rows = groups[key]
    return (
        best,
        (best_dir if best_dir == "up" else ""),
        best_key,
        len(groups),
        best_rows,
    )


def _reduce_numbers(nums: List[float], aggregate: str) -> Optional[float]:
    """One aggregate over numbers that are already the population's members.

    Separate from ``_aggregate_value``, which reads a FIELD off rows and owns the
    projection-gap and monotonicity rules; here each number IS a member, so neither applies.
    """
    if not nums:
        return None
    if aggregate == "count":
        return float(len(nums))
    if aggregate == "distinct":
        return float(len(set(nums)))
    if aggregate in _MODAL_AGGREGATES:
        counts: Dict[float, int] = defaultdict(int)
        for n in nums:
            counts[n] += 1
        # Same tie rule as `_modal_counts`: highest frequency, then the lowest value.
        top = float(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][1])
        return top if aggregate == "mode" else top / float(len(nums))
    if aggregate == "sum":
        return float(sum(nums))
    if aggregate == "min":
        return float(min(nums))
    if aggregate == "max":
        return float(max(nums))
    if aggregate == "avg":
        return float(sum(nums)) / float(len(nums))
    if aggregate == "median":
        return _median(nums)
    return None


def _population_value(
    decl: Dict[str, Any], rows: List[Dict], form: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[float], str]:
    """One number describing a POPULATION -> ``(value, note)``; ``(None, reason)`` if unreadable.

    ``per`` reduces the rows to one value per member (by ``per_aggregate``) before
    ``aggregate`` combines them, because the median of each peer's own total is not the
    median of their rows — and a cohort baseline means the first. Without ``per`` the
    aggregate reads the rows directly, as everywhere else.
    """
    aggregate = str(decl.get("aggregate", "") or "").strip().lower()
    field = str(decl.get("field", "") or "")
    per = str(decl.get("per", "") or "").strip()
    if aggregate not in _AGGREGATES:
        return None, f"declares no readable `aggregate` ({aggregate or '(none)'!r})"
    sel = apply_where(rows, decl.get("where"))
    if aggregate == "ratio":
        if not rows:
            return None, "has no rows to take a proportion of"
        return float(len(sel)) / float(len(rows)), f"{len(sel)} of {len(rows)} row(s)"
    if not per:
        value, _ = _aggregate_value(sel, field, aggregate, form)
        if value is None:
            return None, f"has no readable {aggregate} of {field or 'its rows'}"
        return value, f"over {len(sel)} row(s)"
    inner = str(decl.get("per_aggregate", "") or "").strip().lower()
    if inner not in _AGGREGATES or inner == "ratio":
        return None, (
            "declares `per` with no readable `per_aggregate`, so it has no per-member "
            "value to combine"
        )
    members: Dict[str, List[Dict]] = defaultdict(list)
    for r in sel:
        if not isinstance(r, dict):
            continue
        for v in _collect([r], per):
            key = _norm_identifier(v)
            if key:
                members[key].append(r)
    nums: List[float] = []
    for subset in members.values():
        one, _ = _aggregate_value(subset, field, inner, form)
        if one is not None:
            nums.append(one)
    value = _reduce_numbers(nums, aggregate)
    if value is None:
        return None, f"produced no per-{per} value on any of its {len(sel)} row(s)"
    return value, f"{inner} per {per} over {len(nums)} member(s)"


def _baseline_threshold(
    cond: Dict[str, Any], baseline: Dict[str, Any], multiplier: float
) -> Tuple[float, str, str]:
    """A threshold relative to a population -> ``(threshold, note, refusal)``.

    A non-empty ``refusal`` means the comparison must not run. The baseline's completeness is
    a fact about the baseline's OWN source, stamped by ``_preprocess`` and never inherited
    from the subject side, so a sibling source's truncation cannot suppress a sound reading.
    """
    gap = str(cond.get("_baseline_gap", "") or "")
    if gap:
        return (
            0.0,
            "",
            f"the comparison is relative to a computed baseline, and that population {gap} "
            "— a threshold taken off an incomplete population is not the threshold, and it "
            "understates it, so an ordinary subject reads as an outlier",
        )
    value, note = _population_value(
        baseline, cond.get("_baseline_rows") or [], _projection_form(cond, "normalize")
    )
    if value is None:
        return (
            0.0,
            "",
            f"the comparison is relative to a computed baseline, and that population {note}",
        )
    if value <= 0:
        # At zero every multiplier states the same bound; below zero the comparison inverts.
        # Either way the declared multiple is not what would be tested.
        return (
            0.0,
            "",
            f"the computed baseline is {_fmt_num(value)}, so the declared "
            f"{_fmt_num(multiplier)}x multiple is not a threshold: at zero every multiplier "
            "tests the same bound, and below zero the comparison inverts",
        )
    return multiplier * value, f", baseline = {_fmt_num(value)} ({note})", ""


#: Expected string when a counting condition has no declared bound. Read back by
#: `pack_validate` as the engine's own statement of which kinds require one, so a kind
#: refusing for another reason must say so in its own words rather than borrow this.
_NO_BOUND_EXPECTED = "no bound declared"

#: The same, for an `event_order` refusing to compare: what is missing is the relation
#: between the two events, and nothing about this kind is a magnitude bound.
_NO_ORDER_EXPECTED = "no ordering declared"

#: Default shape for scope-point codes when the ruleset declares none.
_DEFAULT_POINT_RE = re.compile(r"^[A-Z]{3}$")


def _scope_point_codes(
    rows: List[Dict], point_fields: List[str], pattern: str = ""
) -> set:
    """Scope-point codes: declared point fields first, then whole-row scan by shape.

    ``pattern`` (``point_pattern`` in the ruleset) overrides the default 3-letter shape.
    A bad regex falls back to the default rather than raising.
    """
    try:
        shape = re.compile(pattern) if pattern else _DEFAULT_POINT_RE
    except re.error:
        logger.warning(
            "route_membership: point_pattern %r does not compile; using the default "
            "3-letter code shape for the whole-row fallback.",
            pattern,
        )
        shape = _DEFAULT_POINT_RE
    codes: set = set()
    for f in point_fields or []:
        for v in _collect(rows, f):
            s = str(v).strip().upper()
            # The shape filter applies to declared fields too: a scope gate is the one
            # check where a widened input set could flip a jurisdiction call. Making the
            # shape declarable is the fix; packs that declare nothing behave as before.
            if shape.match(s):
                codes.add(s)
    if not codes:  # fall back to scanning all leaves for scope-point-shaped tokens
        for r in rows:
            if isinstance(r, dict):
                for v in _row_values(r):
                    s = str(v).strip().upper()
                    if shape.match(s):
                        codes.add(s)
    return codes


def _count_array_leaves(rows: List[Dict], path: str) -> Optional[int]:
    """Element leaves under an array path for the subject record; None if the path is absent.

    Handles a raw list under a flat dotted key (``{"elem.kind": ["K1"]}``, which
    ``resolve_path`` skips because the value isn't scalar), as well as exploded/nested shapes.

    Returns the max count in any single row, not the sum: a versioned source returns one row
    per snapshot carrying the whole array, so summing would overcount.

    0 and None are distinct: an empty array (``[]`` or ``null``) means "no such elements"
    (a real 0); a path that never arrived means "we did not look". The container is probed
    explicitly because an empty list contributes no leaves either way. The caller depends
    on the distinction: conflating the two lets a retrieval gap read as evidence."""
    best, seen = 0, False
    for r in rows:
        if not isinstance(r, dict):
            continue
        # Direct flat list under the literal dotted key.
        if path in r and isinstance(r[path], list):
            seen = True
            best = max(best, len([x for x in r[path] if x is not None]))
            continue
        leaves = flatten_leaves(r)
        matched = [k for k in leaves if k == path or k.startswith(path + ".")]
        if matched:
            seen = True
            best = max(best, sum(len(leaves[k]) for k in matched))
            continue
        # No leaves under that name: resolve the container via `_resolve_nodes`, which
        # handles underscore-flattened aliases and JSON-string structs, and also
        # distinguishes "container present but empty" (a real 0) from "key never arrived"
        # (None): an empty list or null still resolves; a missing key does not.
        nodes = _resolve_nodes(r, path.split("."))
        if nodes:
            seen = True
            for node in nodes:
                if isinstance(node, list):
                    best = max(best, len([x for x in node if x is not None]))
                elif isinstance(node, dict):
                    best = max(best, 1)
    return best if seen else None


def _max_counter(rows: List[Dict], field: str) -> Optional[int]:
    """Max integer value of a counter leaf across rows; None if the leaf is absent."""
    vals = [_coerce_num(v) for v in _collect(rows, field)]
    vals = [v for v in vals if v is not None]
    return int(max(vals)) if vals else None


_QUOTE_MONTHS = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()


def _quote_day(value: Any) -> str:
    """A date rendered the way operator notations write it (``2026-07-09`` -> ``09JUL``).

    Empty when unparseable; a half-rendered date is worse than an absent one."""
    dt = _parse_ts(value)
    if dt is None:
        return ""
    try:
        return f"{dt.day:02d}{_QUOTE_MONTHS[dt.month - 1]}"
    except (IndexError, ValueError):
        return ""


def _render_element_quote(template: str, node: Any, fields: Dict[str, Any]) -> str:
    """Render a pack ``quote_template`` against one element node -> the operator's notation.

    ``fields`` maps a placeholder name onto the leaf path(s) inside the element, like
    ``subject_links.fields``; a name suffixed ``_day`` gets the ``09JUL`` rendering.
    Domain-specific details (notation, leaf paths) stay in the pack. Placeholders are
    scanned explicitly rather than via ``str.format`` so an unknown name degrades to empty
    instead of raising inside the condition evaluator."""

    def _leaf(key: str) -> str:
        spec = fields.get(key)
        if not spec:
            return ""
        for f in spec if isinstance(spec, list) else [spec]:
            for v in resolve_path(node, str(f)):
                text = str(v).strip()
                if text:
                    return text
        return ""

    out: List[str] = []
    i = 0
    while i < len(template):
        if template[i] == "{":
            end = template.find("}", i)
            if end == -1:
                out.append(template[i:])
                break
            name = template[i + 1 : end].strip()
            if name.endswith("_day"):
                out.append(_quote_day(_leaf(name[: -len("_day")]) or _leaf(name)))
            else:
                out.append(_leaf(name))
            i = end + 1
            continue
        out.append(template[i])
        i += 1
    # Collapse the gaps a missing leaf leaves behind so a partial quote still reads cleanly.
    return re.sub(r"\s{2,}", " ", "".join(out)).strip(" -/")


def _element_nodes(rows: List[Dict], path: str) -> List[Any]:
    """The element dicts at an array path across rows, deduped, order preserved.

    A versioned backend repeats the same element on every later version row (one live grant
    element appeared on 44 of them), so quoting per row would print one identical line per
    version. Dedupe on the element's own JSON so the report quotes each element once."""
    out: List[Any] = []
    seen: set = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        for node in _resolve_nodes(r, path.split(".")):
            for item in node if isinstance(node, list) else [node]:
                if not isinstance(item, dict):
                    continue
                try:
                    key = json.dumps(item, sort_keys=True, default=str)
                except Exception:
                    key = str(item)
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
    return out


#: The composite kinds: one condition whose finding is a combination of its ``children``.
_COMPOSITE_KINDS = ("all_of", "any_of", "none_of")

#: How deep composites may nest. A bound exists because a check importing itself would
#: otherwise recurse until the interpreter ends the verdict rather than the pack.
_MAX_COMPOSITE_DEPTH = 5

#: Which child result decides each (kind, outcome) pair, for the observed line: the reader
#: needs the children that settled it, not the ones that came along.
_PIVOTAL_CHILD = {
    ("all_of", "fail"): "fail",
    ("all_of", "pass"): "pass",
    ("any_of", "pass"): "pass",
    ("any_of", "fail"): "fail",
    ("none_of", "fail"): "pass",
    ("none_of", "pass"): "fail",
}


def _composite_result(kind: str, results: List[str]) -> str:
    """Three-valued combination of a composite's children.

    ``unknown`` never reads as ``pass``: the composite is decided only where the undecided
    children could not have changed it — one failing child settles ``all_of`` whatever else
    is missing, and ``any_of`` needs every child to fail before it may say so.
    """
    if kind == "all_of":
        if any(r == "fail" for r in results):
            return "fail"
        return "pass" if all(r == "pass" for r in results) else "unknown"
    if kind == "any_of":
        if any(r == "pass" for r in results):
            return "pass"
        return "fail" if all(r == "fail" for r in results) else "unknown"
    # none_of is any_of mirrored: a child that passes is the thing that must not happen.
    if any(r == "pass" for r in results):
        return "fail"
    return "pass" if all(r == "fail" for r in results) else "unknown"


def _composite(
    cond: Dict[str, Any], src_rows: Dict[str, List[Dict]], kind: str, depth: int
) -> Tuple[str, str, str, str, bool]:
    """Evaluate a composite condition -> ``mk`` arguments plus its ``own_detail`` flag.

    Produces ONE finding: the children are the mechanics of this check, not checks of their
    own, so the parent owns every weighting key and the report prints one line. The flag is
    set only for the two declaration refusals, whose detail is about the pack rather than the
    data and must not be replaced by the parent's `unknown_detail`.
    """
    children = [c for c in (cond.get("children") or []) if isinstance(c, dict)]
    names = ", ".join(str(c.get("id") or c.get("kind") or "?") for c in children)
    want = f"{kind.replace('_', ' ')}: {names}" if names else kind.replace("_", " ")
    if not children:
        return (
            "unknown",
            want,
            "no children",
            f"this `{kind}` declares no `children`, so there is nothing to combine",
            True,
        )
    if depth >= _MAX_COMPOSITE_DEPTH:
        logger.warning(
            "Condition '%s' nests composites more than %d deep — evaluated as unknown",
            cond.get("id", kind),
            _MAX_COMPOSITE_DEPTH,
        )
        return (
            "unknown",
            want,
            f"nested more than {_MAX_COMPOSITE_DEPTH} deep",
            f"composite conditions nest at most {_MAX_COMPOSITE_DEPTH} deep and this one "
            "goes further, so it was not evaluated",
            True,
        )
    checks = [_eval_condition(child, src_rows, depth + 1) for child in children]
    result = _composite_result(kind, [c.result for c in checks])
    pivot = _PIVOTAL_CHILD.get((kind, result), "unknown")
    deciding = [c for c in checks if c.result == pivot] or checks
    shown = "; ".join(
        f"{c.id} {c.result}" + (f" ({c.observed})" if c.observed else "")
        for c in deciding[:3]
    )
    if len(deciding) > 3:
        shown += f"; and {len(deciding) - 3} more"
    key = {"pass": "pass_detail", "fail": "fail_detail"}.get(result, "")
    said = str((cond.get(key) if key else "") or "")
    return (
        result,
        want,
        shown,
        said
        or "; ".join(c.detail for c in deciding[:3] if c.detail)
        or f"{len(deciding)} of {len(checks)} member check(s) came back {pivot}",
        False,
    )


def _condition_sources(cond: Dict[str, Any]) -> List[str]:
    """Logical source names a condition reads from (for subject-scope override).

    Covers the single-``source`` conditions plus the two-sided ``left``/``right`` of a
    field_equality / time_gap. Only these need the scope override today, but returning
    all referenced logicals keeps it generic. Recurses into a composite's ``children``, or a
    source only a child reads would be invisible to the scope logic built on this."""
    out = []
    if cond.get("source"):
        out.append(cond["source"])
    for side in ("left", "right", "start", "end"):
        s = cond.get(side, {})
        if isinstance(s, dict) and s.get("source"):
            out.append(s["source"])
    for fb in cond.get("fallbacks", []) or []:
        if isinstance(fb, dict) and fb.get("source"):
            out.append(fb["source"])
    for child in cond.get("children", []) or []:
        if isinstance(child, dict):
            out.extend(_condition_sources(child))
    return list(dict.fromkeys(out))


def _eval_condition(
    cond: Dict[str, Any],
    src_rows: Dict[str, List[Dict]],
    depth: int = 0,
) -> ConditionCheck:
    """Evaluate one generic condition against the per-logical-source row subsets.

    ``src_rows`` is keyed by the ruleset's logical source names (e.g. 'records', 'settlement'),
    already filtered to the current subject. Returns a ConditionCheck with pass/fail/
    unknown. Any evaluation error is caught by the caller (best-effort).

    ``depth`` counts composite nesting only, so a `pair_by` re-entry stays at its own level.
    """
    kind = cond.get("kind", "")
    cid = str(cond.get("id", kind))
    label = str(cond.get("label", cid))
    decisive = bool(cond.get("decisive", False))
    polarity = cond.get("polarity", "exclusion")
    if polarity not in ("exclusion", "fraud_indicator"):
        polarity = "exclusion"
    # What kind of evidence an exclusion carries (see ConditionCheck.exclusion_kind).
    # Carried onto the check so every consumer reads one typed field instead of
    # re-deriving it from the spec (or, worse, string-matching a prose note).
    exclusion_kind = (
        str(cond.get("exclusion_kind", "heuristic") or "heuristic").strip().lower()
    )
    if exclusion_kind not in ("heuristic", "categorical"):
        exclusion_kind = "heuristic"
    # `decisive_on` restricts which results the decisive flag applies to.
    # Default is both "fail" and "unknown" (symmetric). Declaring only "fail" makes a
    # check conclusive on FAIL but not degrading on unknown.
    decisive_on = cond.get("decisive_on") or ["fail", "unknown"]
    decisive_on = {str(x).strip().lower() for x in decisive_on}
    # Explains why a check came back `unknown` after row_match narrowing, so an unresolvable
    # identity scope is distinguishable from a source that returned nothing.
    scope_note = str(cond.get("_scope_note", "") or "")

    def mk(result, expected="", observed="", detail="", own_detail=False):
        # All kinds pass through here so `unknown_detail` is applied once.
        # `own_detail=True` skips it for branches whose detail is about the declaration or
        # scope, not the data (they carry their own `truncated_detail`/`absent_detail`).
        if result == "unknown" and not own_detail:
            said = str(cond.get("unknown_detail", "") or "").strip()
            if said:
                detail = said
        if result == "unknown" and scope_note:
            detail = f"{detail}; {scope_note}" if detail else scope_note
        return ConditionCheck(
            id=cid,
            label=label,
            result=result,
            expected=expected,
            observed=observed,
            detail=detail,
            decisive=decisive and result in decisive_on,
            polarity=polarity,
            exclusion_kind=exclusion_kind,
            # Presentational only: which table the report prints this row in. Copied
            # straight from the pack so the engine never has to know the procedure's own
            # sectioning, and an undeclared group falls back to one flat table.
            group=str(cond.get("report_group", "") or ""),
        )

    # `pair_by: record` compares each side within its own record rather than pooling
    # across the source, to avoid a cross-record pair satisfying the comparison.
    if kind in ("field_equality", "time_gap") and (
        str(cond.get("pair_by", "") or "").strip().lower() == "record"
    ):
        bare = {k: v for k, v in cond.items() if k != "pair_by"}
        logical, units, refusal = _pair_units(cond, src_rows)
        if refusal:
            pooled = _eval_condition(bare, src_rows, depth)
            said = (
                "`pair_by: record` was NOT honoured — "
                + refusal
                + "; the two sides were pooled across the source, so a matching pair may sit "
                "on no single record"
            )
            return pooled.model_copy(
                update={"detail": f"{pooled.detail}; {said}" if pooled.detail else said}
            )
        # The unit is the row for the per-record call, so the side must not re-apply the
        # narrowing that produced it: `records:` would resolve inside the entry and find
        # nothing, and `where:` has already run.
        for side_key in ("left", "right", "start", "end"):
            side = bare.get(side_key)
            if isinstance(side, dict):
                bare[side_key] = {
                    k: v for k, v in side.items() if k not in ("records", "where")
                }
        if not units:
            # No record to read: the kind's own no-data answer, unchanged.
            return _eval_condition(bare, {logical: []}, depth)
        return _rollup_paired_records(
            [
                (i, _eval_condition(bare, {logical: [u]}, depth))
                for i, u in enumerate(units)
            ],
            cid,
        )

    if kind == "element_absence":
        rows = src_rows.get(cond.get("source", ""), [])
        counters = cond.get("counters", []) or []
        arrays = cond.get("arrays", []) or []
        want_absent = str(cond.get("expected_label", "") or "") or "all elements absent"
        present_total, any_leaf = 0, False
        detail_bits = []
        # Paths that did not resolve: absence can only be concluded for paths that were read.
        missing: List[str] = []
        for c in counters:
            m = _max_counter(rows, c)
            if m is None:
                missing.append(c)
                continue
            any_leaf = True
            if m > 0:
                present_total += m
                detail_bits.append(f"{c}={m}")
        for a in arrays:
            # Arrays may arrive exploded or as a raw list; flatten_leaves descends both.
            n = _count_array_leaves(rows, a)
            if n is None:
                missing.append(a)
                continue
            any_leaf = True
            if n > 0:
                present_total += n
                detail_bits.append(f"{a}×{n}")
        if not rows or not any_leaf:
            return mk(
                "unknown",
                want_absent,
                "no element data",
                "declared element paths not present in the retrieved rows",
            )
        if present_total > 0:
            # A single present element settles it: the absence check fails with or without gaps.
            # Quote the fired elements where the pack says how; the count stays as arithmetic.
            quote_template = str(cond.get("quote_template", "") or "")
            quote_fields = cond.get("quote_fields", {}) or {}
            quotes: List[str] = []
            if quote_template and quote_fields:
                for a in arrays:
                    for node in _element_nodes(rows, a):
                        q = _render_element_quote(quote_template, node, quote_fields)
                        if q and q not in quotes:
                            quotes.append(q)
            observed = ", ".join(detail_bits)
            if quotes:
                observed = "; ".join(quotes[:5]) + f" [{observed}]"
            return mk(
                "fail",
                want_absent,
                observed,
                str(cond.get("fail_detail", "") or "") or "disallowed elements present",
            )
        if missing:
            # A pass over partial coverage silently converts a retrieval gap into a finding;
            # report unknown instead and name the unretrieved paths.
            return mk(
                "unknown",
                want_absent,
                f"0 for {len(counters) + len(arrays) - len(missing)} path(s); "
                f"not retrieved: {', '.join(missing)}",
                "cannot confirm absence — "
                f"{', '.join(missing)} absent from the retrieved rows",
            )
        return mk(
            "pass",
            want_absent,
            "0",
            str(cond.get("pass_detail", "") or "") or "no disallowed elements",
        )

    if kind == "element_presence":
        # Any found element is a pass. A gap is unknown (not fail). A fail requires full
        # coverage with all paths empty.
        rows = src_rows.get(cond.get("source", ""), [])
        counters = cond.get("counters", []) or []
        arrays = cond.get("arrays", []) or []
        found_bits: List[str] = []
        gaps: List[str] = []
        for c in counters:
            m = _max_counter(rows, c)
            if m is None:
                gaps.append(c)
            elif m > 0:
                found_bits.append(f"{c}={m}")
        for a in arrays:
            n = _count_array_leaves(rows, a)
            if n is None:
                gaps.append(a)
            elif n > 0:
                found_bits.append(f"{a}×{n}")
        want = str(cond.get("expected", "the element is present"))
        if found_bits:
            return mk("pass", want, ", ".join(found_bits), "required element present")
        if not rows or gaps:
            return mk(
                "unknown",
                want,
                (
                    f"not retrieved: {', '.join(gaps)}"
                    if gaps
                    else "source returned no rows"
                ),
                "cannot confirm presence — the declared element path was not retrieved",
            )
        return mk("fail", want, "0", "required element absent")

    if kind == "field_equality":
        left = cond.get("left", {}) or {}
        right = cond.get("right", {}) or {}
        lvals = _collect(_side_rows(src_rows, left), left.get("field", ""))
        rvals = _collect(_side_rows(src_rows, right), right.get("field", ""))
        if not lvals or not rvals:
            return mk(
                "unknown",
                "left == right",
                f"left={lvals[:3]} right={rvals[:3]}",
                "one side of the comparison had no data",
            )
        norm = cond.get("normalize", "")
        if norm == "identifier":
            match = any(_identifiers_match(a, b) for a in lvals for b in rvals)
        else:
            lset = {str(v).strip().upper() for v in lvals}
            rset = {str(v).strip().upper() for v in rvals}
            match = bool(lset & rset)
        obs = f"{sorted({str(v) for v in lvals})[:3]} vs {sorted({str(v) for v in rvals})[:3]}"
        return mk(
            "pass" if match else "fail",
            "equal",
            obs,
            (
                (
                    str(cond.get("pass_detail", "") or "")
                    or "the two sides carry the same value"
                )
                if match
                else (
                    str(cond.get("fail_detail", "") or "")
                    or "the two sides carry no value in common"
                )
            ),
        )

    if kind == "time_gap":
        # An absent or unparseable `max` yields unknown, not a fabricated default bound.
        max_delta = _window_delta(f"within:{cond.get('max', '')}")
        if max_delta is None:
            return mk(
                "unknown",
                _NO_BOUND_EXPECTED,
                "not compared",
                "this interval condition declares no usable `max` (expected a window like "
                "`90m`, `4h` or `2d`), so no gap can be called too long",
                own_detail=True,
            )
        start = cond.get("start", {}) or {}
        end = cond.get("end", {}) or {}
        svals = [
            _parse_ts(v)
            for v in _collect(_side_rows(src_rows, start), start.get("field", ""))
        ]
        evals = [
            _parse_ts(v)
            for v in _collect(_side_rows(src_rows, end), end.get("field", ""))
        ]
        svals = [t for t in svals if t]
        evals = [t for t in evals if t]
        if not svals or not evals:
            return mk(
                "unknown",
                f"gap <= {cond.get('max', '')}",
                "missing timestamps",
                # Replaced by the condition's own `unknown_detail` in `mk`.
                "one end of the interval has no parseable timestamp",
            )
        gap = _norm_dt(min(evals)) - _norm_dt(min(svals))
        # Allow a small negative slack (clock skew / one event just before the other's write).
        ok = timedelta(minutes=-5) <= gap <= max_delta
        return mk(
            "pass" if ok else "fail",
            f"<= {cond.get('max')}",
            f"gap={gap}",
            (
                (
                    str(cond.get("pass_detail", "") or "")
                    or f"the two events are {gap} apart, within {cond.get('max')}"
                )
                if ok
                else (
                    str(cond.get("fail_detail", "") or "")
                    or f"the two events are {gap} apart, more than the {cond.get('max')} allowed"
                )
            ),
        )

    if kind == "event_order":
        # An ORDERING claim, and `time_gap` above is not one: it compares MAGNITUDES and
        # accepts either order, so a check labelled "B followed A" passes on a B that
        # preceded A. Both sides keep `time_gap`'s shape, so a check converts by kind.
        relation = str(cond.get("relation", "") or "").strip().lower()
        quant = str(cond.get("quantifier", "") or "").strip().lower()
        if relation not in _ORDER_RELATIONS or quant not in _ORDER_QUANTIFIERS:
            return mk(
                "unknown",
                _NO_ORDER_EXPECTED,
                "not compared",
                "this ordering condition needs a readable `relation` and `quantifier`; it "
                f"declares relation={relation or '(none)'!r}, quantifier="
                f"{quant or '(none)'!r}. Neither has a default: the relation IS the check, "
                "and `every` and `any` disagree over the same rows",
                own_detail=True,
            )
        strict, later = _ORDER_RELATIONS[relation]
        # An absent `tolerance` is EXACT and never a fabricated slack. `time_gap` forgives
        # five minutes of clock skew from a literal in this file, which is a deployment's
        # judgement about its own sources taken in Python where no pack can restate it.
        raw_tol = str(cond.get("tolerance", "") or "").strip()
        slack = timedelta(0)
        if raw_tol:
            parsed = _window_delta(f"within:{raw_tol}")
            if parsed is None:
                return mk(
                    "unknown",
                    _NO_ORDER_EXPECTED,
                    "not compared",
                    f"this ordering condition declares `tolerance: {raw_tol}`, which is not "
                    "a readable window (expected `90m`, `4h` or `2d`) — comparing at zero "
                    "slack instead would answer a stricter question than the one declared",
                    own_detail=True,
                )
            slack = parsed
        want = f"{quant} `end` {relation.replace('_', ' ')} `start`" + (
            f" (±{raw_tol})" if raw_tol else ""
        )
        start = cond.get("start", {}) or {}
        end = cond.get("end", {}) or {}
        svals = sorted(
            _norm_dt(t)
            for t in (
                _parse_ts(v)
                for v in _collect(_side_rows(src_rows, start), start.get("field", ""))
            )
            if t
        )
        evals = sorted(
            _norm_dt(t)
            for t in (
                _parse_ts(v)
                for v in _collect(_side_rows(src_rows, end), end.get("field", ""))
            )
            if t
        )
        if not svals or not evals:
            return mk(
                "unknown",
                want,
                "missing timestamps",
                "one side of the ordering has no parseable timestamp",
            )
        # The quantifier picks the DECIDING pair rather than a per-side reduction the pack
        # would have to declare twice: a universal is settled by the hardest pair, an
        # existential by the easiest, and which timestamp that is follows from the relation.
        if later == (quant == "every"):
            s_ref, e_ref = svals[-1], evals[0]
        else:
            s_ref, e_ref = svals[0], evals[-1]
        delta = e_ref - s_ref
        if later:
            ok = delta > -slack if strict else delta >= -slack
        else:
            ok = delta < slack if strict else delta <= slack
        sat = (
            "later"
            if delta > timedelta(0)
            else ("earlier" if delta else "simultaneous")
        )
        observed = (
            f"{e_ref.isoformat()} vs {s_ref.isoformat()} "
            f"[{len(svals)} start / {len(evals)} end timestamp(s)]"
        )
        # `every` is universal, so a row that never came back can only BREAK a pass; `any`
        # is existential, so it can only rescue a fail. The surviving outcome is the one the
        # missing rows cannot reach — the same asymmetry `_holds_when` applies to a bound.
        sides = {str(start.get("source", "") or ""), str(end.get("source", "") or "")}
        if ok == (quant == "every") and sides & set(cond.get("_count_truncated") or []):
            return mk(
                "unknown",
                want,
                f"{observed} in a TRUNCATED read",
                str(cond.get("truncated_detail", "") or "")
                or (
                    "a source of this ordering was cut off at its row cap, and the "
                    f"outcome turns on rows that never came back: `{quant}` holds over the "
                    "rows that did, which is not the same claim"
                ),
                own_detail=True,
            )
        said = str(cond.get("pass_detail" if ok else "fail_detail", "") or "")
        return mk(
            "pass" if ok else "fail",
            want,
            observed,
            said
            or (
                f"the deciding `end` is {sat}"
                + (f" by {abs(delta)}" if delta else "")
                + f", which does{'' if ok else ' not'} satisfy: {want}"
            ),
        )

    if kind == "record_absence":
        rows = src_rows.get(cond.get("source", ""), [])
        fields = cond.get("fields", []) or []
        forbidden = [str(x).upper() for x in (cond.get("forbidden_values", []) or [])]
        want = str(cond.get("expected_label", "") or "") or (
            "none of " + ", ".join(str(v) for v in (cond.get("forbidden_values") or []))
            if forbidden
            else "no matching record"
        )
        found_note = (
            str(cond.get("fail_detail", "") or "") or "forbidden value present in this source"
        )
        clear_note = (
            str(cond.get("pass_detail", "") or "") or "no forbidden value in this source"
        )
        if not rows:
            # A resolved-but-empty row_match scope means the identity is genuinely not in
            # the source, not that the source went unasked. Absent `_scope_resolved_empty`
            # the absence is ambiguous and reports unknown.
            if cond.get("_scope_resolved_empty"):
                return mk(
                    "pass",
                    want,
                    "identity not found",
                    "the acting identity is absent from this reference source",
                )
            return mk(
                "unknown",
                want,
                "no rows",
                f"source '{cond.get('source', '')}' did not return",
            )
        # `match: exact` required for single-letter vocabularies (substring hits any value
        # containing that letter). `_collect_any` instead of `_collect` to include booleans.
        exact = str(cond.get("match", "substring")).lower() == "exact"
        hits = []
        for f in fields:
            for v in _collect_any(rows, f):
                sv = str(v).upper()
                if (sv in forbidden) if exact else any(fb in sv for fb in forbidden):
                    hits.append(str(v))
        if hits:
            return mk("fail", want, ", ".join(sorted(set(hits))[:5]), found_note)
        return mk("pass", want, "none", clear_note)

    if kind == "cohort_membership":
        # Asks whether the subject's key appears on another record in the same cohort.
        # `discriminator` names what makes a record different; rows agreeing with the subject
        # on it are excluded before the test. Finding a match on a truncated source is
        # positive evidence; not finding one on a truncated source is unknown (not a clear).
        rows = _side_rows(src_rows, cond)
        key_fields = [f for f in (cond.get("key_fields", []) or []) if f]
        disc = str(cond.get("discriminator", "") or "").strip()
        # `discriminators` lists multiple fields; a row is "other" only when it differs
        # on every one. A single-field discriminator asking for record and date
        # is not a weaker check: a different answer on date alone is a different question.
        #
        # A missing value is treated as not-other: a wrong fail fires an exclusion (high
        # cost), a wrong skip loses one non-decisive corroborator (low cost).
        discs = [
            s
            for s in (
                str(f or "").strip() for f in (cond.get("discriminators", []) or [])
            )
            if s
        ] or ([disc] if disc else [])
        want = (
            cond.get("expected_label", "") or "no other record shares the subject's key"
        )
        # `key_groups` (opt-in) declares composite keys: fields ANDed into a tuple. A partial
        # group is skipped. Absent: `key_fields` is a union match across any listed field.
        key_groups = [
            [str(f) for f in g if f]
            for g in (cond.get("key_groups", []) or [])
            if isinstance(g, (list, tuple)) and [f for f in g if f]
        ]

        def _keys_of(rowset):
            """Identity values in `rowset`: composite tuples when declared, else union."""
            if key_groups:
                out = set()
                for r in rowset:
                    # First fully-satisfied group wins (same rule as `identity_keys`).
                    for g in key_groups:
                        cols = []
                        for f in g:
                            got = [
                                str(v).strip().upper()
                                for v in resolve_path(r, f)
                                if str(v).strip()
                            ]
                            if not got:
                                cols = []  # incomplete: this group cannot name the row
                                break
                            cols.append(got)
                        if not cols:
                            continue
                        # Pair multi-valued fields positionally (one party per index).
                        # A single-valued field broadcasts (the exploded normal case).
                        n = max(len(c) for c in cols)
                        for i in range(n):
                            out.add(
                                " | ".join(c[i] if len(c) > i else c[0] for c in cols)
                            )
                        break
                return out
            return {
                str(v).strip().upper()
                for f in key_fields
                for r in rowset
                for v in resolve_path(r, f)
                if str(v).strip()
            }

        # The subject's own rows carry its key. `subject_rows_where` selects them from the
        # same source (e.g. the alerted record's rows inside a unit-wide sweep).
        subj_sel = cond.get("subject_rows", {}) or {}
        subj_rows = _side_rows({**src_rows, "_": rows}, {**subj_sel, "source": "_"})
        if not rows or not (key_fields or key_groups) or not discs:
            return mk(
                "unknown",
                want,
                "no rows" if not rows else "check not fully declared",
                (
                    f"source '{cond.get('source', '')}' returned no rows"
                    if rows is not None and not rows
                    else "the check needs key_fields and a discriminator"
                ),
            )
        # A missing subject selector causes every cohort row to appear as "the subject's
        # own", leaving nothing to compare. Report unknown, not a pass.
        if not [
            w
            for w in (subj_sel.get("where", []) or [])
            if isinstance(w, dict) and str(w.get("field", "") or "").strip()
        ]:
            return mk(
                "unknown",
                want,
                f"no subject selector declared ({len(rows)} rows returned)",
                "this cohort check declares no `subject_rows`, so the subject's own rows "
                "cannot be told apart from the cohort's others: every row would count as "
                "the subject's own and nothing would be compared. An authoring gap in the "
                "check, NOT a finding about the subject and specifically not evidence that "
                "its key appears nowhere else",
                own_detail=True,
            )
        # An unfilled subject_rows anchor is not a "matched nothing" anchor:
        # `apply_where` skips empty clauses so every row would be counted as the subject's own.
        anchor_gap = [
            str(t).strip()
            for t in (cond.get("_subject_anchor_unresolved") or [])
            if str(t).strip()
        ]
        if anchor_gap:
            return mk(
                "unknown",
                want,
                f"no {', '.join(anchor_gap)} value to locate the subject by",
                "the subject's own rows could not be selected inside the comparison set: the "
                f"incident supplied no {', '.join(anchor_gap)} value, so every row would "
                "count as the subject's own and nothing would be compared. NOT a finding "
                "about the subject",
                own_detail=True,
            )
        subj_keys = _keys_of(subj_rows)
        subj_disc = {
            f: {
                str(v).strip().upper()
                for r in subj_rows
                for v in resolve_path(r, f)
                if str(v).strip()
            }
            for f in discs
        }
        if not subj_keys:
            if not subj_rows:
                return mk(
                    "unknown",
                    want,
                    f"subject NOT IN the cohort ({len(rows)} rows returned)",
                    "the subject's own rows are absent from the comparison set, so it could "
                    "never match: nothing was compared. The cohort's scope (unit, window, "
                    "or subject selector) does not cover the subject — NOT a finding about "
                    "it, and specifically not evidence of no history",
                    own_detail=True,
                )
            return mk(
                "unknown",
                want,
                f"subject present ({len(subj_rows)} row(s)) but its key fields are empty",
                "the subject's rows are in the cohort but carry none of the key fields, so "
                "there is nothing to look for elsewhere — the identity leaf was not returned "
                "by the query (a projection gap), not a fact about the subject",
                own_detail=True,
            )
        # A row is "other" only when it differs from the subject on every discriminator.
        hits = []
        for r in rows:
            rdisc = {
                f: {
                    str(v).strip().upper()
                    for v in resolve_path(r, f)
                    if str(v).strip()
                }
                for f in discs
            }
            if any(not rdisc[f] or rdisc[f] <= subj_disc[f] for f in discs):
                continue  # the subject's own record (or indistinguishable from it)
            if _keys_of([r]) & subj_keys:
                # Label by the first declared discriminator so the note names the other record.
                hits.append(f"{sorted(rdisc[discs[0]])[0]}")
        if hits:
            return mk(
                "fail",
                want,
                f"{len(set(hits))} other record(s): {sorted(set(hits))[:5]}",
                cond.get("fail_detail", "the subject's key appears on other records"),
            )
        if cond.get("_cohort_truncated"):
            return mk(
                "unknown",
                want,
                f"none in {len(rows)} rows returned (TRUNCATED)",
                cond.get(
                    "truncated_detail",
                    "the cohort source was cut off at its row cap, so not finding a match "
                    "in what came back is no evidence that none exists",
                ),
                own_detail=True,
            )
        return mk(
            "pass",
            want,
            f"none among {len(rows)} cohort rows",
            cond.get("pass_detail", "no other record in the cohort shares the key"),
        )

    if kind == "field_flag":
        rows = src_rows.get(cond.get("source", ""), [])
        f, vals = _first_present_flag(rows, cond.get("flag_fields", []) or [])
        if not vals:
            return mk(
                "unknown",
                f"flag == {cond.get('expected')}",
                "flag absent",
                "the declared flag field is not present in the retrieved rows",
            )
        truthy = any(str(v).strip().lower() in _TRUTHY for v in vals)
        expected = bool(cond.get("expected", False))
        ok = truthy == expected
        return mk(
            "pass" if ok else "fail",
            f"{expected}",
            f"{f}={truthy}",
            (
                (
                    str(cond.get("pass_detail", "") or "")
                    or f"the flag holds the expected value ({expected})"
                )
                if ok
                else (
                    str(cond.get("fail_detail", "") or "")
                    or f"the flag is {truthy}, not the expected {expected}"
                )
            ),
        )

    if kind == "distinct_count":
        maxv = _declared_bound(cond)
        if maxv is None:
            return mk(
                "unknown",
                _NO_BOUND_EXPECTED,
                "not compared",
                "this counting condition declares no `max`, so there is nothing to "
                "compare the count against",
                own_detail=True,
            )
        # Primary (source, field) plus optional fallbacks; first that yields a value is used.
        candidates = [
            {"source": cond.get("source", ""), "field": cond.get("field", "")}
        ]
        candidates += cond.get("fallbacks", []) or []
        try:
            # `normalize` replaces the incumbent dedup key with one the PACK declares. It has
            # to reach the subject exclusion below through the same form: change the dedup key
            # alone and `exclude_subject` silently stops matching, inflating the count by one.
            norm_form = _projection_form(cond, "normalize")
        except FormError as exc:
            return mk(
                "unknown", f"<= {maxv}", "not compared", str(exc), own_detail=True
            )
        # Normalise for dedup, keep the original spelling for display.
        vals: Dict[str, str] = {}
        used = ""
        used_source = ""  # kept separate from `used` (display text) for truncation logic
        unresolvable: Set[str] = set()
        for cand in candidates:
            rows = src_rows.get(cand.get("source", ""), [])
            found: Dict[str, str] = {}
            unread: Set[str] = set()
            for v in _collect(rows, cand.get("field", "")):
                raw = str(v).strip()
                if not raw:
                    continue
                try:
                    key = _canonical_key(v, norm_form)
                except FormError as exc:
                    return mk(
                        "unknown",
                        f"<= {maxv}",
                        "not compared",
                        str(exc),
                        own_detail=True,
                    )
                if not key:
                    # Unresolvable under the declared form: excluded and reported, never
                    # counted as one shared value the form's own failure invented.
                    unread.add(raw)
                elif key not in found:
                    found[key] = raw
            if found:
                # Only under a declared form: the incumbent key drops an all-punctuation value
                # too, and reporting that as a form's failure would change every existing line.
                unresolvable = unread if norm_form else set()
                vals = found
                used = f"{cand.get('source', '')}.{cand.get('field', '')}"
                used_source = str(cand.get("source", "") or "")
                break
        # Which form produced the number is part of it, and so is what it could not read: `2
        # distinct` over 40 values of which 38 were unresolvable is not the finding it reads
        # as. Assembled before the early returns below, because the exclusion branch's `0` is
        # a reading the form produced too — it is what made the subject's value match.
        form_note = ""
        if norm_form:
            norm_name = str(cond.get("normalize", "") or "").strip()
            form_note = f" [under the {norm_name} form"
            if unresolvable:
                form_note += (
                    f"; {len(unresolvable)} value(s) unresolvable: "
                    f"{sorted(unresolvable)[:3]}"
                )
            form_note += "]"
        # `exclude_subject: true` subtracts the subject's own value(s) before comparing
        # against the bound, so `max: 0` means "nobody other than the subject". Aliases
        # (from `_merge_co_identified_subjects`) are included so all surface forms are excluded.
        subject_keys: Set[str] = set()
        if cond.get("exclude_subject"):
            try:
                subject_keys = {
                    _canonical_key(v, norm_form) or ""
                    for v in (cond.get("_subject_identity_values") or [])
                    if str(v).strip()
                }
            except FormError as exc:
                return mk(
                    "unknown", f"<= {maxv}", "not compared", str(exc), own_detail=True
                )
            subject_keys.discard("")
        excluded_subject = False
        if subject_keys and vals:
            kept = {k: v for k, v in vals.items() if k not in subject_keys}
            excluded_subject = len(kept) != len(vals)
            vals = kept
        # All collected values were the subject's own: count is 0 (nobody else), not unknown.
        if not vals and excluded_subject:
            ok = 0 <= maxv
            return mk(
                "pass" if ok else "fail",
                f"<= {maxv}",
                f"0 distinct, the subject's own value(s) excluded: []"
                f"{form_note}{f' [from {used}]' if used else ''}",
                str(cond.get("pass_detail" if ok else "fail_detail", "") or "")
                or (
                    "every value counted was the subject's own, so no other party is "
                    "implicated"
                ),
            )
        if not vals:
            # A keyed source returned empty rows: the count is 0, not unknown.
            if cond.get("_count_scope_resolved_empty"):
                ok = 0 <= maxv
                return mk(
                    "pass" if ok else "fail",
                    f"<= {maxv}",
                    "0 distinct: []",
                    str(cond.get("pass_detail" if ok else "fail_detail", "") or "")
                    or (
                        "the query asked about this scope and the source held nothing to "
                        "count"
                    ),
                )
            # Row_match narrowing emptied the source: the identity is not in it.
            # Only applies when `max: 0`; a higher bound presupposes rows to count among.
            if cond.get("_scope_resolved_empty") and maxv == 0:
                ok = 0 <= maxv
                return mk(
                    "pass" if ok else "fail",
                    f"<= {maxv}",
                    "0 distinct: [], the identity lookup found no row of its own",
                    str(cond.get("pass_detail" if ok else "fail_detail", "") or "")
                    or (
                        "the source answered and none of its rows belong to this identity, "
                        "so there is nothing of its own to count"
                    ),
                )
            return mk(
                "unknown",
                f"<= {maxv}",
                "no data",
                "the counted field returned no values on any candidate source",
            )
        n = len(vals)
        ok = n <= maxv
        src_note = f" [from {used}]" if used else ""
        # A count within the bound on a truncated read is unknown (a floor, not a total).
        # A count over the bound stands: more rows cannot mend a broken one.
        if ok and used_source in (cond.get("_count_truncated") or []):
            return mk(
                "unknown",
                f"<= {maxv}",
                f"{n} distinct in a TRUNCATED read: "
                f"{sorted(vals.values())[:5]}{form_note}{src_note}",
                str(cond.get("truncated_detail", "") or "")
                or (
                    "the counted source was cut off at its row cap, so a count within the "
                    "bound is a FLOOR and not the finding — the values that would exceed it "
                    "may be in the rows that never came back"
                ),
                own_detail=True,
            )
        excl_note = ""
        if subject_keys:
            excl_note = (
                ", the subject's own value(s) excluded and present"
                if excluded_subject
                else ", the subject's own value(s) excluded but not among them"
            )
        return mk(
            "pass" if ok else "fail",
            f"<= {maxv}",
            f"{n} distinct{excl_note}: {sorted(vals.values())[:5]}{form_note}{src_note}",
            (
                (
                    str(cond.get("pass_detail", "") or "")
                    or f"{n} distinct value(s), within {maxv}"
                )
                if ok
                else (
                    str(cond.get("fail_detail", "") or "")
                    or f"{n} distinct values, more than the {maxv} allowed"
                )
            ),
        )

    if kind == "numeric_compare":
        # Aggregate-then-compare: the check library declares WHAT is aggregated, the
        # importing ruleset declares the comparison. `bound` rather than `max` because
        # `max` states the wrong thing under `operator: ">="`.
        aggregate = str(cond.get("aggregate", "") or "").strip().lower()
        op = str(cond.get("operator", "") or "").strip()
        bound = _declared_number(cond, "bound")
        if aggregate not in _AGGREGATES or op not in _COMPARE_OPS or bound is None:
            return mk(
                "unknown",
                _NO_BOUND_EXPECTED,
                "not compared",
                "this comparison needs a readable `aggregate`, `operator` and `bound`; it "
                f"declares aggregate={aggregate or '(none)'!r}, operator={op or '(none)'!r}"
                f", bound={cond.get('bound')!r}",
                own_detail=True,
            )
        # A `baseline` makes `bound` a MULTIPLIER of a computed population value rather than
        # an absolute threshold. It rides on `bound` rather than a nested key of its own so
        # the one-level `use:` merge can still override it: a mapping is replaced wholesale,
        # so a nested multiplier could not be re-weighted without restating the mechanics.
        baseline = (
            cond.get("baseline") if isinstance(cond.get("baseline"), dict) else None
        )
        want = (
            f"{op} {_fmt_num(bound)}x baseline"
            if baseline
            else f"{op} {_fmt_num(bound)}"
        )
        group_by = str(cond.get("group_by", "") or "").strip()
        if group_by and op == "==":
            return mk(
                "unknown",
                want,
                "not compared",
                "`group_by` names no deciding group under `==` — the largest group answers "
                "an upper bound, not an equality",
                own_detail=True,
            )
        candidates = [
            {"source": cond.get("source", ""), "field": cond.get("field", "")}
        ]
        candidates += cond.get("fallbacks", []) or []
        value: Optional[float] = None
        direction = used = used_source = shape_note = ""
        try:
            # `normalize` counts equivalence classes instead of spellings. It reaches the
            # baseline through the same resolution, or the two sides of one comparison would
            # be taken over two different questions.
            norm_form = _projection_form(cond, "normalize")
        except FormError as exc:
            return mk("unknown", want, "not compared", str(exc), own_detail=True)
        for cand in candidates:
            logical = str(cand.get("source", "") or "")
            field = str(cand.get("field", "") or "")
            base = src_rows.get(logical, [])
            # No rows and nothing saying the source answered: a retrieval gap, not a zero.
            if not base and not (
                cond.get("_count_scope_resolved_empty") and aggregate in _ZERO_ON_EMPTY
            ):
                continue
            sel = apply_where(base, cond.get("where"))
            if aggregate == "ratio":
                # The denominator is the row set before `where`, so the clauses that
                # select the numerator are the same vocabulary every other kind uses.
                if not base:
                    continue
                value, direction = float(len(sel)) / float(len(base)), ""
                shape_note = f" ({len(sel)} of {len(base)} row(s))"
            elif group_by:
                try:
                    value, direction, top, n_groups, top_rows = _grouped_aggregate(
                        sel, field, aggregate, group_by, norm_form
                    )
                except FormError as exc:
                    return mk(
                        "unknown", want, "not compared", str(exc), own_detail=True
                    )
                if value is not None:
                    shape_note = f" [largest of {n_groups} group(s): {top}]"
                    shape_note += _modal_note(top_rows, field, aggregate, norm_form)
            else:
                try:
                    value, direction = _aggregate_value(
                        sel, field, aggregate, norm_form
                    )
                except FormError as exc:
                    return mk(
                        "unknown", want, "not compared", str(exc), own_detail=True
                    )
                if value is not None:
                    shape_note = _modal_note(sel, field, aggregate, norm_form)
            if value is None:
                continue
            used = f"{logical}.{field}" if field else logical
            used_source = logical
            break
        if value is None:
            return mk(
                "unknown",
                want,
                "no data",
                f"the {aggregate} could not be read on any candidate source",
            )
        threshold, base_note, base_refusal = bound, "", ""
        if baseline:
            try:
                threshold, base_note, base_refusal = _baseline_threshold(
                    cond, baseline, bound
                )
            except FormError as exc:
                # Reachable where the subject side never applied the form (a `count` against
                # a `distinct` baseline), so it cannot ride on the aggregate's own guard.
                return mk("unknown", want, "not compared", str(exc), own_detail=True)
        if base_refusal:
            # Never a comparison against a partial or degenerate baseline: an incomplete
            # population understates the threshold, so a subject that is in fact ordinary
            # reads as an outlier.
            return mk("unknown", want, "no baseline", base_refusal, own_detail=True)
        ok = bool(_compare(value, op, threshold))
        truncated = used_source in (cond.get("_count_truncated") or [])
        if truncated and aggregate in _MODAL_AGGREGATES and shape_note:
            # The frequency is monotone and can carry a `>` conclusion through a truncated
            # read; the winner's identity cannot, because the rows that never came back may
            # carry a different value entirely. So the number decides and the name is marked.
            shape_note = shape_note[:-1] + ", PROVISIONAL: the read was truncated]"
        # Which form produced the number is part of it: `4 distinct` counted over equivalence
        # classes and `4 distinct` counted over spellings are different findings.
        norm_name = str(cond.get("normalize", "") or "").strip()
        norm_note = (
            f" [under the {norm_name} form]"
            if norm_form and aggregate in _FORM_AGGREGATES
            else ""
        )
        observed = f"{aggregate} = {_fmt_num(value)}{shape_note}{norm_note}{base_note}"
        src_note = f" [from {used}]" if used else ""
        # A truncated read bounds the value rather than stating it, so the comparison is
        # reported only where the rows that never came back could not have flipped it.
        if truncated and not _holds_when(op, ok, direction):
            return mk(
                "unknown",
                want,
                f"{observed} in a TRUNCATED read{src_note}",
                str(cond.get("truncated_detail", "") or "")
                or (
                    "the aggregated source was cut off at its row cap, so this "
                    f"{aggregate} is a bound and not the finding — the rows that never "
                    "came back could carry the values that decide it"
                ),
                own_detail=True,
            )
        said = str(cond.get("pass_detail" if ok else "fail_detail", "") or "")
        return mk(
            "pass" if ok else "fail",
            want,
            f"{observed}{src_note}",
            said
            or (
                f"the {aggregate} is {_fmt_num(value)}, which is "
                f"{'' if ok else 'not '}{want}"
                + (f" ({op} {_fmt_num(threshold)})" if baseline else "")
            ),
        )

    if kind == "value_equivalence":
        # How many values are "the same thing" under a form the PACK declares. One question in
        # two framings: with an `anchor`, how many values are equivalent to it (a targeting
        # question); without one, how many fall into the largest equivalence class (a
        # collision). The count compares through the same operator/bound split as every other
        # bounded kind, so nothing about the weighting is new.
        name = str(cond.get("form", "") or "").strip()
        op = str(cond.get("operator", "") or "").strip()
        bound = _declared_number(cond, "bound")
        if not name or op not in _COMPARE_OPS or bound is None:
            return mk(
                "unknown",
                _NO_BOUND_EXPECTED,
                "not compared",
                "this comparison needs a `form` the pack declares, an `operator` and a "
                f"`bound`; it declares form={name or '(none)'!r}, "
                f"operator={op or '(none)'!r}, bound={cond.get('bound')!r}",
                own_detail=True,
            )
        want = f"{op} {_fmt_num(bound)}"
        logical = str(cond.get("source", "") or "")
        field = str(cond.get("field", "") or "")
        base = src_rows.get(logical, [])
        if not base and not cond.get("_count_scope_resolved_empty"):
            return mk(
                "unknown", want, "no data", f"the {logical} source returned no rows"
            )
        rows = apply_where(base, cond.get("where"))
        if rows and field and not _path_retrieved(rows, [field]):
            return mk(
                "unknown",
                want,
                "no data",
                f"no row carries {field}, so there are no values to compare — a projection "
                "gap, not an absence of equivalent values",
            )
        values = _collect(rows, field)
        anchor = cond.get("anchor") if isinstance(cond.get("anchor"), dict) else None
        try:
            form = _resolve_form(cond, "form")
            if anchor:
                anchor_rows = src_rows.get(str(anchor.get("source", "") or ""), [])
                anchor_keys, anchor_shown = [], []
                for av in _collect(anchor_rows, str(anchor.get("field", "") or "")):
                    ak = _canonical_key(av, form)
                    if ak and ak not in anchor_keys:
                        anchor_keys.append(ak)
                        anchor_shown.append(str(av).strip())
                if not anchor_keys:
                    return mk(
                        "unknown",
                        want,
                        "no anchor",
                        "the anchor carried no value this form can read, so there is nothing "
                        "for the other values to be equivalent TO",
                        own_detail=True,
                    )
                matched, unresolvable, comparisons = _anchored_matches(
                    values, anchor_keys, form
                )
                count = len(matched)
                readable = len(
                    {str(v).strip() for v in values if str(v or "").strip()}
                ) - len(set(unresolvable))
                shape = (
                    f"{count} of {readable} value(s) equivalent to "
                    f"{', '.join(sorted(anchor_shown)[:3])}"
                    + (f": {sorted(matched)[:5]}" if matched else "")
                )
            else:
                classes, unresolvable, comparisons = _equivalence_classes(values, form)
                readable = sum(len(m) for _, m in classes)
                count = len(classes[0][1]) if classes else 0
                shape = (
                    f"largest class {count} of {readable} value(s)"
                    + (f": {classes[0][0]} {classes[0][1][:5]}" if classes else "")
                    + (f", {len(classes)} class(es)" if len(classes) > 1 else "")
                )
        except FormError as exc:
            # A malformed form makes every reading taken under it wrong, so it reports the
            # authoring error rather than a count arrived at some other way.
            return mk(
                "unknown",
                want,
                "not compared",
                f"the {name!r} form could not be applied: {exc}",
                own_detail=True,
            )
        # A majority of unresolvable values changes what the count MEANS: a class of 4 drawn
        # from 60 values of which 55 the form could not read is not the finding it reads as.
        if unresolvable:
            shape += (
                f" [{len(unresolvable)} value(s) unresolvable under this form: "
                f"{sorted(set(unresolvable))[:3]}]"
            )
        if comparisons:
            shape += f" [{comparisons} pairwise comparison(s)]"
        ok = bool(_compare(float(count), op, bound))
        truncated = logical in (cond.get("_count_truncated") or [])
        if truncated and not _holds_when(op, ok, "up"):
            # A class can only GROW as rows arrive, so a `>` conclusion carries through a
            # truncated read and nothing else does.
            return mk(
                "unknown",
                want,
                f"{shape} in a TRUNCATED read",
                str(cond.get("truncated_detail", "") or "")
                or "the source was cut off at its row cap, so this is a floor on the "
                "equivalence and not the finding — the values that never came back could "
                "join the class",
                own_detail=True,
            )
        said = str(cond.get("pass_detail" if ok else "fail_detail", "") or "")
        return mk(
            "pass" if ok else "fail",
            want,
            f"{shape} [under the {name} form]",
            said
            or f"{count} value(s) under the {name} form, which is "
            f"{'' if ok else 'not '}{want}",
        )

    if kind == "route_membership":
        rows = src_rows.get(cond.get("source", ""), [])
        codes = _scope_point_codes(
            rows,
            cond.get("point_fields", []) or [],
            str(cond.get("point_pattern", "") or ""),
        )
        routes = cond.get("_routes", []) or []  # injected by evaluate_verdict
        if not codes:
            return mk(
                "unknown",
                "in-scope route",
                "no scope points",
                "no scope-point codes were found in the retrieved rows",
            )
        in_scope = next(((a, b) for a, b in routes if a in codes and b in codes), None)
        if in_scope:
            return mk(
                "pass",
                "in-scope route",
                f"{in_scope[0]}-{in_scope[1]}",
                str(cond.get("pass_detail", "") or "")
                or f"both endpoints of an in-scope pair are present ({in_scope[0]}-{in_scope[1]})",
            )
        return mk(
            "fail",
            "in-scope route",
            ", ".join(sorted(codes)[:8]),
            str(cond.get("fail_detail", "") or "")
            or "no declared in-scope pair has both endpoints in these rows",
        )

    if kind == "value_mismatch":
        # Two value-sets that should overlap; empty intersection => FAIL.
        # An optional `lookup` resolves the left values through pack data (an identifier
        # prefix -> the owning party) as a fallback when the direct left field is empty.
        # `unknown` when either side has no data or the prefix is unmappable.
        left = cond.get("left", {}) or {}
        right = cond.get("right", {}) or {}
        lvals = _collect(_side_rows(src_rows, left), left.get("field", ""))
        rvals = _collect(_side_rows(src_rows, right), right.get("field", ""))
        lookup = cond.get("lookup", {}) or {}
        via_lookup = False
        if lookup:
            # Map raw values to owner codes via pack data. Source is extracted entities
            # (`from_entity`) or a log field (`from`). Used as the left set.
            raw: List[Any] = []
            if lookup.get("from_entity"):
                raw = list(cond.get("_lookup_entities", []) or [])
            if not raw:
                fb = lookup.get("from", {}) or {}
                if fb:
                    raw = _collect(
                        src_rows.get(fb.get("source", ""), []), fb.get("field", "")
                    )
            table = cond.get("_lookup_data", {}) or {}
            prefix_len = int(lookup.get("prefix_len", 3))
            mapped: List[Any] = []
            for v in raw:
                key = str(v).strip().replace("-", "").replace(" ", "")[:prefix_len]
                hit = table.get(key)
                if hit is None:
                    continue
                mapped.extend(hit if isinstance(hit, list) else [hit])
            if mapped and not lvals:
                lvals = mapped
                via_lookup = True

        # Some backends return multi-value fields as JSON-array strings; expand them so
        # the comparison is against the values, not the string encoding.
        def _expand(vals: List[Any]) -> set:
            out: set = set()
            for v in vals:
                node = _maybe_json(v)
                if isinstance(node, list):
                    out.update(str(x).strip().upper() for x in node if str(x).strip())
                elif str(v).strip():
                    out.add(str(v).strip().upper())
            return out

        if not lvals or not rvals:
            return mk(
                "unknown",
                "value-sets overlap",
                f"left={[str(v) for v in lvals][:3]} right={[str(v) for v in rvals][:3]}",
                "one side of the comparison had no data",
            )
        lset = _expand(lvals)
        rset = _expand(rvals)
        overlap = lset & rset
        # The lookup label and pass note are pack-declared; the engine only supplies the
        # arithmetic so domain-specific meaning stays in the pack.
        via = f" [via {lookup.get('via_label') or 'lookup'}]" if via_lookup else ""
        obs = f"{sorted(lset)[:4]} vs {sorted(rset)[:4]}{via}"
        if overlap:
            return mk(
                "pass",
                "value-sets overlap",
                obs,
                str(cond.get("pass_detail", "") or "") or "the two value-sets intersect",
            )
        return mk(
            "fail",
            "value-sets overlap",
            obs,
            cond.get("fail_detail", "value-set mismatch (fraud indicator)"),
        )

    if kind == "value_matches_pattern":
        # Two directions: "forbidden" -> a matching value is suspicious (fail);
        # "allowed" -> a non-matching value is suspicious.
        # `data_map` mode reads a list-of-{key,value} structure.
        import re as _re

        # `_side_rows` applies the `where:` row selector before reading.
        # `reads:` unions values from several sources or row shapes; silent reads are
        # counted and named in the detail. Absent `reads:`, the condition is the read.
        reads = [r for r in (cond.get("reads", []) or []) if isinstance(r, dict)]
        vals: List[Any] = []
        answered: List[str] = []
        silent: List[str] = []
        for read in reads or [cond]:
            rows = _side_rows(src_rows, read)
            data_map = read.get("data_map", {}) or {}
            if data_map:
                got = _collect_kv(
                    rows,
                    data_map.get("field", ""),
                    data_map.get("match_key", ""),
                    data_map.get("key_name", "key"),
                    data_map.get("value_name", "value"),
                )
                which = str(data_map.get("field", "") or "")
            else:
                declared = read.get("fields", []) or []
                resolved, got = _first_present(rows, declared)
                # Use the resolved field when found, otherwise the full declared list
                # so a silent read is diagnosable.
                which = resolved or "|".join(str(f) for f in declared)
            # Named `where_read`, not `label`: `label` is the condition's display label
            # and shadowing it here replaces every report line with a field path.
            where_read = f"{read.get('source', '') or '?'}.{which or '?'}"
            if got:
                vals.extend(got)
                answered.append(where_read)
            else:
                silent.append(where_read)
        coverage = ""  # filled when some declared reads returned no values
        if reads and silent and answered:
            coverage = (
                f"{len(silent)} of {len(reads)} declared read(s) returned no values "
                f"({', '.join(silent)}), so this result covers only what was read"
            )

        def mkp(result, expected="", observed="", detail="", own_detail=False):
            """``mk`` plus the read-coverage clause; not appended on ``unknown``."""
            if coverage and result != "unknown":
                detail = f"{detail}; {coverage}" if detail else coverage
            return mk(result, expected, observed, detail, own_detail=own_detail)

        if not vals:
            return mkp(
                "unknown",
                cond.get("expected", "pattern"),
                (
                    f"field absent in all {len(reads)} declared read(s) "
                    f"({', '.join(silent)})"
                    if reads
                    else "field absent"
                ),
                "the checked field returned no values",
            )
        patterns = cond.get("patterns", []) or []
        if isinstance(patterns, str):
            patterns = [patterns]
        mode = cond.get("match_mode", "forbidden")
        # All comparisons in this kind are case-insensitive. No helper wraps these
        # flags: every comparison branch must carry them explicitly so none silently
        # becomes case-sensitive.
        flags = _re.IGNORECASE
        strvals = [str(v).strip() for v in vals if str(v).strip()]
        # `inconclusive_patterns`: values whose meaning is ambiguous between pass and fail.
        # Matching values are dropped before comparison; if that leaves nothing the result
        # is `unknown` so "vocabulary cannot express this" is distinguishable from "field absent".
        inconclusive = cond.get("inconclusive_patterns", []) or []
        if isinstance(inconclusive, str):
            inconclusive = [inconclusive]
        undecidable: List[str] = []
        if inconclusive:
            undecidable = [
                v
                for v in strvals
                if any(_re.search(p, v, flags) for p in inconclusive)
            ]
            strvals = [v for v in strvals if v not in undecidable]

        # `ordinary_patterns`: values the pack knows are neither forbidden nor inconclusive.
        # Declaring it closes the vocabulary: a value outside all three lists is withheld
        # rather than counted as a non-match. Without it (the default), unrecognised values
        # are treated as ordinary and nothing below changes.
        ordinary = cond.get("ordinary_patterns", []) or []
        if isinstance(ordinary, str):
            ordinary = [ordinary]
        unclassified: List[str] = []
        if ordinary:
            classified = list(patterns) + list(inconclusive) + list(ordinary)
            unclassified = [
                v
                for v in strvals
                if not any(_re.search(p, v, flags) for p in classified)
            ]
            strvals = [v for v in strvals if v not in unclassified]

        def _drop_observed() -> str:
            """Dropped values by bucket; inconclusive and unclassified are distinguished."""
            parts = []
            if undecidable:
                parts.append(", ".join(sorted(set(undecidable))[:4]))
            if unclassified:
                parts.append(
                    f"{', '.join(sorted(set(unclassified))[:4])} "
                    f"(outside every classification this check declares)"
                )
            return "; ".join(parts)

        def _drop_detail() -> str:
            """Detail sentence(s) for each non-empty bucket, in declaration order.

            ``unclassified_detail`` is kept separate from ``inconclusive_detail``:
            the latter explains a known code; reusing it for an unlisted value asserts
            a reason the pack never stated.
            """
            parts = []
            if undecidable:
                parts.append(
                    str(
                        cond.get(
                            "inconclusive_detail",
                            "the field's value does not distinguish the cases this "
                            "check tests",
                        )
                    )
                )
            if unclassified:
                parts.append(
                    str(
                        cond.get(
                            "unclassified_detail",
                            "the field holds a value this check's declared vocabulary does "
                            "not cover, so it can neither match nor be ruled out",
                        )
                    )
                )
            return "; ".join(p for p in parts if p)

        if not strvals and (undecidable or unclassified):
            # Guard on the two buckets, not on `strvals` alone: blanks-only returns
            # the "field absent" path, not this vocabulary path.
            return mkp(
                "unknown",
                cond.get("expected", "pattern"),
                _drop_observed(),
                _drop_detail(),
                own_detail=True,
            )

        # An inconclusive value may or may not be about this subject; the pack declares
        # which via `inconclusive_blocks_pass`. An unclassified value always blocks the
        # clear because the pack has said nothing about it and survivors are not the whole
        # of the evidence. A fail is unaffected in both cases.
        blocks_clear = bool(unclassified) or (
            bool(undecidable) and bool(cond.get("inconclusive_blocks_pass"))
        )

        def _undecided():
            """``unknown`` result when dropped values prevent a clear."""
            beside = f"beside {len(strvals)} value(s) that decide nothing either way"
            return mkp(
                "unknown",
                cond.get("expected", "pattern"),
                f"{_drop_observed()} "
                + (f"(unclassifiable, {beside})" if undecidable else f"({beside})"),
                _drop_detail(),
                own_detail=True,
            )

        if mode == "allowed":
            offending = [
                v for v in strvals if not any(_re.search(p, v, flags) for p in patterns)
            ]
            expected = cond.get("expected", "matches allow-list")
            if offending:
                return mkp(
                    "fail",
                    expected,
                    ", ".join(sorted(set(offending))[:4]),
                    cond.get(
                        "fail_detail", "value outside allow-list (fraud indicator)"
                    ),
                )
            if blocks_clear:
                return _undecided()
            # The pass names the allowed values (not just "all allowed") so a reader can see
            # what cleared the subject. Capped at four with the total beside the cap.
            allowed_seen = ", ".join(sorted(set(strvals))[:4])
            return mkp(
                "pass",
                expected,
                (
                    f"all {len(strvals)} value(s) allowed from "
                    f"{', '.join(answered)}: {allowed_seen}"
                    if reads
                    else f"all allowed: {allowed_seen}"
                ),
                str(cond.get("pass_detail", "") or "") or "no suspicious value",
            )
        # forbidden
        offending = [
            v for v in strvals if any(_re.search(p, v, flags) for p in patterns)
        ]
        expected = cond.get("expected", "no forbidden pattern")
        if offending:
            return mkp(
                "fail",
                expected,
                ", ".join(sorted(set(offending))[:4]),
                cond.get("fail_detail", "suspicious value present (fraud indicator)"),
            )
        if blocks_clear:
            return _undecided()
        return mkp(
            "pass",
            expected,
            # The pass names the count and source so the basis is traceable.
            (
                f"none in {len(strvals)} value(s) from {', '.join(answered)}"
                if reads
                else "none"
            ),
            str(cond.get("pass_detail", "") or "") or "no suspicious value",
        )

    if kind == "delimited_field_mismatch":
        # Compares two positional fields within each delimited value, not across the set.
        # `left_index` / `right_index` name the two positions.
        import re as _re

        rows = _side_rows(src_rows, cond)
        _, vals = _first_present(rows, cond.get("fields", []) or [])
        sep = str(cond.get("separator", "/") or "/")
        li = int(cond.get("left_index", 0))
        ri = int(cond.get("right_index", 0))
        expected = cond.get("expected", "the two fields agree")
        # Only test a value when both positions are non-empty; a short or sparse line
        # is a different record layout, not a mismatch.
        min_parts = max(li, ri) + 1
        pairs, skipped = [], 0
        for v in vals or []:
            parts = str(v).split(sep)
            if len(parts) < min_parts:
                skipped += 1
                continue
            lft, rgt = parts[li].strip(), parts[ri].strip()
            if not lft or not rgt:
                skipped += 1
                continue
            pairs.append((lft, rgt, str(v)))
        if not pairs:
            return mk(
                "unknown",
                expected,
                f"no value carries both fields ({len(vals or [])} value(s), {skipped} skipped)",
                cond.get(
                    "absent_detail",
                    "no record carried both of the compared fields, so nothing could be "
                    "compared — not a finding that they agree",
                ),
                own_detail=True,
            )
        # Normalise using pack-declared equivalences; unlisted values compare literally.
        aliases = {}
        for grp in cond.get("equivalent_values", []) or []:
            canon = sorted(str(g).strip().upper() for g in grp if str(g).strip())
            for g in canon:
                aliases[g] = canon[0]

        def _norm(x: str) -> str:
            u = _re.sub(r"[^A-Z0-9]", "", str(x).upper())
            return aliases.get(u, u)

        bad = [(a, b, raw) for a, b, raw in pairs if _norm(a) != _norm(b)]
        obs_skip = f", {skipped} value(s) without both fields" if skipped else ""
        if bad:
            return mk(
                "fail",
                expected,
                "; ".join(sorted({f"{a} vs {b}" for a, b, _ in bad})[:4])
                + f" ({len(bad)} of {len(pairs)}{obs_skip})",
                cond.get("fail_detail", "the two fields disagree (fraud indicator)"),
            )
        return mk(
            "pass",
            expected,
            f"all {len(pairs)} record(s) agree{obs_skip}",
            cond.get("pass_detail", "the two fields agree on every record"),
        )

    if kind == "velocity_count":
        # Distinct subject keys per actor within a window; > max is a burst (fail).
        # Bound must be declared; a missing `max` key returns unknown rather than a
        # fabricated default. See `_declared_bound`.
        maxv = _declared_bound(cond)
        if maxv is None:
            return mk(
                "unknown",
                _NO_BOUND_EXPECTED,
                "not compared",
                "this velocity condition declares no `max`, so no rate can be called a "
                "burst",
                own_detail=True,
            )
        rows = src_rows.get(cond.get("source", ""), [])
        subject_field = cond.get("subject_field", "")
        actor_field = cond.get("actor_field", "")
        if not rows or not subject_field or not actor_field:
            return mk(
                "unknown",
                f"<= {maxv} per actor",
                "no data",
                "velocity fields not present",
            )
        by_actor: Dict[str, set] = defaultdict(set)
        for r in rows:
            if not isinstance(r, dict):
                continue
            actors = _collect([r], actor_field)
            subs = _collect([r], subject_field)
            if not actors or not subs:
                continue
            akey = _norm_identifier(actors[0])
            for s in subs:
                if str(s).strip():
                    by_actor[akey].add(str(s).strip().upper())
        by_actor.pop("", None)
        if not by_actor:
            return mk(
                "unknown",
                f"<= {maxv} per actor",
                "no actor/subject pairs",
                "velocity fields empty",
            )
        top_actor, top_set = max(by_actor.items(), key=lambda kv: len(kv[1]))
        n = len(top_set)
        if n > maxv:
            return mk(
                "fail",
                f"<= {maxv} per actor",
                f"{top_actor}: {n} distinct ({sorted(top_set)[:6]})",
                cond.get(
                    "fail_detail", "burst of records by one actor (fraud indicator)"
                ),
            )
        return mk("pass", f"<= {maxv} per actor", f"{top_actor}: {n}", "no burst")

    # The three composites. Dispatched separately rather than as `kind in (...)` because
    # `pack_validate.condition_kinds()` derives the vocabulary by regex over this function,
    # and a kind it cannot see becomes an `unknown-condition-kind` error on every pack.
    if kind == "all_of":
        return mk(*_composite(cond, src_rows, "all_of", depth))

    if kind == "any_of":
        return mk(*_composite(cond, src_rows, "any_of", depth))

    if kind == "none_of":
        return mk(*_composite(cond, src_rows, "none_of", depth))

    if kind == "stub":
        # Placeholder for a check whose data path is not confirmed. Always unknown,
        # non-decisive; surfaces the intended-but-unavailable check in the report.
        return mk(
            "unknown",
            cond.get("expected", "planned check"),
            STUB_OBSERVED,
            cond.get("detail", "data path not yet available"),
            own_detail=True,
        )

    # An authoring error, not a deliberate gap like `stub`: without the log line a typo'd
    # `kind:` reads in the report exactly like a source that returned nothing.
    logger.warning(
        "Condition '%s' declares unrecognised kind '%s' — evaluated as unknown",
        cid,
        kind,
    )
    return mk(
        "unknown", "", "", f"unrecognised condition kind '{kind}'", own_detail=True
    )


def _path_retrieved(rows: List[Dict], fields: List[str]) -> bool:
    """True when any of ``fields`` was returned in the rows, even if empty.

    Distinguishes a column projected with no value from a column never projected.
    Uses the same three spelling strategies as value resolution (dotted, underscore alias,
    nested walk).
    """
    for r in rows:
        if not isinstance(r, dict):
            continue
        for f in fields or []:
            segs = str(f).split(".")
            if f in r:
                return True
            if any(
                _dict_get(r, "_".join(segs[:cut]))[0] for cut in range(len(segs), 0, -1)
            ):
                return True
            if _resolve_nodes(r, segs):
                return True
    return False


def _detect_platform_mode(rows: List[Dict], mode_spec: Dict[str, Any]) -> (str, str):
    """Which platform an asset was on, derived from a marker field. Returns ``(prose, class)``.

    ``class`` is the pack-declared id (or ``"unknown"``); ``prose`` is the human label.
    Both labels come from the pack (``marker_label``, ``present.label`` / ``absent.label``).

    An absent field is not an absent marker: a projection gap must not be read as a platform
    determination. A retrieved-but-null field may mean the sub-record was never written;
    the pack can declare ``absent.requires_present`` sibling paths that confirm the
    sub-record exists before the absent determination is made.
    """
    if not mode_spec:
        return "unknown", "unknown"
    fields = [f for f in (mode_spec.get("marker_fields", []) or []) if str(f).strip()]
    marker = str(mode_spec.get("marker_label", "") or "").strip() or "marker"
    present = mode_spec.get("present", {}) or {}
    absent = mode_spec.get("absent", {}) or {}
    if not fields:
        return f"unknown — the ruleset declares no {marker} field", "unknown"
    # A blank leaf is not a marker: an empty string means the column arrived carrying nothing.
    vals = [str(v).strip() for f in fields for v in _collect(rows, f) if str(v).strip()]
    if not vals:
        if not _path_retrieved(rows, fields):
            return (
                f"unknown — no {marker} field was retrieved (paths not in the "
                "projection: "
                + ", ".join(fields)
                + "), so the platform could not be determined",
                "unknown",
            )
        # An empty field is a real absence only if the record was written at all,
        # witnessed by any pack-named sibling carrying a value.
        witnesses = [
            w for w in (absent.get("requires_present", []) or []) if str(w).strip()
        ]
        if witnesses and not [
            v for w in witnesses for v in _collect(rows, w) if str(v).strip()
        ]:
            return (
                f"unknown — the {marker} field was retrieved but empty, and so was every "
                "field that would witness the record being written ("
                + ", ".join(witnesses)
                + "), so the platform could not be determined",
                "unknown",
            )
        # The field is there and empty: a real absence -> the absent-platform determination.
        label = str(absent.get("label", "") or "").strip()
        cls_id = str(absent.get("id", "") or "").strip() or "unknown"
        return (f"{label} (no {marker})" if label else f"no {marker}"), cls_id
    # A marker is present. Flag when its prefix is one the pack recognizes as an extra
    # confirmation, but presence alone is decisive.
    prefixes = [str(p) for p in (present.get("known_prefixes", []) or [])]
    value = vals[0]
    known = " [recognized prefix]" if any(value.startswith(p) for p in prefixes) else ""
    label = str(present.get("label", "") or "").strip()
    cls_id = str(present.get("id", "") or "").strip() or "unknown"
    return (
        f"{label} ({marker} {value}{known})" if label else f"{marker} {value}{known}"
    ), cls_id


def _classify_identity(identity: str, classes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """First pack-declared identity class whose pattern matches; ``{}`` if none."""
    s = _norm_identifier(identity)
    if not s:
        return {}
    for cls in classes or []:
        if not isinstance(cls, dict):
            continue
        for pat in cls.get("patterns", []) or []:
            try:
                if re.match(str(pat), s, re.IGNORECASE):
                    return cls
            except re.error:
                logger.warning("identity_classes: invalid pattern %r ignored", pat)
    return {}


def _finding(check, cond: Optional[Dict[str, Any]] = None) -> str:
    """Sentence stating what a check found, not what it required.

    Uses ``fail_detail`` (pack-declared, phrased as a finding) with the observed value
    appended. Falls back to the label prefixed with ``"FAILED:"`` so the negation is
    visible. ``cond`` may be ``None`` for engine-generated checks.
    """
    detail = str((cond or {}).get("fail_detail", "") or "").strip()
    if not detail:
        # Engine-generated per-kind detail is also finding-phrased, so it is usable here.
        detail = str(getattr(check, "detail", "") or "").strip()
    label = str(getattr(check, "label", "") or "") or str(
        getattr(check, "id", "") or ""
    )
    observed = str(getattr(check, "observed", "") or "").strip()
    text = detail or f"FAILED: {label}"
    return f"{text} ({observed})" if observed else text


def _finding_headline(check, cond: Optional[Dict[str, Any]] = None) -> str:
    """``_finding`` cut to the first sentence of its detail, for use in action steps.

    The cut is on the detail string before the observed value is appended, so the
    evidence is never lost. A condition with no ``fail_detail`` keeps the ``"FAILED:"``
    prefix from ``_finding``.
    """
    text = _finding(check, cond)
    observed = str(getattr(check, "observed", "") or "").strip()
    suffix = f" ({observed})" if observed else ""
    body = text[: -len(suffix)] if suffix and text.endswith(suffix) else text
    first = re.split(r"(?<=[.!?])\s+", body.strip(), maxsplit=1)[0].strip()
    head = first or body.strip()
    if suffix and head.endswith("."):
        head = head[:-1]  # the evidence follows, so the sentence has not ended yet
    return head + suffix


def _unevaluated_labels(checks: List[Any]) -> str:
    """Labels of every check where ``result == "unknown"``; ``"(none)"`` if all resolved.

    Filters on ``result``, not the literal string: real unevaluated checks
    carry the verdict in ``result``.
    Returns ``"(none)"`` rather than ``""`` because callers embed this mid-sentence.
    """
    labels = [
        str(getattr(c, "label", "") or getattr(c, "id", "") or "").strip()
        for c in (checks or [])
        if getattr(c, "result", "") == "unknown"
    ]
    return ", ".join(label for label in labels if label) or "(none)"


def _lock_provenance(
    lock_target: Dict[str, str], lock_spec: Optional[Dict[str, Any]] = None
) -> str:
    """``"<role> from <field>; ..."`` string noting where the named identity was read.

    Role words come from the pack (``scope_label`` / ``identity_label``), falling back to
    the generic role names the engine resolved.
    """
    spec = lock_spec or {}
    roles = {
        "scope_field": str(spec.get("scope_label", "") or "").strip().lower() or "scope",
        "identity_field": str(spec.get("identity_label", "") or "").strip().lower()
        or "identity",
    }
    return "; ".join(
        f"{roles[k]} from {lock_target[k]}"
        for k in ("scope_field", "identity_field")
        if lock_target.get(k)
    )


def _containment_block(
    lock_target: Dict[str, str], lock_spec: Optional[Dict[str, Any]] = None
) -> str:
    """Containment-identity block, or a "none nominated" line when no target was set.

    All labels come from the pack (``heading``, ``scope_label``, ``identity_label``,
    ``provenance_label``, ``class_label``, ``none_nominated``); fallbacks are generic
    role names.
    """
    lt = lock_target or {}
    spec = lock_spec or {}

    def word(key: str, default: str) -> str:
        return str(spec.get(key, "") or "").strip() or default

    if not (lt.get("scope") or lt.get("identity")):
        return word(
            "none_nominated", "No containment target is nominated by this verdict."
        )
    lines = [word("heading", "Containment target identified:")]
    if lt.get("scope"):
        lines.append(f"  {word('scope_label', 'Scope')}: {lt.get('scope', '')}")
    if lt.get("identity"):
        lines.append(
            f"  {word('identity_label', 'Identity')}: {lt.get('identity', '')}"
        )
    prov = _lock_provenance(lt, spec)
    if prov:
        lines.append(f"  {word('provenance_label', 'Read from')}: {prov}")
    if lt.get("identity_class"):
        lines.append(
            f"  {word('class_label', 'Identity class')}: {lt['identity_class']}"
        )
    return "\n".join(lines)


def _render_notification(
    tmpl: Dict[str, Any],
    subjects: List[SubjectVerdict],
    labels: Dict[str, str],
    lock_spec: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the pack notification template for each subject; join the drafts.

    Placeholders are filled from each subject verdict; unknown ones are left blank.
    Non-fatal; returns '' if no template is configured."""
    if not tmpl or not subjects:
        return ""
    body = tmpl.get("template", "") or ""
    fraud_label = labels.get("fraud", "VALID FRAUD")
    closings = {
        fraud_label: tmpl.get("closing_fraud", ""),
        labels.get("false_positive", "FALSE POSITIVE"): tmpl.get(
            "closing_false_positive", ""
        ),
        labels.get("insufficient", "INSUFFICIENT DATA"): tmpl.get(
            "closing_insufficient", ""
        ),
        # Every label the ruleset can emit needs a closing, or that exit renders a
        # notification with no rationale at all -- which reads as "no recommendation"
        # rather than "this exit has one". Keyed by label so a pack that declares
        # neither exit is unaffected (a "" key can never match a verdict).
        labels.get("reject", ""): tmpl.get("closing_reject", ""),
        labels.get("out_of_scope", ""): tmpl.get("closing_out_of_scope", ""),
    }
    closings.pop("", None)
    drafts = []
    for s in subjects:
        cond_summary = "\n".join(
            f"  - [{c.result.upper()}] {c.label}: {c.observed or c.detail}"
            for c in s.checks
        )
        # When fraud indicators are present on a fraud verdict, use the indicator closing.
        fraud_indicators = next(
            (
                n.split("fraud_indicators=")[-1]
                for n in s.notes
                if n.startswith("fraud_indicators=")
            ),
            "",
        )
        closing = closings.get(s.verdict, "")
        if (
            s.verdict == fraud_label
            and fraud_indicators
            and tmpl.get("closing_fraud_indicator")
        ):
            try:
                closing = tmpl.get("closing_fraud_indicator", "").replace(
                    "{fraud_indicators}", fraud_indicators
                )
            except Exception:
                closing = closings.get(s.verdict, "")
        fields = {
            "verdict": s.verdict,
            "subject": s.subject_value,
            "lock_scope": s.lock_target.get("scope", ""),
            "lock_identity": s.lock_target.get("identity", ""),
            "platform_mode": next(
                (
                    n.split("platform_mode=")[-1]
                    for n in s.notes
                    if n.startswith("platform_mode=")
                ),
                "",
            ),
            "asset_count": next(
                (
                    n.split("asset_count=")[-1]
                    for n in s.notes
                    if n.startswith("asset_count=")
                ),
                "",
            ),
            "actors": next(
                (n.split("actors=")[-1] for n in s.notes if n.startswith("actors=")), ""
            ),
            "route": next(
                (n.split("route=")[-1] for n in s.notes if n.startswith("route=")), ""
            ),
            "fraud_indicators": fraud_indicators,
            "decisive_exclusion": next(
                (
                    n.split("decisive_exclusion=")[-1]
                    for n in s.notes
                    if n.startswith("decisive_exclusion=")
                ),
                "",
            ),
            "scope_gate": next(
                (
                    n.split("scope_gate=")[-1]
                    for n in s.notes
                    if n.startswith("scope_gate=")
                ),
                "",
            ),
            # Filtered on result, not the literal string; see _unevaluated_labels.
            "unevaluated": _unevaluated_labels(s.checks),
            "condition_summary": cond_summary,
            "containment_action": str(s.lock_target.get("action", "") or ""),
            "containment_prerequisites": str(
                s.lock_target.get("prerequisites", "") or ""
            ),
            "identity_class": str(s.lock_target.get("identity_class", "") or ""),
            "lock_provenance": _lock_provenance(s.lock_target, lock_spec),
            "distribution": ", ".join(
                str(d) for d in (tmpl.get("distribution", []) or []) if str(d).strip()
            ),
            "containment_block": _containment_block(s.lock_target, lock_spec),
        }

        class _Blank(dict):
            def __missing__(self, k):  # unknown placeholder -> blank
                return ""

        # Expand the closing's own placeholders before substituting it into the body.
        try:
            fields["closing"] = closing.format_map(_Blank(fields))
        except Exception:
            fields["closing"] = closing
        try:
            drafts.append(body.format_map(_Blank(fields)))
        except Exception:  # a malformed template must never break the stage
            drafts.append("")
    return ("\n\n" + "=" * 60 + "\n\n").join(d for d in drafts if d)


def _as_of_boundary(analysis: Any) -> Optional[datetime]:
    """The end of the event window as the as-of cutoff; ``None`` when no window is set."""
    ev = getattr(analysis, "event_time", None)
    if ev is None:
        return None
    for attr in ("end", "start"):  # end preferred; a point window has both the same
        dt = _parse_ts(getattr(ev, attr, "") or "")
        if dt is not None:
            return _norm_dt(dt)
    return None


def _named_actors(analysis: Any, entity_types: List[str]) -> List[str]:
    """Identities the incident named, for the entity types the pack declares as actors.

    Empty when the pack declares no actor types or the incident extracted none.
    The emptiness is load-bearing: see ``_as_of_rows``.
    """
    wanted = {str(t).lower() for t in entity_types if str(t).strip()}
    if not wanted:
        return []
    out = [
        str(e.value)
        for e in (getattr(analysis, "extracted_entities", []) or [])
        if str(getattr(e, "type", "")).lower() in wanted and getattr(e, "value", "")
    ]
    return list(dict.fromkeys(out))


def _as_of_rows(
    rows: List[Dict],
    as_of_spec: Dict[str, Any],
    boundary: Optional[datetime],
    named_actors: Optional[List[str]] = None,
) -> tuple:
    """Rows minus versions a third party wrote after ``boundary``. Returns ``(rows, dropped, who)``.

    A row is excluded only on the intersection of two conditions: written after ``boundary``
    and by an identity the incident did not name. Time alone is insufficient: a version
    written after the alert by the subject itself is that subject's conduct and belongs in
    the adjudication. ``_identifiers_match`` accepts prefix matches.

    Omissions are kept, not dropped:
    - no ``timestamp_fields``, or unparseable timestamp: kept
    - no ``actor_fields``/``actor_entities``, or no writer on a row: kept

    When all rows would be excluded, the full unfiltered set is returned (``dropped == -1``).
    """
    ts_fields = [
        str(f) for f in (as_of_spec.get("timestamp_fields") or []) if str(f).strip()
    ]
    actor_fields = [
        str(f) for f in (as_of_spec.get("actor_fields") or []) if str(f).strip()
    ]
    actors = [a for a in (named_actors or []) if str(a).strip()]
    if not boundary or not ts_fields or not rows or not actor_fields or not actors:
        return rows, 0, []
    kept: List[Dict] = []
    dropped = 0
    third_parties: List[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        _f, vals = _first_present([r], ts_fields)
        ts = next((_parse_ts(v) for v in vals if _parse_ts(v) is not None), None)
        if ts is None or _norm_dt(ts) <= boundary:
            kept.append(r)
            continue
        _af, writers = _first_present([r], actor_fields)
        writers = [str(w).strip() for w in writers if str(w).strip()]
        if not writers or any(
            _identifiers_match(w, a) for w in writers for a in actors
        ):
            kept.append(r)  # named actor's own later conduct, or unattributable
            continue
        dropped += 1
        third_parties.extend(writers)
    # Return the full set when the boundary would exclude everything; zero rows reads as
    # "source had nothing" and the caller reports the fallback.
    if not kept:
        return rows, -1, sorted(set(third_parties))
    return kept, dropped, sorted(set(third_parties))


# -- the incident's own alert record --------------------------------------------------
# Shared by the verdict and the case builder; a second copy would give two answers to the
# same question.


def _entity_values(analysis: Any) -> Dict[str, List[str]]:
    """{entity_type_lower: [values]} from the understanding stage's extracted entities."""
    out: Dict[str, List[str]] = {}
    for e in getattr(analysis, "extracted_entities", []) or []:
        etype = str(getattr(e, "type", "") or "").lower()
        val = str(getattr(e, "value", "") or "").strip()
        if not etype or not val:
            continue
        bucket = out.setdefault(etype, [])
        if val not in bucket:
            bucket.append(val)
    return out


def _clause_values(row: Any, clause: Dict[str, Any]) -> List[str]:
    """Every value a clause's declared fields resolve to on one row/node.

    ``fields`` are candidate paths; all of them are read (not first-wins) because an alert
    payload commonly carries the same fact in a structured field and inside a free-text
    body, and either may be the one that is populated.
    """
    vals: List[str] = []
    for f in clause.get("fields", []) or []:
        for v in resolve_path(row, str(f)):
            s = str(v).strip()
            if s and s not in vals:
                vals.append(s)
    return vals


def _values_match(found: str, want: str, normalize: str = "") -> bool:
    """Compare an observed value to a declared one under the clause's normalisation.

    Modes: ``"identifier"`` uses prefix-tolerant ``_identifiers_match``; ``"id_suffix"``
    strips non-alphanumerics and checks mutual containment (min length 6); default is
    case-insensitive equality or containment of the declared value in the observed one.
    """
    if normalize == "identifier":
        return _identifiers_match(found, want)
    a, b = str(found).strip(), str(want).strip()
    if not a or not b:
        return False
    if normalize == "id_suffix":
        na = re.sub(r"[^A-Za-z0-9]", "", a).upper()
        nb = re.sub(r"[^A-Za-z0-9]", "", b).upper()
        if not na or not nb:
            return False
        if na == nb:
            return True
        shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
        return len(shorter) >= 6 and shorter in longer
    if a.casefold() == b.casefold():
        return True
    return b.casefold() in a.casefold()  # free-text body may contain the value


def _match_row(row: Any, clause: Dict[str, Any], values: List[str]) -> str:
    """The declared value this row carries for a clause, or "" when it carries none."""
    normalize = str(clause.get("normalize", "") or "")
    found = _clause_values(row, clause)
    for want in values:
        for got in found:
            if _values_match(got, want, normalize):
                return want
    return ""


def _label_row(row: Dict, fields: List[str]) -> str:
    """A short human label for an alert record (its ids), for naming it in prose."""
    parts: List[str] = []
    for f in fields:
        for v in resolve_path(row, str(f)):
            s = str(v).strip()
            if s and s not in parts:
                parts.append(s)
                break
    return " / ".join(parts)


def _incident_alert_rows(
    rows: List[Dict],
    alert_record: Dict[str, Any],
    ents: Dict[str, List[str]],
) -> Tuple[List[Dict], List[str]]:
    """Rows that satisfy all ``required`` identify clauses for this incident -> (kept, dropped).

    Clauses whose entity the incident never extracted are skipped (same rule as
    ``build_alert_facts``). If all rows would be dropped, none are: zero rows on the alert
    source is an undetectable failure. ``dropped`` carries each excluded row's label.
    """
    clauses = [
        c
        for c in (alert_record.get("identify", []) or [])
        if isinstance(c, dict) and c.get("required")
    ]
    live = [c for c in clauses if ents.get(str(c.get("from_entity", "") or "").lower())]
    if not live or not rows:
        return rows, []
    label_fields = [str(f) for f in (alert_record.get("label_fields", []) or [])]
    kept: List[Dict] = []
    dropped: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            kept.append(row)
            continue
        ok = True
        for clause in live:
            values = ents.get(str(clause.get("from_entity", "") or "").lower(), [])
            if not _match_row(row, clause, values):
                ok = False
                break
        if ok:
            kept.append(row)
        else:
            dropped.append(_label_row(row, label_fields) or "(unlabelled)")
    if not kept:
        return rows, []
    return kept, dropped


def evaluate_verdict(
    spec: Dict[str, Any],
    logs: Dict[str, List[Dict]],
    analysis: Any,
    entity_map: Optional[Dict[str, Dict[str, str]]] = None,
    pack_data: Optional[Dict[str, Any]] = None,
    row_caps: Optional[Dict[str, int]] = None,
    keyed_sources: Optional[Dict[str, bool]] = None,
    unanswered_sources: Optional[Dict[str, str]] = None,
) -> Optional[ValidationVerdict]:
    """Evaluate a pack validation ruleset over the retrieved rows -> ValidationVerdict.

    Pure and deterministic (no LLM, no IO). ``spec`` is a ``ruleset_spec()`` dict; ``logs``
    is the retrieved rows keyed by real source name; ``analysis`` supplies the subject
    entity values. Returns ``None`` when the ruleset's sources aren't present in the data
    (nothing to validate). Never raises; the caller wraps it best-effort.
    """
    if not spec or not logs:
        return None
    labels = spec.get("labels", {}) or {}
    # No default: a missing key degrades to the whole-set fallback but under a misleading label.
    subject_entity = str(spec.get("subject_entity", "") or "")

    # Map the ruleset's logical source names to real sources present in logs.
    logical_to_real = {}
    for logical, real in (spec.get("sources", {}) or {}).items():
        if real in logs:
            logical_to_real[logical] = real
    # Sources asked but unanswered: absent from `logs` like unasked sources, but the
    # distinction matters for operators. Keyed by logical name; value is the reason.
    unanswered_logical: Dict[str, str] = {}
    for logical, real in (spec.get("sources", {}) or {}).items():
        reason = str((unanswered_sources or {}).get(real, "") or "").strip()
        if reason and logical not in logical_to_real:
            unanswered_logical[logical] = reason
    # Sources present in the spec but neither answered nor asked; conditions on them
    # must not claim an empty answer (the source was never queried).
    unasked_logical: Dict[str, str] = {
        logical: (spec.get("sources", {}) or {}).get(logical, logical)
        for logical in (spec.get("sources", {}) or {})
        if logical not in logical_to_real and logical not in unanswered_logical
    }
    unanswered_notes: List[str] = [
        f"source_unanswered={(spec.get('sources', {}) or {}).get(logical, logical)}: "
        f"this procedure declares it and it {reason} — its rows are MISSING, not empty, so "
        "every condition below that reads it is unresolved rather than answered. Re-run "
        "retrieval for this source (or narrow its scope) before reading an absence here as a "
        "finding."
        for logical, reason in sorted(unanswered_logical.items())
    ]
    if not logical_to_real:
        if unanswered_logical:
            logger.warning(
                "No verdict: every source '%s' declares is missing, and %d of them were "
                "asked and did not answer (%s).",
                spec.get("label_scheme", "") or "this ruleset",
                len(unanswered_logical),
                "; ".join(f"{k}: {v}" for k, v in sorted(unanswered_logical.items())),
            )
        return None
    real_rows = {
        logical: (logs.get(real) or []) for logical, real in logical_to_real.items()
    }

    # As-of filtering applies to conditions only; the chronology, scope sweep and
    # containment assessment read `logs` untouched so they include post-alert conduct.
    as_of_at = _as_of_boundary(analysis)
    as_of_specs = spec.get("as_of", {}) or {}
    as_of_notes: List[str] = []
    for logical, as_of_spec in as_of_specs.items():
        if logical not in real_rows or not isinstance(as_of_spec, dict):
            continue
        before = len(real_rows[logical])
        rows, dropped, third_parties = _as_of_rows(
            real_rows[logical],
            as_of_spec,
            as_of_at,
            _named_actors(analysis, as_of_spec.get("actor_entities") or []),
        )
        real_rows[logical] = rows
        real = logical_to_real.get(logical, logical)
        who = f" (written by {', '.join(third_parties)})" if third_parties else ""
        if dropped < 0:
            logger.warning(
                "As-of boundary %s excluded every row of '%s' (%d rows) — the full "
                "version chain was adjudicated instead.",
                as_of_at.isoformat() if as_of_at else "?",
                real,
                before,
            )
            as_of_notes.append(
                f"as_of_fallback={real}: all {before} version(s) were written after the "
                f"incident by another party{who}, so none could be excluded without leaving "
                f"the source empty and the whole chain was adjudicated — the versions read "
                f"may include changes made in response to the incident"
            )
        elif dropped:
            logger.info(
                "As-of %s: adjudicated %d of %d version(s) of '%s' (%d written later by "
                "another party: %s).",
                as_of_at.isoformat() if as_of_at else "?",
                len(rows),
                before,
                real,
                dropped,
                ", ".join(third_parties) or "?",
            )
            as_of_notes.append(
                f"as_of={real}: {len(rows)} of {before} version(s) adjudicated; {dropped} "
                f"were written after the incident by an identity it did not name{who} and "
                f"are excluded from the verdict as response rather than conduct (they remain "
                f"in the chronology and the containment assessment). The named actor's own "
                f"later changes ARE adjudicated."
            )

    # Narrow the alert source to this incident's own records. The chronology and sweep
    # read `logs` untouched; other incidents' alerts are reported as background.
    alert_scope_notes: List[str] = []
    _ar = spec.get("alert_record", {}) or {}
    _alert_logical = str(_ar.get("source", "") or "")
    if _alert_logical in real_rows:
        _before = len(real_rows[_alert_logical])
        _kept, _dropped = _incident_alert_rows(
            real_rows[_alert_logical], _ar, _entity_values(analysis)
        )
        if _dropped:
            _alert_real = logical_to_real.get(_alert_logical, _alert_logical)
            logger.info(
                "Alert scope: adjudicated %d of %d record(s) of '%s' — %d belong to another "
                "incident (%s).",
                len(_kept),
                _before,
                _alert_real,
                len(_dropped),
                ", ".join(_dropped[:8]),
            )
            alert_scope_notes.append(
                f"alert_scope={_alert_real}: {len(_kept)} of {_before} record(s) "
                f"adjudicated; {len(_dropped)} do NOT carry this incident's identifiers "
                f"({', '.join(_dropped[:8])}) and are excluded from the verdict as another "
                "incident's alert. They remain in the chronology as background activity, and "
                "nothing below is a finding about them."
            )
            real_rows[_alert_logical] = _kept

    # Subject values: extracted entities of the subject type; else distinct values in the
    # primary (first) logical source scanned via the pack/entity-map field.
    subjects = [
        e.value
        for e in (getattr(analysis, "extracted_entities", []) or [])
        if str(getattr(e, "type", "")).lower() == subject_entity.lower() and e.value
    ]
    subjects = list(dict.fromkeys(subjects))  # dedupe, preserve order

    # Merge co-identified subjects. The surviving subject must match rows under dropped
    # values too (kept in `aliases`), or the merge would delete the evidence it licensed.
    co_identity = spec.get("_co_identity") or {}
    # The pack's text-equivalence forms, attached by `ruleset_spec` for the same reason as
    # `_co_identity`: this function receives pack_data rather than the pack.
    equivalence_forms = spec.get("_equivalence_forms") or {}
    co_notes: List[str] = []
    aliases: Dict[str, List[str]] = {}
    if co_identity and len(subjects) > 1:
        subject_forms = {
            e.value: str(getattr(e, "value_form", "") or "")
            for e in (getattr(analysis, "extracted_entities", []) or [])
            if str(getattr(e, "type", "")).lower() == subject_entity.lower() and e.value
        }
        # Use `logs` (all sources), not `real_rows` (ruleset sources only), because the
        # co-identity licence may come from any retrieved source. `entity_map` is keyed by
        # real source name, not logical.
        subjects, co_notes, aliases = _merge_co_identified_subjects(
            subjects, subject_forms, co_identity, logs, entity_map
        )

    # When the incident named no subjects, try to read them from retrieved rows.
    # A ruleset that declares no subject_discovery is unaffected.
    discovered_entities: Dict[str, Dict[str, List[str]]] = {}
    discovered_tuples: Dict[str, List[Dict[str, str]]] = {}
    discovery_notes: List[str] = []
    if not subjects:
        (
            subjects,
            discovered_entities,
            discovered_tuples,
            discovery_notes,
        ) = _discovered_subjects(
            spec.get("subject_discovery") or {}, real_rows, subject_entity
        )
    else:
        # Subject was named; read co-located entity values from its own rows.
        (
            discovered_entities,
            discovered_tuples,
            discovery_notes,
        ) = _named_subject_companions(
            spec.get("subject_discovery") or {},
            real_rows,
            subject_entity,
            subjects,
            aliases,
            {
                str(getattr(e, "type", "")).lower()
                for e in (getattr(analysis, "extracted_entities", []) or [])
                if str(getattr(e, "value", "") or "").strip()
            },
        )

    if not subjects:
        subjects = [""]  # fallback: evaluate the whole set as one subject

    routes = [
        (str(r[0]).upper(), str(r[1]).upper())
        for r in (spec.get("routes", []) or [])
        if isinstance(r, (list, tuple)) and len(r) >= 2
    ]
    lock_spec = spec.get("lock_target", {}) or {}
    mode_spec = spec.get("platform_mode", {}) or {}
    conditions = spec.get("conditions", []) or []
    # Lookup by id for the rollup, which needs spec properties (gate, order) not on the check.
    _cond_by_id = {
        str(c.get("id", "")): c
        for c in conditions
        if isinstance(c, dict) and c.get("id")
    }

    subject_verdicts: List[SubjectVerdict] = []
    any_degraded = False
    for sv in subjects:
        # Per-logical-source rows filtered to this subject.
        src_rows = {
            logical: _rows_for_subject(rows, sv, aliases.get(sv))
            for logical, rows in real_rows.items()
        }

        def _clause_values(ent_type: str) -> List[str]:
            """Entity values for this subject, in priority order.

            Priority: subject's own value + alias forms (when the type matches the subject
            entity), then co-located values read from this subject's rows, then all
            extracted values for the type.
            """
            vals = [
                e.value
                for e in (getattr(analysis, "extracted_entities", []) or [])
                if str(getattr(e, "type", "")).lower() == str(ent_type).lower() and e.value
            ]
            if sv and str(ent_type).lower() == subject_entity.lower():
                vals = [sv] + list(aliases.get(sv) or [])
            else:
                own = (discovered_entities.get(sv) or {}).get(str(ent_type))
                if own:
                    vals = list(own)
            return list(dict.fromkeys(v for v in vals if str(v).strip()))

        def _preprocess(
            cond_in: Dict[str, Any], subject_scoped: Optional[bool] = None
        ) -> Dict[str, Any]:
            """Stamp the runtime facts a condition's kind reads, recursing into ``children``.

            ``subject_scoped`` records whether this condition's rows are THIS subject's, and
            it is resolved from the top-level condition and then inherited: the evaluation
            loop below reads `subject_scope` off the root only, so a child is scoped the way
            its parent is however the child itself is declared.

            Every fact below is derived from THIS condition's own sources, and a composite's
            children are stamped by the same recursion rather than inheriting the parent's
            copy. That is the difference between run-level context (the resolved routes, the
            pack's lookup data, the subject's identity values — per-run or per-subject, so a
            child re-derives the same thing) and a source-derived fact (truncation, a keyed
            source that resolved empty, the scope note). Inheriting the second kind hands a
            child a SIBLING's truncation and turns a sound check into `unknown` — a false
            INSUFFICIENT DATA, the failure this engine is least able to see.
            """
            c = dict(cond_in)
            if subject_scoped is None:
                subject_scoped = cond_in.get("subject_scope") is not False
            # Stamp a scope note when the source was asked but did not answer, so the
            # check reports a retrieval gap rather than "no data". Stamped before any
            # kind-specific handling; skipped if narrowing logic has already set one.
            _dead = [
                (spec.get("sources", {}) or {}).get(logical, logical)
                for logical in _condition_sources(c)
                if logical in unanswered_logical
            ]
            if _dead and not c.get("_scope_note"):
                c["_scope_note"] = (
                    f"the {', '.join(sorted(set(_dead)))} source was asked and did not "
                    "answer, so this check has no rows to read — that is a retrieval gap, "
                    "not an absence of evidence in the source"
                )
            elif not c.get("_scope_note"):
                _unasked = sorted(
                    {
                        real
                        for logical in _condition_sources(c)
                        for real in [unasked_logical.get(logical)]
                        if real
                    }
                )
                if _unasked:
                    c["_scope_note"] = (
                        f"the {', '.join(_unasked)} source was NOT QUERIED on this run, so "
                        "this check has no rows to read — an absent query, never an empty "
                        "answer. Add it to the retrieval plan (the run controls' plan editor) "
                        "and re-run if this check matters to the outcome"
                    )
                else:
                    # Every source this condition reads answered, and answered with rows —
                    # but none of them is THIS subject's. That is a third thing, and the two
                    # notes above cannot say it: an absent field and an absent subject read
                    # identically off one `unknown`, while their remedies are opposite (fix
                    # the projection vs. widen the scope). Requires rows for somebody, or the
                    # claim "the rows are other subjects'" is false about an empty answer.
                    # And requires `subject_scoped`, or it is false in the other direction: a
                    # cohort condition is scoped by its own query and reads the source WHOLE,
                    # so none of its rows naming this subject is the normal state and the
                    # rows it did read are real.
                    _answered = [lg for lg in _condition_sources(c) if lg in src_rows]
                    _elsewhere = sum(len(real_rows.get(lg) or []) for lg in _answered)
                    if (
                        sv
                        and subject_scoped
                        and _answered
                        and _elsewhere
                        and not any(src_rows.get(lg) for lg in _answered)
                    ):
                        _named = ", ".join(
                            sorted(
                                {
                                    (spec.get("sources", {}) or {}).get(lg, lg)
                                    for lg in _answered
                                }
                            )
                        )
                        c["_scope_note"] = (
                            f"the {_named} source answered with {_elsewhere} row(s) on this "
                            f"run and NOT ONE of them names {sv}, so this check has no rows "
                            "of its own to read — a scope gap, not a missing field. Either "
                            "this subject is absent from the source or the run's window and "
                            "filters excluded it; widening the retrieval scope is what "
                            "answers that, and re-reading the projection cannot"
                        )
            if equivalence_forms:
                # Run-level context, so it inherits down the recursion: the pack's forms are
                # the same for every condition. Stamped only when the pack declares some, or a
                # pack that declares none would get a condition dict it never had.
                c["_equivalence_forms"] = equivalence_forms
            if c.get("kind") == "route_membership":
                c["_routes"] = routes
            if c.get("kind") == "value_mismatch" and c.get("lookup"):
                lk = c.get("lookup", {}) or {}
                table_name = lk.get("data")
                if table_name and pack_data:
                    raw = pack_data.get(table_name, {}) or {}
                    # A data file may nest the map under a top-level key (file stem echo);
                    # accept both {prefix: owner} and {map: {prefix: owner}}.
                    if (
                        isinstance(raw, dict)
                        and "map" in raw
                        and isinstance(raw["map"], dict)
                    ):
                        raw = raw["map"]
                    c["_lookup_data"] = {str(k): v for k, v in (raw or {}).items()}
                ent_type = lk.get("from_entity")
                if ent_type:
                    c["_lookup_entities"] = [
                        e.value
                        for e in (getattr(analysis, "extracted_entities", []) or [])
                        if str(getattr(e, "type", "")).lower() == str(ent_type).lower()
                        and e.value
                    ]
            # For the counting kinds: set `_count_scope_resolved_empty` when the source
            # returned zero rows, and the query was keyed (so zero means "not found", not "no
            # data"). Rows-but-no-values stays unknown; unkeyed emptiness proves nothing.
            if c.get("kind") in ("distinct_count", "numeric_compare"):
                empty_and_keyed = True
                for logical in _condition_sources(c):
                    if real_rows.get(logical) or not (keyed_sources or {}).get(
                        logical_to_real.get(logical, "")
                    ):
                        empty_and_keyed = False
                        break
                if empty_and_keyed:
                    c["_count_scope_resolved_empty"] = True
                # Resolve per-subject exclusion values: includes the adjudicated form and
                # all alias forms, since the counted column may store a different form.
                # `distinct_count` only — `numeric_compare` does not read `exclude_subject`
                # (pack_validate errors on it there rather than letting it read as honoured).
                if (
                    c.get("kind") == "distinct_count"
                    and c.get("exclude_subject")
                    and sv
                ):
                    c["_subject_identity_values"] = [sv] + [
                        a for a in (aliases.get(sv) or []) if str(a).strip()
                    ]
            # A read at its row cap bounds the finding rather than stating it, and the three
            # kinds below each carry their own rule for when the bound still decides (see
            # `_holds_when` and the quantifier asymmetry in `event_order`). Stored as a list,
            # not a flag, because only the source that supplied the values matters; a
            # truncated fallback that was never consulted should not hedge a complete read.
            if c.get("kind") in ("distinct_count", "numeric_compare", "event_order"):
                truncated = [
                    logical
                    for logical in _condition_sources(c)
                    if (row_caps or {}).get(logical_to_real.get(logical, ""), 0)
                    and len(real_rows.get(logical, []) or [])
                    >= int((row_caps or {}).get(logical_to_real.get(logical, ""), 0))
                ]
                if truncated:
                    c["_count_truncated"] = truncated
            if c.get("kind") == "numeric_compare" and isinstance(
                c.get("baseline"), dict
            ):
                # The baseline population is deliberately ABSENT from `_condition_sources`:
                # every narrowing built on that list scopes rows to the subject, and a
                # population the subject has been filtered out of is not a baseline. So its
                # rows and its completeness are stamped separately here, and read from the
                # baseline's OWN source — a sibling's truncation says nothing about it.
                b_logical = str((c.get("baseline") or {}).get("source", "") or "")
                b_real = logical_to_real.get(b_logical, "")
                b_rows = list(real_rows.get(b_logical, []) or [])
                b_cap = int((row_caps or {}).get(b_real, 0) or 0)
                c["_baseline_rows"] = b_rows
                if not b_logical:
                    c["_baseline_gap"] = "names no source"
                elif b_logical in unanswered_logical:
                    c["_baseline_gap"] = (
                        f"was asked and did not answer ({b_real or b_logical})"
                    )
                elif unasked_logical.get(b_logical):
                    c["_baseline_gap"] = (
                        f"was NOT QUERIED on this run ({unasked_logical[b_logical]})"
                    )
                elif b_cap and len(b_rows) >= b_cap:
                    c["_baseline_gap"] = f"was cut off at its {b_cap}-row cap"
                elif not b_rows:
                    c["_baseline_gap"] = f"returned no rows ({b_real or b_logical})"
            if c.get("kind") == "cohort_membership":
                logical = str(c.get("source", "") or "")
                real = logical_to_real.get(logical, "")
                cap = (row_caps or {}).get(real, 0)
                # Cap is checked against the unfiltered rows, not the subject-narrowed ones.
                got = len(real_rows.get(logical, []) or [])
                if cap and got >= int(cap):
                    c["_cohort_truncated"] = True
                # Resolve `from_entity` clauses in `subject_rows.where` against this subject's
                # own values via `_clause_values`, so a multi-subject incident gives each
                # subject its own anchor rather than every subject sharing all entity values.
                subj_sel = dict(c.get("subject_rows", {}) or {})
                filled = []
                anchor_gap: List[str] = []
                for clause in subj_sel.get("where", []) or []:
                    if not isinstance(clause, dict):
                        continue
                    clause = dict(clause)
                    ent_type = clause.pop("from_entity", "")
                    if ent_type and not clause.get("any_of"):
                        clause["any_of"] = _clause_values(ent_type)
                        if not clause["any_of"] and ent_type not in anchor_gap:
                            anchor_gap.append(str(ent_type))
                    filled.append(clause)
                if anchor_gap:
                    c["_subject_anchor_unresolved"] = anchor_gap
                if filled:
                    subj_sel["where"] = filled
                    c["subject_rows"] = subj_sel
            kids = [k for k in (c.get("children") or []) if isinstance(k, dict)]
            if kids:
                c["children"] = [_preprocess(k, subject_scoped) for k in kids]
            return c

        checks: List[ConditionCheck] = []
        for cond in conditions:
            c = _preprocess(cond)

            def _scoped_rows(
                c_local: Dict[str, Any], override: Dict[str, str]
            ) -> Dict[str, List[Dict]]:
                """Rows re-scoped to the acting identity for one assignment of its values.

                ``override`` pins one entity type to a single value; other types resolve
                normally via ``_clause_values``.
                """
                picked = {
                    str(k).strip().lower(): str(v)
                    for k, v in (override or {}).items()
                    if str(v or "").strip()
                }

                def _values(ent_type: str) -> List[str]:
                    one = picked.get(str(ent_type).strip().lower())
                    return [one] if one else _clause_values(ent_type)

                out_rows = {**src_rows}
                # `row_match` re-scopes to the acting identity; values come from extracted
                # entities so the ruleset stays free of literals.
                match_spec = []
                for clause in c_local.get("row_match", []) or []:
                    if not isinstance(clause, dict):
                        continue
                    ent_type = clause.get("from_entity", "")
                    # `_values` uses this subject's own value when the clause names the
                    # subject entity, its co-located value for other types, or the override
                    # for quantified passes. Aliases ride along so a co-identified subject
                    # does not go silent on evidence proved to belong to it.
                    match_spec.append({**clause, "values": _values(ent_type)})
                for logical in _condition_sources(c_local):
                    rows = real_rows.get(logical, [])
                    if match_spec:
                        # Truncation guard: a pair match against a capped source is not
                        # evidence of absence; the identity may sit in the unseen rows.
                        real = logical_to_real.get(logical, "")
                        cap = (row_caps or {}).get(real, 0)
                        truncated = bool(cap) and len(rows) >= cap
                        had_rows = bool(rows)
                        # A keyed query answers "not on the list" with zero rows; `had_rows`
                        # alone would read that as missing data.
                        keyed = bool((keyed_sources or {}).get(real))
                        rows, unresolved = _rows_matching(rows, match_spec)
                        if truncated and not rows:
                            c_local["_scope_note"] = (
                                f"the {logical_to_real.get(logical, logical)} source was "
                                f"TRUNCATED at its {cap}-row cap, so not finding the acting "
                                "identity in what was returned does NOT mean it is absent "
                                "from the full list — treat as no evidence, not as a clear"
                            )
                        elif not unresolved and (had_rows or keyed) and not rows:
                            # The lookup ran and found no match; that absence is a genuine answer.
                            c_local["_scope_resolved_empty"] = True
                        if unresolved:
                            # An identity the incident never supplied cannot be looked up.
                            # Report `unknown` (via an empty row set) rather than letting the
                            # unscoped rows stand in for the actor we failed to identify.
                            rows = []
                            c_local["_scope_note"] = (
                                "could not be scoped to the acting identity — the incident "
                                f"supplied no {', '.join(unresolved)} value, so the "
                                f"{logical} rows (which cover other identities) were not "
                                "used"
                            )
                    out_rows[logical] = rows
                return out_rows

            def _evaluated(
                c_local: Dict[str, Any], rows_local: Dict[str, List[Dict]]
            ) -> ConditionCheck:
                """Evaluate one condition; catch and report any exception as unknown."""
                try:
                    return _eval_condition(c_local, rows_local)
                except Exception as e:  # one bad condition must not sink the verdict
                    logger.warning(
                        "Verdict condition '%s' failed: %s", c_local.get("id"), e
                    )
                    return ConditionCheck(
                        id=str(c_local.get("id", "?")),
                        label=str(c_local.get("label", "")),
                        result="unknown",
                        detail=f"evaluation error: {e}",
                        decisive=bool(c_local.get("decisive", False)),
                        group=str(c_local.get("report_group", "") or ""),
                    )

            def _element_scoped(
                c_local: Dict[str, Any], rows_local: Dict[str, List[Dict]]
            ) -> Tuple[Dict[str, List[Dict]], str]:
                """Narrow rows to this subject's entries of the discovery node.

                Returns ``(rows, note)``. Note is non-empty when narrowing was skipped or found
                nothing, because ``unknown`` from a narrow is distinct from ``unknown`` from a
                missing source. Silent when every entry already belongs to this subject.
                """
                if str(c_local.get("subject_scope") or "").strip().lower() != "element":
                    return rows_local, ""
                decl = spec.get("subject_discovery") or {}
                node_path = str(decl.get("path") or "") or "subject"
                targets = [
                    logical
                    for logical in _condition_sources(c_local)
                    if logical and logical == str(decl.get("source") or "")
                ]
                if not targets:
                    return rows_local, (
                        "`subject_scope: element` could not be applied — this check reads no "
                        "source that the ruleset discovers its subjects from, so there is no "
                        "per-identity entry to narrow to and the values below are read as they "
                        "came"
                    )
                subject_values = [
                    v for v in [sv] + list(aliases.get(sv) or []) if str(v or "").strip()
                ]
                if not subject_values:
                    # No identity was discovered; the checks run over all records.
                    return rows_local, (
                        "`subject_scope: element` had no identity to narrow to — no acting "
                        "identity was discovered on this run, so the checks are asked of all "
                        "retrieved records and the values below are pooled across every "
                        "identity on the record"
                    )
                out = dict(rows_local)
                kept = total = 0
                had = 0
                for logical in targets:
                    had += len(out.get(logical) or [])
                    narrowed, k, t = _subject_elements_only(
                        out.get(logical) or [], decl, subject_values
                    )
                    if t < 0:
                        return rows_local, (
                            "`subject_scope: element` could not be applied — the ruleset's "
                            "subject_discovery declares no path or no subject field to narrow "
                            "on, so the values below are pooled across every identity on the "
                            "record"
                        )
                    out[logical] = narrowed
                    kept += k
                    total += t
                if not had:
                    return out, ""
                if not total:
                    return out, (
                        f"could not be narrowed to this identity's own entries — `{node_path}` "
                        f"resolved to no entry on the {had} row(s) read (a transposed projection "
                        "sends one array per leaf, which carries no per-entry identity), so the "
                        "values below are pooled across every identity on the record"
                    )
                if not kept:
                    return out, (
                        f"this identity is on NONE of the {total} `{node_path}` entries the "
                        "record carries, so they cannot be read for it — an answer taken from "
                        "them would be another identity's"
                    )
                if kept == total:
                    return out, ""
                return out, (
                    f"read within the {kept} of {total} `{node_path}` entr"
                    f"{'y' if kept == 1 else 'ies'} this identity is on — the other "
                    f"{total - kept} belong to other identities on the same record"
                )

            # When a subject has several acting (unit, identity) pairs, evaluating the
            # union asks "is any pair on the list", which lets one listed member clear the
            # others. The correct question is the conjunction: "was every acting identity
            # of this kind". So the condition is evaluated once per pair and rolled up.
            # A ruleset without `per_acting_identity` reaches `_scoped_rows(c, {})` unchanged.
            acting: List[Dict[str, str]] = []
            if cond.get("subject_scope") is False and cond.get("per_acting_identity"):
                acting = [t for t in (discovered_tuples.get(sv) or []) if t]
            if acting and len(acting) > 1:
                per: List[Tuple[Dict[str, str], ConditionCheck]] = []
                for combo in acting:
                    c_one = dict(c)
                    per.append((combo, _evaluated(c_one, _scoped_rows(c_one, combo))))
                checks.append(_rollup_acting_identities(per, str(cond.get("id", "?"))))
            else:
                cond_src_rows = src_rows
                if cond.get("subject_scope") is False:
                    cond_src_rows = _scoped_rows(c, acting[0] if acting else {})
                cond_src_rows, element_note = _element_scoped(c, cond_src_rows)
                chk = _evaluated(c, cond_src_rows)
                if element_note:
                    # Append to `detail`, not `_scope_note`: narrowing can change a pass, not
                    # only produce an unknown, so the note must appear on any result.
                    chk = chk.model_copy(
                        update={
                            "detail": (
                                f"{chk.detail}; {element_note}"
                                if chk.detail
                                else element_note
                            )
                        }
                    )
                checks.append(chk)

        # Rollup precedence:
        #   0) categorical exclusion fail -> false positive (beats indicators)
        #   1) decisive indicator fail, or >= indicator_threshold non-decisive -> valid fraud
        #   2) decisive exclusion fail -> false positive
        #   3) decisive unknown -> insufficient data
        #   4) all decisive pass -> `no_exclusion_fired` exit (default: fraud)
        #
        # Step 0 exists because categorical exclusions establish attributed facts (a named
        # action, a confirmed identity), not inferences about record shape. Once such a fact
        # is on the record the behavioural indicators lose their meaning.
        # Step 4 is a pack declaration because "all exclusions passed" implies fraud only
        # where those exclusions enumerate the fraud fingerprint. A procedure whose decisive
        # exclusion establishes only identity should declare `no_exclusion_fired: false_positive`.
        # The reclassification belongs in the pack, not as a fourth tier here: an
        # operator-written element is a categorical fact; an inference about element counts is not.
        # `exclusion_kind` defaults to "heuristic", so every existing condition is unaffected.
        indicators = [c for c in checks if c.polarity == "fraud_indicator"]
        exclusions = [c for c in checks if c.polarity != "fraud_indicator"]
        decisive_excl = [c for c in exclusions if c.decisive]
        categorical_fail = [
            c
            for c in decisive_excl
            if c.result == "fail"
            and getattr(c, "exclusion_kind", "heuristic") == "categorical"
        ]
        decisive_ind_fail = any(c.decisive and c.result == "fail" for c in indicators)
        nondecisive_ind_fails = [
            c for c in indicators if not c.decisive and c.result == "fail"
        ]
        # `indicator_threshold` is required: a missing key means indicators cannot vote
        # (a weaker verdict beats a confident one nobody authorised). Decisive indicators
        # are unaffected. `pack_validate` errors on the omission.
        indicator_threshold = _declared_bound(spec, "indicator_threshold")
        corroborated = (
            indicator_threshold is not None
            and len(nondecisive_ind_fails) >= indicator_threshold
        )
        unweighed_indicators = (
            len(nondecisive_ind_fails) if indicator_threshold is None else 0
        )
        # A categorical exclusion removes the basis for the behavioural indicators.
        fraud_confirmed = (decisive_ind_fail or corroborated) and not categorical_fail

        has_excl_fail = any(c.result == "fail" for c in decisive_excl)
        has_decisive_unknown = any(c.result == "unknown" for c in decisive_excl)

        # A scope gate (`gate: scope`) asks whether the procedure applies at all; a gate fail
        # exits with `out_of_scope`, not a false-positive verdict. The pack names the gate.
        gate_fail = [
            c
            for c in checks
            if _cond_by_id.get(c.id, {}).get("gate") == "scope" and c.result == "fail"
        ]
        gate_unknown = [
            c
            for c in checks
            if _cond_by_id.get(c.id, {}).get("gate") == "scope"
            and c.result == "unknown"
        ]
        # `verdict_class` is the engine branch (false_positive, fraud, etc.); `verdict` is
        # the pack's label. Consumers cannot re-derive the class from `checks` because two
        # branches turn on facts a check does not carry.
        unevaluated_clear = ""
        # `no_fire_exit`: note for a clean no-exclusion-fired exit.
        # `no_fire_partial`: separate wording when some checks were unknown, so a re-run
        # might still change the outcome.
        no_fire_exit = ""
        no_fire_partial = ""
        if gate_fail and labels.get("out_of_scope"):
            verdict, verdict_class = labels["out_of_scope"], "out_of_scope"
        elif categorical_fail:
            verdict = labels.get("false_positive", "FALSE POSITIVE")
            verdict_class = "false_positive"
        elif fraud_confirmed:
            verdict, verdict_class = labels.get("fraud", "VALID FRAUD"), "fraud"
        elif has_excl_fail:
            verdict = labels.get("false_positive", "FALSE POSITIVE")
            verdict_class = "false_positive"
        elif has_decisive_unknown:
            verdict = labels.get("insufficient", "INSUFFICIENT DATA")
            verdict_class = "insufficient"
        else:
            # Step 4: no decisive exclusion or indicator determined the verdict.
            declared_exit = spec.get("no_exclusion_fired", "") or "fraud"
            exit_key = str(declared_exit).strip().lower()
            if exit_key not in ("fraud", "false_positive", "insufficient"):
                logger.warning(
                    "Ruleset declares no_exclusion_fired=%r, which is not one of "
                    "fraud / false_positive / insufficient; using 'fraud', the exit this "
                    "branch has always taken.",
                    exit_key,
                )
                exit_key = "fraud"
            # `min_evaluated_to_clear` gates the false_positive exit: a clear requires at
            # least this many conditions to have produced pass or fail. Default is 1.
            # Stubs count against the denominator separately because they can never be
            # answered by retrieval; the same counts feed the no-fire note below.
            evaluated = [c for c in checks if c.result in ("pass", "fail")]
            attempted = [c for c in checks if c.observed != STUB_OBSERVED]
            unwired = len(checks) - len(attempted)
            floor = spec.get("min_evaluated_to_clear", 1)
            try:
                floor = max(1, int(floor))
            except (TypeError, ValueError):
                floor = 1
            if exit_key == "false_positive" and len(evaluated) < floor:
                unevaluated_clear = (
                    f"{len(evaluated)} of {len(attempted)} condition(s) could be evaluated, "
                    f"below the {floor} this procedure requires before an account is "
                    "cleared, so no verdict is reached on the merits"
                )
                if unwired:
                    unevaluated_clear += (
                        f" (a further {unwired} declare no data path at all, so no re-run "
                        "can answer them)"
                    )
                exit_key = "insufficient"
            if exit_key == "insufficient" and not unevaluated_clear:
                unresolved = len(attempted) - len(evaluated)
                # State what the branch established: no decisive exclusion failed and
                # indicators did not reach the threshold. Non-decisive exclusion fails and
                # sub-threshold indicators are named rather than denied.
                fired_no_vote = [
                    c for c in exclusions if not c.decisive and c.result == "fail"
                ]
                if not nondecisive_ind_fails:
                    ind_clause = "no fraud indicator fired"
                elif indicator_threshold is None:
                    ind_clause = (
                        f"the {len(nondecisive_ind_fails)} fraud indicator(s) that DID fire "
                        "cannot be weighed, because this procedure declares no threshold "
                        "to count them against"
                    )
                else:
                    ind_clause = (
                        f"{len(nondecisive_ind_fails)} fraud indicator(s) fired, short of "
                        f"the {indicator_threshold} this procedure requires"
                    )
                head = (
                    f"{len(evaluated)} of {len(attempted)} condition(s) were evaluated and "
                    "nothing that can decide this subject fired — no decisive exclusion "
                    f"established a reason to close it, and {ind_clause}. "
                )
                if fired_no_vote:
                    head += (
                        f"{len(fired_no_vote)} condition(s) DID fail while voting nothing in "
                        "this procedure ("
                        + ", ".join(sorted(str(c.id) for c in fired_no_vote))
                        + "), so read those as findings and not as an absence. "
                    )
                head += (
                    "The outcome is declared as no verdict rather than as a clear, so the "
                    "PASSes below are the absence of a finding and not a finding of "
                    "absence. "
                )
                if unresolved > 0:
                    tail = (
                        f"The other {unresolved} did not resolve, so retrieval IS still a gap "
                        "here: what is stated above is that nothing which answered fired, not "
                        "that everything answered"
                    )
                else:
                    tail = (
                        "Retrieval is not the gap here and a re-run answers nothing"
                    )
                note_text = head + tail
                if unwired:
                    note_text += (
                        f" (a further {unwired} declare no data path at all, so no re-run "
                        "can answer them either)"
                    )
                if unresolved > 0:
                    no_fire_partial = note_text
                else:
                    no_fire_exit = note_text
            default_label = {
                "fraud": "VALID FRAUD",
                "false_positive": "FALSE POSITIVE",
                "insufficient": "INSUFFICIENT DATA",
            }[exit_key]
            verdict = labels.get(exit_key, default_label)
            verdict_class = exit_key
        # `degraded` is true only when a decisive unknown drives the outcome, not when a
        # decisive fail or indicator already made the verdict terminal.
        degraded = (
            (has_decisive_unknown or bool(unevaluated_clear))
            and not fraud_confirmed
            and not has_excl_fail
        )
        any_degraded = any_degraded or degraded
        has_decisive_fail = has_excl_fail  # kept for back-compat

        # Containment target: identity a fraud verdict nominates for action.
        # Read from the logical source the pack declares in `lock_target.source`.
        lock_source = str(lock_spec.get("source", "") or "")
        lock_rows = src_rows.get(lock_source, [])
        scope_field = lock_spec.get("scope_field", "")
        identity_field = lock_spec.get("identity_field", "")
        lock_scope = (_collect(lock_rows, scope_field) or [""])[0]
        lock_identity = (_collect(lock_rows, identity_field) or [""])[0]
        # The platform decides which action set applies, so it is resolved with the target
        # rather than left as prose in a note for a reader to act on. Same declared-source
        # rule: `platform_mode.source` names the logical source.
        mode_rows = src_rows.get(str(mode_spec.get("source", "") or ""), [])
        platform_mode, platform_class = _detect_platform_mode(mode_rows, mode_spec)
        lock_target = {}
        if lock_scope or lock_identity:
            lock_target = {
                "scope": str(lock_scope or ""),
                "identity": str(lock_identity or ""),
                "source": logical_to_real.get(lock_source, ""),
                "scope_field": str(scope_field or ""),
                "identity_field": str(identity_field or ""),
            }
            # Role-word labels (scope_label, identity_label) are stamped on the target so
            # all three consumers (notification, report, brief) read from one place.
            for _key in ("scope_label", "identity_label"):
                _word = str(lock_spec.get(_key, "") or "").strip()
                if _word:
                    lock_target[_key] = _word
            action_map = lock_spec.get("actions", {}) or {}
            action = str(action_map.get(platform_class, "") or "")
            # Stamped only when the ruleset declares `platform_mode`; absent otherwise so
            # consumers can tell "undetermined" from "not a dimension this procedure uses".
            if mode_spec:
                lock_target["platform"] = platform_class
            if action:
                lock_target["action"] = action
            cls = _classify_identity(
                lock_identity, spec.get("identity_classes", []) or []
            )
            if cls:
                lock_target["identity_class"] = str(cls.get("id", "") or "")
                if cls.get("action"):
                    lock_target["action"] = str(cls["action"])
                if cls.get("rationale"):
                    lock_target["action_rationale"] = str(cls["rationale"])
            prereqs = [
                str(p)
                for p in (lock_spec.get("prerequisites", []) or [])
                if str(p).strip()
            ]
            if prereqs:
                lock_target["prerequisites"] = " | ".join(prereqs)

        # Notes: pack-declared summary facts, used by the template and report.
        notes: List[str] = []
        # Name the decisive exclusion(s) driving a false-positive verdict, ordered by
        # pack `order` (evidential rank), then declaration order.
        if verdict == labels.get("false_positive", "FALSE POSITIVE"):
            excl_fails = [c for c in decisive_excl if c.result == "fail"]
            excl_fails.sort(
                key=lambda c: int(_cond_by_id.get(c.id, {}).get("order", 100) or 100)
            )
            if excl_fails:
                # Uses _finding (not label) because a fail negates an exclusion's label.
                notes.append(
                    "decisive_exclusion="
                    + "; ".join(_finding(c, _cond_by_id.get(c.id)) for c in excl_fails)
                )
        if gate_fail:
            notes.append(
                "scope_gate=This procedure does not apply: "
                + "; ".join(_finding(c, _cond_by_id.get(c.id)) for c in gate_fail)
                + ". The remaining conditions were not weighed, so this is NOT an "
                "adjudication of the subject on its merits."
            )
        elif gate_unknown:
            # A decisive gate that is unknown drives the verdict; a non-decisive one is a
            # caveat. The note states which.
            _decisive_ids = {c.id for c in decisive_excl}
            _gate_decided = verdict_class == "insufficient" and any(
                c.id in _decisive_ids for c in gate_unknown
            )
            notes.append(
                "scope_gate=Scope could not be established ("
                + "; ".join(c.label for c in gate_unknown)
                + (
                    "), and this procedure cannot be concluded without it, so no verdict "
                    "is reached on the merits."
                    if _gate_decided
                    else "), so the verdict below rests on the other conditions only."
                )
            )
        if unevaluated_clear:
            notes.append(f"evidence_floor={unevaluated_clear}")
        if no_fire_exit:
            notes.append(f"no_verdict_reason={no_fire_exit}")
        if no_fire_partial:
            notes.append(f"no_verdict_partial={no_fire_partial}")
        if unweighed_indicators:
            notes.append(
                f"indicator_threshold=undeclared, {unweighed_indicators} "
                "corroborating indicator(s) fired and were not weighed"
            )
        # Platform note only when the ruleset declares `platform_mode`.
        if mode_spec:
            notes.append(f"platform_mode={platform_mode}")
        notes.extend(as_of_notes)
        notes.extend(alert_scope_notes)
        notes.extend(unanswered_notes)
        # Co-identity merge: on the absorbing subject only.
        notes.extend(n for n in co_notes if n.startswith(f"co_identity={sv} "))
        # Subject provenance: where a subject came from (discovery vs named) and any cap.
        notes.extend(
            n
            for n in discovery_notes
            if n.startswith("subject_cap=")
            or n.startswith(f"subject_discovered={sv} ")
            or n.startswith(f"subject_companions={sv} ")
        )
        # Asset count + acting identities: source and fields are pack-declared (`asset_notes:`).
        asset_spec = spec.get("asset_notes", {}) or {}
        asset_rows = src_rows.get(str(asset_spec.get("source") or ""), [])
        assets = {
            (
                _norm_identifier(v)
                if asset_spec.get("normalize") == "identifier"
                else str(v).strip()
            )
            for f in (asset_spec.get("asset_id_fields") or [])
            for v in _collect(asset_rows, f)
            if str(v).strip()
        } - {""}
        if assets:
            notes.append(f"asset_count={len(assets)}")
            _asset_word = str(asset_spec.get("asset_label", "") or "").strip()
            if _asset_word:
                notes.append(f"asset_label={_asset_word}")
        acting = sorted(
            {
                _norm_identifier(v)
                for f in (asset_spec.get("actor_fields") or [])
                for v in _collect(asset_rows, f)
                if str(v).strip()
            }
            - {""}
        )
        if acting:
            notes.append(f"actors={', '.join(acting)}")
            if len(acting) > 1:
                notes.append(
                    "Multiple acting "
                    f"{asset_spec.get('actor_label', 'identities')} on the issued "
                    f"{asset_spec.get('asset_label', 'asset')}s: {', '.join(acting)}."
                )
        route_check = next((c for c in checks if c.id == "route_in_scope"), None)
        if route_check and route_check.result == "pass":
            notes.append(f"route={route_check.observed}")
        # A terminal false-positive with other decisive checks unknown: note the gap
        # without implying a re-run would change the verdict.
        if has_decisive_fail and has_decisive_unknown and not fraud_confirmed:
            unresolved = ", ".join(
                c.label for c in decisive_excl if c.result == "unknown"
            )
            _why = (
                "because their source did not answer"
                if unanswered_logical
                else "because their source returned no rows"
            )
            notes.append(
                "data_coverage=The FALSE POSITIVE rests on the observed decisive "
                "exclusion above and is final. Some corroborating checks could not be "
                f"evaluated {_why} ({unresolved}); this does not affect the verdict."
            )
        if categorical_fail:
            overridden = [c for c in indicators if c.result == "fail"]
            # Exclusions use _finding (fail negates a label); indicators use plain labels
            # (fail affirms them). The asymmetry is polarity, not an inconsistency.
            drivers = ", ".join(
                _finding(c, _cond_by_id.get(c.id)) for c in categorical_fail
            )
            note = (
                f"categorical_exclusion={drivers}. This settles WHO acted, so it "
                "outranks the behavioural fraud indicators"
            )
            if overridden:
                note += (
                    " — which did fail ("
                    + ", ".join(c.label for c in overridden)
                    + "); read them against the exclusion above rather than as an "
                    "independent finding"
                )
            notes.append(note + ".")
        if fraud_confirmed:
            ind_fails = [c for c in indicators if c.result == "fail"]
            if ind_fails:
                notes.append(
                    "fraud_indicators="
                    + ", ".join(
                        f"{c.label} ({c.observed})" if c.observed else c.label
                        for c in ind_fails
                    )
                )
            decisive_driver = next(
                (c for c in indicators if c.decisive and c.result == "fail"), None
            )
            if decisive_driver is not None:
                notes.append(f"decisive_indicator={decisive_driver.label}")
            elif corroborated:
                notes.append(
                    f"corroborated_indicators={len(nondecisive_ind_fails)} "
                    f">= threshold {indicator_threshold}"
                )

        # Attribution gate: withheld when the target scope is not one the incident named.
        # Anchored on the incident's extracted values (not retrieved rows) so a retrieved
        # scope that was itself derived from a wrong query cannot self-validate.
        # Silent when the pack binds no field or the incident extracted no matching type.
        if lock_target:
            _scope_val = str(lock_target.get("scope", "") or "").strip()
            _scope_fld = str(lock_target.get("scope_field", "") or "").strip()
            _lock_real = str(lock_target.get("source", "") or "")
            _bound = (entity_map or {}).get(_lock_real, {}) or {}
            _types = [
                str(t)
                for t, f in _bound.items()
                if _scope_fld and str(f or "") == _scope_fld
            ]
            _declared_scopes: List[str] = []
            for _t in _types:
                _declared_scopes.extend(
                    str(getattr(e, "value", "") or "").strip()
                    for e in (getattr(analysis, "extracted_entities", []) or [])
                    if str(getattr(e, "type", "")).lower() == _t.lower()
                    and getattr(e, "value", "")
                )
            _declared_scopes = [s for s in dict.fromkeys(_declared_scopes) if s]
            if _scope_val and _declared_scopes:
                if not any(_identifiers_match(_scope_val, d) for d in _declared_scopes):
                    notes.append(
                        f"containment_withheld=The evidence's "
                        f"{lock_target.get('scope_label', 'scope')} "
                        f"({_scope_val}) is not one this incident named "
                        f"({', '.join(_declared_scopes[:5])}), so the retrieved rows cannot be "
                        f"attributed to the alerted actor and NO containment target is "
                        f"nominated. Treat this as a retrieval-scoping defect to investigate: "
                        f"the identity may be a different person carrying the same identifier."
                    )
                    lock_target = {}
                    any_degraded = True

        # Gate containment on verdict: which labels may carry a target is pack-declared
        # via `containment_labels` (default: fraud label only).
        containment_labels = spec.get("containment_labels")
        if containment_labels is None:
            containment_labels = [labels.get("fraud", "VALID FRAUD")]
        containment_labels = [str(x) for x in containment_labels if str(x).strip()]
        if lock_target and verdict not in containment_labels:
            # The detail sentence is pack-declared (`withheld_detail`); `{identity}` is
            # substituted with str.replace, never .format.
            withheld = str(lock_spec.get("withheld_detail", "") or "").strip()
            _who = lock_target.get("identity", "")
            _where = lock_target.get("scope", "")
            identity = f"{_who} @ {_where}".strip(" @")
            detail = (
                withheld.replace("{identity}", identity)
                if withheld
                else (
                    f"The identity recorded in the evidence ({identity}) is not nominated "
                    "for action."
                )
            )
            notes.append(
                f"containment_withheld=No containment target is carried on a "
                f"'{verdict}' verdict. {detail}"
            )
            lock_target = {}

        subject_verdicts.append(
            SubjectVerdict(
                subject_type=subject_entity or "subject",
                subject_value=sv or "(all retrieved records)",
                verdict=verdict,
                verdict_class=verdict_class,
                checks=checks,
                lock_target=lock_target,
                notes=notes,
            )
        )

    # Cross-subject exit: `reject_when: no_subject_validates` routes the alert back to
    # the detector owner when no subject produced verdict-bearing evidence. Requires
    # `reject` label in pack. As written this exit is unreachable because `decided` covers
    # all per-subject verdict labels; the guard is left rather than removed so it can be
    # wired properly when retrieval state is available to distinguish a routing decision
    # from a retrieval blackout.
    reject_label = labels.get("reject", "")
    fraud_label_x = labels.get("fraud", "VALID FRAUD")
    fp_label_x = labels.get("false_positive", "FALSE POSITIVE")
    insuff_x = labels.get("insufficient", "INSUFFICIENT DATA")
    oos_x = labels.get("out_of_scope", "\0")
    decided = {fraud_label_x, fp_label_x, insuff_x, oos_x}
    if (
        reject_label
        and subject_verdicts
        and str(spec.get("reject_when", "")).strip() == "no_subject_validates"
        and all(s.verdict not in decided for s in subject_verdicts)
    ):
        for s in subject_verdicts:
            s.verdict = reject_label
            s.verdict_class = "reject"  # must move with the label or the two contradict
            s.notes.append(
                "reject_reason=No alerted subject produced verdict-bearing evidence, so "
                "this ruleset cannot adjudicate the alert and it is returned to the "
                "detector's owner for routing rather than closed on a guess."
            )

    # One-line rollup across subjects.
    counts: Dict[str, int] = defaultdict(int)
    for s in subject_verdicts:
        counts[s.verdict] += 1
    summary = "; ".join(f"{v}: {n}" for v, n in counts.items())

    notification = _render_notification(
        spec.get("notification", {}) or {},
        subject_verdicts,
        labels,
        spec.get("lock_target", {}) or {},
    )
    groups = [
        ConditionGroup(
            id=str(g.get("id", "") or ""),
            title=str(g.get("title", "") or ""),
            description=str(g.get("description", "") or ""),
            role=(
                str(g.get("role", "")).lower()
                if str(g.get("role", "")).lower() in ("gate", "validation", "hint")
                else "validation"
            ),
        )
        for g in (spec.get("condition_groups", []) or [])
        if isinstance(g, dict) and str(g.get("id", "") or "").strip()
    ]
    return ValidationVerdict(
        label_scheme=str(spec.get("label_scheme", "") or ""),
        subjects=subject_verdicts,
        notification_draft=notification,
        summary=summary,
        degraded=any_degraded,
        condition_groups=groups,
        labels={str(k): str(v) for k, v in labels.items() if str(v).strip()},
    )


#: `basis` values, in the order a reader should think about them. Only `no_match` means the
#: caller's fallback to the pack default is a guess rather than a selection.
SELECTION_BASES = ("pinned", "scored", "sole_spec", "no_match", "no_specs")

#: Runners-up carried for the operator to pin. Enough to choose from, short enough to print.
_MAX_SELECTION_CANDIDATES = 4


@dataclass(frozen=True)
class SelectionBasis:
    """How this run's procedure was chosen, and how close the choice was.

    ``basis`` is the load-bearing field. A run whose selection scored nothing is still
    adjudicated — every caller falls back to the pack's default ruleset rather than
    producing no verdict — so ``no_match`` is the only record that the procedure was a
    guess. Without it a defaulted adjudication is indistinguishable from a selected one.
    """

    basis: str
    score: float = 0.0
    runner_up: float = 0.0
    spec_count: int = 0
    #: ``(use_case, score)`` best-first, so a report can name what the operator could pin.
    candidates: Tuple[Tuple[str, float], ...] = ()

    @property
    def defaulted(self) -> bool:
        """True when nothing selected a procedure, so the caller's fallback is a guess."""
        return self.basis == "no_match"

    @property
    def margin(self) -> float:
        """Share of the winning score not also held by the runner-up; 1.0 when unrivalled."""
        if self.score <= 0:
            return 0.0
        return max(0.0, (self.score - self.runner_up) / self.score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "basis": self.basis,
            "score": round(self.score, 4),
            "runner_up": round(self.runner_up, 4),
            "margin": round(self.margin, 4),
            "spec_count": self.spec_count,
            "candidates": [[n, round(s, 4)] for n, s in self.candidates],
        }


def select_correlation_spec(pack, analysis) -> Optional[Dict[str, Any]]:
    """Pick the playbook correlation block most relevant to this incident.

    Scores by weighted keyword overlap against the incident summary (primary) and
    hypotheses/investigation areas (tie-break only). Title tokens are split as prose,
    not as field names. Token weights are inverse spec frequency so generic join-key
    names do not dominate. Declaration order is the last-resort tie-break.

    Module-level so the planner and the verdict share one answer; a second
    implementation would be a second answer to the same question.
    """
    return select_correlation_spec_explained(pack, analysis)[0]


def select_correlation_spec_explained(
    pack, analysis
) -> Tuple[Optional[Dict[str, Any]], SelectionBasis]:
    """``select_correlation_spec``'s answer plus how it was reached.

    Same selection, byte for byte; the second element is the fact the pipeline could not
    otherwise state. Separate entry point rather than a widened return so the ten-odd
    existing callers keep asking the question they ask.
    """
    if pack is None:
        return None, SelectionBasis("no_specs")
    specs = pack.correlation_specs()
    if not specs:
        return None, SelectionBasis("no_specs")

    # A pinned use_case (set on referrals) bypasses scoring; honoured here so the
    # planner and verdict both follow the pin. Unknown pins fall back to scoring.
    pinned = str(getattr(analysis, "pinned_use_case", "") or "").strip()
    if pinned:
        for spec in specs:
            if str(spec.get("use_case", "") or "") == pinned:
                logger.info(
                    "Correlation spec pinned to use case '%s' (%s): this run is a referral, so "
                    "its procedure comes from the referring run's evidence rather than from "
                    "scoring its description.",
                    pinned,
                    spec.get("title", "") or "untitled",
                )
                return spec, SelectionBasis(
                    "pinned", spec_count=len(specs), candidates=((pinned, 0.0),)
                )
        logger.warning(
            "Incident carries a pin to use case '%s', which declares no correlation block; "
            "falling back to scoring the description. The verdict's procedure may not be the "
            "one the referral intended.",
            pinned,
        )

    # Primary evidence: what the incident is. Tie-break only: what it might be.
    primary = str(getattr(analysis, "incident_summary", "") or "").lower()
    secondary = " ".join(
        list(getattr(analysis, "initial_hypotheses", []) or [])
        + list(getattr(analysis, "key_investigation_areas", []) or [])
    ).lower()

    # Each spec's discriminating vocabulary: its prose title plus its join keys.
    tokens_by_index: List[Set[str]] = []
    spec_frequency: Dict[str, int] = defaultdict(int)
    for spec in specs:
        tokens = set(_prose_tokens(str(spec.get("title", "") or "")))
        for key in spec.get("keys") or []:
            tokens.update(_prose_tokens(str(key)))
        tokens.discard("")
        tokens_by_index.append(tokens)
        for token in tokens:
            spec_frequency[token] += 1

    def weigh(tokens: Set[str], text: str) -> float:
        """Summed weight of this spec's tokens present in ``text``.

        A token every spec declares scores 0; it is not evidence for any of them.
        A token unique to one spec carries the full weight. No log needed: the
        weight is just how much of the field this token rules out.
        """
        if not text:
            return 0.0
        total = float(len(specs))
        # float() because an empty sum is an int, and these scores are reported now: a
        # candidate list mixing 0 and 0.0 reads as two different measurements.
        return float(
            sum(
                (total - spec_frequency[t]) / total
                for t in tokens
                if t and t in text
            )
        )

    best, best_key = None, (-1.0, -1.0)
    scored: List[Tuple[Tuple[float, float], str]] = []
    for tokens, spec in zip(tokens_by_index, specs):
        key = (weigh(tokens, primary), weigh(tokens, secondary))
        scored.append((key, str(spec.get("use_case", "") or "")))
        if key > best_key:  # strict: a tie keeps the earlier-declared spec
            best, best_key = spec, key
    # Ranked on the same composite key the winner was chosen by, so the runner-up reported
    # is the one that actually came second rather than the second-highest primary score.
    ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)
    candidates = tuple((name, key[0]) for key, name in ranked[:_MAX_SELECTION_CANDIDATES])
    runner_up = ranked[1][0][0] if len(ranked) > 1 else 0.0

    # If nothing matched by keyword but specs exist, still return the top one only
    # when there is a single spec (unambiguous); otherwise require a real match.
    if best_key[0] <= 0 and best_key[1] <= 0 and len(specs) > 1:
        return None, SelectionBasis(
            "no_match", spec_count=len(specs), candidates=candidates
        )
    # A one-spec pack scores 0 on every token by construction — inverse spec frequency is
    # (1-1)/1 — so its spec wins by being the only one, never by matching. Named apart from
    # `scored` because "unrivalled" and "the description fits" are different claims.
    basis = "sole_spec" if len(specs) == 1 else "scored"
    return best, SelectionBasis(
        basis,
        score=best_key[0],
        runner_up=runner_up,
        spec_count=len(specs),
        candidates=candidates,
    )


class CorrelationModule:
    def __init__(
        self,
        config,
        llm_client,
        rag=None,
        knowledge_pack=None,
        row_caps=None,
        link_probe=None,
        inquiry_probe=None,
    ):
        self.config = config or {}
        self.llm_client = llm_client
        self.rag = rag
        # `link_probe` is the only IO path; injected so the module stays mockable.
        # None leaves rung 3 of the advisory link ladder unreachable.
        self.link_probe = link_probe
        # The same for the other advisory lane: None leaves an open question at the state the
        # free rung reached, which is a reported state and never a silence.
        self.inquiry_probe = inquiry_probe
        self.row_caps = dict(row_caps or {})
        self.knowledge_pack = knowledge_pack
        self.sample_rows = int(self.config.get("sample_rows", 20))
        self.llm_max_records = int(self.config.get("llm_max_records", 2000))
        self.llm_max_sources = int(self.config.get("llm_max_sources", 6))
        self.discovery_key_filter = self.config.get("discovery_key_filter", "strict")
        # How the last run's procedure was chosen. Set here so the attribute always exists;
        # the module is shared across jobs, so the run-recorded copy on `ctx.stage_facts` is
        # what the scorer reads and this is only the fallback for direct-drive callers.
        self.last_selection: Optional[SelectionBasis] = None
        # Which path produced (or did not produce) `findings` last run: "llm" or
        # "deterministic". Same sharing caveat as `last_selection` above. Empty findings mean
        # different things on the two paths and the health scorer charges only one of them.
        self.last_narration: str = ""

    def _entity_hints(self) -> Dict[str, str]:
        """Pack's field-name-token -> entity-type map; ``{}`` when no pack is set."""
        pack = self.knowledge_pack
        if pack is None or not hasattr(pack, "field_name_hints"):
            return {}
        try:
            return pack.field_name_hints()
        except Exception:  # a malformed pack must not fail the stage
            logger.debug("field_name_hints unavailable; using engine fallback tokens")
            return {}

    def _zero_row_meanings(self) -> Dict[str, str]:
        """``{source: meaning}`` for empty results; ``{}`` when no pack is set."""
        pack = self.knowledge_pack
        if pack is None or not hasattr(pack, "zero_row_meanings"):
            return {}
        try:
            return dict(pack.zero_row_meanings() or {})
        except Exception:  # a malformed pack must not fail the stage
            return {}

    def _actor_entity_types(self) -> List[str]:
        """Pack's ``role: actor`` entity types; ``[]`` when no pack is set."""
        pack = self.knowledge_pack
        if pack is None or not hasattr(pack, "actor_entity_types"):
            return []
        try:
            return list(pack.actor_entity_types() or [])
        except Exception:
            return []

    def _scope_entity_types(self) -> Set[str]:
        """Pack's ``role: scope`` types unioned with the engine's generic broad set."""
        pack = self.knowledge_pack
        declared: List[str] = []
        if pack is not None and hasattr(pack, "scope_entity_types"):
            try:
                declared = list(pack.scope_entity_types() or [])
            except Exception:
                declared = []
        return _BROAD_ENTITY_TYPES | {str(t).strip().lower() for t in declared if t}

    def _used_by_hint(self, logs, analysis) -> str:
        """Playbook ids linked (via pack `used_by`) to the retrieved sources/entities."""
        pack = self.knowledge_pack
        if pack is None:
            return ""
        playbooks: set = set()
        for source_name in (logs or {}).keys():
            src = pack.source(source_name)
            if src is not None:
                playbooks.update(getattr(src, "used_by", []) or [])
        entity_types = {
            e.type for e in (getattr(analysis, "extracted_entities", []) or [])
        }
        for entity in pack.entities:
            if entity.type in entity_types:
                playbooks.update(getattr(entity, "used_by", []) or [])
        return ", ".join(sorted(playbooks))

    def _normalized_entity_map(
        self, analysis, schema: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, str]]:
        """``{source: {entity_type: field}}`` verified against the source's schema leaves.

        Returns ``{}`` when there is no pack.
        """
        pack = self.knowledge_pack
        if pack is None:
            return {}
        entity_types = {
            e.type for e in (getattr(analysis, "extracted_entities", []) or [])
        }
        if not entity_types:
            return {}
        result: Dict[str, Dict[str, str]] = {}
        for source, leaves in (schema or {}).items():
            leaf_set = set(leaves)
            # Case-insensitive so a lowercase field matches a backend that uppercases;
            # maps to the real spelling so resolve_path's flat fast-path can hit it.
            leaf_by_lower = {l.lower(): l for l in leaves}
            mapping: Dict[str, str] = {}
            for etype in entity_types:
                for candidate in pack.field_priors_for(etype, source):
                    if candidate in leaf_set:
                        mapping[etype] = candidate
                        break
                    real = leaf_by_lower.get(candidate.lower())
                    if real is not None:
                        mapping[etype] = real
                        break
            if mapping:
                result[source] = mapping
        return result

    def _playbook_correlation_spec(self, analysis) -> Optional[Dict[str, Any]]:
        """Delegate to ``select_correlation_spec`` (module-level for planner sharing)."""
        return select_correlation_spec(self.knowledge_pack, analysis)

    def _playbook_correlation_spec_explained(self, analysis):
        """``(spec, SelectionBasis)`` — the same answer, plus how it was reached."""
        return select_correlation_spec_explained(self.knowledge_pack, analysis)

    def _annotate_procedure_selection(self, result, selection, ruleset_key) -> None:
        """Say on the verdict and brief that the procedure was a fallback, not a selection.

        Nothing happens for any basis other than ``no_match``: every other one chose the
        ruleset that ran, so annotating it would be noise. When the pack declares
        ``adjudication_policy: abstain`` the verdict is additionally converted to that
        ruleset's own ``reject`` label — the pack's declared "cannot adjudicate" state, which
        until now was unreachable — because a routing decision and a finding on the merits are
        different answers and only one of them is honest here.
        """
        if selection is None or not selection.defaulted:
            return
        rivals = ", ".join(
            f"{name} ({score:.2f})" for name, score in selection.candidates if name
        )
        detail = (
            f"No procedure matched this incident's summary, so it was adjudicated under "
            f"'{ruleset_key or 'the pack default'}' because that is the pack's default "
            f"ruleset — not because the incident was recognised. Every condition below was "
            f"evaluated against real rows, so the result reads as confident whether or not "
            f"the procedure is the right one. Scored candidates, all at zero: "
            f"{rivals or 'none'}. Pin one with pinned_use_case, or give the matching "
            f"procedure a title that discriminates."
        )
        try:
            verdict = getattr(result, "verdict", None)
            subjects = list(getattr(verdict, "subjects", None) or []) if verdict else []
            for subject in subjects:
                subject.notes.append(f"procedure_unselected={detail}")
            brief = getattr(result, "brief", None)
            if brief is not None:
                brief.notes.append(f"procedure_unselected={detail}")
            logger.warning(
                "No correlation spec matched this incident; adjudicated under the pack "
                "default ruleset '%s'. Candidates all scored zero: %s.",
                ruleset_key or "(none)",
                rivals or "none",
            )
            if self._adjudication_policy() == "abstain" and subjects:
                self._abstain(verdict, subjects, ruleset_key)
        except Exception as exc:  # noqa: BLE001 — an annotation must never fail the stage
            logger.warning(
                "Could not record that no procedure was selected (%s: %s); the verdict is "
                "unchanged but reads as though its procedure had been chosen.",
                type(exc).__name__,
                exc,
            )

    def _adjudication_policy(self) -> str:
        """The pack's policy for an unselected incident; ``default`` when it declares none."""
        pack = self.knowledge_pack
        if pack is None or not hasattr(pack, "adjudication_policy"):
            return "default"
        try:
            return str(pack.adjudication_policy() or "default")
        except Exception:  # noqa: BLE001 — a malformed pack must not fail the stage
            return "default"

    def _abstain(self, verdict, subjects, ruleset_key) -> None:
        """Convert an unselected run's verdict to the ruleset's own ``reject`` state.

        Refused — loudly, and leaving the verdict as it was — when the ruleset declares no
        ``reject`` label: the label is the word the pack's reporting vocabulary and closing
        templates are keyed on, so inventing one produces a verdict nothing downstream can
        render, which is worse than the finding it replaces.
        """
        label = str((getattr(verdict, "labels", None) or {}).get("reject", "") or "").strip()
        if not label:
            logger.warning(
                "Pack declares adjudication_policy: abstain, but ruleset '%s' declares no "
                "'reject' label, so there is no word to abstain in. Keeping the default "
                "ruleset's verdict, which is annotated as unselected.",
                ruleset_key or "(default)",
            )
            return
        for subject in subjects:
            subject.verdict = label
            subject.verdict_class = "reject"  # must move with the label or the two contradict
            subject.notes.append(
                "reject_reason=No procedure recognised this incident, so it is returned to "
                "the detector's owner for routing rather than adjudicated under a procedure "
                "chosen by default."
            )
            subject.lock_target = {}
        counts: Dict[str, int] = defaultdict(int)
        for subject in subjects:
            counts[subject.verdict] += 1
        verdict.summary = "; ".join(f"{v}: {n}" for v, n in counts.items())
        # The draft was rendered from the verdict this replaces, so it asserts a finding under
        # a procedure that was never selected — the exact sentence an abstention exists to
        # withhold. Cleared rather than re-rendered: what to tell whom about an unrecognised
        # incident is a routing decision, and the pack's templates are keyed on findings.
        verdict.notification_draft = (
            "No notification is drafted: no procedure recognised this incident, so there is "
            "no finding to notify. Route the alert to the detector's owner."
        )

    def _resolve_correlation_keys(
        self, analysis, schema, logs, playbook_spec=None
    ) -> List[CorrelationKey]:
        """Resolve join keys in precedence order: playbook → understanding → discovery.

        Higher layers win; lower layers only fill gaps (never overwrite). Each key is
        verified against the live discovered ``schema`` so a declared/derived key that
        isn't actually in the data is dropped, not invented.
        """
        pack = self.knowledge_pack
        leaves_by_source = {s: set(v) for s, v in (schema or {}).items()}
        lower_by_source = {
            s: {l.lower(): l for l in v} for s, v in (schema or {}).items()
        }

        def real_field(
            source: str,
            entity_type: str,
            explicit: str = "",
            value_form: Optional[str] = None,
        ) -> str:
            """Verified leaf for (source, entity): explicit override, then pack priors.

            Case-insensitive; also accepts underscore-flattened aliases of dotted paths,
            mirroring ``resolve_path`` (e.g. ``locator.red`` matches ``locator_red``).
            ``value_form`` restricts priors to one surface form; unset flattens the map.
            """
            cands: List[str] = []
            if explicit:
                cands.append(explicit)
            if pack is not None:
                cands.extend(
                    pack.field_priors_for(entity_type, source, value_form)
                    if value_form is not None
                    else pack.field_priors_for(entity_type, source)
                )
            leaves = leaves_by_source.get(source, set())
            lower = lower_by_source.get(source, {})
            for c in cands:
                if c in leaves:
                    return c
                if c.lower() in lower:
                    return lower[c.lower()]
                # Underscore-flattened alias, full path then shorter prefixes (a struct
                # column aliased whole, e.g. `pay.method.x` where the row has `pay_method`).
                if "." in c:
                    segments = c.split(".")
                    for cut in range(len(segments), 0, -1):
                        alias = "_".join(segments[:cut])
                        if alias in leaves:
                            return alias
                        if alias.lower() in lower:
                            return lower[alias.lower()]
            return ""

        def time_field(source: str, explicit: str = "") -> str:
            if explicit and (
                explicit in leaves_by_source.get(source, set())
                or explicit.lower() in lower_by_source.get(source, {})
            ):
                return lower_by_source.get(source, {}).get(explicit.lower(), explicit)
            # Schema order is alphabetical; pick_time_field scores on resolution.
            return pick_time_field(
                list(schema.get(source, [])), (logs or {}).get(source) or []
            )

        sources = list((logs or {}).keys())
        # Track which (entity_type) already covered by a higher layer, per source.
        covered: Dict[str, set] = defaultdict(set)
        resolved: List[CorrelationKey] = []

        def joinable(entity_type: str, layer: str) -> bool:
            """False when the type is the run's time bound (see ``_NON_JOINABLE_ENTITY_TYPES``).

            Declines with a log line so a dropped pack declaration is visible.
            """
            if str(entity_type or "").strip().lower() in _NON_JOINABLE_ENTITY_TYPES:
                logger.info(
                    "Correlation key '%s' (%s) is this run's time bound, not an identity; "
                    "every retrieved row is inside it by construction, so the join would "
                    "hold of every pair of rows. Skipping the key — the window still gates "
                    "the other keys.",
                    entity_type,
                    layer,
                )
                return False
            return True

        # --- Layer 1: playbook-declared (authoritative) ---
        if playbook_spec:
            tw = str(playbook_spec.get("time_window", "") or "")
            field_overrides = playbook_spec.get("fields", {}) or {}
            for etype in playbook_spec.get("keys", []) or []:
                etype = str(etype)
                if not joinable(etype, "playbook-declared"):
                    continue
                per_source, tfields = {}, {}
                for src in sources:
                    explicit = (field_overrides.get(src, {}) or {}).get(etype, "")
                    f = real_field(src, etype, explicit)
                    if f:
                        per_source[src] = f
                        declared = (playbook_spec.get("time_fields", {}) or {}).get(
                            src, ""
                        )
                        # Read time_fields with or without a time_window: `time_fields`
                        # names which instant is the event, not just when to gate it.
                        if tw or declared:
                            tf = time_field(src, declared)
                            if tf:
                                tfields[src] = tf
                if len(per_source) >= 2:
                    resolved.append(
                        CorrelationKey(
                            entity_hint=etype,
                            sources=per_source,
                            time_window=tw,
                            time_fields=tfields,
                            origin="playbook",
                        )
                    )
                    for src in per_source:
                        covered[src].add(etype)
                else:
                    logger.info(
                        "Playbook key '%s' not present in >=2 sources' schema; skipping.",
                        etype,
                    )

        # --- Layer 2: understanding-derived (correlation_keys, then entities) ---
        derived_types = list(getattr(analysis, "correlation_keys", []) or [])
        if not derived_types:
            derived_types = [
                e.type for e in (getattr(analysis, "extracted_entities", []) or [])
            ]
        ev = getattr(analysis, "event_time", None)
        tw2 = ""
        if ev is not None and getattr(ev, "start", None):
            tw2 = ""  # event_time bounds retrieval already; not a per-pair gap here
        for etype in dict.fromkeys(str(t) for t in derived_types):
            if not joinable(etype, "understanding-derived"):
                continue
            per_source = {}
            for src in sources:
                if etype in covered[src]:
                    continue
                f = real_field(src, etype)
                if f:
                    per_source[src] = f
            if len(per_source) >= 2:
                resolved.append(
                    CorrelationKey(
                        entity_hint=etype,
                        sources=per_source,
                        time_window=tw2,
                        origin="understanding",
                    )
                )
                for src in per_source:
                    covered[src].add(etype)

        # --- Layer 3: data-driven discovery (fallback) ---
        key_filter = (
            str(playbook_spec.get("key_filter", self.discovery_key_filter))
            if playbook_spec
            else self.discovery_key_filter
        )
        discovered = discover_join_keys(
            logs,
            schema,
            key_filter,
            self._entity_hints(),
            _subject_value_set(analysis),
        )
        for dk in discovered:
            hint = dk["entity_hint"]
            per_source = {
                s: f for s, f in dk["sources"].items() if hint not in covered[s]
            }
            if len(per_source) >= 2:
                resolved.append(
                    CorrelationKey(
                        entity_hint=hint,
                        sources=per_source,
                        origin="discovered",
                        overlap_score=dk.get("overlap_score", 0.0),
                    )
                )
                for src in per_source:
                    covered[src].add(hint)

        # --- Form consistency: ensure all sources on a key compare the same surface form.
        # `field_priors_for` flattens the form map and takes the first candidate per
        # source; two sources binding different forms compare vocabularies that never
        # intersect. The chosen form is scored on the rows, not read from `prefer`.
        for key in [k for k in resolved if len(k.sources) >= 2]:
            if pack is None:
                break
            fixed = _form_consistent_key(
                key, pack, logs, real_field, _field_values, logger
            )
            if fixed:
                key.sources = fixed

        # --- Propagation: extend a resolved key to sources with the same-named leaf.
        # Only confirms, never guesses: the key must already bind >= 2 sources, the
        # column name must match an existing binding (case/alias-insensitive), and the
        # candidate values must overlap the bound set at `_MIN_JK_CONTAINMENT`.
        for key in [k for k in resolved if len(k.sources) >= 2]:
            bound_names = {f.lower() for f in key.sources.values()}
            bound_vals: Set[str] = set()
            for s, f in key.sources.items():
                bound_vals |= _field_values(logs.get(s) or [], f)
            # No floor on count: a targeted investigation may have only one alerted value.
            # The containment check below is what guards against vacuous overlap.
            if not bound_vals:
                continue
            for src in sources:
                if src in key.sources or key.entity_hint in covered[src]:
                    continue
                leaf = next(
                    (
                        real
                        for low, real in lower_by_source.get(src, {}).items()
                        if low in bound_names
                    ),
                    "",
                )
                if not leaf:
                    continue
                cand = _field_values(logs.get(src) or [], leaf)
                if not cand:
                    continue
                shared = bound_vals & cand
                denom = min(len(bound_vals), len(cand))
                if not shared or len(shared) / denom < _MIN_JK_CONTAINMENT:
                    continue
                key.sources[src] = leaf
                covered[src].add(key.entity_hint)
                logger.info(
                    "Propagated correlation key '%s' to source '%s' via same-named leaf "
                    "'%s' (%d/%d values shared with the bound sources).",
                    key.entity_hint,
                    src,
                    leaf,
                    len(shared),
                    denom,
                )
        return resolved

    def _build_deterministic_plan(
        self, resolved_keys: List[CorrelationKey]
    ) -> TransformPlan:
        """Build a no-LLM TransformPlan of cross_source_overlap steps from resolved keys."""
        steps = []
        used_labels: set = set()
        for key in resolved_keys:
            base = f"join_{key.entity_hint}"
            label, n = base, 1
            while label in used_labels:
                n += 1
                label = f"{base}_{n}"
            used_labels.add(label)
            steps.append(
                TransformStep(
                    op="cross_source_overlap",
                    label=label,
                    entity=key.entity_hint,
                    sources=list(key.sources.keys()),
                    time_window=key.time_window,
                    time_fields=key.time_fields,
                )
            )
        return TransformPlan(
            steps=steps, reasoning="deterministic join on resolved correlation keys"
        )

    def _keys_entity_map(
        self, resolved_keys: List[CorrelationKey]
    ) -> Dict[str, Dict[str, str]]:
        """{source: {entity_hint: field}} from resolved keys, for the executor."""
        emap: Dict[str, Dict[str, str]] = defaultdict(dict)
        for key in resolved_keys:
            for source, field in key.sources.items():
                emap[source][key.entity_hint] = field
        return {s: dict(m) for s, m in emap.items()}

    def _should_use_llm(self, aggregations, resolved_keys) -> bool:
        """LLM only when the data is small and the deterministic layer found no join key.

        High-volume or already-resolved incidents skip the LLM.
        """
        total = aggregations.get("total_records", 0)
        n_sources = len(
            [s for s, c in aggregations.get("record_counts", {}).items() if c]
        )
        if total > self.llm_max_records or n_sources > self.llm_max_sources:
            return False
        # Low volume: use the LLM to reason about complex/implicit joins the
        # deterministic layer didn't resolve.
        return len(resolved_keys) == 0

    async def _match_playbook(self, analysis) -> str:
        """Retrieve the playbook(s) most relevant to this incident via RAG."""
        if self.rag is None:
            return ""
        query = getattr(analysis, "incident_summary", "") or analysis.model_dump_json()
        try:
            retrieved = await self.rag.retrieve(query)
            return self.rag.format_context(retrieved)
        except Exception as e:  # retrieval must never break correlation
            logger.warning("Playbook retrieval failed: %s", e)
            return ""

    async def _plan_transforms(
        self,
        analysis,
        schema,
        aggregations,
        playbook,
        used_by_hint="",
        entity_map=None,
        resolved_keys=None,
        discovered_keys=None,
        guidance=None,
    ) -> TransformPlan:
        """Ask the LLM for a pattern-driven TransformPlan. Empty plan on failure."""
        try:
            content = (
                f"Incident:\n{analysis.model_dump_json(indent=2)}\n\n"
                f"Available columns per source:\n{json.dumps(schema, indent=2)}\n\n"
                f"Deterministic aggregates:\n{json.dumps(aggregations, default=str)}"
            )
            if resolved_keys:
                content += (
                    "\n\nResolved correlation keys (already computed; entity_hint -> "
                    "real field per source). Prefer cross_source_overlap on these:\n"
                    + json.dumps([k.model_dump() for k in resolved_keys], default=str)
                )
            if discovered_keys:
                content += (
                    "\n\nCandidate join keys discovered from the data (use for "
                    "cross_source_overlap when no resolved key fits):\n"
                    + json.dumps(discovered_keys, indent=2, default=str)
                )
            if entity_map:
                content += (
                    f"\n\nNormalized entity map (per source, entity type -> real field; "
                    f"use these entity types for cross_source_overlap):\n"
                    f"{json.dumps(entity_map, indent=2)}"
                )
            if used_by_hint:
                content += (
                    f"\n\nThe retrieved sources/entities relate to these fraud "
                    f"playbooks: {used_by_hint}"
                )
            if playbook:
                content += f"\n\nMatched playbook(s):\n{playbook}"
            # Playbook injected explicitly here; pass rag=None to avoid double retrieval.
            messages = [{"role": "system", "content": _PLAN_SYSTEM_PROMPT}]
            human_msg = guidance_message(guidance)
            if human_msg is not None:
                messages.append(human_msg)
            messages.append({"role": "user", "content": content})
            return await self.llm_client.structured_output(
                messages,
                response_model=TransformPlan,
                rag=None,
                stage="correlation",
            )
        except Exception as e:
            logger.warning("Transform planning failed; using aggregates only: %s", e)
            return TransformPlan()

    def _compact_transforms(self, transforms) -> str:
        """Shrink transform results to per-step row counts + top-K sample rows.

        Used when the full results exceed the LLM input budget, so narration can still
        run on real-volume data instead of being skipped (which left findings empty).
        """
        k = max(1, self.sample_rows)
        compact = []
        for t in transforms:
            compact.append(
                {
                    "label": t.label,
                    "op": t.op,
                    "row_count": len(t.rows),
                    "sample_rows": t.rows[:k],
                    "note": t.note,
                }
            )
        return json.dumps(compact, default=str)

    async def _narrate(
        self, analysis, aggregations, transforms, playbook, guidance=None
    ) -> Optional[CorrelationResult]:
        """LLM narration over aggregates + compact transform results. None on failure."""
        transforms_text = json.dumps([t.model_dump() for t in transforms], default=str)
        agg_text = json.dumps(aggregations, default=str)
        # Over budget: compact the transform results (row counts + top-K samples)
        # rather than skip narration entirely, so findings are still produced.
        if len(agg_text) + len(transforms_text) > _MAX_LLM_INPUT_CHARS:
            transforms_text = self._compact_transforms(transforms)
            logger.info(
                "Correlation narration input over budget; compacting transform results."
            )
        content = (
            f"Incident understanding:\n{analysis.model_dump_json(indent=2)}\n\n"
            f"Aggregates:\n{agg_text}\n\n"
            f"Playbook-driven transform results:\n{transforms_text}"
        )
        if playbook:
            content += f"\n\nPlaybook guidance:\n{playbook}"
        try:
            messages = [{"role": "system", "content": _NARRATE_SYSTEM_PROMPT}]
            human_msg = guidance_message(guidance)
            if human_msg is not None:
                messages.append(human_msg)
            messages.append({"role": "user", "content": content})
            return await self.llm_client.structured_output(
                messages,
                response_model=CorrelationResult,
                rag=None,
                stage="correlation",
            )
        except Exception as e:
            logger.warning("LLM correlation narration failed: %s", e)
            return None

    def _deterministic_summary(self, aggregations, transforms, resolved_keys) -> str:
        """No-LLM summary_text for the high-volume path (counts + join results).

        Both projections include ``t.note`` so a truncated row count is not read as a total.
        """
        joins = [
            {
                "label": t.label,
                "matches": len(t.rows),
                "sample": t.rows[:3],
                **({"note": t.note} if t.note else {}),
            }
            for t in transforms
            if t.op == "cross_source_overlap"
        ]
        return json.dumps(
            {
                "record_counts": aggregations.get("record_counts", {}),
                "total_records": aggregations.get("total_records", 0),
                "resolved_correlation_keys": [k.model_dump() for k in resolved_keys],
                "cross_source_joins": joins,
                "transforms": [
                    {
                        "label": t.label,
                        "op": t.op,
                        "row_count": len(t.rows),
                        **({"note": t.note} if t.note else {}),
                    }
                    for t in transforms
                ],
            },
            indent=2,
            default=str,
        )

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def analyze(
        self,
        logs,
        understanding,
        guidance=None,
        keyed_sources=None,
        unanswered_sources=None,
        link_modes=None,
    ) -> CorrelationResult:
        """Correlate the retrieved rows.

        ``guidance`` reaches the two LLM steps (planning + narration); deterministic
        aggregates, resolved keys and the verdict are unaffected.

        ``keyed_sources``: sources whose query was identity-scoped, per the retrieval stage.
        Per-call (not constructor state). Absent: no source is known keyed; zero-row
        lookups stay ``unknown``.

        ``unanswered_sources``: ``{source: why}`` for sources asked but unanswered.
        Absent from ``logs`` like an unasked source; this field makes the reason visible.

        ``link_modes``: ``{target use case: mode}`` operator escalation. Per-call so a gate
        rejection re-run does not silently revert the operator's choice.
        """
        analysis = understanding.analysis
        entity_values = [
            e.value for e in (getattr(analysis, "extracted_entities", []) or [])
        ]
        # Exclude broad population types (org, alert metadata, time window) from
        # person-of-interest flagging so a shared value doesn't tag everyone.
        broad_types = self._scope_entity_types()
        subject_values = [
            e.value
            for e in (getattr(analysis, "extracted_entities", []) or [])
            if str(getattr(e, "type", "")).lower() not in broad_types
        ]
        # Entity values keyed by type: lets the evidence layer attribute rows from a
        # document-bound source using the incident's own extracted value.
        typed_subjects: Dict[str, List[str]] = defaultdict(list)
        for ent in getattr(analysis, "extracted_entities", []) or []:
            etype = str(getattr(ent, "type", "")).lower()
            if etype and ent.value:
                typed_subjects[etype].append(str(ent.value))
        aggregations = aggregate(logs, entity_values)
        total = aggregations["total_records"]

        result = CorrelationResult(aggregations=aggregations, record_count=total)

        if total == 0:
            result.summary_text = "No records were retrieved; nothing to correlate."
            return result

        schema = derive_schema(logs)

        # --- Layered correlation-key resolution: playbook -> understanding -> discovery.
        # The basis is kept off `aggregations`: that dict is json.dumps'ed into the transform
        # and narration prompts, so a key added there changes what two LLM calls read.
        playbook_spec, selection = self._playbook_correlation_spec_explained(analysis)
        self.last_selection = selection
        resolved_keys = self._resolve_correlation_keys(
            analysis, schema, logs, playbook_spec
        )
        # Discovered candidates surfaced separately for the LLM path and the downstream record.
        key_filter = (
            str(playbook_spec.get("key_filter", self.discovery_key_filter))
            if playbook_spec
            else self.discovery_key_filter
        )
        discovered_keys = discover_join_keys(
            logs, schema, key_filter, self._entity_hints()
        )
        # Mutate the stored dict (Pydantic copied `aggregations` at construction).
        result.aggregations["resolved_correlation_keys"] = [
            k.model_dump() for k in resolved_keys
        ]
        result.aggregations["discovered_join_keys"] = discovered_keys
        aggregations = result.aggregations
        if resolved_keys:
            logger.info(
                "Resolved %d correlation key(s): %s",
                len(resolved_keys),
                [(k.entity_hint, k.origin, list(k.sources)) for k in resolved_keys],
            )

        # Augmented entity map: resolved keys extended by pack bindings (pack wins on conflict).
        entity_map = self._normalized_entity_map(analysis, schema)
        keys_map = self._keys_entity_map(resolved_keys)
        augmented_map: Dict[str, Dict[str, str]] = defaultdict(dict)
        for src, m in keys_map.items():
            augmented_map[src].update(m)
        for src, m in entity_map.items():  # pack bindings win on conflict
            augmented_map[src].update(m)
        augmented_map = {s: dict(m) for s, m in augmented_map.items()}

        # --- Volume gate: deterministic by default; LLM only for low-volume/complex.
        used_llm = False
        self.last_narration = ""
        if self._should_use_llm(aggregations, resolved_keys):
            playbook = await self._match_playbook(analysis)
            used_by_hint = self._used_by_hint(logs, analysis)
            plan = await self._plan_transforms(
                analysis,
                schema,
                aggregations,
                playbook,
                used_by_hint,
                augmented_map,
                resolved_keys,
                discovered_keys,
                guidance,
            )
            used_llm = True
        else:
            playbook = ""
            plan = self._build_deterministic_plan(resolved_keys)
        if plan.reasoning:
            logger.info(
                "Transform plan (%s): %s",
                "llm" if used_llm else "deterministic",
                plan.reasoning,
            )
        transforms = execute_plan(plan, logs, augmented_map)
        result.transforms = transforms

        # Compact text that flows downstream regardless of whether narration runs.
        result.summary_text = self._deterministic_summary(
            aggregations, transforms, resolved_keys
        )

        # Narrate findings via LLM only on the low-volume/complex path (cost control).
        # Which path ran is recorded because empty `findings` is a defect on one and the
        # design on the other, and the two are indistinguishable from the result alone.
        self.last_narration = "llm" if used_llm else "deterministic"
        if used_llm:
            narrated = await self._narrate(
                analysis, aggregations, transforms, playbook, guidance
            )
            if narrated is not None:
                result.findings = narrated.findings
                if narrated.summary_text:
                    result.summary_text = narrated.summary_text

        # Resolved once and shared by the evidence trimmer, verdict and case builder so
        # all three read the same procedure. Empty resolves to the pack's declared default.
        ruleset_key = ""
        try:
            if self.knowledge_pack is not None and playbook_spec:
                ruleset_key = self.knowledge_pack.ruleset_key_for(
                    str(playbook_spec.get("use_case", "") or "")
                )
        except Exception:  # selection is an optimisation; never fail the stage for it
            ruleset_key = ""

        # --- Investigation evidence (deterministic, best-effort) -------------------
        try:
            # Local import avoids the module-load cycle (evidence imports correlation).
            from encoded_fields import decoded_tables
            from evidence import build_evidence

            for k in resolved_keys:
                for src in k.sources:
                    if src not in augmented_map:
                        logger.debug(
                            "Resolved key %s references source %s absent from entity_map",
                            k.entity_hint,
                            src,
                        )
            cross_joins = [
                {
                    "value": r.get("value"),
                    "sources": r.get("sources", []),
                    **(
                        {"time_window": r["time_window"]}
                        if r.get("time_window")
                        else {}
                    ),
                }
                for t in transforms
                if t.op == "cross_source_overlap"
                for r in t.rows
            ]
            result.evidence = build_evidence(
                logs,
                aggregations,
                resolved_keys,
                augmented_map,
                entity_values,
                char_budget=int(
                    self.config.get("evidence_char_budget", _MAX_LLM_INPUT_CHARS)
                ),
                schema=schema,
                cross_source_joins=cross_joins,
                subject_values=subject_values,
                row_caps=self.row_caps,
                actor_entity_types=self._actor_entity_types(),
                field_name_hints=self._entity_hints(),
                # decoded_paths: pack-declared tables from encoded fields, not guessed.
                decoded_paths=decoded_tables(self.knowledge_pack),
                zero_row_meanings=self._zero_row_meanings(),
                typed_subjects=dict(typed_subjects),
                # Resolved spec (imports applied) so condition column hints are complete.
                ruleset_spec=(
                    self.knowledge_pack.ruleset_spec(ruleset_key)
                    if self.knowledge_pack is not None
                    else None
                ),
            )
        except Exception as e:  # best-effort; evidence must never break correlation
            logger.warning("Evidence build failed (%s); continuing without it.", e)
            result.evidence = None

        # --- Pack-driven verdict (best-effort) --------------------------------------
        try:
            pack = self.knowledge_pack
            spec = pack.ruleset_spec(ruleset_key) if pack is not None else None
            if spec:
                pack_data = (
                    getattr(pack, "pack_data", None) if pack is not None else None
                )
                result.verdict = evaluate_verdict(
                    spec,
                    logs,
                    analysis,
                    augmented_map,
                    pack_data,
                    row_caps=self.row_caps,
                    keyed_sources=dict(keyed_sources or {}),
                    unanswered_sources=dict(unanswered_sources or {}),
                )
                if result.verdict is not None:
                    logger.info(
                        "Verdict: %s (%d subject(s), degraded=%s).",
                        result.verdict.summary,
                        len(result.verdict.subjects),
                        result.verdict.degraded,
                    )
        except Exception as e:  # best-effort; verdict must never break correlation
            logger.warning("Verdict evaluation failed (%s); continuing without it.", e)
            result.verdict = None

        # --- Use-case brief (best-effort): InvestigationBrief the report and anomaly
        # stages narrate from, so the narrative is bound by the verdict. Deterministic.
        try:
            pack = self.knowledge_pack
            spec = pack.ruleset_spec(ruleset_key) if pack is not None else None
            # The same key the verdict used; must not be re-derived (the brief and verdict
            # must name the same procedure). `default_ruleset_key()` not `ruleset_keys()[0]`
            # because declaration order is directory order, not stability order.
            if not ruleset_key:
                ruleset_key = pack.default_ruleset_key() if pack is not None else ""
            playbook_id = (
                (playbook_spec or {}).get("playbook_id", "") if playbook_spec else ""
            )
            from usecases.registry import get_analyzer

            analyzer = get_analyzer(playbook_id, ruleset_key, pack)
            assessment = analyzer.analyze(
                spec,
                logs,
                analysis,
                entity_map=augmented_map,
                transforms=transforms,
                knowledge_pack=pack,
                playbook_id=playbook_id or ruleset_key,
                # Pass the verdict directly; re-deriving it here loses row_caps and
                # keyed_sources, producing a weaker reading that can contradict the verdict.
                verdict=result.verdict,
                row_caps=self.row_caps,
                keyed_sources=dict(keyed_sources or {}),
            )
            result.brief = assessment.brief
            # Backfill the verdict if the analyzer produced one and the verdict stage didn't.
            if result.verdict is None and assessment.verdict is not None:
                result.verdict = assessment.verdict
            if result.brief is not None:
                logger.info(
                    "Brief: use_case=%s, %d decisive fail(s), %d unknown(s), degraded=%s.",
                    result.brief.use_case,
                    len(result.brief.decisive_fails),
                    len(result.brief.decisive_unknowns),
                    result.brief.degraded,
                )
        except Exception as e:  # best-effort; brief must never break correlation
            logger.warning(
                "Use-case brief build failed (%s); continuing without it.", e
            )
            result.brief = None

        # Both the verdict and the brief now exist, so this is where a defaulted procedure can
        # be stated on each of them. Before the link lane, which reads the verdict.
        self._annotate_procedure_selection(result, selection, ruleset_key)

        # --- Cross-procedure link assessment (best-effort, advisory) ---------------
        # Reads verdict and brief; never reaches this run's verdict, conditions or health.
        try:
            from links import assess_links

            result.links = assess_links(
                self.knowledge_pack,
                analysis,
                logs,
                verdict=result.verdict,
                brief=result.brief,
                ruleset_key=ruleset_key,
                entity_map=augmented_map,
                pack_data=(
                    getattr(self.knowledge_pack, "pack_data", None)
                    if self.knowledge_pack is not None
                    else None
                ),
                row_caps=self.row_caps,
                keyed_sources=dict(keyed_sources or {}),
                unanswered_sources=dict(unanswered_sources or {}),
                # Kept separate from incident_summary so rival-procedure vocabulary
                # does not interfere with playbook selection.
                requested=(
                    list(getattr(analysis, "requested_links", None) or [])
                    if isinstance(getattr(analysis, "requested_links", None), list)
                    else []
                ),
                config=self.config.get("links") or {},
                mode_overrides=(
                    {str(k): str(v) for k, v in link_modes.items()}
                    if isinstance(link_modes, dict)
                    else {}
                ),
            )
            # Run rung-3 probes before the brief copies the links list: a probe re-settles
            # findings in place, so running after the copy would leave the brief stale.
            if self.link_probe is not None and result.links:
                try:
                    from link_probe import run_link_probes

                    await run_link_probes(
                        self.link_probe,
                        result.links,
                        pack=self.knowledge_pack,
                        analysis=analysis,
                        logs=logs,
                        entity_map=augmented_map,
                        pack_data=(
                            getattr(self.knowledge_pack, "pack_data", None)
                            if self.knowledge_pack is not None
                            else None
                        ),
                        row_caps=self.row_caps,
                        keyed_sources=dict(keyed_sources or {}),
                        unanswered_sources=dict(unanswered_sources or {}),
                        config=self.config.get("links") or {},
                        ruleset_key=ruleset_key,
                        mode_overrides=(
                            {str(k): str(v) for k, v in link_modes.items()}
                            if isinstance(link_modes, dict)
                            else {}
                        ),
                    )
                except Exception as e:  # advisory; never break the stage it advises on
                    logger.warning(
                        "Cross-procedure link probing failed (%s); the candidates keep "
                        "their free-rung settlement.",
                        e,
                    )
            if result.brief is not None and isinstance(
                getattr(result.brief, "links", None), list
            ):
                result.brief.links = list(result.links)
        except Exception as e:  # advisory; must never break the stage it advises on
            logger.warning(
                "Cross-procedure link assessment failed (%s); continuing without it.", e
            )
            result.links = []

        # --- This procedure's own open questions (best-effort, advisory) ------------
        # The other axis from the link lane: what THIS procedure could not settle. Reads the
        # verdict and the brief; never reaches this run's verdict, conditions or health.
        try:
            from inquiry import assess_inquiries

            result.inquiries = assess_inquiries(
                self.knowledge_pack,
                analysis,
                logs,
                verdict=result.verdict,
                brief=result.brief,
                ruleset_key=ruleset_key,
                config=self.config.get("inquiries") or {},
            )
            # Probes before the brief copies the list, for the link lane's reason: a probe
            # settles findings in place, so copying first would leave the brief stale.
            if self.inquiry_probe is not None and result.inquiries:
                try:
                    from inquiry_probe import run_inquiries

                    await run_inquiries(
                        self.inquiry_probe,
                        result.inquiries,
                        pack=self.knowledge_pack,
                        analysis=analysis,
                        logs=logs,
                        config=self.config.get("inquiries") or {},
                        # The link lane's budget is an INPUT here: the two lanes share one
                        # ceiling, so this one cannot be resolved without knowing what the
                        # other may spend.
                        link_config=self.config.get("links") or {},
                        ruleset_key=ruleset_key,
                    )
                except Exception as e:  # advisory; never break the stage it advises on
                    logger.warning(
                        "Open-question probing failed (%s); the questions keep the state the "
                        "free rung reached.",
                        e,
                    )
            if result.brief is not None and isinstance(
                getattr(result.brief, "inquiries", None), list
            ):
                result.brief.inquiries = list(result.inquiries)
        except Exception as e:  # advisory; must never break the stage it advises on
            logger.warning(
                "Open-question assessment failed (%s); continuing without it.", e
            )
            result.inquiries = []

        logger.info(
            "Correlation complete: %d records, %d keys, %d transforms, %d findings (%s).",
            total,
            len(resolved_keys),
            len(transforms),
            len(result.findings),
            "llm" if used_llm else "deterministic",
        )
        return result
