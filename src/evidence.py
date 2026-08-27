"""
Investigation-evidence reduction stage (deterministic, domain- and incident-agnostic).

Reduces raw retrieved logs into a compact ``EvidencePack``:
  * a time-ordered chronology across sources (timestamp, source, actor, action, entities);
  * actor attribution rolled up by identity;
  * per-source trimmed aggregates with top-value distributions;
  * cross-source joins computed by correlation.

``render_for_prompt`` turns the pack into compact labeled text for the anomaly-detection
and report-generation stages.
"""

import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set, Tuple)

# Sibling flat-import (matches correlation.py / anomaly_detection.py style).
from correlation import (_MAX_ROWS_PER_SOURCE, _TIME_TOKENS, _bucket_key,
                         _matches_any_identifier, _name_tokens,
                         _norm_identifier, _parse_ts, flatten_leaves,
                         pick_time_field, resolve_path)
from encoded_fields import records_at
from models.pydantic_models import (ActorRollup, ChronologyEvent, EvidencePack,
                                    SourceEvidence)

logger = logging.getLogger(__name__)

# --- budget / aggregation caps (mirror correlation's _MAX_* convention) -----
_DEFAULT_CHAR_BUDGET = 15000
_MAX_CHRONOLOGY_EVENTS = 200  # events kept before volume-triggered aggregation
_MAX_ACTORS = 40
_MAX_EXAMPLES_PER_GROUP = 3
_AGG_VOLUME_TRIGGER = 200  # per-source row count above which we aggregate
_MAX_TRIMMED_COLS = 12  # discriminating columns kept per source
_CONSTANT_COL_RATIO = 0.98  # column dropped if one value covers >= this share

# The last rung of the degradation ladder, exported so stage_health can distinguish it
# from the five above. Rungs 1-5 aggregate but name every source; this one delegates to
# the renderer, which drops whole lines, so a line the downstream stages would have read
# may be absent.
CLIP_NOTE = "evidence over budget after every aggregation step; the render is trimmed to fit"
_BLOB_LEN = 200  # a leaf value longer than this is a "blob"
_MAX_DIST_VALUES = 8  # top values kept per distribution column
_MAX_SOURCE_EXAMPLES = 3  # trimmed example rows kept per source
_MAX_CONDITION_DEPTH = 6  # nesting walked inside one ruleset condition
# Render caps: lead with the subject's events, bound background to keep the input dense.
_MAX_SUBJECT_CHRONO = 60
_MAX_BACKGROUND_CHRONO = 25
_MAX_DIST_COLS_RENDERED = 6  # distribution columns rendered per source
_MAX_SOURCE_EXAMPLES_RENDERED = 2  # example rows rendered per source
_MAX_EXAMPLE_LEN = 300  # chars per rendered example row
# Values of the columns a verdict condition read, rendered on their own line and exempt
# from every budget cut. Marked with a prefix so _thin_source_lines can identify them.
_ADJ_LINE_PREFIX = "    * "
_MAX_ADJ_VALUES = 4  # distinct values listed per adjudicated column
_MAX_ADJ_VALUE_LEN = 60  # chars per listed value
# Floor only; the per-source cap is derived from the budget in _adj_line_caps.
_MIN_ADJ_LINE_LEN = 600  # chars of the per-source adjudicated line, at minimum
# Collective ceiling; exempt lines cannot crowd out the sections that contextualise them.
_ADJ_BUDGET_SHARE = 0.25
_MAX_BACKGROUND_ACTORS = 10  # background (non-subject) actors rendered in attribution
# Per-source decoded-table event cap; hitting it is logged as a lower bound.
_MAX_DECODED_EVENTS_PER_SOURCE = 200
# Budget sentinel: passes the pack's full rendered length without clipping.
_NO_CLIP = 1 << 40
# Fraction of the budget the head sections (chronology + attribution) may use.
_HEAD_BUDGET_SHARE = 0.55

# Field-name tokens that mark an "action"/verb column (best-effort, descriptive).
_ACTION_TOKENS = {
    "action",
    "event",
    "eventtype",
    "type",
    "op",
    "operation",
    "verb",
    "method",
    "result",
    "status",
    "outcome",
    "phase",
    "activity",
    "command",
    "cmd",
}
# Generic actor entity types, tried after the pack's role:actor declaration
# (KnowledgePack.actor_entity_types).
_ACTOR_ENTITY_TYPES = ["user", "actor", "agent", "account", "email", "session"]
# Field-name tokens for the acting identity when no entity is mapped for the source.
# Pack field_name_hints are consulted first via _actor_name_tokens.
_ACTOR_NAME_TOKENS = {
    "user",
    "userid",
    "uid",
    "actor",
    "agent",
    "login",
    "account",
    "acct",
    "email",
    "operator",
    "username",
}
_MASK_RE = re.compile(r"^\*+$|.*\*{3,}.*")
# A value that starts like an ISO date/time (2026-07-24, 2026-07-24T20:31, ...).
_TS_HINT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2})?")


def parse_timestamp(v: Any) -> Optional[datetime]:
    """Epoch-aware timestamp parse. Delegates to ``correlation._parse_ts`` (which now
    handles epoch int/millis + ISO + a few string formats) so evidence and correlation
    always agree on how a value parses."""
    return _parse_ts(v)


def _iso(dt: Optional[datetime]) -> str:
    """Normalized ISO-8601 string, or '' when unparseable."""
    return dt.isoformat() if dt else ""


def _first_scalar(row: Dict[str, Any], field: str) -> str:
    """First resolved scalar for a field path, as a string ('' if none)."""
    vals = resolve_path(row, field)
    return str(vals[0]) if vals else ""


# --- field detection --------------------------------------------------------


def _leaf_field_relevance(
    logs: Dict[str, List[Dict]],
    resolved_keys: List[Any],
    entity_map: Dict[str, Dict[str, str]],
    decoded_paths: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Set[str]]:
    """{source: set(relevant leaf paths)}: union of resolved-key fields, entity-map
    fields, and per-source time fields. These columns are always kept when trimming."""
    relevant: Dict[str, Set[str]] = defaultdict(set)
    for src, m in (entity_map or {}).items():
        for _etype, field in (m or {}).items():
            if field:
                relevant[src].add(field)
    for key in resolved_keys or []:
        for src, field in (getattr(key, "sources", {}) or {}).items():
            if field:
                relevant[src].add(field)
        for src, tfield in (getattr(key, "time_fields", {}) or {}).items():
            if tfield:
                relevant[src].add(tfield)
    # All columns of a decoded table are relevant: the pack named the field and the
    # constant-column rule would otherwise remove them. Read from rows, not declarations.
    for src, paths in (decoded_paths or {}).items():
        for row in (logs or {}).get(src, [])[:_MAX_ROWS_PER_SOURCE]:
            for path in paths:
                for rec in records_at(row, path):
                    for col in rec:
                        relevant[src].add(f"{path}.{col}")
    return {s: set(v) for s, v in relevant.items()}


def _norm_path(path: str) -> str:
    """Normalise a dotted path and its underscore-flattened alias to the same form.

    The SQL-gen prompt aliases struct leaves (``creator.sign`` becomes ``creator_sign``),
    so both sides of a comparison go through here to avoid a false mismatch.
    """
    return str(path).strip().replace(".", "_")


def _condition_paths(node: Any, depth: int = 0) -> Tuple[Set[str], Set[str]]:
    """``(every string in this condition, every logical source name it names)``.

    Every string is a candidate rather than enumerating mechanics keys by name: that
    list grows with every condition kind and a missing key would silently drop the column.
    The rows decide: a string that is a real leaf is a field; otherwise it is a value.
    """
    strings: Set[str] = set()
    sources: Set[str] = set()
    if depth > _MAX_CONDITION_DEPTH:
        return strings, sources
    if isinstance(node, str):
        strings.add(node)
    elif isinstance(node, dict):
        for key, val in node.items():
            if key == "source" and isinstance(val, str):
                sources.add(val)
                continue
            s, src = _condition_paths(val, depth + 1)
            strings |= s
            sources |= src
    elif isinstance(node, (list, tuple)):
        for item in node:
            s, src = _condition_paths(item, depth + 1)
            strings |= s
            sources |= src
    return strings, sources


def _leaf_index(logs: Dict[str, List[Dict]]) -> Callable[[str], Dict[str, str]]:
    """``leaves_of(source) -> {normalised leaf: the leaf as the ROWS spell it}``, lazily.

    Shared by every pin below so a source's rows are flattened once however many
    declarations name it.
    """
    index: Dict[str, Dict[str, str]] = {}

    def leaves_of(source: str) -> Dict[str, str]:
        if source not in index:
            found: Dict[str, str] = {}
            for row in (logs or {}).get(source, [])[:_MAX_ROWS_PER_SOURCE]:
                if isinstance(row, dict):
                    for leaf in flatten_leaves(row):
                        found.setdefault(_norm_path(leaf), leaf)
            index[source] = found
        return index[source]

    return leaves_of


def _pin(found: Dict[str, str], candidates: Iterable[str]) -> Set[str]:
    """The leaves of one source that a declaration's candidate strings name.

    Exact match first. On a miss the candidate is read as a container prefix: a
    ``data_map`` path names an array of records with no leaf of its own, and every
    column beneath it is kept. Prose (spaces in the text) is skipped.
    """
    out: Set[str] = set()
    for cand in candidates:
        text = str(cand).strip()
        if not text or any(ch.isspace() for ch in text):
            continue
        norm = _norm_path(text)
        leaf = found.get(norm)
        if leaf:
            out.add(leaf)
            continue
        stem = norm + "_"
        out.update(spelled for n, spelled in found.items() if n.startswith(stem))
    return out


def _condition_field_relevance(
    ruleset_spec: Optional[Dict[str, Any]],
    logs: Dict[str, List[Dict]],
    leaves_of: Optional[Callable[[str], Dict[str, str]]] = None,
) -> Dict[str, Set[str]]:
    """{source: leaf paths} the adjudicating procedure's own conditions read.

    A condition's field is what its pass/fail rests on; the trimmer has no other way to
    know about it. A condition naming no source is asked of every retrieved source
    (permissive in the direction that keeps a column).
    """
    if not isinstance(ruleset_spec, dict):
        return {}
    conditions = ruleset_spec.get("conditions")
    if not isinstance(conditions, list):
        return {}
    source_map = _logical_sources(ruleset_spec)
    resolve = leaves_of or _leaf_index(logs)

    out: Dict[str, Set[str]] = defaultdict(set)
    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        strings, logicals = _condition_paths(cond)
        targets = {source_map[name] for name in logicals if name in source_map}
        # A condition that names no source (`field_equality` carries its source per side,
        # `stub` carries none) is asked of every retrieved source. Over-keeping costs one
        # render row; under-keeping is the defect this function exists for.
        for source in sorted(targets) or sorted((logs or {}).keys()):
            out[source] |= _pin(resolve(source), strings)
    return {s: set(v) for s, v in out.items()}


def _logical_sources(ruleset_spec: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """The ruleset's own ``{logical name: physical source}`` map, or ``{}``."""
    if not isinstance(ruleset_spec, dict):
        return {}
    return {
        str(k): str(v)
        for k, v in (ruleset_spec.get("sources") or {}).items()
        if k and v
    }


def _alert_facts_field_relevance(
    ruleset_spec: Optional[Dict[str, Any]],
    logs: Dict[str, List[Dict]],
    leaves_of: Optional[Callable[[str], Dict[str, str]]] = None,
) -> Dict[str, Set[str]]:
    """{source: leaf paths} the ``alert_record:`` block reads.

    ``confirm_fields`` are redirected to the entry's own ``confirm_in:`` sources,
    matching what ``pack_validate._path_lists`` applies.
    """
    block = (ruleset_spec or {}).get("alert_record") if isinstance(ruleset_spec, dict) else None
    if not isinstance(block, dict):
        return {}
    source_map = _logical_sources(ruleset_spec)
    resolve = leaves_of or _leaf_index(logs)
    alert_source = source_map.get(
        str(block.get("source") or ""), str(block.get("source") or "")
    )
    out: Dict[str, Set[str]] = defaultdict(set)

    def add(source: str, node: Any) -> None:
        if source and node is not None:
            out[source] |= _pin(resolve(source), _condition_paths(node)[0])

    add(alert_source, block.get("label_fields"))
    entries = [block.get("identify"), block.get("declares")]
    for group in entries:
        for entry in group if isinstance(group, list) else []:
            if not isinstance(entry, dict):
                continue
            add(alert_source, entry.get("fields"))
            confirm = entry.get("confirm_fields")
            for logical in entry.get("confirm_in") or []:
                add(source_map.get(str(logical), str(logical)), confirm)
    return {s: set(v) for s, v in out.items() if v}


def _detect_time_field(
    source: str, leaves: List[str], rows: List[Dict], resolved_keys: List[Any]
) -> str:
    """Best-effort per-source timestamp field: resolved-key time_fields first, then the
    finest-grained leaf whose name-tokens hit _TIME_TOKENS, then a leaf whose values
    actually parse and look like dates."""
    for key in resolved_keys or []:
        tf = (getattr(key, "time_fields", {}) or {}).get(source)
        if tf and tf in leaves:
            return tf
    # Granularity-based selection so the chronology and the correlation keys agree.
    # pick_time_field measures cardinality across sample rows, not schema order.
    picked = pick_time_field(list(leaves), rows or [])
    if picked:
        return picked
    # Value-sniff: first leaf whose values are date-shaped and parse. The shape test
    # keeps bare counters out since _parse_ts treats any digit string as an epoch.
    sample = rows[:25]
    for leaf in leaves:
        parsed = 0
        seen = 0
        for row in sample:
            for v in resolve_path(row, leaf):
                seen += 1
                if _TS_HINT_RE.match(str(v).strip()) and parse_timestamp(v):
                    parsed += 1
        if seen and parsed >= 0.6 * seen:
            return leaf
    return ""


def _actor_types(pack_actor_types: Optional[List[str]] = None) -> List[str]:
    """The pack's actor types first, then the engine's generic ones (deduped)."""
    out = [str(t).lower() for t in (pack_actor_types or []) if str(t).strip()]
    for t in _ACTOR_ENTITY_TYPES:
        if t not in out:
            out.append(t)
    return out


def _actor_name_tokens(
    field_name_hints: Optional[Dict[str, str]] = None,
    actor_entity_types: Optional[List[str]] = None,
) -> Set[str]:
    """Name tokens marking a leaf as the acting identity, with pack tokens unioned in.

    Pack ``field_name_hints`` tokens that map to an actor-role type are added; the
    generic tokens stay so a source may spell it either way.
    """
    types = set(_actor_types(actor_entity_types))
    out = set(_ACTOR_NAME_TOKENS)
    for token, etype in (field_name_hints or {}).items():
        if str(etype).strip().lower() in types:
            tok = str(token).strip().lower()
            if tok:
                out.add(tok)
    return out


def _detect_actor_field(
    source: str,
    leaves: List[str],
    entity_map: Dict[str, Dict[str, str]],
    resolved_keys: List[Any],
    actor_entity_types: Optional[List[str]] = None,
    field_name_hints: Optional[Dict[str, str]] = None,
) -> str:
    """Field carrying the acting identity. Prefer an identity entity mapped for this
    source (via entity_map / resolved keys); else a leaf whose name-tokens suggest an
    identity.

    ``actor_entity_types`` is the pack's ``role: actor`` declaration and is tried in its own
    declaration ORDER before the engine's generic list, because a domain whose acting identity
    has a domain-specific type name would otherwise never be recognised as the actor.
    """
    smap = (entity_map or {}).get(source, {}) or {}
    types = _actor_types(actor_entity_types)
    for etype in types:
        if smap.get(etype) and smap[etype] in leaves:
            return smap[etype]
    for key in resolved_keys or []:
        if str(getattr(key, "entity_hint", "")).lower() in types:
            field = (getattr(key, "sources", {}) or {}).get(source)
            if field and field in leaves:
                return field
    name_tokens = _actor_name_tokens(field_name_hints, actor_entity_types)
    for leaf in leaves:
        if set(_name_tokens(leaf)) & name_tokens:
            return leaf
    return ""


def _detect_action_field(source: str, leaves: List[str], rows: List[Dict]) -> str:
    """A low-cardinality categorical leaf that reads like an action/verb/status.

    A candidate must match an action token, not a time token (``event_timestamp``
    carries ``event`` but is high-cardinality), and be low-cardinality short-valued
    across the sampled rows.
    """
    sample = [r for r in (rows or []) if isinstance(r, dict)][:200]
    for leaf in leaves:
        tokens = set(_name_tokens(leaf))
        if not (tokens & _ACTION_TOKENS):
            continue
        if tokens & _TIME_TOKENS:  # event_timestamp / *_date / *_time → not a verb
            continue
        vals = [
            v for v in (_first_scalar(r, leaf) for r in sample) if v not in (None, "")
        ]
        if not vals:
            continue
        distinct = set(vals)
        # A real action/status column has few distinct, short values; a timestamp or id
        # has roughly one distinct value per row or long values.
        if len(distinct) > max(20, len(vals) // 2):
            continue
        if any(len(str(v)) > 40 for v in distinct):
            continue
        if all(_looks_like_ts(str(v)) for v in list(distinct)[:10]):
            continue
        return leaf
    return ""


def _looks_like_ts(v: str) -> bool:
    """Heuristic: does a string value look like an ISO timestamp / epoch number?"""
    if _TS_HINT_RE.match(v):
        return True
    s = v.strip()
    return s.isdigit() and len(s) >= 10  # epoch seconds/millis


# Minimum subject-value length for containment matching. Short codes occur inside
# unrelated identifiers and would match the whole population.
_SUBJECT_CONTAINMENT_MIN_LEN = 4


def _names_a_subject(text: str, subjects: Set[str]) -> bool:
    """Does a document-length value name one of the subjects?

    Used when a source binds an entity to a document field (a notification body) where
    equality cannot match. Bounded on both sides by a non-alphanumeric character so a
    short identifier cannot match inside a longer unrelated one. Floored at
    :data:`_SUBJECT_CONTAINMENT_MIN_LEN`.
    """
    low = text.lower()
    for subj in subjects:
        if len(subj) < _SUBJECT_CONTAINMENT_MIN_LEN:
            continue
        for m in re.finditer(re.escape(subj), low):
            before = low[m.start() - 1] if m.start() else ""
            after = low[m.end()] if m.end() < len(low) else ""
            if not before.isalnum() and not after.isalnum():
                return True
    return False


def _entities_for_row(
    row: Dict,
    source_map: Dict[str, str],
    typed_subjects: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, str]:
    """{entity_type: value} for the entity fields mapped for this source that resolve.

    When a mapped field holds a document rather than a datum, the entity is replaced by
    the incident's own values of that type named by the document (``typed_subjects``).
    Nothing is emitted when none are found. Without ``typed_subjects`` the value passes
    through unchanged.
    """
    out: Dict[str, str] = {}
    for etype, field in (source_map or {}).items():
        val = _first_scalar(row, field)
        if not val:
            continue
        if len(val) > _BLOB_LEN and typed_subjects is not None:
            named = [
                v
                for v in (typed_subjects.get(etype) or [])
                if _names_a_subject(val, {str(v).strip().lower()})
            ]
            if named:
                out[etype] = ", ".join(named)
            continue
        out[etype] = val
    return out


def _actor_for_row(
    row: Dict,
    afield: str,
    entities: Dict[str, str],
    actor_types: List[str],
) -> str:
    """The row's acting identity, never a document.

    When the actor field holds a document, falls back to the actor-role entity resolved
    for the row. Falls back to empty rather than returning the document.
    """
    val = _first_scalar(row, afield) if afield else ""
    if val and len(val) <= _BLOB_LEN:
        return val
    for etype in actor_types:
        candidate = str(entities.get(etype) or "").strip()
        if candidate and len(candidate) <= _BLOB_LEN:
            # Take only the first if the entity value is a comma-joined pair.
            return candidate.split(", ")[0]
    return ""


# --- chronology -------------------------------------------------------------


def _subject_set(entity_values: Optional[List[str]]) -> Set[str]:
    """Lower-cased surface text for subject matching (see :func:`_names_a_subject`)."""
    return {str(v).strip().lower() for v in (entity_values or []) if v}


def _subject_keys(entity_values: Optional[List[str]]) -> Set[str]:
    """Normalised form for identifier comparison (see ``_matches_any_identifier``)."""
    return {_norm_identifier(v) for v in (entity_values or []) if _norm_identifier(v)}


def _event_is_subject(
    actor: str,
    entities: Dict[str, Any],
    subjects: Set[str],
    subject_keys: Optional[Set[str]] = None,
) -> bool:
    """True if the actor or any touched entity value is an incident person-of-interest.

    ``subject_keys`` is the normalised form of ``subjects`` (:func:`_subject_keys`),
    enabling prefix-tolerant matching against identifiers stored at a different surface
    width. Absent, only exact matching applies.
    """
    if not subjects and not subject_keys:
        return False
    keys = subject_keys or set()
    if actor and (
        actor.strip().lower() in subjects or _matches_any_identifier(actor, keys)
    ):
        return True
    for v in (entities or {}).values():
        for item in v if isinstance(v, (list, set, tuple)) else [v]:
            s = str(item).strip()
            if s.lower() in subjects:
                return True
            if len(s) <= _BLOB_LEN and _matches_any_identifier(s, keys):
                return True
            if len(s) > _BLOB_LEN and _names_a_subject(s, subjects):
                return True
    return False


def _decoded_events(
    source: str,
    rows: List[Dict],
    paths: List[str],
    row_context: Dict[int, Tuple[str, bool]],
    subjects: Set[str],
    subject_keys: Optional[Set[str]] = None,
) -> List[ChronologyEvent]:
    """One event per record in this source's decoded tables (``encoded_fields``).

    Each record becomes a ``ChronologyEvent``; the actor is inherited from the parent row
    since a nested record names what was done, not who did it. Deduped per
    ``(record, path)`` since the same table may appear on multiple rows of one source.
    Subject-hood is also inherited from the parent row.
    """
    out: List[ChronologyEvent] = []
    seen: Set[tuple] = set()
    for idx, row in enumerate(rows):
        for path in paths:
            for rec in records_at(row, path):
                # The record's own values, as strings.
                ents = {
                    str(k): str(v)
                    for k, v in rec.items()
                    if v not in (None, "") and _scalar_like(v)
                }
                if not ents:
                    continue
                dt = None
                for k, v in rec.items():
                    if any(t in _TIME_TOKENS for t in _name_tokens(str(k))):
                        dt = parse_timestamp(v)
                        if dt:
                            break
                key = (path, tuple(sorted(ents.items())))
                if key in seen:
                    continue
                seen.add(key)
                actor, row_is_subject = row_context.get(idx, ("", False))
                out.append(
                    ChronologyEvent(
                        timestamp=_iso(dt),
                        epoch=dt.timestamp() if dt else None,
                        source=source,
                        actor=actor,
                        action=_decoded_action(rec) or path.rsplit(".", 1)[-1],
                        entities=ents,
                        is_subject=row_is_subject
                        or _event_is_subject(actor, ents, subjects, subject_keys),
                    )
                )
        if len(out) >= _MAX_DECODED_EVENTS_PER_SOURCE:
            logger.warning(
                "Source '%s' decoded more than %d records; the chronology holds the first "
                "%d, so any count taken from it is a LOWER BOUND.",
                source,
                _MAX_DECODED_EVENTS_PER_SOURCE,
                _MAX_DECODED_EVENTS_PER_SOURCE,
            )
            return out[:_MAX_DECODED_EVENTS_PER_SOURCE]
    return out


def _scalar_like(v: Any) -> bool:
    """True for a value that renders as one token (a decoded column always should)."""
    return isinstance(v, (str, int, float, bool))


def _decoded_action(rec: Dict[str, Any]) -> str:
    """The record's action/verb column, by the same name tokens the row scan uses."""
    for k, v in rec.items():
        tokens = set(_name_tokens(str(k)))
        if tokens & _ACTION_TOKENS and not (tokens & _TIME_TOKENS) and _scalar_like(v):
            s = str(v).strip()
            if s and len(s) <= 60:
                return s
    return ""


def build_chronology(
    logs: Dict[str, List[Dict]],
    entity_map: Dict[str, Dict[str, str]],
    resolved_keys: List[Any],
    entity_values: List[str],
    max_events: int = _MAX_CHRONOLOGY_EVENTS,
    schema: Optional[Dict[str, List[str]]] = None,
    actor_entity_types: Optional[List[str]] = None,
    field_name_hints: Optional[Dict[str, str]] = None,
    decoded_paths: Optional[Dict[str, List[str]]] = None,
    typed_subjects: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[ChronologyEvent], bool]:
    """Merge rows across sources into time-ordered ChronologyEvents.

    Returns ``(events, was_aggregated)``. When the raw event count exceeds ``max_events``
    or any source is large, events are collapsed into (actor, action, hour-bucket) groups.

    ``decoded_paths`` is ``{source: [path, ...]}`` from ``encoded_fields.decoded_tables``;
    each record becomes its own event.

    ``typed_subjects`` is ``{entity_type: [value, ...]}`` for document-mapped entity
    fields (see :func:`_entities_for_row`).
    """
    schema = schema or {}
    subjects = _subject_set(entity_values)
    subject_keys = _subject_keys(entity_values)
    actor_types = _actor_types(actor_entity_types)
    raw: List[ChronologyEvent] = []
    volume_trigger = False
    for source, rows in (logs or {}).items():
        rows = [r for r in (rows or []) if isinstance(r, dict)]
        if len(rows) > _AGG_VOLUME_TRIGGER:
            volume_trigger = True
        leaves = (
            schema.get(source) or list(flatten_leaves(rows[0]).keys()) if rows else []
        )
        tfield = _detect_time_field(source, leaves, rows, resolved_keys)
        afield = _detect_actor_field(
            source,
            leaves,
            entity_map,
            resolved_keys,
            actor_entity_types,
            field_name_hints,
        )
        actfield = _detect_action_field(source, leaves, rows)
        smap = (entity_map or {}).get(source, {}) or {}
        row_context: Dict[int, Tuple[str, bool]] = {}  # (actor, is_subject) per row index
        for idx, row in enumerate(rows[:_MAX_ROWS_PER_SOURCE]):
            dt = parse_timestamp(row.get(tfield)) if tfield and tfield in row else None
            if dt is None and tfield:
                vals = resolve_path(row, tfield)
                dt = parse_timestamp(vals[0]) if vals else None
            entities = _entities_for_row(row, smap, typed_subjects)
            actor = _actor_for_row(row, afield, entities, actor_types)
            is_subject = _event_is_subject(actor, entities, subjects, subject_keys)
            row_context[idx] = (actor, is_subject)
            raw.append(
                ChronologyEvent(
                    timestamp=_iso(dt),
                    epoch=dt.timestamp() if dt else None,
                    source=source,
                    actor=actor,
                    action=_first_scalar(row, actfield) if actfield else "",
                    entities=entities,
                    is_subject=is_subject,
                )
            )
        for path in (decoded_paths or {}).get(source, []):
            decoded = _decoded_events(
                source,
                rows[:_MAX_ROWS_PER_SOURCE],
                [path],
                row_context,
                subjects,
                subject_keys,
            )
            if decoded:
                raw.extend(decoded)
                logger.info(
                    "Chronology: %d event(s) from '%s' decoded table %s.",
                    len(decoded),
                    source,
                    path,
                )

    raw.sort(key=lambda e: (e.epoch is None, e.epoch or 0.0, e.source))

    if len(raw) <= max_events and not volume_trigger:
        return raw, False

    # Volume-triggered aggregation: collapse by (actor, action, hour-bucket). Subject
    # events keep their own groups (is_subject in the key) so they never merge into a
    # background bucket and can be rendered first.
    groups: Dict[tuple, List[ChronologyEvent]] = defaultdict(list)
    for ev in raw:
        bucket = ""
        if ev.epoch is not None:
            bucket = _bucket_key(
                datetime.fromtimestamp(ev.epoch, tz=timezone.utc), "1h"
            )
        groups[(ev.actor, ev.action, ev.source, bucket, ev.is_subject)].append(ev)

    aggregated: List[ChronologyEvent] = []
    for (actor, action, source, bucket, is_subj), evs in groups.items():
        examples: List[str] = []
        for e in evs[:_MAX_EXAMPLES_PER_GROUP]:
            if e.entities:
                examples.append(", ".join(f"{k}={v}" for k, v in e.entities.items()))
        first = min((e.epoch for e in evs if e.epoch is not None), default=None)
        aggregated.append(
            ChronologyEvent(
                timestamp=bucket or (evs[0].timestamp if evs else ""),
                epoch=first,
                source=source,
                actor=actor,
                action=action,
                entities=evs[0].entities if evs else {},
                count=len(evs),
                examples=examples,
                is_subject=is_subj,
            )
        )
    aggregated.sort(key=lambda e: (e.epoch is None, e.epoch or 0.0, -e.count))
    return aggregated, True


# --- actor attribution ------------------------------------------------------


def attribute_actors(
    events: List[ChronologyEvent], entity_values: Optional[List[str]] = None
) -> List[ActorRollup]:
    """Roll chronology events up by acting identity.

    An actor whose id or touched entity matches an incident value is marked
    ``is_subject=True`` and ranked above higher-volume background actors.
    """
    subjects = _subject_set(entity_values)
    # Prefix-tolerant so an identifier stored wider than the alert names it still ranks
    # as the subject.
    subject_keys = _subject_keys(entity_values)

    def _is_subject(actor: str, entities: Dict[str, Any]) -> bool:
        if actor and (
            actor.strip().lower() in subjects
            or _matches_any_identifier(actor, subject_keys)
        ):
            return True
        for vals in (entities or {}).values():
            for v in vals if isinstance(vals, (list, set)) else [vals]:
                s = str(v).strip()
                if s.lower() in subjects:
                    return True
                # Only a datum is compared as an identifier, not a document-length value.
                if len(s) <= _BLOB_LEN and _matches_any_identifier(s, subject_keys):
                    return True
        return False

    by_actor: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        actor = ev.actor or "(unattributed)"
        agg = by_actor.setdefault(
            actor,
            {
                "event_count": 0,
                "action_counts": defaultdict(int),
                "epochs": [],
                "sources": set(),
                "entities": defaultdict(set),
            },
        )
        agg["event_count"] += ev.count
        if ev.action:
            agg["action_counts"][ev.action] += ev.count
        if ev.epoch is not None:
            agg["epochs"].append(ev.epoch)
        agg["sources"].add(ev.source)
        for etype, val in (ev.entities or {}).items():
            agg["entities"][etype].add(val)

    rollups: List[ActorRollup] = []
    for actor, agg in by_actor.items():
        epochs = sorted(agg["epochs"])
        rollups.append(
            ActorRollup(
                actor=actor,
                event_count=agg["event_count"],
                action_counts=dict(agg["action_counts"]),
                first_seen=(
                    _iso(datetime.fromtimestamp(epochs[0], tz=timezone.utc))
                    if epochs
                    else ""
                ),
                last_seen=(
                    _iso(datetime.fromtimestamp(epochs[-1], tz=timezone.utc))
                    if epochs
                    else ""
                ),
                sources=sorted(agg["sources"]),
                entities_touched={
                    k: sorted(v)[:10] for k, v in agg["entities"].items()
                },
                is_subject=_is_subject(actor, agg["entities"]),
            )
        )
    # Subjects (incident persons-of-interest) first, then by volume within each group.
    rollups.sort(key=lambda r: (not r.is_subject, -r.event_count))
    return rollups[:_MAX_ACTORS]


# --- per-source column trimming --------------------------------------------


def trim_columns(
    rows: List[Dict[str, Any]],
    relevant_fields: Set[str],
    max_cols: int = _MAX_TRIMMED_COLS,
) -> Tuple[List[str], List[str], Dict[str, Dict[str, int]]]:
    """Pick the columns worth keeping for the LLM.

    Returns ``(kept_columns, dropped_columns, distributions)``. Drops constant columns
    (one value >= _CONSTANT_COL_RATIO of rows), fully-masked columns, and blob columns
    (mean value length > _BLOB_LEN), unless the column is in ``relevant_fields``, which
    is always kept. Remaining columns are ranked by distinct-value count (discriminating
    power) and capped at ``max_cols``. ``distributions`` gives top values per kept column.
    """
    # Flatten every row to dotted leaves; collect per-column value lists.
    col_values: Dict[str, List[str]] = defaultdict(list)
    n = 0
    for row in rows[:_MAX_ROWS_PER_SOURCE]:
        if not isinstance(row, dict):
            continue
        n += 1
        for path, vals in flatten_leaves(row).items():
            for v in vals:
                col_values[path].append(str(v))
    if n == 0:
        return [], [], {}

    kept: List[str] = []
    dropped: List[str] = []
    scored: List[Tuple[int, str]] = []  # (distinct_count, column) for ranking
    for col, vals in col_values.items():
        is_relevant = col in relevant_fields
        distinct = set(vals)
        # Constant column.
        if vals:
            top_count = (
                max(vals.count(x) for x in distinct) if len(distinct) <= 50 else 1
            )
            if not is_relevant and len(distinct) == 1:
                dropped.append(col)
                continue
            if not is_relevant and top_count / len(vals) >= _CONSTANT_COL_RATIO:
                dropped.append(col)
                continue
        # Masked column.
        if not is_relevant and vals and all(_MASK_RE.match(v) for v in vals):
            dropped.append(col)
            continue
        # Blob column.
        if not is_relevant and vals:
            mean_len = sum(len(v) for v in vals) / len(vals)
            if mean_len > _BLOB_LEN:
                dropped.append(col)
                continue
        scored.append((len(distinct), col))

    # Relevant fields first (stable), then most-discriminating.
    scored.sort(key=lambda t: (t[1] not in relevant_fields, -t[0], t[1]))
    for _distinct, col in scored:
        # The cap bounds discriminating columns only; relevant columns sort first and are
        # always kept regardless of cap.
        if len(kept) >= max_cols and col not in relevant_fields:
            dropped.append(col)
            continue
        kept.append(col)

    distributions: Dict[str, Dict[str, int]] = {}
    for col in kept:
        counts: Dict[str, int] = defaultdict(int)
        for v in col_values[col]:
            counts[v] += 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[
            :_MAX_DIST_VALUES
        ]
        distributions[col] = {k: c for k, c in top}
    return kept, dropped, distributions


def _adjudicated_values(
    kept: List[str],
    distributions: Dict[str, Dict[str, int]],
    adjudicated: Set[str],
) -> Dict[str, List[str]]:
    """Values behind the columns a condition was decided on, read off the tallies.

    A column is omitted unless it has at least one non-blank value; an all-empty column
    is genuinely unavailable and must not appear as a reading.
    """
    out: Dict[str, List[str]] = {}
    for col in kept:
        if col not in adjudicated:
            continue
        vals = [
            str(v)[:_MAX_ADJ_VALUE_LEN]
            for v in (distributions.get(col) or {})
            if str(v).strip()
        ][:_MAX_ADJ_VALUES]
        if vals:
            out[col] = vals
    return out


def _adj_part(col: str, vals: List[str]) -> str:
    """One ``col=v1,v2`` unit of an adjudicated line. Whole units are the drop granularity:
    half a field path is unmatchable by the reader the line is written for."""
    return f"{col}={','.join(vals)}"


def _adj_line_caps(sources: List["SourceEvidence"], char_budget: int) -> Dict[str, int]:
    """Chars each source may spend restating the values its conditions were decided on.

    Derived from the render budget rather than fixed: when the budget can afford all
    sources in full nothing spills, and no source is squeezed below the minimum.
    """
    needs: Dict[str, int] = {}
    for s in sources:
        if s.adjudicated_values:
            needs[s.source] = sum(
                len(_adj_part(c, v)) + 2 for c, v in s.adjudicated_values.items()
            )
    if not needs:
        return {}
    total = max(_MIN_ADJ_LINE_LEN, int(char_budget * _ADJ_BUDGET_SHARE))
    if sum(needs.values()) <= total:
        return dict(needs)
    # Max-min fair share, ascending need. `_MIN_ADJ_LINE_LEN` is a floor on the share
    # and may overrun the ceiling; that overrun matches the behaviour a fixed cap had.
    caps: Dict[str, int] = {}
    left, remaining = total, len(needs)
    for src, need in sorted(needs.items(), key=lambda kv: (kv[1], kv[0])):
        share = max(_MIN_ADJ_LINE_LEN, max(0, left) // remaining)
        caps[src] = min(need, share)
        left -= caps[src]
        remaining -= 1
    return caps


def _build_source_evidence(
    logs: Dict[str, List[Dict]],
    relevance: Dict[str, Set[str]],
    row_caps: Optional[Dict[str, int]] = None,
    zero_row_meanings: Optional[Dict[str, str]] = None,
    adjudicated: Optional[Dict[str, Set[str]]] = None,
) -> List[SourceEvidence]:
    """One trimmed view per source.

    ``adjudicated`` is ``{source: {path, ...}}`` for the columns the ruleset's conditions
    and ``alert_record`` facts read. Their values are stored in ``adjudicated_values``
    and no budget cut may remove them.
    """
    caps = row_caps or {}
    meanings = zero_row_meanings or {}
    adj = adjudicated or {}
    out: List[SourceEvidence] = []
    for source, rows in (logs or {}).items():
        rows = [r for r in (rows or []) if isinstance(r, dict)]
        kept, dropped, dists = trim_columns(rows, relevance.get(source, set()))
        examples: List[Dict[str, Any]] = []
        for row in rows[:_MAX_SOURCE_EXAMPLES]:
            ex: Dict[str, Any] = {}
            for col in kept:
                vals = resolve_path(row, col)
                if vals:
                    ex[col] = vals[0] if len(vals) == 1 else vals[:5]
            if ex:
                examples.append(ex)
        # Exact match with cap means the backend had more; fewer is a complete answer.
        cap = caps.get(source) or 0
        # zero_rows_meaning is only emitted when the source is actually empty.
        out.append(
            SourceEvidence(
                source=source,
                record_count=len(rows),
                kept_columns=kept,
                dropped_columns=dropped,
                aggregated=len(rows) > _AGG_VOLUME_TRIGGER,
                row_limited=bool(cap and len(rows) >= cap),
                zero_rows_meaning=(
                    str(meanings.get(source) or "").strip() if not rows else ""
                ),
                distributions=dists,
                examples=examples,
                adjudicated_values=_adjudicated_values(
                    kept, dists, adj.get(source) or set()
                ),
            )
        )
    return out


# --- top-level orchestration ------------------------------------------------


def build_evidence(
    logs: Dict[str, List[Dict]],
    correlation_aggregations: Dict[str, Any],
    resolved_keys: List[Any],
    entity_map: Dict[str, Dict[str, str]],
    entity_values: List[str],
    char_budget: int = _DEFAULT_CHAR_BUDGET,
    schema: Optional[Dict[str, List[str]]] = None,
    cross_source_joins: Optional[List[Dict[str, Any]]] = None,
    subject_values: Optional[List[str]] = None,
    row_caps: Optional[Dict[str, int]] = None,
    actor_entity_types: Optional[List[str]] = None,
    field_name_hints: Optional[Dict[str, str]] = None,
    decoded_paths: Optional[Dict[str, List[str]]] = None,
    typed_subjects: Optional[Dict[str, List[str]]] = None,
    zero_row_meanings: Optional[Dict[str, str]] = None,
    ruleset_spec: Optional[Dict[str, Any]] = None,
) -> EvidencePack:
    """Assemble the EvidencePack, then degrade it to fit ``char_budget``.

    ``subject_values``: discriminating identities for person-of-interest flagging;
    defaults to ``entity_values``.
    ``row_caps``: ``{source: max_results}`` so a capped source is marked truncated.
    ``decoded_paths``: ``{source: [path, ...]}`` from ``encoded_fields.decoded_tables``;
    each decoded record becomes its own chronology event and decoded columns are pinned
    so ``trim_columns`` cannot drop them (they are constant-valued per original row).
    ``typed_subjects``: ``{entity_type: [value, ...]}``; consulted where a mapped field
    holds a document rather than a datum.
    ``zero_row_meanings``: ``{source: meaning}`` from ``KnowledgePack.zero_row_meanings``.
    ``ruleset_spec``: the adjudicating procedure's spec, read for condition and
    ``alert_record`` leaf paths to pin as relevant. Optional.
    """
    subjects = subject_values if subject_values is not None else entity_values
    relevance = _leaf_field_relevance(logs, resolved_keys, entity_map, decoded_paths)
    leaves_of = _leaf_index(logs)
    # Ruleset paths are pinned separately because a condition was already decided on them;
    # the budget ladder must not drop them.
    adjudicated: Dict[str, Set[str]] = {}
    for pinned in (
        _condition_field_relevance(ruleset_spec, logs, leaves_of),
        _alert_facts_field_relevance(ruleset_spec, logs, leaves_of),
    ):
        for source, paths in pinned.items():
            relevance.setdefault(source, set()).update(paths)
            adjudicated.setdefault(source, set()).update(paths)
    chronology, aggregated = build_chronology(
        logs,
        entity_map,
        resolved_keys,
        subjects,
        schema=schema,
        actor_entity_types=actor_entity_types,
        field_name_hints=field_name_hints,
        decoded_paths=decoded_paths,
        typed_subjects=typed_subjects,
    )
    actors = attribute_actors(chronology, subjects)
    sources = _build_source_evidence(
        logs, relevance, row_caps, zero_row_meanings, adjudicated
    )
    total = int((correlation_aggregations or {}).get("total_records", 0)) or sum(
        len(r or []) for r in (logs or {}).values()
    )
    joins = list(cross_source_joins or [])
    if not joins:
        # Fall back to the deterministic cross_source_overlap from aggregations.
        overlap = (correlation_aggregations or {}).get("cross_source_overlap", {}) or {}
        joins = [
            {"value": val, "sources": sorted(by_src.keys())}
            for val, by_src in overlap.items()
        ]

    pack = EvidencePack(
        chronology=chronology,
        chronology_aggregated=aggregated,
        actors=actors,
        sources=sources,
        cross_source_joins=joins[:_MAX_CHRONOLOGY_EVENTS],
        total_records=total,
        degraded=False,
        notes=[],
    )
    return degrade_to_budget(pack, char_budget)


def _rendered_len(pack: EvidencePack) -> int:
    """Full rendered length of the pack, ignoring the budget clip.

    The ladder must measure the uncapped length; calling ``render_for_prompt(pack)``
    at the default budget returns ``min(true_length, default_budget)`` and makes
    every rung's gate trivially satisfied.
    """
    return len(render_for_prompt(pack, _NO_CLIP))


def _half_lens(pack: EvidencePack) -> Tuple[int, int]:
    """Rendered lengths of the two halves the renderer sub-budgets: (head, sources).

    Renders once at ``_NO_CLIP`` and splits on the ``[SOURCES]`` heading so each rung
    is charged to the half it actually shrinks.
    """
    text = render_for_prompt(pack, _NO_CLIP)
    marker = "\n[SOURCES]"
    at = text.find(marker)
    if at < 0:
        return len(text), 0
    return at, len(text) - at


def degrade_to_budget(pack: EvidencePack, char_budget: int) -> EvidencePack:
    """Monotonic degradation ladder: shrink the pack until its rendering fits budget.

    Each rung is gated on the half it shrinks, not on the total. The renderer
    sub-budgets: the head gets at most ``_HEAD_BUDGET_SHARE`` and sources the rest.
    A gate on the total is unsatisfiable when one half alone exceeds the budget,
    making the ladder all-or-nothing. Per-half gating is strictly stronger (both
    halves inside their caps implies the total is), so the total stays the early exit.
    """
    if _rendered_len(pack) <= char_budget:
        return pack

    usable = max(0, char_budget - (len(_CLIP_SUFFIX) + _CLIP_NOTE_RESERVE))
    head_cap = int(usable * _HEAD_BUDGET_SHARE)

    def _over_head() -> bool:
        return _half_lens(pack)[0] > head_cap

    def _over_sources() -> bool:
        head, srcs = _half_lens(pack)
        # The head share is a ceiling; unused head budget is available for sources.
        return srcs > max(0, usable - min(head, head_cap))

    def _note(msg: str) -> None:
        pack.degraded = True
        if msg not in pack.notes:
            pack.notes.append(msg)

    # 1. Drop per-source example rows.
    if _over_sources():
        for s in pack.sources:
            s.examples = []
        _note("dropped per-source example rows to fit budget")
        if _rendered_len(pack) <= char_budget:
            return pack

    # 2. Reduce distributions to top-3 per column.
    if _over_sources():
        for s in pack.sources:
            s.distributions = {
                c: dict(list(d.items())[:3]) for c, d in s.distributions.items()
            }
        _note("reduced distributions to top-3")
        if _rendered_len(pack) <= char_budget:
            return pack

    # 3. Re-collapse chronology to a coarser (day) bucket.
    if not pack.chronology_aggregated and _over_head():
        groups: Dict[tuple, List[ChronologyEvent]] = defaultdict(list)
        for ev in pack.chronology:
            bucket = ""
            if ev.epoch is not None:
                bucket = _bucket_key(
                    datetime.fromtimestamp(ev.epoch, tz=timezone.utc), "1d"
                )
            # `is_subject` is part of the group key so subject events are not merged
            # into background actor buckets.
            groups[(ev.actor, ev.action, ev.source, bucket, ev.is_subject)].append(ev)
        collapsed = [
            ChronologyEvent(
                timestamp=b or (evs[0].timestamp if evs else ""),
                epoch=min((e.epoch for e in evs if e.epoch is not None), default=None),
                source=src,
                actor=a,
                action=act,
                entities=evs[0].entities if evs else {},
                count=len(evs),
                is_subject=is_subj,
            )
            for (a, act, src, b, is_subj), evs in groups.items()
        ]
        collapsed.sort(key=lambda e: (e.epoch is None, e.epoch or 0.0, -e.count))
        pack.chronology = collapsed
        pack.chronology_aggregated = True
        _note("collapsed chronology to daily buckets")
        if _rendered_len(pack) <= char_budget:
            return pack

    # 4. Cap chronology and actors, subjects first. Ranking by volume alone would drop
    # incident events (the subject acts infrequently) and keep background identities.
    if _over_head():
        pack.chronology = sorted(
            pack.chronology, key=lambda e: (not e.is_subject, -e.count)
        )[:50]
        pack.actors = sorted(
            pack.actors, key=lambda a: (not a.is_subject, -a.event_count)
        )[:15]
        _note("capped chronology to top-50 events and actors to top-15, subjects first")
        if _rendered_len(pack) <= char_budget:
            return pack

    # 5. Drop distributions entirely. `adjudicated_values` is not a distribution and
    # is never dropped; it records readings a condition already made.
    if _over_sources():
        for s in pack.sources:
            s.distributions = {}
        _note("dropped distributions")
        if _rendered_len(pack) <= char_budget:
            return pack

    # 6. Last resort: hand the overflow to the renderer, which trims whole lines per half
    # and only clips the tail if even that does not fit. Which of the two happened is the
    # renderer's own note; this one records that the ladder ran out of rungs.
    _note(CLIP_NOTE)
    return pack


# --- rendering --------------------------------------------------------------

_CLIP_SUFFIX = "\n... [evidence truncated to budget]"
# Room held back for the notes line's own growth: the trim appends up to two notes of
# its own, and they are the only place the render admits it dropped anything.
_CLIP_NOTE_RESERVE = 180


def _render_notes(notes: List[str]) -> str:
    """The notes line, or '' when there is nothing to note."""
    return ("\n[NOTES] " + "; ".join(notes)) if notes else ""


def _clip_lines(lines: List[str], budget: int, what: str) -> Tuple[List[str], str]:
    """Drop whole lines off the END of a section until it fits ``budget``.

    Returns the kept lines plus a note naming how many went, because a section that
    simply stops early reads exactly like a section that had nothing more to say. Always
    keeps at least the first line.
    """
    kept: List[str] = []
    used = 0
    for i, ln in enumerate(lines):
        cost = len(ln) + 1
        if kept and used + cost > budget:
            return kept, f"{len(lines) - i} {what} line(s) omitted to fit budget"
        kept.append(ln)
        used += cost
    return kept, ""


def _thin_source_lines(lines: List[str], budget: int) -> Tuple[List[str], str]:
    """Fit the sources block by dropping per-source detail, never a source's count line.

    Each ``- name: N record(s)`` line carries the volume finding; indented distribution
    and example lines under it are dropped first. ``_ADJ_LINE_PREFIX`` lines are mandatory
    too: they hold values a condition was already decided on.
    """
    head_idx = {
        i
        for i, ln in enumerate(lines)
        if not ln.startswith("    ") or ln.startswith(_ADJ_LINE_PREFIX)
    }
    mandatory = sum(len(lines[i]) + 1 for i in head_idx)
    if mandatory > budget:
        # Not even the count lines fit: clip them and report how many sources went.
        return _clip_lines([lines[i] for i in sorted(head_idx)], budget, "source")
    keep = set(head_idx)
    used = mandatory
    dropped = 0
    for i, ln in enumerate(lines):
        if i in keep:
            continue
        cost = len(ln) + 1
        if used + cost > budget:
            dropped += 1
            continue
        keep.add(i)
        used += cost
    note = (
        f"{dropped} per-source detail line(s) omitted to fit budget" if dropped else ""
    )
    return [ln for i, ln in enumerate(lines) if i in keep], note


def render_for_prompt(
    pack: EvidencePack, char_budget: int = _DEFAULT_CHAR_BUDGET
) -> str:
    """Compact, labeled TEXT (not JSON) the LLM stages embed as evidence."""
    lines: List[str] = []
    lines.append(
        f"=== INVESTIGATION EVIDENCE (total_records={pack.total_records}"
        f"{', aggregated' if pack.chronology_aggregated else ''}"
        f"{', degraded' if pack.degraded else ''}) ==="
    )

    # Chronology section. Subject events first; bounded background follows for context.
    chrono = list(pack.chronology or [])
    subj_events = [e for e in chrono if getattr(e, "is_subject", False)]
    bg_events = [e for e in chrono if not getattr(e, "is_subject", False)]

    def _fmt_event(ev) -> str:
        ent = ", ".join(f"{k}={v}" for k, v in (ev.entities or {}).items())
        return (
            f"{ev.timestamp or '?'} | {ev.source} | {ev.actor or '-'} | "
            f"{ev.action or '-'} | {ent or '-'} | {ev.count}"
        )

    lines.append("\n[CHRONOLOGY] (time-ordered; subject events first)")
    lines.append("ts | source | actor | action | entities | count")
    if subj_events:
        lines.append(f"# Subject events ({len(subj_events)}):")
        for ev in subj_events[:_MAX_SUBJECT_CHRONO]:
            lines.append(_fmt_event(ev))
    if bg_events:
        shown = bg_events[:_MAX_BACKGROUND_CHRONO]
        label = (
            f"# Background activity in the same window "
            f"(showing {len(shown)} of {len(bg_events)}; not implicated):"
        )
        lines.append(label)
        for ev in shown:
            lines.append(_fmt_event(ev))
    if not subj_events and not bg_events:
        lines.append("(no time-ordered events)")

    # Actor attribution section. Show all subjects; cap background actors.
    lines.append(
        "\n[ACTOR ATTRIBUTION] (who did what, when, where; "
        "SUBJECT = person-of-interest from the incident)"
    )
    subj_actors = [a for a in pack.actors if getattr(a, "is_subject", False)]
    bg_actors = [a for a in pack.actors if not getattr(a, "is_subject", False)]
    rendered_actors = subj_actors + bg_actors[:_MAX_BACKGROUND_ACTORS]

    def _fmt_actor_line(a) -> str:
        actions = ", ".join(f"{k}:{v}" for k, v in a.action_counts.items())
        touched = "; ".join(f"{k}={','.join(v)}" for k, v in a.entities_touched.items())
        tag = "SUBJECT " if getattr(a, "is_subject", False) else ""
        return (
            f"- {tag}{a.actor}: {a.event_count} events "
            f"[{a.first_seen or '?'} .. {a.last_seen or '?'}] "
            f"sources={','.join(a.sources)}"
            + (f" actions=({actions})" if actions else "")
            + (f" entities=({touched})" if touched else "")
        )

    for a in rendered_actors:
        lines.append(_fmt_actor_line(a))
    if len(bg_actors) > _MAX_BACKGROUND_ACTORS:
        lines.append(
            f"- (+{len(bg_actors) - _MAX_BACKGROUND_ACTORS} more background identities "
            "in the same window, not implicated)"
        )

    lines.append("\n[CROSS-SOURCE JOINS] (values appearing in >1 source)")
    if pack.cross_source_joins:
        for j in pack.cross_source_joins:
            val = j.get("value", "")
            srcs = ", ".join(j.get("sources", []))
            extra = f" ({j['time_window']})" if j.get("time_window") else ""
            lines.append(f"- {val} in [{srcs}]{extra}")
    else:
        lines.append("- (none)")

    # Per-source volume and signal. Distributions and examples are bounded to prevent
    # a high-volume source from dominating the budget with low-signal tallies.
    head_lines = list(lines)
    lines = ["\n[SOURCES]"]
    adj_caps = _adj_line_caps(list(pack.sources), char_budget)
    for s in pack.sources:
        # A source that returned exactly its row cap is truncated; the count is an
        # artifact of the cap and must not be treated as a complete total.
        limited = (
            f" [TRUNCATED at the {s.record_count}-row cap — the real total is UNKNOWN and "
            f"larger; this count is an artifact of the limit, so do NOT infer anything "
            f"from its magnitude, completeness, or the breadth of values it spans]"
            if getattr(s, "row_limited", False)
            else ""
        )
        # Zero rows from an exclusion source is the answer, not a gap.
        meaning = str(getattr(s, "zero_rows_meaning", "") or "").strip()
        answered = (
            f" [ZERO ROWS IS THE ANSWER HERE, not a gap: {meaning}. Do NOT describe this "
            f"source as unavailable, failed or unevaluated, and do NOT call what it "
            f"establishes unknown]"
            if meaning and not s.record_count
            else ""
        )
        lines.append(
            f"- {s.source}: {s.record_count} record(s)"
            + limited
            + answered
            + (" [aggregated]" if s.aggregated else "")
            + (f" kept={','.join(s.kept_columns)}" if s.kept_columns else "")
        )
        # Adjudicated-column values rendered before degradable detail and kept by
        # `_thin_source_lines` regardless of budget.
        if s.adjudicated_values:
            # Whole columns are dropped rather than truncated; overflow is counted.
            parts, room = [], adj_caps.get(s.source, _MIN_ADJ_LINE_LEN)
            for col, vals in s.adjudicated_values.items():
                part = _adj_part(col, vals)
                if parts and len(part) + 2 > room:
                    continue
                parts.append(part)
                room -= len(part) + 2
            shown = "; ".join(parts)
            spilled = len(s.adjudicated_values) - len(parts)
            if spilled:
                shown += (
                    f"; (+{spilled} further column(s) listed in kept= were also read, "
                    "values not restated here)"
                )
            lines.append(
                _ADJ_LINE_PREFIX + "values this run's conditions were decided on "
                "(these columns WERE read — never report them as missing or "
                f"unavailable): {shown}"
            )
        for col, dist in list(s.distributions.items())[:_MAX_DIST_COLS_RENDERED]:
            items = list(dist.items())[:_MAX_DIST_VALUES]
            top = ", ".join(f"{k}={c}" for k, c in items)
            more = (
                ""
                if len(dist) <= _MAX_DIST_VALUES
                else f" (+{len(dist) - _MAX_DIST_VALUES} more)"
            )
            lines.append(f"    {col}: {top}{more}")
        for ex in s.examples[:_MAX_SOURCE_EXAMPLES_RENDERED]:
            lines.append(f"    example: {str(ex)[:_MAX_EXAMPLE_LEN]}")

    source_lines = lines
    notes = list(pack.notes or [])
    full = "\n".join(head_lines + source_lines) + _render_notes(notes)
    if len(full) <= char_budget:
        return full

    # Over budget. Sub-budget both halves: a flat tail clip would silently drop source
    # entries. The head share is a ceiling; unused head budget flows to the sources block.
    reserve = len(_CLIP_SUFFIX) + _CLIP_NOTE_RESERVE
    usable = max(0, char_budget - reserve)
    head_cap = int(usable * _HEAD_BUDGET_SHARE)
    head_text = "\n".join(head_lines)
    head_note = ""
    if len(head_text) > head_cap:
        head_lines, head_note = _clip_lines(
            head_lines, head_cap, "chronology/attribution"
        )
        head_text = "\n".join(head_lines)
    source_cap = max(0, usable - len(head_text))
    source_text = "\n".join(source_lines)
    source_note = ""
    if len(source_text) > source_cap:
        source_lines, source_note = _thin_source_lines(source_lines, source_cap)
        source_text = "\n".join(source_lines)

    # Notes assembled after trimming so they can report what was dropped.
    for n in (head_note, source_note):
        if n and n not in notes:
            notes.append(n)
    text = head_text + "\n" + source_text + _render_notes(notes)
    if len(text) > char_budget:
        keep = max(0, char_budget - len(_CLIP_SUFFIX))
        text = text[:keep] + _CLIP_SUFFIX
    return text
