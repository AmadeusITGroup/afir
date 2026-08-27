"""Shared entity->field mapping for retrievers.

``map_entities`` asks the LLM to bind each incident entity to a field chosen only from
the source's real, discovered fields. Pack aliases are a prior; the discovered schema is
authoritative. Entities with no plausible field are dropped.

Three invariants shape every exit path:

- A source's ``entity_bindings`` is a declaration, not a prior: exempt from the confidence
  gate, adopted when the mapper returns nothing, and overriding a confident wrong pick.
  All three paths confirm against the discovered schema first.
- A failed LLM call says nothing about which column was measured, so declared bindings are
  still adopted from the schema.
- An absent schema adopts nothing. An adopted-but-missing field is a predicate matching
  zero rows, which ``key_was_enforced`` licenses as a decisive negative; an empty field
  map leaves the outcome unknown.
"""

import logging
import re
from typing import Dict, List, Optional

from src.models.pydantic_models import FieldMapping, RetrievalQuery
from src.retrievers.query_guards import (conjunction_fields,
                                         resolve_identity_fields,
                                         subject_anchor_values)
from src.utils.projection import expand_projection

logger = logging.getLogger(__name__)

#: Time-window entity type. Travels as ``query.date_from``/``date_to`` and applies as a
#: range bound, never an equality filter. Spelled the same in ``correlation`` and
#: ``api_call_generator``.
_TIME_ENTITY = "time_window"


async def map_entities(
    llm_client,
    query: RetrievalQuery,
    field_schema: str,
    knowledge_pack=None,
    min_confidence: float = 0.5,
) -> Dict[str, str]:
    """Return a verified ``{entity_type: field_name}`` map for one source.

    ``field_schema`` is the rendered, discovered schema of the source (e.g.
    ``table(col type, ...)`` or ``field: type``). Returns an empty dict when there
    are no entities or no schema to map against (callers then fall back to plain
    NL->query generation).
    """
    entities = list(query.entities or [])
    if not entities:
        return {}

    if knowledge_pack is not None and query.target_log_source:
        src = knowledge_pack.source(query.target_log_source)
        allowed = set(getattr(src, "entities", []) or []) if src is not None else set()
        if allowed:
            dropped = [e.type for e in entities if e.type not in allowed]
            entities = [e for e in entities if e.type in allowed]
            if dropped:
                logger.info(
                    "Source %s: not filtering on non-catalog entities %s",
                    query.target_log_source,
                    sorted(set(dropped)),
                )
        if not entities:
            return {}

    declared = _declared_fields(knowledge_pack, entities, query.target_log_source)

    if not field_schema:
        if declared:
            logger.warning(
                "Source %s: no schema was discovered, so nothing can be confirmed and the "
                "field map is EMPTY — the pack's declared bindings (%s) are NOT applied. "
                "Every guard that reads the field map is inert for this query, filters come "
                "only from the source's prose hints, and an empty result cannot be read as "
                "decisive. Re-check the source's schema discovery.",
                query.target_log_source,
                ", ".join(f"{t}->{f[0]}" for t, f in declared.items() if f),
            )
        return {}

    entity_types = [e.type for e in entities]
    alias_hint = ""
    if knowledge_pack is not None:
        alias_hint = knowledge_pack.alias_hints(
            entity_types, source_name=query.target_log_source
        )
        form_hint = _form_alias_hints(knowledge_pack, entities, query.target_log_source)
        if form_hint:
            alias_hint = form_hint

    entity_lines = "\n".join(f"- {e.type}: {e.value}" for e in entities)
    messages = [
        {
            "role": "system",
            "content": (
                "You map incident entities to the real fields of a data source. "
                "For each entity, choose the SINGLE best-matching field NAME taken "
                "ONLY from the provided schema. If no field plausibly holds that "
                "entity, omit it. Never invent a field name. Set a confidence in "
                "[0,1] reflecting how sure the match is."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source schema (only these fields exist):\n{field_schema}\n\n"
                f"Entities to map:\n{entity_lines}\n"
                + (
                    f"\nCandidate field-name hints: {alias_hint}\n"
                    if alias_hint
                    else ""
                )
            ),
        },
    ]

    try:
        result: FieldMapping = await llm_client.structured_output(
            messages, response_model=FieldMapping, stage="log_retrieval"
        )
    except Exception as e:
        logger.warning(
            "Entity field-mapping failed; proceeding on the pack's declared bindings "
            "alone: %s",
            e,
        )
        return _adopt_declared_bindings(
            {}, declared, _schema_field_names(field_schema), query.target_log_source
        )

    schema_names = _schema_field_names(field_schema)

    field_map: Dict[str, str] = {}
    for m in result.mappings:
        if m.confidence >= min_confidence:
            preferred = _declared_override(
                m.entity_type, m.field, declared, schema_names
            )
            if preferred:
                logger.warning(
                    "Source %s: mapper chose %s->%s at confidence %.2f, but the pack "
                    "declares %s for this source and it is in the discovered schema — "
                    "filtering on %s. A declaration is a measurement of this target; a "
                    "confidence is the model's opinion of one.",
                    query.target_log_source,
                    m.entity_type,
                    m.field,
                    m.confidence,
                    declared.get(m.entity_type, []),
                    preferred,
                )
                field_map[m.entity_type] = preferred
                continue
            field_map[m.entity_type] = m.field
            continue
        confirmed = _declared_spelling(
            m.field, declared.get(m.entity_type, []), schema_names
        )
        if confirmed:
            field_map[m.entity_type] = confirmed
            logger.info(
                "Source %s: keeping mapping %s->%s despite low confidence (%.2f) — the "
                "pack declares this binding and the field is in the discovered schema.",
                query.target_log_source,
                m.entity_type,
                confirmed,
                m.confidence,
            )
            continue
        logger.info(
            "Dropping low-confidence mapping %s->%s (%.2f)",
            m.entity_type,
            m.field,
            m.confidence,
        )
    field_map = _adopt_declared_bindings(
        field_map, declared, schema_names, query.target_log_source
    )
    if field_map:
        logger.info("Entity field map for %s: %s", query.target_log_source, field_map)
    return field_map


def _adopt_declared_bindings(
    field_map: Dict[str, str],
    declared: Dict[str, List[str]],
    schema_names: Dict[str, str],
    source_name: str,
) -> Dict[str, str]:
    """Fill entities the mapper left unbound from the source's own declarations (schema-confirmed)."""
    for entity_type, fields in (declared or {}).items():
        if entity_type in field_map:
            continue
        present = [
            spelled
            for spelled in (_in_schema(f, schema_names) for f in fields)
            if spelled
        ]
        if present:
            field_map[entity_type] = present[0]
            logger.info(
                "Source %s: mapping %s->%s from the pack's declared binding (the mapper "
                "offered nothing usable for it).",
                source_name,
                entity_type,
                present[0],
            )
        elif fields and schema_names:
            logger.warning(
                "Source %s: pack declares %s -> %s but none of those fields is in the "
                "discovered schema — STALE BINDING, not filtering on it. Re-measure the "
                "source's entity_bindings.",
                source_name,
                entity_type,
                fields,
            )
    return field_map


def _declared_fields(knowledge_pack, entities, source_name) -> Dict[str, List[str]]:
    """``{entity_type: [field, ...]}`` from this source's own ``entity_bindings`` (form-scoped)."""
    if knowledge_pack is None or not source_name:
        return {}
    try:
        src = knowledge_pack.source(source_name)
    except Exception:  # a pack stub / mock without the accessor
        return {}
    if src is None:
        return {}
    bindings = getattr(src, "entity_bindings", None) or {}
    out: Dict[str, List[str]] = {}
    for ent in entities:
        declared = bindings.get(ent.type)
        if declared is None:
            continue
        if isinstance(declared, dict):
            form = getattr(ent, "value_form", "") or ""
            entry = declared.get(form) if form else None
        else:
            entry = declared
        if isinstance(entry, str):
            fields = [entry]
        elif isinstance(entry, list):
            fields = [str(f) for f in entry]
        else:
            fields = []
        for field in fields:
            bucket = out.setdefault(ent.type, [])
            if field not in bucket:
                bucket.append(field)
    return out


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.$-]*")


def _schema_field_names(field_schema: str) -> Dict[str, str]:
    """``{lowercased: as-spelled}`` for field names in a rendered schema; takes first identifier per fragment."""
    names: Dict[str, str] = {}
    for frag in re.split(r"[,;()\n]", field_schema or ""):
        frag = frag.strip().replace("`", "").replace('"', "").lstrip("- ")
        if not frag:
            continue
        m = _IDENT_RE.match(frag)
        if m:
            names.setdefault(m.group(0).lower(), m.group(0))
    return names


def _schema_field_types(field_schema: str) -> Dict[str, str]:
    """``{lowercased field: type}`` for a rendered schema; missing entry is undecidable, not a default."""
    types: Dict[str, str] = {}
    for frag in re.split(r"[,;()\n]", field_schema or ""):
        frag = frag.strip().replace("`", "").replace('"', "").lstrip("- ")
        if not frag:
            continue
        m = _IDENT_RE.match(frag)
        if not m:
            continue
        rest = frag[m.end() :].strip().lstrip(":").strip()
        if rest:
            types.setdefault(m.group(0).lower(), rest)
    return types


#: Column types a two-sided instant bound can be rendered for. String columns are excluded:
#: an ISO-8601 string compares correctly as a prefix, but a string in any other format
#: returns zero rows, and the two are indistinguishable from the schema.
_BOUNDABLE_TIME_TYPES = ("DATE", "TIMESTAMP", "DATETIME")


def event_time_column(
    field_schema: str,
    bindings: Dict[str, object],
    partitions: Optional[List[Dict]] = None,
    epoch_columns: Optional[List[Dict]] = None,
    source_name: str = "?",
) -> Optional[tuple]:
    """``(column, type)`` for the finer event-time bound beside the partition column, or ``None``.

    Shared by all four retrievers. Declines if partitions use asymmetric pads, if the column
    is absent from the partition set, if it is epoch-integer, or if its type is non-temporal.
    """
    time_fields = bindings.get(_TIME_ENTITY) if bindings else None
    if isinstance(time_fields, str):
        time_fields = [time_fields]
    if not time_fields:
        return None
    specs = list(partitions or [])
    if not specs:
        return None
    for spec in specs:
        if str((spec or {}).get("role") or "date").strip().lower() != "date":
            return None
        if (spec or {}).get("pad_days_after") is not None:
            return None
    excluded = {
        str((spec or {}).get("name") or "").lower() for spec in specs
    } | {str((c or {}).get("name") or "").lower() for c in (epoch_columns or [])}
    types = _schema_field_types(field_schema)
    for field in time_fields:
        field = str(field or "").strip()
        if not field or field.lower() in excluded:
            continue
        col_type = types.get(field.lower())
        if not col_type:
            logger.info(
                "Source '%s': event-time column '%s' is bound to the window but its type "
                "could not be established from the discovered schema, so no bound was "
                "injected — a bound in the wrong literal form returns zero rows, which is "
                "worse than an unbounded scan.",
                source_name,
                field,
            )
            continue
        if not col_type.strip().upper().startswith(_BOUNDABLE_TIME_TYPES):
            logger.info(
                "Source '%s': event-time column '%s' has type '%s', which no literal form "
                "here can be rendered for safely — left unbounded.",
                source_name,
                field,
                col_type,
            )
            continue
        return field, col_type
    return None


def _in_schema(field: str, schema_names: Dict[str, str]) -> Optional[str]:
    """The schema's own spelling of ``field``, or ``None``; case-insensitive fallback returns schema spelling."""
    if not field or not schema_names:
        return None
    if field in schema_names.values():
        return field
    return schema_names.get(field.lower())


def _declared_spelling(
    field: str, declared: List[str], schema_names: Dict[str, str]
) -> Optional[str]:
    """``field`` as the schema spells it, but only if the pack declared it here."""
    if not declared:
        return None
    lowered = {d.lower() for d in declared}
    if field.lower() not in lowered:
        return None
    return _in_schema(field, schema_names)


def _declared_override(
    entity_type: str,
    chosen: str,
    declared: Dict[str, List[str]],
    schema_names: Dict[str, str],
) -> Optional[str]:
    """Declared field that overrides a confident mapper pick; ``None`` when the pick already matches."""
    fields = declared.get(entity_type) or []
    if not fields:
        return None
    if chosen and chosen.lower() in {f.lower() for f in fields}:
        return None
    for field in fields:
        spelled = _in_schema(field, schema_names)
        if spelled:
            return spelled
    return None


def _form_alias_hints(knowledge_pack, entities, source_name) -> str:
    """Per-form alias hint string for entities bound per value form; empty when not form-bound."""
    if knowledge_pack is None:
        return ""
    parts, seen = [], set()
    any_form_binding = False
    for ent in entities:
        try:
            forms = knowledge_pack.form_bindings_for(ent.type, source_name)
        except Exception:  # a pack stub / mock without the accessor
            return ""
        if not forms:
            key = (ent.type, None)
            if key not in seen:
                seen.add(key)
                fields = knowledge_pack.field_priors_for(ent.type, source_name)
                if fields:
                    parts.append(f"{ent.type}: {', '.join(fields)}")
            continue
        any_form_binding = True
        form = getattr(ent, "value_form", "") or ""
        key = (ent.type, form)
        if key in seen:
            continue
        seen.add(key)
        fields = knowledge_pack.field_priors_for(ent.type, source_name, form or None)
        if fields:
            label = f"{ent.type} ({form})" if form else ent.type
            parts.append(f"{label}: {', '.join(fields)}")
    return "; ".join(parts) if any_form_binding else ""


def _form_scoped_fields(knowledge_pack, source_name, entity_type, value_form):
    """Fields this source binds for one (entity, form), or ``None`` when not form-bound."""
    if knowledge_pack is None or not value_form:
        return None
    try:
        forms = knowledge_pack.form_bindings_for(entity_type, source_name)
    except Exception:
        return None
    if not forms:
        return None
    return list(forms.get(value_form, []))


def filter_values_by_field(
    field_map: Dict[str, str], query: RetrievalQuery, knowledge_pack=None
) -> Dict[str, list]:
    """``{real column: [incident value, ...]}`` for one source, form-routed; flattening of :func:`values_by_type_and_field`."""
    out: Dict[str, list] = {}
    for by_field in values_by_type_and_field(
        field_map, query, knowledge_pack
    ).values():
        for field, values in by_field.items():
            out.setdefault(field, [])
            for value in values:
                if value not in out[field]:
                    out[field].append(value)
    return out


def values_by_type_and_field(
    field_map: Dict[str, str], query: RetrievalQuery, knowledge_pack=None
) -> Dict[str, Dict[str, list]]:
    """``{entity type: {real column: [value, ...]}}``; per-type view used by anchor + key-presence guards."""
    if not field_map:
        return {}
    values_by_type: Dict[str, list] = {}
    form_of: Dict[tuple, str] = {}
    for e in query.entities or []:
        if e.value and e.value != "*" and e.type != _TIME_ENTITY:
            values_by_type.setdefault(e.type, [])
            if e.value not in values_by_type[e.type]:
                values_by_type[e.type].append(e.value)
            form = getattr(e, "value_form", "") or ""
            if form:
                form_of[(e.type, e.value)] = form
    out: Dict[str, Dict[str, list]] = {}
    for entity_type, field in field_map.items():
        for value in values_by_type.get(entity_type, []):
            target = _routed_field(
                field,
                knowledge_pack,
                query.target_log_source,
                entity_type,
                value,
                form_of.get((entity_type, value)),
            )
            if target is None:
                continue
            per_type = out.setdefault(entity_type, {})
            per_type.setdefault(target, [])
            if value not in per_type[target]:
                per_type[target].append(value)
            for alt in _stem_alternatives(knowledge_pack, entity_type, value):
                if alt not in per_type[target]:
                    per_type[target].append(alt)
                    logger.info(
                        "Source %s: offering %s %r on %s as its declared stem %r too — the "
                        "target may store the identity core rather than the whole value, and "
                        "a predicate on only the long form is a 0-row match on such a column.",
                        query.target_log_source,
                        entity_type,
                        value,
                        target,
                        alt,
                    )
    return out


def _routed_field(
    field: str,
    knowledge_pack,
    source_name: str,
    entity_type: str,
    value: str,
    value_form: str,
) -> Optional[str]:
    """The column one value belongs on for this source (form-routing); ``None`` to drop it."""
    allowed = _form_scoped_fields(knowledge_pack, source_name, entity_type, value_form)
    if allowed is None:
        return field
    if not allowed:
        logger.info(
            "Source %s: dropping %s value %r — this source binds no field "
            "for its value form %r, so filtering on it would be a "
            "guaranteed 0-row equality on another form's column.",
            source_name,
            entity_type,
            value,
            value_form,
        )
        return None
    if field in allowed:
        return field
    logger.info(
        "Source %s: routing %s value %r (form %s) to %s instead of %s "
        "per the pack's per-form binding.",
        source_name,
        entity_type,
        value,
        value_form,
        allowed[0],
        field,
    )
    return allowed[0]


def value_tuple_columns(
    field_map: Dict[str, str], query: RetrievalQuery, knowledge_pack=None
) -> List[List[tuple]]:
    """Harvested value combinations resolved onto this source's columns.

    ``[[(column, [literal, ...]), ...], ...]``; empty unless a prior pass harvested tuples.
    Components whose type is unbound are dropped; combinations under two are not enforced.
    """
    tuples = getattr(query, "_value_tuples", None) or []
    if not field_map or not tuples:
        return []
    out: List[List[tuple]] = []
    for tup in tuples:
        parts: List[tuple] = []
        for part in tup or []:
            entity_type = str((part or {}).get("type", "") or "")
            value = str((part or {}).get("value", "") or "")
            if not entity_type or not value or entity_type == _TIME_ENTITY:
                continue
            field = field_map.get(entity_type)
            if not field:
                continue
            column = _routed_field(
                field,
                knowledge_pack,
                query.target_log_source,
                entity_type,
                value,
                str((part or {}).get("value_form", "") or ""),
            )
            if column is None:
                continue
            literals = [value]
            for alt in _stem_alternatives(knowledge_pack, entity_type, value):
                if alt not in literals:
                    literals.append(alt)
            parts.append((column, literals))
        if len({c for c, _ in parts}) >= 2:
            out.append(parts)
    return out


def _stem_alternatives(knowledge_pack, entity_type: str, value: str) -> list:
    """Declared stem of one value as ``[stem]``, or ``[]`` when none declared."""
    if knowledge_pack is None:
        return []
    try:
        stem = knowledge_pack.value_stem(entity_type, value)
    except Exception:  # a pack stub / mock without the accessor
        return []
    return [stem] if stem and stem != value else []


def stem_literals(
    field_map: Dict[str, str], query: RetrievalQuery, knowledge_pack=None
) -> Dict[str, Dict[str, str]]:
    """``{real column: {long value: stem}}`` for ``widen_stem_literals``; resolved from filter_values_by_field."""
    if not field_map or knowledge_pack is None:
        return {}
    types_by_value: Dict[str, set] = {}
    for e in query.entities or []:
        if e.value and e.value != "*" and e.type != _TIME_ENTITY:
            types_by_value.setdefault(str(e.value), set()).add(e.type)
    out: Dict[str, Dict[str, str]] = {}
    for column, values in filter_values_by_field(field_map, query, knowledge_pack).items():
        pairs: Dict[str, str] = {}
        for value in values:
            for entity_type in sorted(types_by_value.get(value, ())):
                for stem in _stem_alternatives(knowledge_pack, entity_type, value):
                    if stem in values:
                        pairs[value] = stem
        if pairs:
            out[column] = pairs
    return out


def match_patterns(
    field_map: Dict[str, str], query: RetrievalQuery, knowledge_pack=None
) -> Dict[str, Dict[str, str]]:
    """``{real column: {value: wildcard pattern}}`` for ``widen_match_patterns``; resolved from filter_values_by_field."""
    if not field_map or knowledge_pack is None:
        return {}
    types_by_value: Dict[str, set] = {}
    for e in query.entities or []:
        if e.value and e.value != "*" and e.type != _TIME_ENTITY:
            types_by_value.setdefault(str(e.value), set()).add(e.type)
    out: Dict[str, Dict[str, str]] = {}
    for column, values in filter_values_by_field(field_map, query, knowledge_pack).items():
        pairs: Dict[str, str] = {}
        for value in values:
            for entity_type in sorted(types_by_value.get(value, ())):
                try:
                    pattern = knowledge_pack.value_match_pattern(entity_type, value)
                except Exception:  # a pack stub / mock without the accessor
                    pattern = None
                if pattern and pattern != value:
                    pairs[value] = pattern
        if pairs:
            out[column] = pairs
    return out


def key_presence_values(
    field_map: Dict[str, str],
    query: RetrievalQuery,
    key_fields: Optional[List[str]] = None,
    knowledge_pack=None,
    source_name: str = "?",
) -> Dict[str, Dict[str, list]]:
    """``{entity type: {column: [value, ...]}}`` for the resolved key's members only.

    Shared by all four retrievers. Two key members on the same column are both dropped
    (their ANDed groups would match no row).
    """
    fields = {f for f in (key_fields or []) if f}
    if not fields or not field_map:
        return {}
    key_types = [t for t, f in (field_map or {}).items() if f in fields]
    if not key_types:
        return {}
    per_type = values_by_type_and_field(field_map, query, knowledge_pack)
    out = {t: per_type[t] for t in key_types if per_type.get(t)}
    shared = {
        column
        for column in {c for cols in out.values() for c in cols}
        if sum(1 for cols in out.values() if column in cols) > 1
    }
    if shared:
        collapsed = [t for t, cols in out.items() if shared & set(cols)]
        logger.warning(
            "Source %s: key members %s all route onto %s, so their presence is NOT enforced "
            "here — AND-ing one column against two members' values matches no row. A source "
            "storing several key members in one column is the same-column conjunction's case.",
            source_name,
            " + ".join(collapsed),
            ", ".join(sorted(shared)),
        )
        out = {t: cols for t, cols in out.items() if not (shared & set(cols))}
    return out


def subject_anchor(
    field_map: Dict[str, str],
    query: RetrievalQuery,
    identity_scopes=None,
    identity_synonyms=None,
    knowledge_pack=None,
    source_name: str = "?",
) -> tuple:
    """``(identity_values, scope_values)`` for the subject anchor; shared by all four retrievers."""
    if not field_map or not (identity_scopes or identity_synonyms):
        return {}, {}
    return subject_anchor_values(
        filter_values_by_field(field_map, query, knowledge_pack),
        resolve_identity_fields(identity_scopes, field_map, source_name),
        resolve_identity_fields(identity_synonyms, field_map, source_name),
        source_name,
    )


def render_filters(
    field_map: Dict[str, str],
    query: RetrievalQuery,
    require_all_entities: Optional[List[str]] = None,
    knowledge_pack=None,
    identity_keys: Optional[List[List[str]]] = None,
) -> str:
    """Human-readable filter hints for query generation; form-routed, empty when nothing maps."""
    if not field_map:
        return ""
    values_by_field = filter_values_by_field(field_map, query, knowledge_pack)
    values_by_type = {
        e.type: True
        for e in (query.entities or [])
        if e.value and e.value != "*" and e.type != _TIME_ENTITY
    }
    patterns = match_patterns(field_map, query, knowledge_pack)
    parts = []
    for field, values in values_by_field.items():
        if len(values) == 1:
            parts.append(f"{field} = {values[0]!r}")
        else:
            rendered = ", ".join(repr(v) for v in values)
            parts.append(f"{field} IN ({rendered})  (match ANY of these)")
        for value, pattern in (patterns.get(field) or {}).items():
            parts.append(
                f"{field}: {value!r} is only a PART of the value this column stores — "
                f"match it as the pattern {pattern!r} (? = exactly one character, "
                "* = any run) beside the equality, never as an equality alone"
            )
    time_field = field_map.get(_TIME_ENTITY)
    if time_field and (query.date_from or query.date_to):
        parts.append(
            f"{time_field} is the event-time column for the window "
            f"{query.date_from or '?'}..{query.date_to or '?'} — bound it as a RANGE, "
            "never an equality against a single instant"
        )
    hints = "; ".join(parts)

    required_fields = [
        f
        for f in conjunction_fields(
            field_map,
            require_all_entities=require_all_entities,
            identity_keys=identity_keys,
            present_types=list(values_by_type),
            source_name=query.target_log_source,
        )
        if f in values_by_field
    ]
    if len(required_fields) > 1:
        hints += (
            f". MANDATORY — this source is keyed by the COMBINATION of "
            f"{', '.join(required_fields)}: you MUST join these with AND, never OR. "
            "Neither is selective on its own, so an OR returns rows for OTHER keys "
            "(and truncates at the row limit) instead of answering about this one. "
            "This overrides any general guidance above about OR-ing weak entities"
        )
    return hints


def source_bindings(knowledge_pack, source_name) -> Dict[str, object]:
    """One source's ``entity_bindings`` verbatim (unflattened, unfiltered), or ``{}``."""
    if knowledge_pack is None or not source_name:
        return {}
    try:
        src = knowledge_pack.source(source_name)
    except Exception:  # a pack stub / mock without the accessor
        return {}
    if src is None:
        return {}
    return dict(getattr(src, "entity_bindings", None) or {})


def declared_leaf_paths(knowledge_pack, source_name) -> List[str]:
    """Every field path this source's own config names (entity_bindings + projection), in declaration order; existence unchecked."""
    if knowledge_pack is None or not source_name:
        return []
    try:
        src = knowledge_pack.source(source_name)
    except Exception:  # a pack stub / mock without the accessor
        return []
    if src is None:
        return []
    paths: List[str] = []

    def _add(value) -> None:
        if isinstance(value, str):
            if value and value not in paths:
                paths.append(value)
        elif isinstance(value, dict):
            for inner in value.values():
                _add(inner)
        elif isinstance(value, (list, tuple)):
            for inner in value:
                _add(inner)

    _add(getattr(src, "entity_bindings", None) or {})
    _add(expand_projection(list(getattr(src, "projection", None) or [])))
    return paths


def form_split_bindings(knowledge_pack, source_name) -> List[tuple]:
    """``[(entity type, [(form, [field, ...])])]`` for types with value forms on two or more distinct columns.

    Input to ``relax_form_conjunction``; types with one form or all forms on one column are excluded.
    """
    if knowledge_pack is None or not source_name:
        return []
    try:
        src = knowledge_pack.source(source_name)
    except Exception:  # a pack stub / mock without the accessor
        return []
    out: List[tuple] = []
    for etype, binding in (getattr(src, "entity_bindings", None) or {}).items():
        if not isinstance(binding, dict):
            continue
        forms = knowledge_pack.form_bindings_for(etype, source_name) or {}
        forms = {name: fields for name, fields in forms.items() if fields}
        if len(forms) < 2:
            continue
        columns = {str(f).strip().lower() for fields in forms.values() for f in fields}
        if len(columns) < 2:
            continue
        out.append((str(etype), [(name, list(fields)) for name, fields in forms.items()]))
    return out


def incident_values(query: RetrievalQuery) -> Dict[str, List[str]]:
    """``{entity type: [value, ...]}`` from query entities (excluding time_window), for the fabricated-predicate guard.

    Reads ``query.entities`` + ``_incident_entities`` if present; scalars excluded (unverified types).
    """
    out: Dict[str, List[str]] = {}
    whole = list(getattr(query, "_incident_entities", None) or [])
    for e in whole + list(query.entities or []):
        etype, value = str(getattr(e, "type", "") or ""), str(getattr(e, "value", "") or "")
        if not etype or not value or value == "*" or etype == _TIME_ENTITY:
            continue
        bucket = out.setdefault(etype, [])
        if value not in bucket:
            bucket.append(value)
    return out


def render_identifiers(query: RetrievalQuery) -> str:
    """Identifier values for the query-generation prompt, untyped (fallback when field_map is empty).

    Values come from query.entities + the two scalars; time_window excluded.
    Returns '' when the query carries no identifier values.
    """
    values: List[str] = []
    for value in (
        query.scope_id,
        query.actor_id,
        *(
            e.value
            for e in (query.entities or [])
            if getattr(e, "type", "") != "time_window"
        ),
    ):
        if value and value != "*" and value not in values:
            values.append(value)
    if not values:
        return ""
    rendered = ", ".join(repr(v) for v in values)
    return (
        f"Identifier value(s) from the alert: {rendered}. These are UNTYPED — the "
        "alert's own labels for them are unreliable (two identifiers for the same actor "
        "are routinely both labelled 'user' while binding to DIFFERENT columns). Infer "
        "which column each belongs in from the schema and the entity filters above, "
        "and prefer those mapped filters over this line wherever they cover the same "
        "value.\n"
    )
