"""Harvest the values a follow-up retrieval pass carries forward.

``KnowledgePack.follow_up_passes`` declares what to harvest. Values are stamped with
``value_form`` from pack regexes. Co-occurring values are carried together via
:func:`harvest_value_tuples`; per-type lists are projections of those tuples.
"""

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from correlation import apply_where, resolve_containers, resolve_path

logger = logging.getLogger(__name__)

#: Max chars per harvested value; rejects free-text leaves that would match nothing as a filter.
_MAX_VALUE_CHARS = 128

#: Max distinct values per entity type per pass; backends refuse large IN lists.
_MAX_VALUES_PER_ENTITY = 50

#: Max combinations per pass (OR of ANDs).
_MAX_VALUE_TUPLES = 50

#: Fallback pattern for an uncompilable ``capture``: rejects every value so the pass skips
#: loudly rather than carrying raw leaves the pack declared unusable.
_NEVER_MATCHES = re.compile(r"(?!)")


def harvest_follow_up(
    spec: Dict[str, Any],
    logs: Dict[str, List[Dict[str, Any]]],
    analysis: Any = None,
    knowledge_pack: Any = None,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """The entities one declared follow-up pass carries forward, plus drop notes.

    Returns plain ``{type, value, raw, value_form}`` dicts (not ``ExtractedEntity``
    instances: the dual-import trap makes ``isinstance`` unusable across the import
    boundary). Values already on the incident are dropped unless the pass target has not
    yet answered — the declaration deferred it, so this pass is its only retrieval.
    """
    notes: List[str] = []
    entities: List[Dict[str, str]] = []
    if not isinstance(spec, dict):
        return entities, notes
    targets = [str(t).strip() for t in (spec.get("sources") or []) if str(t).strip()]
    if not targets:
        targets = [t for t in [str(spec.get("source", "") or "").strip()] if t]
    target = ", ".join(targets)
    answered = bool(
        isinstance(logs, dict) and targets and all(logs.get(t) for t in targets)
    )
    known = _known_values(analysis)
    carried_known: Dict[str, int] = {}
    pattern = _compile_capture(str(spec.get("capture", "") or ""), notes)
    for item in spec.get("harvest") or []:
        if not isinstance(item, dict):
            continue
        # Co-occurrence items: per-type lists are projections of accepted tuples.
        components = _tuple_components(item)
        if components is not None:
            tuples, _ = _item_tuples(item, logs, components, pattern, known, answered)
            for comp in components:
                etype = comp["entity"]
                values: List[str] = []
                for tup in tuples:
                    for part in tup:
                        if part["type"] == etype and part["value"] not in values:
                            values.append(part["value"])
                if not values:
                    notes.append(
                        f"no new {etype} value was found in "
                        f"{str(item.get('source', '') or '') or 'the retrieved rows'} "
                        f"({', '.join(comp['fields'])}) as part of a complete combination "
                        f"— {_supply_phrase(logs, str(item.get('source', '') or ''), item.get('where'))}, "
                        "so the follow-up query carries none"
                    )
                    continue
                if len(values) > _MAX_VALUES_PER_ENTITY:
                    values = values[:_MAX_VALUES_PER_ENTITY]
                for value in values:
                    entities.append(
                        {
                            "type": etype,
                            "value": value,
                            "raw": (
                                f"{str(item.get('source', '') or '') or 'retrieved rows'}:"
                                f"{comp['fields'][0]}"
                            ),
                            "value_form": "",
                        }
                    )
            continue
        etype = str(item.get("entity", "") or "").strip()
        fields = [str(f).strip() for f in (item.get("fields") or []) if str(f).strip()]
        if not etype or not fields:
            continue
        source = str(item.get("source", "") or "").strip()
        where = item.get("where") or []
        raw = _read_values(logs, source, fields, where)
        kept: List[str] = []
        rejected = 0
        for value in raw:
            shaped = _apply_capture(value, pattern)
            if shaped is None:
                rejected += 1
                continue
            if shaped.lower() in known and answered:
                continue
            if shaped not in kept:
                kept.append(shaped)
                # Per distinct value, not per row that carried it: counting rows reports a
                # large overlap where there is one value on many rows.
                if shaped.lower() in known:
                    carried_known[etype] = carried_known.get(etype, 0) + 1
        if rejected:
            notes.append(
                f"{rejected} harvested {etype} value(s) did not match the declared "
                "capture pattern and were not carried into the follow-up query"
            )
        if carried_known.get(etype):
            notes.append(
                f"{carried_known[etype]} harvested {etype} value(s) are also on the "
                f"incident, and were carried anyway because '{target or 'the target'}' has "
                "returned nothing yet: this pass is that source's only retrieval, so the "
                "value is new to IT even though the question is not new to the run"
            )
        if not kept:
            scoped = (
                " (restricted to the rows the pass declares, which is the scope of the "
                "question and not a filter)"
                if where
                else ""
            )
            notes.append(
                f"no new {etype} value was found in {source or 'the retrieved rows'} "
                f"({', '.join(fields)}){scoped} — {_supply_phrase(logs, source, where)}, "
                "so the follow-up query carries none"
            )
            continue
        if len(kept) > _MAX_VALUES_PER_ENTITY:
            notes.append(
                f"{len(kept)} distinct {etype} value(s) were harvested and the follow-up "
                f"query is bounded to the first {_MAX_VALUES_PER_ENTITY} — the pass covers "
                "less than the evidence found"
            )
            kept = kept[:_MAX_VALUES_PER_ENTITY]
        for value in kept:
            entities.append(
                {
                    "type": etype,
                    "value": value,
                    "raw": f"{source or 'retrieved rows'}:{fields[0]}",
                    "value_form": "",
                }
            )
        logger.info(
            "Follow-up pass %s: harvested %d distinct %s value(s) from '%s' (%s).",
            spec.get("pass"),
            len(kept),
            etype,
            source or "every retrieved source",
            ", ".join(fields),
        )
    _stamp_value_forms(entities, knowledge_pack)
    return entities, notes


def harvest_value_tuples(
    spec: Dict[str, Any],
    logs: Dict[str, List[Dict[str, Any]]],
    analysis: Any = None,
    knowledge_pack: Any = None,
) -> Tuple[List[List[Dict[str, str]]], List[str]]:
    """Co-occurring combinations one declared follow-up pass carries forward.

    Separate from :func:`harvest_follow_up`; both read the same rows so the per-type
    lists cannot describe different evidence. Returns ``(tuples, notes)``.
    """
    notes: List[str] = []
    out: List[List[Dict[str, str]]] = []
    if not isinstance(spec, dict):
        return out, notes
    targets = [str(t).strip() for t in (spec.get("sources") or []) if str(t).strip()]
    if not targets:
        targets = [t for t in [str(spec.get("source", "") or "").strip()] if t]
    answered = bool(
        isinstance(logs, dict) and targets and all(logs.get(t) for t in targets)
    )
    known = _known_values(analysis)
    pattern = _compile_capture(str(spec.get("capture", "") or ""), [])
    for item in spec.get("harvest") or []:
        if not isinstance(item, dict):
            continue
        components = _tuple_components(item)
        if components is None or len(components) < 2:
            continue  # one component is not a combination; pack_validate tells the author
        tuples, item_notes = _item_tuples(
            item, logs, components, pattern, known, answered
        )
        notes.extend(item_notes)
        for tup in tuples:
            if tup not in out:
                out.append(tup)
    if len(out) > _MAX_VALUE_TUPLES:
        notes.append(
            f"{len(out)} distinct combination(s) were harvested and the follow-up query is "
            f"bounded to the first {_MAX_VALUE_TUPLES} — the pass covers less than the "
            "evidence found"
        )
        out = out[:_MAX_VALUE_TUPLES]
    for tup in out:
        _stamp_value_forms(tup, knowledge_pack)
    return out, notes


def repeats_an_earlier_query(
    source: str,
    scoped: Iterable[Any],
    prior_queries: Optional[Iterable[Any]] = None,
    logs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    row_caps: Optional[Dict[str, Any]] = None,
    window: Tuple[str, str] = ("", ""),
    guidance: str = "",
) -> str:
    """Why this follow-up query adds nothing to what is already held, or ``''``.

    Returns a skip reason when provably redundant. Failure is asymmetric: skipping a
    needed pass yields unknown decisive conditions, so any unknown means the pass runs.
    Five conditions must all hold: no analyst guidance; one earlier query is a superset
    per type (one query, not a union); type sets equal; source answered and not truncated;
    this window is inside the earlier one.
    """
    if str(guidance or "").strip():
        return ""
    types = _scope_by_type(scoped)
    if not types:
        return ""
    rows = (logs or {}).get(source)
    if not rows:
        return ""
    cap = None
    try:
        cap = int((row_caps or {}).get(source) or 0) or None
    except (TypeError, ValueError):
        cap = None
    if cap is None or len(rows) >= cap:
        return ""  # unknown cap or truncated: narrower query may reach more rows
    for query in prior_queries or []:
        if str(getattr(query, "target_log_source", "") or "").strip() != source:
            continue
        prior = _scope_by_type(getattr(query, "entities", None) or [])
        if set(prior) != set(types):
            continue
        if any(not types[etype] <= prior[etype] for etype in types):
            continue
        if not _window_within(window, query):
            continue
        return (
            f"the follow-up pass on '{source}' carries no value the earlier retrieval of it "
            f"did not already ask about ({_scope_text(types)}), over the same window, and "
            f"that query returned {len(rows)} row(s) without reaching its {cap}-row cap — so "
            "the pass was NOT run, because its rows are already in this investigation"
        )
    return ""


def _scope_by_type(entities: Iterable[Any]) -> Dict[str, set]:
    """One query's scope as ``{entity type: {lower-cased values}}``, excluding ``time_window``.

    Reads via both ``getattr`` and dict access, so it works on plain dicts and on
    ``ExtractedEntity`` reached through either import path.
    """
    out: Dict[str, set] = {}
    for ent in entities or []:
        if isinstance(ent, dict):
            etype = str(ent.get("type", "") or "").strip()
            value = str(ent.get("value", "") or "").strip()
        else:
            etype = str(getattr(ent, "type", "") or "").strip()
            value = str(getattr(ent, "value", "") or "").strip()
        if not etype or not value or etype == "time_window":
            continue
        out.setdefault(etype, set()).add(value.lower())
    return out


def _window_within(window: Tuple[str, str], query: Any) -> bool:
    """True when ``window`` is inside the window ``query`` already covered; unknown is False."""
    new_from, new_to = (str(window[0] or "").strip(), str(window[1] or "").strip())
    old_from = str(getattr(query, "date_from", "") or "").strip()
    old_to = str(getattr(query, "date_to", "") or "").strip()
    parts = (new_from, new_to, old_from, old_to)
    if not all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", p) for p in parts):
        return False
    return old_from <= new_from and new_to <= old_to


def _scope_text(types: Dict[str, set]) -> str:
    return "; ".join(
        f"{etype}: {len(values)} value(s)" for etype, values in sorted(types.items())
    )


def _tuple_components(item: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """The ``together:`` components of a harvest item, or ``None`` if absent.

    A per-component ``source`` is ignored: values from two sources did not co-occur.
    ``pack_validate`` refuses it at authoring time.
    """
    raw = item.get("together")
    if not isinstance(raw, list) or not raw:
        return None
    out: List[Dict[str, Any]] = []
    for comp in raw:
        if not isinstance(comp, dict):
            continue
        etype = str(comp.get("entity", "") or "").strip()
        fields = [str(f).strip() for f in (comp.get("fields") or []) if str(f).strip()]
        if not etype or not fields:
            continue
        if str(comp.get("source", "") or "").strip():
            logger.warning(
                "Follow-up co-occurrence component '%s' declares its own source, which is "
                "ignored: values from two sources did not occur together, and pairing them "
                "would invent the combinations this mechanism exists to prevent.",
                etype,
            )
        out.append(
            {
                "entity": etype,
                "fields": fields,
                "capture": str(comp.get("capture", "") or ""),
            }
        )
    return out or None


def _cooccurrence_container(fields: Iterable[str]) -> str:
    """Longest shared dotted prefix of ``fields`` that is a strict ancestor of all of them.

    ``''`` means the row is the container (fields on unrelated branches).
    """
    segments = [str(f).split(".") for f in fields if str(f).strip()]
    if not segments:
        return ""
    prefix: List[str] = []
    for parts in zip(*segments):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    while prefix and any(len(s) <= len(prefix) for s in segments):
        prefix.pop()
    return ".".join(prefix)


def _item_tuples(
    item: Dict[str, Any],
    logs: Dict[str, List[Dict[str, Any]]],
    components: List[Dict[str, Any]],
    default_pattern: Optional[re.Pattern],
    known: set,
    answered: bool,
) -> Tuple[List[List[Dict[str, str]]], List[str]]:
    """Every combination one co-occurrence item's rows carry, and the notes about them."""
    notes: List[str] = []
    source = str(item.get("source", "") or "").strip()
    where = item.get("where") or []
    container = _cooccurrence_container(
        [f for comp in components for f in comp["fields"]]
    )
    for comp in components:
        comp["_relative"] = [
            f[len(container) + 1 :] if container and f.startswith(container + ".") else f
            for f in comp["fields"]
        ]
        comp["_pattern"] = (
            _compile_capture(comp["capture"], notes)
            if comp["capture"]
            else default_pattern
        )
    out: List[List[Dict[str, str]]] = []
    containers = 0
    incomplete = 0
    ambiguous = 0
    already_asked = 0
    for name, rows in _selected_rows(logs, source, where, notes):
        for row in rows:
            for node in resolve_containers(row, container):
                containers += 1
                per_component: List[List[str]] = []
                for comp in components:
                    values: List[str] = []
                    for path in comp["_relative"]:
                        try:
                            leaves = resolve_path(node, path)
                        except Exception as exc:  # noqa: BLE001 — one path, not the harvest
                            logger.warning(
                                "Could not resolve follow-up co-occurrence path '%s': %s",
                                path,
                                exc,
                            )
                            continue
                        for leaf in leaves:
                            shaped = _apply_capture(leaf, comp["_pattern"])
                            if shaped is not None and shaped not in values:
                                values.append(shaped)
                    per_component.append(values)
                if any(not values for values in per_component):
                    incomplete += 1  # partial: would pair this container's value with another's
                    continue
                combos = _product(per_component)
                if len(combos) > 1:
                    ambiguous += 1
                for combo in combos:
                    if answered and all(v.lower() in known for v in combo):
                        # Dedup at tuple level: the whole combination must be new.
                        already_asked += 1
                        continue
                    tup = [
                        {
                            "type": comp["entity"],
                            "value": value,
                            "raw": f"{name or 'retrieved rows'}:{comp['fields'][0]}",
                            "value_form": "",
                        }
                        for comp, value in zip(components, combo)
                    ]
                    if tup not in out:
                        out.append(tup)
    arity = len(components)
    types = " + ".join(comp["entity"] for comp in components)
    if incomplete:
        notes.append(
            f"{incomplete} of {containers} record(s) in {source or 'the retrieved rows'} "
            f"named only some of ({types}), so no combination was carried from them — a "
            "partial combination would pair one record's value with another's"
        )
    if ambiguous:
        notes.append(
            f"{ambiguous} of {containers} record(s) carried more than one value for a "
            f"component of ({types}), so the combinations taken from them are every pairing "
            "WITHIN that record; the declared fields share no tighter enclosing path"
        )
    if already_asked:
        notes.append(
            f"{already_asked} harvested combination(s) of ({types}) are already on the "
            "incident and were not re-asked"
        )
    logger.info(
        "Follow-up co-occurrence harvest on '%s': %d distinct %d-part combination(s) of "
        "(%s) from %d record(s) at container '%s' (%d incomplete, %d ambiguous).",
        source or "every retrieved source",
        len(out),
        arity,
        types,
        containers,
        container or "<row>",
        incomplete,
        ambiguous,
    )
    return out, notes


def _product(per_component: List[List[str]]) -> List[Tuple[str, ...]]:
    """Cartesian product within one container (sound by definition; never across containers)."""
    combos: List[Tuple[str, ...]] = [()]
    for values in per_component:
        combos = [combo + (value,) for combo in combos for value in values]
    return combos


def _known_values(analysis: Any) -> set:
    """Lower-cased values the incident already carries, so a pass never re-asks pass 1's."""
    out = set()
    for ent in getattr(analysis, "extracted_entities", None) or []:
        value = str(getattr(ent, "value", "") or "").strip()
        if value:
            out.add(value.lower())
    return out


def _compile_capture(capture: str, notes: List[str]) -> Optional[re.Pattern]:
    """The declared ``capture`` regex, or ``None`` (keep the value whole).

    An uncompilable pattern becomes ``_NEVER_MATCHES``: ignoring it would carry the values
    the pack declared unusable.
    """
    if not capture:
        return None
    try:
        return re.compile(capture)
    except re.error as exc:
        notes.append(
            "the follow-up pass declares a capture pattern that does not compile "
            f"({exc}), so no harvested value could be used"
        )
        logger.error(
            "Follow-up capture pattern %r does not compile: %s. Harvesting nothing rather "
            "than carrying values the pack declared unusable.",
            capture,
            exc,
        )
        return _NEVER_MATCHES


def _apply_capture(value: Any, pattern: Optional[re.Pattern]) -> Optional[str]:
    """One harvested leaf as a filter literal, or ``None`` when it cannot be one."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text or len(text) > _MAX_VALUE_CHARS:
        return None
    if pattern is None:
        return text
    match = pattern.search(text)
    if match is None:
        return None
    try:
        captured = match.group(1) if match.groups() else match.group(0)
    except IndexError:  # pragma: no cover — groups() already told us it exists
        captured = match.group(0)
    captured = str(captured or "").strip()
    return captured or None


def _read_values(
    logs: Dict[str, List[Dict[str, Any]]],
    source: str,
    fields: Iterable[str],
    where: Any = None,
) -> List[Any]:
    """Every leaf the declared fields resolve to, across the named source's rows.

    An empty ``source`` reads every retrieved source. ``where`` is not an optimisation:
    harvesting from unscoped rows asks the follow-up about out-of-scope values too.
    """
    out: List[Any] = []
    for _name, rows in _selected_rows(logs, source, where):
        for row in rows:
            for path in fields:
                try:
                    out.extend(resolve_path(row, path))
                except Exception as exc:  # noqa: BLE001 — one path, not the harvest
                    logger.warning(
                        "Could not resolve follow-up harvest path '%s': %s", path, exc
                    )
    return out


def _selected_rows(
    logs: Dict[str, List[Dict[str, Any]]],
    source: str,
    where: Any = None,
    notes: Optional[List[str]] = None,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """The rows one harvest item reads; shared by ``_read_values`` and ``_item_tuples``.

    Both must read from the same rows: per-type lists are projections of accepted tuples.
    """
    if not isinstance(logs, dict):
        return []
    if source:
        targets = [(source, logs.get(source) or [])]
        if source not in logs:
            logger.warning(
                "Follow-up harvest names source '%s', which is not in the retrieved logs "
                "(%s). Nothing to harvest from it.",
                source,
                ", ".join(sorted(logs)) or "no sources",
            )
    else:
        targets = list(logs.items())
    out: List[Tuple[str, List[Dict[str, Any]]]] = []
    for name, rows in targets:
        selected = [r for r in (rows or []) if isinstance(r, dict)]
        if where:
            try:
                narrowed = apply_where(selected, where)
            except Exception as exc:  # noqa: BLE001 — a clause, not the harvest
                logger.error(
                    "Follow-up harvest row selector on '%s' could not be applied (%s); "
                    "harvesting nothing from it rather than reading rows the pass did not "
                    "ask about.",
                    name,
                    exc,
                )
                if notes is not None:
                    notes.append(
                        f"the rows of '{name}' the follow-up pass declares could not be "
                        "selected, so no combination was carried from it"
                    )
                continue
            logger.info(
                "Follow-up harvest on '%s': %d of %d row(s) match the declared selector.",
                name,
                len(narrowed),
                len(selected),
            )
            selected = narrowed
        out.append((name, selected))
    return out


def _supply_phrase(
    logs: Dict[str, List[Dict[str, Any]]], source: str, where: Any = None
) -> str:
    """Why a harvest had nothing to give; one of: source absent, source empty, no value at path."""
    if not isinstance(logs, dict):
        return "no source has answered yet"
    if source and source not in logs:
        return (
            f"'{source}' is not among the retrieved sources ({', '.join(sorted(logs)) or 'none'}"
            "): it was not queried, or it did not answer, so nothing could be harvested from it"
        )
    if source:
        total = len(logs.get(source) or [])
        label = f"'{source}'"
    else:
        total = sum(len(rows or []) for rows in logs.values())
        label = "the retrieved rows"
    if not total:
        return f"{label} answered with 0 row(s), so no row could carry one"
    selected = sum(len(rows) for _name, rows in _selected_rows(logs, source, where))
    if where and not selected:
        return (
            f"{label} answered with {total} row(s) and none of them are the rows the pass "
            "declares, so the question's own scope is what came back empty"
        )
    return (
        f"{label} answered with {total} row(s)"
        + (f" ({selected} in the pass's declared scope)" if where else "")
        + ", and none carried a value at those field(s)"
    )


def _stamp_value_forms(entities: List[Dict[str, str]], knowledge_pack: Any) -> None:
    """Stamp ``value_form`` from pack regexes, as the understanding stage does.

    Without this a harvested value of a form-carrying type takes the flat (sibling form's)
    column — the cross-form filter ``value_forms`` exists to prevent.
    """
    if knowledge_pack is None:
        return
    for ent in entities:
        etype, value = ent["type"], ent["value"]
        try:
            form = knowledge_pack.classify_value_form(etype, value)
            declared = bool(knowledge_pack.value_forms_for(etype))
        except Exception as exc:  # noqa: BLE001 — classification is never fatal
            logger.warning("Value-form classification failed for %s: %s", etype, exc)
            continue
        ent["value_form"] = form or ""
        if not form and declared:
            logger.warning(
                "Harvested %s value %r matches none of the declared value_forms; it "
                "cannot be form-scoped to a field.",
                etype,
                value,
            )
