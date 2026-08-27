import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List

# Flat import, not package-qualified: `src/follow_up.py` reaches `correlation` through
# the same module object this file does. Using `src.correlation` here would create a
# second identity and trigger the dual-import trap described in CLAUDE.md.
from correlation import select_correlation_spec
from src.follow_up import (
    harvest_follow_up,
    harvest_value_tuples,
    repeats_an_earlier_query,
)
from src.human_guidance import guidance_message
from src.models.pydantic_models import ExtractedEntity, RetrievalQuery
from src.utils.error_handling import async_retry_with_backoff

logger = logging.getLogger(__name__)


def _retrieval_query_tool():
    """OpenAI tool definition derived from the RetrievalQuery schema."""
    return {
        "type": "function",
        "function": {
            "name": "create_retrieval_query",
            "description": "Create a query describing which logs to retrieve for this incident.",
            "parameters": RetrievalQuery.model_json_schema(),
        },
    }


class ApiCallGenerator:
    def __init__(self, config, llm_client, knowledge_pack=None, available_sources=None):
        # config is the main_config['log_sources'] section.
        self.config = config
        self.llm_client = llm_client
        self.knowledge_pack = knowledge_pack
        # Sources that built a retriever; when set, only these are offered to the LLM.
        self.available_sources = (
            set(available_sources) if available_sources is not None else None
        )
        # Dependency findings per generate() call. Three lists because the remedy differs:
        # undeliverable (no retriever), not_queried (skippable), unscopable (right skip).
        self.undeliverable_required = []
        self.declared_not_queried = []
        self.declared_unscopable = []
        # Planner-selected sources whose query was dropped or unparseable; kept separate
        # so the removal is not silent.
        self.selected_unscopable = []
        self.selected_unparseable = []
        self.tool = _retrieval_query_tool()

    # Endpoint kinds the retrieval engine can serve; mirrors _KIND_TO_TYPE in log_retrieval.py.
    _SERVEABLE_KINDS = {"elasticsearch", "databricks_uc"}

    def _source_names(self):
        # Explicit main_config sources take precedence.
        sources = self.config.get("sources") or []
        names = [s["name"] for s in sources if "name" in s]

        # Add pack sources the engine can serve.
        seen = set(names)
        for src in getattr(self.knowledge_pack, "sources", []) or []:
            if src.name in seen:
                continue
            if src.kind() in self._SERVEABLE_KINDS:
                names.append(src.name)
                seen.add(src.name)

        names = names or self.config.get("names_list", [])

        # Restrict to sources that actually built a retriever; advertise nothing
        # for which retrieval would fail.
        if self.available_sources is not None:
            filtered = [n for n in names if n in self.available_sources]
            if filtered:
                return filtered
            logger.warning(
                "No advertised sources match the engine's built retrievers %s; "
                "offering the unfiltered list.",
                sorted(self.available_sources),
            )
        return names

    def _source_guidance(self, source_names):
        """Describe sources by capability when a catalog is available."""
        if self.knowledge_pack is not None:
            catalog = self.knowledge_pack.catalog_prompt(source_names)
            if catalog:
                return catalog
        return f"target_log_source must be one of: {', '.join(source_names)}."

    def _adjudicating_ruleset_key(self, analysis=None):
        """The ruleset key that will adjudicate this incident, or ``""``.

        Resolved the same way the verdict stage does (playbook → use_case →
        ``ruleset_key_for``). Falls back to the pack default, then ``""``. Never raises.
        """
        pack = self.knowledge_pack
        if pack is None:
            return ""
        try:
            if analysis is not None:
                spec = select_correlation_spec(pack, analysis)
                key = pack.ruleset_key_for(str((spec or {}).get("use_case", "") or ""))
                if key:
                    return key
            return str(pack.default_ruleset_key() or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not resolve which procedure adjudicates this incident (%s); "
                "planning as though no procedure declared a follow-up pass.",
                exc,
            )
            return ""

    def _required_sources(self, source_names, analysis=None):
        """``(required, undeliverable)`` from the adjudicating ruleset's declared sources.

        Unmet sources are reported, never injected. Follow-up pass targets are excluded
        (the pass declaration defers them). Other rulesets' dependencies are not this run's.
        """
        pack = self.knowledge_pack
        if pack is None or not getattr(pack, "rulesets", None):
            return [], []
        key = self._adjudicating_ruleset_key(analysis)
        spec = ((getattr(pack, "rulesets", None) or {}).get("verdicts") or {}).get(key)
        if not isinstance(spec, dict):
            return [], []
        offered = set(source_names)
        deferred_targets = self._follow_up_targets(analysis)
        required, undeliverable, seen = [], [], set()
        for real in (spec.get("sources") or {}).values():
            name = str(real or "")
            if not name or name in seen:
                continue
            seen.add(name)
            if name not in offered:
                undeliverable.append(name)  # reported even when deferred
            elif name in deferred_targets:
                logger.info(
                    "Source '%s' is a declared follow-up pass's target, so it is not "
                    "expected on this pass: the values its question needs do not exist "
                    "until an earlier pass answers.",
                    name,
                )
            else:
                required.append(name)
        return required, undeliverable

    def _follow_up_targets(self, analysis=None):
        """Sources deferred to a declared follow-up pass; this pass must not query them.

        Only the adjudicating ruleset's declarations count: a target deferred by the
        wrong ruleset is deferred by a pass that will never run.
        """
        key = self._adjudicating_ruleset_key(analysis)
        return self._targets_declared_by([key] if key else [])

    def _targets_declared_by(self, keys):
        """The physical sources the given rulesets' follow-up passes target."""
        pack = self.knowledge_pack
        if pack is None or not hasattr(pack, "follow_up_passes"):
            return set()
        out = set()
        try:
            for key in keys:
                for entry in pack.follow_up_passes(key) or []:
                    for name in entry.get("sources") or [entry.get("source")]:
                        if str(name or ""):
                            out.add(str(name))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not read the declared follow-up passes (%s); every declared source "
                "is planned on this pass, which is the single-pass behaviour.",
                exc,
            )
            return set()
        return out

    def _defer_follow_up_targets(self, queries, analysis=None):
        """Drop queries against sources a declared follow-up pass will retrieve later.

        Two queries against one source are not additive: pass-2 rows merge into pass-1
        rows and a condition counting rows reads both scopes at once.
        """
        deferred = self._follow_up_targets(analysis)
        if not deferred:
            return queries
        keep = [q for q in queries if q.target_log_source not in deferred]
        dropped = [
            q.target_log_source for q in queries if q.target_log_source in deferred
        ]
        if dropped:
            logger.info(
                "Dropped %d planned query/queries against %s: a declared follow-up pass "
                "retrieves that source with the values an earlier pass harvests, and the "
                "planner cannot know that from the catalog alone.",
                len(dropped),
                ", ".join(sorted(set(dropped))),
            )
        return keep

    def _fallback_request(self, name, src, purpose, only_types=None):
        """NL request for a source built from catalog declarations.

        ``only_types`` narrows the entity list; omitting it keeps the full declared list.
        """
        entities = [
            e for e in (getattr(src, "entities", []) or []) if e != "time_window"
        ]
        if only_types is not None:
            wanted = {str(t) for t in only_types}
            entities = [e for e in entities if e in wanted]
        scope = (
            f" Filter on the incident's {', '.join(entities)} where present."
            if entities
            else ""
        )
        avoid = ""
        for item in getattr(src, "not_answered_by", []) or []:
            if isinstance(item, dict) and str(item.get("question") or "").strip():
                avoid = (
                    " Do not ask this source for anything it does not hold; see its "
                    "DOES NOT ANSWER notes."
                )
                break
        return (
            f"Retrieve this source's records for the incident.{scope}{avoid} "
            f"{purpose}"
        ).strip()

    def _scopable(self, name, entities):
        """True when at least one incident entity is a non-window type the source can filter on.

        A source declaring no filterable types keeps all entities (``_entities_for``) and
        counts as scopable, so it is reported as a real gap rather than silently excused.
        """
        if not entities:
            return False
        kept = self._entities_for(name, entities)
        return any(str(getattr(e, "type", "") or "") != "time_window" for e in kept)

    def _drop_unscopable(self, queries, analysis):
        """Remove planner picks that cannot be scoped to this incident's entities.

        Skipped when there are no entities; operator-added queries bypass this entirely.
        A false drop indicates a missing ``entities:`` declaration.
        """
        entities = list(getattr(analysis, "extracted_entities", None) or [])
        self.selected_unscopable = []
        if not entities:
            return list(queries)
        kept = []
        for query in queries:
            name = str(getattr(query, "target_log_source", "") or "")
            if not name or self._scopable(name, entities):
                kept.append(query)
                continue
            self.selected_unscopable.append(name)
            logger.warning(
                "Source '%s' was selected but the incident names no entity type it can "
                "filter on, so the query is DROPPED rather than issued as a window-only "
                "scan (measured on this deployment's history: 12 of 16 such queries "
                "returned 0 rows and 4 returned the row cap). Either the source cannot "
                "answer this incident — in which case its catalog entry should say so in "
                "`selection_guidance` / `not_answered_by` — or it can and its `entities:` "
                "list is missing the type this incident carries. Add the query from the run "
                "controls to ask it anyway.",
                name,
            )
        return kept

    def dependency_report(self, queries, source_names=None, analysis=None):
        """Which declared sources this plan does not meet; computed from arguments only.

        Returns ``{"undeliverable": [...], "not_queried": [...], "unscopable": [...]}``;
        each list has a different remedy.
        """
        names = list(
            source_names if source_names is not None else self._source_names()
        )
        required, undeliverable = self._required_sources(names, analysis)
        entities = list(getattr(analysis, "extracted_entities", None) or [])
        chosen = {getattr(q, "target_log_source", "") for q in (queries or [])}
        not_queried, unscopable = [], []
        for name in required:
            if name in chosen:
                continue
            target = not_queried if self._scopable(name, entities) else unscopable
            target.append(name)
        return {
            "undeliverable": list(undeliverable),
            "not_queried": not_queried,
            "unscopable": unscopable,
        }

    def _report_unmet_dependencies(self, queries, source_names, analysis):
        """Record and log dependency findings for this run; never injects queries."""
        report = self.dependency_report(queries, source_names, analysis)
        undeliverable = report["undeliverable"]
        self.undeliverable_required = list(undeliverable)
        self.declared_not_queried = list(report["not_queried"])
        self.declared_unscopable = list(report["unscopable"])
        if undeliverable:
            logger.error(
                "%d ruleset-declared source(s) CANNOT be retrieved on this run (no "
                "retriever was built — missing backend credentials or an unsupported "
                "endpoint kind): %s. Every condition reading them will be `unknown`, so "
                "the verdict is based on less evidence than the procedure requires.",
                len(undeliverable),
                ", ".join(sorted(undeliverable)),
            )
        for name in self.declared_unscopable:
            logger.info(
                "Declared source '%s' was not selected, and could not have been scoped to "
                "this incident: it filters on none of the entity types the incident named. "
                "The conditions reading it cannot be evaluated on this run.",
                name,
            )
        if self.declared_not_queried:
            logger.warning(
                "%d source(s) the adjudicating procedure declares as hard dependencies were "
                "NOT selected by the planner, and could have been scoped to this incident: "
                "%s. Their conditions will be `unknown`. Fix what the catalog tells the "
                "planner about them (selection_guidance / not_answered_by), or add the query "
                "from the run controls.",
                len(self.declared_not_queried),
                ", ".join(sorted(self.declared_not_queried)),
            )

    def build_manual_query(self, analysis, source, question="", window=None):
        """One retrieval query the operator asked for, scoped exactly as a planned one.

        ``question`` blank falls back to ``_fallback_request``. ``window`` is an optional
        ``(date_from, date_to)``; the incident's ``event_time`` overrides it when present.
        Raises ``ValueError`` when the source cannot be retrieved on this run.
        """
        name = str(source or "").strip()
        if not name:
            raise ValueError("A source name is required")
        offered = self._source_names()
        if name not in offered:
            known = self.knowledge_pack.source(name) if self.knowledge_pack else None
            raise ValueError(
                f"Source '{name}' built no retriever on this run (missing backend "
                "credentials or an unsupported endpoint kind), so a query against it "
                "cannot be retrieved"
                if known is not None
                else f"No source named '{name}' — it is not in the catalog"
            )
        src = self.knowledge_pack.source(name) if self.knowledge_pack else None
        purpose = (getattr(src, "description", "") or "").strip().split("\n")[0]
        date_from, date_to = (window or ("", ""))
        event_time = getattr(analysis, "event_time", None)
        if event_time:
            date_from = date_from or _date_part(event_time.start)
            date_to = date_to or _date_part(event_time.end)
        query = RetrievalQuery(
            target_log_source=name,
            natural_language_query=(
                str(question or "").strip()
                or self._fallback_request(name, src, purpose)
            ),
            date_from=date_from or "",
            date_to=date_to or "",
        )
        self._enrich_queries([query], analysis)
        return query

    def unselected_sources(self, queries, analysis=None):
        """Sources not yet queried, annotated for the operator panel.

        Each entry: ``declared`` (procedure hard dependency), ``deferred`` (follow-up
        pass targets it), ``scopable`` (at least one entity type is filterable),
        ``purpose`` (first catalog description line).
        """
        chosen = {getattr(q, "target_log_source", "") for q in (queries or [])}
        deferred = self._follow_up_targets(analysis)
        required, _ = self._required_sources(self._source_names(), analysis)
        # Recomputed from analysis (not instance state): one instance serves every job.
        declared = set(required) | set(deferred)
        entities = list(getattr(analysis, "extracted_entities", None) or [])
        out = []
        for name in self._source_names():
            if name in chosen:
                continue
            src = self.knowledge_pack.source(name) if self.knowledge_pack else None
            out.append(
                {
                    "source": name,
                    "purpose": (getattr(src, "description", "") or "")
                    .strip()
                    .split("\n")[0],
                    "declared": name in declared,
                    "deferred": name in deferred,
                    "scopable": self._scopable(name, entities),
                }
            )
        # Unmet procedure dependencies first: those are the most actionable entries.
        out.sort(key=lambda e: (not e["declared"], e["deferred"], e["source"]))
        return out

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def generate(self, understanding, guidance=None):
        """Turn an UnderstandingResult into a list of RetrievalQuery objects.

        ``guidance`` is analyst direction from a rejected gate (e.g. "the window was
        wrong, widen it to the whole month" / "you missed the sales source").
        """
        incident_id = understanding.incident_id
        analysis = understanding.analysis
        try:
            source_names = self._source_names()
            event_time = getattr(analysis, "event_time", None)
            if event_time:
                date_hint = (
                    f"The incident occurred between {event_time.start} and "
                    f"{event_time.end}; set date_from/date_to to span that window "
                    "(yyyy-MM-dd)."
                )
            else:
                default_from, default_to = self._default_window(
                    getattr(understanding, "incident_timestamp", "")
                )
                date_hint = (
                    f"The incident states no time, so every query is read over the "
                    f"configured default window {default_from} to {default_to}; set "
                    "date_from/date_to to exactly that (yyyy-MM-dd)."
                    if default_from and default_to
                    else "Set date_from a couple of days before and date_to a couple of days "
                    "after the incident (yyyy-MM-dd)."
                )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You decide which logs to retrieve to investigate a fraud incident. "
                        "Call the create_retrieval_query tool ONCE PER RELEVANT SOURCE "
                        "(use a source only when its contents match what the investigation "
                        "needs).\n"
                        f"{self._source_guidance(source_names)}\n"
                        f"{date_hint} Use '*' for scope_id or actor_id when unknown. "
                        "Write natural_language_query as a precise plain-language "
                        "description of the logs needed from that source."
                    ),
                },
                {
                    "role": "user",
                    "content": analysis.model_dump_json(indent=2),
                },
            ]
            human_msg = guidance_message(guidance)
            if human_msg is not None:
                messages.insert(1, human_msg)

            message = await self.llm_client.tool_call(
                messages,
                tools=[self.tool],
                tool_choice={
                    "type": "function",
                    "function": {"name": "create_retrieval_query"},
                },
                stage="api_call_generation",
            )

            queries = self._parse_tool_calls(message, source_names)
            # Defer before the dependency report so deferred sources aren't reported missing.
            queries = self._defer_follow_up_targets(queries, analysis)
            # Drop before the report: a dropped declared source lands in `unscopable`, not
            # `not_queried`. Both use the same _scopable predicate.
            queries = self._drop_unscopable(queries, analysis)
            self._report_unmet_dependencies(queries, source_names, analysis)
            self._enrich_queries(
                queries, analysis, getattr(understanding, "incident_timestamp", "")
            )
            logger.info(
                f"Generated {len(queries)} retrieval queries for incident {incident_id}"
            )
            return queries
        except Exception as e:
            logger.error(
                f"Error generating retrieval queries for incident {incident_id}: {str(e)}"
            )
            raise

    async def generate_follow_up(
        self,
        understanding,
        spec,
        logs,
        prior_queries=None,
        guidance=None,
        row_caps=None,
    ):
        """Queries one declared follow-up pass adds, plus notes. Returns ``(queries, notes)``.

        No LLM call. Never raises; returns ``([], notes)`` when the pass cannot run. One
        pass may target several sources; a per-target failure never aborts siblings.
        ``row_caps`` maps source to ``max_results``; unknown cap means the pass runs.
        """
        analysis = getattr(understanding, "analysis", None)
        notes: List[str] = []
        try:
            harvested, notes = harvest_follow_up(
                spec, logs, analysis, self.knowledge_pack
            )
            # Built from plain dicts so the class identity is this module's (dual-import trap).
            entities = [ExtractedEntity(**e) for e in harvested]
            tuples, tuple_notes = harvest_value_tuples(
                spec, logs, analysis, self.knowledge_pack
            )
            notes.extend(tuple_notes)
        except Exception as exc:  # noqa: BLE001 — a first pass must survive this
            logger.error(
                "Follow-up pass %s: harvesting failed (%s). The pass is skipped and the "
                "run continues on the evidence the earlier pass(es) returned.",
                (spec or {}).get("pass"),
                exc,
            )
            return [], [
                "the follow-up pass could not be built, so its question was not asked"
            ]
        targets = [
            str(t).strip() for t in ((spec or {}).get("sources") or []) if str(t).strip()
        ]
        if not targets:
            targets = [
                t for t in [str((spec or {}).get("source", "") or "").strip()] if t
            ]
        if not entities:
            logger.warning(
                "Follow-up pass %s targeting '%s' harvested no values, so it is SKIPPED. "
                "Querying it with no scope would be a window-only scan whose rows answer no "
                "condition.",
                (spec or {}).get("pass"),
                ", ".join(targets),
            )
            return [], notes
        queries: List[RetrievalQuery] = []
        for name in targets:
            query, target_notes = self._follow_up_query(
                spec,
                name,
                entities,
                analysis,
                prior_queries,
                guidance,
                tuples,
                logs,
                row_caps,
            )
            notes.extend(target_notes)
            if query is not None:
                queries.append(query)
        return queries, notes

    def _follow_up_query(
        self,
        spec,
        name,
        entities,
        analysis,
        prior_queries=None,
        guidance=None,
        tuples=None,
        logs=None,
        row_caps=None,
    ):
        """One follow-up target's query, or ``(None, notes)`` with a reason.

        ``tuples`` projected onto bindable types; fewer than two types yields no
        combination constraint.
        """
        notes: List[str] = []
        available = self._source_names()
        if available and name not in available:
            logger.error(
                "Follow-up pass %s targets '%s', which built no retriever on this run — the "
                "pass is skipped and every condition reading it stays `unknown`.",
                (spec or {}).get("pass"),
                name,
            )
            return None, [
                f"the follow-up source '{name}' could not be retrieved on this run, so the "
                "question it answers is still open"
            ]
        scoped = self._entities_for(name, entities)
        scoped = [e for e in scoped if e.type != "time_window"]
        if not scoped:
            logger.error(
                "Follow-up pass %s: none of the harvested entity types (%s) is filterable on "
                "'%s', so the pass is skipped rather than run as a window-only scan.",
                (spec or {}).get("pass"),
                ", ".join(sorted({e.type for e in entities})),
                name,
            )
            return None, [
                f"the harvested values cannot be used as filters on '{name}', so the "
                "follow-up question was not asked"
            ]
        date_from, date_to = self._follow_up_window(spec, analysis, prior_queries)
        # Checked after window: a widened lookback makes an otherwise identical query different.
        repeat = repeats_an_earlier_query(
            name,
            scoped,
            prior_queries,
            logs,
            row_caps,
            (date_from, date_to),
            guidance,
        )
        if repeat:
            logger.info(
                "Follow-up pass %s on '%s' is SKIPPED: %s",
                (spec or {}).get("pass"),
                name,
                repeat,
            )
            return None, [repeat]
        src = (
            self.knowledge_pack.source(name)
            if self.knowledge_pack is not None
            else None
        )
        purpose = str(((spec or {}).get("purposes") or {}).get(name, "") or "").strip()
        if not purpose:
            purpose = str((spec or {}).get("purpose", "") or "").strip()
        if not purpose:
            purpose = (getattr(src, "description", "") or "").strip().split("\n")[0]
        by_type: Dict[str, List[str]] = {}
        for ent in scoped:
            by_type.setdefault(ent.type, []).append(ent.value)
        scope_text = "; ".join(
            f"{etype} is one of ({', '.join(values)})"
            for etype, values in sorted(by_type.items())
        )
        projected = self._project_tuples(tuples, set(by_type))
        scope_rule = (
            f"Restrict the query to these values, carried forward from the earlier "
            f"retrieval: {scope_text}. Each of these is a separate condition a returned row "
            "must satisfy (combine them with AND); the values listed within one are "
            "alternatives (an IN list)."
        )
        if projected:
            scope_rule = (
                "Restrict the query to these COMBINATIONS of values, carried forward from "
                "the earlier retrieval — each is one row of evidence and the parts of it "
                "occurred together: "
                + "; ".join(
                    "(" + " AND ".join(f"{p['type']}={p['value']}" for p in tup) + ")"
                    for tup in projected
                )
                + ". A returned row must match ALL the parts of ONE combination: OR the "
                "combinations together and AND the parts within each. Do NOT filter each "
                "part independently against a list of its own values — that asks for every "
                "pairing of every value, and the pairings that did not occur belong to other "
                "parties whose rows would come back as though they were this one's."
            )
        request = (
            f"{self._fallback_request(name, src, purpose, only_types=by_type)} "
            f"{scope_rule} Use NO other filter than these and the date range: "
            "any identifier from the incident that is not listed above belongs to a "
            "different party than the one this question is about, and filtering on it "
            "returns nothing."
        ).strip()
        human = str(guidance or "").strip()
        if human:
            request = f"{request} Analyst direction: {human}"
        query = RetrievalQuery(
            target_log_source=name,
            natural_language_query=request,
            date_from=date_from,
            date_to=date_to,
            entities=scoped,
        )
        # Full entity list for the fabricated-predicate guard; bypasses _enrich_queries.
        try:
            query._incident_entities = list(
                getattr(analysis, "extracted_entities", []) or []
            )
        except (AttributeError, ValueError):  # noqa: BLE001 — a mock, or not our model
            pass
        if projected:
            try:
                query._value_tuples = [list(t) for t in projected]
            except (AttributeError, ValueError):  # noqa: BLE001 — a mock, or not our model
                pass
        logger.info(
            "Follow-up pass %s: 1 query for '%s' scoped to %s over %s..%s.",
            (spec or {}).get("pass"),
            name,
            scope_text,
            date_from,
            date_to,
        )
        return query, notes

    @staticmethod
    def _project_tuples(tuples, bindable_types):
        """Harvested combinations projected onto the types this target can filter on.

        Projection to fewer than two types yields nothing (a one-type set is the per-type
        IN list the caller already sends).
        """
        projected: List[List[dict]] = []
        seen: set = set()
        types: set = set()
        for tup in tuples or []:
            parts = [
                p
                for p in (tup or [])
                if str((p or {}).get("type", "")) in (bindable_types or set())
                and str((p or {}).get("value", "")).strip()
            ]
            if len(parts) < 2:
                continue
            key = tuple(sorted((str(p["type"]), str(p["value"])) for p in parts))
            if key in seen:
                continue
            seen.add(key)
            types.update(str(p["type"]) for p in parts)
            projected.append(parts)
        if len(types) < 2:
            return []
        return projected

    def _follow_up_window(self, spec, analysis, prior_queries):
        """Date window for a follow-up pass as ``(from, to)``.

        Modes: ``inherit`` (reuse earlier window), ``onwards`` (event start → today),
        ``lookback:<N>d`` (N days before event start). Unparseable depth degrades to
        ``inherit``.
        """
        prior = list(prior_queries or [])
        date_from = next((q.date_from for q in prior if q.date_from), "")
        date_to = next((q.date_to for q in prior if q.date_to), "")
        event_time = getattr(analysis, "event_time", None)
        if event_time:
            date_from = _date_part(event_time.start) or date_from
        mode = str((spec or {}).get("window", "") or "inherit").strip().lower()
        if mode == "onwards":
            today = datetime.now(timezone.utc).date().isoformat()
            if today > (date_to or ""):
                date_to = today
            return date_from, date_to
        if event_time:
            date_to = _date_part(event_time.end) or date_to
        if mode.startswith("lookback"):
            date_from = self._lookback_from(mode, date_from, spec)
        return date_from, date_to

    @staticmethod
    def _lookback_from(mode, date_from, spec):
        """``date_from`` moved back by ``lookback:<N>d``, or left unchanged on any failure."""
        match = re.match(r"^lookback\s*[:=]?\s*(\d+)\s*d?$", mode)
        if not match:
            logger.error(
                "Follow-up pass %s declares window '%s', which carries no usable depth — "
                "expected 'lookback:<N>d'. Falling back to the inherited window, so the "
                "history question is asked over the incident's own window only.",
                (spec or {}).get("pass"),
                mode,
            )
            return date_from
        days = int(match.group(1))
        if not date_from:
            logger.warning(
                "Follow-up pass %s declares window '%s' but the run has no lower date bound "
                "to move back from, so the window is left unbounded as it already was.",
                (spec or {}).get("pass"),
                mode,
            )
            return date_from
        try:
            start = date.fromisoformat(_date_part(date_from))
        except ValueError:
            logger.error(
                "Follow-up pass %s declares window '%s' but '%s' is not an ISO date, so the "
                "inherited window stands.",
                (spec or {}).get("pass"),
                mode,
                date_from,
            )
            return date_from
        widened = (start - timedelta(days=days)).isoformat()
        logger.info(
            "Follow-up pass %s: window widened %s days back, %s -> %s, to ask an "
            "antecedent question rather than an episode one.",
            (spec or {}).get("pass"),
            days,
            date_from,
            widened,
        )
        return widened

    def referral_window(self, analysis, prior_queries, window_hint=""):
        """``(from, to, resolved)`` for a cross-procedure referral's own window.

        Thin seam onto :meth:`_follow_up_window` so referrals share the same window arithmetic.
        A bare ``lookback`` hint (direction, no depth) degrades to ``inherit`` quietly;
        ``resolved`` names the mode that was actually applied.
        """
        mode = str(window_hint or "").strip().lower() or "inherit"
        resolved = "inherit" if mode == "lookback" else mode
        spec = {"window": resolved, "pass": "link referral"}
        date_from, date_to = self._follow_up_window(spec, analysis, prior_queries)
        return date_from, date_to, resolved

    def _default_window(self, incident_timestamp=""):
        """``(date_from, date_to)`` for an incident that states no time, or ``("", "")``.

        Resolved pack then config; the engine holds no depth of its own. When neither declares
        one, returns nothing and warns, because the alternative reading of an absent declaration
        is a silent constant. Anchored on the ingestion timestamp so a re-run reads the same
        window.
        """
        days = None
        if self.knowledge_pack is not None:
            try:
                days = self.knowledge_pack.default_lookup_days()
            except Exception as e:  # noqa: BLE001 — a stub pack, or a mock
                logger.warning("Could not read the pack's default lookup depth: %s", e)
        if not days:
            days = self.config.get("default_lookup_days")
        try:
            days = int(days or 0)
        except (TypeError, ValueError):
            days = 0
        if days <= 0:
            logger.warning(
                "This incident states no time window and no default lookup depth is declared "
                "(pack retrieval.default_lookup_days, or log_sources.default_lookup_days in the "
                "config), so each query keeps whatever window was generated for it. Declare one: "
                "an invented window is the hard scope of every retrieval."
            )
            return "", ""
        anchor = _date_part(incident_timestamp) or ""
        try:
            end = date.fromisoformat(anchor) if anchor else datetime.now(timezone.utc).date()
        except ValueError:
            end = datetime.now(timezone.utc).date()
        return (end - timedelta(days=days)).isoformat(), end.isoformat()

    def _enrich_queries(self, queries, analysis, incident_timestamp=""):
        """Attach the incident's entities and window to every generated query.

        Entity attachment is a union: the analysis's list is authoritative and the model may
        contribute extras but may not remove entries. Date fields are always overwritten:
        ``event_time`` when stated, else the declared default depth.

        Deduplicated on (type, value, value_form): two forms of one type are two distinct
        entities and collapsing them would silently drop a binding.
        """
        entities = list(getattr(analysis, "extracted_entities", []) or [])
        event_time = getattr(analysis, "event_time", None)
        # Declared depth resolved once per plan so all queries use the same window.
        default_from, default_to = (
            ("", "") if event_time else self._default_window(incident_timestamp)
        )
        # The incident's own per-record combinations, resolved once per plan so all queries
        # use tuples rather than per-type IN lists where applicable.
        incident_tuples = self._incident_value_tuples(entities)
        for query in queries:
            # Set before per-source narrowing: `value_tuple_columns` drops what the source
            # cannot bind rather than dropping the combination.
            if incident_tuples:
                try:
                    query._value_tuples = [list(t) for t in incident_tuples]
                except (AttributeError, ValueError):  # noqa: BLE001 — a mock, or not our model
                    pass
            # Full unfiltered entity list for the fabricated-predicate guard, which needs
            # types the source cannot bind. Assigned before the union.
            try:
                query._incident_entities = list(entities)
            except (AttributeError, ValueError):  # noqa: BLE001 — a mock, or not our model
                pass
            if entities:
                generated = list(query.entities or [])
                if generated:
                    # Apply the allow-list to generated entities for the same reason as the
                    # incident's: overstating scope misleads the reviewer. Guarded because
                    # `_entities_for([])` would warn about a wide scan the union is about to prevent.
                    generated = self._entities_for(query.target_log_source, generated)
                query.entities = self._union_entities(
                    self._entities_for(query.target_log_source, entities),
                    generated,
                    query.target_log_source,
                )
            # Event time is authoritative for the date window when present.
            if event_time:
                query.date_from = _date_part(event_time.start)
                query.date_to = _date_part(event_time.end)
            elif default_from and default_to:
                # Where the incident states no time, the declared default depth is authoritative.
                query.date_from = default_from
                query.date_to = default_to

    @staticmethod
    def _incident_value_tuples(entities):
        """The incident's own per-record combinations, in ``_value_tuples`` shape.

        Reads the ``co_occurrence`` group stamped by
        ``incident_understanding._group_co_occurring_entities`` and emits the same
        ``[[{type, value, value_form}, ...], ...]`` shape ``follow_up.harvest_value_tuples``
        produces. A combination needs two distinct types; a group collapsed to one type is
        already covered by the per-type filter. An entity may carry several labels (a table
        repeats a column); the ``co_occurrence`` field is a space-joined set.

        Returns ``[]`` for incidents with no repeated per-record structure.
        """
        grouped = {}
        for entity in entities:
            for group in str(getattr(entity, "co_occurrence", "") or "").split():
                grouped.setdefault(group, []).append(entity)
        out = []
        for group in sorted(grouped):
            members = grouped[group]
            if len({str(getattr(e, "type", "") or "") for e in members}) < 2:
                continue
            out.append(
                [
                    {
                        "type": str(getattr(e, "type", "") or ""),
                        "value": str(getattr(e, "value", "") or ""),
                        "value_form": str(getattr(e, "value_form", "") or ""),
                    }
                    for e in members
                ]
            )
        if out:
            logger.info(
                "The incident states %d per-record entity combination(s); every query will "
                "be narrowed to those rather than to the cross product of the per-type "
                "value lists.",
                len(out),
            )
        return out

    @staticmethod
    def _entity_key(entity):
        return (
            str(getattr(entity, "type", "") or "").strip().lower(),
            str(getattr(entity, "value", "") or "").strip().lower(),
            str(getattr(entity, "value_form", "") or "").strip().lower(),
        )

    def _union_entities(self, authoritative, generated, source_name="?"):
        """The analysis's entities for this source, plus anything extra the LLM added."""
        merged = list(authoritative)
        seen = {self._entity_key(e) for e in merged}
        added = []
        for entity in generated:
            key = self._entity_key(entity)
            if key in seen:
                continue
            seen.add(key)
            merged.append(entity)
            added.append(f"{key[0]}={key[1]}")
        if added:
            logger.info(
                "Source '%s': keeping %d generated entity value(s) the incident analysis "
                "did not carry (%s).",
                source_name,
                len(added),
                ", ".join(added),
            )
        return merged

    def _entities_for(self, source_name, entities):
        """The incident's entities this source can actually filter on.

        Filtered to the source's declared ``entities`` allow-list, which matches what
        ``field_mapping.map_entities`` uses. An entity naming a column the source lacks
        overstates the query's scope for the reviewer and the gate. ``time_window`` is kept
        as a date bound. Unknown source or a source declaring nothing keeps everything.
        """
        pack = self.knowledge_pack
        src = pack.source(source_name) if pack is not None and source_name else None
        allowed = set(getattr(src, "entities", []) or []) if src is not None else set()
        if not allowed:
            return list(entities)
        kept = [e for e in entities if e.type in allowed or e.type == "time_window"]
        dropped = sorted({e.type for e in entities if e not in kept})
        if dropped:
            logger.info(
                "Source '%s': not attaching entities it cannot filter on (%s); they are "
                "not columns on this source, so listing them overstates the query's scope.",
                source_name,
                ", ".join(dropped),
            )
        if not kept:
            # Worth saying: this source can be bounded only by its date window, so it will
            # be a wide scan. Padding the list with entities it cannot filter on would hide
            # that behind a scope the query does not actually have.
            logger.warning(
                "Source '%s': none of the incident's entities are filterable here — the "
                "query will be bounded by its date window alone.",
                source_name,
            )
        return kept

    def _parse_tool_calls(self, message, source_names):
        self.selected_unparseable = []
        queries = []
        for call in getattr(message, "tool_calls", None) or []:
            raw = getattr(getattr(call, "function", None), "arguments", None)
            try:
                query = RetrievalQuery.model_validate_json(raw)
            except Exception as e:
                query = self._salvage_tool_call(raw, source_names, e)
                if query is None:
                    continue
            if source_names and query.target_log_source not in source_names:
                logger.warning(
                    f"LLM chose unknown log source '{query.target_log_source}'; "
                    f"defaulting to '{source_names[0]}'"
                )
                query.target_log_source = source_names[0]
            queries.append(query)
        return queries

    #: Fields the engine overwrites unconditionally; a call omitting them is incomplete,
    #: not malformed.
    _ENGINE_OWNED_FIELDS = ("date_from", "date_to")

    def _salvage_tool_call(self, raw, source_names, error):
        """A ``RetrievalQuery`` completed from the engine's own fields, or ``None``.

        A selected source whose call fails is indistinguishable from a source nobody chose;
        both leave no query. The call is split into engine-owned fields (``date_from``/``date_to``,
        overwritten by ``_enrich_queries`` anyway) and the analytic intent
        (``natural_language_query``, never invented). A call with a target and no question lands
        on ``selected_unparseable``; a fabricated question turns a scoped lookup into a
        window-wide scan.

        Only a target this run can retrieve is recorded; a hallucinated name is noise.
        """
        try:
            data = json.loads(raw) if isinstance(raw, (str, bytes)) else None
        except Exception:  # noqa: BLE001 — an unparseable argument string is the same drop
            data = None
        if not isinstance(data, dict):
            logger.warning(
                "Skipping malformed retrieval query — its arguments are not an object, so "
                "not even the source it targeted can be reported: %s",
                error,
            )
            return None
        target = data.get("target_log_source")
        target = target.strip() if isinstance(target, str) else ""
        question = data.get("natural_language_query")
        question = question.strip() if isinstance(question, str) else ""
        if not target:
            logger.warning(
                "Skipping malformed retrieval query — it names no target source, so there is "
                "nothing to salvage and nothing to report: %s",
                error,
            )
            return None
        if not question:
            self._record_unparseable(target, source_names, "it carries no question")
            return None
        for field in self._ENGINE_OWNED_FIELDS:
            if not isinstance(data.get(field), str):
                data[field] = ""
        try:
            query = RetrievalQuery.model_validate(data)
        except Exception:  # noqa: BLE001 — one more try, then it is reported
            # `entities` is the only nested shape left; the incident's own are attached by
            # `_enrich_queries` regardless, so dropping them here costs nothing.
            data.pop("entities", None)
            try:
                query = RetrievalQuery.model_validate(data)
            except Exception as e2:  # noqa: BLE001
                self._record_unparseable(target, source_names, f"it did not validate ({e2})")
                return None
        logger.info(
            "Retrieval query for '%s' was completed rather than dropped: the call omitted %s, "
            "which the engine is authoritative for anyway (the event window overwrites both).",
            target,
            " / ".join(self._ENGINE_OWNED_FIELDS),
        )
        return query

    def _record_unparseable(self, target, source_names, reason):
        """Record a planner-chosen source whose call could not become a query."""
        if source_names and target not in source_names:
            logger.warning(
                "Skipping malformed retrieval query targeting '%s' — %s, and it is not a "
                "source this run can retrieve, so it is not reported as a gap either.",
                target,
                reason,
            )
            return
        self.selected_unparseable.append(target)
        logger.warning(
            "Source '%s' WAS selected by the planner and no query could be built from the "
            "call — %s. The question is the one thing the engine cannot supply, so this is "
            "reported rather than repaired: without this it reads exactly like a source the "
            "planner never chose, and the conditions reading it go `unknown` while the stage "
            "reports success. Add the query from the run controls to ask it.",
            target,
            reason,
        )


def _date_part(value: str) -> str:
    """Reduce an ISO date/datetime string to its yyyy-MM-dd prefix."""
    if not value:
        return value
    return value.split("T")[0].strip()
