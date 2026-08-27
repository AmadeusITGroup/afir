"""UseCaseAnalyzer: the single, generic, deterministic case-builder.

Turns raw retrieved rows, the pack verdict, and KB concepts/cases into a compact
:class:`InvestigationBrief`. Pure and deterministic (no LLM, no IO); reuses the
correlation module's row-shape helpers so it works across every backend row shape.
``analyze`` never raises; on any error it returns a degraded brief.

One class, no per-use-case subclasses. Everything that varies between use cases is
declared in the pack ruleset (``verdicts.<key>.case_builder``), so adding a use case
is a knowledge-authoring change, not a Python one.
"""

import logging
from datetime import date
from typing import Any, Dict, List, Optional

from correlation import (_clause_values, _collect_any,
                         _entity_values, _finding_headline, _label_row,
                         _match_row, _matches_any_identifier, _maybe_json,
                         _norm_identifier, _parse_ts, _resolve_nodes,
                         _values_match, evaluate_verdict, resolve_path)
from models.pydantic_models import (AlertFacts, AssetTimelineEntry,
                                    CaseAssessment, CasePrecedent, ConceptRef,
                                    ConditionCheck, DeclaredFact,
                                    ImpactedAsset, InvestigationBrief,
                                    SubjectLink, ValidationVerdict)

logger = logging.getLogger(__name__)

#: Written by ``build_scope_discovery`` when the sweep returns at its cap; read by
#: ``_scope_step`` to distinguish "saw everything" from "stopped looking". A mismatch
#: between writer and reader would silently claim a bounded scope.
_SCOPE_TRUNCATED = "TRUNCATED"

#: Characters of each concept doc that reach the brief. A prohibition placed past this
#: cut is prose no stage reads. Overridable via ``case_builder.concept_snippet_chars``.
DEFAULT_CONCEPT_SNIPPET_CHARS = 220


def _concept_snippet_chars(case_builder: Dict[str, Any]) -> int:
    """The pack's declared concept-snippet length, else the default.

    Absent, unparseable or non-positive all fall back, per the rule that every
    `case_builder:` key must no-op when the ruleset does not write it.
    """
    try:
        n = int(case_builder.get("concept_snippet_chars") or 0)
    except (TypeError, ValueError):
        return DEFAULT_CONCEPT_SNIPPET_CHARS
    return n if n > 0 else DEFAULT_CONCEPT_SNIPPET_CHARS


class UseCaseAnalyzer:
    """The case-builder. One class, every use case, driven by the pack ruleset.

    Everything that varies is declared in ``case_builder:``; the flow lives here, once.
    Every step degrades independently: a pack declaring only ``conditions:`` gets a
    verdict-only brief; a pack declaring the full ``case_builder:`` gets everything.
    """

    def __init__(self, use_case: str = ""):
        #: Use-case id for this run; the key concept/case docs are tagged with it.
        #: Set from the matched ruleset key, never a Python literal.
        self.use_case = use_case or ""

    def analyze(
        self,
        spec: Optional[Dict[str, Any]],
        logs: Dict[str, List[Dict]],
        analysis: Any,
        entity_map: Optional[Dict[str, Dict[str, str]]] = None,
        transforms: Optional[List[Any]] = None,
        knowledge_pack: Any = None,
        playbook_id: str = "",
        verdict: Optional[ValidationVerdict] = None,
        row_caps: Optional[Dict[str, int]] = None,
        keyed_sources: Optional[Dict[str, bool]] = None,
    ) -> CaseAssessment:
        """Build a CaseAssessment (brief + optional verdict) from the pack ruleset.

        Never raises: every step is individually guarded; each step no-ops when the
        ruleset declares nothing for it.

        Takes the caller's ``verdict`` verbatim: re-deriving it here would lose
        ``row_caps`` and ``keyed_sources``, the two facts a row cannot state. Evaluates
        locally only when the caller passes none.

        ``row_caps`` ({source: cap}): a truncated sweep reports counts as floors.
        ``keyed_sources`` ({source: keyed}): gates mismatch claims in reconciliation.
        """
        brief = InvestigationBrief(use_case=self.use_case, playbook_id=playbook_id)
        spec = spec or {}
        cb = spec.get("case_builder", {}) or {}
        pack_data = (
            getattr(knowledge_pack, "pack_data", None) if knowledge_pack else None
        )

        if verdict is None and spec:
            try:
                verdict = evaluate_verdict(spec, logs, analysis, entity_map, pack_data)
            except Exception as e:  # a verdict failure must never sink the analyzer
                logger.warning("[%s] verdict evaluation failed: %s", self.use_case, e)
                verdict = None
        brief.verdict = verdict

        # Locate the incident's own alert record and reconcile facts it states, before
        # anything derived from the logs. A missing record is noted explicitly.
        try:
            brief.alert_facts = self.build_alert_facts(
                logs,
                spec,
                analysis,
                row_caps=row_caps,
                keyed_sources=keyed_sources,
            )
        except Exception as e:
            logger.warning(
                "[%s] alert-fact reconciliation failed: %s", self.use_case, e
            )
            brief.alert_facts = None

        # Decisive fails and unknowns across all subjects, split by polarity: an exclusion
        # fail argues the alert is explained; a fraud-indicator fail is positive evidence.
        decisive_fails: List[ConditionCheck] = []
        decisive_unknowns: List[ConditionCheck] = []
        decisive_indicators: List[ConditionCheck] = []
        # Exclusion fails that moved no verdict class (see `InvestigationBrief`).
        explanatory_fails: List[ConditionCheck] = []
        # Categorical exclusions that settle who acted (see ConditionCheck.exclusion_kind);
        # a fail here outranks the fraud indicators in the verdict engine.
        categorical_fails: List[ConditionCheck] = []
        # Categorical exclusions that could not be answered. Not a duplicate of
        # `decisive_unknowns`: a categorical exclusion is normally declared
        # `decisive_on: [fail]`, so its unknown reaches no other list here.
        unanswered_attributions: List[ConditionCheck] = []
        lock_targets: List[Dict[str, str]] = []
        primary_label = ""
        primary_subject = ""
        route = ""
        # Subjects the alert already named; the scope sweep diffs against this list.
        known_subjects: List[str] = []
        if verdict is not None:
            for sv in verdict.subjects:
                if sv.subject_value:
                    known_subjects.append(sv.subject_value)
                for _c in sv.checks:
                    # Stamp subject before flattening: the lists span all subjects, so
                    # without it a condition on two subjects differs only by `observed`.
                    c = _c.model_copy(update={"subject": sv.subject_value or ""})
                    if getattr(c, "polarity", "exclusion") == "fraud_indicator":
                        if c.result == "fail":
                            decisive_indicators.append(c)
                        continue
                    if c.decisive and c.result == "fail":
                        decisive_fails.append(c)
                        if getattr(c, "exclusion_kind", "heuristic") == "categorical":
                            categorical_fails.append(c)
                    elif c.decisive and c.result == "unknown":
                        decisive_unknowns.append(c)
                    elif c.result == "fail":
                        # A non-decisive exclusion fail found its explanation but left the
                        # verdict class unmoved. Cannot join `decisive_fails` (read by
                        # `collect_precedents`), so it needs its own list.
                        explanatory_fails.append(c)
                    # Not in the elif chain: a categorical exclusion is normally
                    # `decisive_on: [fail]`, so an unanswered one has `decisive=False`
                    # and would be skipped by every earlier branch.
                    if (
                        c.result == "unknown"
                        and getattr(c, "exclusion_kind", "heuristic") == "categorical"
                    ):
                        unanswered_attributions.append(c)
                if sv.lock_target:
                    lock_targets.append(sv.lock_target)
                if not primary_label:
                    primary_label = sv.verdict
                    primary_subject = sv.subject_value
                for n in sv.notes:
                    if n.startswith("route="):
                        route = route or n[len("route=") :]

        try:
            brief.asset_timeline = self.build_asset_timeline(
                logs, spec, primary_subject
            )
        except Exception as e:
            logger.warning("[%s] asset timeline build failed: %s", self.use_case, e)
            brief.asset_timeline = []
        brief.lock_targets = lock_targets

        # scope_status is always populated so a never-ran sweep cannot appear as one
        # that found nothing. Event window passed for out-of-window counting.
        try:
            assets, extra_subjects, scope_status = self.build_scope_discovery(
                logs,
                spec,
                known_subjects=known_subjects,
                event_window=getattr(analysis, "event_time", None),
                # Whether a result was truncated is a fact no row can state; the sweep is
                # the only step that can widen scope past the alert.
                row_caps=row_caps,
            )
        except Exception as e:
            logger.warning("[%s] scope discovery failed: %s", self.use_case, e)
            assets, extra_subjects, scope_status = [], [], f"not attempted (error: {e})"

        # Subject derivation links: resolved after the sweep because a derived subject
        # looks like new unrelated scope until the link reclassifies it.
        try:
            links = self.build_subject_links(logs, spec)
        except Exception as e:
            logger.warning("[%s] subject-link resolution failed: %s", self.use_case, e)
            links = []
        brief.subject_links = links

        assets, extra_subjects, scope_status = self._reclassify_derived(
            links, assets, extra_subjects, scope_status
        )
        brief.impacted_assets = assets
        brief.additional_subjects = extra_subjects
        brief.scope_status = scope_status

        # Next-steps skeleton, KB concepts, and matching precedents. Derived after scope
        # discovery so the scope step states what the sweep actually found.
        brief.action_backbone = self.derive_action_backbone(
            verdict,
            spec,
            additional_subjects=extra_subjects,
            scope_status=scope_status,
            impacted_assets=assets,
        )
        if self.use_case:
            brief.concept_refs = self.collect_concept_refs(
                knowledge_pack,
                self.use_case,
                cb.get("concepts") or None,
                snippet_chars=_concept_snippet_chars(cb),
            )
            brief.precedents = self.collect_precedents(
                knowledge_pack,
                self.use_case,
                verdict_label=primary_label,
                subject=primary_subject,
                route=route,
                decisive_reason_ids=(
                    [c.id for c in decisive_fails] + [c.id for c in decisive_unknowns]
                ),
            )

        # Explicit join status: "ran, N matches" vs "not evaluated", so the report can
        # distinguish a join that found nothing from one never attempted.
        brief.join_status = self.join_status_from_transforms(
            transforms, cb.get("expected_joins") or None
        )

        brief.decisive_fails = decisive_fails
        brief.decisive_unknowns = decisive_unknowns
        brief.decisive_indicators = decisive_indicators
        brief.explanatory_fails = explanatory_fails
        brief.unanswered_attributions = unanswered_attributions

        # containment_gated: an indicator-driven verdict requires expert confirmation
        # before containment. A categorical exclusion outranks indicators, so
        # categorical_fails being set means they did not drive the verdict.
        categorically_excluded = bool(categorical_fails)
        brief.containment_gated = (
            bool(decisive_indicators) and not categorically_excluded
        )

        notes: List[str] = []
        degraded = bool(verdict.degraded) if verdict is not None else bool(spec)
        # A confirmed fraud indicator or a categorical exclusion makes the verdict terminal:
        # the outcome rests on positive evidence, so a missing projection leaf is moot.
        terminal = bool(decisive_indicators) or categorically_excluded
        guard_notes, guard_degraded = self._projection_guard_notes(logs, spec, terminal)
        notes.extend(guard_notes)
        degraded = degraded or guard_degraded

        scope_notes, scope_degraded = self._scope_honesty_notes(
            cb.get("scope_notes") or {}, scope_status, extra_subjects
        )
        notes.extend(scope_notes)
        degraded = degraded or scope_degraded

        brief.notes = notes
        brief.degraded = degraded
        # Not a note and not a degrade: a carve-out the engine cannot apply is a standing
        # property of the adjudication, independent of this run's coverage.
        brief.unenforced_carve_outs = self._carve_out_caveats(spec, verdict)
        return CaseAssessment(brief=brief, verdict=verdict)

    @staticmethod
    def _carve_out_caveats(spec, verdict) -> List[str]:
        """Procedure carve-outs the engine does not apply -> statements for the report.

        A ruleset's ``do_not_consider`` entries name things that must not read as evidence
        but are not enforced by a condition. Two gates keep each statement true: entries
        marked ``enforced: true`` are skipped (reporting a non-existent limitation is
        false); entries are only emitted where a condition they ``affects`` actually failed
        on some subject.

        Entries with no ``affects`` and no ``note`` are silently skipped.
        """
        entries = (spec or {}).get("do_not_consider") or []
        if not isinstance(entries, list) or verdict is None:
            return []
        failed: set = set()
        for sv in getattr(verdict, "subjects", None) or []:
            for c in getattr(sv, "checks", None) or []:
                if str(getattr(c, "result", "")) == "fail":
                    cid = str(getattr(c, "id", "") or "").strip()
                    if cid:
                        failed.add(cid)
        out: List[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("enforced") is True or str(entry.get("enforced")) == "true":
                continue
            note = str(entry.get("note", "") or "").strip()
            affects = entry.get("affects") or []
            if isinstance(affects, str):
                affects = [affects]
            hit = [str(a).strip() for a in affects if str(a).strip() in failed]
            if not note or not hit:
                continue
            out.append(f"{note} (applies to: {', '.join(sorted(hit))})")
        return out

    @staticmethod
    def _reclassify_derived(links, assets, extra_subjects, scope_status):
        """Move sweep-found subjects that derive from an alerted one out of "new scope".

        Reclassifies rather than drops: assets stay in ``impacted_assets`` and only the
        under-reported-scope work-list is corrected.
        """
        derived = {
            str(ln.related_subject).strip().upper()
            for ln in links
            if ln.role == "parent" and str(ln.related_subject).strip()
        }
        derived |= {
            str(ln.subject).strip().upper()
            for ln in links
            if ln.role == "child" and str(ln.subject).strip()
        }
        if not derived:
            return assets, extra_subjects, scope_status
        related = [s for s in extra_subjects if s.strip().upper() in derived]
        if not related:
            return assets, extra_subjects, scope_status
        extra_subjects = [s for s in extra_subjects if s.strip().upper() not in derived]
        for a in assets:
            if str(a.subject).strip().upper() in derived:
                a.known = True
        kinds = sorted({ln.kind for ln in links if ln.kind}) or ["derivation"]
        was = "is" if len(related) == 1 else "are"
        scope_status = (
            f"{scope_status}; {len(related)} of those ({', '.join(related[:10])}) {was} "
            f"NOT new scope — derived from an alerted subject by a {'/'.join(kinds)} "
            "recorded in the data"
        )
        return assets, extra_subjects, scope_status

    @staticmethod
    def _projection_guard_notes(logs, spec, terminal: bool) -> tuple:
        """Flag decisive-check inputs the retrieval dropped -> (notes, degraded).

        A decisive check reading a leaf the projection never selected returns ``unknown``,
        indistinguishable from "the elements really are absent". The pack declares
        (``case_builder.projection_guard``) which leaves must have arrived.

        Skipped when the verdict is ``terminal``.

        Uses ``_collect_any`` rather than ``_collect``: ``_collect`` drops booleans, so a
        flag-field probe path would always report it missing.
        """
        pg = ((spec or {}).get("case_builder", {}) or {}).get(
            "projection_guard", {}
        ) or {}
        logical = str(pg.get("source", "") or "")
        if not logical:
            return [], False
        real = ((spec or {}).get("sources", {}) or {}).get(logical)
        if not real:
            return [], False
        rows = (logs or {}).get(real) or []
        if not rows:
            # Genuinely no rows. This branch keys off row absence only, not the terminal
            # path below.
            tmpl = str(pg.get("empty_note", "") or "")
            note = tmpl.replace("{source}", str(real)).strip()
            return ([note] if note else []), True
        if terminal:
            return [], False
        notes: List[str] = []
        for probe in pg.get("probe_paths") or []:
            if not isinstance(probe, dict):
                continue
            path = str(probe.get("path", "") or "")
            note = str(probe.get("note", "") or "").strip()
            if not path or not note:
                continue
            if not _collect_any(rows, path):
                notes.append(note)
        return notes, bool(notes)

    @staticmethod
    def _scope_honesty_notes(templates, scope_status: str, extra_subjects) -> tuple:
        """Scope-completeness notes -> (notes, degraded).

        A use case whose sweep did not run has unverified impact scope. Extra subjects
        found are not a degradation (that is the sweep working) but are a material note,
        since the alert under-reported the impact.
        """
        if scope_status and not str(scope_status).startswith("ran"):
            tmpl = str(templates.get("unverified", "") or "")
            note = tmpl.replace("{status}", str(scope_status)).strip()
            return ([note] if note else []), True
        if extra_subjects:
            tmpl = str(templates.get("wider", "") or "")
            note = (
                tmpl.replace("{count}", str(len(extra_subjects)))
                .replace("{subjects}", ", ".join(extra_subjects[:10]))
                .strip()
            )
            return ([note] if note else []), False
        return [], False

    # -- shared deterministic tools ---------------------------------------

    @staticmethod
    def build_alert_facts(
        logs: Dict[str, List[Dict]],
        spec: Optional[Dict[str, Any]],
        analysis: Any,
        max_unrelated: int = 8,
        row_caps: Optional[Dict[str, int]] = None,
        keyed_sources: Optional[Dict[str, bool]] = None,
    ) -> Optional[AlertFacts]:
        """Locate the incident's own alert record and reconcile the facts it states.

        Alert-stated fields are ground truth: a divergence from the logs is a finding,
        never a reason to prefer the logs.

        Everything is pack-declared via ``alert_record:``: source, ``identify`` clauses,
        and ``declares`` entries with confirmation sources. Returns ``None`` when not
        declared.

        ``row_caps`` and ``keyed_sources`` gate ``mismatch``: a truncated or unscoped
        source reads as ``not_found`` with a note. Consulted only for disagreements.
        """
        spec = spec or {}
        ar = spec.get("alert_record", {}) or {}
        # Trigger sentence from the trigger: block. Carried on every return path whether
        # or not the alert row was retrieved.
        trigger = _trigger_sentence(spec)
        logical = str(ar.get("source", "") or "")
        if not logical:
            # A trigger may be declared without an alert record. Return an AlertFacts
            # carrying it rather than None, so trigger information is not lost.
            return AlertFacts(trigger=trigger) if trigger else None
        src_map = spec.get("sources", {}) or {}
        real = src_map.get(logical, "")
        facts = AlertFacts(source=real or logical, trigger=trigger)
        if not real:
            facts.locator = (
                f"the '{logical}' alert source is not mapped for this ruleset — the "
                "incident's own alert record was NOT read"
            )
            return facts
        rows = [r for r in (logs.get(real) or []) if isinstance(r, dict)]
        if not rows:
            facts.locator = (
                f"{real} returned no rows — the incident's own alert record was NOT "
                "retrieved, so no alert-stated fact could be confirmed at source"
            )
            return facts

        ents = _entity_values(analysis)

        # --- locate: score every row against the identify clauses. A row must satisfy
        # every `required` clause; among those, the most clauses matched wins. Ties keep
        # the first row (retrieval order), which is stable across runs.
        clauses = [c for c in (ar.get("identify", []) or []) if isinstance(c, dict)]
        best: Optional[Dict] = None
        best_hits: List[str] = []
        best_score = -1
        rejected: List[Dict] = []
        for row in rows:
            hits: List[str] = []
            missing_required = False
            for clause in clauses:
                ent_type = str(clause.get("from_entity", "") or "")
                values = ents.get(ent_type.lower(), [])
                if not values:
                    # Nothing extracted to match on; the clause cannot be evaluated.
                    continue
                hit = _match_row(row, clause, values)
                if hit:
                    hits.append(f"{ent_type}={hit}")
                elif clause.get("required"):
                    missing_required = True
            if missing_required:
                rejected.append(row)
                continue
            if len(hits) > best_score:
                best, best_hits, best_score = row, hits, len(hits)
            elif best is not None:
                rejected.append(row)
            else:
                rejected.append(row)

        label_fields = [str(f) for f in (ar.get("label_fields", []) or [])]
        if best is None or best_score <= 0:
            # No row carries this incident's identifiers: the alert was not retrieved,
            # not silent. The caller must be able to tell those apart.
            got = [_label_row(r, label_fields) for r in rows[:max_unrelated]]
            facts.locator = (
                f"NOT located: none of the {len(rows)} record(s) in {real} carry this "
                "incident's identifiers"
                + (
                    f" — records seen: {', '.join(g for g in got if g)}"
                    if any(got)
                    else ""
                )
            )
            facts.unrelated_records = [g for g in got if g]
            return facts

        facts.located = True
        facts.record_id = _label_row(best, label_fields)
        facts.locator = f"matched on {', '.join(best_hits)}" if best_hits else "matched"
        facts.unrelated_records = [
            lbl
            for lbl in (_label_row(r, label_fields) for r in rejected[:max_unrelated])
            if lbl
        ]

        # --- reconcile: every fact the alert declares, against the evidence sources.
        for entry in ar.get("declares", []) or []:
            if not isinstance(entry, dict):
                continue
            facts.declared_facts.extend(
                _reconcile_declared(
                    entry,
                    best,
                    ents,
                    logs,
                    src_map,
                    alert_source=real,
                    row_caps=row_caps,
                    keyed_sources=keyed_sources,
                )
            )
        return facts

    @staticmethod
    def build_asset_timeline(
        logs: Dict[str, List[Dict]],
        spec: Optional[Dict[str, Any]],
        subject_value: str = "",
    ) -> List[AssetTimelineEntry]:
        """Walk the record and asset-event rows into a sorted asset-impact chronology.

        Deterministic: pulls creation and asset events off mapped logical sources using
        ``resolve_path`` so nested/JSON/variant row shapes resolve. Returns entries sorted
        by epoch (0 last).

        The pack's ``asset_timeline:`` block declares which logical source carries the events
        and which paths hold the timestamp, actor, asset id, and event kind, because those
        names belong to the backend's record shape, not the engine. With no block declared
        the chronology is empty rather than wrong.
        """
        spec = spec or {}
        src_map = spec.get("sources", {}) or {}
        at = spec.get("asset_timeline", {}) or {}
        entries: List[AssetTimelineEntry] = []

        def _rows(logical: str) -> List[Dict]:
            real = src_map.get(logical)
            rows = logs.get(real) if real else None
            rows = rows or []
            if not subject_value:
                return [r for r in rows if isinstance(r, dict)]
            sv = str(subject_value).strip().lower()
            out = []
            for r in rows:
                if not isinstance(r, dict):
                    continue
                blob = " ".join(str(v) for v in _flat_str_values(r)).lower()
                if sv in blob:
                    out.append(r)
            return out

        def _first(row: Dict, fields: List[str]) -> str:
            for f in fields:
                vals = resolve_path(row, f)
                if vals and str(vals[0]).strip():
                    return str(vals[0])
            return ""

        # Subject-creation event: pack-declared, like every other event. With no
        # `subject_created:` block the chronology starts at the first asset event.
        created = at.get("subject_created", {}) or {}
        if created.get("in"):
            c_logical = str(created["in"])
            for row in _rows(c_logical):
                ts = _first(row, created.get("timestamp_fields", []) or [])
                actor = _first(row, created.get("actor_fields", []) or [])
                scope = _first(row, created.get("actor_scope_fields", []) or [])
                subj = (
                    _first(row, created.get("subject_id_fields", []) or [])
                    or subject_value
                )
                entries.append(
                    AssetTimelineEntry(
                        timestamp=ts,
                        epoch=_epoch(ts),
                        event_type=str(created.get("event_type") or "subject_created"),
                        entity_type=str(created.get("entity_type") or "subject"),
                        entity_value=subj,
                        actor=(
                            f"{actor} @ {scope}".strip(" @") if (actor or scope) else ""
                        ),
                        source=src_map.get(c_logical, c_logical),
                        detail=str(created.get("detail") or "subject created"),
                    )
                )

        # Asset events (issuance / reversal / refund). Two pack-declared shapes:
        #  * `documents:`: asset nested inside record rows; each document is one event.
        #  * `source:`: flat rows on a separate source; one row per event.
        events = at.get("documents", {}) or {}
        if events.get("path") and events.get("in"):
            # `in:` is required, not defaulted. A pack that declares `path:` without `in:`
            # would read from an unintended source or produce a wrong-looking empty chronology.
            logical = str(events["in"])
            status_map = {
                str(k).strip().upper(): str(v)
                for k, v in (events.get("status_events", {}) or {}).items()
            }
            for row in _rows(logical):
                for node in _document_nodes(row, str(events["path"])):
                    status = _first(node, events.get("status_fields", []) or [])
                    etype = status_map.get(status.strip().upper(), "")
                    if not etype:
                        continue
                    # The version's write time is when the document reached this status;
                    # the document's own date field is typically day-granular.
                    ts = _first(
                        node, events.get("timestamp_fields", []) or []
                    ) or _first(row, events.get("record_timestamp_fields", []) or [])
                    actor = _first(node, events.get("actor_fields", []) or [])
                    scope = _first(node, events.get("actor_scope_fields", []) or [])
                    asset = _join_parts(
                        node,
                        events.get("asset_id_fields", []) or [],
                        events.get("asset_id_suffix_fields", []) or [],
                        str(events.get("asset_id_separator", "")),
                    )
                    entries.append(
                        AssetTimelineEntry(
                            timestamp=ts,
                            epoch=_epoch(ts),
                            event_type=etype,
                            entity_type=str(events.get("entity_type") or "asset"),
                            entity_value=asset,
                            actor=(
                                f"{actor} @ {scope}".strip(" @")
                                if (actor or scope)
                                else ""
                            ),
                            source=src_map.get(logical, logical),
                            detail=(f"document status {status}" if status else etype),
                        )
                    )
        flat = at.get("events", {}) or {}
        if flat.get("source"):
            kind_map = {
                str(k).strip().upper(): str(v)
                for k, v in (flat.get("type_events", {}) or {}).items()
            }
            for row in _rows(str(flat["source"])):
                ts = _first(row, flat.get("timestamp_fields", []) or [])
                actor = _first(row, flat.get("actor_fields", []) or [])
                scope = _first(row, flat.get("actor_scope_fields", []) or [])
                rectype = " ".join(
                    p
                    for p in (
                        _first(row, [f]) for f in (flat.get("type_fields", []) or [])
                    )
                    if p
                ).strip()
                asset = _first(row, flat.get("asset_id_fields", []) or [])
                up = rectype.upper()
                etype = next(
                    (v for k, v in kind_map.items() if k and k in up),
                    str(flat.get("default_event") or "asset_event"),
                )
                entries.append(
                    AssetTimelineEntry(
                        timestamp=ts,
                        epoch=_epoch(ts),
                        event_type=etype,
                        entity_type=str(flat.get("entity_type") or "asset"),
                        entity_value=asset,
                        actor=(
                            f"{actor} @ {scope}".strip(" @")
                            if (actor or scope)
                            else ""
                        ),
                        source=src_map.get(str(flat["source"]), str(flat["source"])),
                        detail=(rectype or etype),
                    )
                )

        # Dedup identical asset events. A struct-heavy source is often exploded (one row per
        # nested child), and a settlement feed repeats the same event across line-item rows.
        # Collapse on the event's identity so each real event appears once.
        seen = set()
        deduped: List[AssetTimelineEntry] = []
        for e in entries:
            key = (
                e.event_type,
                e.entity_type,
                e.entity_value,
                e.timestamp,
                e.actor,
                e.detail,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(e)

        deduped.sort(key=lambda e: (e.epoch == 0.0, e.epoch))
        return deduped

    @staticmethod
    def build_scope_discovery(
        logs: Dict[str, List[Dict]],
        spec: Optional[Dict[str, Any]],
        known_subjects: Optional[List[str]] = None,
        limit: int = 200,
        event_window: Any = None,
        row_caps: Optional[Dict[str, int]] = None,
    ) -> tuple:
        """Read the pack's actor-scoped scope sweep -> (assets, additional_subjects, status).

        ``status`` is always non-empty: a never-ran sweep cannot appear as one that found
        nothing. Returns ``([], [], "not attempted ...")`` when not declared or no rows.

        Versioned: ``status`` from highest version; descriptive fields from latest. When
        ``event_date`` is declared, out-of-window assets get ``in_window=False``; boundary
        derived from the evidence.

        ``row_caps`` ({source: cap}): a truncated sweep reports floors and ``status`` carries
        ``_SCOPE_TRUNCATED``. Assets unchanged; only counts and "not named" become floors.
        """
        spec = spec or {}
        sd = spec.get("scope_discovery", {}) or {}
        logical = sd.get("source", "")
        real = (spec.get("sources", {}) or {}).get(logical) if logical else ""
        # No sweep declared: return an empty status, not a "not attempted" one. A use case
        # with no scope-widening step has no gap to report; a declared sweep that returned
        # nothing is a real gap.
        if not logical:
            return [], [], ""
        if not real:
            return (
                [],
                [],
                (
                    f"not attempted (the '{logical}' scope-sweep source is not mapped for "
                    "this ruleset)"
                ),
            )
        rows = [r for r in (logs.get(real) or []) if isinstance(r, dict)]
        if not rows:
            return (
                [],
                [],
                (
                    f"not attempted (the scope-sweep source '{real}' returned no rows — the "
                    "wider impact of this actor is UNKNOWN, not clean)"
                ),
            )

        # Truncated at the cap: every count below is a floor. Detected the same way the
        # verdict engine detects it so the same source reads consistently. `>=` because a
        # backend may overshoot by a row.
        _cap = int((row_caps or {}).get(real, 0) or 0)
        _truncated = bool(_cap) and len(rows) >= _cap
        _floor = "at least " if _truncated else ""
        _cap_note = (
            (
                f"; {_SCOPE_TRUNCATED} — the sweep returned {len(rows)} row(s), which is its "
                f"{_cap}-row cap, so every count here is a LOWER BOUND and further "
                "impacted subjects may exist beyond it"
            )
            if _truncated
            else ""
        )

        fields = sd.get("fields", {}) or {}

        def _vals(row: Dict, key: str) -> List[str]:
            """All values of a pack-named field on one row (accepts a list of candidates)."""
            spec_f = fields.get(key)
            if not spec_f:
                return []
            candidates = spec_f if isinstance(spec_f, list) else [spec_f]
            for f in candidates:
                vals = [
                    str(v).strip() for v in resolve_path(row, str(f)) if str(v).strip()
                ]
                if vals:
                    return vals
            return []

        def _one(row: Dict, key: str) -> str:
            vals = _vals(row, key)
            return vals[0] if vals else ""

        def _version(row: Dict) -> float:
            """This row's version number, or -1 when the pack declares none."""
            for v in _vals(row, "version"):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
            return -1.0

        # Optional pack tolerance in days for the event window. The boundary is derived from
        # the evidence; this is an escape hatch for episodes that spill past their own dates.
        try:
            pad_days = int(float(sd.get("event_window_days", 0) or 0))
        except (TypeError, ValueError):
            pad_days = 0

        incident_from = _day_number(str(getattr(event_window, "start", "") or ""))
        incident_to = _day_number(str(getattr(event_window, "end", "") or ""))

        def _derive_event_window() -> tuple:
            """``(from_day, to_day, provenance)`` derived from the evidence.

            The boundary is the span of event dates on the alerted subjects, unioned with
            the alert's own window. Using the outer bound rather than requiring contiguity
            prevents a gap in evidence from dropping a valid document from containment scope.

            Adjacent activity from the sweep at large is not included; it is surfaced and
            labelled out-of-window instead.
            """
            by_day: Dict[int, int] = {}
            for row in rows:
                subj = _one(row, "subject")
                if not subj or not _matches_any_identifier(subj, known):
                    continue
                d = _day_number(_one(row, "event_date"))
                if d:
                    by_day[int(d)] = by_day.get(int(d), 0) + 1
            bounds = [int(d) for d in (incident_from, incident_to) if d] + list(by_day)
            if not bounds:
                return 0.0, 0.0, ""
            lo, hi = min(bounds), max(bounds)
            span = hi - lo + 1
            dated = sum(by_day.values())
            how = "the alert window" if incident_from or incident_to else ""
            if dated:
                how = (
                    f"{how} + the {dated} dated document(s) on the alerted subject(s)"
                    if how
                    else f"the {dated} dated document(s) on the alerted subject(s)"
                )
            try:
                shown = f"{date.fromordinal(lo)} → {date.fromordinal(hi)}"
            except (ValueError, OverflowError):
                shown = "?"
            return float(lo), float(hi), f"{shown}, {span} day(s), derived from {how}"

        # Normalised once: a backend may append a role suffix, so exact membership
        # would read every row as a stranger.
        known = {
            _norm_identifier(s) for s in (known_subjects or []) if _norm_identifier(s)
        }
        win_from, win_to, window_how = _derive_event_window()
        windowed = bool(win_from and win_to and fields.get("event_date"))

        def _in_window(row: Dict) -> tuple:
            """``(event_date, "in" | "out" | "unknown")`` for one row.

            "unknown" is a third state, not "in". Rows with no event date do not decide
            window membership while dated evidence exists. Comparison is per calendar day,
            not per instant, because event_date is typically a bare date while the incident
            window is a timestamp.
            """
            ev = _one(row, "event_date")
            if not windowed or not ev:
                return ev, "unknown"
            d = _day_number(ev)
            if not d:
                return ev, "unknown"
            inside = (win_from - pad_days) <= d <= (win_to + pad_days)
            return ev, ("in" if inside else "out")

        # One entry per (subject, asset): versioned rows are merged; current state from
        # the highest version. State vocabulary belongs to the pack and the report.
        by_key: Dict[tuple, Any] = {}
        states: Dict[tuple, List[tuple]] = {}
        # Version at which each descriptive field's value was last set; later versions win.
        field_ver: Dict[tuple, Dict[str, float]] = {}
        subjects_seen: List[str] = []
        all_subjects: set = set()
        # Window positions per asset, accumulated as a set of "in"/"out"/"unknown";
        # "unknown" must not decide while dated evidence exists so a bool is insufficient.
        asset_window: Dict[tuple, set] = {}
        # `additional_subjects` drives the report's containment work-list; it must name the
        # incident's actual blast radius, not the actor's whole caseload.
        subject_window: Dict[str, set] = {}
        rows_without_asset = 0
        for row in rows:
            subject = _one(row, "subject")
            # The asset id may be split across columns in the backend (e.g. an issuer
            # prefix and a serial), so join the pack-named parts with its separator.
            parts = [p for p in _vals(row, "asset_id") if p]
            if len(parts) < 2:
                extra = [p for p in _vals(row, "asset_id_suffix") if p]
                parts = parts + extra
            asset_id = (
                str(sd.get("asset_id_separator", "-")).join(parts) if parts else ""
            )
            event_date, position = _in_window(row)
            # Subject discovery counts every row including asset-less ones: a subject the
            # actor touched but never issued a document for is still in scope. Only the
            # asset list requires an id.
            if subject:
                all_subjects.add(subject)
                # An asset-less row contributes no window position for its subject: it has
                # no event date of its own, and "unknown" would muddy a subject whose only
                # assets are out of window.
                subject_window.setdefault(subject, set())
                if asset_id:
                    subject_window[subject].add(position)
                if (
                    not _matches_any_identifier(subject, known)
                    and subject not in subjects_seen
                ):
                    subjects_seen.append(subject)
            if not asset_id:
                rows_without_asset += 1
                continue

            actor_id = _one(row, "actor")
            actor_scope = _one(row, "actor_scope")
            state = _one(row, "status")
            version = _version(row)
            key = (subject, asset_id)
            row_vals = {
                "amount": _one(row, "amount"),
                "currency": _one(row, "currency"),
                "actor": f"{actor_id} @ {actor_scope}".strip(" @"),
                "timestamp": _one(row, "timestamp"),
                "event_date": event_date,
            }

            def _merge(asset, seen: Dict[str, float]) -> None:
                """Fold one row's descriptive values into the asset's merged view.

                A value from a later version wins; these fields describe the asset as of
                that version and the reported status is the latest. An empty value never
                overwrites a populated one: a versioned store nulls amount/currency on
                some versions, so the newest populated row is used.
                """
                for field, val in row_vals.items():
                    if not val:
                        continue
                    if not getattr(asset, field) or version > seen.get(field, -2.0):
                        setattr(asset, field, val)
                        seen[field] = version

            existing = by_key.get(key)
            if existing is None:
                if len(by_key) >= limit:
                    continue
                asset = ImpactedAsset(
                    subject=subject,
                    asset_id=asset_id,
                    status=state,
                    known=bool(subject) and _matches_any_identifier(subject, known),
                )
                by_key[key] = asset
                field_ver[key] = {}
                asset_window[key] = {position}
                _merge(asset, field_ver[key])
                states[key] = [(version, state)] if state else []
                continue
            if state:
                states[key].append((version, state))
            asset_window[key].add(position)
            _merge(existing, field_ver[key])

        def _resolve_window(positions: set) -> bool:
            """Collapse "in"/"out"/"unknown" observations into one in-window bool.

            Dated evidence decides; "unknown" only speaks when nothing else can. Any "in"
            wins within the dated evidence. With no dated evidence at all the answer is True:
            the analyzer never narrows scope on a boundary it cannot compute.
            """
            if "in" in positions:
                return True
            if "out" in positions:
                return False
            return True

        assets = list(by_key.values())
        for key, asset in by_key.items():
            asset.in_window = _resolve_window(asset_window.get(key) or set())
            seen_states = states.get(key) or []
            if not seen_states:
                asset.status = ""
                continue
            # Highest version is the current state. Rows arrive in no useful order; an asset
            # voided in one version can be reissued in a later one.
            current_version = max(v for v, _ in seen_states)
            current = next(s for v, s in seen_states if v == current_version)
            superseded = [
                s for v, s in sorted(seen_states, key=lambda vs: vs[0]) if s != current
            ]
            # De-dupe the history, preserving version order.
            hist: List[str] = []
            for s in superseded:
                if s not in hist:
                    hist.append(s)
            if hist and current_version >= 0:
                asset.status = f"{current} (current; was {' -> '.join(hist)})"
            elif hist:
                # No version field declared: the states are unordered, so current is
                # unknown.
                asset.status = " / ".join([current] + hist) + " (order unknown)"
            else:
                asset.status = current

        # In-window assets first, so a report or prompt truncating the list keeps the ones
        # that are actually incident scope.
        assets.sort(key=lambda a: not a.in_window)
        extra_note = (
            f"; {rows_without_asset} row(s) with no asset" if rows_without_asset else ""
        )
        if windowed:
            in_scope = [a for a in assets if a.in_window]
            n_out = len(assets) - len(in_scope)
            # `additional_subjects` is the containment work-list, so it holds only subjects
            # with in-window evidence. The rest stay visible as out-of-window entries in
            # `impacted_assets` and are counted in the status.
            new_in = [
                s
                for s in subjects_seen
                if _resolve_window(subject_window.get(s) or set())
            ]
            out_subj = {a.subject for a in assets if not a.in_window and a.subject} - {
                a.subject for a in assets if a.in_window
            }
            n_out_subj = len(out_subj)
            out_note = (
                f"; also {n_out} asset(s) across {n_out_subj} subject(s) OUTSIDE the "
                "incident window (the same actor's adjacent activity — reported for review, "
                "NOT counted as incident scope)"
                if n_out
                else ""
            )
            # Show the boundary and its provenance; every scope count below depends on it.
            how_note = f"; incident window {window_how}" if window_how else ""
            # Split unalerted subjects by whether they hold an in-window asset. A subject
            # with a document is material exposure; one with none is a record the actor
            # created but never issued against, with nothing to reverse.
            with_asset = {a.subject for a in in_scope if a.subject}
            n_doc = len([s for s in new_in if s in with_asset])
            n_bare = len(new_in) - n_doc
            new_note = (
                f" ({n_doc} holding an in-window document, {n_bare} with none)"
                if n_bare and n_doc
                else (f" ({n_bare} with no document issued)" if n_bare else "")
            )
            status = (
                f"ran, {_floor}{len(in_scope)} asset(s) in the incident window across "
                f"{_floor}{len({a.subject for a in in_scope})} subject(s); "
                f"{_floor}{len(new_in)} NOT named in the alert{new_note}"
                f"{out_note}{extra_note}{how_note}{_cap_note}"
            )
            # Document-holding subjects first: the work-list is truncated downstream, and a
            # subject with a live document must never be pushed out by a document-less one.
            new_in.sort(key=lambda s: s not in with_asset)
            return assets, new_in, status
        status = (
            f"ran, {_floor}{len(assets)} asset(s) across "
            f"{_floor}{len(all_subjects)} subject(s); "
            f"{_floor}{len(subjects_seen)} NOT named in the alert"
            f"{extra_note}{_cap_note}"
        )
        return assets, subjects_seen, status

    @staticmethod
    def build_subject_links(
        logs: Dict[str, List[Dict]],
        spec: Optional[Dict[str, Any]],
    ) -> List[SubjectLink]:
        """Read the pack's derivation links between subjects -> ``[SubjectLink]``.

        Feeds ``_reclassify_derived`` so sweep-found derived subjects are not counted as
        new scope.

        Pack-declared in ``spec['subject_links']``: source, array path, related-subject
        leaf, and a ``roles`` map from type codes to ``parent``/``child``. An unmapped code
        yields ``role='unknown'``; wrong direction inverts the finding.

        Unbounded: deduplication on the relation is the real bound; display bounds belong
        in the report layer. ``quote`` from a pack-declared template.
        """
        spec = spec or {}
        sl = spec.get("subject_links", {}) or {}
        logical = sl.get("source", "")
        real = (spec.get("sources", {}) or {}).get(logical) if logical else ""
        if not logical or not real:
            return []
        rows = [r for r in (logs.get(real) or []) if isinstance(r, dict)]
        if not rows:
            return []

        array_path = str(sl.get("array", "") or "")
        if not array_path:
            return []
        subject_field = str(sl.get("subject_field", "") or "")
        fields = sl.get("fields", {}) or {}
        roles = {
            str(k).strip().upper(): str(v).strip().lower()
            for k, v in (sl.get("roles", {}) or {}).items()
        }
        kind = str(sl.get("kind", "") or "")
        template = str(sl.get("quote_template", "") or "")

        def _leaf(node: Any, key: str) -> str:
            """First value of a pack-named leaf on one link element (accepts a list)."""
            spec_f = fields.get(key)
            if not spec_f:
                return ""
            for f in spec_f if isinstance(spec_f, list) else [spec_f]:
                for v in resolve_path(node, str(f)):
                    text = str(v).strip()
                    if text:
                        return text
            return ""

        out: List[SubjectLink] = []
        seen: set = set()
        for row in rows:
            # The subject this row belongs to (the link names what it is related to).
            owner = ""
            if subject_field:
                vals = [str(v).strip() for v in resolve_path(row, subject_field)]
                owner = next((v for v in vals if v), "")
            # Dedupe on the relation: a versioned backend repeats the link on every version.
            # _resolve_nodes returns the path's node; an array path yields it whole; flatten.
            nodes: List[Any] = []
            for node in _resolve_nodes(row, array_path.split(".")):
                nodes.extend(node if isinstance(node, list) else [node])
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                related = _leaf(node, "related_subject")
                if not related or related == owner:
                    continue
                code = _leaf(node, "role").strip().upper()
                role = roles.get(code, "unknown")
                link = SubjectLink(
                    subject=owner,
                    related_subject=related,
                    role=role if role in ("parent", "child") else "unknown",
                    kind=kind,
                    actor=_leaf(node, "actor"),
                    scope=_leaf(node, "scope"),
                    occurred=_leaf(node, "occurred"),
                    element_id=_leaf(node, "element_id"),
                    status=_leaf(node, "status"),
                )
                key = (link.subject, link.related_subject, link.role, link.element_id)
                if key in seen:
                    continue
                seen.add(key)
                if template:
                    link.quote = _render_link_quote(template, link)
                out.append(link)
        return out

    @staticmethod
    def derive_action_backbone(
        verdict: Optional[ValidationVerdict],
        spec: Optional[Dict[str, Any]],
        additional_subjects: Optional[List[str]] = None,
        scope_status: str = "",
        impacted_assets: Optional[List[Any]] = None,
    ) -> List[str]:
        """Deterministic next-steps skeleton the LLM narrates from, never contradicts.

        Keyed off the rolled-up verdict label. ``additional_subjects``, ``scope_status``,
        and ``impacted_assets`` come from the scope sweep, so the scope step states what
        the sweep actually found rather than recommending a sweep the pipeline may have
        already performed.

        Wording is pack-declared (``case_builder.action_templates``); branching is not.
        The defaults in ``_ACTION_TEMPLATES`` are domain-neutral so a pack declaring no
        templates gets generic prose rather than another use case's procedure wording.
        """
        labels = (spec or {}).get("labels", {}) or {}
        fraud = labels.get("fraud", "VALID FRAUD")
        fp = labels.get("false_positive", "FALSE POSITIVE")
        insuff = labels.get("insufficient", "INSUFFICIENT DATA")
        tpl = _action_templates(spec)

        if verdict is None or not verdict.subjects:
            return [tpl["no_verdict"]]

        steps: List[str] = []
        for sv in verdict.subjects:
            subj = sv.subject_value
            subj_label = tpl["subject_label"].replace("{subject}", str(subj))
            label = sv.verdict
            if label == fraud:
                lt = sv.lock_target if isinstance(sv.lock_target, dict) else {}
                scope_val = lt.get("scope", "")
                identity = lt.get("identity", "")
                who = f"{identity} @ {scope_val}".strip(" @")
                # Role words are read from lock_target (verdict engine stamps
                # scope_label/identity_label there), not written as literals.
                prov = "; ".join(
                    f"{role} from {lt[k]}"
                    for k, role in (
                        (
                            "identity_field",
                            str(lt.get("identity_label", "") or "").strip().lower()
                            or "identity",
                        ),
                        (
                            "scope_field",
                            str(lt.get("scope_label", "") or "").strip().lower()
                            or "scope",
                        ),
                    )
                    if lt.get(k)
                )
                if prov:
                    who = f"{who} ({prov})" if who else f"({prov})"
                # Action verb comes from `lock_target`; no verb declared means say so,
                # not invent one. Verbs are not interchangeable across platform or identity
                # class.
                verb = str(lt.get("action", "") or "").strip()
                rationale = str(lt.get("action_rationale", "") or "").strip()
                key = "contain" if verb else "contain_no_verb"
                contain = (
                    tpl[key]
                    .replace("{subject_label}", subj_label)
                    .replace("{verb}", verb)
                    .replace("{target}", who or tpl["target_fallback"])
                )
                if rationale:
                    contain = f"{contain} {rationale}"
                # A prerequisite is emitted before the action it blocks.
                prereq = str(lt.get("prerequisites", "") or "").strip()
                # A fraud verdict driven by positive indicators rather than the exclusion
                # fingerprint requires expert confirmation before containment.
                indicator_fails = [
                    c.label or c.id
                    for c in sv.checks
                    if getattr(c, "polarity", "exclusion") == "fraud_indicator"
                    and c.result == "fail"
                ]
                scope = _scope_step(
                    subj, additional_subjects, scope_status, impacted_assets, tpl
                )
                prereq_step = (
                    tpl["prerequisite"]
                    .replace("{subject_label}", subj_label)
                    .replace("{prerequisite}", prereq)
                    if prereq
                    else ""
                )
                preserve = tpl["preserve"].replace("{subject_label}", subj_label)
                if indicator_fails:
                    steps.append(
                        tpl["fraud_candidate"]
                        .replace("{subject_label}", subj_label)
                        .replace("{fraud_label}", fraud)
                        .replace("{indicators}", "; ".join(indicator_fails))
                    )
                    steps.append(scope)
                    if prereq_step:
                        steps.append(prereq_step)
                    steps.append(
                        tpl["contain_on_confirmation"].replace("{contain}", contain)
                    )
                    steps.append(preserve)
                else:
                    if prereq_step:
                        steps.append(prereq_step)
                    steps.append(contain)
                    steps.append(scope)
                    steps.append(preserve)
            elif label == fp:
                # Use the finding text, not the label: an exclusion fail negates its label
                # so the requirement wording would state the opposite. A fraud indicator's
                # label states its finding, so a fail there affirms rather than negates.
                fails = [
                    _finding_headline(c)
                    for c in sv.checks
                    if c.decisive and c.result == "fail"
                ]
                reason = f" (decisive exclusion: {'; '.join(fails)})" if fails else ""
                steps.append(
                    tpl["close"]
                    .replace("{subject_label}", subj_label)
                    .replace("{fp_label}", fp)
                    .replace("{reason}", reason)
                )
            elif label == insuff:
                unknowns = [
                    c.label or c.id
                    for c in sv.checks
                    if c.decisive and c.result == "unknown"
                ]
                gap = f" (missing: {'; '.join(unknowns)})" if unknowns else ""
                # Two roads: incomplete evidence (re-run helps) vs. complete with nothing
                # deciding (re-run changes nothing). Read the verdict's own note rather
                # than re-deriving: `no_verdict_reason=` means evidence was complete.
                examined = any(
                    str(n).startswith("no_verdict_reason=")
                    for n in (getattr(sv, "notes", []) or [])
                )
                steps.append(
                    tpl["insufficient_examined" if examined else "insufficient"]
                    .replace("{subject_label}", subj_label)
                    .replace("{gap}", gap)
                )
            else:
                steps.append(
                    tpl["other"]
                    .replace("{subject_label}", subj_label)
                    .replace("{label}", str(label))
                )
        return steps

    @staticmethod
    def collect_concept_refs(
        knowledge_pack: Any,
        use_case: str,
        ids: Optional[List[str]] = None,
        snippet_chars: int = DEFAULT_CONCEPT_SNIPPET_CHARS,
    ) -> List[ConceptRef]:
        """Surface the use case's concept docs as compact ConceptRefs for the brief."""
        if knowledge_pack is None:
            return []
        try:
            docs = knowledge_pack.concepts_for(use_case, ids)
        except Exception:
            return []
        refs: List[ConceptRef] = []
        for d in docs:
            meta = d.get("metadata", {}) or {}
            refs.append(
                ConceptRef(
                    concept_id=meta.get("concept_id", d.get("title", "")),
                    title=meta.get("title") or d.get("title", ""),
                    snippet=_strip_frontmatter(d.get("content", ""))[:snippet_chars],
                    source_path=meta.get("file_path", ""),
                )
            )
        return refs

    @staticmethod
    def collect_precedents(
        knowledge_pack: Any,
        use_case: str,
        verdict_label: str = "",
        subject: str = "",
        route: str = "",
        decisive_reason_ids: Optional[List[str]] = None,
        limit: int = 3,
        snippet_chars: int = 400,
    ) -> List[CasePrecedent]:
        """Deterministically match resolved past investigations to this case.

        Scores each use-case case doc by same-verdict, shared route, and decisive-reason
        keyword overlap, returns the top ``limit`` as CasePrecedents. Same-verdict cases
        are strongly preferred so the LLM anchors to how comparable cases were closed.
        """
        if knowledge_pack is None:
            return []
        try:
            docs = knowledge_pack.cases_for(use_case)
        except Exception:
            return []
        if not docs:
            return []
        want_verdict = str(verdict_label).strip().lower()
        want_route = str(route).strip().lower()
        reason_ids = {str(r).lower() for r in (decisive_reason_ids or [])}

        scored: List[tuple] = []
        for d in docs:
            meta = d.get("metadata", {}) or {}
            score = 0
            if (
                want_verdict
                and str(meta.get("verdict", "")).strip().lower() == want_verdict
            ):
                score += 5
            if want_route and want_route in str(meta.get("route", "")).strip().lower():
                score += 2
            reasons_blob = " ".join(
                str(x) for x in (meta.get("decisive_reasons") or [])
            ).lower()
            score += sum(1 for rid in reason_ids if rid and rid in reasons_blob)
            scored.append((score, d))

        scored.sort(key=lambda t: t[0], reverse=True)
        out: List[CasePrecedent] = []
        for score, d in scored[:limit]:
            meta = d.get("metadata", {}) or {}
            reasons = meta.get("decisive_reasons") or []
            if isinstance(reasons, str):
                reasons = [reasons]
            out.append(
                CasePrecedent(
                    case_id=meta.get("case_id", d.get("title", "")),
                    verdict=str(meta.get("verdict", "")),
                    subject=str(meta.get("subject", "")),
                    decisive_reasons=[str(r) for r in reasons],
                    resolution=str(meta.get("resolution", "")).strip(),
                    date=str(meta.get("date", "")),
                    snippet=_strip_frontmatter(d.get("content", ""))[:snippet_chars],
                    source_path=meta.get("file_path", ""),
                )
            )
        return out

    @staticmethod
    def join_status_from_transforms(
        transforms: Optional[List[Any]], expected: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Explicit per-join outcome so the report can't confuse 'no matches' with 'not run'.

        Reads the executed cross_source_overlap transforms. Each such transform's label
        maps to "ran, N matches"; any ``expected`` join label with no transform maps to
        "not evaluated (no matching data returned)".
        """
        status: Dict[str, str] = {}
        for t in transforms or []:
            op = getattr(t, "op", "")
            if op != "cross_source_overlap":
                continue
            label = getattr(t, "label", "") or "join"
            rows = getattr(t, "rows", []) or []
            note = getattr(t, "note", "") or ""
            status[label] = f"ran, {len(rows)} cross-source match(es)" + (
                f" — {note}" if note else ""
            )
        for label in expected or []:
            status.setdefault(label, "not evaluated (no matching data returned)")
        return status


# -- module helpers -------------------------------------------------------


def _carries_other_facts(value: str, ents: Dict[str, List[str]]) -> bool:
    """True when ``value`` is a prose container that carries facts rather than being one.

    A value that strictly contains some entity value from the incident is a container;
    one that contains none is itself the fact. Comparison is case-insensitive and against
    every entity type. Strict containment: a field that equals the entity it carries is
    the fact stated directly.
    """
    hay = str(value).strip()
    if not hay:
        return False
    low = hay.casefold()
    for values in ents.values():
        for v in values:
            needle = str(v).strip().casefold()
            if needle and needle != low and needle in low:
                return True
    return False


def _render_link_quote(template: str, link: "SubjectLink") -> str:
    """Render a pack ``quote_template`` against one link -> the operator's own notation.

    Placeholders are ``{name}`` for any SubjectLink field, plus ``{occurred_day}`` for the
    day-of-month + month rendering (e.g. ``09JUL``). Uses an explicit scan rather than
    ``str.format`` so an unknown placeholder degrades to empty and a literal brace in a
    template never raises.
    """
    values = {k: str(v or "") for k, v in link.model_dump().items()}
    values["occurred_day"] = ""
    day = _day_number(link.occurred)
    if day:
        try:
            d = date.fromordinal(int(day))
            months = (
                "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()  # noqa: E501
            )
            values["occurred_day"] = f"{d.day:02d}{months[d.month - 1]}"
        except (ValueError, OverflowError, IndexError):
            values["occurred_day"] = ""
    out: List[str] = []
    i = 0
    while i < len(template):
        ch = template[i]
        if ch == "{":
            end = template.find("}", i)
            if end == -1:
                out.append(template[i:])
                break
            out.append(values.get(template[i + 1 : end].strip(), ""))
            i = end + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out).strip()


def _trigger_sentence(spec: Optional[Dict[str, Any]]) -> str:
    """The ruleset's ``trigger:`` block as one sentence (``""`` when it declares none).

    ``description`` is emitted verbatim: it is the only place that can state what the
    detector does not consider. The quantitative keys are appended as a parenthetical when
    present.

    ``scope`` is emitted as the bare declared word; the engine has no generic term for the
    domain's own unit.
    """
    trig = (spec or {}).get("trigger", {}) or {}
    if not isinstance(trig, dict):
        # A scalar `trigger: "..."` is a valid author choice; accept it rather than
        # silently ignoring it.
        return str(trig or "").strip()
    desc = str(trig.get("description", "") or "").strip()
    bits = []
    if trig.get("min_assets") not in (None, ""):
        bits.append(f"at least {trig['min_assets']} subject(s)")
    if str(trig.get("window", "") or "").strip():
        bits.append(f"within {str(trig['window']).strip()}")
    if str(trig.get("scope", "") or "").strip():
        bits.append(f"in one {str(trig['scope']).strip()}")
    quant = ", ".join(bits)
    if desc and quant:
        return f"{desc} (threshold: {quant})"
    return desc or (f"Fires on {quant}." if quant else "")


def _reconcile_declared(
    entry: Dict[str, Any],
    alert_row: Dict,
    ents: Dict[str, List[str]],
    logs: Dict[str, List[Dict]],
    src_map: Dict[str, str],
    alert_source: str = "",
    row_caps: Optional[Dict[str, int]] = None,
    keyed_sources: Optional[Dict[str, bool]] = None,
) -> List[DeclaredFact]:
    """One ``declares:`` entry -> a DeclaredFact per value the alert states.

    Declared side: from the alert record, falling back to extracted entities. Found side:
    the logical sources the entry names.

    ``alert_source`` is excluded: a source cannot corroborate a value it supplied. A fact
    with no independent source is ``stated``, not ``not_found``.

    ``row_caps``/``keyed_sources`` gate ``mismatch``: a truncated or unscoped source reads
    as ``not_found`` with a note. A sibling-only source corroborated those values and is
    silent about this one; it does not contradict it.
    """
    field = str(entry.get("field", "") or entry.get("from_entity", "") or "")
    normalize = str(entry.get("normalize", "") or "")
    declared = _clause_values(alert_row, entry)
    ent_type = str(entry.get("from_entity", "") or "").lower()
    if not declared and ent_type:
        declared = list(ents.get(ent_type, []))
    if not declared:
        return []
    # An alert field may be a structured value or a free-text body. Extracted entities
    # narrow it: a declared value that contains an extracted value declares that value.
    # The body fallback only applies where the declared value strictly contains the needle.
    if ent_type and ents.get(ent_type):
        narrowed = [
            want
            for want in ents[ent_type]
            if any(_values_match(v, want, normalize) for v in declared)
        ]
        if narrowed:
            declared = narrowed
        elif any(_carries_other_facts(v, ents) for v in declared):
            declared = list(ents[ent_type])

    # Evidence from all named sources, computed once per entry (not per declared value).
    observed_by_source: List[tuple] = []
    independent = 0
    # Per source, why a disagreement would not be a contradiction (truncation, unscoped
    # query). Built once here, not per declared value.
    caveats: Dict[str, str] = {}
    for logical in entry.get("confirm_in", []) or []:
        real = src_map.get(str(logical), "")
        # The alert record's own source is not a second reading of the fact; skip it.
        if real and alert_source and real == alert_source:
            continue
        # Counted where it is named, not where it answered: a declared source that returned
        # nothing is a retrieval gap (`not_found`), distinguishable from `stated` (where
        # nothing independent was ever asked).
        independent += 1
        rows = (
            [r for r in (logs.get(real) or []) if isinstance(r, dict)] if real else []
        )
        if not rows:
            continue
        # A disagreement is not a contradiction if truncated (`>=`: may overshoot) or
        # unscoped. `keyed_sources is not None` guards against an empty dict marking
        # every source unkeyed.
        cap = int((row_caps or {}).get(real) or 0)
        if cap and len(rows) >= cap:
            caveats[real] = (
                f"{real} was TRUNCATED at its {cap}-row cap, so the values it carries "
                "are a sample and not its answer — the declared value may be in the "
                "rows that were not returned"
            )
        elif keyed_sources is not None and not keyed_sources.get(real):
            caveats[real] = (
                f"the query on {real} did not constrain this subject, so the values it "
                "carries may belong to other identities and cannot contradict the alert"
            )
        observed: List[str] = []
        for row in rows:
            for v in _clause_values(row, {"fields": entry.get("confirm_fields", [])}):
                if v not in observed:
                    observed.append(v)
        if observed:
            observed_by_source.append((real, observed))

    # A fact with no named independent source is `stated`, not `not_found`: there was
    # never a second reading to produce a gap. An entry whose only named source is the
    # alert's own is in the same position.
    corroborable = independent > 0

    out: List[DeclaredFact] = []
    for want in dict.fromkeys(declared):
        fact = DeclaredFact(
            field=field or ent_type,
            declared=want,
            status="not_found" if corroborable else "stated",
        )
        if not corroborable:
            out.append(fact)
            continue
        # A fact may be confirmable in more than one place, so every named source is tried
        # before concluding: one source lacking this form of the value must not turn a
        # confirmation elsewhere into a mismatch.
        mismatch: Optional[tuple] = None
        # Source carrying only values this same alert also declares: it corroborated the
        # siblings and said nothing about this one.
        siblings_only: Optional[tuple] = None
        others = [d for d in dict.fromkeys(declared) if d != want]
        for real, observed in observed_by_source:
            hit = next((o for o in observed if _values_match(o, want, normalize)), "")
            if hit:
                fact.status = "confirmed"
                fact.found = hit
                fact.source = real
                break
            if others and all(
                any(_values_match(o, other, normalize) for other in others)
                for o in observed
            ):
                if siblings_only is None:
                    siblings_only = (real, observed)
                continue
            if mismatch is None:
                mismatch = (real, observed)
        else:
            if mismatch is None and siblings_only is not None:
                # A source carrying only sibling values corroborated those and is silent
                # about this one; not a contradiction. Only applies where `others` is
                # non-empty and no undeclared value was observed (a stranger contradicts).
                real, observed = siblings_only
                fact.status = "not_found"
                fact.found = ", ".join(observed[:5])
                fact.source = real
                fact.note = (
                    f"every value {real} carries for this fact is one the alert ALSO "
                    "declares, so it corroborates those and is silent about this one — "
                    "not a contradiction"
                )
            elif mismatch is not None:
                # Every named source carries a different value from the declared one.
                real, observed = mismatch
                # A truncated or unscoped result disagrees with the alert only in the
                # rows it happened to return; report as `not_found` with a note.
                fact.status = "not_found" if caveats.get(real) else "mismatch"
                fact.note = caveats.get(real, "")
                fact.found = ", ".join(observed[:5])
                fact.source = real
        out.append(fact)
    return out


# Domain-neutral next-step wording. A pack overrides via ``case_builder.action_templates``.
# Defaults use engine vocabulary so a pack declaring nothing gets generic prose.
# Placeholders use ``str.replace`` not ``.format``; YAML strings may contain braces.
_ACTION_TEMPLATES = {
    "subject_label": "{subject}",
    "target_fallback": "(record creator)",
    "no_verdict": (
        "Review the correlated evidence and confirm whether the alert reflects genuine "
        "fraud before taking containment action."
    ),
    "contain": "{subject_label}: {verb} Target: {target}.",
    "contain_no_verb": (
        "{subject_label}: containment target {target}. NO action verb could be resolved: "
        "the ruleset declares none for this platform/actor class, so the action set must "
        "be selected by a human."
    ),
    "contain_on_confirmation": "On expert confirmation — {contain}",
    "prerequisite": "{subject_label}: {prerequisite}",
    "fraud_candidate": (
        "{subject_label}: {fraud_label} candidate on positive indicators ({indicators}). "
        "An EXPERT MUST CONFIRM before containment — do NOT auto-void/lock/freeze."
    ),
    "preserve": (
        "{subject_label}: raise a follow-up record for the investigating team and preserve "
        "session + transaction evidence."
    ),
    "close": (
        "{subject_label}: CLOSE the incident as {fp_label}{reason}. No containment required."
    ),
    "insufficient": (
        "{subject_label}: pull the full evidence set for the window{gap} and re-run before "
        "deciding."
    ),
    # Same label, evidence complete: a re-run would return identical rows. Escalate
    # for human judgement; do not close and do not contain.
    "insufficient_examined": (
        "{subject_label}: the evidence was retrieved in full and nothing in it decided the "
        "case — a re-run will return the same rows. Escalate for human judgement on whether "
        "this activity was authorised; do not close as clean and do not contain."
    ),
    "other": "{subject_label}: {label} — review the evidence.",
    # -- scope sweep outcomes (three, never conflated)
    "scope_wider": (
        "{subject_label}: SCOPE IS WIDER THAN THE ALERT — the actor-scoped sweep also "
        "returned {subjects}."
    ),
    # Appended to ``scope_wider``; space-joined by the engine so a YAML folded scalar
    # cannot lose its leading space.
    "scope_wider_actionable": (
        "Treat these as in-scope: check each one's assets and neutralise any still-active "
        "document before closing."
    ),
    "scope_wider_no_document": (
        "The sweep also returned {count} record(s) by the same actor with no document "
        "issued in the window ({subjects}) — review them, but there is nothing to void."
    ),
    "scope_wider_none_with_asset": "no further record holding a document",
    "scope_clean": (
        "{subject_label}: the actor-scoped sweep ran and found no subject beyond those "
        "already in the alert — the impact scope is bounded as reported."
    ),
    # The sweep ran but hit its row cap and found no extra subject. Distinct from
    # `scope_clean` (exhaustive, no extra subjects) and `scope_not_run` (no result).
    "scope_truncated": (
        "{subject_label}: the actor-scoped sweep ran but was TRUNCATED at its row cap "
        "({status}), so it named no subject beyond the alert only as far as it could see — "
        "the wider impact of this actor is UNKNOWN, not clean. Re-run the sweep with a "
        "higher row cap, or narrowed by window/scope so it completes, before closing."
    ),
    "scope_not_run": (
        "{subject_label}: the scope sweep did NOT run ({status}), so the wider impact of "
        "this actor is UNKNOWN, not clean. Sweep the acting actor over a MULTI-DAY window "
        "(not the alert day alone) for further impacted subjects before closing."
    ),
}


def _action_templates(spec: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Next-step wording for this use case: pack declarations merged over neutral defaults.

    Merged key-by-key so a pack can re-word one step without restating the rest; a
    mistyped key falls back to the neutral default rather than rendering an empty step.
    """
    declared = ((spec or {}).get("case_builder", {}) or {}).get(
        "action_templates", {}
    ) or {}
    out = dict(_ACTION_TEMPLATES)
    for k, v in declared.items():
        if isinstance(v, str) and v.strip():
            out[str(k)] = v.strip()
    return out


def _scope_step(
    subject: str,
    additional_subjects,
    scope_status: str,
    impacted_assets=None,
    tpl: Optional[Dict[str, str]] = None,
) -> str:
    """The scope line, phrased by what the sweep actually did.

    Four distinct outcomes: extra subjects found (name them); sweep ran clean and saw
    everything (scope bounded); sweep ran but hit its row cap (wider impact unknown, not
    clean); sweep did not run (wider impact unknown).

    When sweep assets are supplied, extra subjects are split by whether they hold an
    in-window document: "neutralise any still-active document" applies only to those.
    """
    tpl = tpl or dict(_ACTION_TEMPLATES)
    subj_label = tpl["subject_label"].replace("{subject}", str(subject))
    extra = [str(s) for s in (additional_subjects or []) if str(s).strip()]
    if extra:
        assets = impacted_assets if isinstance(impacted_assets, list) else []
        with_doc = {
            getattr(a, "subject", "")
            for a in assets
            if getattr(a, "in_window", True) and getattr(a, "asset_id", "")
        }
        with_asset_subj = [s for s in extra if s in with_doc] if with_doc else extra
        bare = [s for s in extra if s not in with_doc] if with_doc else []
        line = (
            tpl["scope_wider"]
            .replace("{subject_label}", subj_label)
            .replace(
                "{subjects}",
                ", ".join(with_asset_subj[:20]) or tpl["scope_wider_none_with_asset"],
            )
        )
        if with_asset_subj:
            line = f"{line} {tpl['scope_wider_actionable']}"
        if bare:
            tail = (
                tpl["scope_wider_no_document"]
                .replace("{count}", str(len(bare)))
                .replace("{subjects}", ", ".join(bare[:20]))
            )
            line = f"{line} {tail}"
        return line
    if scope_status.startswith("ran"):
        # A sweep at its row cap that returned no extra subject cannot claim a bounded
        # scope; `scope_wider` handles the case where a subject was found and carries
        # its own floor counts.
        if _SCOPE_TRUNCATED in scope_status:
            return (
                tpl["scope_truncated"]
                .replace("{subject_label}", subj_label)
                .replace("{status}", scope_status)
            )
        return tpl["scope_clean"].replace("{subject_label}", subj_label)
    return (
        tpl["scope_not_run"]
        .replace("{subject_label}", subj_label)
        .replace("{status}", scope_status or "no result")
    )


def _document_nodes(row: Dict, path: str) -> List[Dict]:
    """The dict documents at ``path`` in ``row``, whatever container shape they arrive in.

    A nested document array reaches the engine three ways depending on the backend and on
    how the query projected it: as a real list, as a JSON string, or already flattened to an
    underscore alias (``parent_child_array``). ``_resolve_nodes`` handles the alias and the
    JSON decode but returns a terminal list as a single node, so unwrap one more level here.
    """
    out: List[Dict] = []
    for node in _resolve_nodes(row, path.split(".")):
        candidates = node if isinstance(node, list) else [node]
        for item in candidates:
            if isinstance(item, str):
                item = _maybe_json(item)
            if isinstance(item, list):
                out.extend(x for x in item if isinstance(x, dict))
            elif isinstance(item, dict):
                out.append(item)
    return out


def _join_parts(
    node: Dict, id_fields: List[str], suffix_fields: List[str], separator: str
) -> str:
    """One asset id from possibly-split parts (an issuer prefix + a serial).

    Mirrors ``build_scope_discovery``'s join so a document read through the timeline and the
    same document read through the sweep produce the same id; the two views of one asset must
    reconcile.
    """
    parts = [str(v) for f in id_fields for v in resolve_path(node, f) if str(v).strip()]
    if len(parts) < 2:
        parts += [
            str(v)
            for f in suffix_fields
            for v in resolve_path(node, f)
            if str(v).strip()
        ]
    return separator.join(parts) if parts else ""


def _epoch(ts: str) -> float:
    dt = _parse_ts(ts)
    if dt is None:
        return 0.0
    try:
        return dt.timestamp()
    except Exception:
        return 0.0


def _day_number(ts: str) -> float:
    """The calendar day of a timestamp as a monotonic day-number, or 0.0 if unparseable.

    Does not go through ``_epoch``: a bare date like ``2026-07-27`` parses naive, so
    ``.timestamp()`` would interpret it in the machine's local zone and, east of UTC, shift
    it back a day. Date fields of this kind are calendar dates in the operator's frame,
    so comparison runs on the calendar triple, not on an instant.
    """
    dt = _parse_ts(ts)
    if dt is None:
        return 0.0
    try:
        return float(dt.toordinal())
    except Exception:
        return 0.0


def _flat_str_values(row: Dict, depth: int = 0) -> List[str]:
    """Shallow flatten of a row's scalar values (for subject substring matching)."""
    out: List[str] = []
    if depth > 3 or not isinstance(row, (dict, list)):
        if row is not None:
            out.append(str(row))
        return out
    items = row.values() if isinstance(row, dict) else row
    for v in items:
        if isinstance(v, (dict, list)):
            out.extend(_flat_str_values(v, depth + 1))
        elif v is not None:
            out.append(str(v))
    return out


def _strip_frontmatter(content: str) -> str:
    """Drop a leading YAML frontmatter fence so snippets are readable prose."""
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            content = content[end + 4 :]
    return content.strip()
