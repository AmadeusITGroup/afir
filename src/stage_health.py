"""Deterministic per-stage health scoring from countable pipeline facts, never self-assessment.

Each stage starts at 1.0; fired signals subtract their weight (fatal = 1.0, so a broken stage
cannot be averaged up). Weights and thresholds are config (stage_gates.weights /
stages.<name>.threshold). Scoring never blocks; the gate that consumes it is separate.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Stages with scoreable outputs; plugins/export/output are excluded (side effects).
GATEABLE_STAGES = (
    "understanding",
    "query_generation",
    "log_retrieval",
    "correlation",
    "anomaly_detection",
    "report_generation",
)

# Weight 1.0: a broken stage scores 0.0 whatever healthy signals also hold.
FATAL = 1.0

# code -> weight subtracted from 1.0. Overridable via stage_gates.weights; 0.0 disables.
_DEFAULT_WEIGHTS: Dict[str, float] = {
    # understanding
    "no_entities": FATAL,
    "no_event_time": 0.3,
    "no_correlation_keys": 0.3,
    "no_sources_named": 0.2,
    # query_generation
    "no_queries": FATAL,
    "source_without_query": 0.2,
    "query_without_window": 0.2,
    # One alone gates; same weight, two codes: only _not_queried is repairable at the gate.
    "required_source_unavailable": 0.5,
    "required_source_not_queried": 0.5,
    # Light: the engine dropped the query; the plan is already correct.
    "selected_source_unscopable": 0.1,
    # Between the two above: not corrected but not declared needed; one doesn't gate, two do.
    "selected_source_unparseable": 0.35,
    # log_retrieval
    "no_rows_at_all": FATAL,
    "empty_sources": 0.4,  # pro-rata: scaled by the fraction of empty sources
    "source_timeout": 0.15,
    "source_failed": 0.15,
    # Lower than a timeout or a failure, and deliberately in the same tier as the other
    # "a configured limit bit, and every reader of the count is told so" signals
    # (detection_truncated_retried, narration_truncated_retried): the row cap is the
    # operator's own `max_results`, the truncation is reported everywhere the count is read
    # (evidence, brief, scope status, report), and a source at its cap still ANSWERED.
    # It stays a deduction because "N rows" and "N rows, and there were more" are different
    # findings — three capped sources gate; one or two are a limit, not a defect.
    "source_truncated": 0.1,
    # correlation
    "no_records": FATAL,
    "unresolved_join_keys": 0.4,
    # Gates alone, like required_source_not_queried: the procedure that adjudicates was the
    # pack default rather than a match, and every condition still resolved against real rows,
    # so the report reads confident either way. A human deciding which procedure applies is
    # the only repair, and it is one click (pin a use case).
    "procedure_not_selected": 0.5,
    # The winner barely beat a rival. Real information, but the match is genuine: must not gate
    # alone, or a domain whose procedures share vocabulary would pause on every run.
    "procedure_selection_thin": 0.15,
    # Aggregation-only: every source is still named; low because the verdict doesn't read the pack.
    "evidence_degraded": 0.1,
    # Content dropped rather than aggregated: "lossier" and "incomplete" are different findings.
    "evidence_clipped": 0.3,
    "verdict_degraded": 0.3,
    # Fires only where the brief degraded for a reason of its OWN (a projection leaf the
    # retrieval dropped, a scope sweep that could not widen). `brief.degraded` is a SUPERSET
    # of `verdict.degraded` by construction (usecases/base.py), so counting both charged one
    # degradation twice — see _signals_correlation.
    "brief_degraded": 0.3,
    # Narration ATTEMPTED and returned nothing. The volume gate's deterministic path is a
    # configured design choice, not a defect, and does not fire this (see _signals_correlation).
    "narration_skipped": 0.1,
    # anomaly_detection
    "detection_llm_failed": FATAL,
    # Retry recovered, so the list is complete; must not gate alone.
    "detection_truncated_retried": 0.1,
    "no_anomalies": 0.2,
    "all_below_threshold": 0.2,
    # report_generation
    "report_fallback_used": FATAL,
    "section_backfilled": 0.15,
    # Retry recovered, fully narrated: must not gate alone.
    "narration_truncated_retried": 0.1,
}

# Default when config names none: passes a single 0.3 signal, gates on two.
DEFAULT_THRESHOLD = 0.6

# Margin below which a procedure selection is reported as thin (see _selection_margin_floor).
# Measured over 80 distinct incident summaries from the local run history against a ten-spec
# pack: every one selected by score, none unmatched, and the two tightest wins sat at 0.130 and
# 0.149 — both correct, both between procedures that genuinely share vocabulary. 0.15 therefore
# fires on exactly those two: enough to exercise the check on real evidence, and at weight 0.15
# it cannot gate a run on its own.
_DEFAULT_SELECTION_MARGIN_FLOOR = 0.15

# Per-stage cap; repeated signals are already aggregated.
_MAX_REASONS = 12


@dataclass
class HealthReason:
    """One fired signal: what it was, what it cost, and why it fired."""

    code: str
    weight: float
    detail: str
    count: int = 1

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "weight": round(self.weight, 4),
            "detail": self.detail,
            "count": self.count,
        }


@dataclass
class StageHealth:
    """A stage's health: one score, the reasons behind it, and the gate verdict."""

    stage: str
    score: float
    threshold: float
    reasons: List[HealthReason] = field(default_factory=list)
    gate_recommended: bool = False
    scored: bool = True

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "threshold": round(self.threshold, 4),
            "gate_recommended": self.gate_recommended,
            "scored": self.scored,
            "reasons": [r.to_dict() for r in self.reasons],
        }

    @property
    def reason_codes(self) -> List[str]:
        return [r.code for r in self.reasons]


# (code, detail, multiplier), where the multiplier scales the configured weight: N for "N
# sources timed out", aggregated into one reason, or a fraction for pro-rata signals such as
# the share of empty sources.
Signal = Tuple[str, str, float]


def _is_list(value) -> bool:
    """A real list, not a MagicMock attribute (every Mock attribute is truthy and iterable-looking)."""
    return isinstance(value, list)


def _nonempty_list(value) -> bool:
    return _is_list(value) and len(value) > 0


def _as_float(value, default=0.0) -> float:
    """A real float, or ``default``. Same MagicMock discipline as ``_is_list``; a TypeError here
    costs the gate, not just a message."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _signals_understanding(output, ctx) -> List[Signal]:
    analysis = getattr(output, "analysis", None)
    if analysis is None:
        return [("no_entities", "No analysis object produced", 1.0)]
    out: List[Signal] = []

    entities = getattr(analysis, "extracted_entities", None)
    if not _nonempty_list(entities):
        # Nothing to filter any query by, so every downstream retrieval is a blind
        # time-window scan.
        out.append(("no_entities", "No entities extracted from the incident", 1.0))

    event_time = getattr(analysis, "event_time", None)
    start = getattr(event_time, "start", None) if event_time is not None else None
    if not start:
        out.append(
            (
                "no_event_time",
                "No event window extracted; queries fall back to the ingestion "
                "timestamp, which is wrong whenever the event predates submission",
                1.0,
            )
        )

    if not _nonempty_list(getattr(analysis, "correlation_keys", None)):
        out.append(
            (
                "no_correlation_keys",
                "No correlation keys proposed; correlation has nothing to join on",
                1.0,
            )
        )

    if not _nonempty_list(getattr(analysis, "log_sources_to_review", None)):
        out.append(("no_sources_named", "No log sources named for review", 1.0))
    return out


def _canon(text: str) -> str:
    """Lowercase alphanumerics only, so ``order_data_lake`` == "Order Data Lake"."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _names_a_source(wanted: str, known) -> Optional[str]:
    """Which known source id this free-text entry names, or None (not "missing").

    log_sources_to_review is LLM prose; target_log_source is a canonical pack id. The safe
    reading is: identify the id named, know nothing about entries that name none.
    """
    hay = _canon(wanted)
    if not hay:
        return None
    best = None
    for name in known:
        needle = _canon(name)
        # A 1-3 char id would substring-match almost any sentence.
        if len(needle) >= 4 and needle in hay:
            # Prefer the longest match: `order_data_lake` over a hypothetical `order`.
            if best is None or len(needle) > len(_canon(best)):
                best = name
    return best


def _looks_like_source_id(text: str) -> bool:
    """A bare canonical id (``auth_events``) rather than a sentence about one."""
    t = text.strip()
    return bool(t) and " " not in t and len(t) <= 64


def _declinable(name: str, ctx, targeted) -> bool:
    """True when NOT querying this source is a decision the pack invited, not a defect.

    Two carve-outs: a legacy/fallback source whose replacement was queried; a source with
    selection_guidance (a stated skip is the mechanism working). Both reads are strict in type:
    a loose read on a MagicMock would make every source declinable.
    """
    try:
        pack = (ctx.modules or {}).get("knowledge_pack") if ctx else None
        if pack is None:
            engine = (ctx.modules or {}).get("log_retrieval") if ctx else None
            pack = getattr(engine, "knowledge_pack", None)
        src = pack.source(name) if pack is not None else None
        if src is None:
            return False
        # Only a dict with non-empty list values counts as a declaration.
        guidance = getattr(src, "selection_guidance", None)
        if isinstance(guidance, dict) and any(
            isinstance(v, (list, tuple)) and v for v in guidance.values()
        ):
            return True
        status = getattr(src, "status", "")
        status = status.lower() if isinstance(status, str) else ""
        if status in ("legacy", "fallback", "deprecated", "historical"):
            # Only excused when something current actually ran, or "legacy" would excuse
            # retrieving nothing at all for that question.
            return bool(targeted)
        return False
    except Exception:  # noqa: BLE001 — scoring must never break a run
        return False


def _known_source_names(ctx) -> set:
    """Source ids the engine can serve (same set main() hands the generator). Never raises."""
    try:
        engine = (ctx.modules or {}).get("log_retrieval") if ctx else None
        names = set(getattr(engine, "retrievers", {}) or {})
        return {n for n in names if isinstance(n, str)}
    except Exception:  # noqa: BLE001 — scoring must never break a run
        return set()


def _signals_query_generation(output, ctx) -> List[Signal]:
    queries = output if _is_list(output) else []
    if not queries:
        return [("no_queries", "No retrieval queries generated", 1.0)]
    out: List[Signal] = []

    # A requested source with no query reads later as "that source had nothing".
    understanding = (ctx.outputs or {}).get("understanding") if ctx else None
    analysis = getattr(understanding, "analysis", None)
    wanted = getattr(analysis, "log_sources_to_review", None)
    if _nonempty_list(wanted):
        targeted = {
            getattr(q, "target_log_source", "")
            for q in queries
            if getattr(q, "target_log_source", "")
        }
        # Against the serveable catalog, not just targeted: partial provisioning must not fire.
        catalog = _known_source_names(ctx)
        known = catalog or targeted
        missing = []
        for entry in wanted:
            if not entry:
                continue
            text = str(entry)
            named = _names_a_source(text, known)
            if named is None and _looks_like_source_id(text):
                # Without a catalog, take bare ids at their word; with one, absent = not serveable.
                if catalog:
                    continue
                named = text
            # Unrecognised prose; scoring it measures phrasing, not retrieval decisions.
            if named is None:
                continue
            if named not in targeted:
                missing.append(named)
        missing = sorted(set(missing))
        missing = [n for n in missing if not _declinable(n, ctx, targeted)]
        if missing:
            out.append(
                (
                    "source_without_query",
                    f"{len(missing)} requested source(s) have no query: "
                    + ", ".join(missing[:5]),
                    float(len(missing)),
                )
            )

    # Undeliverable declared dependency: conditions reading it go unknown while every stage
    # reports success.
    for name in _undeliverable_required(ctx):
        out.append(
            (
                "required_source_unavailable",
                f"'{name}' is declared a hard dependency by a validation ruleset but "
                "built no retriever on this run (missing backend credentials or an "
                "unsupported endpoint kind) — the conditions reading it cannot be "
                "evaluated",
                1.0,
            )
        )

    # Declared, retrievable, not queried: reported beside the plan (one-click repair) and never
    # injected — injecting answered a reasoning defect by bypassing the reasoning.
    for name in _declared_not_queried(ctx):
        out.append(
            (
                "required_source_not_queried",
                f"'{name}' is declared a hard dependency by the adjudicating validation "
                "ruleset, can be retrieved, and no query targets it — the conditions "
                "reading it will be `unknown`. Add a query from the run controls, or fix "
                "what the catalog tells the planner about this source",
                1.0,
            )
        )

    # Engine dropped it (nothing to scope on); plan is already correct, so light and non-gating.
    unscopable = _selected_unscopable(ctx)
    if unscopable:
        out.append(
            (
                "selected_source_unscopable",
                f"{len(unscopable)} selected source(s) filter on none of the entity types "
                "this incident names, so those queries were dropped instead of being "
                "issued as window-only scans: "
                + ", ".join(unscopable[:5])
                + ". The plan is correct as it stands; what misled the planner is the "
                "catalog entry (selection_guidance / not_answered_by, or a missing "
                "entities: type). Add the query from the run controls to ask anyway.",
                float(len(unscopable)),
            )
        )

    # Heavier than unscopable (nothing corrected), lighter than declared (not proven needed).
    unparseable = _selected_unparseable(ctx)
    if unparseable:
        out.append(
            (
                "selected_source_unparseable",
                f"{len(unparseable)} source(s) the planner DID select produced no query — "
                "the call could not be completed from the fields the engine owns: "
                + ", ".join(unparseable[:5])
                + ". Nothing in the plan says so otherwise, which makes this read exactly "
                "like a source that was never chosen. Add the query from the run controls",
                float(len(unparseable)),
            )
        )

    undated = [
        q
        for q in queries
        if not getattr(q, "date_from", None) or not getattr(q, "date_to", None)
    ]
    if undated:
        out.append(
            (
                "query_without_window",
                f"{len(undated)} quer(y|ies) carry no date window — an unbounded "
                "partition scan reads as slowness, then as an empty source",
                float(len(undated)),
            )
        )
    return out


def _signals_log_retrieval(output, ctx) -> List[Signal]:
    logs = output if isinstance(output, dict) else {}
    total_rows = sum(len(v or []) for v in logs.values())
    facts = _stage_facts(ctx, "log_retrieval")
    source_outcomes = facts.get("sources") or {}

    if total_rows == 0:
        return [
            (
                "no_rows_at_all",
                "Every source returned zero rows; nothing downstream can be decided "
                "on this data",
                1.0,
            )
        ]

    out: List[Signal] = []
    empty = [name for name, rows in logs.items() if not (rows or [])]
    if empty and logs:
        # Pro-rata, weighted by zero_rows.health_weight; undeclared sources count in full.
        specs = _zero_row_weights(ctx)
        weighted = sum(specs.get(name, {}).get("weight", 1.0) for name in empty)
        fraction = weighted / len(logs)
        # A source whose emptiness IS its answer (weight 0) is not part of this finding, so it
        # must not be listed inside it either: the arithmetic already excluded it, and naming
        # it first reads as the penalty whatever the number says.
        answered = [n for n in empty if specs.get(n, {}).get("weight", 1.0) <= 0]
        counted = [n for n in empty if n not in set(answered)]
        discounted = [n for n in counted if specs.get(n, {}).get("weight", 1.0) < 1.0]
        detail = (
            f"{len(counted)} of {len(logs)} source(s) returned zero rows and the pack "
            "declares no meaning for that: " + ", ".join(str(e) for e in counted[:5])
        )
        if discounted:
            # Name the discount: a silently smaller number the operator cannot check.
            detail += " — discounted as partly expected: " + "; ".join(
                _named_meaning(n, specs) for n in discounted[:5]
            )
        if answered:
            detail += (
                f"; {len(answered)} further source(s) ANSWERED by being empty and are not "
                "counted here: "
                + "; ".join(_named_meaning(n, specs) for n in answered[:5])
            )
        if fraction > 0:
            out.append(("empty_sources", detail, fraction))
        else:
            # Every empty source answered by being empty: report it, penalise nothing.
            logger.info(
                "Log retrieval: %d source(s) answered by being empty; nothing scored: %s",
                len(answered),
                "; ".join(_named_meaning(n, specs) for n in answered[:5]),
            )

    timed_out = [n for n, o in source_outcomes.items() if o.get("status") == "timeout"]
    if timed_out:
        out.append(
            (
                "source_timeout",
                f"{len(timed_out)} source(s) timed out: "
                + ", ".join(str(t) for t in timed_out[:5]),
                float(len(timed_out)),
            )
        )
    failed = [n for n, o in source_outcomes.items() if o.get("status") == "failed"]
    if failed:
        out.append(
            (
                "source_failed",
                f"{len(failed)} source(s) failed: "
                + ", ".join(str(f) for f in failed[:5]),
                float(len(failed)),
            )
        )

    # At-cap means truncated; indistinguishable from completeness by count alone.
    caps = _row_caps(ctx)
    truncated = [
        name
        for name, rows in logs.items()
        if caps.get(name) is not None and len(rows or []) == caps[name]
    ]
    if truncated:
        out.append(
            (
                "source_truncated",
                f"{len(truncated)} source(s) returned exactly their row cap and are "
                "therefore truncated, not complete: "
                + ", ".join(str(t) for t in truncated[:5]),
                float(len(truncated)),
            )
        )
    return out


def _signals_correlation(output, ctx) -> List[Signal]:
    if output is None:
        # Correlation is optional wiring; an absent module is not a health problem.
        return []
    if getattr(output, "record_count", 0) == 0:
        return [("no_records", "Correlation produced zero records", 1.0)]

    out: List[Signal] = []
    aggregations = getattr(output, "aggregations", None)
    aggregations = aggregations if isinstance(aggregations, dict) else {}
    if not _nonempty_list(aggregations.get("resolved_correlation_keys")):
        out.append(
            (
                "unresolved_join_keys",
                "No correlation keys resolved; sources were not actually joined",
                1.0,
            )
        )

    evidence = getattr(output, "evidence", None)
    if evidence is not None and getattr(evidence, "degraded", False) is True:
        # Which rung ran is the finding: rungs 1-5 aggregate (all sources named), rung 6 cuts
        # text (a source at the tail is absent). Hence two deductions.
        #
        # Imported here: config_store imports this module while loading config, and evidence
        # pulls in correlation — a module-level import would put a large file behind YAML reads.
        from evidence import CLIP_NOTE  # noqa: WPS433 — see above

        notes = getattr(evidence, "notes", None)
        clipped = CLIP_NOTE in (notes if isinstance(notes, list) else [])
        if clipped:
            out.append(
                (
                    "evidence_clipped",
                    "Evidence exhausted all five degradation rungs and the renderer is "
                    "thinning to fit: per-source detail (distributions, example rows) and "
                    "chronology/attribution lines are dropped from what the downstream LLM "
                    "stages read. Every source is still named with its row count, and if "
                    "even those did not fit the render says how many went — raise "
                    "correlation.evidence_char_budget",
                    1.0,
                )
            )
        else:
            out.append(
                (
                    "evidence_degraded",
                    "Evidence pack was aggregated harder to fit the char budget; the "
                    "downstream LLM stages saw a lossier but complete view",
                    1.0,
                )
            )
    verdict = getattr(output, "verdict", None)
    verdict_degraded = (
        verdict is not None and getattr(verdict, "degraded", False) is True
    )
    if verdict_degraded:
        out.append(
            ("verdict_degraded", "Validation verdict is degraded (missing data)", 1.0)
        )
    brief = getattr(output, "brief", None)
    # ONE degradation, charged once. `brief.degraded` starts as `bool(verdict.degraded)` and
    # is then OR-ed with the brief's own two causes (usecases/base.py), so it can only be a
    # superset: firing both codes deducted 0.6 for a single missing-data fact and, stacked
    # with an evidence rung and a skipped narration, summed to exactly 1.0 — a stage scored
    # 0.00 while the verdict, the conditions and every source were intact.
    if brief is not None and getattr(brief, "degraded", False) is True:
        if verdict_degraded:
            logger.info(
                "Correlation: the brief is degraded because the verdict is; scored once."
            )
        else:
            out.append(
                (
                    "brief_degraded",
                    "Investigation brief is degraded for a reason of its own (a decisive "
                    "check's input was dropped by retrieval, or the scope sweep could not "
                    "widen) while the verdict itself is not",
                    1.0,
                )
            )

    if not _nonempty_list(getattr(output, "findings", None)):
        # Empty findings has two causes and only one is a defect. The volume gate choosing
        # the deterministic path is a CONFIGURED choice (correlation.max_records_for_llm),
        # `summary_text` still flows downstream and no verdict reads `findings` — so it is
        # logged, not charged. Narration that ran and came back empty is the defect.
        mode = _narration_mode(ctx)
        if mode == "deterministic":
            logger.info(
                "Correlation: no narrated findings; the volume gate ran the deterministic "
                "path, which is a configured choice and not scored."
            )
        else:
            out.append(
                (
                    "narration_skipped",
                    "No narrated findings: the narration call ran and produced none, so "
                    "the downstream stages read the deterministic summary only"
                    if mode == "llm"
                    else "No narrated findings, and this run did not record which path "
                    "correlation took",
                    1.0,
                )
            )
    out.extend(_selection_signals(ctx))
    return out


def _selection_signals(ctx) -> List[Signal]:
    """How the adjudicating procedure was chosen, where that is itself a finding.

    Two codes, and the difference between them is what a human can do about it. ``no_match``
    means the ruleset that ran was the pack's default rather than a match, and nothing
    downstream can tell: the conditions resolved against real rows and the report reads
    confident. ``thin`` means a real match that a rival nearly took.

    Deliberately silent for ``sole_spec`` and ``no_specs``. A pack with one procedure, or none,
    has nothing to have chosen wrongly, and scoring it would fire on every run of every
    single-procedure deployment.
    """
    selection = _procedure_selection(ctx)
    basis = str(selection.get("basis") or "")
    if basis not in ("no_match", "scored"):
        return []
    if basis == "no_match":
        rivals = ", ".join(
            str(name) for name, _score in (selection.get("candidates") or []) if name
        )
        return [
            (
                "procedure_not_selected",
                "No procedure matched this incident, so it was adjudicated under the pack's "
                "DEFAULT ruleset rather than a recognised one. Every condition still "
                "evaluated against real rows, so the verdict reads as confident whether or "
                "not the procedure fits — which is why this is a finding and not a warning. "
                "Repair is to pin the right procedure on the incident, or to give the "
                "procedure that should have matched a title that discriminates. Candidates, "
                f"all scoring zero: {rivals or 'none declared'}",
                1.0,
            )
        ]
    floor = _selection_margin_floor(getattr(ctx, "config", None))
    margin = _as_float(selection.get("margin"), 1.0)
    if floor <= 0 or margin >= floor:
        return []
    candidates = selection.get("candidates") or []
    named = ", ".join(f"{n} ({s})" for n, s in candidates[:2] if n)
    return [
        (
            "procedure_selection_thin",
            f"The procedure was selected by a margin of {margin:.0%} over the runner-up "
            f"(floor {floor:.0%}), so two procedures read this incident almost equally well "
            f"and the loser's conditions would also have resolved against real rows. The "
            f"selection may well be right; it is worth a look because getting it wrong "
            f"produces a confident verdict under the other procedure's labels. Top two: "
            f"{named or 'unavailable'}",
            1.0,
        )
    ]


def _signals_anomaly_detection(output, ctx) -> List[Signal]:
    module = (ctx.modules or {}).get("anomaly_detection") if ctx else None
    # detect() returns [] for both failure and clean; only the module knows which.
    if module is not None and getattr(module, "last_degraded", False) is True:
        detail = getattr(module, "last_error", "") or "LLM call did not complete"
        return [
            (
                "detection_llm_failed",
                f"Anomaly detection degraded to zero anomalies: {detail}",
                1.0,
            )
        ]

    anomalies = output if _is_list(output) else []
    out: List[Signal] = []
    if module is not None and getattr(module, "last_truncation_retried", False) is True:
        out.append(
            (
                "detection_truncated_retried",
                "The anomaly list hit its token budget and had to be retried at a "
                "wider one — raise anomaly_detection.detect_max_tokens for this "
                "data volume",
                1.0,
            )
        )
    if not anomalies:
        out.append(
            (
                "no_anomalies",
                "No anomalies detected (the LLM call succeeded — this may be a "
                "genuinely clean incident)",
                1.0,
            )
        )
        return out

    # Ask the module for the effective threshold (clamped runs differ from config). Items are
    # marked not dropped. `is True` guards against MagicMock truthy attributes.
    narrated = [a for a in anomalies if getattr(a, "below_threshold", False) is not True]
    if module is not None and not narrated:
        threshold = _as_float(getattr(module, "last_threshold_used", 0.0))
        top = _as_float(getattr(module, "last_filtered_max", 0.0))
        out.append(
            (
                "all_below_threshold",
                f"All {len(anomalies)} anomalies scored below the effective confidence "
                f"threshold {threshold:.2f} (highest {top:.2f}), so none are narrated — "
                "they are retained in the exports and the anomalies table",
                1.0,
            )
        )
    return out


def _signals_report_generation(output, ctx) -> List[Signal]:
    module = (ctx.modules or {}).get("report_generation") if ctx else None
    if module is None:
        return []
    if getattr(module, "last_fallback_used", False) is True:
        detail = getattr(module, "last_error", "") or "narration call failed"
        return [
            (
                "report_fallback_used",
                f"Report was assembled deterministically, not narrated: {detail}",
                1.0,
            )
        ]
    out: List[Signal] = []
    backfilled = getattr(module, "last_backfilled_sections", None)
    if _nonempty_list(backfilled):
        out.append(
            (
                "section_backfilled",
                f"{len(backfilled)} section(s) missing from the narration were "
                "backfilled deterministically: "
                + ", ".join(str(b) for b in backfilled[:5]),
                float(len(backfilled)),
            )
        )
    if getattr(module, "last_truncation_retried", False) is True:
        out.append(
            (
                "narration_truncated_retried",
                "The narration hit its token budget and had to be retried at a wider "
                "one — raise report_generation.report_max_tokens for this data volume",
                1.0,
            )
        )
    return out


_SIGNAL_FNS = {
    "understanding": _signals_understanding,
    "query_generation": _signals_query_generation,
    "log_retrieval": _signals_log_retrieval,
    "correlation": _signals_correlation,
    "anomaly_detection": _signals_anomaly_detection,
    "report_generation": _signals_report_generation,
}


# --- helpers reading facts off the run ------------------------------------------


def _stage_facts(ctx, stage_name) -> dict:
    """Facts a stage recorded about itself on the context (never raises)."""
    try:
        facts = getattr(ctx, "stage_facts", None)
        if isinstance(facts, dict):
            value = facts.get(stage_name)
            return value if isinstance(value, dict) else {}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _procedure_selection(ctx) -> dict:
    """How this run's adjudicating procedure was chosen, as ``SelectionBasis.to_dict()``.

    Run-recorded value wins for the same reason ``_generator_names`` prefers one: the
    correlation module is shared across jobs and its attribute holds the last run's answer.
    ``{}`` when nothing recorded, which reads as "not a defect" — a pack with no correlation
    specs at all is a valid configuration, not an unselected incident.
    """
    try:
        recorded = _stage_facts(ctx, "correlation").get("procedure_selection")
        if isinstance(recorded, dict):
            return recorded
        module = (ctx.modules or {}).get("correlation") if ctx else None
        basis = getattr(module, "last_selection", None)
        as_dict = basis.to_dict() if hasattr(basis, "to_dict") else None
        return as_dict if isinstance(as_dict, dict) else {}
    except Exception:  # noqa: BLE001 — a scoring read must never fail the stage
        return {}


def _narration_mode(ctx) -> str:
    """``"llm"``, ``"deterministic"``, or ``""`` when the run recorded neither.

    Run-recorded value wins over the module attribute for the same reason
    ``_procedure_selection`` prefers it: the correlation module is shared across jobs.
    An unknown mode is scored as before — a defect the run cannot attribute is still a
    defect, and only an explicit "deterministic" is the designed path.
    """
    try:
        recorded = _stage_facts(ctx, "correlation").get("narration")
        if isinstance(recorded, str) and recorded:
            return recorded
        module = (ctx.modules or {}).get("correlation") if ctx else None
        mode = getattr(module, "last_narration", None)
        return mode if isinstance(mode, str) else ""
    except Exception:  # noqa: BLE001 — a scoring read must never fail the stage
        return ""


def _selection_margin_floor(config) -> float:
    """Margin below which a scored win is reported as thin. 0 disables the check.

    Configured rather than fixed because how much vocabulary two procedures share is a
    property of the domain, not of the engine.
    """
    try:
        raw = ((config or {}).get("correlation") or {}).get("selection_margin_floor")
        if raw is None:
            return _DEFAULT_SELECTION_MARGIN_FLOOR
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        logger.warning(
            "correlation.selection_margin_floor is not a number; using the default %s",
            _DEFAULT_SELECTION_MARGIN_FLOOR,
        )
        return _DEFAULT_SELECTION_MARGIN_FLOOR


def _zero_row_weights(ctx) -> Dict[str, Dict[str, Any]]:
    """{source: {"weight": float, "meaning": str}} from SourceDef.zero_rows, or {}.
    Never raises; a missing pack counts every empty source in full (conservative)."""
    out: Dict[str, Dict[str, Any]] = {}
    try:
        engine = (ctx.modules or {}).get("log_retrieval") if ctx else None
        pack = getattr(engine, "knowledge_pack", None)
        for src in getattr(pack, "sources", []) or []:
            spec = getattr(src, "zero_rows", None)
            if not isinstance(spec, dict) or not spec:
                continue
            raw = spec.get("health_weight", 1.0)
            try:
                weight = min(max(float(raw), 0.0), 1.0)
            except (TypeError, ValueError):
                logger.warning(
                    "Source '%s': zero_rows.health_weight %r is not a number; counting "
                    "an empty result in full.",
                    src.name,
                    raw,
                )
                continue
            out[src.name] = {
                "weight": weight,
                "meaning": str(spec.get("meaning") or "").strip(),
            }
    except Exception:  # noqa: BLE001 — scoring must never break a run
        return {}
    return out


def _named_meaning(name, specs: Dict[str, Dict[str, Any]]) -> str:
    """``source (what empty means there)``, or the bare name when the pack declared none."""
    meaning = str((specs.get(name) or {}).get("meaning") or "").strip()
    return f"{name} ({meaning})" if meaning else str(name)


def _generator_names(ctx, attr) -> List[str]:
    """Named dependency findings from the query generator, as a sorted list.

    Run-recorded value wins (the generator is shared across jobs; its attribute holds the last
    plan and would misattribute findings). Falls back to the module attribute for callers that
    drive stages directly. Strict list-of-str: a MagicMock attribute is truthy.
    """
    try:
        recorded = _stage_facts(ctx, "query_generation")
        names = recorded.get(attr) if attr in recorded else None
        if not isinstance(names, list):
            gen = (ctx.modules or {}).get("api_call")
            names = getattr(gen, attr, None)
        if not isinstance(names, list):
            return []
        return sorted({n for n in names if isinstance(n, str) and n})
    except Exception:  # noqa: BLE001 — a scoring read must never fail the stage
        return []


def _undeliverable_required(ctx) -> List[str]:
    """Ruleset-declared sources that built no retriever, from the query generator."""
    return _generator_names(ctx, "undeliverable_required")


def _declared_not_queried(ctx) -> List[str]:
    """Ruleset-declared, retrievable sources the planner did not select."""
    return _generator_names(ctx, "declared_not_queried")


def _selected_unscopable(ctx) -> List[str]:
    """Sources the planner selected that nothing could have scoped to this incident."""
    return _generator_names(ctx, "selected_unscopable")


def _selected_unparseable(ctx) -> List[str]:
    """Sources the planner selected whose tool call could not become a query at all."""
    return _generator_names(ctx, "selected_unparseable")


def _row_caps(ctx) -> Dict[str, Optional[int]]:
    """``{source: max_results}`` from the retrieval engine, or ``{}``."""
    try:
        engine = (ctx.modules or {}).get("log_retrieval")
        caps = engine.row_caps() if engine is not None else None
        return caps if isinstance(caps, dict) else {}
    except Exception:  # noqa: BLE001 — a missing cap is not a scoring failure
        return {}


def _weights(config) -> Dict[str, float]:
    """Configured weights merged over the defaults."""
    merged = dict(_DEFAULT_WEIGHTS)
    try:
        overrides = ((config or {}).get("stage_gates") or {}).get("weights") or {}
        for code, value in overrides.items():
            if code in merged:
                merged[code] = float(value)
            else:
                logger.warning("Ignoring unknown stage-health weight '%s'", code)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read stage_gates.weights (%s); using defaults", exc)
    return merged


def stage_threshold(config, stage_name) -> float:
    """The gate threshold for one stage: per-stage override, else global, else default."""
    gates = (config or {}).get("stage_gates") or {}
    per_stage = (gates.get("stages") or {}).get(stage_name) or {}
    for candidate in (per_stage.get("threshold"), gates.get("threshold")):
        if candidate is not None:
            try:
                return float(candidate)
            except (TypeError, ValueError):
                logger.warning(
                    "Non-numeric gate threshold for %s; using default", stage_name
                )
    return DEFAULT_THRESHOLD


def stage_gate_enabled(config, stage_name) -> bool:
    """Whether this stage may open a gate at all. Non-gateable stages never can."""
    if stage_name not in GATEABLE_STAGES:
        return False
    gates = (config or {}).get("stage_gates") or {}
    per_stage = (gates.get("stages") or {}).get(stage_name) or {}
    enabled = per_stage.get("enabled")
    return True if enabled is None else bool(enabled)


def stage_gate_timeout(config, stage_name) -> Optional[float]:
    """Seconds before on_timeout applies; None holds forever (default).

    0 and negatives read as no timeout: taking them literally would silently un-gate a
    supervised run.
    """
    gates = (config or {}).get("stage_gates") or {}
    per_stage = (gates.get("stages") or {}).get(stage_name) or {}
    for candidate in (per_stage.get("timeout_seconds"), gates.get("timeout_seconds")):
        if candidate is None:
            continue
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            logger.warning(
                "Non-numeric gate timeout_seconds for %s; ignoring", stage_name
            )
            continue
        return value if value > 0 else None
    return None


# `hold` keeps waiting; `proceed` continues as approved; `abort` cancels. `hold` is default:
# the only one that cannot make an unreviewed decision.
GATE_TIMEOUT_ACTIONS = ("hold", "proceed", "abort")
DEFAULT_ON_TIMEOUT = "hold"


def stage_gate_on_timeout(config, stage_name) -> str:
    """What to do when this stage's gate times out. One of ``GATE_TIMEOUT_ACTIONS``."""
    gates = (config or {}).get("stage_gates") or {}
    per_stage = (gates.get("stages") or {}).get(stage_name) or {}
    for candidate in (per_stage.get("on_timeout"), gates.get("on_timeout")):
        if candidate is None:
            continue
        value = str(candidate).strip().lower()
        if value in GATE_TIMEOUT_ACTIONS:
            return value
        logger.warning(
            "Unknown stage_gates on_timeout '%s' for %s; holding instead",
            candidate,
            stage_name,
        )
        return DEFAULT_ON_TIMEOUT
    return DEFAULT_ON_TIMEOUT


# --- the entry point -----------------------------------------------------------


def score_stage(stage_name: str, output: Any, ctx=None, config=None) -> StageHealth:
    """Score one stage. Returns scored=False for unscored stages (plugins/export/output) rather
    than a perfect 1.0. Never raises."""
    cfg = config if config is not None else getattr(ctx, "config", None) or {}
    threshold = stage_threshold(cfg, stage_name)
    fn = _SIGNAL_FNS.get(stage_name)
    if fn is None:
        return StageHealth(
            stage=stage_name,
            score=1.0,
            threshold=threshold,
            reasons=[],
            gate_recommended=False,
            scored=False,
        )

    try:
        signals = fn(output, ctx) or []
    except Exception as exc:  # noqa: BLE001 — scoring must never break a run
        logger.warning(
            "Stage health scoring failed for %s (%s); reporting unscored.",
            stage_name,
            exc,
        )
        return StageHealth(
            stage=stage_name,
            score=1.0,
            threshold=threshold,
            reasons=[],
            gate_recommended=False,
            scored=False,
        )

    weights = _weights(cfg)
    reasons: List[HealthReason] = []
    penalty = 0.0
    for code, detail, multiplier in signals:
        base = weights.get(code)
        if base is None:
            logger.debug("No weight configured for signal '%s'; skipping", code)
            continue
        try:
            mult = float(multiplier)
        except (TypeError, ValueError):
            mult = 1.0
        weight = base * mult
        if weight <= 0:
            continue  # a signal disabled by config (weight 0) fires nothing
        penalty += weight
        reasons.append(
            HealthReason(
                code=code,
                weight=weight,
                detail=detail,
                count=int(mult) if mult >= 1 and float(mult).is_integer() else 1,
            )
        )

    score = max(0.0, min(1.0, 1.0 - penalty))
    return StageHealth(
        stage=stage_name,
        score=score,
        threshold=threshold,
        reasons=reasons[:_MAX_REASONS],
        gate_recommended=score < threshold,
        scored=True,
    )
