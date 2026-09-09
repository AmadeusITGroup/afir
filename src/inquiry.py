"""The adjudicating procedure's own open questions: what this run could not settle.

The sibling of ``src/links.py`` on the other axis. A link asks *does ANOTHER procedure also
apply*; an inquiry asks *the evidence in hand leaves a question about THIS procedure open, and
one further source would say something about it*. Both are advisory and neither reaches the
verdict, the condition lines, the severity or the health score (asserted by
``tests/test_inquiries_never_change_the_verdict.py``).

Cannot fetch: no async, no IO, no clock, no randomness (AST-asserted). The fetching half is
``src/inquiry_probe.py``; rows arrive here as an argument to :func:`settle_with_probe` and go
into that function's own reading, never into ``logs``.

Five states (not collapsed): ``answered``, ``not_asked``, ``empty``, ``unanswered``,
``unreachable``. A question nobody asked, one asked and answered with nothing, and one whose
source never answered license three different next steps, and only the second is a finding.

A pack declaring no ``open_questions:`` produces zero findings and zero probes.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from correlation import apply_where
from models.pydantic_models import InquiryFinding

logger = logging.getLogger(__name__)

#: The five outcomes, ordered actionable first: the answered question, then the one still open,
#: then the two kinds of non-finding, then the one the pack cannot serve. A reader who stops
#: halfway has seen every question that has something to say.
INQUIRY_STATES: Tuple[str, ...] = (
    "answered",
    "not_asked",
    "empty",
    "unanswered",
    "unreachable",
)

#: Condition results a declaration may trigger on. ``unknown`` is the default and the reason the
#: lane exists: an open question is what a check that could not answer leaves behind. ``fail``
#: and ``pass`` are admitted because a decisive result can raise a question too.
INQUIRY_TRIGGERS: Tuple[str, ...] = ("unknown", "fail", "pass")

#: Scope values carried on one finding. The probe asks about all of them at once, so this bounds
#: the query text rather than the number of probes.
_MAX_SCOPE_VALUES = 20

_ADVISORY_PROVENANCE = (
    "added by the open-question assessment, addressed to a human: this run did not adjudicate it"
)


def assess_inquiries(
    pack: Any,
    analysis: Any,
    logs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    verdict: Any = None,
    brief: Any = None,
    ruleset_key: str = "",
    config: Optional[Dict[str, Any]] = None,
) -> List[InquiryFinding]:
    """Raise the open questions this run's own procedure declared, over its own results.

    Pure, deterministic, total: never raises; returns ``[]`` for a pack that declares none.

    A declaration whose trigger did not fire produces NO finding: a question that was not
    raised is not an open question, and listing every declaration on every run would make the
    lane a catalog rather than a finding.
    """
    cfg = config or {}
    if not bool(cfg.get("enabled", True)):
        return []
    if pack is None or not hasattr(pack, "open_questions"):
        return []
    try:
        declarations = pack.open_questions(ruleset_key) or []
        if not declarations:
            return []
        rows = logs if isinstance(logs, dict) else {}
        subject_entity = _subject_entity(pack, ruleset_key)
        findings: List[InquiryFinding] = []
        for declaration in declarations:
            finding = _raise_one(
                declaration, analysis, verdict, brief, subject_entity, rows, cfg
            )
            if finding is not None:
                findings.append(finding)
    except Exception as exc:  # noqa: BLE001 — advisory: never fail the stage
        # An advisory lane that can break the run it advises on is worse than no advisory lane.
        # Logged at error level because a silent empty list is indistinguishable from a pack
        # that declared nothing, and those two need opposite responses.
        logger.error(
            "The open-question assessment failed and is reported as EMPTY for this run (%s). "
            "This is an advisory channel: the verdict, the brief and the health score are "
            "unaffected.",
            exc,
            exc_info=True,
        )
        return []
    order = {state: i for i, state in enumerate(INQUIRY_STATES)}
    findings.sort(key=lambda f: (order.get(f.state, len(order)), f.id))
    logger.info(
        "Open questions raised by this procedure: %d — %s.",
        len(findings),
        ", ".join(f"{f.id}: {f.state}" for f in findings) or "none",
    )
    return findings


def _subject_entity(pack: Any, ruleset_key: str) -> str:
    """The ruleset's own subject entity, or '' when the pack cannot answer."""
    try:
        return str((pack.ruleset_spec(ruleset_key) or {}).get("subject_entity", "") or "")
    except Exception:  # noqa: BLE001 — a pack that cannot answer contributes nothing
        return ""


def _raise_one(
    declaration: Dict[str, Any],
    analysis: Any,
    verdict: Any,
    brief: Any,
    subject_entity: str,
    logs: Dict[str, List[Dict[str, Any]]],
    cfg: Dict[str, Any],
) -> Optional[InquiryFinding]:
    """One declaration, one trigger test, one :class:`InquiryFinding` — or ``None``."""
    triggered, subjects = _triggered(declaration, verdict)
    if not triggered:
        return None

    scope_entity = str(declaration.get("scope_entity", "") or "") or subject_entity
    values = _scope_values(
        subjects,
        scope_entity,
        analysis,
        brief if scope_entity and scope_entity == subject_entity else None,
    )
    finding = InquiryFinding(
        id=str(declaration.get("id", "") or ""),
        source=str(declaration.get("source", "") or ""),
        trigger=_trigger_text(declaration, subjects),
        trigger_condition=str(declaration.get("condition", "") or ""),
        trigger_result=str(declaration.get("result", "") or ""),
        scope_entity=scope_entity,
        scope_values=values,
        note=str(declaration.get("note", "") or ""),
        advisory_note=_ADVISORY_PROVENANCE,
    )
    finding.question = _question_text(declaration, scope_entity, values)

    if not scope_entity or not values:
        # An unscoped question is not a cheap question: it scans the source over the whole
        # window. Reported rather than asked, and reported rather than dropped — the remedy is
        # a `scope_entity` the run actually holds, which is a pack fix.
        finding.state = "unreachable"
        finding.gap_reason = (
            f"this question is scoped by '{scope_entity or '(nothing declared)'}' and this run "
            "holds no value of that type, so there is nothing to ask about: an unscoped probe "
            "would scan the source over the whole window instead of asking about one identity"
        )
        return finding

    # The free rung, and it is the cheap half of the whole lane: where this run already
    # retrieved the source the declaration names, the question is answered from those rows and
    # no probe is licensed at all. Those rows were fetched to answer the procedure's OWN
    # conditions, which is stated in the note — a bounded re-use of evidence in hand is not the
    # same claim as a query asked for this question, and the reader is owed the difference.
    if finding.source in (logs or {}):
        settle_from_rows_in_hand(
            finding,
            declaration,
            (logs or {}).get(finding.source) or [],
            note=(
                f"no probe was spent: this run had already retrieved '{finding.source}' for its "
                "own conditions, and this question is answered by reading those rows again — a "
                "bounded re-use of evidence in hand rather than a query asked for this question"
            ),
        )
        return finding

    finding.state = "not_asked"
    finding.gap_reason = (
        f"settling this needs one query of '{finding.source}', which this run has not spent"
        if _budgeted(cfg)
        else (
            f"settling this needs one query of '{finding.source}', and no probe budget is "
            "configured for open questions, so nothing was spent"
        )
    )
    return finding


def _budgeted(cfg: Dict[str, Any]) -> bool:
    """True when the lane has a probe count to spend.

    Raw key rather than :func:`src.inquiry_probe.inquiry_budget`: that module imports this one,
    and the sentence above only needs to know whether the count is armed.
    """
    try:
        return int(cfg.get("max_inquiry_probes_per_run", 1)) > 0
    except (TypeError, ValueError):
        return False


# --- the trigger ------------------------------------------------------------------------


def _triggered(declaration: Dict[str, Any], verdict: Any) -> Tuple[bool, List[Any]]:
    """Did this declaration's trigger fire, and on which subjects?

    Both halves declared means BOTH must hold on the same subject: a declaration naming a
    condition AND a verdict class is asking about one situation, not two.

    A condition id no subject carries does not fire. That is silent here on purpose — an id
    that matches nothing is a pack defect ``pack_validate`` reports against the declaration,
    and guessing at the nearest condition would raise a question about a check nobody wrote.
    """
    condition = str(declaration.get("condition", "") or "")
    want = str(declaration.get("result", "") or "").strip().lower()
    want_class = str(declaration.get("verdict_class", "") or "").strip()
    hits: List[Any] = []
    for subject in getattr(verdict, "subjects", None) or []:
        if want_class and str(getattr(subject, "verdict_class", "") or "") != want_class:
            continue
        if not condition:
            hits.append(subject)
            continue
        for check in getattr(subject, "checks", None) or []:
            if str(getattr(check, "id", "")) != condition:
                continue
            if str(getattr(check, "result", "") or "").strip().lower() == want:
                hits.append(subject)
                break
    return bool(hits), hits


def _trigger_text(declaration: Dict[str, Any], subjects: Sequence[Any]) -> str:
    """Why the question was raised, in words a report line can print unchanged."""
    condition = str(declaration.get("condition", "") or "")
    result = str(declaration.get("result", "") or "")
    want_class = str(declaration.get("verdict_class", "") or "")
    parts: List[str] = []
    if condition:
        parts.append(f"the check '{condition}' read {result or 'a triggering result'}")
    if want_class:
        parts.append(f"this run's verdict class is '{want_class}'")
    where = f" on {len(subjects)} subject(s)" if subjects else ""
    return " and ".join(parts) + where if parts else f"the declaration fired{where}"


# --- the scope --------------------------------------------------------------------------


def _scope_values(
    subjects: Sequence[Any], scope_entity: str, analysis: Any, brief: Any = None
) -> List[str]:
    """Values of ``scope_entity`` this run holds, subjects first then the incident's entities.

    Subjects first because they are what the trigger fired on: the question is about them. The
    scope sweep's own subjects come next where the question is scoped by the ruleset's subject
    entity — those are the values the alert never named, so a question about them is the one a
    run of one alert cannot otherwise reach. The incident's entities are the last fallback, and
    the only one for a declaration scoped by a type the verdict does not adjudicate.

    ``brief`` is passed as ``None`` by the caller wherever the scope type is NOT the ruleset's
    subject entity: the sweep's values are typed by that entity and nothing else, so reading them
    under another type name would assert a binding the sweep never made.
    """
    out: List[str] = []

    def put(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in out and len(out) < _MAX_SCOPE_VALUES:
            out.append(text)

    if not scope_entity:
        return out
    for subject in subjects:
        if str(getattr(subject, "subject_type", "") or "") == scope_entity:
            put(getattr(subject, "subject_value", ""))
    for value in getattr(brief, "additional_subjects", None) or []:
        put(value)
    for asset in getattr(brief, "impacted_assets", None) or []:
        put(getattr(asset, "subject", ""))
    if out:
        return out
    for ent in getattr(analysis, "extracted_entities", None) or []:
        etype = getattr(ent, "type", None)
        if etype is None and isinstance(ent, dict):
            etype = ent.get("type")
        if str(etype or "") != scope_entity:
            continue
        value = getattr(ent, "value", None)
        if value is None and isinstance(ent, dict):
            value = ent.get("value")
        put(value)
    return out


def _question_text(
    declaration: Dict[str, Any], scope_entity: str, values: Sequence[str]
) -> str:
    """The declaration's question with its two placeholders substituted.

    ``str.replace`` and never ``.format``: procedure prose contains braces, so formatting a
    pack's sentence raises on the first one it did not expect. An absent question is allowed —
    the source plus the scope is a real ask — and ``pack_validate`` says what it costs.
    """
    text = str(declaration.get("question", "") or "")
    if not text:
        return ""
    return text.replace("{value}", ", ".join(values)).replace("{entity}", scope_entity)


# --- the settlement half (the fetching half is `src/inquiry_probe.py`) -------------------


def settle_with_probe(
    finding: InquiryFinding,
    declaration: Dict[str, Any],
    rows: Optional[Sequence[Dict[str, Any]]] = None,
    source: str = "",
    row_cap: Optional[int] = None,
    note: str = "",
) -> None:
    """Read one PROBE's answer and settle the question to one state and one meaning.

    Rung-3 discipline, borrowed whole from the link lane: rows arrive as an argument (this
    module cannot fetch; AST-asserted) and are read HERE only — they are never merged into
    ``logs``, which the census, the co-identity merge and the health scorer all read.

    Three answers, three states: ``rows is None`` is a source that did not answer
    (``unanswered``); a list including ``[]`` is an answer (``empty``); anything the
    declaration's own row selector keeps is ``answered``. The pack's ``meaning`` for that
    outcome is stamped verbatim, because a count with no declared meaning is the failure this
    lane exists to avoid.

    Mutates ``finding`` in place. Never raises.
    """
    _settle(finding, declaration, rows, source, row_cap, note, spent=rows is not None)


def settle_from_rows_in_hand(
    finding: InquiryFinding,
    declaration: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    source: str = "",
    note: str = "",
) -> None:
    """Settle the question from rows this run ALREADY retrieved — the free rung.

    Same reading as :func:`settle_with_probe` and one difference that matters to a reader:
    ``probe_spent`` stays False, because nothing was spent. Separate entry rather than a flag,
    so the call site says which of the two happened and neither can be mistaken for the other.

    No row cap is passed: the cap that bounded those rows belongs to the run's own retrieval and
    is reported where the run reports its sources, not a second time here as though this reading
    had asked for them.
    """
    _settle(finding, declaration, list(rows or []), source, None, note, spent=False)


def _settle(
    finding: InquiryFinding,
    declaration: Dict[str, Any],
    rows: Optional[Sequence[Dict[str, Any]]],
    source: str,
    row_cap: Optional[int],
    note: str,
    spent: bool,
) -> None:
    """The one reading both settlement entries share. Mutates ``finding``; never raises."""
    meaning = declaration.get("meaning") if isinstance(declaration, dict) else {}
    meaning = meaning if isinstance(meaning, dict) else {}
    finding.probe_spent = bool(spent)
    if source:
        finding.source = source
    if note:
        finding.probe_note = note

    if rows is None:
        finding.state = "unanswered"
        finding.meaning = str(meaning.get("unanswered", "") or "")
        finding.gap_reason = (
            f"'{finding.source}' did not answer this question at all, which is a different fact "
            "from answering with no rows: the remedy is a credential or a catalog entry, not a "
            "reading of these rows"
        )
        return

    returned = [r for r in rows if isinstance(r, dict)]
    # Measured on what came BACK, not on what the selector kept: the cap bounds the retrieval,
    # so a selector that keeps two of a capped hundred still owes the reader "and there were more".
    if row_cap is not None and int(row_cap) > 0 and len(returned) >= int(row_cap):
        finding.row_cap_hit = True
    kept = returned
    selector = declaration.get("where") if isinstance(declaration, dict) else None
    if selector:
        try:
            kept = apply_where(kept, selector)
        except Exception as exc:  # noqa: BLE001 — a clause, not the assessment
            # The clause is the scope of the question; reading every row instead would answer a
            # different question confidently. So the spend is recorded and nothing is claimed.
            logger.warning(
                "An open question's row selector could not be applied (%s); the probe's rows "
                "are reported as unread rather than read against a question nobody asked.",
                exc,
            )
            finding.state = "unanswered"
            finding.meaning = str(meaning.get("unanswered", "") or "")
            finding.gap_reason = (
                f"rows from '{finding.source}' are in hand, and this declaration's own row "
                "selector could not be applied to them, so nothing is claimed about those rows"
            )
            return

    finding.rows_matched = len(kept)
    finding.gap_reason = ""
    if kept:
        finding.state = "answered"
        finding.meaning = str(meaning.get("rows", "") or "")
        return
    # An empty answer IS an answer, and it is the one this whole repo is organised around
    # labelling: the pack said what zero rows means here, so the reader gets that sentence and
    # not a zero.
    finding.state = "empty"
    finding.meaning = str(meaning.get("empty", "") or "")
