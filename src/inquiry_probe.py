"""The one rung of the open-question lane that spends a query.

``src/inquiry.py`` raises the questions and settles them; this module is the only half that
fetches. Rows reach the settlement solely via :func:`~src.inquiry.settle_with_probe`, never
merged into ``logs``.

Two bounds and one shared ceiling: the lane has its own count
(``max_inquiry_probes_per_run``, shipping at 1 — armed, because a bound shipped at 0 is an
untested bound), its own per-probe timeout and row cap, and it draws on the SAME clamped product
ceiling as the link lane's rung 3, so the two advisory lanes cannot jointly spend more seconds
than one of them could alone.

Every refusal is a coded row on the finding, never a silence: a question that was not asked and
a question that was asked and came back empty license opposite next steps.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from inquiry import settle_with_probe
from link_escalation import (
    MAX_PROBES_CEILING,
    PROBE_BUDGET_CEILING_SECONDS,
    PROBE_ROW_CAP_DEFAULT,
    PROBE_TIMEOUT_DEFAULT,
    PROBE_TIMEOUT_MAX,
    probe_budget,
    slice_text,
)

# One copy of the dual-import-safe rescoping, borrowed from the lane this one mirrors: a second
# copy would be a second answer to how a probe inherits the incident's window and scope.
from link_probe import _scoped_analysis

logger = logging.getLogger(__name__)

#: Probes one run may spend on its own open questions by default. Narrow but ARMED: shipped at 0
#: every refusal code below would be unreachable outside a test. ``<= 0`` closes the lane.
DEFAULT_MAX_INQUIRY_PROBES = 1

#: Why each refusal happened, in the words the report and the UI both print. A code with no
#: sentence would render as a blank line, which is the silence this table exists to prevent.
REFUSAL_NOTES: Dict[str, str] = {
    "no_source": (
        "no probe was spent: this question names no source the pack could resolve, so there is "
        "nothing to ask — the fix is the declaration's `ask.source`, not this run"
    ),
    "not_open": (
        "no probe was spent: this question is not open — it was already settled, at no retrieval "
        "cost, from rows this run had in hand"
    ),
    "already_asked": (
        "no probe was spent: this question already cost one retrieval in this run, and asking the "
        "same source the same question again would spend a second scan to reproduce the answer"
    ),
    "no_scope_value": (
        "no probe was spent: this run holds no value to scope the question by, so the query would "
        "scan the source over the whole window instead of asking about one identity"
    ),
    "budget_spent": (
        "no probe was spent: this run's open-question probe budget was already spent on the "
        "questions above, and this one keeps its unsettled state rather than overrunning a bound"
    ),
    "lane_closed": (
        "no probe was spent: the advisory probe ceiling this lane shares with the cross-procedure "
        "lane leaves it no seconds to spend on this run, so the question is reported unsettled "
        "rather than asked outside a bound"
    ),
}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def inquiry_budget(
    config: Optional[Dict[str, Any]] = None,
    link_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """``{"max_probes", "timeout", "deadline_seconds"}``: this lane's bounds, resolved and clamped.

    The count and the per-probe timeout are clamped to the link lane's own ceilings — one set of
    ceilings for both advisory lanes, so raising a bound raises it in one place. The collective
    room is then what the link lane's BUDGETED worst case leaves of
    ``PROBE_BUDGET_CEILING_SECONDS``.

    Budgeted rather than actually spent, deliberately: a bound whose value depends on how slow
    the other lane happened to be is not a bound anybody can state in advance, and the whole
    point of a ceiling is that it can be stated before the run.

    ``max_probes <= 0`` closes the lane, and so does a ceiling with no room left — reported as
    ``lane_closed`` on every question it declines, never as silence.
    """
    cfg = config or {}
    max_probes = max(
        0,
        min(
            MAX_PROBES_CEILING,
            _as_int(cfg.get("max_inquiry_probes_per_run"), DEFAULT_MAX_INQUIRY_PROBES),
        ),
    )
    timeout = _as_int(cfg.get("probe_timeout_seconds"), PROBE_TIMEOUT_DEFAULT)
    if timeout <= 0:
        timeout = PROBE_TIMEOUT_DEFAULT
    timeout = min(PROBE_TIMEOUT_MAX, timeout)
    room = max(0, PROBE_BUDGET_CEILING_SECONDS - probe_budget(link_config or {})["deadline_seconds"])
    if max_probes and max_probes * timeout > room:
        timeout = room // max_probes
        if timeout <= 0:
            max_probes = 0
    return {
        "max_probes": max_probes,
        "timeout": timeout,
        "deadline_seconds": max_probes * timeout,
    }


def inquiry_row_cap(config: Optional[Dict[str, Any]] = None) -> int:
    """Rows one probe may hand to the settlement, from config else the shared default."""
    cap = _as_int((config or {}).get("probe_row_cap"), PROBE_ROW_CAP_DEFAULT)
    return cap if cap > 0 else PROBE_ROW_CAP_DEFAULT


def inquiry_refusal(finding: Any, logs: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], str]:
    """The source this question would ask, and the refusal code if it cannot be asked.

    Returns ``(source, "")`` or ``(None, code)``. Budget is not checked here: it belongs to the
    run, not to the question.
    """
    source = str(getattr(finding, "source", "") or "")
    if not source:
        return None, "no_source"
    if str(getattr(finding, "state", "") or "") != "not_asked":
        return None, "not_open"
    if bool(getattr(finding, "probe_spent", False)):
        return None, "already_asked"
    if not list(getattr(finding, "scope_values", None) or []):
        return None, "no_scope_value"
    return source, ""


def _decline(finding: Any, code: str) -> None:
    """Record a refusal on the finding, in one place, with one sentence per code."""
    note = REFUSAL_NOTES.get(code, "")
    if not note:
        # An unknown code is itself reportable: a blank note reads as a question nobody declined.
        note = (
            f"no probe was spent, and the reason was recorded as '{code}', which this build has "
            "no sentence for — treat the question as unsettled"
        )
    try:
        finding.probe_note = note
    except Exception:  # pragma: no cover - a frozen model must not fail the run
        pass


async def run_inquiries(
    probe: Optional[Callable[..., Any]],
    findings: Sequence[Any],
    pack: Any = None,
    analysis: Any = None,
    logs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    config: Optional[Dict[str, Any]] = None,
    link_config: Optional[Dict[str, Any]] = None,
    ruleset_key: str = "",
) -> int:
    """Spend up to the configured budget of probes and settle what they answer.

    Returns probes spent. Never raises. Sequential, for the link lane's reasons: a concurrent
    advisory scan contends with the stage it advises, and a deadline with N in flight can only
    cancel.

    Three probe answers, three states: ``None`` (the source did not answer), a list including
    ``[]`` (it answered), an exception (type logged, never the message — a backend error may
    carry this incident's identifiers).
    """
    cfg = config or {}
    if probe is None or not findings or pack is None:
        return 0
    budget = inquiry_budget(cfg, link_config)
    if budget["max_probes"] <= 0:
        for finding in findings:
            if inquiry_refusal(finding, logs)[0]:
                _decline(finding, "lane_closed")
        return 0
    try:
        declarations = {
            str(d.get("id", "")): d for d in (pack.open_questions(ruleset_key) or [])
        }
    except Exception as exc:  # noqa: BLE001 — advisory, never fatal
        logger.warning(
            "Could not re-read this procedure's open questions (%s); no probe is spent and every "
            "question keeps its unsettled state.",
            type(exc).__name__,
        )
        return 0

    cap = inquiry_row_cap(cfg)
    per_probe = float(budget["timeout"])
    started = time.monotonic()
    spent = 0

    for finding in findings:
        question_id = str(getattr(finding, "id", "") or "")
        declaration = declarations.get(question_id)
        source, refusal = inquiry_refusal(finding, logs)
        if not source or declaration is None:
            # Recorded on the finding for every reason, not only the interesting ones: this lane's
            # whole claim is that an unasked question never reads as an answered one.
            logger.info(
                "No probe for open question %r: %s.",
                question_id,
                refusal or "the pack no longer declares it",
            )
            _decline(finding, refusal or "no_source")
            continue
        if spent >= budget["max_probes"]:
            _decline(finding, "budget_spent")
            continue
        remaining = budget["deadline_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            # Checked before each probe: a probe that overran its slice is already paid for.
            logger.info(
                "The open-question probe budget of %ds is exhausted after %d probe(s); the "
                "remaining questions are reported unsettled.",
                budget["deadline_seconds"],
                spent,
            )
            _decline(finding, "budget_spent")
            continue

        rows: Optional[List[Dict[str, Any]]] = None
        slice_seconds = min(per_probe, remaining)
        note = ""
        try:
            answer = await asyncio.wait_for(
                probe(
                    source,
                    analysis,
                    question=str(getattr(finding, "question", "") or ""),
                    row_cap=cap,
                    timeout=slice_seconds,
                    scope_entity=str(getattr(finding, "scope_entity", "") or ""),
                    scope_values=[
                        str(v) for v in (getattr(finding, "scope_values", None) or [])
                    ],
                ),
                timeout=slice_seconds,
            )
        except asyncio.TimeoutError:
            note = (
                f"a probe of '{source}' was asked and did not answer within its "
                f"{slice_text(slice_seconds)} slice, so this question is unsettled — the advisory "
                "budget is deliberately far below the one the system of record is granted, and a "
                "question that needs longer needs a condition of its own"
            )
        except Exception as exc:  # noqa: BLE001 — advisory, never fatal
            note = (
                f"a probe of '{source}' could not be asked ({type(exc).__name__}), so this "
                "question is unsettled — an environment or catalog gap, not a finding"
            )
            logger.warning(
                "An open-question probe of %r for %r raised %s; the question stays unsettled.",
                source,
                question_id,
                type(exc).__name__,
            )
        else:
            if answer is None:
                note = (
                    f"a probe of '{source}' was licensed and the source did not answer at all, "
                    "which is a different fact from answering with no rows and needs a credential "
                    "or a catalog entry rather than a reading"
                )
            else:
                rows = [r for r in answer if isinstance(r, dict)][:cap]
                spent += 1
                note = (
                    f"one probe of '{source}' was spent on this question and returned "
                    f"{len(rows)} row(s)"
                    + (
                        f", capped at {cap} — the rows this answer reads are bounded"
                        if len(rows) >= cap
                        else ""
                    )
                )

        settle_with_probe(
            finding,
            declaration,
            rows=rows,
            source=source,
            row_cap=cap,
            note=note,
        )

    if spent:
        logger.info(
            "Open-question probes: %d of a budgeted %d spent in %.1fs (ceiling %ds).",
            spent,
            budget["max_probes"],
            time.monotonic() - started,
            budget["deadline_seconds"],
        )
    return spent


def build_inquiry_probe(generator: Any, engine: Any) -> Optional[Callable[..., Any]]:
    """Build the probe fetcher from the generator and engine; ``None`` if either is absent.

    Query built through :meth:`ApiCallGenerator.build_manual_query` — the same seam the plan
    editor and the link probe both take — so every ``_attach_query_guards`` guarantee applies to
    an inquiry query for free. The scope rides on a COPY of the analysis so the shared one is not
    rescoped for the stages that come after.
    """
    if generator is None or engine is None:
        return None

    async def _run(
        source: str,
        analysis: Any,
        question: str = "",
        row_cap: int = PROBE_ROW_CAP_DEFAULT,
        timeout: float = PROBE_TIMEOUT_DEFAULT,
        scope_entity: str = "",
        scope_values: Optional[Sequence[str]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        scoped = _scoped_analysis(analysis, scope_entity, scope_values)
        try:
            query = generator.build_manual_query(scoped, source, question=question)
        except Exception as exc:  # noqa: BLE001 — "could not ask" is an answer
            logger.info(
                "An open-question probe of '%s' could not be built (%s).", source, exc
            )
            return None
        unanswered: Dict[str, str] = {}
        rows = await engine.retrieve([query], unanswered_out=unanswered)
        if not isinstance(rows, dict) or source in unanswered or source not in rows:
            # Absent is the non-answer; present-with-`[]` is the empty answer. Collapsing them
            # makes a timeout indistinguishable from a source with nothing to say — and this lane
            # declares a separate MEANING for each of those two outcomes.
            return None
        return [r for r in (rows.get(source) or []) if isinstance(r, dict)][:row_cap]

    return _run
