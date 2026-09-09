"""Rung 3 of the cross-procedure link ladder: the one rung that spends a query.

Rungs 0-2 read rows already in hand; this rung asks one source the target procedure owns
that this run did not retrieve. Probe rows reach the settlement only via
:func:`~src.links.resettle_with_probe`, never merged into ``logs``. Four licences required:
escalating mode, rung-1 gate pass, at least one signal with ``auto_probe: true``, and budget.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from link_escalation import (
    ESCALATING_MODES,
    PROBE_ROW_CAP_DEFAULT,
    PROBE_TIMEOUT_DEFAULT,
    gate_permits,
    normalise_mode,
    probe_budget,
    probe_row_cap,
    slice_text,
)
from links import resettle_with_probe

logger = logging.getLogger(__name__)


def probe_nominations(pack: Any, use_case: str) -> List[str]:
    """Every source nominated for this spend, best first, before any exclusion.

    Separated from :func:`probe_candidates` because an empty result otherwise cannot distinguish
    'nothing nominated' from 'everything nominated was already answered'. Signals first (keyed as
    in ``logs``), then applicability-gate sources (logical names resolved through the ruleset map).
    """
    ordered: List[str] = []

    def _add(name: Any) -> None:
        text = str(name or "").strip()
        if text and text not in ordered:
            ordered.append(text)

    try:
        signals = pack.entry_signals(use_case) or []
    except Exception:  # pragma: no cover - advisory, never fatal
        signals = []
    for signal in signals:
        if isinstance(signal, dict) and signal.get("auto_probe") is True:
            _add(signal.get("source"))

    try:
        spec = pack.ruleset_spec(use_case) or {}
    except Exception:  # pragma: no cover - advisory, never fatal
        spec = {}
    if isinstance(spec, dict):
        sources = spec.get("sources") if isinstance(spec.get("sources"), dict) else {}
        for condition in spec.get("conditions") or []:
            if not isinstance(condition, dict) or condition.get("gate") != "scope":
                continue
            logical = str(condition.get("source", "") or "")
            _add((sources or {}).get(logical, logical))
    return ordered


def probe_candidates(
    pack: Any, use_case: str, logs: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Nominated sources for ``use_case``, minus what this run already has.

    Zero rows is an answer the free rungs already read; re-asking would spend a query to
    reproduce it.
    """
    have = set(logs or {})
    return [name for name in probe_nominations(pack, use_case) if name not in have]


def probe_nominations_already_answered(
    pack: Any, use_case: str, logs: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Nominated sources this run already retrieved: the reportable half of the exclusion.

    Non-empty means the nomination is inert for this pair on every incident whose planner reaches
    the same source; the fix is in the pack, not the run.
    """
    have = set(logs or {})
    return [name for name in probe_nominations(pack, use_case) if name in have]


def _declares_auto_probe(pack: Any, use_case: str) -> bool:
    """True if any entry signal on the target carries ``auto_probe: true``.

    Read from the pack, not the finding: the finding carries the strongest signal only.
    No nomination → no probe, regardless of mode.
    """
    try:
        signals = pack.entry_signals(use_case) or []
    except Exception:  # pragma: no cover - advisory, never fatal
        return False
    return any(isinstance(s, dict) and s.get("auto_probe") is True for s in signals)


#: The one refusal surfaced to the operator; others restate fields the finding already carries.
REPORTABLE_REFUSAL = "nomination_already_answered"


def probe_refusal(
    finding: Any, pack: Any, logs: Optional[Dict[str, Any]] = None
) -> tuple:
    """The source this finding would probe, and the refusal code if it cannot.

    Returns ``(source, "")`` or ``(None, code)``. Budget is not checked here: it belongs to the
    run, not the candidate.
    """
    use_case = str(getattr(finding, "target_use_case", "") or "")
    if not use_case:
        return None, "no_target"
    # Mode on the finding is the resolved licence; re-deriving here would be a second answer.
    if normalise_mode(getattr(finding, "mode", "")) not in ESCALATING_MODES:
        return None, "mode_not_escalating"
    # Belt-and-braces: this is the function that spends.
    if not gate_permits(getattr(finding, "gate_outcome", "")):
        return None, "gate_withheld"
    # Pivot must be in hand; rung-1 pass sets state=probed_positive so !unsettled is wrong check.
    if bool(getattr(finding, "probe_spent", False)):
        return None, "already_probed"
    if str(getattr(finding, "state", "") or "") == "unreachable":
        return None, "pivot_unreachable"
    if not list(getattr(finding, "pivot_values", None) or []):
        return None, "no_pivot_value"
    if not _declares_auto_probe(pack, use_case):
        return None, "no_declaration_asked"
    candidates = probe_candidates(pack, use_case, logs)
    if candidates:
        return candidates[0], ""
    # Nominated-then-excluded is different from never nominated; only the first is reachable here.
    if probe_nominations_already_answered(pack, use_case, logs):
        return None, REPORTABLE_REFUSAL
    return None, "nothing_nominated"


def probe_eligible(
    finding: Any, pack: Any, logs: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """The source this finding would probe, or ``None``: :func:`probe_refusal`'s source half."""
    return probe_refusal(finding, pack, logs)[0]


async def run_link_probes(
    probe: Optional[Callable[..., Any]],
    findings: Sequence[Any],
    pack: Any = None,
    analysis: Any = None,
    logs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    entity_map: Optional[Dict[str, Dict[str, str]]] = None,
    pack_data: Optional[Dict[str, Any]] = None,
    row_caps: Optional[Dict[str, int]] = None,
    keyed_sources: Optional[Dict[str, bool]] = None,
    unanswered_sources: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
    ruleset_key: str = "",
    mode_overrides: Optional[Dict[str, str]] = None,
) -> int:
    """Spend up to the configured budget of probes and resettle what they answer.

    Returns probes spent. Never raises. Sequential: concurrent advisory scans would contend with
    the stage they advise, and a deadline with N in flight can only cancel.

    Three probe answers: ``None`` (could not ask, ``probe_spent`` stays False); a list including
    ``[]`` (asked and answered, re-settled); exception (type logged, never message).
    """
    cfg = config or {}
    budget = probe_budget(cfg)
    if probe is None or budget["max_probes"] <= 0 or not findings:
        return 0
    if pack is None or not hasattr(pack, "entry_signals"):
        return 0

    cap = probe_row_cap(cfg)
    per_probe = float(budget["timeout"])
    started = time.monotonic()
    spent = 0

    for finding in findings:
        if spent >= budget["max_probes"]:
            break
        remaining = budget["deadline_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            # Checked before each probe: a probe that overran its slice is already paid for.
            logger.info(
                "The rung-3 probe budget of %ds is exhausted after %d probe(s); the remaining "
                "candidates keep their free-rung settlement.",
                budget["deadline_seconds"],
                spent,
            )
            break
        source = None
        refusal = ""
        use_case = str(getattr(finding, "target_use_case", "") or "")
        try:
            source, refusal = probe_refusal(finding, pack, logs)
        except Exception as exc:  # noqa: BLE001 — advisory, never fatal
            logger.warning(
                "Could not decide whether %r is probeable (%s); it keeps its free-rung "
                "settlement.",
                getattr(finding, "target_use_case", ""),
                type(exc).__name__,
            )
        if not source:
            # Logged for every reason; surfaced to the operator for one (per `REPORTABLE_REFUSAL`).
            logger.info(
                "No rung-3 probe for %r: %s.",
                use_case,
                refusal or "no reason recorded",
            )
            if refusal == REPORTABLE_REFUSAL:
                answered = probe_nominations_already_answered(pack, use_case, logs)
                # Via the settlement's no-rows path: `probe_note` has one writer.
                resettle_with_probe(
                    finding,
                    pack,
                    analysis=analysis,
                    logs=logs,
                    probe_source="",
                    probe_rows=None,
                    probe_row_cap=cap,
                    entity_map=entity_map,
                    pack_data=pack_data,
                    row_caps=row_caps,
                    keyed_sources=keyed_sources,
                    unanswered_sources=unanswered_sources,
                    note=(
                        "no probe was spent: this run had already retrieved every source the "
                        "target procedure nominated for one ("
                        + ", ".join(sorted(answered)[:5])
                        + "), and the free rungs have read what those rows answered — re-asking "
                        "would spend a scan to reproduce an answer already in hand. The "
                        "nomination is therefore inert for this pair on any incident whose "
                        "planner reaches the same source"
                    ),
                    config=cfg,
                    ruleset_key=ruleset_key,
                    mode_overrides=mode_overrides,
                )
            continue

        rows: Optional[List[Dict[str, Any]]] = None
        note = ""
        slice_seconds = min(per_probe, remaining)
        try:
            answer = await asyncio.wait_for(
                probe(
                    source,
                    analysis,
                    question="",
                    row_cap=cap,
                    timeout=slice_seconds,
                    pivot_entity=str(getattr(finding, "pivot_entity", "") or ""),
                    pivot_values=[
                        str(v) for v in (getattr(finding, "pivot_values", None) or [])
                    ],
                    window=str(getattr(finding, "window_hint", "") or ""),
                ),
                timeout=slice_seconds,
            )
        except asyncio.TimeoutError:
            note = (
                f"a probe of '{source}' was asked and did not answer within its "
                f"{slice_text(slice_seconds)} slice, so nothing was settled — the advisory budget is "
                "deliberately far below the one the system of record is granted, and a source "
                "that needs longer needs a run of that procedure instead"
            )
        except Exception as exc:  # noqa: BLE001 — advisory, never fatal
            # Exception type, never message: a backend error may carry this incident's identifiers.
            note = (
                f"a probe of '{source}' could not be asked ({type(exc).__name__}), so nothing "
                "was settled here — this is an environment or catalog gap, not a finding about "
                "the target procedure"
            )
            logger.warning(
                "A rung-3 probe of %r for %r raised %s; the candidate keeps its free-rung "
                "settlement.",
                source,
                getattr(finding, "target_use_case", ""),
                type(exc).__name__,
            )
        else:
            if answer is None:
                note = (
                    f"a probe of '{source}' was licensed and could not be asked — the source "
                    "did not answer at all, which is a different fact from answering with no "
                    "rows and needs a credential or a catalog entry rather than a referral"
                )
            else:
                rows = [r for r in answer if isinstance(r, dict)][:cap]
                spent += 1
                note = (
                    f"one probe of '{source}' was spent on this candidate and returned "
                    f"{len(rows)} row(s)"
                    + (
                        f", capped at {cap} — the rows this settlement reads are bounded"
                        if len(rows) >= cap
                        else ""
                    )
                )

        resettle_with_probe(
            finding,
            pack,
            analysis=analysis,
            logs=logs,
            probe_source=source,
            probe_rows=rows,
            probe_row_cap=cap,
            entity_map=entity_map,
            pack_data=pack_data,
            row_caps=row_caps,
            keyed_sources=keyed_sources,
            unanswered_sources=unanswered_sources,
            config=cfg,
            note=note,
            # Settlement re-resolves the mode through all four layers; dropping the override
            # here silently undoes the operator's click.
            ruleset_key=ruleset_key,
            mode_overrides=mode_overrides,
        )

    if spent:
        logger.info(
            "Rung-3 probes: %d of a budgeted %d spent in %.1fs (ceiling %ds).",
            spent,
            budget["max_probes"],
            time.monotonic() - started,
            budget["deadline_seconds"],
        )
    return spent


def build_link_probe(generator: Any, engine: Any) -> Optional[Callable[..., Any]]:
    """Build the probe fetcher from the generator and engine; ``None`` if either absent.

    Query built through :meth:`ApiCallGenerator.build_manual_query` — same seam as the plan
    editor — so pack guards apply. Pivot rides on a copy of the analysis so the shared one is
    not rescoped.
    """
    if generator is None or engine is None:
        return None

    async def _run(
        source: str,
        analysis: Any,
        question: str = "",
        row_cap: int = PROBE_ROW_CAP_DEFAULT,
        timeout: float = PROBE_TIMEOUT_DEFAULT,
        pivot_entity: str = "",
        pivot_values: Optional[Sequence[str]] = None,
        window: str = "",
    ) -> Optional[List[Dict[str, Any]]]:
        scoped = _scoped_analysis(analysis, pivot_entity, pivot_values)
        try:
            query = generator.build_manual_query(scoped, source, question=question)
        except Exception as exc:  # noqa: BLE001 — "could not ask" is an answer
            logger.info("A rung-3 probe of '%s' could not be built (%s).", source, exc)
            return None
        unanswered: Dict[str, str] = {}
        rows = await engine.retrieve([query], unanswered_out=unanswered)
        if not isinstance(rows, dict) or source in unanswered or source not in rows:
            # Absent is the non-answer; present-with-`[]` is the empty answer. Collapsing them
            # makes a timeout indistinguishable from a source with nothing to say.
            return None
        return [r for r in (rows.get(source) or []) if isinstance(r, dict)][:row_cap]

    return _run


def _scoped_analysis(
    analysis: Any, pivot_entity: str, pivot_values: Optional[Sequence[str]]
) -> Any:
    """A copy of the understanding with the pivot entities appended.

    Appended, not substituted: a probe still needs the incident's window and scope. Duck-typed:
    the analysis reaches this module under two import identities.
    """
    values = [str(v) for v in (pivot_values or []) if str(v or "").strip()]
    if not pivot_entity or not values:
        return analysis
    existing = list(getattr(analysis, "extracted_entities", None) or [])
    try:
        scoped = analysis.model_copy(deep=True)
    except Exception:  # pragma: no cover - not a Pydantic model
        return analysis
    if not existing:
        return scoped
    # Built via the existing list's own class to avoid the dual-import identity trap.
    factory = type(existing[0])
    made = []
    for value in values:
        try:
            made.append(factory(type=pivot_entity, value=value))
        except Exception:  # pragma: no cover - an unexpected entity model
            return scoped
    held = {
        (str(getattr(e, "type", "")), str(getattr(e, "value", ""))) for e in existing
    }
    scoped.extracted_entities = list(existing) + [
        e
        for e in made
        if (str(getattr(e, "type", "")), str(getattr(e, "value", ""))) not in held
    ]
    return scoped
