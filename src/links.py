"""Cross-procedure link assessment: could this evidence belong to another procedure?

Cost ladder (stops at settlement): rung 0 = entity in hand (free); rung 1 = gate passes
(verdict engine, free); rung 2 = entry signal fired (rows in hand, free); rung 3 = one
query (link_probe.py; settled here via ``resettle_with_probe``); rung 4 = not implemented.
Cannot fetch: no async, no IO, no clock (AST-asserted). Links are advisory: nothing here
reaches the verdict, condition lines, severity, or health score (asserted by
``tests/test_links_never_change_the_verdict.py``). ``logs`` read-only: probe rows into
scratch. Rolled-up sibling label never read.
Four states (not collapsed): ``unreachable``, ``not_probed``, ``probed_negative``,
``probed_positive`` (referral recommended, not a verdict).
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from correlation import apply_where, evaluate_verdict
from link_escalation import (
    MANUAL_LINK_MODE,
    apply_link_mode,
    escalation_budgeted,
    link_score,
)
from models.pydantic_models import LinkFinding

logger = logging.getLogger(__name__)

#: The four outcomes, ordered actionable first: the ruled-out candidate is listed before the
#: unreachable one so a reader who stops halfway has seen every candidate the engine examined.
LINK_STATES: Tuple[str, ...] = (
    "probed_positive",
    "not_probed",
    "probed_negative",
    "unreachable",
)

#: The causal axis. ``antecedent``: the sibling may have caused the reported event.
#: ``consequent``: the reported event may have caused the sibling's pattern.
#: The window hint derives from this word (``lookback`` vs ``onwards``).
LINK_DIRECTIONS: Tuple[str, ...] = ("antecedent", "consequent")

#: Advisory severity when a candidate is confirmed and the caller configured nothing. A triage
#: default for a human; it is never this run's severity and never an input to it.
_DEFAULT_ADVISORY_SEVERITY = "HIGH"

_ADVISORY_PROVENANCE = "added by the link assessment, addressed to a human"


def assess_links(
    pack: Any,
    analysis: Any,
    logs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    verdict: Any = None,
    brief: Any = None,
    ruleset_key: str = "",
    entity_map: Optional[Dict[str, Dict[str, str]]] = None,
    pack_data: Optional[Dict[str, Any]] = None,
    row_caps: Optional[Dict[str, int]] = None,
    keyed_sources: Optional[Dict[str, bool]] = None,
    unanswered_sources: Optional[Dict[str, str]] = None,
    requested: Sequence[str] = (),
    config: Optional[Dict[str, Any]] = None,
    mode_overrides: Optional[Dict[str, str]] = None,
) -> List[LinkFinding]:
    """Assess every declared cross-procedure candidate over the rows this run retrieved.

    Pure, deterministic, total: never raises; returns ``[]`` for packs with no link surface.
    Three routes: playbook named a sibling, sibling declared entry signals, incident text asked.

    ``requested`` is honored even where nothing is declared and even where it cannot be
    served: an unservable request is reported rather than silently dropped.

    ``mode_overrides`` is the per-job escalation setting, narrowest of the four layers
    ``src.link_escalation.resolve_link_mode`` reads. Still subject to rung 1: the target's
    scope gate withholds automatic spend, not the referral.
    """
    cfg = config or {}
    if not bool(cfg.get("enabled", True)):
        return []
    if pack is None:
        return []
    rows = logs if isinstance(logs, dict) else {}
    try:
        candidates = _candidates(pack, ruleset_key, brief, requested)
        if not candidates:
            return []
        in_hand = _pivots_in_hand(pack, analysis, verdict, brief, ruleset_key)
        reach = pack.entity_binding_map() if hasattr(pack, "entity_binding_map") else {}
        findings = [
            _assess_one(
                pack,
                candidate,
                in_hand,
                reach,
                rows,
                analysis,
                entity_map,
                pack_data,
                row_caps,
                keyed_sources,
                unanswered_sources,
                cfg,
            )
            for candidate in candidates
        ]
        findings = [f for f in findings if f is not None]
        # Resolved after all states are settled, for every candidate including unreachable
        # ones: _assess_one returns early before pack declarations are read.
        for finding in findings:
            apply_link_mode(
                finding,
                pack=pack,
                ruleset_key=ruleset_key,
                config=cfg,
                overrides=mode_overrides or {},
            )
    except Exception as exc:  # noqa: BLE001 — advisory: never fail the stage
        # An advisory lane that can break the run it advises on is worse than no advisory lane.
        # Logged at error level because a silent empty list is indistinguishable from a pack
        # that declared nothing, and those two need opposite responses.
        logger.error(
            "Cross-procedure link assessment failed and is reported as EMPTY for this run "
            "(%s). This is an advisory channel: the verdict, the brief and the health score "
            "are unaffected.",
            exc,
            exc_info=True,
        )
        return []
    order = {state: i for i, state in enumerate(LINK_STATES)}
    findings.sort(key=lambda f: (order.get(f.state, len(order)), f.target_use_case))
    logger.info(
        "Cross-procedure link assessment: %d candidate(s) — %s.",
        len(findings),
        ", ".join(
            f"{f.target_use_case}: {f.state}"
            # Log mode only when non-trivial: not MANUAL_LINK_MODE (the no-spend outcome),
            # or when an escalating mode was asked for but not applied (mode_note is set).
            + (
                f" [{f.mode} via {f.mode_source}]"
                if f.mode != MANUAL_LINK_MODE or str(getattr(f, "mode_note", "") or "").strip()
                else ""
            )
            for f in findings
        )
        or "none",
    )
    return findings


# --- candidates -------------------------------------------------------------------------


def _candidates(
    pack: Any, ruleset_key: str, brief: Any, requested: Sequence[str]
) -> List[Dict[str, Any]]:
    """The sibling procedures worth asking about, in stable order.

    Three routes only: operator request, related playbook, entry signal. Enumerating every
    pack ruleset would report unrelated procedures as candidates. Duplicates are kept at their
    first mention.
    """
    mine = str(ruleset_key or "").strip()
    playbook_id = str(getattr(brief, "playbook_id", "") or "")
    out: List[Dict[str, Any]] = []
    seen = set()

    def add(use_case: str, source: str, playbook: str = "", label: str = "") -> None:
        key = use_case or label
        if not key or key == mine or key in seen:
            return
        seen.add(key)
        out.append(
            {
                "use_case": use_case,
                "playbook_id": playbook,
                "origin": source,
                # What the operator actually wrote, when it resolved to no procedure at all.
                "label": label or use_case,
            }
        )

    for asked in requested or ():
        name = str(asked or "").strip()
        if not name:
            continue
        resolved = ""
        try:
            resolved = pack.ruleset_key_for(name)
        except Exception:  # noqa: BLE001 — a malformed ask must not lose the others
            resolved = ""
        add(resolved, "requested", label=name)

    if playbook_id and hasattr(pack, "related_playbooks"):
        for related in pack.related_playbooks(playbook_id):
            add(
                str(related.get("use_case", "") or ""),
                "related_playbook",
                playbook=str(related.get("playbook_id", "") or ""),
                label=str(related.get("playbook_id", "") or ""),
            )

    if hasattr(pack, "ruleset_keys") and hasattr(pack, "entry_signals"):
        for key in pack.ruleset_keys():
            if str(key) == mine:
                continue
            if pack.entry_signals(str(key)):
                add(str(key), "entry_signal")
    return out


def _score(
    finding: LinkFinding,
    cfg: Dict[str, Any],
    pivot_in_hand: bool = False,
    gate: str = "",
    fired: Sequence[Dict[str, Any]] = (),
) -> None:
    """Stamp the candidate's deterministic score and gate outcome.

    Called at every exit, including exits before rung 0. A finding with no score reads as
    zero; the escalation seam gates on the number.

    Score is computed here where the gate outcome and signal's ``strength``/``base_rate``
    are in scope: they do not survive on the finding in a form arithmetic can read.

    ``gate_outcome`` is also stamped here: it licenses automatic escalation and an override
    handler has no rows to re-derive it from.
    """
    first = dict(fired[0]) if fired else {}
    result = link_score(
        {
            "pivot_in_hand": pivot_in_hand,
            "gate": gate,
            "signal_fired": bool(fired),
            "signal_strength": first.get("strength"),
            "base_rate": first.get("base_rate"),
        },
        cfg,
    )
    finding.gate_outcome = str(gate or "")
    finding.link_score = float(result["score"])
    finding.link_score_reasons = [str(r) for r in result["reasons"]]


def _assess_one(
    pack: Any,
    candidate: Dict[str, Any],
    in_hand: Dict[str, List[str]],
    reach: Dict[str, List[str]],
    logs: Dict[str, List[Dict[str, Any]]],
    analysis: Any,
    entity_map: Optional[Dict[str, Dict[str, str]]],
    pack_data: Optional[Dict[str, Any]],
    row_caps: Optional[Dict[str, int]],
    keyed_sources: Optional[Dict[str, bool]],
    unanswered_sources: Optional[Dict[str, str]],
    cfg: Dict[str, Any],
) -> Optional[LinkFinding]:
    """One candidate, one climb of the free rungs, one :class:`LinkFinding`."""
    use_case = str(candidate.get("use_case", "") or "")
    requested = candidate.get("origin") == "requested"
    finding = LinkFinding(
        target_use_case=use_case or str(candidate.get("label", "") or ""),
        target_playbook_id=str(candidate.get("playbook_id", "") or ""),
    )
    if requested:
        finding.advisory_note = (
            "the incident text asked for this check by name, so it is reported whatever the "
            "free evidence says"
        )

    spec = pack.ruleset_spec(use_case) if use_case else None
    if not use_case or not isinstance(spec, dict):
        # An ask (or a stated relationship) this pack cannot serve. Reported, never dropped:
        # the reader's next step is to install or author the procedure, and silence here reads
        # as "checked, nothing found".
        finding.state = "unreachable"
        finding.gap_reason = (
            f"this pack declares no procedure named "
            f"'{candidate.get('label') or use_case}', so there is nothing to hand a "
            "referral to"
        )
        _score(finding, cfg)
        return finding

    subject = str(spec.get("subject_entity", "") or "")
    finding.pivot_entity = subject
    signals = pack.entry_signals(use_case) if hasattr(pack, "entry_signals") else []
    finding.direction = _direction(signals)
    finding.window_hint = _window_hint(finding.direction, signals)

    # --- rung 0: subject entity value in hand? -------------------------------------------
    # A leg is opened by a subject value; conditions resolve against the subject entity.
    values = sorted(set(in_hand.get(subject, [])))
    if not subject:
        finding.state = "unreachable"
        finding.gap_reason = (
            "this procedure declares no subject entity, so no value can open it"
        )
        _score(finding, cfg)
        return finding
    if not values:
        convertible = sorted(
            {
                held
                for held, reachable in (reach or {}).items()
                if subject in (reachable or []) and in_hand.get(held)
            }
        )
        if not convertible:
            # Unreachable is a pack fact, not an incident fact: no source binds
            # the sibling's subject beside what this run holds. Fix in the catalog.
            finding.state = "unreachable"
            finding.gap_reason = (
                f"no source in this pack binds '{subject}' beside any entity type this run "
                f"holds ({', '.join(sorted(t for t in in_hand if in_hand[t])) or 'none'}), so "
                "no value here can open this procedure"
            )
            _score(finding, cfg)
            return finding
        finding.rung = 0
        finding.state = "not_probed"
        # When a probe budget is configured, the reason notes that a probe confirms a
        # subject rather than harvests one (harvesting is rung 4).
        finding.gap_reason = (
            f"'{subject}' is not in hand, but a source binds it beside "
            f"{', '.join(convertible)} — one retrieval would produce it, "
            + (
                "and a rung-3 probe confirms a subject rather than harvesting one, so this "
                "needs a run of that procedure"
                if escalation_budgeted(cfg)
                else "and no retrieval is spent on a link"
            )
        )
        _score(finding, cfg)
        return finding
    finding.rung = 0
    finding.pivot_values = values[:20]

    # --- rung 1: does the sibling's own applicability test survive these rows? -----------
    gate = _gate_outcome(
        spec,
        logs,
        analysis,
        entity_map,
        pack_data,
        row_caps,
        keyed_sources,
        unanswered_sources,
    )
    if gate["outcome"] in {"fail", "pass", "unknown"}:
        finding.rung = 1

    # --- rung 2: did anything the sibling declares about itself actually fire? -----------
    fired, evaluated, strongest = _fired_signals(signals, logs)
    if evaluated:
        finding.rung = max(finding.rung, 2)
    if fired:
        finding.signal_id = fired[0]["id"]
        finding.base_rate = _base_rate_text(fired[0])

    _score(
        finding, cfg, pivot_in_hand=True, gate=str(gate.get("outcome", "")), fired=fired
    )
    _settle(finding, gate, fired, evaluated, strongest, subject, values, cfg)
    return finding


def _settle(
    finding: LinkFinding,
    gate: Dict[str, Any],
    fired: List[Dict[str, Any]],
    evaluated: int,
    strongest: float,
    subject: str,
    values: List[str],
    cfg: Dict[str, Any],
    probed: bool = False,
) -> None:
    """Settle the candidate to one state, one note, and one advisory line.

    Gate failing outranks any number of indicators firing: the gate answers whether the
    procedure applies at all. ``probed`` changes only the cost-claiming sentences: "for free"
    vs "after one probe".
    """
    subject_line = f"'{subject}' in hand: {', '.join(values[:5])}"
    if len(values) > 5:
        subject_line += f" (+{len(values) - 5} more)"
    # Named once: what this settlement cost, in the words the three cost-claiming sentences
    # below need. Read off the finding rather than passed, because the caller has already
    # stamped it and a second parameter would be a second place for the two to disagree.
    asked = str(getattr(finding, "probe_source", "") or "") or "one further source"
    cost = f"after one probe of '{asked}'" if probed else "at no retrieval cost"

    if gate["outcome"] == "fail":
        finding.state = "probed_negative"
        finding.evidence_note = (
            f"{subject_line}. This procedure's own applicability test FAILS on these rows "
            f"({gate['detail']}) — it was considered and ruled out, {cost}."
        )
        if fired:
            finding.evidence_note += (
                f" Its declared signal '{fired[0]['id']}' did fire "
                f"({fired[0]['matched']} row(s) of '{fired[0]['source']}'); the gate outranks "
                "it, because whether the procedure applies is a prior question to what its "
                "indicators say."
            )
        return

    if fired:
        finding.state = "probed_positive"
        detail = "; ".join(
            f"'{s['id']}' matched {s['matched']} row(s) of '{s['source']}' "
            f"(needs {s['min_rows']})"
            for s in fired
        )
        where = (
            f"on rows this run already retrieved and one probe of '{asked}'"
            if probed
            else "on rows this run already retrieved"
        )
        finding.evidence_note = (
            f"{subject_line}. {len(fired)} signal(s) this procedure declares about itself "
            f"fired {where}: {detail}."
        )
        if gate["outcome"] == "pass":
            finding.evidence_note += (
                f" Its applicability test also holds ({gate['detail']})."
            )
        elif gate["outcome"] == "unknown":
            finding.evidence_note += (
                f" Its applicability test could not be answered here ({gate['detail']}), so "
                "the referral rests on the signal alone."
            )
        _advise(finding, strongest, cfg)
        return

    if gate["outcome"] == "pass":
        finding.state = "probed_positive"
        finding.evidence_note = (
            f"{subject_line}. This procedure's own applicability test HOLDS on these rows "
            f"({gate['detail']}), so its subject matter is present in this evidence. Whether "
            "its fraud is present is a question only its own run can answer."
        )
        _advise(finding, strongest, cfg)
        return

    # State names reflect what was adjudicated, not what was paid: not_probed is honest
    # when nothing was settled, even after a probe. Spend is recorded in probe_spent/probe_note.
    finding.state = "not_probed"
    finding.evidence_note = (
        f"{subject_line}. Nothing was settled "
        + (f"even {cost}" if probed else "for free")
        + ": "
        + {
            "unknown": f"its applicability test read as unresolved ({gate['detail']})",
            "no_gate": "it declares no applicability gate",
            "no_sources": f"none of its declared sources was retrieved here ({gate['detail']})",
            "no_subject": "it adjudicates no subject on these rows",
        }.get(
            gate["outcome"],
            gate["detail"] or "its applicability test was not evaluable",
        )
        + (
            f", and {evaluated} declared signal(s) were checked without firing"
            if evaluated
            else ", and it declares no signal that could be checked from these rows"
        )
        + "."
    )
    finding.gap_reason = (
        (
            f"a probe of '{asked}' was spent here and settled it neither way, so re-asking is "
            "not the remedy — this needs a run of that procedure"
        )
        if probed
        else (
            "settling this needs a source of its own, which costs a retrieval — refer it to a "
            "run of that procedure instead"
        )
    )


def _advise(finding: LinkFinding, strength: float, cfg: Dict[str, Any]) -> None:
    """Stamp the advisory severity triage default on a confirmed finding.

    The field is ``advisory_severity``, not ``severity``: these are separate lanes.
    A signal strength below the configured floor withholds the severity, not the finding.
    """
    floor = _as_float(cfg.get("min_signal_strength"), 0.0)
    severity = str(
        cfg.get("advisory_severity_default", _DEFAULT_ADVISORY_SEVERITY) or ""
    )
    note = _ADVISORY_PROVENANCE
    if strength and strength < floor:
        finding.advisory_note = (
            f"{note}; no severity is claimed, because the declared strength ({strength:g}) is "
            f"below the configured floor ({floor:g})"
        )
        return
    finding.advisory_severity = severity
    # Appended at most once: re-settlement can call this on a finding that already carries
    # the provenance.
    if not finding.advisory_note:
        finding.advisory_note = note
    elif note not in finding.advisory_note:
        finding.advisory_note += f"; {note}"


# --- rung 3: the settlement half (the fetching half is `src/link_probe.py`) --------------


def resettle_with_probe(
    finding: LinkFinding,
    pack: Any,
    analysis: Any = None,
    logs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    probe_source: str = "",
    probe_rows: Optional[Sequence[Dict[str, Any]]] = None,
    probe_row_cap: Optional[int] = None,
    entity_map: Optional[Dict[str, Dict[str, str]]] = None,
    pack_data: Optional[Dict[str, Any]] = None,
    row_caps: Optional[Dict[str, int]] = None,
    keyed_sources: Optional[Dict[str, bool]] = None,
    unanswered_sources: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
    note: str = "",
    ruleset_key: str = "",
    mode_overrides: Optional[Dict[str, str]] = None,
) -> None:
    """Re-read one candidate with one more source in view, and settle it again.

    Rung-3 only: rows arrive as an argument (cannot fetch; AST-asserted).

    Probe rows go into a scratch dict only: ``logs`` must not receive them (census,
    co-identity merge, and health scorer all read ``logs``).

    State may still be ``not_probed``: reflects what was adjudicated, not what was paid.
    ``probe_row_cap`` rides into a local ``row_caps`` copy. Mode is re-resolved: a probe
    may change rung 1, and a disconfirmed gate-pass must not keep escalation licence.

    Mutates ``finding`` in place. Never raises.
    """
    cfg = config or {}
    use_case = str(getattr(finding, "target_use_case", "") or "")
    source = str(probe_source or "")
    for attr, value in (
        ("probe_spent", bool(probe_rows is not None)),
        ("probe_source", source),
        ("probe_note", str(note or "")),
    ):
        setattr(finding, attr, value)
    if probe_rows is None or not source:
        # Nothing fetched; re-settling on the same rows would rewrite free-rung sentences
        # into the probed wording over a probe that never happened.
        return

    try:
        spec = pack.ruleset_spec(use_case) if use_case else None
        if not isinstance(
            spec, dict
        ):  # pragma: no cover - unreachable via the probe path
            return
        scratch = dict(logs or {})
        scratch[source] = [dict(r) for r in probe_rows if isinstance(r, dict)]
        caps = dict(row_caps or {})
        if probe_row_cap is not None:
            caps[source] = int(probe_row_cap)
        # A source that answered is removed from the unanswered register: the register tells
        # "said nothing" from "said nothing came back", and this call has the answer in hand.
        unanswered = {
            k: v for k, v in (unanswered_sources or {}).items() if str(k) != source
        }

        gate = _gate_outcome(
            spec,
            scratch,
            analysis,
            entity_map,
            pack_data,
            caps,
            keyed_sources,
            unanswered,
        )
        signals = pack.entry_signals(use_case) if hasattr(pack, "entry_signals") else []
        fired, evaluated, strongest = _fired_signals(signals, scratch)

        finding.rung = 3
        # Cleared before re-settlement: the free-rung fallthrough left "settling costs a
        # retrieval", which is wrong once a retrieval was actually spent.
        finding.gap_reason = ""
        if fired:
            finding.signal_id = fired[0]["id"]
            finding.base_rate = _base_rate_text(fired[0])
        _score(
            finding,
            cfg,
            pivot_in_hand=True,
            gate=str(gate.get("outcome", "")),
            fired=fired,
        )
        _settle(
            finding,
            gate,
            fired,
            evaluated,
            strongest,
            str(getattr(finding, "pivot_entity", "") or ""),
            [str(v) for v in (getattr(finding, "pivot_values", None) or [])],
            cfg,
            probed=True,
        )
        # After re-settlement: apply_link_mode reads the gate_outcome _score just re-stamped.
        apply_link_mode(
            finding,
            pack=pack,
            ruleset_key=str(ruleset_key or ""),
            config=cfg,
            overrides=mode_overrides or {},
        )
    except Exception as exc:  # noqa: BLE001 — advisory: never fail the stage
        # The spend happened; keep the free-rung settlement and record the failure in probe_note.
        logger.warning(
            "A rung-3 probe of %r came back and could not be re-read for %r (%s); keeping the "
            "free-rung settlement and recording the spend.",
            source,
            use_case,
            exc,
        )
        finding.probe_note = (
            f"{note + '; ' if note else ''}the probed rows could not be re-read here, so this "
            "candidate keeps the settlement the free rungs reached"
        )


# --- rung 0 -----------------------------------------------------------------------------


def _pivots_in_hand(
    pack: Any, analysis: Any, verdict: Any, brief: Any, ruleset_key: str
) -> Dict[str, List[str]]:
    """``{entity type: [value, ...]}`` for every typed value this run holds.

    Only typed values: a value with no established type cannot open a sibling leg.

    Four sources in increasing distance from the incident: incident entities, verdict
    subjects, scope-sweep subjects (typed by the ruleset's subject entity), and sweep
    assets (typed the same way). The last two are most likely to surface values the alert
    never named.
    """
    out: Dict[str, List[str]] = {}

    def put(etype: Any, value: Any) -> None:
        t = str(etype or "").strip()
        v = str(value or "").strip()
        if not t or not v:
            return
        bucket = out.setdefault(t, [])
        if v not in bucket:
            bucket.append(v)

    for ent in getattr(analysis, "extracted_entities", None) or []:
        put(getattr(ent, "type", "") or (ent or {}).get("type"), _entity_value(ent))

    for subject in getattr(verdict, "subjects", None) or []:
        put(getattr(subject, "subject_type", ""), getattr(subject, "subject_value", ""))

    own_subject = ""
    try:
        own_subject = str(
            (pack.ruleset_spec(ruleset_key) or {}).get("subject_entity", "")
        )
    except Exception:  # noqa: BLE001 — a pack that cannot answer contributes nothing
        own_subject = ""
    if own_subject:
        for value in getattr(brief, "additional_subjects", None) or []:
            put(own_subject, value)
        for asset in getattr(brief, "impacted_assets", None) or []:
            put(own_subject, getattr(asset, "subject", ""))
    return {t: v for t, v in sorted(out.items())}


def _entity_value(ent: Any) -> str:
    value = getattr(ent, "value", None)
    if value is None and isinstance(ent, dict):
        value = ent.get("value")
    return str(value or "")


# --- rung 1 -----------------------------------------------------------------------------


def _gate_outcome(
    spec: Dict[str, Any],
    logs: Dict[str, List[Dict[str, Any]]],
    analysis: Any,
    entity_map: Optional[Dict[str, Dict[str, str]]],
    pack_data: Optional[Dict[str, Any]],
    row_caps: Optional[Dict[str, int]],
    keyed_sources: Optional[Dict[str, bool]],
    unanswered_sources: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Evaluate the sibling's ``gate: scope`` conditions over these rows, at zero cost.

    Prunes the spec to its gate conditions only. A spec pruned to its gate falls through the
    rollup to the sibling's own no-exclusion-fired exit, so the result carries the sibling's
    verdict label: only the per-condition results and the ``None`` return are read; the label
    is never touched.

    Returns ``{"outcome": pass|fail|unknown|no_gate|no_sources|no_subject, "detail": str}``.
    """
    conditions = [
        c
        for c in (spec.get("conditions") or [])
        if isinstance(c, dict) and c.get("gate") == "scope"
    ]
    if not conditions:
        # no_gate is not a failure: applicability may be settled by what the incident is
        # about, not by a row condition. An empty list would read as "does not apply".
        return {
            "outcome": "no_gate",
            "detail": "no condition is declared as its scope gate",
        }

    pruned = dict(spec)
    pruned["conditions"] = conditions
    gate_ids = {str(c.get("id", "")) for c in conditions}
    try:
        result = evaluate_verdict(
            pruned,
            logs,
            analysis,
            entity_map,
            pack_data,
            row_caps=row_caps,
            keyed_sources=keyed_sources,
            unanswered_sources=unanswered_sources,
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 — advisory: a sibling's spec is not this run's
        logger.warning(
            "A sibling procedure's scope gate could not be evaluated over this run's rows "
            "(%s); reporting it as unresolved rather than as inapplicable.",
            exc,
        )
        return {"outcome": "unknown", "detail": "its gate could not be evaluated here"}

    if result is None:
        declared = spec.get("sources") or {}
        return {
            "outcome": "no_sources",
            "detail": (
                f"{len(declared)} declared source(s), none present in this run's "
                f"{len(logs)} retrieved source(s)"
            ),
        }
    subjects = getattr(result, "subjects", None) or []
    if not subjects:
        return {
            "outcome": "no_subject",
            "detail": "it adjudicated no subject on these rows",
        }

    results: List[str] = []
    details: List[str] = []
    for subject in subjects:
        for check in getattr(subject, "checks", None) or []:
            if str(getattr(check, "id", "")) not in gate_ids:
                continue
            outcome = str(getattr(check, "result", "") or "")
            results.append(outcome)
            note = str(
                getattr(check, "detail", "") or getattr(check, "observed", "") or ""
            )
            if note and note not in details:
                details.append(note)
    if not results:
        return {
            "outcome": "unknown",
            "detail": "its gate produced no result on these rows",
        }
    detail = "; ".join(details[:2]) or f"{len(results)} gate check(s) evaluated"
    # Gate is all-of: one fail means the procedure declines; unknown only if nothing decided.
    if any(r == "fail" for r in results):
        return {"outcome": "fail", "detail": detail}
    if any(r == "pass" for r in results):
        return {"outcome": "pass", "detail": detail}
    return {"outcome": "unknown", "detail": detail}


# --- rung 2 -----------------------------------------------------------------------------


def _fired_signals(
    signals: Iterable[Dict[str, Any]], logs: Dict[str, List[Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], int, float]:
    """Which declared entry signals fired on rows already retrieved.

    The minimum row count is load-bearing: a shape on one row is not a detector. A source
    not in the run's logs is excluded from the evaluated count, not counted as a miss.

    Returns ``(fired, evaluated, strongest_strength)``.
    """
    fired: List[Dict[str, Any]] = []
    evaluated = 0
    strongest = 0.0
    for signal in signals or []:
        source = str(signal.get("source", "") or "")
        if not source or source not in logs:
            continue
        rows = [r for r in (logs.get(source) or []) if isinstance(r, dict)]
        where = signal.get("where") or []
        if where:
            try:
                rows = apply_where(rows, where)
            except Exception as exc:  # noqa: BLE001 — a clause, not the assessment
                # The clause is the scope of the question; falling through to every row would
                # ask a different question and answer it confidently. This signal contributes nothing.
                logger.warning(
                    "A sibling's entry signal %r declares a row selector that could not be "
                    "applied (%s); it contributes nothing rather than reading rows it did not "
                    "ask about.",
                    signal.get("id"),
                    exc,
                )
                continue
        evaluated += 1
        matched = len(rows)
        if matched >= int(signal.get("min_rows", 1) or 1):
            fired.append({**signal, "matched": matched})
            strongest = max(strongest, _as_float(signal.get("strength"), 0.0))
    fired.sort(key=lambda s: (-_as_float(s.get("strength"), 0.0), str(s.get("id", ""))))
    return fired, evaluated, strongest


def _base_rate_text(signal: Dict[str, Any]) -> str:
    """The signal's ``base_rate`` block as readable text, or a "not measured" note.

    A fired signal with no counted base rate is reported as such so "not measured" does not
    read as "rare".
    """
    rate = signal.get("base_rate") or {}
    if not isinstance(rate, dict) or not rate:
        return "not measured — this signal's firing rate over a corpus has not been counted"
    kind = str(rate.get("kind", "") or "").strip().lower()
    fires_on = rate.get("fires_on")
    of = rate.get("of")
    if kind == "stub" or fires_on is None or of in (None, 0):
        return "declared as a stub — not yet measured over a corpus"
    measured = str(rate.get("measured", "") or "").strip()
    text = f"fired on {fires_on} of {of} incident(s) in the measured corpus"
    return f"{text} (measured {measured})" if measured else text


# --- windows and directions --------------------------------------------------------------


def _direction(signals: Iterable[Dict[str, Any]]) -> str:
    """The candidate's causal direction from its declarations, or '' if none declared.

    When both are declared, ``antecedent`` is reported first (earlier window).
    """
    declared = [
        str(s.get("direction", "") or "").strip().lower() for s in signals or []
    ]
    for direction in LINK_DIRECTIONS:
        if direction in declared:
            return direction
    return ""


def _window_hint(direction: str, signals: Iterable[Dict[str, Any]]) -> str:
    """Window hint for a referral, derived from the signal and direction.

    ``antecedent`` maps to ``lookback``; ``consequent`` maps to ``onwards``.
    """
    for signal in signals or []:
        if str(signal.get("direction", "") or "").strip().lower() != direction:
            continue
        window = str(signal.get("window", "") or "").strip().lower()
        if window and window != "inherit":
            return window
    if direction == "antecedent":
        return "lookback"
    if direction == "consequent":
        return "onwards"
    return "inherit"


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --- the referral, composed and never launched -------------------------------------------


#: Message for ValueError when no pivot value is in hand. A child with no subject would
#: scan the sibling's sources over the whole window.
_NO_PIVOT_REFUSAL = (
    "This link is {state} because the run holds no value of {entity} to scope a referral by, "
    "so there is nothing to compose: a child run with no subject would scan the sibling's "
    "sources over the whole window instead of asking about one identity."
)

#: Note included in every composed request explaining the pin. ``link_pin`` is honoured by
#: ``select_correlation_spec`` ahead of the playbook score, so the child is adjudicated under
#: the target ruleset regardless of how its description scores.
_PIN_ENFORCED = (
    "Applied. `link_pin` in the request body selects the child's procedure through the same "
    "single resolution seam the verdict and the retrieval plan both read, so the child is "
    "adjudicated under this ruleset and not under whatever its description scores as. Submit "
    "the request verbatim: dropping the field falls the child back to scoring its description."
)

_DIRECTION_PROSE = {
    "antecedent": (
        "The earlier run's evidence suggests this procedure's pattern may have come FIRST and "
        "led to what was reported, so ask over the window BEFORE the reported event."
    ),
    "consequent": (
        "The earlier run's evidence suggests what was reported may have LED TO this "
        "procedure's pattern, so ask over the window AFTER the reported event."
    ),
}


def compose_referral(
    finding: Any,
    parent_job_id: str = "",
    parent_incident_id: str = "",
    window: Tuple[str, str] = ("", ""),
    mode: str = "auto",
) -> Dict[str, Any]:
    """The child request a referral would run, composed here and launched nowhere.

    Pure: the same finding composes the same request on every call.

    Parent incident prose is absent from the description: the child's procedure is chosen by
    scoring its description, and the parent's text is precisely what made the parent's
    procedure win. The parent is referenced by opaque identifiers only.

    Raises ``ValueError`` when the link holds no pivot value (see ``_NO_PIVOT_REFUSAL``).
    """
    target = str(getattr(finding, "target_use_case", "") or "").strip()
    entity = str(getattr(finding, "pivot_entity", "") or "").strip() or "subject"
    values = [
        str(v).strip()
        for v in (getattr(finding, "pivot_values", None) or [])
        if str(v).strip()
    ]
    if not values:
        raise ValueError(
            _NO_PIVOT_REFUSAL.format(
                state=str(getattr(finding, "state", "") or "unassessed"), entity=entity
            )
        )
    direction = str(getattr(finding, "direction", "") or "").strip().lower()
    date_from, date_to = (window or ("", ""))[:2]

    lines = [
        f"Referral from job {parent_job_id or '(unknown)'} "
        f"(incident {parent_incident_id or 'unknown'}).",
        f"Investigate {target or 'the linked procedure'} for {entity} "
        f"{', '.join(values)}.",
    ]
    if direction in _DIRECTION_PROSE:
        lines.append(_DIRECTION_PROSE[direction])
    note = str(getattr(finding, "evidence_note", "") or "").strip()
    if note:
        # evidence_note is in the sibling's own terms, safe for a description the scorer reads.
        # Terminated because it is a phrase, not a sentence.
        if not note.endswith((".", "!", "?")):
            note += "."
        lines.append(f"What the earlier run already found: {note}")
    scope = f"Scope: {entity} {', '.join(values)}"
    if date_from or date_to:
        scope += f", {date_from or 'unbounded'} to {date_to or 'unbounded'}"
    lines.append(scope + ".")

    return {
        "request": {
            "description": " ".join(lines),
            "mode": mode or "auto",
            # In the request body as well as the pin block: a client that doesn't recognise
            # the pin block would otherwise drop it.
            "link_pin": target,
        },
        "post_to": "/api/v1/jobs",
        # The pin; see `_PIN_ENFORCED` for why it decides.
        "pin": {
            "use_case": target,
            "playbook_id": str(getattr(finding, "target_playbook_id", "") or ""),
            "enforced": True,
            "note": _PIN_ENFORCED,
        },
        "scope": {
            "pivot_entity": entity,
            "pivot_values": values,
            "direction": direction,
            "date_from": date_from,
            "date_to": date_to,
            "window_hint": str(getattr(finding, "window_hint", "") or ""),
        },
        "parent": {"job_id": parent_job_id, "incident_id": parent_incident_id},
    }
