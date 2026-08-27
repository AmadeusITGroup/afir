"""Renders the deterministic ``InvestigationBrief`` into prompt text.

Shared by anomaly detection and report generation so both stages read the same ground truth.
``phrase`` resolves the pack's wording for a named slot, falling back to a complete generic
sentence so a pack-less domain still gets correct text.
"""

import logging

logger = logging.getLogger(__name__)

#: Char budget for the rendered brief; a sub-budget bounding what the brief takes from the
#: prompt. Overridable per stage via ``brief_char_budget``.
DEFAULT_BRIEF_CHAR_BUDGET = 6000


def phrase(phrases, slot, default: str, **subs) -> str:
    """The pack's wording for a named slot, else ``default``, with placeholders filled.

    Uses ``str.replace``, not ``.format``: pack text is procedure prose that routinely
    contains braces.
    """
    text = ""
    if isinstance(phrases, dict):
        text = str(phrases.get(slot) or "").strip()
    text = text or default
    for key, value in subs.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def render_brief_for_prompt(
    brief, char_budget: int = DEFAULT_BRIEF_CHAR_BUDGET, phrases=None
) -> str:
    """Render the InvestigationBrief as a compact factual block for an LLM prompt.

    Returns "" for a missing brief. Defensive against ``MagicMock``.
    """
    if brief is None:
        return ""
    verdict = getattr(brief, "verdict", None)
    subjects = getattr(verdict, "subjects", None) if verdict is not None else None
    backbone_guard = getattr(brief, "action_backbone", None)
    # Require a real list: a MagicMock's attributes are truthy Mocks, not lists.
    if not isinstance(subjects, list) and not isinstance(backbone_guard, list):
        return ""

    lines = []
    uc = getattr(brief, "use_case", "") or ""
    if uc:
        lines.append(f"Use case: {uc}")
    # Allegation first, as a prohibition: without it the narrator fills in the most
    # suspicious-looking retrieved fact.
    _af = getattr(brief, "alert_facts", None)
    _trigger = str(getattr(_af, "trigger", "") or "").strip() if _af is not None else ""
    if _trigger:
        lines.append(
            "WHAT THE DETECTOR FIRES ON (ground truth — the allegation this case is "
            f"about): {_trigger} Do NOT state or imply any other trigger, and do not "
            "present a retrieved fact as the reason the alert was raised unless it is "
            "this one."
        )
    if verdict is not None and isinstance(subjects, list):
        lines.append(f"VERDICT: {getattr(verdict, 'summary', '')}")
        for sv in subjects:
            lines.append(
                f"- {getattr(sv, 'subject_type', 'subject')} "
                f"{getattr(sv, 'subject_value', '')}: {getattr(sv, 'verdict', '')}"
            )
            lt = getattr(sv, "lock_target", {}) or {}
            # `scope`/`identity` from lock_target; pack-declared `*_label`s when present.
            _who = str(lt.get("identity", "") or "").strip()
            _where = str(lt.get("scope", "") or "").strip()
            if _who or _where:
                _w_lab = str(lt.get("scope_label", "") or "").strip() or "scope"
                _i_lab = str(lt.get("identity_label", "") or "").strip() or "identity"
                _target = ", ".join(
                    f"{lab.lower()}={val}"
                    for lab, val in ((_i_lab, _who), (_w_lab, _where))
                    if val
                )
                lines.append(
                    "    "
                    + phrase(
                        phrases,
                        "containment_target",
                        "containment target — {target}",
                        target=_target,
                    )
                )

    def _check_line(c, suffix=""):
        """One condition bullet led by its subject; subject omitted when blank."""
        who = str(getattr(c, "subject", "") or "").strip()
        head = f"[{who}] " if who else ""
        body = (
            f"{getattr(c, 'label', '') or getattr(c, 'id', '')}: "
            f"{getattr(c, 'observed', '')} ({getattr(c, 'detail', '')})".rstrip(" ()")
        )
        return f"- {head}{body}{suffix}"

    fails = getattr(brief, "decisive_fails", None) or []
    if isinstance(fails, list) and fails:
        # State the direction: a FAIL here argues AGAINST fraud; naming checks alone invites inversion.
        lines.append(
            "EXCLUSION(S) THAT FAILED — READ THE DIRECTION: each of these argues "
            "AGAINST fraud (this is the procedure's own list of innocent explanations, "
            "and a FAIL means one applies). Do NOT re-describe any of them as evidence "
            "OF fraud, a suspicious pattern, or a fraud indicator. Each bullet is prefixed "
            "with the SUBJECT it was measured on: report every value against that subject "
            "and no other, and do not merge two subjects' values into one sentence:"
        )
        for c in fails:
            kind = getattr(c, "exclusion_kind", "") or ""
            mark = (
                "  [ATTRIBUTED FACT — the verdict rests on this]"
                if kind == "categorical"
                else ""
            )
            lines.append(_check_line(c, mark))
    # Same direction; "not decisive" is about the verdict CLASS only, not the finding's strength.
    explanatory = getattr(brief, "explanatory_fails", None) or []
    if isinstance(explanatory, list) and explanatory:
        lines.append(
            "EXCLUSION(S) THAT FAILED WITHOUT CHANGING THE VERDICT CLASS — READ THE "
            "DIRECTION: each of these also argues AGAINST fraud, and a FAIL means the "
            "innocent explanation WAS FOUND. 'Not decisive' is a statement about the "
            "verdict CLASS only — it does not mean the finding is doubtful, unestablished, "
            "or contradicted by other evidence. Do NOT re-describe any of them as evidence "
            "OF fraud, as a suspicious pattern, or as a contradiction/inconsistency in the "
            "evidence, and do NOT report the values they were decided on as disagreeing "
            "with each other. Each bullet is prefixed with the SUBJECT it was measured on:"
        )
        for c in explanatory:
            lines.append(_check_line(c))
    indicators = getattr(brief, "decisive_indicators", None) or []
    if isinstance(indicators, list) and indicators:
        lines.append(
            "POSITIVE fraud indicator(s) that FAILED (evidence OF fraud). Each bullet is "
            "prefixed with the SUBJECT it was measured on:"
        )
        for c in indicators:
            lines.append(_check_line(c))
        lines.append(
            "NOTE: "
            + phrase(
                phrases,
                "indicator_driven_fraud",
                "an indicator-driven fraud verdict requires EXPERT CONFIRMATION "
                "before containment — recommend review plus a wider sweep for other "
                "assets touched by the same identity, not auto-void/lock/freeze.",
                names="; ".join(
                    getattr(c, "label", "") or getattr(c, "id", "") for c in indicators
                ),
            )
        )
    if getattr(brief, "containment_gated", False) is True:
        lines.append(
            "CONTAINMENT IS GATED for this case: every containment action (void, "
            "refund, lock, suspend, freeze) must be written as PENDING EXPERT "
            "CONFIRMATION. Do NOT write 'immediately void/lock/suspend' anywhere."
        )
    unknowns = getattr(brief, "decisive_unknowns", None) or []
    if isinstance(unknowns, list) and unknowns:
        # Subject-prefixed: a flat list across subjects otherwise conflates separate checks.
        lines.append("Decisive checks with NO DATA (drive INSUFFICIENT):")
        for c in unknowns:
            who = str(getattr(c, "subject", "") or "").strip()
            head = f"[{who}] " if who else ""
            lines.append(f"- {head}{getattr(c, 'label', '') or getattr(c, 'id', '')}")
    # Actor-kind attribution is what a narrator invents most readily; needs its own prohibition.
    unattributed = getattr(brief, "unanswered_attributions", None) or []
    if isinstance(unattributed, list) and unattributed:
        lines.append(
            "UNANSWERED — what kind of party acted was NOT established. These checks "
            "could not be evaluated, so whatever each was there to determine is UNKNOWN "
            "and must not be asserted anywhere in your text, in any wording:"
        )
        for c in unattributed:
            who = str(getattr(c, "subject", "") or "").strip()
            head = f"[{who}] " if who else ""
            lines.append(
                f"- {head}{getattr(c, 'label', '') or getattr(c, 'id', '')}: "
                f"{getattr(c, 'detail', '') or 'could not be evaluated'}"
            )

    # Always shown; "sweep did not run" must not read as "no wider impact found".
    scope_status = getattr(brief, "scope_status", "")
    if isinstance(scope_status, str) and scope_status:
        lines.append(
            f"Scope sweep (is the impact wider than the alert?): {scope_status}"
        )
    extra = getattr(brief, "additional_subjects", None) or []
    if isinstance(extra, list) and extra:
        lines.append(
            "ADDITIONAL impacted subjects found by the sweep, NOT named in the alert "
            "(these MUST appear in the report and the containment scope): "
            + ", ".join(str(s) for s in extra[:25])
        )
    assets = getattr(brief, "impacted_assets", None) or []
    if isinstance(assets, list) and assets:
        lines.append(
            "Impacted assets from the scope sweep "
            "(asset | subject | amount | state | issued):"
        )
        for a in assets[:40]:
            # Out-of-window assets are adjacent business, not incident scope; must be labelled.
            if not getattr(a, "in_window", True):
                flag = (
                    "  <-- OUTSIDE the incident window: NOT incident scope, do not "
                    "count it in the exposure or recommend action on it"
                )
            elif not getattr(a, "known", False):
                flag = "  <-- NEW, not in the alert"
            else:
                flag = ""
            lines.append(
                f"- {getattr(a, 'asset_id', '')} | {getattr(a, 'subject', '')} | "
                f"{getattr(a, 'amount', '')} {getattr(a, 'currency', '')} | "
                f"status={getattr(a, 'status', '')} | "
                f"issued={getattr(a, 'event_date', '') or 'unknown'} | "
                f"by {getattr(a, 'actor', '')}{flag}"
            )

    js = getattr(brief, "join_status", None) or {}
    if isinstance(js, dict) and js:
        lines.append("Correlation join status:")
        for k, v in js.items():
            lines.append(f"- {k}: {v}")

    backbone = getattr(brief, "action_backbone", None) or []
    if isinstance(backbone, list) and backbone:
        lines.append("Recommended next-steps backbone (follow this in intent):")
        for s in backbone:
            lines.append(f"- {s}")

    timeline = getattr(brief, "asset_timeline", None) or []
    if isinstance(timeline, list) and timeline:
        lines.append("Asset-impact timeline (chronological):")
        for e in timeline[:25]:
            lines.append(
                f"- {getattr(e, 'timestamp', '')} {getattr(e, 'event_type', '')} "
                f"{getattr(e, 'entity_type', '')} {getattr(e, 'entity_value', '')} "
                f"by {getattr(e, 'actor', '')} [{getattr(e, 'source', '')}] "
                f"{getattr(e, 'detail', '')}".rstrip()
            )

    # In `lines` not `tail`, before data-quality notes: nothing failed to retrieve here.
    carve_outs = getattr(brief, "unenforced_carve_outs", None) or []
    if isinstance(carve_outs, list) and carve_outs:
        lines.append(
            "Procedure carve-outs this engine does NOT apply — the check named in each "
            "reads more broadly than the procedure does, so state the limitation where you "
            "rely on that check:"
        )
        for c in carve_outs:
            lines.append(f"- {c}")

    # Built before concepts (rendered after): snippets fitted to room left, so the budget cut
    # takes from them rather than from the data blocks, which are not negotiable.
    tail = []
    precedents = getattr(brief, "precedents", None) or []
    if isinstance(precedents, list) and precedents:
        tail.append("Past-investigation precedents (how comparable cases were closed):")
        for p in precedents:
            tail.append(
                f"- {getattr(p, 'case_id', '')} ({getattr(p, 'verdict', '')}, "
                f"{getattr(p, 'subject', '')}): "
                f"{(getattr(p, 'resolution', '') or '').strip()[:200]}"
            )

    notes = getattr(brief, "notes", None) or []
    if isinstance(notes, list) and notes:
        tail.append("Data-quality notes:")
        for n in notes:
            tail.append(f"- {n}")

    concepts = getattr(brief, "concept_refs", None) or []
    if isinstance(concepts, list) and concepts:
        header = "Relevant KB concepts:"
        heads = [
            f"- {getattr(c, 'title', '') or getattr(c, 'concept_id', '')}: "
            for c in concepts
        ]
        snips = [(getattr(c, "snippet", "") or "").replace("\n", " ") for c in concepts]
        # Titles and the tail blocks are costed first; snippets get only what remains.
        room = char_budget - len("\n".join(lines + [header] + heads + tail))
        # Split evenly; unused share from a short snippet goes back to the remaining docs.
        allow = [0] * len(snips)
        left, remaining = room, len(snips)
        for i in sorted(range(len(snips)), key=lambda j: len(snips[j])):
            allow[i] = min(len(snips[i]), max(0, left) // max(1, remaining))
            left -= allow[i]
            remaining -= 1
        lines.append(header)
        for head, snip, n in zip(heads, snips, allow):
            lines.append(head + snip[:n])

    lines.extend(tail)

    text = "\n".join(lines)
    if len(text) > char_budget:
        text = text[:char_budget] + "... [truncated]"
    return text
