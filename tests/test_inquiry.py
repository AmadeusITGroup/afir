"""The open-question lane's two halves, asked directly: what it RAISES and what it SPENDS.

`test_inquiries_never_change_the_verdict.py` owns the one property that makes this lane safe — an
inquiry may not move a label, a condition result or a health score by a byte. That file therefore
asserts a *silence*, and a silence is satisfied by a lane that does nothing at all. This file owns
the other side: given that the lane is allowed to run, **does it raise the questions a pack
declared, does it tell its five outcomes apart, and is every spend really bounded?**

Three distinctions carry most of the file, and each is one this repo has already paid for:

* **A question nobody asked is not a question answered with nothing.** Five states, and the two
  silences (`empty` — asked, no rows; `unanswered` — the source never answered) sit at opposite
  ends of a remedy: one is a finding, the other is a credential or a catalog entry. Collapsing
  them is the failure class the whole lane exists to label.
* **A refusal is a coded row and never a silence.** Every way a probe can be declined has a
  sentence (`inquiry_probe.REFUSAL_NOTES`), including a code this build does not know, because a
  blank note renders as a question nobody declined.
* **A bound shipped at 0 is an untested bound.** The lane ships armed at one probe, so the ceiling
  it shares with the cross-procedure lane, the count clamp, the timeout clamp and the wall-clock
  deadline are all reachable — and each is asserted on its own, because a bound that only fires
  when a second bound also would is a bound nobody has tested.

THE PACK IS `knowledge/mock_domain/`, through a copy that declares the open questions
(`tests/mock_domain_inquiries.py`). Nothing here is domain-specific: the trigger, the five states,
the scope resolution and the bounds are engine behaviour over whatever a pack declares. The
declaration goes onto a copy because the shipped fixture pack is `installed_packs.FIXTURE_PACK`,
so declaring there would be read by the whole shared suite.

The doubles come from `tests/test_link_probe.py` rather than being copied: the two lanes go
through the same retrieval seam, and a second `_Engine` that answers `[]` and `None` slightly
differently would be a second opinion about what a non-answer is.
"""

import asyncio
import logging

import pytest

from src.inquiry import (
    INQUIRY_STATES,
    INQUIRY_TRIGGERS,
    assess_inquiries,
    settle_from_rows_in_hand,
    settle_with_probe,
)
from src.inquiry_probe import (
    DEFAULT_MAX_INQUIRY_PROBES,
    REFUSAL_NOTES,
    build_inquiry_probe,
    inquiry_budget,
    inquiry_refusal,
    inquiry_row_cap,
    run_inquiries,
)
from src.link_escalation import (
    MAX_PROBES_CEILING,
    PROBE_BUDGET_CEILING_SECONDS,
    PROBE_ROW_CAP_DEFAULT,
    PROBE_TIMEOUT_DEFAULT,
    PROBE_TIMEOUT_MAX,
    probe_budget,
    slice_text,
)
from src.models.pydantic_models import (
    ConditionCheck,
    ImpactedAsset,
    InquiryFinding,
    InvestigationBrief,
    SubjectVerdict,
    ValidationVerdict,
)
from tests.mock_domain_inquiries import (
    ABSENT_SCOPE_ENTITY,
    IN_HAND_SOURCE,
    MEANINGS,
    PROBE_SOURCE,
    SCOPE_ENTITY,
    inquiry_pack,
    open_question,
    register_row,
)
from tests.test_link_probe import _Engine, _Generator, _analysis

RULESET = "refund_fraud"

#: The logical name `IN_HAND_SOURCE` resolves to through the ruleset's own `sources:` map. The
#: free rung keys on the RESOLVED name, so a `logs` dict keyed by the logical one would leave the
#: question `not_asked` while looking correct.
IN_HAND_PHYSICAL = "device_sessions"


# --- doubles and builders ---------------------------------------------------------------


def _check(condition_id, result="unknown"):
    return ConditionCheck(id=condition_id, result=result)


def _subject(value="RT48192043", verdict_class="insufficient", checks=(), entity=None):
    return SubjectVerdict(
        subject_type=entity or SCOPE_ENTITY,
        subject_value=value,
        verdict="INSUFFICIENT DATA",
        verdict_class=verdict_class,
        checks=list(checks),
    )


def _verdict(*subjects):
    return ValidationVerdict(label_scheme=RULESET, subjects=list(subjects))


def _triggering_verdict(condition="scope_gate", result="unknown", **over):
    """The commonest shape: one in-scope subject whose named check could not answer."""
    return _verdict(_subject(checks=[_check(condition, result)], **over))


def _declaration(**over):
    """A declaration in the flat shape `pack.open_questions` hands the planner."""
    fields = {
        "id": "q",
        "condition": "scope_gate",
        "result": "unknown",
        "verdict_class": "",
        "source": PROBE_SOURCE,
        "question": "Was a handover recorded for {entity} {value}?",
        "scope_entity": SCOPE_ENTITY,
        "where": [],
        "meaning": dict(MEANINGS),
    }
    fields.update(over)
    return fields


def _open_finding(**over):
    """A finding in exactly the state a probe is for: open, sourced, scope in hand."""
    fields = {
        "id": "was_the_handover_recorded_elsewhere",
        "state": "not_asked",
        "question": "Was a handover recorded for shipment RT48192043?",
        "source": PROBE_SOURCE,
        "scope_entity": SCOPE_ENTITY,
        "scope_values": ["RT48192043"],
    }
    fields.update(over)
    return InquiryFinding(**fields)


def _probe_double(answer=None, calls=None, raises=None, delay=0.0):
    """An injected fetcher, with the same three answers the real one can give.

    `None` is "the source did not answer", a list — including `[]` — is "it answered", and an
    exception is "it could not be asked". The three are separate because the pack declares a
    separate MEANING for each.
    """

    async def _run(source, analysis, **kw):
        if calls is not None:
            calls.append({"source": source, "analysis": analysis, **kw})
        if delay:
            await asyncio.sleep(delay)
        if raises is not None:
            raise raises
        return answer

    return _run


class _ExplodingPack:
    """A pack whose accessor raises. The lane is advisory: it must not fail the stage."""

    def __init__(self, error=None):
        self.error = error or RuntimeError("the ruleset could not be read")

    def open_questions(self, key=""):
        raise self.error

    def ruleset_spec(self, key=""):
        return {"subject_entity": SCOPE_ENTITY}


# --- 1. the planner: which questions get RAISED -----------------------------------------


def test_a_pack_that_declares_no_open_question_produces_nothing(tmp_path):
    """The default posture, and the one every shipped pack is in.

    Asserted first because it is the claim the whole lane rests on: a declaration is the only
    thing that turns this on, so a pack that declares none must be byte-identical to a tree with
    no lane at all.
    """
    pack = inquiry_pack(tmp_path, [], label="empty")
    assert assess_inquiries(pack, _analysis(), {}, _triggering_verdict(), None, RULESET) == []


def test_the_lane_can_be_turned_off_by_configuration_as_well_as_by_silence(tmp_path):
    """`enabled: false` is a second off switch, and it precedes reading the pack at all."""
    pack = inquiry_pack(
        tmp_path, [open_question(mode="probe", condition="scope_gate")], label="disabled"
    )
    verdict = _triggering_verdict()
    armed = assess_inquiries(pack, _analysis(), {}, verdict, None, RULESET, {})
    off = assess_inquiries(
        pack, _analysis(), {}, verdict, None, RULESET, {"enabled": False}
    )
    assert [f.id for f in armed] == ["was_the_handover_recorded_elsewhere"]
    assert off == []


def test_a_pack_object_that_cannot_be_asked_contributes_nothing():
    """Duck-typed: an object with no accessor is not an error, it is a pack without the key."""
    assert assess_inquiries(object(), _analysis(), {}, _triggering_verdict(), None, RULESET) == []
    assert assess_inquiries(None, _analysis(), {}, _triggering_verdict(), None, RULESET) == []


def test_a_declaration_whose_trigger_did_not_fire_raises_NO_question(tmp_path):
    """A question that was not raised is not an open question.

    The alternative — listing every declaration on every run — makes the lane a catalog, and a
    reader who has learned that the section lists catalog entries stops reading it.
    """
    pack = inquiry_pack(
        tmp_path, [open_question(mode="probe", condition="scope_gate")], label="untriggered"
    )
    settled = _verdict(_subject(checks=[_check("scope_gate", "pass")]))
    assert assess_inquiries(pack, _analysis(), {}, settled, None, RULESET) == []
    # And the positive control on the same pack, or "nothing fired" is also what a broken
    # trigger reads like.
    assert len(assess_inquiries(pack, _analysis(), {}, _triggering_verdict(), None, RULESET)) == 1


def test_a_condition_trigger_carries_the_words_a_report_line_prints(tmp_path):
    """The trigger is stated in prose, and `unknown` is the default result.

    `result:` defaults to `unknown` in the accessor deliberately — an open question is what a
    check that could not answer leaves behind — so a declaration that omits it must still fire on
    an UNKNOWN and not on everything.
    """
    pack = inquiry_pack(
        tmp_path, [open_question(mode="probe", condition="served_route")], label="cond"
    )
    (finding,) = assess_inquiries(
        pack, _analysis(), {}, _triggering_verdict(condition="served_route"), None, RULESET
    )
    assert finding.trigger_condition == "served_route"
    assert finding.trigger_result == "unknown"
    assert "served_route" in finding.trigger and "unknown" in finding.trigger
    assert "1 subject(s)" in finding.trigger
    assert finding.advisory_note and "human" in finding.advisory_note
    assert "unknown" in INQUIRY_TRIGGERS


def test_a_verdict_class_trigger_needs_no_condition_at_all(tmp_path):
    """A decisive outcome raises questions too, so the class alone is a trigger."""
    pack = inquiry_pack(
        tmp_path, [open_question(mode="probe", verdict_class="insufficient")], label="class"
    )
    (finding,) = assess_inquiries(
        pack, _analysis(), {}, _verdict(_subject(verdict_class="insufficient")), None, RULESET
    )
    assert finding.trigger_condition == ""
    assert "insufficient" in finding.trigger
    other = assess_inquiries(
        pack, _analysis(), {}, _verdict(_subject(verdict_class="fraud")), None, RULESET
    )
    assert other == []


def test_both_halves_of_a_trigger_must_hold_on_the_SAME_subject(tmp_path):
    """A declaration naming a condition AND a class is asking about one situation, not two.

    The failure this pins is the tempting one: satisfy each half on a different subject and the
    question is raised about a run in which neither subject is in the state described.
    """
    pack = inquiry_pack(
        tmp_path,
        [open_question(mode="probe", condition="served_route", verdict_class="insufficient")],
        label="both",
    )
    split = _verdict(
        _subject("RT1", verdict_class="fraud", checks=[_check("served_route", "unknown")]),
        _subject("RT2", verdict_class="insufficient", checks=[_check("served_route", "pass")]),
    )
    assert assess_inquiries(pack, _analysis(), {}, split, None, RULESET) == []
    together = _verdict(
        _subject("RT1", verdict_class="fraud", checks=[_check("served_route", "unknown")]),
        _subject(
            "RT48192043",
            verdict_class="insufficient",
            checks=[_check("served_route", "unknown")],
        ),
    )
    (finding,) = assess_inquiries(pack, _analysis(), {}, together, None, RULESET)
    assert finding.scope_values == ["RT48192043"]


def test_a_source_this_run_ALREADY_RETRIEVED_settles_the_question_for_free(tmp_path):
    """The free rung, and the cheap half of the whole lane.

    Where the declaration names a source the run already holds, the question is answered by
    reading those rows again — and `probe_spent` stays False, because nothing was spent. That is
    the distinction the two settlement seams exist to keep: a bounded re-use of evidence in hand
    is not the same claim as a query asked for this question.
    """
    entry = open_question(
        mode="free",
        condition="scope_gate",
        where=[{"field": "shipment_code", "any_of": ["RT48192043"]}],
    )
    pack = inquiry_pack(tmp_path, [entry], label="free")
    # Resolved rather than spelled: `IN_HAND_SOURCE` is a LOGICAL name, and four fixtures in this
    # suite have already passed while their conditions read `unknown` against rows filed under a
    # physical name no ruleset asks for. If the map is re-pointed, this line fails here.
    assert (pack.ruleset_spec(RULESET)["sources"] or {})[IN_HAND_SOURCE] == IN_HAND_PHYSICAL
    logs = {IN_HAND_PHYSICAL: [register_row(), register_row(shipment_code="RT99999999")]}
    (finding,) = assess_inquiries(
        pack, _analysis(), logs, _triggering_verdict(), None, RULESET
    )
    assert finding.source == IN_HAND_PHYSICAL
    assert finding.state == "answered"
    assert finding.rows_matched == 1
    assert finding.meaning == MEANINGS["rows"]
    assert finding.probe_spent is False
    assert finding.row_cap_hit is False
    assert "no probe was spent" in finding.probe_note
    assert "in hand" in finding.probe_note


def test_the_free_rung_labels_ZERO_rows_rather_than_reporting_a_count(tmp_path):
    """An empty answer IS an answer, and the pack said what it means here."""
    entry = open_question(
        mode="free",
        condition="scope_gate",
        where=[{"field": "shipment_code", "any_of": ["NOBODY"]}],
    )
    pack = inquiry_pack(tmp_path, [entry], label="free_empty")
    (finding,) = assess_inquiries(
        pack,
        _analysis(),
        {IN_HAND_PHYSICAL: [register_row()]},
        _triggering_verdict(),
        None,
        RULESET,
    )
    assert finding.state == "empty"
    assert finding.rows_matched == 0
    assert finding.meaning == MEANINGS["empty"]


def test_a_question_needing_a_source_this_run_lacks_is_OPEN_and_says_what_it_needs(tmp_path):
    """`not_asked` names the source and whether a budget exists to ask it.

    Two wordings and not one, because "nobody has spent this yet" and "this deployment spends
    nothing here" license different next steps: the first waits for the probe rung, the second is
    a configuration decision.
    """
    pack = inquiry_pack(
        tmp_path, [open_question(mode="probe", condition="scope_gate")], label="open"
    )
    (armed,) = assess_inquiries(pack, _analysis(), {}, _triggering_verdict(), None, RULESET)
    assert armed.state == "not_asked"
    assert PROBE_SOURCE in armed.gap_reason
    assert "has not spent" in armed.gap_reason
    (closed,) = assess_inquiries(
        pack,
        _analysis(),
        {},
        _triggering_verdict(),
        None,
        RULESET,
        {"max_inquiry_probes_per_run": 0},
    )
    assert closed.state == "not_asked"
    assert "no probe budget is configured" in closed.gap_reason


def test_a_question_this_run_holds_no_scope_value_for_is_UNREACHABLE(tmp_path):
    """An unscoped question is not a cheap question, so it is reported and never asked.

    The remedy it names is a pack fix: an unscoped probe scans the source over the whole window
    instead of asking about one identity, which is a cost nobody authorised and an answer about
    everybody.
    """
    pack = inquiry_pack(
        tmp_path, [open_question(mode="unscopable", condition="scope_gate")], label="unscoped"
    )
    (finding,) = assess_inquiries(pack, _analysis(), {}, _triggering_verdict(), None, RULESET)
    assert finding.state == "unreachable"
    assert finding.scope_entity == ABSENT_SCOPE_ENTITY
    assert finding.scope_values == []
    assert ABSENT_SCOPE_ENTITY in finding.gap_reason
    assert "whole window" in finding.gap_reason
    # The state is not a finding about the data, so nothing is claimed about rows.
    assert finding.rows_matched == 0 and finding.meaning == ""


def test_a_declaration_the_ACCESSOR_drops_is_reported_and_not_silently_absent(tmp_path, caplog):
    """A dropped declaration reads exactly like a pack that declared nothing.

    So the drop is a warning naming the id and what was missing. `sourceless` is the mode that
    exercises it: the entry is well-formed apart from having nothing to ask.
    """
    with caplog.at_level(logging.WARNING):
        pack = inquiry_pack(
            tmp_path,
            [open_question(mode="sourceless", condition="scope_gate")],
            label="dropped",
        )
        assert pack.open_questions(RULESET) == []
    assert "DROPPED" in caplog.text
    assert "was_the_handover_recorded_elsewhere" in caplog.text
    assert assess_inquiries(pack, _analysis(), {}, _triggering_verdict(), None, RULESET) == []


def test_the_question_text_substitutes_its_two_placeholders_and_survives_BRACES(tmp_path):
    """`str.replace` and never `.format`: procedure prose contains braces.

    Formatting a pack's sentence raises on the first brace it did not expect, and the exception
    is caught by the lane's own guard — so the whole run's questions vanish because one of them
    mentioned a JSON fragment.
    """
    entry = open_question(
        mode="probe",
        condition="scope_gate",
        question="Any handover for {entity} {value} (a row like {\"a\": 1} counts)?",
    )
    pack = inquiry_pack(tmp_path, [entry], label="braces")
    (finding,) = assess_inquiries(pack, _analysis(), {}, _triggering_verdict(), None, RULESET)
    assert finding.question == (
        'Any handover for shipment RT48192043 (a row like {"a": 1} counts)?'
    )


def test_the_scope_is_the_SUBJECTS_the_trigger_fired_on_before_anything_else(tmp_path):
    """Subjects first, because the question is about them; the incident is the last fallback."""
    pack = inquiry_pack(
        tmp_path, [open_question(mode="probe", condition="scope_gate")], label="scope_subj"
    )
    verdict = _verdict(
        _subject("RT48192043", checks=[_check("scope_gate")]),
        _subject("RT77777777", checks=[_check("scope_gate")]),
        # Another type entirely: the sweep may adjudicate one, but it is not this scope.
        _subject("LDS04", checks=[_check("scope_gate")], entity="depot"),
    )
    (finding,) = assess_inquiries(pack, _analysis(), {}, verdict, None, RULESET)
    assert finding.scope_values == ["RT48192043", "RT77777777"]


def test_the_sweeps_own_subjects_join_the_scope_only_under_the_rulesets_OWN_entity(tmp_path):
    """`brief` is read for the subject entity and for no other type name.

    The sweep's values are typed by the ruleset's subject entity and by nothing else, so reading
    them under another type name asserts a binding the sweep never made — a question asked about
    a shipment code in a field that holds depots.
    """
    subject_scoped = inquiry_pack(
        tmp_path, [open_question(mode="probe", condition="scope_gate")], label="brief_yes"
    )
    brief = InvestigationBrief(
        use_case=RULESET,
        additional_subjects=["RT88888888"],
        impacted_assets=[ImpactedAsset(subject="RT66666666", asset_id="A1")],
    )
    (with_sweep,) = assess_inquiries(
        subject_scoped, _analysis(), {}, _triggering_verdict(), brief, RULESET
    )
    assert with_sweep.scope_values == ["RT48192043", "RT88888888", "RT66666666"]

    other = open_question(mode="probe", condition="scope_gate")
    other["ask"]["scope_entity"] = "depot"
    depot_scoped = inquiry_pack(tmp_path, [other], label="brief_no")
    (without,) = assess_inquiries(
        depot_scoped, _analysis(), {}, _triggering_verdict(), brief, RULESET
    )
    # The trigger fired on a shipment, so no subject carries the depot type; the incident's own
    # entity is what remains, and the sweep's shipment codes are NOT read as depots.
    assert without.scope_entity == "depot"
    assert without.scope_values == ["LDS04"]


def test_the_scope_value_list_is_BOUNDED_so_one_probe_stays_one_query(tmp_path):
    """The probe asks about every value at once, so this bounds the query TEXT.

    Not the number of probes — that is the budget's job. Two limits on one lane, and conflating
    them is how a bounded lane produces one unbounded query.
    """
    pack = inquiry_pack(
        tmp_path, [open_question(mode="probe", condition="scope_gate")], label="bounded"
    )
    many = _verdict(
        *[_subject(f"RT{i:08d}", checks=[_check("scope_gate")]) for i in range(40)]
    )
    (finding,) = assess_inquiries(pack, _analysis(), {}, many, None, RULESET)
    assert len(finding.scope_values) == 20
    assert finding.scope_values[0] == "RT00000000"


def test_the_questions_are_ordered_ACTIONABLE_FIRST_and_ties_break_on_the_id(tmp_path):
    """A reader who stops halfway has seen every question with something to say.

    The order is `INQUIRY_STATES` itself, so adding a state means deciding where it sits rather
    than discovering later that it sorted last.
    """
    entries = [
        open_question(question_id="z_unscoped", mode="unscopable", condition="scope_gate"),
        open_question(question_id="m_open", mode="probe", condition="scope_gate"),
        open_question(
            question_id="b_free",
            mode="free",
            condition="scope_gate",
            where=[{"field": "shipment_code", "any_of": ["RT48192043"]}],
        ),
        open_question(question_id="a_open", mode="probe", condition="scope_gate"),
    ]
    pack = inquiry_pack(tmp_path, entries, label="ordered")
    findings = assess_inquiries(
        pack,
        _analysis(),
        {IN_HAND_PHYSICAL: [register_row()]},
        _triggering_verdict(),
        None,
        RULESET,
    )
    assert [f.state for f in findings] == [
        "answered",
        "not_asked",
        "not_asked",
        "unreachable",
    ]
    # Two questions in one state sort by id, so the report's order does not depend on the
    # declaration order of things the reader cannot see.
    assert [f.id for f in findings] == ["b_free", "a_open", "m_open", "z_unscoped"]
    order = [INQUIRY_STATES.index(f.state) for f in findings]
    assert order == sorted(order)


def test_the_assessment_NEVER_fails_the_stage_it_advises_on(caplog):
    """An advisory lane that can break the run it advises on is worse than no lane.

    Logged at ERROR, though, because an empty list is indistinguishable from a pack that declared
    nothing and those two need opposite responses.
    """
    with caplog.at_level(logging.ERROR):
        assert (
            assess_inquiries(
                _ExplodingPack(), _analysis(), {}, _triggering_verdict(), None, RULESET
            )
            == []
        )
    assert "advisory" in caplog.text.lower()
    assert "EMPTY" in caplog.text


def test_the_assessment_is_DETERMINISTIC_over_the_same_inputs(tmp_path):
    """Same evidence, same questions, in the same order — the spine's own contract."""
    entries = [
        open_question(question_id="a", mode="probe", condition="scope_gate"),
        open_question(question_id="b", mode="unscopable", condition="scope_gate"),
    ]
    pack = inquiry_pack(tmp_path, entries, label="determinism")
    args = (pack, _analysis(), {}, _triggering_verdict(), None, RULESET)
    first = [f.model_dump() for f in assess_inquiries(*args)]
    second = [f.model_dump() for f in assess_inquiries(*args)]
    assert first == second and len(first) == 2


# --- 2. the settlement: three answers, and what each one may CLAIM ----------------------


def test_a_source_that_did_not_ANSWER_is_not_a_source_that_answered_with_nothing():
    """The distinction the whole lane is organised around, at the seam that decides it.

    `rows is None` is the non-answer and it names a credential or a catalog entry as the remedy;
    `[]` is an answer and the pack's own sentence for it is stamped instead. Collapsing them
    turns an environment failure into a finding about the subject.
    """
    declaration = _declaration()
    absent = _open_finding()
    settle_with_probe(absent, declaration, rows=None, source=PROBE_SOURCE)
    empty = _open_finding()
    settle_with_probe(empty, declaration, rows=[], source=PROBE_SOURCE)

    assert absent.state == "unanswered"
    assert absent.meaning == MEANINGS["unanswered"]
    assert "did not answer this question at all" in absent.gap_reason
    assert "credential or a catalog entry" in absent.gap_reason
    assert absent.rows_matched == 0

    assert empty.state == "empty"
    assert empty.meaning == MEANINGS["empty"]
    assert empty.rows_matched == 0
    assert empty.gap_reason == ""
    # The pair is the assertion: either half alone passes on a seam that answers both the same.
    assert absent.state != empty.state and absent.meaning != empty.meaning


def test_rows_the_declarations_own_selector_keeps_are_what_is_COUNTED():
    """`where` is the scope of the question, so the count is after it — and only after it."""
    declaration = _declaration(
        where=[{"field": "shipment_code", "any_of": ["RT48192043"]}]
    )
    finding = _open_finding()
    settle_with_probe(
        finding,
        declaration,
        rows=[register_row(), register_row(shipment_code="RT00000001"), register_row()],
        source=PROBE_SOURCE,
    )
    assert finding.state == "answered"
    assert finding.rows_matched == 2
    assert finding.meaning == MEANINGS["rows"]
    assert finding.probe_spent is True


def test_the_row_cap_is_measured_on_what_came_BACK_and_not_on_what_was_kept():
    """A selector that keeps two of a capped hundred still owes the reader "and there were more".

    The cap bounds the RETRIEVAL, so it is a fact about the answer's completeness and not about
    the selector — measured pre-selector or a narrow `where` silently launders a truncated read
    into an exact count.
    """
    declaration = _declaration(
        where=[{"field": "shipment_code", "any_of": ["RT48192043"]}]
    )
    rows = [register_row(shipment_code=f"RT{i:08d}") for i in range(4)] + [register_row()]
    capped = _open_finding()
    settle_with_probe(capped, declaration, rows=rows, source=PROBE_SOURCE, row_cap=5)
    assert capped.row_cap_hit is True
    assert capped.rows_matched == 1

    under = _open_finding()
    settle_with_probe(under, declaration, rows=rows, source=PROBE_SOURCE, row_cap=50)
    assert under.row_cap_hit is False
    assert under.rows_matched == 1


def test_a_selector_that_cannot_be_APPLIED_claims_nothing_about_the_rows_it_has():
    """Reading every row instead would answer a different question, confidently.

    So the spend is recorded — the query really was asked — and the state is the non-answer, with
    a reason that points at the declaration rather than at the data.
    """
    finding = _open_finding()
    settle_with_probe(
        finding,
        _declaration(where=7),
        rows=[register_row()],
        source=PROBE_SOURCE,
        note="one probe was spent",
    )
    assert finding.state == "unanswered"
    assert finding.rows_matched == 0
    assert finding.meaning == MEANINGS["unanswered"]
    assert "row selector could not be applied" in finding.gap_reason
    assert finding.probe_spent is True
    assert finding.probe_note == "one probe was spent"


def test_the_free_rung_claims_neither_a_SPEND_nor_a_CAP_of_its_own():
    """Two named seams rather than a `spent` flag, so a call site cannot mean both.

    The cap that bounded rows in hand belongs to the run's own retrieval and is reported where
    the run reports its sources; claiming it here would report one truncation twice, as though
    this reading had asked for the rows.
    """
    declaration = _declaration()
    rows = [register_row() for _ in range(200)]
    free = _open_finding()
    settle_from_rows_in_hand(free, declaration, rows, source=IN_HAND_PHYSICAL, note="free")
    assert free.state == "answered"
    assert free.rows_matched == 200
    assert free.probe_spent is False
    assert free.row_cap_hit is False
    assert free.source == IN_HAND_PHYSICAL

    probed = _open_finding()
    settle_with_probe(probed, declaration, rows=rows, source=PROBE_SOURCE, row_cap=200)
    assert probed.probe_spent is True and probed.row_cap_hit is True


def test_a_declaration_with_no_MEANING_leaves_the_meaning_empty_and_not_invented():
    """The engine has no sentence of its own to fall back to, and must not acquire one.

    `pack.open_questions` drops an entry missing a meaning, so this is the shape a hand-built
    declaration reaches: nothing is claimed rather than a plausible default being printed as the
    procedure's own words.
    """
    finding = _open_finding()
    settle_with_probe(finding, {"id": "q"}, rows=[register_row()], source=PROBE_SOURCE)
    assert finding.state == "answered"
    assert finding.rows_matched == 1
    assert finding.meaning == ""


# --- 3. the bounds: one ceiling shared with the cross-procedure lane --------------------


def test_the_default_lane_is_ARMED_and_fits_beside_the_link_lanes_own_budget():
    """Narrow but reachable: a bound shipped at 0 makes every refusal code untestable.

    The arithmetic is the assertion, because it is what makes the default reachable at all: the
    link lane's BUDGETED worst case is 240s of the 900s ceiling, so one 120s inquiry probe fits
    with room to spare — and the ceiling still holds over both lanes together.
    """
    link = probe_budget({})
    budget = inquiry_budget({}, {})
    assert budget["max_probes"] == DEFAULT_MAX_INQUIRY_PROBES == 1
    assert budget["timeout"] == PROBE_TIMEOUT_DEFAULT
    assert budget["deadline_seconds"] == PROBE_TIMEOUT_DEFAULT
    assert link["deadline_seconds"] + budget["deadline_seconds"] <= PROBE_BUDGET_CEILING_SECONDS


def test_a_link_lane_that_spends_the_whole_CEILING_closes_this_one():
    """One ceiling for both advisory lanes, so raising a bound raises it in one place.

    Budgeted rather than actually spent, deliberately: a bound whose value depends on how slow
    the other lane happened to be is not a bound anybody can state in advance.
    """
    greedy = {"max_probes_per_run": 3, "probe_timeout_seconds": 300}
    assert probe_budget(greedy)["deadline_seconds"] == PROBE_BUDGET_CEILING_SECONDS
    budget = inquiry_budget({}, greedy)
    assert budget["max_probes"] == 0
    assert budget["deadline_seconds"] == 0


def test_a_narrower_ceiling_shortens_the_slice_before_it_closes_the_lane():
    """Between "fits" and "closed" there is a third answer, and it is the useful one."""
    tight = {"max_probes_per_run": 8, "probe_timeout_seconds": 600}
    room = PROBE_BUDGET_CEILING_SECONDS - probe_budget(tight)["deadline_seconds"]
    budget = inquiry_budget({}, tight)
    assert 0 < budget["max_probes"]
    assert budget["timeout"] < PROBE_TIMEOUT_DEFAULT
    assert budget["deadline_seconds"] <= room


def test_only_a_count_at_or_below_zero_closes_the_lane():
    """`0` and a negative disarm it; `1` does not, which is the direction that gets broken."""
    assert inquiry_budget({"max_inquiry_probes_per_run": 0}, {})["max_probes"] == 0
    assert inquiry_budget({"max_inquiry_probes_per_run": -4}, {})["max_probes"] == 0
    assert inquiry_budget({"max_inquiry_probes_per_run": 1}, {})["max_probes"] == 1
    assert inquiry_budget({"max_inquiry_probes_per_run": 3}, {})["max_probes"] == 3


def test_both_bounds_are_CLAMPED_to_the_ceilings_the_two_lanes_share():
    """A configured number is a request, not a licence."""
    counted = inquiry_budget({"max_inquiry_probes_per_run": 99}, {})
    assert counted["max_probes"] == MAX_PROBES_CEILING
    assert counted["deadline_seconds"] <= PROBE_BUDGET_CEILING_SECONDS

    timed = inquiry_budget({"probe_timeout_seconds": 5000}, {})
    assert timed["timeout"] == PROBE_TIMEOUT_MAX


def test_an_unreadable_bound_falls_back_and_NEVER_widens():
    """A typo must cost the default, never the ceiling — the direction a bad parse must fail in."""
    default = inquiry_budget({}, {})
    for garbage in ("lots", None, [], {"a": 1}, "  "):
        assert inquiry_budget({"max_inquiry_probes_per_run": garbage}, {}) == default
        assert inquiry_budget({"probe_timeout_seconds": garbage}, {}) == default
    assert inquiry_budget({"probe_timeout_seconds": 0}, {}) == default


def test_the_row_cap_comes_from_config_else_the_shared_default():
    """Rows one probe may hand to the settlement — a different limit from every timeout."""
    assert inquiry_row_cap({}) == PROBE_ROW_CAP_DEFAULT
    assert inquiry_row_cap(None) == PROBE_ROW_CAP_DEFAULT
    assert inquiry_row_cap({"probe_row_cap": 25}) == 25
    for garbage in (0, -1, "many", None):
        assert inquiry_row_cap({"probe_row_cap": garbage}) == PROBE_ROW_CAP_DEFAULT


# --- 4. every refusal is a CODE, and every code has a sentence --------------------------


def test_each_way_a_question_cannot_be_asked_has_its_own_code():
    """Five situations, five codes — and the source is checked before everything else.

    Coded rather than a single "not asked", because the remedies differ: a declaration fix, a
    reading already done, a spend already made, a scope the run does not hold.
    """
    assert inquiry_refusal(_open_finding(source=""), {}) == (None, "no_source")
    assert inquiry_refusal(_open_finding(state="answered"), {}) == (None, "not_open")
    assert inquiry_refusal(_open_finding(state="empty"), {}) == (None, "not_open")
    assert inquiry_refusal(_open_finding(probe_spent=True), {}) == (None, "already_asked")
    assert inquiry_refusal(_open_finding(scope_values=[]), {}) == (None, "no_scope_value")
    assert inquiry_refusal(_open_finding(), {}) == (PROBE_SOURCE, "")


def test_every_refusal_code_the_lane_can_produce_carries_a_SENTENCE():
    """A code with no sentence renders as a blank line, which reads as nothing declining it."""
    produced = {"no_source", "not_open", "already_asked", "no_scope_value"}
    spent = {"budget_spent", "lane_closed"}
    assert produced | spent <= set(REFUSAL_NOTES)
    for code, note in REFUSAL_NOTES.items():
        assert note.startswith("no probe was spent"), code
        assert len(note) > 60, code


@pytest.mark.asyncio
async def test_a_code_this_BUILD_has_no_sentence_for_still_renders_as_a_refusal(
    tmp_path, monkeypatch
):
    """The fallback sentence, which exists because the table and the caller can drift.

    Asserted through `run_inquiries` rather than on the private helper: the property is that no
    path through the lane can leave a declined question with an empty note. The pack has to be a
    real one here — a stand-in whose accessor raises stops the lane one guard EARLIER, so the
    per-question refusal is never reached and the test would pass on a note nobody wrote.
    """
    monkeypatch.setattr("src.inquiry_probe.REFUSAL_NOTES", {}, raising=True)
    pack = inquiry_pack(
        tmp_path, [open_question(mode="probe", condition="scope_gate")], label="no_sentence"
    )
    finding = _open_finding(source="")
    spent = await run_inquiries(
        _probe_double(answer=[]),
        [finding],
        pack=pack,
        config={},
        link_config={},
        ruleset_key=RULESET,
    )
    assert spent == 0
    assert finding.probe_note
    assert "no_source" in finding.probe_note
    assert "no sentence for" in finding.probe_note


# --- 5. the spend: run_inquiries -------------------------------------------------------


@pytest.mark.asyncio
async def test_with_no_probe_wired_nothing_is_spent_and_nothing_is_CLAIMED():
    """A deployment with no retriever is the commonest one in a test, and in a laptop run."""
    finding = _open_finding()
    assert await run_inquiries(None, [finding], pack=object()) == 0
    assert await run_inquiries(_probe_double(), [], pack=object()) == 0
    assert await run_inquiries(_probe_double(), [finding], pack=None) == 0
    assert finding.state == "not_asked"
    assert finding.probe_note == ""


@pytest.mark.asyncio
async def test_a_closed_lane_says_so_on_every_question_it_could_have_asked():
    """And on none of the others: a settled question was not declined by a bound.

    The failure this pins is the tidy one — decline everything — which tells a reader that the
    question they can already see the answer to was refused for lack of budget.
    """
    askable = _open_finding(id="open")
    settled = _open_finding(id="done", state="empty")
    spent = await run_inquiries(
        _probe_double(answer=[]),
        [askable, settled],
        pack=object(),
        config={"max_inquiry_probes_per_run": 0},
        link_config={},
    )
    assert spent == 0
    assert askable.probe_note == REFUSAL_NOTES["lane_closed"]
    assert "ceiling" in askable.probe_note
    assert settled.probe_note == ""


@pytest.mark.asyncio
async def test_one_probe_asks_the_declared_source_about_the_scope_it_holds(tmp_path):
    """What the probe SENDS, and that the answer lands through the settlement seam."""
    pack = inquiry_pack(
        tmp_path, [open_question(mode="probe", condition="scope_gate")], label="spend"
    )
    calls = []
    finding = _open_finding()
    spent = await run_inquiries(
        _probe_double(answer=[register_row()], calls=calls),
        [finding],
        pack=pack,
        analysis=_analysis(),
        config={},
        link_config={},
        ruleset_key=RULESET,
    )
    assert spent == 1
    assert len(calls) == 1
    assert calls[0]["source"] == PROBE_SOURCE
    assert calls[0]["question"] == finding.question
    assert calls[0]["scope_entity"] == SCOPE_ENTITY
    assert calls[0]["scope_values"] == ["RT48192043"]
    assert calls[0]["row_cap"] == PROBE_ROW_CAP_DEFAULT
    assert 0 < calls[0]["timeout"] <= PROBE_TIMEOUT_DEFAULT
    assert finding.state == "answered"
    assert finding.meaning == MEANINGS["rows"]
    assert finding.probe_spent is True
    assert "one probe of" in finding.probe_note and "1 row(s)" in finding.probe_note


@pytest.mark.asyncio
async def test_an_EMPTY_answer_is_settled_as_an_answer_and_costs_its_probe(tmp_path):
    """Zero rows from a query that ran is the pack's declared `empty` meaning, not a gap."""
    pack = inquiry_pack(
        tmp_path, [open_question(mode="probe", condition="scope_gate")], label="spend_empty"
    )
    finding = _open_finding()
    spent = await run_inquiries(
        _probe_double(answer=[]),
        [finding],
        pack=pack,
        config={},
        link_config={},
        ruleset_key=RULESET,
    )
    assert spent == 1
    assert finding.state == "empty"
    assert finding.meaning == MEANINGS["empty"]
    assert "0 row(s)" in finding.probe_note


@pytest.mark.asyncio
async def test_a_source_that_never_answered_costs_no_COUNT_and_still_costs_TIME(tmp_path):
    """The count is not the bound here — the wall clock is, and that is deliberate.

    A non-answer is not recorded as a spend, mirroring the link lane's rung 3, because nothing
    was learned; so the count cannot stop a run whose every source is unreachable. The deadline
    can, and does: `deadline_seconds` is checked before each probe, so N dead sources cost one
    budget's worth of seconds rather than N.
    """
    pack = inquiry_pack(
        tmp_path, [open_question(mode="probe", condition="scope_gate")], label="spend_dead"
    )
    calls = []
    first, second = _open_finding(), _open_finding()
    spent = await run_inquiries(
        _probe_double(answer=None, calls=calls),
        [first, second],
        pack=pack,
        config={"max_inquiry_probes_per_run": 1},
        link_config={},
        ruleset_key=RULESET,
    )
    assert spent == 0
    assert len(calls) == 2
    for finding in (first, second):
        assert finding.state == "unanswered"
        assert finding.meaning == MEANINGS["unanswered"]
        assert finding.probe_spent is False
        assert "did not answer at all" in finding.probe_note
    assert inquiry_budget({"max_inquiry_probes_per_run": 1}, {})["deadline_seconds"] > 0


@pytest.mark.asyncio
async def test_a_probe_that_RAISES_reports_the_exception_TYPE_and_never_its_message(
    tmp_path, caplog
):
    """A backend error may carry this incident's identifiers, so the message is not printed."""
    pack = inquiry_pack(
        tmp_path, [open_question(mode="probe", condition="scope_gate")], label="spend_raise"
    )
    finding = _open_finding()
    with caplog.at_level(logging.WARNING):
        spent = await run_inquiries(
            _probe_double(raises=ValueError("RT48192043 is not a valid identifier")),
            [finding],
            pack=pack,
            config={},
            link_config={},
            ruleset_key=RULESET,
        )
    assert spent == 0
    assert finding.state == "unanswered"
    assert "ValueError" in finding.probe_note
    assert "not a valid identifier" not in finding.probe_note
    assert "not a valid identifier" not in caplog.text


@pytest.mark.asyncio
async def test_a_probe_that_OVERRUNS_its_slice_is_cancelled_and_says_which_slice(tmp_path):
    """The advisory budget is deliberately far below the system of record's own.

    So the note says a question needing longer needs a condition of its own — the remedy is the
    pack, not a larger advisory bound.
    """
    pack = inquiry_pack(
        tmp_path, [open_question(mode="probe", condition="scope_gate")], label="spend_slow"
    )
    finding = _open_finding()
    spent = await run_inquiries(
        _probe_double(answer=[register_row()], delay=5.0),
        [finding],
        pack=pack,
        config={"probe_timeout_seconds": 1},
        link_config={},
        ruleset_key=RULESET,
    )
    assert spent == 0
    assert finding.state == "unanswered"
    assert "did not answer within its" in finding.probe_note
    assert "slice" in finding.probe_note
    assert "a condition of its own" in finding.probe_note


def test_a_nearly_EXHAUSTED_deadline_does_not_report_itself_as_a_disarmed_lane():
    """The last probe of a spent budget gets whatever is left, and `int()` calls that zero.

    Two facts under one sentence — a lane configured to spend nothing, and a lane that has spent
    nearly everything — whose remedies are opposite: a larger budget against fewer questions. So
    a sub-second slice keeps a decimal, floored at a tenth so one more decimal place cannot
    reintroduce the same collapse, and only a genuine zero prints as one.
    """
    assert slice_text(0) == "0s"
    assert slice_text(-1) == "0s"
    assert slice_text(120) == "120s"
    assert slice_text(1) == "1s"
    assert slice_text(1.9) == "1s"
    assert slice_text(0.4) == "0.4s"
    assert slice_text(0.04) == "0.1s"
    # The property the three cases above exist for, stated once: nothing but a real zero says 0s.
    for remainder in (0.001, 0.04, 0.4, 0.999):
        assert slice_text(remainder) != "0s"
        assert not slice_text(remainder).startswith("0.0")


@pytest.mark.asyncio
async def test_the_second_question_is_declined_by_the_BUDGET_and_keeps_its_own_state(tmp_path):
    """The count bound, asserted independently of the deadline and of the shared ceiling.

    A bound that only fires when a second bound also would is a bound nobody has tested.
    """
    entries = [
        open_question(question_id="a", mode="probe", condition="scope_gate"),
        open_question(question_id="b", mode="probe", condition="scope_gate"),
    ]
    pack = inquiry_pack(tmp_path, entries, label="spend_budget")
    calls = []
    first, second = _open_finding(id="a"), _open_finding(id="b")
    spent = await run_inquiries(
        _probe_double(answer=[register_row()], calls=calls),
        [first, second],
        pack=pack,
        config={"max_inquiry_probes_per_run": 1},
        link_config={},
        ruleset_key=RULESET,
    )
    assert spent == 1
    assert len(calls) == 1
    assert first.state == "answered"
    assert second.state == "not_asked"
    assert second.probe_note == REFUSAL_NOTES["budget_spent"]


@pytest.mark.asyncio
async def test_a_capped_probe_answer_says_its_count_is_a_FLOOR(tmp_path):
    """`N rows` and `N rows, and there were more` are different findings."""
    pack = inquiry_pack(
        tmp_path, [open_question(mode="probe", condition="scope_gate")], label="spend_cap"
    )
    finding = _open_finding()
    await run_inquiries(
        _probe_double(answer=[register_row() for _ in range(9)]),
        [finding],
        pack=pack,
        config={"probe_row_cap": 3},
        link_config={},
        ruleset_key=RULESET,
    )
    assert finding.rows_matched == 3
    assert finding.row_cap_hit is True
    assert "capped at 3" in finding.probe_note


@pytest.mark.asyncio
async def test_a_question_the_pack_NO_LONGER_declares_is_declined_and_never_settled(tmp_path):
    """A finding can outlive its declaration — a resumed run, a pack edited between passes.

    Settling it would need a meaning nobody declared, which is the unlabelled count this lane
    exists to prevent, so it is declined instead and the probe is never asked.
    """
    pack = inquiry_pack(
        tmp_path, [open_question(question_id="still_here", mode="probe", condition="scope_gate")],
        label="spend_gone",
    )
    calls = []
    stale = _open_finding(id="retired_last_release")
    spent = await run_inquiries(
        _probe_double(answer=[register_row()], calls=calls),
        [stale],
        pack=pack,
        config={},
        link_config={},
        ruleset_key=RULESET,
    )
    assert spent == 0
    assert calls == []
    assert stale.state == "not_asked"
    assert stale.probe_note == REFUSAL_NOTES["no_source"]


@pytest.mark.asyncio
async def test_open_questions_that_cannot_be_RE_READ_spend_nothing(caplog):
    """The declarations are re-read at spend time, and a pack that cannot answer stops the lane."""
    finding = _open_finding()
    with caplog.at_level(logging.WARNING):
        spent = await run_inquiries(
            _probe_double(answer=[register_row()]),
            [finding],
            pack=_ExplodingPack(TypeError("bad ruleset")),
            config={},
            link_config={},
            ruleset_key=RULESET,
        )
    assert spent == 0
    assert finding.state == "not_asked"
    assert "TypeError" in caplog.text
    assert "bad ruleset" not in caplog.text


# --- 6. build_inquiry_probe: the query goes through the operator seam -------------------


def test_no_probe_is_built_without_both_halves():
    """A generator with no engine cannot ask, and an engine with no generator cannot build."""
    assert build_inquiry_probe(None, _Engine(answer=[])) is None
    assert build_inquiry_probe(_Generator(), None) is None
    assert build_inquiry_probe(None, None) is None
    assert build_inquiry_probe(_Generator(), _Engine(answer=[])) is not None


@pytest.mark.asyncio
async def test_the_probe_goes_through_the_OPERATOR_SEAM_and_not_around_it():
    """`build_manual_query` is the seam a human naming a source already takes.

    Which entities a source can bind, which window applies and which guards the text must pass
    are all pack knowledge, so a probe assembling its own query would be a second answer to
    three questions this system answers once — and the ten `_attach_query_guards` guarantees come
    free only on this route.
    """
    generator = _Generator(known=(PROBE_SOURCE,))
    engine = _Engine(answer=[register_row()])
    probe = build_inquiry_probe(generator, engine)
    analysis = _analysis()
    rows = await probe(
        PROBE_SOURCE,
        analysis,
        question="Was a handover recorded?",
        scope_entity=SCOPE_ENTITY,
        scope_values=["RT99999999"],
    )
    assert rows == [register_row()]
    assert len(generator.calls) == 1
    assert generator.calls[0]["source"] == PROBE_SOURCE
    assert generator.calls[0]["question"] == "Was a handover recorded?"
    assert len(engine.calls) == 1


@pytest.mark.asyncio
async def test_the_scope_rides_on_a_COPY_of_the_understanding():
    """The stages after correlation read the shared analysis, so it is not rescoped in place.

    Appended and not substituted, either: a probe still needs the incident's window and scope.
    """
    generator = _Generator(known=(PROBE_SOURCE,))
    probe = build_inquiry_probe(generator, _Engine(answer=[]))
    analysis = _analysis()
    before = [(e.type, e.value) for e in analysis.extracted_entities]
    await probe(
        PROBE_SOURCE, analysis, scope_entity=SCOPE_ENTITY, scope_values=["RT99999999"]
    )
    scoped = generator.calls[0]["analysis"]
    assert scoped is not analysis
    assert [(e.type, e.value) for e in analysis.extracted_entities] == before
    assert (SCOPE_ENTITY, "RT99999999") in [
        (e.type, e.value) for e in scoped.extracted_entities
    ]
    assert set(before) <= {(e.type, e.value) for e in scoped.extracted_entities}


@pytest.mark.asyncio
async def test_a_query_that_cannot_be_BUILT_is_a_non_answer_and_not_an_empty_one():
    """"Could not ask" is an answer, and it is the one that must not read as "nothing found"."""
    unbuildable = build_inquiry_probe(_Generator(known=()), _Engine(answer=[register_row()]))
    assert await unbuildable(PROBE_SOURCE, _analysis()) is None
    raising = build_inquiry_probe(
        _Generator(raises=RuntimeError("no retriever")), _Engine(answer=[register_row()])
    )
    assert await raising(PROBE_SOURCE, _analysis()) is None


@pytest.mark.asyncio
async def test_an_ABSENT_source_and_one_answering_with_nothing_come_back_differently():
    """The same line `_gather` draws with `unanswered_out`, drawn again one lane over.

    `_Engine(answer=None)` leaves the source out of the returned dict and names it unanswered —
    the non-answer, `None`. `answer=[]` returns it present and empty — the answer, `[]`.
    """
    absent = build_inquiry_probe(_Generator(known=(PROBE_SOURCE,)), _Engine(answer=None))
    empty = build_inquiry_probe(_Generator(known=(PROBE_SOURCE,)), _Engine(answer=[]))
    assert await absent(PROBE_SOURCE, _analysis()) is None
    assert await empty(PROBE_SOURCE, _analysis()) == []


@pytest.mark.asyncio
async def test_the_probes_own_ROW_CAP_bounds_what_reaches_the_settlement():
    """Bounded at the fetcher too, not only at the caller: two readers, one limit."""
    engine = _Engine(answer=[register_row(shipment_code=f"RT{i:08d}") for i in range(7)])
    probe = build_inquiry_probe(_Generator(known=(PROBE_SOURCE,)), engine)
    rows = await probe(PROBE_SOURCE, _analysis(), row_cap=2)
    assert len(rows) == 2
    assert await probe(PROBE_SOURCE, _analysis(), row_cap=50) is not None
    assert len(await probe(PROBE_SOURCE, _analysis(), row_cap=50)) == 7
