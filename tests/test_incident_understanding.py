"""Deterministic post-checks over what the understanding stage produced.

These exist because a prompt instruction cannot be relied on to beat another. The window
check asymmetry: reading a time expression that is not one leaves the generated window;
reading none where the text states one overwrites it. So the vocabulary is generous.

The grouping check recovers which values arrived together. The asymmetry inverts: a group
that is not a record ANDs values that never co-occurred, so the vocabulary is strict.

The requested_links check requires the text to support every token of the ask. It can only
report, not edit: the model writes `incident_summary` and the engine cannot edit prose.
"""

from types import SimpleNamespace

from src.incident_understanding import (
    _TIME_EXPRESSION_RE,
    IncidentUnderstandingModule,
    _is_traceable_ask,
)
from src.models.pydantic_models import EventWindow, ExtractedEntity, IncidentAnalysis


def _module():
    """No LLM, no pack: both post-checks read the analysis and the incident text only."""
    return IncidentUnderstandingModule(llm_client=None)


def _analysis(start="2026-08-17T00:00:00Z", end="2026-08-17T09:31:00Z"):
    return IncidentAnalysis(
        incident_summary="s",
        severity_reasoning="r",
        impact_assessment="i",
        key_investigation_areas=[],
        log_sources_to_review=[],
        initial_hypotheses=[],
        recommended_actions=[],
        stakeholder_notification=[],
        event_time=EventWindow(start=start, end=end) if start else None,
    )


def _incident(description, **extra):
    return {"id": "INC-1", "description": description, **extra}


def _ent(type_, value, form="", raw=""):
    return ExtractedEntity(type=type_, value=value, value_form=form, raw=raw)


def _grouped(text, *entities):
    """Run the grouper over one description and return the entities it stamped."""
    analysis = _analysis(start=None)
    analysis.extracted_entities = list(entities)
    _module()._group_co_occurring_entities(analysis, _incident(text))
    return list(entities)


def _labels(entities):
    return [e.co_occurrence for e in entities]


def test_a_window_parsed_from_a_text_that_states_no_time_is_DISCARDED():
    """The measured case, and it is the run's own clock arriving as the incident's window.

    What replaces it is not this module's business — it is a declared default depth, applied
    where the window becomes a query (`ApiCallGenerator._default_window`). Here the only job is
    to stop an invented window from being authoritative, because a window the incident did not
    state is indistinguishable downstream from one it did.
    """
    module, analysis = _module(), _analysis()
    module._drop_invented_event_window(
        analysis,
        _incident(
            "Abusive access detected. Office LLL1M13YZ read records belonging to office "
            "DACSV0994 outside its own scope. Please investigate."
        ),
    )
    assert analysis.event_time is None
    # A window that was already absent is left absent, and nothing is invented here either.
    empty = _analysis(start=None)
    module._drop_invented_event_window(empty, _incident("no time anywhere"))
    assert empty.event_time is None


def test_a_STATED_time_keeps_the_window_in_every_shape_the_text_can_state_it():
    """Every accepted shape is a shape a real incident description uses.

    This is the direction that must not fire, so it is asserted per shape rather than on one
    representative: overwriting a stated window is strictly worse than leaving an invented one,
    which is why the vocabulary is deliberately generous and why each entry needs its own case.
    """
    module = _module()
    for text in (
        "Suspicious access on 2026-08-15 from an unknown terminal.",
        "Reported 15/08/2026 by the security desk.",
        "The reads happened around 14:05 local time.",
        "Alert fired at 1755244800 (epoch seconds).",
        "Ongoing since the August release.",
        "Activity observed in 2025 and never closed.",
        "The office called yesterday about it.",
        "Repeated reads over the last 3 weeks.",
        "First seen 6 days ago.",
        "Access continued overnight.",
        "Reported on Monday by the duty analyst.",
        "Between 2 and 4 records were read each hour.",
    ):
        analysis = _analysis()
        module._drop_invented_event_window(analysis, _incident(text))
        assert analysis.event_time is not None, text
        assert _TIME_EXPRESSION_RE.search(text), text


def test_the_time_expression_is_looked_for_in_EVERY_field_that_carries_the_TEXT():
    """An incident's prose does not always arrive under `description`.

    Input structure is not a contract — the same report reaches this stage as a title, a summary
    or a raw blob depending on who forwarded it — so a stated date under any of them keeps the
    window. Reading one key only would discard a correct window on a shape the pipeline already
    accepts, which is the one direction forbidden here.
    """
    module = _module()
    for key in ("description", "title", "summary", "raw_text"):
        analysis = _analysis()
        module._drop_invented_event_window(
            analysis, {"id": "INC-1", key: "unauthorised reads on 2026-08-15"}
        )
        assert analysis.event_time is not None, key
    # ...and a field carrying no text at all is not a crash: a `None`, a number and a nested
    # structure all reach this stage from one ingestion route or another.
    analysis = _analysis()
    module._drop_invented_event_window(
        analysis, {"id": "INC-1", "description": None, "title": 42, "summary": {"a": 1}}
    )
    assert analysis.event_time is None


def test_a_model_that_cannot_be_written_is_reported_and_never_raises():
    """This is a post-check, not a gate: it must not take the first stage down.

    The suite hands this module mocks and namespaces, and a live run may reach it with a model
    whose field is frozen. Failing to clear the window is a worse verdict; failing to return is
    no verdict at all.
    """

    class Frozen(SimpleNamespace):
        @property
        def event_time(self):
            return EventWindow(start="2026-08-17T00:00:00Z", end="2026-08-17T09:31:00Z")

    frozen = Frozen()
    _module()._drop_invented_event_window(frozen, _incident("no time in this text"))
    assert frozen.event_time is not None


def test_ordinary_ENGLISH_is_not_a_time_expression_and_the_check_stays_live():
    """The guard was inert: the month alternation `(jan|...|dec)[a-z]*` matched "may" and common
    English words like "marked", "decided", "separate". `_drop_invented_event_window` found a time
    expression in almost any text and discarded nothing.

    A vacuous check is worse than absent, so this asserts the false direction word by word,
    alongside the sentence case that proves the discard still fires on genuine time prose.
    """
    module = _module()

    for word in (
        "may",  # the modal — the one that made this permanent
        "marked",
        "decided",
        "declined",
        "separate",
        "novel",
        "junction",
        "augment",
        "marketing",
        "junior",
        "octopus",
        "aprons",
        "decommissioned",
        "novelty",
        "deception",
    ):
        assert not _TIME_EXPRESSION_RE.search(word), word

    # And a whole description built from them, which is what a real one looks like: the window is
    # discarded, so the declared default depth is what reaches retrieval.
    analysis = _analysis()
    module._drop_invented_event_window(
        analysis,
        _incident(
            "The analyst marked the alert and decided it may involve separate offices; a novel "
            "pattern at the junction of two roles, and the review should augment the earlier one."
        ),
    )
    assert analysis.event_time is None

    # THE ABBREVIATIONS ARE STILL ACCEPTED — with a day-or-year number beside them, which is the
    # only thing that tells a month from a prefix of English. This is the half a fix that merely
    # deleted the alternation would have lost.
    for text in (
        "Reported on 17 Aug by the duty desk.",
        "Reported 17th August.",
        "Aug 17 access from an unknown terminal.",
        "Sept. 4 is when the reads started.",
        "May 2026 saw the first of them.",
        "May 17 was the alert date.",
        "Dec 2025 onwards.",
        # A FULL month name stands alone: unambiguous, and the asymmetry says a stated month
        # must keep the window even with no number in reach.
        "In December the office read records it does not own.",
        "The failed checks began in september and continued.",
    ):
        analysis = _analysis()
        module._drop_invented_event_window(analysis, _incident(text))
        assert analysis.event_time is not None, text


# --- the third post-check: which values arrived TOGETHER -----------------------------------------


def test_a_TABULATED_incident_recovers_one_group_per_record_in_the_incidents_own_order():
    """The reason this check exists: a tabulated incident reaches the stage as 3N independent
    entities; downstream reads the cross product (1000 combinations for 10 real records), which
    comes back full and the row cap then truncates on other parties' rows.

    Two claims: grouping is by line (domain-neutral, no entity type named), and labels are
    zero-padded so the arm order matches the incident's own line order.
    """
    header = "The record was read by the following unit, code and login combinations:"
    rows = [f"UNIT{n:02d} | CODE{n:02d} | LOGIN{n:02d}" for n in range(1, 11)]
    entities = []
    for n in range(1, 11):
        entities += [
            _ent("org_unit", f"UNIT{n:02d}"),
            _ent("user", f"CODE{n:02d}", "code"),
            _ent("user", f"LOGIN{n:02d}", "login"),
        ]
    stamped = _grouped("\n".join([header, ""] + rows), *entities)

    per_record = list(zip(stamped[0::3], stamped[1::3], stamped[2::3]))
    labels = []
    for record in per_record:
        assert len(set(_labels(record))) == 1, _labels(record)
        assert record[0].co_occurrence, "every tabulated record is a group"
        labels.append(record[0].co_occurrence)
    # One group per record, and no record shares another's.
    assert len(set(labels)) == 10
    # The two forms of one type are NOT a second group: a record's three values are one group.
    assert labels[0] == stamped[1].co_occurrence == stamped[2].co_occurrence
    # ...and the labels sort in the order the incident printed them, which is what the padding is
    # for: without it, the tenth record (line 12) would sort between the first and the second.
    assert sorted(labels) == labels
    assert labels[0] == "L000003" and labels[-1] == "L000012"


def test_a_line_that_is_PROSE_and_not_a_record_is_refused_three_ways():
    """The asymmetry inverts here, so the refusals carry the weight.

    Reading a record that is not one ANDs values that never co-occurred, and the query is then
    narrowed onto combinations that are *worse* than the cross product rather than better — a
    result that is neither full nor empty but wrong, on rows that exist. So each refusal gets its
    own case, and each is a shape a real report writes.
    """
    # 1. A REPEATED FIELD means prose. A record names each of its fields once; a sentence naming
    #    three units is prose that happens to mention several. Two such sentences repeat the same
    #    shape, so nothing but this test stops them.
    prose = _grouped(
        "Summary of the escalation.\n"
        "Units UNIT01, UNIT02 and UNIT03 read the record under code CODE01.\n"
        "Units UNIT04, UNIT05 and UNIT06 read the record under code CODE02.",
        *[_ent("org_unit", f"UNIT{n:02d}") for n in range(1, 7)],
        _ent("user", "CODE01", "code"),
        _ent("user", "CODE02", "code"),
    )
    assert _labels(prose) == [""] * 8

    #    ...and the test is per (type, FORM) and not per type, which is the headline case refusing
    #    itself: a table naming its subject by both a code and a login puts two values of type
    #    `user` on every row, and read per type that is every record read as prose.
    both_forms = _grouped(
        "UNIT01 | CODE01 | LOGIN01\nUNIT02 | CODE02 | LOGIN02",
        _ent("org_unit", "UNIT01"),
        _ent("user", "CODE01", "code"),
        _ent("user", "LOGIN01", "login"),
        _ent("org_unit", "UNIT02"),
        _ent("user", "CODE02", "code"),
        _ent("user", "LOGIN02", "login"),
    )
    assert all(_labels(both_forms))

    # 2. FEWER THAN TWO DISTINCT TYPES is not a combination. Two forms of ONE type are alternative
    #    spellings of the same fact, and AND-ing them asserts both spellings appear on one row —
    #    the claim `relax_form_conjunction` exists to refuse.
    one_type = _grouped(
        "CODE01 | LOGIN01\nCODE02 | LOGIN02",
        _ent("user", "CODE01", "code"),
        _ent("user", "LOGIN01", "login"),
        _ent("user", "CODE02", "code"),
        _ent("user", "LOGIN02", "login"),
    )
    assert _labels(one_type) == [""] * 4

    # 3. A SHAPE THAT OCCURS ON ONE LINE is a sentence, not a table — and the refusal costs
    #    nothing it should not, because with one record the per-type lists hold one value each and
    #    their cross product IS that record.
    sentence = _grouped(
        "Office UNIT01 read the record under code CODE01.",
        _ent("org_unit", "UNIT01"),
        _ent("user", "CODE01", "code"),
    )
    assert _labels(sentence) == ["", ""]

    #    The refusal is about REPETITION and not about the sentence form: the same prose twice,
    #    with different values, is a table an operator wrote in words.
    twice = _grouped(
        "Office UNIT01 read the record under code CODE01.\n"
        "Office UNIT02 read the record under code CODE02.",
        _ent("org_unit", "UNIT01"),
        _ent("user", "CODE01", "code"),
        _ent("org_unit", "UNIT02"),
        _ent("user", "CODE02", "code"),
    )
    assert all(_labels(twice))
    assert twice[0].co_occurrence == twice[1].co_occurrence != twice[2].co_occurrence


def test_a_value_is_matched_as_a_WHOLE_TOKEN_and_never_as_a_substring():
    """Load-bearing rather than tidy, because one real entity type is a substring of another.

    An organisation code is routinely carried inside the identifiers of the org units that belong
    to it, so a substring match puts the organisation on every record line — which turns a
    per-record combination into a claim about all of them, and every arm of the rewritten query
    then AND-s an incident-wide constant that the query already carries once.
    """
    stamped = _grouped(
        "Organization ORGX was notified of the reads.\n"
        "UNITORGX01 | CODE01\n"
        "UNITORGX02 | CODE02",
        _ent("organization", "ORGX"),
        _ent("org_unit", "UNITORGX01"),
        _ent("user", "CODE01", "code"),
        _ent("org_unit", "UNITORGX02"),
        _ent("user", "CODE02", "code"),
    )
    org, unit1, code1, unit2, code2 = stamped
    # The organisation is named on its own line, whose single type is not a combination.
    assert org.co_occurrence == ""
    # ...and it is not a member of either record's group, which a substring match would make it.
    assert unit1.co_occurrence == code1.co_occurrence
    assert unit2.co_occurrence == code2.co_occurrence
    assert unit1.co_occurrence != unit2.co_occurrence
    sizes = {}
    for ent in stamped:
        if ent.co_occurrence:
            sizes[ent.co_occurrence] = sizes.get(ent.co_occurrence, 0) + 1
    assert sorted(sizes.values()) == [2, 2]


def test_a_value_on_SEVERAL_records_is_stamped_with_all_of_their_labels():
    """A table repeats a column, so the stamp is a set of labels, not one.

    One entity per distinct value: an org unit on N rows arrives once; a single-valued stamp
    keeps only the last line, deleting it from earlier records, which then have two forms of
    one type and are discarded by `_incident_value_tuples`.
    Both halves: a repeated value carries every label; a single-record value carries exactly one.
    """
    stamped = _grouped(
        "Reads by unit, code and login:\n"
        "UNIT01 | CODE01 | LOGIN01\n"
        "UNIT01 | CODE02 | LOGIN02\n"
        "UNIT02 | CODE03 | LOGIN03",
        _ent("org_unit", "UNIT01"),
        _ent("user", "CODE01", "code"),
        _ent("user", "LOGIN01", "login"),
        _ent("user", "CODE02", "code"),
        _ent("user", "LOGIN02", "login"),
        _ent("org_unit", "UNIT02"),
        _ent("user", "CODE03", "code"),
        _ent("user", "LOGIN03", "login"),
    )
    unit1, code1, login1, code2, login2, unit2, code3, login3 = stamped

    # The unit named on two rows carries BOTH labels, whitespace-joined and in the incident's own
    # order — which the padding gives for free, since the labels are appended line by line.
    assert unit1.co_occurrence.split() == ["L000002", "L000003"]
    assert unit1.co_occurrence == "L000002 L000003"
    # ...and the unit named on one row carries exactly one label, with no separator at all.
    assert unit2.co_occurrence == "L000004"
    for ent in (code1, login1, code2, login2, code3, login3):
        assert len(ent.co_occurrence.split()) == 1, ent.value

    # Every row is a group, and the shared unit is a member of the two it appears on.
    per_row = [(code1, login1), (code2, login2), (code3, login3)]
    for members in per_row:
        assert len({e.co_occurrence for e in members}) == 1, _labels(members)
    rows = [members[0].co_occurrence for members in per_row]
    assert len(set(rows)) == 3
    assert set(rows[:2]) == set(unit1.co_occurrence.split())
    assert rows[2] == unit2.co_occurrence
    # No value the table named is left out of every record: that count is what the guard
    # downstream proves before it narrows anything, and 18 of 66 is what made it decline.
    assert all(e.co_occurrence for e in stamped)


def test_SEVERAL_records_written_on_ONE_line_are_read_as_several_and_not_as_one():
    """A line is the coarsest record boundary; prose does not respect it.

    When subjects arrive as flowing prose, the line reading loses records. Entity spans (each
    entity records the span it was extracted from) give a finer boundary. The finer reading is
    preferred where it finds more records: two disjoint records on one line ANDed together is
    the same cross product, on values that never co-occurred.
    """
    stamped = _grouped(
        "Records were read by, for example, UNIT01/CODE01,\n"
        "UNIT02/CODE02, UNIT03/CODE03 and roughly twenty more.",
        _ent("org_unit", "UNIT01", raw="UNIT01/CODE01"),
        _ent("user", "CODE01", "code", raw="UNIT01/CODE01"),
        _ent("org_unit", "UNIT02", raw="UNIT02/CODE02"),
        _ent("user", "CODE02", "code", raw="UNIT02/CODE02"),
        _ent("org_unit", "UNIT03", raw="UNIT03/CODE03"),
        _ent("user", "CODE03", "code", raw="UNIT03/CODE03"),
    )
    unit1, code1, unit2, code2, unit3, code3 = stamped
    # Three records, each of its two members and nobody else's.
    assert unit1.co_occurrence == code1.co_occurrence
    assert unit2.co_occurrence == code2.co_occurrence
    assert unit3.co_occurrence == code3.co_occurrence
    assert len({e.co_occurrence for e in stamped}) == 3
    assert all(_labels(stamped))

    # SPANS AND LINES ARE COUNTED TOGETHER for the shape refusal, which is what lets the pair on
    # line 1 recognise the two on line 2 as the same table. Counted per reading, that first pair
    # is a shape of one and is discarded — and it is the record the alert led with.
    assert unit1.co_occurrence.startswith("L000001")
    assert unit2.co_occurrence.startswith("L000002")
    # ...and the labels still sort in the incident's own order, which is what a span's suffix is
    # for: an operator reads the rewritten arms back against the text that named them.
    assert sorted(_labels(stamped)) == _labels(stamped)


def test_a_SPAN_is_verified_against_its_own_value_and_ONE_span_defers_to_the_line():
    """Two refusals, and both are about not trusting a field the model wrote.

    A span is LLM-authored, so it is evidence and not authority: read credulously it is a machine
    for asserting co-occurrence, which is the one direction that cannot be recovered downstream.
    """
    # 1. A member must appear in its own span. An incident-wide constant recorded in a
    #    record's span would be AND-ed onto that arm as a fact about all subjects.
    stamped = _grouped(
        "Organization ORGX. Reads by UNIT01/CODE01, UNIT02/CODE02.",
        _ent("organization", "ORGX", raw="UNIT01/CODE01"),
        _ent("org_unit", "UNIT01", raw="UNIT01/CODE01"),
        _ent("user", "CODE01", "code", raw="UNIT01/CODE01"),
        _ent("org_unit", "UNIT02", raw="UNIT02/CODE02"),
        _ent("user", "CODE02", "code", raw="UNIT02/CODE02"),
    )
    org, unit1, code1, unit2, code2 = stamped
    assert org.co_occurrence == ""
    assert unit1.co_occurrence == code1.co_occurrence != unit2.co_occurrence
    assert unit2.co_occurrence == code2.co_occurrence

    #    ...and a span that swallowed the WHOLE line leaves ONE candidate, so the line reading
    #    decides — and refuses it as prose, because one span holding both records repeats every
    #    field. That is the model's worst case costing nothing: the coarse reading is the floor.
    whole = _grouped(
        "Reads by UNIT01/CODE01 and UNIT02/CODE02.",
        *[
            _ent(t, v, f, raw="Reads by UNIT01/CODE01 and UNIT02/CODE02.")
            for t, v, f in (
                ("org_unit", "UNIT01", ""),
                ("user", "CODE01", "code"),
                ("org_unit", "UNIT02", ""),
                ("user", "CODE02", "code"),
            )
        ],
    )
    assert _labels(whole) == [""] * 4

    # 2. One span is not evidence of a boundary; the line reading stands. A tabulated
    #    incident with spans must group byte-identically to one without.
    for raw1, raw2 in ((("UNIT01 | CODE01"), ("UNIT02 | CODE02")), ("", "")):
        table = _grouped(
            "Reads by unit and code:\nUNIT01 | CODE01\nUNIT02 | CODE02",
            _ent("org_unit", "UNIT01", raw=raw1),
            _ent("user", "CODE01", "code", raw=raw1),
            _ent("org_unit", "UNIT02", raw=raw2),
            _ent("user", "CODE02", "code", raw=raw2),
        )
        assert _labels(table) == ["L000002", "L000002", "L000003", "L000003"]


def test_a_PROSE_span_beside_two_real_ones_is_refused_without_costing_them():
    """The two per-span refusals, and they are only REACHABLE in this shape.

    A single bad span is harmless — one candidate means the line reading decides, and it applies
    the same two refusals — so a fixture with one prose span proves nothing about them. They bite
    where a line mixes a tabulation with a summarising clause, which is what a report actually
    writes: two pairs, then "and units X and Y under code Z". The good spans must survive
    (refusing the whole line would lose them) and the prose span must not become a record.

    Both are asserted with the prose shape occurring TWICE, because a shape occurring once is
    discarded by the third refusal anyway — the only way to tell "refused as prose" from "not
    repeated" is to repeat it.
    """
    # 1. A REPEATED FIELD inside a span is prose. Admitted, it AND-s two org units that never
    #    co-occurred onto one code — a combination worse than the cross product rather than better,
    #    on rows that exist.
    stamped = _grouped(
        "Reads by UNIT01/CODE01, UNIT02/CODE02, then UNIT03 and UNIT04 under CODE03, "
        "then UNIT05 and UNIT06 under CODE04.",
        _ent("org_unit", "UNIT01", raw="UNIT01/CODE01"),
        _ent("user", "CODE01", "code", raw="UNIT01/CODE01"),
        _ent("org_unit", "UNIT02", raw="UNIT02/CODE02"),
        _ent("user", "CODE02", "code", raw="UNIT02/CODE02"),
        _ent("org_unit", "UNIT03", raw="UNIT03 and UNIT04 under CODE03"),
        _ent("org_unit", "UNIT04", raw="UNIT03 and UNIT04 under CODE03"),
        _ent("user", "CODE03", "code", raw="UNIT03 and UNIT04 under CODE03"),
        _ent("org_unit", "UNIT05", raw="UNIT05 and UNIT06 under CODE04"),
        _ent("org_unit", "UNIT06", raw="UNIT05 and UNIT06 under CODE04"),
        _ent("user", "CODE04", "code", raw="UNIT05 and UNIT06 under CODE04"),
    )
    assert _labels(stamped[4:]) == [""] * 6
    assert stamped[0].co_occurrence == stamped[1].co_occurrence
    assert stamped[2].co_occurrence == stamped[3].co_occurrence
    assert stamped[0].co_occurrence != stamped[2].co_occurrence

    # 2. FEWER THAN TWO DISTINCT TYPES inside a span is not a combination either. Two FORMS of one
    #    type are alternative spellings of one fact, and AND-ing them asserts both spellings sit on
    #    one row — the claim `relax_form_conjunction` exists to refuse, arriving as a record.
    stamped = _grouped(
        "Reads by UNIT01/CODE01, UNIT02/CODE02, also seen as CODE05/LOGIN05 and CODE06/LOGIN06.",
        _ent("org_unit", "UNIT01", raw="UNIT01/CODE01"),
        _ent("user", "CODE01", "code", raw="UNIT01/CODE01"),
        _ent("org_unit", "UNIT02", raw="UNIT02/CODE02"),
        _ent("user", "CODE02", "code", raw="UNIT02/CODE02"),
        _ent("user", "CODE05", "code", raw="CODE05/LOGIN05"),
        _ent("user", "LOGIN05", "login", raw="CODE05/LOGIN05"),
        _ent("user", "CODE06", "code", raw="CODE06/LOGIN06"),
        _ent("user", "LOGIN06", "login", raw="CODE06/LOGIN06"),
    )
    assert _labels(stamped[4:]) == [""] * 4
    assert all(_labels(stamped[:4]))


def test_the_grouper_is_a_post_check_and_never_takes_the_first_stage_down():
    """No groups is the normal answer, and it must be byte-identical to having no groups at all.

    Most incidents state no repeated per-record structure, so every silence here is the common
    path: too few entities to pair, no text to read them in, and a model whose field cannot be
    written — the suite hands this module namespaces and a live run may reach it with a frozen
    field. Failing to group is a wider query; failing to return is no verdict at all.
    """
    module = _module()

    lone = _grouped("UNIT01 | CODE01\nUNIT02 | CODE02", _ent("org_unit", "UNIT01"))
    assert _labels(lone) == [""]

    blank = _grouped("", _ent("org_unit", "UNIT01"), _ent("user", "CODE01", "code"))
    assert _labels(blank) == ["", ""]

    # A text-bearing key that carries no text is not a crash either.
    analysis = _analysis(start=None)
    analysis.extracted_entities = [
        _ent("org_unit", "UNIT01"),
        _ent("user", "CODE01", "code"),
    ]
    module._group_co_occurring_entities(
        analysis, {"id": "INC-1", "description": None, "title": 42, "summary": {"a": 1}}
    )
    assert _labels(analysis.extracted_entities) == ["", ""]

    class Frozen(SimpleNamespace):
        @property
        def co_occurrence(self):
            return ""

    frozen = _analysis(start=None)
    frozen.extracted_entities = [
        Frozen(type="org_unit", value="UNIT01"),
        Frozen(type="user", value="CODE01"),
        Frozen(type="org_unit", value="UNIT02"),
        Frozen(type="user", value="CODE02"),
    ]
    module._group_co_occurring_entities(
        frozen, _incident("UNIT01 | CODE01\nUNIT02 | CODE02")
    )
    assert _labels(frozen.extracted_entities) == [""] * 4


# --- the fourth post-check: an explicitly requested link, confined to what the text says ---------


def _confined(asks, text, summary="s", **extra):
    """Run the confinement over one description and return the asks that survived it."""
    analysis = _analysis(start=None)
    analysis.incident_summary = summary
    analysis.requested_links = list(asks)
    _module()._confine_requested_links(analysis, _incident(text, **extra))
    return analysis.requested_links


def test_an_ask_the_text_does_not_MAKE_is_discarded_and_the_drop_NAMES_it(caplog):
    """The field's only authority is that the text asks, so an ask it does not make is a lie.

    Everything else reaching the link assessment carries a declared, base-rate-measured signal
    behind it. This one carries a sentence, and skips the strength threshold on the strength of
    it — a human asked. So a fabricated ask does not merely add noise: it produces the one
    finding in the set that says *the reporter wanted this checked*, about a procedure nobody
    mentioned.
    """
    text = (
        "Ticketing fraud suspected on record ABC123. The issuing office is LLL1M13YZ. "
        "Please also confirm whether the account takeover detector fired for the same sign."
    )
    with caplog.at_level("INFO"):
        kept = _confined(
            [
                "confirm whether the account takeover detector fired",
                "also verify there was no loyalty points abuse",
            ],
            text,
        )
    assert kept == ["confirm whether the account takeover detector fired"]
    # The drop is named rather than counted: an operator reading it can see whether the model
    # invented the request or the ingestion route dropped the sentence that made it.
    assert "loyalty points abuse" in caplog.text
    assert "DISCARDED" in caplog.text
    # ...and the surviving ask is logged too, because it changes what the run reports.
    assert any("explicitly requested" in r.message for r in caplog.records)

    # Traced against every text field: input structure is not a contract; the same report
    # arrives as title, summary or raw blob. `description` alone would miss a forwarded ask.
    for key in ("description", "title", "summary", "raw_text"):
        analysis = _analysis(start=None)
        analysis.requested_links = ["confirm the loyalty points abuse"]
        _module()._confine_requested_links(
            analysis, {"id": "INC-1", key: "please confirm the loyalty points abuse too"}
        )
        assert analysis.requested_links == ["confirm the loyalty points abuse"], key


def test_the_quantifier_is_ALL_because_one_ORDINARY_word_is_not_a_request():
    """Under `any`, an entirely fabricated ask rides in on a preposition.

    This is the vacuity floor, and it is the reason the filler list is safe to be incomplete.
    Read as "some substantive word of the ask appears in the text", a list that happens not to
    name `for` lets `also check for account takeover` through against any text containing the
    word `for` — which is every text. Read as `all`, a fabrication has to fabricate nothing.
    """
    text = "Unauthorised reads were reported for office LLL1M13YZ by the duty analyst."
    # `for` is in the text; `account` and `takeover` are not. One shared word, no ask.
    assert _confined(["also check for account takeover"], text) == []
    # An ask made only of request phrasing names no procedure at all, whatever the text says.
    for filler_only in ("please also confirm", "verify there was no related fraud", "check"):
        assert _confined([filler_only], text) == [], filler_only
    # Cost of `all` stated: one missing word drops the whole ask. The strict direction is
    # cheap: a declared signal raises the link whether or not the sentence survived.
    assert _confined(["urgently confirm the LLL1M13YZ reads"], text) == []
    assert _confined(["please confirm the LLL1M13YZ reads"], text) == [
        "please confirm the LLL1M13YZ reads"
    ]


def test_a_token_stands_as_its_OWN_word_and_not_inside_an_identifier():
    """A report that names a log source has not asked for that source's procedure.

    Incident text routinely names sources and columns, and those names carry procedure
    abbreviations inside them. `\\b` is what makes this the strict reading rather than the loose
    one — Python counts `_` as a word character, so the identifier does not answer for the word —
    which is the inverse of how the same fact bites the domain-neutrality scanner, where `_`
    counting as a word character is what let four leaks through.
    """
    named_source = "The ato_events index was reviewed and no rows matched office LLL1M13YZ."
    assert _is_traceable_ask("check ato", named_source) is False
    assert _confined(["check ato"], named_source.lower()) == []
    # The same word written as a word IS the ask.
    spoken = "Reads by office LLL1M13YZ; also check ato for the same sign."
    assert _is_traceable_ask("check ato", spoken.lower()) is True


def test_a_VERBATIM_echo_in_the_INCIDENT_SUMMARY_is_reported_and_never_EDITED(caplog):
    """The half that cannot be enforced, and the distinction is the point.

    Keeping the ask out of `incident_summary` is structural in the engine — nothing under `src/`
    copies it there — but the summary is written by the model, and it is the only text scored to
    decide which single procedure adjudicates this incident. An engine cannot delete a clause
    from prose without judging what else the clause was carrying, so the one shape that is
    mechanically decidable is reported: the ask appearing in the summary word for word. The run
    may then have been adjudicated by the wrong procedure, and nothing else would say so.
    """
    ask = "confirm whether the account takeover detector also fired"
    text = f"Ticketing fraud on ABC123 at office LLL1M13YZ. Please {ask}."
    summary = (
        "Ticketing fraud is suspected at office LLL1M13YZ, and the reporter asks to "
        f"{ask} for the same sign."
    )
    with caplog.at_level("WARNING"):
        analysis = _analysis(start=None)
        analysis.incident_summary = summary
        analysis.requested_links = [ask]
        _module()._confine_requested_links(analysis, _incident(text))

    assert any("VERBATIM" in r.message for r in caplog.records)
    # Reported, and nothing else: the summary is untouched and so is the ask, because the field
    # this rides in is not the field that was polluted.
    assert analysis.incident_summary == summary
    assert analysis.requested_links == [ask]

    # The two bounds, and both are about NOT firing. A few words are a phrase any paraphrase of
    # the incident shares with the ask, so a short one is not evidence the ask was folded in —
    # and a warning on it is a warning about the summary naming its own subject.
    short = "ato check"
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert _confined(
            [short], "reads by LLL1M13YZ; also ato check", summary=f"An {short} was requested."
        ) == [short]
    assert "VERBATIM" not in caplog.text

    # ...and a summary that merely shares the incident's own vocabulary with the ask says nothing:
    # a token-overlap test here would fire on every incident whose subject is named in both, which
    # ships as noise and gets switched off. Only the whole ask, word for word, is decidable.
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert _confined([ask], text.lower(), summary="Ticketing fraud at office LLL1M13YZ.")
    assert "VERBATIM" not in caplog.text


def test_no_ASK_is_the_normal_answer_and_the_check_never_takes_the_stage_down(caplog):
    """Almost every incident asks for nothing, so every silence here is the common path.

    A post-check, not a gate — the suite hands this module namespaces and a live run may reach it
    with a frozen field. Failing to confine an ask states one referral too many; failing to
    return is no verdict at all.
    """
    module = _module()

    # The field defaults empty, which is what makes the whole link surface additive: an incident
    # that asks for nothing reaches the assessment with nothing, and every reader can duck-type
    # a list rather than testing for the key.
    assert _analysis(start=None).requested_links == []
    assert _analysis(start=None).model_dump()["requested_links"] == []

    assert _confined([], "anything at all") == []
    # A field that is not a list at all (a model swap, a hand-built namespace) is left alone.
    analysis = _analysis(start=None)
    analysis.requested_links = "account takeover"
    module._confine_requested_links(analysis, _incident("account takeover"))
    assert analysis.requested_links == "account takeover"
    # An analysis with no such field is not a crash.
    module._confine_requested_links(SimpleNamespace(), _incident("account takeover"))

    # A blank entry is dropped and is NOT reported as an invention: an empty string is a model
    # emitting the field it was told to emit, not a claim about the text, and a warning naming
    # `''` sends a reader looking for a sentence nobody wrote.
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert _confined(["", "   "], "account takeover was reported") == []
    assert "DISCARDED" not in caplog.text
    # A text-bearing key that carries no text is not a crash either, and with no text at all
    # every ask is untraceable — which is the strict direction, deliberately.
    analysis = _analysis(start=None)
    analysis.requested_links = ["account takeover"]
    module._confine_requested_links(
        analysis, {"id": "INC-1", "description": None, "title": 42, "summary": {"a": 1}}
    )
    assert analysis.requested_links == []

    class Frozen(SimpleNamespace):
        @property
        def requested_links(self):
            return ["account takeover"]

    frozen = Frozen(incident_summary="s")
    module._confine_requested_links(frozen, _incident("no such words here"))
    assert frozen.requested_links == ["account takeover"]


def test_the_PROMPT_asks_for_the_field_and_forbids_repeating_it_into_the_summary():
    """A field the model is never told about is a field that is always empty.

    The confinement above can only ever narrow what arrives, so it cannot make the path work —
    and an inert route is indistinguishable from a corpus of incidents that ask for nothing. The
    prohibition is asserted beside it because it is the one instruction here that protects
    something already measured: `incident_summary` is the only text `select_correlation_spec`
    scores, and rival-procedure vocabulary in it is what took playbook selection from 27/27 to
    10/27. The engine's own separation is structural, so this instruction is not what enforces it
    — but it is what stops the model writing the sentence twice, which the engine cannot undo.
    """
    prompt = _module().system_prompt
    assert "requested_links" in prompt
    # Emitted only where the text asks, so the default is empty rather than a guess.
    assert "ONLY if" in prompt and "empty otherwise" in prompt
    # And named as its own field, with the summary named as the place it must not be repeated.
    ask_para = prompt.split("requested_links", 1)[1]
    assert "incident_summary" in ask_para
    for other in ("initial_hypotheses", "key_investigation_areas"):
        assert other in ask_para, other
