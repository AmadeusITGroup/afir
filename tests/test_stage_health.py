"""
Tests for the deterministic stage-health scorer (``src/stage_health.py``).

Three things these guard:

1. **A fatal signal cannot be averaged away.** ``no_rows_at_all`` carries weight 1.0;
   averaging it into a mean lets a stage that retrieved nothing score 0.7.
2. **The scorer never raises.** A scorer that throws turns a successful stage into a
   failed job. It reports ``scored=False`` on unexpected shapes.
3. **MagicMock outputs do not fake a healthy stage.** Every attribute is truthy, so
   ``isinstance(list)`` must be checked rather than truthiness.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.evidence import CLIP_NOTE, degrade_to_budget
from src.models.pydantic_models import EvidencePack
from src.stage_health import (DEFAULT_THRESHOLD, GATEABLE_STAGES, score_stage,
                              stage_gate_enabled, stage_threshold)

# --- helpers -----------------------------------------------------------------


class _Ctx:
    """A stand-in for JobContext carrying only what the scorer reads."""

    def __init__(self, outputs=None, modules=None, config=None, stage_facts=None):
        self.outputs = outputs or {}
        self.modules = modules or {}
        self.config = config or {}
        self.stage_facts = stage_facts or {}


def _analysis(**over):
    """An `IncidentAnalysis`-shaped object with every health signal satisfied."""
    a = MagicMock()
    a.extracted_entities = over.get("extracted_entities", [MagicMock()])
    a.correlation_keys = over.get("correlation_keys", ["user"])
    a.log_sources_to_review = over.get("log_sources_to_review", ["src_a"])
    event_time = over.get("event_time", MagicMock(start="2026-01-01", end="2026-01-02"))
    a.event_time = event_time
    return a


def _understanding(**over):
    u = MagicMock()
    u.analysis = _analysis(**over)
    return u


def _query(source="src_a", date_from="2026-01-01", date_to="2026-01-02"):
    q = MagicMock()
    q.target_log_source = source
    q.date_from = date_from
    q.date_to = date_to
    return q


def _anomaly(score=0.9, below=False):
    a = MagicMock()
    a.confidence_score = score
    # Set on both sides: `filter_anomalies` stamps this on every returned item, so an
    # anomaly missing it is not a real production shape; a bare MagicMock is truthy.
    a.below_threshold = below
    return a


def _correlation(**over):
    c = MagicMock()
    c.record_count = over.get("record_count", 10)
    c.aggregations = over.get(
        "aggregations", {"resolved_correlation_keys": [{"entity_hint": "user"}]}
    )
    c.findings = over.get("findings", [MagicMock()])
    c.evidence = over.get("evidence", MagicMock(degraded=False))
    c.verdict = over.get("verdict", None)
    c.brief = over.get("brief", None)
    return c


def codes(health):
    return set(health.reason_codes)


# --- understanding -----------------------------------------------------------


def test_healthy_understanding_scores_one_and_does_not_gate():
    h = score_stage("understanding", _understanding(), _Ctx())
    assert h.score == 1.0
    assert h.reasons == []
    assert h.gate_recommended is False
    assert h.scored is True


def test_no_entities_is_fatal():
    """A MagicMock analysis with entities=[] must fire — not be read as truthy."""
    h = score_stage("understanding", _understanding(extracted_entities=[]), _Ctx())
    assert h.score == 0.0
    assert "no_entities" in codes(h)
    assert h.gate_recommended is True


def test_magicmock_attribute_does_not_pass_for_a_populated_list():
    """The dual-import/MagicMock trap: a Mock attribute is truthy but not a list."""
    u = MagicMock()
    u.analysis = MagicMock()  # every attribute is a truthy Mock, no real lists
    h = score_stage("understanding", u, _Ctx())
    assert "no_entities" in codes(h)
    assert "no_correlation_keys" in codes(h)
    assert h.score == 0.0


def test_missing_event_time_and_keys_accumulate_without_being_fatal():
    h = score_stage(
        "understanding",
        _understanding(event_time=None, correlation_keys=[]),
        _Ctx(),
    )
    assert codes(h) == {"no_event_time", "no_correlation_keys"}
    assert h.score == pytest.approx(1.0 - 0.3 - 0.3)
    assert h.gate_recommended is True  # 0.4 < 0.6 default


def test_a_single_mid_weight_signal_stays_above_the_default_threshold():
    h = score_stage("understanding", _understanding(event_time=None), _Ctx())
    assert h.score == pytest.approx(0.7)
    assert h.gate_recommended is False


def test_understanding_with_no_analysis_is_fatal():
    h = score_stage("understanding", MagicMock(analysis=None), _Ctx())
    assert h.score == 0.0


# --- query_generation --------------------------------------------------------


def test_zero_queries_is_fatal():
    h = score_stage("query_generation", [], _Ctx())
    assert h.score == 0.0
    assert "no_queries" in codes(h)


def test_a_requested_source_with_no_query_fires_once_per_source():
    ctx = _Ctx(
        outputs={
            "understanding": _understanding(log_sources_to_review=["src_a", "src_b"])
        }
    )
    h = score_stage("query_generation", [_query("src_a")], ctx)
    assert "source_without_query" in codes(h)
    # one missing source at 0.2
    assert h.score == pytest.approx(0.8)
    assert (
        "src_b" in [r.detail for r in h.reasons if r.code == "source_without_query"][0]
    )


def test_a_ruleset_declared_source_that_cannot_be_retrieved_gates_the_stage():
    """An unmet hard dependency must gate on its own.

    Job 4da14f65: the ruleset's `alert` source (the incident's own alert record) had no
    ELK credentials, built no retriever, was dropped silently from the required-source
    top-up, and the run produced a confident verdict with every stage green.
    """
    gen = SimpleNamespace(undeliverable_required=["siem_alerts_current"])
    h = score_stage(
        "query_generation", [_query("src_a")], _Ctx(modules={"api_call": gen})
    )
    assert "required_source_unavailable" in codes(h)
    assert h.score == pytest.approx(0.5)
    assert h.gate_recommended is True
    assert (
        "siem_alerts_current"
        in [r.detail for r in h.reasons if r.code == "required_source_unavailable"][0]
    )


def test_a_declared_source_the_planner_did_not_query_gates_the_stage():
    """The replacement for the deleted force-add, and it must gate alone.

    A dependency the procedure declares, that CAN be retrieved, and that no query targets is
    a defect in what the catalog tells the planner — and the conditions reading it will be
    `unknown` while every stage reports success. That gap used to be closed by injecting the
    query, which bypassed the reasoning it was compensating for. It is now a finding, so this
    signal is the whole safety net: at 0.5 it lands on the 0.6 threshold on its own.
    """
    gen = SimpleNamespace(declared_not_queried=["automation_registry"])
    h = score_stage(
        "query_generation", [_query("src_a")], _Ctx(modules={"api_call": gen})
    )
    assert "required_source_not_queried" in codes(h)
    assert h.score == pytest.approx(0.5)
    assert h.gate_recommended is True
    detail = [
        r.detail for r in h.reasons if r.code == "required_source_not_queried"
    ][0]
    assert "automation_registry" in detail
    # It names both ways out: the operator's own click, and the catalog fix that answers
    # every future incident rather than this one.
    assert "run controls" in detail and "catalog" in detail

    # One reason per source, so a stage missing three dependencies does not read as one.
    many = SimpleNamespace(declared_not_queried=["a", "b", "c"])
    h2 = score_stage(
        "query_generation", [_query("src_a")], _Ctx(modules={"api_call": many})
    )
    assert len([r for r in h2.reasons if r.code == "required_source_not_queried"]) == 3

    # And the two dependency findings are distinct: one needs credentials, the other needs
    # a catalog fix or a click, and a reader who cannot tell them apart acts on neither.
    both = SimpleNamespace(
        undeliverable_required=["siem_alerts"], declared_not_queried=["registry"]
    )
    h3 = score_stage(
        "query_generation", [_query("src_a")], _Ctx(modules={"api_call": both})
    )
    assert {"required_source_unavailable", "required_source_not_queried"} <= set(codes(h3))


def test_a_selected_source_nothing_could_scope_is_reported_but_does_not_gate():
    """The same predicate as `required_source_not_queried`, read from the other end.

    A source the planner SELECTED although the incident names no entity type it can filter
    on is dropped by the generator rather than issued as a window-only scan, so by the time
    anybody reads the plan it is correct — which is why this is 0.1 and must NOT gate alone.
    The deduction exists because the alternative is a silent correction: the operator sees a
    plan that never mentions the source and cannot tell it was considered. Contrast the two
    weights deliberately — an unmet dependency (0.5) is work the run still owes; this one is
    work the engine already did.
    """
    gen = SimpleNamespace(selected_unscopable=["auth_events"])
    h = score_stage(
        "query_generation", [_query("src_a")], _Ctx(modules={"api_call": gen})
    )
    assert "selected_source_unscopable" in codes(h)
    assert h.score == pytest.approx(0.9)
    assert h.gate_recommended is False
    detail = [
        r.detail for r in h.reasons if r.code == "selected_source_unscopable"
    ][0]
    assert "auth_events" in detail
    # Names both remedies, like its sibling: the catalog fix that answers every future
    # incident, and the one click that answers this one.
    assert "catalog" in detail and "run controls" in detail

    # Scaled per source, so five dropped picks do not read as one.
    many = SimpleNamespace(selected_unscopable=["a", "b", "c", "d", "e"])
    h2 = score_stage(
        "query_generation", [_query("src_a")], _Ctx(modules={"api_call": many})
    )
    assert h2.score == pytest.approx(0.5)
    assert h2.gate_recommended is True

    # And it is read strictly, or the MagicMock modules most tests inject would deduct
    # from every run — the `_declinable` lesson again.
    for module in (MagicMock(), SimpleNamespace(selected_unscopable="a_string"), None):
        h3 = score_stage(
            "query_generation", [_query("src_a")], _Ctx(modules={"api_call": module})
        )
        assert "selected_source_unscopable" not in codes(h3)


def test_a_selected_source_whose_call_produced_no_query_is_reported_between_the_two():
    """The fifth finding, and its WEIGHT is the whole assertion.

    A pick that never became a query sits between the two it is easy to confuse it with. Unlike
    an unscopable pick (0.1) nothing was corrected — the planner's reasoning was right and the
    plan is missing the source anyway — and unlike a declared dependency (0.5) the engine cannot
    know the source was needed, so it must not gate alone. Two of them do, which is the intended
    reading: one lost pick is a nudge, a pattern of them is a planner or schema problem an
    operator should see before approving the plan.
    """
    gen = SimpleNamespace(selected_unparseable=["ticket_access"])
    h = score_stage(
        "query_generation", [_query("src_a")], _Ctx(modules={"api_call": gen})
    )
    assert "selected_source_unparseable" in codes(h)
    assert h.score == pytest.approx(0.65)
    assert h.gate_recommended is False
    detail = [r.detail for r in h.reasons if r.code == "selected_source_unparseable"][0]
    assert "ticket_access" in detail
    # Names the consequence, not the symptom: without the finding this reads as a source that
    # was never chosen, which is what made the live occurrence invisible.
    assert "never chosen" in detail and "run controls" in detail

    two = SimpleNamespace(selected_unparseable=["a", "b"])
    h2 = score_stage(
        "query_generation", [_query("src_a")], _Ctx(modules={"api_call": two})
    )
    assert h2.score == pytest.approx(0.3)
    assert h2.gate_recommended is True

    # And read strictly, or the MagicMock modules most tests inject deduct from every run.
    for module in (MagicMock(), SimpleNamespace(selected_unparseable="a_string"), None):
        h3 = score_stage(
            "query_generation", [_query("src_a")], _Ctx(modules={"api_call": module})
        )
        assert "selected_source_unparseable" not in codes(h3)


def test_undeliverable_required_is_read_strictly_not_truthily():
    """A MagicMock module must not gate every run — the `_declinable` lesson, inverted.

    This signal ADDS a deduction, so a truthy read (rather than list-of-str) would fire
    against the MagicMock modules that most tests inject.
    """
    for module in (
        MagicMock(),
        SimpleNamespace(undeliverable_required="a_string"),
        None,
    ):
        h = score_stage(
            "query_generation", [_query("src_a")], _Ctx(modules={"api_call": module})
        )
        assert "required_source_unavailable" not in codes(h)


def test_this_runs_own_dependency_findings_outrank_the_shared_generators():
    """The generator is ONE object shared by every job, so its attributes are the last plan's.

    Every finding on this stage is read off `modules["api_call"]`, and that module is built once
    for the process — so a concurrent sibling's plan decides this run's deduction, and nothing
    about the score says which run it describes. The run records its own copy under
    `stage_facts["query_generation"]`, and a recorded key must WIN.

    Asserted in both directions, because only one of them is the leak: a recorded finding the
    module does not carry must fire (this run's gap, absolved by the sibling), and a finding the
    module carries that this run recorded as empty must NOT (the sibling's gap, charged here).
    """
    stale = SimpleNamespace(
        declared_not_queried=["a_sibling_runs_gap"],
        selected_unparseable=["another_siblings_pick"],
        undeliverable_required=[],
        declared_unscopable=[],
    )
    mine = {
        "declared_not_queried": ["this_runs_gap"],
        "selected_unparseable": [],
        "undeliverable_required": [],
        "declared_unscopable": [],
    }
    h = score_stage(
        "query_generation",
        [_query("src_a")],
        _Ctx(modules={"api_call": stale}, stage_facts={"query_generation": mine}),
    )
    detail = [r.detail for r in h.reasons if r.code == "required_source_not_queried"]
    assert len(detail) == 1 and "this_runs_gap" in detail[0]
    assert "a_sibling_runs_gap" not in detail[0]
    # The sibling's unparseable pick is charged to nobody: this run recorded an empty list,
    # which is an ANSWER and not an absence.
    assert "selected_source_unparseable" not in codes(h)

    # An EMPTY recorded record still wins — that is the half a truthy check would lose, and it
    # is the direction that over-charges an innocent run.
    h2 = score_stage(
        "query_generation",
        [_query("src_a")],
        _Ctx(
            modules={"api_call": stale},
            stage_facts={"query_generation": dict(mine, declared_not_queried=[])},
        ),
    )
    assert "required_source_not_queried" not in codes(h2)
    assert h2.score == pytest.approx(1.0)


def test_a_run_that_recorded_nothing_still_scores_off_the_module():
    """The fallback is deliberate: recording nothing must not read as finding nothing.

    A caller that drives the stages directly (no job context recording facts) records no key at
    all, and requiring one would silently absolve every plan it makes — a scorer that measures
    nothing while reporting `scored`. Wrong-across-jobs beats blind, so the attribute stays the
    fallback and only a RECORDED key overrides it.
    """
    gen = SimpleNamespace(declared_not_queried=["automation_registry"])
    for facts in ({}, {"query_generation": {}}, {"query_generation": "not a dict"}):
        h = score_stage(
            "query_generation",
            [_query("src_a")],
            _Ctx(modules={"api_call": gen}, stage_facts=facts),
        )
        assert "required_source_not_queried" in codes(h)


def _engine_ctx(wanted, available):
    """A ctx whose retrieval engine offers `available` and whose analysis wants `wanted`."""
    engine = MagicMock()
    engine.retrievers = {name: MagicMock() for name in available}
    return _Ctx(
        outputs={"understanding": _understanding(log_sources_to_review=wanted)},
        modules={"log_retrieval": engine},
    )


# The verbatim prose two live runs produced, kept as the fixture because inventing
# tidier strings is exactly how this bug survived a 740-test suite.
_REAL_PROSE = [
    "record Data Lake / Easy record — full record version for ABC123: creator sign, issuing "
    "sign, TENDER element, AUX elements (DOCS, CTCM), asset_document.",
    "siem_alerts_current (raw.prd.siem-alerts type=scheme) — SCHEME alert details for "
    "org_unit ORG1A0955 on 2026-07-20.",
    "AUTH_SVC authentication logs — login session for sign 6009JJ: IP, MFA, RBA, outcome.",
    "PAYMENT_SVC / Fraud Management — check if the tender card BIN is on stolen lists.",
    "app appEvents — issuance application events for ISSUE and VOID on record ABC123.",
]


def test_prose_that_names_a_covered_source_does_not_fire():
    """`log_sources_to_review` is prose; `target_log_source` is a canonical id.

    Comparing them literally made this signal fire on every real run, pinning
    query_generation at 0.00 while the suite stayed green (its fixtures use clean ids
    like `src_a`). Only entries that NAME a known source can be judged.
    """
    ctx = _engine_ctx(
        _REAL_PROSE, ["record_lake", "siem_alerts_current", "payment_alerts"]
    )
    queries = [
        _query("record_lake"),
        _query("siem_alerts_current"),
        _query("payment_alerts"),
    ]
    h = score_stage("query_generation", queries, ctx)
    assert "source_without_query" not in codes(h)
    assert h.score == pytest.approx(1.0)


def test_prose_naming_no_known_source_is_not_treated_as_missing():
    """ "AUTH_SVC authentication logs" names no id — silence is the only honest reading.

    `auth_events` is not a substring of it, so claiming it is missing would
    measure the model's phrasing rather than under-retrieval.
    """
    ctx = _engine_ctx(
        ["AUTH_SVC authentication logs — login session for sign 6009JJ."],
        ["auth_events"],
    )
    h = score_stage("query_generation", [_query("auth_events")], ctx)
    assert "source_without_query" not in codes(h)


def test_a_named_but_untargeted_source_still_fires():
    """The resolver must not swallow real under-retrieval."""
    ctx = _engine_ctx(
        [
            "record_lake — retrieve the version for ABC123.",
            "settlement_report (SETTLEMENT/BATCH settlement) — refund accounting.",
        ],
        ["record_lake", "settlement_report"],
    )
    h = score_stage("query_generation", [_query("record_lake")], ctx)
    assert "source_without_query" in codes(h)
    reason = [r for r in h.reasons if r.code == "source_without_query"][0]
    assert reason.count == 1
    assert "settlement_report" in reason.detail


# ── declining a source the pack made optional is not under-retrieval ───────────────
# `selection_guidance` and `skip_when` make asking a judgement call; scoring a skip
# makes those keys unusable.


class _Src:
    def __init__(self, selection_guidance=None, status=""):
        self.selection_guidance = selection_guidance
        self.status = status


def _pack_ctx(wanted, available, sources):
    """`_engine_ctx`, plus a real pack whose `source()` returns plain objects.

    Deliberately NOT a MagicMock: every attribute of one is truthy, which is how a first
    cut of `_declinable` excused every source and silently disabled the whole signal.
    """
    engine = MagicMock()
    engine.retrievers = {name: MagicMock() for name in available}
    pack = MagicMock()
    pack.source = lambda n: sources.get(n)
    return _Ctx(
        outputs={"understanding": _understanding(log_sources_to_review=wanted)},
        modules={"log_retrieval": engine, "knowledge_pack": pack},
    )


def test_a_source_the_pack_says_when_to_skip_is_not_scored_for_being_skipped():
    ctx = _pack_ctx(
        ["record_lake — the version.", "raw_access_log — the record access trail."],
        ["record_lake", "raw_access_log"],
        {
            "record_lake": _Src(),
            "raw_access_log": _Src(
                selection_guidance={
                    "choose_when": ["the question is who DISPLAYED the record"],
                    "skip_when": ["the record data already answers it"],
                }
            ),
        },
    )
    h = score_stage("query_generation", [_query("record_lake")], ctx)
    assert "source_without_query" not in codes(h)


def test_a_legacy_source_is_excused_only_because_something_current_ran():
    sources = {"scheme_alerts": _Src(status="legacy"), "siem_alerts_current": _Src()}
    wanted = [
        "scheme_alerts (legacy fallback index)",
        "siem_alerts_current — the live alert.",
    ]
    ctx = _pack_ctx(wanted, list(sources), sources)
    h = score_stage("query_generation", [_query("siem_alerts_current")], ctx)
    assert "source_without_query" not in codes(h)


def test_an_empty_selection_guidance_excuses_nothing():
    """The key present but empty declares no decision, so there is none to respect."""
    ctx = _pack_ctx(
        ["record_lake — the version.", "settlement_report — settlement."],
        ["record_lake", "settlement_report"],
        {
            "record_lake": _Src(),
            "settlement_report": _Src(
                selection_guidance={"choose_when": [], "skip_when": []}
            ),
        },
    )
    h = score_stage("query_generation", [_query("record_lake")], ctx)
    assert "source_without_query" in codes(h)


def test_a_non_mapping_selection_guidance_excuses_nothing():
    """Guard against the truthy read: any object at all would otherwise delete the
    deduction, which fails in the direction that HIDES real under-retrieval."""
    ctx = _pack_ctx(
        ["record_lake — the version.", "settlement_report — settlement."],
        ["record_lake", "settlement_report"],
        {
            "record_lake": _Src(),
            "settlement_report": _Src(selection_guidance="ask when relevant"),
        },
    )
    h = score_stage("query_generation", [_query("record_lake")], ctx)
    assert "source_without_query" in codes(h)


def test_a_source_the_engine_cannot_serve_is_not_this_stage_s_fault():
    """An unprovisioned source (no creds) isn't under-retrieval by query_generation.

    Judging against the engine's real catalog keeps this from firing on every run in
    a partially-provisioned environment.
    """
    ctx = _engine_ctx(
        [
            "record_lake — the version.",
            "gdpr_audit_trail — access audit (no Snowflake creds in this env).",
        ],
        ["record_lake"],  # gdpr_audit_trail was skipped at startup
    )
    h = score_stage("query_generation", [_query("record_lake")], ctx)
    assert "source_without_query" not in codes(h)


def test_a_short_source_id_does_not_match_arbitrary_prose():
    """A 1-3 char id would substring-match almost any sentence; require >= 4 chars."""
    ctx = _engine_ctx(["Some entirely unrelated data feed"], ["db"])
    h = score_stage("query_generation", [_query("db")], ctx)
    assert "source_without_query" not in codes(h)


def test_one_reason_per_source_even_when_prose_repeats_it():
    """Two paragraphs naming the same missing source is one missing source."""
    ctx = _engine_ctx(
        [
            "settlement_report — amounts for the issued documents.",
            "settlement_report (SETTLEMENT/BATCH) — again, for the voided ones.",
        ],
        ["settlement_report", "record_lake"],
    )
    h = score_stage("query_generation", [_query("record_lake")], ctx)
    reason = [r for r in h.reasons if r.code == "source_without_query"][0]
    assert reason.count == 1


def test_undated_queries_scale_with_count():
    h = score_stage(
        "query_generation",
        [_query(date_from=None), _query(date_to=None), _query()],
        _Ctx(),
    )
    reason = [r for r in h.reasons if r.code == "query_without_window"][0]
    assert reason.count == 2
    assert h.score == pytest.approx(1.0 - 0.4)


# --- log_retrieval -----------------------------------------------------------


def test_zero_rows_everywhere_is_fatal():
    h = score_stage("log_retrieval", {"a": [], "b": []}, _Ctx())
    assert h.score == 0.0
    assert codes(h) == {"no_rows_at_all"}


def test_empty_sources_penalty_is_pro_rata():
    logs = {"a": [{"x": 1}], "b": [], "c": [], "d": []}  # 3 of 4 empty
    h = score_stage("log_retrieval", logs, _Ctx())
    reason = [r for r in h.reasons if r.code == "empty_sources"][0]
    assert reason.weight == pytest.approx(0.4 * 0.75)


# --- empty is not always a defect (SourceDef.zero_rows) -----------------------
# The domain judgement lives in the pack (`SourceDef.zero_rows`), the arithmetic here.
# An empty result from an exclusion check is the finding, not a defect.


def _pack_with_zero_rows(spec):
    """A ctx whose log_retrieval engine carries a pack declaring ``zero_rows``."""
    sources = []
    for name, zero_rows in spec.items():
        src = MagicMock()
        src.name = name
        src.zero_rows = zero_rows
        sources.append(src)
    engine = MagicMock()
    engine.knowledge_pack.sources = sources
    engine.row_caps.return_value = {}
    return _Ctx(modules={"log_retrieval": engine})


def test_a_source_that_answers_by_being_empty_is_not_a_defect():
    """weight 0.0 means an empty result is a valid ANSWER, so nothing fires at all."""
    ctx = _pack_with_zero_rows(
        {"automation_registry": {"health_weight": 0.0, "meaning": "actor is HUMAN"}}
    )
    logs = {"scheme": [{"x": 1}], "automation_registry": []}
    h = score_stage("log_retrieval", logs, ctx)
    assert "empty_sources" not in codes(h)
    assert h.score == pytest.approx(1.0)


def test_the_measured_incident_scores_the_defect_not_the_valid_negatives():
    """Job ae050dbf's exact shape: 4 of 9 empty, but only one is a real gap."""
    ctx = _pack_with_zero_rows(
        {
            "automation_registry": {"health_weight": 0.0, "meaning": "actor is HUMAN"},
            "payment_alerts": {"health_weight": 0.2, "meaning": "card-tender only"},
            "admin_action_history": {
                "health_weight": 0.3,
                "meaning": "conditional for SCHEME",
            },
        }
    )
    logs = {f"ok{i}": [{"x": 1}] for i in range(5)}
    logs.update(
        {
            "app_session_events": [],  # the real defect — undeclared, counts in full
            "admin_action_history": [],
            "payment_alerts": [],
            "automation_registry": [],
        }
    )
    reason = [
        r
        for r in score_stage("log_retrieval", logs, ctx).reasons
        if r.code == "empty_sources"
    ][0]
    # 1.0 + 0.3 + 0.2 + 0.0 = 1.5 over 9 sources, not 4 over 9.
    assert reason.weight == pytest.approx(0.4 * (1.5 / 9))
    assert reason.weight < 0.4 * (4 / 9)


def test_the_discount_is_visible_in_the_reason_not_just_the_number():
    """A silently smaller penalty is one an operator cannot check."""
    ctx = _pack_with_zero_rows(
        {
            "automation_registry": {
                "health_weight": 0.0,
                "meaning": "no row means the actor is not a registered robot",
            }
        }
    )
    logs = {"a": [{"x": 1}], "b": [], "automation_registry": []}
    reason = [
        r
        for r in score_stage("log_retrieval", logs, ctx).reasons
        if r.code == "empty_sources"
    ][0]
    assert "automation_registry" in reason.detail
    assert "not a registered robot" in reason.detail
    # ...and the source that IS a defect is still named.
    assert "b" in reason.detail


def test_a_source_that_answers_by_being_empty_is_not_listed_as_a_gap():
    """The arithmetic excluded it; the sentence must too.

    Reported from a live run: the reason read "3 of 8 source(s) returned zero rows:
    <eight names>" with a pack-discounted registry lookup first in that list and the
    discount note cut off by the 5-name cap, so a check that succeeded by coming back
    empty read as the penalty.
    """
    ctx = _pack_with_zero_rows(
        {
            "automation_registry": {
                "health_weight": 0.0,
                "meaning": "no row means the actor is not a registered robot",
            }
        }
    )
    logs = {"a": [{"x": 1}], "real_gap": [], "automation_registry": []}
    reason = [
        r
        for r in score_stage("log_retrieval", logs, ctx).reasons
        if r.code == "empty_sources"
    ][0]
    counted, answered = reason.detail.split("ANSWERED by being empty")
    assert "real_gap" in counted
    assert "automation_registry" not in counted
    assert "automation_registry" in answered
    assert "not a registered robot" in answered
    # One counted source of three, not three of three.
    assert reason.detail.startswith("1 of 3 source(s)")


def test_an_undeclared_pack_scores_exactly_as_before():
    ctx = _pack_with_zero_rows({})
    logs = {"a": [{"x": 1}], "b": [], "c": [], "d": []}
    plain = score_stage("log_retrieval", logs, _Ctx())
    with_pack = score_stage("log_retrieval", logs, ctx)
    assert with_pack.score == pytest.approx(plain.score)


def test_a_malformed_weight_is_ignored_rather_than_crashing_the_scorer():
    """The scorer runs in every job's stage-completion path; it must never raise."""
    ctx = _pack_with_zero_rows(
        {
            "bad": {"health_weight": "not-a-number"},
            "clamped_high": {"health_weight": 7},
            "clamped_low": {"health_weight": -3},
        }
    )
    logs = {"a": [{"x": 1}], "bad": [], "clamped_high": [], "clamped_low": []}
    h = score_stage("log_retrieval", logs, ctx)
    assert h.scored
    reason = [r for r in h.reasons if r.code == "empty_sources"][0]
    # bad -> ignored (full 1.0), high -> 1.0, low -> 0.0
    assert reason.weight == pytest.approx(0.4 * (2.0 / 4))


def test_timeouts_and_failures_come_from_recorded_facts_not_the_output():
    """A timed-out source and an empty one look identical in `logs`."""
    logs = {"a": [{"x": 1}], "slow": [], "broken": []}
    facts = {
        "log_retrieval": {
            "sources": {
                "a": {"status": "completed"},
                "slow": {"status": "timeout"},
                "broken": {"status": "failed"},
            }
        }
    }
    h = score_stage("log_retrieval", logs, _Ctx(stage_facts=facts))
    assert "source_timeout" in codes(h)
    assert "source_failed" in codes(h)


def test_a_source_at_exactly_its_cap_is_reported_as_truncated():
    """Truncation is invisible from the row count alone — this is the 500-of-89,937 bug."""
    engine = MagicMock()
    engine.row_caps.return_value = {"capped": 3, "fine": 500}
    logs = {"capped": [{}, {}, {}], "fine": [{}]}
    h = score_stage("log_retrieval", logs, _Ctx(modules={"log_retrieval": engine}))
    assert "source_truncated" in codes(h)
    detail = [r for r in h.reasons if r.code == "source_truncated"][0].detail
    assert "capped" in detail and "fine" not in detail


def test_one_capped_source_cannot_gate_the_stage_on_its_own():
    """The row cap is the operator's own `max_results`, and a capped source ANSWERED.

    It stays a deduction — "N rows" and "N rows, and there were more" are different findings,
    and truncation is reported everywhere the count is read — but in the same tier as the
    other configured-limit signals, so it pauses a run only when it is pervasive.
    """
    at_cap = [{}, {}]
    engine = MagicMock()
    engine.row_caps.return_value = {f"s{i}": 2 for i in range(5)}
    ctx = _Ctx(modules={"log_retrieval": engine})
    one = score_stage("log_retrieval", {"s0": at_cap, "ok": [{}]}, ctx)
    assert one.score == pytest.approx(0.9)
    assert one.gate_recommended is False
    many = score_stage(
        "log_retrieval", {f"s{i}": at_cap for i in range(5)} | {"ok": [{}]}, ctx
    )
    assert many.score == pytest.approx(0.5)
    assert many.gate_recommended is True


def test_row_caps_raising_does_not_break_scoring():
    engine = MagicMock()
    engine.row_caps.side_effect = RuntimeError("engine gone")
    h = score_stage(
        "log_retrieval", {"a": [{"x": 1}]}, _Ctx(modules={"log_retrieval": engine})
    )
    assert h.scored is True
    assert "source_truncated" not in codes(h)


# --- correlation -------------------------------------------------------------


def test_healthy_correlation_scores_one():
    h = score_stage("correlation", _correlation(), _Ctx())
    assert h.score == 1.0


def test_absent_correlation_module_is_not_a_health_problem():
    """`correlation` is optional wiring; None means "not configured", not "broken"."""
    h = score_stage("correlation", None, _Ctx())
    assert h.score == 1.0
    assert h.reasons == []


def test_zero_records_is_fatal():
    h = score_stage("correlation", _correlation(record_count=0), _Ctx())
    assert h.score == 0.0


def test_unresolved_join_keys_and_degraded_evidence_accumulate():
    h = score_stage(
        "correlation",
        _correlation(
            aggregations={"resolved_correlation_keys": []},
            evidence=MagicMock(degraded=True),
        ),
        _Ctx(),
    )
    assert {"unresolved_join_keys", "evidence_degraded"} <= codes(h)
    # 0.1, not the 0.3 this line carried while one flat weight covered all six rungs of the
    # degradation ladder. The property under test is that the two signals ACCUMULATE; the
    # constant is calibration, and it moved deliberately (see the two tests below).
    assert h.score == pytest.approx(1.0 - 0.4 - 0.1)
    # A `MagicMock` evidence object has a `notes` attribute that is not a list, so the clip
    # is read strictly and does not fire — the aggregation reading is the safe default.
    assert "evidence_clipped" not in codes(h)


def test_the_evidence_ladders_last_rung_is_a_different_finding_from_the_five_above_it():
    """Rungs 1-5 aggregate; rung 6 cuts the text and a source can vanish from the prompt.

    Measured over this deployment's 21 distinct incidents: 9 never degraded, 2 stopped at an
    aggregation rung, 4 reached rung 5 and **6 reached the clip**. One flat weight scored all
    twelve the same, so the rung with a proven evidence-loss incident behind it (5 of 9
    sources gone from the prompt, the system of record among them) read like dropping a
    handful of example rows — while a live run landed on exactly its own gate threshold
    because aggregation alone cost it 0.3.

    The clip is not a worse degree of aggregation. Aggregation costs detail the deterministic
    verdict never reads (it reads the rows, not this pack); the clip costs whole sources from
    the narration's input, which is the same class of defect as a truncated result.
    """
    aggregated = score_stage(
        "correlation",
        _correlation(
            evidence=SimpleNamespace(
                degraded=True, notes=["dropped per-source example rows to fit budget"]
            )
        ),
        _Ctx(),
    )
    assert "evidence_degraded" in codes(aggregated)
    assert "evidence_clipped" not in codes(aggregated)
    assert aggregated.score == pytest.approx(0.9)

    clipped = score_stage(
        "correlation",
        _correlation(
            evidence=SimpleNamespace(
                degraded=True,
                notes=["dropped per-source example rows to fit budget", CLIP_NOTE],
            )
        ),
        _Ctx(),
    )
    # Exclusive: the clip REPLACES the milder reading rather than adding to it, or one
    # degradation would be reported twice under two headings.
    assert "evidence_clipped" in codes(clipped)
    assert "evidence_degraded" not in codes(clipped)
    assert clipped.score == pytest.approx(0.7)
    detail = [r.detail for r in clipped.reasons if r.code == "evidence_clipped"][0]
    # It names the remedy, because the operator's action is a number in the config.
    assert "evidence_char_budget" in detail
    # The reason must not say "missing": that described the flat tail clip the renderer had
    # before `_thin_source_lines`, which now keeps every source's count line and drops only the
    # detail beneath it. A missing source looks like a retrieval failure, while this run
    # consulted every source and the narrator merely read less about each.
    assert "missing" not in detail
    assert "still named" in detail


def test_the_clip_note_the_scorer_looks_for_is_the_one_the_ladder_writes():
    """A constant compared across two files, so drift has to be a test failure.

    `stage_health` reads the note by value; `evidence.degrade_to_budget` writes it. Spelled
    twice, a reworded note would silently downgrade every clipped run to the aggregation
    weight — and nothing else in the run would change.
    """
    pack = EvidencePack(
        chronology=[],
        actors=[],
        sources=[],
        total_records=0,
        degraded=False,
        notes=[],
    )
    # Budget 1 is unreachable by any rung, so the ladder must fall through to the last one.
    degraded = degrade_to_budget(pack, 1)
    assert degraded.degraded is True
    assert CLIP_NOTE in degraded.notes
    assert degraded.notes[-1] == CLIP_NOTE


def test_one_degradation_is_charged_once_not_twice():
    """`brief.degraded` is a SUPERSET of `verdict.degraded`, so both codes double-charged.

    `usecases/base.py` seeds the brief's flag with `bool(verdict.degraded)` and only ever
    OR-s its own two causes onto it. Firing both deducted 0.6 for one missing-data fact and,
    stacked with an evidence rung and a skipped narration, hit exactly 1.0 — a stage scored
    0.00 while the verdict, the conditions and every source were intact.
    """
    h = score_stage(
        "correlation",
        _correlation(verdict=MagicMock(degraded=True), brief=MagicMock(degraded=True)),
        _Ctx(),
    )
    assert "verdict_degraded" in codes(h)
    assert "brief_degraded" not in codes(h)
    assert h.score == pytest.approx(0.7)


def test_a_brief_degraded_on_its_own_still_scores():
    """Its own two causes (a dropped projection leaf, an unwidenable sweep) are real."""
    h = score_stage(
        "correlation",
        _correlation(verdict=MagicMock(degraded=False), brief=MagicMock(degraded=True)),
        _Ctx(),
    )
    assert "brief_degraded" in codes(h)
    assert "verdict_degraded" not in codes(h)
    assert h.score == pytest.approx(0.7)


def test_the_deterministic_narration_path_is_not_a_defect():
    """The volume gate is a CONFIGURED choice; `summary_text` still flows downstream."""
    ctx = _Ctx(stage_facts={"correlation": {"narration": "deterministic"}})
    h = score_stage("correlation", _correlation(findings=[]), ctx)
    assert h.reasons == []
    assert h.score == 1.0


def test_narration_that_ran_and_produced_nothing_is_a_defect():
    ctx = _Ctx(stage_facts={"correlation": {"narration": "llm"}})
    h = score_stage("correlation", _correlation(findings=[]), ctx)
    assert codes(h) == {"narration_skipped"}
    detail = [r for r in h.reasons if r.code == "narration_skipped"][0].detail
    assert "ran and produced none" in detail
    assert h.score == pytest.approx(0.9)


def test_the_narration_path_falls_back_to_the_shared_module_attribute():
    """Callers that drive the stage directly record no fact; the module still knows."""
    module = MagicMock()
    module.last_narration = "deterministic"
    h = score_stage(
        "correlation", _correlation(findings=[]), _Ctx(modules={"correlation": module})
    )
    assert h.reasons == []


def test_an_unrecorded_narration_path_is_still_scored():
    """Unknown is not "designed": a defect the run cannot attribute is still a defect."""
    h = score_stage("correlation", _correlation(findings=[]), _Ctx())
    assert codes(h) == {"narration_skipped"}
    assert h.score == pytest.approx(0.9)
    assert h.gate_recommended is False


# --- anomaly_detection ------------------------------------------------------


def _detection_module(
    degraded=False,
    error="",
    threshold=0.5,
    truncation_retried=False,
    filtered_out=0,
    filtered_max=0.0,
):
    m = MagicMock()
    m.last_degraded = degraded
    m.last_error = error
    m.last_truncation_retried = truncation_retried
    # The scorer reads these three directly, not `effective_threshold()`: a clamped run
    # filters against the false-positive ceiling, and re-deriving the cut-off would fire
    # the signal on every clamped run.
    m.last_threshold_used = threshold
    m.last_filtered_out = filtered_out
    m.last_filtered_max = filtered_max
    return m


def test_a_failed_detection_llm_is_fatal_and_distinguished_from_a_clean_incident():
    """Both return []. Only the module's own flag separates them."""
    broken = _Ctx(
        modules={"anomaly_detection": _detection_module(True, "429 rate limit")}
    )
    clean = _Ctx(modules={"anomaly_detection": _detection_module(False)})

    h_broken = score_stage("anomaly_detection", [], broken)
    h_clean = score_stage("anomaly_detection", [], clean)

    assert h_broken.score == 0.0
    assert "detection_llm_failed" in codes(h_broken)
    assert "429 rate limit" in h_broken.reasons[0].detail

    assert h_clean.score == pytest.approx(0.8)
    assert codes(h_clean) == {"no_anomalies"}


def test_a_recovered_truncation_is_a_config_note_not_a_fatal_degradation():
    """The retry succeeded, so the anomaly list is COMPLETE — the only thing left to
    report is that detect_max_tokens is undersized for this data volume. Scoring it as
    the FATAL detection_llm_failed would gate a stage whose output is fine."""
    ctx = _Ctx(
        modules={"anomaly_detection": _detection_module(truncation_retried=True)}
    )
    h = score_stage("anomaly_detection", [_anomaly(0.9)], ctx)

    assert codes(h) == {"detection_truncated_retried"}
    assert h.score == pytest.approx(0.9)
    assert h.gate_recommended is False
    assert "detect_max_tokens" in h.reasons[0].detail


def test_all_anomalies_below_the_effective_threshold_fires():
    """Every item marked => the report narrates nothing, which is worth saying.

    The signal reads the MARK, not the scores: the filter no longer drops sub-threshold
    items, so "all of them" is answerable from the returned list — and it is the only
    remaining way to notice that a stage reporting N anomalies narrates zero of them.
    """
    ctx = _Ctx(
        modules={
            "anomaly_detection": _detection_module(
                threshold=0.9, filtered_out=2, filtered_max=0.5
            )
        }
    )
    h = score_stage(
        "anomaly_detection", [_anomaly(0.4, below=True), _anomaly(0.5, below=True)], ctx
    )
    assert "all_below_threshold" in codes(h)
    # The numbers come from the module's record of what it filtered, so the operator can
    # see how near the line the best finding was — 0.5 against 0.9 is a different story
    # from 0.89 against 0.9.
    detail = [r.detail for r in h.reasons if r.code == "all_below_threshold"][0]
    assert "0.90" in detail and "0.50" in detail


async def test_a_partly_marked_list_is_not_all_below_threshold():
    """One narrated item is enough. Asserted against a REAL `filter_anomalies` call,
    because the mark is the contract between these two modules and a hand-set attribute
    would pass even if the filter stopped stamping it."""
    from src.anomaly_detection import AnomalyDetectionModule
    from src.models.pydantic_models import AnomalyItem

    module = AnomalyDetectionModule({"threshold": 0.8}, MagicMock())
    marked = module.filter_anomalies(
        [
            AnomalyItem(
                description=d,
                supporting_data="rows",
                potential_implications="—",
                confidence_score=s,
                recommended_actions="—",
                patterns="—",
            )
            for d, s in (("kept", 0.9), ("held", 0.4))
        ]
    )

    h = score_stage("anomaly_detection", marked, _Ctx(modules={"anomaly_detection": module}))

    assert [a.below_threshold for a in marked] == [False, True]
    assert "all_below_threshold" not in codes(h)


def test_anomalies_above_threshold_score_clean():
    ctx = _Ctx(modules={"anomaly_detection": _detection_module(threshold=0.5)})
    h = score_stage("anomaly_detection", [_anomaly(0.9)], ctx)
    assert h.score == 1.0


def test_an_unmarked_anomaly_list_does_not_fire_the_signal():
    """A list whose items carry no mark at all — an older stored job, a hand-built
    fixture, a MagicMock — must read as NARRATED. Defaulting the other way would gate a
    stage whose findings are all in the report."""
    ctx = _Ctx(modules={"anomaly_detection": _detection_module()})
    bare = MagicMock()
    bare.confidence_score = 0.1
    h = score_stage("anomaly_detection", [bare], ctx)
    assert "all_below_threshold" not in codes(h)
    assert h.scored is True


# --- report_generation ------------------------------------------------------


def _report_module(fallback=False, backfilled=None, error=""):
    m = MagicMock()
    m.last_fallback_used = fallback
    m.last_backfilled_sections = backfilled if backfilled is not None else []
    m.last_error = error
    return m


def test_a_deterministic_fallback_report_is_fatal():
    """A fallback report looks like a report — only the module knows it degraded."""
    ctx = _Ctx(modules={"report_generation": _report_module(True, error="timeout")})
    h = score_stage("report_generation", {"sections": []}, ctx)
    assert h.score == 0.0
    assert "report_fallback_used" in codes(h)
    assert "timeout" in h.reasons[0].detail


def test_backfilled_sections_scale_with_count():
    ctx = _Ctx(
        modules={
            "report_generation": _report_module(
                backfilled=["Executive Summary", "Analysis and Findings"]
            )
        }
    )
    h = score_stage("report_generation", {"sections": []}, ctx)
    assert h.score == pytest.approx(1.0 - 0.3)
    assert [r for r in h.reasons if r.code == "section_backfilled"][0].count == 2


def test_a_fully_narrated_report_scores_one():
    ctx = _Ctx(modules={"report_generation": _report_module()})
    h = score_stage("report_generation", {"sections": []}, ctx)
    assert h.score == 1.0


# --- unscored stages, config, robustness ------------------------------------


@pytest.mark.parametrize("stage", ["plugins", "export", "output"])
def test_non_gateable_stages_are_reported_unscored_not_perfect(stage):
    """An unscored stage must not masquerade as a suspiciously perfect 1.0."""
    h = score_stage(stage, ["anything"], _Ctx())
    assert h.scored is False
    assert h.gate_recommended is False
    assert stage_gate_enabled({}, stage) is False


def test_every_gateable_stage_has_signals_defined():
    for stage in GATEABLE_STAGES:
        h = score_stage(stage, None, _Ctx())
        assert h.scored is True, f"{stage} has no signal function"


def test_configured_weight_overrides_the_default():
    cfg = {"stage_gates": {"weights": {"no_event_time": 0.05}}}
    h = score_stage("understanding", _understanding(event_time=None), _Ctx(), cfg)
    assert h.score == pytest.approx(0.95)


def test_a_weight_of_zero_disables_a_signal_entirely():
    cfg = {"stage_gates": {"weights": {"narration_skipped": 0.0}}}
    h = score_stage("correlation", _correlation(findings=[]), _Ctx(), cfg)
    assert h.reasons == []
    assert h.score == 1.0


def test_an_unknown_weight_key_is_ignored_not_fatal():
    cfg = {"stage_gates": {"weights": {"not_a_signal": 0.5}}}
    h = score_stage("understanding", _understanding(), _Ctx(), cfg)
    assert h.score == 1.0


def test_a_non_numeric_weight_falls_back_to_defaults():
    cfg = {"stage_gates": {"weights": {"no_event_time": "loads"}}}
    h = score_stage("understanding", _understanding(event_time=None), _Ctx(), cfg)
    assert h.scored is True
    assert h.score == pytest.approx(0.7)


def test_per_stage_threshold_beats_global_beats_default():
    cfg = {
        "stage_gates": {
            "threshold": 0.5,
            "stages": {"report_generation": {"threshold": 0.9}},
        }
    }
    assert stage_threshold(cfg, "report_generation") == 0.9
    assert stage_threshold(cfg, "understanding") == 0.5
    assert stage_threshold({}, "understanding") == DEFAULT_THRESHOLD


def test_a_non_numeric_threshold_falls_back_to_the_default():
    cfg = {"stage_gates": {"threshold": "high"}}
    assert stage_threshold(cfg, "understanding") == DEFAULT_THRESHOLD


def test_gate_enablement_defaults_on_for_gateable_stages_and_is_disableable():
    assert stage_gate_enabled({}, "correlation") is True
    cfg = {"stage_gates": {"stages": {"correlation": {"enabled": False}}}}
    assert stage_gate_enabled(cfg, "correlation") is False


def test_the_threshold_is_carried_on_the_result_so_a_score_is_interpretable():
    cfg = {"stage_gates": {"threshold": 0.75}}
    h = score_stage("understanding", _understanding(), _Ctx(), cfg)
    assert h.threshold == 0.75
    assert h.to_dict()["threshold"] == 0.75


def test_scoring_never_raises_on_a_garbage_output():
    """The scorer runs on every stage completion; it must not fail a good job."""
    for junk in (None, 42, "text", object(), {"unexpected": True}, [1, 2, 3]):
        for stage in GATEABLE_STAGES:
            h = score_stage(stage, junk, _Ctx())
            assert 0.0 <= h.score <= 1.0


def test_scoring_with_no_context_at_all_is_safe():
    h = score_stage("log_retrieval", {"a": [{"x": 1}]}, None)
    assert 0.0 <= h.score <= 1.0


def test_a_signal_function_that_explodes_yields_unscored_not_an_exception(monkeypatch):
    import src.stage_health as sh

    monkeypatch.setitem(
        sh._SIGNAL_FNS, "correlation", lambda o, c: 1 / 0  # noqa: ARG005
    )
    h = sh.score_stage("correlation", _correlation(), _Ctx())
    assert h.scored is False
    assert h.gate_recommended is False


def test_score_is_clamped_to_zero_never_negative():
    """Many signals at once must floor at 0.0, not go negative."""
    logs = {f"s{i}": [] for i in range(5)}
    logs["ok"] = [{"x": 1}]
    facts = {
        "log_retrieval": {"sources": {f"s{i}": {"status": "failed"} for i in range(5)}}
    }
    h = score_stage("log_retrieval", logs, _Ctx(stage_facts=facts))
    assert h.score == 0.0


def test_to_dict_is_json_safe_and_carries_the_reasons():
    import json

    h = score_stage("understanding", _understanding(event_time=None), _Ctx())
    doc = h.to_dict()
    json.dumps(doc)  # must not raise
    assert doc["reasons"][0]["code"] == "no_event_time"
    assert doc["reasons"][0]["detail"]
    assert doc["scored"] is True
