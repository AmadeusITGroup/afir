"""Tests for ApiCallGenerator source selection.

Focus: the LLM must be offered ONLY sources the retrieval engine actually built a
retriever for. A serveable-kind pack source whose backend creds are missing is skipped
at engine build, so advertising it leads to "No retriever configured for log source"
at query time. `available_sources` closes that gap.

Second focus: sources a validation ruleset DECLARES are a data dependency — but the
selection is still the planner's, so the dependency is REPORTED and never injected. See
`test_declared_ruleset_sources_are_reported_and_never_added`.
"""

from types import SimpleNamespace

import pytest

from src.api_call_generator import ApiCallGenerator
from src.models.pydantic_models import ExtractedEntity, RetrievalQuery


def _pack(*specs, verdicts=None):
    """Build a fake KnowledgePack exposing .sources with .name + .kind()."""
    sources = [
        SimpleNamespace(name=name, kind=lambda k=kind: k, description="")
        for name, kind in specs
    ]
    # Three selection methods the planner calls. Without them, a deferred target whose
    # ruleset is not adjudicating is asked on no pass, degrading to single-pass behaviour.
    keys = list((verdicts or {}).keys())
    return SimpleNamespace(
        sources=sources,
        rulesets={"verdicts": verdicts} if verdicts else {},
        source=lambda n: next((s for s in sources if s.name == n), None),
        correlation_specs=lambda: {},
        ruleset_key_for=lambda use_case: (
            str(use_case) if str(use_case) in (verdicts or {}) else ""
        ),
        default_ruleset_key=lambda: keys[0] if keys else "",
    )


def test_source_names_unfiltered_when_no_available_sources():
    """Back-compat: with available_sources=None, every serveable-kind source is offered."""
    pack = _pack(
        ("scheme_alerts", "elasticsearch"),
        ("record_lake", "databricks_uc"),
        ("ff_activity_snowflake", "snowflake"),  # not serveable-kind
    )
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=pack)
    names = gen._source_names()
    assert "scheme_alerts" in names
    assert "record_lake" in names
    assert "ff_activity_snowflake" not in names  # snowflake kind not advertised


def test_source_names_filtered_to_built_retrievers():
    """A serveable-kind source with no built retriever must NOT be advertised."""
    pack = _pack(
        ("scheme_alerts", "elasticsearch"),  # skipped at engine build (no creds)
        ("siem_alerts", "elasticsearch"),  # skipped
        ("record_lake", "databricks_uc"),  # built
        ("admin_action_history", "databricks_uc"),  # built
    )
    gen = ApiCallGenerator(
        {},
        llm_client=None,
        knowledge_pack=pack,
        available_sources=["record_lake", "admin_action_history"],
    )
    names = gen._source_names()
    assert set(names) == {"record_lake", "admin_action_history"}
    assert "scheme_alerts" not in names
    assert "siem_alerts" not in names


def test_source_names_falls_back_when_no_overlap():
    """If nothing overlaps the built set, offer the unfiltered list rather than empty."""
    pack = _pack(("scheme_alerts", "elasticsearch"))
    gen = ApiCallGenerator(
        {},
        llm_client=None,
        knowledge_pack=pack,
        available_sources=["some_other_source"],
    )
    names = gen._source_names()
    # No overlap -> fall back to the unfiltered serveable list (never empty).
    assert names == ["scheme_alerts"]


def _ruleset_pack():
    """A pack whose SCHEME ruleset declares 5 logical sources -> 5 real ones."""
    return _pack(
        ("record_lake", "databricks_uc"),
        ("settlement_report", "databricks_uc"),
        ("auth_events", "databricks_uc"),
        ("automation_registry", "databricks_uc"),
        ("record_document_scope_sweep", "databricks_uc"),
        ("payment_alerts", "elasticsearch"),  # not a ruleset dependency
        verdicts={
            "scheme": {
                "sources": {
                    "record": "record_lake",
                    "settlement": "settlement_report",
                    "auth": "auth_events",
                    "automation": "automation_registry",
                    "scope_sweep": "record_document_scope_sweep",
                }
            }
        },
    )


def _analysis(entities=None, event_time=None, summary=""):
    """An analysis stand-in that DID identify a subject.

    Non-empty `extracted_entities` is the realistic case, and it is what makes a declared
    source *scopable*: every ruleset condition is a question about the incident's entities,
    so a dependency the incident can bind no entity to could only be asked as a bare
    date-window scan. One predicate, read from two ends: for a source the planner did NOT
    pick it is reported (`unscopable`) and never injected; for one the planner DID pick the
    query is dropped rather than issued (`_drop_unscopable`). Reporting never adds evidence
    and the drop never invents any, which is why only the second acts.
    """
    if entities is None:
        entities = [ExtractedEntity(type="record", value="ABC123", raw="ABC123")]
    return SimpleNamespace(
        model_dump_json=lambda indent=None: "{}",
        event_time=event_time,
        extracted_entities=entities,
        # The prose the ruleset SELECTION reads. Empty for every test that does not care,
        # which is the honest default: with no summary the selector has no evidence and the
        # planner falls back to the pack's declared default ruleset.
        incident_summary=summary,
        initial_hypotheses=[],
        key_investigation_areas=[],
    )


def test_declared_ruleset_sources_are_reported_and_never_added():
    """A ruleset's declared sources are a data dependency and a finding, not an injected query.

    Both halves are asserted: the report names the unmet dependency AND the query list is
    untouched. Either alone is the old defect (silent gap) or the old cure (silent injection).
    """
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_ruleset_pack())
    names = gen._source_names()
    # The planner chose only two of the five declared sources.
    chosen = [
        RetrievalQuery(
            target_log_source="record_lake",
            natural_language_query="record versions",
            date_from="2026-07-16",
            date_to="2026-07-18",
        ),
        RetrievalQuery(
            target_log_source="payment_alerts",
            natural_language_query="payment_svc",
            date_from="2026-07-16",
            date_to="2026-07-18",
        ),
    ]
    report = gen.dependency_report(chosen, names, _analysis())
    assert set(report["not_queried"]) == {
        "settlement_report",
        "auth_events",
        "automation_registry",
        "record_document_scope_sweep",
    }
    # The one the planner DID pick is not a gap, and a source no ruleset declares is in no
    # list at all — `payment_alerts` is the planner's own judgement and stays unremarked.
    assert "record_lake" not in report["not_queried"]
    assert "payment_alerts" not in report["not_queried"]
    assert report["undeliverable"] == []

    # And the mechanism that used to close the gap is GONE, not merely unreached: a helper
    # left behind is a helper a later edit re-wires.
    assert not hasattr(gen, "_add_missing_required")
    assert not hasattr(gen, "_applicable_rulesets")
    # Reporting is pure: the plan it was handed is the plan that comes out.
    assert [q.target_log_source for q in chosen] == ["record_lake", "payment_alerts"]


def test_required_sources_not_added_when_unavailable_or_no_ruleset():
    """Never advertise a source the engine did not build, and stay inert with no ruleset.

    An unbuildable dependency must still be REPORTED as one: dropping it silently made a
    mandatory source indistinguishable from an undeclared one (job 4da14f65).
    """
    # automation_registry declared by the ruleset but NOT built (no creds) -> not added,
    # because retrieval would fail with "No retriever configured".
    pack = _ruleset_pack()
    gen = ApiCallGenerator(
        {},
        llm_client=None,
        knowledge_pack=pack,
        available_sources=["record_lake", "payment_alerts"],
    )
    required, undeliverable = gen._required_sources(gen._source_names())
    assert required == ["record_lake"]
    # Not merely absent from `required` — every unbuildable dependency is NAMED, so the
    # run can gate on it. `payment_alerts` was built but is not a dependency, so it is
    # in neither list.
    assert sorted(undeliverable) == [
        "auth_events",
        "automation_registry",
        "record_document_scope_sweep",
        "settlement_report",
    ]

    # A pack with no ruleset adds nothing.
    plain = _pack(("record_lake", "databricks_uc"))
    gen2 = ApiCallGenerator({}, llm_client=None, knowledge_pack=plain)
    assert gen2._required_sources(gen2._source_names()) == ([], [])
    assert ApiCallGenerator({}, llm_client=None)._required_sources(["x"]) == ([], [])


def _two_ruleset_pack():
    """A pack shipping TWO procedures about DIFFERENT subjects, sharing one source.

    The shape that makes the scoping matter and the only shape that can show it: with one
    ruleset there is nothing to exclude, and with two rulesets about the SAME subject
    neither is ever excluded. `auth_events` is declared by both, so the test also pins that
    a shared dependency survives — an exclusion that took a source away from the ruleset
    that DOES apply would be the far worse bug.
    """
    pack = _pack(
        ("record_lake", "databricks_uc"),
        ("settlement_report", "databricks_uc"),
        ("auth_events", "databricks_uc"),
        ("admin_action_history", "databricks_uc"),
        ("automation_registry", "databricks_uc"),
        verdicts={
            "scheme": {
                "subject_entity": "record",
                "sources": {
                    "record": "record_lake",
                    "settlement": "settlement_report",
                    "auth": "auth_events",
                },
            },
            "admin": {
                "subject_entity": "operator",
                "sources": {
                    "admin": "admin_action_history",
                    "automation": "automation_registry",
                    "auth": "auth_events",
                },
            },
        },
    )
    # Two distinct prose titles, because WHICH procedure adjudicates is the whole scope of a
    # dependency and a fake that answered the same for both incidents would be vacuous.
    pack.correlation_specs = lambda: [
        {"use_case": "scheme", "title": "Refund scheme on a record", "keys": ["record"]},
        {"use_case": "admin", "title": "Operator privilege misuse", "keys": ["operator"]},
    ]
    return pack


def test_another_rulesets_hard_dependencies_do_not_attach():
    """Only the adjudicating procedure's declared sources are this run's dependencies.

    The key is resolved via `ruleset_key_for`, matching the verdict stage's own resolution,
    so the dependency report and the procedure it reports on cannot disagree.
    """
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_two_ruleset_pack())
    names = gen._source_names()

    # An incident the admin procedure adjudicates. The scheme procedure's record/settlement
    # sources are not this run's dependencies — no condition of it will be evaluated.
    admin_incident = _analysis(
        entities=[ExtractedEntity(type="operator", value="OP1", raw="OP1")],
        summary="operator OP1 committed privilege misuse",
    )
    required, undeliverable = gen._required_sources(names, admin_incident)
    assert set(required) == {
        "admin_action_history",
        "automation_registry",
        "auth_events",
    }
    assert "record_lake" not in required
    assert "settlement_report" not in required
    # Not silently reclassified as unmeetable either — they are simply not dependencies.
    assert undeliverable == []

    # And symmetrically, so this is selection and not a hard-coded preference. `auth_events`
    # is declared by both and survives either way: a shared dependency dropped from the
    # ruleset that DOES apply would be the far worse bug.
    record_incident = _analysis(
        entities=[ExtractedEntity(type="record", value="ABC123", raw="ABC123")],
        summary="a refund scheme was run against record ABC123",
    )
    required2, _ = gen._required_sources(names, record_incident)
    assert set(required2) == {"record_lake", "settlement_report", "auth_events"}
    assert "automation_registry" not in required2


def test_an_unresolved_procedure_falls_back_to_one_ruleset_and_never_the_union():
    """The three ways selection can fail to resolve, and none of them may widen the scope.

    The mechanism this replaced was permissive on failure — it kept EVERY ruleset's sources,
    on the theory that a wrong exclusion degrades a decisive check two stages later. That is
    true of an injected query and false of a reported one: over-reporting names sources no
    condition will read as gaps in the run, which is how the report's "returned nothing" list
    stopped being readable. So an unresolved selection falls back to the pack's DEFAULT
    ruleset — the same fallback the verdict stage takes — and one procedure's sources can
    never ride on another procedure's run.
    """
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_two_ruleset_pack())
    names = gen._source_names()
    default_only = {"record_lake", "settlement_report", "auth_events"}
    others = {"admin_action_history", "automation_registry"}

    # 1. No analysis at all (the unscoped call site).
    assert set(gen._required_sources(names)[0]) == default_only
    # 2. An analysis that extracted nothing.
    assert set(gen._required_sources(names, _analysis(entities=[]))[0]) == default_only
    # 3. Entities and a summary that score for no procedure in particular.
    unrelated = _analysis(
        entities=[ExtractedEntity(type="org_unit", value="OU1", raw="OU1")],
        summary="something happened at org unit OU1",
    )
    assert set(gen._required_sources(names, unrelated)[0]) == default_only
    for case in (None, _analysis(entities=[]), unrelated):
        got = set(gen._required_sources(names, case)[0])
        assert not (got & others), f"{sorted(got & others)} attached to another run"

    # And a pack that resolves nothing at all — no default either — reports no dependency
    # rather than every one of them.
    nameless = _pack(
        ("record_lake", "databricks_uc"),
        verdicts={"scheme": {"sources": {"a": "record_lake"}}},
    )
    nameless.default_ruleset_key = lambda: ""
    gen2 = ApiCallGenerator({}, llm_client=None, knowledge_pack=nameless)
    assert gen2._required_sources(gen2._source_names(), unrelated) == ([], [])


@pytest.mark.asyncio
async def test_generate_reports_the_gap_and_returns_the_planners_own_plan():
    """The whole `generate` path: the LLM omits four declared sources and nothing is added.

    The wiring test for the removal. `_report_unmet_dependencies` returns nothing precisely
    so that this cannot regress by accident — but a call site can still be re-plumbed, so
    the assertion is on the returned plan (unchanged) AND on the recorded finding (present).
    """
    from unittest.mock import AsyncMock, MagicMock

    call = SimpleNamespace(
        function=SimpleNamespace(
            name="create_retrieval_query",
            arguments=RetrievalQuery(
                target_log_source="record_lake",
                natural_language_query="record versions",
                date_from="2026-07-16",
                date_to="2026-07-18",
            ).model_dump_json(),
        )
    )
    llm = MagicMock()
    llm.tool_call = AsyncMock(return_value=SimpleNamespace(tool_calls=[call]))

    pack = _ruleset_pack()
    pack.catalog_prompt = lambda names=None: "catalog"
    gen = ApiCallGenerator({}, llm_client=llm, knowledge_pack=pack)
    understanding = SimpleNamespace(incident_id="i1", analysis=_analysis())
    queries = await gen.generate(understanding)
    targets = [q.target_log_source for q in queries]
    # Exactly what the planner asked for. The four declared sources it did not pick are the
    # stage's finding, which `stage_health` scores as `required_source_not_queried`.
    assert targets == ["record_lake"]
    assert set(gen.declared_not_queried) == {
        "settlement_report",
        "auth_events",
        "automation_registry",
        "record_document_scope_sweep",
    }
    assert gen.undeliverable_required == []


@pytest.mark.asyncio
async def test_no_entities_means_no_declared_source_scan():
    """An incident with no entities must not queue the declared sources.

    With no entities, a declared source's `_fallback_request` degrades to an unbounded
    date-window scan whose rows answer no condition. These are reported as `unscopable`,
    not as gaps: declining them was correct.
    """
    from unittest.mock import AsyncMock, MagicMock

    call = SimpleNamespace(
        function=SimpleNamespace(
            name="create_retrieval_query",
            arguments=RetrievalQuery(
                target_log_source="record_lake",
                natural_language_query="record versions",
                date_from="2026-07-16",
                date_to="2026-07-18",
            ).model_dump_json(),
        )
    )
    llm = MagicMock()
    llm.tool_call = AsyncMock(return_value=SimpleNamespace(tool_calls=[call]))

    pack = _ruleset_pack()
    pack.catalog_prompt = lambda names=None: "catalog"
    gen = ApiCallGenerator({}, llm_client=llm, knowledge_pack=pack)
    understanding = SimpleNamespace(incident_id="i1", analysis=_analysis(entities=[]))

    queries = await gen.generate(understanding)

    assert [q.target_log_source for q in queries] == ["record_lake"]
    # And the four unmet dependencies are reported as unscopable, not as gaps: with no
    # entities, no query against them could have been scoped to this incident, so declining
    # them was right and the health gate must not read them as a planner defect.
    assert gen.declared_not_queried == []
    assert set(gen.declared_unscopable) == {
        "settlement_report",
        "auth_events",
        "automation_registry",
        "record_document_scope_sweep",
    }


# ── a follow-up pass's target is not asked on THIS pass ──────────────────────


def _follow_up_pack(*, declared_target=True):
    """A pack whose ruleset declares a follow-up pass over one of its own sources.

    A follow-up target is reachable two ways and both have to be closed, which is why the
    fixture declares it in `sources:` (so it is a hard dependency the report would name)
    while the tests also hand the planner a query against it (so the LLM's own pick, the
    route that fired live, is covered).
    """
    sources = {"record": "record_lake", "auth": "auth_events"}
    if declared_target:
        sources["access"] = "access_trail_lake"
    pack = _pack(
        ("record_lake", "databricks_uc"),
        ("auth_events", "databricks_uc"),
        ("access_trail_lake", "databricks_uc"),
        verdicts={"scheme": {"sources": sources}},
    )
    pack.follow_up_passes = lambda key="": [
        {
            "pass": 2,
            "source": "access_trail_lake",
            "harvest": [
                {"entity": "counterparty", "source": "record_lake", "fields": ["x"]}
            ],
            "capture": "",
            "window": "onwards",
            "skip_when_empty": True,
            "purpose": "",
        }
    ]
    return pack


def test_a_follow_up_targets_source_is_not_a_dependency_of_this_pass():
    """The declaration that makes the pass VALID must not report the query it replaces missing.

    `pack_validate` requires a follow-up `source:` to be in the ruleset's `sources:` map —
    that is the only place a logical name resolves — so every follow-up target is also, by
    construction, a hard dependency. Read both ways at once, the stage names as a gap the
    exact wrong-scope first-pass scan the pass exists to remove (measured: 0 rows in 306.8s),
    and an operator acting on that report re-creates it. The declaration wins.
    """
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_follow_up_pack())
    names = gen._source_names()
    required, undeliverable = gen._required_sources(names, _analysis())
    assert "access_trail_lake" not in required
    # The other dependencies are untouched — deferring one target is not opting out of the
    # dependency read.
    assert set(required) == {"record_lake", "auth_events"}
    assert undeliverable == []


def test_a_follow_up_target_the_planner_picked_is_dropped_from_this_pass():
    """The planner reaches the source from the CATALOG and cannot see the declaration.

    Which is how the live session-anomaly run asked the access trail about the acting identity as the
    retriever: a coherent question, not the procedure's — and the only route left now that
    nothing is added, so this deferral is the whole mechanism rather than half of it.
    """
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_follow_up_pack())
    chosen = [
        RetrievalQuery(
            target_log_source="record_lake",
            natural_language_query="record versions",
            date_from="2026-08-01",
            date_to="2026-08-08",
        ),
        RetrievalQuery(
            target_log_source="access_trail_lake",
            natural_language_query="reads by the acting identity",
            date_from="2026-08-01",
            date_to="2026-08-08",
        ),
    ]
    kept = gen._defer_follow_up_targets(chosen, _analysis())
    assert [q.target_log_source for q in kept] == ["record_lake"]


def test_EVERY_target_of_a_shared_pass_is_deferred_out_of_this_pass():
    """Deferral reads the whole target list, or the second target keeps its wrong-scope query.

    A shared pass exists because the harvest is what its targets have in common, and the
    harvest is a value no first-pass query can carry — so the reason to defer applies to each
    target identically. Reading only the first would leave the second asked on pass 1 with
    whatever the incident happened to carry, which is the measured failure this mechanism
    replaces (a record locator on an office-name column, `completed, 0 rows`) — and it would
    then be asked AGAIN on pass 2, paying for the scan twice.
    """
    pack = _follow_up_pack()
    pack.follow_up_passes = lambda key="": [
        {
            "pass": 2,
            "source": "access_trail_lake",
            "sources": ["access_trail_lake", "auth_events"],
            "harvest": [
                {"entity": "counterparty", "source": "record_lake", "fields": ["x"]}
            ],
            "capture": "",
            "window": "onwards",
            "skip_when_empty": True,
            "purpose": "",
        }
    ]
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=pack)
    assert gen._follow_up_targets(_analysis()) == {"access_trail_lake", "auth_events"}
    chosen = [
        RetrievalQuery(
            target_log_source=name,
            natural_language_query="q",
            date_from="2026-08-01",
            date_to="2026-08-08",
        )
        for name in ("record_lake", "access_trail_lake", "auth_events")
    ]
    kept = gen._defer_follow_up_targets(chosen, _analysis())
    assert [q.target_log_source for q in kept] == ["record_lake"]
    # ...and neither is reported as a missing dependency of the pass that no longer asks it.
    required, _ = gen._required_sources(gen._source_names(), _analysis())
    assert required == ["record_lake"]


def test_deferral_is_inert_for_a_pack_declaring_no_follow_up_pass():
    """The no-regression pin: every pack that exists today declares no pass.

    Asserted on object identity, because "unchanged" here means the planner's own list
    reaches enrichment untouched — not an equal copy.
    """
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_ruleset_pack())
    chosen = [
        RetrievalQuery(
            target_log_source="record_lake",
            natural_language_query="record versions",
            date_from="2026-08-01",
            date_to="2026-08-08",
        )
    ]
    assert gen._defer_follow_up_targets(chosen, _analysis()) is chosen
    assert gen._follow_up_targets(_analysis()) == set()
    # And a pack with no ruleset at all, reached through the same call.
    bare = ApiCallGenerator({}, llm_client=None, knowledge_pack=None)
    assert bare._defer_follow_up_targets(chosen, _analysis()) is chosen


def test_an_undeliverable_follow_up_target_is_still_reported():
    """Deferring is about WHICH pass asks, never about whether the dependency is met.

    A follow-up target that built no retriever asks nothing on any pass, and the conditions
    reading it go `unknown` exactly as they do for a first-pass dependency — so it must stay
    in `undeliverable`, which is the list the run gates on.
    """
    gen = ApiCallGenerator(
        {},
        llm_client=None,
        knowledge_pack=_follow_up_pack(),
        available_sources=["record_lake", "auth_events"],
    )
    required, undeliverable = gen._required_sources(gen._source_names(), _analysis())
    assert required == ["record_lake", "auth_events"]
    assert undeliverable == ["access_trail_lake"]


@pytest.mark.asyncio
async def test_generate_does_not_plan_a_follow_up_target_end_to_end():
    """The WIRING, which every unit test above passes without.

    The route that survives the force-add's removal is the one that fired live: the LLM picks
    the target itself, from the catalog, having no way to see the declaration. So `generate`
    must call the deferral — and a deferred target must also be absent from the dependency
    report, or the stage reports a gap against the pass that was told not to ask.
    """
    from unittest.mock import AsyncMock, MagicMock

    def _call(source):
        return SimpleNamespace(
            function=SimpleNamespace(
                name="create_retrieval_query",
                arguments=RetrievalQuery(
                    target_log_source=source,
                    natural_language_query="rows",
                    date_from="2026-08-01",
                    date_to="2026-08-08",
                ).model_dump_json(),
            )
        )

    llm = MagicMock()
    llm.tool_call = AsyncMock(
        return_value=SimpleNamespace(
            tool_calls=[_call("record_lake"), _call("access_trail_lake")]
        )
    )
    pack = _follow_up_pack()
    pack.catalog_prompt = lambda names=None: "catalog"
    gen = ApiCallGenerator({}, llm_client=llm, knowledge_pack=pack)
    queries = await gen.generate(
        SimpleNamespace(incident_id="i1", analysis=_analysis())
    )
    targets = [q.target_log_source for q in queries]
    # The planner picked it and the declaration overrules the pick — it is dropped, not
    # merely left un-added, because the LLM reaches the source from the CATALOG and cannot
    # see the pass. The planner's other choice stands untouched.
    assert targets == ["record_lake"]
    # And a deferred target is a gap in NO list: the procedure said which pass asks it.
    report = gen.dependency_report(queries, gen._source_names(), _analysis())
    assert "access_trail_lake" not in report["not_queried"]
    assert "access_trail_lake" not in report["undeliverable"]
    assert report["not_queried"] == ["auth_events"]


@pytest.mark.parametrize("lazy", [False, True])
def test_a_pack_whose_follow_up_declaration_raises_defers_nothing(lazy):
    """Permissive on failure, like every other read of a pack key in this module.

    A broken declaration must degrade to the single-pass behaviour — every declared source
    planned — not to a run that silently skips a hard dependency.

    Both parametrisations matter and only the second can catch a half-read: raising up
    front leaves nothing collected, so keeping the partial set would look identical to
    abandoning it. The LAZY case yields one target and then fails, which is the shape a
    real multi-ruleset read has — and the answer must still be "defer nothing", because a
    declaration that could not be read whole is not one the engine may act on half of.
    """

    def _boom(key=""):
        raise RuntimeError("malformed")

    def _half(key=""):
        yield {"pass": 2, "source": "access_trail_lake", "harvest": [{}]}
        raise RuntimeError("malformed")

    pack = _follow_up_pack()
    pack.follow_up_passes = _half if lazy else _boom
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=pack)
    assert gen._follow_up_targets(_analysis()) == set()
    required, _ = gen._required_sources(gen._source_names(), _analysis())
    assert "access_trail_lake" in required


@pytest.mark.asyncio
async def test_a_follow_up_request_asks_only_about_the_harvested_sides():
    """The whole text of the query a follow-up pass builds, on real harvested rows.

    Two properties, and both were live defects. The types offered must be the question's own
    (the source's full list invited the acting identity onto a relational query, whose
    predicate matches nothing), and the sides must be AND-ed while the values within a side
    are alternatives (OR-ing the sides answers about every office that read anything OR was
    read by anyone).
    """
    src = SimpleNamespace(
        name="access_trail_lake",
        kind=lambda: "databricks_uc",
        entities=["org_unit", "user", "reader_unit", "owner_unit", "time_window"],
        entity_bindings={
            "reader_unit": ["reader_col"],
            "owner_unit": ["owner_col"],
            "org_unit": ["reader_col"],
            "user": ["sign_col"],
        },
        not_answered_by=[],
        description="Access trail.",
    )
    pack = SimpleNamespace(
        sources=[src],
        rulesets={},
        source=lambda n: src if n == src.name else None,
        follow_up_passes=lambda key="": [],
        classify_value_form=lambda etype, value: "",
        value_forms_for=lambda etype: [],
    )
    gen = ApiCallGenerator(
        {}, llm_client=None, knowledge_pack=pack, available_sources=[src.name]
    )
    spec = {
        "pass": 2,
        "source": src.name,
        "harvest": [
            {"entity": "reader_unit", "source": "grant", "fields": ["decoded.reader"]},
            {"entity": "owner_unit", "source": "grant", "fields": ["decoded.owner"]},
        ],
        "capture": "^([A-Z0-9]{6,9})",
        "window": "onwards",
        "skip_when_empty": True,
        "purpose": "Did the readers read the owner's records.",
    }
    logs = {
        "grant": [
            {"decoded": {"reader": "UUU1V21QR/**", "owner": "JJJ1K10UV/**"}},
            {"decoded": {"reader": "UUU1V21QR/**", "owner": "JJJ1K11UV/**"}},
        ]
    }
    analysis = _analysis(
        entities=[ExtractedEntity(type="user", value="0202YZ", raw="0202YZ")]
    )
    queries, notes = await gen.generate_follow_up(
        SimpleNamespace(incident_id="i1", analysis=analysis), spec, logs
    )
    assert len(queries) == 1
    text = queries[0].natural_language_query
    # The harvested sides, with the mask suffix stripped by `capture`.
    assert "reader_unit is one of (UUU1V21QR)" in text
    assert "owner_unit is one of (JJJ1K10UV, JJJ1K11UV)" in text
    assert "/**" not in text
    # AND across the sides, alternatives within one.
    assert "combine them with AND" in text
    # The acting identity the incident carried is offered NOWHERE — not as a value, and not
    # as a type the generator is invited to filter on.
    assert "0202YZ" not in text
    assert "user" not in text
    assert "org_unit" not in text
    # The values also ride on the query itself, which is what `render_filters` reads — the
    # prose is for the LLM, the entities are the enforcement.
    carried = {(e.type, e.value) for e in (queries[0].entities or [])}
    assert carried == {
        ("reader_unit", "UUU1V21QR"),
        ("owner_unit", "JJJ1K10UV"),
        ("owner_unit", "JJJ1K11UV"),
    }
    # A clean harvest manufactures no caveat: `notes` is where a rejected value, an empty
    # side or a bounded list is reported, and inventing one here would put a gap on a run
    # that has none.
    assert notes == []


@pytest.mark.asyncio
async def test_a_shared_follow_up_pass_asks_each_TARGET_its_own_question():
    """One harvest, two targets, two queries — and the second is not a copy of the first.

    A pass number is a scarce slot (``jobs.max_retrieval_passes``), so the question that loses
    the numbering race is not asked at all: measured, it shipped a pass-1 query putting a
    record locator on an office-name column and reported 0 rows against a ground truth of 2.
    Sharing one pass costs no extra scan — but the targets share the HARVEST, not the
    question, and the purpose is the text the backend query gets written from. One purpose
    over both asks the reference table the event log's question, which it can only answer
    empty, so each target's own text has to reach its own request.
    """
    trail = SimpleNamespace(
        name="access_trail_lake",
        kind=lambda: "databricks_uc",
        entities=["reader_unit", "time_window"],
        entity_bindings={"reader_unit": ["reader_col"]},
        not_answered_by=[],
        description="Access trail.",
    )
    reference = SimpleNamespace(
        name="unit_reference",
        kind=lambda: "databricks_uc",
        entities=["reader_unit"],
        entity_bindings={"reader_unit": ["unit_code"]},
        not_answered_by=[],
        description="Unit reference data.",
    )
    by_name = {s.name: s for s in (trail, reference)}
    pack = SimpleNamespace(
        sources=[trail, reference],
        rulesets={},
        source=by_name.get,
        follow_up_passes=lambda key="": [],
        classify_value_form=lambda etype, value: "",
        value_forms_for=lambda etype: [],
    )
    gen = ApiCallGenerator(
        {}, llm_client=None, knowledge_pack=pack, available_sources=list(by_name)
    )
    spec = {
        "pass": 2,
        "source": trail.name,
        "sources": [trail.name, reference.name],
        "harvest": [
            {"entity": "reader_unit", "source": "grant", "fields": ["decoded.reader"]}
        ],
        "capture": "",
        "window": "inherit",
        "skip_when_empty": True,
        "purposes": {
            trail.name: "What the unit DID.",
            reference.name: "What KIND of unit it is.",
        },
    }
    logs = {"grant": [{"decoded": {"reader": "UUU1V21QR"}}]}
    queries, notes = await gen.generate_follow_up(
        SimpleNamespace(incident_id="i1", analysis=_analysis()), spec, logs
    )
    assert [q.target_log_source for q in queries] == [trail.name, reference.name]
    text = {q.target_log_source: q.natural_language_query for q in queries}
    assert "What the unit DID." in text[trail.name]
    assert "What KIND of unit it is." not in text[trail.name]
    assert "What KIND of unit it is." in text[reference.name]
    assert "What the unit DID." not in text[reference.name]
    # The one harvest reaches both, which is the whole point of sharing the pass.
    for query in queries:
        assert [(e.type, e.value) for e in query.entities] == [
            ("reader_unit", "UUU1V21QR")
        ]
    assert notes == []


@pytest.mark.asyncio
async def test_an_unavailable_follow_up_target_does_not_cancel_its_SIBLING():
    """Every per-target refusal is a note; only an empty harvest drops the pass whole.

    A target with no retriever, or one that can filter on none of the harvested types, is a
    fact about THAT source. Returned from the whole pass it would delete a sibling's only
    retrieval on the strength of a name that may simply be stale — and that sibling's
    conditions would then read `unknown` with nothing in the run saying which source failed.
    """
    trail = SimpleNamespace(
        name="access_trail_lake",
        kind=lambda: "databricks_uc",
        entities=["reader_unit", "time_window"],
        entity_bindings={"reader_unit": ["reader_col"]},
        not_answered_by=[],
        description="Access trail.",
    )
    # Present in the pack, absent from `available_sources`: the shape a source whose backend
    # credentials are unset really has — declared, planned, and no retriever built.
    absent = SimpleNamespace(
        name="offline_reference",
        kind=lambda: "elasticsearch",
        entities=["reader_unit"],
        entity_bindings={"reader_unit": ["unit_code"]},
        not_answered_by=[],
        description="Offline reference.",
    )
    # Available, but binds none of what the harvest carries: the authoring error, which must
    # also stay local to its own target.
    unbindable = SimpleNamespace(
        name="unrelated_lake",
        kind=lambda: "databricks_uc",
        entities=["document"],
        entity_bindings={"document": ["doc_col"]},
        not_answered_by=[],
        description="Unrelated.",
    )
    by_name = {s.name: s for s in (trail, absent, unbindable)}
    pack = SimpleNamespace(
        sources=list(by_name.values()),
        rulesets={},
        source=by_name.get,
        follow_up_passes=lambda key="": [],
        classify_value_form=lambda etype, value: "",
        value_forms_for=lambda etype: [],
    )
    gen = ApiCallGenerator(
        {},
        llm_client=None,
        knowledge_pack=pack,
        available_sources=[trail.name, unbindable.name],
    )
    spec = {
        "pass": 2,
        "source": absent.name,  # the FIRST target is the broken one
        "sources": [absent.name, unbindable.name, trail.name],
        "harvest": [
            {"entity": "reader_unit", "source": "grant", "fields": ["decoded.reader"]}
        ],
        "capture": "",
        "window": "inherit",
        "skip_when_empty": True,
        "purpose": "Was the unit seen elsewhere.",
    }
    logs = {"grant": [{"decoded": {"reader": "UUU1V21QR"}}]}
    queries, notes = await gen.generate_follow_up(
        SimpleNamespace(incident_id="i1", analysis=_analysis()), spec, logs
    )
    assert [q.target_log_source for q in queries] == [trail.name]
    # Both refusals are REPORTED, and each names its own reason: an operator reading "the
    # question is still open" needs to know whether to fix credentials or fix the pack.
    assert any(absent.name in n and "could not be retrieved" in n for n in notes), notes
    assert any(
        unbindable.name in n and "cannot be used as filters" in n for n in notes
    ), notes


# ── the auto-added query must ask for what the source actually holds ─────────


def test_the_added_query_names_the_sources_own_entities_not_a_fixed_list():
    """The fallback request used to hard-code "org_unit / user / record / document as applicable".

    Two faults in one sentence. It is a domain literal in `src/` (the engine naming this
    domain's entity types), and it is a request no reference table can satisfy: on job
    047a603c it asked `automation_registry` — three columns, `orgUnitId, sign, profile` — for
    the incident's record and document. The list now comes from the source's own catalog entry,
    which is already the allow-list `map_entities` filters against, so a type named here is
    one that can really become a filter.
    """
    src = SimpleNamespace(
        name="robots",
        entities=["org_unit", "user", "automation_flag", "time_window"],
        not_answered_by=[],
        description="Reference list of automated accounts.",
    )
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=None)
    text = gen._fallback_request("robots", src, "Reference list of automated accounts.")

    assert "org_unit, user, automation_flag" in text
    # `time_window` is the date scope, carried in date_from/date_to — not an entity to
    # "filter on where present".
    assert "time_window" not in text
    # The entity types this source does NOT declare must not be asked for.
    assert "asset" not in text
    assert "document" not in text
    # The source's own purpose still rides along.
    assert "Reference list of automated accounts." in text


def test_the_request_can_be_narrowed_to_the_types_the_question_is_about():
    """A relational follow-up must not invite the acting identity into its WHERE clause.

    The source's full entity list is right for a first pass ("filter on whatever this
    incident carries") and wrong for a question about two OTHER parties: the actor who
    granted an access is not the one exercising it, so that predicate returns nothing and
    the source reads as empty. Narrowing is opt-in, so every first-pass caller is unaffected.
    """
    src = SimpleNamespace(
        name="access_trail",
        entities=["org_unit", "user", "reader_unit", "owner_unit", "time_window"],
        not_answered_by=[],
        description="Access trail.",
    )
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=None)
    text = gen._fallback_request(
        "access_trail", src, "p", only_types=["reader_unit", "owner_unit"]
    )
    assert "reader_unit, owner_unit" in text
    assert "org_unit" not in text
    assert "user" not in text
    # Omitted keeps every declared type — the first-pass behaviour.
    whole = gen._fallback_request("access_trail", src, "p")
    assert "org_unit, user, reader_unit, owner_unit" in whole


def test_the_added_query_warns_off_a_declared_non_answer():
    src = SimpleNamespace(
        name="audit_trail",
        entities=["document"],
        not_answered_by=[
            {"question": "whether a document was voided", "ask_instead": "lake"}
        ],
        description="Access audit trail.",
    )
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=None)
    assert "DOES NOT ANSWER" in gen._fallback_request(
        "audit_trail", src, "Access audit trail."
    )
    # A source with nothing declared gets no such clause.
    bare = SimpleNamespace(
        name="s", entities=["org_unit"], not_answered_by=[], description="d"
    )
    assert "DOES NOT ANSWER" not in gen._fallback_request("s", bare, "d")


def test_a_source_declaring_no_entities_still_gets_a_complete_request():
    """Every pack key must no-op when absent — including this one."""
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=None)
    text = gen._fallback_request("s", SimpleNamespace(), "The purpose.")
    assert text.startswith("Retrieve this source's records for the incident.")
    assert "The purpose." in text
    assert "Filter on" not in text


# ── the incident's entity list is authoritative, not a fallback ───────────────


def test_a_partial_llm_entity_list_does_not_displace_the_analysis_entities():
    """`_enrich_queries` is a UNION. It used to be a fallback, and that was the whole of
    one incident's run-to-run verdict variance.

    Measured over three runs of one incident: the enrichment ran only `if not
    query.entities`, so whatever the generator emitted — however partial — replaced the
    deterministic list outright. One run's query carried the login but not the operator
    code; another carried neither the code nor the time window. Three differently scoped
    pass-1 queries, three different verdicts, on identical input.

    The two `user` values here are two distinct FORMS, which is why the dedup key carries
    `value_form`: `render_filters` routes each form to its own column, so collapsing them
    to one `user` would silently drop a binding rather than deduplicate a repeat.
    """
    pack = _pack(("auth", "databricks_uc"))
    pack.sources[0].entities = ["user", "org_unit", "time_window"]
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=pack)

    analysis = _analysis(
        [
            ExtractedEntity(type="user", value="ALOGIN", value_form="login"),
            ExtractedEntity(type="user", value="00X1YZ", value_form="code"),
            ExtractedEntity(type="org_unit", value="UNIT001"),
        ]
    )
    # What the LLM emitted: one of the two forms, nothing else.
    query = RetrievalQuery(
        target_log_source="auth",
        natural_language_query="q",
        date_from="2025-01-01",
        date_to="2025-01-02",
        entities=[ExtractedEntity(type="user", value="ALOGIN", value_form="login")],
    )

    gen._enrich_queries([query], analysis)

    values = {(e.type, e.value) for e in query.entities}
    assert ("user", "ALOGIN") in values
    # The form the generator dropped is back. Pre-fix this is the failing assertion.
    assert ("user", "00X1YZ") in values
    assert ("org_unit", "UNIT001") in values
    # And the value it did emit is carried once, not twice.
    assert len(query.entities) == 3


def test_the_union_keeps_a_value_only_the_generator_found():
    """A model may still CONTRIBUTE; it may no longer REMOVE.

    The incident text is not the only place a value can come from — a generator reading
    the alert prose may pick up an identifier the extractor missed. So the union keeps it,
    subject to the same allow-list the incident's own entities pass through.
    """
    pack = _pack(("auth", "databricks_uc"))
    pack.sources[0].entities = ["user", "org_unit", "time_window"]
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=pack)

    analysis = _analysis([ExtractedEntity(type="user", value="ALOGIN")])
    query = RetrievalQuery(
        target_log_source="auth",
        natural_language_query="q",
        date_from="2025-01-01",
        date_to="2025-01-02",
        entities=[
            ExtractedEntity(type="user", value="BLOGIN"),
            # Not filterable here — dropped by the source's own allow-list, exactly as an
            # incident entity naming an absent column is.
            ExtractedEntity(type="alert_id", value="IR1"),
        ],
    )

    gen._enrich_queries([query], analysis)

    values = {(e.type, e.value) for e in query.entities}
    assert values == {("user", "ALOGIN"), ("user", "BLOGIN")}


def test_the_unfiltered_entity_set_rides_beside_the_scoped_one():
    """The filtered list is right for three readers and wrong for the fourth.

    `entities` is narrowed to what this source can BIND, which the gate reviewer, the generation
    prompt and the health scorer all need. The fabricated-predicate guard needs the opposite: its
    question is whether a literal of a type the source binds NOTHING for was written onto a
    column it binds to another type, so the narrowing deletes the only evidence of the
    contradiction. Hence `_incident_entities`, and hence the assertion that it survives on a
    query whose own allow-list dropped the value entirely.
    """
    pack = _pack(("auth", "databricks_uc"))
    pack.sources[0].entities = ["user", "time_window"]
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=pack)

    analysis = _analysis(
        [
            ExtractedEntity(type="user", value="ALOGIN"),
            ExtractedEntity(type="record", value="8J6ONY"),
        ]
    )
    query = RetrievalQuery(
        target_log_source="auth",
        natural_language_query="q",
        date_from="2025-01-01",
        date_to="2025-01-02",
    )

    gen._enrich_queries([query], analysis)

    # The scope is unchanged: the locator is not filterable here and stays out.
    assert {(e.type, e.value) for e in query.entities} == {("user", "ALOGIN")}
    # And the guard's second input carries it anyway.
    assert {(e.type, e.value) for e in query._incident_entities} == {
        ("user", "ALOGIN"),
        ("record", "8J6ONY"),
    }
    from src.retrievers.field_mapping import incident_values

    assert incident_values(query)["record"] == ["8J6ONY"]


@pytest.mark.asyncio
async def test_a_follow_up_query_carries_the_unfiltered_set_too():
    """The follow-up path builds its `RetrievalQuery` itself, and it is where this now BITES.

    A follow-up target is by definition a source deferred out of pass 1, so the source the
    measured fabrication was written against (job ca4240c0: a record locator on an office
    column) is asked on pass 2 — a path that never passes through `_enrich_queries`. The prose
    this path writes does tell the generator to filter on nothing else, and that is a prompt;
    relying on it is what this guard exists to not do.
    """
    src = SimpleNamespace(
        name="office_profile",
        kind=lambda: "databricks_uc",
        entities=["org_unit", "time_window"],
        entity_bindings={"org_unit": ["OFFICE_NAME"]},
        not_answered_by=[],
        description="Office reference data.",
    )
    pack = SimpleNamespace(
        sources=[src],
        rulesets={},
        source=lambda n: src if n == src.name else None,
        follow_up_passes=lambda key="": [],
        classify_value_form=lambda etype, value: "",
        value_forms_for=lambda etype: [],
    )
    gen = ApiCallGenerator(
        {}, llm_client=None, knowledge_pack=pack, available_sources=[src.name]
    )
    spec = {
        "pass": 2,
        "sources": [src.name],
        "harvest": [{"entity": "org_unit", "source": "grant", "fields": ["office"]}],
        "purpose": "What type of office each of these is.",
    }
    analysis = _analysis(
        [
            ExtractedEntity(type="record", value="8J6ONY", raw="8J6ONY"),
            ExtractedEntity(type="org_unit", value="JJJ1K10UV", raw="JJJ1K10UV"),
        ]
    )
    queries, _ = await gen.generate_follow_up(
        SimpleNamespace(incident_id="i1", analysis=analysis),
        spec,
        {"grant": [{"office": "PPP1Q16EF"}]},
    )
    assert len(queries) == 1
    # The harvested value is the scope, exactly as before.
    assert {(e.type, e.value) for e in queries[0].entities} == {
        ("org_unit", "PPP1Q16EF")
    }
    # And the incident's own locator — of a type this source binds nothing for — is visible to
    # the guard, which is the whole point.
    from src.retrievers.field_mapping import incident_values

    values = incident_values(queries[0])
    assert values["record"] == ["8J6ONY"]
    assert sorted(values["org_unit"]) == ["JJJ1K10UV", "PPP1Q16EF"]


# --- the two ways a dependency reaches the wrong pass -------------------------
# Both of these were measured on real runs of the shipped pack, and both fail the same
# invisible way: the source is never queried, the stage reports success, and the conditions
# that read it go `unknown` — which the report renders as INSUFFICIENT DATA.


def test_a_subject_discovery_rulesets_sources_ride_on_no_other_run():
    """The heuristic that had to keep such a ruleset is why the whole superset read is gone.

    A ruleset declaring `subject_discovery` says, in the pack, that its subject is NOT in the
    incident text: it is inside the rows, knowable only after retrieval. So "did the incident
    name this ruleset's subject?" — the question the removed scoping asked in order to prune
    the union of every ruleset's dependencies — cannot be answered for it, and the only safe
    answer was to keep it. That permission is what made two of the shipped pack's sources a
    dependency of every incident it would ever process, whatever the incident said.

    Reading the ONE adjudicating ruleset needs no such permission: a discovery-based procedure
    contributes its own dependencies when it adjudicates, and none when it does not.
    """
    pack = _pack(
        ("record_lake", "databricks_uc"),
        ("pricing_alerts", "databricks_uc"),
        ("automation_registry", "databricks_uc"),
        verdicts={
            "scheme": {
                "subject_entity": "record",
                "sources": {"record": "record_lake"},
            },
            "pricing": {
                # Declared, and deliberately absent from the incident below.
                "subject_entity": "operator",
                "subject_discovery": {
                    "source": "pricing_alerts",
                    "path": "alert.entries",
                    "subject": ["actor.sign"],
                },
                "sources": {
                    "alert": "pricing_alerts",
                    "automation": "automation_registry",
                },
            },
        },
    )
    pack.correlation_specs = lambda: [
        {"use_case": "scheme", "title": "Refund scheme on a record", "keys": ["record"]},
        {"use_case": "pricing", "title": "Abnormal fare pricing", "keys": ["operator"]},
    ]
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=pack)

    # An incident the scheme procedure adjudicates. The discovery-based procedure's two
    # sources are NOT its dependencies — this is the exact leak the user's objection named.
    scheme_incident = _analysis(
        entities=[ExtractedEntity(type="record", value="ABC123", raw="ABC123")],
        summary="a refund scheme was run against record ABC123",
    )
    required, undeliverable = gen._required_sources(
        gen._source_names(), scheme_incident
    )
    assert required == ["record_lake"]
    assert undeliverable == []

    # And when it DOES adjudicate, its own sources are dependencies — with the incident
    # naming none of its subject, which is the whole point of `subject_discovery`. Nothing
    # about the discovery declaration is read here: adjudicating is the only qualification.
    pricing_incident = _analysis(
        entities=[ExtractedEntity(type="record", value="ABC123", raw="ABC123")],
        summary="abnormal fare pricing was detected",
    )
    required2, _ = gen._required_sources(gen._source_names(), pricing_incident)
    assert set(required2) == {"pricing_alerts", "automation_registry"}
    assert "record_lake" not in required2


def _two_ruleset_follow_up_pack(shared=True):
    """Two procedures; only the SECOND declares a follow-up pass over `auth_events`.

    `shared=True` makes the first procedure declare `auth_events` too, which is the case that
    decides whether deferral may be read across rulesets at all.
    """
    scheme = {"subject_entity": "record", "sources": {"record": "record_lake"}}
    if shared:
        scheme["sources"]["auth"] = "auth_events"
    pack = _pack(
        ("record_lake", "databricks_uc"),
        ("admin_action_history", "databricks_uc"),
        ("auth_events", "databricks_uc"),
        verdicts={
            "scheme": scheme,
            "admin": {
                "subject_entity": "operator",
                "sources": {"admin": "admin_action_history", "auth": "auth_events"},
            },
        },
    )
    # The playbook frontmatter the selector scores. Two distinct prose titles, because
    # WHICH procedure adjudicates is what scopes the deferral, and a fake that cannot answer
    # differently for two incidents would make the assertions below vacuous.
    pack.correlation_specs = lambda: [
        {"use_case": "scheme", "title": "Refund scheme on a record", "keys": ["record"]},
        {"use_case": "admin", "title": "Operator privilege misuse", "keys": ["operator"]},
    ]
    pack.follow_up_passes = lambda key="": (
        [
            {
                "pass": 2,
                "source": "auth_events",
                "harvest": [
                    {"entity": "user", "source": "admin_action_history", "fields": ["x"]}
                ],
                "capture": "",
                "window": "inherit",
                "skip_when_empty": True,
                "purpose": "",
            }
        ]
        if key == "admin"
        else []
    )
    return pack


def test_a_follow_up_declared_by_another_procedure_does_not_defer_this_run():
    """Deferral and collection are two halves of ONE mechanism, so they need one scope.

    `follow_up_passes` is COLLECTED for the ruleset that adjudicates. Read for deferral
    across every *applicable* ruleset instead, a source another procedure defers is taken out
    of pass 1 here and asked on pass 2 by nobody — so it is queried on no pass at all, which
    is indistinguishable from the pack never declaring it.

    Measured on the shipped pack: one source was being deferred out of every run of the
    procedure that needs it on pass 1, by a sibling procedure's declaration, before this
    ruleset existed to notice.
    """
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_two_ruleset_follow_up_pack())
    record_incident = _analysis(
        entities=[ExtractedEntity(type="record", value="ABC123", raw="ABC123")],
        summary="a refund scheme was run against record ABC123",
    )
    # `scheme` adjudicates and declares no pass, so its own dependency stands.
    assert gen._follow_up_targets(record_incident) == set()
    required, _ = gen._required_sources(gen._source_names(), record_incident)
    assert "auth_events" in required

    # And for the procedure that DOES declare the pass, the same source is deferred — the
    # scoping is a change of scope, not an opt-out.
    admin_incident = _analysis(
        entities=[ExtractedEntity(type="operator", value="OP1", raw="OP1")],
        summary="operator OP1 committed privilege misuse",
    )
    assert gen._follow_up_targets(admin_incident) == {"auth_events"}
    required2, _ = gen._required_sources(gen._source_names(), admin_incident)
    assert "auth_events" not in required2
    assert "admin_action_history" in required2


def test_a_source_only_its_deferring_procedure_declares_is_not_reported_on_other_runs():
    """Where the ONLY procedure declaring a source defers it, no other run reports it missing.

    Either that procedure adjudicates, and its own pass 2 asks — or it does not, and none of
    its conditions is evaluated, so naming the source on this run's gap list points at
    evidence nothing here would read. It falls out of reading the adjudicating ruleset alone,
    which is why this is a pin and not a mechanism: measured, two sibling procedures' pass-2
    target was force-added to every run of a third, against the primary time budget, for zero
    rows read by zero conditions.
    """
    gen = ApiCallGenerator(
        {}, llm_client=None, knowledge_pack=_two_ruleset_follow_up_pack(shared=False)
    )
    record_incident = _analysis(
        entities=[ExtractedEntity(type="record", value="ABC123", raw="ABC123")],
        summary="a refund scheme was run against record ABC123",
    )
    required, undeliverable = gen._required_sources(gen._source_names(), record_incident)
    assert required == ["record_lake"]
    # Not reclassified as a missing dependency: it is deliverable and simply not asked here.
    assert undeliverable == []


# --- the operator's own edits to the plan -------------------------------------
# When the planner declines a declared source, an operator can add it. Which entities
# a source can filter on is pack knowledge, so the server builds the query, not the client.


def _scopable_pack():
    """Three sources, two of which declare what they can be filtered on.

    `member_list` declares only a type the incidents below never name, which is the case
    `_scopable` exists for; `open_log` declares nothing, which must stay permissive.
    """
    pack = _pack(
        ("record_lake", "databricks_uc"),
        ("member_list", "databricks_uc"),
        ("open_log", "databricks_uc"),
        verdicts={
            "scheme": {
                "sources": {
                    "record": "record_lake",
                    "members": "member_list",
                }
            }
        },
    )
    by_name = {s.name: s for s in pack.sources}
    by_name["record_lake"].entities = ["record", "time_window"]
    by_name["record_lake"].description = "every version of a record\nsecond line"
    by_name["member_list"].entities = ["membership_id", "time_window"]
    by_name["member_list"].description = "the membership reference list"
    return pack


def test_scopable_separates_a_failed_dependency_from_a_declined_one():
    """A source bound to no type the incident named could only be a date-window scan.

    Which makes it a dependency the run correctly DECLINED, not one it failed to meet: the
    condition that wanted it resolves its values out of `analysis.extracted_entities`, so a
    value the incident never carried is unavailable to it however many rows arrive. Reporting
    the two the same way puts the item the operator cannot act on in front of the one they
    can — and on a `subject_discovery` procedure it would fire on every single run.
    """
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_scopable_pack())
    entities = [ExtractedEntity(type="record", value="ABC123", raw="ABC123")]

    assert gen._scopable("record_lake", entities) is True
    assert gen._scopable("member_list", entities) is False
    # A source declaring nothing keeps every entity, so it counts as scopable: a missing
    # declaration must not excuse a real gap.
    assert gen._scopable("open_log", entities) is True
    # No entities at all — nothing to scope by, whatever the source declares.
    assert gen._scopable("record_lake", []) is False
    # time_window alone does not count, or this would be true of every source on every
    # incident and the question would be vacuous.
    window_only = [ExtractedEntity(type="time_window", value="2026-08-01", raw="")]
    assert gen._scopable("record_lake", window_only) is False

    # And the report splits on exactly that: both are unmet, only one is actionable.
    report = gen.dependency_report([], gen._source_names(), _analysis(entities=entities))
    assert report["not_queried"] == ["record_lake"]
    assert report["unscopable"] == ["member_list"]


def test_a_manual_query_is_scoped_by_the_same_code_as_a_planned_one():
    """The reason the panel names a source and not a query.

    A `RetrievalQuery` carries a typed `entities` list, and which of the incident's entities
    a source can bind is pack knowledge. So the operator supplies the two things only a human
    has — which source, and what to ask it — and the scope comes from `_enrich_queries`, the
    one place the planner's own queries are scoped.
    """
    from src.models.pydantic_models import EventWindow

    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_scopable_pack())
    analysis = _analysis(
        entities=[
            ExtractedEntity(type="record", value="ABC123", raw="ABC123"),
            # Not a column on record_lake — dropped, exactly as it is for a planned query.
            ExtractedEntity(type="membership_id", value="M1", raw="M1"),
        ],
        event_time=EventWindow(start="2026-08-02T10:00:00", end="2026-08-03T10:00:00"),
    )
    query = gen.build_manual_query(analysis, "record_lake", "who reissued it")
    assert query.target_log_source == "record_lake"
    assert query.natural_language_query == "who reissued it"
    assert [(e.type, e.value) for e in query.entities] == [("record", "ABC123")]
    # The incident's own window wins over anything borrowed from the plan.
    assert (query.date_from, query.date_to) == ("2026-08-02", "2026-08-03")

    # No event_time: the window borrowed from the queries already in the plan is used, so an
    # added query is bounded the same way its neighbours are rather than unbounded.
    windowless = _analysis(
        entities=[ExtractedEntity(type="record", value="ABC123", raw="ABC123")]
    )
    borrowed = gen.build_manual_query(
        windowless, "record_lake", "", window=("2026-07-01", "2026-07-08")
    )
    assert (borrowed.date_from, borrowed.date_to) == ("2026-07-01", "2026-07-08")
    # A blank question falls back to the request built from the source's OWN declarations —
    # what the planner would have been asked to write, never a fixed engine sentence.
    assert "record" in borrowed.natural_language_query
    assert "every version of a record" in borrowed.natural_language_query


def test_a_manual_query_against_an_unretrievable_source_is_refused_by_reason():
    """Two different failures, and the operator can act on only one of them.

    A name the pack does not know is a typo; a name it knows that built no retriever is an
    environment gap. Accepting either produces a query that fails at retrieval time, after
    the operator has been told the plan was edited — the shape of the silent degrade this
    whole seam exists to make visible.
    """
    gen = ApiCallGenerator(
        {},
        llm_client=None,
        knowledge_pack=_scopable_pack(),
        available_sources=["record_lake", "open_log"],
    )
    with pytest.raises(ValueError, match="not in the catalog"):
        gen.build_manual_query(_analysis(), "no_such_source")
    with pytest.raises(ValueError, match="built no retriever"):
        gen.build_manual_query(_analysis(), "member_list")
    with pytest.raises(ValueError, match="required"):
        gen.build_manual_query(_analysis(), "  ")


def test_the_menu_puts_the_procedures_own_unmet_dependencies_first():
    """The panel's list, and it must be stateless.

    One generator instance serves every job on the server, so reading the last run's
    attributes would annotate this job's menu with another incident's dependencies. Every
    flag is recomputed from the analysis it was handed.
    """
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_scopable_pack())
    analysis = _analysis(
        entities=[ExtractedEntity(type="record", value="ABC123", raw="ABC123")]
    )
    chosen = [
        RetrievalQuery(
            target_log_source="record_lake",
            natural_language_query="q",
            date_from="2026-08-01",
            date_to="2026-08-02",
        )
    ]
    menu = gen.unselected_sources(chosen, analysis)
    assert [e["source"] for e in menu] == ["member_list", "open_log"]
    declared = {e["source"]: e["declared"] for e in menu}
    assert declared == {"member_list": True, "open_log": False}
    assert {e["source"]: e["scopable"] for e in menu} == {
        "member_list": False,
        "open_log": True,
    }
    # The purpose is the first line only: the menu is one line per source.
    assert menu[0]["purpose"] == "the membership reference list"
    # A source with a query is not offered twice.
    assert "record_lake" not in [e["source"] for e in menu]

    # Statelessness, asserted the way it can actually fail: run one incident, then ask for
    # another's menu on the SAME generator and expect no trace of the first.
    gen._report_unmet_dependencies(chosen, gen._source_names(), analysis)
    assert gen.declared_unscopable == ["member_list"]
    other = _analysis(
        entities=[ExtractedEntity(type="membership_id", value="M1", raw="M1")]
    )
    again = gen.unselected_sources([], other)
    by_name = {e["source"]: e for e in again}
    assert by_name["member_list"]["scopable"] is True
    assert by_name["record_lake"]["scopable"] is False


def test_a_deferred_target_is_offered_but_flagged():
    """Listed rather than hidden — the operator may have a reason — and flagged, because a
    pass-1 query MERGES its rows into the pass-2 result under one source name, and a
    condition counting distinct rows then reads both scopes at once."""
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_follow_up_pack())
    menu = gen.unselected_sources([], _analysis())
    entry = next(e for e in menu if e["source"] == "access_trail_lake")
    assert entry["deferred"] is True
    # Still a declared dependency of the procedure — `sources:` is where a pass names its
    # target — so the menu says both things rather than either one alone.
    assert entry["declared"] is True
    # And it sorts after the dependencies this pass really is expected to have.
    order = [e["source"] for e in menu]
    assert order.index("auth_events") < order.index("access_trail_lake")


def _scoping_pack():
    """Two sources that DECLARE what they can filter on, which is what makes this testable.

    `_pack`'s sources declare no `entities` at all, and a source declaring none keeps every
    entity by construction (`_entities_for`) — so a fake built from it can never be
    unscopable, and every assertion below would pass vacuously against the unfixed engine.
    """
    pack = _pack(
        ("auth_events", "databricks_uc"),
        ("record_lake", "databricks_uc"),
    )
    pack.sources[0].entities = ["user", "office", "time_window"]
    pack.sources[1].entities = ["record", "time_window"]
    pack.catalog_prompt = lambda names=None: "catalog"
    return pack


def test_a_selected_source_the_incident_cannot_scope_is_dropped():
    """Applied to the incident's entities, the same scopability rule as for harvested values.

    Both halves are asserted: the query is gone AND the drop is named. A silent removal is
    the degrade shape this seam exists to prevent.
    """
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_scoping_pack())
    analysis = _analysis([ExtractedEntity(type="record", value="ABC123", raw="ABC123")])
    queries = [
        RetrievalQuery(
            target_log_source=name,
            natural_language_query="q",
            date_from="2026-07-16",
            date_to="2026-07-18",
        )
        for name in ("auth_events", "record_lake")
    ]

    kept = gen._drop_unscopable(queries, analysis)

    # `auth_events` filters on user/office and the incident named neither.
    assert [q.target_log_source for q in kept] == ["record_lake"]
    assert gen.selected_unscopable == ["auth_events"]


def test_an_unscopable_source_is_kept_when_the_incident_named_nothing_at_all():
    """The bound that stops this from deleting the whole plan.

    With no entities `_scopable` is False for every source, so the same predicate that removes
    one bad pick would remove all of them. An incident that named no subject is already FATAL
    at `understanding` (`no_entities`), so the decision belongs to the operator reading that —
    not to this seam quietly returning an empty plan.
    """
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_scoping_pack())
    queries = [
        RetrievalQuery(
            target_log_source="auth_events",
            natural_language_query="q",
            date_from="2026-07-16",
            date_to="2026-07-18",
        )
    ]

    kept = gen._drop_unscopable(queries, _analysis(entities=[]))

    assert [q.target_log_source for q in kept] == ["auth_events"]
    assert gen.selected_unscopable == []


def test_a_source_that_declares_nothing_filterable_is_never_dropped():
    """Permissive where the pack is silent, for the same reason the reporting side is.

    A source declaring no `entities` keeps every entity (`_entities_for`), so it is scopable by
    construction. Reading an absent declaration as "unscopable" would make this seam delete
    queries in proportion to how little a pack documents — punishing the authoring gap by
    removing the evidence.
    """
    pack = _pack(("undocumented", "databricks_uc"))
    pack.catalog_prompt = lambda names=None: "catalog"
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=pack)
    queries = [
        RetrievalQuery(
            target_log_source="undocumented",
            natural_language_query="q",
            date_from="2026-07-16",
            date_to="2026-07-18",
        )
    ]

    kept = gen._drop_unscopable(
        queries, _analysis([ExtractedEntity(type="record", value="ABC123", raw="ABC123")])
    )

    assert [q.target_log_source for q in kept] == ["undocumented"]
    assert gen.selected_unscopable == []


@pytest.mark.asyncio
async def test_a_dropped_dependency_is_reported_as_unscopable_not_missing():
    """The two seams share one predicate, so the removal and the classification agree.

    A dropped pick becomes an unselected source, and if the ruleset DECLARES it the dependency
    report has to say something about it. `unscopable` is the honest bucket — "declining it was
    correct" — and `not_queried` would be wrong twice: it accuses the planner of a mistake the
    engine just made on its behalf, and it carries a 0.5 weight that gates the stage alone. The
    ordering in `generate` is what makes this hold, so it is asserted end to end.
    """
    from unittest.mock import AsyncMock, MagicMock

    calls = [
        SimpleNamespace(
            function=SimpleNamespace(
                name="create_retrieval_query",
                arguments=RetrievalQuery(
                    target_log_source=name,
                    natural_language_query="q",
                    date_from="2026-07-16",
                    date_to="2026-07-18",
                ).model_dump_json(),
            )
        )
        for name in ("record_lake", "auth_events")
    ]
    llm = MagicMock()
    llm.tool_call = AsyncMock(return_value=SimpleNamespace(tool_calls=calls))

    pack = _ruleset_pack()
    pack.catalog_prompt = lambda names=None: "catalog"
    # `auth_events` is one of this ruleset's five declared dependencies, and it filters on an
    # identity this incident does not carry.
    by_name = {s.name: s for s in pack.sources}
    by_name["auth_events"].entities = ["user", "time_window"]
    by_name["record_lake"].entities = ["record", "time_window"]
    gen = ApiCallGenerator({}, llm_client=llm, knowledge_pack=pack)
    understanding = SimpleNamespace(incident_id="i1", analysis=_analysis())

    queries = await gen.generate(understanding)

    assert [q.target_log_source for q in queries] == ["record_lake"]
    assert gen.selected_unscopable == ["auth_events"]
    # Reported as correctly declined, and NOT as a gap the planner should have filled.
    assert "auth_events" in gen.declared_unscopable
    assert "auth_events" not in gen.declared_not_queried


def test_an_operators_own_query_for_an_unscopable_source_is_still_built():
    """Asking anyway is the decision the plan editor exists to let a human make.

    `build_manual_query` does not route through `generate`, so the drop cannot reach it. That
    is the design and not an oversight: the engine declines on what the catalog says, and an
    operator who knows this source holds the answer must be able to overrule it — which is
    also the remedy the drop's own log line names.
    """
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_scoping_pack())
    analysis = _analysis([ExtractedEntity(type="record", value="ABC123", raw="ABC123")])

    query = gen.build_manual_query(analysis, "auth_events", "look anyway")

    assert query is not None
    assert query.target_log_source == "auth_events"


@pytest.mark.asyncio
async def test_a_harvest_scoped_follow_up_survives_the_drop():
    """The collision the drop could cause, asserted so it stays impossible.

    `_drop_unscopable` reads the incident's entities; a follow-up pass is scoped by harvested
    values on `RetrievalQuery.entities`, which never reach `analysis.extracted_entities`. A
    target the incident cannot scope is exactly a valid pass-2 query. Pass >= 2 goes through
    `generate_follow_up`, not `generate`; routing it through `generate` silently disables them.
    """
    target = SimpleNamespace(
        name="reader_trail",
        kind=lambda: "databricks_uc",
        # NOT `record` — nothing the incident below carries can scope this source.
        entities=["reader_unit", "time_window"],
        entity_bindings={"reader_unit": ["reader_col"]},
        not_answered_by=[],
        description="Reader trail.",
    )
    pack = SimpleNamespace(
        sources=[target],
        rulesets={},
        source=lambda n: target if n == target.name else None,
        follow_up_passes=lambda key="": [],
        classify_value_form=lambda etype, value: "",
        value_forms_for=lambda etype: [],
    )
    gen = ApiCallGenerator(
        {}, llm_client=None, knowledge_pack=pack, available_sources=[target.name]
    )
    analysis = _analysis(
        [ExtractedEntity(type="record", value="ABC123", raw="ABC123")]
    )
    # The incident cannot scope this source: the predicate the drop reads is False for it.
    assert gen._scopable(target.name, list(analysis.extracted_entities)) is False

    queries, _ = await gen.generate_follow_up(
        SimpleNamespace(incident_id="i1", analysis=analysis),
        {
            "pass": 2,
            "source": target.name,
            "harvest": [
                {"entity": "reader_unit", "source": "grant", "fields": ["reader"]}
            ],
            "window": "onwards",
            "skip_when_empty": True,
            "purpose": "Which units read it.",
        },
        {"grant": [{"reader": "UNIT01"}]},
    )

    assert len(queries) == 1
    assert queries[0].target_log_source == target.name
    assert {(e.type, e.value) for e in queries[0].entities or []} == {
        ("reader_unit", "UNIT01")
    }
    # And the drop stayed out of it entirely, so nothing was reported as a catalog defect.
    assert gen.selected_unscopable == []


def _window_gen():
    """A generator with no pack behaviour — `_follow_up_window` reads only the spec and the
    analysis, so a pack would be scenery."""
    return ApiCallGenerator({}, llm_client=None, knowledge_pack=_scopable_pack())


def _prior(date_from="2026-08-02", date_to="2026-08-03"):
    return [
        RetrievalQuery(
            target_log_source="record_lake",
            natural_language_query="pass 1",
            date_from=date_from,
            date_to=date_to,
        )
    ]


def test_a_lookback_window_moves_the_LOWER_bound_and_leaves_the_upper():
    """The third window mode, and the only one that can ask an ANTECEDENT question.

    `inherit` and `onwards` both anchor `date_from` to the event start and move only
    `date_to`, so neither can reach a time BEFORE the incident. Measured on a cross-detector
    history question: the pass inherited a single-day window and matched 0 rows, against 1 at
    ±30d, 3 at ±180d and 4 unbounded — every one of them strictly before the incident, so
    `onwards` would also have returned nothing. A 0-row answer to a question about an
    identity's history is not evidence the history is clean.
    """
    from src.models.pydantic_models import EventWindow

    gen = _window_gen()
    analysis = _analysis(
        event_time=EventWindow(start="2026-08-02T10:00:00", end="2026-08-03T10:00:00")
    )
    lo, hi = gen._follow_up_window({"pass": 2, "window": "lookback:30d"}, analysis, _prior())
    assert lo == "2026-07-03"
    # The upper bound is the EPISODE's, untouched: widening forward as well would ask two
    # questions at once and spend the scan on the half nobody declared.
    assert hi == "2026-08-03"

    # `inherit` on the same inputs is the window the lookback moved away from — which is what
    # makes the assertion above a delta rather than a coincidence of the fixture.
    assert gen._follow_up_window({"pass": 2}, analysis, _prior()) == (
        "2026-08-02",
        "2026-08-03",
    )


def test_an_unusable_lookback_depth_narrows_to_inherit_rather_than_widening():
    """Every failure resolves to the window the caller already had, and never to a wider one.

    A depth that does not parse, a `date_from` that is not a date and no lower bound at all
    are three different problems with one safe answer. The conservative direction is the
    NARROW scan: an unbounded lower bound is the full-retention read `follow_up_passes` exists
    to avoid, so a lookback that silently became one would trade a wrong answer for a slow one
    on a system of record — where a timeout decides the verdict instead of degrading it.
    """
    from src.models.pydantic_models import EventWindow

    gen = _window_gen()
    analysis = _analysis(
        event_time=EventWindow(start="2026-08-02T10:00:00", end="2026-08-03T10:00:00")
    )
    for mode in ("lookback", "lookback:", "lookback:d", "lookback:many", "lookbackward"):
        lo, hi = gen._follow_up_window({"pass": 2, "window": mode}, analysis, _prior())
        assert (lo, hi) == ("2026-08-02", "2026-08-03"), mode

    # No lower bound to move: left unbounded as it already was, NOT re-anchored to the run
    # date, which would silently make an antecedent question a recent one.
    lo, hi = gen._follow_up_window(
        {"pass": 2, "window": "lookback:30d"}, _analysis(), _prior(date_from="")
    )
    assert lo == ""

    # And the spellings that DO parse, so the strictness above is not just a rejecting regex:
    # the depth is what matters, the `d` and the separator are conveniences.
    for mode in ("lookback:365d", "lookback=365d", "lookback:365", "LOOKBACK:365D"):
        lo, _ = gen._follow_up_window({"pass": 2, "window": mode}, analysis, _prior())
        assert lo == "2025-08-02", mode


def test_a_referral_window_is_the_SAME_arithmetic_a_follow_up_pass_gets():
    """One implementation of `lookback:<N>d`, reached by both callers.

    A referral and a follow-up pass ask the same depth question, so one implementation answers
    both. A second would drift: both produce a plausible window, but only one matches the pack.

    A depthless `lookback` degrades to the inherited window and reports `inherit`, so a caller
    claiming "lookback" over a single-day window is corrected.
    """
    from src.models.pydantic_models import EventWindow

    gen = _window_gen()
    analysis = _analysis(
        event_time=EventWindow(start="2026-08-02T10:00:00", end="2026-08-03T10:00:00")
    )
    assert gen.referral_window(analysis, _prior(), "lookback:30d") == (
        "2026-07-03",
        "2026-08-03",
        "lookback:30d",
    )
    # A depthless hint inherits, and says `inherit` — not the word it was handed.
    assert gen.referral_window(analysis, _prior(), "lookback") == (
        "2026-08-02",
        "2026-08-03",
        "inherit",
    )
    # Absent entirely is inherit too: a link with no direction is still a link.
    assert gen.referral_window(analysis, _prior(), "")[2] == "inherit"
    # And a consequent looks FORWARD, which is the whole reason the direction rides along.
    lo, hi, mode = gen.referral_window(analysis, _prior(), "onwards")
    assert (lo, mode) == ("2026-08-02", "onwards")
    assert hi >= "2026-08-03"


def _repeat_pack():
    """One source that is both the harvest's origin and the follow-up's target.

    That is the realistic shape of a REPEAT: pass 1 retrieved a source, the pass-2 declaration
    harvests values out of the rows it returned, and the target is the same source again. A
    different target could not be a repeat at all — nothing was asked of it yet.
    """
    src = SimpleNamespace(
        name="grant_lake",
        kind=lambda: "databricks_uc",
        entities=["reader_unit", "owner_unit", "time_window"],
        entity_bindings={"reader_unit": ["reader_col"], "owner_unit": ["owner_col"]},
        not_answered_by=[],
        description="Access grants.",
    )
    return SimpleNamespace(
        sources=[src],
        rulesets={},
        source=lambda n: src if n == src.name else None,
        follow_up_passes=lambda key="": [],
        classify_value_form=lambda etype, value: "",
        value_forms_for=lambda etype: [],
    )


def _repeat_case(prior_owners=("JJJ1K10UV", "JJJ1K11UV")):
    """The inputs of a follow-up pass whose values a pass-1 query already carried.

    Returns `(gen, spec, logs, analysis, prior, understanding)`. The four tests below each
    change exactly ONE of the five facts `repeats_an_earlier_query` requires, so the fixture
    has to be a provable repeat as it stands — otherwise a test asserting "the pass runs"
    would pass for the wrong reason.
    """
    from src.models.pydantic_models import EventWindow

    pack = _repeat_pack()
    gen = ApiCallGenerator(
        {}, llm_client=None, knowledge_pack=pack, available_sources=["grant_lake"]
    )
    spec = {
        "pass": 2,
        "source": "grant_lake",
        "harvest": [
            {"entity": "reader_unit", "source": "grant_lake", "fields": ["reader"]},
            {"entity": "owner_unit", "source": "grant_lake", "fields": ["owner"]},
        ],
        "purpose": "Did the readers read the owner's records.",
    }
    logs = {
        "grant_lake": [
            {"reader": "UUU1V21QR", "owner": "JJJ1K10UV"},
            {"reader": "UUU1V21QR", "owner": "JJJ1K11UV"},
        ]
    }
    # The incident's entity is neither harvested type deliberately: seeding `reader_unit`
    # would drop it from scope, and a one-type scope is refused, making assertions vacuous.
    analysis = _analysis(
        entities=[ExtractedEntity(type="record", value="ABC123", raw="ABC123")],
        event_time=EventWindow(start="2026-08-02T10:00:00", end="2026-08-03T10:00:00"),
    )
    prior = [
        RetrievalQuery(
            target_log_source="grant_lake",
            natural_language_query="pass 1",
            date_from="2026-08-01",
            date_to="2026-08-05",
            entities=[ExtractedEntity(type="reader_unit", value="UUU1V21QR", raw="x")]
            + [
                ExtractedEntity(type="owner_unit", value=v, raw=v)
                for v in prior_owners
            ],
        )
    ]
    return (
        gen,
        spec,
        logs,
        analysis,
        prior,
        SimpleNamespace(incident_id="i1", analysis=analysis),
    )


@pytest.mark.asyncio
async def test_a_follow_up_pass_that_asks_only_what_pass_1_asked_is_NOT_run():
    """A harvest carrying nothing new buys nothing, and the scan is the cost.

    Every value this pass would filter on rode on the pass-1 query to the same source, over a
    window containing this one, and that query came back well short of its row cap — so its
    rows are a superset of what the second scan would return, and they are already in `logs`.
    On a `retrieval_class: primary` source that scan is a budget measured in hours, and it is
    the same budget every OTHER condition on the source is competing for.
    """
    gen, spec, logs, _analysis_, prior, understanding = _repeat_case()

    queries, notes = await gen.generate_follow_up(
        understanding, spec, logs, prior_queries=prior, row_caps={"grant_lake": 500}
    )

    assert queries == []
    # AND IT SAYS SO. A pass that vanishes reads exactly like a source with nothing to say,
    # which is the failure mode this whole seam exists to keep out of a report: the note names
    # the source, the scope, the window and the row count that proves the earlier answer whole.
    assert len(notes) == 1
    assert "grant_lake" in notes[0]
    assert "no value the earlier retrieval" in notes[0]
    assert "2 row(s)" in notes[0] and "500-row cap" in notes[0]


@pytest.mark.asyncio
async def test_a_TRUNCATED_earlier_answer_keeps_the_follow_up_pass():
    """The one case where re-asking the same values narrower returns rows the first answer lost.

    A result cut off at `max_results` is not an answer about the values it carried — it is an
    answer about the first N rows that matched. So a second, narrower query over a subset of
    those values reaches rows the wider one dropped, and the repeat reasoning inverts. A cap
    the caller cannot state reads the same way (the second half), because "not truncated" is
    a claim and an absent cap is not evidence for it.
    """
    gen, spec, logs, _analysis_, prior, understanding = _repeat_case()

    # Two rows against a two-row cap: full page, so possibly cut off.
    queries, notes = await gen.generate_follow_up(
        understanding, spec, logs, prior_queries=prior, row_caps={"grant_lake": 2}
    )
    assert len(queries) == 1
    assert notes == []

    # And with no cap known at all — which is every caller written before `row_caps` existed,
    # so the pre-existing behaviour is preserved exactly rather than approximately.
    queries, notes = await gen.generate_follow_up(
        understanding, spec, logs, prior_queries=prior
    )
    assert len(queries) == 1
    assert notes == []


@pytest.mark.asyncio
async def test_a_WIDENED_lookback_over_the_same_values_is_not_a_repeat():
    """Same source, same values, different question.

    `lookback:` exists precisely to ask an ANTECEDENT question the first pass could not: the
    values are the incident's, and the point is the history before it. Reading that as a repeat
    would skip the only pass whose window was declared on purpose — and it would do so most
    confidently on the runs where pass 1 returned a small, clean result.
    """
    gen, spec, logs, _analysis_, prior, understanding = _repeat_case()
    spec = dict(spec, window="lookback:30d")

    queries, notes = await gen.generate_follow_up(
        understanding, spec, logs, prior_queries=prior, row_caps={"grant_lake": 500}
    )

    assert len(queries) == 1
    # 30 days before the episode start, which is outside the pass-1 window's 2026-08-01 floor.
    assert queries[0].date_from == "2026-07-03"
    assert notes == []


@pytest.mark.asyncio
async def test_ONE_new_harvested_value_keeps_the_whole_pass():
    """The skip is proven per source, not per value, so one unasked value runs the scan.

    There is no such thing as a query for the remainder: the pass asks its whole scope at
    once, and a value pass 1 never carried is a row that may not be in `logs`. The same holds
    for a whole entity TYPE — an earlier query naming fewer types AND-ed fewer predicates and
    is therefore not the wider read its superset values suggest.
    """
    gen, spec, logs, _analysis_, prior, understanding = _repeat_case(
        prior_owners=("JJJ1K10UV",)  # JJJ1K11UV was harvested but never asked
    )

    queries, notes = await gen.generate_follow_up(
        understanding, spec, logs, prior_queries=prior, row_caps={"grant_lake": 500}
    )
    assert len(queries) == 1
    assert notes == []

    # A missing TYPE, with every value still a superset: the pass runs for the same reason.
    _, _, _, _, prior_one_type, _ = _repeat_case()
    prior_one_type[0].entities = [
        e for e in prior_one_type[0].entities if e.type != "owner_unit"
    ]
    queries, notes = await gen.generate_follow_up(
        understanding, spec, logs, prior_queries=prior_one_type,
        row_caps={"grant_lake": 500},
    )
    assert len(queries) == 1
    assert notes == []


@pytest.mark.asyncio
async def test_analyst_guidance_beats_the_repeat_reasoning():
    """A reviewer who rejected the gate asked a different question, whatever the values are.

    `guidance` is appended to the request text, so the query the retriever writes is not the
    one pass 1 wrote — and a human who looked at this run and said "ask again" is the last
    input the engine should overrule with an inference about scope.
    """
    gen, spec, logs, _analysis_, prior, understanding = _repeat_case()

    queries, notes = await gen.generate_follow_up(
        understanding,
        spec,
        logs,
        prior_queries=prior,
        guidance="include the parent office too",
        row_caps={"grant_lake": 500},
    )
    assert len(queries) == 1
    assert "include the parent office too" in queries[0].natural_language_query
    assert notes == []


# ── an undated incident: the window is declared, never invented ──────────────
# Without a cleared event_time and a declared default depth, the generator picks a different
# window per source. Both are needed: the clear alone leaves the per-source guess.


def _depth_pack(days=30, sources=(("record_lake", "databricks_uc"),)):
    pack = _pack(*sources)
    pack.default_lookup_days = lambda: days
    return pack


def _undated(timestamp="2026-08-17T09:30:00Z"):
    return SimpleNamespace(
        incident_id="INC-1",
        incident_timestamp=timestamp,
        analysis=SimpleNamespace(extracted_entities=[], event_time=None),
    )


def test_an_undated_incident_reads_the_DECLARED_depth_and_the_engine_holds_none():
    """Resolved pack → config, and nothing when neither declares.

    How far back a store must be read to find the conduct behind an undated report is a fact
    about the domain's retention, its alerting lag and how long the behaviour typically runs —
    so a number in `src/` would be that judgement made in the one place that cannot know it,
    applied to every domain. An absent declaration therefore leaves the generated window
    standing (with a warning naming both places), because the alternative reading of "nobody
    declared one" is a silent engine constant.
    """
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_depth_pack(30))
    assert gen._default_window("2026-08-17T09:30:00Z") == ("2026-07-18", "2026-08-17")
    # The config is the fallback, and only the fallback: a pack that declares one wins.
    cfg = ApiCallGenerator({"default_lookup_days": 7}, llm_client=None, knowledge_pack=None)
    assert cfg._default_window("2026-08-17T09:30:00Z") == ("2026-08-10", "2026-08-17")
    both = ApiCallGenerator(
        {"default_lookup_days": 7}, llm_client=None, knowledge_pack=_depth_pack(30)
    )
    assert both._default_window("2026-08-17T09:30:00Z") == ("2026-07-18", "2026-08-17")
    # Nothing declared anywhere: no window, so every query keeps the one generated for it.
    assert ApiCallGenerator({}, llm_client=None)._default_window("2026-08-17") == ("", "")
    # A depth that cannot be a depth is the same answer — never a fabricated fallback.
    for bad in (0, -7, "thirty", None):
        assert ApiCallGenerator(
            {}, llm_client=None, knowledge_pack=_depth_pack(bad)
        )._default_window("2026-08-17T09:30:00Z") == ("", ""), bad
    # A pack whose accessor raises degrades to the config rather than taking retrieval down.
    broken = _depth_pack(30)
    broken.default_lookup_days = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    assert ApiCallGenerator(
        {"default_lookup_days": 7}, llm_client=None, knowledge_pack=broken
    )._default_window("2026-08-17T09:30:00Z") == ("2026-08-10", "2026-08-17")


def test_the_default_window_is_anchored_on_the_INCIDENT_and_not_on_the_run():
    """Re-running the same incident a week later must read the same window.

    The ingestion timestamp is the one time an undated incident does state. Anchoring on
    `now()` instead makes the scope of the investigation a property of when somebody pressed
    the button — so two runs of one incident disagree with nothing changed, which is the same
    failure mode as the invented window this replaces.
    """
    from datetime import datetime, timedelta, timezone

    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_depth_pack(30))
    assert gen._default_window("2026-08-17T09:30:00Z") == ("2026-07-18", "2026-08-17")
    assert gen._default_window("2026-08-17") == ("2026-07-18", "2026-08-17")
    # No timestamp or unparseable: anchor on today in UTC, asserted relatively.
    today = datetime.now(timezone.utc).date()
    for absent in ("", None, "not a timestamp"):
        start, end = gen._default_window(absent)
        assert end == today.isoformat(), absent
        assert start == (today - timedelta(days=30)).isoformat(), absent


def test_the_declared_window_is_written_onto_EVERY_query_and_resolved_ONCE():
    """One window per plan, not one per query — the per-source guess is the defect.

    And it is authoritative for the same reason a stated `event_time` is: the window is the
    hard scope of the query, so leaving it to the generator makes the scope of the
    investigation a per-source coincidence.
    """
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=_depth_pack(30))
    queries = [
        RetrievalQuery(
            target_log_source="record_lake",
            natural_language_query="q1",
            date_from="2026-08-16",
            date_to="2026-08-17",
        ),
        RetrievalQuery(
            target_log_source="record_lake",
            natural_language_query="q2",
            date_from="2026-05-01",
            date_to="2026-08-17",
        ),
    ]
    und = _undated()
    gen._enrich_queries(queries, und.analysis, und.incident_timestamp)
    assert [(q.date_from, q.date_to) for q in queries] == [
        ("2026-07-18", "2026-08-17"),
        ("2026-07-18", "2026-08-17"),
    ]
    # A STATED window still wins over the declared depth: the default is a fallback below any
    # window the incident really carries, and above the generator's guess.
    stated = SimpleNamespace(
        extracted_entities=[],
        event_time=SimpleNamespace(start="2026-03-01T00:00:00Z", end="2026-03-02T00:00:00Z"),
    )
    dated = [
        RetrievalQuery(
            target_log_source="record_lake",
            natural_language_query="q",
            date_from="2026-08-16",
            date_to="2026-08-17",
        )
    ]
    gen._enrich_queries(dated, stated, "2026-08-17T09:30:00Z")
    assert (dated[0].date_from, dated[0].date_to) == ("2026-03-01", "2026-03-02")
    # And with nothing declared, the generated windows are left exactly as they were — the
    # engine adds no depth of its own, so a pack that opts out is byte-identical to before.
    plain = ApiCallGenerator({}, llm_client=None, knowledge_pack=_pack(("record_lake", "databricks_uc")))
    untouched = [
        RetrievalQuery(
            target_log_source="record_lake",
            natural_language_query="q",
            date_from="2026-08-16",
            date_to="2026-08-17",
        )
    ]
    plain._enrich_queries(untouched, und.analysis, und.incident_timestamp)
    assert (untouched[0].date_from, untouched[0].date_to) == ("2026-08-16", "2026-08-17")


def test_the_incidents_own_combinations_ride_on_EVERY_query_in_the_plan():
    """Incident-side half of the tuple guard; projection unifies it with the harvested shape.

    Both produce `_value_tuples` on `RetrievalQuery`. Two claims: groups arrive in the
    incident's own order (what the label padding is for), and they ride on every query before
    per-source narrowing (field_mapping decides projectability later, per source).
    """
    pack = _pack(("access_a", "databricks_uc"), ("access_b", "elasticsearch"))
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=pack)

    # Two ungrouped constants of different types, each a conjunct and not a record member.
    # One alone could not form a combination and cannot distinguish "skipped" from "not combinable".
    entities = [
        ExtractedEntity(type="record", value="ABC123"),
        ExtractedEntity(type="organization", value="ORGX"),
    ]
    # The SECOND record's values first: the model's entity list is in no particular order, and the
    # only thing carrying the incident's layout is the label — which is why it is padded and why
    # the projection sorts on it rather than trusting the order it was handed.
    for n, label in ((2, "L000004"), (1, "L000003")):
        entities += [
            ExtractedEntity(type="org_unit", value=f"UNIT{n:02d}", co_occurrence=label),
            ExtractedEntity(
                type="user",
                value=f"CODE{n:02d}",
                value_form="code",
                co_occurrence=label,
            ),
            ExtractedEntity(
                type="user",
                value=f"LOGIN{n:02d}",
                value_form="login",
                co_occurrence=label,
            ),
        ]
    analysis = _analysis(entities)

    queries = [
        RetrievalQuery(
            target_log_source=name,
            natural_language_query="q",
            date_from="2025-01-01",
            date_to="2025-01-02",
        )
        for name in ("access_a", "access_b")
    ]
    gen._enrich_queries(queries, analysis)

    for query in queries:
        assert query._value_tuples == [
            [
                {"type": "org_unit", "value": "UNIT01", "value_form": ""},
                {"type": "user", "value": "CODE01", "value_form": "code"},
                {"type": "user", "value": "LOGIN01", "value_form": "login"},
            ],
            [
                {"type": "org_unit", "value": "UNIT02", "value_form": ""},
                {"type": "user", "value": "CODE02", "value_form": "code"},
                {"type": "user", "value": "LOGIN02", "value_form": "login"},
            ],
        ], query.target_log_source
    # Neither ungrouped constant is a member of any record, and the two of them are not a record
    # of their own — they arrived in the prose around the table, not in a row of it.
    named = {m["value"] for t in queries[0]._value_tuples for m in t}
    assert "ABC123" not in named and "ORGX" not in named


def test_a_group_of_ONE_TYPE_is_not_a_combination_and_most_incidents_produce_NONE():
    """Both silences, and each is the byte-identical-to-before path.

    A group that collapsed to a single type constrains one column, which the per-type filter hints
    already do — calling it a combination claims a narrowing that is not one, and the guard would
    then rewrite an OR of that type's own values into an OR of one-member ANDs: the same rows, in a
    shape an operator reads as a set of records. And an incident that tabulates nothing produces no
    groups at all, which is most of them.
    """
    tuples = ApiCallGenerator._incident_value_tuples(
        [
            ExtractedEntity(
                type="user", value="CODE01", value_form="code", co_occurrence="L000003"
            ),
            ExtractedEntity(
                type="user",
                value="LOGIN01",
                value_form="login",
                co_occurrence="L000003",
            ),
            ExtractedEntity(
                type="user", value="CODE02", value_form="code", co_occurrence="L000004"
            ),
            ExtractedEntity(
                type="user",
                value="LOGIN02",
                value_form="login",
                co_occurrence="L000004",
            ),
        ]
    )
    assert tuples == []

    assert (
        ApiCallGenerator._incident_value_tuples(
            [
                ExtractedEntity(type="org_unit", value="UNIT01"),
                ExtractedEntity(type="user", value="CODE01", value_form="code"),
            ]
        )
        == []
    )

    # A group that keeps two types beside a repeated one is still a combination: the arity is not
    # fixed anywhere, and nothing here knows the word "pair".
    kept = ApiCallGenerator._incident_value_tuples(
        [
            ExtractedEntity(type="org_unit", value="UNIT01", co_occurrence="L000003"),
            ExtractedEntity(type="user", value="CODE01", co_occurrence="L000003"),
            ExtractedEntity(type="organization", value="ORGX", co_occurrence="L000003"),
        ]
    )
    assert [m["type"] for m in kept[0]] == ["org_unit", "user", "organization"]


def test_a_value_carrying_SEVERAL_labels_is_a_member_of_EVERY_record_it_names():
    """Reader half of the multi-label stamp; the two halves are a fix only together.

    One entity per distinct value: an org unit servicing several subjects carries a
    space-joined set of labels. Read as one label it becomes a one-type group, discarded;
    every record that shares it loses it; `enforce_value_tuples` then finds literals in no
    combination and refuses the rewrite, so the query goes out as the cross product.
    The assertion: every grouped value is a member of some combination.
    """
    entities = [
        ExtractedEntity(
            type="org_unit", value="UNIT01", co_occurrence="L000002 L000003"
        ),
        ExtractedEntity(
            type="user", value="CODE01", value_form="code", co_occurrence="L000002"
        ),
        ExtractedEntity(
            type="user", value="LOGIN01", value_form="login", co_occurrence="L000002"
        ),
        ExtractedEntity(
            type="user", value="CODE02", value_form="code", co_occurrence="L000003"
        ),
        ExtractedEntity(
            type="user", value="LOGIN02", value_form="login", co_occurrence="L000003"
        ),
    ]
    tuples = ApiCallGenerator._incident_value_tuples(entities)

    assert [[m["value"] for m in t] for t in tuples] == [
        ["UNIT01", "CODE01", "LOGIN01"],
        ["UNIT01", "CODE02", "LOGIN02"],
    ]
    # Nothing the incident grouped is unharvested — the count the guard proves before it narrows.
    assert {m["value"] for t in tuples for m in t} == {e.value for e in entities}
    # And the label SET is not itself a group: there is no third combination holding the unit alone.
    assert len(tuples) == 2


# ── a pick the call could not carry ───────────────────────────────────────────
# A call that failed validation was dropped with one warning, making it indistinguishable
# from a source nobody chose. Engine-owned fields are completed; the planner's question
# is reported, not invented.


def _call(arguments):
    """One tool call carrying whatever the planner wrote — valid JSON or not."""
    return SimpleNamespace(
        function=SimpleNamespace(name="create_retrieval_query", arguments=arguments)
    )


def test_a_call_missing_only_the_window_is_completed_not_dropped():
    """The dates are the engine's, so a call omitting them is incomplete and not malformed.

    `_enrich_queries` overwrites `date_from`/`date_to` unconditionally — from the incident's
    `event_time` where it states one, else from the declared default depth — so whatever the
    planner wrote there was never going to survive. Dropping the whole call over them loses the
    source; completing them loses nothing, and a query that ends up with no window anyway is
    already scored by `query_without_window`.
    """
    gen = ApiCallGenerator({}, llm_client=None)
    message = SimpleNamespace(
        tool_calls=[
            _call('{"target_log_source": "record_lake", '
                  '"natural_language_query": "who read this record"}')
        ]
    )
    queries = gen._parse_tool_calls(message, ["record_lake"])
    assert [q.target_log_source for q in queries] == ["record_lake"]
    assert queries[0].natural_language_query == "who read this record"
    # Empty rather than invented: the enrichment is what fills them, and a fabricated window
    # would become the hard scope of the retrieval.
    assert (queries[0].date_from, queries[0].date_to) == ("", "")
    assert gen.selected_unparseable == []


def test_a_null_window_is_completed_too_because_a_null_is_not_a_missing_key():
    """The `.get(k, default)` trap, in the one place a model validates the value.

    A planner that writes `"date_from": null` has not omitted the key, so nothing keyed on
    absence fires — and the model rejects `None` for a `str` exactly as it rejects a missing
    field. Both readings must reach the same completion or the salvage covers half the shape.
    """
    gen = ApiCallGenerator({}, llm_client=None)
    message = SimpleNamespace(
        tool_calls=[
            _call('{"target_log_source": "record_lake", "natural_language_query": "q", '
                  '"date_from": null, "date_to": null}')
        ]
    )
    queries = gen._parse_tool_calls(message, ["record_lake"])
    assert [q.target_log_source for q in queries] == ["record_lake"]
    assert (queries[0].date_from, queries[0].date_to) == ("", "")


def test_a_call_with_no_question_is_REPORTED_and_never_invented():
    """The question is the analytic intent, and the engine has nothing to write there.

    THE LIVE DEFECT: a run emitted `{"target_log_source": "<source>"}` with no question and no
    dates, the call was discarded, that source is not one the adjudicating ruleset declares so
    no `required_source_not_queried` fired, and the plan an operator approved simply did not
    contain it. The remedy is one click in the plan editor, where a human supplies the question
    — which is why the finding names the source and the query does not exist.
    """
    gen = ApiCallGenerator({}, llm_client=None)
    message = SimpleNamespace(tool_calls=[_call('{"target_log_source": "record_lake"}')])
    queries = gen._parse_tool_calls(message, ["record_lake"])
    assert queries == []
    assert gen.selected_unparseable == ["record_lake"]


def test_a_blank_question_is_the_same_as_no_question():
    """Whitespace is not an analytic intent, and it would reach the retriever prompt as one."""
    gen = ApiCallGenerator({}, llm_client=None)
    message = SimpleNamespace(
        tool_calls=[
            _call('{"target_log_source": "record_lake", "natural_language_query": "   "}')
        ]
    )
    assert gen._parse_tool_calls(message, ["record_lake"]) == []
    assert gen.selected_unparseable == ["record_lake"]


def test_a_malformed_entity_list_costs_the_entities_and_not_the_source():
    """The only nested shape left, and the incident's own entities are attached anyway.

    `_enrich_queries` unions the incident's entities onto every query, so dropping the
    planner's restatement of them costs the query nothing it will not get back — where dropping
    the call costs the whole source. Asserted because the second validation attempt is the kind
    of fallback that silently stops being reachable.
    """
    gen = ApiCallGenerator({}, llm_client=None)
    message = SimpleNamespace(
        tool_calls=[
            _call('{"target_log_source": "record_lake", "natural_language_query": "q", '
                  '"entities": ["not an entity object"]}')
        ]
    )
    queries = gen._parse_tool_calls(message, ["record_lake"])
    assert [q.target_log_source for q in queries] == ["record_lake"]
    assert list(queries[0].entities or []) == []
    assert gen.selected_unparseable == []


def test_an_unnamed_or_unparseable_call_reports_nothing_because_there_is_nothing_to_report():
    """Two drops that must NOT become findings: a finding an operator cannot act on is noise.

    Arguments that are not an object at all, and an object naming no source, leave nothing to
    add to the plan — there is no source to click. Both still log, and both still drop.
    """
    gen = ApiCallGenerator({}, llm_client=None)
    message = SimpleNamespace(
        tool_calls=[
            _call("not json at all"),
            _call('["a", "list"]'),
            _call('{"natural_language_query": "who read this record"}'),
        ]
    )
    assert gen._parse_tool_calls(message, ["record_lake"]) == []
    assert gen.selected_unparseable == []


def test_a_salvaged_target_this_run_cannot_retrieve_is_not_reported_as_a_gap():
    """A hallucinated source name in a findings list is a click that goes nowhere.

    The validated path already has its own branch for an unknown source (it defaults to the
    first offered one). Here there is no query to redirect, so the only question is whether the
    name reaches the operator — and it must not, because the source does not exist on this run.
    """
    gen = ApiCallGenerator({}, llm_client=None)
    message = SimpleNamespace(tool_calls=[_call('{"target_log_source": "invented_source"}')])
    assert gen._parse_tool_calls(message, ["record_lake"]) == []
    assert gen.selected_unparseable == []


def test_the_unparseable_finding_is_reset_per_parse():
    """One generator instance serves every job on the server.

    The same reason `dependency_report` reads no instance state at all: a list left over from
    the previous incident is a finding reported against this one. Reset where it is filled, so
    a run with no malformed call clears it even though nothing else touches the attribute.
    """
    gen = ApiCallGenerator({}, llm_client=None)
    first = SimpleNamespace(tool_calls=[_call('{"target_log_source": "record_lake"}')])
    gen._parse_tool_calls(first, ["record_lake"])
    assert gen.selected_unparseable == ["record_lake"]

    clean = SimpleNamespace(
        tool_calls=[
            _call(
                RetrievalQuery(
                    target_log_source="record_lake",
                    natural_language_query="q",
                    date_from="2026-07-16",
                    date_to="2026-07-18",
                ).model_dump_json()
            )
        ]
    )
    queries = gen._parse_tool_calls(clean, ["record_lake"])
    assert len(queries) == 1
    assert gen.selected_unparseable == []
