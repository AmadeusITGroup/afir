"""Tests for the correlation/aggregation stage."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.correlation import (
    CorrelationModule,
    aggregate,
    derive_schema,
    discover_join_keys,
    execute_plan,
    flatten_leaves,
    resolve_path,
)
from src.models.pydantic_models import (
    STUB_OBSERVED,
    CorrelationResult,
    EventWindow,
    ExtractedEntity,
    IncidentAnalysis,
    TransformPlan,
    TransformResult,
    TransformStep,
    UnderstandingResult,
)


def _understanding(entities, correlation_keys=None, summary="s"):
    return UnderstandingResult(
        incident_id="INC-1",
        analysis=IncidentAnalysis(
            incident_summary=summary,
            severity="5",
            severity_reasoning="r",
            impact_assessment="i",
            key_investigation_areas=[],
            log_sources_to_review=[],
            initial_hypotheses=[],
            recommended_actions=[],
            stakeholder_notification=[],
            extracted_entities=entities,
            correlation_keys=correlation_keys or [],
        ),
    )


# --- deterministic aggregate (unchanged base) -------------------------------


def test_aggregate_counts_and_cross_source_overlap():
    logs = {
        "application_logs": [
            {"org_unit_id": "ORGUNIT2301", "record": "SUBJ00"},
            {"org_unit_id": "ORGUNIT2301", "record": "OTHER"},
        ],
        "transaction_logs": [
            {"org_unit": "ORGUNIT2301", "document": "300", "record_locator": "SUBJ00"},
        ],
    }
    agg = aggregate(logs, ["ORGUNIT2301", "SUBJ00"])

    assert agg["record_counts"] == {"application_logs": 2, "transaction_logs": 1}
    assert agg["total_records"] == 3
    assert "ORGUNIT2301" in agg["cross_source_overlap"]
    assert "SUBJ00" in agg["cross_source_overlap"]
    assert agg["entity_occurrences"]["ORGUNIT2301"]["application_logs"] == 2


def test_aggregate_ignores_wildcards():
    agg = aggregate({"s": [{"a": "x"}]}, ["*", ""])
    assert agg["entity_occurrences"] == {}


# --- schema derivation ------------------------------------------------------


def test_derive_schema_unions_row_keys():
    logs = {
        "auth_svc": [{"uid": "U1", "asn": "A1"}, {"uid": "U2", "ts": "t"}],
        "empty": [],
    }
    schema = derive_schema(logs)
    assert set(schema["auth_svc"]) == {"uid", "asn", "ts"}
    assert schema["empty"] == []


# --- transform executor (pure Python, no LLM) -------------------------------

_LOGS = {
    "auth": [
        {"uid": "U1", "asn": "A1", "ts": "2024-08-26T10:00:00", "result": "fail"},
        {"uid": "U1", "asn": "A2", "ts": "2024-08-26T10:05:00", "result": "fail"},
        {
            "uid": "U1",
            "asn": "A2",
            "ts": "2024-08-26T10:10:00",
            "result": "ok",
            "sessionId": "S1",
        },
        {"uid": "U2", "asn": "A3", "ts": "2024-08-27T09:00:00", "result": "ok"},
    ],
    "issuance_alerts": [{"sessionId": "S1", "document": "300"}],
}


def test_executor_group_by_and_threshold():
    plan = TransformPlan(
        steps=[
            TransformStep(op="group_by", label="by_uid", source="auth", keys=["uid"]),
            TransformStep(
                op="threshold", label="hot", over="by_uid", operator=">", value=2
            ),
        ]
    )
    results = {r.label: r for r in execute_plan(plan, _LOGS)}
    assert results["by_uid"].rows[0] == {"uid": "U1", "metric": 3}
    # Only U1 (3 events) clears the >2 threshold.
    assert results["hot"].rows == [{"uid": "U1", "metric": 3}]


def test_executor_group_by_distinct():
    plan = TransformPlan(
        steps=[
            TransformStep(
                op="group_by",
                label="asns",
                source="auth",
                keys=["uid"],
                agg="distinct",
                field="asn",
            ),
        ]
    )
    rows = execute_plan(plan, _LOGS)[0].rows
    by_uid = {r["uid"]: r["metric"] for r in rows}
    assert by_uid == {"U1": 2, "U2": 1}  # U1 hit A1+A2


def test_executor_time_bucket_and_cross_source():
    plan = TransformPlan(
        steps=[
            TransformStep(
                op="time_bucket",
                label="hourly",
                source="auth",
                field="ts",
                bucket="1h",
            ),
            TransformStep(
                op="cross_source_overlap",
                label="sess",
                entity="sessionId",
                sources=["auth", "issuance_alerts"],
            ),
        ]
    )
    results = {r.label: r for r in execute_plan(plan, _LOGS)}
    hourly = {r["bucket"]: r["metric"] for r in results["hourly"].rows}
    assert hourly["2024-08-26 10:00"] == 3
    assert results["sess"].rows == [
        {"value": "S1", "sources": ["auth", "issuance_alerts"]}
    ]


def test_a_capped_transform_result_says_it_was_capped():
    """`N rows` and `N rows, and there were more` must be different findings.

    Each executor sliced to `_MAX_RESULT_ROWS` itself and returned bare rows, so a step
    that hit the ceiling was published as `row_count: 200` exactly like a step that
    genuinely found 200 — the row-cap defect one layer in. The slice now happens once, in
    `execute_plan`, next to the note that records it.
    """
    from src.correlation import _MAX_RESULT_ROWS

    n = _MAX_RESULT_ROWS + 37
    logs = {"wide": [{"uid": f"U{i}"} for i in range(n)]}
    plan = TransformPlan(
        steps=[
            TransformStep(op="group_by", label="by_uid", source="wide", keys=["uid"])
        ]
    )
    res = execute_plan(plan, logs)[0]
    assert len(res.rows) == _MAX_RESULT_ROWS  # still bounded
    assert "TRUNCATED" in res.note and str(n) in res.note


def test_a_transform_that_fits_carries_no_note():
    """The note must mean something: exactly-at-the-cap is NOT truncated."""
    from src.correlation import _MAX_RESULT_ROWS

    logs = {"wide": [{"uid": f"U{i}"} for i in range(_MAX_RESULT_ROWS)]}
    plan = TransformPlan(
        steps=[
            TransformStep(op="group_by", label="by_uid", source="wide", keys=["uid"])
        ]
    )
    res = execute_plan(plan, logs)[0]
    assert len(res.rows) == _MAX_RESULT_ROWS
    assert res.note == ""


def test_a_threshold_over_a_capped_step_is_flagged_too():
    """`threshold` filters a PRIOR label, so a truncated group_by shrinks the population
    its outlier test ran over. The cap applies to every op through the one seam."""
    from src.correlation import _MAX_RESULT_ROWS

    # Every uid appears twice, so all groups have metric 2 and all clear `>1`.
    n = _MAX_RESULT_ROWS + 10
    logs = {"wide": [{"uid": f"U{i % n}"} for i in range(2 * n)]}
    plan = TransformPlan(
        steps=[
            TransformStep(op="group_by", label="by_uid", source="wide", keys=["uid"]),
            TransformStep(
                op="threshold", label="hot", over="by_uid", operator=">", value=1
            ),
        ]
    )
    results = {r.label: r for r in execute_plan(plan, logs)}
    assert "TRUNCATED" in results["by_uid"].note
    # `hot` reads the already-capped 200, so it is not itself over the cap — and must
    # therefore NOT claim a truncation of its own.
    assert len(results["hot"].rows) == _MAX_RESULT_ROWS
    assert results["hot"].note == ""


def test_the_deterministic_summary_publishes_the_truncation_note():
    """The high-volume path is where the cap is most likely hit, and a bare `row_count`
    there is exactly the number a reader treats as a total."""
    mod = CorrelationModule(MagicMock(), {})
    tr = [
        TransformResult(
            label="joined",
            op="cross_source_overlap",
            rows=[{"value": "V"}],
            note="TRUNCATED: 900 rows computed, the top 200 kept",
        ),
        TransformResult(label="clean", op="group_by", rows=[{"uid": "U1"}]),
    ]
    text = mod._deterministic_summary({"total_records": 900}, tr, [])
    payload = json.loads(text)
    assert payload["cross_source_joins"][0]["note"].startswith("TRUNCATED")
    by_label = {t["label"]: t for t in payload["transforms"]}
    assert by_label["joined"]["note"].startswith("TRUNCATED")
    assert "note" not in by_label["clean"]  # absent, not empty-string noise


def test_executor_skips_bad_step_without_raising():
    plan = TransformPlan(
        steps=[
            # group_by on a field-less source still runs; unknown op is skipped.
            TransformStep(op="group_by", label="ok", source="auth", keys=["uid"]),
        ]
    )
    # Inject an invalid op post-validation to prove the executor is defensive.
    plan.steps.append(
        TransformStep(op="distinct", label="bad", source="nope", field="x")
    )
    results = execute_plan(plan, _LOGS)
    labels = {r.label for r in results}
    assert "ok" in labels and "bad" in labels  # both recorded, no exception


# --- analyze() orchestration ------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_empty_logs_skips_llm():
    llm = MagicMock()
    llm.structured_output = AsyncMock()
    module = CorrelationModule({}, llm)

    result = await module.analyze({}, _understanding([]))

    assert result.record_count == 0
    assert "nothing to correlate" in result.summary_text
    llm.structured_output.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_plans_executes_and_narrates():
    # Two structured_output calls: first the TransformPlan, then the narration.
    plan = TransformPlan(
        steps=[
            TransformStep(
                op="group_by",
                label="by_org_unit",
                source="transaction_logs",
                keys=["org_unit"],
            )
        ]
    )
    narration = CorrelationResult(
        findings=[], summary_text="llm summary", record_count=0
    )
    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=[plan, narration])
    module = CorrelationModule({}, llm)  # rag=None -> no playbook retrieval
    logs = {"transaction_logs": [{"org_unit": "ORGUNIT2301", "document": "300"}]}

    result = await module.analyze(
        logs, _understanding([ExtractedEntity(type="org_unit", value="ORGUNIT2301")])
    )

    # Deterministic aggregates preserved.
    assert result.record_count == 1
    assert result.aggregations["total_records"] == 1
    # The planned transform was executed in Python.
    assert any(t.label == "by_org_unit" for t in result.transforms)
    assert result.transforms[0].rows == [{"org_unit": "ORGUNIT2301", "metric": 1}]
    # Narration folded in.
    assert result.summary_text == "llm summary"
    assert llm.structured_output.await_count == 2


@pytest.mark.asyncio
async def test_analyze_survives_plan_failure():
    # If planning raises, analyze still returns deterministic aggregates.
    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=RuntimeError("boom"))
    module = CorrelationModule({}, llm)
    logs = {"transaction_logs": [{"org_unit": "ORGUNIT2301"}]}

    result = await module.analyze(
        logs, _understanding([ExtractedEntity(type="org_unit", value="ORGUNIT2301")])
    )

    assert result.record_count == 1
    assert result.transforms == []  # plan failed -> empty, no crash
    assert result.aggregations["total_records"] == 1


# --- used_by hint (pack-aware transform planning) ---------------------------


def test_used_by_hint_collects_playbooks_from_sources_and_entities():
    from src.knowledge.pack import EntityDef, KnowledgePack, SourceDef

    pack = KnowledgePack(
        entities=[EntityDef(type="org_unit", used_by=["PB-APP-001"])],
        sources=[
            SourceDef(name="transaction_logs", used_by=["PB-APP-001", "PB-ATO-002"])
        ],
    )
    module = CorrelationModule({}, MagicMock(), knowledge_pack=pack)
    analysis = _understanding([ExtractedEntity(type="org_unit", value="X")]).analysis
    hint = module._used_by_hint({"transaction_logs": [{}]}, analysis)
    assert "PB-APP-001" in hint
    assert "PB-ATO-002" in hint


def test_used_by_hint_empty_without_pack():
    module = CorrelationModule({}, MagicMock())
    analysis = _understanding([]).analysis
    assert module._used_by_hint({"s": [{}]}, analysis) == ""


# --- path resolution (nested Kibana _source + flat-dotted ES|QL/Databricks) --

# A nested Kibana-shaped row: entities live under actor.*, alert.allDiscounts[].*,
# and a list-of-dicts orders[].locator.
_NESTED_ROW = {
    "actor": {"orgUnitId": "ORGUNIT2301", "userId": "U1"},
    "alert": {
        "allDiscounts": [
            {
                "loginArea": {"org_unit": "ORGUNIT2301"},
                "documentDetails": {"locator": "SUBJ00"},
            },
            {
                "loginArea": {"org_unit": "ORG1A0980"},
                "documentDetails": {"locator": "ABC123"},
            },
        ],
        "raisedTime": "2024-08-26T10:00:00",
    },
    "orders": [{"locator": "SUBJ00"}, {"locator": "ZZZ999"}],
}


def test_resolve_path_nested_scalar():
    assert resolve_path(_NESTED_ROW, "actor.orgUnitId") == ["ORGUNIT2301"]


def test_resolve_path_list_of_dicts_yields_many():
    vals = resolve_path(_NESTED_ROW, "alert.allDiscounts.loginArea.org_unit")
    assert vals == ["ORGUNIT2301", "ORG1A0980"]
    locators = resolve_path(_NESTED_ROW, "orders.locator")
    assert locators == ["SUBJ00", "ZZZ999"]


def test_resolve_path_flat_dotted_key_esql_case():
    # ES|QL / Databricks return flat dicts with dotted-string keys.
    flat = {"actor.orgUnitId": "ORGUNIT2301"}
    assert resolve_path(flat, "actor.orgUnitId") == ["ORGUNIT2301"]


def test_resolve_path_missing_and_nonscalar():
    assert resolve_path(_NESTED_ROW, "actor.nope") == []
    # A path stopping on a dict (or a list OF dicts) yields nothing.
    assert resolve_path(_NESTED_ROW, "actor") == []
    assert resolve_path(_NESTED_ROW, "orders") == []


def test_resolve_path_terminal_array_of_scalars_yields_every_element():
    """A path may legitimately END on a repeated field. Returning [] for those made the
    leaf indistinguishable from ABSENT, so a pack reference to e.g. the document serial
    (`pricing.asset_document.document.numbers`, a 1-element array in the record lake) silently
    resolved to UNKNOWN and the scope sweep produced document ids missing their serial.
    """
    nested = {
        "pricing": {
            "asset_document": [
                {"document": {"provider": "400", "numbers": ["2000000006"]}},
                {"document": {"provider": "200", "numbers": ["2000000004"]}},
            ]
        }
    }
    assert resolve_path(nested, "pricing.asset_document.document.numbers") == [
        "2000000006",
        "2000000004",
    ]
    # Multi-valued leaf, flat-dotted key, underscore alias, and JSON-string array all agree.
    assert resolve_path(
        {"marketing_providers": ["XY", "LX"]}, "marketing_providers"
    ) == [
        "XY",
        "LX",
    ]
    assert resolve_path({"t.numbers": ["1", "2"]}, "t.numbers") == ["1", "2"]
    assert resolve_path({"document_numbers": ["1", "2"]}, "document.numbers") == [
        "1",
        "2",
    ]
    assert resolve_path({"nbs": '["1", "2"]'}, "nbs") == ["1", "2"]


# --- backend-specific row shapes (Databricks / Snowflake / ServiceNow) ------


def test_resolve_path_databricks_json_string_struct():
    # Databricks SQL API returns a STRUCT/ARRAY selected whole as a JSON STRING.
    row = {"metadata": '{"device": "mobile", "ip": "1.2.3.4"}'}
    assert resolve_path(row, "metadata.device") == ["mobile"]
    # JSON-string array of dicts descends + fans out too.
    row2 = {"orders": '[{"locator": "SUBJ00"}, {"locator": "ZZZ999"}]'}
    assert resolve_path(row2, "orders.locator") == ["SUBJ00", "ZZZ999"]


def test_resolve_path_snowflake_variant_dict_and_case():
    # Snowflake connector returns VARIANT/OBJECT as a live Python dict, keys UPPERCASED.
    row = {"METADATA": {"device": "mobile"}, "USER_ID": "U1"}
    # Path written lowercase (pack style) still resolves against uppercase keys.
    assert resolve_path(row, "metadata.device") == ["mobile"]
    assert resolve_path(row, "user_id") == ["U1"]


def test_resolve_path_servicenow_reference_field():
    # ServiceNow with display values returns reference fields as nested dicts.
    row = {
        "sys_id": "abc123",
        "assigned_to": {"display_value": "John Smith", "link": "https://x/u001"},
    }
    assert resolve_path(row, "assigned_to.display_value") == ["John Smith"]


def test_resolve_path_ordinary_string_not_parsed_as_json():
    # A normal string that isn't JSON is returned as-is (not swallowed by the parser).
    assert resolve_path({"note": "hello"}, "note") == ["hello"]
    # A string that only starts like JSON but is invalid stays a scalar leaf.
    assert resolve_path({"x": "{not json"}, "x") == ["{not json"]


def test_flatten_leaves_parses_json_string_struct():
    row = {"metadata": '{"device": "mobile"}'}
    leaves = flatten_leaves(row)
    assert leaves["metadata.device"] == ["mobile"]


def test_flatten_leaves_nested_and_lists():
    leaves = flatten_leaves(_NESTED_ROW)
    assert leaves["actor.orgUnitId"] == ["ORGUNIT2301"]
    # List-of-dicts collect all values under one dotted path.
    assert leaves["alert.allDiscounts.loginArea.org_unit"] == [
        "ORGUNIT2301",
        "ORG1A0980",
    ]
    assert leaves["orders.locator"] == ["SUBJ00", "ZZZ999"]


def test_flatten_leaves_depth_cap():
    deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": "too-deep"}}}}}}}
    leaves = flatten_leaves(deep)
    # Depth-capped at _MAX_NEST_DEPTH=5, so the deepest scalar is not surfaced.
    assert "a.b.c.d.e.f.g" not in leaves


# --- nested-aware aggregate / derive_schema ---------------------------------


def test_aggregate_matches_nested_entity_values():
    logs = {
        "payment_alerts": [_NESTED_ROW],
        "scheme_alerts": [{"org_unit": "ORGUNIT2301", "record": "SUBJ00"}],
    }
    agg = aggregate(logs, ["ORGUNIT2301", "SUBJ00"])
    # OrgUnit nested under actor.orgUnitId is now matched (the regression this fixes).
    assert agg["entity_occurrences"]["ORGUNIT2301"]["payment_alerts"] == 1
    # And it spans both sources -> cross-source overlap.
    assert "ORGUNIT2301" in agg["cross_source_overlap"]
    assert "SUBJ00" in agg["cross_source_overlap"]


def test_derive_schema_emits_dotted_leaves():
    schema = derive_schema({"payment_alerts": [_NESTED_ROW]})
    leaves = set(schema["payment_alerts"])
    assert "actor.orgUnitId" in leaves
    assert "alert.allDiscounts.loginArea.org_unit" in leaves
    assert "orders.locator" in leaves


# --- executor over nested + multi-valued fields -----------------------------


def test_executor_group_by_nested_multivalued():
    logs = {"payment_alerts": [_NESTED_ROW]}
    plan = TransformPlan(
        steps=[
            TransformStep(
                op="group_by",
                label="by_org_unit",
                source="payment_alerts",
                keys=["alert.allDiscounts.loginArea.org_unit"],
            )
        ]
    )
    rows = execute_plan(plan, logs)[0].rows
    by_org_unit = {
        r["alert.allDiscounts.loginArea.org_unit"]: r["metric"] for r in rows
    }
    # Each discount org_unit contributes one increment from the single row.
    assert by_org_unit == {"ORGUNIT2301": 1, "ORG1A0980": 1}


def test_executor_distinct_nested():
    logs = {"payment_alerts": [_NESTED_ROW]}
    plan = TransformPlan(
        steps=[
            TransformStep(
                op="distinct",
                label="locators",
                source="payment_alerts",
                field="orders.locator",
            )
        ]
    )
    rows = execute_plan(plan, logs)[0].rows
    assert rows == [{"field": "orders.locator", "distinct_count": 2}]


def test_executor_time_bucket_nested():
    logs = {"payment_alerts": [_NESTED_ROW]}
    plan = TransformPlan(
        steps=[
            TransformStep(
                op="time_bucket",
                label="hourly",
                source="payment_alerts",
                field="alert.raisedTime",
                bucket="1h",
            )
        ]
    )
    rows = execute_plan(plan, logs)[0].rows
    assert rows == [{"bucket": "2024-08-26 10:00", "metric": 1}]


# --- cross_source_overlap by entity type (different field per source) --------


def test_cross_source_overlap_by_entity_type():
    from src.knowledge.pack import EntityDef, KnowledgePack, SourceDef

    # 'record' lives under orders.locator in payment_alerts but as flat 'record' in scheme_alerts.
    pack = KnowledgePack(
        entities=[EntityDef(type="record", field_aliases=["record", "locator"])],
        sources=[
            SourceDef(
                name="payment_alerts",
                entity_bindings={"record": ["orders.locator"]},
            ),
            SourceDef(name="scheme_alerts", entity_bindings={"record": ["record"]}),
        ],
    )
    module = CorrelationModule({}, MagicMock(), knowledge_pack=pack)
    logs = {
        "payment_alerts": [_NESTED_ROW],
        "scheme_alerts": [{"record": "SUBJ00"}],
    }
    analysis = _understanding([ExtractedEntity(type="record", value="SUBJ00")]).analysis
    schema = derive_schema(logs)
    entity_map = module._normalized_entity_map(analysis, schema)
    # Verified against real schema leaves.
    assert entity_map["payment_alerts"]["record"] == "orders.locator"
    assert entity_map["scheme_alerts"]["record"] == "record"

    plan = TransformPlan(
        steps=[
            TransformStep(
                op="cross_source_overlap",
                label="record_overlap",
                entity="record",
                sources=["payment_alerts", "scheme_alerts"],
            )
        ]
    )
    rows = execute_plan(plan, logs, entity_map)[0].rows
    # SUBJ00 appears in both sources despite different field names.
    overlap = next(r for r in rows if r["value"] == "SUBJ00")
    assert set(overlap["sources"]) == {"scheme_alerts", "payment_alerts"}


def _co_identity_pack():
    """A pack whose subject entity has two co-identifying forms, bound per form.

    Mirrors the shape every real pack uses: one source keys on the long form, another on
    the short one, and the two forms are declared to name one identity.
    """
    from src.knowledge.pack import CoIdentity, EntityDef, KnowledgePack, SourceDef, ValueForm

    return KnowledgePack(
        entities=[
            EntityDef(
                type="operator",
                field_aliases=["operator"],
                value_forms=[
                    ValueForm(name="code", pattern=r"^[0-9]{4}[A-Z]{2}$"),
                    ValueForm(name="login", pattern=r"^[A-Z][A-Z0-9]{1,9}$"),
                ],
                co_identity=CoIdentity(
                    forms=["code", "login"], via=["unit"], prefer="code"
                ),
            ),
            EntityDef(type="unit", field_aliases=["unit"]),
        ],
        sources=[
            # Keys on the LOGIN form, and also carries the code (an alert-shaped feed).
            SourceDef(
                name="access_events",
                entity_bindings={
                    "operator": {"login": ["actor_login"], "code": ["actor_code"]},
                    "unit": ["actor_unit"],
                },
            ),
            # Keys on the CODE form only — no login column exists here at all.
            SourceDef(
                name="record_writes",
                entity_bindings={
                    "operator": {"code": ["writer_code"]},
                    "unit": ["writer_unit"],
                },
            ),
        ],
    )


def test_cross_source_overlap_joins_two_forms_of_one_identity():
    """A code-keyed source and a login-keyed source must join on the SAME actor.

    Live run 1b716b0d: the resolved `user` key bound the login column on the
    authentication feed and the sign column on both record feeds, so `join_user`
    computed 0 rows and every correlation key scored 0.0 overlap — the cross-source
    checks then read `no data` on an actor present in all three sources. The forms are
    declared to co-identify and a retrieved row binds both, so the join must see one
    identity, not two unequal strings.
    """
    pack = _co_identity_pack()
    module = CorrelationModule({}, MagicMock(), knowledge_pack=pack)
    logs = {
        "access_events": [
            {"actor_login": "TOPERATOR", "actor_code": "6001AA", "actor_unit": "UNIT01"}
        ],
        "record_writes": [{"writer_code": "6001AA", "writer_unit": "UNIT01"}],
    }
    analysis = _understanding(
        [
            ExtractedEntity(type="operator", value="TOPERATOR", value_form="login"),
            ExtractedEntity(type="operator", value="6001AA", value_form="code"),
            ExtractedEntity(type="unit", value="UNIT01"),
        ],
        correlation_keys=["operator", "unit"],
    ).analysis
    schema = derive_schema(logs)
    keys = module._resolve_correlation_keys(analysis, schema, logs)
    key = next(k for k in keys if k.entity_hint == "operator")

    # BOTH sources bound, on the form each one actually carries.
    assert set(key.sources) == {"access_events", "record_writes"}
    assert key.sources["record_writes"] == "writer_code"

    plan = module._build_deterministic_plan(keys)
    emap = module._keys_entity_map(keys)
    results = {t.label: t.rows for t in execute_plan(plan, logs, emap)}
    # The join must be non-empty: one actor, two forms, two sources.
    assert results["join_operator"], "the co-identified actor joined no sources"
    joined = results["join_operator"][0]
    assert set(joined["sources"]) == {"access_events", "record_writes"}


def test_cross_source_overlap_joins_a_suffixed_form_of_one_identifier():
    """The same actor written bare in one source and suffixed in another must join.

    Live run 1b716b0d: the authentication feed carried the actor as `6001AA` and the
    record feed as `6001AAGS` — the pack's own form regex declares that trailing
    two-letter qualifier as optional, and the verdict engine already treats such a pair
    as one identity (`_identifiers_match`). The join compared raw strings, so it computed
    0 rows on an actor present in both sources, and every resolved key reported 0.0
    overlap. Absence and unlike spellings must not render identically.
    """
    logs = {
        "access_events": [{"actor": "6001AA"}],
        "record_writes": [{"actor": "6001AAGS"}],
    }
    plan = TransformPlan(
        steps=[
            TransformStep(
                op="cross_source_overlap",
                label="actor_overlap",
                entity="actor",
                sources=["access_events", "record_writes"],
            )
        ]
    )
    rows = execute_plan(plan, logs)[0].rows
    assert rows, "the suffixed form of one identifier joined no sources"
    assert set(rows[0]["sources"]) == {"access_events", "record_writes"}
    # The value REPORTED is a real spelling from the data, never a normalised stand-in:
    # an operator has to be able to search for it.
    assert rows[0]["value"] in {"6001AA", "6001AAGS"}


def test_cross_source_overlap_does_not_join_unrelated_short_identifiers():
    """A prefix join must not fabricate one: two distinct actors stay two.

    The guard on the suffix-tolerant join. `_identifiers_match` demands >=4 shared
    characters and a true prefix, so neither a short coincidence nor two different
    codes may collapse into a spurious cross-source hit.
    """
    logs = {
        "access_events": [{"actor": "6001AA"}, {"actor": "AB1"}],
        "record_writes": [{"actor": "0099ZZ"}, {"actor": "AB2"}],
    }
    plan = TransformPlan(
        steps=[
            TransformStep(
                op="cross_source_overlap",
                label="actor_overlap",
                entity="actor",
                sources=["access_events", "record_writes"],
            )
        ]
    )
    assert execute_plan(plan, logs)[0].rows == []


def test_normalized_entity_map_only_schema_present_bindings():
    from src.knowledge.pack import EntityDef, KnowledgePack, SourceDef

    pack = KnowledgePack(
        entities=[EntityDef(type="org_unit", field_aliases=["orgUnitId", "org_unit"])],
        sources=[SourceDef(name="s1", entity_bindings={"org_unit": ["missing.field"]})],
    )
    module = CorrelationModule({}, MagicMock(), knowledge_pack=pack)
    analysis = _understanding([ExtractedEntity(type="org_unit", value="X")]).analysis
    # schema has none of the candidate fields -> no mapping for that source.
    schema = {"s1": ["unrelated.col"]}
    assert module._normalized_entity_map(analysis, schema) == {}


def test_normalized_entity_map_empty_without_pack():
    module = CorrelationModule({}, MagicMock())
    analysis = _understanding([ExtractedEntity(type="org_unit", value="X")]).analysis
    assert module._normalized_entity_map(analysis, {"s1": ["org_unit"]}) == {}


def test_normalized_entity_map_case_insensitive_snowflake():
    from src.knowledge.pack import EntityDef, KnowledgePack, SourceDef

    # Pack uses lowercase field names; Snowflake schema is uppercased.
    pack = KnowledgePack(
        entities=[EntityDef(type="org_unit", field_aliases=["org_unit_id"])],
        sources=[SourceDef(name="sf", entity_bindings={"org_unit": ["org_unit_id"]})],
    )
    module = CorrelationModule({}, MagicMock(), knowledge_pack=pack)
    analysis = _understanding([ExtractedEntity(type="org_unit", value="X")]).analysis
    schema = {"sf": ["ORG_UNIT_ID", "AMOUNT"]}
    emap = module._normalized_entity_map(analysis, schema)
    # Maps to the REAL uppercase leaf so resolve_path's fast-path hits it.
    assert emap["sf"]["org_unit"] == "ORG_UNIT_ID"


# --- narration compaction (over-budget input still narrates) ----------------


@pytest.mark.asyncio
async def test_narrate_compacts_over_budget_instead_of_skipping():
    from src.models.pydantic_models import TransformResult

    # A transform result far larger than the 15000-char budget.
    big_rows = [{"org_unit": f"O{i}", "metric": i} for i in range(2000)]
    transforms = [TransformResult(label="big", op="group_by", rows=big_rows)]
    narration = CorrelationResult(findings=[], summary_text="narrated", record_count=0)
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=narration)
    module = CorrelationModule({"sample_rows": 5}, llm)
    analysis = _understanding([]).analysis

    result = await module._narrate(analysis, {"total_records": 2000}, transforms, "")

    # Narration ran (not skipped) and the compacted prompt was small enough to send.
    assert result is not None
    assert result.summary_text == "narrated"
    llm.structured_output.assert_awaited_once()
    sent = llm.structured_output.await_args.args[0][1]["content"]
    assert "row_count" in sent  # compacted representation used
    assert len(sent) < 15000


# --- data-driven join-key discovery (domain-agnostic, no pack) --------------

# Two NOVEL sources sharing a user id under different field names AND different
# shapes: source A nested (Kibana), source B flat (ES|QL). Plus decoy fields:
# a low-cardinality `status` and an ISO-timestamp that must NOT be flagged as keys.
_NOVEL_LOGS = {
    "kibana_alerts": [
        {
            "actor": {"userId": f"U{1000 + i}"},
            "alert": {"raisedTime": f"2024-08-26T0{i % 10}:00:00"},
            "status": "open" if i % 2 == 0 else "closed",
        }
        for i in range(30)
    ],
    "txn_flat": [
        {"uid": f"U{1000 + i}", "amount": 100 + i, "status": "ok" if i % 2 else "ko"}
        for i in range(25)
    ],
}


def test_discover_join_keys_finds_shared_key_diff_names_and_shapes():
    keys = discover_join_keys(_NOVEL_LOGS, derive_schema(_NOVEL_LOGS))
    user_keys = [
        k
        for k in keys
        if "kibana_alerts" in k["sources"] and "txn_flat" in k["sources"]
    ]
    assert user_keys, f"expected a cross-source user key, got {keys}"
    k = user_keys[0]
    assert k["sources"]["kibana_alerts"] == "actor.userId"  # nested resolved
    assert k["sources"]["txn_flat"] == "uid"  # flat resolved
    assert k["entity_hint"] == "user"
    assert k["overlap_score"] >= 0.35


def test_discover_join_keys_decoy_status_not_flagged():
    keys = discover_join_keys(_NOVEL_LOGS, derive_schema(_NOVEL_LOGS))
    for k in keys:
        assert "status" not in k["sources"].values()  # low-cardinality word enum


def test_discover_join_keys_timestamp_not_flagged():
    keys = discover_join_keys(_NOVEL_LOGS, derive_schema(_NOVEL_LOGS))
    for k in keys:
        assert "alert.raisedTime" not in k["sources"].values()


def test_discover_join_keys_empty_and_single_source():
    assert discover_join_keys({}, {}) == []
    single = {"only": [{"uid": f"U{i}"} for i in range(10)]}
    assert discover_join_keys(single, derive_schema(single)) == []


def test_discover_join_keys_alpha_only_needs_cardinality_filter():
    # Alpha-only IDs (no digits) are missed by strict, caught by cardinality filter.
    logs = {
        "a": [
            {"code": c} for c in "alpha beta gamma delta epsilon zeta eta theta".split()
        ],
        "b": [
            {"ref": c} for c in "alpha beta gamma delta epsilon zeta eta theta".split()
        ],
    }
    assert discover_join_keys(logs, derive_schema(logs), "strict") == []
    loose = discover_join_keys(logs, derive_schema(logs), "cardinality")
    assert any("a" in k["sources"] and "b" in k["sources"] for k in loose)


# --- the pack LABELS a discovered key; it never finds one -------------------


def test_pack_field_name_hints_label_a_discovered_key():
    """A domain's own identifier token gets the domain's own entity type.

    Discovery is pack-free by construction — it compares value sets to catch a key nobody
    declared — so the pack is consulted for exactly one thing: the NAME to put on what was
    found. Without the pack map the engine falls back to a small set of universal computing
    tokens, and every domain-specific identifier comes back ``unknown``.
    """
    logs = {
        "orders": [{"locator": f"AB{1000 + i}"} for i in range(20)],
        "audit": [{"record_ref": f"AB{1000 + i}"} for i in range(20)],
    }
    schema = derive_schema(logs)

    # No hints: nothing in the engine's universal table matches `locator`/`record`.
    bare = discover_join_keys(logs, schema, "cardinality")
    assert bare, "the key itself must still be discovered without any pack"
    assert bare[0]["entity_hint"] == "unknown"

    # The pack's map names it.
    hinted = discover_join_keys(
        logs, schema, "cardinality", {"locator": "record", "record": "record"}
    )
    assert (
        hinted[0]["sources"] == bare[0]["sources"]
    )  # same key, only the label changed
    assert hinted[0]["entity_hint"] == "record"


def test_pack_hint_wins_over_the_engine_fallback_token():
    """A pack token outranks the engine's generic one for the same field name."""
    from src.correlation import _infer_entity_hint

    assert _infer_entity_hint("agent_user_id") == "user"  # fallback: `user`
    assert _infer_entity_hint("agent_user_id", hints={"agent": "courier"}) == "courier"
    # Scanned per TOKEN, so an earlier domain token beats a later generic one and a later
    # domain token does not beat an earlier generic one.
    assert _infer_entity_hint("user_agent_id", hints={"agent": "courier"}) == "user"


def test_field_name_hints_drops_a_type_the_pack_never_declared():
    """A typo costs a label, never invents an entity nothing downstream can bind."""
    from src.knowledge.pack import EntityDef, KnowledgePack

    pack = KnowledgePack(
        entities=[EntityDef(type="shipment")],
        field_hints={"tracking": "shipment", "parcel": "shipmnet"},
    )
    assert pack.field_name_hints() == {"tracking": "shipment"}


def test_the_correlation_module_passes_the_packs_hints_to_discovery():
    from src.knowledge.pack import EntityDef, KnowledgePack

    pack = KnowledgePack(
        entities=[EntityDef(type="shipment")], field_hints={"locator": "shipment"}
    )
    module = CorrelationModule({}, MagicMock(), knowledge_pack=pack)
    assert module._entity_hints() == {"locator": "shipment"}
    # No pack at all is a working configuration, not a failure.
    assert CorrelationModule({}, MagicMock())._entity_hints() == {}


# --- layered key resolution: playbook -> understanding -> discovery ---------


def _pack_with_spec(spec=None, entities=None, sources=None):
    from src.knowledge.pack import KnowledgePack

    pack = KnowledgePack(entities=entities or [], sources=sources or [])
    if spec is not None:
        pack.playbook_documents = [
            {
                "title": "pb",
                "content": "c",
                "type": "playbook",
                "metadata": {"playbook_id": "PB-1", "correlation": spec},
            }
        ]
    return pack


def test_resolve_keys_playbook_wins():
    from src.knowledge.pack import EntityDef, SourceDef

    spec = {"keys": ["user"], "time_window": "same_day", "title": "pb"}
    pack = _pack_with_spec(
        spec=spec,
        entities=[EntityDef(type="user", field_aliases=["uid", "actor.userId"])],
        sources=[
            SourceDef(name="kibana_alerts", entity_bindings={"user": ["actor.userId"]}),
            SourceDef(name="txn_flat", entity_bindings={"user": ["uid"]}),
        ],
    )
    module = CorrelationModule({}, MagicMock(), knowledge_pack=pack)
    schema = derive_schema(_NOVEL_LOGS)
    analysis = _understanding([]).analysis
    spec_matched = module._playbook_correlation_spec(analysis)
    keys = module._resolve_correlation_keys(analysis, schema, _NOVEL_LOGS, spec_matched)
    user = [k for k in keys if k.entity_hint == "user"][0]
    assert user.origin == "playbook"
    assert user.time_window == "same_day"
    assert user.sources["kibana_alerts"] == "actor.userId"


def test_a_declared_time_field_is_read_without_a_time_window():
    """WHICH INSTANT IS THE EVENT IS NOT A PROPERTY OF WANTING A JOIN WINDOW.

    `time_fields` was populated only `if tw:`, which made the declaration inert for exactly
    the packs that need it most: a procedure whose sources must not be window-gated (a
    month-wide cohort sweep compared against one booking) could not say which column times
    its rows, and the pick fell to measured resolution alone — which on a live source is a
    constant epoch placeholder carrying a real time-of-day, and on another a *scheduled*
    departure instant. The join is unaffected either way: the matcher reads `time_fields`
    only when a window exists, which the second half asserts.
    """
    from src.knowledge.pack import EntityDef, SourceDef

    logs = {
        "a": [
            {
                "uid": "U1",
                "creation_date_time": "2025-09-23T16:42:00.000+0000",
                "placeholder_time": "2000-01-01T16:42:00.000Z",
            }
        ],
        "b": [{"uid": "U1", "creation_date_time": "2025-09-23T16:45:00.000+0000"}],
    }
    spec = {
        "keys": ["user"],
        "title": "pb",
        "time_fields": {"a": "placeholder_time"},
    }  # NO time_window
    pack = _pack_with_spec(
        spec=spec,
        entities=[EntityDef(type="user", field_aliases=["uid"])],
        sources=[
            SourceDef(name="a", entity_bindings={"user": ["uid"]}),
            SourceDef(name="b", entity_bindings={"user": ["uid"]}),
        ],
    )
    module = CorrelationModule({}, MagicMock(), knowledge_pack=pack)
    analysis = _understanding([]).analysis
    keys = module._resolve_correlation_keys(
        analysis,
        derive_schema(logs),
        logs,
        module._playbook_correlation_spec(analysis),
    )
    user = [k for k in keys if k.entity_hint == "user"][0]
    assert user.origin == "playbook"
    assert user.time_window == ""  # still unbounded, which is the point
    assert user.time_fields["a"] == "placeholder_time"
    # The declaration reaches the chronology...
    from src.evidence import _detect_time_field

    assert _detect_time_field("a", list(logs["a"][0].keys()), logs["a"], keys) == (
        "placeholder_time"
    )
    # ...and does NOT gate the join, because there is no window to gate it with.
    plan = module._build_deterministic_plan(keys)
    step = [s for s in plan.steps if s.entity == "user"][0]
    assert step.time_window == ""
    emap = {src: {"user": field} for src, field in user.sources.items()}
    joined = [r for r in execute_plan(plan, logs, emap) if r.label == step.label][0]
    assert [d["value"] for d in joined.rows] == ["U1"]


def test_resolve_keys_falls_to_understanding_when_no_playbook():
    from src.knowledge.pack import EntityDef, SourceDef

    pack = _pack_with_spec(
        entities=[EntityDef(type="user", field_aliases=["uid", "actor.userId"])],
        sources=[
            SourceDef(name="kibana_alerts", entity_bindings={"user": ["actor.userId"]}),
            SourceDef(name="txn_flat", entity_bindings={"user": ["uid"]}),
        ],
    )
    module = CorrelationModule({}, MagicMock(), knowledge_pack=pack)
    schema = derive_schema(_NOVEL_LOGS)
    # correlation_keys from understanding drives the join.
    analysis = _understanding([], correlation_keys=["user"]).analysis
    keys = module._resolve_correlation_keys(analysis, schema, _NOVEL_LOGS, None)
    user = [k for k in keys if k.entity_hint == "user"][0]
    assert user.origin == "understanding"


def test_resolve_keys_falls_to_discovery_when_pack_and_understanding_silent():
    module = CorrelationModule({}, MagicMock())  # no pack
    schema = derive_schema(_NOVEL_LOGS)
    analysis = _understanding([]).analysis  # no correlation_keys, no entities
    keys = module._resolve_correlation_keys(analysis, schema, _NOVEL_LOGS, None)
    assert any(k.origin == "discovered" and k.entity_hint == "user" for k in keys)


def test_resolve_keys_higher_layer_not_overwritten():
    # Playbook covers 'user'; discovery must not add a second 'user' key for same sources.
    from src.knowledge.pack import EntityDef, SourceDef

    spec = {"keys": ["user"], "title": "pb"}
    pack = _pack_with_spec(
        spec=spec,
        entities=[EntityDef(type="user", field_aliases=["uid", "actor.userId"])],
        sources=[
            SourceDef(name="kibana_alerts", entity_bindings={"user": ["actor.userId"]}),
            SourceDef(name="txn_flat", entity_bindings={"user": ["uid"]}),
        ],
    )
    module = CorrelationModule({}, MagicMock(), knowledge_pack=pack)
    schema = derive_schema(_NOVEL_LOGS)
    analysis = _understanding([]).analysis
    spec_matched = module._playbook_correlation_spec(analysis)
    keys = module._resolve_correlation_keys(analysis, schema, _NOVEL_LOGS, spec_matched)
    user_keys = [k for k in keys if k.entity_hint == "user"]
    assert len(user_keys) == 1 and user_keys[0].origin == "playbook"


def test_resolve_keys_binds_underscore_aliased_struct_leaf():
    """A playbook key declared as a dotted struct path must bind to the row's
    underscore-flattened alias.

    Regression: the Databricks SQL-gen prompt aliases struct leaves (``locator.red`` ->
    ``locator_red``) to avoid last-segment collisions. ``real_field`` only tried exact and
    case-insensitive matches, so the declared key was dropped as "not in the schema" and
    the record join was reported "not evaluated" on a run that had 53 real record rows.
    """
    from src.knowledge.pack import EntityDef, SourceDef

    logs = {
        # Databricks-style aliased columns (dots flattened to underscores).
        "record_lake": [
            {"locator_red": "SUBJ01", "creation_date_time": "2026-07-27T16:09:00"}
        ],
        # Settlement side keeps its plain column name.
        "settlement_report": [
            {
                "relatedRecordLocator": "SUBJ01",
                "transactionDateTime": "2026-07-27T18:15:04",
            }
        ],
    }
    spec = {
        "keys": ["record"],
        "title": "pb",
        "fields": {
            "record_lake": {"record": "locator.red"},  # dotted, as the pack declares it
            "settlement_report": {"record": "relatedRecordLocator"},
        },
    }
    pack = _pack_with_spec(
        spec=spec,
        entities=[
            EntityDef(
                type="record", field_aliases=["locator.red", "relatedRecordLocator"]
            )
        ],
        sources=[
            SourceDef(name="record_lake", entity_bindings={"record": ["locator.red"]}),
            SourceDef(
                name="settlement_report",
                entity_bindings={"record": ["relatedRecordLocator"]},
            ),
        ],
    )
    module = CorrelationModule({}, MagicMock(), knowledge_pack=pack)
    schema = derive_schema(logs)
    analysis = _understanding([]).analysis
    keys = module._resolve_correlation_keys(
        analysis, schema, logs, module._playbook_correlation_spec(analysis)
    )
    record_keys = [k for k in keys if k.entity_hint == "record"]
    assert (
        record_keys
    ), "the record key must resolve despite the underscore-aliased column"
    # Bound to the ALIASED column actually present in the rows, both sides present.
    assert record_keys[0].sources["record_lake"] == "locator_red"
    assert record_keys[0].sources["settlement_report"] == "relatedRecordLocator"


# --- the run's own window is a BOUND on a join, never a key of one ---------------------

# Two sources binding the same identity and carrying the same instant. Timestamps agree
# because retrieval returns only rows inside the window by construction.
_WINDOWED_LOGS = {
    "alert_feed": [
        {"actor": {"userId": "U1"}, "raisedTime": "2026-07-24T20:31:00"},
        {"actor": {"userId": "U2"}, "raisedTime": "2026-07-24T20:31:00"},
    ],
    "activity_feed": [
        {"uid": "U1", "eventTime": "2026-07-24T20:31:00"},
        {"uid": "U3", "eventTime": "2026-07-24T20:31:00"},
    ],
}


def _windowed_pack(spec=None):
    from src.knowledge.pack import EntityDef, SourceDef

    return _pack_with_spec(
        spec=spec,
        entities=[
            EntityDef(type="user", field_aliases=["uid", "actor.userId"]),
            # A pack MAY declare the window as an entity — the retrieval layer needs the
            # aliases to recognise a time column. Declaring it must not make it joinable.
            EntityDef(type="time_window", field_aliases=["raisedTime", "eventTime"]),
        ],
        sources=[
            SourceDef(
                name="alert_feed",
                entity_bindings={
                    "user": ["actor.userId"],
                    "time_window": ["raisedTime"],
                },
            ),
            SourceDef(
                name="activity_feed",
                entity_bindings={"uid": ["uid"], "time_window": ["eventTime"]},
            ),
        ],
    )


def test_the_run_window_is_not_admitted_as_a_join_key_from_the_understanding(caplog):
    """An engine-owned time window named in ``correlation_keys`` resolves to NO key.

    Regression, measured on a live run: the understanding emitted ``time_window`` beside the
    real identities and the event window was a zero-width instant. Layer 2 consults neither
    the type's declared ``role`` nor the retrieval layer's own exclusion of it, so it bound a
    timestamp column on 11 of the run's sources and a ``join_time_window`` overlap step ran
    across all of them. The join is a TAUTOLOGY — every row is inside the window already —
    and it is reported in the same shape, under the same heading, as a shared identifier,
    with ``overlap_score`` 0.0 on every declared/derived key so nothing distinguishes them.
    """
    module = CorrelationModule({}, MagicMock(), knowledge_pack=_windowed_pack())
    schema = derive_schema(_WINDOWED_LOGS)
    analysis = _understanding([], correlation_keys=["user", "time_window"]).analysis
    with caplog.at_level(logging.INFO):
        keys = module._resolve_correlation_keys(analysis, schema, _WINDOWED_LOGS, None)

    assert not [k for k in keys if k.entity_hint == "time_window"]
    # NOT vacuous: the identity named beside it in the same list still resolves, so the
    # refusal is about the type and not about this fixture failing to bind anything.
    assert [k for k in keys if k.entity_hint == "user"]
    # And it is LOGGED. A declaration dropped in silence reads as a column missing from the
    # schema, which sends an author looking for a binding that was never the problem.
    assert any(
        "time_window" in r.getMessage() and "time bound, not an identity" in r.getMessage()
        for r in caplog.records
    )


def test_the_run_window_is_not_admitted_as_a_join_key_from_a_playbook_either():
    """The same refusal one layer up: whichever layer names it, the key is the same tautology.

    Checked separately because the layers do not share a code path — the live defect was in
    the understanding layer, and a fix applied only there would leave a pack able to declare
    the identical meaningless join under the authoritative origin.
    """
    spec = {"keys": ["time_window", "user"], "title": "pb"}
    module = CorrelationModule({}, MagicMock(), knowledge_pack=_windowed_pack(spec))
    schema = derive_schema(_WINDOWED_LOGS)
    analysis = _understanding([]).analysis
    keys = module._resolve_correlation_keys(
        analysis, schema, _WINDOWED_LOGS, module._playbook_correlation_spec(analysis)
    )

    assert not [k for k in keys if k.entity_hint == "time_window"]
    user = [k for k in keys if k.entity_hint == "user"]
    assert user and user[0].origin == "playbook"


def test_discovery_still_refuses_a_shared_instant_as_a_key_on_its_own():
    """The third layer needs no guard, and this is what keeps that true.

    Discovery works from value SETS, and ``_is_candidate_field`` rejects a timestamp field
    outright — *they belong to the time-window, not the join key* — which is why the two
    name-bound layers above are the only ones that needed the refusal added. The rule is
    load-bearing rather than incidental: a shared instant is the easiest key in any run to
    discover, since every row really is inside the window, so it would arrive with a HIGH
    overlap score rather than the 0.0 a declared key carries. If that rejection ever
    loosens, the third layer reopens the defect in its worst form and nothing else says so.
    """
    logs = {
        "alert_feed": [
            {
                "actor": {"userId": f"U{1000 + i}"},
                "raisedTime": f"2026-07-24T20:{i:02d}:00",
            }
            for i in range(30)
        ],
        "activity_feed": [
            {"uid": f"U{1000 + i}", "eventTime": f"2026-07-24T20:{i:02d}:00"}
            for i in range(30)
        ],
    }
    found = discover_join_keys(logs, derive_schema(logs), "strict")
    bound = {f for dk in found for f in dk["sources"].values()}
    # NOT vacuous: the identity sharing exactly the same 30-value overlap IS discovered.
    assert {"actor.userId", "uid"} <= bound
    assert not ({"raisedTime", "eventTime"} & bound)


def test_refusing_the_window_key_leaves_the_windows_real_role_untouched():
    """The window still GATES the other keys — that is what the type is for in this module.

    The bound direction of the same fix: refusing the key must not cost a declared
    ``time_window:``/``time_fields:`` gap, which is the window doing its actual job (a pair
    of rows may only join if their instants are within the declared distance). A fix that
    dropped the type wholesale would silently widen every declared join to unbounded.
    """
    spec = {
        "keys": ["time_window", "user"],
        "title": "pb",
        "time_window": "within:24h",
        "time_fields": {"alert_feed": "raisedTime", "activity_feed": "eventTime"},
    }
    module = CorrelationModule({}, MagicMock(), knowledge_pack=_windowed_pack(spec))
    schema = derive_schema(_WINDOWED_LOGS)
    analysis = _understanding([]).analysis
    keys = module._resolve_correlation_keys(
        analysis, schema, _WINDOWED_LOGS, module._playbook_correlation_spec(analysis)
    )
    user = [k for k in keys if k.entity_hint == "user"][0]
    assert user.time_window == "within:24h"
    assert user.time_fields == {
        "alert_feed": "raisedTime",
        "activity_feed": "eventTime",
    }
    # ...and it reaches the plan step, which is where the gate is actually applied.
    step = [s for s in module._build_deterministic_plan(keys).steps if s.entity == "user"]
    assert step and step[0].time_window == "within:24h"


# --- playbook match: it selects the RULESET, so it may not tie on declaration order ---


def _pack_with_two_specs():
    """A pack shipping TWO procedures whose join keys are identical.

    The realistic shape, and the one that broke: two procedures over one domain read the
    same identities, so the KEYS cannot discriminate and only the titles can.
    """
    from src.knowledge.pack import KnowledgePack

    pack = KnowledgePack()
    pack.playbook_documents = [
        {
            "title": "Ledger Settlement Monitoring — Suspected Settlement Fraud",
            "content": "c",
            "type": "playbook",
            "metadata": {
                "playbook_id": "PB-A-1",
                "use_case": "settlement",
                "correlation": {"keys": ["user", "org_unit", "session"]},
            },
        },
        {
            "title": "Custodian Rights Abuse — Unusual Delegated Administration",
            "content": "c",
            "type": "playbook",
            "metadata": {
                "playbook_id": "PB-B-2",
                "use_case": "delegation",
                "correlation": {"keys": ["user", "org_unit", "session"]},
            },
        },
    ]
    return pack


def test_a_prose_title_is_tokenised_as_prose_not_as_a_field_name():
    """The defect: the title went through the DOTTED/UNDERSCORED FIELD-NAME splitter.

    `_name_tokens` splits on `.` and `_` only, so a multi-word title came back as ONE
    token — a whole sentence, which no incident text can ever contain. Every title
    therefore scored zero and the match collapsed onto the join keys, which two
    procedures in one domain share. This asserts the two tokenizers stay distinct, in
    both directions, because reusing either one for the other's job is the bug.
    """
    from src.correlation import _name_tokens, _prose_tokens

    title = "Custodian Rights Abuse — Unusual Delegated Administration"
    # The field-name splitter cannot see inside prose: one unmatchable token.
    assert _name_tokens(title) == [title.lower()]
    # The prose splitter yields words an incident summary can actually contain.
    assert "custodian" in _prose_tokens(title)
    assert "delegated" in _prose_tokens(title)
    assert "—" not in " ".join(_prose_tokens(title))
    # And the prose splitter must NOT be used on field names: it would split the dotted
    # path into segments that no longer identify the leaf.
    assert _name_tokens("scoredLocationFeature.ip_addr") == [
        "scoredlocationfeature",
        "ip",
        "addr",
    ]


def test_the_matched_playbook_is_the_one_the_incident_names_not_the_first_declared():
    """A tie broken by declaration order chose the WRONG PROCEDURE on a live run.

    The matched playbook's `use_case` selects the ruleset the verdict is adjudicated
    under, so this is not a join-key nicety: the losing procedure's conditions resolve
    against real rows and produce a confident wrong verdict under the wrong labels.
    Both specs here declare identical keys, so ONLY the title can decide — and the
    expected winner is declared SECOND, so passing by luck is not available.
    """
    module = CorrelationModule({}, MagicMock(), knowledge_pack=_pack_with_two_specs())
    analysis = _understanding(
        [],
        summary=(
            "An alert fired for unusual delegated administration by a custodian: one "
            "user granted rights over another org_unit's records in a single session."
        ),
    ).analysis
    assert module._playbook_correlation_spec(analysis)["use_case"] == "delegation"

    # ... and the sibling procedure still wins on ITS OWN incident (declared first, so
    # this half would pass even with the defect — it is here to prove the fix did not
    # simply invert the order).
    other = _understanding(
        [],
        summary=(
            "A ledger settlement monitoring alert reports suspected settlement fraud "
            "by one user in a single session."
        ),
    ).analysis
    assert module._playbook_correlation_spec(other)["use_case"] == "settlement"


def test_hypotheses_are_a_tiebreak_only_because_they_enumerate_rival_patterns():
    """The understanding stage NAMES the alternatives — that is a good hypothesis list.

    So scoring them as primary evidence votes for every procedure at once. Measured over
    27 labelled live runs: summary alone 27/27, summary + investigation areas 17/27, all
    three fields 10/27. Here the summary points at one procedure while the hypotheses
    argue for the other; the summary must win.
    """
    module = CorrelationModule({}, MagicMock(), knowledge_pack=_pack_with_two_specs())
    analysis = _understanding(
        [],
        summary="Unusual delegated administration by a custodian in one session.",
    ).analysis
    analysis.initial_hypotheses = [
        "Ledger settlement fraud: the suspected settlement was monitored and diverted",
        "Suspected settlement fraud again, to weight this rival pattern twice",
    ]
    assert module._playbook_correlation_spec(analysis)["use_case"] == "delegation"

    # With NOTHING in the summary, the rival-pattern text is the only evidence there is,
    # and using it beats returning no spec at all.
    blank = _understanding([], summary="An alert fired.").analysis
    blank.initial_hypotheses = ["Ledger settlement fraud is suspected"]
    assert module._playbook_correlation_spec(blank)["use_case"] == "settlement"


def test_a_token_most_procedures_declare_cannot_discriminate_between_them():
    """A term the whole domain shares is not evidence FOR any one procedure.

    Needs THREE specs to bite, and that is the point: with two, a shared token adds the
    same constant to both and raw counting happens to order them correctly. With three,
    two specs sharing four words each out-count the one spec the incident actually names
    — so raw counting picks a procedure on the strength of vocabulary its rivals use
    just as much. The pack here is built so `gamma` scores LOWER on raw hits (3 vs 4)
    and higher on weight, which is exactly the disagreement being pinned.
    """
    from src.knowledge.pack import KnowledgePack

    shared = "Custodian Delegated Review"
    pack = KnowledgePack()
    pack.playbook_documents = [
        {
            "title": f"Alpha {shared}",
            "content": "c",
            "type": "playbook",
            "metadata": {
                "playbook_id": "PB-A",
                "use_case": "alpha",
                "correlation": {"keys": ["user"]},
            },
        },
        {
            "title": f"Beta {shared}",
            "content": "c",
            "type": "playbook",
            "metadata": {
                "playbook_id": "PB-B",
                "use_case": "beta",
                "correlation": {"keys": ["user"]},
            },
        },
        {
            "title": "Gamma Transfer",
            "content": "c",
            "type": "playbook",
            "metadata": {
                "playbook_id": "PB-C",
                "use_case": "gamma",
                "correlation": {"keys": ["user"]},
            },
        },
    ]
    module = CorrelationModule({}, MagicMock(), knowledge_pack=pack)
    # Names gamma's two distinctive words, plus the three every procedure shares.
    analysis = _understanding(
        [],
        summary=(
            "A gamma transfer alert, raised during custodian delegated review, "
            "naming one user."
        ),
    ).analysis
    assert module._playbook_correlation_spec(analysis)["use_case"] == "gamma"


def test_no_keyword_match_at_all_still_declines_to_guess():
    """Unchanged contract: several specs and no evidence returns None, never a guess.

    Selecting a procedure the incident gave no reason to select is the failure this whole
    seam exists to prevent, so "no match" must stay distinguishable from "matched".
    """
    module = CorrelationModule({}, MagicMock(), knowledge_pack=_pack_with_two_specs())
    analysis = _understanding([], summary="zzqqxx").analysis
    assert module._playbook_correlation_spec(analysis) is None

    # A single spec is unambiguous, so it is still returned with no match at all.
    from src.knowledge.pack import KnowledgePack

    one = KnowledgePack()
    one.playbook_documents = [
        {
            "title": "Only Procedure",
            "content": "c",
            "type": "playbook",
            "metadata": {"playbook_id": "PB-1", "correlation": {"keys": ["user"]}},
        }
    ]
    solo = CorrelationModule({}, MagicMock(), knowledge_pack=one)
    assert solo._playbook_correlation_spec(analysis) is not None


# --- propagation: a resolved key reaches the sources the priors missed -------


_PROPAGATION_LOGS = {
    # Two sources the pack binds for `record`, sharing one value.
    "alerts": [{"rec_id": "R100", "who": "A1"}],
    "audit": [{"rec_id": "R100", "who": "A1"}],
    # SAME column name, SAME population — the pack just never bound it for `record`.
    "record_lake": [{"rec_id": "R100", "detail": "d"}],
    # SAME column name, DIFFERENT population: a cohort sweep of other records.
    "cohort_sweep": [{"rec_id": f"R{200 + i}"} for i in range(40)],
    # Same value, DIFFERENT column name — a judgement only the pack may make.
    "renamed": [{"booking_ref": "R100"}],
}


def _propagation_pack():
    from src.knowledge.pack import EntityDef, KnowledgePack, SourceDef

    return KnowledgePack(
        # NO global `field_aliases`: those are pack-wide and would bind every source that
        # has the column, which is layer 2 doing the work and would leave this block
        # untested. What is under test is the source the pack bound NOWHERE.
        entities=[EntityDef(type="record")],
        sources=[
            SourceDef(name="alerts", entity_bindings={"record": ["rec_id"]}),
            SourceDef(name="audit", entity_bindings={"record": ["rec_id"]}),
            # record_lake / cohort_sweep / renamed: deliberately unbound for `record`.
        ],
    )


def _propagated_record_key():
    module = CorrelationModule({}, MagicMock(), knowledge_pack=_propagation_pack())
    analysis = _understanding(
        [ExtractedEntity(type="record", value="R100")], correlation_keys=["record"]
    ).analysis
    keys = module._resolve_correlation_keys(
        analysis, derive_schema(_PROPAGATION_LOGS), _PROPAGATION_LOGS, None
    )
    return [k for k in keys if k.entity_hint == "record"][0]


def test_a_resolved_key_reaches_a_source_the_packs_priors_never_bound():
    """A join over 2 of 3 sources holding the value looks exactly like a complete one.

    Every layer binds a source only through the pack's priors, so a source carrying the
    key under the same column name as its siblings — but never bound for that entity — is
    silently left out. Measured on job 4da14f65: the subject-record key spanned 2 sources
    while three more carried the same fully-populated column, the system of record among
    them.
    """
    key = _propagated_record_key()
    assert key.sources["alerts"] == "rec_id"  # declared
    assert key.sources["audit"] == "rec_id"  # declared
    assert key.sources["record_lake"] == "rec_id"  # PROPAGATED


def test_propagation_refuses_a_same_named_column_holding_a_DIFFERENT_population():
    """The name is not the evidence — the value overlap is.

    `cohort_sweep.rec_id` is spelled identically and is a real record column, but holds 40
    OTHER records. Joining it would manufacture "this record appears in 4 sources" out of
    a name collision, which is the failure this whole layer would otherwise introduce.
    """
    assert "cohort_sweep" not in _propagated_record_key().sources


def test_propagation_refuses_a_differently_named_column_holding_the_SAME_value():
    """Same value, different name, is a MEANING judgement — the pack's alone to make.

    `renamed.booking_ref` carries exactly the alerted value, and it may well be the same
    entity. The engine cannot know that: this layer is allowed to confirm a binding the
    pack already made elsewhere, never to invent one. Declaring it is a one-line pack edit
    (`entity_bindings`), which is the right place for the claim.
    """
    assert "renamed" not in _propagated_record_key().sources


def test_propagation_needs_two_already_bound_sources_so_it_cannot_bootstrap():
    """A key resting on ONE source is unverified — widening it would launder a guess."""
    from src.knowledge.pack import EntityDef, KnowledgePack, SourceDef

    logs = {
        "alerts": [{"rec_id": "R100"}],  # the only bound source
        "record_lake": [{"rec_id": "R100"}],  # unbound, same column, same value
    }
    pack = KnowledgePack(
        entities=[EntityDef(type="record")],  # NO global aliases
        sources=[SourceDef(name="alerts", entity_bindings={"record": ["rec_id"]})],
    )
    module = CorrelationModule({}, MagicMock(), knowledge_pack=pack)
    analysis = _understanding(
        [ExtractedEntity(type="record", value="R100")], correlation_keys=["record"]
    ).analysis
    keys = module._resolve_correlation_keys(analysis, derive_schema(logs), logs, None)
    # One bound source never became a key at all (>=2 sources required), so there is
    # nothing for propagation to widen — and it must not create one.
    assert not [k for k in keys if k.entity_hint == "record"]


# --- the cardinality floor inverts on a targeted query ----------------------


def test_a_field_holding_only_the_alerted_value_is_not_low_cardinality():
    """The floor assumes few distinct values means few identities. A filter inverts it.

    `_MIN_JK_DISTINCT` rejects status enums and small counters. But a targeted query has
    already narrowed to one actor and one record, so the columns the investigation is
    ABOUT hold exactly one value each and are discarded for it. Measured on job 4da14f65:
    31 single-valued fields rejected across 8 sources, every one an alerted entity.
    """
    logs = {
        "auth": [{"login": "WHO1", "status": "ok"} for _ in range(30)],
        "records": [{"acting_login": "WHO1", "status": "ok"} for _ in range(30)],
    }
    schema = derive_schema(logs)
    # Both columns hold ONE value, so today the pair is never even compared.
    assert discover_join_keys(logs, schema, "strict") == []

    found = discover_join_keys(logs, schema, "strict", None, {"who1"})
    assert found, "the alerted identity is the one value that cannot be a coincidence"
    assert found[0]["sources"] == {"auth": "login", "records": "acting_login"}
    # `status` is single-valued too — and is NOT an incident entity, so it stays out.
    for k in found:
        assert "status" not in k["sources"].values()


def test_the_exemption_needs_EVERY_value_named_not_merely_one():
    """A column that happens to CONTAIN the alerted value is still low-information.

    The narrow form ("all of them") is what distinguishes a field the query resolved from
    a broad field the subject happens to appear in — the latter is exactly what the floor
    is for.
    """
    from src.correlation import _is_candidate_field

    subjects = {"who1"}
    assert _is_candidate_field("login", {"WHO1"}, "strict", subjects)
    assert not _is_candidate_field(
        "login", {"WHO1", "other", "third"}, "strict", subjects
    )
    # Case-insensitive: a backend may echo the identifier in another case.
    assert _is_candidate_field("login", {"who1"}, "strict", subjects)
    # No subject set at all is the old behaviour, unchanged.
    assert not _is_candidate_field("login", {"WHO1"}, "strict", None)


def test_a_timestamp_is_still_rejected_even_when_the_incident_named_it():
    """The exemption relaxes the CARDINALITY floor only, never the shape rules.

    An incident always names its event time, so an exemption applied one step too broadly
    would promote every timestamp column to a join key — and a join on a timestamp groups
    by when, not by who.
    """
    from src.correlation import _is_candidate_field

    ts = "2026-08-04T03:03:00.000Z"
    assert not _is_candidate_field("event_time", {ts}, "strict", {ts.lower()})
    assert not _is_candidate_field("raisedTime", {ts}, "strict", {ts.lower()})


def test_the_subject_value_set_comes_from_the_incidents_own_entities():
    from src.correlation import _subject_value_set

    analysis = _understanding(
        [
            ExtractedEntity(type="record", value="R100"),
            ExtractedEntity(type="user", value=" Who1 "),  # stripped + lowercased
            ExtractedEntity(type="user", value=""),  # empty contributes nothing
        ]
    ).analysis
    assert _subject_value_set(analysis) == {"r100", "who1"}
    assert _subject_value_set(_understanding([]).analysis) == set()


# --- time-window co-occurrence ----------------------------------------------


def test_cross_source_overlap_time_window_same_day():
    logs = {
        "a": [{"record": "SUBJ00", "ts": "2024-08-26T10:00:00"}],
        "b": [{"record": "SUBJ00", "ts": "2024-08-26T15:00:00"}],  # same day -> counts
        "c": [{"record": "SUBJ00", "ts": "2024-09-01T10:00:00"}],  # different day
    }
    step = TransformStep(
        op="cross_source_overlap",
        label="j",
        entity="record",
        sources=["a", "b"],
        time_window="same_day",
        time_fields={"a": "ts", "b": "ts"},
    )
    rows = execute_plan(TransformPlan(steps=[step]), logs)[0].rows
    assert rows and rows[0]["value"] == "SUBJ00"

    # Now a + c are different days -> excluded.
    step2 = TransformStep(
        op="cross_source_overlap",
        label="j2",
        entity="record",
        sources=["a", "c"],
        time_window="same_day",
        time_fields={"a": "ts", "c": "ts"},
    )
    rows2 = execute_plan(TransformPlan(steps=[step2]), logs)[0].rows
    assert rows2 == []


# --- volume gate: deterministic vs LLM --------------------------------------


@pytest.mark.asyncio
async def test_analyze_high_volume_skips_llm():
    # Many rows across novel sources -> deterministic path, no LLM call.
    logs = {
        "src_a": [{"uid": f"U{i}"} for i in range(1500)],
        "src_b": [{"uid": f"U{i}"} for i in range(1500)],
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock()
    module = CorrelationModule({}, llm)  # no pack
    result = await module.analyze(logs, _understanding([], correlation_keys=[]))

    llm.structured_output.assert_not_called()  # deterministic
    # Discovered the shared uid deterministically and recorded it.
    assert "discovered_join_keys" in result.aggregations
    assert any(t.op == "cross_source_overlap" for t in result.transforms)


@pytest.mark.asyncio
async def test_analyze_low_volume_unresolved_uses_llm():
    # Tiny data, no resolvable keys (single source) -> LLM path.
    logs = {"only": [{"weird_col": "x"}]}
    plan = TransformPlan(steps=[])
    narration = CorrelationResult(findings=[], summary_text="llm", record_count=0)
    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=[plan, narration])
    module = CorrelationModule({}, llm)
    result = await module.analyze(logs, _understanding([]))
    assert llm.structured_output.await_count == 2  # plan + narrate
    assert result.summary_text == "llm"


@pytest.mark.asyncio
async def test_analyze_records_resolved_keys_in_aggregations():
    logs = {
        "src_a": [{"uid": f"U{i}"} for i in range(50)],
        "src_b": [{"uid": f"U{i}"} for i in range(50)],
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock()
    module = CorrelationModule({}, llm)
    result = await module.analyze(logs, _understanding([], correlation_keys=[]))
    assert isinstance(result.aggregations.get("resolved_correlation_keys"), list)


def test_should_use_llm_gate():
    module = CorrelationModule(
        {"llm_max_records": 100, "llm_max_sources": 3}, MagicMock()
    )
    # High volume -> never LLM.
    assert (
        module._should_use_llm({"total_records": 500, "record_counts": {"a": 500}}, [])
        is False
    )
    # Low volume, no resolved keys -> LLM.
    assert (
        module._should_use_llm({"total_records": 10, "record_counts": {"a": 10}}, [])
        is True
    )
    # Low volume but keys already resolved -> deterministic.
    from src.models.pydantic_models import CorrelationKey

    k = [CorrelationKey(entity_hint="user", sources={"a": "uid", "b": "uid"})]
    assert (
        module._should_use_llm({"total_records": 10, "record_counts": {"a": 10}}, k)
        is False
    )


def test_build_deterministic_plan_from_keys():
    from src.models.pydantic_models import CorrelationKey

    keys = [
        CorrelationKey(
            entity_hint="user",
            sources={"a": "uid", "b": "userId"},
            time_window="same_day",
            time_fields={"a": "ts", "b": "ts"},
        ),
        CorrelationKey(entity_hint="record", sources={"a": "record", "b": "locator"}),
    ]
    module = CorrelationModule({}, MagicMock())
    plan = module._build_deterministic_plan(keys)
    assert len(plan.steps) == 2
    assert all(s.op == "cross_source_overlap" for s in plan.steps)
    user_step = [s for s in plan.steps if s.entity == "user"][0]
    assert user_step.time_window == "same_day"


# --- epoch timestamps + relaxed time-window (evidence prerequisites) --------


def test_parse_ts_epoch_int_and_millis():
    from src.correlation import _parse_ts

    sec = _parse_ts(1721853060)  # epoch seconds
    ms = _parse_ts(1721853060000)  # epoch millis
    assert sec is not None and ms is not None
    assert sec == ms
    assert _parse_ts("1721853060") == sec  # numeric string epoch
    assert _parse_ts("2024-07-24T20:31:00Z") is not None  # ISO still works
    assert _parse_ts("junk") is None


def test_within_window_ignores_sources_without_times():
    from src.correlation import _within_window

    # Source "b" has no parsed times; it must NOT veto the join (old bug returned False).
    tbs = {"a": [_parse_ts_or(1721853060)], "b": []}
    assert _within_window(tbs, "within:24h") is True
    # Two sources with times far apart still fail the window (gating still meaningful).
    tbs2 = {"a": [_parse_ts_or(1721853060)], "b": [_parse_ts_or(1731853060)]}
    assert _within_window(tbs2, "within:1h") is False


def _parse_ts_or(v):
    from src.correlation import _parse_ts

    return _parse_ts(v)


def test_cross_source_overlap_survives_unparseable_time_in_one_source():
    """Epoch in one source, missing time in the other -> join still populates."""
    logs = {
        "a": [{"record": "SUBJ00", "ts": 1721853060000}],  # epoch millis
        "b": [{"record": "SUBJ00"}],  # no time field
    }
    step = TransformStep(
        op="cross_source_overlap",
        label="j",
        entity="record",
        sources=["a", "b"],
        time_window="within:48h",
        time_fields={"a": "ts"},
    )
    rows = execute_plan(TransformPlan(steps=[step]), logs)[0].rows
    assert rows and rows[0]["value"] == "SUBJ00"


@pytest.mark.asyncio
async def test_analyze_populates_evidence():
    logs = {
        "src_a": [{"uid": f"U{i}", "ts": 1721853060 + i} for i in range(50)],
        "src_b": [{"uid": f"U{i}", "ts": 1721853060 + i} for i in range(50)],
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock()
    module = CorrelationModule({}, llm)
    result = await module.analyze(logs, _understanding([], correlation_keys=[]))
    assert result.evidence is not None
    assert result.evidence.chronology  # events built
    assert result.evidence.total_records == 100


@pytest.mark.asyncio
async def test_analyze_evidence_none_on_build_failure(monkeypatch):
    logs = {
        "src_a": [{"uid": f"U{i}"} for i in range(50)],
        "src_b": [{"uid": f"U{i}"} for i in range(50)],
    }
    import evidence as evidence_mod

    def _boom(*a, **k):
        raise RuntimeError("evidence exploded")

    monkeypatch.setattr(evidence_mod, "build_evidence", _boom)
    llm = MagicMock()
    llm.structured_output = AsyncMock()
    module = CorrelationModule({}, llm)
    result = await module.analyze(logs, _understanding([], correlation_keys=[]))
    # Best-effort: analyze still returns, evidence is None, pipeline unaffected.
    assert result.evidence is None
    assert result.record_count == 100


# --- pack-driven validation verdict engine (evaluate_verdict) ---------------

from src.correlation import (
    _MAX_ACTING_TUPLES,
    _identifiers_match,
    evaluate_verdict,
)


def _scheme_spec():
    """A compact SCHEME-shaped ruleset mirroring rulesets.yaml (kept inline so the test
    doesn't depend on the shipped pack file)."""
    return {
        "label_scheme": "scheme",
        "subject_entity": "record",
        "labels": {
            "fraud": "VALID FRAUD",
            "false_positive": "FALSE POSITIVE",
            "insufficient": "INSUFFICIENT DATA",
        },
        "sources": {"record": "record_lake", "settlement": "settlement_report"},
        "conditions": [
            {
                "id": "bare",
                "label": "Bare",
                "kind": "element_absence",
                "source": "record",
                "counters": [
                    "element_counters.AUX",
                    "element_counters.OSI",
                    "element_counters.RM",
                    "element_counters.INS",
                ],
                "decisive": True,
            },
            {
                "id": "no_es",
                "label": "No ES",
                "kind": "element_absence",
                "source": "record",
                "counters": ["element_counters.ES"],
                "decisive": False,
            },
            {
                "id": "no_split",
                "label": "No split",
                "kind": "element_absence",
                "source": "record",
                "counters": ["element_counters.SP"],
                "arrays": ["xref.sp"],
                "decisive": False,
            },
            {
                "id": "same_agent",
                "label": "Same agent",
                "kind": "field_equality",
                "left": {"source": "record", "field": "creator.sign.red"},
                "right": {"source": "settlement", "field": "retrieverUserSign"},
                "normalize": "identifier",
                "decisive": True,
            },
            {
                "id": "single_actor",
                "label": "Single actor",
                "kind": "distinct_count",
                "source": "settlement",
                "field": "retrieverUserSign",
                "max": 1,
                "decisive": True,
            },
            {
                "id": "immediate_issuance",
                "label": "Immediate",
                "kind": "time_gap",
                "start": {"source": "record", "field": "creation_date_time"},
                "end": {"source": "settlement", "field": "transactionDateTime"},
                "max": "1h",
                "decisive": True,
            },
            {
                "id": "no_void_refund",
                "label": "No void/refund",
                "kind": "record_absence",
                "source": "settlement",
                "fields": ["recordType"],
                "forbidden_values": ["VOID", "REFUND"],
                "decisive": True,
            },
            {
                "id": "route_in_scope",
                "label": "In-scope route",
                "kind": "route_membership",
                "source": "record",
                "point_fields": [
                    "route.air.board_point",
                    "route.air.off_point",
                ],
                "decisive": True,
            },
        ],
        "lock_target": {
            "source": "record",
            "scope_field": "creator.org_unit_id",
            "identity_field": "creator.sign.red",
        },
        # WHICH source and fields carry the issued assets + who acted on them is
        # pack-declared, not engine-wired (these used to be audit_trail column names hard-coded in
        # correlation.py, where the "actor" was whoever DISPLAYED the record).
        "asset_notes": {
            "source": "settlement",
            "asset_label": "document",
            "actor_label": "signs",
            "asset_id_fields": ["relatedDocumentNumber", "relatedDocuments"],
            "actor_fields": ["retrieverUserSign"],
        },
        "platform_mode": {
            "source": "record",
            "marker_label": "ATID",
            "marker_fields": ["contextual_data.sec.atid.red"],
            "present": {
                "id": "classic",
                "label": "Sell Classic",
                "known_prefixes": ["58", "67", "8C"],
            },
            "absent": {"id": "connect", "label": "Sell Connect"},
        },
        "routes": [["LBV", "CMN"], ["ABJ", "CMN"]],
        "notification": {
            "template": "Outcome for record {subject}: {verdict}. Lock {lock_scope}/{lock_identity}. {closing}",
            "closing_fraud": "Contain now.",
            "closing_false_positive": "Close as FP.",
            "closing_insufficient": "Re-run.",
        },
    }


class _Analysis:
    def __init__(self, record="SUBJ03"):
        self.extracted_entities = [ExtractedEntity(type="record", value=record)]


def _fraud_logs():
    return {
        "record_lake": [
            {
                "locator.red": "SUBJ03",
                "creation_date_time": "2026-07-24T09:00:00",
                "creator.org_unit_id": "ORG2428D4",
                "creator.sign.red": "0201GPSU",
                "element_counters.AUX": 0,
                "element_counters.OSI": 0,
                "element_counters.RM": 0,
                "element_counters.INS": 0,
                "element_counters.ES": 0,
                "element_counters.SP": 0,
                "route.air.board_point": "LBV",
                "route.air.off_point": "CMN",
                "contextual_data.sec.atid.red": "58ABC",
            }
        ],
        "settlement_report": [
            {
                "relatedRecordLocator": "SUBJ03",
                "retrieverUserSign": "0201GP",
                "retrieverOrgUnitId": "ORG2428D4",
                "transactionDateTime": "2026-07-24T09:20:00",
                "recordType": "SALE",
                "relatedDocumentNumber": "0571234567890",
            }
        ],
    }


def test_signs_match_prefix_and_mismatch():
    assert _identifiers_match("0201GPSU", "0201GP")  # numeric sign is a prefix
    assert _identifiers_match("0201GP", "0201GP")
    assert not _identifiers_match("0201GP", "USERNAMEX")  # login/name != numeric sign
    assert not _identifiers_match("0201GP", "0007VP")


def test_evaluate_verdict_valid_fraud():
    v = evaluate_verdict(_scheme_spec(), _fraud_logs(), _Analysis())
    assert v is not None and len(v.subjects) == 1
    s = v.subjects[0]
    assert s.verdict == "VALID FRAUD"
    assert s.lock_target["scope"] == "ORG2428D4"
    assert s.lock_target["identity"] == "0201GPSU"
    # order≠issuance-name still matches on numeric sign prefix
    assert next(c for c in s.checks if c.id == "same_agent").result == "pass"
    assert next(c for c in s.checks if c.id == "route_in_scope").result == "pass"
    assert "Sell Classic" in " ".join(s.notes)


def test_a_ruleset_declaring_no_platform_dimension_reports_no_platform_at_all():
    """An undeclared dimension is absent, not unknown. The two are different findings:
    unknown means the marker was sought and not found; absent means the check was not declared.

    Both legs are asserted: undeclared produces no platform note and no platform key in the
    lock target. Declared but undetermined still reports unknown, because that is a real gap
    that gates a choice between non-interchangeable action sets.
    """
    spec = _scheme_spec()
    assert spec.get("platform_mode"), "the fixture's positive leg needs the block"

    # Declared: the note is emitted and the target carries the class the action map reads.
    s = evaluate_verdict(spec, _fraud_logs(), _Analysis()).subjects[0]
    assert any(n.startswith("platform_mode=") for n in s.notes), s.notes
    assert s.lock_target.get("platform") == "classic", s.lock_target

    # Undeclared: neither. Not "unknown" — ABSENT, because the two are different findings and
    # a consumer handed `unknown` for both cannot tell them apart.
    bare = {k: v for k, v in spec.items() if k != "platform_mode"}
    s2 = evaluate_verdict(bare, _fraud_logs(), _Analysis()).subjects[0]
    assert not [n for n in s2.notes if n.startswith("platform_mode=")], s2.notes
    assert "platform" not in s2.lock_target, s2.lock_target
    # The rest of the containment target is untouched: the identity is still nominated, and it
    # is the verdict that authorises action, never the platform.
    assert s2.lock_target.get("identity") == "0201GPSU", s2.lock_target
    assert s2.verdict == s.verdict


def test_an_unwritten_sub_record_is_not_an_absent_marker():
    """A projected leaf arrives on every row; a written one does not. `_path_retrieved` can
    only answer the first, so an all-null leaf inside a sub-record that was never written reads
    as a confident absence. `requires_present` lets a pack require sibling witnesses before
    treating null as absence. Declaring nothing preserves the prior face-value reading, which
    is why the fourth leg asserts it unchanged.
    """
    from src.correlation import _detect_platform_mode

    spec = {
        "marker_label": "MARKER",
        "marker_fields": ["ctx.block.marker"],
        "present": {"id": "one", "label": "Platform One"},
        "absent": {
            "id": "two",
            "label": "Platform Two",
            "requires_present": ["ctx.block.scope", "ctx.block.actor"],
        },
    }
    # The sub-record WAS written — a sibling carries a value — so an empty marker is a real
    # absence and the absent-platform determination stands.
    prose, cls = _detect_platform_mode(
        [{"ctx.block.marker": "", "ctx.block.scope": "ORG1", "ctx.block.actor": None}],
        spec,
    )
    assert cls == "two" and "Platform Two" in prose

    # Nothing was written into it: every witness is empty too → unknown, and the prose names
    # the witnesses so a reader can see WHY it is not the absent determination.
    prose, cls = _detect_platform_mode(
        [{"ctx.block.marker": None, "ctx.block.scope": "", "ctx.block.actor": None}],
        spec,
    )
    assert cls == "unknown", prose
    assert "witness" in prose and "ctx.block.scope" in prose
    assert "Platform Two" not in prose and "Platform One" not in prose

    # A marker that IS there outranks the witnesses entirely — they only license an ABSENCE.
    _prose, cls = _detect_platform_mode(
        [
            {
                "ctx.block.marker": "1A2B3C4D",
                "ctx.block.scope": "",
                "ctx.block.actor": "",
            }
        ],
        spec,
    )
    assert cls == "one"

    # A ruleset declaring no witnesses is byte-for-byte the older reading.
    bare = dict(spec, absent={"id": "two", "label": "Platform Two"})
    _prose, cls = _detect_platform_mode([{"ctx.block.marker": None}], bare)
    assert cls == "two"
    # And the projection gap still wins over the witness rule: neither was retrieved, and
    # "no column" is a different sentence from "a column nobody wrote".
    prose, cls = _detect_platform_mode([{"unrelated": "x"}], spec)
    assert cls == "unknown" and "not in the projection" in prose


# --- the exit taken when nothing decisive fired -----------------------------------------
# `test_evaluate_verdict_valid_fraud` covers the branch where decisive exclusions enumerate
# the fraud fingerprint. For a procedure whose decisive exclusions only establish who acted,
# "nothing fired" reads as "guilty" rather than as an inconclusive result.


def test_the_no_fingerprint_exit_defaults_to_fraud_so_existing_rulesets_are_unchanged():
    """The whole back-compat claim in one assertion: no declaration, no behaviour change.

    Asserted on the spec that DECLARES NOTHING rather than trusting the test above to keep
    covering it — a later edit adding the key to `_scheme_spec` would silently retire the
    guarantee while both tests still passed.
    """
    spec = _scheme_spec()
    assert "no_exclusion_fired" not in spec
    s = evaluate_verdict(spec, _fraud_logs(), _Analysis()).subjects[0]
    assert (s.verdict, s.verdict_class) == ("VALID FRAUD", "fraud")


def test_a_ruleset_may_declare_that_nothing_fired_means_the_subject_is_CLEARED():
    """Same rows, same passing checks, opposite verdict — decided by the pack, not the engine.

    And the CLASS moves with the label, because every downstream consumer asks the class:
    the confidence clamp, the framing, and the containment gate, which would otherwise
    nominate an account for suspension on a verdict that just cleared it.
    """
    spec = _scheme_spec()
    spec["no_exclusion_fired"] = "false_positive"
    s = evaluate_verdict(spec, _fraud_logs(), _Analysis()).subjects[0]
    assert (s.verdict, s.verdict_class) == ("FALSE POSITIVE", "false_positive")
    assert not s.lock_target, "a cleared subject may not carry a containment target"
    # The exit is the LAST branch, so it must not shadow the ones before it: an actual
    # indicator FAIL still reaches fraud on the very same declaration.
    assert all(c.result == "pass" for c in s.checks if c.decisive)


def test_the_declared_exit_does_not_shadow_an_earlier_branch():
    """It is step 4 of 5. A decisive exclusion FAIL must still be adjudicated as one, or a
    ruleset declaring the key would answer identically however its checks came out."""
    spec = _scheme_spec()
    spec["no_exclusion_fired"] = "false_positive"
    logs = _fraud_logs()
    logs["record_lake"][0]["element_counters.AUX"] = 3  # not bare → decisive FAIL
    s = evaluate_verdict(spec, logs, _Analysis()).subjects[0]
    assert s.verdict_class == "false_positive"
    assert next(c for c in s.checks if c.id == "bare").result == "fail", (
        "the two exits agree on the LABEL here, so the assertion that separates them is "
        "which check drove it"
    )


def test_an_unreadable_exit_keeps_the_historical_one_rather_than_inventing_a_class():
    """A value the engine cannot map must not produce a class no consumer recognises: the
    label and `verdict_class` would then contradict each other for everything downstream.
    `pack_validate` reports it as an error; the engine's job is to stay coherent."""
    spec = _scheme_spec()
    spec["no_exclusion_fired"] = "cleared"  # not one of the three classes
    s = evaluate_verdict(spec, _fraud_logs(), _Analysis()).subjects[0]
    assert (s.verdict, s.verdict_class) == ("VALID FRAUD", "fraud")


def test_the_exit_prints_the_packs_own_label_and_not_the_engines_default():
    """The engine's fallback wording exists for a ruleset that declares no label at all. A
    pack that declared one must see it here — this is the verdict a reader meets most often.
    """
    spec = _scheme_spec()
    spec["no_exclusion_fired"] = "false_positive"
    spec["labels"]["false_positive"] = "EXAMINED AND CLEARED"
    s = evaluate_verdict(spec, _fraud_logs(), _Analysis()).subjects[0]
    assert s.verdict == "EXAMINED AND CLEARED"
    assert (
        s.verdict_class == "false_positive"
    ), "the class is the engine's, the words the pack's"


# --- the evidence floor on the CLEAR exit ------------------------------------------------
# Step 4 means "nothing failed", which a subject whose checks all passed and one whose checks
# were all unknown produce equally. The fraud direction catches decisive unknowns; a clear
# direction without a floor passes unknowns silently, naming a person as exonerated.


def _no_evidence_logs():
    """Every declared source present and EMPTY — the live shape. Not a missing key: a source
    that returned zero rows is what the pipeline actually hands the verdict, and it is what
    drives every condition to `unknown` rather than to a FAIL."""
    return {"record_lake": [], "settlement_report": []}


def _few_decisive_spec():
    """The ruleset shape the floor exists for, and the reason a plainer test proves nothing.

    On a spec with many DECISIVE checks, empty sources reach INSUFFICIENT DATA through the
    branch above step 4 (`has_decisive_unknown`), so the floor is never consulted and a test
    asserting the verdict would pass with the floor deleted. The exposed shape is a ruleset
    whose decisive exclusions establish only WHO ACTED — few decisive checks by construction —
    so the UNKNOWNs land on NON-decisive conditions, nothing is decisive-unknown, and step 4
    clears the subject. That is the live authentication ruleset: 1 decisive of 10.

    So this keeps exactly one decisive check and gives it a source that ANSWERS (a reference
    lookup that legitimately returns rows), leaving every other condition unevaluable.
    """
    spec = _scheme_spec()
    for cond in spec["conditions"]:
        cond["decisive"] = cond["id"] == "no_es"
    return spec


def _one_answered_source_logs():
    """The decisive check's source answers; every other source is empty."""
    return {
        "record_lake": [{"locator.red": "SUBJ03", "element_counters.ES": 0}],
        "settlement_report": [],
    }


def test_a_clear_needs_something_to_clear_on():
    """THE REGRESSION TEST. Against the prior engine this returned FALSE POSITIVE.

    One condition PASSED (the reference lookup) and seven could not be evaluated, which is the
    live shape — and it reaches step 4, because nothing DECISIVE is unknown.
    """
    spec = _few_decisive_spec()
    spec["no_exclusion_fired"] = "false_positive"
    spec["min_evaluated_to_clear"] = (
        4  # this procedure wants real coverage before clearing
    )
    v = evaluate_verdict(spec, _one_answered_source_logs(), _Analysis())
    s = v.subjects[0]
    assert (s.verdict, s.verdict_class) == ("INSUFFICIENT DATA", "insufficient")
    assert not any(c.result == "unknown" for c in s.checks if c.decisive), (
        "THE PREMISE. If a decisive check is unknown, the branch above step 4 produces this "
        "same verdict and the assertion proves nothing about the floor"
    )
    assert len([c for c in s.checks if c.result in ("pass", "fail")]) == 1
    assert v.degraded, (
        "a verdict reached because the conditions could not be evaluated is degraded by "
        "definition — a consumer reading the flag rather than the prose must not see a "
        "verdict resting on nothing as fully evidenced"
    )
    assert any(
        n.startswith("evidence_floor=") for n in s.notes
    ), "a verdict that changed class without saying why is the defect this fixes"


def test_the_floors_denominator_is_what_could_have_been_evaluated():
    """The one sentence an operator reads to decide whether a re-run is worth it.

    A `stub` is a condition the procedure requires and the pack declares no data path for, so
    it returns `unknown` without reading a row and can never join the `pass`/`fail` count the
    floor is compared against. Counted in the denominator it states a data loss this run did
    not suffer — here `1 of 10` for a run that answered one of eight answerable questions.
    Named separately rather than dropped, because a check nobody wired is still a gap in the
    procedure; it is just not one a re-run can close, and the two remedies are different.

    The numerator and the verdict are deliberately asserted unchanged: this is a prose fix,
    and a denominator change that moved a verdict would be a different and much worse bug.
    """
    spec = _few_decisive_spec()
    spec["no_exclusion_fired"] = "false_positive"
    spec["min_evaluated_to_clear"] = 4
    n_real = len(spec["conditions"])
    for i in range(2):
        spec["conditions"].append(
            {"id": f"unwired_{i}", "kind": "stub", "detail": "no data path yet"}
        )
    s = evaluate_verdict(spec, _one_answered_source_logs(), _Analysis()).subjects[0]
    assert (s.verdict, s.verdict_class) == ("INSUFFICIENT DATA", "insufficient")
    assert len([c for c in s.checks if c.result in ("pass", "fail")]) == 1
    note = next(n for n in s.notes if n.startswith("evidence_floor="))
    assert f"1 of {n_real} condition(s) could be evaluated" in note, note
    assert (
        f"1 of {n_real + 2}" not in note
    ), "the two stubs are not a loss this run suffered"
    assert "a further 2 declare no data path at all" in note, note

    # And the same ruleset with nothing unwired says nothing about data paths, so a pack that
    # declares no stub reads exactly as it did before this split existed.
    plain = _few_decisive_spec()
    plain["no_exclusion_fired"] = "false_positive"
    plain["min_evaluated_to_clear"] = 4
    p = evaluate_verdict(plain, _one_answered_source_logs(), _Analysis()).subjects[0]
    plain_note = next(n for n in p.notes if n.startswith("evidence_floor="))
    assert "no data path" not in plain_note, plain_note
    assert f"1 of {n_real} condition(s) could be evaluated" in plain_note, plain_note


def test_a_stub_is_the_only_check_the_marker_selects():
    """The marker is a string on `observed`, so what it must NOT match is the point.

    Every other unevaluated condition reports what it MEASURED (`field absent`, a cohort it
    was not in) and carries the verdict in `result`. If a real unknown ever wrote this literal
    the split would silently reclassify a retrieval gap as authoring work — the direction that
    tells an operator not to bother re-running.
    """
    spec = _few_decisive_spec()
    spec["conditions"].append({"id": "unwired", "kind": "stub", "detail": "not wired"})
    s = evaluate_verdict(spec, _one_answered_source_logs(), _Analysis()).subjects[0]
    unknown = [c for c in s.checks if c.result == "unknown"]
    assert len(unknown) > 2, "the premise: real unevaluated checks beside the stub"
    marked = [c for c in unknown if c.observed == STUB_OBSERVED]
    assert [c.id for c in marked] == ["unwired"], [(c.id, c.observed) for c in unknown]


def test_the_floor_does_not_touch_a_subject_that_WAS_examined():
    """The other half, and the one that would break the feature if it flipped: a ruleset
    declaring the clear exit must still clear a subject whose checks actually resolved.
    """
    spec = _scheme_spec()
    spec["no_exclusion_fired"] = "false_positive"
    v = evaluate_verdict(spec, _fraud_logs(), _Analysis())
    s = v.subjects[0]
    assert (s.verdict, s.verdict_class) == ("FALSE POSITIVE", "false_positive")
    assert not v.degraded
    assert not any(n.startswith("evidence_floor=") for n in s.notes)


def test_the_floor_is_the_packs_number_and_only_bites_on_the_clear_exit():
    """Two claims one test would leave ambiguous, so both are asserted on the same rows.

    A pack may demand more than one evaluated condition before it will clear an account; and
    the floor is scoped to the `false_positive` exit, so a ruleset exiting to `fraud` — every
    ruleset that declares nothing — cannot have its verdict changed by this key at all.
    """
    logs = _fraud_logs()
    # One decisive check unevaluable: the settlement side is gone, the record side is intact.
    logs["settlement_report"] = []
    spec = _scheme_spec()
    spec["no_exclusion_fired"] = "false_positive"
    spec["min_evaluated_to_clear"] = 99  # more than this ruleset has conditions
    s = evaluate_verdict(spec, logs, _Analysis()).subjects[0]
    assert s.verdict_class == "insufficient"

    fraud_spec = _scheme_spec()
    fraud_spec["min_evaluated_to_clear"] = 99
    assert "no_exclusion_fired" not in fraud_spec
    f = evaluate_verdict(fraud_spec, _no_evidence_logs(), _Analysis()).subjects[0]
    assert not any(
        n.startswith("evidence_floor=") for n in f.notes
    ), "the key is read only on the clear exit; a fraud-exiting ruleset must never consult it"


def test_an_unreadable_floor_falls_back_to_one_rather_than_disabling_itself():
    """A pack typo must not silently restore the behaviour this floor exists to prevent —
    the failure mode is a clear on no evidence, so an unusable value takes the safe floor.
    """
    spec = _scheme_spec()
    spec["no_exclusion_fired"] = "false_positive"
    for bad in ("many", None, 0, -5):
        spec["min_evaluated_to_clear"] = bad
        s = evaluate_verdict(spec, _no_evidence_logs(), _Analysis()).subjects[0]
        assert s.verdict_class == "insufficient", f"floor {bad!r} disabled the check"


# --- the same question on the NO-VERDICT exit --------------------------------------------
# `no_exclusion_fired: insufficient` is a third declaration: "nothing fired" is not a clear.
# The exit splits by whether every data path resolved: `no_verdict_reason=` when complete,
# `no_verdict_partial=` when some conditions had no source. At most one is emitted.


def _no_fire_spec():
    """`_scheme_spec()` with the one condition that cannot resolve removed.

    `no_split` reads `xref.sp`, which `_fraud_logs()` does not carry, so the shared fixture is
    the PARTIAL reading — useful, and tested as such below, but it cannot pin the sentence that
    only a complete subject may be told.
    """
    spec = _scheme_spec()
    spec["no_exclusion_fired"] = "insufficient"
    spec["conditions"] = [c for c in spec["conditions"] if c["id"] != "no_split"]
    return spec


def test_the_no_verdict_exit_says_that_NOTHING_FIRED_rather_than_nothing_came_back():
    """THE REGRESSION TEST. Three roads reach one label; the note is which road this was.

    `degraded` already separates them mechanically — it is False here, since no decisive check
    went `unknown` — but a flag is not what a reader reads, and the two remedies are opposite.
    """
    spec = _no_fire_spec()
    v = evaluate_verdict(spec, _fraud_logs(), _Analysis())
    s = v.subjects[0]
    assert (s.verdict, s.verdict_class) == ("INSUFFICIENT DATA", "insufficient")
    assert not v.degraded, (
        "THE PREMISE. If this were degraded the subject reached the label through missing "
        "data and the note under test would be the wrong sentence"
    )
    note = next(n for n in s.notes if n.startswith("no_verdict_reason="))
    n_eval = len([c for c in s.checks if c.result in ("pass", "fail")])
    assert n_eval == len(spec["conditions"]), "THE PREMISE of the complete reading"
    assert f"{n_eval} of {len(spec['conditions'])} condition(s) were evaluated" in note
    assert not any(c.result == "fail" for c in s.checks), (
        "THE SECOND PREMISE. Nothing failed at all here, which is the only shape the flat "
        "reading below is true of — see the voteless-FAIL test for the other one"
    )
    assert "nothing that can decide this subject fired" in note, note
    assert "no fraud indicator fired" in note, note
    assert "DID fail while voting nothing" not in note, note
    assert "re-run answers nothing" in note, note


def test_a_partially_answered_no_verdict_exit_is_not_told_that_a_re_run_is_pointless():
    """The other half of the exit, and the defect that split it.

    A subject whose conditions ALL resolved and one whose 7 of 8 resolved reach this exit
    identically — same class, same `degraded: false`, same census of PASSes — and the first
    draft told both of them "retrieval is not the gap here and a re-run answers nothing". On the
    second that is a false claim about the run, and it points the operator away from the one
    action that could still change the outcome. So the shortfall is named, under its own key,
    and the sentence a complete subject gets is absent.
    """
    spec = _scheme_spec()
    spec["no_exclusion_fired"] = "insufficient"
    v = evaluate_verdict(spec, _fraud_logs(), _Analysis())
    s = v.subjects[0]
    assert (s.verdict_class, v.degraded) == ("insufficient", False), (
        "THE PREMISE. Same exit, same flag as the complete reading — which is exactly why the "
        "note is the only thing that can tell them apart"
    )
    unresolved = [c.id for c in s.checks if c.result not in ("pass", "fail")]
    assert unresolved == ["no_split"], unresolved
    assert not any(n.startswith("no_verdict_reason=") for n in s.notes), (
        "the complete key is what the case builder reads to suppress the re-run step"
    )
    note = next(n for n in s.notes if n.startswith("no_verdict_partial="))
    assert "7 of 8 condition(s) were evaluated" in note, note
    assert "nothing that can decide this subject fired" in note, note
    assert "The other 1 did not resolve" in note, note
    assert "retrieval IS still a gap" in note, note
    assert "re-run answers nothing" not in note, note


def test_a_no_verdict_exit_names_the_fails_that_vote_nothing_rather_than_denying_them():
    """The third wrong sentence in the same note, and the one that denied real findings.

    Reaching this exit establishes two things only: no DECISIVE exclusion failed, and the fraud
    indicators did not reach the declared threshold. A FAIL on a NON-decisive exclusion — which
    is where every condition carrying no `polarity` lands, `exclusions` being the complement of
    `fraud_indicator` — routes nowhere. So "NONE of them fired" was false on any subject that had
    one, and a live companion subject printed it directly beneath two `fail` rows.
    """
    spec = _no_fire_spec()
    spec["conditions"].append(
        {
            "id": "solo_retriever",
            "label": "Nobody retrieved it",
            "kind": "distinct_count",
            "source": "settlement",
            "field": "retrieverUserSign",
            "max": 0,
            "decisive": False,
        }
    )
    s = evaluate_verdict(spec, _fraud_logs(), _Analysis()).subjects[0]
    fails = [c.id for c in s.checks if c.result == "fail"]
    assert fails == ["solo_retriever"], (
        "THE PREMISE. One FAIL, non-decisive and of no polarity, so it changes no verdict"
    )
    assert s.verdict_class == "insufficient", (
        "and the exit is unchanged — step 3 tests the decisive exclusions only, which is "
        "exactly why the note could contradict the census"
    )
    note = next(n for n in s.notes if n.startswith("no_verdict_reason="))
    assert "1 condition(s) DID fail while voting nothing" in note, note
    assert "solo_retriever" in note, note
    assert "read those as findings and not as an absence" in note, note
    assert "NONE of them fired" not in note, note


def test_the_no_verdict_note_counts_the_indicators_against_the_declared_threshold():
    """A sub-threshold indicator FIRED, and the note used to say none did.

    Two indicators against a threshold of three is the shape that reaches this exit with real
    positive evidence in hand: not corroboration, and not nothing. The count and the requirement
    are both stated because "no indicator fired" and "two fired and three are required" license
    different next steps, and only the second is worth a human's time.
    """
    spec = _no_fire_spec()
    spec["indicator_threshold"] = 3
    for i in range(2):
        spec["conditions"].append(
            {
                "id": f"ind_{i}",
                "label": f"Indicator {i}",
                "kind": "distinct_count",
                "source": "settlement",
                "field": "retrieverUserSign",
                "max": 0,
                "decisive": False,
                "polarity": "fraud_indicator",
            }
        )
    s = evaluate_verdict(spec, _fraud_logs(), _Analysis()).subjects[0]
    assert s.verdict_class == "insufficient", "THE PREMISE: 2 < 3, so no corroboration"
    note = next(n for n in s.notes if n.startswith("no_verdict_reason="))
    assert "2 fraud indicator(s) fired, short of the 3 this procedure requires" in note, note
    assert "no fraud indicator fired" not in note, note
    assert "DID fail while voting nothing" not in note, (
        "an indicator is not voteless — it voted and lost, which the clause above states"
    )


def test_the_no_verdict_note_shares_the_floors_denominator():
    """One denominator, one arithmetic — the reason `attempted` is computed above both branches.

    A `stub` can never enter the numerator, so counting it states a data loss this run did not
    suffer. The two sentences answer the same reader question and a reader who has learned one
    reading of `N of M` must not meet a second one two lines down.
    """
    spec = _no_fire_spec()
    n_real = len(spec["conditions"])
    for i in range(2):
        spec["conditions"].append(
            {"id": f"unwired_{i}", "kind": "stub", "detail": "no data path yet"}
        )
    s = evaluate_verdict(spec, _fraud_logs(), _Analysis()).subjects[0]
    note = next(n for n in s.notes if n.startswith("no_verdict_reason="))
    assert f"of {n_real} condition(s) were evaluated" in note, note
    assert f"of {n_real + 2}" not in note, "the two stubs are not a loss this run suffered"
    assert "a further 2 declare no data path at all" in note, note
    # AND A STUB DOES NOT MAKE THE READING PARTIAL. It never had a data path, so it is not an
    # unresolved question a re-run could answer — counting it as one would send every subject of
    # a stub-carrying ruleset (which is the procedure that motivated this exit) to the wrong key.
    assert not any(n.startswith("no_verdict_partial=") for n in s.notes), s.notes

    p = evaluate_verdict(_no_fire_spec(), _fraud_logs(), _Analysis()).subjects[0]
    plain_note = next(n for n in p.notes if n.startswith("no_verdict_reason="))
    assert "no data path" not in plain_note, plain_note


def test_the_two_historical_exits_gain_no_sentence():
    """The whole back-compat claim: every pack shipped before this declares `fraud` (by
    omission) or `false_positive`, and neither may grow a note. Asserted on both, because a
    branch scoped by a `!=` would leave exactly one of them intact.
    """
    fraud = _scheme_spec()
    assert "no_exclusion_fired" not in fraud
    f = evaluate_verdict(fraud, _fraud_logs(), _Analysis()).subjects[0]
    assert f.verdict_class == "fraud"
    assert not any(n.startswith("no_verdict_reason=") for n in f.notes)

    clear = _scheme_spec()
    clear["no_exclusion_fired"] = "false_positive"
    c = evaluate_verdict(clear, _fraud_logs(), _Analysis()).subjects[0]
    assert c.verdict_class == "false_positive"
    assert not any(n.startswith("no_verdict_reason=") for n in c.notes)


def test_the_evidence_floor_and_the_no_verdict_note_are_mutually_exclusive():
    """Both explain an INSUFFICIENT outcome and they render under ONE heading, so a subject
    carrying both would state two different reasons for one verdict — and the floor's is the
    true one, since it is what MOVED the exit. The floor wins by construction; this pins it.
    """
    spec = _few_decisive_spec()
    spec["no_exclusion_fired"] = "false_positive"
    spec["min_evaluated_to_clear"] = 4
    s = evaluate_verdict(spec, _one_answered_source_logs(), _Analysis()).subjects[0]
    assert s.verdict_class == "insufficient"
    assert any(n.startswith("evidence_floor=") for n in s.notes)
    assert not any(n.startswith("no_verdict_reason=") for n in s.notes), (
        "the floor already said why, and it says it about the exit it changed — a second "
        "sentence claiming every condition was evaluated would contradict it"
    )


def test_a_decisive_unknown_reaches_the_label_by_another_road_and_says_nothing_here():
    """The third road. `has_decisive_unknown` routes to `insufficient` several branches ABOVE
    step 4, so the declared exit is never consulted — and the note must not fire there, because
    its sentence ("retrieval is not the gap") would be exactly wrong.
    """
    spec = _scheme_spec()
    spec["no_exclusion_fired"] = "insufficient"
    v = evaluate_verdict(spec, _no_evidence_logs(), _Analysis())
    s = v.subjects[0]
    assert s.verdict_class == "insufficient"
    assert any(c.result == "unknown" for c in s.checks if c.decisive), "the premise"
    assert v.degraded, "a verdict resting on unevaluable decisive checks IS degraded"
    assert not any(n.startswith("no_verdict_reason=") for n in s.notes)


def test_evaluate_verdict_false_positive_not_bare():
    logs = _fraud_logs()
    logs["record_lake"][0]["element_counters.AUX"] = 3  # not bare → decisive FAIL
    v = evaluate_verdict(_scheme_spec(), logs, _Analysis())
    s = v.subjects[0]
    assert s.verdict == "FALSE POSITIVE"
    assert next(c for c in s.checks if c.id == "bare").result == "fail"


def test_false_positive_terminal_not_degraded_when_settlement_missing():
    """A decisive FAIL that is OBSERVED (AUX present → not bare) makes the FALSE
    POSITIVE terminal: even though the settlement-dependent decisive checks come back UNKNOWN
    (settlement_report returned no rows — e.g. same-day audit-trail ingestion lag), the verdict
    must NOT be flagged degraded (no misleading 'INSUFFICIENT DATA, re-run' note), and a
    data_coverage note must explain the unavailable corroboration. This is the SUBJ06 case.
    """
    logs = _fraud_logs()
    logs["record_lake"][0]["element_counters.AUX"] = 1  # not bare → decisive FAIL
    logs["settlement_report"] = []  # audit trail not yet ingested
    v = evaluate_verdict(_scheme_spec(), logs, _Analysis())
    s = v.subjects[0]
    assert s.verdict == "FALSE POSITIVE"
    assert v.degraded is False  # terminal FP is not weakened by missing corroboration
    # the settlement-dependent decisive checks are UNKNOWN but moot
    assert next(c for c in s.checks if c.id == "single_actor").result == "unknown"
    # a transparency note explains the missing corroboration without alarming
    assert any(n.startswith("data_coverage=") for n in s.notes)


def test_evaluate_verdict_false_positive_late_issuance():
    logs = _fraud_logs()
    logs["settlement_report"][0]["transactionDateTime"] = "2026-07-24T13:00:00"  # +4h
    v = evaluate_verdict(_scheme_spec(), logs, _Analysis())
    assert v.subjects[0].verdict == "FALSE POSITIVE"
    assert (
        next(c for c in v.subjects[0].checks if c.id == "immediate_issuance").result
        == "fail"
    )


def test_evaluate_verdict_false_positive_void_and_multi_actor():
    logs = _fraud_logs()
    logs["settlement_report"].append(
        {
            "relatedRecordLocator": "SUBJ03",
            "retrieverUserSign": "0007VP",
            "transactionDateTime": "2026-07-24T09:25:00",
            "recordType": "REFUND",
        }
    )
    v = evaluate_verdict(_scheme_spec(), logs, _Analysis())
    s = v.subjects[0]
    assert s.verdict == "FALSE POSITIVE"
    assert next(c for c in s.checks if c.id == "single_actor").result == "fail"
    assert next(c for c in s.checks if c.id == "no_void_refund").result == "fail"
    # multi-actor note surfaced
    assert any("Multiple acting signs" in n for n in s.notes)


def test_containment_is_withheld_on_a_cleared_verdict():
    """Containment is a §4 step, reachable only from a fraud verdict.

    `lock_target` is resolved from the record creator BEFORE any verdict exists, so without
    a gate a cleared order travels downstream carrying a named org_unit+sign and every
    reader has to remember to re-check the label. The gate sits with the verdict, once.
    """
    logs = _fraud_logs()
    logs["record_lake"][0]["element_counters.AUX"] = 3  # not bare → decisive FAIL
    s = evaluate_verdict(_scheme_spec(), logs, _Analysis()).subjects[0]
    assert s.verdict == "FALSE POSITIVE"
    assert s.lock_target == {}
    # ...and the withholding is RECORDED, not silent: the creator is still evidence.
    note = next(n for n in s.notes if n.startswith("containment_withheld="))
    assert "0201GPSU" in note and "ORG2428D4" in note
    assert "not nominated for action" in note


def test_containment_is_withheld_on_insufficient_data():
    """ "We do not know yet" is not grounds to suspend somebody."""
    logs = {"record_lake": _fraud_logs()["record_lake"], "settlement_report": []}
    s = evaluate_verdict(_scheme_spec(), logs, _Analysis()).subjects[0]
    assert s.verdict == "INSUFFICIENT DATA"
    assert s.lock_target == {}


def test_containment_labels_are_pack_declared_not_hardcoded():
    """The engine stays generic: WHICH verdicts warrant containment is a pack decision."""
    # default (no declaration) → the fraud label only, and it keeps its target
    fraud = evaluate_verdict(_scheme_spec(), _fraud_logs(), _Analysis()).subjects[0]
    assert fraud.verdict == "VALID FRAUD"
    assert fraud.lock_target.get("identity") == "0201GPSU"
    assert not any(n.startswith("containment_withheld=") for n in fraud.notes)

    # a ruleset that declares the FP label as containment-worthy is honoured
    spec = _scheme_spec()
    spec["containment_labels"] = ["FALSE POSITIVE"]
    logs = _fraud_logs()
    logs["record_lake"][0]["element_counters.AUX"] = 3
    s = evaluate_verdict(spec, logs, _Analysis()).subjects[0]
    assert s.verdict == "FALSE POSITIVE"
    assert s.lock_target.get("identity") == "0201GPSU"
    # ...and conversely the fraud label now falls outside the declared set
    fraud2 = evaluate_verdict(spec, _fraud_logs(), _Analysis()).subjects[0]
    assert fraud2.verdict == "VALID FRAUD"
    assert fraud2.lock_target == {}


def _lake_issuance_spec():
    """The SCHEME ruleset with `immediate_issuance` reading the LAKE, as it ships now.

    The issuance instant is not a column: in a versioned record store it is the write time
    of the FIRST version whose document shows an issued document. A side's `where` clause is
    what lets a ruleset say that, instead of the engine hard-coding one backend's shape.
    """
    spec = _scheme_spec()
    for cond in spec["conditions"]:
        if cond["id"] == "immediate_issuance":
            cond["end"] = {
                "source": "record",
                "field": "modification_date_time",
                "where": [
                    {
                        "field": "pricing.asset_document.status",
                        "any_of": ["T"],
                        "match": "exact",
                    }
                ],
            }
    return spec


def _version_logs(issued_at="2026-07-24T09:02:34"):
    """One record as a SERIES OF ENVELOPES: only the later one carries an issued document."""
    base = _fraud_logs()["record_lake"][0]
    creation = dict(
        base, modification_date_time="2026-07-24T09:00:14", version_number="0"
    )
    issued = dict(
        base,
        modification_date_time=issued_at,
        version_number="6",
        **{"pricing.asset_document.status": "T"},
    )
    return {"record_lake": [creation, issued], "settlement_report": []}


def test_immediate_issuance_reads_the_versioned_records_own_write_time():
    """The gap is creation -> the FIRST version showing a document (SUBJ01: 2m34s → FRAUD).

    The un-`where`d version (09:00:14, no document) must NOT be taken as the issuance
    instant, and the audit_trail access trail is not consulted at all (settlement_report is empty).
    """
    spec = _lake_issuance_spec()
    v = evaluate_verdict(spec, _version_logs(), _Analysis())
    check = next(c for c in v.subjects[0].checks if c.id == "immediate_issuance")
    assert check.result == "pass"
    assert "0:02:34" in check.observed


def test_immediate_issuance_fails_when_the_issuance_version_is_late():
    v = evaluate_verdict(
        _lake_issuance_spec(), _version_logs("2026-07-24T13:00:00"), _Analysis()
    )
    check = next(c for c in v.subjects[0].checks if c.id == "immediate_issuance")
    assert check.result == "fail"


def test_immediate_issuance_unknown_when_no_version_shows_a_document():
    """No version carries an issued document -> the issuance time is genuinely UNKNOWN.

    Without the `where` filter the check would silently read SOME version's write time and
    report a confident gap for an event that never happened."""
    logs = _version_logs()
    logs["record_lake"] = [logs["record_lake"][0]]  # creation version only
    v = evaluate_verdict(_lake_issuance_spec(), logs, _Analysis())
    check = next(c for c in v.subjects[0].checks if c.id == "immediate_issuance")
    assert check.result == "unknown"


def test_side_where_is_exact_by_default_for_single_letter_status():
    """`match: exact` must not let 'T' match a value that merely CONTAINS a T.

    The lake's status vocabulary is single letters (T/V/I), so substring matching would
    read a voided document as issued."""
    from correlation import _side_rows

    rows = [{"status": "TRANSFERRED"}, {"status": "T"}]
    side = {"source": "s", "where": [{"field": "status", "any_of": ["T"]}]}
    assert _side_rows({"s": rows}, side) == [{"status": "T"}]
    loose = {
        "source": "s",
        "where": [{"field": "status", "any_of": ["T"], "match": "substring"}],
    }
    assert len(_side_rows({"s": rows}, loose)) == 2


def test_the_row_selector_is_ONE_implementation_shared_with_a_side():
    """`apply_where` is public because a second consumer speaks the same vocabulary.

    A follow-up pass's harvest scopes WHICH rows it reads its values out of, and a pack author
    who has written a condition's `where` must not have to learn a second spelling for the
    same idea — two spellings of one predicate become two behaviours the moment either is
    fixed. So a side's selector IS this function, asserted by running both over the same rows
    rather than by reading the code.

    The skip on an incomplete clause is pinned here too, and it is not tidiness: a clause that
    emptied the row set would look downstream exactly like a source that returned nothing,
    which is the one outcome this whole layer exists to keep distinguishable. Its cost — the
    harvest then reads every row — is why `pack_validate` reports such a clause as an error.
    """
    from correlation import _side_rows, apply_where

    rows = [{"status": "T", "id": 1}, {"status": "V", "id": 2}]
    clauses = [{"field": "status", "any_of": ["T"], "match": "exact"}]
    assert apply_where(rows, clauses) == [rows[0]]
    assert _side_rows({"s": rows}, {"source": "s", "where": clauses}) == apply_where(
        rows, clauses
    )
    # Absent, empty, and unusable clauses all mean EVERY row — never zero.
    assert apply_where(rows, None) == rows
    assert apply_where(rows, []) == rows
    assert apply_where(rows, [{"field": "status"}]) == rows
    assert apply_where(rows, [{"any_of": ["T"]}]) == rows
    assert apply_where(rows, ["not-a-mapping"]) == rows
    # Clauses AND together.
    assert apply_where(rows, clauses + [{"field": "id", "any_of": ["2"]}]) == []


def test_evaluate_verdict_out_of_scope_route():
    logs = _fraud_logs()
    logs["record_lake"][0]["route.air.board_point"] = "JFK"
    logs["record_lake"][0]["route.air.off_point"] = "LAX"
    v = evaluate_verdict(_scheme_spec(), logs, _Analysis())
    assert v.subjects[0].verdict == "FALSE POSITIVE"
    assert (
        next(c for c in v.subjects[0].checks if c.id == "route_in_scope").result
        == "fail"
    )


def test_the_scope_gate_rewrites_the_label_AND_its_class():
    """`out_of_scope` replaces a label the rollup already assigned — the class moves too.

    Leaving the earlier branch's class behind would leave the two fields contradicting
    each other, and a consumer reading the class (the confidence clamp, the framing)
    would act on an adjudication this branch just withdrew. Same defect shape as the
    containment target that survived a cleared verdict.
    """
    spec = _scheme_spec()
    spec["labels"]["out_of_scope"] = "OUT OF SCOPE"
    for c in spec["conditions"]:
        if c["id"] == "route_in_scope":
            c["gate"] = "scope"
    logs = _fraud_logs()
    logs["record_lake"][0]["route.air.board_point"] = "JFK"
    logs["record_lake"][0]["route.air.off_point"] = "LAX"
    s = evaluate_verdict(spec, logs, _Analysis()).subjects[0]
    # Not "FALSE POSITIVE": the ruleset declined to adjudicate, it did not clear anybody.
    assert (s.verdict, s.verdict_class) == ("OUT OF SCOPE", "out_of_scope")


def _unknown_gate_subject(decisive):
    """One subject whose scope gate is `unknown` because its points never arrived."""
    spec = _scheme_spec()
    spec["labels"]["out_of_scope"] = "OUT OF SCOPE"
    for c in spec["conditions"]:
        if c["id"] == "route_in_scope":
            c["gate"] = "scope"
            c["decisive"] = decisive
    logs = _fraud_logs()
    logs["record_lake"][0].pop("route.air.board_point")
    logs["record_lake"][0].pop("route.air.off_point")
    s = evaluate_verdict(spec, logs, _Analysis()).subjects[0]
    assert next(c for c in s.checks if c.id == "route_in_scope").result == "unknown"
    return s, next(n for n in s.notes if n.startswith("scope_gate="))


def test_an_unknown_DECISIVE_scope_gate_says_no_verdict_was_reached_at_all():
    """A gate the ruleset declared `decisive` IS the verdict when it cannot be established.

    `has_decisive_unknown` routes such a subject to `insufficient`, so the headline says
    nothing was adjudicated — while the note used to tell the operator the remaining
    conditions had carried the case. That is not a caveat, it is the opposite of what
    happened, and the two halves of one report contradicting each other is worse than
    either half alone: the reader who trusts the note closes a case the headline says
    was never opened.
    """
    s, note = _unknown_gate_subject(decisive=True)
    assert (s.verdict, s.verdict_class) == ("INSUFFICIENT DATA", "insufficient")
    assert note == (
        "scope_gate=Scope could not be established (In-scope route), and this "
        "procedure cannot be concluded without it, so no verdict is reached on the merits."
    )
    assert "rests on the other conditions only" not in note


def test_an_unknown_NON_decisive_scope_gate_keeps_the_historical_caveat_verbatim():
    """The other reading is legitimate, so it must survive the fix byte-identically.

    A gate the ruleset did NOT declare decisive really is a caveat: the rollup reached a
    verdict from the other conditions and the note qualifies it. Both directions are pinned
    separately because a fix that only added the new sentence — or only replaced the old one
    — would be indistinguishable from one that reports the gate's decisiveness, and every
    existing ruleset's gate is the non-decisive kind.
    """
    s, note = _unknown_gate_subject(decisive=False)
    assert (s.verdict, s.verdict_class) == ("VALID FRAUD", "fraud")
    assert note == (
        "scope_gate=Scope could not be established (In-scope route), so the verdict "
        "below rests on the other conditions only."
    )
    assert "no verdict is reached on the merits" not in note


def test_a_total_blackout_is_INSUFFICIENT_not_a_reject_to_the_detectors_owner():
    """No rows for any source is a retrieval failure, not a routing decision.

    The `reject` exit means "this alert is unactionable by this procedure" — a claim about
    the detector, not about the data. A blackout usually means credential or scope issues on
    our side, so firing it here discards a case nobody has read. The per-subject rollup can
    only emit fraud/false_positive/insufficient/out_of_scope, so the reject guard is
    unreachable as written regardless.
    """
    spec = _scheme_spec()
    spec["labels"]["reject"] = "REJECT TO OWNER"
    spec["reject_when"] = "no_subject_validates"
    barren = {"record_lake": [{"locator.red": "SUBJ03"}], "settlement_report": []}
    for s in evaluate_verdict(spec, barren, _Analysis()).subjects:
        assert (s.verdict, s.verdict_class) == ("INSUFFICIENT DATA", "insufficient")
        assert not any(n.startswith("reject_reason=") for n in s.notes)


def test_evaluate_verdict_insufficient_when_settlement_missing():
    logs = {"record_lake": _fraud_logs()["record_lake"], "settlement_report": []}
    v = evaluate_verdict(_scheme_spec(), logs, _Analysis())
    s = v.subjects[0]
    assert s.verdict == "INSUFFICIENT DATA"
    assert v.degraded is True
    # the settlement-dependent decisive checks are unknown
    assert next(c for c in s.checks if c.id == "single_actor").result == "unknown"


def test_evaluate_verdict_split_fails_no_split():
    logs = _fraud_logs()
    logs["record_lake"][0]["xref.sp"] = ["SP1"]  # split present
    v = evaluate_verdict(_scheme_spec(), logs, _Analysis())
    # no_split is non-decisive, so a split alone stays FRAUD but the check fails
    assert next(c for c in v.subjects[0].checks if c.id == "no_split").result == "fail"


def test_element_absence_cannot_claim_absence_for_a_path_it_never_read():
    """A partial-coverage PASS turns a retrieval gap into evidence.

    `element_absence` is used for the SCHEME EXCLUSION checks, where PASS is the
    fraud-CONSISTENT answer ("the record really is bare"). It used to need only ONE
    declared path to resolve before reporting PASS on the rest, so a projection that
    dropped `service.osi` produced "bare" from the counters that happened to arrive —
    an unretrieved element read as a confirmed absent one. §3.2 requires the opposite:
    unretrievable ⇒ NOT EVALUATED.
    """
    from src.correlation import _eval_condition

    cond = {
        "id": "bare",
        "label": "Bare",
        "kind": "element_absence",
        "source": "record",
        "counters": ["element_counters.AUX"],
        "arrays": ["remark.rm", "service.osi"],
        "decisive": True,
    }
    # `service.osi` was never projected; everything that DID arrive is empty.
    c = _eval_condition(
        cond, {"record": [{"element_counters": {"AUX": 0}, "remark": {"rm": []}}]}
    )
    assert c.result == "unknown", c.detail
    assert "service.osi" in c.detail, c.detail
    # Full coverage with everything empty is a genuine PASS — the gap check must not
    # swallow the real answer.
    c = _eval_condition(
        cond,
        {
            "record": [
                {
                    "element_counters": {"AUX": 0},
                    "remark": {"rm": []},
                    "service": {"osi": []},
                }
            ]
        },
    )
    assert c.result == "pass", c.detail
    # A present element still FAILS decisively despite the gap: finding one is enough, and
    # re-checking the unread paths could only find more.
    c = _eval_condition(
        cond, {"record": [{"element_counters": {"AUX": 7}, "remark": {"rm": []}}]}
    )
    assert c.result == "fail", c.detail


# The real `security.es` element as the lake returns it (live 2026-07-31, SUBJ04): the
# whole struct arrives as a JSON STRING, and the same element is repeated on every later
# version row (44 of them on this record).
_ES_ELEMENT = (
    '[{"element_id":"0-record-ES-61","version_number":"7","element_status":"ACT",'
    '"last_updator_agent_id":{"red":"GGSU","orange":"GGSU","green":"D1_VAW=="},'
    '"last_update_date":"2026-07-09","last_updator_org_unit_id":"ORG262206",'
    '"receivers":[{"access_right":"B","receiver":"MMM1N14AB"}],'
    '"receiver_type":"G","activity":"A"}]'
)


def _es_cond():
    """`no_es` as the pack ships it — an element_absence that QUOTES what it finds."""
    return {
        "id": "no_es",
        "label": "No ES element (extended-security access grant)",
        "kind": "element_absence",
        "source": "record",
        "arrays": ["security.es"],
        "decisive": True,
        "decisive_on": ["fail"],
        "expected_label": "no ES access grant on the record",
        "fail_detail": "an extended-security element grants another org_unit access",
        "pass_detail": "no extended-security access grant on this record",
        "quote_template": "ES/{grant_type} {granted_day}/{actor}/{org_unit} {receiver}-{access}",
        "quote_fields": {
            "grant_type": ["receiver_type"],
            "granted": ["last_update_date"],
            "actor": ["last_updator_agent_id.red"],
            "org_unit": ["last_updator_org_unit_id"],
            "receiver": ["receivers.receiver"],
            "access": ["receivers.access_right"],
        },
    }


def test_a_found_element_is_quoted_in_the_operators_own_notation():
    """§3.2 asks for the element VERBATIM, including the org_unit reference it grants.

    A count (`security.es×1`) proves the check ran but lets nobody verify WHAT was found,
    and the reviewer's own resolution of 83e94dd6 cites the element as
    `ES/G 09JUL/GGSU/ORG262206 MMM1N14AB-B`. So the engine reproduces that notation from
    the element's leaves; the count stays alongside as the corroborating arithmetic.
    """
    from src.correlation import _eval_condition

    c = _eval_condition(_es_cond(), {"record": [{"security_es": _ES_ELEMENT}]})
    assert c.result == "fail"
    assert "ES/G 09JUL/GGSU/ORG262206 MMM1N14AB-B" in c.observed, c.observed
    # The count is still there — a quote is the evidence, the count is how much of it.
    assert "security.es×1" in c.observed, c.observed
    # And the row says what was REQUIRED in the pack's words, not the bare check's.
    assert c.expected == "no ES access grant on the record"
    assert "grants another org_unit access" in c.detail


def test_the_same_element_on_many_versions_is_quoted_once():
    """A versioned backend repeats an element on every later row — quote the ELEMENT.

    `security.es` arrives on 44 of SUBJ04's versions; quoting per row would print 44
    identical lines and make one grant read as a pattern of them.
    """
    from src.correlation import _eval_condition

    rows = [
        {"version_number": str(n), "security_es": _ES_ELEMENT} for n in range(7, 51)
    ]
    c = _eval_condition(_es_cond(), {"record": rows})
    assert c.observed.count("ES/G 09JUL") == 1, c.observed


def test_an_element_quote_is_never_a_reason_the_check_fired():
    """Quoting is presentational: it must not change pass/fail/unknown or decisiveness.

    Same rows, with and without the pack's quote declaration — only the observed STRING
    may differ, because a report's notation is not allowed to move a verdict.
    """
    from src.correlation import _eval_condition

    quoted, plain = _es_cond(), _es_cond()
    plain.pop("quote_template"), plain.pop("quote_fields")
    rows = {"record": [{"security_es": _ES_ELEMENT}]}
    a, b = _eval_condition(quoted, rows), _eval_condition(plain, rows)
    assert (a.result, a.decisive) == (b.result, b.decisive) == ("fail", True)
    assert "ES/G" in a.observed and "ES/G" not in b.observed


def test_a_missing_leaf_degrades_a_quote_instead_of_breaking_the_check():
    """A pack template naming a leaf this element does not carry must not raise.

    `_eval_condition` runs inside the verdict loop: an exception here is a lost condition,
    and a lost DECISIVE condition is a changed verdict. So an unresolvable placeholder
    renders empty and the surrounding notation closes up.
    """
    from src.correlation import _eval_condition

    cond = _es_cond()
    cond["quote_fields"]["org_unit"] = ["no.such.leaf"]
    cond["quote_template"] += " {unknown_placeholder}"
    c = _eval_condition(cond, {"record": [{"security_es": _ES_ELEMENT}]})
    assert c.result == "fail"
    assert "ORG262206" not in c.observed
    # No double spaces left behind where the missing leaves were.
    assert "  " not in c.observed, repr(c.observed)
    assert "MMM1N14AB-B" in c.observed


def test_element_absence_without_the_element_is_a_clean_pass_not_a_quote():
    """The record that carries no grant passes, in the pack's words, with nothing quoted."""
    from src.correlation import _eval_condition

    c = _eval_condition(_es_cond(), {"record": [{"security_es": []}]})
    assert c.result == "pass"
    assert c.expected == "no ES access grant on the record"
    assert c.detail == "no extended-security access grant on this record"
    assert "ES/" not in c.observed


def test_an_unretrieved_es_struct_is_not_evaluated_rather_than_absent():
    """The stub this replaced existed for this case: a gap must not read as a clearance.

    Before `security.es` was projected the column simply was not there. That is NOT
    "no grant on this record" — it is a condition §3.2 requires reported as NOT EVALUATED,
    and it must not be decisive either way.
    """
    from src.correlation import _eval_condition

    c = _eval_condition(_es_cond(), {"record": [{"locator_red": "SUBJ04"}]})
    assert c.result == "unknown"
    assert c.decisive is False  # decisive_on: [fail] only
    assert c.expected == "no ES access grant on the record"


def test_count_array_leaves_distinguishes_an_empty_array_from_an_absent_one():
    """0 and None are different answers, and the lake shape makes it easy to conflate them.

    An array that arrived EMPTY means "no such elements" (a real 0); a path that never
    arrived means "we did not look". Counting leaves alone cannot tell them apart — an
    empty list contributes no leaves either way — and the real rows return the whole struct
    as a JSON STRING with every element array explicitly null.
    """
    from src.correlation import _count_array_leaves

    assert _count_array_leaves([{"remark": {"rm": []}}], "remark.rm") == 0
    assert _count_array_leaves([{"remark": {"rm": None}}], "remark.rm") == 0
    # The real record_table shape: `remark` selected whole, returned as a JSON string.
    assert _count_array_leaves([{"remark": '{"rm":null,"ry":null}'}], "remark.ry") == 0
    assert (
        _count_array_leaves([{"remark": {"rm": [{"a": 1}, {"a": 2}]}}], "remark.rm")
        == 2
    )
    # Never retrieved -> None, which is what lets a caller report NOT EVALUATED.
    assert _count_array_leaves([{"element_counters": {"AUX": 0}}], "remark.rm") is None
    assert _count_array_leaves([{"remark": '{"rm":null}'}], "service.osi") is None


def test_distinct_count_falls_back_to_record_updators_when_settlement_empty():
    """single_actor must survive an empty Settlement Report by reading the record lake's updator
    chain — a order touched by many signs before issuance fails the single-actor check
    off record_table alone (the expert's SUBJ06 multi-actor FALSE-POSITIVE reasoning).
    """
    from src.correlation import _eval_condition

    cond = {
        "id": "single_actor",
        "label": "Single actor",
        "kind": "distinct_count",
        "source": "settlement",
        "field": "retrieverUserSign",
        "fallbacks": [{"source": "record", "field": "last_updator.sign.red"}],
        "max": 1,
        "decisive": True,
    }
    # Settlement empty; record version rows carry several distinct updator signs.
    src_rows = {
        "settlement": [],
        "record": [
            {"last_updator.sign.red": "6002BBSU"},
            {"last_updator.sign.red": "9107RKAS"},
            {"last_updator.sign.red": "6003CCSU"},
        ],
    }
    c = _eval_condition(cond, src_rows)
    assert c.result == "fail"  # >1 distinct acting sign
    assert "from record.last_updator.sign.red" in c.observed
    # And when the settlement sign IS present, the primary source wins (no fallback used).
    src_rows2 = {
        "settlement": [{"retrieverUserSign": "6003CC"}],
        "record": [
            {"last_updator.sign.red": "6002BBSU"},
            {"last_updator.sign.red": "9107RKAS"},
        ],
    }
    c2 = _eval_condition(cond, src_rows2)
    assert c2.result == "pass"  # single settlement sign
    assert "from settlement.retrieverUserSign" in c2.observed


def test_a_counting_condition_with_no_declared_bound_invents_none():
    """An undeclared bound is no comparison, not a small one.

    Three condition kinds substitute a literal when `max` is absent, printing that invented
    number as the procedure's own threshold. A declared-but-unparseable bound (`time_gap`
    accepts only `<N>h`/`<N>d`/`<N>m`) lands in the same state as an absent one. Both
    installed packs declare every bound; this pins behaviour the latent defect can still reach.
    """
    from src.correlation import _eval_condition

    rows = {
        "orders": [
            {"handler": "AAA1", "locator": "L1"},
            {"handler": "BBB2", "locator": "L2"},
            {"handler": "CCC3", "locator": "L3"},
            {"handler": "DDD4", "locator": "L4"},
        ]
    }

    # distinct_count: four distinct handlers, and nothing to compare four against.
    unbounded = {
        "id": "handler_spread",
        "label": "One handler only",
        "kind": "distinct_count",
        "source": "orders",
        "field": "handler",
        "decisive": True,
    }
    c = _eval_condition(unbounded, rows)
    assert c.result == "unknown"
    assert "no bound declared" in c.expected
    # And the number is NOT reported as a finding against an invented limit.
    assert "allowed" not in c.observed
    assert "4" not in c.observed

    # The same condition with the bound declared still works exactly as before.
    bounded = dict(unbounded, max=1)
    assert _eval_condition(bounded, rows).result == "fail"
    assert _eval_condition(dict(unbounded, max=9), rows).result == "pass"
    # Zero is a declared bound, not an absent one — `max: 0` is the commonest value in the
    # installed packs and must never be read as "unset".
    assert _eval_condition(dict(unbounded, max=0), rows).result == "fail"

    # velocity_count: one actor over four locators, and no declared burst rate.
    vel = {
        "id": "handler_burst",
        "label": "No burst",
        "kind": "velocity_count",
        "source": "orders",
        "subject_field": "locator",
        "actor_field": "handler",
    }
    burst_rows = {"orders": [{"handler": "AAA1", "locator": f"L{i}"} for i in range(5)]}
    v = _eval_condition(vel, burst_rows)
    assert v.result == "unknown"
    assert "no bound declared" in v.expected
    assert _eval_condition(dict(vel, max=2), burst_rows).result == "fail"
    assert _eval_condition(dict(vel, max=9), burst_rows).result == "pass"

    # time_gap: an absent window and an unparseable one both refuse to adjudicate.
    gap_rows = {
        "orders": [{"opened": "2026-03-01T00:00:00Z", "closed": "2026-03-01T06:00:00Z"}]
    }
    gap = {
        "id": "closed_promptly",
        "label": "Closed promptly",
        "kind": "time_gap",
        "start": {"source": "orders", "field": "opened"},
        "end": {"source": "orders", "field": "closed"},
    }
    for absent in ({}, {"max": ""}, {"max": "1 week"}, {"max": "PT48H"}):
        g = _eval_condition(dict(gap, **absent), gap_rows)
        assert g.result == "unknown", absent
        assert "no bound declared" in g.expected, absent
    # A six-hour gap against a declared hour fails; against a declared day it passes.
    assert _eval_condition(dict(gap, max="1h"), gap_rows).result == "fail"
    assert _eval_condition(dict(gap, max="1d"), gap_rows).result == "pass"


def test_distinct_count_renders_the_original_values_not_the_normalised_ones():
    """Normalise to COMPARE, render the ORIGINAL. Dedup and display are two different jobs.

    `_norm_identifier` strips punctuation so two spellings of one identifier dedup to one
    value — right for the count, and the counted token was also what got PRINTED. A live run
    reported `40 distinct: ['10203212354826', ...]` for a set of `host:port` addresses, and
    the containment block repeated it: an operator cannot act on a value with the dots and
    the colon removed, and cannot even tell what kind of thing it is.
    """
    from src.correlation import _eval_condition

    cond = {
        "id": "address_count",
        "label": "One client address",
        "kind": "distinct_count",
        "source": "auth",
        "field": "addr",
        "max": 1,
    }
    src_rows = {
        "auth": [
            {"addr": "10.1.1.10:54826"},
            {"addr": "10.2.2.20:1080"},
        ]
    }
    c = _eval_condition(cond, src_rows)
    assert c.result == "fail"
    assert "2 distinct" in c.observed
    # The real spellings, punctuation intact.
    assert "10.1.1.10:54826" in c.observed
    assert "10.2.2.20:1080" in c.observed
    assert "10203212354826" not in c.observed

    # DEDUP IS STILL NORMALISED: two spellings of one identifier stay ONE value, and the
    # rendered survivor is a real spelling rather than the stripped token.
    same = {"auth": [{"addr": "6001AA-SU"}, {"addr": "6001AASU"}]}
    c2 = _eval_condition(cond, same)
    assert c2.result == "pass", c2.observed
    assert "1 distinct" in c2.observed
    assert "6001AA" in c2.observed and "6001AASU" in c2.observed.replace("-", "")


def test_count_array_leaves_is_per_record_max_not_cross_row_sum():
    """record_table returns one row per version snapshot for the same record; each row carries the
    whole array. _count_array_leaves must return the per-row MAX, not the sum across rows
    (which produced the spurious remark.rm×32 on a record with ~1 real remark)."""
    from src.correlation import _count_array_leaves

    rows = [
        {"remark.rm": ["EML-ITI-SENT"]} for _ in range(24)
    ]  # same 1 remark, 24 rows
    assert _count_array_leaves(rows, "remark.rm") == 1  # not 24
    # Absent path still returns None (distinguishes "0" from "not projected").
    assert _count_array_leaves([{"other": 1}], "remark.rm") is None
    # A row with more elements sets the max.
    rows2 = [{"xref.sp": ["A"]}, {"xref.sp": ["A", "B"]}]
    assert _count_array_leaves(rows2, "xref.sp") == 2


def test_degraded_reflects_only_decisive_unknowns():
    """A non-decisive check coming back UNKNOWN must NOT mark the verdict degraded —
    degraded means a DECISIVE check couldn't be evaluated (→ INSUFFICIENT-leaning)."""
    spec = _scheme_spec()
    # Add a non-decisive check on a source that isn't present → it evaluates UNKNOWN.
    spec["sources"]["auth"] = "auth_events"
    spec["conditions"].append(
        {
            "id": "not_automated",
            "label": "Not automated",
            "kind": "field_flag",
            "source": "auth",
            "flag_fields": ["robot_flag"],
            "expected": False,
            "decisive": False,
        }
    )
    v = evaluate_verdict(spec, _fraud_logs(), _Analysis())  # auth_events absent
    s = v.subjects[0]
    assert s.verdict == "VALID FRAUD"  # decisive checks all pass
    assert next(c for c in s.checks if c.id == "not_automated").result == "unknown"
    assert v.degraded is False  # non-decisive unknown does not degrade


def test_subject_scope_false_evaluates_all_rows():
    """A condition with subject_scope: false reads ALL rows of its source, not the
    subset filtered to the subject record — needed for AUTH_SVC auth / automated checks whose
    rows are keyed by acting identity, not the record."""
    from src.correlation import _condition_sources

    spec = _scheme_spec()
    spec["sources"]["auth"] = "auth_events"
    spec["conditions"].append(
        {
            "id": "not_automated",
            "label": "Not automated",
            "kind": "field_flag",
            "source": "auth",
            "flag_fields": ["robot_flag"],
            "expected": False,
            "decisive": False,
            "subject_scope": False,
        }
    )
    logs = _fraud_logs()
    # An auth row that does NOT contain the record value anywhere (keyed by login/org_unit).
    logs["auth_events"] = [
        {"login": "USERNAMEX", "org_unit": "ORG2428D4", "robot_flag": False}
    ]
    v = evaluate_verdict(spec, logs, _Analysis())
    check = next(c for c in v.subjects[0].checks if c.id == "not_automated")
    # Without subject_scope:false this would be UNKNOWN (record filter drops the row);
    # with it, the row is seen and robot_flag=false → PASS (not automated).
    assert check.result == "pass"
    assert _condition_sources(spec["conditions"][-1]) == ["auth"]


# --- IR10000002: the automated exclusion must be able to decide the verdict -----
# The automation check should be decisive but three independent flaws each prevent it.
# One test below covers each.


def _automation_spec():
    """_scheme_spec() plus the pair-matched, fail-decisive automated reference check as shipped."""
    spec = _scheme_spec()
    spec["sources"]["automation"] = "automation_registry"
    spec["conditions"].append(
        {
            "id": "not_automated_ref",
            "label": "Not a known AUTOMATED account",
            "kind": "record_absence",
            "source": "automation",
            "subject_scope": False,
            "row_match": [
                {
                    "label": "org_unit",
                    "from_entity": "org_unit",
                    "fields": ["orgUnitId"],
                },
                {
                    "label": "sign",
                    "from_entity": "user",
                    "fields": ["sign"],
                    "normalize": "identifier",
                },
            ],
            "fields": ["profile"],
            "forbidden_values": ["AUTOMATED"],
            "match": "exact",
            "decisive": True,
            "decisive_on": ["fail"],
        }
    )
    return spec


class _AutomationAnalysis:
    """The IR10000002 identity: alerted org_unit + the acting WS sign."""

    def __init__(self, org_unit="QQQ1R17GH", sign="6009JJ", record="SUBJ03"):
        self.extracted_entities = [
            ExtractedEntity(type="record", value=record),
            ExtractedEntity(type="org_unit", value=org_unit),
            ExtractedEntity(type="user", value=sign),
        ]


def test_automated_pair_match_drives_false_positive():
    """A confirmed AUTOMATED (org_unit, sign) pair is a §3.2 exclusion -> FALSE POSITIVE.

    Previously `not_automated_ref` was non-decisive, so the engine reported the FAIL and then
    ignored it: the verdict came back INSUFFICIENT DATA while the human expert closed the
    case on exactly this evidence."""
    logs = _fraud_logs()
    logs["automation_registry"] = [
        {"orgUnitId": "QQQ1R17GH", "sign": "6009JJ", "profile": "AUTOMATED"}
    ]
    v = evaluate_verdict(_automation_spec(), logs, _AutomationAnalysis())
    s = v.subjects[0]
    assert next(c for c in s.checks if c.id == "not_automated_ref").result == "fail"
    assert s.verdict == "FALSE POSITIVE"


def test_automated_other_org_units_rows_do_not_convict():
    """The (org_unit, sign) PAIR must match — a sign shared by other org_units must not hit.

    Sign 6009JJ is the standard web-service sign in 89,937 org_units (measured), so the
    retriever's OR-ed filter returns thousands of OTHER org_units' robots. Matching on
    `orgUnitId OR sign` "found" the alerted identity in a list it was absent from."""
    logs = _fraud_logs()
    # Same sign, different org_units — the alerted QQQ1R17GH is NOT among them.
    logs["automation_registry"] = [
        {"orgUnitId": "DDD1E05JK", "sign": "6009JJ", "profile": "AUTOMATED"},
        {"orgUnitId": "EEE1F06LM", "sign": "6009JJ", "profile": "AUTOMATED"},
    ]
    v = evaluate_verdict(_automation_spec(), logs, _AutomationAnalysis())
    check = next(c for c in v.subjects[0].checks if c.id == "not_automated_ref")
    assert check.result == "pass"  # this identity is not on the list


def test_a_listed_automation_does_not_clear_its_unlisted_co_subject():
    """One incident, two acting identities: the reference row belongs to one of them.

    `row_match` was filling its clauses from every entity of the named type, so one subject's
    registered row satisfied the lookup for its unregistered sibling. The narrowing can only
    remove a match (a wrong fail becomes a pass; a pass never becomes a fail), so the
    automation's verdict below is the other half of the assertion: the fix must not buy the
    human's adjudication at the automation's expense.
    """
    spec = _automation_spec()
    # Subject on the ACTING IDENTITY, as a per-identity ruleset does, and take the exclusion
    # at its shipped strength — `categorical` is what makes a wrong clear unappealable.
    spec["subject_entity"] = "user"
    for cond in spec["conditions"]:
        if cond["id"] == "not_automated_ref":
            cond["exclusion_kind"] = "categorical"

    class _MixedAnalysis:
        """One alert, one registered automation and one human sign acting under it."""

        extracted_entities = [
            ExtractedEntity(type="record", value="SUBJ03"),
            ExtractedEntity(type="org_unit", value="QQQ1R17GH"),
            ExtractedEntity(type="user", value="6009JJ"),
            ExtractedEntity(type="user", value="0042AB"),
        ]

    logs = _fraud_logs()
    # The register holds the automation's pair ONLY. The human's sign is absent from it.
    logs["automation_registry"] = [
        {"orgUnitId": "QQQ1R17GH", "sign": "6009JJ", "profile": "AUTOMATED"}
    ]
    v = evaluate_verdict(spec, logs, _MixedAnalysis())
    by_value = {s.subject_value: s for s in v.subjects}
    assert set(by_value) == {"6009JJ", "0042AB"}, sorted(by_value)

    # The automation: still cleared, on its own row.
    robot = by_value["6009JJ"]
    assert next(c for c in robot.checks if c.id == "not_automated_ref").result == "fail"
    assert robot.verdict_class == "false_positive"
    assert any(n.startswith("categorical_exclusion=") for n in robot.notes), robot.notes

    # The human: adjudicated on its own merit, and NOT on its sibling's registration.
    human = by_value["0042AB"]
    check = next(c for c in human.checks if c.id == "not_automated_ref")
    assert check.result == "pass", check.detail
    assert human.verdict_class != "false_positive", human.verdict_class
    assert not [
        n
        for n in human.notes
        if n.startswith(("categorical_exclusion=", "decisive_exclusion="))
    ], human.notes


def test_automated_absence_is_unknown_when_source_truncated():
    """Absence from a TRUNCATED source is no evidence: it must read `unknown`, not `pass`.

    automation_registry returned exactly its 500-row cap out of 89,937 real rows, so the
    alerted pair could sit in the rows never returned."""
    logs = _fraud_logs()
    logs["automation_registry"] = [
        {"orgUnitId": f"OFF{i:05d}", "sign": "6009JJ", "profile": "AUTOMATED"}
        for i in range(500)
    ]
    v = evaluate_verdict(
        _automation_spec(),
        logs,
        _AutomationAnalysis(),
        row_caps={"automation_registry": 500},
    )
    check = next(c for c in v.subjects[0].checks if c.id == "not_automated_ref")
    assert check.result == "unknown"
    assert "TRUNCATED" in check.detail
    # ...and an `unknown` here must NOT drag the case to INSUFFICIENT DATA (decisive_on).
    assert check.decisive is False


def test_zero_rows_from_a_KEYED_lookup_is_the_answer_not_a_gap():
    """A query that named this identity and returned nothing ANSWERED the question.

    The two tests above both hand the check real rows to narrow, and that hid an
    assumption: absence was only credited when the source returned SOMETHING. But a keyed
    reference lookup puts the identity in its own WHERE clause, so the backend says "not on
    the list" by returning zero rows — the one shape the engine read as missing data.

    Measured on incident 0184a3ce: the automated register was queried by (office, sign),
    returned 0 rows — meaning the actor is a HUMAN, which is the finding the check exists
    to produce — and the report printed `[NOT EVALUATED] ... no rows | source did not
    return`, while the narration went on to call the subject "the identified automation".
    """
    logs = _fraud_logs()
    logs["automation_registry"] = []  # present-and-empty: the backend ANSWERED

    # Unkeyed (the pre-existing behaviour, and the safe default): nothing tells the engine
    # the query asked about this identity, so the question stays open.
    v = evaluate_verdict(_automation_spec(), logs, _AutomationAnalysis())
    check = next(c for c in v.subjects[0].checks if c.id == "not_automated_ref")
    assert check.result == "unknown"

    # Keyed, as measured by the retrieval stage on the text that actually ran.
    v = evaluate_verdict(
        _automation_spec(),
        logs,
        _AutomationAnalysis(),
        keyed_sources={"automation_registry": True},
    )
    check = next(c for c in v.subjects[0].checks if c.id == "not_automated_ref")
    assert check.result == "pass"
    assert "not found" in check.detail or "absent" in check.detail.lower()


def test_a_keyed_but_TRUNCATED_source_still_reads_unknown():
    """Truncation outranks the key flag: the two guards must not cancel out.

    A keyed query can still be cut off at the row cap — a key on one member of a pair, or a
    prefix match — and then absence from what came back is no evidence at all. If the key
    flag were checked first, the fix for a real UNKNOWN would have manufactured a PASS in
    exactly the case the truncation guard was written for."""
    logs = _fraud_logs()
    logs["automation_registry"] = [
        {"orgUnitId": f"OFF{i:05d}", "sign": "6009JJ", "profile": "AUTOMATED"}
        for i in range(500)
    ]
    v = evaluate_verdict(
        _automation_spec(),
        logs,
        _AutomationAnalysis(),
        row_caps={"automation_registry": 500},
        keyed_sources={"automation_registry": True},
    )
    check = next(c for c in v.subjects[0].checks if c.id == "not_automated_ref")
    assert check.result == "unknown"
    assert "TRUNCATED" in check.detail


def test_a_keyed_flag_cannot_answer_for_an_identity_the_incident_never_carried():
    """`unresolved` still wins: with no value to look up, no query could have asked.

    The flag says the query constrained the source's key — it cannot say the constraint was
    THIS subject's, and an incident missing the org_unit has no pair to be absent."""

    class _NoOrgUnit:
        extracted_entities = [
            ExtractedEntity(type="record", value="SUBJ03"),
            ExtractedEntity(type="user", value="6009JJ"),
        ]

    logs = _fraud_logs()
    logs["automation_registry"] = []
    v = evaluate_verdict(
        _automation_spec(),
        logs,
        _NoOrgUnit(),
        keyed_sources={"automation_registry": True},
    )
    check = next(c for c in v.subjects[0].checks if c.id == "not_automated_ref")
    assert check.result == "unknown"


def _counting_spec():
    """_scheme_spec() plus a `distinct_count` over a source scoped entirely by its query.

    The realized-exposure shape: the count's scope is not the subject and is not re-derived
    here — the query carried it (harvested from an earlier pass), so `subject_scope: false`
    and no `row_match`. `max: 0` because any value at all is the finding.
    """
    spec = _scheme_spec()
    spec["sources"]["access"] = "access_trail"
    spec["conditions"].append(
        {
            "id": "nothing_was_read",
            "label": "No record was read through the granted access",
            "kind": "distinct_count",
            "source": "access",
            "field": "record_locator",
            "max": 0,
            "subject_scope": False,
            "polarity": "fraud_indicator",
            "pass_detail": "the access was not exercised in the window examined",
            "unknown_detail": "the access trail returned nothing to count",
        }
    )
    return spec


def test_a_count_over_an_empty_KEYED_source_is_zero_and_not_unknown():
    """Zero rows from a query that carried the whole scope is a COUNT OF ZERO.

    Same reading as the keyed `record_absence` above, for the other kind. Measured on the
    two-pass session-anomaly run: the follow-up query named both sides of the access
    (`retriever_office IN (...) AND record_owner_office IN (...)`) and returned 0 rows —
    "the receivers read nothing", the finding the pass exists to produce. The check reported
    `unknown / no data`, indistinguishable from the pass never having run, and with `max: 0`
    its PASS branch was unreachable at all: any value FAILs, and no values read `unknown`.
    """
    logs = _fraud_logs()
    logs["access_trail"] = []  # present-and-empty: the backend ANSWERED

    # Unkeyed stays UNKNOWN — the safe default, and the pre-existing behaviour.
    v = evaluate_verdict(_counting_spec(), logs, _AutomationAnalysis())
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "unknown"
    assert "nothing to count" in check.detail

    v = evaluate_verdict(
        _counting_spec(),
        logs,
        _AutomationAnalysis(),
        keyed_sources={"access_trail": True},
    )
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "pass"
    assert check.observed == "0 distinct: []"
    # The PACK's sentence, not the engine's: an empty access trail is not containment, and
    # only the procedure can say so.
    assert check.detail == "the access was not exercised in the window examined"


def test_a_count_over_a_source_that_RETURNED_rows_without_the_field_stays_unknown():
    """Rows-but-no-values is missing data, and the keyed flag must not convert it.

    Both halves of the stamp are load-bearing. A record came back and the counted field was
    absent from it — that is the ordinary unknown every other user of this kind sees, and
    reading it as zero would state "nothing was read" over a source that never carried the
    column. Only an EMPTY source can be a count of zero.
    """
    logs = _fraud_logs()
    logs["access_trail"] = [{"some_other_column": "x"}, {"some_other_column": "y"}]
    v = evaluate_verdict(
        _counting_spec(),
        logs,
        _AutomationAnalysis(),
        keyed_sources={"access_trail": True},
    )
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "unknown"
    assert "nothing to count" in check.detail


def _keyed_counting_spec():
    """`_counting_spec()`'s count, scoped to ONE identity by `row_match` instead of by its query.

    The other shape this kind is asked in: the source answers for a whole population and the
    narrowing happens here, so neither half of the keyed-empty stamp above applies — the source
    is not empty and its query carried no identity.
    """
    spec = _counting_spec()
    cond = next(c for c in spec["conditions"] if c["id"] == "nothing_was_read")
    cond["row_match"] = [
        {"label": "org_unit", "from_entity": "org_unit", "fields": ["orgUnitId"]},
        {
            "label": "sign",
            "from_entity": "user",
            "fields": ["sign"],
            "normalize": "identifier",
        },
    ]
    return spec


def test_a_count_whose_IDENTITY_LOOKUP_ran_and_matched_nothing_is_zero():
    """A source that answered for others, with none of its rows ours, is a count of ZERO.

    The mirror of the keyed-empty stamp, and the shape it cannot see: this source is NOT empty
    and its query named no identity, so the lookup runs here — and when it matches nothing, the
    rows came back in quantity and the narrowing is what emptied them. `record_absence` has read
    exactly this signal since an automation reference list needed it; this kind fell through to
    `no data`, and with `max: 0` its PASS branch was then unreachable altogether: any value FAILs
    and no values hedged.
    """
    logs = _fraud_logs()
    # 3 rows, all of them another (org_unit, sign) pair's.
    logs["access_trail"] = [
        {"orgUnitId": "OTHER001", "sign": "0404EF", "record_locator": f"REC{i}"}
        for i in range(3)
    ]
    v = evaluate_verdict(_keyed_counting_spec(), logs, _AutomationAnalysis())
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "pass"
    assert "the identity lookup found no row of its own" in check.observed
    # The PACK's sentence, as everywhere else on this kind.
    assert check.detail == "the access was not exercised in the window examined"

    # And the FAIL direction still works through the same narrowing — one row of ours among
    # the strangers is the finding, which is what makes the PASS above worth anything.
    logs["access_trail"].append(
        {"orgUnitId": "QQQ1R17GH", "sign": "6009JJ", "record_locator": "MINE01"}
    )
    v = evaluate_verdict(_keyed_counting_spec(), logs, _AutomationAnalysis())
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "fail"
    assert "MINE01" in check.observed


def test_that_zero_is_only_a_count_where_ZERO_IS_THE_BOUND():
    """The bound on the test above, and the direction it reads backwards.

    `max: 0` asks an absence question: no rows matching the subject is the answer. A bound
    that admits rows presupposes the rows the narrowing failed to find; a zero then measures
    the retrieval, not the subject. Reported as a satisfied bound, it counts toward
    `min_evaluated_to_clear` and can license a clear on a record nobody read.
    """
    logs = _fraud_logs()
    logs["access_trail"] = [
        {"orgUnitId": "OTHER001", "sign": "0404EF", "record_locator": f"REC{i}"}
        for i in range(3)
    ]
    spec = _keyed_counting_spec()
    cond = next(c for c in spec["conditions"] if c["id"] == "nothing_was_read")
    cond["max"] = 1
    v = evaluate_verdict(spec, logs, _AutomationAnalysis())
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "unknown", (check.result, check.observed)

    # And the same run at `max: 0` is still the PASS the stamp exists to return, so the guard is a
    # reading of the bound and not a retreat from the stamp.
    cond["max"] = 0
    v = evaluate_verdict(spec, logs, _AutomationAnalysis())
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "pass", (check.result, check.observed)
    assert "the identity lookup found no row of its own" in check.observed

    # A row of ours among the strangers still FAILs at the admitting bound — two distinct values on
    # rows that did match, which is the finding `max: 1` is there for.
    cond["max"] = 1
    logs["access_trail"] += [
        {"orgUnitId": "QQQ1R17GH", "sign": "6009JJ", "record_locator": "MINE01"},
        {"orgUnitId": "QQQ1R17GH", "sign": "6009JJ", "record_locator": "MINE02"},
    ]
    v = evaluate_verdict(spec, logs, _AutomationAnalysis())
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "fail", (check.result, check.observed)


def test_a_TRUNCATED_identity_lookup_that_matched_nothing_is_not_that_zero():
    """The bound on the test above, and the reason the flag is exclusive rather than additive.

    Not finding the pair in a result cut off at its row cap is no evidence that it is absent —
    the measured case, where a register returned 500 rows belonging to 500 other units out of
    89,937. So a capped read stays `unknown` and says the cap did it, while an unfillable clause
    keeps naming the value the incident never supplied.
    """
    logs = _fraud_logs()
    logs["access_trail"] = [
        {"orgUnitId": "OTHER001", "sign": "0404EF", "record_locator": f"REC{i}"}
        for i in range(3)
    ]
    v = evaluate_verdict(
        _keyed_counting_spec(),
        logs,
        _AutomationAnalysis(),
        row_caps={"access_trail": 3},
    )
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "unknown"
    assert "TRUNCATED" in check.detail

    class _NoOrgUnit:
        extracted_entities = [
            ExtractedEntity(type="record", value="SUBJ03"),
            ExtractedEntity(type="user", value="6009JJ"),
        ]

    v = evaluate_verdict(_keyed_counting_spec(), logs, _NoOrgUnit())
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "unknown"
    assert "org_unit" in check.detail


def test_an_empty_FALLBACK_source_does_not_make_the_count_zero_on_its_own():
    """Every candidate source must be empty AND keyed, or the count is not zero.

    A `fallbacks` chain exists because one side may be empty while another answers — so an
    empty primary beside an unkeyed fallback proves nothing: the fallback might have held
    the values and was never asked about this scope.
    """
    spec = _counting_spec()
    cond = next(c for c in spec["conditions"] if c["id"] == "nothing_was_read")
    spec["sources"]["access_alt"] = "access_trail_alt"
    cond["fallbacks"] = [{"source": "access_alt", "field": "recordLocator"}]
    logs = _fraud_logs()
    logs["access_trail"] = []
    logs["access_trail_alt"] = []
    v = evaluate_verdict(
        spec,
        logs,
        _AutomationAnalysis(),
        keyed_sources={"access_trail": True},  # the fallback is NOT keyed
    )
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "unknown"
    # Both keyed and both empty: now the zero is real.
    v = evaluate_verdict(
        spec,
        logs,
        _AutomationAnalysis(),
        keyed_sources={"access_trail": True, "access_trail_alt": True},
    )
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "pass"


# A bounded count over a capped read. `distinct_count`'s bound is an upper bound, so
# truncation is asymmetric: a count over the bound stands, a count within it proves nothing
# because the values that would break it may be in the rows the cap excluded.
def test_a_count_within_its_bound_on_a_TRUNCATED_read_is_unknown():
    spec = _counting_spec()
    cond = next(c for c in spec["conditions"] if c["id"] == "nothing_was_read")
    cond["max"] = 5
    cond["truncated_detail"] = "the cohort was cut off, so this count is a floor"
    logs = _fraud_logs()
    logs["access_trail"] = [{"record_locator": f"REC{i}"} for i in range(3)]

    # Uncapped, or capped above what came back: the count answers.
    v = evaluate_verdict(spec, logs, _AutomationAnalysis())
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "pass"
    v = evaluate_verdict(spec, logs, _AutomationAnalysis(), row_caps={"access_trail": 10})
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "pass"

    # Cap reached: 3 distinct is a FLOOR, so "within 5" is not the finding.
    v = evaluate_verdict(spec, logs, _AutomationAnalysis(), row_caps={"access_trail": 3})
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "unknown"
    assert "TRUNCATED" in check.observed
    assert check.detail == "the cohort was cut off, so this count is a floor"


def test_a_count_OVER_its_bound_on_a_truncated_read_still_fails():
    """The asymmetry, which is the whole reason this is not "capped -> unknown".

    The returned values already break the bound and more of them cannot mend it, so the
    finding stands. Hedging it would delete the indicator on the very keys that earn it.
    """
    spec = _counting_spec()
    cond = next(c for c in spec["conditions"] if c["id"] == "nothing_was_read")
    cond["max"] = 2
    logs = _fraud_logs()
    logs["access_trail"] = [{"record_locator": f"REC{i}"} for i in range(4)]
    v = evaluate_verdict(spec, logs, _AutomationAnalysis(), row_caps={"access_trail": 4})
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "fail"
    assert "4 distinct" in check.observed
    assert "TRUNCATED" not in check.observed


def test_a_truncated_FALLBACK_does_not_hedge_the_count_the_primary_answered():
    """Only the candidate that SUPPLIED the values matters — hence a list, not a flag.

    A `fallbacks` chain is ordered and the first source with any value wins. A flag set by
    any truncated source would hedge a complete primary read on account of a fallback that
    was never consulted: the safe direction, and still a check lost for no reason.
    """
    spec = _counting_spec()
    cond = next(c for c in spec["conditions"] if c["id"] == "nothing_was_read")
    cond["max"] = 5
    spec["sources"]["access_alt"] = "access_trail_alt"
    cond["fallbacks"] = [{"source": "access_alt", "field": "recordLocator"}]
    logs = _fraud_logs()
    logs["access_trail"] = [{"record_locator": "REC1"}]  # the primary answers, in full
    logs["access_trail_alt"] = [{"recordLocator": f"R{i}"} for i in range(2)]
    v = evaluate_verdict(
        spec,
        logs,
        _AutomationAnalysis(),
        row_caps={"access_trail_alt": 2},  # the FALLBACK is the truncated one
    )
    check = next(c for c in v.subjects[0].checks if c.id == "nothing_was_read")
    assert check.result == "pass"
    assert "TRUNCATED" not in check.observed


def test_record_absence_states_its_own_finding_not_the_void_refund_one():
    """A generic condition kind must not narrate one use of itself over another.

    `record_absence` means "no row here carries a forbidden value" and serves both the
    void/refund check and the AUTOMATED reference-list lookup. Its messages were hard-coded to
    the first, so on 83e94dd6 the automated FAIL — the row that decided the case — printed
    "reversal record present / required: no void/refund", which is a statement the evidence
    does not support. The condition phrases its own row; the engine only supplies neutral
    fallbacks.
    """
    spec = _automation_spec()
    cond = next(c for c in spec["conditions"] if c["id"] == "not_automated_ref")
    cond["expected_label"] = (
        "the acting identity is not on the AUTOMATED reference list"
    )
    cond["fail_detail"] = (
        "the acting (org_unit, sign) pair is listed with profile AUTOMATED"
    )
    logs = _fraud_logs()
    logs["automation_registry"] = [
        {"orgUnitId": "QQQ1R17GH", "sign": "6009JJ", "profile": "AUTOMATED"}
    ]
    check = next(
        c
        for c in evaluate_verdict(spec, logs, _AutomationAnalysis()).subjects[0].checks
        if c.id == "not_automated_ref"
    )
    assert check.result == "fail"
    assert "AUTOMATED" in check.expected and "void" not in check.expected.lower()
    assert "reversal" not in check.detail.lower()
    assert "profile AUTOMATED" in check.detail


def test_record_absence_falls_back_to_neutral_wording_when_the_pack_is_silent():
    """A pack that declares no wording must still not get another check's wording.

    The fallback describes what the kind actually tested (the forbidden values it looked
    for), so an unphrased condition is vague rather than wrong.
    """
    spec = _automation_spec()  # declares no expected_label/fail_detail
    logs = _fraud_logs()
    logs["automation_registry"] = [
        {"orgUnitId": "QQQ1R17GH", "sign": "6009JJ", "profile": "AUTOMATED"}
    ]
    check = next(
        c
        for c in evaluate_verdict(spec, logs, _AutomationAnalysis()).subjects[0].checks
        if c.id == "not_automated_ref"
    )
    assert check.expected == "none of AUTOMATED"
    assert "void" not in check.expected.lower() and "refund" not in check.detail.lower()


def test_decisive_on_fail_does_not_force_insufficient_when_unknown():
    """`decisive_on: [fail]` is asymmetric: conclusive on FAIL, inert on unknown/pass.

    Plain `decisive: true` also makes an `unknown` force INSUFFICIENT DATA — wrong for a
    check whose absence of evidence is the ordinary human case."""
    spec = _automation_spec()
    logs = _fraud_logs()
    logs["automation_registry"] = []  # source returned nothing -> unknown
    v = evaluate_verdict(spec, logs, _AutomationAnalysis())
    s = v.subjects[0]
    check = next(c for c in s.checks if c.id == "not_automated_ref")
    assert check.result == "unknown" and check.decisive is False
    # The rest of the fingerprint is intact, so the verdict is unchanged by this unknown.
    assert s.verdict == "VALID FRAUD"
    assert v.degraded is False


def test_row_match_unresolved_identity_reads_unknown():
    """If the incident never supplied one side of the key, the check cannot be evaluated.

    Falling back to the unscoped rows would let other identities' rows answer for the actor
    we failed to identify."""
    spec = _automation_spec()
    logs = _fraud_logs()
    logs["automation_registry"] = [
        {"orgUnitId": "QQQ1R17GH", "sign": "6009JJ", "profile": "AUTOMATED"}
    ]

    class _NoOrgUnit:
        extracted_entities = [
            ExtractedEntity(type="record", value="SUBJ03"),
            ExtractedEntity(type="user", value="6009JJ"),
        ]

    v = evaluate_verdict(spec, logs, _NoOrgUnit())
    check = next(c for c in v.subjects[0].checks if c.id == "not_automated_ref")
    assert check.result == "unknown"
    assert "org_unit" in check.detail


def test_field_flag_bool_and_alias_tolerant():
    """field_flag must (a) read BOOLEAN leaves (resolve_path drops bools) and (b) match
    the flag under whatever alias the SQL gave it (robot_flag / is_robot / ...), keyed on
    the real leaf segment 'robot'."""
    from src.correlation import _eval_condition

    cond = {
        "id": "not_automated",
        "label": "Not automated",
        "kind": "field_flag",
        "source": "auth",
        "flag_fields": ["value.payload.userInfo.robot", "robot"],
        "expected": False,
        "decisive": False,
    }
    # (a) boolean False under an LLM alias the rule never named ("is_robot").
    c = _eval_condition(cond, {"auth": [{"login": "X", "is_robot": False}]})
    assert c.result == "pass"  # not automated
    # (b) a automated sign-in (True) under yet another alias.
    c2 = _eval_condition(cond, {"auth": [{"login": "BOT", "robot_flag": True}]})
    assert c2.result == "fail"  # automated account
    # absent entirely → unknown (honest, non-decisive).
    c3 = _eval_condition(cond, {"auth": [{"login": "X", "org_unit": "Y"}]})
    assert c3.result == "unknown"


def test_evaluate_verdict_notification_rendered():
    v = evaluate_verdict(_scheme_spec(), _fraud_logs(), _Analysis())
    draft = v.notification_draft
    assert "SUBJ03" in draft and "VALID FRAUD" in draft
    assert "ORG2428D4" in draft and "0201GPSU" in draft
    assert "Contain now." in draft  # fraud closing


def test_the_closing_names_the_conditions_the_data_could_not_answer():
    """`{unevaluated}` must read `result`, not the `observed` string.

    Only the `stub` kind literally writes "NOT EVALUATED" into `observed`; a real check the
    data could not answer reports its MEASUREMENT there (`field absent`, `subject NOT IN the
    cohort (61 rows returned)`) and carries the verdict in `result`. Filtering on the string
    therefore found nothing on a live case with two unevaluated conditions, and the closing
    printed `Conditions that could not be evaluated from the retrieved data: .` — while the
    report's own gaps section, which reads `result`, listed both a page earlier.
    """
    spec = _scheme_spec()
    spec["notification"][
        "closing_false_positive"
    ] = "Close as FP. Unevaluated: {unevaluated}."
    logs = _fraud_logs()
    logs["record_lake"][0]["element_counters.AUX"] = 3  # decisive FAIL → FALSE POSITIVE
    del logs["record_lake"][0]["element_counters.ES"]  # `no_es` has nothing to read
    v = evaluate_verdict(spec, logs, _Analysis())
    s = v.subjects[0]
    assert (
        s.verdict == "FALSE POSITIVE"
    )  # so closing_false_positive is the one rendered
    unknown = [c for c in s.checks if c.result == "unknown"]
    assert unknown, "fixture must leave at least one condition unanswered"
    # None of them says "NOT EVALUATED" — each reports the measurement it managed to take,
    # which is why the old `observed`-string filter matched none of them.
    assert all(str(c.observed) != "NOT EVALUATED" for c in unknown)
    for c in unknown:
        assert c.label in v.notification_draft
    assert "Unevaluated: ." not in v.notification_draft


def test_an_empty_unevaluated_list_is_not_a_dangling_sentence():
    """Every template embeds `{unevaluated}` MID-SENTENCE, so blank leaves `...: .`, which
    reads as unrendered rather than as "there were none"."""
    from src.correlation import _unevaluated_labels

    class _C:
        def __init__(self, result, label):
            self.result, self.label, self.id = result, label, "cid"

    assert _unevaluated_labels([_C("pass", "A"), _C("fail", "B")]) == "(none)"
    assert _unevaluated_labels([]) == "(none)"
    # A check with no label at all falls back to its id rather than emitting an empty item.
    nameless = _C("unknown", "")
    assert _unevaluated_labels([nameless]) == "cid"


def test_evaluate_verdict_none_without_spec_or_sources():
    assert evaluate_verdict({}, _fraud_logs(), _Analysis()) is None
    # spec present but its sources aren't in the logs → None
    assert (
        evaluate_verdict(_scheme_spec(), {"other_src": [{"x": 1}]}, _Analysis()) is None
    )


# --- positive fraud indicators (polarity) + rollup precedence ----------------


def _indicator_spec(threshold=2):
    """The compact SCHEME spec plus positive fraud-indicator conditions (polarity)."""
    spec = _scheme_spec()
    spec["indicator_threshold"] = threshold
    spec["conditions"] += [
        {
            "id": "document_stock_mismatch",
            "label": "Stock mismatch",
            "kind": "value_mismatch",
            "polarity": "fraud_indicator",
            "decisive": False,
            "left": {"source": "record", "field": ""},
            "right": {"source": "record", "field": "marketing_providers"},
            "lookup": {
                "from_entity": "document",
                "data": "issuer_prefix_map",
                "prefix_len": 3,
            },
        },
        {
            "id": "non_agency_email",
            "label": "Non-agency email",
            "kind": "value_matches_pattern",
            "polarity": "fraud_indicator",
            "decisive": False,
            "source": "record",
            "fields": ["address.addr_detail.email.red"],
            "match_mode": "allowed",
            "patterns": ["@examplecorp\\.", "examplecorp"],
        },
        {
            "id": "cash_tender",
            "label": "Cash Tender",
            "kind": "value_matches_pattern",
            "polarity": "fraud_indicator",
            "decisive": False,
            "source": "record",
            "data_map": {"field": "pricing.payment.tender.data_map", "match_key": "PM"},
            "match_mode": "forbidden",
            "patterns": ["^CA$"],
        },
        {
            "id": "record_velocity",
            "label": "record velocity",
            "kind": "velocity_count",
            "polarity": "fraud_indicator",
            "decisive": False,
            "source": "record",
            "subject_scope": False,
            "subject_field": "locator.red",
            "actor_field": "creator.sign.red",
            "max": 1,
        },
    ]
    return spec


_PREFIX_DATA = {"issuer_prefix_map": {"map": {"400": "XY"}}}


class _AnalysisApp:
    def __init__(self, records, document="400-2000000006"):
        self.extracted_entities = [
            ExtractedEntity(type="record", value=p) for p in records
        ] + [ExtractedEntity(type="document", value=document)]


def _yryuxz_logs():
    """Two records by one sign, non-agency email + cash tender, consistent 100/XX stock."""

    def _record(record, aux):
        return {
            "locator": {"orange": record, "red": record},
            "creator": {"sign": {"red": "0303CDSU"}, "org_unit_id": "ORGUNIT01"},
            "marketing_providers": '["XY"]',
            "address": {"addr_detail": [{"email": {"red": "ORG_UNIT@EXAMPLECO.RS"}}]},
            "pricing": {
                "payment": [{"tender": [{"data_map": [{"key": "PM", "value": "CA"}]}]}]
            },
            "element_counters": {
                "AUX": aux,
                "OSI": 0,
                "RM": 0,
                "INS": 0,
                "ES": 0,
                "SP": 0,
            },
            "route.air.board_point": "ABJ",
            "route.air.off_point": "CMN",
            "creation_date_time": "2026-07-27T18:00:00",
        }

    return {
        "record_lake": [_record("SUBJ01", 6), _record("YK7LM5", 9)],
        "settlement_report": [
            {
                "relatedRecordLocator": "SUBJ01",
                "retrieverUserSign": "0303CD",
                "recordType": "SALE",
                "transactionDateTime": "2026-07-27T20:17:44",
            },
            {
                "relatedRecordLocator": "YK7LM5",
                "retrieverUserSign": "6005EE",
                "recordType": "SALE",
                "transactionDateTime": "2026-07-27T20:45:40",
            },
        ],
    }


def test_indicators_override_exclusions_valid_fraud():
    """Not-bare (exclusion FAIL) is OVERRIDDEN by >= threshold corroborating
    fraud indicators → VALID FRAUD (the SUBJ01 pattern the FP-only engine missed)."""
    v = evaluate_verdict(
        _indicator_spec(),
        _yryuxz_logs(),
        _AnalysisApp(["SUBJ01", "YK7LM5"]),
        None,
        _PREFIX_DATA,
    )
    assert v is not None
    for s in v.subjects:
        assert s.verdict == "VALID FRAUD", (s.subject_value, s.verdict)
        # stock is CONSISTENT (157=QR vs marketing QR) → PASS, not a fabricated mismatch.
        stock = next(c for c in s.checks if c.id == "document_stock_mismatch")
        assert stock.result == "pass"
        # email + cash + velocity FAIL → >= 2 corroborators.
        fails = {
            c.id
            for c in s.checks
            if c.polarity == "fraud_indicator" and c.result == "fail"
        }
        assert {"non_agency_email", "cash_tender", "record_velocity"} <= fails
    assert not v.degraded


def test_an_undeclared_indicator_threshold_does_not_invent_a_voting_rule():
    """An absent `indicator_threshold` forges a voting rule: with a default of 2, any two
    indicators reach the fraud label under a threshold the procedure never stated. An undeclared
    threshold means the indicators cannot vote at all (the weaker direction), and the reader is
    told, because "no indicator fired" and "three fired but there is no rule for counting them"
    otherwise render identically.
    """
    logs = _yryuxz_logs()
    subjects = _AnalysisApp(["SUBJ01", "YK7LM5"])

    # The declared baseline: three indicators fire, the threshold is 2, the verdict is fraud.
    declared = evaluate_verdict(
        _indicator_spec(), logs, subjects, None, _PREFIX_DATA
    )
    assert [s.verdict for s in declared.subjects] == ["VALID FRAUD"] * 2

    # The same evidence with the threshold undeclared: the indicators still FAIL, and they no
    # longer add up to anything.
    spec = _indicator_spec()
    del spec["indicator_threshold"]
    v = evaluate_verdict(spec, logs, subjects, None, _PREFIX_DATA)
    for s in v.subjects:
        assert s.verdict != "VALID FRAUD", s.subject_value
        fails = {
            c.id
            for c in s.checks
            if c.polarity == "fraud_indicator" and c.result == "fail"
        }
        assert {"non_agency_email", "cash_tender", "record_velocity"} <= fails
        # And the report says so, naming the count that was not weighed.
        note = [n for n in s.notes if n.startswith("indicator_threshold=undeclared")]
        assert len(note) == 1, s.notes
        assert f"{len(fails)} corroborating indicator(s)" in note[0]

    # A declared threshold emits no such note — the note is a report of a missing rule, not a
    # running commentary on every voting ruleset.
    assert not [
        n
        for s in declared.subjects
        for n in s.notes
        if n.startswith("indicator_threshold=")
    ]

    # A ruleset with no indicators at all is unaffected either way: the number governs
    # nothing there, so its absence is not a gap and must not be narrated as one.
    bare = _scheme_spec()
    bare.pop("indicator_threshold", None)
    bv = evaluate_verdict(bare, logs, subjects, None, _PREFIX_DATA)
    assert bv is not None
    assert not [
        n for s in bv.subjects for n in s.notes if n.startswith("indicator_threshold=")
    ]


def test_single_corroborator_does_not_override_exclusion():
    """Mirror-bug guard: ONE non-decisive indicator + an exclusion FAIL stays FALSE
    POSITIVE (a lone cash-tender legit order must not flip to fraud)."""
    logs = _yryuxz_logs()
    # Make it a single record (kills velocity), agency email (kills email indicator),
    # keep ONLY cash tender as the one corroborator; not-bare stays the exclusion FAIL.
    logs["record_lake"] = [logs["record_lake"][0]]
    logs["record_lake"][0]["address"]["addr_detail"][0]["email"][
        "red"
    ] = "agent@examplecorp.com"
    logs["settlement_report"] = [logs["settlement_report"][0]]
    v = evaluate_verdict(
        _indicator_spec(), logs, _AnalysisApp(["SUBJ01"]), None, _PREFIX_DATA
    )
    s = v.subjects[0]
    assert s.verdict == "FALSE POSITIVE"
    ind_fails = [
        c for c in s.checks if c.polarity == "fraud_indicator" and c.result == "fail"
    ]
    assert [c.id for c in ind_fails] == ["cash_tender"]


def test_no_indicator_still_false_positive():
    """SUBJ03 regression: not-bare exclusion FAIL, NO indicator FAILs → FALSE
    POSITIVE (the discriminator is presence of indicators, not absence of exclusions).
    """
    logs = _yryuxz_logs()
    logs["record_lake"] = [logs["record_lake"][0]]
    # Clean order: agency email, non-cash tender, single record, consistent stock.
    logs["record_lake"][0]["address"]["addr_detail"][0]["email"][
        "red"
    ] = "agent@examplecorp.com"
    logs["record_lake"][0]["pricing"]["payment"][0]["tender"][0]["data_map"] = [
        {"key": "PM", "value": "CC"}
    ]
    logs["settlement_report"] = [logs["settlement_report"][0]]
    v = evaluate_verdict(
        _indicator_spec(), logs, _AnalysisApp(["SUBJ01"]), None, _PREFIX_DATA
    )
    s = v.subjects[0]
    assert s.verdict == "FALSE POSITIVE"
    assert not any(
        c.polarity == "fraud_indicator" and c.result == "fail" for c in s.checks
    )


# --- categorical exclusions outrank the indicator vote (IR10000002) ----------


def _categorical_spec(threshold=2):
    """The indicator spec plus the automated reference check as a CATEGORICAL exclusion."""
    spec = _indicator_spec(threshold)
    spec["sources"]["automation"] = "automation_registry"
    spec["conditions"].append(
        {
            "id": "not_automated_ref",
            "label": "Not a known AUTOMATED account",
            "kind": "record_absence",
            "source": "automation",
            "subject_scope": False,
            "row_match": [
                {
                    "label": "org_unit",
                    "from_entity": "org_unit",
                    "fields": ["orgUnitId"],
                },
                {
                    "label": "sign",
                    "from_entity": "user",
                    "fields": ["sign"],
                    "normalize": "identifier",
                },
            ],
            "fields": ["profile"],
            "forbidden_values": ["AUTOMATED"],
            "match": "exact",
            "decisive": True,
            "decisive_on": ["fail"],
            "exclusion_kind": "categorical",
        }
    )
    return spec


class _AnalysisAppIdentity(_AnalysisApp):
    """_AnalysisApp plus the acting (org_unit, sign) the automated lookup is keyed by."""

    def __init__(self, records, org_unit="ORGUNIT01", sign="0303CD", **kw):
        super().__init__(records, **kw)
        self.extracted_entities += [
            ExtractedEntity(type="org_unit", value=org_unit),
            ExtractedEntity(type="user", value=sign),
        ]


def test_categorical_exclusion_outranks_indicator_vote():
    """IR10000002: a confirmed AUTOMATED actor beats 3 failing fraud indicators.

    Most exclusions are heuristic tripwires that positive indicators can rightly outweigh.
    A categorical one establishes WHO ACTED — and once the actor is an automation, the
    indicators lose their MEANING, not just their weight: non-agency email + cash tender +
    burst is what a travel agency's own web-service application looks like when it books.
    The engine had this FAIL and still returned VALID FRAUD (3 indicators outvoted it);
    the expert closed the case NOT A FRAUD on precisely this evidence."""
    logs = _yryuxz_logs()
    logs["automation_registry"] = [
        {"orgUnitId": "ORGUNIT01", "sign": "0303CD", "profile": "AUTOMATED"}
    ]
    v = evaluate_verdict(
        _categorical_spec(),
        logs,
        _AnalysisAppIdentity(["SUBJ01", "YK7LM5"]),
        None,
        _PREFIX_DATA,
    )
    for s in v.subjects:
        assert s.verdict == "FALSE POSITIVE", (s.subject_value, s.verdict)
        assert next(c for c in s.checks if c.id == "not_automated_ref").result == "fail"
        # The indicators DID fail — the override is a ranking, not a re-evaluation...
        fails = {
            c.id
            for c in s.checks
            if c.polarity == "fraud_indicator" and c.result == "fail"
        }
        assert {"non_agency_email", "cash_tender", "record_velocity"} <= fails
        # ...and the report must be able to say so honestly.
        note = next(n for n in s.notes if n.startswith("categorical_exclusion="))
        assert "outranks" in note and "Non-agency email" in note
        # No "fraud_indicators=" note, which reads as "this verdict rests on fraud evidence".
        assert not any(n.startswith("fraud_indicators=") for n in s.notes)
    assert not v.degraded


def test_the_override_note_asserts_nothing_about_WHO_the_actor_IS():
    """The engine may rank the exclusion above the indicators; it may not say what it found.

    The branch knows one thing: some `categorical`-tagged condition failed. What that means
    is the pack's to say via `fail_detail`. The engine must not append its own claim about the
    actor, because a categorical exclusion is any question about who acted, and one pack's
    may be a unit allowlist that says nothing about automation. The fixture uses an unknown
    automation check and a different categorical exclusion as the decisive one.
    """
    spec = _categorical_spec()
    # The automation lookup can answer nothing: its source returns no rows at all, which is
    # `unknown` and not `pass` (absence is only an answer when the lookup RAN over real rows).
    spec["sources"]["automation"] = "registry_that_did_not_return"
    # A second categorical exclusion, on an entirely different question, is what FAILs.
    spec["sources"]["allowlist"] = "unit_allowlist"
    spec["conditions"].append(
        {
            "id": "unit_on_allowlist",
            "kind": "field_flag",
            "source": "allowlist",
            "flag_fields": ["excluded"],
            "expected": False,
            "subject_scope": False,
            "decisive": True,
            "decisive_on": ["fail"],
            "exclusion_kind": "categorical",
            "label": "The acting unit is not on the detector's own exclusion list",
            "fail_detail": (
                "the action came from a unit the detector was never meant to flag"
            ),
        }
    )
    logs = _yryuxz_logs()
    logs["registry_that_did_not_return"] = []
    logs["unit_allowlist"] = [{"excluded": True}]
    v = evaluate_verdict(
        spec, logs, _AnalysisAppIdentity(["SUBJ01", "YK7LM5"]), None, _PREFIX_DATA
    )
    for s in v.subjects:
        checks = {c.id: c for c in s.checks}
        # The premise of the test: the automation question is genuinely unanswered, and the
        # override is driven by the other exclusion entirely.
        assert checks["not_automated_ref"].result == "unknown"
        assert checks["unit_on_allowlist"].result == "fail"
        note = next(n for n in s.notes if n.startswith("categorical_exclusion="))
        # The pack's own wording for what the FAIL means is there...
        assert "never meant to flag" in note
        # ...the outranked indicators are NAMED, so the report can be honest about them...
        assert "Non-agency email" in note and "outranks" in note
        # ...and nothing is asserted about what they are, least of all automation.
        low = note.lower()
        assert "automation" not in low and "automated" not in low, note
        assert "robot" not in low, note


def test_exclusion_kind_rides_on_the_check_not_a_note_prefix():
    """The engine, the case-builder and the report must agree on ONE typed field.

    The override was originally signalled by three files independently string-matching a
    `categorical_exclusion=` prefix on a free-text `notes` list — a contract that any
    wording change breaks silently. `ConditionCheck.exclusion_kind` carries it instead, and
    the rollup ranks on that same field, so consumers cannot drift from the engine."""
    logs = _yryuxz_logs()
    logs["automation_registry"] = [
        {"orgUnitId": "ORGUNIT01", "sign": "0303CD", "profile": "AUTOMATED"}
    ]
    v = evaluate_verdict(
        _categorical_spec(),
        logs,
        _AnalysisAppIdentity(["SUBJ01", "YK7LM5"]),
        None,
        _PREFIX_DATA,
    )
    for s in v.subjects:
        by_id = {c.id: c for c in s.checks}
        assert by_id["not_automated_ref"].exclusion_kind == "categorical"
        # Every other condition keeps the default, so the ranking is opt-in per condition.
        assert all(
            c.exclusion_kind == "heuristic"
            for cid, c in by_id.items()
            if cid != "not_automated_ref"
        )


def test_categorical_exclusion_absent_leaves_indicator_vote_intact():
    """The override is opt-in per condition AND per outcome: a PASS/UNKNOWN automated check
    must not disturb the SUBJ01 indicator path."""
    logs = _yryuxz_logs()
    # The lookup ran over real rows and this identity is NOT on the list -> pass.
    logs["automation_registry"] = [
        {"orgUnitId": "OTHER1234", "sign": "0303CD", "profile": "AUTOMATED"}
    ]
    v = evaluate_verdict(
        _categorical_spec(),
        logs,
        _AnalysisAppIdentity(["SUBJ01", "YK7LM5"]),
        None,
        _PREFIX_DATA,
    )
    for s in v.subjects:
        assert s.verdict == "VALID FRAUD", (s.subject_value, s.verdict)
        assert not any(n.startswith("categorical_exclusion=") for n in s.notes)


def test_the_rollup_stamps_a_machine_readable_class_on_every_branch():
    """`verdict_class` is the rollup's own answer, recorded because it cannot be recovered.

    Consumers ask one question of a verdict — was this subject cleared? — and two ways of
    answering it downstream have shipped wrong. Matching the label PROSE compared against
    the literal "FALSE POSITIVE" that no shipped ruleset emits, so the confidence clamp was
    dead on every live run. RE-DERIVING the rollup from `checks` ("a decisive exclusion FAIL
    with no decisive indicator FAIL") misread stored VALID FRAUD subjects as dismissed,
    because two branches below turn on things a check does not carry: fraud reached through
    CORROBORATED NON-DECISIVE indicators, and a categorical exclusion outranking the lot.

    So every branch is asserted here, including those two, and each class is checked
    against the LABEL it was stamped with — the pair is the contract.
    """
    # fraud, via decisive checks passing
    v = evaluate_verdict(_scheme_spec(), _fraud_logs(), _Analysis())
    assert (v.subjects[0].verdict, v.subjects[0].verdict_class) == (
        "VALID FRAUD",
        "fraud",
    )

    # false_positive, via an observed decisive exclusion FAIL
    logs = _fraud_logs()
    logs["record_lake"][0]["element_counters.AUX"] = 3
    s = evaluate_verdict(_scheme_spec(), logs, _Analysis()).subjects[0]
    assert (s.verdict, s.verdict_class) == ("FALSE POSITIVE", "false_positive")

    # insufficient, via a decisive UNKNOWN
    s = evaluate_verdict(
        _scheme_spec(),
        {"record_lake": _fraud_logs()["record_lake"], "settlement_report": []},
        _Analysis(),
    ).subjects[0]
    assert (s.verdict, s.verdict_class) == ("INSUFFICIENT DATA", "insufficient")

    # fraud, via CORROBORATED NON-DECISIVE indicators — the branch a check cannot show.
    logs = _yryuxz_logs()
    for s in evaluate_verdict(
        _indicator_spec(), logs, _AnalysisApp(["SUBJ01", "YK7LM5"]), None, _PREFIX_DATA
    ).subjects:
        assert s.verdict_class == "fraud", (s.subject_value, s.verdict)
        # Precisely the shape the removed re-derivation got wrong: the indicators that
        # carried this verdict are NOT decisive.
        assert not any(
            c.decisive for c in s.checks if c.polarity == "fraud_indicator"
        ), "if these become decisive the branch is no longer covered"

    # false_positive, via a categorical exclusion outranking that same indicator vote.
    logs = _yryuxz_logs()
    logs["automation_registry"] = [
        {"orgUnitId": "ORGUNIT01", "sign": "0303CD", "profile": "AUTOMATED"}
    ]
    for s in evaluate_verdict(
        _categorical_spec(),
        logs,
        _AnalysisAppIdentity(["SUBJ01", "YK7LM5"]),
        None,
        _PREFIX_DATA,
    ).subjects:
        assert (s.verdict, s.verdict_class) == ("FALSE POSITIVE", "false_positive")


def test_heuristic_exclusion_still_loses_to_indicators():
    """Regression guard for SUBJ01: an exclusion WITHOUT `exclusion_kind: categorical`
    keeps the old behaviour — the indicator vote overrides it."""
    spec = _categorical_spec()
    # Demote the automated check to an ordinary (heuristic) exclusion.
    for c in spec["conditions"]:
        if c["id"] == "not_automated_ref":
            c.pop("exclusion_kind")
    logs = _yryuxz_logs()
    logs["automation_registry"] = [
        {"orgUnitId": "ORGUNIT01", "sign": "0303CD", "profile": "AUTOMATED"}
    ]
    v = evaluate_verdict(
        spec, logs, _AnalysisAppIdentity(["SUBJ01", "YK7LM5"]), None, _PREFIX_DATA
    )
    for s in v.subjects:
        assert s.verdict == "VALID FRAUD", (s.subject_value, s.verdict)


def test_value_mismatch_unknown_when_prefix_unmapped():
    """An unmapped document prefix yields UNKNOWN (never a fabricated mismatch)."""
    from src.correlation import _eval_condition

    cond = {
        "id": "stock",
        "kind": "value_mismatch",
        "polarity": "fraud_indicator",
        "left": {"source": "record", "field": ""},
        "right": {"source": "record", "field": "marketing_providers"},
        "lookup": {"from_entity": "document", "prefix_len": 3},
        "_lookup_data": {"400": "XY"},
        "_lookup_entities": ["999-0000000000"],
    }
    rows = {"record": [{"marketing_providers": '["XY"]'}]}
    c = _eval_condition(cond, rows)
    assert c.result == "unknown"


def test_value_matches_pattern_datamap_cash():
    """Cash-tender scan reads {key:FP, value:CA} out of the nested data_map list."""
    from src.correlation import _eval_condition

    cond = {
        "id": "cash",
        "kind": "value_matches_pattern",
        "polarity": "fraud_indicator",
        "source": "record",
        "data_map": {"field": "pricing.payment.tender.data_map", "match_key": "PM"},
        "match_mode": "forbidden",
        "patterns": ["^CA$"],
    }
    rows = {
        "record": [
            {
                "pricing": {
                    "payment": [
                        {
                            "tender": [
                                {
                                    "data_map": [
                                        {"key": "ET", "value": "MS"},
                                        {"key": "PM", "value": "CA"},
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }
        ]
    }
    assert _eval_condition(cond, rows).result == "fail"
    # non-cash tender → pass
    rows2 = {
        "record": [
            {
                "pricing": {
                    "payment": [{"tender": [{"data_map": [{"key": "PM", "value": "CC"}]}]}]
                }
            }
        ]
    }
    assert _eval_condition(cond, rows2).result == "pass"


def test_value_matches_pattern_datamap_cash_aliased_column():
    """Databricks aliases `pricing.payment` to a flat `pricing_payment` JSON-string column — the cash-tender
    scan must resolve through the underscore alias (this returned UNKNOWN on live data
    before the fix, silently dropping the expert's cash-tender signal)."""
    import json as _json

    from src.correlation import _eval_condition

    cond = {
        "id": "cash",
        "kind": "value_matches_pattern",
        "polarity": "fraud_indicator",
        "source": "record",
        "data_map": {"field": "pricing.payment.tender.data_map", "match_key": "PM"},
        "match_mode": "forbidden",
        "patterns": ["^CA$"],
    }
    # pricing.payment collapsed to `pricing_payment` holding a JSON-string, exactly as the retriever returns.
    row = {
        "pricing_payment": _json.dumps(
            [
                {
                    "tender": [
                        {
                            "data_map": [
                                {"key": "ET", "value": "MS"},
                                {"key": "PM", "value": "CA"},
                            ]
                        }
                    ]
                }
            ]
        )
    }
    assert _eval_condition(cond, {"record": [row]}).result == "fail"


# ── value_matches_pattern on an EAV source ───────────────────────────────────────────────
# Without a `where:` row selector every attribute's value is compared; without a third outcome
# for inconclusive patterns, an ambiguous code clears or convicts with false certainty.


def _org_unit_eav_rows(org_unit_type: str):
    """One org_unit's `org_unit_profile` rows — EAV, one row per attribute (~130 in real life).

    Trimmed to the shapes that matter: the typed org_unit-type attribute the check reads, plus
    the free-text and code attributes that sit beside it and would be compared too.
    """
    return [
        {"CODE": "AD1", "VALUE": "52 ALLEN AVENUE"},
        {"CODE": "ARP", "VALUE": "NG"},
        {"CODE": "CTO", "VALUE": "PRD"},
        {"CODE": "NDC", "VALUE": "TA : TRAVEL AGENCY"},
        {"CODE": "UST", "VALUE": org_unit_type},
    ]


def _org_unit_type_cond():
    return {
        "id": "not_transit_org_unit",
        "label": "Order org_unit is not located at an transit_hub",
        "kind": "value_matches_pattern",
        "source": "org_unit_ref",
        "where": [{"field": "CODE", "any_of": ["UST"], "match": "exact"}],
        "fields": ["VALUE"],
        "match_mode": "allowed",
        "patterns": [r"^N\b", r"^T\b", r"^E\b", r"^X\b"],
        "inconclusive_patterns": [r"^A\b"],
        "inconclusive_detail": "org_unit-type lumps transit_hub and city desks under one value",
    }


def test_value_matches_pattern_where_selects_the_attribute_row():
    """On an EAV source the `where:` selector is what makes the check read the right field.

    Without it every attribute's VALUE is compared against the allow-list, so a street line
    and a country code become "values outside the allow-list" — a FAIL manufactured from an
    org_unit's postal address. The constant 'UST' is not an entity, so nothing in the retrieval
    path adds it; only the condition can."""
    from src.correlation import _eval_condition

    rows = {"org_unit_ref": _org_unit_eav_rows("T : SCHEME_CODE TRAVEL AGENT")}
    with_where = _eval_condition(_org_unit_type_cond(), rows)
    assert with_where.result == "pass"

    without_where = dict(_org_unit_type_cond())
    without_where.pop("where")
    assert _eval_condition(without_where, rows).result == "fail"


def test_value_matches_pattern_inconclusive_value_is_unknown_not_a_verdict():
    """A value that lumps both cases the check distinguishes must read `unknown`.

    `A : PROVIDER ORG_UNIT (ATO/CTO)` covers 119,758 org_units and names the transit_hub desk §3.3
    excludes AND the city org_unit it does not. Listing it as allowed would clear every transit_hub
    desk; listing it as forbidden would flag every provider city org_unit."""
    from src.correlation import _eval_condition

    cond = _org_unit_type_cond()
    check = _eval_condition(
        cond, {"org_unit_ref": _org_unit_eav_rows("A : PROVIDER ORG_UNIT (ATO/CTO)")}
    )
    assert check.result == "unknown"
    # The value is reported, so "the vocabulary cannot decide" is distinguishable from
    # "the field was absent" — which is the other way this check reaches `unknown`.
    assert "PROVIDER ORG_UNIT" in check.observed
    assert "lumps" in check.detail

    absent = _eval_condition(
        cond,
        {"org_unit_ref": [r for r in _org_unit_eav_rows("x") if r["CODE"] != "UST"]},
    )
    assert absent.result == "unknown"
    assert absent.observed != check.observed


def test_value_matches_pattern_allowed_org_unit_types_pass():
    """The three values that soundly exclude an provider transit_hub desk."""
    from src.correlation import _eval_condition

    for org_unit_type in (
        "N : NON-SCHEME_CODE AGENT",
        "T : SCHEME_CODE TRAVEL AGENT",
        "E : ELECTRONIC SYSTEM",
    ):
        check = _eval_condition(
            _org_unit_type_cond(), {"org_unit_ref": _org_unit_eav_rows(org_unit_type)}
        )
        assert check.result == "pass", org_unit_type


# ── a blank is not a value ───────────────────────────────────────────────────────────────
# `_collect` drops None but keeps ''. An unset column arrives as an empty string from SQL
# backends, and on a multi-shape source a class-conditional column is blank on every shape but
# one. Both consequences below report a clean ANSWER through `_first_present` for a question
# that could not be asked: neither raises, neither empties a source.


def _blank_cond(mode: str, fields=("primary",)):
    return {
        "id": "outcome_not_refused",
        "label": "No read carried the refusal outcome",
        "kind": "value_matches_pattern",
        "source": "pivot",
        "fields": list(fields),
        "match_mode": mode,
        "patterns": [r"^N$"] if mode == "forbidden" else [r"\S"],
    }


def test_a_field_blank_on_every_row_is_unknown_and_not_a_clean_pass():
    """Nothing to compare must read `unknown`, which is what the empty-result guard intends.

    The guard three lines above the comparison already returns `unknown` for a field that
    resolved to no values — blanks defeated it by making the result non-empty. The comparison
    then ran over zero comparable values, found no offender, and reported "none": a `forbidden`
    exclusion CLEARING its subject, and an `allowed` scope gate ADMITTING rows that carry no
    identity at all. Both are the invisible direction of the same defect the pack's zero-row
    work addresses — a source that answered nothing, read as an answer of nothing."""
    from src.correlation import _eval_condition

    for mode in ("forbidden", "allowed"):
        for blank in ("", "   "):
            check = _eval_condition(
                _blank_cond(mode), {"pivot": [{"primary": blank}, {"primary": blank}]}
            )
            assert check.result == "unknown", (mode, repr(blank))
        # A real value on the same field still decides, in both modes.
        real = _eval_condition(_blank_cond(mode), {"pivot": [{"primary": "B"}]})
        assert real.result == "pass", mode
    assert (
        _eval_condition(_blank_cond("forbidden"), {"pivot": [{"primary": "N"}]}).result
        == "fail"
    )


def test_a_blank_first_field_does_not_shadow_a_populated_fallback():
    """`fields:` is first-present so a check survives one spelling being absent.

    The empty string was the one form of absent that stopped the search: the first field
    "resolved", the sibling holding the value was never read, and the check answered from a
    column that had nothing in it. Measured shape: an identity column empty on 100% of one
    record class while the sibling login column carries the value on the same rows."""
    from src.correlation import _eval_condition

    rows = {"pivot": [{"primary": "", "fallback": "N"}]}
    check = _eval_condition(_blank_cond("forbidden", ("primary", "fallback")), rows)
    assert check.result == "fail"
    assert "N" in check.observed

    # And the fallback is read for the clearing direction too, not only the convicting one.
    clear = _eval_condition(
        _blank_cond("allowed", ("primary", "fallback")),
        {"pivot": [{"primary": "  ", "fallback": "AB1234"}]},
    )
    assert clear.result == "pass"
    assert "AB1234" in clear.observed


def test_an_allow_list_PASS_names_the_value_that_cleared_the_subject():
    """A pass on an allow-list must name the value that cleared the subject.

    A `forbidden` pass is an absence, so it reports the count and source. An `allowed` pass
    is the opposite: every value present is a reason, so the bare conclusion discards evidence
    while the fail side names the offending values. The last assertion checks that the observed
    string varies with the data, not a constant containing "allowed".
    """
    from src.correlation import _eval_condition

    cond = _org_unit_type_cond()
    cleared = _eval_condition(
        cond, {"org_unit_ref": _org_unit_eav_rows("T : SCHEME_CODE TRAVEL AGENT")}
    )
    assert cleared.result == "pass"
    assert "T : SCHEME_CODE TRAVEL AGENT" in cleared.observed

    # The symmetry claim: the FAIL side names its values, so the PASS side must too.
    convicted = _eval_condition(
        cond, {"org_unit_ref": _org_unit_eav_rows("Z : SOMETHING ELSE")}
    )
    assert convicted.result == "fail"
    assert "Z : SOMETHING ELSE" in convicted.observed

    # A `forbidden` PASS is left exactly as it was: an absence has no value to quote, and
    # what it reports instead is the size of the read.
    forbidden = {
        "id": "no_forbidden_mode",
        "kind": "value_matches_pattern",
        "source": "record",
        "fields": ["mode"],
        "match_mode": "forbidden",
        "patterns": ["^W$"],
    }
    absent = _eval_condition(forbidden, {"record": [{"mode": "R"}]})
    assert absent.result == "pass"
    assert "none" in absent.observed
    assert "R" not in absent.observed.replace("record", "")

    # NOT a constant: a different allowed value must produce a different sentence.
    other = _eval_condition(
        cond, {"org_unit_ref": _org_unit_eav_rows("N : NON-SCHEME_CODE AGENT")}
    )
    assert other.result == "pass"
    assert "N : NON-SCHEME_CODE AGENT" in other.observed
    assert other.observed != cleared.observed


def test_value_matches_pattern_prints_the_packs_own_PASS_wording():
    """A PASS sentence is the pack's, on this kind as on every other one.

    This kind serves BOTH polarities, so one hardcoded PASS sentence cannot be right for
    both: under `polarity: fraud_indicator` a PASS means the indicator did not fire, and
    "no suspicious value" over an exclusion whose PASS is the fraud-consistent side states
    the opposite of what was found. Ten sibling kinds already read `pass_detail`; this one
    ignored it, so a pack could declare the sentence and see NO effect — the expensive shape
    of the defect, because nothing errors and the report reads as considered prose.

    Asserted on the emitted detail (what a reader sees), both match modes, and the absent
    case still yields the old default so no existing pack's report changes.
    """
    from src.correlation import _eval_condition

    allow = {
        "id": "read_only",
        "kind": "value_matches_pattern",
        "source": "record",
        "fields": ["mode"],
        "match_mode": "allowed",
        "patterns": ["^R$"],
        "pass_detail": "every access mode granted is read-only",
    }
    passed = _eval_condition(allow, {"record": [{"mode": "R"}]})
    assert passed.result == "pass"
    assert passed.detail == "every access mode granted is read-only"

    forbid = {
        "id": "no_write_grant",
        "kind": "value_matches_pattern",
        "source": "record",
        "fields": ["apps"],
        "match_mode": "forbidden",
        "patterns": ["[A-Z]{3}[BW]"],
        "pass_detail": "no write-bearing grant in the session",
    }
    passed = _eval_condition(forbid, {"record": [{"apps": "-PNOR /PNNR"}]})
    assert passed.result == "pass"
    assert passed.detail == "no write-bearing grant in the session"

    # Absent, or declared blank, keeps the wording every shipped pack's report already has.
    for cond in (
        {k: v for k, v in forbid.items() if k != "pass_detail"},
        {**forbid, "pass_detail": ""},
    ):
        fallback = _eval_condition(cond, {"record": [{"apps": "-PNOR"}]})
        assert fallback.result == "pass"
        assert fallback.detail == "no suspicious value"

    # And the FAIL side is untouched by any of this.
    failed = _eval_condition(forbid, {"record": [{"apps": "-PNOB"}]})
    assert failed.result == "fail"
    assert "suspicious value present" in failed.detail


def test_inconclusive_patterns_do_not_disturb_a_forbidden_check():
    """A condition declaring no `inconclusive_patterns` behaves exactly as before."""
    from src.correlation import _eval_condition

    cond = {
        "id": "cash",
        "kind": "value_matches_pattern",
        "source": "record",
        "fields": ["tender"],
        "match_mode": "forbidden",
        "patterns": ["^CA$"],
    }
    assert _eval_condition(cond, {"record": [{"tender": "CA"}]}).result == "fail"
    assert _eval_condition(cond, {"record": [{"tender": "CC"}]}).result == "pass"
    # And an inconclusive pattern that matches NOTHING leaves both outcomes alone.
    cond2 = {**cond, "inconclusive_patterns": ["^ZZ$"]}
    assert _eval_condition(cond2, {"record": [{"tender": "CA"}]}).result == "fail"
    assert _eval_condition(cond2, {"record": [{"tender": "CC"}]}).result == "pass"


# ── `inconclusive_blocks_pass`: two meanings behind one key ─────────────────────────────
# `inconclusive_patterns` drops values before comparison. The key decides what the survivors do:
# absent (default): the dropped value is a third party's; survivors decide the question.
# true: the dropped value is the subject's and unclassifiable; survivors cannot clear.
# A fail is unaffected in both: a matched value is evidence regardless of dropped values.


def _coded_attribute_cond(**over):
    cond = {
        "id": "no_entitling_status",
        "label": "No status entitling the outcome is recorded",
        "kind": "value_matches_pattern",
        "source": "record",
        "fields": ["status_code"],
        "match_mode": "forbidden",
        "patterns": ["^ENT$"],
        "inconclusive_patterns": ["^ZZZ$"],
        "inconclusive_detail": "the recorded status is a code whose meaning is not established",
    }
    cond.update(over)
    return cond


def test_an_undecidable_value_beside_a_clean_one_withholds_the_clear_when_declared():
    """The case the key exists for, and the only one in which it changes an answer.

    The field holds two values: one the comparison can read and clears on, one it dropped. The
    arithmetic alone reports the clear — a sentence saying no entitling status is recorded, over
    a record that holds a status nobody could classify. With the declaration the check reads
    `unknown`, names the dropped value, and says how many survivors decided nothing either way,
    so the reader can tell this from an absent field."""
    from src.correlation import _eval_condition

    rows = {"record": [{"status_code": ["OTH", "ZZZ"]}]}
    check = _eval_condition(_coded_attribute_cond(inconclusive_blocks_pass=True), rows)
    assert check.result == "unknown"
    assert "ZZZ" in check.observed, check.observed
    # BOTH counts, because the two ways this check reaches `unknown` are otherwise identical in
    # the report: nothing was read, versus something was read and could not be classified.
    assert "unclassifiable" in check.observed, check.observed
    assert "1 value(s)" in check.observed, check.observed
    assert "meaning is not established" in check.detail

    # Declared, and nothing dropped: the ordinary value still clears. A withholding rule that
    # withholds everywhere is not a stricter check, it is a disabled one.
    ordinary = _eval_condition(
        _coded_attribute_cond(inconclusive_blocks_pass=True),
        {"record": [{"status_code": "OTH"}]},
    )
    assert ordinary.result == "pass"


def test_the_same_mix_without_the_declaration_still_clears():
    """The DEFAULT is the other reading, and it must be byte-identical to what shipped.

    Every declaration that existed when this key was added means "the dropped value is somebody
    else's", where the survivors genuinely decide. So the absence of the key is not "unset" — it
    is the first reading, and this test is what keeps the fix from silently changing the answer
    of every condition that never asked for it."""
    from src.correlation import _eval_condition

    rows = {"record": [{"status_code": ["OTH", "ZZZ"]}]}
    check = _eval_condition(_coded_attribute_cond(), rows)
    assert check.result == "pass"
    # And explicitly false is the same as absent — a pack may write it down to record the
    # judgement, which must not then behave like a third state.
    off = _eval_condition(_coded_attribute_cond(inconclusive_blocks_pass=False), rows)
    assert off.result == "pass"


def test_a_forbidden_match_among_the_survivors_still_fails_either_way():
    """The guard withholds a CLEAR and never a finding.

    A dropped value cannot argue against a match that was found: the match is present in the
    data whatever sits beside it. So the FAIL side is identical with and without the key — the
    one property that keeps this from being a way to bury findings under an unreadable code."""
    from src.correlation import _eval_condition

    rows = {"record": [{"status_code": ["ENT", "ZZZ"]}]}
    for cond in (
        _coded_attribute_cond(),
        _coded_attribute_cond(inconclusive_blocks_pass=True),
    ):
        check = _eval_condition(cond, rows)
        assert check.result == "fail", cond.get("inconclusive_blocks_pass")
        assert "ENT" in check.observed


def test_the_guard_holds_in_allowed_mode_too():
    """`allowed` reaches its clear by the opposite route and needs the same withholding.

    In `forbidden` mode a clear means nothing matched; in `allowed` mode it means everything
    read matched. A dropped value breaks the second claim more directly than the first — "every
    value is on the allow-list" is false of a record holding one that was never compared — so a
    guard implemented on one branch only would leave the mode where the sentence is plainly
    wrong. Both clear returns are covered, which is why this is a separate test."""
    from src.correlation import _eval_condition

    allow = _coded_attribute_cond(
        match_mode="allowed", patterns=["^OTH$"], inconclusive_blocks_pass=True
    )
    mixed = _eval_condition(allow, {"record": [{"status_code": ["OTH", "ZZZ"]}]})
    assert mixed.result == "unknown"
    assert "ZZZ" in mixed.observed

    # The clean case still passes, and the out-of-list case still fails.
    assert (
        _eval_condition(allow, {"record": [{"status_code": "OTH"}]}).result == "pass"
    )
    assert (
        _eval_condition(allow, {"record": [{"status_code": "XXX"}]}).result == "fail"
    )


# ── `ordinary_patterns`: the third tier ─────────────────────────────────────────────────
# A value outside both `patterns` and `inconclusive_patterns` produces a spurious clean check.
# Declaring `ordinary_patterns` closes the vocabulary and withholds the clear on an outsider.


def _closed_vocabulary_cond(**over):
    """The vocabulary shape: three tiers, all three declared, nothing left implicit."""
    cond = {
        "id": "no_entitling_code",
        "label": "No code entitling the outcome is recorded",
        "kind": "value_matches_pattern",
        "source": "record",
        "fields": ["type_code"],
        "match_mode": "forbidden",
        "patterns": ["^ENT$"],
        "inconclusive_patterns": ["^ZZZ$"],
        "inconclusive_detail": "the recorded code is one whose meaning is not established",
        "ordinary_patterns": ["^ORD$", "^DEF$"],
    }
    cond.update(over)
    return cond


def test_a_value_outside_every_declared_list_withholds_the_clear():
    """The case the key exists for: a code the pack has never classified.

    The arithmetic alone reports the clear, because an unlisted value simply fails to match a
    forbidden pattern — a sentence saying no entitling code is recorded, over a record holding a
    code nobody has read. With the vocabulary closed the check reads `unknown` and names the
    value as an outsider, which is a different remedy from every other `unknown` this kind can
    reach: extend the pack's vocabulary, rather than find an authority for a code it knows.
    """
    from src.correlation import _eval_condition

    rows = {"record": [{"type_code": ["ORD", "QQQ"]}]}
    check = _eval_condition(_closed_vocabulary_cond(), rows)
    assert check.result == "unknown"
    assert "QQQ" in check.observed, check.observed
    assert "outside every classification" in check.observed, check.observed
    # The surviving ordinary value is counted, so this is distinguishable from an absent field.
    assert "1 value(s)" in check.observed, check.observed
    assert "declared vocabulary does not cover" in check.detail, check.detail

    # And a record holding only classified values still clears. A closed vocabulary is not a
    # check that withholds everywhere — that would be a disabled check, not a stricter one.
    clean = _eval_condition(_closed_vocabulary_cond(), {"record": [{"type_code": "ORD"}]})
    assert clean.result == "pass"


def test_the_same_outsider_reads_as_an_ordinary_non_match_when_nothing_closes_it():
    """The DEFAULT, and it must be byte-identical to what shipped.

    Every declaration that existed when this key was added enumerates a shape or accepts the
    default reading, so the absence of `ordinary_patterns` is not "unset": it is the open
    vocabulary, where an unmatched value is an ordinary non-match and clears. This test is what
    keeps the fix from changing the answer of every condition that never asked for it.
    """
    from src.correlation import _eval_condition

    rows = {"record": [{"type_code": ["ORD", "QQQ"]}]}
    cond = _closed_vocabulary_cond()
    cond.pop("ordinary_patterns")
    assert _eval_condition(cond, rows).result == "pass"
    # An EMPTY list is the same as absent — a pack may leave the key behind while removing its
    # entries, which must not become a third state that withholds on every value.
    assert _eval_condition({**cond, "ordinary_patterns": []}, rows).result == "pass"


def test_an_outsider_blocks_the_clear_without_inconclusive_blocks_pass():
    """The two withholdings are NOT the same key, and the difference is not an oversight.

    An enumerated drop has two legitimate meanings — the value is a third party's, or it is this
    subject's and unreadable — and only the pack knows which, so `inconclusive_blocks_pass`
    exists to say. A value outside every list has ONE meaning: the pack said nothing about it.
    There is no reading in which the survivors are the whole of the evidence, so this bucket
    withholds unconditionally and needs no second declaration to switch it on.
    """
    from src.correlation import _eval_condition

    rows = {"record": [{"type_code": ["ORD", "QQQ"]}]}
    for flag in (None, False, True):
        cond = _closed_vocabulary_cond()
        if flag is not None:
            cond["inconclusive_blocks_pass"] = flag
        assert _eval_condition(cond, rows).result == "unknown", flag


def test_the_two_buckets_are_reported_separately_because_the_remedies_differ():
    """Both drops at once: each value named under its own bucket, each with its own sentence.

    A single count would send the reader to the wrong remedy — an authority for a code the pack
    knows, versus an extension of what the pack knows. So `ZZZ` prints as an enumerated drop and
    `QQQ` as an outsider, and the pack's `inconclusive_detail` is not reused as the reason for a
    value it never mentioned.
    """
    from src.correlation import _eval_condition

    check = _eval_condition(
        _closed_vocabulary_cond(inconclusive_blocks_pass=True),
        {"record": [{"type_code": ["ORD", "ZZZ", "QQQ"]}]},
    )
    assert check.result == "unknown"
    assert "ZZZ" in check.observed and "QQQ" in check.observed, check.observed
    # The bucket marker sits on the outsider only — `ZZZ` is a code the pack DID classify.
    assert check.observed.index("ZZZ") < check.observed.index("QQQ"), check.observed
    assert "meaning is not established" in check.detail, check.detail
    assert "declared vocabulary does not cover" in check.detail, check.detail

    # And when EVERYTHING is dropped there is nothing to compare, so the same two sentences
    # arrive without a survivor count rather than falling through to a clear on zero values.
    all_dropped = _eval_condition(
        _closed_vocabulary_cond(),
        {"record": [{"type_code": ["ZZZ", "QQQ"]}]},
    )
    assert all_dropped.result == "unknown"
    assert "QQQ" in all_dropped.observed


def test_the_enumerated_drop_wording_is_unchanged_by_the_new_bucket():
    """A regression on the STRING, because the report is where this kind is read.

    The enumerated bucket's sentence shipped before the outsider bucket existed and a shared
    formatter is exactly where it would quietly acquire the new one's wording. With no
    `ordinary_patterns` declared the observed string must still read as it did.
    """
    from src.correlation import _eval_condition

    check = _eval_condition(
        _coded_attribute_cond(inconclusive_blocks_pass=True),
        {"record": [{"status_code": ["OTH", "ZZZ"]}]},
    )
    assert check.result == "unknown"
    assert "ZZZ (unclassifiable, beside 1 value(s) that decide nothing either way)" in (
        check.observed
    ), check.observed
    assert "outside every classification" not in check.observed
    assert "declared vocabulary" not in check.detail


def test_a_forbidden_match_beside_an_outsider_still_fails():
    """Closing the vocabulary withholds a CLEAR and never a finding.

    A value nobody classified cannot argue against a match that was found: the match is in the
    data whatever sits beside it. This is the property that keeps the key from becoming a way to
    bury a finding under a code the pack simply never listed.
    """
    from src.correlation import _eval_condition

    check = _eval_condition(
        _closed_vocabulary_cond(), {"record": [{"type_code": ["ENT", "QQQ"]}]}
    )
    assert check.result == "fail"
    assert "ENT" in check.observed


def test_the_closed_vocabulary_holds_in_allowed_mode_too():
    """`allowed` reaches its clear by the opposite route and needs the same withholding.

    In `forbidden` mode a clear means nothing matched; in `allowed` mode it means everything read
    matched — a claim a record holding an unclassified value breaks more directly. Both clear
    returns are covered here, since the second is the one a fix implemented on one branch forgets.
    """
    from src.correlation import _eval_condition

    allow = _closed_vocabulary_cond(match_mode="allowed", patterns=["^OK$"])
    mixed = _eval_condition(allow, {"record": [{"type_code": ["OK", "QQQ"]}]})
    assert mixed.result == "unknown"
    assert "outside every classification" in mixed.observed

    # A value the pack declared ORDINARY is still not on the allow-list, so it fails — closing
    # the vocabulary classifies a value, it does not excuse one.
    assert (
        _eval_condition(allow, {"record": [{"type_code": ["OK", "ORD"]}]}).result == "fail"
    )
    assert _eval_condition(allow, {"record": [{"type_code": "OK"}]}).result == "pass"


def test_a_closed_vocabulary_is_case_insensitive_like_every_other_comparison():
    """All three lists are compared with `re.IGNORECASE`, and the third must not differ.

    A dead helper omitting those flags used to sit in this evaluator, so the failure mode is
    concrete: one branch folding case while the next does not, which produces a well-formed
    verdict and no output distinguishes it.
    """
    from src.correlation import _eval_condition

    lower = _eval_condition(
        _closed_vocabulary_cond(), {"record": [{"type_code": "ord"}]}
    )
    assert lower.result == "pass", lower.observed


# cohort_membership: three silent failure modes, one test each. Without a discriminator every
# subject matches its own row. Truncation is asymmetric: a match found survives a cap, a
# non-match does not. The subject's own rows must be locatable in the cohort.


def _cohort_rows(*triples):
    """(record, creation_date, party_ref) rows as the org_unit sweep returns them."""
    return [{"record": p, "creation_date": d, "party_ref": u} for p, d, u in triples]


def _pax_history_cond():
    """Mirrors the SCHEME ruleset's `party_has_history`, with the subject already filled
    in (the engine substitutes `from_entity` -> the subject value in evaluate_verdict).
    """
    return {
        "id": "party_has_history",
        "label": "Party has no other records in this org_unit on other dates",
        "kind": "cohort_membership",
        "source": "pax_history",
        "subject_scope": False,
        "key_fields": ["party_ref", "p.party_ref"],
        "discriminator": "record",
        "subject_rows": {
            "where": [{"field": "record", "any_of": ["SUBJ02"], "match": "exact"}]
        },
        "expected_label": "no other record in this org_unit carries this party",
        "fail_detail": "repeat customer of this agency",
        "pass_detail": "no prior order history here",
        "truncated_detail": "cohort cut off at its row cap",
        "decisive": False,
    }


def test_cohort_membership_finds_history_on_another_record():
    """The FAIL direction: a shared party id on a different record is the §3.3 exclusion."""
    from src.correlation import _eval_condition

    cond = _pax_history_cond()
    rows = _cohort_rows(
        ("SUBJ02", "2026-07-01", "310C2673007500A1"),
        ("7952H8", "2026-06-10", "310C2673007500A1"),
    )
    chk = _eval_condition(cond, {"pax_history": rows})
    assert chk.result == "fail"
    # The report has to name WHICH other record, or the finding cannot be checked.
    assert "7952H8" in chk.observed


def test_cohort_membership_subject_own_rows_are_not_history():
    """WITHOUT the discriminator every subject matches itself. The subject's own record — even
    across several party rows — must never read as history."""
    from src.correlation import _eval_condition

    cond = _pax_history_cond()
    rows = _cohort_rows(
        ("SUBJ02", "2026-07-01", "310C2673007500A1"),
        ("SUBJ02", "2026-07-01", "310C2673007500A2"),
        ("7952H8", "2026-06-10", "310F7675004EE20F"),
    )
    chk = _eval_condition(cond, {"pax_history": rows})
    assert chk.result == "pass"
    assert "no prior order history" in chk.detail
    # And the guard is really the discriminator, not luck: drop it and the check can no
    # longer tell "another record" from "this one", so it must refuse to answer.
    blind = {k: v for k, v in cond.items() if k != "discriminator"}
    assert _eval_condition(blind, {"pax_history": rows}).result == "unknown"


def test_cohort_membership_truncation_is_asymmetric():
    """A cap-limited miss is `unknown`; a cap-limited HIT is still a fail."""
    from src.correlation import _eval_condition

    cond = _pax_history_cond()
    miss = _cohort_rows(
        ("SUBJ02", "2026-07-01", "310C2673007500A1"),
        ("7952H8", "2026-06-10", "310F7675004EE20F"),
    )
    # Same rows, same absence of a match — only the truncation flag differs.
    assert _eval_condition(cond, {"pax_history": miss}).result == "pass"
    trunc = _eval_condition({**cond, "_cohort_truncated": True}, {"pax_history": miss})
    assert trunc.result == "unknown"
    assert "row cap" in trunc.detail
    # Finding the key survives truncation — the row came back, so the history is real.
    hit = _cohort_rows(
        ("SUBJ02", "2026-07-01", "310C2673007500A1"),
        ("7952H8", "2026-06-10", "310C2673007500A1"),
    )
    assert (
        _eval_condition(
            {**cond, "_cohort_truncated": True}, {"pax_history": hit}
        ).result
        == "fail"
    )


def test_cohort_membership_needs_the_subject_inside_the_cohort():
    """No subject rows => no left-hand side. `unknown`, and distinguishable from no rows.

    THREE reasons for the same `unknown`, and each must read differently. Two of them used
    to share the note "subject key absent", which sent a live investigation (job 7d3eea1e)
    looking at the key when the actual cause was the cohort's SCOPE — it was built around
    the org_unit the alert names, and the subject record was created by a different one.
    """
    from src.correlation import _eval_condition

    cond = _pax_history_cond()
    absent = _eval_condition(
        cond,
        {"pax_history": _cohort_rows(("7952H8", "2026-06-10", "310F7675004EE20F"))},
    )
    assert absent.result == "unknown"
    # A scope error, named as one: the subject is not in the comparison set at all.
    assert "NOT IN the cohort" in absent.observed
    # And it must not read as a finding about the party — an exclusion that reports
    # "no history" here would be asserting a fact from the absence of a comparison.
    assert "not evidence of no history" in absent.detail

    empty = _eval_condition(cond, {"pax_history": []})
    assert empty.result == "unknown"
    # Different reasons must read differently, or the operator cannot tell a window that
    # missed the record from a source that returned nothing.
    assert empty.observed != absent.observed

    # The third: the subject IS in the cohort, but the query never returned the identity
    # leaf. A projection gap, not a fact about the party — and not the same as a scope
    # error, because the fix is in a different place.
    no_key = _eval_condition(
        cond,
        {
            "pax_history": [
                {"record": "SUBJ02", "creation_date": "2026-07-04"},
                {"record": "7952H8", "creation_date": "2026-06-10"},
            ]
        },
    )
    assert no_key.result == "unknown"
    assert "key fields are empty" in no_key.observed
    assert no_key.observed != absent.observed


def _anchor_spec(anchor_entity="record"):
    """A ruleset whose cohort check is keyed on the SUBJECT and anchored on another entity.

    The shape `_pax_history_cond` cannot reach: there, the subject entity and the entity
    that locates the subject's rows are the same, so substituting the subject value for
    every `from_entity` clause happens to be right. Here the cohort is a bag of ONE party's
    rows across many records, the key IS the party, and the anchor is the RECORD under
    adjudication — the only clause value that can point at "the subject's own rows".
    """
    return {
        "label_scheme": "scheme",
        "subject_entity": "party_ref",
        "labels": {
            "fraud": "VALID FRAUD",
            "false_positive": "FALSE POSITIVE",
            "insufficient": "INSUFFICIENT DATA",
        },
        "sources": {"pax_history": "pax_history"},
        "conditions": [
            {
                "id": "party_has_history",
                "label": "Party has no other records on other dates",
                "kind": "cohort_membership",
                "source": "pax_history",
                "subject_scope": False,
                "key_fields": ["party_ref"],
                "discriminators": ["record", "creation_date"],
                "subject_rows": {
                    "where": [
                        {
                            "field": "record",
                            "from_entity": anchor_entity,
                            "match": "exact",
                        }
                    ]
                },
                "fail_detail": "the party is on other records too",
                "pass_detail": "no other record carries this party",
                "decisive": False,
            }
        ],
    }


def test_a_subject_anchor_resolves_from_the_entity_the_clause_NAMES():
    """`subject_rows.where[].from_entity` is filled by TYPE, not by "the subject, always".

    The fill used to substitute the subject value for every clause whatever `from_entity`
    named. That is right when the named type IS the subject's, and silently wrong when it is
    not: a cohort keyed on the subject has to be anchored by the RECORD under adjudication,
    and comparing the subject's identity against a record column matches no row — so the
    engine reported `subject NOT IN the cohort`, a scope error blaming the pack's sweep for a
    value the verdict loop itself supplied. Measured live on two referrals of one procedure:
    the anchor row was present in both (49 and 25 distinct records) and both read `unknown`.
    """
    from src.correlation import evaluate_verdict

    rows = _cohort_rows(
        ("SUBJ02", "2026-07-01", "PARTY-1"),
        ("7952H8", "2026-06-10", "PARTY-1"),
    )
    v = evaluate_verdict(
        _anchor_spec(),
        {"pax_history": rows},
        _understanding(
            [
                ExtractedEntity(type="party_ref", value="PARTY-1"),
                ExtractedEntity(type="record", value="SUBJ02"),
            ]
        ).analysis,
    )
    chk = next(c for c in v.subjects[0].checks if c.id == "party_has_history")
    # The comparison HAPPENED — which is the whole assertion. Whether it fails or passes is
    # the pack's business; that it could not be attempted was the defect.
    assert chk.result == "fail", chk.observed
    assert "7952H8" in chk.observed
    assert "NOT IN the cohort" not in chk.observed


def test_an_unfillable_subject_anchor_is_unknown_and_never_a_clear():
    """A clause naming an entity the incident never supplied must not clear the subject.

    An empty `any_of` is SKIPPED by `apply_where` — right everywhere else, because an
    incomplete declaration must not silently empty a row set. Here that default inverts:
    every cohort row becomes "the subject's own", nothing is left to compare, and an
    exclusion phrased "has no other records" returns the exculpatory PASS on a cohort that
    contains the answer. So the gap is named instead.
    """
    from src.correlation import evaluate_verdict

    rows = _cohort_rows(
        ("SUBJ02", "2026-07-01", "PARTY-1"),
        ("7952H8", "2026-06-10", "PARTY-1"),
    )
    v = evaluate_verdict(
        _anchor_spec(anchor_entity="order_ref"),  # never extracted
        {"pax_history": rows},
        _understanding([ExtractedEntity(type="party_ref", value="PARTY-1")]).analysis,
    )
    chk = next(c for c in v.subjects[0].checks if c.id == "party_has_history")
    assert chk.result == "unknown"
    assert "order_ref" in chk.observed
    # And it must not read as a fact about the party, the way `pass_detail` would have.
    assert "no other record carries this party" not in chk.detail
    assert "NOT a finding about the subject" in chk.detail


def test_a_cohort_check_with_no_subject_selector_is_unknown_and_never_a_clear():
    """A selector never declared fails the same as one that could not be filled.

    Without a `where` clause, `_side_rows` returns all rows unchanged, so every cohort row
    becomes the subject's own, every row is dropped, and the check reports the pack's
    `pass_detail` over a cohort that holds the answer. Three spellings of "no selector" are
    asserted equivalent: key absent, empty `where`, and a clause with no `field`.
    """
    from src.correlation import _eval_condition

    rows = {
        "pax_history": _cohort_rows(
            ("SUBJ02", "2026-07-01", "PARTY-1"),
            ("7952H8", "2026-06-10", "PARTY-1"),
        )
    }
    declared = _eval_condition(_pax_history_cond(), rows)
    assert declared.result == "fail"  # the answer the cohort really holds

    for label, sel in (
        ("absent", None),
        ("empty", {"where": []}),
        ("fieldless", {"where": [{"any_of": ["SUBJ02"], "match": "exact"}]}),
    ):
        cond = _pax_history_cond()
        if sel is None:
            cond.pop("subject_rows")
        else:
            cond["subject_rows"] = sel
        chk = _eval_condition(cond, rows)
        assert chk.result == "unknown", label
        assert "no subject selector declared" in chk.observed, label
        # Not the pack's clear, and not a claim about the party either way.
        assert "no prior order history here" not in chk.detail, label
        assert "NOT a finding about the subject" in chk.detail, label

    # And it stays distinguishable from the two neighbouring reasons, because each is fixed
    # somewhere else: an empty source is a retrieval gap, a subject outside the cohort is a
    # scope error, and this one is the ruleset.
    empty = _eval_condition({k: v for k, v in _pax_history_cond().items()}, {"pax_history": []})
    no_sel = _eval_condition(
        {k: v for k, v in _pax_history_cond().items() if k != "subject_rows"}, rows
    )
    out_of_scope = _eval_condition(
        _pax_history_cond(),
        {"pax_history": _cohort_rows(("7952H8", "2026-06-10", "PARTY-1"))},
    )
    assert len({empty.observed, no_sel.observed, out_of_scope.observed}) == 3


def test_velocity_count_burst():
    """velocity_count FAILs when one actor exceeds max distinct subjects."""
    from src.correlation import _eval_condition

    cond = {
        "id": "vel",
        "kind": "velocity_count",
        "polarity": "fraud_indicator",
        "source": "record",
        "subject_field": "locator.red",
        "actor_field": "creator.sign.red",
        "max": 1,
    }
    rows = {
        "record": [
            {"locator": {"red": "A"}, "creator": {"sign": {"red": "0303CD"}}},
            {"locator": {"red": "B"}, "creator": {"sign": {"red": "0303CD"}}},
        ]
    }
    assert _eval_condition(cond, rows).result == "fail"


@pytest.mark.asyncio
async def test_analyze_attaches_verdict_from_pack():
    """CorrelationModule.analyze attaches result.verdict when the pack ships a ruleset."""
    pack = MagicMock()
    pack.ruleset_spec.return_value = _scheme_spec()
    pack.source.return_value = None
    pack.correlation_specs.return_value = []
    pack.field_priors_for.return_value = []
    pack.entities = []
    llm = MagicMock()
    llm.structured_output = AsyncMock()
    module = CorrelationModule({}, llm, knowledge_pack=pack)
    result = await module.analyze(
        _fraud_logs(), _understanding([ExtractedEntity(type="record", value="SUBJ03")])
    )
    assert result.verdict is not None
    assert result.verdict.subjects[0].verdict == "VALID FRAUD"


@pytest.mark.asyncio
async def test_analyze_no_verdict_without_pack():
    llm = MagicMock()
    llm.structured_output = AsyncMock()
    module = CorrelationModule({}, llm)  # no pack
    result = await module.analyze(
        _fraud_logs(), _understanding([ExtractedEntity(type="record", value="SUBJ03")])
    )
    assert result.verdict is None


# --- Databricks struct-leaf alias resolution (projection collision fix) ------


def test_resolve_path_underscore_alias_scalar():
    # locator.red / creator.sign.red aliased to underscore columns (unique, no collision).
    row = {
        "locator_red": "SUBJ03",
        "creator_sign_red": "0201GPSU",
        "element_counters_AUX": "7",
    }
    assert resolve_path(row, "locator.red") == ["SUBJ03"]
    assert resolve_path(row, "creator.sign.red") == ["0201GPSU"]
    assert resolve_path(row, "element_counters.AUX") == ["7"]


def test_resolve_path_underscore_alias_with_json_nested_tail():
    # route.air aliased to route_air holding a JSON struct; a nested tail
    # (.board_point) descends into the parsed JSON.
    row = {"route_air": '{"board_point":"DSS","off_point":"CMN"}'}
    assert resolve_path(row, "route.air.board_point") == ["DSS"]
    assert resolve_path(row, "route.air.off_point") == ["CMN"]


def test_resolve_path_alias_does_not_break_flat_or_nested():
    # No regression: plain flat-dotted and nested shapes still resolve.
    assert resolve_path({"actor.id": "A"}, "actor.id") == ["A"]
    assert resolve_path({"actor": {"id": "A"}}, "actor.id") == ["A"]
    # A dotted path with no alias and no nested match yields nothing (not an error).
    assert resolve_path({"other": "x"}, "a.b.c") == []


# --- use-case brief attachment (best-effort) --------------------------------


def _scheme_pack():
    """A minimal KnowledgePack carrying a SCHEME verdict ruleset for analyze() to use."""
    from src.knowledge.pack import KnowledgePack

    pack = KnowledgePack(name="t")
    pack.rulesets = {
        "verdicts": {
            "scheme": {
                "label_scheme": "scheme",
                "subject_entity": "record",
                "labels": {
                    "fraud": "VALID FRAUD",
                    "false_positive": "FALSE POSITIVE",
                    "insufficient": "INSUFFICIENT DATA",
                },
                "sources": {"record": "record_lake"},
                "conditions": [
                    {
                        "id": "bare",
                        "label": "Bare record",
                        "kind": "element_absence",
                        "source": "record",
                        "counters": ["element_counters.AUX"],
                        "decisive": True,
                    }
                ],
                "lock_target": {
                    "source": "record",
                    "scope_field": "creator.org_unit_id",
                    "identity_field": "creator.sign.red",
                },
            }
        }
    }
    return pack


@pytest.mark.asyncio
async def test_analyze_attaches_brief_best_effort():
    llm = MagicMock()
    llm.structured_output = AsyncMock()  # high-volume gate keeps LLM unused here
    module = CorrelationModule({}, llm, knowledge_pack=_scheme_pack())
    logs = {
        "record_lake": [
            {
                "locator.red": "SUBJ03",
                "creator.sign.red": "0201GPSU",
                "creator.org_unit_id": "ORG2428D4",
                "element_counters.AUX": 3,  # not bare -> FALSE POSITIVE
            }
        ]
    }
    result = await module.analyze(
        logs, _understanding([ExtractedEntity(type="record", value="SUBJ03")])
    )
    assert result.brief is not None
    # Duck-type across the flat/src. import boundary.
    assert result.brief.use_case == "scheme"
    assert "FALSE POSITIVE" in result.verdict.summary
    assert any(c.id == "bare" for c in result.brief.decisive_fails)


@pytest.mark.asyncio
async def test_the_unselected_incident_is_adjudicated_by_the_DECLARED_default():
    """No playbook matched, so both the verdict and the brief fall back — to the same key.

    The fallback used to be the first ruleset in declaration order, which is use-case
    DIRECTORY order, so adding a procedure under an alphabetically earlier name silently
    re-pointed it. Two procedures in one domain read the same sources, so the newcomer's
    conditions resolve against real rows: not a degrade to no-verdict but a confident wrong
    verdict under the wrong labels, and a brief carrying the wrong procedure's concepts.

    Asserted on the verdict SUMMARY and the brief's `use_case` together, because they are two
    call sites that must not be able to disagree — the brief is what the report narrates from.
    """
    pack = _scheme_pack()
    # An alphabetically EARLIER procedure over the same subject and the same source, whose
    # one condition would reach the opposite conclusion on these rows.
    pack.rulesets["verdicts"] = {
        "aardvark": {
            "label_scheme": "aardvark",
            "subject_entity": "record",
            "labels": {
                "fraud": "AARDVARK POSITIVE",
                "false_positive": "AARDVARK NEGATIVE",
                "insufficient": "INSUFFICIENT DATA",
            },
            "sources": {"record": "record_lake"},
            "conditions": [
                {
                    "id": "present",
                    "label": "Servicing elements present",
                    "kind": "element_presence",
                    "source": "record",
                    "counters": ["element_counters.AUX"],
                    "decisive": True,
                }
            ],
        },
        **pack.rulesets["verdicts"],
    }
    assert pack.ruleset_keys() == [
        "aardvark",
        "scheme",
    ]  # the shape that used to decide
    pack.rulesets["default_ruleset"] = "scheme"

    llm = MagicMock()
    llm.structured_output = AsyncMock()
    module = CorrelationModule({}, llm, knowledge_pack=pack)
    logs = {
        "record_lake": [
            {
                "locator.red": "SUBJ03",
                "creator.sign.red": "0201GPSU",
                "creator.org_unit_id": "ORG2428D4",
                "element_counters.AUX": 3,
            }
        ]
    }
    result = await module.analyze(
        logs, _understanding([ExtractedEntity(type="record", value="SUBJ03")])
    )
    assert result.brief.use_case == "scheme"
    assert "FALSE POSITIVE" in result.verdict.summary  # scheme's vocabulary
    assert "AARDVARK" not in result.verdict.summary
    assert [c.id for c in result.verdict.subjects[0].checks] == ["bare"]


@pytest.mark.asyncio
async def test_analyze_brief_none_on_analyzer_error(monkeypatch):
    # If the analyzer raises, analyze still returns (brief stays None), never crashes.
    # correlation.py does a flat `from usecases.registry import get_analyzer`, so patch
    # the flat module identity (not src.usecases.registry).
    import usecases.registry as reg

    def _boom(*a, **k):
        raise RuntimeError("analyzer down")

    monkeypatch.setattr(reg, "get_analyzer", _boom)
    llm = MagicMock()
    llm.structured_output = AsyncMock()
    module = CorrelationModule({}, llm, knowledge_pack=_scheme_pack())
    logs = {"record_lake": [{"locator.red": "X", "element_counters.AUX": 0}]}
    result = await module.analyze(
        logs, _understanding([ExtractedEntity(type="record", value="X")])
    )
    assert result.brief is None  # degraded gracefully


# ── a decisive note must state the FINDING, not the requirement it broke ──────


def _automation_spec_phrased():
    """The shipped ruleset's wording for the automated exclusion."""
    spec = _automation_spec()
    cond = next(c for c in spec["conditions"] if c["id"] == "not_automated_ref")
    cond["label"] = "Issuance user is not automated"
    cond["expected_label"] = (
        "the acting identity is not on the AUTOMATED reference list"
    )
    cond["fail_detail"] = (
        "the acting (org_unit, sign) pair is listed with profile AUTOMATED — an automated "
        "process issued this document, which §3.2 excludes"
    )
    return spec


def _automation_notes(spec):
    logs = _fraud_logs()
    logs["automation_registry"] = [
        {"orgUnitId": "QQQ1R17GH", "sign": "6009JJ", "profile": "AUTOMATED"}
    ]
    v = evaluate_verdict(spec, logs, _AutomationAnalysis())
    return v.subjects[0].notes


def test_a_failed_exclusion_is_reported_as_what_was_found_not_its_negation():
    """The live defect, verbatim from the report of job 047a603c:

        DECISIVE CONDITION(S): Issuance identity is not a known AUTOMATED account (AUTOMATED);
        Issuance user is not automated (value.payload.userInfo.robot=True)

    Read as prose that says the actor was HUMAN, under a verdict resting on it being an
    automation. No LLM is involved — the string is assembled here, from the condition
    LABEL, which states the REQUIREMENT. A FAIL means the label's negation is what was
    found, so the label is exactly the wrong half of the pair to print.
    """
    notes = _automation_notes(_automation_spec_phrased())
    note = next(n for n in notes if n.startswith("decisive_exclusion="))
    named = note.split("decisive_exclusion=")[-1]
    assert "profile AUTOMATED" in named, named
    # The requirement's phrasing must be GONE, not merely accompanied by the finding: a
    # reader stopping at the first clause must not read the opposite of the verdict.
    assert "is not automated" not in named.lower(), named
    assert "not a known automated" not in named.lower(), named


def test_the_finding_falls_back_to_the_engines_own_note_when_the_pack_is_silent():
    """A pack that declares no `fail_detail` must still not print the requirement. The
    evaluator's per-kind note is already phrased as a finding, so it is usable here —
    vague beats reversed."""
    notes = _automation_notes(_automation_spec())  # no fail_detail declared
    named = next(n for n in notes if n.startswith("decisive_exclusion=")).split(
        "decisive_exclusion="
    )[-1]
    assert "not a known automated" not in named.lower(), named
    assert named.strip(), named


def test_an_indicator_keeps_its_label_because_a_fail_affirms_it():
    """The asymmetry is the `polarity`/kind of the condition, not an inconsistency.

    An EXCLUSION's label states a requirement ("is not automated") so a FAIL negates it; a
    fraud INDICATOR's label already states the finding ("Cash form of payment") so a FAIL
    affirms it. Treating both the same is what produced the reversed sentence above; so is
    treating them the same in the other direction.
    """
    logs = _yryuxz_logs()
    v = evaluate_verdict(
        _categorical_spec(), logs, _AnalysisAppIdentity(["SUBJ01"]), None, _PREFIX_DATA
    )
    joined = " | ".join(v.subjects[0].notes)
    assert "fraud_indicators=" in joined, joined
    named = joined.split("fraud_indicators=")[-1].split(" | ")[0]
    # Indicator labels reach the report unchanged — they need no inversion.
    assert named.strip(), joined


# --- cohort_membership: composite keys ------------------------------------------------


def _name_cohort_rows(*quads):
    """(record, creation_date, family_name_token, first_name_token) as the pax sweep returns it.

    The `.orange` leaves are deterministic pseudonyms of the encrypted name version —
    comparable across records, not readable. The family_name token is shared by a family.
    """
    return [
        {
            "record": p,
            "creation_date": d,
            "party_name_token": sn,
            "party_first_name_token": fn,
        }
        for p, d, sn, fn in quads
    ]


def _pax_history_composite_cond():
    """`party_has_history` keyed on the (family_name, given-name) token PAIR."""
    return {
        "id": "party_has_history",
        "label": "Party has no other records in this org_unit on other dates",
        "kind": "cohort_membership",
        "source": "pax_history",
        "subject_scope": False,
        "key_groups": [["party_name_token", "party_first_name_token"]],
        "discriminator": "record",
        "subject_rows": {
            "where": [{"field": "record", "any_of": ["SUBJ02"], "match": "exact"}]
        },
        "expected_label": "no other record in this org_unit carries this party",
        "fail_detail": "repeat customer of this agency",
        "pass_detail": "no prior order history here",
        "decisive": False,
    }


def test_composite_key_does_not_match_a_namesake():
    """A shared FAMILY_NAME is not a shared identity.

    `name.orange` is the family_name token and a family travelling together shares it, so a
    union match over both name fields reports "order history" for any namesake. This is
    an EXCLUSION: firing it wrongly clears a real fraud, which is the expensive direction.
    """
    from src.correlation import _eval_condition

    rows = _name_cohort_rows(
        ("SUBJ02", "2026-07-04", "SN_FAMILY_NAME", "FN_GIVENONE"),  # the subject
        (
            "7952H8",
            "2026-06-10",
            "SN_FAMILY_NAME",
            "FN_GIVENTWO",
        ),  # same family_name, other person
    )
    out = _eval_condition(_pax_history_composite_cond(), {"pax_history": rows})
    assert out.result == "pass", out.observed

    # The union form on the same data is exactly the false exclusion being prevented.
    union = dict(_pax_history_composite_cond())
    union.pop("key_groups")
    union["key_fields"] = ["party_name_token", "party_first_name_token"]
    assert _eval_condition(union, {"pax_history": rows}).result == "fail"


def test_composite_key_matches_the_same_person_on_another_record():
    """Both tokens equal => the same party => real history => FAIL (the exclusion).

    On an EARLIER date: history means a PRIOR order. The same-day case is the opposite
    finding and is pinned separately below.
    """
    from src.correlation import _eval_condition

    rows = _name_cohort_rows(
        ("SUBJ02", "2026-07-04", "SN_FAMILY_NAME", "FN_GIVENONE"),
        (
            "7952H8",
            "2026-06-10",
            "SN_FAMILY_NAME",
            "FN_GIVENONE",
        ),  # same person, earlier
    )
    out = _eval_condition(_pax_history_composite_cond(), {"pax_history": rows})
    assert out.result == "fail"
    assert "7952H8" in out.observed


def test_same_day_sibling_records_are_not_order_history():
    """A second record created the SAME DAY is a co-conspirator order, not history.

    MEASURED on SUBJ02: the subject party's only other appearances anywhere in seven
    months are two records created on the alert's own date — the burst's sibling orders. On
    the locator alone they read as "other records" and this EXCLUSION fires, so the fraud's
    own siblings become the evidence clearing it. `discriminators: [record, creation_date]`
    requires a row to differ on BOTH.
    """
    from src.correlation import _eval_condition

    cond = dict(_pax_history_composite_cond())
    cond.pop("discriminator")
    cond["discriminators"] = ["record", "creation_date"]
    rows = _name_cohort_rows(
        ("SUBJ02", "2026-07-04", "SN_FAMILY_NAME", "FN_GIVENONE"),
        ("9EDRBE", "2026-07-04", "SN_FAMILY_NAME", "FN_GIVENONE"),  # sibling, same day
        ("9F9BIM", "2026-07-04", "SN_FAMILY_NAME", "FN_GIVENONE"),  # sibling, same day
    )
    assert _eval_condition(cond, {"pax_history": rows}).result == "pass"

    # The single-discriminator form on the SAME data is the false exclusion being prevented.
    assert (
        _eval_condition(_pax_history_composite_cond(), {"pax_history": rows}).result
        == "fail"
    )

    # And a genuine prior order still FAILS — the fix must not disable the check.
    out = _eval_condition(
        cond,
        {
            "pax_history": rows
            + _name_cohort_rows(
                ("7952H8", "2026-06-10", "SN_FAMILY_NAME", "FN_GIVENONE")
            )
        },
    )
    assert out.result == "fail"
    assert "7952H8" in out.observed


def test_a_row_missing_a_discriminator_is_not_counted_as_other():
    """An ambiguous row is dropped, not counted — the directions are not symmetric.

    With no creation_date there is no way to tell a prior order from a same-day sibling.
    A wrong FAIL fires an exclusion and clears a real fraud; a wrong skip costs one
    non-decisive corroborator. So the ambiguity resolves toward NOT firing.
    """
    from src.correlation import _eval_condition

    cond = dict(_pax_history_composite_cond())
    cond.pop("discriminator")
    cond["discriminators"] = ["record", "creation_date"]
    rows = _name_cohort_rows(
        ("SUBJ02", "2026-07-04", "SN_FAMILY_NAME", "FN_GIVENONE")
    ) + [
        {
            "record": "7952H8",  # a different record, but its date did not come back
            "party_name_token": "SN_FAMILY_NAME",
            "party_first_name_token": "FN_GIVENONE",
        }
    ]
    assert _eval_condition(cond, {"pax_history": rows}).result == "pass"


def test_singular_discriminator_still_works():
    """Every existing check declares `discriminator` (singular) and must be untouched."""
    from src.correlation import _eval_condition

    cond = _pax_history_composite_cond()
    assert "discriminators" not in cond
    rows = _name_cohort_rows(
        ("SUBJ02", "2026-07-04", "SN_FAMILY_NAME", "FN_GIVENONE"),
        ("7952H8", "2026-06-10", "SN_FAMILY_NAME", "FN_GIVENONE"),
    )
    assert _eval_condition(cond, {"pax_history": rows}).result == "fail"
    # And a cohort holding only the subject still passes rather than matching itself.
    assert (
        _eval_condition(
            cond,
            {
                "pax_history": _name_cohort_rows(
                    ("SUBJ02", "2026-07-04", "SN_FAMILY_NAME", "FN_GIVENONE")
                )
            },
        ).result
        == "pass"
    )


def _docs_cond():
    """`identity_citizenship_mismatch`: DOCS position 1 (issuer) vs 3 (citizenship).

    The layout is SCHEME_CODE-positional, verbatim from live data:
        P/NGA/A11111111/NGA/01JAN10/F/01JAN25/FAMILY_NAME ONE/GIVENONE
        0:type 1:issuer 2:number 3:citizenship 4:dob 5:sex 6:expiry 7:family_name 8:given
    """
    return {
        "id": "identity_citizenship_mismatch",
        "label": "Party passport issuer matches stated citizenship",
        "kind": "delimited_field_mismatch",
        "source": "record",
        "fields": ["aux_docs_text"],
        "separator": "/",
        "left_index": 1,
        "right_index": 3,
        "equivalent_values": [["NG", "NGA"], ["US", "USA"], ["GB", "GBR"]],
        "polarity": "fraud_indicator",
        "decisive": False,
        "expected": "passport issuing state matches the stated citizenship",
        "fail_detail": "issuer and citizenship disagree",
        "pass_detail": "issuer and citizenship agree",
        "absent_detail": "no DOCS line carried both fields",
    }


def _docs(*texts):
    return [{"aux_docs_text": t} for t in texts]


def test_docs_consistent_passport_passes():
    """The subject record's real lines: NGA passport, NGA citizenship, five parties."""
    from src.correlation import _eval_condition

    out = _eval_condition(
        _docs_cond(),
        {
            "record": _docs(
                "P/NGA/A11111111/NGA/01JAN10/F/01JAN25/FAMILY_NAME ONE/GIVENONE",
                "P/NGA/B22222222/NGA/02FEB08/M/02FEB28/FAMILY_NAME ONE/GIVENTWO MIDDLE",
                "P/NGA/B33333333/NGA/03MAR78/F/03MAR32/SURNAMEZ/GIVENTHREE MIDDLE",
            )
        },
    )
    assert out.result == "pass", out.observed


def test_docs_mismatch_is_detected_per_line_not_across_the_set():
    """The pairing IS the finding — a set-overlap comparison would miss this entirely.

    Nationalities {NGA, GBR} and issuers {GBR, NGA} overlap perfectly as SETS while every
    party is mismatched, which is why `value_mismatch` cannot express this check.
    """
    from src.correlation import _eval_condition

    out = _eval_condition(
        _docs_cond(),
        {
            "record": _docs(
                "P/GBR/A11111111/NGA/01JAN10/F/01JAN25/FAMILY_NAME/GIVENONE",
                "P/NGA/B22222222/GBR/02FEB08/M/02FEB28/FAMILY_NAME/GIVENTWO",
            )
        },
    )
    assert out.result == "fail"
    assert "GBR vs NGA" in out.observed or "NGA vs GBR" in out.observed


def test_alpha2_and_alpha3_country_codes_are_not_a_mismatch():
    """`P/NG/.../NGA/...` and `P/US/.../US/...` are BOTH real, self-consistent lines.

    Measured in one org_unit-week. This is a fraud INDICATOR, so a wrong FAIL alleges fraud
    against an ordinary party — an ISO alpha-2/alpha-3 spelling difference must not.
    """
    from src.correlation import _eval_condition

    out = _eval_condition(
        _docs_cond(),
        {
            "record": _docs(
                "P/NG/B44444444/NGA/04APR80/F/04APR35/SURNAMETWO/GIVENFOUR/H",
                "P/US/A55555555/USA/05MAY12/F/05MAY28/SURNAMETHREE/GIVENFIVE/H",
                "P/US/666666666/US/06JUN22/M/06JUN27/SURNAMETHREE/GIVENSIX/H",
            )
        },
    )
    assert out.result == "pass", out.observed


def test_document_less_docs_lines_are_skipped_not_compared():
    """`////07JUL66/M//SURNAMEFOUR/GIVENSEVEN MIDDLE` — name and DOB only, positions 1 and 3 empty.

    This is the MAJORITY form in live data. Comparing empty to empty would silently clear a
    order with no passport on file; comparing populated to empty would allege a mismatch
    on one. Both invent a finding, so the line is skipped and the check says so.
    """
    from src.correlation import _eval_condition

    out = _eval_condition(
        _docs_cond(),
        {
            "record": _docs(
                "////07JUL66/M//SURNAMEFOUR/GIVENSEVEN MIDDLE",
                "////04APR80/F//SURNAMETWO/GIVENFOUR MIDDLE",
            )
        },
    )
    assert out.result == "unknown", out.observed
    assert "both" in out.observed or "no value" in out.observed

    # A populated line alongside document-less ones is still judged on its own.
    out = _eval_condition(
        _docs_cond(),
        {
            "record": _docs(
                "////07JUL66/M//SURNAMEFOUR/GIVENSEVEN MIDDLE",
                "P/GBR/A11111111/NGA/01JAN10/F/01JAN25/FAMILY_NAME/GIVENONE",
            )
        },
    )
    assert out.result == "fail"
    # The note states how many lines were NOT compared — a finding drawn from 1 of 2 lines
    # must not read as if it were drawn from both.
    assert "1 value(s) without both fields" in out.observed


def test_doco_layout_is_not_read_with_docs_positions():
    """A DOCO (visa) line has a DIFFERENT layout and must be excluded by the row selector.

    Live on SUBJ02: `/V/777777777/GBR//GBR//08AUG26`. Read with the DOCS positions, index 1
    is a document number and index 3 is blank — a comparison between unrelated fields. The
    ruleset's `where: aux_code == DOCS` is what prevents it, so this asserts the selector
    is honoured rather than that the parse happens to be safe.
    """
    from src.correlation import _eval_condition

    cond = dict(_docs_cond())
    cond["fields"] = ["text"]
    cond["where"] = [{"field": "aux_code", "any_of": ["DOCS"], "match": "exact"}]
    rows = [
        {"aux_code": "DOCO", "text": "/V/777777777/GBR//GBR//08AUG26"},
        {"aux_code": "DOCS", "text": "P/NGA/A11111111/NGA/01JAN10/F/01JAN25/O/C"},
    ]
    out = _eval_condition(cond, {"record": rows})
    assert out.result == "pass", out.observed
    # Without the selector the DOCO line is parsed too, and it decides nothing correctly.
    cond_no_sel = dict(cond)
    cond_no_sel.pop("where")
    assert _eval_condition(cond_no_sel, {"record": rows}).result != "pass"


def test_aux_entries_nested_in_one_record_row_are_selected_as_records():
    """THE LIVE SHAPE: one record row whose `service.service` is an array of AUX elements.

    Every other test here hands the evaluator pre-flattened per-AUX rows, which the real
    projection does not produce — Databricks returns the sub-struct as an array on a single
    row. Without `records` the `where: code == DOCS` clause looks for `code` on the record row,
    finds nothing, drops the row and the check reports `unknown` about data that is present:
    the same failure the stub had, one layer down. And skipping the clause instead would be
    worse — the DOCO element's text would be read with the DOCS positions.
    """
    from src.correlation import _eval_condition

    cond = dict(_docs_cond())
    cond["records"] = "service.service"
    cond["fields"] = ["free_text.red"]
    cond["where"] = [{"field": "code", "any_of": ["DOCS"], "match": "exact"}]
    row = {
        "locator_red": "SUBJ02",
        "service": {
            "service": [
                {
                    "code": "DOCO",
                    "free_text": {"red": "/V/777777777/GBR//GBR//08AUG26"},
                },
                {"code": "SEAT", "free_text": {"red": "12A"}},
                {
                    "code": "DOCS",
                    "free_text": {
                        "red": "P/NGA/A11111111/NGA/01JAN10/F/01JAN25/FAMILY_NAME/GIVENONE"
                    },
                },
                {
                    "code": "DOCS",
                    "free_text": {
                        "red": "P/GBR/B22222222/NGA/02FEB08/M/02FEB28/FAMILY_NAME/GIVENTWO"
                    },
                },
            ]
        },
    }
    out = _eval_condition(cond, {"record": [row]})
    # Both DOCS entries were reached and only they were: 2 pairs, one of them mismatched.
    assert out.result == "fail", out.observed
    assert "GBR vs NGA" in out.observed, out.observed
    assert "of 2" in out.observed, out.observed
    # The DOCO entry was excluded, not merely outvoted — its number/blank pair would have
    # produced a THIRD comparison had the selector been applied to the wrong unit.
    assert "777777777" not in out.observed

    # A flat underscore alias of the same sub-struct (the projection's other spelling)
    # reaches the same entries, so the check does not depend on which one Databricks emits.
    aliased = {"service_service": row["service"]["service"]}
    assert _eval_condition(cond, {"record": [aliased]}).result == "fail"


def test_parallel_leaf_arrays_are_zipped_back_into_records():
    """THE CHEAPER PROJECTION, which returns the same records TRANSPOSED.

    MEASURED: selecting `service.service` whole timed out at 1500s on one record, so the check
    is served by projecting two of its 45 leaves instead. Databricks returns a leaf selection
    on an array-of-struct as PARALLEL ARRAYS — one per leaf — not as entries. The records are
    all there; only the shape differs, so index i of each array is element i.
    """
    from src.correlation import _eval_condition

    cond = dict(_docs_cond())
    cond["records"] = "service.service"
    cond["fields"] = ["free_text_red"]
    cond["where"] = [{"field": "code", "any_of": ["DOCS"], "match": "exact"}]
    row = {
        "locator_red": "SUBJ02",
        "code": ["DOCO", "SEAT", "DOCS", "DOCS"],
        "free_text_red": [
            "/V/777777777/GBR//GBR//08AUG26",
            "12A",
            "P/NGA/A11111111/NGA/01JAN10/F/01JAN25/FAMILY_NAME/GIVENONE",
            "P/GBR/B22222222/NGA/02FEB08/M/02FEB28/FAMILY_NAME/GIVENTWO",
        ],
    }
    out = _eval_condition(cond, {"record": [row]})
    # The two DOCS elements were paired with THEIR OWN text: 2 comparisons, one mismatch.
    assert out.result == "fail", out.observed
    assert "GBR vs NGA" in out.observed, out.observed
    assert "of 2" in out.observed, out.observed


def test_ragged_leaf_arrays_are_refused_rather_than_mispaired():
    """Unequal lengths cannot be aligned, and guessing attaches one element's type to
    another element's text — a DOCS filter reading a SEAT line with passport positions,
    which produces a confident finding about nothing. `unknown` is the honest answer for a
    shape that cannot be read, and it is the SAFE one for a fraud indicator."""
    from src.correlation import _eval_condition

    cond = dict(_docs_cond())
    cond["records"] = "service.service"
    cond["fields"] = ["free_text_red"]
    cond["where"] = [{"field": "code", "any_of": ["DOCS"], "match": "exact"}]
    ragged = {
        "code": ["DOCO", "SEAT", "DOCS"],
        "free_text_red": [  # one short — the DOCS text is missing, not merely misplaced
            "/V/777777777/GBR//GBR//08AUG26",
            "12A",
        ],
    }
    assert _eval_condition(cond, {"record": [ragged]}).result == "unknown"


def test_exploded_one_row_per_element_is_read_as_records():
    """THE THIRD SHAPE, and the one the pipeline is most likely to send.

    The Databricks projection prompt tells the generator to `LATERAL VIEW OUTER explode(...)`
    any array leaf, so the same records can arrive already flattened — one row per AUX
    element, columns named after the full path (`service_service_code`). Nothing about that
    is wrong, but it is a THIRD shape, and a mechanism that handled only the other two would
    report `unknown` on the query the pipeline actually wrote. The prefix is stripped so the
    ruleset names the leaf as it stands inside the record (`code`), not as the SELECT
    happened to alias it.
    """
    from src.correlation import _eval_condition

    cond = dict(_docs_cond())
    cond["records"] = "service.service"
    cond["fields"] = ["free_text.red"]
    cond["where"] = [{"field": "code", "any_of": ["DOCS"], "match": "exact"}]
    rows = [
        {
            "locator_red": "SUBJ02",
            "service_service_code": c,
            "service_service_free_text_red": t,
        }
        for c, t in [
            ("DOCO", "/V/777777777/GBR//GBR//08AUG26"),
            ("SEAT", "12A"),
            ("DOCS", "P/NGA/A11111111/NGA/01JAN10/F/01JAN25/FAMILY_NAME/GIVENONE"),
            ("DOCS", "P/GBR/B22222222/NGA/02FEB08/M/02FEB28/FAMILY_NAME/GIVENTWO"),
        ]
    ]
    out = _eval_condition(cond, {"record": rows})
    assert out.result == "fail", out.observed
    assert "GBR vs NGA" in out.observed, out.observed
    assert "of 2" in out.observed, out.observed
    # The DOCO number must not have been read with the DOCS positions.
    assert "777777777" not in out.observed, out.observed


def test_records_selection_is_opt_in_and_leaves_row_level_wheres_alone():
    """`records` absent => the row is the record, exactly as before.

    Asserted because this seam is shared by every condition kind that uses `_side_rows`, so
    a change here that quietly re-interpreted flat rows would move checks that work today.
    """
    from src.correlation import _side_rows

    rows = [{"code": "DOCS", "free_text": {"red": "x"}}, {"code": "SEAT"}]
    assert _side_rows({"record": rows}, {"source": "record"}) == rows
    kept = _side_rows(
        {"record": rows},
        {"source": "record", "where": [{"field": "code", "any_of": ["DOCS"]}]},
    )
    assert kept == [rows[0]]
    # A `records` path that resolves to no struct entries yields no rows — not the raw row
    # as a fallback, which would be read with the entry's field names and select nothing.
    assert (
        _side_rows({"record": rows}, {"source": "record", "records": "service.service"})
        == []
    )


def test_missing_docs_field_is_unknown_not_a_clear():
    """No DOCS at all => `unknown`. Measured: only 44% of records carry any DOCS."""
    from src.correlation import _eval_condition

    out = _eval_condition(_docs_cond(), {"record": [{"other_field": "x"}]})
    assert out.result == "unknown"


def test_composite_key_skips_an_incomplete_tuple_rather_than_half_applying_it():
    """A group is usable only when EVERY field in it resolves.

    Half a composite key is a different, looser check than the one declared — the same
    reason `identity_keys` skips a partially-satisfied candidate instead of applying it.
    """
    from src.correlation import _eval_condition

    rows = [
        # Subject: complete tuple.
        {
            "record": "SUBJ02",
            "creation_date": "2026-07-04",
            "party_name_token": "SN_FAMILY_NAME",
            "party_first_name_token": "FN_GIVENONE",
        },
        # Other: family_name only. Must NOT match on the family_name alone.
        {
            "record": "7952H8",
            "creation_date": "2026-06-10",
            "party_name_token": "SN_FAMILY_NAME",
        },
    ]
    assert (
        _eval_condition(_pax_history_composite_cond(), {"pax_history": rows}).result
        == "pass"
    )


def test_key_fields_union_behaviour_is_unchanged_without_key_groups():
    """`key_groups` is additive: every existing check must behave exactly as before."""
    from src.correlation import _eval_condition

    cond = _pax_history_cond()
    assert "key_groups" not in cond
    hit = _eval_condition(
        cond,
        {
            "pax_history": _cohort_rows(
                ("SUBJ02", "2026-07-04", "310C2673007500A2"),
                ("7952H8", "2026-06-10", "310C2673007500A2"),
            )
        },
    )
    assert hit.result == "fail"


def test_composite_key_pairs_multi_valued_fields_positionally():
    """An unexploded row carries every party; each must keep their OWN given name.

    Taking only the first value would key the whole row on party #1 (so parties 2..n
    become invisible to the check); crossing the fields would invent parties who are not
    on the order, and an invented pair can match a namesake elsewhere.
    """
    from src.correlation import _eval_condition

    rows = [
        {
            "record": "SUBJ02",
            "creation_date": "2026-07-04",
            "party_name_token": ["SN_FAMILY_NAME", "SN_FAMILY_NAME"],
            "party_first_name_token": ["FN_GIVENONE", "FN_GIVENTWO"],
        },
        # GRACE — the SECOND subject party — booked before. Real history.
        {
            "record": "7952H8",
            "creation_date": "2026-06-10",
            "party_name_token": "SN_FAMILY_NAME",
            "party_first_name_token": "FN_GIVENTWO",
        },
        # A namesake pair that exists only if the fields are CROSSED, never on any order.
        {
            "record": "8888XX",
            "creation_date": "2026-06-11",
            "party_name_token": "SN_OTHER",
            "party_first_name_token": "FN_GIVENONE",
        },
    ]
    out = _eval_condition(_pax_history_composite_cond(), {"pax_history": rows})
    assert out.result == "fail"
    assert "7952H8" in out.observed  # the real match was found...
    assert "8888XX" not in out.observed  # ...and no crossed pair was invented


def test_key_groups_are_priority_ordered_not_unioned():
    """First fully-satisfied group wins — the `identity_keys` rule, for the same reason.

    The groups are alternative SPELLINGS of one identity (a flat projection alias and the
    raw struct path), not extra identities. If every group were tried and unioned, a looser
    later spelling could widen a match the earlier one already decided.
    """
    from src.correlation import _eval_condition

    cond = _pax_history_composite_cond()
    cond["key_groups"] = [
        ["party_name_token", "party_first_name_token"],  # precise, resolves
        ["loose_token"],  # a 1-field group that would match a namesake
    ]
    rows = [
        {
            "record": "SUBJ02",
            "creation_date": "2026-07-04",
            "party_name_token": "SN_FAMILY_NAME",
            "party_first_name_token": "FN_GIVENONE",
            "loose_token": "SN_FAMILY_NAME",
        },
        {
            "record": "7952H8",
            "creation_date": "2026-06-10",
            "party_name_token": "SN_FAMILY_NAME",
            "party_first_name_token": "FN_GIVENTWO",  # different person
            "loose_token": "SN_FAMILY_NAME",  # would match if groups were unioned
        },
    ]
    assert _eval_condition(cond, {"pax_history": rows}).result == "pass"


# --- the containment target: declared source, declared role words ------------


def _lock_spec_verdict(lock_spec, rows, extra_spec=None, extra_conditions=None):
    """Minimal fraud-reaching ruleset whose only job is to resolve a lock_target."""
    from src.correlation import evaluate_verdict

    spec = {
        "subject_entity": "asset",
        "labels": {"fraud": "FRAUD", "false_positive": "CLEARED"},
        "sources": {"record": "record_source", "other": "other_source"},
        "conditions": [
            {
                "id": "flag_is_clear",
                "kind": "field_flag",
                "source": "record",
                "flag_fields": ["suspicious"],
                "label": "Asset is not flagged",
                "decisive": True,
                "decisive_on": ["fail"],
                "polarity": "fraud_indicator",
            }
        ],
        "lock_target": lock_spec,
        "notification": {"template": "{containment_block}"},
    }
    spec["conditions"] = spec["conditions"] + list(extra_conditions or [])
    spec.update(extra_spec or {})
    analysis = _understanding([ExtractedEntity(type="asset", value="A1")]).analysis
    return evaluate_verdict(spec, rows, analysis)


# --- attribution: a target the incident never named -------------------------------------
# A separate gate from the verdict-label check: whether the containment target belongs to
# this incident at all. A confident label on a stranger passes the label gate cleanly.


def _attribution_verdict(rows, entities, entity_map=None):
    """The lock fixture, with an entity_map binding the scope field and chosen entities."""
    from src.correlation import evaluate_verdict

    spec = {
        "subject_entity": "asset",
        "labels": {"fraud": "FRAUD", "false_positive": "CLEARED"},
        "sources": {"record": "record_source"},
        "conditions": [
            {
                "id": "flag_is_clear",
                "kind": "field_flag",
                "source": "record",
                "flag_fields": ["suspicious"],
                "label": "Asset is not flagged",
                "decisive": True,
                "decisive_on": ["fail"],
                "polarity": "fraud_indicator",
            }
        ],
        "lock_target": {
            "source": "record",
            "scope_field": "site",
            "identity_field": "who",
        },
    }
    analysis = _understanding(entities).analysis
    return evaluate_verdict(spec, rows, analysis, entity_map=entity_map)


def test_a_containment_target_outside_the_incidents_scope_is_withheld_and_degrades():
    """The evidence's scope is not one the incident named -> no target, and degraded.

    The rows are internally consistent and the conditions all resolve; what is wrong is that
    they belong to somebody else. Anchored on the EXTRACTION, because on the live run the
    alert record was located in these same foreign rows, so the reconciled "declared" scope
    WAS the stranger's — a check comparing declared against retrieved compares a value to
    itself and can never fire.
    """
    v = _attribution_verdict(
        {
            "record_source": [
                {"asset": "A1", "suspicious": True, "site": "FOREIGN", "who": "X"}
            ]
        },
        [
            ExtractedEntity(type="asset", value="A1"),
            ExtractedEntity(type="site", value="HOME"),
        ],
        entity_map={"record_source": {"site": "site", "asset": "asset"}},
    )
    s = v.subjects[0]
    assert (
        s.lock_target == {}
    ), "a stranger's scope may not be nominated for containment"
    assert v.degraded is True
    assert any("containment_withheld=" in n for n in s.notes)


def test_a_containment_target_inside_the_incidents_scope_is_kept():
    """The other side: the same shape, the scope the incident DID name -> target stands.

    Without this the guard would be indistinguishable from never nominating a target.
    """
    v = _attribution_verdict(
        {
            "record_source": [
                {"asset": "A1", "suspicious": True, "site": "HOME", "who": "X"}
            ]
        },
        [
            ExtractedEntity(type="asset", value="A1"),
            ExtractedEntity(type="site", value="HOME"),
        ],
        entity_map={"record_source": {"site": "site", "asset": "asset"}},
    )
    s = v.subjects[0]
    assert s.lock_target.get("scope") == "HOME"
    assert not any("containment_withheld=" in n for n in s.notes)


def test_attribution_is_silent_when_the_pack_binds_no_entity_to_the_scope_field():
    """No binding for the scope field means nothing to contradict — not a degradation.

    Every unknown here is a different question than the one this gate asks, and a pack that
    never bound the field must not start reading as mis-attributed. The rows deliberately
    carry a scope the incident did not name: it is the missing BINDING, not agreement, that
    keeps the guard quiet.
    """
    v = _attribution_verdict(
        {
            "record_source": [
                {"asset": "A1", "suspicious": True, "site": "FOREIGN", "who": "X"}
            ]
        },
        [
            ExtractedEntity(type="asset", value="A1"),
            ExtractedEntity(type="site", value="HOME"),
        ],
        entity_map={"record_source": {"asset": "asset"}},
    )
    s = v.subjects[0]
    assert s.lock_target.get("scope") == "FOREIGN"
    assert not any("containment_withheld=" in n for n in s.notes)


def test_the_containment_target_is_read_from_the_declared_logical_source():
    """`lock_target.source` was declared by every ruleset and IGNORED by the engine.

    The logical name `record` was hardcoded, so the block worked only for a ruleset that happened
    to use that name and any other pack silently got `lock_target: {}` — with no error
    anywhere. That is the expensive shape: a fraud verdict nominating NOBODY is exactly what a
    verdict which deliberately withheld containment looks like (the "no target" sentence exists
    for that legitimate case), so a missing containment instruction cannot be told apart from a
    withheld one by the person reading the report.
    """
    v = _lock_spec_verdict(
        {"source": "other", "scope_field": "site", "identity_field": "who"},
        {
            "record_source": [{"asset": "A1", "suspicious": True}],
            # The identity is on the OTHER source, which is the one the ruleset declares.
            # It carries the subject key because these rows go through the same
            # subject-scoping as every other source's.
            "other_source": [{"asset": "A1", "site": "DEPOT-9", "who": "OP-42"}],
        },
    )
    lt = v.subjects[0].lock_target
    assert lt.get("scope") == "DEPOT-9" and lt.get("identity") == "OP-42"
    assert lt.get("source") == "other_source"  # the REAL source name, for the report


def test_the_containment_block_names_no_domain_when_the_pack_declares_no_words():
    """The engine's fallbacks name the two generic ROLES it resolved, and nothing else.

    `§4.1.1`, `OrgUnit:`, `Sign:` and "the CREATOR, not the issuance agent" were engine
    literals in the single most consequential paragraph of the report — the one naming a real
    account for action, where a clause citation reads as authoritative and cannot be checked
    against anything.
    """
    v = _lock_spec_verdict(
        {"source": "record", "scope_field": "site", "identity_field": "who"},
        {
            "record_source": [
                {"asset": "A1", "suspicious": True, "site": "D9", "who": "OP42"}
            ]
        },
    )
    block = v.notification_draft
    assert "D9" in block and "OP42" in block
    for term in ("§", "OrgUnit", "Sign", "record", "order", "issuance", "agent"):
        assert term not in block, f"engine containment block names a domain: {term!r}"


def test_the_pack_declares_the_containment_blocks_own_wording():
    """And a pack that DOES declare gets its words verbatim, including a clause number."""
    v = _lock_spec_verdict(
        {
            "source": "record",
            "scope_field": "site",
            "identity_field": "who",
            "heading": "Operator identified (§9.2 — the DISPATCHER, not the driver):",
            "scope_label": "Depot",
            "identity_label": "Operator",
            "provenance_label": "Read from",
        },
        {
            "record_source": [
                {"asset": "A1", "suspicious": True, "site": "D9", "who": "OP42"}
            ]
        },
    )
    block = v.notification_draft
    assert "Operator identified (§9.2 — the DISPATCHER, not the driver):" in block
    assert "Depot: D9" in block
    assert "Operator: OP42" in block
    # The provenance reuses the SAME declared words, lower-cased, so the block and the inline
    # provenance cannot disagree about what the two fields are.
    assert "depot from site" in block and "operator from who" in block
    # ...and the labels ride on the target itself, for the report and brief consumers.
    lt = v.subjects[0].lock_target
    assert lt.get("scope_label") == "Depot" and lt.get("identity_label") == "Operator"


def test_a_verdict_with_no_nominated_target_says_so_in_one_line():
    """A stack of empty labels reads as missing data on a case that named nobody."""
    v = _lock_spec_verdict(
        {"source": "record", "scope_field": "site", "identity_field": "who"},
        {"record_source": [{"asset": "A1", "suspicious": True}]},
    )
    assert not v.subjects[0].lock_target
    assert "nominated" in v.notification_draft.lower()
    assert "§" not in v.notification_draft


_CLEARING_CONDITION = {
    "id": "was_reviewed",
    "kind": "field_flag",
    "source": "record",
    "flag_fields": ["reviewed"],
    # The label states the REQUIREMENT ("was not reviewed"), so `expected: False` is the
    # passing state and a FAIL — the flag IS set — negates the label and clears the case.
    "expected": False,
    "label": "Record was not manually reviewed",
    "decisive": True,
    "decisive_on": ["fail"],
    "exclusion_kind": "categorical",
}


def test_a_withheld_containment_note_carries_the_packs_own_sentence():
    """The identity a verdict resolved but declined to act on.

    Read by exactly the person most likely to act on it anyway, so the sentence has to say what
    ROLE the identity played in the record — which is why it is a whole declared sentence rather
    than a template assembled from the two field labels. Assembling it read "the sign ... as the
    org_unit's acting identity": grammatical, generic, and less informative than the shipped text.
    """
    lock_spec = {
        "source": "record",
        "scope_field": "site",
        "identity_field": "who",
        "withheld_detail": "The dispatcher ({identity}) opened the record, not the driver.",
    }
    # `suspicious` clear -> the indicator PASSES -> no fraud -> containment is withheld.
    v = _lock_spec_verdict(
        lock_spec,
        {
            "record_source": [
                {
                    "asset": "A1",
                    "suspicious": False,
                    "reviewed": True,
                    "site": "D9",
                    "who": "OP42",
                }
            ]
        },
        extra_conditions=[_CLEARING_CONDITION],
    )
    s = v.subjects[0]
    assert s.verdict != "FRAUD"
    assert not s.lock_target, "a non-containment verdict must nominate nobody"
    note = next(n for n in s.notes if n.startswith("containment_withheld="))
    assert "The dispatcher (OP42 @ D9) opened the record, not the driver." in note


def test_the_withheld_note_names_no_domain_by_default():
    v = _lock_spec_verdict(
        {"source": "record", "scope_field": "site", "identity_field": "who"},
        {
            "record_source": [
                {
                    "asset": "A1",
                    "suspicious": False,
                    "reviewed": True,
                    "site": "D9",
                    "who": "OP42",
                }
            ]
        },
        extra_conditions=[_CLEARING_CONDITION],
    )
    note = next(n for n in v.subjects[0].notes if n.startswith("containment_withheld="))
    for term in ("consignment", "waybill", "agent", "§"):
        assert term not in note, f"withheld note names a domain: {term!r}"


# --- as-of adjudication: a version log is not read whole ----------------------
#
# An append-per-change source keeps accruing rows after the incident is raised, so
# adjudicating the chain whole reads the responder's containment as the subject's conduct.
# A time-only cut is also wrong: a row the named actor wrote late is that actor's own conduct,
# not a third party's. The rule is the intersection: later AND by an identity not in the alert.


def _as_of_verdict(rows, as_of=None, event=("2026-07-27T16:45:00Z",), actor="OP42"):
    """A three-condition ruleset over one append-per-change source.

    `flagged` is the indicator that makes the subject reachable at all; `one_actor` is an
    exclusion on WHO acted; `authorised` is an exclusion whose evidence may arrive late. All
    three read the same source, so the only variable is which of its versions they were given.

    `actor` is the identity the INCIDENT named (an entity of the type the declaration points
    at) — `None` for an incident that named nobody.
    """
    from src.correlation import evaluate_verdict

    spec = {
        "subject_entity": "asset",
        "labels": {"fraud": "FRAUD", "false_positive": "CLEARED"},
        "sources": {"record": "record_source"},
        "conditions": [
            {
                "id": "flagged",
                "kind": "field_flag",
                "source": "record",
                "flag_fields": ["suspicious"],
                "label": "Asset is flagged",
                "polarity": "fraud_indicator",
            },
            {
                "id": "one_actor",
                "kind": "distinct_count",
                "source": "record",
                "field": "who",
                "max": 1,
                "label": "One identity acted on this asset",
            },
            {
                "id": "authorised",
                "kind": "field_flag",
                "source": "record",
                "flag_fields": ["authorised"],
                "expected": True,
                "label": "The change carries an authorisation",
            },
        ],
    }
    if as_of is not None:
        spec["as_of"] = as_of
    entities = [ExtractedEntity(type="asset", value="A1")]
    if actor:
        entities.append(ExtractedEntity(type="operator", value=actor))
    u = _understanding(entities)
    if event:
        u.analysis.event_time = EventWindow(start=event[0], end=event[-1])
    return evaluate_verdict(spec, {"record_source": rows}, u.analysis)


#: Two versions written BEFORE the alert by the named actor, and one written after by the
#: responder who contained it. Identical shape — that is the point.
_VERSIONS = [
    {
        "asset": "A1",
        "version": "1",
        "who": "OP42",
        "written": "2026-07-27T16:11:07.000Z",
    },
    {
        "asset": "A1",
        "version": "2",
        "who": "OP42",
        "written": "2026-07-27T16:40:00.000Z",
        "suspicious": True,
    },
    {
        "asset": "A1",
        "version": "3",
        "who": "RESPONDER",
        "written": "2026-07-27T18:15:51.000Z",
    },
]
#: Both halves declared, because either alone is a known-wrong rule.
_AS_OF = {
    "record": {
        "timestamp_fields": ["written"],
        "actor_fields": ["who"],
        "actor_entities": ["operator"],
    }
}


def test_a_version_a_third_party_wrote_after_the_incident_is_not_the_subjects_behaviour():
    """The whole point: the responder's row must not make the actor count 2."""
    whole = _as_of_verdict(_VERSIONS)
    scoped = _as_of_verdict(_VERSIONS, as_of=_AS_OF)
    assert (
        next(c for c in whole.subjects[0].checks if c.id == "one_actor").result
        == "fail"
    )
    assert (
        next(c for c in scoped.subjects[0].checks if c.id == "one_actor").result
        == "pass"
    )


def test_a_later_version_the_named_actor_wrote_is_adjudicated_not_excluded():
    """The other half of the rule: a later row the named actor wrote is that actor's own conduct.
    Only the writer differs between the two fixture variants: authorised by the named actor is
    read; authorised by a third party is not. Nothing else in these fixtures separates them.
    """
    late_auth = [dict(r) for r in _VERSIONS[:2]] + [
        {
            "asset": "A1",
            "version": "3",
            "who": "OP42",
            "written": "2026-07-27T16:51:00.000Z",
            "authorised": True,
        }
    ]
    v = _as_of_verdict(late_auth, as_of=_AS_OF)
    assert (
        next(c for c in v.subjects[0].checks if c.id == "authorised").result == "pass"
    )
    assert not [n for n in v.subjects[0].notes if "as_of" in n], "nothing was excluded"

    # ...and had a third party written that very row, the authorisation is not the subject's.
    by_other = late_auth[:2] + [dict(late_auth[2], who="RESPONDER")]
    other = _as_of_verdict(by_other, as_of=_AS_OF)
    assert (
        next(c for c in other.subjects[0].checks if c.id == "authorised").result
        == "unknown"
    )


def test_a_padded_form_of_the_named_actor_still_counts_as_the_named_actor():
    """The incident names `0303CD`, the version log stores `0303CDSU` — one identity, two
    surface forms, and an equality test would read the actor's own work as a third party's.
    """
    padded = [dict(r, who="OP42SU") for r in _VERSIONS[:2]] + [
        {
            "asset": "A1",
            "version": "3",
            "who": "OP42SU",
            "written": "2026-07-27T18:15:51.000Z",
            "authorised": True,
        }
    ]
    v = _as_of_verdict(padded, as_of=_AS_OF, actor="OP42")
    assert (
        next(c for c in v.subjects[0].checks if c.id == "authorised").result == "pass"
    )


def test_the_narrowing_is_stated_on_the_subject_not_left_implicit():
    """A condition that passes because 1 later version was excluded is not the same finding
    as one that passes on the whole record, and a reader must be able to tell them apart —
    including WHO wrote what was excluded, which is what makes it checkable."""
    scoped = _as_of_verdict(_VERSIONS, as_of=_AS_OF)
    note = next(n for n in scoped.subjects[0].notes if n.startswith("as_of="))
    assert "2 of 3" in note and "RESPONDER" in note
    # ...and the unnarrowed run says nothing, so the note's presence carries information.
    assert not [n for n in _as_of_verdict(_VERSIONS).subjects[0].notes if "as_of" in n]


def test_an_undeclared_source_is_untouched():
    """`as_of` is opt-in per source: a pack that declares nothing behaves exactly as before,
    and so does a source the declaration does not name."""
    other = _as_of_verdict(_VERSIONS, as_of={"not_this_one": dict(_AS_OF["record"])})
    assert (
        next(c for c in other.subjects[0].checks if c.id == "one_actor").result
        == "fail"
    )


def test_a_declaration_missing_either_half_narrows_nothing():
    """Both halves are load-bearing, and a half-declared rule is one of the two known-wrong
    rules — so it does not run at all. Silence here is safe (the pre-existing whole-chain
    read); guessing the other half is not, and `pack_validate` is what makes it loud."""
    for partial in (
        {"timestamp_fields": ["written"]},  # time-only: deletes evidence
        {
            "actor_fields": ["who"],
            "actor_entities": ["operator"],
        },  # actor-only: no boundary
    ):
        v = _as_of_verdict(_VERSIONS, as_of={"record": partial})
        assert (
            next(c for c in v.subjects[0].checks if c.id == "one_actor").result
            == "fail"
        )
        assert not [n for n in v.subjects[0].notes if "as_of" in n]


def test_an_incident_that_named_no_actor_reads_the_whole_chain():
    """The declaration is present and correct; the INCIDENT extracted no identity of that
    type. With nothing to compare a writer against, every later row would be "third party" —
    so the rule abstains rather than excluding the lot."""
    v = _as_of_verdict(_VERSIONS, as_of=_AS_OF, actor=None)
    assert next(c for c in v.subjects[0].checks if c.id == "one_actor").result == "fail"
    assert not [n for n in v.subjects[0].notes if "as_of" in n]


def test_a_version_with_no_writer_on_it_is_kept():
    """Unattributable is not third-party. A row whose actor field is absent or empty cannot
    be shown to be somebody else's, and the cost of guessing is deleting evidence."""
    anon = _VERSIONS[:2] + [
        {
            "asset": "A1",
            "version": "3",
            "who": "",
            "written": "2026-07-27T18:15:51.000Z",
            "authorised": True,
        }
    ]
    v = _as_of_verdict(anon, as_of=_AS_OF)
    assert (
        next(c for c in v.subjects[0].checks if c.id == "authorised").result == "pass"
    )


def test_an_incident_with_no_event_window_reads_the_whole_chain():
    """No as-of instant -> the unfiltered read, which is the correct degradation: guessing
    one (ingestion time, or now) would narrow the evidence on a boundary nobody stated.
    """
    v = _as_of_verdict(_VERSIONS, as_of=_AS_OF, event=None)
    assert next(c for c in v.subjects[0].checks if c.id == "one_actor").result == "fail"
    assert not [n for n in v.subjects[0].notes if "as_of" in n]


def test_a_version_whose_write_time_does_not_parse_is_kept():
    """A version that cannot be placed in time is not evidence that it is late. Dropping it
    would silently narrow the evidence on a parse failure."""
    rows = _VERSIONS[:2] + [dict(_VERSIONS[2], written="not a timestamp")]
    v = _as_of_verdict(rows, as_of=_AS_OF)
    assert next(c for c in v.subjects[0].checks if c.id == "one_actor").result == "fail"
    assert not [n for n in v.subjects[0].notes if "as_of" in n]


def test_the_boundary_never_empties_a_source():
    """If the boundary excludes every row, the boundary is wrong about this source (a lagging
    feed, a window extracted a day off), so the unfiltered rows stand. The fallback is reported,
    because a silent as-of read that gave up is indistinguishable from one with nothing to drop.
    """
    late = [
        dict(r, who="RESPONDER", written="2026-07-28T09:00:00.000Z") for r in _VERSIONS
    ]
    v = _as_of_verdict(late, as_of=_AS_OF)
    assert next(c for c in v.subjects[0].checks if c.id == "one_actor").result == "pass"
    note = next(n for n in v.subjects[0].notes if n.startswith("as_of_fallback="))
    assert "3 version(s)" in note and "RESPONDER" in note


def test_a_day_granular_write_time_keeps_the_same_days_later_versions():
    """The boundary is only as precise as the column. A day-granular write time cannot separate
    16:11 from 18:15, so every same-day version is kept. Declare the sub-day column first."""
    daily = [dict(r, written=r["written"][:10]) for r in _VERSIONS]
    v = _as_of_verdict(daily, as_of=_AS_OF)
    assert next(c for c in v.subjects[0].checks if c.id == "one_actor").result == "fail"


def test_the_narrowing_does_not_leak_into_the_evidence_the_reviewer_needs():
    """The split is the design. The CONDITIONS ask what the record looked like when the alert
    fired; the chronology and the sweep ask what has happened to it SINCE — including the
    responder's actions. `evaluate_verdict` narrows its own copy and never the caller's rows.
    """
    logs = {"record_source": list(_VERSIONS)}
    before = [dict(r) for r in logs["record_source"]]
    from src.correlation import evaluate_verdict

    u = _understanding(
        [
            ExtractedEntity(type="asset", value="A1"),
            # The named actor, so the filter actually RUNS — without an identity to compare
            # writers against it abstains, and this test would pass by doing nothing.
            ExtractedEntity(type="operator", value="OP42"),
        ]
    )
    u.analysis.event_time = EventWindow(
        start="2026-07-27T16:45:00Z", end="2026-07-27T16:45:00Z"
    )
    evaluate_verdict(
        {
            "subject_entity": "asset",
            "labels": {"fraud": "FRAUD"},
            "sources": {"record": "record_source"},
            "as_of": _AS_OF,
            "conditions": [
                {
                    "id": "one_actor",
                    "kind": "distinct_count",
                    "source": "record",
                    "field": "who",
                    "max": 1,
                    "label": "One identity acted",
                }
            ],
        },
        logs,
        u.analysis,
    )
    assert logs["record_source"] == before, "the verdict mutated the caller's evidence"


# `reads:` — a value-pattern condition over several shapes of the same source
#
# `fields:` is first-present (right within a shape, wrong across them) and cannot carry
# per-shape row selectors. `reads:` gives each shape its own field list and optional selector.
# A partial union narrows what a pass means: silence on one read is not evidence of nothing.


def _pattern_cond(**over):
    """A forbidden-pattern condition; `over` replaces any key."""
    cond = {
        "id": "wildcard_receiver",
        "kind": "value_matches_pattern",
        "polarity": "fraud_indicator",
        "source": "sessions",
        "match_mode": "forbidden",
        "patterns": [r"^[^/]*\*"],
    }
    cond.update(over)
    return cond


def test_reads_union_spans_shapes_one_fields_list_cannot():
    """A wildcard on the SECOND declared read still fails — `fields:` would have stopped at
    the first shape that resolved and reported a clean receiver."""
    from src.correlation import _eval_condition

    cond = _pattern_cond(
        reads=[
            {"source": "sessions", "fields": ["payload.ReceiverOffice"]},
            {"source": "sessions", "fields": ["actions.receiverOfficeCode"]},
        ]
    )
    rows = {
        "sessions": [
            {"payload": {"ReceiverOffice": "RRR1S18JK/**"}},  # specific
            {"actions": [{"receiverOfficeCode": "***XZ9***/**"}]},  # unbounded
        ]
    }
    assert _eval_condition(cond, rows).result == "fail"

    # The same evidence read as ONE first-present `fields:` list: shape B resolves, so the
    # wildcard on shape A is never looked at. This is the defect the union removes.
    single = _pattern_cond(
        fields=["payload.ReceiverOffice", "actions.receiverOfficeCode"]
    )
    assert _eval_condition(single, rows).result == "pass"


def test_reads_union_row_selector_is_per_read():
    """A read may carry `records:`/`where:` without imposing it on its siblings — the shape
    whose two sides share one array needs the selector, the flat shapes must not have it.
    """
    from src.correlation import _eval_condition

    cond = _pattern_cond(
        reads=[
            {"source": "sessions", "fields": ["payload.RcvLocation"]},
            {
                "source": "sessions",
                "records": "payload.views",
                "where": [{"field": "role", "any_of": ["RECEIVING"], "match": "exact"}],
                "fields": ["addr.location"],
            },
        ]
    )
    # The wildcard is on the DELEGATING side, which is the ordinary way an office shares its
    # own data out. Selecting the receiving entry is what keeps this a PASS.
    rows = {
        "sessions": [
            {
                "payload": {
                    "views": [
                        {"role": "DELEGATING", "addr": {"location": "*"}},
                        {"role": "RECEIVING", "addr": {"location": "LHR"}},
                    ]
                }
            }
        ]
    }
    assert _eval_condition(cond, rows).result == "pass"

    rows["sessions"][0]["payload"]["views"][1]["addr"]["location"] = "*"
    assert _eval_condition(cond, rows).result == "fail"


def test_reads_pass_states_what_was_examined_and_what_was_silent():
    """A PASS over a partly-silent read set says so. 'none' is equally true of a read that
    saw 267 offices and of one that saw nothing, and only one of those clears a subject.
    """
    from src.correlation import _eval_condition

    cond = _pattern_cond(
        reads=[
            {"source": "sessions", "fields": ["payload.ReceiverOffice"]},
            {"source": "sessions", "fields": ["actions.receiverOfficeCode"]},
        ]
    )
    rows = {"sessions": [{"actions": [{"receiverOfficeCode": "UUU1V21QR/**"}]}]}
    c = _eval_condition(cond, rows)
    assert c.result == "pass"
    # WHICH read answered, and how many values it saw.
    assert "actions.receiverOfficeCode" in c.observed
    assert "1 value" in c.observed
    # WHICH read was silent, and that the result covers only what was read.
    assert "1 of 2 declared read(s) returned no values" in c.detail
    assert "payload.ReceiverOffice" in c.detail
    assert "covers only what was read" in c.detail


def test_reads_all_silent_is_unknown_not_pass():
    """Every declared read absent stays UNKNOWN. A forbidden-pattern check that finds no
    values has not cleared anything, and naming the paths is what makes the gap fixable.
    """
    from src.correlation import _eval_condition

    cond = _pattern_cond(
        reads=[
            {"source": "sessions", "fields": ["payload.ReceiverOffice"]},
            {"source": "sessions", "fields": ["actions.receiverOfficeCode"]},
        ]
    )
    c = _eval_condition(cond, {"sessions": [{"unrelated": 1}]})
    assert c.result == "unknown"
    assert "all 2 declared read(s)" in c.observed
    assert "actions.receiverOfficeCode" in c.observed


def test_reads_fully_answered_carries_no_coverage_caveat():
    """No silent read, no caveat — the clause must not become boilerplate on every result."""
    from src.correlation import _eval_condition

    cond = _pattern_cond(
        reads=[
            {"source": "sessions", "fields": ["payload.ReceiverOffice"]},
            {"source": "sessions", "fields": ["actions.receiverOfficeCode"]},
        ]
    )
    rows = {
        "sessions": [
            {"payload": {"ReceiverOffice": "RRR1S18JK/**"}},
            {"actions": [{"receiverOfficeCode": "UUU1V21QR/**"}]},
        ]
    }
    c = _eval_condition(cond, rows)
    assert c.result == "pass"
    assert "covers only what was read" not in (c.detail or "")


def test_reads_absent_leaves_plain_fields_behaviour_unchanged():
    """No `reads:` key, no change: every existing condition in every pack reads exactly as
    before, which is what keeps this a capability rather than a migration."""
    from src.correlation import _eval_condition

    cond = _pattern_cond(fields=["actions.receiverOfficeCode"])
    rows = {"sessions": [{"actions": [{"receiverOfficeCode": "***XZ9***/**"}]}]}
    c = _eval_condition(cond, rows)
    assert c.result == "fail"
    assert "declared read(s)" not in (c.detail or "")
    assert "declared read(s)" not in c.observed


def test_reads_bookkeeping_never_overwrites_the_condition_label():
    """The read-coverage bookkeeping names PATHS; the condition's `label` names the FINDING.

    A near miss, and a cross-domain one: the loop that records which read answered bound its
    path string to `label`, shadowing the enclosing scope's condition label. Every
    `value_matches_pattern` line in every report — including conditions declaring no `reads:`
    at all — then printed `record.address.addr_detail.email.red` where "Non-agency email"
    belonged, and the categorical-exclusion note that lists the outranked indicators listed
    field paths. Nothing raised; the results were all correct.
    """
    from src.correlation import _eval_condition

    for cond in (
        _pattern_cond(id="c", label="Non-agency email", fields=["a.b.c"]),
        _pattern_cond(
            id="c",
            label="Non-agency email",
            reads=[{"source": "sessions", "fields": ["a.b.c"]}],
        ),
    ):
        c = _eval_condition(cond, {"sessions": [{"a": {"b": {"c": "UUU1V21QR/**"}}}]})
        assert c.label == "Non-agency email", c.label
        assert "a.b.c" not in c.label


# --- co-identity: two forms of one identity are ONE subject -------------------
# The defect: an alert naming both an actor's forms ("User: <login> … Sign: <sign>") produced
# TWO subjects — value-membership matches a row carrying either, so identical conditions ran
# over identical rows and printed under two headings, with a rollup claiming two actors were
# adjudicated. Measured on a live job: both values selected the same 73 rows.
#
# The merge is licensed by EVIDENCE, never by shape, and these tests assert both directions:
# a licensing row merges, its absence does not, and the surviving subject keeps the dropped
# value's rows (or the fix trades a duplicate subject for a set of fabricated UNKNOWNs).


def _co_spec(**over):
    """A two-condition ruleset whose subject entity declares two co-identifying forms."""
    spec = {
        "label_scheme": "scheme",
        "subject_entity": "actor",
        "labels": {
            "fraud": "VALID FRAUD",
            "false_positive": "FALSE POSITIVE",
            "insufficient": "INSUFFICIENT DATA",
        },
        "sources": {"sessions": "session_feed", "auth": "auth_feed"},
        "conditions": [
            {
                "id": "clean",
                "label": "Nothing forbidden",
                "kind": "record_absence",
                "source": "sessions",
                "fields": ["verb"],
                "forbidden_values": ["PURGE"],
                "decisive": True,
            }
        ],
        "_co_identity": {"forms": ["sign", "login"], "via": ["unit"], "prefer": "sign"},
    }
    spec.update(over)
    return spec


class _CoAnalysis:
    """Both forms extracted, classified, exactly as the engine stamps them."""

    def __init__(self, pairs=(("LOGIN1", "login"), ("0202YZ", "sign"))):
        self.extracted_entities = [
            ExtractedEntity(type="actor", value=v, value_form=f) for v, f in pairs
        ]


def _co_map():
    return {"session_feed": {"unit": "office"}, "auth_feed": {"unit": "org_unit"}}


def _co_logs(**over):
    logs = {
        # ONE row binding both forms and carrying one unit — the licence.
        "session_feed": [
            {"user": "LOGIN1", "sign": "0202YZ", "office": "BBB1C03EF", "verb": "READ"}
        ],
        # A feed keyed on the OTHER form only. If the merge drops `LOGIN1` and the survivor
        # stops matching it, this source goes silent and its condition reads `unknown`.
        "auth_feed": [{"userId": "LOGIN1", "org_unit": "BBB1C03EF", "verb": "READ"}],
    }
    logs.update(over)
    return logs


def test_co_identity_merges_two_forms_into_one_subject():
    v = evaluate_verdict(_co_spec(), _co_logs(), _CoAnalysis(), _co_map())
    assert v is not None
    assert len(v.subjects) == 1, [s.subject_value for s in v.subjects]


def test_co_identity_keeps_the_preferred_form_as_the_heading():
    """`prefer` is not cosmetic: the containment paragraph names the sign, so a subject
    heading carrying the login makes an operator confirm the link by hand."""
    v = evaluate_verdict(_co_spec(), _co_logs(), _CoAnalysis(), _co_map())
    assert v.subjects[0].subject_value == "0202YZ"

    # And the preference is honoured whichever order the alert gave the forms in.
    v2 = evaluate_verdict(
        _co_spec(),
        _co_logs(),
        _CoAnalysis((("0202YZ", "sign"), ("LOGIN1", "login"))),
        _co_map(),
    )
    assert v2.subjects[0].subject_value == "0202YZ"


def test_co_identity_states_the_licence_in_a_note():
    """A collapse that is not stated reads as an alert that named one identity — the reader
    can tell neither that a merge happened nor what authorised it."""
    v = evaluate_verdict(_co_spec(), _co_logs(), _CoAnalysis(), _co_map())
    note = next(
        (n for n in v.subjects[0].notes if n.startswith("co_identity=")),
        "",
    )
    assert note, v.subjects[0].notes
    assert "0202YZ" in note and "LOGIN1" in note
    assert "session_feed" in note, note  # which row licensed it
    assert "unit=BBB1C03EF" in note, note  # and on what agreement


def test_co_identity_without_an_evidencing_row_keeps_two_subjects():
    """No row binds both values: the merge REFUSES. Two subjects is the honest reading —
    asserting an identity relationship the data never showed is the worse error."""
    logs = {
        "session_feed": [{"sign": "0202YZ", "office": "BBB1C03EF", "verb": "READ"}],
        "auth_feed": [{"userId": "LOGIN1", "org_unit": "BBB1C03EF", "verb": "READ"}],
    }
    v = evaluate_verdict(_co_spec(), logs, _CoAnalysis(), _co_map())
    assert len(v.subjects) == 2, [s.subject_value for s in v.subjects]
    assert not any(
        n.startswith("co_identity=") for s in v.subjects for n in s.notes
    ), "a refused merge must not claim one"


def test_co_identity_refuses_when_the_composite_disagrees():
    """A row binding both values but carrying TWO units is ambiguous about which identity it
    describes, so it licenses nothing. This is the domain's own definition of a user: neither
    identifier is an identity by itself, and shape is no licence at all."""
    logs = {
        "session_feed": [
            {
                "user": "LOGIN1",
                "sign": "0202YZ",
                "office": ["BBB1C03EF", "AAA1B0955"],
                "verb": "READ",
            }
        ]
    }
    v = evaluate_verdict(_co_spec(), logs, _CoAnalysis(), _co_map())
    assert len(v.subjects) == 2, [s.subject_value for s in v.subjects]


def test_co_identity_refuses_when_the_row_omits_the_composite():
    """`via` is an AND and every named type must resolve. A source that cannot state the whole
    pair does not get to decide a merge that is irreversible in the report."""
    logs = {"session_feed": [{"user": "LOGIN1", "sign": "0202YZ", "verb": "READ"}]}
    v = evaluate_verdict(_co_spec(), logs, _CoAnalysis(), _co_map())
    assert len(v.subjects) == 2, [s.subject_value for s in v.subjects]


def test_the_merged_subject_still_reads_the_dropped_value_rows():
    """The other half of the fix. The survivor is the sign; `auth_feed` carries only the
    login. Selecting on the survivor alone would empty that source and turn its condition
    `unknown`, trading a duplicate subject for a fabricated gap."""
    spec = _co_spec()
    spec["conditions"] = spec["conditions"] + [
        {
            "id": "auth_clean",
            "label": "No forbidden auth event",
            "kind": "record_absence",
            "source": "auth",
            "fields": ["verb"],
            "forbidden_values": ["PURGE"],
            "decisive": False,
        }
    ]
    v = evaluate_verdict(spec, _co_logs(), _CoAnalysis(), _co_map())
    s = v.subjects[0]
    assert s.subject_value == "0202YZ"
    auth = next(c for c in s.checks if c.id == "auth_clean")
    assert auth.result == "pass", (auth.result, auth.observed, auth.detail)

    # And the row is really being read, not merely present: a forbidden value in the
    # login-keyed feed must FAIL the merged subject.
    logs = _co_logs(
        auth_feed=[{"userId": "LOGIN1", "org_unit": "BBB1C03EF", "verb": "PURGE"}]
    )
    v2 = evaluate_verdict(spec, logs, _CoAnalysis(), _co_map())
    auth2 = next(c for c in v2.subjects[0].checks if c.id == "auth_clean")
    assert auth2.result == "fail", (auth2.result, auth2.observed)


def test_co_identity_leaves_an_unclassified_value_alone():
    """A value whose form is not one of the declared two is not known to be either, so it is
    never folded into a classified one — a third, genuinely separate actor stays separate.
    """
    analysis = _CoAnalysis((("LOGIN1", "login"), ("0202YZ", "sign"), ("SOMEONE", "")))
    logs = _co_logs(
        session_feed=[
            {
                "user": "LOGIN1",
                "sign": "0202YZ",
                "other": "SOMEONE",
                "office": "BBB1C03EF",
                "verb": "READ",
            }
        ]
    )
    v = evaluate_verdict(_co_spec(), logs, analysis, _co_map())
    assert sorted(s.subject_value for s in v.subjects) == ["0202YZ", "SOMEONE"]


def test_no_co_identity_declaration_leaves_every_value_its_own_subject():
    """Absent the declaration nothing changes — which is what keeps this a pack-driven
    capability rather than a behaviour change for every existing ruleset."""
    spec = _co_spec()
    spec.pop("_co_identity")
    v = evaluate_verdict(spec, _co_logs(), _CoAnalysis(), _co_map())
    assert len(v.subjects) == 2, [s.subject_value for s in v.subjects]


# `exclude_subject` on a distinct_count: the subject's own value is not blast radius.
#
# A pivot count asks "did anybody other than the subject use this?" No single `max:` can
# answer it: whether the subject appears in the values is a property of the data. The
# subtraction must reach every form the subject is known by (the adjudicated form and the
# counted column often differ) and must not turn an all-subject result into `unknown`.


def _pivot_cond(**over):
    """A pivot `distinct_count` bounded at zero others, with the subject subtracted."""
    cond = {
        "id": "co_parties",
        "label": "No party other than the subject used the pivoted address",
        "kind": "distinct_count",
        "source": "pivot",
        "field": "party",
        "max": 0,
        "exclude_subject": True,
        "_subject_identity_values": ["SUBJ-A", "SUBJALT"],
    }
    cond.update(over)
    return cond


def test_exclude_subject_subtracts_the_subjects_own_value_from_the_count():
    """The count the bound is compared against must be the count of OTHERS."""
    from src.correlation import _eval_condition

    rows = {"pivot": [{"party": "SUBJ-A"}, {"party": "STRANGER"}]}
    c = _eval_condition(_pivot_cond(), rows)
    assert "1 distinct" in c.observed, c.observed
    assert "STRANGER" in c.observed
    assert "SUBJ-A" not in c.observed, c.observed
    assert c.result == "fail"  # one other party is one too many at max: 0


def test_exclude_subject_reads_only_the_subject_as_a_real_count_of_zero():
    """Subtracting the subject can empty the value set, and the empty-values branch reports
    "no data" rather than passing. A full result read as an empty one is the defect this
    condition kind was hardened against; this is the same defect one step further along.
    """
    from src.correlation import _eval_condition

    rows = {"pivot": [{"party": "SUBJ-A"}, {"party": "SUBJALT"}, {"party": "subj-a"}]}
    c = _eval_condition(_pivot_cond(), rows)
    assert c.result == "pass", c.observed
    assert "0 distinct" in c.observed, c.observed
    assert "unknown" not in c.result
    assert "excluded" in c.observed, c.observed


def test_exclude_subject_reaches_every_form_the_subject_is_known_by():
    """Half the fix is the whole bug: the counted column often carries the OTHER form.

    The subject is adjudicated under one surface form and the pivoted field holds another,
    which `_merge_co_identified_subjects` proved name the same actor. Excluding only the
    adjudicated value would count the subject's own second form as a stranger — a FAIL
    naming the subject as its own blast radius.
    """
    from src.correlation import _eval_condition

    rows = {"pivot": [{"party": "SUBJALT"}]}
    c = _eval_condition(_pivot_cond(), rows)
    assert c.result == "pass", c.observed
    assert "0 distinct" in c.observed, c.observed

    # And with only the adjudicated form on the exclusion list, the alias is a stranger —
    # which is what makes passing the aliases load-bearing rather than defensive.
    half = _eval_condition(_pivot_cond(_subject_identity_values=["SUBJ-A"]), rows)
    assert half.result == "fail", half.observed


def test_exclude_subject_says_so_whether_or_not_the_subject_was_there():
    """`1 distinct` against `max: 0` is a different finding depending on whether the
    subject was already taken out, and the number alone cannot tell the reader which.

    The absent case is stated too, because "excluded but not among them" is itself
    information — it says the subject did not use the pivoted address on this source at
    all, which is the shape the live defect actually had.
    """
    from src.correlation import _eval_condition

    present = _eval_condition(
        _pivot_cond(), {"pivot": [{"party": "SUBJ-A"}, {"party": "STRANGER"}]}
    )
    absent = _eval_condition(_pivot_cond(), {"pivot": [{"party": "STRANGER"}]})
    assert "excluded and present" in present.observed, present.observed
    assert "excluded but not among them" in absent.observed, absent.observed
    # Same number, different finding — which is the whole reason the note exists.
    assert "1 distinct" in present.observed and "1 distinct" in absent.observed


def test_without_the_declaration_a_distinct_count_is_untouched():
    """No `exclude_subject` means the pre-existing arithmetic and the pre-existing wording,
    or this is a behaviour change for every ruleset that never asked for one."""
    from src.correlation import _eval_condition

    cond = _pivot_cond(max=1)
    cond.pop("exclude_subject")
    rows = {"pivot": [{"party": "SUBJ-A"}, {"party": "STRANGER"}]}
    c = _eval_condition(cond, rows)
    assert "2 distinct" in c.observed, c.observed
    assert "excluded" not in c.observed, c.observed
    assert c.result == "fail"


def _excl_spec(**over):
    """`_co_spec` plus a subject-excluding pivot count over a query-scoped source."""
    spec = _co_spec()
    spec["sources"] = dict(spec["sources"], pivot="pivot_feed")
    spec["conditions"] = [
        {
            "id": "co_parties",
            "label": "No party other than the subject used the pivoted address",
            "kind": "distinct_count",
            "source": "pivot",
            "field": "party",
            "max": 0,
            "exclude_subject": True,
            "subject_scope": False,
            "decisive": True,
        }
    ]
    spec.update(over)
    return spec


def test_exclude_subject_is_stamped_from_the_merged_subjects_aliases():
    """End to end: a ruleset never names a subject, so the values come from the caller.

    The pivoted column here carries the login while the subject is adjudicated under the
    sign, which is exactly the case `_merge_co_identified_subjects` exists for — so this
    also proves the alias survives the merge and reaches the count.
    """
    logs = _co_logs(pivot_feed=[{"party": "LOGIN1"}])
    v = evaluate_verdict(_excl_spec(), logs, _CoAnalysis(), _co_map())
    assert len(v.subjects) == 1, [s.subject_value for s in v.subjects]
    c = next(c for c in v.subjects[0].checks if c.id == "co_parties")
    assert c.result == "pass", c.observed
    assert "0 distinct" in c.observed, c.observed


def test_each_subject_excludes_only_its_own_values_not_its_siblings():
    """One shared entity list would have every subject discount the others out of its own
    count — a two-subject incident would then clear both of implicating each other.

    So the exclusion set is resolved per subject inside the adjudication loop, from that
    subject's own value and aliases.
    """
    spec = _excl_spec()
    spec.pop("_co_identity")  # two independent subjects, no merge
    logs = _co_logs(pivot_feed=[{"party": "0202YZ"}, {"party": "LOGIN1"}])
    v = evaluate_verdict(spec, logs, _CoAnalysis(), _co_map())
    assert len(v.subjects) == 2, [s.subject_value for s in v.subjects]
    for s in v.subjects:
        c = next(c for c in s.checks if c.id == "co_parties")
        # Its own value is gone; the sibling's is still counted and still implicates it.
        assert "1 distinct" in c.observed, (s.subject_value, c.observed)
        assert c.result == "fail", (s.subject_value, c.observed)
        assert s.subject_value not in c.observed, (s.subject_value, c.observed)


# --- companions for a subject the incident named ------------------------------
# `subject_discovery:` answers two questions: who the subject is, and what values sit beside
# it on its own rows. When extraction named the subject, companions fill `from_entity` clauses
# that would otherwise resolve to nothing. The licence is unanimity: all subject rows must
# agree on the companion value; a disputed value leaves the clause unresolved.


def _comp_spec(**over):
    """A reference-lookup ruleset whose subject is named and whose actor is not."""
    spec = {
        "label_scheme": "scheme",
        "subject_entity": "asset",
        "labels": {
            "fraud": "VALID FRAUD",
            "false_positive": "FALSE POSITIVE",
            "insufficient": "INSUFFICIENT DATA",
        },
        "sources": {"records": "record_feed", "register": "register_feed"},
        "subject_discovery": {
            "source": "records",
            "subject": ["asset_id"],
            "entities": {"actor": ["actor_code"], "unit": ["unit_code"]},
        },
        "conditions": [
            {
                "id": "not_registered",
                "label": "Acting identity is not on the register",
                "kind": "record_absence",
                "source": "register",
                "subject_scope": False,
                "row_match": [
                    {"label": "unit", "from_entity": "unit", "fields": ["unit"]},
                    {"label": "actor", "from_entity": "actor", "fields": ["code"]},
                ],
                "fields": ["profile"],
                "forbidden_values": ["AUTOMATED"],
                "match": "exact",
                "decisive": True,
                "decisive_on": ["fail"],
            }
        ],
    }
    spec.update(over)
    return spec


class _CompAnalysis:
    """The subject named, the acting identity NOT — the shape this path exists for."""

    def __init__(self, extra=()):
        self.extracted_entities = [ExtractedEntity(type="asset", value="AST-1")] + [
            ExtractedEntity(type=t, value=v) for t, v in extra
        ]


def _comp_logs(rows=None, register=None):
    return {
        "record_feed": rows
        if rows is not None
        else [
            {"asset_id": "AST-1", "actor_code": "X1", "unit_code": "U1", "seq": 1},
            {"asset_id": "AST-1", "actor_code": "X1", "unit_code": "U1", "seq": 2},
        ],
        "register_feed": register
        if register is not None
        else [{"unit": "U1", "code": "X1", "profile": "AUTOMATED"}],
    }


def _comp_check(v, cid="not_registered"):
    return next(c for c in v.subjects[0].checks if c.id == cid)


def test_a_named_subjects_companion_value_fills_the_clause_the_incident_left_empty():
    v = evaluate_verdict(_comp_spec(), _comp_logs(), _CompAnalysis())
    assert v is not None and [s.subject_value for s in v.subjects] == ["AST-1"]
    c = _comp_check(v)
    assert c.result == "fail", (c.result, c.observed)
    # And the subject is still the NAMED one: the companion read must not re-scope the verdict.
    note = " ".join(v.subjects[0].notes or [])
    assert "subject_companions=AST-1" in note, note
    assert "actor=X1" in note and "unit=U1" in note, note
    assert "subject_discovered" not in note, note


def test_companion_values_the_rows_disagree_on_are_left_unresolved():
    """Two actors on the subject's rows -> no single value answers "THE acting one"."""
    rows = [
        {"asset_id": "AST-1", "actor_code": "X1", "unit_code": "U1", "seq": 1},
        {"asset_id": "AST-1", "actor_code": "X2", "unit_code": "U1", "seq": 2},
    ]
    v = evaluate_verdict(_comp_spec(), _comp_logs(rows=rows), _CompAnalysis())
    c = _comp_check(v)
    assert c.result == "unknown", (c.result, c.observed)
    note = " ".join(v.subjects[0].notes or [])
    # The disagreement is STATED: an empty value set says nothing, and this check is decisive.
    assert "do NOT agree on actor (2 distinct values)" in note, note
    # The unit they DO agree on is still read — a filled clause is not withheld because a
    # sibling clause could not be filled.
    assert "unit=U1" in note, note


def test_a_ruleset_declaring_no_subject_discovery_is_untouched():
    """The pre-fix reading verbatim: no declaration, no companions, an unresolved clause."""
    spec = _comp_spec()
    spec.pop("subject_discovery")
    v = evaluate_verdict(spec, _comp_logs(), _CompAnalysis())
    assert _comp_check(v).result == "unknown"
    assert not any(
        "subject_companions" in n for s in v.subjects for n in (s.notes or [])
    )


# A nested declaration against the flattened alias the projection actually returns.
#
# `resolve_path` bridges `a.b.c` to `a_b_c` because a SQL backend aliases nested leaves after
# their last segment. A path written as nested must still resolve on a genuinely nested row
# (flattening it moves the silence rather than removing it). Three tests cover both seams:
# alias resolution, nested-row resolution, and both discovery paths sharing the helper.


def _nested_comp_spec():
    """`_comp_spec` with every discovery path written as a nested one."""
    spec = _comp_spec()
    spec["subject_discovery"] = {
        "source": "records",
        "subject": ["record.asset.id"],
        "entities": {"actor": ["record.actor.code"], "unit": ["record.unit.code"]},
    }
    return spec


def test_a_nested_discovery_path_resolves_against_the_flattened_alias():
    """The row carries the aliases the SELECT list produced, not the nested structs."""
    rows = [
        {
            "record_asset_id": "AST-1",
            "record_actor_code": "X1",
            "record_unit_code": "U1",
            "seq": 1,
        },
        {
            "record_asset_id": "AST-1",
            "record_actor_code": "X1",
            "record_unit_code": "U1",
            "seq": 2,
        },
    ]
    v = evaluate_verdict(_nested_comp_spec(), _comp_logs(rows=rows), _CompAnalysis())
    c = _comp_check(v)
    assert c.result == "fail", (c.result, c.observed)
    note = " ".join(v.subjects[0].notes or [])
    assert "actor=X1" in note and "unit=U1" in note, note


def test_the_same_nested_path_still_resolves_against_a_genuinely_nested_row():
    """The bound, and the reason this is a bridge rather than a flattening.

    One backend returns the alias and another returns the struct; the declaration cannot know
    which, and a reading that only handled the flat name would move the silence rather than
    remove it.
    """
    rows = [
        {
            "record": {
                "asset": {"id": "AST-1"},
                "actor": {"code": "X1"},
                "unit": {"code": "U1"},
            },
            "seq": 1,
        }
    ]
    v = evaluate_verdict(_nested_comp_spec(), _comp_logs(rows=rows), _CompAnalysis())
    c = _comp_check(v)
    assert c.result == "fail", (c.result, c.observed)
    note = " ".join(v.subjects[0].notes or [])
    assert "actor=X1" in note and "unit=U1" in note, note


def test_a_subject_nobody_named_is_discovered_from_a_nested_path_under_its_alias():
    """The OTHER seam sharing the helper: the subject itself is read from the rows.

    The companion test above cannot see this one — there the subject was named and only the
    clause values came from the rows, so a broken subject path leaves the verdict looking right
    and the failure lands one stage later.
    """

    class _Nobody:
        extracted_entities = []

    rows = [
        {
            "record_asset_id": "AST-9",
            "record_actor_code": "X1",
            "record_unit_code": "U1",
            "seq": 1,
        }
    ]
    v = evaluate_verdict(_nested_comp_spec(), _comp_logs(rows=rows), _Nobody())
    assert v is not None and [s.subject_value for s in v.subjects] == ["AST-9"], (
        [s.subject_value for s in (v.subjects if v else [])]
    )
    note = " ".join(v.subjects[0].notes or [])
    assert "subject_discovered=AST-9" in note, note
    assert _comp_check(v).result == "fail", _comp_check(v).observed


def test_an_entity_the_incident_named_is_never_overwritten_by_a_column():
    """The alert's own statement outranks the rows. This path fills gaps, nothing else.

    Without the skip, a referral naming the acting identity would have it silently replaced by
    whatever the retrieved rows agree on — the rows correcting the alert, on the one clause a
    decisive categorical exclusion is resolved from.
    """
    v = evaluate_verdict(
        _comp_spec(),
        _comp_logs(),
        _CompAnalysis(extra=(("actor", "X9"), ("unit", "U1"))),
    )
    # X9 is not on the register, so the exclusion PASSES — the row's X1 must not reach it.
    c = _comp_check(v)
    assert c.result == "pass", (c.result, c.observed)
    note = " ".join(v.subjects[0].notes or [])
    assert "actor=" not in note, note


def test_a_subject_the_rows_do_not_carry_yields_no_companions():
    """The named subject is absent from the declared source -> nothing to read, no note."""
    rows = [{"asset_id": "AST-9", "actor_code": "X1", "unit_code": "U1"}]
    v = evaluate_verdict(_comp_spec(), _comp_logs(rows=rows), _CompAnalysis())
    assert _comp_check(v).result == "unknown"
    assert not any(
        "subject_companions" in n for s in v.subjects for n in (s.notes or [])
    )


# --- one subject, several identities that acted on it --------------------------
# A union across acting identities is wrong in two ways: one registered member clears all
# the others, and OR-ed value sets cross-multiply over pairs that never acted together.
# `per_acting_identity: true` evaluates the condition once per observed combination,
# rolling up with fail requiring unanimity. This can never strengthen a verdict.


def _acting_spec(**over):
    """`_comp_spec` with the quantifier declared on its categorical exclusion."""
    spec = _comp_spec(**over)
    spec["conditions"] = [{**spec["conditions"][0], "per_acting_identity": True}]
    return spec


_TWO_PAIRS = [
    {"asset_id": "AST-1", "actor_code": "X1", "unit_code": "U1", "seq": 1},
    {"asset_id": "AST-1", "actor_code": "X2", "unit_code": "U2", "seq": 2},
]


def test_one_unregistered_identity_among_several_answers_the_categorical_check():
    """A human hand among machine ones is still a human hand -> PASS, and the split is stated."""
    v = evaluate_verdict(
        _acting_spec(),
        _comp_logs(
            rows=_TWO_PAIRS,
            register=[{"unit": "U1", "code": "X1", "profile": "AUTOMATED"}],
        ),
        _CompAnalysis(),
    )
    c = _comp_check(v)
    assert c.result == "pass", (c.result, c.observed, c.detail)
    # The COUNT is the finding: a check answered over two identities and one answered over a single
    # identity are different findings, and only the note says which happened.
    assert "asked of each of the 2 identities that acted" in c.detail, c.detail
    assert "actor=X2+unit=U2" in c.detail, c.detail
    assert "reported as PASS" in c.detail, c.detail


def test_the_categorical_fail_requires_every_acting_identity():
    v = evaluate_verdict(
        _acting_spec(),
        _comp_logs(
            rows=_TWO_PAIRS,
            register=[
                {"unit": "U1", "code": "X1", "profile": "AUTOMATED"},
                {"unit": "U2", "code": "X2", "profile": "AUTOMATED"},
            ],
        ),
        _CompAnalysis(),
    )
    c = _comp_check(v)
    assert c.result == "fail", (c.result, c.observed, c.detail)
    assert "every one of them did" in c.detail, c.detail
    # And the decisive FAIL still reaches the verdict as the categorical exclusion it is.
    assert v.subjects[0].verdict == "FALSE POSITIVE", v.subjects[0].verdict


def test_a_pair_that_never_acted_cannot_answer_for_one_that_did():
    """The cross-product row: the register holds U1+X2, which no identity ever was.

    A union reading matches it (`unit IN (U1,U2) AND code IN (X1,X2)` is satisfied on that single
    row) and fires the decisive exclusion. Asked per observed pair, neither real identity is on the
    register and the check passes.
    """
    v = evaluate_verdict(
        _acting_spec(),
        _comp_logs(
            rows=_TWO_PAIRS,
            register=[{"unit": "U1", "code": "X2", "profile": "AUTOMATED"}],
        ),
        _CompAnalysis(),
    )
    c = _comp_check(v)
    assert c.result == "pass", (c.result, c.observed, c.detail)
    assert "asked of each of the 2 identities" in c.detail, c.detail


def test_a_ruleset_not_declaring_per_acting_identity_still_abstains():
    """The pre-fix reading verbatim, and the reason the key exists at all."""
    v = evaluate_verdict(_comp_spec(), _comp_logs(rows=_TWO_PAIRS), _CompAnalysis())
    c = _comp_check(v)
    assert c.result == "unknown", (c.result, c.detail)
    assert "asked of each of the" not in c.detail, c.detail


def test_too_many_acting_identities_withhold_the_set_rather_than_truncating_it():
    """Unanimity over a subset is not unanimity, so past the bound the check reads `unknown`.

    A truncating bound would let the engine assert "every identity that acted" over the first N it
    happened to keep — the same shape as a row cap read as a complete answer.
    """
    n = _MAX_ACTING_TUPLES + 1
    rows = [
        {"asset_id": "AST-1", "actor_code": f"X{i}", "unit_code": f"U{i}", "seq": i}
        for i in range(n)
    ]
    v = evaluate_verdict(
        _acting_spec(),
        # ONE of them registered, which is what makes this assertion discriminating: a bound that
        # truncated to the first 25 would find 24 unregistered pairs among them and return the
        # existential PASS over a subset it never established was the whole set.
        _comp_logs(
            rows=rows,
            register=[{"unit": "U0", "code": "X0", "profile": "AUTOMATED"}],
        ),
        _CompAnalysis(),
    )
    c = _comp_check(v)
    assert c.result == "unknown", (c.result, c.detail)
    note = " ".join(v.subjects[0].notes or [])
    assert f"more than the bound of {_MAX_ACTING_TUPLES}" in note, note
    assert f"{n} distinct combination(s) were observed" in note, note


# --- a two-sided comparison pools its sides across the source ------------------
# `field_equality` and `time_gap` compare two collected sets, so the cross product can be
# satisfied by a pair on no single record. Two independent opt-in narrowings address this:
# `pair_by: record` (compare within each record) and `subject_scope: element` (each subject
# reads only its own entries of a repeated node). Either alone still misreports.


def _pair_spec(cond=None, **over):
    """A ruleset with ONE two-sided condition on a single source, and nothing else."""
    spec = {
        "label_scheme": "scheme",
        "subject_entity": "actor",
        "labels": {
            "fraud": "VALID FRAUD",
            "false_positive": "FALSE POSITIVE",
            "insufficient": "INSUFFICIENT DATA",
        },
        "sources": {"records": "record_feed"},
        "conditions": [
            cond
            or {
                "id": "one_unit",
                "label": "Both units on the record are the same unit",
                "kind": "field_equality",
                "left": {"source": "records", "field": "origin_unit"},
                "right": {"source": "records", "field": "login_unit"},
                "pair_by": "record",
            }
        ],
    }
    spec.update(over)
    return spec


class _PairAnalysis:
    """One named subject, so the rows are not re-scoped underneath the pairing."""

    def __init__(self, value="AC0001"):
        self.extracted_entities = [ExtractedEntity(type="actor", value=value)]


def _pair_check(v, cid="one_unit"):
    return next(c for c in v.subjects[0].checks if c.id == cid)


#: Neither record carries two equal units; pooled, record 1's `origin_unit` matches record 2's
#: `login_unit` and the check PASSES on a pair nobody ever wrote.
_CROSS_ONLY = [
    {"actor": "AC0001", "origin_unit": "U-AAA", "login_unit": "U-BBB"},
    {"actor": "AC0001", "origin_unit": "U-CCC", "login_unit": "U-AAA"},
]


def test_a_pooled_comparison_is_satisfied_by_a_pair_on_no_record():
    """The defect itself, so the fix is measured against it rather than asserted about."""
    spec = _pair_spec()
    spec["conditions"][0].pop("pair_by")
    c = _pair_check(evaluate_verdict(spec, {"record_feed": _CROSS_ONLY}, _PairAnalysis()))
    assert c.result == "pass", (c.result, c.observed)
    assert "WITHIN each" not in (c.detail or ""), c.detail


def test_pairing_within_each_record_drops_the_cross_pair_match():
    c = _pair_check(
        evaluate_verdict(_pair_spec(), {"record_feed": _CROSS_ONLY}, _PairAnalysis())
    )
    assert c.result == "fail", (c.result, c.observed, c.detail)
    # The COUNT is the finding: a comparison made on one record and one made on two are
    # different findings, and only the note says which happened.
    assert "compared WITHIN each of the 2 record(s)" in c.detail, c.detail
    assert "FAIL on 2" in c.detail, c.detail
    assert "every one of them did" in c.detail, c.detail


def test_one_record_carrying_both_sides_is_enough():
    """A PASS is existential — one record satisfying the comparison is a fact about the
    incident that no other record can retract."""
    rows = [_CROSS_ONLY[0], {**_CROSS_ONLY[1], "login_unit": "U-CCC"}]
    c = _pair_check(evaluate_verdict(_pair_spec(), {"record_feed": rows}, _PairAnalysis()))
    assert c.result == "pass", (c.result, c.observed, c.detail)
    assert "PASS on 1" in c.detail and "FAIL on 1" in c.detail, c.detail
    # The quantifier is STATED, not appealed to — a fallback note is scanned word by word
    # against the pack's own vocabulary, so engine prose says the arithmetic it performed.
    assert "a PASS on one of them holds for the set" in c.detail, c.detail


def test_a_record_missing_one_side_blocks_the_unanimity_a_fail_needs():
    """Unanimity over the resolvable records is unanimity over a subset, so it stays UNKNOWN."""
    rows = [_CROSS_ONLY[0], {"actor": "AC0001", "origin_unit": "U-CCC"}]
    c = _pair_check(evaluate_verdict(_pair_spec(), {"record_feed": rows}, _PairAnalysis()))
    assert c.result == "unknown", (c.result, c.observed, c.detail)
    assert "carry only one side" in c.detail, c.detail


def test_sides_that_select_different_records_refuse_the_pairing_out_loud():
    """An interval whose ends are two versions of one entity cannot be paired at all.

    The pooled reading is kept — there is nothing else to fall back to — and the refusal is
    reported, because a declaration that silently does nothing is the defect it was written to
    prevent.
    """
    spec = _pair_spec(
        cond={
            "id": "one_unit",
            "label": "Both units on the record are the same unit",
            "kind": "field_equality",
            "left": {"source": "records", "field": "origin_unit"},
            "right": {
                "source": "records",
                "field": "login_unit",
                "where": [{"field": "actor", "any_of": ["AC0001"]}],
            },
            "pair_by": "record",
        }
    )
    c = _pair_check(evaluate_verdict(spec, {"record_feed": _CROSS_ONLY}, _PairAnalysis()))
    assert c.result == "pass", (c.result, c.observed, c.detail)
    assert "`pair_by: record` was NOT honoured" in c.detail, c.detail
    assert "select different records through `where:`" in c.detail, c.detail


def test_a_time_gap_is_paired_the_same_way():
    """The other gated kind, and the one where pooling is least visible: the gap is measured
    between the EARLIEST timestamp on each side, which is a cross pair by construction.

    Record 1's own interval is five hours and record 2's end precedes its own start (a stale or
    out-of-order write), so each record FAILS on its own. Pooled, the earliest start belongs to
    record 1 and the earliest end to record 2, and the half hour between them is an interval
    neither record contains.
    """
    cond = {
        "id": "quick_refund",
        "label": "The refund followed the claim within an hour",
        "kind": "time_gap",
        "start": {"source": "records", "field": "claimed_at"},
        "end": {"source": "records", "field": "refunded_at"},
        "max": "1h",
        "pair_by": "record",
    }
    rows = [
        {
            "actor": "AC0001",
            "claimed_at": "2026-01-01T10:00:00Z",
            "refunded_at": "2026-01-01T15:00:00Z",
        },
        {
            "actor": "AC0001",
            "claimed_at": "2026-01-01T12:00:00Z",
            "refunded_at": "2026-01-01T10:30:00Z",
        },
    ]
    logs = {"record_feed": rows}
    pooled = dict(cond)
    pooled.pop("pair_by")
    assert (
        _pair_check(
            evaluate_verdict(_pair_spec(cond=pooled), logs, _PairAnalysis()),
            "quick_refund",
        ).result
        == "pass"
    )
    c = _pair_check(
        evaluate_verdict(_pair_spec(cond=cond), logs, _PairAnalysis()), "quick_refund"
    )
    assert c.result == "fail", (c.result, c.observed, c.detail)
    assert "compared WITHIN each of the 2 record(s)" in c.detail, c.detail


def test_pairing_a_source_with_no_rows_reads_as_it_always_did():
    """No record to pair within is not a new outcome — it is the same UNKNOWN as before."""
    declared = _pair_check(evaluate_verdict(_pair_spec(), {"record_feed": []}, _PairAnalysis()))
    plain = _pair_spec()
    plain["conditions"][0].pop("pair_by")
    pooled = _pair_check(evaluate_verdict(plain, {"record_feed": []}, _PairAnalysis()))
    assert declared.result == pooled.result == "unknown"
    assert declared.detail == pooled.detail, (declared.detail, pooled.detail)


# --- one record, one entry per identity ----------------------------------------
# Where entries of a repeated node are one per identity, every subject is selected for the
# whole row and an existential pass carries the other identity's entry. `subject_scope: element`
# narrows the node to the subject's own entries, composing with `records:` and `pair_by`.


def _elem_spec(cond=None, **over):
    spec = {
        "label_scheme": "scheme",
        "subject_entity": "actor",
        "labels": {
            "fraud": "VALID FRAUD",
            "false_positive": "FALSE POSITIVE",
            "insufficient": "INSUFFICIENT DATA",
        },
        "sources": {"alert": "alert_feed"},
        "subject_discovery": {
            "source": "alert",
            "path": "alert.lines",
            "subject": ["acting_code"],
        },
        "conditions": [
            cond
            or {
                "id": "one_unit",
                "label": "Both units on the entry are the same unit",
                "kind": "field_equality",
                "left": {"source": "alert", "field": "alert.lines.origin_unit"},
                "right": {"source": "alert", "field": "alert.lines.login_unit"},
                "subject_scope": "element",
            }
        ],
    }
    spec.update(over)
    return spec


class _ElemAnalysis:
    """No subject named, so the two identities are DISCOVERED from the alert's own entries —
    the live shape, where the alert is the list of subjects."""

    extracted_entities: list = []


#: One row, two entries, one per identity. AC0001's units differ; AC0002's are identical.
_TWO_ENTRY_ALERT = [
    {
        "alert": {
            "incident": "INC-1",
            "lines": [
                {"acting_code": "AC0001", "origin_unit": "U-AAA", "login_unit": "U-BBB"},
                {"acting_code": "AC0002", "origin_unit": "U-CCC", "login_unit": "U-CCC"},
            ],
        }
    }
]


def _by_subject(v, cid="one_unit"):
    return {
        s.subject_value: next(c for c in s.checks if c.id == cid) for s in v.subjects
    }


def test_one_row_holding_every_identitys_entry_answers_both_with_the_other_one():
    """The defect, measured: without the narrowing both identities read the pooled PASS."""
    spec = _elem_spec()
    spec["conditions"][0].pop("subject_scope")
    got = _by_subject(evaluate_verdict(spec, {"alert_feed": _TWO_ENTRY_ALERT}, _ElemAnalysis()))
    assert sorted(got) == ["AC0001", "AC0002"], sorted(got)
    assert [c.result for c in got.values()] == ["pass", "pass"], got
    # AC0001's own entry differs, and the PASS is reported with the other identity's values in it.
    assert "U-CCC" in got["AC0001"].observed, got["AC0001"].observed


def test_each_identity_is_answered_out_of_its_own_entry():
    got = _by_subject(
        evaluate_verdict(_elem_spec(), {"alert_feed": _TWO_ENTRY_ALERT}, _ElemAnalysis())
    )
    assert got["AC0001"].result == "fail", (got["AC0001"].result, got["AC0001"].observed)
    assert got["AC0002"].result == "pass", (got["AC0002"].result, got["AC0002"].observed)
    # The other identity's values are GONE from both readings, not merely outvoted.
    assert "U-CCC" not in got["AC0001"].observed, got["AC0001"].observed
    assert "U-AAA" not in got["AC0002"].observed, got["AC0002"].observed
    # And the narrowing is stated, because a check read on one entry of two and a check read on
    # the whole record are different findings.
    assert "read within the 1 of 2 `alert.lines` entry" in got["AC0001"].detail, got[
        "AC0001"
    ].detail
    assert "belong to other identities" in got["AC0002"].detail, got["AC0002"].detail


def test_an_identity_on_none_of_the_entries_cannot_be_answered_from_them():
    """The honest reading: the entries are somebody else's, so they say nothing about this
    subject — and it must not read as a source that returned nothing.

    The subject is on the record (the alert reports it) and on none of its entries, which is the
    case worth a note. A subject absent from the record altogether has no rows at all and reads
    as the ordinary no-data UNKNOWN it always did.
    """
    rows = [{"alert": {**_TWO_ENTRY_ALERT[0]["alert"], "reported_by": "AC0009"}}]
    v = evaluate_verdict(_elem_spec(), {"alert_feed": rows}, _PairAnalysis(value="AC0009"))
    c = _pair_check(v)
    assert c.result == "unknown", (c.result, c.observed, c.detail)
    assert "is on NONE of the 2 `alert.lines` entries" in c.detail, c.detail


def test_every_entry_being_this_identitys_says_nothing_extra():
    """Silence in exactly one case: where nothing was removed, the narrowed and pooled readings
    ARE the same reading, and a note would report that nothing happened."""
    rows = [
        {
            "alert": {
                "lines": [
                    {"acting_code": "AC0001", "origin_unit": "U-AAA", "login_unit": "U-AAA"}
                ]
            }
        }
    ]
    c = _pair_check(evaluate_verdict(_elem_spec(), {"alert_feed": rows}, _PairAnalysis()))
    assert c.result == "pass", (c.result, c.observed)
    assert "entr" not in (c.detail or ""), c.detail


def test_an_element_scope_that_cannot_be_applied_says_so_rather_than_pooling_silently():
    """Two ways the declaration cannot reach an element, and both fall back to the pooled
    reading — so both have to be reported, or the pack reads as though it had narrowed."""
    # (a) the condition reads another source entirely.
    spec = _elem_spec()
    spec["sources"]["other"] = "other_feed"
    spec["conditions"][0]["left"] = {"source": "other", "field": "origin_unit"}
    spec["conditions"][0]["right"] = {"source": "other", "field": "login_unit"}
    c = _pair_check(
        evaluate_verdict(
            spec,
            {"alert_feed": _TWO_ENTRY_ALERT, "other_feed": _CROSS_ONLY},
            _PairAnalysis(),
        )
    )
    assert c.result == "pass", (c.result, c.observed)
    assert "reads no source that the ruleset discovers its subjects from" in c.detail
    # (b) the ruleset declares no repeated path to narrow within.
    spec = _elem_spec()
    spec["subject_discovery"] = {"source": "alert", "subject": ["acting_code"]}
    c = _pair_check(
        evaluate_verdict(spec, {"alert_feed": _TWO_ENTRY_ALERT}, _PairAnalysis())
    )
    assert "subject_discovery declares no path" in c.detail, c.detail


def test_no_identity_discovered_at_all_is_a_different_reason_and_says_which():
    """The one inapplicable case that is nobody's defect — and it must not borrow the note that
    blames the declaration.

    Where neither the incident nor the discovery walk yields an identity, the checks are asked of
    the retrieved records AS A WHOLE under one unnamed subject. There is no identity to narrow to,
    so the values are pooled — the same *reading* as an undeclared narrowing, for the opposite
    reason. Both have to be reported (the pack read as narrowed and it did not), and they have to
    be told apart: one is a pack fix, the other is what this run had to work with.
    """
    rows = [
        {
            "alert": {
                "lines": [
                    {"origin_unit": "U-AAA", "login_unit": "U-BBB"},
                    {"origin_unit": "U-CCC", "login_unit": "U-AAA"},
                ]
            }
        }
    ]
    v = evaluate_verdict(_elem_spec(), {"alert_feed": rows}, _ElemAnalysis())
    # ONE unnamed subject, carrying the engine's own stand-in heading rather than an identity.
    assert [s.subject_value for s in v.subjects] == ["(all retrieved records)"], [
        s.subject_value for s in v.subjects
    ]
    c = next(c for c in v.subjects[0].checks if c.id == "one_unit")
    # Pooled, so the cross pair still matches — which is exactly why the note is needed.
    assert c.result == "pass", (c.result, c.observed)
    assert "had no identity to narrow to" in c.detail, c.detail
    assert "pooled across every identity" in c.detail, c.detail
    # And NOT the declaration's reason: this ruleset declares a perfectly good path.
    assert "declares no path" not in c.detail, c.detail


def test_a_transposed_projection_carries_no_per_entry_identity_and_says_so():
    """The cheap projection sends one ARRAY PER LEAF rather than the entries, and a positional
    zip cannot tell which identity a value belongs to. Reported, never narrowed on a guess."""
    rows = [
        {
            "alert_lines_acting_code": ["AC0001", "AC0002"],
            "alert_lines_origin_unit": ["U-AAA", "U-CCC"],
            "alert_lines_login_unit": ["U-BBB", "U-CCC"],
        }
    ]
    c = _pair_check(evaluate_verdict(_elem_spec(), {"alert_feed": rows}, _PairAnalysis()))
    assert "resolved to no entry on the 1 row(s) read" in c.detail, c.detail


def test_the_two_narrowings_compose_within_one_identitys_own_entries():
    """The full live shape: one identity holding TWO entries on a record shared with another.

    Element scope removes the stranger's entry; `pair_by: record` over `records:` then keeps the
    identity's own two entries from answering for each other. Either alone still PASSes on a
    cross pair.
    """
    rows = [
        {
            "alert": {
                "lines": [
                    {"acting_code": "AC0001", "origin_unit": "U-AAA", "login_unit": "U-BBB"},
                    {"acting_code": "AC0001", "origin_unit": "U-CCC", "login_unit": "U-AAA"},
                    {"acting_code": "AC0002", "origin_unit": "U-DDD", "login_unit": "U-DDD"},
                ]
            }
        }
    ]
    cond = {
        "id": "one_unit",
        "label": "Both units on the entry are the same unit",
        "kind": "field_equality",
        "left": {"source": "alert", "records": "alert.lines", "field": "origin_unit"},
        "right": {"source": "alert", "records": "alert.lines", "field": "login_unit"},
        "subject_scope": "element",
        "pair_by": "record",
    }
    c = _pair_check(evaluate_verdict(_elem_spec(cond=cond), {"alert_feed": rows}, _PairAnalysis()))
    assert c.result == "fail", (c.result, c.observed, c.detail)
    assert "read within the 2 of 3 `alert.lines` entries" in c.detail, c.detail
    assert "compared WITHIN each of the 2 record(s)" in c.detail, c.detail
    # Neither narrowing alone gets there: the stranger's entry, or this identity's own two
    # entries crossed, each supply a matching pair.
    for drop in ("subject_scope", "pair_by"):
        half = dict(cond)
        half.pop(drop)
        assert (
            _pair_check(
                evaluate_verdict(_elem_spec(cond=half), {"alert_feed": rows}, _PairAnalysis())
            ).result
            == "pass"
        ), drop


# --- a source that was asked and did not answer --------------------------------
# `logs` has no key for a non-answer: a timed-out source is absent, exactly like an unplanned
# one. Both produce `unknown` conditions; `unanswered_sources` carries the reason.


def test_a_source_that_never_answered_is_named_with_its_reason():
    logs = dict(_comp_logs())
    logs.pop("register_feed")  # timed out -> absent, not empty
    v = evaluate_verdict(
        _comp_spec(),
        logs,
        _CompAnalysis(),
        unanswered_sources={"register_feed": "did not answer within its 1800s budget"},
    )
    c = _comp_check(v)
    assert c.result == "unknown", (c.result, c.detail)
    # The condition itself carries the reason, so a reader of the checks table is not left
    # comparing it against a source that answered with nothing.
    assert "did not answer" in c.detail, c.detail
    note = " ".join(v.subjects[0].notes or [])
    assert "source_unanswered=register_feed" in note, note
    assert "1800s" in note and "MISSING, not empty" in note, note


def test_a_source_that_answered_with_nothing_is_not_reported_as_a_non_answer():
    """The distinction only exists if the empty case stays untouched."""
    logs = _comp_logs(register=[])
    v = evaluate_verdict(_comp_spec(), logs, _CompAnalysis(), unanswered_sources={})
    note = " ".join(v.subjects[0].notes or [])
    assert "source_unanswered" not in note, note
    assert "did not answer" not in (_comp_check(v).detail or "")


def test_a_source_nobody_asked_is_not_reported_as_one_that_answered_with_nothing():
    """The third state: declared, absent, and not among the failures — never queried.

    Two ways it happens: the planner declined the source, or a follow-up pass was skipped
    because its harvest came back empty. Every evaluator reads `src_rows.get(logical, [])`,
    so both read as zero rows, and the condition must not phrase that as an empty answer.
    """
    logs = dict(_comp_logs())
    logs.pop("register_feed")  # never planned / pass skipped -> absent, and no failure either
    v = evaluate_verdict(_comp_spec(), logs, _CompAnalysis(), unanswered_sources={})
    c = _comp_check(v)
    assert c.result == "unknown", (c.result, c.detail)
    assert "NOT QUERIED" in (c.detail or ""), c.detail
    assert "never an empty answer" in (c.detail or ""), c.detail
    # ...and it must not borrow the vocabulary of the source that DID fail: an operator acts
    # differently on the two (a query to add vs a retrieval to re-run).
    assert "was asked and did not answer" not in (c.detail or ""), c.detail
    note = " ".join(v.subjects[0].notes or [])
    assert "source_unanswered" not in note, note


def test_an_empty_answer_is_still_an_answer_and_keeps_its_own_wording():
    """The new state only exists if the empty case is untouched: `logical_to_real` is keyed on
    MEMBERSHIP in logs, so a source answering `[]` is present and must not read as unasked."""
    v = evaluate_verdict(
        _comp_spec(), _comp_logs(register=[]), _CompAnalysis(), unanswered_sources={}
    )
    assert "NOT QUERIED" not in (_comp_check(v).detail or "")


# --- a source that answered, about somebody else --------------------------------
# The fourth state, and the one the three above cannot express. A source answered, with rows,
# and none of them is THIS subject's: the check reads `unknown` in the wording reserved for a
# field the projection did not return, so the operator is sent to fix the projection when the
# remedy is the retrieval scope. Every evaluator reads `src_rows.get(logical, [])`, which is
# already narrowed per subject, so the distinction exists in the engine and only in the engine.


def _absent_subject_spec():
    return {
        "key": "roster_case",
        "title": "Roster check",
        "subject_entity": "record",
        "subject_field": "rec",
        "sources": {"roster": "roster_src"},
        "conditions": [
            {
                "id": "flagged",
                "label": "The record carries the marker",
                "kind": "field_flag",
                "source": "roster",
                "flag_fields": ["marker"],
                "expected": True,
            }
        ],
    }


class _TwoRecords:
    def __init__(self, records=("SUBJ03", "SUBJ99")):
        self.extracted_entities = [
            ExtractedEntity(type="record", value=r) for r in records
        ]


def _absent_subject_checks(logs):
    v = evaluate_verdict(
        _absent_subject_spec(), logs, _TwoRecords(), unanswered_sources={}
    )
    return {s.subject_value: s.checks[0] for s in v.subjects}


def test_a_subject_absent_from_rows_that_came_back_is_not_a_missing_field():
    """The fix, with its control in the same run.

    `SUBJ99` is one of the records the incident named and no retrieved row mentions it, so its
    check is unresolved for a reason the engine knows and was not saying: the rows came back and
    belong to somebody else. The remedy is the scope, so the note has to name it — a reader sent
    to the projection re-reads a declaration that is already correct.
    """
    checks = _absent_subject_checks({"roster_src": [{"rec": "SUBJ03", "marker": "yes"}]})
    # The control: the subject the rows DO name resolves, so the note is not boilerplate on
    # every condition of every multi-subject run.
    assert checks["SUBJ03"].result == "pass", checks["SUBJ03"].detail
    assert "NOT ONE of them names" not in (checks["SUBJ03"].detail or "")

    absent = checks["SUBJ99"]
    assert absent.result == "unknown", (absent.result, absent.detail)
    assert "NOT ONE of them names SUBJ99" in absent.detail, absent.detail
    assert "scope gap, not a missing field" in absent.detail, absent.detail
    # ...and it must not borrow either neighbouring vocabulary: an operator acts differently on
    # a source that did not answer, one nobody asked, and one that answered about other rows.
    assert "NOT QUERIED" not in absent.detail, absent.detail
    assert "was asked and did not answer" not in absent.detail, absent.detail


def test_a_source_that_answered_with_nothing_is_not_reported_as_answering_about_others():
    """The bound that makes the note true rather than merely new.

    With zero rows for everybody there are no other subjects' rows to point at, and claiming
    otherwise would describe an empty answer as a populated one — the same conflation in the
    opposite direction. The check stays `unknown` on its own wording, which is where the pack's
    `zero_rows` meaning speaks.
    """
    checks = _absent_subject_checks({"roster_src": []})
    absent = checks["SUBJ99"]
    assert absent.result == "unknown", (absent.result, absent.detail)
    assert "NOT ONE of them names" not in (absent.detail or ""), absent.detail
    assert "scope gap" not in (absent.detail or ""), absent.detail


def test_a_subject_with_rows_of_its_own_still_reads_as_a_missing_field():
    """The inverse direction, and the one the fix must not take away.

    `SUBJ03` has a row of its own and that row carries no `marker`, so the projection note is the
    TRUE one — this is the case the third note was written for. Both subjects here read `unknown`
    off the same source, for opposite reasons, which is the whole distinction: telling the one
    whose rows are present that none of them names it would be the original conflation running
    backwards, sending an operator to widen a scope that is already right.
    """
    checks = _absent_subject_checks(
        {"roster_src": [{"rec": "SUBJ03"}, {"rec": "SUBJ99", "marker": "yes"}]}
    )
    present = checks["SUBJ03"]
    assert present.result == "unknown", (present.result, present.detail)
    assert "not present in the retrieved rows" in present.detail, present.detail
    assert "NOT ONE of them names" not in present.detail, present.detail
    assert "scope gap" not in present.detail, present.detail
    # And the subject whose row DOES carry the field resolves, so neither reading is a blanket
    # property of this source on this run.
    assert checks["SUBJ99"].result == "pass", checks["SUBJ99"].detail


def test_a_cohort_condition_is_not_told_its_rows_are_somebody_elses():
    """The other direction the note can be false in, and the one a live pack really declares.

    `subject_scope: false` means the QUERY carried the scope, so the condition reads the source
    whole and none of its rows naming this subject is the normal state — the rows it read are
    real and its `unknown` is about something else entirely (here: the marker is on no row). The
    note is a claim about the subject-narrowed rows, so it may only be stamped where those are
    what the condition reads.
    """
    spec = _absent_subject_spec()
    spec["conditions"][0]["subject_scope"] = False
    v = evaluate_verdict(spec, {"roster_src": [{"rec": "SUBJ03"}]}, _TwoRecords())
    absent = {s.subject_value: s.checks[0] for s in v.subjects}["SUBJ99"]
    assert absent.result == "unknown", (absent.result, absent.detail)
    assert "not present in the retrieved rows" in absent.detail, absent.detail
    assert "NOT ONE of them names" not in absent.detail, absent.detail
    assert "scope gap" not in absent.detail, absent.detail


def test_a_cohort_composites_children_are_scoped_the_way_the_parent_is():
    """The same rule one level down, which is a separate assertion because the flag is elsewhere.

    Only the ROOT condition's `subject_scope` is read by the evaluation loop, so a child is
    scoped the way its parent is however the child is declared — and a child's detail rolls up
    into the parent's report line. Stamping the note per condition dict rather than inheriting it
    would put the false claim on the one line an operator actually reads.
    """
    spec = _absent_subject_spec()
    leaf = spec["conditions"][0]
    spec["conditions"] = [
        {
            "id": "flagged_all",
            "label": "Every marker requirement holds",
            "kind": "all_of",
            "subject_scope": False,
            "children": [leaf],
        }
    ]
    v = evaluate_verdict(spec, {"roster_src": [{"rec": "SUBJ03"}]}, _TwoRecords())
    absent = {s.subject_value: s.checks[0] for s in v.subjects}["SUBJ99"]
    assert absent.result == "unknown", (absent.result, absent.detail)
    assert "NOT ONE of them names" not in absent.detail, absent.detail


def test_a_source_that_never_answered_keeps_its_own_note_over_the_absent_subject_one():
    """Precedence, asserted rather than inferred from the branch order.

    A subject absent from rows that arrived and a subject whose source never answered are both
    `unknown` with no rows for this subject, and only the first is a scope question. The second
    is the more specific fact and states a different remedy, so it wins.
    """
    spec = _absent_subject_spec()
    # A second source that DID answer, or `logical_to_real` is empty and there is no verdict to
    # read at all — a run where nothing answered is a different case from the one under test.
    spec["sources"]["other"] = "other_src"
    v = evaluate_verdict(
        spec,
        {"other_src": [{"rec": "SUBJ03"}]},
        _TwoRecords(),
        unanswered_sources={"roster_src": "timed out"},
    )
    absent = {s.subject_value: s.checks[0] for s in v.subjects}["SUBJ99"]
    assert absent.result == "unknown", (absent.result, absent.detail)
    assert "was asked and did not answer" in absent.detail, absent.detail
    assert "NOT ONE of them names" not in absent.detail, absent.detail


def test_a_non_answer_never_overwrites_the_narrowing_note_that_ran():
    """A note about rows that DID come back is the more specific of the two, so a source
    that answered keeps its own note even while a sibling is reported unanswered."""
    logs = dict(_comp_logs())
    v = evaluate_verdict(
        _comp_spec(),
        logs,
        _CompAnalysis(),
        # Names a source this ruleset DOES declare but which came back with rows: the
        # engine must trust `logs` over a stale claim, not annotate a source it read.
        unanswered_sources={"register_feed": "did not answer within its 1800s budget"},
    )
    assert _comp_check(v).result == "fail"
    assert "source_unanswered" not in " ".join(v.subjects[0].notes or [])


def test_a_source_the_ruleset_does_not_declare_is_not_this_procedures_gap():
    v = evaluate_verdict(
        _comp_spec(),
        _comp_logs(),
        _CompAnalysis(),
        unanswered_sources={"some_other_procedures_feed": "did not answer"},
    )
    assert "source_unanswered" not in " ".join(v.subjects[0].notes or [])


def test_omitting_the_argument_leaves_the_verdict_byte_identical():
    """Every caller that predates the argument must produce what it produced before."""
    logs = dict(_comp_logs())
    logs.pop("register_feed")
    a = evaluate_verdict(_comp_spec(), logs, _CompAnalysis())
    b = evaluate_verdict(_comp_spec(), logs, _CompAnalysis(), unanswered_sources={})
    assert a.model_dump() == b.model_dump()


# --- the incident's own alert record is the only one adjudicated ------------------------
# An alert index scopes to the detector's window, returning documents for other incidents too.
# `alert_record.identify` declares how to tell them apart; the verdict filters on those clauses.


def _foreign_spec(**over):
    """One ruleset that identifies its alert record and discovers subjects from it."""
    spec = {
        "label_scheme": "scheme",
        "subject_entity": "actor",
        "labels": {
            "fraud": "VALID FRAUD",
            "false_positive": "FALSE POSITIVE",
            "insufficient": "INSUFFICIENT DATA",
        },
        "sources": {"alert": "alert_feed"},
        "alert_record": {
            "source": "alert",
            "label_fields": ["alert.record"],
            "identify": [
                {"from_entity": "record", "fields": ["alert.record"], "required": True},
                # Not required, and true of every document in the window — the shape that
                # makes the over-broad retrieval happen in the first place.
                {"from_entity": "carrier", "fields": ["alert.carrier"]},
            ],
        },
        "subject_discovery": {
            "source": "alert",
            "path": "alert.lines",
            "subject": ["acting_code"],
        },
        "conditions": [
            {
                "id": "one_actor",
                "label": "One acting identity across the whole alert",
                "kind": "distinct_count",
                "source": "alert",
                "field": "alert.lines.acting_code",
                "max": 1,
            }
        ],
    }
    spec.update(over)
    return spec


def _foreign_analysis(*records):
    """The incident names its record(s) and the carrier — never the acting identity."""
    from types import SimpleNamespace

    return SimpleNamespace(
        extracted_entities=(
            [ExtractedEntity(type="record", value=r) for r in records]
            + [ExtractedEntity(type="carrier", value="ZZ")]
        ),
        event_time=None,
    )


#: Three documents on one carrier in one window: the incident's, and two other incidents'.
_THREE_ALERTS = [
    {"alert": {"record": "OURS", "carrier": "ZZ", "lines": [{"acting_code": "AC0001"}]}},
    {"alert": {"record": "THEIRS1", "carrier": "ZZ", "lines": [{"acting_code": "AC0002"}]}},
    {"alert": {"record": "THEIRS2", "carrier": "ZZ", "lines": [{"acting_code": "AC0003"}]}},
]


def test_another_incidents_alert_document_is_neither_a_subject_nor_a_count():
    v = evaluate_verdict(_foreign_spec(), {"alert_feed": _THREE_ALERTS}, _foreign_analysis("OURS"))
    # Only this incident's acting identity is adjudicated. The other two are not outvoted,
    # not cleared and not reported as explained — they are not subjects at all.
    assert [s.subject_value for s in v.subjects] == ["AC0001"], [
        s.subject_value for s in v.subjects
    ]
    # ...and the alert-spanning count is over this incident's document only, so a
    # single-identity alert reads as one. Unfiltered it counted 3 and FAILED.
    check = next(c for c in v.subjects[0].checks if c.id == "one_actor")
    assert check.result == "pass", (check.result, check.observed)
    assert "AC0002" not in check.observed and "AC0003" not in check.observed


def test_the_narrowing_is_stated_and_names_the_records_it_cut():
    """A filter that changes which rows an adjudication saw is part of the finding.

    Named, not counted: a reader comparing the report to the alert has to be able to check
    the cut, and "2 records were excluded" invites the question the label answers.
    """
    v = evaluate_verdict(_foreign_spec(), {"alert_feed": _THREE_ALERTS}, _foreign_analysis("OURS"))
    note = next((n for n in v.subjects[0].notes if n.startswith("alert_scope=")), "")
    assert note, v.subjects[0].notes
    assert "alert_feed" in note and "1 of 3" in note
    assert "THEIRS1" in note and "THEIRS2" in note
    assert "another incident's alert" in note


def test_an_incident_that_named_no_key_drops_nothing():
    """The first bound. A required clause whose entity was never extracted is not evaluable —
    exactly as in `build_alert_facts` — so there is nothing to key on and every row stands.
    A pack whose incident text named only an amount gets the prior behaviour byte for byte."""
    from types import SimpleNamespace

    analysis = SimpleNamespace(
        extracted_entities=[ExtractedEntity(type="carrier", value="ZZ")], event_time=None
    )
    v = evaluate_verdict(_foreign_spec(), {"alert_feed": _THREE_ALERTS}, analysis)
    assert sorted(s.subject_value for s in v.subjects) == ["AC0001", "AC0002", "AC0003"]
    assert not any(n.startswith("alert_scope=") for s in v.subjects for n in s.notes)


def test_when_no_record_carries_the_incidents_identifiers_nothing_is_dropped():
    """The second bound, and the one that must fail OPEN.

    Zero rows on the alert source is the invisible failure this engine fights everywhere
    else: every condition goes `unknown` and the verdict reads INSUFFICIENT DATA, which is
    indistinguishable from a detector that alleged nothing. "No row carries this incident's
    identifiers" is a finding about the RETRIEVAL, and `build_alert_facts` already reports it
    in those words — so the verdict adjudicates what it has and says nothing about a cut.
    """
    v = evaluate_verdict(
        _foreign_spec(), {"alert_feed": _THREE_ALERTS}, _foreign_analysis("NOT-IN-ANY-ROW")
    )
    assert sorted(s.subject_value for s in v.subjects) == ["AC0001", "AC0002", "AC0003"]
    assert not any(n.startswith("alert_scope=") for s in v.subjects for n in s.notes)


def test_a_ruleset_declaring_no_alert_record_is_untouched():
    spec = _foreign_spec()
    spec.pop("alert_record")
    a = evaluate_verdict(spec, {"alert_feed": _THREE_ALERTS}, _foreign_analysis("OURS"))
    b = evaluate_verdict(
        _foreign_spec(), {"alert_feed": _THREE_ALERTS}, _foreign_analysis("OURS")
    )
    assert sorted(s.subject_value for s in a.subjects) == ["AC0001", "AC0002", "AC0003"]
    assert [s.subject_value for s in b.subjects] == ["AC0001"]


def test_several_documents_can_be_one_incidents_alert():
    """The narrowing is per ROW against the incident's OWN values, not "keep exactly one".

    A detector may emit one document per record, and an incident naming two records is then
    two documents — both this incident's. Selecting a single winning row (as the fact
    reconciliation does, which needs exactly one) would silently halve such a case.
    """
    v = evaluate_verdict(
        _foreign_spec(),
        {"alert_feed": _THREE_ALERTS},
        _foreign_analysis("OURS", "THEIRS1"),
    )
    assert sorted(s.subject_value for s in v.subjects) == ["AC0001", "AC0002"]
    note = next((n for n in v.subjects[0].notes if n.startswith("alert_scope=")), "")
    assert "2 of 3" in note and "THEIRS2" in note


def test_the_verdict_and_the_reconciliation_read_the_same_clauses():
    """One question, one answer. The two seams share `_match_row`/`_entity_values` rather than
    each carrying a copy, because the defect above WAS the two disagreeing: the reconciliation
    named THEIRS1/THEIRS2 as unrelated in the same report that adjudicated them."""
    from src.usecases.base import UseCaseAnalyzer

    spec = _foreign_spec()
    logs = {"alert_feed": _THREE_ALERTS}
    analysis = _foreign_analysis("OURS")
    facts = UseCaseAnalyzer.build_alert_facts(logs, spec, analysis)
    assert facts is not None and facts.located and facts.record_id == "OURS"
    assert sorted(facts.unrelated_records) == ["THEIRS1", "THEIRS2"]
    v = evaluate_verdict(spec, logs, analysis)
    note = next(n for n in v.subjects[0].notes if n.startswith("alert_scope="))
    # The same two records, from the same declaration, on both sides.
    assert all(r in note for r in facts.unrelated_records)


def test_unknown_detail_is_honoured_by_every_kind_not_only_the_four_that_read_it():
    """A pack's `unknown_detail` was load-bearing prose on one kind and a silent no-op on ten.

    Four evaluators read the key themselves (`time_gap`, `field_flag`, `distinct_count`,
    `route_membership`); the rest hard-coded their sentence, so the same declaration reached
    the report from one condition and vanished from the next — the `expected_label` trap one
    key over, and invisible from the pack, where both look identical. The seam is `mk`, so
    this asserts the kind that motivated it AND one that already worked, because a fix that
    moves the honouring must not drop it where it was.
    """
    from src.correlation import _eval_condition

    said = (
        "either the creating sign or the issuing sign did not resolve on the versions of "
        "this identity's own record"
    )
    two_sided = {
        "id": "same_agent",
        "label": "Created and ticketed by the same sign",
        "kind": "field_equality",
        "source": "record",
        "left": {"field": "creator.sign"},
        "right": {"field": "issuer.sign"},
        "normalize": "identifier",
        "unknown_detail": said,
    }
    # One side present, the other not: the branch that hard-coded its own sentence.
    c = _eval_condition(two_sided, {"record": [{"creator.sign": "6001AASU"}]})
    assert c.result == "unknown"
    assert c.detail == said
    # Undeclared -> the engine's own default, unchanged.
    bare = {k: v for k, v in two_sided.items() if k != "unknown_detail"}
    assert (
        _eval_condition(bare, {"record": [{"creator.sign": "6001AASU"}]}).detail
        == "one side of the comparison had no data"
    )
    # And the kind that already honoured it still does.
    gap = {
        "id": "issued_promptly",
        "label": "Issued within an hour of creation",
        "kind": "time_gap",
        "source": "record",
        "start": {"field": "created_at"},
        "end": {"field": "issued_at"},
        "max": "1h",
        "unknown_detail": "this record carries no issuing timestamp",
    }
    ct = _eval_condition(gap, {"record": [{"created_at": "2026-08-01T10:00:00Z"}]})
    assert ct.result == "unknown"
    assert ct.detail == "this record carries no issuing timestamp"


def test_a_pack_sentence_does_not_overwrite_an_engine_scope_diagnosis():
    """`unknown_detail` says what an absence of DATA means, and some unknowns are not that.

    "This check declares no `subject_rows`", "the subject is not in the cohort", "the source
    was cut off at its cap" are statements about the declaration or the scope — a pack
    sentence there would report something that was never measured, and each of those branches
    already carries its own key or its own diagnosis. So they are exempt, and the exemption is
    asserted rather than assumed: it is the whole reason the honouring is a parameter of `mk`
    and not an unconditional override.
    """
    from src.correlation import _eval_condition

    said = "the register returned nothing for this identity"
    cohort = {
        "id": "key_elsewhere",
        "label": "The subject's key appears on no other record",
        "kind": "cohort_membership",
        "source": "cohort",
        "key_fields": ["email"],
        "discriminators": ["locator"],
        "subject_rows": {"where": [{"field": "locator", "any_of": ["SUBJ01"]}]},
        "unknown_detail": said,
    }
    # Rows came back, and the subject's own selector matched none of them: a SCOPE error.
    rows = {"cohort": [{"locator": "OTHER1", "email": "a@b.c"}]}
    c = _eval_condition(cohort, rows)
    assert c.result == "unknown"
    assert "subject NOT IN the cohort" in c.observed
    assert said not in c.detail
    assert "does not cover the subject" in c.detail
    # Same for a placeholder, whose `detail` is the pack's own declaration already.
    stub = {
        "id": "planned",
        "label": "A check whose data path is not confirmed",
        "kind": "stub",
        "detail": "probe-confirmed absent from the structured record",
        "unknown_detail": said,
    }
    cs = _eval_condition(stub, {})
    assert cs.result == "unknown"
    assert cs.detail == "probe-confirmed absent from the structured record"


def test_a_scope_note_still_reaches_the_reader_beside_a_pack_sentence():
    """Two clauses, two jobs: WHAT an absence means (the pack) and WHY these rows (the engine).

    The scope note is appended to whatever detail the condition ended with, so replacing the
    engine's default must not consume it — a `row_match` that could not be filled is exactly
    the case where the pack's sentence is least sufficient on its own.
    """
    from src.correlation import _eval_condition

    cond = {
        "id": "same_agent",
        "label": "Created and ticketed by the same sign",
        "kind": "field_equality",
        "source": "record",
        "left": {"field": "creator.sign"},
        "right": {"field": "issuer.sign"},
        "unknown_detail": "this identity's own record carried neither sign",
        "_scope_note": "scope=no row of this source carried this identity's record",
    }
    c = _eval_condition(cond, {"record": [{"creator.sign": "6001AASU"}]})
    assert c.result == "unknown"
    assert c.detail.startswith("this identity's own record carried neither sign")
    assert "scope=no row of this source carried this identity's record" in c.detail


# ------------------------------------------------------------------ numeric_compare


def _num_rows():
    return [
        {"actor": "A1", "amount": 100, "ref": "R1"},
        {"actor": "A1", "amount": 250, "ref": "R2"},
        {"actor": "A2", "amount": 40, "ref": "R3"},
        {"actor": "A2", "amount": 40, "ref": "R4"},
    ]


def _num_cond(**over):
    cond = {
        "id": "vol",
        "label": "Volume within the declared bound",
        "kind": "numeric_compare",
        "source": "events",
        "aggregate": "count",
        "operator": "<=",
        "bound": 3,
    }
    cond.update(over)
    return cond


def test_numeric_compare_computes_each_aggregate():
    """One kind covering the aggregate-then-compare family, so a new shape is YAML not Python."""
    from src.correlation import _eval_condition

    rows = {"events": _num_rows()}
    cases = [
        ("count", "", "<=", 4, "pass", "count = 4"),
        ("count", "", "<", 4, "fail", "count = 4"),
        ("distinct", "actor", "<=", 1, "fail", "distinct = 2"),
        ("distinct", "actor", "==", 2, "pass", "distinct = 2"),
        ("sum", "amount", ">", 400, "pass", "sum = 430"),
        ("min", "amount", ">=", 40, "pass", "min = 40"),
        ("max", "amount", "<", 250, "fail", "max = 250"),
        ("avg", "amount", "<=", 107.5, "pass", "avg = 107.5"),
    ]
    for agg, field, op, bound, want, observed in cases:
        c = _eval_condition(
            _num_cond(aggregate=agg, field=field, operator=op, bound=bound), rows
        )
        assert c.result == want, (agg, op, bound, c.result, c.observed)
        assert observed in c.observed, (agg, c.observed)


def test_numeric_compare_reads_a_fractional_bound():
    """`_declared_bound` coerces with `int()`, so a ratio bound routed through it becomes 0 —
    a bound every value satisfies. `numeric_compare` reads its bound as a real number."""
    from src.correlation import _declared_bound, _declared_number

    assert _declared_bound({"max": 0.25}) == 0  # the coercion this kind must not use
    assert _declared_number({"bound": 0.25}, "bound") == 0.25

    from src.correlation import _eval_condition

    # 2 of 4 rows match, i.e. 0.5 — above 0.25 and below 0.75, which a 0 bound cannot see.
    cond = _num_cond(
        aggregate="ratio",
        operator="<=",
        bound=0.25,
        where=[{"field": "actor", "any_of": ["A1"]}],
    )
    c = _eval_condition(cond, {"events": _num_rows()})
    assert c.result == "fail", c.observed
    assert "ratio = 0.5" in c.observed and "2 of 4 row(s)" in c.observed
    assert _eval_condition({**cond, "bound": 0.75}, {"events": _num_rows()}).result == "pass"


def test_numeric_compare_groups_and_reports_the_deciding_group():
    """`group_by` asks the question per group; the largest group answers either reading of an
    upper bound, which is why an equality against it is refused rather than answered."""
    from src.correlation import _eval_condition

    rows = {"events": _num_rows()}
    over = _num_cond(aggregate="count", group_by="actor", operator="<=", bound=1)
    c = _eval_condition(over, rows)
    assert c.result == "fail" and "count = 2" in c.observed
    assert "largest of 2 group(s): a1" in c.observed.lower(), c.observed
    assert _eval_condition({**over, "bound": 2}, rows).result == "pass"

    refused = _eval_condition({**over, "operator": "==", "bound": 2}, rows)
    assert refused.result == "unknown"
    assert "names no deciding group" in refused.detail


def test_numeric_compare_holds_only_where_truncation_cannot_flip_it():
    """A truncated read is a BOUND on the value, not the value. The finding stands only where
    the rows that never came back could not have changed it — a count already past its ceiling
    stays past it, one below it does not stay below."""
    from src.correlation import _eval_condition

    rows = {"events": _num_rows()}

    # count rises with more rows: `> 3` survives, `<= 4` does not.
    over = _eval_condition(
        _num_cond(operator=">", bound=3, _count_truncated=["events"]), rows
    )
    assert over.result == "pass" and "TRUNCATED" not in over.observed

    under = _eval_condition(
        _num_cond(operator="<=", bound=4, _count_truncated=["events"]), rows
    )
    assert under.result == "unknown"
    assert "TRUNCATED" in under.observed and "bound and not the finding" in under.detail

    # min falls with more rows, so the directions invert.
    low = _num_cond(aggregate="min", field="amount", _count_truncated=["events"])
    assert _eval_condition({**low, "operator": "<", "bound": 50}, rows).result == "pass"
    assert _eval_condition({**low, "operator": ">", "bound": 10}, rows).result == "unknown"

    # avg and ratio move either way, so neither ever survives a truncated read.
    for agg in ("avg", "ratio"):
        c = _eval_condition(
            _num_cond(
                aggregate=agg,
                field="amount",
                operator="<",
                bound=1000,
                _count_truncated=["events"],
            ),
            rows,
        )
        assert c.result == "unknown", (agg, c.result)

    # A sum over a column carrying negatives is not monotone either.
    credits = {"events": [{"amount": 10}, {"amount": -4}]}
    c = _eval_condition(
        _num_cond(
            aggregate="sum", field="amount", operator=">", bound=1,
            _count_truncated=["events"],
        ),
        credits,
    )
    assert c.result == "unknown", c.observed


def test_numeric_compare_splits_a_real_zero_from_a_retrieval_gap():
    """Zero rows is an answer only where the query was keyed and the aggregate has one.
    Nothing has no minimum, and rows-with-no-field is a projection gap, never a zero."""
    from src.correlation import _eval_condition

    keyed = {"_count_scope_resolved_empty": True}
    c = _eval_condition(
        _num_cond(aggregate="count", operator="==", bound=0, **keyed), {"events": []}
    )
    assert c.result == "pass" and "count = 0" in c.observed

    for agg in ("min", "max", "avg", "ratio"):
        gap = _eval_condition(
            _num_cond(aggregate=agg, field="amount", operator=">", bound=0, **keyed),
            {"events": []},
        )
        assert gap.result == "unknown", agg

    # No flag at all: an unkeyed empty source proves nothing.
    assert (
        _eval_condition(_num_cond(operator="==", bound=0), {"events": []}).result
        == "unknown"
    )
    # Rows present, field absent from the projection.
    absent = _eval_condition(
        _num_cond(aggregate="sum", field="missing", operator=">", bound=0),
        {"events": _num_rows()},
    )
    assert absent.result == "unknown" and "could not be read" in absent.detail


def test_numeric_compare_refuses_an_unreadable_declaration():
    """An aggregate, operator or bound the engine cannot read is `unknown` WITH the reason —
    a decisive one reads as INSUFFICIENT DATA, so the report must name the declaration."""
    from src.correlation import _eval_condition

    rows = {"events": _num_rows()}
    for over in ({"aggregate": "p95"}, {"operator": "!="}, {"bound": None}):
        c = _eval_condition(_num_cond(**over), rows)
        assert c.result == "unknown", over
        assert "readable `aggregate`, `operator` and `bound`" in c.detail, over


def test_numeric_compare_falls_back_to_a_second_source():
    """Same fallback vocabulary as the counting kinds: the first candidate that yields a
    readable value decides, and the report names which one it was."""
    from src.correlation import _eval_condition

    cond = _num_cond(
        source="primary",
        aggregate="sum",
        field="amount",
        operator=">",
        bound=100,
        fallbacks=[{"source": "events", "field": "amount"}],
    )
    c = _eval_condition(cond, {"primary": [], "events": _num_rows()})
    assert c.result == "pass" and "[from events.amount]" in c.observed


def test_a_modal_aggregate_names_the_value_it_found():
    """The winner's identity IS the finding, so it has to reach `observed`.

    `sum`/`min`/`max`/`avg`/`median` need numbers, so before these two a concentration question
    could only be asked about a value the pack named up front (`where` + `ratio`). "Whichever
    value is most frequent, and what share it holds" is what a targeting pattern is — and a
    report reading `mode = 2 (>= 2)` states a concentration and names nothing, which is the same
    defect as a label printing its requirement instead of its finding.
    """
    from src.correlation import _eval_condition

    rows = {"events": _num_rows()}  # actor: A1, A1, A2, A2 — two values, two rows each

    top = _eval_condition(
        _num_cond(aggregate="mode", field="actor", operator=">=", bound=2), rows
    )
    assert top.result == "pass" and "mode = 2" in top.observed
    assert "most frequent: A1 on 2 of 4 row(s)" in top.observed, top.observed
    assert "2 distinct value(s)" in top.observed, top.observed

    shape = _num_cond(aggregate="mode_share", field="actor", operator=">=", bound=0.5)
    share = _eval_condition(shape, rows)
    assert share.result == "pass" and "mode_share = 0.5" in share.observed
    assert "most frequent: A1" in share.observed, share.observed
    # A fractional bound is the whole point of a share: routed through `int()` it would be 0.
    assert _eval_condition({**shape, "bound": 0.75}, rows).result == "fail"


def test_a_modal_tie_resolves_the_same_way_every_time():
    """A dict-iteration winner would name a different value on two evaluations of one row set.

    Both values here occur twice, so the tie is real, and the rule is the lowest normalised
    value — asserted over REVERSED rows too, because insertion order is what a backend decides
    and a verdict that is not reproducible is not a verdict.
    """
    from src.correlation import _eval_condition, _modal_counts

    cond = _num_cond(aggregate="mode", field="actor", operator=">=", bound=2)
    first = _eval_condition(cond, {"events": _num_rows()})
    second = _eval_condition(cond, {"events": _num_rows()})
    assert first.observed == second.observed
    assert "most frequent: A1" in first.observed, first.observed

    flipped = _eval_condition(cond, {"events": list(reversed(_num_rows()))})
    assert "most frequent: A1" in flipped.observed, flipped.observed

    # The rule stated directly: frequency descending, then the NORMALISED value ascending —
    # `_norm_identifier` folds to upper case, so that is the form the tie is broken on.
    ranked = _modal_counts([{"a": "z"}, {"a": "z"}, {"a": "m"}, {"a": "m"}, {"a": "q"}], "a")
    assert ranked == [("M", 2), ("Z", 2), ("Q", 1)], ranked


def test_a_truncated_modal_read_keeps_the_number_and_marks_the_name():
    """The two halves of a modal answer have DIFFERENT monotonicity, so they degrade apart.

    The top frequency only rises, so a `>=` conclusion survives the rows that never came back;
    the winner's identity does not, because those rows can carry a different value entirely. A
    share is non-monotone in both terms and is therefore always `unknown` under truncation,
    exactly like `ratio`.
    """
    from src.correlation import _eval_condition

    rows = {"events": _num_rows()}

    stands = _eval_condition(
        _num_cond(
            aggregate="mode",
            field="actor",
            operator=">=",
            bound=2,
            _count_truncated=["events"],
        ),
        rows,
    )
    assert stands.result == "pass", stands.observed
    assert "PROVISIONAL: the read was truncated" in stands.observed, stands.observed

    falls = _eval_condition(
        _num_cond(
            aggregate="mode",
            field="actor",
            operator="<=",
            bound=2,
            _count_truncated=["events"],
        ),
        rows,
    )
    assert falls.result == "unknown", falls.observed

    for op, bound in ((">=", 0.5), ("<=", 0.5)):
        share = _eval_condition(
            _num_cond(
                aggregate="mode_share",
                field="actor",
                operator=op,
                bound=bound,
                _count_truncated=["events"],
            ),
            rows,
        )
        assert share.result == "unknown", (op, share.observed)

    # And an untruncated read carries no marker, or every reader learns to ignore it.
    clean = _eval_condition(
        _num_cond(aggregate="mode", field="actor", operator=">=", bound=2), rows
    )
    assert "PROVISIONAL" not in clean.observed, clean.observed


def test_a_modal_aggregate_with_no_field_has_nothing_to_rank():
    """`count` reads the rows themselves; a modal aggregate cannot. With no field there are no
    values to rank, so it reports `unknown` with the reason rather than a frequency of rows."""
    from src.correlation import _eval_condition

    c = _eval_condition(
        _num_cond(aggregate="mode", field="", operator=">=", bound=1),
        {"events": _num_rows()},
    )
    assert c.result == "unknown" and "could not be read" in c.detail, c.detail

    # A field that never arrived in the projection is the same answer, not a zero.
    absent = _eval_condition(
        _num_cond(aggregate="mode", field="missing", operator=">=", bound=1),
        {"events": _num_rows()},
    )
    assert absent.result == "unknown", absent.observed


def test_a_grouped_modal_aggregate_names_the_winner_inside_its_group():
    """`group_by` and a modal aggregate answer two different questions at once — which group is
    most concentrated, and on which value — so `observed` has to carry both names."""
    from src.correlation import _eval_condition

    rows = {
        "events": [
            {"actor": "A1", "ref": "R1"},
            {"actor": "A1", "ref": "R1"},
            {"actor": "A1", "ref": "R2"},
            {"actor": "A2", "ref": "R9"},
        ]
    }
    c = _eval_condition(
        _num_cond(
            aggregate="mode", field="ref", group_by="actor", operator=">=", bound=2
        ),
        rows,
    )
    assert c.result == "pass" and "mode = 2" in c.observed
    assert "largest of 2 group(s): A1" in c.observed, c.observed
    assert "most frequent: R1 on 2 of 3 row(s)" in c.observed, c.observed


def test_a_tie_between_groups_resolves_on_the_group_name():
    """The deciding group was picked in dict-insertion order, i.e. in the order the BACKEND
    returned the rows — so two runs of one incident could name two different groups off
    identical data. Same rule as the modal tie: the lowest key wins."""
    from src.correlation import _eval_condition

    rows = [
        {"actor": "A2", "amount": 10},
        {"actor": "A2", "amount": 10},
        {"actor": "A1", "amount": 10},
        {"actor": "A1", "amount": 10},
    ]
    cond = _num_cond(aggregate="count", group_by="actor", operator="<=", bound=1)
    forward = _eval_condition(cond, {"events": rows})
    reverse = _eval_condition(cond, {"events": list(reversed(rows))})
    assert "largest of 2 group(s): A1" in forward.observed, forward.observed
    assert forward.observed == reverse.observed


# ------------------------------------------------------- all_of / any_of / none_of


def _leaf(cid, result):
    """A child whose outcome is fixed by the rows it is handed."""
    if result == "unknown":
        return {"id": cid, "kind": "field_flag", "source": "missing", "field": "flag"}
    return {
        "id": cid,
        "kind": "numeric_compare",
        "source": "events",
        "aggregate": "count",
        "operator": "<=" if result == "pass" else "<",
        "bound": 4,
    }


def _composite(kind, *results):
    from src.correlation import _eval_condition

    cond = {
        "id": "combo",
        "label": "The combination holds",
        "kind": kind,
        "children": [_leaf(f"c{i}", r) for i, r in enumerate(results)],
        "fail_detail": "the combination did not hold",
    }
    return _eval_condition(cond, {"events": _num_rows()})


def test_composites_follow_three_valued_logic():
    """`unknown` must never read as `pass`. A composite is decided only where the undecided
    children could not have changed it — one fail settles `all_of`, one pass settles `any_of`."""
    cases = [
        ("all_of", ("pass", "pass"), "pass"),
        ("all_of", ("pass", "fail"), "fail"),
        ("all_of", ("pass", "unknown"), "unknown"),
        ("all_of", ("fail", "unknown"), "fail"),
        ("any_of", ("fail", "pass"), "pass"),
        ("any_of", ("fail", "fail"), "fail"),
        ("any_of", ("fail", "unknown"), "unknown"),
        ("any_of", ("pass", "unknown"), "pass"),
        ("none_of", ("fail", "fail"), "pass"),
        ("none_of", ("fail", "pass"), "fail"),
        ("none_of", ("fail", "unknown"), "unknown"),
        ("none_of", ("pass", "unknown"), "fail"),
    ]
    for kind, results, want in cases:
        c = _composite(kind, *results)
        assert c.result == want, (kind, results, c.result, c.observed)


def test_a_composite_is_one_finding_carrying_the_parents_weight():
    """The children are this check's mechanics, not checks of their own: one report line, the
    parent's label and `fail_detail`, and the deciding children named in `observed`."""
    from src.correlation import _eval_condition

    cond = {
        "id": "combo",
        "label": "Both identity checks hold",
        "kind": "all_of",
        "decisive": True,
        "polarity": "exclusion",
        "report_group": "identity",
        "fail_detail": "one leg of the identity check did not hold",
        "children": [_leaf("first", "pass"), _leaf("second", "fail")],
    }
    c = _eval_condition(cond, {"events": _num_rows()})
    assert c.id == "combo" and c.label == "Both identity checks hold"
    assert c.result == "fail" and c.decisive is True and c.group == "identity"
    assert c.detail == "one leg of the identity check did not hold"
    assert "second fail" in c.observed and "first" not in c.observed
    assert "all of: first, second" in c.expected


def test_a_composite_refuses_a_declaration_it_cannot_combine():
    """No children and runaway nesting are authoring errors, so each keeps its own detail
    rather than being overwritten by the parent's `unknown_detail` about the data."""
    from src.correlation import _MAX_COMPOSITE_DEPTH, _eval_condition

    empty = _eval_condition(
        {"id": "combo", "kind": "any_of", "unknown_detail": "no rows"}, {}
    )
    assert empty.result == "unknown" and "declares no `children`" in empty.detail

    deep = {"id": "leaf", "kind": "all_of", "children": [_leaf("a", "pass")] * 2}
    for i in range(_MAX_COMPOSITE_DEPTH + 1):
        deep = {"id": f"n{i}", "kind": "all_of", "children": [deep, _leaf("x", "pass")]}
    c = _eval_condition(deep, {"events": _num_rows()})
    assert c.result == "unknown"
    assert f"nest at most {_MAX_COMPOSITE_DEPTH} deep" in c.detail


def test_a_composites_children_are_stamped_from_their_own_sources():
    """The parent->child rule. Run-level context (the resolved routes) is re-derived for each
    child; a source-derived fact is computed per child, so a SIBLING's truncated source must
    not turn a sound check into `unknown` — that is a false INSUFFICIENT DATA."""
    from src.correlation import evaluate_verdict

    spec = {
        "key": "combo_case",
        "title": "Composite scoping",
        "subject_entity": "record",
        "subject_field": "rec",
        "sources": {"sound": "sound_src", "capped": "capped_src"},
        "conditions": [
            {
                "id": "combo",
                "label": "Both legs hold",
                "kind": "all_of",
                "fail_detail": "a leg did not hold",
                "children": [
                    {
                        "id": "sound_leg",
                        "kind": "numeric_compare",
                        "source": "sound",
                        "aggregate": "count",
                        "operator": "<=",
                        "bound": 9,
                    },
                    {
                        "id": "capped_leg",
                        "kind": "numeric_compare",
                        "source": "capped",
                        "aggregate": "count",
                        "operator": "<=",
                        "bound": 9,
                    },
                ],
            },
            {
                "id": "sound_alone",
                "label": "The uncapped leg alone",
                "kind": "numeric_compare",
                "source": "sound",
                "aggregate": "count",
                "operator": "<=",
                "bound": 9,
            },
        ],
    }
    logs = {
        "sound_src": [{"rec": "SUBJ03"}],
        "capped_src": [{"rec": "SUBJ03"}, {"rec": "SUBJ03"}],
    }
    s = evaluate_verdict(
        spec, logs, _Analysis(), row_caps={"capped_src": 2}
    ).subjects[0]
    by_id = {c.id: c for c in s.checks}
    # The capped child hedges, so the combination is unknown...
    assert by_id["combo"].result == "unknown"
    # ...but the standalone condition reading only the sound source is unaffected, which is
    # what proves the truncation was not inherited across siblings.
    assert by_id["sound_alone"].result == "pass"


def test_a_composites_child_reaches_a_source_only_it_declares():
    """`_condition_sources` recurses into `children`, so the scope note for a source that did
    not answer reaches a composite whose parent names no source of its own."""
    from src.correlation import evaluate_verdict

    spec = {
        "key": "combo_case",
        "title": "Composite scoping",
        "subject_entity": "record",
        "subject_field": "rec",
        "sources": {"seen": "seen_src", "quiet": "quiet_src"},
        "conditions": [
            {
                "id": "combo",
                "label": "The combination holds",
                "kind": "any_of",
                "children": [
                    {
                        "id": "leg",
                        "kind": "numeric_compare",
                        "source": "quiet",
                        "aggregate": "count",
                        "operator": ">",
                        "bound": 0,
                    },
                    {"id": "other", "kind": "field_flag", "source": "quiet", "field": "f"},
                ],
            }
        ],
    }
    s = evaluate_verdict(
        spec,
        {"seen_src": [{"rec": "SUBJ03"}]},
        _Analysis(),
        unanswered_sources={"quiet_src": "timed out"},
    ).subjects[0]
    c = s.checks[0]
    assert c.result == "unknown"
    assert "quiet_src source was asked and did not answer" in c.detail, c.detail


def test_an_unrecognised_kind_is_logged_rather_than_only_reported(caplog):
    """A typo'd `kind:` reads in the report exactly like a source that returned nothing, so
    the engine must say so somewhere a reader of logs will find it. `stub` is the deliberate
    case and keeps its own wording."""
    from src.correlation import _eval_condition

    with caplog.at_level(logging.WARNING, logger="src.correlation"):
        c = _eval_condition({"id": "typo", "kind": "distinct_counts"}, {})
    assert c.result == "unknown"
    assert "unrecognised condition kind" in c.detail
    assert any("unrecognised kind" in r.getMessage() for r in caplog.records)


def _baseline_spec(**over):
    """A ruleset whose one condition compares a subject against a computed population."""
    cond = {
        "id": "rel_vol",
        "label": "Volume within a multiple of the population",
        "kind": "numeric_compare",
        "source": "acts",
        "aggregate": "count",
        "operator": "<=",
        "bound": 2,
        "baseline": {"source": "peers", "aggregate": "count"},
    }
    cond.update(over)
    return {
        "key": "rel_case",
        "title": "Relative comparison",
        "subject_entity": "record",
        "subject_field": "rec",
        "sources": {"acts": "acts_src", "peers": "peers_src"},
        "conditions": [cond],
    }


def test_a_baseline_makes_the_bound_a_multiple_of_a_population():
    """The shape the compositional vocabulary was extended for: a threshold the pack cannot
    state as a literal, because it is a property of the data rather than of the procedure."""
    from src.correlation import evaluate_verdict

    logs = {
        "acts_src": [{"rec": "SUBJ03"}, {"rec": "SUBJ03"}, {"rec": "SUBJ03"}],
        "peers_src": [{"rec": "P1"}, {"rec": "P2"}],
    }
    c = evaluate_verdict(_baseline_spec(), logs, _Analysis()).subjects[0].checks[0]
    # 3 subject rows against 2x a baseline of 2 -> within the bound.
    assert c.result == "pass", c.detail
    assert "baseline = 2" in c.observed, c.observed
    # The resolved absolute threshold is stated too: `<= 2x baseline` alone is not checkable
    # by a reader holding the report.
    assert "<= 4" in c.detail, c.detail
    tight = evaluate_verdict(
        _baseline_spec(bound=1), logs, _Analysis()
    ).subjects[0].checks[0]
    assert tight.result == "fail", tight.detail


def test_a_baseline_reads_the_population_and_not_the_subjects_own_rows():
    """The trap this branch is built around: `src_rows` is subject-filtered, so a baseline
    read through it would compare the subject against ITSELF — a threshold that moves with the
    value it is testing, and a comparison that can never find an outlier."""
    from src.correlation import evaluate_verdict

    logs = {
        "acts_src": [{"rec": "SUBJ03"}] * 6,
        # The population is other records only; a subject-narrowed read finds none of them.
        "peers_src": [{"rec": "P1"}, {"rec": "P2"}],
    }
    c = evaluate_verdict(_baseline_spec(), logs, _Analysis()).subjects[0].checks[0]
    assert c.result == "fail", c.detail
    assert "baseline = 2" in c.observed, c.observed


def test_a_truncated_baseline_is_never_compared_against():
    """A partial population UNDERSTATES the threshold, so the error is one-directional: an
    ordinary subject reads as an outlier. Both directions asserted, or the guard is
    indistinguishable from a comparison that happened to come out `unknown`."""
    from src.correlation import evaluate_verdict

    logs = {
        "acts_src": [{"rec": "SUBJ03"}] * 6,
        "peers_src": [{"rec": "P1"}, {"rec": "P2"}],
    }
    capped = evaluate_verdict(
        _baseline_spec(), logs, _Analysis(), row_caps={"peers_src": 2}
    ).subjects[0].checks[0]
    assert capped.result == "unknown", capped.detail
    assert "row cap" in capped.detail and "outlier" in capped.detail, capped.detail
    # Same rows, cap not reached: the comparison runs.
    complete = evaluate_verdict(
        _baseline_spec(), logs, _Analysis(), row_caps={"peers_src": 9}
    ).subjects[0].checks[0]
    assert complete.result == "fail", complete.detail


def test_a_truncated_subject_side_does_not_read_as_a_baseline_gap():
    """The baseline's completeness is a fact about the baseline's OWN source. A cap on the
    subject side must still be reported as a truncated READING — hedged on its own terms and
    never as an absent population, because the two license different next steps."""
    from src.correlation import evaluate_verdict

    logs = {
        "acts_src": [{"rec": "SUBJ03"}] * 3,
        "peers_src": [{"rec": "P1"}, {"rec": "P2"}],
    }
    c = evaluate_verdict(
        _baseline_spec(bound=9), logs, _Analysis(), row_caps={"acts_src": 3}
    ).subjects[0].checks[0]
    assert c.result == "unknown", c.detail
    assert "TRUNCATED read" in c.observed, c.observed
    assert "baseline" not in c.detail, c.detail


def test_a_baseline_that_did_not_answer_is_not_a_baseline_of_zero():
    """Each non-answer names itself, because the remedies differ: a timeout is re-runnable, a
    source nobody queried needs a plan edit, and an empty population needs neither."""
    from src.correlation import evaluate_verdict

    subject = {"acts_src": [{"rec": "SUBJ03"}] * 6}
    quiet = evaluate_verdict(
        _baseline_spec(),
        subject,
        _Analysis(),
        unanswered_sources={"peers_src": "timed out"},
    ).subjects[0].checks[0]
    assert quiet.result == "unknown"
    assert "did not answer" in quiet.detail, quiet.detail
    empty = evaluate_verdict(
        _baseline_spec(), {**subject, "peers_src": []}, _Analysis()
    ).subjects[0].checks[0]
    assert empty.result == "unknown"
    assert "no rows" in empty.detail, empty.detail


def test_a_zero_baseline_is_refused_because_every_multiplier_states_one_bound():
    """At a baseline of zero, `1x` and `3x` are the same threshold and every value above zero
    fails it — a bound no declaration can change, which is the shape of a fabricated finding
    rather than a strict one."""
    from src.correlation import evaluate_verdict

    logs = {
        "acts_src": [{"rec": "SUBJ03"}],
        "peers_src": [{"rec": "P1", "amount": 0}, {"rec": "P2", "amount": 0}],
    }
    c = evaluate_verdict(
        _baseline_spec(baseline={"source": "peers", "aggregate": "sum", "field": "amount"}),
        logs,
        _Analysis(),
    ).subjects[0].checks[0]
    assert c.result == "unknown", c.detail
    assert "every multiplier tests the same bound" in c.detail, c.detail


def test_a_cohort_baseline_reduces_per_member_before_combining():
    """`per` is the two-level shape a cohort comparison needs: the median of each peer's own
    total is not the median of their rows, and a peer with 50 rows would otherwise dominate a
    population of peers with one each."""
    from src.correlation import evaluate_verdict

    logs = {
        "acts_src": [{"rec": "SUBJ03", "amount": 10}] * 4,
        "peers_src": [
            {"rec": "P1", "amount": 1},
            {"rec": "P1", "amount": 1},
            {"rec": "P1", "amount": 1},
            {"rec": "P1", "amount": 1},
            {"rec": "P1", "amount": 1},
            {"rec": "P2", "amount": 1},
            {"rec": "P3", "amount": 1},
        ],
    }
    baseline = {
        "source": "peers",
        "aggregate": "median",
        "per": "rec",
        "per_aggregate": "count",
        "field": "",
    }
    c = evaluate_verdict(
        _baseline_spec(bound=2, baseline=baseline), logs, _Analysis()
    ).subjects[0].checks[0]
    # Per-member counts are [5, 1, 1]; their median is 1, so the threshold is 2 and the
    # subject's 4 rows exceed it. Read over the rows instead, the baseline would be 7.
    assert c.result == "fail", c.detail
    assert "baseline = 1" in c.observed, c.observed
    assert "count per rec over 3 member(s)" in c.observed, c.observed


# ------------------------------------------------------------------------ event_order


def _order_cond(**over):
    cond = {
        "id": "order",
        "label": "The closing event followed the opening one",
        "kind": "event_order",
        "relation": "after",
        "quantifier": "every",
        "start": {"source": "events", "field": "opened_at"},
        "end": {"source": "events", "field": "closed_at"},
    }
    cond.update(over)
    return cond


def _order_rows(*pairs):
    return {
        "events": [
            {"opened_at": opened, "closed_at": closed} for opened, closed in pairs
        ]
    }


def test_event_order_verifies_an_ordering_that_time_gap_only_bounds():
    """The defect this kind exists for. `time_gap` compares MAGNITUDES and forgives five
    minutes of negative slack, so a closing event three minutes BEFORE its opening passes a
    check labelled "followed" — the label states an ordering the evaluator never tested."""
    from src.correlation import _eval_condition

    rows = _order_rows(("2026-01-01T10:00:00Z", "2026-01-01T09:57:00Z"))
    gap = _eval_condition(
        {
            "id": "gap",
            "kind": "time_gap",
            "start": {"source": "events", "field": "opened_at"},
            "end": {"source": "events", "field": "closed_at"},
            "max": "1h",
        },
        rows,
    )
    assert gap.result == "pass", gap.observed

    c = _eval_condition(_order_cond(), rows)
    assert c.result == "fail", (c.result, c.observed)
    assert "earlier by 0:03:00" in c.detail, c.detail


def test_event_order_reads_each_relation_including_simultaneity():
    """Four relations rather than two because equality is a real reading: a day-granular
    column produces it, and whether it satisfies "followed" is the pack's call, not the
    engine's."""
    from src.correlation import _eval_condition

    same = _order_rows(("2026-01-01T10:00:00Z", "2026-01-01T10:00:00Z"))
    later = _order_rows(("2026-01-01T10:00:00Z", "2026-01-01T11:00:00Z"))
    for relation, on_same, on_later in (
        ("after", "fail", "pass"),
        ("not_before", "pass", "pass"),
        ("before", "fail", "fail"),
        ("not_after", "pass", "fail"),
    ):
        assert _eval_condition(_order_cond(relation=relation), same).result == on_same, relation
        assert (
            _eval_condition(_order_cond(relation=relation), later).result == on_later
        ), relation
    c = _eval_condition(_order_cond(relation="not_before"), same)
    assert "simultaneous" in c.detail, c.detail


def test_event_order_quantifier_picks_the_deciding_pair():
    """A side may carry many timestamps, and the two quantifiers answer different questions
    over one row set: the universal is settled by the hardest pair and the existential by the
    easiest. Without a declared quantifier the engine would have to pick one and call it a
    default, which is the pack's judgement taken in Python."""
    from src.correlation import _eval_condition

    # One closing event follows both openings; the other precedes the later opening.
    rows = _order_rows(
        ("2026-01-01T10:00:00Z", "2026-01-01T12:00:00Z"),
        ("2026-01-01T11:00:00Z", "2026-01-01T10:30:00Z"),
    )
    every = _eval_condition(_order_cond(quantifier="every"), rows)
    assert every.result == "fail", every.observed
    assert "2 start / 2 end timestamp(s)" in every.observed, every.observed
    assert _eval_condition(_order_cond(quantifier="any"), rows).result == "pass"


def test_event_order_tolerance_is_declared_and_never_defaulted():
    """An absent tolerance is EXACT, not a small one: the five minutes `time_gap` forgives are
    a deployment's judgement about its own clocks, taken from a literal in the engine where no
    pack can restate it. An unreadable one refuses to compare rather than silently asking the
    stricter question."""
    from src.correlation import _eval_condition

    rows = _order_rows(("2026-01-01T10:00:00Z", "2026-01-01T09:59:00Z"))
    assert _eval_condition(_order_cond(), rows).result == "fail"
    forgiving = _eval_condition(_order_cond(tolerance="90m"), rows)
    assert forgiving.result == "pass", forgiving.detail
    assert "(±90m)" in forgiving.expected, forgiving.expected

    unreadable = _eval_condition(_order_cond(tolerance="1 week"), rows)
    assert unreadable.result == "unknown"
    assert "stricter question" in unreadable.detail, unreadable.detail


def test_event_order_refuses_an_undeclared_relation_or_quantifier():
    """Neither has a default, so neither absence is a loose check — it is one that never
    compares. Reported by naming what is missing, because a decisive `unknown` reaches the
    reader as INSUFFICIENT DATA with nothing saying the declaration emptied it."""
    from src.correlation import _eval_condition

    rows = _order_rows(("2026-01-01T10:00:00Z", "2026-01-01T11:00:00Z"))
    for over in ({"relation": ""}, {"relation": "follows"}, {"quantifier": ""}):
        c = _eval_condition(_order_cond(**over), rows)
        assert c.result == "unknown", (over, c.result)
        assert "the relation IS the check" in c.detail, c.detail

    missing = _eval_condition(_order_cond(), {"events": [{"opened_at": "2026-01-01"}]})
    assert missing.result == "unknown"
    assert "no parseable timestamp" in missing.detail, missing.detail


def test_event_order_holds_only_where_truncation_cannot_flip_it():
    """The quantifier's own asymmetry, and it runs opposite ways: `every` is universal, so a
    row that never came back can only BREAK a pass; `any` is existential, so it can only
    rescue a fail. Each surviving outcome is the one the missing rows cannot reach."""
    from src.correlation import _eval_condition

    rows = _order_rows(
        ("2026-01-01T10:00:00Z", "2026-01-01T12:00:00Z"),
        ("2026-01-01T11:00:00Z", "2026-01-01T13:00:00Z"),
    )
    capped = {"_count_truncated": ["events"]}
    every_pass = _eval_condition(_order_cond(quantifier="every", **capped), rows)
    assert every_pass.result == "unknown", every_pass.observed
    assert "TRUNCATED" in every_pass.observed
    assert "not the same claim" in every_pass.detail, every_pass.detail
    assert _eval_condition(_order_cond(quantifier="any", **capped), rows).result == "pass"

    inverted = _order_rows(
        ("2026-01-01T10:00:00Z", "2026-01-01T09:00:00Z"),
        ("2026-01-01T11:00:00Z", "2026-01-01T09:30:00Z"),
    )
    assert (
        _eval_condition(_order_cond(quantifier="every", **capped), inverted).result == "fail"
    )
    rescued = _eval_condition(_order_cond(quantifier="any", **capped), inverted)
    assert rescued.result == "unknown", rescued.observed


def test_a_row_cap_reaches_an_event_order_through_the_run():
    """The flag is stamped in `_preprocess`, so a cap on the source has to arrive by itself —
    a hand-set `_count_truncated` proves the reading and not the wiring, and the three kinds
    that hedge on truncation each had to be listed there."""
    from src.correlation import evaluate_verdict

    spec = {
        "key": "order_case",
        "title": "Ordering",
        "subject_entity": "record",
        "subject_field": "rec",
        "sources": {"events": "events_src"},
        "conditions": [_order_cond(start={"source": "events", "field": "opened_at"})],
    }
    logs = {
        "events_src": [
            {
                "rec": "SUBJ03",
                "opened_at": "2026-01-01T10:00:00Z",
                "closed_at": "2026-01-01T12:00:00Z",
            }
        ]
    }
    c = evaluate_verdict(spec, logs, _Analysis(), row_caps={"events_src": 1}).subjects[0].checks[0]
    assert c.result == "unknown", c.detail
    assert "row cap" in c.detail, c.detail
    uncapped = evaluate_verdict(spec, logs, _Analysis(), row_caps={"events_src": 9})
    assert uncapped.subjects[0].checks[0].result == "pass"


# ------------------------------------------------------------- equivalence forms


#: A form naming every shape the four incumbent normalisation seams had opinions about. The
#: engine supplies the OPERATIONS; every threshold in here is a declaration.
_FORMS = {
    "folded": {"project": [{"case": "upper"}, {"keep": "alnum"}]},
    # `id_suffix` (`_values_match`) reproduced as YAML: fold, keep alphanumerics, accept mutual
    # containment, and refuse a key shorter than 6 — the magic number it hardcoded.
    "id_suffix": {
        "project": [{"case": "upper"}, {"keep": "alnum"}],
        "compare": {"contains": True},
        "min_length": 6,
        "linkage": "single",
    },
    "near": {"compare": {"edit_distance": 1}, "linkage": "single"},
    "near_complete": {"compare": {"edit_distance": 1}, "linkage": "complete"},
    "first_two": {"project": [{"case": "upper"}, {"prefix": 2}]},
    "alpha_only": {"project": [{"keep": "alpha"}], "min_length": 3},
    # The fourth seam: `equivalent_values` on `delimited_field_mismatch` is a substitution
    # table reachable from exactly one kind. `map` is the same relation, composable.
    "iso": {"project": [{"case": "upper"}, {"map": {"NGA": "NG", "USA": "US"}}]},
    "anchorless": {"compare": {"exact": True}},
    "no_n": {"compare": {"edit_distance": True}, "linkage": "single"},
    "no_prefix_n": {"compare": {"shared_prefix": True}, "linkage": "single"},
    "unknown_op": {"project": [{"nope": 1}]},
    "two_ops": {"project": [{"case": "upper", "prefix": 2}]},
    "always_empty": {"project": [{"keep": "alpha"}, {"prefix": 2}, {"suffix": 1}]},
}


def _eq_cond(**over):
    cond = {
        "id": "collision",
        "label": "No identity is shared under the declared form",
        "kind": "value_equivalence",
        "source": "events",
        "field": "ref",
        "form": "folded",
        "operator": "<=",
        "bound": 1,
        "_equivalence_forms": _FORMS,
    }
    cond.update(over)
    return cond


def _eq_rows(*refs):
    return {"events": [{"ref": r} for r in refs]}


def test_the_engine_supplies_no_relation_and_no_threshold_of_its_own():
    """Every undeclared parameter reads `unknown`; none falls back to equality.

    This is the test the whole vocabulary rests on. A silent fallback to `exact` is the one
    wrong answer that still looks like a working check — the condition returns a well-formed
    count, the report prints it, and the relation it was counted under is not the one any pack
    declared. Same rule as `_declared_bound`: an undeclared bound is not a small bound.
    """
    from src.correlation import _eval_condition

    rows = _eq_rows("ABCDEF", "ABCDEG", "ABCDEH")
    undeclared = [
        # An operation whose parameter the pack did not state.
        ("no_n", "takes the largest number of edits"),
        ("no_prefix_n", "takes how many leading characters"),
        # A pairwise relation with neither an `anchor` nor a `linkage`: which of the two the
        # pack means changes the classes, so the engine cannot pick one.
        ("anchorless", "needs `linkage`"),
        # An operation the engine does not have. An ignored op is a DIFFERENT relation, so
        # every reading taken under it is wrong.
        ("unknown_op", "is not one of the projection operations"),
        ("two_ops", "exactly one operation"),
        # A name no pack file declares. Falling back to the incumbent key would answer a
        # different question under the authority of a declaration nobody wrote.
        ("never_declared", "no pack file declares the form"),
    ]
    for name, fragment in undeclared:
        c = _eval_condition(_eq_cond(form=name), rows)
        assert c.result == "unknown", (name, c.result, c.observed)
        assert fragment in c.detail, (name, c.detail)

    # And a condition naming no form at all: not "the default relation", no relation.
    for over in ({"form": ""}, {"operator": ""}, {"bound": None}):
        c = _eval_condition(_eq_cond(**over), rows)
        assert c.result == "unknown", (over, c.observed)
        assert "needs a `form` the pack declares" in c.detail, (over, c.detail)

    # The proof that none of the above silently compared on equality: under a form that IS
    # declared, these three values are one class of 3 and the same bound FAILS.
    declared = _eval_condition(_eq_cond(form="near"), rows)
    assert declared.result == "fail", declared.observed
    assert "largest class 3 of 3" in declared.observed, declared.observed


def test_no_form_is_the_incumbent_reading_byte_for_byte():
    """A pack that declares no forms behaves exactly as it did — the checkpoint this whole
    vocabulary rides behind. `_canonical_key(v, None)` IS `_norm_identifier`, and the counting
    kinds that gained a `normalize:` hook produce the same line without one."""
    from src.correlation import _canonical_key, _eval_condition, _norm_identifier

    for raw in ("ab-CD_12", "  x.y  ", "999", "ABCDEF"):
        assert _canonical_key(raw, None) == (_norm_identifier(raw) or None), raw

    cond = {
        "id": "dc",
        "label": "One value",
        "kind": "distinct_count",
        "source": "events",
        "field": "ref",
        "max": 1,
    }
    rows = _eq_rows("999", "abc", "---")
    plain = _eval_condition(cond, rows)
    # `---` keeps no alphanumerics, and with no form declared that stays the incumbent silent
    # drop: reporting it as a form's failure would change every existing condition line.
    assert plain.observed == "2 distinct: ['999', 'abc'] [from events.ref]", plain.observed
    assert "form" not in plain.observed and "unresolvable" not in plain.observed

    # An absent `_equivalence_forms` stamp reads identically to an empty one, because a pack
    # declaring none must not receive a condition dict it never had.
    assert _eval_condition({**cond, "_equivalence_forms": {}}, rows).observed == plain.observed


def test_a_hardcoded_engine_mode_is_reproducible_as_a_declaration():
    """`_values_match`'s `id_suffix` was mutual containment plus a minimum length of 6 — a
    pack's decision taken in Python, where no pack could restate it. The same relation is one
    form, and the 6 is the pack's number."""
    from src.correlation import _canonical_key, _eval_condition

    form = _FORMS["id_suffix"]
    assert _canonical_key("ab-cd_123456", form) == "ABCD123456"
    # Too short to satisfy the declared floor: unresolvable, not silently passed through.
    assert _canonical_key("abc", form) is None

    # Containment: the short key sits inside the long one, so both are one class.
    c = _eval_condition(
        _eq_cond(form="id_suffix", operator="<=", bound=1),
        _eq_rows("XX-123456", "123456", "ZZZZZZ"),
    )
    assert c.result == "fail", c.observed
    assert "largest class 2 of 3" in c.observed, c.observed

    # And the floor is the pack's: raise it and the same values stop being readable at all.
    strict = dict(form, min_length=12)
    c2 = _eval_condition(
        _eq_cond(form="strict", _equivalence_forms={**_FORMS, "strict": strict}),
        _eq_rows("XX-123456", "123456"),
    )
    assert c2.result == "pass", c2.observed
    assert "2 value(s) unresolvable under this form" in c2.observed, c2.observed


def test_a_value_the_form_cannot_read_is_excluded_counted_and_named():
    """Unresolvable is not `""`. A form that reduced a value to nothing would make it
    equivalent to every OTHER value it could not read, fabricating a class out of the form's
    own failures — and a class of 2 drawn from 40 values of which 38 were unreadable is not the
    finding it reads as. Same rule as an absent `encoded_fields` part being omitted, not `""`.
    """
    from src.correlation import _canonical_key, _eval_condition

    # `keep: alpha` against a numeric value yields nothing, which is the collapse `min_length`
    # exists to make visible.
    assert _canonical_key("12345", _FORMS["alpha_only"]) is None
    assert _canonical_key("xy", _FORMS["alpha_only"]) is None  # below the declared floor

    # And a form declaring NO floor is the case that has to hold on its own: `min_length` is a
    # warning and not a requirement, so a shortening form without one is a pack a validator
    # lets through, and there the empty projection is the ONLY thing standing between the
    # form's failures and one fabricated class of them.
    assert _canonical_key("---", _FORMS["folded"]) is None
    assert _canonical_key("ab-CD", _FORMS["folded"]) == "ABCD"
    floorless = _eval_condition(
        _eq_cond(form="folded", operator="<=", bound=1),
        _eq_rows("ab-CD", "---", "***"),
    )
    assert floorless.result == "pass", floorless.observed
    assert "largest class 1 of 1 value(s)" in floorless.observed, floorless.observed
    assert "2 value(s) unresolvable under this form" in floorless.observed, floorless.observed

    c = _eval_condition(
        _eq_cond(form="alpha_only", operator="<=", bound=1),
        _eq_rows("abc", "abc", "111", "222", "333"),
    )
    # Two spellings of one readable value; the three numerics are reported, never one class.
    assert c.result == "pass", c.observed
    assert "largest class 1 of 1 value(s)" in c.observed, c.observed
    assert "3 value(s) unresolvable under this form" in c.observed, c.observed
    assert "['111', '222', '333']" in c.observed, c.observed


def test_value_equivalence_names_the_class_it_counted():
    """The class key and its members ARE the finding, per the modal rule one step over: a line
    reading `largest class 3 (<= 1)` states a collision and names nothing."""
    from src.correlation import _eval_condition

    c = _eval_condition(
        _eq_cond(form="first_two", operator="<=", bound=2),
        _eq_rows("ABxx", "AByy", "ABzz", "QQ11"),
    )
    assert c.result == "fail", c.observed
    assert "largest class 3 of 4 value(s): AB ['ABxx', 'AByy', 'ABzz']" in c.observed
    assert "2 class(es)" in c.observed, c.observed
    assert "[under the first_two form]" in c.observed, c.observed


def test_an_anchor_asks_the_targeting_question_and_refuses_without_one():
    """One kind, two framings: with an anchor, how many values are equivalent to it; without
    one, how many collide. An anchor the form cannot read is refused rather than counted as
    zero matches — there is nothing for the other values to be equivalent TO."""
    from src.correlation import _eval_condition

    rows = {
        "events": [{"ref": r} for r in ("ABCDEF", "ABCDEG", "ZZZZZZ")],
        "subject": [{"own": "ABCDEF"}],
    }
    anchored = _eval_condition(
        _eq_cond(
            form="near",
            operator="<=",
            bound=1,
            anchor={"source": "subject", "field": "own"},
        ),
        rows,
    )
    assert anchored.result == "fail", anchored.observed
    assert "2 of 3 value(s) equivalent to ABCDEF" in anchored.observed, anchored.observed

    # An anchor the form reduces to nothing: `unknown`, not "0 are equivalent to it".
    blank = _eval_condition(
        _eq_cond(
            form="alpha_only",
            operator="<=",
            bound=1,
            anchor={"source": "subject", "field": "own"},
        ),
        {**rows, "subject": [{"own": "12"}]},
    )
    assert blank.result == "unknown", blank.observed
    assert "nothing for the other values to be equivalent TO" in blank.detail


def test_single_and_complete_linkage_are_two_verdicts_over_identical_rows():
    """Which one applies is the pack's declaration, and this is why the engine cannot pick.

    Under `edit_distance: 1` these four values form a CHAIN: each is one edit from the next and
    two edits from the one after. Single linkage takes connected components and returns one
    class of 4; complete linkage requires every member to match every other and returns two.
    Same rows, same operation, different count — so a `<= 3` bound FAILS under one and PASSES
    under the other, which is a verdict the engine would be authoring.
    """
    from src.correlation import _eval_condition, _equivalence_classes

    chain = ["AAAA", "AAAB", "AABB", "ABBB"]
    single, _, single_cmps = _equivalence_classes(chain, _FORMS["near"])
    complete, _, complete_cmps = _equivalence_classes(chain, _FORMS["near_complete"])
    assert [len(m) for _, m in single] == [4], single
    assert [len(m) for _, m in complete] == [2, 2], complete
    # The comparison counts differ per linkage, so neither is the other's estimate.
    assert single_cmps != complete_cmps, (single_cmps, complete_cmps)

    rows = _eq_rows(*chain)
    assert _eval_condition(_eq_cond(form="near", bound=3), rows).result == "fail"
    assert _eval_condition(_eq_cond(form="near_complete", bound=3), rows).result == "pass"


def test_the_pairwise_cost_is_reported_rather_than_capped():
    """A projection is O(n) and a pairwise pass is O(n²), so the engine states the work it did.
    A cap that silently truncated a class is the fabricated-finding shape one layer down — the
    same rule as every other cut in this system stating its remainder."""
    from src.correlation import _eval_condition

    rows = _eq_rows("AAAA", "AAAB", "AABB", "ABBB")
    pairwise = _eval_condition(_eq_cond(form="near", bound=9), rows)
    assert "6 pairwise comparison(s)" in pairwise.observed, pairwise.observed

    # A projection-only form does no pairwise work and therefore claims none.
    projected = _eval_condition(_eq_cond(form="folded", bound=9), rows)
    assert "pairwise comparison" not in projected.observed, projected.observed


def test_a_truncated_equivalence_read_stands_only_upward():
    """A class can only GROW as the missing rows arrive, so a `>` conclusion carries through a
    truncated read and a `<=` PASS does not. The values that never came back could join it."""
    from src.correlation import _eval_condition

    rows = _eq_rows("ABxx", "AByy", "ABzz")
    cut = {"_count_truncated": ["events"]}

    stands = _eval_condition(_eq_cond(form="first_two", operator=">=", bound=2, **cut), rows)
    assert stands.result == "pass", stands.observed

    falls = _eval_condition(_eq_cond(form="first_two", operator="<=", bound=9, **cut), rows)
    assert falls.result == "unknown", falls.observed
    assert "TRUNCATED" in falls.observed, falls.observed
    assert "could join the class" in falls.detail, falls.detail

    # An untruncated read of the same rows decides it.
    assert _eval_condition(_eq_cond(form="first_two", operator="<=", bound=9), rows).result == "pass"


def test_a_form_reaches_both_sides_of_a_distinct_counts_subject_exclusion():
    """`normalize:` has to change the dedup key AND the subject's own key together.

    Change the dedup key alone and `exclude_subject` silently stops matching, so the subject's
    own value counts as one more distinct value — the count inflates by one in the convicting
    direction, off a condition that looks like it is working.
    """
    from src.correlation import _eval_condition

    cond = {
        "id": "dc",
        "label": "No other identity on this record",
        "kind": "distinct_count",
        "source": "events",
        "field": "ref",
        "max": 0,
        "exclude_subject": True,
        "_subject_identity_values": ["ABxx"],
        "_equivalence_forms": _FORMS,
    }
    rows = _eq_rows("ABzz")
    # Under the form both spellings reduce to `AB`, so the only value IS the subject's.
    under = _eval_condition({**cond, "normalize": "first_two"}, rows)
    assert under.result == "pass", under.observed
    assert "[under the first_two form]" in under.observed, under.observed
    # Without it they are two values and the same rows read as another identity.
    assert _eval_condition(cond, rows).result == "fail"


def test_a_distinct_count_refuses_a_relation_that_has_no_distinct_count():
    """A projection is transitive so "how many distinct" has one answer; a `compare` relation
    is not, and the same question becomes "how many classes under which linkage". Applying only
    the projection half would answer a different question under the form's name."""
    from src.correlation import _eval_condition

    cond = {
        "id": "dc",
        "label": "One value",
        "kind": "distinct_count",
        "source": "events",
        "field": "ref",
        "max": 1,
        "normalize": "id_suffix",
        "_equivalence_forms": _FORMS,
    }
    c = _eval_condition(cond, _eq_rows("XX-123456", "123456"))
    assert c.result == "unknown", c.observed
    assert "not transitive" in c.detail, c.detail
    assert "`value_equivalence`" in c.detail, c.detail

    # The same refusal at the other seam that deduplicates on a key.
    num = _eval_condition(
        {
            "id": "nc",
            "label": "Few values",
            "kind": "numeric_compare",
            "source": "events",
            "field": "ref",
            "aggregate": "distinct",
            "operator": "<=",
            "bound": 1,
            "normalize": "id_suffix",
            "_equivalence_forms": _FORMS,
        },
        _eq_rows("XX-123456", "123456"),
    )
    assert num.result == "unknown" and "not transitive" in num.detail, num.detail


def test_a_form_leaves_a_row_count_untouched_and_claims_nothing():
    """`count` reads text but counts ROWS, so a form cannot change its answer — and a note
    claiming otherwise attributes the number to a projection that never ran."""
    from src.correlation import _eval_condition, _FORM_AGGREGATES

    assert _FORM_AGGREGATES == ("distinct", "mode", "mode_share"), _FORM_AGGREGATES

    base = {
        "id": "nc",
        "label": "Volume",
        "kind": "numeric_compare",
        "source": "events",
        "field": "ref",
        "operator": "<=",
        "bound": 9,
        "normalize": "first_two",
        "_equivalence_forms": _FORMS,
    }
    rows = _eq_rows("ABxx", "AByy", "ABzz")

    counted = _eval_condition({**base, "aggregate": "count"}, rows)
    assert counted.result == "pass" and "count = 3" in counted.observed
    assert "form" not in counted.observed, counted.observed

    # The aggregates a form DOES change say so, because which form produced the number is
    # part of the number.
    for agg, observed in (("distinct", "distinct = 1"), ("mode", "mode = 3")):
        c = _eval_condition({**base, "aggregate": agg}, rows)
        assert observed in c.observed, (agg, c.observed)
        assert "[under the first_two form]" in c.observed, (agg, c.observed)


def test_a_substitution_table_is_one_of_the_operations():
    """`equivalent_values` on `delimited_field_mismatch` is the fourth normalisation seam — the
    same relation, reachable from exactly one kind and from no counting kind, so no aggregate
    could ever be taken over it. `map` is that relation as a composable operation."""
    from src.correlation import _canonical_key, _eval_condition

    assert _canonical_key("nga", _FORMS["iso"]) == "NG"
    assert _canonical_key("NG", _FORMS["iso"]) == "NG"
    assert _canonical_key("GBR", _FORMS["iso"]) == "GBR"  # unlisted values compare literally

    c = _eval_condition(
        _eq_cond(form="iso", operator="<=", bound=1),
        _eq_rows("NGA", "NG", "GBR"),
    )
    assert c.result == "fail", c.observed
    assert "largest class 2 of 3 value(s): NG ['NG', 'NGA']" in c.observed, c.observed


def test_a_pack_declared_form_reaches_every_condition_through_the_ruleset_spec():
    """The forms are run-level context, so they inherit down a composite the same way `_routes`
    and `_subject_identity_values` do — a child evaluated without them would read `unknown` on
    a form its own parent's pack declares."""
    from src.correlation import evaluate_verdict

    child = {
        "id": "inner",
        "label": "No identity collides under the declared form",
        "kind": "value_equivalence",
        "source": "records",
        "field": "actor_code",
        "form": "first_two",
        "operator": "<=",
        "bound": 0,
    }
    spec = _comp_spec()
    spec["_equivalence_forms"] = _FORMS
    spec["conditions"] = [
        {
            "id": "outer",
            "label": "Nothing collides",
            "kind": "all_of",
            "fail_detail": "a shared identity was found",
            "children": [child, dict(child, id="inner2")],
        }
    ]
    c = evaluate_verdict(spec, _comp_logs(), _CompAnalysis()).subjects[0].checks[0]
    assert c.result == "fail", (c.result, c.observed, c.detail)

    # Without the stamp the same ruleset cannot read its own declaration, which is what makes
    # this a wiring test and not a second reading of the evaluator: an undeclared form is not
    # the incumbent key.
    bare = {k: v for k, v in spec.items() if k != "_equivalence_forms"}
    b = evaluate_verdict(bare, _comp_logs(), _CompAnalysis()).subjects[0].checks[0]
    assert b.result == "unknown", (b.result, b.observed)
