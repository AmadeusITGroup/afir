"""Tests for the shared entity->field mapping helper."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.knowledge.pack import EntityDef, KnowledgePack, SourceDef
from src.models.pydantic_models import (EntityFieldMap, ExtractedEntity,
                                        FieldMapping, RetrievalQuery)
from src.retrievers.field_mapping import (map_entities, render_filters,
                                          render_identifiers)


def _query():
    return RetrievalQuery(
        target_log_source="transaction_logs",
        natural_language_query="issuance for org_unit",
        date_from="2024-08-25",
        date_to="2024-08-27",
        entities=[
            ExtractedEntity(type="org_unit", value="ORGUNIT2301"),
            ExtractedEntity(type="document", value="300-2000000001"),
        ],
    )


@pytest.mark.asyncio
async def test_maps_entities_to_real_fields_and_filters_confidence():
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=FieldMapping(
            mappings=[
                EntityFieldMap(
                    entity_type="org_unit", field="org_unit_id", confidence=0.95
                ),
                EntityFieldMap(
                    entity_type="document", field="document_nbr", confidence=0.3
                ),
            ]
        )
    )
    schema = "transactions(org_unit_id string, document_nbr string)"

    field_map = await map_entities(llm, _query(), schema)

    # Low-confidence document mapping is dropped; org_unit is kept.
    assert field_map == {"org_unit": "org_unit_id"}


@pytest.mark.asyncio
async def test_no_entities_or_no_schema_short_circuits():
    llm = MagicMock()
    llm.structured_output = AsyncMock()

    empty_q = RetrievalQuery(
        target_log_source="t",
        natural_language_query="x",
        date_from="2024-01-01",
        date_to="2024-01-02",
    )
    assert await map_entities(llm, empty_q, "t(col int)") == {}
    assert await map_entities(llm, _query(), "") == {}
    llm.structured_output.assert_not_called()


@pytest.mark.asyncio
async def test_alias_hints_passed_from_pack():
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=FieldMapping(
            mappings=[EntityFieldMap(entity_type="org_unit", field="org_unit_id")]
        )
    )
    pack = KnowledgePack(
        entities=[
            EntityDef(type="org_unit", field_aliases=["orgUnitId", "org_unit_id"])
        ]
    )

    await map_entities(llm, _query(), "transactions(org_unit_id string)", pack)

    # The alias hint string reaches the LLM prompt.
    sent = llm.structured_output.call_args.args[0]
    user_msg = sent[-1]["content"]
    assert "org_unit_id" in user_msg


@pytest.mark.asyncio
async def test_source_specific_bindings_preferred_in_hint():
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=FieldMapping(
            mappings=[EntityFieldMap(entity_type="org_unit", field="pos_org_unit")]
        )
    )
    # Global alias says orgUnitId; the source binding says pos_org_unit. The source
    # binding must appear first in the hint sent to the LLM.
    pack = KnowledgePack(
        entities=[EntityDef(type="org_unit", field_aliases=["orgUnitId"])],
        sources=[
            SourceDef(
                name="transaction_logs",
                entity_bindings={"org_unit": ["pos_org_unit"]},
            )
        ],
    )

    await map_entities(llm, _query(), "transactions(pos_org_unit string)", pack)

    user_msg = llm.structured_output.call_args.args[0][-1]["content"]
    assert "org_unit: pos_org_unit" in user_msg


def test_render_filters_pairs_field_with_value():
    field_map = {"org_unit": "org_unit_id", "document": "document_nbr"}
    hints = render_filters(field_map, _query())
    assert "org_unit_id = 'ORGUNIT2301'" in hints
    assert "document_nbr = '300-2000000001'" in hints


def test_render_filters_skips_wildcards_and_empty():
    q = RetrievalQuery(
        target_log_source="t",
        natural_language_query="x",
        date_from="2024-01-01",
        date_to="2024-01-02",
        entities=[ExtractedEntity(type="org_unit", value="*")],
    )
    assert render_filters({"org_unit": "org_unit_id"}, q) == ""
    assert render_filters({}, q) == ""


def test_render_filters_ors_multiple_values_of_same_type():
    """A login and an agent sign both typed `user` must NOT collapse to the last
    value — they are alternatives, rendered as an IN (...) 'match ANY' hint."""
    q = RetrievalQuery(
        target_log_source="t",
        natural_language_query="x",
        date_from="2024-01-01",
        date_to="2024-01-02",
        entities=[
            ExtractedEntity(type="user", value="USERNAMEX"),
            ExtractedEntity(type="user", value="0201GP"),
        ],
    )
    hints = render_filters({"user": "user_id"}, q)
    # Both values survive (the old dict-collapse dropped USERNAMEX).
    assert "USERNAMEX" in hints and "0201GP" in hints
    assert "IN (" in hints and "ANY" in hints


def test_render_filters_unions_two_types_onto_one_field():
    q = RetrievalQuery(
        target_log_source="t",
        natural_language_query="x",
        date_from="2024-01-01",
        date_to="2024-01-02",
        entities=[
            ExtractedEntity(type="user", value="USERNAMEX"),
            ExtractedEntity(type="sign", value="0201GP"),
        ],
    )
    # Both entity types map to the same field -> unioned as alternatives.
    hints = render_filters({"user": "uid", "sign": "uid"}, q)
    assert "USERNAMEX" in hints and "0201GP" in hints
    assert hints.count("uid") == 1  # one field clause, not two


def test_render_filters_requires_conjunction_for_composite_key_source():
    """`require_all_entities` must state the AND requirement, naming the mapped fields.

    A lookup keyed by (orgUnitId, sign) has no selective column of its own: OR-ing them asks
    for most of the table (sign '6009JJ' is on 89,937 rows across as many org_units), the row
    cap truncates it, and the rows returned belong to other keys. IR10000002 lost a clean
    AUTOMATED exclusion exactly that way."""
    q = RetrievalQuery(
        target_log_source="automation_registry",
        natural_language_query="x",
        date_from="2026-07-17",
        date_to="2026-07-18",
        entities=[
            ExtractedEntity(type="org_unit", value="QQQ1R17GH"),
            ExtractedEntity(type="user", value="6009JJ"),
        ],
    )
    field_map = {"org_unit": "orgUnitId", "user": "sign"}
    plain = render_filters(field_map, q)
    assert "MANDATORY" not in plain  # unchanged when not declared

    hints = render_filters(field_map, q, require_all_entities=["org_unit", "user"])
    assert "MANDATORY" in hints and "AND" in hints
    assert "orgUnitId" in hints and "sign" in hints
    # Only one identity available -> nothing to conjoin, so no spurious requirement.
    q_one = RetrievalQuery(
        target_log_source="automation_registry",
        natural_language_query="x",
        date_from="2026-07-17",
        date_to="2026-07-18",
        entities=[ExtractedEntity(type="org_unit", value="QQQ1R17GH")],
    )
    assert "MANDATORY" not in render_filters(
        field_map, q_one, require_all_entities=["org_unit", "user"]
    )


def test_identifier_values_reach_the_prompt_without_asserting_their_kind():
    """`RetrievalQuery.actor_id` is the planner's GUESS, and nothing verifies it.

    On IR10000004 the alert carried a login (`ASURNAME`) and an agent sign
    (`0606GK`); the planner put the sign in `actor_id` on all nine queries. Those two
    identifiers bind to DIFFERENT columns, so a prompt line labelling one of them
    `actor_id:` argues for the wrong column on any source that has both fields. The
    values must still reach the generator — they are the scan bound on a source with
    no mapped entity — but as untyped candidates."""
    q = RetrievalQuery(
        target_log_source="auth_events",
        natural_language_query="x",
        scope_id="AAA1B0955",
        actor_id="0606GK",
        date_from="2026-07-29",
        date_to="2026-07-29",
    )
    line = render_identifiers(q)
    assert "AAA1B0955" in line and "0606GK" in line
    # The kind is NOT asserted: no `actor_id:`/`scope_id:` labelling.
    assert "actor_id:" not in line and "scope_id:" not in line
    assert "UNTYPED" in line
    # Wildcards/absent scalars contribute nothing at all.
    assert (
        render_identifiers(
            RetrievalQuery(
                target_log_source="t",
                natural_language_query="x",
                date_from="2026-07-29",
                date_to="2026-07-29",
            )
        )
        == ""
    )


def test_identifiers_fall_back_to_the_query_entities_when_the_scalars_are_unset():
    """The two scalars are optional, so they cannot be the only fallback input.

    `query.entities` is attached deterministically by `_enrich_queries` and does not depend
    on schema discovery or the request text. When every other channel is empty (scalars at
    their `'*'` default, empty `field_map`), the entities channel must still supply the
    filter values.
    """
    q = RetrievalQuery(
        target_log_source="office_profile",
        natural_language_query="Retrieve this source's records for the incident.",
        date_from="2026-08-01",
        date_to="2026-08-01",
        entities=[
            ExtractedEntity(type="office", value="NNN1P15CD"),
            ExtractedEntity(type="time_window", value="2026-08-01T13:53:00.000Z"),
        ],
    )
    line = render_identifiers(q)
    assert "NNN1P15CD" in line
    # `time_window` rides as a date bound, not a filter value: offering a timestamp as an
    # identifier candidate invites a predicate on it.
    assert "13:53" not in line
    # Still untyped, and still no `office:` labelling — the entity's type is not asserted.
    assert "UNTYPED" in line and "office:" not in line


def test_identifier_values_are_de_duplicated_across_the_scalars_and_the_entities():
    """The same value routinely arrives on both channels; it must be named once."""
    q = RetrievalQuery(
        target_log_source="s",
        natural_language_query="x",
        actor_id="6008HH",
        date_from="2026-08-01",
        date_to="2026-08-01",
        entities=[
            ExtractedEntity(type="user", value="6008HH", value_form="sign"),
            ExtractedEntity(type="office", value="NNN1P15CD"),
        ],
    )
    line = render_identifiers(q)
    assert line.count("6008HH") == 1
    assert "NNN1P15CD" in line


@pytest.mark.asyncio
async def test_map_entities_respects_catalog_allow_list():
    """An entity type the source catalog does NOT list must not be mapped/filtered,
    even if a similarly-named field exists in the discovered schema."""
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=FieldMapping(
            mappings=[
                EntityFieldMap(
                    entity_type="org_unit", field="orgUnitId", confidence=0.95
                )
            ]
        )
    )
    pack = KnowledgePack(
        entities=[EntityDef(type="org_unit"), EntityDef(type="alert_id")],
        sources=[
            # Catalog says only `org_unit` is filterable here (no alert_id).
            SourceDef(name="ic_sessions", entities=["org_unit"]),
        ],
    )
    q = RetrievalQuery(
        target_log_source="ic_sessions",
        natural_language_query="x",
        date_from="2024-01-01",
        date_to="2024-01-02",
        entities=[
            ExtractedEntity(type="org_unit", value="ORG2428D4"),
            ExtractedEntity(type="alert_id", value="scheme-alert-xyz"),
        ],
    )
    await map_entities(llm, q, "ic(orgUnitId string, alertId string)", pack)
    # Only the allowed entity reached the mapping prompt.
    entity_lines = llm.structured_output.call_args.args[0][-1]["content"]
    assert "org_unit: ORG2428D4" in entity_lines
    assert "scheme-alert-xyz" not in entity_lines


# ── value forms: a login and an agent sign are one entity type, two columns ──
#
# Two values, both typed `user`, must route to different columns per source. Prose cannot
# restore a distinction the data model discarded; the form is classified from the value's
# shape and the source binds its columns per form.


def _form_pack():
    """A pack whose `user` has two forms, bound to different fields per source."""
    return KnowledgePack(
        entities=[
            EntityDef(
                type="user",
                field_aliases=["userId", "sign"],
                value_forms=[
                    # Narrow first: a sign would also satisfy a loose login pattern.
                    {"name": "sign", "pattern": r"^[0-9]{4}[A-Z]{2}([A-Z]{2})?$"},
                    {"name": "login", "pattern": r"^[A-Z][A-Z0-9]{1,9}$"},
                ],
            ),
            EntityDef(type="org_unit", field_aliases=["orgUnitId"]),
        ],
        sources=[
            SourceDef(
                name="ic_sessions",
                entities=["user", "org_unit"],
                entity_bindings={
                    "user": {
                        "login": ["user.userId"],
                        "sign": ["loginArea.sign"],
                    },
                    "org_unit": ["orgUnitId"],
                },
            ),
            # A record source stores ONLY the sign — a login has no column at all.
            SourceDef(
                name="record_lake",
                entities=["user"],
                entity_bindings={"user": {"sign": ["creator.sign.red"]}},
            ),
        ],
    )


def _two_form_query(source):
    return RetrievalQuery(
        target_log_source=source,
        natural_language_query="x",
        date_from="2026-07-04",
        date_to="2026-07-04",
        entities=[
            ExtractedEntity(type="user", value="BSURNAME", value_form="login"),
            ExtractedEntity(type="user", value="6009JJ", value_form="sign"),
        ],
    )


def test_value_forms_are_classified_from_the_value_not_its_label():
    """An alert reading `User: 0606GK` is naming a SIGN. Only the shape decides."""
    pack = _form_pack()
    assert pack.classify_value_form("user", "6009JJ") == "sign"
    assert pack.classify_value_form("user", "0303GHSU") == "sign"  # with duty code
    assert pack.classify_value_form("user", "0606GK") == "sign"
    assert pack.classify_value_form("user", "BSURNAME") == "login"
    assert pack.classify_value_form("user", "ASURNAME") == "login"
    # An entity type with no declared forms, an unknown type, and an empty value are all
    # simply unclassified — never an error, so nothing changes for a pack without forms.
    assert pack.classify_value_form("org_unit", "HHH1J09ST") is None
    assert pack.classify_value_form("nope", "x") is None
    assert pack.classify_value_form("user", "") is None


def test_a_sign_and_a_login_render_onto_their_OWN_fields():
    """The exact defect: two `user` values must not collapse onto one column.

    `map_entities` returns one field per entity TYPE, so without form routing both values
    land on whichever field it picked — the live `user.userId IN ('BSURNAME', '6009JJ')`,
    whose sign half can never match.
    """
    pack = _form_pack()
    q = _two_form_query("ic_sessions")
    field_map = {"user": "user.userId"}

    # Without the pack: the old behaviour, both values on one field.
    before = render_filters(field_map, q)
    assert "user.userId IN ('BSURNAME', '6009JJ')" in before

    after = render_filters(field_map, q, knowledge_pack=pack)
    assert "user.userId = 'BSURNAME'" in after
    assert "loginArea.sign = '6009JJ'" in after
    # ...and the sign is NOWHERE near the login column.
    assert "user.userId = '6009JJ'" not in after
    assert "IN (" not in after


def test_a_value_whose_form_the_source_cannot_hold_is_dropped_not_guessed():
    """A login on a record source is not a narrower filter — it is a 0-row one.

    Dropping it leaves the query bounded by the sign (and the date window); keeping it
    would AND/OR in a predicate on a column that does not exist for that form.
    """
    pack = _form_pack()
    after = render_filters(
        {"user": "creator.sign.red"},
        _two_form_query("record_lake"),
        knowledge_pack=pack,
    )
    assert after == "creator.sign.red = '6009JJ'"
    assert "BSURNAME" not in after


def test_an_unclassified_value_behaves_exactly_as_before():
    """No form on the value -> no routing. Absence of a form is never a filter change."""
    pack = _form_pack()
    q = RetrievalQuery(
        target_log_source="ic_sessions",
        natural_language_query="x",
        date_from="2026-07-04",
        date_to="2026-07-04",
        # value_form deliberately left empty, as it is for any entity type without forms.
        entities=[ExtractedEntity(type="user", value="BSURNAME")],
    )
    assert "user.userId = 'BSURNAME'" in render_filters(
        {"user": "user.userId"}, q, knowledge_pack=pack
    )


def test_field_priors_are_scoped_to_one_form():
    """The mapper must not be OFFERED the other form's columns as candidates."""
    pack = _form_pack()
    assert pack.field_priors_for("user", "ic_sessions", "sign") == ["loginArea.sign"]
    assert pack.field_priors_for("user", "ic_sessions", "login") == ["user.userId"]
    # A form this source does not bind gets NOTHING — not the other form's fields, and
    # not the global aliases (which span both forms and would reintroduce the wrong one).
    assert pack.field_priors_for("user", "record_lake", "login") == []
    # With no form requested the map is flattened, so form-unaware callers are unchanged.
    flat = pack.field_priors_for("user", "ic_sessions")
    assert "user.userId" in flat and "loginArea.sign" in flat
    # A flatly-bound entity is untouched by any of this.
    assert pack.field_priors_for("org_unit", "ic_sessions", "sign")[0] == "orgUnitId"


@pytest.mark.asyncio
async def test_the_mapping_prompt_labels_each_form_separately():
    """`user: user.userId, loginArea.sign` invites one column for two values."""
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=FieldMapping(
            mappings=[EntityFieldMap(entity_type="user", field="user.userId")]
        )
    )
    await map_entities(
        llm,
        _two_form_query("ic_sessions"),
        "ic(user.userId string, loginArea.sign string)",
        _form_pack(),
    )
    hint = llm.structured_output.call_args.args[0][-1]["content"]
    assert "user (sign): loginArea.sign" in hint
    assert "user (login): user.userId" in hint


# ── an actor is named by a COMBINATION, and several combinations work ─────────


def _raw_access_pack():
    """A pack shaped like raw_access_log: two candidate keys, priority-ordered."""
    from src.knowledge.pack import (EntityDef, KnowledgePack, SourceDef,
                                    ValueForm)

    return KnowledgePack(
        entities=[
            EntityDef(
                type="user",
                value_forms=[
                    ValueForm(name="sign", pattern=r"^[0-9]{4}[A-Z]{2}([A-Z]{2})?$"),
                    ValueForm(name="login", pattern=r"^[A-Z][A-Z0-9]{1,9}$"),
                ],
            ),
        ],
        sources=[
            SourceDef(
                name="raw_access",
                require_all_entities=["org_unit", "user"],
                identity_keys=[["org_unit", "user"], ["organization", "user"]],
                entity_bindings={
                    "org_unit": ["retriever_org_unit"],
                    "organization": ["retriever_organization"],
                    "user": {
                        "sign": ["retriever_sign"],
                        "login": ["retriever_user_id"],
                    },
                },
            )
        ],
    )


def _q(pairs):
    pack = _raw_access_pack()
    return RetrievalQuery(
        target_log_source="raw_access",
        natural_language_query="x",
        date_from="2026-07-03",
        date_to="2026-07-05",
        entities=[
            ExtractedEntity(
                type=t, value=v, value_form=pack.classify_value_form(t, v) or ""
            )
            for t, v in pairs
        ],
    )


def test_the_first_satisfiable_candidate_key_is_the_one_enforced():
    """(org_unit, sign) is present on every RAW_ACCESS row, so it is candidate #1 and wins."""
    pack = _raw_access_pack()
    field_map = {"org_unit": "retriever_org_unit", "user": "retriever_sign"}
    hints = render_filters(
        field_map,
        _q([("org_unit", "HHH1J09ST"), ("user", "6009JJ")]),
        require_all_entities=["org_unit", "user"],
        knowledge_pack=pack,
        identity_keys=pack.source("raw_access").identity_keys,
    )
    assert "COMBINATION of retriever_org_unit, retriever_sign" in hints
    assert "AND, never OR" in hints


def test_a_missing_key_member_falls_through_to_the_next_COMPLETE_key():
    """This is the defect: a fixed tuple degraded to ONE field and disabled the guard.

    `required_fields` resolves entity types through the field map, and
    `enforce_conjunction` needs two fields to rewrite anything. An incident carrying
    (organization, login) but no org_unit therefore produced a single-element list, the
    rewrite became a no-op, and the generated OR survived — answering about other actors
    while the pack's declaration read like a guarantee. MEASURED on one RAW_ACCESS day:
    retriever_user_id is null on 82,776,709 of 513,444,742 rows and 24,683 of 173,978
    distinct logins appear in more than one org_unit, so neither an unqualified login nor a
    half-applied key identifies anybody.
    """
    pack = _raw_access_pack()
    # No org_unit in the incident, so candidate #1 cannot be met; #2 can.
    field_map = {
        "organization": "retriever_organization",
        "user": "retriever_user_id",
    }
    query = _q([("organization", "ORG-ALFAND"), ("user", "BSURNAME")])

    # BEFORE: the fixed tuple resolves to one field and states no requirement.
    before = render_filters(field_map, query, require_all_entities=["org_unit", "user"])
    assert "MANDATORY" not in before

    after = render_filters(
        field_map,
        query,
        require_all_entities=["org_unit", "user"],
        knowledge_pack=pack,
        identity_keys=pack.source("raw_access").identity_keys,
    )
    assert "COMBINATION of retriever_organization, retriever_user_id" in after


def test_a_partially_satisfiable_candidate_is_skipped_not_half_applied():
    """Half a composite key is not a narrower filter, it is a different question."""
    from src.retrievers.query_guards import resolve_identity_key

    field_map = {"org_unit": "retriever_org_unit", "user": "retriever_sign"}
    # Only `user` is carried: candidate #1 needs org_unit, #2 needs organization.
    assert resolve_identity_key(field_map, [["org_unit", "user"]], ["user"]) == []


# ── a complete one-column key is not a partial two-column one ────────────────
#
# A half-satisfied pair is refused; a one-member key that fully resolves must not be. A
# source whose key is one column had its declaration silently discarded: zero rows read as
# "said nothing" instead of "identity not present". The arity was never the rule.


def test_a_one_column_candidate_RESOLVES_while_a_partial_pair_still_does_not():
    """Both halves in one place, because the distinction is the whole fix.

    Same arity of RESULT — one field — and opposite meanings: one is every member of a key
    that names one column, the other is half of a key that names two.
    """
    from src.retrievers.query_guards import resolve_identity_key

    field_map = {"org_unit": "retriever_org_unit", "user": "retriever_sign"}
    assert resolve_identity_key(field_map, [["org_unit", "user"]], ["user"]) == []
    assert resolve_identity_key(field_map, [["org_unit"]], ["org_unit"]) == [
        "retriever_org_unit"
    ]
    # Unchanged when the incident does not carry the one member either.
    assert resolve_identity_key(field_map, [["org_unit"]], ["user"]) == []
    # ...or when this source binds no column for it.
    assert (
        resolve_identity_key({"user": "retriever_sign"}, [["org_unit"]], ["org_unit"])
        == []
    )


def test_a_one_column_candidate_NEVER_preempts_a_satisfiable_conjunction():
    """Otherwise a pack weakens its own source by adding a fallback candidate.

    Declaration order is a measured priority among keys of EQUAL strength; a one-column key
    is a weaker claim than any pair, so it is resolved in a second tier regardless of where
    it sits in the list. Written first here on purpose — that is the shape a hand-edited
    catalog produces.
    """
    from src.retrievers.query_guards import resolve_identity_key

    field_map = {"org_unit": "retriever_org_unit", "user": "retriever_sign"}
    assert resolve_identity_key(
        field_map, [["user"], ["org_unit", "user"]], ["org_unit", "user"]
    ) == ["retriever_org_unit", "retriever_sign"]
    # And it IS taken once the pair cannot be satisfied.
    assert resolve_identity_key(
        field_map, [["user"], ["org_unit", "user"]], ["user"]
    ) == ["retriever_sign"]


def test_a_conjunction_from_EITHER_declaration_outranks_a_one_column_key():
    """`require_all_entities` is unconditional, so its pair must still win over one column."""
    from src.retrievers.query_guards import conjunction_fields

    field_map = {"org_unit": "retriever_org_unit", "user": "retriever_sign"}
    assert conjunction_fields(
        field_map,
        require_all_entities=["org_unit", "user"],
        identity_keys=[["user"]],
        present_types=["org_unit", "user"],
        source_name="raw_access",
    ) == ["retriever_org_unit", "retriever_sign"]


def test_a_satisfied_candidate_does_not_DISCARD_the_unconditional_declaration(caplog):
    """`identity_keys` picks one alternative key; `require_all_entities` always AND-s all members.

    Ranking them against each other loses conjuncts: after the winning candidate is resolved,
    `require_all_entities` members not in that candidate are still required. The union cannot
    invent a conjunct; only types this source binds contribute.
    """
    import logging

    from src.retrievers.query_guards import conjunction_fields

    field_map = {
        "record": "record_locator",
        "office": "retriever_office",
        "user": "retriever_sign",
        "organization": "retriever_organization",
    }
    with caplog.at_level(logging.INFO, logger="src.retrievers.query_guards"):
        fields = conjunction_fields(
            field_map,
            require_all_entities=["record", "office", "user", "organization"],
            identity_keys=[["office", "user"], ["organization", "user"]],
            present_types=["record", "office", "user", "organization"],
            source_name="raw_access",
        )
    # The actor key leads, in its own declared order, and the rest of the fixed tuple follows.
    assert fields == [
        "retriever_office",
        "retriever_sign",
        "record_locator",
        "retriever_organization",
    ]
    # And it is SAID — a member silently added is as hard to review as one silently dropped.
    assert "require_all_entities" in caplog.text
    assert "record_locator" in caplog.text

    # A TYPE THE INCIDENT DOES NOT CARRY CONTRIBUTES NOTHING, which is what makes it safe for a
    # pack to state the rule broadly. `required_fields` reads the field_map, so the type absent
    # here is absent from the conjunction for the same reason it always was.
    assert conjunction_fields(
        {"office": "retriever_office", "user": "retriever_sign"},
        require_all_entities=["record", "office", "user", "organization"],
        identity_keys=[["office", "user"]],
        present_types=["office", "user"],
        source_name="raw_access",
    ) == ["retriever_office", "retriever_sign"]

    # A MEMBER ALREADY IN THE WINNING CANDIDATE IS NOT REPEATED: `col = A AND col = A` reads to
    # an operator as two facts about the row when it is one.
    assert conjunction_fields(
        field_map,
        require_all_entities=["office", "user"],
        identity_keys=[["office", "user"]],
        present_types=["office", "user"],
        source_name="raw_access",
    ) == ["retriever_office", "retriever_sign"]


def test_a_one_column_candidate_is_ANDED_onto_the_fixed_tuple_not_ranked_against_it():
    """The union's other side, and the case that pins the ORDER.

    A one-column candidate is a weaker claim than a pair, so it must not displace the head of a
    fixed tuple — before the union existed the winning declaration's order was what shipped, and
    a field list reordered here is a key the operator reading the published predicate would have
    to re-derive. But it is still a declared key member, so where it names a column the fixed
    tuple does not, it is AND-ed on rather than thrown away.
    """
    from src.retrievers.query_guards import conjunction_fields

    field_map = {
        "org_unit": "retriever_org_unit",
        "user": "retriever_sign",
        "session": "channel_id",
    }
    # Already inside the fixed tuple: same fields, fixed tuple's order, nothing appended.
    assert conjunction_fields(
        field_map,
        require_all_entities=["org_unit", "user"],
        identity_keys=[["user"]],
        present_types=["org_unit", "user"],
        source_name="raw_access",
    ) == ["retriever_org_unit", "retriever_sign"]
    # Naming a column the fixed tuple does not: appended, never promoted.
    assert conjunction_fields(
        field_map,
        require_all_entities=["org_unit", "user"],
        identity_keys=[["session"]],
        present_types=["org_unit", "user", "session"],
        source_name="raw_access",
    ) == ["retriever_org_unit", "retriever_sign", "channel_id"]


def test_a_one_column_key_ANDs_nothing_and_the_log_does_not_claim_it_did(caplog):
    """What it changes is the CLAIM about an empty result, and nothing else.

    There is no AND to write for one column — `enforce_conjunction` correctly leaves the
    query exactly as generated — so the log must not report this source the way it reports a
    satisfied pair. `key_was_enforced` is the consumer that gains, and it is sound at one
    field by its own argument: a subset of a conjunction returning zero rows proves the
    conjunction is empty too.
    """
    import logging

    from src.retrievers.query_guards import (conjunction_fields,
                                             enforce_conjunction,
                                             key_was_enforced)

    field_map = {"org_unit": "retriever_org_unit", "user": "retriever_sign"}
    with caplog.at_level(logging.INFO, logger="src.retrievers.query_guards"):
        fields = conjunction_fields(
            field_map,
            identity_keys=[["org_unit"]],
            present_types=["org_unit"],
            source_name="office_ref",
        )
    assert fields == ["retriever_org_unit"]
    assert "single column retriever_org_unit" in caplog.text
    assert "Nothing is AND-ed" in caplog.text
    assert "these are AND-ed" not in caplog.text

    sql = "SELECT * FROM t WHERE retriever_org_unit='U1' AND code IN ('a','b')"
    assert enforce_conjunction(sql, fields, "office_ref") == sql
    assert key_was_enforced(sql, fields) is True


def test_a_one_column_key_leaves_the_PROMPT_untouched():
    """The hint states a COMBINATION, and one column is not one — so it must stay silent.

    The rewrite and the hint share `conjunction_fields`; this pins that the seam's new
    one-field answer reaches the claim without inventing prompt prose about an AND that
    cannot exist.
    """
    field_map = {"org_unit": "retriever_org_unit"}
    hints = render_filters(
        field_map,
        _q([("org_unit", "HHH1J09ST")]),
        identity_keys=[["org_unit"]],
    )
    assert "MANDATORY" not in hints
    assert "retriever_org_unit = 'HHH1J09ST'" in hints


def test_the_unresolved_key_warning_names_WHICH_of_the_three_causes(caplog):
    """One sentence for three causes sent every reader to the same wrong place.

    A declaration no incident can satisfy, a binding this source does not have, and an
    incident that genuinely carries half a key are three different problems with three
    different owners — and the log said "not satisfied by this incident's entities" for all
    of them, which is only true of the third.
    """
    import logging

    from src.retrievers.query_guards import conjunction_fields

    def warn(field_map, identity_keys, present):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="src.retrievers.query_guards"):
            assert (
                conjunction_fields(
                    field_map,
                    identity_keys=identity_keys,
                    present_types=present,
                    source_name="raw_access",
                )
                == []
            )
        return caplog.text

    # (1) A candidate spelled as a bare string names no type the resolver can look up.
    text = warn({"user": "retriever_sign"}, ["user"], ["user"])
    assert "catalog defect" in text
    assert "not a property of this incident" in text

    # (2) The incident carries the whole candidate; the source binds no column for a member.
    text = warn(
        {"user": "retriever_sign"}, [["org_unit", "user"]], ["org_unit", "user"]
    )
    assert "binds no field for org_unit" in text
    assert "stale entity binding" in text

    # (3) The one case the original wording was right about.
    text = warn(
        {"org_unit": "retriever_org_unit", "user": "retriever_sign"},
        [["org_unit", "user"]],
        ["user"],
    )
    assert "no declared identity_keys candidate is fully satisfied" in text
    assert "catalog defect" not in text


def test_a_ONE_FIELD_fallback_does_not_silence_the_unresolved_key_warning(caplog):
    """One column is not a conjunction, so the fallback that resolves one must not mute it.

    The warning was suppressed as soon as `require_all_entities` resolved anything at all,
    and at one field nothing is AND-ed — so the weakest outcome the seam can return was the
    one it announced least, indistinguishable in the log from a satisfied pair. It now fires
    whenever a declaration went unmet and NAMES what the fallback got, because "nothing" and
    "one column" are different states with different remedies.
    """
    import logging

    from src.retrievers.query_guards import conjunction_fields

    field_map = {"org_unit": "retriever_org_unit", "user": "retriever_sign"}
    with caplog.at_level(logging.WARNING, logger="src.retrievers.query_guards"):
        fields = conjunction_fields(
            field_map,
            require_all_entities=["org_unit", "nope"],
            identity_keys=[["org_unit", "user"]],
            present_types=["org_unit"],
            source_name="raw_access",
        )
    assert fields == ["retriever_org_unit"]
    assert "will NOT be constrained to a single actor" in caplog.text
    assert "fallback resolved retriever_org_unit" in caplog.text


def test_a_one_member_require_all_entities_reads_like_a_one_member_key(caplog):
    """The two declarations spell the same weak answer, so they log the same way."""
    import logging

    from src.retrievers.query_guards import conjunction_fields

    field_map = {"org_unit": "retriever_org_unit"}
    with caplog.at_level(logging.INFO, logger="src.retrievers.query_guards"):
        fields = conjunction_fields(
            field_map,
            require_all_entities=["org_unit"],
            source_name="office_ref",
        )
    assert fields == ["retriever_org_unit"]
    assert "single column retriever_org_unit" in caplog.text
    assert "Nothing is AND-ed" in caplog.text
    assert "these are AND-ed" not in caplog.text


def test_a_malformed_key_member_cannot_RAISE_from_inside_a_guard():
    """A declaration nested one level too deep used to take the retrieval down.

    `{}.get(["a", "b"])` raises TypeError, so `identity_keys: [[[a, b]]]` — one bracket
    more than the schema — crashed the resolver, from inside a guard whose every other
    failure path degrades to "this declaration constrains nothing".
    """
    from src.retrievers.query_guards import (conjunction_fields,
                                             required_fields,
                                             resolve_identity_key)

    field_map = {"org_unit": "retriever_org_unit", "user": "retriever_sign"}
    assert resolve_identity_key(field_map, [[["org_unit", "user"]]], None) == []
    assert required_fields(field_map, [["org_unit"]]) == []
    assert (
        conjunction_fields(
            field_map,
            require_all_entities=[{"org_unit": 1}],
            identity_keys=[[None, 3]],
            present_types=["org_unit", "user"],
            source_name="raw_access",
        )
        == []
    )
    # A member that is merely unmapped is unchanged: dropped, and the candidate with it.
    assert resolve_identity_key(field_map, [["org_unit", "nope"]], None) == []


def test_the_prompt_hint_and_the_post_generation_guard_name_the_SAME_key():
    """A hint the rewrite does not back is just another prompt instruction.

    The two used to be computed by different code paths over different inputs
    (`render_filters` filtered by "has a value", the retrievers called `required_fields`
    directly), so they could name different tuples — indistinguishable from no guard.
    """
    from src.retrievers.query_guards import (conjunction_fields,
                                             enforce_conjunction)

    pack = _raw_access_pack()
    field_map = {
        "organization": "retriever_organization",
        "user": "retriever_user_id",
    }
    query = _q([("organization", "ORG-ALFAND"), ("user", "BSURNAME")])
    ik = pack.source("raw_access").identity_keys

    hints = render_filters(
        field_map,
        query,
        require_all_entities=["org_unit", "user"],
        knowledge_pack=pack,
        identity_keys=ik,
    )
    fields = conjunction_fields(
        field_map,
        require_all_entities=["org_unit", "user"],
        identity_keys=ik,
        present_types=[e.type for e in query.entities],
        source_name="raw_access",
    )
    for field in fields:
        assert field in hints
    # ...and those exact fields are what the rewrite acts on.
    sql = (
        "SELECT * FROM v WHERE access_date='2026-07-04' AND ("
        f"{fields[0]}='a' OR {fields[1]}='b')"
    )
    assert f"{fields[0]}='a' AND {fields[1]}='b'" in enforce_conjunction(
        sql, fields, "raw_access"
    )


def test_an_event_log_declaring_no_key_still_ORs_its_identities():
    """A lookup must AND its key; an event log must OR its identities. Opposite needs.

    auth_events deliberately declares neither key, and the measurement backs it:
    `login` is null on 30,548,028 of 95,422,170 rows (32.0%), the INVERSE of RAW_ACCESS's
    ordering — so ANDing (org_unit, login) there would drop a third of the population.
    """
    field_map = {"org_unit": "org_unit", "user": "sign"}
    hints = render_filters(
        field_map, _q([("org_unit", "ORG2428D4"), ("user", "0201GP")])
    )
    assert "MANDATORY" not in hints
    assert "org_unit = 'ORG2428D4'" in hints
    assert "sign = '0201GP'" in hints


def test_identity_keys_absent_changes_nothing():
    """Every pack key must no-op when absent."""
    field_map = {"org_unit": "orgUnitId", "user": "sign"}
    query = _q([("org_unit", "HHH1J09ST"), ("user", "6009JJ")])
    plain = render_filters(field_map, query, require_all_entities=["org_unit", "user"])
    with_empty = render_filters(
        field_map, query, require_all_entities=["org_unit", "user"], identity_keys=[]
    )
    assert plain == with_empty
    # The fixed tuple still works on its own.
    assert "COMBINATION of orgUnitId, sign" in plain


# ── a declared binding is not a prior, and the confidence gate cannot veto it ──
#
# The gate measures the model's uncertainty; a field name somebody measured is not
# something the model may withhold. A prompt rescuing a dropped binding is a guarantee
# that holds only as long as the prompt does.


def _blob_pack():
    """A source that binds two entity types onto one text blob, as a real one does."""
    return KnowledgePack(
        entities=[EntityDef(type="org_unit"), EntityDef(type="user")],
        sources=[
            SourceDef(
                name="alerts",
                entities=["org_unit", "user"],
                entity_bindings={"org_unit": ["alert.body"], "user": ["alert.body"]},
            )
        ],
    )


def _blob_query():
    return RetrievalQuery(
        target_log_source="alerts",
        natural_language_query="x",
        date_from="2026-03-05",
        date_to="2026-03-08",
        entities=[
            ExtractedEntity(type="org_unit", value="ORGUNIT2301"),
            ExtractedEntity(type="user", value="BSURNAME"),
        ],
    )


@pytest.mark.asyncio
async def test_a_declared_binding_survives_the_confidence_gate():
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=FieldMapping(
            mappings=[
                EntityFieldMap(
                    entity_type="org_unit", field="alert.body", confidence=0.35
                ),
                EntityFieldMap(entity_type="user", field="alert.body", confidence=0.40),
            ]
        )
    )
    schema = "alert.id, alert.body, timestamp, type"

    field_map = await map_entities(llm, _blob_query(), schema, _blob_pack())

    assert field_map == {"org_unit": "alert.body", "user": "alert.body"}


@pytest.mark.asyncio
async def test_a_declared_binding_is_adopted_when_the_mapper_offers_nothing():
    """Silence from the mapper must not delete the pack's measurement either.

    The gate is one way to lose a declared binding; an empty `mappings` list is the other,
    and it leaves exactly the same trace — a query with no actor predicate that still
    returns rows.
    """
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=FieldMapping(mappings=[]))

    field_map = await map_entities(
        llm, _blob_query(), "alert.id, alert.body, timestamp", _blob_pack()
    )

    assert field_map == {"org_unit": "alert.body", "user": "alert.body"}


@pytest.mark.asyncio
async def test_a_stale_declaration_is_dropped_not_filtered_on():
    """The discovered schema stays authoritative — this is the stale-binding failure.

    A predicate on a field the source no longer has matches nothing and still returns
    rows, so honouring the declaration here would be worse than dropping it: it reads as a
    successful retrieval that simply had no actor. This is the defect the whole re-point
    exercise turned up, so the fix above must not re-introduce it with a declaration's
    authority behind it.
    """
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=FieldMapping(mappings=[]))
    pack = KnowledgePack(
        entities=[EntityDef(type="org_unit")],
        sources=[
            SourceDef(
                name="alerts",
                entities=["org_unit"],
                # The pre-migration field name; the new index has no such field.
                entity_bindings={"org_unit": ["org_unit_id"]},
            )
        ],
    )
    q = RetrievalQuery(
        target_log_source="alerts",
        natural_language_query="x",
        date_from="2026-03-05",
        date_to="2026-03-08",
        entities=[ExtractedEntity(type="org_unit", value="ORGUNIT2301")],
    )

    assert await map_entities(llm, q, "alert.id, alert.body, timestamp", pack) == {}


@pytest.mark.asyncio
async def test_an_UNDECLARED_low_confidence_mapping_is_still_dropped():
    """The exemption is narrow: only the source's own bindings, never a global alias.

    A pack-wide `field_aliases` entry is a guess about what a column might be called
    somewhere; it has no business overriding the gate. If it did, the gate would be dead
    for every entity the glossary happens to alias.
    """
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=FieldMapping(
            mappings=[
                EntityFieldMap(
                    entity_type="org_unit", field="org_unit_id", confidence=0.3
                )
            ]
        )
    )
    pack = KnowledgePack(
        entities=[EntityDef(type="org_unit", field_aliases=["org_unit_id"])],
        sources=[SourceDef(name="transaction_logs", entities=["org_unit"])],
    )

    assert await map_entities(llm, _query(), "t(org_unit_id string)", pack) == {}


@pytest.mark.asyncio
async def test_a_declared_binding_for_the_WRONG_FORM_is_not_adopted():
    """The adoption path must not re-create the cross-form filter one layer earlier.

    `render_filters` drops a value whose form the source binds nothing for. A declared
    binding read flatly would hand it the sibling form's column instead — the same defect,
    with a declaration's authority behind it.
    """
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=FieldMapping(mappings=[]))
    pack = KnowledgePack(
        entities=[EntityDef(type="user")],
        sources=[
            SourceDef(
                name="auth",
                entities=["user"],
                entity_bindings={"user": {"login": ["userId"]}},
            )
        ],
    )
    q = RetrievalQuery(
        target_log_source="auth",
        natural_language_query="x",
        date_from="2026-03-05",
        date_to="2026-03-08",
        entities=[ExtractedEntity(type="user", value="6009JJ", value_form="sign")],
    )

    # `sign` binds nothing here, so the login column must NOT be adopted for it.
    assert await map_entities(llm, q, "auth(userId string, sign string)", pack) == {}


@pytest.mark.asyncio
async def test_the_schema_spelling_wins_over_the_declaration_s():
    """One backend upper-cases every unquoted identifier; the filter must name the real
    column, not the pack's casing of it."""
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=FieldMapping(mappings=[]))
    pack = KnowledgePack(
        entities=[EntityDef(type="org_unit")],
        sources=[
            SourceDef(
                name="warehouse",
                entities=["org_unit"],
                entity_bindings={"org_unit": ["orgUnitId"]},
            )
        ],
    )
    q = RetrievalQuery(
        target_log_source="warehouse",
        natural_language_query="x",
        date_from="2026-03-05",
        date_to="2026-03-08",
        entities=[ExtractedEntity(type="org_unit", value="ORGUNIT2301")],
    )

    field_map = await map_entities(
        llm, q, "T(ORGUNITID VARCHAR, TIME_STAMP TIMESTAMP)", pack
    )
    assert field_map == {"org_unit": "ORGUNITID"}


@pytest.mark.asyncio
async def test_no_discovered_schema_concludes_nothing_about_a_declaration():
    """Discovery can fail; absence must not be inferred from an empty schema.

    With no schema at all `map_entities` already short-circuits, so the guarantee here is
    that the confirmation step never turns a discovery failure into a silent unfiltered
    query by *adopting* an unconfirmable field.
    """
    llm = MagicMock()
    llm.structured_output = AsyncMock()
    pack = _blob_pack()

    assert await map_entities(llm, _blob_query(), "", pack) == {}
    llm.structured_output.assert_not_called()


@pytest.mark.asyncio
async def test_an_absent_schema_maps_nothing_but_NAMES_WHAT_THAT_COSTS(caplog):
    """The test above owns the ``{}``; this one owns the fact that it is not silent.

    Returning an empty field map is the right answer (see above) and it is also expensive in
    a way nothing downstream reports: `render_filters` emits no predicate, every guard in
    `query_guards` that reads the field map is inert, and `key_was_enforced` then answers
    False over whatever the prose hints produced. Measured on job 49e3cf7d, a source with a
    fully declared key came back with no field map, the SQL was correct anyway because the
    pack's `query_hints` re-create the clause, and one verdict line moved PASS -> UNKNOWN
    between two runs of one incident — with no log line anywhere saying the mapping step had
    concluded nothing. So the exit must state the consequence, and the declarations it is
    declining to apply are what makes the line actionable.
    """
    caplog.set_level(logging.WARNING, logger="src.retrievers.field_mapping")
    llm = MagicMock()
    llm.structured_output = AsyncMock()

    assert await map_entities(llm, _blob_query(), "", _blob_pack()) == {}

    warned = "\n".join(
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    )
    assert "no schema was discovered" in warned
    # The bindings it declined to apply, so the reader knows what was on the table.
    assert "org_unit->alert.body" in warned
    # And the consequence, not just the symptom: a reader who sees only "no schema" has no
    # reason to connect it to a verdict line that came back unknown.
    assert "inert" in warned and "decisive" in warned


@pytest.mark.asyncio
async def test_a_FAILED_mapping_call_does_not_discard_the_DECLARATION(caplog):
    """A throttled mapper says nothing about which column somebody measured.

    The other two ways to lose a declaration are answered above — the confidence gate and an
    empty ``mappings`` list. This is the third: the call itself raising. It used to return
    ``{}`` before the adoption step could run, which is the same silent outcome as the
    stale-binding case but for the opposite reason — nothing was wrong with the binding, and
    the schema was right there to confirm it against. A retrieval fan-out of ~19 sources
    against one rate-limited endpoint makes this the most reachable of the three.
    """
    caplog.set_level(logging.WARNING, logger="src.retrievers.field_mapping")
    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=RuntimeError("429 rate limited"))

    field_map = await map_entities(
        llm, _blob_query(), "alert.id, alert.body, timestamp", _blob_pack()
    )

    assert field_map == {"org_unit": "alert.body", "user": "alert.body"}
    # Still reported: the run has no model opinion on any entity the pack does NOT declare.
    assert "Entity field-mapping failed" in "\n".join(
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    )


@pytest.mark.asyncio
async def test_a_failed_mapping_call_still_drops_a_STALE_declaration():
    """The fix above must not become a route around the discovered schema.

    Adopting on failure is licensed only because the schema is available to confirm against.
    A field the source does not have would become a predicate matching zero rows, arriving
    with `key_was_enforced` True and read as a decisive negative — a fabricated PASS out of a
    broken query, which is the one direction forbidden here.
    """
    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=RuntimeError("429 rate limited"))
    pack = KnowledgePack(
        entities=[EntityDef(type="org_unit")],
        sources=[
            SourceDef(
                name="alerts",
                entities=["org_unit"],
                entity_bindings={"org_unit": ["org_unit_id"]},
            )
        ],
    )
    q = RetrievalQuery(
        target_log_source="alerts",
        natural_language_query="x",
        date_from="2026-03-05",
        date_to="2026-03-08",
        entities=[ExtractedEntity(type="org_unit", value="ORGUNIT2301")],
    )

    assert await map_entities(llm, q, "alert.id, alert.body, timestamp", pack) == {}


@pytest.mark.asyncio
async def test_a_declared_binding_outranks_a_CONFIDENT_wrong_pick():
    """The other half of the exemption: a high confidence is no more of a measurement.

    Measured live: a source whose alert discriminator lives in a free-text remarks field
    also has a same-named `type` column holding a different vocabulary. The mapper picked
    `type` confidently — it is in the schema and it is literally named after the entity — and
    the resulting term filter matched **0 documents**. The retrieval only survived because
    other clauses in the same boolean did the work, so the wrong binding left no trace.
    """
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=FieldMapping(
            mappings=[
                EntityFieldMap(entity_type="alert_type", field="type", confidence=0.95)
            ]
        )
    )
    pack = KnowledgePack(
        entities=[EntityDef(type="alert_type")],
        sources=[
            SourceDef(
                name="alerts",
                entities=["alert_type"],
                entity_bindings={"alert_type": ["ir.userRemarks"]},
            )
        ],
    )
    q = RetrievalQuery(
        target_log_source="alerts",
        natural_language_query="x",
        date_from="2026-03-05",
        date_to="2026-03-08",
        entities=[ExtractedEntity(type="alert_type", value="UBA")],
    )

    field_map = await map_entities(
        llm, q, "ir.id, ir.type, ir.userRemarks, ir.text, type", pack
    )
    assert field_map == {"alert_type": "ir.userRemarks"}


@pytest.mark.asyncio
async def test_a_STALE_declaration_does_not_override_a_confident_pick():
    """A declaration that cannot be honoured must change nothing.

    This is the mutation the test above would otherwise pass under: overriding on the
    declaration ALONE trades a wrong predicate for one on a field the source does not have,
    which matches nothing while the stage still reports success. The discovered schema stays
    authoritative in both directions.
    """
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=FieldMapping(
            mappings=[
                EntityFieldMap(entity_type="org_unit", field="orgUnitId", confidence=0.9)
            ]
        )
    )
    pack = KnowledgePack(
        entities=[EntityDef(type="org_unit")],
        sources=[
            SourceDef(
                name="alerts",
                entities=["org_unit"],
                entity_bindings={"org_unit": ["org_unit_id_v1"]},  # migrated away
            )
        ],
    )
    q = RetrievalQuery(
        target_log_source="alerts",
        natural_language_query="x",
        date_from="2026-03-05",
        date_to="2026-03-08",
        entities=[ExtractedEntity(type="org_unit", value="ORGUNIT2301")],
    )

    field_map = await map_entities(llm, q, "orgUnitId, timestamp", pack)
    assert field_map == {"org_unit": "orgUnitId"}


@pytest.mark.asyncio
async def test_the_override_never_crosses_a_value_FORM():
    """A form-bound entity whose form this source binds nothing for is not overridden.

    Adopting a flat reading here would put one form's value on the sibling's column with a
    declaration's authority behind it — the cross-form filter `render_filters` exists to
    prevent, one layer earlier. `_declared_fields` is form-scoped, so the override simply
    does not fire and the mapper's pick stands.
    """
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=FieldMapping(
            mappings=[EntityFieldMap(entity_type="user", field="sign", confidence=0.9)]
        )
    )
    pack = KnowledgePack(
        entities=[EntityDef(type="user")],
        sources=[
            SourceDef(
                name="auth",
                entities=["user"],
                entity_bindings={"user": {"login": ["userId"]}},
            )
        ],
    )
    q = RetrievalQuery(
        target_log_source="auth",
        natural_language_query="x",
        date_from="2026-03-05",
        date_to="2026-03-08",
        entities=[ExtractedEntity(type="user", value="6009JJ", value_form="sign")],
    )

    field_map = await map_entities(llm, q, "auth(userId string, sign string)", pack)
    assert field_map == {"user": "sign"}


def test_a_column_is_never_confused_with_its_own_TYPE():
    """`_schema_field_names` takes the first identifier of each fragment only.

    Otherwise `date` as a type name would confirm a stale binding to a `date` column, and
    the stale-binding check above would pass on a source that has no such field.
    """
    from src.retrievers.field_mapping import _in_schema, _schema_field_names

    names = _schema_field_names("t(creation_date date, amount decimal, sign string)")
    assert _in_schema("creation_date", names) == "creation_date"
    assert _in_schema("date", names) is None
    assert _in_schema("string", names) is None


# ── a value's precision: the same identity, two lengths ──
#
# A form may carry an optional trailing part; a target may store only the stem. The reading
# side has always been prefix-tolerant; the predicate must offer both lengths. A lookup
# rendering the long form on a column holding only stems returned 0 rows on a source
# whose empty result is the decisive answer.


def _stem_pack(stem=r"^([0-9]{4}[A-Z]{2})"):
    """A `user` whose `sign` form names its identity CORE with one capture group."""
    return KnowledgePack(
        entities=[
            EntityDef(
                type="user",
                field_aliases=["sign"],
                value_forms=[
                    {
                        "name": "sign",
                        "pattern": r"^[0-9]{4}[A-Z]{2}([A-Z]{2})?$",
                        "stem": stem,
                    },
                    {"name": "login", "pattern": r"^[A-Z][A-Z0-9]{1,9}$"},
                ],
            ),
            EntityDef(type="org_unit", field_aliases=["unitId"]),
        ],
        sources=[
            SourceDef(
                name="robot_register",
                entities=["user", "org_unit"],
                entity_bindings={"user": {"sign": ["sign"]}, "org_unit": ["unitId"]},
            )
        ],
    )


def _stem_query(value="7777ABCD"):
    return RetrievalQuery(
        target_log_source="robot_register",
        natural_language_query="x",
        date_from="2026-08-01",
        date_to="2026-08-02",
        entities=[
            ExtractedEntity(type="user", value=value, value_form="sign"),
            ExtractedEntity(type="org_unit", value="OFFICE01"),
        ],
    )


def test_the_declared_stem_of_a_value_is_derived_from_its_form():
    """`value_stem` is a pack accessor, so nothing in `src/` names a domain shape."""
    pack = _stem_pack()
    assert pack.value_stem("user", "7777ABCD") == "7777AB"
    # Already the stem: there is no OTHER form of it, so there is nothing to offer.
    assert pack.value_stem("user", "7777AB") is None
    # A different form of the same entity type declares no stem.
    assert pack.value_stem("user", "BSURNAME") is None
    # Unclassifiable, unknown type, empty — all simply None, never an error.
    assert pack.value_stem("user", "??") is None
    assert pack.value_stem("nope", "7777ABCD") is None
    assert pack.value_stem("user", "") is None
    # A pack declaring no stem at all behaves exactly as before.
    assert _stem_pack(stem="").value_stem("user", "7777ABCD") is None
    # An unparseable regex is a pack defect, reported and skipped — not a crash mid-retrieval.
    assert _stem_pack(stem="^([0-9").value_stem("user", "7777ABCD") is None
    # A pattern that captures nothing (no group) cannot say what the stem is.
    assert _stem_pack(stem="^[0-9]{4}").value_stem("user", "7777ABCD") is None


def test_the_stem_is_offered_BESIDE_the_value_and_never_instead_of_it():
    """Which precision a column stores is declared nowhere, so the data decides.

    The epoch guard REPLACES its literal because a unit conversion is computable. This one
    cannot know whether the target keeps the trailing part, so both spellings go and a
    column holding the long form is unaffected.
    """
    pack = _stem_pack()
    out = render_filters(
        {"user": "sign", "org_unit": "unitId"}, _stem_query(), knowledge_pack=pack
    )
    assert "sign IN ('7777ABCD', '7777AB')" in out
    # The stem is a strict prefix, so on a long-form column the extra literal matches nothing:
    # it can only add rows the narrow predicate would have missed.
    assert "unitId = 'OFFICE01'" in out
    # Without the pack, and with a pack that declares no stem, byte-identical to before.
    plain = "sign = '7777ABCD'"
    assert plain in render_filters({"user": "sign"}, _stem_query())
    assert plain in render_filters(
        {"user": "sign"}, _stem_query(), knowledge_pack=_stem_pack(stem="")
    )


def test_stem_literals_resolves_only_what_the_HINT_already_offered():
    """The guard's whole confinement: same seam, same column, same literal.

    `filter_values_by_field` is shared verbatim by the prompt hint and by this resolver, so
    the rewrite cannot widen a column the hint never asked about — which is what keeps the one
    additive guard in the module inside the never-additive contract.
    """
    from src.retrievers.field_mapping import stem_literals

    pack = _stem_pack()
    field_map = {"user": "sign", "org_unit": "unitId"}
    assert stem_literals(field_map, _stem_query(), pack) == {
        "sign": {"7777ABCD": "7777AB"}
    }
    # No pack, no stem declared, no field map, and a value already at stem precision: {}.
    assert stem_literals(field_map, _stem_query()) == {}
    assert stem_literals(field_map, _stem_query(), _stem_pack(stem="")) == {}
    assert stem_literals({}, _stem_query(), pack) == {}
    assert stem_literals(field_map, _stem_query("7777AB"), pack) == {}
    # A pack-shaped object without the accessor contributes nothing rather than raising: the
    # resolvers here run against mocks and stubs all over this suite.
    assert stem_literals(field_map, _stem_query(), object()) == {}


def _tuple_query(tuples, value="7777ABCD"):
    """A pass-2 query carrying the combinations an earlier pass's rows actually showed."""
    query = _stem_query(value)
    query._value_tuples = tuples
    return query


def test_value_tuple_columns_resolves_combinations_through_the_SAME_router():
    """Which column holds a value is answered ONCE, or the prompt and the guard disagree.

    The per-value filter hints and this resolver both go through `_routed_field`, so a form
    routed to its own column for the hint cannot be routed to the sibling's column for the
    guard that narrows the combinations — the way a declaration comes to be honoured on one
    seam and silently ignored on the other.
    """
    from src.retrievers.field_mapping import value_tuple_columns

    pack = _stem_pack()
    field_map = {"user": "sign", "org_unit": "unitId"}
    query = _tuple_query(
        [
            [
                {"type": "user", "value": "7777ABCD", "value_form": "sign"},
                {"type": "org_unit", "value": "OFFICE01"},
            ],
            [
                {"type": "user", "value": "1111ABCD", "value_form": "sign"},
                {"type": "org_unit", "value": "OFFICE02"},
            ],
        ]
    )
    assert value_tuple_columns(field_map, query, pack) == [
        [("sign", ["7777ABCD", "7777AB"]), ("unitId", ["OFFICE01"])],
        [("sign", ["1111ABCD", "1111AB"]), ("unitId", ["OFFICE02"])],
    ]
    # The declared stem rides along as an ALTERNATIVE spelling of the same component, because
    # which precision a column stores is declared nowhere: the guard matches whichever the
    # generator wrote. It is not a second component and cannot pair with anything.
    assert value_tuple_columns(field_map, query, _stem_pack(stem="")) == [
        [("sign", ["7777ABCD"]), ("unitId", ["OFFICE01"])],
        [("sign", ["1111ABCD"]), ("unitId", ["OFFICE02"])],
    ]


def test_a_component_this_source_cannot_bind_is_dropped_not_the_combination():
    """A projection of a real co-occurrence is narrower but still TRUE.

    Dropping a part during the harvest would invent a combination; dropping one here only
    forgets a constraint this source has no column for. What must not survive is a projection
    down to ONE column — that is the per-type filter hint, and enforcing it as a combination
    would publish a narrowing that is not one.
    """
    from src.retrievers.field_mapping import value_tuple_columns

    pack = _stem_pack()
    tuples = [
        [
            {"type": "user", "value": "7777ABCD", "value_form": "sign"},
            {"type": "org_unit", "value": "OFFICE01"},
            {"type": "unmapped_type", "value": "X1"},
        ]
    ]
    query = _tuple_query(tuples)
    # The third component's type is in no field map: the other two still occurred together.
    assert value_tuple_columns(
        {"user": "sign", "org_unit": "unitId"}, query, pack
    ) == [[("sign", ["7777ABCD", "7777AB"]), ("unitId", ["OFFICE01"])]]
    # One bindable component left -> nothing to enforce.
    assert value_tuple_columns({"user": "sign"}, query, pack) == []
    # Two components routed onto ONE column is one column, so also nothing: the arity is
    # counted in COLUMNS, never in parts. (`user` is left out of the map here because its form
    # binding names its column outright — see below.)
    assert value_tuple_columns({"org_unit": "c", "unmapped_type": "c"}, query, pack) == []
    # And the router, not the field map, is what named that column: a form-bound component
    # goes to its own form's column whatever the mapper returned for the type.
    assert value_tuple_columns({"user": "c", "org_unit": "unitId"}, query, pack) == [
        [("sign", ["7777ABCD", "7777AB"]), ("unitId", ["OFFICE01"])]
    ]
    # A form this source binds no column for is dropped by the router, same as an unknown type.
    login = _tuple_query(
        [
            [
                {"type": "user", "value": "BSURNAME", "value_form": "login"},
                {"type": "org_unit", "value": "OFFICE01"},
            ]
        ]
    )
    assert value_tuple_columns({"user": "sign", "org_unit": "unitId"}, login, pack) == []


def test_value_tuple_columns_is_empty_for_every_single_pass_query():
    """The default must be inert: a pack declaring no follow-up pass runs byte-identically."""
    from src.retrievers.field_mapping import value_tuple_columns

    pack = _stem_pack()
    field_map = {"user": "sign", "org_unit": "unitId"}
    # No combinations harvested (pass 1, and every query of a single-pass run).
    assert value_tuple_columns(field_map, _stem_query(), pack) == []
    assert value_tuple_columns(field_map, _tuple_query([]), pack) == []
    # No field map, a malformed part, a blank value, and the window entity — all just skipped.
    assert value_tuple_columns({}, _tuple_query([[{"type": "user", "value": "x"}]]), pack) == []
    junk = _tuple_query(
        [
            None,
            [None, {}, {"type": "user", "value": ""}, {"type": "", "value": "v"}],
            [
                {"type": "user", "value": "7777ABCD", "value_form": "sign"},
                {"type": "time_window", "value": "2026-08-01"},
            ],
        ]
    )
    assert value_tuple_columns(
        {"user": "sign", "org_unit": "unitId", "time_window": "d"}, junk, pack
    ) == []
    # And a query object that has never heard of combinations (a mock, an older model) is not
    # an error: this resolver runs on every retrieval.
    assert value_tuple_columns(field_map, MagicMock(spec=[]), pack) == []


# --- key_presence_values: columns holding the resolved key ----------------------
# Input is what `conjunction_fields` already resolved; no second key ranking. Output is
# per entity type: columns for one type are OR-ed, types are AND-ed between them.


def _key_query(value="7777ABCD", extra=None):
    query = _stem_query(value)
    if extra:
        query.entities = list(query.entities) + list(extra)
    return query


def test_key_presence_values_resolves_ONLY_the_resolved_keys_members():
    """A mapped entity that is not part of the key contributes nothing.

    This is the whole safety argument of the guard it feeds: AND-ing a non-key entity is the
    whole-population narrowing that module refuses to invent, and it is also the shape that
    genuinely voids `key_was_enforced`'s absence proof.
    """
    from src.retrievers.field_mapping import key_presence_values

    pack = _stem_pack(stem="")
    field_map = {"user": "sign", "org_unit": "unitId"}
    query = _key_query()
    # Both members of a two-column key, each under its own type.
    assert key_presence_values(field_map, query, ["sign", "unitId"], pack, "reg") == {
        "user": {"sign": ["7777ABCD"]},
        "org_unit": {"unitId": ["OFFICE01"]},
    }
    # A ONE-column key over the same map and the same incident: `user` is mapped, carries a
    # value, and is still absent, because the key resolution did not name its column.
    assert key_presence_values(field_map, query, ["unitId"], pack, "reg") == {
        "org_unit": {"unitId": ["OFFICE01"]}
    }
    # The types are recovered from the SAME field_map the resolution read, so a key field
    # this source maps to no type resolves to nothing rather than to a guess.
    assert key_presence_values(field_map, query, ["some_other_column"], pack, "reg") == {}


def test_key_presence_values_is_empty_wherever_there_is_nothing_to_enforce():
    """Four silences, and each is a source or an incident the guard must leave alone."""
    from src.retrievers.field_mapping import key_presence_values

    pack = _stem_pack(stem="")
    field_map = {"user": "sign", "org_unit": "unitId"}
    # No key resolved: an event log, or a candidate no incident satisfies. `conjunction_fields`
    # returns nothing and there is no conjunction to complete.
    assert key_presence_values(field_map, _key_query(), [], pack, "reg") == {}
    assert key_presence_values(field_map, _key_query(), None, pack, "reg") == {}
    assert key_presence_values(field_map, _key_query(), [""], pack, "reg") == {}
    # No field map: nothing is bound on this source, so nothing can be injected onto it.
    assert key_presence_values({}, _key_query(), ["sign", "unitId"], pack, "reg") == {}
    # A member the INCIDENT does not carry: there is no literal, and inventing one is the one
    # direction forbidden in the guard this feeds.
    only_office = RetrievalQuery(
        target_log_source="robot_register",
        natural_language_query="x",
        date_from="2026-08-01",
        date_to="2026-08-02",
        entities=[ExtractedEntity(type="org_unit", value="OFFICE01")],
    )
    assert key_presence_values(field_map, only_office, ["sign", "unitId"], pack, "reg") == {
        "org_unit": {"unitId": ["OFFICE01"]}
    }
    # And a member whose value FORM this source binds no column for is dropped by the router,
    # exactly as it is for the filter hints — never unioned onto the sibling form's column.
    login = _key_query(value="BSURNAME")
    login.entities[0].value_form = "login"
    assert key_presence_values(field_map, login, ["sign", "unitId"], pack, "reg") == {
        "org_unit": {"unitId": ["OFFICE01"]}
    }


def test_two_key_members_on_ONE_column_are_REFUSED_not_injected(caplog):
    """The one refusal, and it is a sibling guard's case rather than an edge case.

    Two members routing onto one column would be AND-ed as `col IN (a) AND col IN (b)`, which
    matches no row — so the wrong query would become an empty one, in the guard added to stop
    exactly that. A source storing several key members in one column is
    `enforce_conjunction_same_column`'s shape, and it is enforced there.
    """
    import logging

    from src.retrievers.field_mapping import key_presence_values

    pack = _stem_pack(stem="")
    query = _key_query(extra=[ExtractedEntity(type="session", value="SESS7")])
    field_map = {"org_unit": "actor_key", "session": "actor_key"}
    with caplog.at_level(logging.WARNING):
        assert key_presence_values(field_map, query, ["actor_key"], pack, "reg") == {}
    assert "route onto" in caplog.text and "actor_key" in caplog.text
    # It is a per-COLUMN refusal and not a whole-key one: a member on a column of its own is
    # still enforced beside the collapsed pair, or one blob column would disable the guard for
    # every other member of the same key.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        out = key_presence_values(
            {"org_unit": "actor_key", "session": "actor_key", "user": "sign"},
            query,
            ["actor_key", "sign"],
            pack,
            "reg",
        )
    assert out == {"user": {"sign": ["7777ABCD"]}}


def test_key_presence_values_offers_a_declared_STEM_beside_the_value():
    """It is the same resolution as the filter hints, so it inherits their fourth property.

    A value stored at the identity CORE rather than in full is the epoch guard's defect one
    type over: a valid predicate matching nothing. The injected group must therefore offer the
    same alternatives the hint would have, or the guard narrows onto a column the pack already
    measured as holding the short form.
    """
    from src.retrievers.field_mapping import filter_values_by_field, key_presence_values

    pack = _stem_pack()  # the `sign` form declares its stem
    field_map = {"user": "sign", "org_unit": "unitId"}
    query = _key_query()
    assert key_presence_values(field_map, query, ["sign", "unitId"], pack, "reg") == {
        "user": {"sign": ["7777ABCD", "7777AB"]},
        "org_unit": {"unitId": ["OFFICE01"]},
    }
    # And it is the SAME resolution and not a second one: flattening it yields exactly what
    # the filter hints were built from, which is the property that keeps a hint and an
    # injected predicate from naming different columns or different literals.
    flat = {}
    for cols in key_presence_values(field_map, query, ["sign", "unitId"], pack, "reg").values():
        for col, vals in cols.items():
            flat.setdefault(col, []).extend(v for v in vals if v not in flat.get(col, []))
    assert flat == filter_values_by_field(field_map, query, pack)


def test_filter_values_by_field_is_unchanged_by_the_per_type_refactor():
    """The flat resolver is now a flattening of the per-type one, so pin it independently.

    Two types sharing one column must still UNION there — that is the shape the per-type
    reading exists to keep separate, and a refactor that let the flattening drop one of them
    would be invisible in every per-type assertion above.
    """
    from src.retrievers.field_mapping import filter_values_by_field

    pack = _stem_pack(stem="")
    query = _key_query(extra=[ExtractedEntity(type="session", value="SESS7")])
    assert filter_values_by_field(
        {"org_unit": "actor_key", "session": "actor_key", "user": "sign"}, query, pack
    ) == {"actor_key": ["OFFICE01", "SESS7"], "sign": ["7777ABCD"]}


# ── a value's width: a fixed-width part of the stored identifier ──
#
# A `match` places a short value as a positional window inside the stored one. Inverse of
# the stem guard: stem extracts the core from a long value; match locates a short value
# within a longer one.


def _match_pack(match="{value}??????"):
    """An `org_unit` whose short form declares WHERE it sits inside the stored value."""
    return KnowledgePack(
        entities=[
            EntityDef(
                type="org_unit",
                field_aliases=["unitId"],
                value_forms=[
                    {"name": "full", "pattern": r"^[A-Z]{3}[A-Z0-9]{6}$"},
                    {
                        "name": "location_code",
                        "pattern": r"^[A-Z]{3}$",
                        "match": match,
                    },
                    {
                        "name": "corporate_code",
                        "pattern": r"^[A-Z0-9]{2,3}$",
                        "match": "???{value}????",
                    },
                ],
            ),
            EntityDef(type="user", field_aliases=["sign"]),
        ],
        sources=[
            SourceDef(
                name="robot_register",
                entities=["org_unit", "user"],
                entity_bindings={"org_unit": ["unitId"], "user": ["sign"]},
            )
        ],
    )


def _match_query(*values):
    return RetrievalQuery(
        target_log_source="robot_register",
        natural_language_query="x",
        date_from="2026-08-01",
        date_to="2026-08-02",
        entities=[ExtractedEntity(type="org_unit", value=v) for v in (values or ("DEL",))]
        + [ExtractedEntity(type="user", value="7777ABCD")],
    )


def test_match_patterns_resolves_only_what_the_HINT_already_offered():
    """The same confinement the stem resolver has, for the same reason and through one seam.

    `filter_values_by_field` is shared verbatim by the generator's filter hint and by this
    resolver, so the guard can only offer a second way of comparing a value the query already
    carried on a column the hint already named — which is what keeps an additive rewrite inside
    the never-additive contract.
    """
    from src.retrievers.field_mapping import match_patterns

    pack = _match_pack()
    field_map = {"org_unit": "unitId", "user": "sign"}
    assert match_patterns(field_map, _match_query("DEL", "DAC"), pack) == {
        "unitId": {"DEL": "DEL??????", "DAC": "DAC??????"}
    }
    # A value already at the STORED width has no other form to compare against, so it
    # contributes nothing and the column drops out entirely.
    assert match_patterns(field_map, _match_query("LLL1M13YZ"), pack) == {}
    # The pattern is resolved per FORM, not per type: the two-character form declares its own
    # window, and both forms of one type land on that type's column together.
    assert match_patterns(field_map, _match_query("DEL", "2A"), pack) == {
        "unitId": {"DEL": "DEL??????", "2A": "???2A????"}
    }
    # A value belonging to a type whose forms declare no `match` is untouched — which is every
    # entity of every pack until one opts in, and the reason the guard is a no-op and not a
    # migration.
    assert "sign" not in match_patterns(field_map, _match_query(), pack)
    assert match_patterns(field_map, _match_query(), _match_pack(match="")) == {}
    # No pack, no field map, a column the map does not carry: {} rather than a guess.
    assert match_patterns(field_map, _match_query()) == {}
    assert match_patterns({}, _match_query(), pack) == {}
    assert match_patterns({"user": "sign"}, _match_query(), pack) == {}
    # A pack-shaped object with no accessor contributes nothing rather than raising: these
    # resolvers run against mocks and stubs throughout this suite.
    assert match_patterns(field_map, _match_query(), object()) == {}


def test_the_match_pattern_is_offered_BESIDE_the_value_like_the_stem():
    """Both widenings are additive, and neither may become a substitution.

    Which precision a column stores is declared nowhere per column — that is the whole premise
    — so the equality stays and the pattern rides beside it. The hint keeps the value because a
    column that really does store the segment must still match, and the guard keeps the pattern
    because a column storing the whole identifier otherwise matches nothing.
    """
    from src.retrievers.field_mapping import match_patterns, stem_literals

    pack = _match_pack()
    field_map = {"org_unit": "unitId", "user": "sign"}
    query = _match_query("DEL")
    # The filter hint itself is unchanged by the declaration: the widening happens in the
    # guard, over the published text, so the two cannot disagree about what was asked.
    assert "unitId = 'DEL'" in render_filters(field_map, query, knowledge_pack=pack)
    assert "unitId LIKE" not in render_filters(field_map, query, knowledge_pack=pack)
    # And the two resolvers are independent: a `match` is not a `stem` and vice versa, so a
    # pack declaring one gets exactly one of them.
    assert match_patterns(field_map, query, pack) == {"unitId": {"DEL": "DEL??????"}}
    assert stem_literals(field_map, query, pack) == {}
