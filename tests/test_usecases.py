"""Tests for the use-case case-builders (src/usecases/)."""

from types import SimpleNamespace

from src.knowledge.pack import KnowledgePack, load_knowledge_pack
from src.models.pydantic_models import ExtractedEntity, TransformResult
from src.usecases.base import UseCaseAnalyzer
from src.usecases.registry import get_analyzer
from src.utils.paths import REPO_ROOT

# A minimal SCHEME-shaped ruleset spec exercising the decisive conditions the analyzer reads.
_SPEC = {
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
            "label": "Bare record",
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
            "id": "same_agent",
            "label": "Same agent",
            "kind": "field_equality",
            "left": {"source": "record", "field": "creator.sign.red"},
            "right": {"source": "settlement", "field": "retrieverUserSign"},
            "normalize": "identifier",
            "decisive": True,
        },
        {
            "id": "immediate_issuance",
            "label": "Issued within 1h",
            "kind": "time_gap",
            "start": {"source": "record", "field": "creation_date_time"},
            "end": {"source": "settlement", "field": "transactionDateTime"},
            "max": "1h",
            "decisive": True,
        },
    ],
    "lock_target": {
        "source": "record",
        "scope_field": "creator.org_unit_id",
        "identity_field": "creator.sign.red",
    },
    # A FLAT per-event asset feed (one row per issue/void/refund) — the shape a real
    # settlement report has. The nested-document shape the examplecorp SCHEME pack uses is
    # covered separately below.
    "asset_timeline": {
        # The creation event is pack-declared now — the engine no longer carries any
        # backend's creation-column names, so a fixture that wants the entry declares it.
        "subject_created": {
            "in": "record",
            "event_type": "record_created",
            "entity_type": "record",
            "detail": "record created",
            "subject_id_fields": ["locator.red", "recordLocator"],
            "timestamp_fields": ["creation_date_time"],
            "actor_fields": ["creator.sign.red"],
            "actor_scope_fields": ["creator.org_unit_id"],
        },
        "events": {
            "source": "settlement",
            "entity_type": "document",
            "timestamp_fields": ["transactionDateTime"],
            "actor_fields": ["retrieverUserSign"],
            "actor_scope_fields": ["retrieverOrgUnitId"],
            "type_fields": ["recordType", "recordSubType"],
            "asset_id_fields": ["relatedDocumentNumber"],
            "type_events": {"VOID": "voided", "REF": "refunded"},
            "default_event": "issued",
        }
    },
    "routes": [["DSS", "CDG"]],
    # Pack declarations the analyzer surfaces. A fixture that wants the projection guard
    # or scope notes must declare them (pack that declares none gets none).
    "case_builder": {
        "concepts": ["bare_record"],
        "expected_joins": ["join_record", "join_user"],
        "projection_guard": {
            "source": "record",
            "probe_paths": [
                {
                    "path": "element_counters.AUX",
                    "note": (
                        "record element_counters.AUX missing from the returned rows — the "
                        "projection did not surface the element counters."
                    ),
                }
            ],
            "empty_note": (
                "No rows returned for the record source ({source}); verdict conditions over "
                "the record are UNKNOWN."
            ),
        },
        "scope_notes": {
            "unverified": (
                "Impact scope UNVERIFIED: the actor-scoped sweep did not run ({status}). "
                "More may be impacted by the same sign."
            ),
            "wider": (
                "The sweep found {count} impacted subject(s) the alert did NOT name "
                "({subjects}) — the alert under-reported the scope."
            ),
        },
    },
}


# `_SPEC` plus the mock_domain pack's own next-step wording (read from the pack, not
# restated). mock_domain ships on every branch; an installed-pack dependency would need
# skipping wherever it is absent, which reads identical to passing.
_SPEC_WITH_PACK_WORDING = dict(_SPEC)
_SPEC_WITH_PACK_WORDING["case_builder"] = dict(
    _SPEC["case_builder"],
    action_templates=(
        (
            (
                load_knowledge_pack(
                    REPO_ROOT / "knowledge" / "mock_domain"
                ).ruleset_spec("refund_fraud")
                or {}
            ).get("case_builder", {})
            or {}
        ).get("action_templates", {})
        or {}
    ),
)


def _analysis(record="SUBJ03"):
    return SimpleNamespace(
        incident_summary="SCHEME issuance fraud",
        initial_hypotheses=[],
        key_investigation_areas=[],
        correlation_keys=[],
        extracted_entities=[ExtractedEntity(type="record", value=record)],
        event_time=None,
    )


def _pack_with_case(verdict="FALSE POSITIVE", route="DSS-CDG"):
    pack = KnowledgePack(name="t")
    pack.concept_documents = [
        {
            "title": "Bare record",
            "content": "---\nconcept_id: bare_record\n---\nA bare record carries only NM.",
            "type": "concept",
            "metadata": {"use_case": "scheme", "concept_id": "bare_record"},
        }
    ]
    pack.case_documents = [
        {
            "title": "case_a1b2c3",
            "content": "---\ncase_id: case_a1b2c3\n---\nClosed FALSE POSITIVE.",
            "type": "case",
            "metadata": {
                "use_case": "scheme",
                "case_id": "case_a1b2c3",
                "verdict": verdict,
                "subject": "SUBJ03",
                "route": route,
                "decisive_reasons": ["Not bare: AUX present"],
                "resolution": "Closed by GET-TOM.",
                "date": "2026-07-24",
            },
        }
    ]
    return pack


# --- verdict paths -----------------------------------------------------------


def test_scheme_analyzer_false_positive_not_bare():
    logs = {
        "record_lake": [
            {
                "locator.red": "SUBJ03",
                "creator.sign.red": "0201GPSU",
                "creator.org_unit_id": "ORG2428D4",
                "creation_date_time": "2026-07-24T19:14:00Z",
                "element_counters.AUX": 3,  # NOT bare -> decisive FAIL
                "element_counters.OSI": 0,
                "element_counters.RM": 0,
                "element_counters.INS": 0,
                "route.air.board_point": "DSS",
                "route.air.off_point": "CDG",
            }
        ],
        "settlement_report": [
            {
                "retrieverUserSign": "0201GP",
                "transactionDateTime": "2026-07-24T20:19:00Z",
                "recordType": "SALE",
            }
        ],
    }
    a = UseCaseAnalyzer(use_case="scheme").analyze(
        _SPEC, logs, _analysis(), knowledge_pack=_pack_with_case()
    )
    assert a.verdict is not None
    assert "FALSE POSITIVE" in a.verdict.summary
    brief = a.brief
    assert brief.use_case == "scheme"
    assert any(c.id == "bare" for c in brief.decisive_fails)
    # `_SPEC` has no `action_templates`, so this tests the domain-neutral default wording.
    # The pack's own "CLOSE the IR" phrasing is asserted separately in
    # test_pack_declared_action_templates_replace_the_neutral_wording.
    joined = " ".join(brief.action_backbone).lower()
    assert "close the incident as false positive" in joined
    assert "suspend" not in joined and "lock the order" not in joined


def test_scheme_analyzer_valid_fraud_all_pass():
    logs = {
        "record_lake": [
            {
                "locator.red": "AAAAAA",
                "creator.sign.red": "0201GP",
                "creator.org_unit_id": "ORG2428D4",
                "creation_date_time": "2026-07-24T20:00:00Z",
                "element_counters.AUX": 0,
                "element_counters.OSI": 0,
                "element_counters.RM": 0,
                "element_counters.INS": 0,
            }
        ],
        "settlement_report": [
            {
                "recordLocator": "AAAAAA",  # so the subject filter keeps this row
                "retrieverUserSign": "0201GP",  # same agent (prefix match)
                "transactionDateTime": "2026-07-24T20:30:00Z",  # within 1h
                "recordType": "SALE",
            }
        ],
    }
    a = UseCaseAnalyzer(use_case="scheme").analyze(
        _SPEC, logs, _analysis("AAAAAA"), knowledge_pack=_pack_with_case("VALID FRAUD")
    )
    assert "VALID FRAUD" in a.verdict.summary
    joined = " ".join(a.brief.action_backbone).lower()
    # The containment target is named, with the fields it was read from — but this spec
    # declares no per-platform action set, so NO action verb is invented. "LOCK" is not a
    # safe default: Sell Connect is FROZEN, not locked, and the two sets do not mix.
    assert "containment target 0201gp @ org2428d4" in joined
    assert "creator.sign.red" in joined and "creator.org_unit_id" in joined
    assert "no action verb could be resolved" in joined
    assert "lock" not in joined.replace("lock or freeze", "")
    # lock target names the creator identity/scope.
    assert a.brief.lock_targets and a.brief.lock_targets[0]["identity"] == "0201GP"


def test_scheme_analyzer_action_verb_follows_the_selling_platform():
    """Classic locks, Connect freezes, an automated sign is escalated instead.

    The three are not interchangeable, so the verb is resolved from evidence (an ATID's
    presence, the sign's class) rather than defaulted — and the blocking customer-contact
    prerequisite is emitted BEFORE the action it blocks, since a precondition rendered
    anywhere else is one a reader can act past.
    """
    import copy

    def _run(spec, creator_sign="0201GP", atid=None):
        row = {
            "locator.red": "AAAAAA",
            "creator.sign.red": creator_sign,
            "creator.org_unit_id": "ORG2428D4",
            "creation_date_time": "2026-07-24T20:00:00Z",
            "element_counters.AUX": 0,
            "element_counters.OSI": 0,
            "element_counters.RM": 0,
            "element_counters.INS": 0,
        }
        if atid is not None:
            row["contextual_data.sec.atid.red"] = atid
        logs = {
            "record_lake": [row],
            "settlement_report": [
                {
                    "recordLocator": "AAAAAA",
                    "retrieverUserSign": creator_sign,
                    "transactionDateTime": "2026-07-24T20:30:00Z",
                    "recordType": "SALE",
                }
            ],
        }
        a = UseCaseAnalyzer(use_case="scheme").analyze(spec, logs, _analysis("AAAAAA"))
        return " ".join(a.brief.action_backbone), a.verdict.subjects[0].lock_target

    spec = copy.deepcopy(_SPEC)
    spec["lock_target"]["actions"] = {
        "classic": "LOCK the Sell Classic agent account.",
        "connect": "FREEZE the Sell Connect user (NOT locked).",
        "unknown": "DETERMINE the selling platform first.",
    }
    spec["lock_target"]["prerequisites"] = [
        "BLOCKING PREREQUISITE — contact the customer and log it first."
    ]
    spec["platform_mode"] = {
        "source": "record",
        "marker_label": "ATID",
        "marker_fields": ["contextual_data.sec.atid.red"],
        "present": {
            "id": "classic",
            "label": "Sell Classic",
            "known_prefixes": ["58"],
        },
        "absent": {"id": "connect", "label": "Sell Connect"},
    }
    spec["identity_classes"] = [
        {
            "id": "web_service_sign",
            "patterns": [r"^600\dJJ$"],
            "action": "ESCALATE to the market ACO — do NOT lock or freeze.",
            "rationale": "An automated identity: locking it stops the application.",
        }
    ]

    # An ATID is present → Classic → LOCK, and the prerequisite precedes it.
    steps, lt = _run(spec, atid="58ABC")
    assert lt["platform"] == "classic" and "LOCK the Sell Classic" in lt["action"]
    assert steps.index("BLOCKING PREREQUISITE") < steps.index("LOCK the Sell Classic")

    # The field is retrieved and EMPTY → a real absence → Connect → FREEZE, never lock.
    steps, lt = _run(spec, atid="")
    assert lt["platform"] == "connect" and "FREEZE" in lt["action"]
    assert "LOCK" not in lt["action"]

    # The field was never retrieved → the platform is UNKNOWN, not Connect. Reading a
    # projection gap as "no ATID" selects one of two mutually exclusive action sets on no
    # evidence; on incident 83e94dd6 no ATID path was in the projection at all.
    steps, lt = _run(spec)
    assert lt["platform"] == "unknown"
    assert "DETERMINE the selling platform first" in lt["action"]

    # An automated sign outranks the platform verb entirely: escalate, do not contain.
    steps, lt = _run(spec, creator_sign="6009JJ", atid="58ABC")
    assert lt["identity_class"] == "web_service_sign"
    assert "ESCALATE to the market ACO" in lt["action"]
    assert "LOCK the Sell Classic" not in steps


def test_scheme_analyzer_insufficient_missing_settlement():
    # No settlement rows -> same_agent / immediate_issuance are UNKNOWN (decisive) -> INSUFFICIENT.
    logs = {
        "record_lake": [
            {
                "locator.red": "BBBBBB",
                "creator.sign.red": "0201GP",
                "creation_date_time": "2026-07-24T20:00:00Z",
                "element_counters.AUX": 0,
                "element_counters.OSI": 0,
                "element_counters.RM": 0,
                "element_counters.INS": 0,
            }
        ],
        "settlement_report": [],
    }
    a = UseCaseAnalyzer(use_case="scheme").analyze(
        _SPEC, logs, _analysis("BBBBBB"), knowledge_pack=_pack_with_case()
    )
    assert "INSUFFICIENT DATA" in a.verdict.summary
    assert a.brief.decisive_unknowns
    assert " ".join(a.brief.action_backbone).lower().count("pull the full") == 1


def _examined_in_full_logs():
    """Logs on which EVERY condition of `_SPEC` resolves and none of them fires.

    The settlement row is what makes the fixture honest rather than convenient: with
    `settlement_report: []` two of the three conditions read `unknown`, the verdict says so with
    `no_verdict_partial=` instead, and a subject with an open retrieval gap must NOT be the one
    proving that a re-run is pointless. Same agent, 20 minutes later — both conditions PASS.
    """
    return {
        "record_lake": [
            {
                "locator.red": "BBBBBB",
                "creator.sign.red": "0201GP",
                "creation_date_time": "2026-07-24T20:00:00Z",
                "element_counters.AUX": 0,
                "element_counters.OSI": 0,
                "element_counters.RM": 0,
                "element_counters.INS": 0,
            }
        ],
        "settlement_report": [
            {
                "recordLocator": "BBBBBB",  # so the subject filter keeps this row
                "retrieverUserSign": "0201GP",
                "transactionDateTime": "2026-07-24T20:20:00Z",
            }
        ],
    }


def test_a_subject_examined_in_full_is_not_told_to_re_run_the_retrieval():
    """Two roads reach the INSUFFICIENT label and only one of them wants a re-run.

    A ruleset may declare `no_exclusion_fired: insufficient` — "nothing fired" is not a clear in
    this procedure — and a subject then arrives at the label having been examined in FULL: every
    evaluable condition answered, nothing fired. The default next step sends that operator to
    re-pull an evidence set that was already complete, i.e. the one action guaranteed to change
    nothing, on the commonest outcome of such a procedure.

    The branch reads the verdict's own `no_verdict_reason=` note rather than re-deriving the
    condition, and the test above is the other half of the pair: it reaches the same label with a
    decisive UNKNOWN, has no such note, and must keep the re-run line. Neither test alone
    discriminates — a fix that always escalated would pass this one and break that one.
    """
    logs = _examined_in_full_logs()
    spec = dict(_SPEC)
    spec["no_exclusion_fired"] = "insufficient"
    # Nothing decisive may be unknown, or the OTHER road produces this label and the
    # assertion proves nothing about the branch under test.
    spec["conditions"] = [dict(c, decisive=c["id"] == "bare") for c in _SPEC["conditions"]]
    a = UseCaseAnalyzer(use_case="scheme").analyze(
        spec, logs, _analysis("BBBBBB"), knowledge_pack=_pack_with_case()
    )
    sv = a.verdict.subjects[0]
    assert sv.verdict_class == "insufficient"
    assert not any(c.result == "unknown" for c in sv.checks if c.decisive), "the premise"
    assert all(c.result == "pass" for c in sv.checks), (
        "THE SECOND HALF OF THE PREMISE. Examined IN FULL means nothing was left unresolved — a "
        "partially answered subject reaches the same exit and keeps the re-run line"
    )
    joined = " ".join(a.brief.action_backbone)
    assert "a re-run will return the same rows" in joined, joined
    assert "pull the full evidence set" not in joined
    assert "Escalate for human judgement" in joined
    assert "do not close as clean and do not contain" in joined, (
        "the escalation must forbid BOTH wrong closures — an unexplained subject is neither "
        "cleared nor convicted, and a next step naming only one of them invites the other"
    )


def test_a_partially_answered_no_fire_subject_keeps_the_re_run_next_step():
    """THREE roads reach this label, not two, and the third one wants the default line back.

    The no-fire exit itself splits: all evaluable conditions answered, or only some of them. The
    second still has a retrieval gap — nothing decisive is unknown, so `degraded` is False and
    the census of PASSes looks identical — and telling that operator "a re-run will return the
    same rows" removes the one remedy left. The engine separates the two at the rollup by key, so
    this branch's `no_verdict_reason=` match falls through here by construction; that
    fall-through is the assertion, and it is what makes the pair above discriminating rather than
    a rule that fires on the class.
    """
    logs = _examined_in_full_logs()
    logs["settlement_report"] = []  # two of three conditions can no longer resolve
    spec = dict(_SPEC)
    spec["no_exclusion_fired"] = "insufficient"
    spec["conditions"] = [dict(c, decisive=c["id"] == "bare") for c in _SPEC["conditions"]]
    a = UseCaseAnalyzer(use_case="scheme").analyze(
        spec, logs, _analysis("BBBBBB"), knowledge_pack=_pack_with_case()
    )
    sv = a.verdict.subjects[0]
    assert sv.verdict_class == "insufficient"
    assert not any(c.result == "unknown" for c in sv.checks if c.decisive), (
        "THE PREMISE. Nothing DECISIVE is unknown, so this is the no-fire exit and not the "
        "decisive-unknown road — the two are indistinguishable to a reader of the class alone"
    )
    assert [c.id for c in sv.checks if c.result == "unknown"] == [
        "same_agent",
        "immediate_issuance",
    ]
    joined = " ".join(a.brief.action_backbone)
    assert "pull the full evidence set" in joined, joined
    assert "a re-run will return the same rows" not in joined, joined


def test_the_examined_next_step_is_the_packs_prose_like_every_other_branch():
    """The engine owns the branch, the pack owns the words — asserted on the NEW template too,
    because a branch wired with a hard-coded string is a domain leak waiting for its first
    override, and `_ACTION_TEMPLATES` is the only seam a knowledge author has here.
    """
    logs = _examined_in_full_logs()
    spec = dict(_SPEC)
    spec["no_exclusion_fired"] = "insufficient"
    spec["conditions"] = [dict(c, decisive=c["id"] == "bare") for c in _SPEC["conditions"]]
    spec["case_builder"] = dict(
        _SPEC.get("case_builder", {}) or {},
        action_templates={"insufficient_examined": "{subject_label}: ASK THE DUTY OFFICER."},
    )
    a = UseCaseAnalyzer(use_case="scheme").analyze(
        spec, logs, _analysis("BBBBBB"), knowledge_pack=_pack_with_case()
    )
    joined = " ".join(a.brief.action_backbone)
    assert "ASK THE DUTY OFFICER." in joined, joined
    assert "a re-run will return the same rows" not in joined


def test_scheme_analyzer_indicator_driven_fraud_brief():
    """A not-bare order with >= threshold fraud indicators -> VALID FRAUD; the
    brief carries decisive_indicators and the action backbone recommends expert
    confirmation (not auto-containment)."""
    spec = dict(_SPEC)
    spec["indicator_threshold"] = 2
    spec["conditions"] = list(_SPEC["conditions"]) + [
        {
            "id": "non_agency_email",
            "label": "Non-agency email",
            "kind": "value_matches_pattern",
            "polarity": "fraud_indicator",
            "decisive": False,
            "source": "record",
            "fields": ["address.addr_detail.email.red"],
            "match_mode": "allowed",
            "patterns": ["@examplecorp\\."],
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
    ]
    logs = {
        "record_lake": [
            {
                "locator": {"red": "SUBJ01"},
                "creator": {"sign": {"red": "0303CD"}, "org_unit_id": "ORGUNIT01"},
                "creation_date_time": "2026-07-27T18:00:00Z",
                "element_counters": {
                    "AUX": 6,
                    "OSI": 0,
                    "RM": 0,
                    "INS": 0,
                },  # not bare -> exclusion FAIL
                "address": {"addr_detail": [{"email": {"red": "ORG_UNIT@EXAMPLECO.RS"}}]},
                "pricing": {
                    "payment": [{"tender": [{"data_map": [{"key": "PM", "value": "CA"}]}]}]
                },
            }
        ],
        "settlement_report": [
            {
                "recordLocator": "SUBJ01",
                "retrieverUserSign": "0303CD",
                "transactionDateTime": "2026-07-27T20:30:00Z",
                "recordType": "SALE",
            }
        ],
    }
    a = UseCaseAnalyzer(use_case="scheme").analyze(
        spec, logs, _analysis("SUBJ01"), knowledge_pack=_pack_with_case("VALID FRAUD")
    )
    assert "VALID FRAUD" in a.verdict.summary
    ind_ids = {c.id for c in a.brief.decisive_indicators}
    assert {"non_agency_email", "cash_tender"} <= ind_ids
    joined = " ".join(a.brief.action_backbone).lower()
    assert "expert must confirm" in joined
    assert "auto-void/lock/freeze" in joined

    # Regression: confident verdict on present rows must not set brief.degraded.
    # The projection guard's elif once fired on a confirmed-indicator short-circuit.
    assert a.verdict.degraded is False
    assert (
        a.brief.degraded is False
    ), "confident verdict on real rows must not be degraded"
    assert not any("no rows returned" in n.lower() for n in (a.brief.notes or []))


def test_scheme_brief_degraded_only_when_record_rows_truly_absent():
    """The 'no rows returned' note must key off row ABSENCE, not on the indicator
    short-circuit — and must still fire when the record source genuinely came back empty.
    """
    spec = dict(_SPEC)
    logs = {
        "record_lake": [],  # genuinely empty
        "settlement_report": [
            {
                "recordLocator": "SUBJ01",
                "retrieverUserSign": "0303CD",
                "transactionDateTime": "2026-07-27T20:30:00Z",
                "recordType": "SALE",
            }
        ],
    }
    a = UseCaseAnalyzer(use_case="scheme").analyze(
        spec, logs, _analysis("SUBJ01"), knowledge_pack=_pack_with_case("VALID FRAUD")
    )
    assert a.brief.degraded is True
    assert any("no rows returned" in n.lower() for n in (a.brief.notes or []))


# --- shared tools ------------------------------------------------------------


def test_asset_timeline_orders_events():
    logs = {
        "record_lake": [
            {"locator.red": "SUBJ03", "creation_date_time": "2026-07-24T19:14:00Z"}
        ],
        "settlement_report": [
            {
                "recordLocator": "SUBJ03",
                "relatedDocumentNumber": "0575033837736",
                "transactionDateTime": "2026-07-24T20:20:00Z",
                "recordType": "VOID",
            },
            {
                "recordLocator": "SUBJ03",
                "relatedDocumentNumber": "0575033837736",
                "transactionDateTime": "2026-07-24T20:19:00Z",
                "recordType": "SALE",
            },
        ],
    }
    tl = UseCaseAnalyzer.build_asset_timeline(logs, _SPEC, "SUBJ03")
    kinds = [e.event_type for e in tl]
    # Sorted by time: created (19:14) -> issued (20:19) -> voided (20:20).
    assert kinds == ["record_created", "issued", "voided"]


# Asset nested in record versions: event time is the version write time, actor is the
# issuing sign, not a retriever (which can be the investigator).
_NESTED_TIMELINE_SPEC = {
    **_SPEC,
    "asset_timeline": {
        "subject_created": {
            "in": "record",
            "event_type": "record_created",
            "entity_type": "record",
            "detail": "record created",
            "subject_id_fields": ["locator.red"],
            "timestamp_fields": ["creation_date_time"],
            "actor_fields": ["creator.sign.red"],
            "actor_scope_fields": ["creator.org_unit_id"],
        },
        "documents": {
            "in": "record",
            "path": "pricing.asset_document",
            "entity_type": "document",
            "status_fields": ["status"],
            "status_events": {"T": "issued", "V": "voided"},
            "timestamp_fields": [],
            "record_timestamp_fields": ["modification_date_time", "creation_date_time"],
            "actor_fields": ["security.creator.sign"],
            "actor_scope_fields": ["org_unit_id"],
            "asset_id_fields": ["document.provider"],
            "asset_id_suffix_fields": ["document.numbers"],
            "asset_id_separator": "-",
        }
    },
}


def test_asset_timeline_reads_nested_documents_per_version():
    doc = lambda status, sign: {  # noqa: E731
        "status": status,
        "org_unit_id": "BEGQR08AA",
        "security": {"creator": {"sign": sign}},
        "document": {"provider": "400", "numbers": ["2000000006"]},
    }
    logs = {
        "record_lake": [
            # Version 0: no document yet -> contributes only the creation event.
            {
                "locator.red": "SUBJ01",
                "creation_date_time": "2026-07-27T16:09:00Z",
                "modification_date_time": "2026-07-27T16:09:14Z",
                "version_number": "0",
            },
            # Version 6: the FIRST version carrying an issued document -> issued.
            {
                "locator.red": "SUBJ01",
                "creation_date_time": "2026-07-27T16:09:00Z",
                "modification_date_time": "2026-07-27T16:11:34Z",
                "version_number": "6",
                "pricing": {"asset_document": [doc("T", "0303CD")]},
            },
            # Version 13: the same document, now voided by a different sign.
            {
                "locator.red": "SUBJ01",
                "creation_date_time": "2026-07-27T16:09:00Z",
                "modification_date_time": "2026-07-27T18:15:53Z",
                "version_number": "13",
                "pricing": {"asset_document": [doc("V", "0404EF")]},
            },
            # Version 15: status I (intermediate) is deliberately unmapped -> no event.
            {
                "locator.red": "SUBJ01",
                "creation_date_time": "2026-07-27T16:09:00Z",
                "modification_date_time": "2026-07-27T18:32:03Z",
                "version_number": "15",
                "pricing": {"asset_document": [doc("I", "0303CD")]},
            },
        ]
    }
    tl = UseCaseAnalyzer.build_asset_timeline(logs, _NESTED_TIMELINE_SPEC, "SUBJ01")
    assert [e.event_type for e in tl] == ["record_created", "issued", "voided"]
    issued = tl[1]
    # The number is stored SPLIT and rejoined the same way the scope sweep joins it, so
    # both views of one document carry the same id.
    assert issued.entity_value == "400-2000000006"
    # The event time is the ENVELOPE's write time (the document's own date is day-granular).
    assert issued.timestamp == "2026-07-27T16:11:34Z"
    # The ISSUING sign, not a retriever.
    assert issued.actor == "0303CD @ BEGQR08AA"
    assert tl[2].actor.startswith("0404EF")


# --- §3.4 scope discovery ----------------------------------------------------

# A pack-declared actor-scoped sweep: same shape the examplecorp pack uses, with the document
# number stored SPLIT (provider + serial) as the real lake does.
_SCOPE_SPEC = {
    **_SPEC,
    "sources": {**_SPEC["sources"], "scope_sweep": "record_document_scope_sweep"},
    "scope_discovery": {
        "source": "scope_sweep",
        "asset_id_separator": "-",
        "fields": {
            "subject": "locator.red",
            "asset_id": "pricing.asset_document.document.provider",
            "asset_id_suffix": "pricing.asset_document.document.numbers",
            "amount": "pricing.asset_document.amount",
            "currency": "pricing.asset_document.currency",
            "status": "pricing.asset_document.status",
            "actor": "creator.sign.red",
            "actor_scope": "creator.org_unit_id",
            "timestamp": "creation_date_time",
            # The record lake is ENVELOPE-versioned: version 0 is the creation and every
            # modification appends a new one, so the highest version is the CURRENT record.
            "version": [
                "document_version_number",
                "version_number",
                "pricing.asset_document.version_number",
            ],
            # The document's own ISSUE date — what bounds §3.4 scope, distinct from the record's
            # creation date. The sweep is actor-scoped over a multi-day window, so it also
            # returns that sign's ordinary business.
            "event_date": ["document_issue_date", "pricing.asset_document.date"],
        },
        # No `event_window_days`, matching the real pack: the boundary is DERIVED from the
        # alerted subjects' own documents, not configured.
    },
}


class _Window:
    """Stand-in for the pipeline's ``EventWindow`` (duck-typed, per the dual-import trap)."""

    def __init__(self, start, end):
        self.start, self.end = start, end


def _sweep_row(
    record,
    provider,
    serial,
    amount,
    status,
    date="2026-07-27T18:00:00Z",
    version=0,
    doc_date="2026-07-27",
):
    return {
        "locator": {"red": record},
        "creation_date_time": date,
        "version_number": version,
        "creator": {"sign": {"red": "0303CDSU"}, "org_unit_id": "ORGUNIT01"},
        "pricing": {
            "asset_document": [
                {
                    "document": {"provider": provider, "numbers": [serial]},
                    "amount": amount,
                    "currency": "RSD",
                    "status": status,
                    "version_number": version,
                    "date": doc_date,
                }
            ]
        },
    }


_LINK_SPEC = {
    **_SPEC,
    "sources": {**_SPEC["sources"], "links": "record_lake"},
    "subject_links": {
        "source": "links",
        "array": "elements.related",
        "subject_field": "locator.red",
        "kind": "split",
        "roles": {"SP": "parent", "SC": "child"},
        "fields": {
            "related_subject": "related_locator",
            "role": "type_code",
            "element_id": "element_number",
            "actor": "signed_by",
        },
    },
}


def _link_row(subject, related, element_no, code="SP"):
    return {
        "locator": {"red": subject},
        "elements": {
            "related": [
                {
                    "related_locator": related,
                    "type_code": code,
                    "element_number": str(element_no),
                    "signed_by": "0303CDSU",
                }
            ]
        },
    }


def test_the_derivation_list_is_not_cut_so_a_late_link_still_corrects_the_scope():
    """A cap on the derivation links manufactures the finding it was cutting.

    `build_subject_links` stopped at 40; `_reclassify_derived` reads the list to move derived
    subjects out of "under-reported scope", so the 41st link left an actor-created record
    standing there. The cap was the engine's own constant, undeclared and silent.

    This builds past any fixed bound and asserts on the last link: the far end is where a cap
    is invisible. The reclassification count is also asserted: a fix returning every link
    but reclassifying none reads the same from the scope line's first clause.
    """
    n = 60
    logs = {
        "record_document_scope_sweep": [
            _sweep_row("SUBJ01", "400", "2000000006", "89211", "T")
        ]
        + [
            _sweep_row(f"DERV{i:03d}", "400", f"20000010{i:02d}", "1000", "T")
            for i in range(n)
        ],
        "record_lake": [
            _link_row("SUBJ01", f"DERV{i:03d}", 1000 + i) for i in range(n)
        ],
    }
    spec = {**_LINK_SPEC, **{"scope_discovery": _SCOPE_SPEC["scope_discovery"]}}
    spec["sources"] = {**_LINK_SPEC["sources"], **_SCOPE_SPEC["sources"]}

    links = UseCaseAnalyzer.build_subject_links(logs, spec)
    assert len(links) == n
    assert links[-1].related_subject == f"DERV{n - 1:03d}"
    assert links[-1].role == "parent"

    _, extra, status = UseCaseAnalyzer.build_scope_discovery(
        logs, spec, known_subjects=["SUBJ01"]
    )
    assert len(extra) == n
    assets, extra, status = UseCaseAnalyzer._reclassify_derived(
        links, [], extra, status
    )
    # Every one of them is the same episode under a second identifier — none is new scope.
    assert extra == []
    assert f"{n} of those" in status and "NOT new scope" in status


def test_an_unmapped_role_code_is_never_reclassified_as_a_derivation():
    """The bound on the fix above: only a MAPPED role licenses the correction.

    `roles` maps the backend's own codes onto parent/child and an unknown code yields
    `role='unknown'`, because getting the direction backwards would make the alerted record
    the derived one. Removing the cap must not turn that abstention into a reclassification —
    an unmapped code is a link the engine can see and cannot read, and the honest outcome is
    that the subject stays in the work-list.
    """
    logs = {"record_lake": [_link_row("SUBJ01", "DERV000", 1000, code="ZZ")]}
    links = UseCaseAnalyzer.build_subject_links(logs, _LINK_SPEC)
    assert len(links) == 1 and links[0].role == "unknown"
    _, extra, status = UseCaseAnalyzer._reclassify_derived(
        links, [], ["DERV000"], "ran, 1 NOT named in the alert"
    )
    assert extra == ["DERV000"]
    assert "NOT new scope" not in status


def test_scope_discovery_finds_subjects_the_alert_never_named():
    """The §3.4 gap from IR10000001: the sweep must surface records outside the alert.

    The expert found 3 records / 6 documents; AFIR reported 2 and missed SUBJ09, whose document
    was still ACTIVE. The actor-scoped sweep is what closes that.
    """
    logs = {
        "record_document_scope_sweep": [
            _sweep_row("SUBJ01", "400", "2000000006", "89211", "T"),  # already known
            _sweep_row("YEFGHJ", "100", "2000000005", "121964", "T"),  # NEW
            # Created the DAY BEFORE the alert — the case for a multi-day window.
            _sweep_row(
                "SUBJ09", "200", "2000000004", "11755", "T", date="2026-07-26T09:00:00Z"
            ),  # NEW
        ]
    }
    assets, extra, status = UseCaseAnalyzer.build_scope_discovery(
        logs, _SCOPE_SPEC, known_subjects=["SUBJ01"]
    )
    # Document numbers are reconstructed from the SPLIT provider + serial columns.
    assert {a.asset_id for a in assets} == {
        "400-2000000006",
        "100-2000000005",
        "200-2000000004",
    }
    assert extra == ["YEFGHJ", "SUBJ09"]
    assert [a.known for a in assets] == [True, False, False]
    assert status.startswith(
        "ran, 3 asset(s) in the incident window across 3 subject(s)"
    )
    assert "2 NOT named in the alert" in status
    # Amount/status ride along so the report can total the exposure and spot live documents.
    assert {a.amount for a in assets} == {"89211", "121964", "11755"}


def test_a_duty_code_suffixed_subject_is_not_reported_as_unnamed():
    """The sweep must read a WIDER SURFACE FORM of the alerted identifier as the subject.

    A backend can store an identifier at more than one width: the alert names the base
    identifier and the rows carry it with a role/duty suffix appended. Measured live on
    IR 30556527 — the sweep returned 500 rows on the alerted actor under a two-character
    suffix and the scope line reported 100% of its subjects "NOT named in the alert", while
    ``evaluate_verdict`` had already adjudicated those same rows AS the subject through the
    engine's own prefix-tolerant ``_identifiers_match``. One report, both readings.

    So the reading side asks the question the verdict side already asks. The last block is
    the required negative: tolerance is a PREFIX relation, not a shared beginning.
    """
    logs = {
        "record_document_scope_sweep": [
            # The alert's own identifier, verbatim.
            _sweep_row("SUBJ01", "400", "2000000006", "89211", "T"),
            # The SAME identity, stored one suffix wider — not a second subject.
            _sweep_row("SUBJ01GS", "400", "2000000007", "12000", "T"),
        ]
    }
    assets, extra, status = UseCaseAnalyzer.build_scope_discovery(
        logs, _SCOPE_SPEC, known_subjects=["SUBJ01"]
    )
    assert extra == []
    assert all(a.known for a in assets)
    assert "0 NOT named in the alert" in status

    # A genuinely different identifier stays a stranger even though it shares a prefix
    # boundary — the ">=4 shared characters AND a true prefix" contract, not "looks alike".
    other = {
        "record_document_scope_sweep": [
            _sweep_row("SUBJ01", "400", "2000000006", "89211", "T"),
            _sweep_row("SUBJ02", "100", "2000000005", "121964", "T"),
        ]
    }
    _, extra2, status2 = UseCaseAnalyzer.build_scope_discovery(
        other, _SCOPE_SPEC, known_subjects=["SUBJ01"]
    )
    assert extra2 == ["SUBJ02"]
    assert "1 NOT named in the alert" in status2


def test_scope_discovery_reports_a_truncated_sweep_as_a_floor():
    """A sweep cut off at the backend's row cap must not read as an exhaustive one.

    Measured live: 500 rows returned against 3509 available, and the status line said
    `ran, N asset(s) across M subject(s)` — the exact wording an exhaustive sweep
    produces. The sweep is the only step that can widen scope beyond the alert, so a
    ceiling reported as a total understates the blast radius while looking complete.
    """
    logs = {
        "record_document_scope_sweep": [
            _sweep_row("SUBJ01", "400", "2000000006", "89211", "T"),
            _sweep_row("YEFGHJ", "100", "2000000005", "121964", "T"),
        ]
    }
    # The retriever returned exactly its cap, so there may be more behind it.
    _, _, status = UseCaseAnalyzer.build_scope_discovery(
        logs,
        _SCOPE_SPEC,
        known_subjects=["SUBJ01"],
        row_caps={"record_document_scope_sweep": 2},
    )
    assert status.startswith("ran"), status
    # The counts must be marked as floors, and the cap named so it is checkable.
    assert "at least" in status.lower() or "≥" in status, status
    assert "2" in status and "truncated" in status.lower(), status

    # A sweep BELOW its cap is exhaustive and must be unchanged.
    _, _, exhaustive = UseCaseAnalyzer.build_scope_discovery(
        logs,
        _SCOPE_SPEC,
        known_subjects=["SUBJ01"],
        row_caps={"record_document_scope_sweep": 500},
    )
    assert "truncated" not in exhaustive.lower()
    assert "≥" not in exhaustive


def test_scope_discovery_accepts_the_sql_gens_flattened_aliases():
    """The LLM-generated sweep SQL FLATTENS the exploded document into per-column aliases
    (`document_number` already concatenated, `document_amount`, ...) rather than returning the
    nested struct. Live-verified 2026-07-29: mapping only the struct paths resolved every
    document leaf to EMPTY — the sweep found the extra records but reported them with no document
    number, amount or status. Each pack field therefore lists CANDIDATE paths.
    """
    spec = {
        **_SCOPE_SPEC,
        "scope_discovery": {
            "source": "scope_sweep",
            "asset_id_separator": "-",
            "fields": {
                "subject": ["locator.red", "locator_red"],
                "asset_id": ["document_number", "pricing.asset_document.document.provider"],
                "asset_id_suffix": ["pricing.asset_document.document.numbers"],
                "amount": ["document_amount", "pricing.asset_document.amount"],
                "status": ["document_status", "pricing.asset_document.status"],
                "actor": ["document_issuing_sign", "creator.sign.red"],
            },
        },
    }
    flat = [
        {
            "locator_red": "SUBJ01",
            "document_number": "400-2000000006",
            "document_amount": "89211",
            "document_status": "T",
            "document_issuing_sign": "0303CDSU",
        },
        {
            "locator_red": "SUBJ09",
            "document_number": "200-2000000004",
            "document_amount": "11755",
            "document_status": "T",
            "document_issuing_sign": "0303CDSU",
        },
    ]
    assets, extra, status = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": flat}, spec, known_subjects=["SUBJ01"]
    )
    assert extra == ["SUBJ09"]
    # The already-concatenated alias must NOT be re-joined with the separator.
    assert [a.asset_id for a in assets] == ["400-2000000006", "200-2000000004"]
    assert [a.amount for a in assets] == ["89211", "11755"]
    assert all(a.status == "T" for a in assets)
    assert status.startswith("ran, 2 asset(s)")

    # The nested struct shape still resolves through the same mapping (2nd candidate).
    assets, extra, _ = UseCaseAnalyzer.build_scope_discovery(
        {
            "record_document_scope_sweep": [
                _sweep_row("YEFGHJ", "100", "2000000005", "121964", "T")
            ]
        },
        spec,
        known_subjects=["SUBJ01"],
    )
    assert extra == ["YEFGHJ"]
    assert assets[0].asset_id == "100-2000000005" and assets[0].amount == "121964"


def test_scope_discovery_collapses_one_asset_per_lifecycle_not_per_row():
    """The lake returns each document document once PER ENVELOPE, and nulls the amount on
    superseded ones. Keying assets on state made one document three "impacted assets" and any
    amount total a 3x overcount; the versions must merge into one entry that still carries
    the amount from whichever row had it, reporting the HIGHEST version as current.
    """
    rows = [
        # Version 0: the document is only issued-pending, no amount yet.
        _sweep_row("SUBJ01", "400", "2000000006", "", "I", version=0),
        _sweep_row("SUBJ01", "400", "2000000006", "89211", "T", version=1),
        _sweep_row("SUBJ01", "400", "2000000006", "89211", "V", version=2),
    ]
    assets, _, status = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": rows}, _SCOPE_SPEC, known_subjects=["SUBJ01"]
    )
    assert len(assets) == 1
    assert assets[0].asset_id == "400-2000000006"
    assert assets[0].amount == "89211"  # recovered from a later version
    # Version 2 is the current version; the earlier states survive as history.
    assert assets[0].status == "V (current; was I -> T)"
    assert status.startswith(
        "ran, 1 asset(s) in the incident window across 1 subject(s)"
    )


def test_scope_discovery_reports_the_highest_version_as_current():
    """The version model: version 0 is the record's creation and every modification appends
    a new one, so the LAST version is the current version of the record. Rows come back in no
    useful order, so taking the first/last row's state would report a document that was
    voided and then REISSUED as still voided — i.e. a LIVE fraudulent document read as
    already contained.
    """
    rows = [
        _sweep_row(
            "SUBJ01", "400", "2000000006", "89211", "V", version=4
        ),  # voided...
        _sweep_row(
            "SUBJ01", "400", "2000000006", "89211", "T", version=7
        ),  # ...reissued
        _sweep_row("SUBJ01", "400", "2000000006", "", "I", version=2),
    ]
    assets, _, _ = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": rows}, _SCOPE_SPEC, known_subjects=["SUBJ01"]
    )
    assert len(assets) == 1
    # T (live) is current even though V arrived first in the result set.
    assert assets[0].status.startswith("T (current;")
    assert assets[0].status == "T (current; was I -> V)"


def test_scope_discovery_attributes_the_asset_to_the_actor_of_its_CURRENT_version():
    """Descriptive fields must come from the LATEST version, not the first row seen.

    Live shape from document 100-2000000005 (IR10000001): an intermediate document version
    was issued by 0404EF @ CCC1D04GH while the CURRENT version is 0303CD @ ORGUNIT01.
    First-non-empty reported the foreign org_unit, misattributing the act — the report pairs
    `actor` with the current `status`, so they must describe the same version.
    """
    rows = [
        _sweep_row("YEFGHJ", "100", "2000000005", "121964", "T", version=8),
        _sweep_row("YEFGHJ", "100", "2000000005", "121964", "T", version=12),
        _sweep_row("YEFGHJ", "100", "2000000005", "", "I", version=18),
    ]
    for r in rows[:2]:  # the superseded versions: a different org_unit
        r["creator"] = {"sign": {"red": "0404EF"}, "org_unit_id": "CCC1D04GH"}
    assets, _, _ = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": rows}, _SCOPE_SPEC, known_subjects=["YEFGHJ"]
    )
    assert len(assets) == 1
    assert (
        assets[0].actor == "0303CDSU @ ORGUNIT01"
    )  # version 18 = the current version
    assert "CCC1D04GH" not in assets[0].actor
    assert assets[0].status == "I (current; was T)"
    # ...but an EMPTY value never overwrites a populated one: the lake nulls the amount on
    # pending/superseded versions, so the newest row that HAS a figure still wins.
    assert assets[0].amount == "121964"


def test_scope_discovery_refuses_to_claim_currency_without_a_version_field():
    """No ``version`` declared -> the states are UNORDERED observations. Guessing which one
    is current would be a fabrication in exactly the direction that hurts (claiming a
    document is voided), so the analyzer reports them all and says the order is unknown.
    """
    spec = {
        **_SCOPE_SPEC,
        "scope_discovery": {
            **_SCOPE_SPEC["scope_discovery"],
            "fields": {
                k: v
                for k, v in _SCOPE_SPEC["scope_discovery"]["fields"].items()
                if k != "version"
            },
        },
    }
    rows = [
        _sweep_row("SUBJ01", "400", "2000000006", "89211", "V", version=4),
        _sweep_row("SUBJ01", "400", "2000000006", "89211", "T", version=7),
    ]
    assets, _, _ = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": rows}, spec, known_subjects=["SUBJ01"]
    )
    assert len(assets) == 1
    assert assets[0].status == "V / T (order unknown)"
    assert "current" not in assets[0].status


def test_scope_discovery_separates_incident_scope_from_the_actors_adjacent_business():
    """The §6 boundary from IR10000001: scope is the document's ISSUE date, not the sweep's.

    The sweep must span more than the alert day (SUBJ09's record was created 26JUL for a 27JUL
    alert), so it necessarily also returns the sign's ordinary work. The expert reviewed the
    Settlement Report for 0303CD ON 27JUL and found 6 documents; probing the live lake with
    ``td.date = '2026-07-27'`` over a 25-28JUL creation window returns exactly that set.
    Reporting the 25JUL and 28JUL documents as extra fraud scope overstated the case 5x.
    """
    logs = {
        "record_document_scope_sweep": [
            _sweep_row("SUBJ01", "400", "2000000006", "89211", "T"),  # known, alert day
            # Created the day BEFORE the alert but ISSUED on the alert day -> in scope.
            _sweep_row(
                "SUBJ09",
                "200",
                "2000000004",
                "11755",
                "T",
                date="2026-07-26T09:00:00Z",
                doc_date="2026-07-27",
            ),
            # Same sign, issued two days earlier: this agent's normal business.
            _sweep_row(
                "XYZABC",
                "200",
                "2000000002",
                "40100",
                "T",
                date="2026-07-25T08:00:00Z",
                doc_date="2026-07-25",
            ),
        ]
    }
    assets, extra, status = UseCaseAnalyzer.build_scope_discovery(
        logs,
        _SCOPE_SPEC,
        known_subjects=["SUBJ01"],
        event_window=_Window("2026-07-27T00:00:00Z", "2026-07-27T23:59:59Z"),
    )
    in_window = {a.asset_id for a in assets if a.in_window}
    assert in_window == {"400-2000000006", "200-2000000004"}
    # Nothing is DROPPED — the operator still sees the neighbouring activity, flagged.
    assert {a.asset_id for a in assets if not a.in_window} == {"200-2000000002"}
    assert assets[-1].asset_id == "200-2000000002"  # out-of-window entries sort last
    # ...but the containment work-list holds only real incident scope.
    assert extra == ["SUBJ09"]
    assert "2 asset(s) in the incident window across 2 subject(s)" in status
    assert "1 NOT named in the alert" in status
    assert "1 asset(s) across 1 subject(s) OUTSIDE the incident window" in status
    # The issue date rides along so the report can show WHY something is out of scope.
    assert {a.event_date for a in assets} == {"2026-07-27", "2026-07-25"}


def test_an_undated_version_cannot_drag_an_old_asset_back_into_scope():
    """An UNDATED row is a third state, not "in window".

    Live bug (IR10000001 rerun): the lake returns a order's pre-issuance versions with a
    NULL document date, and document 200-2000000002 — issued 16JUL — rode 15 such versions of
    its alert-day record back into a 27JUL incident. Undated evidence must only decide when
    there is no dated evidence to decide with.
    """
    rows = [
        _sweep_row(
            "SUBJ09",
            "200",
            "2000000002",
            "40100",
            "T",
            version=1,
            doc_date="2026-07-16",
        ),
        _sweep_row(
            "SUBJ09",
            "200",
            "2000000004",
            "11755",
            "T",
            version=9,
            doc_date="2026-07-27",
        ),
    ]
    # ...plus the same record's undated versions, on BOTH documents.
    for env in (2, 3, 4):
        for serial in ("2000000002", "2000000004"):
            r = _sweep_row("SUBJ09", "200", serial, "", "I", version=env)
            r["pricing"]["asset_document"][0].pop("date")
            rows.append(r)
    assets, extra, status = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": rows},
        _SCOPE_SPEC,
        known_subjects=[],
        event_window=_Window("2026-07-27T00:00:00Z", "2026-07-27T23:59:59Z"),
    )
    by_id = {a.asset_id: a for a in assets}
    assert by_id["200-2000000002"].in_window is False  # issued 16JUL, stays out
    assert by_id["200-2000000004"].in_window is True
    # The subject still makes the work-list — it DOES hold an in-window document.
    assert extra == ["SUBJ09"]
    assert "1 asset(s) in the incident window" in status


def test_out_of_window_subjects_are_counted_only_when_nothing_of_theirs_is_in_scope():
    """The companion live bug: every out-of-window subject also had undated versions, so
    each read as in-window and the status said "0 subject(s) OUTSIDE" while listing 6
    out-of-window assets. A subject is OUT only when none of its assets is in scope."""
    rows = [
        # Wholly out-of-window subject, with undated versions of its own.
        _sweep_row("SUBJ12", "200", "2000000003", "40100", "T", doc_date="2026-07-25"),
        # Mixed subject: one asset in, one out -> counts as IN, not as an extra out-subject.
        _sweep_row("SUBJ09", "200", "2000000004", "11755", "T", doc_date="2026-07-27"),
        _sweep_row("SUBJ09", "200", "2000000008", "9000", "T", doc_date="2026-07-28"),
    ]
    bare = _sweep_row("SUBJ12", "200", "2000000003", "", "I", version=3)
    bare["pricing"]["asset_document"][0].pop("date")
    rows.append(bare)
    documentless = _sweep_row("SUBJ10", "", "", "", "")  # a order, never issued
    documentless["pricing"] = {"asset_document": []}
    rows.append(documentless)

    assets, extra, status = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": rows},
        _SCOPE_SPEC,
        known_subjects=[],
        event_window=_Window("2026-07-27T00:00:00Z", "2026-07-27T23:59:59Z"),
    )
    assert {a.asset_id for a in assets if not a.in_window} == {
        "200-2000000003",
        "200-2000000008",
    }
    # 2 out-of-window ASSETS but only ONE wholly-out subject: SUBJ09 has an in-scope document.
    assert "2 asset(s) across 1 subject(s) OUTSIDE the incident window" in status
    # A order with no document at all is still worth reviewing, but it is not exposure —
    # the count says which is which instead of presenting 2 as the impacted-record total.
    assert extra == ["SUBJ09", "SUBJ10"]  # document-holding subject sorts first
    assert "1 holding an in-window document, 1 with none" in status


def test_scope_discovery_keeps_an_asset_in_scope_if_any_version_is_in_window():
    """An asset issued during the incident stays in scope even if a later version carries a
    different date — the fraud happened, whatever was done to the document afterwards.
    """
    rows = [
        _sweep_row(
            "YEFGHJ",
            "100",
            "2000000005",
            "121964",
            "T",
            version=4,
            doc_date="2026-07-27",
        ),
        _sweep_row(
            "YEFGHJ",
            "100",
            "2000000005",
            "121964",
            "V",
            version=9,
            doc_date="2026-07-29",
        ),
    ]
    assets, extra, _ = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": rows},
        _SCOPE_SPEC,
        known_subjects=[],
        event_window=_Window("2026-07-27T00:00:00Z", "2026-07-27T23:59:59Z"),
    )
    assert len(assets) == 1 and assets[0].in_window is True
    assert extra == ["YEFGHJ"]


def test_scope_discovery_never_narrows_scope_on_a_boundary_it_cannot_compute():
    """No ``event_date`` field, or no dates anywhere -> everything stays in scope.

    Silently narrowing on an uncomputable boundary is the dangerous direction: it would drop
    a live fraudulent document out of the containment list.
    """
    rows = [
        _sweep_row("SUBJ01", "400", "2000000006", "89211", "T"),
        _sweep_row("XYZABC", "200", "2000000002", "40100", "T", doc_date="2026-07-25"),
    ]
    # (a) a window, but the pack names no event_date field.
    spec = {
        **_SCOPE_SPEC,
        "scope_discovery": {
            **_SCOPE_SPEC["scope_discovery"],
            "fields": {
                k: v
                for k, v in _SCOPE_SPEC["scope_discovery"]["fields"].items()
                if k != "event_date"
            },
        },
    }
    assets, extra, _ = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": rows},
        spec,
        known_subjects=["SUBJ01"],
        event_window=_Window("2026-07-27T00:00:00Z", "2026-07-27T23:59:59Z"),
    )
    assert all(a.in_window for a in assets) and extra == ["XYZABC"]

    # (b) the field declared, but no row carries a value for it AND no window was passed —
    # there is genuinely nothing to derive a boundary from.
    undated = []
    for record, serial in (("SUBJ01", "2000000006"), ("XYZABC", "2000000002")):
        r = _sweep_row(record, "200", serial, "40100", "T")
        r["pricing"]["asset_document"][0].pop("date")
        undated.append(r)
    assets, extra, status = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": undated}, _SCOPE_SPEC, known_subjects=["SUBJ01"]
    )
    assert all(a.in_window for a in assets) and extra == ["XYZABC"]
    assert "OUTSIDE the incident window" not in status

    # (c) a window and dates, but THIS row carries none — unknown never narrows.
    bare = _sweep_row("XYZABC", "200", "2000000002", "40100", "T")
    bare["pricing"]["asset_document"][0].pop("date")
    assets, _, _ = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": [bare]},
        _SCOPE_SPEC,
        known_subjects=[],
        event_window=_Window("2026-07-27T00:00:00Z", "2026-07-27T23:59:59Z"),
    )
    assert assets[0].in_window is True


def test_scope_discovery_derives_its_window_from_the_alerted_subjects_own_documents():
    """The boundary is EVIDENCE-DERIVED, not a configured tolerance.

    An analyst reads an episode's duration off the documents in front of them; so does the
    engine. The span of the ALERTED subjects' own issue dates IS the incident, because those
    documents are what the alert is about. A static ``event_window_days`` was right only for
    the single-day case it was tuned on: set to 0 it writes off the later days of a
    genuinely multi-day fraud as the actor's ordinary business.
    """
    # A fraud worked across three days: the alerted record holds documents on 25/26/27 JUL.
    # A same-sign document on 26JUL is INSIDE that span and must be scope; one on 20JUL is not.
    rows = [
        _sweep_row("SUBJ01", "400", "2000000006", "89211", "T", doc_date="2026-07-25"),
        _sweep_row("SUBJ01", "400", "2000000007", "89211", "T", doc_date="2026-07-27"),
        _sweep_row("YEFGHJ", "100", "2000000005", "121964", "T", doc_date="2026-07-26"),
        _sweep_row("XYZABC", "200", "2000000002", "40100", "T", doc_date="2026-07-20"),
    ]
    assets, extra, status = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": rows},
        _SCOPE_SPEC,
        known_subjects=["SUBJ01"],
        # The alert names one instant; the documents are what widen it.
        event_window=_Window("2026-07-27T16:45:00Z", "2026-07-27T16:45:00Z"),
    )
    by_id = {a.asset_id: a for a in assets}
    assert by_id["100-2000000005"].in_window is True  # 26JUL, inside the derived span
    assert by_id["200-2000000002"].in_window is False  # 20JUL, the actor's own business
    assert extra == ["YEFGHJ"]
    assert "3 day(s)" in status and "2026-07-25 → 2026-07-27" in status

    # A quiet day inside the episode must not split it: requiring a contiguous run writes
    # off the far side of a fraud that paused for a day.
    paused = [
        _sweep_row("SUBJ01", "400", "2000000006", "89211", "T", doc_date="2026-07-25"),
        _sweep_row("SUBJ01", "400", "2000000007", "89211", "T", doc_date="2026-07-27"),
        _sweep_row("YEFGHJ", "100", "2000000005", "121964", "T", doc_date="2026-07-26"),
    ]
    assets, extra, status = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": paused},
        _SCOPE_SPEC,
        known_subjects=["SUBJ01"],
        event_window=_Window("2026-07-27T16:45:00Z", "2026-07-27T16:45:00Z"),
    )
    assert all(a.in_window for a in assets) and extra == ["YEFGHJ"]
    assert "3 day(s)" in status

    # The SAME code, a single-day alert: the span collapses to that day with no pack edit.
    single = [
        _sweep_row("SUBJ01", "400", "2000000006", "89211", "T", doc_date="2026-07-27"),
        _sweep_row("YEFGHJ", "100", "2000000005", "121964", "T", doc_date="2026-07-26"),
    ]
    assets, extra, status = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": single},
        _SCOPE_SPEC,
        known_subjects=["SUBJ01"],
        event_window=_Window("2026-07-27T16:45:00Z", "2026-07-27T16:45:00Z"),
    )
    assert {a.asset_id for a in assets if not a.in_window} == {"100-2000000005"}
    assert extra == [] and "1 day(s)" in status


def test_scope_discovery_does_not_stretch_its_window_to_merely_adjacent_activity():
    """The derived span must not creep onto the actor's NEXT day of ordinary work.

    A "the gap is only one day, so it's probably continuation" rule reads as smart and is
    the reason this test exists: on IR10000001 it pulls in the 28JUL document the expert
    excluded. Adjacent activity is surfaced and labelled, and the operator decides.
    """
    rows = [
        _sweep_row("SUBJ01", "400", "2000000006", "89211", "T", doc_date="2026-07-27"),
        _sweep_row("SUBJ09", "200", "2000000008", "9000", "T", doc_date="2026-07-28"),
    ]
    assets, extra, _ = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": rows},
        _SCOPE_SPEC,
        known_subjects=["SUBJ01"],
        event_window=_Window("2026-07-27T16:45:00Z", "2026-07-27T16:45:00Z"),
    )
    assert {a.asset_id for a in assets if not a.in_window} == {"200-2000000008"}
    assert extra == []


def test_scope_discovery_still_honours_an_explicit_pack_tolerance():
    """``event_window_days`` remains an escape hatch: it PADS the derived span.

    Packs normally omit it (the examplecorp SCHEME ruleset does), but a domain whose episode is
    known to spill past its own documents must be able to say so without a code change.
    """
    rows = [
        _sweep_row("SUBJ01", "400", "2000000006", "89211", "T", doc_date="2026-07-27"),
        _sweep_row("SUBJ09", "200", "2000000008", "9000", "T", doc_date="2026-07-28"),
    ]
    padded = {
        **_SCOPE_SPEC,
        "scope_discovery": {**_SCOPE_SPEC["scope_discovery"], "event_window_days": 1},
    }
    assets, extra, _ = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": rows},
        padded,
        known_subjects=["SUBJ01"],
        event_window=_Window("2026-07-27T16:45:00Z", "2026-07-27T16:45:00Z"),
    )
    assert all(a.in_window for a in assets) and extra == ["SUBJ09"]


def test_scope_discovery_windows_on_the_calendar_day_not_the_instant():
    """A bare issue date vs a mid-day alert timestamp: comparing instants would read a
    document issued at 18:15 on the alert day as PRECEDING a 16:45 alert, and (on a host east
    of UTC) would floor a naive bare date onto the previous day."""
    rows = [
        _sweep_row("SUBJ01", "400", "2000000006", "89211", "T", doc_date="2026-07-27"),
        _sweep_row(
            "YK7LM5", "400", "2000000007", "89211", "T", doc_date="2026-07-27T23:45:00Z"
        ),
    ]
    assets, _, status = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": rows},
        _SCOPE_SPEC,
        known_subjects=["SUBJ01"],
        # The alert fired mid-afternoon; both documents are same-day and in scope.
        event_window=_Window("2026-07-27T16:45:00Z", "2026-07-27T16:45:00Z"),
    )
    assert all(a.in_window for a in assets)
    assert "2 asset(s) in the incident window" in status


def test_scope_discovery_counts_asset_less_subjects_but_not_as_assets():
    """The sweep uses an OUTER explode on purpose: a record the actor created but never
    issued is still in scope as a order. It must count as a SUBJECT (and be reported)
    while never inflating the asset list."""
    bare = _sweep_row("SUBJ10", "", "", "", "")
    bare["pricing"] = {"asset_document": []}
    rows = [_sweep_row("SUBJ01", "400", "2000000006", "89211", "T"), bare]
    assets, extra, status = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": rows}, _SCOPE_SPEC, known_subjects=["SUBJ01"]
    )
    assert [a.asset_id for a in assets] == ["400-2000000006"]
    assert extra == ["SUBJ10"]
    assert "1 asset(s) in the incident window across 1 subject(s)" in status
    assert "1 NOT named in the alert (1 with no document issued)" in status
    assert "1 row(s) with no asset" in status


def test_scope_discovery_distinguishes_not_run_from_found_nothing():
    """An unrun sweep must NEVER read as a clean one — the whole point of scope_status."""
    # Declared but the source returned no rows: a real gap.
    assets, extra, status = UseCaseAnalyzer.build_scope_discovery(
        {"record_document_scope_sweep": []}, _SCOPE_SPEC, known_subjects=["SUBJ01"]
    )
    assert (assets, extra) == ([], [])
    assert status.startswith("not attempted")
    assert "UNKNOWN, not clean" in status

    # Ran and genuinely found only the known subject: bounded scope.
    assets, extra, status = UseCaseAnalyzer.build_scope_discovery(
        {
            "record_document_scope_sweep": [
                _sweep_row("SUBJ01", "400", "2000000006", "89211", "T")
            ]
        },
        _SCOPE_SPEC,
        known_subjects=["SUBJ01"],
    )
    assert extra == [] and status.startswith("ran, 1 asset(s)")

    # No sweep DECLARED (a use case with no §3.4 step) -> empty status, not a gap.
    assert UseCaseAnalyzer.build_scope_discovery({}, _SPEC)[2] == ""


def test_scope_discovery_drives_backbone_and_degrades_unverified_scope():
    """The brief must name the extra records in the next steps, and a sweep that did not run
    must mark the case degraded — reporting a bounded scope you never verified is the
    failure mode that let a live fraudulent document go unactioned."""
    base_logs = {
        "record_lake": [
            {
                "locator": {"red": "SUBJ01"},
                "creator": {"sign": {"red": "0303CD"}, "org_unit_id": "ORGUNIT01"},
                "creation_date_time": "2026-07-27T18:00:00Z",
                "element_counters": {"AUX": 0, "OSI": 0, "RM": 0, "INS": 0},
            }
        ],
        "settlement_report": [
            {
                "recordLocator": "SUBJ01",
                "retrieverUserSign": "0303CD",
                "transactionDateTime": "2026-07-27T18:30:00Z",
                "recordType": "SALE",
            }
        ],
    }
    spec = dict(_SCOPE_SPEC)
    spec["routes"] = [["BEG", "DOH"]]

    # (a) Sweep ran and widened the scope.
    logs = {
        **base_logs,
        "record_document_scope_sweep": [
            _sweep_row("SUBJ01", "400", "2000000006", "89211", "T"),
            _sweep_row("SUBJ09", "200", "2000000004", "11755", "T"),
        ],
    }
    a = UseCaseAnalyzer(use_case="scheme").analyze(spec, logs, _analysis("SUBJ01"))
    assert a.brief.additional_subjects == ["SUBJ09"]
    assert "SUBJ09" in " ".join(a.brief.action_backbone)
    assert any("under-reported the scope" in n for n in a.brief.notes)

    # (b) Sweep never ran -> scope UNVERIFIED -> degraded + an explicit ASK.
    a = UseCaseAnalyzer(use_case="scheme").analyze(spec, base_logs, _analysis("SUBJ01"))
    assert a.brief.degraded is True
    assert any("scope UNVERIFIED" in n for n in a.brief.notes)
    joined = " ".join(a.brief.action_backbone).lower()
    assert "did not run" in joined and "multi-day window" in joined


def test_the_scope_step_only_tells_you_to_void_where_a_document_exists():
    """ "VOID any still-active document" is actionable only on a subject that HAS one.

    The sweep OUTER-explodes, so it also returns records the actor booked but never issued.
    Naming those in the same breath told the operator to void documents on six records that had
    none (IR10000001, against the expert's two impacted records).
    """
    from src.models.pydantic_models import ImpactedAsset
    from src.usecases.base import _action_templates, _scope_step

    assets = [
        ImpactedAsset(subject="SUBJ09", asset_id="200-2000000004", in_window=True),
        # Out-of-window: adjacent business, must not make the void list either.
        ImpactedAsset(subject="SUBJ12", asset_id="200-2000000003", in_window=False),
    ]
    # Rendered with a real pack's own templates, so this asserts the SPLIT (the engine's
    # job) in a procedure's own wording (the pack's) — the two together.
    tpl = _action_templates(_SPEC_WITH_PACK_WORDING)
    line = _scope_step(
        "SUBJ01", ["SUBJ09", "SUBJ10", "SUBJ11"], "ran, 1 asset(s)", assets, tpl
    )
    with_asset, without = line.split("The sweep also returned")
    assert "SUBJ09" in with_asset and "HOLD any unsettled payout" in with_asset
    assert "SUBJ10" not in with_asset and "SUBJ11" not in with_asset
    assert "2 shipment(s)" in without and "no payout to hold" in without
    assert "SUBJ12" not in line  # out-of-window subject is not a scope step at all

    # No asset list supplied (or none in window) -> unchanged behaviour, name them all.
    plain = _scope_step("SUBJ01", ["SUBJ09", "SUBJ10"], "ran, 2 asset(s)", None, tpl)
    assert "SUBJ09" in plain and "SUBJ10" in plain and "shipment(s)" not in plain


def test_the_scope_step_does_not_call_a_truncated_sweep_bounded():
    """A sweep cut off at its row cap that returned no EXTRA subject is not a clean sweep.

    ``scope_status`` still starts with "ran", so the backbone read the clean branch and told
    the operator "the impact scope is bounded as reported" — the one sentence a truncated
    sweep cannot support, since the subjects beyond the cap were never looked at. Measured
    live at 500 rows returned of 3509 available.
    """
    from src.usecases.base import _action_templates, _scope_step

    tpl = _action_templates(None)
    truncated = _scope_step(
        "SUBJ01",
        [],
        "ran, at least 2 asset(s) across at least 1 subject(s); at least 0 NOT named in "
        "the alert; TRUNCATED — the sweep returned 2 row(s), which is its 2-row cap, so "
        "every count here is a LOWER BOUND and further impacted subjects may exist "
        "beyond it",
        None,
        tpl,
    )
    assert "bounded as reported" not in truncated
    low = truncated.lower()
    assert "cap" in low and "unknown" in low
    # It must still say the sweep RAN — an unrun sweep is a different finding.
    assert "did NOT run" not in truncated

    # An exhaustive clean sweep is unchanged.
    clean = _scope_step("SUBJ01", [], "ran, 2 asset(s) across 1 subject(s)", None, tpl)
    assert "bounded as reported" in clean


def test_containment_gated_only_on_indicator_driven_fraud():
    """An indicator-driven VALID FRAUD is a CANDIDATE: containment must be gated so the
    report can't append 'immediately void the documents' under 'do NOT auto-void'."""
    spec = dict(_SPEC)
    spec["indicator_threshold"] = 1
    spec["conditions"] = list(_SPEC["conditions"]) + [
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
    ]
    logs = {
        "record_lake": [
            {
                "locator": {"red": "SUBJ01"},
                "creator": {"sign": {"red": "0303CD"}, "org_unit_id": "ORGUNIT01"},
                "creation_date_time": "2026-07-27T18:00:00Z",
                "element_counters": {"AUX": 6, "OSI": 0, "RM": 0, "INS": 0},
                "pricing": {
                    "payment": [{"tender": [{"data_map": [{"key": "PM", "value": "CA"}]}]}]
                },
            }
        ],
        "settlement_report": [
            {
                "recordLocator": "SUBJ01",
                "retrieverUserSign": "0303CD",
                "transactionDateTime": "2026-07-27T18:30:00Z",
                "recordType": "SALE",
            }
        ],
    }
    a = UseCaseAnalyzer(use_case="scheme").analyze(spec, logs, _analysis("SUBJ01"))
    assert a.brief.containment_gated is True

    # A clean FALSE POSITIVE has no indicator fail -> nothing to gate.
    fp = UseCaseAnalyzer(use_case="scheme").analyze(_SPEC, logs, _analysis("SUBJ01"))
    assert fp.brief.containment_gated is False


def test_categorical_exclusion_ungates_containment_via_typed_field():
    """The case-builder must read `ConditionCheck.exclusion_kind`, the same field the
    verdict engine ranks on — not a prose note prefix that any rewording breaks.

    A categorical exclusion outranks the indicators in the engine, so the indicators did
    NOT drive a fraud verdict: gating containment on them would narrate a FALSE POSITIVE as
    a fraud candidate awaiting expert sign-off."""
    spec = dict(_SPEC)
    spec["indicator_threshold"] = 1
    spec["sources"] = dict(_SPEC["sources"], automated="automation_registry")
    spec["conditions"] = list(_SPEC["conditions"]) + [
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
            "id": "not_automated_ref",
            "label": "Not a known AUTOMATED account",
            "kind": "record_absence",
            "source": "automated",
            "subject_scope": False,
            "fields": ["profile"],
            "forbidden_values": ["AUTOMATED"],
            "match": "exact",
            "decisive": True,
            "decisive_on": ["fail"],
            "exclusion_kind": "categorical",
        },
    ]
    logs = {
        "record_lake": [
            {
                "locator": {"red": "SUBJ01"},
                "creator": {"sign": {"red": "0303CD"}, "org_unit_id": "ORGUNIT01"},
                "creation_date_time": "2026-07-27T18:00:00Z",
                "element_counters": {"AUX": 6, "OSI": 0, "RM": 0, "INS": 0},
                "pricing": {
                    "payment": [{"tender": [{"data_map": [{"key": "PM", "value": "CA"}]}]}]
                },
            }
        ],
        "settlement_report": [
            {
                "recordLocator": "SUBJ01",
                "retrieverUserSign": "0303CD",
                "transactionDateTime": "2026-07-27T18:30:00Z",
                "recordType": "SALE",
            }
        ],
        "automation_registry": [
            {"orgUnitId": "ORGUNIT01", "sign": "0303CD", "profile": "AUTOMATED"}
        ],
    }
    a = UseCaseAnalyzer(use_case="scheme").analyze(spec, logs, _analysis("SUBJ01"))
    # The indicator still failed (the override is a ranking, not a re-evaluation)...
    assert any(c.id == "cash_tender" for c in a.brief.decisive_indicators)
    # ...but containment is NOT gated on it, because the actor is an automation.
    assert a.brief.containment_gated is False
    assert any(
        c.exclusion_kind == "categorical" and c.result == "fail"
        for c in a.brief.decisive_fails
    )
    # Answered, so nothing is unattributed.
    assert a.brief.unanswered_attributions == []


def test_an_unanswered_attribution_reaches_the_brief_and_the_prompt():
    """WHO acted, when nothing established it, must be stated as unknown to the narrator.

    A categorical exclusion is declared `decisive_on: [fail]` — an UNKNOWN on a reference
    lookup must not degrade the case to INSUFFICIENT — so an unanswered one is
    `decisive=False` and reached NO list on the brief. That silence is not neutral: the
    remaining evidence describes how the actor behaved, which reads as a hint about what the
    actor is. On a live run (0184a3ce) a different exclusion carried the verdict, the
    automation check returned no rows, and the executive summary called the subject "the
    identified automation" — the engine's own most-read paragraph asserting the one finding
    the evidence had failed to establish."""
    spec = dict(_SPEC)
    spec["sources"] = dict(_SPEC["sources"], automated="automation_registry")
    spec["conditions"] = list(_SPEC["conditions"]) + [
        {
            "id": "not_automated_ref",
            "label": "Not a known AUTOMATED account",
            "kind": "record_absence",
            "source": "automated",
            "subject_scope": False,
            "fields": ["profile"],
            "forbidden_values": ["AUTOMATED"],
            "match": "exact",
            "decisive": True,
            "decisive_on": ["fail"],
            "exclusion_kind": "categorical",
        },
    ]
    logs = {
        "record_lake": [
            {
                "locator": {"red": "SUBJ01"},
                "creator": {"sign": {"red": "0303CD"}, "org_unit_id": "ORGUNIT01"},
                "creation_date_time": "2026-07-27T18:00:00Z",
                "element_counters": {"AUX": 6, "OSI": 0, "RM": 0, "INS": 0},
            }
        ],
        # Present and EMPTY, and nothing says the query named this identity: the register
        # answered nothing, so what kind of party acted is genuinely unknown.
        "automation_registry": [],
    }
    a = UseCaseAnalyzer(use_case="scheme").analyze(spec, logs, _analysis("SUBJ01"))
    check = next(
        (c for c in a.brief.unanswered_attributions if c.id == "not_automated_ref"), None
    )
    assert check is not None and check.result == "unknown"
    # It is NOT in `decisive_unknowns` — that is why a separate list exists.
    assert not any(c.id == "not_automated_ref" for c in a.brief.decisive_unknowns)

    # ...and it reaches the narrator, as a prohibition rather than a bare list entry.
    from src.brief_prompt import render_brief_for_prompt

    text = render_brief_for_prompt(a.brief)
    assert "UNANSWERED" in text and "Not a known AUTOMATED account" in text
    assert "must not be asserted" in text


def test_the_concept_snippet_length_is_one_number_and_the_pack_can_declare_it():
    """There were TWO independent cuts and only the smaller one was visible in the prompt.

    `collect_concept_refs` kept 500 chars and `brief_prompt` re-cut the same string to 220, so
    56% of every snippet reached nothing and no signal said so. That is expensive rather than
    untidy: a concept doc is written to correct a specific misreading, and a prohibition placed
    past the cut is prose nobody will ever read — while the author, checking the doc against
    the budget the ruleset documents, has no way to see it. One number now, it is the number
    the prompt shows, and the pack may declare it. The default is the old EFFECTIVE 220, so no
    existing pack's brief changes."""
    from src.brief_prompt import render_brief_for_prompt
    from src.usecases.base import DEFAULT_CONCEPT_SNIPPET_CHARS, _concept_snippet_chars

    assert DEFAULT_CONCEPT_SNIPPET_CHARS == 220
    # Absent, blank, unparseable and non-positive all no-op to the default.
    assert _concept_snippet_chars({}) == DEFAULT_CONCEPT_SNIPPET_CHARS
    assert _concept_snippet_chars({"concept_snippet_chars": None}) == 220
    assert _concept_snippet_chars({"concept_snippet_chars": "no"}) == 220
    assert _concept_snippet_chars({"concept_snippet_chars": 0}) == 220
    assert _concept_snippet_chars({"concept_snippet_chars": 900}) == 900

    class _Pack:
        def concepts_for(self, use_case, ids=None):
            return [
                {
                    "content": "---\nconcept_id: c1\ntitle: T\n---\n" + ("x" * 4000),
                    "metadata": {"concept_id": "c1", "title": "A concept"},
                }
            ]

    at_default = UseCaseAnalyzer.collect_concept_refs(_Pack(), "scheme")
    assert len(at_default[0].snippet) == 220
    declared = UseCaseAnalyzer.collect_concept_refs(_Pack(), "scheme", snippet_chars=900)
    assert len(declared[0].snippet) == 900

    # What the collector kept is what the render shows: no second cut behind it.
    # Duck-typed: `collect_concept_refs` returns a flat `ConceptRef` that `InvestigationBrief`
    # rejects (dual-import trap); the render only getattrs.
    class _Brief:
        use_case = "scheme"
        action_backbone = ["no containment."]
        concept_refs = declared

    text = render_brief_for_prompt(_Brief(), char_budget=100000)
    assert "x" * 900 in text


def test_a_non_decisive_exclusion_fail_reaches_the_brief_under_its_own_heading():
    """A FAIL is a FINDING whether or not it moved the verdict class, and the direction
    statement that stops it being inverted is needed either way.

    FOUND BY A LIVE RUN (`ed265d70`, 2026-08-15). The triage was `if c.decisive and result ==
    "fail"` / `elif c.decisive and result == "unknown"`, so a non-decisive exclusion FAIL fell
    through every branch and the brief was silent — while the same builder appends a
    fraud-INDICATOR fail regardless of decisiveness. The scorer, handed the rows and no
    direction, called the check's own columns "contradictory" at 0.68 and that reached a
    recommended action. It must NOT be merged into `decisive_fails`: that list is read as the
    verdict's reasons and `collect_precedents` matches cases on its ids, so merging would
    make a non-decisive check look like one the class rests on. A non-decisive UNKNOWN stays
    out of both — unlike a FAIL it established nothing, so there is no direction to state."""
    spec = dict(_SPEC)
    spec["sources"] = dict(_SPEC["sources"], reference="entitlement_ref")
    spec["conditions"] = list(_SPEC["conditions"]) + [
        {
            "id": "no_holder_entitlement",
            "label": "No holder entitlement explains the reduction",
            "kind": "value_matches_pattern",
            "source": "record",
            "subject_scope": False,
            "fields": ["holder_class"],
            "match_mode": "forbidden",
            "patterns": ["^CHD$"],
            "decisive": False,
            "fail_detail": "an entitlement is recorded on this record",
        },
        {
            "id": "unanswerable_check",
            "label": "A check with no data at all",
            "kind": "value_matches_pattern",
            "source": "reference",
            "subject_scope": False,
            "fields": ["never_present"],
            "match_mode": "forbidden",
            "patterns": ["^X$"],
            "decisive": False,
        },
    ]
    logs = {
        "record_lake": [
            {
                "locator": {"red": "SUBJ01"},
                "creator": {"sign": {"red": "0303CD"}, "org_unit_id": "ORGUNIT01"},
                "creation_date_time": "2026-07-27T18:00:00Z",
                "element_counters": {"AUX": 6, "OSI": 0, "RM": 0, "INS": 0},
                "holder_class": "CHD",
            }
        ],
        "entitlement_ref": [],
    }
    a = UseCaseAnalyzer(use_case="scheme").analyze(spec, logs, _analysis("SUBJ01"))
    got = next(
        (c for c in a.brief.explanatory_fails if c.id == "no_holder_entitlement"), None
    )
    assert got is not None and got.result == "fail" and got.decisive is False
    # NOT in the verdict-reasons list, and not in the precedent-matching ids either.
    assert not any(c.id == "no_holder_entitlement" for c in a.brief.decisive_fails)
    # The non-decisive UNKNOWN reaches neither list.
    assert not any(c.id == "unanswerable_check" for c in a.brief.explanatory_fails)
    assert not any(c.id == "unanswerable_check" for c in a.brief.decisive_unknowns)

    from src.brief_prompt import render_brief_for_prompt

    text = render_brief_for_prompt(a.brief)
    assert "WITHOUT CHANGING THE VERDICT CLASS" in text
    assert "No holder entitlement explains the reduction" in text
    assert "A check with no data at all" not in text


def _carve_out_spec(entry):
    """`_SPEC` plus one §"do not consider as evidence" entry."""
    return dict(_SPEC, do_not_consider=[entry])


_CARVE_OUT_LOGS = {
    "record_lake": [
        {
            "locator.red": "SUBJ03",
            "creator.sign.red": "0201GPSU",
            "creator.org_unit_id": "ORG2428D4",
            "creation_date_time": "2026-07-24T19:14:00Z",
            "element_counters.AUX": 3,  # NOT bare -> `bare` FAILs, i.e. it FOUND something
            "element_counters.OSI": 0,
            "element_counters.RM": 0,
            "element_counters.INS": 0,
        }
    ],
    "settlement_report": [
        {
            "retrieverUserSign": "0201GP",
            "transactionDateTime": "2026-07-24T20:19:00Z",
            "recordType": "SALE",
        }
    ],
}

_CARVE_OUT_NOTE = (
    "The procedure exempts one shape of this element and the engine cannot tell that shape "
    "from any other, so the element was accepted at face value."
)


def test_an_unenforceable_procedure_carve_out_is_stated_where_its_check_found_something():
    """A rule the procedure states and the engine cannot apply is a standing bias of the
    verdict, and it reached NO output at all.

    Every procedure names things that must not be read as evidence. Some are enforceable by a
    condition and some are not, and the second kind leaves the check it touches reading WIDER
    here than the procedure reads — which is a limitation of the adjudication, not of this
    run's coverage. A live pack declared three such carve-outs under a comment saying "declared
    here so the report states them"; nothing in `src/` read the key, so the report stated none
    of them and the only trace was the comment (`pack_validate`'s `unread-pack-key` exists
    because of this one). It rides in its own brief field and not in `notes`, whose every other
    producer is about coverage and whose heading in the prompt says so, and it must NOT set
    `degraded`: the run was not impaired.
    """
    spec = _carve_out_spec(
        {
            "id": "airline_performed",
            "enforced": False,
            "affects": ["bare"],
            "note": _CARVE_OUT_NOTE,
        }
    )
    a = UseCaseAnalyzer(use_case="scheme").analyze(
        spec, _CARVE_OUT_LOGS, _analysis(), knowledge_pack=_pack_with_case()
    )
    assert any(c.id == "bare" and c.result == "fail" for c in a.brief.decisive_fails)
    assert len(a.brief.unenforced_carve_outs) == 1
    stated = a.brief.unenforced_carve_outs[0]
    assert _CARVE_OUT_NOTE in stated
    # The check it applies to is NAMED: a limitation nobody can attach to a finding is one
    # nobody can act on, and the reader's next move is to judge that check by hand.
    assert "applies to: bare" in stated
    # Not a coverage note and not a degrade.
    assert not any(_CARVE_OUT_NOTE in n for n in a.brief.notes)

    from src.brief_prompt import render_brief_for_prompt

    text = render_brief_for_prompt(a.brief)
    assert "Procedure carve-outs this engine does NOT apply" in text
    assert _CARVE_OUT_NOTE in text
    # In the body, not the tail: the tail is what a global cut takes first, and a narrator that
    # sees the finding without the limitation will present it as procedure-conformant.
    assert text.index(_CARVE_OUT_NOTE) < text.index("Relevant KB concepts:")


def test_a_carve_out_says_nothing_where_its_check_found_nothing():
    """The gate that stops this becoming boilerplate on every report.

    A carve-out NARROWS what counts as evidence, so it can only have changed a reading where
    the check FOUND something. On a PASS there was nothing to exclude; on an `unknown` nothing
    was read at all. Both directions are asserted here because a fix that emitted the statement
    unconditionally would look identical on the incident it was written for — and then the one
    incident where the limitation really applies is a paragraph the reader has already learned
    to skip.
    """
    spec = _carve_out_spec(
        {
            "id": "airline_performed",
            "enforced": False,
            "affects": ["bare"],
            "note": _CARVE_OUT_NOTE,
        }
    )
    passing = {
        "record_lake": [
            dict(
                _CARVE_OUT_LOGS["record_lake"][0],
                **{"locator.red": "AAAAAA", "element_counters.AUX": 0},
            )
        ],
        "settlement_report": _CARVE_OUT_LOGS["settlement_report"],
    }
    a = UseCaseAnalyzer(use_case="scheme").analyze(spec, passing, _analysis("AAAAAA"))
    assert any(
        c.id == "bare" and c.result == "pass"
        for c in a.brief.verdict.subjects[0].checks
    )
    assert a.brief.unenforced_carve_outs == []

    # ...and where the check could not be answered at all.
    unknown = {"record_lake": [{"locator.red": "SUBJ03"}], "settlement_report": []}
    b = UseCaseAnalyzer(use_case="scheme").analyze(spec, unknown, _analysis())
    assert any(
        c.id == "bare" and c.result == "unknown"
        for c in b.brief.verdict.subjects[0].checks
    )
    assert b.brief.unenforced_carve_outs == []


def test_an_enforced_carve_out_and_a_bare_one_state_nothing():
    """Two silences, for two different reasons, and both would be a FALSE statement otherwise.

    `enforced: true` says the engine DOES apply it — announcing a limitation there would invent
    one. An entry with no `affects` and no `note` is the shape a pack uses for a carve-out over
    something no condition reads: there is no path by which it could become evidence, so there
    is nothing to warn about. A pack declaring the key at all must not be punished for
    documenting the carve-outs that are safe.
    """
    enforced = _carve_out_spec(
        {
            "id": "handled",
            "enforced": True,
            "affects": ["bare"],
            "note": _CARVE_OUT_NOTE,
        }
    )
    a = UseCaseAnalyzer(use_case="scheme").analyze(
        enforced, _CARVE_OUT_LOGS, _analysis()
    )
    assert a.brief.unenforced_carve_outs == []

    documented = _carve_out_spec(
        {
            "id": "not_read",
            "enforced": "not_read",
            "description": "not read by any condition",
        }
    )
    b = UseCaseAnalyzer(use_case="scheme").analyze(
        documented, _CARVE_OUT_LOGS, _analysis()
    )
    assert b.brief.unenforced_carve_outs == []

    # And a ruleset declaring no carve-outs at all is byte-identical to before.
    c = UseCaseAnalyzer(use_case="scheme").analyze(_SPEC, _CARVE_OUT_LOGS, _analysis())
    assert c.brief.unenforced_carve_outs == []
    from src.brief_prompt import render_brief_for_prompt

    assert "carve-out" not in render_brief_for_prompt(c.brief)


# --- what the ALERT declares, when the alert and the extraction disagree ----------------
# Alert fields are ground truth; divergence is a finding. These tests pin the boundary:
# a free-text body may be replaced by the extraction, but a structured field may not.


def _alert_spec(fields, **entry):
    """A ruleset whose whole content is one alert record declaring one fact."""
    return {
        "sources": {"alert": "alert_feed", "record": "record_lake"},
        "alert_record": {
            "source": "alert",
            "label_fields": ["alert_id"],
            "identify": [{"from_entity": "record", "fields": ["body"], "required": True}],
            "declares": [dict({"field": "kind", "fields": fields}, **entry)],
        },
    }


def _alert_analysis(*entities):
    return SimpleNamespace(
        extracted_entities=[ExtractedEntity(type=t, value=v) for t, v in entities],
        event_time=None,
    )


def test_a_structured_alert_field_outranks_a_stale_extraction():
    """The alert states X, the extraction says Y, no source corroborates either -> X.

    Reported as `stated` (nothing was declared to confirm it against) and carrying the
    ALERT's value. Replacing it with the extraction inverts the premise of the whole
    function: the report would name the extraction while claiming to quote the alert.
    Measured on a synthesised detector variant of job 0184a3ce — a structured field read one
    detector and the reconciliation printed the other.
    """
    spec = _alert_spec(["detector"], from_entity="kind")
    logs = {"alert_feed": [{"alert_id": "A1", "body": "SUBJ01 acted", "detector": "B"}]}
    facts = UseCaseAnalyzer.build_alert_facts(
        logs, spec, _alert_analysis(("record", "SUBJ01"), ("kind", "A"))
    )
    assert facts is not None and facts.located
    declared = [(f.declared, f.status) for f in facts.declared_facts if f.field == "kind"]
    assert declared == [("B", "stated")]


def test_a_free_text_body_still_falls_back_to_the_extraction():
    """The other side of the same boundary, and why the rule cannot be "always the alert".

    A body carries the fact rather than equalling it, so declaring the body would print the
    whole email as one "declared value". The body is recognised by what it CARRIES — it
    strictly contains another value the incident named — which needs no length threshold, no
    field-name list and no domain vocabulary.

    The extracted value is deliberately NOT a substring of the body: an ordinary containment
    hit would narrow without ever reaching the fallback, and the fallback is what this test
    is for. That is also the real shape — a DERIVED value (a normalised window, a padded
    identifier) is what the extraction contributes that the body does not spell out.
    """
    spec = _alert_spec(["body"], from_entity="kind")
    logs = {
        "alert_feed": [
            {"alert_id": "A1", "body": "SUBJ01 acted; the kind is not spelt out here"}
        ]
    }
    facts = UseCaseAnalyzer.build_alert_facts(
        logs, spec, _alert_analysis(("record", "SUBJ01"), ("kind", "ESCALATED"))
    )
    assert facts is not None and facts.located
    declared = [f.declared for f in facts.declared_facts if f.field == "kind"]
    assert declared == ["ESCALATED"]


# --- a source cannot corroborate a value it supplied ------------------------------------
# The other half: these pin what counts as a confirmation. If `confirm_in` names the alert
# source, the comparison is the declared value against the row it came from (`x == x`), so
# `confirmed` is the only reachable outcome.


def test_a_source_cannot_confirm_a_value_it_supplied():
    """`confirm_in` naming the alert record's own source is not corroboration.

    Measured on job `fc2b1b0e`: retrieval returned 428 rows of a DIFFERENT identity sharing
    the subject's login string, the locator selected one of them as this incident's alert
    record, and that stranger's scope was reported `[CONFIRMED]` against itself. The same
    shape would confirm a typo or an empty string just as confidently. Reads `stated` — the
    alert asserts it and nothing independent was ever asked — not `confirmed`.
    """
    spec = _alert_spec(["scope"], from_entity="scope", confirm_in=["alert"])
    spec["alert_record"]["declares"][0]["confirm_fields"] = ["scope"]
    logs = {
        "alert_feed": [{"alert_id": "A1", "body": "SUBJ01 acted", "scope": "FOREIGN"}]
    }
    facts = UseCaseAnalyzer.build_alert_facts(
        logs, spec, _alert_analysis(("record", "SUBJ01"))
    )
    assert facts is not None and facts.located
    declared = [
        (f.declared, f.status) for f in facts.declared_facts if f.field == "kind"
    ]
    assert declared == [("FOREIGN", "stated")]


def test_an_independent_source_still_confirms():
    """The other side of that boundary: a genuinely second reading still reads `confirmed`.

    Same declared value, same alert row — the only difference is that `confirm_in` names a
    source the declared side was NOT read from. Without this, the fix above would be
    indistinguishable from disabling reconciliation.
    """
    spec = _alert_spec(["scope"], from_entity="scope", confirm_in=["record"])
    spec["alert_record"]["declares"][0]["confirm_fields"] = ["scope"]
    logs = {
        "alert_feed": [{"alert_id": "A1", "body": "SUBJ01 acted", "scope": "HOME"}],
        "record_lake": [{"scope": "HOME"}],
    }
    facts = UseCaseAnalyzer.build_alert_facts(
        logs, spec, _alert_analysis(("record", "SUBJ01"))
    )
    assert facts is not None and facts.located
    declared = [
        (f.declared, f.status) for f in facts.declared_facts if f.field == "kind"
    ]
    assert declared == [("HOME", "confirmed")]


def test_a_declared_source_that_returned_nothing_is_still_a_gap():
    """An independent source that was named and stayed SILENT is `not_found`, not `stated`.

    The distinction the self-confirmation fix must not collapse: `stated` means nothing was
    ever going to corroborate this, `not_found` means something was asked and did not answer.
    Only the second sends the reader looking for a retrieval gap, and conflating them would
    hide exactly the gap this reconciliation exists to report.
    """
    spec = _alert_spec(["scope"], from_entity="scope", confirm_in=["alert", "record"])
    spec["alert_record"]["declares"][0]["confirm_fields"] = ["scope"]
    logs = {"alert_feed": [{"alert_id": "A1", "body": "SUBJ01 acted", "scope": "HOME"}]}
    facts = UseCaseAnalyzer.build_alert_facts(
        logs, spec, _alert_analysis(("record", "SUBJ01"))
    )
    assert facts is not None and facts.located
    declared = [
        (f.declared, f.status) for f in facts.declared_facts if f.field == "kind"
    ]
    assert declared == [("HOME", "not_found")]


# --- a mismatch is a contradiction, not a disagreement of values ────────────────────
# Third layer. A mismatch requires the source to have answered in full (`keyed_sources` +
# `row_caps`). Asserted in both directions: a cap cannot un-match a value that matched.


def _mismatch_spec():
    """One declared value, one independent source carrying a DIFFERENT value."""
    spec = _alert_spec(["scope"], from_entity="scope", confirm_in=["record"])
    spec["alert_record"]["declares"][0]["confirm_fields"] = ["scope"]
    return spec


def test_a_truncated_source_cannot_contradict_the_alert():
    """A source cut off at its row cap disagrees only about the rows it happened to show.

    Measured on job `ad74e724`: the declared organisation was reported MISMATCH against five
    OTHER organisations pooled out of an event log that came back at exactly its 500-row cap.
    A mismatch tells the reader the data denies the alert — here the data had not finished
    answering. Reads `not_found` (the gap it always was) and carries the reason in `note`,
    while still SHOWING the sample, because suppressing it would hide the retrieval limit
    too. `>=` for the same reason the verdict engine uses it: a backend may overshoot by a
    row, and one source must not read truncated there and exhaustive here.
    """
    logs = {
        "alert_feed": [{"alert_id": "A1", "body": "SUBJ01 acted", "scope": "HOME"}],
        "record_lake": [{"scope": "AWAY1"}, {"scope": "AWAY2"}],
    }
    facts = UseCaseAnalyzer.build_alert_facts(
        logs,
        _mismatch_spec(),
        _alert_analysis(("record", "SUBJ01")),
        row_caps={"record_lake": 2},
        keyed_sources={"record_lake": True},
    )
    assert facts is not None and facts.located
    fact = next(f for f in facts.declared_facts if f.field == "kind")
    assert (fact.declared, fact.status) == ("HOME", "not_found")
    assert "TRUNCATED" in fact.note and "2-row cap" in fact.note
    assert fact.found == "AWAY1, AWAY2" and fact.source == "record_lake"


def test_a_check_lifted_out_of_its_subject_carries_the_subject_with_it():
    """Builder half of the cross-subject attribution fix.

    `decisive_fails` / `explanatory_fails` / `decisive_indicators` / `decisive_unknowns` are
    flat across subjects: the `ValidationSubject` is gone when a check is appended. The
    subject is stamped at the lift, the only place that still knows it.

    Two properties: the lifted copy carries the subject, and the check inside the subject is
    untouched (`model_copy`, not mutation). The render half lives in
    `test_report_generation.py` and survives the mutation that kills this one.
    """
    logs = {
        "record_lake": [
            {
                "locator": {"red": "SUBJ01"},
                "creator": {"sign": {"red": "0303CD"}, "org_unit_id": "ORGUNIT01"},
                "creation_date_time": "2026-07-27T18:00:00Z",
                # FAILS the absence check, and the counters DIFFER from the sibling's — which
                # is the whole shape of the defect: one number apart, everything else equal.
                "element_counters": {"AUX": 6, "OSI": 0, "RM": 80, "INS": 0},
            },
            {
                "locator": {"red": "SUBJ02"},
                "creator": {"sign": {"red": "0303CD"}, "org_unit_id": "ORGUNIT01"},
                "creation_date_time": "2026-07-27T18:00:00Z",
                "element_counters": {"AUX": 6, "OSI": 0, "RM": 88, "INS": 0},
            },
        ],
        "settlement_report": [],
    }
    analysis = SimpleNamespace(
        incident_summary="SCHEME issuance fraud",
        initial_hypotheses=[],
        key_investigation_areas=[],
        correlation_keys=[],
        extracted_entities=[
            ExtractedEntity(type="record", value="SUBJ01"),
            ExtractedEntity(type="record", value="SUBJ02"),
        ],
        event_time=None,
    )
    a = UseCaseAnalyzer(use_case="scheme").analyze(_SPEC, logs, analysis)
    subs = {sv.subject_value: sv for sv in a.verdict.subjects}
    assert set(subs) == {"SUBJ01", "SUBJ02"}, sorted(subs)

    lifted = [c for c in a.brief.decisive_fails if c.id == "bare"]
    assert len(lifted) == 2, [c.observed for c in lifted]
    # EVERY lifted check names a subject, and it names the one its own value came from.
    for c in lifted:
        assert c.subject in ("SUBJ01", "SUBJ02"), c.subject
        truth = next(
            k for k in subs[c.subject].checks if k.id == "bare"
        )
        assert c.observed == truth.observed, (c.subject, c.observed, truth.observed)
    # The two attributions are DISTINCT — a stamp that put the same subject on both would
    # satisfy every assertion above and reproduce the defect.
    assert {c.subject for c in lifted} == {"SUBJ01", "SUBJ02"}
    # And the check inside the subject is untouched: the parent states it there.
    for sv in a.verdict.subjects:
        assert all(k.subject == "" for k in sv.checks), sv.subject_value


def test_a_subject_unscoped_source_cannot_contradict_the_alert():
    """The other half: a query that named no identity carries other people's values.

    Same rows, no cap in play — the query simply did not constrain the subject, so what it
    returned may be about anyone. `keyed_sources` is the only thing that can say so, and it
    is read for its ABSENCE of the key rather than for a False, since a source that was never
    measured is not a source that was measured unkeyed.
    """
    logs = {
        "alert_feed": [{"alert_id": "A1", "body": "SUBJ01 acted", "scope": "HOME"}],
        "record_lake": [{"scope": "AWAY1"}],
    }
    facts = UseCaseAnalyzer.build_alert_facts(
        logs,
        _mismatch_spec(),
        _alert_analysis(("record", "SUBJ01")),
        row_caps={"record_lake": 500},
        keyed_sources={"record_lake": False},
    )
    assert facts is not None and facts.located
    fact = next(f for f in facts.declared_facts if f.field == "kind")
    assert (fact.declared, fact.status) == ("HOME", "not_found")
    assert "did not constrain this subject" in fact.note
    assert fact.found == "AWAY1"


def test_neither_register_can_un_confirm_a_value_that_matched():
    """The asymmetry that makes the downgrade sound, and the test that bounds it.

    A cap and an unscoped query are reasons a source's SILENCE proves nothing. Neither is a
    reason to doubt a value it actually carries: the matching row is real regardless of how
    many rows followed it or whom the rest belong to. So the qualification applies to the
    negative direction only — exactly as `key_was_enforced` is consulted for an EMPTY result
    alone. Without this, "be careful about mismatches" would degrade into "trust nothing",
    and the reconciliation would report a gap on every capped source in the run.
    """
    logs = {
        "alert_feed": [{"alert_id": "A1", "body": "SUBJ01 acted", "scope": "HOME"}],
        "record_lake": [{"scope": "HOME"}, {"scope": "AWAY1"}],
    }
    facts = UseCaseAnalyzer.build_alert_facts(
        logs,
        _mismatch_spec(),
        _alert_analysis(("record", "SUBJ01")),
        row_caps={"record_lake": 2},
        keyed_sources={"record_lake": False},
    )
    assert facts is not None and facts.located
    fact = next(f for f in facts.declared_facts if f.field == "kind")
    assert (fact.declared, fact.status) == ("HOME", "confirmed")
    assert fact.note == "" and fact.found == "HOME"


def test_a_caller_that_measures_neither_register_still_reports_the_mismatch():
    """A standalone caller passing no registers keeps the pre-existing reading.

    `build_alert_facts` is reachable without the pipeline's two per-run registers (the
    standalone `analyze` path, the replay harness), and there the honest reading is the one
    the rows support: the source answered, it carries another value, that is a mismatch.
    Defaulting the other way would silence every disagreement in every caller that does not
    measure retrieval — the fix would then be indistinguishable from deleting the status.
    """
    logs = {
        "alert_feed": [{"alert_id": "A1", "body": "SUBJ01 acted", "scope": "HOME"}],
        "record_lake": [{"scope": "AWAY1"}],
    }
    facts = UseCaseAnalyzer.build_alert_facts(
        logs, _mismatch_spec(), _alert_analysis(("record", "SUBJ01"))
    )
    assert facts is not None and facts.located
    fact = next(f for f in facts.declared_facts if f.field == "kind")
    assert (fact.declared, fact.status) == ("HOME", "mismatch")
    assert fact.note == "" and fact.found == "AWAY1"


# --- the alert's own other facts are never evidence against one of them ─────────────
# Fourth layer: arithmetic over the declared list. A source carrying only siblings
# corroborated them and is silent about this one. Single-valued case must be byte-identical.


def _set_valued_spec(confirm_in=("record",)):
    """The same one-fact ruleset, with the alert stating a SET of values for it."""
    spec = _alert_spec(["scope"], from_entity="scope", confirm_in=list(confirm_in))
    spec["alert_record"]["declares"][0]["confirm_fields"] = ["scope"]
    spec["sources"]["audit"] = "audit_log"
    return spec


def test_a_value_the_same_alert_declares_cannot_contradict_a_sibling():
    """The data reproduced one of the set and not the other: a gap, not a contradiction.

    Measured on two live runs. `aca5fbbc` — an amount alert grouping two discounts on
    two different records — had the record source carry the first ticket, and the second
    was reported MISMATCH against it. `ad74e724` named one actor twice, as a login and
    as a sign; the record source binds only the sign column, so the sign read CONFIRMED
    and the login read MISMATCH against its own other spelling. In both the value quoted
    as contradicting the alert was a value the alert itself declared and the same table
    marked CONFIRMED two lines above, and the DELTA paragraph sent the reader after a
    parsing defect.
    """
    logs = {
        "alert_feed": [
            {"alert_id": "A1", "body": "SUBJ01 acted", "scope": ["HOME", "AWAY9"]}
        ],
        "record_lake": [{"scope": "HOME"}],
    }
    facts = UseCaseAnalyzer.build_alert_facts(
        logs, _set_valued_spec(), _alert_analysis(("record", "SUBJ01"))
    )
    assert facts is not None and facts.located
    got = {f.declared: f for f in facts.declared_facts if f.field == "kind"}
    assert sorted(got) == ["AWAY9", "HOME"]
    assert got["HOME"].status == "confirmed" and got["HOME"].note == ""
    assert got["AWAY9"].status == "not_found"
    assert "the alert ALSO declares" in got["AWAY9"].note
    # The sample still prints, exactly as it does for a capped or unkeyed source: the
    # reader sees what the source held, and the note is what stops it reading as a
    # mismatch.
    assert got["AWAY9"].found == "HOME" and got["AWAY9"].source == "record_lake"


def test_a_source_carrying_a_sibling_AND_a_stranger_still_contradicts():
    """The conservative direction: one undeclared value is enough to keep the mismatch.

    The rule is "every value this source carries is one the alert also declares", not
    "some value is". A source holding a sibling beside a value the alert never named has
    produced a reading of this fact that disagrees, and the sibling does not excuse it —
    otherwise declaring a set would immunise the whole entry against contradiction.
    """
    logs = {
        "alert_feed": [
            {"alert_id": "A1", "body": "SUBJ01 acted", "scope": ["HOME", "AWAY9"]}
        ],
        "record_lake": [{"scope": "HOME"}, {"scope": "STRANGER"}],
    }
    facts = UseCaseAnalyzer.build_alert_facts(
        logs, _set_valued_spec(), _alert_analysis(("record", "SUBJ01"))
    )
    fact = next(
        f for f in facts.declared_facts if f.field == "kind" and f.declared == "AWAY9"
    )
    assert (fact.status, fact.note) == ("mismatch", "")
    assert fact.found == "HOME, STRANGER"


def test_a_sibling_only_source_does_not_outrank_another_sources_contradiction():
    """Two independent sources, and the mismatch must name the one that disagrees.

    This is the `ad74e724` shape and the reason the sibling test selects a candidate
    rather than short-circuiting the first non-matching source: the record lake carried
    the actor's other spelling, and the auth trail carried four strangers. Deciding on
    the first source alone would report the gap and drop a real disagreement; deciding
    on the last would report the alert's own value. Live, the surviving mismatch is then
    downgraded by the truncation caveat above — the two rules compose, and neither
    subsumes the other.
    """
    logs = {
        "alert_feed": [
            {"alert_id": "A1", "body": "SUBJ01 acted", "scope": ["HOME", "AWAY9"]}
        ],
        "record_lake": [{"scope": "HOME"}],
        "audit_log": [{"scope": "STRANGER"}],
    }
    facts = UseCaseAnalyzer.build_alert_facts(
        logs,
        _set_valued_spec(("record", "audit")),
        _alert_analysis(("record", "SUBJ01")),
    )
    fact = next(
        f for f in facts.declared_facts if f.field == "kind" and f.declared == "AWAY9"
    )
    assert (fact.status, fact.source) == ("mismatch", "audit_log")
    assert fact.found == "STRANGER" and fact.note == ""


def test_a_trigger_without_an_alert_record_attempts_no_location_and_says_so():
    """The locator is the discriminator between "did not look" and "looked and missed".

    Every failing path above writes a non-empty `locator` naming which way it failed. The path
    that declares no `alert_record:` at all writes none, because nothing was attempted — and the
    report section keys the "ALERT RECORD NOT READ" line on the locator rather than on the falsy
    `located` for exactly that reason (see `test_report_generation.py`). Pinned here because the
    two halves live in different files and a plausible-looking `locator="not declared"` on this
    path would restore the defect from the other end.
    """
    spec = {
        "sources": {"record": "record_lake"},
        "trigger": {"description": "Fires on a BURST. It is NOT amount-based."},
    }
    facts = UseCaseAnalyzer.build_alert_facts(
        {"record_lake": [{"scope": "HOME"}]}, spec, _alert_analysis(("record", "SUBJ01"))
    )
    assert facts is not None
    assert facts.located is False
    assert facts.locator == ""
    assert facts.declared_facts == [] and facts.source == ""
    assert "NOT amount-based" in facts.trigger


def test_join_status_explicit_ran_vs_not_evaluated():
    transforms = [
        TransformResult(
            label="join_user", op="cross_source_overlap", rows=[{"value": "x"}]
        ),
    ]
    status = UseCaseAnalyzer.join_status_from_transforms(
        transforms, ["join_user", "join_record"]
    )
    assert "ran, 1" in status["join_user"]
    assert "not evaluated" in status["join_record"]


def test_collect_precedents_prefers_same_verdict_and_route():
    pack = _pack_with_case(verdict="FALSE POSITIVE", route="DSS-CDG")
    precs = UseCaseAnalyzer.collect_precedents(
        pack, "scheme", verdict_label="FALSE POSITIVE", route="DSS-CDG"
    )
    assert precs and precs[0].case_id == "case_a1b2c3"
    assert precs[0].verdict == "FALSE POSITIVE"
    # No pack -> empty, never raises.
    assert UseCaseAnalyzer.collect_precedents(None, "scheme") == []


# --- registry ----------------------------------------------------------------


def test_registry_returns_one_generic_analyzer_scoped_to_the_use_case():
    """No use case gets its own class — only its own NAME.

    The registry used to map a hard-coded ``{"pb-app-scheme-001", "scheme"}`` to a bespoke
    ``SchemeAnalyzer``, which put a domain name in the engine and meant the next use case
    needed both a Python class and an entry here. The ruleset key already IS the use-case
    name (the pack's ``use_cases/<name>/`` folder, and how concept/case docs are tagged), so
    every incident gets the same analyzer carrying that name.
    """
    # Duck-type on class name: the registry uses flat imports (`usecases.base`) while this
    # test uses `src.usecases.base`, so isinstance across the two module identities is False.
    scheme = get_analyzer("PB-APP-SCHEME-001", "scheme")
    assert type(scheme).__name__ == "UseCaseAnalyzer"
    assert scheme.use_case == "scheme"
    other = get_analyzer("PB-ATO-002", "ato")
    assert type(other).__name__ == "UseCaseAnalyzer"
    assert other.use_case == "ato"
    # No ruleset matched -> fall back to the playbook id so KB lookups still have a key.
    assert get_analyzer("PB-ATO-002", "").use_case == "pb-ato-002"
    # Nothing known -> empty, which makes the KB lookups no-op rather than mismatch.
    assert get_analyzer("", "").use_case == ""


def test_analyzer_is_evidence_only_without_a_spec():
    a = UseCaseAnalyzer(use_case="ato").analyze(
        None, {}, _analysis(), knowledge_pack=None
    )
    assert a.verdict is None
    assert a.brief.use_case == "ato"
    assert a.brief.action_backbone  # a generic review step is always present


def test_a_use_case_with_no_case_builder_block_still_gets_a_verdict_brief():
    """Every ``case_builder:`` step must no-op when the pack declares none of it.

    Otherwise folding the SCHEME orchestration into the one shared analyzer would have made
    every OTHER use case pay for declarations it never wrote.
    """
    spec = {k: v for k, v in _SPEC.items() if k != "case_builder"}
    a = UseCaseAnalyzer(use_case="ato").analyze(
        spec, {"record_lake": [{"locator_red": "AAAAAA"}]}, _analysis("AAAAAA")
    )
    assert a.verdict is not None
    assert a.brief.join_status == {}  # no expected_joins declared
    assert a.brief.concept_refs == []  # no concept ids declared
    # No projection_guard / scope_notes declared -> no notes invented from thin air.
    assert not any("projection" in n.lower() for n in (a.brief.notes or []))


def test_pack_declared_action_templates_replace_the_neutral_wording():
    """The engine picks the BRANCH; the pack supplies the PROSE.

    `derive_action_backbone` used to f-string one domain's subject noun, its clause citations
    and its follow-up wording — one procedure compiled into a generic engine, which every
    OTHER use case would then have rendered verbatim. Same verdict, same branch, two
    wordings: the engine's neutral default and a shipped pack's own.
    """
    logs = {
        "record_lake": [
            {
                "locator.red": "SUBJ03",
                "creator.sign.red": "0201GPSU",
                "creator.org_unit_id": "ORG2428D4",
                "creation_date_time": "2026-07-24T19:14:00Z",
                "element_counters.AUX": 3,  # NOT bare -> decisive FAIL -> close
                "route.air.board_point": "DSS",
                "route.air.off_point": "CDG",
            }
        ],
        "settlement_report": [
            {
                "retrieverUserSign": "0201GP",
                "transactionDateTime": "2026-07-24T20:19:00Z",
            }
        ],
    }
    neutral = UseCaseAnalyzer(use_case="scheme").analyze(_SPEC, logs, _analysis())
    packed = UseCaseAnalyzer(use_case="scheme").analyze(
        _SPEC_WITH_PACK_WORDING, logs, _analysis()
    )
    n = " ".join(neutral.brief.action_backbone)
    p = " ".join(packed.brief.action_backbone)
    # Same branch (both CLOSE), different prose — and the pack's own nouns appear ONLY under
    # the pack that declares them.
    assert "CLOSE" in n and "CLOSE" in p
    assert "shipment SUBJ03" in p and "shipment SUBJ03" not in n
    assert "CLOSE the claim" in p and "CLOSE the incident" in n
    assert "SUBJ03:" in n  # neutral subject_label is the bare identifier


def test_no_domain_procedure_wording_is_hard_coded_in_the_engine():
    """The neutral defaults must name no domain. A grep-level guard, deliberately.

    This is the check that would have caught the leak: the strings sat in Python for months
    while every test asserted on them, because a test that asserts "§4.1.1 appears" cannot
    tell whether the engine or the pack produced it.
    """
    from src.usecases.base import _ACTION_TEMPLATES

    blob = " ".join(_ACTION_TEMPLATES.values())
    for term in (
        "consignment",
        "§",
        "IR ",
        "SMC",
        "Settlement Report",
        "waybill",
        "manifest",
        "haulier",
    ):
        assert term not in blob, f"engine default wording names a domain: {term!r}"


def test_analyzer_never_raises_on_bad_input():
    # Garbage logs/spec must degrade to a brief, not raise.
    a = UseCaseAnalyzer(use_case="scheme").analyze(
        {"sources": {"record": "x"}}, {"x": "notalist"}, _analysis()
    )
    assert a.brief is not None


def _logs():
    """Logs the analyzer CAN adjudicate on its own — `bare` fails decisively here.

    Shared by the two tests below because the point of both is which of two possible
    verdicts ends up on the brief: this shape re-derives to FALSE POSITIVE, so a handed
    verdict saying anything else can only have arrived by being carried through.
    """
    return {
        "record_lake": [
            {
                "locator.red": "SUBJ03",
                "creator.sign.red": "0201GPSU",
                "creator.org_unit_id": "ORG2428D4",
                "creation_date_time": "2026-07-24T19:14:00Z",
                "element_counters.AUX": 3,  # NOT bare -> decisive FAIL
                "element_counters.OSI": 0,
                "element_counters.RM": 0,
                "element_counters.INS": 0,
            }
        ],
        "settlement_report": [
            {
                "retrieverUserSign": "0201GP",
                "transactionDateTime": "2026-07-24T20:19:00Z",
                "recordType": "SALE",
            }
        ],
    }


def test_the_callers_verdict_is_used_verbatim_and_not_re_derived():
    """The brief must carry the caller's verdict, not re-derive it.

    `analyze` can evaluate a ruleset from `logs`, but the correlation stage additionally
    passes `row_caps` and `keyed_sources`: zero rows is an answer, not a gap, only when the
    query was keyed and not truncated. The two readings diverge where it matters most.

    Asserted by passing a verdict the analyzer could not have produced from these logs,
    so inheriting it is the only way to pass.
    """
    # Flat import: `base.py` builds `CaseAssessment` with flat models and rejects
    # `src.`-qualified ones (dual-import trap). This test asserts object identity, so it
    # must hand over the same class the analyzer validates against.
    from models.pydantic_models import (  # noqa: E402
        ConditionCheck,
        SubjectVerdict,
        ValidationVerdict,
    )

    handed = ValidationVerdict(
        summary="HANDED DOWN: 1",
        subjects=[
            SubjectVerdict(
                subject_type="record",
                subject_value="SUBJ03",
                verdict="HANDED DOWN",
                verdict_class="false_positive",
                checks=[
                    ConditionCheck(
                        id="lookup_absence",
                        label="The subject is absent from the register",
                        result="pass",
                        observed="identity not found",
                        detail="the subject is absent from this reference source",
                        decisive=True,
                        polarity="exclusion",
                        exclusion_kind="categorical",
                    )
                ],
            )
        ],
    )
    a = UseCaseAnalyzer(use_case="scheme").analyze(
        _SPEC, _logs(), _analysis(), verdict=handed
    )
    assert a.verdict is handed, "the caller's verdict object must be carried, not rebuilt"
    assert a.brief.verdict is handed
    assert a.brief.verdict.summary == "HANDED DOWN: 1"
    # And the PASS must not resurface as an unanswered attribution, which is the exact
    # shape that made the narration contradict the verdict.
    unanswered_ids = {c.id for c in (a.brief.unanswered_attributions or [])}
    assert "lookup_absence" not in unanswered_ids
    assert not any(c.id == "lookup_absence" for c in (a.brief.decisive_unknowns or []))


def test_the_analyzer_still_evaluates_when_no_verdict_is_handed_to_it():
    """Standalone use (its own tests, a dry-run script) must keep working unchanged.

    `verdict=None` is not "no verdict" — it means the caller did not evaluate one, so the
    analyzer falls back to evaluating the ruleset itself. Losing that would make the
    analyzer unusable outside the pipeline.
    """
    a = UseCaseAnalyzer(use_case="scheme").analyze(_SPEC, _logs(), _analysis())
    assert a.verdict is not None
    assert a.verdict.subjects, "the fallback evaluation must still produce subjects"
