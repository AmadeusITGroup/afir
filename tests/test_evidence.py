"""Tests for the deterministic investigation-evidence reduction (src/evidence.py).

All functions are pure (no LLM, no I/O), so these exercise them directly:
epoch/ISO timestamp parsing, cross-source chronology ordering + volume-triggered
aggregation, column trimming (constant/masked/blob dropped, relevant kept),
actor attribution, build_evidence determinism, and budget degradation.
"""

import pytest

from src.evidence import (CLIP_NOTE, _DEFAULT_CHAR_BUDGET, _half_lens,
                          attribute_actors, build_chronology, build_evidence,
                          degrade_to_budget, parse_timestamp,
                          render_for_prompt, trim_columns)
from src.models.pydantic_models import (ActorRollup, ChronologyEvent,
                                        EvidencePack, SourceEvidence)


class _Key:
    """Minimal duck-typed CorrelationKey (fields evidence reads)."""

    def __init__(self, entity_hint, sources, time_fields=None):
        self.entity_hint = entity_hint
        self.sources = sources
        self.time_fields = time_fields or {}


# --- timestamp parsing ------------------------------------------------------


def test_parse_timestamp_epoch_seconds_and_millis():
    sec = parse_timestamp(1721853060)  # 10-digit epoch seconds
    ms = parse_timestamp(1721853060000)  # 13-digit epoch millis
    assert sec is not None and ms is not None
    assert sec == ms  # same instant
    assert sec.year == 2024 and sec.month == 7 and sec.day == 24


def test_parse_timestamp_numeric_string_epoch():
    assert parse_timestamp("1721853060") == parse_timestamp(1721853060)


def test_parse_timestamp_iso_and_junk():
    assert parse_timestamp("2026-07-24T20:31:00Z") is not None
    assert parse_timestamp("not-a-date") is None
    assert parse_timestamp(None) is None
    assert parse_timestamp(True) is None  # bool is not an epoch


@pytest.mark.parametrize(
    "value,micros",
    [
        ("2026-08-05T11:19:56.34", 340000),  # the live shape: a trimmed trailing zero
        ("2026-08-05T11:19:56.3", 300000),
        ("2026-08-05T11:19:56.157", 157000),  # 3 digits: accepted natively
        ("2026-08-05T11:19:56.123456", 123456),  # 6 digits: accepted natively
        ("2026-08-05T11:19:56.1234567", 123456),  # 9 digits: truncated, not rejected
    ],
)
def test_a_fractional_second_of_any_width_still_parses(value, micros):
    """A FRACTIONAL SECOND OF THE WRONG WIDTH IS NOT AN UNPARSEABLE TIMESTAMP.

    Before Python 3.11 `fromisoformat` accepts exactly 3 or 6 fractional digits and
    rejects every other count — so a producer that trims its trailing zero sends
    `...T11:19:56.34` and the WHOLE value reads as no timestamp at all. Measured live: one
    of thirteen decoded administrative actions lost its instant that way and sorted to the
    end of the chronology, behind events two hours later, with nothing in the artifact
    saying why. The failure is invisible because a missing time reads as "the source did
    not record one".
    """
    dt = parse_timestamp(value)
    assert dt is not None, value
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) == (
        2026,
        8,
        5,
        11,
        19,
        56,
    )
    assert dt.microsecond == micros


@pytest.mark.parametrize(
    "value,offset_seconds",
    [
        ("2026-08-05T11:19:56.000+0000", 0),  # the live shape: Spark's own rendering
        ("2026-08-05T11:19:56+0000", 0),  # same, with no fraction at all
        ("2026-08-05T11:19:56.000+0530", 19800),  # a real offset, colon-less
        ("2026-08-05T11:19:56-0500", -18000),
        ("2026-08-05T11:19:56+02", 7200),  # hours only
        ("2026-08-05T11:19:56.000+00:00", 0),  # already legal: unchanged
    ],
)
def test_an_offset_without_its_colon_still_parses(value, offset_seconds):
    """NOR IS AN OFFSET WRITTEN WITHOUT ITS COLON, and this one is worse than the fraction.

    Before Python 3.11 `fromisoformat` requires `±HH:MM`, so Spark's default rendering —
    and therefore every Databricks `timestamp` column serialised to JSON — reads as no
    timestamp at all. One step further than a lost instant: a column whose values never
    parse scores `(0, 0)` in `_time_field_granularity` and is REJECTED as the source's time
    field, so the event time falls to whichever other candidate does parse. Measured over
    two live runs' evidence sidecars: 1200 of 1200 values shaped `...(.fff)?±HHMM` returned
    None, `creation_date_time` among them.
    """
    dt = parse_timestamp(value)
    assert dt is not None, value
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) == (
        2026,
        8,
        5,
        11,
        19,
        56,
    )
    assert dt.utcoffset().total_seconds() == offset_seconds


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2026-01-20", (2026, 1, 20)),  # a bare date ends in `-20`, not an offset
        ("2026-08-05 11:19:56", (2026, 8, 5)),
        ("2026/08/05 11:19:56", (2026, 8, 5)),
        ("76254", None),  # the id floor still holds underneath
    ],
)
def test_the_offset_normalisation_leaves_a_value_with_no_offset_alone(value, expected):
    """The rewrite requires a TIME OF DAY before the group, because a bare `2026-01-20`
    ends in `-20` — which a naive anchor turns into a `-20:00` offset and a date into a
    parse failure."""
    dt = parse_timestamp(value)
    if expected is None:
        assert dt is None
    else:
        assert dt is not None and (dt.year, dt.month, dt.day) == expected


def test_the_source_time_field_is_not_the_only_column_that_parses():
    """The consequence the parse fix exists for, asserted end to end.

    Two candidates, both carrying a time of day: the real event time in Spark's colon-less
    rendering, and a constant epoch placeholder written `Z`. While only the placeholder
    parsed it WAS the source's time field — which is how two live reports came to carry a
    `[0.60] Timestamp integrity defect` anomaly over `2000-01-01`.
    """
    from src.evidence import _detect_time_field

    rows = [
        {
            "creation_date_time": "2025-09-23T16:42:00.000+0000",
            "svc_note_creation_data_creation_time": "2000-01-01T16:42:00.000Z",
        }
    ]
    leaves = ["creation_date_time", "svc_note_creation_data_creation_time"]
    assert _detect_time_field("s", leaves, rows, []) == "creation_date_time"
    # A pack's declaration still outranks the measurement, in either column order.
    key = _Key("card", {"s": "id"}, {"s": "svc_note_creation_data_creation_time"})
    assert (
        _detect_time_field("s", list(reversed(leaves)), rows, [key])
        == "svc_note_creation_data_creation_time"
    )


def test_a_padded_fraction_does_not_eat_the_timezone():
    """The pad is anchored on the seconds field, so an offset survives it intact."""
    aware = parse_timestamp("2026-08-05T11:19:56.34+02:00")
    zulu = parse_timestamp("2026-08-05T11:19:56.34Z")
    assert aware is not None and zulu is not None
    assert aware.utcoffset().total_seconds() == 7200
    assert zulu.utcoffset().total_seconds() == 0
    assert aware.microsecond == zulu.microsecond == 340000


# --- chronology -------------------------------------------------------------


def test_build_chronology_orders_across_sources_and_epoch():
    logs = {
        "a": [{"ts": 1721853120000, "user": "U2", "action": "B"}],  # later
        "b": [{"ts": "2024-07-24T20:31:00Z", "user": "U1", "action": "A"}],  # earlier
        "c": [{"user": "U3", "action": "C"}],  # no timestamp -> sinks last
    }
    emap = {
        "a": {"user": "user"},
        "b": {"user": "user"},
        "c": {"user": "user"},
    }
    events, aggregated = build_chronology(logs, emap, [], ["U1", "U2", "U3"])
    assert not aggregated
    assert [e.source for e in events] == ["b", "a", "c"]  # time order, unparseable last
    assert events[0].actor == "U1" and events[0].action == "A"
    assert events[-1].epoch is None


def test_chronology_aggregates_when_over_volume():
    # One source well over the volume trigger -> collapse to (actor,action,bucket) groups.
    rows = [
        {"ts": 1721853060 + i, "user": "BOT", "action": "LOGIN"} for i in range(300)
    ]
    logs = {"big": rows}
    emap = {"big": {"user": "user"}}
    events, aggregated = build_chronology(logs, emap, [], [])
    assert aggregated
    # All 300 rows share actor+action and fall in one hour bucket -> one group.
    assert len(events) == 1
    assert events[0].count == 300
    assert events[0].actor == "BOT" and events[0].action == "LOGIN"


# --- decoded tables become events -------------------------------------------
# A decoded table (`encoded_fields`) nests events inside one row, and that row is a single
# event to every other reader. So a decode can succeed completely — the records present and
# reachable by `resolve_path` — and the narrated reconstruction still have nothing per-action
# to enumerate, because the chronology it walks reads rows and not decoded entries.


def _alert_row(alert_id, actions):
    """An alert row carrying a decoded action table, as `decode_logs` writes one."""
    return {
        "alertId": alert_id,
        "ir": {
            "recordId": alert_id,
            "decoded_actions": [dict(a) for a in actions],
        },
        "ts": "2026-08-05T11:22:34.095+00:00",
        "user": "0202YZ",
    }


_A1 = {"action": "EOU-OWN1/**-RCV9/**", "time": "2026-08-05T11:17:31.157"}
_A2 = {"action": "EOU-OWN2/**-RCV9/**", "time": "2026-08-05T11:18:26.277"}
_A3 = {"action": "EOU-OWN3/**-RCV9/**", "time": "2026-08-05T11:19:56.34"}


def test_each_decoded_record_becomes_its_own_event():
    logs = {"alerts": [_alert_row("a1", [_A1, _A2, _A3])]}
    emap = {"alerts": {"user": "user"}}
    events, _agg = build_chronology(
        logs, emap, [], ["0202YZ"], decoded_paths={"alerts": ["ir.decoded_actions"]}
    )
    # The parent row is STILL an event: the alert has its own instant and its own body,
    # and the records are what happened inside it. 1 + 3, not 3.
    assert len(events) == 4
    decoded = [e for e in events if e.entities.get("action")]
    assert len(decoded) == 3
    assert [e.entities["action"] for e in decoded] == [
        _A1["action"],
        _A2["action"],
        _A3["action"],
    ]
    # Ordered by the record's OWN instant, and every one of them has one — including the
    # trimmed-fraction value, which is why they sort at all.
    assert all(e.epoch is not None for e in decoded)
    assert [e.epoch for e in decoded] == sorted(e.epoch for e in decoded)


def test_a_decoded_record_inherits_the_acting_identity_and_the_subject_flag():
    """A nested record names WHAT was done, not who did it — so it inherits both.

    Subject-hood is inherited from the parent row as a whole rather than re-derived from
    the record's own values. Those values are masks, codes and instants that match no
    extracted identity, so testing them alone files the incident's own actions as
    background — and the render puts background second and caps it, which is how a decoded
    table reaches the artifact and still not the report.
    """
    logs = {"alerts": [_alert_row("a1", [_A1, _A2])]}
    emap = {"alerts": {"user": "user"}}
    events, _agg = build_chronology(
        logs, emap, [], ["0202YZ"], decoded_paths={"alerts": ["ir.decoded_actions"]}
    )
    decoded = [e for e in events if e.entities.get("action")]
    assert decoded and all(e.actor == "0202YZ" for e in decoded)
    assert all(e.is_subject for e in decoded)
    # And a record whose parent row is NOT about the subject stays background.
    other = _alert_row("a2", [_A1])
    other["user"] = "6004DD"
    events2, _ = build_chronology(
        {"alerts": [other]},
        emap,
        [],
        ["0202YZ"],
        decoded_paths={"alerts": ["ir.decoded_actions"]},
    )
    assert not any(e.is_subject for e in events2)


def test_two_identical_records_on_two_rows_are_one_event():
    """A re-notification carries the session's list again; a count of two reads as
    corroboration, which is the most expensive kind of wrong.

    Sized on the live shape: two retrieved alert rows holding thirteen records between
    them, of which ten are distinct.
    """
    logs = {
        "alerts": [
            _alert_row("a1", [_A1, _A2, _A3]),
            _alert_row("a2", [_A1, _A2, _A3, {**_A1, "time": "2026-08-05T11:17:37.402"}]),
        ]
    }
    emap = {"alerts": {"user": "user"}}
    events, _agg = build_chronology(
        logs, emap, [], ["0202YZ"], decoded_paths={"alerts": ["ir.decoded_actions"]}
    )
    decoded = [e for e in events if e.entities.get("action")]
    # 7 records, 4 distinct: the three repeats collapse, the differing timestamp does not.
    assert len(decoded) == 4
    assert len({tuple(sorted(e.entities.items())) for e in decoded}) == 4


def test_a_pack_declaring_no_decoded_path_is_unchanged():
    """The seam is opt-in, so every other pack's chronology is byte-identical."""
    logs = {"alerts": [_alert_row("a1", [_A1, _A2, _A3])]}
    emap = {"alerts": {"user": "user"}}
    base, agg = build_chronology(logs, emap, [], ["0202YZ"])
    none_paths, agg2 = build_chronology(logs, emap, [], ["0202YZ"], decoded_paths=None)
    empty, agg3 = build_chronology(logs, emap, [], ["0202YZ"], decoded_paths={})
    assert len(base) == 1  # the alert row, and nothing from the nested table
    assert [e.model_dump() for e in base] == [e.model_dump() for e in none_paths]
    assert [e.model_dump() for e in base] == [e.model_dump() for e in empty]
    assert agg is agg2 is agg3 is False


def test_a_decoded_columns_value_survives_the_trimmer_even_when_constant():
    """The trimmer's constant-column rule targets exactly the columns worth having.

    One grant repeated across thirteen actions has ONE receiver on all thirteen records, so
    `trim_columns` drops it as constant — which puts the report back where the decode found
    it: the value in hand, reported unavailable. `build_evidence` therefore marks every
    column under a declared decoded path relevant, read from the ROWS because a payload's
    header names its own columns.
    """
    rows = [
        _alert_row(
            f"a{i}",
            [
                {
                    "action": f"EOU-OWN{i}/**-RCV9/**",
                    "action_code": "EOU",
                    "owner_office": f"OWN{i}/**",
                    "receiver_office": "RCV9/**",  # constant across every record
                    "time": f"2026-08-05T11:17:{i:02d}.157",
                }
            ],
        )
        for i in range(6)
    ]
    emap = {"alerts": {"user": "user"}}
    paths = {"alerts": ["ir.decoded_actions"]}

    without = build_evidence({"alerts": rows}, {}, [], emap, ["0202YZ"])
    with_paths = build_evidence(
        {"alerts": rows}, {}, [], emap, ["0202YZ"], decoded_paths=paths
    )
    kept_without = [s for s in without.sources if s.source == "alerts"][0]
    kept_with = [s for s in with_paths.sources if s.source == "alerts"][0]
    col = "ir.decoded_actions.receiver_office"
    assert col in kept_without.dropped_columns  # the defect, still reproducible
    assert col in kept_with.kept_columns
    # And the constant column is genuinely constant, or this test proves nothing.
    assert kept_with.distributions.get(col, {}) == {"RCV9/**": 6}


# --- column trimming --------------------------------------------------------


def test_trim_columns_drops_constant_masked_blob_keeps_relevant():
    rows = [
        {
            "record": f"record{i}",  # discriminating identifier
            "env": "PROD",  # constant -> dropped
            "card": "************1234",  # masked -> dropped
            "blob": "x" * 400,  # blob -> dropped
            "status": "OK" if i % 2 else "ERR",  # discriminating categorical
        }
        for i in range(10)
    ]
    kept, dropped, dists = trim_columns(rows, relevant_fields={"record"})
    assert "record" in kept  # relevant always kept
    assert "env" in dropped  # constant
    assert "card" in dropped  # masked
    assert "blob" in dropped  # blob
    assert "status" in kept  # discriminating
    assert dists["status"] == {"OK": 5, "ERR": 5} or set(dists["status"]) == {
        "OK",
        "ERR",
    }


def test_trim_columns_keeps_masked_when_relevant():
    rows = [{"card": "****1234"} for _ in range(6)]
    kept, dropped, _ = trim_columns(rows, relevant_fields={"card"})
    assert "card" in kept and "card" not in dropped


def test_trim_columns_caps_at_max():
    rows = [{f"c{i}": f"v{i}{j}" for i in range(30)} for j in range(5)]
    kept, dropped, _ = trim_columns(rows, relevant_fields=set(), max_cols=12)
    assert len(kept) == 12
    assert len(dropped) >= 18


# --- actor attribution ------------------------------------------------------


def test_attribute_actors_rollup():
    events = [
        ChronologyEvent(
            timestamp="2024-07-24T20:31:00+00:00",
            epoch=1721853060.0,
            source="a",
            actor="U1",
            action="LOGIN",
            entities={"org_unit": "OFF1"},
        ),
        ChronologyEvent(
            timestamp="2024-07-24T20:35:00+00:00",
            epoch=1721853300.0,
            source="b",
            actor="U1",
            action="ISSUE",
            entities={"record": "P1"},
        ),
        ChronologyEvent(
            timestamp="2024-07-24T21:00:00+00:00",
            epoch=1721854800.0,
            source="a",
            actor="U2",
            action="LOGIN",
        ),
    ]
    rollups = attribute_actors(events)
    assert rollups[0].actor == "U1"  # most active first
    u1 = rollups[0]
    assert u1.event_count == 2
    assert u1.action_counts == {"LOGIN": 1, "ISSUE": 1}
    assert set(u1.sources) == {"a", "b"}
    assert u1.first_seen and u1.last_seen and u1.first_seen <= u1.last_seen
    assert u1.entities_touched.get("org_unit") == ["OFF1"]


def test_attribute_actors_subject_ranked_above_noise():
    """The incident's person-of-interest (matches an extracted entity value) is flagged
    is_subject and ranked ABOVE a higher-volume background actor that merely shares the
    time window — so containment targets the suspect, not innocent same-day logins."""
    events = [
        ChronologyEvent(
            timestamp="t",
            epoch=1.0,
            source="raw_access",
            actor="0201GP",
            action="R",
            entities={"record": "SUBJ03"},
        ),
    ] + [
        ChronologyEvent(
            timestamp="t",
            epoch=2.0 + i,
            source="auth_svc",
            actor="BSANJAY",
            action="SIGNIN",
        )
        for i in range(50)
    ]
    rollups = attribute_actors(events, entity_values=["0201GP", "USERNAMEX", "SUBJ03"])
    assert rollups[0].actor == "0201GP" and rollups[0].is_subject is True
    # The high-volume background actor is present but NOT a subject and ranked lower.
    assert rollups[1].actor == "BSANJAY" and rollups[1].is_subject is False


def test_attribute_actors_subject_via_touched_entity():
    """An actor is a subject if any entity it touched matches an incident value, even
    when the actor id itself doesn't (e.g. login id differs from the flagged sign)."""
    events = [
        ChronologyEvent(
            timestamp="t",
            epoch=1.0,
            source="raw_access",
            actor="LOGINX",
            action="R",
            entities={"record": "SUBJ03"},
        ),
    ]
    rollups = attribute_actors(events, entity_values=["SUBJ03"])
    assert rollups[0].is_subject is True


def test_a_duty_code_suffixed_actor_is_marked_is_subject():
    """A wider surface form of the alerted identifier is the SUBJECT, not background.

    Same defect as the scope line's "NOT named in the alert" count, one artifact along: the
    alert names a base identifier and the rows carry it with a role/duty suffix, so exact
    matching files the incident's own events as background — and the render puts background
    second and CAPS it, while this ranking is also what containment would target. Measured
    live on IR 30556527: every event of the alerted actor read ``is_subject: false`` while
    the verdict engine adjudicated the same rows as the subject's.
    """
    events = [
        ChronologyEvent(
            timestamp="t",
            epoch=1.0,
            source="auth_svc",
            actor="6001AAGS",  # the alerted identifier plus a two-character suffix
            action="SIGNIN",
        ),
    ] + [
        ChronologyEvent(
            timestamp="t",
            epoch=2.0 + i,
            source="auth_svc",
            actor="BSANJAY",
            action="SIGNIN",
        )
        for i in range(50)
    ]
    rollups = attribute_actors(events, entity_values=["6001AA"])
    by_actor = {r.actor: r for r in rollups}
    assert by_actor["6001AAGS"].is_subject is True
    assert by_actor["BSANJAY"].is_subject is False
    # And the subject outranks the higher-volume background actor.
    assert rollups[0].actor == "6001AAGS"

    # A subject reached through a TOUCHED entity, likewise (a login id differing from the
    # coded identifier is what makes the entity path the one that answers).
    touched = [
        ChronologyEvent(
            timestamp="t",
            epoch=1.0,
            source="raw_access",
            actor="LOGINX",
            action="R",
            entities={"user": "6001AAGS"},
        ),
    ]
    assert attribute_actors(touched, entity_values=["6001AA"])[0].is_subject is True


def test_an_unrelated_identifier_sharing_a_prefix_boundary_does_not_match():
    """The required negative: tolerance is a PREFIX relation, never a family resemblance.

    Without this the fix trades one wrong reading for a worse one — a whole population of
    identifiers sharing four leading characters read as the subject, which is the failure
    the discriminating-subject filter exists to prevent.

    The relation is deliberately SYMMETRIC, so a truncation of the alerted identifier that
    is still >=4 characters (``0013`` here) is a match and is not in this list: the alert may
    name the suffixed form while the source stores the base, and that direction is the same
    fact. What must not match is anything that is not a prefix at all.
    """
    events = [
        ChronologyEvent(
            timestamp="t",
            epoch=1.0,
            source="auth_svc",
            actor=actor,
            action="SIGNIN",
        )
        for actor in ("6001AB", "013AG", "XX6001AA", "0014AG")
    ]
    rollups = attribute_actors(events, entity_values=["6001AA"])
    assert not any(r.is_subject for r in rollups), [
        r.actor for r in rollups if r.is_subject
    ]

    # Nor does the chronology's own subject test admit them.
    logs = {"a": [{"user": a, "action": "SIGNIN"} for a in ("6001AB", "XX6001AA")]}
    chron, _ = build_chronology(logs, {"a": {"user": "user"}}, [], ["6001AA"])
    assert not any(e.is_subject for e in chron)
    # ...while the suffixed form does.
    logs2 = {"a": [{"user": "6001AAGS", "action": "SIGNIN"}]}
    chron2, _ = build_chronology(logs2, {"a": {"user": "user"}}, [], ["6001AA"])
    assert all(e.is_subject for e in chron2)


def test_detect_action_field_rejects_timestamp_columns():
    """A high-cardinality timestamp column carrying the 'event' token
    (event_timestamp) must NOT be chosen as the action field; the real low-cardinality
    verb (event_type) is picked instead. This kept action_counts from being polluted
    with one bucket per distinct timestamp."""
    from src.evidence import _detect_action_field

    rows = [
        {
            "event_timestamp": f"2026-07-24T08:16:{i:02d}.000Z",
            "event_type": "SIGNIN_LOCAL",
            "user_id": "BSANJAY",
        }
        for i in range(30)
    ]
    leaves = list(rows[0].keys())
    assert _detect_action_field("auth_events", leaves, rows) == "event_type"


def test_actor_field_falls_back_to_the_packs_own_name_token():
    """A store spelling its operator column with the DOMAIN's word still attributes.

    The engine's generic token list cannot contain every domain's word for an acting
    identity, and when no entity was mapped for the source the name tokens are the only
    signal left — so a column like `initiator_sign` in a domain whose actor entity is `user`
    matched nothing and every event on that source came back unattributed. The pack
    already declares `field_name_hints: {sign: user}` and `role: actor` on `user`, so the
    token is DERIVABLE; no second declaration, and the generic tokens are UNIONed, never
    replaced.
    """
    from src.evidence import _detect_actor_field

    leaves = ["created_at", "initiator_sign", "amount"]
    # No entity_map / resolved keys: the name-token path is the only one available.
    assert _detect_actor_field("s", leaves, {}, []) == ""
    assert (
        _detect_actor_field(
            "s",
            leaves,
            {},
            [],
            actor_entity_types=["user"],
            field_name_hints={"sign": "user", "locator": "record"},
        )
        == "initiator_sign"
    )
    # A hint pointing at a NON-actor type must not promote its column.
    assert (
        _detect_actor_field(
            "s",
            ["created_at", "record_locator"],
            {},
            [],
            actor_entity_types=["user"],
            field_name_hints={"locator": "record"},
        )
        == ""
    )
    # The generic tokens survive the union.
    assert (
        _detect_actor_field(
            "s",
            ["created_at", "username"],
            {},
            [],
            actor_entity_types=["courier"],
            field_name_hints={"badge": "courier"},
        )
        == "username"
    )


# --- build_evidence + determinism ------------------------------------------


def _sample_logs():
    return {
        "app": [
            {
                "user": "USERNAMEX",
                "org_unit": "LBV",
                "record": "SUBJ03",
                "ts": 1721853060000,
                "action": "ISSUE",
            },
            {
                "user": "USERNAMEX",
                "org_unit": "LBV",
                "record": "SUBJ03",
                "ts": 1721853120000,
                "action": "ISSUE",
            },
        ],
        "record_lake": [
            {"locator": "SUBJ03", "off": "LBV", "cdate": "2026-07-24", "status": "HK"},
        ],
    }


def _sample_map():
    return {
        "app": {"user": "user", "org_unit": "org_unit", "record": "record"},
        "record_lake": {"record": "locator", "org_unit": "off"},
    }


def test_build_evidence_deterministic():
    logs = _sample_logs()
    emap = _sample_map()
    keys = [_Key("record", {"app": "record", "record_lake": "locator"}, {"app": "ts"})]
    agg = {
        "total_records": 3,
        "cross_source_overlap": {"SUBJ03": {"app": 2, "record_lake": 1}},
    }
    p1 = build_evidence(logs, agg, keys, emap, ["USERNAMEX", "SUBJ03", "LBV"])
    p2 = build_evidence(logs, agg, keys, emap, ["USERNAMEX", "SUBJ03", "LBV"])
    assert p1.model_dump() == p2.model_dump()  # stable
    assert p1.total_records == 3
    assert any(a.actor == "USERNAMEX" for a in p1.actors)
    assert any(j["value"] == "SUBJ03" for j in p1.cross_source_joins)


def test_render_for_prompt_contains_sections_and_no_giant_json():
    logs = _sample_logs()
    pack = build_evidence(
        logs, {"total_records": 3}, [], _sample_map(), ["USERNAMEX", "SUBJ03"]
    )
    text = render_for_prompt(pack)
    assert "[CHRONOLOGY]" in text
    assert "[ACTOR ATTRIBUTION]" in text
    assert "[CROSS-SOURCE JOINS]" in text
    assert "USERNAMEX" in text
    # Not a raw per-row JSON dump.
    assert '[app] {"' not in text


def test_render_is_subject_first_and_bounds_background():
    """The render must lead with the subject's events + mark subject actors, and cap the
    same-window background so a busy population can't bury the incident (which also made
    the serving endpoint return empty JSON). One suspect + 300 unrelated logins."""
    logs = {
        "raw_access": [
            {"sign": "0201GP", "record": "SUBJ03", "act": "R", "ts": 1721853060 + i}
            for i in range(5)
        ],
        "auth": [
            {"uid": f"NOISE{i}", "act": "SIGNIN", "ts": 1721853000 + i}
            for i in range(300)
        ],
    }
    emap = {
        "raw_access": {"user": "sign", "record": "record"},
        "auth": {"user": "uid"},
    }
    pack = build_evidence(
        logs,
        {"total_records": 305},
        [],
        emap,
        ["0201GP", "SUBJ03"],
        subject_values=["0201GP", "SUBJ03"],
    )
    # The subject actor is flagged and ranked first despite the 300 noise logins.
    assert pack.actors[0].actor == "0201GP" and pack.actors[0].is_subject is True
    text = render_for_prompt(pack)
    # The chronology leads with a subject section and labels the bounded background.
    assert "Subject events" in text
    assert "Background activity" in text
    # The subject appears before the noise in the rendered text.
    assert text.index("0201GP") < text.index("NOISE")
    # Background is bounded (chrono cap + actor cap + capped distributions), nowhere
    # near a 300-row dump.
    assert text.count("NOISE") <= 100
    # The chronology section itself shows only a bounded background sample: at most
    # _MAX_BACKGROUND_CHRONO rows (each row names NOISE twice — actor + entity).
    chrono_section = text.split("[ACTOR ATTRIBUTION]")[0]
    assert chrono_section.count("\n2024-") <= 30  # bounded rows, not 300


# --- budget degradation -----------------------------------------------------


def test_degrade_to_budget_monotonic_and_fits():
    # Build a large pack that overflows a tiny budget.
    chrono = [
        ChronologyEvent(
            timestamp=f"2024-07-24T{h:02d}:00:00+00:00",
            epoch=1721800000.0 + h * 3600,
            source=f"s{h % 5}",
            actor=f"U{h}",
            action="ACT",
            entities={"record": f"P{h}", "org_unit": f"O{h}"},
        )
        for h in range(150)
    ]
    pack = EvidencePack(chronology=chrono, total_records=150)
    small = 500
    degraded = degrade_to_budget(pack, small)
    assert degraded.degraded is True
    assert degraded.notes  # recorded what it dropped
    # Callers always render WITH the budget (as anomaly_detection/report do), which
    # applies the final hard clip; structural degradation minimizes what's lost to it.
    assert len(render_for_prompt(degraded, small)) <= small


def _subject_and_background_pack(n_subject=13, n_background=400, bg_count=9):
    """A pack shaped like every real run: a few subject events, a busy background.

    The asymmetry is the point and it is not a fixture convenience — it is what production
    looks like. The subject acts a handful of times inside one session; a whole-population
    authentication log carries hundreds of events per identity in the same window. Measured
    on job 0184a3ce: 104 subject events against 61 background groups, the background groups
    counting up to 32 events each against the subject's 1–3.
    """
    chrono = [
        ChronologyEvent(
            timestamp="2026-08-05T11:17:00+00:00",
            epoch=1785928651.0 + i,
            source="alerts",
            actor="RSTUVW",
            action=f"A_SUBJECT_ACTION_{i}",
            entities={"receiver": "UUU1V21QR", "owner": f"CPT{i}"},
            is_subject=True,
        )
        for i in range(n_subject)
    ]
    chrono += [
        ChronologyEvent(
            timestamp="2026-08-05T11:00:00+00:00",
            epoch=1785928000.0 + i,
            source="population_auth",
            actor=f"BG{i:03d}",
            action="A_BACKGROUND_ACTION_OF_REALISTIC_LENGTH",
            entities={"user": f"BG{i:03d}", "office": f"OFF{i:03d}"},
            count=bg_count,
            is_subject=False,
        )
        for i in range(n_background)
    ]
    actors = [
        ActorRollup(actor="RSTUVW", event_count=n_subject, is_subject=True),
    ] + [
        ActorRollup(actor=f"BG{i:03d}", event_count=bg_count * 5, is_subject=False)
        for i in range(40)
    ]
    return EvidencePack(
        chronology=chrono,
        actors=actors,
        sources=[],
        total_records=n_subject + n_background,
    )


def test_the_daily_recollapse_keeps_the_person_of_interest_flag():
    """Rung 3 rebuilt every event WITHOUT `is_subject`, so the flag was erased wholesale.

    Two consequences, both of them the pack lying about what it holds rather than admitting
    a trim: `render_for_prompt` finds no subject events and files the entire chronology
    under "Background activity in the same window … not implicated", and the actor
    attribution loses every SUBJECT tag — the flag containment decisions are read from.
    And because `is_subject` was not in the grouping key either, a subject's action MERGED
    into a background actor's bucket whenever the two shared actor + action + day.

    Asserted on the count, not on presence: 13 subject events in must not become 0 out.
    """
    pack = _subject_and_background_pack()
    before = sum(1 for e in pack.chronology if e.is_subject)
    assert before == 13  # premise
    # A budget that forces rung 3 (the collapse) but leaves the pack recognisable.
    out = degrade_to_budget(pack, 5000)
    assert out.chronology_aggregated is True  # the rung really fired
    assert any("daily buckets" in n for n in out.notes)
    kept = [e for e in out.chronology if e.is_subject]
    assert kept, "every person-of-interest flag was erased by the collapse"
    assert sum(e.count for e in kept) == before
    # And the render says so, rather than calling the incident background.
    text = render_for_prompt(out, 5000)
    assert "# Subject events" in text


def test_the_top_50_cap_keeps_the_incident_not_the_busiest_background():
    """Rung 4 ranked by volume alone, which is precisely backwards.

    The subject acts a few times; a same-window population log carries hundreds of events
    per identity. `-count` therefore puts EVERY subject event below EVERY background one,
    so the rung that exists to fit the budget deletes the incident and keeps the noise.
    Measured on job 0184a3ce at the shipped 15,000-char budget: 104 subject events became
    20, and all five decoded administrative actions — the evidence the run exists to
    examine — were cut, leaving the report to state the action list was unavailable while
    holding it in the same pack.
    """
    pack = _subject_and_background_pack()
    out = degrade_to_budget(pack, 1500)  # deep enough to reach rung 4
    assert any("subjects first" in n for n in out.notes)
    assert len(out.chronology) <= 50
    subj = [e for e in out.chronology if e.is_subject]
    assert len(subj) == 13, "the subject's own events were dropped for busier background"
    # Same rule for the actor cap: the one subject must survive 40 busier identities.
    assert out.actors[0].actor == "RSTUVW" and out.actors[0].is_subject is True
    assert len(out.actors) <= 15


def test_a_pack_of_only_background_still_degrades_and_reports_it():
    """The subjects-first rule must not become a no-op ladder when nothing is a subject.

    With no person-of-interest to prefer, the ordering degenerates to the volume rank it
    always was — so the rung must still fire, still cap, and still say so.
    """
    pack = _subject_and_background_pack(n_subject=0)
    out = degrade_to_budget(pack, 1500)
    assert out.degraded is True
    assert len(out.chronology) <= 50
    assert any("subjects first" in n for n in out.notes)
    # Ranked by volume within the (single) group, as before.
    assert out.chronology[0].count >= out.chronology[-1].count


# --- truncation must be visible ---------------------------------------------


def _wide_pack(n_sources=9, n_events=400):
    """A pack whose render overflows any realistic budget, with distinguishable sources."""
    chrono = [
        ChronologyEvent(
            timestamp=f"2026-07-24T{h % 24:02d}:00:00+00:00",
            epoch=1721800000.0 + h,
            source=f"s{h % n_sources}",
            actor=f"U{h}",
            action="AN_ACTION_NAME_OF_REALISTIC_LENGTH",
            entities={"record": f"R{h}", "org_unit": f"O{h}", "asset": f"A{h}"},
            count=1,
            is_subject=(h < 60),
        )
        for h in range(n_events)
    ]
    sources = [
        SourceEvidence(
            source=f"source_number_{i}",
            record_count=100 + i,
            kept_columns=[f"col_{j}" for j in range(10)],
            distributions={
                f"col_{j}": {f"value_{k}": k for k in range(8)} for j in range(6)
            },
            examples=[{"payload": "x" * 150}],
            row_limited=(i == 3),
        )
        for i in range(n_sources)
    ]
    return EvidencePack(
        chronology=chrono, actors=[], sources=sources, total_records=1000
    )


def test_the_ladder_measures_the_pack_not_the_renderers_own_clip():
    """`degrade_to_budget` must fire at the SHIPPED budget.

    Its gates called `len(render_for_prompt(pack))` with no budget argument, so the
    renderer applied its DEFAULT budget and hard-clipped first: the length returned was
    `min(true_length, _DEFAULT_CHAR_BUDGET)`, and the then-shipped
    `correlation.evidence_char_budget` was exactly that default. Every gate compared
    `15000 <= 15000` and returned satisfied, making the whole ladder a no-op at the only
    budget anyone runs. Measured on job 4da14f65: a 21,828-char pack reported itself as
    fitting, `degraded` stayed False and the tail clip ate 5 of 9 sources in silence.
    """
    pack = _wide_pack()
    assert len(render_for_prompt(pack, 1 << 40)) > _DEFAULT_CHAR_BUDGET  # premise
    out = degrade_to_budget(pack, _DEFAULT_CHAR_BUDGET)
    assert out.degraded is True
    assert out.notes


def test_no_source_vanishes_from_an_over_budget_render():
    """A source dropped by the clip is indistinguishable from one never retrieved.

    The flat tail clip always landed inside [SOURCES] — it is the last section — so the
    first sources rendered complete and the rest disappeared with nothing saying so. Every
    source's count line must survive, detail lines being what gets sacrificed.
    """
    pack = _wide_pack()
    text = render_for_prompt(pack, 6000)
    assert len(text) <= 6000
    for i in range(9):
        assert f"- source_number_{i}:" in text
    # And the row-cap warning rides along with the count it qualifies.
    assert "TRUNCATED at the" in text


def test_zero_rows_is_rendered_as_the_answer_where_the_pack_declared_one():
    """A count of 0 cannot say whether the source answered or failed, so the pack does.

    `zero_rows.meaning` already discounts the stage-health score; it had never reached the
    text the LLM narrates FROM. On the live session-anomaly run a registered-automation lookup returned
    no rows — which is what "a human acted" LOOKS like, and the verdict read it as a PASS —
    and the narration then said in three separate passages that the check "did not return"
    and the actor's kind was UNKNOWN, contradicting the condition beside it.

    Asserted on the RENDER, because that string is the whole mechanism: the declaration is
    only worth carrying if the prompt shows it.
    """
    logs = {"robot_register": [], "alert_store": [], "sessions": [{"id": "A"}]}
    pack = build_evidence(
        logs,
        {},
        [],
        {},
        [],
        zero_row_meanings={
            "robot_register": "no row means the pair is not a registered robot",
            # Declared but NOT empty — must not be rendered: it describes a result that
            # did not happen, and printed there it reads as the finding.
            "sessions": "no session means the identity did not act",
        },
    )
    by_name = {s.source: s for s in pack.sources}
    assert by_name["robot_register"].zero_rows_meaning
    assert by_name["sessions"].zero_rows_meaning == ""
    # An empty source the pack said nothing about stays a bare count.
    assert by_name["alert_store"].zero_rows_meaning == ""

    text = render_for_prompt(pack, 1 << 40)
    assert "ZERO ROWS IS THE ANSWER HERE" in text
    assert "not a registered robot" in text
    assert "do NOT call what it establishes unknown" in text
    # The undeclared empty source is NOT dressed up as an answer.
    line = next(ln for ln in text.split("\n") if ln.startswith("- alert_store:"))
    assert "ZERO ROWS IS THE ANSWER" not in line


def test_the_notes_line_reports_the_trim_that_would_have_removed_it():
    """[NOTES] is assembled AFTER trimming, so it can name what the trim dropped.

    Emitted before, the one line that says what is missing was itself the first thing the
    tail clip removed — the record of the loss went out with the loss.
    """
    text = render_for_prompt(_wide_pack(), 6000)
    notes = text[text.index("[NOTES]") :]  # noqa: E203
    assert "chronology/attribution line(s) omitted" in notes
    assert "per-source detail line(s) omitted" in notes


def test_a_pack_that_fits_is_rendered_whole_with_no_trim_notes():
    """The sub-budget is a CEILING on the head, not a reservation.

    A run with a short chronology must still render every source's detail — the head share
    only binds when the head actually overflows it.
    """
    pack = _wide_pack(n_sources=2, n_events=3)
    text = render_for_prompt(pack, 1 << 40)
    assert "omitted to fit budget" not in text
    assert "[NOTES]" not in text
    assert "example: " in text  # detail intact
    # Same pack, generous-but-finite budget: still whole.
    assert render_for_prompt(pack, 60000) == text


# ---- the values a condition was decided on outlive every budget cut ---------------
# Every adjudicated value must survive every aggregation and clip rung. A budget that
# fires every rung must not drop the adjudicated-value line.


def _spec_reading(source, *fields):
    """A minimal ruleset naming `fields` of one logical source. No domain vocabulary."""
    return {
        "sources": {"target": source},
        "conditions": [
            {"id": "c1", "kind": "allowed", "source": "target", "field": f}
            for f in fields
        ],
    }


def _adjudicated_pack(n_rows=40, budget=1500):
    """An over-budget pack whose ruleset reads one column of one source."""
    logs = {
        "ledger": [
            {
                "row_id": f"R{i:04d}",
                "state_code": "OPEN" if i else "CLOSED",
                "filler": "x" * 120,
            }
            for i in range(n_rows)
        ],
        "noise": [{"other": f"N{i}", "blob": "y" * 150} for i in range(n_rows)],
    }
    spec = _spec_reading("ledger", "state_code")
    pack = build_evidence(logs, {}, [], {}, [], char_budget=budget, ruleset_spec=spec)
    return pack, budget


def test_the_value_a_condition_was_decided_on_survives_the_whole_ladder():
    """The defect: the ladder drops examples then distributions, so the only place a
    condition's column carried a VALUE is gone — while `kept=` still names the column, and
    the narrator reads that as "retrieved but not supplied" and denies the finding."""
    pack, budget = _adjudicated_pack()
    assert pack.degraded is True and pack.notes  # premise: the ladder ran
    ledger = next(s for s in pack.sources if s.source == "ledger")
    assert ledger.adjudicated_values.get("state_code")

    # Asserted on the RENDER, because that string is the whole mechanism.
    text = render_for_prompt(pack, budget)
    assert len(text) <= budget
    assert "CLOSED" in text, text
    line = next(ln for ln in text.split("\n") if ln.startswith("    * "))
    assert "state_code=" in line
    assert "never report them as missing or unavailable" in line


def test_a_column_no_condition_reads_is_still_degradable():
    """Scoped, not a blanket exemption: the ladder must still be able to shrink.

    The exemption covers the adjudicated READING only — the rungs still take this source's
    example rows and still cut its value tallies down, which is what keeps the ladder a
    ladder rather than a floor.
    """
    pack, _ = _adjudicated_pack()
    noise = next(s for s in pack.sources if s.source == "noise")
    assert noise.adjudicated_values == {}
    assert noise.examples == []
    assert "dropped per-source example rows to fit budget" in pack.notes

    # And at a budget tight enough to reach the rung that drops value tallies OUTRIGHT,
    # the tallies go from both sources and the adjudicated reading still stands — which is
    # the whole invariant, since that rung is where the live run lost it.
    tight, budget = _adjudicated_pack(budget=700)
    assert "dropped distributions" in tight.notes, tight.notes
    assert all(s.distributions == {} for s in tight.sources)
    ledger = next(s for s in tight.sources if s.source == "ledger")
    assert ledger.adjudicated_values.get("state_code")
    assert "state_code=" in render_for_prompt(tight, budget)


def test_a_pack_with_no_ruleset_promises_nothing():
    """No adjudicating procedure means no column was read off, so nothing is exempt and the
    ladder behaves exactly as it did before the exemption existed."""
    logs = {
        "ledger": [{"row_id": f"R{i}", "state_code": "OPEN"} for i in range(40)],
    }
    pack = build_evidence(logs, {}, [], {}, [], char_budget=1500)
    assert all(s.adjudicated_values == {} for s in pack.sources)
    assert "    * " not in render_for_prompt(pack, 1500)


def test_an_adjudicated_column_with_no_value_makes_no_promise():
    """A condition may name a field the rows do not carry — the `pack_validate` warning
    case. The line must then say nothing about it rather than print an empty reading."""
    logs = {"ledger": [{"row_id": f"R{i}"} for i in range(5)]}
    spec = _spec_reading("ledger", "state_code")
    pack = build_evidence(logs, {}, [], {}, [], ruleset_spec=spec)
    ledger = next(s for s in pack.sources if s.source == "ledger")
    assert "state_code" not in ledger.adjudicated_values


def test_a_column_whose_whole_population_is_blank_makes_no_promise():
    """The INVERSE defect, and it is reachable on real data: one live source's currency leaf
    is the empty string on 94,210 of 94,755 documents. Such a column is genuinely
    unavailable, so promising it would make the narrator assert a reading nobody made —
    exactly the failure this list exists to prevent, pointed the other way."""
    logs = {
        "ledger": [
            {"row_id": f"R{i}", "state_code": "", "other_code": "OPEN"}
            for i in range(20)
        ]
    }
    spec = _spec_reading("ledger", "state_code", "other_code")
    pack = build_evidence(logs, {}, [], {}, [], ruleset_spec=spec)
    ledger = next(s for s in pack.sources if s.source == "ledger")
    assert "state_code" not in ledger.adjudicated_values
    assert ledger.adjudicated_values.get("other_code") == ["OPEN"]  # the control


def _adj_line(pack, budget):
    """The one adjudicated-values line of a single-source pack, rendered at ``budget``."""
    return next(
        ln
        for ln in render_for_prompt(pack, budget).split("\n")
        if ln.startswith("    * ")
    )


def test_a_promise_spills_only_when_the_budget_cannot_afford_it_and_counts_what_it_drops():
    """The adjudicated-value line's allowance comes from the budget, not a fixed cap.

    With room, nothing spills and every adjudicated value is present. Without room:
    the dropped unit is a whole `col=vals` pair, the spill is counted, and the count
    matches what was shown."""
    n = 40
    logs = {
        "ledger": [
            {f"col_{i:02d}_with_a_longish_name": f"VALUE_{i:02d}" for i in range(n)}
            for _ in range(3)
        ]
    }
    spec = _spec_reading(
        "ledger", *[f"col_{i:02d}_with_a_longish_name" for i in range(n)]
    )
    pack = build_evidence(logs, {}, [], {}, [], char_budget=1 << 30, ruleset_spec=spec)
    ledger = next(s for s in pack.sources if s.source == "ledger")
    assert len(ledger.adjudicated_values) == n  # the model keeps them all

    # AMPLE. The whole rendered line is ~1.6k chars, so a budget that can seat it seats all
    # of it — at 1 << 30 the old constant still dropped 20 of the 40.
    ample = _adj_line(pack, 1 << 30)
    assert "further column(s) listed in kept=" not in ample
    for i in range(n):
        assert f"col_{i:02d}_with_a_longish_name=VALUE_{i:02d}" in ample

    # CONSTRAINED. 3,000 chars cannot, and the spill is governed.
    import re

    line = _adj_line(pack, 3000)
    assert "further column(s) listed in kept= were also read" in line
    shown = [p for p in line.split(": ", 1)[1].split("; ") if p.startswith("col_")]
    assert 0 < len(shown) < n
    assert all("=" in p and p.split("=")[1] for p in shown)  # no half pair
    spilled = int(re.search(r"\+(\d+) further", line).group(1))
    assert spilled == n - len(shown)


def test_one_wide_source_does_not_spend_the_promise_allowance_of_the_narrow_ones():
    """The allowance is COLLECTIVE, so it has to be shared, and two rules bound the sharing:
    no source is squeezed below what a fixed cap always gave it, and a source whose whole
    line is short keeps it whole however wide its neighbour is. Otherwise the fix trades the
    original defect for a positional one — the same value lost, now depending on which other
    source happened to be retrieved alongside it."""
    from src.evidence import _MIN_ADJ_LINE_LEN, _adj_line_caps

    wide = SourceEvidence(
        source="wide",
        record_count=3,
        adjudicated_values={f"a_long_column_name_{i:03d}": ["V"] for i in range(400)},
    )
    narrow = SourceEvidence(
        source="narrow", record_count=3, adjudicated_values={"status": ["OPEN"]}
    )
    plain = SourceEvidence(source="plain", record_count=3)

    caps = _adj_line_caps([wide, narrow, plain], 15000)
    assert "plain" not in caps  # nothing to promise, no allowance
    assert caps["narrow"] >= len("status=OPEN")  # kept whole
    assert caps["wide"] >= _MIN_ADJ_LINE_LEN  # never worse than the fixed cap
    # The narrow source used almost none of its share, so the leftover went to the wide one
    # rather than being reserved and wasted.
    assert caps["wide"] > 15000 * 0.25 / 2

    # And the floor outranks the ceiling: a budget too small to share still promises.
    tiny = _adj_line_caps([wide, narrow], 1000)
    assert tiny["wide"] >= _MIN_ADJ_LINE_LEN
    assert tiny["narrow"] >= len("status=OPEN")


def test_the_thinner_keeps_the_adjudicated_line_beside_the_count():
    """`_thin_source_lines` sacrifices per-source DETAIL to fit [SOURCES]. The adjudicated
    line is not detail: dropping it leaves the count line and the `kept=` list making a
    promise the render does not keep."""
    from src.evidence import _thin_source_lines

    lines = []
    for i in range(9):
        lines.append(f"- source_{i}: 100 record(s) kept=col_a,col_b")
        lines.append(f"    * values this run's conditions were decided on: col_a=V{i}")
        lines.append("    col_b: " + "z" * 200)
        lines.append("    example: " + "z" * 200)
    kept, note = _thin_source_lines(lines, 1200)
    assert note and "detail line(s) omitted" in note
    for i in range(9):
        assert f"- source_{i}: 100 record(s) kept=col_a,col_b" in kept
        assert f"    * values this run's conditions were decided on: col_a=V{i}" in kept
    assert not any(ln.startswith("    example:") for ln in kept)


# ---- a rung is charged to the half it shrinks -------------------------------------


def test_a_source_rung_is_not_charged_for_a_head_overflow():
    """The renderer sub-budgets, so dropping source detail cannot bring an oversized
    chronology back under its own ceiling. Gated on the total, the ladder spends every
    source rung on an overflow none of them can reach — and on job 562a60ad the head alone
    exceeded the whole budget, which made the gate unsatisfiable and the ladder
    all-or-nothing: the render came out 3,731 chars UNDER budget with every value gone."""
    pack = _subject_and_background_pack(n_subject=13, n_background=400)
    pack.sources = [
        SourceEvidence(
            source="small",
            record_count=3,
            kept_columns=["col_a"],
            distributions={"col_a": {"V1": 2, "V2": 1}},
            examples=[{"col_a": "V1"}],
        )
    ]
    head, srcs = _half_lens(pack)
    assert head > srcs * 5  # premise: the overflow is entirely in the head
    out = degrade_to_budget(pack, 4000)
    assert out.degraded is True
    # The head rungs did their work...
    assert out.chronology_aggregated is True
    assert len(out.chronology) <= 50
    # ...and the source half, which was never over its own cap, is untouched.
    assert out.sources[0].examples == [{"col_a": "V1"}]
    assert out.sources[0].distributions == {"col_a": {"V1": 2, "V2": 1}}
    assert "dropped distributions" not in out.notes


def test_a_head_rung_is_not_charged_for_a_source_overflow():
    """The mirror image: an oversized [SOURCES] block must not re-bucket a chronology that
    fits, which would erase per-event timestamps nothing was over budget for."""
    pack = _wide_pack(n_sources=9, n_events=4)
    head, srcs = _half_lens(pack)
    assert srcs > head * 3  # premise: the overflow is entirely in [SOURCES]
    out = degrade_to_budget(pack, 3000)
    assert out.degraded is True
    assert out.notes
    assert out.chronology_aggregated is False
    assert len(out.chronology) == 4
    assert "collapsed chronology to daily buckets" not in out.notes
    assert (
        "capped chronology to top-50 events and actors to top-15, subjects first"
        not in out.notes
    )


def test_the_last_rungs_note_does_not_claim_a_clip_the_renderer_did_not_make():
    """Rung 6 is a PREDICTION about the renderer, and the renderer has two ways to trim.

    It said "hard-clipped rendered evidence to budget" for as long as the renderer's only
    move was a tail clip. Since `_thin_source_lines` sub-budgets the halves, the usual loss
    is whole indented lines and the tail clip often does not run at all — measured on job
    562a60ad, where this note fired, `_CLIP_SUFFIX` was absent from the render, and the real
    loss (27 head lines) was reported by a different note. Both the LLM and the health scorer
    read this sentence, so it has to be true of whichever branch ran.
    """
    from src.evidence import _CLIP_SUFFIX

    pack = _wide_pack(n_sources=9, n_events=40)
    out = degrade_to_budget(pack, 3000)
    assert CLIP_NOTE in out.notes  # premise: the ladder reached its last rung
    text = render_for_prompt(out, 3000)
    assert not text.endswith(_CLIP_SUFFIX)  # premise: nothing was tail-clipped
    # What DID happen, reported by the renderer's own note rather than by this one.
    assert "line(s) omitted to fit budget" in text
    assert CLIP_NOTE in text
    assert "clip" not in CLIP_NOTE
