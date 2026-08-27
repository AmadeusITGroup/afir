"""`src/encoded_fields.py` — a pack-declared encoded payload becomes real columns.

WHAT THIS FILE IS GUARDING
==========================
An encoded field is the one shape where a source returns its richest evidence and the
investigation still does not see it: the base64 blob IS in the rows, every stage reports
success, and the report says the detail is unavailable. So the assertions here are all
about what a READER sees afterwards, never about the decode having "run":

* `resolve_path` must return the decoded values. That is the assertion, because writing the
  records to the literal flat key `"ir.decoded"` decodes perfectly and reaches NOBODY —
  `resolve_path`'s fast path asks a list of dicts for its scalar leaves and gets none. A test
  asserting `"ir.decoded" in row` passes on exactly that bug.
* `derive_schema` must list the decoded leaves as columns, since that is what makes them
  visible to the stages that read a schema rather than a row.
* An absent derived part is ABSENT, not `""` — an empty string reads downstream as a field
  that was checked and found empty.
* A source declaring nothing is byte-identical afterwards, which is what keeps this seam from
  being a behaviour change for every other pack.
"""

import base64

import pytest

from src.correlation import derive_schema, resolve_path
from src.encoded_fields import decode_logs, decoded_tables, records_at
from src.knowledge.pack import KnowledgePack, SourceDef

#: The live payload's shape, verbatim in the part that matters: the records are separated by
#: the two characters `\` `n` — an escaped newline nobody unescaped — so `splitlines()` on
#: the decoded text yields ONE record. Measured on job 0184a3ce (0 real newlines, 3 escaped).
ESCAPED_CSV = (
    "action,time"
    "\\nEOU-JJJ1K10UV/**-UUU1V21QR/**,2026-08-05T11:17:31.157"
    "\\nEOU-JJJ1K11UV/**-UUU1V21QR/**,2026-08-05T11:18:26.277"
)

#: The same table with real newlines — the shape the producer would send if the escaping were
#: ever fixed upstream. Both must decode identically, or the fix breaks the investigation.
REAL_NEWLINE_CSV = (
    "action,time\n"
    "EOU-JJJ1K10UV/**-UUU1V21QR/**,2026-08-05T11:17:31.157\n"
    "EOU-JJJ1K11UV/**-UUU1V21QR/**,2026-08-05T11:18:26.277"
)

ACTION_SPEC = {
    "field": "ir.attachmentContent",
    "encoding": "base64",
    "format": "delimited",
    "record_separator": ["\\n", "\n"],
    "field_separator": ",",
    "header": True,
    "into": "ir.decoded_actions",
    "derive": [
        {
            "from": "action",
            "separator": "-",
            "into": ["action_code", "owner_office", "receiver_office"],
            "remainder": "application_ids",
        }
    ],
}


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _pack(*specs, name="alerts", extra_sources=()):
    """A pack whose `alerts` source declares ``specs`` and nothing else."""
    sources = [SourceDef(name=name, encoded_fields=list(specs))]
    sources.extend(SourceDef(name=n) for n in extra_sources)
    return KnowledgePack(name="test", sources=sources)


def _row(payload: str):
    return {"alertId": "a1", "ir": {"recordId": "39056769", "attachmentContent": payload}}


# --- the reader's view ------------------------------------------------------------------


@pytest.mark.parametrize("text", [ESCAPED_CSV, REAL_NEWLINE_CSV], ids=["escaped", "real"])
def test_the_decoded_records_are_reachable_by_resolve_path(text):
    """THE assertion of this file: a condition's dotted path returns the decoded values.

    Both separator spellings must give the same answer. A pack cannot pin how a producer
    escapes its rows, and picking a separator that does not occur yields exactly one record
    — a whole table read as one malformed row, which tells a reader nothing while looking
    like a successful decode.
    """
    logs = {"alerts": [_row(_b64(text))]}
    assert decode_logs(logs, _pack(ACTION_SPEC)) == {"alerts": 2}

    row = logs["alerts"][0]
    assert resolve_path(row, "ir.decoded_actions.receiver_office") == [
        "UUU1V21QR/**",
        "UUU1V21QR/**",
    ]
    assert resolve_path(row, "ir.decoded_actions.owner_office") == [
        "JJJ1K10UV/**",
        "JJJ1K11UV/**",
    ]
    assert resolve_path(row, "ir.decoded_actions.action_code") == ["EOU", "EOU"]
    assert resolve_path(row, "ir.decoded_actions.time") == [
        "2026-08-05T11:17:31.157",
        "2026-08-05T11:18:26.277",
    ]


def test_the_records_are_nested_not_stored_under_a_flat_dotted_key():
    """The write target is a NESTED child, and this is the test that can tell.

    `row["ir.decoded_actions"] = [...]` decodes correctly and is invisible: `resolve_path`
    finds the exact flat key, asks a list of dicts for its terminal scalar leaves, and
    returns nothing. So assert the nesting directly as well as the round trip above — the
    two assertions fail in different places and only together name the cause.
    """
    logs = {"alerts": [_row(_b64(ESCAPED_CSV))]}
    decode_logs(logs, _pack(ACTION_SPEC))
    row = logs["alerts"][0]
    assert "ir.decoded_actions" not in row
    assert isinstance(row["ir"]["decoded_actions"], list)
    assert resolve_path(row, "ir.decoded_actions") == []  # a list of dicts has no leaves


def test_the_decoded_leaves_appear_in_the_derived_schema():
    """Stages that read a schema rather than a row must see the new columns too."""
    logs = {"alerts": [_row(_b64(ESCAPED_CSV))]}
    decode_logs(logs, _pack(ACTION_SPEC))
    leaves = derive_schema(logs)["alerts"]
    for leaf in (
        "ir.decoded_actions.action",
        "ir.decoded_actions.time",
        "ir.decoded_actions.action_code",
        "ir.decoded_actions.owner_office",
        "ir.decoded_actions.receiver_office",
    ):
        assert leaf in leaves


# --- the table as a table, not as more leaves -------------------------------------------


def test_records_at_returns_the_table_resolve_path_cannot():
    """The two readers answer different questions and BOTH answers are needed.

    `resolve_path` returns scalar LEAVES, and a list of dicts has none — so a consumer
    asking it for the table gets `[]` while the table is plainly there. That is the whole
    reason `records_at` exists as a separate function rather than a call into correlation.
    """
    logs = {"alerts": [_row(_b64(ESCAPED_CSV))]}
    decode_logs(logs, _pack(ACTION_SPEC))
    row = logs["alerts"][0]
    assert resolve_path(row, "ir.decoded_actions") == []
    recs = records_at(row, "ir.decoded_actions")
    assert [r["owner_office"] for r in recs] == ["JJJ1K10UV/**", "JJJ1K11UV/**"]
    # Absent, non-list and non-dict members are all `[]`/skipped rather than an error: a
    # reader of an undecoded row must get "no table", never a crash.
    assert records_at(row, "ir.nothing_here") == []
    assert records_at({"ir": {"x": "scalar"}}, "ir.x") == []
    assert records_at({"ir": {"t": [{"a": 1}, "junk", None]}}, "ir.t") == [{"a": 1}]


def test_decoded_tables_names_the_paths_a_decode_will_have_written():
    """The declaration's shape is known in ONE module, so a consumer cannot mis-derive it.

    Asserted against where `decode_logs` actually wrote, not against the literal, because
    the two agreeing is the only thing that makes this function usable — a consumer marking
    the wrong path relevant silently protects nothing.
    """
    pack = _pack(ACTION_SPEC, extra_sources=["other"])
    assert decoded_tables(pack) == {"alerts": ["ir.decoded_actions"]}
    logs = {"alerts": [_row(_b64(ESCAPED_CSV))]}
    decode_logs(logs, pack)
    for src, paths in decoded_tables(pack).items():
        for path in paths:
            assert records_at(logs[src][0], path), (src, path)
    # No declaration, no entry — not an empty list, which a caller would iterate happily
    # while believing a source declares a table.
    assert "other" not in decoded_tables(pack)
    assert decoded_tables(KnowledgePack(name="bare")) == {}
    assert decoded_tables(None) == {}


def test_the_default_into_path_is_the_one_reported():
    """A spec with no `into` still has a knowable path, and the two must not drift."""
    spec = {"field": "ir.attachmentContent", "header": True}
    pack = _pack(spec)
    assert decoded_tables(pack) == {"alerts": ["ir.attachmentContent_decoded"]}
    logs = {"alerts": [_row(_b64(ESCAPED_CSV))]}
    decode_logs(logs, pack)
    assert len(records_at(logs["alerts"][0], "ir.attachmentContent_decoded")) == 2


# --- what must NOT change --------------------------------------------------------------


def test_a_source_declaring_nothing_is_untouched():
    """The seam is opt-in per source, so every other pack's rows are byte-identical.

    Asserted on a source that carries a plausible base64 value: the decode must be driven
    by the DECLARATION and never by the engine guessing that a value looks encoded.
    """
    logs = {"other": [_row(_b64(ESCAPED_CSV))]}
    before = repr(logs)
    assert decode_logs(logs, _pack(ACTION_SPEC, extra_sources=["other"])) == {}
    assert repr(logs) == before


def test_no_pack_and_no_logs_are_no_ops():
    assert decode_logs({"alerts": [_row("x")]}, None) == {}
    assert decode_logs({}, _pack(ACTION_SPEC)) == {}


def test_a_second_pass_does_not_double_the_table():
    """Idempotent: a retry (or a resumed job) must not report 26 actions for 13."""
    logs = {"alerts": [_row(_b64(ESCAPED_CSV))]}
    assert decode_logs(logs, _pack(ACTION_SPEC)) == {"alerts": 2}
    assert decode_logs(logs, _pack(ACTION_SPEC)) == {}
    assert len(logs["alerts"][0]["ir"]["decoded_actions"]) == 2


def test_an_undecodable_payload_costs_only_that_field():
    """A bad payload is logged and skipped WITH the encoded value still on the row.

    The rest of the retrieval is evidence that was paid for; a decode failure must not
    discard it. And leaving the encoded value in place keeps the failure diagnosable from
    the exported evidence rather than only from a log line.
    """
    logs = {"alerts": [_row("not-base64-$$$"), _row(_b64(ESCAPED_CSV))]}
    assert decode_logs(logs, _pack(ACTION_SPEC)) == {"alerts": 2}
    assert "decoded_actions" not in logs["alerts"][0]["ir"]
    assert logs["alerts"][0]["ir"]["attachmentContent"] == "not-base64-$$$"
    assert len(logs["alerts"][1]["ir"]["decoded_actions"]) == 2


# --- derive: the parts, and the parts that are not there --------------------------------


def test_a_part_the_value_does_not_carry_is_absent_not_empty():
    """An absent part and an empty one are different findings.

    A check reading `receiver_office == ""` would report on a value that was never present;
    the honest answer is that the field has no value, which is what `unknown` is for.
    """
    logs = {"alerts": [_row(_b64("action,time\\nEOZ,2026-08-05T11:17:31.157"))]}
    decode_logs(logs, _pack(ACTION_SPEC))
    rec = logs["alerts"][0]["ir"]["decoded_actions"][0]
    assert rec["action_code"] == "EOZ"
    assert "owner_office" not in rec
    assert "receiver_office" not in rec
    assert resolve_path(logs["alerts"][0], "ir.decoded_actions.receiver_office") == []


def test_the_remainder_keeps_everything_past_the_named_parts():
    """A trailing application-id list is one part, not silently dropped."""
    payload = "action,time\\nEOU-OWN1/**-RCV1/**-APPA-APPB,2026-08-05T11:17:31.157"
    logs = {"alerts": [_row(_b64(payload))]}
    decode_logs(logs, _pack(ACTION_SPEC))
    rec = logs["alerts"][0]["ir"]["decoded_actions"][0]
    assert rec["receiver_office"] == "RCV1/**"
    assert rec["application_ids"] == "APPA-APPB"


def test_a_derive_rule_whose_column_is_absent_no_ops():
    """Two rules may name the two spellings one feed uses; the absent one must be inert.

    The live pack declares `from: action` (the current header) beside `from: name` (the
    legacy one) so a single declaration serves both feeds.
    """
    spec = dict(ACTION_SPEC)
    spec["derive"] = [
        {"from": "name", "separator": "-", "into": ["action_code"]},
        *ACTION_SPEC["derive"],
    ]
    logs = {"alerts": [_row(_b64(ESCAPED_CSV))]}
    decode_logs(logs, _pack(spec))
    rec = logs["alerts"][0]["ir"]["decoded_actions"][0]
    assert rec["action_code"] == "EOU"
    assert "name" not in rec


# --- format / header / cap variations ---------------------------------------------------


def test_headerless_columns_come_from_the_declaration():
    spec = {
        "field": "ir.attachmentContent",
        "header": False,
        "columns": ["name", "epoch"],
        "into": "ir.decoded_actions",
    }
    logs = {"alerts": [_row(_b64("EOU-OWN1/**-RCV1/**,1785928651157"))]}
    decode_logs(logs, _pack(spec))
    assert logs["alerts"][0]["ir"]["decoded_actions"] == [
        {"name": "EOU-OWN1/**-RCV1/**", "epoch": "1785928651157"}
    ]


def test_a_headerless_payload_with_no_declared_columns_names_by_position():
    """No header and no names: position is the only honest name available."""
    spec = {"field": "ir.attachmentContent", "header": False, "into": "ir.decoded"}
    logs = {"alerts": [_row(_b64("EOU,2026-08-05"))]}
    decode_logs(logs, _pack(spec))
    assert logs["alerts"][0]["ir"]["decoded"] == [
        {"column_1": "EOU", "column_2": "2026-08-05"}
    ]


def test_format_text_writes_the_decoded_string_whole():
    spec = {"field": "ir.attachmentContent", "format": "text", "into": "ir.body"}
    logs = {"alerts": [_row(_b64("a free-text note"))]}
    assert decode_logs(logs, _pack(spec)) == {"alerts": 1}
    assert resolve_path(logs["alerts"][0], "ir.body") == ["a free-text note"]


def test_encoding_none_reads_the_value_as_text():
    spec = {
        "field": "ir.attachmentContent",
        "encoding": "none",
        "record_separator": ["\n"],
        "into": "ir.decoded",
    }
    logs = {"alerts": [_row("a,b\n1,2")]}
    decode_logs(logs, _pack(spec))
    assert logs["alerts"][0]["ir"]["decoded"] == [{"a": "1", "b": "2"}]


def test_max_records_truncates_and_the_cap_is_declared_not_silent(caplog):
    """A truncated table that does not say so is read as the whole table."""
    payload = "action,time" + "".join(
        f"\\nEOU-OWN{i}/**-RCV1/**,2026-08-05T11:17:0{i}" for i in range(5)
    )
    spec = {**ACTION_SPEC, "max_records": 2}
    logs = {"alerts": [_row(_b64(payload))]}
    with caplog.at_level("WARNING"):
        assert decode_logs(logs, _pack(spec)) == {"alerts": 2}
    assert "max_records" in caplog.text and "LOWER BOUND" in caplog.text


def test_a_quoted_field_separator_inside_a_value_survives():
    payload = 'action,time\\n"EOU,X-OWN1/**-RCV1/**",2026-08-05'
    logs = {"alerts": [_row(_b64(payload))]}
    decode_logs(logs, _pack(ACTION_SPEC))
    rec = logs["alerts"][0]["ir"]["decoded_actions"][0]
    assert rec["action"] == "EOU,X-OWN1/**-RCV1/**"
    assert rec["time"] == "2026-08-05"


def test_a_non_mapping_intermediate_does_not_lose_the_records(caplog):
    """A row whose `ir` is a scalar cannot be nested into; the decode must still land.

    Writing the flat key is visible in the exported evidence, which is strictly better than
    dropping the table — and the warning names the path so the cause is diagnosable.
    """
    logs = {"alerts": [{"ir": "a string", "ir.attachmentContent": _b64(ESCAPED_CSV)}]}
    with caplog.at_level("WARNING"):
        assert decode_logs(logs, _pack(ACTION_SPEC)) == {"alerts": 2}
    assert len(logs["alerts"][0]["ir.decoded_actions"]) == 2
    assert "not a mapping" in caplog.text


def test_an_empty_or_missing_encoded_value_is_not_an_error():
    logs = {"alerts": [{"ir": {"attachmentContent": ""}}, {"ir": {}}, {}]}
    assert decode_logs(logs, _pack(ACTION_SPEC)) == {}
