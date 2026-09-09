"""Shared builders for the open-question lane, over `knowledge/mock_domain/`.

Imported by `test_inquiries_never_change_the_verdict.py` and `test_inquiry.py`. Helper, not a
test file: collects nothing, imported as `tests.mock_domain_inquiries`. Nothing here names a
domain.

The declaration goes onto a COPY and never onto the shipped fixture pack, for the reason
`mock_domain_links.py` and the abstention round both found: `knowledge/mock_domain/` is
`installed_packs.FIXTURE_PACK`, so a key declared there is read by the whole shared suite and
would move output the rest of it asserts on. A fixture that has to change other tests to be
installed is not a fixture.
"""

import shutil

import yaml

from src.knowledge.pack import load_knowledge_pack
from src.utils.paths import REPO_ROOT

MOCK_DOMAIN_DIR = REPO_ROOT / "knowledge" / "mock_domain"

#: Source in neither the fixture pack's catalog nor any ruleset's `sources:` map, and retrieved
#: by nothing here. Its absence from `logs` is what leaves a question `not_asked` and therefore
#: probeable — the same role `mock_domain_links.PROBE_SOURCE` plays one lane over, and a separate
#: name because the two lanes must be arm-able independently.
PROBE_SOURCE = "handover_register"

#: Logical name of a source the fixture rulesets DO declare, so this run retrieves it and the
#: question is settled from rows in hand at no retrieval cost.
IN_HAND_SOURCE = "sessions"

#: How the declaration is scoped and sourced. Four entries because they are the four rungs a
#: reader has to be able to tell apart, and `!= "free"` would collapse three of them.
ASK_MODES = ("free", "probe", "unscopable", "sourceless")

#: Entity type the fixture rulesets adjudicate, and the one the incident always carries.
SCOPE_ENTITY = "shipment"

#: Entity type declared in the fixture glossary that no test incident carries a value of, so a
#: question scoped by it is `unreachable` rather than merely unanswered.
ABSENT_SCOPE_ENTITY = "refund_claim"

#: The three outcomes a declaration must label, in the words the report prints back. Every entry
#: this helper builds carries all three: the accessor DROPS an entry missing any of them, and a
#: dropped declaration is indistinguishable from a pack that declared nothing.
MEANINGS = {
    "rows": (
        "the handover was recorded elsewhere, so the gap in this procedure's own evidence has an "
        "explanation a human should read before treating it as an omission"
    ),
    "empty": (
        "nothing recorded a handover for this subject anywhere, which is what the procedure's own "
        "silence would mean if it were the only source asked"
    ),
    "unanswered": (
        "whether a handover was recorded elsewhere is still unknown: the source was asked and said "
        "nothing at all, so this is a credential or catalog gap and not a finding"
    ),
}


def open_question(
    question_id="was_the_handover_recorded_elsewhere",
    mode="free",
    condition="",
    verdict_class="",
    result="unknown",
    where=None,
    question="Was a handover recorded for {entity} {value} outside this procedure's own sources?",
    note="",
):
    """One `open_questions:` entry, in the shape `pack.open_questions` reads back.

    `mode` selects the rung (see :data:`ASK_MODES`): `free` names a source the ruleset declares
    and this run therefore holds, `probe` one nothing retrieved, `unscopable` a scope type no
    incident carries, `sourceless` names no source at all — which the accessor DROPS, and that
    is the point of having it here rather than in one test.
    """
    ask = {
        "question": question,
        "scope_entity": (
            ABSENT_SCOPE_ENTITY if mode == "unscopable" else SCOPE_ENTITY
        ),
    }
    if mode == "free":
        ask["source"] = IN_HAND_SOURCE
    elif mode in ("probe", "unscopable"):
        # Physical name, deliberately outside the ruleset's `sources:` map: naming it there would
        # make it a hard dependency retrieved on every run, which is the opposite of this lane.
        ask["source"] = PROBE_SOURCE
    if where:
        ask["where"] = list(where)
    when = {"result": result}
    if condition:
        when["condition"] = condition
    if verdict_class:
        when["verdict_class"] = verdict_class
    entry = {"id": question_id, "when": when, "ask": ask, "meaning": dict(MEANINGS)}
    if note:
        entry["note"] = note
    return entry


def inquiry_pack(tmp_path, entries, use_case="refund_fraud", label=""):
    """Fixture pack copy whose `use_case` ruleset declares `entries` as its open questions.

    `label` distinguishes two copies built from the same entry ids in one test; without it the
    second call reuses the first tree, which is the caching `mock_domain_links.probe_pack`
    relies on and a surprise wherever the entries differ.
    """
    ids = "_".join(str(e.get("id", "")) for e in entries) or "none"
    root = tmp_path / f"inquiry_pack_{use_case}_{label or ids}"
    if not root.exists():
        shutil.copytree(MOCK_DOMAIN_DIR, root)
    rules = root / "use_cases" / use_case / "rules.yaml"
    data = yaml.safe_load(rules.read_text(encoding="utf-8")) or {}
    spec = (data.get("verdicts") or {})[use_case]
    spec["open_questions"] = [dict(e) for e in entries]
    rules.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_knowledge_pack(root)


def register_row(**over):
    """One probe-source row recording a handover for the subject the fixtures adjudicate.

    Carries the subject value: the settlement applies the declaration's own `where` to these
    rows, and a row naming no subject answers a question about somebody else.
    """
    row = {
        "shipment_code": "RT48192043",
        "depot_code": "LDS04",
        "handover.recorded": True,
        "handover.recorded_at": "2026-07-20T09:14:00Z",
    }
    row.update(over)
    return row
