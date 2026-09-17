"""A run's human name: fixed width, minted once, and never a claim the run cannot make."""

import re

import pytest

from src.incident_label import (
    DATE_WIDTH,
    ID_LENGTH,
    LABEL_WIDTH,
    PAD,
    PROCEDURE_WIDTH,
    SUBJECT_WIDTH,
    TAIL_WIDTH,
    build_label,
    new_incident_id,
    procedure_tokens,
    stamp_label,
)
from src.storage.base import safe_key


class FakeEntity:
    def __init__(self, etype, value):
        self.type = etype
        self.value = value


class FakeWindow:
    def __init__(self, start, end=""):
        self.start = start
        self.end = end or start


class FakeAnalysis:
    def __init__(self, summary="", entities=(), window=None):
        self.incident_summary = summary
        self.extracted_entities = list(entities)
        self.event_time = window
        self.initial_hypotheses = []
        self.key_investigation_areas = []


class FakePack:
    """Enough of a pack for the label: two rival procedures and two entity roles."""

    def __init__(
        self,
        keys=("alpha_rules", "beta_rules"),
        actors=("operator",),
        scopes=("unit",),
        subjects=None,
    ):
        self._keys = list(keys)
        self._actors = list(actors)
        self._scopes = list(scopes)
        self._subjects = dict(subjects or {})

    def ruleset_spec(self, key):
        return {"subject_entity": self._subjects.get(key, "")}

    def correlation_specs(self):
        return [
            {"use_case": "alpha", "title": "windmill inspection", "keys": ["shared_key"]},
            {"use_case": "beta", "title": "harbour dredging", "keys": ["shared_key"]},
        ]

    def ruleset_keys(self):
        return list(self._keys)

    def ruleset_key_for(self, use_case):
        return {"alpha": self._keys[0], "beta": self._keys[-1]}.get(use_case, "")

    def default_ruleset_key(self):
        return self._keys[0]

    def actor_entity_types(self):
        return list(self._actors)

    def scope_entity_types(self):
        return list(self._scopes)


def _fields(label):
    """The label split at its fixed offsets: (procedure, subject, date, tail)."""
    at = 0
    out = []
    for width in (PROCEDURE_WIDTH, SUBJECT_WIDTH, DATE_WIDTH, TAIL_WIDTH):
        end = at + width
        out.append(label[at:end])
        at = end + 1
    return tuple(out)


def _incident(**over):
    base = {"id": "9V6RK7QP2Z", "timestamp": "2026-09-15T10:11:12+00:00", "description": "x"}
    base.update(over)
    return base


def test_a_full_label_names_the_procedure_the_subject_the_date_and_the_id():
    analysis = FakeAnalysis(
        summary="a windmill inspection went wrong",
        entities=[FakeEntity("operator", "4417OP")],
        window=FakeWindow("2026-09-14T08:00:00Z"),
    )
    label, detail = build_label(FakePack(), analysis, _incident())
    procedure, subject, date, tail = _fields(label)
    assert procedure == "ALPH"
    assert subject == "4417OP" + PAD * 2
    assert date == "260914"
    assert tail == "9V6R"  # the id's LEADING characters, not its trailing ones
    assert "procedure=alpha_rules" in detail


def test_every_label_is_exactly_one_width_however_little_resolved():
    full = build_label(
        FakePack(),
        FakeAnalysis("a windmill inspection", [FakeEntity("operator", "4417OP")]),
        _incident(),
    )[0]
    # Nothing resolves: no pack, no analysis, and an id shorter than the tail field.
    empty = build_label(None, None, {"id": "A"})[0]
    assert len(full) == LABEL_WIDTH
    assert len(empty) == LABEL_WIDTH
    # A column of them aligns because the widths are the value's, not the renderer's.
    assert len({len(full), len(empty)}) == 1


def test_a_field_that_did_not_resolve_is_all_padding_and_says_why():
    label, detail = build_label(FakePack(), FakeAnalysis("a windmill inspection"), _incident())
    _, subject, date, _ = _fields(label)
    assert subject == PAD * SUBJECT_WIDTH
    assert "named no entity" in detail
    # The date is still there: an ingestion timestamp always exists, and which one it is is
    # the difference between "when it happened" and "when we heard".
    assert date == "260915"
    assert "ingestion date" in detail


def test_an_over_long_value_is_cut_to_its_field_and_never_overflows():
    analysis = FakeAnalysis(
        "a windmill inspection", [FakeEntity("operator", "abcdefghijklmnop")]
    )
    label, _ = build_label(FakePack(), analysis, _incident())
    assert _fields(label)[1] == "ABCDEFGH"
    assert len(label) == LABEL_WIDTH


def test_the_subject_is_an_actor_before_a_population_in_the_packs_own_order():
    analysis = FakeAnalysis(
        "a windmill inspection",
        # Declared order is what decides, not extraction order: the scope entity is first here.
        [FakeEntity("unit", "UNIT01"), FakeEntity("operator", "4417OP")],
    )
    assert _fields(build_label(FakePack(), analysis, _incident())[0])[1].startswith("4417OP")


def test_the_procedures_own_declared_subject_outranks_every_role():
    """The pack says what its procedure is about, and the verdict anchors on that same type.

    Measured on 80 stored runs: the type holding a card's issuer prefix is a declared SCOPE
    type and the type holding the card itself is declared by no role at all, so role order
    alone named the issuer as the subject of a case about one card — and the label then
    disagreed with the verdict it sits beside.
    """
    pack = FakePack(subjects={"alpha_rules": "card"})
    analysis = FakeAnalysis(
        "a windmill inspection",
        # Declaration order is against us on purpose: the scope type is extracted first.
        [
            FakeEntity("unit", "UNIT01"),
            FakeEntity("operator", "4417OP"),
            FakeEntity("card", "C7788"),
        ],
    )
    label, detail = build_label(pack, analysis, _incident())
    assert _fields(label)[1] == "C7788" + PAD * 3
    assert "subject=card C7788" in detail
    # A pack that declares no subject for the matched ruleset keeps the role ordering.
    plain = build_label(FakePack(), analysis, _incident())[0]
    assert _fields(plain)[1].startswith("4417OP")


def test_a_date_is_never_the_subject_because_the_next_field_is_the_date():
    """Two of the 80 stored runs read `ALPH-20260814-260814-…`: the date twice and no party."""
    analysis = FakeAnalysis(
        "a windmill inspection",
        [
            FakeEntity("window", "2026-08-14T09:01:00Z"),
            FakeEntity("window", "2026-01"),
            # An interval is a whole value made of nothing but two timestamps, which is the
            # form the stored runs actually carried — one pattern per value could not see it.
            FakeEntity("window", "2026-08-14T08:51:00Z/2026-08-14T08:52:00Z"),
            FakeEntity("window", "2026-08-14 to 2026-08-15"),
        ],
    )
    label, detail = build_label(FakePack(), analysis, _incident())
    assert _fields(label)[1] == PAD * SUBJECT_WIDTH
    assert "is a date, not a party" in detail
    # A party beside the date still wins — the guard skips a candidate, it does not stop the search.
    beside = FakeAnalysis(
        "a windmill inspection",
        [FakeEntity("window", "2026-08-14"), FakeEntity("operator", "4417OP")],
    )
    assert _fields(build_label(FakePack(), beside, _incident())[0])[1].startswith("4417OP")


def test_an_all_digit_identifier_is_not_read_as_a_date():
    """The guard is anchored and year-first, or every numeric account id becomes unnameable."""
    for value in ("700481523", "88450012997", "20260814", "4417OP"):
        analysis = FakeAnalysis("a windmill inspection", [FakeEntity("operator", value)])
        subject = _fields(build_label(FakePack(), analysis, _incident())[0])[1]
        assert subject.startswith(value[:SUBJECT_WIDTH]), value


def test_an_entity_of_no_declared_role_still_names_the_run():
    analysis = FakeAnalysis("a windmill inspection", [FakeEntity("reference", "RC-000123")])
    label, detail = build_label(FakePack(), analysis, _incident())
    assert _fields(label)[1] == "RC000123"
    assert "reference" in detail


def test_an_address_is_named_by_its_local_part():
    """The domain is shared by everyone on it, so it identifies nobody."""
    analysis = FakeAnalysis("a windmill inspection", [FakeEntity("operator", "jbrown@example.com")])
    assert _fields(build_label(FakePack(), analysis, _incident())[0])[1] == "JBROWN" + PAD * 2


def test_the_event_window_beats_the_ingestion_timestamp():
    entities = [FakeEntity("operator", "4417OP")]
    stated = build_label(
        FakePack(),
        FakeAnalysis("a windmill inspection", entities, FakeWindow("2026-01-02")),
        _incident(),
    )
    assert _fields(stated[0])[2] == "260102"
    assert "event date" in stated[1]


def test_a_procedure_no_scorer_matched_is_left_blank_rather_than_defaulted():
    """The pack's default still adjudicates; the label may not claim it was selected.

    A token here would be indistinguishable from a scored match, which is the one thing a
    run's name must not fake — and this is the state an operator most needs to see.
    """
    analysis = FakeAnalysis(
        "nothing in this sentence resembles either procedure",
        [FakeEntity("operator", "4417OP")],
    )
    label, detail = build_label(FakePack(), analysis, _incident())
    assert _fields(label)[0] == PAD * PROCEDURE_WIDTH
    assert "no procedure matched" in detail
    # Not the default's token, which is what a "fall back to the pack default" fix would print.
    assert "ALPH" not in label


def test_two_procedures_that_truncate_alike_get_different_tokens():
    pack = FakePack(keys=("refund_rules", "reference_rules", "other"))
    tokens = procedure_tokens(pack)
    assert len(set(tokens.values())) == 3
    assert tokens["other"] == "OTHE"
    # The shared prefix survives; only the last character discriminates, so the token is
    # still invertible by eye.
    assert all(t.startswith("REF") for t in (tokens["refund_rules"], tokens["reference_rules"]))
    assert all(len(t) == PROCEDURE_WIDTH for t in tokens.values())


def test_stamp_writes_the_label_and_never_renames_a_run():
    incident = _incident()
    analysis = FakeAnalysis("a windmill inspection", [FakeEntity("operator", "4417OP")])
    first = stamp_label(incident, FakePack(), analysis)
    assert incident["label"] == first
    assert incident["label_detail"]
    # A retried or resumed understanding stage runs this again, with a different analysis.
    # A rename would break every reference an operator has already written down.
    again = stamp_label(incident, FakePack(), FakeAnalysis("a harbour dredging", []))
    assert again == first
    assert incident["label"] == first


def test_stamp_survives_a_pack_that_cannot_answer():
    class Broken:
        def ruleset_keys(self):
            raise RuntimeError("no pack")

        def correlation_specs(self):
            raise RuntimeError("no pack")

        def actor_entity_types(self):
            raise RuntimeError("no pack")

        def scope_entity_types(self):
            raise RuntimeError("no pack")

    incident = _incident()
    label = stamp_label(incident, Broken(), FakeAnalysis("x", [FakeEntity("operator", "4417OP")]))
    assert len(label) == LABEL_WIDTH
    # The subject survives a pack that cannot state its roles: the fallback is extraction order.
    assert _fields(label)[1].startswith("4417OP")


def test_stamp_is_a_no_op_on_a_non_dict_incident():
    assert stamp_label(None, FakePack(), FakeAnalysis()) == ""


def test_a_minted_id_is_a_usable_storage_key_and_carries_no_ambiguous_letters():
    ids = {new_incident_id() for _ in range(200)}
    assert len(ids) == 200  # 32**10; a collision here is a broken generator, not luck
    for value in ids:
        assert len(value) == ID_LENGTH
        assert safe_key(f"fraud_report_{value}.pdf")
        # I/L/O/U are excluded so an id read off a screen cannot be typed as another run.
        assert not set(value) & set("ILOU")
        assert re.fullmatch(r"[0-9A-Z]+", value)


def test_two_ids_that_end_alike_still_get_different_tails():
    """A derived id carries its provenance as a SUFFIX, so the trailing characters are shared.

    Measured on the stored runs: every link child is `<parent>-link-1`, so a tail read off the
    end was `INK1` for all of them and three different incidents of one procedure over one
    subject on one day collapsed onto a single label.
    """
    analysis = FakeAnalysis("a windmill inspection", [FakeEntity("operator", "4417OP")])
    labels = {
        build_label(FakePack(), analysis, _incident(id=f"{parent}-link-1"))[0]
        for parent in ("2A9D6963", "49E3CF7D", "A5729413")
    }
    assert len(labels) == 3


@pytest.mark.parametrize("bad", [{}, {"id": ""}, {"id": None}])
def test_a_label_is_built_even_with_no_id_to_end_it(bad):
    label, _ = build_label(FakePack(), FakeAnalysis(), bad)
    assert len(label) == LABEL_WIDTH
    assert _fields(label)[3] == PAD * TAIL_WIDTH
