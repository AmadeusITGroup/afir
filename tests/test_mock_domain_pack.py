"""The GENERIC engine, proven against `knowledge/mock_domain/` — the pack that ships on main.

A domain pack's tests prove that pack encodes its procedure correctly, over real production rows.
This file proves what a pack test structurally cannot: that the ENGINE holds no procedure. It runs
the same `evaluate_verdict`, `UseCaseAnalyzer` and pack loader against an invented domain — parcel
courier refund fraud — and asserts every stage still works and no sentence in the output names
another domain.

Writing this pack surfaced five defects every domain regression had missed, all of one species, an
engine detail that happened to be true of one domain:

  1. `field_flag` returned `unknown` on any NESTED boolean, because `resolve_path` drops bools by
     design (`_scalar` excludes them so a boolean is never mistaken for a join key).
  2. `record_absence` PASSED on every row for a BOOLEAN forbidden vocabulary, so the check could
     not fire and reported "no forbidden value" over rows where the flag was set on all of them.
  3. `route_membership` hard-coded a 3-letter location-code regex, so a domain whose scope point
     is a 5-character depot code got `unknown` on its scope GATE.
  4. Four evaluators wrote domain literals into their notes — one domain's facts on every
     domain's report.
  5. `projection_guard` probed with `_collect`, so a probe path naming a BOOLEAN could never be
     satisfied and a fully-evidenced verdict was marked `degraded`.

Each is invisible in the shipped domain, and each produces a check that CANNOT fire and is
indistinguishable in the report from one that had nothing to find.

The fixtures are synthetic on purpose. `tests/fixtures/*.json.gz` are real production rows,
committed because a fixture that isn't cannot pin behaviour; these rows exist to exercise the
engine's dispatch, so they are hand-built to this pack's declared shapes and hold nothing real.
"""

import pathlib
import re
import tempfile
from types import SimpleNamespace

import pytest

from src.correlation import evaluate_verdict
from src.knowledge.pack import load_knowledge_pack
from src.models.pydantic_models import ExtractedEntity
from src.usecases.base import UseCaseAnalyzer
from src.utils.paths import REPO_ROOT

MOCK_DOMAIN_DIR = REPO_ROOT / "knowledge" / "mock_domain"

#: Words the engine may use that this pack's own files never happen to spell. The anti-leak
#: assertions below are CLOSED-WORLD, so this is the residue: arithmetic and set-relation
#: vocabulary, measured over both rulesets on five row shapes with the pack's wording stripped.
#: A word joining it must be one no domain could own — a plausible-sounding noun is a real leak.
_ENGINE_ARITHMETIC = frozenset(
    {
        "allow",
        "disallowed",
        "distinct",
        "equal",
        "common",
        "parseable",
        "checked",
        # The shape of a two-sided comparison (`field_compare`'s left/right).
        "side",
        "sides",
        "carry",
        # `velocity_count` is an engine CONDITION KIND, so the note naming it is the engine
        # saying which arithmetic it performed — not a domain noun.
        "velocity",
        # `resolve_path` reports which candidate paths it tried.
        "paths",
    }
)


def _pack_vocabulary(pack_dir=MOCK_DOMAIN_DIR):
    """Every word this pack's own files spell, from its LIVE content only.

    COMMENTS ARE STRIPPED, and that is load-bearing rather than tidiness. A pack comment is
    prose written by the author to explain a decision, and it routinely mentions other domains
    to do so — one line of this pack's `rules.yaml` says "an exploded itinerary scatters the
    codes", which is a perfectly good explanation and would silently ADD `itinerary` to the set
    of words the engine is then allowed to emit. A leak-detector whose allowlist grows every
    time someone writes a comment is not a detector. Only what the engine can actually READ
    counts, so only live YAML/markdown body text is harvested.
    """
    words = set()
    for path in sorted(pack_dir.rglob("*")):
        if not path.is_file() or path.suffix not in {".yaml", ".yml", ".md"}:
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            live = line.split("#", 1)[0] if path.suffix != ".md" else line
            words |= {w.lower() for w in re.findall(r"[A-Za-z_]{2,}", live)}
    return words


def _foreign_words(text, pack_words):
    """The words in `text` that neither the pack nor the engine's arithmetic accounts for.

    This is the whole anti-leak mechanism, and it is closed-world BY DESIGN. The list it
    replaced enumerated one real domain's nouns — which meant the check could only ever catch
    a leak from *that* domain, it went stale the moment the engine learned a new word, and
    (the reason it had to go) it made this file name a domain it has no business knowing.
    Asking "did this word come from the pack?" needs no such list and catches every domain.
    """
    return sorted(
        {
            w.lower()
            for w in re.findall(r"[A-Za-z_]{2,}", text)
            if w.lower() not in pack_words and w.lower() not in _ENGINE_ARITHMETIC
        }
    )


@pytest.fixture(scope="module")
def pack():
    return load_knowledge_pack(MOCK_DOMAIN_DIR)


@pytest.fixture(scope="module")
def refund_spec(pack):
    return pack.ruleset_spec("refund_fraud")


@pytest.fixture(scope="module")
def collusion_spec(pack):
    return pack.ruleset_spec("courier_collusion")


# --- row builders ------------------------------------------------------------
# Shapes taken from `knowledge/mock_domain/schemas/shipment_ledger.yaml`, so a schema doc that
# drifts from what the checks read shows up here.


def ledger_row(**over):
    """One shipment_ledger row: bare, unreviewed, refunded, on an in-scope route."""
    row = {
        "tracking_code": "RT48192043",
        "depot_code": "LDS04",
        "handler.badge": "0192C",
        "handler.device_login": "jdunne",
        "created_at": "2026-07-20T08:00:00Z",
        "element_counters.INS": 0,
        "element_counters.SIG": 0,
        # PRESENT AND EMPTY, which is a different answer from absent: an absence check may
        # only claim "absent" for a path it actually read. `scans: []` resolves (a real 0);
        # omitting the key entirely would correctly yield `unknown`.
        "scans": [],
        "pod.signature_captured": False,
        "pod.captured_by": "0192C",
        "refund.amount": 41.5,
        "refund.reason_code": "LOST",
        "refund.manually_reviewed": False,
        "refund.approved_by": None,
        "refund.claimed_at": "2026-07-20T15:00:00Z",
        "route.origin": "LDS04",
        "route.destination": "BHM11",
    }
    row.update(over)
    return row


SERVICED = {
    "element_counters.INS": 1,
    "element_counters.SIG": 1,
    "scans": [
        {"scan": "PU", "scanned_at": "2026-07-20T09:00:00Z", "scanned_by": "0192C"}
    ],
}
REVIEWED = {"refund.manually_reviewed": True, "refund.approved_by": "SUP9"}


def session_row(**over):
    row = {
        "depot_code": "LDS04",
        "login": "jdunne",
        "badge": "0192C",
        "device_id": "TERM-1",
        "session.automated": True,
        "session.started_at": "2026-07-20T07:55:00Z",
    }
    row.update(over)
    return row


def alert_row(**over):
    row = {
        "alert.incident_id": "IR-REF-0001",
        "alert.depot_code": "LDS04",
        "alert.courier_badge": "0192C",
        "alert.device_login": "jdunne",
        "alert.claim_id": "CLM-9001",
        "alert.claimed_shipments": ["RT48192043", "QQ00771265"],
        # Nested and repeated, one element per claimed shipment, each naming its own handler:
        # what `subject_discovery` reads. The two flat lists above cannot say which handler
        # goes with which shipment.
        "alert": {
            "claim_lines": [
                {"shipment_code": "RT48192043", "handler_badge": "0192C"},
                {"shipment_code": "QQ00771265", "handler_badge": "7741B"},
            ]
        },
        "alert.raised_at": "2026-07-20T16:00:00Z",
    }
    row.update(over)
    return row


def analysis(shipment="RT48192043"):
    return SimpleNamespace(
        incident_summary="Burst of refund claims in one depot",
        initial_hypotheses=[],
        key_investigation_areas=[],
        correlation_keys=[],
        extracted_entities=[
            ExtractedEntity(type="shipment", value=shipment),
            ExtractedEntity(type="depot", value="LDS04"),
            ExtractedEntity(type="courier", value="0192C"),
        ],
        event_time=None,
    )


def verdict_for(spec, ledger, sessions=None, alerts=None, shipment="RT48192043"):
    logs = {"shipment_ledger": ledger, "device_sessions": sessions or []}
    if alerts is not None:
        logs["refund_alerts"] = alerts
    return evaluate_verdict(spec, logs, analysis(shipment))


def label(v):
    assert v is not None and v.subjects, "the ruleset produced no verdict at all"
    return v.subjects[0].verdict


def check(v, cid):
    return next(c for c in v.subjects[0].checks if c.id == cid)


# --- the pack loads ----------------------------------------------------------


def test_the_mock_domain_pack_loads_completely(pack):
    """Every folder a pack may ship is present and non-empty here.

    The template documents what CAN exist; this asserts one pack where all of it DOES, so a
    loader change that quietly stops reading a folder fails a test rather than producing a
    thinner pack nobody notices.
    """
    assert [e.type for e in pack.entities] == [
        "shipment",
        "courier",
        "depot",
        "refund_claim",
        "incident_ref",
    ]
    assert [s.name for s in pack.sources] == [
        "refund_alerts",
        "shipment_ledger",
        "device_sessions",
    ]
    assert sorted(pack.shared_checks) == [
        "actor_identity/automated_terminal_session",
        "actor_identity/pod_captured_by_handler",
        "shipment_elements/manually_reviewed",
        "shipment_elements/no_servicing_elements",
        "shipment_elements/signature_captured",
        "shipment_elements/single_handler",
    ]
    assert sorted((pack.rulesets.get("verdicts") or {})) == [
        "courier_collusion",
        "refund_fraud",
    ]
    assert sorted(pack.pack_data) == ["refund_reason_map"]
    assert len(pack.schema_documents) == 1  # one document per TABLE
    assert len(pack.playbook_documents) == 2
    assert sorted((pack.reporting.get("use_cases") or {})) == ["refund_fraud"]


def test_no_foreign_domain_vocabulary_survives_into_a_verdict(refund_spec):
    """Every string the verdict engine emits is either the pack's or arithmetic.

    Closed-world: harvest the words this pack spells, add the measured arithmetic terms
    (`_ENGINE_ARITHMETIC`), require that nothing else appears. A word from neither source came
    from `src/`, so it appears on every other domain's report too — which is what makes this
    general where an enumerated ban-list catches only the leaks somebody thought of.

    Checked on every path and not just the happy one: `unknown` is where a fallback string is
    most likely to be forgotten, and it is the branch a new domain hits first.
    """
    pack_words = _pack_vocabulary()
    rows = [
        [ledger_row()],
        [ledger_row(**SERVICED)],
        [ledger_row(**REVIEWED)],
        [ledger_row(**{"route.destination": "ZZZ99"})],
        # Nothing resolves: every check takes its `unknown` branch.
        [{"tracking_code": "RT48192043"}],
    ]
    for ledger in rows:
        v = verdict_for(refund_spec, ledger)
        for c in v.subjects[0].checks:
            text = " ".join([c.detail, c.expected, c.label])
            foreign = _foreign_words(text, pack_words)
            assert not foreign, f"{c.id} ({c.result}) leaked {foreign}: {text}"


@pytest.mark.parametrize("use_case", ["refund_fraud", "courier_collusion"])
def test_the_engines_own_fallback_notes_name_no_domain(pack, use_case):
    """The assertion above is not enough on its own, hence the duplication.

    This pack declares `pass_detail`/`fail_detail`/`unknown_detail` on most of its checks, so the
    engine's own fallback strings never execute and a domain literal re-introduced into one of
    them passes the scan above untouched. Verified by mutation: restoring `"itinerary is an
    in-scope record-misuse route"` as `route_membership`'s fallback left every other test here green.

    So the notes are STRIPPED, forcing every evaluator down its fallback branch. Same closed-world
    rule, and the residue is condition IDs and field paths (`refund_within_window`,
    `pod.captured_by`) — the pack's own identifiers, and the only way an unworded check can say
    what it checked.
    """
    pack_words = _pack_vocabulary()
    spec = dict(pack.ruleset_spec(use_case))
    spec["conditions"] = [
        {
            k: v
            for k, v in cond.items()
            # Every key through which a pack supplies WORDS, `expected`/`expected_label`
            # included: leaving those in scans the ruleset rather than the engine.
            if k
            not in (
                "pass_detail",
                "fail_detail",
                "unknown_detail",
                "label",
                "expected",
                "expected_label",
            )
        }
        for cond in spec["conditions"]
    ]
    row_sets = [
        [ledger_row()],
        [ledger_row(**SERVICED, **REVIEWED)],
        [ledger_row(**{"route.destination": "ZZZ99", "pod.captured_by": "0999Z"})],
        [{"tracking_code": "RT48192043"}],
    ]
    seen = 0
    for ledger in row_sets:
        for sessions in ([], [session_row()]):
            v = evaluate_verdict(
                spec,
                {"shipment_ledger": ledger, "device_sessions": sessions},
                analysis(),
            )
            for c in v.subjects[0].checks:
                seen += 1
                for field in (c.detail, c.expected, c.label):
                    foreign = _foreign_words(str(field), pack_words)
                    assert not foreign, (
                        f"{use_case}/{c.id} ({c.result}) fallback names a domain: "
                        f"{foreign} in {field!r}"
                    )
    assert seen, "no checks were evaluated, so nothing was actually scanned"


def test_the_leak_detector_can_still_fail():
    """A mutation test on the two assertions above, which pass by finding NOTHING.

    Every way the closed-world check could break yields an empty result and a green test: a
    `_pack_vocabulary` that swallowed the dictionary, a tokeniser that matched nothing, an
    `_ENGINE_ARITHMETIC` grown until it covers any word. So the detector is pointed at four known
    leaks, in the vocabulary they shipped in, and must object to each. Text rather than a spec,
    because what is under test is the detector.
    """
    pack_words = _pack_vocabulary()
    for leaked in (
        "the account is a automated account",
        "the legs are an in-scope airline itinerary",
        "creation or ticketing time not available",
        "no record was retrieved for the booking",
    ):
        assert _foreign_words(leaked, pack_words), f"the detector no longer objects to {leaked!r}"

    # ...and not to what the engine legitimately says: a detector that flags everything is
    # worthless in the other direction.
    assert not _foreign_words("the two sides carry the same value", pack_words)
    assert not _foreign_words("no rows were retrieved from the source", pack_words)

    # A pack COMMENT must not widen the allowlist, asserted on the mechanism rather than on a
    # word that is in a comment today: pinning a real word is self-defeating, since rewording
    # that one comment turns the test green forever while proving nothing.
    with tempfile.TemporaryDirectory() as tmp:
        probe = pathlib.Path(tmp)
        (probe / "entity_glossary.yaml").write_text(
            "# zzcommentword is mentioned only in this comment\nentities: []\n"
        )
        harvested = _pack_vocabulary(probe)
    assert "zzcommentword" not in harvested, (
        "a pack COMMENT reached the allowlist — the leak-detector now permits whatever an "
        "author happens to mention while explaining a decision"
    )
    assert "entities" in harvested, "the harvester stopped reading live YAML altogether"


# --- the shared check library ------------------------------------------------


def test_both_rulesets_resolve_their_use_imports(refund_spec, collusion_spec):
    """A `use:` that survived to here would be an unevaluated check, not an error.

    An unresolvable import is fatal at LOAD (`test_knowledge_pack.py` pins that); this
    asserts the resolved end of the same contract — every condition reaching the engine is a
    real condition with a `kind` it can dispatch on.
    """
    for spec in (refund_spec, collusion_spec):
        for cond in spec["conditions"]:
            assert "use" not in cond, f"unresolved import: {cond}"
            assert cond.get("id"), f"condition with no id: {cond}"
            assert cond.get("kind"), f"condition {cond['id']} has no kind"


def test_the_same_check_carries_identical_mechanics_and_different_weighting(
    refund_spec, collusion_spec
):
    """THE POINT OF THE SHARED LAYER, asserted on a real pack rather than a fixture.

    `manually_reviewed` is a DECISIVE CATEGORICAL EXCLUSION for refund fraud (a named person
    adjudicated the claim, so the behavioural indicators lose their meaning) and a NON-DECISIVE
    FRAUD INDICATOR for courier collusion (a colluding pair can route a claim past a complicit
    reviewer). Two procedures legitimately disagreeing about what one recorded fact MEANS is
    exactly why the weighting stays in the ruleset — and neither restates the field path, so
    they cannot drift apart about where the fact lives.
    """
    a = next(c for c in refund_spec["conditions"] if c["id"] == "manually_reviewed")
    b = next(c for c in collusion_spec["conditions"] if c["id"] == "manually_reviewed")

    # Mechanics — declared ONCE in shared/checks/, identical in both.
    for key in ("kind", "source", "flag_fields", "expected"):
        assert a[key] == b[key], f"mechanics diverged on {key}"

    # Weighting — each procedure's own.
    assert a["decisive"] is True and a["exclusion_kind"] == "categorical"
    assert b["decisive"] is False and b["polarity"] == "fraud_indicator"
    # The label follows the polarity: an exclusion's label states the REQUIREMENT, an
    # indicator's states its own FINDING, so the importing ruleset overrides it.
    assert a["label"] != b["label"]


def test_an_indicator_import_is_not_silently_an_exclusion(collusion_spec):
    """`decisive` and `polarity` are independent, and confusing them inverts a verdict.

    `pod_captured_by_handler` was first written into this pack as a decisive CATEGORICAL
    EXCLUSION, which would have reported the handler/POD divergence it exists to catch and
    then concluded NO COLLUSION from it — a decisive exclusion FAIL rolls up to
    `false_positive`, a decisive indicator FAIL to `fraud`. `exclusion_kind` is absent on an
    indicator by design: it grades an exclusion's evidence and means nothing here.
    """
    cond = next(
        c for c in collusion_spec["conditions"] if c["id"] == "pod_captured_by_handler"
    )
    assert cond["polarity"] == "fraud_indicator"
    assert cond["decisive"] is True and cond["decisive_on"] == ["fail"]
    assert "exclusion_kind" not in cond


# --- every verdict label is reachable ----------------------------------------


def test_fraud_on_the_textbook_fingerprint(refund_spec):
    v = verdict_for(refund_spec, [ledger_row()])
    assert label(v) == "REFUND FRAUD"


def test_a_categorical_exclusion_reaches_false_positive(refund_spec):
    v = verdict_for(refund_spec, [ledger_row(**REVIEWED)])
    assert label(v) == "NOT FRAUD"
    assert check(v, "manually_reviewed").result == "fail"


def test_the_scope_gate_exits_before_the_procedure_applies(refund_spec):
    """A gate FAIL is the engine declining to adjudicate, not an adjudicated clearance.

    Reported as its own label because "this ruleset has no opinion on this shipment" and
    "this shipment was examined and cleared" are different statements and only one is true.
    Without the distinction an out-of-scope case fails an ordinary exclusion and prints as
    NOT FRAUD.
    """
    v = verdict_for(refund_spec, [ledger_row(**{"route.destination": "ZZZ99"})])
    assert label(v) == "OUT OF SCOPE — PROCEDURE DOES NOT APPLY"
    assert check(v, "served_route").result == "fail"


def test_a_five_character_scope_point_is_recognised(refund_spec):
    """`point_pattern` is why the gate works here at all.

    The engine's default point shape is a 3-letter airport code. A depot code is five
    characters, so before the pattern was declarable this check returned `unknown` — on the
    condition that is answered FIRST, which suppresses the jurisdiction question for the
    entire procedure. `unknown` on a gate is the expensive direction of wrong.
    """
    v = verdict_for(refund_spec, [ledger_row()])
    served = check(v, "served_route")
    assert served.result == "pass"
    assert "LDS04" in served.observed or "LDS04" in served.detail


def test_insufficient_data_when_a_decisive_check_cannot_be_evaluated(refund_spec):
    """A row that resolves nothing must read as UNDECIDED, never as clean.

    `decisive_on: [fail]` is what makes the other checks' `unknown`s harmless, so this fires
    on the one decisive-symmetric path. The failure this guards is the whole reason the codebase
    distinguishes 0-rows from no-lookup: an `unknown` silently scored as a PASS produces a
    confident verdict over data that never arrived.
    """
    spec = dict(refund_spec)
    spec["conditions"] = [
        dict(c, decisive=True, decisive_on=["fail", "unknown"])
        if c["id"] == "no_servicing_elements"
        else c
        for c in refund_spec["conditions"]
    ]
    # `scans` omitted entirely (not empty) and no counters -> the absence check cannot claim
    # a PASS for paths it never read.
    v = verdict_for(
        spec,
        [
            {
                "tracking_code": "RT48192043",
                "depot_code": "LDS04",
                "route.origin": "LDS04",
                "route.destination": "BHM11",
            }
        ],
    )
    assert check(v, "no_servicing_elements").result == "unknown"
    assert label(v) == "INSUFFICIENT DATA"


def test_indicator_corroboration_outweighs_a_heuristic_exclusion(refund_spec):
    """>= `indicator_threshold` non-decisive indicator FAILs reach fraud on their own.

    Without this path an engine can only ever clear a case or match the textbook fingerprint,
    so a NON-textbook fraud (elements present, ordinary refund reason) is unreachable. Here
    the decisive exclusion FAILs — but it is HEURISTIC, an inference about the shipment's
    shape, and two positive indicators outrank an inference.
    """
    rows = [
        ledger_row(**SERVICED, **{"handler.badge": "0192C", "refund.reason_code": "MISC"}),
        ledger_row(**SERVICED, **{"handler.badge": "0288D", "refund.reason_code": "MISC"}),
    ]
    v = verdict_for(refund_spec, rows)
    assert check(v, "no_servicing_elements").result == "fail"  # decisive, heuristic
    fails = [
        c.id
        for c in v.subjects[0].checks
        if c.polarity == "fraud_indicator" and c.result == "fail"
    ]
    assert len(fails) >= refund_spec["indicator_threshold"]
    assert label(v) == "REFUND FRAUD"


def test_a_categorical_exclusion_outranks_the_same_indicator_vote(refund_spec):
    """Identity evidence is prior to behavioural evidence, and the ordering is the engine's.

    Same two rows as above, plus a named approver. A heuristic exclusion can be outvoted by
    indicators; a CATEGORICAL one cannot — once an attributed fact (who acted, and when) is on
    the record the indicators lose their MEANING rather than their weight.
    """
    rows = [
        ledger_row(
            **SERVICED, **REVIEWED, **{"handler.badge": "0192C", "refund.reason_code": "MISC"}
        ),
        ledger_row(
            **SERVICED, **REVIEWED, **{"handler.badge": "0288D", "refund.reason_code": "MISC"}
        ),
    ]
    v = verdict_for(refund_spec, rows)
    assert check(v, "manually_reviewed").result == "fail"
    assert len([c for c in v.subjects[0].checks
                if c.polarity == "fraud_indicator" and c.result == "fail"]) >= 2
    assert label(v) == "NOT FRAUD"


def test_a_decisive_indicator_reaches_fraud_in_the_second_ruleset(collusion_spec):
    """The second procedure over the SAME data, reaching its own label.

    `courier_collusion` declares no field path of its own — every check it uses was written
    for refund fraud. Its decisive check is an INDICATOR, so a FAIL argues FOR its label.
    """
    v = verdict_for(collusion_spec, [ledger_row(**{"pod.captured_by": "0999Z"})])
    assert label(v) == "COLLUSION CONFIRMED"
    assert check(v, "pod_captured_by_handler").result == "fail"


def test_the_decisive_comparison_is_asked_within_one_shipment(collusion_spec):
    """`pair_by: record`, on the shape that needs it.

    Two legs of one shipment, each handled by one courier and signed for by the OTHER. Every
    record diverges, so the pooled reading of a two-sided comparison gathers each side over both
    rows and asks only whether the sets intersect. They do, on a pair sitting on no leg: 0192C is
    a handler on one row and a POD capturer on the other, so without the declaration the decisive
    indicator PASSES and the ruleset clears the case it was written to catch. Asserted in both
    directions on the same rows.

    THE ASSERTION IS ON THE CHECK AND NOT ON THE LABEL: two legs handled by two couriers is what
    a cross pair needs, so `single_handler` FAILs here too and the ruleset reaches its fraud label
    either way. The label is blind to this defect, which is why a verdict-level test cannot see it.
    """
    rows = [
        ledger_row(**{"handler.badge": "0192C", "pod.captured_by": "0288D"}),
        ledger_row(**{"handler.badge": "0288D", "pod.captured_by": "0192C"}),
    ]
    v = verdict_for(collusion_spec, rows)
    c = check(v, "pod_captured_by_handler")
    assert c.result == "fail", (c.result, c.observed, c.detail)
    assert c.decisive is True, c
    assert label(v) == "COLLUSION CONFIRMED"
    # The count is part of the finding: a comparison made within each leg and one made across
    # the shipment are different questions, and only the note says which was asked.
    assert "compared WITHIN each of the 2 record(s)" in c.detail, c.detail

    # The same rows without the declaration — the fabricated PASS this pack must not report.
    pooled = dict(collusion_spec)
    pooled["conditions"] = [
        {k: val for k, val in cond.items() if k != "pair_by"}
        for cond in collusion_spec["conditions"]
    ]
    pc = check(verdict_for(pooled, rows), "pod_captured_by_handler")
    assert pc.result == "pass", (pc.result, pc.observed, pc.detail)
    assert "WITHIN each" not in (pc.detail or ""), pc.detail


def test_an_automated_terminal_session_explains_the_divergence_away(collusion_spec):
    """A boolean forbidden vocabulary must be able to FIRE.

    `record_absence` compares upper-cased strings and `_collect` drops bools, so before
    `_collect_any` this predicate could not match: the check PASSED on every row and reported
    "no forbidden value" over a source where `session.automated` was `true` on all of them.
    Here the divergence has an innocent explanation — an unattended depot terminal captured the
    POD, no second person involved — and the categorical exclusion must beat the decisive
    indicator.
    """
    v = verdict_for(
        collusion_spec,
        [ledger_row(**{"pod.captured_by": "0999Z"})],
        sessions=[session_row()],
    )
    assert check(v, "automated_terminal_session").result == "fail"
    assert check(v, "pod_captured_by_handler").result == "fail"
    assert label(v) == "NO COLLUSION"


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param({"session.automated": True}, id="flat-dotted"),
        pytest.param({"session": {"automated": True}}, id="nested-dict"),
        pytest.param({"session_automated": True}, id="underscore-alias"),
        pytest.param({"session": '{"automated": true}'}, id="json-string-struct"),
    ],
)
def test_a_boolean_forbidden_vocabulary_matches_in_every_row_shape(collusion_spec, shape):
    """The predicate's own resolver, swept over the same four shapes as the flag check.

    `record_absence` and `field_flag` are separate evaluators with separate resolvers, and
    fixing one says nothing about the other — verified by mutation: disabling the NESTED-bool
    branch of `_collect_any` left every other test in this file green, because the session
    fixture happened to use the flat-dotted shape. Which shape a source returns is a property
    of the backend, so a check that works on one of them works by luck; and here the cost is
    the whole exclusion, since an unmatchable forbidden value PASSES on every row.
    """
    row = session_row()
    row.pop("session.automated")
    row.update(shape)
    v = verdict_for(
        collusion_spec, [ledger_row(**{"pod.captured_by": "0999Z"})], sessions=[row]
    )
    assert check(v, "automated_terminal_session").result == "fail"
    assert label(v) == "NO COLLUSION"


def test_a_human_courier_session_does_not_degrade_the_case(collusion_spec):
    """NOT finding an automated session is what every human courier looks like.

    The asymmetry `decisive_on: [fail]` exists for: a PASS here is the ordinary case and must
    never force INSUFFICIENT DATA, while the FAIL above is conclusive.
    """
    v = verdict_for(
        collusion_spec,
        [ledger_row(**{"pod.captured_by": "0999Z"})],
        sessions=[session_row(**{"session.automated": False})],
    )
    assert check(v, "automated_terminal_session").result == "pass"
    assert label(v) == "COLLUSION CONFIRMED"


# --- boolean-shaped data ----------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param({"refund.manually_reviewed": True}, id="flat-dotted"),
        pytest.param({"refund": {"manually_reviewed": True}}, id="nested-dict"),
        pytest.param({"refund_manually_reviewed": True}, id="underscore-alias"),
        pytest.param({"refund": '{"manually_reviewed": true}'}, id="json-string-struct"),
    ],
)
def test_a_boolean_flag_is_read_in_every_row_shape(refund_spec, shape):
    """One flag, four row shapes, one answer.

    `resolve_path` drops bools by design — `_scalar` excludes them so a boolean is never
    mistaken for a join key, which is right for key discovery and wrong for reading a flag.
    Before `_flag_leaves`, only the FLAT-DOTTED shape was seen: a Kibana `_source`, a
    Snowflake VARIANT, a JSON-string struct or a Databricks underscore-aliased sub-struct all
    reported "flag absent" over a flag that was plainly set. Which shape a source returns is
    a property of the backend, so a check that only works on one of them works by luck.
    """
    row = ledger_row()
    row.pop("refund.manually_reviewed")
    row.update(shape)
    v = verdict_for(refund_spec, [row])
    assert check(v, "manually_reviewed").result == "fail"
    assert label(v) == "NOT FRAUD"


def test_an_absent_flag_is_unknown_not_a_pass(refund_spec):
    """The distinction the bool fix must NOT blur: absent is not false.

    "No review is recorded" and "we did not retrieve the review column" are different facts,
    and only the first is evidence. `unknown` on a decisive-on-fail exclusion is harmless by
    design; silently reading it as a PASS would be a verdict resting on a projection gap.
    """
    row = ledger_row()
    row.pop("refund.manually_reviewed")
    v = verdict_for(refund_spec, [row])
    assert check(v, "manually_reviewed").result == "unknown"


def test_an_empty_array_is_a_real_zero_and_a_missing_one_is_not(refund_spec):
    """`element_absence` may only claim absence for a path it actually READ.

    `scans: []` resolves — the container arrived and holds nothing, so "no scan chain" is the
    finding. Omit the key and nothing was read, so the honest answer is `unknown`. Counting
    leaves alone cannot tell these apart (an empty list contributes none either way), which is
    why the container is probed explicitly. Conflating them lets a retrieval gap read as
    evidence — here, as the fabricated-movement fingerprint.
    """
    empty = verdict_for(refund_spec, [ledger_row()])
    assert check(empty, "no_servicing_elements").result == "pass"

    row = ledger_row()
    row.pop("scans")
    row.pop("element_counters.INS")
    row.pop("element_counters.SIG")
    missing = verdict_for(refund_spec, [row])
    assert check(missing, "no_servicing_elements").result == "unknown"


# --- the case builder -------------------------------------------------------


def build_brief(pack, spec, ledger=None, sessions=None, alerts=None):
    return UseCaseAnalyzer(use_case="refund_fraud").analyze(
        spec,
        {
            "shipment_ledger": ledger if ledger is not None else [ledger_row()],
            "device_sessions": sessions or [],
            "refund_alerts": alerts if alerts is not None else [alert_row()],
        },
        analysis(),
        knowledge_pack=pack,
        playbook_id="PB-REFUND-001",
    )


def test_the_case_builder_attaches_the_declared_shared_concepts(pack, refund_spec):
    """A shared id is nameable in `case_builder.concepts` exactly like a local one.

    The union is specific-first, so an author never has to know which folder a concept lives
    in — and a use case that re-states a shared id gets ITS reading. RAG retrieval is separate
    and wider: it can surface any shared concept by similarity without it being listed here.
    """
    brief = build_brief(pack, refund_spec).brief
    ids = [c.concept_id for c in brief.concept_refs]
    assert ids == ["shipment_elements", "actor_identity_forms", "bare_shipment_is_suspicious"]
    # Two shared, one the use case's own — and every one carries real text for the prompt.
    assert all(c.snippet.strip() for c in brief.concept_refs)


def test_the_alert_record_is_located_and_reconciled(pack, refund_spec):
    """The one input the pipeline did not derive is treated as ground truth to REPRODUCE.

    Every fact the detector stated must be confirmed against the logs, and a divergence is a
    finding (wrong record selected, stale index) rather than a reason to prefer the logs
    quietly.
    """
    facts = build_brief(pack, refund_spec).brief.alert_facts
    assert facts is not None and facts.located is True
    assert facts.record_id == "IR-REF-0001"
    assert {f.field for f in facts.declared_facts} == {"depot", "courier", "shipment"}
    assert all(f.status == "confirmed" for f in facts.declared_facts)


def test_the_declared_trigger_reaches_the_brief_and_carries_its_negative(pack, refund_spec):
    """What the detector fires on is unreconstructible from the retrieved rows.

    Nothing in the data says which condition raised the alert, so a narration asked to explain
    the case reaches for whichever retrieved fact looks most suspicious and states THAT as the
    trigger. Measured on a live record-misuse run: the report said the alert fired on the ABSENCE of a
    payment element, for a detector that never looks at payment, on a record that did carry
    one — a premise both unalleged and false.

    The `description` is emitted VERBATIM because it is the only place that can say what the
    detector does NOT look at, and that negative is the half a reader most needs: without it,
    a large refund reads as the allegation.
    """
    trigger = build_brief(pack, refund_spec).brief.alert_facts.trigger
    assert "not amount-based" in trigger.lower(), trigger
    # The quantitative keys ride along as a parenthetical, so the burst threshold is checkable
    # without opening the pack.
    assert "3" in trigger and "24h" in trigger and "depot" in trigger, trigger


def test_a_ruleset_that_declares_no_trigger_says_nothing_about_one(pack, refund_spec):
    """Silence is correct; an invented trigger is not.

    The engine has no generic sentence to fall back on here, deliberately: a sentence about
    triggers in general would read as the trigger having been ESTABLISHED, which is the exact
    confusion this whole mechanism exists to prevent. So an undeclared trigger yields an empty
    string, the report section omits the line, and the narration prompt says nothing.
    """
    spec = dict(refund_spec)
    spec.pop("trigger", None)
    assert build_brief(pack, spec).brief.alert_facts.trigger == ""


def test_a_projection_guard_probing_a_boolean_does_not_flag_degraded(pack, refund_spec):
    """The guard exists to stop a data gap reading as a finding; it must not invent one.

    `_collect` drops bools, so a probe path naming a FLAG could never be satisfied — the leaf
    arrives, the guard reports "missing from the returned rows", and a fully-evidenced verdict
    is marked `degraded`. Worst on exactly the leaves most worth guarding: a decisive
    exculpatory exclusion is usually a boolean.
    """
    brief = build_brief(pack, refund_spec).brief
    assert not [n for n in brief.notes if "manually_reviewed" in n]
    assert brief.degraded is False


def test_a_genuinely_missing_leaf_still_flags_degraded(pack, refund_spec):
    """The other half: with the counters gone the guard MUST fire.

    A bare-shipment check reading a leaf the projection dropped returns `unknown`, which is
    indistinguishable from "the elements really are absent" — so the pack declares which leaves
    must have ARRIVED and the note says the check is unknown for a DATA reason.
    """
    row = ledger_row()
    row.pop("element_counters.INS")
    brief = build_brief(pack, refund_spec, ledger=[row]).brief
    assert any("element_counters.INS" in n for n in brief.notes)
    assert brief.degraded is True


def test_an_empty_system_of_record_is_reported_as_such(pack, refund_spec):
    """0 rows from the primary source must never read as 0 findings."""
    brief = build_brief(pack, refund_spec, ledger=[]).brief
    assert any("shipment_ledger" in n for n in brief.notes)
    assert brief.degraded is True


def test_the_next_steps_backbone_invents_no_action_verb(pack, refund_spec):
    """The engine owns the BRANCH, the pack owns the PROSE.

    This ruleset declares no `action_templates`, and the correct output is to say so: badge
    suspension and device-login revocation are not interchangeable, so a defaulted verb is an
    operational error rather than a cosmetic one.
    """
    steps = " ".join(build_brief(pack, refund_spec).brief.action_backbone).lower()
    assert steps
    assert "no action verb could be resolved" in steps


def test_the_close_step_states_the_finding_and_not_the_negated_requirement(pack, refund_spec):
    """The label-vs-finding defect, in the ONE line an operator acts on.

    `manually_reviewed` is labelled as a requirement — *Refund was not manually reviewed* — and
    a FAIL negates it: the review DID happen, which is what closes the case. The backbone used
    to print the label, so the checklist line read `CLOSE … (decisive exclusion: Refund was not
    manually reviewed)` — the opposite of its grounds, directly beneath a verdict note stating
    the review happened. The pack already carries the right words in `fail_detail`; the engine's
    job is to prefer them, which is what `_finding` does for the verdict note and what this
    branch now does too.

    Nothing here is domain-specific: any exclusion-phrased label has this polarity, which is
    why the regression lives on the fixture pack and not on the pack that surfaced it.
    """
    brief = build_brief(pack, refund_spec, ledger=[ledger_row(**REVIEWED)]).brief
    close = [s for s in brief.action_backbone if "decisive exclusion:" in s]
    assert len(close) == 1, brief.action_backbone

    assert "the refund was manually reviewed and approved by a named person" in close[0], close[0]
    assert "Refund was not manually reviewed" not in close[0], (
        "the requirement wording asserts the opposite of the finding that closed the case"
    )
    assert "(refund.manually_reviewed=True)" in close[0], (
        "and the observed value must survive the sentence cut — it is the one part of the "
        "clause no prose can be wrong about"
    )


def test_a_decisive_exclusion_with_no_fail_detail_is_never_the_negated_label(pack, refund_spec):
    """A pack that declares no `fail_detail` still may not ship an unmarked false statement.

    Six of the 22 decisive exclusions across the installed packs declare none, so this is the
    normal path and not a corner. It does not fall through to the label: each evaluator writes
    its own finding-phrased note onto the check ("the flag is True, not the expected False"),
    which is engine prose but still states what was FOUND. The label is reached only when
    neither exists, and then it is fenced — see the unit assertion at the bottom.
    """
    import copy as _copy

    from src.correlation import _finding_headline

    spec = _copy.deepcopy(refund_spec)
    for cond in spec["conditions"]:
        if cond["id"] == "manually_reviewed":
            cond.pop("fail_detail", None)

    brief = build_brief(pack, spec, ledger=[ledger_row(**REVIEWED)]).brief
    close = next(s for s in brief.action_backbone if "decisive exclusion:" in s)
    assert "the flag is True, not the expected False" in close, close
    assert "Refund was not manually reviewed" not in close, close

    bare = SimpleNamespace(
        id="manually_reviewed",
        label="Refund was not manually reviewed",
        detail="",
        observed="refund.manually_reviewed=True",
        result="fail",
        decisive=True,
    )
    assert _finding_headline(bare) == (
        "FAILED: Refund was not manually reviewed (refund.manually_reviewed=True)"
    ), _finding_headline(bare)


# --- the pack's own declarations are wired, not decorative -------------------


def test_the_query_guarantees_reach_a_retriever_config(pack):
    """All five guarantees ride ONE seam, so a new backend cannot no-op a declaration.

    They were databricks-only once, which meant `never_filter` on an Elasticsearch source did
    nothing at all — a pack field that silently no-ops is the bug it was written to prevent.
    `_attach_query_guards` is that seam; this asserts this pack's declarations arrive
    through it, over an ES source and a Databricks source alike.
    """
    from log_retrieval import LogRetrievalEngine

    for name, expected in (
        ("shipment_ledger", {"never_filter", "partition_columns", "epoch_time_columns"}),
        ("device_sessions", {"require_all_entities", "identity_keys", "never_filter"}),
        ("refund_alerts", set()),
    ):
        src = pack.source(name)
        merged = {}
        LogRetrievalEngine._attach_query_guards(src, merged)
        for key in expected:
            assert merged.get(key), f"{name}: {key} did not ride the seam"

    ledger = {}
    LogRetrievalEngine._attach_query_guards(pack.source("shipment_ledger"), ledger)
    assert ledger["retrieval_class"] == "primary"
    assert ledger["never_filter"] == ["refund.manually_reviewed", "pod.signature_captured"]
    # An epoch column declares its UNIT only — the window is COMPUTED, never written as a
    # literal into query_hints (one such literal went stale by exactly a year and cost a live
    # source all of its rows).
    assert ledger["epoch_time_columns"] == [
        {"name": "pod_captured_ms", "unit": "milliseconds"}
    ]
    for spec in ledger["partition_columns"]:
        assert "pad_days" in spec, "declare only what backend metadata cannot say"


def test_zero_rows_from_a_lookup_source_is_declared_as_an_answer(pack):
    """An empty result is sometimes the finding, and the pack is what says which.

    `device_sessions` answers an exclusion by being empty: no session row means no automated
    terminal acted, i.e. a person did. The health scorer's pro-rata empty-source penalty would
    otherwise invert that evidence. The weight is the arithmetic; which sources qualify is a
    pack judgement.
    """
    assert pack.source("device_sessions").zero_rows["health_weight"] == 0.0
    assert pack.source("device_sessions").zero_rows["meaning"].strip()
    # And the primary source is NOT discounted — an empty system of record is a real gap.
    assert not pack.source("shipment_ledger").zero_rows


def test_the_planner_is_told_what_a_source_cannot_answer(pack):
    """`description` answers neither question the planner actually has.

    `not_answered_by` states what a source cannot answer AND who can; `selection_guidance`
    states when it is worth asking at all. Note the two ways to get a source retrieved are not
    interchangeable — a ruleset's `sources:` map is a hard dependency, right when a condition
    READS the source; `selection_guidance` leaves the call with the planner, right when
    relevance is incident-dependent.
    """
    ledger = pack.source("shipment_ledger")
    assert ledger.not_answered_by
    assert all(
        entry.get("question") and entry.get("ask_instead")
        for entry in ledger.not_answered_by
    )
    sessions = pack.source("device_sessions")
    assert sessions.selection_guidance.get("choose_when")
    assert sessions.selection_guidance.get("skip_when")
    # ...and it is the one source NOT in either ruleset's hard `sources:` dependency map by
    # accident: both declare it, because both have a condition that reads it.
    assert "sessions" in pack.ruleset_spec("refund_fraud")["sources"]


def test_a_courier_is_bound_per_value_form(pack):
    """One entity type, two non-interchangeable surface forms.

    A badge and a device login name the same person and live in different columns. Bind both
    to one column and the query returns zero rows — indistinguishable from "this courier did
    nothing". A value whose form a source binds nothing for is DROPPED, never unioned onto the
    sibling form's column, and no prompt can restore a distinction the data model discarded.
    """
    assert pack.classify_value_form("courier", "0192C") == "badge"
    assert pack.classify_value_form("courier", "jdunne") == "device_login"
    assert pack.classify_value_form("courier", "??") is None

    forms = pack.form_bindings_for("courier", "device_sessions")
    assert forms == {"badge": ["badge"], "device_login": ["login"]}
    assert pack.field_priors_for("courier", "device_sessions", "badge")[0] == "badge"
    assert pack.field_priors_for("courier", "device_sessions", "device_login")[0] == "login"


def test_the_prompts_the_pack_builds_are_non_empty_and_domain_specific(pack):
    """The pack must actually reach the LLM, not merely load.

    `glossary_prompt` seeds entity extraction and closes the abbreviation list (an LLM asked to
    summarise expands an unfamiliar acronym, and a plausible wrong expansion reads exactly like
    a right one). `catalog_prompt` is what source selection chooses from — including the
    negative guidance, which is the half a `description` cannot carry.
    """
    glossary = pack.glossary_prompt()
    assert "Proof Of Delivery" in glossary  # the closed abbreviation list
    assert "shipment" in glossary and "^[A-Z]{2}[0-9]{8}$" in glossary

    catalog = pack.catalog_prompt()
    assert all(
        name in catalog
        for name in ("refund_alerts", "shipment_ledger", "device_sessions")
    )
    assert "DOES NOT ANSWER" in catalog
    assert "CHOOSE WHEN" in catalog
    # Schema docs are RAG-only: they are far too large for a selection prompt.
    assert "populated_pct" not in catalog


def test_report_vocabulary_is_scoped_to_the_use_case_that_owns_it(pack):
    """A clause number belongs to ONE procedure, so its phrases are scoped per use case.

    Hoisted to the domain root, one procedure's wording mis-cites every other procedure's
    report — and a clause citation is the worst kind of leak, because it reads as authoritative
    and cannot be checked from the engine. An unscoped lookup falls back to the engine's own
    procedure-free sentence.
    """
    assert pack.report_phrase("containment_target", "", use_case="refund_fraud")
    # The ENGINE'S default is what a use case declaring nothing gets — not the other use
    # case's clause wording, and not an empty string either.
    assert (
        pack.report_phrase(
            "containment_target", "the acting identity", use_case="courier_collusion"
        )
        == "the acting identity"
    )
    labels = [r["label"] for r in pack.phase_rules("refund_fraud")]
    assert "Refund claim activity" in labels
    assert pack.phase_rules("courier_collusion") == []


# --- subjects discovered from rows -------------------------------------------
# The subject list comes from the entities the understanding stage extracted, and that stage runs
# BEFORE retrieval. An alert naming its subjects only INSIDE the document yields none, so the
# fallback adjudicates one degenerate subject over every retrieved row and the report says
# INSUFFICIENT DATA after a run in which every stage reported success. `subject_discovery:` is the
# pack-driven primitive for that case; every path, source and entity type below is read out of
# `knowledge/mock_domain/`.


def test_subjects_are_discovered_from_rows_when_the_incident_named_none(refund_spec):
    """No extracted shipment -> one subject per declared array element, not one degenerate."""
    no_subject = SimpleNamespace(
        incident_summary="Refund claim CLM-9001 raised in one depot",
        initial_hypotheses=[],
        key_investigation_areas=[],
        correlation_keys=[],
        # The depot and the claim, which is what an alert handle carries. No shipment: that is
        # the whole case this primitive exists for.
        extracted_entities=[
            ExtractedEntity(type="depot", value="LDS04"),
            ExtractedEntity(type="refund_claim", value="CLM-9001"),
        ],
        event_time=None,
    )
    ledger = [
        ledger_row(),
        ledger_row(tracking_code="QQ00771265", **{"handler.badge": "7741B"}),
    ]
    v = evaluate_verdict(
        refund_spec,
        {"shipment_ledger": ledger, "device_sessions": [], "refund_alerts": [alert_row()]},
        no_subject,
    )
    assert v is not None and v.subjects
    values = [s.subject_value for s in v.subjects]
    assert values == ["RT48192043", "QQ00771265"], (
        "the two claim lines must adjudicate as two subjects; one degenerate subject over "
        f"every row is the defect this replaces (got {values})"
    )
    # Not merely present: each subject must have been ADJUDICATED, i.e. its checks resolved
    # against its own rows rather than the whole retrieved set.
    for s in v.subjects:
        assert s.checks, f"subject {s.subject_value} was listed but no check was evaluated for it"


def test_a_discovered_subject_is_given_only_its_own_elements_values(refund_spec):
    """Per-ELEMENT grouping, asserted on the rendered note — the union is the real bug.

    `_resolve_nodes` returns the node AT the declared path, so a repeated node arrives as ONE
    list and every relative path read from it resolves across all elements at once. Without an
    explicit one-level expansion each subject is handed every sibling's co-located values —
    and a reference lookup belonging to one identity then answers another's question, through
    the most decisive exit the engine has.
    """
    no_subject = SimpleNamespace(
        incident_summary="Refund claim CLM-9001 raised in one depot",
        initial_hypotheses=[],
        key_investigation_areas=[],
        correlation_keys=[],
        extracted_entities=[ExtractedEntity(type="depot", value="LDS04")],
        event_time=None,
    )
    ledger = [
        ledger_row(),
        ledger_row(tracking_code="QQ00771265", **{"handler.badge": "7741B"}),
    ]
    v = evaluate_verdict(
        refund_spec,
        {"shipment_ledger": ledger, "device_sessions": [], "refund_alerts": [alert_row()]},
        no_subject,
    )
    notes = {s.subject_value: " ".join(s.notes or []) for s in v.subjects}
    assert "courier=0192C" in notes["RT48192043"], notes["RT48192043"]
    assert "courier=7741B" in notes["QQ00771265"], notes["QQ00771265"]
    # THE REGRESSION: the sibling's badge must NOT be attributed to this subject.
    assert "7741B" not in notes["RT48192043"], notes["RT48192043"]
    assert "0192C" not in notes["QQ00771265"], notes["QQ00771265"]


def test_an_extracted_subject_still_wins_over_the_rows(refund_spec):
    """The declaration is a FALLBACK. Where the incident named its subject, nothing changes.

    This is what keeps `subject_discovery` inert on every incident that carries its own
    identity — and it is asserted rather than assumed, because a fallback that fires when it
    should not silently re-scopes a verdict the operator already scoped.
    """
    v = verdict_for(refund_spec, [ledger_row()], alerts=[alert_row()])
    assert [s.subject_value for s in v.subjects] == ["RT48192043"]
    assert not any("subject_discovered" in n for s in v.subjects for n in (s.notes or []))


def test_a_pack_declaring_no_subject_discovery_is_unaffected(collusion_spec):
    """The second ruleset declares none, so the degenerate fallback still applies to it."""
    assert "subject_discovery" not in collusion_spec
    no_subject = SimpleNamespace(
        incident_summary="Two couriers repeatedly signing for each other",
        initial_hypotheses=[],
        key_investigation_areas=[],
        correlation_keys=[],
        extracted_entities=[ExtractedEntity(type="depot", value="LDS04")],
        event_time=None,
    )
    v = evaluate_verdict(
        collusion_spec,
        {"shipment_ledger": [ledger_row()], "device_sessions": [session_row()]},
        no_subject,
    )
    # The degenerate single subject, which is the pre-primitive behaviour verbatim: one
    # pseudo-subject standing for the whole retrieved set. That it is still reached here is
    # what makes `subject_discovery` opt-in rather than a change of default.
    assert [s.subject_value for s in v.subjects] == ["(all retrieved records)"]
    assert not any(
        "subject_discovered" in n for s in v.subjects for n in (s.notes or [])
    )


# --- the columns the procedure's own conditions read -------------------------
# `trim_columns` had three routes to "relevant" — `entity_bindings`, a resolved join key, a decoded
# table's columns — and a condition's OWN field was in none of them, so the field a PASS/FAIL rests
# on was dropped from the evidence the report narrates from and published under `dropped_columns`
# beside a finding that says it was read. The constant-value rule makes it systematic: a vocabulary
# column is constant across a reference source's rows BECAUSE every row says the same thing, which
# is what makes reading it decisive.


def _ledger_cohort(n=60):
    """`n` ledger rows that vary in two columns and are CONSTANT in the rest.

    That is the shape a reference or per-record lookup really returns, and the shape the
    trimmer's `_CONSTANT_COL_RATIO` rule strips down to the two varying columns.
    """
    return [
        ledger_row(tracking_code=f"RT{i:08d}", **{"refund.amount": float(i)})
        for i in range(n)
    ]


def test_a_condition_read_column_survives_the_constant_value_trimmer(refund_spec):
    """Measured on the mock pack: the trimmer keeps 2 of 13 columns, and 9 are condition fields."""
    from src.evidence import _condition_field_relevance, trim_columns

    rows = _ledger_cohort()
    kept_before, dropped_before, _ = trim_columns(rows, set())
    assert "refund.reason_code" in dropped_before, dropped_before
    assert set(kept_before) == {"tracking_code", "refund.amount"}, kept_before

    relevance = _condition_field_relevance(refund_spec, {"shipment_ledger": rows})
    kept_after, dropped_after, _ = trim_columns(rows, relevance["shipment_ledger"])

    # Every column this ruleset's conditions read is now shown, and the constant-value rule is
    # what would have taken all but one of them.
    assert relevance["shipment_ledger"] <= set(kept_after), sorted(
        relevance["shipment_ledger"] - set(kept_after)
    )
    assert "refund.reason_code" in kept_after, kept_after

    # SCOPED, not a blanket disable of the rule: a column no condition reads is still dropped,
    # which is the whole reason the trimmer exists.
    assert "depot_code" in dropped_after, dropped_after
    assert set(dropped_after) < set(dropped_before), (sorted(dropped_after), sorted(dropped_before))


def test_a_pinned_column_survives_the_column_cap(refund_spec):
    """"Always kept" has to mean always: the cap used to evict pinned columns after `max_cols`."""
    from src.evidence import _condition_field_relevance, trim_columns

    rows = _ledger_cohort()
    pinned = _condition_field_relevance(refund_spec, {"shipment_ledger": rows})[
        "shipment_ledger"
    ]
    assert len(pinned) > 4, sorted(pinned)
    kept, _dropped, _ = trim_columns(rows, pinned, max_cols=4)
    assert pinned <= set(kept), sorted(pinned - set(kept))


def test_an_underscore_flattened_column_is_recognised_by_its_dotted_path(refund_spec):
    """A struct leaf arrives flattened by the BACKEND, so a pack path and its column differ.

    A nested leaf comes back from a SQL warehouse as one column named with underscores, and
    `resolve_path` reads the dotted pack path off it. The trimmer keys on the column as the rows
    spell it, so comparing a pack path to a column name directly matches nothing and the pin is
    silently inert — which looks identical to no pin at all.
    """
    from src.evidence import _condition_field_relevance, trim_columns

    rows = [
        {
            (k.replace(".", "_") if k.startswith("handler.") else k): v
            for k, v in row.items()
        }
        for row in _ledger_cohort()
    ]
    assert "handler_badge" in rows[0] and "handler.badge" not in rows[0]

    relevance = _condition_field_relevance(refund_spec, {"shipment_ledger": rows})[
        "shipment_ledger"
    ]
    assert "handler_badge" in relevance, sorted(relevance)
    kept, _dropped, _ = trim_columns(rows, relevance)
    assert "handler_badge" in kept, kept


def test_a_run_with_no_resolved_ruleset_trims_exactly_as_before(refund_spec):
    """The pin is additive: no ruleset resolved (or none declaring conditions) changes nothing."""
    from src.evidence import _condition_field_relevance, trim_columns

    rows = _ledger_cohort()
    baseline = trim_columns(rows, set())
    for absent in (None, {}, {"conditions": "not a list"}, {"conditions": []}):
        assert _condition_field_relevance(absent, {"shipment_ledger": rows}) == {}, absent
    assert trim_columns(rows, set()) == baseline


def test_the_evidence_pack_publishes_the_condition_columns_it_kept(pack, refund_spec):
    """End of the seam: `build_evidence` wires the ruleset through, not just the helper.

    Asserted on the PUBLISHED `SourceEvidence`, because `kept_columns`/`dropped_columns` are what
    the LLM narrates from and what the report prints — a helper that computes the right set and is
    never consulted is the defect, not the fix.
    """
    from src.evidence import build_evidence

    rows = _ledger_cohort()
    logs = {"shipment_ledger": rows}

    def _evidence(spec):
        ev = build_evidence(
            logs,
            {},
            [],
            pack.entity_bindings() if hasattr(pack, "entity_bindings") else {},
            ["RT00000000"],
            ruleset_spec=spec,
        )
        return next(s for s in ev.sources if s.source == "shipment_ledger")

    without, with_spec = _evidence(None), _evidence(refund_spec)
    assert "refund.reason_code" in without.dropped_columns, without.dropped_columns
    assert "refund.reason_code" in with_spec.kept_columns, with_spec.kept_columns
    assert "refund.reason_code" not in with_spec.dropped_columns, with_spec.dropped_columns


# --- a declared path may name a CONTAINER, which is a leaf of nothing ---------
# The pin above matched a declared path against the real leaves exactly, which is right for a scalar
# and wrong for the mechanics whose path names a struct or an array (`arrays:`, `counters:`,
# `records:`, a `data_map:` field), because a container resolves to no leaf of its own. The match
# missed, nothing was pinned, and a report stated the value was unavailable next to a FAIL resting
# on it — a contradiction inside one report rather than a gap in it.
#
# Exact-first, then read the candidate as a container: a declared string that is a proper PREFIX of
# a real leaf path names a container in these rows, and a comparison value does not.


def _serviced_cohort(n=60):
    """`_ledger_cohort`, but each row carries a scan record — so `scans` holds leaves."""
    return [
        ledger_row(tracking_code=f"RT{i:08d}", **{**SERVICED, "refund.amount": float(i)})
        for i in range(n)
    ]


def test_a_container_path_pins_every_leaf_beneath_it(refund_spec):
    """`arrays: [scans]` is declared as the container it is, and pins the three leaves under it."""
    from src.evidence import _condition_field_relevance, _leaf_index, trim_columns

    rows = _serviced_cohort()
    logs = {"shipment_ledger": rows}
    under = {"scans.scan", "scans.scanned_at", "scans.scanned_by"}

    # The negative half, and the whole reason an exact match was not enough: the declared path is
    # not a column. Every leaf that exists is one level BELOW it.
    leaves = _leaf_index(logs)("shipment_ledger")
    assert "scans" not in leaves, sorted(leaves)
    assert under <= set(leaves.values()), sorted(under - set(leaves.values()))

    _kept_before, dropped_before, _ = trim_columns(rows, set())
    assert under <= set(dropped_before), sorted(under - set(dropped_before))

    pinned = _condition_field_relevance(refund_spec, logs)["shipment_ledger"]
    assert under <= pinned, sorted(under - pinned)
    kept, dropped, _ = trim_columns(rows, pinned)
    assert under <= set(kept), sorted(under - set(kept))

    # Still scoped: expanding a container is not a blanket keep. A column that no condition reads
    # and sits under no declared container is dropped as before.
    assert "pod.captured_by" in dropped, dropped


def test_a_container_expansion_does_not_widen_a_scalar_pin(refund_spec):
    """The prefix rule must not fire on a path that ALREADY resolved, nor on a sibling stem.

    `handler.badge` is a real leaf and `handler.device_login` is its sibling — if the expansion ran
    unconditionally, or matched on the parent of a resolved path, reading one would keep both. The
    ruleset reads only the badge, and the trimmer's job is to say so.
    """
    from src.evidence import _condition_field_relevance, trim_columns

    rows = _serviced_cohort()
    pinned = _condition_field_relevance(refund_spec, {"shipment_ledger": rows})[
        "shipment_ledger"
    ]
    assert "handler.badge" in pinned and "handler.device_login" not in pinned, sorted(pinned)
    _kept, dropped, _ = trim_columns(rows, pinned)
    assert "handler.device_login" in dropped, dropped


# --- the columns the ALERT's own facts are read from -------------------------
# `alert_record:` names a second class of column — the fields the alert is parsed for, and the fields
# of the corroborating sources its values are confirmed against. Neither is a condition path, so
# neither was pinned, and the reconciliation reported a MISMATCH on a column published as dropped.
# `confirm_fields` are redirected to the entry's `confirm_in:` sources, the same redirect
# `pack_validate` applies, because a confirmation field is spelled in the corroborating source's
# vocabulary and never the alert's.


def test_the_alert_record_columns_are_pinned_on_every_source_it_names(refund_spec):
    """One block, three sources' worth of columns: the alert's own, and each `confirm_in` target."""
    from src.evidence import _alert_facts_field_relevance, trim_columns

    ledger, sessions = _serviced_cohort(), [
        session_row(login=f"u{i:03d}") for i in range(30)
    ]
    logs = {"shipment_ledger": ledger, "device_sessions": sessions}
    pinned = _alert_facts_field_relevance(refund_spec, logs)

    # `identify`/`declares` name the alert's own fields; `confirm_fields` are spelled in the
    # confirming source's vocabulary — `handler.badge`/`handler.device_login` on the ledger, the
    # same two facts as `badge`/`login` on the sessions source.
    assert {"tracking_code", "depot_code", "handler.badge", "handler.device_login"} <= pinned[
        "shipment_ledger"
    ], sorted(pinned["shipment_ledger"])
    assert {"badge", "login"} <= pinned["device_sessions"], sorted(pinned["device_sessions"])

    # And the two that the CONDITION pin does not reach are the point: without this block they are
    # dropped, which is what published a mismatch on a column the report called unavailable.
    _kept_before, dropped_before, _ = trim_columns(ledger, set())
    assert {"depot_code", "handler.device_login"} <= set(dropped_before), sorted(dropped_before)
    kept, _dropped, _ = trim_columns(ledger, pinned["shipment_ledger"])
    assert {"depot_code", "handler.device_login"} <= set(kept), sorted(kept)


def test_a_ruleset_declaring_no_alert_record_pins_nothing(collusion_spec):
    """Additive, like the condition pin: a ruleset without the block trims exactly as before."""
    from src.evidence import _alert_facts_field_relevance, trim_columns

    rows = _serviced_cohort()
    logs = {"shipment_ledger": rows}
    assert _alert_facts_field_relevance(collusion_spec, logs) == {}
    for absent in (None, {}, {"alert_record": "not a mapping"}, {"alert_record": {}}):
        assert _alert_facts_field_relevance(absent, logs) == {}, absent
    assert trim_columns(rows, set()) == trim_columns(rows, set())


def test_the_evidence_pack_publishes_the_alert_record_columns_too(pack, refund_spec):
    """Both pins ride through `build_evidence`, not just the one that was wired first."""
    from src.evidence import build_evidence

    logs = {"shipment_ledger": _serviced_cohort()}

    def _ledger(spec):
        ev = build_evidence(
            logs,
            {},
            [],
            pack.entity_bindings() if hasattr(pack, "entity_bindings") else {},
            ["RT00000000"],
            ruleset_spec=spec,
        )
        return next(s for s in ev.sources if s.source == "shipment_ledger")

    without, with_spec = _ledger(None), _ledger(refund_spec)
    for column in ("depot_code", "scans.scan"):
        assert column in without.dropped_columns, (column, without.dropped_columns)
        assert column in with_spec.kept_columns, (column, with_spec.kept_columns)
        assert column not in with_spec.dropped_columns, (column, with_spec.dropped_columns)


# The `no_exclusion_fired: insufficient` exit, over a PACK and not an inline dict.
# One ruleset of 8 across both installed packs declares it, so the note block that branch writes had
# one real reader and its only permanent coverage was a synthetic spec dict in `test_correlation.py`
# — which cannot disagree with the loader, the validator or a real condition set. So the exit runs
# here on `refund_fraud`'s own conditions and rows. Only the exit KEY is switched, on a deep copy,
# so the fixture on disk still declares nothing.
#
# The default ledger row already carries a FAIL on a non-decisive exclusion, so the flat sentence
# the engine used to write — "and NONE of them fired" — was false on this domain's most ordinary
# row too.


def _insufficient_exit(refund_spec):
    """`refund_fraud` with only its no-fire EXIT changed, on a copy the fixture never sees."""
    import copy as _copy

    spec = _copy.deepcopy(refund_spec)
    spec["no_exclusion_fired"] = "insufficient"
    return spec


def test_a_pack_declared_no_verdict_exit_names_the_fails_that_vote_nothing(refund_spec):
    """Reaching this exit does not mean nothing fired — and on this pack's default row, one did.

    `signature_captured` is a non-decisive exclusion, which is where every condition carrying
    no `polarity` lands. Its FAIL routes nowhere: step 3 tests the DECISIVE exclusions only,
    so the subject falls through to the declared no-fire exit with a real finding on it. The
    note has to name that finding, because this exit exists to hand the subject to a human and
    a FAIL that changes no verdict is still the material they have to read.
    """
    spec = _insufficient_exit(refund_spec)
    verdict = verdict_for(spec, [ledger_row()])
    subject = verdict.subjects[0]

    fails = [c.id for c in subject.checks if c.result == "fail"]
    assert fails == ["signature_captured"], (
        "THE PREMISE: exactly one condition fails on the default row, it is not decisive, and "
        "it is not an indicator — so it votes nothing and the rollup falls to step 4"
    )
    assert subject.verdict_class == "insufficient", subject.verdict
    assert subject.verdict == "INSUFFICIENT DATA", (
        "the label comes from the PACK's own map, so a wrong exit key would show up here"
    )

    note = next(n for n in subject.notes if n.startswith("no_verdict_reason="))
    assert "8 of 8 condition(s) were evaluated" in note, note
    assert "nothing that can decide this subject fired" in note, note
    assert "no fraud indicator fired" in note, note
    assert "1 condition(s) DID fail while voting nothing" in note, note
    assert "signature_captured" in note, note
    assert "read those as findings and not as an absence" in note, note
    assert "NONE of them fired" not in note, (
        "the claim this pack's most ordinary row falsified — it printed directly beneath the "
        "`fail` row above"
    )
    assert not any(
        word in note.lower() for word in ("record", "ticket", "office", "loyalty")
    ), f"a generic note may not name the domain that needed the fix: {note}"


def test_a_pack_declared_no_verdict_exit_counts_indicators_against_the_threshold(refund_spec):
    """A sub-threshold indicator FIRED, and *1 of the 2 required* is not *none*.

    This pack declares `indicator_threshold: 2` over three indicators, so one firing is the
    shape that reaches the no-fire exit having found something. Reporting it as "no fraud
    indicator fired" spends a reviewer's attention on the wrong subject in both directions.
    """
    spec = _insufficient_exit(refund_spec)
    row = ledger_row(
        **{"refund.reason_code": "OTHER", "pod.signature_captured": True}
    )
    verdict = verdict_for(spec, [row])
    subject = verdict.subjects[0]

    fired = [
        c.id
        for c in subject.checks
        if c.result == "fail"
        and next(
            k.get("polarity") for k in spec["conditions"] if k["id"] == c.id
        )
        == "fraud_indicator"
    ]
    assert fired == ["refund_reason_is_generic"], (
        "THE PREMISE: exactly one of the three indicators fires, and 1 < 2 so there is no "
        "corroboration — the subject must NOT be `fraud`"
    )
    assert subject.verdict_class == "insufficient", subject.verdict
    assert not any(c.result == "fail" for c in subject.checks if c.id != fired[0]), (
        "and nothing else fails, so the voteless-FAIL clause must stay silent here"
    )

    note = next(n for n in subject.notes if n.startswith("no_verdict_reason="))
    assert (
        "1 fraud indicator(s) fired, short of the 2 this procedure requires" in note
    ), note
    assert "no fraud indicator fired" not in note, note
    assert "DID fail while voting nothing" not in note, (
        "an indicator is not voteless — it voted and lost, which the clause above states"
    )
