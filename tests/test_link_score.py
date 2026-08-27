"""The deterministic score that gates `semi_auto`, and the ordering of the two refusals.

WHY THIS FILE EXISTS SEPARATELY FROM THE INVARIANT FILE
=======================================================
`tests/test_links_never_change_the_verdict.py` proves the advisory lane cannot move the verdict.
This file proves the one number inside that lane is arithmetic: reproducible from the finding
printed beside it, computed from the free rungs and a declared base rate only, and never from
prose or a model. Those are different failure modes — a score that drifted would keep the
verdict byte-identical while escalating on a guess, which is the spend this whole mechanism is
built to withhold.

It is deliberately a PURE test file: no pack, no fixture rows, no LLM. Every subject here is a
primitive in/primitive out (`src/link_escalation.py` is a leaf for exactly that reason), so these
assertions hold on every branch and for every domain — the shared-test rule from
`docs/architecture/engine-coverage.md`, applied to `min_escalation_score` and `score_weights`.

THE POLARITY IS THE THING MOST LIKELY TO BE "FIXED" BACK
========================================================
`src/stage_health.py` starts at 1.0 and subtracts. This starts at 0.0 and adds. Someone
harmonising the two would score a link about which nothing at all is known at 1.0 and escalate
it, so `test_the_score_ADDS_and_an_unassessable_link_scores_ZERO` is written to fail loudly on
that edit rather than on the arithmetic.
"""

import ast
import logging

import pytest

from src import link_escalation as le
from src.utils.paths import REPO_ROOT

LINKS_PY = REPO_ROOT / "src" / "links.py"


def _facts(**over):
    """The five primitives the climb establishes, with every free rung held.

    Full ladder by default so each test can knock ONE term out and read the delta — a builder
    whose default is empty makes every assertion a statement about two absent things at once.
    """
    facts = {
        "pivot_in_hand": True,
        "gate": "pass",
        "signal_fired": True,
        "signal_strength": None,
        # 3 of 27 is the shape a real declaration ships: measured, well over the corpus bar, and
        # discriminating rather than firing on most incidents.
        "base_rate": {"fires_on": 3, "of": 27, "measured": "2026-08-19"},
    }
    facts.update(over)
    return facts


# --- the polarity and the two vetoes -----------------------------------------------------


def test_the_score_ADDS_and_an_unassessable_link_scores_ZERO():
    """A candidate about which nothing is known is not a confident one.

    The stage-health polarity would return 1.0 here: nothing countable went wrong, because
    nothing was countable at all. That is the escalation this file exists to stop.
    """
    got = le.link_score({"pivot_in_hand": False, "gate": "", "signal_fired": False})
    assert got["score"] == 0.0
    # And the zero is EXPLAINED. A bare 0.0 is indistinguishable from a finding nobody scored.
    assert got["reasons"], "a held score must name what held it"


def test_a_FAILING_sibling_gate_is_a_VETO_and_not_a_missing_term():
    """`probed_negative` is the ladder's one free DISPROOF, and it outranks every weight.

    Asserted with every other term at its maximum, because a veto implemented as a weight of
    zero would still score 0.65 here and clear the default 0.60 threshold — the score would then
    escalate a pair whose own applicability test says it does not apply, and the printed state
    beside it would read `probed_negative`.
    """
    got = le.link_score(_facts(gate="fail"))
    assert got["score"] == 0.0
    assert len(got["reasons"]) == 1
    assert "does not apply" in got["reasons"][0]


def test_no_pivot_in_hand_is_a_VETO_because_a_referral_is_SCOPED_by_the_pivot():
    got = le.link_score(_facts(pivot_in_hand=False))
    assert got["score"] == 0.0
    assert len(got["reasons"]) == 1
    assert "nothing to scope a referral with" in got["reasons"][0]


def test_the_vetoes_are_asked_before_the_weights_are_read():
    """A veto must not be reachable through a weight override, in either direction.

    A deployment that zeroed `sibling_gate_holds` cannot turn a FAIL into a neutral term, and one
    that inflated the others cannot buy past it.
    """
    cfg = {"score_weights": {"sibling_gate_holds": 0.0, "entry_signal_fired": 1.0}}
    assert le.link_score(_facts(gate="fail"), cfg)["score"] == 0.0
    assert le.link_score(_facts(pivot_in_hand=False), cfg)["score"] == 0.0


# --- the four additive terms -------------------------------------------------------------


def test_the_weights_are_a_partition_of_one_so_a_FULL_ladder_scores_exactly_one():
    """Every free rung held and a perfectly discriminating declaration → 1.0.

    This is what makes the threshold readable as "the fraction of the free evidence this
    deployment insists on". A weight table summing to anything else turns the configured 0.6 into
    a number whose meaning depends on the table.
    """
    assert sum(le._DEFAULT_LINK_SCORE_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(le._DEFAULT_LINK_SCORE_WEIGHTS) == set(le.LINK_SCORE_SIGNALS)
    perfect = le.link_score(_facts(base_rate={"fires_on": 0, "of": 40}))
    assert perfect["score"] == pytest.approx(1.0)
    assert len(perfect["reasons"]) == 4


def test_each_term_contributes_its_OWN_weight_and_nothing_else():
    """Knock one rung out at a time and the delta is that rung's declared weight.

    Written as four subtractions from the full ladder rather than four additions to an empty one,
    so a term accidentally counted twice (the easy edit when a reason line is added) shows up as a
    delta that does not match its weight.
    """
    weights = dict(le._DEFAULT_LINK_SCORE_WEIGHTS)
    full = le.link_score(_facts(base_rate={"fires_on": 0, "of": 40}))["score"]

    no_gate = le.link_score(_facts(gate="unknown", base_rate={"fires_on": 0, "of": 40}))
    assert full - no_gate["score"] == pytest.approx(weights["sibling_gate_holds"])

    # Dropping the signal drops its OWN term and the discrimination term with it: an unfired
    # signal's base rate says nothing about this incident.
    no_signal = le.link_score(_facts(signal_fired=False))
    assert full - no_signal["score"] == pytest.approx(
        weights["entry_signal_fired"] + weights["signal_discriminates"]
    )
    # Pivot alone, with nothing else established — necessary, and on its own almost nothing.
    assert no_gate["score"] > no_signal["score"]
    bare = le.link_score({"pivot_in_hand": True, "gate": "unknown"})
    assert bare["score"] == pytest.approx(weights["pivot_in_hand"])


def test_an_ABSENT_strength_is_no_discount_and_a_declared_one_scales_only_ITS_term():
    """The precedent is `_advise`'s `if strength and strength < floor` — silence is not a zero.

    An author who declared no strength has said nothing about the signal, and inventing a penalty
    here would make this module disagree with the severity function one file over about what that
    silence means.
    """
    weights = dict(le._DEFAULT_LINK_SCORE_WEIGHTS)
    absent = le.link_score(_facts(signal_strength=None))
    blank = le.link_score(_facts(signal_strength=""))
    assert absent["score"] == blank["score"]

    half = le.link_score(_facts(signal_strength=0.5))
    assert absent["score"] - half["score"] == pytest.approx(
        weights["entry_signal_fired"] * 0.5
    )
    assert "declared strength 0.50" in "; ".join(half["reasons"])
    # A declared 1.0 is the full weight, not a special case — the same number either way.
    assert le.link_score(_facts(signal_strength=1.0))["score"] == absent["score"]


def test_the_base_rate_contributes_DISCRIMINATION_and_not_presence():
    """A signal that fires on most incidents is not a detector, so it earns less, not more.

    The measured lesson one directory over (`tender=CC` on 340 of 586 alerts). The term is
    `weight * (1 - fires_on/of)`, so a well-measured signal that fires constantly is scored below
    a well-measured signal that fires rarely — the opposite of counting the measurement itself.
    """
    rare = le.link_score(_facts(base_rate={"fires_on": 1, "of": 40}))["score"]
    common = le.link_score(_facts(base_rate={"fires_on": 38, "of": 40}))["score"]
    assert rare > common
    # Firing on every incident in the corpus earns exactly nothing, and never a negative.
    always = le.link_score(_facts(base_rate={"fires_on": 40, "of": 40}))
    unmeasured = le.link_score(_facts(base_rate=None))
    assert always["score"] == pytest.approx(unmeasured["score"])
    assert le.link_score(_facts(base_rate={"fires_on": 99, "of": 40}))["score"] >= 0.0


def test_discrimination_requires_the_CORPUS_bar_the_licence_requires():
    """Below `MIN_BASE_RATE_CORPUS` the term does not contribute at all.

    One bar, one home — the same function the authoring-time licence asks. A score that credited
    a rate counted over three runs would let a pair look confident on the number that is not yet
    allowed to license it, which is the two-homed bar this constant exists to prevent.
    """
    thin = {"fires_on": 0, "of": le.MIN_BASE_RATE_CORPUS - 1}
    fat = {"fires_on": 0, "of": le.MIN_BASE_RATE_CORPUS}
    assert le.link_score(_facts(base_rate=thin))["score"] < (
        le.link_score(_facts(base_rate=fat))["score"]
    )
    # A stub is the honest placeholder, and it is unmeasured too.
    stub = le.link_score(_facts(base_rate={"kind": "stub"}))
    assert stub["score"] == pytest.approx(
        le.link_score(_facts(base_rate=thin))["score"]
    )


def test_the_score_never_raises_and_never_leaves_the_unit_interval():
    """It feeds an advisory lane, which may not fail the run it advises on."""
    for facts in (
        None,
        {},
        {"pivot_in_hand": True, "gate": None, "signal_fired": "yes"},
        {
            "pivot_in_hand": 1,
            "gate": "PASS",
            "signal_fired": True,
            "base_rate": "prose",
        },
        {
            "pivot_in_hand": True,
            "gate": "pass",
            "signal_fired": True,
            "signal_strength": "not a number",
            "base_rate": {"fires_on": "x", "of": "y"},
        },
    ):
        got = le.link_score(facts)
        assert 0.0 <= got["score"] <= 1.0
        assert isinstance(got["reasons"], list)
        assert all(isinstance(r, str) for r in got["reasons"])
    # A weight table that oversubscribes is clamped rather than trusted.
    big = le.link_score(_facts(), {"score_weights": {"entry_signal_fired": 40}})
    assert big["score"] == 1.0


def test_the_score_is_DETERMINISTIC_for_the_same_facts():
    assert le.link_score(_facts()) == le.link_score(_facts())


# --- the weights and the threshold, as CONFIG --------------------------------------------


def test_a_deployment_overrides_ONE_weight_and_keeps_the_other_three():
    """Merged over the defaults, exactly as `stage_health._weights` does it.

    Replacing rather than merging is how a signal added later silently keeps a weight of zero on
    every deployment that tuned one term.
    """
    got = le.link_score_weights({"score_weights": {"pivot_in_hand": 0.5}})
    assert got["pivot_in_hand"] == 0.5
    for code in ("sibling_gate_holds", "entry_signal_fired", "signal_discriminates"):
        assert got[code] == le._DEFAULT_LINK_SCORE_WEIGHTS[code]
    # No config at all, and a config that says nothing, are the defaults untouched.
    assert le.link_score_weights() == le._DEFAULT_LINK_SCORE_WEIGHTS
    assert le.link_score_weights({"score_weights": "nonsense"}) == (
        le._DEFAULT_LINK_SCORE_WEIGHTS
    )
    # And the defaults are not mutable through the accessor.
    got["pivot_in_hand"] = 99
    assert le._DEFAULT_LINK_SCORE_WEIGHTS["pivot_in_hand"] != 99


def test_an_unknown_weight_code_is_WARNED_about_and_dropped(caplog):
    """A typo must not buy a weight that contributes to nothing.

    From the config file, a silently accepted `sibling_gate_hold` reads as a tuned deployment; the
    scores it produces are the untuned ones.
    """
    with caplog.at_level(logging.WARNING, logger="src.link_escalation"):
        got = le.link_score_weights({"score_weights": {"sibling_gate_hold": 0.9}})
    assert got == le._DEFAULT_LINK_SCORE_WEIGHTS
    assert "sibling_gate_hold" in caplog.text
    # A non-numeric value on a KNOWN code keeps that code's default rather than zeroing it.
    assert le.link_score_weights({"score_weights": {"pivot_in_hand": "high"}}) == (
        le._DEFAULT_LINK_SCORE_WEIGHTS
    )


def test_the_threshold_is_a_CONFIG_value_and_a_blank_one_is_not_zero():
    """Blank means inherit. Zero means "act on anything", and they must not be the same key.

    The config patcher cannot delete a key, so a cleared field arrives as `""` — reading that as
    0.0 would ship `semi_auto` behaving identically to `auto` on the deployment that cleared it.
    """
    assert le.min_escalation_score() == le.DEFAULT_MIN_ESCALATION_SCORE
    assert le.min_escalation_score({}) == le.DEFAULT_MIN_ESCALATION_SCORE
    assert le.min_escalation_score({"min_escalation_score": None}) == (
        le.DEFAULT_MIN_ESCALATION_SCORE
    )
    assert le.min_escalation_score({"min_escalation_score": ""}) == (
        le.DEFAULT_MIN_ESCALATION_SCORE
    )
    assert le.min_escalation_score({"min_escalation_score": "junk"}) == (
        le.DEFAULT_MIN_ESCALATION_SCORE
    )
    assert le.min_escalation_score({"min_escalation_score": 0.25}) == 0.25
    assert le.min_escalation_score({"min_escalation_score": "0.9"}) == 0.9
    assert le.min_escalation_score({"min_escalation_score": 0}) == 0.0


def test_the_default_threshold_is_reachable_by_more_than_ONE_combination():
    """A threshold only one combination can reach is that combination with extra steps.

    Two independent routes over it, both without the discrimination term, so the number stays
    meaningful for the pairs whose declarations are honest stubs.
    """
    bar = le.DEFAULT_MIN_ESCALATION_SCORE
    gate_and_signal = le.link_score(_facts(base_rate=None))["score"]
    assert gate_and_signal >= bar
    signal_and_measurement = le.link_score(
        _facts(gate="unknown", base_rate={"fires_on": 0, "of": 40})
    )["score"]
    assert signal_and_measurement >= bar
    # And it is not reachable by the pivot alone, which is the floor and not evidence.
    assert le.link_score({"pivot_in_hand": True})["score"] < bar


# --- the score GATE inside the one resolution seam ---------------------------------------


def _licensed(**over):
    """`resolve_link_mode` kwargs for a link whose target gate PASSED on this run's rows."""
    kwargs = {
        "config_mode": "semi_auto",
        "gate_outcome": "pass",
        "escalation_available": True,
    }
    kwargs.update(over)
    return kwargs


def test_semi_auto_BELOW_the_threshold_composes_a_referral_and_says_so_with_both_numbers():
    got = le.resolve_link_mode(**_licensed(score=0.5, min_score=0.6))
    assert got["mode"] == le.MANUAL_LINK_MODE
    assert got["source"] == "score"
    assert "0.50" in got["note"] and "0.60" in got["note"]
    # The action printed is the action TAKEN, not the mode that was asked for.
    assert got["action"] == le.proposed_action(le.MANUAL_LINK_MODE)


def test_semi_auto_AT_the_threshold_acts_because_the_comparison_is_not_strict():
    """At-or-above, so a deployment setting the bar to a reachable score can reach it."""
    got = le.resolve_link_mode(**_licensed(score=0.6, min_score=0.6))
    assert (got["mode"], got["source"]) == ("semi_auto", "config")
    assert got["note"] == ""


def test_auto_is_NOT_score_gated_by_its_own_definition():
    """The operator asking for `auto` has already answered this question.

    A score gate on `auto` would make the three modes two, and the middle one unreachable.
    """
    got = le.resolve_link_mode(
        **_licensed(config_mode="auto", score=0.0, min_score=0.9)
    )
    assert (got["mode"], got["source"]) == ("auto", "config")
    assert got["note"] == ""


def test_rung_1_is_asked_before_the_score_gate():
    """A link whose target gate did not hold is `clamp`, never `score`, and the ordering is why.

    The two refusals have different remedies: `clamp` says this run never established that the
    target procedure applies (the remedy, if any, is more evidence), `score` says it did and the
    confidence fell short, which wants nothing at all. Reporting a below-threshold score on a link
    that was refused by the gate sends the reader to weigh a number that was never consulted.
    """
    got = le.resolve_link_mode(
        **_licensed(gate_outcome="unknown", score=0.1, min_score=0.6)
    )
    assert got["source"] == "clamp"
    assert "does not license" in got["note"] and "UNRESOLVED" in got["note"]
    assert "0.10" not in got["note"]


def test_every_non_PASS_rung_1_outcome_clamps_and_names_ITSELF():
    """Six outcomes, one licence, six sentences — because the reader's next action differs.

    `fail` is the target saying it does not apply; `unknown` is rows that were not retrieved;
    `no_sources` is a retrieval gap; `no_subject` is a binding gap; `no_gate` is a pack that never
    authored an applicability test. One sentence for all five would send a reader chasing data on a
    link whose data was never the problem.
    """
    seen = set()
    for outcome in ("fail", "unknown", "no_gate", "no_sources", "no_subject", ""):
        got = le.resolve_link_mode(
            **_licensed(gate_outcome=outcome, config_mode="auto")
        )
        assert (got["mode"], got["source"]) == (le.MANUAL_LINK_MODE, "clamp"), outcome
        assert got["note"] not in seen or outcome == ""
        seen.add(got["note"])
    # And the one that does license, for contrast — same kwargs, opposite answer.
    assert le.resolve_link_mode(**_licensed(config_mode="auto"))["mode"] == "auto"


def test_gate_permits_is_PASS_and_not_merely_not_FAIL():
    """`no_gate` is silence, and an engine reading silence as consent escalates everything.

    The bar has one home for the same reason: three callers ask it (the seam, the probe rung and
    the child rung), and a licence with two implementations is one that gets loosened in one of
    them.
    """
    assert le.gate_permits("pass") is True
    assert le.gate_permits("PASS ") is True
    for outcome in ("fail", "unknown", "no_gate", "no_sources", "no_subject", "", None):
        assert le.gate_permits(outcome) is False, outcome


def test_a_refusal_falls_back_to_the_MANUAL_mode_and_not_to_the_DEFAULT():
    """The two used to be one constant, and the day the default moved that would have escalated.

    `DEFAULT_LINK_MODE` is what a deployment gets when nobody says anything; `MANUAL_LINK_MODE` is
    what a refusal leaves behind. With the default now `semi_auto`, a single name would make every
    clamp and every score refusal fall back INTO an escalating mode.
    """
    assert le.DEFAULT_LINK_MODE == "semi_auto"
    assert le.MANUAL_LINK_MODE == "planned"
    assert le.MANUAL_LINK_MODE not in le.ESCALATING_MODES
    for outcome in ("fail", "unknown", ""):
        assert le.resolve_link_mode(gate_outcome=outcome)["mode"] == le.MANUAL_LINK_MODE
    assert (
        le.resolve_link_mode(gate_outcome="pass", score=0.0, min_score=0.9)["mode"]
        == le.MANUAL_LINK_MODE
    )


def test_the_DEFAULT_with_nothing_declared_is_semi_auto():
    """Nobody configured anything, and a gate that held: the deployment default acts.

    `planned` by default is a lane nobody exercises; `auto` by default cannot express "a thin score
    is a case a human should see". What holds the spend down is the gate plus the threshold.
    """
    got = le.resolve_link_mode(gate_outcome="pass", escalation_available=True)
    assert (got["mode"], got["source"]) == ("semi_auto", "default")


def test_a_declaration_that_LOWERS_a_wider_escalating_ask_is_NAMED():
    """`planned` from a declaration and `planned` from nobody-asked are the same two words.

    Measured live: a run with the deployment set to `auto` resolved one candidate to
    mode=`planned`, source=`pack`, rung 1 = PASS — and an EMPTY note, beside nine clamped
    candidates each carrying a full sentence. That is the row where the reader's expectation of
    escalation is strongest (its gate held), so the silence lands exactly where the default reading
    — nobody set this pair up — is most wrong, and an operator concludes their setting was lost.

    The seam explained the three outcomes it imposes ITSELF and none that a declaration imposes,
    which is the same asymmetry the clamp's own comment exists to prevent one layer up.
    """
    got = le.resolve_link_mode(
        config_mode="auto",
        pack_mode="planned",
        gate_outcome="pass",
        escalation_available=True,
        score=0.5,
        min_score=0.6,
    )
    assert (got["mode"], got["source"]) == (le.MANUAL_LINK_MODE, "pack")
    assert "'auto' was asked for by the config layer" in got["note"], got["note"]
    assert "narrower pack layer declares 'planned'" in got["note"], got["note"]
    # The action printed is the action TAKEN, as everywhere else in this seam.
    assert got["action"] == le.proposed_action(le.MANUAL_LINK_MODE)
    # And the AUTHOR stays the layer that declared it: a clamp and the score gate are the engine
    # refusing a setting, this is a setting. Three sources, three different remedies.
    assert got["source"] not in ("clamp", "score")
    # The same precedence one layer down — a per-job hold over the pack's own `auto`.
    other = le.resolve_link_mode(
        pack_mode="auto", job_mode="planned", gate_outcome="pass", escalation_available=True
    )
    assert (other["mode"], other["source"]) == (le.MANUAL_LINK_MODE, "job")
    assert "'auto' was asked for by the pack layer" in other["note"], other["note"]
    assert "narrower job layer" in other["note"], other["note"]
    # Both wider asks named when both were made: an operator who set one of the two cannot tell
    # from a single-layer sentence whether the setting they own is the one being overruled.
    both = le.resolve_link_mode(
        config_mode="auto",
        pack_mode="semi_auto",
        job_mode="planned",
        gate_outcome="pass",
        escalation_available=True,
    )
    assert "config layer" in both["note"] and "pack layer" in both["note"], both["note"]


def test_a_hold_NOTHING_wider_asked_for_stays_SILENT_because_the_DEFAULT_escalates():
    """The note is a claim about an ASK, and the deployment default is not one.

    `DEFAULT_LINK_MODE` is itself escalating, so a predicate that compared the winner against the
    mode-so-far instead of against an explicit declaration would print "overruling a wider setting"
    on every `planned` declaration in every default deployment — boilerplate on the majority of
    candidates, which is how a reader learns to skip the line that matters. That is the same
    inversion that landed the day this default moved from `planned` to `semi_auto`.
    """
    for kwargs in (
        {"pack_mode": "planned"},
        {"config_mode": "planned"},
        {"job_mode": "planned"},
        {"config_mode": "planned", "pack_mode": "planned"},
    ):
        got = le.resolve_link_mode(
            gate_outcome="pass", escalation_available=True, **kwargs
        )
        assert got["mode"] == le.MANUAL_LINK_MODE, kwargs
        assert got["note"] == "", (kwargs, got["note"])
    # And nobody declaring anything at all is not a hold either — that pair reaches the default,
    # which escalates, so it takes one of the other three branches or none.
    assert le.resolve_link_mode(gate_outcome="pass", escalation_available=True)["mode"] == (
        le.DEFAULT_LINK_MODE
    )


def test_a_DECLARED_hold_is_a_HOLD_even_where_rung_1_would_also_have_refused():
    """Two reasons for one outcome, and the declaration is the one that decided it.

    The clamp explains an escalating mode rung 1 took AWAY. Here nothing was taken away, because
    the narrower layer never granted it — so reporting this as a clamp would send a reader to go and
    retrieve rows for a link that stays held after they arrive. One note, not two.
    """
    got = le.resolve_link_mode(config_mode="auto", pack_mode="planned", gate_outcome="fail")
    assert (got["mode"], got["source"]) == (le.MANUAL_LINK_MODE, "pack")
    assert "narrower pack layer declares 'planned'" in got["note"], got["note"]
    assert "does not license" not in got["note"], got["note"]
    assert got["note"].count("asked for by") == 1, got["note"]


def test_the_log_line_prints_a_mode_where_a_REASON_exists_and_not_a_LIST_of_refusals():
    """Read as text, because the alternative is a full `assess_links` run to reach one log call.

    The predicate enumerated the two engine-imposed sources, so the newest of the three reasons a
    candidate sits at a composed referral printed nothing — on the one row whose `planned` is least
    expected. Keyed on the presence of a reason it cannot go stale the next time a fourth arrives.
    """
    text = LINKS_PY.read_text(encoding="utf-8")
    assert 'if f.mode != MANUAL_LINK_MODE or str(getattr(f, "mode_note"' in text, (
        "the assessment log line must print a mode wherever the resolution seam wrote a reason"
    )
    assert 'f.mode_source in ("clamp", "score")' not in text


def test_an_UNMEASURED_pair_escalates_exactly_like_a_measured_one():
    """The demotion, asserted at the seam: a base rate may ADD confidence and never withhold it.

    A corpus nobody has counted is the normal state of a young deployment, and a bar that read an
    absent measurement as a refusal made the automatic path unreachable by construction.
    """
    facts = {"pivot_in_hand": True, "gate": "pass", "signal_fired": True}
    unmeasured = le.link_score(dict(facts, base_rate={"kind": "stub"}))["score"]
    measured = le.link_score(dict(facts, base_rate={"fires_on": 1, "of": 40}))["score"]
    thin = le.link_score(dict(facts, base_rate={"fires_on": 1, "of": 3}))["score"]
    # Additive only: the measured one is HIGHER, and neither of the others is penalised.
    assert measured > unmeasured
    assert thin == unmeasured
    # And every one of the three clears the bar off this run's own evidence alone.
    for score in (unmeasured, measured, thin):
        got = le.resolve_link_mode(
            config_mode="semi_auto", gate_outcome="pass", score=score
        )
        assert (got["mode"], got["source"]) == ("semi_auto", "config")


def test_clamp_and_score_are_DIFFERENT_sources_and_neither_is_a_configurable_layer():
    for name in ("clamp", "score"):
        assert name in le.MODE_SOURCES
    assert len(set(le.MODE_SOURCES)) == len(le.MODE_SOURCES)
    # The four authors are the layers; the two refusals are what those layers leave behind.
    assert le.MODE_SOURCES[:4] == ("default", "config", "pack", "job")


def test_score_None_leaves_every_pre_existing_resolution_BYTE_identical():
    """The gate must be invisible to a caller with no score to offer.

    `set_job_link_modes` resolves a mode for a pair this run has no link for YET — there is no
    assessment, so there is no score, and a 0.0 there would refuse the setting the operator just
    made in the same request.
    """
    for mode in le.LINK_MODES:
        for outcome in ("pass", "fail", "unknown", ""):
            for available in (True, False):
                kwargs = {
                    "config_mode": mode,
                    "gate_outcome": outcome,
                    "escalation_available": available,
                }
                assert le.resolve_link_mode(**kwargs) == le.resolve_link_mode(
                    score=None, **kwargs
                )


def test_a_never_scored_finding_gates_in_the_CONSERVATIVE_direction():
    """Absent means 0.0 at the seam that reads the finding, which withholds the spend.

    `apply_link_mode` re-runs on an operator's override, where the score rides on the finding
    rather than being recomputed. A finding whose score never landed must not read as a confident
    one.
    """

    class _Bare:
        target_use_case = "sibling"
        gate_outcome = "pass"

    bare = _Bare()
    got = le.apply_link_mode(
        bare, config={"escalation_mode": "semi_auto", "max_probes_per_run": 2}
    )
    assert (got["mode"], got["source"]) == (le.MANUAL_LINK_MODE, "score")
    assert bare.mode_source == "score"


def test_the_score_on_the_FINDING_is_what_an_override_is_adjudicated_against():
    """One number, computed once, read by both callers — never recomputed downstream.

    The link pass has the gate outcome and the declaration's raw `base_rate` in scope; the
    override handler has neither (the finding carries `base_rate` as prose, for the reader). So a
    second attempt at this number would have to reconstruct it from a sentence.
    """

    class _Scored:
        target_use_case = "sibling"
        gate_outcome = "pass"
        link_score = 0.9

    cfg = {"escalation_mode": "semi_auto", "max_probes_per_run": 2}
    got = le.apply_link_mode(_Scored(), config=cfg)
    assert (got["mode"], got["source"]) == ("semi_auto", "config")

    low = _Scored()
    low.link_score = 0.2
    assert le.apply_link_mode(low, config=cfg)["source"] == "score"

    # And a deployment that lowers the bar lets the same finding act.
    lowered = dict(cfg, min_escalation_score=0.1)
    assert le.apply_link_mode(low, config=lowered)["source"] == "config"


# --- the climb stamps a score at EVERY exit ----------------------------------------------


def test_every_exit_of_the_climb_stamps_a_score():
    """Read as an AST, because the four early exits are the ones a fixture is least likely to hit.

    A finding that returns unscored reads 0.0 to the escalation seam and "never scored" to
    nothing at all — the same class as a source that did not answer being indistinguishable from
    one that answered with nothing. So the structural claim is asserted directly: inside
    `_assess_one`, every `return` is preceded by a `_score(...)` call in its own block.
    """
    tree = ast.parse(LINKS_PY.read_text(encoding="utf-8"))
    climb = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_assess_one"
    )

    def _walk(body):
        seen = 0
        exits = 0
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Name) and func.id == "_score":
                    seen += 1
            elif isinstance(stmt, ast.Return):
                exits += 1
                assert seen, "a return in _assess_one that no _score() precedes"
            for attr in ("body", "orelse", "finalbody"):
                nested = getattr(stmt, attr, None)
                if isinstance(nested, list) and nested:
                    inner_seen, inner_exits = _walk(nested)
                    seen += inner_seen
                    exits += inner_exits
        return seen, exits

    scored, returns = _walk(climb.body)
    assert returns >= 5, "the climb's five exits are the ones this test is about"
    assert scored >= returns
