"""Which procedure adjudicated, and whether anything chose it.

`select_correlation_spec` has always returned `None` when a pack holds more than one
procedure and the incident's summary scores against none of them. Four downstream sites then
substitute the pack's default ruleset, so that honest `None` reached no reader: the
substituted ruleset's conditions resolve against real rows, the stage reports success, and the
report reads exactly as confident as one whose procedure was recognised. The failure mode is a
confident WRONG verdict rather than a missing one.

Nothing here changes WHICH ruleset runs. Every test is about the run being able to SAY how its
procedure was reached — `SelectionBasis` carried out of the selector, one gating stage-health
finding, a note on the verdict and the brief, and, for a pack that asks for it, a conversion to
that ruleset's own already-declared `reject` state.

Three properties the file exists to hold down:

* A pack that declares nothing is byte-identical. `sole_spec` and `no_specs` are silent, the
  default policy leaves every field alone but the note, and a *stale* pin still falls back to
  scoring (`test_link_children.py` deliberately prefers a scored guess to no verdict).
* An abstention that cannot be spelled is refused rather than invented: a ruleset with no
  `reject` label keeps its verdict, loudly.
* The finding gates on its own, at the weight `required_source_not_queried` carries, because a
  procedure nobody chose is the same class of defect as a dependency nobody retrieved.

The pack is ``knowledge/mock_domain/``.
"""

import logging
import shutil
from types import SimpleNamespace

import pytest

from src.api_call_generator import ApiCallGenerator
from src.correlation import (
    SELECTION_BASES,
    CorrelationModule,
    SelectionBasis,
    select_correlation_spec,
    select_correlation_spec_explained,
)
from src.knowledge.pack import load_knowledge_pack
from src.pipeline_runner import _summ_correlation
from src.report_generation import ReportGenerationModule
from src.models.pydantic_models import (
    CorrelationResult,
    InvestigationBrief,
    SubjectVerdict,
    ValidationVerdict,
)
from src.stage_health import _DEFAULT_WEIGHTS, score_stage, stage_threshold
from tests.mock_domain_links import MOCK_DOMAIN_DIR

# The two fixture procedures. `refund_fraud` is what the summaries below are written to select;
# `courier_collusion` is the rival, and — because there is no flat-root `rulesets.yaml` — also
# the pack's default ruleset, which is what makes the substitution visible without any edit.
_MATCHED = "refund_fraud"
_RIVAL = "courier_collusion"

#: Scores `_MATCHED` and nothing else.
_RECOGNISED = "refund claim on a bare shipment"
#: Scores nothing at all: no procedure in this pack recognises it.
_UNRECOGNISED = "an unrelated database migration failed overnight"


def _pack():
    return load_knowledge_pack(MOCK_DOMAIN_DIR)


def _analysis(summary=_RECOGNISED, pin=""):
    """Only the four fields the selector reads."""
    return SimpleNamespace(
        incident_summary=summary,
        pinned_use_case=pin,
        initial_hypotheses=[],
        key_investigation_areas=[],
    )


def _abstaining_pack(tmp_path, pack_dir_name="abstaining", reject=True, policy="abstain"):
    """A copy of the fixture pack that declares `adjudication_policy` at its root.

    The key is read off the flat-root `rulesets.yaml`, which `mock_domain` does not ship, so
    the fixture writes one — which is also the honest fixture, because a pack declaring a
    policy is a pack that declares a `default_ruleset` beside it. `reject=False` produces the
    refusal case: a pack asking to abstain in a ruleset that has no word for it.
    """
    root = tmp_path / pack_dir_name
    shutil.copytree(MOCK_DOMAIN_DIR, root)
    (root / "rulesets.yaml").write_text(
        f"default_ruleset: {_MATCHED}\nadjudication_policy: {policy}\n",
        encoding="utf-8",
    )
    if reject:
        rules = root / "use_cases" / _MATCHED / "rules.yaml"
        text = rules.read_text(encoding="utf-8")
        anchor = '      out_of_scope: "OUT OF SCOPE — PROCEDURE DOES NOT APPLY"'
        # Asserted rather than assumed: a pack edit that renames or re-indents the label block
        # would otherwise leave this fixture declaring no `reject` label and every abstention
        # test below passing for the refusal reason instead of the one it is testing.
        assert anchor in text, "the fixture pack's label block moved; re-anchor the insertion"
        rules.write_text(
            text.replace(anchor, anchor + '\n      reject: "RETURNED — NO PROCEDURE"', 1),
            encoding="utf-8",
        )
    pack = load_knowledge_pack(str(root))
    assert pack.adjudication_policy() == ("abstain" if policy == "abstain" else "default")
    return pack


def _verdict(pack, key=_MATCHED, subjects=2):
    """A verdict shaped the way `evaluate_verdict` builds one: labels come from the ruleset."""
    labels = dict((pack.ruleset_spec(key) or {}).get("labels") or {})
    return ValidationVerdict(
        label_scheme=key,
        labels=labels,
        summary=f"{labels.get('fraud', 'FRAUD')}: {subjects}",
        notification_draft="Dear owner, the subject below is confirmed fraudulent.",
        subjects=[
            SubjectVerdict(
                subject_type="shipment",
                subject_value=f"S{n}",
                verdict=labels.get("fraud", "FRAUD"),
                verdict_class="fraud",
                lock_target={"identity": f"courier-{n}"},
            )
            for n in range(subjects)
        ],
    )


def _result(pack, key=_MATCHED, subjects=2, brief=True):
    return CorrelationResult(
        record_count=10,
        verdict=_verdict(pack, key, subjects),
        brief=InvestigationBrief(use_case=key) if brief else None,
    )


def _notes(holder):
    return [n for n in holder.notes if n.startswith("procedure_unselected=")]


# --- the selector reports how it was reached ---------------------------------------


def test_a_matching_incident_reports_the_score_that_selected_it():
    spec, basis = select_correlation_spec_explained(_pack(), _analysis())
    assert spec and spec["use_case"] == _MATCHED
    assert basis.basis == "scored"
    assert basis.score > 0
    assert basis.defaulted is False
    assert basis.spec_count == 2
    # Best-first, so the first name is the one that ran and the rest are what an operator
    # could pin instead.
    assert [name for name, _ in basis.candidates] == [_MATCHED, _RIVAL]


def test_an_unrecognised_incident_returns_no_spec_and_says_the_default_is_a_guess():
    """The whole defect in one assertion: `None` here, and every caller adjudicates anyway.

    `defaulted` is the only record that the ruleset which ran was not selected — the
    conditions still evaluate against real rows either way.
    """
    spec, basis = select_correlation_spec_explained(_pack(), _analysis(_UNRECOGNISED))
    assert spec is None
    assert basis.basis == "no_match"
    assert basis.defaulted is True
    assert basis.spec_count == 2
    assert basis.score == 0.0
    # Both rivals are named with their zero scores: the finding's remedy is to pin one.
    assert sorted(name for name, _ in basis.candidates) == [_RIVAL, _MATCHED]
    assert all(score == 0.0 for _, score in basis.candidates)


def test_a_resolving_pin_is_never_second_guessed():
    """A referral's pin bypasses scoring, and is not reported as a fallback."""
    spec, basis = select_correlation_spec_explained(
        _pack(), _analysis(_RECOGNISED, pin=_RIVAL)
    )
    assert spec and spec["use_case"] == _RIVAL  # the text scores the OTHER one
    assert basis.basis == "pinned"
    assert basis.defaulted is False


def test_a_stale_pin_still_falls_back_to_scoring():
    """Guards `test_link_children.py::test_the_pin_decides_the_procedure_...`.

    A job document can outlive the pack edit that deleted a use case, and a scored guess is a
    better answer than no verdict — so a pin naming nothing must not become `no_match`, which
    would abstain on a pack that asked to.
    """
    spec, basis = select_correlation_spec_explained(
        _pack(), _analysis(_RECOGNISED, pin="deleted_proc")
    )
    assert spec and spec["use_case"] == _MATCHED
    assert basis.basis == "scored"
    assert basis.defaulted is False


def test_a_one_procedure_pack_reports_sole_spec_because_nothing_scored_it():
    """The fifth fallback path, and the reason it is not a finding.

    With one spec every token weighs `(1 - 1) / 1 = 0`, so the lone procedure wins unscored
    whatever the incident says. That is worth naming — it is why the health scorer stays
    silent here rather than reporting a zero-score win on every single-procedure deployment.
    """
    only = {"use_case": _MATCHED, "title": "refund fraud", "keys": []}
    pack = SimpleNamespace(correlation_specs=lambda: [only])
    spec, basis = select_correlation_spec_explained(pack, _analysis(_UNRECOGNISED))
    assert spec is only
    assert basis.basis == "sole_spec"
    assert basis.defaulted is False
    assert basis.spec_count == 1


@pytest.mark.parametrize("pack", [None, SimpleNamespace(correlation_specs=lambda: [])])
def test_a_pack_with_no_correlation_specs_reports_no_specs(pack):
    """Not a defect: a pack may ship playbooks with no `correlation:` block at all."""
    spec, basis = select_correlation_spec_explained(pack, _analysis())
    assert spec is None
    assert basis.basis == "no_specs"
    assert basis.defaulted is False


@pytest.mark.parametrize(
    "summary,pin",
    [
        (_RECOGNISED, ""),
        (_UNRECOGNISED, ""),
        (_RECOGNISED, _RIVAL),
        (_RECOGNISED, "deleted_proc"),
    ],
)
def test_the_wrapper_answers_exactly_what_the_explained_form_selected(summary, pin):
    """The ten-odd existing callers must be reading the same selection they always did."""
    pack = _pack()
    analysis = _analysis(summary, pin)
    assert select_correlation_spec(pack, analysis) == (
        select_correlation_spec_explained(pack, analysis)[0]
    )


def test_every_basis_is_declared_and_only_no_match_defaults():
    """`defaulted` is what the pipeline branches on, so it is asserted over the closed set."""
    assert set(SELECTION_BASES) == {
        "pinned",
        "scored",
        "sole_spec",
        "no_match",
        "no_specs",
    }
    for basis in SELECTION_BASES:
        assert SelectionBasis(basis).defaulted is (basis == "no_match")


def test_the_margin_is_the_share_of_the_win_the_runner_up_did_not_hold():
    assert SelectionBasis("scored", score=1.0, runner_up=0.0).margin == 1.0
    assert SelectionBasis("scored", score=1.0, runner_up=0.9).margin == pytest.approx(0.1)
    assert SelectionBasis("scored", score=0.5, runner_up=0.5).margin == 0.0
    # Nothing scored: an unrivalled zero is not a confident win.
    assert SelectionBasis("no_match", score=0.0).margin == 0.0
    # A runner-up cannot beat the winner, but arithmetic on a hand-built basis must not go
    # negative and then read as a confident win against a floor.
    assert SelectionBasis("scored", score=0.5, runner_up=0.9).margin == 0.0


def test_the_reported_scores_are_all_floats():
    """A candidate list mixing `0` and `0.0` reads as two different measurements.

    `weigh` sums a generator, and an empty sum is an `int` — so the zero-scoring rivals came
    back typed differently from the winner in the one place both are printed side by side.
    """
    for summary in (_RECOGNISED, _UNRECOGNISED):
        _, basis = select_correlation_spec_explained(_pack(), _analysis(summary))
        assert basis.candidates
        for name, score in basis.candidates:
            assert isinstance(name, str)
            assert isinstance(score, float), f"{name} scored {score!r}, not a float"
        as_dict = basis.to_dict()
        assert as_dict["basis"] == basis.basis
        assert as_dict["margin"] == pytest.approx(basis.margin, abs=1e-4)


# --- the four fallback sites now report ---------------------------------------------


def test_the_verdict_and_brief_both_carry_the_substitution():
    """The note the report renders, on every subject and on the brief.

    Both, because they are read by different consumers: the verdict is the acceptance
    artifact's table and the brief is what the two narrating LLM stages are grounded on. A
    brief that omits it grounds the narration on a procedure it has no reason to doubt.
    """
    pack = _pack()
    module = CorrelationModule({}, None, knowledge_pack=pack)
    result = _result(pack)
    _, basis = select_correlation_spec_explained(pack, _analysis(_UNRECOGNISED))

    module._annotate_procedure_selection(result, basis, _RIVAL)

    for subject in result.verdict.subjects:
        assert len(_notes(subject)) == 1
    assert len(_notes(result.brief)) == 1
    detail = _notes(result.brief)[0]
    # The three things a reader can act on: which ruleset ran, that it was not chosen, and how
    # to choose one. Naming the rivals is what turns the finding into one click.
    assert _RIVAL in detail
    assert "pinned_use_case" in detail
    assert _MATCHED in detail


def test_every_reader_of_one_unrecognised_run_carries_it():
    """One `no_match` run, and all five surfaces that could hide it, in a single pass.

    The per-reader tests around this one each hold their own seam, and every one of them can
    pass while a run still hides the substitution somewhere — the finding scored but no note
    rendered, the note written but the job summary reading empty. This asserts the whole set
    off ONE selection, which is what a live run would produce.

    It is also as close to a live firing as this path gets on a mature pack: `no_match` needs
    EVERY procedure to score zero on both the summary and the hypotheses, and a ten-procedure
    pack's vocabulary is wide enough that an out-of-domain incident still scored 2.5 vs 1.5
    live (`docs/architecture/verdict-engine.md`). So the two-procedure fixture is the only
    place the state is reachable, and the readers are what has to be checked there.
    """
    pack = _pack()
    module = CorrelationModule({}, None, knowledge_pack=pack)
    result = _result(pack)
    # Every OTHER correlation signal satisfied, so the score is this finding alone.
    result.aggregations = {"resolved_correlation_keys": [{"entity_hint": "shipment"}]}
    result.findings = ["one narrated finding"]
    spec, basis = select_correlation_spec_explained(pack, _analysis(_UNRECOGNISED))
    assert spec is None and basis.basis == "no_match"

    module._annotate_procedure_selection(result, basis, _RIVAL)

    # 1. the gate: it scores, and on its own.
    health = score_stage(
        "correlation",
        result,
        SimpleNamespace(
            outputs={},
            modules={},
            config={},
            stage_facts={"correlation": {"procedure_selection": basis.to_dict()}},
        ),
    )
    assert health.reason_codes == ["procedure_not_selected"]
    assert health.gate_recommended is True

    # 2/3. the verdict's subjects and the brief the narrating stages are grounded on.
    assert all(_notes(s) for s in result.verdict.subjects)
    assert _notes(result.brief)

    # 4. the report the operator accepts: rendered, in words, not as a note prefix.
    section = ReportGenerationModule._verdict_section(result)
    body = "\n".join(str(v) for v in (section or {}).values())
    assert "PROCEDURE NOT SELECTED" in body
    assert "procedure_unselected=" not in body

    # 5. the job summary the UI run card reads.
    assert _RIVAL in _summ_correlation(result)["procedure_unselected"]


@pytest.mark.parametrize("summary,pin", [(_RECOGNISED, ""), (_RECOGNISED, _RIVAL)])
def test_a_selected_procedure_is_annotated_nowhere(summary, pin):
    """Every basis but `no_match` chose the ruleset that ran, so a note would be noise."""
    pack = _pack()
    module = CorrelationModule({}, None, knowledge_pack=pack)
    result = _result(pack)
    before = result.model_dump_json()
    _, basis = select_correlation_spec_explained(pack, _analysis(summary, pin))

    module._annotate_procedure_selection(result, basis, _MATCHED)

    assert result.model_dump_json() == before


def test_an_annotation_that_cannot_be_written_does_not_fail_the_stage():
    """A note is not worth a run. Asserted with a verdict whose notes cannot be appended to."""
    pack = _pack()
    module = CorrelationModule({}, None, knowledge_pack=pack)
    broken = SimpleNamespace(
        verdict=SimpleNamespace(subjects=[SimpleNamespace(notes=None)]), brief=None
    )
    _, basis = select_correlation_spec_explained(pack, _analysis(_UNRECOGNISED))

    module._annotate_procedure_selection(broken, basis, _RIVAL)  # must not raise


def test_the_pass_decision_names_the_substitution_rather_than_the_default():
    """`pipeline_runner._adjudicating_ruleset_key`: same key, a reason that distinguishes.

    An operator reading "the pack's default ruleset" cannot tell a pack with one procedure
    from an incident no procedure recognised, and only one of those is a defect.
    """
    from src.pipeline_runner import JobContext, _adjudicating_ruleset_key

    pack = _pack()

    def ctx_for(summary):
        return JobContext(
            job_id="j",
            incident={},
            modules={
                "api_call": SimpleNamespace(knowledge_pack=pack),
                "correlation": CorrelationModule({}, None, knowledge_pack=pack),
            },
            config={},
            outputs={"understanding": SimpleNamespace(analysis=_analysis(summary))},
        )

    key, how = _adjudicating_ruleset_key(ctx_for(_RECOGNISED))
    assert key == _MATCHED
    assert "NO procedure" not in how

    key, how = _adjudicating_ruleset_key(ctx_for(_UNRECOGNISED))
    assert key == pack.default_ruleset_key()  # unchanged: the pass decision still resolves
    assert "NO procedure matched" in how


def test_the_planner_says_its_hard_dependencies_come_from_the_default(caplog):
    """`api_call_generator` plans the DEFAULT ruleset's sources for an unrecognised incident.

    It still does — a plan is repairable and a missing source is not — but this is where the
    substitution starts costing scan time, so it is where it is logged.
    """
    pack = _pack()
    gen = ApiCallGenerator({}, llm_client=None, knowledge_pack=pack)

    with caplog.at_level(logging.WARNING):
        key = gen._adjudicating_ruleset_key(_analysis(_UNRECOGNISED))
    assert key == pack.default_ruleset_key()
    assert "No procedure matched" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert gen._adjudicating_ruleset_key(_analysis(_RECOGNISED)) == _MATCHED
    assert "No procedure matched" not in caplog.text


def test_the_pack_accessor_still_answers_the_default_because_it_cannot_know_why():
    """`ruleset_spec("")` is deliberately unchanged: it is handed a key, not an incident.

    The fourth fallback site is a pure accessor with no access to the selection, so reporting
    lives at the three callers above. Pinned here so a later "fix" at this seam — which would
    have to guess — reads as the wrong place.
    """
    pack = _pack()
    assert (pack.ruleset_spec("") or {}) == (pack.ruleset_spec(pack.default_ruleset_key()) or {})
    assert pack.default_ruleset_key() == _RIVAL


# --- the stage-health finding -------------------------------------------------------


class _Ctx:
    """A stand-in for JobContext carrying only what the scorer reads."""

    def __init__(self, selection=None, modules=None, config=None):
        self.outputs = {}
        self.modules = modules or {}
        self.config = config or {}
        self.stage_facts = (
            {"correlation": {"procedure_selection": selection}} if selection else {}
        )


def _healthy_correlation():
    """A correlation output every OTHER signal is satisfied by, so the score is the finding."""
    return SimpleNamespace(
        record_count=10,
        aggregations={"resolved_correlation_keys": [{"entity_hint": "shipment"}]},
        findings=[object()],
        evidence=SimpleNamespace(degraded=False),
        verdict=None,
        brief=None,
    )


def _health(selection=None, **kw):
    return score_stage("correlation", _healthy_correlation(), _Ctx(selection, **kw))


def test_the_baseline_correlation_output_scores_a_clean_one():
    """The positive control. Without it every assertion below could be another signal's."""
    health = _health()
    assert health.score == 1.0
    assert health.reason_codes == []
    assert health.gate_recommended is False


def test_an_unselected_procedure_gates_the_stage_on_its_own():
    """Weight 0.5 against the 0.6 threshold — the same posture `required_source_not_queried`
    carries, and for the same reason: the run produced a well-formed answer to a question
    nobody asked."""
    assert _DEFAULT_WEIGHTS["procedure_not_selected"] == 0.5
    health = _health({"basis": "no_match", "candidates": [[_MATCHED, 0.0], [_RIVAL, 0.0]]})
    assert health.score == pytest.approx(0.5)
    assert health.reason_codes == ["procedure_not_selected"]
    assert health.score < stage_threshold({}, "correlation")
    assert health.gate_recommended is True
    reason = " ".join(r.detail for r in health.reasons)
    assert _MATCHED in reason and _RIVAL in reason
    assert "pin" in reason.lower()


def test_a_thin_margin_is_reported_and_cannot_gate_alone():
    """Real information, and not enough of it to hold a run: the selection may well be right.

    A rival that nearly won would also have resolved its conditions against real rows, so the
    wrong choice here is a confident verdict under the other procedure's labels — worth a
    look, not worth a gate.
    """
    assert _DEFAULT_WEIGHTS["procedure_selection_thin"] == 0.15
    health = _health(
        {
            "basis": "scored",
            "margin": 0.05,
            "candidates": [[_MATCHED, 0.42], [_RIVAL, 0.40]],
        }
    )
    assert health.reason_codes == ["procedure_selection_thin"]
    assert health.score == pytest.approx(0.85)
    assert health.score > stage_threshold({}, "correlation")
    assert health.gate_recommended is False


@pytest.mark.parametrize(
    "selection",
    [
        None,
        {},
        {"basis": "scored", "margin": 1.0, "candidates": [[_MATCHED, 0.5]]},
        {"basis": "pinned"},
        {"basis": "sole_spec", "spec_count": 1},
        {"basis": "no_specs"},
    ],
)
def test_a_chosen_procedure_and_a_single_procedure_pack_are_both_silent(selection):
    """The byte-identical checkpoint: a pack that cannot have chosen wrongly scores as before.

    `sole_spec` is the one worth stating — its win really is unscored, and reporting that on
    every run of every single-procedure deployment is how a finding gets switched off.
    """
    health = _health(selection)
    assert health.score == 1.0
    assert health.reason_codes == []


def test_the_run_recorded_fact_beats_the_shared_modules_last_answer():
    """The correlation module is shared across jobs, so its attribute holds the LAST run.

    Same reason `_generator_names` prefers a recorded fact. The fallback is still asserted,
    because a resumed or replayed job has no `stage_facts` entry to read.
    """
    module = SimpleNamespace(last_selection=SelectionBasis("no_match", spec_count=2))

    recorded = score_stage(
        "correlation",
        _healthy_correlation(),
        _Ctx({"basis": "scored", "margin": 1.0}, modules={"correlation": module}),
    )
    assert recorded.reason_codes == []

    fell_back = score_stage(
        "correlation", _healthy_correlation(), _Ctx(modules={"correlation": module})
    )
    assert fell_back.reason_codes == ["procedure_not_selected"]


def test_the_margin_floor_is_configurable_and_zero_disables_the_check():
    """How much vocabulary two procedures share is a property of the domain, not the engine."""
    thin = {"basis": "scored", "margin": 0.2, "candidates": [[_MATCHED, 0.5]]}
    assert _health(thin).reason_codes == []  # 0.20 clears the shipped 0.15 floor

    raised = _health(thin, config={"correlation": {"selection_margin_floor": 0.5}})
    assert raised.reason_codes == ["procedure_selection_thin"]

    off = _health(
        {"basis": "scored", "margin": 0.01},
        config={"correlation": {"selection_margin_floor": 0}},
    )
    assert off.reason_codes == []

    # An unreadable floor falls back to the default rather than disabling the check: a typo
    # must not silently switch a finding off.
    typo = _health(
        {"basis": "scored", "margin": 0.01},
        config={"correlation": {"selection_margin_floor": "soon"}},
    )
    assert typo.reason_codes == ["procedure_selection_thin"]


def test_a_hostile_context_scores_as_though_nothing_was_recorded():
    """Tests hand the scorer MagicMocks, and a scoring read must never fail the stage.

    A `MagicMock` module's `last_selection.to_dict()` returns another Mock, not a dict — the
    exact shape that would otherwise deduct 0.5 from every run in the suite.
    """
    from unittest.mock import MagicMock

    health = score_stage(
        "correlation", _healthy_correlation(), _Ctx(modules={"correlation": MagicMock()})
    )
    assert health.score == 1.0
    assert health.reason_codes == []

    class _Exploding:
        @property
        def stage_facts(self):
            raise RuntimeError("no facts here")

        outputs = {}
        modules = {}
        config = {}

    assert score_stage("correlation", _healthy_correlation(), _Exploding()).score == 1.0


# --- abstention, declared by the pack and off by default ----------------------------


def _annotated(pack, key, summary=_UNRECOGNISED, subjects=2):
    module = CorrelationModule({}, None, knowledge_pack=pack)
    result = _result(pack, key, subjects)
    _, basis = select_correlation_spec_explained(pack, _analysis(summary))
    module._annotate_procedure_selection(result, basis, key)
    return result


def test_abstain_converts_the_verdict_to_the_rulesets_own_reject_label(tmp_path):
    """`reject` was declared vocabulary and unreachable; an abstention is what reaches it.

    No new verdict class and no new schema value — `verdict_class` is a plain `str` and
    `reject` is already one of the five the rollup stamps.
    """
    pack = _abstaining_pack(tmp_path)
    result = _annotated(pack, _MATCHED)

    for subject in result.verdict.subjects:
        assert subject.verdict == "RETURNED — NO PROCEDURE"
        assert subject.verdict_class == "reject"
        assert any(n.startswith("reject_reason=") for n in subject.notes)
        assert _notes(subject)  # the finding survives the conversion; it IS the reason
        # Nothing to contain: a routing decision names no target.
        assert subject.lock_target == {}
    assert result.verdict.summary == "RETURNED — NO PROCEDURE: 2"
    # The draft was rendered from the verdict this replaces, so it asserted a finding under a
    # procedure never selected — the one sentence an abstention exists to withhold.
    assert "fraudulent" not in result.verdict.notification_draft
    assert "no procedure recognised" in result.verdict.notification_draft.lower()


def test_abstain_without_a_reject_label_refuses_loudly_and_changes_nothing(
    tmp_path, caplog
):
    """The label is the word the reporting vocabulary and closing templates are keyed on.

    Inventing one produces a verdict nothing downstream can render, which is worse than the
    finding it replaces — so the refusal keeps the annotated verdict and says why.
    """
    pack = _abstaining_pack(tmp_path, "no_reject_label", reject=False)
    assert "reject" not in ((pack.ruleset_spec(_MATCHED) or {}).get("labels") or {})

    with caplog.at_level(logging.WARNING):
        result = _annotated(pack, _MATCHED)

    for subject in result.verdict.subjects:
        assert subject.verdict_class == "fraud"  # untouched
        assert not any(n.startswith("reject_reason=") for n in subject.notes)
        assert _notes(subject)  # but still says the procedure was not selected
    assert "no 'reject' label" in caplog.text


def test_the_default_policy_leaves_every_field_alone_but_the_note():
    """What every pack did before this key existed, and what `mock_domain` still does."""
    pack = _pack()
    assert pack.adjudication_policy() == "default"
    baseline = _result(pack, _RIVAL)
    result = _annotated(pack, _RIVAL)

    for before, after in zip(baseline.verdict.subjects, result.verdict.subjects):
        assert after.verdict == before.verdict
        assert after.verdict_class == before.verdict_class
        assert after.lock_target == before.lock_target
        assert [n for n in after.notes if not n.startswith("procedure_unselected=")] == list(
            before.notes
        )
    assert result.verdict.summary == baseline.verdict.summary
    assert result.verdict.notification_draft == baseline.verdict.notification_draft


@pytest.mark.parametrize("policy", ["abstin", "ABSTAIN?", "off"])
def test_an_unspellable_policy_reads_as_default(tmp_path, policy):
    """A pack must not stop adjudicating for a typo; `pack_validate` reports it as an error."""
    pack = _abstaining_pack(tmp_path, f"typo_{abs(hash(policy))}", policy=policy)
    result = _annotated(pack, _MATCHED)
    assert all(s.verdict_class == "fraud" for s in result.verdict.subjects)
    assert all(_notes(s) for s in result.verdict.subjects)


def test_abstain_never_fires_for_a_selected_procedure(tmp_path):
    """The policy is about an unselected incident and nothing else."""
    pack = _abstaining_pack(tmp_path, "selected")
    result = _annotated(pack, _MATCHED, summary=_RECOGNISED)
    assert all(s.verdict_class == "fraud" for s in result.verdict.subjects)
    assert all(not _notes(s) for s in result.verdict.subjects)


def test_a_verdict_with_no_subjects_is_not_abstained(tmp_path):
    """Nothing to convert, and `summary` would be rewritten to the empty string.

    The note is still written — the substitution happened whether or not a subject came out
    of it — but there is no verdict to replace.
    """
    pack = _abstaining_pack(tmp_path, "no_subjects")
    result = _annotated(pack, _MATCHED, subjects=0)
    assert result.verdict.summary  # the original rollup, not an empty rewrite
    assert _notes(result.brief)
