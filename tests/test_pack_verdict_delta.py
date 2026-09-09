"""The verdict delta: what an authoring edit does to the FINDINGS of the runs on record.

Why these tests are shaped the way they are. The module answers a question with **two sides** —
the same stored rows adjudicated by the base pack and by the candidate — so almost every test
here asserts a *difference*, and the ones that do not are asserting a **silence**: `compared is
False` with an empty `changes` is "this deployment cannot say", and rendering that as "nothing
moved" is the failure the module exists to avoid. Each of those has its own test, because a
guard that only fires when a sibling guard would is a guard nobody has tested.

Four properties get a test of their own because each fails quietly:

* **The short-circuit is exact.** An adjudication is a function of the ruleset specs, the entity
  bindings, the pack data and the rows, so an edit touching none of those cannot move a
  verdict — and the check must then read no corpus at all, asserted by making the corpus read
  RAISE. A store that is merely unused and one that is unreachable look identical otherwise.
* **A determination that CHANGED is not a determination that went quiet.** `pass` → `fail` is a
  different finding about a person; `pass` → `unknown` is a check that stopped answering. Both
  are reported, and only the first is a `decided_flip`.
* **The comparison proves it can agree with itself first.** Two packs are compared, so a
  difference is only attributable to the edit if evaluating one pack twice is stable — a
  nondeterministic or self-contaminating evaluator would otherwise report the author's edit as
  the cause of a change it did not make.
* **The two blind spots of a replay CANCEL, and one thing does not.** A flattened sidecar and
  absent row caps make both sides equally weak, so a difference is still real — but a condition
  reading `unknown` for the replay's own reasons reads `unknown` under both packs, so an edit
  meant to FIX it shows nothing. The limits have to say so.

Nothing here reads the real job history, no test names an installed pack except the two that
mean to, and the corpus arrives through injected `runs=` or a fake with a `load_all`.
"""

import shutil
from types import SimpleNamespace

import pytest
import yaml

from src.knowledge import pack_dry_run, pack_verdict_delta as pvd
from src.knowledge.pack import load_knowledge_pack
from tests.installed_packs import FIXTURE_PACK, PACKS_ROOT

SUBJECT = "W-1"
SOURCE = "widget_ledger"

# --------------------------------------------------------------------------- the fixtures


def _spec(*, maxv=1, fail_detail="", conditions=None, subject="widget"):
    """One counting condition over a leaf the rows carry, so both sides really decide.

    `max` is the knob nearly every test here turns, because it is the smallest possible edit
    that moves a finding without touching a field path, a source or a polarity — exactly the
    shape the other three plan checks cannot see.
    """
    cond = {
        "id": "one_handler",
        "label": "One handler touched the widget",
        "kind": "distinct_count",
        "source": "ledger",
        "field": "handler",
        "max": maxv,
        "decisive": True,
    }
    if fail_detail:
        cond["fail_detail"] = fail_detail
    return {
        "label_scheme": "widgets",
        "subject_entity": subject,
        "labels": {
            "fraud": "CONFIRMED",
            "false_positive": "CLEARED",
            "insufficient": "INSUFFICIENT DATA",
        },
        "sources": {"ledger": SOURCE},
        "conditions": [cond] if conditions is None else conditions,
    }


def _pack(tmp_path, name, verdicts, *, data=None):
    """A pack on disk, loaded through the pipeline's own loader.

    Written and loaded rather than hand-built, because `replay_surface` reads the pack through
    the accessors (`ruleset_spec`, `field_priors_for`, `pack_data`) and a stand-in object would
    let a resolution rule the loader applies go untested.
    """
    root = tmp_path / name
    root.mkdir()
    (root / "rulesets.yaml").write_text(
        yaml.safe_dump({"verdicts": verdicts}, sort_keys=False), encoding="utf-8"
    )
    if data:
        (root / "data").mkdir()
        for stem, payload in data.items():
            (root / "data" / f"{stem}.yaml").write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
    return load_knowledge_pack(root)


def _analysis(subject=SUBJECT, etype="widget"):
    """Through the module's own reader, so the shape is the one production passes."""
    return pack_dry_run._analysis_of(
        {
            "extracted_entities": [{"entity_type": etype, "value": subject}],
            "incident_summary": "a widget was handled twice",
        }
    )


def _run(job_id="job-aaaaaaaa", key="alpha", handlers=("H1", "H2"), subject=SUBJECT):
    return pack_dry_run.StoredRun(
        job_id=job_id,
        ruleset_key=key,
        logs={SOURCE: [{"widget": subject, "handler": h} for h in handlers]},
        analysis=_analysis(subject),
    )


class FakeStore:
    """A `JobStore` as this module uses it: one `load_all`, and a record that it was asked."""

    def __init__(self, docs=(), raises=None):
        self.docs = list(docs)
        self.raises = raises
        self.calls = 0

    def load_all(self):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return list(self.docs)


def _patch_evaluator(monkeypatch, evaluate):
    """Swap the verdict engine for `evaluate`, keeping the real `flatten_leaves`.

    `compare` reaches the engine through `pack_dry_run._correlation()`, which is also what
    resolves the entity map — so a fake that answers only `evaluate_verdict` would break the
    map instead of the evaluation and every test would fail for the wrong reason.
    """
    real = pack_dry_run._correlation()
    monkeypatch.setattr(
        pack_dry_run,
        "_correlation",
        lambda: SimpleNamespace(
            evaluate_verdict=evaluate, flatten_leaves=real.flatten_leaves
        ),
    )


def _line(delta, line, *, kind=None):
    matches = [
        c for c in delta.changes if c.line == line and (kind is None or c.kind == kind)
    ]
    assert matches, f"no {kind or 'change'} on {line}: {[c.to_dict() for c in delta.changes]}"
    return matches[0]


# ------------------------------------------------------------------------- a moved finding


def test_a_threshold_edit_that_flips_a_determination_names_the_run_and_the_line(tmp_path):
    """The deliverable: the job, the ruleset, the subject, the condition, and both readings.

    Nothing else in `plan_checks` can see this edit. The pack validates on both sides, the
    condition answers on both sides, and the same procedure adjudicates — only the finding
    about the subject is different.
    """
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"alpha": _spec(maxv=9)})

    delta = pvd.verdict_delta(base, candidate, runs=[_run()])

    assert delta.compared is True
    assert delta.replayed == 1 and delta.runs_changed == 1
    assert delta.rulesets == ("alpha",)
    change = _line(delta, "one_handler", kind="condition")
    assert change.job_id == "job-aaaaaaaa"
    assert change.ruleset_key == "alpha"
    assert change.subject == "widget W-1"
    assert (change.before, change.after) == ("fail", "pass")
    assert change.decided_flip is True
    assert len(delta.decided_flips) == 1


def test_a_moved_subject_disposition_is_its_own_line_and_outranks_the_condition(tmp_path):
    """What a report reader sees first must sort first.

    A condition table is where an author looks; a headline is where everyone else does. Both are
    reported, and the ordering is the fix rather than a convention — a summary buried under
    forty condition flips is a summary nobody reads.
    """
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"alpha": _spec(maxv=9)})

    delta = pvd.verdict_delta(base, candidate, runs=[_run()])

    kinds = [c.kind for c in delta.changes]
    assert kinds.index("summary") < kinds.index("condition")
    assert kinds.index("verdict") < kinds.index("condition")
    assert len(delta.verdicts_moved) == 2
    subject_line = _line(delta, "_verdict", kind="verdict")
    assert subject_line.before != subject_line.after
    assert "[" in subject_line.before and "[" in subject_line.after, (
        "the verdict CLASS rides beside the label, because two procedures can print one label "
        "for different classes and the class is what the pipeline branches on"
    )


def test_a_check_that_stopped_answering_is_reported_and_is_not_a_flip(tmp_path):
    """Two changes, two meanings, and only one is a different finding about a person.

    `pass` -> `fail` alleges something new; `pass` -> `unknown` says the check went quiet, which
    is usually an author's own mistake and never a conclusion. Counted together, the first
    disappears inside the second on any real edit.
    """
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(
        tmp_path,
        "cand",
        {"alpha": _spec(conditions=[dict(_spec()["conditions"][0], field="depot.code")])},
    )

    delta = pvd.verdict_delta(base, candidate, runs=[_run()])

    change = _line(delta, "one_handler", kind="condition")
    assert (change.before, change.after) == ("fail", "unknown")
    assert change.decided_flip is False
    assert delta.decided_flips == ()
    assert delta.runs_changed == 1, "it still MOVED — it is just not a determination"


def test_a_detail_that_moved_under_an_unchanged_result_is_not_a_finding(tmp_path):
    """Prose is regenerated per run; reporting it would bury the flips.

    A reworded `fail_detail` is the commonest edit in this whole surface, and the result it
    explains is identical — so the condition line is unchanged and the delta says so.
    """
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1, fail_detail="too many hands")})
    candidate = _pack(
        tmp_path, "cand", {"alpha": _spec(maxv=1, fail_detail="more than one handler")}
    )

    delta = pvd.verdict_delta(base, candidate, runs=[_run()])

    assert delta.compared is True and delta.replayed == 1
    assert delta.changes == ()
    assert delta.runs_changed == 0
    assert delta.surface_changed == ("specs",), "the edit WAS seen — it just changed no result"


# ------------------------------------------------------------------------ the short-circuit


def test_an_edit_that_moves_no_replay_surface_reads_no_corpus(tmp_path, monkeypatch):
    """The common case costs nothing, and that is exact rather than a hope.

    An adjudication is a pure function of the specs, the bindings, the pack data and the rows —
    and the rows are the same on both sides — so an edit leaving all three pack parts alone
    cannot move a verdict. Asserted by making the corpus read RAISE, because a store that is
    merely unused and one that is unreachable look identical from a passing test otherwise.
    """
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"alpha": _spec(maxv=1)})
    monkeypatch.setattr(
        pack_dry_run,
        "stored_runs",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("read the corpus")),
    )

    delta = pvd.verdict_delta(base, candidate)

    assert delta.compared is False
    assert delta.surface_changed == ()
    assert delta.changes == ()
    assert "adjudicates exactly as it did" in delta.reason


@pytest.mark.parametrize(
    "part,mutate",
    [
        ("specs", lambda kw: kw.update(verdicts={"alpha": _spec(maxv=9)})),
        ("data", lambda kw: kw.update(data={"thresholds": {"handlers": 9}})),
    ],
)
def test_the_surface_names_which_part_the_edit_moved(tmp_path, part, mutate):
    """Reported per part because the parts license different readings.

    A `bindings` change moves what every ruleset can see; a `specs` change is usually confined
    to the use case that was edited. One bit would make those the same finding.
    """
    kw = {"verdicts": {"alpha": _spec(maxv=1)}, "data": None}
    base = _pack(tmp_path, "base", kw["verdicts"], data=kw["data"])
    mutate(kw)
    candidate = _pack(tmp_path, "cand", kw["verdicts"], data=kw["data"])

    assert pvd.surface_changed(base, candidate) == (part,)


# ------------------------------------------------------------------- the determinism control


def test_a_pack_that_does_not_agree_with_itself_withholds_the_delta(tmp_path, monkeypatch):
    """A difference between two packs is only the edit's if one pack is stable.

    Without this control an evaluator that mutated the rows it was handed, or a condition
    reading a clock, reports the author's edit as the cause of a change it did not make — on
    every preview, with a well-formed list of moved findings.
    """
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"alpha": _spec(maxv=9)})
    real = pack_dry_run._correlation().evaluate_verdict
    calls = {"n": 0}

    def drifting(spec, logs, analysis, **kwargs):
        calls["n"] += 1
        verdict = real(spec, logs, analysis, **kwargs)
        verdict.summary = f"{verdict.summary} (call {calls['n']})"
        return verdict

    _patch_evaluator(monkeypatch, drifting)

    delta = pvd.verdict_delta(base, candidate, runs=[_run()])

    assert delta.compared is False
    assert delta.changes == ()
    assert "twice" in delta.reason and "attributed to the edit" in delta.reason


def test_the_control_costs_one_extra_replay_and_runs_once(tmp_path, monkeypatch):
    """A control charged per run would double the budget of the whole check.

    So it is the FIRST replayed run only: three runs cost seven evaluations, not eight, and
    a fourth run would still cost one control.
    """
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"alpha": _spec(maxv=9)})
    real = pack_dry_run._correlation().evaluate_verdict
    seen = []

    def counted(spec, logs, analysis, **kwargs):
        seen.append(spec.get("conditions")[0].get("max"))
        return real(spec, logs, analysis, **kwargs)

    _patch_evaluator(monkeypatch, counted)

    runs = [_run(job_id=f"job-{i}{i}{i}{i}{i}{i}{i}{i}") for i in range(3)]
    delta = pvd.verdict_delta(base, candidate, runs=runs)

    assert delta.replayed == 3
    assert len(seen) == 7, seen
    assert seen[:3] == [1, 1, 9], "base, base again as the control, then the candidate"
    assert seen.count(1) == 4 and seen.count(9) == 3


# --------------------------------------------------------------------------- the silences


def test_no_corpus_is_a_silence_and_not_a_clean_bill(tmp_path):
    """A deployment with no stored run cannot say anything about past findings.

    Reporting `0 changed` there is the exact shape this module exists to avoid: it reads as a
    guarantee, and it is an absence.
    """
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"alpha": _spec(maxv=9)})

    delta = pvd.verdict_delta(base, candidate, runs=[])

    assert delta.compared is False
    assert delta.corpus == 0 and delta.replayed == 0
    assert delta.changes == ()
    assert "cannot say" in delta.reason
    assert delta.surface_changed == ("specs",), "what the edit moved is still reported"


def test_no_change_over_a_real_corpus_says_what_it_does_not_claim(tmp_path):
    """`compared is True` with no change is a real result — and still not a verdict on the edit.

    The limits carry the one caveat that does NOT cancel between two replays of the same rows.
    """
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    # A spec differing only in a top-level key no evaluator reads, so the surface moves and
    # the findings cannot.
    candidate = _pack(
        tmp_path, "cand", {"alpha": dict(_spec(maxv=1), notes=["rephrased"])}
    )

    delta = pvd.verdict_delta(base, candidate, runs=[_run()])

    assert delta.compared is True and delta.changes == ()
    assert any("BOTH packs" in limit for limit in delta.limits)
    assert any("did nothing" in limit for limit in delta.limits)


def test_a_ruleset_only_the_candidate_declares_is_named_and_not_diffed(tmp_path):
    """A new ruleset has no before-state, so every line of it would read as new.

    That noise would bury the changes to the procedures that DO have one — and "this procedure
    is new" is the actionable form of the same fact.
    """
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(
        tmp_path, "cand", {"alpha": _spec(maxv=1), "beta": _spec(maxv=9)}
    )

    delta = pvd.verdict_delta(base, candidate, runs=[_run()])

    assert any("beta" in p and "before-state" in p for p in delta.problems)
    assert all(c.ruleset_key != "beta" for c in delta.changes)


def test_two_packs_sharing_no_ruleset_key_compare_nothing(tmp_path):
    """The degenerate case of the rule above: nothing has a before-state at all."""
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"beta": _spec(maxv=9)})

    delta = pvd.verdict_delta(base, candidate, runs=[_run()])

    assert delta.compared is False
    assert "share no ruleset key" in delta.reason


def test_a_ruleset_no_stored_run_adjudicated_is_flagged_as_unpaired(tmp_path):
    """Source overlap is the weaker pairing and must never read as the paired one.

    A changed line there is still real — both sides read the same rows — but the run was never
    decided by this procedure, so it cannot be described as a past finding of it.
    """
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"alpha": _spec(maxv=9)})

    delta = pvd.verdict_delta(base, candidate, runs=[_run(key="somebody_else")])

    assert delta.compared is True and delta.replayed == 1
    assert any("shared sources only" in p for p in delta.problems)


# ---------------------------------------------------------------------------- the bounds


def test_one_run_that_cannot_be_re_adjudicated_costs_only_itself(tmp_path, monkeypatch):
    """A comparison that aborts on one bad job document tells an author less than 11 of 12."""
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"alpha": _spec(maxv=9)})
    real = pack_dry_run._correlation().evaluate_verdict

    def sometimes(spec, logs, analysis, **kwargs):
        if any(r.get("widget") == "W-2" for r in logs.get(SOURCE) or []):
            raise RuntimeError("boom")
        return real(spec, logs, analysis, **kwargs)

    _patch_evaluator(monkeypatch, sometimes)

    delta = pvd.verdict_delta(
        base,
        candidate,
        runs=[_run(job_id="job-good1234"), _run(job_id="job-bad12345", subject="W-2")],
    )

    assert delta.replayed == 1
    assert delta.compared is True
    assert any("job-bad1" in p and "RuntimeError" in p for p in delta.problems)
    assert delta.changes and all(c.job_id == "job-good1234" for c in delta.changes)


def test_the_run_budget_bounds_the_replay_and_the_limits_say_which_bound(tmp_path):
    """Replaying more can only ADD changes, so a cut list has to name its own cause.

    An operator reading "2 of 9" acts differently from one reading "2 of 2", and the two are
    indistinguishable without the count.
    """
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"alpha": _spec(maxv=9)})
    runs = [_run(job_id=f"job-{i:08d}") for i in range(5)]

    delta = pvd.verdict_delta(base, candidate, runs=runs, max_runs=2)

    assert delta.corpus == 5 and delta.replayed == 2
    cut = [limit for limit in delta.limits if "not replayed" in limit]
    assert cut and "3 of 5" in cut[0]
    assert "run budget" in cut[0] and "can only ADD" in cut[0]


def test_a_cut_change_list_says_how_many_it_did_not_list(tmp_path, monkeypatch):
    """A reader who has learned these lists are complete reads a truncated one as the whole
    consequence."""
    monkeypatch.setattr(pvd, "MAX_CHANGES", 2)
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"alpha": _spec(maxv=9)})

    delta = pvd.verdict_delta(base, candidate, runs=[_run()])

    assert len(delta.changes) == 2
    assert delta.changes_cut == 1
    assert "1 more changed line(s) not listed" in pvd.render(delta)


def test_a_subject_that_appears_on_only_one_side_is_its_own_kind(tmp_path):
    """A subject the candidate stops adjudicating is not forty missing condition lines.

    Reported as one `subject_gone` carrying its old disposition, because that disposition is the
    thing that is now unsaid.
    """
    two_subjects = {SOURCE: [{"widget": "W-1", "handler": "H1"}, {"widget": "W-2", "handler": "H2"}]}
    run = pack_dry_run.StoredRun(
        job_id="job-twosubs",
        ruleset_key="alpha",
        logs=two_subjects,
        analysis=pack_dry_run._analysis_of(
            {
                "extracted_entities": [
                    {"entity_type": "widget", "value": "W-1"},
                    {"entity_type": "widget", "value": "W-2"},
                ],
                "incident_summary": "two widgets",
            }
        ),
    )
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"alpha": _spec(maxv=1, subject="gadget")})

    delta = pvd.verdict_delta(base, candidate, runs=[run])

    gone = [c for c in delta.changes if c.kind == "subject_gone"]
    assert len(gone) == 2, [c.to_dict() for c in delta.changes]
    assert {c.subject for c in gone} == {"widget W-1", "widget W-2"}
    assert all(c.before and not c.after for c in gone)


def test_a_repeated_subject_key_is_kept_rather_than_overwritten():
    """Two subjects of one type and value is not a shape the engine should produce.

    Silently dropping the second would report its every condition as missing from both sides —
    a fabricated defect out of a fixture nobody expected.
    """
    def subject(result):
        return SimpleNamespace(
            subject_type="widget",
            subject_value=SUBJECT,
            verdict="CLEARED",
            verdict_class="false_positive",
            checks=[SimpleNamespace(id="one_handler", result=result, detail="")],
        )

    out = pvd.lines(
        SimpleNamespace(
            subjects=[subject("pass"), subject("fail")], summary="", degraded=False
        )
    )

    assert len(out["subjects"]) == 2
    results = {v["one_handler"][0] for v in out["subjects"].values()}
    assert results == {"pass", "fail"}


def test_a_detail_is_bounded_and_the_bound_is_the_modules_own():
    """The detail is context for a result that moved, not the finding — so it is cut, and cut
    at one place."""
    long_detail = "x" * (pvd.DETAIL_CHARS * 3)
    out = pvd.lines(
        SimpleNamespace(
            subjects=[
                SimpleNamespace(
                    subject_type="widget",
                    subject_value=SUBJECT,
                    verdict="CLEARED",
                    verdict_class="false_positive",
                    checks=[
                        SimpleNamespace(id="one_handler", result="fail", detail=long_detail)
                    ],
                )
            ],
            summary="",
            degraded=False,
        )
    )

    assert len(out["subjects"]["widget W-1"]["one_handler"][1]) == pvd.DETAIL_CHARS


def test_a_degraded_reading_that_cleared_does_not_render_as_absent():
    """`degraded` is a bool, and `False or ""` reads as a line that is not there at all —
    the one wording this module reserves for exactly that."""
    def verdict(degraded):
        return SimpleNamespace(subjects=[], summary="s", degraded=degraded)

    changes = pvd.diff(pvd.lines(verdict(True)), pvd.lines(verdict(False)))

    assert len(changes) == 1
    assert changes[0].kind == "degraded"
    assert changes[0].before == "degraded" and changes[0].after == "complete"


# --------------------------------------------------------------------------- the rendering


def test_the_serialised_delta_carries_the_servers_own_sentences(tmp_path):
    """The model proposing the edit and the operator approving it must read one description of
    one measurement, so the render rides on the payload."""
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"alpha": _spec(maxv=9)})

    payload = pvd.verdict_delta(base, candidate, runs=[_run()]).to_dict()

    assert payload["compared"] is True
    assert payload["decided_flips"] == 1 and payload["verdicts_moved"] == 2
    assert payload["surface_changed"] == ["specs"]
    assert "determination CHANGED" in payload["text"]
    assert isinstance(payload["changes"], list) and payload["changes"][0]["kind"]


def test_an_uncompared_render_leads_with_the_reason_and_not_a_count(tmp_path):
    """Rendered as a count, a silence reads as a clean result."""
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"alpha": _spec(maxv=9)})

    text = pvd.render(pvd.verdict_delta(base, candidate, runs=[]))

    assert text.startswith("verdict lines: not compared")
    assert "read differently" not in text


def test_a_compared_run_with_no_change_still_states_what_it_did_not_prove(tmp_path):
    """"No stored run's findings move" is the sentence that must not be read as "the edit is
    correct"."""
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"alpha": _spec(maxv=1)})

    text = pvd.render(pvd.compare(base, candidate, [_run()]))

    assert "no stored run's findings move" in text
    assert "not a claim that the edit is correct" in text


# --------------------------------------------------------------------- the directory entry


def test_two_pack_directories_are_compared_through_the_pipeline_loader(tmp_path):
    """The plan-preview entry point, over the real fixture pack and a copy with one bound moved.

    Both sides go through `load_knowledge_pack`, so what is asserted here is that a real pack's
    edit is *seen* — the loader (with its shared-check imports and its data directory) is the
    only thing between a YAML file and a replay surface.
    """
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    shutil.copytree(PACKS_ROOT / FIXTURE_PACK, base)
    shutil.copytree(PACKS_ROOT / FIXTURE_PACK, candidate)

    edited = 0
    for path in sorted(candidate.rglob("*.yaml")):
        text = path.read_text()
        if "max: 1\n" in text:
            path.write_text(text.replace("max: 1\n", "max: 7\n", 1))
            edited += 1
            break
    assert edited == 1, "the fixture pack's counting bounds moved; re-point this test"

    delta = pvd.verdict_delta_for_dirs(base, candidate, store=FakeStore())

    assert delta.problems == ()
    assert delta.surface_changed == ("specs",)
    # No corpus in this fixture, so the comparison itself is a silence — and that is the
    # assertion: the surface change is reported even where the changes cannot be.
    assert delta.compared is False
    assert delta.corpus == 0


def test_a_pack_directory_that_will_not_load_is_reported_not_raised(tmp_path, monkeypatch):
    """An unloadable candidate is another check's finding; refusing to answer here would hide
    the moved findings in a plan that has a second, unrelated defect.

    The loader is patched rather than pointed at a missing directory, because an absent pack
    dir loads *cleanly* into an empty pack by design — the failure this branch exists for is a
    tree that exists and cannot be read.
    """
    from src.knowledge import pack as pack_module

    monkeypatch.setattr(
        pack_module,
        "load_knowledge_pack",
        lambda path: (_ for _ in ()).throw(ValueError("could not compose the anchors")),
    )

    delta = pvd.verdict_delta_for_dirs(
        tmp_path / "base", tmp_path / "candidate", store=FakeStore()
    )

    assert delta.compared is False
    assert "could not be loaded" in delta.reason or delta.reason
    assert any("could not be loaded" in p for p in delta.problems)
    assert any("anchors" in p for p in delta.problems), "the loader's own reason survives"


def test_an_unreadable_surface_is_compared_rather_than_short_circuited(tmp_path):
    """The one place a raise is the right answer, and the reason it is not caught deeper.

    `replay_surface` reads a pack, and its only alternative to failing on a pack it cannot
    read is to return an EMPTY surface — which compares equal to the other side's empty
    surface, short-circuits as "no ruleset spec changed", and reports the reassuring silence
    this whole module exists to prevent. So it raises, and `verdict_delta` degrades: the
    surface reads `unknown` and the corpus is compared anyway.
    """
    real = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    with pytest.raises(Exception):
        pvd.surface_changed(None, real)

    delta = pvd.verdict_delta(None, real, runs=[_run()])

    assert delta.surface_changed == ("unknown",), (
        "an unreadable surface must not read as an unchanged one"
    )
    assert delta.compared is False and delta.problems, "and the failure is still named"


def test_nothing_in_the_module_raises(tmp_path):
    """Every failure at an entry point becomes a reported field.

    Scoped to the functions that HAVE a field to report into — a preview that 500s on one
    unreadable job document is a preview an author stops opening, while a pure surface reader
    has nowhere to put a problem and one caller that needs to hear about it.
    """
    base = _pack(tmp_path, "base", {"alpha": _spec(maxv=1)})
    candidate = _pack(tmp_path, "cand", {"alpha": _spec(maxv=9)})
    for call in (
        lambda: pvd.compare(base, candidate, [_run()]),
        lambda: pvd.compare(base, candidate, []),
        lambda: pvd.compare(None, None, [_run()]),
        lambda: pvd.verdict_delta(base, candidate, store=FakeStore(raises=OSError)),
        lambda: pvd.verdict_delta_for_dirs(tmp_path / "gone", tmp_path / "also-gone"),
        lambda: pvd.lines(None),
        lambda: pvd.diff(pvd.lines(None), pvd.lines(None)),
        lambda: pvd.render(pvd.VerdictDelta()),
        lambda: pvd.main([]),
    ):
        call()
