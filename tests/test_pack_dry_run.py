"""Replaying a candidate ruleset over stored evidence: the two findings, and their bounds.

WHAT THIS FILE IS GUARDING. The module exists to catch a condition that is valid YAML, names
a real leaf, and reads `unknown` on every row that has ever come back — so the assertions
that matter are the ones about a finding it must NOT make. A dry run is a *warning surface*:
it reports two things an author is expected to act on, and either of them fired on a shape
that is correct by design ships as noise and gets switched off. Three such shapes are pinned
here, each of which the module would otherwise report:

* **A composite's child.** `evaluate_verdict` emits ONE check line for the parent and none
  for its children, so every child of every composite reads "no check line at all" — the
  module's headline finding, fabricated by the walk that puts children in the table.
* **A declared `kind: stub`.** A pack ships one to say a question was considered and cannot
  be asked yet; it is the DECLARED not-evaluated kind, so it cannot be the reported one.
* **A ruleset no stored run adjudicated.** Source overlap admits nearly every run to nearly
  every ruleset in one domain, and each replay then reads one procedure's conditions against
  another's rows and answers `unknown` on all of them. That reading is kept, because a newly
  authored ruleset has no other, but it is FLAGGED — never presented as the paired one.

Everything is built on injected `runs=`, a fake store, and a synthetic pack written into
`tmp_path`: no test here reads the real job history or names an installed pack.
"""

from types import SimpleNamespace

import pytest
import yaml

from src.knowledge import pack_dry_run
from src.knowledge.pack_dry_run import (
    StoredRun,
    _condition_index,
    _eligible,
    _limits,
    _select,
    as_dict,
    dry_run,
    render,
    stored_runs,
)

SUBJECT = "W-1"

# The mechanics of one counting condition, twice: once on a leaf the rows carry, once on a
# leaf no row has. The second is the module's reason to exist — it is well-formed, it names a
# plausible path, and it answers nothing.
ONE_HANDLER = {
    "id": "one_handler",
    "label": "One handler touched the widget",
    "kind": "distinct_count",
    "source": "ledger",
    "field": "handler",
    "max": 1,
    "decisive": True,
}
ABSENT_LEAF = {
    "id": "absent_leaf",
    "label": "One depot appears on the widget",
    "kind": "distinct_count",
    "source": "ledger",
    "field": "depot.code",
    "max": 1,
}
STUB = {
    "id": "parked",
    "label": "No bound has been established for this yet",
    "kind": "stub",
}
COMPOSITE = {
    "id": "either_reading",
    "label": "Either reading of the handler count holds",
    "kind": "any_of",
    "fail_detail": "neither reading of the handler count held",
    "children": [
        {
            "id": "child_strict",
            "kind": "distinct_count",
            "source": "ledger",
            "field": "handler",
            "max": 1,
        },
        {
            "id": "child_loose",
            "kind": "distinct_count",
            "source": "ledger",
            "field": "handler",
            "max": 9,
        },
    ],
}


def _spec(conditions, *, subject="widget", sources=None):
    return {
        "label_scheme": "alpha",
        "subject_entity": subject,
        "labels": {
            "fraud": "CONFIRMED",
            "false_positive": "CLEARED",
            "insufficient": "INSUFFICIENT DATA",
        },
        "sources": {"ledger": "widget_ledger"} if sources is None else sources,
        "conditions": conditions,
    }


def _write_pack(tmp_path, verdicts, name="candidate"):
    root = tmp_path / name
    root.mkdir()
    (root / "rulesets.yaml").write_text(
        yaml.safe_dump({"verdicts": verdicts}, sort_keys=False), encoding="utf-8"
    )
    return root


def _analysis(subject=SUBJECT, etype="widget"):
    """Built through the module's own reader, so the shape is the one production passes."""
    return pack_dry_run._analysis_of(
        {
            "extracted_entities": [{"entity_type": etype, "value": subject}],
            "incident_summary": "a widget was handled twice",
        }
    )


def _rows(*handlers, subject=SUBJECT):
    return [{"widget": subject, "handler": h} for h in (handlers or ("H1",))]


def _run(job_id="job-1", key="alpha", logs=None, subject=SUBJECT):
    return StoredRun(
        job_id=job_id,
        ruleset_key=key,
        logs={"widget_ledger": _rows(subject=subject)} if logs is None else logs,
        analysis=_analysis(subject),
    )


def _tally(report, key):
    return next(rs for rs in report.rulesets if rs.key == key)


def _cond(report, key, cid):
    return next(c for c in _tally(report, key).conditions if c.condition_id == cid)


# ----------------------------------------------------------------- reading the stored runs


class _Store:
    """Stands in for `JobStore`, which is the one seam `stored_runs` reads through."""

    def __init__(self, docs, error=None):
        self._docs = docs
        self._error = error

    def load_all(self):
        if self._error:
            raise self._error
        return list(self._docs)


def _doc(job_id, *, logs=None, analysis=None, use_case="alpha", summary=""):
    return {
        "job_id": job_id,
        "outputs": {
            "logs": {"widget_ledger": _rows()} if logs is None else logs,
            "understanding": {
                "analysis": (
                    {"extracted_entities": [{"entity_type": "widget", "value": SUBJECT}]}
                    if analysis is None
                    else analysis
                )
            },
            "correlation": {
                "brief": {"use_case": use_case},
                "verdict": {"summary": summary},
            },
        },
    }


def test_a_run_with_no_retrieved_rows_is_skipped_without_a_problem():
    """Most stored jobs are not evidence. That is not a finding, so it must stay silent."""
    runs, problems = stored_runs(_Store([_doc("a", logs={}), _doc("b")]))
    assert [r.job_id for r in runs] == ["b"]
    assert problems == []


def test_a_run_whose_understanding_is_not_a_mapping_is_reported_and_skipped():
    """No subject can be resolved from it, so the replay would be silently subject-less."""
    runs, problems = stored_runs(_Store([_doc("badshape", analysis="not a mapping")]))
    assert runs == []
    assert len(problems) == 1
    assert "badshape" in problems[0] and "skipped" in problems[0]


def test_runs_come_back_newest_first():
    """`load_all` is oldest-first, and the newest evidence is the relevant evidence."""
    runs, _ = stored_runs(_Store([_doc("old"), _doc("mid"), _doc("new")]))
    assert [r.job_id for r in runs] == ["new", "mid", "old"]


def test_an_unreadable_history_is_one_problem_and_never_a_raise():
    """A dry run that aborts tells the author less than one that reports 11 of 12."""
    runs, problems = stored_runs(_Store([], error=OSError("disk is gone")))
    assert runs == []
    assert len(problems) == 1 and "disk is gone" in problems[0]


def test_the_pairing_comes_off_the_brief_and_non_row_values_are_dropped():
    """`use_case` is what pairs a run with the ruleset that really adjudicated it."""
    runs, problems = stored_runs(
        _Store(
            [
                _doc(
                    "j",
                    logs={"widget_ledger": [{"handler": "H1"}, "not a row"], "x": "text"},
                    use_case="beta",
                    summary="CLEARED",
                )
            ]
        )
    )
    assert problems == []
    assert runs[0].ruleset_key == "beta"
    assert runs[0].recorded_summary == "CLEARED"
    assert runs[0].logs == {"widget_ledger": [{"handler": "H1"}]}


# ------------------------------------------------------------------------- the two findings


def test_a_condition_answered_on_no_run_is_reported_as_always_unknown(tmp_path):
    """The defect the module exists for: asked on every subject, answering nothing."""
    pack = _write_pack(tmp_path, {"alpha": _spec([ONE_HANDLER, ABSENT_LEAF])})
    report = dry_run(pack, runs=[_run()])

    assert report.exercised
    assert _cond(report, "alpha", "one_handler").passes == 1
    absent = _cond(report, "alpha", "absent_leaf")
    assert (absent.passes, absent.fails, absent.unknowns) == (0, 0, 1)
    assert absent.always_unknown and not absent.never_evaluated
    assert [c.condition_id for c in _tally(report, "alpha").mute_conditions] == [
        "absent_leaf"
    ]
    assert "ALWAYS UNKNOWN  absent_leaf" in render(report)


def test_a_declared_stub_is_not_reported_as_always_unknown(tmp_path):
    """A stub IS the declared not-evaluated kind, so reporting it is reporting a decision."""
    pack = _write_pack(tmp_path, {"alpha": _spec([ONE_HANDLER, STUB])})
    report = dry_run(pack, runs=[_run()])

    parked = _cond(report, "alpha", "parked")
    assert parked.unknowns == 1
    assert not parked.always_unknown and not parked.never_evaluated
    assert _tally(report, "alpha").mute_conditions == []


def test_a_composites_child_is_never_reported_as_never_evaluated(tmp_path):
    """The parent produces the one line; a child's zeroes are the design, not a defect.

    Without this, every composite child of every pack lands in the module's headline finding
    — a fabricated defect on a correct declaration, which is how a warning surface earns
    being switched off. The child is still listed, because a reader looking for its id has to
    find it, and it is marked so the zeroes are not read as silence.
    """
    pack = _write_pack(tmp_path, {"alpha": _spec([COMPOSITE])})
    report = dry_run(pack, runs=[_run()])

    parent = _cond(report, "alpha", "either_reading")
    assert parent.passes == 1 and not parent.nested
    for cid in ("child_strict", "child_loose"):
        child = _cond(report, "alpha", cid)
        assert child.nested
        assert child.checks == 0
        assert not child.never_evaluated
        # A child is silent on every run by construction, so the count would read as the
        # whole replay and say nothing at all.
        assert child.runs_silent == 0
    assert _tally(report, "alpha").silent_conditions == []

    text = render(report)
    assert "NEVER EVALUATED" not in text
    assert "folded  child_strict" in text
    assert "folded` marks a composite's child" in text


def test_a_top_level_condition_with_no_check_line_is_still_reported(tmp_path):
    """The other direction of the same fix: the real finding must survive it.

    Reached through the no-verdict path — a run paired with the ruleset by its brief whose
    logs hold none of the ruleset's sources — which is also what `runs_no_verdict` counts.
    """
    pack = _write_pack(tmp_path, {"alpha": _spec([ONE_HANDLER, COMPOSITE])})
    report = dry_run(pack, runs=[_run(logs={"another_source": [{"handler": "H1"}]})])

    rs = _tally(report, "alpha")
    assert (rs.runs_replayed, rs.runs_no_verdict) == (1, 1)
    assert not report.exercised
    assert [c.condition_id for c in rs.silent_conditions] == [
        "one_handler",
        "either_reading",
    ]
    assert "NEVER EVALUATED  one_handler" in render(report)


def test_a_child_promoted_to_a_line_is_counted_rather_than_dropped(tmp_path):
    """A check the condition list does not name still has to reconcile with the totals."""
    pack = _write_pack(tmp_path, {"alpha": _spec([ONE_HANDLER])})
    real = pack_dry_run._correlation()

    def evaluate(*args, **kwargs):
        verdict = real.evaluate_verdict(*args, **kwargs)
        verdict.subjects[0].checks.append(
            SimpleNamespace(id="rewritten_id", label="Rewritten", result="fail")
        )
        return verdict

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            pack_dry_run,
            "_correlation",
            lambda: SimpleNamespace(
                flatten_leaves=real.flatten_leaves, evaluate_verdict=evaluate
            ),
        )
        report = dry_run(pack, runs=[_run()])

    extra = _cond(report, "alpha", "rewritten_id")
    assert extra.fails == 1 and extra.label == "Rewritten"


def test_a_condition_silent_on_one_run_of_two_counts_the_run_not_the_line(tmp_path):
    """`runs silent` counts RUNS, and the units footer is what stops it reading as lines."""
    pack = _write_pack(tmp_path, {"alpha": _spec([ONE_HANDLER])})
    report = dry_run(
        pack,
        runs=[_run("job-1"), _run("job-2", logs={"widget_ledger": _rows("H1", "H2")})],
    )

    tally = _cond(report, "alpha", "one_handler")
    assert (tally.passes, tally.fails, tally.runs_silent) == (1, 1, 0)
    assert "except `runs silent`, which counts RUNS." in render(report)


# ------------------------------------------------------------ which runs, and how many


def test_the_adjudicating_run_wins_and_source_overlap_is_only_the_fallback():
    """Overlap admitted 60 of 65 runs to each of nine rulesets, all answering `unknown`."""
    spec = _spec([ONE_HANDLER])
    adjudicated = _run("paired", key="alpha")
    overlapping = _run("overlap", key="beta")

    picked, unpaired = _eligible("alpha", spec, [adjudicated, overlapping])
    assert [r.job_id for r in picked] == ["paired"] and unpaired is False

    picked, unpaired = _eligible("alpha", spec, [overlapping])
    assert [r.job_id for r in picked] == ["overlap"] and unpaired is True

    # Neither adjudicated by it nor holding one of its sources.
    elsewhere = _run("elsewhere", key="beta", logs={"other": [{"a": 1}]})
    assert _eligible("alpha", spec, [elsewhere]) == ([], True)


def test_an_unpaired_replay_says_so_rather_than_reporting_its_unknowns(tmp_path):
    """A blanket `unknown` off another procedure's rows would send an author to a rewrite."""
    pack = _write_pack(tmp_path, {"alpha": _spec([ONE_HANDLER])})
    report = dry_run(pack, runs=[_run(key="beta")])

    assert _tally(report, "alpha").unpaired
    assert "NO stored run was adjudicated by this ruleset" in render(report)


def test_a_ruleset_no_run_touched_says_its_conditions_are_unmeasured(tmp_path):
    """"Unmeasured" and "silent" license opposite next steps, so the wording is the fix."""
    pack = _write_pack(tmp_path, {"alpha": _spec([ONE_HANDLER])})
    report = dry_run(pack, runs=[_run(key="beta", logs={"other": [{"a": 1}]})])

    rs = _tally(report, "alpha")
    assert rs.runs_replayed == 0
    assert any("unmeasured" in p for p in rs.problems)
    # The zeroes are still marked, but the finding paragraph — the part that sends a reader to
    # `subject_entity` — is gated on a run having been replayed at all.
    assert "produced no check line at all" not in render(report)


def test_the_budget_is_shared_round_robin_across_the_rulesets():
    """One procedure dominates any real history; spent on it, every other reads as silent."""
    beta = [_run(f"b{i}", key="beta") for i in range(3)]
    alpha = [_run("a0", key="alpha")]
    per_key = {"beta": beta, "alpha": alpha}

    assert [k for k, _ in _select(per_key, 2)] == ["alpha", "beta"]
    assert [k for k, _ in _select(per_key, 3)] == ["alpha", "beta", "beta"]
    assert len(_select(per_key, 99)) == 4
    assert _select(per_key, 0) == []


def test_the_run_budget_clips_and_says_which_budget_clipped_it(tmp_path):
    pack = _write_pack(tmp_path, {"alpha": _spec([ONE_HANDLER])})
    report = dry_run(pack, runs=[_run("job-1"), _run("job-2")], max_runs=1)

    assert report.runs_replayed == 1
    assert any("1 of 2 stored runs were not replayed" in lim for lim in report.limits)
    assert any("(the run budget)" in lim for lim in report.limits)


def test_the_time_budget_clips_and_says_which_budget_clipped_it(tmp_path):
    """Checked between runs, because one run's cost scales with the rows it retrieved."""
    pack = _write_pack(tmp_path, {"alpha": _spec([ONE_HANDLER])})
    report = dry_run(pack, runs=[_run("job-1"), _run("job-2")], max_seconds=-1.0)

    assert report.runs_replayed == 0
    assert any("the time budget was spent" in lim for lim in report.limits)


# ----------------------------------------------------------------------------- the bounds


def test_every_blind_spot_names_the_direction_it_errs_in():
    """A bound with no direction is worse than no bound: a clean run reads as a guarantee."""
    limits = _limits(None, None, 0, 0, False)
    assert len(limits) == 3
    assert sum("OPTIMISTIC" in lim for lim in limits) == 2
    assert sum("PESSIMISTIC" in lim for lim in limits) == 1
    assert any("row cap" in lim for lim in limits)
    assert any("acting identity" in lim for lim in limits)
    assert any("did not answer" in lim for lim in limits)


def test_supplying_the_two_facts_a_stored_run_lacks_drops_their_bounds():
    """They are the caller's to supply and are never invented, so they can be supplied."""
    limits = _limits({"widget_ledger": 500}, {"widget_ledger": True}, 0, 0, False)
    assert len(limits) == 1
    assert "did not answer" in limits[0]


def test_a_clipped_reading_says_a_replay_can_only_add_outcomes():
    """The direction that matters for the headline finding: `never evaluated` may be wrong."""
    limits = _limits({"s": 1}, {"s": True}, 4, 12, False)
    assert any("may well fire on a run that was not replayed" in lim for lim in limits)


def test_the_bounds_are_reported_even_when_nothing_could_be_replayed(tmp_path):
    """A reading with no runs still has to say what it could not see."""
    pack = _write_pack(tmp_path, {"alpha": _spec([ONE_HANDLER])})
    report = dry_run(pack, runs=[])

    assert report.runs_available == 0
    assert any("not a finding about the pack" in p for p in report.problems)
    assert len(report.limits) == 3


# ---------------------------------------------------------------- never raises, always says


def test_a_pack_that_will_not_load_is_a_problem_and_not_a_raise(tmp_path):
    """An unresolvable `use:` raises at load by design — here it has to become a sentence."""
    pack = _write_pack(
        tmp_path, {"alpha": _spec([{"id": "imported", "use": "nowhere/nothing"}])}
    )
    report = dry_run(pack, runs=[_run()])

    assert report.rulesets == []
    assert any("the pack could not be loaded" in p for p in report.problems)
    assert report.limits


def test_a_run_that_raises_is_one_ruleset_problem_and_the_others_still_count(tmp_path):
    """One bad run must not lose the other eleven."""
    pack = _write_pack(tmp_path, {"alpha": _spec([ONE_HANDLER])})
    real = pack_dry_run._correlation()
    calls = {"n": 0}

    def evaluate(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("a row the engine could not read")
        return real.evaluate_verdict(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            pack_dry_run,
            "_correlation",
            lambda: SimpleNamespace(
                flatten_leaves=real.flatten_leaves, evaluate_verdict=evaluate
            ),
        )
        report = dry_run(pack, runs=[_run("job-1"), _run("job-2")])

    rs = _tally(report, "alpha")
    assert rs.runs_replayed == 1
    assert any("raised ValueError" in p for p in rs.problems)
    assert _cond(report, "alpha", "one_handler").passes == 1


def test_a_ruleset_key_the_pack_does_not_declare_reports_itself(tmp_path):
    pack = _write_pack(tmp_path, {"alpha": _spec([ONE_HANDLER])})
    report = dry_run(pack, ruleset_keys=["alpha", "ghost"], runs=[_run()])

    assert _tally(report, "ghost").problems == [
        "the pack declares no ruleset under this key"
    ]
    assert _cond(report, "alpha", "one_handler").passes == 1


def test_a_ruleset_declaring_no_sources_says_the_engine_returns_no_verdict(tmp_path):
    """`evaluate_verdict` returns `None` with no declared source in `logs`, so say that."""
    pack = _write_pack(tmp_path, {"alpha": _spec([ONE_HANDLER], sources={})})
    report = dry_run(pack, runs=[_run()])

    assert any("no verdict for it at all" in p for p in _tally(report, "alpha").problems)


def test_a_ruleset_with_no_conditions_says_so_rather_than_printing_an_empty_table(
    tmp_path,
):
    pack = _write_pack(tmp_path, {"alpha": _spec([])})
    report = dry_run(pack, runs=[_run()])

    assert "(this ruleset declares no conditions)" in render(report)


# -------------------------------------------------------------------------- the two shapes


def test_the_index_lists_a_child_but_marks_it_nested():
    """The table lists what the ruleset declares; only the marking differs by depth."""
    index = _condition_index(_spec([ONE_HANDLER, COMPOSITE]))

    assert list(index) == [
        "one_handler",
        "either_reading",
        "child_strict",
        "child_loose",
    ]
    assert index["either_reading"].nested is False
    assert index["child_strict"].nested is True


def test_the_json_carries_every_flag_the_text_marks(tmp_path):
    """The UI renders the text verbatim, so a consumer reading the JSON needs the same facts."""
    pack = _write_pack(tmp_path, {"alpha": _spec([ABSENT_LEAF, COMPOSITE])})
    payload = as_dict(dry_run(pack, runs=[_run()]))

    assert payload["pack"] == "candidate"
    assert payload["runs_replayed"] == 1 and payload["exercised"] is True
    by_id = {c["id"]: c for c in payload["rulesets"][0]["conditions"]}
    assert by_id["absent_leaf"]["always_unknown"] is True
    assert by_id["child_strict"]["nested"] is True
    assert by_id["child_strict"]["never_evaluated"] is False
    # One renderer, shared: the JSON carries the same text the tool result and the preview show.
    assert payload["text"].startswith("dry run over stored evidence")
    assert "folded  child_strict" in payload["text"]
