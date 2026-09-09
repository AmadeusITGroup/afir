"""The selection delta: what an authoring edit does to every past incident's procedure.

Why these tests are shaped the way they are. The module answers a question with **two sides**
— the same corpus under the base pack and under the candidate — so almost every test here
asserts a *difference*, and the ones that do not are asserting a **silence**: `compared is
False` with an empty `flips` is "this deployment cannot say", and reporting it as "nothing
flipped" is the failure the module exists to avoid. Each of those has its own test, because a
guard that only fires when a sibling guard would is a guard nobody has tested.

Nothing here reaches a pack directory except the two tests that mean to, and nothing reaches a
real `JobStore`: the corpus arrives through a fake with a `load_all`, which is the whole seam.
"""

import shutil

import pytest

from src.knowledge import pack_selection_delta as psd
from tests.installed_packs import FIXTURE_PACK, PACKS_ROOT

# --------------------------------------------------------------------------- the fixtures

#: Two specs whose vocabularies do not overlap, so the base selection is unambiguous.
BASE = [
    {"use_case": "alpha", "title": "Widget tampering", "keys": []},
    {"use_case": "beta", "title": "Gadget skimming", "keys": []},
]

#: `beta` retitled to take the summary's own word. Only `beta`'s vocabulary moves; `alpha`'s
#: score collapses anyway, because the weights are inverse SPEC FREQUENCY over both sets.
BETA_TAKES_IT = [
    {"use_case": "alpha", "title": "Widget tampering", "keys": []},
    {"use_case": "beta", "title": "Widget tampering counter", "keys": []},
]

#: Both specs declaring one vocabulary: every token is in both, so every weight is zero and
#: the selector abstains. The one direction an author never intends.
NOBODY_WINS = [
    {"use_case": "alpha", "title": "Widget tampering", "keys": []},
    {"use_case": "beta", "title": "Widget tampering", "keys": []},
]

SUMMARY = "Report of widget tampering at the counter."


def row(job_id="j1", summary=SUMMARY, label="", hypotheses=(), areas=()):
    return psd.CorpusRow(
        job_id=job_id,
        summary=summary,
        label=label,
        hypotheses=tuple(hypotheses),
        areas=tuple(areas),
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


def job(job_id, summary, hypotheses=None, areas=None):
    return {
        "job_id": job_id,
        "outputs": {
            "understanding": {
                "analysis": {
                    "incident_summary": summary,
                    "initial_hypotheses": hypotheses or [],
                    "key_investigation_areas": areas or [],
                }
            }
        },
    }


# ------------------------------------------------------------------------------- the flip


def test_a_title_edit_that_flips_a_labelled_run_names_that_run():
    """The deliverable: the run id, both readings, and which way it moved relative to the label."""
    delta = psd.compare(BASE, BETA_TAKES_IT, [row(job_id="job-7", label="alpha")])

    assert delta.compared is True
    assert delta.scored == 1
    assert delta.vocabulary_changed == ("beta",)
    assert len(delta.flips) == 1
    flip = delta.flips[0]
    assert flip.job_id == "job-7"
    assert (flip.before, flip.after) == ("alpha", "beta")
    assert (flip.before_basis, flip.after_basis) == ("scored", "scored")
    assert flip.before_score > flip.after_score > 0
    # The label is an independent reading, so the direction is reportable: this edit moved the
    # incident AWAY from the procedure a human said adjudicates it.
    assert flip.toward_label is False
    assert flip.lost_recognition is False
    assert "job-7" in psd.render(delta)
    assert "away from the expected procedure (alpha)" in psd.render(delta)


def test_a_flip_toward_the_label_is_reported_as_such():
    """The opposite direction of the same field, because a flip is usually the point of the edit.

    Asserted separately rather than as a second case of the test above: `toward_label` has
    three values and a property that only ever returns one of them reads as working.
    """
    delta = psd.compare(BASE, BETA_TAKES_IT, [row(label="beta")])
    assert delta.flips[0].toward_label is True
    assert "toward the expected procedure (beta)" in psd.render(delta)


def test_an_unlabelled_flip_states_no_direction():
    """No label means no opinion. `None` and `False` license different reactions."""
    delta = psd.compare(BASE, BETA_TAKES_IT, [row()])
    assert delta.flips[0].toward_label is None
    assert "expected procedure" not in psd.render(delta)


def test_recognition_lost_is_called_out_by_name():
    """`scored -> no_match` is the one direction an author never intends, so it says so.

    The candidate here gives both specs one vocabulary, which is what merging two procedures'
    titles does: every token is now in both specs, every inverse-frequency weight is zero, and
    an incident that scored 1.0 selects nothing at all.
    """
    delta = psd.compare(BASE, NOBODY_WINS, [row(job_id="job-9", label="alpha")])

    flip = delta.flips[0]
    assert (flip.before, flip.after) == ("alpha", "")
    assert (flip.before_basis, flip.after_basis) == ("scored", "no_match")
    assert flip.lost_recognition is True
    assert delta.lost_recognition == (flip,)
    assert "recognition LOST" in psd.render(delta)


def test_a_procedure_reached_by_being_the_only_one_is_not_the_same_selection():
    """The BASIS is part of a flip's identity, not just the use-case name.

    A one-spec pack returns its spec with no keyword matching at all — every weight is
    `(1-1)/1` — so `sole_spec` and `scored` are different claims about the same name, and a
    pack going from one spec to two is exactly the edit that makes the difference matter.
    """
    one = [{"use_case": "alpha", "title": "Widget tampering", "keys": []}]
    delta = psd.compare(one, BASE, [row()])

    assert len(delta.flips) == 1
    flip = delta.flips[0]
    assert flip.before == flip.after == "alpha"
    assert (flip.before_basis, flip.after_basis) == ("sole_spec", "scored")


# ---------------------------------------------------------------------------- the silences


def test_an_unrelated_edit_reports_nothing_and_reads_no_corpus():
    """The short-circuit, and it is exact rather than an optimisation.

    A score is a function of the token sets alone — the inverse-frequency weights are counts
    over those same sets — so identical sets score identically on every input. Which is what
    makes it safe to answer without looking at the corpus at all, and the assertion on
    `store.calls` is the half that matters: a check costing seconds on every save is a check
    that gets turned off.
    """
    candidate = [dict(s, conditions=[{"id": "c1"}]) for s in BASE]
    store = FakeStore([job("j1", SUMMARY)])

    delta = psd.selection_delta(BASE, candidate, store=store)

    assert store.calls == 0
    assert delta.compared is False
    assert delta.flips == ()
    assert "no playbook title or join key changed" in delta.reason
    assert "not compared" in psd.render(delta)


def test_no_corpus_is_a_silence_and_not_a_clean_bill():
    """Mirrors `pack_validate._check_field_paths`: nothing to check against means say nothing.

    The vocabulary DID move here, so the comparison was wanted and could not be made. Reading
    the empty `flips` as "no incident flips" is the whole failure — it is a guarantee this
    deployment is in no position to give.
    """
    delta = psd.compare(BASE, BETA_TAKES_IT, [])

    assert delta.compared is False
    assert delta.flips == ()
    assert delta.corpus == 0
    assert delta.vocabulary_changed == ("beta",)
    assert "cannot say" in delta.reason
    rendered = psd.render(delta)
    assert "not compared" in rendered
    assert "no stored incident's selection moves" not in rendered


def test_no_flip_over_a_real_corpus_says_what_it_does_not_claim():
    """The other empty list, which is a measurement — and it still refuses to bless the edit."""
    unrelated = [
        {"use_case": "alpha", "title": "Widget tampering", "keys": []},
        {"use_case": "beta", "title": "Gadget skimming trawl", "keys": []},
    ]
    delta = psd.compare(BASE, unrelated, [row()])

    assert delta.compared is True
    assert delta.scored == 1
    assert delta.flips == ()
    rendered = psd.render(delta)
    assert "no stored incident's selection moves" in rendered
    assert "not a claim that the edit is correct" in rendered


def test_neither_pack_declaring_a_correlation_block_is_reported_as_such():
    """Two empty spec lists is a third silence, and it is not the no-corpus one."""
    delta = psd.compare([], [], [row()])
    assert delta.compared is False
    assert "neither pack declares" in delta.reason


# -------------------------------------------------------------------------------- the bounds


def test_a_cut_flip_list_says_so():
    """A reader who has learned these lists are complete reads a truncated one as the whole
    consequence — so the remainder is a field, not a dropped tail."""
    rows = [row(job_id=f"j{i}") for i in range(psd.MAX_FLIPS + 5)]
    delta = psd.compare(BASE, BETA_TAKES_IT, rows)

    assert delta.scored == psd.MAX_FLIPS + 5
    assert len(delta.flips) == psd.MAX_FLIPS
    assert delta.flips_cut == 5
    assert f"and 5 more flipped incident(s) not listed (bound: {psd.MAX_FLIPS})" in (
        psd.render(delta)
    )


def test_an_unscored_row_names_its_own_direction_of_error():
    """`max_rows` bounds the report, and an unscored incident's flip is invisible — said, not
    implied, because the reader's remedy is to raise the bound."""
    rows = [row(job_id=f"j{i}") for i in range(3)]
    delta = psd.compare(BASE, BETA_TAKES_IT, rows, max_rows=2)

    assert delta.scored == 2
    assert delta.corpus == 3
    assert any("would not appear here" in p for p in delta.problems)
    assert any("bound: 2" in p for p in delta.problems)


def test_the_serialised_delta_carries_the_servers_own_sentences():
    """The preview must not re-derive the reading in JavaScript.

    `to_dict` is the whole payload the plan preview gets, so the rendered text rides on it the
    way it does on `pack_dry_run.as_dict`: two formatters over one measurement is two readings,
    and the one the operator approves would not be the one the model proposed. Asserted for a
    delta that was NOT compared too, because that is the case with no flips to print and the
    reason is the entire content.
    """
    compared = psd.compare(BASE, BETA_TAKES_IT, [row()]).to_dict()
    assert compared["text"] == psd.render(psd.compare(BASE, BETA_TAKES_IT, [row()]))
    assert "beta" in compared["text"]

    unmoved = psd.compare(BASE, BASE, [row()]).to_dict()
    assert unmoved["compared"] is False
    assert unmoved["vocabulary_changed"] == []
    assert unmoved["reason"] and unmoved["reason"] in unmoved["text"]


# ------------------------------------------------------------------------------ the scoring


def test_the_score_comes_from_the_engine_including_its_tie_break():
    """One scorer, not two — asserted on the one input a summary-only copy gets wrong.

    The selector reads the incident summary as primary evidence and the hypotheses and
    investigation areas as a TIE-BREAK, so a summary carrying no spec's vocabulary can still
    select on a hypothesis. A local re-implementation of the scorer read the summary only and
    answered `no_match` here; that is the drift this module exists to have removed.
    """
    scored = row(
        summary="Unspecified anomaly reported by the overnight desk.",
        hypotheses=("Possible gadget skimming at the counter",),
    )
    use_case, basis = psd.score(BASE, scored)

    assert use_case == "beta"
    assert basis.basis == "scored"
    assert basis.score == 0.0  # nothing matched the summary; the tie-break decided


def test_the_vocabulary_is_split_as_prose_and_covers_the_join_keys():
    """Both halves of the scoring input, through the engine's own prose splitter.

    A field-name splitter returns a multi-word title as one unmatchable token, so a copy of
    the splitter here could confirm an unchanged vocabulary while the scores moved.
    """
    vocab = psd.spec_vocabulary(
        [{"use_case": "alpha", "title": "Widget tampering", "keys": ["depot_id"]}]
    )
    assert vocab == {"alpha": {"widget", "tampering", "depot", "id"}}


def test_a_removed_or_added_spec_counts_as_a_vocabulary_change():
    """Not only an edited title: a use case appearing or disappearing moves every weight."""
    assert psd.vocabulary_changed(BASE, BASE) == ()
    assert psd.vocabulary_changed(BASE, BASE[:1]) == ("beta",)
    assert psd.vocabulary_changed(BASE[:1], BASE) == ("beta",)


# ------------------------------------------------------------------------------- the corpus


def test_the_corpus_keeps_a_run_that_retrieved_nothing():
    """The reason this module reads its own corpus instead of `pack_dry_run.stored_runs`.

    That function keeps only runs that retrieved rows — correctly, for a replay — and a run
    that retrieved nothing is precisely what a mis-selected procedure produces. Filtering
    those out would hide the failure being reported.
    """
    docs = [job("empty-run", SUMMARY)]  # no `logs` output at all
    rows, problems = psd.corpus(FakeStore(docs))

    assert problems == []
    assert [r.job_id for r in rows] == ["empty-run"]
    assert rows[0].summary == SUMMARY


def test_the_corpus_dedupes_on_the_summary_and_keeps_the_newest():
    """The history is re-runs of a much smaller set of incidents.

    Counting one replayed alert twelve times turns one flip into twelve, which is a blast
    radius that reads as twelve times the size it is. `load_all` is oldest-first, so the
    newest reading of an incident is the one kept.
    """
    docs = [job("old", SUMMARY), job("new", SUMMARY), job("other", "A different alert.")]
    rows, problems = psd.corpus(FakeStore(docs))

    assert problems == []
    assert [r.job_id for r in rows] == ["other", "new"]


def test_the_corpus_carries_the_tie_break_fields():
    """Hypotheses and areas are read, or the engine's secondary evidence is silently dropped."""
    docs = [job("j1", SUMMARY, hypotheses=["h one"], areas=["a one"])]
    rows, _ = psd.corpus(FakeStore(docs))

    assert rows[0].hypotheses == ("h one",)
    assert rows[0].areas == ("a one",)


def test_a_run_with_no_summary_is_skipped_without_a_problem():
    """An un-run or cancelled job has no understanding output; that is ordinary, not a fault."""
    docs = [{"job_id": "queued"}, job("done", SUMMARY)]
    rows, problems = psd.corpus(FakeStore(docs))

    assert [r.job_id for r in rows] == ["done"]
    assert problems == []


def test_one_unreadable_job_document_costs_only_itself():
    """A comparison that aborts on one bad document tells an author less than one reporting 60
    of 61 — so the failure is a `problems` entry naming the run, and the rest are scored."""
    docs = [{"job_id": "broken", "outputs": ["not", "a", "mapping"]}, job("fine", SUMMARY)]
    rows, problems = psd.corpus(FakeStore(docs))

    assert [r.job_id for r in rows] == ["fine"]
    assert len(problems) == 1
    assert "broken" in problems[0]
    assert "AttributeError" in problems[0]


def test_an_unreadable_history_is_a_finding_and_not_a_crash():
    """The whole store failing is the same class of answer, one level up."""
    rows, problems = psd.corpus(FakeStore(raises=OSError("volume unreachable")))

    assert rows == []
    assert len(problems) == 1
    assert "volume unreachable" in problems[0]


def test_a_labels_map_reaches_the_rows_it_names():
    """The label is supplied per job id and never derived — what an incident is about is domain
    knowledge, and this module is engine surface."""
    docs = [job("j1", SUMMARY), job("j2", "A different alert.")]
    rows, _ = psd.corpus(FakeStore(docs), labels={"j1": "alpha"})

    by_id = {r.job_id: r.label for r in rows}
    assert by_id == {"j1": "alpha", "j2": ""}


def test_the_corpus_problems_ride_on_the_delta():
    """A partial corpus must be visible on the result, or 1 of 61 rows reads like all of them."""
    docs = [{"job_id": "broken", "outputs": ["x"]}, job("fine", SUMMARY)]
    delta = psd.selection_delta(BASE, BETA_TAKES_IT, store=FakeStore(docs))

    assert delta.compared is True
    assert delta.scored == 1
    assert any("broken" in p for p in delta.problems)


# --------------------------------------------------------------------- the directory entry


def test_two_pack_directories_are_compared_through_the_pipeline_loader(tmp_path):
    """The plan-preview entry point, over the real fixture pack and a retitled copy.

    Both sides go through `load_knowledge_pack`, so what is asserted here is that a playbook
    title edit is *seen* — the loader is the only thing between a `.md` file and a score.
    """
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    shutil.copytree(PACKS_ROOT / FIXTURE_PACK, base)
    shutil.copytree(PACKS_ROOT / FIXTURE_PACK, candidate)

    playbooks = sorted(candidate.rglob("playbooks/*.md"))
    edited = 0
    for path in playbooks:
        text = path.read_text()
        if "Courier Collusion" in text:
            path.write_text(text.replace("Courier Collusion", "Handler Substitution"))
            edited += 1
    assert edited == 1, "the fixture pack's playbook titles moved; re-point this test"

    delta = psd.selection_delta_for_dirs(base, candidate, store=FakeStore())

    assert delta.problems == ()
    assert delta.base_specs == delta.candidate_specs > 1
    assert delta.vocabulary_changed  # the retitled use case, whatever it is called
    # No corpus in this fixture, so the comparison itself is a silence — and that is the
    # assertion: the vocabulary change is reported even when the flips cannot be.
    assert delta.compared is False


def test_a_pack_directory_that_will_not_load_is_reported_not_raised(tmp_path):
    """An unloadable candidate is another check's finding; refusing to answer here would hide
    the flips in a plan that has a second, unrelated defect."""
    delta = psd.selection_delta_for_dirs(
        tmp_path / "no-such-pack", tmp_path / "also-missing", store=FakeStore()
    )

    assert delta.compared is False
    assert isinstance(delta.reason, str)


@pytest.mark.parametrize(
    "call",
    [
        lambda: psd.compare(BASE, BETA_TAKES_IT, [row()]),
        lambda: psd.compare([], [], []),
        lambda: psd.selection_delta(BASE, BETA_TAKES_IT, store=FakeStore(raises=OSError)),
        lambda: psd.corpus(FakeStore(raises=RuntimeError("boom"))),
    ],
)
def test_nothing_in_the_module_raises(call):
    """Every failure becomes a reported field. A preview that 500s on an unreadable job
    document is a preview an author stops opening."""
    call()
