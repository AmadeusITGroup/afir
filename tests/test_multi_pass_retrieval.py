"""Multi-pass retrieval: the engine allows N passes, the knowledge pack declares them.

A pipeline that retrieves once can only ask about values the incident already carried. The live run
that prompted this mechanism scoped its query to the only identity the incident named, matched zero
rows in 306.8s, and reported success at every stage.

Four properties, only the first about the new feature:

* a single-pass run is byte-identical to before — every record key is the bare stage name, the
  snapshot shape unchanged, `passes` reading `{total: 1, current: 1}`;
* a follow-up pass ADDS: `queries` accumulates, `logs` merges per source, and only the new pass's
  queries execute, re-running the first pass's costing a primary source's whole scan budget for
  rows the run already holds;
* the cap stops the loop and SAYS what it dropped, a pass that never ran and a pass that found
  nothing leaving the same absent evidence;
* every control action addresses one pass, so retrying pass 2's fetch must not discard pass 1's
  rows.
"""

import pytest

from src.models.pydantic_models import (
    IncidentAnalysis,
    RetrievalQuery,
    UnderstandingResult,
)
from src.notifications import EventEmitter
from src.pipeline_runner import (
    DEFAULT_MAX_RETRIEVAL_PASSES,
    JobManager,
    JobRunMode,
    JobStatus,
    StageDescriptor,
    StageStatus,
    max_retrieval_passes,
    pass_key,
    split_pass_key,
)


def _emitter_with_manager(stages, **kwargs):
    emitter = EventEmitter()
    jm = JobManager(stages, emitter, **kwargs)
    emitter.set_job_manager(jm)
    return jm


def _incident():
    return {"id": "INC-1", "description": "test", "timestamp": "2024-01-01T00:00"}


# --- the pass key: pass 1 keeps the bare name ------------------------------
#
# `THINKING_STAGES`, `stage_gates.stages.*`, every UI label, every persisted job doc and
# `import_job` address a stage by its NAME, so a composite key for pass 1 unmatches all of them.


def test_pass_one_keys_on_the_bare_stage_name():
    assert pass_key("log_retrieval", 1) == "log_retrieval"
    assert pass_key("log_retrieval") == "log_retrieval"
    assert pass_key("log_retrieval", 2) == "log_retrieval#2"
    # A stage that is not repeatable never takes a suffix, whatever it is asked for: there
    # is one correlation and one report per run by design.
    assert pass_key("correlation", 3) == "correlation"


def test_split_pass_key_round_trips_and_is_defensive():
    assert split_pass_key("log_retrieval") == ("log_retrieval", 1)
    assert split_pass_key("log_retrieval#2") == ("log_retrieval", 2)
    # A malformed suffix reads as pass 1 rather than raising: the key may come from a
    # persisted document or a hand-edited export, and a KeyError there loses the whole job.
    assert split_pass_key("log_retrieval#x") == ("log_retrieval#x", 1)
    assert split_pass_key("log_retrieval#0") == ("log_retrieval", 1)


def test_the_cap_is_configurable_and_floored_at_one():
    assert max_retrieval_passes({}) == DEFAULT_MAX_RETRIEVAL_PASSES
    assert max_retrieval_passes({"jobs": {"max_retrieval_passes": 5}}) == 5
    # 1 is "no follow-up pass ever runs" — a valid operator choice, and the behaviour of
    # every deployment before this key existed.
    assert max_retrieval_passes({"jobs": {"max_retrieval_passes": 1}}) == 1
    assert max_retrieval_passes({"jobs": {"max_retrieval_passes": 0}}) == 1
    assert max_retrieval_passes({"jobs": {"max_retrieval_passes": "nonsense"}}) == (
        DEFAULT_MAX_RETRIEVAL_PASSES
    )
    assert max_retrieval_passes(None) == DEFAULT_MAX_RETRIEVAL_PASSES


# --- a fake pack + generator, declaring one follow-up pass -----------------
#
# Minimal duck types rather than real `KnowledgePack` / `ApiCallGenerator` objects: the RUNNER's
# loop is what is under test, and the declaration seam has its own tests.


class _Pack:
    """One ruleset, declaring whatever passes the test asked for.

    ``ruleset_keys`` / ``default_ruleset_key`` are part of the seam and not padding: the
    runner resolves WHICH procedure adjudicates before asking whether it declares a pass, so
    that another applicable ruleset's declaration cannot be run on this incident. A pack that
    could not name its procedures would leave that unanswerable.
    """

    def __init__(self, passes=None, keys=("only",), by_key=None):
        self._passes = passes or {}
        self._keys = list(keys)
        # {ruleset key: {pass number: spec}} — for the tests about WHOSE declaration counts.
        # `passes` is the shorthand for "the default ruleset declares these", which is every
        # other test in this file.
        self._by_key = by_key or {}

    def ruleset_keys(self):
        return list(self._keys)

    def default_ruleset_key(self):
        return self._keys[0] if self._keys else ""

    def ruleset_key_for(self, use_case):
        return str(use_case) if str(use_case) in self._keys else ""

    def follow_up_pass(self, key, number):
        if self._by_key:
            return (self._by_key.get(key) or {}).get(int(number))
        return (
            self._passes.get(int(number)) if key == self.default_ruleset_key() else None
        )


def _query(source, pass_number):
    """A real `RetrievalQuery`, because export/import runs it through the typed codec."""
    return RetrievalQuery(
        target_log_source=source,
        natural_language_query=f"pass {pass_number}: everything from {source}",
        date_from="2024-01-01",
        date_to="2024-01-02",
    )


def _tag(query):
    """`<source>@<pass>` — the batch label the assertions read."""
    return f"{query.target_log_source}@{query.natural_language_query.split(':')[0][-1]}"


class _Generator:
    """Plans two queries on pass 1 and one on each follow-up pass."""

    def __init__(self, pack):
        self.knowledge_pack = pack
        self.calls = []

    # No `_applicable_rulesets` stub: that method is gone from the real generator (the pass
    # spec now comes from the ONE adjudicating ruleset), and a fake offering a method the
    # real class does not have is how a dead seam keeps looking wired.

    async def generate(self, understanding, guidance=None):
        self.calls.append(("generate", None))
        return [_query("alerts", 1), _query("sessions", 1)]

    async def generate_follow_up(
        self,
        understanding,
        spec,
        logs,
        prior_queries=None,
        guidance=None,
        # Unused here, but part of the real signature: a double missing it raises `TypeError`,
        # which the stage reports as a failed plan.
        row_caps=None,
    ):
        number = int(spec["pass"])
        self.calls.append(("follow_up", number))
        # A real one harvests the values out of `logs` first; the shape is all the runner reads.
        # One query per target, since one entry may answer several sources off one harvest and a
        # fake reading only `source` would pass whatever the runner did with the rest.
        targets = [str(t).strip() for t in (spec.get("sources") or []) if str(t).strip()]
        if not targets:
            targets = [str(spec["source"]).strip()]
        return [_query(t, number) for t in targets], [f"note for pass {number}"]


class _Engine:
    """Returns one row per query, keyed by the query's own target source."""

    config = {}
    retrievers = {}

    def __init__(self):
        self.batches = []

    def row_caps(self):
        """THE REAL SIGNATURE again, and empty on purpose.

        The runner reads every source's `max_results` off the engine so a follow-up target
        whose harvest carries nothing new can be skipped as a repeat. An empty mapping is the
        conservative answer this fixture wants — an unknown cap means "the earlier answer may
        have been truncated", so no pass is ever skipped for that reason here and the tests
        below measure the pass machinery rather than the skip.
        """
        return {}

    async def retrieve(
        self,
        queries,
        progress_cb=None,
        extended=False,
        keyed_out=None,
        guidance=None,
        queries_out=None,
        # A DOUBLE MUST CARRY THE REAL SIGNATURE, however unused the parameter is here: the
        # stage calls the engine with every out-parameter it has, and a missing one is a
        # `TypeError` the stage reports as a failed retrieval rather than as a stale fake.
        unanswered_out=None,
    ):
        self.batches.append([_tag(q) for q in queries])
        out = {}
        for q in queries:
            out.setdefault(q.target_log_source, []).append({"from": _tag(q)})
            # The query text as PUBLISHED, filled before the source settles — the real engine
            # fills this at publish time so a reader never has to consult the retriever's own
            # `last_generated_query`, which outlives the run and can hold an earlier one's.
            if queries_out is not None:
                queries_out[q.target_log_source] = f"QUERY {_tag(q)}"
            if progress_cb:
                progress_cb(q.target_log_source, "completed", f"1 row from {_tag(q)}")
        if keyed_out is not None:
            keyed_out[f"keyed_by_pass_{len(self.batches)}"] = {"enforced": True}
        return out


async def _understanding(ctx):
    """A real `UnderstandingResult`, for the same reason `_query` is real."""
    return UnderstandingResult(
        incident_id=str(ctx.incident.get("id") or ""),
        analysis=IncidentAnalysis(
            incident_summary="a grant was made",
            severity_reasoning="",
            impact_assessment="",
            key_investigation_areas=[],
            log_sources_to_review=[],
            initial_hypotheses=[],
            recommended_actions=[],
            stakeholder_notification=[],
        ),
    )


def _stages():
    from src.pipeline_runner import _run_log_retrieval, _run_query_generation

    return [
        StageDescriptor("understanding", _understanding, "understanding"),
        StageDescriptor("query_generation", _run_query_generation, "queries"),
        StageDescriptor("log_retrieval", _run_log_retrieval, "logs"),
    ]


# The three stage names of this fixture's plan, pass 1. Named once so a test asserting on
# the plan reads as a statement about the PASSES and not about the stage list.
_PASS_1 = ["understanding", "query_generation", "log_retrieval"]


class _Correlation:
    """The seam the runner asks WHICH procedure adjudicates — the same one the verdict asks.

    Only `_playbook_correlation_spec` is implemented, because that is the whole question at
    this point in the run: it is pure token arithmetic over the incident summary and needs no
    logs, which is what lets the follow-up decision be taken a stage early.
    """

    def __init__(self, use_case="", boom=False):
        self._use_case = use_case
        self._boom = boom
        self.calls = 0

    def _playbook_correlation_spec(self, analysis):
        self.calls += 1
        if self._boom:
            raise RuntimeError("no playbook could be matched")
        return {"use_case": self._use_case} if self._use_case else {}


def _manager(passes=None, config=None, pack=None, correlation=None):
    pack = pack if pack is not None else _Pack(passes)
    generator = _Generator(pack)
    engine = _Engine()
    modules = {"api_call": generator, "log_retrieval": engine}
    if correlation is not None:
        modules["correlation"] = correlation
    jm = _emitter_with_manager(
        _stages(),
        modules=modules,
        config=config or {},
    )
    return jm, generator, engine


async def _run(jm):
    job = jm.create_job(_incident(), run_mode=JobRunMode.AUTO)
    await jm.run_job(job)
    return job


async def _settle(predicate, tries=400):
    """Wait for a control action's own scheduled task to reach ``predicate``.

    Polls the observable EFFECT rather than the job status: `control` returns as soon as it
    has scheduled `run_from_stage`, and the job it acts on is already COMPLETED, so a
    status poll succeeds before the re-run has started.
    """
    import asyncio

    for _ in range(tries):
        if predicate():
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)  # let the scheduled task unwind
    assert predicate(), "the control action never took effect"


# --- the no-regression pin -------------------------------------------------


async def test_a_pack_declaring_no_follow_up_runs_exactly_one_pass():
    """Every pack that exists today. Nothing about the run may change.

    Asserted on the record KEYS and the snapshot, not on the row counts: the compatibility
    guarantee is that a single-pass run addresses its stages by exactly the names it always
    did, because everything from the gate config to the UI labels keys on them.
    """
    jm, generator, engine = _manager()
    job = await _run(jm)

    assert job.status == JobStatus.COMPLETED
    assert job.stage_keys == _PASS_1
    assert set(job.stage_statuses) == set(_PASS_1)
    assert [c[0] for c in generator.calls] == ["generate"]
    assert len(engine.batches) == 1
    snap = job.snapshot()
    assert [s["name"] for s in snap["stages"]] == _PASS_1
    assert all(s["pass"] == 1 for s in snap["stages"])
    assert snap["passes"] == {"total": 1, "current": 1}


# --- a follow-up pass ADDS, it does not replace ----------------------------


async def test_a_second_pass_adds_queries_and_merges_rows():
    jm, generator, engine = _manager(
        passes={2: {"pass": 2, "source": "access_trail", "harvest": [{}]}}
    )
    job = await _run(jm)

    assert job.status == JobStatus.COMPLETED
    assert job.stage_keys == _PASS_1 + ["query_generation#2", "log_retrieval#2"]
    # ACCUMULATED, with the new plan last so a reader sees it in the order it was made.
    assert [_tag(q) for q in job.context.outputs["queries"]] == [
        "alerts@1",
        "sessions@1",
        "access_trail@2",
    ]
    # Pass 1's rows survive, and the follow-up source is there beside them.
    logs = job.context.outputs["logs"]
    assert sorted(logs) == ["access_trail", "alerts", "sessions"]
    assert logs["alerts"] == [{"from": "alerts@1"}]
    assert logs["access_trail"] == [{"from": "access_trail@2"}]
    # ONLY the new query ran. Re-running the first pass's would spend a primary source's
    # whole scan budget again for rows the run already holds — measured in hours.
    assert engine.batches == [["alerts@1", "sessions@1"], ["access_trail@2"]]
    assert generator.calls == [("generate", None), ("follow_up", 2)]


async def test_a_re_queried_source_keeps_BOTH_passes_rows():
    """The merge is the point: a pass that ADDS evidence must not subtract any.

    A follow-up legitimately re-reads a source the first pass already read, under a
    different scope — the first pass's rows answer the conditions it was planned for, and
    the second's answer the question that could not be asked yet. Replacing would delete
    one of the two silently, and the row count is the only place it would show.
    """
    jm, generator, engine = _manager(
        passes={2: {"pass": 2, "source": "alerts", "harvest": [{}]}}
    )
    job = await _run(jm)

    assert job.context.outputs["logs"]["alerts"] == [
        {"from": "alerts@1"},
        {"from": "alerts@2"},
    ]


async def test_each_pass_keeps_its_own_record_and_the_run_keeps_the_accumulation():
    """`pass_outputs` is the per-pass split; `outputs` is what correlation reads.

    Both are needed and neither substitutes: "pass 2 found nothing" and "pass 2 was never
    asked" are different findings, and the merged view cannot tell them apart.
    """
    jm, _generator, _engine = _manager(
        passes={2: {"pass": 2, "source": "access_trail", "harvest": [{}]}}
    )
    job = await _run(jm)

    records = job.context.pass_outputs
    assert sorted(records) == [1, 2]
    assert [_tag(q) for q in records[2]["queries"]] == ["access_trail@2"]
    assert records[2]["logs"] == {"access_trail": [{"from": "access_trail@2"}]}
    # The generator's notes ride the pass record, so a harvest that carried nothing (or
    # truncated) is readable rather than showing up only as a thin follow-up.
    assert records[2]["notes"] == ["note for pass 2"]
    # Pass 1 keeps a record on the same terms — it is what the panel's first page reads, and
    # what the accumulation is rebuilt from when a later pass runs.
    assert [_tag(q) for q in records[1]["queries"]] == ["alerts@1", "sessions@1"]
    assert sorted(records[1]["logs"]) == ["alerts", "sessions"]


async def test_three_passes_run_in_the_order_the_work_happened():
    """The keys are spliced in after the previous pass, not appended.

    Appended, a reader scrolling the stage cards would find the second fetch below the
    delivery step it ran hours before.
    """
    jm, generator, _engine = _manager(
        passes={
            2: {"pass": 2, "source": "access_trail", "harvest": [{}]},
            3: {"pass": 3, "source": "profile", "harvest": [{}]},
        },
        config={"jobs": {"max_retrieval_passes": 3}},
    )
    job = await _run(jm)

    assert job.stage_keys == _PASS_1 + [
        "query_generation#2",
        "log_retrieval#2",
        "query_generation#3",
        "log_retrieval#3",
    ]
    assert [c[1] for c in generator.calls] == [None, 2, 3]
    assert job.snapshot()["passes"] == {"total": 3, "current": 3}


async def test_a_pass_that_harvested_nothing_does_not_close_the_next_one():
    """Pass 3 opens after a pass 2 that added no query, and that is load-bearing.

    A procedure with TWO deferred questions has to number them 2 and 3, because one entry
    carries one target source. If the later pass depended on the earlier one HARVESTING
    something, those two questions would be coupled through nothing they share: the values
    come from the accumulated logs of pass 1, and each pass re-reads them. A pack author
    reading this loop the other way would decline to declare pass 3 at all — which is what
    `use_cases/abnormal_amount/rules.yaml` did until this was verified, leaving the one
    source that can widen the scope past the alert asked with no values on it.

    So the decision reads the declaration and the cap, and NOTHING about pass 2's result.
    """

    class _EmptyPass2(_Generator):
        async def generate_follow_up(
            self,
            understanding,
            spec,
            logs,
            prior_queries=None,
            guidance=None,
            row_caps=None,
        ):
            number = int(spec["pass"])
            self.calls.append(("follow_up", number))
            if number == 2:
                # The real generator's outcome when a harvest found no value: no queries,
                # and a note saying so. `skip_when_empty` is what produced it.
                return [], [f"pass {number} harvested nothing"]
            return [_query(spec["source"], number)], [f"note for pass {number}"]

    pack = _Pack(
        passes={
            2: {"pass": 2, "source": "access_trail", "harvest": [{}]},
            3: {"pass": 3, "source": "scope_sweep", "harvest": [{}]},
        }
    )
    generator = _EmptyPass2(pack)
    engine = _Engine()
    jm = _emitter_with_manager(
        _stages(),
        modules={"api_call": generator, "log_retrieval": engine},
        config={"jobs": {"max_retrieval_passes": 3}},
    )
    job = await _run(jm)

    assert [c[1] for c in generator.calls] == [None, 2, 3]
    assert job.context.pass_outputs[2]["queries"] == []
    assert [_tag(q) for q in job.context.pass_outputs[3]["queries"]] == ["scope_sweep@3"]
    # And the pass that ran is the one that fetched: pass 2 added no rows, pass 3 did.
    assert "access_trail" not in job.context.outputs["logs"]
    assert job.context.outputs["logs"]["scope_sweep"] == [{"from": "scope_sweep@3"}]


# --- WHOSE declaration opens the pass ------------------------------------
#
# A pass harvests values so the ADJUDICATING ruleset's conditions can be asked. Run from another
# procedure's declaration it is a full extra scan whose rows no condition reads, and it REPLACES
# the pass the adjudicating procedure would have run. Invisible from the outside: both produce a
# pass-2 query that runs against a real source and returns real rows.


_OTHERS_PASS_2 = {"pass": 2, "source": "somebody_elses_source", "harvest": [{}]}
_OWN_PASS_2 = {"pass": 2, "source": "access_trail", "harvest": [{}]}


async def test_a_pass_declared_only_by_ANOTHER_procedure_is_not_run(caplog):
    """The regression that has no artifact: a verified pass silently swapped for a stranger's.

    Both rulesets are applicable — same subject type — and only the non-adjudicating one
    declares pass 2. The run must be single-pass, and the log must name both procedures,
    because a declaration skipped for a reason is readable while a declaration honoured for
    the wrong procedure is not.
    """
    pack = _Pack(
        keys=("adjudicating", "other"),
        by_key={"other": {2: _OTHERS_PASS_2}},
    )
    jm, generator, engine = _manager(
        pack=pack, correlation=_Correlation("adjudicating")
    )
    with caplog.at_level("INFO"):
        job = await _run(jm)

    assert job.stage_keys == _PASS_1
    assert [c[0] for c in generator.calls] == ["generate"]
    assert len(engine.batches) == 1  # nobody else's scan was spent
    assert "'other'" in caplog.text and "adjudicating" in caplog.text


async def test_the_ADJUDICATING_procedures_own_pass_still_runs(caplog):
    """The positive control, and it is what proves the skip above is about OWNERSHIP.

    Same two rulesets, same shapes; the only difference is which one declares the pass. A fix
    that simply stopped opening pass 2 whenever a pack declares two procedures would pass the
    test above and fail this one.
    """
    pack = _Pack(
        keys=("adjudicating", "other"),
        by_key={"adjudicating": {2: _OWN_PASS_2}, "other": {2: _OTHERS_PASS_2}},
    )
    jm, generator, engine = _manager(
        pack=pack, correlation=_Correlation("adjudicating")
    )
    with caplog.at_level("INFO"):
        job = await _run(jm)

    assert job.stage_keys == _PASS_1 + ["query_generation#2", "log_retrieval#2"]
    # ITS source, not the other procedure's — the assertion the source name exists for.
    assert engine.batches[1] == ["access_trail@2"]
    assert "somebody_elses_source" not in caplog.text


async def test_the_matched_playbook_decides_and_the_DEFAULT_is_only_the_fallback():
    """Two readings of the same pack, differing only in which playbook the incident matched.

    The runner must ask the same seam the verdict asks, or the pass belongs to whichever
    procedure sorts first — which is what directory order decided before. Here the default
    (first-declared) is `adjudicating` and the matched use case is `other`, so a
    default-only reading would run nothing.
    """
    by_key = {"adjudicating": {2: _OWN_PASS_2}, "other": {2: _OTHERS_PASS_2}}
    jm, _generator, engine = _manager(
        pack=_Pack(keys=("adjudicating", "other"), by_key=by_key),
        correlation=_Correlation("other"),
    )
    job = await _run(jm)
    assert job.stage_keys == _PASS_1 + ["query_generation#2", "log_retrieval#2"]
    assert engine.batches[1] == ["somebody_elses_source@2"]

    # ...and with nothing matched, the pack's declared DEFAULT answers — the same fallback
    # the verdict itself takes, so the two stages cannot disagree about the procedure.
    jm2, _g2, engine2 = _manager(
        pack=_Pack(keys=("adjudicating", "other"), by_key=by_key),
        correlation=_Correlation(""),
    )
    job2 = await _run(jm2)
    assert job2.stage_keys == _PASS_1 + ["query_generation#2", "log_retrieval#2"]
    assert engine2.batches[1] == ["access_trail@2"]


async def test_selection_failing_falls_back_to_the_default_rather_than_the_run(caplog):
    """A follow-up pass is ADDITIONAL evidence, so failing to work out whose it is must cost
    the run nothing beyond that pass.

    The fallback is the pack's declared default and not "no pass", because a procedure that
    declared one still wants it; what must never happen is an exception reaching the stage
    loop and failing a run that already holds every pass-1 row.
    """
    jm, _generator, engine = _manager(
        pack=_Pack(
            keys=("adjudicating", "other"), by_key={"adjudicating": {2: _OWN_PASS_2}}
        ),
        correlation=_Correlation(boom=True),
    )
    with caplog.at_level("WARNING"):
        job = await _run(jm)

    assert job.status == JobStatus.COMPLETED
    assert job.stage_keys == _PASS_1 + ["query_generation#2", "log_retrieval#2"]
    assert engine.batches[1] == ["access_trail@2"]


async def test_a_run_with_no_correlation_module_still_opens_the_declared_pass():
    """The no-regression pin for every caller that wires a partial `modules` dict.

    The classic pipeline and several tests build one without a correlation module, and the
    follow-up decision must not start depending on it — the default ruleset is a property of
    the pack, answerable with no module at all.
    """
    jm, _generator, engine = _manager(passes={2: _OWN_PASS_2})
    job = await _run(jm)
    assert job.stage_keys == _PASS_1 + ["query_generation#2", "log_retrieval#2"]
    assert engine.batches[1] == ["access_trail@2"]


# --- a pass that asked nothing -------------------------------------------


async def test_a_pass_that_planned_NO_query_is_recorded_with_its_reason(caplog):
    """A declared pass can legitimately ask nothing, and then it must not vanish.

    Three things make a follow-up plan empty — the harvest found no value, no harvested type
    is filterable on the target, or every value it found is one the earlier retrieval of that
    source already asked — and in all three the fetch is skipped and the accumulated rows come
    back unchanged. The run is then byte-identical to one where the pass was never declared,
    which is the artifact a source with nothing to say produces: the exact confusion the
    multi-pass mechanism exists inside. So the reason becomes durable state and an event, and
    the notes the generator returned are what carries it — they were written by
    `generate_follow_up` and, before this, read by nothing at all.
    """
    jm, generator, engine = _manager(passes={2: _OWN_PASS_2})

    async def _asked_nothing(understanding, spec, logs, prior_queries=None,
                             guidance=None, row_caps=None):
        generator.calls.append(("follow_up", int(spec["pass"])))
        return [], ["nothing new was harvested for 'access_trail'"]

    generator.generate_follow_up = _asked_nothing
    with caplog.at_level("WARNING"):
        job = await _run(jm)

    # The pass RAN — it is in the plan, and its stages completed. A pass that asked nothing is
    # not a pass that failed, and collapsing the two would make the run unresumable at it.
    assert job.status == JobStatus.COMPLETED
    assert job.stage_keys == _PASS_1 + ["query_generation#2", "log_retrieval#2"]
    # And nothing was fetched twice: the second retrieval had no query to run, so the engine
    # was never called again and pass 1's rows come back unchanged.
    assert len(engine.batches) == 1

    asked = [i for i in job.interventions if i["action"] == "retrieval_pass_asked_nothing"]
    assert len(asked) == 1
    assert "nothing new was harvested for 'access_trail'" in asked[0]["detail"]
    # `engine`, because no human and no config decided this — the evidence did.
    assert asked[0]["actor"] == "engine"
    assert asked[0]["stage"] == "query_generation"

    # The operator watching the run is the one who can add the query by hand from the plan
    # editor if the skip was wrong, so it is an EVENT as well as a record.
    skipped = [e for e in job._event_history if e.get("type") == "pass_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["data"]["pass"] == 2
    assert skipped[0]["data"]["notes"] == ["nothing new was harvested for 'access_trail'"]
    assert "asked nothing" in skipped[0]["message"]
    assert "nothing new was harvested" in caplog.text


async def test_a_pass_that_DID_plan_a_query_records_nothing_of_the_kind():
    """The other direction, which is what makes the test above an assertion about emptiness.

    Every ordinary follow-up pass returns notes too (a bounded value list, a rejected value),
    so a reader keyed on "were there notes" would file every pass as one that asked nothing.
    """
    jm, _generator, engine = _manager(passes={2: _OWN_PASS_2})
    job = await _run(jm)

    assert len(engine.batches) == 2
    assert [i["action"] for i in job.interventions] == []
    assert [e for e in job._event_history if e.get("type") == "pass_skipped"] == []


# --- the cap -------------------------------------------------------------


async def test_the_cap_stops_the_loop_and_records_what_it_dropped(caplog):
    """A mis-authored pack that re-qualifies for its own next pass would loop forever.

    And a run that never finishes reports as one still running. The cap has to name the
    pass it refused: a pass that never ran and a pass that found nothing leave the same
    (absent) evidence.
    """
    jm, generator, _engine = _manager(
        passes={
            2: {"pass": 2, "source": "access_trail", "harvest": [{}]},
            3: {
                "pass": 3,
                "source": "profile",
                "harvest": [{}],
                "purpose": "the third question",
            },
        },
        config={"jobs": {"max_retrieval_passes": 2}},
    )
    with caplog.at_level("WARNING"):
        job = await _run(jm)

    assert job.stage_keys == _PASS_1 + ["query_generation#2", "log_retrieval#2"]
    assert [c[1] for c in generator.calls] == [None, 2]
    capped = [i for i in job.interventions if i["action"] == "retrieval_pass_capped"]
    assert len(capped) == 1
    assert "profile" in capped[0]["detail"]
    assert capped[0]["actor"] == "config"
    assert "profile" in caplog.text


async def test_a_cap_of_one_refuses_every_follow_up():
    jm, generator, engine = _manager(
        passes={2: {"pass": 2, "source": "access_trail", "harvest": [{}]}},
        config={"jobs": {"max_retrieval_passes": 1}},
    )
    job = await _run(jm)

    assert job.stage_keys == _PASS_1
    assert len(engine.batches) == 1
    assert [i["action"] for i in job.interventions] == ["retrieval_pass_capped"]


# --- per-pass state the downstream consumers read -------------------------


async def test_retrieval_facts_are_carried_forward_across_passes():
    """The scorer and the verdict engine read ONE dict for the whole run.

    Reset per pass, a first-pass keyed lookup would read as never-keyed the moment a second
    pass ran — turning an answered question ("this identity is not on the list") back into
    a gap, which is the difference between a finding and a shrug.
    """
    jm, _generator, _engine = _manager(
        passes={2: {"pass": 2, "source": "access_trail", "harvest": [{}]}}
    )
    job = await _run(jm)

    facts = job.context.stage_facts["log_retrieval"]
    assert sorted(facts["sources"]) == ["access_trail", "alerts", "sessions"]
    assert sorted(facts["keyed_sources"]) == ["keyed_by_pass_1", "keyed_by_pass_2"]
    # The follow-up pass ALSO keeps its own view, because merged, "3 sources completed"
    # hides that the one source this pass existed to ask timed out.
    own = job.context.stage_facts["log_retrieval#2"]
    assert sorted(own["sources"]) == ["access_trail"]
    assert own["pass"] == 2


async def test_the_pass_number_rides_the_source_progress_event():
    """The console keys its per-source table on this.

    Without it a follow-up pass's query text and row count overwrite the first pass's, and
    the panel shows one query where two ran — which is precisely the "pass 2 replaced pass
    1" failure the operator asked not to have.
    """
    import src.pipeline_runner as pr

    seen = []
    original = pr.emitter.emit

    def _capture(*args, **kwargs):
        if kwargs.get("event_type") == "source_progress" or (
            len(args) > 1 and args[1] == "source_progress"
        ):
            seen.append(dict(kwargs.get("data") or {}))

    pr.emitter.emit = _capture
    try:
        jm, _generator, _engine = _manager(
            passes={2: {"pass": 2, "source": "alerts", "harvest": [{}]}}
        )
        await _run(jm)
    finally:
        pr.emitter.emit = original

    by_pass = {}
    for item in seen:
        by_pass.setdefault(item.get("pass"), []).append(item.get("source"))
    assert by_pass[1] == ["alerts", "sessions"]
    assert by_pass[2] == ["alerts"]


async def test_a_new_pass_announces_itself_before_the_stages_re_run():
    """`pass_started` is what stops a follow-up pass reading as a retry.

    The two repeatable cards go from completed back to running, which is exactly what a
    retry looks like — so the console gets one line naming the pass, the source it is for
    and the stages about to repeat, and `BASIC_TYPES` lists it (`src/ui/script_tabs.py`)
    because a run that appears to retry itself is the shape an operator escalates.
    """
    jm, _generator, _engine = _manager(
        passes={
            2: {
                "pass": 2,
                "source": "access_trail",
                "harvest": [{}],
                "purpose": "was the exposure realised",
            }
        }
    )
    job = await _run(jm)

    # Read from the job's own replay buffer rather than a patched emitter: that buffer is
    # what a client attaching after the fact receives, so asserting on it also pins that a
    # re-attach can still tell a follow-up pass from a retry.
    seen = [e for e in job._event_history if e.get("type") == "pass_started"]
    assert len(seen) == 1, "one announcement per pass, and none for the first"
    event = seen[0]
    assert event["data"]["pass"] == 2
    # Named, because a pass whose purpose is unstated is a second scan of a system of
    # record that nobody can justify from the console.
    assert "access_trail" in event["message"]
    assert event["data"]["purpose"] == "was the exposure realised"
    # WHICH stages are about to repeat — the page uses it to decide that two cards going
    # backwards is a plan and not a failure.
    assert event["data"]["stages"] == ["query_generation#2", "log_retrieval#2"]
    # The bare stage name, like every other event: the pass rides in `data`, so a client
    # that knows nothing about passes still files this under a stage it recognises.
    assert event["stage"] == "query_generation"


async def test_a_pass_with_several_targets_names_EVERY_one_and_every_purpose():
    """One entry, several sources, one harvest — and the console must show the whole pass.

    The list form exists because a pass number is a scarce slot (`jobs.max_retrieval_passes`)
    and a procedure whose slots are exhausted cannot ask its next question at all. Once a pass
    carries two sources, every operator-facing sentence built from `spec["source"]` alone names
    only the first: the announcement reads as a pass that asked one source, and a reader then
    looks for a second declaration that does not exist. The purpose is per target for the same
    reason — the targets share the harvest, not the question — so a pass that spells it that
    way must not render as a pass with no stated purpose at all.
    """
    jm, _generator, engine = _manager(
        passes={
            2: {
                "pass": 2,
                "source": "access_trail",
                "sources": ["access_trail", "unit_reference"],
                "harvest": [{}],
                "purposes": {
                    "access_trail": "what the unit did",
                    "unit_reference": "what kind of unit it is",
                },
            }
        }
    )
    job = await _run(jm)

    # Both targets really were retrieved, in one pass, and each row is tagged with it.
    assert engine.batches[-1] == ["access_trail@2", "unit_reference@2"]
    assert sorted(job.context.stage_facts["log_retrieval#2"]["sources"]) == [
        "access_trail",
        "unit_reference",
    ]
    event = next(e for e in job._event_history if e.get("type") == "pass_started")
    assert "access_trail" in event["message"]
    assert "unit_reference" in event["message"]
    # Both questions, because `purpose` was never declared and reading only that key would
    # publish an empty purpose for a pass that stated two.
    assert "what the unit did" in event["data"]["purpose"]
    assert "what kind of unit it is" in event["data"]["purpose"]


async def test_a_pass_asked_for_but_no_longer_declared_returns_the_accumulation():
    """Reached by re-running the stage out of band — a retry after the pack was edited.

    The truthful outcome is that this pass adds nothing; deleting the accumulated plan
    would take the first pass's queries out of the job document, the export and the panel.
    """
    from src.pipeline_runner import _run_query_generation

    jm, _generator, _engine = _manager()
    job = jm.create_job(_incident())
    job.context.outputs["understanding"] = object()
    planned = [_query("alerts", 1)]
    job.context.outputs["queries"] = planned
    job.context.current_pass = 2

    assert await _run_query_generation(job.context) == planned


# --- control + gates address one pass ------------------------------------


async def test_retrying_one_pass_does_not_touch_the_other():
    jm, generator, engine = _manager(
        passes={2: {"pass": 2, "source": "access_trail", "harvest": [{}]}}
    )
    job = await _run(jm)
    assert job.status == JobStatus.COMPLETED

    jm.control(job.job_id, "retry_stage", "log_retrieval", pass_number=2)
    await _settle(lambda: len(engine.batches) >= 3)

    # The follow-up query ran a second time; the first pass's two did not.
    assert engine.batches == [
        ["alerts@1", "sessions@1"],
        ["access_trail@2"],
        ["access_trail@2"],
    ]
    # And re-fetching a pass REPLACES that pass's rows rather than merging onto them: the
    # run holds one row per source, not two for the re-fetched one. A doubled row count is a
    # wrong finding and not a cosmetic one — every distinct-count condition reads it.
    assert {k: len(v) for k, v in job.context.outputs["logs"].items()} == {
        "alerts": 1,
        "sessions": 1,
        "access_trail": 1,
    }
    retried = [i for i in job.interventions if i["action"] == "retry_stage"]
    assert retried and "pass 2" in retried[0]["detail"]


async def test_an_absent_pass_on_a_control_action_means_the_CURRENT_pass():
    """Every client written before multi-pass retrieval sends no `pass`.

    Defaulting to 1 instead would address the FIRST fetch on a run that took a follow-up,
    so an operator retrying the stage they are looking at would re-scan every source the
    first plan named and discard the pass-2 rows on screen.
    """
    jm, _generator, engine = _manager(
        passes={2: {"pass": 2, "source": "access_trail", "harvest": [{}]}}
    )
    job = await _run(jm)

    jm.control(job.job_id, "retry_stage", "log_retrieval")
    await _settle(lambda: len(engine.batches) >= 3)

    assert engine.batches[-1] == ["access_trail@2"]


async def test_retry_all_takes_the_plan_back_to_the_fixed_stage_list():
    """A follow-up pass is a consequence of results `retry_all` has just discarded.

    Left in place, pass 2's keys would sit PENDING for a pass that may never be declared
    again — a stage the UI shows as queued and nothing ever runs.
    """
    jm, _generator, engine = _manager(
        passes={2: {"pass": 2, "source": "access_trail", "harvest": [{}]}}
    )
    job = await _run(jm)
    assert len(job.stage_keys) == len(_PASS_1) + 2

    jm.control(job.job_id, "retry_all")
    await _settle(lambda: len(engine.batches) >= 4)

    # Re-declared on the re-run — but through `register_pass`, from an honest starting
    # point, rather than left over from the run that was just discarded.
    assert job.stage_keys == _PASS_1 + ["query_generation#2", "log_retrieval#2"]
    assert all(s == StageStatus.COMPLETED for s in job.stage_statuses.values())


async def test_an_override_can_be_booked_against_one_pass():
    jm, _generator, _engine = _manager(
        passes={2: {"pass": 2, "source": "access_trail", "harvest": [{}]}}
    )
    job = await _run(jm)

    jm.set_stage_output(
        job.job_id, "log_retrieval", {"fixed": [{"x": 1}]}, actor="me", pass_number=2
    )

    assert job.stage_summaries["log_retrieval#2"]
    assert "log_retrieval#2" in job.stage_health
    trail = [i for i in job.interventions if i["action"] == "override_stage_output"]
    assert trail and "pass 2" in trail[0]["detail"]
    # The stage NAME is what the trail records, so a stage-name filter still finds it.
    assert trail[0]["stage"] == "log_retrieval"


async def test_the_pass_state_survives_an_export_import_round_trip():
    """A restored run that took a follow-up pass must not read as one that never did.

    A pass is declared FROM results, and `import_job` has no pack to ask — so without the
    exported plan the pass-2 statuses would have no key to land on, be dropped, and
    `resume` would re-plan a pass whose rows the run already holds.
    """
    jm, _generator, _engine = _manager(
        passes={2: {"pass": 2, "source": "access_trail", "harvest": [{}]}}
    )
    job = await _run(jm)
    doc = jm.export_job(job.job_id)

    fresh, _g, _e = _manager()  # a manager that knows NOTHING about the declaration
    restored = fresh.import_job(doc)

    assert restored.stage_keys == _PASS_1 + ["query_generation#2", "log_retrieval#2"]
    assert restored.stage_statuses["log_retrieval#2"] == StageStatus.COMPLETED
    assert restored.snapshot()["passes"]["total"] == 2


async def test_an_export_from_before_multi_pass_still_imports():
    """A persisted doc written by an earlier release carries no `stage_keys`."""
    jm, _generator, _engine = _manager()
    job = await _run(jm)
    doc = jm.export_job(job.job_id)
    doc.pop("stage_keys")
    doc.pop("current_pass")

    restored = jm.import_job(doc)
    assert restored.stage_keys == _PASS_1
    assert restored.stage_statuses["log_retrieval"] == StageStatus.COMPLETED


# --- the gate on a follow-up pass ---------------------------------------
#
# `log_retrieval` gates every time (a threshold above any attainable score) and
# `query_generation` never does (0.0), so each gate below is unambiguously the fetch and the
# assertions are about which pass's.

_GATE_ON_RETRIEVAL = {
    "stage_gates": {
        "stages": {
            "log_retrieval": {"threshold": 1.1},
            # 0.0 is "never gate on health" — the fixture's understanding and query plans
            # are deliberately thin and would otherwise gate first, making every assertion
            # below about the wrong stage.
            "query_generation": {"threshold": 0.0},
            "understanding": {"threshold": 0.0},
        }
    }
}


async def _next_gate(job, after=None, timeout=3.0):
    """The next gate to open, distinct from ``after``.

    Keyed on `opened_at` identity rather than on mere presence: `resolve_gate` sets the
    event but the runner clears `open_gate` only once it wakes, so a presence poll returns
    the STALE gate and every assertion after it races.
    """
    import asyncio

    stale = (after or {}).get("opened_at")

    async def _poll():
        while job.open_gate is None or (
            stale is not None and job.open_gate.get("opened_at") == stale
        ):
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout=timeout)
    return dict(job.open_gate)


async def test_a_gate_on_a_follow_up_pass_names_its_pass():
    """The gate record carries `pass` beside `stage`, and resolving it addresses that one.

    Two records of one stage otherwise resolve each other: an operator approving the
    follow-up fetch would release the gate on the first one.
    """
    import asyncio

    jm, _generator, _engine = _manager(
        passes={2: {"pass": 2, "source": "access_trail", "harvest": [{}]}},
        config=_GATE_ON_RETRIEVAL,
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))

    first = await _next_gate(job)
    assert (first["stage"], first["pass"]) == ("log_retrieval", 1)
    jm.resolve_gate(job.job_id, "approve", actor="me")

    # The follow-up pass's fetch gates under the same stage NAME, one pass along.
    second = await _next_gate(job, after=first)
    assert (second["stage"], second["pass"]) == ("log_retrieval", 2)
    jm.resolve_gate(job.job_id, "approve", actor="me")

    await asyncio.wait_for(task, timeout=3)
    assert job.status == JobStatus.COMPLETED


async def test_rejecting_a_follow_up_gate_re_runs_that_pass_not_the_first():
    """`restart_pass` defaults to the gate's own pass.

    Resolving to pass 1 would discard the accumulated rows and re-scan every source the
    first plan named — for a correction whose whole scope is the follow-up question.
    """
    import asyncio

    jm, _generator, engine = _manager(
        passes={2: {"pass": 2, "source": "access_trail", "harvest": [{}]}},
        config=_GATE_ON_RETRIEVAL,
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))

    first = await _next_gate(job)
    jm.resolve_gate(job.job_id, "approve", actor="me")
    second = await _next_gate(job, after=first)
    assert second["pass"] == 2
    jm.resolve_gate(job.job_id, "reject", guidance="widen the receivers", actor="me")

    # Re-gated on the RE-RUN of pass 2, not on a fresh pass 1.
    third = await _next_gate(job, after=second)
    assert (third["stage"], third["pass"]) == ("log_retrieval", 2)
    jm.resolve_gate(job.job_id, "approve", actor="me")
    await asyncio.wait_for(task, timeout=3)

    assert job.status == JobStatus.COMPLETED
    # Pass 1's queries ran once; the follow-up's ran twice (the rejection re-ran it).
    assert engine.batches.count(["alerts@1", "sessions@1"]) == 1
    assert engine.batches.count(["access_trail@2"]) == 2
    # The guidance landed on the FOLLOW-UP pass's key, so it steers that pass's re-run and
    # not pass 1's — the two are one stage NAME and would otherwise share one guidance list.
    assert job.context.stage_guidance["log_retrieval#2"] == ["widen the receivers"]
    assert "log_retrieval" not in job.context.stage_guidance


async def test_a_reject_can_send_the_run_back_to_the_follow_up_PLAN():
    """`restart_from` + `restart_pass` reach one pass's query generation.

    "The follow-up asked the wrong question" is a correction to the PLAN, not to the fetch,
    and the plan for pass 2 is a different record from the plan for pass 1: re-planning pass
    1 would discard the accumulated rows and re-scan every source the first plan named.
    """
    import asyncio

    jm, generator, engine = _manager(
        passes={2: {"pass": 2, "source": "access_trail", "harvest": [{}]}},
        config=_GATE_ON_RETRIEVAL,
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))

    first = await _next_gate(job)
    jm.resolve_gate(job.job_id, "approve", actor="me")
    second = await _next_gate(job, after=first)
    jm.resolve_gate(
        job.job_id,
        "reject",
        guidance="the receiver side, not the retriever side",
        restart_from="query_generation",
        restart_pass=2,
        actor="me",
    )

    third = await _next_gate(job, after=second)
    assert (third["stage"], third["pass"]) == ("log_retrieval", 2)
    jm.resolve_gate(job.job_id, "approve", actor="me")
    await asyncio.wait_for(task, timeout=3)

    assert job.status == JobStatus.COMPLETED
    # The follow-up plan was made twice, pass 1's once.
    assert [c[1] for c in generator.calls] == [None, 2, 2]
    assert job.context.stage_guidance["query_generation#2"] == [
        "the receiver side, not the retriever side"
    ]
    # And re-planning did not multiply the accumulation: the pass's own record replaced its
    # earlier queries rather than appending to them.
    assert [_tag(q) for q in job.context.outputs["queries"]] == [
        "alerts@1",
        "sessions@1",
        "access_trail@2",
    ]


@pytest.mark.parametrize("number", [2, 3])
def test_register_pass_is_idempotent(number):
    """A pass can be re-entered: a rejected gate re-runs it, a restored job may too.

    Re-registering must not duplicate a key or reset a status the earlier attempt
    recorded — `_reset_stage_for_retry` is what clears an outcome, deliberately and per
    stage.
    """
    jm, _generator, _engine = _manager()
    job = jm.create_job(_incident())
    for n in range(2, number + 1):
        job.register_pass(n)
    job.stage_statuses[pass_key("log_retrieval", number)] = StageStatus.COMPLETED

    assert job.register_pass(number) == []
    assert len(job.stage_keys) == len(set(job.stage_keys))
    assert job.stage_statuses[pass_key("log_retrieval", number)] == (
        StageStatus.COMPLETED
    )
