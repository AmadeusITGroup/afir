"""
Tests for the job-based pipeline runner (src/pipeline_runner.py).

Stages are replaced with lightweight async callables via a custom stage list, so
these run with no LLM / retriever / network. They exercise the control surface
(pause/resume/step/cancel/retry), SSE fan-out, and export/import round-tripping.
"""

import asyncio
import logging
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.models.pydantic_models import (AnomalyItem, CorrelationResult,
                                        ExtractedEntity, IncidentAnalysis,
                                        RetrievalQuery, UnderstandingResult)
from src.notifications import EventEmitter
from src.pipeline_runner import (JobManager, JobRunMode, JobStatus,
                                 StageDescriptor, StageStatus)


def _emitter_with_manager(stages, **kwargs):
    emitter = EventEmitter()
    jm = JobManager(stages, emitter, **kwargs)
    emitter.set_job_manager(jm)
    return jm


def _incident():
    return {"id": "INC-1", "description": "test", "timestamp": "2024-01-01T00:00"}


# --- basic run -------------------------------------------------------------


async def test_all_stages_complete():
    order = []

    async def s1(ctx):
        order.append("a")
        return "ra"

    async def s2(ctx):
        order.append("b")
        return "rb"

    stages = [
        StageDescriptor("a", s1, "oa"),
        StageDescriptor("b", s2, "ob"),
    ]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident(), run_mode=JobRunMode.AUTO)
    await jm.run_job(job)

    assert job.status == JobStatus.COMPLETED
    assert order == ["a", "b"]
    assert job.context.outputs == {"oa": "ra", "ob": "rb"}
    assert all(s == StageStatus.COMPLETED for s in job.stage_statuses.values())


async def test_best_effort_failure_does_not_fail_job():
    async def ok(ctx):
        return 1

    async def boom(ctx):
        raise RuntimeError("delivery down")

    stages = [
        StageDescriptor("ok", ok, "x"),
        StageDescriptor("out", boom, "y", is_best_effort=True),
    ]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    await jm.run_job(job)

    assert job.status == JobStatus.COMPLETED
    assert job.stage_statuses["out"] == StageStatus.COMPLETED
    assert "y" not in job.context.outputs  # best-effort output not stored on failure


# --- failure + retry -------------------------------------------------------


async def test_stage_failure_then_retry_stage_completes():
    calls = {"n": 0}

    async def flaky(ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first attempt fails")
        return "ok"

    async def after(ctx):
        return "done"

    stages = [
        StageDescriptor("flaky", flaky, "f"),
        StageDescriptor("after", after, "a"),
    ]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    await jm.run_job(job)

    assert job.status == JobStatus.STAGE_FAILED
    assert job.stage_statuses["flaky"] == StageStatus.FAILED

    jm.control(job.job_id, "retry_stage")
    # retry_stage schedules run_from_stage as a task; let it finish.
    await asyncio.sleep(0.05)

    assert job.status == JobStatus.COMPLETED
    assert job.context.outputs == {"f": "ok", "a": "done"}


async def test_retry_all_clears_outputs():
    async def s1(ctx):
        return "v1"

    async def s2(ctx):
        raise RuntimeError("boom")

    stages = [StageDescriptor("s1", s1, "o1"), StageDescriptor("s2", s2, "o2")]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    await jm.run_job(job)
    assert job.context.outputs == {"o1": "v1"}

    jm.control(job.job_id, "retry_all")
    await asyncio.sleep(0.05)
    # s2 still fails, but outputs were cleared and s1 recomputed.
    assert job.status == JobStatus.STAGE_FAILED
    assert job.context.outputs == {"o1": "v1"}


# --- cancel ----------------------------------------------------------------


async def test_cancel_all_mid_stage_is_fast():
    started = asyncio.Event()

    async def slow(ctx):
        started.set()
        await asyncio.sleep(100)  # would hang without cancel

    stages = [StageDescriptor("slow", slow, "s")]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    task = asyncio.ensure_future(jm.run_job(job))

    await asyncio.wait_for(started.wait(), timeout=1)
    jm.control(job.job_id, "cancel_all")
    await asyncio.wait_for(task, timeout=1)

    assert job.status == JobStatus.CANCELLED
    # CANCELLED, not SKIPPED: a skip is a human deciding the run proceeds WITHOUT the
    # stage, a cancel is a human stopping work in flight. Reporting one as the other
    # told the operator their cancel was ignored and the pipeline moved on.
    assert job.stage_statuses["slow"] == StageStatus.CANCELLED


async def test_cancel_all_terminates_even_if_stage_swallows_cancel():
    # A stage that ignores CancelledError (like a library that internally retries)
    # must not keep the job hostage: the cancel event terminates the job at once.
    started = asyncio.Event()

    async def stubborn(ctx):
        started.set()
        while True:
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                pass  # swallow — refuse to die

    stages = [StageDescriptor("stubborn", stubborn, "s")]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    task = asyncio.ensure_future(jm.run_job(job))

    await asyncio.wait_for(started.wait(), timeout=1)
    jm.control(job.job_id, "cancel_all")
    # run_job returns promptly despite the orphaned, uncancellable stage task.
    await asyncio.wait_for(task, timeout=1)
    assert job.status == JobStatus.CANCELLED


async def test_cancel_stage_stops_the_stage_and_leaves_the_job_resumable():
    """cancel_stage is not a synonym for cancel_all.

    It used to be: both flags reached the same _mark_cancelled, so cancelling ONE stage
    terminated the whole run. The distinction is the point of having two controls — this
    one ends the work in flight and hands the run back to the operator, who can then
    retry that stage or resume past it.
    """
    started = asyncio.Event()

    async def slow(ctx):
        started.set()
        await asyncio.sleep(100)

    async def after(ctx):
        return "ran"

    stages = [StageDescriptor("slow", slow, "s"), StageDescriptor("after", after, "a")]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    task = asyncio.ensure_future(jm.run_job(job))

    await asyncio.wait_for(started.wait(), timeout=1)
    jm.control(job.job_id, "cancel_stage")
    await asyncio.wait_for(task, timeout=1)

    assert job.status == JobStatus.PAUSED
    assert job.stage_statuses["slow"] == StageStatus.CANCELLED
    assert job.stage_statuses["after"] == StageStatus.PENDING
    # The cancel flag is consumed by the cancel it served: left set, it would kill the
    # next stage the operator starts.
    assert job._cancel_stage_flag is False


async def test_cancel_stage_then_resume_continues_the_run():
    """A stage-cancelled job has no loop, so `resume` can only set an event nobody is
    waiting on. It has to START one — otherwise the console reports RUNNING with nothing
    running, which is what made pause/resume look broken."""
    started = asyncio.Event()
    ran = []

    async def slow(ctx):
        started.set()
        await asyncio.sleep(100)

    async def after(ctx):
        ran.append("after")
        return "a"

    stages = [StageDescriptor("slow", slow, "s"), StageDescriptor("after", after, "a")]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    task = asyncio.ensure_future(jm.run_job(job))
    await asyncio.wait_for(started.wait(), timeout=1)
    jm.control(job.job_id, "cancel_stage")
    await asyncio.wait_for(task, timeout=1)

    # Skip past the cancelled stage; the rest of the pipeline still runs.
    jm.control(job.job_id, "skip_stage", "slow")
    for _ in range(40):
        if job.status == JobStatus.COMPLETED:
            break
        await asyncio.sleep(0.02)
    assert ran == ["after"]
    assert job.status == JobStatus.COMPLETED


async def test_retry_stage_works_on_a_stage_that_did_not_fail():
    """retry_stage used to require STAGE_FAILED, so the only way to re-run a stage that
    succeeded with the WRONG answer was retry_all — discarding every other stage's work
    with it."""
    calls = []

    async def s1(ctx):
        calls.append("s1")
        return "v1"

    async def s2(ctx):
        calls.append("s2")
        return "v2"

    stages = [StageDescriptor("s1", s1, "o1"), StageDescriptor("s2", s2, "o2")]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    await jm.run_job(job)
    assert calls == ["s1", "s2"]
    assert job.status == JobStatus.COMPLETED

    jm.control(job.job_id, "retry_stage", "s1")
    for _ in range(40):
        if calls.count("s1") == 2:
            break
        await asyncio.sleep(0.02)
    # Re-runs from that stage onward, and is recorded as an analyst action.
    assert calls == ["s1", "s2", "s1", "s2"]
    assert any(i["action"] == "retry_stage" for i in job.interventions)


async def test_retry_stage_cancels_a_running_stage_first():
    """ "Retry" on a stage that is running means cancel it and run it again — not queue a
    second loop over the same JobContext."""
    starts = []
    release = asyncio.Event()

    async def slow(ctx):
        starts.append(1)
        if len(starts) == 1:
            await asyncio.sleep(100)  # cancelled
        await release.wait()
        return "done"

    stages = [StageDescriptor("slow", slow, "s")]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    task = asyncio.ensure_future(jm.run_job(job))
    for _ in range(50):
        if starts:
            break
        await asyncio.sleep(0.02)

    jm.control(job.job_id, "retry_stage", "slow")
    await asyncio.wait_for(task, timeout=2)  # the first loop unwinds
    for _ in range(60):
        if len(starts) == 2:
            break
        await asyncio.sleep(0.02)
    assert len(starts) == 2, "the stage was not re-run"
    assert job.stage_statuses["slow"] == StageStatus.RUNNING
    release.set()


async def test_retry_all_clears_the_replay_buffer_but_not_the_audit_trail():
    """The replay buffer is the run's transcript. After retry_all it belongs to a run
    that no longer exists, so a subscriber would be handed the OLD terminal job_status
    and break its stream on it — the "jobs panel says running, monitor says cancelled"
    split. The interventions, by contrast, record what a human DID and a re-run does not
    un-do a decision they took."""
    started = asyncio.Event()
    runs = []

    async def s1(ctx):
        runs.append(1)
        started.set()
        if len(runs) == 1:
            await asyncio.sleep(100)  # cancelled
        return "v1"

    stages = [StageDescriptor("s1", s1, "o1")]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    task = asyncio.ensure_future(jm.run_job(job))
    await asyncio.wait_for(started.wait(), timeout=1)

    jm.control(job.job_id, "cancel_all")  # recorded as an intervention
    await asyncio.wait_for(task, timeout=2)
    assert any(
        e["type"] == "job_status" and e.get("status") == "cancelled"
        for e in job._event_history
    )

    jm.control(job.job_id, "retry_all")
    # Nothing from the previous run survives in the replay buffer...
    assert not [
        e
        for e in job._event_history
        if e["type"] == "job_status" and e.get("status") == "cancelled"
    ]
    assert any(e["type"] == "run_reset" for e in job._event_history)
    # ...but the operator's actions do.
    actions = [i["action"] for i in job.interventions]
    assert "cancel_all" in actions and "retry_all" in actions
    for _ in range(40):
        if job.status == JobStatus.COMPLETED:
            break
        await asyncio.sleep(0.02)
    assert job.status == JobStatus.COMPLETED


async def test_a_retired_loop_does_not_publish_over_the_current_one():
    """Job 9b3f9631: cancel all → retry all → cancel all, and the jobs panel still read
    "running". The first loop was still unwinding and emitted its own "Job running" after
    the second cancel had landed."""
    started = asyncio.Event()

    async def stubborn(ctx):
        started.set()
        while True:
            try:
                await asyncio.sleep(0.02)
            except asyncio.CancelledError:
                pass  # swallow, like a library that internally retries

    stages = [StageDescriptor("stubborn", stubborn, "s")]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    task = asyncio.ensure_future(jm.run_job(job))
    await asyncio.wait_for(started.wait(), timeout=1)

    jm.control(job.job_id, "cancel_all")
    await asyncio.wait_for(task, timeout=2)
    assert job.status == JobStatus.CANCELLED

    jm.control(job.job_id, "retry_all")
    await asyncio.sleep(0.05)
    jm.control(job.job_id, "cancel_all")
    await asyncio.sleep(0.2)
    assert job.status == JobStatus.CANCELLED, "a retired loop revived the job"


# --- pause / step ----------------------------------------------------------


async def test_pause_halts_at_boundary():
    ran = []

    async def s1(ctx):
        ran.append("s1")

    async def s2(ctx):
        ran.append("s2")

    stages = [StageDescriptor("s1", s1), StageDescriptor("s2", s2)]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    # Pause before starting: the loop should block at the first boundary.
    jm.control(job.job_id, "pause")
    task = asyncio.ensure_future(jm.run_job(job))
    await asyncio.sleep(0.05)

    assert ran == []
    assert job.status == JobStatus.PAUSED

    jm.control(job.job_id, "resume")
    await asyncio.wait_for(task, timeout=1)
    assert ran == ["s1", "s2"]
    assert job.status == JobStatus.COMPLETED


async def test_step_advances_one_stage_at_a_time():
    ran = []

    async def s1(ctx):
        ran.append("s1")

    async def s2(ctx):
        ran.append("s2")

    stages = [StageDescriptor("s1", s1), StageDescriptor("s2", s2)]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident(), run_mode=JobRunMode.STEP)
    task = asyncio.ensure_future(jm.run_job(job))
    await asyncio.sleep(0.05)
    assert ran == []  # STEP mode waits for the first step

    jm.control(job.job_id, "step")
    await asyncio.sleep(0.05)
    assert ran == ["s1"]

    jm.control(job.job_id, "step")
    await asyncio.wait_for(task, timeout=1)
    assert ran == ["s1", "s2"]
    assert job.status == JobStatus.COMPLETED


# --- SSE fan-out -----------------------------------------------------------


async def test_multiple_subscribers_each_receive_events():
    async def s1(ctx):
        await asyncio.sleep(0.02)

    stages = [StageDescriptor("s1", s1)]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())

    async def collect():
        events = []
        async for ev in jm.subscribe(job.job_id):
            events.append(ev)
            if ev.get("type") == "job_status" and ev.get("status") == "completed":
                break
        return events

    sub1 = asyncio.ensure_future(collect())
    sub2 = asyncio.ensure_future(collect())
    await asyncio.sleep(0)  # let subscribers register
    await jm.run_job(job)
    events1 = await asyncio.wait_for(sub1, timeout=1)
    events2 = await asyncio.wait_for(sub2, timeout=1)

    types1 = [e["type"] for e in events1]
    assert "stage_started" in types1 and "stage_completed" in types1
    assert types1 == [e["type"] for e in events2]


async def test_late_subscriber_gets_replay():
    async def s1(ctx):
        return "x"

    stages = [StageDescriptor("s1", s1, "o")]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    await jm.run_job(job)  # completes before anyone subscribes

    events = []
    async for ev in jm.subscribe(job.job_id):
        events.append(ev)
        if ev.get("type") == "job_status" and ev.get("status") == "completed":
            break
    assert any(e["type"] == "stage_completed" for e in events)


# --- export / import -------------------------------------------------------


def _understanding():
    return UnderstandingResult(
        incident_id="INC-1",
        analysis=IncidentAnalysis(
            incident_summary="s",
            severity="5",
            severity_reasoning="r",
            impact_assessment="i",
            key_investigation_areas=[],
            log_sources_to_review=[],
            initial_hypotheses=[],
            recommended_actions=[],
            stakeholder_notification=[],
        ),
    )


async def test_export_import_round_trips_outputs_to_models():
    async def s1(ctx):
        return _understanding()

    stages = [StageDescriptor("understanding", s1, "understanding")]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    # Seed richer outputs to exercise the codecs.
    await jm.run_job(job)
    job.context.outputs["queries"] = [
        RetrievalQuery(
            target_log_source="src",
            natural_language_query="q",
            date_from="2024-01-01",
            date_to="2024-01-02",
        )
    ]
    job.context.outputs["correlation"] = CorrelationResult(
        record_count=3, summary_text="c"
    )
    job.context.outputs["anomalies"] = [
        AnomalyItem(
            description="d",
            supporting_data="s",
            potential_implications="p",
            confidence_score=0.5,
            recommended_actions="a",
            patterns="pt",
        )
    ]
    job.context.outputs["report"] = "final report"

    doc = jm.export_job(job.job_id)
    assert doc["status"] == "completed"
    assert isinstance(doc["outputs"]["understanding"], dict)  # serialized

    reloaded = jm.import_job(doc)
    assert reloaded.job_id == job.job_id
    # Duck-type, not isinstance: pipeline_runner uses flat imports (models.*) while
    # this test imports src.models.* — two class identities for the same model
    # (documented dual-import caveat). Assert the rehydrated fields instead.
    u = reloaded.context.outputs["understanding"]
    assert u.incident_id == "INC-1" and u.analysis.severity == "5"
    q = reloaded.context.outputs["queries"][0]
    assert q.target_log_source == "src" and q.natural_language_query == "q"
    corr = reloaded.context.outputs["correlation"]
    assert corr.record_count == 3 and corr.summary_text == "c"
    anomaly = reloaded.context.outputs["anomalies"][0]
    assert anomaly.description == "d" and anomaly.confidence_score == 0.5
    assert reloaded.context.outputs["report"] == "final report"
    assert reloaded.stage_statuses["understanding"] == StageStatus.COMPLETED


# --- evidence artifacts export ---------------------------------------------


def test_run_export_writes_evidence_artifacts(tmp_path, monkeypatch):
    """_run_export writes the raw + transformed evidence files and returns 4 paths;
    it reads logs/correlation from ctx.outputs without KeyError when absent."""
    import json

    import src.pipeline_runner as pr
    from src.pipeline_runner import JobContext

    monkeypatch.setattr(pr, "exports_dir", lambda: tmp_path)

    ctx = JobContext(
        job_id="J1",
        incident={"id": "INC-1", "timestamp": "2024-01-01T00:00", "description": "d"},
        modules={},
        config={},
        outputs={
            "understanding": _understanding(),
            "anomalies": [],
            "logs": {"app": [{"record": "P1"}, {"record": "P2"}]},
            "correlation": CorrelationResult(record_count=2, summary_text="c"),
        },
    )
    paths = pr._run_export(ctx)
    assert len(paths) == 4
    raw = tmp_path / "evidence_raw_INC-1.json"
    transformed = tmp_path / "evidence_transformed_INC-1.json"
    assert raw.exists() and transformed.exists()
    assert str(raw) in paths and str(transformed) in paths
    assert len(json.loads(raw.read_text())["app"]) == 2
    assert json.loads(transformed.read_text())["record_count"] == 2


def test_run_export_no_correlation_or_logs(tmp_path, monkeypatch):
    """Absent logs/correlation must not raise; the transformed file notes the gap."""
    import json

    import src.pipeline_runner as pr
    from src.pipeline_runner import JobContext

    monkeypatch.setattr(pr, "exports_dir", lambda: tmp_path)

    ctx = JobContext(
        job_id="J1",
        incident={"id": "INC-2", "timestamp": "2024-01-01T00:00", "description": "d"},
        modules={},
        config={},
        outputs={"understanding": _understanding(), "anomalies": []},
    )
    paths = pr._run_export(ctx)
    assert len(paths) == 4
    assert json.loads((tmp_path / "evidence_raw_INC-2.json").read_text()) == {}
    assert "note" in json.loads(
        (tmp_path / "evidence_transformed_INC-2.json").read_text()
    )


def _q(source):
    """A minimally valid `RetrievalQuery` — only its target matters to the dependency report."""
    return RetrievalQuery(
        target_log_source=source,
        natural_language_query="q",
        date_from="2024-01-01",
        date_to="2024-01-02",
    )


def _dep_ctx(report, unparseable=(), **over):
    """A ctx whose `api_call` module answers `dependency_report` with `report`."""
    from src.pipeline_runner import JobContext

    gen = MagicMock()
    gen.dependency_report.return_value = dict(report)
    gen.selected_unparseable = list(unparseable)
    return (
        gen,
        JobContext(
            job_id="J1",
            incident=_incident(),
            modules={"api_call": gen},
            config={},
            outputs={"understanding": _understanding()},
            **over,
        ),
    )


def test_plan_dependency_findings_are_recorded_on_the_run_not_the_module():
    """The shared generator's attributes are the LAST plan's; the run needs its own copy.

    `modules["api_call"]` is one object for the whole process, so a finding read off it is
    whichever job planned most recently. `_record_dependency_findings` pins this run's answer to
    `stage_facts`, which is per-JobContext — so what is asserted here is that the recorded copy
    is complete (all four keys the scorer reads) and comes from THIS plan's queries.
    """
    import src.pipeline_runner as pr

    gen, ctx = _dep_ctx(
        {
            "undeliverable": ["no_creds"],
            "not_queried": ["never_asked"],
            "unscopable": ["nothing_to_scope"],
        },
        unparseable=["lost_pick"],
    )
    plan = [_q("src_a")]
    pr._record_dependency_findings(ctx, 1, plan)

    facts = ctx.stage_facts["query_generation"]
    assert facts == {
        "undeliverable_required": ["no_creds"],
        "declared_not_queried": ["never_asked"],
        "declared_unscopable": ["nothing_to_scope"],
        "selected_unparseable": ["lost_pick"],
    }
    # Computed FROM the plan, not from generator state — the report is a pure function and this
    # is the seam that supplies its arguments.
    assert gen.dependency_report.call_args[0][0] == plan


def test_a_follow_up_pass_retracts_a_dependency_it_met_and_keeps_the_lost_pick():
    """A pass ADDS, so a met dependency stops being a finding and an unparseable pick does not.

    The two halves differ because only one is recomputable. `dependency_report` is a function of
    the accumulated plan, so a pass-2 query that finally targets a declared source has MET it and
    the finding must go — a stale one reports a hole in a plan that no longer has it.
    `selected_unparseable` is a side effect of parsing a model's tool calls, which only the
    planning pass does, so pass 2 has nothing to say about it and silence must not clear it.
    """
    import src.pipeline_runner as pr

    gen, ctx = _dep_ctx(
        {"undeliverable": [], "not_queried": ["late_source"], "unscopable": []},
        unparseable=["lost_pick"],
    )
    pr._record_dependency_findings(ctx, 1, [_q("a")])
    assert ctx.stage_facts["query_generation"]["declared_not_queried"] == ["late_source"]

    # Pass 2 targets it, and the generator no longer carries the pass-1 side effect.
    gen.dependency_report.return_value = {
        "undeliverable": [],
        "not_queried": [],
        "unscopable": [],
    }
    gen.selected_unparseable = []
    pr._record_dependency_findings(
        ctx,
        2,
        [
            _q("a"),
            _q("late_source"),
        ],
    )
    facts = ctx.stage_facts["query_generation"]
    assert facts["declared_not_queried"] == []          # retracted: the pass met it
    assert facts["selected_unparseable"] == ["lost_pick"]  # carried: pass 2 cannot recompute it


def test_recording_a_dependency_finding_never_fails_the_stage():
    """A scoring input that cannot be recorded degrades to the old read; it does not raise."""
    import src.pipeline_runner as pr
    from src.pipeline_runner import JobContext

    gen = MagicMock()
    gen.dependency_report.side_effect = RuntimeError("boom")
    ctx = JobContext(
        job_id="J1", incident=_incident(), modules={"api_call": gen}, config={}
    )
    pr._record_dependency_findings(ctx, 1, [])
    assert "query_generation" not in ctx.stage_facts

    # And no generator at all is a no-op, not an AttributeError.
    bare = JobContext(job_id="J2", incident=_incident(), modules={}, config={})
    pr._record_dependency_findings(bare, 1, [])
    assert bare.stage_facts == {}


# --- control validation ----------------------------------------------------


def test_control_unknown_job_raises_keyerror():
    jm = _emitter_with_manager([StageDescriptor("s", lambda ctx: None)])
    with pytest.raises(KeyError):
        jm.control("nope", "pause")


async def test_retry_stage_invalid_state_raises():
    async def s1(ctx):
        return 1

    jm = _emitter_with_manager([StageDescriptor("s1", s1, "o")])
    job = jm.create_job(_incident())
    await jm.run_job(job)  # COMPLETED, not STAGE_FAILED
    with pytest.raises(ValueError):
        jm.control(job.job_id, "retry_stage")


# --- stage_output events, timing, and summarizers --------------------------


def _collect_events(job):
    """All events emitted for a job (replay history is the full record)."""
    return list(job._event_history)


async def test_stage_completed_carries_duration_ms():
    async def s1(ctx):
        return 1

    jm = _emitter_with_manager([StageDescriptor("understanding", s1, "understanding")])
    job = jm.create_job(_incident())
    await jm.run_job(job)

    completed = [e for e in _collect_events(job) if e["type"] == "stage_completed"]
    assert completed
    assert all("duration_ms" in e["data"] for e in completed)
    assert all(isinstance(e["data"]["duration_ms"], int) for e in completed)


async def test_stage_output_event_emitted_with_summary():
    async def s1(ctx):
        return 42

    # unknown stage name -> summarizer returns {} but the event still fires
    jm = _emitter_with_manager([StageDescriptor("mystery", s1, "m")])
    job = jm.create_job(_incident())
    await jm.run_job(job)

    outputs = [e for e in _collect_events(job) if e["type"] == "stage_output"]
    assert len(outputs) == 1
    assert outputs[0]["stage"] == "mystery"
    assert "summary" in outputs[0]["data"]


async def test_stage_failed_carries_duration_ms():
    async def boom(ctx):
        raise RuntimeError("nope")

    jm = _emitter_with_manager([StageDescriptor("understanding", boom, "u")])
    job = jm.create_job(_incident())
    await jm.run_job(job)

    failed = [e for e in _collect_events(job) if e["type"] == "stage_failed"]
    assert failed and "duration_ms" in failed[0]["data"]


# --- summarize_stage: defensive + shape ------------------------------------


def test_summarize_understanding_extracts_key_fields():
    from src.pipeline_runner import summarize_stage

    analysis = IncidentAnalysis(
        incident_summary="user 4821 compromised",
        severity="8",
        severity_reasoning="high value transfer",
        impact_assessment="funds at risk",
        key_investigation_areas=["auth"],
        log_sources_to_review=["auth_events", "svc_alerts"],
        initial_hypotheses=["ATO"],
        recommended_actions=["lock account"],
        stakeholder_notification=["SOC"],
        correlation_keys=["user", "session"],
    )
    result = UnderstandingResult(incident_id="INC-1", analysis=analysis)
    s = summarize_stage("understanding", result)
    assert s["severity"] == "8"
    assert s["correlation_keys"] == ["user", "session"]
    assert "auth_events" in s["log_sources_to_review"]


def test_a_stages_own_prose_is_not_clipped_to_a_label_length():
    """One cap was serving two purposes, and prose lost.

    `_SUM_STR` (300) is right for an id, a source name, a section title. It is wrong for
    an analytical paragraph: on IR10000004 the understanding summary and severity
    reasoning were cut mid-word at exactly 300 chars, and the operator read that as the
    analysis being truncated rather than its display. Prose the human reads gets
    `_PROSE_STR`; labels keep the tight cap (bounded either way — the replay buffer must
    not bloat)."""
    from src.pipeline_runner import _PROSE_STR, _SUM_STR, summarize_stage

    assert _PROSE_STR > _SUM_STR  # the two caps must stay distinct
    long_prose = "A SCHEME issuance fraud alert fired. " * 40  # ~1400 chars
    assert len(long_prose) > _SUM_STR
    analysis = IncidentAnalysis(
        incident_summary=long_prose,
        severity="HIGH",
        severity_reasoning=long_prose,
        impact_assessment=long_prose,
        key_investigation_areas=["auth"],
        log_sources_to_review=["auth_events"],
        initial_hypotheses=[long_prose],
        recommended_actions=[long_prose],
        stakeholder_notification=["SOC"],
        correlation_keys=["user"],
    )
    s = summarize_stage(
        "understanding", UnderstandingResult(incident_id="INC-1", analysis=analysis)
    )
    for field in ("summary", "severity_reasoning", "impact_assessment"):
        assert s[field] == long_prose, f"{field} was clipped"
    assert s["initial_hypotheses"][0] == long_prose
    assert s["recommended_actions"][0] == long_prose
    # Still bounded: past the prose cap it clips with the ellipsis marker.
    huge = "x" * (_PROSE_STR + 50)
    analysis.incident_summary = huge
    s2 = summarize_stage(
        "understanding", UnderstandingResult(incident_id="INC-2", analysis=analysis)
    )
    assert s2["summary"].endswith("…") and len(s2["summary"]) == _PROSE_STR + 1


def test_a_backend_query_is_not_clipped_to_a_display_snippet():
    """A statement is the one string that is useless partially.

    Prose clipped at 4000 still carries its argument; SQL clipped anywhere loses whichever
    clause came after the cut — and on these sources the cut lands in the worst place,
    because the projection is generated from the discovered schema. Measured on job
    30887c6b at the old 2000-char cap: `auth_events` spent ~1900 chars on 32
    `value.payload…AS …` lines, so the visible text ended at
    `AND ( value.payload.userInfo.login = '…` — every filter, every partition bound and
    every guard rewrite hidden, which is precisely what the operator opens it to check.
    """
    from src.pipeline_runner import _PROSE_STR, _QUERY_STR, _clip

    # Must clear a real generated statement, not just a label or a paragraph.
    assert _QUERY_STR > _PROSE_STR
    projection = ",\n  ".join(
        f"value.payload.userInfo.f{i} AS value_payload_userInfo_f{i}" for i in range(32)
    )
    sql = (
        f"SELECT\n  {projection}\nFROM cat.sch.auth_raw_events\n"
        "WHERE date >= DATE'2026-07-03' AND date <= DATE'2026-07-05'\n"
        "  AND ((value.payload.userInfo.login = 'BSURNAME'"
        " OR value.payload.userInfo.userId = 'BSURNAME')"
        " AND value.payload.userInfo.sign = '6009JJ'"
        " AND value.payload.userInfo.org_unit = 'HHH1J09ST')"
    )
    assert len(sql) > 2000, "fixture must exceed the old cap to regress it"
    shown = _clip(sql, _QUERY_STR)
    assert shown == sql
    # The clauses that used to fall off the end.
    for clause in (
        "WHERE date >=",
        "userInfo.sign = '6009JJ'",
        "org_unit = 'HHH1J09ST'",
    ):
        assert clause in shown
    # Still bounded — the replay buffer must not carry an unbounded string.
    assert _clip("x" * (_QUERY_STR + 10), _QUERY_STR).endswith("…")


def test_the_query_a_gate_asks_about_is_shown_in_full():
    """The `query_generation` gate must not ask for approval of a clipped request.

    `natural_language_query` was summarized at the 300-char LABEL cap; a real one measured
    268 chars, i.e. already at the edge, so the next one is cut and the operator signs off
    on a sentence whose end they cannot see.
    """
    from src.pipeline_runner import _PROSE_STR, _SUM_STR, summarize_stage

    request = (
        "Retrieve SCHEME alert records for org_unit HHH1J09ST on 2026-07-04, including IR "
        "title, recordId, severity, status, alert type, and the email/SMS notification "
        "lifecycle. " * 3
    )
    assert len(request) > _SUM_STR
    q = RetrievalQuery(
        target_log_source="siem_alerts_current",
        natural_language_query=request,
        date_from="2026-07-04",
        date_to="2026-07-05",
    )
    s = summarize_stage("query_generation", [q])
    assert s["queries"][0]["query"] == request, "the gate showed a clipped request"
    assert len(request) < _PROSE_STR  # still bounded by the prose cap


def test_summarize_logs_counts_rows_and_samples():
    from src.pipeline_runner import summarize_stage

    logs = {"esql": [{"a": 1}, {"a": 2}], "dbx": [{"b": 3}]}
    s = summarize_stage("log_retrieval", logs)
    assert s["total_rows"] == 3
    assert s["source_count"] == 2
    # sorted by row count desc
    assert s["sources"][0]["source"] == "esql"
    assert s["sources"][0]["rows"] == 2
    assert len(s["sources"][0]["samples"]) <= 3


def test_summarize_correlation_surfaces_resolved_keys():
    from src.pipeline_runner import summarize_stage

    corr = CorrelationResult(
        aggregations={
            "resolved_correlation_keys": [
                {
                    "entity_hint": "user",
                    "sources": {"a": "userId"},
                    "origin": "playbook",
                }
            ],
            "discovered_join_keys": [],
            "record_count_by_source": {"a": 5},
        },
        record_count=5,
    )
    s = summarize_stage("correlation", corr)
    assert s["record_count"] == 5
    assert s["resolved_correlation_keys"][0]["entity_hint"] == "user"
    assert s["resolved_correlation_keys"][0]["origin"] == "playbook"
    assert s["evidence"] is None  # no evidence pack -> defensively None


def test_summarize_correlation_surfaces_evidence():
    from src.models.pydantic_models import (ActorRollup, ChronologyEvent,
                                            EvidencePack)
    from src.pipeline_runner import summarize_stage

    corr = CorrelationResult(
        aggregations={"resolved_correlation_keys": [], "discovered_join_keys": []},
        record_count=100,
        evidence=EvidencePack(
            chronology=[ChronologyEvent(source="a", actor="USERNAMEX")],
            chronology_aggregated=True,
            actors=[ActorRollup(actor="USERNAMEX", event_count=5)],
            cross_source_joins=[{"value": "SUBJ03", "sources": ["a", "b"]}],
            total_records=100,
            degraded=True,
        ),
    )
    s = summarize_stage("correlation", corr)
    ev = s["evidence"]
    assert ev["chronology_events"] == 1
    assert ev["chronology_aggregated"] is True
    assert ev["actor_count"] == 1
    assert ev["top_actors"] == ["USERNAMEX"]
    assert ev["cross_source_joins"] == 1
    assert ev["degraded"] is True


def test_summarize_correlation_surfaces_brief():
    from src.models.pydantic_models import (ConditionCheck, InvestigationBrief,
                                            SubjectVerdict, ValidationVerdict)
    from src.pipeline_runner import summarize_stage

    verdict = ValidationVerdict(
        summary="FALSE POSITIVE: 1",
        subjects=[
            SubjectVerdict(
                subject_type="record", subject_value="SUBJ03", verdict="FALSE POSITIVE"
            )
        ],
    )
    corr = CorrelationResult(
        aggregations={"resolved_correlation_keys": [], "discovered_join_keys": []},
        record_count=5,
        brief=InvestigationBrief(
            use_case="scheme",
            playbook_id="PB-APP-SCHEME-001",
            verdict=verdict,
            decisive_fails=[ConditionCheck(id="bare", result="fail", decisive=True)],
            join_status={"join_record": "ran, 2 match(es)"},
            degraded=True,
        ),
    )
    s = summarize_stage("correlation", corr)
    bv = s["brief"]
    assert bv is not None
    assert bv["use_case"] == "scheme"
    assert bv["verdict_summary"] == "FALSE POSITIVE: 1"
    assert bv["decisive_fails"] == ["bare"]
    assert bv["join_status"] == {"join_record": "ran, 2 match(es)"}
    assert bv["degraded"] is True
    # No brief -> defensively None.
    assert summarize_stage("correlation", CorrelationResult())["brief"] is None


def test_summarize_correlation_carries_the_link_STATE_verbatim():
    """The four link states are the whole artifact, so the summary may not fold them.

    A boolean "linked / not linked" is the collapse the lane exists to prevent: `not_probed`
    ("we did not look") and `probed_negative` ("we looked and it does not apply") license
    OPPOSITE next steps, and the second is a finding rather than a silence. The card is the
    only surface an operator reads before deciding whether to refer, and it renders from this
    summary and never from the stage output.
    """
    from src.links import LINK_STATES
    from src.models.pydantic_models import LinkFinding
    from src.pipeline_runner import summarize_stage

    corr = CorrelationResult(
        aggregations={"resolved_correlation_keys": [], "discovered_join_keys": []},
        record_count=5,
        links=[
            LinkFinding(
                target_use_case="sibling",
                direction="antecedent",
                state=state,
                rung=2,
                pivot_entity="account",
                pivot_values=["A1"],
            )
            for state in LINK_STATES
        ],
    )
    s = summarize_stage("correlation", corr)
    assert [f["state"] for f in s["links"]] == list(LINK_STATES)
    assert s["link_count"] == len(LINK_STATES)
    assert s["links"][0]["pivot_entity"] == "account"
    assert s["links"][0]["pivot_values"] == ["A1"]
    assert s["links"][0]["rung"] == 2
    # No links -> an empty list and a zero, never a missing key: the renderer branches on
    # the array, and a pack declaring nothing is the normal case.
    empty = summarize_stage("correlation", CorrelationResult())
    assert empty["links"] == [] and empty["link_count"] == 0


def test_the_link_card_carries_rung_1s_OUTCOME_and_not_only_its_boolean():
    """`mode_licensed` says WHETHER the target's gate held; it cannot say which way it did not.

    Five outcomes arrive as one `False` — `fail`, `unknown`, `no_gate`, `no_sources`,
    `no_subject` — and "its own applicability test says this is not that procedure" and "the
    rows to decide it were never retrieved in this run" license opposite next steps: the first
    is a finding, the second a retrieval gap. `mode_note` carries a reason only where a mode was
    CLAMPED, so a candidate nobody asked to escalate had an empty note and the distinction
    reached no surface an operator reads.
    """
    from src.links import _gate_outcome  # noqa: F401 — the vocabulary's owner
    from src.models.pydantic_models import LinkFinding
    from src.pipeline_runner import summarize_stage

    outcomes = ["pass", "fail", "unknown", "no_gate", "no_sources", "no_subject"]
    corr = CorrelationResult(
        aggregations={"resolved_correlation_keys": [], "discovered_join_keys": []},
        record_count=5,
        links=[
            LinkFinding(
                target_use_case=f"uc{i}",
                state="probed_negative",
                gate_outcome=outcome,
                # Every non-`pass` one is unlicensed, which is exactly the collapse: without
                # `gate_outcome` these five rows are indistinguishable on the card.
                mode_licensed=(outcome == "pass"),
            )
            for i, outcome in enumerate(outcomes)
        ],
    )
    s = summarize_stage("correlation", corr)
    assert [f["gate_outcome"] for f in s["links"]] == outcomes
    assert [f["mode_licensed"] for f in s["links"]] == [
        True,
        False,
        False,
        False,
        False,
        False,
    ], "the boolean is the licence and stays; the outcome is what it cannot express"
    # A gate never reached is empty, not one of the six: "we did not get there" is a seventh
    # fact, and spelling it `unknown` would claim the gate ran and could not decide.
    unreached = summarize_stage(
        "correlation",
        CorrelationResult(links=[LinkFinding(target_use_case="uc", state="not_probed")]),
    )
    assert unreached["links"][0]["gate_outcome"] == ""


def test_a_bounded_link_list_reports_its_TRUE_total():
    """`_counted` bounds the list; the count beside it must not be its length.

    A candidate an operator never sees is a candidate nobody refers, and a list that stops is
    shaped exactly like a list that ended. Same rule as every other bounded summary field.
    """
    from src.models.pydantic_models import LinkFinding
    from src.pipeline_runner import _SUM_MAX_ITEMS, summarize_stage

    n = _SUM_MAX_ITEMS + 5
    corr = CorrelationResult(
        aggregations={"resolved_correlation_keys": [], "discovered_join_keys": []},
        links=[
            LinkFinding(target_use_case=f"uc{i}", state="not_probed") for i in range(n)
        ],
    )
    s = summarize_stage("correlation", corr)
    assert len(s["links"]) == _SUM_MAX_ITEMS
    assert s["link_count"] == n, "the count describes the display, not the assessment"


def test_the_link_SCORE_reaches_the_card_as_a_NUMBER_with_its_terms():
    """The card badges the score and shows the terms behind it, so both must be echoed.

    The score is what gates `semi_auto`, and the card is where an operator sees why a candidate
    only proposed. A number reaching the page as a string renders as `score NaN` off
    `toFixed`, so the summarizer answers with a real float — clamped, because the page trusts
    this value's range rather than re-checking it.
    """
    from types import SimpleNamespace

    from src.models.pydantic_models import LinkFinding
    from src.pipeline_runner import summarize_stage

    corr = CorrelationResult(
        aggregations={"resolved_correlation_keys": [], "discovered_join_keys": []},
        links=[
            LinkFinding(
                target_use_case="sibling",
                state="probed_positive",
                link_score=0.9778,
                link_score_reasons=[f"term {i} (+0.10)" for i in range(9)],
            )
        ],
    )
    s = summarize_stage("correlation", corr)
    row = s["links"][0]
    assert row["link_score"] == 0.9778
    assert isinstance(row["link_score"], float)
    # Bounded like every other list on this summary, and the terms are the tooltip's whole
    # content — an unbounded one would be the largest field on the page.
    assert row["link_score_reasons"][0] == "term 0 (+0.10)"
    assert len(row["link_score_reasons"]) == 6

    # A finding whose score never landed as a number reads 0.0 rather than reaching the page:
    # the badge is the operator's only view of the gate, and `score NaN` beside a candidate
    # held at a referral is worse than a conservative zero.
    corr.links = [
        SimpleNamespace(
            target_use_case="sibling",
            state="not_probed",
            link_score="not a number",
            link_score_reasons="not a list",
        )
    ]
    row = summarize_stage("correlation", corr)["links"][0]
    assert row["link_score"] == 0.0
    assert row["link_score_reasons"] == []


async def test_a_rung_4_CHILD_reaches_the_SUMMARY_and_not_only_the_result():
    """The spawn runs after the summary was built, so the summary has to be rebuilt.

    Rung 4 stamps `child_job_id` on the finding, and it necessarily runs LATE — after the
    approval gate, because a rejected gate re-runs correlation and spawning first would launch
    a full run off an output a human is about to reject. But the summary was frozen before the
    gate, and the summary is what every reader actually reads: the job document, the replayed
    `stage_output`, the link card's own button. So the id stayed empty on findings whose child
    was demonstrably running, and rung 4's only surface could not reach the run it launched.

    The spawn itself is planned and bounded in `test_link_children.py`, which needs no loop;
    what only a runner test can assert is that the mutation it makes is VISIBLE afterwards.
    Hence the stub: it stamps what the real one stamps and returns the same count.
    """
    from src.models.pydantic_models import LinkFinding

    corr = CorrelationResult(
        aggregations={"resolved_correlation_keys": [], "discovered_join_keys": []},
        record_count=3,
        links=[
            LinkFinding(
                target_use_case="sibling",
                state="probed_positive",
                gate_outcome="pass",
                mode="auto",
                rung=4,
            )
        ],
    )

    async def correlation(ctx):
        return corr

    jm = _emitter_with_manager(
        [StageDescriptor("correlation", correlation, "correlation")]
    )

    async def spawn(job, result):
        result.links[0].child_job_id = "child-1"
        result.links[0].child_note = "a full run was launched as job child-1"
        return 1

    jm._spawn_link_children = spawn
    job = jm.create_job(_incident(), run_mode=JobRunMode.AUTO)
    await jm.run_job(job)

    row = job.stage_summaries["correlation"]["links"][0]
    assert row["child_job_id"] == "child-1"
    assert "child-1" in row["child_note"]

    # And on the replay, which is what fills the page for a job that finished before anyone
    # attached. The event is keyed by stage, so the later copy overwrites the earlier one on
    # the client rather than appending — but only if it is emitted at all.
    outputs = [
        e
        for e in job._event_history
        if e["type"] == "stage_output" and e.get("stage") == "correlation"
    ]
    assert len(outputs) == 2, "the post-spawn summary must be re-emitted, not only stored"
    assert outputs[0]["data"]["summary"]["links"][0]["child_job_id"] == ""
    assert outputs[-1]["data"]["summary"]["links"][0]["child_job_id"] == "child-1"
    # Health is settled before the gate the operator answered, and the advisory lane may not
    # move it — so the re-emit carries the SAME score rather than a re-scored one.
    assert outputs[-1]["data"]["health"] == outputs[0]["data"]["health"]


async def test_a_run_that_spawned_NOTHING_emits_the_stage_output_once():
    """A pack declaring no link, and every run whose candidates only proposed, is untouched.

    The re-summarise is conditional on a launch for exactly this reason: a second identical
    `stage_output` on every correlation stage would be a duplicate row in the console for the
    overwhelmingly common case, and a byte-identical run is what makes the advisory lane
    auditable at all.
    """

    async def correlation(ctx):
        return CorrelationResult(record_count=1, summary_text="c")

    jm = _emitter_with_manager(
        [StageDescriptor("correlation", correlation, "correlation")]
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.AUTO)
    await jm.run_job(job)

    outputs = [
        e
        for e in job._event_history
        if e["type"] == "stage_output" and e.get("stage") == "correlation"
    ]
    assert len(outputs) == 1
    assert job.stage_summaries["correlation"]["links"] == []


async def test_a_rung_4_REFUSAL_reaches_the_SUMMARY_the_way_a_LAUNCH_does():
    """A bound that held is a row, and the row is the summary, not a log line.

    Four of the nine codes are reconstructible from nothing else on the candidate:
    `unpinnable`, `depth_cap`, `cycle`, `run_budget`. The refusal must appear in the
    summary the same way a launch does.

    Reached through the real planner, not a stub: the summary is rebuilt only when the
    rung reports it changed `result`, and keyed on the launch count that condition was
    false on exactly the runs this test is about.
    """
    from src.models.pydantic_models import LinkFinding

    corr = CorrelationResult(
        aggregations={"resolved_correlation_keys": [], "discovered_join_keys": []},
        record_count=3,
        links=[
            LinkFinding(
                target_use_case="sibling",
                state="probed_positive",
                gate_outcome="pass",
                mode="auto",
                pivot_entity="shipment",
                pivot_values=["SHP-1"],
                rung=3,
            )
        ],
    )

    async def correlation(ctx):
        return corr

    jm = _emitter_with_manager(
        [StageDescriptor("correlation", correlation, "correlation")],
        config={"correlation": {"links": {"max_children_per_run": 2}}},
    )
    # A run that is ITSELF a referral, which is the whole of the bound: two chain entries is one
    # hop, and the default cap is one.
    incident = dict(_incident())
    incident["link_chain"] = [
        {"use_case": "the_reported_procedure", "pivot": ""},
        {"use_case": "somewhere", "pivot": "SHP-9"},
    ]
    job = jm.create_job(incident, run_mode=JobRunMode.AUTO)
    await jm.run_job(job)

    row = job.stage_summaries["correlation"]["links"][0]
    # Nothing was launched, and the row says which bound held rather than saying nothing.
    assert row["child_job_id"] == ""
    assert row["child_note"], "a refused candidate carries no reason on the one surface read"
    assert "depth" in row["child_note"]
    # And on the replay, which is what fills the page for a job that finished before anyone
    # attached — the same re-emit a launch gets, for the same reason.
    outputs = [
        e
        for e in job._event_history
        if e["type"] == "stage_output" and e.get("stage") == "correlation"
    ]
    assert (
        len(outputs) == 2
    ), "the post-rung-4 summary is re-emitted only when something was LAUNCHED"
    assert outputs[0]["data"]["summary"]["links"][0]["child_note"] == ""
    assert outputs[-1]["data"]["summary"]["links"][0]["child_note"] == row["child_note"]
    # The advisory lane may not move the score it was settled beside, refusal or launch.
    assert outputs[-1]["data"]["health"] == outputs[0]["data"]["health"]
    # And the state the assessment reached is untouched: rung 4 declining to spend a run is not a
    # re-adjudication of the candidate, and a reader must still see what rungs 0-2 concluded.
    assert corr.links[0].state == "probed_positive"


async def test_a_RUN_scoped_rung_4_refusal_does_not_land_on_the_row_it_rides_on():
    """`children_disabled` is reported once, on the first candidate, and is not ABOUT it.

    `plan_child_spawns` reports the two run-level refusals against `findings[0]` rather than N
    times, because a fact that applies to every candidate buries the per-candidate ones when it is
    repeated. That makes the first row a place to sit and nothing more, so stamping it would tell one arbitrary candidate
    that the whole rung being switched off is a finding about it — and every other candidate in the
    same list, with the identical situation, would show nothing.

    Run with the rung switched off, which is the shape a deployment declaring `0` renders —
    exactly as every run did before rung 4 existed.
    """
    from src.models.pydantic_models import LinkFinding

    def _candidate(pivot):
        return LinkFinding(
            target_use_case="sibling",
            state="probed_positive",
            gate_outcome="pass",
            mode="auto",
            pivot_entity="shipment",
            pivot_values=[pivot],
            rung=3,
        )

    corr = CorrelationResult(
        aggregations={"resolved_correlation_keys": [], "discovered_join_keys": []},
        record_count=3,
        links=[_candidate("SHP-1"), _candidate("SHP-2")],
    )

    async def correlation(ctx):
        return corr

    jm = _emitter_with_manager(
        [StageDescriptor("correlation", correlation, "correlation")],
        config={"correlation": {"links": {"max_children_per_run": 0}}},
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.AUTO)
    await jm.run_job(job)

    for row in job.stage_summaries["correlation"]["links"]:
        assert row["child_job_id"] == ""
        assert row["child_note"] == ""
    outputs = [
        e
        for e in job._event_history
        if e["type"] == "stage_output" and e.get("stage") == "correlation"
    ]
    assert len(outputs) == 1, "a run with the rung off must emit the stage output once"


def test_summarize_anomalies_sorted_by_confidence():
    from src.pipeline_runner import summarize_stage

    anomalies = [
        AnomalyItem(
            description="low",
            supporting_data="",
            potential_implications="",
            confidence_score=0.3,
            recommended_actions="",
            patterns="",
        ),
        AnomalyItem(
            description="high",
            supporting_data="",
            potential_implications="",
            confidence_score=0.9,
            recommended_actions="",
            patterns="",
        ),
    ]
    s = summarize_stage("anomaly_detection", anomalies)
    assert s["count"] == 2
    assert s["anomalies"][0]["description"] == "high"


def test_the_count_and_the_list_a_gate_shows_cannot_disagree():
    """13 queries reported, 12 listed — the live report, and the operator did the arithmetic.

    `count` came from the uncapped `len()` while `queries` was sliced at a hardcoded 12, so
    the query-generation card stated one number and displayed a different set. That card IS
    the `query_generation` gate's approval surface: approving 12 queries when 13 will run is
    approving something other than what happens. The dropped one (`svc_alerts`) was visible
    in Monitor, because `_summ_logs` is the one summarizer that never capped its sources —
    i.e. the two views of the same run disagreed with each other.
    """
    from src.pipeline_runner import _SUM_MAX_ITEMS, summarize_stage

    # 13 sources, the count from the live report, and above the old bound of 12.
    queries = [
        RetrievalQuery(
            target_log_source=f"source_{i}",
            natural_language_query=f"retrieve from source {i}",
            date_from="2026-07-04",
            date_to="2026-07-05",
        )
        for i in range(13)
    ]
    assert len(queries) <= _SUM_MAX_ITEMS, "bound is below real data again"
    s = summarize_stage("query_generation", queries)
    assert s["count"] == 13
    assert len(s["queries"]) == 13, "the gate is listing fewer queries than will run"
    # By NAME, not just by count: the whole defect was one specific source going missing.
    assert [q["source"] for q in s["queries"]] == [f"source_{i}" for i in range(13)]


def test_every_extracted_entity_reaches_the_review_surface():
    """The same defect one stage earlier, and this one had nothing that disagreed.

    A live run classified 14 entities and the understanding card showed 12, silently: unlike
    the queries there was no total beside the list, so a bounded list was indistinguishable
    from a complete one. Entities are what every downstream filter is built from and what the
    `value_form` classification routes to a column, so a reviewer seeing 12 of 14 is checking
    a subset of the decisions the engine made.
    """
    from src.pipeline_runner import summarize_stage

    analysis = IncidentAnalysis(
        incident_summary="many entities",
        severity="HIGH",
        severity_reasoning="r",
        impact_assessment="i",
        key_investigation_areas=["auth"],
        log_sources_to_review=[f"src_{i}" for i in range(13)],
        initial_hypotheses=["h"],
        recommended_actions=["a"],
        stakeholder_notification=["SOC"],
        correlation_keys=["user"],
        extracted_entities=[
            ExtractedEntity(type="user", value=f"U{i}") for i in range(14)
        ],
    )
    s = summarize_stage(
        "understanding", UnderstandingResult(incident_id="INC-1", analysis=analysis)
    )
    assert len(s["entities"]) == 14
    assert [e["value"] for e in s["entities"]] == [f"U{i}" for i in range(14)]
    # The TRUE total rides alongside, so the card can never report the bound as the finding.
    assert s["entity_count"] == 14
    assert len(s["log_sources_to_review"]) == 13
    assert s["source_review_count"] == 13


def test_a_bounded_summary_list_reports_its_own_bound():
    """The bound still has to exist — summaries ride the SSE replay buffer — but not silently.

    `_counted` returns the true total beside the kept slice so a cut is a STATED cut. Without
    this the failure mode returns: raise the bound and a bigger incident walks straight back
    into a full-looking list that isn't.
    """
    from src.pipeline_runner import _counted

    kept, total = _counted(list(range(50)), limit=10)
    assert kept == list(range(10))
    assert total == 50, "the true total is the only thing that can report a cut"
    # At or under the bound, nothing is claimed to be missing.
    assert _counted([1, 2, 3], limit=10) == ([1, 2, 3], 3)
    assert _counted(None, limit=10) == ([], 0)


def test_the_summary_bound_is_the_operators_number_not_the_codes():
    """`jobs.summary_max_items` must actually reach the summarizers, or it is decoration.

    The bound lives in a module global read by module-level functions, which is exactly the
    shape where a config field gets accepted and does nothing: a `limit=_SUM_MAX_ITEMS`
    DEFAULT ARGUMENT would bind once at import, so every summarizer defined before the
    config was applied would keep the old value while the UI reported the new one.
    """
    from src import pipeline_runner as pr

    original = pr._SUM_MAX_ITEMS
    try:
        assert pr.configure_summaries({"jobs": {"summary_max_items": 3}}) == 3
        entities = [ExtractedEntity(type="user", value=f"U{i}") for i in range(6)]
        analysis = IncidentAnalysis(
            incident_summary="s",
            severity="HIGH",
            severity_reasoning="r",
            impact_assessment="i",
            key_investigation_areas=["auth"],
            log_sources_to_review=["a"],
            initial_hypotheses=["h"],
            recommended_actions=["a"],
            stakeholder_notification=["SOC"],
            correlation_keys=["user"],
            extracted_entities=entities,
        )
        s = pr.summarize_stage(
            "understanding", UnderstandingResult(incident_id="I", analysis=analysis)
        )
        assert (
            len(s["entities"]) == 3
        ), "the configured bound never reached the summarizer"
        assert s["entity_count"] == 6, "...and the total must still be the truth"
        # Absent, unparseable and out-of-range all fall back to the default rather than
        # raising: a DISPLAY bound must never be why a run cannot start.
        assert pr.configure_summaries({}) == pr._SUM_MAX_ITEMS_DEFAULT
        assert pr.configure_summaries({"jobs": {}}) == pr._SUM_MAX_ITEMS_DEFAULT
        for bad in ("lots", None, 0, -5, 10**9):
            assert (
                pr.configure_summaries({"jobs": {"summary_max_items": bad}})
                == pr._SUM_MAX_ITEMS_DEFAULT
            ), bad
    finally:
        pr._SUM_MAX_ITEMS = original


def test_summarize_is_defensive_on_bad_input():
    from src.pipeline_runner import summarize_stage

    # Wrong types must not raise — they yield a minimal/empty summary.
    assert summarize_stage("understanding", None) == {}
    assert summarize_stage("log_retrieval", "not a dict")["total_rows"] == 0
    assert summarize_stage("correlation", None) == {"skipped": True}
    assert isinstance(summarize_stage("anomaly_detection", None), dict)


# --- enriched snapshot + list_jobs (polling API parity) --------------------


async def test_snapshot_includes_duration_and_summary():
    async def s1(ctx):
        return {"esql": [{"a": 1}]}

    jm = _emitter_with_manager([StageDescriptor("log_retrieval", s1, "logs")])
    job = jm.create_job(_incident())
    await jm.run_job(job)

    snap = job.snapshot()
    stage = snap["stages"][0]
    assert stage["name"] == "log_retrieval"
    assert stage["status"] == "completed"
    assert isinstance(stage["duration_ms"], int)
    assert stage["summary"]["total_rows"] == 1


async def test_list_jobs_newest_first():
    jm = _emitter_with_manager([StageDescriptor("s", lambda ctx: None, "o")])
    j1 = jm.create_job({"id": "A", "description": "x"})
    j2 = jm.create_job({"id": "B", "description": "y"})
    rows = jm.list_jobs()
    ids = [r["job_id"] for r in rows]
    assert set(ids) == {j1.job_id, j2.job_id}
    assert all("status" in r and "current_stage" in r for r in rows)


async def test_process_incident_via_job_return_job():
    from src.pipeline_runner import process_incident_via_job

    async def s1(ctx):
        return "REPORT-TEXT"

    stages = [StageDescriptor("report_generation", s1, "report")]
    jm = _emitter_with_manager(stages)
    # report-only (default)
    report = await process_incident_via_job(_incident(), {}, {}, jm)
    assert report == "REPORT-TEXT"
    # return_job=True yields the Job with populated timings
    job = await process_incident_via_job(_incident(), {}, {}, jm, return_job=True)
    assert job.context.outputs["report"] == "REPORT-TEXT"
    assert "report_generation" in job.stage_durations


# --- stage health riding a real run (Phase A: scoring only, no gating) ------


async def test_health_rides_the_snapshot_and_the_stage_output_event():
    """A real run must carry a score + reasons per stage, with no behaviour change."""

    async def retrieval(ctx):
        # 1 of 2 sources empty -> a pro-rata `empty_sources` penalty, not fatal.
        return {"src_a": [{"x": 1}], "src_b": []}

    jm = _emitter_with_manager([StageDescriptor("log_retrieval", retrieval, "logs")])
    job = jm.create_job(_incident())
    await jm.run_job(job)

    assert job.status == JobStatus.COMPLETED  # scoring never blocks
    health = job.snapshot()["stages"][0]["health"]
    assert 0.0 < health["score"] < 1.0
    assert health["scored"] is True
    assert "empty_sources" in [r["code"] for r in health["reasons"]]

    outputs = [e for e in job._event_history if e["type"] == "stage_output"]
    assert outputs[0]["data"]["health"]["score"] == health["score"]


async def test_health_survives_the_export_import_round_trip():
    async def retrieval(ctx):
        return {"src_a": []}

    jm = _emitter_with_manager([StageDescriptor("log_retrieval", retrieval, "logs")])
    job = jm.create_job(_incident())
    await jm.run_job(job)

    doc = jm.export_job(job.job_id)
    assert doc["stage_health"]["log_retrieval"]["score"] == 0.0  # no rows at all

    jm2 = _emitter_with_manager([StageDescriptor("log_retrieval", retrieval, "logs")])
    revived = jm2.import_job(doc)
    assert revived.stage_health["log_retrieval"]["score"] == 0.0


async def test_a_link_escalation_mode_SURVIVES_the_export_import_round_trip():
    """The escalation mode must survive an export/import round-trip.

    The setting lives on the job because correlation re-runs on a gate rejection or a
    restart that restored the queue as PAUSED. Dropped from the export, the re-run
    resolves every link back to the engine's default with nothing saying it happened.

    Two halves: the doc must carry it, and the restore must read it back onto the
    context the correlation call takes its `link_modes` from. An unspellable value is
    restored verbatim; the vocabulary has one home in `resolve_link_mode`.
    """

    async def noop(ctx):
        return "r"

    stages = [StageDescriptor("understanding", noop, "understanding")]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    job.context.link_modes = {"sibling_proc": "auto", "third_proc": "semi-auto"}

    doc = jm.export_job(job.job_id)
    assert doc["link_modes"] == {"sibling_proc": "auto", "third_proc": "semi-auto"}

    revived = _emitter_with_manager(stages).import_job(doc)
    assert revived.context.link_modes == {
        "sibling_proc": "auto",
        "third_proc": "semi-auto",
    }

    # A doc from before the field existed restores to "nobody asked", not to a crash: the
    # export format grows, and a queue persisted by an older build is exactly what a restart
    # after an upgrade reads.
    doc.pop("link_modes")
    assert _emitter_with_manager(stages).import_job(doc).context.link_modes == {}


def _link(**over):
    """One advisory finding, shaped by field access only.

    A ``SimpleNamespace`` rather than a mock on purpose: every field the recorder reads is
    read with ``getattr``, and a ``MagicMock`` answers every one of them truthily — so the
    single assertion that matters most here (a `planned` link records **nothing**) would pass
    against a recorder that records everything.
    """
    f = {
        "target_use_case": "sibling_proc",
        "mode": "auto",
        "mode_source": "pack",
        "probe_source": "depot_roster",
        "probe_spent": True,
        "probe_note": "returned 2 row(s)",
        "score": 0.9,
    }
    f.update(over)
    return SimpleNamespace(**f)


async def test_an_AUTOMATIC_link_escalation_is_RECORDED_and_survives_the_round_trip(
    caplog,
):
    """An automatic escalation must be recorded, and the record must survive export/import.

    Four separate claims, each failing on a different edit:

    * an escalating mode with a probe attempted records one entry — reaching, not settling;
    * a `planned` link records nothing — composing a referral is not an escalation;
    * an escalating mode that never reached a backend records nothing — a permission not
      exercised is not an action;
    * the entry survives ``export_job`` → ``import_job`` — the trail is read after the
      writing process exits (a restart restoring a PAUSED job).
    """

    async def noop(ctx):
        return "r"

    stages = [StageDescriptor("correlation", noop, "correlation")]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())

    result = SimpleNamespace(
        links=[
            _link(),
            _link(
                target_use_case="third_proc",
                mode="semi_auto",
                mode_source="job",
                probe_source="gate_ledger",
                probe_spent=False,
                probe_note="'gate_ledger' did not answer (TimeoutError)",
            ),
            # Strongest signal in the list, and deliberately at the default mode.
            _link(target_use_case="unasked_proc", mode="planned", score=1.0),
            # Licensed, and it never got as far as a backend.
            _link(target_use_case="unreached_proc", probe_source="", probe_note=""),
        ]
    )

    assert jm._record_automatic_escalations(job, result) == 2
    assert len(job.interventions) == 2

    for entry in job.interventions:
        assert entry["action"] == "link_escalated"
        assert entry["stage"] == "correlation"
        # The arithmetic decided this link, not a person — and which LAYER licensed the mode
        # at all is the different question the detail answers.
        assert entry["actor"] == "engine"

    first, second = (e["detail"] for e in job.interventions)
    assert "auto" in first and "set by pack" in first
    assert "one probe" in first and "depot_roster" in first and "sibling_proc" in first
    assert "returned 2 row(s)" in first
    assert "semi_auto" in second and "set by job" in second
    # A licensed scan that settled nothing is still an escalation, and it says which it was.
    assert "no probe" in second and "gate_ledger" in second and "third_proc" in second
    assert "TimeoutError" in second

    trail = " ".join(e["detail"] for e in job.interventions)
    assert "unasked_proc" not in trail
    assert "unreached_proc" not in trail

    # A result with no links, or where `links` is not a list: nothing recorded, nothing
    # raised, and nothing warned. The broad `except` swallows the error either way;
    # the shape check earns its place only in what it does not say.
    # Cleared first, and read by level not text: the escalations above log an INFO line,
    # and a text match fails in a full run if an earlier test left the root level set.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert jm._record_automatic_escalations(job, SimpleNamespace()) == 0
        assert jm._record_automatic_escalations(job, SimpleNamespace(links=None)) == 0
        assert jm._record_automatic_escalations(job, MagicMock()) == 0
        assert jm._record_automatic_escalations(job, "not a result at all") == 0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(job.interventions) == 2

    doc = jm.export_job(job.job_id)
    assert [e["action"] for e in doc["interventions"]] == [
        "link_escalated",
        "link_escalated",
    ]

    revived = _emitter_with_manager(stages).import_job(doc)
    assert [e["detail"] for e in revived.interventions] == [first, second]
    assert {e["actor"] for e in revived.interventions} == {"engine"}


async def test_a_RUN_records_its_own_automatic_escalations_off_the_correlation_stage():
    """The recorder is only a recorder if a run asks it, and only correlation can.

    Separate from the test above on purpose: that one proves the entry is shaped and durable,
    this one proves it is ever WRITTEN. A trail no stage reaches is the same as no trail —
    the shape this file keeps re-learning about guards that were never called — and the call
    site is the one part of it a unit test on the method cannot see.

    Two further claims ride along, both about *where* the call sits. It is in the runner and
    not inside correlation, because that module is deliberately IO-free and holds no reference
    to the job it is running under; and it is before the approval gate, so a reviewer holding
    the stage reads what was already spent on its behalf rather than authorising it afterwards.
    Both are asserted through the surface an operator actually reads — the snapshot.
    """
    seen = []

    async def correlation(ctx):
        seen.append("ran")
        return SimpleNamespace(
            links=[_link(), _link(target_use_case="quiet", mode="planned")]
        )

    stages = [StageDescriptor("correlation", correlation, "correlation")]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    await jm.run_job(job)

    assert seen == ["ran"]
    assert job.status == JobStatus.COMPLETED
    trail = [
        e for e in job.snapshot()["interventions"] if e["action"] == "link_escalated"
    ]
    assert len(trail) == 1
    assert "sibling_proc" in trail[0]["detail"] and "quiet" not in trail[0]["detail"]


async def test_a_restored_job_keeps_the_time_it_actually_RAN():
    """WHEN a job ran is a property of the run, not of the process that reloaded it.

    ``import_job`` goes through ``create_job``, which stamps ``created_at`` with *now*. It
    restored fifteen other fields and not this one, so restoring a queue gave every job the
    same timestamp to the microsecond — the restore moment. Two visible consequences, and
    the second is the one that made the history useless: the Monitor's job list showed one
    time repeated down every row, and ``list_jobs`` sorts newest-first **keyed on this
    field**, so the ordering collapsed into dict order.

    Asserted against a persisted time that is unambiguously not "now", because a test using
    a freshly-created job would pass on the broken code by coincidence.
    """

    async def noop(ctx):
        return "r"

    stages = [StageDescriptor("understanding", noop, "understanding")]
    jm = _emitter_with_manager(stages)
    job = jm.create_job(_incident())
    doc = jm.export_job(job.job_id)
    doc["created_at"] = "2026-08-01T09:15:00+00:00"
    doc["updated_at"] = "2026-08-01T09:47:30+00:00"

    jm2 = _emitter_with_manager(stages)
    revived = jm2.import_job(doc)
    assert revived.created_at == "2026-08-01T09:15:00+00:00"
    assert revived.updated_at == "2026-08-01T09:47:30+00:00"
    # The numeric twin is deliberately NOT restored: it drives the in-memory TTL, and a
    # real age there makes every reloaded job evictable on the next create_job — which
    # would empty the history this restore just loaded.
    assert revived.created_ts > 0
    assert (time.time() - revived.created_ts) < 60

    # And the list the operator reads is sorted by it, so a restored older job must not
    # jump ahead of a newer one.
    newer = jm2.create_job(_incident())
    rows = jm2.list_jobs()
    assert [r["job_id"] for r in rows][0] == newer.job_id
    assert [r["created_at"] for r in rows][-1] == "2026-08-01T09:15:00+00:00"


async def test_a_malformed_persisted_timestamp_does_not_lose_the_job():
    """A doc hand-edited to a null/blank time must still restore — with a usable stamp.

    The restore path's whole promise is that one bad field cannot cost the queue an entry,
    and `created_at` is what `list_jobs` sorts on: a `None` there would raise inside the
    sort and take out the whole listing, not just this row.
    """

    async def noop(ctx):
        return "r"

    stages = [StageDescriptor("understanding", noop, "understanding")]
    jm = _emitter_with_manager(stages)
    doc = jm.export_job(jm.create_job(_incident()).job_id)
    for bad in (None, "", 12345):
        doc["created_at"] = bad
        revived = _emitter_with_manager(stages).import_job(doc)
        assert isinstance(revived.created_at, str) and revived.created_at
    assert isinstance(jm.list_jobs(), list)


async def test_log_retrieval_records_per_source_outcomes_for_the_scorer():
    """A timed-out source and an empty one are indistinguishable in `logs`."""

    class _Engine:
        config = {}
        retrievers = {}

        async def retrieve(
            self,
            queries,
            progress_cb=None,
            extended=False,
            keyed_out=None,
            queries_out=None,
            # The real signature, including the out-parameter this test does not read — see
            # tests/CLAUDE.md: the stage passes all of them, and an unexpected-kwarg
            # `TypeError` here surfaces as a retrieval failure with the cause in a log line.
            unanswered_out=None,
        ):
            progress_cb("ok", "completed", "2 rows")
            progress_cb("slow", "timeout", "timed out after 20s")
            return {"ok": [{"a": 1}, {"a": 2}], "slow": []}

    from src.pipeline_runner import _run_log_retrieval

    jm = _emitter_with_manager(
        [StageDescriptor("log_retrieval", _run_log_retrieval, "logs")],
        modules={"log_retrieval": _Engine()},
    )
    job = jm.create_job(_incident())
    job.context.outputs["queries"] = []
    await jm.run_job(job)

    facts = job.context.stage_facts["log_retrieval"]["sources"]
    assert facts["slow"]["status"] == "timeout"
    codes = [r["code"] for r in job.stage_health["log_retrieval"]["reasons"]]
    assert "source_timeout" in codes


async def test_query_ready_annotates_a_running_source_without_settling_it():
    """`query_ready` reports that a source's query exists without settling the source.
    If it landed in `source_outcomes` it would overwrite `running` with a non-terminal
    status; the scorer reads that dict to tell an empty source from a timed-out one.

    The event must carry the current pass's query. The engine fills a `queries_out` dict
    at publish time rather than reading `retriever.last_generated_query`: that attribute
    is state on a retriever that outlives the run. The `running` event must carry no
    query; only events after publication may carry one.
    """

    class _Engine:
        config = {}
        # A retriever holding a previous run's query, which is the live shape: one instance
        # per source, shared by every job, and its `last_generated_query` never cleared.
        retrievers = {"slow": SimpleNamespace(last_generated_query="STALE FROM AN OLDER RUN")}

        async def retrieve(
            self,
            queries,
            progress_cb=None,
            extended=False,
            keyed_out=None,
            queries_out=None,
            # The real signature — see the note on the double above.
            unanswered_out=None,
        ):
            progress_cb("slow", "running", "Querying slow…")
            if queries_out is not None:
                queries_out["slow"] = "SELECT this_run FROM t"
            progress_cb("slow", "query_ready", "Query generated for slow")
            progress_cb("slow", "timeout", "timed out after 20s")
            return {"slow": []}

    import src.pipeline_runner as pr

    # `progress_cb` publishes through the module-level emitter (one process-wide fan-out),
    # so capture there rather than on the test's own JobManager.
    seen = []
    original = pr.emitter.emit
    pr.emitter.emit = lambda *a, **kw: seen.append(
        (kw.get("status"), (kw.get("data") or {}).get("generated_query"))
    )
    try:
        jm = _emitter_with_manager(
            [StageDescriptor("log_retrieval", pr._run_log_retrieval, "logs")],
            modules={"log_retrieval": _Engine()},
        )
        job = jm.create_job(_incident())
        job.context.outputs["queries"] = []
        await jm.run_job(job)
    finally:
        pr.emitter.emit = original

    facts = job.context.stage_facts["log_retrieval"]["sources"]
    # The terminal status survived; the annotation did not displace it.
    assert facts["slow"]["status"] == "timeout"
    by_status = dict(seen)
    assert "query_ready" in by_status  # still reported to the operator
    # Published: the query this pass generated, never the retriever's leftover.
    assert by_status["query_ready"] == "SELECT this_run FROM t"
    assert by_status["timeout"] == "SELECT this_run FROM t"
    # Before publication there IS no query for this pass, and reporting the stale one would
    # put two contradictory queries in the record for one source in one pass.
    assert by_status["running"] is None


async def test_an_override_rescores_the_stage_rather_than_keeping_the_bad_score():
    """The override exists to FIX a stage; the score must reflect the fix."""

    async def retrieval(ctx):
        return {"src_a": []}  # fatal: no rows at all

    jm = _emitter_with_manager([StageDescriptor("log_retrieval", retrieval, "logs")])
    job = jm.create_job(_incident())
    await jm.run_job(job)
    assert job.stage_health["log_retrieval"]["score"] == 0.0

    jm.set_stage_output(job.job_id, "log_retrieval", {"src_a": [{"x": 1}]}, actor="me")
    assert job.stage_health["log_retrieval"]["score"] == 1.0


# --- gate durability against a queueing store ------------------------------
# Gate state must reach durable storage before the operator is notified. A backend that
# queues writes breaks this silently unless the gate save is flushed: every save reports
# success and the loss shows up only at restart.


class _QueueingStore:
    """A JobStore whose writes only land when ``flush`` is called.

    Stands in for ``DatabricksStorage``'s writer thread without needing one: what is being
    asserted is that the runner *asks*, not how a backend queues.
    """

    def __init__(self):
        self.queued = {}
        self.landed = {}
        self.flushes = 0

    def save(self, doc, evidence_changed=False):
        self.queued[doc["job_id"]] = doc
        return True

    def flush(self, timeout=30.0):
        self.flushes += 1
        self.landed.update(self.queued)
        self.queued.clear()
        return True

    def prune(self):
        return 0


async def test_a_gate_is_durable_before_it_is_announced():
    """Queued is not persisted. A container death here loses the pending approval."""
    store = _QueueingStore()

    async def retrieval(ctx):
        return {"src_a": []}  # scores 0.0, so semi_auto gates

    jm = _emitter_with_manager(
        [StageDescriptor("log_retrieval", retrieval, "logs")], store=store
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.create_task(jm.run_job(job))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if job.open_gate:
            break
    assert job.open_gate, "the stage did not gate"
    # Give the flush its turn on the loop — it runs in a thread, so the gate can be open
    # for an instant before it completes.
    for _ in range(100):
        await asyncio.sleep(0.01)
        if store.flushes:
            break

    # Asserted WHILE the gate is open, which is the whole point: a flush that only
    # happened at the end of the run would leave the days-long wait unprotected.
    assert store.flushes >= 1, "the gate save was never flushed to durable storage"
    assert job.job_id in store.landed
    assert store.landed[job.job_id]["status"] == "awaiting_approval"
    assert not store.queued, "a write was still queued with the gate already announced"

    jm.resolve_gate(job.job_id, "approve", actor="tester")
    await asyncio.wait_for(task, timeout=5)
    assert job.status == JobStatus.COMPLETED


async def test_a_local_store_without_flush_still_runs():
    """Every existing deployment. `flush` is optional on the store, not required."""

    class _PlainStore:
        def __init__(self):
            self.saves = 0

        def save(self, doc, evidence_changed=False):
            self.saves += 1
            return True

    store = _PlainStore()

    async def retrieval(ctx):
        return {"src_a": [{"x": 1}]}

    jm = _emitter_with_manager(
        [StageDescriptor("log_retrieval", retrieval, "logs")], store=store
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.AUTO)
    await jm.run_job(job)

    assert job.status == JobStatus.COMPLETED
    assert store.saves > 0
