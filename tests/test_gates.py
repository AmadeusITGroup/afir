"""
Tests for approval gates (src/pipeline_runner.py + src/human_guidance.py).

Gates are the supervision surface: in ``semi_auto`` a stage whose deterministic health
is below threshold stops and asks a human; in ``supervised`` every gateable stage does.
Three regressions these guard specifically:

1. **A gate must actually block.** A gate that opens and then lets the run continue is
   worse than no gate — the report ships as if it were reviewed.
2. **A reject must change the retry's prompt.** Re-running an identical prompt gets an
   identical answer, so a reject whose guidance never reaches the model presents the
   analyst with the same output they just rejected and looks like their correction was
   ignored.
3. **A gate must not outlive its run.** A cancel while a gate is open has to terminate;
   and a control action that restarts the pipeline must not leave a second stage loop
   parked on the old gate, racing the new one over one outputs dict.

Stages are lightweight async callables — no LLM, no retrievers, no network.
"""

import asyncio

import pytest

from src.human_guidance import (MAX_ITEMS, guidance_message,
                                guidance_prompt_line, render_guidance)
from src.models.pydantic_models import (EventWindow, ExtractedEntity,
                                        IncidentAnalysis, UnderstandingResult)
from src.notifications import EventEmitter
from src.pipeline_runner import (GATE_ACTIONS, JobManager, JobRunMode,
                                 JobStatus, StageDescriptor)

# Real gateable stage names, so stage_health scores them and stage_gate_enabled
# allows a gate. A made-up name would be `scored=False` and never gate in semi_auto.
GATEABLE = "understanding"


def _manager(stages, **kwargs):
    emitter = EventEmitter()
    jm = JobManager(stages, emitter, **kwargs)
    emitter.set_job_manager(jm)
    return jm


def _incident():
    return {"id": "INC-G", "description": "gate test", "timestamp": "2026-07-30T00:00"}


def _events(job, event_type):
    return [e for e in job._event_history if e.get("type") == event_type]


async def _wait_for_gate(job, timeout=2.0, after=None):
    """Block until a gate is open. Fails the test rather than hanging forever.

    ``after`` is a previously-returned gate: resolving a gate sets the event but the
    runner clears ``open_gate`` only once it wakes, so waiting on mere presence would
    return the STALE gate and the assertion would race. Keying on ``opened_at``
    identity makes "the next gate" unambiguous.
    """
    stale = (after or {}).get("opened_at")

    async def _poll():
        while job.open_gate is None or (
            stale is not None and job.open_gate.get("opened_at") == stale
        ):
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout=timeout)
    return job.open_gate


# --- mode behaviour --------------------------------------------------------


async def test_auto_mode_never_gates():
    """AUTO is the default and the classic endpoints' mode: it must be untouched."""

    async def understanding(ctx):
        return None  # score 0.0 — would gate in any gating mode

    jm = _manager([StageDescriptor(GATEABLE, understanding, "understanding")])
    job = jm.create_job(_incident(), run_mode=JobRunMode.AUTO)
    await jm.run_job(job)

    assert job.status == JobStatus.COMPLETED
    assert job.open_gate is None
    assert _events(job, "gate_opened") == []


async def test_step_mode_does_not_gate():
    """STEP is a PRE-stage wait, not a gate — the two must stay distinct."""

    async def understanding(ctx):
        return None

    jm = _manager([StageDescriptor(GATEABLE, understanding, "understanding")])
    job = jm.create_job(_incident(), run_mode=JobRunMode.STEP)
    task = asyncio.ensure_future(jm.run_job(job))
    await asyncio.sleep(0.02)
    jm.control(job.job_id, "step")
    await asyncio.wait_for(task, timeout=2)

    assert job.status == JobStatus.COMPLETED
    assert _events(job, "gate_opened") == []


async def test_semi_auto_gates_only_below_threshold():
    """A healthy stage passes unattended; an unhealthy one stops for a human."""

    async def healthy(ctx):
        # A real UnderstandingResult-shaped object would be needed for a perfect score;
        # a duck-typed stand-in with the fields the scorer reads is enough here.
        return _Understanding()

    jm = _manager([StageDescriptor(GATEABLE, healthy, "understanding")])
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    await asyncio.wait_for(jm.run_job(job), timeout=2)

    assert job.status == JobStatus.COMPLETED
    assert _events(job, "gate_opened") == []
    assert job.stage_health[GATEABLE]["score"] == 1.0


async def test_semi_auto_gate_blocks_until_resolved():
    ran = []

    async def understanding(ctx):
        ran.append("understanding")
        return None  # fatal signal -> score 0.0

    async def later(ctx):
        ran.append("later")
        return "x"

    jm = _manager(
        [
            StageDescriptor(GATEABLE, understanding, "understanding"),
            StageDescriptor("export", later, "export_paths"),
        ]
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    gate = await _wait_for_gate(job)

    # Blocked: the gated stage ran, the next one did NOT.
    assert ran == ["understanding"]
    assert job.status == JobStatus.AWAITING_APPROVAL
    assert gate["stage"] == GATEABLE
    assert gate["health"]["score"] == 0.0
    assert gate["actions"] == list(GATE_ACTIONS)
    # The opened event carries enough to render a decision screen in one shot.
    opened = _events(job, "gate_opened")[-1]
    assert opened["data"]["health"]["reasons"]
    assert "summary" in opened["data"]

    jm.resolve_gate(job.job_id, "approve", actor="ana")
    await asyncio.wait_for(task, timeout=2)

    assert ran == ["understanding", "later"]
    assert job.status == JobStatus.COMPLETED
    assert job.open_gate is None


async def test_supervised_gates_even_a_healthy_stage():
    jm = _manager([StageDescriptor(GATEABLE, _healthy_stage, "understanding")])
    job = jm.create_job(_incident(), run_mode=JobRunMode.SUPERVISED)
    task = asyncio.ensure_future(jm.run_job(job))
    gate = await _wait_for_gate(job)

    assert gate["reason"] == "supervised mode"
    assert gate["health"]["score"] == 1.0  # healthy, and gated anyway
    jm.resolve_gate(job.job_id, "approve")
    await asyncio.wait_for(task, timeout=2)
    assert job.status == JobStatus.COMPLETED


async def test_non_gateable_stage_never_gates_even_supervised():
    """plugins/export/output are mechanical; there is nothing for a human to judge."""
    jm = _manager([StageDescriptor("export", _healthy_stage, "export_paths")])
    job = jm.create_job(_incident(), run_mode=JobRunMode.SUPERVISED)
    await asyncio.wait_for(jm.run_job(job), timeout=2)

    assert job.status == JobStatus.COMPLETED
    assert _events(job, "gate_opened") == []


async def test_config_can_disable_a_stages_gate():
    jm = _manager(
        [StageDescriptor(GATEABLE, _healthy_stage, "understanding")],
        config={"stage_gates": {"stages": {GATEABLE: {"enabled": False}}}},
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SUPERVISED)
    await asyncio.wait_for(jm.run_job(job), timeout=2)

    assert job.status == JobStatus.COMPLETED
    assert _events(job, "gate_opened") == []


async def test_semi_auto_respects_a_per_stage_threshold():
    """A threshold of 0 means "never gate on health" for that stage."""

    async def understanding(ctx):
        return None  # score 0.0

    jm = _manager(
        [StageDescriptor(GATEABLE, understanding, "understanding")],
        config={"stage_gates": {"stages": {GATEABLE: {"threshold": 0.0}}}},
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    await asyncio.wait_for(jm.run_job(job), timeout=2)

    assert job.status == JobStatus.COMPLETED
    assert _events(job, "gate_opened") == []


async def test_semi_auto_does_not_gate_an_unscored_stage():
    """`plugins` scores 1.0 as a placeholder; that is not a measurement to act on.

    Belt-and-braces alongside the non-gateable check: even if a stage were made
    gateable, semi_auto must not read an unscored placeholder as "healthy".
    """
    jm = _manager(
        [StageDescriptor("plugins", _healthy_stage, "plugin_results")],
        config={"stage_gates": {"stages": {"plugins": {"enabled": True}}}},
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    await asyncio.wait_for(jm.run_job(job), timeout=2)

    assert job.status == JobStatus.COMPLETED
    assert _events(job, "gate_opened") == []


# --- reject: the guidance must reach the retry ------------------------------


async def test_reject_reruns_the_stage_with_guidance_in_context():
    seen = []

    async def understanding(ctx):
        seen.append(list((ctx.stage_guidance or {}).get(GATEABLE) or []))
        # Healthy on the second attempt, so the re-run's gate can be approved.
        return None if len(seen) == 1 else _Understanding()

    jm = _manager([StageDescriptor(GATEABLE, understanding, "understanding")])
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)

    jm.resolve_gate(
        job.job_id,
        "reject",
        guidance="the org_unit id is 12345678, not the agent sign",
        reason_code="wrong_entity",
        actor="ana",
    )
    # The re-run gates again (score 1.0 is healthy, so only if supervised) — here it is
    # semi_auto and the second attempt is healthy, so the run completes on its own.
    await asyncio.wait_for(task, timeout=2)

    assert len(seen) == 2, "the rejected stage must actually re-run"
    assert seen[0] == [], "first attempt has no guidance"
    assert seen[1] == ["the org_unit id is 12345678, not the agent sign"]
    assert job.status == JobStatus.COMPLETED
    # Audit trail: the rejection, its reason code, and who did it.
    actions = [i["action"] for i in job.interventions]
    assert "gate_reject" in actions
    reject = [i for i in job.interventions if i["action"] == "gate_reject"][0]
    assert reject["actor"] == "ana"
    assert "wrong_entity" in reject["detail"]
    assert job.gate_history[0]["action"] == "reject"
    assert job.gate_history[0]["reason_code"] == "wrong_entity"


async def test_reject_without_guidance_is_refused():
    """A reject with no correction is a retry of an identical prompt — refuse it."""

    async def understanding(ctx):
        return None

    jm = _manager([StageDescriptor(GATEABLE, understanding, "understanding")])
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)

    with pytest.raises(ValueError, match="guidance"):
        jm.resolve_gate(job.job_id, "reject", guidance="   ")
    # The gate is still open and still resolvable.
    assert job.open_gate is not None
    jm.resolve_gate(job.job_id, "approve")
    await asyncio.wait_for(task, timeout=2)


async def test_reject_with_restart_from_reruns_the_earlier_stage():
    """Rejecting retrieval for a bad WINDOW must be fixable at query generation."""
    calls = []

    async def understanding(ctx):
        calls.append("understanding")
        return _Understanding()

    async def query_generation(ctx):
        calls.append("query_generation")
        return [_Query()]

    async def log_retrieval(ctx):
        calls.append("log_retrieval")
        # Empty on the first pass (score 0.0), rows once the window is corrected.
        return {} if calls.count("log_retrieval") == 1 else {"src": [{"a": 1}]}

    jm = _manager(
        [
            StageDescriptor("understanding", understanding, "understanding"),
            StageDescriptor("query_generation", query_generation, "queries"),
            StageDescriptor("log_retrieval", log_retrieval, "logs"),
        ]
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)
    assert job.open_gate["stage"] == "log_retrieval"

    jm.resolve_gate(
        job.job_id,
        "reject",
        guidance="widen the window to the whole month",
        restart_from="query_generation",
    )
    await asyncio.wait_for(task, timeout=3)

    # understanding ran once; query_generation and log_retrieval ran twice.
    assert calls.count("understanding") == 1
    assert calls.count("query_generation") == 2
    assert calls.count("log_retrieval") == 2
    # The guidance is attached to the stage that RE-RAN, not the gated one.
    assert job.context.stage_guidance["query_generation"] == [
        "widen the window to the whole month"
    ]
    assert "log_retrieval" not in job.context.stage_guidance
    assert job.status == JobStatus.COMPLETED


async def test_restart_from_a_later_stage_is_refused():
    """Re-running a stage that comes AFTER the gate cannot fix the gated output."""

    async def understanding(ctx):
        return None

    async def query_generation(ctx):
        return [_Query()]

    jm = _manager(
        [
            StageDescriptor("understanding", understanding, "understanding"),
            StageDescriptor("query_generation", query_generation, "queries"),
        ]
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)

    with pytest.raises(ValueError, match="runs after"):
        jm.resolve_gate(
            job.job_id, "reject", guidance="x", restart_from="query_generation"
        )
    with pytest.raises(ValueError, match="Unknown stage"):
        jm.resolve_gate(job.job_id, "reject", guidance="x", restart_from="nope")

    jm.resolve_gate(job.job_id, "approve")
    await asyncio.wait_for(task, timeout=2)


# --- override --------------------------------------------------------------


async def test_override_at_a_gate_replaces_the_output_and_continues():
    downstream = []

    async def log_retrieval(ctx):
        return {}  # score 0.0

    async def correlation(ctx):
        downstream.append(dict(ctx.outputs["logs"]))
        # A zero-record result, NOT None: `None` means "correlation module not wired"
        # and is deliberately not a health problem, so it would score 1.0 and not gate.
        return _Correlation(record_count=0)

    jm = _manager(
        [
            StageDescriptor("log_retrieval", log_retrieval, "logs"),
            StageDescriptor("correlation", correlation, "correlation"),
        ]
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    first = await _wait_for_gate(job)

    jm.resolve_gate(job.job_id, "override", value={"src_a": [{"x": 1}]}, actor="ana")
    # correlation's own gate (zero records -> score 0.0) opens next.
    second = await _wait_for_gate(job, after=first)
    assert second["stage"] == "correlation"
    jm.resolve_gate(job.job_id, "approve")
    await asyncio.wait_for(task, timeout=2)

    assert downstream == [{"src_a": [{"x": 1}]}], "downstream saw the analyst's data"
    # Overriding re-scores, so the corrected stage no longer reads as broken.
    assert job.stage_health["log_retrieval"]["score"] == 1.0
    actions = [i["action"] for i in job.interventions]
    assert "override_stage_output" in actions and "gate_override" in actions


# --- lifecycle: a gate must not outlive its run -----------------------------


async def test_cancel_while_a_gate_is_open_terminates_the_job():
    ran = []

    async def understanding(ctx):
        ran.append("understanding")
        return None

    async def later(ctx):
        ran.append("later")
        return "x"

    jm = _manager(
        [
            StageDescriptor(GATEABLE, understanding, "understanding"),
            StageDescriptor("export", later, "export_paths"),
        ]
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)

    jm.control(job.job_id, "cancel_all")
    await asyncio.wait_for(task, timeout=2)

    assert job.status == JobStatus.CANCELLED
    assert ran == ["understanding"], "the run must not continue past a cancelled gate"
    assert job.open_gate is None


async def test_release_gate_continues_and_records_that_nobody_reviewed():
    ran = []

    async def understanding(ctx):
        return None

    async def later(ctx):
        ran.append("later")
        return "x"

    jm = _manager(
        [
            StageDescriptor(GATEABLE, understanding, "understanding"),
            StageDescriptor("export", later, "export_paths"),
        ]
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)

    jm.control(job.job_id, "release_gate")
    await asyncio.wait_for(task, timeout=2)

    assert ran == ["later"]
    assert job.status == JobStatus.COMPLETED
    actions = [i["action"] for i in job.interventions]
    assert "gate_released" in actions, "an unreviewed stage must not look reviewed"
    assert "gate_approve" not in actions


async def test_retry_all_while_gated_does_not_leave_two_loops_running():
    """The parked gate loop must stand down, or two loops race one outputs dict."""
    starts = []

    async def understanding(ctx):
        starts.append(1)
        return None

    jm = _manager([StageDescriptor(GATEABLE, understanding, "understanding")])
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)
    assert len(starts) == 1

    jm.control(job.job_id, "retry_all")
    await asyncio.wait_for(task, timeout=2)  # the OLD loop must return, not hang
    # The retry re-runs the stage and opens its own gate.
    await _wait_for_gate(job)
    assert len(starts) == 2, "exactly one re-run, from the new loop only"
    actions = [i["action"] for i in job.interventions]
    assert "gate_abandoned" in actions
    jm.control(job.job_id, "cancel_all")
    await asyncio.sleep(0.05)


async def test_resolving_when_no_gate_is_open_raises():
    jm = _manager([StageDescriptor(GATEABLE, _healthy_stage, "understanding")])
    job = jm.create_job(_incident(), run_mode=JobRunMode.AUTO)
    await asyncio.wait_for(jm.run_job(job), timeout=2)

    with pytest.raises(ValueError, match="No gate is open"):
        jm.resolve_gate(job.job_id, "approve")
    with pytest.raises(KeyError):
        jm.resolve_gate("no-such-job", "approve")


async def test_unknown_gate_action_raises():
    async def understanding(ctx):
        return None

    jm = _manager([StageDescriptor(GATEABLE, understanding, "understanding")])
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)

    with pytest.raises(ValueError, match="Unknown gate action"):
        jm.resolve_gate(job.job_id, "yolo")
    jm.resolve_gate(job.job_id, "approve")
    await asyncio.wait_for(task, timeout=2)


async def test_awaiting_approval_is_not_pruned():
    """A gate may be open for days; the job holding it must survive the TTL sweep."""

    async def understanding(ctx):
        return None

    jm = _manager(
        [StageDescriptor(GATEABLE, understanding, "understanding")], ttl_seconds=0
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)

    jm.create_job(_incident(), run_mode=JobRunMode.AUTO)  # triggers _prune()
    assert jm.get_job(job.job_id) is job

    jm.control(job.job_id, "cancel_all")
    await asyncio.wait_for(task, timeout=2)


# --- discovery surfaces ----------------------------------------------------


async def test_open_gate_is_discoverable_by_snapshot_and_inbox():
    async def understanding(ctx):
        return None

    jm = _manager([StageDescriptor(GATEABLE, understanding, "understanding")])
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)

    snap = job.snapshot()
    assert snap["status"] == "awaiting_approval"
    assert snap["open_gate"]["stage"] == GATEABLE
    assert jm.list_jobs()[0]["awaiting_stage"] == GATEABLE
    inbox = jm.list_open_gates()
    assert len(inbox) == 1
    assert inbox[0]["job_id"] == job.job_id
    assert inbox[0]["incident_id"] == "INC-G"
    assert inbox[0]["run_mode"] == "semi_auto"

    jm.resolve_gate(job.job_id, "approve")
    await asyncio.wait_for(task, timeout=2)
    assert jm.list_open_gates() == []
    assert job.snapshot()["open_gate"] is None
    assert job.snapshot()["gate_history"][0]["action"] == "approve"


async def test_a_cancelled_job_leaves_the_approvals_inbox():
    """The inbox is read as "here is what needs me", so a cancelled run in it is a decision
    nobody can take — and attaching to that job reopened the whole decision panel on a run
    that had ended. Live report: "Awaiting approval should be present only on the jobs that
    are awaiting approval, not all."

    Two independent guarantees, because one missed clear turns the list back into a queue of
    dead work: the cancel clears the gate, AND `list_open_gates` derives answerability from
    the job's STATUS rather than trusting the field.
    """

    async def understanding(ctx):
        return None

    jm = _manager([StageDescriptor(GATEABLE, understanding, "understanding")])
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)
    assert len(jm.list_open_gates()) == 1

    jm.control(job.job_id, "cancel_all")
    await asyncio.wait_for(task, timeout=2)

    assert job.status is JobStatus.CANCELLED
    assert job.open_gate is None
    assert job.pending_gate is None
    assert jm.list_open_gates() == []
    # And the list view stops claiming a stage needs a human.
    assert jm.list_jobs()[0]["awaiting_stage"] is None
    # A gate that was never answered must not read as one that was: the abandon is on the
    # trail, and no decision was appended to gate_history.
    actions = [iv.get("action") for iv in job.interventions]
    assert "gate_abandoned" in actions or "cancel_all" in actions
    assert job.gate_history == []


async def test_a_terminal_job_is_never_in_the_inbox_even_with_a_gate_set():
    """The status is the authority. This is the belt to the cancel path's braces: every route
    that ends a run clears the gate, and this list must be right even if one of them stops.
    """

    async def understanding(ctx):
        return None

    jm = _manager([StageDescriptor(GATEABLE, understanding, "understanding")])
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    gate = await _wait_for_gate(job)

    jm.resolve_gate(job.job_id, "approve")
    await asyncio.wait_for(task, timeout=2)
    assert job.status is JobStatus.COMPLETED
    # Force the state a missed clear would leave behind.
    job.open_gate = dict(gate)
    assert jm.list_open_gates() == []


async def test_a_live_gate_carries_its_jobs_status_to_the_inbox():
    """So a client can tell an answerable gate from one on a run that has since ended,
    without a follow-up request per row."""

    async def understanding(ctx):
        return None

    jm = _manager([StageDescriptor(GATEABLE, understanding, "understanding")])
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)

    assert jm.list_open_gates()[0]["status"] == "awaiting_approval"

    jm.control(job.job_id, "cancel_all")
    await asyncio.wait_for(task, timeout=2)


async def test_export_import_round_trips_supervision_state():
    async def understanding(ctx):
        return None if not ctx.stage_guidance else _real_understanding()

    jm = _manager([StageDescriptor(GATEABLE, understanding, "understanding")])
    job = jm.create_job(_incident(), run_mode=JobRunMode.SUPERVISED)
    task = asyncio.ensure_future(jm.run_job(job))
    first = await _wait_for_gate(job)
    jm.resolve_gate(
        job.job_id, "reject", guidance="look at the document, not the record"
    )
    # The re-run gates again (supervised gates every gateable stage).
    await _wait_for_gate(job, after=first)

    doc = jm.export_job(job.job_id)
    assert doc["gate_history"][0]["action"] == "reject"
    assert doc["stage_guidance"][GATEABLE] == ["look at the document, not the record"]
    assert doc["open_gate"]["stage"] == GATEABLE

    jm.control(job.job_id, "cancel_all")
    await asyncio.wait_for(task, timeout=2)

    jm2 = _manager([StageDescriptor(GATEABLE, understanding, "understanding")])
    revived = jm2.import_job(doc)
    assert revived.gate_history[0]["action"] == "reject"
    # The correction survives, so a re-run does not repeat the rejected prompt.
    assert revived.context.stage_guidance[GATEABLE] == [
        "look at the document, not the record"
    ]
    # The gate is NOT restored open — nothing on this side is waiting to read a
    # decision, so an approve button there would resolve nothing.
    assert revived.open_gate is None
    assert revived.status != JobStatus.AWAITING_APPROVAL


# --- the guidance renderer -------------------------------------------------


def test_render_guidance_states_its_own_subordination():
    body = render_guidance(["check the document leg status"])
    assert "check the document leg status" in body
    # It must say what it does NOT outrank, or the model can read it as authoritative.
    assert "knowledge pack" in body
    assert "official procedure" in body
    assert "outrank" in body


def test_render_guidance_is_bounded_and_keeps_the_newest():
    items = [f"correction {i}" for i in range(20)]
    body = render_guidance(items)
    assert "correction 19" in body, "the newest correction must survive"
    assert "correction 0" not in body
    assert body.count("- correction") == MAX_ITEMS
    assert len(body) <= 2100


def test_render_guidance_handles_junk_without_raising():
    assert render_guidance(None) == ""
    assert render_guidance([]) == ""
    assert render_guidance(["", "   "]) == ""
    assert render_guidance("a bare string") == ""  # not a list: ignored, not exploded
    assert guidance_message(None) is None
    assert guidance_message(["x"])["role"] == "system"


def test_guidance_prompt_line_is_inline_and_subordinate():
    line = guidance_prompt_line("filter on the org_unit, not the sign")
    assert "filter on the org_unit, not the sign" in line
    assert "unless it conflicts" in line
    assert line.endswith("\n")
    assert guidance_prompt_line("") == ""
    assert guidance_prompt_line(None) == ""


# --- stand-ins -------------------------------------------------------------
#
# Duck-typed rather than real Pydantic models: these tests only need the fields the
# scorer reads, and building a full IncidentAnalysis here would couple every gate test
# to that model's required fields.


class _EventWindow:
    start = "2026-07-01"
    end = "2026-07-31"


class _Analysis:
    def __init__(self):
        self.extracted_entities = [object()]
        self.event_time = _EventWindow()
        self.correlation_keys = ["org_unit_id"]
        self.log_sources_to_review = ["src_a"]


class _Understanding:
    def __init__(self):
        self.incident_id = "INC-G"
        self.analysis = _Analysis()


def _real_understanding():
    """A REAL UnderstandingResult, for the one test that exports.

    Export/import runs outputs through `_OUTPUT_CODECS`, which really calls
    `.model_dump()` and `.model_validate()` — a duck-typed stand-in would fail there
    for reasons unrelated to gating, and hand-writing a valid dict would drift from
    the model. Everything else in this file stays duck-typed on purpose.
    """
    return UnderstandingResult(
        incident_id="INC-G",
        analysis=IncidentAnalysis(
            incident_summary="gate test",
            severity="5",
            severity_reasoning="n/a",
            impact_assessment="n/a",
            key_investigation_areas=[],
            log_sources_to_review=["src_a"],
            initial_hypotheses=[],
            recommended_actions=[],
            stakeholder_notification=[],
            extracted_entities=[ExtractedEntity(type="org_unit", value="GGG1H08QR")],
            event_time=EventWindow(start="2026-07-01", end="2026-07-31"),
            correlation_keys=["org_unit_id"],
        ),
    )


class _Correlation:
    """Duck-typed CorrelationResult stand-in (the scorer and summarizer both getattr)."""

    def __init__(self, record_count=0):
        self.record_count = record_count
        self.aggregations = {}
        self.findings = []
        self.transform_plan = None
        self.evidence = None
        self.verdict = None
        self.brief = None


class _Query:
    target_log_source = "src_a"
    natural_language_query = "everything"
    date_from = "2026-07-01"
    date_to = "2026-07-31"


async def _healthy_stage(ctx):
    return _Understanding()
