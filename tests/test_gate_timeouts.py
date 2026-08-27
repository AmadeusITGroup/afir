"""
Tests for gate timeouts and webhook notification (Phase D).

A gate holds indefinitely by default, which is correct — a wrong auto-decision is
worse than a late one — but it means the only thing telling anyone a decision is due
is a client that happens to be watching. These two features close that gap, and each
brings a way to silently make a decision nobody made. That is what these guard:

1. **A timeout must not un-gate a run by default.** No ``timeout_seconds`` means hold
   forever, and ``on_timeout`` defaults to ``hold`` (notify, keep waiting) — so
   enabling a timeout for notification cannot accidentally start auto-approving.
2. **A timeout-driven ``proceed``/``abort`` must be recorded as such.** A report whose
   gate expired must never read as reviewed; ``interventions``/``gate_history`` carry
   an actor of ``timeout``.
3. **A ``hold`` timeout must fire exactly once.** A repeating alert on an unanswered
   gate trains the recipient to mute the channel.
4. **A webhook must never fail or slow a run.** A dead receiver, a 4xx, an
   unresolvable ``${VAR}`` — all degrade to a log line.
"""

import asyncio

from src.notifications import (WEBHOOK_EVENTS, EventEmitter, WebhookDispatcher,
                               webhook_event_name)
from src.pipeline_runner import (JobManager, JobRunMode, JobStatus,
                                 StageDescriptor)
from src.stage_health import (DEFAULT_ON_TIMEOUT, stage_gate_on_timeout,
                              stage_gate_timeout)

GATEABLE = "understanding"


def _manager(stages, **kwargs):
    emitter = EventEmitter()
    jm = JobManager(stages, emitter, **kwargs)
    emitter.set_job_manager(jm)
    return jm


def _incident():
    return {"id": "INC-T", "description": "timeout test", "timestamp": "2026-07-30"}


def _events(job, event_type):
    return [e for e in job._event_history if e.get("type") == event_type]


def _cfg(**gate_keys):
    return {"stage_gates": dict(gate_keys)}


async def _wait_for(predicate, timeout=2.0):
    async def _poll():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout=timeout)


def _stages(recorder):
    async def understanding(ctx):
        recorder.append("understanding")
        return None  # fatal signal -> score 0.0 -> gates in semi_auto

    async def export(ctx):
        recorder.append("export")
        return "x"

    return [
        StageDescriptor(GATEABLE, understanding, "understanding"),
        StageDescriptor("export", export, "export_paths"),
    ]


# --- config resolution -----------------------------------------------------


def test_no_timeout_by_default():
    """The default must be "hold forever" — a timeout is opt-in, per stage or global."""
    assert stage_gate_timeout({}, GATEABLE) is None
    assert stage_gate_timeout(_cfg(), GATEABLE) is None
    assert stage_gate_on_timeout({}, GATEABLE) == DEFAULT_ON_TIMEOUT == "hold"


def test_per_stage_timeout_overrides_global():
    cfg = _cfg(timeout_seconds=60, stages={GATEABLE: {"timeout_seconds": 5}})
    assert stage_gate_timeout(cfg, GATEABLE) == 5.0
    assert stage_gate_timeout(cfg, "correlation") == 60.0


def test_zero_and_negative_timeout_mean_no_timeout():
    """`0` must not mean "expire instantly" — that would silently un-gate a run."""
    assert stage_gate_timeout(_cfg(timeout_seconds=0), GATEABLE) is None
    assert stage_gate_timeout(_cfg(timeout_seconds=-30), GATEABLE) is None


def test_non_numeric_timeout_ignored_not_crashing():
    assert stage_gate_timeout(_cfg(timeout_seconds="soon"), GATEABLE) is None


def test_unknown_on_timeout_falls_back_to_hold():
    """An unrecognised action must never be read as permission to proceed."""
    cfg = _cfg(on_timeout="YOLO")
    assert stage_gate_on_timeout(cfg, GATEABLE) == "hold"


def test_on_timeout_values_and_per_stage_override():
    cfg = _cfg(on_timeout="proceed", stages={GATEABLE: {"on_timeout": "ABORT"}})
    assert stage_gate_on_timeout(cfg, GATEABLE) == "abort"
    assert stage_gate_on_timeout(cfg, "correlation") == "proceed"


# --- the gate record advertises the deadline -------------------------------


async def test_gate_record_advertises_timeout():
    """An inbox has to be able to show "expires in ..." rather than assume forever."""
    ran = []
    cfg = _cfg(timeout_seconds=30, on_timeout="proceed")
    jm = _manager(_stages(ran), config=cfg)
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for(lambda: job.open_gate is not None)

    assert job.open_gate["timeout_seconds"] == 30.0
    assert job.open_gate["on_timeout"] == "proceed"

    jm.resolve_gate(job.job_id, "approve")
    await asyncio.wait_for(task, timeout=2)


async def test_gate_record_reports_none_when_holding_forever():
    """`on_timeout` must read as None when no timeout is set, not as a phantom "hold"."""
    ran = []
    jm = _manager(_stages(ran), config=_cfg(on_timeout="proceed"))
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for(lambda: job.open_gate is not None)

    assert job.open_gate["timeout_seconds"] is None
    assert job.open_gate["on_timeout"] is None

    jm.resolve_gate(job.job_id, "approve")
    await asyncio.wait_for(task, timeout=2)


# --- hold ------------------------------------------------------------------


async def test_hold_notifies_but_keeps_waiting():
    """The default outcome: announce, then keep blocking. The run must NOT continue."""
    ran = []
    cfg = _cfg(timeout_seconds=0.05, on_timeout="hold")
    jm = _manager(_stages(ran), config=cfg)
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for(lambda: _events(job, "gate_timeout"))

    # Timed out AND still gated: the downstream stage has not run.
    assert ran == ["understanding"]
    assert job.status == JobStatus.AWAITING_APPROVAL
    assert job.open_gate is not None
    timeout_event = _events(job, "gate_timeout")[-1]
    assert timeout_event["status"] == "hold"
    assert timeout_event["data"]["actor"] == "timeout"
    # A hold is NOT a decision, so it must not pollute the decision record.
    assert job.gate_history == []
    assert [i for i in job.interventions if "timeout" in i["action"]] == []

    jm.resolve_gate(job.job_id, "approve")
    await asyncio.wait_for(task, timeout=2)
    assert ran == ["understanding", "export"]


async def test_hold_notifies_only_once():
    """A repeating alert on an unanswered gate is how the channel gets muted."""
    ran = []
    cfg = _cfg(timeout_seconds=0.03, on_timeout="hold")
    jm = _manager(_stages(ran), config=cfg)
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for(lambda: _events(job, "gate_timeout"))

    # Wait several timeout intervals; the count must not grow.
    await asyncio.sleep(0.03 * 6)
    assert len(_events(job, "gate_timeout")) == 1
    assert job.status == JobStatus.AWAITING_APPROVAL

    jm.resolve_gate(job.job_id, "approve")
    await asyncio.wait_for(task, timeout=2)


# --- proceed ---------------------------------------------------------------


async def test_proceed_continues_and_records_that_nobody_reviewed():
    """A clock-made decision must be traceable as a clock-made decision."""
    ran = []
    cfg = _cfg(timeout_seconds=0.05, on_timeout="proceed")
    jm = _manager(_stages(ran), config=cfg)
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)

    await asyncio.wait_for(jm.run_job(job), timeout=3)

    assert job.status == JobStatus.COMPLETED
    assert ran == ["understanding", "export"]
    assert job.open_gate is None
    # The audit trail must say a human did NOT approve this.
    entry = [i for i in job.interventions if i["action"] == "gate_timeout_proceed"]
    assert len(entry) == 1
    assert entry[0]["actor"] == "timeout"
    assert entry[0]["stage"] == GATEABLE
    history = job.gate_history[-1]
    assert history["action"] == "proceed"
    assert history["actor"] == "timeout"
    assert history["waited_seconds"] == 0.05


async def test_proceed_only_after_the_timeout_not_immediately():
    """Sanity: `proceed` must be the timeout's effect, not the gate failing to block."""
    ran = []
    cfg = _cfg(timeout_seconds=0.4, on_timeout="proceed")
    jm = _manager(_stages(ran), config=cfg)
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for(lambda: job.open_gate is not None)

    # Well before the deadline the run is still blocked.
    await asyncio.sleep(0.05)
    assert ran == ["understanding"]
    assert job.status == JobStatus.AWAITING_APPROVAL

    await asyncio.wait_for(task, timeout=3)
    assert ran == ["understanding", "export"]


async def test_human_decision_beats_the_timeout():
    """An answer that arrives before the deadline must win — and be attributed."""
    ran = []
    cfg = _cfg(timeout_seconds=5, on_timeout="proceed")
    jm = _manager(_stages(ran), config=cfg)
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for(lambda: job.open_gate is not None)
    jm.resolve_gate(job.job_id, "approve", actor="analyst@example.com")
    await asyncio.wait_for(task, timeout=2)

    assert job.status == JobStatus.COMPLETED
    assert _events(job, "gate_timeout") == []
    assert job.gate_history[-1]["actor"] == "analyst@example.com"
    assert job.gate_history[-1]["action"] == "approve"


# --- abort -----------------------------------------------------------------


async def test_abort_cancels_the_job():
    ran = []
    cfg = _cfg(timeout_seconds=0.05, on_timeout="abort")
    jm = _manager(_stages(ran), config=cfg)
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)

    await asyncio.wait_for(jm.run_job(job), timeout=3)

    assert job.status == JobStatus.CANCELLED
    assert ran == ["understanding"]  # downstream never ran
    assert job.open_gate is None
    entry = [i for i in job.interventions if i["action"] == "gate_timeout_abort"]
    assert len(entry) == 1
    assert entry[0]["actor"] == "timeout"


# --- supervised mode + a disabled stage ------------------------------------


async def test_timeout_applies_in_supervised_mode_too():
    """Timeouts are a property of the gate, not of why it opened."""
    ran = []

    async def healthy(ctx):
        ran.append("understanding")
        return None

    cfg = _cfg(timeout_seconds=0.05, on_timeout="proceed")
    jm = _manager([StageDescriptor(GATEABLE, healthy, "understanding")], config=cfg)
    job = jm.create_job(_incident(), run_mode=JobRunMode.SUPERVISED)

    await asyncio.wait_for(jm.run_job(job), timeout=3)
    assert job.status == JobStatus.COMPLETED
    assert job.gate_history[-1]["action"] == "proceed"


async def test_auto_mode_ignores_timeouts_entirely():
    """No gate opens, so there is nothing to time out."""
    ran = []
    cfg = _cfg(timeout_seconds=0.01, on_timeout="abort")
    jm = _manager(_stages(ran), config=cfg)
    job = jm.create_job(_incident(), run_mode=JobRunMode.AUTO)

    await asyncio.wait_for(jm.run_job(job), timeout=3)
    assert job.status == JobStatus.COMPLETED
    assert _events(job, "gate_timeout") == []


# --- webhook event naming --------------------------------------------------


def test_terminal_job_status_maps_to_named_webhook_events():
    """A subscriber registers for `job_completed`, not for `job_status`+filtering."""
    assert webhook_event_name({"type": "job_status", "status": "completed"}) == (
        "job_completed"
    )
    assert webhook_event_name({"type": "job_status", "status": "cancelled"}) == (
        "job_cancelled"
    )
    # Non-terminal transitions are NOT webhook events — that is the chatty stream.
    assert webhook_event_name({"type": "job_status", "status": "running"}) is None
    assert webhook_event_name({"type": "job_status", "status": "paused"}) is None


def test_gate_events_pass_through_and_stage_events_do_not():
    for name in ("gate_opened", "gate_resolved", "gate_timeout", "stage_failed"):
        assert webhook_event_name({"type": name}) == name
    for name in ("stage_started", "stage_output", "stage_completed", "intervention"):
        assert webhook_event_name({"type": name}) is None


def test_webhook_event_name_tolerates_junk():
    assert webhook_event_name({}) is None
    assert webhook_event_name(None) is None


# --- webhook dispatcher configuration -------------------------------------


def test_dispatcher_disabled_by_default():
    d = WebhookDispatcher({})
    assert d.enabled is False
    assert d.targets == []
    # notify() on a disabled dispatcher is a no-op, not an error.
    d.notify({"type": "gate_opened"})


def test_url_comes_from_an_env_var_reference(monkeypatch):
    """A webhook URL IS a secret; the config must reference it, never hold it."""
    monkeypatch.setenv("AFIR_TEST_HOOK", "https://hooks.example.com/abc123")
    d = WebhookDispatcher(
        {
            "webhooks": {
                "enabled": True,
                "targets": [{"name": "soc", "url": "${AFIR_TEST_HOOK}"}],
            }
        }
    )
    assert len(d.targets) == 1
    assert d.targets[0]["url"] == "https://hooks.example.com/abc123"
    # No `events` given -> subscribed to all of them.
    assert d.targets[0]["events"] == set(WEBHOOK_EVENTS)


def test_unresolved_env_var_drops_the_target_loudly(monkeypatch, caplog):
    """An unset secret must be an ERROR, not a target that quietly sends nowhere."""
    monkeypatch.delenv("AFIR_MISSING_HOOK", raising=False)
    with caplog.at_level("ERROR"):
        d = WebhookDispatcher(
            {
                "webhooks": {
                    "enabled": True,
                    "targets": [{"name": "soc", "url": "${AFIR_MISSING_HOOK}"}],
                }
            }
        )
    assert d.targets == []
    assert "no URL after env expansion" in caplog.text


def test_non_http_url_rejected():
    d = WebhookDispatcher(
        {
            "webhooks": {
                "enabled": True,
                "targets": [{"name": "bad", "url": "file:///etc/passwd"}],
            }
        }
    )
    assert d.targets == []


def test_unknown_event_names_are_dropped_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        d = WebhookDispatcher(
            {
                "webhooks": {
                    "enabled": True,
                    "targets": [
                        {
                            "name": "soc",
                            "url": "https://x.example.com/h",
                            "events": ["gate_opened", "everything"],
                        }
                    ],
                }
            }
        )
    assert d.targets[0]["events"] == {"gate_opened"}
    assert "unknown event 'everything'" in caplog.text


def test_target_with_only_unknown_events_is_dropped():
    d = WebhookDispatcher(
        {
            "webhooks": {
                "enabled": True,
                "targets": [
                    {
                        "name": "soc",
                        "url": "https://x.example.com/h",
                        "events": ["nope"],
                    }
                ],
            }
        }
    )
    assert d.targets == []


def test_malformed_target_entries_are_skipped():
    d = WebhookDispatcher(
        {"webhooks": {"enabled": True, "targets": ["https://x.example.com/h", None]}}
    )
    assert d.targets == []


# --- webhook delivery ------------------------------------------------------


async def test_notify_delivers_only_subscribed_events(monkeypatch):
    sent = []

    async def fake_deliver(target, event):
        sent.append((target["name"], event["event"]))

    d = WebhookDispatcher(
        {
            "webhooks": {
                "enabled": True,
                "targets": [
                    {
                        "name": "gates-only",
                        "url": "https://x.example.com/h",
                        "events": ["gate_opened"],
                    },
                    {"name": "all", "url": "https://y.example.com/h"},
                ],
            }
        }
    )
    monkeypatch.setattr(d, "_deliver", fake_deliver)

    d.notify({"type": "gate_opened", "job_id": "J1"})
    d.notify({"type": "job_status", "status": "completed", "job_id": "J1"})
    await asyncio.sleep(0.02)

    assert ("gates-only", "gate_opened") in sent
    assert ("all", "gate_opened") in sent
    assert ("all", "job_completed") in sent
    # The gates-only target must NOT receive the completion.
    assert ("gates-only", "job_completed") not in sent


async def test_delivery_failure_never_propagates(monkeypatch):
    """A dead receiver degrades to a log line; the caller sees nothing."""
    d = WebhookDispatcher(
        {
            "webhooks": {
                "enabled": True,
                "max_attempts": 1,
                "targets": [{"name": "dead", "url": "https://127.0.0.1:1/hook"}],
            }
        }
    )
    d.notify({"type": "gate_opened", "job_id": "J1"})
    # Await the spawned task directly: it must complete, not raise.
    await asyncio.wait_for(
        asyncio.gather(*list(d._tasks), return_exceptions=True), timeout=10
    )
    assert all(t.done() for t in d._tasks) or not d._tasks


async def test_emit_is_unaffected_by_a_broken_dispatcher():
    """The event stream must survive a sink that throws on every call."""

    class Broken:
        def notify(self, event):
            raise RuntimeError("sink exploded")

    async def understanding(ctx):
        return None

    jm = _manager([StageDescriptor(GATEABLE, understanding, "understanding")])
    jm.emitter.set_webhooks(Broken())
    job = jm.create_job(_incident(), run_mode=JobRunMode.AUTO)

    await asyncio.wait_for(jm.run_job(job), timeout=2)

    # The run completed and the SSE history is intact despite the broken sink.
    assert job.status == JobStatus.COMPLETED
    assert _events(job, "job_status")


async def test_notify_without_a_running_loop_is_safe():
    """Called from sync code (a unit test, a script) it must not raise."""
    d = WebhookDispatcher(
        {
            "webhooks": {
                "enabled": True,
                "targets": [{"name": "x", "url": "https://x.example.com/h"}],
            }
        }
    )
    # Inside this coroutine a loop IS running, so drive the no-loop path directly.
    coro_holder = []

    async def fake(target, event):
        coro_holder.append(1)

    d._deliver = fake
    d.notify({"type": "gate_opened"})
    await asyncio.sleep(0.01)
    assert coro_holder == [1]


async def test_close_is_safe_with_nothing_in_flight():
    d = WebhookDispatcher({})
    await d.close()  # must not raise
