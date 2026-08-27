"""
Tests for the run queue (src/job_queue.py) and its seam in the job manager.

Two halves, split the way the code is: the arithmetic holds ids only and is tested
with no pipeline at all, then the manager tests assert the four places a slot changes
hands — a loop exiting, a run parking on a human, an operator acting on a queued job,
and a restart.
"""

import asyncio

import pytest

from src.job_queue import (DEFAULT_MAX_CONCURRENT, DEFAULT_MAX_QUEUED,
                           MAX_CONCURRENT_CEILING, MAX_QUEUED_CEILING,
                           JobQueue, QueueFull)
from src.notifications import EventEmitter
from src.pipeline_runner import (JobManager, JobRunMode, JobStatus,
                                 StageDescriptor)


def _queue(width=2, max_queued=256):
    return JobQueue(
        {"jobs": {"max_concurrent_jobs": width, "max_queued_jobs": max_queued}}
    )


# --- the arithmetic --------------------------------------------------------


def test_the_backlog_drains_in_the_order_it_filled():
    """FIFO is not a preference here: `position` is the only thing a caller holding a
    job id can act on, and a queue that reorders reports a position that is a lie."""
    q = _queue(width=1)

    assert q.admit("a") is True
    for job_id in ("b", "c", "d"):
        assert q.admit(job_id) is False
    assert q.waiting() == ["b", "c", "d"]
    assert [q.position(j) for j in ("b", "c", "d")] == [1, 2, 3]

    assert q.release("a") == ["b"]
    assert q.release("b") == ["c"]
    assert q.release("c") == ["d"]
    assert q.waiting() == []


def test_a_full_backlog_is_refused_with_the_numbers_that_refused_it():
    """A job id for a run that may never start reads exactly like a run that is merely
    slow, so the submission is refused — and the refusal carries the depth and the limit,
    because "the backlog is full" must be distinguishable from any other server error."""
    q = _queue(width=1, max_queued=2)
    q.admit("running")
    q.admit("wait-1")
    q.admit("wait-2")

    with pytest.raises(QueueFull) as exc:
        q.admit("wait-3")

    assert exc.value.depth == 2
    assert exc.value.limit == 2
    assert "2" in str(exc.value)
    assert q.stats()["refused"] == 1
    assert "wait-3" not in q.waiting()


def test_releasing_an_id_that_holds_no_slot_frees_nothing():
    """Idempotence is the whole reason this returns a list rather than a bool.

    A slot is released both when a run's stage loop exits and when it parks on a gate,
    and the same run does both. Meanwhile a resumed run, a stage retry and a link lane's
    child all release without ever having been admitted — if any of those freed a slot,
    each would start a queued job and put the width over.
    """
    q = _queue(width=1)
    q.admit("a")
    q.admit("b")

    assert q.release("never-admitted") == []
    assert q.running() == ["a"]
    assert q.waiting() == ["b"]

    assert q.release("a") == ["b"]
    assert q.release("a") == []  # the same run releasing twice
    assert q.running() == ["b"]


def test_release_drains_up_to_the_width_and_no_further():
    q = _queue(width=3)
    for job_id in ("a", "b", "c", "d", "e"):
        q.admit(job_id)
    assert q.running() == ["a", "b", "c"]

    assert q.release("a") == ["d"]
    assert len(q.running()) == 3
    assert q.waiting() == ["e"]


def test_withdraw_takes_a_job_out_of_the_backlog_and_frees_no_slot():
    """A cancel of a RUNNING job releases through `release` when its loop unwinds, so
    freeing a slot here as well would admit two jobs for one departure."""
    q = _queue(width=1)
    q.admit("running")
    q.admit("waiting")

    assert q.withdraw("waiting") is True
    assert q.withdraw("waiting") is False
    assert q.withdraw("running") is False

    assert q.running() == ["running"]
    assert q.waiting() == []
    assert q.stats()["withdrawn"] == 1


def test_a_resubmission_is_a_duplicate_request_and_not_a_second_run():
    q = _queue(width=1)
    assert q.admit("a") is True
    assert q.admit("a") is True
    assert q.admit("b") is False
    assert q.admit("b") is False
    assert q.running() == ["a"]
    assert q.waiting() == ["b"]
    assert q.stats()["admitted"] == 1
    assert q.stats()["queued_total"] == 1


def test_position_is_zero_for_a_running_or_unknown_job():
    """Zero rather than None because this rides a job-list row: "queued, 4th" and
    "running" are both answers, and an absent field is neither."""
    q = _queue(width=1)
    q.admit("running")
    q.admit("waiting")

    assert q.position("running") == 0
    assert q.position("nobody") == 0
    assert q.position("waiting") == 1


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, DEFAULT_MAX_CONCURRENT),
        (True, DEFAULT_MAX_CONCURRENT),
        ("four", DEFAULT_MAX_CONCURRENT),
        (0, 1),
        (-5, 1),
        (MAX_CONCURRENT_CEILING + 100, MAX_CONCURRENT_CEILING),
        ("3", 3),
    ],
)
def test_an_unreadable_bound_falls_back_and_never_widens(value, expected):
    """A typo must not become the most permissive setting available: an unparseable
    width reads as the default, and a number outside the range is clamped into it."""
    assert JobQueue({"jobs": {"max_concurrent_jobs": value}}).width() == expected


def test_the_backlog_bound_is_clamped_the_same_way():
    assert JobQueue({}).max_queued() == DEFAULT_MAX_QUEUED
    assert JobQueue({"jobs": {}}).max_queued() == DEFAULT_MAX_QUEUED
    assert (
        JobQueue({"jobs": {"max_queued_jobs": MAX_QUEUED_CEILING * 2}}).max_queued()
        == MAX_QUEUED_CEILING
    )


def test_the_bounds_are_re_read_on_every_admission():
    """Held by reference, not copied: an operator raising the width in the Configuration
    tab affects the next admission rather than the next restart."""
    config = {"jobs": {"max_concurrent_jobs": 1}}
    q = JobQueue(config)
    q.admit("a")
    assert q.admit("b") is False

    config["jobs"]["max_concurrent_jobs"] = 3
    assert q.admit("c") is True
    assert q.width() == 3


def test_stats_reports_both_bounds_beside_the_counters():
    q = _queue(width=1, max_queued=1)
    q.admit("a")
    q.admit("b")
    with pytest.raises(QueueFull):
        q.admit("c")
    q.release("a")

    stats = q.stats()
    assert stats == {
        "width": 1,
        "max_queued": 1,
        "running": 1,
        "queued": 0,
        "admitted": 2,
        "queued_total": 1,
        "refused": 1,
        "withdrawn": 0,
    }


# --- the manager seam ------------------------------------------------------


def _manager(stages, width=1, max_queued=256, **kwargs):
    emitter = EventEmitter()
    config = {
        "jobs": {"max_concurrent_jobs": width, "max_queued_jobs": max_queued}
    }
    jm = JobManager(stages, emitter, config=config, **kwargs)
    emitter.set_job_manager(jm)
    return jm


def _incident(n=1):
    return {
        "id": f"INC-{n}",
        "description": "test",
        "timestamp": "2024-01-01T00:00",
    }


async def _settle(predicate, timeout=2.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


async def test_a_submission_past_the_width_is_queued_and_starts_when_a_slot_frees():
    ran = []

    async def stage(ctx):
        ran.append(ctx.incident["id"])
        return "ok"

    jm = _manager([StageDescriptor("a", stage, "oa")], width=1)
    first = jm.create_job(_incident(1))
    second = jm.create_job(_incident(2))

    assert jm.submit_job(first) is True
    assert jm.submit_job(second) is False
    assert second.status == JobStatus.QUEUED
    assert jm.queue.position(second.job_id) == 1

    assert await _settle(lambda: second.status == JobStatus.COMPLETED)
    assert ran == ["INC-1", "INC-2"]
    assert jm.queue.running() == []


async def test_a_run_parked_on_a_human_gives_its_slot_back():
    """A gate can stay open for days and a pause has no clock at all, so a parked run
    must not hold a slot the backlog could be draining into. What resumes it is not
    re-queued: the human has already decided."""
    ran = []

    async def stage(ctx):
        ran.append(ctx.incident["id"])
        return "ok"

    jm = _manager([StageDescriptor("a", stage, "oa")], width=1)
    parked = jm.create_job(_incident(1), run_mode=JobRunMode.STEP)
    queued = jm.create_job(_incident(2))

    jm.submit_job(parked)
    assert jm.submit_job(queued) is False

    assert await _settle(lambda: queued.status == JobStatus.COMPLETED)
    assert parked.status == JobStatus.PAUSED
    assert ran == ["INC-2"]
    assert jm.queue.running() == []

    # Resuming re-enters the run outside the width; it is not sent back to the backlog.
    jm.control(parked.job_id, "step")
    assert await _settle(lambda: parked.status == JobStatus.COMPLETED)
    assert parked.job_id not in jm.queue.waiting()
    assert ran == ["INC-2", "INC-1"]


async def test_an_operator_acting_on_a_queued_run_takes_it_out_of_the_backlog():
    """Otherwise the drain later starts a run that was cancelled, or starts a second
    time one the operator already resumed."""
    started = asyncio.Event()

    async def slow(ctx):
        started.set()
        await asyncio.sleep(100)

    jm = _manager([StageDescriptor("slow", slow, "s")], width=1)
    running = jm.create_job(_incident(1))
    queued = jm.create_job(_incident(2))
    jm.submit_job(running)
    jm.submit_job(queued)
    await asyncio.wait_for(started.wait(), timeout=1)

    jm.control(queued.job_id, "cancel_all")

    assert queued.status == JobStatus.CANCELLED
    assert jm.queue.waiting() == []
    assert [i["action"] for i in queued.interventions] == ["dequeued", "cancel_all"]

    # And the slot the running job holds is not handed to the job that just left.
    jm.control(running.job_id, "cancel_all")
    assert await _settle(lambda: running.status == JobStatus.CANCELLED)
    assert jm.queue.running() == []


async def test_skipping_the_last_stage_of_a_running_job_hands_its_slot_on():
    """That branch retires the loop with `_run_gen += 1` rather than letting it exit, so
    the loop's own `finally` sees a stale generation and hands nothing back. Without an
    explicit release the slot is lost for the life of the process."""
    started = asyncio.Event()

    async def slow(ctx):
        started.set()
        await asyncio.sleep(100)

    jm = _manager([StageDescriptor("slow", slow, "s")], width=1)
    first = jm.create_job(_incident(1))
    second = jm.create_job(_incident(2))
    jm.submit_job(first)
    jm.submit_job(second)
    await asyncio.wait_for(started.wait(), timeout=1)

    jm.control(first.job_id, "skip_stage", "slow")

    assert first.status == JobStatus.COMPLETED
    assert await _settle(lambda: second.status != JobStatus.QUEUED)
    assert first.job_id not in jm.queue.running()
    jm.control(second.job_id, "cancel_all")


async def test_a_restored_queued_job_goes_back_to_the_backlog_and_not_to_a_pause():
    """A queued run has done nothing, so the honest restore is the backlog it was in —
    not a `paused` an operator has to notice and clear."""

    async def stage(ctx):
        return "ok"

    jm = _manager([StageDescriptor("a", stage, "oa")], width=1)
    first = jm.create_job(_incident(1))
    second = jm.create_job(_incident(2))
    third = jm.create_job(_incident(3))
    for job in (first, second, third):
        job.status = JobStatus.QUEUED

    assert jm.resume_queued() == 3
    assert await _settle(
        lambda: all(
            j.status == JobStatus.COMPLETED for j in (first, second, third)
        )
    )


async def test_resume_queued_stops_at_a_bound_that_was_lowered_between_runs(caplog):
    async def stage(ctx):
        await asyncio.sleep(100)

    jm = _manager([StageDescriptor("a", stage, "oa")], width=1, max_queued=1)
    jobs = [jm.create_job(_incident(n)) for n in range(1, 5)]
    for job in jobs:
        job.status = JobStatus.QUEUED

    with caplog.at_level("ERROR"):
        assert jm.resume_queued() == 2  # one started, one queued, the rest refused
    assert "max_queued_jobs" in caplog.text
    assert jobs[2].status == JobStatus.QUEUED
    for job in jobs:
        jm.control(job.job_id, "cancel_all")


# --- batches ---------------------------------------------------------------


async def test_a_batch_is_a_label_on_its_jobs_and_never_a_second_record():
    """So it round-trips through export/import for free, a restart rebuilds it from the
    jobs themselves, and it can never disagree with them about what happened."""

    async def stage(ctx):
        return "ok"

    jm = _manager([StageDescriptor("a", stage, "oa")], width=1)
    result = jm.submit_batch([_incident(1), _incident(2), _incident(3)])

    batch_id = result["batch_id"]
    assert len(result["accepted"]) == 3
    assert result["refused"] == []
    assert [row["queue_position"] for row in result["accepted"]] == [0, 1, 2]

    jobs = jm.batch_jobs(batch_id)
    assert [j.incident["batch_id"] for j in jobs] == [batch_id] * 3
    assert [row["batch_id"] for row in jm.list_jobs()] == [batch_id] * 3

    status = jm.batch_status(batch_id)
    assert status["total"] == 3
    assert sum(status["counts"].values()) == 3

    assert await _settle(lambda: jm.batch_status(batch_id)["done"] == 3)
    assert [r["batch_id"] for r in jm.list_batches()] == [batch_id]

    # The label is derived, so a batch nobody submitted has no record to find.
    with pytest.raises(KeyError):
        jm.batch_status("no-such-batch")


async def test_a_batch_whose_tail_is_refused_keeps_the_jobs_already_accepted():
    """Partial by design: a batch whose last incident exceeds the backlog bound must not
    discard the ones already admitted — and a job id for a run nobody will start is worse
    than no job id, so the refused ones are dropped rather than parked."""
    started = asyncio.Event()

    async def slow(ctx):
        started.set()
        await asyncio.sleep(100)

    jm = _manager([StageDescriptor("slow", slow, "s")], width=1, max_queued=1)
    result = jm.submit_batch([_incident(n) for n in range(1, 5)])

    assert len(result["accepted"]) == 2
    assert len(result["refused"]) == 2
    assert [r["incident_id"] for r in result["refused"]] == ["INC-3", "INC-4"]
    assert result["refused"][0]["depth"] == 1
    assert result["refused"][0]["limit"] == 1
    assert result["queue"]["refused"] == 2

    # Only the accepted ones exist as jobs.
    assert len(jm.batch_jobs(result["batch_id"])) == 2

    await asyncio.wait_for(started.wait(), timeout=1)
    cancel = jm.cancel_batch(result["batch_id"])
    assert len(cancel["cancelled"]) == 2
    assert await _settle(lambda: jm.batch_status(result["batch_id"])["done"] == 2)


async def test_cancelling_a_batch_counts_a_race_as_already_finished():
    """Cancelling a backlog is a bulk action; one job ending between the read and the
    cancel must not refuse the other 299."""

    async def stage(ctx):
        return "ok"

    jm = _manager([StageDescriptor("a", stage, "oa")], width=2)
    result = jm.submit_batch([_incident(1), _incident(2)])
    batch_id = result["batch_id"]
    assert await _settle(lambda: jm.batch_status(batch_id)["done"] == 2)

    cancel = jm.cancel_batch(batch_id)
    assert cancel["cancelled"] == []
    assert len(cancel["already_finished"]) == 2

    with pytest.raises(KeyError):
        jm.cancel_batch("no-such-batch")


async def test_a_batch_run_mode_is_resolved_once_for_every_job_in_it():
    async def stage(ctx):
        return "ok"

    jm = _manager([StageDescriptor("a", stage, "oa")], width=2)
    result = jm.submit_batch([_incident(1), _incident(2)], run_mode="supervised")
    jobs = jm.batch_jobs(result["batch_id"])

    assert [j.run_mode for j in jobs] == [JobRunMode.SUPERVISED] * 2
    for job in jobs:
        jm.control(job.job_id, "cancel_all")
