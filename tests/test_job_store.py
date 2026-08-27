"""
Tests for durable job persistence (src/job_store.py + JobManager.restore).

The regressions these guard:

1. **A pending approval must survive a restart.** That is the entire reason the store
   exists — a gate holds indefinitely, so the process will be restarted under it.
2. **A restored job must not lie about its state.** A job that was RUNNING or
   AWAITING_APPROVAL when the process died has no task and no answerable gate; coming
   back in either status would present a run that is waiting on nothing.
3. **A dropped evidence sidecar must be visible.** Resuming on silently-empty logs
   would build a report on no data and present it as though it were built on data.
4. **A storage failure must not fail a run.** Losing durability must not lose the
   investigation.
"""

import json

import pytest

from src.job_store import DEFAULT_MAX_EVIDENCE_BYTES, JobStore, build_job_store
from src.notifications import EventEmitter
from src.pipeline_runner import (JobManager, JobRunMode, JobStatus,
                                 StageDescriptor, StageStatus)


def _manager(stages, store=None, **kwargs):
    emitter = EventEmitter()
    jm = JobManager(stages, emitter, store=store, **kwargs)
    emitter.set_job_manager(jm)
    return jm


def _incident(iid="INC-P"):
    return {"id": iid, "description": "persist test", "timestamp": "2026-07-30T00:00"}


def _store(tmp_path, **kwargs):
    return JobStore(base_dir=tmp_path / "jobs", **kwargs)


async def _wait_for_gate(job, timeout=2.0):
    import asyncio

    async def _poll():
        while job.open_gate is None:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout=timeout)
    return job.open_gate


# --- the core guarantee ----------------------------------------------------


async def test_a_job_parked_on_a_gate_is_restored_after_a_restart(tmp_path):
    """The reason this module exists: the analyst's queue must survive a deploy."""
    import asyncio

    async def understanding(ctx):
        return None  # scores 0.0 -> gates in semi_auto

    store = _store(tmp_path)
    jm = _manager(
        [StageDescriptor("understanding", understanding, "understanding")], store=store
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)

    # Simulate the process dying under the open gate.
    jm.control(job.job_id, "cancel_all")
    await asyncio.wait_for(task, timeout=2)

    fresh_store = _store(tmp_path)
    jm2 = _manager(
        [StageDescriptor("understanding", understanding, "understanding")],
        store=fresh_store,
    )
    assert jm2.restore() == 1
    revived = jm2.get_job(job.job_id)
    assert revived is not None, "the job survived the restart"
    assert revived.incident["id"] == "INC-P"


async def test_shutdown_under_a_gate_does_not_persist_the_job_as_cancelled(tmp_path):
    """A graceful shutdown must not destroy the pending approval.

    Found by a real restart, not by a unit test: the outer CancelledError handler
    overwrote AWAITING_APPROVAL with CANCELLED and persisted THAT, so the analyst's
    queue came back empty. Only an explicit operator cancel really cancels.
    """
    import asyncio

    async def understanding(ctx):
        return None

    store = _store(tmp_path)
    jm = _manager(
        [StageDescriptor("understanding", understanding, "understanding")], store=store
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)

    task.cancel()  # process shutting down, NOT an operator cancel
    with pytest.raises(asyncio.CancelledError):
        await task

    assert job.status == JobStatus.AWAITING_APPROVAL, "the gate state is preserved"
    persisted = json.loads((tmp_path / "jobs" / f"{job.job_id}.json").read_text())
    assert persisted["status"] == "awaiting_approval"
    assert persisted["open_gate"]["stage"] == "understanding"


async def test_an_operator_cancel_under_a_gate_still_cancels(tmp_path):
    """The other side of the coin: an explicit cancel must not be preserved as pending."""
    import asyncio

    async def understanding(ctx):
        return None

    jm = _manager(
        [StageDescriptor("understanding", understanding, "understanding")],
        store=_store(tmp_path),
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)

    jm.control(job.job_id, "cancel_all")
    await asyncio.wait_for(task, timeout=2)
    assert job.status == JobStatus.CANCELLED


async def test_a_restored_gate_is_rearmed_and_answerable(tmp_path):
    """The whole point: the analyst can still answer a gate opened before the restart."""
    import asyncio

    ran = []

    async def understanding(ctx):
        return None

    async def retrieval(ctx):
        ran.append("retrieval")
        return {"src_a": [{"row": 1}]}

    stages = [
        StageDescriptor("understanding", understanding, "understanding"),
        StageDescriptor("log_retrieval", retrieval, "logs"),
    ]
    jm = _manager(stages, store=_store(tmp_path))
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)
    jid = job.job_id

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    jm2 = _manager(stages, store=_store(tmp_path))
    assert jm2.restore() == 1
    revived = jm2.get_job(jid)
    # Not answerable until re-armed: no runner is waiting to read a decision yet.
    assert revived.open_gate is None and revived.pending_gate is not None

    assert jm2.rearm_gates() == 1
    await asyncio.sleep(0.05)
    assert revived.status == JobStatus.AWAITING_APPROVAL
    inbox = jm2.list_open_gates()
    assert len(inbox) == 1
    assert inbox[0]["stage"] == "understanding"
    # The person answering should be able to see this gate predates a restart.
    assert inbox[0]["reopened_after_restart"] is True

    # Approving must carry the run to completion — not stop at the import-time pause.
    jm2.resolve_gate(jid, "approve", actor="ana")
    await asyncio.sleep(0.2)
    assert revived.status == JobStatus.COMPLETED, "approval resumed the pipeline"
    assert ran == ["retrieval"], "the remaining stages actually ran"
    assert revived.context.outputs["logs"] == {"src_a": [{"row": 1}]}


async def test_a_rearmed_gate_reruns_with_guidance_on_reject(tmp_path):
    """A reject across a restart must still change the retry's prompt."""
    import asyncio

    seen = []

    async def understanding(ctx):
        seen.append(dict(ctx.stage_guidance))
        return None

    stages = [StageDescriptor("understanding", understanding, "understanding")]
    jm = _manager(stages, store=_store(tmp_path))
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)
    jid = job.job_id
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    jm2 = _manager(stages, store=_store(tmp_path))
    jm2.restore()
    jm2.rearm_gates()
    await asyncio.sleep(0.05)

    jm2.resolve_gate(jid, "reject", guidance="check the document leg", actor="ana")
    await asyncio.sleep(0.2)
    assert seen[-1] == {"understanding": ["check the document leg"]}

    # The retry re-gated; close it out so no waiter outlives the test.
    jm2.control(jid, "cancel_all")
    await asyncio.sleep(0.05)


async def test_an_unarmed_pending_gate_still_exports_as_open(tmp_path):
    """A second restart before rearm_gates() must not lose the decision."""
    store = _store(tmp_path)
    jm = _manager(
        [StageDescriptor("understanding", _noop, "understanding")], store=store
    )
    job = jm.create_job(_incident())
    job.pending_gate = {"stage": "understanding", "opened_at": "2026-07-30T00:00"}

    doc = jm.export_job(job.job_id)
    assert doc["open_gate"]["stage"] == "understanding"


async def test_a_rearmed_gate_left_open_is_cancellable(tmp_path):
    """Re-armed waiters must respond to cancel like any other gate — no orphan tasks."""
    import asyncio

    stages = [StageDescriptor("understanding", _noop_none, "understanding")]
    jm = _manager(stages, store=_store(tmp_path))
    job = jm.create_job(_incident(), run_mode=JobRunMode.SEMI_AUTO)
    task = asyncio.ensure_future(jm.run_job(job))
    await _wait_for_gate(job)
    jid = job.job_id
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    jm2 = _manager(stages, store=_store(tmp_path))
    jm2.restore()
    jm2.rearm_gates()
    await asyncio.sleep(0.05)

    jm2.control(jid, "cancel_all")
    await asyncio.sleep(0.1)
    assert jm2.get_job(jid).status == JobStatus.CANCELLED


async def test_a_pending_gate_on_an_unknown_stage_is_dropped(tmp_path):
    """A doc written by a build with different stages must not wedge the restore."""
    jm = _manager(
        [StageDescriptor("understanding", _noop, "understanding")],
        store=_store(tmp_path),
    )
    job = jm.create_job(_incident())
    job.pending_gate = {"stage": "a_stage_that_no_longer_exists"}

    assert jm.rearm_gates() == 0
    assert job.pending_gate is None


async def test_a_restored_job_does_not_claim_to_be_running_or_awaiting(tmp_path):
    """Neither status can be honest without a live task / answerable gate."""
    store = _store(tmp_path)
    jm = _manager(
        [StageDescriptor("understanding", _noop, "understanding")], store=store
    )

    for status in (JobStatus.RUNNING, JobStatus.AWAITING_APPROVAL):
        job = jm.create_job(_incident())
        job.status = status
        job.stage_statuses["understanding"] = StageStatus.RUNNING
        store.save(jm.export_job(job.job_id))

        jm2 = _manager(
            [StageDescriptor("understanding", _noop, "understanding")],
            store=_store(tmp_path),
        )
        jm2.restore()
        revived = jm2.get_job(job.job_id)
        assert revived.status == JobStatus.PAUSED, f"{status} must restore as PAUSED"
        # A stage that was mid-flight recorded no result, so it is pending work.
        assert revived.stage_statuses["understanding"] == StageStatus.PENDING
        store.delete(job.job_id)


async def test_a_never_started_job_stays_pending(tmp_path):
    """PENDING is honest on a rehydrated job; calling it PAUSED would misdescribe it."""
    store = _store(tmp_path)
    jm = _manager(
        [StageDescriptor("understanding", _noop, "understanding")], store=store
    )
    job = jm.create_job(_incident())
    store.save(jm.export_job(job.job_id))

    jm2 = _manager(
        [StageDescriptor("understanding", _noop, "understanding")],
        store=_store(tmp_path),
    )
    jm2.restore()
    assert jm2.get_job(job.job_id).status == JobStatus.PENDING


async def test_gate_history_and_guidance_survive_the_round_trip(tmp_path):
    """A restored job must not re-run a rejected stage with the ORIGINAL prompt."""
    store = _store(tmp_path)
    jm = _manager(
        [StageDescriptor("understanding", _noop, "understanding")], store=store
    )
    job = jm.create_job(_incident())
    job.gate_history.append({"stage": "understanding", "action": "reject"})
    job.context.stage_guidance["understanding"] = ["look at the document"]
    store.save(jm.export_job(job.job_id))

    jm2 = _manager(
        [StageDescriptor("understanding", _noop, "understanding")],
        store=_store(tmp_path),
    )
    jm2.restore()
    revived = jm2.get_job(job.job_id)
    assert revived.gate_history[0]["action"] == "reject"
    assert revived.context.stage_guidance["understanding"] == ["look at the document"]


# --- evidence sidecar ------------------------------------------------------


async def test_evidence_goes_to_a_sidecar_and_is_merged_back(tmp_path):
    store = _store(tmp_path)
    doc = {
        "job_id": "j1",
        "created_at": "2026-07-30T00:00",
        "outputs": {"logs": {"src_a": [{"x": 1}]}, "understanding": None},
    }
    assert store.save(doc, evidence_changed=True)

    main = json.loads((tmp_path / "jobs" / "j1.json").read_text())
    assert "logs" not in main["outputs"], "bulk evidence stays out of the main doc"
    assert (tmp_path / "jobs" / "j1.evidence.json").exists()

    loaded = store.load_all()
    assert loaded[0]["outputs"]["logs"] == {"src_a": [{"x": 1}]}


async def test_oversized_evidence_is_dropped_and_the_omission_is_recorded(tmp_path):
    """Silently resuming on empty logs would present a report built on no data."""
    store = _store(tmp_path, max_evidence_bytes=50)
    doc = {
        "job_id": "j2",
        "created_at": "2026-07-30T00:00",
        "outputs": {"logs": {"src_a": [{"pad": "x" * 500}]}},
    }
    store.save(doc, evidence_changed=True)

    main = json.loads((tmp_path / "jobs" / "j2.json").read_text())
    assert main["evidence_omitted"], "the doc must say its evidence is gone"
    assert "exceeded" in main["evidence_omitted"]
    assert not (tmp_path / "jobs" / "j2.evidence.json").exists()

    loaded = store.load_all()
    assert "logs" not in loaded[0]["outputs"]
    assert loaded[0]["evidence_omitted"]


async def test_an_oversized_save_clears_a_stale_smaller_sidecar(tmp_path):
    """The doc and the sidecar must not disagree about what evidence exists."""
    store = _store(tmp_path, max_evidence_bytes=10_000)
    base = {"job_id": "j3", "created_at": "2026-07-30T00:00"}
    store.save({**base, "outputs": {"logs": {"a": [1]}}}, evidence_changed=True)
    assert (tmp_path / "jobs" / "j3.evidence.json").exists()

    store.max_evidence_bytes = 20
    store.save(
        {**base, "outputs": {"logs": {"a": [{"pad": "y" * 400}]}}},
        evidence_changed=True,
    )
    assert not (tmp_path / "jobs" / "j3.evidence.json").exists()


async def test_evidence_is_not_rewritten_when_unchanged(tmp_path):
    store = _store(tmp_path)
    doc = {
        "job_id": "j4",
        "created_at": "2026-07-30T00:00",
        "outputs": {"logs": {"a": [1]}},
    }
    store.save(doc, evidence_changed=True)
    sidecar = tmp_path / "jobs" / "j4.evidence.json"
    before = sidecar.stat().st_mtime_ns

    store.save(doc, evidence_changed=False)  # a plain status transition
    assert sidecar.stat().st_mtime_ns == before, "sidecar untouched on a metadata save"


# --- write integrity -------------------------------------------------------


async def test_a_write_keeps_the_previous_version(tmp_path):
    store = _store(tmp_path)
    store.save({"job_id": "j5", "created_at": "1", "status": "running"})
    store.save({"job_id": "j5", "created_at": "1", "status": "completed"})

    prev = json.loads((tmp_path / "jobs" / "j5.json.prev").read_text())
    assert prev["status"] == "running"
    current = json.loads((tmp_path / "jobs" / "j5.json").read_text())
    assert current["status"] == "completed"


async def test_a_corrupt_doc_falls_back_to_prev(tmp_path):
    store = _store(tmp_path)
    store.save({"job_id": "j6", "created_at": "1", "status": "running"})
    store.save({"job_id": "j6", "created_at": "1", "status": "completed"})
    (tmp_path / "jobs" / "j6.json").write_text("{ truncated")

    loaded = store.load_all()
    assert len(loaded) == 1 and loaded[0]["status"] == "running"


async def test_one_unreadable_file_does_not_cost_the_whole_queue(tmp_path):
    store = _store(tmp_path)
    store.save({"job_id": "good", "created_at": "1"})
    (tmp_path / "jobs" / "bad.json").write_text("not json at all")

    loaded = store.load_all()
    assert [d["job_id"] for d in loaded] == ["good"]


async def test_a_job_id_cannot_escape_the_store_directory(tmp_path):
    store = _store(tmp_path)
    store.save({"job_id": "../../etc/pwned", "created_at": "1"})
    written = list((tmp_path / "jobs").glob("*.json"))
    assert len(written) == 1
    assert ".." not in written[0].name


async def test_a_doc_without_a_job_id_is_refused(tmp_path):
    store = _store(tmp_path)
    assert store.save({"created_at": "1"}) is False
    assert list((tmp_path / "jobs").glob("*.json")) == []


# --- failure handling ------------------------------------------------------


async def test_a_persistence_failure_does_not_fail_the_run(tmp_path):
    """Losing durability must not lose the investigation."""
    import asyncio

    class _Broken(JobStore):
        def save(self, doc, evidence_changed=False):
            raise OSError("volume is gone")

    ran = []

    async def understanding(ctx):
        ran.append(1)
        return None

    jm = _manager(
        [StageDescriptor("understanding", understanding, "understanding")],
        store=_Broken(base_dir=tmp_path / "jobs"),
    )
    job = jm.create_job(_incident(), run_mode=JobRunMode.AUTO)
    await asyncio.wait_for(jm.run_job(job), timeout=2)

    assert job.status == JobStatus.COMPLETED
    assert ran == [1]


async def test_a_read_only_directory_falls_back_and_says_so(tmp_path, caplog):
    import logging

    target = tmp_path / "ro" / "jobs"
    target.parent.mkdir()
    target.parent.chmod(0o500)  # no write permission
    try:
        store = JobStore(base_dir=target)
        with caplog.at_level(logging.ERROR):
            d = store.directory
        if d is None:
            pytest.skip("no writable fallback in this environment")
        assert store.using_fallback
        # The operator asked for durable storage and is not getting it; a quiet
        # downgrade here is how a pending approval disappears without explanation.
        assert any("LOCAL DISK" in r.message for r in caplog.records)
    finally:
        target.parent.chmod(0o700)


async def test_persistence_disabled_writes_nothing(tmp_path):
    store = JobStore(base_dir=tmp_path / "jobs", enabled=False)
    assert store.save({"job_id": "j7", "created_at": "1"}) is False
    assert store.load_all() == []
    assert not (tmp_path / "jobs").exists()


async def test_a_manager_without_a_store_is_purely_in_memory(tmp_path):
    """The classic blocking endpoints construct no store; that path must be untouched."""
    import asyncio

    jm = _manager([StageDescriptor("understanding", _noop, "understanding")])
    job = jm.create_job(_incident())
    await asyncio.wait_for(jm.run_job(job), timeout=2)
    assert job.status == JobStatus.COMPLETED
    assert jm.restore() == 0


# --- retention -------------------------------------------------------------


async def test_prune_removes_files_past_retention(tmp_path):
    import os
    import time

    store = _store(tmp_path, retention_days=1)
    store.save({"job_id": "old", "created_at": "1"}, evidence_changed=True)
    store.save({"job_id": "new", "created_at": "2"})

    old = tmp_path / "jobs" / "old.json"
    ancient = time.time() - (3 * 86400)
    os.utime(old, (ancient, ancient))

    assert store.prune() == 1
    remaining = {p.stem for p in (tmp_path / "jobs").glob("*.json")}
    assert "new" in remaining and "old" not in remaining


async def test_retention_zero_disables_pruning(tmp_path):
    import os
    import time

    store = _store(tmp_path, retention_days=0)
    store.save({"job_id": "old", "created_at": "1"})
    old = tmp_path / "jobs" / "old.json"
    ancient = time.time() - (999 * 86400)
    os.utime(old, (ancient, ancient))

    assert store.prune() == 0
    assert old.exists()


async def test_in_memory_pruning_keeps_the_persisted_file(tmp_path):
    """Evicting a finished job from memory must not destroy the audit record."""
    import asyncio

    store = _store(tmp_path)
    jm = _manager(
        [StageDescriptor("understanding", _noop, "understanding")],
        store=store,
        ttl_seconds=-1,  # everything terminal is immediately stale
    )
    job = jm.create_job(_incident())
    await asyncio.wait_for(jm.run_job(job), timeout=2)

    jm.create_job(_incident("INC-Q"))  # triggers _prune
    assert jm.get_job(job.job_id) is None, "dropped from memory"
    assert (tmp_path / "jobs" / f"{job.job_id}.json").exists(), "file survives"


# --- config ----------------------------------------------------------------


def test_build_job_store_reads_the_jobs_section():
    store = build_job_store(
        {"jobs": {"persist": True, "max_evidence_mb": 8, "retention_days": 3}}
    )
    assert store.enabled
    assert store.max_evidence_bytes == 8 * 1024 * 1024
    assert store.retention_days == 3


def test_build_job_store_defaults_when_the_section_is_absent():
    store = build_job_store({})
    assert store.enabled
    assert store.max_evidence_bytes == DEFAULT_MAX_EVIDENCE_BYTES


def test_build_job_store_honours_persist_false():
    assert build_job_store({"jobs": {"persist": False}}).enabled is False


# --- the storage seam ------------------------------------------------------
#
# The store's POLICY (evidence split, the cap, the .prev fallback, retention) is
# backend-independent on purpose: duplicating it per backend is how a remote path comes
# to disagree with the local one about what a job is, and the divergence would surface
# as a restored job that looks complete while missing its evidence. These prove the
# policy runs over an injected backend, not just over a directory.


def test_an_injected_backend_receives_the_job_documents(tmp_path):
    from src.storage import LocalStorage

    backend = LocalStorage(root=tmp_path / "remote")
    store = build_job_store({}, storage=backend)
    assert store.save({"job_id": "s1", "created_at": "1", "status": "running"})

    # Under a jobs/ prefix, which is where an upgraded VM's files already are.
    assert (tmp_path / "remote" / "jobs" / "s1.json").is_file()
    assert [d["job_id"] for d in store.load_all()] == ["s1"]


def test_the_evidence_split_holds_over_an_injected_backend(tmp_path):
    from src.storage import LocalStorage

    store = build_job_store({}, storage=LocalStorage(root=tmp_path / "remote"))
    store.save(
        {"job_id": "s2", "created_at": "1", "outputs": {"logs": {"a": [1]}, "x": 2}},
        evidence_changed=True,
    )
    jobs = tmp_path / "remote" / "jobs"
    doc = json.loads((jobs / "s2.json").read_text())
    assert "logs" not in doc["outputs"] and doc["outputs"]["x"] == 2
    assert json.loads((jobs / "s2.evidence.json").read_text()) == {"logs": {"a": [1]}}

    # ...and is merged back on load, or the restored job silently has no evidence.
    assert store.load_all()[0]["outputs"]["logs"] == {"a": [1]}


def test_an_oversize_sidecar_is_recorded_as_omitted_over_a_backend(tmp_path):
    from src.storage import LocalStorage

    store = build_job_store(
        {"jobs": {"max_evidence_mb": 0.0001}},
        storage=LocalStorage(root=tmp_path / "remote"),
    )
    store.save(
        {
            "job_id": "s3",
            "created_at": "1",
            "outputs": {"logs": {"a": [{"pad": "y" * 4000}]}},
        },
        evidence_changed=True,
    )
    loaded = store.load_all()[0]
    assert loaded["evidence_omitted"], "a reloaded job must be able to say so"
    assert not (tmp_path / "remote" / "jobs" / "s3.evidence.json").exists()


def test_a_backend_view_does_not_list_another_consumers_keys(tmp_path):
    """Several consumers share one backend; a job listing must see only jobs."""
    from src.storage import LocalStorage

    backend = LocalStorage(root=tmp_path / "remote")
    backend.put_text("exports/fraud_report_INC1.md", "# not a job")
    backend.append_text("feedback_log.jsonl", '{"n": 1}\n')

    store = build_job_store({}, storage=backend)
    store.save({"job_id": "s4", "created_at": "1"})
    assert [d["job_id"] for d in store.load_all()] == ["s4"]


def test_a_torn_document_falls_back_to_the_previous_version_over_a_backend(tmp_path):
    from src.storage import LocalStorage

    store = build_job_store({}, storage=LocalStorage(root=tmp_path / "remote"))
    store.save({"job_id": "s5", "created_at": "1", "status": "running"})
    store.save({"job_id": "s5", "created_at": "1", "status": "completed"})
    (tmp_path / "remote" / "jobs" / "s5.json").write_text("{ truncated")

    loaded = store.load_all()
    assert len(loaded) == 1 and loaded[0]["status"] == "running"


def test_prune_keeps_a_job_whose_age_the_backend_cannot_report(tmp_path):
    """Retention is housekeeping; a pending approval is not. Unknown age must keep."""
    from src.storage import LocalStorage, StoredObject

    class _NoMtime(LocalStorage):
        def list_keys(self, prefix):
            return [
                StoredObject(key=o.key, size=o.size, mtime=0.0)
                for o in super().list_keys(prefix)
            ]

    store = build_job_store(
        {"jobs": {"retention_days": 1}}, storage=_NoMtime(root=tmp_path / "remote")
    )
    store.save({"job_id": "s6", "created_at": "1"})
    assert store.prune() == 0
    assert store.load_all(), "a job of unknown age must survive the sweep"


def test_a_delete_takes_the_evidence_sidecar_with_it(tmp_path):
    from src.storage import LocalStorage

    store = build_job_store({}, storage=LocalStorage(root=tmp_path / "remote"))
    store.save(
        {"job_id": "s7", "created_at": "1", "outputs": {"logs": {"a": [1]}}},
        evidence_changed=True,
    )
    store.delete("s7")
    assert list((tmp_path / "remote" / "jobs").glob("s7*")) == []


def test_no_storage_argument_keeps_the_existing_directory_layout(tmp_path, monkeypatch):
    """The existing deployment. build_job_store({}) must write where it always did."""
    monkeypatch.setattr("src.job_store.data_dir", lambda: tmp_path)
    store = build_job_store({})
    store.save({"job_id": "s8", "created_at": "1"})
    assert (tmp_path / "jobs" / "s8.json").is_file()


async def _noop(ctx):
    return None


# Distinct name for stages used specifically to force a gate (health 0.0), so the
# intent is readable at the call site even though the body matches _noop.
async def _noop_none(ctx):
    return None
