"""
Tests for the human-in-the-loop surfaces.

Three things the pipeline could not do before and must not silently regress:

1. **Feedback closes the loop.** Distilled insights are rendered into the
   understanding + anomaly-detection prompts (they used to be written to a file
   nothing ever read), and the guidance is bounded + advisory.
2. **A partial batch survives a restart.** Reviews persist on arrival, so a new
   FeedbackLoop over the same data dir picks up an un-distilled batch.
3. **An analyst can override a stage's output mid-run** and continue the pipeline
   on the corrected data (`skip_stage`), with the intervention recorded.

Plus HTTP-level coverage of the feedback endpoints, which had none.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src import link_children, link_escalation
from src.anomaly_detection import AnomalyDetectionModule
from src.feedback_loop import FeedbackLoop
from src.incident_input import IncidentInputInterface
from src.incident_understanding import IncidentUnderstandingModule
from src.models.pydantic_models import (AnomalyItem, AnomalyList,
                                        CorrelationResult, ExtractedEntity,
                                        FeedbackInsights, IncidentAnalysis,
                                        LinkFinding, RetrievalQuery,
                                        UnderstandingResult)
from src.notifications import EventEmitter
from src.pipeline_runner import (JobManager, JobRunMode, JobStatus,
                                 StageDescriptor, StageStatus)
from src.utils.rate_limiter import AsyncRateLimiter

# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the feedback store at a tmp dir (it writes under data_dir())."""
    monkeypatch.setattr("src.feedback_loop.data_dir", lambda: tmp_path)
    return tmp_path


def _insights(**over):
    base = dict(
        common_success_patterns=["cross-check the org_unit against the issuance agent"],
        frequently_missed_anomalies=["cash payment on a same-day international document"],
        accuracy_improvements=["do not treat a automated sign as fraud"],
        confidence_threshold_recommendations=[
            "raise confidence when email is disposable"
        ],
        recommended_action_effectiveness=[],
        new_fraud_patterns=["voided-then-reissued document"],
        process_improvements=["state the scope boundary explicitly"],
    )
    base.update(over)
    return FeedbackInsights(**base)


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


# --- 1. durability: a partial batch survives a restart ----------------------


async def test_entries_persist_on_arrival_and_reload_after_restart(data_dir):
    loop = FeedbackLoop(MagicMock(), {"batch_size": 10})
    for i in range(3):
        await loop.collect_feedback(f"INC{i}", None, "looks right")

    # Both files exist immediately — not only once the batch fills.
    assert (data_dir / "feedback_log.jsonl").exists()
    assert (data_dir / "feedback_pending.jsonl").exists()

    # A fresh instance (process restart) recovers the un-distilled batch.
    revived = FeedbackLoop(MagicMock(), {"batch_size": 10})
    assert len(revived.feedback_data) == 3
    assert revived.feedback_data[0]["incident_id"] == "INC0"


async def test_distilling_clears_pending_but_keeps_the_audit_log(data_dir):
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=_insights())
    loop = FeedbackLoop(llm, {"batch_size": 2})

    await loop.collect_feedback("INC0", None, "a")
    await loop.collect_feedback("INC1", None, "b")  # triggers the distillation

    llm.structured_output.assert_awaited_once()
    assert loop.feedback_data == []
    assert not (data_dir / "feedback_pending.jsonl").exists()
    # History is permanent.
    assert len((data_dir / "feedback_log.jsonl").read_text().strip().splitlines()) == 2
    # A restart now starts with an empty batch, not a re-distillation of the same two.
    assert FeedbackLoop(MagicMock()).feedback_data == []


async def test_structured_review_fields_are_recorded(data_dir):
    loop = FeedbackLoop(MagicMock(), {"batch_size": 10})
    record = await loop.collect_feedback(
        "INC-9",
        None,
        None,
        agrees_with_verdict=False,
        analyst_verdict="VALID FRAUD",
        missed_anomalies=["cash + disposable email"],
        analyst="mehdi",
    )
    assert record["agrees_with_verdict"] is False
    assert record["analyst_verdict"] == "VALID FRAUD"
    assert record["missed_anomalies"] == ["cash + disposable email"]
    stats = loop.stats()
    assert stats["total_reviews"] == 1 and stats["disagreed_with_verdict"] == 1


# --- 2. the loop actually closes -------------------------------------------


async def test_guidance_is_per_stage_and_scoped(data_dir):
    loop = FeedbackLoop(MagicMock())
    await loop.apply_insights(_insights())

    understanding = loop.guidance_prompt("understanding")
    anomalies = loop.guidance_prompt("anomaly_detection")

    # Each stage sees only its own categories.
    assert "voided-then-reissued document" in understanding  # new_fraud_patterns
    assert "cash payment" not in understanding  # missed anomalies -> other stage
    assert "cash payment" in anomalies  # frequently_missed_anomalies
    assert "voided-then-reissued" not in anomalies
    # An unknown stage leaks nothing.
    assert loop.guidance_prompt("report_generation") == ""


async def test_guidance_is_marked_advisory_and_subordinate(data_dir):
    """The guidance must never read as authority over the pack/procedure/verdict."""
    loop = FeedbackLoop(MagicMock())
    await loop.apply_insights(_insights())
    text = loop.guidance_prompt("anomaly_detection")
    assert "ADVISORY" in text
    assert "NOT evidence" in text
    assert "deterministic verdict" in text


async def test_guidance_empty_when_nothing_distilled_or_disabled(data_dir):
    assert FeedbackLoop(MagicMock()).guidance_prompt("understanding") == ""

    loop = FeedbackLoop(MagicMock(), {"apply_to_prompts": False})
    await loop.apply_insights(_insights())
    assert loop.guidance_prompt("understanding") == ""


async def test_guidance_is_capped(data_dir):
    """A long feedback history cannot crowd the incident out of the prompt."""
    loop = FeedbackLoop(MagicMock())
    await loop.apply_insights(
        _insights(
            frequently_missed_anomalies=[f"pattern {i} " + "x" * 200 for i in range(40)]
        )
    )
    text = loop.guidance_prompt("anomaly_detection")
    assert len(text) < 2500
    assert text.count("pattern ") <= 5  # _MAX_ITEMS_PER_CATEGORY


async def test_merged_insights_dedupe_newest_first(data_dir):
    loop = FeedbackLoop(MagicMock())
    await loop.apply_insights(_insights(new_fraud_patterns=["old pattern", "shared"]))
    await loop.apply_insights(_insights(new_fraud_patterns=["new pattern", "shared"]))
    merged = loop.merged_insights(refresh=True)["new_fraud_patterns"]
    assert merged[0] == "new pattern"  # newest distillation leads
    assert merged.count("shared") == 1  # de-duplicated


async def test_understanding_stage_injects_guidance(data_dir):
    loop = FeedbackLoop(MagicMock())
    await loop.apply_insights(_insights())

    captured = {}

    async def _capture(messages, **kwargs):
        captured["messages"] = messages
        # A dict, not the `src.`-imported model: the module builds UnderstandingResult
        # from its FLAT model identity, which rejects the other one (dual-import trap).
        return _understanding().analysis.model_dump()

    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=_capture)
    module = IncidentUnderstandingModule(llm, feedback=loop)
    await module.process({"id": "I", "timestamp": "t", "description": "d"})

    systems = "\n".join(
        m["content"] for m in captured["messages"] if m["role"] == "system"
    )
    assert "ANALYST FEEDBACK GUIDANCE" in systems
    assert "voided-then-reissued document" in systems
    # The guidance sits among the system messages, before the user content.
    assert captured["messages"][-1]["role"] == "user"


async def test_anomaly_stage_injects_guidance(data_dir):
    loop = FeedbackLoop(MagicMock())
    await loop.apply_insights(_insights())

    captured = {}

    async def _capture(messages, **kwargs):
        captured["messages"] = messages
        return AnomalyList(anomalies=[])

    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=_capture)
    module = AnomalyDetectionModule({"threshold": 0.5}, llm, feedback=loop)
    await module.detect({"src": [{"a": 1}]}, _understanding(), correlation=None)

    systems = "\n".join(
        m["content"] for m in captured["messages"] if m["role"] == "system"
    )
    assert "cash payment on a same-day international document" in systems


async def test_stages_work_without_feedback_and_survive_broken_guidance(data_dir):
    """Feedback is optional, and a failure rendering it must not fail the stage."""
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=AnomalyList(anomalies=[]))
    assert (
        await AnomalyDetectionModule({"threshold": 0.5}, llm).detect(
            {"s": [{"a": 1}]}, _understanding(), None
        )
        == []
    )

    broken = MagicMock()
    broken.guidance_prompt.side_effect = RuntimeError("disk gone")
    module = AnomalyDetectionModule({"threshold": 0.5}, llm, feedback=broken)
    assert await module.detect({"s": [{"a": 1}]}, _understanding(), None) == []


# --- 3. mid-run stage-output override --------------------------------------


def _jm(stages, **kw):
    emitter = EventEmitter()
    jm = JobManager(stages, emitter, **kw)
    emitter.set_job_manager(jm)
    return jm


def _incident():
    return {"id": "INC-1", "description": "test", "timestamp": "2024-01-01T00:00"}


async def test_override_then_skip_runs_downstream_on_the_analyst_value():
    """The point of the feature: the rest of the pipeline consumes the human's data."""
    seen = {}

    async def failing(ctx):
        raise RuntimeError("LLM misread the incident")

    async def downstream(ctx):
        seen["input"] = ctx.outputs["understanding"]
        return "report"

    stages = [
        StageDescriptor("understanding", failing, "understanding"),
        StageDescriptor("report_generation", downstream, "report"),
    ]
    jm = _jm(stages)
    job = jm.create_job(_incident())
    await jm.run_job(job)
    assert job.status == JobStatus.STAGE_FAILED

    jm.set_stage_output(
        job.job_id,
        "understanding",
        _understanding().model_dump(mode="json"),
        actor="mehdi",
    )
    # The failed marker is cleared and the value is decoded to the TYPED model.
    assert job.stage_statuses["understanding"] == StageStatus.COMPLETED
    assert job.error is None
    assert job.context.outputs["understanding"].analysis.severity == "5"

    jm.control(job.job_id, "skip_stage", stage="understanding")

    await asyncio.sleep(0.05)

    assert job.status == JobStatus.COMPLETED
    assert job.context.outputs["report"] == "report"
    # Downstream really read the analyst's object, not a dict.
    assert seen["input"].analysis.incident_summary == "s"


async def test_override_records_an_intervention_and_rides_the_export():
    async def ok(ctx):
        return _understanding()

    stages = [StageDescriptor("understanding", ok, "understanding")]
    jm = _jm(stages)
    job = jm.create_job(_incident())
    await jm.run_job(job)

    jm.set_stage_output(
        job.job_id,
        "understanding",
        _understanding().model_dump(mode="json"),
        actor="mehdi",
    )
    iv = job.interventions
    assert len(iv) == 1
    assert iv[0]["action"] == "override_stage_output" and iv[0]["actor"] == "mehdi"
    # Audit trail is visible on the snapshot and survives export/import — a report
    # built on hand-edited data must be traceable as such.
    assert job.snapshot()["interventions"][0]["stage"] == "understanding"
    doc = jm.export_job(job.job_id)
    assert doc["interventions"] == iv
    reimported = _jm(stages).import_job(doc)
    assert reimported.interventions[0]["action"] == "override_stage_output"


async def test_override_rejects_a_value_that_breaks_the_contract():
    async def ok(ctx):
        return _understanding()

    jm = _jm([StageDescriptor("understanding", ok, "understanding")])
    job = jm.create_job(_incident())
    await jm.run_job(job)

    with pytest.raises(ValueError):
        jm.set_stage_output(job.job_id, "understanding", {"not": "an understanding"})
    # The good value is untouched by the rejected write.
    assert job.context.outputs["understanding"].incident_id == "INC-1"


async def test_override_rejected_while_the_stage_is_running():
    """A running stage would overwrite the override the moment it returned."""

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(ctx):
        started.set()
        await release.wait()
        return _understanding()

    jm = _jm([StageDescriptor("understanding", slow, "understanding")])
    job = jm.create_job(_incident())
    task = asyncio.ensure_future(jm.run_job(job))
    await started.wait()

    with pytest.raises(ValueError, match="running"):
        jm.set_stage_output(
            job.job_id, "understanding", _understanding().model_dump(mode="json")
        )
    release.set()
    await task


async def test_override_unknown_job_or_stage_raises_keyerror():
    jm = _jm([StageDescriptor("a", AsyncMock(), "oa")])
    job = jm.create_job(_incident())
    with pytest.raises(KeyError):
        jm.set_stage_output("no-such-job", "a", 1)
    with pytest.raises(KeyError):
        jm.set_stage_output(job.job_id, "no-such-stage", 1)


async def test_skip_stage_marks_a_non_overridden_stage_skipped():
    """Skipping without an override is honest about it: status is SKIPPED."""

    async def boom(ctx):
        raise RuntimeError("source down")

    async def after(ctx):
        return "done"

    jm = _jm([StageDescriptor("s1", boom, "o1"), StageDescriptor("s2", after, "o2")])
    job = jm.create_job(_incident())
    await jm.run_job(job)

    jm.control(job.job_id, "skip_stage")  # defaults to the failed stage
    await asyncio.sleep(0.05)

    assert job.stage_statuses["s1"] == StageStatus.SKIPPED
    assert job.status == JobStatus.COMPLETED
    assert job.context.outputs["o2"] == "done"


async def test_skip_last_stage_completes_the_job():
    async def boom(ctx):
        raise RuntimeError("delivery down")

    jm = _jm([StageDescriptor("only", boom, "o")])
    job = jm.create_job(_incident())
    await jm.run_job(job)

    jm.control(job.job_id, "skip_stage", stage="only")
    assert job.status == JobStatus.COMPLETED


async def test_skip_stage_needs_a_target():
    async def ok(ctx):
        return 1

    jm = _jm([StageDescriptor("a", ok, "oa")])
    job = jm.create_job(_incident())
    await jm.run_job(job)  # completes, so there is no failed stage to infer
    with pytest.raises(ValueError):
        jm.control(job.job_id, "skip_stage")
    with pytest.raises(ValueError):
        jm.control(job.job_id, "skip_stage", stage="nope")


# --- 4. HTTP surface -------------------------------------------------------


def _config():
    return {
        "post_incident_endpoint": "/api/v1/incidents",
        "post_ir_endpoint": "/api/v1/ir",
        "post_freetext_endpoint": "/api/v1/incidents/freetext",
        "port": 5000,
        "rate_limit": {"requests": 1000, "per_seconds": 60},
    }


async def _client(iface):
    iface.rate_limiter = AsyncRateLimiter(1000, 60)
    client = TestClient(TestServer(iface.app))
    await client.start_server()
    return client


async def test_post_feedback_accepts_structured_review(data_dir):
    loop = FeedbackLoop(MagicMock(), {"batch_size": 10})

    async def feedback_fn(incident_id, result, human_feedback, **structured):
        return await loop.collect_feedback(
            incident_id, result, human_feedback, **structured
        )

    iface = IncidentInputInterface(
        _config(), feedback_fn=feedback_fn, feedback_loop=loop
    )
    client = await _client(iface)
    try:
        resp = await client.post(
            "/api/v1/feedback",
            json={
                "incident_id": "INC-7",
                "agrees_with_verdict": False,
                "analyst_verdict": "VALID FRAUD",
                "missed_anomalies": "cash payment",  # a bare string is accepted
                "analyst": "mehdi",
            },
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["status"] == "received"
        assert body["stats"]["total_reviews"] == 1
        # The string was normalized to a list.
        assert loop.feedback_data[0]["missed_anomalies"] == ["cash payment"]
    finally:
        await client.close()


async def test_post_feedback_requires_something_substantive(data_dir):
    iface = IncidentInputInterface(_config(), feedback_fn=AsyncMock())
    client = await _client(iface)
    try:
        assert (await client.post("/api/v1/feedback", json={})).status == 400
        # incident_id alone says nothing about the investigation.
        r = await client.post("/api/v1/feedback", json={"incident_id": "X"})
        assert r.status == 400
        # Back-compat: the original free-form shape still works.
        r = await client.post(
            "/api/v1/feedback",
            json={
                "incident_id": "X",
                "human_feedback": "false positive — known org_unit",
            },
        )
        assert r.status == 200
    finally:
        await client.close()


async def test_get_feedback_returns_history_stats_and_insights(data_dir):
    loop = FeedbackLoop(MagicMock(), {"batch_size": 10})
    await loop.collect_feedback("INC-1", None, "good catch")
    await loop.apply_insights(_insights())

    iface = IncidentInputInterface(_config(), feedback_loop=loop)
    client = await _client(iface)
    try:
        resp = await client.get("/api/v1/feedback")
        assert resp.status == 200
        body = await resp.json()
        assert body["stats"]["total_reviews"] == 1
        assert body["history"][0]["incident_id"] == "INC-1"
        assert body["pending"][0]["incident_id"] == "INC-1"
        assert "voided-then-reissued document" in body["insights"]["new_fraud_patterns"]
    finally:
        await client.close()


async def test_get_feedback_guidance_shows_what_is_injected(data_dir):
    loop = FeedbackLoop(MagicMock())
    await loop.apply_insights(_insights())
    iface = IncidentInputInterface(_config(), feedback_loop=loop)
    client = await _client(iface)
    try:
        body = await (await client.get("/api/v1/feedback/guidance")).json()
        assert body["apply_to_prompts"] is True
        assert "voided-then-reissued" in body["guidance"]["understanding"]
        assert "cash payment" in body["guidance"]["anomaly_detection"]
    finally:
        await client.close()


async def test_force_distill_endpoint(data_dir):
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=_insights())
    loop = FeedbackLoop(llm, {"batch_size": 10})
    iface = IncidentInputInterface(_config(), feedback_loop=loop)
    client = await _client(iface)
    try:
        # Nothing pending -> 409 rather than a pointless LLM call.
        assert (await client.post("/api/v1/feedback/process")).status == 409

        await loop.collect_feedback("INC-1", None, "note")
        resp = await client.post("/api/v1/feedback/process")
        assert resp.status == 200
        body = await resp.json()
        assert body["processed"] == 1
        assert body["stats"]["distillations"] == 1
        assert loop.feedback_data == []
    finally:
        await client.close()


async def test_feedback_read_endpoints_503_without_a_loop():
    iface = IncidentInputInterface(_config(), feedback_fn=AsyncMock())
    client = await _client(iface)
    try:
        assert (await client.get("/api/v1/feedback")).status == 503
        assert (await client.get("/api/v1/feedback/guidance")).status == 503
        assert (await client.post("/api/v1/feedback/process")).status == 503
    finally:
        await client.close()


async def test_override_endpoint_applies_and_validates():
    async def ok(ctx):
        return _understanding()

    jm = _jm([StageDescriptor("understanding", ok, "understanding")])
    job = jm.create_job(_incident())
    await jm.run_job(job)

    iface = IncidentInputInterface(_config(), job_manager=jm)
    client = await _client(iface)
    try:
        # Missing 'value'.
        r = await client.post(
            f"/api/v1/jobs/{job.job_id}/outputs/understanding", json={"actor": "m"}
        )
        assert r.status == 400

        # A value that doesn't fit the contract -> 400, not a poisoned job.
        r = await client.post(
            f"/api/v1/jobs/{job.job_id}/outputs/understanding",
            json={"value": {"bogus": True}},
        )
        assert r.status == 400

        # Unknown stage -> 404.
        r = await client.post(
            f"/api/v1/jobs/{job.job_id}/outputs/nope", json={"value": {}}
        )
        assert r.status == 404

        # The real thing.
        good = _understanding().model_dump(mode="json")
        good["analysis"]["severity"] = "9"
        r = await client.post(
            f"/api/v1/jobs/{job.job_id}/outputs/understanding",
            json={"value": good, "actor": "mehdi"},
        )
        assert r.status == 200
        snap = await r.json()
        assert snap["interventions"][0]["action"] == "override_stage_output"
        assert job.context.outputs["understanding"].analysis.severity == "9"
    finally:
        await client.close()


async def test_skip_stage_is_an_accepted_control_action():
    async def boom(ctx):
        raise RuntimeError("down")

    async def after(ctx):
        return "done"

    jm = _jm([StageDescriptor("s1", boom, "o1"), StageDescriptor("s2", after, "o2")])
    job = jm.create_job(_incident())
    await jm.run_job(job)

    iface = IncidentInputInterface(_config(), job_manager=jm)
    client = await _client(iface)
    try:
        r = await client.post(
            f"/api/v1/jobs/{job.job_id}/control",
            json={"action": "skip_stage", "stage": "s1"},
        )
        assert r.status == 200
        # An unknown stage is a 409 (bad state), not a 500.
        r = await client.post(
            f"/api/v1/jobs/{job.job_id}/control",
            json={"action": "skip_stage", "stage": "ghost"},
        )
        assert r.status == 409
    finally:
        await client.close()


async def test_console_ui_exposes_the_hitl_controls():
    iface = IncidentInputInterface(_config())
    client = await _client(iface)
    try:
        body = await (await client.get("/")).text()
        assert "Analyst review" in body
        assert "/api/v1/feedback" in body
        assert "skip_stage" in body
        assert "/outputs/" in body
        # The numeric half is visible too, including the revert path.
        assert "/api/v1/feedback/threshold" in body
        assert "Anomaly confidence threshold" in body
        assert "/api/v1/feedback/threshold/reset" in body
    finally:
        await client.close()


# --- 4b. the retrieval plan, edited by an operator --------------------------
#
# The operator names a source; the server builds the query, because which entities a source
# can filter by is pack knowledge.


def _plan_jm(**kw):
    """A job manager whose `api_call` module is a real-enough generator for the endpoints.

    Deliberately NOT a MagicMock: `_plan_context` refuses a module with no
    `dependency_report`, and a MagicMock satisfies every `hasattr` — which would make the
    503-refusal test vacuous and let the handlers "work" against a mock that answers anything.
    """

    class _Gen:
        def __init__(self):
            self.built = []
            self.window_hints = []

        def dependency_report(self, queries, source_names=None, analysis=None):
            chosen = {q.target_log_source for q in queries or []}
            return {
                "undeliverable": [],
                "not_queried": [n for n in ("declared_src",) if n not in chosen],
                "unscopable": [],
            }

        def unselected_sources(self, queries, analysis=None):
            chosen = {q.target_log_source for q in queries or []}
            return [
                {
                    "source": n,
                    "purpose": "p",
                    "declared": n == "declared_src",
                    "deferred": False,
                    "scopable": True,
                }
                for n in ("declared_src", "other_src")
                if n not in chosen
            ]

        def referral_window(self, analysis, prior_queries, window_hint=""):
            # Records the hint and returns a fixed window whose mode differs from the hint.
            # A depthless `lookback` degrades to `inherit`, so the double must not echo the
            # hint back — that would make the assertion pass whichever mode was reported.
            self.window_hints.append(window_hint)
            return "2026-07-02", "2026-08-08", "inherit"

        def build_manual_query(self, analysis, source, question="", window=None):
            if source not in ("declared_src", "other_src"):
                raise ValueError(
                    f"No source named '{source}' — it is not in the catalog"
                )
            self.built.append((source, question, window))
            return RetrievalQuery(
                target_log_source=source,
                natural_language_query=question or "the source's own purpose",
                date_from=(window or ("", ""))[0],
                date_to=(window or ("", ""))[1],
            )

    async def plan(ctx):
        return [
            RetrievalQuery(
                target_log_source="src_a",
                natural_language_query="q a",
                date_from="2026-08-01",
                date_to="2026-08-08",
                entities=[
                    ExtractedEntity(type="record", value="ABC123"),
                    ExtractedEntity(type="time_window", value="2026-08-01"),
                ],
            )
        ]

    gen = _Gen()
    jm = _jm(
        [
            StageDescriptor("understanding", _ok_understanding, "understanding"),
            StageDescriptor("query_generation", plan, "queries"),
        ],
        modules={"api_call": gen},
        **kw,
    )
    return jm, gen


async def _ok_understanding(ctx):
    return _understanding()


async def test_the_plan_endpoint_names_what_each_query_is_scoped_by():
    """The reason this is an endpoint and not a JSON textarea.

    A query's `entities` are filtered to the ones the source can bind, so "which of the
    incident's identifiers does this narrow on" is not something an operator can read off the
    incident text — and an unscoped query is a bare date-window scan, the exact defect the
    removed force-add produced. `row_counts` is beside it because a source that already
    answered and a source about to be removed are the two facts the decision turns on.
    """
    jm, _gen = _plan_jm()
    job = jm.create_job(_incident())
    await jm.run_job(job)
    job.context.outputs["logs"] = {"src_a": [{"r": 1}, {"r": 2}]}

    iface = IncidentInputInterface(_config(), job_manager=jm)
    client = await _client(iface)
    try:
        r = await client.get(f"/api/v1/jobs/{job.job_id}/queries")
        assert r.status == 200
        d = await r.json()
        assert d["pass"] == 1
        assert d["queries"] == [
            {
                "index": 0,
                "source": "src_a",
                "question": "q a",
                "date_from": "2026-08-01",
                "date_to": "2026-08-08",
                # time_window excluded: every query carries one, so listing it would say
                # nothing about scope.
                "scoped_by": ["record"],
            }
        ]
        assert d["row_counts"] == {"src_a": 2}
        assert [s["source"] for s in d["unselected"]] == ["declared_src", "other_src"]
        assert d["dependencies"]["not_queried"] == ["declared_src"]
    finally:
        await client.close()


async def test_editing_the_plan_is_all_or_nothing_and_indexes_before_it_appends():
    """Both rules, and each is a way the operator's own handles stop meaning what they meant.

    A partial apply leaves a plan neither the operator nor the run intended; and an addition
    appended before removals resolve shifts every index the client was given. The edit lands
    through `set_stage_output`, so a report built on a hand-edited plan is traceable as one.
    """
    jm, gen = _plan_jm()
    job = jm.create_job(_incident())
    await jm.run_job(job)

    iface = IncidentInputInterface(_config(), job_manager=jm)
    client = await _client(iface)
    url = f"/api/v1/jobs/{job.job_id}/queries"
    try:
        # Nothing staged is a refusal, not a no-op that reports success.
        assert (await client.post(url, json={})).status == 400
        # An out-of-range index is a 409 naming the size, because the plan may have moved.
        r = await client.post(url, json={"remove": [7]})
        assert r.status == 409
        assert "the plan holds 1" in (await r.json())["error"]
        # A non-integer index, and a malformed addition.
        assert (await client.post(url, json={"remove": ["a"]})).status == 400
        assert (await client.post(url, json={"add": ["src"]})).status == 400
        # An unretrievable source refuses the WHOLE request — the removal beside it must not
        # have been applied.
        r = await client.post(
            url, json={"remove": [0], "add": [{"source": "no_such_source"}]}
        )
        assert r.status == 400
        assert [q.target_log_source for q in job.context.outputs["queries"]] == ["src_a"]
        assert job.interventions == []

        # The real thing: one request, one intervention, removals resolved against the
        # indices the GET returned.
        r = await client.post(
            url,
            json={
                "remove": [0],
                "add": [{"source": "declared_src", "question": "ask it anyway"}],
                "actor": "mehdi",
            },
        )
        assert r.status == 200
        d = await r.json()
        assert d["added"] == ["declared_src"] and d["removed"] == ["src_a"]
        assert d["queries"] == 1
        assert d["dependencies"]["not_queried"] == []
        plan = job.context.outputs["queries"]
        assert [q.target_log_source for q in plan] == ["declared_src"]
        assert plan[0].natural_language_query == "ask it anyway"
        # The window was borrowed from the plan the operator was looking at, so the addition
        # is bounded the way its neighbours were rather than unbounded.
        assert gen.built == [("declared_src", "ask it anyway", ("2026-08-01", "2026-08-08"))]

        # Audited as a hand-supplied output, and it says WHAT changed: "the plan was edited"
        # is not a trail anyone can act on.
        iv = job.interventions[-1]
        assert iv["action"] == "override_stage_output"
        assert iv["stage"] == "query_generation"
        assert iv["actor"] == "mehdi"
        assert "added declared_src" in iv["detail"]
        assert "removed src_a" in iv["detail"]
    finally:
        await client.close()


async def test_the_plan_endpoints_refuse_for_the_reason_that_applies():
    """Four refusals, and a caller can act on only some of them.

    Chiefly: no understanding yet is a 409 and not an empty list, because a plan that cannot
    exist yet and a planner that chose nothing are different answers — and the second is one
    an operator would try to fix by adding queries to a run that has not read the incident.
    """
    jm, _gen = _plan_jm()
    job = jm.create_job(_incident())

    iface = IncidentInputInterface(_config(), job_manager=jm)
    client = await _client(iface)
    try:
        # Created but not run: no understanding.
        for call in (client.get, client.post):
            r = await call(f"/api/v1/jobs/{job.job_id}/queries", json={"remove": [0]})
            assert r.status == 409
            assert "no understanding yet" in (await r.json())["error"]
        assert (await client.get("/api/v1/jobs/nope/queries")).status == 404
    finally:
        await client.close()

    # No generator on this server instance is a 503, not a 500 from the first attribute read.
    bare = _jm([StageDescriptor("understanding", _ok_understanding, "understanding")])
    job2 = bare.create_job(_incident())
    await bare.run_job(job2)
    client = await _client(IncidentInputInterface(_config(), job_manager=bare))
    try:
        r = await client.get(f"/api/v1/jobs/{job2.job_id}/queries")
        assert r.status == 503
    finally:
        await client.close()

    # And no job manager at all — the 404 a `_lookup_job`-first order would have given is
    # worse than useless: it says the job is gone rather than that jobs are not served.
    client = await _client(IncidentInputInterface(_config()))
    try:
        assert (await client.get("/api/v1/jobs/x/queries")).status == 503
        assert (await client.post("/api/v1/jobs/x/queries", json={})).status == 503
    finally:
        await client.close()


# --- 4c. the advisory lane, acted on by naming an index ---------------------
#
# Same rule as the plan editor: the server composes because construction is pack knowledge.
# The endpoint composes a referral run and does not launch it.


#: A distinctive sentence, long enough that absence assertions don't match by accident.
#: `_incident()` uses `"test"`, which is too short.
_PARENT_PROSE = (
    "an unexplained refund reversal on booking QQ7L4P, flagged by the night desk"
)


def _linked_job(jm, links, description=_PARENT_PROSE):
    """A run whose correlation output carries `links`, ready to refer."""

    async def go():
        job = jm.create_job({**_incident(), "description": description})
        await jm.run_job(job)
        job.context.outputs["correlation"] = CorrelationResult(links=links)
        return job

    return go


def _link(**kw):
    base = {
        "target_use_case": "sibling_proc",
        "target_playbook_id": "PB-SIB",
        "direction": "antecedent",
        "state": "probed_positive",
        "rung": 2,
        "pivot_entity": "actor",
        "pivot_values": ["A-1"],
        "evidence_note": "3 rejected attempts by this actor",
        "window_hint": "lookback:30d",
        "advisory_severity": "HIGH",
    }
    base.update(kw)
    return LinkFinding(**base)


async def test_a_link_referral_is_COMPOSED_and_never_launched():
    """The whole endpoint in one assertion: a body to POST, and nothing started.

    A `refer` that spawned would be the most expensive silent side effect in the API — a
    38-minute median run against a primary source estate, from one click, with no depth cap and
    no cycle guard. So `launched` is a FIELD and not an omission: an operator reading a 200 with
    a composed request in it must not have to infer whether a run is already going.

    And the composed description must NOT carry the parent's own prose, however much of the
    incident it holds. Which procedure adjudicates the child is decided by scoring its
    description, and the parent's description is precisely the text that made the PARENT's
    procedure win — a referral composed from it re-adjudicates what was already adjudicated,
    under the same headings, and reads as a confirmation.
    """
    jm, gen = _plan_jm()
    job = await _linked_job(jm, [_link()])()

    iface = IncidentInputInterface(_config(), job_manager=jm)
    client = await _client(iface)
    try:
        r = await client.post(
            f"/api/v1/jobs/{job.job_id}/links/0/refer", json={"actor": "mehdi"}
        )
        assert r.status == 200
        d = await r.json()
        assert d["launched"] is False
        assert d["post_to"] == "/api/v1/jobs"

        # Scoped to the pivot the sibling's leg opens on, over its implied window.
        assert gen.window_hints == ["lookback:30d"]
        assert d["scope"]["pivot_values"] == ["A-1"]
        assert (d["scope"]["date_from"], d["scope"]["date_to"]) == (
            "2026-07-02",
            "2026-08-08",
        )
        # Declaration and applied mode are two fields; the fixture rigs them to differ
        # (see `_Gen.referral_window`). Reporting the declaration in the applied slot would
        # misrepresent the actual window.
        assert d["scope"]["window_hint"] == "lookback:30d"
        assert d["scope"]["window_applied"] == "inherit"

        # The description names the TARGET and the pivot, and the parent only by id.
        desc = d["request"]["description"]
        assert "sibling_proc" in desc and "A-1" in desc
        assert job.job_id in desc and "INC-1" in desc
        assert _PARENT_PROSE not in desc
        assert "refund reversal" not in desc and "QQ7L4P" not in desc
        # The mode defaults to the parent's, because a referral out of a supervised run is
        # being watched by somebody.
        assert d["request"]["mode"] == job.run_mode.value

        # The pin rides in the request, not only the sibling block: a pin an operator has
        # to transcribe out of a neighbouring field is one a client may silently drop.
        assert d["pin"] == {
            "use_case": "sibling_proc",
            "playbook_id": "PB-SIB",
            "enforced": True,
            "note": d["pin"]["note"],
        }
        assert "Applied" in d["pin"]["note"]
        assert d["request"]["link_pin"] == "sibling_proc"

        # Audited like every other analyst action, in the word that describes what happened.
        iv = job.interventions[-1]
        assert iv["action"] == "link_referral_composed"
        assert iv["stage"] == "correlation"
        assert iv["actor"] == "mehdi"
        assert "sibling_proc" in iv["detail"] and "not launched" in iv["detail"]
    finally:
        await client.close()


async def test_referring_an_UNREACHABLE_link_is_refused_with_its_reason():
    """The state that has nothing to refer, and why refusing beats composing.

    `unreachable` means no value of the sibling's subject entity is in hand. A child run
    composed anyway would be scoped by nothing at all — a window-wide scan of a primary source
    estate, which is the defect `selected_source_unparseable` exists to stop one stage earlier,
    arriving here dressed as the operator's own click. 409 and not 400: the request is well
    formed, the STATE is what cannot be acted on, and the reason names the binding that is
    missing so the fix lands in the pack.
    """
    jm, _gen = _plan_jm()
    job = await _linked_job(
        jm,
        [_link(state="unreachable", pivot_values=[], pivot_entity="loyalty_id")],
    )()

    client = await _client(IncidentInputInterface(_config(), job_manager=jm))
    try:
        r = await client.post(f"/api/v1/jobs/{job.job_id}/links/0/refer")
        assert r.status == 409
        err = (await r.json())["error"]
        assert "loyalty_id" in err and "unreachable" in err
        # Nothing was recorded: an action that was refused is not an intervention.
        assert [
            i for i in job.interventions if i["action"] == "link_referral_composed"
        ] == []
    finally:
        await client.close()


async def test_the_referral_endpoint_refuses_for_the_reason_that_applies():
    """Five refusals, and each one a different next action.

    The specific reason comes FIRST: an index that does not resolve is almost always a run
    that has not correlated yet, and answering that with "no such link" gives it the word for
    a typo. A stale index is a 409 for the plan editor's reason — a rejected gate re-runs
    correlation, so the handle the operator read a minute ago can be right-then and wrong-now.
    """
    jm, _gen = _plan_jm()
    job = jm.create_job(_incident())
    await jm.run_job(job)  # ran, but nothing set a correlation output

    client = await _client(IncidentInputInterface(_config(), job_manager=jm))
    try:
        r = await client.post(f"/api/v1/jobs/{job.job_id}/links/0/refer")
        assert r.status == 409
        assert "has not correlated yet" in (await r.json())["error"]

        # Correlated, and holding fewer links than the index asks for.
        job.context.outputs["correlation"] = CorrelationResult(links=[_link()])
        r = await client.post(f"/api/v1/jobs/{job.job_id}/links/7/refer")
        assert r.status == 409
        assert "this run holds 1" in (await r.json())["error"]
        assert (await client.post("/api/v1/jobs/nope/links/0/refer")).status == 404
    finally:
        await client.close()

    # A generator that cannot derive a window refuses rather than composing an unscoped one:
    # the window IS the direction, and a referral over the parent's own window would silently
    # ask an antecedent question about the episode.
    class _NoWindow:
        def dependency_report(self, queries, source_names=None, analysis=None):
            return {}

    jm2, _g2 = _plan_jm()
    job2 = await _linked_job(jm2, [_link()])()
    jm2.modules["api_call"] = _NoWindow()
    client = await _client(IncidentInputInterface(_config(), job_manager=jm2))
    try:
        r = await client.post(f"/api/v1/jobs/{job2.job_id}/links/0/refer")
        assert r.status == 503
        assert "full-window scan" in (await r.json())["error"]
    finally:
        await client.close()

    # And no job manager at all — jobs are not served here, which is not the same as a job
    # that is gone.
    client = await _client(IncidentInputInterface(_config()))
    try:
        assert (await client.post("/api/v1/jobs/x/links/0/refer")).status == 503
    finally:
        await client.close()


class _ModePack:
    """A pack double for the mode endpoint: which procedures exist, and what each has declared.

    Two methods, and the split is the whole reason a double is needed here rather than a
    `MagicMock`. `ruleset_spec` decides whether a NAME is real — a typo must not be accepted into
    a setting nothing will ever read, and that is the only question this endpoint asks the pack.
    `entry_signals` carries the pair's declaration, whose measured `base_rate` is informational:
    it feeds the confidence score's discrimination term additively and licenses NOTHING, so the
    corpus a target is constructed with here changes no mode. What licenses an escalation rides on
    the FINDING — `gate_outcome`, rung 1 over this run's own rows — which is why the tests below
    set that and not a corpus.
    """

    def __init__(self, corpus_by_target):
        self._corpus = dict(corpus_by_target)

    def ruleset_spec(self, key=""):
        return {"subject_entity": "actor"} if key in self._corpus else {}

    def ruleset_keys(self):
        return sorted(self._corpus)

    def entry_signals(self, key=""):
        corpus = self._corpus.get(key, 0)
        if not corpus:
            return []
        return [
            {
                "id": "s1",
                "base_rate": {"fires_on": 3, "of": corpus, "measured": "2026-08-19"},
            }
        ]


def _mode_jm(corpus_by_target, **kw):
    """`_plan_jm` with a pack wired where the mode endpoint looks for it."""
    jm, gen = _plan_jm(**kw)
    jm.modules["correlation"] = MagicMock(knowledge_pack=_ModePack(corpus_by_target))
    return jm, gen


async def test_a_link_escalation_mode_is_SET_on_the_job_and_on_the_cards_at_once():
    """Mode is written on the job and re-stamped on the findings.

    On the job: a rejected gate re-runs correlation and re-resolves the mode; a setting only
    on the findings would silently revert. On the findings: the card is the only surface the
    operator reads. The response reports what took effect, not what was asked. Both findings
    carry `gate_outcome="pass"`; one also carries a corpus count, and the two resolve
    identically — a historical statistic is not a permission.
    """
    jm, _gen = _mode_jm({"sibling_proc": 27})
    job = await _linked_job(
        jm,
        [
            _link(gate_outcome="pass", mode_licensed=True, mode_corpus=27),
            _link(gate_outcome="pass", mode_licensed=True),
        ],
    )()

    client = await _client(IncidentInputInterface(_config(), job_manager=jm))
    try:
        r = await client.post(
            f"/api/v1/jobs/{job.job_id}/links/mode",
            json={"mode": "AUTO", "actor": "mehdi"},
        )
        assert r.status == 200
        d = await r.json()
        # Folded, so the vocabulary has one spelling wherever it is read back.
        assert d["mode"] == "auto"
        assert d["targets_checked_against_pack"] is True
        assert d["applied"] == [
            {
                "target_use_case": "sibling_proc",
                "asked": "auto",
                "mode": "auto",
                "mode_source": "job",
                "mode_note": d["applied"][0]["mode_note"],
                "proposed_action": link_escalation.proposed_action("auto"),
                "links": 2,
            }
        ]
        # On the job, which is what the next correlation reads.
        assert d["link_modes"] == {"sibling_proc": "auto"}
        assert job.context.link_modes == {"sibling_proc": "auto"}
        # And on every finding of that pair, not just the first.
        got = job.context.outputs["correlation"].links
        assert [f.mode for f in got] == ["auto", "auto"]
        assert [f.mode_source for f in got] == ["job", "job"]
        assert got[0].proposed_action == link_escalation.proposed_action("auto")

        iv = job.interventions[-1]
        assert iv["action"] == "link_mode_set"
        assert iv["stage"] == "correlation"
        assert iv["actor"] == "mehdi"
        assert "auto" in iv["detail"] and "sibling_proc" in iv["detail"]
        # Nothing was held, so the clause about the clamp is absent rather than negated.
        assert "held at" not in iv["detail"]
    finally:
        await client.close()


async def test_an_UNLICENSED_escalation_is_CLAMPED_and_the_request_still_succeeds():
    """An unlicensed `auto` comes back as `planned`, but this is a 200, not a 4xx.

    `mode_source` distinguishes a clamped link from one nobody set a mode for: both show
    `planned`, but the first needs the target's gate to resolve, the second needs a decision.
    The clamp note names which rung-1 outcome refused it; a corpus count makes no difference.
    """
    jm, _gen = _mode_jm({"sibling_proc": 3})
    job = await _linked_job(
        jm, [_link(gate_outcome="fail", mode_licensed=False, mode_corpus=3)]
    )()

    client = await _client(IncidentInputInterface(_config(), job_manager=jm))
    try:
        r = await client.post(
            f"/api/v1/jobs/{job.job_id}/links/mode", json={"mode": "auto"}
        )
        assert r.status == 200
        d = await r.json()
        entry = d["applied"][0]
        assert entry["asked"] == "auto"
        # `MANUAL_LINK_MODE` and not `DEFAULT_LINK_MODE`: the default is `semi_auto`, and a clamp
        # that landed on the default would be an escalating mode reported as a refusal.
        assert entry["mode"] == link_escalation.MANUAL_LINK_MODE == "planned"
        assert entry["mode_source"] == "clamp"
        assert "auto" in entry["mode_note"] and "job layer" in entry["mode_note"]
        assert link_escalation._GATE_REFUSALS["fail"] in entry["mode_note"]
        # And the outcome that was NOT this run's is absent, or the note names two answers.
        assert link_escalation._GATE_REFUSALS["unknown"] not in entry["mode_note"]
        assert entry["proposed_action"] == link_escalation.proposed_action("planned")

        # The ask is stored, not the clamped result: the bar re-applies on the next pass.
        assert job.context.link_modes == {"sibling_proc": "auto"}
        assert job.context.outputs["correlation"].links[0].mode == "planned"
        assert job.context.outputs["correlation"].links[0].mode_source == "clamp"

        # The audit trail records that the mode was held, not what was asked.
        iv = job.interventions[-1]
        assert iv["action"] == "link_mode_set"
        assert f"held at '{link_escalation.MANUAL_LINK_MODE}'" in iv["detail"]
        assert "sibling_proc" in iv["detail"]
        # Assert `MANUAL_LINK_MODE`, not `DEFAULT_LINK_MODE`: a clamp at the default would
        # be recorded as a refusal in a case where none was intended.
        assert f"held at '{link_escalation.DEFAULT_LINK_MODE}'" not in iv["detail"]
    finally:
        await client.close()


async def test_a_mode_can_be_set_for_a_pair_this_run_has_no_link_for_YET():
    """Named targets, resolved against the pack, before correlation has produced any links.

    Names are checked against the pack, not the link list: a real procedure with no link yet
    is legitimate; a typo is a setting nothing reads. Both come back `planned` via
    `NO_CANDIDATE` (not the generic gate refusal). The ask is stored and re-resolved when a
    candidate appears.
    """
    jm, _gen = _mode_jm({"sibling_proc": 27, "quiet_proc": 3})
    job = jm.create_job(_incident())
    await jm.run_job(job)  # correlated nothing

    client = await _client(IncidentInputInterface(_config(), job_manager=jm))
    try:
        r = await client.post(
            f"/api/v1/jobs/{job.job_id}/links/mode",
            json={"mode": "semi_auto", "targets": ["sibling_proc", " quiet_proc "]},
        )
        assert r.status == 200
        d = await r.json()
        assert [a["target_use_case"] for a in d["applied"]] == [
            "sibling_proc",
            "quiet_proc",
        ]
        assert [a["links"] for a in d["applied"]] == [0, 0]
        # Both clamp, and the well-counted pair clamps exactly like the thin one: 27 incidents of
        # history is not evidence about THIS incident, which is the only thing rung 1 reads.
        assert [a["mode"] for a in d["applied"]] == ["planned", "planned"]
        assert [a["mode_source"] for a in d["applied"]] == ["clamp", "clamp"]
        for entry in d["applied"]:
            assert link_escalation._GATE_REFUSALS[link_escalation.NO_CANDIDATE] in (
                entry["mode_note"]
            )
            # NOT the sentence a real non-passing outcome gets: no gate was evaluated here.
            assert link_escalation._GATE_REFUSALS["no_gate"] not in entry["mode_note"]
        # The ask, not the clamp, is what rides on the job for both.
        assert job.context.link_modes == {
            "sibling_proc": "semi_auto",
            "quiet_proc": "semi_auto",
        }
    finally:
        await client.close()


async def test_the_link_mode_endpoint_refuses_for_the_reason_that_applies():
    """Seven refusals, and each one a different next action.

    The order matters as much as the codes. A body that names no mode is a 400 listing the three
    words, because the vocabulary is closed and a typo is the likeliest reason to be here. A
    target the pack does not name is a 400 too — the request is malformed, in the one way no
    schema can catch. But an omitted `targets` on a run with no links is a **409**: the request
    is fine and the STATE cannot answer it, so the refusal names the request that would work
    instead of teaching the operator that their body was wrong.
    """
    jm, _gen = _mode_jm({"sibling_proc": 27})
    job = await _linked_job(jm, [_link(mode_licensed=True)])()
    url = f"/api/v1/jobs/{job.job_id}/links/mode"

    client = await _client(IncidentInputInterface(_config(), job_manager=jm))
    try:
        assert (await client.post(url, data="")).status == 400
        assert (await client.post(url, data="{oops")).status == 400
        assert (await client.post(url, json=["auto"])).status == 400

        r = await client.post(url, json={"mode": "semi-auto"})
        assert r.status == 400
        err = (await r.json())["error"]
        for word in link_escalation.LINK_MODES:
            assert word in err

        r = await client.post(url, json={"mode": "auto", "targets": "sibling_proc"})
        assert r.status == 400
        assert "list" in (await r.json())["error"]
        assert (
            await client.post(url, json={"mode": "auto", "targets": ["", "x"]})
        ).status == 400

        r = await client.post(url, json={"mode": "auto", "targets": ["nope_proc"]})
        assert r.status == 400
        assert "nope_proc" in (await r.json())["error"]

        assert (
            await client.post("/api/v1/jobs/nope/links/mode", json={"mode": "auto"})
        ).status == 404

        # Nothing above was applied, which is the all-or-nothing half: a refused batch that
        # left one target set is the state no error message covers.
        assert job.context.link_modes == {}
        assert [i for i in job.interventions if i["action"] == "link_mode_set"] == []
    finally:
        await client.close()

    # A run holding no links, with `targets` omitted: there is nothing to enumerate.
    jm2, _g2 = _mode_jm({"sibling_proc": 27})
    job2 = jm2.create_job(_incident())
    await jm2.run_job(job2)
    client = await _client(IncidentInputInterface(_config(), job_manager=jm2))
    try:
        r = await client.post(
            f"/api/v1/jobs/{job2.job_id}/links/mode", json={"mode": "auto"}
        )
        assert r.status == 409
        assert "`targets`" in (await r.json())["error"]
    finally:
        await client.close()

    # And no job manager at all — jobs are not served here, which is not a job that is gone.
    client = await _client(IncidentInputInterface(_config()))
    try:
        assert (
            await client.post("/api/v1/jobs/x/links/mode", json={"mode": "auto"})
        ).status == 503
    finally:
        await client.close()


async def test_a_mode_set_with_NO_pack_reachable_says_so_instead_of_refusing():
    """The degradation that must not become a refusal, and must not be silent either.

    The name check needs the loaded pack, and a deployment can be up without one reachable from
    here. Refusing every request would make an advisory control fail on a run that is otherwise
    fine; accepting silently would let a typo become a setting nothing reads. So the request is
    honoured and `targets_checked_against_pack` reports which of the two happened — and with no
    pack there is no licence either, so an escalating ask on a link the pass never measured is
    clamped rather than granted on the strength of a missing check.
    """
    jm, _gen = _plan_jm()  # no `correlation` module, hence no pack
    job = await _linked_job(jm, [_link()])()

    client = await _client(IncidentInputInterface(_config(), job_manager=jm))
    try:
        r = await client.post(
            f"/api/v1/jobs/{job.job_id}/links/mode",
            json={"mode": "auto", "targets": ["anything_at_all"]},
        )
        assert r.status == 200
        d = await r.json()
        assert d["targets_checked_against_pack"] is False
        assert d["applied"][0]["mode"] == "planned"
        assert d["applied"][0]["mode_source"] == "clamp"
        assert job.context.link_modes == {"anything_at_all": "auto"}
    finally:
        await client.close()


async def test_the_escalation_BUDGET_is_readable_before_a_probe_could_be_WASTED():
    """Read endpoint for escalation budget, reported per RUNG and not for the lane.

    Both paid rungs ship armed, so what a deployment reads with nothing configured is the engine's
    own defaults — and each rung is flagged on its own budget, because a run can reach a probe and
    stop there. `escalation_budgeted` answers the weaker lane-wide question. Caps come back clamped
    through the engine's resolvers. Procedure names are included because the mode must be settable
    before correlation. The constructor-slice path here is not the shape `main()` builds; the next
    test covers the production wiring.
    """
    jm, _gen = _mode_jm({"sibling_proc": 27, "other_proc": 0})

    # No `correlation.links` block at all: every deployment that has not opted in.
    job = jm.create_job(_incident())
    await jm.run_job(job)
    client = await _client(IncidentInputInterface(_config(), job_manager=jm))
    try:
        r = await client.get(f"/api/v1/jobs/{job.job_id}/links")
        assert r.status == 200
        d = await r.json()
        e = d["escalation"]
        assert e["escalation_budgeted"] is True
        assert e["probes_budgeted"] is True
        assert e["max_probes_per_run"] == link_escalation.DEFAULT_MAX_PROBES_PER_RUN
        assert e["children_budgeted"] is True
        assert e["max_children_per_run"] == link_children.DEFAULT_MAX_CHILDREN_PER_RUN
        # The vocabulary and the engine's own default, so the picker never types its own.
        assert e["modes"] == list(link_escalation.LINK_MODES)
        assert e["engine_default"] == link_escalation.DEFAULT_LINK_MODE
        # "" and not the default: nothing was configured, which is a different fact from a
        # deployment that chose the default on purpose.
        assert e["config_mode"] == ""
        # Answers BEFORE correlation — this is the one surface that can prevent a wasted run,
        # and refusing here would make it reachable only after the run it was meant to shape.
        assert d["correlated"] is False and d["link_count"] == 0
        assert d["job_modes"] == {}
        # Both procedures, including the one with no measured corpus: a base rate is
        # informational and licenses nothing, so it cannot decide what is settable.
        assert d["procedures"] == ["other_proc", "sibling_proc"]
    finally:
        await client.close()

    # And with a budget configured above the ceilings — the clamp is what gets reported.
    cfg = {
        **_config(),
        "correlation": {
            "links": {
                "escalation_mode": "auto",
                "max_probes_per_run": 99,
                "max_children_per_run": 99,
                "max_child_depth": 9,
            }
        },
    }
    client = await _client(IncidentInputInterface(cfg, job_manager=jm))
    try:
        d = await (await client.get(f"/api/v1/jobs/{job.job_id}/links")).json()
        e = d["escalation"]
        assert e["config_mode"] == "auto"
        assert e["escalation_budgeted"] is True
        assert e["max_probes_per_run"] == link_escalation.MAX_PROBES_CEILING
        assert e["children_budgeted"] is True
        assert e["max_children_per_run"] == link_children.MAX_CHILDREN_PER_RUN_CEILING
        assert e["max_child_depth"] == link_children.MAX_CHILD_DEPTH_CEILING
        # The per-probe timeout is inside the collective deadline, not multiplied by it.
        assert e["max_probes_per_run"] * e["probe_timeout_seconds"] <= (
            link_escalation.PROBE_BUDGET_CEILING_SECONDS
        )
        assert e["probe_row_cap"] > 0
        # The child concurrency is the LLM layer's width minus the parent's own slot, read from
        # the manager rather than re-derived — a second reader could report a bound the spawner
        # does not apply.
        assert e["max_concurrent_children"] >= 1
    finally:
        await client.close()

    # A job that is gone, and a deployment not serving jobs at all: two different answers.
    client = await _client(IncidentInputInterface(_config(), job_manager=jm))
    try:
        assert (await client.get("/api/v1/jobs/nope/links")).status == 404
    finally:
        await client.close()
    client = await _client(IncidentInputInterface(_config()))
    try:
        assert (await client.get("/api/v1/jobs/x/links")).status == 503
    finally:
        await client.close()


async def test_the_escalation_BUDGET_is_read_from_the_WHOLE_config_and_not_from_THIS_slice():
    """Verifies that `correlation.links` is read from `live_config`, not the constructor slice.

    `main()` passes `main_config["incident_input"]` as `config` and the full dict as
    `live_config`, so `self.config["correlation"]` is absent in production. Both callers must
    read from `live_config`. Both halves of the note are asserted (present and absent) because
    a `not in` on a phrase nobody emits cannot fail.
    """
    jm, _gen = _mode_jm({"sibling_proc": 27})
    job = await _linked_job(jm, [_link(gate_outcome="pass")])()
    links = {
        "escalation_mode": "auto",
        "max_probes_per_run": 4,
        "max_children_per_run": 2,
        "max_child_depth": 1,
    }
    live = {"incident_input": _config(), "correlation": {"links": links}}

    client = await _client(
        IncidentInputInterface(_config(), job_manager=jm, live_config=live)
    )
    try:
        d = await (await client.get(f"/api/v1/jobs/{job.job_id}/links")).json()
        e = d["escalation"]
        assert e["config_mode"] == "auto"
        assert e["escalation_budgeted"] is True and e["max_probes_per_run"] == 4
        assert e["children_budgeted"] is True and e["max_children_per_run"] == 2

        r = await client.post(
            f"/api/v1/jobs/{job.job_id}/links/mode",
            json={"mode": "auto", "targets": ["sibling_proc"]},
        )
        assert r.status == 200
        applied = (await r.json())["applied"][0]
        assert (applied["mode"], applied["mode_source"]) == ("auto", "job")
        assert "no escalation budget is configured" not in applied["mode_note"]

        # A `live` edit, in place, and the endpoint moves with it. Only rung 3's own flag
        # moves: rung 4 still has a budget, so the lane can still spend and saying otherwise
        # would send a reader to raise a number that is already raised.
        links["max_probes_per_run"] = 0
        e = (await (await client.get(f"/api/v1/jobs/{job.job_id}/links")).json())[
            "escalation"
        ]
        assert e["probes_budgeted"] is False and e["max_probes_per_run"] == 0
        assert e["escalation_budgeted"] is True and e["children_budgeted"] is True
    finally:
        await client.close()

    # The other direction of the note, so the assertion above is not vacuous: with BOTH paid
    # rungs zeroed, the same licensed link is told so — in words, beside a mode that is still
    # `auto`, because an empty budget is a bound and not a refusal.
    jm2, _gen2 = _mode_jm({"sibling_proc": 27})
    job2 = await _linked_job(jm2, [_link(gate_outcome="pass")])()
    client = await _client(
        IncidentInputInterface(
            _config(),
            job_manager=jm2,
            live_config={
                "correlation": {
                    "links": {"max_probes_per_run": 0, "max_children_per_run": 0}
                }
            },
        )
    )
    try:
        d = await (
            await client.post(
                f"/api/v1/jobs/{job2.job_id}/links/mode",
                json={"mode": "auto", "targets": ["sibling_proc"]},
            )
        ).json()
        applied = d["applied"][0]
        assert applied["mode"] == "auto"
        assert "no escalation budget is configured" in applied["mode_note"]
    finally:
        await client.close()


async def test_a_referral_LAUNCHED_by_hand_is_recorded_on_the_job_that_COMPOSED_it():
    """Rung 4 by hand, and the trail that keeps it from being a rumour.

    The endpoint composes and never launches, so an operator who acts on it POSTs the request
    themselves — and the parent then holds a referral its own audit trail says was "not launched"
    while a child run exists. That gap is the one this optional field closes: the child's id comes
    back, is recorded as `link_referral_launched`, and the two interventions stay separate words
    because composing and launching are separate decisions.

    `launched` stays False regardless, which is not a contradiction: it is a claim about what THIS
    endpoint did, and flipping it on a caller's say-so would make the field unable to answer the
    question it exists for. And an id that resolves to no job is REFUSED — an unverifiable claim in
    an audit trail is worse than a gap, because a reader cannot tell it from a run that happened.
    """
    jm, _gen = _mode_jm({"sibling_proc": 27})
    job = await _linked_job(jm, [_link()])()
    child = jm.create_job({**_incident(), "id": "INC-CHILD"})

    client = await _client(IncidentInputInterface(_config(), job_manager=jm))
    try:
        url = f"/api/v1/jobs/{job.job_id}/links/0/refer"
        r = await client.post(
            url, json={"actor": "mehdi", "launched_job_id": child.job_id}
        )
        assert r.status == 200
        d = await r.json()
        assert d["launched"] is False
        assert d["launch_recorded"] == child.job_id
        iv = job.interventions[-1]
        assert iv["action"] == "link_referral_launched"
        assert iv["stage"] == "correlation"
        assert iv["actor"] == "mehdi"
        assert child.job_id in iv["detail"] and "sibling_proc" in iv["detail"]

        # An id naming no job: refused, and nothing recorded — including the compose, because a
        # partial record of a claim that was rejected is the state no error message covers.
        before = len(job.interventions)
        r = await client.post(url, json={"launched_job_id": "not-a-job"})
        assert r.status == 409
        assert "not-a-job" in (await r.json())["error"]
        assert len(job.interventions) == before

        # Omitted, and it is the composed wording again — the old path, byte for byte.
        r = await client.post(url, json={})
        assert r.status == 200
        assert (await r.json())["launch_recorded"] == ""
        assert job.interventions[-1]["action"] == "link_referral_composed"
        assert "not launched" in job.interventions[-1]["detail"]
    finally:
        await client.close()


async def test_links_reach_the_job_JSON_through_the_endpoint_an_operator_POSTS_to():
    """End-to-end: a link from a real correlation stage, serialised through `snapshot()`.

    Other tests attach correlation output after the run; none exercises the full path
    `CorrelationResult` -> `summarize_stage` -> `snapshot()` -> `GET /jobs/{id}`. Also
    asserts the export/import round trip: `outputs["correlation"].links` must come back
    typed, not just as the summary's clipped view.
    """

    async def correlate(ctx):
        return CorrelationResult(
            record_count=3,
            links=[
                _link(state="probed_positive", pivot_values=["A-1", "A-2"]),
                _link(
                    target_use_case="other_proc",
                    state="probed_negative",
                    direction="consequent",
                    rung=1,
                    evidence_note="checked, and its own scope gate excluded it",
                    advisory_severity="",
                ),
                _link(
                    target_use_case="third_proc",
                    state="not_probed",
                    rung=0,
                    evidence_note="",
                ),
                _link(
                    target_use_case="fourth_proc",
                    state="unreachable",
                    rung=0,
                    pivot_entity="",
                    pivot_values=[],
                    gap_reason="no pivot in this pack binds that subject entity",
                ),
            ],
        )

    jm = _jm(
        [
            StageDescriptor("understanding", _ok_understanding, "understanding"),
            StageDescriptor("correlation", correlate, "correlation"),
        ]
    )
    launched = []

    def launch_fn(incident, mode):
        job = jm.create_job(incident, run_mode=JobRunMode(mode))
        launched.append(asyncio.ensure_future(jm.run_job(job)))
        return job

    iface = IncidentInputInterface(_config(), job_manager=jm, launch_fn=launch_fn)
    client = await _client(iface)
    try:
        r = await client.post("/api/v1/jobs", json={"description": _PARENT_PROSE})
        assert r.status == 201
        job_id = (await r.json())["job_id"]
        await asyncio.gather(*launched)

        r = await client.get(f"/api/v1/jobs/{job_id}")
        assert r.status == 200
        snap = await r.json()
        stage = next(s for s in snap["stages"] if s["name"] == "correlation")
        assert stage["status"] == StageStatus.COMPLETED.value
        links = stage["summary"]["links"]
        assert stage["summary"]["link_count"] == 4

        # The four states arrive whole and stay four. Anything that folded them into a boolean
        # (`probed: true/false`) would erase the pair that licenses OPPOSITE next steps — a
        # `probed_negative` rendered like a `not_probed` is a finding rendered as silence.
        assert [f["state"] for f in links] == [
            "probed_positive",
            "probed_negative",
            "not_probed",
            "unreachable",
        ]
        # And the fields the card and the refer button read arrive with them.
        assert links[0]["pivot_values"] == ["A-1", "A-2"]
        assert links[0]["pivot_entity"] == "actor"
        assert links[0]["direction"] == "antecedent"
        assert links[0]["advisory_severity"] == "HIGH"
        assert links[1]["evidence_note"].startswith("checked, and its own scope gate")
        assert links[3]["gap_reason"].startswith("no pivot in this pack")
        assert [f["target_use_case"] for f in links[1:]] == [
            "other_proc",
            "third_proc",
            "fourth_proc",
        ]

        # The verdict lane is untouched by any of it: this run reached no verdict, and the
        # advisory lane must not have invented one.
        assert stage["summary"]["verdict"] is None

        # A restored job can still be acted on, which needs the TYPED findings and not the
        # summary's clipped view.
        r = await client.get(f"/api/v1/jobs/{job_id}/export")
        assert r.status == 200
        doc = await r.json()
        jm2 = _jm([StageDescriptor("correlation", correlate, "correlation")])
        restored = jm2.import_job(doc)
        got = restored.context.outputs["correlation"].links
        assert [f.state for f in got] == [
            "probed_positive",
            "probed_negative",
            "not_probed",
            "unreachable",
        ]
        assert got[0].pivot_values == ["A-1", "A-2"]
    finally:
        await client.close()


# --- 5. numeric tuning: feedback -> anomaly_detection.threshold -------------


def _tuner(data_dir, **cfg):
    """A FeedbackLoop with tuning ON and a 0.80 baseline unless overridden.

    batch_size is parked out of reach so these tests exercise tuning alone — a
    distillation would fire the (mocked) LLM and confuse the assertion.
    """
    base = {
        "batch_size": 10_000,
        "auto_tune_threshold": True,
        "min_reviews_for_tuning": 3,
        "threshold_step": 0.05,
        "max_threshold_drift": 0.15,
    }
    base.update(cfg)
    return FeedbackLoop(MagicMock(), base, anomaly_config={"threshold": 0.8})


async def _submit(loop, n, *, false_positives=None, missed=None):
    for i in range(n):
        kw = {}
        if false_positives:
            kw["false_positives"] = [f"fp {i}"]
        if missed:
            kw["missed_anomalies"] = [f"miss {i}"]
        await loop.collect_feedback(f"INC{i}-{id(kw)}", None, "review", **kw)


async def test_false_positive_reports_raise_the_threshold(data_dir):
    loop = _tuner(data_dir)
    await _submit(loop, 3, false_positives=True)

    rec = await loop.tune_threshold()
    assert rec["direction"] == "raise"
    assert rec["recommended"] == 0.85
    assert rec["persisted"] is True
    assert loop.effective_threshold() == 0.85


async def test_missed_anomaly_reports_lower_the_threshold(data_dir):
    loop = _tuner(data_dir)
    await _submit(loop, 3, missed=True)

    rec = await loop.tune_threshold()
    assert rec["direction"] == "lower"
    assert rec["recommended"] == 0.75
    assert loop.effective_threshold() == 0.75


async def test_mixed_reports_leave_the_threshold_alone(data_dir):
    """Balanced complaints mean the ranking is wrong, not the cut-off."""
    loop = _tuner(data_dir)
    await _submit(loop, 2, false_positives=True)
    await _submit(loop, 2, missed=True)

    rec = await loop.tune_threshold()
    assert rec["direction"] == "hold"
    assert rec["persisted"] is False
    assert loop.effective_threshold() == 0.8
    assert "No clear direction" in rec["reason"]


async def test_a_review_reporting_both_does_not_vote_twice(data_dir):
    loop = _tuner(data_dir)
    for i in range(4):
        await loop.collect_feedback(
            f"INC{i}", None, None, false_positives=["a"], missed_anomalies=["b"]
        )
    rec = await loop.tune_threshold()
    assert rec["false_positive_reports"] == 0
    assert rec["missed_anomaly_reports"] == 0
    assert rec["direction"] == "hold"


async def test_too_few_reviews_is_not_enough_evidence(data_dir):
    """One analyst's single review must not retune detection sensitivity."""
    loop = _tuner(data_dir, min_reviews_for_tuning=5)
    await _submit(loop, 2, false_positives=True)

    rec = await loop.tune_threshold()
    assert rec["direction"] == "hold"
    assert rec["persisted"] is False
    assert "Not enough new reviews" in rec["reason"]
    assert loop.effective_threshold() == 0.8


async def test_free_text_alone_never_moves_the_number(data_dir):
    """Prose is not scanned for a direction — only structured fields count."""
    loop = _tuner(data_dir)
    for i in range(6):
        await loop.collect_feedback(
            f"INC{i}", None, "way too many false positives, raise the threshold!"
        )
    rec = await loop.tune_threshold()
    assert rec["false_positive_reports"] == 0
    assert rec["direction"] == "hold"
    assert loop.effective_threshold() == 0.8


async def test_drift_from_the_configured_baseline_is_capped(data_dir):
    """Sustained one-sided feedback cannot walk the threshold anywhere it likes."""
    loop = _tuner(data_dir, max_threshold_drift=0.10)
    for round_no in range(6):
        for i in range(3):
            await loop.collect_feedback(
                f"INC{round_no}-{i}", None, None, false_positives=["noise"]
            )
        await loop.tune_threshold()

    # baseline 0.80 + drift cap 0.10 -> 0.90, and no further.
    assert loop.effective_threshold() == 0.9
    rec = loop.threshold_recommendation()
    assert rec["direction"] == "hold"
    assert "already at the bound" in rec["reason"]


async def test_the_hard_ceiling_wins_over_the_drift_allowance(data_dir):
    loop = FeedbackLoop(
        MagicMock(),
        {
            "auto_tune_threshold": True,
            "min_reviews_for_tuning": 1,
            "threshold_step": 0.2,
            "max_threshold_drift": 0.9,
            "threshold_max": 0.95,
            "batch_size": 10_000,
        },
        anomaly_config={"threshold": 0.9},
    )
    await _submit(loop, 3, false_positives=True)
    await loop.tune_threshold()
    assert loop.effective_threshold() == 0.95


async def test_the_same_reviews_cannot_be_counted_twice(data_dir):
    """The watermark is what stops integral windup on a single batch of complaints."""
    loop = _tuner(data_dir)
    await _submit(loop, 3, false_positives=True)

    first = await loop.tune_threshold()
    assert first["persisted"] is True
    second = await loop.tune_threshold()  # no NEW reviews since
    assert second["persisted"] is False
    assert second["reviews_considered"] == 0
    assert loop.effective_threshold() == 0.85  # moved once, not twice


async def test_tuning_is_opt_in(data_dir):
    """Off by default: the recommendation is computed but never applied."""
    loop = FeedbackLoop(
        MagicMock(),
        {"min_reviews_for_tuning": 3, "batch_size": 10_000},
        anomaly_config={"threshold": 0.8},
    )
    assert loop.auto_tune_threshold is False
    await _submit(loop, 4, false_positives=True)

    rec = await loop.tune_threshold()
    # It still tells a human what it WOULD do — that's the point of computing it.
    assert rec["direction"] == "raise"
    assert rec["recommended"] == 0.85
    assert rec["persisted"] is False
    assert rec["applied"] is False
    assert loop.effective_threshold() == 0.8


async def test_disabling_tuning_reverts_without_deleting_history(data_dir):
    loop = _tuner(data_dir)
    await _submit(loop, 3, false_positives=True)
    await loop.tune_threshold()
    assert loop.effective_threshold() == 0.85

    loop.auto_tune_threshold = False
    assert loop.effective_threshold() == 0.8  # configured baseline is back
    assert len(loop.load_tuning_records()) == 1  # history intact


async def test_reset_restores_the_configured_baseline(data_dir):
    loop = _tuner(data_dir)
    await _submit(loop, 3, false_positives=True)
    await loop.tune_threshold()
    assert loop.effective_threshold() == 0.85

    assert loop.reset_threshold() is True
    assert loop.effective_threshold() == 0.8
    assert loop.load_tuning_records() == []


async def test_a_tuned_value_is_reclamped_when_the_operator_edits_the_config(data_dir):
    """A value tuned under old settings must not survive as an out-of-range threshold."""
    loop = _tuner(data_dir)
    await _submit(loop, 3, false_positives=True)
    await loop.tune_threshold()
    assert loop.effective_threshold() == 0.85

    # The operator later tightens the allowance to 0.02.
    loop.max_threshold_drift = 0.02
    assert loop.effective_threshold() == 0.82
    # ...and separately moves the baseline itself.
    assert loop.effective_threshold(configured=0.5) == 0.52


async def test_a_corrupt_tuning_file_falls_back_to_configured(data_dir):
    loop = _tuner(data_dir)
    (data_dir / "feedback_thresholds.json").write_text("{not json")
    assert loop.effective_threshold(refresh=True) == 0.8


async def test_distillation_also_retunes_the_threshold(data_dir):
    """The numeric half rides the same batch boundary as the prose half."""
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=_insights())
    loop = FeedbackLoop(
        llm,
        {"batch_size": 3, "auto_tune_threshold": True, "min_reviews_for_tuning": 3},
        anomaly_config={"threshold": 0.8},
    )
    await _submit(loop, 3, false_positives=True)

    llm.structured_output.assert_awaited_once()  # prose distilled
    assert loop.effective_threshold() == 0.85  # and the number moved


async def test_anomaly_detection_filters_on_the_tuned_threshold(data_dir):
    """End to end: the stage that filters must actually see the tuned value."""
    loop = _tuner(data_dir)
    await _submit(loop, 3, missed=True)  # -> lower to 0.75
    await loop.tune_threshold()

    def _anom(score):
        return AnomalyItem(
            description="d",
            supporting_data="s",
            potential_implications="i",
            confidence_score=score,
            recommended_actions="a",
            patterns="p",
        )

    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=AnomalyList(anomalies=[_anom(0.78), _anom(0.60)])
    )
    module = AnomalyDetectionModule({"threshold": 0.8}, llm, feedback=loop)
    assert module.effective_threshold() == 0.75

    found = await module.detect({"s": [{"a": 1}]}, _understanding(), None)
    # 0.78 would be held back at 0.80; feedback lowers the threshold. Both items come
    # back; the cut-off affects narration, not list length.
    assert [a.confidence_score for a in found] == [0.78, 0.60]
    assert [a.below_threshold for a in found] == [False, True]
    assert module.last_threshold_used == 0.75


async def test_anomaly_detection_falls_back_when_the_loop_misbehaves(data_dir):
    """Detection sensitivity must not depend on the feedback store being readable."""
    llm = MagicMock()
    broken = MagicMock()
    broken.effective_threshold.side_effect = RuntimeError("disk gone")
    module = AnomalyDetectionModule({"threshold": 0.8}, llm, feedback=broken)
    assert module.effective_threshold() == 0.8

    # And with no feedback loop at all, behaviour is exactly as before the feature.
    assert AnomalyDetectionModule({"threshold": 0.8}, llm).effective_threshold() == 0.8


async def test_threshold_endpoints(data_dir):
    loop = _tuner(data_dir)
    await _submit(loop, 3, false_positives=True)
    await loop.tune_threshold()

    iface = IncidentInputInterface(_config(), feedback_loop=loop)
    client = await _client(iface)
    try:
        body = await (await client.get("/api/v1/feedback/threshold")).json()
        assert body["effective"] == 0.85
        assert body["threshold"]["baseline"] == 0.8
        assert len(body["adjustments"]) == 1
        assert body["adjustments"][0]["direction"] == "raise"

        # The baseline can be overridden for a what-if calculation.
        body = await (
            await client.get("/api/v1/feedback/threshold?configured=0.6")
        ).json()
        assert body["threshold"]["baseline"] == 0.6
        assert (
            await client.get("/api/v1/feedback/threshold?configured=abc")
        ).status == 400

        # Reset reverts to the configured baseline.
        r = await client.post("/api/v1/feedback/threshold/reset")
        assert r.status == 200
        assert (await r.json())["effective"] == 0.8
    finally:
        await client.close()


async def test_threshold_endpoints_503_without_a_loop():
    iface = IncidentInputInterface(_config(), feedback_fn=AsyncMock())
    client = await _client(iface)
    try:
        assert (await client.get("/api/v1/feedback/threshold")).status == 503
        assert (await client.post("/api/v1/feedback/threshold/reset")).status == 503
    finally:
        await client.close()


async def test_stats_report_the_tuning_state(data_dir):
    loop = _tuner(data_dir)
    await _submit(loop, 3, false_positives=True)
    await loop.tune_threshold()

    stats = loop.stats()
    assert stats["auto_tune_threshold"] is True
    assert stats["threshold_adjustments"] == 1
    assert stats["effective_threshold"] == 0.85
    assert stats["false_positive_reports"] == 3
