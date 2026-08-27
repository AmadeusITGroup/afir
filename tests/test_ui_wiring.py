"""
The contract between the redesigned UI and the live job machinery.

`test_webui.py` type-checks the page in isolation and `test_ui_server.py` covers the server
modules. Neither runs a job, so a rename inside `Job.snapshot()` or a changed SSE event name
would leave both green while the Monitor tab silently renders nothing. This file drives the real
``IncidentInputInterface`` + ``JobManager`` over a TestServer and asserts on the precise fields
the client reads:

1. the SSE stream carries every type the basic log mode filters on. Basic mode is an allowlist
   (`BASIC_TYPES` in ``src/ui/script_tabs.py``), so a renamed event does not error, it vanishes
   from the operator's view;
2. `health` lives inside `stages[]`, not at the top level. `attachTo` read a non-existent
   `snap.stage_health` before the redesign;
3. a gate is answerable through the approvals inbox, with the `actions` the gate panel renders
   buttons for;
4. a rejection re-gates the same stage, so the panel survives being reopened on a stage it
   already showed;
5. the override editor round-trips real payloads: the value loaded out of `/export` under
   `OUTPUT_KEY[stage]` is the one the override endpoint accepts back, a broken one is a 400 and
   not a poisoned job, and `skip_stage` continues on the analyst's data.

The interface builds its rate limiter in ``start_server()``, which ``TestServer`` bypasses, hence
the explicit assignment in ``_client``.
"""

import asyncio
import json

from aiohttp.test_utils import TestClient, TestServer

from src.incident_input import IncidentInputInterface
from src.models.pydantic_models import (CorrelationResult, ExtractedEntity,
                                        IncidentAnalysis, UnderstandingResult)
from src.notifications import EventEmitter
from src.pipeline_runner import JobManager, JobRunMode, StageDescriptor
from src.ui.script_core import SCRIPT_CORE_JS
from src.ui.script_tabs import SCRIPT_TABS_JS
from src.utils.rate_limiter import AsyncRateLimiter

# The two stages are stand-ins for the real pipeline's first and fourth. Typed
# outputs matter: `_OUTPUT_CODECS` calls `.model_dump()`, so a dict here would make
# /export 500 and the assertion would blame the endpoint instead of the fixture.
UNDERSTANDING = UnderstandingResult(
    incident_id="INC-UI",
    analysis=IncidentAnalysis(
        incident_summary="Suspicious issuance on one org_unit",
        severity="7",
        severity_reasoning="high-value documents in a burst",
        impact_assessment="revenue loss",
        key_investigation_areas=["issuance"],
        log_sources_to_review=["record_lake"],
        initial_hypotheses=["SCHEME"],
        recommended_actions=["void the documents"],
        stakeholder_notification=["fraud team"],
        extracted_entities=[ExtractedEntity(type="org_unit", value="ORG2428D4")],
        correlation_keys=["org_unit"],
    ),
)
CORRELATION = CorrelationResult(
    record_count=12, correlated_data={"record": [1, 2]}, summary="12 rows"
)


def _stage(name, output):
    async def run(ctx):
        await asyncio.sleep(0)
        return output

    return StageDescriptor(name, run, name)


def _config():
    return {
        "host": "127.0.0.1",
        "port": 0,
        "post_incident_endpoint": "/api/v1/incidents",
        "post_ir_endpoint": "/api/v1/ir",
        "rate_limit": {"requests": 1000, "per_seconds": 60},
    }


async def _client(iface):
    # Built in start_server(), which TestServer skips; without it every handler
    # raises AttributeError on `self.rate_limiter.acquire()`.
    iface.rate_limiter = AsyncRateLimiter(rate_limit=1000, time_period=1)
    client = TestClient(TestServer(iface.app))
    await client.start_server()
    return client


def _failing_stage(name):
    async def run(ctx):
        raise RuntimeError("backend refused the query")

    return StageDescriptor(name, run, name)


async def _serve(stages=None, store=None):
    """A two-stage manager wired the way ``main()`` wires the real one."""
    stages = stages or [
        _stage("understanding", UNDERSTANDING),
        _stage("correlation", CORRELATION),
    ]
    emitter = EventEmitter()
    jm = JobManager(stages, emitter, store=store)
    emitter.set_job_manager(jm)
    # A test that leaves a job parked on a gate leaves its run_job task alive; without
    # somewhere to find it, teardown prints "Task was destroyed but it is pending".
    jm._test_tasks = []

    def launch_fn(incident, mode="auto"):
        job = jm.create_job(incident, run_mode=JobRunMode(str(mode).lower()))
        jm._test_tasks.append(asyncio.ensure_future(jm.run_job(job)))
        return job

    iface = IncidentInputInterface(_config(), job_manager=jm, launch_fn=launch_fn)
    client = await _client(iface)
    client._afir_iface = iface  # so _shutdown can stop the limiter's refill task
    return jm, client


async def _shutdown(jm, client):
    """Cancel the parked stage loops + the limiter refill task, then close the client.

    A test that ends with a job on a gate leaves `run_job` waiting on a future that
    will never resolve. Cancelling here keeps the noise out of the whole suite's
    output — an unrelated failure is hard enough to read without it.
    """
    for task in getattr(jm, "_test_tasks", []):
        task.cancel()
    await asyncio.gather(*getattr(jm, "_test_tasks", []), return_exceptions=True)
    iface = getattr(client, "_afir_iface", None)
    if iface is not None and iface.rate_limiter is not None:
        iface.rate_limiter.close()
    await client.close()


async def _start(client, mode="supervised"):
    resp = await client.post(
        "/api/v1/jobs", json={"description": "ui wiring incident", "mode": mode}
    )
    assert resp.status == 201
    return (await resp.json())["job_id"]


async def _await_gate(client, stage=None, tries=60):
    """Poll the approvals inbox until a gate (optionally on `stage`) is open."""
    for _ in range(tries):
        gates = (await (await client.get("/api/v1/gates")).json())["gates"]
        match = [g for g in gates if stage is None or g["stage"] == stage]
        if match:
            return match[0]
        await asyncio.sleep(0.02)
    raise AssertionError(f"no gate opened on {stage or 'any stage'}")


async def _await_status(client, job_id, status, tries=80):
    for _ in range(tries):
        snap = await (await client.get(f"/api/v1/jobs/{job_id}")).json()
        if snap["status"] == status:
            return snap
        await asyncio.sleep(0.02)
    raise AssertionError(f"job never reached {status}; last was {snap['status']}")


def _js():
    return SCRIPT_CORE_JS + SCRIPT_TABS_JS


def _js_array(name):
    """Read a `const NAME = [...]` literal out of the UI source."""
    js = _js()
    start = js.index("const " + name + " = [")
    return set(json.loads(js[js.index("[", start) : js.index("]", start) + 1]))


# --- 1. the event stream the Monitor tab lives on ---------------------------


async def test_sse_carries_every_type_basic_mode_shows():
    """Basic mode is an allowlist; a renamed event silently disappears from it."""
    jm, client = await _serve()
    try:
        job_id = await _start(client)
        seen = []

        async def listen():
            resp = await client.get(f"/api/v1/jobs/{job_id}/events")
            async for raw in resp.content:
                line = raw.decode().strip()
                if line.startswith("data:"):
                    seen.append(json.loads(line[5:]))

        task = asyncio.ensure_future(listen())
        try:
            gate = await _await_gate(client, "understanding")
            await client.post(
                f"/api/v1/jobs/{job_id}/gate",
                json={"action": "approve", "actor": "tester"},
            )
            await _await_gate(client, "correlation")
            await client.post(
                f"/api/v1/jobs/{job_id}/gate",
                json={"action": "approve", "actor": "tester"},
            )
            await _await_status(client, job_id, "completed")
            await asyncio.sleep(0.05)  # let the last frames drain
        finally:
            task.cancel()

        types = {e.get("type") for e in seen}
        # Every event carries the keys appendLog()/advancedFields() read.
        for event in seen:
            assert "type" in event and "job_id" in event
        # The lifecycle types basic mode filters TO must all actually be emitted.
        for expected in (
            "job_status",
            "stage_started",
            "stage_completed",
            "gate_opened",
            "gate_resolved",
        ):
            assert expected in types, f"{expected} never emitted"
        assert gate["stage"] == "understanding"
    finally:
        await _shutdown(jm, client)


async def test_basic_mode_allowlist_only_names_real_event_types():
    """A typo in BASIC_TYPES costs the operator an event class with no error."""
    jm, client = await _serve()
    try:
        job_id = await _start(client)
        seen = set()

        async def listen():
            resp = await client.get(f"/api/v1/jobs/{job_id}/events")
            async for raw in resp.content:
                line = raw.decode().strip()
                if line.startswith("data:"):
                    seen.add(json.loads(line[5:]).get("type"))

        task = asyncio.ensure_future(listen())
        try:
            await _await_gate(client, "understanding")
            await client.post(f"/api/v1/jobs/{job_id}/gate", json={"action": "approve"})
            await _await_gate(client, "correlation")
            await client.post(f"/api/v1/jobs/{job_id}/gate", json={"action": "approve"})
            await _await_status(client, job_id, "completed")
            await asyncio.sleep(0.05)
        finally:
            task.cancel()

        # Each of these needs something this run does not have (a clock, an override, a broken
        # stage, a pack declaring a follow-up pass) and is observed in its own test. Everything
        # else in the allowlist had to appear on this ordinary run.
        excused = {"gate_timeout", "intervention", "stage_failed", "pass_started"}
        unexplained = _js_array("BASIC_TYPES") - seen - excused
        assert not unexplained, f"BASIC_TYPES names types never emitted: {unexplained}"
    finally:
        await _shutdown(jm, client)


async def test_a_failing_stage_emits_stage_failed_and_offers_a_retry():
    """`stage_failed` is in the basic allowlist and the retry button depends on it."""
    jm, client = await _serve(
        [_stage("understanding", UNDERSTANDING), _failing_stage("correlation")]
    )
    try:
        job_id = await _start(client, mode="auto")
        seen = []

        async def listen():
            resp = await client.get(f"/api/v1/jobs/{job_id}/events")
            async for raw in resp.content:
                line = raw.decode().strip()
                if line.startswith("data:"):
                    seen.append(json.loads(line[5:]))

        task = asyncio.ensure_future(listen())
        try:
            snap = await _await_status(client, job_id, "stage_failed")
            await asyncio.sleep(0.05)
        finally:
            task.cancel()

        assert "stage_failed" in {e.get("type") for e in seen}
        # The card shows the error text and the retry control targets the stage.
        assert snap["error"]
        assert snap["current_stage"] == "correlation"
        assert (
            await client.post(
                f"/api/v1/jobs/{job_id}/control",
                json={"action": "skip_stage", "stage": "correlation"},
            )
        ).status == 200
    finally:
        await _shutdown(jm, client)


# --- 2. the snapshot shape the stage cards read ----------------------------


async def test_snapshot_carries_health_inside_the_stages_list():
    """`attachTo` reads `stages[].health`; there is no top-level `stage_health`."""
    jm, client = await _serve()
    try:
        job_id = await _start(client)
        await _await_gate(client, "understanding")
        snap = await (await client.get(f"/api/v1/jobs/{job_id}")).json()

        assert "stage_health" not in snap
        by_name = {s["name"]: s for s in snap["stages"]}
        health = by_name["understanding"]["health"]
        # The four fields the health chip renders.
        assert set(health) >= {"score", "threshold", "gate_recommended", "scored"}
        assert 0.0 <= health["score"] <= 1.0
        # A stage that has not run yet reports no health rather than a fake zero —
        # an unscored stage must not render as a failing one.
        assert by_name["correlation"]["health"] is None
        # The keys the drawer + timer read.
        assert set(snap) >= {
            "job_id",
            "incident_id",
            "status",
            "run_mode",
            "current_stage",
            "stages",
            "open_gate",
            "gate_history",
            "interventions",
        }
    finally:
        await _shutdown(jm, client)


async def test_gate_advertises_the_actions_the_panel_renders():
    jm, client = await _serve()
    try:
        await _start(client)
        gate = await _await_gate(client, "understanding")
        assert gate["actions"] == ["approve", "reject", "override"]
        # The inbox row needs these to be answerable from any tab.
        assert set(gate) >= {
            "job_id",
            "incident_id",
            "stage",
            "reason",
            "health",
            "actions",
            "opened_at",
        }
    finally:
        await _shutdown(jm, client)


class _QueueingStore:
    """A store whose writes only land on ``flush`` — a queueing backend's shape.

    ``DatabricksStorage`` hands writes to a background thread, so "saved" and "durable"
    stop being the same instant. The two saves that must not sit in a queue are the one
    that opens a gate and the one that records its resolution: both are followed by a
    wait measured in days, and a container death in the queued window asks the same
    analyst the same question twice.
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


async def test_resolving_a_gate_over_http_flushes_before_it_answers_200():
    """The flush is in the *handler*, so only a real request proves it is reached.

    ``test_pipeline_runner.py`` covers the gate-opening half against the runner directly.
    This half lives on the HTTP boundary — the analyst's 200 is the promise that the
    decision is recorded — and a guard nothing asks is not a guard.
    """
    store = _QueueingStore()
    jm, client = await _serve(store=store)
    try:
        job_id = await _start(client)
        await _await_gate(client, "understanding")
        flushes_at_gate = store.flushes

        resp = await client.post(
            f"/api/v1/jobs/{job_id}/gate",
            json={"action": "approve", "actor": "tester"},
        )
        assert resp.status == 200
        # Asserted on the 200 itself, not eventually: the response IS the receipt.
        assert store.flushes > flushes_at_gate, "the resolution was never flushed"
        assert store.landed[job_id]["gate_history"], "the decision did not land"
        assert not store.queued, "a write was still queued when the analyst got a 200"
    finally:
        await _shutdown(jm, client)


async def test_a_gate_resolves_over_http_with_a_store_that_cannot_flush():
    """Every current deployment: a synchronous store with no ``flush`` at all."""

    class _PlainStore:
        def __init__(self):
            self.saves = 0

        def save(self, doc, evidence_changed=False):
            self.saves += 1
            return True

        def prune(self):
            return 0

    store = _PlainStore()
    jm, client = await _serve(store=store)
    try:
        job_id = await _start(client)
        await _await_gate(client, "understanding")
        resp = await client.post(
            f"/api/v1/jobs/{job_id}/gate",
            json={"action": "approve", "actor": "tester"},
        )
        assert resp.status == 200
        assert store.saves
    finally:
        await _shutdown(jm, client)


# --- 3. a rejection re-opens the gate on the same stage --------------------


async def test_rejection_re_gates_the_same_stage_and_records_the_decision():
    jm, client = await _serve()
    try:
        job_id = await _start(client)
        await _await_gate(client, "understanding")
        assert (
            await client.post(
                f"/api/v1/jobs/{job_id}/gate",
                json={"action": "approve", "actor": "tester"},
            )
        ).status == 200

        await _await_gate(client, "correlation")
        resp = await client.post(
            f"/api/v1/jobs/{job_id}/gate",
            json={
                "action": "reject",
                "actor": "tester",
                "reason_code": "missing_evidence",
                "guidance": "also join on the document number",
            },
        )
        assert resp.status == 200
        # The stage re-runs with the guidance injected, then gates AGAIN.
        await _await_gate(client, "correlation")
        await client.post(
            f"/api/v1/jobs/{job_id}/gate", json={"action": "approve", "actor": "tester"}
        )
        snap = await _await_status(client, job_id, "completed")

        decisions = [(h["stage"], h["action"]) for h in snap["gate_history"]]
        assert decisions == [
            ("understanding", "approve"),
            ("correlation", "reject"),
            ("correlation", "approve"),
        ]
        rejection = snap["gate_history"][1]
        # renderTrail() reads exactly these three off a history row.
        assert rejection["actor"] == "tester"
        assert rejection["reason_code"] == "missing_evidence"
        assert rejection["guidance"]
    finally:
        await _shutdown(jm, client)


# --- 4. the override editor's round trip ----------------------------------


async def test_override_round_trips_through_export_and_continues_the_run():
    """ovLoad reads /export[OUTPUT_KEY], ovApply POSTs it back, ovSkip continues."""
    jm, client = await _serve()
    try:
        job_id = await _start(client)
        await _await_gate(client, "understanding")

        # ovLoad
        doc = await (await client.get(f"/api/v1/jobs/{job_id}/export")).json()
        value = doc["outputs"]["understanding"]  # OUTPUT_KEY["understanding"]
        assert value["analysis"]["severity"] == "7"

        # ovApply — a hand-edited value
        value["analysis"]["severity"] = "9"
        value["analysis"]["incident_summary"] = "Analyst-corrected summary"
        resp = await client.post(
            f"/api/v1/jobs/{job_id}/outputs/understanding",
            json={"value": value, "actor": "tester"},
        )
        assert resp.status == 200

        # A value that breaks the contract is a client error, not a poisoned job.
        bad = await client.post(
            f"/api/v1/jobs/{job_id}/outputs/understanding",
            json={"value": {"nope": 1}, "actor": "tester"},
        )
        assert bad.status == 400
        assert "understanding" in (await bad.json())["error"]

        # ovSkip
        assert (
            await client.post(
                f"/api/v1/jobs/{job_id}/control",
                json={"action": "skip_stage", "stage": "understanding"},
            )
        ).status == 200
        await _await_gate(client, "correlation")
        await client.post(
            f"/api/v1/jobs/{job_id}/gate", json={"action": "approve", "actor": "tester"}
        )
        snap = await _await_status(client, job_id, "completed")

        # The run finished on the analyst's number, and says so.
        doc = await (await client.get(f"/api/v1/jobs/{job_id}/export")).json()
        assert doc["outputs"]["understanding"]["analysis"]["severity"] == "9"
        overrides = [
            i for i in snap["interventions"] if i["action"] == "override_stage_output"
        ]
        assert overrides and overrides[0]["stage"] == "understanding"
        # renderTrail() reads `iv.action`; a rename would blank the What column.
        assert all("action" in i for i in snap["interventions"])
    finally:
        await _shutdown(jm, client)


async def test_override_emits_the_intervention_event_the_console_shows():
    jm, client = await _serve()
    try:
        job_id = await _start(client)
        seen = []

        async def listen():
            resp = await client.get(f"/api/v1/jobs/{job_id}/events")
            async for raw in resp.content:
                line = raw.decode().strip()
                if line.startswith("data:"):
                    seen.append(json.loads(line[5:]))

        task = asyncio.ensure_future(listen())
        try:
            await _await_gate(client, "understanding")
            doc = await (await client.get(f"/api/v1/jobs/{job_id}/export")).json()
            await client.post(
                f"/api/v1/jobs/{job_id}/outputs/understanding",
                json={"value": doc["outputs"]["understanding"], "actor": "tester"},
            )
            await asyncio.sleep(0.05)
        finally:
            task.cancel()
        assert "intervention" in {e.get("type") for e in seen}
    finally:
        await _shutdown(jm, client)


async def test_export_outputs_are_keyed_the_way_the_editor_expects():
    """OUTPUT_KEY maps stage -> output key; a mismatch loads `undefined`."""
    jm, client = await _serve()
    try:
        job_id = await _start(client, mode="auto")
        await _await_status(client, job_id, "completed")
        doc = await (await client.get(f"/api/v1/jobs/{job_id}/export")).json()

        js = _js()
        start = js.index("const OUTPUT_KEY = {")
        mapping = js[start : js.index("}", start)]
        # Only the two stages this fixture runs can be asserted present; the point is
        # that the key the client asks for is the key the export actually uses.
        assert (
            '"understanding"' in mapping
            or 'understanding:"understanding"' in mapping.replace(" ", "")
        )
        assert set(doc["outputs"]) >= {"understanding", "correlation"}
        assert doc["outputs"]["correlation"]["record_count"] == 12
    finally:
        await _shutdown(jm, client)


# --- 5. the jobs drawer ---------------------------------------------------


async def test_jobs_list_rows_carry_what_the_drawer_renders():
    jm, client = await _serve()
    try:
        job_id = await _start(client)
        await _await_gate(client, "understanding")
        rows = (await (await client.get("/api/v1/jobs")).json())["jobs"]
        row = [r for r in rows if r["job_id"] == job_id][0]
        assert set(row) >= {
            "job_id",
            "incident_id",
            "status",
            "run_mode",
            "current_stage",
            "created_at",
            "updated_at",
        }
        # A waiting job advertises WHERE it waits, so the drawer can badge it.
        assert row["awaiting_stage"] == "understanding"
    finally:
        await _shutdown(jm, client)


# --- 6. the config importer -----------------------------------------------


async def test_import_rejects_every_bad_file_before_writing_any(tmp_path, monkeypatch):
    """`POST /api/v1/config/import` must validate ALL files, then write.

    The endpoint promises "validates EVERY file before writing ANY of them", but the
    top-level-mapping rule was only enforced inside `replace_file` — at write time.
    So a non-mapping file returned a 500 (an I/O fault, not a bad request) *after*
    the valid files ahead of it in the batch had already landed, which is precisely
    the mismatched-half-an-environment state the all-or-nothing check exists to
    prevent. The importer had no HTTP-level test at all, which is how it survived.
    """
    from src import config_store

    monkeypatch.setattr(config_store, "config_dir", lambda: tmp_path)
    good = "anomaly_detection:\n  threshold: 0.5\n"
    (tmp_path / "main_config.yaml").write_text(good, encoding="utf-8")
    (tmp_path / "llm_config.yaml").write_text("model: keep-me\n", encoding="utf-8")

    iface = IncidentInputInterface(_config())
    client = await _client(iface)
    try:
        # A list is valid YAML but not a config file. Batched with a good file so a
        # write-time failure would be observable as a half-applied import.
        resp = await client.post(
            "/api/v1/config/import",
            json={
                "files": {
                    "main_config.yaml": good,
                    "llm_config.yaml": "- not\n- a\n- mapping\n",
                }
            },
        )
        assert resp.status == 400, await resp.text()
        body = await resp.json()
        assert body["error"] == "Import rejected; nothing was written"
        assert any("mapping" in e for e in body["errors"])
        # Nothing written: the untouched file still holds its original value.
        assert (tmp_path / "llm_config.yaml").read_text() == "model: keep-me\n"

        # And the happy path still writes.
        resp = await client.post(
            "/api/v1/config/import",
            json={"files": {"llm_config.yaml": "model: replaced\n"}},
        )
        assert resp.status == 200, await resp.text()
        assert (await resp.json())["restart_required"] is True
        assert "replaced" in (tmp_path / "llm_config.yaml").read_text()
    finally:
        if iface.rate_limiter is not None:
            iface.rate_limiter.close()
        await client.close()


async def test_editing_the_primary_budget_reaches_the_retrieval_engine(
    tmp_path, monkeypatch
):
    """`PUT /api/v1/config` on the primary timeout must actually tell the engine.

    ``applies="live"`` is a claim about EFFECT, not about storage. This field is
    genuinely live in the config dict — and completely inert there, because
    ``_apply_primary_budget`` COPIES the resolved budget onto every retriever's own
    config at engine-build time and nothing re-reads it. Marking it live without the
    ``on_live_reload`` hook would therefore report a number as applied while the old
    cap kept firing, which is worse than being told to restart: the operator raises the
    budget, watches the primary source time out at the previous one, and has no reason
    to doubt the screen.

    So both halves are pinned from the outside: wired, the path comes back under
    ``reloaded.applied`` AND the hook ran; unwired, the same path comes back under
    ``restart_required`` and never claims to be applied.
    """
    from src import config_store

    monkeypatch.setattr(config_store, "config_dir", lambda: tmp_path)
    (tmp_path / "main_config.yaml").write_text(
        "log_sources:\n  primary_source_timeout_seconds: 7200\n", encoding="utf-8"
    )

    async def _put(iface):
        client = await _client(iface)
        try:
            resp = await client.put(
                "/api/v1/config",
                json={"updates": {"log_sources.primary_source_timeout_seconds": 5400}},
            )
            assert resp.status == 200, await resp.text()
            return await resp.json()
        finally:
            if iface.rate_limiter is not None:
                iface.rate_limiter.close()
            await client.close()

    # 1. Wired: the file is written, the live dict updated, and the engine told.
    refreshes = []
    live = {"log_sources": {"primary_source_timeout_seconds": 7200}}
    body = await _put(
        IncidentInputInterface(
            _config(),
            live_config=live,
            on_live_reload=lambda: refreshes.append(True),
        )
    )
    assert "5400" in (tmp_path / "main_config.yaml").read_text()
    assert live["log_sources"]["primary_source_timeout_seconds"] == 5400
    assert body["reloaded"]["applied"] == [
        "log_sources.primary_source_timeout_seconds"
    ]
    assert body["reloaded"]["restart_required"] == []
    # The whole point: the module that CACHED the old budget was actually asked to
    # re-resolve it. Without this assertion the test passes on a dict write alone.
    assert refreshes == [True]

    # 2. Unwired (no hook): honest, not optimistic.
    (tmp_path / "main_config.yaml").write_text(
        "log_sources:\n  primary_source_timeout_seconds: 7200\n", encoding="utf-8"
    )
    live = {"log_sources": {"primary_source_timeout_seconds": 7200}}
    body = await _put(IncidentInputInterface(_config(), live_config=live))
    assert body["reloaded"]["applied"] == []
    assert body["reloaded"]["restart_required"] == [
        "log_sources.primary_source_timeout_seconds"
    ]


def test_the_retrieval_budgets_section_is_reachable_from_the_ui():
    """A field the operator cannot see is not configurable.

    The Configuration tab renders one control per descriptor in
    ``config_store.SECTIONS``, so a budget that exists only in the YAML is editable
    only by hand-editing a file on a Databricks App — the thing this section exists to
    avoid. Pinned here rather than in ``test_ui_server.py`` because what matters is the
    round trip: the section is declared, its field validates the value the form sends,
    and it is marked live.
    """
    from src import config_store

    section = next(
        (s for s in config_store.SECTIONS if s[0] == "retrieval"),
        None,
    )
    assert section is not None, "no 'retrieval' section — the budgets are invisible"
    paths = [f.path for f in section[2]]
    assert "log_sources.primary_source_timeout_seconds" in paths
    field = config_store.FIELDS["log_sources.primary_source_timeout_seconds"]
    assert field.applies == "live"
    assert field.default == 7200, "the shipped default must be the documented 2 hours"
    # The bounds are the operator's guard rails: generous up, never absurdly low.
    accepted, errors = config_store.validate_updates(
        {"log_sources.primary_source_timeout_seconds": 7200}
    )
    assert not errors and accepted
    _, errors = config_store.validate_updates(
        {"log_sources.primary_source_timeout_seconds": 5}
    )
    assert errors, "a 5-second cap on a primary source must not validate"


# --- 7. the report route, both spellings ----------------------------------


async def test_job_keyed_and_incident_keyed_report_routes_agree(tmp_path, monkeypatch):
    """`/jobs/{id}/report` must serve what `/incidents/{id}/report` serves.

    The job-keyed handler read `job.outputs`, but outputs live on the job's CONTEXT
    (`Job.context = ctx`). So every job-keyed report request raised AttributeError →
    500, while the incident-keyed spelling — same files, read off disk — worked. The
    Report tab uses whichever id it has, so half the buttons were dead and no test
    touched this route.
    """
    from src import report_delivery

    monkeypatch.setattr(report_delivery, "exports_dir", lambda: tmp_path)
    jm, client = await _serve()
    try:
        job_id = await _start(client, mode="auto")
        await _await_status(client, job_id, "completed")
        incident_id = (await (await client.get(f"/api/v1/jobs/{job_id}")).json())[
            "incident_id"
        ]
        # A report on disk is what both spellings resolve to.
        (tmp_path / f"fraud_report_{incident_id}.md").write_text(
            "# Report\n\nbody\n", encoding="utf-8"
        )
        for base in (f"/api/v1/jobs/{job_id}", f"/api/v1/incidents/{incident_id}"):
            resp = await client.get(base + "/report?format=md")
            assert resp.status == 200, f"{base}: {await resp.text()}"
            assert "# Report" in await resp.text()
        # A bad format is a 400 on both, not a 500.
        for base in (f"/api/v1/jobs/{job_id}", f"/api/v1/incidents/{incident_id}"):
            resp = await client.get(base + "/report?format=bogus")
            assert resp.status == 400, f"{base}: {resp.status}"
        # An unknown job is 404 — never a 500 from touching a missing attribute.
        resp = await client.get("/api/v1/jobs/nosuchjob/report?format=md")
        assert resp.status == 404
    finally:
        await _shutdown(jm, client)
