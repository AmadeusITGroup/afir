"""Tests for incident input normalization, validation, and inline serving."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.config_store import REDACTED
from src.incident_input import (
    IncidentInputInterface,
    _audit_changes,
    _pass_number,
    validate_incident,
)
from src.knowledge.pack import EntityDef, KnowledgePack, SourceDef
from src.utils.rate_limiter import AsyncRateLimiter


def _config():
    return {
        "post_incident_endpoint": "/api/v1/incidents",
        "post_ir_endpoint": "/api/v1/ir",
        "post_freetext_endpoint": "/api/v1/incidents/freetext",
        "port": 5000,
        "rate_limit": {"requests": 100, "per_seconds": 60},
    }


def test_validate_incident_requires_only_description():
    assert validate_incident({"description": "something happened"})
    assert not validate_incident({"description": ""})
    assert not validate_incident({"id": "X"})


def test_normalize_fills_id_and_timestamp():
    incident = IncidentInputInterface._normalize_incident(
        {"description": "x"}, source="freetext"
    )
    assert incident["description"] == "x"
    assert incident["id"]  # auto-generated uuid
    assert incident["timestamp"]  # auto-generated iso timestamp
    assert incident["source"] == "freetext"


def test_normalize_preserves_existing_fields():
    incident = IncidentInputInterface._normalize_incident(
        {"id": "INC-1", "timestamp": "2024-01-01", "description": "x"}, source="api"
    )
    assert incident["id"] == "INC-1"
    assert incident["timestamp"] == "2024-01-01"


def test_constructor_registers_freetext_route_and_defers_rate_limiter():
    iface = IncidentInputInterface(_config())
    # rate limiter is created in start_server (needs a running loop), not __init__
    assert iface.rate_limiter is None
    paths = {r.resource.canonical for r in iface.app.router.routes()}
    assert "/api/v1/incidents/freetext" in paths
    assert "/health" in paths
    assert "/" in paths


async def _client(iface):
    """Build an aiohttp test client and arm the (loop-bound) rate limiter."""
    iface.rate_limiter = AsyncRateLimiter(100, 60)
    client = TestClient(TestServer(iface.app))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_health_returns_ok():
    iface = IncidentInputInterface(_config())
    client = await _client(iface)
    try:
        resp = await client.get("/health")
        assert resp.status == 200
        assert (await resp.json())["status"] == "ok"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_the_deep_health_answer_is_opt_in_and_names_the_credential():
    """Two endpoints in one, and the split matters in both directions.

    Bare ``/health`` is the platform's liveness probe: fast, dependency-free, and asserted
    byte-for-byte above — it must not start depending on a collaborator being wired.
    ``?deep=1`` answers the question the probe cannot, which is the failure that actually
    costs a run: a reachable server with an empty LLM credential 401s all six LLM stages,
    and today the app says so only in a container log the operator cannot read.

    ``None`` and ``False`` are deliberately different answers. Every test that builds this
    interface directly has no ``llm_client``, and a pure-export deployment legitimately has
    none either — reporting *"not wired"* as a failure is how an indicator learns to cry
    wolf and stops being read.
    """
    iface = IncidentInputInterface(_config())
    client = await _client(iface)
    try:
        shallow = await (await client.get("/health")).json()
        assert shallow == {"status": "ok"}, "the platform probe changed shape"

        deep = await (await client.get("/health?deep=1")).json()
        assert deep["status"] == "ok"
        assert set(deep) == {
            "status",
            "llm_credential",
            "pack",
            "pack_detail",
            "jobs",
            "pipeline",
            "storage",
            "storage_ok",
            "storage_detail",
            "sources_declared",
            "sources_unavailable",
            "retrieval_cache",
            "run_queue",
        }
        assert deep["llm_credential"] is None  # nothing wired, not "unusable"
        assert deep["pipeline"] is False
        assert deep["storage"] is None and deep["storage_ok"] is None
        assert deep["sources_declared"] is None
        assert deep["sources_unavailable"] is None
        assert deep["retrieval_cache"] is None
        assert deep["run_queue"] is None
        assert deep["pack"] is None and deep["pack_detail"] is None

        # Wired but with no usable credential is the amber state the dot exists for.
        class _Client:
            credential_available = False

        iface.llm_client = _Client()
        deep = await (await client.get("/health?deep=1")).json()
        assert deep["llm_credential"] is False

        # And an unrecognised value of the parameter is not a deep request.
        assert await (await client.get("/health?deep=0")).json() == {"status": "ok"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_finished_incidents_are_listable_on_the_incidents_path():
    """A GET beside the existing POST, on the same configured path: aiohttp keys on
    (method, path), and registering them together means a deployment that renames the
    endpoint cannot split the pair."""
    iface = IncidentInputInterface(_config())
    client = await _client(iface)
    try:
        body = await (await client.get("/api/v1/incidents?limit=5")).json()
        assert isinstance(body["incidents"], list)
        # A junk limit must not 500 the Report tab's first load.
        assert (await client.get("/api/v1/incidents?limit=abc")).status == 200
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_index_serves_console_ui():
    iface = IncidentInputInterface(_config())
    client = await _client(iface)
    try:
        resp = await client.get("/")
        assert resp.status == 200
        assert resp.content_type == "text/html"
        body = await resp.text()
        # New SOC-console UI markers (not the old plain page).
        assert "Fraud Investigation Console" in body
        assert "EventSource" in body
        assert "stage_output" in body
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_the_named_alias_serves_the_same_page_and_needs_no_identity():
    """`/afir` exists so a bookmark says what it opens under a proxy prefix that does not.

    Both spellings, because no trailing-slash normalisation middleware is installed, and
    both unauthenticated like `/` — the page has to load before it can say who the caller
    is, and an alias that 403s where the canonical path loads is worse than no alias.
    """
    iface = IncidentInputInterface(_config())
    client = await _client(iface)
    try:
        canonical = await (await client.get("/")).text()
        for path in ("/afir", "/afir/"):
            resp = await client.get(path)
            assert resp.status == 200, path
            assert resp.content_type == "text/html", path
            assert await resp.text() == canonical, path
        assert "/afir" in iface._UNAUTHENTICATED_PATHS
        assert "/afir/" in iface._UNAUTHENTICATED_PATHS
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_freetext_runs_pipeline_inline():
    process_fn = AsyncMock(return_value={"sections": [{"section_title": "Summary"}]})
    iface = IncidentInputInterface(_config(), process_fn=process_fn)
    client = await _client(iface)
    try:
        resp = await client.post(
            "/api/v1/incidents/freetext", json={"description": "suspicious transfer"}
        )
        assert resp.status == 200
        body = await resp.json()
        assert "incident_id" in body
        assert body["report"] == {"sections": [{"section_title": "Summary"}]}
        process_fn.assert_awaited_once()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_freetext_verbose_returns_stages_and_events():
    """?verbose=1 exposes per-stage timings + event history alongside the report."""

    class _FakeJob:
        job_id = "job-xyz"
        _event_history = [{"type": "stage_completed", "stage": "understanding"}]

        class _Ctx:
            outputs = {"report": "REPORT-TEXT"}

        context = _Ctx()

        def snapshot(self):
            return {
                "stages": [
                    {
                        "name": "understanding",
                        "status": "completed",
                        "duration_ms": 12,
                        "summary": {"severity": "8"},
                    }
                ]
            }

    async def process_fn(incident, verbose=False):
        assert verbose is True
        return _FakeJob()

    iface = IncidentInputInterface(_config(), process_fn=process_fn)
    client = await _client(iface)
    try:
        resp = await client.post(
            "/api/v1/incidents/freetext?verbose=1",
            json={"description": "suspicious transfer"},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["job_id"] == "job-xyz"
        assert body["report"] == "REPORT-TEXT"
        assert body["stages"][0]["duration_ms"] == 12
        assert body["events"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_list_jobs_endpoint():
    class _JM:
        # queue as well as list_jobs: the response carries the backlog's counters beside
        # the rows, because a caller reading `queued` on a job needs the width and the
        # depth to know what it is waiting for.
        queue = SimpleNamespace(stats=lambda: {"width": 2, "queued": 0})

        def list_jobs(self):
            return [{"job_id": "j1", "status": "running"}]

    iface = IncidentInputInterface(_config(), job_manager=_JM())
    client = await _client(iface)
    try:
        resp = await client.get("/api/v1/jobs")
        assert resp.status == 200
        body = await resp.json()
        assert body["jobs"][0]["job_id"] == "j1"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_freetext_rejects_empty_description():
    iface = IncidentInputInterface(_config(), process_fn=AsyncMock())
    client = await _client(iface)
    try:
        resp = await client.post("/api/v1/incidents/freetext", json={"description": ""})
        assert resp.status == 400
    finally:
        await client.close()


# --- the blocking endpoints behind a platform ingress ------------------------
#
# Every test above builds `_config()` with no `blocking_endpoints` key and expects the
# inline endpoints to work, which is the invariant that matters most here: the platform
# default is ON everywhere except a Databricks App, so the VM deployment is untouched by
# this switch existing. What follows is the other half.


@pytest.mark.asyncio
async def test_blocking_endpoints_are_refused_with_a_pointer_inside_an_app(monkeypatch):
    """A 501 naming the replacement, not a socket held until the ingress cuts it.

    The refusal is the *useful* answer. Behind an ingress that closes a request at ~120s
    these endpoints cannot return a report — a measured run is 38 min at the median — and
    what the caller gets instead of a timeout is nothing they can act on: no job id, no
    handle, while the investigation continues invisibly. So the body has to carry the
    replacement route, and `process_fn` must never be awaited: refusing after starting a
    40-minute run would be the worst of both.
    """
    monkeypatch.setenv("DATABRICKS_APP_PORT", "8000")
    process_fn = AsyncMock(return_value={"sections": []})
    iface = IncidentInputInterface(_config(), process_fn=process_fn)
    client = await _client(iface)
    try:
        for path, payload in (
            ("/api/v1/incidents/freetext", {"description": "x"}),
            ("/api/v1/incidents", {"description": "x"}),
            ("/api/v1/ir", {"id": "IR1"}),
        ):
            resp = await client.post(path, json=payload)
            assert resp.status == 501, path
            body = await resp.json()
            assert body["use_instead"] == "POST /api/v1/jobs"
            assert "/api/v1/jobs" in body["error"]
        process_fn.assert_not_awaited()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_the_jobs_api_still_works_inside_an_app(monkeypatch):
    """Otherwise this is not a redirection, it is an outage.

    The 501 above tells the caller to use `POST /api/v1/jobs`; a switch that reached the
    job routes too would make that instruction false, and the test asserting the refusal
    would pass either way.
    """
    monkeypatch.setenv("DATABRICKS_APP_PORT", "8000")

    class _JM:
        # queue as well as list_jobs: the response carries the backlog's counters beside
        # the rows, because a caller reading `queued` on a job needs the width and the
        # depth to know what it is waiting for.
        queue = SimpleNamespace(stats=lambda: {"width": 2, "queued": 0})

        def list_jobs(self):
            return [{"job_id": "j1", "status": "running"}]

    iface = IncidentInputInterface(_config(), job_manager=_JM())
    client = await _client(iface)
    try:
        resp = await client.get("/api/v1/jobs")
        assert resp.status == 200
        assert (await resp.json())["jobs"][0]["job_id"] == "j1"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_an_operator_can_override_the_platform_default_both_ways(monkeypatch):
    """The ingress limit belongs to the platform, so the operator keeps the last word.

    An App reached through a tunnel may want them on; a VM behind somebody else's proxy
    may want them off. Read per request out of the config dict — which is what makes the
    field's `live` claim true rather than a promise the handler cannot keep.
    """
    process_fn = AsyncMock(return_value={"sections": []})
    config = _config()
    iface = IncidentInputInterface(config, process_fn=process_fn)
    client = await _client(iface)
    try:
        monkeypatch.setenv("DATABRICKS_APP_PORT", "8000")
        config["blocking_endpoints"] = "on"
        resp = await client.post(
            "/api/v1/incidents/freetext", json={"description": "x"}
        )
        assert resp.status == 200

        monkeypatch.delenv("DATABRICKS_APP_PORT", raising=False)
        config["blocking_endpoints"] = "off"
        resp = await client.post(
            "/api/v1/incidents/freetext", json={"description": "x"}
        )
        assert resp.status == 501
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_an_unrecognised_override_is_loud_and_still_resolves(monkeypatch, caplog):
    """A typo must not decide this silently in either direction.

    Reading `enabld` as "off" would take the endpoints away from a VM deployment whose
    operator believes they turned them on; reading it as a working value with no complaint
    is how a deliberate override becomes a mystery. So: resolve to the platform default,
    and say so.
    """
    monkeypatch.delenv("DATABRICKS_APP_PORT", raising=False)
    config = _config()
    config["blocking_endpoints"] = "enabld"
    iface = IncidentInputInterface(config, process_fn=AsyncMock(return_value={}))
    client = await _client(iface)
    try:
        with caplog.at_level("ERROR"):
            resp = await client.post(
                "/api/v1/incidents/freetext", json={"description": "x"}
            )
        assert resp.status == 200  # the platform default, off-platform
        assert "blocking_endpoints" in caplog.text
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_an_ir_lookup_is_refused_before_the_ticketing_system_is_called(
    monkeypatch,
):
    """`get_ir` fetches a record from a third party *before* it has an incident to run.

    Checked in the handler as well as the shared funnel, because a request we already know
    we will refuse should not first make somebody else's system do work for it. Asserted
    by leaving `win_url` pointed at nowhere: if the refusal came later, the outbound call
    would be attempted and the status would not be 501.
    """
    monkeypatch.setenv("DATABRICKS_APP_PORT", "8000")
    config = _config()
    config.update(
        {
            "win_url": "http://127.0.0.1:1",  # nothing listens here
            "win_username": "u",
            "win_password": "p",
        }
    )
    iface = IncidentInputInterface(config, process_fn=AsyncMock())
    client = await _client(iface)
    try:
        resp = await client.post("/api/v1/ir", json={"id": "IR1"})
        assert resp.status == 501
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_deep_health_reports_where_state_lives_and_whether_it_works():
    """WHERE durable state lives is switchable from the Configuration tab in every
    deployment mode, so it needs a read-back that is not a boot log.

    The failure this exists for: a remote store that refuses every write — a blank
    catalog, a workspace-scoped PAT answering 403 — behaves exactly like a working one
    until the restart that finds nothing there. `main()` logs `Storage is DEGRADED` once,
    at boot, which is precisely the moment an operator who switches the backend at runtime
    is not reading.

    Three states, deliberately distinct: `None` for not wired (every test that builds this
    interface directly, and a deployment that passes no store), `True` for usable, and
    `False` **with a reason** — a boolean alone would tell an operator that something is
    wrong and nothing about which of the four possible causes it is.
    """
    iface = IncidentInputInterface(_config())
    client = await _client(iface)
    try:

        class _Healthy:
            kind = "databricks"
            degradation = None

        iface.storage = _Healthy()
        deep = await (await client.get("/health?deep=1")).json()
        assert deep["storage"] == "databricks"
        assert deep["storage_ok"] is True
        assert deep["storage_detail"] is None

        class _Broken:
            kind = "databricks"
            degradation = "catalog is not set, so there is no Volume path to write to"

        iface.storage = _Broken()
        deep = await (await client.get("/health?deep=1")).json()
        assert deep["storage_ok"] is False
        assert "catalog" in deep["storage_detail"]

        # A store whose own reporter raises is still a store that answers. The health
        # endpoint exists to report failures, so it must not become one.
        class _Hostile:
            kind = "databricks"

            @property
            def degradation(self):
                raise RuntimeError("boom")

        iface.storage = _Hostile()
        deep = await (await client.get("/health?deep=1")).json()
        assert deep["storage_ok"] is False
        assert deep["storage_detail"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_deep_health_reports_which_sources_cannot_be_queried():
    """The graceful degrade that is silent, one stage before the storage one.

    A pack source whose backend credentials are missing builds no retriever and is skipped
    with a log line — correct, because one unreachable source must not fail a run. But the
    consequence arrives at the far end wearing the wrong name: the retrieval stage reports
    success, the conditions that needed that source go `unknown`, and the verdict is
    INSUFFICIENT DATA, indistinguishable from "the sources had nothing to say". Measured on
    a real job: 18 of 30 sources skipped, every ELK one, for unset credentials.

    So it is a COUNT plus the reasons, not a boolean: zero unavailable and "18 of 30" are
    different answers, and only the second names a credential to go and set.
    """
    iface = IncidentInputInterface(_config())
    client = await _client(iface)
    try:

        class _Engine:
            retrievers = {"a": object(), "b": object()}
            unavailable_sources = {}

        iface.retrieval_engine = _Engine()
        deep = await (await client.get("/health?deep=1")).json()
        assert deep["sources_declared"] == 2
        assert deep["sources_unavailable"] == {}, "fully wired is {}, not None"

        class _Degraded:
            retrievers = {"a": object()}
            unavailable_sources = {
                "auth_events": "no credentials for elasticsearch cluster 'primary'",
                "tickets": "no retriever implements type 'snowflake'",
            }

        iface.retrieval_engine = _Degraded()
        deep = await (await client.get("/health?deep=1")).json()
        # The total is what makes the count legible: 2 of 3, not "2 missing".
        assert deep["sources_declared"] == 3
        assert set(deep["sources_unavailable"]) == {"auth_events", "tickets"}
        assert "elasticsearch" in deep["sources_unavailable"]["auth_events"]
        # `status` stays ok on purpose. The server IS up; a degraded source set is not a
        # liveness failure, and conflating the two is how the probe starts flapping.
        assert deep["status"] == "ok"

        # An engine whose own attributes raise must not take the health endpoint with it.
        class _Hostile:
            @property
            def unavailable_sources(self):
                raise RuntimeError("boom")

        iface.retrieval_engine = _Hostile()
        deep = await (await client.get("/health?deep=1")).json()
        assert deep["status"] == "ok"
        assert deep["sources_unavailable"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_deep_health_reports_whether_the_pack_LOADED_not_whether_it_is_named():
    """The widest silent degrade of the four, and the one a configured name cannot see.

    ``load_knowledge_pack`` answers a directory it cannot read with an *empty* pack and one
    ``logger.warning``, so a ``pack_dir`` that did not ship — or is still the template's
    placeholder — leaves every stage running and answering nothing: no glossary to extract
    entities with, no catalog to pick a source from, no ruleset to adjudicate. Reading the
    configured name would report that as healthy, because the name is set either way.

    The counts are the answer rather than a bare boolean, because a pack that loaded still
    has to be checked for the ruleset a verdict needs, and only a number says so.
    """
    iface = IncidentInputInterface(_config())
    client = await _client(iface)
    try:
        # What `load_knowledge_pack` returns for a directory that is not there.
        iface.knowledge_pack = KnowledgePack(name="a_domain")
        deep = await (await client.get("/health?deep=1")).json()
        assert deep["pack"] is False, "an empty pack is a degradation, not a null"
        assert "0 sources" in deep["pack_detail"]
        assert "every condition reads" in deep["pack_detail"]
        # Still up: a run will complete and explain nothing, which is what makes it easy
        # to miss and the reason it is on the dot at all.
        assert deep["status"] == "ok"

        iface.knowledge_pack = KnowledgePack(
            name="a_domain",
            entities=[EntityDef(type="user")],
            sources=[SourceDef(name="auth_events")],
        )
        deep = await (await client.get("/health?deep=1")).json()
        assert deep["pack"] is True
        # Loaded and still worth reading: no ruleset means no deterministic verdict.
        assert "0 rulesets" in deep["pack_detail"]
        assert "1 entities, 1 sources" in deep["pack_detail"]

        # And the count is of the RULESETS and not of the rules file's top-level keys. Sizing
        # `pack.rulesets` reported "2" for a pack shipping three procedures and would report
        # the same 2 for a pack shipping one — a number that moves with nothing is worse than
        # no number, because a reader checking whether their procedure shipped is reassured.
        iface.knowledge_pack = KnowledgePack(
            name="a_domain",
            entities=[EntityDef(type="user")],
            sources=[SourceDef(name="auth_events")],
            rulesets={
                "verdicts": {"first": {}, "second": {}, "third": {}},
                "default_ruleset": "first",
            },
        )
        deep = await (await client.get("/health?deep=1")).json()
        assert "3 rulesets" in deep["pack_detail"]

        # A pack object whose contents are not sized must not take the endpoint with it.
        iface.knowledge_pack = SimpleNamespace(name="hostile", entities=object())
        deep = await (await client.get("/health?deep=1")).json()
        assert deep["status"] == "ok"
        assert deep["pack"] is False
    finally:
        await client.close()


# --- the run queue, batches, and three config-reading rules ------------------


def test_pass_number_none_means_the_pass_the_run_is_on():
    assert _pass_number({}) is None
    assert _pass_number({"pass": None}) is None
    assert _pass_number(None) is None
    assert _pass_number("not a body") is None


def test_pass_number_refuses_a_malformed_value_instead_of_400ing():
    assert _pass_number({"pass": "later"}) is None
    assert _pass_number({"pass": []}) is None
    assert _pass_number({"pass": True}) is None  # a bool is not a pass index
    assert _pass_number({"pass": 0}) is None
    assert _pass_number({"pass": -3}) is None


def test_pass_number_accepts_a_positive_int_or_its_digits():
    assert _pass_number({"pass": 2}) == 2
    assert _pass_number({"pass": "3"}) == 3
    assert _pass_number({"pass": 1.0}) == 1


def test_a_full_run_queue_is_refused_with_the_depth_and_the_limit():
    class _QueueFull(RuntimeError):
        depth = 50
        limit = 50

    refusal = IncidentInputInterface._queue_refusal(_QueueFull("queue is full"))
    assert refusal.status == 429
    body = json.loads(refusal.text)
    assert body["queued"] == 50 and body["max_queued"] == 50
    assert body["retry"]


def test_queue_refusal_is_duck_typed_across_the_two_import_styles():
    # `except QueueFull` misses the class reached under the other import style, so a
    # backlog refusal is recognised by its fields and not its type.
    assert IncidentInputInterface._queue_refusal(RuntimeError("boom")) is None
    assert (
        IncidentInputInterface._queue_refusal(
            SimpleNamespace(depth="50", limit=50)
        )
        is None
    )
    borrowed = RuntimeError("queue is full")
    borrowed.depth, borrowed.limit = 7, 4
    assert IncidentInputInterface._queue_refusal(borrowed).status == 429


def _batch_iface(submitted):
    class _JM:
        queue = SimpleNamespace(stats=lambda: {"width": 2, "queued": 0})

        def submit_batch(self, incidents, run_mode="auto", batch_id=None):
            submitted.extend(incidents)
            return {
                "batch_id": batch_id or "B1",
                "mode": run_mode,
                "submitted": [i["id"] for i in incidents],
            }

    return IncidentInputInterface(
        _config(), job_manager=_JM(), launch_fn=lambda incident, mode: None
    )


@pytest.mark.asyncio
async def test_an_oversized_batch_is_refused_whole_and_nothing_is_submitted():
    submitted = []
    client = await _client(_batch_iface(submitted))
    try:
        resp = await client.post(
            "/api/v1/batches",
            json={"incidents": ["x"] * (IncidentInputInterface._MAX_BATCH_SIZE + 1)},
        )
        assert resp.status == 413
        body = await resp.json()
        assert str(IncidentInputInterface._MAX_BATCH_SIZE) in body["error"]
        assert submitted == [], "a truncated batch reads as a batch that ran"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_batch_reports_what_it_rejected_beside_what_it_accepted():
    submitted = []
    client = await _client(_batch_iface(submitted))
    try:
        resp = await client.post(
            "/api/v1/batches",
            json={
                "incidents": [
                    "first alert",
                    {"description": "second alert", "extended_retrieval": "yes"},
                    {"description": "  "},
                    42,
                ],
                "mode": "semi_auto",
            },
        )
        assert resp.status == 201
        body = await resp.json()
        assert len(body["submitted"]) == 2
        assert [r["index"] for r in body["rejected"]] == [2, 3]
        assert body["mode"] == "semi_auto"
        assert submitted[1]["extended_retrieval"] is True
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_batch_with_no_usable_incident_is_a_400_naming_each_reason():
    submitted = []
    client = await _client(_batch_iface(submitted))
    try:
        resp = await client.post("/api/v1/batches", json={"incidents": [{}, ""]})
        assert resp.status == 400
        assert len((await resp.json())["rejected"]) == 2
        assert submitted == []
    finally:
        await client.close()


def test_link_config_is_read_from_the_whole_config_not_from_this_slice():
    # self.config is main_config["incident_input"], which carries no `correlation` key,
    # so reading the block from it reports a budget of zero.
    links = {"max_probes_per_run": 4}
    iface = IncidentInputInterface(
        _config(), live_config={"correlation": {"links": links}}
    )
    assert iface._link_config() == links
    assert IncidentInputInterface(_config())._link_config() == {}


def test_link_config_falls_back_to_a_slice_that_is_a_whole_config():
    iface = IncidentInputInterface({**_config(), "correlation": {"links": {"a": 1}}})
    assert iface._link_config() == {"a": 1}


def test_set_live_writes_only_a_direct_child_of_each_segment():
    live = {"anomaly_detection": {"threshold": 0.5}, "jobs": {"nested": {"width": 1}}}
    iface = IncidentInputInterface(_config(), live_config=live)
    assert iface._set_live("anomaly_detection.threshold", 0.8)
    assert live["anomaly_detection"]["threshold"] == 0.8
    # An absent leaf is refused rather than created: a key nobody reads would report
    # success and change nothing.
    assert not iface._set_live("anomaly_detection.absent", 1)
    assert "absent" not in live["anomaly_detection"]
    # And a same-named key one level deeper is not the target.
    assert not iface._set_live("jobs.width", 4)
    assert live["jobs"]["nested"]["width"] == 1
    assert not iface._set_live("no_such_section.key", 1)


def test_audit_changes_redacts_a_secret_shaped_key_both_ways():
    """The patcher's change records carry `from` and `to` verbatim, because the operator reads
    them back in the response. The journal keeps the same records for months and every
    administrator may read it, so a credential must not survive the trip.

    Prospective by design: no field descriptor is secret-shaped today (the form offers
    `api_key_env`, never `api_key`), so this guard is what keeps adding one from turning an
    append-only file into a credential store.
    """
    out = _audit_changes([
        {"path": "llm.api_key", "from": "sk-old", "to": "sk-new", "applies": "restart"},
        {"path": "storage.sql.dsn", "from": "postgres://u:p@h/db", "to": "x"},
        {"path": "llm.api_key_env", "from": "OLD", "to": "NEW"},
    ])
    assert out[0] == {"path": "llm.api_key", "from": REDACTED, "to": REDACTED,
                      "applies": "restart"}
    assert out[1]["from"] == REDACTED and out[1]["to"] == REDACTED
    # An env-var NAME is not a secret; redacting it would hide the one thing worth recording.
    assert out[2] == {"path": "llm.api_key_env", "from": "OLD", "to": "NEW"}


def test_audit_changes_bounds_one_value_and_says_how_long_it_was():
    """A pasted certificate would push a day's other entries out of a bounded buffer."""
    (out,) = _audit_changes([{"path": "identity.admin_users", "from": "", "to": "x" * 5000}])
    assert len(out["to"]) < 300 and "5000 chars" in out["to"]
    # An absent side is omitted rather than recorded as an empty string: an inserted key had
    # no previous value, which is not the same as having had a blank one.
    assert "from" not in _audit_changes([{"path": "a.b", "from": None, "to": "1"}])[0]


def test_audit_changes_keeps_a_number_a_number():
    """A threshold read back as "0.8" and one read back as 0.8 are the same edit, and a
    reader comparing two entries must not have to guess which."""
    (out,) = _audit_changes([{"path": "anomaly_detection.threshold", "from": 0.8, "to": 0.5}])
    assert out["from"] == 0.8 and out["to"] == 0.5
