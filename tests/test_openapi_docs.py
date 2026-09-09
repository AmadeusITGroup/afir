"""Tests for the OpenAPI description and the two routes that serve it.

The spec is hand-authored rather than generated, so the thing worth asserting is
that it stays in agreement with the router: a route the server registers and the
spec omits is a route nobody outside the codebase knows about, and a route the
spec claims and the server does not register is worse, because a client generator
will emit a call for it.
"""

import re

import pytest
import yaml
from aiohttp.test_utils import TestClient, TestServer

from src import openapi_docs
from src.incident_input import IncidentInputInterface
from src.utils.rate_limiter import AsyncRateLimiter


def _config():
    return {
        "post_incident_endpoint": "/api/v1/incidents",
        "post_ir_endpoint": "/api/v1/ir",
        "post_freetext_endpoint": "/api/v1/incidents/freetext",
        "port": 5000,
        "rate_limit": {"requests": 100, "per_seconds": 60},
    }


async def _client(iface):
    iface.rate_limiter = AsyncRateLimiter(100, 60)
    client = TestClient(TestServer(iface.app))
    await client.start_server()
    return client


def _spec():
    spec = openapi_docs.load_spec()
    assert spec is not None, openapi_docs.unavailable_reason()
    return spec


def test_the_spec_parses_and_declares_the_expected_shape():
    spec = _spec()
    assert spec["openapi"].startswith("3.1")
    assert spec["info"]["title"]
    assert spec["info"]["version"]
    assert spec["info"]["license"]["name"] == "Apache-2.0"
    assert spec["paths"]


def test_every_operation_has_an_id_a_summary_and_a_tag():
    """An untagged or unsummarised operation renders as an unexplained row."""
    missing = []
    for tag, method, path, op in openapi_docs._operations(_spec()):
        if not op.get("operationId"):
            missing.append(f"{method} {path}: operationId")
        if not op.get("summary"):
            missing.append(f"{method} {path}: summary")
        if not op.get("responses"):
            missing.append(f"{method} {path}: responses")
        if tag == "Other":
            missing.append(f"{method} {path}: tag")
    assert not missing, missing


def test_operation_ids_are_unique():
    ids = [op.get("operationId") for _t, _m, _p, op in openapi_docs._operations(_spec())]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, dupes


def test_every_ref_resolves():
    """A dangling ``$ref`` renders as a bare schema name that leads nowhere."""
    spec = _spec()
    declared = set((spec.get("components") or {}).get("schemas") or {})
    declared |= set((spec.get("components") or {}).get("responses") or {})
    declared |= set((spec.get("components") or {}).get("parameters") or {})
    refs = set(re.findall(r"#/components/\w+/(\w+)", yaml.safe_dump(spec)))
    assert refs <= declared, refs - declared


def test_the_spec_documents_every_route_the_server_registers():
    """The agreement that makes a hand-authored spec trustworthy.

    ``canonical`` gives aiohttp's own template form (``/api/v1/jobs/{job_id}``),
    which is the same spelling OpenAPI uses, so the two sets are comparable
    without normalising either side.
    """
    iface = IncidentInputInterface(_config())
    registered = set()
    for route in iface.app.router.routes():
        if route.method == "HEAD":
            continue
        registered.add((route.method.lower(), route.resource.canonical))

    documented = set()
    for path, item in _spec()["paths"].items():
        for method in item:
            if method in openapi_docs._METHODS:
                documented.add((method, path))

    assert not (registered - documented), sorted(registered - documented)
    assert not (documented - registered), sorted(documented - registered)


@pytest.mark.asyncio
async def test_openapi_json_serves_the_spec():
    iface = IncidentInputInterface(_config())
    client = await _client(iface)
    try:
        resp = await client.get("/openapi.json")
        assert resp.status == 200
        assert resp.content_type == "application/json"
        body = await resp.json()
        assert body["openapi"].startswith("3.1")
        assert len(body["paths"]) == len(_spec()["paths"])
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_docs_renders_html_with_no_external_asset():
    """The App has no egress, so a CDN-hosted viewer renders a blank page. The
    only permitted ``href`` targets are in-page anchors and ``/openapi.json``."""
    iface = IncidentInputInterface(_config())
    client = await _client(iface)
    try:
        resp = await client.get("/docs")
        assert resp.status == 200
        assert resp.content_type == "text/html"
        html = await resp.text()
        assert "<script" not in html and "<link" not in html and "<img" not in html
        external = re.findall(r'(?:src|href)\s*=\s*"(?!#|/openapi\.json)([^"]+)"', html)
        assert not external, external
        for tag in ("Jobs", "Gates", "Knowledge"):
            assert tag in html
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_an_absent_spec_answers_503_and_says_why(monkeypatch, tmp_path):
    """Not a traceback and not an empty page: ``docs/`` ships in the deployment
    bundle by virtue of the exclude list, not by guarantee, and the reader has to
    be able to tell a missing file from a broken server."""
    monkeypatch.setattr(openapi_docs, "docs_dir", lambda: tmp_path)
    monkeypatch.setattr(openapi_docs, "_CACHE", None)
    iface = IncidentInputInterface(_config())
    client = await _client(iface)
    try:
        resp = await client.get("/openapi.json")
        assert resp.status == 503
        body = await resp.json()
        assert body["error"] == "specification unavailable"
        assert "openapi.yaml" in body["reason"]

        resp = await client.get("/docs")
        assert resp.status == 503
        assert "openapi.yaml" in await resp.text()
    finally:
        await client.close()


def test_the_renderer_escapes_markup_from_the_spec():
    """The spec is a checked-in file, but it is still rendered into a page; an
    unescaped angle bracket in a description would break the document."""
    spec = {
        "openapi": "3.1.0",
        "info": {"title": "T", "version": "1", "description": "<script>x</script>"},
        "paths": {
            "/a": {
                "get": {
                    "tags": ["X"],
                    "summary": "<b>s</b>",
                    "operationId": "a",
                    "responses": {"200": {"description": "ok"}},
                }
            }
        },
    }
    html = openapi_docs.render_html(spec)
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html
    assert "<b>s</b>" not in html


def _deref(spec: dict, schema: dict) -> dict:
    """Follow a local ``$ref`` one hop, which is the only depth this spec uses."""
    ref = schema.get("$ref")
    if not ref:
        return schema
    node = spec
    for part in ref.lstrip("#/").split("/"):
        node = node[part]
    return node


def _dummy(schema: dict, spec: dict):
    """The smallest value satisfying one property's declared type.

    Type-driven rather than name-driven on purpose: a value chosen from the property NAME
    would encode the same guess the spec already encodes, so the two could agree while both
    disagreed with the handler.
    """
    schema = _deref(spec, schema)
    if schema.get("enum"):
        return schema["enum"][0]
    kind = schema.get("type") or "string"
    if kind == "array":
        return [_dummy(schema.get("items") or {}, spec)]
    if kind == "object":
        return {}
    return {"integer": 1, "number": 1, "boolean": True}.get(kind, "PROBE")


#: Path parameters substituted with a value each handler can at least parse. A 404 from one of
#: these is a fine answer — what this test is about is a body refused as malformed.
_PATH_VALUES = {
    "job_id": "00000000-0000-0000-0000-000000000000",
    "batch_id": "00000000-0000-0000-0000-000000000000",
    "pack": "mock_domain",
    "stage": "understanding",
    "index": "0",
    "session": "s1",
    "name": "PROBE_TOKEN",
}

#: ``POST /api/v1/ir`` is the one operation whose documented body cannot be proven accepted by
#: a status code. It calls the incident tracker BEFORE it has an incident to run, so a
#: correctly-shaped body still ends at 400 once that call fails — which is why its agreement is
#: asserted by the refusal CHANGING, below, rather than exempted silently.
_CALLS_A_THIRD_PARTY = {"POST /api/v1/ir"}


def _documented_bodies(spec):
    """Every operation whose body schema names at least one REQUIRED property.

    A schema declaring none makes no claim a client can violate — the free-form ``{...}``
    bodies (a stage override, a config import) are refused for having no content, which is the
    handler's own rule and not a disagreement with the spec.
    """
    for _tag, method, path, op in openapi_docs._operations(spec):
        body = op.get("requestBody") or {}
        schema = ((body.get("content") or {}).get("application/json") or {}).get("schema")
        if not schema:
            continue
        schema = _deref(spec, schema)
        props = schema.get("properties") or {}
        # `required` plus the FIRST `anyOf` branch, which is how 3.1 spells "and at least one
        # of these". A handler enforcing that rule refuses a body built from `required` alone,
        # so a spec stating it only in prose is as unusable as one stating it nowhere.
        names = list(schema.get("required") or [])
        for branch in schema.get("anyOf") or []:
            names += list(_deref(spec, branch).get("required") or [])
            break
        # A required name that is not a declared property cannot be given a value, so it would
        # silently shrink the payload — and a shrunk payload is how this probe passes on the
        # very defect it exists to find. Reported rather than skipped.
        undeclared = [n for n in names if n not in props]
        assert not undeclared, f"{method} {path}: {undeclared} not in {sorted(props)}"
        payload = {name: _dummy(props[name], spec) for name in names if name in props}
        if not payload:
            continue
        target = path
        for param, value in _PATH_VALUES.items():
            target = target.replace("{" + param + "}", value)
        yield method, path, target, payload


async def _probe_client(monkeypatch, tmp_path):
    """A server with every optional collaborator wired, because 503 masks a refusal.

    ``create_job`` answers 503 *before* validating its body when ``launch_fn`` is absent, so an
    unwired probe passes on exactly the defect this test exists to catch. The pack root is
    redirected for the reason every pack-editor test redirects it: an unredirected
    scaffold writes a real pack into ``knowledge/``.
    """
    from unittest.mock import AsyncMock, MagicMock

    from src.knowledge import pack_store

    monkeypatch.setattr(pack_store, "knowledge_pack_dir", lambda name: tmp_path / name)
    assert pack_store.packs_root() == tmp_path

    manager = MagicMock()
    manager.get_job.return_value = None
    manager.list_jobs.return_value = []
    iface = IncidentInputInterface(
        _config(),
        process_fn=AsyncMock(return_value={"report": {}}),
        feedback_fn=AsyncMock(),
        job_manager=manager,
        launch_fn=MagicMock(side_effect=RuntimeError("no pipeline in this test")),
    )
    return await _client(iface)


@pytest.mark.asyncio
async def test_every_documented_request_body_is_accepted_by_its_handler(monkeypatch, tmp_path):
    """The agreement no other test in this file makes: spec-to-ROUTER both ways is already
    asserted, and a body is the other half of a call.

    A generated client can only send what the spec's ``required`` list names, so a handler
    reading some other key answers 400 on a well-formed request — a defect invisible from
    either side alone, and one that shipped six times over (``CreateJobRequest`` taking a
    nested incident, ``FreetextIncidentInput`` naming ``text`` for ``description``, ``/ir``
    documented as an incident rather than a record id, ``FeedbackSubmission`` requiring
    ``job_id`` instead of ``incident_id``, and two undocumented required keys on the knowledge
    routes). Every one of them was correct in ``docs/API.md``, so prose is not the check.
    """
    spec = _spec()
    client = await _probe_client(monkeypatch, tmp_path)
    refused = []
    checked = 0
    try:
        for method, path, target, payload in _documented_bodies(spec):
            if f"{method} {path}" in _CALLS_A_THIRD_PARTY:
                continue
            checked += 1
            resp = await client.request(method.upper(), target, json=payload)
            if resp.status == 400:
                refused.append(f"{method} {path} {sorted(payload)} -> {await resp.text()}")
    finally:
        await client.close()
    assert not refused, refused
    # A spec whose bodies stopped being read would pass the loop above vacuously. The floor
    # sits under the measured 13 rather than on it, so adding a route is not a test failure.
    assert checked >= 10, checked


@pytest.mark.asyncio
async def test_the_ir_body_is_read_even_though_its_handler_cannot_answer(monkeypatch, tmp_path):
    """The one exemption above, proven rather than assumed.

    ``get_ir`` validates the body, then asks the tracker — so both the documented body and an
    empty one end at 400 and a status code cannot tell them apart. What can is the REASON: the
    documented body clears validation and fails at the outbound call, the empty one does not
    reach it. If the spec ever names some other key, both bodies fail validation and the two
    messages converge, which is what this asserts they do not.
    """
    spec = _spec()
    bodies = _documented_bodies(spec)
    body = next(p for m, path, _t, p in bodies if f"{m} {path}" in _CALLS_A_THIRD_PARTY)
    assert body, "the /ir body must document at least one required property"
    client = await _probe_client(monkeypatch, tmp_path)
    try:
        documented = await (await client.post("/api/v1/ir", json=body)).text()
        empty = await (await client.post("/api/v1/ir", json={})).text()
    finally:
        await client.close()
    assert documented != empty, documented


def test_the_deprecated_blocking_endpoints_are_marked_as_such():
    """They answer 501 naming the job endpoint; a generated client should not
    reach for them first."""
    spec = _spec()
    for path in (
        "/api/v1/incidents",
        "/api/v1/incidents/freetext",
        "/api/v1/ir",
    ):
        op = spec["paths"][path]["post"]
        assert op.get("deprecated") is True, path
        assert "501" in op["responses"], path


def test_no_emoji_and_no_literal_secret_in_the_spec():
    text = openapi_docs.spec_path().read_text(encoding="utf-8")
    assert not re.search("[\U0001f300-\U0001faff☀-➿]", text)
    # A template offers the env-var NAME, never the value.
    assert "api_key_env" in text or "api_key" not in text
    assert not re.search(r"(?i)\b(dapi[0-9a-f]{16,}|sk-[A-Za-z0-9]{20,})", text)
