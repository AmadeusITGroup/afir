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
