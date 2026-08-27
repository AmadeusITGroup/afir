"""`/api/v1/knowledge/...` — the HTTP layer of the pack editor, over a real server.

`test_pack_store.py` proves the store refuses what it must; it cannot prove the HTTP layer asks it
to. A store guard is only a guard if the handler routes into it: the config importer enforced a
validation rule only at write time, had no HTTP-level test, and so returned a 500 for what was
really a bad request, breaking its own all-or-nothing promise for every file queued behind it.

The assertion that matters most is a status code, not a body. A traversal `?path=` must be 400 and
not 200 with somebody else's file, and the difference is invisible to a test that only checks the
store: the handler could `normpath` the input, get a real readable file, and answer 200 with
content it was never meant to serve. `test_ui_server.py` pins the same rule for report ids.

`test_the_mutating_routes_are_exactly_these_four` replaces the structural read-only guarantee the
read half shipped with: the set of mutating methods is enumerated, so a fifth cannot appear
without somebody restating the blast radius on purpose.

The write tests assert that the handler routes into the store's guards, not that the guards work
(`test_pack_store.py` owns that). So they check the OUTCOME on disk — the file byte-identical
after a refusal, the prior bytes in history, an import rejecting one file writing none — since "the
handler called the right function" is what a mock would let pass with the route wired to nothing.

Every test runs against COPIES of the shipped packs under `tmp_path`, with
`pack_store.knowledge_pack_dir` redirected. Nothing here writes into `knowledge/`.
"""

import asyncio
import base64
import json
import shutil

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.incident_input import IncidentInputInterface
from src.knowledge import pack_assistant, pack_attachments, pack_store
from src.utils.paths import REPO_ROOT
from src.utils.rate_limiter import AsyncRateLimiter
from tests.installed_packs import FIXTURE_PACK, installed_packs

REAL_PACKS = REPO_ROOT / "knowledge"
CATALOG = "source_catalog.yaml"

#: The pack every test names. The engine's own fixture pack, so this file names no domain.
PACK = FIXTURE_PACK

#: The pack every traversal test tries to reach INSTEAD. Built by the `packs` fixture as a
#: copy of `PACK`, so it is a real pack on every branch — see that fixture for why.
SIBLING = "sibling_pack"


def _config():
    return {
        "host": "127.0.0.1",
        "port": 0,
        "post_incident_endpoint": "/api/v1/incidents",
        "post_ir_endpoint": "/api/v1/ir",
        "rate_limit": {"requests": 1000, "per_seconds": 60},
    }


@pytest.fixture
def packs(tmp_path, monkeypatch):
    """Writable copies of the shipped packs, with the store pointed at them.

    `knowledge_pack_dir` is the single seam every path in `pack_store` resolves through,
    which is why the redirect is one line — and why getting it wrong would edit the real
    pack. The fixture asserts the redirect took, rather than assuming it.

    A second pack always exists here, `SIBLING`, copied from the fixture pack rather than
    taken from whatever else is installed. The traversal tests need a real, readable,
    perfectly parseable file OUTSIDE the pack the caller named — that is the whole reason
    sanitising a path is dangerous — and a sibling that appears only on the branch shipping
    a domain pack would make those the assertions that quietly stop existing.
    """
    root = tmp_path / "knowledge"
    root.mkdir()
    for name in installed_packs():
        shutil.copytree(REAL_PACKS / name, root / name)
    shutil.copytree(root / PACK, root / SIBLING)
    monkeypatch.setattr(pack_store, "knowledge_pack_dir", lambda name: root / name)
    assert pack_store.packs_root() == root
    assert (root / SIBLING / CATALOG).is_file(), "the traversal target must be real"
    return root


@pytest.fixture
async def client(packs):
    """The real interface with NO pipeline wired at all.

    Deliberately `process_fn=None, job_manager=None`: the pack editor must work on a
    deployment that has never run an incident, because authoring a pack is what happens
    BEFORE there is anything to run. It also mirrors how `test_webui.py` constructs the
    class, so a route that only appears with a job manager would be one this file and that
    one are both blind to.
    """
    iface = IncidentInputInterface(
        _config(), live_config={"knowledge": {"pack_dir": "knowledge/mock_domain"}}
    )
    # Built in start_server(), which TestServer bypasses.
    iface.rate_limiter = AsyncRateLimiter(rate_limit=1000, time_period=1)
    c = TestClient(TestServer(iface.app))
    await c.start_server()
    yield c
    iface.rate_limiter.close()
    await c.close()


async def json_of(resp):
    assert resp.content_type == "application/json", await resp.text()
    return await resp.json()


# ------------------------------------------------------------------ listing and shape


async def test_listing_names_every_pack_and_which_one_is_loaded(client):
    """`loaded` is the flag that makes "restart to activate" actionable.

    Editing a pack deliberately does not hot-reload it. Without knowing which pack the
    running process is actually using, an operator cannot tell whether the file they just
    changed affects the next run or nothing at all.
    """
    body = await json_of(await client.get("/api/v1/knowledge"))
    names = [p["name"] for p in body["packs"]]
    assert "mock_domain" in names
    loaded = [p["name"] for p in body["packs"] if p["loaded"]]
    assert loaded == ["mock_domain"]
    assert body["loaded"] == "mock_domain"
    for entry in body["packs"]:
        assert entry["files"] > 0
        assert entry["bytes"] > 0


async def test_the_loaded_pack_is_matched_on_name_not_on_path_spelling(packs):
    """The config holds a PATH, this endpoint speaks in NAMES.

    An absolute `pack_dir` is the normal shape in a deployment, and comparing it to a bare
    name would mark every pack unloaded — the flag would silently always be false, which
    reads as "none of these is live" rather than as a broken comparison.
    """
    iface = IncidentInputInterface(
        _config(),
        live_config={"knowledge": {"pack_dir": "/srv/afir/knowledge/mock_domain/"}},
    )
    iface.rate_limiter = AsyncRateLimiter(rate_limit=1000, time_period=1)
    c = TestClient(TestServer(iface.app))
    await c.start_server()
    try:
        body = await json_of(await c.get("/api/v1/knowledge"))
        assert body["loaded"] == "mock_domain"
        assert [p["name"] for p in body["packs"] if p["loaded"]] == ["mock_domain"]
    finally:
        iface.rate_limiter.close()
        await c.close()


async def test_a_pack_get_returns_the_tree_and_the_diagnostics_together(client):
    """ONE payload, mirroring `GET /api/v1/config`, and the reason is the failure mode.

    A pack whose catalog does not parse looks completely normal in a file listing — same
    files, same sizes. A UI that fetched the tree and the diagnostics separately would
    render a clean, browsable, entirely wrong picture for the interval between the two
    responses, and that picture is precisely the silent-empty state this editor exists to
    make visible.
    """
    body = await json_of(await client.get("/api/v1/knowledge/mock_domain"))
    assert body["pack"] == "mock_domain"
    assert body["counts"]["files"] > 0
    assert body["validate"]["ok"] is True
    assert body["validate"]["errors"] == 0
    paths = [n["path"] for n in body["nodes"]]
    assert CATALOG in paths
    assert "entity_glossary.yaml" in paths


async def test_the_tree_declares_what_is_editable_inline_and_what_is_not(client):
    """`editable` is a fact about the FILE, not a permission.

    Three generated schema files in a shipped pack are ~456 KB. Loading one into a textarea
    and PUTting it back is the fastest available way to lose it, so the tree says so and
    the UI offers a download and a line-range edit instead.
    """
    body = await json_of(await client.get("/api/v1/knowledge/mock_domain"))
    limit = body["limits"]["inline_edit_max_bytes"]
    assert limit > 0
    files = [n for n in body["nodes"] if not n["dir"]]
    assert files, "the fixture pack has no files"
    for node in files:
        expected = node["bytes"] <= limit and node.get("text")
        assert bool(node["editable"]) == bool(expected), node["path"]
    kinds = {n["kind"] for n in files}
    assert {"catalog", "glossary"} <= kinds


async def test_a_dir_node_precedes_its_own_contents(client):
    """Tree ORDER is the payload's contract; the UI renders it as given.

    Sorting on the joined path string would make a directory's position depend on where
    "/" falls against the next character, so `shared/checks/...` and a sibling
    `shared_notes.md` could interleave. The browser would then indent a file under a
    directory it does not live in.
    """
    nodes = (await json_of(await client.get("/api/v1/knowledge/mock_domain")))["nodes"]
    index = {n["path"]: i for i, n in enumerate(nodes)}
    for node in nodes:
        if node["path"].count("/"):
            parent = node["path"].rsplit("/", 1)[0]
            assert index[parent] < index[node["path"]], node["path"]


async def test_an_unknown_pack_is_404_and_a_malformed_name_is_400(client):
    """Different codes because they are different mistakes.

    "No pack by that name" is a 404 a UI can offer to fix by listing what exists. A name
    the editor could never address at all is a 400 — the request itself was wrong, and
    answering 404 would suggest that creating it would help.

    A bare `..` segment is deliberately NOT probed here: measured, `/api/v1/knowledge/..`
    and its `%2e%2e` spelling both normalise to `/api/v1/` before aiohttp routes them, so
    the handler is never reached and the 404 comes from there being no such route. That
    input is guarded one level down, over the function that actually sees it
    (`safe_pack_name`, `tests/test_pack_store.py`). The forms below DO arrive: an encoded
    slash keeps the traversal inside a single segment, and `.history` is a real name.
    """
    assert (await client.get("/api/v1/knowledge/no_such_pack")).status == 404
    assert (
        await client.get(f"/api/v1/knowledge/{PACK}%2f..%2f{SIBLING}")
    ).status == 400
    assert (await client.get("/api/v1/knowledge/.history")).status == 400
    assert (await client.get("/api/v1/knowledge/bad$name")).status == 400


# ----------------------------------------------------------------------- reading a file


async def test_reading_a_file_returns_its_text_and_a_concurrency_token(client):
    body = await json_of(
        await client.get(f"/api/v1/knowledge/mock_domain/file?path={CATALOG}")
    )
    assert body["path"] == CATALOG
    assert body["kind"] == "catalog"
    assert "sources" in body["text"]
    assert len(body["sha256"]) == 64
    assert body["lines"] > 1
    assert body["bytes"] == len(body["text"].encode("utf-8"))


async def test_the_read_returns_the_bytes_on_disk_not_a_reserialisation(client, packs):
    """Comments and anchors are load-bearing pack content, so a read must be verbatim.

    One shipped ruleset is roughly 60% comment and the catalog carries live YAML anchors. A
    read that went through `safe_load` would silently drop the first and expand the second,
    and the operator would then save that reserialised text back over the real file — the
    editor destroying the pack by round-tripping it.
    """
    on_disk = (packs / "mock_domain" / CATALOG).read_text()
    body = await json_of(
        await client.get(f"/api/v1/knowledge/mock_domain/file?path={CATALOG}")
    )
    assert body["text"] == on_disk


@pytest.mark.parametrize(
    "hostile",
    [
        "../../etc/passwd",
        f"../{SIBLING}/source_catalog.yaml",
        "/etc/passwd",
        "shared/../../escape.yaml",
        ".history/index.json",
        "sub/../../../outside.yaml",
    ],
)
async def test_a_traversal_path_is_rejected_not_sanitised(client, hostile):
    """THE ASSERTION THIS FILE IS FOR: 400, and never 200 with another file's contents.

    Sanitising is the tempting failure. `normpath` on `../<sibling>/source_catalog.yaml`
    yields a real, readable, perfectly parseable file — so a sanitising handler answers 200
    with content from a pack the caller never named, and neither the caller nor the log can
    tell that happened. Rejecting is the only outcome that cannot silently serve the wrong
    file. Same rule `test_ui_server.py` pins for report ids.
    """
    resp = await client.get(f"/api/v1/knowledge/mock_domain/file?path={hostile}")
    assert resp.status == 400, await resp.text()
    body = await resp.json()
    assert "error" in body
    assert "sources" not in str(body), "a rejection must not leak the file's contents"


async def test_a_missing_file_is_404_and_a_missing_path_is_400(client):
    """No `?path=` at all is a malformed request; a path naming nothing is a missing file."""
    assert (
        await client.get("/api/v1/knowledge/mock_domain/file?path=nope.yaml")
    ).status == 404
    assert (await client.get("/api/v1/knowledge/mock_domain/file")).status == 400


async def test_a_binary_suffix_is_refused_even_though_the_file_exists(client, packs):
    """The suffix allowlist is half the bound on this API, so it is enforced on READ too.

    A pack directory can contain anything an operator dropped in it. Serving arbitrary
    bytes from a path-addressed endpoint with no authentication is a different and much
    larger promise than serving the pack's text files, and this endpoint makes the smaller
    one.
    """
    (packs / "mock_domain" / "diagram.png").write_bytes(b"\x89PNG\r\n\x1a\n binary")
    resp = await client.get("/api/v1/knowledge/mock_domain/file?path=diagram.png")
    assert resp.status == 400
    assert "editable" in (await resp.json())["error"]


async def test_an_oversized_file_is_413_with_advice_rather_than_400(client, packs):
    """413 because the request was fine and the operator's next move is different.

    A 400 says "you asked wrong". A 413 says "this one is too big to read inline" and the
    message names the alternative — which is the actual situation for the ~456 KB schema
    files in a shipped pack.
    """
    big = "# padding\n" * (pack_store.READ_MAX_BYTES // 10 + 100)
    (packs / "mock_domain" / "huge.yaml").write_text(big)
    resp = await client.get("/api/v1/knowledge/mock_domain/file?path=huge.yaml")
    assert resp.status == 413
    assert "download" in (await resp.json())["error"]


async def test_download_serves_the_file_as_an_attachment(client):
    """The answer for a file too large to edit inline — and the only one, for now.

    A file over the inline limit still has to be *gettable*, or the editor is a tool that
    can show you a pack it refuses to let you look at.
    """
    resp = await client.get(
        f"/api/v1/knowledge/mock_domain/file?path={CATALOG}&download=1"
    )
    assert resp.status == 200
    disposition = resp.headers["Content-Disposition"]
    assert disposition.startswith("attachment")
    assert CATALOG in disposition
    assert "sources" in await resp.text()


async def test_a_nested_path_is_served_and_classified(client, packs):
    """A pack is a tree, so the common case is a path with separators in it.

    `kind` comes from the DIRECTORY for shared checks, not the filename — the loader finds
    them structurally, and the UI's per-file help has to agree with the loader about what
    a file is.
    """
    checks = sorted((packs / "mock_domain" / "shared" / "checks").glob("*.yaml"))
    if not checks:
        pytest.skip("the fixture pack ships no shared checks")
    rel = f"shared/checks/{checks[0].name}"
    body = await json_of(
        await client.get(f"/api/v1/knowledge/mock_domain/file?path={rel}")
    )
    assert body["path"] == rel
    assert body["kind"] == "shared_check"


# ------------------------------------------------------------------------- validate


async def test_validate_answers_200_even_when_the_pack_has_errors(client, packs):
    """A diagnostics report is not an HTTP error, and this is the distinction.

    A client written the ordinary way treats non-200 as "the request failed" and renders
    nothing. If a pack with errors answered 4xx by default, the editor would show an empty
    diagnostics panel for exactly the pack that most needs one.
    """
    (packs / "mock_domain" / CATALOG).write_text("sources: [ *dangling ]\n")
    resp = await client.get("/api/v1/knowledge/mock_domain/validate")
    assert resp.status == 200
    body = await resp.json()
    assert body["ok"] is False
    assert body["errors"] >= 1
    assert any(d["code"].startswith("yaml-") for d in body["diagnostics"])


async def test_strict_turns_the_same_body_into_a_422(client, packs):
    """For a CI job that wants a non-zero exit without parsing JSON.

    The BODY is identical — strict changes the status only. Two endpoints, or two shapes,
    would let the human view and the automated gate disagree about what a pack's state is.
    """
    (packs / "mock_domain" / CATALOG).write_text("sources: [ *dangling ]\n")
    plain = await (await client.get("/api/v1/knowledge/mock_domain/validate")).json()
    resp = await client.get("/api/v1/knowledge/mock_domain/validate?strict=1")
    assert resp.status == 422
    assert await resp.json() == plain


async def test_strict_on_a_clean_pack_is_still_200(client):
    resp = await client.get("/api/v1/knowledge/mock_domain/validate?strict=1")
    assert resp.status == 200
    assert (await resp.json())["ok"] is True


async def test_validate_on_an_unknown_pack_is_404(client):
    """A malformed name is 400 here too — the validator must not be a second door.

    Probed with an encoded slash rather than a bare `..`: the latter is normalised out of
    the path before routing (see
    `test_an_unknown_pack_is_404_and_a_malformed_name_is_400`), so it would assert the
    router's behaviour, not this handler's.
    """
    assert (await client.get("/api/v1/knowledge/nope/validate")).status == 404
    assert (
        await client.get(f"/api/v1/knowledge/{PACK}%2f..%2f{SIBLING}/validate")
    ).status == 400


async def test_a_validator_fault_is_reported_as_a_diagnostic_not_a_500(
    client, monkeypatch
):
    """The lint must never take the file browser down with it.

    The editor's entire reason for existing is to be usable on a pack that is ALREADY
    broken. A checker that 500s on the pack it cannot parse locks the operator out of the
    one pack they need to open, so a checker fault is reported in the same shape as a
    finding and says whose fault it is.
    """
    from src.knowledge import pack_validate

    def boom(_):
        raise RuntimeError("checker exploded")

    monkeypatch.setattr(pack_validate, "validate_pack", boom)
    resp = await client.get("/api/v1/knowledge/mock_domain/validate")
    assert resp.status == 200
    body = await resp.json()
    assert body["ok"] is False
    assert [d["code"] for d in body["diagnostics"]] == ["validator-failed"]
    assert "untouched" in body["diagnostics"][0]["hint"]

    # And the tree still renders, which is the point of the degradation.
    tree = await json_of(await client.get("/api/v1/knowledge/mock_domain"))
    assert tree["counts"]["files"] > 0
    assert tree["validate"]["diagnostics"][0]["code"] == "validator-failed"


async def test_the_installed_packs_validate_clean_over_http(client):
    """The same guarantee `test_pack_validate.py` asserts, through the transport.

    Worth repeating here because a handler can hand the validator the wrong directory —
    a pack NAME where a PATH was wanted validates an empty directory and reports a clean
    bill of health for a pack it never looked at.
    """
    for name in installed_packs():
        body = await json_of(await client.get(f"/api/v1/knowledge/{name}/validate"))
        assert body["errors"] == 0, body["diagnostics"]
        assert body["counts"]["sources"] > 0, "an empty count means the wrong directory"


# -------------------------------------------------------------------------- history


async def test_history_is_empty_but_present_on_an_untouched_pack(client):
    """An empty undo list is a real answer, not a 404.

    The panel renders "no earlier versions yet" from this; a 404 would render as an error
    on every freshly copied pack.
    """
    body = await json_of(await client.get("/api/v1/knowledge/mock_domain/history"))
    assert body["entries"] == []
    assert body["pack"] == "mock_domain"


async def test_history_lists_a_snapshot_and_serves_its_stored_text(client, packs):
    """The read half of undo: list the versions, then fetch one to preview or diff.

    Written through the STORE rather than through HTTP so this stays a test of the history
    READ. The write route generates history too (asserted in the write section), but a test
    that goes through it would fail here for a reason that has nothing to do with reading.
    """
    original = (packs / "mock_domain" / CATALOG).read_text()
    pack_store.write_file(
        "mock_domain", CATALOG, original + "\n# a change\n", actor="tester"
    )

    entries = (
        await json_of(await client.get("/api/v1/knowledge/mock_domain/history"))
    )["entries"]
    assert len(entries) == 1
    assert entries[0]["path"] == CATALOG
    assert entries[0]["reason"] == "write"
    assert entries[0]["actor"] == "tester"

    body = await json_of(
        await client.get(
            f"/api/v1/knowledge/mock_domain/history?snapshot={entries[0]['id']}"
        )
    )
    assert body["text"] == original, "a snapshot must be the PRIOR bytes, exactly"


async def test_history_filters_by_path(client, packs):
    for rel in (CATALOG, "entity_glossary.yaml"):
        text = (packs / "mock_domain" / rel).read_text()
        pack_store.write_file("mock_domain", rel, text + "\n# touched\n")
    body = await json_of(
        await client.get(f"/api/v1/knowledge/mock_domain/history?path={CATALOG}")
    )
    assert [e["path"] for e in body["entries"]] == [CATALOG]


async def test_an_unknown_snapshot_is_404_and_a_traversal_path_is_400(client):
    assert (
        await client.get("/api/v1/knowledge/mock_domain/history?snapshot=nope")
    ).status == 404
    assert (
        await client.get("/api/v1/knowledge/mock_domain/history?path=../../etc/passwd")
    ).status == 400


# --------------------------------------------------------------------- saving a file


async def _read(client, rel="", pack="mock_domain"):
    return await json_of(
        await client.get(f"/api/v1/knowledge/{pack}/file?path={rel or CATALOG}")
    )


async def test_a_save_writes_the_bytes_and_returns_fresh_diagnostics(client, packs):
    """The save response has to carry the pack's state, not just the file's.

    A file can be individually valid and still break the pack — the store proved the bytes
    parse before replacing anything, so a per-file "ok" is guaranteed and therefore says
    nothing. What the operator needs to know is whether the PACK still validates, and a
    caller who must make a second request to find that out generally will not.
    """
    before = await _read(client)
    resp = await client.put(
        f"/api/v1/knowledge/mock_domain/file?path={CATALOG}",
        json={"text": before["text"] + "\n# appended by a save\n", "actor": "tester"},
    )
    body = await json_of(resp)
    assert resp.status == 200
    assert body["changed"] is True
    assert body["restart_required"] is True
    assert body["validate"]["errors"] == 0
    assert body["snapshot"]
    on_disk = (packs / "mock_domain" / CATALOG).read_text()
    assert on_disk.endswith("# appended by a save\n")
    assert on_disk.startswith(before["text"][:200])
    assert body["sha256"] != before["sha256"]


async def test_a_save_that_would_make_the_file_load_empty_is_refused(client, packs):
    """THE defect this whole editor is built around, at the HTTP boundary.

    `pack._read_yaml` swallows a parse failure and returns `{}`, so a catalog replaced with
    a comment loads as a pack with ZERO sources: every condition goes `unknown` and the
    report reads INSUFFICIENT DATA, indistinguishable from the sources having had nothing.
    A 200 here would be the editor manufacturing that state on request.

    Asserted on the BYTES afterwards, not on the status code alone — a refusal that still
    truncated the file would pass a status-only check.
    """
    original = (packs / "mock_domain" / CATALOG).read_text()
    resp = await client.put(
        f"/api/v1/knowledge/mock_domain/file?path={CATALOG}",
        json={"text": "# everything commented out\n"},
    )
    assert resp.status == 400
    body = await resp.json()
    assert "empty" in body["error"]
    assert body["path"] == CATALOG
    assert (packs / "mock_domain" / CATALOG).read_text() == original


async def test_a_save_of_unparseable_yaml_carries_the_parser_s_own_line(client, packs):
    """ "Does not parse" without the line sends the operator hunting through 3000 lines.

    PyYAML already located the fault exactly; dropping that and reporting only the message
    is throwing away the one piece of information the editor could act on (it scrolls there).
    """
    original = (packs / "mock_domain" / CATALOG).read_text()
    resp = await client.put(
        f"/api/v1/knowledge/mock_domain/file?path={CATALOG}",
        json={"text": "sources:\n  - name: a\n   bad_indent: x\n"},
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["line"] > 0
    assert (packs / "mock_domain" / CATALOG).read_text() == original


async def test_a_stale_sha_is_409_and_does_not_overwrite(client, packs):
    """A concurrent-edit surface a config API does not need, and this one does.

    The editor holds several files open. Answering 200 to a save computed against bytes that
    have since moved silently discards whoever saved first — and neither of them sees an
    error, which is the worst version of this failure.
    """
    before = await _read(client)
    pack_store.write_file(
        "mock_domain", CATALOG, before["text"] + "\n# somebody else got here first\n"
    )
    theirs = (packs / "mock_domain" / CATALOG).read_text()
    resp = await client.put(
        f"/api/v1/knowledge/mock_domain/file?path={CATALOG}",
        json={"text": before["text"] + "\n# mine\n", "expect_sha": before["sha256"]},
    )
    assert resp.status == 409
    assert "reload" in (await resp.json())["error"]
    assert (packs / "mock_domain" / CATALOG).read_text() == theirs


async def test_a_line_range_save_keeps_the_rest_of_the_file_byte_identical(
    client, packs
):
    """The primitive that makes an assistant patch safe, asserted on bytes.

    A pack's rulesets are ~60% comment and `source_catalog.yaml` carries YAML anchors,
    neither of which survives a safe_load/safe_dump round trip. Replacing a line range in
    the raw text is the only edit shape that cannot lose them — so the assertion is that
    every OTHER line is unchanged, not that the file still parses.
    """
    original = (packs / "mock_domain" / CATALOG).read_text()
    lines = original.splitlines(keepends=True)
    target = next(i for i, ln in enumerate(lines) if ln.strip().startswith("#"))
    resp = await client.put(
        f"/api/v1/knowledge/mock_domain/file?path={CATALOG}",
        json={
            "text": "# replaced comment",
            "start_line": target + 1,
            "end_line": target + 1,
            "expect_first_line": lines[target].rstrip("\n"),
        },
    )
    body = await json_of(resp)
    assert resp.status == 200
    assert body["replaced"] == [target + 1, target + 1]
    after = (packs / "mock_domain" / CATALOG).read_text().splitlines(keepends=True)
    assert len(after) == len(lines)
    assert after[target] == "# replaced comment\n"
    for i, line in enumerate(lines):
        if i != target:
            assert after[i] == line, f"line {i + 1} moved and should not have"


async def test_a_line_range_whose_anchor_no_longer_matches_is_409(client, packs):
    """A stale line NUMBER is worse than a stale hash: it still points at a line.

    An assistant computes a patch against a snapshot; a human saves an edit above it; the
    numbers now name different content and the patch would land on it silently. The anchor
    is what turns that into a refusal, and the message names both sides so the caller can
    see what it found instead.
    """
    original = (packs / "mock_domain" / CATALOG).read_text()
    resp = await client.put(
        f"/api/v1/knowledge/mock_domain/file?path={CATALOG}",
        json={
            "text": "whatever: 1",
            "start_line": 1,
            "end_line": 1,
            "expect_first_line": "a line this file does not contain",
        },
    )
    assert resp.status == 409
    assert (packs / "mock_domain" / CATALOG).read_text() == original


async def test_a_save_snapshots_the_prior_bytes_before_replacing_them(client, packs):
    """Undo has to exist by the time the save returns, not on the next request.

    The prior bytes are the only copy — there is no git here — so a snapshot taken after
    the write, or on a later request, is a window in which the edit is irreversible.
    """
    original = (packs / "mock_domain" / CATALOG).read_text()
    await client.put(
        f"/api/v1/knowledge/mock_domain/file?path={CATALOG}",
        json={"text": original + "\n# v2\n"},
    )
    entries = (
        await json_of(await client.get("/api/v1/knowledge/mock_domain/history"))
    )["entries"]
    assert len(entries) == 1
    stored = await json_of(
        await client.get(
            f"/api/v1/knowledge/mock_domain/history?snapshot={entries[0]['id']}"
        )
    )
    assert stored["text"] == original


async def test_a_refused_save_leaves_no_history_entry(client, packs):
    """A history list that records saves which never happened is not an undo list.

    Its ids would stop lining up with the file's actual states, so restoring "the version
    before the last change" would reinstate something that was never on disk.
    """
    await client.put(
        f"/api/v1/knowledge/mock_domain/file?path={CATALOG}",
        json={"text": "# commented out\n"},
    )
    body = await json_of(await client.get("/api/v1/knowledge/mock_domain/history"))
    assert body["entries"] == []


async def test_an_unchanged_save_is_a_no_op_with_no_snapshot(client, packs):
    """Saving an untouched buffer must not fill the undo list with identical versions.

    The editor's save button is pressed on files that were only scrolled through. Recording
    each of those pushes the version the operator actually wants off the visible list.
    """
    before = await _read(client)
    body = await json_of(
        await client.put(
            f"/api/v1/knowledge/mock_domain/file?path={CATALOG}",
            json={"text": before["text"]},
        )
    )
    assert body["changed"] is False
    assert body["snapshot"] is None
    assert (await json_of(await client.get("/api/v1/knowledge/mock_domain/history")))[
        "entries"
    ] == []


@pytest.mark.parametrize(
    "hostile",
    ["../../etc/passwd", f"../{SIBLING}/{CATALOG}", ".history/index.json"],
)
async def test_a_save_to_a_hostile_path_is_400_and_writes_nothing(
    client, packs, hostile
):
    """Rejected, not sanitised — and here the stakes are a write, not a read.

    `normpath` on `../<sibling>/source_catalog.yaml` yields a real, writable file in a
    DIFFERENT pack. Silently editing that one is the failure mode; a 400 is the only answer
    that cannot be mistaken for success.
    """
    sibling = packs / SIBLING / CATALOG
    before = sibling.read_text() if sibling.is_file() else ""
    resp = await client.put(
        f"/api/v1/knowledge/mock_domain/file?path={hostile}", json={"text": "x: 1\n"}
    )
    assert resp.status == 400
    if before:
        assert sibling.read_text() == before
    assert not (packs / "mock_domain" / ".history" / "index.json").exists()


async def test_a_save_to_a_non_pack_suffix_is_refused(client, packs):
    """The suffix allowlist is a write bound, not a display convenience.

    Nothing in a pack is executable, and the loader reads only these suffixes — so a `.py`
    or `.sh` landing in a pack directory could only be there to be run by something else.
    """
    resp = await client.put(
        "/api/v1/knowledge/mock_domain/file?path=evil.py", json={"text": "print(1)\n"}
    )
    assert resp.status == 400
    assert not (packs / "mock_domain" / "evil.py").exists()


async def test_a_save_to_a_missing_file_is_404_not_a_new_file(client, packs):
    """A mistyped path on a save must NOT create a file.

    This is why create is a separate POST. A save that silently creates leaves the edit in a
    file no loader reads, and the operator's change appears to have simply vanished.
    """
    resp = await client.put(
        "/api/v1/knowledge/mock_domain/file?path=typoed_name.yaml",
        json={"text": "a: 1\n"},
    )
    assert resp.status == 404
    assert not (packs / "mock_domain" / "typoed_name.yaml").exists()


async def test_a_save_over_the_write_cap_is_413(client, packs):
    """413, not 400: the request was well-formed and the next move is different.

    The app's own `client_max_size` is a far larger ceiling (a PDF attachment has to fit), so
    a per-handler cap is the only thing that makes this bound real.
    """
    original = (packs / "mock_domain" / CATALOG).read_text()
    huge = original + "\n" + ("# padding\n" * (pack_store.WRITE_MAX_BYTES // 8))
    resp = await client.put(
        f"/api/v1/knowledge/mock_domain/file?path={CATALOG}", json={"text": huge}
    )
    assert resp.status == 413
    assert (packs / "mock_domain" / CATALOG).read_text() == original


async def test_a_save_body_must_be_an_object_with_text(client):
    for body in ({}, {"text": 5}, [1, 2]):
        resp = await client.put(
            f"/api/v1/knowledge/mock_domain/file?path={CATALOG}", json=body
        )
        assert resp.status == 400, body


async def test_a_non_numeric_line_number_is_400_not_500(client):
    """A malformed request, not a store fault — the distinction the caller acts on.

    A 500 tells the operator the server is broken and there is nothing they can do; a 400
    tells them to fix the request.
    """
    resp = await client.put(
        f"/api/v1/knowledge/mock_domain/file?path={CATALOG}",
        json={"text": "x", "start_line": "one", "end_line": 2},
    )
    assert resp.status == 400


# -------------------------------------------------------------------- creating a file


async def test_creating_a_file_returns_its_kind_and_lands_on_disk(client, packs):
    """`kind` comes back because the editor's per-file help is driven by it.

    A newly created `shared/checks/*.yaml` has to be recognised as a check library
    immediately — otherwise the author gets the generic help for the one file they most
    need the specific help for.
    """
    resp = await client.post(
        "/api/v1/knowledge/mock_domain/file?path=shared/checks/new_library.yaml",
        json={"text": "one_actor:\n  kind: distinct_count\n", "actor": "tester"},
    )
    body = await json_of(resp)
    assert resp.status == 200
    assert body["created"] is True
    assert body["kind"] == "shared_check"
    assert body["validate"]["errors"] == 0
    assert (
        packs / "mock_domain" / "shared" / "checks" / "new_library.yaml"
    ).read_text() == "one_actor:\n  kind: distinct_count\n"


async def test_creating_an_existing_file_is_409_and_does_not_overwrite(client, packs):
    original = (packs / "mock_domain" / CATALOG).read_text()
    resp = await client.post(
        f"/api/v1/knowledge/mock_domain/file?path={CATALOG}", json={"text": "x: 1\n"}
    )
    assert resp.status == 409
    assert (packs / "mock_domain" / CATALOG).read_text() == original


async def test_an_empty_new_file_is_allowed(client, packs):
    """A stub is a legitimate first state, and refusing it makes the editor unusable.

    The "parses to empty" refusal exists to stop an edit DESTROYING facts. There are none
    to lose in a file that does not exist yet, so applying it here would mean a new file
    could only ever be created with its final content already typed.
    """
    resp = await client.post(
        "/api/v1/knowledge/mock_domain/file?path=notes.md", json={"text": ""}
    )
    assert resp.status == 200
    assert (packs / "mock_domain" / "notes.md").read_text() == ""


async def test_creating_in_an_unknown_pack_is_404(client, packs):
    resp = await client.post(
        "/api/v1/knowledge/no_such_pack/file?path=a.yaml", json={"text": "a: 1\n"}
    )
    assert resp.status == 404
    assert not (packs / "no_such_pack").exists()


# -------------------------------------------------------------------- deleting a file


async def test_a_delete_without_confirm_is_refused(client, packs):
    """The operator's instruction was that nothing is deleted unless they ask.

    So the ask lives in the REQUEST, not in a dialog the UI happens to show: a script, a
    retried request or a second front-end would otherwise not be covered by it.
    """
    resp = await client.delete(f"/api/v1/knowledge/mock_domain/file?path={CATALOG}")
    assert resp.status == 400
    assert "confirm=1" in (await resp.json())["error"]
    assert (packs / "mock_domain" / CATALOG).is_file()


async def test_a_confirmed_delete_removes_the_file_and_keeps_its_content(client, packs):
    """Deletion is reversible, which is what makes offering it at all defensible."""
    original = (packs / "mock_domain" / CATALOG).read_text()
    body = await json_of(
        await client.delete(
            f"/api/v1/knowledge/mock_domain/file?path={CATALOG}&confirm=1"
        )
    )
    assert body["deleted"] is True
    assert not (packs / "mock_domain" / CATALOG).exists()
    stored = await json_of(
        await client.get(
            f"/api/v1/knowledge/mock_domain/history?snapshot={body['snapshot']}"
        )
    )
    assert stored["text"] == original


async def test_deleting_the_catalog_is_reported_as_a_broken_pack(client, packs):
    """The response says what the pack now IS, not merely that the request succeeded.

    Deleting the source catalog is exactly the silent-empty state: the pack still loads,
    with zero sources. The diagnostics riding on the delete response are what turn that from
    something discovered on the next run into something visible now.
    """
    body = await json_of(
        await client.delete(
            f"/api/v1/knowledge/mock_domain/file?path={CATALOG}&confirm=1"
        )
    )
    codes = {d["code"] for d in body["validate"]["diagnostics"]}
    assert "no-sources-declared" in codes


async def test_deleting_a_missing_file_is_404(client):
    resp = await client.delete(
        "/api/v1/knowledge/mock_domain/file?path=never_existed.yaml&confirm=1"
    )
    assert resp.status == 404


async def test_a_delete_of_a_hostile_path_is_400(client, packs):
    sibling = packs / SIBLING / CATALOG
    before = sibling.read_text() if sibling.is_file() else ""
    resp = await client.delete(
        f"/api/v1/knowledge/{PACK}/file?path=../{SIBLING}/{CATALOG}&confirm=1"
    )
    assert resp.status == 400
    if before:
        assert sibling.read_text() == before


# ------------------------------------------------------------------------- restoring


async def test_a_restore_puts_the_prior_bytes_back(client, packs):
    original = (packs / "mock_domain" / CATALOG).read_text()
    saved = await json_of(
        await client.put(
            f"/api/v1/knowledge/mock_domain/file?path={CATALOG}",
            json={"text": original + "\n# regrettable\n"},
        )
    )
    resp = await client.post(
        "/api/v1/knowledge/mock_domain/history/restore",
        json={"snapshot": saved["snapshot"], "actor": "tester"},
    )
    body = await json_of(resp)
    assert resp.status == 200
    assert body["parses"] is True
    assert (packs / "mock_domain" / CATALOG).read_text() == original


async def test_a_restore_is_itself_undoable(client, packs):
    """Otherwise the first undo destroys the state it was undoing FROM.

    An operator who restores the wrong version has then lost the work they were trying to
    keep, and nothing in the UI would have warned them that undo was one-way.
    """
    original = (packs / "mock_domain" / CATALOG).read_text()
    edited = original + "\n# work in progress\n"
    saved = await json_of(
        await client.put(
            f"/api/v1/knowledge/mock_domain/file?path={CATALOG}", json={"text": edited}
        )
    )
    await client.post(
        "/api/v1/knowledge/mock_domain/history/restore",
        json={"snapshot": saved["snapshot"]},
    )
    entries = (
        await json_of(await client.get("/api/v1/knowledge/mock_domain/history"))
    )["entries"]
    reasons = [e["reason"] for e in entries]
    assert "restore" in reasons
    restore_entry = next(e for e in entries if e["reason"] == "restore")
    back = await json_of(
        await client.get(
            f"/api/v1/knowledge/mock_domain/history?snapshot={restore_entry['id']}"
        )
    )
    assert back["text"] == edited, "the state being undone from must be recoverable"


async def test_a_restore_recreates_a_deleted_file(client, packs):
    original = (packs / "mock_domain" / CATALOG).read_text()
    deleted = await json_of(
        await client.delete(
            f"/api/v1/knowledge/mock_domain/file?path={CATALOG}&confirm=1"
        )
    )
    body = await json_of(
        await client.post(
            "/api/v1/knowledge/mock_domain/history/restore",
            json={"snapshot": deleted["snapshot"]},
        )
    )
    assert body["recreated"] is True
    assert (packs / "mock_domain" / CATALOG).read_text() == original
    assert body["validate"]["errors"] == 0


async def test_restoring_content_that_does_not_parse_still_works_and_says_so(
    client, packs
):
    """Undo must not be conditional on the broken version having been valid.

    The moment undo is most needed is right after an edit went wrong — refusing to reinstate
    bytes that were previously on disk would make it unavailable precisely then. So the
    restore proceeds and `parses` reports the truth, rather than the operator finding out on
    the next run.
    """
    broken = "sources: [ *dangling ]\n"
    (packs / "mock_domain" / CATALOG).write_text(broken)
    snap = pack_store.snapshot("mock_domain", CATALOG, broken, reason="write")
    (packs / "mock_domain" / CATALOG).write_text("sources:\n  - name: fine\n")
    body = await json_of(
        await client.post(
            "/api/v1/knowledge/mock_domain/history/restore",
            json={"snapshot": snap["id"]},
        )
    )
    assert body["parses"] is False
    assert body["parse_error"]
    assert (packs / "mock_domain" / CATALOG).read_text() == broken
    assert body["validate"]["errors"] > 0, "and the pack is reported as broken"


async def test_an_unknown_snapshot_restore_is_404_and_a_missing_id_is_400(client):
    assert (
        await client.post(
            "/api/v1/knowledge/mock_domain/history/restore", json={"snapshot": "nope"}
        )
    ).status == 404
    assert (
        await client.post("/api/v1/knowledge/mock_domain/history/restore", json={})
    ).status == 400


# --------------------------------------------------------------------------- import


async def test_an_import_writes_every_file_and_returns_diagnostics(client, packs):
    body = await json_of(
        await client.post(
            "/api/v1/knowledge/mock_domain/import",
            json={
                "files": {
                    "shared/concepts/a_note.md": "# a note\n\nSome text.\n",
                    "shared/checks/imported.yaml": "one:\n  kind: distinct_count\n",
                },
                "actor": "tester",
            },
        )
    )
    assert len(body["written"]) == 2
    assert body["restart_required"] is True
    assert (packs / "mock_domain" / "shared" / "checks" / "imported.yaml").is_file()
    assert (packs / "mock_domain" / "shared" / "concepts" / "a_note.md").is_file()


async def test_an_import_validates_every_file_before_writing_any(client, packs):
    """The all-or-nothing promise, asserted on the file that was VALID.

    A multi-file import is usually one coherent change — a ruleset and the check it imports.
    Landing half of it leaves a pack broken in a way neither file's author would recognise,
    and the config importer's own history is the precedent: enforcing a rule only at write
    time returned a 500 and wrote the files ahead of the bad one.
    """
    resp = await client.post(
        "/api/v1/knowledge/mock_domain/import",
        json={
            "files": {
                "shared/checks/good.yaml": "one:\n  kind: distinct_count\n",
                "shared/checks/bad.yaml": "one:\n  - a\n   bad: indent\n",
            }
        },
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["error"] == "Import rejected; nothing was written"
    assert any("bad.yaml" in e for e in body["errors"])
    assert not (packs / "mock_domain" / "shared" / "checks" / "good.yaml").exists()
    assert not (packs / "mock_domain" / "shared" / "checks" / "bad.yaml").exists()


async def test_an_import_over_an_existing_file_snapshots_it(client, packs):
    """Overwriting is the point of an import, so it has to be reversible file by file."""
    original = (packs / "mock_domain" / CATALOG).read_text()
    await client.post(
        "/api/v1/knowledge/mock_domain/import",
        json={"files": {CATALOG: original + "\n# from an import\n"}},
    )
    entries = (
        await json_of(
            await client.get(f"/api/v1/knowledge/mock_domain/history?path={CATALOG}")
        )
    )["entries"]
    assert [e["reason"] for e in entries] == ["import"]
    stored = await json_of(
        await client.get(
            f"/api/v1/knowledge/mock_domain/history?snapshot={entries[0]['id']}"
        )
    )
    assert stored["text"] == original


async def test_an_import_rejects_a_hostile_path_without_writing_the_others(
    client, packs
):
    sibling = packs / SIBLING / CATALOG
    before = sibling.read_text() if sibling.is_file() else ""
    resp = await client.post(
        "/api/v1/knowledge/mock_domain/import",
        json={
            "files": {
                "shared/checks/fine.yaml": "one:\n  kind: distinct_count\n",
                f"../{SIBLING}/{CATALOG}": "sources: []\n",
            }
        },
    )
    assert resp.status == 400
    assert not (packs / "mock_domain" / "shared" / "checks" / "fine.yaml").exists()
    if before:
        assert sibling.read_text() == before


async def test_an_import_body_must_carry_files_and_the_pack_must_exist(client):
    assert (
        await client.post("/api/v1/knowledge/mock_domain/import", json={"files": {}})
    ).status == 400
    assert (
        await client.post(
            "/api/v1/knowledge/nope/import", json={"files": {"a.yaml": "a: 1\n"}}
        )
    ).status == 404


# -------------------------------------------------------------------------- scaffold


async def test_scaffolding_creates_a_pack_that_validates(client, packs):
    """The baseline has to be GREEN, or the author cannot tell their edit from the template.

    A scaffold that ships with errors means the first diagnostics an author sees are not
    theirs, and they learn to ignore the panel.
    """
    body = await json_of(
        await client.post(
            "/api/v1/knowledge/scaffold",
            json={"name": "new_pack", "vocabulary": ["widget", "widget id"]},
        )
    )
    assert body["created"] is True
    assert body["validate"]["errors"] == 0, body["validate"]["diagnostics"]
    assert body["restart_required"] is True
    assert (packs / "new_pack" / "domain_vocabulary.yaml").is_file()
    assert "README.md" not in body["files"]
    listed = await json_of(await client.get("/api/v1/knowledge"))
    assert "new_pack" in [p["name"] for p in listed["packs"]]


async def test_a_scaffold_writes_the_callers_vocabulary_not_the_templates(
    client, packs
):
    """A verbatim copy would fail the TEMPLATE's own test, not just this pack's.

    The template's placeholder vocabulary shares a stem with a key in its own example
    ruleset, and the neutrality scan reads every *installed* pack's vocabulary against the
    template. So the one file a scaffold must not copy is the one that declares what the
    engine may not say.
    """
    await client.post(
        "/api/v1/knowledge/scaffold",
        json={"name": "vocab_pack", "vocabulary": ["Widget", "widget", "gizmo"]},
    )
    text = (packs / "vocab_pack" / "domain_vocabulary.yaml").read_text()
    assert "- widget\n" in text
    assert "- gizmo\n" in text
    assert text.count("- widget\n") == 1, "case-folded duplicates must collapse"
    template_vocab = (pack_store.TEMPLATE_DIR / "domain_vocabulary.yaml").read_text()
    declared = [
        ln.strip()[2:]
        for ln in template_vocab.splitlines()
        if ln.strip().startswith("- ")
    ]
    for word in declared:
        assert f"- {word}\n" not in text, f"the template's own word {word!r} was copied"


async def test_a_scaffold_without_a_vocabulary_is_refused(client, packs):
    """Not paperwork: a pack with no vocabulary file takes the whole SUITE down.

    `test_every_installed_pack_declares_its_vocabulary` scans installed packs, and the
    guarantee that file installs — the engine provably does not speak this domain's words —
    would be quietly absent for the pack most likely to introduce a leak.
    """
    for payload in (
        {"name": "no_vocab"},
        {"name": "no_vocab", "vocabulary": []},
        {"name": "no_vocab", "vocabulary": ["  "]},
    ):
        resp = await client.post("/api/v1/knowledge/scaffold", json=payload)
        assert resp.status == 400, payload
        assert not (packs / "no_vocab").exists()


async def test_a_scaffold_over_an_existing_pack_is_409(client, packs):
    resp = await client.post(
        "/api/v1/knowledge/scaffold",
        json={"name": "mock_domain", "vocabulary": ["widget"]},
    )
    assert resp.status == 409
    assert (packs / "mock_domain" / CATALOG).is_file(), "the real pack is untouched"


async def test_a_scaffold_with_a_hostile_name_writes_nothing(client, packs):
    for name in ("../escaped", "has/slash", ".history", "has space"):
        resp = await client.post(
            "/api/v1/knowledge/scaffold", json={"name": name, "vocabulary": ["widget"]}
        )
        assert resp.status == 400, name
    assert sorted(p.name for p in packs.iterdir()) == sorted(
        installed_packs() + [SIBLING]
    )


# ------------------------------------------------------------------ routing invariants


def _knowledge_routes():
    iface = IncidentInputInterface(_config())
    out = []
    for route in iface.app.router.routes():
        path = getattr(route.resource, "canonical", "") or ""
        if path.startswith("/api/v1/knowledge"):
            out.append((route.method, path))
    return out


def test_the_routes_exist_without_a_job_manager_or_a_pipeline():
    """The pack editor has no dependency on the pipeline, and that is deliberate.

    Authoring a pack is what happens BEFORE anything can run. Registering these routes
    conditionally would also make `test_webui.py` blind to them — it builds this class with
    no job manager, so its fetch-URL scan would pass vacuously while every button 404s.
    """
    routes = _knowledge_routes()
    paths = {path for _, path in routes}
    assert "/api/v1/knowledge" in paths
    assert "/api/v1/knowledge/{pack}" in paths
    assert "/api/v1/knowledge/{pack}/file" in paths
    assert "/api/v1/knowledge/{pack}/validate" in paths
    assert "/api/v1/knowledge/{pack}/history" in paths
    assert "/api/v1/knowledge/{pack}/tree" in paths


def test_the_mutating_routes_are_exactly_these_four():
    """THE BLAST RADIUS, enumerated against the router instead of trusted as prose.

    The read half could assert "no mutating route exists at all". Now that writes are
    registered, the equivalent guarantee is that the mutating surface is *exactly* this
    list — so a fifth one fails here and whoever added it has to state, in this file, what
    it can destroy. Every entry below is one `pack_store` call with a snapshot behind it.

    Deletion is on the file route rather than a route of its own, deliberately: the method
    carries the intent (DELETE), and the `confirm=1` requirement is asserted separately.

    THREE OF THESE ARE THE ASSISTANT'S, and only one of the three writes. `POST /assist`
    and `/reject` are POSTs because they CREATE a session, not because they change a pack —
    the assistant has no write tool, so `/assist/{session}/apply` is the single point where
    a proposal can reach disk, and it needs a human to call it.
    """
    mutating = {
        (method, path)
        for method, path in _knowledge_routes()
        if method not in ("GET", "HEAD")
    }
    assert mutating == {
        ("PUT", "/api/v1/knowledge/{pack}/file"),
        ("POST", "/api/v1/knowledge/{pack}/file"),
        ("DELETE", "/api/v1/knowledge/{pack}/file"),
        ("POST", "/api/v1/knowledge/{pack}/history/restore"),
        ("POST", "/api/v1/knowledge/{pack}/import"),
        ("POST", "/api/v1/knowledge/scaffold"),
        ("POST", "/api/v1/knowledge/{pack}/assist"),
        ("POST", "/api/v1/knowledge/{pack}/assist/{session}/apply"),
        ("POST", "/api/v1/knowledge/{pack}/assist/{session}/reject"),
    }


def test_the_literal_scaffold_path_is_not_swallowed_by_the_dynamic_pack_route():
    """Registration ORDER is the only mechanism here, and it is now load-bearing.

    aiohttp matches in registration order. `/{pack}` registered ahead of the literal
    `/scaffold` would swallow it, and the create-pack POST would arrive at the read handler
    as a pack named "scaffold" — a 404 for a route that exists, which reads as "the feature
    is missing" rather than "the routes are in the wrong order".
    """
    iface = IncidentInputInterface(_config())
    ordered = [
        getattr(r.resource, "canonical", "")
        for r in iface.app.router.routes()
        if str(getattr(r.resource, "canonical", "")).startswith("/api/v1/knowledge")
    ]
    dynamic = ordered.index("/api/v1/knowledge/{pack}")
    literals = [p for p in ordered[:dynamic] if "{" not in p]
    assert (
        "/api/v1/knowledge" in literals
    ), "the collection route must precede the dynamic one"
    assert (
        "/api/v1/knowledge/scaffold" in literals
    ), "the scaffold literal must be registered before /{pack}"


async def test_the_scaffold_literal_really_reaches_its_own_handler(client, packs):
    """The order test reads the router; this proves the request lands.

    A POST to `/api/v1/knowledge/scaffold` must not arrive as "pack named scaffold". The
    tell is which error comes back: the scaffold handler complains about the BODY, whereas
    the dynamic route would answer 404/405 for a pack that does not exist.
    """
    resp = await client.post("/api/v1/knowledge/scaffold", json={})
    assert resp.status == 400
    assert "name" in (await resp.json())["error"]
    assert not (packs / "scaffold").exists()


# ------------------------------------------------------------------ the assistant


class ScriptedClient:
    """A minimal stand-in for `LLMClient` — see `tests/CLAUDE.md`: never a real model.

    `plan` is what `structured_output` returns. `turns` scripts `tool_call`; empty means
    the model asks for nothing, which is the loop's own fallback path.
    """

    def __init__(self, plan=None, turns=None, accept_images=True):
        self.plan = plan
        self.turns = list(turns or [])
        self.calls = 0
        self.accept_images = accept_images
        self.seen = []

    async def complete(self, messages, rag=None):
        """The image probe's one call. Present because ABSENT would also pass.

        The assistant treats any exception as a refusal, so a client with no `complete` at
        all degrades to `text_only` — which is the right behaviour but would leave the
        accepting path untested, and that is the path a real multimodal endpoint takes.
        """
        self.seen.append(messages)
        if not self.accept_images:
            raise ValueError("this endpoint cannot read images")
        return "ok"

    async def tool_call(
        self, messages, tools, tool_choice="auto", rag=None, stage=None
    ):
        self.calls += 1
        self.seen.append(messages)
        if not self.turns:
            return _Msg()
        return self.turns.pop(0)

    async def structured_output(
        self, messages, response_model, rag=None, max_tokens=None, stage=None
    ):
        return self.plan if self.plan is not None else pack_assistant.EditPlan()


class _Msg:
    tool_calls = None
    content = ""


@pytest.fixture
async def assist_client(packs):
    """The interface WITH an llm_client, so the assist routes are live.

    Separate from `client` on purpose: that fixture wires none, and the assist endpoints
    must answer 503 there rather than 500. Both states are real deployments — a pack editor
    on a box with no model endpoint is exactly the case the manual half exists for.
    """
    iface = IncidentInputInterface(
        _config(),
        live_config={"knowledge": {"pack_dir": "knowledge/mock_domain"}},
        llm_client=ScriptedClient(),
    )
    iface.rate_limiter = AsyncRateLimiter(rate_limit=1000, time_period=1)
    c = TestClient(TestServer(iface.app))
    await c.start_server()
    yield c, iface
    iface.rate_limiter.close()
    await c.close()


@pytest.fixture(autouse=True)
def _clean_assist_sessions():
    pack_assistant._SESSIONS.clear()
    yield
    pack_assistant._SESSIONS.clear()


async def _await_session(c, pack, session_id, want=("proposed", "failed")):
    """Poll the session until it reaches a terminal state.

    The start endpoint answers 202 and runs the loop in the background, so a test that read
    the plan immediately would be asserting against a session mid-exploration. Bounded, so a
    loop that never finishes fails here rather than hanging the suite.
    """
    for _ in range(200):
        body = await json_of(
            await c.get(f"/api/v1/knowledge/{pack}/assist/{session_id}")
        )
        if body["status"] in want:
            return body
        await asyncio.sleep(0.01)
    raise AssertionError(f"session {session_id} never reached {want}: {body['status']}")


async def test_the_assist_routes_answer_503_without_an_llm_endpoint(client):
    """No model wired is a real deployment, and the manual editor is unaffected.

    503 rather than 500, and the message says which half still works — an operator who
    reads "internal server error" concludes the whole tab is broken and stops using the
    part that isn't.
    """
    resp = await client.post(
        "/api/v1/knowledge/mock_domain/assist", json={"question": "add a check"}
    )
    assert resp.status == 503
    body = await resp.json()
    assert "manual editor is unaffected" in body["error"]
    # And the read routes are unaffected too.
    assert (await client.get("/api/v1/knowledge/mock_domain")).status == 200


async def test_an_assist_request_returns_a_session_immediately_and_writes_nothing(
    assist_client, packs
):
    """202, not 200: exploration is several round trips.

    Holding the request open for them means a proxy timeout is indistinguishable from a
    model that produced nothing — and the trail, which is how the proposal gets judged, is
    only watchable if the caller has the session id while it is still running.
    """
    c, iface = assist_client
    before = (packs / "mock_domain" / CATALOG).read_bytes()
    iface.llm_client.plan = pack_assistant.EditPlan(
        summary="add a comment",
        ops=[
            pack_assistant.EditOp(
                op="create",
                path="shared/concepts/proposed.md",
                text="# proposed\n\nA note.\n",
                reason="the operator asked for it",
            )
        ],
    )
    resp = await c.post(
        "/api/v1/knowledge/mock_domain/assist", json={"question": "add a note"}
    )
    assert resp.status == 202
    started = await resp.json()
    assert started["status"] in ("queued", "exploring")
    body = await _await_session(c, "mock_domain", started["session"])
    assert body["status"] == "proposed"
    assert body["plan"]["summary"] == "add a comment"
    assert body["preview"]["ops"][0]["diff"].startswith(
        "--- a/shared/concepts/proposed.md"
    )
    # THE POINT: a proposal is not a write.
    assert not (packs / "mock_domain" / "shared/concepts/proposed.md").exists()
    assert (packs / "mock_domain" / CATALOG).read_bytes() == before


async def test_an_assist_request_needs_a_question(assist_client):
    c, _ = assist_client
    resp = await c.post("/api/v1/knowledge/mock_domain/assist", json={})
    assert resp.status == 400
    assert "question" in (await resp.json())["error"]


async def test_an_oversized_question_is_413(assist_client):
    c, _ = assist_client
    resp = await c.post(
        "/api/v1/knowledge/mock_domain/assist", json={"question": "x" * 20_001}
    )
    assert resp.status == 413


async def test_an_assist_on_an_unknown_pack_is_404(assist_client):
    c, _ = assist_client
    resp = await c.post("/api/v1/knowledge/nope/assist", json={"question": "hi"})
    assert resp.status == 404


async def test_an_assist_on_a_hostile_pack_name_is_rejected(assist_client):
    """Rejected, not sanitised — the same rule as every other route in this file."""
    c, _ = assist_client
    resp = await c.post(
        "/api/v1/knowledge/..%2F..%2Fetc/assist", json={"question": "hi"}
    )
    assert resp.status in (400, 404)


async def test_applying_a_proposal_writes_it_and_reports_restart_required(
    assist_client, packs
):
    """The one route that writes, and it still cannot skip the store's guards.

    `restart_required` rides on the response for the same reason every manual write carries
    it: the pack this process holds is not reloaded, and claiming otherwise would be worse
    than offering no assistant.
    """
    c, iface = assist_client
    iface.llm_client.plan = pack_assistant.EditPlan(
        summary="a note",
        ops=[
            pack_assistant.EditOp(
                op="create", path="shared/concepts/applied.md", text="# applied\n"
            )
        ],
    )
    started = await (
        await c.post("/api/v1/knowledge/mock_domain/assist", json={"question": "note"})
    ).json()
    await _await_session(c, "mock_domain", started["session"])
    resp = await c.post(
        f"/api/v1/knowledge/mock_domain/assist/{started['session']}/apply",
        json={"actor": "tester"},
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["applied"] is True
    assert body["restart_required"] is True
    assert body["validate"]["errors"] == 0
    assert (
        packs / "mock_domain" / "shared/concepts/applied.md"
    ).read_text() == "# applied\n"


async def test_an_apply_that_fails_validation_is_400_and_writes_nothing(
    assist_client, packs
):
    """All-or-nothing, over HTTP, with the per-op reasons in the body.

    The reasons are what make the next attempt a correction instead of a re-run — a bare
    "rejected" leaves the operator with no idea which of four ops was the problem.
    """
    c, iface = assist_client
    target = packs / "mock_domain" / CATALOG
    before = target.read_bytes()
    first = target.read_text().splitlines()[0]
    iface.llm_client.plan = pack_assistant.EditPlan(
        ops=[
            pack_assistant.EditOp(
                op="patch",
                path=CATALOG,
                text="# fine",
                start_line=1,
                end_line=1,
                expect_first_line=first,
                expect_last_line=first,
            ),
            pack_assistant.EditOp(op="delete", path="entity_glossary.yaml"),
        ]
    )
    started = await (
        await c.post("/api/v1/knowledge/mock_domain/assist", json={"question": "edit"})
    ).json()
    await _await_session(c, "mock_domain", started["session"])
    resp = await c.post(
        f"/api/v1/knowledge/mock_domain/assist/{started['session']}/apply", json={}
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["applied"] is False
    assert any("deletion box" in e for e in body["errors"])
    assert target.read_bytes() == before
    assert (packs / "mock_domain" / "entity_glossary.yaml").is_file()


async def test_an_edited_plan_overrides_the_proposal_and_is_recorded_as_edited(
    assist_client, packs
):
    """ "Edit first" — because an operator who can only accept or reject accepts a wrong one.

    The edited plan runs through the identical guards, and the session records what actually
    landed rather than what was proposed: an apply that reads back as the model's own plan
    makes the intervention untraceable.
    """
    c, iface = assist_client
    iface.llm_client.plan = pack_assistant.EditPlan(
        summary="the model's version",
        ops=[
            pack_assistant.EditOp(
                op="create", path="shared/concepts/model.md", text="# model\n"
            )
        ],
    )
    started = await (
        await c.post("/api/v1/knowledge/mock_domain/assist", json={"question": "note"})
    ).json()
    await _await_session(c, "mock_domain", started["session"])
    resp = await c.post(
        f"/api/v1/knowledge/mock_domain/assist/{started['session']}/apply",
        json={
            "plan": {
                "summary": "the operator's version",
                "ops": [
                    {
                        "op": "create",
                        "path": "shared/concepts/operator.md",
                        "text": "# operator\n",
                    }
                ],
            }
        },
    )
    assert resp.status == 200
    assert (packs / "mock_domain" / "shared/concepts/operator.md").is_file()
    assert not (packs / "mock_domain" / "shared/concepts/model.md").exists()
    body = await json_of(
        await c.get(f"/api/v1/knowledge/mock_domain/assist/{started['session']}")
    )
    assert body["status"] == "applied"
    assert body["plan"]["summary"] == "the operator's version"


async def test_a_reject_needs_a_correction_and_starts_a_new_session(assist_client):
    """A reject with no guidance would re-send the same request and return the same plan.

    Which reads, to the operator, as the assistant ignoring them. And the retry is a NEW
    session so the rejected proposal stays inspectable beside its replacement.
    """
    c, iface = assist_client
    iface.llm_client.plan = pack_assistant.EditPlan(summary="v1")
    started = await (
        await c.post("/api/v1/knowledge/mock_domain/assist", json={"question": "note"})
    ).json()
    await _await_session(c, "mock_domain", started["session"])
    bare = await c.post(
        f"/api/v1/knowledge/mock_domain/assist/{started['session']}/reject", json={}
    )
    assert bare.status == 400
    assert "same plan" in (await bare.json())["error"]
    iface.llm_client.plan = pack_assistant.EditPlan(summary="v2")
    resp = await c.post(
        f"/api/v1/knowledge/mock_domain/assist/{started['session']}/reject",
        json={"guidance": "put it in the shared library instead"},
    )
    assert resp.status == 202
    body = await resp.json()
    assert body["rejected"] == started["session"]
    assert body["session"] != started["session"]
    retried = await _await_session(c, "mock_domain", body["session"])
    assert retried["plan"]["summary"] == "v2"
    old = await json_of(
        await c.get(f"/api/v1/knowledge/mock_domain/assist/{started['session']}")
    )
    assert old["status"] == "rejected"
    assert old["plan"]["summary"] == "v1", "the rejected proposal must stay readable"


async def test_a_session_from_another_pack_is_409_not_applied(assist_client, packs):
    """Every op's path is relative to a pack root, and the same path exists in most packs.

    So a stale tab must not be able to apply one pack's plan into another — the ops would
    validate and the write would succeed.
    """
    c, iface = assist_client
    iface.llm_client.plan = pack_assistant.EditPlan(
        summary="x",
        ops=[
            pack_assistant.EditOp(
                op="create", path="shared/concepts/x.md", text="# x\n"
            )
        ],
    )
    started = await (
        await c.post("/api/v1/knowledge/mock_domain/assist", json={"question": "note"})
    ).json()
    await _await_session(c, "mock_domain", started["session"])
    resp = await c.post(
        f"/api/v1/knowledge/{SIBLING}/assist/{started['session']}/apply", json={}
    )
    assert resp.status == 409
    assert not (packs / SIBLING / "shared/concepts/x.md").exists()


async def test_an_unknown_session_is_404_and_says_why_it_is_gone(assist_client):
    """Sessions are in memory by design; "no such session" without the reason reads as a bug."""
    c, _ = assist_client
    resp = await c.get("/api/v1/knowledge/mock_domain/assist/assist-99999")
    assert resp.status == 404
    assert "dropped on restart" in (await resp.json())["error"]


async def test_the_session_stream_replays_the_trail_and_closes(assist_client):
    """SSE, and it must END — a stream that stays open after a terminal status leaks.

    Replay-then-tail is the same contract as the job stream, for the same reason: which
    files were read is how the proposal is judged, and a UI that subscribes after the first
    tool round would otherwise see none of it.
    """
    c, iface = assist_client
    iface.llm_client.plan = pack_assistant.EditPlan(summary="done")
    started = await (
        await c.post("/api/v1/knowledge/mock_domain/assist", json={"question": "note"})
    ).json()
    await _await_session(c, "mock_domain", started["session"])
    resp = await c.get(
        f"/api/v1/knowledge/mock_domain/assist/{started['session']}/events"
    )
    assert resp.status == 200
    assert resp.headers["Content-Type"].startswith("text/event-stream")
    text = await asyncio.wait_for(resp.text(), timeout=5)
    events = [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]
    assert events, text
    assert events[-1]["type"] == "assist_status"
    assert events[-1]["status"] == "proposed"


async def test_the_session_list_names_this_packs_sessions_only(assist_client):
    c, iface = assist_client
    iface.llm_client.plan = pack_assistant.EditPlan(summary="x")
    started = await (
        await c.post("/api/v1/knowledge/mock_domain/assist", json={"question": "note"})
    ).json()
    await _await_session(c, "mock_domain", started["session"])
    mine = await json_of(await c.get("/api/v1/knowledge/mock_domain/assist"))
    assert [s["session"] for s in mine["sessions"]] == [started["session"]]
    assert mine["max_sessions"] == pack_assistant.MAX_SESSIONS
    other = await json_of(await c.get(f"/api/v1/knowledge/{SIBLING}/assist"))
    assert other["sessions"] == []


def test_the_literal_assist_path_is_not_swallowed_by_the_session_route():
    """Registration order again: `/assist/{session}` ahead of `/assist` would swallow it."""
    iface = IncidentInputInterface(_config())
    ordered = [
        getattr(r.resource, "canonical", "")
        for r in iface.app.router.routes()
        if str(getattr(r.resource, "canonical", "")).startswith("/api/v1/knowledge")
    ]
    assert ordered.index("/api/v1/knowledge/{pack}/assist") < ordered.index(
        "/api/v1/knowledge/{pack}/assist/{session}"
    )


# --- attachments over the wire


async def _started(resp):
    """The body of a 202, with the status asserted rather than assumed.

    A 400 also carries JSON, so reading the body without checking the code turns "the upload
    was refused" into a KeyError several lines later, pointing at the wrong thing.
    """
    assert resp.status == 202, await resp.text()
    return await json_of(resp)


def _upload(name, data):
    """An upload shaped as the UI sends it: base64, optionally a data URL."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return {"name": name, "content": base64.b64encode(data).decode("ascii")}


async def test_a_document_is_converted_and_reported_in_the_202(assist_client, packs):
    """Conversion happens BEFORE the session starts, and that is the point.

    A PDF that cannot be read is a fact the operator should get in the response to their
    click, not discover as a missing attachment three exploration turns later.
    """
    c, iface = assist_client
    started = await _started(
        await c.post(
            "/api/v1/knowledge/mock_domain/assist",
            json={
                "question": "fold this table into the catalogue",
                "attachments": [_upload("table.csv", b"name,kind\nalpha,stub\n")],
            },
        )
    )
    assert [a["name"] for a in started["attachments"]] == ["table.csv"]
    assert started["attachments"][0]["kind"] == "document"
    assert started["attachment_errors"] == []
    body = await _await_session(c, "mock_domain", started["session"])
    assert body["status"] == "proposed"
    # The converted text reached the model, not the base64.
    text = "".join(str(m.get("content")) for m in iface.llm_client.seen[0])
    assert "alpha" in text


async def test_a_bad_attachment_is_named_and_the_good_ones_still_run(
    assist_client, packs
):
    """NOT all-or-nothing, unlike the importer — and the asymmetry is deliberate.

    An import writes files, so a partial one leaves a pack half-changed. An attachment only
    adds context, so refusing four good uploads over a fifth unreadable scan is friction with
    no safety behind it. What must never happen is the refusal going unmentioned.
    """
    c, iface = assist_client
    started = await _started(
        await c.post(
            "/api/v1/knowledge/mock_domain/assist",
            json={
                "question": "use these",
                "attachments": [
                    _upload("good.md", b"# a draft"),
                    _upload("nope.zip", b"PK\x03\x04"),
                    {"name": "broken.md", "content": "!!!not base64!!!"},
                ],
            },
        )
    )
    assert [a["name"] for a in started["attachments"]] == ["good.md"]
    assert len(started["attachment_errors"]) == 2
    joined = " ".join(started["attachment_errors"])
    assert "nope.zip" in joined and "broken.md" in joined
    body = await _await_session(c, "mock_domain", started["session"])
    assert body["status"] == "proposed", "one bad upload must not fail the run"


async def test_an_image_reaches_a_multimodal_endpoint_as_a_content_block(
    assist_client, packs
):
    c, iface = assist_client
    started = await _started(
        await c.post(
            "/api/v1/knowledge/mock_domain/assist",
            json={
                "question": "follow this layout",
                "attachments": [_upload("layout.png", b"\x89PNG\r\n\x1a\npixels")],
            },
        )
    )
    body = await _await_session(c, "mock_domain", started["session"])
    assert body["image_mode"] == "read"
    blocks = [
        b
        for messages in iface.llm_client.seen
        for m in messages
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "image_url"
    ]
    assert blocks, "the image must ride as a block, not as base64 in the prose"
    assert blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_an_endpoint_that_refuses_images_says_so_on_the_session(
    assist_client, packs
):
    """`text_only` is the whole reason the probe exists: the alternative is an assistant
    that answers about a diagram it never saw, inside a diff a human then approves."""
    c, iface = assist_client
    iface.llm_client.accept_images = False
    started = await _started(
        await c.post(
            "/api/v1/knowledge/mock_domain/assist",
            json={
                "question": "follow this layout",
                "attachments": [_upload("layout.png", b"\x89PNG\r\n\x1a\npixels")],
            },
        )
    )
    body = await _await_session(c, "mock_domain", started["session"])
    assert body["image_mode"] == "text_only"
    assert body["status"] == "proposed", "a refused image must not fail the run"
    # The exploration turn carries NO image block, and names the file in prose instead — so
    # the model is told the diagram exists and that it did not see it.
    sent = iface.llm_client.seen[-1]
    assert not any(isinstance(m.get("content"), list) for m in sent)
    prose = "".join(str(m.get("content")) for m in sent)
    assert "layout.png" in prose and "NOT" in prose


async def test_a_rejection_keeps_the_attachments_so_they_are_not_re_uploaded(
    assist_client, packs
):
    """A correction is usually ABOUT the attached diagram.

    Making the operator re-upload it to say "you read the wrong box" would be the most
    predictable friction on this surface.
    """
    c, iface = assist_client
    started = await _started(
        await c.post(
            "/api/v1/knowledge/mock_domain/assist",
            json={
                "question": "use this",
                "attachments": [_upload("spec.md", b"# the spec")],
            },
        )
    )
    await _await_session(c, "mock_domain", started["session"])
    retry = await _started(
        await c.post(
            f"/api/v1/knowledge/mock_domain/assist/{started['session']}/reject",
            json={"guidance": "wrong section"},
        )
    )
    assert [a["name"] for a in retry["attachments"]] == ["spec.md"]


async def test_the_upload_body_is_bounded_by_the_per_attachment_cap(
    assist_client, packs
):
    """The app-wide `client_max_size` is a ceiling; this is the policy.

    A request under 32 MB is still read, so the refusal has to come from the handler with a
    number in it rather than from aiohttp with a 413 and no explanation.
    """
    c, iface = assist_client
    oversize = b"\x89PNG" + b"z" * pack_attachments.MAX_IMAGE_BYTES
    started = await _started(
        await c.post(
            "/api/v1/knowledge/mock_domain/assist",
            json={
                "question": "use this",
                "attachments": [_upload("huge.png", oversize)],
            },
        )
    )
    assert started["attachments"] == []
    assert str(pack_attachments.MAX_IMAGE_BYTES) in started["attachment_errors"][0]
