"""Who is asking, and where their writes land — over a real server.

`test_identity.py` proves the resolver reads headers correctly and `test_user_overlay.py` proves
a three-way merge keeps both sides' work. Neither can prove the handlers route into them, which
is the failure that matters: a layer nothing writes to reads exactly like a layer that works,
and a caller whose edit went to the shared tree finds out when somebody else's run changes
behaviour.

The first test is the one to keep green above all the others. AFIR runs unchanged on a laptop, a
VM, an Azure App Service and a Databricks App — none of which forwards a caller identity — and on
those the whole per-caller model must be *unreachable*, not merely unused: no identity resolves,
so every caller is the single local admin and every write goes to the base exactly where it went
before any of this existed.
"""

import asyncio
import json
import shutil

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from src import config_store, report_delivery
from src.audit_journal import AuditJournal
from src.identity import (ADMIN, USER, IdentityResolver, LOCAL_IDENTITY,
                          NAME_HEADER, VALIDATED_HEADER, owner_scoped)
from src.user_secrets import (UserSecretStore, offered_names, personal_value,
                              set_secret_store)
from src.incident_input import IncidentInputInterface
from src.knowledge import pack_store
from src.notifications import EventEmitter
from src.pipeline_runner import JobManager, JobRunMode, StageDescriptor
from src.storage import LocalStorage
from src.user_overlay import CONFLICT, KNOWLEDGE_LAYER, MERGED, UserLayer
from src.utils.paths import REPO_ROOT
from src.utils.rate_limiter import AsyncRateLimiter
from tests.installed_packs import FIXTURE_PACK, installed_packs

PACK = FIXTURE_PACK
REAL_PACKS = REPO_ROOT / "knowledge"

#: A file every installed pack has, so nothing here names a domain.
CATALOG = "source_catalog.yaml"

#: The headers the driver proxy forwards on the browser path: a platform-asserted name with no
#: token behind it. Measured, not invented — see `docs/architecture/identity.md`.
AS_USER = {VALIDATED_HEADER: "true", NAME_HEADER: "someone@example.test"}
AS_OWNER = {VALIDATED_HEADER: "true", NAME_HEADER: "owner@example.test"}


def _config():
    return {
        "host": "127.0.0.1",
        "port": 0,
        "post_incident_endpoint": "/api/v1/incidents",
        "post_ir_endpoint": "/api/v1/ir",
        "rate_limit": {"requests": 1000, "per_seconds": 60},
    }


@pytest.fixture
def cfg_dir(tmp_path, monkeypatch):
    """A throwaway config directory, redirected the way `test_ui_server.py` does it."""
    root = tmp_path / "config"
    root.mkdir()
    (root / "main_config.yaml").write_text(
        "anomaly_detection:\n"
        "  threshold: 0.8\n"
        "  max_anomalies: 50\n"
    )
    (root / "llm_config.yaml").write_text("model: reasoning-endpoint\ntemperature: 0.2\n")
    monkeypatch.setattr(config_store, "config_dir", lambda: root)
    import src.incident_input as incident_input

    monkeypatch.setattr(incident_input, "config_dir", lambda: root)
    return root


@pytest.fixture
def packs(tmp_path, monkeypatch):
    """Writable copies of the installed packs, with the store pointed at them."""
    root = tmp_path / "knowledge"
    root.mkdir()
    for name in installed_packs():
        shutil.copytree(REAL_PACKS / name, root / name)
    monkeypatch.setattr(pack_store, "knowledge_pack_dir", lambda name: root / name)
    assert pack_store.packs_root() == root
    return root


@pytest.fixture
def store(tmp_path):
    return LocalStorage(root=tmp_path / "store")


def _resolver():
    """A resolver in the state a driver-proxy deployment is in: identity on, one named owner.

    `mode="on"` rather than `auto` so the test does not depend on the environment it runs in —
    which is the same reason the platform-neutrality test below injects nothing at all.
    """
    return IdentityResolver(mode="on", admin_users=["owner@example.test"])


async def _client(iface):
    iface.rate_limiter = AsyncRateLimiter(rate_limit=1000, time_period=1)
    client = TestClient(TestServer(iface.app))
    await client.start_server()
    return client


@pytest.fixture
async def client(cfg_dir, packs, store):
    """The interface with identity enforced and a durable store behind it."""
    iface = IncidentInputInterface(
        _config(),
        live_config={"knowledge": {"pack_dir": f"knowledge/{PACK}"}},
        storage=store,
        identity_resolver=_resolver(),
    )
    got = await _client(iface)
    yield got
    iface.rate_limiter.close()
    await got.close()


async def json_of(resp):
    assert resp.content_type == "application/json", await resp.text()
    return await resp.json()


# -- the platform-neutrality guarantee ------------------------------------


async def test_with_no_ingress_identity_every_write_goes_to_the_base(cfg_dir, packs, store):
    """A laptop, a VM, an App Service and a Databricks App, all in one assertion.

    Nothing is injected: the interface builds its own resolver from a config with no
    `identity:` block, exactly as `main()` does. If this ever fails, a deployment that never
    had users has started routing its operator's edits into a layer nobody reads.
    """
    iface = IncidentInputInterface(
        _config(),
        live_config={"knowledge": {"pack_dir": f"knowledge/{PACK}"}},
        storage=store,
    )
    client = await _client(iface)
    try:
        assert iface.identity_resolver.enabled is False
        who = await json_of(await client.get("/api/v1/whoami"))
        assert who["role"] == ADMIN and who["source"] == "local"

        body = await json_of(
            await client.put(
                "/api/v1/config",
                json={"updates": {"anomaly_detection.threshold": 0.42}},
            )
        )
        assert body.get("layer") is not True
        assert "0.42" in (cfg_dir / "main_config.yaml").read_text()

        before = pack_store.read_file(PACK, CATALOG)
        saved = await json_of(
            await client.put(
                f"/api/v1/knowledge/{PACK}/file?path={CATALOG}",
                json={"text": before["text"] + "\n# edited by the operator\n"},
            )
        )
        assert saved.get("layer") is not True and saved["restart_required"] is True
        assert "edited by the operator" in pack_store.read_file(PACK, CATALOG)["text"]
        # And nothing was written into anybody's per-caller namespace.
        assert [obj.key for obj in store.list_keys("users")] == []
    finally:
        iface.rate_limiter.close()
        await client.close()


async def test_the_local_identity_is_the_admin_and_owns_nothing(cfg_dir, packs, store):
    """The two halves of "single-operator": admin role, and no owner stamped on a run.

    Stamping an owner would move every job under `users/local/`, which a restart of an
    existing deployment would read as an empty queue.
    """
    from src.identity import owner_of, stamp_owner

    assert LOCAL_IDENTITY.role == ADMIN
    assert owner_of(stamp_owner({"description": "x"}, LOCAL_IDENTITY)) == ""


# -- config: an administrator edits the base ------------------------------


async def test_an_owner_patches_the_shared_file(client, cfg_dir):
    body = await json_of(
        await client.put(
            "/api/v1/config",
            headers=AS_OWNER,
            json={"updates": {"anomaly_detection.threshold": 0.5}},
        )
    )
    assert body.get("layer") is not True
    assert "0.5" in (cfg_dir / "main_config.yaml").read_text()


# -- config: everybody else edits their own layer -------------------------


async def test_a_users_patch_lands_in_their_layer_and_not_in_the_shared_file(client, cfg_dir):
    before = (cfg_dir / "main_config.yaml").read_text()
    body = await json_of(
        await client.put(
            "/api/v1/config",
            headers=AS_USER,
            json={"updates": {"anomaly_detection.threshold": 0.11}},
        )
    )
    assert body["layer"] is True and body["effect"] == "draft"
    assert body["note"], "a draft that does not say it is a draft reads as an apply"
    # A `live` field drafted rather than applied still needs no restart, and the three layered
    # writes have to agree about that: a caller reading the field on a whole-file save and no
    # field on a patch has to guess, and the guess that costs something is "restart and it lands".
    assert body["restart_required"] is False
    assert (cfg_dir / "main_config.yaml").read_text() == before


async def test_a_user_reads_their_own_draft_back(client):
    await client.put(
        "/api/v1/config",
        headers=AS_USER,
        json={"updates": {"anomaly_detection.threshold": 0.11}},
    )
    text = await (
        await client.get("/api/v1/config/main_config.yaml", headers=AS_USER)
    ).text()
    assert "0.11" in text
    # And the shared text is still reachable, which is what makes a diff possible.
    shared = await (
        await client.get(
            "/api/v1/config/main_config.yaml?base=1", headers=AS_USER
        )
    ).text()
    assert "0.11" not in shared


async def test_one_users_draft_is_invisible_to_another(client):
    await client.put(
        "/api/v1/config",
        headers=AS_USER,
        json={"updates": {"anomaly_detection.threshold": 0.11}},
    )
    other = {VALIDATED_HEADER: "true", NAME_HEADER: "third@example.test"}
    text = await (
        await client.get("/api/v1/config/main_config.yaml", headers=other)
    ).text()
    assert "0.11" not in text


async def test_a_whole_file_save_by_a_user_is_validated_before_it_is_drafted(client, cfg_dir):
    """The rules are about the TEXT, not about who wrote it.

    A draft allowed to hold YAML the base would refuse defers the refusal to whoever
    promotes it — who did not write the mistake and cannot see where it came from.
    """
    resp = await client.put(
        "/api/v1/config/main_config.yaml",
        headers=AS_USER,
        json={"text": "anomaly_detection: [not, a, mapping\n"},
    )
    assert resp.status == 400
    assert "yaml" in (await json_of(resp))["error"].lower()


async def test_an_import_by_a_user_is_all_or_nothing_into_the_layer(client, cfg_dir):
    before = (cfg_dir / "llm_config.yaml").read_text()
    resp = await client.post(
        "/api/v1/config/import",
        headers=AS_USER,
        json={"files": {"llm_config.yaml": "model: mine\n",
                        "nonexistent.yaml": "x: 1\n"}},
    )
    assert resp.status == 400
    assert (cfg_dir / "llm_config.yaml").read_text() == before
    text = await (
        await client.get("/api/v1/config/llm_config.yaml", headers=AS_USER)
    ).text()
    assert "mine" not in text, "a refused import must have written none of its files"


async def test_a_draft_needs_somewhere_durable_to_live(cfg_dir, packs):
    """With no store the refusal names the one route that still works: ask an administrator."""
    iface = IncidentInputInterface(
        _config(), storage=None, identity_resolver=_resolver()
    )
    client = await _client(iface)
    try:
        resp = await client.put(
            "/api/v1/config",
            headers=AS_USER,
            json={"updates": {"anomaly_detection.max_anomalies": 7}},
        )
        assert resp.status == 503
        assert "administrator" in (await json_of(resp))["error"]
    finally:
        iface.rate_limiter.close()
        await client.close()


# -- knowledge pack: the same split ---------------------------------------


async def test_a_users_pack_save_lands_in_their_layer(client, packs):
    before = pack_store.read_file(PACK, CATALOG)["text"]
    body = await json_of(
        await client.put(
            f"/api/v1/knowledge/{PACK}/file?path={CATALOG}",
            headers=AS_USER,
            json={"text": before + "\n# my own note\n"},
        )
    )
    assert body["layer"] is True and body["restart_required"] is False
    assert body["durable"] is True
    assert pack_store.read_file(PACK, CATALOG)["text"] == before


async def test_the_diagnostics_beside_a_draft_say_which_pack_they_describe(client, packs):
    """They are computed from disk, so beside a draft they are a verdict on the base."""
    before = pack_store.read_file(PACK, CATALOG)["text"]
    body = await json_of(
        await client.put(
            f"/api/v1/knowledge/{PACK}/file?path={CATALOG}",
            headers=AS_USER,
            json={"text": before + "\n# my own note\n"},
        )
    )
    assert body["validate_scope"] == "base"


async def test_a_user_reads_their_own_pack_draft_and_can_still_reach_the_shared_one(client):
    before = pack_store.read_file(PACK, CATALOG)["text"]
    await client.put(
        f"/api/v1/knowledge/{PACK}/file?path={CATALOG}",
        headers=AS_USER,
        json={"text": before + "\n# my own note\n"},
    )
    mine = await json_of(
        await client.get(f"/api/v1/knowledge/{PACK}/file?path={CATALOG}", headers=AS_USER)
    )
    assert "my own note" in mine["text"] and mine["layer"] is True
    # The concurrency token must describe what was served, or the 409 path is disabled.
    assert mine["sha256"] == pack_store.sha256_text(mine["text"])
    shared = await json_of(
        await client.get(
            f"/api/v1/knowledge/{PACK}/file?path={CATALOG}&base=1", headers=AS_USER
        )
    )
    assert "my own note" not in shared["text"]


async def test_a_ranged_save_splices_against_the_callers_own_text(client):
    """A line number means what the editor showed them, not what the base holds."""
    base = pack_store.read_file(PACK, CATALOG)["text"]
    await client.put(
        f"/api/v1/knowledge/{PACK}/file?path={CATALOG}",
        headers=AS_USER,
        json={"text": "# first draft line\n" + base},
    )
    resp = await client.put(
        f"/api/v1/knowledge/{PACK}/file?path={CATALOG}",
        headers=AS_USER,
        json={"text": "# replaced line one\n", "start_line": 1, "end_line": 1},
    )
    body = await json_of(resp)
    assert resp.status == 200, body
    mine = await json_of(
        await client.get(f"/api/v1/knowledge/{PACK}/file?path={CATALOG}", headers=AS_USER)
    )
    assert mine["text"].startswith("# replaced line one\n")
    assert "# first draft line" not in mine["text"]


async def test_a_users_draft_cannot_hold_yaml_the_pack_would_refuse(client):
    resp = await client.put(
        f"/api/v1/knowledge/{PACK}/file?path={CATALOG}",
        headers=AS_USER,
        json={"text": "sources: [unterminated\n"},
    )
    assert resp.status == 400


async def test_a_file_a_user_creates_exists_only_for_them_and_is_listed(client, packs):
    rel = "rulesets/my_draft.yaml"
    resp = await client.post(
        f"/api/v1/knowledge/{PACK}/file?path={rel}",
        headers=AS_USER,
        json={"text": "rulesets:\n  mine:\n    conditions: []\n"},
    )
    body = await json_of(resp)
    assert resp.status == 200, body
    assert body["layer"] is True and body["in_base"] is False
    assert not (packs / PACK / rel).exists()

    tree = await json_of(
        await client.get(f"/api/v1/knowledge/{PACK}/tree", headers=AS_USER)
    )
    paths = {node["path"] for node in tree["nodes"]}
    assert rel in paths, "a draft nobody can find in the browser cannot be opened again"
    assert "rulesets" in paths, "its parent directory has to be there for the indent"
    # And it is not in anybody else's tree.
    other = await json_of(await client.get(f"/api/v1/knowledge/{PACK}/tree", headers=AS_OWNER))
    assert rel not in {node["path"] for node in other["nodes"]}


async def test_a_user_cannot_delete_a_shared_file(client, packs):
    resp = await client.delete(
        f"/api/v1/knowledge/{PACK}/file?path={CATALOG}&confirm=1", headers=AS_USER
    )
    assert resp.status == 403
    assert (packs / PACK / CATALOG).is_file()


async def test_a_user_discards_their_own_draft_and_sees_the_shared_file_again(client):
    base = pack_store.read_file(PACK, CATALOG)["text"]
    await client.put(
        f"/api/v1/knowledge/{PACK}/file?path={CATALOG}",
        headers=AS_USER,
        json={"text": base + "\n# mine\n"},
    )
    body = await json_of(
        await client.delete(
            f"/api/v1/knowledge/{PACK}/file?path={CATALOG}&confirm=1", headers=AS_USER
        )
    )
    assert body["dropped"] is True and body["restored_to_base"] is True
    served = await json_of(
        await client.get(f"/api/v1/knowledge/{PACK}/file?path={CATALOG}", headers=AS_USER)
    )
    assert served["text"] == base


# -- what an administrator's release does to a draft ----------------------


async def test_an_owners_pack_release_merges_a_users_draft_and_reports_it(client, store):
    """The requirement in one test: the edit persists where there is no conflict.

    Reported on the administrator's own response, because a release that silently conflicted
    with somebody's draft is a release nobody knows to look at.
    """
    rel = "concepts/note.md"
    (pack_store.pack_dir(PACK) / "concepts").mkdir(exist_ok=True)
    (pack_store.pack_dir(PACK) / rel).write_text("---\nid: note\n---\n\nfirst\n\nlast\n")

    await client.put(
        f"/api/v1/knowledge/{PACK}/file?path={rel}",
        headers=AS_USER,
        json={"text": "---\nid: note\n---\n\nMINE\n\nlast\n"},
    )
    body = await json_of(
        await client.put(
            f"/api/v1/knowledge/{PACK}/file?path={rel}",
            headers=AS_OWNER,
            json={"text": "---\nid: note\n---\n\nfirst\n\nTHEIRS\n"},
        )
    )
    states = [state for files in body["rebased"].values() for state in files.values()]
    assert states == [MERGED], body["rebased"]
    merged = UserLayer(store, "someone@example.test", KNOWLEDGE_LAYER).read(f"{PACK}/{rel}")
    assert "MINE" in merged and "THEIRS" in merged


async def test_a_release_over_the_same_lines_conflicts_without_losing_the_draft(client, store):
    rel = "concepts/note.md"
    (pack_store.pack_dir(PACK) / "concepts").mkdir(exist_ok=True)
    (pack_store.pack_dir(PACK) / rel).write_text("---\nid: note\n---\n\nfirst\n")

    await client.put(
        f"/api/v1/knowledge/{PACK}/file?path={rel}",
        headers=AS_USER,
        json={"text": "---\nid: note\n---\n\nMINE\n"},
    )
    body = await json_of(
        await client.put(
            f"/api/v1/knowledge/{PACK}/file?path={rel}",
            headers=AS_OWNER,
            json={"text": "---\nid: note\n---\n\nTHEIRS\n"},
        )
    )
    states = [state for files in body["rebased"].values() for state in files.values()]
    assert states == [CONFLICT]
    kept = UserLayer(store, "someone@example.test", KNOWLEDGE_LAYER).read(f"{PACK}/{rel}")
    assert "MINE" in kept and "THEIRS" in kept


async def test_an_owners_config_release_rebases_the_config_layers(client, cfg_dir):
    await client.put(
        "/api/v1/config",
        headers=AS_USER,
        json={"updates": {"anomaly_detection.threshold": 0.11}},
    )
    body = await json_of(
        await client.put(
            "/api/v1/config",
            headers=AS_OWNER,
            json={"updates": {"anomaly_detection.max_anomalies": 7}},
        )
    )
    assert "rebased" in body


# -- the three writes with no draft to be ----------------------------------
#
# Every other write in this file layers. These three cannot, and each for its own reason, so
# each is refused rather than quietly redirected — a 200 on a write that went nowhere is the
# failure this whole file exists to catch, in the one place a layer is not the answer.


async def test_a_user_cannot_push_a_release_over_the_shared_pack(client, packs):
    """The import route IS the release verb, so it is the one a reader must not reach.

    The per-file routes beside it accept the same caller's edit into a draft, which is what
    makes this a refusal about the DESTINATION rather than about the person.
    """
    before = (packs / PACK / CATALOG).read_bytes()
    refused = await client.post(
        f"/api/v1/knowledge/{PACK}/import",
        headers=AS_USER,
        json={"files": {CATALOG: "sources: []\n"}},
    )
    assert refused.status == 403
    body = await json_of(refused)
    assert body["role"] == USER and "/api/v1/whoami/elevate" in body["remedy"]
    assert (packs / PACK / CATALOG).read_bytes() == before

    landed = await client.post(
        f"/api/v1/knowledge/{PACK}/import",
        headers=AS_OWNER,
        json={"files": {CATALOG: "sources: []\n"}},
    )
    assert landed.status == 200
    assert (packs / PACK / CATALOG).read_bytes() != before


async def test_a_user_cannot_scaffold_a_pack_into_the_shared_tree(client, packs):
    """A new pack has no base, so there is nowhere for a draft of one to sit."""
    refused = await client.post(
        "/api/v1/knowledge/scaffold",
        headers=AS_USER,
        json={"name": "mine", "vocabulary": ["widget"]},
    )
    assert refused.status == 403
    assert not (packs / "mine").exists()


async def test_a_user_may_ask_the_assistant_and_may_not_apply_its_plan(client):
    """The refusal is on `apply` alone, and it precedes the session lookup.

    A user reaching for the assistant is reaching for help, so the read side stays open; only
    the write is refused. Asserted through a session id that does not exist, because that is
    what tells the two orderings apart: the guard first answers 403 for everyone who may not
    write, while a session check first would answer 404 and leak that the ordering is wrong
    only once somebody had a real session.
    """
    assert (await client.get(f"/api/v1/knowledge/{PACK}/assist", headers=AS_USER)).status == 200

    refused = await client.post(
        f"/api/v1/knowledge/{PACK}/assist/no-such-session/apply", headers=AS_USER, json={}
    )
    assert refused.status == 403
    admin = await client.post(
        f"/api/v1/knowledge/{PACK}/assist/no-such-session/apply", headers=AS_OWNER, json={}
    )
    assert admin.status == 404


# -- who the caller is told they are --------------------------------------


async def test_whoami_explains_the_role_it_gave(client):
    body = await json_of(await client.get("/api/v1/whoami", headers=AS_USER))
    assert body["role"] == USER and body["role_reason"]
    assert body["edits_the_base"] is False
    owner = await json_of(await client.get("/api/v1/whoami", headers=AS_OWNER))
    assert owner["role"] == ADMIN and owner["edits_the_base"] is True


async def test_a_request_with_no_validated_identity_is_refused_when_identity_is_on(client):
    resp = await client.get("/api/v1/config")
    assert resp.status == 403
    assert "driver proxy" in (await json_of(resp))["error"]


async def test_the_overlay_endpoint_lists_both_trees(client):
    await client.put(
        "/api/v1/config",
        headers=AS_USER,
        json={"updates": {"anomaly_detection.max_anomalies": 7}},
    )
    body = await json_of(await client.get("/api/v1/overlay", headers=AS_USER))
    assert [row["path"] for row in body["config"]] == ["main_config.yaml"]
    assert body["knowledge"] == []


async def test_dropping_a_config_override_restores_the_shared_view(client, cfg_dir):
    await client.put(
        "/api/v1/config",
        headers=AS_USER,
        json={"updates": {"anomaly_detection.max_anomalies": 7}},
    )
    resp = await client.delete(
        "/api/v1/overlay/config?path=main_config.yaml", headers=AS_USER
    )
    assert resp.status == 200, await resp.text()
    text = await (
        await client.get("/api/v1/config/main_config.yaml", headers=AS_USER)
    ).text()
    assert "max_anomalies: 7" not in text
    body = await json_of(await client.get("/api/v1/overlay", headers=AS_USER))
    assert body["config"] == [] and body["knowledge"] == []


# -- whose run is it -------------------------------------------------------
#
# A run is owned by whoever submitted it, and the stamp rides on the incident so it survives
# `export_job` / `import_job` and a restart. Two rules are asserted here rather than in
# `test_pipeline_runner.py`, because both are properties of the HTTP surface: a run belongs to
# the caller the *request* resolved to, and somebody else's run reads as ABSENT.


def _echo_stage(name):
    async def run(ctx):
        return name

    return StageDescriptor(name, run, name)


@pytest.fixture
async def jobs_client(cfg_dir, packs, store):
    """The interface with identity enforced and a real `JobManager` behind it.

    A real manager rather than a stub, because the ownership stamp passes through
    `launch_fn` -> `create_job` -> `list_jobs`, and a stub returning rows with an `owner` key
    would assert the filter while letting the stamp be broken.

    The one stage is named for a codec-free key: `_OUTPUT_CODECS` decodes `understanding` into
    a Pydantic model, so an echo stage returning a string would make `export_job` raise — and
    the export is one of the routes asserted below.
    """
    emitter = EventEmitter()
    jm = JobManager([_echo_stage("log_retrieval")], emitter)
    emitter.set_job_manager(jm)
    jm._test_tasks = []

    def launch_fn(incident, mode="auto"):
        job = jm.create_job(incident, run_mode=JobRunMode(str(mode).lower()))
        jm._test_tasks.append(asyncio.ensure_future(jm.run_job(job)))
        return job

    iface = IncidentInputInterface(
        _config(),
        live_config={"knowledge": {"pack_dir": f"knowledge/{PACK}"}},
        storage=store,
        identity_resolver=_resolver(),
        job_manager=jm,
        launch_fn=launch_fn,
    )
    got = await _client(iface)
    yield jm, got
    for task in jm._test_tasks:
        task.cancel()
    iface.rate_limiter.close()
    await got.close()


async def _submit(client, headers, description="a run"):
    body = await json_of(
        await client.post("/api/v1/jobs", headers=headers, json={"description": description})
    )
    return body["job_id"]


async def test_a_run_is_listed_for_its_own_caller_and_for_an_administrator(jobs_client):
    _jm, client = jobs_client
    job_id = await _submit(client, AS_USER)

    mine = await json_of(await client.get("/api/v1/jobs", headers=AS_USER))
    assert [row["job_id"] for row in mine["jobs"]] == [job_id]
    assert mine["jobs"][0]["owner_name"] == AS_USER[NAME_HEADER]

    stranger = {VALIDATED_HEADER: "true", NAME_HEADER: "third@example.test"}
    assert (await json_of(await client.get("/api/v1/jobs", headers=stranger)))["jobs"] == []

    everything = await json_of(await client.get("/api/v1/jobs", headers=AS_OWNER))
    assert [row["job_id"] for row in everything["jobs"]] == [job_id]


async def test_another_callers_run_reads_as_absent_and_not_as_forbidden(jobs_client):
    """404, because a 403 confirms the id exists — the one thing a caller with no claim on it
    should not learn. Asserted on the status and not on the body: an error message naming the
    owner would leak the same fact the status code was chosen to withhold.
    """
    _jm, client = jobs_client
    job_id = await _submit(client, AS_USER)
    stranger = {VALIDATED_HEADER: "true", NAME_HEADER: "third@example.test"}

    hidden = await client.get(f"/api/v1/jobs/{job_id}", headers=stranger)
    assert hidden.status == 404
    assert AS_USER[NAME_HEADER] not in await hidden.text()
    # Same id, same instant, two callers: the owner and an administrator both see it.
    assert (await client.get(f"/api/v1/jobs/{job_id}", headers=AS_USER)).status == 200
    assert (await client.get(f"/api/v1/jobs/{job_id}", headers=AS_OWNER)).status == 200


async def test_a_run_from_before_ownership_existed_is_administrator_only(jobs_client):
    """An unowned run is not everybody's run.

    Every job submitted before this seam existed carries no owner, as does every run of a
    deployment that resolves no identity. Attributing one to whoever asks would be inventing a
    claim, so it stays visible to an administrator alone — who is also the only caller a
    single-operator deployment has.
    """
    jm, client = jobs_client
    legacy = jm.create_job({"id": "INC-legacy", "description": "before ownership"})

    assert (await json_of(await client.get("/api/v1/jobs", headers=AS_USER)))["jobs"] == []
    assert (await client.get(f"/api/v1/jobs/{legacy.job_id}", headers=AS_USER)).status == 404
    seen = await json_of(await client.get("/api/v1/jobs", headers=AS_OWNER))
    assert [row["job_id"] for row in seen["jobs"]] == [legacy.job_id]
    assert seen["jobs"][0]["owner"] == ""


async def test_a_batch_is_scoped_like_the_jobs_it_labels(jobs_client):
    """A batch is a label and not a second record, so it inherits its jobs' owner.

    All three batch routes are asserted, because each reaches the check by its own path: the
    list filters rows, and `GET`/`cancel` raise the same KeyError a missing batch raises — a
    cancel that refused with a 403 would let one caller stop another's runs by guessing.
    """
    _jm, client = jobs_client
    created = await json_of(
        await client.post(
            "/api/v1/batches", headers=AS_USER, json={"incidents": ["one", "two"]}
        )
    )
    batch_id = created["batch_id"]
    stranger = {VALIDATED_HEADER: "true", NAME_HEADER: "third@example.test"}

    assert (await json_of(await client.get("/api/v1/batches", headers=stranger)))["batches"] == []
    assert (await client.get(f"/api/v1/batches/{batch_id}", headers=stranger)).status == 404
    assert (
        await client.post(f"/api/v1/batches/{batch_id}/cancel", headers=stranger)
    ).status == 404

    mine = await json_of(await client.get(f"/api/v1/batches/{batch_id}", headers=AS_USER))
    assert mine["total"] == 2
    assert (await client.get(f"/api/v1/batches/{batch_id}", headers=AS_OWNER)).status == 200


async def test_every_route_that_NAMES_a_run_takes_the_ownership_funnel(jobs_client):
    """One test over all of them, because the funnel is the guarantee and a route that
    reaches for ``job_manager`` directly is outside it whatever its own checks say.

    They are asserted together and not one per test because they fail as a class: each was
    written before ownership existed and each reads the id straight out of ``match_info``.
    The three writes are the ones that cost something — a cancel is not undoable, a gate
    decision moves somebody else's run past a human, and an override is recorded as *their*
    intervention on a report that then has to be read as hand-edited. The two reads are the
    widest on the router: the export doc is the whole run, and the event stream replays it.
    """
    _jm, client = jobs_client
    job_id = await _submit(client, AS_USER)
    stranger = {VALIDATED_HEADER: "true", NAME_HEADER: "third@example.test"}

    refused = [
        await client.get(f"/api/v1/jobs/{job_id}/export", headers=stranger),
        await client.get(f"/api/v1/jobs/{job_id}/events", headers=stranger),
        await client.post(
            f"/api/v1/jobs/{job_id}/control", headers=stranger, json={"action": "cancel"}
        ),
        await client.post(
            f"/api/v1/jobs/{job_id}/gate", headers=stranger, json={"action": "proceed"}
        ),
        await client.post(
            f"/api/v1/jobs/{job_id}/outputs/log_retrieval",
            headers=stranger,
            json={"value": "rewritten"},
        ),
    ]
    for r in refused:
        assert r.status == 404, f"{r.url.path} answered {r.status}"
        assert AS_USER[NAME_HEADER] not in await r.text()

    # The owner still reaches every one of them, or the funnel is a wall rather than a filter.
    assert (await client.get(f"/api/v1/jobs/{job_id}/export", headers=AS_USER)).status == 200
    assert (await client.get(f"/api/v1/jobs/{job_id}/export", headers=AS_OWNER)).status == 200


async def test_an_imported_run_belongs_to_whoever_imported_it(jobs_client):
    """An export from another deployment, or from before ownership existed, carries no owner.

    Left that way it would be visible to an administrator alone — including from the importer,
    who would watch their own import vanish. So the importer is stamped; but an export that
    already NAMES an owner keeps it, because import is a restore and not a transfer of claim.
    """
    _jm, client = jobs_client
    job_id = await _submit(client, AS_USER)
    doc = await json_of(await client.get(f"/api/v1/jobs/{job_id}/export", headers=AS_USER))

    orphan = json.loads(json.dumps(doc))
    orphan["incident"].pop("_owner", None)
    orphan["incident"].pop("owner", None)
    for key in [k for k in orphan["incident"] if "owner" in k.lower()]:
        orphan["incident"].pop(key)
    stranger = {VALIDATED_HEADER: "true", NAME_HEADER: "third@example.test"}
    made = await json_of(await client.post("/api/v1/jobs/import", headers=stranger, json=orphan))
    assert (
        await client.get(f"/api/v1/jobs/{made['job_id']}", headers=stranger)
    ).status == 200, "the importer cannot see their own import"
    assert (await client.get(f"/api/v1/jobs/{made['job_id']}", headers=AS_USER)).status == 404

    # The owned doc is restored to its own owner and not to the caller who carried it.
    kept = await json_of(await client.post("/api/v1/jobs/import", headers=stranger, json=doc))
    assert (await client.get(f"/api/v1/jobs/{kept['job_id']}", headers=AS_USER)).status == 200
    assert (await client.get(f"/api/v1/jobs/{kept['job_id']}", headers=stranger)).status == 404


# -- whose artifacts are they ---------------------------------------------
#
# A run's report and evidence are WRITTEN through `owner_scoped`, so they land under
# `users/<segment>/`; reading them back is a separate decision, which is why every
# `report_delivery` reader takes an `owners` argument. `test_ui_server.py` proves those
# resolve. What only a server can prove is that a handler passes the right segments — and
# both halves of the live defect were exactly there: every owned report answered 404 while
# `/artifacts` beside it reported the same file present with a byte count, and the finished
# list matched a BASENAME across every subtree, so it carried other callers' incident ids.


@pytest.fixture
def exports(store):
    """`report_delivery` pointed at the same store, at the prefix `main()` gives it."""
    from src.storage import PrefixedStorage

    view = PrefixedStorage(store, "exports")
    report_delivery.set_storage(view)
    yield view
    report_delivery.set_storage(None)


def _publish(exports, incident, body="# a finished report\n"):
    """Write a finished run's artifacts exactly where the pipeline writes them.

    Through `owner_scoped` and not to a path this test spells, or it would assert its own
    idea of the layout rather than the write side's.
    """
    view = owner_scoped(exports, incident)
    ident = incident["id"]
    view.put_text(report_delivery.report_key(ident, "md"), body)
    view.put_text(
        report_delivery.evidence_key(ident, "raw"), '{"a_source": [{"row": 1}]}\n'
    )
    return ident


async def test_an_owned_runs_artifacts_are_served_from_the_view_they_were_written_to(
    jobs_client, exports, store
):
    """The live defect, in the direction that was broken.

    All three routes are asserted together because they failed apart: the two readers 404'd
    on the root-relative key while the inventory kept working by accident — it matched a
    basename over a recursive walk — so the page drew a download button, with a byte count,
    for a link that could not resolve. Disagreement between them is the symptom to catch.
    """
    jm, client = jobs_client
    job_id = await _submit(client, AS_USER)
    ident = _publish(exports, jm.hydrate(job_id).incident)

    # Written where the read side has to look for it, and nowhere else.
    assert report_delivery._backend().get_text(
        report_delivery.report_key(ident, "md")
    ) is None
    assert [obj.key for obj in store.list_keys("exports/users")] != []

    md = await client.get(f"/api/v1/jobs/{job_id}/report?format=md", headers=AS_USER)
    assert md.status == 200 and "a finished report" in await md.text()
    outline = await json_of(
        await client.get(f"/api/v1/jobs/{job_id}/evidence", headers=AS_USER)
    )
    assert outline["total_rows"] == 1
    inventory = await json_of(
        await client.get(f"/api/v1/jobs/{job_id}/artifacts", headers=AS_USER)
    )
    assert inventory["artifacts"]["report_md"] == {
        "exists": True,
        "bytes": len("# a finished report\n"),
        "filename": report_delivery.report_key(ident, "md"),
    }

    # An administrator reading somebody else's finished run gets THAT run's segment, which
    # is why the job-scoped routes pass the run's owner and not the caller's own.
    assert (
        await client.get(f"/api/v1/jobs/{job_id}/report?format=md", headers=AS_OWNER)
    ).status == 200


async def test_a_stranger_reaches_none_of_the_three_artifact_routes(jobs_client, exports):
    """404 on all three, and the reason is the funnel rather than the scope.

    `_lookup_job` refuses first, so an artifact route never gets as far as naming a segment
    — asserted here because passing the *caller's* segment instead of the run's would answer
    404 too, for the wrong reason, and would then serve an admin nothing.
    """
    jm, client = jobs_client
    job_id = await _submit(client, AS_USER)
    _publish(exports, jm.hydrate(job_id).incident)
    stranger = {VALIDATED_HEADER: "true", NAME_HEADER: "third@example.test"}

    for path in ("report?format=md", "evidence", "artifacts"):
        r = await client.get(f"/api/v1/jobs/{job_id}/{path}", headers=stranger)
        assert r.status == 404, f"{r.url.path} answered {r.status}"
        assert AS_USER[NAME_HEADER] not in await r.text()


async def test_an_incident_keyed_read_offers_only_the_callers_own_subtree(
    jobs_client, exports
):
    """The route family with no run in hand, so the owner scope IS the whole check.

    A job-scoped URL is refused by `_lookup_job`; this one carries an id and nothing else,
    so a caller who learns another caller's incident id must still get nothing — while an
    administrator, who may already list every row, reaches both.
    """
    jm, client = jobs_client
    stranger = {VALIDATED_HEADER: "true", NAME_HEADER: "third@example.test"}
    mine = _publish(exports, jm.hydrate(await _submit(client, AS_USER)).incident)
    theirs = _publish(exports, jm.hydrate(await _submit(client, stranger)).incident)
    assert mine != theirs

    for who, own, other in ((AS_USER, mine, theirs), (stranger, theirs, mine)):
        assert (
            await client.get(f"/api/v1/incidents/{own}/report?format=md", headers=who)
        ).status == 200
        assert (
            await client.get(f"/api/v1/incidents/{other}/report?format=md", headers=who)
        ).status == 404
        hidden = await json_of(
            await client.get(f"/api/v1/incidents/{other}/artifacts", headers=who)
        )
        assert hidden["artifacts"]["report_md"]["exists"] is False

    for ident in (mine, theirs):
        assert (
            await client.get(
                f"/api/v1/incidents/{ident}/report?format=md", headers=AS_OWNER
            )
        ).status == 200


async def test_the_finished_list_carries_the_shared_root_and_no_other_callers_rows(
    jobs_client, exports
):
    """The leak half of the same defect: a row here is an id, and an id is the one thing a
    caller with no claim on a run must not learn — the fact `_lookup_job` answers 404 for.

    The unowned row is in the fixture because it is not an edge case: it is where a
    deployment that resolves no identity writes, and where every run from before ownership
    existed still lives. It must stay visible to everyone, or the fix trades a leak for a
    disappearance.
    """
    jm, client = jobs_client
    stranger = {VALIDATED_HEADER: "true", NAME_HEADER: "third@example.test"}
    mine = _publish(exports, jm.hydrate(await _submit(client, AS_USER)).incident)
    theirs = _publish(exports, jm.hydrate(await _submit(client, stranger)).incident)
    shared = _publish(exports, {"id": "INC-before-ownership"})

    async def listed(headers):
        body = await json_of(await client.get("/api/v1/incidents", headers=headers))
        return {row["incident_id"] for row in body["incidents"]}

    assert await listed(AS_USER) == {shared, mine}
    assert await listed(stranger) == {shared, theirs}
    assert await listed(AS_OWNER) == {shared, mine, theirs}


# -- proving membership the browser path cannot forward --------------------


async def test_a_browser_caller_elevates_with_their_own_token(cfg_dir, packs, store):
    """The browser path forwards a name and no credential, so an owner arrives as a user.

    Elevation is how they show otherwise. The validator is injected — the point is that the
    handler routes into it and that the *new* role is what the response and the next request
    both report, not that SCIM works.
    """
    resolver = IdentityResolver(
        mode="on",
        admin_groups=["platform-owners"],
        host="https://example.test",
        validator=lambda token: (
            {"id": "42", "userName": AS_USER[NAME_HEADER], "groups": [{"display": "platform-owners"}]}
            if token == "good"
            else None
        ),
    )
    iface = IncidentInputInterface(
        _config(),
        live_config={"knowledge": {"pack_dir": f"knowledge/{PACK}"}},
        storage=store,
        identity_resolver=resolver,
    )
    client = await _client(iface)
    try:
        before = await json_of(await client.get("/api/v1/whoami", headers=AS_USER))
        assert before["role"] == USER and before["can_elevate"] is True

        refused = await client.post(
            "/api/v1/whoami/elevate", headers=AS_USER, json={"token": "wrong"}
        )
        assert refused.status == 403
        still = await json_of(await client.get("/api/v1/whoami", headers=AS_USER))
        assert still["role"] == USER, "a rejected token must not grant anything"

        accepted = await json_of(
            await client.post(
                "/api/v1/whoami/elevate", headers=AS_USER, json={"token": "good"}
            )
        )
        assert accepted["accepted"] is True and accepted["role"] == ADMIN
        # And it sticks for the requests that follow, which is the whole point: the next pack
        # save has to land on the base.
        after = await json_of(await client.get("/api/v1/whoami", headers=AS_USER))
        assert after["role"] == ADMIN and after["edits_the_base"] is True
    finally:
        iface.rate_limiter.close()
        await client.close()


async def test_elevation_is_refused_where_it_is_switched_off(cfg_dir, packs, store):
    """`allow_self_elevation: false` leaves the named-users list as the only route.

    Asserted through the endpoint rather than on the resolver, because the flag is worth
    nothing if the handler answers 200 and simply ignores it.
    """
    resolver = IdentityResolver(
        mode="on",
        admin_groups=["platform-owners"],
        host="https://example.test",
        allow_elevation=False,
        validator=lambda token: {
            "id": "42",
            "userName": AS_USER[NAME_HEADER],
            "groups": [{"display": "platform-owners"}],
        },
    )
    iface = IncidentInputInterface(
        _config(),
        live_config={"knowledge": {"pack_dir": f"knowledge/{PACK}"}},
        storage=store,
        identity_resolver=resolver,
    )
    client = await _client(iface)
    try:
        assert (await json_of(await client.get("/api/v1/whoami", headers=AS_USER)))[
            "can_elevate"
        ] is False
        resp = await client.post(
            "/api/v1/whoami/elevate", headers=AS_USER, json={"token": "good"}
        )
        assert resp.status == 403
        assert (await json_of(await client.get("/api/v1/whoami", headers=AS_USER)))["role"] == USER
    finally:
        iface.rate_limiter.close()
        await client.close()


# -- the access journal ----------------------------------------------------
#
# `test_audit_journal.py` proves the journal records and reads back what it is told. It cannot
# prove the HTTP layer tells it anything, which is the failure that matters: a deployment
# recording nothing looks exactly like one nobody uses. So these go through the middleware —
# the only place that sees a caller at all, and the only place that sees the two events no
# handler ever runs for (a refusal at the door, and a page load by someone who then leaves).


@pytest.fixture
async def audited(cfg_dir, packs, store):
    """The interface with identity enforced AND a journal wired behind it."""
    journal = AuditJournal(storage=store, enabled=True, flush_seconds=3600)
    iface = IncidentInputInterface(
        _config(),
        live_config={"knowledge": {"pack_dir": f"knowledge/{PACK}"}},
        storage=store,
        identity_resolver=_resolver(),
        audit_journal=journal,
    )
    got = await _client(iface)
    yield journal, got
    iface.rate_limiter.close()
    await got.close()


def _journalled(journal):
    """Everything recorded so far, read back the way an operator reads it — through the store.

    Flushed first because entries are buffered: asserting on the buffer would pass for a
    journal whose sink never worked.
    """
    journal.flush_now()
    return journal.tail(limit=500)


async def test_a_request_is_journalled_with_the_caller_the_ingress_named(audited):
    journal, client = audited
    assert (await client.get("/api/v1/whoami", headers=AS_USER)).status == 200
    (entry,) = [e for e in _journalled(journal) if e.get("path") == "/api/v1/whoami"]
    assert entry["kind"] == "request"
    assert entry["user"] == AS_USER[NAME_HEADER]
    assert entry["status"] == 200 and entry["auth"] == "header"
    assert entry["ms"] >= 0


async def test_a_refusal_at_the_door_is_journalled_with_its_reason(audited):
    """The handler never runs, so nothing but the middleware can record this — and a caller
    reaching AFIR by a URL that strips the identity headers sees only a bare 403."""
    journal, client = audited
    assert (await client.get("/api/v1/jobs")).status == 403
    (entry,) = [e for e in _journalled(journal) if e["kind"] == "refused"]
    assert entry["status"] == 403 and entry["path"] == "/api/v1/jobs"
    assert entry["detail"]


async def test_a_handler_that_raises_is_still_journalled(audited):
    """A 404 or a 500 is a use of the app; recording only the successes reports a deployment
    that never has a problem."""
    journal, client = audited
    assert (await client.get("/api/v1/knowledge/no-such-pack/file?path=x", headers=AS_USER)).status >= 400
    statuses = [e["status"] for e in _journalled(journal)
                if "no-such-pack" in str(e.get("path"))]
    assert statuses and statuses[0] >= 400


async def test_a_config_change_names_who_made_it_and_where_it_landed(audited, cfg_dir):
    """Two writes of the same key, by two callers, with two different effects. Before this the
    only record was a filename and a key count."""
    journal, client = audited
    await client.put("/api/v1/config", headers=AS_OWNER,
                     json={"updates": {"anomaly_detection.threshold": 0.55}})
    await client.put("/api/v1/config", headers=AS_USER,
                     json={"updates": {"anomaly_detection.threshold": 0.66}})
    changes = [e for e in _journalled(journal) if e["kind"] == "config_change"]
    by_user = {e["user"]: e for e in changes}
    assert by_user[AS_OWNER[NAME_HEADER]]["target"] == "base"
    assert by_user[AS_USER[NAME_HEADER]]["target"] == "layer"
    # Before AND after: "somebody lowered the threshold" and "somebody set it to 0.05" are
    # different findings, and only one of them is actionable.
    assert by_user[AS_OWNER[NAME_HEADER]]["changed"] == [
        {"path": "anomaly_detection.threshold", "from": "0.8", "to": "0.55",
         "applies": "live"}
    ]


async def test_the_journal_is_readable_only_by_an_administrator(audited):
    """It names every other caller, which is exactly the question a plain user has no
    business asking."""
    journal, client = audited
    journal.record("request", None, path="/x")
    journal.flush_now()
    refused = await client.get("/api/v1/audit", headers=AS_USER)
    assert refused.status == 403
    assert "administrator" in (await json_of(refused))["error"]

    body = await json_of(await client.get("/api/v1/audit?limit=5", headers=AS_OWNER))
    assert body["limit"] == 5
    assert body["journal"]["enabled"] is True
    assert body["count"] == len(body["entries"])


async def test_the_journal_endpoint_filters_by_user_and_bounds_the_answer(audited):
    journal, client = audited
    for _ in range(3):
        await client.get("/api/v1/whoami", headers=AS_USER)
    await client.get("/api/v1/whoami", headers=AS_OWNER)
    journal.flush_now()
    mine = await json_of(
        await client.get(f"/api/v1/audit?user={AS_USER[NAME_HEADER]}", headers=AS_OWNER)
    )
    assert mine["count"] == 3
    assert {e["user"] for e in mine["entries"]} == {AS_USER[NAME_HEADER]}


async def test_a_page_load_is_journalled_unattributed_where_an_identity_is_enforced(audited):
    """The page has to load before it can ask who the caller is, so the HANDLER runs as the
    local operator — but nobody is `local` on an enforced deployment, and an anonymous browser
    hit recorded under that name is a census naming a caller who does not exist."""
    journal, client = audited
    assert (await client.get("/afir")).status == 200
    (entry,) = [e for e in _journalled(journal) if e.get("path") == "/afir"]
    assert entry["auth"] == "unresolved"
    assert "user" not in entry


async def test_the_same_page_load_is_the_local_operator_where_nothing_is_enforced(cfg_dir, packs, store):
    """The other direction, and the one that keeps the fix from being a deletion: on a laptop
    every caller really IS the single local operator, and dropping the attribution there would
    turn a complete record into an anonymous one."""
    journal = AuditJournal(storage=store, enabled=True, flush_seconds=3600)
    iface = IncidentInputInterface(
        _config(),
        live_config={"knowledge": {"pack_dir": f"knowledge/{PACK}"}},
        storage=store,
        identity_resolver=IdentityResolver(mode="off"),
        audit_journal=journal,
    )
    client = await _client(iface)
    try:
        assert (await client.get("/afir")).status == 200
        (entry,) = [e for e in _journalled(journal) if e.get("path") == "/afir"]
        assert entry["auth"] == LOCAL_IDENTITY.source
        assert entry["user"] == LOCAL_IDENTITY.user_name
    finally:
        iface.rate_limiter.close()
        await client.close()


async def test_a_disabled_journal_answers_an_empty_list_and_says_it_is_off(cfg_dir, packs, store):
    """200 and not 404: "nobody has done anything" and "nothing is being recorded" are
    different answers, and an operator checking on a deployment needs to tell them apart."""
    iface = IncidentInputInterface(
        _config(),
        live_config={"knowledge": {"pack_dir": f"knowledge/{PACK}"}},
        storage=store,
        identity_resolver=_resolver(),
    )
    client = await _client(iface)
    try:
        body = await json_of(await client.get("/api/v1/audit", headers=AS_OWNER))
        assert body["entries"] == [] and body["journal"]["enabled"] is False
    finally:
        iface.rate_limiter.close()
        await client.close()


# -- the actor on a durable decision --------------------------------------


async def test_a_gate_decision_records_the_resolved_caller_and_not_the_claimed_one(jobs_client):
    """The actor box is a free-text field on a form. Where an identity is enforced, what it
    holds is a claim and the resolved caller is a fact — so a decision cannot be signed with
    somebody else's name."""
    jm, client = jobs_client
    job_id = await _submit(client, AS_USER)
    job = jm.get_job(job_id)
    job.open_gate = {"stage": "log_retrieval", "opened_at": "now"}
    resp = await client.post(
        f"/api/v1/jobs/{job_id}/gate",
        headers=AS_USER,
        json={"action": "approve", "actor": "somebody.else@example.test"},
    )
    assert resp.status in (200, 409), await resp.text()
    recorded = [i.get("actor") for i in (job.interventions or [])]
    assert "somebody.else@example.test" not in recorded
    if recorded:
        assert recorded[0] == AS_USER[NAME_HEADER]


# -- my own credentials ------------------------------------------------------
#
# `test_user_secrets.py` proves the store's own semantics — a name nothing reads is refused, the
# value never comes back out, one caller's is not another's. It cannot prove the three routes
# reach any of that, which is the failure that matters here: a surface that accepts a token and
# changes no run reads exactly like one that works, and the caller has no way to tell, because
# the one thing that would confirm it is the value they are never shown again.
#
# The administrator asymmetry is the other half. Everywhere else in this file an admin edits the
# shared thing; here there is no shared thing to edit, so an admin has no route to anybody's
# credential in either direction — including their own view of it.

#: A name the credentials fixture's config really reads, so the store will accept it.
CRED_NAME = "AFIR_TEST_REST_TOKEN"

#: One it does not, for the refusal.
UNREAD_NAME = "AFIR_TEST_NOBODY_READS_THIS"

#: A deployment that reads one credential by name, keeps its durable store on another, and
#: signs in to a third backend with a username — the three classes the surface must report.
CRED_CONFIG = {
    "knowledge": {"pack_dir": f"knowledge/{PACK}"},
    "log_sources": {
        "backends": {
            "rest": {"tickets": {"token_env": CRED_NAME}},
            "elasticsearch": {"main": {"hosts": ["https://es.example.test:9200"]}},
        }
    },
    "storage": {"databricks": {"token_env": "AFIR_TEST_STORE_TOKEN"}},
}

SECRET = "paste-of-a-real-token"


@pytest.fixture
async def creds(cfg_dir, packs, store):
    """Identity enforced, a journal wired, and a personal-credential store installed.

    Installed through `set_secret_store`, which is process-global for the same reason the
    mirrors are — so it is cleared on the way out, or a leaked store lands in whichever test
    file runs next.
    """
    journal = AuditJournal(storage=store, enabled=True, flush_seconds=3600)
    secrets = UserSecretStore(store, offered=offered_names(CRED_CONFIG, {}))
    assert CRED_NAME in secrets.offered and secrets.available
    set_secret_store(secrets)
    iface = IncidentInputInterface(
        _config(),
        live_config=CRED_CONFIG,
        storage=store,
        identity_resolver=_resolver(),
        audit_journal=journal,
    )
    got = await _client(iface)
    yield journal, got
    set_secret_store(None)
    iface.rate_limiter.close()
    await got.close()


def _row(body, name=CRED_NAME):
    (row,) = [r for r in body["secrets"] if r["name"] == name]
    return row


async def test_a_caller_replaces_a_credential_and_never_reads_it_back(creds):
    """The whole surface in one pass: it is accepted, it is in force, and it is gone. The
    read-back is asserted over the WHOLE response text and not over a `value` key, because the
    leak that matters is the secret appearing under any name at all."""
    _, client = creds
    saved = await client.put(f"/api/v1/secrets/{CRED_NAME}", headers=AS_USER,
                             json={"value": SECRET})
    assert saved.status == 200, await saved.text()
    body = await json_of(saved)
    assert body["saved"] is True and body["personal"] is True
    assert body["fingerprint"] and SECRET not in json.dumps(body)

    listed = await client.get("/api/v1/secrets", headers=AS_USER)
    text = await listed.text()
    assert SECRET not in text
    row = _row(json.loads(text))
    assert row["personal"] is True and row["source"] == "personal"
    assert row["fingerprint"] == body["fingerprint"]
    assert row["used_by"], "a caller cannot judge a replacement without knowing what reads it"


async def test_a_clear_puts_the_deployments_credential_back(creds):
    _, client = creds
    await client.put(f"/api/v1/secrets/{CRED_NAME}", headers=AS_USER, json={"value": SECRET})
    cleared = await client.delete(f"/api/v1/secrets/{CRED_NAME}", headers=AS_USER)
    assert cleared.status == 200, await cleared.text()
    assert (await json_of(cleared))["cleared"] is True
    row = _row(await json_of(await client.get("/api/v1/secrets", headers=AS_USER)))
    assert row["personal"] is False and row["source"] == "shared"
    assert row["fingerprint"] == ""


async def test_an_administrator_has_no_route_to_anybody_elses_credential(creds):
    """The inverse of every other write in this file. There is no shared thing here to
    administer, so the admin's own view is their own — and a DELETE by them is about their own
    state, which is why it is a 404 and not somebody's credential being revoked."""
    _, client = creds
    await client.put(f"/api/v1/secrets/{CRED_NAME}", headers=AS_USER, json={"value": SECRET})

    theirs = await client.get("/api/v1/secrets", headers=AS_OWNER)
    body = await json_of(theirs)
    assert body["you"] == AS_OWNER[NAME_HEADER]
    assert _row(body)["personal"] is False and _row(body)["fingerprint"] == ""
    assert SECRET not in json.dumps(body)

    revoked = await client.delete(f"/api/v1/secrets/{CRED_NAME}", headers=AS_OWNER)
    assert revoked.status == 404, await revoked.text()

    survived = _row(await json_of(await client.get("/api/v1/secrets", headers=AS_USER)))
    assert survived["personal"] is True


async def test_a_second_caller_holds_their_own_value_for_the_same_name(creds):
    _, client = creds
    first = await client.put(f"/api/v1/secrets/{CRED_NAME}", headers=AS_USER,
                             json={"value": SECRET})
    second = await client.put(f"/api/v1/secrets/{CRED_NAME}", headers=AS_OWNER,
                              json={"value": "a different paste"})
    assert (await json_of(first))["fingerprint"] != (await json_of(second))["fingerprint"]
    mine = _row(await json_of(await client.get("/api/v1/secrets", headers=AS_USER)))
    assert mine["fingerprint"] == (await json_of(first))["fingerprint"]


async def test_a_name_no_reader_resolves_is_refused_and_says_which_are(creds):
    """A stored no-op is the failure this codebase forbids: a 200, a credential on disk, and
    every run unchanged. Both routes refuse it, and both name the alternatives, because the
    caller's mistake is usually a spelling."""
    _, client = creds
    for resp in (
        await client.put(f"/api/v1/secrets/{UNREAD_NAME}", headers=AS_USER,
                         json={"value": SECRET}),
        await client.delete(f"/api/v1/secrets/{UNREAD_NAME}", headers=AS_USER),
    ):
        assert resp.status == 400, await resp.text()
        body = await json_of(resp)
        assert UNREAD_NAME in body["error"]
        assert CRED_NAME in body["offerable"]


async def test_a_name_you_never_set_is_a_different_answer_from_one_nobody_reads(creds):
    """404 against 400, and the distinction is the remedy: one is a mistake about this
    deployment, the other about the caller's own state."""
    _, client = creds
    missing = await client.delete(f"/api/v1/secrets/{CRED_NAME}", headers=AS_USER)
    assert missing.status == 404
    assert CRED_NAME in (await json_of(missing))["error"]


async def test_an_empty_paste_is_refused_rather_than_stored(creds):
    _, client = creds
    resp = await client.put(f"/api/v1/secrets/{CRED_NAME}", headers=AS_USER, json={"value": " "})
    assert resp.status == 400, await resp.text()
    row = _row(await json_of(await client.get("/api/v1/secrets", headers=AS_USER)))
    assert row["personal"] is False


async def test_the_surface_reports_what_it_cannot_offer(creds):
    """A panel listing one name and silently dropping the deployment's store token and its
    username-based backends reads as a panel that covers everything."""
    _, client = creds
    body = await json_of(await client.get("/api/v1/secrets", headers=AS_USER))
    assert body["available"] is True and "reason" not in body
    withheld = {row["name"] for row in body["withheld"]}
    assert "AFIR_TEST_STORE_TOKEN" in withheld
    assert any("Elasticsearch" in name for name in withheld)
    assert all(row["reason"] for row in body["withheld"])
    # The three things a 200 does not say on its own.
    assert "your own runs only" in body["note"]
    assert "never displayed again" in body["note"]
    assert "next run" in body["note"]


async def test_the_one_name_this_deployment_offers_is_not_also_listed_as_unreplaceable(
    cfg_dir, packs, store
):
    """The shipped config keeps the store on the SAME token the reasoning endpoint reads, so the
    panel offered one name at the top and, at the bottom, advised that replacing it "would do
    nothing" — about the only name the feature applies to. Asserted here rather than only in
    `test_user_secrets.py` because dropping the argument at this call site is the whole defect
    and leaves the unit test green."""
    config = json.loads(json.dumps(CRED_CONFIG))
    config["storage"] = {"databricks": {"token_env": CRED_NAME}}
    secrets = UserSecretStore(store, offered=offered_names(config, {}))
    set_secret_store(secrets)
    iface = IncidentInputInterface(
        _config(), live_config=config, storage=store, identity_resolver=_resolver()
    )
    client = await _client(iface)
    try:
        body = await json_of(await client.get("/api/v1/secrets", headers=AS_USER))
        assert _row(body)["name"] == CRED_NAME
        withheld = {row["name"] for row in body["withheld"]}
        assert CRED_NAME not in withheld
        # Still reported, because the limit is real: durable state keeps the deployment's.
        assert any(CRED_NAME in row["reason"] for row in body["withheld"])
    finally:
        set_secret_store(None)
        iface.rate_limiter.close()
        await client.close()


async def test_a_deployment_with_nowhere_durable_refuses_rather_than_accepting(client):
    """The `client` fixture installs no store. An accepted save that resolves nowhere is the
    same defect as the config switch that reports success onto a disk the restart discards."""
    set_secret_store(None)
    body = await json_of(await client.get("/api/v1/secrets", headers=AS_USER))
    assert body["available"] is False and body["secrets"] == []
    assert body["reason"]
    for resp in (
        await client.put(f"/api/v1/secrets/{CRED_NAME}", headers=AS_USER,
                         json={"value": SECRET}),
        await client.delete(f"/api/v1/secrets/{CRED_NAME}", headers=AS_USER),
    ):
        assert resp.status == 503, await resp.text()


async def test_the_journal_records_the_name_and_the_fingerprint_and_never_the_value(creds):
    """This journal is readable by an administrator, and the whole point of the surface is that
    the value is not — so the record has to be enough to answer "who changed what, when" and no
    more than that."""
    journal, client = creds
    saved = await json_of(
        await client.put(f"/api/v1/secrets/{CRED_NAME}", headers=AS_USER,
                         json={"value": SECRET})
    )
    await client.delete(f"/api/v1/secrets/{CRED_NAME}", headers=AS_USER)
    entries = [e for e in _journalled(journal) if e["kind"] == "secret_change"]
    by_action = {e["action"]: e for e in entries}
    assert sorted(by_action) == ["clear", "set"]
    assert all(e["user"] == AS_USER[NAME_HEADER] for e in entries)
    assert all(e["name"] == CRED_NAME for e in entries)
    assert by_action["set"]["fingerprint"] == saved["fingerprint"]
    assert SECRET not in json.dumps(entries)


async def test_a_request_binds_its_caller_so_the_work_it_starts_reads_their_credential(
    cfg_dir, packs, store
):
    """The seam the other tests cannot see. A retriever, an embedding provider and the LLM
    client each read `resolve_env` with no notion of a caller, so the request binds the segment
    on a ContextVar — and a run it starts inherits that binding. Asserted through a synthetic
    route, because the property is of the middleware every handler passes through and not of any
    one handler.
    """
    secrets = UserSecretStore(store, offered=offered_names(CRED_CONFIG, {}))
    set_secret_store(secrets)
    seen = {}

    async def probe(request):
        seen[request.headers.get(NAME_HEADER)] = personal_value(CRED_NAME)
        return web.json_response({"ok": True})

    iface = IncidentInputInterface(
        _config(),
        live_config=CRED_CONFIG,
        storage=store,
        identity_resolver=_resolver(),
    )
    iface.app.router.add_get("/api/v1/_probe", probe)
    client = await _client(iface)
    try:
        await client.put(f"/api/v1/secrets/{CRED_NAME}", headers=AS_USER,
                         json={"value": SECRET})
        assert (await client.get("/api/v1/_probe", headers=AS_USER)).status == 200
        assert (await client.get("/api/v1/_probe", headers=AS_OWNER)).status == 200
    finally:
        set_secret_store(None)
        iface.rate_limiter.close()
        await client.close()
    assert seen[AS_USER[NAME_HEADER]] == SECRET
    # The second request is the isolation half, and it is only meaningful because it ran AFTER
    # the first: one caller's binding reaching the next request is the failure a shared session
    # already had. The `finally` reset is not asserted here and cannot be — a task inherits a
    # *copy* of the context, so nothing set inside a request escapes it either way; what the
    # reset protects is a reader on the same context after the handler, in `test_user_secrets.py`.
    assert seen[AS_OWNER[NAME_HEADER]] is None
