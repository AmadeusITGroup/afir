"""
`src/storage/mirror.py` — durability for the two trees that are edited byte by byte.

Config and the knowledge pack are the only durable state that is not a blob. Both are edited in
place by code that exists because a YAML round trip destroys the file, so neither can move behind
the storage seam: the local tree stays the working copy and the bytes are mirrored after each
write. Four failure modes, every one of which reports success while losing the edit:

1. the push is skipped — the save answers 200, the file is on the container's disk, and the restart
   discards it. So the reported flag is asserted and not just the store's contents;
2. the whole tree is seeded instead of only the edits, freezing the pack at whatever the first
   container uploaded. Asserted directly: a file added to the bundle must appear after a restart
   while an edit to another file survives it;
3. a delete reverts, removing a file the bundle ships working until the next boot re-seeds it.
   Hence the tombstone and a restart test asserting the file is still gone;
4. the undo history does not survive, so undo is available exactly until it is needed.

The mirror is inert on a local deployment, asserted too: none of this may run on the VM path, where
the working copy already is the durable copy and a mirror would overlay the checked-in pack with a
stale duplicate.

``LocalStorage`` is the durable store in most tests, being the seam's behavioural oracle, with one
round trip over the fake Files API server because a key carrying the ``.history`` alias has to
survive the remote backend's own validation as well as ``safe_key``.
"""

import json
import shutil

import pytest

from src import config_store
from src.knowledge import pack_store as ps
from src.storage import LocalStorage
from src.storage.mirror import (CONFIG_PREFIX, KNOWLEDGE_PREFIX,
                                MAX_FILE_BYTES, TOMBSTONE_KEY, MirrorSet,
                                TreeMirror, build_mirrors, seed_working_copies,
                                working_copies_are_writable)
from tests.installed_packs import FIXTURE_PACK

# A config file with the shape that matters: a comment carrying the reason for a value,
# which is the thing a `safe_load`/`safe_dump` round trip would delete and which therefore
# has to survive the mirror's round trip too.
CONFIG_TEXT = """\
# The anomaly threshold. 0.7 was measured against 20 real incidents; below it the
# report fills with noise.
anomaly_detection:
  threshold: 0.7
  # Kept deliberately low.
  max_anomalies: 15

knowledge:
  pack_dir: mock_domain
"""


class _RefusingStorage(LocalStorage):
    """A store that accepts reads and refuses writes to a named key.

    Not a broken store — a *selectively* broken one, because "the push failed" has to be
    distinguishable from "the local write failed", and a store that refuses everything
    cannot tell those apart.
    """

    def __init__(self, root, refuse=()):
        super().__init__(root=root)
        self.refuse = tuple(refuse)

    def put_bytes(self, key, blob):
        if any(key.endswith(suffix) for suffix in self.refuse):
            return False
        return super().put_bytes(key, blob)


@pytest.fixture
def trees(tmp_path, monkeypatch):
    """A bundle, a writable working copy of each tree, and a durable store.

    Three separate roots on purpose: in an App they are three different things (a
    read-only deployed bundle, container scratch, a Volume), and a fixture that collapsed
    any two of them would make the interesting tests vacuous.
    """
    from src.storage import mirror as mirror_mod

    bundle = tmp_path / "bundle"
    (bundle / "config").mkdir(parents=True)
    (bundle / "config" / "main_config.yaml").write_text(CONFIG_TEXT, encoding="utf-8")
    (bundle / "config" / "main_config.yaml.bak").write_text(
        "backends:\n  db:\n    password: a-real-credential\n", encoding="utf-8"
    )
    (bundle / "config" / "templates").mkdir()
    (bundle / "config" / "templates" / "main_config.yaml").write_text(
        "# reference only\n", encoding="utf-8"
    )
    pack = bundle / "knowledge" / FIXTURE_PACK
    (pack / "shared" / "checks").mkdir(parents=True)
    (pack / "source_catalog.yaml").write_text(
        "sources:\n  - id: src_a\n    endpoint: e1\n", encoding="utf-8"
    )
    (pack / "shared" / "checks" / "shape.yaml").write_text(
        "checks:\n  one_actor:\n    kind: distinct_count\n", encoding="utf-8"
    )

    working_config = tmp_path / "work" / "config"
    working_knowledge = tmp_path / "work" / "knowledge"
    working_config.mkdir(parents=True)
    working_knowledge.mkdir(parents=True)

    monkeypatch.setattr(mirror_mod, "REPO_ROOT", bundle)
    monkeypatch.setattr(mirror_mod, "config_dir", lambda: working_config)
    monkeypatch.setattr(config_store, "config_dir", lambda: working_config)
    monkeypatch.setattr(ps, "knowledge_pack_dir", lambda name: working_knowledge / name)

    store = LocalStorage(root=tmp_path / "durable")

    yield {
        "bundle": bundle,
        "config": working_config,
        "knowledge": working_knowledge,
        "store": store,
        "tmp": tmp_path,
    }

    config_store.set_mirror(None)
    ps.set_mirror(None)


def _mirrors(trees, store=None):
    """The mirror set the app builds, installed. Returns it for direct assertions."""
    mirrors = build_mirrors(
        {"storage": {"backend": "local", "mirror_config_and_pack": "always"}},
        store or trees["store"],
    )
    mirrors.install()
    return mirrors


def _boot(trees, store=None):
    """One container start: seed the bundle's defaults, then lay the durable copy over."""
    seed_working_copies()
    mirrors = _mirrors(trees, store)
    report = mirrors.sync_down()
    return mirrors, report


def _restart(trees, store=None):
    """A restart: the container's disk is gone, the durable store is not."""
    shutil.rmtree(trees["config"], ignore_errors=True)
    shutil.rmtree(trees["knowledge"], ignore_errors=True)
    trees["config"].mkdir(parents=True)
    trees["knowledge"].mkdir(parents=True)
    return _boot(trees, store)


# --- the mirror is inert on a local deployment ------------------------------


def test_no_mirror_is_built_for_a_local_backend():
    """The VM path must not acquire one.

    Not a preference: the working copy there IS the durable copy, so ``sync_down`` would
    overlay the repo's checked-in pack with a copy of itself, and every pack write would
    pay an extra upload for nothing.
    """
    assert build_mirrors({}, LocalStorage(root="/tmp")) is None
    assert build_mirrors({"storage": {"backend": "local"}}, LocalStorage()) is None


def test_mirroring_can_be_forced_on_and_off_independently_of_the_backend():
    local = LocalStorage(root="/tmp")
    assert (
        build_mirrors(
            {"storage": {"backend": "local", "mirror_config_and_pack": "always"}}, local
        )
        is not None
    )

    class _Remote(LocalStorage):
        kind = "databricks"

    assert build_mirrors({"storage": {"backend": "databricks"}}, _Remote()) is not None
    assert (
        build_mirrors(
            {
                "storage": {
                    "backend": "databricks",
                    "mirror_config_and_pack": "never",
                }
            },
            _Remote(),
        )
        is None
    )


def test_an_unknown_mirror_mode_does_not_silently_disable_durability(caplog):
    """A typo must not read as ``never``.

    ``auto`` is the fallback, so a remote deployment with a mis-spelled value still
    mirrors — and says so loudly. Reading it as "off" would answer every save with a 200
    and lose them all at the restart.
    """

    class _Remote(LocalStorage):
        kind = "databricks"

    with caplog.at_level("ERROR"):
        built = build_mirrors(
            {
                "storage": {
                    "backend": "databricks",
                    "mirror_config_and_pack": "smetimes",
                }
            },
            _Remote(),
        )
    assert built is not None
    assert "mirror_config_and_pack" in caplog.text


# --- seeding the working copy ----------------------------------------------


def test_seeding_copies_the_bundle_into_the_working_copy(trees):
    report = seed_working_copies()
    assert report["config"] == 1  # main_config.yaml; NOT the .bak, NOT templates/
    assert (trees["config"] / "main_config.yaml").read_text() == CONFIG_TEXT
    assert not (trees["config"] / "main_config.yaml.bak").exists()
    assert not (trees["config"] / "templates").exists()
    assert (
        trees["knowledge"] / FIXTURE_PACK / "shared" / "checks" / "shape.yaml"
    ).is_file()


def test_seeding_falls_back_to_the_templates_a_bundle_actually_ships(
    tmp_path, monkeypatch
):
    """The deployed bundle has NO ``config/*.yaml``, only ``config/templates/``.

    Those files hold live tokens and are gitignored — and ``bundle deploy`` honours
    ``.gitignore`` on top of ``sync.exclude``, so the synced tree's ``config/`` contains
    nothing but ``templates/``. Seeding depth-1 only therefore copied **zero** files and
    the App died in ``load_config``'s bare ``open()`` on its first start, before logging
    was configured. Reproduced against the CLI's own sync manifest.

    The `trees` fixture cannot catch this: it ships a real ``main_config.yaml`` in the
    fake bundle, which the deployed one can never contain. Hence a bundle built to the
    shape the platform gets.
    """
    from src.storage import mirror as mirror_mod

    bundle = tmp_path / "bundle"
    (bundle / "config" / "templates").mkdir(parents=True)
    for name in ("main_config.yaml", "llm_config.yaml", "plugin_config.yaml"):
        (bundle / "config" / "templates" / name).write_text(
            f"# shipped default: {name}\nknowledge:\n  pack_dir: mock_domain\n",
            encoding="utf-8",
        )
    (bundle / "knowledge" / FIXTURE_PACK).mkdir(parents=True)
    working = tmp_path / "work" / "config"
    working.mkdir(parents=True)
    monkeypatch.setattr(mirror_mod, "REPO_ROOT", bundle)
    monkeypatch.setattr(mirror_mod, "config_dir", lambda: working)
    monkeypatch.setattr(
        ps, "knowledge_pack_dir", lambda name: tmp_path / "work" / "knowledge" / name
    )

    report = seed_working_copies()

    # Flattened: `templates/main_config.yaml` must arrive as `main_config.yaml`, which is
    # where `config_path()` looks. Landing it under a `templates/` subdirectory would
    # report three files copied and still fail the boot.
    assert report["config"] == 3
    assert (working / "main_config.yaml").is_file()
    assert (working / "llm_config.yaml").is_file()
    assert not (working / "templates").exists()


def test_a_real_config_wins_over_the_template_of_the_same_name(trees):
    """Locally the templates are inert, and per FILE rather than per tree.

    The `trees` bundle ships a real `main_config.yaml` *and* a template of that name, so
    this pins which one the working copy gets — and an operator who has only some of the
    four still gets the shipped defaults for the rest.
    """
    (trees["bundle"] / "config" / "templates" / "llm_config.yaml").write_text(
        "# shipped default\nbase_url: https://example.invalid\n", encoding="utf-8"
    )

    report = seed_working_copies()

    assert (trees["config"] / "main_config.yaml").read_text() == CONFIG_TEXT
    assert "shipped default" in (trees["config"] / "llm_config.yaml").read_text()
    assert report["config"] == 2  # the real main_config, plus the template-only llm one


def test_seeding_never_overwrites_what_is_already_there(trees):
    """A warm restart must not replace an operator's edit with the shipped default.

    The order at boot is seed-then-sync-down, so an overwriting seed would also undo the
    previous line of the boot on the second pass.
    """
    seed_working_copies()
    (trees["config"] / "main_config.yaml").write_text("edited: yes\n", encoding="utf-8")
    seed_working_copies()
    assert (trees["config"] / "main_config.yaml").read_text() == "edited: yes\n"


def test_seeding_is_a_no_op_when_the_bundle_is_the_working_copy(tmp_path, monkeypatch):
    """Which is every local deployment — with both env overrides unset the roots are one.

    Structural rather than a branch on deployment mode: there is nothing to copy when the
    source and the destination are the same directory.
    """
    from src.storage import mirror as mirror_mod

    bundle = tmp_path / "repo"
    (bundle / "config").mkdir(parents=True)
    (bundle / "config" / "main_config.yaml").write_text("a: 1\n", encoding="utf-8")
    (bundle / "knowledge" / FIXTURE_PACK).mkdir(parents=True)
    monkeypatch.setattr(mirror_mod, "REPO_ROOT", bundle)
    monkeypatch.setattr(mirror_mod, "config_dir", lambda: bundle / "config")
    monkeypatch.setattr(
        ps, "knowledge_pack_dir", lambda name: bundle / "knowledge" / name
    )
    assert seed_working_copies() == {"config": 0, "knowledge": 0}


def test_a_read_only_working_copy_is_reported_at_boot(trees, monkeypatch):
    """The alternative is finding out from a 500 on the operator's first save.

    The unwritable root here is a *path under a regular file* — an OSError from the real
    filesystem rather than a patched one, because the probe exists precisely because a
    stand-in once answered for a directory that was not the store.
    """
    assert working_copies_are_writable() == {"config": True, "knowledge": True}
    blocker = trees["tmp"] / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setattr(ps, "knowledge_pack_dir", lambda name: blocker / "packs" / name)
    assert working_copies_are_writable() == {"config": True, "knowledge": False}


# --- config: the round trip ------------------------------------------------


def test_a_config_edit_survives_a_restart_with_its_comments(trees):
    _boot(trees)
    result = config_store.apply_updates({"anomaly_detection.threshold": 0.55})
    assert result["durable"] is True
    assert [c["path"] for c in result["changed"]] == ["anomaly_detection.threshold"]

    _restart(trees)
    after = (trees["config"] / "main_config.yaml").read_text()
    assert "threshold: 0.55" in after
    # The comments are the documentation, and they are also the thing that proves the
    # mirror carried the BYTES rather than a re-serialised mapping.
    assert "0.7 was measured against 20 real incidents" in after
    assert "# Kept deliberately low." in after


def test_a_whole_file_replacement_survives_a_restart(trees):
    _boot(trees)
    result = config_store.replace_file(
        "main_config.yaml", "knowledge:\n  pack_dir: m\n"
    )
    assert result["durable"] is True
    _restart(trees)
    assert (trees["config"] / "main_config.yaml").read_text() == (
        "knowledge:\n  pack_dir: m\n"
    )


def test_a_bak_is_never_pushed_to_durable_storage(trees):
    """``config_store`` writes a ``.bak`` beside every file it patches.

    It holds the file as it was *before* the edit — including any literal credential the
    read path redacts — so it must not be uploaded anywhere. It is also this module's own
    private mechanics, and a store carrying one would lay it back down at boot.
    """
    _boot(trees)
    config_store.apply_updates({"anomaly_detection.threshold": 0.6})
    keys = {o.key for o in trees["store"].list_keys(CONFIG_PREFIX)}
    assert keys == {f"{CONFIG_PREFIX}/main_config.yaml"}


def test_a_config_write_that_cannot_be_stored_reports_not_durable(trees, caplog):
    """The values are patched and in effect; what was lost is the restart.

    ``durable: false`` rather than a raise, because the edit HAS taken effect for this
    process — reporting it as refused would be a second untruth on top of the first.
    """
    store = _RefusingStorage(trees["tmp"] / "durable", refuse=(".yaml",))
    mirrors, _ = _boot(trees, store)
    with caplog.at_level("ERROR"):
        result = config_store.apply_updates({"anomaly_detection.threshold": 0.42})
    assert result["durable"] is False
    assert "threshold: 0.42" in (trees["config"] / "main_config.yaml").read_text()
    assert mirrors.degradation and "config" in mirrors.degradation


# --- the pack: the round trip ---------------------------------------------


def test_a_pack_edit_and_its_undo_history_both_survive_a_restart(trees):
    """Undo that does not outlive the container is not undo.

    The snapshot is the only copy of what the edit replaced, so a mirror that carried the
    new content and not the history would leave the operator with a change they cannot
    revert — available right up to the moment it is needed.
    """
    _boot(trees)
    original = (trees["knowledge"] / FIXTURE_PACK / "source_catalog.yaml").read_text()
    result = ps.write_file(
        FIXTURE_PACK,
        "source_catalog.yaml",
        "sources:\n  - id: src_a\n    endpoint: e2\n",
        actor="tester",
    )
    assert result["durable"] is True
    snap_id = result["snapshot"]
    assert snap_id

    _restart(trees)
    assert (
        "endpoint: e2"
        in (trees["knowledge"] / FIXTURE_PACK / "source_catalog.yaml").read_text()
    )
    # The history index AND the blob it points at, which is what makes the restore real
    # rather than a listed id that resolves to nothing.
    assert [h["id"] for h in ps.history(FIXTURE_PACK, "source_catalog.yaml")] == [
        snap_id
    ]
    ps.restore(FIXTURE_PACK, snap_id)
    assert (
        trees["knowledge"] / FIXTURE_PACK / "source_catalog.yaml"
    ).read_text() == original


def test_the_history_directory_rides_under_an_alias_and_comes_back_as_itself(trees):
    """A storage key segment may not begin with a dot — relaxing that would admit ``..``.

    So ``.history`` is carried as ``history`` and translated back on the way down. The
    round trip is what matters: a one-way alias would store the undo store under a name
    ``sync_down`` then writes to a directory the pack editor does not read.
    """
    _boot(trees)
    ps.write_file(FIXTURE_PACK, "source_catalog.yaml", "sources: []\n")
    keys = {o.key for o in trees["store"].list_keys(KNOWLEDGE_PREFIX)}
    assert f"{KNOWLEDGE_PREFIX}/{FIXTURE_PACK}/history/index.json" in keys
    assert not any("/.history/" in k for k in keys)

    _restart(trees)
    assert (trees["knowledge"] / FIXTURE_PACK / ".history" / "index.json").is_file()


def test_a_created_pack_file_survives_a_restart(trees):
    _boot(trees)
    result = ps.create_file(
        FIXTURE_PACK, "shared/concepts/new_note.md", "---\nid: n1\n---\nBody.\n"
    )
    assert result["durable"] is True
    _restart(trees)
    assert (
        trees["knowledge"] / FIXTURE_PACK / "shared" / "concepts" / "new_note.md"
    ).is_file()


def test_a_deleted_pack_file_does_not_come_back_from_the_bundle(trees):
    """The failure this tombstone exists for: a delete that reverts at the next boot.

    The bundle still ships the file, so a boot with no record of the removal seeds it
    straight back — a change that reports success and then quietly undoes itself.
    """
    _boot(trees)
    result = ps.delete_file(FIXTURE_PACK, "shared/checks/shape.yaml")
    assert result["durable"] is True

    _restart(trees)
    assert not (
        trees["knowledge"] / FIXTURE_PACK / "shared" / "checks" / "shape.yaml"
    ).exists()
    record = json.loads(trees["store"].get_text(f"{KNOWLEDGE_PREFIX}/{TOMBSTONE_KEY}"))
    assert record["deleted"] == [f"{FIXTURE_PACK}/shared/checks/shape.yaml"]


def test_recreating_a_deleted_file_clears_its_tombstone(trees):
    """Otherwise the next boot deletes the file the operator just wrote back."""
    _boot(trees)
    ps.delete_file(FIXTURE_PACK, "shared/checks/shape.yaml")
    ps.create_file(FIXTURE_PACK, "shared/checks/shape.yaml", "checks: {}\n")

    _restart(trees)
    path = trees["knowledge"] / FIXTURE_PACK / "shared" / "checks" / "shape.yaml"
    assert path.is_file()
    assert path.read_text() == "checks: {}\n"


def test_a_pack_write_that_cannot_be_stored_reports_not_durable(trees, caplog):
    store = _RefusingStorage(trees["tmp"] / "durable", refuse=("source_catalog.yaml",))
    _boot(trees, store)
    with caplog.at_level("ERROR"):
        result = ps.write_file(FIXTURE_PACK, "source_catalog.yaml", "sources: []\n")
    assert result["durable"] is False
    # The local write still happened — the editor is not reporting a refusal it did not
    # perform — and the reason names the file.
    assert (
        trees["knowledge"] / FIXTURE_PACK / "source_catalog.yaml"
    ).read_text() == "sources: []\n"
    assert "source_catalog.yaml" in caplog.text


def test_a_no_op_write_is_durable_because_nothing_is_at_risk(trees):
    """Reporting a non-change as non-durable sends the operator hunting a lost edit."""
    store = _RefusingStorage(trees["tmp"] / "durable", refuse=(".yaml",))
    _boot(trees, store)
    current = (trees["knowledge"] / FIXTURE_PACK / "source_catalog.yaml").read_text()
    result = ps.write_file(FIXTURE_PACK, "source_catalog.yaml", current)
    assert result["changed"] is False
    assert result["durable"] is True


# --- the store holds EDITS, not the tree ----------------------------------


def test_a_new_release_delivers_new_files_while_an_edit_survives_it(trees):
    """The rule that makes a redeploy work, and the reason the tree is never seeded.

    Store the whole tree and the pack freezes at whatever the first container uploaded:
    every later bundle — a corrected rule, a new source — is invisible behind a stale copy
    of itself, and no error is raised anywhere.
    """
    _boot(trees)
    ps.write_file(FIXTURE_PACK, "source_catalog.yaml", "sources: []\n")

    # A release: the bundle gains a file and changes one the operator never touched.
    (
        trees["bundle"]
        / "knowledge"
        / FIXTURE_PACK
        / "shared"
        / "checks"
        / "extra.yaml"
    ).write_text("checks:\n  new_one: {}\n", encoding="utf-8")
    (
        trees["bundle"]
        / "knowledge"
        / FIXTURE_PACK
        / "shared"
        / "checks"
        / "shape.yaml"
    ).write_text(
        "checks:\n  one_actor:\n    kind: distinct_count\n    max: 1\n",
        encoding="utf-8",
    )

    _restart(trees)
    pack_root = trees["knowledge"] / FIXTURE_PACK
    assert (pack_root / "shared" / "checks" / "extra.yaml").is_file()
    assert "max: 1" in (pack_root / "shared" / "checks" / "shape.yaml").read_text()
    # ...and the operator's edit still wins over the bundle's own copy of that file.
    assert (pack_root / "source_catalog.yaml").read_text() == "sources: []\n"


def test_only_edited_files_are_in_the_durable_store(trees):
    _boot(trees)
    ps.write_file(FIXTURE_PACK, "source_catalog.yaml", "sources: []\n")
    stored = {
        o.key
        for o in trees["store"].list_keys(KNOWLEDGE_PREFIX)
        if not o.key.endswith("index.json") and "/history/" not in o.key
    }
    assert stored == {f"{KNOWLEDGE_PREFIX}/{FIXTURE_PACK}/source_catalog.yaml"}


# --- what the mirror refuses ----------------------------------------------


def test_an_oversize_file_is_refused_loudly_rather_than_uploaded(trees, caplog):
    """Every real pack file is far under the cap; something over it is an accident.

    Refused *and reported*: silently skipping it is a file that lives on the container's
    disk while the operator has been told it saved.
    """
    _boot(trees)
    big = trees["knowledge"] / FIXTURE_PACK / "huge.yaml"
    big.write_text("k: " + "x" * (MAX_FILE_BYTES + 10), encoding="utf-8")
    mirror = TreeMirror(
        trees["store"], KNOWLEDGE_PREFIX, trees["knowledge"], label="knowledge"
    )
    with caplog.at_level("ERROR"):
        assert mirror.push_path(big) is False
    assert "mirror cap" in caplog.text
    assert mirror.degradation


def test_a_path_outside_the_tree_is_not_a_durability_failure(trees):
    """``push_path`` is called with whatever the write path wrote.

    A path this mirror was never responsible for must answer ``True``: the caller's
    question is "is my edit durable?", and a file elsewhere is not an edit it lost.
    """
    mirror = TreeMirror(
        trees["store"], KNOWLEDGE_PREFIX, trees["knowledge"], label="knowledge"
    )
    stray = trees["tmp"] / "elsewhere.yaml"
    stray.write_text("a: 1\n", encoding="utf-8")
    assert mirror.push_path(stray) is True
    assert mirror.degradation is None
    assert trees["store"].list_keys(KNOWLEDGE_PREFIX) == []


@pytest.mark.parametrize(
    "rel",
    [
        ".DS_Store",
        f"{FIXTURE_PACK}/.DS_Store",
        f"{FIXTURE_PACK}/catalog.yaml.bak",
        f"{FIXTURE_PACK}/catalog.yaml.tmp8123",
        f"{FIXTURE_PACK}/catalog.yaml.prev",
        f"{FIXTURE_PACK}/.git/config",
        "history/index.json",
    ],
)
def test_private_and_hidden_files_are_not_mirrored(trees, rel):
    """Each entry is somebody's private mechanics, and the last one is a collision.

    A real directory literally named ``history`` at the tree root would land in the same
    key space as the aliased ``.history``, so it is refused rather than merged — a merge
    resolves back down to ``.history`` and writes into the undo store.
    """
    mirror = TreeMirror(
        trees["store"], KNOWLEDGE_PREFIX, trees["knowledge"], label="knowledge"
    )
    path = trees["knowledge"] / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    assert mirror.push_path(path) is True
    assert trees["store"].list_keys(KNOWLEDGE_PREFIX) == []


def test_the_config_mirror_carries_only_top_level_yaml(trees):
    """``config/templates/`` is reference material shipped in the bundle, never edited.

    Mirroring it would upload ~100 KB of unchanging text on the first write of any config
    file, and lay a stale copy of the shipped documentation back down at every boot.
    """
    mirror = build_mirrors(
        {"storage": {"mirror_config_and_pack": "always"}}, trees["store"]
    ).config
    seed_working_copies()
    (trees["config"] / "templates").mkdir(exist_ok=True)
    template = trees["config"] / "templates" / "main_config.yaml"
    template.write_text("# reference\n", encoding="utf-8")
    assert mirror.push_path(template) is True
    assert mirror.push_path(trees["config"] / "main_config.yaml") is True
    assert [o.key for o in trees["store"].list_keys(CONFIG_PREFIX)] == [
        f"{CONFIG_PREFIX}/main_config.yaml"
    ]


# --- against a real remote backend ---------------------------------------


def test_the_round_trip_works_over_the_remote_backend(trees):
    """One end-to-end pass over the Files API implementation, not just the oracle.

    The mirror is backend-agnostic by construction, but the keys it invents are not
    obviously acceptable to every backend: the ``history`` alias adds a segment, and the
    remote store validates keys of its own. A local-only suite would find that out from a
    deployment.
    """
    from tests.fake_files_api import serve_fake_volume
    from tests.test_storage import CATALOG, FlushingDatabricksStorage

    base_url, _volume, shutdown = serve_fake_volume()
    store = FlushingDatabricksStorage(
        {
            "host": base_url,
            "catalog": CATALOG,
            "schema": "afir",
            "volume": "state",
            "token_env": "AFIR_TEST_MIRROR_TOKEN",
            "verify_ssl": False,
        }
    )
    try:
        _boot(trees, store)
        result = ps.write_file(FIXTURE_PACK, "source_catalog.yaml", "sources: []\n")
        assert result["durable"] is True
        store.flush(timeout=30)
        _restart(trees, store)
        assert (
            trees["knowledge"] / FIXTURE_PACK / "source_catalog.yaml"
        ).read_text() == "sources: []\n"
        assert (trees["knowledge"] / FIXTURE_PACK / ".history" / "index.json").is_file()
    finally:
        store.close(timeout=5)
        shutdown()


def test_the_knowledge_pack_and_config_round_trip_through_a_DATABASE(trees):
    """The pack in a database, which is a destination and not only a state backend.

    Same reasoning as the remote test above: the mirror is backend-agnostic by
    construction, but the keys it invents are not obviously acceptable to every store, and
    this one turns them into a primary-key column with a ``LIKE`` prefix scan. The
    ``history`` alias adds a segment and the config tree and pack tree share one table, so
    a prefix that over-matched would have the config mirror sync the pack's files down into
    ``config/`` — which a local-only suite finds out from a deployment.
    """
    from src.storage.sql import SqlStorage

    store = SqlStorage(
        {"dialect": "sqlite", "dsn": str(trees["tmp"] / "durable-state.sqlite3")}
    )
    try:
        assert store.degradation is None
        _boot(trees, store)

        pack_result = ps.write_file(
            FIXTURE_PACK, "source_catalog.yaml", "sources: []\n"
        )
        assert pack_result["durable"] is True
        config_result = config_store.apply_updates(
            {"anomaly_detection.threshold": 0.71}
        )
        assert not config_result.get("errors")

        _restart(trees, store)

        assert (
            trees["knowledge"] / FIXTURE_PACK / "source_catalog.yaml"
        ).read_text() == "sources: []\n"
        assert (trees["knowledge"] / FIXTURE_PACK / ".history" / "index.json").is_file()
        # The config came back as a config, not as a pack file under the wrong tree.
        config_text = (trees["config"] / "main_config.yaml").read_text()
        assert "0.71" in config_text
        assert not (trees["config"] / FIXTURE_PACK).exists()
    finally:
        store.close()


@pytest.fixture(autouse=True)
def _remote_token(monkeypatch):
    monkeypatch.setenv("AFIR_TEST_MIRROR_TOKEN", "test-token-not-a-real-secret")


# --- the set, and what it installs ---------------------------------------


def test_installing_wires_both_stores_and_clearing_unwires_them(trees):
    """One object because it is one decision.

    A deployment whose pack survives a restart and whose config does not is nobody's
    intent, and the two stores are reached through the same pair of hooks — so the failure
    mode to guard is wiring one and forgetting the other.
    """
    mirrors = _mirrors(trees)
    assert isinstance(mirrors, MirrorSet)
    assert config_store.mirror() is mirrors.config
    assert ps.mirror() is mirrors.knowledge
    config_store.set_mirror(None)
    ps.set_mirror(None)
    assert config_store.mirror() is None and ps.mirror() is None
