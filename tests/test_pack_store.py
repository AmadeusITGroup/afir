"""`src/knowledge/pack_store.py` — the file layer under the knowledge-pack editor.

The pack loader is deliberately forgiving: `pack._read_yaml` catches every parse error and returns
`{}` so a malformed pack cannot crash startup. The price is that a broken file is
indistinguishable from an absent one — a `source_catalog.yaml` with a dangling `*alias` loads as
zero sources, every condition goes `unknown`, and the report reads INSUFFICIENT DATA, which is
what an investigation that genuinely found nothing looks like. So the assertions that earn this
file are the refusals:

* a write that would leave the file parsing to empty is rejected AND the file is byte-identical
  afterwards, not mostly unchanged;
* a dangling alias introduced into the REAL `source_catalog.yaml` is rejected, that being the only
  place PyYAML's `ComposerError` is ever visible;
* a rejected write leaves no history entry, so the undo list never lists a save that did not
  happen.

The second theme is preservation, asserted on OUTPUT BYTES rather than on intent.
`use_cases/*/rules.yaml` is ~60% comment and `source_catalog.yaml` carries live YAML anchors, so a
`safe_load` → `safe_dump` round trip would delete the first and expand the second. The line-patch
test diffs the real file against itself and requires exactly one changed line with the anchor and
comment counts unchanged; a test asserting "we use line replacement" would pass over an
implementation that reformats.

Every test runs against a COPY of a real pack under `tmp_path`, through the fixtures' redirect of
`pack_store.knowledge_pack_dir` — the single seam every path in the module goes through. Nothing
here may write into `knowledge/`: this module's job is writing to packs, so a test that got the
target wrong would edit the shipped pack.
"""

import re
import shutil

import pytest
import yaml

from src.knowledge import pack_store as ps
from src.knowledge.pack import load_knowledge_pack
from src.utils.paths import REPO_ROOT
from tests.installed_packs import SCALE_PACK, installed_packs

REAL_PACKS = REPO_ROOT / "knowledge"


@pytest.fixture
def packs(tmp_path, monkeypatch):
    """A writable copy of every shipped pack, with the store pointed at it."""
    root = tmp_path / "knowledge"
    root.mkdir()
    for name in installed_packs():
        shutil.copytree(REAL_PACKS / name, root / name)
    monkeypatch.setattr(ps, "knowledge_pack_dir", lambda name: root / name)
    return root


def _anchor_defs(text):
    """Every `&anchor` DEFINITION in `text`, as a sorted list.

    Counted by shape so the assertion says nothing about how a pack names its anchors —
    and so an `&` inside prose (there are two in one shipped catalog) is not mistaken for
    one. A dropped anchor makes the whole file load empty, so this is the sharpest thing
    to compare across a write.
    """
    return sorted(re.findall(r"&([A-Za-z_][\w-]*)", text))


# ------------------------------------------------------------------ path rejection


@pytest.mark.parametrize(
    "hostile",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "a/../../b",
        "..",
        "./x.yaml",
        ".history/index.json",
        ".history/blobs/aa/bb",
        "a/b/c/d/e/f/g.yaml",
        "x y.yaml",
        "a;b.yaml",
        "sub\\file.yaml",
        "C:/windows/x.yaml",
        "",
        "   ",
    ],
)
def test_a_hostile_path_is_rejected_not_repaired(hostile):
    """Reject, never sanitise — the precedent is ``report_delivery._safe_id``.

    A sanitised ``../../etc/passwd`` becomes some *other* real path, so a caller who
    asked for one file and silently received another is a worse outcome than a refusal.
    ``.history`` is in the list for its own reason: it matches the segment allowlist
    perfectly well, and letting the editor edit its own undo store would let a bad save
    destroy the record of the good one.
    """
    with pytest.raises(ps.PackStoreError):
        ps.safe_rel_path(hostile)


def test_a_dot_pair_inside_a_filename_is_still_allowed():
    """``..`` is refused AS A SEGMENT, not as a substring.

    A substring test looks equivalent and is not: a real filename may contain a dot
    pair, and refusing it would reject a legitimate file in order to catch a traversal
    the segment test already catches.
    """
    assert str(ps.safe_rel_path("notes..draft.md")) == "notes..draft.md"


@pytest.mark.parametrize(
    "hostile", ["../other", "/abs", ".history", ".hidden", "a/b", ""]
)
def test_a_hostile_pack_name_is_rejected(hostile):
    with pytest.raises(ps.PackStoreError):
        ps.safe_pack_name(hostile)


def test_the_deepest_real_pack_file_is_within_the_depth_bound(packs):
    """The bound is measured against the real tree, not chosen.

    The deepest shipped file is ``use_cases/<name>/playbooks/<file>.md`` at 4 levels. If
    a pack ever needs a fifth, this test says so by failing here rather than by a
    404 from the editor on one file nobody thought to open.
    """
    deepest = max(
        len(node["path"].split("/")) for node in ps.pack_tree("mock_domain")["nodes"]
    )
    assert deepest <= ps.MAX_DEPTH


# ------------------------------------------------------------------------- reading


def test_the_tree_lists_files_with_the_metadata_the_editor_needs(packs):
    tree = ps.pack_tree("mock_domain")
    by_path = {n["path"]: n for n in tree["nodes"]}
    catalog = by_path["source_catalog.yaml"]
    assert catalog["kind"] == "catalog"
    assert catalog["editable"] is True
    assert catalog["bytes"] > 0 and catalog["lines"] > 0
    assert by_path["shared/checks"]["dir"] is True
    assert by_path["shared/checks/actor_identity.yaml"]["kind"] == "shared_check"
    assert tree["counts"]["files"] == sum(1 for n in tree["nodes"] if not n["dir"])


def test_a_directory_sorts_immediately_before_its_own_contents(packs):
    """Tree order is segment-wise, so a folder cannot be separated from its children.

    Sorting the joined path string makes this depend on where ``/`` falls relative to
    the next character, which would put ``shared_notes.md`` between ``shared/`` and
    ``shared/checks/``. Nothing breaks, but the browser renders children under the wrong
    parent, which is indistinguishable from the file being in the wrong place.
    """
    paths = [n["path"] for n in ps.pack_tree("mock_domain")["nodes"]]
    first = paths.index("shared")
    children = [i for i, p in enumerate(paths) if p.startswith("shared/")]
    assert children, "the mock pack has a shared/ subtree"
    assert children == list(range(first + 1, first + 1 + len(children)))


def test_the_history_store_never_appears_in_the_tree(packs):
    ps.write_file("mock_domain", "README.md", "# changed\n")
    paths = [n["path"] for n in ps.pack_tree("mock_domain")["nodes"]]
    assert (packs / "mock_domain" / ps.HISTORY_DIR).is_dir()
    assert not [p for p in paths if p.startswith(ps.HISTORY_DIR)]


def test_a_mid_write_temp_file_never_appears_in_the_tree(packs):
    """A crashed write leaves a ``.tmpNNN``; presenting it as a pack file would offer
    the operator a half-written copy of a file they already have."""
    (packs / "mock_domain" / "entity_glossary.yaml.tmp999").write_text("junk")
    paths = [n["path"] for n in ps.pack_tree("mock_domain")["nodes"]]
    assert not [p for p in paths if ".tmp" in p]


def test_reading_a_file_returns_the_sha_a_safe_write_needs(packs):
    doc = ps.read_file("mock_domain", "entity_glossary.yaml")
    on_disk = (packs / "mock_domain" / "entity_glossary.yaml").read_text()
    assert doc["text"] == on_disk
    assert doc["sha256"] == ps.sha256_text(on_disk)
    assert doc["kind"] == "glossary"


def test_reading_a_missing_file_raises_rather_than_returning_empty(packs):
    with pytest.raises(ps.PackFileNotFound):
        ps.read_file("mock_domain", "no_such_file.yaml")


def test_a_non_text_suffix_is_not_editable(packs):
    (packs / "mock_domain" / "diagram.png").write_bytes(b"\x89PNG\r\n")
    with pytest.raises(ps.PackStoreError):
        ps.read_file("mock_domain", "diagram.png")
    node = {n["path"]: n for n in ps.pack_tree("mock_domain")["nodes"]}["diagram.png"]
    assert node["editable"] is False and node["text"] is False


def test_a_generated_schema_is_listed_but_not_inline_editable(packs):
    """The real pack's biggest generated inventory is ~456 KB.

    ``editable: false`` is what tells the UI to offer a download and a line-range patch
    instead of a textarea. Putting half a megabyte in a browser control and PUTting it
    back is the fastest way to lose a file, so this is asserted against the real file
    rather than a fabricated one.
    """
    if SCALE_PACK is None:
        pytest.skip("no large pack installed")
    big = [
        n
        for n in ps.pack_tree(SCALE_PACK)["nodes"]
        if not n["dir"] and n["bytes"] > ps.INLINE_EDIT_MAX_BYTES
    ]
    assert big, "expected at least one file over the inline-edit cap"
    assert all(n["editable"] is False for n in big)
    assert all(n["text"] is True for n in big)


# -------------------------------------------------------------- the write refusals


def test_a_write_that_would_load_as_empty_is_refused_and_changes_nothing(packs):
    """The defence against the silent-empty trap.

    The candidate is valid YAML and not blank — it is all comments — so nothing before
    this check would object, and the pack would load with the file's whole contribution
    missing while reporting no error anywhere. Both halves are asserted: the refusal,
    and that the file on disk is the SAME BYTES afterwards.
    """
    target = packs / "mock_domain" / "entity_glossary.yaml"
    before = target.read_bytes()
    with pytest.raises(ps.PackWriteRejected) as exc:
        ps.write_file(
            "mock_domain", "entity_glossary.yaml", "# nothing but a comment\n"
        )
    assert "empty" in str(exc.value)
    assert target.read_bytes() == before


def test_a_write_of_unparsable_yaml_is_refused_with_the_parser_s_own_line(packs):
    target = packs / "mock_domain" / "entity_glossary.yaml"
    before = target.read_bytes()
    with pytest.raises(ps.PackWriteRejected) as exc:
        ps.write_file("mock_domain", "entity_glossary.yaml", "entities:\n  - [1,\n")
    assert exc.value.line > 0, "the operator needs to be told WHERE"
    assert target.read_bytes() == before


def test_a_dangling_anchor_in_the_real_catalog_is_refused(packs):
    """The one check no other layer performs.

    A `*ref` whose `&anchor` was deleted raises `ComposerError` for the WHOLE file, and
    `_read_yaml` turns that into `{}` — a pack with zero sources. The mutation here is
    the realistic one: an author tidies the line that happens to carry the anchor
    definition, several thousand lines above the reference that needs it.

    The anchor is chosen by finding one with a LIVE reference, not just the first one
    declared. Most anchors in the shipped catalog are read only from blocks that are
    currently commented out, so deleting one of those is legitimately harmless — and a
    test that picked the first `&` would assert a refusal that should not happen.
    """
    if SCALE_PACK is None:
        pytest.skip("no anchored catalog installed")
    target = packs / SCALE_PACK / "source_catalog.yaml"
    lines = target.read_text().splitlines()
    live_refs = {
        m.group(1)
        for line in lines
        if not line.lstrip().startswith("#")
        for m in [re.search(r":\s*\*(\w+)\s*$", line)]
        if m
    }
    idx = next(
        (
            i
            for i, line in enumerate(lines)
            for m in [re.search(r":\s*&(\w+)\s*$", line)]
            if m and m.group(1) in live_refs
        ),
        None,
    )
    if idx is None:
        pytest.skip("no anchor with a live reference in the installed catalog")
    stripped = lines[idx].split("&")[0].rstrip()
    before = target.read_bytes()
    with pytest.raises(ps.PackWriteRejected) as exc:
        ps.replace_lines(
            SCALE_PACK,
            "source_catalog.yaml",
            idx + 1,
            idx + 1,
            stripped,
            expect_first_line=lines[idx],
        )
    assert "alias" in str(exc.value)
    assert target.read_bytes() == before


def test_a_refused_write_leaves_no_history_entry(packs):
    """The undo list must not record a save that never happened.

    Snapshotting before verifying is the natural implementation and it is wrong: the ids
    then stop lining up with the file's actual states, so the operator's third-from-last
    entry is not the file's third-from-last state.
    """
    ps.write_file("mock_domain", "README.md", "# one real edit\n")
    for bad in ("# comment only\n", "a: [1,\n"):
        with pytest.raises(ps.PackWriteRejected):
            ps.write_file("mock_domain", "entity_glossary.yaml", bad)
    assert ps.history("mock_domain", "entity_glossary.yaml") == []
    assert len(ps.history("mock_domain", "README.md")) == 1


def test_no_temp_file_survives_a_refused_write(packs):
    with pytest.raises(ps.PackWriteRejected):
        ps.write_file("mock_domain", "entity_glossary.yaml", "# comment only\n")
    assert not list((packs / "mock_domain").glob("*.tmp*"))


def test_a_write_over_the_size_cap_is_refused_as_too_large(packs):
    """A distinct exception type, so the HTTP layer can answer 413 rather than 400.

    The aiohttp app's ``client_max_size`` is a 32 MB *ceiling* raised for the editor's
    biggest legitimate payload; it is not this endpoint's policy, so the policy has to
    be stated here.
    """
    with pytest.raises(ps.PackTooLarge):
        ps.write_file("mock_domain", "README.md", "x" * (ps.WRITE_MAX_BYTES + 1))


def test_a_stale_sha_is_a_conflict_not_an_overwrite(packs):
    """Two editors, one file. The second save must not silently win.

    A config API can do without this because it patches one scalar at a time; a
    multi-file editor holds a whole file's text in a textarea for as long as the
    operator is reading it.
    """
    doc = ps.read_file("mock_domain", "README.md")
    ps.write_file("mock_domain", "README.md", doc["text"] + "\nfirst\n")
    with pytest.raises(ps.PackConflict):
        ps.write_file(
            "mock_domain",
            "README.md",
            doc["text"] + "\nsecond\n",
            expect_sha=doc["sha256"],
        )
    assert "first" in (packs / "mock_domain" / "README.md").read_text()


def test_saving_identical_content_is_a_no_op_with_no_snapshot(packs):
    doc = ps.read_file("mock_domain", "README.md")
    res = ps.write_file(
        "mock_domain", "README.md", doc["text"], expect_sha=doc["sha256"]
    )
    assert res["changed"] is False and res["snapshot"] is None
    assert ps.history("mock_domain", "README.md") == []


# --------------------------------------------------------------- line-range editing


def test_a_line_patch_of_the_real_catalog_changes_exactly_one_line(packs):
    """Comments and anchors survive because nothing re-serialises the file.

    Asserted on OUTPUT BYTES: the diff must have exactly one added and one removed
    line, and the `&anchor` and `#` counts must be unchanged. A test asserting the
    implementation "uses line replacement" would pass over one that reformats.
    """
    if SCALE_PACK is None:
        pytest.skip("no anchored catalog installed")
    target = packs / SCALE_PACK / "source_catalog.yaml"
    before = target.read_text()
    lines = before.splitlines()
    idx = next(
        i for i, line in enumerate(lines) if line.strip().startswith("# ") and i > 20
    )
    ps.replace_lines(
        SCALE_PACK,
        "source_catalog.yaml",
        idx + 1,
        idx + 1,
        "    # rewritten by the pack editor",
        expect_first_line=lines[idx],
    )
    after = target.read_text()
    diff = ps.unified_diff(before, after, "source_catalog.yaml").splitlines()
    added = [
        line for line in diff if line.startswith("+") and not line.startswith("+++")
    ]
    removed = [
        line for line in diff if line.startswith("-") and not line.startswith("---")
    ]
    assert len(added) == 1 and len(removed) == 1
    # Anchors counted by SHAPE, not by name: a pack's anchor-naming convention is the
    # pack's business, and `count("&")` would also match an ampersand in prose.
    assert _anchor_defs(after) == _anchor_defs(before)
    assert after.count("#") == before.count("#")
    assert len(after.splitlines()) == len(before.splitlines())


def test_a_line_patch_of_a_comment_heavy_ruleset_keeps_every_other_comment(packs):
    """`rules.yaml` is roughly 60% comment, and those comments are the pack's
    documentation of why each weighting is what it is."""
    if SCALE_PACK is None:
        pytest.skip("no large pack installed")
    root = packs / SCALE_PACK
    rel = next(
        (str(p.relative_to(root)) for p in (root / "use_cases").rglob("rules.yaml")),
        None,
    )
    if rel is None:
        pytest.skip("the installed large pack has no ruleset")
    target = root / rel
    before = target.read_text()
    lines = before.splitlines()
    idx = next(i for i, line in enumerate(lines) if line.strip().startswith("#"))
    ps.replace_lines(
        SCALE_PACK, rel, idx + 1, idx + 1, "# note", expect_first_line=lines[idx]
    )
    after = target.read_text()
    assert after.count("#") == before.count("#") - lines[idx].count("#") + 1
    assert len(after.splitlines()) == len(before.splitlines())


def test_an_anchored_catalog_keeps_its_anchors_through_a_line_patch(packs):
    """The anchor guarantee, on a fixture rather than on an installed pack.

    The two tests above skip when the large pack is absent, so on a fresh checkout the
    anchor half of "nothing re-serialises" is asserted nowhere — and a `safe_load`→`dump`
    regression would land green. This writes its own anchored catalog, so it never skips.

    `*ref` resolution is the specific stake: PyYAML expands an alias on load, so a
    round-tripped file keeps the VALUES and loses the `&anchor`/`*ref` pair. The document
    still parses and still means the same thing today, which is why nothing downstream
    notices — while a pack author's next edit to the anchor now silently updates one source
    instead of four.
    """
    rel = "catalog_with_anchors.yaml"
    text = (
        "# a catalog that shares one binding across sources\n"
        "sources:\n"
        "  - name: alpha\n"
        "    entity_bindings: &shared_bindings\n"
        "      actor: actor_id\n"
        "  - name: beta\n"
        "    entity_bindings: *shared_bindings\n"
        "  - name: gamma\n"
        "    entity_bindings: *shared_bindings\n"
    )
    ps.create_file("mock_domain", rel, text)
    target = packs / "mock_domain" / rel
    ps.replace_lines(
        "mock_domain",
        rel,
        1,
        1,
        "# a catalog that shares one binding across sources (edited)",
        expect_first_line=text.splitlines()[0],
    )
    after = target.read_text()
    assert "&shared_bindings" in after
    assert after.count("*shared_bindings") == 2
    assert len(after.splitlines()) == len(text.splitlines())


def test_a_pre_image_anchor_mismatch_names_both_sides(packs):
    """The plan an assistant computed was read against a snapshot.

    If a human saved in between, the line numbers now name different content, and
    applying the patch anyway edits the wrong lines while every layer reports success.
    The message has to carry both what was expected and what is there, because the
    operator is the one who has to work out which edit to keep.
    """
    doc = ps.read_file("mock_domain", "README.md")
    real_first = doc["text"].splitlines()[0]
    with pytest.raises(ps.PackConflict) as exc:
        ps.replace_lines(
            "mock_domain", "README.md", 1, 1, "# new", expect_first_line="# not this"
        )
    assert "not this" in str(exc.value) and real_first in str(exc.value)
    assert (packs / "mock_domain" / "README.md").read_text() == doc["text"]


@pytest.mark.parametrize("start,end", [(0, 1), (5, 2), (1, 10**6), (-1, 3)])
def test_a_bad_line_range_is_refused(packs, start, end):
    with pytest.raises(ps.PackStoreError):
        ps.replace_lines("mock_domain", "README.md", start, end, "x")


def test_a_multi_line_replacement_can_shrink_and_grow_the_file(packs):
    doc = ps.read_file("mock_domain", "README.md")
    original_lines = len(doc["text"].splitlines())
    res = ps.replace_lines("mock_domain", "README.md", 2, 4, "one\ntwo\nthree\nfour\n")
    assert res["changed"] is True and res["replaced"] == [2, 4]
    assert (
        len((packs / "mock_domain" / "README.md").read_text().splitlines())
        == original_lines + 1
    )


def test_a_patch_at_the_end_of_a_file_without_a_final_newline_adds_none(packs):
    """Adding a trailing newline would be a change outside the range the caller named,
    which shows up as a spurious line in every subsequent diff."""
    ps.create_file("mock_domain", "notes.txt", "alpha\nbeta")
    ps.replace_lines("mock_domain", "notes.txt", 2, 2, "gamma")
    assert (packs / "mock_domain" / "notes.txt").read_text() == "alpha\ngamma"


# ------------------------------------------------------------- create, delete, undo


def test_creating_a_file_that_exists_is_refused_rather_than_overwriting(packs):
    """ "Save" and "add" are different intents; conflating them turns a mistyped path
    into a silent overwrite of a file the operator did not have open."""
    with pytest.raises(ps.PackFileExists):
        ps.create_file("mock_domain", "README.md", "# clobbered\n")
    assert "# clobbered" not in (packs / "mock_domain" / "README.md").read_text()


def test_creating_a_file_in_a_new_subdirectory_works(packs):
    res = ps.create_file(
        "mock_domain",
        "use_cases/refund_fraud/concepts/new_note.md",
        "# a note\n",
    )
    assert res["created"] is True and res["kind"] == "concept"
    assert (
        packs / "mock_domain" / "use_cases/refund_fraud/concepts/new_note.md"
    ).read_text() == "# a note\n"


def test_creating_an_unparsable_file_is_refused_and_nothing_is_left_behind(packs):
    with pytest.raises(ps.PackWriteRejected):
        ps.create_file("mock_domain", "broken.yaml", "a: [1,\n")
    assert not (packs / "mock_domain" / "broken.yaml").exists()


def test_delete_keeps_the_exact_prior_bytes_and_restore_returns_them(packs):
    """Nothing is destroyed. The snapshot is taken BEFORE the unlink, and the id it
    returns is what puts the file back — byte-for-byte, not approximately."""
    original = (packs / "mock_domain" / "README.md").read_bytes()
    res = ps.delete_file("mock_domain", "README.md")
    assert res["deleted"] is True and res["snapshot"]
    assert not (packs / "mock_domain" / "README.md").exists()
    restored = ps.restore("mock_domain", res["snapshot"])
    assert restored["recreated"] is True
    assert (packs / "mock_domain" / "README.md").read_bytes() == original


def test_deleting_a_missing_file_raises(packs):
    with pytest.raises(ps.PackFileNotFound):
        ps.delete_file("mock_domain", "never_existed.md")


def test_restore_is_itself_snapshotted_so_undo_is_undoable(packs):
    """Otherwise the first undo destroys the state it was undoing from, and a
    mis-aimed restore is unrecoverable — which is the moment undo matters most."""
    doc = ps.read_file("mock_domain", "README.md")
    ps.write_file("mock_domain", "README.md", doc["text"] + "\nedit one\n")
    ps.write_file("mock_domain", "README.md", doc["text"] + "\nedit two\n")
    first = [h for h in ps.history("mock_domain", "README.md")][-1]
    ps.restore("mock_domain", first["id"])
    assert (packs / "mock_domain" / "README.md").read_text() == doc["text"]
    reasons = [h["reason"] for h in ps.history("mock_domain", "README.md")]
    assert "restore" in reasons
    latest = ps.history("mock_domain", "README.md")[0]
    ps.restore("mock_domain", latest["id"])
    assert "edit two" in (packs / "mock_domain" / "README.md").read_text()


def test_restore_reinstates_content_that_does_not_parse_and_says_so(packs):
    """Verification is downgraded to a REPORT on restore, on purpose.

    These bytes were on disk before. Refusing to reinstate a file that does not parse
    would make undo unavailable exactly when an edit has gone wrong — so the operation
    succeeds and the result carries ``parses: False`` for the caller to surface.
    """
    broken = "entities:\n  - [1,\n"
    (packs / "mock_domain" / "README.md").write_text(broken)
    snap = ps.snapshot("mock_domain", "notes.yaml", broken, reason="write")
    ps.create_file("mock_domain", "notes.yaml", "notes: [ok]\n")
    res = ps.restore("mock_domain", snap["id"])
    assert res["parses"] is False and res["parse_error"]
    assert (packs / "mock_domain" / "notes.yaml").read_text() == broken


def test_history_blobs_are_content_addressed_so_a_revert_costs_nothing(packs):
    """Content addressing is what makes snapshotting EVERY write affordable.

    Three saves cycling between two states store two blobs, not three, so there is
    never a reason to make snapshotting conditional.
    """
    doc = ps.read_file("mock_domain", "README.md")
    ps.write_file("mock_domain", "README.md", doc["text"] + "\nA\n")
    ps.write_file("mock_domain", "README.md", doc["text"])
    ps.write_file("mock_domain", "README.md", doc["text"] + "\nA\n")
    blobs = list((packs / "mock_domain" / ps.HISTORY_DIR / "blobs").rglob("*"))
    files = [b for b in blobs if b.is_file()]
    assert len(files) == 2
    assert all(ps.sha256_text(b.read_text()) == b.name for b in files)
    assert len(ps.history("mock_domain", "README.md")) == 3


def test_history_is_per_file_and_newest_first(packs):
    ps.write_file("mock_domain", "README.md", "# a\n")
    ps.write_file("mock_domain", "README.md", "# b\n")
    ps.write_file("mock_domain", "entity_glossary.yaml", "entities:\n  - name: x\n")
    per_file = ps.history("mock_domain", "README.md")
    assert [h["path"] for h in per_file] == ["README.md", "README.md"]
    whole = ps.history("mock_domain")
    assert len(whole) == 3
    assert whole[0]["path"] == "entity_glossary.yaml"


def test_a_corrupt_history_index_does_not_block_editing(packs):
    """History is the safety net, not the system of record. A net that has torn must
    not stop the work — it must be reported and rebuilt."""
    ps.write_file("mock_domain", "README.md", "# a\n")
    (packs / "mock_domain" / ps.HISTORY_DIR / "index.json").write_text("{not json")
    assert ps.history("mock_domain") == []
    res = ps.write_file("mock_domain", "README.md", "# b\n")
    assert res["changed"] is True


def test_an_unknown_snapshot_id_raises(packs):
    with pytest.raises(ps.PackFileNotFound):
        ps.restore("mock_domain", "snap-999999-deadbeefdead")


# -------------------------------------------------------------------------- scaffold


def test_a_scaffolded_pack_loads_and_reaches_no_verdict(packs, tmp_path):
    """The property that matters is that the author's first edit has a green baseline
    to diff against, so the copy is asserted to LOAD, not merely to exist."""
    res = ps.scaffold_pack("gizmo_domain", vocabulary=["gizmo", "gizmos"])
    assert res["created"] is True
    pack = load_knowledge_pack(packs / "gizmo_domain")
    assert pack.sources and pack.entities and pack.rulesets


def test_a_scaffolded_pack_does_not_inherit_the_template_s_own_vocabulary(packs):
    """The measured hazard behind this: the template declares ``example_platform``, and
    the template's own ``rules.yaml`` contains ``example_platform_a:``. The neutrality
    scan reads EVERY installed pack's vocabulary and treats ``_`` as a word boundary, so
    ``cp -r template knowledge/new`` makes the **template's** test fail — a failure
    whose cause is in a directory the author never touched.
    """
    ps.scaffold_pack("gizmo_domain", vocabulary=["gizmo"])
    vocab = (packs / "gizmo_domain" / "domain_vocabulary.yaml").read_text()
    assert "example_platform" not in vocab
    assert "example_widget" not in vocab
    assert yaml.safe_load(vocab)["domain_vocabulary"] == ["gizmo"]


def test_a_scaffolded_pack_keeps_the_vocabulary_file_s_explanatory_header(packs):
    """The header is the only place the omission rule is written down — which words to
    leave out and why an omission must carry its reason. Regenerating that prose here
    would put two versions of it in the tree, to be corrected in one."""
    ps.scaffold_pack("gizmo_domain", vocabulary=["gizmo"])
    vocab = (packs / "gizmo_domain" / "domain_vocabulary.yaml").read_text()
    assert "THE HARD PART IS WHAT TO LEAVE OUT" in vocab


def test_a_scaffolded_pack_does_not_inherit_the_template_s_readme(packs):
    """The template's README says "copy this to start a pack", which inside a real pack
    is simply false — and it is the first file an author opens."""
    ps.scaffold_pack("gizmo_domain", vocabulary=["gizmo"])
    assert not (packs / "gizmo_domain" / "README.md").exists()


def test_scaffolding_without_a_vocabulary_is_refused(packs):
    """A pack directory with no vocabulary file fails
    ``test_every_installed_pack_declares_its_vocabulary`` and takes the whole suite
    down — and the guarantee that file installs would be silently absent for the new
    pack, which is the state that check exists to make impossible.
    """
    for empty in ([], "", None, ["   "]):
        with pytest.raises(ps.PackStoreError):
            ps.scaffold_pack("gizmo_domain", vocabulary=empty)
        assert not (packs / "gizmo_domain").exists()


def test_a_vocabulary_word_that_would_need_quoting_is_refused(packs):
    with pytest.raises(ps.PackStoreError):
        ps.scaffold_pack("gizmo_domain", vocabulary=["fine", "bad: word"])
    assert not (packs / "gizmo_domain").exists()


def test_a_multi_word_vocabulary_entry_is_kept(packs):
    """Real vocabularies contain them, so refusing spaces outright would be wrong."""
    ps.scaffold_pack("gizmo_domain", vocabulary=["Sprocket Ledger", "GIZMO", "gizmo"])
    words = yaml.safe_load(
        (packs / "gizmo_domain" / "domain_vocabulary.yaml").read_text()
    )["domain_vocabulary"]
    assert words == ["sprocket ledger", "gizmo"]


def test_scaffolding_over_an_existing_pack_is_refused(packs):
    with pytest.raises(ps.PackFileExists):
        ps.scaffold_pack("mock_domain", vocabulary=["gizmo"])


def test_a_failed_scaffold_leaves_no_half_copied_pack(packs, monkeypatch):
    """A half-copied pack is SELECTABLE, and would load with pieces missing — the exact
    failure shape this module exists to avoid, arriving through the door meant to
    prevent it."""
    real_copy = shutil.copy2
    calls = {"n": 0}

    def exploding_copy(src, dst, **kw):
        calls["n"] += 1
        if calls["n"] > 3:
            raise OSError("disk full")
        return real_copy(src, dst, **kw)

    monkeypatch.setattr(ps.shutil, "copy2", exploding_copy)
    with pytest.raises(OSError):
        ps.scaffold_pack("gizmo_domain", vocabulary=["gizmo"])
    assert not (packs / "gizmo_domain").exists()


# ------------------------------------------------------------------------- classify


@pytest.mark.parametrize(
    "rel,kind",
    [
        ("entity_glossary.yaml", "glossary"),
        ("source_catalog.yaml", "catalog"),
        ("domain_vocabulary.yaml", "vocabulary"),
        ("reporting.yaml", "reporting"),
        ("rulesets.yaml", "ruleset"),
        ("use_cases/uc/rules.yaml", "ruleset"),
        ("use_cases/uc/reporting.yaml", "reporting"),
        ("use_cases/uc/playbooks/p.md", "playbook"),
        ("use_cases/uc/concepts/c.md", "concept"),
        ("use_cases/uc/cases/case.md", "case"),
        ("use_cases/uc/data/lookup.yaml", "data"),
        ("shared/checks/x.yaml", "shared_check"),
        ("shared/concepts/x.md", "concept"),
        ("schemas/table.yaml", "schema"),
        ("data/lookup.yaml", "data"),
        ("VERDICT.md", "notes"),
        ("notes.txt", "notes"),
        ("use_cases/uc/router_manifest.yaml", "reference"),
        ("diagram.png", "other"),
    ],
)
def test_a_file_s_kind_is_recognised_from_where_it_lives(rel, kind):
    """The KIND drives which help the editor shows beside a file, and the loader finds
    shared checks and concepts by DIRECTORY, so this reads the directory too — a
    filename-only rule would mislabel every file an author names something sensible."""
    assert ps.classify(rel) == kind


def test_every_editable_suffix_gets_a_kind_and_other_is_left_to_attachments():
    """``other`` is the one kind the editor shows no help for, so a file an author can
    TYPE INTO must never land there — and the two that used to
    (an unrecognised ``.yaml``, a ``.txt``) are precisely the shapes a pack invents when
    it needs documentation the loader must not ingest. A YAML one is kept apart from
    prose because the editor parses it on write and prose it does not.
    """
    for suffix in ps.EDITABLE_SUFFIXES:
        assert ps.classify("some_pack_file" + suffix) != "other", suffix
    assert ps.classify("attachment.png") == "other"
    assert ps.classify("procedure.pdf") == "other"


def test_the_real_packs_have_no_unclassifiable_pack_file(packs):
    """A file the editor cannot label gets no help text, which is how an author ends up
    editing a ruleset with a schema's guidance beside it."""
    for name in installed_packs():
        unknown = [
            n["path"]
            for n in ps.pack_tree(name)["nodes"]
            if not n["dir"] and n["kind"] == "other"
        ]
        assert unknown == [], f"{name}: unclassified {unknown}"


# ----------------------------------------------------------------- editing the real
# packs is impossible from here


def test_nothing_in_this_module_writes_outside_the_pack_root(packs):
    """Every path in the module goes through one seam — ``knowledge_pack_dir`` — which
    is why the fixture can redirect it with one line. A helper that built a path any
    other way would write into the SHIPPED pack when these tests run.
    """
    real = ps.packs_root()
    assert real.is_relative_to(packs.parent) or real == packs
