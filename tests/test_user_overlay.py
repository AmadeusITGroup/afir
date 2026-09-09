"""A caller's own edits, and what an administrator's release does to them.

The regression that matters: an admin push must not delete a caller's non-conflicting edit,
and must not silently overwrite a conflicting one either. Both directions lose work.
"""

import pytest

from src.storage import LocalStorage
from src.user_overlay import (ADOPTED, CLEAN, CONFLICT, CONFLICT_SPLIT, MERGED,
                              OverlaySet, UserLayer, merge_three_way,
                              rebase_all)


@pytest.fixture
def storage(tmp_path):
    return LocalStorage(root=tmp_path / "store")


@pytest.fixture
def layer(storage):
    return UserLayer(storage, "4242", "config")


# -- the merge itself ------------------------------------------------------


def test_only_one_side_changed_takes_that_side():
    base = "a\nb\nc\n"
    merged, conflicted = merge_three_way(base, "a\nB\nc\n", base)
    assert (merged, conflicted) == ("a\nB\nc\n", False)


def test_the_other_side_alone_also_wins():
    base = "a\nb\nc\n"
    merged, conflicted = merge_three_way(base, base, "a\nb\nC\n")
    assert (merged, conflicted) == ("a\nb\nC\n", False)


def test_disjoint_changes_both_survive():
    """The case the whole feature exists for: a release and a caller's edit coexisting."""
    merged, conflicted = merge_three_way(
        "one\ntwo\nthree\nfour\n",
        "one\nMINE\nthree\nfour\n",
        "one\ntwo\nthree\nTHEIRS\n",
    )
    assert merged == "one\nMINE\nthree\nTHEIRS\n"
    assert conflicted is False


def test_the_same_change_on_both_sides_is_not_a_conflict():
    merged, conflicted = merge_three_way("a\nb\n", "a\nX\n", "a\nX\n")
    assert (merged, conflicted) == ("a\nX\n", False)


def test_the_same_line_changed_differently_conflicts_and_keeps_both():
    merged, conflicted = merge_three_way("a\nb\nc\n", "a\nMINE\nc\n", "a\nTHEIRS\nc\n")
    assert conflicted is True
    assert "MINE" in merged and "THEIRS" in merged and CONFLICT_SPLIT in merged


def test_a_conflict_marker_starts_on_its_own_line_without_a_final_newline():
    merged, conflicted = merge_three_way("a\nb", "a\nMINE", "a\nTHEIRS")
    assert conflicted is True
    for line in merged.splitlines():
        assert line in ("a", "MINE", "THEIRS") or line.startswith(("<<<", "===", ">>>"))


def test_an_addition_at_the_end_of_each_side():
    merged, conflicted = merge_three_way("a\n", "a\nmine\n", "a\n")
    assert (merged, conflicted) == ("a\nmine\n", False)


def test_an_empty_base_with_one_side_editing():
    merged, conflicted = merge_three_way("", "mine\n", "")
    assert (merged, conflicted) == ("mine\n", False)


# -- the layer -------------------------------------------------------------


def test_an_untouched_file_reads_as_absent(layer):
    assert layer.read("main_config.yaml") is None


def test_a_write_is_read_back(layer):
    layer.write("main_config.yaml", "port: 9000\n", "port: 8080\n")
    assert layer.read("main_config.yaml") == "port: 9000\n"
    assert layer.base_of("main_config.yaml") == "port: 8080\n"


def test_a_second_write_keeps_the_original_fork_point(layer):
    """Otherwise every later rebase compares the caller's text to itself and merges nothing."""
    layer.write("c.yaml", "v: 1\n", "v: 0\n")
    layer.write("c.yaml", "v: 2\n", "v: 1\n")
    assert layer.base_of("c.yaml") == "v: 0\n"


def test_dropping_an_override_restores_the_base_view(layer):
    layer.write("c.yaml", "mine\n", "base\n")
    layer.drop("c.yaml")
    assert layer.read("c.yaml") is None and layer.base_of("c.yaml") is None


def test_paths_excludes_the_bookkeeping_blobs(layer):
    layer.write("a.yaml", "1\n", "0\n")
    layer.write("b.yaml", "1\n", "0\n")
    assert layer.paths() == ["a.yaml", "b.yaml"]


def test_a_nested_knowledge_path_round_trips(storage):
    layer = UserLayer(storage, "4242", "knowledge")
    layer.write("mock_domain/rulesets/example.yaml", "x: 1\n", "x: 0\n")
    assert layer.paths() == ["mock_domain/rulesets/example.yaml"]
    assert layer.read("mock_domain/rulesets/example.yaml") == "x: 1\n"


def test_an_unavailable_store_is_inert_rather_than_fatal():
    layer = UserLayer(None, "4242", "config")
    assert layer.write("c.yaml", "x\n", "y\n") is False
    assert layer.read("c.yaml") is None and layer.paths() == []


# -- rebasing onto a moved base -------------------------------------------


def test_a_release_that_does_not_touch_the_layered_lines_merges_clean(layer):
    layer.write("c.yaml", "a\nMINE\nc\n", "a\nb\nc\n")
    assert layer.rebase("c.yaml", "a\nb\nCHANGED\n") == MERGED
    assert layer.read("c.yaml") == "a\nMINE\nCHANGED\n"


def test_a_release_touching_the_same_line_conflicts_and_keeps_the_edit(layer):
    layer.write("c.yaml", "a\nMINE\n", "a\nb\n")
    assert layer.rebase("c.yaml", "a\nTHEIRS\n") == CONFLICT
    assert "MINE" in layer.read("c.yaml")


def test_a_release_matching_the_callers_edit_drops_the_now_pointless_override(layer):
    layer.write("c.yaml", "a\nX\n", "a\nb\n")
    assert layer.rebase("c.yaml", "a\nX\n") == CLEAN
    assert layer.read("c.yaml") is None


def test_an_unmoved_base_is_not_a_rebase(layer):
    layer.write("c.yaml", "mine\n", "base\n")
    assert layer.rebase("c.yaml", "base\n") == CLEAN
    assert layer.read("c.yaml") == "mine\n"


def test_a_deleted_base_keeps_the_callers_file(layer):
    layer.write("c.yaml", "mine\n", "base\n")
    assert layer.rebase("c.yaml", None) == CLEAN
    assert layer.read("c.yaml") == "mine\n"


def test_a_layer_with_no_fork_point_is_adopted_not_merged(storage, layer):
    layer.write("c.yaml", "mine\n", "base\n")
    storage.delete("users/4242/layers/config/c.yaml.base")
    assert layer.rebase("c.yaml", "moved\n") == ADOPTED
    assert layer.read("c.yaml") == "mine\n"


def test_an_override_identical_to_its_fork_point_is_not_an_edit(layer):
    """It would otherwise pin the file against every future release while changing nothing."""
    layer.write("c.yaml", "a\nb\n", "a\nb\n")
    assert layer.rebase("c.yaml", "a\nRELEASED\n") == CLEAN
    assert layer.read("c.yaml") is None


def test_the_fork_point_advances_so_a_second_release_merges_against_the_first(layer):
    layer.write("c.yaml", "a\nMINE\nc\n", "a\nb\nc\n")
    layer.rebase("c.yaml", "a\nb\nSECOND\n")
    assert layer.rebase("c.yaml", "a\nb\nTHIRD\n") == MERGED
    assert layer.read("c.yaml") == "a\nMINE\nTHIRD\n"


def test_the_state_is_recorded_for_the_tab_that_shows_it(layer):
    layer.write("c.yaml", "a\nMINE\n", "a\nb\n")
    layer.rebase("c.yaml", "a\nTHEIRS\n")
    [row] = layer.describe()
    assert row["path"] == "c.yaml" and row["conflict"] is True


# -- across every caller ---------------------------------------------------


def test_rebase_all_reports_only_what_is_worth_reporting(storage):
    UserLayer(storage, "aaa", "config").write("c.yaml", "a\nMINE\nc\n", "a\nb\nc\n")
    # Edits the same line the release changes, so this one cannot merge.
    UserLayer(storage, "bbb", "config").write("c.yaml", "a\nb\nOTHER\n", "a\nb\nc\n")
    UserLayer(storage, "ccc", "config").write("c.yaml", "a\nb\nc\n", "a\nb\nc\n")

    report = rebase_all(storage, "config", lambda rel: "a\nb\nRELEASED\n")

    assert report["aaa"] == {"c.yaml": MERGED}
    assert report["bbb"] == {"c.yaml": CONFLICT}
    assert "ccc" not in report


def test_rebase_all_ignores_a_caller_with_only_jobs(storage):
    storage.put_text("users/dddd/jobs/j1.json", "{}")
    assert rebase_all(storage, "config", lambda rel: "x\n") == {}


def test_rebase_all_leaves_the_other_tree_alone(storage):
    UserLayer(storage, "aaa", "knowledge").write("p/r.yaml", "mine\n", "base\n")
    assert rebase_all(storage, "config", lambda rel: "moved\n") == {}
    assert UserLayer(storage, "aaa", "knowledge").read("p/r.yaml") == "mine\n"


def test_an_overlay_set_carries_both_trees(storage):
    overlay = OverlaySet(storage, "4242")
    overlay.config.write("c.yaml", "1\n", "0\n")
    overlay.knowledge.write("p/r.yaml", "1\n", "0\n")
    assert overlay.config.paths() == ["c.yaml"]
    assert overlay.knowledge.paths() == ["p/r.yaml"]
    assert overlay.layer("config") is overlay.config
