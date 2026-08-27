"""The pack-authoring skill library: it loads, it is SELECTED rather than offered, and the
neutrality scan really covers it.

WHAT THIS FILE IS GUARDING, and why each one fails silently otherwise:

* **A skill that stopped loading.** The library is read with a frontmatter parser, and a
  malformed document is skipped so one bad file cannot take the assistant down. That is the
  right behaviour and it is also invisible: the assistant proposes a slightly worse plan and
  nothing says why. So `problems()` must NAME what it could not load, and the shipped library
  must have no problems.
* **Selection that quietly matches nothing.** The whole design decision here is that the method
  documents are injected by the engine rather than offered as a tool the model may call, because
  an opt-in library is one a model can decline invisibly. A trigger list that matches nothing
  returns that library to opt-in without changing any observable behaviour.
* **A budget that can delete the spine.** The always-on document is the one that must be in
  front of the model for every change; a character budget that can drop it is a budget that
  silently removes the rules.
* **A scan that does not cover the thing it certifies.** The library sits under `src/` for one
  reason: `pack_validate`'s collision check scans every `.md` there against every installed
  pack's vocabulary, which is what makes "these documents teach method, not a domain" mechanical
  instead of a promise. Move the directory and that guarantee is gone with no test failing —
  the same shape as a harness that splits on a literal and stops asserting. So it is asserted
  here, by path.
"""

import pytest

from src.knowledge import pack_assistant, pack_skills
from src.knowledge.pack_validate import _NEUTRALITY_ROOTS, _NEUTRALITY_SUFFIXES

SKILL = """\
---
name: {name}
title: "A title"
when: "When the thing happens"
always: {always}
triggers: {triggers}
---

# Body

{body}
"""


def write_skill(directory, name, *, always=False, triggers=(), body="a rule"):
    path = directory / f"{name}.md"
    path.write_text(
        SKILL.format(
            name=name,
            always=str(bool(always)).lower(),
            triggers=list(triggers),
            body=body,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def library(tmp_path):
    """A three-skill library: one always-on, two triggered, one of them on two words."""
    root = tmp_path / "skills"
    root.mkdir()
    write_skill(root, "a-spine", always=True)
    write_skill(root, "b-sources", triggers=["source", "never_filter", "projection"])
    write_skill(root, "c-conditions", triggers=["condition", "polarity"])
    return root


# --------------------------------------------------------------- the shipped library


def test_every_shipped_skill_loads_and_declares_what_it_is_for():
    """The library the assistant actually ships with. Empty would be a silent regression.

    `when` is asserted because it is the only thing in the system prompt's index: a skill with
    no `when` is listed by name alone, and the model has no basis to call `read_skill` on it.
    """
    skills = pack_skills.load()
    assert pack_skills.problems() == []
    assert len(skills) >= 5, "the shipped method library has shrunk unexpectedly"
    for skill in skills:
        assert skill.name and skill.when and skill.body
        assert skill.name == skill.name.strip().lower()
        assert len(skill.body) > 500, f"{skill.name} is too short to be method knowledge"


def test_exactly_one_shipped_skill_is_always_on():
    """More than one 'always' and the per-question budget is spent before selection runs."""
    always = [s for s in pack_skills.load() if s.always]
    assert len(always) == 1, [s.name for s in always]
    assert always[0].triggers == [], "an always-on skill needs no triggers"


def test_the_shipped_library_is_inside_the_neutrality_scan():
    """The scan is what makes the library domain-free; assert it actually reaches the files.

    By path and not by rglob, per the reasoning in the neutrality test itself: a layout change
    that moves the directory out of the scanned tree would otherwise pass everything.
    """
    assert ".md" in _NEUTRALITY_SUFFIXES
    roots = [str(r) for r in _NEUTRALITY_ROOTS]
    assert any(str(pack_skills.SKILLS_DIR).startswith(r) for r in roots), (
        f"{pack_skills.SKILLS_DIR} is not under a scanned root — the skills' "
        "domain-neutrality is no longer enforced by anything"
    )
    assert list(pack_skills.SKILLS_DIR.glob("*.md")), "the scan would cover nothing"


# ------------------------------------------------------------------------ selection


def test_the_always_on_skill_is_selected_for_a_question_matching_nothing(library):
    """The spine is not subject to the trigger match — that is what makes it the spine."""
    picked = pack_skills.select("please rename a heading", directory=library)
    assert [s.name for s, _ in picked] == ["a-spine"]
    assert picked[0][1] == []


def test_a_triggered_skill_is_selected_and_says_what_it_matched(library):
    picked = pack_skills.select("add a source to the catalog", directory=library)
    assert [s.name for s, _ in picked] == ["a-spine", "b-sources"]
    assert dict((s.name, t) for s, t in picked)["b-sources"] == ["source"]


def test_more_distinct_triggers_ranks_higher(library):
    """The rank is a proxy for how much of the question is about that skill's subject."""
    picked = pack_skills.select(
        "the projection and never_filter on a source, and one condition",
        directory=library,
    )
    names = [s.name for s, _ in picked]
    assert names == ["a-spine", "b-sources", "c-conditions"]


def test_selection_is_stable_for_the_same_question(library):
    """Deterministic, because a plan that varies run to run cannot be reviewed as a change."""
    once = pack_skills.select("a condition and a source", directory=library)
    twice = pack_skills.select("a condition and a source", directory=library)
    assert [s.name for s, _ in once] == [s.name for s, _ in twice]


def test_a_trigger_matches_across_a_separator_but_not_inside_a_word(tmp_path):
    """`filter` must find `never_filter`; `cap` must not find `capture`.

    The costs are asymmetric — an extra skill spends context, a missing one spends a live run —
    so the boundary is deliberately permissive at a separator and strict at a letter.
    """
    root = tmp_path / "skills"
    root.mkdir()
    write_skill(root, "f", triggers=["filter"])
    write_skill(root, "c", triggers=["cap"])
    names = [s.name for s, _ in pack_skills.select("never_filter", directory=root)]
    assert names == ["f"]
    assert pack_skills.select("capture the output", directory=root) == []


def test_the_focus_files_are_matched_too(library):
    """"add the new one here" names no subject; the file the operator pointed at does."""
    picked = pack_skills.select(
        "add the new one here", focus=["source_catalog.yaml"], directory=library
    )
    assert "b-sources" in [s.name for s, _ in picked]


def test_the_character_budget_drops_a_triggered_skill_but_never_the_spine(library):
    """A budget that can remove the always-on document is a budget that removes the rules."""
    picked = pack_skills.select(
        "a source and a condition", char_budget=10, directory=library
    )
    assert [s.name for s, _ in picked] == ["a-spine"]


def test_the_count_limit_bounds_the_injection_and_never_drops_the_spine(library):
    """The other half of the same defect, and the half a budget test cannot see.

    The spine is exempt from the character budget, so ranking it below a well-matched skill
    still admitted it — until the COUNT limit was reached first, and then it was dropped
    outright with no note anywhere. Asserted at the tightest limit there is.
    """
    picked = pack_skills.select(
        "a projection and never_filter on a source, and a condition and polarity",
        limit=1,
        directory=library,
    )
    assert [s.name for s, _ in picked] == ["a-spine"]
    assert len(
        pack_skills.select("a source and a condition", limit=2, directory=library)
    ) == 2


# -------------------------------------------------------------------------- loading


def test_a_malformed_skill_is_reported_and_the_others_still_load(tmp_path):
    """One bad file must cost one skill, not the library — and must not be silent."""
    root = tmp_path / "skills"
    root.mkdir()
    write_skill(root, "good")
    (root / "no-frontmatter.md").write_text("# just a heading\n", encoding="utf-8")
    (root / "bad-yaml.md").write_text(
        '---\nname: x\ntitle: "unclosed\n---\n\nbody\n', encoding="utf-8"
    )
    (root / "nameless.md").write_text("---\ntitle: x\n---\n\nbody\n", encoding="utf-8")
    (root / "empty-body.md").write_text("---\nname: e\n---\n", encoding="utf-8")
    assert [s.name for s in pack_skills.load(root)] == ["good"]
    problems = pack_skills.problems(root)
    assert len(problems) == 4
    assert any("no-frontmatter.md" in p for p in problems)
    assert any("nameless.md" in p and "name" in p for p in problems)
    assert any("empty-body.md" in p for p in problems)


def test_a_duplicate_name_is_reported(tmp_path):
    """Two files claiming one name makes `read_skill` return whichever sorted first."""
    root = tmp_path / "skills"
    root.mkdir()
    write_skill(root, "one")
    (root / "two.md").write_text(
        SKILL.format(name="one", always="false", triggers=[], body="x"), encoding="utf-8"
    )
    assert any("duplicate" in p for p in pack_skills.problems(root))


def test_a_missing_directory_is_not_an_error(tmp_path):
    """An installation may ship no library; the assistant then behaves as it did before one."""
    root = tmp_path / "absent"
    assert pack_skills.load(root) == []
    assert pack_skills.problems(root) == []
    assert pack_skills.index_text(root) == ""


def test_an_edited_skill_is_picked_up_without_a_restart(tmp_path):
    """The cache is keyed on a signature over the listing, so authoring one is not a deploy."""
    root = tmp_path / "skills"
    root.mkdir()
    write_skill(root, "one")
    assert [s.name for s in pack_skills.load(root)] == ["one"]
    write_skill(root, "two")
    assert [s.name for s in pack_skills.load(root)] == ["one", "two"]


# ------------------------------------------------------------------------ rendering


def test_the_index_names_every_skill_so_an_uninjected_one_can_be_asked_for(library):
    index = pack_skills.index_text(library)
    for skill in pack_skills.load(library):
        assert skill.name in index
    assert "[included below]" in index, "the always-on one must say it is already here"


def test_the_injected_text_says_why_each_skill_is_present(library):
    picked = pack_skills.select("a source", directory=library)
    text = pack_skills.injected_text(picked)
    assert "always applies" in text
    assert "matched: source" in text
    assert "--- skill: b-sources" in text


def test_injected_text_is_empty_when_nothing_was_selected():
    assert pack_skills.injected_text([]) == ""


def test_reading_an_unknown_skill_lists_the_real_names(library):
    out = pack_skills.read("no-such-thing", directory=library)
    assert "no such skill" in out.lower() or "there is no skill" in out
    assert "a-spine" in out


def test_a_truncated_read_says_so(tmp_path, monkeypatch):
    """Same rule as the pack reader: a model that cannot see it got a fragment acts on it."""
    root = tmp_path / "skills"
    root.mkdir()
    write_skill(root, "long", body="x" * 5000)
    monkeypatch.setattr(pack_skills, "READ_CHARS_PER_CALL", 100)
    out = pack_skills.read("long", directory=root)
    assert "TRUNCATED" in out


def test_the_tool_schema_enumerates_the_installed_skills():
    schema = pack_skills.tool_schema()
    assert schema["function"]["name"] == "read_skill"
    names = schema["function"]["parameters"]["properties"]["name"]["enum"]
    assert set(names) == {s.name for s in pack_skills.load()}


# ------------------------------------------------------- as the assistant sees them


#: Every tool the assistant may be handed, by name. A CLOSED SET rather than a screen for
#: forbidden words, because a shadow run watched the model call `edit_file` — a name no list of
#: ("write", "patch", "apply", "delete") contains, which is what the first version of the test
#: below checked. A guarantee stated as "none of these four substrings" is satisfied by every
#: write tool nobody thought to name, so the assertion is membership and the list is the review.
_ALLOWED_TOOLS = {
    "pack_summary",
    "list_files",
    "read_file",
    "search",
    "validate",
    "dry_run",
    "read_skill",
}


def test_the_assistant_offers_read_skill_and_still_has_no_write_tool():
    """The no-write guarantee is a fact about this list, so adding a tool has to be checked."""
    names = {t["function"]["name"] for t in pack_assistant._tool_schemas()}
    assert "read_skill" in names
    assert names == set(pack_assistant._TOOLS)
    assert names == _ALLOWED_TOOLS


def test_an_invented_tool_name_comes_back_naming_the_real_ones():
    """What the model actually does when it wants to write: it invents a tool.

    Observed live — `edit_file`, in a session that then recovered and proposed a plan. The
    recovery is the assertion: a bare refusal leaves the model guessing a second name, so the
    result has to carry the list. Nothing is written either way, but a session that spends its
    remaining turns hunting for a write tool comes back with no plan, which reads to the
    operator exactly like an endpoint that cannot call tools.
    """
    out = pack_assistant.dispatch_tool("any-pack", "edit_file", {"path": "x.yaml"})
    assert "there is no tool called 'edit_file'" in out
    for real in _ALLOWED_TOOLS - {"read_skill"}:
        assert real in out


def test_read_skill_dispatches_through_the_tool_seam():
    """Every tool result comes back as TEXT, including a wrong name."""
    out = pack_assistant.dispatch_tool("any-pack", "read_skill", {"name": "nope"})
    assert "there is no skill" in out
    real = pack_skills.load()[0].name
    assert real in pack_assistant.dispatch_tool("any-pack", "read_skill", {"name": real})


def test_the_system_prompt_carries_the_index_and_the_original_rules():
    prompt = pack_assistant.system_prompt()
    assert pack_assistant.SYSTEM_PROMPT in prompt
    for skill in pack_skills.load():
        assert skill.name in prompt
