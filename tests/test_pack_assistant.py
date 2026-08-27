"""The pack assistant: the tool loop, the plan guards, and all-or-nothing apply.

No test in this suite calls a model (`tests/CLAUDE.md`). Beyond cost, the loop is what is under
test, and a real model makes the loop's behaviour depend on what it chose to do; every test below
scripts the model's turns exactly, so a failure names a defect in the loop rather than a bad sample.

The assistant has no write tool, so "the model cannot write" is a fact about the call graph rather
than an assertion. What a test can pin is everything downstream of the proposal, each of which has a
way of failing that looks like success:

* a plan applies completely or not at all — a second op failing validation must leave the first op's
  file byte-identical, or the pack is in a state neither the operator's before nor their after
  describes;
* a pre-image anchor is checked and not just a line number, a stale NUMBER still pointing at a line
  so the write succeeds at the wrong place;
* an unknown `op` is skipped rather than raised, a model inventing `rename` not destroying a
  proposal whose other ops are good;
* a truncated read says so, since a model that cannot tell it received a fragment proposes line
  numbers correct for text it never saw;
* exhausting a budget is announced to the model, a plan built on partial exploration being a
  different artifact from one built on a complete reading;
* a tool fault is a tool RESULT, a raise on a mistyped path ending a session that may already have
  found what it needed.

Every test runs against a COPY of the shipped packs under `tmp_path`. Nothing here writes into
`knowledge/`.
"""

import shutil
from pathlib import Path

import pytest

from src.knowledge import (
    pack_assistant,
    pack_dry_run,
    pack_skills,
    pack_store,
    pack_validate,
)
from src.knowledge.pack_assistant import EditOp, EditPlan, PackAssistant
from src.utils.paths import REPO_ROOT
from tests.installed_packs import FIXTURE_PACK, SCALE_PACK, installed_packs

PACK = FIXTURE_PACK
CATALOG = "source_catalog.yaml"


@pytest.fixture
def packs(tmp_path, monkeypatch):
    """A writable copy of the shipped packs, with the store pointed at it.

    `knowledge_pack_dir` is the seam every path in `pack_store` resolves through — and
    `packs_root()` is repo-anchored and ignores `AFIR_DATA_DIR`, so this redirect is the
    only thing between a test and the real pack. Asserted rather than assumed.
    """
    root = tmp_path / "knowledge"
    root.mkdir()
    for name in installed_packs():
        shutil.copytree(REPO_ROOT / "knowledge" / name, root / name)
    monkeypatch.setattr(pack_store, "knowledge_pack_dir", lambda name: root / name)
    assert pack_store.packs_root() == root
    return root


@pytest.fixture(autouse=True)
def clean_sessions():
    """Sessions live in a module-level dict; a leak between tests would be shared state."""
    pack_assistant._SESSIONS.clear()
    yield
    pack_assistant._SESSIONS.clear()


# --------------------------------------------------------------------- the fake client


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, ident, name, arguments):
        self.id = ident
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls
        self.content = content


class FakeClient:
    """Scripted turns. `turns` is a list of FakeMessage, or an exception to raise.

    Records every `messages` list it was handed, because half of what this file asserts is
    about what the loop SENDS — that a tool result carries the matching `tool_call_id`, that
    the budget notice really reaches the model, that the plan call includes the trail.
    """

    def __init__(self, turns=None, plan=None, plan_error=None):
        self.turns = list(turns or [])
        self.plan = plan
        self.plan_error = plan_error
        self.tool_call_messages = []
        self.plan_messages = []

    async def tool_call(
        self, messages, tools, tool_choice="auto", rag=None, stage=None
    ):
        self.tool_call_messages.append(list(messages))
        if not self.turns:
            return FakeMessage(content="done")
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn

    async def structured_output(
        self, messages, response_model, rag=None, max_tokens=None, stage=None
    ):
        self.plan_messages.append(list(messages))
        if self.plan_error is not None:
            raise self.plan_error
        return self.plan if self.plan is not None else EditPlan()


def call(name, arguments, ident="c1"):
    return FakeToolCall(ident, name, arguments)


def session_for(pack=PACK, question="add a check"):
    return pack_assistant.new_session(pack, question)


# ------------------------------------------------------------------------ pack_summary


def test_the_summary_names_what_the_model_must_not_invent(packs):
    """`pack_summary` is the mandatory first call, and this is why.

    Only the kinds the evaluator dispatches on are ever evaluated: any other `kind` falls
    straight through and the condition is never assessed at all — which in the final report
    is indistinguishable from a source that returned no rows. The names in this payload are
    the model's whole defence against inventing one.
    """
    summary = pack_assistant.pack_summary(PACK)
    assert summary["pack"] == PACK
    assert summary[
        "entity_types"
    ], "a pack without entity types cannot be edited usefully"
    assert summary["source_names"]
    assert summary["condition_kinds"] == sorted(
        pack_assistant.pack_validate.condition_kinds()
    )
    # `stub` is in the list and `databricks_table` is not: the first IS a condition kind,
    # the second is a RAG source kind that a naive grep over src/ would have swept up.
    assert "stub" in summary["condition_kinds"]
    assert "databricks_table" not in summary["condition_kinds"]
    assert summary["expected_label_kinds"], "three evaluators honour it; the rest no-op"


def test_the_summary_reads_the_yaml_directly_and_not_through_the_loader(packs):
    """The loader swallows a parse error and hands back an empty document.

    A summary built on it would describe a BROKEN pack as an EMPTY one, and an assistant
    told a pack has no sources proposes adding the thirty already there. So the unreadable
    file is named as unreadable instead.
    """
    (packs / PACK / CATALOG).write_text("sources:\n  - name: a\n   bad: indent\n")
    summary = pack_assistant.pack_summary(PACK)
    assert summary["source_names"] == []
    assert any(CATALOG in u for u in summary.get("unreadable") or [])


# ------------------------------------------------------------------------- the tools


def test_a_read_says_when_it_returned_a_fragment(packs, monkeypatch):
    """Truncation is MARKED, never silent.

    A model that cannot see it received part of a file will propose an edit against the
    part — and the line numbers in that proposal are perfectly plausible for text it never
    read. The marker is what turns a wrong edit into a second, narrower read.
    """
    monkeypatch.setattr(pack_assistant, "READ_CHARS_PER_CALL", 200)
    out = pack_assistant.dispatch_tool(PACK, "read_file", {"path": CATALOG})
    assert "TRUNCATED at line" in out
    assert "of" in out.split("TRUNCATED at line", 1)[1]


def test_a_read_is_line_numbered_from_the_requested_start(packs):
    """The numbers in a read are the numbers an op has to use.

    If a ranged read restarted its numbering at 1, every patch proposed from it would be
    off by exactly the offset — and would still apply cleanly, somewhere else.
    """
    out = pack_assistant.dispatch_tool(
        PACK, "read_file", {"path": CATALOG, "start_line": 5, "end_line": 7}
    )
    body = out.split("\n", 2)[2]
    assert body.splitlines()[0].split("\t")[0].strip() == "5"
    assert len(body.splitlines()) == 3


def test_a_bad_path_comes_back_as_a_tool_result_not_an_exception(packs):
    """A mistyped path is ordinary model behaviour, and the fix is to say so.

    A raise here ends a session that may already have read everything it needed. Traversal
    is included deliberately: the store REJECTS rather than sanitises, and that refusal must
    reach the model as text it can act on.
    """
    for bad in ("does/not/exist.yaml", "../../etc/passwd", ".history/index.json"):
        out = pack_assistant.dispatch_tool(PACK, "read_file", {"path": bad})
        assert out, bad
        assert "refused" in out or "failed" in out or "no such file" in out


def test_an_unknown_tool_name_is_reported_with_the_real_names(packs):
    """The model chose from a list it was given, so a name outside it is worth reporting."""
    out = pack_assistant.dispatch_tool(PACK, "write_file", {"path": CATALOG})
    assert "no tool called" in out
    assert "read_file" in out and "search" in out
    # And the point of the whole design: there is no write tool to have called.
    assert "write" not in sorted(pack_assistant._TOOLS)


def test_search_bounds_its_hits_and_says_it_stopped(packs, monkeypatch):
    monkeypatch.setattr(pack_assistant, "MAX_SEARCH_HITS", 3)
    out = pack_assistant.dispatch_tool(PACK, "search", {"pattern": "name"})
    assert len(out.splitlines()) == 4  # 3 hits + the notice
    assert "stopped at 3 hits" in out


def test_a_broken_regex_is_answered_not_raised(packs):
    out = pack_assistant.dispatch_tool(PACK, "search", {"pattern": "([unclosed"})
    assert "not a valid regular expression" in out


def test_validate_reports_the_pack_as_the_checker_sees_it(packs):
    out = pack_assistant.dispatch_tool(PACK, "validate", {})
    assert "error(s)" in out and "warning(s)" in out


# --------------------------------------------------------------------------- the loop


async def test_the_loop_feeds_each_result_back_under_its_own_call_id(packs):
    """A tool result is matched to its call by id, and nothing else.

    Two calls in one turn with the ids crossed would have the model reading the answer to
    the other question — plausibly, and with no error anywhere.
    """
    client = FakeClient(
        turns=[
            FakeMessage(
                tool_calls=[
                    call("pack_summary", "{}", ident="a1"),
                    call("list_files", '{"glob": "shared/*"}', ident="a2"),
                ]
            ),
            FakeMessage(content="ready"),
        ],
        plan=EditPlan(summary="nothing to do"),
    )
    session = session_for()
    await PackAssistant(client).run(session)
    sent = client.plan_messages[-1]
    tool_turns = [m for m in sent if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_turns] == ["a1", "a2"]
    assert "entity_types" in tool_turns[0]["content"]
    assert session.status == "proposed"
    assert [t["tool"] for t in session.trail] == ["pack_summary", "list_files"]


async def test_the_turn_budget_tells_the_model_instead_of_stopping_quietly(
    packs, monkeypatch
):
    """Exhausting a budget is a fact the plan has to be produced in knowledge of.

    A loop that just stopped and asked for a plan would get one that reads exactly like a
    plan from complete exploration. The operator cannot tell those apart from the diff, so
    the model is told, and asked to put what it could not check in `questions`.
    """
    monkeypatch.setattr(pack_assistant, "MAX_TURNS", 2)
    client = FakeClient(
        turns=[
            FakeMessage(tool_calls=[call("pack_summary", "{}")]),
            FakeMessage(tool_calls=[call("validate", "{}")]),
            FakeMessage(tool_calls=[call("validate", "{}")]),
        ],
        plan=EditPlan(summary="partial", questions=["could not read the ruleset"]),
    )
    session = session_for()
    await PackAssistant(client, max_turns=2).run(session)
    assert session.turns == 2
    assert "all 2 exploration turns" in session.budget_spent
    notice = [
        m
        for m in client.plan_messages[-1]
        if m.get("role") == "user" and "budget is spent" in str(m.get("content"))
    ]
    assert notice, "the model must be told its budget ran out"
    assert "propose from what you have" not in session.question


async def test_the_byte_budget_stops_the_loop_too(packs, monkeypatch):
    """Three of the shipped schema files are ~456 KB each; two reads blow any context."""
    client = FakeClient(
        turns=[
            FakeMessage(tool_calls=[call("read_file", '{"path": "' + CATALOG + '"}')])
        ]
        * 4,
        plan=EditPlan(summary="partial"),
    )
    session = session_for()
    await PackAssistant(client, max_tool_bytes=50).run(session)
    assert session.turns == 1
    assert "bytes" in session.budget_spent
    assert session.tool_bytes >= 50


async def test_no_tool_support_degrades_visibly_and_still_returns_a_plan(packs):
    """An endpoint that cannot call tools must not mean no assistant at all.

    But the degradation is REPORTED — `tool_mode` is rendered in the UI — because a
    proposal written from a pre-loaded summary and one written after reading the files are
    different things, and only one of them checked the line numbers.
    """
    client = FakeClient(
        turns=[RuntimeError("tools are not supported by this endpoint")],
        plan=EditPlan(summary="from the bundle"),
    )
    session = session_for()
    await PackAssistant(client).run(session)
    assert session.status == "proposed"
    assert session.tool_mode == "single_shot"
    bundle = str(client.plan_messages[-1][-2]["content"])
    assert "SUMMARY:" in bundle and "CHECKER:" in bundle and "FILES:" in bundle
    assert any(e["type"] == "assist_note" for e in session.events)


async def test_a_model_that_asks_for_nothing_gets_the_pack_anyway(packs):
    """Offered tools and used none: not an error, but the plan would be written blind."""
    client = FakeClient(
        turns=[FakeMessage(content="I will just guess")], plan=EditPlan()
    )
    session = session_for()
    await PackAssistant(client).run(session)
    assert session.tool_mode == "single_shot"
    assert "SUMMARY:" in str(client.plan_messages[-1][-2]["content"])


async def test_the_focus_files_are_preloaded_in_the_fallback(packs):
    """The operator pointed at a file; a fallback that ignores it wastes the one hint."""
    client = FakeClient(turns=[FakeMessage(content="")], plan=EditPlan())
    session = session_for()
    await PackAssistant(client).run(session, focus=[CATALOG])
    bundle = str(client.plan_messages[-1][-2]["content"])
    assert CATALOG in bundle
    assert session.focus == [CATALOG]


async def test_a_failure_lands_on_the_session_and_never_raises(packs):
    """The caller is an HTTP handler that already answered 202.

    An exception escaping `run` surfaces only in the log while the UI waits on a session
    that will never finish — the failure mode is a spinner, not an error.
    """
    client = FakeClient(
        turns=[FakeMessage(content="")], plan_error=RuntimeError("endpoint refused")
    )
    session = session_for()
    await PackAssistant(client).run(session)
    assert session.status == "failed"
    assert "endpoint refused" in session.error
    assert any(
        e.get("status") == "failed"
        for e in session.events
        if e["type"] == "assist_status"
    )


async def test_a_rejection_carries_the_prior_plan_and_the_correction(packs):
    """A paraphrase loses which op was wrong, so the rejected plan is quoted verbatim."""
    client = FakeClient(turns=[FakeMessage(content="")], plan=EditPlan())
    session = session_for()
    await PackAssistant(client).run(
        session,
        guidance="wrong ruleset — it belongs in the shared library",
        prior_plan={"summary": "put it in the ruleset", "ops": []},
    )
    opening = str(client.plan_messages[-1][1]["content"])
    assert "was REJECTED" in opening
    assert "put it in the ruleset" in opening
    assert "shared library" in opening


# ---------------------------------------------------------------- the plan guards


def _existing_line(packs, rel, n, pack=PACK):
    return (packs / pack / rel).read_text().splitlines()[n - 1]


def patch_op(packs, rel, line, text, pack=PACK):
    """A patch op that replaces exactly line `line`, with its real anchors."""
    current = _existing_line(packs, rel, line, pack)
    return EditOp(
        op="patch",
        path=rel,
        text=text,
        start_line=line,
        end_line=line,
        expect_first_line=current,
        expect_last_line=current,
        reason="test",
    )


def test_an_unknown_op_is_skipped_and_the_good_ones_survive(packs):
    """A model inventing `rename` must not cost the operator three correct edits."""
    plan = EditPlan(
        ops=[
            EditOp(op="rename", path="a.yaml", text="x"),
            patch_op(packs, CATALOG, 1, "# a new first line"),
        ]
    )
    preview = pack_assistant.plan_preview(PACK, plan)
    assert preview["skipped"] and "rename" in preview["skipped"][0]
    assert preview["errors"] == []
    assert [o["path"] for o in preview["ops"]] == [CATALOG]


def test_a_patch_without_anchors_is_refused(packs):
    """A line number alone is the dangerous form of staleness: it still points at a line.

    So a patch missing its pre-image anchors is refused outright rather than applied to
    whatever happens to be there — the write would succeed and land in the wrong place.
    """
    plan = EditPlan(
        ops=[EditOp(op="patch", path=CATALOG, text="# x", start_line=1, end_line=1)]
    )
    preview = pack_assistant.plan_preview(PACK, plan)
    assert preview["errors"]
    assert "expect_first_line" in preview["errors"][0]


def test_a_moved_anchor_names_both_the_expected_and_the_actual_text(packs):
    """The operator's next move depends on knowing WHAT is there instead."""
    op = patch_op(packs, CATALOG, 1, "# x")
    op.expect_first_line = "# something else entirely"
    op.expect_last_line = "# something else entirely"
    preview = pack_assistant.plan_preview(PACK, EditPlan(ops=[op]))
    assert preview["errors"]
    message = preview["errors"][0]
    assert "something else entirely" in message
    assert "moved since the plan was made" in message


def test_a_line_range_outside_the_file_is_refused_with_its_bounds(packs):
    op = patch_op(packs, CATALOG, 1, "# x")
    op.start_line, op.end_line = 90_000, 90_001
    preview = pack_assistant.plan_preview(PACK, EditPlan(ops=[op]))
    assert preview["errors"] and "outside the file" in preview["errors"][0]


def test_an_edit_that_would_make_the_file_load_as_empty_is_refused(packs):
    """THE defect this whole editor exists to prevent, reached through the assistant.

    A catalog that parses to nothing loads the pack with zero sources; every condition then
    resolves to unknown and the report reads as the data having nothing to say. The store
    owns this check — this asserts the assistant's preview routes into it, before any file
    is touched.
    """
    lines = (packs / PACK / CATALOG).read_text().splitlines()
    op = EditOp(
        op="patch",
        path=CATALOG,
        text="# every declaration replaced by a comment",
        start_line=1,
        end_line=len(lines),
        expect_first_line=lines[0],
        expect_last_line=lines[-1],
    )
    preview = pack_assistant.plan_preview(PACK, EditPlan(ops=[op]))
    assert preview["errors"]
    assert "empty" in preview["errors"][0]


def test_a_delete_is_refused_unless_it_was_separately_allowed(packs):
    plan = EditPlan(ops=[EditOp(op="delete", path=CATALOG)])
    blocked = pack_assistant.plan_preview(PACK, plan)
    assert blocked["errors"] and "deletion box" in blocked["errors"][0]
    allowed = pack_assistant.plan_preview(PACK, plan, allow_delete=True)
    assert allowed["errors"] == []
    assert (packs / PACK / CATALOG).is_file(), "a preview must never write or unlink"


def test_a_create_over_an_existing_file_is_refused_as_a_create(packs):
    """ "Add" and "save" are different intents, and conflating them overwrites silently."""
    plan = EditPlan(ops=[EditOp(op="create", path=CATALOG, text="sources: []")])
    preview = pack_assistant.plan_preview(PACK, plan)
    assert preview["errors"] and "already exists" in preview["errors"][0]


def test_a_patch_of_a_missing_file_points_at_create(packs):
    plan = EditPlan(
        ops=[
            EditOp(
                op="patch",
                path="shared/checks/nope.yaml",
                text="x",
                start_line=1,
                end_line=1,
                expect_first_line="a",
                expect_last_line="a",
            )
        ]
    )
    preview = pack_assistant.plan_preview(PACK, plan)
    assert preview["errors"] and "`create`" in preview["errors"][0]


def test_a_non_editable_suffix_is_refused(packs):
    plan = EditPlan(ops=[EditOp(op="create", path="notes.pdf", text="x")])
    preview = pack_assistant.plan_preview(PACK, plan)
    assert preview["errors"]


def test_the_preview_carries_a_server_rendered_diff(packs):
    """Rendered here so the UI never computes one of its own.

    Two implementations that disagree produce a diff that is not what gets written — and
    the diff is the entire basis on which the change was approved.
    """
    plan = EditPlan(ops=[patch_op(packs, CATALOG, 1, "# a brand new first line")])
    preview = pack_assistant.plan_preview(PACK, plan)
    diff = preview["ops"][0]["diff"]
    assert diff.startswith("--- a/" + CATALOG)
    assert "+# a brand new first line" in diff
    assert preview["ops"][0]["bytes_after"] != preview["ops"][0]["bytes_before"]


# ------------------------------------------------------------------ apply: all or none


def test_an_invalid_second_op_leaves_the_first_files_bytes_untouched(packs):
    """ALL OR NOTHING, asserted on the BYTES rather than on the response.

    Half-applied, the pack is in a state neither the before nor the after describes, and
    the diff the operator approved names neither of them.
    """
    target = packs / PACK / CATALOG
    before = target.read_bytes()
    plan = EditPlan(
        ops=[
            patch_op(packs, CATALOG, 1, "# this one is fine"),
            EditOp(
                op="delete", path="entity_glossary.yaml"
            ),  # refused: no allow_delete
        ]
    )
    result = pack_assistant.apply_plan(PACK, plan)
    assert result["applied"] is False
    assert result["errors"]
    assert target.read_bytes() == before
    assert (packs / PACK / "entity_glossary.yaml").is_file()


def test_a_valid_plan_writes_every_op_and_keeps_the_prior_version(packs):
    """Nothing is lost: every write snapshots first, which is what makes any of this safe."""
    plan = EditPlan(
        summary="two edits",
        ops=[
            patch_op(packs, CATALOG, 1, "# edited by the assistant"),
            EditOp(
                op="create",
                path="shared/concepts/assist_note.md",
                text="# a note\n\nSomething worth writing down.\n",
            ),
        ],
    )
    result = pack_assistant.apply_plan(PACK, plan, actor="tester", session="assist-1")
    assert result["applied"] is True
    assert len(result["written"]) == 2
    assert (packs / PACK / CATALOG).read_text().startswith("# edited by the assistant")
    assert (packs / PACK / "shared/concepts/assist_note.md").is_file()
    entries = pack_store.history(PACK, CATALOG)
    assert entries and entries[0]["reason"] == "assist"
    assert pack_store.snapshot_text(PACK, entries[0]["id"]).startswith("#")


def test_a_patch_keeps_the_comments_and_the_anchors_around_it(packs):
    """The reason patches exist at all.

    Run against the LARGEST installed pack, not the engine's own fixture, and that choice is
    the test: the anchor half is vacuous over a catalog with no anchors in it, and the
    fixture pack's has zero. One deployed catalog carries 17 `&` definitions, 198 `*`
    references and 1218 comment lines across 3153 lines — a whole-file rewrite through
    safe_load/dump loses every one, and a dropped anchor makes the WHOLE file load empty. So
    this asserts surviving bytes.

    Skipping when no such pack is installed is honest but silent, which is why
    `test_pack_store.py` also asserts the anchor guarantee over a catalog it writes itself.
    """
    pack = SCALE_PACK
    if pack is None:
        pytest.skip("no pack larger than the fixture one is installed")
    target = packs / pack / CATALOG
    original = target.read_text()
    anchors, aliases = original.count("&"), original.count("*")
    if not (anchors and aliases):
        pytest.skip("the largest installed pack's catalog uses no anchors")
    comments = sum(1 for line in original.splitlines() if line.lstrip().startswith("#"))
    line_no = next(
        i for i, line in enumerate(original.splitlines(), 1) if line.startswith("#")
    )
    plan = EditPlan(
        ops=[patch_op(packs, CATALOG, line_no, "# a replaced comment", pack=pack)]
    )
    assert pack_assistant.apply_plan(pack, plan)["applied"] is True
    after = target.read_text()
    assert after.count("&") == anchors
    assert after.count("*") == aliases
    assert (
        sum(1 for line in after.splitlines() if line.lstrip().startswith("#"))
        == comments
    )
    assert "# a replaced comment" in after
    # And exactly one line differs, out of the whole file.
    differing = [
        (a, b) for a, b in zip(original.splitlines(), after.splitlines()) if a != b
    ]
    assert len(differing) == 1
    assert len(after.splitlines()) == len(original.splitlines())


def test_an_allowed_delete_keeps_the_content_recoverable(packs):
    rel = "shared/concepts"
    victim = next((packs / PACK / rel).glob("*.md"), None)
    if victim is None:
        pytest.skip("this pack ships no shared concept to delete")
    relpath = f"{rel}/{victim.name}"
    before = victim.read_text()
    plan = EditPlan(ops=[EditOp(op="delete", path=relpath)])
    result = pack_assistant.apply_plan(PACK, plan, allow_delete=True)
    assert result["applied"] is True
    assert not victim.exists()
    entry = pack_store.history(PACK, relpath)[0]
    assert pack_store.snapshot_text(PACK, entry["id"]) == before


def test_apply_accepts_a_plain_dict_plan(packs):
    """ "Edit first" arrives over HTTP as JSON, and runs through the identical guards.

    An operator who must accept a proposal verbatim or reject it will accept a wrong one —
    so the edited plan is a first-class input, not a bypass.
    """
    current = _existing_line(packs, CATALOG, 1)
    plan = {
        "summary": "hand-edited",
        "ops": [
            {
                "op": "patch",
                "path": CATALOG,
                "text": "# hand-edited by the operator",
                "start_line": 1,
                "end_line": 1,
                "expect_first_line": current,
                "expect_last_line": current,
            }
        ],
    }
    assert pack_assistant.apply_plan(PACK, plan)["applied"] is True
    assert (packs / PACK / CATALOG).read_text().startswith("# hand-edited")


# ------------------------------------------------------------------------- sessions


def test_sessions_are_capped_and_evict_the_oldest(packs, monkeypatch):
    """A plan is only meaningful against the snapshot it was computed from.

    So there is nothing to gain from keeping more sessions — and something to lose: a stale
    plan applied to files that moved underneath it.
    """
    monkeypatch.setattr(pack_assistant, "MAX_SESSIONS", 3)
    ids = [session_for(question=f"q{i}").id for i in range(5)]
    live = [s["session"] for s in pack_assistant.list_sessions()]
    assert len(live) == 3
    assert ids[0] not in live and ids[-1] in live


def test_the_session_list_omits_the_bulk_and_the_detail_carries_it(packs):
    """An index is for choosing; the plan and the trail are what a detail read is for."""
    session = session_for()
    session.plan = EditPlan(
        summary="s", ops=[EditOp(op="create", path="a.md", text="x")]
    )
    session.trail.append({"turn": 1, "tool": "validate"})
    listed = pack_assistant.list_sessions(PACK)[0]
    assert "plan" not in listed and "trail" not in listed
    assert listed["session"] == session.id
    assert session.snapshot()["plan"]["summary"] == "s"


async def test_a_late_subscriber_still_sees_every_earlier_event(packs):
    """Which files were read is how an operator judges the proposal.

    Replayed rather than tailed-only, the same as the job stream: a tool round that
    happened but is invisible reads as a model that did nothing.
    """
    client = FakeClient(
        turns=[
            FakeMessage(tool_calls=[call("validate", "{}")]),
            FakeMessage(content=""),
        ],
        plan=EditPlan(summary="done"),
    )
    session = session_for()
    await PackAssistant(client).run(session)
    seen = [e async for e in pack_assistant.subscribe(session.id)]
    assert [e["type"] for e in seen][0] == "assist_status"
    assert any(e["type"] == "assist_tool" and e["tool"] == "validate" for e in seen)
    assert seen[-1]["status"] == "proposed"


async def test_subscribing_to_an_unknown_session_ends_quietly(packs):
    assert [e async for e in pack_assistant.subscribe("assist-nope")] == []


# ------------------------------------------------------------------ the method library
#
# The library itself is tested in `test_pack_skills.py`; the WIRING is what is asserted here. A
# skill selected and never reaching a turn produces a proposal written without it, which reads
# exactly like one written with it.


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A synthetic library in place of the shipped one.

    Synthetic on purpose: asserting on the real documents' wording would make every test here
    a hostage to editing them, and the wiring is what is under test. One test below does use
    the real library, for the one claim only it can make.
    """
    root = tmp_path / "skills"
    root.mkdir()
    (root / "aaa-spine.md").write_text(
        '---\nname: aaa-spine\ntitle: "S"\nwhen: "always"\nalways: true\n---\n\n'
        "SPINE-BODY: measure before you declare.\n",
        encoding="utf-8",
    )
    (root / "bbb-binding.md").write_text(
        '---\nname: bbb-binding\ntitle: "B"\nwhen: "binding"\n'
        "triggers:\n  - projection\n---\n\nBINDING-BODY: check the column exists.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pack_skills, "SKILLS_DIR", root)
    return root


async def test_the_selected_skills_reach_the_first_user_turn(packs, library):
    """Injected into the request, not offered as a tool the model may decline.

    Asserted on the message the client was HANDED, because that is the only place the claim
    is checkable: `session.skill_text` being populated says the selection ran, not that the
    model saw it.
    """
    client = FakeClient(turns=[FakeMessage(content="")], plan=EditPlan())
    session = session_for(question="fix the projection on a source")
    await PackAssistant(client).run(session)
    first = client.tool_call_messages[0]
    opening = str(first[-1]["content"])
    assert "SPINE-BODY" in opening, "the always-on skill did not reach the model"
    assert "BINDING-BODY" in opening, "the triggered skill did not reach the model"
    assert opening.index("SPINE-BODY") < opening.index("BINDING-BODY")


async def test_the_skill_index_is_in_the_system_turn_and_the_original_rules_survive(
    packs, library
):
    """The index is what makes `read_skill` usable: a name the model cannot see it cannot ask
    for. Built from the directory rather than written into the prompt literal, so installing a
    skill needs no code change — and the pre-existing rules must still be there, since a
    prompt that gained a catalogue and lost its instructions is a worse assistant."""
    client = FakeClient(turns=[FakeMessage(content="")], plan=EditPlan())
    await PackAssistant(client).run(session_for())
    system = str(client.tool_call_messages[0][0]["content"])
    assert "bbb-binding" in system and "aaa-spine" in system
    assert pack_assistant.SYSTEM_PROMPT in system


async def test_the_skills_survive_the_endpoint_that_cannot_call_tools(packs, library):
    """The degraded path is where method knowledge matters MOST, and it was free to lose.

    `single_shot` proposes from a pre-loaded bundle with no exploration at all, so the rules
    about measuring before declaring are the only thing standing between that and a plan of
    invented values.

    The bundle is its own user turn appended AFTER the opening one, so what this asserts is
    that both are in the same request and the skills come first — the model reads the method
    before the material. Asserted across the whole message list rather than on one index,
    because which turn the bundle lands in is the fallback's business and not this claim's.
    """
    client = FakeClient(
        turns=[RuntimeError("tools are not supported by this endpoint")],
        plan=EditPlan(summary="from the bundle"),
    )
    session = session_for(question="fix the projection")
    await PackAssistant(client).run(session)
    assert session.tool_mode == "single_shot"
    request = client.plan_messages[-1]
    text = "\n".join(str(m.get("content")) for m in request)
    assert "SPINE-BODY" in text and "BINDING-BODY" in text
    assert "SUMMARY:" in text
    assert text.index("SPINE-BODY") < text.index("SUMMARY:")


async def test_the_applied_skills_are_named_on_the_session_and_in_an_event(packs, library):
    """Which method the proposal was written under is part of judging the proposal.

    On the snapshot AND as an event, because they answer to different readers: the snapshot is
    what a late subscriber and the panel render, the event is what an operator watching the run
    sees at the moment it happens.
    """
    client = FakeClient(turns=[FakeMessage(content="")], plan=EditPlan())
    session = session_for(question="fix the projection")
    await PackAssistant(client).run(session)
    names = [s["name"] for s in session.snapshot()["skills"]]
    assert names == ["aaa-spine", "bbb-binding"]
    assert all(s["chars"] > 0 for s in session.snapshot()["skills"])
    notes = [e for e in session.events if e["type"] == "assist_note"]
    assert any("bbb-binding" in str(e.get("message", "")) for e in notes)


async def test_the_operators_correction_is_matched_too(packs, library):
    """A rejection is where the real subject of the change usually first gets named.

    The question here mentions nothing the library triggers on; the correction does. Selecting
    on the question alone means the second attempt is made with less method than it needs,
    which is the attempt that already went wrong once.
    """
    client = FakeClient(turns=[FakeMessage(content="")], plan=EditPlan())
    session = session_for(question="change that other thing")
    await PackAssistant(client).run(
        session, guidance="you got the projection wrong"
    )
    assert [s["name"] for s in session.skills] == ["aaa-spine", "bbb-binding"]


async def test_a_skill_that_cannot_be_loaded_is_reported_on_the_session(packs, library):
    """Reported, not silent — the whole reason `problems()` exists.

    A malformed document is skipped so one bad file cannot take the assistant down, and that
    is the right behaviour AND an invisible one: the visible symptom is a proposal that quietly
    stopped knowing something. Same posture as `tool_mode` and `image_mode`.
    """
    (library / "broken.md").write_text("no frontmatter here\n", encoding="utf-8")
    client = FakeClient(turns=[FakeMessage(content="")], plan=EditPlan())
    session = session_for()
    await PackAssistant(client).run(session)
    assert session.status == "proposed", "one bad skill must not fail the session"
    assert any(
        "broken.md" in str(e.get("message", ""))
        for e in session.events
        if e["type"] == "assist_note"
    )


async def test_no_library_installed_behaves_as_it_did_before_one_existed(
    packs, tmp_path, monkeypatch
):
    """A deployment may ship none. The turn must then carry no skill scaffolding at all."""
    monkeypatch.setattr(pack_skills, "SKILLS_DIR", tmp_path / "absent")
    client = FakeClient(turns=[FakeMessage(content="")], plan=EditPlan())
    session = session_for()
    await PackAssistant(client).run(session)
    assert session.status == "proposed"
    assert session.skills == []
    opening = str(client.tool_call_messages[0][-1]["content"])
    assert "METHOD KNOWLEDGE" not in opening


async def test_read_skill_is_dispatchable_inside_the_loop(packs, library):
    """The widening path: a skill selection did not reach, asked for by name mid-session.

    Through the same tool seam as every other call, so it lands in the trail the operator
    reads and in `session.files_read`-style accounting rather than happening invisibly.
    """
    client = FakeClient(
        turns=[
            FakeMessage(
                tool_calls=[call("read_skill", '{"name": "bbb-binding"}', "s1")]
            ),
            FakeMessage(content=""),
        ],
        plan=EditPlan(),
    )
    session = session_for(question="rename a heading")
    await PackAssistant(client).run(session)
    result = [
        m
        for m in client.tool_call_messages[-1]
        if m.get("role") == "tool" and m.get("tool_call_id") == "s1"
    ]
    assert result and "BINDING-BODY" in str(result[0]["content"])
    assert any(
        e["type"] == "assist_tool" and e["tool"] == "read_skill" for e in session.events
    )


# ------------------------------------------------------------- the after-state checks
#
# `plan_checks` measures the pack the plan would LEAVE, in a candidate tree, and the apply is
# gated on it. Three properties, each of which fails silently the other way:
#
# * the gate is a DELTA. A pack being worked on normally carries validation errors, so gating
#   on the candidate's total makes the very commit that fixes the first one unappliable — the
#   editor would refuse every plan for a pack that needs one.
# * error identity excludes the LINE. A patch shifts every line below it, so keying a
#   diagnostic on its line reports a file's pre-existing errors as introduced on any edit near
#   the top. That is the negative test this whole design exists for.
# * a measurement that could not be taken must SAY so. Folded into a clean result it reads as
#   a plan that was checked; turned into a refusal it blames the plan for a temp directory.


@pytest.fixture(autouse=True)
def clean_baseline_cache():
    """The validation baseline is a module-level cache, so a leak between tests is shared state.

    Keyed on a tree fingerprint, so a stale entry cannot actually be served to a later test —
    but a test that asserts the baseline was RE-READ can only do so from a known start.
    """
    pack_assistant._BASELINE_CACHE.clear()
    yield
    pack_assistant._BASELINE_CACHE.clear()


@pytest.fixture
def dry_run_calls(monkeypatch):
    """`plan_checks`'s dry run captured rather than run.

    The real one reads the deployment's job history through `job_store`, which a unit test
    must not depend on; what these tests assert about it is its SCOPE and which tree it was
    pointed at. `pack_dry_run` has its own file.

    The candidate tree's contents are read HERE and not by the test, because the tree is a
    context manager: by the time `plan_checks` has returned there is nothing left to read.
    """
    calls = []

    def fake_dry_run(pack_dir, *, ruleset_keys=None, **kwargs):
        root = Path(pack_dir)
        calls.append(
            {
                "dir": root,
                "keys": ruleset_keys,
                "kwargs": kwargs,
                "files": {
                    str(p.relative_to(root)): p.read_text(errors="replace")
                    for p in sorted(root.rglob("*"))
                    if p.is_file()
                },
            }
        )
        return pack_dry_run.DryRunReport(pack=root.name)

    monkeypatch.setattr(pack_dry_run, "dry_run", fake_dry_run)
    return calls


def _a_ruleset_with_an_inline_kind(packs, pack=PACK):
    """A use-case ruleset that declares a `kind:` inline, found rather than named.

    Named, the path would be an assertion about which procedures a pack happens to ship —
    and a ruleset built entirely of `use:` imports declares no kind to break.
    """
    for path in sorted((packs / pack / "use_cases").glob("*/rules.yaml")):
        lines = path.read_text().splitlines()
        hits = [i for i, line in enumerate(lines) if line.strip().startswith("kind:")]
        if hits:
            rel = str(path.relative_to(packs / pack))
            return rel, path.parent.name, hits[0] + 1
    pytest.skip("no installed ruleset declares an inline condition kind")


def _break_kinds(packs, rel, count=1, pack=PACK):
    """Write `count` unknown condition kinds into the pack copy, one error each.

    Written with `write_text` and not through the store on purpose: these are the pack's
    PRE-EXISTING errors, the state a plan arrives into, not something a plan did. Returns
    `[(1-based line, the text that was there)]` so a plan can put one of them back.
    """
    path = packs / pack / rel
    lines = path.read_text().splitlines()
    hits = [i for i, line in enumerate(lines) if line.strip().startswith("kind:")]
    assert len(hits) >= count
    broken = []
    for i in hits[:count]:
        broken.append((i + 1, lines[i]))
        lines[i] = f"{lines[i].split('kind:')[0]}kind: no_such_kind_{i}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return broken


def _bogus_kind_op(packs, rel, line, pack=PACK):
    """A patch op replacing a real condition kind with one the evaluator does not dispatch.

    The indent comes off the line being replaced: re-indented, the op would be a YAML error
    rather than the one unknown-kind error the test is measuring.
    """
    current = _existing_line(packs, rel, line, pack)
    return patch_op(
        packs, rel, line, f"{current.split('kind:')[0]}kind: no_such_kind", pack
    )


def _insert_comment_op(packs, rel, comment, pack=PACK):
    """A patch op that keeps line 1 and adds a line under it, shifting the rest down.

    Keeping the line matters: replacing it would change the document rather than move it,
    and then a shifted diagnostic could not be told from one the plan caused.
    """
    return patch_op(
        packs, rel, 1, f"{_existing_line(packs, rel, 1, pack)}\n{comment}", pack
    )


def _error_lines(packs, pack=PACK):
    return [
        d.get("line")
        for d in pack_validate.validate_pack(packs / pack).get("diagnostics", [])
        if d.get("severity") == "error"
    ]


def test_an_introduced_validation_error_refuses_the_plan_and_writes_nothing(packs):
    """A plan that leaves the pack failing validation is one the next run loads as empty.

    The operator approved a diff, not that outcome — so the refusal is on the after-state and
    the assertion is on the BYTES, the same all-or-nothing rule every other write here takes.
    """
    rel, _key, line = _a_ruleset_with_an_inline_kind(packs)
    target = packs / PACK / rel
    before = target.read_bytes()
    plan = EditPlan(ops=[_bogus_kind_op(packs, rel, line)])
    result = pack_assistant.apply_plan(PACK, plan)
    assert result["applied"] is False
    assert any("no_such_kind" in e for e in result["errors"])
    assert result["checks"]["introduced"] and result["checks"]["baseline_errors"] == 0
    assert target.read_bytes() == before


def test_a_line_shifting_patch_does_not_report_a_pre_existing_error_as_introduced(packs):
    """THE negative test: identity excludes the line, so a shift introduces nothing.

    The pack arrives with one error. The plan adds a line above it, which moves that error's
    reported line — asserted directly, or the test passes on a patch that shifted nothing.
    Keyed on the line, the diagnostic at 146 and the same diagnostic at 147 are two different
    errors and the plan is refused for a defect it did not cause.
    """
    rel, _key, _line = _a_ruleset_with_an_inline_kind(packs)
    _break_kinds(packs, rel)
    was = _error_lines(packs)
    plan = EditPlan(ops=[_insert_comment_op(packs, rel, "# one added line")])
    checks = pack_assistant.plan_checks(PACK, plan, dry_run=False)
    assert checks["ran"] is True
    assert checks["baseline_errors"] == 1
    assert checks["candidate_errors"] == 1
    assert checks["introduced"] == []
    assert checks["resolved"] == []
    assert pack_assistant.apply_plan(PACK, plan, checks=checks)["applied"] is True
    assert _error_lines(packs) == [n + 1 for n in was], "the lines did not actually move"


def test_a_plan_that_fixes_one_error_applies_while_the_pack_still_fails(packs):
    """The delta stated the other way round, and the reason it is a delta at all.

    Two pre-existing errors, one of them fixed: gated on the candidate's total this plan is
    refused, which means the first fix to a broken pack can never be applied through here.
    """
    rel, _key, _line = _a_ruleset_with_an_inline_kind(packs)
    broken = _break_kinds(packs, rel, count=2)
    line, original = broken[0]
    plan = EditPlan(ops=[patch_op(packs, rel, line, original)])
    checks = pack_assistant.plan_checks(PACK, plan, dry_run=False)
    assert checks["baseline_errors"] == 2
    assert checks["candidate_errors"] == 1
    assert checks["introduced"] == []
    assert any("no_such_kind" in r for r in checks["resolved"])
    assert pack_assistant.apply_plan(PACK, plan, checks=checks)["applied"] is True


def test_the_preview_carries_an_introduced_error_as_a_plan_error(packs):
    """The preview is where the operator decides, so the finding has to block THERE.

    Folded into `errors` rather than reported beside them, because that is the one list the
    apply already refuses on — a second channel would be a second answer to the same question.
    The ops still render: a plan an operator cannot read is one they cannot correct.
    """
    rel, _key, line = _a_ruleset_with_an_inline_kind(packs)
    plan = EditPlan(ops=[_bogus_kind_op(packs, rel, line)])
    checks = pack_assistant.plan_checks(PACK, plan, dry_run=False)
    preview = pack_assistant.plan_preview(PACK, plan, checks=checks)
    assert any("no_such_kind" in e for e in preview["errors"])
    assert preview["checks"]["candidate_errors"] == 1
    assert preview["ops"] and preview["ops"][0]["diff"]


def test_a_plan_that_cannot_be_prepared_is_reported_as_unmeasured(packs):
    """There is no after-state to check, and that is not the same as a clean one."""
    op = patch_op(packs, CATALOG, 1, "# x")
    op.expect_first_line = op.expect_last_line = "# something else entirely"
    checks = pack_assistant.plan_checks(PACK, EditPlan(ops=[op]), dry_run=False)
    assert checks["ran"] is False
    assert checks["introduced"] == []
    assert any("could not be prepared" in p for p in checks["problems"])


def test_a_plan_that_changes_no_file_is_reported_as_unmeasured(packs):
    checks = pack_assistant.plan_checks(PACK, EditPlan(), dry_run=False)
    assert checks["ran"] is False
    assert any("changes no file" in p for p in checks["problems"])


def test_a_measurement_that_raises_is_a_problem_and_never_a_refusal(packs, monkeypatch):
    """A plan is not at fault for a temp directory, and `ran: False` is not `introduced`.

    Turned into a refusal, an unwritable /tmp would block every apply; folded into a clean
    result, an unmeasured plan would read as a checked one. It is a third state and says so.
    """
    monkeypatch.setattr(
        pack_assistant.pack_validate,
        "validate_pack",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no space left on device")),
    )
    plan = EditPlan(ops=[patch_op(packs, CATALOG, 1, "# a comment")])
    checks = pack_assistant.plan_checks(PACK, plan, dry_run=False)
    assert checks["ran"] is False
    assert checks["introduced"] == []
    assert any("no space left on device" in p for p in checks["problems"])
    assert any("Nothing here says the plan is sound" in p for p in checks["problems"])
    assert pack_assistant.apply_plan(PACK, plan, checks=checks)["applied"] is True


def test_the_dry_run_is_scoped_to_the_ruleset_the_plan_touches(packs, dry_run_calls):
    """A ruleset key IS a use-case directory name, so the run budget can be spent on it.

    And it is spent against the CANDIDATE tree — the after-state is the whole question — with
    the live pack untouched, since measuring a plan must never write it.
    """
    rel, key, _line = _a_ruleset_with_an_inline_kind(packs)
    live = packs / PACK / rel
    before = live.read_bytes()
    plan = EditPlan(ops=[_insert_comment_op(packs, rel, "# a scoped edit")])
    checks = pack_assistant.plan_checks(PACK, plan)
    assert checks["ran"] is True
    assert checks["rulesets"] == [key]
    assert dry_run_calls[0]["keys"] == [key]
    assert dry_run_calls[0]["dir"] != packs / PACK
    assert "# a scoped edit" in dry_run_calls[0]["files"][rel]
    assert live.read_bytes() == before


def test_a_pack_root_edit_leaves_the_dry_run_unscoped(packs, dry_run_calls):
    """Anything at the pack root is readable by every use case, so the answer is "any".

    Erring toward all of them rather than none: a narrowed dry run that missed the ruleset a
    plan actually changed reports it as unmeasured, which in a preview reads as clean.
    """
    plan = EditPlan(ops=[patch_op(packs, CATALOG, 1, "# a shared edit")])
    checks = pack_assistant.plan_checks(PACK, plan)
    assert checks["rulesets"] is None
    assert dry_run_calls[0]["keys"] is None


def test_the_apply_does_not_spend_a_dry_run_of_its_own(packs, dry_run_calls):
    """The gate is "does this plan break the pack"; the dry run is an authoring read.

    It gates nothing, so computing it here would spend a replay budget on every write to
    produce a number no branch reads.
    """
    plan = EditPlan(ops=[patch_op(packs, CATALOG, 1, "# a comment")])
    assert pack_assistant.apply_plan(PACK, plan)["applied"] is True
    assert dry_run_calls == []


def test_the_baseline_is_re_read_when_the_pack_changed_underneath_it(packs):
    """The cache is keyed on the tree and not on time, so any other writer invalidates it.

    Keyed on time, an edit through the file endpoint between two previews would leave the
    delta computed against a before-state that no longer exists — reporting somebody else's
    error as this plan's.
    """
    plan = EditPlan(ops=[patch_op(packs, CATALOG, 1, "# a comment")])
    first = pack_assistant.plan_checks(PACK, plan, dry_run=False)
    assert first["baseline_errors"] == 0
    rel, _key, _line = _a_ruleset_with_an_inline_kind(packs)
    _break_kinds(packs, rel)
    second = pack_assistant.plan_checks(PACK, plan, dry_run=False)
    assert second["baseline_errors"] == 1
    assert second["introduced"] == []


async def test_the_proposal_arrives_with_its_after_state_already_measured(
    packs, dry_run_calls
):
    """The operator reads the checks in the preview, so the session must carry them.

    Emitted as its own status first, because the measurement costs seconds on a real pack and
    a panel that sits on `exploring` with the model already finished reads as a stalled run.
    """
    rel, _key, _line = _a_ruleset_with_an_inline_kind(packs)
    plan = EditPlan(
        summary="one edit", ops=[patch_op(packs, rel, 1, "# a measured edit")]
    )
    client = FakeClient(turns=[FakeMessage(content="")], plan=plan)
    session = session_for()
    await PackAssistant(client).run(session)
    assert session.status == "proposed"
    checks = session.snapshot()["preview"]["checks"]
    assert checks["ran"] is True and checks["introduced"] == []
    assert [e["status"] for e in session.events if e["type"] == "assist_status"].count(
        "checking"
    ) == 1


async def test_the_shipped_library_really_reaches_a_real_session(packs):
    """The one claim the synthetic fixture cannot make: the DOCUMENTS that ship are wired.

    Without this, renaming the directory or emptying it would leave every test above green —
    they would be asserting the wiring against a library only the tests install.
    """
    client = FakeClient(turns=[FakeMessage(content="")], plan=EditPlan())
    session = session_for(question="bind a new source and write a condition for it")
    await PackAssistant(client).run(session)
    assert len(session.skills) >= 2
    opening = str(client.tool_call_messages[0][-1]["content"])
    for skill in session.skills:
        assert f"--- skill: {skill['name']}" in opening
