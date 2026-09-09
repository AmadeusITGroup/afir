"""The sacred channel, restated for the OTHER advisory lane: an open question may not adjudicate.

`src/inquiry.py` adds, at the end of correlation, a deterministic pass asking of the procedure
that just adjudicated "what could this run NOT settle, and would one further source say something
about it?" — emitting an `InquiryFinding` per declaration whose trigger fired. It is the sibling of
the link lane on the other axis (a link asks whether ANOTHER procedure applies) and it inherits the
link lane's one refusal whole: **a lane that adds no evidence to the verdict must not add confidence
to it either.** An inquiry is addressed to a human. It may not move a label, a severity, a
condition result or a health score by a byte.

Nothing about wiring an inquiry into a condition FAILS. It produces a verdict that is more
confident and differently wrong, with every stage green — and this lane is the more tempting of the
two to wire in, because its findings are about THIS procedure's own subject and read like evidence
the verdict simply has not folded in yet. So the invariant is asserted the same three ways, each
catching a different way of breaking it:

  1. **By value** — the sacred fields serialise identically with the pass off and on, over a pack
     that really declares open questions. Catches an inquiry that feeds a condition or a rollup.
  2. **By identity** — `logs` gains no key and no list is replaced, with a probe in the loop.
     Catches the tempting implementation where a probe's rows are namespaced INTO `logs`: sixteen
     places enumerate that dict, `aggregate` would count a probe's rows toward a real `no_records`,
     and `_merge_co_identified_subjects` would license a merge from a row the verdict never saw.
  3. **By structure** — the call site read as an AST: the pass runs AFTER the verdict and the
     brief, and its return value lands on nothing but an `inquiries` attribute. Catches a future
     edit that threads a question EARLIER, whose effect on this fixture happens to be nil.

And two guards on the guards: `test_the_comparison_can_actually_fail` mutates each sacred field and
demands the comparison object to it, because byte-equality between two runs of the same code is the
easiest test in the world to write vacuously; and every probe-rung test carries an anti-vacuity
assertion that a probe really fired, since a refused probe satisfies "the verdict did not move" for
the wrong reason.

The pack is `knowledge/mock_domain/` **through a copy that declares the open questions**
(`tests/mock_domain_inquiries.py`), because the invariant is a property of the ENGINE and the
shipped fixture pack is `installed_packs.FIXTURE_PACK`: a declaration made there would be read by
the whole shared suite and would move output the rest of it asserts on.
"""

import ast
import copy
import json
import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.correlation import CorrelationModule
from src.inquiry import INQUIRY_STATES
from src.knowledge.pack import load_knowledge_pack
from src.stage_health import score_stage
from src.utils.paths import REPO_ROOT

from tests.mock_domain_inquiries import (
    IN_HAND_SOURCE,
    PROBE_SOURCE,
    inquiry_pack,
    open_question,
    register_row,
)

# The incident and the rows are the LINK lane's, imported rather than copied on purpose: both
# advisory lanes have to be invariant over the SAME run, and a second set of fixture-pack rows here
# could drift from that one without either file failing. `tests/__init__.py` exists, so this is the
# module pytest already collected and not a second identity of it.
from tests.test_links_never_change_the_verdict import _logs, _understanding

MOCK_DOMAIN_DIR = REPO_ROOT / "knowledge" / "mock_domain"
CORRELATION_PY = REPO_ROOT / "src" / "correlation.py"

#: The fields whose bytes the open-question pass may not touch. `inquiries` is excluded from the
#: brief comparison — it is the ONE field the pass is allowed to write, and comparing it would make
#: the assertion vacuous the moment the feature works. `links` is deliberately NOT excluded: the
#: link lane runs before this one and its output is part of what must not move.
_BRIEF_EXCLUDE = {"inquiries"}


# --- fixtures ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plain_pack():
    """The shipped fixture pack, which declares no `open_questions:` at all."""
    return load_knowledge_pack(MOCK_DOMAIN_DIR)


def _module(pack, inquiries_enabled, probe=None, budget=1):
    """A correlation module whose LLM cannot be reached, so the run is fully deterministic.

    The narration and the transform plan are the only two LLM calls in the stage and both are
    best-effort, so a raising client leaves the deterministic aggregates, the verdict and the brief
    — every field this test compares — intact and reproducible.

    `inquiries` rides in the CORRELATION config slice, beside `links`, because that slice is the
    one `main()` already hands this module.
    """
    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=RuntimeError("no llm in this test"))
    return CorrelationModule(
        {
            "inquiries": {
                "enabled": inquiries_enabled,
                "max_inquiry_probes_per_run": budget,
                "probe_timeout_seconds": 30,
            }
        },
        llm,
        knowledge_pack=pack,
        inquiry_probe=probe,
    )


def _probe(rows=None, calls=None):
    """An injected fetcher: records what it was asked, answers with `rows`.

    `None` rows is the "the source did not answer" answer and a list is the "it answered" answer,
    including the empty list — the same distinction `unanswered_out` draws one stage earlier, and
    the reason the callable returns an Optional rather than a list.
    """

    async def _run(source, analysis, **kw):
        if calls is not None:
            calls.append({"source": source, **kw})
        return None if rows is None else [dict(r) for r in rows]

    return _run


def _trigger_for(verdict):
    """A `when:` that really fires on THIS run, read off the run rather than guessed.

    A declaration whose trigger never fires produces no finding, and a file full of empty
    `inquiries` lists would satisfy every byte-equality assertion below while proving nothing. So
    the trigger is derived from the verdict the fixture actually reaches: a condition and the
    result it read, else the verdict class, which every adjudicated subject carries.
    """
    subject = verdict.subjects[0]
    for check in getattr(subject, "checks", None) or []:
        result = str(getattr(check, "result", "") or "").strip().lower()
        if result in ("unknown", "fail", "pass"):
            return {"condition": str(check.id), "result": result}
    return {"verdict_class": str(getattr(subject, "verdict_class", "") or "")}


async def _armed(plain_pack, tmp_path, modes=("free",), label=""):
    """`(declaring pack, baseline result)`: a pack copy whose questions fire on this fixture.

    The baseline run is returned as well because it is what the trigger was read off — a caller
    asserting against it is comparing like with like.
    """
    baseline = await _module(plain_pack, False).analyze(_logs(), _understanding())
    assert baseline.verdict is not None and baseline.verdict.subjects, (
        "the fixture produced no adjudicated subject, so no trigger can be derived and every "
        "assertion in this file would be vacuous"
    )
    trigger = _trigger_for(baseline.verdict)
    entries = [
        open_question(question_id=f"q_{mode}", mode=mode, **trigger) for mode in modes
    ]
    pack = inquiry_pack(tmp_path, entries, label=label or "_".join(modes))
    return pack, baseline


def _sacred(result):
    """Every channel the operator reads as THIS run's finding, as comparable bytes.

    Deliberately not `result.model_dump_json()` whole: that would compare `inquiries` too and the
    test would fail for the one reason it must not — the feature working.
    """
    brief = None
    if result.brief is not None:
        brief = result.brief.model_dump(mode="json")
        for key in _BRIEF_EXCLUDE:
            brief.pop(key, None)
    return {
        "verdict": (
            result.verdict.model_dump_json() if result.verdict is not None else None
        ),
        "brief": json.dumps(brief, sort_keys=True, default=str),
        "record_count": result.record_count,
        "summary_text": result.summary_text,
        "aggregations": json.dumps(result.aggregations, sort_keys=True, default=str),
        "findings": json.dumps(
            [f.model_dump(mode="json") for f in result.findings],
            sort_keys=True,
            default=str,
        ),
        "transforms": json.dumps(
            [t.model_dump(mode="json") for t in result.transforms],
            sort_keys=True,
            default=str,
        ),
        "evidence": (
            result.evidence.model_dump_json() if result.evidence is not None else None
        ),
    }


# --- 1. by value ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_verdict_is_byte_identical_with_and_without_the_pass(
    plain_pack, tmp_path
):
    """The one test that has to fail if an open question is ever wired into a condition."""
    pack, _ = await _armed(plain_pack, tmp_path, modes=("free", "probe"))
    off = await _module(pack, False).analyze(_logs(), _understanding())
    on = await _module(pack, True).analyze(_logs(), _understanding())

    # A vacuous fixture is the failure mode this whole suite is built to avoid: with no verdict,
    # byte-equality below is satisfied by two `None`s.
    assert (
        off.verdict is not None
    ), "the fixture produced no verdict — nothing is guarded"
    assert off.verdict.subjects, "the verdict adjudicated no subject"
    assert off.brief is not None, "the fixture produced no brief"
    assert on.inquiries, "the pass raised no question — the comparison below proves nothing"

    a, b = _sacred(off), _sacred(on)
    for field in sorted(a):
        assert a[field] == b[field], (
            f"the open-question pass changed `{field}` — an InquiryFinding has reached the "
            "sacred channel. An inquiry is advisory and addressed to a human; it may not be "
            "readable by a condition, feed a severity, or move a rollup."
        )


@pytest.mark.asyncio
async def test_the_pass_writes_only_the_inquiries_field(plain_pack, tmp_path):
    """The complement of the test above: something DID change, and it is only `inquiries`.

    Without this, disabling the feature by accident (a config key read from the wrong slice, an
    exception swallowed by the best-effort wrapper) makes the invariant test pass for the worst
    possible reason.
    """
    pack, _ = await _armed(plain_pack, tmp_path, modes=("free",))
    on = await _module(pack, True).analyze(_logs(), _understanding())
    assert hasattr(on, "inquiries"), "CorrelationResult carries no `inquiries` field"
    assert on.inquiries, "the declaring pack raised no open question"
    assert on.brief is not None and list(on.brief.inquiries) == list(on.inquiries), (
        "the brief did not receive the same findings — the case builder's readers would see a "
        "different lane from the report's"
    )


@pytest.mark.asyncio
async def test_a_pack_declaring_no_open_question_is_a_byte_identical_no_op(plain_pack):
    """Absence is a no-op — the same guarantee `follow_up_passes` gives a single-pass pack.

    Asserted over the SHIPPED fixture pack rather than by disabling the feature, because the two
    are different claims: a config switch proves the switch works, a pack that never heard of the
    key proves that an installation which never heard of this feature is unaffected by it. This is
    also the assertion that keeps the rest of the shared suite honest, since every other test that
    runs this pack asserts on output this lane must not have touched.
    """
    off = await _module(plain_pack, False).analyze(_logs(), _understanding())
    on = await _module(plain_pack, True).analyze(_logs(), _understanding())

    assert on.inquiries == [], (
        "a pack declaring no `open_questions:` produced findings — the pass invented a question "
        "the pack never declared"
    )
    assert on.verdict is not None, "the fixture stopped producing a verdict"
    assert _sacred(off) == _sacred(on)


@pytest.mark.asyncio
async def test_the_five_states_are_the_declared_ones(plain_pack, tmp_path):
    """Whatever the lane reports, it reports in the vocabulary the module declares.

    An undeclared state renders as an unlabelled row: the report keys its blocks on these five
    names, so a sixth would print a question with no heading — the silence this lane exists to
    avoid, arriving through the field that says what became of the question.
    """
    pack, _ = await _armed(
        plain_pack, tmp_path, modes=("free", "probe", "unscopable"), label="states"
    )
    on = await _module(pack, True, probe=_probe(rows=[register_row()])).analyze(
        _logs(), _understanding()
    )
    assert on.inquiries, "no question was raised"
    for finding in on.inquiries:
        assert finding.state in INQUIRY_STATES, (
            f"open question {finding.id!r} reported state {finding.state!r}, which is not one of "
            f"{list(INQUIRY_STATES)} — every reader keys its rendering on that list"
        )


# --- 2. by identity ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_pass_adds_no_source_to_the_logs_the_verdict_reads(
    plain_pack, tmp_path
):
    """`logs` is not a scratch space. Sixteen places enumerate it.

    The rejected implementation namespaces a probe's rows into `logs` under a prefixed key. It is
    quiet and it is wrong in both directions: `aggregate` counts them (so a run with no real
    records reports a healthy total and never reaches the FATAL `no_records` signal), and
    `_merge_co_identified_subjects` may license a co-identity merge from a row no condition was
    allowed to see. Probe rows are read inside `settle_with_probe` and nowhere else.

    Asserted with the probe ARMED and firing, because a lane that spends nothing cannot fail this.
    """
    pack, _ = await _armed(plain_pack, tmp_path, modes=("probe",))
    calls = []
    logs = _logs()
    before_keys = set(logs)
    before_ids = {k: id(v) for k, v in logs.items()}
    before_counts = {k: len(v) for k, v in logs.items()}
    before_copy = copy.deepcopy(logs)

    result = await _module(
        pack, True, probe=_probe(rows=[register_row(), register_row()], calls=calls)
    ).analyze(logs, _understanding())

    assert calls, "no probe was asked — this test's own premise did not hold"
    assert any(
        f.probe_spent for f in result.inquiries
    ), "no probe was spent, so nothing could have been merged into logs"

    assert (
        set(logs) == before_keys
    ), f"the open-question pass added {set(logs) - before_keys} to logs"
    assert {k: id(v) for k, v in logs.items()} == before_ids, "a log list was replaced"
    assert {k: len(v) for k, v in logs.items()} == before_counts, "a row was appended"
    assert logs == before_copy, "a retrieved row was mutated in place"


@pytest.mark.asyncio
async def test_stage_health_scores_the_stage_identically(plain_pack, tmp_path):
    """An open question may not move the number that decides whether a gate opens.

    `semi_auto` proceeds on `health.scored` and the score; an advisory finding that nudged it
    would turn "a human reviews this" into "the machine continued" — a HITL decision made by a
    question nobody has read yet. Asserted with a probe in the loop, since an ANSWERED question is
    the one a scorer would most plausibly be taught to read.
    """
    pack, _ = await _armed(plain_pack, tmp_path, modes=("free", "probe"))
    off = await _module(pack, False).analyze(_logs(), _understanding())
    on = await _module(pack, True, probe=_probe(rows=[register_row()])).analyze(
        _logs(), _understanding()
    )
    assert on.inquiries, "no question was raised — the comparison is vacuous"

    a = score_stage("correlation", off)
    b = score_stage("correlation", on)
    assert a.scored, "the correlation stage scored nothing — the comparison is vacuous"
    assert (a.score, a.gate_recommended, list(a.reasons)) == (
        b.score,
        b.gate_recommended,
        list(b.reasons),
    ), "the open-question pass moved the stage health score"


@pytest.mark.asyncio
async def test_a_second_run_over_the_same_logs_reproduces_the_verdict(
    plain_pack, tmp_path
):
    """The gate-rejection path: a rejected stage RE-RUNS correlation on the same inputs.

    `src/human_guidance.py` re-runs the stage with the operator's guidance injected and re-gates
    on the result. So the pass has to be idempotent with respect to everything it read — and this
    is precisely the assertion a namespaced-logs implementation fails, because pass two sees pass
    one's leftovers.
    """
    pack, _ = await _armed(plain_pack, tmp_path, modes=("free", "probe"))
    module = _module(pack, True, probe=_probe(rows=[register_row()]))
    logs = _logs()
    first = await module.analyze(logs, _understanding())
    second = await module.analyze(logs, _understanding())
    assert _sacred(first) == _sacred(second), (
        "correlation is not reproducible across two runs over one `logs` — the open-question "
        "pass left state behind"
    )


@pytest.mark.asyncio
async def test_the_pass_is_deterministic(plain_pack, tmp_path):
    """Same inputs, same findings — no ordering by set iteration, no clock, no randomness."""
    pack, _ = await _armed(plain_pack, tmp_path, modes=("free", "probe", "unscopable"))
    module = _module(pack, True, probe=_probe(rows=[register_row()]))
    logs = _logs()
    first = await module.analyze(logs, _understanding())
    second = await module.analyze(logs, _understanding())
    assert len(first.inquiries) > 1, "one finding cannot demonstrate an ordering"
    assert [f.model_dump(mode="json") for f in first.inquiries] == [
        f.model_dump(mode="json") for f in second.inquiries
    ], "the findings are not deterministic — a set or a clock reached the ordering"


# --- the rung that SPENDS something -----------------------------------------------------
# Everything above is about a pass that mostly READS, so the three assertions are restated here
# with a probe in the loop, each paired with an anti-vacuity assertion that the probe really ran.


@pytest.mark.asyncio
async def test_a_probe_really_fires_and_asks_about_the_declared_scope(
    plain_pack, tmp_path
):
    """The anti-vacuity assertion the tests above rest on: the probe rung is reachable at all.

    Shipped narrow but ARMED (one probe), because a bound shipped at 0 makes every refusal code
    unreachable outside a test. And what it asks is asserted, not just that it asked: a probe that
    reached the source without the scope values would be a window-wide scan wearing this lane's
    name.
    """
    pack, _ = await _armed(plain_pack, tmp_path, modes=("probe",))
    calls = []
    result = await _module(
        pack, True, probe=_probe(rows=[register_row()], calls=calls)
    ).analyze(_logs(), _understanding())

    assert len(calls) == 1, f"expected exactly one probe, got {len(calls)}"
    assert calls[0]["source"] == PROBE_SOURCE
    assert calls[0]["scope_values"], "the probe was asked with no scope value at all"
    assert calls[0]["row_cap"] > 0 and calls[0]["timeout"] > 0, (
        "the probe was asked with no row cap or no timeout — the two bounds that make an "
        "advisory query safe"
    )
    answered = [f for f in result.inquiries if f.state == "answered"]
    assert answered, "the probe's rows settled no question"
    assert answered[0].probe_spent is True
    assert answered[0].meaning, (
        "an answered question carries no meaning — a count with no declared meaning is the "
        "failure this lane exists to avoid"
    )


@pytest.mark.asyncio
async def test_the_free_rung_settles_from_rows_in_hand_and_spends_nothing(
    plain_pack, tmp_path
):
    """A question whose source this run already retrieved costs no query at all.

    The cheap half of the whole lane, and the one a reader must be able to tell apart from a
    spend: `probe_spent` stays False and the note says the rows were already in hand.
    """
    pack, _ = await _armed(plain_pack, tmp_path, modes=("free",))
    calls = []
    result = await _module(pack, True, probe=_probe(rows=[], calls=calls)).analyze(
        _logs(), _understanding()
    )
    assert result.inquiries, "no question was raised"
    finding = result.inquiries[0]
    assert calls == [], (
        "a probe was spent on a question whose source this run had already retrieved — the free "
        "rung is what keeps this lane affordable"
    )
    assert finding.probe_spent is False
    assert finding.state in ("answered", "empty"), (
        f"the free rung left the question at {finding.state!r}; rows in hand settle it one way "
        "or the other"
    )
    # Resolved through the ruleset's own `sources:` map, so the finding names the PHYSICAL source.
    assert finding.source in _logs(), (
        f"the free rung claims to have read {finding.source!r}, which this run did not retrieve"
    )
    assert IN_HAND_SOURCE not in _logs(), (
        "the fixture's logical name is also a physical log key, so this test cannot tell a "
        "resolved source from an unresolved one"
    )


@pytest.mark.asyncio
async def test_a_source_that_did_not_answer_is_not_a_source_that_answered_with_nothing(
    plain_pack, tmp_path
):
    """The distinction the whole repo is organised around, at this lane's own seam.

    `None` from the probe is a source that never answered; `[]` is an answer. They license
    opposite next steps — a credential or a catalog entry against a reading of the evidence — so
    they are two states with two declared meanings and never one zero.
    """
    pack, _ = await _armed(plain_pack, tmp_path, modes=("probe",))
    silent = await _module(pack, True, probe=_probe(rows=None)).analyze(
        _logs(), _understanding()
    )
    empty = await _module(pack, True, probe=_probe(rows=[])).analyze(
        _logs(), _understanding()
    )
    assert silent.inquiries and empty.inquiries
    a, b = silent.inquiries[0], empty.inquiries[0]
    assert (a.state, b.state) == ("unanswered", "empty"), (
        f"a non-answer and an empty answer collapsed into {a.state!r} and {b.state!r} — the "
        "failure class this lane exists to report"
    )
    assert a.meaning and b.meaning and a.meaning != b.meaning, (
        "the two outcomes carry the same declared meaning, so a reader cannot act on the "
        "difference"
    )


@pytest.mark.asyncio
async def test_a_closed_budget_refuses_by_a_coded_note_and_never_by_silence(
    plain_pack, tmp_path
):
    """`max_inquiry_probes_per_run: 0` closes the lane — and says so on every question.

    A refused question that renders as a blank line reads exactly like a question nobody raised,
    which is the one reading this lane may not produce. Also the invariance assertion in the
    direction that spends nothing: a closed lane must not move the verdict either.
    """
    pack, _ = await _armed(plain_pack, tmp_path, modes=("probe",))
    calls = []
    closed = await _module(
        pack, True, probe=_probe(rows=[register_row()], calls=calls), budget=0
    ).analyze(_logs(), _understanding())
    off = await _module(pack, False).analyze(_logs(), _understanding())

    assert calls == [], "the lane spent a probe with a budget of 0"
    assert closed.inquiries, "a closed budget dropped the question instead of reporting it"
    finding = closed.inquiries[0]
    assert finding.state == "not_asked"
    assert finding.probe_note, (
        "a question refused for want of budget carries no note — a refusal that renders as "
        "silence is indistinguishable from a question nobody raised"
    )
    assert _sacred(off) == _sacred(closed)


# --- 3. by structure --------------------------------------------------------------------


def _correlation_tree():
    return ast.parse(CORRELATION_PY.read_text(encoding="utf-8"))


def _call_lines(tree, func_name):
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == func_name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == func_name)
        )
    ]


def test_the_pass_runs_after_the_verdict_and_the_brief():
    """Read the call site as an AST, not as a comment.

    A question computed BEFORE the verdict is one refactor away from being available to it, and
    the value comparison above cannot see the difference on a fixture where the question happens
    to change nothing. Position is the structural half of the invariant — and it is load-bearing
    twice over here, since `assess_inquiries` takes the verdict and the brief as ARGUMENTS: moving
    it earlier would not merely risk a leak, it would hand the pass two `None`s and report a lane
    that raised nothing.
    """
    tree = _correlation_tree()
    inquiries = _call_lines(tree, "assess_inquiries")
    assert inquiries, "`assess_inquiries` is not called from src/correlation.py"
    assert (
        len(inquiries) == 1
    ), f"the open-question pass is called {len(inquiries)} times; it must be once"

    verdicts = _call_lines(tree, "evaluate_verdict")
    briefs = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "analyze"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "analyzer"
    ]
    assert (
        verdicts and briefs
    ), "the verdict/brief call sites moved — re-anchor this test"
    assert inquiries[0] > max(verdicts), "the open-question pass runs BEFORE the verdict"
    assert inquiries[0] > max(briefs), "the open-question pass runs BEFORE the case builder"


def test_the_findings_land_on_nothing_but_an_inquiries_attribute():
    """Whatever `assess_inquiries` returns may be assigned to `*.inquiries` and to nothing else.

    This is the assertion that survives a refactor: it does not care what the pass computes, only
    that its output has exactly one destination. A finding reaching any other attribute of
    `result` is a finding reaching a channel some reader treats as this run's own.
    """
    tree = _correlation_tree()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Call)
            and (
                (
                    isinstance(value.func, ast.Name)
                    and value.func.id == "assess_inquiries"
                )
                or (
                    isinstance(value.func, ast.Attribute)
                    and value.func.attr == "assess_inquiries"
                )
            )
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            name = (
                target.attr
                if isinstance(target, ast.Attribute)
                else getattr(target, "id", "?")
            )
            assert name in {"inquiries", "inquiry_findings"}, (
                f"`assess_inquiries(...)` is assigned to `{name}` at line {node.lineno} — an "
                "open question may only reach the advisory lane"
            )


def test_the_verdict_engine_takes_no_inquiry_argument():
    """`evaluate_verdict` may not learn about open questions, by parameter or by keyword.

    A parameter is how "advisory" becomes "an input the rollup happens to read". The engine's own
    signature is the cheapest place to make that impossible, and the case builder's is checked
    beside it because the brief is the other thing a downstream reader treats as this run's.
    """
    import inspect

    from src.correlation import evaluate_verdict
    from src.usecases.base import UseCaseAnalyzer

    for func in (evaluate_verdict, UseCaseAnalyzer.analyze):
        params = set(inspect.signature(func).parameters)
        assert not {
            p for p in params if "inquir" in p.lower() or "question" in p.lower()
        }, f"{func.__qualname__} accepts an inquiry-shaped parameter: {sorted(params)}"


# --- the guard on the guards -------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_comparison_can_actually_fail(plain_pack):
    """A mutation test on `_sacred`: prove the comparison objects to a real difference.

    Two runs of the same code are trivially equal, so every assertion above is one silent bug in
    this helper away from being vacuous — the exact shape of "a check that cannot fire is
    indistinguishable from one that found nothing". Each sacred field is perturbed independently,
    because a helper that only reads `verdict` would still pass a single-field mutation test.
    """
    result = await _module(plain_pack, False).analyze(_logs(), _understanding())
    baseline = _sacred(result)

    result.record_count = (result.record_count or 0) + 1
    assert _sacred(result)["record_count"] != baseline["record_count"]

    result.summary_text = (result.summary_text or "") + " perturbed"
    assert _sacred(result)["summary_text"] != baseline["summary_text"]

    result.aggregations = dict(result.aggregations or {}, _perturbed=1)
    assert _sacred(result)["aggregations"] != baseline["aggregations"]

    result.transforms = []
    assert _sacred(result)["transforms"] != baseline["transforms"]

    result.verdict.summary = (result.verdict.summary or "") + " perturbed"
    assert _sacred(result)["verdict"] != baseline["verdict"]

    result.brief.use_case = (result.brief.use_case or "") + "_perturbed"
    assert _sacred(result)["brief"] != baseline["brief"]

    result.evidence = None
    assert _sacred(result)["evidence"] != baseline["evidence"]


def test_the_advisory_lane_carries_no_verdict_vocabulary():
    """An `InquiryFinding` may not wear the verdict's own field names.

    The operator has to be able to read "verdict INSUFFICIENT DATA (this procedure, N conditions)"
    and "open question, unanswered (advisory, nobody adjudicated it)" as two different sentences.
    A finding carrying `verdict_class` or a bare `severity` is how the second gets quoted as the
    first in a handover — the same reason `LinkFinding.advisory_severity` is not called `severity`.
    """
    from src.models.pydantic_models import InquiryFinding

    fields = set(InquiryFinding.model_fields)
    assert not fields & {"severity", "verdict_class", "result", "checks"}, (
        f"InquiryFinding carries verdict vocabulary: {sorted(fields & {'severity', 'verdict_class', 'result', 'checks'})}"
    )
    assert "advisory_note" in fields, (
        "the lane states its own provenance nowhere — a finding with no `advisory_note` is one a "
        "reader may take for an adjudication"
    )
    assert {"state", "meaning"} <= fields, (
        "a state with no meaning is the unlabelled count this lane exists to prevent"
    )


def test_every_inquiry_state_is_declared_and_distinct():
    """The five states may never collapse: "did not ask" is not "asked and got nothing".

    The second is a FINDING — the same distinction `unanswered_out` draws against an empty result,
    and the one a reader loses first. And `unanswered` is a third thing again: the source was asked
    and said nothing at all, which needs a credential rather than a reading.
    """
    from src.inquiry import INQUIRY_STATES as states

    assert set(states) == {
        "answered",
        "not_asked",
        "empty",
        "unanswered",
        "unreachable",
    }, f"the inquiry states drifted: {sorted(states)}"
    assert len(states) == len(set(states)), "a state is declared twice"
    # Ordered actionable-first: a reader who stops halfway has seen every question with something
    # to say. Asserted because the sort key in `assess_inquiries` is this tuple's ORDER.
    assert states[0] == "answered" and states[1] == "not_asked"


def test_the_inquiry_module_is_pure():
    """No LLM, no IO, no clock, no randomness — `evaluate_verdict`'s posture, restated.

    Read as an AST rather than trusted from a docstring. A pass that could fetch would be a
    retrieval stage hiding inside correlation, with none of the budgets, none of the caps and none
    of the per-source timeouts that make retrieval safe. A clock would make the pass
    non-reproducible, which is what breaks the re-run assertion above.

    One module and not two, unlike the link lane's pair: everything that spends lives in
    `src/inquiry_probe.py`, which is asserted separately below.
    """
    name = "src/inquiry.py"
    tree = ast.parse((REPO_ROOT / "src" / "inquiry.py").read_text(encoding="utf-8"))
    banned_modules = {
        "requests",
        "httpx",
        "aiohttp",
        "urllib",
        "socket",
        "random",
        "subprocess",
        "asyncio",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in banned_modules, f"{name} imports {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in banned_modules, f"{name} imports from {node.module}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert not (
                node.func.attr in {"now", "utcnow", "today", "time", "monotonic"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"datetime", "date", "time"}
            ), f"{name} reads the clock at line {node.lineno}"
        # No `async def`: an await is where IO gets added later without anyone noticing.
        elif isinstance(node, ast.AsyncFunctionDef):
            raise AssertionError(
                f"{name} declares `async def {node.name}` — the pass must be pure"
            )


_INQUIRY_SEAMS = {"assess_inquiries", "settle_with_probe", "settle_from_rows_in_hand"}


def test_the_module_exposes_ONE_seam_PER_THING_IT_DOES():
    """Three seams, so there are three places to audit — and no fourth.

    Each is a different verb over the same declarations: `assess_inquiries` reads the run and
    raises the questions, `settle_with_probe` reads ONE probe's answer, `settle_from_rows_in_hand`
    reads rows this run already had. The last two are separate entries rather than one function
    with a `spent` flag on purpose: `probe_spent` is a claim about COST, so the call site has to
    say which of the two happened, and a default argument is how the free rung would come to claim
    a spend nobody made.

    The set is asserted exactly: a fourth public name is how this lane would grow a path that
    reaches a backend without the invariant tests above ever seeing it.
    """
    import src.inquiry as inquiry_mod

    public = {
        n
        for n in dir(inquiry_mod)
        if not n.startswith("_") and callable(getattr(inquiry_mod, n))
    }
    # Imported helpers are not this module's API; only names DEFINED here count.
    tree = ast.parse(pathlib.Path(inquiry_mod.__file__).read_text(encoding="utf-8"))
    defined = {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not n.name.startswith("_")
    }
    assert defined == _INQUIRY_SEAMS, (
        f"src/inquiry.py exposes {sorted(defined)} — this lane has exactly three seams: one that "
        "raises the questions, one that reads a probe's answer, one that reads rows in hand"
    )
    assert _INQUIRY_SEAMS <= public


def test_only_the_FETCHING_module_may_reach_a_backend():
    """The split that makes the probe rung safe, asserted as a fact about two files.

    `src/inquiry.py` is scanned for purity above, so the settlement cannot fetch. The complement
    is that `src/inquiry_probe.py` must not re-implement the settlement: it may call the seam, and
    it may not read this module's privates. An underscore name crossing that line is how the
    five-state reading would quietly acquire a second implementation on the side that spends.

    `_scoped_analysis` is imported from `src/link_probe.py` and is deliberately not in scope here:
    it is ONE copy of how a probe inherits the incident's window, shared with the lane this one
    mirrors, and a second copy would be a second answer to it.
    """
    tree = ast.parse(
        (REPO_ROOT / "src" / "inquiry_probe.py").read_text(encoding="utf-8")
    )
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("inquiry"):
            imported |= {a.name for a in node.names}
    assert imported, "src/inquiry_probe.py reaches no seam in src/inquiry.py at all"
    assert imported <= _INQUIRY_SEAMS, (
        f"src/inquiry_probe.py imports {sorted(imported - _INQUIRY_SEAMS)} from src/inquiry.py — "
        "a private reached across modules is a private that has stopped being one, and the "
        "five-state reading may have exactly one implementation"
    )
