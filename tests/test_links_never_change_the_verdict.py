"""The sacred channel: a cross-procedure LINK may never change this run's verdict.

Cross-use-case linking adds, at the end of correlation, a deterministic pass asking of every
SIBLING procedure "could this evidence also be yours?" and emitting a `LinkFinding` per candidate.
The design rests on one refusal, recorded as R3 in
`knowledge/<domain>/use_cases/cross_silo/ROUTER.md`: **a router adds no evidence, so it must not add
confidence either.** A link is addressed to a human — not evidence, not a condition input, and it
may not move a label, a severity or a health score by a byte.

Nothing about wiring a link into a condition FAILS; it produces a verdict that is more confident
and differently wrong, with every stage green. So the invariant is asserted three ways, each
catching a different way of breaking it:

  1. **By value** — the sacred fields serialise identically with the link pass off and on. Catches
     a link that feeds a condition, a rollup or a severity.
  2. **By identity** — `logs` gains no key and no list is replaced. Catches the tempting
     implementation where a probe's rows are namespaced INTO `logs`: sixteen places enumerate that
     dict, and `aggregate` counting a probe's rows turns a real `no_records` into a
     healthy-looking total while `_merge_co_identified_subjects` licenses a merge from a row the
     verdict never saw.
  3. **By structure** — the call site read as an AST: the link pass runs AFTER the verdict and the
     brief, and its return value lands on nothing but a `links` attribute. Catches a future edit
     that threads a link EARLIER, whose effect on this fixture happens to be nil.

And one guard on the guards: `test_the_comparison_can_actually_fail` mutates a sacred field and
demands the comparison object to it — a byte-equality assertion between two runs of the same code
is the easiest test in the world to write vacuously.

The pack is `knowledge/mock_domain/` because the invariant is a property of the ENGINE: two
rulesets over one subject entity sharing two of their three sources, one declaring a `scope_gate`
and one not. That is the shape the link ladder reads, and it is available on every branch.
"""

import ast
import copy
import json
import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.correlation import CorrelationModule
from src.knowledge.pack import load_knowledge_pack
from src.link_children import plan_child_spawns
from src.link_escalation import ESCALATING_MODES
from src.models.pydantic_models import (
    ExtractedEntity,
    IncidentAnalysis,
    UnderstandingResult,
)
from src.stage_health import score_stage
from src.utils.paths import REPO_ROOT

# Shared with `test_link_probe.py`, which asserts the other half of rung 3 — the fetch. One
# builder, because a second copy of the probeable pack would drift from this one silently.
from tests.mock_domain_links import GATE_FLIPS_ON_PROBE
from tests.mock_domain_links import probe_pack as _probe_pack
from tests.mock_domain_links import roster_row as _roster_row

MOCK_DOMAIN_DIR = REPO_ROOT / "knowledge" / "mock_domain"
CORRELATION_PY = REPO_ROOT / "src" / "correlation.py"

#: The fields whose bytes the link pass may not touch. `links` itself is excluded from the
#: brief comparison — it is the ONE field the pass is allowed to write, and comparing it would
#: make the assertion vacuous the moment the feature works.
_BRIEF_EXCLUDE = {"links"}


# --- fixtures ---------------------------------------------------------------------------
# Row shapes are `knowledge/mock_domain/schemas/shipment_ledger.yaml`'s, kept deliberately
# close to `tests/test_mock_domain_pack.py`'s builders so a schema drift shows up in both.


@pytest.fixture(scope="module")
def pack():
    return load_knowledge_pack(MOCK_DOMAIN_DIR)


def _ledger_row(**over):
    """One shipment: bare, unreviewed, refunded, handler and POD signer DIVERGE.

    Chosen so both procedures have something to say about it — `refund_fraud` reads its
    servicing counters and its review flag, `courier_collusion` reads the handler/POD
    divergence that is its own decisive indicator. A fixture only one procedure can read
    would make the link ladder's rung 1 untestable in the direction that matters.
    """
    row = {
        "tracking_code": "RT48192043",
        "depot_code": "LDS04",
        "handler.badge": "0192C",
        "handler.device_login": "jdunne",
        "created_at": "2026-07-20T08:00:00Z",
        "element_counters.INS": 0,
        "element_counters.SIG": 0,
        "scans": [],
        "pod.signature_captured": True,
        # DIFFERENT from handler.badge — the collusion shape.
        "pod.captured_by": "0288D",
        "refund.amount": 41.5,
        "refund.reason_code": "LOST",
        "refund.manually_reviewed": False,
        "refund.approved_by": None,
        "refund.claimed_at": "2026-07-20T15:00:00Z",
        "route.origin": "LDS04",
        "route.destination": "BHM11",
    }
    row.update(over)
    return row


def _session_row(**over):
    row = {
        "depot_code": "LDS04",
        "login": "jdunne",
        "badge": "0192C",
        "device_id": "TERM-1",
        "session.automated": False,
        "session.started_at": "2026-07-20T07:55:00Z",
    }
    row.update(over)
    return row


def _alert_row(**over):
    row = {
        "alert.incident_id": "IR-REF-0001",
        "alert.depot_code": "LDS04",
        "alert.courier_badge": "0192C",
        "alert.device_login": "jdunne",
        "alert.claim_id": "CLM-9001",
        "alert.claimed_shipments": ["RT48192043"],
        "alert": {
            "claim_lines": [{"shipment_code": "RT48192043", "handler_badge": "0192C"}]
        },
        "alert.raised_at": "2026-07-20T16:00:00Z",
    }
    row.update(over)
    return row


def _logs():
    return {
        "shipment_ledger": [_ledger_row()],
        "device_sessions": [_session_row()],
        "refund_alerts": [_alert_row()],
    }


def _understanding():
    return UnderstandingResult(
        incident_id="INC-LINK-1",
        analysis=IncidentAnalysis(
            incident_summary="Burst of refund claims in one depot",
            severity="5",
            severity_reasoning="r",
            impact_assessment="i",
            key_investigation_areas=[],
            log_sources_to_review=[],
            initial_hypotheses=[],
            recommended_actions=[],
            stakeholder_notification=[],
            extracted_entities=[
                ExtractedEntity(type="shipment", value="RT48192043"),
                ExtractedEntity(type="depot", value="LDS04"),
                ExtractedEntity(type="courier", value="0192C"),
            ],
            correlation_keys=[],
        ),
    )


def _probe(rows=None, calls=None):
    """An injected rung-3 fetcher: records what it was asked, answers with `rows`.

    `None` rows is the "I could not ask" answer and a list is the "I asked" answer, including
    the empty list — the same distinction `unanswered_out` draws one stage earlier, and the
    reason the callable returns an Optional rather than a list.
    """

    async def _run(source, analysis, **kw):
        if calls is not None:
            calls.append({"source": source, **kw})
        return None if rows is None else [dict(r) for r in rows]

    return _run


def _module(pack, links_enabled):
    """A correlation module whose LLM cannot be reached, so the run is fully deterministic.

    The narration and the transform plan are the only two LLM calls in the stage and both are
    best-effort, so a raising client leaves the deterministic aggregates, the verdict and the
    brief — every field this test compares — intact and reproducible. A narrated
    `summary_text` would make byte-equality a statement about the mock, not about the engine.

    `links` rides in the CORRELATION config slice (`main_config.yaml`'s `correlation:` block),
    because that slice is the one `main()` already hands this module — a top-level `links:`
    section would need a new constructor argument, and the build order forbids changing an
    existing signature for this feature.
    """
    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=RuntimeError("no llm in this test"))
    return CorrelationModule(
        {"links": {"enabled": links_enabled}}, llm, knowledge_pack=pack
    )


def _probe_module(pack, probe=None, budget=2, mode="auto"):
    """The same module with rung 3 ARMED — a probe callable and a non-zero budget.

    Both halves are needed and neither implies the other: `max_probes_per_run` is what
    `escalation_budgeted` reads to decide an escalating mode has anything to spend, and the
    callable is the only thing in this module that can reach a backend. A module with the
    budget and no callable is the shape every deployment that never wires one keeps.
    """
    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=RuntimeError("no llm in this test"))
    return CorrelationModule(
        {
            "links": {
                "enabled": True,
                "escalation_mode": mode,
                "max_probes_per_run": budget,
                "probe_timeout_seconds": 30,
            }
        },
        llm,
        knowledge_pack=pack,
        link_probe=probe,
    )


def _sacred(result):
    """Every channel the operator reads as THIS run's finding, as comparable bytes.

    Deliberately not `result.model_dump_json()` whole: that would compare `links` too and the
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
async def test_the_verdict_is_byte_identical_with_and_without_the_link_pass(pack):
    """R3, as bytes. The one test that has to fail if a link is ever wired into a condition."""
    off = await _module(pack, False).analyze(_logs(), _understanding())
    on = await _module(pack, True).analyze(_logs(), _understanding())

    # A vacuous fixture is the failure mode this whole suite is built to avoid: if the pack
    # produced no verdict, byte-equality below is satisfied by two `None`s.
    assert (
        off.verdict is not None
    ), "the fixture produced no verdict — nothing is guarded"
    assert off.verdict.subjects, "the verdict adjudicated no subject"
    assert off.brief is not None, "the fixture produced no brief"

    a, b = _sacred(off), _sacred(on)
    for field in sorted(a):
        assert a[field] == b[field], (
            f"the link pass changed `{field}` — a LinkFinding has reached the sacred "
            "channel. Links are advisory and addressed to a human (ROUTER.md R3); they may "
            "not be readable by a condition, feed a severity, or move a rollup."
        )


@pytest.mark.asyncio
async def test_the_link_pass_writes_only_the_links_field(pack):
    """The complement of the test above: something DID change, and it is only `links`.

    Without this, disabling the feature by accident (a config key read from the wrong slice,
    an exception swallowed by the best-effort wrapper) makes the invariant test pass for the
    worst possible reason.
    """
    on = await _module(pack, True).analyze(_logs(), _understanding())
    assert hasattr(on, "links"), "CorrelationResult carries no `links` field"
    # A pack that declares no `entry_signals` still reports the states it can compute for
    # free, so mock_domain — whose two playbooks name each other in `related_playbooks:` —
    # must produce at least one candidate. `[]` here means the pass did not run.
    assert (
        on.links
    ), "the link pass produced no candidate on a pack whose playbooks relate"


@pytest.mark.asyncio
async def test_a_pack_declaring_no_link_surface_is_a_byte_identical_no_op(
    pack, tmp_path
):
    """Absence is a no-op — the same guarantee `follow_up_passes` gives a single-pass pack.

    Asserted over a STRIPPED COPY rather than by disabling the feature, because the two are
    different claims: a config switch proves the switch works, a stripped pack proves that an
    installation which never heard of this feature is unaffected by it.
    """
    import shutil

    import yaml

    stripped = tmp_path / "stripped"
    shutil.copytree(MOCK_DOMAIN_DIR, stripped)
    # Remove every inbound declaration and every `related_playbooks:` edge.
    for rules in stripped.rglob("rules.yaml"):
        text = rules.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        for spec in (data.get("verdicts") or {}).values():
            if isinstance(spec, dict):
                spec.pop("entry_signals", None)
        rules.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    for md in stripped.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        if "related_playbooks:" in text:
            md.write_text(
                "\n".join(
                    line
                    for line in text.splitlines()
                    if not line.startswith("related_playbooks:")
                )
                + "\n",
                encoding="utf-8",
            )

    bare_pack = load_knowledge_pack(stripped)
    result = await _module(bare_pack, True).analyze(_logs(), _understanding())
    assert result.links == [], (
        "a pack declaring no link surface produced link findings — the pass invented an "
        "edge the pack never declared"
    )
    assert result.verdict is not None, "the stripped copy stopped producing a verdict"


# --- 2. by identity ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_link_pass_adds_no_source_to_the_logs_the_verdict_reads(pack):
    """`logs` is not a scratch space. Sixteen places enumerate it.

    The rejected implementation namespaces a probe's rows into `logs` under a prefixed key.
    It is quiet and it is wrong in both directions: `aggregate` counts them (so a run with no
    real records reports a healthy total and never reaches the FATAL `no_records` signal),
    and `_merge_co_identified_subjects` may license a co-identity merge from a row no
    condition was allowed to see. Probe rows belong in their own container.
    """
    logs = _logs()
    before_keys = set(logs)
    before_ids = {k: id(v) for k, v in logs.items()}
    before_counts = {k: len(v) for k, v in logs.items()}
    before_copy = copy.deepcopy(logs)

    await _module(pack, True).analyze(logs, _understanding())

    assert (
        set(logs) == before_keys
    ), f"the link pass added {set(logs) - before_keys} to logs"
    assert {k: id(v) for k, v in logs.items()} == before_ids, "a log list was replaced"
    assert {k: len(v) for k, v in logs.items()} == before_counts, "a row was appended"
    assert logs == before_copy, "a retrieved row was mutated in place"


@pytest.mark.asyncio
async def test_stage_health_scores_the_stage_identically(pack):
    """A link may not move the number that decides whether a gate opens.

    `semi_auto` proceeds on `health.scored` and the score; an advisory finding that nudged it
    would turn "a human reviews this" into "the machine continued" — a HITL decision made by a
    link nobody has read yet.
    """
    off = await _module(pack, False).analyze(_logs(), _understanding())
    on = await _module(pack, True).analyze(_logs(), _understanding())

    a = score_stage("correlation", off)
    b = score_stage("correlation", on)
    assert a.scored, "the correlation stage scored nothing — the comparison is vacuous"
    assert (a.score, a.gate_recommended, list(a.reasons)) == (
        b.score,
        b.gate_recommended,
        list(b.reasons),
    ), "the link pass moved the stage health score"


@pytest.mark.asyncio
async def test_a_second_run_over_the_same_logs_reproduces_the_verdict(pack):
    """The gate-rejection path: a rejected stage RE-RUNS correlation on the same inputs.

    `src/human_guidance.py` re-runs the stage with the operator's guidance injected and
    re-gates on the result. So the link pass has to be idempotent with respect to everything
    it read — and this is precisely the assertion a namespaced-logs implementation fails,
    because pass two sees pass one's leftovers. Verdict-affecting drift between two runs of an
    unchanged incident is the defect that once answered `PASS` then `UNKNOWN` with nothing
    changed but a predicate's wording.
    """
    module = _module(pack, True)
    logs = _logs()
    first = await module.analyze(logs, _understanding())
    second = await module.analyze(logs, _understanding())
    assert _sacred(first) == _sacred(second), (
        "correlation is not reproducible across two runs over one `logs` — the link pass "
        "left state behind"
    )


@pytest.mark.asyncio
async def test_the_link_pass_is_deterministic(pack):
    """Same inputs, same findings — no ordering by set iteration, no clock, no randomness."""
    module = _module(pack, True)
    logs = _logs()
    first = await module.analyze(logs, _understanding())
    second = await module.analyze(logs, _understanding())
    assert [f.model_dump(mode="json") for f in first.links] == [
        f.model_dump(mode="json") for f in second.links
    ], "the link findings are not deterministic — a set or a clock reached the ordering"


# --- rung 3: the one rung that SPENDS something -----------------------------------------
# Everything above is about a pass that only READS, so each of the three assertions is restated
# here with a probe in the loop, paired with an anti-vacuity assertion that the probe really ran:
# a refused probe satisfies "the verdict did not move" for the wrong reason.


@pytest.mark.asyncio
async def test_a_probe_really_fires_on_an_armed_licensed_pair(pack, tmp_path):
    """The anti-vacuity assertion the three below rest on: rung 3 is reachable at all.

    Four things must line up before a query is spent — an escalating mode, a licensed pair, a
    signal declaring `auto_probe`, a budget — and any one silently missing makes every isolation
    assertion in this section pass while proving nothing. So the spend, its target and the
    RESETTLEMENT are all asserted.
    """
    calls = []
    probed = _probe_pack(tmp_path)
    module = _probe_module(probed, probe=_probe([_roster_row()], calls))
    result = await module.analyze(_logs(), _understanding())

    spent = [f for f in result.links if f.probe_spent]
    assert spent, (
        "no probe was spent on a pack with an armed, licensed, auto_probe pair and a "
        f"budget — links: {[(f.target_use_case, f.state, f.mode, f.probe_note) for f in result.links]}"
    )
    assert [c["source"] for c in calls] == ["depot_roster"], (
        "the probe asked the wrong source: rung 3 asks a source the TARGET procedure owns "
        f"and this run did not retrieve, got {calls}"
    )
    finding = spent[0]
    assert finding.probe_source == "depot_roster"
    assert finding.rung == 3, f"a spent probe left the rung at {finding.rung}"
    assert finding.state == "probed_positive", (
        "the probed rows fire the sibling's own declared signal, so the candidate must "
        f"resettle — state is {finding.state!r}, note {finding.evidence_note!r}"
    )
    assert (
        finding.probe_note
    ), "a probe was spent and said nothing about what it settled"


@pytest.mark.asyncio
async def test_a_probe_cannot_move_the_parent_verdict(pack, tmp_path):
    """R3 with a query behind it. The single most important assertion in this file.

    A probe's rows are real rows from a real source, and the whole failure mode is that they
    look exactly like the ones the verdict read. If they ever reach `logs`, this comparison is
    the only thing standing between an advisory scan and a parent verdict adjudicated on
    evidence no condition was allowed to see.
    """
    probed = _probe_pack(tmp_path)
    off = await _probe_module(probed, probe=None, budget=0).analyze(
        _logs(), _understanding()
    )
    on = await _probe_module(probed, probe=_probe([_roster_row()])).analyze(
        _logs(), _understanding()
    )

    assert (
        off.verdict is not None
    ), "the fixture produced no verdict — nothing is guarded"
    assert any(
        f.probe_spent for f in on.links
    ), "no probe ran, so this comparison proves nothing"
    assert not any(
        f.probe_spent for f in off.links
    ), "a probe ran with no budget and no callable"

    a, b = _sacred(off), _sacred(on)
    for field in sorted(a):
        assert a[field] == b[field], (
            f"a rung-3 probe changed `{field}` — probed rows have reached the sacred "
            "channel. They are fetched on the advisory lane's own initiative, outside every "
            "condition, every cap and every dependency the verdict is allowed to rest on."
        )


@pytest.mark.asyncio
async def test_probe_rows_never_enter_the_logs_the_verdict_reads(pack, tmp_path):
    """The rejected implementation, asserted against directly: namespaced keys inside `logs`.

    Sixteen places enumerate that dict. `aggregate` counting probe rows turns a real
    `no_records` into a healthy-looking total; `_merge_co_identified_subjects` may license a
    co-identity merge off a probed row; `src/stage_health.py` scores the stage from it. A
    probe's rows are transient by construction — they exist inside one call and are stored on
    no result field, because a new field on `CorrelationResult` would flow through
    `model_dump` into the report prompt, the anomaly prompt and the job document.
    """
    probed = _probe_pack(tmp_path)
    logs = _logs()
    before_keys = set(logs)
    before_ids = {k: id(v) for k, v in logs.items()}
    before_counts = {k: len(v) for k, v in logs.items()}
    before_copy = copy.deepcopy(logs)

    result = await _probe_module(probed, probe=_probe([_roster_row()])).analyze(
        logs, _understanding()
    )
    assert any(f.probe_spent for f in result.links), "no probe ran"

    assert (
        set(logs) == before_keys
    ), f"a probe added {set(logs) - before_keys} to the logs the verdict reads"
    assert {k: id(v) for k, v in logs.items()} == before_ids, "a log list was replaced"
    assert {
        k: len(v) for k, v in logs.items()
    } == before_counts, "a probed row was appended"
    assert logs == before_copy, "a retrieved row was mutated by the probe pass"
    # And nowhere on the result either: no field may carry them out of this call.
    dumped = json.dumps(result.model_dump(mode="json"), default=str)
    assert (
        "shift.on_duty" not in dumped
    ), "a probed row's own field reached the correlation result"


@pytest.mark.asyncio
async def test_stage_health_is_identical_with_a_probe(pack, tmp_path):
    """A probe may not move the number that decides whether a gate opens.

    Restated from the free-rung version above because the failure mode is different: there,
    a nudge could only come from a computed field; here it can come from a row count, which
    is what `stage_health` reads most of.
    """
    probed = _probe_pack(tmp_path)
    off = await _probe_module(probed, probe=None, budget=0).analyze(
        _logs(), _understanding()
    )
    on = await _probe_module(probed, probe=_probe([_roster_row()])).analyze(
        _logs(), _understanding()
    )
    assert any(f.probe_spent for f in on.links), "no probe ran"

    a, b = score_stage("correlation", off), score_stage("correlation", on)
    assert a.scored, "the correlation stage scored nothing — the comparison is vacuous"
    assert (a.score, a.gate_recommended, list(a.reasons)) == (
        b.score,
        b.gate_recommended,
        list(b.reasons),
    ), "a rung-3 probe moved the stage health score"


@pytest.mark.asyncio
async def test_two_runs_with_a_probe_reproduce_the_same_findings(pack, tmp_path):
    """The gate-rejection path with a probe in it: a rejected stage RE-RUNS correlation.

    So the probe budget is per CALL and not per module, and the pass must not accumulate. A
    per-module counter would make run two of an unchanged incident probe less than run one and
    settle less — two runs disagreeing with nothing changed, which is the defect that once
    answered `PASS` then `UNKNOWN` on one incident.
    """
    probed = _probe_pack(tmp_path)
    module = _probe_module(probed, probe=_probe([_roster_row()]))
    logs = _logs()
    first = await module.analyze(logs, _understanding())
    second = await module.analyze(logs, _understanding())
    assert any(f.probe_spent for f in first.links), "no probe ran on the first pass"
    assert any(
        f.probe_spent for f in second.links
    ), "the second run probed nothing — the budget is per module, not per call"
    assert [f.model_dump(mode="json") for f in first.links] == [
        f.model_dump(mode="json") for f in second.links
    ], "a re-run of correlation produced different link findings"
    assert _sacred(first) == _sacred(second), "the probe left state behind"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,why",
    [
        ({"budget": 0}, "no budget is configured"),
        ({"mode": "planned"}, "the mode is planned"),
        (
            {"mode": "semi_auto"},
            "semi_auto is score-gated and this link scores below it",
        ),
    ],
)
async def test_the_bounds_each_refuse_the_spend_on_their_own(
    pack, tmp_path, kwargs, why
):
    """Four things license a probe, so each has to be shown to refuse it ALONE.

    A bound that only holds when another one also holds is a bound that disappears the day the
    other is configured. `semi_auto` is in the list because it is the only one whose refusal is
    the setting WORKING rather than something missing.
    """
    probed = _probe_pack(tmp_path)
    calls = []
    result = await _probe_module(
        probed, probe=_probe([_roster_row()], calls), **kwargs
    ).analyze(_logs(), _understanding())
    assert not calls, f"a probe was spent although {why}: {calls}"
    assert not any(
        f.probe_spent for f in result.links
    ), f"a probe was recorded although {why}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pack_kwargs,why",
    [
        (
            {"gate": "fail"},
            "rung 1 says the target procedure does not apply to this run",
        ),
        ({"gate": "unknown"}, "rung 1 could not be resolved on this run's rows"),
        (
            {"gate": "none"},
            "the target procedure declares no applicability test at all",
        ),
        ({"auto_probe": False}, "no signal on the pair declares auto_probe"),
    ],
)
async def test_an_UNLICENSED_pair_is_never_probed(pack, tmp_path, pack_kwargs, why):
    """The escalation licence, from the engine's side of it — and it is rung 1.

    Three of the four rows are the same licence at its three non-PASS values, three different facts
    about the target procedure of which only one is "it does not apply": `unknown` means the
    question was asked and not answered, `none` that the procedure never declared one. Both would
    auto-launch under a check written as `!= "fail"`, so each is asserted on its own.

    A base rate is NOT in this list: a historical statistic can only ADD to the semi_auto
    confidence score, never block an escalation this run's own evidence licensed
    (`test_link_probe.py::test_an_UNMEASURED_pair_is_probed_exactly_like_a_measured_one`).
    """
    probed = _probe_pack(tmp_path, **pack_kwargs)
    calls = []
    result = await _probe_module(probed, probe=_probe([_roster_row()], calls)).analyze(
        _logs(), _understanding()
    )
    assert not calls, f"a probe was spent although {why}: {calls}"
    assert not any(
        f.probe_spent for f in result.links
    ), f"a probe was recorded although {why}"
    # And the candidate is NOT lost: refusing to spend on it is not the same as not finding it.
    # The free rungs already ran, so the link is still in the report for a human to act on.
    assert any(
        f.target_use_case == "courier_collusion" for f in result.links
    ), "a link that was declined for a spend disappeared from the findings altogether"


@pytest.mark.asyncio
async def test_rung_1_PASS_is_what_makes_the_SAME_pack_escalate(pack, tmp_path):
    """The positive half of the four refusals above, on packs that differ in the gate ALONE.

    Every other declaration is identical, so the delta between spending a query and refusing to is
    the target procedure's own applicability test over this run's retrieved rows. Asserted as a
    pair because the claim is the difference: a licensed run that spends proves nothing on its own
    if the refusing run would have spent too.
    """
    licensed = await _probe_module(
        _probe_pack(tmp_path, gate="pass"), probe=_probe([_roster_row()])
    ).analyze(_logs(), _understanding())
    refused = await _probe_module(
        _probe_pack(tmp_path, gate="unknown"), probe=_probe([_roster_row()])
    ).analyze(_logs(), _understanding())

    spent = [f for f in licensed.links if f.probe_spent]
    assert spent, "rung-1 PASS did not license the spend the whole design rests on"
    assert (
        spent[0].gate_outcome == "pass"
    ), f"the finding did not record the licence it escalated on: {spent[0].gate_outcome!r}"
    assert not any(f.probe_spent for f in refused.links)
    held = [f for f in refused.links if f.target_use_case == "courier_collusion"]
    assert held and held[0].gate_outcome == "unknown", (
        "the unresolved gate was not recorded on the finding, so a reader cannot tell this link "
        f"from one nobody looked at: {[f.gate_outcome for f in refused.links]}"
    )


@pytest.mark.asyncio
async def test_a_DISCONFIRMING_probe_withdraws_the_licence_it_arrived_with(
    pack, tmp_path
):
    """Rung 3's whole reason to exist: one cheap query that may take rung 4's licence away.

    The gate that licensed the probe is re-evaluated over the probe's own rows, so a query costing
    one scan can refuse an escalation that would otherwise cost a whole child run. A probe that
    could only ever confirm would be a formality between rung 2 and rung 4.

    The pack differs from the licensing one in ONE declaration, a second `gate: scope` condition on
    the un-retrieved source (:data:`GATE_FLIPS_ON_PROBE`), and that shape is not contrived: a gate
    declared SOLELY on the probed source reads `unknown` on the free rungs, so nothing licenses the
    probe that resolves it. Beside a condition that already passes it is reachable, a gate
    aggregating FAIL before PASS.

    Both halves are asserted, the withdrawal being worth nothing if rung 4 does not read it.
    """
    flips = _probe_pack(tmp_path, gate=GATE_FLIPS_ON_PROBE)
    # The control is the SAME pack with a probe that could not ask, so the flip below is
    # attributable to the answer and not to a pack that was never licensed in the first place.
    unasked = await _probe_module(flips, probe=_probe(None)).analyze(
        _logs(), _understanding()
    )
    disconfirmed = await _probe_module(flips, probe=_probe([_roster_row()])).analyze(
        _logs(), _understanding()
    )

    before = [f for f in unasked.links if f.target_use_case == "courier_collusion"][0]
    after = [f for f in disconfirmed.links if f.target_use_case == "courier_collusion"][
        0
    ]
    assert before.gate_outcome == "pass" and before.mode in ESCALATING_MODES, (
        "the control never held the licence, so the flip below proves nothing: "
        f"{before.gate_outcome!r} / {before.mode!r}"
    )
    assert after.probe_spent is True, "the licensed probe was never spent"
    assert after.gate_outcome == "fail", (
        "the probe's rows were fetched and the gate was not re-evaluated over them — rung 3 "
        f"spent a query and read nothing back into the licence: {after.gate_outcome!r}"
    )
    assert after.state == "probed_negative", (
        f"a disconfirmed candidate settled {after.state!r}: the four states must not collapse, "
        "and this is the one an operator reads as 'looked, and it is not there'"
    )
    assert after.mode not in ESCALATING_MODES, (
        f"the mode stayed {after.mode!r} after the licence was withdrawn, so the next rung is "
        "still armed on a gate that has since failed"
    )

    # THE OTHER HALF: rung 4 must actually refuse it — the mode and the state are advisory fields
    # until something reads them. The outer `correlation` block, which is what the runner passes:
    # an inner-block-only config refuses on `children_disabled` and every assertion below then
    # passes for that reason.
    child_config = {"links": {"max_children_per_run": 2}}
    spawns, refusals = plan_child_spawns(
        [after],
        incident={},
        config=child_config,
        pack=flips,
        parent_use_case="refund_fraud",
    )
    assert (
        spawns == []
    ), "a child run was launched on a candidate the probe had just refuted"
    assert [r["code"] for r in refusals] == ["not_escalating"], (
        "the withdrawal reached rung 4 as something other than the mode it clamped: "
        f"{[r['code'] for r in refusals]}"
    )

    # And a SECOND barrier, independent of that clamp — re-arming the mode by hand is the only way
    # to ask whether rung 4 has one. It must: the clamp is a decision taken from the score and the
    # mode is authored in four places, so a candidate can reach here escalating over a licence that
    # no longer holds. The barrier is the state the probe wrote. `gate_not_pass` is a THIRD and is
    # deliberately not what fires here — it is for a candidate that never had a rung-1 PASS at all,
    # and it is asserted on that shape in `test_link_children.py`.
    after.mode = "auto"
    rearmed_spawns, rearmed_refusals = plan_child_spawns(
        [after],
        incident={},
        config=child_config,
        pack=flips,
        parent_use_case="refund_fraud",
    )
    assert (
        rearmed_spawns == []
    ), "an escalating mode overrode a state the probe had refuted"
    assert [r["code"] for r in rearmed_refusals] == ["not_confirmed"], (
        "rung 4 declined for some other reason, so the refusal would disappear the moment that "
        f"other bound was raised: {[r['code'] for r in rearmed_refusals]}"
    )


@pytest.mark.asyncio
async def test_a_probe_that_could_not_ASK_is_not_a_probe_that_found_nothing(
    pack, tmp_path
):
    """`None` and `[]` are different answers, and the finding must say which it got.

    The same rule as `unanswered_out` one stage earlier: a source that did not answer must not
    read as one that answered with nothing. Here it decides what an operator does next — a
    refusal needs a credential or a catalog entry, an empty answer needs a referral.
    """
    probed = _probe_pack(tmp_path)
    refused = await _probe_module(probed, probe=_probe(None)).analyze(
        _logs(), _understanding()
    )
    empty = await _probe_module(probed, probe=_probe([])).analyze(
        _logs(), _understanding()
    )

    a = [f for f in refused.links if f.target_use_case == "courier_collusion"][0]
    b = [f for f in empty.links if f.target_use_case == "courier_collusion"][0]
    assert a.probe_spent is False, "a probe that could not ask was recorded as spent"
    assert a.probe_note, "a refused probe left no trace at all"
    assert (
        b.probe_spent is True
    ), "a probe that answered with zero rows was not recorded"
    assert a.probe_note != b.probe_note, (
        "a refused probe and an empty answer print the same sentence — the reader cannot "
        "tell a missing credential from a source with nothing to say"
    )
    # An empty answer settles nothing OF ITS OWN, and the state says what IS settled rather than
    # what this probe found: rung 1 held before the probe was spent and an empty answer neither
    # confirms nor withdraws that. So the state must be unchanged, not reset.
    assert b.state == a.state == "probed_positive", (
        f"an empty probe answer moved the settlement to {b.state!r} against the non-answer's "
        f"{a.state!r} — a scan that found nothing is not a ruling-out, and neither is one that "
        "could not be run"
    )
    assert "0 row(s)" in b.probe_note, (
        "the empty answer is not visible anywhere: the state is the free rungs' and the note is "
        "the only place a reader learns a query was spent and came back with nothing"
    )


@pytest.mark.asyncio
async def test_a_raising_probe_leaves_the_stage_and_the_verdict_intact(pack, tmp_path):
    """The advisory lane may not fail the run it advises on, probe included."""

    async def _boom(source, analysis, **kw):
        raise RuntimeError("the backend refused")

    probed = _probe_pack(tmp_path)
    off = await _probe_module(probed, probe=None, budget=0).analyze(
        _logs(), _understanding()
    )
    on = await _probe_module(probed, probe=_boom).analyze(_logs(), _understanding())
    assert on.verdict is not None, "a raising probe cost the run its verdict"
    assert _sacred(off) == _sacred(on), "a raising probe moved the sacred channel"
    assert on.links, "a raising probe cost the run its free-rung findings"


def test_the_probe_budget_can_never_reach_the_primary_source_timeout():
    """The fan-out bound, as arithmetic rather than as a comment.

    `primary_source_timeout_seconds` defaults to 7200s and a probe is an advisory scan: the
    worst case of the whole rung-3 fan-out must not be able to approach the budget the system
    of record is granted, or a link nobody has read yet delays the verdict it is advisory to.
    Asserted against the engine's own constant, so raising one without the other is a failure
    here rather than a discovery in production.
    """
    # The bounds live beside `escalation_budgeted`, in the module resolving every other escalation
    # setting from the same config block: one reader of `max_probes_per_run`, not two that could
    # disagree. The config editor imports them too, which is what keeps a field's `maximum=` and
    # the engine's ceiling from becoming two numbers.
    from src.link_escalation import (
        MAX_PROBES_CEILING,
        PROBE_BUDGET_CEILING_SECONDS,
        PROBE_TIMEOUT_MAX,
        probe_budget,
    )
    from src.log_retrieval import LogRetrievalEngine

    # Read off the engine class, where it lives: a copy of the number here would be the second
    # answer this assertion exists to prevent.
    _PRIMARY_TIMEOUT_SECONDS = LogRetrievalEngine._PRIMARY_TIMEOUT_SECONDS

    assert PROBE_BUDGET_CEILING_SECONDS < _PRIMARY_TIMEOUT_SECONDS, (
        f"the whole probe fan-out may run for {PROBE_BUDGET_CEILING_SECONDS}s against a "
        f"{_PRIMARY_TIMEOUT_SECONDS}s primary-source budget"
    )
    assert MAX_PROBES_CEILING * PROBE_TIMEOUT_MAX >= PROBE_BUDGET_CEILING_SECONDS, (
        "the per-probe caps cannot reach the collective ceiling, so the ceiling is dead code "
        "and the real bound is somewhere else"
    )
    # A configured value cannot escape either ceiling, in both directions and at the absurd end.
    wild = probe_budget({"max_probes_per_run": 10_000, "probe_timeout_seconds": 10_000})
    assert wild["max_probes"] <= MAX_PROBES_CEILING
    assert wild["timeout"] <= PROBE_TIMEOUT_MAX
    assert wild["max_probes"] * wild["timeout"] <= PROBE_BUDGET_CEILING_SECONDS
    assert wild["deadline_seconds"] <= PROBE_BUDGET_CEILING_SECONDS
    # A negative value disarms rather than inverting a comparison, while an absent or unreadable
    # one reads as the shipped budget — the same reading `escalation_budgeted` takes, so the note
    # about an empty budget cannot disagree with the budget the rung resolves.
    from src.link_escalation import DEFAULT_MAX_PROBES_PER_RUN, escalation_budgeted

    assert probe_budget({"max_probes_per_run": -3})["max_probes"] == 0
    for same in ({"max_probes_per_run": "many"}, {}):
        assert probe_budget(same)["max_probes"] == DEFAULT_MAX_PROBES_PER_RUN, same
        assert escalation_budgeted(same) is True, same


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


def test_the_link_pass_runs_after_the_verdict_and_the_brief():
    """Read the call site as an AST, not as a comment.

    A link computed BEFORE the verdict is one refactor away from being available to it, and
    the value comparison above cannot see the difference on a fixture where the link happens
    to change nothing. Position is the structural half of the invariant.
    """
    tree = _correlation_tree()
    links = _call_lines(tree, "assess_links")
    assert links, "`assess_links` is not called from src/correlation.py"
    assert (
        len(links) == 1
    ), f"the link pass is called {len(links)} times; it must be once"

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
    assert links[0] > max(verdicts), "the link pass runs BEFORE the verdict"
    assert links[0] > max(briefs), "the link pass runs BEFORE the case builder"


def test_the_link_findings_land_on_nothing_but_a_links_attribute():
    """Whatever `assess_links` returns may be assigned to `*.links` and to nothing else.

    This is the assertion that survives a refactor: it does not care what the pass computes,
    only that its output has exactly one destination. A link reaching any other attribute of
    `result` is a link reaching a channel some reader treats as this run's finding.
    """
    tree = _correlation_tree()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Call)
            and (
                (isinstance(value.func, ast.Name) and value.func.id == "assess_links")
                or (
                    isinstance(value.func, ast.Attribute)
                    and value.func.attr == "assess_links"
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
            assert name in {"links", "link_findings"}, (
                f"`assess_links(...)` is assigned to `{name}` at line {node.lineno} — a link "
                "may only reach the advisory lane"
            )


def test_the_verdict_engine_takes_no_link_argument():
    """`evaluate_verdict` may not learn about links, by parameter or by keyword.

    A parameter is how "advisory" becomes "an input the rollup happens to read". The engine's
    own signature is the cheapest place to make that impossible.
    """
    import inspect

    from src.correlation import evaluate_verdict

    params = set(inspect.signature(evaluate_verdict).parameters)
    assert not {
        p for p in params if "link" in p.lower()
    }, f"evaluate_verdict accepts a link-shaped parameter: {sorted(params)}"


# --- the guard on the guards -------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_comparison_can_actually_fail(pack):
    """A mutation test on `_sacred`: prove the comparison objects to a real difference.

    Two runs of the same code are trivially equal, so every assertion above is one silent bug
    in this helper away from being vacuous — the exact shape of "a check that cannot fire is
    indistinguishable from one that found nothing". Each sacred field is perturbed
    independently, because a helper that only reads `verdict` would still pass a single-field
    mutation test.
    """
    result = await _module(pack, False).analyze(_logs(), _understanding())
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


def test_the_advisory_lane_is_not_the_verdict_severity(pack):
    """A `LinkFinding`'s severity is its own field, named so a reader cannot confuse the two.

    The operator must be able to read "verdict severity HIGH (this procedure, N conditions)"
    and "correlation advisory HIGH (router-added, act via Refer)" as two different sentences.
    A `LinkFinding` reusing the verdict's own `severity` name is how the second gets quoted as
    the first in a handover.
    """
    from src.models.pydantic_models import LinkFinding

    fields = set(LinkFinding.model_fields)
    assert "advisory_severity" in fields, "the advisory lane has no severity of its own"
    assert "severity" not in fields, (
        "LinkFinding carries a bare `severity` — it must be `advisory_severity`, because the "
        "name is what stops a router-added triage default being read as a verdict"
    )


def test_every_link_state_is_declared_and_distinct():
    """The four states may never collapse: "did not look" is not "looked and found nothing".

    The second is a FINDING — the same distinction `unanswered_out` draws against an empty
    result, and the one a reader loses first. `probed_negative` rendering as silence is how a
    procedure that was checked and ruled out becomes indistinguishable from one nobody asked
    about.
    """
    from src.links import LINK_STATES

    assert set(LINK_STATES) == {
        "unreachable",
        "not_probed",
        "probed_negative",
        "probed_positive",
    }, f"the link states drifted: {sorted(LINK_STATES)}"


@pytest.mark.parametrize("module", ["links.py", "link_escalation.py"])
def test_the_links_module_is_pure(module):
    """No LLM, no IO, no clock — `assess_links` is `evaluate_verdict`'s posture, restated.

    Read as an AST rather than trusted from a docstring. A link pass that could fetch would be
    a retrieval stage hiding inside correlation, with none of the budgets, none of the caps,
    and none of the per-source timeouts that make retrieval safe.

    Both files, because the escalation seam is where the pressure to reach outward will land: it
    is the module that decides whether something gets SPENT, so a future edit resolving a mode by
    asking a service — or by consulting a clock, which is how a "quiet hours" rule would arrive —
    would make the mode non-reproducible, and the re-run assertions above rest on the whole pass
    resolving the same way twice.
    """
    name = f"src/{module}"
    tree = ast.parse((REPO_ROOT / "src" / module).read_text(encoding="utf-8"))
    banned_modules = {
        "requests",
        "httpx",
        "aiohttp",
        "urllib",
        "socket",
        "random",
        "subprocess",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in banned_modules, f"{name} imports {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in banned_modules, f"{name} imports from {node.module}"
        # A clock makes the pass non-reproducible, which is what breaks the re-run above.
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


_LINKS_SEAMS = {"assess_links", "compose_referral", "resettle_with_probe"}


def test_the_module_exposes_ONE_seam_PER_THING_IT_DOES():
    """Three seams, so there are three places to audit — and no fourth.

    Each is a different verb over the same declarations: `assess_links` reads the run and produces
    the findings, `compose_referral` turns ONE finding into the request a child run would need,
    `resettle_with_probe` re-reads ONE finding with one more source in view. All three are public
    because something outside this file reaches each of them, and all three are HERE rather than in
    their callers because they must stay as pure and as deterministic as each other.

    `resettle_with_probe` is the one worth justifying, the tempting shape being to put it in
    `src/link_probe.py` beside the fetch. A sibling's own applicability gate outranking any number
    of its indicators is a rule that may have exactly one implementation, and a copy in the module
    that fetches would be a second answer to which reading wins.

    The set is asserted exactly: a fourth public name is how the advisory lane would grow a path
    that launches a run without the invariant tests above ever seeing it.
    """
    import src.links as links_mod

    public = {
        n
        for n in dir(links_mod)
        if not n.startswith("_") and callable(getattr(links_mod, n))
    }
    # Imported helpers are not this module's API; only names DEFINED here count.
    tree = ast.parse(pathlib.Path(links_mod.__file__).read_text(encoding="utf-8"))
    defined = {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not n.name.startswith("_")
    }
    assert defined == _LINKS_SEAMS, (
        f"src/links.py exposes {sorted(defined)} — the advisory lane has exactly three seams: "
        "one that assesses, one that composes, one that re-settles"
    )
    assert _LINKS_SEAMS <= public


def test_only_the_FETCHING_module_may_reach_a_backend():
    """The split that makes rung 3 safe, asserted as a fact about two files.

    `src/links.py` is scanned for purity above, so the settlement cannot fetch. The complement
    is that `src/link_probe.py` must not re-implement the settlement: it may call the seam, and
    it may not read this module's privates. An underscore name crossing that line is how the
    precedence rule would quietly acquire a second implementation on the side that spends.
    """
    tree = ast.parse((REPO_ROOT / "src" / "link_probe.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("links"):
            imported |= {a.name for a in node.names}
    assert imported, "src/link_probe.py reaches no seam in src/links.py at all"
    assert imported <= _LINKS_SEAMS, (
        f"src/link_probe.py imports {sorted(imported - _LINKS_SEAMS)} from src/links.py — a "
        "private reached across modules is a private that has stopped being one, and the "
        "ladder's precedence may have exactly one implementation"
    )
