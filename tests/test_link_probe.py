"""Rung 3's FETCHING half: what a probe asks, and every way it is bounded.

`test_links_never_change_the_verdict.py` owns the other half — that a probe's rows cannot reach
the verdict, the health score or `logs`. This file owns the question that one cannot ask: given
that a probe is licensed, **what does it actually send, and what does it do with each of the
three answers it can get back?**

Two distinctions carry most of the file, and both are ones this codebase has already paid for
elsewhere:

* **A source that did not ANSWER is not a source that answered with nothing.** `_gather` draws
  that line one stage earlier with `unanswered_out`, and a probe has to draw it again or a timed
  -out scan reads as "the sibling procedure's shape is absent" — a confident negative produced by
  an environment failure.
* **A row cap is not a time cap.** A probe returning exactly its cap is TRUNCATED, and it says
  so, because the sibling's own applicability test is entitled to know it read a bounded result.

THE PACK IS `knowledge/mock_domain/`. Nothing here is domain-specific: the ladder, the bounds and
the three answers are engine behaviour over whatever a pack declares, so they are proven on the
fixture pack that ships on every branch. The one source a probe can go and get
(`mock_domain_links.PROBE_SOURCE`) is deliberately absent from that pack's catalog — a pack whose
every source is already retrieved cannot express this rung at all.
"""

import asyncio

import pytest

from src.link_probe import (
    REPORTABLE_REFUSAL,
    build_link_probe,
    probe_candidates,
    probe_eligible,
    probe_nominations,
    probe_nominations_already_answered,
    probe_refusal,
    run_link_probes,
)
from src.models.pydantic_models import (
    ExtractedEntity,
    IncidentAnalysis,
    LinkFinding,
)
from tests.mock_domain_links import PROBE_SOURCE, probe_pack, roster_row

# --- doubles ----------------------------------------------------------------------------


class _Generator:
    """The half of `ApiCallGenerator` a probe uses, and nothing else.

    `build_manual_query` is the seam the operator's plan editor already goes through, so a probe
    going through it too is the assertion `test_the_probe_goes_through_the_operator_seam` makes.
    It raises for a source that built no retriever, which is the real one's contract.
    """

    def __init__(self, known=(PROBE_SOURCE,), raises=None):
        self.known = set(known)
        self.raises = raises
        self.calls = []

    def build_manual_query(self, analysis, source, question="", window=None):
        self.calls.append(
            {"analysis": analysis, "source": source, "question": question}
        )
        if self.raises is not None:
            raise self.raises
        if source not in self.known:
            raise ValueError(f"no retriever is configured for '{source}'")
        return {"source": source, "question": question}


class _Engine:
    """`LogRetrievalEngine.retrieve`'s contract, in the two shapes that matter.

    `answer=None` means the source is ABSENT from the returned dict and named in
    `unanswered_out` — the non-answer. `answer=[]` means PRESENT with an empty list — the empty
    answer. Collapsing the two is the defect this double exists to make visible.
    """

    def __init__(self, answer=None, unanswered_reason="TimeoutError"):
        self.answer = answer
        self.unanswered_reason = unanswered_reason
        self.calls = []

    async def retrieve(self, queries, **kw):
        self.calls.append({"queries": list(queries), **kw})
        source = (queries[0] or {}).get("source")
        if self.answer is None:
            out = kw.get("unanswered_out")
            if isinstance(out, dict):
                out[source] = self.unanswered_reason
            return {}
        return {source: [dict(r) for r in self.answer]}


def _analysis():
    return IncidentAnalysis(
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
        ],
        correlation_keys=[],
    )


def _finding(**over):
    """A candidate in exactly the state a probe is for: licensed, unspent, pivot in hand.

    `gate_outcome: "pass"` is the licence, not decoration. Rung 1 — the target procedure's own
    scope gate re-evaluated over THIS run's rows — is what permits an escalation, so a candidate
    without it is refused before the pack is ever read. `state` is `not_probed` here only because
    that is what an unspent candidate looks like; it is no longer a licence of its own.
    """
    fields = {
        "target_use_case": "courier_collusion",
        "direction": "consequent",
        "state": "not_probed",
        "rung": 2,
        "pivot_entity": "shipment",
        "pivot_values": ["RT48192043"],
        "gate_outcome": "pass",
        "mode": "auto",
        "mode_source": "config",
    }
    fields.update(over)
    return LinkFinding(**fields)


# --- 1. what the probe SENDS ------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_probe_goes_through_the_operator_seam_and_not_around_it():
    """One query, built by `build_manual_query`, handed to `retrieve` — and rows come back.

    The seam matters more than the round trip: which entities a source can bind, which window
    applies and which guards the text must pass are all pack knowledge, and a probe assembling
    its own query would be a second answer to three questions this system answers once. It is
    also the seam that already exists for a human naming a source, which is what a probe is.
    """
    gen, engine = _Generator(), _Engine(answer=[roster_row()])
    probe = build_link_probe(gen, engine)

    rows = await probe(PROBE_SOURCE, _analysis(), row_cap=50)

    assert [c["source"] for c in gen.calls] == [PROBE_SOURCE]
    assert len(engine.calls) == 1, "a probe is ONE query, whatever it finds"
    assert rows == [roster_row()]


@pytest.mark.asyncio
async def test_the_pivot_is_ADDED_to_the_understanding_and_never_substituted():
    """The probe is scoped by the sibling's subject value AND the incident's own facts.

    Substituting would widen the scan the bounds exist to keep small — the incident's window and
    its organisational scope are what make one source answerable in seconds. And the copy is the
    point: the shared analysis scoped every query the verdict rests on, so a probe rescoping it
    in place would change what anything reading it afterwards sees.
    """
    gen, engine = _Generator(), _Engine(answer=[])
    analysis = _analysis()
    before = [(e.type, e.value) for e in analysis.extracted_entities]

    await build_link_probe(gen, engine)(
        PROBE_SOURCE, analysis, pivot_entity="depot", pivot_values=["BHM11"]
    )

    scoped = gen.calls[0]["analysis"]
    seen = [(e.type, e.value) for e in scoped.extracted_entities]
    assert ("depot", "BHM11") in seen, "the pivot never reached the query"
    for pair in before:
        assert (
            pair in seen
        ), f"the incident's own {pair} was dropped from the probe's scope"
    assert [
        (e.type, e.value) for e in analysis.extracted_entities
    ] == before, "the probe mutated the analysis the verdict was built from"


@pytest.mark.asyncio
async def test_a_pivot_already_in_hand_is_not_added_twice():
    """A pivot the incident already named is not duplicated onto the query.

    Two identical entities are two identical OR arms, which is harmless in the result and noise
    in the text — and the text is what a query guard reads.
    """
    gen, engine = _Generator(), _Engine(answer=[])
    await build_link_probe(gen, engine)(
        PROBE_SOURCE, _analysis(), pivot_entity="shipment", pivot_values=["RT48192043"]
    )
    seen = [(e.type, e.value) for e in gen.calls[0]["analysis"].extracted_entities]
    assert seen.count(("shipment", "RT48192043")) == 1


# --- 2. the three answers ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_source_that_did_not_ANSWER_is_not_one_that_answered_with_nothing():
    """The distinction `unanswered_out` draws, redrawn here — `None` against `[]`.

    Both come back from `retrieve` looking like "no rows for this source" unless the caller reads
    the register. The consequence of collapsing them is a confident negative: an environment
    failure would settle the candidate as "the sibling's shape is absent from this evidence",
    which is a finding, and the empty answer genuinely is one.
    """
    gen = _Generator()
    unanswered = await build_link_probe(gen, _Engine(answer=None))(
        PROBE_SOURCE, _analysis()
    )
    empty = await build_link_probe(gen, _Engine(answer=[]))(PROBE_SOURCE, _analysis())

    assert unanswered is None
    assert empty == [], "an empty ANSWER must stay a list, or it reads as a non-answer"


@pytest.mark.asyncio
async def test_a_query_that_could_not_be_BUILT_is_a_non_answer_and_not_a_crash():
    """A source no retriever was built for is "could not ask", not an exception.

    `build_manual_query` raises `ValueError` for exactly that, and it is a normal outcome: a
    declared source with no credentials in this deployment is the case the whole `unavailable`
    reporting lane exists for. Raising here would take the advisory pass — and the log line it
    was about to write — down with it.
    """
    probe = build_link_probe(_Generator(known=()), _Engine(answer=[roster_row()]))
    assert await probe(PROBE_SOURCE, _analysis()) is None

    # And any other exception the same way: a generator is LLM-backed in production, so the
    # failure modes are not enumerable and the conservative reading is the only safe one.
    boom = build_link_probe(
        _Generator(raises=RuntimeError("endpoint refused")), _Engine(answer=[])
    )
    assert await boom(PROBE_SOURCE, _analysis()) is None


@pytest.mark.asyncio
async def test_the_probe_truncates_to_its_OWN_row_cap():
    """A row cap is not a time cap, and the probe's cap is tighter than the backend's.

    A probe asks whether a shape is present. Reading a page to answer that makes it a retrieval,
    which is rung 4's job — so the cap is applied to what the settlement is handed, not left to
    whatever the source's own `max_results` happened to be.
    """
    engine = _Engine(answer=[roster_row(badge=f"{i:04d}") for i in range(40)])
    rows = await build_link_probe(_Generator(), engine)(
        PROBE_SOURCE, _analysis(), row_cap=7
    )
    assert len(rows) == 7


def test_nothing_is_built_when_either_half_is_missing():
    """No generator or no engine means the rung is unreachable — and that is not an error.

    Opt-in at the WIRING level as well as at the config level. A deployment with no retrieval
    engine cannot probe, and returning `None` here is what lets `CorrelationModule` treat "not
    wired" and "budget of zero" as the same answer instead of two.
    """
    assert build_link_probe(None, _Engine()) is None
    assert build_link_probe(_Generator(), None) is None
    assert build_link_probe(None, None) is None
    assert callable(build_link_probe(_Generator(), _Engine()))


# --- 3. which source, and whether at all ------------------------------------------------


def test_the_nominated_source_is_preferred_and_a_retrieved_one_is_never_re_asked(
    tmp_path,
):
    """Order is by which declaration ASKED, and `logs` subtracts — including a zero-row answer.

    A signal carrying `auto_probe: true` names the source the pack's author nominated for exactly
    this spend, so it comes first. And a source already in `logs` is excluded whatever it
    answered: an empty result is an ANSWER the free rungs already read, so re-asking would spend
    a query to reproduce it.
    """
    pack = probe_pack(tmp_path)
    assert probe_candidates(pack, "courier_collusion", {})[0] == PROBE_SOURCE

    already = probe_candidates(pack, "courier_collusion", {PROBE_SOURCE: []})
    assert (
        PROBE_SOURCE not in already
    ), "a source that answered with zero rows was queued for a second, identical query"


@pytest.mark.parametrize(
    "over,why",
    [
        ({"mode": "planned"}, "a non-escalating mode may compose, never spend"),
        ({"gate_outcome": "fail"}, "rung 1 said the target procedure does not apply"),
        ({"gate_outcome": "unknown"}, "an unresolved rung 1 is not a rung-1 PASS"),
        ({"gate_outcome": ""}, "rung 1 was never reached, so nothing licenses a spend"),
        (
            {"probe_spent": True},
            "one probe per candidate; a budget is not a retry allowance",
        ),
        ({"state": "unreachable"}, "no query closes a binding gap in the pack"),
        ({"pivot_values": []}, "a probe confirms a pivot; it does not go and find one"),
        ({"target_use_case": ""}, "there is no procedure to probe on behalf of"),
    ],
)
def test_each_licence_refuses_on_its_own(tmp_path, over, why):
    """Eight refusals, one at a time — because a gate that only works in combination is one gate.

    Asserted through `probe_eligible` rather than through the loop, so the answer to "which of
    them refused" is a property of one function. The BUDGET is deliberately not among them: it
    belongs to the run rather than to the candidate, and the loop that spends it owns it.

    Three of the eight rows are one licence — rung 1 — read at its three non-PASS values, because
    `fail`, `unknown` and "never reached" are three different things an operator does three
    different things about, and a check written as `!= "fail"` would pass the first row and let the
    other two spend. What is deliberately NOT a refusal any more is a SETTLED candidate: rung-1
    PASS settles a finding at `probed_positive` by construction, so requiring an unsettled state
    would leave rung 3 unreachable for exactly the candidates that earned it.
    """
    pack = probe_pack(tmp_path)
    assert probe_eligible(_finding(), pack) == PROBE_SOURCE, "the control is vacuous"
    assert probe_eligible(_finding(**over), pack) is None, why


def test_a_SETTLED_candidate_is_still_probeable_because_a_probe_may_DISCONFIRM_it(
    tmp_path,
):
    """The inverse of the refusal above, and the whole reason rung 3 survived the redesign.

    Rung 1 is now the escalation licence, and `_settle` maps a gate PASS to `probed_positive`. So
    every candidate a probe could be spent on is already settled, and rung 3's job is no longer to
    break a tie: it is the one cheap query that can take a gate-PASS candidate away before rung 4
    spends a whole child run on it. A `state` licence here would silently disarm that.
    """
    pack = probe_pack(tmp_path)
    settled = _finding(state="probed_positive", rung=3)
    assert probe_eligible(settled, pack) == PROBE_SOURCE


def test_an_UNMEASURED_pair_is_probed_exactly_like_a_measured_one(tmp_path):
    """A historical base rate gates nothing — it is an optional term in the confidence score.

    This test is the inversion of one that asserted the opposite: a pair below the measurement
    corpus used to be refused by the code that spends, mirroring a `pack_validate` ERROR. Both are
    gone. The licence is the pack's declaration plus THIS run's rung-1 outcome, so a pair nobody
    has ever measured is probed on the same terms as one measured over a thousand incidents — and
    a stub measurement can no longer block an escalation the evidence licensed.
    """
    for corpus in (0, 4):
        assert probe_eligible(_finding(), probe_pack(tmp_path, corpus=corpus)) == (
            PROBE_SOURCE
        ), f"a corpus of {corpus} refused a link this run's own evidence licensed"


def test_a_pair_whose_declarations_never_ASKED_is_not_probed_off_its_gate_source(
    tmp_path,
):
    """The other half, and it needs the ruleset that HAS a scope gate to be provable at all.

    A candidate source comes from two places: a signal that nominated one, and the target's own
    applicability-gate sources — a gate reading `unknown` for want of its source being the other
    way a candidate settles neither way. On `courier_collusion`, which declares no gate, the
    `auto_probe` refusal is over-determined: with the check removed the candidate list is empty
    anyway, so a test there passes whether or not the declaration is read. `refund_fraud` declares
    a gate, so its gate source stands as a candidate on its own merits — and refusing it is then
    attributable to the declaration and to nothing else.

    This is the two-edit shape `tests/CLAUDE.md` records: a mutation caught by a second guard
    downstream reads exactly like an assertion that holds, and only the case where the second
    guard is silent discriminates the two.
    """
    asked = probe_pack(tmp_path, use_case="refund_fraud")
    finding = _finding(target_use_case="refund_fraud")
    assert probe_eligible(finding, asked) == PROBE_SOURCE, "the control is vacuous"

    unasked = probe_pack(tmp_path, use_case="refund_fraud", auto_probe=False)
    assert probe_candidates(
        unasked, "refund_fraud", {}
    ), "the gate source must still be a candidate, or this test proves nothing"
    assert (
        probe_eligible(finding, unasked) is None
    ), "a pair whose declarations never nominated a probe was probed off its gate source"


# --- 3b. WHY a probe was refused, which is not the same fact as that it was ---------------


def test_every_refusal_names_ITSELF_and_no_two_of_them_share_a_name(tmp_path):
    """The eight refusals return eight distinct codes, and the licensed case returns none.

    The sibling of `test_each_licence_refuses_on_its_own`, asking the question that one cannot: it
    proves each licence refuses ALONE, and this proves each refusal is DISTINGUISHABLE afterwards.
    Both are needed, because the defect they bracket is not a gate that fails to fire — it is
    eight unrelated facts arriving at every surface as one silence, so a reader cannot tell a pack
    that declared no probe from one whose licence was withheld from one whose nominated source was
    already in hand. Asserted as a SET rather than row by row: any two codes collapsing into one
    re-creates exactly the ambiguity the codes exist to remove, and a per-row assertion would pass
    while two rows agreed.
    """
    pack = probe_pack(tmp_path)
    source, reason = probe_refusal(_finding(), pack)
    assert (source, reason) == (PROBE_SOURCE, ""), "the control is vacuous"

    refusals = {
        "mode": {"mode": "planned"},
        "gate_fail": {"gate_outcome": "fail"},
        "spent": {"probe_spent": True},
        "unreachable": {"state": "unreachable"},
        "no_pivot": {"pivot_values": []},
        "no_target": {"target_use_case": ""},
    }
    codes = {}
    for label, over in refusals.items():
        src, code = probe_refusal(_finding(**over), pack)
        assert src is None, f"{label} was licensed"
        assert code, f"{label} refused without saying why"
        codes[label] = code
    assert len(set(codes.values())) == len(codes), f"two refusals share a code: {codes}"


def test_a_NOMINATION_this_run_already_answered_is_its_own_refusal_and_not_silence(
    tmp_path,
):
    """The one refusal that is a fact about the DECLARATION, so the one that must be visible.

    This is the shape a live run exhibited. The pack nominated a probe source, every licence
    held, the budget was there — and the source was already in `logs`, so the spend was correctly
    declined: the free rungs have read those rows and re-asking would pay a scan to reproduce an
    answer in hand. Correct, and until now indistinguishable from a pair that never declared a
    probe at all, which is the opposite conclusion about the pack.

    Both directions are asserted because only their DIFFERENCE carries the meaning: an
    already-answered nomination and no nomination at all both leave `probe_candidates` empty, so a
    test that only checked "no probe was spent" passes on either reading.
    """
    pack = probe_pack(tmp_path)
    # Built FROM the nomination list rather than pinned to one name: this ruleset nominates its
    # gate source beside the signal's, and the refusal is "every nominated source is in hand".
    # A hand-written one-key dict leaves a candidate standing and the probe is licensed after all.
    nominated = probe_nominations(pack, "courier_collusion")
    assert len(nominated) >= 2, "the fixture must nominate more than one source here"
    logs = {name: [] for name in nominated}

    src, code = probe_refusal(_finding(), pack, logs)
    assert src is None, "a source already in logs was queued for a second identical query"
    assert code == REPORTABLE_REFUSAL
    assert probe_nominations_already_answered(pack, "courier_collusion", logs) == nominated

    # The same empty candidate list reached the other way: nothing was ever nominated. A
    # DIFFERENT code, or the declaration's own inertness reads as the pack's silence.
    unasked = probe_pack(tmp_path, use_case="refund_fraud", auto_probe=False)
    _, other = probe_refusal(_finding(target_use_case="refund_fraud"), unasked, {})
    assert other != REPORTABLE_REFUSAL, (
        "a pair that nominated nothing was reported as one whose nomination was already answered"
    )
    assert probe_nominations_already_answered(unasked, "refund_fraud", {}) == []


@pytest.mark.asyncio
async def test_an_already_answered_nomination_is_NOTED_on_the_finding_and_costs_nothing(
    tmp_path,
):
    """It reaches a surface, it spends no probe, and it leaves the free-rung settlement alone.

    Three separate claims, and the third is the one that makes this safe to report at all: the
    note goes on through the settlement's own no-rows path, so `probe_spent` stays False and the
    state, rung and evidence wording a probe never touched are unchanged. A note that also
    rewrote the settlement would be narrating a probe that did not happen.
    """
    pack = probe_pack(tmp_path)
    finding = _finding()
    before = (finding.state, finding.rung, finding.evidence_note)
    logs = {name: [roster_row()] for name in probe_nominations(pack, "courier_collusion")}

    async def _never(*a, **kw):  # pragma: no cover — asserted never awaited
        raise AssertionError("a probe was spent on a source this run had already retrieved")

    spent = await run_link_probes(
        _never,
        [finding],
        pack=pack,
        analysis=_analysis(),
        logs=logs,
        config={"max_probes_per_run": 4, "probe_timeout_seconds": 5},
    )
    assert spent == 0
    assert finding.probe_spent is False
    assert PROBE_SOURCE in finding.probe_note
    assert "already retrieved" in finding.probe_note
    assert (finding.state, finding.rung, finding.evidence_note) == before, (
        "the note rewrote a settlement no probe had re-read"
    )


@pytest.mark.asyncio
async def test_a_pair_that_nominated_NOTHING_gets_no_note_at_all(tmp_path):
    """The silence that must stay silent, which is why the note above is scoped to one code.

    Nine candidates per run and one procedure in ten declaring a probe: a note on every refusal
    is a note its reader learns to skip, and then the one that matters is skipped with it. So
    only the reportable code writes, and `probe_note` here must come back untouched.
    """
    pack = probe_pack(tmp_path, use_case="refund_fraud", auto_probe=False)
    finding = _finding(target_use_case="refund_fraud")

    async def _never(*a, **kw):  # pragma: no cover — asserted never awaited
        raise AssertionError("a pair that nominated no probe was probed")

    spent = await run_link_probes(
        _never,
        [finding],
        pack=pack,
        analysis=_analysis(),
        logs={},
        config={"max_probes_per_run": 4, "probe_timeout_seconds": 5},
    )
    assert spent == 0
    assert finding.probe_note == "", "a routine refusal wrote prose onto the finding"


# --- 4. the run-level bounds ------------------------------------------------------------


async def _spend(pack, findings, probe, **cfg):
    config = {"max_probes_per_run": 4, "probe_timeout_seconds": 5}
    config.update(cfg)
    return await run_link_probes(
        probe, findings, pack=pack, analysis=_analysis(), logs={}, config=config
    )


@pytest.mark.asyncio
async def test_the_budget_bounds_the_COUNT_and_the_rest_keep_their_free_settlement(
    tmp_path,
):
    """Three eligible candidates and a budget of one: one query, and no half-finding.

    The candidates that were not probed must be untouched rather than marked as anything —
    "nothing was spent here" is the free-rung settlement they already had, and a note claiming a
    refusal would send the reader to authorise a spend that was never declined on its merits.
    """
    pack = probe_pack(tmp_path)
    asked = []

    async def _probe(source, analysis, **kw):
        asked.append(source)
        return [roster_row()]

    findings = [_finding(), _finding(), _finding()]
    spent = await _spend(pack, findings, _probe, max_probes_per_run=1)

    assert spent == 1
    assert asked == [PROBE_SOURCE]
    assert findings[0].probe_spent is True
    assert [f.probe_spent for f in findings[1:]] == [False, False]
    assert [f.probe_note for f in findings[1:]] == ["", ""]


@pytest.mark.asyncio
async def test_a_zero_budget_a_missing_callable_and_a_packless_run_all_spend_nothing(
    tmp_path,
):
    """The three ways the rung is simply off, asserted together because they must agree.

    `max_probes_per_run: 0` is the one spelling that disarms the rung a deployment ships armed;
    the other two are a build with no probe callable and a run with no pack. All three answer 0
    rather than raising, and none of them writes a note: an inert rung must be indistinguishable
    from a build without one.
    """
    pack = probe_pack(tmp_path)

    async def _probe(source, analysis, **kw):  # pragma: no cover - must never run
        raise AssertionError("a probe was spent with the rung disarmed")

    for label, args, cfg in (
        ("zero budget", (pack, [_finding()], _probe), {"max_probes_per_run": 0}),
        ("no callable", (pack, [_finding()], None), {}),
        ("no pack", (None, [_finding()], _probe), {}),
    ):
        finding = args[1][0]
        assert await _spend(*args, **cfg) == 0, label
        assert finding.probe_spent is False, label
        assert finding.probe_note == "", label


@pytest.mark.asyncio
async def test_a_probe_that_TIMED_OUT_cost_nothing_and_settled_nothing(tmp_path):
    """A timeout is a non-answer, and the note says which — it is not a negative finding.

    The candidate stays `not_probed` with `probe_spent` False, because the point of the four
    states is what was ADJUDICATED. A timeout that settled the candidate as "does not apply"
    would turn an advisory budget being too small into evidence about the sibling procedure.
    """
    pack = probe_pack(tmp_path)

    async def _slow(source, analysis, **kw):
        await asyncio.sleep(5)
        return [roster_row()]

    finding = _finding()
    spent = await _spend(
        pack, [finding], _slow, probe_timeout_seconds=1, max_probes_per_run=1
    )

    assert spent == 0
    assert finding.probe_spent is False
    assert finding.state == "not_probed"
    assert "did not answer" in finding.probe_note
    assert finding.probe_source == PROBE_SOURCE


@pytest.mark.asyncio
async def test_a_raising_probe_records_its_TYPE_and_never_its_message(tmp_path):
    """The note names the exception class and nothing from the exception.

    A backend error commonly quotes the statement that failed, and a probe's statement carries
    this incident's identifiers — the note rides into the job document, the report and the UI. So
    the message is dropped by construction rather than sanitised, which is the only version of
    this that cannot be defeated by an unusual backend.
    """
    pack = probe_pack(tmp_path)
    secret = "SELECT * WHERE tracking_code = 'RT48192043'"

    async def _boom(source, analysis, **kw):
        raise RuntimeError(secret)

    finding = _finding()
    assert await _spend(pack, [finding], _boom, max_probes_per_run=1) == 0
    assert "RuntimeError" in finding.probe_note
    assert secret not in finding.probe_note
    assert "RT48192043" not in finding.probe_note
    assert finding.probe_spent is False


@pytest.mark.asyncio
async def test_an_EMPTY_answer_is_spent_and_a_capped_one_says_it_was_bounded(tmp_path):
    """Two things `state` alone cannot say: what the scan COST, and whether it read everything.

    An empty answer is a real answer, so it is spent — a candidate left unsettled with
    `probe_spent` True is the one an operator must not be asked to authorise a second time. And a
    probe returning exactly its cap is truncated: the note says so, and the cap rides into the
    settlement's own `row_caps` so the sibling's applicability test reads a bounded result as
    bounded rather than as exhaustive.
    """
    pack = probe_pack(tmp_path)

    async def _empty(source, analysis, **kw):
        return []

    async def _full(source, analysis, **kw):
        return [roster_row(badge=f"{i:04d}") for i in range(9)]

    empty = _finding()
    assert await _spend(pack, [empty], _empty, max_probes_per_run=1) == 1
    assert empty.probe_spent is True
    assert "0 row(s)" in empty.probe_note
    assert "capped" not in empty.probe_note

    capped = _finding()
    assert (
        await _spend(pack, [capped], _full, max_probes_per_run=1, probe_row_cap=3) == 1
    )
    assert capped.probe_spent is True
    assert "capped at 3" in capped.probe_note
