"""Rung 4: the four bounds, the nine refusals, and the pin that decides the child's procedure.

`test_link_probe.py` owns the probe rung; this file owns the run. Three claims:

* A refusal is a row, not silence: every code is reached by its own test.
* The total-budget bound is independent of depth, cycle and per-candidate checks; a
  shallow acyclic wide fan-out satisfies all three and is still unbounded.
* The pin flows through `select_correlation_spec` (the engine's own field), cannot be
  set by the LLM, and survives an export/restore round-trip.

The pack is ``knowledge/mock_domain/``.
"""

import pytest

from src.knowledge.pack import load_knowledge_pack
from src.link_children import (
    DEFAULT_MAX_CHILD_DEPTH,
    DEFAULT_MAX_CHILDREN_PER_RUN,
    MAX_CHILD_DEPTH_CEILING,
    MAX_CHILDREN_PER_RUN_CEILING,
    MAX_CONCURRENT_CHILDREN_CEILING,
    MAX_TOTAL_CHILDREN_DEFAULT,
    REFUSAL_NOTES,
    child_budget,
    child_chain,
    child_depth,
    child_incident,
    parent_job_id,
    plan_child_spawns,
)
from src.links import compose_referral
from src.models.pydantic_models import LinkFinding
from tests.mock_domain_links import MOCK_DOMAIN_DIR

# The two fixture procedures, both over the same subject entity, which is what makes one of them a
# usable target for the other without any domain knowledge entering this file.
_PARENT = "refund_fraud"
_TARGET = "courier_collusion"


def _finding(**over):
    """A candidate that WOULD spawn: gate-PASSed, confirmed, pivoted, at an escalating mode.

    Every test below turns exactly one of those off, so the fixture is the positive control for
    all of them — and `test_the_positive_control_spawns` is what stops the whole file from being a
    set of assertions about a candidate that could never have spawned for an unrelated reason.

    `gate_outcome: "pass"` is the licence rung 4 spends a whole run against: the target
    procedure's own scope gate, re-evaluated over THIS run's retrieved rows. It is carried
    separately from `state` because `probed_positive` is reachable on a fired signal alone with
    that gate still unresolved — see `test_a_CONFIRMED_candidate_with_an_unresolved_gate_...`.
    """
    fields = {
        "target_use_case": _TARGET,
        "state": "probed_positive",
        "gate_outcome": "pass",
        "mode": "auto",
        "direction": "consequent",
        "pivot_entity": "shipment",
        "pivot_values": ["SHP-1"],
        "rung": 3,
    }
    fields.update(over)
    return LinkFinding(**fields)


def _config(**over):
    """The `correlation` config slice, with the rung switched ON.

    Note the shape: `plan_child_spawns` is handed the `correlation` block and reads `links` out of
    it, so a test passing the inner block directly would pass while the runner's own call — which
    passes the outer one — could still be reading nothing.
    """
    links = {"max_children_per_run": 2}
    links.update(over)
    return {"links": links}


def _pack():
    return load_knowledge_pack(MOCK_DOMAIN_DIR)


def _codes(refusals):
    return [r["code"] for r in refusals]


# --- the budget ------------------------------------------------------------------------


def test_the_rung_ships_armed_and_narrow():
    """An empty config resolves to the shipped budget, and every bound holds it to one lineage.

    One child per run, one hop, at most one at a time: the rung acts without being configured, and
    what keeps it cheap is the four bounds rather than a zero. A deployment that raises only the
    count must still get a working depth and concurrency bound rather than zeros.
    """
    budget = child_budget({})
    assert budget["max_children"] == DEFAULT_MAX_CHILDREN_PER_RUN == 1
    assert budget["max_depth"] == DEFAULT_MAX_CHILD_DEPTH
    assert budget["max_concurrent"] >= 1
    assert budget["max_total"] == MAX_TOTAL_CHILDREN_DEFAULT


def test_a_zero_count_is_the_one_spelling_that_disarms_the_rung():
    """The opposite direction of the default, and the only way to switch rung 4 off.

    Asserted beside the default because the two are one decision: an operator who does not want
    child runs has to say so, and a deployment that says nothing gets the shipped one.
    """
    budget = child_budget({"links": {"max_children_per_run": 0}})
    assert budget["max_children"] == 0
    assert budget["max_depth"] == DEFAULT_MAX_CHILD_DEPTH


def test_every_bound_is_clamped_to_the_engines_own_ceiling():
    """A config cannot raise any of the four past what the engine holds.

    The backstop is the one that matters most here: a number an operator can raise without bound
    is not a backstop, so its ceiling IS its default.
    """
    budget = child_budget(
        {
            "links": {
                "max_children_per_run": 99,
                "max_child_depth": 99,
                "max_concurrent_children": 99,
                "max_total_children": 999,
            }
        }
    )
    assert budget["max_children"] == MAX_CHILDREN_PER_RUN_CEILING
    assert budget["max_depth"] == MAX_CHILD_DEPTH_CEILING
    assert budget["max_concurrent"] == MAX_CONCURRENT_CHILDREN_CEILING
    assert budget["max_total"] == MAX_TOTAL_CHILDREN_DEFAULT


def test_an_unusable_number_reads_as_the_default_and_never_raises():
    """A config is where a string lands where a number was meant.

    Failing here would take the run down over the advisory lane, and reading an unusable value as
    the shipped default is the same answer as reading no value at all — which is what an operator
    who typed a word into a number field has effectively declared. Switching a rung off is the one
    thing that has to be spelled correctly, since `0` is a number and never a typo.
    """
    budget = child_budget(
        {"links": {"max_children_per_run": "two", "max_child_depth": None}}
    )
    assert budget["max_children"] == DEFAULT_MAX_CHILDREN_PER_RUN
    assert budget["max_depth"] == DEFAULT_MAX_CHILD_DEPTH


def test_the_concurrency_bound_is_a_claim_about_the_shared_llm_semaphore():
    """A deployment throttled to one LLM call runs no child beside the parent.

    This is the bound that is not about counting: every job in this process contends on one
    `Semaphore(max_concurrency)`, so the child allowance is `max_concurrency - 1` and the parent
    keeps a slot of its own. Asserted in both directions, because a bound that only ever floors to
    1 would be indistinguishable from a constant.
    """
    asked = {"links": {"max_concurrent_children": 2}}
    assert child_budget(asked, llm_concurrency=1)["max_concurrent"] == 1
    assert child_budget(asked, llm_concurrency=4)["max_concurrent"] == 2
    # And an unknown cap leaves the engine's own ceiling to apply alone rather than inventing one.
    assert child_budget(asked, llm_concurrency=None)["max_concurrent"] == 2


# --- the chain, and its durability -----------------------------------------------------


def test_the_chain_rides_the_incident_dict_and_is_read_defensively():
    """A malformed hop is dropped and the rest still counts — a SHORTER chain, never a longer one.

    The value comes back from a job document that may have been written by an older build or
    hand-edited, and failing toward a shorter chain means the cap is reached sooner rather than
    later, which is the direction a bound must fail in.
    """
    incident = {
        "link_chain": [
            {"use_case": "a", "pivot": "x"},
            "not a hop",
            {"pivot": "y"},
            {"use_case": "b", "pivot": "z"},
        ],
        "parent_job_id": "job-1",
    }
    assert child_chain(incident) == [
        {"use_case": "a", "pivot": "x"},
        {"use_case": "b", "pivot": "z"},
    ]
    # Two entries, ONE hop: the first entry is the reported run itself, which nobody referred. The
    # distinction is the bound — counting entries made `max_child_depth` 1 and 2 identical.
    assert child_depth(incident) == 1
    assert parent_job_id(incident) == "job-1"
    # An incident a human reported is depth 0, which is what makes the default cap of 1 mean
    # "one hop from a real report".
    assert (
        child_depth({}) == 0 and child_chain(None) == [] and parent_job_id(None) == ""
    )


# --- the positive control, then one test per refusal -----------------------------------


def test_the_positive_control_spawns():
    """The fixture candidate spawns, and the spawn carries everything a child needs.

    Every other test below turns one thing off, so this is what makes each of them a statement
    about that thing.
    """
    spawns, refusals = plan_child_spawns(
        [_finding()],
        incident={},
        config=_config(),
        pack=_pack(),
        parent_use_case=_PARENT,
    )
    assert refusals == []
    assert len(spawns) == 1
    spawn = spawns[0]
    assert spawn["target_use_case"] == _TARGET
    assert spawn["pivot_value"] == "SHP-1"
    assert spawn["depth"] == 1
    # The chain the child will carry ends with its own hop and begins with the parent's, which is
    # what the cycle guard reads on the next lap — and it is exactly two entries for one hop, so
    # the depth the planner recorded and the depth the child reads back agree.
    assert spawn["chain"] == [
        {"use_case": _PARENT, "pivot": ""},
        {"use_case": _TARGET, "pivot": "SHP-1"},
    ]
    assert child_depth({"link_chain": spawn["chain"]}) == spawn["depth"]


@pytest.mark.parametrize(
    "over,code",
    [
        ({"mode": "planned"}, "not_escalating"),
        ({"state": "not_probed"}, "not_confirmed"),
        ({"state": "probed_negative"}, "not_confirmed"),
        ({"gate_outcome": "unknown"}, "gate_not_pass"),
        ({"gate_outcome": "fail"}, "gate_not_pass"),
        ({"gate_outcome": ""}, "gate_not_pass"),
        ({"pivot_values": []}, "no_pivot"),
        ({"target_use_case": "no_such_procedure"}, "unpinnable"),
    ],
)
def test_each_per_candidate_refusal_is_reached_and_named(over, code):
    """One test per way a single candidate is declined, each with its own sentence.

    `unpinnable` is the one worth reading twice: a target with no `correlation:` block cannot be
    pinned through the one resolution seam, so a child of it would resolve its own procedure by
    scoring a description — the exact risk the pin exists to remove — and launching it would be
    worse than refusing.

    `gate_not_pass` is asserted at all three of its non-PASS values because they are three
    different facts about the target procedure — it does not apply, it might, it was never asked —
    and a check spelled `!= "fail"` would let the last two spend a whole run.
    """
    candidate = _finding(**over)
    spawns, refusals = plan_child_spawns(
        [candidate],
        incident={},
        config=_config(),
        pack=_pack(),
        parent_use_case=_PARENT,
    )
    assert spawns == []
    assert _codes(refusals) == [code]
    assert refusals[0]["note"] == REFUSAL_NOTES[code]
    # And each row says WHICH candidate it is about and at what scope, because the caller is what
    # persists: a refusal this function can only return is a refusal an operator cannot read.
    assert refusals[0]["scope"] == "candidate"
    assert refusals[0]["finding"] is candidate


def test_a_CONFIRMED_candidate_with_an_unresolved_gate_is_a_REFERRAL_and_not_a_LAUNCH():
    """`probed_positive` is NOT the escalation licence, and the two are not redundant.

    `_settle` reaches `probed_positive` two ways: the target's own scope gate PASSed on this run's
    rows, or a declared entry signal fired while that gate stayed `unknown`. The second is a real
    finding and belongs in the report — but a fired signal is a reason for a human to look, not a
    reason to spend a whole run, so rung 4 reads rung 1 in its own right. Leaning on `state` alone
    would auto-launch on the signal, which is exactly what the design forbids.

    And the candidate is not lost: it is refused with a code and a sentence, which is how this lane
    surfaces everything it declines to spend on.
    """
    signal_only = _finding(gate_outcome="unknown")
    spawns, refusals = plan_child_spawns(
        [signal_only],
        incident={},
        config=_config(),
        pack=_pack(),
        parent_use_case=_PARENT,
    )
    assert (
        spawns == []
    ), "a fired signal under an unresolved gate auto-launched a child run"
    assert _codes(refusals) == ["gate_not_pass"]
    assert (
        signal_only.state == "probed_positive"
    ), "the refusal rewrote the finding — the candidate must still be reported as confirmed"
    # The same candidate with rung 1 resolved is the one that spawns, so the delta is the gate and
    # nothing else about the finding.
    licensed, _ = plan_child_spawns(
        [_finding(gate_outcome="pass")],
        incident={},
        config=_config(),
        pack=_pack(),
        parent_use_case=_PARENT,
    )
    assert len(licensed) == 1


def test_a_disabled_rung_refuses_once_and_not_once_per_candidate():
    """A deployment's `0`, and it says so — a run-level fact reported N times buries the per-candidate ones."""
    findings = [_finding(), _finding(pivot_values=["SHP-2"])]
    spawns, refusals = plan_child_spawns(
        findings,
        incident={},
        config={"links": {"max_children_per_run": 0}},
        pack=_pack(),
        parent_use_case=_PARENT,
    )
    assert spawns == []
    assert _codes(refusals) == ["children_disabled"]
    # Reported ONCE means it rides on the first candidate merely to have somewhere to sit, so the row says so: it is
    # scoped to the run, and a reader stamping it onto that candidate would tell one arbitrary row
    # that a fact about every row is about it.
    assert refusals[0]["scope"] == "run"
    assert refusals[0]["finding"] is findings[0]


def test_a_refusal_says_which_candidate_and_at_what_SCOPE_without_writing_on_either():
    """A bound that held has to reach the candidate row; only this function knows which row.

    Four codes are underivable from the persisted candidate: `unpinnable`, `depth_cap`,
    `cycle` and `run_budget`. `total_budget` is refused at both scopes: run-wide before
    any candidate is considered, and per candidate once the backstop is reached part-way
    through a list. The caller must not stamp a run-level fact onto the row it rides on.

    This function is asserted pure; a declined candidate still carries its assessment,
    including its `child_note` (the caller's write, not a planning output).
    """
    over_budget = [_finding(pivot_values=[f"SHP-{i}"]) for i in range(4)]
    spawns, refusals = plan_child_spawns(
        over_budget,
        incident={},
        config=_config(max_children_per_run=2),
        pack=_pack(),
        parent_use_case=_PARENT,
    )
    assert len(spawns) == 2 and _codes(refusals) == ["run_budget"]
    # The candidate it is about is the third one — the one the budget stopped — and not the first
    # in the list, which spawned. A row that named the wrong candidate would be worse than none.
    assert refusals[0]["finding"] is over_budget[2]
    assert refusals[0]["scope"] == "candidate"

    # The same code at the OTHER scope, from the same list and the same config: reached run-wide,
    # before any candidate is looked at, so the row it rides on is not a row it is ABOUT.
    _, run_wide = plan_child_spawns(
        over_budget,
        incident={},
        config=_config(max_total_children=2),
        pack=_pack(),
        parent_use_case=_PARENT,
        spawned_total=2,
    )
    assert _codes(run_wide) == ["total_budget"]
    assert run_wide[0]["scope"] == "run"

    # And nothing was written on any of them. The note a refusal earns is stamped by the runner,
    # which is what persists; writing it here would make this function impure and would put the
    # sentence on an object the job document is not built from.
    for finding in over_budget:
        assert finding.child_note == ""
        assert finding.child_job_id == ""
        assert finding.state == "probed_positive"


def test_the_depth_cap_stops_a_referral_of_a_referral():
    """A run that is ITSELF a referral spawns nothing at the default depth.

    Read off the chain the incident carries, which is why the export/restore test below is part of
    this bound and not a separate nicety.
    """
    incident = {
        "link_chain": [
            {"use_case": "the_reported_procedure", "pivot": ""},
            {"use_case": "somewhere", "pivot": "SHP-9"},
        ]
    }
    spawns, refusals = plan_child_spawns(
        [_finding()],
        incident=incident,
        config=_config(),
        pack=_pack(),
        parent_use_case=_PARENT,
    )
    assert spawns == [] and _codes(refusals) == ["depth_cap"]
    # And raising the cap makes the same candidate spawn, so the refusal is attributable to the
    # bound and not to the chain having broken something else.
    spawns, refusals = plan_child_spawns(
        [_finding()],
        incident=incident,
        config=_config(max_child_depth=2),
        pack=_pack(),
        parent_use_case=_PARENT,
    )
    assert len(spawns) == 1 and spawns[0]["depth"] == 2
    # And the number means hops: raising the cap by one buys exactly one more hop, which is the
    # whole reason the chain's first entry is not counted.
    assert child_depth({"link_chain": spawns[0]["chain"]}) == 2


def test_the_cycle_guard_reads_the_procedure_AND_the_identity():
    """A(x) -> B(y) -> A(x) terminates; A(x) -> B(y) -> A(z) does not.

    Both halves in one test on purpose: a guard keyed on the procedure alone would pass the first
    assertion and silently refuse every second look at a procedure under a NEW identity, which is
    a different question and the one a chain exists to follow.
    """
    incident = {"link_chain": [{"use_case": _TARGET, "pivot": "SHP-1"}]}
    cfg = _config(max_child_depth=3)
    spawns, refusals = plan_child_spawns(
        [_finding()],
        incident=incident,
        config=cfg,
        pack=_pack(),
        parent_use_case=_PARENT,
    )
    assert spawns == [] and _codes(refusals) == ["cycle"]

    spawns, refusals = plan_child_spawns(
        [_finding(pivot_values=["SHP-2"])],
        incident=incident,
        config=cfg,
        pack=_pack(),
        parent_use_case=_PARENT,
    )
    assert len(spawns) == 1 and refusals == []


def test_the_cycle_guard_includes_the_PARENTS_own_hop():
    """A root incident's chain is empty, so the parent is not in it — and has to be added.

    Without this, a candidate pointing back at the procedure that just ran is only caught on the
    SECOND lap, which is one full run too late. Every pivot the parent adjudicated counts, not
    just the first: a run whose scope sweep widened to three identities answered for all three.
    """
    spawns, refusals = plan_child_spawns(
        [_finding(target_use_case=_PARENT, pivot_values=["SHP-7"])],
        incident={},
        config=_config(),
        pack=_pack(),
        parent_use_case=_PARENT,
        parent_pivots=["SHP-1", "SHP-7"],
    )
    assert spawns == [] and _codes(refusals) == ["cycle"]


def test_the_per_run_budget_spawns_up_to_it_and_refuses_the_rest():
    """The candidates after the ones that spawned keep their referrals, and are told so."""
    findings = [_finding(pivot_values=[f"SHP-{i}"]) for i in range(4)]
    spawns, refusals = plan_child_spawns(
        findings,
        incident={},
        config=_config(max_children_per_run=2),
        pack=_pack(),
        parent_use_case=_PARENT,
    )
    assert len(spawns) == 2
    assert _codes(refusals) == ["run_budget"]


def test_the_backstop_is_INDEPENDENT_of_the_depth_cap_and_the_cycle_guard():
    """The reason it exists: a wide, shallow, acyclic fan-out is within every per-lineage bound.

    So it is asserted where nothing else is close to firing — depth 0, no repeated identity, and a
    per-run budget deliberately HIGHER than the number of candidates. A test that reached the
    backstop while also over the per-run budget would pass whether or not the two checks are
    separate.
    """
    findings = [_finding(pivot_values=[f"SHP-{i}"]) for i in range(4)]
    cfg = _config(
        max_children_per_run=MAX_CHILDREN_PER_RUN_CEILING, max_total_children=2
    )
    spawns, refusals = plan_child_spawns(
        findings, incident={}, config=cfg, pack=_pack(), parent_use_case=_PARENT
    )
    assert len(spawns) == 2
    assert _codes(refusals) == ["total_budget"]
    assert refusals[0]["note"] == REFUSAL_NOTES["total_budget"]

    # And it counts what this PROCESS already spent, not what this run did: a parent arriving with
    # the backstop already reached spawns nothing at all, whatever its own budget says.
    spawns, refusals = plan_child_spawns(
        findings,
        incident={},
        config=cfg,
        pack=_pack(),
        parent_use_case=_PARENT,
        spawned_total=2,
    )
    assert spawns == [] and _codes(refusals) == ["total_budget"]


def test_planning_is_pure_and_deterministic():
    """Same inputs, same two lists — no clock, no IO, nothing accumulated between calls."""
    args = dict(incident={}, config=_config(), pack=_pack(), parent_use_case=_PARENT)
    findings = [_finding(), _finding(mode="planned", pivot_values=["SHP-2"])]

    def strip(out):
        # The finding itself is carried by reference on BOTH lists — a spawn's and a refusal's —
        # so it is dropped before comparing: two identical objects compare equal here for the
        # wrong reason, and what is being asserted is that the DECISIONS match.
        return tuple(
            [{k: v for k, v in row.items() if k != "finding"} for row in rows]
            for rows in out
        )

    assert strip(plan_child_spawns(findings, **args)) == strip(
        plan_child_spawns(findings, **args)
    )


# --- the child incident, and the pin -----------------------------------------------------


def test_the_child_incident_carries_the_three_durable_fields_and_NOT_the_parents_prose():
    """The pin, the chain and the parent id ride the incident dict; the parent's description does not.

    That absence is the design and not an omission: the parent's description is the text that made
    the PARENT's procedure win, so a child described by it re-adjudicates the incident already
    adjudicated, under the same headings, and reads as a confirmation.
    """
    spawns, _ = plan_child_spawns(
        [_finding()],
        incident={},
        config=_config(),
        pack=_pack(),
        parent_use_case=_PARENT,
    )
    composed = compose_referral(
        spawns[0]["finding"], parent_job_id="job-1", parent_incident_id="INC-1"
    )
    parent_incident = {
        "description": "A DISTINCTIVE PARENT SENTENCE about the reported thing.",
        "timestamp": "2026-08-19T10:00:00Z",
    }
    incident = child_incident(spawns[0], parent_incident, composed)
    assert incident["link_pin"] == _TARGET
    assert incident["link_chain"][-1] == {"use_case": _TARGET, "pivot": "SHP-1"}
    assert (
        incident["parent_job_id"] == ""
    )  # no parent job was handed to the planner here
    assert "DISTINCTIVE PARENT SENTENCE" not in incident["description"]
    assert _TARGET in incident["description"] and "SHP-1" in incident["description"]
    # The event time is INHERITED, so the child asks about the reported event's window rather than
    # about the moment the referral happened to be composed.
    assert incident["timestamp"] == parent_incident["timestamp"]


def test_a_child_incident_composes_with_no_referral_text_at_all():
    """A pinned run with a thin description is correct, so composition failing is not fatal.

    The pin decides the procedure and the pivot decides the scope, so neither depends on prose —
    which is why the runner treats a failed compose as a log line and launches anyway.
    """
    spawns, _ = plan_child_spawns(
        [_finding()],
        incident={},
        config=_config(),
        pack=_pack(),
        parent_use_case=_PARENT,
        parent_job="job-7",
    )
    incident = child_incident(spawns[0], {}, None)
    assert incident["link_pin"] == _TARGET
    assert incident["parent_job_id"] == "job-7"
    assert "job-7" in incident["description"] and _TARGET in incident["description"]


def test_the_chain_survives_an_export_and_a_restore():
    """The whole reason the chain rides the incident: `export_job` carries that dict verbatim.

    A chain lost by a restart resets the depth cap to 0 on exactly the runs it exists to bound, so
    this is asserted through the real job document rather than by inspecting the dict.
    """
    from src.pipeline_runner import JobManager, build_stage_descriptors
    from src.notifications import EventEmitter

    manager = JobManager(build_stage_descriptors(), EventEmitter())
    spawns, _ = plan_child_spawns(
        [_finding()],
        incident={},
        config=_config(),
        pack=_pack(),
        parent_use_case=_PARENT,
        parent_job="job-parent",
    )
    child = manager.create_job(child_incident(spawns[0], {}, None))
    doc = manager.export_job(child.job_id)

    restored = JobManager(build_stage_descriptors(), EventEmitter())
    back = restored.import_job(doc)
    assert child_depth(back.incident) == 1
    assert parent_job_id(back.incident) == "job-parent"
    assert back.incident["link_pin"] == _TARGET
    # And the restored chain is what the guard then reads, so a re-plan off it refuses the repeat.
    _, refusals = plan_child_spawns(
        [_finding()],
        incident=back.incident,
        config=_config(),
        pack=_pack(),
        parent_use_case=_PARENT,
    )
    assert _codes(refusals) == ["depth_cap"]


def test_the_pin_decides_the_procedure_through_the_ONE_resolution_seam():
    """A pinned incident adjudicates the pinned procedure, whatever its description scores as.

    Asserted against a description written to make the OTHER procedure win, and both directions
    are in one test: without the pin the same text picks the rival, with it the pin holds. A
    one-directional version would pass on a fixture whose text scored the pinned spec anyway,
    which is the reading a pin exists to make unnecessary.

    The seam is `select_correlation_spec`, and it is the seam because its answer is ALSO what
    tells the planner which sources are hard dependencies: a pin honoured at the verdict alone
    would adjudicate one procedure over another procedure's retrieval plan.
    """
    from types import SimpleNamespace

    from src.correlation import select_correlation_spec

    pack = _pack()
    rival = select_correlation_spec(
        pack,
        SimpleNamespace(incident_summary=_PARENT.replace("_", " "), pinned_use_case=""),
    )
    assert rival and rival["use_case"] == _PARENT

    pinned = select_correlation_spec(
        pack,
        SimpleNamespace(
            incident_summary=_PARENT.replace("_", " "), pinned_use_case=_TARGET
        ),
    )
    assert pinned and pinned["use_case"] == _TARGET

    # A pin naming nothing the pack declares falls back to the score rather than resolving to
    # nothing: a job document can outlive the pack edit that removed a use case, and no verdict at
    # all is a worse answer than the honest scored one.
    stale = select_correlation_spec(
        pack,
        SimpleNamespace(
            incident_summary=_PARENT.replace("_", " "), pinned_use_case="deleted_proc"
        ),
    )
    assert stale and stale["use_case"] == _PARENT


def test_the_pin_comes_from_the_INCIDENT_and_a_model_supplied_one_is_discarded():
    """`pinned_use_case` is the one field on the analysis no prompt may write.

    It decides which procedure adjudicates, so a model that returns one has effectively chosen the
    ruleset — the outcome the whole selection seam exists to take away from prose. The stamp is
    therefore unconditional: "the model invented a pin" and "there is no pin" have to end up as
    the same state, and a guarded assignment would let the first one through, which is the
    direction that produces a confident wrong verdict rather than none.
    """
    from unittest.mock import MagicMock

    from src.incident_understanding import IncidentUnderstandingModule
    from src.models.pydantic_models import IncidentAnalysis

    def analysis(**over):
        fields = dict(
            incident_summary="a reported thing",
            severity_reasoning="",
            impact_assessment="",
            key_investigation_areas=[],
            log_sources_to_review=[],
            initial_hypotheses=[],
            recommended_actions=[],
            stakeholder_notification=[],
        )
        fields.update(over)
        return IncidentAnalysis(**fields)

    module = IncidentUnderstandingModule(MagicMock())

    invented = analysis(pinned_use_case="a_procedure_nobody_asked_for")
    module._pin_requested_use_case(invented, {"id": "INC-1"})
    assert invented.pinned_use_case == ""

    referred = analysis()
    module._pin_requested_use_case(referred, {"id": "INC-2", "link_pin": _TARGET})
    assert referred.pinned_use_case == _TARGET
