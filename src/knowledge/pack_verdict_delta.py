"""Re-adjudicate every stored run under both packs, and name the condition lines that moved.

WHAT THIS CATCHES that nothing else does. ``pack_validate`` answers "will the engine read
this declaration", ``pack_dry_run`` answers "does this ruleset decide anything over rows that
really came back", and ``pack_selection_delta`` answers "which procedure would adjudicate".
None of the three can say what an edit does to the *findings* of the runs already on record:
a reworded field path, a threshold moved by one, a check imported under the other polarity —
each of those leaves the pack valid, the dry run non-empty and every selection where it was,
and changes what the report concludes about a named person's conduct.

So this is a **difference**, and it needs both sides replayed. The base pack and the candidate
adjudicate the same stored evidence, and the per-subject, per-condition lines are diffed — the
same comparison one makes between two runs of one incident, made here between
two packs over one run. It decides nothing: moving a finding is usually the point of the edit,
and "is this the right finding" is a judgement no arithmetic settles.

Four properties it holds, each because the alternative fails quietly:

* **Base against candidate, never candidate against the RECORDED verdict.** A stored run's
  evidence sidecar is flattened and carries neither ``row_caps`` nor ``keyed_sources``, so a
  replay legitimately reaches a weaker reading than the run did. Diffed against the recording
  that shows up as a regression on every line; diffed against the base pack's replay of the
  same rows it cancels exactly, because both sides suffer it identically.
* **An identical replay surface cannot move a verdict**, so an edit that leaves every ruleset
  spec, every entity binding and every data file alone short-circuits before reading the
  corpus. Most pack edits are that edit, and a check that costs seconds on every save is a
  check that gets turned off.
* **The comparison is proved able to disagree with itself first.** The base pack replays one
  run twice, and unless those two agree the delta is withheld: a difference between two packs
  is only attributable to the edit if evaluating the same pack twice is stable. Without that
  control, a condition reading the clock — or an evaluator that mutated the rows it was handed
  — reports the author's edit as the cause of a change it did not make.
* **No corpus means silence, not a clean result.** A deployment with no stored runs can say
  nothing about findings, and reporting "0 changed" there would read as a guarantee. The
  counts ride on the result either way, or a pack that stopped being checked looks like one
  that passed.

Pure and read-only: no LLM, no network, no writes. Never raises — every failure becomes a
``problems`` entry, because a comparison that aborts on one unreadable job document tells an
author less than one that reports 11 of 12.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Runs replayed at most. Half ``pack_dry_run``'s budget because every run is adjudicated
#: TWICE here, and this shares an operator's request with the dry run itself.
DEFAULT_MAX_RUNS = 6

#: Wall-clock budget for both sides together, checked between runs. Belt and braces with the
#: run count: one run's cost scales with the rows it retrieved, which no count can bound.
DEFAULT_MAX_SECONDS = 20.0

#: Changed lines listed. A cut list says so (``changes_cut``), because a reader who has
#: learned that these lists are complete reads a truncated one as the whole consequence.
MAX_CHANGES = 40

#: Detail text kept per line. The detail is context for a result that moved, not the finding.
DETAIL_CHARS = 160

#: Line kinds, worst first — the order the cut keeps and the render prints. A summary or a
#: subject's disposition is what a reader acts on; a condition's own result is why.
_KIND_ORDER = (
    "summary",
    "verdict",
    "subject_gone",
    "subject_new",
    "degraded",
    "condition",
    "line_gone",
    "line_new",
)

#: Results that are a determination. A flip between two of these is a changed FINDING; a
#: transition through anything else is a check that started or stopped answering, which
#: matters less to a reader and much more often to the author of the edit.
_DECIDED = ("pass", "fail")


def _dry_run():
    """``pack_dry_run``, imported late so this module has no import-time cost of its own."""
    from src.knowledge import pack_dry_run

    return pack_dry_run


# ------------------------------------------------------------------------ the data model


@dataclass(frozen=True)
class LineChange:
    """One verdict line that reads differently under the candidate pack.

    ``line`` is a condition id, or one of ``_verdict`` / ``_summary`` / ``_degraded`` — the
    three lines that are not a condition and are the ones a reader looks at first.
    """

    job_id: str = ""
    ruleset_key: str = ""
    subject: str = ""
    line: str = ""
    kind: str = "condition"
    before: str = ""
    after: str = ""
    detail: str = ""

    @property
    def decided_flip(self) -> bool:
        """True when a determination became a different determination.

        A ``pass`` that became a ``fail`` is a changed finding about a person. A ``pass`` that
        became ``unknown`` is a check that stopped answering — worth reporting, and a
        different thing, so the report must not print them in one undifferentiated list.
        """
        return (
            self.kind == "condition"
            and self.before in _DECIDED
            and self.after in _DECIDED
            and self.before != self.after
        )

    @property
    def rank(self) -> Tuple[int, int, str, str]:
        """Sort key: kind first, then a decided flip ahead of a check that went quiet."""
        try:
            kind = _KIND_ORDER.index(self.kind)
        except ValueError:
            kind = len(_KIND_ORDER)
        return (kind, 0 if self.decided_flip else 1, self.job_id, self.line)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "ruleset_key": self.ruleset_key,
            "subject": self.subject,
            "line": self.line,
            "kind": self.kind,
            "before": self.before,
            "after": self.after,
            "detail": self.detail,
            "decided_flip": self.decided_flip,
        }


@dataclass
class VerdictDelta:
    """What the candidate pack does to the findings of the runs already on record.

    ``compared`` is the field a reader must check first. False means no comparison happened —
    no corpus, no replayable ruleset, an unchanged replay surface, or a failed determinism
    control — and then an empty ``changes`` is a silence rather than a clean bill.
    """

    compared: bool = False
    reason: str = ""
    corpus: int = 0
    replayed: int = 0
    runs_changed: int = 0
    rulesets: Tuple[str, ...] = ()
    surface_changed: Tuple[str, ...] = ()
    changes: Tuple[LineChange, ...] = ()
    changes_cut: int = 0
    limits: Tuple[str, ...] = ()
    problems: Tuple[str, ...] = field(default_factory=tuple)
    seconds: float = 0.0

    @property
    def decided_flips(self) -> Tuple[LineChange, ...]:
        """The changes that moved one determination to another, rather than to ``unknown``."""
        return tuple(c for c in self.changes if c.decided_flip)

    @property
    def verdicts_moved(self) -> Tuple[LineChange, ...]:
        """The changes a report reader sees without opening a condition table."""
        return tuple(c for c in self.changes if c.kind in ("summary", "verdict"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compared": self.compared,
            "reason": self.reason,
            "corpus": self.corpus,
            "replayed": self.replayed,
            "runs_changed": self.runs_changed,
            "rulesets": list(self.rulesets),
            "surface_changed": list(self.surface_changed),
            "changes": [c.to_dict() for c in self.changes],
            "changes_cut": self.changes_cut,
            "decided_flips": len(self.decided_flips),
            "verdicts_moved": len(self.verdicts_moved),
            "limits": list(self.limits),
            "problems": list(self.problems),
            "seconds": round(self.seconds, 2),
            # The rendered text rides along, as it does on ``pack_dry_run.as_dict`` and on
            # the selection delta: the preview shows the server's own sentences rather than
            # re-deriving them in JS, or the operator approving the edit and the model
            # proposing it are reading two descriptions of one measurement.
            "text": render(self),
        }


# ------------------------------------------------------------------- the replay surface


def replay_surface(pack: Any) -> Dict[str, Any]:
    """Everything about a pack that an adjudication of stored rows can read.

    Three things and no more, because the replay's inputs are exactly three: the resolved
    ruleset specs, the pack data files the engine is handed, and the entity-to-column
    bindings the entity map resolves through. A ``description`` reworded, a query hint
    rewritten, a comment added — none of those can move a verdict over rows that are already
    retrieved, and including them here would spend two replays on every catalog edit.

    The bindings are read through ``field_priors_for`` rather than off ``entity_bindings``
    directly: that accessor is where per-form bindings, per-source overrides and the global
    aliases are resolved into the list the replay actually consults, and a second reading of
    that precedence would be a second answer to it.
    """
    specs: Dict[str, Any] = {}
    for key in pack.ruleset_keys() or []:
        try:
            specs[str(key)] = pack.ruleset_spec(str(key))
        except Exception as e:  # noqa: BLE001 — an unresolvable import is a finding
            specs[str(key)] = f"<unresolvable: {type(e).__name__}: {e}>"
    types = list(pack.entity_types() or [])
    bindings = {
        str(src.name): {t: list(pack.field_priors_for(t, str(src.name))) for t in types}
        for src in (pack.sources or [])
    }
    return {
        "specs": specs,
        "data": getattr(pack, "pack_data", None) or {},
        "bindings": bindings,
    }


def surface_changed(base: Any, candidate: Any) -> Tuple[str, ...]:
    """Which parts of the replay surface the edit moved: ``specs`` / ``data`` / ``bindings``.

    Empty means no verdict can move, and that is exact rather than a heuristic — the
    adjudication is a pure function of these three plus the stored rows, and the rows are the
    same on both sides. Which is what licenses the short-circuit in :func:`verdict_delta`.

    Reported per part rather than as one bit because the parts license different readings: a
    ``bindings`` change moves what every ruleset can see, while a ``specs`` change is usually
    confined to the use case that was edited.
    """
    before, after = replay_surface(base), replay_surface(candidate)
    out: List[str] = []
    for part in ("specs", "data", "bindings"):
        if json.dumps(before[part], sort_keys=True, default=str) != json.dumps(
            after[part], sort_keys=True, default=str
        ):
            out.append(part)
    return tuple(out)


# -------------------------------------------------------------------- the verdict lines


def lines(verdict: Any) -> Dict[str, Any]:
    """``{"subjects": {subject: {line: (result, detail)}}, "_summary", "_degraded"}``.

    The same shape a diff of two job documents reads, built here off the verdict OBJECT — so
    one comparison serves a replay and a stored run, and a line a reader can diff by hand is
    a line this can diff.

    A repeated subject key is suffixed rather than overwritten: two subjects of one type and
    value is not a shape the engine is expected to produce, and silently dropping the second
    would report its every condition as missing from both sides.
    """
    subjects: Dict[str, Dict[str, Tuple[str, str]]] = {}
    for subject in getattr(verdict, "subjects", None) or []:
        key = (
            f"{getattr(subject, 'subject_type', '') or ''} "
            f"{getattr(subject, 'subject_value', '') or ''}"
        ).strip()
        if key in subjects:
            key = f"{key} #{sum(1 for k in subjects if k.startswith(key)) + 1}"
        checks: Dict[str, Tuple[str, str]] = {
            "_verdict": (
                str(getattr(subject, "verdict", "") or ""),
                str(getattr(subject, "verdict_class", "") or ""),
            )
        }
        for check in getattr(subject, "checks", None) or []:
            cid = str(getattr(check, "id", "") or "?")
            checks[cid] = (
                str(getattr(check, "result", "") or ""),
                str(getattr(check, "detail", "") or "")[:DETAIL_CHARS],
            )
        subjects[key] = checks
    return {
        "subjects": subjects,
        "_summary": str(getattr(verdict, "summary", "") or ""),
        # Spelled rather than `str(bool)`: `degraded` is a bool, and `False or ""` renders as
        # an ABSENT value, so a run that stopped being degraded would print `(absent) -> ...`
        # — the one wording this module reserves for a line that is not there at all.
        "_degraded": "degraded" if getattr(verdict, "degraded", False) else "complete",
    }


def _fmt(value: Tuple[str, str], *, with_class: bool) -> str:
    if with_class:
        return f"{value[0] or '(none)'} [{value[1] or '-'}]"
    return value[0] or "(none)"


def diff(
    before: Dict[str, Any], after: Dict[str, Any], *, job_id: str = "", key: str = ""
) -> List[LineChange]:
    """Every line that reads differently, in no particular order."""
    out: List[LineChange] = []

    def add(kind: str, line: str, b: str, a: str, subject: str = "", detail: str = ""):
        out.append(
            LineChange(
                job_id=job_id,
                ruleset_key=key,
                subject=subject,
                line=line,
                kind=kind,
                before=b,
                after=a,
                detail=detail,
            )
        )

    if before["_summary"] != after["_summary"]:
        add("summary", "_summary", before["_summary"], after["_summary"])
    if before["_degraded"] != after["_degraded"]:
        add("degraded", "_degraded", before["_degraded"], after["_degraded"])

    b_subjects, a_subjects = before["subjects"], after["subjects"]
    for subject, b_lines in b_subjects.items():
        a_lines = a_subjects.get(subject)
        if a_lines is None:
            add(
                "subject_gone",
                "_verdict",
                _fmt(b_lines["_verdict"], with_class=True),
                "",
                subject=subject,
            )
            continue
        for line, b_value in b_lines.items():
            a_value = a_lines.get(line)
            if a_value is None:
                add("line_gone", line, b_value[0], "", subject=subject)
                continue
            if a_value == b_value:
                continue
            if line == "_verdict":
                if a_value[0] != b_value[0] or a_value[1] != b_value[1]:
                    add(
                        "verdict",
                        line,
                        _fmt(b_value, with_class=True),
                        _fmt(a_value, with_class=True),
                        subject=subject,
                    )
                continue
            # A detail that moved under an unchanged result is prose, not a finding: the
            # narration is regenerated per run, so reporting it would bury the flips.
            if a_value[0] != b_value[0]:
                add(
                    "condition",
                    line,
                    b_value[0],
                    a_value[0],
                    subject=subject,
                    detail=a_value[1],
                )
    for subject, a_lines in a_subjects.items():
        b_lines = b_subjects.get(subject)
        if b_lines is None:
            add(
                "subject_new",
                "_verdict",
                "",
                _fmt(a_lines["_verdict"], with_class=True),
                subject=subject,
            )
            continue
        for line, a_value in a_lines.items():
            if line not in b_lines:
                add(
                    "line_new",
                    line,
                    "",
                    a_value[0],
                    subject=subject,
                    detail=a_value[1],
                )
    return out


# ------------------------------------------------------------------------- the comparison


def _limits(cut: int, total: int, over_budget: bool) -> Tuple[str, ...]:
    """What this reading cannot see, each stated with the direction it errs in.

    Deliberately shorter than ``pack_dry_run._limits``: the caveats that dominate a single
    replay — the flattened sidecar, the absent row caps, the absent keyed-source facts — are
    identical on both sides here and therefore cancel. What survives is the one that does not
    cancel, which is that a condition reading ``unknown`` for a replay's own reasons is
    ``unknown`` under both packs, so an edit meant to FIX such a condition shows no change.
    """
    out = [
        "A condition that cannot resolve against stored evidence reads `unknown` under BOTH "
        "packs, so an edit whose whole purpose is to make such a check answer will show no "
        "change here. Absence of a change is not evidence the edit did nothing."
    ]
    if cut:
        out.append(
            f"{cut} of {total} eligible run(s) were not replayed"
            + (" (the time budget was spent)" if over_budget else " (the run budget)")
            + ". Replaying more can only ADD changes."
        )
    return tuple(out)


def compare(
    base_pack: Any,
    candidate_pack: Any,
    runs: Sequence[Any],
    *,
    max_runs: int = DEFAULT_MAX_RUNS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    problems: Optional[Sequence[str]] = None,
) -> VerdictDelta:
    """Adjudicate ``runs`` under both packs and report the lines that moved."""
    started = time.monotonic()
    out = VerdictDelta(corpus=len(runs), problems=tuple(problems or ()))
    try:
        dr = _dry_run()
        keys = sorted(set(base_pack.ruleset_keys() or []) & set(candidate_pack.ruleset_keys() or []))
        added = sorted(set(candidate_pack.ruleset_keys() or []) - set(keys))
        if added:
            # A ruleset the base pack does not have has no before-state, so every line of it
            # would read as `line_new` — noise that would bury the changes to the procedures
            # that DO have one. Named instead, which is the actionable form.
            out.problems = out.problems + (
                "new ruleset(s) with no before-state to compare against, so their findings "
                f"are not diffed here: {', '.join(added)}",
            )
        if not keys:
            out.reason = "the two packs share no ruleset key, so no run has a before-state"
            return out
        if not runs:
            out.reason = (
                "no stored run carries retrieved rows, so this deployment cannot say which "
                "past findings the edit moves"
            )
            return out

        per_key: Dict[str, List[Any]] = {}
        specs: Dict[str, Tuple[Any, Any]] = {}
        for key in keys:
            try:
                base_spec = base_pack.ruleset_spec(key)
                cand_spec = candidate_pack.ruleset_spec(key)
            except Exception as e:  # noqa: BLE001 — an unresolvable import raises by design
                out.problems = out.problems + (
                    f"{key}: its spec could not be resolved ({type(e).__name__}: {e})",
                )
                continue
            if not base_spec or not cand_spec:
                continue
            specs[key] = (base_spec, cand_spec)
            eligible, unpaired = dr._eligible(key, cand_spec, runs)
            if eligible:
                per_key[key] = eligible
            if eligible and unpaired:
                # No stored run was ADJUDICATED by this ruleset, so the runs below are paired
                # on source overlap alone — the weaker reading, and it must never be
                # presented as the paired one. Both sides suffer it identically, so a change
                # it reports is still real; what it cannot claim is that the run in question
                # would ever have been decided by this procedure.
                out.problems = out.problems + (
                    f"{key}: no stored run was adjudicated by it, so its runs are paired on "
                    "shared sources only — a changed line there is real, but the run was "
                    "never decided by this procedure",
                )
        if not per_key:
            out.reason = (
                "no stored run was adjudicated by a shared ruleset and none retrieved its "
                "sources, so there is nothing to re-adjudicate"
            )
            return out

        plan = dr._select(per_key, max(0, int(max_runs)))
        evaluate = dr._correlation().evaluate_verdict
        base_data = getattr(base_pack, "pack_data", None)
        cand_data = getattr(candidate_pack, "pack_data", None)

        def replay(pack: Any, spec: Any, run: Any, data: Any) -> Dict[str, Any]:
            etypes = dr._entity_types(run)
            verdict = evaluate(
                spec,
                run.logs,
                run.analysis,
                entity_map=dr._entity_map(pack, run.logs, etypes),
                pack_data=data,
            )
            return lines(verdict) if verdict is not None else lines(None)

        changes: List[LineChange] = []
        cut = 0
        replayed_keys: List[str] = []
        changed_runs = 0
        over_budget = False
        control_done = False
        for key, run in plan:
            if time.monotonic() - started > max_seconds:
                over_budget = True
                break
            base_spec, cand_spec = specs[key]
            try:
                before = replay(base_pack, base_spec, run, base_data)
                if not control_done:
                    # The comparison proved able to agree with itself, once, before any
                    # difference is attributed to the edit. Cheaper than it looks — one extra
                    # replay of one run — and it is the only thing standing between "the
                    # candidate changed this" and "this evaluation is not reproducible".
                    control_done = True
                    if diff(before, replay(base_pack, base_spec, run, base_data)):
                        out.reason = (
                            "the base pack adjudicated one stored run twice and did not "
                            "reach the same lines, so a difference between the two packs "
                            "could not be attributed to the edit"
                        )
                        out.limits = _limits(0, 0, False)
                        return out
                after = replay(candidate_pack, cand_spec, run, cand_data)
            except Exception as e:  # noqa: BLE001 — one bad run must not lose the others
                logger.warning(
                    "Verdict delta failed on job %s: %s", str(run.job_id)[:8], e
                )
                out.problems = out.problems + (
                    f"job {str(run.job_id)[:8]} raised {type(e).__name__}: {e}",
                )
                continue
            out.replayed += 1
            if key not in replayed_keys:
                replayed_keys.append(key)
            found = sorted(
                diff(before, after, job_id=str(run.job_id), key=key),
                key=lambda c: c.rank,
            )
            if found:
                changed_runs += 1
            for change in found:
                if len(changes) >= MAX_CHANGES:
                    cut += 1
                    continue
                changes.append(change)

        out.rulesets = tuple(replayed_keys)
        out.runs_changed = changed_runs
        out.changes = tuple(sorted(changes, key=lambda c: c.rank))
        out.changes_cut = cut
        total_eligible = len({r.job_id for items in per_key.values() for r in items})
        out.limits = _limits(max(0, total_eligible - out.replayed), total_eligible, over_budget)
        if not out.replayed:
            out.reason = (
                "every eligible run failed to re-adjudicate, so nothing was compared — the "
                "problems below are the finding"
            )
            return out
        out.compared = True
    except Exception as e:  # noqa: BLE001 — a broken comparison is a finding, not a crash
        logger.warning("Verdict delta failed: %s", e, exc_info=True)
        out.compared = False
        out.problems = out.problems + (
            f"the verdict comparison could not be completed ({type(e).__name__}: {e})",
        )
    finally:
        out.seconds = round(time.monotonic() - started, 2)
    return out


def verdict_delta(
    base_pack: Any,
    candidate_pack: Any,
    *,
    store: Any = None,
    jobs_dir: Optional[Path] = None,
    runs: Optional[Sequence[Any]] = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> VerdictDelta:
    """:func:`compare` over the stored corpus — the one call a caller with two packs needs.

    The corpus is read only once the replay surface is known to have moved, so the common
    case — an edit that touches no spec, no binding and no data file — costs two surface
    builds and no IO at all.
    """
    moved: Tuple[str, ...] = ()
    try:
        moved = surface_changed(base_pack, candidate_pack)
        if not moved:
            out = VerdictDelta(
                reason=(
                    "no ruleset spec, entity binding or data file changed, so every stored "
                    "run adjudicates exactly as it did"
                )
            )
            return out
    except Exception as e:  # noqa: BLE001 — an unreadable surface is compared, not trusted
        logger.warning("Verdict delta could not read the replay surface: %s", e)
        moved = ("unknown",)
    if runs is None:
        runs, problems = _dry_run().stored_runs(store, jobs_dir=jobs_dir)
    else:
        problems = []
    delta = compare(
        base_pack,
        candidate_pack,
        runs,
        max_runs=max_runs,
        max_seconds=max_seconds,
        problems=problems,
    )
    delta.surface_changed = moved
    return delta


def verdict_delta_for_dirs(
    base_dir: Any,
    candidate_dir: Any,
    *,
    store: Any = None,
    jobs_dir: Optional[Path] = None,
    runs: Optional[Sequence[Any]] = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> VerdictDelta:
    """:func:`verdict_delta` between two pack DIRECTORIES — the plan-preview entry point.

    Both sides go through ``load_knowledge_pack``, the loader the pipeline uses. A pack that
    will not load is a ``problems`` entry and not a refusal: this is not the check that
    reports a broken pack, and declining to answer would hide the moved findings in a plan
    that has a second, unrelated defect.
    """
    from src.knowledge.pack import load_knowledge_pack

    packs: Dict[str, Any] = {}
    problems: List[str] = []
    for side, path in (("base", base_dir), ("candidate", candidate_dir)):
        try:
            packs[side] = load_knowledge_pack(path)
        except Exception as e:  # noqa: BLE001 — an unloadable pack is somebody else's finding
            packs[side] = None
            problems.append(f"the {side} pack could not be loaded: {e}")
    if problems:
        return VerdictDelta(
            reason="a pack could not be loaded, so no adjudication was replayed",
            problems=tuple(problems),
        )
    return verdict_delta(
        packs["base"],
        packs["candidate"],
        store=store,
        jobs_dir=jobs_dir,
        runs=runs,
        max_runs=max_runs,
        max_seconds=max_seconds,
    )


# --------------------------------------------------------------------------- rendering


def render(delta: VerdictDelta) -> str:
    """The delta as text, for the plan preview and the CLI.

    One renderer, shared, for the reason ``pack_dry_run.render`` gives: the model, the
    operator and the author at a terminal must be shown the same numbers.
    """
    out: List[str] = []
    if not delta.compared:
        out.append(f"verdict lines: not compared — {delta.reason or 'unknown reason'}")
        for p in delta.problems:
            out.append(f"  ! {p}")
        return "\n".join(out)

    moved = ", ".join(delta.surface_changed) or "(none)"
    out.append(
        f"verdict lines: {delta.runs_changed} of {delta.replayed} re-adjudicated run(s) "
        f"read differently ({len(delta.changes) + delta.changes_cut} line(s) moved; "
        f"ruleset(s): {', '.join(delta.rulesets) or '(none)'}; surface changed: {moved})"
    )
    if not delta.changes:
        out.append(
            "  no stored run's findings move. This is not a claim that the edit is correct "
            "— only that no past adjudication changes because of it."
        )
    if delta.verdicts_moved:
        out.append(
            f"  {len(delta.verdicts_moved)} of them change a summary or a subject's "
            "disposition, which is what a report reader sees first."
        )
    for c in delta.changes:
        where = f"{c.job_id[:8] or '?'} {c.ruleset_key}"
        subject = f" / {c.subject}" if c.subject else ""
        arrow = f"{c.before or '(absent)'} -> {c.after or '(absent)'}"
        note = "  <- determination CHANGED" if c.decided_flip else ""
        out.append(f"  [{c.kind}] {where}{subject} / {c.line}: {arrow}{note}")
        if c.detail:
            out.append(f"      {c.detail}")
    if delta.changes_cut:
        out.append(
            f"  ... and {delta.changes_cut} more changed line(s) not listed (bound: "
            f"{MAX_CHANGES})"
        )
    for limit in delta.limits:
        out.append(f"  - {limit}")
    for p in delta.problems:
        out.append(f"  ! {p}")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m src.knowledge.pack_verdict_delta <base dir> <candidate dir>``."""
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) < 2:
        print(__doc__)
        print("usage: pack_verdict_delta <base pack dir> <candidate pack dir>")
        return 2
    print(render(verdict_delta_for_dirs(args[0], args[1])))
    return 0


if __name__ == "__main__":  # pragma: no cover — the CLI is a thin wrapper
    raise SystemExit(main())
