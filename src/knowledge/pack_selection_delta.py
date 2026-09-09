"""Re-score every stored incident under a candidate pack, and name the runs that flip.

WHAT THIS CATCHES that nothing else does. ``pack_validate`` answers "will the engine read
this declaration" and ``pack_dry_run`` answers "does this ruleset decide anything over rows
that really came back" — both about the candidate pack **in isolation**. Neither can see the
one side effect an authoring edit has on every procedure at once: which ruleset adjudicates
is decided by a keyword score over the playbook titles and join keys, so adding a use case,
or rewording one title, silently re-scores every incident the pack has ever seen. Getting
that wrong does not fail: the losing procedure's conditions still resolve against real rows,
every stage reports success, and the output is a confident verdict from the wrong procedure.

So this is a **difference**, and it needs both sides. It scores the same corpus under the
base pack's specs and the candidate's, through the engine's own selector, and reports the
runs whose selected procedure moved. It decides nothing: a flip is very often the point of
the edit, and "is this flip correct" is a judgement no arithmetic settles.

Three properties it holds, each because the alternative fails quietly:

* **One scorer, not two.** Every score comes from ``correlation.select_correlation_spec_
  explained`` via a duck-typed pack and analysis. A re-implementation here would be a second
  answer to the same question, and it would drift — the tie-break on the secondary
  hypotheses is exactly the kind of detail a copy loses.
* **An unchanged vocabulary cannot move a selection**, so an edit that leaves every spec's
  title and keys alone short-circuits before reading the corpus. Most pack edits are that
  edit, and a check that costs seconds on every save is a check that gets turned off.
* **No corpus means silence, not a clean result.** A deployment with no stored runs can say
  nothing about flips, and reporting "0 flipped" there would read as a guarantee. The counts
  ride on the result either way, or a pack that stopped being checked looks like one that
  passed.

Pure and read-only: no LLM, no network, no writes. Never raises — every failure becomes a
``problems`` entry, because a comparison that aborts on one unreadable job document tells an
author less than one that reports 60 of 61.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

#: Rows scored at most, per side. Scoring is pure string work — ~0.1ms a row — so this is a
#: bound on the report's length as much as on its cost: an author cannot read 400 flips.
DEFAULT_MAX_ROWS = 400

#: Flips listed. A cut list says so (``flips_cut``), because a reader who has learned that
#: these lists are complete reads a truncated one as the whole consequence.
MAX_FLIPS = 25


def _correlation():
    """The verdict engine, imported late — see ``pack_dry_run._correlation`` for why.

    Delegated rather than duplicated: two spellings of the same path fix-up is two answers
    to which module identity ``correlation`` has, and this codebase already carries a rule
    about that.
    """
    from src.knowledge import pack_dry_run

    return pack_dry_run._correlation()


# ------------------------------------------------------------------------- the results


@dataclass(frozen=True)
class CorpusRow:
    """One past incident, in the shape the selector reads it.

    ``label`` is an independent reading of which procedure *should* adjudicate this
    incident, supplied by the caller and empty when there is none. It is never derived
    here: what an incident is about is domain knowledge, and this module is engine surface.
    """

    job_id: str = ""
    summary: str = ""
    hypotheses: Tuple[str, ...] = ()
    areas: Tuple[str, ...] = ()
    label: str = ""


@dataclass(frozen=True)
class Flip:
    """One incident whose selected procedure moved, with both readings side by side.

    Both bases are carried, not just the use-case names, because the three ways a
    selection can move license different reactions: ``scored -> scored`` is a contest the
    edit re-decided, ``no_match -> scored`` is a procedure that now recognises an incident
    nothing recognised before (usually the point of the edit), and ``scored -> no_match``
    is recognition LOST, which is the one direction an author never intends.
    """

    job_id: str = ""
    before: str = ""
    after: str = ""
    before_basis: str = ""
    after_basis: str = ""
    before_score: float = 0.0
    after_score: float = 0.0
    label: str = ""

    @property
    def lost_recognition(self) -> bool:
        """True when the candidate pack recognises an incident the base pack selected for."""
        return self.after_basis == "no_match" and self.before_basis != "no_match"

    @property
    def toward_label(self) -> Optional[bool]:
        """Whether the flip moved toward the caller's own reading; ``None`` if unlabelled."""
        if not self.label:
            return None
        return self.after == self.label and self.before != self.label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "before": self.before,
            "after": self.after,
            "before_basis": self.before_basis,
            "after_basis": self.after_basis,
            "before_score": round(self.before_score, 4),
            "after_score": round(self.after_score, 4),
            "label": self.label,
            "lost_recognition": self.lost_recognition,
            "toward_label": self.toward_label,
        }


@dataclass
class SelectionDelta:
    """What the candidate pack's vocabulary does to the corpus's selections.

    ``compared`` is the field a reader must check first. False means no comparison happened
    — no corpus, no specs, or an unchanged vocabulary — and then an empty ``flips`` is a
    silence rather than a clean bill.
    """

    compared: bool = False
    reason: str = ""
    scored: int = 0
    corpus: int = 0
    base_specs: int = 0
    candidate_specs: int = 0
    vocabulary_changed: Tuple[str, ...] = ()
    flips: Tuple[Flip, ...] = ()
    flips_cut: int = 0
    problems: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def lost_recognition(self) -> Tuple[Flip, ...]:
        return tuple(f for f in self.flips if f.lost_recognition)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compared": self.compared,
            "reason": self.reason,
            "scored": self.scored,
            "corpus": self.corpus,
            "base_specs": self.base_specs,
            "candidate_specs": self.candidate_specs,
            "vocabulary_changed": list(self.vocabulary_changed),
            "flips": [f.to_dict() for f in self.flips],
            "flips_cut": self.flips_cut,
            "problems": list(self.problems),
            # The rendered text rides along, as it does on ``pack_dry_run.as_dict``: the
            # preview shows the server's own sentences rather than re-deriving them in JS,
            # or the operator approving the edit and the model proposing it are reading two
            # descriptions of one measurement.
            "text": render(self),
        }


# --------------------------------------------------------------------- reading the corpus


def corpus(
    store: Any = None,
    *,
    jobs_dir: Optional[Path] = None,
    labels: Optional[Dict[str, str]] = None,
) -> Tuple[List[CorpusRow], List[str]]:
    """Every stored run that carries an incident summary, newest first, deduped by summary.

    Read through ``JobStore.load_all`` like ``pack_dry_run.stored_runs``, but **not** through
    that function: it keeps only runs that retrieved rows, and a run that retrieved nothing
    is precisely the run a mis-selected procedure produces. Excluding those would hide the
    failure this module exists to report.

    Deduped on the summary text because the history is re-runs of a much smaller set of
    incidents — the same alert replayed a dozen times is one selection, and counting it
    twelve times turns one flip into twelve. Newest kept, matching ``stored_runs``' reason
    for reversing: the newest reading of an incident is the relevant one.
    """
    problems: List[str] = []
    try:
        if store is None:
            from src.job_store import JobStore

            store = JobStore(base_dir=jobs_dir) if jobs_dir else JobStore()
        docs = store.load_all() or []
    except Exception as e:  # noqa: BLE001 — an unreadable history is a finding, not a crash
        logger.warning("Selection delta could not read the job history: %s", e)
        return [], [f"the stored job history could not be read: {e}"]

    rows: List[CorpusRow] = []
    for doc in reversed(list(docs)):  # load_all is oldest-first
        try:
            raw = ((doc.get("outputs") or {}).get("understanding") or {}).get("analysis")
            if not isinstance(raw, dict):
                continue
            summary = str(raw.get("incident_summary") or "").strip()
            if not summary:
                continue
            job_id = str(doc.get("job_id") or "")
            rows.append(
                CorpusRow(
                    job_id=job_id,
                    summary=summary,
                    hypotheses=tuple(
                        str(h) for h in (raw.get("initial_hypotheses") or [])
                    ),
                    areas=tuple(
                        str(a) for a in (raw.get("key_investigation_areas") or [])
                    ),
                    label=str((labels or {}).get(job_id) or ""),
                )
            )
        except Exception as e:  # noqa: BLE001 — one bad document must not lose the rest
            problems.append(
                f"{str(doc.get('job_id') or '?')[:8]}: its understanding output could not "
                f"be read ({type(e).__name__}) — skipped"
            )

    seen: Set[str] = set()
    unique: List[CorpusRow] = []
    for row in rows:
        if row.summary in seen:
            continue
        seen.add(row.summary)
        unique.append(row)
    return unique, problems


# ------------------------------------------------------------------------- the scoring


def _shim_analysis(row: CorpusRow) -> Any:
    """The four attributes the selector reads off an analysis, and nothing else.

    Duck-typed rather than an ``IncidentAnalysis``: the selector reads through
    ``getattr``, and building the Pydantic model would drag its required fields into a
    module whose whole input is a summary string.
    """
    return SimpleNamespace(
        pinned_use_case="",
        incident_summary=row.summary,
        initial_hypotheses=list(row.hypotheses),
        key_investigation_areas=list(row.areas),
    )


def score(specs: Sequence[Dict[str, Any]], row: CorpusRow) -> Tuple[str, Any]:
    """``(use_case, SelectionBasis)`` for this row under these specs, via the engine.

    An empty use case means the selector abstained — ``basis`` says which of the two
    abstentions it was, and the caller's own fallback is what decides what runs.
    """
    corr = _correlation()
    pack = SimpleNamespace(correlation_specs=lambda: list(specs))
    spec, basis = corr.select_correlation_spec_explained(pack, _shim_analysis(row))
    return (str((spec or {}).get("use_case") or ""), basis)


def spec_vocabulary(specs: Sequence[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Each spec's discriminating tokens, keyed by use case — the whole scoring input.

    Uses the engine's own splitter rather than a local one: a field-name splitter returns a
    multi-word title as one unmatchable token, and a copy of the prose splitter here could
    disagree with the scorer about what a token even is, which would make this check confirm
    an unchanged vocabulary while the scores moved.
    """
    prose_tokens = _correlation()._prose_tokens
    out: Dict[str, Set[str]] = {}
    for spec in specs:
        tokens = set(prose_tokens(str(spec.get("title", "") or "")))
        for key in spec.get("keys") or []:
            tokens.update(prose_tokens(str(key)))
        tokens.discard("")
        out[str(spec.get("use_case", "") or "")] = tokens
    return out


def vocabulary_changed(
    base: Sequence[Dict[str, Any]], candidate: Sequence[Dict[str, Any]]
) -> Tuple[str, ...]:
    """The use cases whose scoring vocabulary the edit moved, added or removed.

    Empty means no selection can move, and that is exact rather than a heuristic: the score
    is a function of the token sets alone — including the inverse-spec-frequency weights,
    which are counts over those same sets — so identical sets score identically on every
    input. Which is what licenses the short-circuit in :func:`compare`.
    """
    before, after = spec_vocabulary(base), spec_vocabulary(candidate)
    return tuple(
        sorted(
            name
            for name in set(before) | set(after)
            if before.get(name) != after.get(name)
        )
    )


def compare(
    base_specs: Sequence[Dict[str, Any]],
    candidate_specs: Sequence[Dict[str, Any]],
    rows: Sequence[CorpusRow],
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    problems: Optional[Sequence[str]] = None,
) -> SelectionDelta:
    """Score ``rows`` under both spec sets and report the selections that moved."""
    out = SelectionDelta(
        corpus=len(rows),
        base_specs=len(base_specs),
        candidate_specs=len(candidate_specs),
        problems=tuple(problems or ()),
    )
    try:
        if not base_specs and not candidate_specs:
            out.reason = "neither pack declares a playbook correlation block"
            return out
        out.vocabulary_changed = vocabulary_changed(base_specs, candidate_specs)
        if not out.vocabulary_changed:
            out.reason = (
                "no playbook title or join key changed, so every incident scores exactly "
                "as it did"
            )
            return out
        if not rows:
            out.reason = (
                "no stored run carries an incident summary, so this deployment cannot say "
                "which past selections the edit moves"
            )
            return out

        flips: List[Flip] = []
        cut = 0
        for row in list(rows)[:max_rows]:
            out.scored += 1
            before, before_basis = score(base_specs, row)
            after, after_basis = score(candidate_specs, row)
            # The BASIS is part of the identity, not just the name: the same procedure
            # reached by `sole_spec` rather than by `scored` was not selected at all, it was
            # the only candidate — and a pack going from one spec to two is exactly the edit
            # that makes that distinction start to matter.
            if before == after and before_basis.basis == after_basis.basis:
                continue
            if len(flips) >= MAX_FLIPS:
                cut += 1
                continue
            flips.append(
                Flip(
                    job_id=row.job_id,
                    before=before,
                    after=after,
                    before_basis=before_basis.basis,
                    after_basis=after_basis.basis,
                    before_score=float(before_basis.score),
                    after_score=float(after_basis.score),
                    label=row.label,
                )
            )
        out.flips = tuple(flips)
        out.flips_cut = cut
        out.compared = True
        if len(rows) > max_rows:
            out.problems = out.problems + (
                f"{len(rows) - max_rows} of {len(rows)} stored incidents were not scored "
                f"(bound: {max_rows}) — a flip among them would not appear here",
            )
    except Exception as e:  # noqa: BLE001 — a broken comparison is a finding, not a crash
        logger.warning("Selection delta failed: %s", e, exc_info=True)
        out.compared = False
        out.problems = out.problems + (
            f"the selection comparison could not be completed ({type(e).__name__}: {e})",
        )
    return out


def selection_delta(
    base_specs: Sequence[Dict[str, Any]],
    candidate_specs: Sequence[Dict[str, Any]],
    *,
    store: Any = None,
    jobs_dir: Optional[Path] = None,
    labels: Optional[Dict[str, str]] = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> SelectionDelta:
    """:func:`compare` over the stored corpus — the one call a caller with two packs needs.

    The corpus is read only once the vocabulary is known to have moved, so the common case
    (an edit that touches no title) costs two token-set builds and no IO at all.
    """
    try:
        if not vocabulary_changed(base_specs, candidate_specs):
            return compare(base_specs, candidate_specs, [], max_rows=max_rows)
    except Exception:  # noqa: BLE001 — fall through to compare, which reports its failures
        pass
    rows, problems = corpus(store, jobs_dir=jobs_dir, labels=labels)
    return compare(
        base_specs, candidate_specs, rows, max_rows=max_rows, problems=problems
    )


def selection_delta_for_dirs(
    base_dir: Any,
    candidate_dir: Any,
    *,
    store: Any = None,
    jobs_dir: Optional[Path] = None,
    labels: Optional[Dict[str, str]] = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> SelectionDelta:
    """:func:`selection_delta` between two pack DIRECTORIES — the plan-preview entry point.

    Both sides go through ``load_knowledge_pack``, which is the loader the pipeline uses, so
    a candidate whose playbooks no longer parse yields no specs and the delta says so rather
    than comparing against a silently emptied pack. A pack that will not load at all is a
    ``problems`` entry: this is not the check that reports it, and refusing to answer here
    would hide the flips in a plan that has a second, unrelated defect.
    """
    from src.knowledge.pack import load_knowledge_pack

    specs: Dict[str, List[Dict[str, Any]]] = {}
    problems: List[str] = []
    for side, path in (("base", base_dir), ("candidate", candidate_dir)):
        try:
            specs[side] = list(load_knowledge_pack(path).correlation_specs() or [])
        except Exception as e:  # noqa: BLE001 — an unloadable pack is somebody else's finding
            specs[side] = []
            problems.append(f"the {side} pack could not be loaded: {e}")
    if problems:
        return compare(
            specs["base"], specs["candidate"], [], max_rows=max_rows, problems=problems
        )
    return selection_delta(
        specs["base"],
        specs["candidate"],
        store=store,
        jobs_dir=jobs_dir,
        labels=labels,
        max_rows=max_rows,
    )


# --------------------------------------------------------------------------- rendering


def render(delta: SelectionDelta) -> str:
    """The delta as text, for the plan preview and the CLI.

    One renderer, shared, for the reason ``pack_dry_run.render`` gives: the model, the
    operator and the author at a terminal must be shown the same numbers.
    """
    lines: List[str] = []
    if not delta.compared:
        lines.append(
            f"procedure selection: not compared — {delta.reason or 'unknown reason'}"
        )
        for p in delta.problems:
            lines.append(f"  ! {p}")
        return "\n".join(lines)

    changed = ", ".join(delta.vocabulary_changed) or "(none)"
    lines.append(
        f"procedure selection: {len(delta.flips) + delta.flips_cut} of {delta.scored} "
        f"stored incident(s) select a different procedure "
        f"({delta.base_specs} spec(s) before, {delta.candidate_specs} after; "
        f"vocabulary changed: {changed})"
    )
    if not delta.flips:
        lines.append(
            "  no stored incident's selection moves. This is not a claim that the edit is "
            "correct — only that no past incident's procedure changes because of it."
        )
    for f in delta.flips:
        before = f"{f.before or '(none)'} [{f.before_basis} {f.before_score:.2f}]"
        after = f"{f.after or '(none)'} [{f.after_basis} {f.after_score:.2f}]"
        note = ""
        if f.lost_recognition:
            note = "  <- recognition LOST: nothing now selects for this incident"
        elif f.toward_label is True:
            note = f"  <- toward the expected procedure ({f.label})"
        elif f.toward_label is False:
            note = f"  <- away from the expected procedure ({f.label})"
        lines.append(f"  {f.job_id[:8] or '?'}: {before} -> {after}{note}")
    if delta.flips_cut:
        lines.append(
            f"  ... and {delta.flips_cut} more flipped incident(s) not listed (bound: "
            f"{MAX_FLIPS})"
        )
    for p in delta.problems:
        lines.append(f"  ! {p}")
    lines.append(
        "  A flip is not a defect: re-scoring past incidents is what a title edit is for. "
        "What it is, is the edit's blast radius — read it before saving."
    )
    return "\n".join(lines)
