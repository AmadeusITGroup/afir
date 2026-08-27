"""Replay a candidate pack's rulesets over the evidence of runs already stored.

WHAT THIS CATCHES that nothing else does. ``pack_validate`` answers "will the engine read
this declaration"; a schema check answers "is this leaf on the source". Neither answers the
one that costs a live investigation: a condition that is valid YAML, names a real leaf, and
reads ``unknown`` on every row that has ever actually come back. That condition never
votes, the verdict is unchanged, and the only trace is one line in a report nobody reads as
a defect. So this evaluates the ruleset against real retrieved rows and counts the three
outcomes per condition, plus a fourth the counts cannot express.

**Two findings, and they are not the same defect.** ``always unknown`` is the one above: the
condition WAS asked, on every subject of every replayed run, and answered nothing — so the
fix is a field path, a source, a row selector or a bound. ``never evaluated`` is the other:
no check line at all. ``evaluate_verdict`` emits one line per condition per SUBJECT
unconditionally, with no branch that skips a condition, so this second bucket is really a
statement about the ruleset — it resolved no subject — arriving per condition. Neither is
folded into ``unknown``, and ``kind: stub`` is excluded from the first, because a pack ships
one to declare that a question was considered and cannot be asked yet.

**Every bound names its DIRECTION of error.** Two facts the verdict engine is given by its
caller are absent from a stored run — the per-source row cap and whether a query constrained
the acting identity — so this reading is not the reading the run took, and which way it
leans depends on the kind. Stating "row_caps unavailable" without the direction lets a
reader take a clean dry run as a guarantee. See :func:`_limits`.

Pure and read-only: no LLM, no network, no writes. Never raises — every failure becomes a
``problems`` entry, because a dry run that aborts on one unreadable job document tells the
author less than one that reports 11 of 12.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

#: Runs replayed at most, across every ruleset. A replay is ~1s of pure CPU per run on the
#: installed packs, and this runs inside an HTTP request an operator is waiting on.
DEFAULT_MAX_RUNS = 12

#: Wall-clock budget, checked between runs. Belt and braces with the run count: a single
#: run's cost scales with the rows it retrieved, which no count can bound.
DEFAULT_MAX_SECONDS = 25.0

#: Rows per source scanned when resolving the entity map, mirroring what the pipeline's own
#: resolution reads. More rows cannot add a leaf a source does not have.
ENTITY_MAP_ROWS = 50

#: The budget when the drafting assistant calls this as a tool, rather than the operator's
#: one-shot preview. Lower on both axes because the model may call it several times inside
#: one exploration loop, and every second of it is spent inside the operator's request.
TOOL_MAX_RUNS = 6
TOOL_MAX_SECONDS = 12.0


def _correlation():
    """The verdict engine, imported late.

    ``correlation`` is a flat-import module (``from anomaly_detection import ...``), so it
    resolves only with the repo's ``src`` directory on the path — which ``app.py`` and
    ``pytest.ini`` both arrange. Imported at module scope here, this module would drag that
    requirement onto every importer of ``pack_assistant``, including the pack editor's own
    HTTP handlers, which have no business needing the pipeline. Imported flat rather than as
    ``src.correlation`` deliberately: the same module under two names is the dual-import trap
    this codebase already carries a rule about, and the models it builds are compared by
    identity downstream.
    """
    import sys

    from src.utils.paths import REPO_ROOT

    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import correlation  # noqa: E402 — after the path is set up

    return correlation


# ------------------------------------------------------------------------- the results


@dataclass
class ConditionTally:
    """One condition's outcomes across every run replayed.

    ``checks`` counts check LINES, not runs: a ruleset with two subjects evaluates each
    condition twice in one run, and both readings are real. ``runs_silent`` counts RUNS in
    which the condition produced no line at all. The two units are stated wherever this is
    rendered, because a reader who takes ``checks`` for runs will read a two-subject run as
    two corroborating measurements.
    """

    condition_id: str
    label: str = ""
    kind: str = ""
    passes: int = 0
    fails: int = 0
    unknowns: int = 0
    runs_silent: int = 0
    #: A composite's child, which the engine folds into its parent's single check line.
    nested: bool = False

    @property
    def checks(self) -> int:
        return self.passes + self.fails + self.unknowns

    @property
    def never_evaluated(self) -> bool:
        """No replayed run produced a single check line, and one was expected.

        Reachable only where the ruleset resolved no subject, since the engine emits a line
        per condition per subject with no branch that skips one. Kept as its own bucket
        anyway: a zero is not an outcome, and the remedy is the ruleset's subject scoping
        rather than anything on this condition.

        A composite's child is excluded for the same reason ``stub`` is excluded from
        ``always_unknown``: it produces no line BY DESIGN — the parent owns the report line
        and the children decide it — so counting one here would report every composite in
        every pack as this module's headline finding, which is a fabricated defect and the
        shape that gets a check switched off.
        """
        return self.checks == 0 and not self.nested

    @property
    def always_unknown(self) -> bool:
        """Asked on every replayed run and answered on none.

        ``stub`` is excluded because it is the DECLARED not-evaluated kind: a pack ships one
        to state that a question was considered and cannot be asked yet, so reporting it
        beside a check that is silently broken would bury the second in the first.
        """
        return bool(self.checks) and not (self.passes or self.fails) and self.kind != "stub"


@dataclass
class RulesetTally:
    """One ruleset's replay: which runs it could be evaluated against, and what it said."""

    key: str
    runs_replayed: int = 0
    runs_no_verdict: int = 0
    runs_available: int = 0
    #: True when no stored run was adjudicated by this ruleset, so the runs replayed were
    #: paired with it only because they retrieved one of its sources. See :func:`_eligible`.
    unpaired: bool = False
    conditions: List[ConditionTally] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def silent_conditions(self) -> List[ConditionTally]:
        return [c for c in self.conditions if c.never_evaluated]

    @property
    def mute_conditions(self) -> List[ConditionTally]:
        """Conditions asked on every replayed run and answered on none."""
        return [c for c in self.conditions if c.always_unknown]


@dataclass
class DryRunReport:
    """What a dry run concluded, plus what it could not see."""

    pack: str = ""
    runs_available: int = 0
    runs_replayed: int = 0
    seconds: float = 0.0
    rulesets: List[RulesetTally] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def exercised(self) -> bool:
        """True when at least one condition of one ruleset produced a check line."""
        return any(
            c.checks for rs in self.rulesets for c in rs.conditions
        )


@dataclass
class StoredRun:
    """One past run's evidence, in the shape the verdict engine takes.

    ``ruleset_key`` is the procedure that adjudicated the run, recorded on its own brief. It
    is what makes the replay comparable to what happened rather than an arbitrary pairing of
    a ruleset with somebody else's rows; a run whose brief names none is still usable, since
    what decides evaluability is whether the ruleset's sources are in ``logs``.
    """

    job_id: str = ""
    ruleset_key: str = ""
    logs: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    analysis: Any = None
    recorded_summary: str = ""


# --------------------------------------------------------------------- reading the runs


def _analysis_of(raw: Dict[str, Any]) -> Any:
    """The three things the verdict engine reads off an analysis, and nothing else.

    ``event_time`` is included and matters: it is the as-of cutoff, so omitting it keeps
    every version of an append-per-change source in the read. That is the *stronger* reading
    — the run narrowed and this would not — which is why it is supplied rather than left off.
    """
    entities = [
        SimpleNamespace(
            type=str(e.get("entity_type") or e.get("type") or ""),
            value=str(e.get("value") or ""),
            value_form=str(e.get("value_form") or ""),
        )
        for e in (raw.get("extracted_entities") or [])
        if isinstance(e, dict)
    ]
    window = raw.get("event_time") if isinstance(raw.get("event_time"), dict) else None
    return SimpleNamespace(
        extracted_entities=entities,
        incident_summary=str(raw.get("incident_summary") or ""),
        correlation_keys=list(raw.get("correlation_keys") or []),
        event_time=(
            SimpleNamespace(
                start=str((window or {}).get("start") or ""),
                end=str((window or {}).get("end") or ""),
            )
            if window
            else None
        ),
    )


def stored_runs(
    store: Any = None, *, jobs_dir: Optional[Path] = None
) -> Tuple[List[StoredRun], List[str]]:
    """Every stored run that carries retrieved rows, newest first, plus what was unreadable.

    Read through ``JobStore.load_all`` rather than off the sidecar files directly: that
    function is where "the evidence sidecar is the retrieval stage's own ``logs`` output"
    is expressed, and a second reader of the same two files is a second answer to which of
    them wins.
    """
    problems: List[str] = []
    try:
        if store is None:
            from src.job_store import JobStore

            store = JobStore(base_dir=jobs_dir) if jobs_dir else JobStore()
        docs = store.load_all() or []
    except Exception as e:  # noqa: BLE001 — an unreadable history is a finding, not a crash
        logger.warning("Dry run could not read the job history: %s", e)
        return [], [f"the stored job history could not be read: {e}"]

    runs: List[StoredRun] = []
    for doc in docs:
        outputs = doc.get("outputs") or {}
        logs = outputs.get("logs") or {}
        if not isinstance(logs, dict) or not logs:
            continue
        corr = outputs.get("correlation") or {}
        analysis_raw = (outputs.get("understanding") or {}).get("analysis") or {}
        if not isinstance(analysis_raw, dict):
            problems.append(
                f"{str(doc.get('job_id') or '?')[:8]}: its understanding output is not a "
                "mapping, so no subject could be resolved — skipped"
            )
            continue
        runs.append(
            StoredRun(
                job_id=str(doc.get("job_id") or ""),
                ruleset_key=str(((corr.get("brief") or {}).get("use_case") or "")),
                logs={
                    src: [r for r in (rows or []) if isinstance(r, dict)]
                    for src, rows in logs.items()
                    if isinstance(rows, list)
                },
                analysis=_analysis_of(analysis_raw),
                recorded_summary=str((corr.get("verdict") or {}).get("summary") or ""),
            )
        )
    runs.reverse()  # load_all is oldest-first; the newest evidence is the relevant evidence
    return runs, problems


# ------------------------------------------------------------------------ the entity map


def _entity_map(
    pack: Any, logs: Dict[str, List[Dict[str, Any]]], etypes: Set[str]
) -> Dict[str, Dict[str, str]]:
    """The same first-present-leaf resolution the pipeline does, over the replayed rows.

    Rebuilt from the rows rather than read from the job document because no document carries
    it. The consequence is bounded: a leaf the rows do not have cannot be bound here either,
    which is the direction that yields ``unknown`` rather than a fabricated reading.
    """
    flatten_leaves = _correlation().flatten_leaves
    out: Dict[str, Dict[str, str]] = {}
    for source, rows in logs.items():
        leaves: Set[str] = set()
        for row in (rows or [])[:ENTITY_MAP_ROWS]:
            leaves.update(flatten_leaves(row).keys())
        lower = {leaf.lower(): leaf for leaf in leaves}
        mapping: Dict[str, str] = {}
        for etype in etypes:
            for candidate in pack.field_priors_for(etype, source):
                if candidate in leaves:
                    mapping[etype] = candidate
                    break
                real = lower.get(str(candidate).lower())
                if real is not None:
                    mapping[etype] = real
                    break
        if mapping:
            out[source] = mapping
    return out


def _entity_types(run: StoredRun) -> Set[str]:
    return {
        str(getattr(e, "type", "") or "")
        for e in (getattr(run.analysis, "extracted_entities", None) or [])
    } - {""}


# ------------------------------------------------------------------------ the replay plan


#: ``{condition id: tally}``, in declaration order.
ConditionIndex = Dict[str, ConditionTally]


def _condition_index(spec: Dict[str, Any]) -> ConditionIndex:
    """Every condition the ruleset declares, by id, in declaration order.

    Composite kinds are walked into so the table lists what the ruleset actually declares —
    a reader looking for a child id has to find it. But a composite emits **one** check line
    for the parent and none for its children, so a child is marked ``nested`` and excluded
    from the never-evaluated finding: a zero there is the design, not a defect.
    """
    index: Dict[str, ConditionTally] = {}

    def walk(conds: Iterable[Any], depth: int) -> None:
        if depth > 6:
            return
        for cond in conds or []:
            if not isinstance(cond, dict):
                continue
            cid = str(cond.get("id") or "").strip()
            if cid and cid not in index:
                index[cid] = ConditionTally(
                    condition_id=cid,
                    label=str(cond.get("label") or ""),
                    kind=str(cond.get("kind") or ""),
                    nested=depth > 0,
                )
            children = cond.get("children")
            if isinstance(children, list):
                walk(children, depth + 1)

    walk(spec.get("conditions") or [], 0)
    return index


def _spec_sources(spec: Dict[str, Any]) -> Set[str]:
    """The real source names the ruleset declares. Empty means it can never be evaluated."""
    return {
        str(real)
        for real in (spec.get("sources") or {}).values()
        if str(real or "").strip()
    }


def _eligible(
    key: str, spec: Dict[str, Any], runs: Sequence[StoredRun]
) -> Tuple[List[StoredRun], bool]:
    """``(runs to replay this ruleset against, whether the pairing is unpaired)``.

    A run is PAIRED with the ruleset that actually adjudicated it, recorded on its brief.
    That is the only pairing whose outcomes mean anything: procedures in one domain share
    sources heavily, so "this run retrieved one of its sources" admitted nearly every run to
    nearly every ruleset — measured 60 of 65 for each of nine rulesets — and each replay then
    read a procedure's conditions against a different procedure's rows and returned `unknown`
    on all of them. A blanket `unknown` reported as a finding is the fabricated-defect shape,
    and it would send an author to rewrite a check that works.

    Source overlap survives only as the FALLBACK for a ruleset no stored run adjudicated —
    which is exactly a newly authored one, the case this module exists for — and it is
    flagged ``unpaired`` so the weaker reading is never presented as the paired one.
    """
    paired = [r for r in runs if r.ruleset_key and r.ruleset_key == key]
    if paired:
        return paired, False
    wanted = _spec_sources(spec)
    return [r for r in runs if wanted & set(r.logs)], True


def _select(
    per_key: Dict[str, List[StoredRun]], budget: int
) -> List[Tuple[str, StoredRun]]:
    """``(ruleset key, run)`` pairs to replay, round-robin across the keys.

    Round-robin rather than newest-first overall: one procedure's runs dominate the recent
    history on any real deployment, and a budget spent entirely on it would report every
    other ruleset's conditions as ``never evaluated`` — the module's headline finding,
    fabricated by the selection rule.
    """
    pending = {key: list(items) for key, items in per_key.items() if items}
    out: List[Tuple[str, StoredRun]] = []
    while pending and len(out) < budget:
        for key in sorted(pending):
            if len(out) >= budget:
                break
            queue = pending[key]
            out.append((key, queue.pop(0)))
            if not queue:
                pending.pop(key, None)
    return out


def _limits(
    row_caps: Optional[Dict[str, int]],
    keyed_sources: Optional[Dict[str, bool]],
    clipped: int,
    total: int,
    over_budget: bool,
) -> List[str]:
    """What this reading cannot see, each stated with the direction it errs in.

    A bound with no direction is worse than no bound: an operator told "row caps were
    unavailable" cannot tell whether to trust a clean result, and a clean result is exactly
    what an author is hoping to see.
    """
    out: List[str] = []
    if not row_caps:
        out.append(
            "No per-source row cap was supplied, and no stored run records one. A "
            "truncated read is a bound on the value rather than the value, so the "
            "`unknown` the live engine returns for a counting condition whose source hit "
            "its cap CANNOT appear here: this reading is OPTIMISTIC for the counting kinds."
        )
    if not keyed_sources:
        out.append(
            "Whether a query constrained the acting identity was not supplied, and no "
            "stored run records it. An empty result therefore reads as a gap rather than "
            "as the absence it may have been, so a condition whose finding IS its emptiness "
            "reads `unknown` here where the run decided it: PESSIMISTIC for absence checks."
        )
    out.append(
        "A source that was asked and did not answer is not distinguishable in stored "
        "evidence from one that answered with nothing, so a condition over a timed-out "
        "source reads as answered-empty: OPTIMISTIC."
    )
    if clipped:
        out.append(
            f"{clipped} of {total} stored runs were not replayed"
            + (" (the time budget was spent)" if over_budget else " (the run budget)")
            + ". Replaying more can only ADD outcomes, so a condition reported "
            "`never evaluated` here may well fire on a run that was not replayed."
        )
    return out


# --------------------------------------------------------------------------- the dry run


def dry_run(
    pack_dir: Any,
    *,
    ruleset_keys: Optional[Sequence[str]] = None,
    runs: Optional[Sequence[StoredRun]] = None,
    store: Any = None,
    row_caps: Optional[Dict[str, int]] = None,
    keyed_sources: Optional[Dict[str, bool]] = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> DryRunReport:
    """Evaluate ``pack_dir``'s rulesets over stored evidence and count the outcomes.

    ``pack_dir`` may be a candidate tree, which is the point: the question an author has is
    what the pack would do AFTER the edit. ``row_caps`` / ``keyed_sources`` are accepted and
    never invented — they are the caller's to supply, are resolvable only from built
    retrievers, and their absence is reported with its direction rather than papered over.
    """
    started = time.monotonic()
    report = DryRunReport(pack=str(Path(pack_dir).name))
    try:
        from src.knowledge.pack import load_knowledge_pack

        pack = load_knowledge_pack(pack_dir)
    except Exception as e:  # noqa: BLE001 — a pack that will not load is the finding
        report.problems.append(f"the pack could not be loaded: {e}")
        report.limits = _limits(row_caps, keyed_sources, 0, 0, False)
        return report

    if runs is None:
        runs, problems = stored_runs(store)
        report.problems.extend(problems)
    report.runs_available = len(runs)
    if not runs:
        report.problems.append(
            "No stored run carries retrieved rows, so nothing could be replayed. A dry run "
            "needs a completed retrieval to read; this is not a finding about the pack."
        )
        report.limits = _limits(row_caps, keyed_sources, 0, 0, False)
        return report

    keys = [str(k) for k in (ruleset_keys or pack.ruleset_keys())]
    tallies: Dict[str, RulesetTally] = {}
    indexes: Dict[str, ConditionIndex] = {}
    specs: Dict[str, Dict[str, Any]] = {}
    per_key: Dict[str, List[StoredRun]] = {}
    for key in keys:
        tally = RulesetTally(key=key)
        tallies[key] = tally
        try:
            spec = pack.ruleset_spec(key)
        except Exception as e:  # noqa: BLE001 — an unresolvable import raises by design
            tally.problems.append(f"its spec could not be resolved: {e}")
            continue
        if not spec:
            tally.problems.append("the pack declares no ruleset under this key")
            continue
        specs[key] = spec
        indexes[key] = _condition_index(spec)
        tally.conditions = list(indexes[key].values())
        wanted = _spec_sources(spec)
        if not wanted:
            tally.problems.append(
                "it declares no sources, so the engine returns no verdict for it at all"
            )
            continue
        eligible, tally.unpaired = _eligible(key, spec, runs)
        tally.runs_available = len(eligible)
        if not eligible:
            tally.problems.append(
                "no stored run was adjudicated by it and none retrieved any of its "
                "sources, so it was not exercised — the conditions below are unmeasured, "
                "not silent"
            )
            continue
        per_key[key] = eligible

    plan = _select(per_key, max(0, int(max_runs)))
    replayed_ids: Set[str] = set()
    over_budget = False
    evaluate = _correlation().evaluate_verdict
    pack_data = getattr(pack, "pack_data", None)
    for key, run in plan:
        if time.monotonic() - started > max_seconds:
            over_budget = True
            break
        tally = tallies[key]
        index = indexes[key]
        etypes = _entity_types(run)
        try:
            verdict = evaluate(
                specs[key],
                run.logs,
                run.analysis,
                entity_map=_entity_map(pack, run.logs, etypes),
                pack_data=pack_data,
                row_caps=row_caps,
                keyed_sources=keyed_sources,
            )
        except Exception as e:  # noqa: BLE001 — one bad run must not lose the other eleven
            logger.warning("Dry run failed on job %s: %s", run.job_id[:8], e)
            tally.problems.append(f"job {run.job_id[:8]} raised {type(e).__name__}: {e}")
            continue
        tally.runs_replayed += 1
        replayed_ids.add(run.job_id)
        if verdict is None:
            tally.runs_no_verdict += 1
            continue
        seen_here: Set[str] = set()
        for subject in getattr(verdict, "subjects", None) or []:
            for check in getattr(subject, "checks", None) or []:
                cid = str(getattr(check, "id", "") or "")
                entry = index.get(cid)
                if entry is None:
                    # A check the spec's own condition list does not name: a composite's
                    # child promoted to a line, or an id the resolver rewrote. Counted
                    # rather than dropped, so the totals reconcile.
                    entry = ConditionTally(
                        condition_id=cid or "?",
                        label=str(getattr(check, "label", "") or ""),
                    )
                    index[cid] = entry
                    tally.conditions.append(entry)
                result = str(getattr(check, "result", "") or "")
                if result == "pass":
                    entry.passes += 1
                elif result == "fail":
                    entry.fails += 1
                else:
                    entry.unknowns += 1
                seen_here.add(cid)
        for cid, entry in index.items():
            if cid not in seen_here and not entry.nested:
                entry.runs_silent += 1

    report.rulesets = [tallies[k] for k in keys]
    report.runs_replayed = len(replayed_ids)
    total_eligible = len({r.job_id for items in per_key.values() for r in items})
    report.limits = _limits(
        row_caps,
        keyed_sources,
        max(0, total_eligible - report.runs_replayed),
        total_eligible,
        over_budget,
    )
    report.seconds = round(time.monotonic() - started, 2)
    return report


# ---------------------------------------------------------------------------- rendering


def render(report: DryRunReport) -> str:
    """The report as text, for the assistant's tool result and the plan preview.

    One renderer, shared: the model and the operator must be shown the same numbers, and a
    second formatter is a second chance for one of them to be reassured by a different
    reading of the same run.
    """
    lines: List[str] = [
        f"dry run over stored evidence: {report.runs_replayed} run(s) replayed of "
        f"{report.runs_available} available, {report.seconds}s"
    ]
    for rs in report.rulesets:
        head = f"\nruleset {rs.key}: "
        if rs.problems:
            lines.append(head + "; ".join(rs.problems))
        else:
            head += f"{rs.runs_replayed} run(s) replayed of {rs.runs_available} " + (
                "paired with it merely by a shared source — NO stored run was "
                "adjudicated by this ruleset, so these outcomes are about another "
                "procedure's rows and an `unknown` here says little"
                if rs.unpaired
                else "it adjudicated"
            )
            if rs.runs_no_verdict:
                head += f", {rs.runs_no_verdict} produced no verdict"
            lines.append(head)
        if not rs.conditions:
            lines.append("  (this ruleset declares no conditions)")
            continue
        lines.append("  condition\tkind\tpass\tfail\tunknown\truns silent")
        for c in rs.conditions:
            # Lower case for the folded marker and upper case for the two findings: a
            # composite's child carrying zeroes is the design, and a marker that shouts
            # reads as the defect the two above it are.
            if c.nested:
                mark = "  folded  "
            elif c.never_evaluated:
                mark = "  NEVER EVALUATED  "
            else:
                mark = "  ALWAYS UNKNOWN  " if c.always_unknown else "  "
            lines.append(
                f"{mark}{c.condition_id}\t{c.kind}\t{c.passes}\t{c.fails}\t{c.unknowns}"
                f"\t{c.runs_silent}"
            )
        if any(c.nested for c in rs.conditions):
            lines.append(
                "  `folded` marks a composite's child: the parent produces the one check "
                "line the children decide, so a child's own counts are always zero and "
                "the parent's row is where its effect shows."
            )
        mute = rs.mute_conditions
        if mute and rs.runs_replayed:
            lines.append(
                f"  {len(mute)} condition(s) were asked on every subject of every "
                "replayed run and answered on none. Each one is declared, resolves, votes "
                "on nothing, and costs the scan of its source. Check the field path, the "
                "row selector and the bound — a `kind: stub` is excluded from this count, "
                "so none of these is a declared not-evaluated entry."
            )
        silent = rs.silent_conditions
        if silent and rs.runs_replayed:
            lines.append(
                f"  {len(silent)} condition(s) produced no check line at all. The engine "
                "emits one line per condition per SUBJECT, so this says the ruleset "
                "resolved no subject on these runs — read it against `subject_entity` and "
                "the subject-discovery source, not against the conditions."
            )
    lines.append(
        "\nCounts are check LINES (one per subject per run), except `runs silent`, "
        "which counts RUNS."
    )
    if report.problems:
        lines.append("\nproblems:")
        lines.extend(f"  - {p}" for p in report.problems)
    if report.limits:
        lines.append("\nwhat this reading cannot see:")
        lines.extend(f"  - {lim}" for lim in report.limits)
    return "\n".join(lines)


def as_dict(report: DryRunReport) -> Dict[str, Any]:
    """The report as JSON, for the plan preview the UI renders."""
    return {
        "pack": report.pack,
        "runs_available": report.runs_available,
        "runs_replayed": report.runs_replayed,
        "seconds": report.seconds,
        "exercised": report.exercised,
        "rulesets": [
            {
                "key": rs.key,
                "runs_replayed": rs.runs_replayed,
                "runs_available": rs.runs_available,
                "runs_no_verdict": rs.runs_no_verdict,
                "unpaired": rs.unpaired,
                "problems": list(rs.problems),
                "conditions": [
                    {
                        "id": c.condition_id,
                        "label": c.label,
                        "kind": c.kind,
                        "pass": c.passes,
                        "fail": c.fails,
                        "unknown": c.unknowns,
                        "runs_silent": c.runs_silent,
                        "nested": c.nested,
                        "never_evaluated": c.never_evaluated,
                        "always_unknown": c.always_unknown,
                    }
                    for c in rs.conditions
                ],
            }
            for rs in report.rulesets
        ],
        "limits": list(report.limits),
        "problems": list(report.problems),
        "text": render(report),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m src.knowledge.pack_dry_run <pack dir> [ruleset key ...]``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pack_dir")
    parser.add_argument("ruleset_keys", nargs="*")
    parser.add_argument("--max-runs", type=int, default=DEFAULT_MAX_RUNS)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = dry_run(
        args.pack_dir,
        ruleset_keys=args.ruleset_keys or None,
        max_runs=args.max_runs,
        max_seconds=args.max_seconds,
    )
    print(json.dumps(as_dict(report), indent=1) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
