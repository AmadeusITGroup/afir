#!/usr/bin/env python3
"""Measure cross-procedure link signals over the real runs in ``jobs/``.

Every declaration ships with ``base_rate: {kind: stub}`` until this script has counted it.

What it measures per (source -> target) pair:
  corpus  -- distinct incidents where the target was a candidate
  reach   -- of those, how many had the target's subject entity in hand (rung 0)
  fire    -- how many had a declared entry signal fire on already-retrieved rows
  agree   -- of fired+evaluable runs, the fraction where the target's gate did not fail

A high fire rate is a failing grade: it means the signal describes the population, not a link.

Does not gate anything. The signal_discriminates term is additive; an unmeasured pair
escalates the same as a measured one. --check exits non-zero on CONTRADICTED alone; the
reporting run always exits 0.

Nothing is written to the pack from here; updating a base_rate is an authored act.

Replay limitation: the export does not carry row_caps, keyed_sources, or unanswered_sources,
so the agreement column is biased upward (a truncated source reads as complete).

Usage::

    .venv/bin/python scripts/measure_link_signals.py
    .venv/bin/python scripts/measure_link_signals.py --json
    .venv/bin/python scripts/measure_link_signals.py --check
    .venv/bin/python scripts/measure_link_signals.py --pack knowledge/<domain> --min-corpus 10
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
# Flat path for src/links.py; package path for models and pack loader.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from link_escalation import MIN_BASE_RATE_CORPUS, base_rate_measured  # noqa: E402
from links import _fired_signals, _gate_outcome, assess_links  # noqa: E402
from src.knowledge.pack import load_knowledge_pack  # noqa: E402
from src.models.pydantic_models import CorrelationResult  # noqa: E402
from src.models.pydantic_models import UnderstandingResult  # noqa: E402

# MIN_CORPUS is imported from link_escalation, not restated: a local copy could drift
# so this script blesses a pair that the scorer still ignores. Changing any bar is a
# visible diff.
MIN_CORPUS = MIN_BASE_RATE_CORPUS
# A signal on more than half the runs of the procedure that retrieved the rows is describing the
# population, not a link. The selector that produced this bar was on 340 of 586 alerts — it
# matched both the alert title and the subject, and would have fired on 58% of them regardless.
MAX_FIRE_RATE = 0.5
# Of the runs where it fired and the target's own gate could answer, the gate must mostly agree.
MIN_AGREEMENT = 0.5

_STATES = ("probed_positive", "not_probed", "probed_negative", "unreachable")


# --- corpus ------------------------------------------------------------------------------


def _load_pack(pack_dir: Optional[str]):
    """The installed pack, by path or by discovery. Never by name."""
    if pack_dir:
        path = Path(pack_dir)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not (path / "source_catalog.yaml").is_file():
            raise SystemExit(f"{path} carries no source_catalog.yaml")
        return load_knowledge_pack(path)
    packs = sorted(
        p
        for p in (REPO_ROOT / "knowledge").iterdir()
        if (p / "source_catalog.yaml").is_file()
    )
    if not packs:
        raise SystemExit("no installed pack carries a source_catalog.yaml")
    return load_knowledge_pack(packs[0])


def _richness(doc: Dict[str, Any]) -> Tuple[int, int, str]:
    """Completion key for deduplication: complete first, then most outputs, then newest."""
    outputs = doc.get("outputs") or {}
    complete = 1 if (outputs.get("correlation") and not doc.get("error")) else 0
    return (complete, len(outputs), str(doc.get("created_at") or ""))


def _runs(jobs_dir: Path) -> List[Dict[str, Any]]:
    """One job doc per distinct incident text (deduped on alert text, not job id)."""
    by_incident: Dict[str, Dict[str, Any]] = {}
    for path in sorted(glob.glob(str(jobs_dir / "*.json"))):
        try:
            doc = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict) or "job_id" not in doc:
            continue  # an evidence sidecar, not a job doc
        text = str((doc.get("incident") or {}).get("description") or "").strip()
        if not text or not (doc.get("outputs") or {}).get("correlation"):
            continue
        key = text[:400]
        prev = by_incident.get(key)
        if prev is None or _richness(doc) > _richness(prev):
            doc["_path"] = path
            by_incident[key] = doc
    return [by_incident[k] for k in sorted(by_incident)]


def _evidence(doc: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieved rows from the evidence sidecar. Returns {} when the sidecar is missing
    so the caller can drop that run rather than treating it as empty-retrieved.
    """
    path = Path(str(doc.get("_path") or ""))
    sidecar = path.with_suffix("").with_suffix(".evidence.json")
    if not sidecar.is_file():
        sidecar = Path(str(path)[: -len(".json")] + ".evidence.json")
    if not sidecar.is_file():
        return {}
    try:
        blob = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return {}
    logs = (blob or {}).get("logs")
    if not isinstance(logs, dict):
        return {}
    return {
        str(k): [r for r in (v or []) if isinstance(r, dict)]
        for k, v in logs.items()
        if isinstance(v, list)
    }


def _playbook_use_cases(pack) -> Dict[str, str]:
    """``{playbook_id: use_case}`` over the loaded playbooks, first declaration winning.

    Used to detect runs where the recorded playbook no longer resolves to the adjudicating
    procedure (pack edited after the run); those runs are flagged in the header.
    """
    out: Dict[str, str] = {}
    for doc in getattr(pack, "playbook_documents", None) or []:
        meta = (doc or {}).get("metadata", {}) or {}
        pid = str(meta.get("playbook_id", "") or "").strip()
        if pid and pid not in out:
            out[pid] = str(meta.get("use_case", "") or "").strip()
    return out


def _decode(doc: Dict[str, Any]):
    """``(analysis, correlation)`` for one run, through the models the pipeline itself uses."""
    outputs = doc.get("outputs") or {}
    understanding = UnderstandingResult.model_validate(outputs["understanding"])
    correlation = CorrelationResult.model_validate(outputs["correlation"])
    return understanding.analysis, correlation


# --- one run -----------------------------------------------------------------------------


def _measure_run(
    pack, doc: Dict[str, Any], playbooks: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """Replay one run's link assessment and return the per-candidate rows it produced."""
    try:
        analysis, correlation = _decode(doc)
    except (
        Exception
    ) as exc:  # noqa: BLE001 — an undecodable export is not a measurement
        return {"error": f"{type(exc).__name__}: {exc}"}
    logs = _evidence(doc)
    if not logs:
        return {"error": "no evidence sidecar — its rows are not recoverable"}

    brief = correlation.brief
    use_case = str(getattr(brief, "use_case", "") or "")
    ruleset_key = ""
    if use_case:
        try:
            ruleset_key = pack.ruleset_key_for(use_case) or ""
        except Exception:  # noqa: BLE001
            ruleset_key = ""
    if not ruleset_key and use_case and pack.ruleset_spec(use_case):
        ruleset_key = use_case
    if not ruleset_key:
        return {"error": f"no ruleset resolves for use_case {use_case!r}"}

    pack_data = getattr(pack, "pack_data", None)
    findings = assess_links(
        pack,
        analysis,
        logs,
        verdict=correlation.verdict,
        brief=brief,
        ruleset_key=ruleset_key,
        entity_map=None,
        pack_data=pack_data,
        # Passed as None; the export does not carry these. When it does, update here.
        row_caps=None,
        keyed_sources=None,
        unanswered_sources=None,
        requested=list(getattr(analysis, "requested_links", None) or []),
        config={"enabled": True},
    )

    rows: List[Dict[str, Any]] = []
    for finding in findings:
        target = str(finding.target_use_case or "")
        spec = pack.ruleset_spec(target) if target else None
        gate = {"outcome": "no_spec", "detail": ""}
        fired: List[Dict[str, Any]] = []
        evaluated = 0
        evaluated_ids: List[str] = []
        if isinstance(spec, dict):
            # gate FAIL outranks a fired signal, so state alone hides whether one fired;
            # read both through the same functions as assess_links.
            gate = _gate_outcome(
                spec, logs, analysis, None, pack_data, None, None, None
            )
            declared = pack.entry_signals(target)
            fired, evaluated, _ = _fired_signals(declared, logs)
            # base_rate is declared per signal, so each needs its own denominator.
            # "evaluated" means the source was retrieved; a signal whose source wasn't is
            # neither fired nor silent. Re-use _fired_signals to keep denominator consistent.
            for one in declared:
                if _fired_signals([one], logs)[1]:
                    evaluated_ids.append(str(one.get("id") or ""))
        rows.append(
            {
                "target": target,
                "origin_playbook": str(finding.target_playbook_id or ""),
                "state": str(finding.state or ""),
                "direction": str(finding.direction or ""),
                "gate": str(gate.get("outcome") or ""),
                "signals_evaluated": evaluated,
                "evaluated_ids": evaluated_ids,
                "fired": [str(s.get("id") or "") for s in fired],
                "pivot_entity": str(finding.pivot_entity or ""),
                "pivot_count": len(finding.pivot_values or []),
            }
        )
    playbook_id = str(getattr(brief, "playbook_id", "") or "")
    recorded = playbooks.get(playbook_id, "")
    resolves = ""
    if recorded:
        try:
            resolves = pack.ruleset_key_for(recorded) or ""
        except Exception:  # noqa: BLE001
            resolves = ""
    return {
        "job_id": str(doc.get("job_id") or ""),
        "created_at": str(doc.get("created_at") or ""),
        "ruleset_key": ruleset_key,
        "playbook_id": playbook_id,
        # True where the run's own recorded playbook still resolves to the procedure that
        # adjudicated it. False means the pack was edited after the run.
        "playbook_current": bool(resolves) and resolves == ruleset_key,
        "sources_retrieved": len(logs),
        "candidates": rows,
    }


# --- aggregation -------------------------------------------------------------------------


def _verdict(pair: Dict[str, Any], min_corpus: int) -> Tuple[str, str]:
    """``(verdict, why)`` for one pair, against the pre-committed bars."""
    corpus = pair["corpus"]
    if corpus < min_corpus:
        return "STUB", f"corpus {corpus} < {min_corpus}"
    if not pair["reach"]:
        return (
            "UNREACHABLE",
            "the target's subject entity was never in hand — a binding gap, not a rate",
        )
    if not pair["fire"]:
        return "WEAK", "no declared signal ever fired on rows already retrieved"
    rate = pair["fire"] / corpus
    if rate > MAX_FIRE_RATE:
        return (
            "WEAK",
            f"fires on {rate:.0%} of runs, over the {MAX_FIRE_RATE:.0%} bar — "
            "that describes the population, not a link",
        )
    if pair["gate_evaluable_and_fired"] == 0:
        return (
            "WEAK",
            "it fired, but the target's own gate was never evaluable beside it",
        )
    if pair["gate_fail"] == 0 and pair["gate_pass"] == pair["corpus"]:
        # A gate that holds on every run of the corpus can label nothing; 100% agreement
        # against it is arithmetic, not evidence.
        return (
            "WEAK",
            "the target's gate held on every run of the corpus, so it can label nothing — "
            "agreement against it measures the gate, not the signal",
        )
    agreement = pair["agree"] / pair["gate_evaluable_and_fired"]
    if agreement < MIN_AGREEMENT:
        return (
            "WEAK",
            f"the target's own gate contradicted it on {1 - agreement:.0%} of the runs "
            "where it fired",
        )
    return "MEASURED", f"fires on {rate:.0%}, gate agrees on {agreement:.0%}"


def _aggregate(runs: List[Dict[str, Any]], min_corpus: int) -> List[Dict[str, Any]]:
    pairs: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "corpus": 0,
            "reach": 0,
            "fire": 0,
            "agree": 0,
            "gate_evaluable_and_fired": 0,
            "gate_pass": 0,
            "gate_fail": 0,
            "states": defaultdict(int),
            "gates": defaultdict(int),
            "signals": defaultdict(int),
            "direction": "",
        }
    )
    for run in runs:
        source = run["ruleset_key"]
        for row in run["candidates"]:
            pair = pairs[(source, row["target"])]
            pair["corpus"] += 1
            pair["states"][row["state"]] += 1
            pair["gates"][row["gate"]] += 1
            pair["direction"] = pair["direction"] or row["direction"]
            if row["gate"] == "pass":
                pair["gate_pass"] += 1
            elif row["gate"] == "fail":
                pair["gate_fail"] += 1
            if row["state"] != "unreachable":
                pair["reach"] += 1
            if row["fired"]:
                pair["fire"] += 1
                for sid in row["fired"]:
                    pair["signals"][sid] += 1
                if row["gate"] in {"pass", "fail"}:
                    pair["gate_evaluable_and_fired"] += 1
                    if row["gate"] != "fail":
                        pair["agree"] += 1

    out: List[Dict[str, Any]] = []
    for (source, target), pair in pairs.items():
        verdict, why = _verdict(pair, min_corpus)
        out.append(
            {
                "source": source,
                "target": target,
                "direction": pair["direction"],
                "corpus": pair["corpus"],
                "reach": pair["reach"],
                "fire": pair["fire"],
                "agree": pair["agree"],
                "gate_evaluable_and_fired": pair["gate_evaluable_and_fired"],
                "gate_pass": pair["gate_pass"],
                "gate_fail": pair["gate_fail"],
                "states": dict(pair["states"]),
                "gates": dict(pair["gates"]),
                "signals": dict(pair["signals"]),
                "verdict": verdict,
                "why": why,
            }
        )
    out.sort(key=lambda p: (-p["fire"], -p["corpus"], p["source"], p["target"]))
    return out


# --- the flip ----------------------------------------------------------------------------


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _signal_counts(runs: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """``{(target, signal_id): {of, fires_on, sources, fired_sources}}`` per declaration.

    base_rate is declared per signal; its denominator is every run where the signal was asked,
    which differs from the pair corpus (every run where the target was a candidate).
    """
    counts: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"of": 0, "fires_on": 0, "sources": set(), "fired_sources": set()}
    )
    for run in runs:
        source = run["ruleset_key"]
        for row in run["candidates"]:
            fired = set(row.get("fired") or [])
            for sid in row.get("evaluated_ids") or []:
                seen = counts[(str(row["target"]), str(sid))]
                seen["of"] += 1
                seen["sources"].add(source)
                if sid in fired:
                    seen["fires_on"] += 1
                    seen["fired_sources"].add(source)
    return counts


def _reconcile(
    pack, runs: List[Dict[str, Any]], pairs: List[Dict[str, Any]], min_corpus: int
) -> List[Dict[str, Any]]:
    """One row per declared entry signal: what it claims, against what this corpus says.

    A declared base_rate is never re-read by the engine after it is written, so the claim
    must be revocable by the same instrument that supported it. Only a contradiction fails
    (--check exits non-zero); UNCHECKABLE and THIN mean "cannot check here", not "wrong".
    """
    counts = _signal_counts(runs)
    measured_pairs = {
        (p["source"], p["target"]) for p in pairs if p["verdict"] == "MEASURED"
    }
    pair_why = {(p["source"], p["target"]): p["why"] for p in pairs}
    try:
        keys = list(pack.ruleset_keys() or [])
    except (
        Exception
    ):  # noqa: BLE001 — a pack that cannot list its rulesets declares nothing
        keys = []
    out: List[Dict[str, Any]] = []
    for target in sorted(keys):
        for signal in pack.entry_signals(target) or []:
            sid = str(signal.get("id") or "")
            rate = signal.get("base_rate")
            rate = rate if isinstance(rate, dict) else {}
            cleared, declared_corpus = base_rate_measured(rate)
            stub = not rate or str(rate.get("kind", "") or "").strip().lower() == "stub"
            seen = counts.get((target, sid), {})
            of = _int(seen.get("of"))
            fires_on = _int(seen.get("fires_on"))
            declared_fires = _int(rate.get("fires_on"), -1)
            measured_rate = (fires_on / float(of)) if of else None
            declared_rate = (
                (declared_fires / float(declared_corpus))
                if declared_fires >= 0 and declared_corpus > 0
                else None
            )
            eligible_pair = any(
                (src, target) in measured_pairs
                for src in seen.get("fired_sources") or ()
            )

            if stub:
                if not of:
                    state, why = "STUB", "never asked on this corpus"
                elif of < min_corpus:
                    state, why = "STUB", f"asked on {of} incident(s) < {min_corpus}"
                elif measured_rate is not None and measured_rate > MAX_FIRE_RATE:
                    state, why = (
                        "STUB",
                        f"fires on {measured_rate:.0%} of the {of} it was asked on, over the "
                        f"{MAX_FIRE_RATE:.0%} bar",
                    )
                elif not eligible_pair:
                    # A signal is inbound, so its corpus spans every procedure that retrieved
                    # its source; it can clear the corpus and fire-rate bars while no pair clears
                    # the precision bar. Name which bar is missing so a large corpus isn't mistaken
                    # for a fully cleared one.
                    blocked = sorted(
                        {
                            pair_why.get((src, target), "")
                            for src in seen.get("fired_sources") or ()
                        }
                        - {""}
                    )
                    state, why = (
                        "STUB",
                        f"fires on {fires_on} of the {of} it was asked on"
                        + (
                            f" ({measured_rate:.0%})"
                            if measured_rate is not None
                            else ""
                        )
                        + ", so its OWN bars clear — but no pair carrying it does: "
                        + "; ".join(blocked or ["it fired on no pair"]),
                    )
                else:
                    state, why = (
                        "ELIGIBLE",
                        f"fires on {fires_on} of {of}, and its pair clears every bar",
                    )
            elif not cleared:
                state, why = (
                    "INERT",
                    f"declared over {declared_corpus} incident(s) — under the "
                    f"{min_corpus} the score reads a rate from, so it adds no confidence",
                )
            elif not of:
                state, why = (
                    "UNCHECKABLE",
                    "the signal was never asked on this corpus, so nothing here bears on it",
                )
            elif of < min_corpus:
                state, why = (
                    "THIN",
                    f"asked on {of} incident(s) here — too few to contradict a declaration",
                )
            elif measured_rate is not None and measured_rate > MAX_FIRE_RATE:
                state, why = (
                    "CONTRADICTED",
                    f"declared {declared_fires} of {declared_corpus}"
                    + (f" ({declared_rate:.0%})" if declared_rate is not None else "")
                    + f", measured {fires_on} of {of} ({measured_rate:.0%}) — over the "
                    f"{MAX_FIRE_RATE:.0%} bar, so it adds confidence off a population",
                )
            else:
                state, why = (
                    "AGREES",
                    f"declared {declared_fires} of {declared_corpus}"
                    + (f" ({declared_rate:.0%})" if declared_rate is not None else "")
                    + f", measured {fires_on} of {of} ({measured_rate:.0%})",
                )
            out.append(
                {
                    "target": target,
                    "signal": sid,
                    "declared": dict(rate),
                    "declared_counts": bool(cleared),
                    "measured_of": of,
                    "measured_fires_on": fires_on,
                    "sources": sorted(str(s) for s in seen.get("sources") or ()),
                    "state": state,
                    "why": why,
                }
            )
    out.sort(key=lambda r: (r["state"], r["target"], r["signal"]))
    return out


# --- report ------------------------------------------------------------------------------


def _print(
    runs: List[Dict[str, Any]],
    pairs: List[Dict[str, Any]],
    decls: List[Dict[str, Any]],
    errors,
    min_corpus,
):
    print("Cross-procedure link signals, measured over the local job history")
    print("=" * 96)
    print(
        f"{len(runs)} distinct incident(s) replayed (deduped on alert text, most complete run "
        f"of each); {len(errors)} skipped."
    )
    print(
        "A REPLAY IS WEAKER THAN THE RUN: row_caps, keyed_sources and unanswered_sources are "
        "not exported,"
    )
    print(
        "so a truncated source reads as complete and an unanswered one as empty — the "
        "agreement column is biased UPWARD."
    )
    print(
        f"Bars: corpus >= {min_corpus}, fire rate <= {MAX_FIRE_RATE:.0%}, "
        f"gate agreement >= {MIN_AGREEMENT:.0%}."
    )
    print()
    by_source: Dict[str, int] = defaultdict(int)
    for run in runs:
        by_source[run["ruleset_key"]] += 1
    print(
        "Runs per adjudicating procedure: "
        + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items()))
    )
    stale = [r for r in runs if not r["playbook_current"]]
    if stale:
        print(
            f"{len(stale)} of {len(runs)} run(s) recorded a playbook that no longer resolves to "
            "the procedure that adjudicated them,"
        )
        print(
            "so their candidate list is today's declaration and not the one that ran: "
            + ", ".join(
                sorted({f"{r['playbook_id']}->{r['ruleset_key']}" for r in stale})
            )
        )
    print()
    header = (
        f"{'source -> target':<46} {'dir':<11} {'corpus':>6} {'reach':>6} "
        f"{'fire':>5} {'agree':>7} {'gate P/F':>9}  verdict"
    )
    print(header)
    print("-" * len(header))
    for pair in pairs:
        agree = (
            f"{pair['agree']}/{pair['gate_evaluable_and_fired']}"
            if pair["gate_evaluable_and_fired"]
            else "-"
        )
        # gate_pass/gate_fail beside agree makes a gate that never fails visible.
        print(
            f"{pair['source'] + ' -> ' + pair['target']:<46} "
            f"{pair['direction'] or '-':<11} {pair['corpus']:>6} {pair['reach']:>6} "
            f"{pair['fire']:>5} {agree:>7} "
            f"{str(pair['gate_pass']) + '/' + str(pair['gate_fail']):>9}  "
            f"{pair['verdict']}: {pair['why']}"
        )
    print()
    verdicts: Dict[str, int] = defaultdict(int)
    for pair in pairs:
        verdicts[pair["verdict"]] += 1
    print("Pairs: " + ", ".join(f"{k}={v}" for k, v in sorted(verdicts.items())))
    print()
    print("Declarations, against this corpus")
    print("-" * 96)
    if not decls:
        print("  no ruleset declares an entry signal — there is nothing to measure.")
    states: Dict[str, int] = defaultdict(int)
    for row in decls:
        states[row["state"]] += 1
    for row in decls:
        print(f"  [{row['state']:<13}] {row['target']}/{row['signal']}: {row['why']}")
    if decls:
        print()
        print(
            "Declarations: " + ", ".join(f"{k}={v}" for k, v in sorted(states.items()))
        )
    eligible = [r for r in decls if r["state"] == "ELIGIBLE"]
    if eligible:
        print()
        print(
            "ELIGIBLE — every bar cleared. Transcribe into the signal's own declaration, which "
            "is what lets the"
        )
        print(
            "confidence score credit the signal with discriminating; nothing is applied from "
            "here, because a pack edit is an authored act:"
        )
        for row in eligible:
            print(
                f"  {row['target']}/{row['signal']}: base_rate: {{fires_on: "
                f"{row['measured_fires_on']}, of: {row['measured_of']}, corpus: jobs}}"
                f"   (asked on runs of: {', '.join(row['sources']) or '-'})"
            )
    else:
        print()
        print(
            "NO declaration clears every bar on this corpus. That is a measurement, not a "
            "refusal of anything:"
        )
        print(
            "every declaration keeps `base_rate: {kind: stub}` and escalates on its target's own "
            "scope gate exactly as"
        )
        print(
            "before — what the thin corpus withholds is the score's discrimination term, never "
            "permission to escalate."
        )
    bad = [r for r in decls if r["state"] == "CONTRADICTED"]
    if bad:
        print()
        print(
            f"CONTRADICTED — {len(bad)} declaration(s) claim a discrimination this corpus no "
            "longer supports."
        )
        print(
            "`--check` exits non-zero on these: the number was produced by a run of this script "
            "and is withdrawn by one."
        )
    if errors:
        print()
        print("Skipped runs:")
        for job_id, why in errors:
            print(f"  {job_id}: {why}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pack", default=None, help="pack directory (default: discover)"
    )
    parser.add_argument("--jobs", default=None, help="job history directory")
    parser.add_argument("--min-corpus", type=int, default=MIN_CORPUS)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "exit non-zero if a declared base_rate claims a discrimination this corpus "
            "contradicts (a gate for CI; reporting is unchanged)"
        ),
    )
    args = parser.parse_args()

    # The link assessment logs an INFO line per run and a warning per unevaluable sibling gate;
    # over a corpus that is noise on top of the table.
    logging.basicConfig(level=logging.ERROR)

    pack = _load_pack(args.pack)
    playbooks = _playbook_use_cases(pack)
    jobs_dir = Path(args.jobs) if args.jobs else REPO_ROOT / "jobs"
    docs = _runs(jobs_dir)
    if not docs:
        print(f"no usable job docs under {jobs_dir}", file=sys.stderr)
        return 1

    runs: List[Dict[str, Any]] = []
    errors: List[Tuple[str, str]] = []
    for doc in docs:
        measured = _measure_run(pack, doc, playbooks)
        if measured is None:
            continue
        if measured.get("error"):
            errors.append((str(doc.get("job_id") or "")[:8], measured["error"]))
            continue
        runs.append(measured)

    pairs = _aggregate(runs, args.min_corpus)
    decls = _reconcile(pack, runs, pairs, args.min_corpus)
    if args.json:
        print(
            json.dumps(
                {
                    "runs": len(runs),
                    "skipped": [{"job": j, "why": w} for j, w in errors],
                    "bars": {
                        "min_corpus": args.min_corpus,
                        "max_fire_rate": MAX_FIRE_RATE,
                        "min_agreement": MIN_AGREEMENT,
                    },
                    "pairs": pairs,
                    "declarations": decls,
                },
                indent=2,
            )
        )
    else:
        _print(runs, pairs, decls, errors, args.min_corpus)
    # --check exits non-zero only on CONTRADICTED so the plain reporting run always exits 0.
    if args.check and any(r["state"] == "CONTRADICTED" for r in decls):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
