"""Does the planner select the procedure's sources without being forced to?

The force-add mechanism (_add_missing_required) was removed. This script verifies that the
planner already picks the dependencies unaided. One LLM call per job (query-generation
stage only); understanding is replayed from the persisted job document.

Per job it prints:
  PICKED       the planner's own selection, after follow-up deferral
  DECLARED     the adjudicating ruleset's sources: map (the only hard dependencies of this run)
  UNSCOPABLE   declared, but the incident names no entity type the source is bound to;
               the planner declining is correct (an unscoped query returns nothing useful)
  MISSING      DECLARED - PICKED - deferred - undeliverable - UNSCOPABLE  <<< must be empty
  CROSS        what the old force-add would have injected from non-adjudicating procedures

MISSING is the gate. Fix in the pack's selection_guidance / not_answered_by, then re-measure.

Read-only: opens no writer, mutates no job document, touches no config/.
--jobs is required; the curated validation set belongs in the pack's ledger.
The adjudicating procedure is read from each job's stored verdict, not annotated here.

Run: set -a; source .afir_env; set +a
     python -u scripts/validate_source_selection.py --jobs 88194ae1,6e4bf9a6,...
     python -u scripts/validate_source_selection.py --jobs 88194ae1 --repeat 3   # stability
     python -u scripts/validate_source_selection.py --jobs ... --pack <name>
"""

import argparse
import asyncio
import glob
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def _job_path(prefix):
    hits = [
        p
        for p in glob.glob(str(REPO_ROOT / "jobs" / f"{prefix}*.json"))
        if not p.endswith(".prev") and ".evidence." not in p
    ]
    return hits[0] if hits else ""


def _live_scheme(doc):
    """Which procedure adjudicated the live run, from the job's own stored verdict."""
    c = (doc.get("outputs") or {}).get("correlation")
    if isinstance(c, str):
        try:
            c = json.loads(c)
        except Exception:  # noqa: BLE001
            return ""
    return str((((c or {}).get("verdict") or {}) or {}).get("label_scheme") or "")


def _row_counts(prefix):
    """What each source actually returned on the live run, from the evidence sidecar."""
    hits = glob.glob(str(REPO_ROOT / "jobs" / f"{prefix}*.evidence.json"))
    if not hits:
        return {}
    try:
        logs = (json.load(open(hits[0])) or {}).get("logs") or {}
    except Exception:  # noqa: BLE001
        return {}
    return {k: len(v or []) for k, v in logs.items()}


def _declared(pack, key):
    spec = ((getattr(pack, "rulesets", None) or {}).get("verdicts") or {}).get(key) or {}
    return {str(v) for v in (spec.get("sources") or {}).values() if str(v or "")}


def _unscopable(pack, names, analysis):
    """Of ``names``, those bound to no entity type the incident named.

    Uses the source's entities: list. time_window is excluded: every source and incident
    carries it, so including it would make the intersection non-empty for everything.
    """
    present = {
        str(getattr(e, "type", "") or "").lower()
        for e in (getattr(analysis, "extracted_entities", None) or [])
    }
    present.discard("")
    present.discard("time_window")
    out = set()
    for n in names:
        src = pack.source(n) if hasattr(pack, "source") else None
        bound = {
            str(x).lower() for x in (getattr(src, "entities", None) or []) if str(x or "")
        }
        bound.discard("time_window")
        if bound and not (bound & present):
            out.add(n)
    return out


def _all_declared(pack):
    out = {}
    for k, spec in (
        (getattr(pack, "rulesets", None) or {}).get("verdicts") or {}
    ).items():
        if isinstance(spec, dict):
            out[k] = {
                str(v) for v in (spec.get("sources") or {}).values() if str(v or "")
            }
    return out


async def one(gen, pack, prefix, offered, repeat):
    from models.pydantic_models import UnderstandingResult

    path = _job_path(prefix)
    if not path:
        print(f"\n### {prefix}  SKIPPED — no persisted job document", flush=True)
        return None
    doc = json.load(open(path))
    stored = (doc.get("outputs") or {}).get("understanding")
    if isinstance(stored, str):
        stored = json.loads(stored)
    if not stored:
        print(f"\n### {prefix}  SKIPPED — job stored no understanding", flush=True)
        return None
    und = UnderstandingResult.model_validate(stored)
    analysis = und.analysis
    uc = _live_scheme(doc)
    note = " ".join(
        str((doc.get("incident") or {}).get("description") or "").split()
    )[:88]

    key = gen._adjudicating_ruleset_key(analysis)
    declared = _declared(pack, key)
    deferred = gen._follow_up_targets(analysis)
    unscopable = _unscopable(pack, declared, analysis)
    counts = _row_counts(prefix)

    print(f"\n{'=' * 96}", flush=True)
    print(f"### {prefix}  live={uc or '(no stored verdict)'} — {note}", flush=True)
    print(f"{'=' * 96}", flush=True)
    ents = sorted(
        {str(getattr(e, "type", "") or "") for e in (analysis.extracted_entities or [])}
    )
    print(f"  entity types      : {', '.join(ents) or '(none)'}", flush=True)
    print(
        f"  adjudicating      : {key or '(unresolved)'}"
        f"{'   <<< DISAGREES with the live run' if uc and key != uc else ''}",
        flush=True,
    )
    print(f"  declared by it    : {', '.join(sorted(declared)) or '(none)'}", flush=True)
    if deferred:
        print(f"  follow-up target  : {', '.join(sorted(deferred))}", flush=True)
    if unscopable:
        print(
            f"  UNSCOPABLE        : {', '.join(sorted(unscopable))}"
            "  — bound to no entity type this incident named; the CHECK is the loss,"
            " not the query (ENGINE-GAP)",
            flush=True,
        )

    # generate() returns the planner's own selection. _required_sources is read separately
    # (it is pure) to show what the old force-add would have injected.
    rounds = []
    for i in range(repeat):
        required, undel = gen._required_sources(offered, analysis)
        queries = await gen.generate(und)

        picked = {q.target_log_source for q in queries}
        undel = set(undel)
        missing = declared - picked - deferred - undel - unscopable
        rounds.append((picked, missing, undel, set(required)))

        tag = f"  run {i + 1}/{repeat}" if repeat > 1 else "  planner"
        print(f"{tag}  PICKED ({len(picked)}): {', '.join(sorted(picked))}", flush=True)
        if undel:
            print(
                f"           UNDELIVERABLE (no retriever): {', '.join(sorted(undel))}",
                flush=True,
            )
        if missing:
            print(
                f"           MISSING  <<< SELECTION DEFECT: {', '.join(sorted(missing))}",
                flush=True,
            )
        else:
            print(
                "           MISSING  : none — every hard dependency chosen unaided",
                flush=True,
            )

    # What the removed mechanism would have injected, and what it cost live.
    would_force = rounds[0][3] - rounds[0][0]
    cross = {n for n in would_force if n not in declared}
    if would_force:
        parts = []
        for n in sorted(would_force):
            c = counts.get(n)
            owners = [k for k, v in _all_declared(pack).items() if n in v and k != key]
            parts.append(
                f"{n}({'no rows recorded' if c is None else f'{c} rows'}"
                f"{'; owner ' + '/'.join(sorted(owners)) if owners else ''})"
            )
        print(f"  old force-add would inject: {', '.join(parts)}", flush=True)
    if cross:
        print(
            f"  of which CROSS-PROCEDURE ({len(cross)}): {', '.join(sorted(cross))}",
            flush=True,
        )
    return dict(
        uc=uc,
        prefix=prefix,
        key=key,
        rounds=rounds,
        cross=cross,
        unscopable=unscopable,
    )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jobs",
        default="",
        help="REQUIRED: comma-separated job id prefixes; the curated set lives in the "
        "pack's ledger, not in this script",
    )
    ap.add_argument("--repeat", type=int, default=1, help="runs per job (stability)")
    ap.add_argument(
        "--pack", default="", help="pack name (default: knowledge.pack_dir from the config)"
    )
    args = ap.parse_args()

    wanted = [w.strip() for w in args.jobs.split(",") if w.strip()]
    if not wanted:
        ap.error(
            "--jobs is required. Which incidents are worth measuring is pack knowledge: take "
            "the handles from the pack's own ledger, where the reason each was chosen (and the "
            "synthetic ones excluded) is written down."
        )

    from knowledge.pack import load_knowledge_pack
    from api_call_generator import ApiCallGenerator
    from log_retrieval import LogRetrievalEngine
    from main import load_config
    from utils.databricks_auth import try_build_auth
    from utils.llm_client import LLMClient
    from utils.paths import config_path, knowledge_pack_dir

    mc = load_config(config_path("main_config.yaml"))
    lc = load_config(config_path("llm_config.yaml"))
    auth = try_build_auth()
    llm = LLMClient(lc, auth=auth if auth else None)
    pack_name = args.pack or str(
        (mc.get("knowledge") or {}).get("pack_dir") or ""
    ).strip()
    if not pack_name:
        ap.error("no pack: pass --pack, or set knowledge.pack_dir in main_config.yaml")
    pack = load_knowledge_pack(knowledge_pack_dir(pack_name))

    # The offered set must be the one the live run offered, or "the planner did not pick it"
    # is indistinguishable from "it was never on the menu". Built the same way `main()` does.
    eng = LogRetrievalEngine(mc["log_sources"], llm, auth=auth, knowledge_pack=pack)
    offered = list(eng.retrievers.keys())
    print(f"offered sources ({len(offered)}): {', '.join(sorted(offered))}", flush=True)

    gen = ApiCallGenerator(
        mc["log_sources"], llm, knowledge_pack=pack, available_sources=offered
    )

    results = []
    for prefix in wanted:
        try:
            r = await one(gen, pack, prefix, offered, args.repeat)
        except Exception as exc:  # noqa: BLE001 — one job must not stop the sweep
            print(f"\n### {prefix}  EXC {type(exc).__name__}: {exc}", flush=True)
            r = None
        if r:
            results.append(r)

    print(f"\n{'=' * 96}\n### SUMMARY — MISSING is the gate\n{'=' * 96}", flush=True)
    bad = 0
    for r in results:
        miss = set().union(*[m for _, m, _, _ in r["rounds"]]) if r["rounds"] else set()
        flag = "FAIL" if miss else "ok  "
        bad += 1 if miss else 0
        print(
            f"  {flag}  {r['uc'] or '(unknown)':<20} {r['prefix']}"
            f"  adjudicating={r['key'] or '?':<20}"
            f" missing={', '.join(sorted(miss)) or 'none':<48}"
            f" cross_saved={len(r['cross'])}",
            flush=True,
        )
    print(
        f"\n  {len(results) - bad}/{len(results)} job(s) select every hard dependency unaided."
        + (
            "  The force-add is not load-bearing on this set."
            if not bad
            else "  Fix the pack guidance for the FAIL rows, then re-measure."
        ),
        flush=True,
    )
    await eng.close()


if __name__ == "__main__":
    asyncio.run(main())
