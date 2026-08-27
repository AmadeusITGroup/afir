"""Audit every saved run, grouped by use case. Read-only; prints to stdout, writes nothing.

A test says the code still does what it did; it cannot say whether the shipped use cases still land
on the disposition the human investigation reached, since that lives in each use case's `VERDICT.md`
while the run that produced it lives in `jobs/*.json`.

`jobs/` is a re-run log and not a population: the same incident appears many times as the pack was
refined, and a cancelled run counts twice if you count files. Runs are grouped by the incident
DESCRIPTION, the only stable identity a job doc carries (`incident.id` is minted per submission),
and within a group the latest completed run wins, or the latest of any status, marked.

Accessors are shared with `grade_live_run.py`, two hand-written reads of `outputs` having produced
two different answers once: `use_case` / `playbook_id` from `correlation.brief`, the disposition
from `correlation.verdict`, nowhere else.

    .venv/bin/python scripts/audit_use_case_verdicts.py            # deduped, grouped by use case
    .venv/bin/python scripts/audit_use_case_verdicts.py --all      # every run, no dedup
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

JOBS = Path("jobs")


def job_docs():
    """Every job document, evidence sidecars excluded (they carry `logs` and nothing else)."""
    for path in sorted(JOBS.glob("*.json")):
        if path.name.endswith(".evidence.json"):
            continue
        try:
            yield path, json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue


def read(path, doc):
    out = doc.get("outputs") or {}
    corr = out.get("correlation") or {}
    verdict = corr.get("verdict") or {}
    brief = corr.get("brief") or {}
    incident = doc.get("incident") or {}
    text = (incident.get("description") or "").strip()
    subjects = verdict.get("subjects") or []
    return {
        "job_id": doc.get("job_id") or path.stem,
        "status": doc.get("status"),
        "updated": doc.get("updated_at") or "",
        "use_case": brief.get("use_case") or "(none — no verdict)",
        "playbook": brief.get("playbook_id") or "",
        "summary": (verdict.get("summary") or "").strip(),
        "scheme": verdict.get("label_scheme") or "",
        "degraded": verdict.get("degraded"),
        "classes": [s.get("verdict_class") for s in subjects],
        "n_subjects": len(subjects),
        "scope": (brief.get("scope_status") or "")[:60],
        "incident_key": hashlib.sha1(text.encode()).hexdigest()[:8] if text else "-",
        "incident_head": " ".join(text.split())[:70],
    }


def main():
    show_all = "--all" in sys.argv[1:]
    runs = [read(p, d) for p, d in job_docs()]
    print(f"job documents: {len(runs)}")

    groups = defaultdict(list)
    for run in runs:
        groups[(run["use_case"], run["incident_key"])].append(run)

    kept = []
    for (_use_case, _key), members in groups.items():
        members.sort(key=lambda r: (r["status"] == "completed", r["updated"]))
        kept.append((members[-1], len(members)))
    print(f"distinct (use case, incident) pairs: {len(kept)}")

    by_use_case = defaultdict(list)
    for run, reruns in kept:
        by_use_case[run["use_case"]].append((run, reruns))

    for use_case in sorted(by_use_case):
        entries = by_use_case[use_case]
        print("=" * 78)
        print(f"### {use_case}   —   {len(entries)} distinct incident(s)")
        for run, reruns in sorted(entries, key=lambda e: e[0]["updated"]):
            flag = "" if run["status"] == "completed" else f"  <<< {run['status']}"
            print(f"  {run['job_id'][:8]}  {run['updated'][:19]}  reruns={reruns}{flag}")
            print(f"      incident : {run['incident_head']}")
            print(f"      playbook : {run['playbook']}   scheme={run['scheme']}")
            print(f"      verdict  : {run['summary'][:120] or '(none)'}")
            print(
                f"      subjects : {run['n_subjects']}  classes={run['classes']}"
                f"  degraded={run['degraded']}"
            )
            if run["scope"]:
                print(f"      scope    : {run['scope']}")
            if show_all:
                for other in sorted(
                    groups[(use_case, run["incident_key"])],
                    key=lambda r: r["updated"],
                ):
                    print(
                        f"        · {other['job_id'][:8]} {other['updated'][:19]} "
                        f"{other['status']:10s} {other['summary'][:70]}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
