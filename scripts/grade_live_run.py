"""Grade one live run on what is gradeable without an answer key. Read-only; writes nothing.

Every run must be measured by the same code, or a difference between two use cases cannot be told
from a difference between two readings of the output: one hand-written grading reported 13 of 14
sections as "2 chars" by reading `len(list)` on a nested `content`. None of the five checks needs to
know the right verdict. Disposition is graded by hand into each use case's VERDICT.md, not here.

  G1 CONCEPT WINDOW      Every `concept_refs` snippet's length and position. A concept doc reaches
                         the prompt as its first N chars only, so a snippet at the cut length is the
                         load-bearing measurement and a snippet of 0 means a declared concept
                         reached the brief as nothing.
  G2 SELF-CONTRADICTION  The report is narrated FROM a deterministic verdict and may not deny it.
                         Two shapes: the verdict's own status must appear, and an indicator heading
                         whose text negates itself.
  G3 SILENT EMPTINESS    A source that did not ANSWER must not read as one that answered with
                         nothing, and an unfilled `'<...>'` predicate is a non-answer too. Both per
                         pass, beside the row counts and the truncation markers.
  G4 DEPENDENCIES        `required_source_not_queried` and the undeliverable/unscopable lists.
  G5 STAGE HEALTH        Every stage's score and reasons, plus `scored`: an unscored stage is not a
                         healthy one.

Run:  .venv/bin/python scripts/grade_live_run.py <job_id> [<job_id> ...]
      .venv/bin/python scripts/grade_live_run.py --state    # every job in live_runs_four.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

JOBS = Path("jobs")
STATE = Path("scripts/out/live_runs_four.json")

# An indicator heading states its finding; an exclusion's label states its requirement. A heading
# that affirms a positive count and then negates itself in the same breath is the polarity trap.
_POLARITY = re.compile(
    r"(positive[^\n]{0,40}indicator[^\n]{0,200}?\b(?:NOT|no longer|was not|did not)\b[^\n]{0,80})",
    re.IGNORECASE,
)
_PLACEHOLDER = re.compile(r"'<[^>]*>'|\"<[^>]*>\"|<[A-Z_]{3,}>")


def flatten_text(node, acc):
    """`content` is a nested list of strings and subsections — read it recursively or lie."""
    if isinstance(node, str):
        acc.append(node)
    elif isinstance(node, list):
        for item in node:
            flatten_text(item, acc)
    elif isinstance(node, dict):
        for key in ("section_title", "title", "heading", "content", "body", "text"):
            if key in node:
                flatten_text(node[key], acc)
    return acc


def report_text(raw):
    """The report may be a JSON string, a dict, or already prose. Return (text, n_sections).

    On this pack `outputs.report` is a JSON **string** whose payload is a LIST of
    `{section_title, content}` — and `content` is itself a nested list of strings and subsections.
    Counting `len()` on the wrong level is how a 57,982-char report first read as 14 sections of
    "2 chars" each, so the section count comes from the list and the text from a recursive walk.
    """
    doc = raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                doc = json.loads(stripped)
            except json.JSONDecodeError:
                return raw, 0
        else:
            return raw, 0
    if isinstance(doc, dict):
        sections = doc.get("sections") or []
        return "\n".join(flatten_text(doc, [])), len(sections)
    if isinstance(doc, list):
        n = sum(
            1
            for s in doc
            if isinstance(s, dict) and ("section_title" in s or "title" in s)
        )
        return "\n".join(flatten_text(doc, [])), n
    return "\n".join(flatten_text(doc, [])), 0


def load(job_id):
    path = JOBS / f"{job_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_evidence(job_id):
    path = JOBS / f"{job_id}.evidence.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    return (doc or {}).get("logs") or {}


def grade(job_id):
    doc = load(job_id)
    if doc is None:
        print(f"  no job document for {job_id} — not finished, or a different store")
        return
    out = doc.get("outputs") or {}
    corr = out.get("correlation") or {}
    verdict = corr.get("verdict") or {}
    brief = corr.get("brief") or {}

    stage_keys = doc.get("stage_keys") or []
    # `current_pass` reads 1 on a finished three-pass run, so the pass count is COUNTED off the
    # stage keys — a pass is a `log_retrieval` that ran, and `pass_key` spells pass 1 bare.
    n_passes = sum(1 for k in stage_keys if k.split("#")[0] == "log_retrieval")
    print(f"  status            {doc.get('status')}   stages {len(stage_keys)}")
    print(f"  passes            {n_passes} (log_retrieval stages)")
    print(f"  use_case/playbook {brief.get('use_case')} / {brief.get('playbook_id')}")
    summary = (verdict.get("summary") or "").strip()
    print(f"  verdict           {summary[:150] or '(none)'}")
    print(f"  degraded          {verdict.get('degraded')}")

    # ---- G1 concept window ------------------------------------------------------------------
    refs = brief.get("concept_refs") or []
    print(f"  G1 concepts       {len(refs)}")
    for i, ref in enumerate(refs):
        if isinstance(ref, dict):
            title = ref.get("title") or ref.get("id") or ref.get("name") or "?"
            body = ref.get("snippet") or ref.get("text") or ref.get("body") or ""
        else:
            title, body = str(ref)[:40], ""
        flag = "  <<< EMPTY" if not body else ""
        print(f"       {i}. {len(body):5d} ch  {str(title)[:52]}{flag}")

    for key in (
        "decisive_fails",
        "decisive_unknowns",
        "decisive_indicators",
        "explanatory_fails",
    ):
        val = brief.get(key)
        mark = "absent" if val is None else f"{len(val)}"
        print(f"  brief.{key:22s} {mark}")

    # ---- G2 self-contradiction -------------------------------------------------------------
    text, n_sections = report_text(out.get("report"))
    print(f"  G2 report         {len(text)} chars, {n_sections} sections")
    if summary:
        head = summary.split("—")[0].split(".")[0].strip()
        token = head[:48]
        print(
            f"     verdict phrase in report: {bool(token) and token.lower() in text.lower()}"
            f"  ({token!r})"
        )
    hits = _POLARITY.findall(text)
    print(f"     polarity trap: {len(hits)} hit(s)")
    for h in hits[:3]:
        print(f"       ! {' '.join(h.split())[:150]}")

    # ---- G3 silent emptiness ---------------------------------------------------------------
    # Retrieved rows are NOT in `outputs` — they live in the `<job>.evidence.json` sidecar, which is
    # FLATTENED (`an-export-shape-is-not-the-run-shape`). Row COUNTS survive that flattening, which
    # is all G3 reads; nothing here may be used to re-derive a binding or an entity map.
    logs = load_evidence(job_id)
    queries = out.get("queries") or []
    print(
        f"  G3 queries        {len(queries)}  (rows from the flattened evidence sidecar)"
    )
    ph = [q for q in queries if _PLACEHOLDER.search(json.dumps(q))]
    if ph:
        print(f"     UNFILLED placeholder in {len(ph)} query/queries:")
        for q in ph[:4]:
            src = q.get("source") if isinstance(q, dict) else "?"
            found = _PLACEHOLDER.search(json.dumps(q))
            print(f"       ! {src}: {found.group(0)[:60]}")
    if isinstance(logs, dict) and logs:
        zero = [s for s, rows in logs.items() if isinstance(rows, list) and not rows]
        print(f"     sources answered: {len(logs)}   empty: {len(zero)}")
        for src, rows in sorted(logs.items(), key=lambda kv: -len(kv[1] or [])):
            n = len(rows or [])
            # A count that equals a round cap is a FLOOR, not a total — flagged, never assumed.
            cap = (
                "  (== a round number: check the cap)"
                if n in (100, 200, 500, 1000, 5000)
                else ""
            )
            print(f"       {n:6d}  {src}{cap}")
    for key in ("unanswered_sources", "unanswered", "keyed_sources", "row_caps"):
        if key in out:
            print(f"     {key}: {json.dumps(out[key])[:220]}")

    # ---- G4 dependencies -------------------------------------------------------------------
    health = doc.get("stage_health") or {}
    dep = []
    for stage, h in health.items():
        for reason in (h or {}).get("reasons") or []:
            if "source" in (reason.get("code") or ""):
                dep.append((stage, reason.get("code"), reason.get("detail")))
    print(f"  G4 source findings {len(dep)}")
    for stage, code, detail in dep:
        print(f"       {stage:20s} {code:28s} {str(detail)[:90]}")

    # ---- G5 stage health -------------------------------------------------------------------
    unscored = [s for s, h in health.items() if not (h or {}).get("scored")]
    worst = sorted(
        ((h or {}).get("score", 1.0), s)
        for s, h in health.items()
        if (h or {}).get("scored")
    )[:3]
    print(f"  G5 health         unscored={unscored or 'none'}  worst={worst}")
    for _score, stage in worst:
        for reason in (health.get(stage) or {}).get("reasons") or []:
            print(
                f"       {stage:20s} {reason.get('code'):28s} {str(reason.get('detail'))[:80]}"
            )


def main():
    args = sys.argv[1:]
    if not args or args[0] == "--state":
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        pairs = [(k, (v or {}).get("job_id")) for k, v in state.items()]
    else:
        pairs = [(a[:8], a) for a in args]
    for label, job_id in pairs:
        print("=" * 78)
        print(f"### {label}  job {job_id}")
        if not job_id:
            print("  no job id recorded")
            continue
        grade(job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
