# AFIR HTTP API

Everything the control-panel UI does is driven by this HTTP API — the UI is just
one client of it. This document shows how to launch an investigation, watch it
live, control it, and read results using `curl` or Python, with **full parity**
with the UI.

The server runs on one port (`incident_input.port`, default `5000`; a Databricks
App overrides it with `$DATABRICKS_APP_PORT`). Examples below use
`BASE=http://127.0.0.1:8137`.

## Two ways to run an incident

| | Controllable **job** | Classic **blocking** call |
|---|---|---|
| Endpoint | `POST /api/v1/jobs` | `POST /api/v1/incidents/freetext` (+ `/incidents`, `/ir`) |
| Returns | job id immediately (non-blocking) | the full report in one response |
| Progress | live via SSE / polling | none, unless `?verbose=1` |
| Control | pause/resume/step/cancel/retry + approval gates | none |
| Use when | you want visibility + control (what the UI uses) | you just want the report from a script |
| In a Databricks App | the only way | **`501`** — the ingress cuts a request long before a run finishes (§ Classic blocking endpoints) |

> **No authentication.** Nothing in this API is authenticated. That was already true
> of the pipeline endpoints; approval gates make it consequential, because an
> unauthenticated caller can approve a stage or inject prompt guidance
> (`POST /jobs/{id}/gate`), and the configuration endpoints (§10) make it more so — an
> unauthenticated caller can repoint a backend URL. Literal secret values never leave
> the server (they are redacted in both the structured read and the raw text), which is
> the one guarantee that does not depend on the ingress. Everything else does:
> terminate TLS and require auth at the ingress (a Databricks App does this for you)
> before exposing this port anywhere.

---

## Machine-readable specification

`docs/openapi.yaml` is the OpenAPI 3.1 specification for this surface. The running server exposes the same document at:

- `GET /openapi.json` — the specification as JSON
- `GET /docs` — a rendered, interactive reference

Both routes are unauthenticated and carry the same access caveat as every other endpoint on this API.

## Control panel

`GET /` returns the single-page control panel as `text/html`. The page bundles its own CSS and requires no external asset — it functions offline, including in a Databricks App. Every action the UI takes is driven by the endpoints in this document.

---

## 1. Launch a controllable job

```bash
curl -s -X POST "$BASE/api/v1/jobs" \
  -H 'Content-Type: application/json' \
  -d '{"description": "Multiple failed logins for user 4821, then a large transfer on 2024-08-26", "mode": "auto"}'
# -> {"job_id": "39f4…", "incident_id": "3e31…", "status": "running"}
```

`status` is `running` or **`queued`**: a run costs ~38 minutes at the median and fans out
~19 retrievers against one shared LLM semaphore, so `jobs.max_concurrent_jobs` (default 2)
bounds how many run stages at once and the rest wait in a FIFO backlog. A queued job is a
real job — every endpoint below works on it, and the `job_status` event carries its
`queue_position`. Past `jobs.max_queued_jobs` (default 256) the submission is refused with
**429** rather than accepted, because a job id for a run that may never start reads exactly
like a run that is merely slow:

```json
{"error": "the run queue is full: 256 submission(s) waiting against a limit of 256",
 "queued": 256, "max_queued": 256,
 "retry": "poll GET /api/v1/jobs and resubmit once the backlog drains"}
```

The width bounds *submitted* runs only. A resume, a stage retry and a link lane's child run
all start outside it — each is a decision already taken, and backpressuring one of those is
how a control surface stops being trusted.

`mode` picks how much human involvement the run requires:

| `mode` | behaviour |
|---|---|
| `auto` (default) | every stage runs back-to-back; no gate ever opens |
| `semi_auto` | a stage stops for approval **only** when its deterministic health score is below its threshold |
| `supervised` | every gateable stage stops for approval |
| `step` | pauses **before** each stage; advances on a `step` control action |

`step` is deliberately not a gate: it asks permission to *start* a stage, whereas a
gate is a review of what a stage *produced*. An unknown value falls back to `auto`
with a server-side warning — so a typo un-gates the run. Check `run_mode` in the
snapshot if that matters to you.

## 2. Watch it live — Server-Sent Events (what the UI uses)

```bash
curl -N "$BASE/api/v1/jobs/<job_id>/events"
```

Each line is `data: {json}`. Event `type`s:

| type | meaning | notable `data` |
|---|---|---|
| `job_status` | job-level state change | `status` = running/paused/awaiting_approval/completed/cancelled/stage_failed |
| `stage_started` | a stage began | `started_at` (epoch seconds) |
| `stage_output` | **what the stage produced** (compact summary) | `summary` (see below), `health` (see §7) |
| `stage_completed` | a stage finished | `duration_ms` |
| `stage_failed` | a stage errored | `duration_ms`, `message` |
| `source_progress` | per-source log-retrieval progress | `source`, `status` (completed/timeout/failed) |
| `gate_opened` | **a stage is waiting on you** | the whole gate record (§7) |
| `gate_resolved` | the gate was answered | `action`, `actor`, `reason_code`, `guidance`, `restart_from` |
| `gate_timeout` | nobody answered in time | `action` (`hold`/`proceed`/`abort`), `waited_seconds` |
| `intervention` | something changed the data | `actor` (`"timeout"` when a clock decided) |

The stream ends when `job_status` reaches `completed` or `cancelled`.
`awaiting_approval` is **not** terminal — the stream stays open while the gate holds,
which for a `supervised` run can be days. A client that cannot hold a connection that
long should poll `GET /api/v1/gates` (§7) instead, or register a webhook (§8).

Python (streaming client):

```python
import json, requests

BASE = "http://127.0.0.1:8137"
job = requests.post(f"{BASE}/api/v1/jobs",
                    json={"description": "…", "mode": "auto"}).json()
jid = job["job_id"]

with requests.get(f"{BASE}/api/v1/jobs/{jid}/events", stream=True) as r:
    for line in r.iter_lines():
        if not line or not line.startswith(b"data:"):
            continue
        ev = json.loads(line[5:].strip())
        if ev["type"] == "stage_output":
            print(ev["stage"], "->", ev["data"]["summary"])
        elif ev["type"] == "stage_completed":
            print(ev["stage"], "done in", ev["data"]["duration_ms"], "ms")
        elif ev["type"] == "job_status" and ev["status"] in ("completed", "cancelled"):
            break
```

### `stage_output` summary shapes

The `data.summary` object per stage (all fields best-effort, bounded in size):

- **understanding** — `severity_score`, `severity_reasoning`, `summary`,
  `impact_assessment`, `entities[]` (`type`/`value`), `correlation_keys[]`,
  `event_time` (`start`/`end`), `log_sources_to_review[]`, `initial_hypotheses[]`,
  `recommended_actions[]`.
- **query_generation** — `count`, `queries[]` (`source`, `query`, `date_from`,
  `date_to`, `entities[]`).
- **log_retrieval** — `total_rows`, `source_count`, `sources[]` (`source`, `rows`,
  `samples[]`).
- **correlation** — `record_count`, `resolved_correlation_keys[]` (`entity_hint`,
  `sources{}`, `time_window`, `origin`), `discovered_join_keys[]`, `transforms[]`,
  `findings[]`, `summary_text`.
- **anomaly_detection** — `count`, `anomalies[]` (`description`,
  `confidence_score`, `potential_implications`, `recommended_actions`), sorted by
  confidence.
- **plugins** — `count`, `plugins[]`.
- **report_generation** — `sections[]` (outline).
- **export** — `paths[]`. **output** — `delivered`.

## 3. Watch it live — polling (no streaming client needed)

`GET /api/v1/jobs/{job_id}` returns the enriched snapshot. Unlike the SSE stream,
this is a plain request you can poll — and it now carries the **same per-stage
`duration_ms` and `summary`** the UI shows:

```bash
curl -s "$BASE/api/v1/jobs/<job_id>" | python -m json.tool
```

```json
{
  "job_id": "39f4…",
  "status": "running",
  "run_mode": "auto",
  "current_stage": "log_retrieval",
  "stages": [
    {"name": "understanding", "pass": 1, "status": "completed", "duration_ms": 2118,
     "summary": {"severity_score": 8, "correlation_keys": ["user", "session"], "...": "…"}},
    {"name": "query_generation", "pass": 1, "status": "completed", "duration_ms": 840, "summary": {"count": 3, "...": "…"}},
    {"name": "log_retrieval", "pass": 1, "status": "running", "duration_ms": null, "summary": null}
  ],
  "passes": {"total": 1, "current": 1},
  "error": null,
  "created_at": "…", "updated_at": "…",
  "open_gate": null,
  "pending_gate": null,
  "gate_history": [],
  "interventions": []
}
```

Each `stages[]` entry also carries `health` (see §7) once the stage has produced an
output. `open_gate` / `pending_gate` / `gate_history` are covered in §7.

`name` is always the **bare** stage name, so a client keys on what it always did, and
`pass` beside it is what tells two records of one stage apart — a run that took a
follow-up retrieval pass has two `query_generation` entries and two `log_retrieval`
ones. `passes` is always present and reads `{"total": 1, "current": 1}` on a run that
took none, so a pass selector can be rendered from that one field rather than inferred
from duplicate names.

## 4. Discover jobs

`GET /api/v1/jobs` lists all live jobs (newest first):

```bash
curl -s "$BASE/api/v1/jobs" | python -m json.tool
# -> {"jobs": [{"job_id", "incident_id", "status", "run_mode", "current_stage",
#               "created_at", "updated_at", "batch_id", "queue_position"}, …],
#     "queue": {"width": 2, "max_queued": 256, "running": 2, "queued": 3,
#               "admitted": 11, "queued_total": 4, "refused": 0, "withdrawn": 1}}
```

`queue_position` is 1-based and read live off the backlog rather than stored on the job, so
it cannot go stale as the queue drains; **0** means the job is not waiting. The `queue`
block rides alongside the rows because a caller reading `queued` on one job needs the width
and the depth to know what it is waiting for.

### `GET /api/v1/incidents` — what finished, on disk

`GET /api/v1/jobs` only knows about jobs the *current* process is holding. An incident
whose run finished (or that a restart forgot) is reachable only by its id, which the
console's Report tab otherwise asks an operator to type from memory. This lists what has
an artifact on disk, newest first:

```bash
curl -s "$BASE/api/v1/incidents?limit=50" | python -m json.tool
# -> {"incidents": [{"incident_id": "IR10000001", "mtime": 1754300000.0,
#                    "has_report": true, "has_pdf": true}, …]}
```

`limit` defaults to 200; a non-numeric value falls back to the default rather than
erroring, because this is the Report tab's first load and a 500 there costs the whole tab.
`mtime` is the newest of the incident's artifacts (epoch seconds). `has_report` /
`has_pdf` say which `?format=` values of §9 will resolve; an incident with neither (a JSON
export only) still appears.

**An id recovered from a filename is not trusted.** Every one goes back through the same
`_safe_id` guard the artifact endpoints use and anything that fails is **dropped, not
sanitised** — a file named `fraud_report_../../etc/passwd.md` yields no row at all, since
the alternative is handing a caller a string it will interpolate into a path. A missing
exports directory returns `{"incidents": []}`.

This `GET` shares the configured `post_incident_endpoint` path with the `POST` of the
classic endpoints (below): aiohttp routes on (method, path), and the pair is registered
together so renaming the endpoint cannot split them.

## 4b. Submit a backlog — batches

A batch is a **label** shared by its jobs (`incident["batch_id"]`) and nothing else: no
batch record, no batch state, no second copy of a run's progress. So it survives
export/import for free, a restart rebuilds it from the jobs themselves, and it can never
disagree with them about what happened. Every endpoint in §5–§9 works on each member
unchanged.

```bash
curl -s -X POST "$BASE/api/v1/batches" \
  -H 'Content-Type: application/json' \
  -d '{"mode": "auto", "incidents": [
        "Card changed then ticket issued for record ABC123",
        {"description": "Bulk export at 02:14 by sign 0404WX", "extended_retrieval": true}]}'
# -> {"batch_id": "b71c…",
#     "accepted": [{"job_id", "incident_id", "status": "running",  "queue_position": 0},
#                  {"job_id", "incident_id", "status": "queued",   "queue_position": 1}],
#     "refused":  [], "rejected": [],
#     "queue": { … as in §4 … }}
```

Each entry is a description string or an object taking the same `description`,
`extended_retrieval` and `link_pin` keys as §1. At most 500 incidents per request; past that
the whole batch is refused with **413** — a partially-applied bulk submission is worse than
a rejected one when the caller cannot see which half landed.

**Admission is per job and partial by design.** A batch whose tail exceeds the backlog bound
must not discard the incidents already accepted, so the refused ones come back in `refused`
with the `depth`/`limit` that refused them, and unparseable entries in `rejected` with their
`index`. Both are reported *beside* what was accepted; neither is an error status.

```bash
curl -s "$BASE/api/v1/batches"                 # every batch with at least one live job
curl -s "$BASE/api/v1/batches/b71c…"
# -> {"batch_id", "total": 2, "counts": {"running": 1, "queued": 1}, "done": 0,
#     "jobs": [{"job_id", "incident_id", "status", "current_stage", "queue_position", "error"}, …]}
curl -s -X POST "$BASE/api/v1/batches/b71c…/cancel"
# -> {"batch_id", "cancelled": ["39f4…"], "already_finished": ["a81b…"]}
```

`done` is a **floor, not a total**: the counts come from the jobs and nowhere else, and
`_prune` drops terminal jobs after the in-memory TTL, so a batch is exactly as durable as
its jobs. A batch id no live job carries is **404**. The cancel goes per job through the same
`control` path an operator uses, so each is recorded on its own audit trail, and a job that
ended between the read and the cancel is counted as `already_finished` rather than failing
the call — cancelling a backlog is a bulk action and one race in it must not refuse the other
299.

## 5. Control a running job

`POST /api/v1/jobs/{job_id}/control` with `{"action": "<action>"}`:

| action | effect |
|---|---|
| `pause` | pause at the next stage boundary |
| `resume` | resume from a pause |
| `step` | advance exactly one stage (STEP mode) |
| `cancel_stage` | cancel the current stage and stop the job |
| `cancel_all` | cancel the whole job immediately |
| `retry_stage` | re-run the failed stage (only after `stage_failed`) |
| `retry_all` | re-run the whole job from the first stage |
| `skip_stage` | leave a stage's output as-is and continue from the next stage. Accepts `{"stage": "<name>"}`; defaults to the failed stage. This is the continuation path after an override (§5b) — `retry_stage` would re-run the stage and throw the human's value away. |
| `release_gate` | abandon the open approval gate and continue as if approved, without review. Recorded as not reviewed in both `gate_history` and `interventions`. Distinct from `approve` on the gate endpoint (`POST .../gate`): `approve` means reviewed and accepted; `release_gate` means the gate is cleared without review, which a clock-driven timeout also records for the same reason. |

```bash
curl -s -X POST "$BASE/api/v1/jobs/<job_id>/control" \
  -H 'Content-Type: application/json' -d '{"action": "pause"}'
```

Returns the job snapshot. `409` if the action is invalid for the current state
(e.g. `retry_stage` when nothing failed, or `skip_stage` with no resolvable
target); `404` for an unknown job.

### Addressing one retrieval pass

Where a use case declares a follow-up retrieval pass, `query_generation` and
`log_retrieval` run more than once, and any action that names a `stage` also takes an
optional integer `pass`:

```bash
# re-run the SECOND retrieval pass only, leaving pass 1's rows alone
curl -s -X POST "$BASE/api/v1/jobs/<job_id>/control" \
  -H 'Content-Type: application/json' \
  -d '{"action": "retry_stage", "stage": "log_retrieval", "pass": 2}'
```

**`pass` omitted means the pass the run is currently on** — not pass 1. That is what every
client written before multi-pass retrieval sends, and defaulting it to 1 would make a
retry issued while pass 2 is in flight discard the pass-2 rows the operator is looking at.
On a single-pass run the two readings coincide, so nothing changes there.

A non-positive or non-numeric `pass` is **ignored** rather than refused: it is an optional
refinement of an action that is already fully specified without it, and a `400` over a typo
in a field the caller need not send would refuse the control action itself. A pass that is
well-formed but does not exist on this job addresses no stage record and comes back `409`,
the same as any other action invalid for the current state.

The snapshot reports `passes: {"total": N, "current": M}`, and each entry in `stages[]`
carries its own `pass`; a pass-1 entry keeps the bare stage name as its key, so nothing
that read the snapshot before needs to change.

## 5b. Override a stage's output (human in the loop)

`POST /api/v1/jobs/{job_id}/outputs/{stage}` replaces one stage's result with a
value you supply, so the rest of the pipeline runs on corrected data instead of
being restarted:

```bash
# 1. the job stopped with understanding wrong (or failed)
curl -s -X POST "$BASE/api/v1/jobs/<job_id>/outputs/understanding" \
  -H 'Content-Type: application/json' \
  -d '{"value": {"incident_id": "…", "analysis": { … }}, "actor": "analyst@corp"}'

# 2. continue from the next stage (do NOT use retry_stage — it would overwrite you)
curl -s -X POST "$BASE/api/v1/jobs/<job_id>/control" \
  -H 'Content-Type: application/json' -d '{"action": "skip_stage", "stage": "understanding"}'
```

`value` is decoded through the same codec the export/import path uses, so it must
match that stage's contract (e.g. an `UnderstandingResult` shape for
`understanding`, a `RetrievalQuery` list for `query_generation`). A shape mismatch
is a `400` — never a poisoned job. Overridable stages: `understanding`,
`query_generation`, `log_retrieval`, `correlation`, `anomaly_detection`,
`report_generation`. On a repeatable stage an optional `"pass": N` books the override
against one retrieval pass's record (absent = the pass the run is on), so the
`interventions` trail says *which* retrieval was hand-edited and not merely that one was.

Returns the job snapshot; the stage is marked `completed` and its `summary` is
recomputed. `400` if `value` is missing, malformed, or the stage is currently
`running` (it would overwrite your value on return); `404` for an unknown job or
stage name.

Every override and skip is appended to the job's `interventions` list, which rides
the snapshot, the SSE stream (a new `intervention` event, plus `stage_output` with
`"overridden": true`) and the export — so a report built on hand-edited data is
traceable as such.

## 5c. Edit the retrieval plan (add / remove queries)

§5b can replace the whole `query_generation` output, but for the one edit an analyst
actually makes — *this source should have been queried and wasn't* — that means
hand-writing a `RetrievalQuery` list, and a hand-written query is unscoped or scoped
by columns the source does not have. Which entities a source can bind is pack
knowledge, so these two endpoints let the client **name** a source and have the
server build the query through the same enrichment a planned one goes through.

```bash
# what is planned, what isn't, and what the procedure expected
curl -s "$BASE/api/v1/jobs/<job_id>/queries" | python -m json.tool
```

```json
{
  "job_id": "…", "pass": 1,
  "queries": [
    {"index": 0, "source": "record_lake", "question": "retrieve …",
     "date_from": "2026-08-01", "date_to": "2026-08-08", "scoped_by": ["record"]}
  ],
  "unselected": [
    {"source": "member_list", "purpose": "enrolment records for …",
     "declared": true, "deferred": false, "scopable": true}
  ],
  "dependencies": {"undeliverable": [], "not_queried": ["member_list"], "unscopable": []},
  "row_counts": {"record_lake": 173}
}
```

- `index` is the handle `remove` takes. `scoped_by` is the entity **types** the query
  actually narrows on (`time_window` excluded) — the query's entities are filtered to
  what the source can bind, so this is not readable off the incident text.
- `unselected` is the addable menu, ordered with the adjudicating procedure's own unmet
  dependencies first. `declared` = the procedure lists it as a hard dependency;
  `deferred` = a follow-up pass is due to retrieve it later (offered, but adding it now
  merges two scopes under one source name); `scopable` = at least one incident entity is
  a type it can filter on, and `false` means any query against it is a bare date scan.
- `dependencies` is the same three-way finding the planning stage reports and the health
  scorer reads (`undeliverable` needs credentials, `not_queried` needs a catalog fix or
  one click here, `unscopable` needs neither — declining it was correct). Nothing is
  ever force-added; this endpoint is how a human acts on that finding.
- `row_counts` is what each source has returned **so far on this run**, so you can see
  that a source already answered — and that the one you are about to remove is holding
  the rows.

```bash
curl -s -X POST "$BASE/api/v1/jobs/<job_id>/queries" \
  -H 'Content-Type: application/json' \
  -d '{"add": [{"source": "member_list"}], "remove": [2], "actor": "analyst@corp"}'
# -> {"job_id", "added": ["member_list"], "removed": ["src_c"], "queries": 3, "dependencies": {…}}
```

Both keys are optional (but not both absent → `400`). `question` is optional per
addition and falls back to a request built from the source's own declarations; the
window is borrowed from the existing plan. **All-or-nothing**, and removals resolve
against the indices the `GET` returned *before* any addition is appended — otherwise a
removal could target a row the same request just added.

Refusals name the reason that applies: `400` for a body that is not an object, a
non-integer index, an addition that is not an object, or a source that has no retriever
(the message says which); `409` for an index the plan does not hold (it says how many it
does), for a job with no `understanding` output yet, and for a `query_generation` stage
currently `running` — it would overwrite the edit the moment planning returned, which
reads as the edit having vanished; `404` for an unknown job; `503` when jobs or the
generator are not wired.

The edit is applied through the same `set_stage_output` path as §5b, so it lands in
`interventions` (`"retrieval plan edited by analyst: added …; removed …"`) and is
persisted before the response returns. Continue a paused/gated job with
`skip_stage` — `retry_stage` would re-plan and discard the edit. On a repeatable stage
an optional `"pass": N` books the edit against one retrieval pass.

## 5d. Cross-procedure links — inspect, refer, and set modes

A cross-procedure link is an advisory finding computed at the end of correlation. It says whether evidence from this run implicates a sibling procedure — not a verdict, not a finding that changes this run's score or health, but a question addressed to a human. Three endpoints serve this advisory lane. They do not duplicate the stage outputs: the correlation summary already carries every link finding, and these endpoints exist because three things a finding cannot itself state are only answerable here.

### `GET /api/v1/jobs/{job_id}/links` — budget, procedures, and current mode settings

```bash
curl -s "$BASE/api/v1/jobs/<job_id>/links" | python -m json.tool
```

```json
{
  "job_id": "…",
  "correlated": true,
  "link_count": 2,
  "job_modes": {"proc_b": "auto"},
  "procedures": ["proc_a", "proc_b", "proc_c"],
  "escalation": {
    "modes": ["planned", "semi_auto", "auto"],
    "engine_default": "semi_auto",
    "config_mode": "semi_auto",
    "min_escalation_score": 0.6,
    "escalation_budgeted": true,
    "probes_budgeted": true,
    "max_probes_per_run": 2,
    "probe_timeout_seconds": 120,
    "probe_deadline_seconds": 240,
    "probe_row_cap": 200,
    "max_children_per_run": 1,
    "max_child_depth": 1,
    "max_concurrent_children": 1,
    "max_total_children": 8,
    "children_budgeted": true
  }
}
```

`procedures` is the list of procedure names the loaded pack declares — the menu the mode endpoint below accepts. A mode can be set by name before correlation has run; without this list the pre-correlation ask has no valid names to supply. `job_modes` is what this run currently holds and what the next correlation will read.

`escalation_budgeted` says the lane can spend *something* — either paid rung having a budget answers it. `probes_budgeted` and `children_budgeted` say which one, because a run can reach a probe and stop there, and one number cannot say which rung is disarmed. When a flag is `false`, every escalating mode (`semi_auto` / `auto`) accepts the setting, records it, and then spends nothing on that rung at run time. The budgets and the mode are separate fields because each can independently be the reason a run escalated nothing, and their remedies differ. All budget values are resolved through the same functions the run itself reads, never re-derived at this endpoint.

The shipped defaults above are the engine's: 2 probes and 1 child run per run. Setting a budget to `0` disarms that rung and keeps the referral; an absent or unreadable value takes the default, so switching a rung off is an explicit act rather than a typo.

Answers before correlation: `correlated: false`, `link_count: 0`. Everything else — budget, procedure list, current per-job mode settings — is knowable from the config and the pack, and waiting for correlation would leave the one endpoint that can prevent an escalation-budgeted-but-never-spending run available only after the run.

**`404`** for an unknown job. **`503`** when no job manager is wired.

### `POST /api/v1/jobs/{job_id}/links/{index}/refer` — compose a referral for one link

`{index}` is the 0-based position of the link in the correlation output's link list. The endpoint derives the child run's window from the link's causal direction and the current retrieval plan, composes the full request body a child job would need, and records the composition in the parent's audit trail. **Nothing is launched.**

```bash
curl -s -X POST "$BASE/api/v1/jobs/<job_id>/links/0/refer" \
  -H 'Content-Type: application/json' \
  -d '{"mode": "auto", "actor": "analyst@corp"}'
```

```json
{
  "job_id": "…",
  "index": 0,
  "launched": false,
  "launch_recorded": "",
  "request": {
    "description": "Referral from job … (incident …). Investigate proc_b for entity_type val1, val2. …",
    "mode": "auto",
    "link_pin": "proc_b"
  },
  "post_to": "/api/v1/jobs",
  "pin": {
    "use_case": "proc_b",
    "playbook_id": "",
    "enforced": true,
    "note": "Applied. `link_pin` in the request body selects the child's procedure through the same single resolution seam the verdict and the retrieval plan both read …"
  },
  "scope": {
    "pivot_entity": "entity_type",
    "pivot_values": ["val1", "val2"],
    "direction": "antecedent",
    "date_from": "2026-07-01",
    "date_to": "2026-08-15",
    "window_hint": "lookback",
    "window_applied": "lookback:45d"
  },
  "parent": {"job_id": "…", "incident_id": "…"},
  "link": {"state": "probed_positive", "rung": 2, "advisory_severity": "HIGH"}
}
```

Submit `request` verbatim to `POST /api/v1/jobs` to start the child run. `link_pin` is what routes the child to the correct procedure through the same single selection seam the parent's verdict used — dropping it falls the child back to scoring its description, which is the parent's prose and would re-select the parent's procedure.

`launched` is always `false`: this endpoint composes and never spawns. If the child has already been launched separately, supply `launched_job_id` to record that launch in the parent's audit trail; the id is verified against the job registry and refused if not found.

| field | required | notes |
|---|---|---|
| `mode` | no | run mode for the composed request; absent = the parent run's own mode |
| `actor` | no | recorded in the audit trail |
| `launched_job_id` | no | record an already-launched child job id in the audit trail; the endpoint does not launch it |

**`200`** with the composed referral. **`400`** if the index is not an integer or the body is not a JSON object. **`409`** if the job has not correlated yet (no links to index into), if no link exists at the given index (correlation may have re-run since the index was read — re-read the job snapshot), if the link has no pivot values (an unreachable candidate has nothing to refer), or if `launched_job_id` does not match a known job. **`503`** when the job manager or referral-window generator is not available. **`404`** for an unknown job.

### `POST /api/v1/jobs/{job_id}/links/mode` — set escalation mode per target procedure

Set how this run handles one or more target procedures: whether a confirmed link is composed for a human to execute (`planned`), escalated automatically only when the link score meets the configured threshold (`semi_auto`), or escalated automatically whenever the rung-1 gate permits (`auto`). The setting is stored on the job and survives a gate rejection and re-correlation.

```bash
curl -s -X POST "$BASE/api/v1/jobs/<job_id>/links/mode" \
  -H 'Content-Type: application/json' \
  -d '{"mode": "auto", "targets": ["proc_b"], "actor": "analyst@corp"}'
```

```json
{
  "job_id": "…",
  "mode": "auto",
  "applied": [
    {
      "target_use_case": "proc_b",
      "asked": "auto",
      "mode": "planned",
      "mode_source": "clamp",
      "mode_note": "rung-1 gate did not pass on this run's rows — retrieve the source that opens this procedure's leg first",
      "proposed_action": "compose",
      "links": 1
    }
  ],
  "link_modes": {"proc_b": "auto"},
  "targets_checked_against_pack": true
}
```

`mode` in each `applied` entry is the **effective** mode after clamping, which may differ from `asked`. A rung-1 gate that did not hold clamps the effective mode to `planned` regardless of the request; the ask is stored on the job either way and re-resolved at the next correlation against real rows. `link_modes` reflects what is actually stored on the job.

`targets` omitted means every link this run currently holds (requires that correlation has run). Naming targets explicitly allows setting the mode before correlation, provided the names are procedures the pack declares.

All-or-nothing: every target name is resolved before anything is written. Where the pack is reachable, names are validated against it; `targets_checked_against_pack: false` in the response means the validation was skipped because no pack handle was available, not that validation passed.

| field | required | notes |
|---|---|---|
| `mode` | **yes** | `planned` \| `semi_auto` \| `auto` |
| `targets` | no | list of procedure names; absent = all current links (requires correlation to have run) |
| `actor` | no | recorded in the audit trail |

**`200`** with the effective settings per target (check `mode` not `asked` to see what the run will do). **`400`** if the body is missing or not an object, `mode` is not one of the valid values, `targets` is not a list of non-empty strings, or a named target is not a procedure in the loaded knowledge pack. **`409`** if `targets` is omitted and this job holds no cross-procedure links yet (name targets explicitly to pre-set the mode). **`503`** when no job manager is wired. **`404`** for an unknown job.

---

## 6. Get the result / export

```bash
curl -s "$BASE/api/v1/jobs/<job_id>/export" | python -m json.tool
```

Returns the full job document: `stage_statuses`, `stage_durations`,
`stage_summaries`, `event_history`, and `outputs` (including `outputs.report`).
This is the easiest way for a non-streaming client to get everything after the
run. The same document can be re-imported for study/retry:

```bash
curl -s -X POST "$BASE/api/v1/jobs/import" -H 'Content-Type: application/json' \
  --data @exported_job.json
```

---

## 7. Approval gates (semi_auto / supervised)

A gate is a stage that has finished, published its output, and is now **waiting for a
human** before the run continues. It is the review counterpart to `step`: `step` asks
permission to start a stage, a gate asks whether what a stage produced is good enough
to build on.

Gates open on the six LLM stages only — `understanding`, `query_generation`,
`log_retrieval`, `correlation`, `anomaly_detection`, `report_generation`. `plugins`,
`export` and `output` are never gateable: their outputs are side-effect records, so
there is nothing an approval would change.

### Stage health — why a gate opened

Whether `semi_auto` gates is decided by a **deterministic** score, never by the LLM
judging its own work. `score_stage` starts at 1.0 and subtracts a configured weight per
*countable* defect (no entities extracted, sources that returned nothing, unresolved
join keys, a fallback report…). Some signals are fatal (weight 1.0 → score 0.0); one,
`empty_sources`, is pro-rata in the fraction of sources that came back empty.

```json
{
  "score": 0.65,
  "threshold": 0.6,
  "gate_recommended": false,
  "scored": true,
  "reasons": [
    {"code": "empty_sources", "weight": 0.2,
     "detail": "3 of 6 source(s) returned zero rows: source_a, source_b, source_c", "count": 1},
    {"code": "source_timeout", "weight": 0.15,
     "detail": "1 source(s) timed out: source_d", "count": 1}
  ]
}
```

Note the arithmetic: `empty_sources` has a configured weight of `0.4` but cost `0.2`
here, because it is scaled by the fraction of sources that came back empty (3/6).
`weight` in a reason is what was **actually** subtracted, not the configured maximum.

This object rides `stage_output` events, every `stages[]` entry in the snapshot, and the
gate record. Two fields are easy to misread:

- `gate_recommended` is `score < threshold`. It is a *recommendation*: whether a gate
  actually opens also depends on the run mode and on the stage being enabled.
- **`scored: false` does not mean healthy** — it means no signal was computable for
  that stage, so nothing was measured. `semi_auto` will not gate on an unscored stage
  (there is no evidence to gate on); `supervised` gates regardless.

Thresholds and weights are configured, not sent: `stage_gates.threshold`,
`stage_gates.stages.<name>.{enabled,threshold}` and `stage_gates.weights.<code>` in
`main_config.yaml`. Setting a weight to `0.0` disables that signal without a code
change.

### Is anything waiting on me?

```bash
# one job
curl -s "$BASE/api/v1/jobs/<job_id>/gate" | python -m json.tool
# -> {"job_id", "status", "open_gate": { … } | null, "gate_history": [ … ]}

# every job, oldest gate first — the approvals inbox
curl -s "$BASE/api/v1/gates" | python -m json.tool
```

`GET /api/v1/gates` is the endpoint an integrating app should build on. It reports
gates across **all** jobs, including ones this process restored from disk after a
restart and ones launched by a different client — neither of which any per-job
subscription would ever show you. Each row is the gate record plus `job_id`,
`incident_id` and `run_mode`.

The gate record:

```json
{
  "stage": "correlation",
  "pass": 1,
  "reason": "health below threshold",
  "health": { … },
  "summary": { … },
  "actions": ["approve", "reject", "override"],
  "opened_at": "2026-07-30T09:14:02+00:00",
  "timeout_seconds": null,
  "on_timeout": null,
  "reopened_after_restart": true
}
```

Everything needed to render a decision screen is in that one object — health, the
stage's summary, the allowed actions — so no second call is needed. `reason` is
`"supervised mode"` or `"health below threshold"`. `pass` says which retrieval pass this
gate belongs to (`1` unless the use case declared a follow-up), and the health it carries
is that pass's own — a clean first retrieval plan does not vouch for a second one.

The snapshot distinguishes two states that look alike and are not:

| field | meaning |
|---|---|
| `open_gate` | a runner **is** waiting. Answerable now. |
| `pending_gate` | a gate carried across a restart that has not been re-armed yet. **Not** answerable — a decision would resolve nothing. |

`pending_gate` normally becomes `open_gate` within moments of startup
(`rearm_gates()`). Render it as "being re-armed", not as a button.

### Resolve a gate

`POST /api/v1/jobs/{job_id}/gate`:

```bash
# Approve — accept the output and continue.
curl -s -X POST "$BASE/api/v1/jobs/<job_id>/gate" \
  -H 'Content-Type: application/json' \
  -d '{"action": "approve", "actor": "analyst@corp", "reason_code": "verified_correct"}'

# Reject — re-run the stage with your correction injected into its prompt.
curl -s -X POST "$BASE/api/v1/jobs/<job_id>/gate" \
  -H 'Content-Type: application/json' \
  -d '{"action": "reject",
       "guidance": "That value is the org-unit id, not the actor id. Re-extract entities.",
       "reason_code": "wrong_entities",
       "restart_from": "understanding",
       "actor": "analyst@corp"}'

# Override — replace the output with a corrected value, then continue.
curl -s -X POST "$BASE/api/v1/jobs/<job_id>/gate" \
  -H 'Content-Type: application/json' \
  -d '{"action": "override", "value": { … }, "actor": "analyst@corp"}'
```

| field | applies to | notes |
|---|---|---|
| `action` | all | `approve` \| `reject` \| `override` |
| `guidance` | `reject` | **required.** A reject without a correction re-runs an identical prompt and returns an identical answer. |
| `restart_from` | `reject` | re-run from an **earlier** stage instead of this one. A later stage cannot be what produced this output, so pointing forward is a `400`. |
| `pass` | `reject`, `override` | on a repeatable stage (`query_generation` / `log_retrieval`), which retrieval pass to address — see §5. Absent: the gate's own pass. |
| `value` | `override` | required; decoded through the same codec as §5b, so a shape mismatch is a `400`. |
| `actor` | all | recorded in the audit trail. |
| `reason_code` | all | free-form; the UI offers `incomplete_data`, `wrong_entities`, `wrong_scope`, `false_positive`, `missed_finding`, `verified_correct`, `other`. |
| `note` | all | free text, recorded instead of `guidance` when approving. |

Returns the job snapshot. **`409`** when no gate is open — a stale client racing a
decision someone else already made, not a server fault; distinguish it from `400`
(bad action, missing `guidance`, forward `restart_from`, un-decodable `value`).

After a `reject` the stage re-runs with the guidance injected, and then **gates
again** — a correction that did not land is a decision you get to make twice, not a
silent pass. Every resolution is appended to `gate_history` (the supervision trail)
and to `interventions` (the trail of things that changed the data).

### Gates hold indefinitely by default

A gate waits forever unless a timeout is configured, because a timeout that quietly
expires is a decision made by a clock — the opposite of what choosing `semi_auto` or
`supervised` asked for. Per-stage or globally:

```yaml
stage_gates:
  timeout_seconds: 3600          # omit or null = hold forever (the default)
  on_timeout: hold               # hold | proceed | abort
  stages:
    report_generation: {timeout_seconds: 7200, on_timeout: hold}
```

`0` and negative values read as "no timeout" rather than "expire immediately".

| `on_timeout` | effect |
|---|---|
| `hold` (default) | notify once, then keep waiting. Nothing is decided. |
| `proceed` | continue as if approved — recorded as **not reviewed** |
| `abort` | cancel the job |

`hold` fires its `gate_timeout` event exactly once and then reverts to waiting
forever: a repeating alarm on a gate nobody has answered is how a recipient learns to
mute the channel. `proceed` and `abort` are decisions taken without a human, so each
writes to `gate_history` **and** `interventions` with `actor: "timeout"` — a report
must never look reviewed when the reviewing was done by a clock.

### Durability

A gate that waits days outlives the process. Jobs persist to
`<AFIR_DATA_DIR>/jobs/`, and a restart restores an in-flight job as `PAUSED` with its
gate re-armed on the **existing** output — never silently re-running a stage whose
side effects are unknown. A re-armed gate carries `reopened_after_restart: true`;
surface it, because "this has been waiting since before a restart" changes how urgent
it is. A shutdown under a gate is not a cancel.

## 8. Webhooks — get told instead of polling

SSE needs a client connected *and staying* connected, which a two-day gate outlives.
Webhooks POST the events an external system acts on, so an integrating app can be
notified while nothing of its own is running. Configured in `main_config.yaml`, not
via the API:

```yaml
webhooks:
  enabled: true
  timeout_seconds: 10
  max_attempts: 3
  targets:
    - name: ops-slack
      url: ${AFIR_WEBHOOK_SLACK_URL}        # a Slack URL IS a secret — env var, never a literal
      events: [gate_opened, gate_timeout]
    - name: case-system
      url: ${AFIR_WEBHOOK_CASE_URL}
      headers: {Authorization: "Bearer ${AFIR_WEBHOOK_CASE_TOKEN}"}
```

Subscribable events: `gate_opened`, `gate_resolved`, `gate_timeout`, `job_completed`,
`job_cancelled`, `stage_failed`. The per-stage stream is deliberately not
subscribable — firing on all of it is a self-inflicted DoS on the receiver.

The payload is the SSE event object plus an `event` key holding the webhook name you
subscribed to (`job_completed` arrives on the wire as `type: "job_status"`,
`status: "completed"`; match on `event`).

**Not a delivery guarantee.** Retries are bounded and there is no durable outbox: a
POST in flight when the process dies is gone. That is acceptable because the state it
announces *is* durable — the gate is still open, still in `GET /api/v1/gates`, still
in the job store. Treat a webhook as a latency optimisation over polling; anything
that treats it as the system of record will drop a gate at the first restart.

Delivery never blocks or fails a run. An unresolved `${VAR}` is logged as an ERROR
and that target is dropped, rather than silently sending nothing.

---

## 9. Artifacts — the report and the evidence behind it

The report is the acceptance artifact and the evidence is what makes it checkable, so
both are served directly rather than only as a JSON blob inside the job document.

Every artifact endpoint exists in **two spellings** that resolve to the same files:

| by job | by incident |
|---|---|
| `GET /api/v1/jobs/{job_id}/report` | `GET /api/v1/incidents/{incident_id}/report` |
| `GET /api/v1/jobs/{job_id}/evidence` | `GET /api/v1/incidents/{incident_id}/evidence` |
| `GET /api/v1/jobs/{job_id}/artifacts` | `GET /api/v1/incidents/{incident_id}/artifacts` |

The incident-keyed spelling is the durable one: artifacts live under `exports_dir()`
and outlive the in-memory job, so they still resolve after a restart, and they exist
for an incident submitted through a classic blocking call that never had a job.

### The report — `?format=`

```bash
curl -s   "$BASE/api/v1/incidents/<id>/report?format=view"        # JSON: {html, toc, …}
curl -s   "$BASE/api/v1/incidents/<id>/report?format=html"        # standalone HTML page
curl -sO  "$BASE/api/v1/incidents/<id>/report?format=md&download=1"
curl -sO  "$BASE/api/v1/incidents/<id>/report?format=pdf&download=1"
curl -s   "$BASE/api/v1/incidents/<id>/report?format=json"        # the InvestigationReport
```

| format | body | notes |
|---|---|---|
| `view` | `{"incident_id", "html", "toc": [{"id","title","level"}], "chars"}` | for embedding in a page that already has styling — what the UI's Report tab renders; `toc` ids are unique anchors into the `html` |
| `html` | a standalone document | carries its own CSS; needs no external asset, so it works offline |
| `md` | the Markdown the report stage wrote | |
| `pdf` | `application/pdf` | |
| `json` | the `InvestigationReport` structure | `sections` come from the **in-memory** report, so the job-keyed spelling returns the real per-section split; read off disk there is only the Markdown, which degrades to one `{"section_title": "Report"}` section holding the whole document (deliberate — better than a 404 for a report that plainly exists) |

`&download=1` adds `Content-Disposition: attachment` with a stable filename.

**Both spellings serve all five formats.** The job-keyed handler reads its in-memory
report from the job's *context* (`Job.context.outputs`); reading `job.outputs` — an
attribute that does not exist — made every `/jobs/{id}/report` request a 500 while the
incident-keyed route worked, so half the Report tab's buttons were dead depending on
which id it held. `format=json` is the one place the two legitimately differ in
content, per the note above; `md`/`pdf`/`view`/`html` are byte-identical.
Markdown is rendered **server-side** (`src/report_delivery.py::markdown_to_html`) —
no CDN, no client-side library, so the UI works in an offline Databricks App. The
renderer escapes HTML in the source before it applies any markup, so LLM-authored
prose cannot inject script.

A `404` distinguishes *"the report stage has not run"* from *"the renderer broke"*;
an unknown `format` is a `400` listing the accepted ones.

### The evidence — `?kind=raw|transformed`

```bash
curl -s  "$BASE/api/v1/incidents/<id>/evidence?kind=raw"            # bounded OUTLINE
curl -sO "$BASE/api/v1/incidents/<id>/evidence?kind=raw&download=1" # the real file
curl -s  "$BASE/api/v1/incidents/<id>/evidence?kind=transformed&full=1"
```

- **`raw`** — every row retrieved from every source, before correlation. One group per
  source, ordered by row count.
- **`transformed`** — the correlated/derived view the findings were actually read off.
  One group per named block (`record_count`, `summary_text`, aggregations, findings,
  transforms, evidence pack), surfaced by name so a client renders a card per block
  without knowing the schema.

The default response is a **bounded outline**, not the artifact: raw evidence is every
row from up to ~19 sources and routinely tens of MB (926 KB and 244 KB on one small
real incident), which no browser should be handed to draw a preview.

```json
{"incident_id": "…", "kind": "raw", "bytes": 926813, "total_rows": 599,
 "groups": [{"name": "auth_events", "count": 500, "truncated": true, "rows": [ … ]}, …]}
```

Each group carries `rows` (the first 25, with `truncated: true` when it cut) — or
`value` instead, for a `transformed` block that is a scalar rather than a list.
Bounding the *preview* and not the *artifact* is the distinction that matters: nothing
is hidden, it is just not all rendered at once. `?full=1` (or `?download=1`) serves the
real file.

An unknown `kind` is a `400`.

### What exists — `/artifacts`

```bash
curl -s "$BASE/api/v1/incidents/<id>/artifacts" | python -m json.tool
```

```json
{"incident_id": "…",
 "artifacts": {"report_md":          {"exists": true,  "bytes": 11357, "filename": "…"},
               "report_pdf":         {"exists": true,  "bytes": 9807,  "filename": "…"},
               "evidence_raw":       {"exists": true,  "bytes": 926813, "filename": "…"},
               "evidence_transformed": {"exists": true, "bytes": 244250, "filename": "…"},
               "export_json":        {"exists": true,  "bytes": 10255, "filename": "…"},
               "export_csv":         {"exists": true,  "bytes": 1800,  "filename": "…"}}}
```

Absence is reported (`"exists": false`), never raised — a client asks this first and
offers only the downloads that will succeed. An incident id that isn't a plain
identifier is **rejected** with a `400` rather than sanitised: a cleaned-up hostile id
would read some *other* incident's file and hand it back as the caller's.

---

## 10. Configuration — read, change, import

The config editor's API. `config/*.yaml` is the master; these endpoints read it,
patch individual fields, and replace or import whole files.

> **Literal secrets never leave the server.** Every value under a secret-shaped key
> (any key containing `token`/`password`/`passwd`/`secret`/`credential(s)`, or named
> `api_key`/`apikey`/`private_key`/`access_key`) is replaced with
> `__redacted__` in the structured view **and** in the raw text. An `${ENV_VAR}`
> reference comes back verbatim, because that is a variable *name* and an operator
> needs to see which one a backend reads. Per the repo convention, secrets are
> referenced by env-var name in YAML — so the editable form offers `api_key_env`, never
> `api_key`.

### `GET /api/v1/config` — everything, redacted

```json
{"config_dir": "/…/config",
 "redacted_placeholder": "__redacted__",
 "files": {"main_config.yaml": {"exists": true, "values": { … }, "raw": "<yaml text>"}, …},
 "sections": [{"key": "llm", "title": "LLM endpoint",
               "fields": [{"path": "base_url", "file": "llm_config.yaml",
                           "kind": "string", "applies": "restart",
                           "label": "base url", "help": "…",
                           "value": "https://…", "set": true}, …]}, …]}
```

- `kind` ∈ `number` | `integer` | `boolean` | `string` | `choice` | `text` — drives the
  control a client renders; `choice` fields carry `choices`, numeric ones carry
  `minimum`/`maximum`. (Everything not covered by a typed field is still editable
  through the raw YAML editor below.)
- `applies` ∈ `live` | `restart` — whether the running process can pick the value up.
- `set` is `false` for a key absent from the file; the field then reports its `default`.

### `PUT /api/v1/config` — patch fields

```bash
curl -s -X PUT "$BASE/api/v1/config" -H 'Content-Type: application/json' \
  -d '{"updates": {"anomaly_detection.threshold": 0.85, "logging.format": "json"}}'
# -> {"changed": [{"path": "anomaly_detection.threshold", "from": "0.8", "to": "0.85",
#                  "applies": "live"}],          # + "inserted": true for a new key
#     "skipped": [{"path": …, "reason": "…"}],
#     "reloaded": {"applied": ["anomaly_detection.threshold"], "restart_required": []}}
```

- **All-or-nothing.** Every value is validated against its descriptor *first*; if
  anything fails, nothing is written and the response is a `400` with per-path
  `errors`. A config write that half-lands leaves the app in a state nobody chose.
- **A value equal to `__redacted__` is dropped as "unchanged"**, so a client that
  renders the redacted read and PUTs it straight back cannot overwrite a password with
  the placeholder string.
- **`reloaded` is the honest half.** `live` fields are re-read into the running config
  and listed under `applied`; the rest are reported as `restart_required` instead of
  being presented as applied. A path the running process cannot reach is *not* created
  — a phantom write that appears to succeed is worse than an honest "restart required".
- Changing `logging.*` re-invokes `configure_logging()` (idempotent), which is what
  makes logging genuinely live rather than restart-required.
- Editing the config does **not** disturb feedback-driven threshold tuning: the
  configured value stays the baseline every bound is measured against (see §Feedback).

### `GET /api/v1/config/{name}` — export one file

```bash
curl -sO "$BASE/api/v1/config/main_config.yaml?download=1"
```

Raw text, redacted — safe to share or commit as a template. Unknown name → `404`.

### `PUT /api/v1/config/{name}` — whole-file save

```bash
curl -s -X PUT "$BASE/api/v1/config/main_config.yaml" \
  -H 'Content-Type: application/json' -d '{"text": "<yaml>"}'
# -> {"file": "main_config.yaml", "bytes": 4218, "keys": ["anomaly_detection", …],
#     "restart_required": true}
```

Parses the candidate before it replaces anything, writes atomically (temp file +
rename), and keeps the previous version as `<name>.bak` — gitignored, because a
rollback copy of a real config holds real credentials. Invalid YAML, YAML that isn't a
top-level mapping, or text still containing `__redacted__` is a `400`.

### `POST /api/v1/config/import` — multi-file import

```bash
curl -s -X POST "$BASE/api/v1/config/import" -H 'Content-Type: application/json' \
  -d '{"files": {"main_config.yaml": "<yaml>", "llm_config.yaml": "<yaml>"}}'
# -> {"written": [ … ], "restart_required": true}
```

**Validates every file before writing any of them.** Importing a matched pair where
only the first is valid would leave the app running a mismatched half of somebody's
environment. Text still containing `__redacted__` is refused — so a redacted export
cannot be re-imported over real credentials by accident.

All four rejections return `400` with `{"error": "Import rejected; nothing was
written", "errors": [...]}`: an unrecognised filename, non-string contents, invalid
YAML, and YAML whose top level is not a mapping (a bare list is valid YAML but not a
config file). That last rule used to be enforced only inside `replace_file`, i.e. at
*write* time — which returned a `500` for what is really a bad request and, in a
multi-file batch, only after the valid files ahead of it had already landed. A `500`
here therefore means a genuine I/O fault, and it reports `written` so the operator
knows the real on-disk state rather than having to guess.

---

## 11. Knowledge packs — browse, edit, validate, and the assistant

The pack editor's API. A knowledge pack is the domain brain — entities, sources,
rulesets, playbooks, concepts, report wording — and everything domain-specific lives
there rather than in `src/`. These endpoints read a pack's files, edit them in place,
lint the whole pack, keep every prior version, and run an LLM assistant that *proposes*
changes as a diff for approval.

Registered **unconditionally**, unlike the job endpoints: authoring a pack is what an
operator does before there is anything to run, so none of this needs a wired pipeline.
Only the five `assist` routes need an LLM client, and they answer `503` without one.

> **Three properties hold across every write.**
>
> - **Nothing is deleted as a side effect.** `DELETE` requires `confirm=1` in the query
>   itself — not in a dialog the UI happens to show — and even then the content is
>   snapshotted first, so it is undoable rather than merely warned about.
> - **Every write keeps the prior bytes**, content-addressed under
>   `knowledge/<pack>/.history/` (gitignored). A restore is itself snapshotted, so undo
>   is undoable.
> - **Edits are surgical, never a re-serialisation.** A line-range save keeps comments
>   and YAML anchors byte-for-byte; `use_cases/*/rules.yaml` is ~60% comments and
>   `source_catalog.yaml` relies on `&anchor`/`*ref`, both of which a
>   `safe_load` → `dump` round trip destroys.
>
> **And no write can leave the pack silently sourceless.** `pack._read_yaml` swallows
> every parse error and returns `{}`, so a broken `source_catalog.yaml` does not raise —
> the pack loads with *zero sources*, every condition goes `unknown`, and the next report
> reads INSUFFICIENT DATA, indistinguishable from "the sources had nothing". So the
> candidate is written to a temp file, **re-parsed from disk**, and refused if it parses
> to empty when the previous version did not. On refusal the target is byte-identical and
> the response is a `400` carrying the parser's own message *and line*.
>
> **`restart_required` is always `true`.** Editing a pack does not reload the one this
> process holds — the same honest answer the Configuration tab gives for a non-live
> field. `GET /api/v1/knowledge` says which pack is loaded so that advice is actionable.

Path safety is **rejection, not sanitisation** (as for report ids): each `?path=`
segment must match `[A-Za-z0-9._\-]{1,128}`, depth ≤ 6, no absolute path, no `..` as a
segment, and `.history` is not addressable. A traversal is a `400`, never a `200`
carrying somebody else's file. Editable suffixes are `.yaml .yml .md .txt`.

### `GET /api/v1/knowledge` — every pack, and which one is loaded

```bash
curl -s "$BASE/api/v1/knowledge"
# -> {"packs": [{"name": "mock_domain", "files": 16, "bytes": 104772, "loaded": true}, …],
#     "loaded": "mock_domain", "root": "/…/knowledge"}
```

`loaded` is matched on the pack *name*: the config holds a `pack_dir` path, so an
absolute `/srv/afir/knowledge/x` and a relative `knowledge/x` both resolve to `x`.

### `GET /api/v1/knowledge/{pack}` — the tree AND the diagnostics, in one payload

```json
{"pack": "mock_domain",
 "counts": {"files": 16, "dirs": 11, "bytes": 104772},
 "nodes": [{"path": "use_cases", "dir": true, "depth": 0, "name": "use_cases"},
           {"path": "source_catalog.yaml", "dir": false, "depth": 0,
            "name": "source_catalog.yaml", "bytes": 11902, "lines": 219,
            "kind": "catalog", "editable": true, "text": true}, …],
 "limits": {"inline_edit_max_bytes": 262144, "read_max_bytes": 1048576,
            "write_max_bytes": 2097152,
            "editable_suffixes": [".md", ".txt", ".yaml", ".yml"]},
 "validate": {"ok": true, "errors": 0, "warnings": 0, "infos": 0, "diagnostics": [],
              "counts": {"entities": 5, "sources": 3, "rulesets": 2, …}}}
```

`limits` comes from the server rather than being hardcoded by a client, so the read
limit, the write limit, the inline-edit threshold and the editable suffixes cannot drift
out of agreement with the store that enforces them.

One payload rather than two round trips, mirroring `GET /api/v1/config`, and for the
same failure mode: a pack whose catalog does not parse looks completely normal in a file
listing — same files, same sizes. A client that fetched the tree first would render a
clean, browsable, entirely wrong picture.

- `kind` ∈ `glossary` | `catalog` | `ruleset` | `shared_check` | `concept` | `playbook` |
  `case` | `data` | `schema` | `reporting` | `vocabulary` | `notes` | `other` — drives
  per-file help. Structural where it can be (the loader finds shared checks by
  *directory*, not by filename).
- **`editable` is a fact about the file, not a permission**: `false` above 256 KB.
  Three generated schema files in the installed packs are ~456 KB, and putting one in a
  textarea to `PUT` it back is the fastest way to lose it — a client offers the download
  and a line-range edit instead.

`GET /api/v1/knowledge/{pack}/tree` returns the node list alone.

### `GET /api/v1/knowledge/{pack}/file?path=` — one file

```bash
curl -s "$BASE/api/v1/knowledge/mock_domain/file?path=entity_glossary.yaml"
# -> {"pack": "mock_domain", "path": "entity_glossary.yaml", "kind": "glossary",
#     "text": "…", "bytes": 3140, "lines": 96,
#     "sha256": "9c1f…", "editable": true}
```

`?download=1` serves it as an attachment. Over 1 MB → `413` with a message that says to
download it. **`sha256` is the concurrency token** the write half takes back.

### `PUT /api/v1/knowledge/{pack}/file?path=` — save, whole file or by line range

```bash
# whole file
curl -s -X PUT "$BASE/api/v1/knowledge/mock_domain/file?path=entity_glossary.yaml" \
  -H 'Content-Type: application/json' -d '{"text": "<yaml>", "expect_sha": "9c1f…"}'

# one line range — this is the form that preserves comments and anchors
curl -s -X PUT "$BASE/api/v1/knowledge/mock_domain/file?path=source_catalog.yaml" \
  -H 'Content-Type: application/json' \
  -d '{"start_line": 412, "end_line": 412, "text": "    retrieval_class: primary",
       "expect_first_line": "    retrieval_class: secondary", "actor": "me"}'
# -> {"pack": "mock_domain", "path": "source_catalog.yaml", "changed": true,
#     "replaced": [412, 412], "bytes": 11849, "lines": 219, "sha256": "5801…",
#     "snapshot": "snap-000001-5a7b16d2debc",
#     "validate": { … }, "restart_required": true}
```

- `start_line`/`end_line` present (1-indexed, inclusive) → line-range replacement;
  absent → whole-file.
- **`409`** on an `expect_sha` mismatch: the file moved under the editor and answering
  `200` would silently discard whoever saved first. `expect_first_line` /
  `expect_last_line` do the same job for a range, where a stale line *number* is the more
  dangerous form of the same staleness — it still points at a line, just the wrong one.
- **`400`** on parse rejection, with `path` and `line`. **`413`** over 2 MB.
- The response carries the pack-wide `validate` payload, because a save is exactly when a
  *pack-level* break appears: the file parses fine on its own — the store proved that
  before replacing anything — while the ruleset now imports a check that no longer exists.

`POST` to the same URL **creates** (`409` if it exists); the two are separate intents
because conflating them means a mistyped path on a save writes a file no loader reads and
the operator's edit appears to have vanished.

### `DELETE /api/v1/knowledge/{pack}/file?path=&confirm=1`

```bash
curl -s -X DELETE "$BASE/api/v1/knowledge/mock_domain/file?path=notes.md"
# -> 400 {"error": "refusing to delete 'notes.md' without confirm=1 — repeat the
#          request with &confirm=1 if that is what you mean", "path": "notes.md"}
```

With `confirm=1` the content is snapshotted **first**, and the unlink is refused if that
snapshot could not be stored.

### `GET /api/v1/knowledge/{pack}/history[?path=][&snapshot=]` — the undo store

Three views of one record: the whole pack's snapshots, one file's, or with `?snapshot=`
the stored **text** of a single entry (what a revert preview and a diff both need).

```json
{"pack": "mock_domain", "path": "", "entries":
  [{"id": "snap-000001-5a7b16d2debc", "path": "source_catalog.yaml",
    "at": "2026-08-04T19:34:06+00:00", "reason": "write", "actor": "me", "session": "",
    "sha256": "5a7b16d2…", "bytes": 11902, "stored": true}, …]}
```

`reason` ∈ `write` | `create` | `delete` | `restore` | `assist`; `session` carries the
assist session id when the assistant made the change, so an applied plan is traceable to
the proposal that was approved. Newest first. Blobs are content-addressed, so
edit-then-revert costs one blob and snapshotting every write is cheap. No `git`, no
`subprocess`.

### `POST /api/v1/knowledge/{pack}/history/restore`

```bash
curl -s -X POST "$BASE/api/v1/knowledge/mock_domain/history/restore" \
  -H 'Content-Type: application/json' -d '{"snapshot": "snap-000001-5a7b16d2debc"}'
# -> {"pack": "mock_domain", "path": "source_catalog.yaml",
#     "restored": "snap-000001-5a7b16d2debc", "recreated": false,
#     "sha256": "5a7b16d2…", "bytes": 11902, "parses": true, "parse_error": "",
#     "validate": { … }, "restart_required": true}
```

`recreated` is `true` when the restore brought back a file that had been deleted.

The current content is snapshotted before the restore lands. A restore does **not**
require the restored bytes to parse — the one moment undo is most needed is right after
an edit went wrong — so `parses`/`parse_error` are reported instead of enforced.

### `GET /api/v1/knowledge/{pack}/validate[?strict=1]` — lint the whole pack

Nothing else lints a pack, and two defect classes are otherwise invisible: Pydantic's
`extra='ignore'` **silently drops a mistyped key** (the declaration loads clean and does
nothing), and a condition `kind` the evaluator does not dispatch falls through and is
never evaluated.

```json
{"pack": "<a large installed pack>", "ok": true, "errors": 0, "warnings": 2, "infos": 9,
 "counts": {"files": 66, "yaml_files": 33, "vocabulary": 56, "entities": 53,
            "sources": 30, "shared_checks": 19, "rulesets": 1, "conditions": 19,
            "use_cases": 10, "playbooks": 10, "concepts": 18, "schemas": 23},
 "diagnostics": [{"severity": "warning", "code": "unread-pack-key",
                  "path": "use_cases/…/rules.yaml", "line": 1225,
                  "message": "…", "detail": "…", "hint": "…"}, …]}
```

Diagnostics are sorted with errors first. `ok` is `errors == 0`. **Warnings never block a
save** — they are what the operator is told. Plain form is always `200`; a diagnostics
report is not an HTTP error. `?strict=1` returns the identical body at **`422`**, but only
when `ok` is false — a strict call on a clean pack is still a `200`, so a CI job can gate
on the status code without parsing the body.

Both packs shipped in this repo validate with **zero errors** today. The warnings that
remain on the larger one are real and deliberately left visible: every one is a source
binding a form-declaring entity as a flat list while sixteen of its siblings bind it per
form, and each needs its own column measured before it can be converted — which is a fact
about the data, not about the pack, so the lint reports the list rather than guessing.

- **`error` — the pack does not work, or works while lying:** `yaml-parse-failed`,
  `yaml-anchor-unresolved` (its own code: a `*ref` without its `&anchor` empties the
  *whole file*), `yaml-empty-but-nonblank`, `unresolvable-check-import` (the one
  deliberately fatal path in pack loading), `unknown-condition-kind`,
  `unknown-ruleset-source`, `unknown-physical-source`, `unknown-lookup-data`,
  `duplicate-source-name`, `duplicate-condition-id`, `duplicate-data-stem`,
  `missing-domain-vocabulary`, `neutrality-collision`, `frontmatter-parse-failed`,
  `ruleset-not-a-mapping`, `file-unreadable`, `unusable-value-form-stem` (an invalid
  regex or one without exactly one capture group: `value_stem` then returns `None` for
  every value, so the widening guard never fires and the pack reads as one that never
  declared a stem — while the predicate it exists to repair matches no row and reports
  0 rows as a success), `unknown-value-form-binding` (a source binds an entity under a
  form name the glossary does not declare: no value can classify as one, so those fields
  are never filtered on and every value of the form they were meant to catch is dropped).
- **`warning` — it works, but a declaration is inert or a human is misled:**
  `unread-pack-key`, `unknown-model-key`, `expected-label-noop` (honoured by only three
  of the evaluators — `distinct_count` is not one), `label-polarity-unaffirmed` (the
  defect that shipped with 1083 tests green; it cannot be an error because the engine
  cannot read prose), `missing-out-of-scope-label`, `orphan-concept-id`,
  `orphan-logical-source`, `ruleset-no-conditions`, `condition-missing-id`,
  `unknown-report-group`, `no-entities-declared`, `no-sources-declared`,
  `value-form-binding-mixed` (some sources bind a form-declaring entity per form while
  others bind it as a flat list, so the routing is real on some and absent on the rest,
  where one form's value lands on the other form's column — a warning and not an error
  because flat can be the honest answer, one column really holding both forms, which only
  a per-source measurement settles).
- **`info`, deliberately not a warning:** `playbook-only-use-case`. Nine of one installed
  pack's ten use-case directories are playbooks-only — nine permanent false alarms is how
  a check gets weakened and then ignored.

The last two error codes are the ones that fail a *suite-wide* test rather than a run:
a pack with no `domain_vocabulary.yaml`, and a vocabulary word colliding with the
engine's own terms (which fails **retroactively**, for packs installed earlier).

### `POST /api/v1/knowledge/{pack}/import` — multi-file

```bash
curl -s -X POST "$BASE/api/v1/knowledge/mock_domain/import" \
  -H 'Content-Type: application/json' \
  -d '{"files": {"shared/checks/lib.yaml": "<yaml>", "use_cases/x/rules.yaml": "<yaml>"}}'
```

**Validates every file before writing any**, byte-shape identical to
`POST /api/v1/config/import`: `400` with `{"error": "Import rejected; nothing was
written", "errors": [...]}`. A multi-file import is usually one coherent change — a
ruleset and the shared check it imports — and landing half of it leaves a pack broken in
a way neither file's author would recognise. Existing files are overwritten *with their
prior bytes snapshotted*, so an import is reversible file by file.

### `POST /api/v1/knowledge/scaffold` — a new pack from the template

```bash
curl -s -X POST "$BASE/api/v1/knowledge/scaffold" -H 'Content-Type: application/json' \
  -d '{"name": "my_domain", "vocabulary": ["widget", "sprocket", "depot"]}'
# -> {"pack": "my_domain", "created": true, "files": [ … ],
#     "vocabulary": ["widget", "sprocket", "depot"],
#     "template": "docs/knowledge-pack-template",
#     "validate": { … }, "restart_required": true}
```

Registered **before** `/{pack}` so the dynamic route cannot swallow it. `409` if the pack
exists. The copy *loads*, which is the property that matters: the author gets a green
baseline to diff their first edit against instead of an empty directory.

**`vocabulary` is required and an empty list is refused.** It is not paperwork — a pack
with no `domain_vocabulary.yaml` fails
`test_every_installed_pack_declares_its_vocabulary` and takes the whole suite down, and
the guarantee that file installs (the engine provably does not speak this domain's words)
would be quietly absent for the pack most likely to introduce a leak. The template's own
vocabulary is deliberately **not** copied: its placeholder words collide with its own
example ruleset, so `cp -r` would make the *template's* test fail.

### The assistant — propose, review, apply

`503` on all five routes when no LLM client is wired.

**Nothing here writes.** The assistant's tools are read-only — `list_files`,
`read_file`, `search`, `pack_summary`, `validate` — and there is no write tool at all,
which is what makes "nothing touches disk until you approve" structural rather than
procedural. `pack_summary` is also what stops the model inventing a condition `kind`: it
returns the entity types, source names, ruleset keys, shared-check ids and the kinds the
evaluator actually dispatches.

#### `POST /api/v1/knowledge/{pack}/assist` → `202`

```bash
curl -s -X POST "$BASE/api/v1/knowledge/mock_domain/assist" \
  -H 'Content-Type: application/json' -d '{
    "question": "Add a condition that flags more than one actor on a single record.",
    "focus": ["use_cases/x/rules.yaml", "shared/checks/lib.yaml"],
    "attachments": [{"name": "flow.png", "content": "<base64>"},
                    {"name": "procedure.pdf", "content": "<base64>"}]}'
# -> 202 {"session": "assist-00003", "pack": "mock_domain", "status": "exploring",
#         "tool_mode": "tools", "image_mode": "none", "turns": 0, "trail": [],
#         "plan": null, "preview": null,
#         "attachments": [{"name": "procedure.pdf", "kind": "document",
#                          "bytes": 88210, "chars": 12043, "note": ""}],
#         "attachment_errors": ["scan.tiff: .tiff is not a readable format — …"]}
```

`202` and a background task, not a held-open request: exploration is several LLM round
trips, and a proxy timeout would look identical to a model that produced nothing. The
caller then watches `/events` or polls the session.

**Attachments.** Documents (`.pdf .docx .doc .csv .json`) are converted to text
server-side; text formats (`.md .yaml .txt .log …`) are read directly; images
(`.png .jpg .gif .webp`) are passed through as content blocks. Caps: 8 attachments,
8 MB each (4 MB for an image), 40k characters per document and 120k in total — and **a
cap is always stated**, both in `note` and in the text the model sees, because a
truncated document that does not say so is read as the whole file.

**`attachment_errors` is the honest half, and this endpoint is deliberately *not*
all-or-nothing** (the asymmetry with the import above is intentional: an attachment only
adds context to a question, so refusing four good diagrams because a fifth was an
unreadable scan would make the operator re-upload all five). Every rejection comes back
and is rendered beside the accepted files, so nothing is dropped quietly.

**Two degradations, both reported rather than hidden:**

- **`tool_mode`** ∈ `tools` | `single_shot`. An endpoint that cannot call tools gets a
  bounded pre-loaded bundle — tree, diagnostics, the `focus` files — in one message and
  goes straight to the plan.
- **`image_mode`** ∈ `none` | `read` | `text_only`. Whether an image works is a property
  of the deployed model, so it is **probed with its own call** before the exploration
  loop; on refusal each image becomes a text placeholder naming the file and saying it was
  **not** seen, and the mode says `text_only`. An assistant that silently ignored an
  attached diagram would be exactly the *ran-and-found-nothing* vs *never-evaluated*
  confusion this codebase exists to prevent.

Two hard budgets bound exploration: **8 turns** and **200 KB of tool output** (three
456 KB schema files blow any context). Exceeding either appends an explicit "your
exploration budget is spent, propose from what you have" turn rather than stopping
silently, so the plan the operator sees was produced knowingly — `budget_spent` and a
`assist_note` event say so.

#### `GET /api/v1/knowledge/{pack}/assist/{session}` and `.../{session}/events`

The session snapshot, or the same trail as SSE. Events are **buffered as well as
pushed**, so a client that subscribes late still sees the whole trail — a tool round that
happened but is not visible reads, to the operator, as a model that did nothing. Event
types: `assist_status` (`exploring` → `proposing` → `proposed`, or `failed` / `applied` /
`rejected`), `assist_tool` (`turn`, `tool`, `args`, `bytes`) and `assist_note` — which is
where every degradation and budget exhaustion announces itself. The stream closes once
`assist_status` reaches `proposed`, `failed`, `applied` or `rejected`.
`GET /api/v1/knowledge/{pack}/assist` lists this pack's sessions with the plan, trail and
preview stripped.

Sessions are **in-memory, capped at 20**. A plan is computed against a snapshot of the
files, so it is worthless after a restart and persisting one would only invite applying a
stale plan to moved files.

The proposal is a validated `EditPlan`: `{summary, ops[], notes[], questions[]}`, each op
`{op: create|patch|delete, path, text, start_line, end_line, expect_first_line,
expect_last_line, reason}`. `questions` is where the model says what it could not
determine. The session's `preview` carries the same plan with **server-rendered
per-op diffs** plus `errors[]` and `skipped[]`; the client never computes a diff of its
own, because two implementations that disagree produce a diff that is not what gets
written — and the diff is the entire basis on which the change was approved.

An **unknown `op` is skipped and reported, never raised**: a model inventing `rename`
must not sink a proposal whose other three ops are fine.

The preview also carries `checks`, measured on the plan's **after-state** in a candidate
tree rather than described: `{ran, introduced[], resolved[], baseline_errors,
candidate_errors, candidate_warnings, rulesets, dry_run, problems[], seconds}`. Two
questions, and the second is the one nothing else asks. **Would the pack still validate** —
reported as the errors the plan *adds* (`introduced`) and never as the candidate's total,
because a pack with pre-existing errors is the normal state of one being worked on and gating
on the total makes the first fix unappliable. **And would the new conditions ever answer
anything** — `dry_run` replays the touched rulesets over the evidence of stored runs and
reports per-condition `pass`/`fail`/`unknown` counts plus the two findings a count cannot
express, `always unknown` and `never evaluated` (`src/knowledge/pack_dry_run.py`,
`docs/architecture/knowledge.md`). It states the bounds of its own reading: a stored run
carries neither the per-source row cap nor whether a query constrained the acting identity,
so each missing fact is reported **with its direction of error**. `ran: false` with
`problems[]` means the measurement did not happen — never folded into a clean result, and
never turned into a refusal either, since a plan is not at fault for a temp directory.

#### `POST /api/v1/knowledge/{pack}/assist/{session}/apply` — all or nothing

```bash
curl -s -X POST "$BASE/api/v1/knowledge/mock_domain/assist/assist-00003/apply" \
  -H 'Content-Type: application/json' -d '{"actor": "me", "allow_delete": false}'
# -> {"applied": true, "written": [ … ], "skipped": [ … ], "summary": "…",
#     "validate": { … }, "restart_required": true}
```

Every op is prepared and verified **in memory first** — path safety, suffix, size,
exists/not-exists, line bounds, the pre-image anchors, the final text parses, and it does
not become empty when the pre-image was not. **An error the plan *introduces* refuses the
whole write** on the same terms, because a proposal that leaves the pack failing validation is
one the next run loads as an *empty* pack, and the operator approved a diff rather than that
outcome — introduced, not total, so a pack already failing can still be repaired here. The
dry run gates nothing: *will this condition ever fire* is an authoring question for a human
reading the preview, and running a replay on every write would spend the budget to produce a
number no branch reads. One problem → **`400`, nothing written**,
with the per-op reasons so the next attempt is a correction rather than a re-run. A
`delete` op requires `allow_delete: true`; without it the op is blocked *visibly* in the
preview rather than omitted. An I/O failure mid-plan restores the files this call already
changed and answers **`500`** with `rolled_back` — a partly-applied plan leaves the pack
in a state neither the operator's before nor their after describes.

**"Edit first"**: the body may carry an overriding `plan`, which runs through the
identical validation. An operator who must accept a proposal verbatim or reject it will
accept a wrong one. `409` if the session has no plan.

#### `POST /api/v1/knowledge/{pack}/assist/{session}/reject` → `202`

```bash
curl -s -X POST "$BASE/api/v1/knowledge/mock_domain/assist/assist-00003/reject" \
  -H 'Content-Type: application/json' \
  -d '{"guidance": "Put the check in the shared library, not inline in the ruleset."}'
# -> 202 {"rejected": "assist-00003", "session": "assist-00004", "status": "exploring", …}
```

Starts a **new** session carrying the rejected plan, the original attachments and the
correction, so the proposal that was turned down stays inspectable beside its
replacement. **`guidance` is required**: a reject with no correction would re-send the
identical request and return the same plan, which reads as the assistant ignoring the
operator.

---

## Classic blocking endpoints

Run the whole pipeline and return the report in one response. Default body is
`{"incident_id", "report"}` (unchanged).

> **All three answer `501` when AFIR runs as a Databricks App.** The ingress closes a request
> at ~60–120s and a measured run is 38 min at the median, 476 min at the longest, so they
> cannot complete there — the refusal names `POST /api/v1/jobs` in `use_instead` rather than
> letting the caller wait for a cut connection. `incident_input.blocking_endpoints` is
> `auto | on | off` and `live`-editable: `on` re-enables them inside an App (useful against a
> short incident), `off` retires them on a VM. Everything else under `/api/v1/` is unaffected.

```bash
curl -s -X POST "$BASE/api/v1/incidents/freetext" \
  -H 'Content-Type: application/json' \
  -d '{"description": "suspicious transfer"}'
```

- `POST /api/v1/incidents/freetext` — body `{"description": "..."}`.
- `POST /api/v1/incidents` — body a full incident (`description` required;
  `id`/`timestamp` auto-filled).
- `POST /api/v1/ir` — body `{"id": "<IR id>"}`; looks the incident up first.

### `?verbose=1` — timings + events in one blocking call

Append `?verbose=1` to any of the three to also get the job id, per-stage
status + `duration_ms` + `summary`, and the full `events` history alongside the
report — parity with the jobs API, without streaming:

```bash
curl -s -X POST "$BASE/api/v1/incidents/freetext?verbose=1" \
  -H 'Content-Type: application/json' -d '{"description": "suspicious transfer"}'
```

```json
{
  "incident_id": "…",
  "job_id": "…",
  "report": { "sections": [ … ] },
  "stages": [ {"name": "understanding", "status": "completed", "duration_ms": 2118, "summary": { … }}, … ],
  "events": [ { "type": "stage_output", "stage": "understanding", "data": {"summary": { … }} }, … ]
}
```

---

## Feedback

An analyst review of a finished investigation. Reviews are persisted on arrival,
distilled into insights every `feedback.batch_size` reviews (default 10), and the
distilled insights are injected as **advisory** guidance into the
incident-understanding and anomaly-detection prompts of subsequent runs — that is
the loop that closes.

### `POST /api/v1/feedback` — submit a review

`incident_id` is required, plus either `human_feedback` (free text, the original
shape) or at least one structured field:

| field | meaning |
|---|---|
| `agrees_with_verdict` | bool — did the analyst agree with the system's verdict |
| `analyst_verdict` | the analyst's own conclusion, e.g. `"VALID FRAUD"` |
| `missed_anomalies` | list (a bare string is accepted) — what the system should have flagged |
| `false_positives` | list — what it flagged that it shouldn't have |
| `notes` | free-form rationale |
| `analyst` | who reviewed it |
| `job_id` | the job the review is about |

```bash
curl -s -X POST "$BASE/api/v1/feedback" -H 'Content-Type: application/json' \
  -d '{"incident_id": "…", "agrees_with_verdict": false,
       "analyst_verdict": "VALID FRAUD",
       "missed_anomalies": ["cash payment on a same-day high-value transaction"],
       "analyst": "analyst@corp"}'
# -> {"status": "received", "incident_id": "…", "stats": {"total_reviews": 1, "pending_in_batch": 1, …}}
```

`400` if `incident_id` is missing or the body carries nothing substantive.

### `GET /api/v1/feedback` — read what has been collected

```bash
curl -s "$BASE/api/v1/feedback" | python -m json.tool
# -> {"stats": {…}, "history": [ … ], "pending": [ … ], "insights": {"new_fraud_patterns": [ … ], …}}
```

`history` is the permanent review log (newest last), `pending` the un-distilled
batch, `insights` the merged distillations (newest first, de-duped).

### `GET /api/v1/feedback/guidance` — see the exact text injected into prompts

```bash
curl -s "$BASE/api/v1/feedback/guidance" | python -m json.tool
# -> {"apply_to_prompts": true, "guidance": {"understanding": "…", "anomaly_detection": "…"}}
```

Empty strings mean nothing has been distilled yet, or
`feedback.apply_to_prompts` is `false`. The guidance is capped (5 items per
category, ~1800 chars) and prefixed with a preamble that ranks it **below** the
knowledge pack, the official procedure, and any deterministic verdict.

### `POST /api/v1/feedback/process` — distill now, without waiting for the batch

```bash
curl -s -X POST "$BASE/api/v1/feedback/process"
# -> {"processed": 3, "insights": { … }, "stats": { … }}
```

`409` when nothing is pending (no point spending an LLM call). The response also
carries `threshold` — distillation re-evaluates the numeric threshold too (below).

### `GET /api/v1/feedback/threshold` — the feedback-derived confidence threshold

The numeric half of "apply": reviews also tune `anomaly_detection.threshold`.

```bash
curl -s "$BASE/api/v1/feedback/threshold" | python -m json.tool
# -> {"effective": 0.85,
#     "threshold": {"baseline": 0.8, "current": 0.85, "recommended": 0.85,
#                   "direction": "hold", "applied": true, "reviews_considered": 0,
#                   "false_positive_reports": 3, "missed_anomaly_reports": 0,
#                   "bounds": {"min": 0.65, "max": 0.95}, "reason": "…"},
#     "adjustments": [ … ]}
```

The direction comes from **counted structured review fields**, never from the LLM's
prose: reviews listing `false_positives` mean noise got through (raise the
threshold); reviews listing `missed_anomalies` mean real findings were filtered out
(lower it). A review listing both cancels out, and free text is not keyword-scanned
— guessing a number from prose is how a threshold starts moving for reasons nobody
can audit.

Tuning is **opt-in** (`feedback.auto_tune_threshold`, default `false`). With it off,
`recommended` is still computed and `applied` is `false` — advice a human can act on
by editing the config. With it on, an adjustment requires
`min_reviews_for_tuning` (default 5) new reviews since the last one and a clear net
direction, moves at most `threshold_step` (0.05), and stays inside
`[threshold_min, threshold_max]` *and* within `max_threshold_drift` (0.15) of the
configured baseline. `?configured=0.7` overrides the baseline for a what-if
calculation; a non-numeric value is a `400`.

The tuned value lives in `feedback_thresholds.json`, not in your YAML — the config
stays the declared baseline that every bound is measured against, and the whole
adjustment history stays auditable.

### `POST /api/v1/feedback/threshold/reset` — revert to the configured value

```bash
curl -s -X POST "$BASE/api/v1/feedback/threshold/reset"
# -> {"status": "reset", "effective": 0.8, "stats": { … }}
```

Discards the tuning history so the configured baseline applies again. (Setting
`auto_tune_threshold: false` also reverts immediately, without discarding history.)

All six feedback endpoints return `503` when the feedback loop isn't wired.

## Health

`GET /health` → `{"status": "ok"}` (dependency-free probe). **Unchanged, and deliberately
so:** this is the platform's liveness probe, it must stay fast, and it must not start
depending on a collaborator being wired. No query parameter means a byte-identical response
to what it has always returned.

`GET /health?deep=1` answers the question the bare probe cannot. **Three failures leave a
server that is genuinely up and a run that reports success**, and every one of them is
announced only in a container log the operator cannot read: an empty LLM credential (all six
LLM stages 401), a durable store that refuses every write (the run finishes and its pending
approvals are gone on the next restart), and declared sources that built no retriever (the
retrieval stage reports success, the conditions that needed them read `unknown`, and the
verdict comes back INSUFFICIENT DATA — indistinguishable from *"the sources had nothing"*).

```bash
curl -s "$BASE/health?deep=1" | python -m json.tool
# -> {"status": "ok", "llm_credential": true, "pack": true, "jobs": 2, "pipeline": true,
#     "storage": "databricks", "storage_ok": true, "storage_detail": null,
#     "sources_declared": 30, "sources_unavailable": {},
#     "retrieval_cache": {"enabled": true, "entries": 12, "rows": 4103, "hits": 7,
#                         "misses": 23, "hit_rate": 0.233, ...},
#     "run_queue": {"width": 2, "max_queued": 256, "running": 2, "queued": 3, ...}}
```

| key | meaning |
|---|---|
| `llm_credential` | `true` a token resolves · `false` wired but unusable (every LLM stage will 401) · `null` no LLM client wired at all |
| `pack` | whether `knowledge.pack_dir` is configured |
| `jobs` | count of live jobs, or `null` if no job manager is wired / the lister raised |
| `pipeline` | whether an incident-processing function is wired |
| `storage` | where durable state lives (`local` / `databricks` / `sql`), or `null` if no store is wired |
| `storage_ok` | `true` usable · `false` **with `storage_detail` naming the cause** · `null` not wired |
| `storage_detail` | why the store is unusable (a blank catalog, a 403 from a workspace-scoped PAT, an unset `$AFIR_SQL_DSN`, …), else `null` |
| `sources_declared` | how many sources the pack declared *and* the engine resolved — built plus unavailable |
| `sources_unavailable` | `{source: reason}` for every declared source that built no retriever on this boot; `{}` when all are queryable, `null` when no engine is wired |
| `retrieval_cache` | the retrieval cache's counters and its four bounds, or `null` when no retrieval engine is wired. `enabled: false` is the shipped default and means every lookup misses; `hits`/`misses` are reported beside `hit_rate` because a rate over three lookups is not a measurement |
| `run_queue` | the run queue's counters and its two live bounds (§4), or `null` when no job manager is wired. A `queued` depth that never falls is the shape of a width set to 1 on a box that can afford more, and a rising `refused` is the shape of one set too high for the LLM endpoint |

`null` and `false` are different answers on purpose. A pure-export deployment legitimately
has no LLM client, and reporting *"not wired"* as a failure is how an indicator learns to
cry wolf and stops being read.

`sources_unavailable` is a map rather than a boolean for the same reason: *0 unavailable* and
*18 of 30, every ES one* are different answers, and only the second names a credential to go
and set. That set is decided **once, at boot** from the config's backend credentials, so it
is knowable the moment the app is up — a run that discovers it the hard way spends its whole
budget (38 min at the median) to reach a verdict nobody can act on.

**`status` stays `"ok"` through all three**, deliberately. The server is up; a degraded
dependency is not a liveness failure, and conflating them would make the platform's probe
flap. Only `?deep=` in `1`/`true`/`yes` is a deep request — `?deep=0` gets the fast probe.
The console's topbar dot polls this every 60s and turns **amber** for any of the three,
collecting them rather than ranking them (a deployment missing one credential usually misses
several, so reporting only the first makes each fix reveal the next). Red is reserved for no
answer at all.

---

## Server logs

Every event is also logged (`[event] job=… type=… stage=… status=… : <message>`),
now with a short data digest appended — `| duration_ms=… summary={…keys…}` or
`| source=…` — so an operator watching only stdout/logs sees the same signal the
SSE stream carries (minus the full nested summary, which the API/SSE deliver).
