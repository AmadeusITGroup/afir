# AFIR User Guide

This guide covers the operator workflow: submitting incidents, monitoring runs,
using the web UI, controlling gates, editing retrieval plans, and retrieving
results. For the full HTTP API reference with request/response schemas, see
[API.md](API.md).

## Submitting an incident

### Via the web UI

Open the server's root URL (default `http://localhost:5000/`) in a browser. The
**Investigate** tab presents a text area for the incident description and a radio
group for the run mode. Enter a free-text description of the incident and click
**Investigate** (or **Start job**). The UI switches to the Monitor tab and streams
progress live.

### Via the HTTP API

The recommended submission path is `POST /api/v1/jobs`, which returns a job id
immediately and never holds the connection open:

```bash
curl -s -X POST "http://localhost:5000/api/v1/jobs" \
  -H 'Content-Type: application/json' \
  -d '{
    "description": "Multiple failed logins for user 4821, then a large transfer on 2024-08-26",
    "mode": "auto"
  }'
# -> {"job_id": "39f4…", "incident_id": "3e31…"}
```

The `mode` field selects how much human involvement the run requires (see "Run
modes" below).

Poll for the job status:

```bash
curl -s "http://localhost:5000/api/v1/jobs/<job_id>" | python -m json.tool
```

Stream live events over SSE:

```bash
curl -s "http://localhost:5000/api/v1/jobs/<job_id>/events"
```

### Classic blocking endpoints

Three older endpoints — `POST /api/v1/incidents`, `POST /api/v1/incidents/freetext`,
and `POST /api/v1/ir` — hold the HTTP connection open until the investigation
finishes and return the full report inline. They work on a local server but are
returned as **501** when running as a Databricks App (where the platform ingress
closes long-lived requests before a run can complete) or when
`incident_input.blocking_endpoints: off` is set. The 501 response body names
`POST /api/v1/jobs` as the alternative. See [API.md](API.md) for the full request
format for these endpoints.

## Run modes

Each job runs in one of four modes, chosen per request:

| Mode | Behaviour |
|---|---|
| `auto` | Every stage runs back-to-back with no human intervention. |
| `semi_auto` | A stage stops for approval only when its deterministic health score falls below the configured threshold. |
| `supervised` | Every gateable stage stops for human approval, regardless of score. |
| `step` | The run pauses *before* each stage and advances only when an operator sends a `step` action. Unlike the gate modes, `step` asks a different question: not "is this output acceptable" but "are you ready to proceed". |

The mode is submitted with the job request as the `mode` field. An unknown value
falls back to `auto`.

## The web UI — five tabs

### Investigate

Launch a new investigation. Select a run mode, enter the incident description, and
submit. The run starts immediately.

### Monitor

Streams the event log for the selected job. Two views share the same buffer: the
basic view shows stage transitions and health scores; the detailed view shows
individual stage outputs, extracted entities, generated queries, and retrieved-row
counts. Selecting a different job from the dropdown replays its event history from
the beginning, so all events for a completed job are still visible.

### Report

Renders the finished investigation report once the job completes. The report
includes the incident summary, correlation findings, verdict, detected anomalies,
risk assessment, and recommended actions. Controls to download the report in
different formats are on this tab.

### Configuration

Editable form for every modelled configuration key, plus a raw YAML editor for
keys not yet exposed as form controls. Changes marked `live` take effect on the
next stage that reads them with no restart. Changes marked `restart` require
restarting the server. Each field shows which applies. See
[configuration.md](configuration.md) for the full key reference.

Literal secret values are never displayed: the form shows the environment variable
name (`${VAR}`) rather than the resolved value.

### Knowledge

The domain pack editor: a file tree of the loaded knowledge pack, a text editor,
and a pack assistant (an LLM-backed authoring loop that can propose edits and run
validation). Changes written here are mirrored to durable storage when a remote
storage backend is configured.

## Approval gates

A gate is a review checkpoint that holds the run and waits for a human decision.
Gates appear on the gate panel, which overlays the current tab, and are also
reachable via `GET /api/v1/jobs/{id}/gate` and `GET /api/v1/gates` (all open
gates across all jobs).

### What a gate shows

The gate panel displays:
- Which stage is paused and why (health score below threshold, or supervised mode)
- A compact summary of the stage's output (entities extracted, queries generated,
  sources retrieved, verdict, anomalies)
- The health score and the signals behind it

### Gate actions

Send a gate decision via `POST /api/v1/jobs/{id}/gate`:

| Action | Effect |
|---|---|
| `approve` | Continue the run from the next stage. |
| `reject` | Re-run the same stage with the guidance you supply (optional `guidance` field). The re-run result is presented at a new gate. |
| `override` | Replace the stage output with a value you supply and continue. This is recorded in the job's intervention log so the finished report is traceable as hand-edited. |
| `skip_stage` | Skip the current stage entirely and move to the next one. |

All decisions are recorded in `Job.interventions` with the actor, timestamp, and
reason code. A gate that was approved by a timeout is marked with `actor: "timeout"` 
so a finished report cannot be read as reviewed when nobody reviewed it.

### Gate timeouts

By default, gates hold indefinitely. A timeout is opt-in via
`stage_gates.timeout_seconds` and `stage_gates.on_timeout` (`hold`, `proceed`, or
`abort`). See [configuration.md](configuration.md).

## Editing the retrieval plan

After the query-generation stage, an operator can add or remove retrieval queries
before log retrieval runs:

```bash
# Read the current plan
curl -s "http://localhost:5000/api/v1/jobs/<job_id>/queries" | python -m json.tool

# Add a source and remove query at index 2 (indices resolved atomically)
curl -s -X POST "http://localhost:5000/api/v1/jobs/<job_id>/queries" \
  -H 'Content-Type: application/json' \
  -d '{"add": [{"source": "my_source"}], "remove": [2], "actor": "analyst@corp"}'
```

The server builds the added query through the same enrichment path a planned query
takes (entity binding, date window, partition guards), because which columns a
source can bind is pack knowledge. A manually added query cannot be submitted as raw
SQL or DSL. The edit is recorded in `Job.interventions`.

This is the correct place to act on a dependency finding: when the investigation
report says a declared source was not queried, add it here, not by modifying pack
files mid-run.

## Retrieving results

### Report formats

Reports are available at `GET /api/v1/incidents/{incident_id}/report` (also
`GET /api/v1/jobs/{job_id}/report`):

| Format (`?format=`) | Returns |
|---|---|
| `view` | JSON object: rendered HTML, table of contents, metadata |
| `html` | Standalone HTML page |
| `md` | Markdown source |
| `pdf` | PDF file |
| `json` | The `InvestigationReport` Pydantic model as JSON |

Add `&download=1` to trigger a file download.

### Evidence

Raw and transformed evidence are available at
`GET /api/v1/incidents/{incident_id}/evidence`:

| Kind (`?kind=`) | Returns |
|---|---|
| `raw` | Every row retrieved from every source, before correlation |
| `transformed` | The correlated/derived view that the findings were read off |

The default response is a bounded outline (byte count, row count, top-level keys).
Add `&download=1` to download the full artifact, or `&full=1` to receive it inline.

### Export files

The server also writes export files to the `exports/` directory
(configurable via `AFIR_DATA_DIR`): JSON, CSV, XML, and Excel (`.xlsx`) exports
are generated by default at the end of each run.

## Submitting analyst feedback

Analyst reviews improve future runs. Submit a review with:

```bash
curl -s -X POST "http://localhost:5000/api/v1/feedback" \
  -H 'Content-Type: application/json' \
  -d '{
    "incident_id": "<incident_id>",
    "agrees_with_verdict": false,
    "analyst_verdict": "FRAUD",
    "missed_anomalies": ["unusual transfer pattern"],
    "analyst": "analyst@corp"
  }'
```

Reviews accumulate; once `feedback.batch_size` (default 10) have been collected
they are distilled into guidance that is injected into subsequent runs as advisory
context. When `feedback.apply_to_prompts` is `true` (the default), the distilled
insights are added to the incident-understanding and anomaly-detection prompts as
lower-priority context below the pack rules and procedure. The distilled guidance
is readable at `GET /api/v1/feedback/guidance`. Reviews also inform an optional
automated tuning of the anomaly-detection confidence threshold; see
`feedback.auto_tune_threshold` in [configuration.md](configuration.md).

## API reference

For the complete request/response schema, all endpoints, error codes, and curl
examples, see [API.md](API.md).

Endpoint paths are fixed in the server code. They are not configurable in
`main_config.yaml`. The `incident_input.post_incident_endpoint` and similar keys
in the template are informational comments, not routing configuration.
