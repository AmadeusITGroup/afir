<div align="center">

<img src="assets/BMWE2025_NextGenEU_gef_en_RGB.svg" alt="IPCEI Next Generation Cloud Infrastructure and Services" width="50%">

</div>


The content of this repository comprises the work which is being developed in the scope of the **RESCUE** (**RES**ilient **C**loud for **EU**ropE) project, a part of the **IPCEI-CIS** (IPCEI Next Generation Cloud Infrastructure and Services), the key digital policy project aimed at strengthening Europe's digital and technological sovereignty.


# Automated Fraud Investigation and Reporting system

AFIR is an LLM-driven pipeline for automated security-incident investigation. It
understands a free-text incident description, generates and executes log-retrieval
queries against one or more configured backends, correlates the retrieved evidence,
detects anomalies, and produces a structured investigation report. The engine is
domain-generic; domain knowledge (entity glossary, source catalog, investigation
playbooks, and verdict rules) is supplied by a knowledge pack under
`knowledge/<domain>/`.

The verdict itself is **not** an LLM judgement: the pack's rulesets are evaluated
deterministically in code, and the LLM's job is to understand the incident, shape
the queries, and narrate what the rules found. Those rulesets are written in a
compositional vocabulary — Boolean combination, aggregate-then-compare, ordering,
and pack-declared textual equivalence — so a new fraud pattern is new YAML rather
than new code; and they can be *drafted* by the LLM at authoring time, proposed as
a validated, dry-run edit plan that a human approves before anything is written.

The system is described in `docs/architecture/` and is covered by the Apache 2.0
license.

## System Architecture

An incident enters over HTTP and is processed by one pipeline. Each stage is a
class in its own module, constructed with the shared LLM client and a slice of
configuration; the knowledge pack feeds every stage that needs domain knowledge,
and the retrieval pair may repeat when the pack asks it to. Incidents may be
submitted one at a time or as a batch; either way they are admitted through one
FIFO run queue, which starts as many as `jobs.max_concurrent_jobs` allows and
holds the rest with a position an operator can see.

```
                         incident  (POST /api/v1/jobs)
                                       o
   knowledge pack                      v
   knowledge/<domain>/    +-----------------------------+
   +------------------+   |    Incident Understanding   |
   | entity glossary  |-->+-----------------------------+
   | source catalog   |                 o
   | playbooks        |                 v
   | verdict rulesets |   +-----------------------------+
   | concept docs     |-->|      Query Generation       |<--+
   +------------------+   +-----------------------------+   |
          |                              o                  | follow-up pass:
          |                              v                  | the ruleset says
          |               +-----------------------------+   | what to harvest
          +-------------->|        Log Retrieval        |---+ and where to ask
          |               | Elasticsearch / Kibana,     |
          |               | Databricks SQL, Snowflake,  |
          |               | REST                        |
          |               +-----------------------------+
          |                              o
          |                              v
          |               +-----------------------------+
          +-------------->|    Correlation + Verdict    |
          |               | deterministic rule engine   |
          |               +-----------------------------+
          |                              o
          |                              v
          |               +-----------------------------+
          |               |      Anomaly Detection      |
          |               +-----------------------------+
          |                              o
          |                              v
          |               +-----------------------------+
          |               |           Plugins           |
          |               +-----------------------------+
          |                              o
          |                              v
          |               +-----------------------------+
          +-------------->|      Report Generation      |
                          +-----------------------------+
                                         o
                                         v
                          +-----------------------------+
                          |     Export -> Delivery      |
                          | json, csv, xml, xlsx, md,   |
                          | pdf; email / file / API     |
                          +-----------------------------+

   o = a human-in-the-loop gate: the run can hold there for an analyst to
       approve the stage, reject it or override its output.
```

Query generation and log retrieval are the only stages that may repeat: a pack
declares what a follow-up pass should harvest and which source to ask next, and
`jobs.max_retrieval_passes` (default 3) bounds the number of passes. A pack that
declares nothing runs single-pass.

Every `o` on the diagram is a **human-in-the-loop gate**, and which of them arm is
the run mode's decision (`auto`, `semi_auto`, `supervised`, and the older `step`,
which pauses *before* a stage instead of reviewing its output). A gate holds the
run indefinitely by default: a rejection re-runs the stage with the analyst's
guidance and re-gates it, an override replaces the stage's output, and either lands
in the job's `interventions` trail so a report built on hand-edited data is
traceable as such. The gate after query generation is the one where the operator
edits the plan itself — a query is added or dropped by *naming* its source, and the
server builds it, because which entities a source can bind is pack knowledge.

The one thing acting across the pipeline that is not on the diagram is the
**feedback loop**: it persists analyst reviews and injects a distilled,
deliberately weak form of them into later runs.

## Components

| Component | Where |
|---|---|
| Incident input, HTTP API, web UI | `src/incident_input.py`, `src/webui.py`, `src/ui/` |
| Job machinery: stages, gates, cancel, SSE | `src/pipeline_runner.py`, `src/job_store.py` |
| Run queue and batch admission | `src/job_queue.py` |
| Incident understanding | `src/incident_understanding.py` |
| Query generation | `src/api_call_generator.py`, `src/follow_up.py` |
| Log retrieval and the four backends | `src/log_retrieval.py`, `src/retrievers/` |
| Retrieval cache (opt-in, per process) | `src/retrieval_cache.py` |
| Correlation, verdict engine, evidence pack | `src/correlation.py`, `src/evidence.py` |
| Investigation brief per use case | `src/usecases/`, `src/brief_prompt.py` |
| Cross-procedure link lane (advisory) | `src/links.py`, `src/link_*.py` |
| Anomaly detection | `src/anomaly_detection.py` |
| Plugin system | `src/plugin_system.py`, `plugins/` |
| Report generation and delivery | `src/report_generation.py`, `src/report_delivery.py` |
| Export, output channels | `src/export_results.py`, `src/output_interface.py` |
| Knowledge pack loader, validator, editor | `src/knowledge/` |
| RAG retrieval and embedding providers | `src/rag/` |
| Durable state (local disk, UC Volume, SQL) | `src/storage/` |
| Feedback loop, mid-run guidance | `src/feedback_loop.py`, `src/human_guidance.py` |
| Stage health scoring | `src/stage_health.py` |
| Typed pipeline contracts | `src/models/pydantic_models.py` |
| Configuration read/patch surface | `src/config_store.py` |

These are supported by utility modules for LLM integration, Databricks
authentication, path anchoring, error handling, input validation, and rate
limiting (`src/utils/`).

## Features

**Implemented**

- Automated incident understanding using LLMs: classification, entity extraction,
  and the event time window, from any incident shape
- Dynamic query generation for log retrieval (`src/api_call_generator.py`), with
  pack-declared guarantees enforced after generation rather than requested in the
  prompt (`src/retrievers/query_guards.py`)
- Asynchronous log retrieval from Elasticsearch, Kibana gateway, Databricks SQL,
  Snowflake, and REST sources, with schema discovery, partition discovery,
  per-source timeouts and row caps
- Multi-pass retrieval, where a pack declares a follow-up pass scoped by values
  that only a first retrieval could reveal
- An opt-in retrieval cache keyed on the question that was asked, so a rejected
  gate, a follow-up pass, a link child run and a comparison re-run do not each pay
  a full scan; only an *answer* is cached, and every hit reports its age
  (`src/retrieval_cache.py`)
- A **deterministic verdict engine**: the pack's rulesets are evaluated in code,
  not by the LLM, and the verdict the report narrates from is computed once
  (`docs/architecture/verdict-engine.md`)
- A **compositional** condition vocabulary, so a pattern the engine has not seen is
  new YAML and not new Python: Boolean composition over child conditions, a generic
  aggregate-then-compare (including baseline-relative bounds and modal
  concentration), an ordering primitive, and textual equivalence under a form the
  **pack** declares — the engine supplies the text operations and owns no notion of
  "the same thing" of its own
- Correlation across sources with a three-layer join-key resolution and an
  evidence pack that keeps the values a condition was decided on
- An advisory **cross-procedure link lane** (`src/links.py`): a second procedure's
  fraud recognised in this run's own evidence, escalated up a cost ladder whose
  every rung is bounded, and reported beside the verdict rather than inside it —
  by construction it can never change one
  (`docs/architecture/cross-procedure-links.md`)
- LLM-powered anomaly detection
- Automated report generation with a deterministic fallback, so the report always
  exists
- Output delivery: email, file, or an external API endpoint
- Human-in-the-loop: `auto` / `semi_auto` / `supervised` / `step` run modes,
  gates that hold indefinitely, stage rejection with guidance, mid-run output
  override, and an operator-editable retrieval plan — all recorded as
  interventions (`docs/architecture/hitl.md`)
- Feedback loop: analyst reviews are persisted, distilled, and injected as
  advisory guidance into subsequent runs
- Web UI over five tabs (Investigate / Monitor / Report / Configuration /
  Knowledge), served inline with no bundler and no CDN so it works offline
- Knowledge-pack editor and validator, including an LLM assistant that proposes
  pack edits and a probe that measures a source before a pack binds it
- RAG retrieval over the pack, with either a local embedding model or a serving
  endpoint
- Configurable durable state: local disk, a Databricks UC Volume, or a SQL
  database (`docs/architecture/storage.md`)
- Deployment as a Databricks App or as a plain process on a laptop or VM
  (`docs/architecture/databricks-deployment.md`)
- Plugin system for custom analysis extensions
- Export of investigation results in JSON, CSV, XML, and Excel formats, plus
  Markdown and PDF reports and raw/transformed evidence sidecars
- HTTP API for submitting incidents, streaming progress, answering gates, and
  reading results (`docs/API.md`, `docs/openapi.yaml`)
- Batch submission and a FIFO run queue: a backlog is submitted in one request,
  drains at a configured width, refuses rather than over-accepting, and reports
  each run's position (`src/job_queue.py`,
  `docs/architecture/run-modes.md`)
- Rate limiting, input validation, LLM throttling, and retry with backoff

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/AmadeusITGroup/afir.git
   cd afir
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   To run the test suite or the formatters, install the development set instead —
   it starts with `-r requirements.txt`, so it is a superset:
   ```
   pip install -r requirements-dev.txt
   ```
   **Why two files.** Databricks Apps installs `requirements.txt` by that exact
   filename, with no way to point it elsewhere, so that file *is* the deployed
   runtime environment. Anything only a developer needs — pytest, black, flake8,
   isort, jupyter, and the optional local embedding model with its torch
   dependency — has to stay out of it, or every deployment pays for it in image
   size and install time. `requirements-dev.txt` is where those live.

4. Copy the configuration templates and fill in your values:
   ```
   cp config/templates/main_config.yaml    config/main_config.yaml
   cp config/templates/llm_config.yaml     config/llm_config.yaml
   cp config/templates/logging_config.yaml config/logging_config.yaml
   cp config/templates/plugin_config.yaml  config/plugin_config.yaml
   ```
   At minimum, set an LLM endpoint in `config/llm_config.yaml` (`base_url`,
   `api_key_env`, `model`), at least one log-source backend in
   `config/main_config.yaml` (`log_sources.backends`), and the knowledge pack to
   use (`knowledge.pack_dir`). The pack `knowledge/mock_domain` ships with the
   repository and runs end to end without credentials.

   Real `config/*.yaml` files are gitignored. See [docs/configuration.md](docs/configuration.md)
   for a full reference.

5. Start the main system:
   ```
   python app.py
   ```
   It works from any working directory, and serves the web UI, the HTTP API and
   the OpenAPI docs (`/docs`) from one port.

6. Open the UI at `http://localhost:5000/` (the port is
   `incident_input.port`; a Databricks App overrides it with
   `$DATABRICKS_APP_PORT`), or submit incidents over the API as
   described in [docs/user_guide.md](docs/user_guide.md) and
   [docs/API.md](docs/API.md). Use `POST /api/v1/jobs` — a real run takes tens of
   minutes, so the older blocking endpoints answer `501` behind an ingress that
   cannot wait that long.

## Extending the System

### Adding Plugins

1. Create a new Python file in the `plugins/` directory.
2. Implement your plugin logic and a `register_plugin()` function that returns a
   dict with `name` and an async `execute`.
3. Add the module name to `active_plugins` in `config/plugin_config.yaml`.
4. The plugin will be automatically loaded by the PluginManager.

### Adding a Domain

Domain knowledge is not code. Copy `docs/knowledge-pack-template/` (or
`knowledge/mock_domain/`), fill in the entity glossary, source catalog,
playbooks, verdict rulesets and concept docs, point `knowledge.pack_dir` at it,
and check it with:

```
python -m src.knowledge.pack_validate knowledge/<domain>
```

The engine stays generic — a test enforces that no domain literal appears under
`src/`. See [docs/architecture/knowledge-pack-authoring.md](docs/architecture/knowledge-pack-authoring.md)
for every declarable key and what breaks when it is absent, and the Knowledge tab
in the UI for editing a pack in place.

### Deploying as a Databricks App

`app.yaml` and `databricks.yml` ship with the repository. The platform installs
`requirements.txt`, resolves credentials through the App's OAuth service
principal, and reaches the workspace's own model-serving endpoints, so no API key
is needed. Durable state should be pointed at a UC Volume or a SQL database,
since container disk does not survive a restart. See
[docs/architecture/databricks-deployment.md](docs/architecture/databricks-deployment.md).

## Testing

The suite needs `requirements-dev.txt` installed. No test calls a real LLM or a
real backend.

```
pytest                                              # everything
pytest tests/test_main.py::test_process_incident    # one test
pytest -k retrieval                                 # by pattern
pytest --cov=src                                    # with coverage
```

## Logging

The system uses Python's built-in logging module. Log output format is controlled
by `logging.format` in `config/main_config.yaml` (`console` or `json`).
`AFIR_LOG_FORMAT` and `AFIR_LOG_LEVEL` environment variables override these values
at process start.

## Documentation

| Document | Contents |
|---|---|
| [docs/setup.md](docs/setup.md) | Full install and run walkthrough |
| [docs/user_guide.md](docs/user_guide.md) | Submitting incidents, using the UI, gates, reports |
| [docs/configuration.md](docs/configuration.md) | All configuration keys, section by section |
| [docs/API.md](docs/API.md) | HTTP API reference with curl examples |
| [docs/openapi.yaml](docs/openapi.yaml) | OpenAPI specification, also served at `/openapi.json` and `/docs` |
| [docs/architecture/knowledge-pack-authoring.md](docs/architecture/knowledge-pack-authoring.md) | Authoring a domain pack: every declarable key |
| [docs/architecture/verdict-engine.md](docs/architecture/verdict-engine.md) | Rulesets, condition kinds, how a verdict is reached |
| [docs/architecture/retrieval.md](docs/architecture/retrieval.md) | The four backends, query shaping, timeouts and row caps |
| [docs/architecture/hitl.md](docs/architecture/hitl.md) | Gates, run modes, overrides, the feedback loop |
| [docs/architecture/databricks-deployment.md](docs/architecture/databricks-deployment.md) | Deploying as a Databricks App |
| [docs/architecture/](docs/architecture/) | Design documents for every subsystem, including storage, RAG embeddings, report generation, extended thinking, run modes, correlation and the link lane |

## License

Apache License 2.0. See the `LICENSE` file.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first —
it covers the code and testing conventions, the deliberate formatting posture
(there is no repo-wide `flake8`/`black` config, so format only what you touched),
and how a measurement is cited. Participation is governed by
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

To report a vulnerability, see [SECURITY.md](SECURITY.md) — please do not open a
public issue for one.
