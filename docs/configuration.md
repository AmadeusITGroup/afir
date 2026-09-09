# AFIR Configuration Guide

Configuration lives in YAML files under `config/`. Only the template files in
`config/templates/` are tracked in git; real files (`config/*.yaml`) are gitignored
because they hold credentials. Copy the templates before the first run:

```bash
cp config/templates/main_config.yaml    config/main_config.yaml
cp config/templates/llm_config.yaml     config/llm_config.yaml
cp config/templates/logging_config.yaml config/logging_config.yaml
cp config/templates/plugin_config.yaml  config/plugin_config.yaml
```

The four files are editable from the **Configuration** tab in the web UI without
restarting the server (for `live` keys) or after a restart (for `restart` keys).
Literal secret values are never echoed back by the UI or the API: a field holding
an `${ENV_VAR}` reference is shown as-is; a literal value under a secret-shaped key
is replaced with `__redacted__` on read and that placeholder is accepted on write to
mean "leave the stored value alone."

## Two rules that bite operators

**Live vs restart.** Each key is marked `live` or `restart` in the source
(`src/config_store.py`). A `live` key is re-read from the config dict on every use,
so writing it from the UI takes effect on the next stage that reads it with no
restart. A `restart` key is consumed once during `main()` startup; writing it takes
effect only after the server is restarted.

**Dotted path must resolve through direct children.** The config patcher locates a
key by walking a dotted path (`section.subsection.key`) and rewriting the scalar it
finds. Every segment must name a direct child of the previous segment. A key that
lives deeper than the path expects is rejected rather than silently targeting a
same-named key at a different depth, because a patch that half-applies is worse than
one that fails.

---

## `llm_config.yaml` — LLM endpoint

This file configures the single LLM endpoint the pipeline uses for all stages.

| Key | Type | Applies | Notes |
|---|---|---|---|
| `base_url` | string | restart | OpenAI-compatible endpoint URL. Leave blank for Databricks Apps (the SDK resolves the workspace automatically). |
| `api_key_env` | string | restart | **Name** of the environment variable holding the token. Never paste the token here. |
| `model` | string | restart | Model name served at the endpoint. |
| `max_tokens` | integer | restart | Output cap for prose responses. |
| `structured_output_max_tokens` | integer | restart | Output cap for JSON-schema responses (understanding, queries, correlation, anomalies). Raise this if a stage reports a required field missing; a truncated schema response presents as a missing field, not as a truncation. Default 8000. |
| `temperature` | number | restart | Sampling temperature. Default 0.2. |
| `timeout` | integer | restart | Per-request timeout in seconds. Default 180. |
| `max_concurrency` | integer | restart | Cap on in-flight LLM requests. Lower this if the endpoint returns 429 errors. Default 4. |
| `requests_per_minute` | integer | restart | Token-bucket QPS limit. 0 disables it. Default 60. |
| `thinking` | choice | live | Extended thinking mode: `disabled` (default), `adaptive`, or `unset` (endpoint decides). |
| `thinking_effort` | choice | live | Only read when `thinking: adaptive`. `low`, `medium`, or `high`. |
| `thinking_by_stage` | mapping | live | Per-stage overrides: `mode`, `effort`, and `max_tokens` floor. Recognised stages: `incident_understanding`, `api_call_generation`, `log_retrieval`, `correlation`, `anomaly_detection`, `report_generation`, `pack_assistant`, `feedback_distillation`. |
| `context` | string | restart | Static system context injected when `rag.use_rag` is `false`. Leave blank to send no extra context. |

### Minimal example

```yaml
base_url: "https://<workspace-host>/serving-endpoints"
api_key_env: "DATABRICKS_TOKEN"
model: "my-served-model"
max_tokens: 4096
structured_output_max_tokens: 8000
temperature: 0.2
timeout: 180
thinking: disabled
```

---

## `main_config.yaml` — main system configuration

### `incident_input`

Controls how the server accepts incidents and binds its network socket.

| Key | Default | Notes |
|---|---|---|
| `blocking_endpoints` | `auto` | Whether `POST /api/v1/incidents`, `/freetext`, and `/ir` are active. `auto` enables them except on a Databricks App; `on` and `off` override. |
| `host` | `0.0.0.0` | Bind address. |
| `port` | `5000` | Listen port. Overridden by `$DATABRICKS_APP_PORT` on a Databricks App. |
| `rate_limit.requests` | `1000` | Maximum requests per window. |
| `rate_limit.per_seconds` | `3600` | Window duration in seconds. |
| `win_url`, `win_username`, `win_password` | — | Optional Win@proach retrieval credentials (password via `${ENV_VAR}`). |

### `log_sources`

Configures retrieval backends. The `backends` section is the primary way to wire up
credentials. The `sources` list (explicit full connection blocks) is a legacy
override that wins by name over a same-named pack source.

#### Timeouts and row caps

| Key | Applies | Default | Notes |
|---|---|---|---|
| `per_source_timeout_seconds` | live | 20 | Time cap for an ordinary source. A source the pack marks `retrieval_class: primary` uses the budget below instead. |
| `primary_source_timeout_seconds` | live | 7200 | Budget for primary sources. Set it above the worst measured cost, not just the typical one: a cap that fires here decides the verdict rather than degrading it (every decisive condition goes `unknown` while the stage still reports success). |
| `extended_retrieval` | live | false | Global opt-in to extend time budgets for slow sources. Also settable per-incident (`"extended_retrieval": true` in the job request). |
| `extended_retrieval_timeout_seconds` | live | 4x normal | The cap applied when extended retrieval is on. Never lowers a source's normal cap. |
| `default_lookup_days` | live | unset | Days to look back when the incident text states no date or time at all, counted from the incident's ingestion timestamp. The pack's `retrieval.default_lookup_days` wins over this. Leave unset only if the pack always declares a window — unset with a pack that also declares none caused 29 queries to carry eight different windows on one measured run. |
| `max_results` | restart | 500 | Row cap per source (fallback; a per-endpoint `max_results` wins). A row cap and a time cap are separate limits: extended retrieval buys more time, not more rows. |

#### `cache`

Off unless declared, and a deployment that declares nothing behaves exactly as it did
before this block existed. What is cached is one source's *answer*, keyed on the question
that was asked — the source, the window, the entities, the row cap and any analyst
guidance — never on the generated backend query, which does not exist until the retriever
has run. Only what the run actually worked from is stored, so a timeout, a backend error,
a cancel and an empty result from an unfilled `'<...>'` placeholder can never be cached:
all four are non-answers and are absent from the retrieval result by design.

| Key | Applies | Default | Notes |
|---|---|---|---|
| `enabled` | restart | false | Switch. Built once with the retrieval engine, hence `restart`. |
| `ttl_seconds` | restart | 3600 | How long a non-empty answer stays usable. Long enough to cover a re-run, a rejected gate and a child run of the same incident; short enough that a source repaired mid-shift is asked again without an operator having to know the cache exists. |
| `empty_ttl_seconds` | restart | 0 | How long an *empty* answer stays usable. Zero means empty answers are never kept: zero rows from a keyed lookup is a finding, but it is also the answer a re-ask most often changes, and a cached empty is indistinguishable from a source that has nothing until it expires. |
| `max_entries` | restart | 64 | Entries kept before the least recently used is dropped. |
| `max_rows` | restart | 200000 | Rows kept across all entries. A single answer larger than this is refused rather than admitted, because admitting it would evict every other entry to hold one truncated source. |

A served answer says so on the source's own status line — `Retrieved N rows … (cached 4m
ago, not re-queried)` — and `/health?deep=1` reports `retrieval_cache` with the hit and
miss counts beside the rate. The cache is per process and not durable: it holds result
sets, one replica does not see another's, and a restart empties it.

#### `backends`

The backends section maps credentials to the logical endpoint names the pack's
`source_catalog.yaml` declares. Each kind has its own keying convention.

**Elasticsearch** (keyed by `endpoints.cluster` in the pack):

```yaml
backends:
  elasticsearch:
    my-cluster:
      url: "https://<es-host>:9200"   # direct ES node, or Kibana gateway host
      # gateway: kibana               # uncomment if the host is a Kibana gateway
      username: "${ES_USER}"
      password: "${ES_PASSWORD}"
      timeout: 30
      max_results: 500
      # verify_ssl: false             # dev only; prefer ca_bundle in production
      # ca_bundle: "/path/to/ca.pem"
```

For a Kibana gateway (`gateway: kibana`), the URL is the Kibana host without a
port; queries route through `/internal/search/es` using Query DSL. Omit `gateway`
for a direct ES node; queries use ES|QL.

**Databricks** (keyed by `endpoints.workspace` in the pack):

```yaml
backends:
  databricks:
    my-workspace:
      workspace_url: "https://<workspace-host>"  # blank = SDK-resolved host
      warehouse_id: "<16-hex-sql-warehouse-id>"  # NOT the workspace/org id from the URL
      api_key_env: "DATABRICKS_TOKEN"
      max_results: 500
      statement_timeout_seconds: 1800
      retrieval_timeout_seconds: 1800            # must match statement_timeout_seconds
      poll_interval_seconds: 5
      max_poll_attempts: 360
      max_consecutive_poll_errors: 5
```

A PAT is workspace-scoped: a token issued for workspace A returns 403 on workspace B.
For multiple workspaces, add one entry per workspace, each with its own
`api_key_env`. The flat form (no per-workspace keys) is still supported for packs
that declare no `workspace`.

**Snowflake** (keyed by `endpoints.account` in the pack):

```yaml
backends:
  snowflake:
    my-account:
      account: "<account-id>"        # e.g. xy12345.eu-west-1
      user: "${SNOWFLAKE_USER}"
      password: "${SNOWFLAKE_PASSWORD}"
      # private_key_env: "SNOWFLAKE_PRIVATE_KEY"  # alternative: key-pair auth
      role: "<role>"
      warehouse: "<warehouse>"
      max_results: 500
```

**REST** (keyed by `endpoints.service` in the pack):

```yaml
backends:
  rest:
    my-service:
      base_url: "https://<service-host>"
      username: "${REST_USER}"
      password: "${REST_PASSWORD}"
      # token_env: "REST_TOKEN"       # alternative: Bearer token
      timeout: 30
      max_results: 500
```

#### SSH tunnel (optional)

```yaml
log_sources:
  use_ssh_tunnel: false
  tunnel:
    url: "<tunnel-host>"
    user: "<tunnel-user>"
    password: "${TUNNEL_PASSWORD}"
    remote_bind_url: "remote.bind.url"
    remote_bind_port: 9000
    local_bind_url: "local.bind.url"
    local_bind_port: 9000
```

### `knowledge`

| Key | Default | Notes |
|---|---|---|
| `pack_dir` | `mock_domain` | Subdirectory under `knowledge/` (or `$AFIR_KNOWLEDGE_DIR`) containing the domain pack. Set `$AFIR_PACK_DIR` in the environment to override without editing this file (the template ships `${AFIR_PACK_DIR:-mock_domain}`). |

The domain pack holds all domain-specific knowledge: entity glossary, source
catalog, investigation playbooks, and verdict rules. Swap `pack_dir` to target a
different domain; the engine itself stays unchanged.

### `correlation`

Controls the correlation and verdict stage.

| Key | Applies | Default | Notes |
|---|---|---|---|
| `sample_rows` | live | 20 | Rows per source shown to the LLM when it narrates. |
| `llm_max_records` | live | 2000 | Volume gate: above this the deterministic plan is used and the LLM is skipped. |
| `llm_max_sources` | live | 6 | Source gate for the same decision. |
| `discovery_key_filter` | live | `strict` | How aggressively join keys are discovered when no playbook overrides: `strict`, `cardinality`, or `both`. |
| `evidence_char_budget` | live | 80000 | Character budget for the evidence pack fed to anomaly detection and report generation. The pack degrades (aggregates harder) to fit this before reaching the LLM stages, so a budget below what a run actually renders costs adjudicated values and scores `evidence_clipped`. A ceiling, not a target: a full 41-source investigation renders at ~37,000 chars and a small one costs far less. The engine's own fallback where the key is absent stays 15,000. |

#### `correlation.links`

The advisory cross-procedure link lane. A pack that declares no `entry_signals`
produces an empty list and a byte-identical report. Both paid rungs ship armed and
narrow (`max_probes_per_run: 2`, `max_children_per_run: 1`); each is disarmed on its
own by setting its budget to `0`, which keeps the referral.

| Key | Applies | Default | Notes |
|---|---|---|---|
| `enabled` | live | true | Whether the link lane runs at all. |
| `min_signal_strength` | live | 0.0 | Floor on signal strength; 0.0 keeps every declared signal. |
| `advisory_severity_default` | live | `MEDIUM` | Severity label carried by a confirmed link (separate from the verdict's severity). |
| `escalation_mode` | live | `semi_auto` | `planned`, `semi_auto`, or `auto`. A link escalates only where the target procedure's own scope gate also passes on this run's evidence. |
| `min_escalation_score` | live | 0.6 | The deterministic link score at or above which `semi_auto` acts by itself. 0.0 makes `semi_auto` behave like `auto`. |
| `max_probes_per_run` | live | 2 | Rung-3 budget: candidate-confirmation queries. A probe also needs an escalating mode, a rung-1 PASS, and `auto_probe: true` on the declaration. 0 disables this rung entirely. |
| `probe_timeout_seconds` | live | 120 | Seconds one probe may run. |
| `probe_row_cap` | live | 200 | Rows a probe may return. A probe returning exactly this many is reported as truncated. |
| `max_children_per_run` | live | 1 | Rung-4 budget: full child runs of a sibling procedure. The most expensive rung, so it ships at one; 0 disables it and keeps the referral. |
| `max_child_depth` | live | 1 | How many referral steps deep a lineage may go. |
| `max_concurrent_children` | live | 1 | Child runs in flight at once. |
| `max_total_children` | live | 8 | Total child runs this process may launch; the runaway backstop. |

### `anomaly_detection`

| Key | Applies | Default | Notes |
|---|---|---|---|
| `use_llm` | live | true | Whether the LLM stage runs. |
| `threshold` | live | 0.8 | Confidence floor for reporting an anomaly. This is the one value `feedback.auto_tune_threshold` may adjust, bounded against this declared baseline. |
| `max_anomalies` | live | 50 | Maximum anomalies to keep. |

### `report_generation`

| Key | Applies | Default | Notes |
|---|---|---|---|
| `output_format` | live | `pdf` | `pdf` or `txt`. Markdown and PDF are always written regardless. |
| `max_anomalies_in_prompt` | live | 15 | Top-N anomalies by confidence sent to the LLM. All anomalies appear in the exports regardless. |
| `llm_input_char_budget` | live | 15000 | Correlation JSON is truncated to this many characters before the LLM stage. |
| `report_min_tokens` | live | 4000 | Floor on the LLM output budget so the fixed report sections always fit. |
| `report_max_tokens` | live | 12000 | Ceiling on the LLM output budget. `max_tokens` is a cap, not a reservation. |

### `output_interface`

Configures how the finished report is delivered. The `type` key selects the channel.

| `type` value | Description |
|---|---|
| `email` | Send via SMTP. Configure `smtp_server`, `smtp_port`, `smtp_username`, `smtp_password` (via `${ENV_VAR}`), `sender_email`, `recipients`, and `cc`. |
| `file` | Write to the configured output path. |
| `api` | POST to an external endpoint. |

### `stage_gates`

Controls when the run pauses for human approval.

| Key | Applies | Default | Notes |
|---|---|---|---|
| `threshold` | live | 0.6 | Gate a stage whose health score is below this. Health is deterministic (never self-assessed by the LLM); calibrate against your own runs. |
| `timeout_seconds` | live | 0 (no timeout) | 0 or unset means hold forever; a timeout is opt-in and its expiry is recorded as a clock-made decision. |
| `on_timeout` | live | `hold` | `hold` (notify once, keep waiting), `proceed` (continue, recorded as not reviewed), or `abort`. |
| `stages.<name>.enabled` | live | true | Whether this stage may open a gate at all. Off = stage never pauses even in `supervised` mode. |
| `stages.<name>.threshold` | live | global | Per-stage threshold override. |

The six gateable stages are `understanding`, `query_generation`, `log_retrieval`,
`correlation`, `anomaly_detection`, and `report_generation`.

### `webhooks`

Outbound push notifications. Delivery is best-effort (bounded retries, background
POST); a failure degrades to a log line, never stalls a run.

| Key | Default | Notes |
|---|---|---|
| `enabled` | false | Master switch. |
| `timeout_seconds` | 10 | Per-delivery timeout. |
| `max_attempts` | 3 | Retry limit. |
| `targets` | `[]` | List of webhook targets. Each needs `name`, `url` (via `${ENV_VAR}` — a webhook URL is a bearer credential), and optionally `events` and `headers`. |

Subscribable events: `gate_opened`, `gate_resolved`, `gate_timeout`,
`job_completed`, `job_cancelled`, `stage_failed`.

### `storage`

Where durable state lives (job documents, evidence sidecars, exports, feedback
log). One decision for the whole process; not a property of the deployment mode.

| Key | Applies | Default | Notes |
|---|---|---|---|
| `backend` | restart | `local` | `local`, `databricks`, or `sql`. |
| `root` | restart | `$AFIR_DATA_DIR` | Only for `backend: local`. A filesystem path; `/Volumes/...` is not mounted in an App container, so pointing at it reports success while losing state on restart. |
| `mirror_config_and_pack` | restart | `auto` | `auto` (mirror when backend is remote), `always`, or `never`. |

**`storage.databricks`** (Unity Catalog Volume over the Files API):

```yaml
storage:
  backend: "${AFIR_STORAGE_BACKEND:-local}"
  databricks:
    catalog: "${AFIR_UC_CATALOG:-}"
    schema: "${AFIR_UC_SCHEMA:-afir}"
    volume: "${AFIR_UC_VOLUME:-state}"
    host: ""           # blank = SDK-resolved workspace
    token_env: DATABRICKS_TOKEN
    verify_ssl: true
```

**`storage.sql`** (SQLite or PostgreSQL; the only option safe for multiple replicas):

```yaml
storage:
  backend: sql
  sql:
    dialect: "${AFIR_SQL_DIALECT:-sqlite}"
    dsn_env: AFIR_SQL_DSN   # env var holding the connection string; preferred over dsn
    dsn: ""                 # explicit path (sqlite) or libpq URL (postgresql)
    table: afir_blobs
```

Do not use a Databricks SQL warehouse or Snowflake here: Databricks SQL caps
combined statement parameters at 1 MiB (evidence sidecars reach 2.07 MB in
production) and Snowflake cannot concatenate binary columns atomically.

### `jobs`

| Key | Default | Notes |
|---|---|---|
| `persist` | true | Write job state to disk. |
| `max_evidence_mb` | 64 | Evidence sidecar size cap; evidence is dropped (and the omission recorded) above this limit. |
| `retention_days` | 14 | How long job files are kept. |
| `summary_max_items` | 30 | Items kept in per-stage summary lists shown in the UI and gates. Raise for incidents naming many entities or sources. |
| `max_retrieval_passes` | 3 | Ceiling on query-generation + retrieval iterations. A runaway guard; set to 1 to refuse follow-up passes entirely. |
| `max_concurrent_jobs` | 2 | Submitted runs executing stages at once; the rest wait in a FIFO backlog. Read on every admission, so raising it in the Configuration tab applies to the next one rather than the next restart. Engine ceiling 8: past that the shared LLM semaphore is what actually orders the work, and twenty runs at once do not finish sooner — they all finish slower, ordered by a queue nobody can see. |
| `max_queued_jobs` | 256 | Submissions that may wait. Past this `POST /api/v1/jobs` answers **429** rather than accepting, because a job id for a run that may never start reads exactly like a run that is merely slow. Engine ceiling 4096. |

Both bounds cover *submitted* runs only. A human's resume, a stage retry and a link lane's
child run all start outside the width — each is a decision already taken. An unreadable
value (a typo, a bool) falls back to the default rather than to the most permissive
setting, and `/health?deep=1` reports the live counters under `run_queue`.

### `feedback`

| Key | Applies | Default | Notes |
|---|---|---|---|
| `batch_size` | live | 10 | Reviews required before a distillation run. |
| `apply_to_prompts` | live | true | Whether distilled insights are injected into prompts. Off = collect reviews without steering the LLM. |
| `auto_tune_threshold` | restart | false | Whether the anomaly-detection threshold is auto-tuned from review data. Off by default; the recommendation is still computed and readable at `GET /api/v1/feedback/threshold`. |
| `threshold_step` | restart | 0.05 | Maximum move per adjustment. |
| `min_reviews_for_tuning` | restart | 5 | New reviews required since the last adjustment. |
| `max_threshold_drift` | restart | 0.15 | Maximum total distance from the configured baseline. |
| `threshold_min` / `threshold_max` | restart | 0.3 / 0.95 | Hard floor and ceiling. |

### `rag`

Controls the retrieval-augmented generation layer that retrieves playbook context
for each stage.

| Key | Applies | Default | Notes |
|---|---|---|---|
| `use_rag` | restart | true | Off = keyword fallback only; the `context` key in `llm_config.yaml` is used instead. |
| `embedding_provider` | restart | `databricks` | `databricks` (Model Serving endpoint) or `sentence_transformers` (local model). |
| `embedding_model` | restart | `databricks-qwen3-embedding-0-6b` | Endpoint name (Databricks) or model id (sentence-transformers). |
| `embedding_host` | restart | blank | Blank = same workspace the LLM auth resolves to. |
| `embedding_token_env` | restart | blank | Env var holding a PAT for the embedding endpoint. Blank = unified auth. |
| `max_retrieved_documents` | restart | 5 | Documents returned per RAG query. |
| `similarity_threshold` | restart | 0.5 | Minimum cosine similarity for a document to be returned. |
| `embedding_batch_size` | restart | 100 | Inputs per encode request; serving endpoints refuse above 150. |

The index records its `provider:model` signature. A model swap costs one boot to
rebuild the index; it never searches in the wrong vector space.

The `rag.sources` list declares where knowledge documents come from:

```yaml
rag:
  sources:
    - type: playbook        # the pack's playbooks/*.md (always declare this first)
      name: playbook
    - type: pack_schema     # per-table field inventories from schemas/*.yaml
      name: pack_schema
    - type: document        # local folders of pdf/docx/txt/csv/json
      name: local_documents
      enabled: false
      paths: []
    - type: confluence
      name: confluence
      enabled: false
      url: ""
      username: ""
      password: ""
      spaces: []
    - type: databricks_table
      name: db_knowledge
      enabled: false
      table: "catalog.schema.articles"
      title_col: "title"
      content_col: "body"
```

### `logging`

| Key | Default | Notes |
|---|---|---|
| `format` | `console` | `console` (aligned human-readable text) or `json` (one JSON object per line with structured fields for querying). `AFIR_LOG_FORMAT` overrides at runtime. |
| `level` | `INFO` | Minimum log level. `AFIR_LOG_LEVEL` overrides at runtime. |
| `levels` | `{}` | Per-logger overrides. Named chatty third-party loggers are raised to `WARNING` automatically; list one here to get its HTTP detail back when debugging. |

---

## `logging_config.yaml`

A standard Python logging configuration dictionary. The entries under `formatters`,
`handlers`, and `loggers` follow the `logging.config.dictConfig` schema. The
pipeline's own logs are also influenced by `logging.format` in `main_config.yaml`;
`logging_config.yaml` is for fine-grained handler and formatter control beyond what
the simple format/level keys expose.

---

## `plugin_config.yaml`

| Key | Notes |
|---|---|
| `active_plugins` | List of module names (without `.py`) to load from the `plugins/` directory. |
| `plugin_settings.<name>` | Per-plugin settings dict passed to the plugin at load time. |

```yaml
active_plugins:
  - my_plugin

plugin_settings:
  my_plugin:
    setting_key: value
```

Plugins are loaded dynamically at startup. A plugin module that cannot be imported
is skipped with a log warning.

---

## Environment variable reference

| Variable | Purpose |
|---|---|
| `AFIR_CONFIG_DIR` | Override the config directory (default: `<repo-root>/config`). |
| `AFIR_DATA_DIR` | Override the writable artifacts directory (default: `<repo-root>`). |
| `AFIR_KNOWLEDGE_DIR` | Override the parent directory of knowledge packs (default: `<repo-root>/knowledge`). |
| `AFIR_PACK_DIR` | Pack name to load; referenced as `${AFIR_PACK_DIR:-mock_domain}` in the template. |
| `AFIR_STORAGE_BACKEND` | Override `storage.backend` for containers (default: `local`). |
| `AFIR_STATE_DIR` | Override `storage.root`. |
| `AFIR_UC_CATALOG` | Databricks storage catalog. |
| `AFIR_UC_SCHEMA` | Databricks storage schema (default: `afir`). |
| `AFIR_UC_VOLUME` | Databricks storage volume (default: `state`). |
| `AFIR_SQL_DIALECT` | SQL storage dialect (`sqlite` or `postgresql`, default: `sqlite`). |
| `AFIR_SQL_DSN` | Connection string for SQL storage. |
| `AFIR_LLM_BASE_URL` | LLM base URL; referenced as `${AFIR_LLM_BASE_URL:-}` in the template. |
| `AFIR_LLM_MODEL` | LLM model name; referenced as `${AFIR_LLM_MODEL:-...}` in the template. |
| `AFIR_LOG_FORMAT` | Override `logging.format` (`console` or `json`). |
| `AFIR_LOG_LEVEL` | Override `logging.level`. |
| `DATABRICKS_APP_PORT` | Injected by the Databricks App platform; overrides the listen port. |
