# AFIR Setup Guide

This guide walks through installing and running AFIR from a fresh clone.

## Prerequisites

- Python 3.10
- Access to an OpenAI-compatible LLM endpoint (Databricks Model Serving, OpenAI, or
  any compatible service)
- At least one supported log-source backend (Elasticsearch, Kibana gateway,
  Databricks SQL, Snowflake, or REST), or use the included `knowledge/mock_domain`
  pack which runs end to end without credentials

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<domain>ITGroup/afir.git
cd afir
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install runtime dependencies

```bash
pip install -r requirements.txt
```

To run the test suite, also install development dependencies (this installs
`requirements.txt` first):

```bash
pip install -r requirements-dev.txt
```

### 4. Create configuration files

Only the template files under `config/templates/` are tracked in git. Real
configuration files go in `config/` and are gitignored (they hold credentials).
Copy the templates before starting:

```bash
cp config/templates/main_config.yaml    config/main_config.yaml
cp config/templates/llm_config.yaml     config/llm_config.yaml
cp config/templates/logging_config.yaml config/logging_config.yaml
cp config/templates/plugin_config.yaml  config/plugin_config.yaml
```

### 5. Configure the LLM endpoint (`config/llm_config.yaml`)

At minimum, set:

```yaml
base_url: "https://<your-workspace-host>/serving-endpoints"
api_key_env: "DATABRICKS_TOKEN"   # name of the env var holding the token
model: "<your-served-model-name>"
```

- `base_url`: leave blank to have the Databricks SDK resolve its own workspace
  automatically (correct for Databricks Apps and local runs authenticated via the
  SDK). Set it explicitly for OpenAI or a specific Databricks workspace.
- `api_key_env`: the **name** of an environment variable holding the token. Never
  paste a literal token into the config file. Export the variable before running:
  ```bash
  export DATABRICKS_TOKEN="<your-token>"
  ```

### 6. Configure at least one log-source backend (`config/main_config.yaml`)

The `log_sources.backends` section maps backend credentials to the logical
endpoints that the knowledge pack's source catalog references. The supported kinds
are:

| Kind | Notes |
|---|---|
| `elasticsearch` | Direct ES node (ES\|QL) or Kibana gateway (Query DSL, set `gateway: kibana`) |
| `databricks` | Databricks SQL Statement Execution API |
| `snowflake` | Snowflake SQL |
| `rest` | Generic REST service |

Each entry is keyed by the `endpoints.cluster` (or `workspace`, `account`, or
`service`) value declared in the pack's `source_catalog.yaml`. Secrets use
environment variable references (`${VAR}`), never literal values:

```yaml
log_sources:
  backends:
    elasticsearch:
      my-cluster:
        url: "https://<es-host>:9200"
        username: "${ES_USER}"
        password: "${ES_PASSWORD}"
```

See [configuration.md](configuration.md) for the full key reference for each
backend kind.

### 7. Configure the knowledge pack

The `knowledge.pack_dir` key in `config/main_config.yaml` names the subdirectory
under `knowledge/` (or under `$AFIR_KNOWLEDGE_DIR`) to load:

```yaml
knowledge:
  pack_dir: "mock_domain"    # or your own pack name
```

`knowledge/mock_domain` ships with the repository and runs end to end without
backend credentials, making it suitable for verifying the install. Copy it to
`knowledge/<your-domain>/` and fill in the source catalog and rules for your own
domain.

## Running the application

Start the server from the repository root:

```bash
python app.py
```

Do not use `python src/main.py` — paths are anchored to the repository root and
launching from `src/` will not resolve them correctly.

The server binds to `0.0.0.0:5000` by default. Both the port and the bind address
are configurable under `incident_input.host` and `incident_input.port` in
`config/main_config.yaml`.

### Verify the server is running

```bash
curl -s http://localhost:5000/health
# -> {"status": "ok"}
```

The deep health check reports whether storage, source backends, and the knowledge
pack are all reachable:

```bash
curl -s "http://localhost:5000/health?deep=1" | python -m json.tool
```

### Open the web UI

Navigate to `http://localhost:5000/` in a browser. The UI is a single inline HTML
document (no CDN, works offline in a Databricks App) with five tabs: Investigate,
Monitor, Report, Configuration, and Knowledge.

## Running the tests

```bash
pytest
```

Run a single test:

```bash
pytest tests/test_main.py::test_process_incident
```

Run all tests matching a pattern:

```bash
pytest -k "retrieval"
```

Run with coverage:

```bash
pytest --cov=src
```

`pytest.ini` sets `pythonpath = src` so that flat imports in `src/main.py` and
package-qualified imports in `tests/` both resolve correctly.

## Environment variable overrides

The following environment variables redirect the paths the application uses at
runtime. They are useful for deployment and for pointing an existing installation
at different directories without editing config files:

| Variable | Overrides | Default |
|---|---|---|
| `AFIR_CONFIG_DIR` | Directory holding `*.yaml` config files | `<repo-root>/config` |
| `AFIR_DATA_DIR` | Writable base for exports and generated artifacts | `<repo-root>` |
| `AFIR_KNOWLEDGE_DIR` | Parent directory of knowledge packs | `<repo-root>/knowledge` |
| `AFIR_LOG_FORMAT` | Log output format (`console` or `json`) | value in `main_config.yaml` |
| `AFIR_LOG_LEVEL` | Log level (`DEBUG`, `INFO`, `WARNING`, …) | value in `main_config.yaml` |

The `knowledge.pack_dir` key in `main_config.yaml` can reference `$AFIR_PACK_DIR`
to choose the pack name at deploy time without modifying the config file
(`${AFIR_PACK_DIR:-mock_domain}` is the shipped default).

## Secrets handling

Credentials are never stored literally in configuration files. Every secret-bearing
key (`password`, `api_key`, private keys, DSN connection strings) must reference an
environment variable by name:

```yaml
password: "${MY_SECRET_ENV_VAR}"
```

The application expands `${VAR}` references from the environment at load time. An
unresolvable variable expands to an empty string and the relevant source or backend
is skipped with a log warning (graceful degrade), not a fatal error.

The configuration API redacts literal values from GET responses and refuses to echo
them back. The one guarantee that does not depend on an ingress is that literal
secrets never leave the server process.

## Troubleshooting

- **"No module named 'src'"** — run `python app.py` from the repository root, not
  from `src/`. The `pythonpath = src` in `pytest.ini` applies only to the test
  runner.
- **LLM calls return 401 or 403** — verify the token in the referenced env var is
  valid for the workspace named in `base_url`. A PAT issued for workspace A will
  reject requests to workspace B.
- **A source returns 0 rows** — check `GET /health?deep=1` for
  `sources_unavailable`. A backend whose credentials are missing is skipped
  silently; the verdict conditions over that source will report `unknown`.
- **Server starts but the web UI is blank** — the UI is served at `/`; make sure
  the browser is not hitting a proxy that strips the HTML response.
