# Deploying AFIR as a Databricks App

Status: **in progress.** The storage seam and the remote backend
(`docs/architecture/storage.md`) are both written and under test; the config/pack mirror, the
bundle and the deploy are not. This file records the platform facts and the *measurements*
the design rests on, so nothing here is re-derived or re-guessed.

## The target workspace

**Deployment target: `<app-workspace>`, `https://adb-<app-workspace>.<n>.azuredatabricks.net`.**
Durable state and the LLM endpoint both live here — the App runs in the same workspace it
serves its model from, so once deployed the OAuth service principal covers the LLM with no
injected secret. The **log sources do not move**: they stay in `<reporting-warehouse>`
(`adb-<reporting-workspace>`) and `<analytics-warehouse>` (`adb-<analytics-workspace>`), and because **a PAT
is workspace-scoped** those still need their own tokens as secrets. Three workspaces, three
tokens — `APP_DATABRICKS_TOKEN`, `DATABRICKS_TOKEN`, `ANALYTICS_DATABRICKS_TOKEN`.

**Provisioned** (`scripts/provision_uc_state.py`, idempotent, 2026-08-06): schema
`<app-catalog>.afir` and MANAGED volume `…afir.state`. Identity
`<operator>@example.com` holds `ALL_PRIVILEGES` there via the workspace's own owners group.
The durable root is
`/Volumes/<app-catalog>/afir/state`.

**A workspace switch is two coupled values, not one.** `base_url` comes from
`llm_config.yaml` but `LLMClient._refresh_auth` overwrites `api_key` from `auth.token()`
before *every* call, so a `DatabricksAuth` resolving to a different workspace than `base_url`
403s with a token that is itself perfectly valid. Locally `try_build_auth()` returns `None`
(no ambient `DATABRICKS_HOST`) and the static `api_key_env` path applies; **in the App it will
not be None**, so the App's workspace and `base_url` must agree.

## Why the current posture does not survive the platform

Four verified facts, each of which breaks something that works locally:

1. **`/Volumes` is not mounted in an App container.** `open('/Volumes/...')` raises
   `FileNotFoundError`; the Files API is the only path. So `job_store`'s `os.replace`
   cannot reach a Volume — and its writability probe *succeeds* against the container's
   `/tmp`, which `app.yaml` currently sets `AFIR_DATA_DIR` to. The loud fallback ERROR
   never fires and durability silently degrades to "lost on restart".
2. **The App filesystem is explicitly ephemeral** — "lost when the app restarts". Every
   export, the feedback log and the FAISS index live there today.
3. **The ingress caps a request at ~60–120s, not configurable.** Measured from
   `jobs/*.json`: median run 38 min, longest 476 min. The three blocking endpoints
   (`/api/v1/incidents`, `/freetext`, `/ir`) cannot complete behind it.
4. **`knowledge_pack_dir()` does not route through `data_dir()`** — it has its own
   `AFIR_KNOWLEDGE_DIR`, which `app.yaml` never sets, so pack writes target the read-only
   deployed bundle and every Knowledge-tab save fails.

## Decisions taken

| Question | Answer |
|---|---|
| Where does the pipeline execute? | **In-App, as background asyncio tasks.** A redeploy therefore kills in-flight runs; `job_store` restores them as PAUSED so nothing is silently lost, but a 38-minute run must be re-triggered. |
| Where does durable state live? | **Unity Catalog** — a Volume, over the Files API, and *only* a Volume: the measurement below is why the Delta index was dropped. |
| Where does the embedding model live? | **Nowhere — it is a serving endpoint.** `databricks-qwen3-embedding-0-6b` is the default in *every* deployment mode, so there is no 420 MB Volume sync and no torch cold start. A local model stays first-class for an air-gapped install, one config value away. Measured in `docs/architecture/rag-embeddings.md`. |
| Authorization? | **App `CAN_USE` permissions alone, no code change.** AFIR's HTTP API has no authentication of its own; anyone with `CAN_USE` can approve another analyst's gate or repoint a backend URL. Grant it narrowly. |

## Measurements

### The Statement Execution API parameter limit is real, 1 MiB, and loud

Measured against `<analytics-warehouse>` (2026-08-06):

| Parameter size | Result |
|---|---|
| 1 KB → 768 KB | OK, echoed length exact, ~1s round trip |
| 1024 KB | **HTTP 400 `INVALID_PARAMETER_VALUE`** — "the combined size of parameters is 1048585, exceeding the limit of 1048576 characters" |

Two things this settles. The ~1 MB figure was previously only an *inference* from the
connector repo; it is a stated server-side limit on the **combined** size of parameters.
And it **fails loudly** — a 400 with the exact number, not a silent truncation. That
matters more than the limit itself: a truncated parameter would have persisted a corrupt
job document while reporting success.

**The probe was wrong first, in the direction that flatters.** Its payload was
`"afir-storage-probe-" * 1000` — 19 KB — sliced to the target size, so every row above
19 KB measured 19 KB and the run reported ten passes and a ceiling of 8 MB. The repeat
count is now derived from the target size with an assertion, because a probe that cannot
build its own input measures nothing.

### Job documents are larger than the plan assumed

Measured over 31 real job documents and 20 evidence sidecars (2026-08-06):

| | median | p90 | max |
|---|---|---|---|
| Job document (sidecar excluded) | 322 KB | 512 KB | **599 KB** |
| Evidence sidecar | 1.06 MB | — | **2.07 MB** |

The plan said 522 KB max; the real figure is 599 KB. **So the parameterised-MERGE path is
not viable for documents.** 599 KB against a 768 KB verified ceiling is 1.28x margin, and
by this repo's own rule (`primary-cap-margin-is-the-real-budget`) the margin *is* the
budget — a document that grows past it fails to persist at exactly the moment a gate is
opening. Evidence sidecars exceed the ceiling outright.

**Therefore: the document itself goes to the Volume via the Files API, and Delta carries
only the index** — `job_id`, status, timestamps, gate state, and the pointer. That keeps
the listing/pruning queries cheap and puts no size ceiling anywhere on the path. The
largest document's composition, for whoever wonders whether it could just be trimmed
instead: `outputs` 324 KB, `event_history` 80 KB, `stage_summaries` 46 KB. None is
incidental — `event_history` is what the Monitor tab replays on re-attach.

### The Files API has no ceiling in play, and it is fast

Against the provisioned volume (2026-08-06), payloads built from a repeating *pattern* so a
value that compressed to nothing could not flatter the result, and **verified by counting the
bytes that come back** rather than by the upload's status code — a truncated file that reports
204 is the failure mode that matters:

| Payload | Upload | Verified | Round trip |
|---|---|---|---|
| 599 KB (measured max job doc) | HTTP 204 | exact | 1.3s |
| 2120 KB (measured max evidence sidecar) | HTTP 204 | exact | 0.9s |
| 8192 KB (headroom) | HTTP 204 | exact | 1.2s |

This closes the design question the parameter probe opened: the sizes that break a Delta
parameter are unremarkable here, and latency is flat across a 14x size range. It does *not*
remove the need for the background writer thread — ~1s on the event loop × 15 saves per run is
still a stalled UI, and this was measured from a laptop, not from a container. Warm sequential
322 KB writes measured a **0.59s median**, which is the number the coalescing window is sized
against.

### The Delta index has no consumer, so it is not built

The plan said "documents to the Volume, index to Delta", and the Delta half rested on one
assumption: that a Volume listing gives you names and nothing else, while `StoredObject` needs
`size` and `mtime` (`report_delivery` orders the Report tab newest-first; `JobStore.prune`
compares an age to a retention window). Measured against the provisioned volume, the Files API
listing carries **both** — `file_size` and `last_modified` per entry.

That leaves the index with nothing reading it. Building it anyway would put a SQL warehouse on
the write path of **every** job save — a cold start (the Serverless Starter Warehouse is
STOPPED), a MERGE's latency, a second set of credentials and a second failure mode — for a
table no caller queries. **Dropped from the design.** The remaining Delta prerequisite,
warm MERGE latency, is therefore no longer needed either.

Four more semantics the probe pinned, each of which the fake in `tests/fake_files_api.py`
encodes so the contract suite is testing something real:

| Question | Measured |
|---|---|
| Overwrite an existing path | `PUT ?overwrite=true` → 204, read-back byte-exact |
| Absent file, and `DELETE` of one | **404 both** — so absence is distinguishable from a 403 |
| `last_modified` unit | **milliseconds** |
| Recursive listing | **one level per call**; there is no `rglob`, so the backend walks directories itself |

The millisecond unit is the mirror image of this repo's `mtime = 0.0` near miss: passed
through raw it places every job ~54,000 years in the future, so `prune` never expires
anything — a retention policy that reports success and does nothing.

### The <app-workspace> serving endpoint answers, and it reasons

The workspace's own reasoning endpoint is READY, task `llm/v1/chat`, no external-model
indirection —
config-identical to the endpoint it replaces. Verified through `LLMClient` itself, not just
curl: `complete` 2.1s, and `structured_output` filled a real `AnomalyList` in 38.1s.

Two behaviours to carry, neither a fault:

- **The reasoning block bills against the completion budget.** For one 3-word answer, <app-workspace>
  spent 308 completion tokens where the old endpoint spent 32. So `max_tokens` here is *not*
  the visible answer's budget, which is the exact shape of
  `truncation-masquerades-as-schema-error`: a starved schema call is reported as whatever key
  went missing. The configured 4096 / 8000 budgets are unchanged and were adequate, but they
  now have less real headroom than the same numbers had before.
- **The endpoint rejects `temperature`** with a 400. `LLMClient` already handles this: it warns
  once, drops the parameter for the rest of the process, and the endpoint's own default
  applies. So `temperature: 0.2` in `llm_config.yaml` is inert on this endpoint.

### Where the identity can write

Against the `<analytics-warehouse>` warehouse (2026-08-06): catalogs visible are
`<analytics-catalog>` (schemas `default`, `information_schema`, `audit_datalake`,
`<a-sandbox>`), `<third-catalog>`, `<reporting-catalog>` (30
schemas), plus `hive_metastore`, `samples`, `system`. **No `afir` schema exists, and no
volumes exist in the analytics catalog.** `SHOW GRANTS ON CATALOG` returned nothing for the
current identity, so read access here does not evidence CREATE — that has to be attempted
or granted deliberately, and it lands in a **production** catalog.

## The bundle, and the three things it decided rather than configured

`databricks.yml` owns *deployment* — which files are synced, which resources must exist, who
may use the app. It deliberately templates **no value into a config file**: `config/*.yaml`
and the pack are edited through the UI at runtime and mirrored to the Volume, so a redeploy
that wrote the shipped default over an operator's edit would be the mirror's whole purpose
defeated. Two targets: `dev` (`mode: development`, default, no `permissions:` block because
the deploying user has it implicitly) and `prod`, whose `permissions:` block *is* the entire
authorization model.

Three details are not arbitrary:

**`sync.exclude` denies by content, not by convenience.** `bundle deploy` syncs the working
**tree**, not the git index, and an untracked `config/main_config.yaml` holding live tokens is
exactly what a developer has on disk. So the list names credentials (`config/*.bak` — written
beside every file `config_store` patches, holding the *pre-redaction* content — `config/certs/**`,
`.afir_env`, `.databrickscfg`), real operational data (`jobs/**`, `exports/**`, `feedback_*`,
`knowledge/*/use_cases/*/cases/**`, `knowledge/*/.history/**`), and local bulk.

**`.gitignore` is honoured too, on top of this list** — measured from the deploy's own sync
manifest, where the confidential procedure PDF, `.DS_Store`, `.idea/` and `config/main_config.yaml`
are all absent while `sync.exclude` names none of them. Belt and braces on the credential side,
and a load-bearing consequence on the other: the bundle therefore ships **no** `config/*.yaml`,
which is what makes seeding from `config/templates/` the App's only source of a config file (see
below). The explicit list still earns its place — it must hold for a file that is *tracked*, and
`.gitignore` answers a different question in a repo where `tests/**` is committed on purpose.

**`requirements.txt` IS the App's install list, and the platform reads that filename with no
override.** The plan assumed a separate `requirements-app.txt` could be pointed at; it cannot.
So the split is inverted: `requirements.txt` is now the **runtime** set and
`requirements-dev.txt` (`-r requirements.txt` + pytest/black/flake8/isort/torch/notebooks) is
what a developer installs. Nothing the server imports was dropped — flask, fastapi, uvicorn,
celery, redis, splunk-sdk, matplotlib and plotly were never imported by it. **torch and
sentence-transformers go too**, which is only possible because the default embedding provider
is an endpoint; the one-line reinstall for an air-gapped deployment is a comment in the file.
Finding this also surfaced five module-top imports in `src/rag/knowledge_base_manager.py`
(`asyncio`, `pickle`, `faiss`, `numpy`, `SentenceTransformer`) that nothing used — flake8 F401
×5 — and were single-handedly what made torch mandatory in the image.

**`app.yaml` sets three separate directories, and one of them is a lie if you read it as a
store.** `AFIR_CONFIG_DIR` and `AFIR_KNOWLEDGE_DIR` are the mirror's *working* copies (the
byte-level patcher needs a real filesystem with `os.replace`); `AFIR_DATA_DIR` is a **cache**
holding only what can be rebuilt. Anything that must outlive a restart goes to the Volume
through the `storage:` block. `DATABRICKS_DISABLE_EXPERIMENTAL_FILES_API_CLIENT=true` is not a
performance setting — SDK issues #1148/#1153 are open at v0.125.0 and the blocked presigned-URL
path returns no error, so an SDK Volume transfer *hangs*.

## The three blocking endpoints answer 501, and it is an operator switch

`POST /api/v1/incidents`, `/api/v1/incidents/freetext` and `/api/v1/ir` run the pipeline inline
and return the report. Behind the ingress's ~60–120s cap against a **38 min median / 476 min
longest** run, they cannot finish — so inside an App they answer **501 with `use_instead:
"POST /api/v1/jobs"`** rather than being cut off at 120s with nothing to show. A timeout and a
refusal look identical to a caller; only one of them says what to do next.

Where the check lives matters twice. It is at `_run_pipeline`, the one funnel all three share,
**plus** a second time at the top of `get_ir` — that handler fetches the record from the
external tracker *before* it has an incident to run, and a request we already know we will
refuse should not first make somebody else's system do work for it. The test asserts this by
pointing `win_url` at a dead port.

Not a hard-coded platform branch: `incident_input.blocking_endpoints` is `auto | on | off`,
`live`-editable from the Configuration tab, and `auto` reads **only** `DATABRICKS_APP_PORT`
(the OAuth vars also exist on a laptop with the SDK installed, so they cannot mean "in an App").
An unrecognised value logs an error *and* resolves to `auto` — the resolution helpers live in
`src/utils/deployment.py` and nothing in them branches behaviour on its own.

## The second shape: a cluster driver behind the driver proxy

An App's container is not on the corporate network. Measured: a run inside one finishes green
with `sources: []` because **every** query raised — serverless Apps answer `Unauthorized
cross-workspace request` for the warehouses and cannot resolve the ELK estate at all, while
classic compute in the same VNet reaches both. That is not a configuration gap, so the App is
not the only supported shape: `scripts/deploy_driver_proxy.py` runs the same server as a
child of a **cluster driver**, published by the driver proxy at
`/driver-proxy/o/<org>/<cluster>/<port>/` (`-api/` for a Bearer token). No grant is needed —
arbitrary ports, `CAN_ATTACH_TO` is the only permission a viewer needs.

**Exactly one part of that URL is AFIR's**, which matters only because an operator has to
recognise it: `driver-proxy`, the org and the cluster *id* (never its name) are fixed and name
no service, and **the port is not free either** — "arbitrary ports" above is the permission,
not the range. The proxy allowlists ports and answers a forbidden one with `403
PERMISSION_DENIED: Traffic on this port is not permitted`, so a mnemonic port (2347, AFIR on a
keypad) is simply unreachable; 1024 and 2347 are refused where 3000–8501 are permitted, and
the deployment stays on **8080**. Measured from outside with **no listener anywhere**, which
is the cheap part: the proxy decides before it connects, so a permitted port with nothing
behind it answers **502** and a forbidden one **403** — the two errors are the instrument.

So the name goes in the **path**: the page is also served at `…/<port>/afir`, an **alias and
not a base path** — `afirBasePath()` reads segments 1–5 and ignores the tail, so every call
underneath is unchanged. A real hostname would need a reverse proxy holding one Databricks
credential for everybody, which is exactly the shared-identity ingress `src/identity.py`
exists to refuse.

What the shape costs, stated rather than discovered later: the service is a child of the
driver, so **a cluster restart or an autotermination drops it**, there is no
restart-on-failure and no health management, and bringing it back is `stage → install →
launch` again because `/local_disk0` does not survive. Single replica comes free, which is
what `DatabricksStorage`'s single writer thread needs anyway. The next section removes that
cost without changing anything else; `deploy_driver_proxy.py` stays as the ad-hoc path,
because pushing the working tree at a cluster in one step is still the fastest way to try
something.

Six platform facts the script encodes, each measured on DBR 16.4 LTS rather than assumed:

- **The proxy strips its own prefix and sends no `X-Forwarded-Prefix`**, so the *server* needs
  no change and the *browser* does — `src/ui/script_base.py`, and a no-op off the proxy.
- **`DATABRICKS_APP_PORT` is deliberately not set.** The 501 is right for an ingress that cuts
  at ~120s; this one does not, and the blocking endpoints stay usable.
- **No `python3-venv`** on the node — stdlib `venv` fails at `ensurepip` on every interpreter
  there — but `virtualenv` is preinstalled and carries its own pip. `pip install --target`
  plus `PYTHONPATH` is the fallback.
- **Secrets are resolved on the driver by `dbutils.secrets.get`** into the child process env,
  from an overlay naming only `{scope, key}`: an address is not a credential, and no token is
  written to a file or a command line.
- **`DATABRICKS_TOKEN` is popped from that env.** With it beside the driver's own credentials
  the SDK refuses to choose (`more than one authorization method configured`), and losing
  `DatabricksAuth` costs the LLM endpoint and every Databricks retriever's host at once. The
  config names its own token vars, so the ambiguous one is dropped rather than added to.
- **`COPYFILE_DISABLE=1` on the tar.** macOS bsdtar otherwise ships every xattr as a sibling
  `._name` entry, and `._main_config.yaml` is a file every `*.yaml` glob in the pack loader
  picks up and cannot parse.

Two config facts follow from the network rather than from the shape: the workspace host
verifies against the node's default trust store, so `verify_ssl` is **true** here (it is off
locally only for a self-signed corporate chain), and ELK is reached **node-direct** —
`gateway: node` in the `elasticsearch` backend block, which keeps Query DSL and every pack
`query_hints` written for it while addressing `POST /{index}/_search` on 9200 instead of
Kibana's port. There is no VIP: the cluster-wide SAN does not resolve, so the URL names a data
node.

Acceptance for a deployment of this shape is **not** "the run finished". A run whose every
source returned nothing finishes, green, in minutes, and reads exactly like success — so it is
`/health?deep=1` **through the proxy** (`storage_ok`, a loaded pack, `sources_unavailable`
accounted for one by one) and then one real incident diffed against a local baseline verdict
line by line.

## The same shape, continuous: nothing on a laptop is load-bearing

The shape above works and is not a service, and the difference is one question: **after a
restart, what has to happen before the page answers again?** With `deploy_driver_proxy.py` the
answer is a person with this repository checked out. `scripts/afir_service.py` moves that answer
onto the platform by moving **where each thing lives**, and nothing else:

| what | where | why there |
|---|---|---|
| release tarballs, the baked venv, `manifest.env` | `/Volumes/<catalog>/<schema>/dist` | shared, durable, governed by the same grant as the state |
| the bootstrap | `dist/driver_init.sh` (a workspace file only in dedicated mode), a cluster-scoped init script | runs before the driver process, on every boot, with nobody watching |
| run state, exports, feedback | `/Volumes/<catalog>/<schema>/state/…` | already the `storage:` block — unchanged |
| each caller's runs, artifacts and drafts | `state/users/<segment>/…` | already `owner_scoped` + `user_overlay` — unchanged |

The last two rows are the point of listing all four: **the per-user half is not something a
deployment arranges.** `identity.mode` decides whether a caller is read at all, `owner_scoped`
puts their runs and exports under their own prefix, `user_overlay` keeps their config and pack
edits as drafts against the base, and a run of somebody else's is a 404. Provisioning does not
add that; what it can do is switch it off by accident, which is why `provision` **refuses to
deploy on `identity.mode: auto`** — `auto` resolves through `running_on_databricks_driver()`,
which reads `DATABRICKS_RUNTIME_VERSION`, and a root-owned init script's children are not
guaranteed to carry it. Resolved the wrong way it does not fail: every caller becomes the one
local administrator writing to the shared tree. The deployment says `on`.

A **separate `dist` volume** rather than a prefix inside the state one, for a reason visible in
the UI rather than in a traceback: `report_delivery.artifact_inventory` does one `list_keys("")`
over the store root on every Report-tab render, and release tarballs under that root would be a
listing the page walks to draw a button. Same grant, same schema — one name, not a second
destination.

### Every location, and what is actually in it

Three tiers, and the only interesting question about each is what a cluster restart does to it.
**Durable, shared** — `/Volumes/<catalog>/<schema>/dist`, written by the operator, read by every
boot:

| key | what it is |
|---|---|
| `manifest.env` | the two archive names + the `requirements.txt` hash, released-at and released-by. Written **last** in a publish; rolling back is rewriting this file |
| `releases/app-<ts>.tar.gz` | one full deployable tree per publish, ~19 MiB extracted. Kept, not pruned — a rollback needs the old one to still be there |
| `runtime/venv-<sha>.tar.gz` | the baked venv, ~100 MiB, keyed on the `requirements.txt` hash so a release that did not change dependencies reuses it |
| `driver_init.sh` | the bootstrap the cluster spec points at (shared modes; see the access-mode section) |

**Durable, per-deployment** — `/Volumes/<catalog>/<schema>/state`, the `storage:` block's own
root, written by the running service. Nothing here is deployment-specific; it is the same layout
`LocalStorage` writes under `data/`, which is what makes the local backend the test oracle:

| key | what it is |
|---|---|
| `jobs/<job-id>.json` | the job document: stages, gates, interventions, the verdict. `.prev` beside it is the previous version, written by `DatabricksStorage` before an overwrite |
| `jobs/<job-id>.evidence.json` | the evidence sidecar, rewritten only when the evidence changed (most saves are status updates) |
| `exports/fraud_report_<id>.md` / `.pdf` | the acceptance artifact, written unconditionally |
| `exports/evidence_raw_<id>.json`, `evidence_transformed_<id>.json` | what the retrievers returned and what the transform plan made of it |
| `jobs/users/<segment>/…`, `exports/users/<segment>/…` | the same four shapes for an **owned** run. The segment sits *inside* the `jobs`/`exports` prefix, because `owner_scoped` narrows the already-prefixed store. An unowned run keeps the flat key it had before the seam existed |
| `users/<segment>/layers/{config,knowledge}/<name>` | a non-admin caller's **draft** edits, with `<name>.base` recording the base they were merged forward from. Root-level, not under a stage prefix — `user_overlay` narrows the store itself |
| `feedback_insights.json` | the distilled reviews, appended |
| `feedback_thresholds.json` | the one tuned number's full history. Never written back to YAML, which stays the baseline anchor |
| `config/*.yaml`, `knowledge/**` | the mirrors, and they hold **edits only**. Empty is the normal state and means nobody has edited through the UI — not that mirroring is off. A delete is a tombstone here, or the next boot re-seeds the file it removed |

**Ephemeral, node-local** — `/local_disk0`, gone on every restart, and that is the design rather
than a limitation: everything here is either re-fetched from `dist` or re-derived.
`/local_disk0/afir` is the extracted tree, `/local_disk0/afir-venv` the restored venv at the
pinned path a venv's shebangs bind it to, `afir-bootstrap.log` and `afir_status` the boot record,
`afir/afir.log` + `.log.1` the app log under one 256 MiB rotation, `afir.pid` /
`afir.supervisor.pid` the two live processes, and `afir/STOP` the sentinel that makes stopping
distinguishable from crashing.

What the release tarball ships is one list (`STAGE` in `deploy_driver_proxy.py`, shared by both
deployment paths so they cannot diverge): `app.py`, `requirements.txt`, `src`, `config`,
`knowledge`, `knowledge_base`, `plugins`, `docs`, `assets`, `scripts`. Deliberately absent:
`vendor/` (cp311 wheels against a 3.12 driver), `tests/`, and `exports/`/`jobs/`/`model_cache/`
— which on this deployment are not merely regenerated but **live somewhere else entirely**, the
two Volumes above. One asset in that list earns less than its name suggests: `knowledge_base/`
carries `documents.json` plus a `faiss_index.pkl`, and the index is **not** a warm start —
`rag/orchestrator.py` calls `update_index(force_rebuild=True)` at every boot, deliberately,
because the first encode is where a missing embedding model raises and triggers the deterministic
fallback. So each start re-embeds the corpus, and it is cheap enough that this is not a defect:
**2.6 s** for 90 documents against the qwen3 endpoint (14:01:54.2 → 14:01:56.8 on the boot
recorded below), inside a 13.8-second app start and a 21-second boot. Shipping the pickle costs
360 KiB and saves nothing; `documents.json` is the asset that matters. The same log lines carry a
finding that is *not* about deployment: half a dozen pack documents are truncated from ~300k
characters to the endpoint's 96k, so their vectors describe only their opening — a pack-authoring
concern, logged where it happens.

Four things measured on the way, each of which had a plausible wrong answer:

- **`pip` does not work from an init script on this network, and does from a notebook on the
  same driver.** The init context gets a connection reset per index request and reports
  `Could not find a version that satisfies the requirement PyYAML~=6.0.1 (from versions:
  none)`; the identical command through the driver's REPL downloads at 12 MB/s. No proxy
  variable and no `pip.conf` exists to point at — the node's egress is simply not in place that
  early. So the baked venv is **not** a boot-time optimisation, it is the only path that works
  here: `bake` builds it in the notebook context and publishes a 100 MiB tarball, and
  `driver_init.sh` keeps its pip fallback as the loud diagnostic for a workspace where this is
  not true. First boot: `no_deps`, cluster **up**, marker readable. After baking: **21 seconds**
  from init-script start to `/health` answering.
- **A venv that imports today can be a venv that is broken on the next boot.** The notebook's
  interpreter lives in an ephemeral env under `/local_disk0/.ephemeral_nfs`; a venv symlinked to
  it passes every check in the session that built it and is a dangling symlink after the restart
  this shape exists to survive. `bake` therefore builds `--copies` from
  `/databricks/python3/bin/python` and **verifies the result** (`realpath` of `bin/python` plus
  `pyvenv.cfg`, refusing anything containing `ephemeral`; the accepted one reads
  `base-executable = /usr/bin/python3.12`). A venv is also path-bound, so the bake path and the
  restore path are the same pinned `/local_disk0/afir-venv`, written in both files.
- **The UC Volume FUSE mount IS available to a cluster-scoped init script** on DBR 16.4 —
  measured, not assumed, because the init script runs as root before the driver and `dbutils`
  does not exist there at all. `fetch()` tries the Volume mount, then `/dbfs`, then `curl`
  against the Files API with a secret-supplied token, and **logs which one answered**; all three
  boots so far read `via volume fuse`. That log line is the measurement, and it is the first
  thing to read when a boot comes up empty.
- **macOS `bsdtar` writes an xattr header per file** (`LIBARCHIVE.xattr.com.apple.provenance`)
  that GNU tar on the driver ignores one warning line at a time. Harmless, and the first 1800
  characters of the first failed bootstrap were nothing else — the log is the diagnostic surface,
  so `build_tarball` adds `--no-xattrs` beside the older `COPYFILE_DISABLE=1`.

Two contracts hold the rest together. **The init script never fails the cluster**: every failure
path logs, writes a one-word outcome to `/local_disk0/afir_status` (`no_manifest`,
`bad_manifest`, `no_release`, `bad_release`, `no_venv`, `no_deps`, `no_supervisor`, `started`,
`up`) and exits 0, because a cluster that refuses to start is one nobody can debug from — and
the driver is also how an operator probes the estate. Whether the *service* is running is the
marker and `/health`, never the cluster's state. And **stopping is a file**: `driver_supervise.sh`
restarts the app for as long as the driver lives, so a `kill` is indistinguishable from a crash
and the supervisor wins that argument — `app-stop` writes `$ROOT/STOP` first, and
`deploy_driver_proxy.py stop` is therefore a *restart* while the supervisor is up.

`manifest.env` is the whole release protocol: two archive names, read and **never sourced**
(`source` on a file from a shared location is arbitrary code execution by design), written last
in a publish so a restart mid-release cannot come up on a name that is not there yet, and
carrying the venv forward keyed on a `requirements.txt` hash that `bake` refuses to mislabel.
Rolling back is rewriting that one file. A release with no cluster restart is `publish` +
`reload`, which re-runs the bootstrap from a **copy of the script in `/tmp`** — its first real
act is to delete the tree it lives in, and bash reads a script incrementally.

`provision` builds the cluster spec in full and applies it by **name**, so the cluster is a
function of that script plus the overlay rather than of whatever was clicked on the way, and
editing the existing one in place keeps its id — which is the proxy URL people have bookmarked.
`autotermination_minutes: 0` is what continuous costs: an always-on driver. Any other value is a
correct, cheaper thing that is not a service.

### Who may reach the page is the cluster ACL, which costs one access mode

A driver-proxy service has no login of its own, so **the cluster's permission list *is* the
guest list** — and `--access-mode` decides whether that list is consulted. The three modes are
not three degrees of the same thing:

- **`dedicated`** (`SINGLE_USER`) puts a principal check *above* the ACL. `CAN_ATTACH_TO` is
  granted, accepted, and then the command is refused anyway: `Single-user check failed: user 'X'
  attempted to run a command on single-user cluster …, but the single user of this cluster is
  'Y'`. A group is accepted as that principal on this workspace, which turns "who may use AFIR"
  into a SCIM membership request per person — the ACL is left saying something that is not true.
- **`no-isolation`** (`NONE`) makes the ACL the gate and gives every attached user the driver's
  own filesystem and process namespace. AFIR's three secrets arrive as `spark_env_vars` and live
  in the service process's environment, so this mode publishes them to everyone on the list.
- **`standard`** (`USER_ISOLATION`) makes the ACL the gate and keeps the isolation. **Measured on
  this cluster**, from an execution context on the driver: it runs as `uid=1004` in a per-user
  sandbox, and `open("/proc/<service-pid>/environ")` is `PermissionError [Errno 13]`. The tree
  under `/local_disk0/afir` *is* world-readable at `0644`, which is the right answer for the
  wrong-looking reason — `config/*.yaml` names its secrets by **env-var name** and holds no
  value, so a readable config is not a readable credential.

`standard` is therefore what `provision` defaults to, and `single_user_name` is written **only**
in dedicated mode. Two consequences to know before switching:

- **A shared cluster refuses a workspace-file init script**: `Error: Shared clusters do not
  support init script storage type wsfs`. So the destination follows the mode — the `dist` Volume
  for `standard`/`no-isolation`, a workspace file for `dedicated` — and the Volume path is the one
  exercised continuously. (DBFS init scripts are EOL and are not an option in either.)
- **The per-user sandbox has its own network namespace**, so `127.0.0.1:8080` on the driver
  answers `Connection refused` to a probe running in an execution context while the same endpoint
  returns 200 through the proxy. `status` therefore asks `/health?deep=1` from the operator's
  machine through `databricks api get /driver-proxy-api/…`, which also keeps the token out of the
  script: the two proxy routes each refuse the other's credential — a browser session for
  `/driver-proxy/`, a bearer token for `/driver-proxy-api/` (a token on the browser route is a
  303, and no credential at all is a 401 at the ingress, never reaching AFIR).

What the mode does **not** change is whose grants the data reads use. Every warehouse and ELK
call is made with the PAT in `spark_env_vars`, so **the estate is read as that token's owner** no
matter who is looking at the page. AFIR's identity layer segregates *state*, not grants: an
attached caller sees only their own runs and drafts, over data fetched with one shared identity.

Verified end to end on the deployment cluster: `publish` → `provision` → `start` (first boot
`no_deps`, cluster up, diagnosable) → `bake --rebuild` → `reload` → **cluster restart with
nothing running locally** → bootstrap `up` in 21s, `/health?deep=1` through the proxy reporting
`storage_ok`, the pack loaded with all ten rulesets, and the three pre-existing job documents
still there. `/api/v1/whoami` on the token URL answers `mode: on`, `enforced: true`, `role:
admin` by group membership, `edits_the_base: true`; the same URL with no credential is refused
by the ingress at 401 and never reaches AFIR.

Then re-verified after the switch out of dedicated mode, because that changes the init-script
storage type and the isolation the service runs behind: the cluster came back
`data_security_mode: USER_ISOLATION` with `single_user_name: null`, its init script read from the
`dist` Volume, all three artifacts `fetched … via volume fuse`, bootstrap `up` at 14:02:00 —
21 seconds again — supervisor and app pids live, `/health?deep=1` still `ok` with `jobs: 3` and
`storage_ok`, and 42 of 42 declared sources built except the four whose backends this deployment
has no URL for. The `/api/1.2` execution-context tooling `bake`, `reload` and `status` depend on
keeps working in the shared mode; the only thing that broke was the health probe those commands
used to run *on* the driver, which is now asked through the proxy instead.

## The one gap the suite cannot close, and how it was closed

Everything `DatabricksStorage` and `TreeMirror` assert is asserted against a **local fake**
built from the measurements above. If the platform differs from the fake, the fake is what the
suite agrees with — so the seam was run against the real Volume,
`/Volumes/<app-catalog>/afir/state` in the deployment workspace. **29 checks, 29 passed**, in nine sections chosen to be
the ones a fake can get wrong: a put/flush/get round trip byte-exact; that a second `put_text`
**replaces** rather than appends and `get_previous_text` still returns the prior bytes; that a
failed `verify` stores *nothing*; that `list_keys` fills size and an mtime in **POSIX seconds**
(the storage layer treats `mtime == 0.0` as unknown and *milliseconds read as seconds* expire
nothing, so the unit is load-bearing); `append_text` against a cold second backend object;
durability read back through a second independent backend rather than the writer's own cache;
a mirror push → wipe the working copy → `sync_down`, including the `.history` tree under its
`history` alias; a tombstone surviving `sync_down`; and that absence never raises.

Then AFIR itself was booted in App mode with **all** durable state on that Volume: mirror
sync-down, pack load, 25 retrievers, server ready. That boot also exercised the RAG degrade
path for real — the embedding endpoint answered 403 from the probe shell and the boot came up
on the deterministic `PlaybookFallback` instead of failing.

`var.catalog` in `databricks.yml` therefore defaults to that catalog rather than `main`. A
placeholder that *resolves* is worse than one that does not: the deploy succeeds and every
Volume write 404s into a store that reports itself degraded in a boot log nobody reads —
which is why `/health?deep=1` now answers `storage`, `storage_ok` and `storage_detail`.

## What `bundle validate` corrected, and the boot defect it exposed

The bundle is no longer unvalidated: CLI **v1.11.0** ran `validate` and `deploy -t dev`, which
created the app `afir-dev`. Four things the hand-written file got wrong, three of them silent:

- **`source_code_path: ../` → `.`** — this file sits *at* the repo root, so the bundle root
  already is the repo. `../` is outside the sync root, which `validate` **rejects** rather than
  resolving. The only one of the four that failed loudly.
- **There is no `volume:` app resource.** An app resource is one of exactly four kinds (`job`,
  `secret`, `serving_endpoint`, `sql_warehouse`); UC access is a **grant** to the app's service
  principal, not a binding declared here. The invented block was reported as `Warning: unknown
  field: volume` — a *warning*, so the deploy would have succeeded and dropped it, and the app
  would come up with no Volume permission while `/health?deep=1` was the only thing that said
  so. The `GRANT` statements now live in the file as a comment.
- **`CAN_USE` is app-scoped, not target-scoped.** A target-level `permissions:` block accepts
  only `[CAN_MANAGE, CAN_VIEW, CAN_RUN]` and governs the *bundle*; `CAN_USE` there is a hard
  error. It moved under `resources.apps.afir`.
- **`prod` needs an explicit `workspace.root_path`**, or it lands under whoever ran the deploy —
  and two operators then produce two copies, which is the second replica this deployment cannot
  have. Not `/Workspace/Shared` (the CLI warns it is read/write for every workspace user, and
  this root holds the deployed config and pack). The chosen folder is a **prerequisite**: with
  `/Workspace/Applications` absent, `validate -t prod` answers `403 PERMISSION_DENIED`, which is
  the right way round — a deploy that quietly relocated itself somewhere writable is the defect.

**Then the deployed bundle could not boot, and the suite could not have caught it.** Replaying
the CLI's own sync manifest (315 files) into a container-shaped tree and running `app.py` with
the App's env died in `load_config`'s bare `open()`, **before logging was configured**:

```
FileNotFoundError: .../work/config/main_config.yaml
```

`config/*.yaml` hold live tokens and are gitignored — and **`bundle deploy` honours
`.gitignore` on top of `sync.exclude`**, which is also why the confidential procedure PDF and
`.DS_Store` never synced despite neither being in the exclude list. So the deployed tree's
`config/` contains nothing but `templates/`, and `seed_working_copies` copied depth-1 only:
**zero files**. It now falls back to `config/templates/`, **flattened** (`templates/main_config.yaml`
arrives as `main_config.yaml`) and decided per file, so an operator who ships one real config
still gets the shipped defaults for the other three. Locally the templates stay inert — the real
depth-1 files are found first.

One more failure was hiding behind the first, same shape as the `volume:` warning: a key that is
**present with a null value**. `plugin_config.yaml` ships every plugin commented out, so
`active_plugins:` parses to `None`, `config.get("active_plugins", [])` never applies its default
(the key is not absent), and `None` reached `in self.active_plugins` → `TypeError`, killing the
boot. `PluginManager` was constructed in `main()` and mocked as a `MagicMock()` in every test, so
`__init__` had **never run** under pytest; `tests/test_plugin_system.py` now exists. It is the
only such key in the four templates — checked, not assumed.

With both fixed, a first boot from the deployed file list reaches *server ready*: config seeded
(4 files), pack loaded, `storage_ok: true`, the UI serving 267 KB, and `POST /api/v1/incidents`
answering **501** naming `POST /api/v1/jobs`.

## The bootstrap trap: four decisions the UI cannot make about itself

The App now booted, and was **unconfigurable in exactly the four places that matter**. Everything
about AFIR is editable from the Configuration tab and mirrored to the Volume — but only once there
*is* a Volume. In a container the first config comes from the read-only bundle, whose template says
`backend: local`, meaning the container's own disk. Measured through the fake Files API: an
operator switching that in the UI gets `applied`, and `durable: true` — truthfully, because
`build_mirrors` returned `None` for a local backend, so there is no mirror to push to and nothing
to report as skipped. The restart that makes a `restart` field take effect then discards the disk
the switch was written to, and the app comes back on `local` again. **A configuration surface
cannot configure where its own writes are stored.**

So `${VAR:-default}` was added to `_expand_env` and the shipped template's four bootstrap values
read through it, driven from `app.yaml`: `AFIR_STORAGE_BACKEND`, `AFIR_UC_CATALOG` /
`_SCHEMA` / `_VOLUME`, plus `AFIR_PACK_DIR` (which domain this instance investigates — left unset,
the app answers real incidents plausibly against `mock_domain`'s rules). Env is the only channel
the platform offers that early. Two properties are load-bearing:

- **An empty value counts as unset.** A platform can inject a declared-but-blank var — an
  `app.yaml` entry left with no value, a resource not filled in. Treating `""` as a value turns the
  default into `""`, and for `storage.backend` the empty string means ephemeral container disk:
  every pending approval lost on the next restart while the app reports itself healthy.
- **The bare `${VAR}` form keeps its old semantics** — an unset secret must stay empty so the
  source is skipped with a log line, not acquire a literal that only 401s at query time. The two
  forms must not converge.

A laptop or VM that sets none of them expands to the shipped values and behaves exactly as before,
which is what `tests/test_config_env.py` asserts against the **real** template files.

**The same shape one layer up, in `llm_config.yaml`: a placeholder that resolves is worse than a
blank.** The template shipped `base_url: "https://<workspace-host>/serving-endpoints"`, and it
would have 401'd every LLM stage in a way that names the wrong problem. It passed `main()`'s
`"serving-endpoints" in base_url` test, so auth was used; then it **won over**
`auth.serving_base_url()`, because an explicit value is meant to; and it reached the SDK
URL-encoded as `https://%3cworkspace-host%3e/serving-endpoints`. Blank now means "Model Serving on
the workspace the SDK resolves", which is precisely what injected OAuth credentials give an App —
in the template, in `main()`'s auth test (`not configured_base_url or "serving-endpoints" in …`),
and in `LLMClient.__init__`. The third site matters on its own: `AsyncOpenAI(base_url=None)`
defaults to `api.openai.com`, so a blank URL with no SDK available would send a Databricks token to
OpenAI and report a 401 from the wrong provider. It is an ERROR at construction instead.

## Still to do
- **A deployed config must set `verify_ssl: true`** on both Databricks backends and on
  `rag.embedding_verify_ssl`. The shipped template already does; the local
  `config/main_config.yaml` has them off for a self-signed corporate chain, and that file is
  gitignored, so a deployment that copies a developer's config inherits the wrong posture.
  Leave `embedding_host` and `embedding_token_env` **blank** so the endpoint resolves through
  the OAuth service principal rather than an injected PAT.

## Carried risks

- **Single replica, and it must be enforced.** `JobManager._jobs`, the gate
  `asyncio.Event`s and the SSE queues are per-process: a second replica cannot release a
  gate held on the first, and its `restore()` would re-arm gates for jobs the first is
  actively running. **`DatabricksStorage` now depends on this too, and more quietly.** A
  Volume has no append verb, so `append_text` is read-modify-write, serialised by the single
  writer thread. A second replica has its own thread and no lock between them: two analysts'
  reviews interleave and one is lost, in a feedback log that still parses cleanly. **On the
  continuous driver shape this is discharged structurally rather than carried**: one driver runs
  one supervisor which runs one `app.py`, and the supervisor's restart loop starts a replacement
  only after `wait` on the previous child returns. There is no replica count to get wrong.
- **SSE may not survive the ingress** (one report of a stream frozen at 120s regardless of
  keepalives), and `es.onerror` is currently `() => {}` — a dead stream is invisible. The
  5s inbox poll is the fallback; it needs wiring to the error, not inventing.
- ~~**RAG cold start**~~ — **closed.** The escape hatch was taken rather than carried:
  embeddings now come from `databricks-qwen3-embedding-0-6b` by default, so a boot indexes
  53 documents in **1.7s over the network** with no weights to fetch and no torch to warm.
  Measured locally through a real boot, and again on the deployment cluster at **2.6s for 90
  documents** — every boot, since the rebuild is forced. The residual risk is the inverse one — the App now
  depends on a serving endpoint at startup — and it degrades correctly: an unreachable
  endpoint activates the deterministic keyword `PlaybookFallback` rather than failing the
  boot.
- **The two source workspaces need static tokens.** Sources span `<reporting-warehouse>` and
  `<analytics-warehouse>`, and the App's OAuth service principal covers only <app-workspace>, so
  `DATABRICKS_TOKEN` and `ANALYTICS_DATABRICKS_TOKEN` must both be injected as secrets. A PAT
  issued for one workspace 403s on another, and that failure looks like a bad token rather
  than a misrouted one.
- **`verify_ssl: false`** on both Databricks backends (`main_config.yaml:105,144`) is
  dev-only for a self-signed corporate chain. The App sees the public chain — flip it.
