# The storage seam

Where durable state lives is one decision, taken once, in `build_storage(config)`
(`src/storage/__init__.py`). Everything else in the system asks a backend for a named blob
and does not know or care what is underneath.

    storage:
      backend: local        # local | databricks | sql   (absent = local)

**A `storage:` block absent from `main_config.yaml` — which is every deployment that
existed before this package — yields `LocalStorage` over `data_dir()`, i.e. exactly the
behaviour that shipped before.** The layout on disk is unchanged: `<root>/jobs/<id>.json`,
`<root>/exports/fraud_report_<id>.md`, `<root>/feedback_log.jsonl`. A VM upgraded to this
code reads its own existing files.

### Four destinations, three implementations

| Destination | Config | Implementation |
|---|---|---|
| Local disk | `backend: local` (or no block) | `LocalStorage` over `data_dir()` |
| An external / mounted volume, NFS share, attached disk | `backend: local` + `storage.root: /mnt/afir-state` | `LocalStorage` over that root |
| A Unity Catalog Volume, **managed or external** | `backend: databricks` + `catalog`/`schema`/`volume` | `DatabricksStorage` over the Files API |
| A transactional database | `backend: sql` + `storage.sql.*` | `SqlStorage`, one row per blob |

**An `external_volume` option would be a second spelling of an existing one, in both
directions, which is why there are three implementations and not four.** A *mounted* volume
is a filesystem: it takes `os.replace` and `fsync`, which is `LocalStorage` exactly, and the
only thing that differs is the root — so `storage.root` is the whole feature, and it is now
an exposed field rather than something reachable only by knowing the constructor signature.
A *UC* Volume is the same one backend whether the volume is managed or external: both live
at `/Volumes/<catalog>/<schema>/<volume>` and are addressed by the identical Files API, so
the distinction is a Unity Catalog property that never reaches this code. A database is the
one genuinely different mechanism, and it gets a genuinely different class.

## Why the seam exists

A Databricks App's container filesystem is explicitly ephemeral, and `/Volumes` is **not
mounted** in it — `open('/Volumes/...')` raises `FileNotFoundError`, and the Files API is
the only path. So `job_store`'s durable write (`os.replace`, which is what makes the swap
atomic) cannot reach durable storage there at all. Worse, its writability probe *succeeds*
against the container's `/tmp`, so the loud local-disk fallback never fires: the module
written to make sure a pending human approval outlives a restart reports success while
protecting nothing.

That is the `a-graceful-degrade-that-is-silent` shape, and it is why a remote backend has
to be a different **implementation** rather than the same code pointed at a different
root.

## The contract

`StorageBackend` (`src/storage/base.py`) is named blobs and nothing else:

| Method | Contract |
|---|---|
| `put_text(key, text, verify=None)` | Replace, never append. `verify` is called on the bytes **read back**, and a raised exception must leave the previous content intact. Returns a bool. |
| `put_bytes(key, blob)` | Same, no verification — a PDF has no cheap validity check. |
| `append_text(key, text)` | Append. Not read-modify-write, so two concurrent appenders cannot lose each other's record. |
| `get_text` / `get_bytes` | The content, or `None` for an absent key. **Never raises for "not there"** — absence is an answer. |
| `get_previous_text(key)` | The prior good version, or `None`. On the interface deliberately (see below). |
| `exists`, `list_keys(prefix)`, `delete(key)` | `list_keys` returns `StoredObject(key, size, mtime)`; `mtime = 0.0` means *the backend cannot say*. |
| `degradation` | A human-readable string when the backend is not doing what was asked, else `None`. |
| `close(timeout=10.0)` | Bounded. A remote backend may hold a queue. |

**Blobs only, deliberately.** An earlier design had `upsert_row`/`query_rows` alongside,
so a Delta table could be addressed as rows. But every actual consumer — job docs,
evidence sidecars, exports, the feedback JSONL — is a *named document*, and the one
listing consumer (`report_delivery.list_incidents`) needs nothing but name + mtime. Adding
a second contract would have leaked "this deployment uses Delta" into `job_store` and
`report_delivery`. With one contract, Delta is an implementation detail of
`DatabricksStorage`.

### `get_previous_text` is on the interface on purpose

The `.prev` file is `LocalStorage`'s own mechanic, so on the face of it the *recovery*
belongs there too. It doesn't: the policy — "a torn current document falls back to the
previous one rather than costing the operator the job" — lives in `JobStore.load_all`,
where it is written once. Putting the method on the interface (default `None`) is what
stops a future backend from silently forgetting the recovery.

## `LocalStorage` is the oracle

`LocalStorage` is not a new persistence scheme. It is the write path `job_store` has always
used, lifted behind the interface: temp file → `fsync` → parse it back → copy the target to
`.prev` → `os.replace`. Each step earns its place:

- **Parse-back before committing.** A truncated write that is still valid JSON is
  indistinguishable from a real document at load time, and the caller that would find out
  is a restart trying to recover a pending approval.
- **`.prev` is a copy, not a rename.** The target stays present throughout, so a crash
  mid-write leaves a readable *current* document rather than only a backup.
- **`os.replace` is atomic on POSIX**, which is what makes a concurrent reader see either
  the old document or the new one and never half of each.

Because it is the oracle, `tests/test_storage.py` is **one contract suite parametrised over
a `BACKENDS` dict**:

```python
BACKENDS = {"local": _local_backend, "databricks": _databricks_backend}
```

A new backend adds one entry and either passes or is wrong. `DatabricksStorage` runs there
against `tests/fake_files_api.py` — a real local HTTP server speaking the four Files API
calls, so the backend's own `requests` code path is exercised rather than mocked out. What
that still cannot prove is that the *platform* behaves the way the fake does; the fake
encodes what was measured against a live UC Volume, and a live run is a separate phase.

**A queueing backend needs a flush in the harness or the whole suite is vacuous.**
`FlushingDatabricksStorage` overrides the three write methods to `flush()` before returning.
Without it every contract test would pass off the in-memory queue with the remote never
consulted — 68 assertions about HTTP that never made a request, the
`a-test-harness-that-voids-itself` shape.

**A `.prev` or `.tmp` file is not a stored object.** `list_keys` hides both: a caller
enumerating jobs must not find two entries for one job, and one enumerating exports must
not offer a temp file as a download. For the same reason `delete` removes the `.prev` too —
otherwise `load_all`'s fallback resurrects a deleted job.

## `DatabricksStorage`: a UC Volume over the Files API

    storage:
      backend: databricks
      databricks:
        catalog: analytics        # required — there is no sensible default
        schema: afir
        volume: state            # → /Volumes/analytics/afir/state
        # host / token: omitted, so the SDK supplies both. An App has neither configured.
        verify_ssl: true

**One HTTP surface, four calls**, all under `/api/2.0/fs`: `PUT ?overwrite=true`,
`GET`, `DELETE`, and `GET /directories{path}` for a listing. Not the SDK's `w.files.*` —
`DATABRICKS_DISABLE_EXPERIMENTAL_FILES_API_CLIENT` exists because the SDK's presigned-URL
path is network-blocked inside an App and *hangs* rather than failing over. The SDK is still
used for what it is good at: `try_build_auth()` resolves the host and vends a currently-valid
token per call, so a local PAT and an App's short-lived OAuth service principal are one code
path.

`verify_ssl` **defaults to `true` here**, unlike `log_sources.backends.databricks.*` where
`false` is a dev accommodation for a self-signed corporate chain. Copying that default would
have inherited the insecure posture into the one component that carries the audit trail.

### It queues, and that decides most of the design

`JobManager._persist` has 15 call sites and several are plain `def` (`resolve_gate`,
`set_stage_output`, `_mark_cancelled`, `_on_gate_timeout`), so `StorageBackend.put_text` and
`JobStore.save` must stay **synchronous**. A ~0.6s round trip × 15 per run on the event loop
would stall every other job's SSE stream. So writes go to a **single background writer
thread**, coalescing by key: every save is a full snapshot, so only the newest per key matters.

Three consequences that are easy to get wrong, each with a test:

- **There are two dicts, not one.** `_pending` is the queue; `_inflight` is the write the
  thread is performing *right now*. A payload the writer has already popped is on neither the
  queue nor the Volume — so every read path merges both. Miss the in-flight half and
  `append_text`'s read-modify-write silently drops a record.
- **`flush` waits for both.** It exists for exactly one guarantee — a gate is durable before
  it is announced — and a flush that returned with an upload still in progress would report
  that guarantee while not providing it.
- **A full queue refuses rather than writing inline.** An inline write would break the
  single-writer property that makes `append_text` safe. `_enqueue` waits 30s for room and then
  returns `False`, which the caller already handles as a failed write.

**Single replica only, and this is where it becomes load-bearing rather than cosmetic.**
`append_text` on a Volume is read-modify-write (there is no append verb), serialised by the
one writer thread. A second replica has its own thread and no lock between them, so two
analysts' reviews interleave and one is lost — with a feedback log that still parses.

### Two unit conversions, each a whole class of bug

`last_modified` from a listing is **milliseconds**. Passed through raw it puts every job
~54,000 years in the future, so `prune` never expires anything — the exact mirror of the
`mtime = 0.0` defect below, and just as invisible. `_seconds()` divides when the value
exceeds 10^12 and returns `0.0` ("cannot say") for a missing one.

`DELETE` and `GET` on an absent path both answer **404**, which is absence. Anything else —
403 above all — is **logged at WARNING and still returns `None`**, because reading a
permission failure as "not there" turns a misconfigured grant into an empty job queue.

### What was dropped from the plan, and why

The original design paired the Volume with a Delta table (`afir.jobs`) indexed by job id, on
the assumption that listing a Volume gives you names and nothing else. The probe measured
otherwise: a Files API listing returns `file_size` and `last_modified` per entry, which is
precisely `StoredObject`. That left the Delta half with no consumer — and building it anyway
would have put a SQL warehouse on the write path of every job save, plus a MERGE's latency,
for an index nothing reads. It is not in the code.

## `SqlStorage`: one row per named blob

    storage:
      backend: sql
      sql:
        dialect: sqlite          # sqlite | postgresql
        dsn_env: AFIR_SQL_DSN    # the variable holding the connection string
        dsn: ""                  # an explicit one wins, for a hand-written config
        table: afir_blobs

For a deployment with neither a durable disk nor a Unity Catalog workspace — a container
fleet next to a Postgres it already runs, or a VM that wants an ACID single-file store on a
mounted share rather than a directory tree of loose files.

**A database, deliberately, and not a warehouse.** Two obvious candidates are refused, for
measured reasons rather than taste:

- **Databricks SQL.** The Statement Execution API caps *combined statement parameters* at
  1 MiB — measured against `<analytics-warehouse>`: 768 KB accepted, 1024 KB answered
  HTTP 400 `INVALID_PARAMETER_VALUE`. Real evidence sidecars here reach 2.07 MB, so a
  parameterised upsert does not merely run close to the ceiling, it cannot carry the payload
  at all. Job documents at 599 KB leave a 1.28× margin, and in this codebase **the margin is
  the budget, not the number**. A UC Volume has no such limit, which is why
  `backend: databricks` writes files.
- **Snowflake.** A columnar warehouse rewrites micro-partitions to update one row, has no
  `ON CONFLICT` upsert, and cannot concatenate `BINARY` — so `append_text` would become a
  read-modify-write with no statement-level atomicity. The connector is already a dependency
  for *log retrieval*, where reading many rows once is exactly what it is good at. Job state
  is the opposite shape: one row, written 15 times a run.

So the dialects are the two that do a single-statement atomic upsert **and** a
single-statement binary append. `sqlite` is the one registered in `BACKENDS`, so it is held
to the whole contract against the oracle on every run; the dialects differ in four values
(`_Dialect`) and in no statement, so what passes there is what Postgres executes.

### The atomic append is the capability the Volume backend does not have

    INSERT INTO afir_blobs (storage_key, body, prev_body, byte_size, modified_at)
    VALUES (?, ?, NULL, ?, ?)
    ON CONFLICT (storage_key) DO UPDATE SET
      body = afir_blobs.body || excluded.body, ...

One statement, so two replicas appending to the feedback log cannot lose each other's
record. `DatabricksStorage` reaches the same guarantee through a single writer thread and is
therefore **single-replica only**; `SqlStorage` has no such restriction, and that inversion
is the reason to pick it. The upsert deliberately does *not* rotate `prev_body` on an append
— a `.prev` of a log is not a recovery point, and paying a full copy per review would be a
per-append doubling of the table.

Writes are **synchronous** here, unlike the Volume backend, and `flush()` is an honest
no-op returning `not self._fatal`. The queueing there exists because a Files API upload is
0.59s over HTTP; a local sqlite commit is sub-millisecond and a Postgres one is a LAN round
trip, so queueing would import the "saved is not durable" hazard for no latency win.

### Four mechanics that mirror the oracle

- **Parse-back inside the transaction.** `verify` runs on the row *read back*, not on the
  string in hand, and a failure rolls back — including leaving `prev_body` untouched, so a
  refused write cannot poison the recovery copy.
- **`prev_body` is a column, not a second row.** A `.prev` *row* would have to be hidden from
  `list_keys` the way `LocalStorage` hides the file, and the first listing that forgot would
  show two entries per job. As a column it cannot leak into a listing, and `delete` takes it
  with the row — required, or `load_all`'s fallback resurrects a deleted job.
- **`mtime` is a POSIX second stamped in Python**, never `0.0` for a row that exists. The
  epoch read as a real age deletes the entire queue.
- **One connection per thread.** The interface is synchronous and called from the event loop
  *and* from the mirror's push hooks; sqlite forbids sharing a connection across threads, and
  a shared Postgres connection would interleave two transactions.

`list_keys` matches `storage_key = ? OR storage_key LIKE '<prefix>/%'`, so a prefix `jobs`
cannot pick up `jobsarchive` — the table is flat and shared, so the prefix is the only
scoping there is, and over-matching would hand `job_store` an export to parse.

### The connection string is read from the environment

`dsn_env` names the variable; `dsn` is honoured where someone has written one by hand and
wins, being the more specific statement. This is the indirection the repo uses for every
other credential (`api_key_env`, `token_env`), and the reason is specific to this value: a
libpq URL carries its password *inside* the string, so a `dsn` form control would be a
secret-shaped field on a page whose reads are redacted — and **a redacted control PUTs the
placeholder back on the next save**. So `dsn` is in `_SECRET_LEAVES` (a word-based rule would
have passed `postgresql://user:pw@host/db` straight through the unauthenticated
`GET /api/v1/config`), and the *field* offered by the form is `dsn_env`.

`table` is validated against `[A-Za-z_][A-Za-z0-9_]*` per dot-separated segment rather than
quoted — it is the one value that cannot be a bound parameter, being interpolated into DDL,
and a rejected name is a config error the operator can read where a quoted one silently
accepts something that is not a table.

## Keys are names, not paths

`safe_key` validates every segment against `[A-Za-z0-9][A-Za-z0-9._\-]*` and rejects an
absolute or empty key outright. `MAX_SEGMENT_LEN` is 160.

**Only a *trailing* slash on a prefix is cosmetic.** `key_prefix` uses `rstrip("/")`, and
the reason is a caught near miss: relaxing it to `.strip("/")` — so `JobStore` could list
with an empty prefix — silently turned the absolute `/etc/passwd` into an accepted
*relative* prefix, the traversal every other path in the module rejects. Exactly one
listing test failed while the whole rest of the suite stayed green.

## Who gets which view

`main()` builds the backend once and hands out views of it:

| Consumer | View | Why |
|---|---|---|
| `job_store` | `PrefixedStorage(storage, "jobs")` | `load_all` lists; without the prefix it would sweep up exports and the feedback log. |
| `report_generation`, both `ResultExporter` sites, `report_delivery` | `PrefixedStorage(storage, "exports")` — **the same object** | The stage writes the `.md`/`.pdf` that `report_delivery` serves. Two views of one backend work locally and diverge the moment either is remote. |
| `feedback_loop` | the **un-prefixed root** | The three feedback files have always lived directly under `AFIR_DATA_DIR`. A prefix would relocate them, so an upgraded VM would read *zero* analyst reviews from a log still sitting on its disk and tuning would silently revert to the configured baseline. Safe because `feedback_loop` addresses its files by exact name and never lists. |

**`PrefixedStorage.list_keys` strips the prefix back off.** A caller must get out what it
put in; otherwise `job_store` asks for `<id>.json`, gets back `jobs/<id>.json`, and every
evidence sidecar orphans. Its `close()` is deliberately *not* forwarded — the wrapper does
not own the backend.

**Local is not a branch.** `main()` wires the same code path for both backends. A
`if remote:` at any consumer would mean the VM run never exercises what a remote deploy
depends on, and the divergence would surface as a Report tab that works on a VM and 404s
on the App.

## Where state lives is not a property of the deployment MODE

Both directions are supported and both are ordinary:

- **A laptop or VM keeping its state on an external Volume.** `storage.backend: databricks`
  with a `host` and a `token_env` needs no App and no platform OAuth — it is a PAT and
  `requests`. Verified live, not inferred: the real `DatabricksStorage` and `TreeMirror` were
  run from a local process against `/Volumes/<app-catalog>/afir/state`,
  **29 checks, 29 passed**. That is the
  answer for a VM whose disk is not backed up, or two machines that must see one queue —
  subject to the single-replica rule below, which is about *writers*, not about hosts.
  Then verified one level up, through the config file rather than the class: a full `app.py`
  boot with `running_as_databricks_app()` **False** and a `storage:` block naming the Volume
  came up on `Durable storage: databricks`, no degradation, mirror active, and answered
  `{"storage": "databricks", "storage_ok": true}`. Flipping `catalog` to `""` in that same
  file answered `storage_ok: false` with *"storage.databricks.catalog is not set, so there is
  no Volume path to write to"* — beside `"status": "ok"`, which is the whole point: the server
  is live, the store is not, and one of those must not report the other.
- **An App keeping config and the pack in the image.** `mirror_config_and_pack: never` for a
  deployment that treats them as part of the release and wants a UI edit to be discarded on
  the next restart rather than to outlive it.

Both were already true in the code and reachable only by hand-editing a **commented-out**
YAML block, which is not "configurable" — it is "possible", and the difference is who can do
it. So `config/templates/main_config.yaml` ships a real `storage:` block with the pre-seam
local defaults, and `config_store.SECTIONS` carries a *Durable state (local, volume, or
database)* group over the twelve fields — every destination in the table above, `storage.root`
and the `sql.*` trio included, so the switch is genuinely made from the UI and not only from a
file. The block must be **real, not commented**: the patcher is
line-anchored, so a key that exists only inside a comment is a key it will not find, and the
switch reports `skipped` against a file that looks like it declares one
(`test_the_shipped_template_carries_a_real_storage_block`).

Every one of those fields is `restart`, and honestly so. The backend is built **once** in
`main()` and handed to five consumers; `applies: "live"` is a claim about *effect*, and
nothing here delivers it.

**But in a container, `restart` is not merely "later" — it is "never", and the report says
otherwise.** The form's whole point is that an operator can switch this from the UI; on an App they
cannot, and nothing tells them so. The first config comes from the read-only bundle, so the switch
is written to the *working* copy on ephemeral disk; the mirror that would push it durable does not
exist yet (`build_mirrors` returns `None` for a local backend — correctly), so `durable: true` is
reported with nothing skipped; and the restart that applies a `restart` field throws that disk away.
Measured end to end against `tests/fake_files_api.py`: accepted, `durable: true`, `local` again on
the next boot. **A configuration surface cannot configure where its own writes are stored** — the
one decision that has to be made before it can be trusted.

So the four bootstrap values read through `${VAR:-default}` in the shipped template and are set in
`app.yaml`'s env block (`AFIR_STORAGE_BACKEND`, `AFIR_UC_CATALOG` / `_SCHEMA` / `_VOLUME`), which is
the only channel the platform offers that early. Once they are set the loop closes — the store
exists, the mirror activates, and every later edit *is* durable from the UI, including the
`storage.*` fields themselves. A laptop or VM that sets none of them expands to the shipped local
defaults and is byte-for-byte unaffected. The trap and the empty-counts-as-unset rule are in
`docs/architecture/databricks-deployment.md`.

**A switch that is accepted and does not work looks exactly like one that works** — until
the restart that finds nothing. A blank `catalog`, or a workspace-scoped PAT answering 403,
sets `_fatal` and makes every `put_text` return `False`, which `main()` announces once, in a
boot log, at the one moment an operator who just changed the setting from the Configuration
tab is not reading one. So `/health?deep=1` answers `storage` (the kind), `storage_ok` and
`storage_detail`. Three states, deliberately distinct: `null` for not wired — a legitimate
configuration, and reporting it as a failure is how an indicator learns to cry wolf — `true`,
and `false` **with the reason**, because a bare boolean names none of the four causes.

## The two trees that are mirrored, not moved

`config/*.yaml` and `knowledge/<domain>/` are the only durable state that does **not** go
through the seam, and `src/storage/mirror.py` is why. Both are edited *at the byte level*
by code that exists precisely because a YAML round trip destroys the file: `config_store`
patches the value half of one line and preserves ~25k of load-bearing comments;
`pack_store` replaces a line range, verifies by re-reading the bytes **off disk**, and
keeps a content-addressed `.history` beside the tree. None of that survives being
re-expressed as "read a blob, mutate a string, write a blob" — and none of it has to,
because `os.replace` works perfectly on the local filesystem those modules are handed.

So the local tree stays the **working** copy, the durable store holds a copy, and two
movements bracket the unchanged write path:

- **down, at boot** — `seed_working_copies()` fills the working roots from the deployed
  bundle (only what is missing), then `MirrorSet.sync_down()` lays the durable store's
  contents over the top;
- **up, after each write** — the bytes that just landed are pushed. Exactly two hook
  points per store: `config_store._atomic_write`, and `pack_store._atomic_write_verified`
  plus the history index, the snapshot blob, `delete_file`'s unlink and `scaffold_pack`'s
  template copies.

Five rules, each of which fails invisibly if broken:

- **The store holds EDITS, not the tree.** Only a file written through an editor is ever
  pushed; everything else is re-read from the bundle on every boot. Seeding the whole tree
  would freeze the pack at whatever the first container uploaded, and every later release
  — a new catalog source, a corrected rule — would be invisible behind a stale copy of
  itself, with no error anywhere. With only edits stored, a redeploy delivers new files
  *and* an operator's edit to another file still wins.
- **A delete is an edit too**, hence `mirror_deleted.json`. The file being removed is one
  the bundle still ships, so a boot with no record of the removal seeds it straight back —
  a change that reports success and then quietly reverts. Pushing a file clears its
  tombstone, so delete-then-recreate resolves to "present".
- **`durable` is an explicit result field on every mutating function.** A write that
  reaches the working copy but not the durable store is strictly worse than a refusal: it
  looks identical to a save until the restart. `durable: false` reaches both UI surfaces as
  an error banner reading *NOT saved durably — in effect now, lost on restart*. It is
  `True` on every local deployment, where the file on disk **is** the durable copy. A
  no-op write is `True` too — nothing changed, so nothing is at risk, and reporting
  otherwise sends an operator hunting an edit they never made.
- **A push failure is reported, never raised.** The local write has already committed and
  is in effect for this process; turning the mirror's failure into a raise would report
  the edit as refused while the file on disk holds it. It also lands on
  `TreeMirror.degradation` — the same channel the seam already uses for "you asked for
  durable and are not getting it".
- **Nothing here runs on a local deployment.** `build_mirrors` returns `None` unless the
  backend is remote (`storage.mirror_config_and_pack: auto|always|never` overrides both
  ways; an unrecognised value is loud and treated as `auto`, because reading a typo as
  "off" would answer every save 200 and lose them all), and `seed_working_copies` no-ops
  when the working root and the bundle root resolve equal — which they do whenever
  `AFIR_CONFIG_DIR` / `AFIR_KNOWLEDGE_DIR` are unset. A mirror over local disk would copy
  a tree onto itself and, worse, `sync_down` would overlay the repo's checked-in pack with
  a stale duplicate.

Three details that are not arbitrary:

- **`.history` rides under the alias `history`.** `safe_key` requires a segment to start
  alphanumeric and *rejects* rather than repairs, because relaxing it to admit a leading
  dot would admit `..`. The alias is translated back on the way down — a one-way alias
  would store the undo history under a name `sync_down` then writes somewhere the editor
  does not read. A real directory colliding with the alias is refused, not merged. Every
  other dotfile is skipped in both directions, which is also what keeps `.DS_Store` and a
  stray `.git` out.
- **The pack tree has no suffix allowlist.** History blobs are digest-named with no
  extension, so an allowlist would carry the index and drop the content — a history
  listing snapshots it cannot restore. The config tree does filter (`.yaml`/`.yml`, depth
  1), which is what keeps `config/templates/` out.
- **`.bak` is never pushed.** `config_store` writes one beside every file it patches, and
  it holds the file as it was *before* the edit — including any literal credential the
  read path redacts.

**`main()` must re-read the config after `sync_down`.** Deciding *where* the durable store
is takes the config, so the first `load_config` can only see the bundle's defaults;
without the re-read the process runs on shipped values while the UI displays the
operator's, and every "restart to take effect" message in the app is false.

`tests/test_storage_mirror.py` covers all of it, with the round trip run once over the
fake Files API server as well as over `LocalStorage` — the `history` alias adds a segment
the remote backend validates on its own terms, and a local-only suite would learn that
from a deployment.

## Near misses worth keeping

- **`mtime = 0.0` is "unknown", not "the epoch".** The rewritten `prune` compared
  `obj.mtime >= cutoff`, so a backend that cannot report a modification time would have
  looked infinitely old and deleted the entire job queue on the first prune. It now skips
  an unreportable age; `tests/test_job_store.py` has a backend subclass that forces
  `mtime=0.0` and proves the job survives.
- **A monkeypatch seam is part of the contract.** The first `feedback_loop` edit set
  `self.storage = LocalStorage()` in `__init__`, which resolves `src.storage.local.data_dir`
  — not `src.feedback_loop.data_dir`, the symbol the whole suite monkeypatches. The backend
  is now resolved **per call** in `_backend()` for exactly that reason, and the same applies
  to `report_delivery`'s `exports_dir` seam.
- **An in-memory value that nothing on disk justifies.** `_persist_tuning` does not update
  `self._tuning_cache` when the write fails: a tuned threshold that exists only in memory
  would apply for the process lifetime and then vanish, which is worse than not tuning.
- **A returned path is a claim.** `ResultExporter` honours the caller's directory when no
  backend is injected, because `_run_export` builds `exports_dir() / name` and then
  *returns those paths* as its record of where the artifacts are. Defaulting to
  `exports_dir()` regardless made the record true only by coincidence — and false for any
  caller with a directory of its own. With a backend injected the directory is correctly
  irrelevant: the backend owns the location.
- **A root is not necessarily a `Path`.** `PrefixedStorage.root` did `base / self._prefix`.
  A remote root is a string, so it raised `TypeError` — out of a *property*, through
  `JobStore._describe`, out of `JobStore.save`, which is documented never to raise. The one
  caller only formats it into a message, so string joining is both sufficient and the only
  form both backends can supply.
- **A forwarding gap makes a guarantee a no-op that reports success.** `PrefixedStorage` did
  not forward `flush`, and `build_job_store` wraps in `PrefixedStorage(storage, "jobs")` — so
  `JobStore.flush` resolved to nothing and answered `True` having waited for nothing. The
  gate-durability guarantee, present in code and absent in effect.
- **A restore that is not checked is not a restore.** `_verify_readback` logged "restoring the
  previous content" without confirming the restore landed. Whatever tore the first upload can
  tear the restore, and an unverified one converts a loud ERROR into a stored document nothing
  knows is broken until a restart tries to load it. It now verifies, and **removes the object**
  when it cannot.
- **One `requests.Session` shared between the writer thread and its callers.** A Session's
  connection pool is not thread-safe; interleaved responses on one socket read back as "a
  document that does not verify" rather than as a threading bug. One session per thread via
  `threading.local()`.
- **A queueing backend was never drained on shutdown.** `main()`'s `finally` closed the
  retrievers and not the storage, so every write still on `DatabricksStorage`'s queue died
  with the process — "saved" reported at the moment of enqueue. It now calls
  `storage.close(10.0)` in a thread: bounded, because the platform's SIGTERM budget is 15s,
  and off the loop, because the drain is blocking.
- **Six round trips per button draw.** `report_delivery.artifact_inventory` did
  `os.path.isfile` + `getsize` six times per incident. Locally that is free; against a
  remote store it is six calls on every Report-tab render. It is now one `list_keys("")`.

## Adding a backend

1. Implement `StorageBackend`. Never raise for an absent key; never raise past `put_*` —
   return `False` and log. Set `kind` and populate `degradation` when you are not doing what
   was asked.
2. Register it in `build_storage`. An unrecognised name and a failed import both **fall back
   to local and say so loudly at ERROR, naming LOCAL DISK** — a typo in one config key must
   not stop the system from investigating, but it must not look honoured either.
3. Add it to `BACKENDS` in `tests/test_storage.py`. If it cannot run in-process, add a
   `scripts/probe_*.py` and record the measurement.
4. If it queues writes, `close(timeout)` must drain under that bound, and any write whose
   whole purpose is to outlive the process — a gate-opening save above all — must be
   flushed synchronously before the gate is announced. A queued write that dies with the
   container rebuilds the silent-degrade bug the store exists to prevent.
