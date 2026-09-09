# Who is asking

AFIR has no login of its own and does not want one. Behind the Databricks driver proxy every
request already carries an identity the platform validated; `src/identity.py` reads it,
`src/incident_input.py` acts on it, and `src/user_overlay.py` holds what each caller writes.

**One rule decides every ambiguous case: anything unresolved, unreadable or unverifiable costs
privilege and never grants it.** A header that cannot be trusted is a 403, a role that cannot
be established is `user`, a run whose owner cannot be established is visible to an
administrator only, and a token that fails to validate is not evidence of anything.

    identity:
      mode: auto            # auto | on | off
      admin_groups: ""      # comma-separated; a YAML list works too
      admin_users: ""
      workspace_host: ""    # falls back to the `databricks:` block
      validation_ttl_seconds: 900
      allow_self_elevation: true

## The platform-neutrality guarantee

**AFIR runs unchanged on a laptop, a VM, an Azure App Service, a Databricks App and a
Databricks cluster driver — and on the first four the whole per-caller model is *unreachable*,
not merely unused.** That is the property to protect when editing any of this.

`IdentityResolver.enforced = resolve_mode(mode, platform_default=running_on_databricks_driver())`,
and `running_on_databricks_driver()` reads `DATABRICKS_RUNTIME_VERSION`, which only the
Databricks runtime image sets. Off the driver, `auto` resolves to *not enforced*, so:

| | Result |
|---|---|
| `resolve(headers)` | returns `LOCAL_IDENTITY` without reading a header |
| `LOCAL_IDENTITY.role` | `admin`, because a single-operator deployment has nobody to be segregated from |
| `_edits_the_base(request)` | True — every write takes the path it took before this package existed |
| `stamp_owner(incident, …)` | returns the incident **unchanged** for `source == "local"` |
| `owner_scoped(storage, incident)` | returns `storage` unchanged, not a shared bucket |

So the layer code, the ownership filter and the per-caller storage prefix are all downstream of
a branch that never runs. `tests/test_identity_api.py`'s first test is the one to keep green
above all the others: it builds the interface **injecting no resolver at all** and asserts a
config patch and a pack save both land in the shared tree.

### Three states out of one tri-state, and the missing fourth is deliberate

`off` and a laptop `auto` need no identity at all; `on` and a driver `auto` require one. The
state that must not exist is *read an identity but accept its absence*: it makes an
unauthenticated request indistinguishable from the single-operator case, **at admin**. A typo
in `mode` resolves to `auto` and is reported (`mode_recognised`) rather than stopping the boot.

## The two proxy paths carry different amounts of identity

Measured against a real cluster driver proxy, not inferred from documentation:

| Header | API path (`Authorization: Bearer <PAT>`) | Browser path (cookie) |
|---|---|---|
| `x-databricks-auth-validated` | `true` | `true` |
| `x-databricks-auth-type` | — | `DB_AAD` |
| `x-databricks-user-name` | forwarded | forwarded **twice**, equal values |
| `x-databricks-user-id` | forwarded | forwarded **twice**, equal values |
| `x-databricks-user-token` | **forwarded** | `absent` |
| `x-databricks-non-uc-user-token` | forwarded | `absent` |
| `x-rstudio-username` | forgeable — see below | forgeable |

That difference is the whole design. A **forwarded token can be validated** — one call to
`/api/2.0/preview/scim/v2/Me` returns the caller's `userName`, `id` and **groups** — so the API
path yields `source="token"` and a group-derived role. A **name header can only be trusted**, so
the browser path yields `source="header"` and no groups at all.

`GET /Users/{id}` would answer the group question for the browser path and it is **403 without
workspace-admin**, so the server cannot look up its own caller's groups. Hence two fallbacks:

- **`identity.admin_users`** — a name list, the one that works on the browser path with no
  extra step and no grant.
- **`POST /api/v1/whoami/elevate`** — the caller pastes their own workspace token, it is
  validated, and the proven `userName` must **equal the name the platform already asserted**, so
  one user's token cannot elevate another. The proven groups are cached per caller for the
  process; the token itself is never stored and never logged (a validation failure is reported
  by exception *type*). `allow_self_elevation: false` switches the surface off.

### Forgery

Three rules, each with a test named after it:

- **`auth-validated` is read before the name.** A bare forged name earns nothing.
- **Disagreeing duplicates are refused, not chosen between** (`_agreed`). The proxy sends the
  identity headers twice with equal values, so a caller-supplied third value makes them
  disagree — and picking either one is picking a coin flip.
- **`x-rstudio-username` is never read** (`NEVER_READ`). Measured: the proxy strips a forged
  `x-databricks-*` but forwards this one, and a forgery sorts *first* among the duplicates.

A refusal is **403 and not 401**, because there is no credential for the caller to supply at
this layer — the ingress supplies it, so the remedy is which URL they used. Four paths are
reachable with no identity (`/`, `/health`, `/openapi.json`, `/docs`): a platform probe carries
none, and the page has to load before it can tell the caller who they are.

## Two roles

`admin` and `user`, because the question every surface asks is binary: **may this caller change
what everyone else sees?** Owners are administrators; contributors and readers are users.

`role_for` resolves in one order — a group named in `admin_groups` (evidence), then a name in
`admin_users` (declaration), then `user`. The reason is carried on the identity
(`role_reason`) and surfaced by `/api/v1/whoami`, because a caller who cannot see *why* they
are a reader has nothing to act on.

**Both lists are read from either shape** (`_folded`): a comma/newline-separated string or a
YAML sequence. The string is not a convenience — the Configuration tab's line-anchored patcher
refuses a key that opens a block, so a scalar is the only shape it can write, and a string
folded as a sequence would iterate *letters* and make every single-character name an
administrator.

## Two trees, and the asymmetry is WHERE a write lands

Both editable trees — `config/*.yaml` and `knowledge/<pack>/` — work the same way, and neither
is admin-only-write:

- an **administrator edits the base**: the working copy this process loaded, mirrored exactly as
  it was before any of this existed;
- **everyone else edits their own layer**, keyed by the same relative path the base uses.

The seam is one predicate, `_edits_the_base(request)`, at the top of every mutating handler.
A layer holds **only the files that caller changed**, for the same reason `storage.mirror` holds
edits rather than a tree: a layer carrying the whole tree would silently override the next
release.

### A layer is a DRAFT, and every layered write says so

This is the boundary to state out loud, because a 200 will not state it: the pack and the
config **this process runs on** are built once at boot from the base. A layer is durable, it is
merged forward when the base moves, and it is what its author reads back — but it does not
change a running verdict. So a layered response carries `layer: true`, `effect: "draft"`, a
`note` naming who can promote it, and `restart_required: false`, because no restart applies a
draft either. A layered pack write's diagnostics are stamped `validate_scope: "base"`, since
they describe the files on disk and beside a draft would read as a verdict on what was just
saved.

A non-administrator with **no durable store configured** has no destination at all, so the
refusal is a 503 naming the one thing that still works: ask an administrator to apply it.

### When the administrator moves the base

Every base write passes one seam (`_pack_write_result` for the pack, the config handlers for
the config) and that seam re-merges everybody's layers (`rebase_all`), reporting the result on
the **administrator's own response** — a release that silently conflicted with somebody else's
draft is a release nobody knows to look at. The merge is line-based (`difflib`
`SequenceMatcher` opcodes) against the text the caller forked from, kept beside their edit in
`<file>.base`: `.history` also holds it, but a history blob can be pruned, and a missing merge
base turns a clean merge into a conflict the caller did not cause.

| State | Meaning |
|---|---|
| `clean` | the base did not move under this file |
| `merged` | both sides' edits applied |
| `conflict` | conflict markers written **into the draft**, which is kept — losing an edit is worse than keeping one that no longer applies cleanly |
| `adopted` | the caller's text equals the new base, so the override is redundant |

`DELETE /api/v1/overlay/{label}?path=` is the way back to the base view and the only way out of
a conflict; a conflicted draft is kept deliberately, so discarding it has to be a choice.

### The three writes with no draft to be

Every other write layers. These three are **refused** for a non-administrator
(`_forbid_non_admin`, a 403 naming what was refused, the caller's role, the reason, and the
elevation remedy):

| Route | Why not a layer |
|---|---|
| `POST /api/v1/knowledge/scaffold` | a new pack has no base to layer over |
| `POST /api/v1/knowledge/{pack}/import` | this is the **release verb** — pushing the version everybody runs — and a bulk overwrite of the shared tree is the one thing a draft is not |
| `POST .../assist/{session}/apply` | the plan's ops resolve anchors against the base and are all-or-nothing across several files; applying one into a layer would be a second implementation of the whole op vocabulary, and a half-applied plan is worse than a refused one |

The assistant's asymmetry is deliberate: **anyone may ask for a plan and read its preview**,
because that is where the help is; only the write is refused. The guard precedes the session
lookup, which is what the test asserts — a nonexistent session answers 403 to a user and 404 to
an administrator.

## Whose run is it

`stamp_owner` records `owner` (the caller's storage segment) and `owner_name` on the
**incident**, rather than beside it, so ownership rides through `export_job` / `import_job` and
`Job.snapshot` with no field of their own. Then:

- **`_visible(request, rows)`** filters jobs, batches and open gates on the row's own `owner`.
  An administrator sees everything.
- **An unowned row is administrator-only.** That covers pre-seam runs and every run of a
  deployment that resolves no identity — attributing one to whoever happens to ask would be
  inventing a claim.
- **`_lookup_job` returns None for another caller's job, so the answer is 404 and not 403.** A
  403 confirms the id exists; a job id is guessable in a way a report is not, and the response
  must not name the owner either.
- **A batch is scoped like the jobs it labels**, on all three of its routes. A cancel that
  refused with 403 would still let one caller stop another's runs by guessing.

Artifacts and jobs are segregated at the storage seam that already existed:
`PrefixedStorage(storage, f"users/{segment}/…")` via `owner_scoped`. The segment is folded from
the user **id** where there is one (stable across a display-name change) and satisfies
`src.storage.base._SEGMENT` — an email's `@` becomes `-`, a traversal attempt becomes a flat
name, and an empty value becomes `unknown` rather than the prefix root.

    jobs/users/<segment>/<id>.json
    exports/users/<segment>/fraud_report_<id>.md
    users/<segment>/layers/config/<file>            + .base + .meta.json
    users/<segment>/layers/knowledge/<pack>/<rel>   + .base + .meta.json

The segment sits **inside** the subsystem prefix for the first two and **above** the layer root
for the last two, because the first two are narrowings of a store somebody else already
prefixed (`PrefixedStorage(storage, "jobs")`, `…, "exports"`) while `UserLayer` spells its whole
key. Neither order is a decision the reading side may guess at — see below.

### Reading an artifact back is a SECOND decision, and it was not taken

`owner_scoped` is on the write path only, so every reader in `report_delivery` resolved the
shared `exports/` root. The two listers appeared to work and the four readers did not, for the
same reason: `list_keys` walks recursively and `StoredObject.name` is the basename, so matching
the name found `users/<other>/fraud_report_X.md` under its bare filename while
`get_text("fraud_report_X.md")` — the root-relative key — found nothing. Measured live before
the fix, on one owned run and one unowned one: the owned run's report and evidence answered
FAIL while `/artifacts` beside them reported all six artifacts present with real byte counts
(`report_md` 70491, `evidence_raw` 1817496), and the unowned run served. So **every** report
produced since the identity round was unreachable through the API and the UI, and the Report
tab listed other callers' incident ids.

The fix is one `owners: Sequence[str] = ()` argument on all six read entry points
(`artifact_inventory`, `list_incidents`, `read_markdown`/`read_pdf`, `resolve_report`,
`resolve_evidence`, `evidence_outline`), resolved through `_views(owners)`:

- **The shared root is tried FIRST and is never in `owners`.** A deployment that stamps no
  owner passes no segment, hits on the first view, and makes exactly the one call it made
  before the argument existed — which is what keeps the platform-neutrality guarantee above
  true of the read side too.
- **A job-scoped route passes the RUN's own owner**, not the caller's. `_lookup_job` has
  already refused another caller's run, so the run's segment is authoritative and exact: an
  administrator reading somebody else's finished investigation gets that run's segment and no
  other.
- **An incident-keyed route has only an id**, so it offers the caller their own segment — and
  an administrator every segment, which is the visibility `_visible` already grants them.
  `owner_segments()` enumerates the subtrees; it strips `users/` first, because `list_keys`
  answers root-relative keys **whatever prefix it is given**, so reading `key.split("/")[0]`
  off that listing returns the literal `users` for every row and resolves to no segment at all.
- **Both listers match `obj.key` and the shared view keeps only its flat keys.** Two
  mechanisms for one property, so neither mutation kills a route-level test on its own — hence
  `_own_keys` has a unit test of its own.

## What a run does not record: the access journal

Everything above answers *whose run is this*. The complementary question — **who is using this
deployment, and what did they do** — has no answer in any of it, because every durable record
AFIR keeps is about a run. A caller who browses the console, reads someone else's report, is
refused at the door or edits the shared configuration produces no job document, and neither
does the fact that anybody was here at all. On a shared deployment that is most of the traffic.

`src/audit_journal.py` closes it. The HTTP layer is the only place that sees a caller, so
`_identity_middleware` records there and not per handler — a handler that forgot to record
would be an unrecorded one, and the middleware is also the only place that sees the two events
no handler ever runs for: a refusal, and a page load by someone who then leaves.

Four decisions, each taken against an obvious alternative that fails silently:

- **Durable, because the obvious sink is not.** A log line goes to stdout, which on a cluster
  driver is a file on ephemeral local disk — an audit trail a restart erases is not one. Entries
  are appended to the storage backend beside the feedback log, one object per UTC day
  (`audit/access-YYYY-MM-DD.jsonl`), so retention is a listing plus a delete and no object grows
  without bound. `prune()` compares **the date in the key** and not an mtime: an appended file's
  mtime is the last *write*, so today's object looks freshly created every day and an old one
  looks new the moment it is read back.
- **Batched, because the backend charges per write.** A UC Volume / DBFS append is a
  read-modify-write of the whole object serialised on one writer thread. An append per request
  would be quadratic over a day *and* would put an investigation's writes behind a queue of page
  loads. Entries land every `audit.flush_seconds` or every `audit.max_buffer` entries, whichever
  is first; the loss window is stated in the config template rather than hidden, and a clean
  shutdown drains it before the storage queue closes.
- **Best-effort in one direction only.** Nothing here may fail, slow or reorder a request: every
  method swallows its own errors, a full buffer drops its **oldest** entries and journals how
  many (`journal_overflow`, so a census over a record with a hole in it cannot read as a complete
  one), and a sink that is refusing writes is reported **once** rather than per entry.
- **A no-store instance is disabled, not degraded.** Buffering into a process with nowhere to put
  the result is worse than saying so, so `enabled` is false whenever `storage is None`.

And two rules it inherits from this file rather than inventing:

- **`enabled: auto` follows the identity layer** (`running_on_databricks_driver()`), for the same
  reason `identity.mode` does: on a laptop every caller is the same local operator, and a journal
  there records one person visiting their own machine. `auth` is stamped on every entry, so a
  `local` entry can never be read as an authenticated one — and where an identity IS enforced
  nobody is local, so the pages that run before a caller can be named (`/`, `/afir`) are
  journalled `unresolved` rather than under the fallback's name. An anonymous browser hit
  recorded as `local` is a census naming a caller who does not exist.
- **`GET /api/v1/audit` is administrator-only**, because it names every other caller. It answers
  `200` with an empty list when the journal is off rather than `404`: "nobody has done anything"
  and "nothing is being recorded" are different answers, and `journal.enabled` is which one the
  reader has.

The one thing the journal made possible elsewhere: **the actor on a durable decision is now the
resolved caller.** `_actor(request, claimed)` returns `identity.user_name` wherever the identity
did not come from the `local` fallback, and the client-supplied `actor` field only where it did.
A gate decision, a stage override, a plan edit and every pack write are signed by whoever the
ingress named — a free-text form field is a claim, and a durable decision may not be signed with
somebody else's name. The console fills that box from `/api/v1/whoami` and locks it, filled
rather than hidden, because who is being recorded is the point.

The two records are read together (runs per owner, and the journal) through the same storage
seam the app writes with, so a census answers about a Volume, DBFS or local disk without being
told which.

## Whose credential is it

Everything above segregates what a caller *reads and writes*. `src/user_secrets.py` segregates
what their runs **authenticate with**: the deployment ships working credentials, and a caller
with a token of their own may replace one for their own runs. Four decisions, each against an
alternative that fails silently.

**Keyed by environment-variable name, not by subsystem.** One name commonly backs several
subsystems at once — on this deployment a single token serves the LLM, the embeddings and every
SQL warehouse — so a per-subsystem form asks the same secret of the same person four times and
lets three of the four go stale. `offered_names(main_config, llm_config)` reads the *live
config* for every name a reader resolves and maps it to what it reaches; `withheld_names`
reports the rest with their reason, because a surface listing four names and dropping three
others reads as complete.

**A name nothing reads is refused.** Accepting one would store a secret, report success and
change nothing about the caller's runs. `resolve` re-checks the offered set on every read too:
a name that stopped being read must stop being honoured, not silently apply to nothing.

**The value is never returned**, in either direction — no API read-back, no UI field, no log
line. A `fingerprint` (8 hex of SHA-256) confirms a paste landed and tells two credentials
apart. **Stored durably and not encrypted, which is a statement rather than an omission:** the
same store already holds job evidence under the same access control, so encrypting only the
token — with a key that would have to live beside it — protects nothing while reading as if it
did. The requirement is about display, and display is closed.

**Durable state is excluded on purpose.** The job store, the audit journal and the config
mirror are shared by construction, so a personal credential there would either do nothing or
hide the caller's own history from them. So would ES / Snowflake sign-in, for a different
reason: a username and a password are expanded to *values* by `_expand_env` at boot and the
client is built once, so there is no name to replace and no per-call seam to replace it at.
That is reported rather than omitted.

### Who is asking, out of band

The readers are a retriever, an embedding provider and the LLM client, and none of them has any
notion of a caller — nor should acquire one. So the segment travels on a `ContextVar`, bound at
**two** seams with deliberately different sources:

- `_recorded` (`src/incident_input.py`) binds `request["identity"].segment` for the duration of
  the request, and resets it in the same `finally` that clears the recorded actor. A task
  created inside that window inherits the context copy, so a detached run keeps the binding
  after the request returns.
- `_run_stages` (`src/pipeline_runner.py`) binds `owner_of(job.incident)`, **not** whoever
  started the loop: a job submitted by one caller and resumed, retried or restored by an
  administrator is still the submitter's run, and must not start using the administrator's
  token half way through. An unowned run keeps the inherited segment, which on a deployment
  that resolves no identity is the one operator.

`personal_value(name)` returns `None` with no store installed and no caller bound, so a laptop,
a VM and an App Service are byte-identical to the tree that shipped before the file existed.
`resolve_env(name)` is `os.environ.get` there, exactly.

### A session header is fixed at construction

The two aiohttp readers were the interesting fix. `DatabricksRetriever` and `RestRetriever`
baked `Authorization` into the shared session, which outlives a run — so an override arriving
later was ignored *and* a session opened during one caller's run would have handed that
caller's token to everybody after them. Both sessions are now built with **no** Authorization at
all and every request site passes `headers=self._auth_headers()`, so a future site that forgets
it gets a 401 rather than silently sending somebody else's credential. The poll loop re-derives
per poll, because a statement can outlive the token it was submitted with.

Priority is the same in all four readers — **personal → SDK-refreshed → static** — and the
embedding provider had to start carrying its credential's *name* beside the value
(`token_env=`), because a value alone cannot say which name it came from.
`src/rag/sources/databricks_source.py` is deliberately not wired: it builds the shared RAG index
at boot, before any caller exists.

## Near misses

Four options were considered and not taken, and each is worth knowing about because each looks
like the obvious answer:

- **Personal workspace files as the per-user store.** Offered, and rejected in favour of the
  Volume prefix: it would be a second storage implementation reachable only in one deployment
  mode — the unexercised copy that rots — where `PrefixedStorage` is the seam jobs and exports
  already use, so every backend gets per-user state for free and `LocalStorage` stays the
  oracle it is in `tests/test_storage.py`.
- **Cluster ACLs as the role source.** They decide who reaches the proxy at all, which is a
  different question from who may push a pack; and reading them needs a permissions API call
  per request against a grant this deployment does not have.
- **Admin-only *write* on the config.** The first draft of this; wrong, because a user editing
  their own copy of a config file is exactly as legitimate as editing their own copy of a pack
  file. Both trees layer.
- **A `list` field kind for the two admin lists.** There is no such kind, and adding one would
  need the patcher to write a YAML block it cannot write — it would report `skipped` on every
  attempt. A comma-separated scalar plus a reader that accepts both shapes is the whole feature.

And two refusals on the credential routes were caught by one `except` clause. `KeyError` **is**
a `LookupError`, and `_require` raised `LookupError` for "no durable store" while raising
`KeyError` for "nothing reads that name" — so with the wider clause written first, every
unreadable name came back **503 with the repr of that name** and the branch that lists the names
this deployment *does* read was unreachable, as was the DELETE's 404. The two need opposite
answers (come back later, against you asked for the wrong thing), so the unavailable case has a
class of its own, `SecretsUnavailable(RuntimeError)`, and cannot be caught by a clause meant for
the other. Both were found by the route tests and neither by the store's own: at store level the
exception *types* are right, and what was wrong was which one a handler saw. In the same pass the
`GET` learned to say `reason` when there is no store at all and not only when there is one that
reads nothing — an amber state that names no consequence sends the reader to the wrong fix.

The third was the panel contradicting itself on the one name that matters. `withheld_names`
listed the durable store's `token_env` flatly, and on the driver-proxy deployment that is the
*same* env var the reasoning endpoint and both warehouses read — the only offered name — so the
panel offered it at the top and, at the bottom, said replacing it "would either do nothing or
hide your own runs from you". Both halves were true of a different reader, which is why the flat
row was wrong rather than merely terse: the store reads `os.environ` **directly**, on purpose, so
a personal value never reaches shared state, while every per-call reader goes through
`resolve_env`. The offered set is therefore an argument to `withheld_names`, and where the name is
offered the row is named for the subsystem that keeps the deployment's value rather than for the
credential. Untestable from the store alone in the other direction, too: dropping the argument at
the call site leaves the unit test green, so the route test builds a config with both on one name.

And one field name had to change rather than the detector that caught it:
`allow_token_elevation` tripped `is_secret_key` (which splits a leaf on `[_-]` against a word
list containing `token`), so the field is `allow_self_elevation`. Widening the secret detector
with an exception would have been a hole in the redaction path for the sake of a spelling.

## Where it is tested

| File | Covers |
|---|---|
| `tests/test_identity.py` | the resolver: modes, both paths, the three forgery rules, elevation, the segment fold, construction |
| `tests/test_user_overlay.py` | the three-way merge, the four states, the splice primitive |
| `tests/test_identity_api.py` | that the **handlers route into** all three — over a real server, starting with the platform-neutrality guarantee, and ending with the three credential routes: the admin with no override in either direction, 400-versus-404, and the request binding its own caller |
| `tests/test_audit_journal.py` | the journal itself, grouped by the failure each property prevents |
| `tests/test_user_secrets.py` | the store: the refused name, the absent read-back, per-caller isolation, and the off-deployment byte-identity |
| `tests/test_ui_server.py` | the shipped template carries a real `identity:` block whose two lists are scalars |

A layer nothing writes to reads exactly like a layer that works, and a caller whose edit went
to the shared tree finds out when somebody else's run changes behaviour. That is why the
routing has its own file: the resolver and the merge can both be perfect while no handler asks
either of them.
