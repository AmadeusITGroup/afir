# Retrieval: retrievers, robustness, query shaping, source wiring

## Retriever abstraction

`LogRetrievalEngine` (`src/log_retrieval.py`) is a dispatcher that builds one `DataRetriever` per configured source, keyed by `type`, and routes each `RetrievalQuery` by `target_log_source` (it passes `auth` only to the `databricks` retriever). `DataRetriever.retrieve(query) -> list[dict]` is the contract. Current implementations have the **LLM generate the backend query** (ES|QL for `ElasticsearchRetriever`, SQL for `DatabricksRetriever` via the `/api/2.0/sql/statements` REST API with async polling), and the retriever — not the LLM — caps row count. **Databricks Genie is a planned future drop-in** behind the same interface (`type: databricks_genie`), no pipeline changes needed.

`DatabricksRetriever._execute_sql` is **cold-start tolerant**: it submits with the API-max `wait_timeout: "50s"`, then polls the **same `statement_id`** (never resubmits — a resubmit would kick off another warehouse start) up to `max_poll_attempts × poll_interval_seconds` (default 60 × 5s ≈ 5 min, covering a cold/auto-stopped warehouse start). On budget exhaustion / per-source timeout / outer cancel it best-effort **cancels** the orphaned statement (`POST .../statements/{id}/cancel`, `_cancel_statement`, never raises).

**`warehouse_id` must be the 16-hex SQL-warehouse id, NOT the workspace/org id from the URL** — using the workspace id yields `403 Forbidden` on every statement (this was the real cause of the "Databricks 403" — the serving endpoint 200s with the same token).

Implementations: `ElasticsearchRetriever` (ES|QL), `DatabricksRetriever` (UC SQL), `SnowflakeRetriever` (Snowflake SQL via `snowflake-connector-python`, sync driver wrapped in `asyncio.to_thread`, schema discovery from `information_schema.columns` scoped to the pack's `databases`/`schemas`/`objects`), and `RestRetriever` (ServiceNow Table API — maps entities to ServiceNow fields from the pack's per-source `entity_bindings`, builds an encoded `sysparm_query`, paginates with `sysparm_limit`/`sysparm_offset`). Both of the latter use the standard `(config, llm_client, knowledge_pack=)` constructor (non-auth branch).

## Retrieval robustness

`_gather` wraps each source's retrieval in its own task under a **per-source `asyncio.wait_for` timeout** (`log_sources.per_source_timeout_seconds`, default 20) so one slow/unreachable backend fails in ~20s instead of burning minutes through transport retries; unfinished tasks are cancelled and drained in a `finally` (sibling cleanup on outer cancel).

A retriever may **raise its own cap** via `retrieval_timeout_seconds` in its config (read by `_gather` via `_source_timeout`, falling back to the global default) — Databricks sets ~330s because a cold warehouse start alone can take minutes; the key rides from `backends.databricks` through `_merge_endpoint`.

**A pack-declared `retrieval_class: primary` source gets a far larger budget**, resolved at config-build time by `_apply_primary_budget` (`log_sources.primary_source_timeout_seconds`, default **7200s / 2h**; per-source `primary_retrieval_timeout_seconds` wins). It writes all three coupled budgets from ONE number — the engine's `retrieval_timeout_seconds`, the retriever's `statement_timeout_seconds`, and the `max_poll_attempts` floor — because the lowest wins silently. `_source_timeout` deliberately does **not** re-derive it: computing the same number in two places is how the two caps drift apart. Why it can't be a backend setting, and why a cap firing here *decides* the verdict instead of degrading it (measured on job 047a603c), is in [knowledge.md](knowledge.md#retrieval_class-primary--what-a-sources-absence-costs).

**Why 2h and not 3000s: the MARGIN is the budget, not the number.** An accepted baseline completed its primary source in **2990.6s against a 3000s cap** — 27 seconds of margin, 0.9%. A re-run of the same incident with a byte-identical final query timed out, and the cause was not that statement: the whole warehouse was slower that night (a 2-row reference lookup on the same backend went 11.9s → 386.6s; an event-log source 71.8s → 424.6s). The result was **15 conditions `unknown`** where the baseline had 2 — the exact failure `retrieval_class: primary` exists to prevent, reintroduced by a cap set just under the observed cost. A cap tuned to a measurement is tuned to that measurement's *load*, so the only safe setting is one a genuinely-stuck statement reaches and a merely-slow one does not.

**It is editable in the running process** (`log_sources.primary_source_timeout_seconds`, exposed as the *Retrieval budgets* section of the Configuration tab; takes effect on the next retrieval, not on a statement already in flight). That needs one extra step to be true rather than merely claimed: the resolved budget is **copied** onto each retriever's own config dict at build time and nothing re-reads it, so `LogRetrievalEngine.refresh_primary_budgets()` re-resolves every primary source from the current config, wired from `main()` into `IncidentInputInterface(on_live_reload=...)`. A deployment that wires nothing gets the field reported as `restart_required` instead — `applies="live"` is a claim about *effect*, not about storage, and a phantom apply here means the operator raises the cap, watches the old one fire, and has no reason to doubt the screen. Re-application is bidirectional: `_apply_primary_budget` stamps `_primary_budget_applied` so it can tell **the number this mechanism wrote** (ours to rewrite, up or down) from **a number the operator configured** (only ever raised), which is what lets a budget be *lowered* rather than silently no-op'd by the only-ever-raises rule.

**Opt-in extended retrieval:** the defaults above are always respected; a user can grant slow sources more time per-incident (`extended_retrieval: true` on the incident dict — accepted on the freetext and job endpoints) or globally (`log_sources.extended_retrieval`). When on, `_gather(extended=True)` raises each source's cap to `extended_retrieval_timeout_seconds` (per-source override → global → `4×` the normal cap) via `_source_timeout(extended=True)`, and **also widens each Databricks retriever's poll budget** (`_apply_extended_poll_budgets` bumps `max_poll_attempts` to cover the extended cap, restored in the `finally` by `_restore_poll_budgets`) — because the retriever's own `max_poll_attempts × poll_interval` (~300s) would otherwise give up before the engine's extended cap. **Both caps must move together or the lower one wins.** This is the escape hatch for genuinely slow sources (e.g. a huge unpartitioned view) where the analyst is willing to wait.

**The row cap is a DIFFERENT limit from every budget above, and the label said otherwise.** `max_results` bounds *rows*; every timeout on this page bounds *seconds*. Extended retrieval buys a slow source more seconds and not one more row — which the Investigate checkbox used to deny, reading *"longer caps for slow sources"*: an operator ticked it, got `Retrieved 500 rows from <source>` and asked why the cap still bit. It now reads *"longer **time** for slow sources (not more rows)"* and its tooltip names `max_results` as the thing to change instead. Three consequences worth knowing:

- The default lives in **one** place (`_DEFAULT_ROW_CAP = 500` in `log_retrieval.py`, read through `_row_cap_default()`), not copied into each of the four `_merge_endpoint` backend branches as it was. `log_sources.max_results` overrides it globally and is editable from the Configuration tab; `_row_cap_default()` is deliberately forgiving of a non-numeric or absent value (a *defaulting* helper must not be the thing that raises).
- **An endpoint's own `max_results` still wins.** `backends.<kind>.<name>.max_results` is read first, so a config that pins `500` per endpoint is unaffected by the global knob — raise it per endpoint, or delete those lines to inherit. The field is `applies="restart"` because the cap is copied onto each retriever when the engine is built.
- **A truncated result now says so.** `_gather` compares each source's row count against `row_caps()` and appends `— TRUNCATED at the N-row cap (max_results); the real total is higher` to the `completed` progress message. `N rows` and `N rows, and there were more` are different findings; reporting them identically is the same class of failure as an empty source reading like a slow one. **The same `row_caps` map now also reaches the scope sweep** — `build_scope_discovery(…, row_caps=…)`, so a truncated widen-the-scope step reports floors and an explicit "unknown, not clean" instead of the wording an exhaustive one uses (see [knowledge.md](knowledge.md) and `src/usecases/base.py`).

`retrieve`/`retrieve_with_tunnel`/`_gather` take an optional `progress_cb(source, status, message)` (`completed`/`timeout`/`failed`) that the `log_retrieval` job stage wires to the emitter as live `source_progress` SSE events.

**The backend query is published BEFORE it is sent.** `DataRetriever.retrieve` takes an `on_query(text)` callback, called by every implementation through `publish_query` (`src/retrievers/base.py`) the instant the final query exists — after every guard has rewritten it, so what is published is what runs. `_gather` builds one callback per source (a factory, not a closure over the loop variable) and reports it as a `query_ready` **annotation**: an extra fact about a source that is still `running`, deliberately *not* a lifecycle status. So `progress_cb` in `pipeline_runner` skips it when recording `source_outcomes` (the dict the health scorer reads to tell an empty source from a timed-out one), and the UI row carries the query forward without touching the badge or the done-count.

Why the ordering is the whole point: `last_generated_query` used to be assigned on the line *after* the backend call returned, so the sources whose query most needed review — the slow ones — were exactly the ones that reported none. On job 047a603c the primary source ran 30 minutes, hit its cap, and its SQL never appeared anywhere; the operator could see *that* it timed out but not *what* it asked. Unlike `guidance` (whose loss would silently discard an analyst's correction, so an undeliverable one fails loudly), `on_query` is observability: a retriever that does not accept it still retrieves, and a raising callback is swallowed.

**TLS toggle:** both `DatabricksRetriever` (module-level `_build_ssl_context` → `aiohttp.TCPConnector(ssl=...)`) and `ElasticsearchRetriever` honor optional per-backend `verify_ssl: false` (disable verification — dev/self-signed only) or `ca_bundle: <path>` (trust a CA); neither key set = unchanged default. Keys ride from `backends.<kind>` through `_merge_endpoint` into the retriever config. This fixed the corporate self-signed-cert `SSLCertVerificationError` that made Databricks queries appear to hang.

## The retrieval cache — what may be cached is an ANSWER, and only the caller can say what one is

`src/retrieval_cache.py`, off unless `log_sources.cache.enabled`. A deployment that declares nothing gets a disabled cache: every `get` misses and every `store` refuses, so switching it on is the only way to change a single row.

**The question is the key, never the query.** `cache_key` hashes the source, the question, the window, the sorted entities (type + value + form), the row cap and the analyst `guidance`. The generated backend query is deliberately excluded: it does not exist until the retriever has run, and it is LLM-authored, so keying on it would mean a cache that can only ever miss. The row cap is in the key because **a result at the cap is a floor rather than an answer** — re-serving a 500-row entry to a run configured for 5000 would keep a truncation the operator has already paid to remove — and `guidance` because a rejected gate rewrites the query, so a corrected ask that hit the uncorrected answer would return exactly what the rejection was about.

**Every non-answer is excluded by construction, not by a second implementation of the rule.** The four ways a source fails to answer — a timeout, a backend error, a cancel, and an empty result from a query still carrying an unfilled `'<...>'` — are all *absent* from `logs`, which is the invariant the sections below exist for. `store` is called from `logs` at the end of `_gather`, so it needs no idea any of them exist, and caching one (which would turn a transient gap into a persistent INSUFFICIENT DATA) is not something a future branch can reintroduce by forgetting a check.

**An empty answer is an answer, and it is still not stored by default.** Zero rows from a keyed reference lookup IS the finding, so re-serving it is sound; but it is also the answer a re-ask most often changes, and a cached empty is indistinguishable from a source that has nothing until the TTL expires. `empty_ttl_seconds` therefore defaults to `0` — nobody keeps them unless a deployment says so — while a non-empty answer rides `ttl_seconds` (default 3600).

**A hit replays two facts rather than recomputing them.** `CacheHit` carries the published query text and `key_enforced`, because no retriever ran and `last_key_enforced` on a shared retriever would answer for whichever job published last. Without them a cached keyed lookup reads as unkeyed and its zero-row *finding* degrades to `unknown` — the `key_was_enforced` defect arriving through a new seam.

**And a hit says how old it is.** The row count a stage announces is read by the health scorer and narrated in the report, so a hit that looked like a fresh retrieval would put an earlier scan's rows under today's heading with nothing anywhere saying so: the `completed` line reads `Retrieved N rows from <source> (cached 4m ago, not re-queried)`, and it still carries the truncation marker, because a served answer that was capped is as capped as it was when it was stored.

**`extended=True` bypasses the read and still refreshes on the way out.** Extended retrieval is the operator deliberately asking again with a bigger budget; answering that from a store answers a different question.

Stored **after** `_decode_encoded_fields`, so what a hit returns is what the storing run worked from. That is safe only because `decode_logs` is idempotent — a hit runs through the decoder again rather than needing an "already decoded" flag nothing else would read.

The cache is bounded on entries **and** rows (a dict-as-LRU, 64 / 200,000 by default), because one truncated primary source can carry more rows than sixty small ones together; a single answer larger than the whole row budget is **refused** rather than admitted, since admitting it would evict every other entry to hold one source. An entry whose `stored_at` is missing or unreadable is *discarded* with a warning rather than read as very old or brand new: the timestamp is what makes an entry expirable, and an entry nobody can date is one nobody can retire. It is per process and not durable — it holds result sets, one replica does not see another's, and a restart empties it. `/health?deep=1` reports `retrieval_cache` with the hits and misses **beside** the rate, because a rate over three lookups is not a measurement.

## Query shaping — entities are OR-ed EVIDENCE, not all-mandatory ANDs

Applies to all three query-generating retrievers. An incident carries several entities — an org unit, a login, an actor identifier, the subject's own locator, its assets, and often an alert-derived value that is really something else in disguise. ANDing them all into one WHERE returns **0 rows the moment any single value is wrong or belongs to a different actor**, and that is the ordinary case: the identity that *created* a record routinely differs from the one an alert names as having acted on it.

So the ES|QL (`_generate_esql`), Kibana-DSL (`_generate_dsl`) and Databricks-SQL (`_generate_sql`) prompts all instruct: the **date range is the hard scope**; entity filters are **evidence to find**, combined with **OR** (`bool.should minimum_should_match:1` for ES; `col IN (...)` / OR-groups for SQL) so a row matching *any* entity is returned. When **no** entity maps to a filterable field, the prompt restricts to the time range only (never invents unmatchable clauses).

For huge backends the SQL prompt additionally requires **one selective identifier** (the subject's own locator) AND-ed as the scan-bounding filter, with the weak entities OR-ed on top — selectivity for performance without over-constraining for correctness. `render_filters` renders multiple values of one entity type, and multiple types sharing a field, as `field IN (...)  (match ANY)` rather than collapsing to the last value. This is what makes a real incident's data actually come back; verified end-to-end on a live pack, where the heaviest source went **0 → 45 rows** once the org-unit/actor/asset-class over-ANDs were dropped and the subject locator kept as the selective filter.

### One actor, several ROLES — and no mechanism here can OR two columns

The rule above is about several *entities*. A distinct failure is one entity on several **columns**: a record stores the identity that CREATED it separately from the identity that ACTED on it later, and an alert names whichever one it happened to observe. A filter pinned to the wrong side returns **0 rows**, and on a scope-widening source that reads identically to *"this actor touched nothing else"* — a containment miss dressed as a clean result. Measured on the deployed pack's scope-sweep source (its largest table): creator-side, as the hints mandated, **0 rows in 306.8s** against an incident whose own paperwork listed ~60 issued documents; the other actor side alone **159 subjects / 1185 documents (168.4s)**; **the OR of both sides 593 / 3509 in 76.3s** — *faster than either arm*, so the two-sided form is not a cost to weigh. Both arms stay load-bearing: on an earlier incident the alerted identity really was the creator and the creator-side sweep returned exactly the expert's set.

**Neither `entity_bindings` nor any of the five guards can express this, and it is worth knowing why before reaching for them.** The second actor's columns are leaves of a `LATERAL VIEW` alias, so they are absent from the discovered schema and `_in_schema` drops a binding naming them as a **STALE BINDING** — correctly, since a binding claims a *filterable column*. And `map_entities` returns **one field per entity type**, which `render_filters` then renders: listing both sides makes the mapper *choose*, replacing "always the wrong side" with "whichever it picked" — the same defect on half the incidents rather than none. The guards AND (`require_all_entities`, `identity_keys`), drop (`never_filter`), or repair a bound (`partition_columns`, `epoch_time_columns`); **none of them unions two columns**. So this one legitimately lives in the source's `query_hints`, which is the only mechanism that can state a disjunction — an exception to the "a prompt cannot beat a prompt" rule below, because there is no competing instruction here, only an absent capability. Verified live: the generator emitted the OR of both arms with the partition and the document date bounded, and the source returned rows where it had returned none.

**A fix that unblocks a source can hand its result straight into the row cap.** The same run then delivered exactly **500** rows — the backend's `max_results`, which is a *backend credential shared by every source on that warehouse* and so cannot be widened for one of them. Bounding the document's own date narrowed 3509 → 1184, still over. `DISTINCT` collapses nothing when a version/envelope number is deliberately projected. And `UseCaseAnalyzer.build_scope_discovery` was **not** passed the engine's `row_caps` — the truncation guard in `evaluate_verdict` covers conditions and does not reach the sweep — so a truncated sweep reported as `ran, N assets` and only the report's evidence-limits section said `row-limited at 500 rows … a lower bound`. It takes them now, on the same `len(rows) >= cap` test the verdict uses, so one source cannot read truncated in one surface and exhaustive in the next; the counts become floors and the status carries the marker the action backbone matches on.

## Pack-declared query guarantees (`src/retrievers/query_guards.py`)

The OR-ing rule above is the right *default* and the wrong answer for some sources, and the generation prompt cannot be trusted to make the exception — **a prompt instruction cannot be relied on to beat another prompt instruction**. Five `SourceDef` fields therefore state the exception in the pack and the engine ENFORCES it after generation (`require_all_entities`, `identity_keys`, `never_filter`, `partition_columns`, `epoch_time_columns`), and a **sixth** guard enforces a contradiction *between* two declarations rather than a declaration of its own (fabricated literals, below). (Two of the five follow the same pattern for a different reason — the prompt cannot state the fact at all without a literal that goes stale: see partition pruning and epoch time windows below.) Both describe the **source** (its key shape; which of its fields are evidence), not the backend that stores it, so `_attach_query_guards` copies them onto every retriever config regardless of kind — a new backend branch cannot silently drop them, which is exactly what happened while they were Databricks-only (declaring `never_filter` on an Elasticsearch source was a no-op).

- **`require_all_entities: [<entity types>]`** — this source is a lookup keyed by that TUPLE and no member is selective alone, so its fields must be **AND-ed**. `render_filters` states the requirement in the prompt naming the resolved fields; `enforce_conjunction` (SQL/ES|QL text, `=` and `==`) and `enforce_conjunction_dsl` (promotes a `bool.should` over exactly the required fields to `bool.must`) rewrite the OR afterwards. Measured motivation: on one reference-lookup source the alerted key *pair* returns 1 row, while one component of it alone matches **89,937 org units** — an OR asks for ~90k rows, truncates at the cap, and answers about *other* keys.
- **`never_filter: [<dotted field paths>]`** — fields that are EVIDENCE: the query may **RETURN** them, never **FILTER** on them. Some read exactly like noise reducers while being the answer (`AND <flag path> = false` looks like dropping service accounts, and deletes exactly the rows that decide whether the actor *was* an automation). `guard_prompt_line` derives the prohibition sentence from the pack field, so declaring `never_filter` is self-sufficient — no source hand-writes the prose into `query_hints`, where it would drift. `strip_evidence_predicates` then removes the predicate from generated SQL/ES|QL (including a whole `| WHERE` pipe stage), and `strip_evidence_filters_dsl` prunes the matching leaf clauses from a Query DSL tree.

Both rewrites are deliberately **conservative**: only shapes whose meaning is unambiguous are touched. A predicate inside an OR-group, or a `should` list carrying clauses beyond the required key fields, is **left alone and logged loudly** — corrupting someone's boolean tree is worse than an over-filtered read. Matching is on the field's **last path segment**, since a generated query may reference it bare, dotted, backticked, or via a table alias. ServiceNow's encoded query is built in code (`^` is already AND, so no conjunction rewrite applies) but still drops a `never_filter` field rather than making it a clause.

### Fabricated literals — the sixth guard, and the only one whose second input is the INCIDENT

The five above all repair the **shape** of a predicate the generator was right to write. This one removes a predicate that should not exist: **a column bound to entity type A carrying a literal that is a value of entity type B.** It is not a new pack field — both inputs are already declared, and it is enforced where all the others are (`strip_fabricated_predicates` for SQL/ES|QL text, `strip_fabricated_filters_dsl` for Query DSL, wired into all four query-generating retrievers immediately after the evidence strip and **before** the bounds).

**Why none of the others can see it.** `render_filters` decides which value goes on which column and gets it right — a value whose form the source binds nothing for is dropped there. But the filter hints are *one* input to a prompt, and the prompt's others (`render_identifiers`, the source's `query_hints`, the request prose) carry the incident's values **UNTYPED, by construction**: the alert's own labels for its identifiers are unreliable, so the generator is deliberately told to infer each value's column from the schema. That inference is the hole. Handed a value it has no column for, a generator following a mandatory-predicate instruction does not abstain — it writes the closest-looking column. The result is a **real** literal in a valid predicate that can never match, which is why `base.unresolved_placeholders` (which looks for a quoted `'<...>'`) cannot catch it: same silence, arriving in a shape that check cannot see.

**Measured on job `0184a3ce`, two halves of one defect.** A locator-keyed source was handed the actor's login on the locator's column (`WHERE <locator> = '<the actor login>'`) — on an incident that extracted **no** entity of the locator's type at all, so no such value existed anywhere in the run. And that source's sibling equated the **alert instant** to a record-creation timestamp. Both returned 0 rows, both reported success, and 0 rows is indistinguishable downstream from "the source had nothing to say".

The two halves have different root causes and are fixed in different places:

- The **timestamp** half was traced to `render_filters` itself, which paired the window's value with its column as an *equality*. Fixed at source: it now emits a RANGE instruction naming the column (`<col> is the event-time column for the window A..B — bound it as a RANGE, never an equality against a single instant`). Not dropped outright, because the hint carries a second fact worth keeping — *which* of several date columns is the event time.
- The **cross-entity literal** half cannot be fixed in the hint channel at all (the other three channels are untyped by design), so it is the post-generation guard.

**The test is declarative on both sides**, hence domain-free: take the predicate's column, ask which entity types the SOURCE binds to it (`entity_bindings`), ask which type the LITERAL is a value of in THIS incident (the incident's typed entities — never the planner's guessed `scope_id`/`actor_id` scalars), and drop only when those two sets are disjoint and both non-empty.

**And the second input has to be the UNFILTERED entity set, which is the same defect this guard is about, one layer down.** `incident_values` read `RetrievalQuery.entities`, and that list has already been narrowed by `_entities_for` to the entity types **this source can bind** — correct for its three readers (the gate reviewer, the generation prompt, the health scorer) and precisely inverted here, because the contradiction being tested is *about* a type the source binds no column for. So the one value that could produce a verdict was deleted before the comparison and the guard failed open on its own headline case, silently: a guard that declines reads exactly like a guard with nothing to do. Measured on job `ca4240c0` on **two** sources at once — `WHERE OFFICE_NAME = '<record locator>'` against a source binding only `org_unit`, and `term(actor.sign = "<the same locator>")` against one binding `actor` — both returned unchanged, one of them the source whose `zero_rows.meaning` then told the operator the office carried neither attribute. The incident's whole typed set therefore rides beside the scoped one on `RetrievalQuery._incident_entities`:

- **A `PrivateAttr`, not a field.** This model is also the tool schema handed to the LLM, so a real field would ask the planner to fill the one guard input that *deletes* predicates.
- **Set on both seams that build a query** — `_enrich_queries` (planned and manual) and `_follow_up_query`, which constructs its own `RetrievalQuery` and does not pass through the first. That second one is not padding: a follow-up target is by definition a source **deferred out of pass 1**, so the measured case now runs on exactly that path.
- **Unioned with `entities`, never preferred over it.** A follow-up pass's harvested values ride on `entities` alone and are deliberately absent from the incident's entities, so preferring the incident set would make the guard blind on pass 2 to a literal it could see on pass 1. The union is monotone — strictly more visible, never differently typed, since `_entities_for` filters and does not retype — and `_types_by_value` keeps both types of a value carried twice, so a predicate survives if **any** of its value's types is bound to the column.
- **Absent, it reduces to `entities`.** A query rebuilt from an export does not carry private state, and it must then guard exactly as it did before this existed rather than differently.

- Matching is on the **path, not the leaf** — the opposite of the other guards, and the measurement is why: the fabricated predicate was written on a field one level *inside* the bound struct, so a leaf test asks about a segment the pack never mentioned and fails open on the very shape the guard exists for.
- An **enclosing** binding is consulted only about a leaf the pack names nowhere else. A struct is bound by what it *identifies*, and a struct declared as the actor still carries fields of other types (that actor's own org unit) — a subtree-wide claim would drop correct predicates.
- **Fails open** on an unbound column, on a literal the incident never carried (a status/flag/code read out of the schema), and on any predicate in a shape not safely rewritable (left alone, logged). There is one further escape it is tempting to add and must **not** have: keeping the predicate when the source binds the value's type on no column at all. That inverts the guard on the source that needs it most — the measured case is exactly it, a two-binding source handed an actor login, and the narrower a source's declaration the more freely a generator invents against it.
- **The window is excluded** from the incident's values. It is a bound, every source binds a column to it, and a guard that removed a time predicate could delete the query's only partition bound — which is the slow-scan failure `enforce_partition_bounds` exists to prevent. Two guards pulling in opposite directions on one clause is worse than either defect. In the DSL half the same reasoning excludes `range` (and `exists`) from the clause kinds examined.
- In the DSL, removal is safe in **every** occurrence position, which is not the usual argument: a fabricated literal matches no document, so the clause contributes nothing to a `should` and nothing to a `must_not`, and removing it from a `must`/`filter` widens the result to what the query would have returned had the generator not invented it. An emptied occurrence list is deleted rather than left as `[]`, and a clause reached outside any `bool` degrades to `match_all` so the query stays valid.

The REST retriever is deliberately **not** wired: it builds its clauses in code from `field_map`, so there is no generated query to repair.

### `identity_keys` — an actor is named by a COMBINATION, and several combinations work

`require_all_entities` is **one fixed tuple**, which is right for a lookup table whose key never varies and wrong for an *actor*: a real deployment identifies a user by org-unit+identifier, org-unit+userId **or** organization+userId, and which of those the incident carries varies run to run. The fixed tuple then degrades in the worst possible way — with one tuple declared and an incident carrying only the members of another, `required_fields` resolves **one** field, `enforce_conjunction` needs ≥2 to act, so the generated OR survives untouched and the query answers about other actors *while the pack declaration reads like a guarantee*. A silent no-op is the exact failure class these guards exist to prevent.

So `SourceDef.identity_keys` is a **priority-ordered list of candidate keys** (`[[org_unit, user], [organization, user]]`) and `resolve_identity_key` takes the first candidate whose **every** member the incident carries and the source binds. A candidate is satisfied wholly or skipped — falling through to the next COMPLETE candidate is the correct degradation; half-applying one is the bug. `conjunction_fields` is the single resolver behind **both** the prompt hint (`render_filters`) and the post-generation rewrite, because a hint naming one tuple while the guard enforces another is indistinguishable from no guard; it falls back to `require_all_entities` and logs a warning when nothing resolves, so "the query is not scoped to one actor" appears in the log rather than in the verdict. Like the other guarantees it rides `_attach_query_guards` onto every retriever config.

**A COMPLETE one-column key is not a PARTIAL two-column one, and conflating them made a whole shape of declaration inert.** The paragraph above is about a key whose members were declared and only *some* resolved; `resolve_identity_key` implemented it as "skip anything under two members", which also refused a candidate that declares **one** member and resolves it — a lookup on a table whose key *is* one column, which exists in every domain. Refusing it bought nothing: it could not have been enforced either way (`enforce_conjunction` needs two fields, correctly — there is no AND to write), so the only casualty was the claim `key_was_enforced` makes about an **empty** result, which is sound at one field and says so in its own docstring (any subset of a conjunction returning zero rows proves the conjunction is empty too). The same key spelled `require_all_entities: [<one type>]` has always resolved one field and made that claim, so **the arity was never the rule — the asymmetry between the two declarations was the defect**. Two consequences of the fix: candidates resolve in **two tiers** (every conjunction first in declaration order, then the one-column candidates), so a fallback candidate can never preempt a real pair whatever order it is written in and a pack cannot weaken its own source by adding one; and a conjunction from **either** declaration outranks a one-column key, because two AND-ed columns bound the query to one actor while one column only licenses reading an emptiness. The log distinguishes them (`… actor identified by A + B; these are AND-ed` vs `… actor key is the single column A. Nothing is AND-ed …`), or a source enforcing no conjunction reads exactly like a satisfied pair.

**And when nothing resolves, WHICH of the three causes it was decides who can fix it.** One sentence served all three — "no declared `identity_keys` candidate is fully satisfied by this incident's entities" — which is true of only one of them, and sends the reader to the understanding stage to look for an entity that was never the problem. `identity_key_diagnosis` names them apart: a **declaration** no incident can satisfy (a candidate that names no usable entity type — a catalog defect), a **binding** this source does not have for a member the incident *did* carry (the stale-binding failure), or an **incident** that genuinely carries part of every candidate (not a defect at all). Prose only — nothing branches on it, so a wrong guess costs a reader one line and never a predicate. Two related hardenings: a member that is not a string is dropped in `_entity_types` rather than at the field lookup, because `{}.get(["a", "b"])` raises `TypeError` and a declaration nested one bracket too deep took the whole retrieval down from inside a guard; and `pack_validate` reports the shapes that are decidable from the catalog text — a candidate that is not a list, a member that cannot name a type (**errors**: the pack states a key and none is ever enforced), a candidate repeating one type, and a key member the source declares no `entity_bindings` for (**warnings**: it still resolves, or the mapper may still bind the type off the discovered schema). Measured zero across every installed pack and the template.

**The order is per-source and MEASURED, never an engine-wide constant** — that is the whole reason it lives in the pack. Two sources in one real pack invert each other. On an access-audit view (513,444,742 rows on one day's partition) the org-unit and actor-identifier columns are null/empty on **0** rows, while the userId column is on 16.1% and the organization column on 15.8% → (org_unit, identifier) is candidate #1. On an authentication event table (95,422,170 rows on one day) the ranking **inverts**: userId null on 0.3%, identifier on 1.5%, login on **32.0%**. And a bare identifier is genuinely ambiguous: of 173,978 distinct logins in one day, 24,683 (14.2%) appear in more than one org unit. Selectivity on the first source over a 3-day window: the identifier alone → **316,398,310 rows across 11,424 org units**; adding the org unit → **998**. An **event log** declares neither key — but the reason is **not** that it "must OR its identities", which is how this sentence used to end and is wrong in a way that returns a FULL result: what an event log may OR is one entity's alternative **columns** for the same value (two spellings of one login — that is `identity_synonyms`, and here the 32.0% null rate above is exactly why no fixed order over them holds), and every SEPARATE dimension beside it — the org unit, the agent sign, the organization — is `identity_scopes` and therefore **AND-ed on**. The rule is *OR within one entity, AND between different entities*; a lookup differs only in that its tuple is FIXED, which is what `require_all_entities`/`identity_keys` express and an event log cannot.

### One entity TYPE may sit on SEVERAL columns, and grouping per field made the conjunction OPTIONAL

`enforce_conjunction_dsl` promotes a generated `should` (OR) over the resolved key's columns into a `must` (AND), and it grouped the arms **per field**. Its own docstring stated that as a convenience — the rule *AND between entity types, OR within one type's values and forms* could be honoured "without needing to know the types" — and that holds only while a type has one column. It usually does not, by two independent routes: `value_forms` routing binds one type's agent-sign and login forms to two columns, and a source may bind one type **flat** to three columns for robustness (the same fact recorded under three paths, none reliably filled). Either way the extra arms sit on leaves the resolved key does not name.

So they became **spare** arms. Spare arms are kept OR-ed beside the promoted conjunction (correctly — a key says how to identify the subject, not that nothing else may be asked), which means the whole conjunction was demoted to one **alternative** under `minimum_should_match: 1`, and each spare arm was bounded by nothing at all:

```
should: [ {must: [unit=UNIT01, sign=SIGN01]},      <- the key, now optional
          {userId=LOGIN01} ]                          <- every unit, unbounded
minimum_should_match: 1
```

The result comes back **FULL**, which is the failure no "did anything come back" check can see, and the row cap then truncates on other parties' rows. It is the same defect class as the ninth guard's measured 17,863 identities, in the opposite direction and through a different seam. **Measured on a live alert-store query** whose key resolved to a subject reference plus two identity types: **11 `should` arms, 7 of them spare**, and the source returned its 500-row cap truncated.

The fix is to group per entity **TYPE**, and "type" needs **two attribution routes**, because neither sees the other's case:

* **The pack's declaration** — `field_mapping.form_split_bindings`, the *same* one `relax_form_conjunction_dsl` reads, so the two cannot come to disagree about which columns are forms of one type. It survives a literal no lookup recognises, and it is the only route where the type is known without the value.
* **The arms' own LITERALS** — `_arm_entity_type` over `_literal_value_types` (end-only wildcard unwrapping, so a widened `LIKE` still attributes). This is the only route to a type bound **flat** to several columns, and to a column the pack never declared at all.

On that live query the declaration route **alone** folds 4 of the 7 spare arms and leaves 3 — the flat-bound type's two extra columns and one undeclared sibling — and **3 unbounded arms are as FULL a result as 7**. Both routes together fold all 7, leaving 0 spares and exactly the shape the rule asks for: `subject AND (unit OR unit' OR unit'') AND (sign OR sign' OR login OR login')`.

The two groupings are **unioned**, and every leaf **two types both claim is dropped** from the grouping — that shape is the *eighth* guard's (`enforce_conjunction_same_column`), where the arms must be AND-ed, so OR-ing them would be the opposite of the right answer. Arms on a folded type's columns are then OR-ed **inside** that member's conjunct (`{must: [unit, {should: [sign, userId], minimum_should_match: 1}]}`). Folding also makes the key reachable in a second shape: where the generator constrained the type on a **sibling column alone**, per field the key read as partial, so the whole rewrite declined and the full OR shipped.

Grouping is resolved **per `should` group**, not once per query, because half the attribution reads that group's own literals.

Five refusals keep it conservative:

* **A type the key does not name keeps the spare-arm reading.** The fold is not a licence to treat any declared column as part of the key.
* **A group in which the key names TWO of one type's columns is left per field.** Folding them together would delete a conjunction the pack asked for, so the declaration wins. (Only reachable at three or more columns, which is what makes it a real bound rather than a shape the field map cannot produce.)
* **An arm whose literal is of no known type is not attributed** — an environment constant stays spare rather than being annexed to whichever member it sits beside.
* **A mixed `terms` list is not attributed.** `_arm_entity_type` requires *every* literal in the arm to be a value of the same one type; two types in one list yield no attribution.
* **One value that two types share attributes to neither**, since the intersection is not a single type.

Plus one bookkeeping property: **a fold is counted and logged apart from several values of ONE column.** The headline log line is the same either way, so that clause is the only trace of which of the two readings ran — and the two readings of one input ("an arm bounded by nothing" vs "a sibling column OR-ed inside the member's conjunct") are otherwise indistinguishable in the record.

A pack declaring no split forms and an incident whose values attribute to nothing produce empty groupings, and the guard is byte-for-byte what it was. `key_was_enforced` is unchanged and stays per FIELD, so a member honoured only through its sibling column reads as *not* enforced — the conservative direction: an empty result is then reported as "the source said nothing" rather than claimed as a decisive absence.

### `enforce_key_presence` — the key member the query never MENTIONED, and why an `OR`-rewriter cannot see it

Everything above is about a key the generator *did* express and expressed wrongly. The remaining hole is the one shape no rewriter can reach: a query that constrains a resolved member on **no column at all**. `enforce_conjunction*` turns an `OR` it can find into an `AND`; handed a query with nothing to convert it returns the text untouched, emits no log line, and both `require_all_entities` and `identity_keys` are a silent no-op. Nothing else notices either — nothing is fabricated (so the sixth declines), nothing is vacuous (so the seventh declines), and `enforce_subject_anchor`, the only other additive guard, is gated on `identity_scopes`/`identity_synonyms`, which a keyed lookup deliberately does not declare.

**Measured over the local job history:** of **57** real queries targeting a source that declares `identity_keys`, **52** constrain every resolved member, **4** constrain one and omit the other, and **1** constrains none. All five are the same source — a composite-key reference register — over two different incidents, asked `WHERE <org unit> = '<value>'` with the actor's own identifier absent. On a ~940k-row register that is a question about a *population*, truncated at the row cap; and because a reference lookup is read for MEMBERSHIP, the omission does not degrade the check, it **decides** it on the wrong question. One live run then printed `[UNKNOWN] … : no rows` for an absence the pack had declared to be the finding, because `key_was_enforced` is per FIELD and one omitted member makes it `False` (§ below).

So `enforce_key_presence` / `_dsl` **adds** the missing member. Five narrowings, four of them the subject anchor's:

* **Per entity TYPE, never per column.** `key_values` is `{type: {column: [value, …]}}`; a type's columns are tested as one group (`_is_constrained_conjunctively` over all of them) and `OR`-ed when injected — the eighth and ninth guards' shared rule, *AND between entity types, OR within one type's values and forms*. Per column, a type whose two forms sit on two columns would be AND-ed into the claim that both spellings appear on one row, which is the ninth guard's measured 17,863-identity failure arriving through this seam.
* **Already constrained → untouched**, on the strong conjunctive test and not the loose one: a comparison `OR`-ed beside a wider arm binds no row of its own, so "there is a comparison somewhere" is not the question. A member the generator bound *tighter* than this guard would is left exactly as written.
* **Only what the caller resolved a VALUE for.** A member the incident does not carry, or one whose value form this source binds no column for, contributes nothing. Inventing a literal is the one direction forbidden here.
* **All-or-nothing per member, and members are AND-ed**, because a key is a conjunction by definition — that is precisely what makes an empty result decisive.
* **The columns come from the pack.** `field_mapping.key_presence_values` is a per-TYPE reading of the same `values_by_type_and_field` the generator's filter hints flatten, so the hint and the injection cannot name different columns, and it routes each value through the same `value_forms` router every other value takes. Two types resolving onto ONE column are **refused** rather than injected — there is no conjunction to write between a field and itself — and the refusal is per column, so a sibling member on its own column still gets injected.

It runs **after `enforce_subject_anchor` and before `widen_stem_literals`** in all four retrievers, asserted in the wiring tests: the anchor is the other additive guard and must not read a clause this one just wrote, and a declared stem must be offered beside the injected literal too, since which precision a column stores is declared nowhere per column.

**A declaration is only safe on a source read as a LOOKUP about one identity**, and that is a pack judgement this guard makes consequential. Because the rewrite is additive, `identity_keys` stops being advice the generator may decline: whatever a candidate names becomes a predicate. On a **scope sweep** — a source whose whole purpose is to return records the alert never named — the same declaration deletes exactly the other parties' rows it was asked for. On a source binding one entity across several **roles** (creator / owner / last-updator), the mapper resolves one column per type and per form, so a key can pair the actor on one role with the org unit on another: a valid predicate no row satisfies. And where a target-side column is an explode alias rather than a filterable path, injecting the actor-side column answers "what did this identity do" under the heading "what was done to it". None of those three is decidable from the engine, so a pack that declines to declare a key should record the reason where the binding is (one installed pack pins its eleven exceptions by name, with the reason, in its own test file).

### `enforce_value_tuples` — a combination is a fact ACROSS columns, and per-type lists lose it

Every guard above reasons per entity type: which column a type sits on, whether it is constrained, whether it was mentioned at all. That is one level too coarse for an incident that names N *records* — one org unit, one operator code and one login per row, repeated. Per-type lists AND-ed together ask for their **CROSS PRODUCT**: 25 real identities go out as 800 pairings, 775 of which belong to other parties. The result comes back **FULL**, so no "did anything come back" check sees it, every literal in it is real, every column is bound, and the row cap then truncates on rows nobody asked about.

So a combination is rewritten as an **OR of AND-groups**, one group per record — this page's shape rule read one level up: OR *between* records, AND *within* one. **Two sources of combinations, deliberately indistinguishable downstream:** values harvested together from retrieved rows (a `follow_up_passes` entry's `together:`, via `follow_up.harvest_value_tuples`) and the incident text's own per-record layout (`incident_understanding._group_co_occurring_entities` stamps `co_occurrence` from the text's LINES; `ApiCallGenerator._incident_value_tuples` reads it). One `RetrievalQuery._value_tuples` field, one resolver, one guard, because a second shape would be a second enforcement path to keep honest.

**A value belongs to every record that named it**, which is why the stamp is a whitespace-joined SET of labels and not one label: an entity list holds one entity per distinct *value*, a table repeats a column by construction, and a single-valued stamp keeps only the LAST line that named a shared org unit — deleting it from every earlier record, whose remainder is then two FORMS of one type, i.e. fewer than two distinct types, i.e. discarded outright. Measured on one tabulated incident (27 records, 16 org units, 6 of them on more than one record): 27 records became **17** usable combinations and **18 of 66** asked-for literals sat in no surviving combination — at which point the guard, which proves it harvested every literal the body constrains before it narrows anything, declined the whole rewrite and the query went out as the cross product. The set stamp takes the same incident to 27 combinations and 0 unharvested literals. A single-record value still stamps exactly one label.

**Where it runs is the fix, not a comment.** After **both** additive guards (`enforce_subject_anchor`, `enforce_key_presence`) and before **both** widenings, asserted in the wiring tests on all four routes. What the additive pair splices is `WHERE <clause> AND (<old body>)`, and that clause is per-TYPE value **LISTS** — the very cross product this guard exists to narrow — so run first it read a body whose components were still buried inside an OR-group, declined, and the injection then published the cross product unnarrowably: measured on one live 2-arm statement, **800 pairings for 25 real identities**; run in this position the same statement narrows to those 25. A widening run first turns a component's equality into a `LIKE` group, which is not a value list it can read; and it stays after the identity rewrite, which reversed would read the OR-of-ANDs as a flat identity group and flatten it back into `unit = 'U1' AND unit = 'U2'` — zero rows.

**Heterogeneous shapes choose a FAMILY.** One incident yields combinations of several shapes at once (25 three-slot records beside 2 prose two-slot facts), so requiring every slot skips every combination: the largest family wins (then slot count, then slot order) and slots outside it keep the conjuncts they already had.

**And a twin is only a twin while both sides read the same input.** That family choice was written on the textual side alone, so the DSL twin still required every slot — the same silent no-op this whole page is about, one level in, and wearing the decline that reads like a *finding about the pass* ("none of the harvested combinations is fully within the values it asked for") rather than like a guard that cannot read its own input. **Measured on job e3ca805c:** one DSL group constrained the record slot — named only by the 2 prose combinations, its sibling organisation column already deleted upstream by `never_filter` — beside the unit and identity slots of the 25 tabulated ones. Every combination missed a slot, 0 arms, declined, and 400 pairings for 25 real identities went out over the Query DSL **in the same run** in which the textual twin narrowed a sibling source from 800 to 25. The port is the same code: the family is chosen per `must`/`filter` list, only that shape's clauses are replaced, and the proof runs against the chosen family's own harvest and not the global one (a value harvested only by a *discarded* combination appears in no arm, so the global harvest would pass the proof and the rewrite would delete that value from the query).

### A SET OPERATION is not one statement, and four guards used to read it as one

`UNION`/`EXCEPT`/`INTERSECT` is the shape a generator writes whenever a source spans two views, and it is where a filter-region reader fails silently in the one direction nothing checks. `_filter_region` returns everything after the FIRST `WHERE`, so a depth-zero `AND` split cuts across the arm boundary and hands back a fragment holding arm 1's group *plus* arm 2's `SELECT … WHERE` prefix — which decomposes no further, falls through to the LOOSE test, and the loose test says yes. Measured on a live 2-arm `UNION ALL` over two views of one relation: no member of the declared key was conjunctively constrained anywhere in it, `_is_constrained_conjunctively` returned a false `True` in silence, and `enforce_key_presence` and `enforce_subject_anchor` both declined. An arm the guards skip contributes exactly the rows they exist to exclude, and the statement still returns a **full** result.

So: `_is_constrained_conjunctively` reads **per arm with every arm required** (the question is about every row the query can return, and a union returns the union of its arms'); the two additive guards splice their clause onto **every** arm through one `_and_clause_onto_arms` helper (shared with `enforce_default_filters`, which already had to); and `_filter_bodies` yields one span per arm for the tuple rewrite. A **parenthesised** arm or one holding a subquery declines the whole statement with a warning rather than half-doing it — a refusal costs the slice, a half-rewrite costs the answer. And both of the tuple rewrite's refusals now **log**: *a guard that declines is indistinguishable from a guard with nothing to do*, and these two were silent.

### `key_was_enforced` — the flag that says what ZERO rows means, and which direction it is sound in

`publish_query` records, per source, whether the query that actually ran constrained every field of the resolved key (`last_key_enforced` → `keyed_sources` → `evaluate_verdict`). It exists so an empty result from a reference lookup can be read as **the finding** — "this identity is not on the list" — instead of as missing data. Always from the published text and never from the declaration: four things can stop a declared key reaching the backend, and each would otherwise turn a real UNKNOWN into a fabricated PASS.

**Its claim used to be the wrong one, and any `OR` disqualified.** "Every row returned is about this identity" is what a *non-empty* result needs; this flag is only ever consulted for an **empty** one, where the entailment runs the other way. `sign = 'X' OR sign LIKE 'X%'` returning nothing proves `sign = 'X'` returns nothing, because the query asked for a **superset** — so a disjunct that *widens one key field* is sound, and sound in the hard direction. Rejecting it put the engine in direct contradiction with a pack: the source's own `query_hints` instruct exactly that widening (a concatenated duty code must still match), and the reading of the result then depended on whether the generator took the advice. Measured: **one incident, two runs, `PASS` then `UNKNOWN`** on the check that establishes what kind of party acted, with nothing changed but the wording of the predicate. Now a group is collapsed to the key equality it widens when — and only when — every arm constrains the **same** key field and one arm is that field's recognised equality; a group spanning two fields, naming a non-key column, or nested past what the reader can parse still disqualifies. Verified as a **strict relaxation** by replaying the local job history: of 259 distinct generated queries, 23 target a source that declares `identity_keys`, and the change moves **zero `True → False`** and exactly one `False → True` — the query that regressed. Both query families, since a seam wired for one makes the declaration a silent no-op on the other. **The field set has to come from the pack, per source**, or the measurement is not one: an earlier pass applied a guessed list of candidate key tuples to every query in the corpus, including sources declaring no key at all, and produced counts that do not reproduce.

**And the shape that really does void the proof is still accepted:** an AND-ed conjunct on a **non-key** column can be the reason the result is empty. It is on **10 of the 23** live `True` queries, almost all mandatory partition or window bounds — so tightening it would strip the flag from a large share of the sources that need it, and belongs in its own evidence-led change rather than a drive-by. Worth knowing before trusting a `True`.

### `identity_synonyms` / `identity_scopes` — the THIRD identity shape, and it comes in FAMILIES

`identity_keys` ANDs a composite key (right for a keyed lookup: an OR there returns other keys' rows) and declaring nothing leaves a flat OR-group (right only when every disjunct names the same actor). An **event log** is neither, because its identity columns are not all the same kind of fact:

- **`identity_synonyms`** — columns that may hold **one** value, stored under whichever the producer populated. They must be OR-ed or the row is missed: on one auth-event table (95,422,170 rows, one date) the login column is null on 32.0% and the userId column on 0.3%, so ANDing them drops a third of the population.
- **`identity_scopes`** — **separate** identity facts carried by the same document (the unit it belongs to, the role it acted under). OR-ing these does not loosen the filter, it changes the question to "anyone sharing either". Measured on job 30887c6b: the flat group `login OR userId OR sign OR office` returned exactly **500 rows — the cap** — for ONE account over three days, i.e. the unit's whole population crowding out the subject.

Enforced shape: `(synonym OR synonym) AND scope AND scope`, rewritten post-generation by `enforce_identity_scope` — because `query_hints` **already** asked for "SELECTIVE IDENTITY fields ONLY" and the generator emitted the flat OR anyway.

**One family is not the general case, so `identity_synonyms` also accepts a list of lists: OR inside each, AND between.** A source may store two *independent* identifiers each spread over several columns — an actor id under one column per document shape, and the correlation id of the episode under another per shape. Flattened into one group they ask "this actor's rows OR this episode's rows", and the second arm is unbounded by the first. Measured on job 0184a3ce, 15 flat clauses under `minimum_should_match: 1` over 30 days: **500 rows (the cap) holding 100 distinct episodes of 33 distinct actors, of which 17 were the subject's** — the whole-organisation arms matched all 500, so the episode id the alert names constrained nothing. Enforcing the declared families took the same data to **17 rows / 1 episode**, well inside the **unchanged** cap: the cap was never the limit.

Three rules the seam holds, each because the failure is silent:

- **A family must be COMPLETE to be enforced** (`_group_by_family`, shared by both routes). On that source a three-shape unit family was generated for 2 of the 3; enforced as written it cut the correct 17 rows to 12 and deleted every row of the third shape — the five carrying the per-action detail the checks read. An incomplete family's clauses are **discarded** from the rewrite with a WARNING: each names a fact about the subject, so a row matching only through the dropped disjunct matched on that fact alone (a different actor in the same unit). Leaving it OR-ed is the original bug; declining the rewrite entirely leaves the flat group that returned the cap.
- **Which key a column goes in is not taste, because scopes AND against each other.** Per-shape spellings of ONE fact are a family *even when that fact is an organisational unit*: declared as three scopes, the three organisation columns became `orgA AND orgB AND orgC`, and **no document carries more than one** (257 / 243 / 0 rows, and 0 of 500 carry all three) — a predicate nothing satisfies, which returns zero rows and reads exactly like a source with nothing to say. A scope is a second fact on the *same* document; a per-shape alternative spelling never is.
- **Both routes, or it is not a shape.** `enforce_identity_scope` existed for textual SQL only and was wired into the Databricks retriever alone, so for a source reached over a Query DSL gateway the declaration was a **silent no-op** — the very defect class this page exists for, and the one `_attach_query_guards`' own comment warns about. `enforce_identity_scope_dsl` is the twin; both share `synonym_families` (where a family boundary is), `resolve_identity_fields` (which columns are in it) and `_group_by_family` (when it counts as constrained), so the two cannot disagree.

Two conservatism properties, both asserted: **nothing is ever added** (only clauses the generator already wrote are moved; a family the query never constrained is not invented), and a group carrying **any** clause on a field the pack does not declare is left exactly as generated — the guard cannot know which family it belongs to. That last one is why the identity guard runs **after both strips** in all four retrievers: on job 0184a3ce two predicates putting the actor's own unit on the counterparty-unit columns matched 0 rows of 500 and their only other effect was to block the rewrite entirely. `resolve_identity_fields` tells an entity type from a column by whether the name is a key of the field map — *not* by whether it contains a dot, which reads a top-level column as an unresolvable type and drops it, emptying the declaration only on sources that have flat columns (invisible on a fully-nested backend; it discarded two real columns on the measured run).

### And WHICH of the three operations an organisation column gets is a MEASUREMENT, per relation

An organisation column is the one binding where all three mechanisms above are available and each has
a different failure, so choosing between them by reading the column's *name* is choosing at random:

- **OR-ed into the identity group** (the generator's default) it satisfies the group **alone**, so a
  one-identity question becomes "every event of this whole organisation" and comes back **FULL** —
  the failure no "did anything come back" check can see, with the row cap then truncating on other
  parties' rows. This is the shape the operator reported on three live jobs
  (`OR <actor org> = '1A'`, `OR payload.retrieverOrganization = 'SV'`).
- **AND-ed** — `identity_scopes` for an event log, `require_all_entities`/`identity_keys` for a keyed
  lookup — it deletes every row where the column is NULL or `''`, silently, in the direction that
  reads as a clean source.
- **Stripped** (`never_filter`) it can neither widen nor delete: a query that does not mention the
  column returns a superset of one that does.

So the deciding number is the **CONDITIONAL** absence rate — the share of rows *the actor arm could
itself have matched* where the organisation is NULL or `''` — and not the overall rate, because only
those rows can be lost by the AND. `populated_pct` is useless here for a reason this estate has
already paid for: it counts non-null and `''` is non-null (one live column was empty on 148,226 of
148,229 rows), so the two absence shapes are counted apart.

Measured 2026-08-20, one bounded aggregate per source, over 3-day windows on every source in one pack that binds an organisation column:

| source | rows | distinct orgs | conditional loss | declared as |
|---|---|---|---|---|
| `audit_profile_access` | 49,278,711 | 230 | **68.68%** | `never_filter` |
| `audit_raw_access` | 2,055,575,826 | 423 | **13.38%** | `never_filter` |
| `audit_record_security_checks` | 2,058,198,685 | 423 | **13.37%** | `never_filter` |
| `boarding_access` | 425,667,347 | 172 | 0.00% | `require_all_entities` |
| `boarding_list_access` | 34,931,228 | 165 | 0.00% | `require_all_entities` |
| `loyalty_profile_access` | 64,083,439 | 14 | 0.00% | `require_all_entities` |
| `order_access` | 17,474 | 2 | 0.00% | `require_all_entities` |
| `customer_entity_access` | 2,587 | 18 | 0.00% | `require_all_entities` |
| `auth_events` | 389,898,443 | 1,987 | 0.00% | `identity_scopes` |

**The nine split three ways, and nothing about the column's name, its estate or its meaning predicts
which answer a relation gets.** The first four were measured on the assumption that one family would
answer once; 13.38% and 0.00% came back from the same estate on the same day, which is why the
remaining five were measured rather than inferred from a sibling. (The explanation — three views
left-join the organisation, five trail relations have their producer write it inline — was found
*after* the numbers and is not a rule anyone could have applied beforehand.)

**Removing the member from `identity_keys` is mandatory on a stripped column, not tidiness**, and the
reason is the guard ORDER: `strip_evidence_filters` runs early and `enforce_key_presence` — the
additive guard, the one that ADDS the member a query never mentioned — runs late. A column that is
both `never_filter` and an `identity_keys` member is therefore **re-injected after its own strip**,
with nothing left to remove it, and `key_was_enforced` then returns `True` — licensing an empty page
as *this identity is not on the list*. The strip and the key removal are one change.

### `value_forms` — one entity type, several non-interchangeable surface forms

Upstream of the keys is a subtler version of the same problem: one entity *type* can carry two *forms* — a business **identifier** and an authentication **login** — stored in different columns, with the alert labelling both the same thing. The mapper returns one field per entity *type*, so `render_filters` used to union both values onto that one field, which is how a filter putting one form's value on the *other* form's column reached three live runs: a guaranteed-0-row equality, indistinguishable downstream from an empty source.

A prompt cannot fix what the data model discarded, so the form is **classified deterministically in the engine** from a pack-declared regex and stamped on `ExtractedEntity.value_form`. A source may then bind the entity **per form** (`user: {identifier: [<its own column>], login: [<its login columns>]}`), and `render_filters` is the enforcement seam: a form-bound value is routed to its own form's field, and a value whose form this source binds **nothing** for is **dropped with a log line** rather than filtered onto the sibling form's column. `_form_alias_hints` also narrows the mapper's prior to the forms actually present, since offering both forms' columns under one entity type is what let the mis-binding happen in the first place. Sources with no `value_forms` binding are unchanged.

## Partition pruning — the failure that looks like an empty source

**An unbounded partition column is not a correctness bug that looks wrong. It looks like slowness, and past the timeout it looks like "the source had nothing"** — 0 rows → decisive conditions `unknown` → INSUFFICIENT DATA. That is how a physical-layout detail becomes a *wrong verdict*, and it is why this is enforced rather than left to the prompt. Measured on one pack's biggest table (**84 TB, 112,697 files**): the same `count(*)` takes **185.7s** bounded only on the logical event date and **65.7s** once the *partition* column is bounded; the full ruleset projection went from **TIMEOUT (>960s)** to **717.4s**, and the actor-scoped scope sweep from TIMEOUT to **147.2s**.

**Discover, don't declare.** The physical layout comes from the backend's own metadata, because a hand-written hint cannot notice that a table was repartitioned: `DatabricksRetriever._discover_columns` already selects `partition_index` in the schema-discovery round-trip it was making anyway (non-null exactly on partition columns, its value being the position in the partition key), and `_partitions_from_columns` turns those rows into ordered specs. Snowflake reads `information_schema.tables.clustering_key`. A live sweep of every Databricks source in one real pack found all four outcomes the mechanism has to handle: the big event tables report one DATE partition column each — and it is often *not* the column the investigation is about; two reference tables are genuinely unpartitioned (single-digit file counts); and several sources are **VIEWs**, which report nothing at all.

**The pack declaration is the override for what metadata cannot say**, and only that: a VIEW hides its underlying table's layout, and no catalog anywhere carries `role` or `pad_days`. `SourceDef.partition_columns` is a list of `{name, role, type, pad_days}` — one live source declares `year`/`month`/`day` as zero-padded STRINGs because `DESCRIBE DETAIL` and `SHOW PARTITIONS` both fail on it with `EXPECT_TABLE_NOT_VIEW`. `merge_partition_specs` layers the declaration over the discovered specs (discovered wins on name), so a source can declare a layout without pinning one that the metadata already knows.

**One dialect-independent computation, four formatters.** `query_guards.partition_bounds(specs, date_from, date_to)` turns the incident window into per-column bounds: `role: date` (or an unrolled DATE/TIMESTAMP type) → a two-sided range; `role: year|month|day` → an **enumeration of the calendar parts the window spans**, which is why a window of 2023-12-31..2024-01-06 correctly yields `year IN ('2023','2024')` and not `year = '2024'`. `pad_days` (default **1**) widens both sides, because the partition column is rarely the column the incident is about — a record's later versions are written *after* the alert. Those bounds feed `enforce_partition_bounds` (SQL / ES|QL pipe stage), `enforce_partition_bounds_dsl` (ES Query DSL), `partition_clauses_encoded` (ServiceNow `^`-encoded), and `partition_prompt_line`.

**Asked AND enforced.** `partition_prompt_line` names the column in the generation prompt, and `_enforce_partition_bounds` rewrites the query afterwards regardless — the generator bounds the column the *request* talks about (creation/event time), not the one the table is physically *laid out* by, and those differ on exactly the table where it matters. The rewrite is as conservative as the other guards: a column already compared anywhere in the query is left alone; more than one `WHERE` (i.e. a subquery) is left alone and logged; an existing `WHERE` body is **parenthesised** before the bounds are AND-ed on, so a top-level `OR` cannot re-associate into `A OR (B AND bounds)`. Like `never_filter`/`require_all_entities`, `partition_columns` rides through `_attach_query_guards` — the single seam every retriever config passes — because a pack field honoured on only some backends is the same class of bug those guards were written to prevent.

Because this is discovered and injected, sources **do not hand-write partition prose into `query_hints`**; a well-authored hint tells the LLM only that the partition is bounded *for* it, and to write the *logical* bound the investigation needs — which on a versioned store is part of the record's identity rather than merely a scan optimisation, because locators get recycled.

## Epoch time windows — the same failure, caused by a stale literal

The incident's window travels the pipeline as `YYYY-MM-DD`, but some sources store their event time as an **epoch integer**. Somebody has to convert, and the generator is the wrong somebody: an epoch integer advertises neither its **unit** nor its **era**, so a wrong one is not wrong-*looking*. It is a plausible number, AND-ed into the filter, matching nothing — and "matching nothing" is the failure mode above: 0 rows → decisive conditions `unknown` → INSUFFICIENT DATA.

**Measured, not hypothetical.** One source returned **0 rows** on a live incident (job `ae050dbf`) whose document was sitting in the index. The generated Query DSL asked for `timestamp` between `1753574400000` and `1753660799999`; the document is at `1785170060721`. The delta is `31,536,000,000` ms — **exactly 365 days**. The cause was in the pack, not the model: `query_hints` carried a worked example, `1753315200000 = 2026-07-24T00:00:00Z`, and that value is **2025**-07-24. The generator copied the era faithfully. Every other source in the same run emitted a correct `2026-07-27` ISO window, which is what localises the defect to this one hand-written literal.

**Prose is structurally unable to fix this.** The only way to state a unit in a sentence is an example, and an epoch example is a literal that cannot be checked by reading it — it is precisely the class of fact the partition work already banned from `query_hints`. So `SourceDef.epoch_time_columns` declares `{name, unit}` and **nothing numeric**: `epoch_window` computes this incident's closed interval (midnight UTC on `date_from` → the last instant of `date_to`, so both end days are covered whole), `epoch_prompt_line` hands the generator the already-converted numbers, and the conversion is re-checked afterwards. Units accepted: seconds / milliseconds (default) / microseconds / nanoseconds; an unrecognised one is refused rather than guessed at.

**Enforced conservatively, in two shapes only.** `enforce_epoch_window` (SQL / ES|QL) and `enforce_epoch_window_dsl` (Query DSL) rewrite a bound when — and only when — it cannot be right:

- a **date/ISO literal** on an epoch column, which cannot compare against an integer at all, is always converted (`<=` takes the last instant of the day, so the end day stays included);
- an **integer interval that does not intersect** the incident's window — the query is asking about a different span of time than the incident. This is the year-off case.

An interval that *overlaps* the window is left exactly as generated even when it is narrower, because a generator that tightened to a few hours around the alert wrote a **better** query than the padded window would be. A rewritten DSL clause drops `format`/`time_zone`, which describe a date literal that is no longer there. Both are wired on every backend through `_attach_query_guards` for the usual reason — and here it is not theoretical: which retriever runs for that very source depends only on whether the cluster's creds carry `gateway: kibana`, so a guarantee honoured on one route and not the other is not a guarantee. The REST backend builds its query in code, so `epoch_clauses_encoded` simply appends the bounds.

### A literal in the wrong PRECISION (`value_forms[].stem`, `widen_stem_literals`)

**The sibling of the epoch repair above, one type over — and the only WIDENING rewrite in
`query_guards.py`.** That guard exists because a date literal on an integer column is a valid
predicate matching nothing; this one because the LONG form of an identifier is a valid predicate
matching nothing on a column storing its core. Same invisibility: 0 rows, reported as a success,
read downstream as "the source had nothing to say".

**MEASURED 2026-08-14** (one pack's reference register, the whole table with no window because its date column is `never_filter`): the key column holds the
6-character core on **946,805 of 946,805** non-empty rows and the 8-character operational form on
**zero**. The sources the incident's values are harvested from carry 8 characters on 100% of rows.
So a pass-2 lookup rendering the harvested value verbatim could never match — on any incident, for
any of the five rulesets declaring that source — and it returned **0 rows** where the cores match
11 of 11. The pack had already stated the fact (the trailing pair is optional in the form's own
`pattern`) and the reading side had always been prefix-tolerant (`correlation._identifiers_match`
accepts a stored value that is a prefix of the incident's, which is the only reason a check over a
reference table ever matched at all). **Only the predicate did not honour it**, and that asymmetry
is what made it invisible: nothing downstream would have mismatched had the rows arrived.

**And here the fabricated empty does not degrade the check, it INVERTS it.** That source declares
`zero_rows: {health_weight: 0.0, meaning: ...}` — empty *means* "this identity is not on the
register". So every identity read as absent from a list none of them had been compared against,
which on a categorical exclusion is a decisive PASS. Two graded cases were right by accident.

**Two differences from the epoch guard, both about what is KNOWABLE.** A unit conversion is
computable, so that guard **replaces** the literal. Which precision a column stores is declared
nowhere per column — the same pack faces one source at each precision — so this one **adds** the
core beside the value and lets the data decide. A core is a strict prefix, so on a long-form column
the extra literal selects nothing; it can only add rows the narrow predicate would have missed.

Because it widens, three properties keep it inside this module's never-additive contract:

- **The column and the literal both come from `field_mapping.stem_literals`**, resolved through the
  same `filter_values_by_field` the generator's own filter hint was built from. Nothing is added to
  a column the incident does not bind, and no literal appears that was not already in the
  predicate — only a second spelling of one that was, which the pack declares is the same identity.
  (The hint offers both spellings too, so a generator that follows it needs no rewrite at all.)
- **Two shapes only.** `col = 'v'` becomes `col IN ('v', '<core>')`; a literal `IN` list gains the
  core. A `LIKE`, a comparison against an expression, or a list this cannot parse is left exactly as
  generated and logged. On the Query DSL route (`widen_stem_literals_dsl`) `term` becomes `terms`
  and a `terms` list gains the core — both families, or it is not a guarantee.
- **Never a negation.** A widened `NOT IN` / `<>` / `must_not` NARROWS, and getting that backwards
  on an exclusion check is worse than the defect: it would delete the subject's own rows and report
  the emptiness as a clean. `sign NOT IN (...)` also happens not to match the naive pattern, so the
  `NOT` is matched on **both** sides of the column and declined explicitly — an accident is not a
  guarantee, and is not visible in the log.

It runs **after** both strips and after the subject anchor in all four retrievers (asserted): the
anchor may be the very predicate that needs widening, and a fabricated predicate is *dropped*
rather than repaired. A pack declaring no `stem` resolves to `{}` and every query is byte-identical,
which is every pack until one opts in — so `pack_validate` reports an unusable stem (invalid regex,
or not exactly one capture group) as an **error**: it is mechanically decidable, provably inert, and
an inert declaration reads as a pack that never opted in while the predicate it exists to repair
goes on matching nothing.

### A literal that is only PART of the stored value (`value_forms[].match`, `widen_match_patterns`)

**The stem guard's inverse, and the second widening rewrite.** A `stem` takes a shorter form OUT of
a long value; a `match` places a SHORT value as a fixed-width **window inside** the stored one. Same
premise, same invisibility, opposite direction — and the same reason neither can be a substitution:
**which precision a column stores is declared nowhere per column, so the data decides.**

**MEASURED live on the abusive-access incident of 2026-08-17.** An organisational unit is 9
characters (`AAASV0991`); the report named it by the 3-character segment a human reads (`DEL`,
`DAC`). Every route rendered that verbatim — `unitId = 'DEL'`, `{"term": {…: "DEL"}}` — against
columns holding the whole identifier, so every predicate was a **valid predicate matching nothing**:
0 rows, every stage green, and downstream indistinguishable from a source that had nothing to say.
Nothing else in `query_guards.py` can see it. The value is real and the incident carried it (so the
fabricated-predicate guard fails open, correctly), the column is bound (so no stale-binding warning),
the predicate carries a literal and is not vacuous, and `base.unresolved_placeholders` is about a
marker that is still `'<...>'`.

**The declaration is a form's, not a column's.** A `value_forms` entry may state a third fact beside
`pattern` and `stem`:

```yaml
value_forms:
  - name: full
    pattern: "^[A-Z]{3}[A-Z0-9]{6}$"          # the stored width: nothing to widen
  - name: location_code
    pattern: "^[A-Z]{3}$"
    match: "{value}??????"                     # three letters, then six of anything
  - name: corporate_code
    pattern: "^[A-Z0-9]{2,3}$"
    match: "???{value}????"                    # the segment sits INSIDE the identifier
```

`?` is exactly one character and `*` is any run, in the canonical spelling — `KnowledgePack.value_match_pattern`
substitutes `{value}` and the guard translates to the route's syntax (`LIKE 'DEL______'` on SQL, a
`wildcard` clause on the DSL, `LIKE "DEL??????"` on ES|QL). `pack_validate` errors on a template that
does not name `{value}`: it is mechanically decidable and provably inert. **A width is not a prefix.**
`DEL%` would also select a 12-character value on some other column, which is why the pattern is
positional and why the same mechanism can express a segment that is not at the front.

Three refusals keep it inside the never-additive contract, and the first two are the stem guard's:

- **Offered BESIDE the equality, never instead of it** — `unitId = 'DEL' OR unitId LIKE 'DEL______'`,
  and on the DSL route the `term`/`terms` clause is wrapped in a `bool.should` holding itself plus one
  `wildcard` per patterned value, with an explicit `minimum_should_match: 1` (the default is 1 only
  while the `bool` carries no `must`, and this clause may be spliced under one later). Widening is
  licensed because `key_was_enforced` is read **only for an EMPTY result** and its entailment runs
  through supersets: a widened predicate returning nothing proves the narrow one does.
- **Never a negation.** `NOT IN`, `NOT`, `<>`, `!=`, and `must_not` at any depth are all declined,
  because a widened exclusion NARROWS — on a check whose emptiness IS its finding that deletes the
  subject's own rows and reports the emptiness as a clean.
- **Never a value carrying the route's own wildcards.** `DE_` substituted into a `LIKE` template turns
  one character of the subject's own identifier into "any character", so such a value publishes as
  generated. The forbidden set follows the **dialect** (`%`/`_` on SQL, `?`/`*` on ES|QL and the DSL)
  rather than being one global list, since the same value is safe on one route and not the other.

The column and the literal both come from `field_mapping.match_patterns`, resolved through the same
`filter_values_by_field` the generator's filter hint was built from — so, exactly as with the stem, the
rewrite can only offer a second way of comparing a value the query already carried on a column the
hint already named. It runs **after `widen_stem_literals`** in all four retrievers (asserted): a stem
offers a shorter form that is itself a candidate for a positional window, and a pattern rewritten first
leaves a `LIKE` the stem guard does not read. A pack declaring no `match` resolves to `{}` and every
query is byte-identical — which is every pack until one opts in.

### A constant in the OR: `never_filter` and `default_filters` are OPPOSITES

Both keys are about one shape, and it is the one the OR-ing default above produces for a column that
is not evidence about anybody: the generator writes the **environment constant** into the same flat
`should` as every subject clause, where it satisfies `minimum_should_match: 1` on its own and the
subject scoping becomes decoration. Measured on the same run: `... AND ( record_locator = '…' OR
application_phase = 'PRD' OR retriever_office IN (…) ) AND …`. **The failure is a FULL result**, so no
"did anything come back" check can see it, and the row cap then truncates on the population's rows.

Which key repairs it is a **measurement**, never a family or a column name:

| the column holds | declare | why the other one is wrong |
|---|---|---|
| **one** real value | `never_filter` alone | a constant cannot narrow, so the predicate has exactly two outcomes — no-op, or **deletion** of every row that omits the field |
| **several** real values | `never_filter` **and** `default_filters` | the strip alone drops a real environment scope and mixes test traffic into a fraud verdict; the pin alone leaves the vacuous OR arm standing beside the pinned conjunct |

Measured 2026-08-17 on one estate, and the two answers sit one catalog apart: cardinality is **1** on
every ELK source binding the column (three of them only partially populated, so AND-ing it would have
deleted 25,511,308 documents in the name of a scope that excludes nothing), while the audit-trail views
next door hold **five** values with the production one at 99.82% — where dropping the predicate really
does widen the read. So the column NAME decides nothing: `application_phase` is not `phase`, and a
spelling that binds no row is its own defect, a declaration on it being a valid predicate matching
nothing. Both readings — the ELK sources and the SQL audit-trail views — separate `''` from `NULL`,
because a mandatory conjunct excludes a row with the field absent and only one of those is visible to
a `COUNT(col)` population figure.

Three rules in the enforcement:

- **Strip THEN pin, and the order IS the guarantee.** `enforce_default_filters` runs last, after the
  `never_filter` strip has lifted the constant out of wherever the generator put it, and AND-s it back
  on exactly once. Reversed, the strip deletes the conjunct the pin just wrote and the declaration is
  gone. The wiring test asserts the order on all four routes.
- **A UNION is not a wall here — it is every arm.** This is the opposite of `relax_form_conjunction`,
  the one guard that MOVES text: a pinned conjunct must hold of **every returned row**, so it is
  spliced onto each arm of a top-level `UNION`/`EXCEPT`/`INTERSECT` (depth- and quote-aware; a `UNION`
  inside a subquery or a string literal is not an arm boundary). Pinning only the first arm is valid
  SQL and half a guarantee — on the source this key exists for, which generates one `SELECT` per view
  of a shared relation and unions them, the slice would hold on one view while the others contributed
  their test rows to the same result. A **parenthesised** arm declines the whole rewrite with a warning:
  there is no depth-zero `WHERE` inside it to extend, and a refusal costs the slice while a broken
  statement costs the source.
- **The risk is asymmetric, and it decides who may declare what.** `never_filter` can never lose a row.
  A pin on an unmeasured column or spelling deletes every row or fails the statement — which is why an
  unreachable source (no credentials, so neither its vocabulary nor its identifier case has been read)
  gets the strip only, and the pin the day the account is filled in and the vocabulary is measured.

**The pin was ES-only for as long as the key existed.** Both SQL retrievers read
`config["default_filters"]` into an attribute and never applied it, so a pack pinning a shared relation
got the slice on an Elasticsearch source and the whole table on a SQL one — a full result of other
slices' rows, which passes every check that asks whether anything came back.

### An UNDATED incident: the window is DECLARED, never invented

A window is the **hard scope** of every query, so where it comes from is not a detail. Measured on one
live incident whose text carries no date anywhere: the understanding stage returned one — the start of
the run's own day until the moment it ran — and a parsed window is authoritative over every query's
date range, so that invented day became the scope of all **29** retrievals. Clearing it is only half:
asked to pick a window per source, the generator picked **eight different ones across 29 queries**, so
sources that have to be compared to each other were read over different spans.

Both halves are deterministic, and neither invents a number:

- `IncidentUnderstandingModule._drop_invented_event_window` clears `event_time` when the incident text
  contains nothing that could plausibly have produced a window. **The asymmetry decides the
  vocabulary**: reading a time expression that is not one leaves the generated window exactly as it is
  (nothing is lost), while reading NO expression where the text states one would overwrite a *correct,
  stated* window with a default. So a shape counts whenever it weakly could be a time reference — a
  bare four-digit year is a poor window and still evidence the text talks about when — and every field
  the prose can arrive under is searched, because input structure is not a contract.
- `ApiCallGenerator._default_window` resolves the depth **pack → config** (`retrieval.default_lookup_days`,
  else `log_sources.default_lookup_days`, both editable from the Configuration tab) and writes the same
  window onto every query in the plan, once per plan. **The engine holds no depth of its own**: how far
  back a store must be read to find the conduct behind an undated report is a fact about the domain's
  retention, its alerting lag and how long the behaviour typically runs, so a number in `src/` would be
  that judgement made in the one place that cannot know it — and it would apply to every domain. When
  neither declares, the generated windows stand exactly as they do today and a warning names both
  places, because the alternative reading of an absent declaration is a silent engine constant.

It is anchored on the incident's **ingestion timestamp** — the one time an undated incident does state —
and never on the moment the run starts, or re-running the same incident a week later reads a different
window and two runs disagree with nothing changed.

## Pack-driven source wiring

`LogRetrievalEngine._build_source_configs` / `_merge_endpoint`: retrievers are built from two merged places.

1. Explicit `log_sources.sources[]` in main_config carry full connection info and **win by name**.
2. Otherwise, each knowledge-pack `SourceDef` contributes a retriever by merging its `endpoints` block (backend *coordinates*) with credentials from `log_sources.backends`.

The pack `endpoints.kind` maps to a retriever `type` via `_KIND_TO_TYPE`: `elasticsearch`→`elasticsearch` (creds keyed `backends.elasticsearch.<cluster>`), `databricks_uc`→`databricks` (creds keyed `backends.databricks.<workspace>`), `snowflake`→`snowflake` (`backends.snowflake.<account>`), `rest`→`rest` (`backends.rest.<service>`). Pack sources whose backend creds are missing are **skipped with a log line** (graceful degrade). With creds for all four kinds present, one real pack builds all **22 retrievers** (10 ES + 9 Databricks + 2 Snowflake + 1 REST); that pack's own README lists which backend key each of its sources needs. This keeps backend coordinates in the pack (one source of truth) while secrets stay in the gitignored config. `ApiCallGenerator` only advertises serveable kinds to the LLM (`_SERVEABLE_KINDS`).

### A graceful degrade is not a silent one, and an unmet DEPENDENCY is a finding

Skipping an unconfigured source is right; skipping it *quietly* is the same invisible failure as the four limits above. On job **4da14f65**, 18 of 30 pack sources built no retriever (every ELK source: the `analytics-prd` URL is real, so `${ELK_USER}`/`${ELK_PASS}` expanding to `""` fell to the *second* check — a URL with no credentials), the run queried the 12 that remained, and **every stage reported success**. `log_retrieval` scored 0.24 on truncation alone; nothing anywhere scored the 18 sources that were never asked, because a source that was never wired is absent from `logs` exactly as a source that returned nothing is.

Three mechanisms now close that:

- **`LogRetrievalEngine.unavailable_sources`** — `{source: reason}` for every source that built no retriever, recorded via `_note_skip` in each `_merge_endpoint` branch (so a new backend cannot forget) and logged once as a warning at build. A log line is not an output; downstream needs the roster to tell *"this source had nothing"* from *"this source was never asked"*.
- **`/health?deep=1` answers it before a run, not after one.** The roster above is decided **once, at boot**, from the config's backend credentials — so it is knowable the moment the app is up, and the run that discovers it the hard way spends 38 minutes to reach INSUFFICIENT DATA. The endpoint returns `sources_declared` (a count, since *0 unavailable* and *18 of 30* are different answers and only the second names a credential to go and set) and `sources_unavailable` (`{source: reason}`), beside the `storage_ok` that already existed for the same reason one stage later. The UI dot collects **all three** amber states — no LLM credential, an unusable store, unqueryable sources — rather than ranking them, because a deployment missing one credential usually misses several and reporting only the first makes each fix reveal the next. Each hover names the *consequence*: `401`, "lost on the next restart", "INSUFFICIENT DATA — not an empty result". None turns the dot red: the server **is** up in all three, which is exactly what made them invisible, and conflating them with unreachable is how an indicator starts flapping.
- **`required_source_unavailable`** (weight `0.5`, so one alone gates a 0.6 threshold) — `ApiCallGenerator._required_sources` returns `(required, undeliverable)` instead of intersecting the declared set with the offered one and discarding the remainder. A ruleset's `sources:` map is a **hard dependency**, so a declared source that cannot be retrieved was previously indistinguishable from one the pack never declared. On 4da14f65 that dropped `siem_record_misuse_alerts_current` — the ruleset's `alert` source, holding *the incident's own alert record* — and the run reached `NOT A FRAUD` with the only trace being one `alert_facts.locator` line deep in the brief. The scorer reads `undeliverable_required` off the module strictly (list-of-`str`), because a truthy read against the MagicMock modules in tests would gate every run — the `_declinable` lesson, inverted.

The verdict itself was not wrong here (a categorical AUTOMATED-ACTOR exclusion decided it, and that evidence *was* retrieved). The defect is that the run could not have told anyone if it had been.

### A source that was ASKED and did not ANSWER — the same absence, one stage later

The three mechanisms above cover a source that was never *asked*. A source that **was** asked and did not come back is invisible in `logs` in both directions: a non-answer is **absent** from the dict (the shape of a source nobody planned), and an empty answer is **present** with `[]` (which every consumer reads as an answer). Measured on a loyalty referral: two `retrieval_class: primary` sources timed out at 1800s and 7200s, the stage reported `completed` with `source_count: 10`, and the verdict wrote `data_coverage=… because their source returned no rows` — an assertion about an empty result set, made about a query that never came back. The only trace anywhere was stage health 0.282 against the sibling case's 0.610.

**`unanswered_out`** on `retrieve` / `retrieve_with_tunnel` / `_gather` is the seam, beside `keyed_out` and `queries_out` and for the same reason they are out-parameters rather than engine state: one retriever instance serves every concurrent job, so the fact is **per-run**. `pipeline_runner` carries it in `stage_facts["log_retrieval"]["unanswered"]` and hands it to `evaluate_verdict(unanswered_sources=…)`; `main.py`'s inline path does the same. Four rules the shape forces:

- **An answer RETRACTS an earlier non-answer.** A pass ADDS, and the caller carries one dict for the whole run, so a success `pop`s the source — or pass 2 answering what pass 1 could not leaves the run reporting a gap it closed.
- **A failure records the exception TYPE, never its message.** A backend error text carries DSNs and tokens, and this string is rendered in a report; the source's own progress line already carries the message.
- **It is not keyed on `retrieval_class: primary`.** That key is strictly a budget signal, and a source that did not answer is a defect whatever its class.
- **The verdict reports it three ways, because a reader can do three different things.** A `source_unanswered=<source>` note on every subject (rendered under its own heading — *"Data coverage"* says the verdict stands without that corroboration, which is the opposite claim), a per-condition scope note on every check that reads it, and the `data_coverage=` sentence corrected to *"did not answer"*. Only sources the **adjudicating** ruleset declares count, and only where the source is genuinely absent from `logs`: a stale claim about a source that did answer is ignored, since `logs` is the evidence and the claim is not.

`source_outcomes` already carried enough for the health scorer to score `source_timeout`, which is why the divergence was *visible* as a number and nowhere as a fact. Deriving the fact back out of that dict was rejected: `progress_cb` is a UI callback a caller may omit (`main.py` passes none), and parsing a status string would be a second answer to the same question.

### A source ASKED WITH A PLACEHOLDER answered nothing, and the detection existed but only logged

The fourth way `unanswered_out` fills, and the only one where retrieval **succeeded**: the query still carried an unfilled `'<...>'` template marker. That is a syntactically valid predicate, so the backend runs it happily and returns zero rows — the identical shape a source with genuinely nothing to say returns, which downstream is an *answer* (`zero_rows` can declare it decisive).

`base.unresolved_placeholders` has caught that shape since job c6c11f89, and for one release its entire consequence was a `logger.warning`. Measured on job **ca4240c0** (`abnormal_amount`, Run C): the `primary` scope sweep published

```sql
… WHERE creator.office_id = '<unitId>' AND creator.sign.red LIKE '<sign>%'
```

reported *"Retrieved 0 rows"*, the stage reported `completed`, and the report told the operator the actor's wider impact was UNKNOWN "because the scope sweep returned no result" — which attributes to the source what the query did. The table held **81** office-scoped records for that actor. Two distinct sub-defects:

- **The regex missed the wildcard form entirely.** It matched a whole-literal `'<x>'`, so `LIKE '<sign>%'` was invisible: on the measured query the office half was caught and the sign half was not, and a hint spelling both sides with `LIKE` would have been caught on neither. `_PLACEHOLDER_LITERAL` now allows **wildcards only** (`%`, `_`, whitespace) around the marker, so the literal is still nothing *but* a placeholder — `'%<script>%'` is the one shape it cannot tell from a template, and reporting a search for markup as UNKNOWN rather than clean is the safe direction.
- **A warning is not a consequence.** The health scorer cannot take it either: it sees a source with zero rows, and what zero MEANS is a pack declaration that cannot say "empty because we asked with a placeholder". So `_gather` captures the placeholders per source in `_on_query` — the same instant `keyed[source]` and `queries_out[source]` are captured, because the published text is this query's only at the moment it is published — and at the settle point an **empty** result from such a query is converted to a non-answer: omitted from `logs`, named in `unanswered_out`, notified `failed` (scored `source_failed`, 0.15).

**The bound is `key_was_enforced`'s, and it is what makes this safe:** this is a claim about the query that ran, so it is read **only for an empty result**. A placeholder can sit somewhere harmless — a projected label, an `ORDER BY` — and rows that came back are real rows whatever the text looked like. Converting those would delete evidence on the strength of a regex; converting an empty result deletes nothing. Pre-execution refusal was rejected for the same reason.

**And the root cause was a pack gap, not a prompt failure.** The sweep's scan bound *is* the (office, sign) pair, which lives only inside the alert document — the query had no value to substitute. That is what `follow_up_passes` exists for, so `abnormal_amount` now harvests both from `abnormal_amount_alert` on pass 3. The engine half above is what makes the *next* such gap loud instead of invisible.

### An unmet dependency is REPORTED, never injected — and why the force-add was removed

A ruleset's `sources:` map is a hard dependency, and for a while the engine met it the direct way: `_add_missing_required` appended a query for every declared source the planner had not chosen. That is the wrong seam, for three reasons, in increasing order of how badly they bite.

- **It answers a reasoning defect by bypassing the reasoning.** A declared source the planner declined is a defect in what the catalog *tells* the planner — a missing `selection_guidance`, a `description` that reads like an inventory entry. Fixed there, every future incident benefits; injected behind the planner, one incident is patched and the cause is hidden. `scripts/validate_source_selection.py` is the measurement that made this actionable: it replays each persisted understanding, runs *only* query generation, and reports `PICKED` / `DECLARED` / `UNSCOPABLE` / `MISSING`, where a non-empty `MISSING` is the selection defect to repair in the pack.
- **It cannot know which procedure is adjudicating without over-reading.** The pruning heuristic had to keep any ruleset that *could* apply — and a `subject_discovery` procedure applies to everything, since its subject is in the rows by construction. So one pack held two sources on **every** incident it would ever process, regardless of entities. That is not a tuning problem: an engine deciding retrieval from pack declarations is the coupling the whole knowledge-pack design exists to prevent.
- **An unscoped query is not a cheaper check, it is a more expensive absence.** For a source bound to no entity type the incident named, the injected query is a bare date-window scan: measured on a live loyalty run it returned 0 rows and left the condition `unknown` exactly as skipping it would have, having spent a query and its share of the primary budget to do so. What the run loses there is the **check**, not the retrieval — the condition's row selector resolves `from_entity` out of `analysis.extracted_entities`, so a value the incident never carried is unavailable to it however many rows arrive. That is an ENGINE-GAP to report as an unevaluated condition, and issuing an unscoped query whose emptiness then reads as evidence is the opposite of reporting it.

So `_required_sources` still resolves the declaration — for the **ONE** adjudicating ruleset, via the same `select_correlation_spec` → `ruleset_key_for` path the verdict takes — and `_report_unmet_dependencies` reports the result **three ways**, because the reader can do three different things about them:

| Finding | Means | What acts on it |
|---|---|---|
| `undeliverable` | declared, but no retriever was built | a credential; `/health?deep=1` says so before the run |
| `not_queried` | retrievable, scopable, and the planner did not choose it | a catalog fix — or one operator click (`docs/API.md` §5c) |
| `unscopable` | retrievable and unselected, and the incident names no entity type it can filter on | nothing: declining it was correct |

`_scopable(name, entities)` is the split, and it excludes `time_window` deliberately — every source binds it and every incident carries one, so counting it would make the intersection non-empty for everything and the category vacuous.

**`required_source_not_queried`** (weight `0.5`, so one alone gates a 0.6 threshold) is the health signal, and it counts **only** `not_queried`: scoring `unscopable` would gate every run of a `subject_discovery` procedure, which is the same over-reading that made the force-add unprunable. Its detail names the sources and points at the run controls, because a gate the operator cannot act on is a pause, not a review.

### A FIFTH finding, one step earlier: a source the planner CHOSE that produced no query

All four findings above are about a source the planner did **not** choose. The fifth is the opposite and reads identically from the outside: `_parse_tool_calls` used to drop a tool call that failed `RetrievalQuery` validation with one `logger.warning` and nothing else, so a source the planner *selected* left exactly the same trace as a source nobody selected — none. Live: a call arrived as `{"target_log_source": "<source>"}` with no question and no dates, the whole call was discarded, that source is not one the adjudicating ruleset declares so no `not_queried` finding could fire, and the plan an operator approved simply did not contain it. Same shape as the four limits and as `unanswered_out`, one stage earlier: the stage reports `completed` and the conditions reading the source go `unknown`.

`_salvage_tool_call` splits the call into **what the engine owns and what only the planner can say**:

- `date_from` / `date_to` are **engine-owned** (`_ENGINE_OWNED_FIELDS`), because `_enrich_queries` overwrites both unconditionally — from `event_time` where the incident states one, else from the declared default depth. So a call omitting them is *incomplete*, not malformed: they are defaulted to `""` and the enrichment fills them. A **null** is defaulted too, for the reason a null always needs its own branch — a null is not a missing key, so nothing keyed on absence fires, while the model rejects it exactly as it rejects a missing field. A salvaged query that still ends up with no window is already scored, by `query_without_window`.
- `natural_language_query` is the **analytic intent and is never invented.** The engine has nothing to write there, and a fabricated question is how a scoped lookup becomes a window-wide scan. So a call carrying a target and no question is **reported, not repaired**: it lands on `selected_unparseable`, scored as `selected_source_unparseable`, and the remedy is one operator click in the plan editor where a human supplies the question and `build_manual_query` builds the rest. That is the same posture the force-add removal took — a source that should have been queried and was not is a finding on this stage, never a query injected behind the planner.
- `entities` is the only nested shape left, so a validation failure retries once **without** it. The incident's own entities are unioned onto every query by `_enrich_queries` regardless, so dropping the planner's restatement of them costs the query nothing it will not get back — where dropping the call costs the whole source.

Two drops stay silent on purpose, because a finding an operator cannot act on is noise: arguments that are not an object at all, and an object naming no source — in both there is nothing to click. A target that names a source **this run cannot retrieve** is not reported either; the validated path has its own branch for an unknown source name (it redirects to the first offered one), and here there is no query to redirect.

The weight is `0.35`, and its position is the assertion: heavier than `selected_source_unscopable` (`0.1`), where the engine's removal left the plan *correct*, and lighter than `required_source_not_queried` (`0.5`), where the pack said the source was mandatory. Nothing was corrected here and the engine cannot know the source was needed, so one lost pick does not gate and **two do** — one is a nudge, a pattern of them is a planner or schema problem to see before approving the plan. `selected_unparseable` is reset where it is filled, so a run with no malformed call clears the previous incident's list.

Both `dependency_report` and `unselected_sources` are **pure over their arguments** and read no instance state: one `ApiCallGenerator` serves every job on the server, so an attribute holds whichever incident ran last, and the control panel asks about a job that may have finished.

**Measured before removing it, and again after** (2026-08-13, `scripts/validate_source_selection.py --repeat 2` over nine real production jobs, two per installed procedure, understanding replayed from the persisted job document so only selection varies): **9/9 select every hard dependency unaided, on both runs**, and the adjudicating key matched the live run's use case on all nine. Exactly one source the force-add would have injected was not chosen — `automated_users` on a loyalty incident, which the incident cannot scope (it binds neither `loyalty_number` nor `record_locator`), classed `unscopable`, and which the live run had retrieved **0 rows** from. That is the whole case in one row: the injection bought nothing, and what the run actually loses is a *check* the report must name.

**Re-measured 2026-08-16 with a SIXTH procedure installed** (`--repeat 2` over a curated ten jobs, 20 live planner runs; the 2026-08-13 numbers above stand as measured and are not re-pointed): **10/10 select every hard dependency unaided on both runs**, `MISSING: none` on every run, `cross_saved=0`, and the adjudicating key matched the live run's use case on all ten. The new procedure declares nine sources of its own and the gate stayed empty, which is the claim worth having: a catalog entry written for one procedure did not have to be widened for the planner to find it.

**Read BOTH directions of the picked-set diff, because the gate can only see one of them.** Adding nine access sources put two of them (`boarding_access`, `audit_profile_access`) on **one** ATO job, on **1 of 2** runs — a source appearing half the time is the shape the `--repeat` rule exists for, and it is not a defect here: ATO reads access evidence, and the pick is a widening the planner is entitled to. In the other direction two `abnormal_amount` jobs **narrowed**, dropping `audit_raw_access` and `monitor_alerts` they had previously picked, so the cost of the new entries moved *down* rather than up. Neither movement is visible in a `MISSING` column, and a run that only checks the gate would have reported the same green while the corpus shifted under it. Both directions were read off one `--repeat 2` log, which is kept locally and not in the repository (`scripts/out/` is gitignored) — every count that log settled is in the paragraph above, which is the form these citations take throughout (see CONTRIBUTING.md).

## Multiple retrieval passes — a question whose SCOPE the first retrieval discovers

Some questions cannot be asked on the first pass, because the values that scope them do not exist
until something has been retrieved and decoded. The engine therefore **allows** repeating
`query_generation` → `log_retrieval`, and the **use case declares** when that is worth doing. A
pack that declares nothing runs single-pass, byte-identically to before.

**Why this could not be a pack fix.** `process_incident` called `log_retrieval.retrieve()` exactly
once; `decode_logs` runs at the *end* of `_gather`, so an encoded field's contents are known only
after retrieval has finished; and a `TransformStep` reshapes rows already retrieved, it cannot
fetch. The concrete case: an access-grant alert carries its per-event action list encoded, and the
impact question — *did the offices that RECEIVED the grant go on to read the owner's records?* — is
scoped by identities that appear only inside that blob. Asked on pass 1, the query can only be
scoped to the identity the incident named, which is the *administrator*: measured 0 rows in 306.8s,
and 0 rows is the failure that reads as INSUFFICIENT DATA while the stage reports success.

**Only those two stages repeat** (`_PASS_STAGES`). Correlation onward runs once over the
accumulated logs — `evaluate_verdict` is called once by design, and a second reading of the same
rows is how the *weaker* verdict lands on the brief.

**Pass identity is `pass` alongside `stage`**, never a composite stage name: `pass_key(name, n)`
returns the bare name at `n == 1`, so `THINKING_STAGES`, `GATEABLE_STAGES`, `stage_gates.stages.*`,
persisted job docs and every UI label keep matching, and `pass` omitted on an API call means the
current pass. Gates, `control`, `retry_stage` and a mid-run override all address `(stage, pass)`.

**A pass ADDS; it never replaces.** `logs[source]` becomes pass 1's rows plus pass 2's, `queries`
accumulates, `keyed_sources` unions (a source keyed in *any* pass stays keyed), and row-cap
truncation stays **per pass** — merged counts must not hide that one pass hit its cap. The UI keys
its source rows on `source + "#" + pass` for the same reason: keyed on source alone, pass 2's query
text and row count silently overwrite pass 1's in the panel.

**The declaration** (a ruleset key, since whether a follow-up earns its scan is a procedure
judgement — the same rule as `sources:` vs `selection_guidance`):

```yaml
follow_up_passes:
  - pass: 2
    source: access_trail          # a LOGICAL name from this ruleset's sources: map — or a
                                  # LIST, when one harvest answers several sources
    harvest:                      # entity type <- paths in a PRIOR pass's rows
      - entity: receiver_unit
        source: alert
        fields: [decoded.actions.receiver_unit]
    capture: "^([A-Z0-9]{6,9})"   # optional; drops a masked value rather than filtering on it
    window: onwards               # onwards | inherit
    skip_when_empty: true         # nothing harvested -> no pass, and SAY so
    purpose: >                    # or a MAPPING of logical source -> text, per target
      Whether the receivers read the owner's records after the grant.
```

**One entry may target SEVERAL sources, because a pass number is a scarce slot.** One entry
occupied one pass and `jobs.max_retrieval_passes` defaults to 3, so a procedure with three
deferred questions could not ask one of them *at all* — measured on job `ca4240c0`: the source
that lost the numbering race shipped its pass-1 query with a **record locator on an office-name
column** for want of a value that lives only inside the alert, reported `completed, 0 rows`
against a ground truth of 2, and its condition read `unknown / field absent` where the answer
was a PASS quoting a value. The passes it lost to harvested the *identical* leaves from the
*identical* source, i.e. they were one pass wearing two numbers. So `source:` may be a list:
the harvest is read **once**, each target gets its own query, and no extra scan is paid for.
Three properties this needs, each of which fails silently otherwise:

* **`purpose` may be keyed per source**, because the targets share the harvest and not the
  question — and that text is what the retriever writes its backend query *from*. One text over
  two targets asks a reference table the event log's question, which it can only answer empty.
  `pass_purpose` / `pass_targets` (`src/pipeline_runner.py`) render the whole pass for every
  operator-facing sentence: reading `spec["source"]` alone announces a pass that asked one source.
* **A refusal is per target, never collective.** A name with no retriever, or one that can filter
  on none of the harvested types, comes back as a *note* from `generate_follow_up` while its
  siblings are still queried — otherwise one stale name deletes another source's only retrieval.
* **The deferral premise is per target too**: `answered = all(logs.get(t) for t in targets)`. A
  sibling that already replied is not evidence that the one still waiting was ever asked, so
  reading it as "any answered" drops the harvest and with it that source's retrieval.

`pack_validate` checks each target against the `sources:` map individually and warns on a
duplicate target, a `purpose` keyed to a non-target, and a target with no `purpose` under a
per-source mapping.

`src/follow_up.py` `harvest_follow_up(spec, logs, analysis, pack)` reads those paths through the
same `resolve_path` everything else uses, applies `capture`, de-duplicates, drops values the
incident already carries, and stamps `value_form` from the pack's regexes — **unstamped, a
harvested value takes the flat binding, which is the sibling form's column**, arriving at the
cross-form filter through the back door. It returns `(entities, notes)` and never raises into the
pipeline; nothing harvested means the pass is skipped *and reported*, not run unscoped.

**The follow-up target is DEFERRED out of the earlier passes, by both routes.** A source a
follow-up will ask is dropped from the ruleset's hard-dependency *reckoning*
(`_required_sources` — so it is not reported as `not_queried` on pass 1, where it is not yet due)
**and** from whatever the planner chose (`_defer_follow_up_targets`). Both routes matter and the
*planner* one is what fired live — with nothing force-added any more, it is the only one that
removes a query. A deferred target that can build **no retriever** is still reported as
undeliverable rather than silently deferred: a pass that cannot ask its question is a finding
either way. And a pack whose declaration raises defers **nothing** — the run degrades to
single-pass rather than dropping a source nobody will now ask. The plan editor lists a deferred
target too, flagged: adding it on pass 1 merges two scopes under one source name.

**The pass-N query is the narrowing, so a condition reading it must not try to be.** Harvested
values ride on `RetrievalQuery.entities` and deliberately never reach
`analysis.extracted_entities`, so a `row_match` clause over a harvested type resolves to *no
values* and reads `unknown` over the exact rows that answer it. A presence count with no row
selector is the right shape, and it is safe precisely *because* of the deferral: the only rows that
source contributed are the pass's own.

**The request is narrowed to the types the question is about** (`_fallback_request(only_types=)`).
Offering the source's full entity list invited the acting identity back onto a relational query —
the predicate the whole pass exists to remove — so the text names only the harvested sides, states
that they combine with **AND** while values within a side are alternatives, and says in as many
words that any other identifier from the incident belongs to a different party and matches nothing.

`jobs.max_retrieval_passes` (default **3**, `live`) bounds the loop; reaching it logs what was
dropped rather than truncating silently.

## Multi-workspace Databricks

`_merge_endpoint` databricks branch + `LogRetrievalEngine._auth_for_workspace`: different `databricks_uc` catalogs can live in **different Databricks workspaces**, and a PAT is **workspace-scoped** (a token for workspace A gets `403` on workspace B — this is real, not a permission gap). So `backends.databricks` is a **map keyed by workspace name** (like `elasticsearch.<cluster>`): each entry has its own `workspace_url` + `warehouse_id` + `api_key_env`. A pack `databricks_uc` source selects one via `endpoints.workspace: <name>`; `_merge_endpoint` looks up `backends.databricks[workspace]` (missing → skip-with-log).

**Auth per workspace:** `__init__` seeds an `_auth_by_host` cache with the shared `self.auth` under its own host; `_auth_for_workspace(workspace_url)` reuses that SDK/OAuth auth **only on an exact host match**, else returns `None` so the retriever falls back to that workspace's `api_key_env` static token. One real pack spans three workspaces, each with its own token env var — one of which has no credentials configured, so its 2 sources are skipped-with-log.

**Back-compat:** a flat `backends.databricks` block (warehouse_id directly, no per-workspace keys) still works for a source that declares no `workspace` (`endpoints.workspace` absent → `_merge_endpoint` reads the flat block, `workspace=None`). Verified live: `SELECT 1` succeeds on two different workspaces in one process, each with its own token.

**A cross-catalog reroute is a per-source pack declaration, not a config edit.** A warehouse can read **another workspace's catalog** wherever its principal holds `SELECT`, so a source whose own workspace warehouse is unhealthy can be pinned to a different one via `endpoints.workspace:`, catalog unchanged. Not hypothetical tidying: one workspace's warehouse persistently failed *every* statement against its own catalog with `INTERNAL: IO error … Too many open files` (Spark-executor file-descriptor exhaustion — byte-identical across attempts, independent of query shape), returning 0 rows, and 0 rows is the failure that reads as INSUFFICIENT DATA. Rerouting is one line per source in the pack.

The cost lands on the receiving warehouse, so its poll budget must move with the reroute: on one pack the rerouted workhorse went to **960s / 192 polls** (from 660s / 132) because the heaviest projection measures ~223s standalone and 2–3× that under concurrent queueing — and the old cap timed it out → 0 rows → wrong verdict.

## Schema discovery

`DatabricksRetriever._get_field_schema()` has three discovery modes driven by config:

1. explicit `field_schema` → used verbatim;
2. `catalog`+`schema` → discover that one schema's columns from `information_schema.columns`;
3. `catalog` only → scan **all** schemas (`information_schema.tables`), have the LLM curate the fraud-relevant tables once (`SchemaSelection` model), then fetch columns for just those.

The result is cached as `_field_schema` and reused for every per-incident SQL generation. **`self.tables` (from the pack's `endpoints.tables`, threaded through `_merge_endpoint`) scopes discovery to an allow-list** — `_discover_columns` adds `AND table_name IN (...)` so the LLM sees only the intended table's columns, not the whole schema (this stopped it hallucinating truncated table names — a real table's trailing `_4` silently dropped — on large schemas).

**STRUCT/ARRAY handling:** `_discover_columns` also selects `full_data_type`; `_render_schema` calls `_flatten_struct(col, type)` to render nested `STRUCT<...>` columns as dotted leaf paths (`parent.leaf STRING`, depth-capped at 3 / 60 leaves per table), so the LLM can select real struct fields. **Arrays are terminal leaves — never descended** (a dotted path into an `ARRAY<...>` is invalid SQL; elements need `explode()`/`LATERAL VIEW`). After SQL-gen, `_normalize_struct_paths` deterministically rewrites the LLM's `` `a.b.c` `` (one backtick-quoted token Databricks can't resolve) → `` `a`.`b`.`c` `` (valid per-segment field access).

`ElasticsearchRetriever._get_field_schema()` is the analogue: it discovers real index fields via the **field-caps API** (`field_caps(index, fields="*")`), cached, falling back to the configured `field_schema` (or empty) on failure.

**`KibanaRetriever._get_field_schema()` has no mapping endpoint behind `/internal/search/es`, so it samples documents — and it samples PER CONCRETE INDEX, because an index pattern is a set of indices that need not share a document shape.** A flat `match_all` over a pattern is ordered by nothing in particular. Measured on a source spanning two patterns and 293k documents across three shapes (a scored-sessions index, and a raw stream whose pre- and post-migration spellings never co-occur): all 50 docs of the flat sample came back from the **single oldest monthly index**, so the schema handed to the generator held 32 paths from one obsolete shape and none of the two current ones. The generated query then filtered a field no document in the incident's window has and returned 0 rows — and 0 rows from a source is indistinguishable from a source that had nothing to say, while the stage reports success. A `terms` agg on `_index` with `top_hits` per bucket makes coverage a property of the sampling rather than of shard ordering: **32 paths became 121 across 27 indices**, and every identity column the pack binds became visible. Buckets are ordered `_key: desc` — a date-suffixed index name sorts reverse-chronologically, so the current shape survives a truncated bucket cap. A cluster that buckets nothing gets an explicit **flat re-ask with a WARNING** (`_sampled_hits` reads either response shape), because the aggregation request carries `size: 0` and has no hits to degrade to; the fallback is what makes this change unable to be *worse* than what it replaced, and an empty schema is not a loud failure here — the generator is told to infer reasonable field names, so it invents plausible ones.

The same defect lived in `scripts/generate_source_schemas.py`, whose entire purpose is a complete inventory: its flat 500-doc sample wrote down one shape and recorded the other two as absent — the "field that IS in the data but written down as absent" failure, produced by the artifact meant to prevent it. It now stratifies the same way (`_stratified_sample`), so `present_in_sampled_pct` is a fraction of a sample that weights each **index** equally rather than each document, and `sampled_indices` says how many it covered.

**The same sampling had a second, independent hole: an array of objects was not descended, so half of a nested source's field names did not exist as far as discovery was concerned.** `term: {a.b: v}` on Elasticsearch matches whether `a` holds one object or fifty — the array is transparent to the *query*, and must therefore be transparent to *discovery*. `_collect_keys` walked dicts only, so every path under a bucketed field was absent from the schema; `_in_schema` then rejected the pack's binding naming it, logged **STALE BINDING** against a field somebody had measured, dropped the predicate, and left a window-only scan. Measured across one pack's 13 ES sources: **27 declared bindings on 6 sources** were invisible this way, 0 regressions — among them the record locator on three record-misuse alert sources (`svcAlerts.record_ref`, `bookings.record_ref`, `alert.allDiscounts.itemDetails.record_ref`), and one source went from **9 discoverable field names to 73**. `scripts/generate_source_schemas.py` had descended arrays since it was written and says so in its `walk()` docstring, so two readings of one backend disagreed and the *live* one was wrong — verified per concrete index over three rule versions (pre-fix, shipped, uncapped), 0 regressions. Note which way that failure points: **a false stale binding is worse than a true one**, because the pack is corrected toward the defect.

Two things about the cap, both of which were wrong before and neither of which fails loudly. **It counts dotted segments, not recursion levels** — a field name is measured in segments and an array is not one, so `_collect_keys` recurses into list elements at the *same* prefix and the *same* depth. And **it is a runaway bound, so it is set from the measurement rather than to it**: `_SCHEMA_MAX_SEGMENTS = 6` against a corpus whose deepest real path is 5 (217/232/155/58/11 paths at 1–5 segments; ≤6 extra paths per source versus pre-fix, so this is not a token-budget change). The old value admitted 4 and thereby hid `alert.allDiscounts.itemDetails.tenders.tenderType`, a mapped `keyword` naming the payment method on a pricing-fraud alert.

**And once STALE BINDING is trustworthy, it is worth checking — with `field_caps`, and never with a named field list.** A `field_caps` check takes the authoritative reading the retriever cannot (no mapping endpoint behind the gateway), but only `fields=*` is safe on that proxy: a request naming the paths returned nothing for two fields the document sample had just found populated, and `fields=*type*` matched nothing on an index whose mapping carries `…tenders.tenderType`. A named negative there is a machine for deleting correct catalog entries. **A sampled absence and a mapping absence need opposite fixes** — record a discovery bound, versus delete a stale binding — which is why the two measurements — the document sample and the `field_caps` check — are cross-checked for contradictions (0, after the array fix). What survives is real pack work: **49 declared paths named nowhere in any mapping**, plus 3 sources whose patterns resolve to no index at all.

Two consequences for a pack: a source spanning several shapes must bind **every** shape's column for each entity (the bindings are a list), and since nothing here can OR two columns, naming the other shapes' spellings for the generator is `query_hints` work — the one thing that seam legitimately owns.

### A projected struct leaf comes back named after its LAST segment

Discovery says which nested leaves *exist*; `projection:` says which are *selected*; and this is the
third question, which nothing in the pack could answer by reading either — **what the row calls
them.** A statement result is columns plus values and the row is `dict(zip(columns, values))`
(`databricks_retriever.py:1651`, and the same line in the ES|QL and Snowflake readers), so a backend
that names `a.b.leaf` just `leaf` collapses two entries ending in the same word into **one** key, and
`dict` keeps the last. Measured on one live source: **10 projection entries came back as 7 row keys.**
Nothing is empty, retrieval succeeds, and a condition reading either path is answered with the other
one's value.

The engine's own defence is a **prompt instruction** — the generator is told to alias every dotted
path as that path with `.` → `_` — which is the shape this whole guard layer exists to distrust. It
cannot be a guard here, because the alias is chosen while the SQL is written and a rewrite would
have to invent the name. So the *declaration* settles it instead: an entry may carry its own
`AS <flattened path>`, and `pack_validate` reports the rest —

- **`projection-name-collision`**, two severities on purpose. Every colliding entry pinning its own
  alias is an **ERROR** (the row provably cannot carry both, and no generator behaviour changes
  that); at least one colliding by falling back to its last segment is a **WARNING** (the
  flattening convention may still name them apart — a defect conditional on behaviour nothing
  enforces, reported where it can be removed rather than asserted as broken).
- **`field-path-outside-projection`**, the coverage check, whose candidate set is the *projected
  sub-inventory* and therefore has to read the aliases the other way round: **the sub-inventory
  grows from the path an entry READS, never from the name it lands under** (`projection_renames` →
  `_check_projection_coverage`). Grown from the alias, one installed pack's newly-aliased
  projections produced **44 findings**, every one of them a leaf under a projected struct that
  resolves fine at run time.

Both readings live in `src/utils/projection.py` and neither is the other's inverse:
`projection_names` is optimistic (*what will this be called* — the alias, else the dotted entry),
`returned_names`/`colliding_row_names` pessimistic (*what can the row call it* — the alias, else the
last segment), and `expand_projection`/`projection_sources` answer *what does this entry read*, which
is the only one a schema or coverage question may use. Aliasing to the exact flattened path is
**behaviour-preserving**: `resolve_path` already tries that spelling at every prefix cut, so a
condition keeps its dotted path and an aliased struct keeps its nested tail.

## Entity→field mapping

`src/retrievers/field_mapping.py`: before generating the backend query, both retrievers call `map_entities(llm, query, field_schema, knowledge_pack)` — the LLM binds each incident entity to a field **chosen only from the discovered schema** (knowledge-pack `field_aliases` passed as a prior, never authoritative; low-confidence and unmappable entities dropped, not invented). `render_filters()` turns the resulting `{entity_type: field}` map + the entity values into `field = 'value'` hints injected into the SQL/ES|QL prompt. This is what makes "same entity, different field name per source" work automatically — each source maps against its own real schema.

**A measured per-source binding outranks a high-confidence mapper pick.** The confidence gate measures the model's uncertainty about a field name, and the model has no standing to be *confident* about one somebody measured on this exact target either. So where the pack declares `entity_bindings` for a source **and** the declared spelling is in the discovered schema, that spelling wins and the override is logged with both candidates. Found on `session_anomaly_alerts`, where an `alert_type` alias in the glossary steered the mapper onto `ir.type` at high confidence: that column holds `OTH`, so the generated `term` clause matched zero of the feed's rows while the real discriminator (`ir.userRemarks = "UBA"`, measured 91:4) sat unqueried. The retrieval survived on three unrelated substring clauses, which is the shape of the problem — a dead clause in an `OR` costs nothing visible until it is the only clause left.

Three bounds on the override, each load-bearing: it fires **only** when the declaration is confirmed present in the discovered schema (a declaration is a measurement of the target, but a schema doc can be stale and a stale one must not beat a live read); it **never crosses a value form**, or it would re-create exactly the cross-form filter `render_filters` exists to prevent; and the low-confidence path keeps its own separate behaviour — a rejected pick whose *spelling* the pack confirms is restored rather than dropped, since the gate was measuring the wrong thing there too.

## A pack's own retrieval notes

Everything above is the mechanism. A source's *measured* facts — its partition column and why that
is not the column the investigation is about, its version model, the column aliases a downstream
stage reads by name, the identity-suffix trap that makes `LIKE` mandatory where `=` returns nothing
— are per-source knowledge and belong with the pack (`knowledge/<domain>/`, in that pack's own
retrieval notes). None of them may become a literal in this document, for exactly the reason they
may not become one in `query_hints`: a fact nobody can check by reading it goes stale silently, and
the first symptom is a source that returns nothing.
