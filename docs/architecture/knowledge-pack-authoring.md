# Authoring a knowledge pack

How to point AFIR at a new domain. This is the *how-to*; the mechanisms it walks are specified in
[`knowledge.md`](knowledge.md) (loader + RAG), [`retrieval.md`](retrieval.md) (query guarantees +
partition discovery) and [`verdict-engine.md`](verdict-engine.md) (every condition kind's exact
semantics).

The engine in `src/` contains no entity name, no source name, no field path and no fraud rule.
Everything domain-specific is data in `knowledge/<domain>/`, so re-targeting is an authoring job.
The pack that ships plus the template beside it are the two ways in:

```bash
cp -r docs/knowledge-pack-template knowledge/mydomain   # the template: every key, documented
less knowledge/mock_domain/README.md          # a tiny pack that RUNS, in an invented domain
```

Then point at it:

```yaml
# config/main_config.yaml
knowledge:
  pack_dir: "mydomain"        # or set AFIR_KNOWLEDGE_DIR to relocate knowledge/ entirely
```

**The method, as opposed to the keys, is written down in `src/knowledge/skills/*.md`** — seven
documents on how to do this work rather than what is declarable: probing a source before binding
it, the four questions to measure before writing a line, the three ways a condition silently never
fires, why a fixture that passes can be worthless, what to check before concluding the engine is at
fault. Read them alongside this file; they are also injected into the in-app pack assistant, which
picks the ones a request needs (`src/knowledge/pack_skills.py`). They are engine surface and
therefore domain-free — every rule in them is technique, and `validate_pack`'s
`neutrality-collision` check is what keeps it that way.

**An empty or missing pack loads cleanly.** Every accessor returns empty, and the pipeline infers
what it can from the data — as it did before packs existed. A pack only adds precision; it never
gates execution. That is what makes incremental authoring safe: you can run after every step.

---

## 1. The one structural rule

**Pack root = shared. `use_cases/<name>/` = specific.**

The seam is not a matter of taste. Knowledge about the **data** — what an element is, which column
holds it, the trap somebody measured on it — is true for every procedure that reads that element,
so a second use case must not have to re-discover it or copy a field path. Knowledge about the
**procedure** — whether that element being present clears the case or condemns it — is one team's
judgement and belongs to their use case.

The operative test: **if a second use case might legitimately disagree, it belongs in the use
case.**

```text
knowledge/<domain>/
├── entity_glossary.yaml      ← shared: what to recognise in incident text
├── source_catalog.yaml       ← shared: where the data is, how to query it SAFELY
├── schemas/                  ← shared: complete field inventories (RAG-only)
├── data/                     ← shared: reference lookup tables
├── reporting.yaml            ← shared: domain-wide report vocabulary
├── shared/
│   ├── concepts/             ← shared: prose about the DATA (loads UNTAGGED)
│   └── checks/               ← shared: importable check MECHANICS
└── use_cases/<name>/
    ├── rules.yaml            ← specific: the verdict ruleset (the procedure)
    ├── playbooks/            ← specific: narrative investigation guidance
    ├── concepts/             ← specific: prose about the PROCEDURE (tagged)
    ├── cases/                ← specific: resolved investigations (GITIGNORE THESE)
    ├── data/                 ← specific: lookups only this procedure uses
    └── reporting.yaml        ← specific: this procedure's wording
```

Loader mechanics worth knowing before you name a file:

- `shared/concepts/` loads **untagged**, and that is the entire sharing mechanism — untagged, a
  concept is RAG-retrievable by every use case. Tagging it makes an ownership claim the loader
  then enforces by hiding it from everybody else.
- `use_cases/` **wins on key collision** with a flat-root file of the same key.
- Playbooks are deduped by `playbook_id` across the flat root and every use case.
- `data/` from the root **and** from every use case flattens into one global `pack_data` dict keyed
  by **file stem**. Stems must therefore be unique pack-wide; prefix a use case's own tables.
- An unresolvable `use:` is **fatal at load**, checked eagerly over every ruleset. A silently
  dropped condition is a check that was never evaluated, and in a report that is indistinguishable
  from a check whose source returned no rows.
- **Pydantic silently drops unknown YAML keys.** A brand-new pack key does nothing until it is
  added to `EntityDef` / `SourceDef` (or, for a playbook's `correlation:` block, retained
  explicitly in `_read_playbooks`). If a key you invented has no effect, this is why.

---

## 2. The order to build in

Each step leaves the pack loadable.

### 2.1 Entities — `entity_glossary.yaml`

Pure vocabulary: what to recognise in incident text, plus candidate field names as a **prior** for
mapping. No query logic.

```yaml
abbreviations:            # a CLOSED list — see below
  SHP: "Shipment — one parcel movement, identified by a 10-character tracking code"

entities:
  - type: shipment
    description: >
      One parcel movement. The subject of a refund-fraud investigation.
    recognition_hints: >
      A 10-character code, two uppercase letters then eight digits.
    field_aliases: [tracking_code, shipment_id, parcel_ref]
    value_pattern: "^[A-Z]{2}[0-9]{8}$"
    cardinality: one-per-refund-claim
    examples: [RT48192043]
    used_by: [PB-REFUND-001]
```

Two things to get right here, because nothing downstream can repair them:

**`abbreviations:` is a prohibition, not a convenience.** It is emitted *ahead of* the entities as
a closed list — use only these expansions, and write an unlisted acronym as-is rather than
guessing. An LLM asked to summarise an incident will expand an unfamiliar acronym rather than leave
it alone, and a plausible wrong expansion is indistinguishable from a right one once it is in the
report's prose. Measured: a three-letter code the pack uses for a *region* was expanded into an
invented alert-type name, in a sentence that otherwise read as authoritative.

**`value_forms:` when one entity type has several non-interchangeable surface FORMS.** In one real
pack a business identifier and an authentication login are both labelled `user` and bind to
different columns. The form is classified in the engine from the pack's regex, stamped on
`ExtractedEntity.value_form`, and `render_filters` routes each value to its own form's binding —
dropping it if the source binds none, never unioning it onto the sibling form's column. **No later
prose can restore a distinction the data model discarded**, so if your domain has one, declare it
now.

**And then bind it the same way everywhere, or nowhere.** A source binds a form-declaring entity
either per form (`entity_bindings: {user: {sign: [a], login: [b]}}` — each value routed to its own
column, a value whose form the map does not name dropped) or as a flat list (`{user: [a, b]}` —
every form's value filtered onto every column). Both are legitimate: flat is right for a source
that stores the whole incident text in one field. Doing *both* for one type across a pack is not,
because then the routing is real on some sources and absent on the rest — where one form's value
lands on the other form's column, a valid predicate matching nothing, reported as a source that had
nothing to say. `pack_validate` warns per flat source (`value-form-binding-mixed`) whenever any
sibling binds the same type per form, and **errors** on a form name the glossary does not declare
(`unknown-value-form-binding`), which is the typo the conversion invites: no value can classify as
one, so every value of the form it was meant to catch is silently dropped. All-flat is silent —
forms still drive classification, `stem` widening and `co_identity`, none of which reads a binding.

**A form may also declare WHERE it sits inside the stored value, and there are two directions.** Both
exist for the same failure and it is the worst kind: a **valid predicate matching nothing**. The value
is real, the column is bound, the query is well-formed, and the source answers 0 rows — which reads
downstream as a source that had nothing to say.

| Key | Says | Rewrite |
|---|---|---|
| `stem: "^(...)"` (one capture group) | this form CONTAINS a shorter one — take it OUT | `widen_stem_literals` offers the captured core **beside** the value |
| `match: "{value}??????"` | this form IS a segment of a longer stored one — place it INSIDE | `widen_match_patterns` offers a positional wildcard **beside** the value |

Measured for each: an identifier's long form asked of a column storing its core (946,805 rows of one
length, zero of the other), and a 3-character organisational segment a report names asked of a column
storing the 9-character identifier (`unitId = 'DEL'` against `AAASV0991`). **Which precision or which
width a column stores is declared nowhere per column, so both are offered rather than substituted and
the data decides** — sound because a widened predicate returning nothing proves the narrow one does,
and `key_was_enforced` is read only for an EMPTY result. In `match`, `?` is exactly one character and
`*` is any run; `{value}` is mandatory (`pack_validate` errors on a template without it — a pattern
that cannot name the value is mechanically decidable and provably inert). Declare the STORED width as a
form with no `match` at all: there is nothing to widen, and a template on it would only match longer
values on some other column. Neither rewrite ever touches a negation, and neither accepts a value
already carrying the route's own wildcard characters. Details and the refusals in
`docs/architecture/retrieval.md`.

`field_aliases` is only a prior. Real column names are discovered live (Databricks
`information_schema.columns`, Elasticsearch `_field_caps`, Snowflake `information_schema.columns`)
and that discovery is authoritative — so the same entity maps to a different column per backend,
automatically.

### 2.2 Sources — `source_catalog.yaml`

Where the data is, and **how to query it safely**. This is the highest-leverage file in the pack.

Everything in it is advisory *except* seven keys, which are rewritten into the generated query
**after** generation by `src/retrievers/query_guards.py`. That is deliberate: **a prompt
instruction cannot be relied on to beat another prompt instruction.** All six ride through
`_attach_query_guards`, the one seam every retriever config passes, so a new backend branch cannot
silently turn a pack declaration into a no-op.

| Key | Enforces |
|---|---|
| `require_all_entities: true` | a composite-key source gets **AND**, never OR. Measured: one identifier appears on 89,937 rows across 89,937 *different* org units, so OR-ing the two halves of the key returns other keys' rows |
| `identity_keys: [[a, b], [a, c]]` | an actor is named by a *combination*, and several work. Priority-ordered; the first **fully** satisfied candidate wins, and a partial one is skipped rather than half-applied. The order is per-source and should be MEASURED |
| `identity_synonyms` / `identity_scopes` | the shape an **event log** has instead of a fixed key: `identity_synonyms` are several columns that may hold the **same** value (OR them, or the row is missed — one auth table nulls its login column on 32.0%), `identity_scopes` are **separate** identity facts on the same document (the unit, the role) which are **AND**-ed. `identity_synonyms` also takes a list of lists — OR inside each family, AND between — because two independent identifiers each spread over several columns, flattened into one group, leave the second arm unbounded by the first |
| `never_filter: [f, …]` | evidence fields: returnable, never filterable. A filter on one deletes the very rows the check exists to read |
| `default_filters: {f: v}` | a mandatory slice, **AND**-ed onto the query last and onto **every arm** of a set operation. The opposite key to the one above, and which one a column needs is a MEASUREMENT — see below |
| `partition_columns` | the **override only for what backend metadata cannot say**. Layout is normally *discovered* (Databricks `partition_index`, Snowflake `clustering_key`); a VIEW reports no partitions, and no catalog carries `role` / `pad_days` |
| `epoch_time_columns: [{name, unit}]` | declares only the name and unit. The engine converts *this incident's* window, injects it already-converted, and repairs a non-intersecting bound afterwards |

The last two exist because **a slow query is a wrong answer, and so is a query about the wrong
year** — both fail the same invisible way. Motivating measurement: a worked example in a prose
hint read `1753315200000 = 2026-07-24T00:00:00Z`, off by exactly one year (that value is 2025).
The generator copied the era, and a live source returned 0 rows with its document sitting in the
index. **Never hand-write an epoch literal or a partition layout into `query_hints`.**

**`never_filter` and `default_filters` are opposites, and the choice between them is a MEASUREMENT of
the column, never a reading of its name.** Both address one shape: a column that is not evidence about
anybody — a deployment environment, a record type, a calling application — written by the generator
into the same OR-ed group as every subject clause, where it satisfies the group on its own and the
subject scoping becomes decoration. **The failure is a FULL result**, so nothing that asks whether the
source answered can see it, and the row cap then truncates on the population's rows.

- **Cardinality 1 → `never_filter` alone.** A constant cannot narrow, so the predicate has exactly two
  outcomes: no-op, or **deletion** of every row that omits the field. Refusing it loses nothing.
  Measured on one estate: cardinality 1 on all seven sources binding the column, three of them only
  partially populated — AND-ing it would have deleted 25,511,308 documents in the name of a scope that
  excludes nothing.
- **Cardinality N → declare BOTH.** `never_filter` lifts the constant out of wherever the generator put
  it and `default_filters` AND-s it back on once. The strip alone drops a real environment scope and
  mixes test traffic into a fraud verdict; the pin alone leaves the vacuous OR arm standing beside the
  pinned conjunct. Measured on the audit-trail views of the same estate: five values, the production one
  at 99.82% — there, dropping the predicate really does widen the read.

So the column NAME decides nothing (`application_phase` is not `phase`), the FAMILY decides nothing, and
a spelling that binds no row is its own defect: a declaration on it is a valid predicate matching
nothing. Probe each relation and each bound spelling separately, and separate `''` from `NULL` — a
mandatory conjunct excludes a row whose field is absent, and only one of those is visible to a
`COUNT(col)` figure. **Do not pin what you have not read**: an unreachable source (no credentials, so
neither its vocabulary nor its identifier case has been measured) gets the strip only, and the pin the
day the account is filled in.

**One more key sits at the TOP of this file rather than under a source, because it is about an incident
and not about a store:**

```yaml
retrieval:
  default_lookup_days: 30      # how far back to read when the incident states no date at all
```

An incident whose text carries no time expression has no window, and a window is the **hard scope** of
every query. Measured: the understanding stage invented one (the start of the run's own day), it became
the scope of all 29 retrievals, and with that cleared the generator then picked **eight different
windows across 29 queries** — so sources that must be compared to each other were read over different
spans. The engine holds no depth of its own, and deliberately: how far back a store must be read to
find the conduct behind an undated report is a fact about *your* domain's retention, its alerting lag
and how long the behaviour typically runs. Declare it here, or in `log_sources.default_lookup_days`
(`main_config.yaml`, also editable from the Configuration tab) — this key wins, since retention is pack
knowledge. Declare neither and the generated per-query windows stand as they are, with a warning naming
both places. The window is anchored on the incident's **ingestion timestamp**, so re-running the same
incident later reads the same span.

Three further keys shape *planning* rather than the query, and each is emitted as its own labelled
line under the source — a negative or conditional fact buried in a paragraph reads as a feature
list:

- **`not_answered_by: [{question, ask_instead}]`** — what this source *cannot* answer, naming the
  one that can. A `description` says what a source holds, which does not stop a planner asking it
  the wrong question; the rows then cannot contain the answer, and the absence reads as "nothing
  found" rather than "asked the wrong source".
- **`selection_guidance: {choose_when, skip_when}`** — whether it is worth asking at all.
- **`zero_rows: {health_weight, meaning}`** — because an empty result is sometimes **the finding**
  (no row on an automation list means the actor is human). Without this the health scorer counts
  your answer as a defect. `0.0` fires nothing, and the discount plus its reason are printed,
  since a silently smaller number cannot be checked. **`0.0` is also a claim about the REPORT**,
  not only the score: at that weight the `meaning` is printed in the evidence the LLM narrates
  from and moved out of the report's gap list, so write it as a statement of what was
  established. At any other weight it is read as a recorded reason for a gap that remains a
  gap — so if the emptiness leaves something `unknown`, say so there and do **not** use `0.0`.
  Both readings are honoured; picking the wrong one either buries your finding or invents one.

**Two ways to get a source retrieved, and they are NOT interchangeable.** Listing it in a
ruleset's `sources:` map makes it a hard **data dependency** — right when a *condition reads it*,
since an unevaluable condition is not the planner's call. `selection_guidance` leaves the decision
with the planner — right when relevance is genuinely incident-dependent. Choosing the first for a
source no condition reads spends its full scan cost on every incident.

**But a hard dependency is not a hard ADD.** It used to be: the engine appended a query for every
declared source the planner had not chosen, and that made the declaration an engine-level retrieval
decision taken from pack contents — the coupling this whole design exists to prevent, and measured
to put two primary-class scans on every incident of four unrelated procedures. Today the unmet
dependency is **reported** (undeliverable / not queried / unscopable), scored by the stage-health
gate, and addable by an operator (`docs/architecture/retrieval.md`, "An unmet dependency is
REPORTED, never injected"). **What that means for you as an author:** the `sources:` map states
what the procedure needs and makes a shortfall visible, but the thing that actually gets the source
picked is still `selection_guidance` + `not_answered_by` on the source itself. If a run reports
`not_queried`, the fix is in those two keys — verify with
`scripts/validate_source_selection.py`, whose `MISSING` column is exactly this gap.

**`encoded_fields:`** — a column whose value is a *payload* rather than a datum (a base64 blob
holding a delimited table). This is the one shape where a source returns the richest evidence it
has and the investigation still does not see it: the blob **is** in the rows, every stage reports
success, and the report states the detail "is not available" — a stronger and more damaging claim
than "we did not look". Measured on one live job: the alerting source carried the full per-event
list, base64-encoded, on both of its rows; nothing decoded it, so the narration declined to
enumerate the events and the one condition that reads them reported `unknown`.

The pack has to name it, and it cannot be a `query_hints` sentence: base64 is indistinguishable
from an opaque identifier without knowing the field *means* to be decoded, and the decode happens
**after** retrieval, where no prompt runs. So the pack declares the field and its separators
(`src/knowledge/pack.py`'s `SourceDef.encoded_fields` lists every key) and `src/encoded_fields.py`
does the mechanics, writing real nested columns back onto the row before correlation sees it.
Three traps worth knowing before you declare one:

- **`into:` must resolve to a NESTED child** (`ir.decoded_x`), never a flat dotted key. A list of
  dicts under a flat key resolves to zero leaves silently — the decode succeeds and every reader
  sees nothing.
- **A separator is a candidate LIST, not a constant.** How a producer escaped its rows is not
  something a pack can pin: the live payload separates records with the two characters `\` `n`, so
  a `splitlines()` reading returns one record and the table reads as a single malformed row.
- **These are also the archetypal `never_filter` entries** (a predicate on a blob matches the
  encoding, not the content), and both declarations are needed: `never_filter` gets the value
  RETURNED, `encoded_fields` gets it READ.

**`projection:`** — the columns to SELECT, and on a wide nested table a **lower bound on what any
condition can read**. Nothing outside it arrives, so a leaf you forgot reads `unknown` with the
spelling correct, retrieval successful and the rows real. `pack_validate` reports that as
`field-path-outside-projection` — but only where the pack ships a schema inventory for the target,
and it emits `projected_path_lists` beside the finding so a pack that stopped being checked does not
read like a pack that passed.

**Alias every nested entry as its own path, `.` → `_`:**

```yaml
projection:
  - payload.userInfo.sign AS payload_userInfo_sign
  - fare.ticket_document.ticket.numbers AS fare_ticket_document_ticket_numbers
```

Because **a backend names a projected struct leaf after its LAST segment only.** A row is
`dict(zip(columns, values))`, which keeps the last on a collision, so two entries ending in the same
segment arrive as **one** column and one of the two values is served under the other's name — a
determination made on a value from somewhere else, with nothing empty and nothing to notice.
Measured on one live source: **10 projection entries came back as 7 row keys.** The engine's only
generic defence is a *prompt instruction* telling the generator to alias, which is the shape this
whole file exists to distrust — pinning the alias in the pack is the declarative enforcement of it.

Aliasing to the exact underscore-flattened path is **behaviour-preserving**: `resolve_path` already
tries that spelling at every prefix cut, so a condition keeps reading `payload.userInfo.sign`, and a
struct alias keeps its nested tail (`fare.ticket_document AS fare_ticket_document` still resolves
`fare.ticket_document.ticket.numbers`). Two consequences worth knowing:

- **Project the LEAF, not the parent struct**, where a condition reads a leaf. A bare `SELECT` of a
  large struct is how a query stops returning at all, and the wide read is charged to a source whose
  timeout decides the verdict.
- **An alias the entry cannot pin makes reads under it unverifiable.** An expression
  (`concat(a, b) AS both`) names a column no schema records, so `both.whatever` is accepted as an
  opaque prefix and the coverage check says nothing about it. That is the resolver's own rule, not a
  licence: what you spell there is on you.

**`retrieval_class: primary`** says what a source's *absence* costs: the investigation cannot be
concluded. It buys a far larger timeout, resolved once onto all three coupled budgets. It belongs
in the pack because a timeout is configured per *backend* and a backend's sources are not equally
important — a multi-TB projection beside second-long reference lookups. It changes neither
selection nor zero-row scoring.

**YAML anchors.** If you use `&`/`*` (one real pack does, heavily), a `*ref` whose `&anchor`
is missing raises a ComposerError that makes the **whole pack load empty** — not just that entry.
Nothing crashes; you get a pack-less run.

**No secrets, ever.** Endpoints here are coordinates: index name, catalog/schema/table, service
nickname. URLs, users and tokens live in `config/main_config.yaml` under `log_sources.backends`,
by env-var name, keyed to match the pack's `endpoints`.

### 2.3 Schemas — `schemas/<source>.yaml`

The **complete** field inventory per source, arrays descended, `explode_path` per leaf. RAG-only:
returned exclusively to a caller passing `filter_type="schema"`, so adding one cannot change an
existing prompt.

It exists so **a field that EXISTS is never recorded as absent.** Retriever schema discovery stops
at an `ARRAY<STRUCT<…>>` — a dotted path into an array is not valid SQL — which is right for
generating SQL and wrong as a statement about the data.

Half of the document is generated from backend metadata (structure, `explode_path`,
`populated_pct`, `partition_index`) and half is hand-written (`description:` per leaf). That split
is the point: the structure is discovered so it cannot go stale, the meaning is authored because no
catalog carries it. Declare `kind:` per leaf — `flag` vs `array_container` vs `array_leaf` — because
matching a check's kind to the data's shape is the single most common authoring error (see §2.5).

### 2.4 Playbooks — `use_cases/<name>/playbooks/*.md`

Narrative investigation guidance, embedded into the FAISS KB and retrieved by similarity to the
incident text. Front-matter carries a stable `playbook_id`; **never change one after first
ingestion** — the vector store and every `used_by:` reference depend on it. Keep the H2 (`##`)
section structure, because the retriever chunks on it.

The optional `correlation:` block is a **functional** input, not documentation — the join spec for
that pattern (`keys`, per-source `fields`/`time_fields`, `time_window`, `key_filter`), consumed by
correlation's layer-1 key resolution, which outranks both the LLM's guess and data-driven
discovery.

Write the playbook **before** the ruleset. Writing down what to retrieve and how to join it is how
you discover what the ruleset needs.

### 2.5 The ruleset — `use_cases/<name>/rules.yaml`

The procedure. Twenty condition kinds are available (`element_absence`, `element_presence`,
`field_equality`, `time_gap`, `record_absence`, `cohort_membership`, `field_flag`,
`distinct_count`, `route_membership`, `value_mismatch`, `value_matches_pattern`,
`delimited_field_mismatch`, `velocity_count`, `stub`, plus the six compositional ones in §2.5.3 —
`all_of`, `any_of`, `none_of`, `numeric_compare`, `event_order`, `value_equivalence`), all dispatched
on `kind` in `src/correlation.py`. Their parameters are specified in
[`verdict-engine.md`](verdict-engine.md); the template documents the first fourteen inline.

Replace the template's single `stub` with one real condition and run before writing twelve.

**Five things to get right.** Each of these has produced a wrong live report, or would on the
first pack to omit it:

**Every number the procedure chose must be DECLARED, because the engine no longer supplies one.**
`distinct_count` and `velocity_count` need `max` as a whole number; `time_gap` needs `max` as a
window in the engine's own grammar — `90m`, `4h`, `2d`, and **not** `24`, `1 week` or `PT48H`; a
ruleset with corroborating (non-decisive) `fraud_indicator` conditions needs
`indicator_threshold`. Omit one and the check reports `unknown` and `pack_validate` errors. These
four used to fall back to 3, 1, one hour and 2 respectively, which printed the invented figure as
the procedure's own finding — `4 distinct values, more than the 1 allowed` over a bound nothing
in the pack states — and in `indicator_threshold`'s case reached the FRAUD label under a voting
rule nobody wrote. `max: 0` is a perfectly good declared bound ("nobody other than the subject",
with `exclude_subject: true`) and is never read as unset; `indicator_threshold: 0` is rejected,
since zero indicators already satisfy it and every subject would be accused.

**A `label` is the REQUIREMENT, not the finding.** For an exclusion a FAIL *negates* the label, so
printing the label states the opposite of what was found. A live report read
`DECISIVE CONDITION: <actor> is not automated (<flag path>=True)` under a verdict resting on the
actor being an automation — the label and the evidence beside it contradicting each other, with no
way for the reader to tell which half to trust. No LLM is involved: the note comes from
`fail_detail`, never the label. (A fraud *indicator*'s label already states its finding, so a FAIL
affirms it and it prints unchanged. The asymmetry is the polarity, not an inconsistency.)

**`polarity` decides which way a FAIL argues; `decisive` only decides whether it settles.** A
decisive *indicator* FAIL reaches the positive label; a decisive *exclusion* FAIL reaches the
negative one. Confuse them and the verdict inverts while every row of the condition table still
reads correctly. Without any `fraud_indicator` polarity a ruleset can only ever reach FALSE
POSITIVE or the textbook-fingerprint path, so a non-textbook fraud is *unreachable* — that was a
real defect, caught by a third live incident the engine cleared and the expert called fraud.

**`decisive_on: [fail]` is what most checks want.** `decisive: true` alone also makes an `unknown`
force INSUFFICIENT DATA — wrong for a check whose FAIL is conclusive but whose absence of evidence
is *ordinary*. Without the asymmetry, every ordinary subject reads as insufficient.

**`exclusion_kind: categorical` vs heuristic.** A categorical exclusion rests on an **attributed
fact** — who acted, and when — and cannot be outvoted by indicators: once such a fact is on the
record the indicators lose their *meaning*, not merely their weight. A heuristic exclusion (an
inference about the subject's shape) can rightly be outvoted. Measured: an *automated* actor beats
three behavioural indicators at once, because identity evidence is prior to behavioural evidence.

**And match the kind to the data's shape.** `counters:` and `arrays:` name paths holding a COUNT or
a COLLECTION, and the engine concludes presence by counting leaves under them. A **boolean** holds
neither, so it contributes no countable leaf and the path reads as one that never arrived — the
check returns `unknown` on every row, *including* the rows where the flag is plainly `true`. A
boolean is a `field_flag`.

Other blocks the ruleset may declare: `trigger:` (what the detector fires on — see §5),
`condition_groups` (presentational only; a group decides which table a check prints in), a `gate:`
condition (asked first; a FAIL exits with `out_of_scope`, **not** an adjudicated NOT FRAUD),
`indicator_threshold`, `lock_target` (the containment recommendation's vocabulary and per-platform
action sets — there is deliberately **no default action verb** anywhere in the engine, because two
platforms' action sets are rarely interchangeable and a guessed verb is an operational error),
`case_builder`, `notification` and `scope_discovery`.

Name all four rollup labels, including **`out_of_scope`** — the one most often omitted. Without it
a gate FAIL prints as the false-positive label, which claims the case was examined and cleared when
the procedure in fact declined to adjudicate.

### 2.5.1 `entry_signals:` — how this procedure's fraud shows up in someone else's rows

Optional, and a pack declaring none is byte-identical to one that never heard of the key. It feeds
the advisory link lane (`src/links.py`, `docs/architecture/correlation.md`): at the end of
correlation the engine asks, of every *other* ruleset in the pack, whether this run's evidence
argues for that procedure too — and reports it as a referral addressed to a human, never as an
input to this run's verdict.

**The declaration is INBOUND: a ruleset declares how its OWN pattern looks in another procedure's
evidence.** Adding an eleventh procedure is then one new file. Declared outbound it would be N²,
and every new procedure would have to edit ten files it does not own.

```yaml
entry_signals:
  - id: rejected_authentication_burst
    direction: antecedent          # I may be the CAUSE of the incident in hand
    opens_with: {entity: user}     # MUST equal this ruleset's own subject_entity
    when:
      source: auth_events          # a logical name in MY OWN `sources:` map
      where: [{field: value.payload.action.action, any_of: [Rejected], match: exact}]
      min_rows: 3
    window: lookback:30d           # the vocabulary `follow_up_passes` already uses
    strength: 0.6
    base_rate: {kind: stub}        # or {fires_on: N, of: M, measured: "<date>", corpus: jobs}
```

**Six rules, and only four of them have a validator.**

**`opens_with.entity` must equal this ruleset's `subject_entity`** — a `pack_validate` **error**. A
leg is opened by a subject *value* and by nothing else, because `subject_entity` is what the
sibling ruleset iterates and every one of its conditions is written against. A signal opening on
anything else declares a leg that procedure cannot walk.

**`base_rate` is mandatory, and it licenses NOTHING.** `kind: stub` is the honest form of "not
counted yet". A signal firing on most runs is not a detector — one live discriminator was on 340 of
586 alerts — and a reader given no rate cannot discount one that fired, so absent, `stub`, and a
corpus too small each report `entry-signal-unmeasured` (a **warning**, three sentences, one code).
But a rate is a *historical statistic*, and what licenses spending on a link is this run's own
evidence: **rung-1 PASS**, the target ruleset's `gate: scope` conditions re-evaluated against the
rows this run retrieved. So `auto_probe: true` on an unmeasured signal is **not** an error, and a
measured rate buys exactly one thing — an additive term in the `semi_auto` confidence score
(`signal_discriminates`, 0.20). See `cross-procedure-links.md` for the precondition and the caps.

**Which makes `when.where` the safety surface, so authoring it IS the control.** A signal with no
`where` clause fires on any `min_rows` rows of the source, i.e. on the source having been
*retrieved* — a fact about this run's plan rather than about its evidence — and reports
`entry-signal-broad-selector` (a **warning**, loudest with `auto_probe: true`, which spends a query
on it). A warning and not an error because it is sometimes the honest declaration: a source that
only exists when the shape does really is answered by its own presence, and no mechanical check can
tell that from an author who left the clause out.

**The `where:` vocabulary is small, and it is the whole vocabulary.** Clauses AND together, values
within one clause OR, both sides are compared as `str(v).strip().upper()`, and `match` is `exact`
or `substring`. It cannot express a negation, a numeric bound, or a comparison between two columns
of one row — so a shape that needs one of those is not authorable as a signal, and the right thing
to do is say so in a comment where the block would have gone. One ruleset in the shipped <domain>
pack does exactly that: its only discriminating shape is a two-column row comparison, and bare
presence on the source it would read fires on nearly every administrator.

**A signal source must be SUBJECT-KEYED, and nothing can check that for you.** The firing test
reads *every row* of `logs[source]` and is deliberately not subject-scoped — it runs after
retrieval, over whatever came back. So a signal declared on a **cohort or scope-sweep** source
fires on rows belonging to other parties and hands them to the sibling procedure under *this*
subject's name: a finding whose row is real, whose subject is real, and which says nothing
whatever about the two being related. Whether a source is subject-keyed is a judgement about what
it returns, so `pack_validate` cannot decide it — the shipped pack pins its nine chosen sources
**by name** in its own test, with the sweeps listed as the exclusion set. Prefer the source whose
query the incident's own subject value scopes.

**`direction` picks the window, and the window is the causal claim.** `antecedent` means "I may be
what caused this" and looks *back* (`lookback:<N>d`); `consequent` means "this may have caused me"
and looks *onwards*; `inherit` keeps the incident's own window. It is the same three-valued
vocabulary `follow_up_passes` already implements, and a referral composed from a link carries the
window the direction implies.

### 2.5.2 `link_escalation:` — may a sibling's run act on a link to me by itself?

Optional, one mode for every source procedure plus an optional per-source override. Omitted, the
deployment's `correlation.links.mode` decides (default `semi_auto`).

```yaml
link_escalation:
  mode: semi_auto                  # planned | semi_auto | auto
  from:
    abnormal_amount: auto      # this ONE sibling's evidence, I trust further
```

**An escalating mode requires that this ruleset declare a `gate: scope` condition** — a
`pack_validate` **error** (`link-escalation-no-scope-gate`) otherwise. Rung-1 PASS is the licence,
and a ruleset with no applicability test has no rung 1 to pass: the mode is not "refused this time"
but unreachable on every incident forever, which reads from every artifact exactly like declaring
nothing. `planned` is silent here, because a composed referral a human executes needs no licence.

The other two errors are the ones an author cannot self-check by reading this file: an unspellable
mode is DROPPED by the resolver (`link-escalation-bad-mode`, closed vocabulary), and a `from:` key
naming no ruleset in the pack can never match, so the pair silently takes the general mode
(`link-escalation-unknown-source`) — it is another procedure's name, so a rename elsewhere breaks it.

### 2.5.3 The compositional kinds

Six kinds exist so that a pattern the first fourteen cannot express becomes new YAML rather than a
Python commit. Full semantics in [`verdict-engine.md`](verdict-engine.md) §"The compositional
vocabulary"; what breaks when a key is absent is below. All six are **polarity-neutral** — what a
FAIL argues for is the importing ruleset's declaration, exactly as for every other kind.

**`all_of` / `any_of` / `none_of`** — `children:` is a list of full condition dicts, each of which may
be a `use:` import or another composite. Three-valued: `all_of` fails on any child fail and reads
`unknown` when an unknown child could still have decided it; `none_of` passes only when every child
fails. **The parent produces one report line and owns every weighting key** — `decisive`,
`decisive_on`, `polarity`, `exclusion_kind`, `report_group`, `order`, `label`, `fail_detail`. A child
declaring one of those is silently inert, so it is a `pack_validate` **error**
(`composite-child-weighting`); fewer than two children is `composite-children` (a one-child composite
is the child, wrapped); nesting deeper than five is `composite-too-deep`. Give the parent a
`fail_detail` for the same reason every other condition needs one: the label states the requirement.

**`numeric_compare`** — `source`, `field`, `aggregate`, optional `where` / `group_by` / `fallbacks`,
plus `operator` and `bound`. `aggregate` is one of `count`, `distinct`, `sum`, `min`, `max`, `avg`,
`median`, `ratio`, `mode`, `mode_share`; `operator` is one of `>`, `>=`, `<`, `<=`, `==`. **`bound`
here is a float**, unlike `max` on the older counting kinds — `bound: 0.25` is read as 0.25 and not
as 0. Omit `aggregate`, `operator` or `bound` and the check reads `unknown` and the validator errors:
an undeclared bound is not a small bound. `group_by` compares the **largest** group, so `==` cannot
be answered under it (`numeric-compare-group-equality`, a warning). `ratio`'s denominator is the row
set **before** `where`, so `where` selects the numerator. **`exclude_subject` is not available here**
(`numeric-compare-exclude-subject`, an error): it is a three-way reading of what an emptiness means,
not a filter, and it belongs to `distinct_count`.

Two things decide whether a reading survives a row cap, and both are the engine's arithmetic rather
than a declaration: `count` / `distinct` / `max` / `mode` only rise, `min` only falls, `sum` only
rises while nothing in the column is negative, and `avg` / `median` / `ratio` / `mode_share` move
either way and are therefore always `unknown` on a truncated source. `==` never survives truncation.
Pick the aggregate that your question can be answered with under a cap, or accept that the check will
read `unknown` on the runs that matter most.

**`mode` and `mode_share` answer with a VALUE**, which is why they exist: every other numeric
aggregate needs the value named up front (`where` + `ratio`), and "whichever value is most frequent"
is the question a targeting pattern is. The winner's identity is printed as part of the finding; ties
resolve by highest frequency then lowest key, so two evaluations of one row set name the same value.

**`baseline:`** — makes `bound` a **multiplier** of a second computed aggregate rather than an
absolute figure: `baseline: {aggregate, source, field, where, group_by, per}` with `bound: 3` is
"three times the cohort figure". The baseline population is measured from its own source and its
completeness is never inherited from the subject side, so a truncated or scope-empty baseline reads
`unknown` rather than comparing against a partial cohort — `median` and the percentile family are
only ever answerable under a complete read. Seven errors bound the declaration; the one worth knowing
is `baseline-unread-kind`, because a `baseline` on a kind that does not read it is a threshold the
author believes is relative and the engine applies as absolute.

**`event_order`** — an ordering claim, which `time_gap` is not. Both sides keep `time_gap`'s shape
(`start:` / `end:`, each `{source, fields, where}`), and add `relation` (`after`, `not_before`,
`before`, `not_after`), `quantifier` (`every` or `any`) and an optional `tolerance` in the engine's
window grammar. **Neither `relation` nor `quantifier` has a default** — the relation *is* the check,
and `every` and `any` disagree over the same rows — and an absent `tolerance` means **exact**, never
borrowed slack. Convert a `time_gap` whose label contains an ordering word: `time_gap` compares
magnitudes and accepts either order, so a check labelled *"the reversal followed the booking"* passes
on a reversal timestamped minutes before it.

**`value_equivalence`** — `source`, `field`, `form`, plus `operator` and `bound`, where `form` names a
pipeline from §2.5.4. Without an `anchor:` it counts the largest equivalence class (a *collision*);
with `anchor: {source, field}` it counts the values equivalent to that anchor (a *targeting*
question). The class key and its members are printed, and so is the number of values the form could
not read — a class of 4 drawn from 60 values of which 55 were unresolvable is not the finding it
reads as. A class can only grow, so the reading survives truncation upward only.

### 2.5.4 Equivalence forms — `shared/equivalence_forms.yaml`

The third pack-root sharing mechanism, beside `shared/concepts/` and `shared/checks/`. A form is a
named answer to "when are two textual values *the same thing* for the purpose being adjudicated" —
which is a domain judgement, differs between deployments of one domain, and is therefore declared
here and owned nowhere in `src/`. Flat `{name: pipeline}`, no namespace, pack-wide.

```yaml
staff_login:
  description: An operator login, compared on its alphanumeric core   # optional, for the reader
  project:                          # value -> key; equal keys are equivalent
    - case: upper                   # upper | lower | fold
    - keep: alnum                   # alnum | alpha | digits
  compare: {edit_distance: 1}       # optional; a pairwise relation over the projected keys
  linkage: single                   # required with `compare:` unless the condition names an anchor
  min_length: 4                     # a shorter key is UNRESOLVABLE, not a match
```

**Projection ops** (value → key, transitive, O(n)): `case`, `keep`, `strip: <chars>`, `prefix: N`,
`suffix: N`, `tokens: {split, order, take}` (`order` is `as_written` or `sorted`; omit it for
`as_written`, since sorting collapses *more* values together), `collapse_repeats: true`,
`map: {from: to}`. **Comparison ops** (two keys → bool, not transitive, O(n²)): `exact: true`,
`contains: true`, `shared_prefix: N`, `shared_suffix: N`, `edit_distance: N` (Levenshtein),
`min_overlap: N`. One op per `project:` step. A form may declare both families — project first, then
compare.

**The engine defaults nothing.** No relation, no threshold, no minimum length, and no form: an
`edit_distance` with no N, a `shared_prefix` with no N, an anchorless pairwise form with no `linkage`,
and a condition naming a form no pack file declares each read `unknown` with an authoring sentence.
A silent fallback to `exact` is the one wrong answer that still looks like a working check — the count
is well formed, the report prints it, and the relation it was counted under is not the one you wrote.

**`linkage` is a question about the shape of the question, not a tuning knob.** A pairwise relation is
not transitive, so *single* linkage (A~B, B~C ⇒ one class of three) and *complete* linkage (every
member must match every other) produce different classes over identical rows and therefore different
verdicts. Declare the one your procedure means. It is required whenever a `compare:` form is used for
grouping; with an `anchor:` there is nothing to cluster, and declaring it there is a warning.

**`min_length` is the collapse guard, and its absence is only a warning.** `prefix: 3` against a
two-character value yields the value; `keep: alpha` against a numeric value yields nothing — and then
every such value matches every other, which fabricates a class out of the form's own failures. A
value the form cannot read is **unresolvable: excluded from every class, counted, and named in the
finding** — never canonicalised to `""` and never passed through unchanged, the same rule as an absent
`encoded_fields` part being omitted rather than blank. Skip `min_length` only for a form over a
fixed-width code, where there is no shortening step to guard.

**`normalize: <form>` is the other way in**, available on `distinct_count` and on
`numeric_compare`'s `distinct` / `mode` / `mode_share` — the aggregates a projection actually changes
(`count` reads text but counts rows, so a form there is inert and reported as such). **`normalize` is
one key over two vocabularies**: on `field_equality` and on a `row_match:` selector it names a fixed
engine mode (`identifier` / `exact` / `id_suffix`), so a *form* name there is an **error** — it would
silently mean nothing. And a form declaring `compare:` is refused on those seams rather than
half-applied, since they canonicalise and have nowhere to run a pairwise pass.

`pack_validate` reports every one of these; run it after writing a form, because each failure mode
here changes the relation instead of raising.

### 2.6 Report vocabulary — `reporting.yaml`

Two keys, both optional, at two scopes (pack root = domain-wide, `use_cases/<name>/` = one
procedure).

- **`phases:`** — an ordered `[{keywords, label}]` list classifying one chronology event into an
  incident phase. Matched against `"<source> <action>"` lower-cased, **first hit wins**, so the
  ordering *is* the semantics: the specific rule goes above the source-name rule, or an event in a
  source whose name contains "alert" is filed under "Alert and detection" instead of under the
  thing being investigated.
- **`phrases:`** — `{slot: text}` for eleven named slots (`containment_target`,
  `containment_action`, `containment_action_unknown`, `selling_platform`,
  `indicator_driven_fraud`, `categorical_exclusion`, `notification_draft_heading`,
  `notification_closure_heading`, `wider_evidence_framing`, `narration_concepts`,
  `reconstruction_shape`).

Three properties a new slot must preserve:

- **Every slot no-ops when absent.** The engine's own default is a complete, procedure-free
  sentence, so a pack shipping no `reporting.yaml` still produces a correct report.
- **Substitution is `str.replace`, never `.format`.** Procedure prose contains braces, on which
  `.format` raises. An unknown placeholder is left verbatim — visible in the artifact, therefore
  self-correcting.
- **Scope it.** A `§` clause number at the pack root cites it in **every** use case's report,
  which is exactly the defect that moved these strings out of `src/`.

**Resolution: a mapping merges per sub-key, a list replaces.** `phrases` is a bag of independent
named slots, so a use case re-wording one must inherit the rest. `phases` is an ordered decision
procedure, and merging two authored orderings produces a precedence nobody wrote. Resolved
whole-key (as it was), declaring a single scoped phrase silently discarded the entire domain-wide
map — and the loss is invisible, because every slot falls back to the engine's default, so the
result reads as a pack that never declared wording rather than as one whose wording was dropped. A
well-authored pack *hides* that defect: a single-use-case pack ships no root `reporting.yaml` at
all, which is precisely why it never surfaced on the deployed one.

### 2.7 Promote to `shared/` — once a *second* use case needs it

Not before. This is a de-duplication mechanism, not a style rule.

Split each half by where it can be disagreed with:

```yaml
# shared/checks/shipment_elements.yaml — MECHANICS ONLY
manually_reviewed:
  kind: field_flag
  source: ledger
  flag_fields: [refund.manually_reviewed]
  expected: false
  label: "Refund was not manually reviewed"     # neutral, states the finding
  pass_detail: "no manual review is recorded against this refund"
  unknown_detail: "the review flag was not returned, so an adjudicated claim …"
```

```yaml
# use_cases/refund_fraud/rules.yaml — WEIGHTING
- use: shipment_elements/manually_reviewed
  report_group: validation
  decisive: true
  decisive_on: [fail]
  exclusion_kind: categorical
  order: 20
  fail_detail: "the refund was manually reviewed and approved by a named person"
```

```yaml
# use_cases/courier_collusion/rules.yaml — SAME CHECK, OPPOSITE MEANING
- use: shipment_elements/manually_reviewed
  report_group: fraud_indicators
  polarity: fraud_indicator
  decisive: false
  order: 30
  label: "Refund passed review despite a handler/POD divergence"
  fail_detail: "a review approved this claim even though the POD names a different courier"
```

The same FAIL clears one case and incriminates the other, and neither ruleset restates the field
path. The library file therefore carries **no** `decisive`, `decisive_on`, `polarity`,
`exclusion_kind`, `report_group` or `gate` — a library that leaked one procedure's opinion would
quietly become the place weighting decisions get made.

The file **stem is the namespace**: `shared/checks/shipment_elements.yaml` → `use:
shipment_elements/<id>`.

The override is a **one-level merge**: these keys replace the library's, and a **list replaces
rather than concatenating**. That matters — a deep merge unioning a `counters` list would point the
check at a leaf that does not exist on the source, and selecting a non-existent struct leaf fails
the **whole** query, not just that leaf.

Concepts split the same way. `shared/concepts/x.md` says what the elements *are*;
`use_cases/<name>/concepts/y.md` says what they *mean for this procedure*. Link across with
`[[concept_id]]`.

Equivalence forms (§2.5.4) are the third mechanism and the only one with **no scoped half**: a form
answers when two values are the same thing, which is a fact about the data rather than a judgement
about a procedure, so `shared/equivalence_forms.yaml` is flat and pack-wide with no per-use-case
override. Two procedures that need different relations need two named forms, not one form resolved
twice.

---

## 3. Wiring the RAG

The pack's Markdown is embedded into a FAISS knowledge base and retrieved by similarity;
everything in §2's YAML is injected **deterministically**, not by similarity. Two different
mechanisms, and the distinction matters when something does not reach a prompt.

```yaml
# config/main_config.yaml
rag:
  use_rag: true
  sentence_transformer_model: "all-mpnet-base-v2"
  embedding_dim: 768
  max_retrieved_documents: 5
  similarity_threshold: 0.5

  sources:
    - type: playbook          # DECLARE FIRST — see below
      name: playbook
    - type: pack_schema
      name: pack_schema
    - type: document          # local folders of pdf/docx/txt/csv/json (or a UC Volume)
      name: local_documents
      enabled: false
      paths: []
    - type: confluence
      name: confluence
      enabled: false
    - type: databricks_table  # rows of a UC table mapped to docs
      name: db_fraud_kb
      enabled: false
```

- **`type: playbook`** yields the pack's playbooks **plus** its concepts **plus** its cases — all
  three already in the homogeneous `title`/`content`/`type`/`metadata` schema, so it is a
  zero-I/O adapter. **Declare it first**: the deterministic fallback used when the embedding model
  cannot load reads playbook docs by that key, so playbook context is never lost.
- **`type: pack_schema`** yields `schemas/*.yaml`, one doc per table. Retrieval-only and opt-in
  (`filter_type="schema"`), so enabling it cannot change an existing prompt.
- `name` becomes the knowledge-base source key and must be unique. Adding a new backend is one
  `KnowledgeSource` subclass plus one entry here.

**What is NOT similarity-retrieved**, and therefore cannot be fixed by tuning
`similarity_threshold`: the glossary (seeded into the understanding prompt via
`glossary_prompt()`), the source catalog (`catalog_prompt()`), the resolved verdict spec, the
report vocabulary, and the concepts a ruleset names in `case_builder.concepts` — those are
injected into the brief **by id**, deterministically. Omit `case_builder.concepts` to surface every
concept the use case ships.

So a concept can be RAG-retrievable and *not* in the brief. That is the intended default:
**RAG sees everything, brief injection is opt-in**, because the brief is the deterministic channel
and its contents should be a decision rather than a similarity score.

**Offline note.** `conftest.py` sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` and resolves the
embedding model from the local HF cache. If the model is absent and the network is blocked, the RAG
layer degrades to the deterministic playbook fallback rather than failing the run.

---

## 4. The tests that keep a pack honest

The suite is split along a line worth understanding before adding to it:

| File | Proves | Lives on |
|---|---|---|
| `tests/test_pack_template.py` | the **template** loads, and every key it documents is a key `src/` reads | main |
| `tests/test_mock_domain_pack.py` | the **engine** holds no procedure — same code, second domain, no leaked vocabulary | main |
| `tests/test_knowledge_pack.py` | the **loader**: collision precedence, shared/scoped tagging, reporting resolution | main |
| a pack's own `test_<pack>_regression.py` / `test_<pack>_source_hierarchy.py` | that **pack** encodes its procedure correctly, over real production rows | that pack's branch |

**Main proves the ENGINE; a pack's own branch proves the PACK.** A pack test cannot show the engine
is generic, and an engine test cannot show a pack is right.

### Write your pack's tests like this

**Assert on the OUTPUT, never on the pack's own declaration.** A test that reads the YAML back
proves only that somebody typed the block. `trigger:` was declared by three shipped rulesets with a
comment saying it made the report state the real trigger, guarded by a test asserting the pack's
own declaration — and nothing in `src/` read it. The block was inert, so the report defect it was
written to fix stayed live in every run while the test stayed green.

**Declared-but-unread is the worst of the three states.** Unread-and-undeclared is honest;
declared-and-read works. Declared-but-unread makes an author reading the pack believe the fact is
in force.

**Verify a fix by mutation.** Break the fix, watch the guarding test fail, restore, confirm
byte-identical. A test you have not seen fail is not known to test anything.

**Fixture data tidier than production data hides bugs.** A health check comparing the LLM's *prose*
to a canonical source id fired on every entry of every real run, pinning one stage's score at
0.00 — and it was invisible in the suite, because the fixtures used ids like `src_a` where the
real field holds a whole descriptive sentence naming the source, the identity and the session.
When a field is LLM-authored free text, the fixture must be verbatim output from a real run.

**Commit your fixtures.** A fixture that isn't committed cannot pin behaviour. If they contain real
records, they travel with the pack's branch and the pack's gitignore rules.

### Standing test rules

Never call a real LLM or backend; mock `llm_client.structured_output` and the backend clients.
Duck-type rather than `isinstance` across the flat / `src.` import boundary — the same Pydantic
class reached both ways has two module identities, so `isinstance` can be `False`.

---

## 5. The failure mode every convention here defends against

**A check that cannot fire is indistinguishable, in the report, from a check that had nothing to
find.** Zero rows → decisive conditions `unknown` → INSUFFICIENT DATA, while the retrieval stage
still reports success.

It has arrived by every route, each one measured on a live run:

| Route | What it looked like |
|---|---|
| An unbounded partition scan | slow, then empty, then "insufficient" |
| An epoch literal in a prose hint, off by one year | a live source returned 0 rows with its document sitting in the index |
| A filter on an evidence field | the query deleted its own evidence |
| An absence check naming an array's *leaf* instead of its container | the very case the check existed to catch reported `unknown` on every row |
| A boolean read by a counting check | `unknown` on rows where the flag was plainly `true` |
| A timeout firing on the system of record | the cap *decided* the verdict rather than degrading it |
| A pack key declared, documented and read by nothing | the defect it was written to fix stayed live, invisibly |

Which is why the pack's guarantees are **enforced after generation rather than prompted**, why
partition layout is **discovered** and epoch windows **computed**, why `zero_rows` lets a source
answer by being empty, and why the `trigger:` a ruleset declares is now injected into the narration
prompt as ground truth *plus* an explicit prohibition:

> WHAT THE DETECTOR FIRES ON (ground truth — the allegation this case is about): … Do NOT state or
> imply any other trigger, and do not present a retrieved fact as the reason the alert was raised
> unless it is this one.

The prohibition is the load-bearing half. Nothing in the retrieved rows says which condition raised
the alert, so a narration asked to explain the case reaches for whichever retrieved fact looks most
suspicious and states *that* as the trigger — which on one live run produced "the alert fired
because no payment was recorded", for a detector that never looks at payment, on a record that did
carry a payment element. The report's premise was both unalleged and false, and it sent the reader
hunting for evidence of something nobody had claimed. Absent a declared `trigger:`, the report says
**nothing** about one: silence is correct, an invented trigger is not.

---

## 6. Checklist before you ship a pack

- [ ] `load_knowledge_pack()` returns non-empty for every subsystem you populated.
- [ ] No secret, URL, username or token anywhere in the pack — coordinates only.
- [ ] `cases/` is gitignored, and you verified it with `git status --ignored` **before** the first
      real case file landed.
- [ ] Every `use:` resolves (an unresolvable one is fatal at load, so a clean load proves it).
- [ ] Every `data/` file stem is unique pack-wide.
- [ ] Each source that can legitimately return zero rows declares `zero_rows`.
- [ ] Each source whose absence blocks a conclusion declares `retrieval_class: primary`.
- [ ] Each condition's `fail_detail` states the finding, not the requirement.
- [ ] `out_of_scope` is among the declared labels.
- [ ] Each check's `kind` matches the data's shape (a boolean is a `field_flag`).
- [ ] No epoch literal and no hand-written partition layout in `query_hints`.
- [ ] Your pack's tests assert on **output**, and you have watched each one fail.
