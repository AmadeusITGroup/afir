---
name: binding-a-source
title: "Binding a source: the declarations that are enforced, and what each one costs to get wrong"
when: "Editing source_catalog.yaml — adding a source, a filter guarantee, a projection or an entity binding."
always: false
triggers:
  - source
  - catalog
  - source_catalog
  - retrieve
  - retrieval
  - query
  - filter
  - never_filter
  - default_filters
  - projection
  - partition
  - epoch
  - endpoint
  - selection
  - planner
  - not_answered_by
  - selection_guidance
  - encoded
  - decode
  - identity
  - identity_keys
  - value form
  - alias
  - primary
  - zero_rows
---

# Binding a source

`source_catalog.yaml` is the highest-leverage file in a pack. Most of it is advisory — text a
model reads while writing a query. A small set of keys is different: they are **enforced after
generation**, by rewriting the query the model produced. That split exists because *a prompt
instruction cannot be relied on to beat another prompt instruction*. If a guarantee matters,
it has to be one of the enforced keys; written as prose it is a suggestion, and a suggestion
that loses is invisible.

## The enforced keys

| Key | What it guarantees |
|---|---|
| `require_all_entities: true` | a composite-key source gets **AND**, never OR |
| `identity_keys: [[a, b], [a, c]]` | an actor is named by a *combination*; priority-ordered, first fully-satisfied candidate wins, a partial one is skipped |
| `identity_synonyms` | several columns that may hold the **same** value — OR them, or the row is missed. A list of lists means OR inside each family, AND between |
| `identity_scopes` | separate identity facts on one document (the unit, the role) — **AND**-ed |
| `never_filter: [f, …]` | evidence fields: returnable, never filterable |
| `default_filters: {f: v}` | a mandatory slice, AND-ed on last and onto **every arm** of a set operation |
| `partition_columns` | override only for what backend metadata cannot report |
| `epoch_time_columns: [{name, unit}]` | name and unit only; the engine computes the window |

### The composite-key rule

If an actor is identified by a combination of parts, the predicate must be conjunctive.
OR-ing the parts returns every row matching *either*, and for a part that is widely reused
that is effectively the whole population — measured on one estate at tens of thousands of rows
spread across as many distinct units. `require_all_entities` and `identity_keys` are how you
say so; the ordering inside `identity_keys` is a measurement (which combination actually binds
rows), not a preference — `pack_probe pair <source> <table> "<part a>" "<part b>"` is that
measurement: it returns all four cells, and `left` > 0 with `both` == 0 is the answer that
stops you declaring the pair as one key at all.

### `never_filter` and `default_filters` are opposites, and the choice is a measurement

Both address one shape: a column that is not evidence about anybody — a deployment
environment, a record type, a calling application. The generator writes it into the same OR-ed
group as the subject clauses, where it satisfies the group *on its own* and the subject scoping
becomes decoration. **The failure is a FULL result**, so no "did anything come back" check can
see it, and the row cap then truncates on the population's rows.

Which key applies is decided by the column's **cardinality**, never by its name, and the
cardinality is one command: `pack_probe population <source> <table> <column>` gives it beside
the populated counts, `pack_probe values` gives the spellings.


* **Cardinality 1 → `never_filter` alone.** A constant cannot narrow, so the predicate has
  exactly two outcomes: no-op, or deletion of every row where the field is absent. Refusing it
  loses nothing.
* **Cardinality N → declare both**, and understand the order: the strip lifts the constant out
  of wherever the generator put it, and the pin AND-s it back on once. Strip alone drops a real
  scope; pin alone leaves the vacuous OR arm standing beside the pinned conjunct.

The risk is asymmetric, which is why the two keys are not equally free: `never_filter` can
never lose a row, while a pin on an unmeasured column — or an unmeasured *spelling* — deletes
every row. So: probe each relation and each spelling separately, separate `''` from `NULL`
(a mandatory conjunct excludes a row whose field is absent, and only one of those shows up in a
count of non-null values), and **do not pin what you have not read** — an unreachable source
gets the strip only, and the pin the day its credentials exist.

### Partitions and epochs are computed, never written down

A partition layout is *discovered* from backend metadata; declare `partition_columns` only for
what metadata cannot say (`pack_probe cost` reports the elapsed time against the budget the
source will really run under — record the *margin*, not the number). An epoch window is *computed* from the declared `{name, unit}`.
Never hand-write an epoch literal or a worked example of a converted boundary into prose: one
such example was a year out, the generator copied the era, and a live source returned zero rows
with its documents sitting right there in the index.

## The keys that decide whether a source is asked at all

A `description` answers neither question a planner has. It says what the source holds, which
does not stop the planner asking it a question it cannot answer — the rows then cannot contain
the answer, and the absence reads as *nothing found* rather than *asked the wrong source*.

* **`not_answered_by: [{question, ask_instead}]`** — what it cannot answer, naming the source
  that can.
* **`selection_guidance: {choose_when, skip_when}`** — whether it is worth asking at all.

These two are the *only* way a source gets chosen. If a run reports a declared source as
not queried, the fix is here, in what the catalogue **tells** the planner — not a query
injected behind the planner. Fixing it here fixes every future incident; injecting one query
fixes one incident and hides the cause.

One more trap on selection: **a name outranks its own description.** The planner reads the
identifier first, and a name that suggests the wrong subject loses against three paragraphs of
perfectly correct caveat. If a source keeps being chosen for the wrong question, or skipped for
the right one, consider whether its *name* is the thing that is wrong.

**Two ways to get a source retrieved, not interchangeable.** Listing it in a ruleset's
`sources:` map makes it a hard data dependency — correct when a *condition reads it*, since an
unevaluable condition is not the planner's call. `selection_guidance` leaves the decision with
the planner — correct when relevance is genuinely incident-dependent. Choosing the first for a
source no condition reads spends its full scan cost on every incident. A hard dependency is
**not** a hard add: an unmet one is *reported*, not injected.

## `zero_rows` — when empty is the answer

An empty result is sometimes the finding (no row on an exclusion list means the subject is not
on it). Declare `zero_rows: {health_weight, meaning}` so the health scorer does not count your
answer as a defect. Two readings, both honoured, and picking the wrong one either buries a
finding or invents one:

* **`health_weight: 0.0`** is a claim about the *report* as well as the score: the `meaning`
  is printed in the evidence the narration is built from and moved out of the gap list. Use it
  only when the emptiness genuinely establishes something.
* **Any other weight** reads as a recorded reason for a gap that *remains* a gap. If the
  emptiness leaves anything `unknown`, say so and do not use `0.0`.

## `projection` — a lower bound on what any condition can read

Nothing outside the projection arrives. A leaf you forgot reads `unknown` with the spelling
correct, retrieval successful and the rows real.

**Alias every nested entry as its own underscore-flattened path**, because a backend names a
projected struct leaf after its **last segment only**. A row is a zip of columns and values,
which keeps the last on a collision, so two entries ending in the same segment arrive as *one*
column and one value is served under the other's name — a determination made on a value from
somewhere else, with nothing empty and nothing to notice:

```yaml
projection:
  - payload.actor.login AS payload_actor_login
  - payload.target.login AS payload_target_login
```

Two consequences: **project the leaf, not the parent struct** where a condition reads a leaf (a
bare select of a large struct is how a query stops returning at all, charged to a source whose
timeout decides the verdict); and an alias over an *expression* names a column no schema
records, so reads beneath it cannot be checked by anything.

## `encoded_fields` — evidence the rows carry but nothing reads

A column may hold a *payload* rather than a datum. This is the one shape where a source returns
its richest evidence and the investigation still does not see it: the blob is in the rows, every
stage reports success, and the report states the detail *is not available* — a stronger and more
damaging claim than "we did not look". It cannot be a prose hint, because an encoded value is
indistinguishable from an opaque identifier without knowing the field means to be decoded, and
the decode happens after retrieval where no prompt runs. Three traps:

* **`into:` must resolve to a NESTED child**, never a flat dotted key — a flat key resolves to
  zero leaves, silently, and the decode "succeeds".
* **A separator is a candidate LIST, not a constant.** How a producer escaped its records is
  not something a pack can pin; a payload separating records with a literal backslash-n reads as
  one malformed record under a line-splitting assumption.
* **An absent part is omitted, never empty-string.**

The same field almost always needs `never_filter` too. The two declarations are about different
things: `never_filter` is about the value being **returned**, `encoded_fields` about it being
**read**.

## Surface forms, precision and position

One entity type can carry several non-interchangeable surface forms — a business identifier and
an authentication login, both labelled the same type. The form is classified from a pack regex
and each value is routed to *its own* form's binding; a value whose form the source binds is
dropped rather than unioned onto the sibling form's column. Three related declarations, all
about a valid predicate that matches nothing:

* **`value_forms[].stem`** — one capture group naming the *core* of a longer identifier, for a
  column that stores only the core. Offered *beside* the full value, never instead of it, because
  which precision a column stores is declared nowhere per column: the data decides.
* **`value_forms[].match`** — a `{value}` template with `?`/`*`, for a form that is a *segment*
  of the stored value. Offered beside the equality.
* **`co_identity: {forms, via, prefer}`** in the glossary — two forms that are **one** identity,
  which the verdict would otherwise adjudicate twice under two headings. The merge is licensed
  by *evidence*, never by shape: a pair merges only on a retrieved row binding both values.

Both widening declarations are hinted per source through `field_mapping` (`stem_literals`,
`match_patterns`), refuse a negation, and refuse a value already carrying the wildcards.

## `retrieval_class: primary`

Says what a source's *absence* costs: the investigation cannot be concluded. It buys a far
larger timeout and nothing else — it changes neither selection nor zero-row scoring. It belongs
in the pack because a timeout is configured per *backend*, and a backend's sources are not
equally important: a multi-terabyte projection sits beside second-long reference lookups.

## Two things that make a whole pack vanish

* **YAML anchors.** A `*ref` whose `&anchor` is missing raises a composer error that makes the
  **whole pack load empty** — not just that entry. Nothing crashes; you get a pack-less run.
* **No secrets, ever.** Endpoints here are coordinates: index name, catalog/schema/table, a
  service nickname. URLs, users and tokens live in the config, by env-var name, keyed to match
  the pack's `endpoints`.
