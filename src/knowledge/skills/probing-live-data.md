---
name: probing-live-data
title: "Probing a live source before you declare anything about it"
when: "Before binding a source or column, choosing a threshold, or explaining a zero-row run."
always: false
triggers:
  - probe
  - measure
  - column
  - schema
  - field
  - binding
  - bind
  - zero rows
  - no rows
  - empty
  - cardinality
  - threshold
  - namespace
  - identifier
  - value form
  - index
  - table
  - cost
  - slow
  - timeout
---

# Probing a live source

A probe turns a belief about the data into a measurement. It is not a test — it needs
credentials, it hits the real system, and its output is numbers a human reads once and then
records in the pack as a comment.

## Do not hand-roll one: the tooling already exists

Two helpers ship with the engine, both pack-agnostic, both asking through **the retrievers a
real run builds** — so credentials, TLS, coordinates, row caps and per-source timeouts are
whatever the config and the pack actually say, and never a constant in a scratch script.

* **`src/knowledge/pack_probe.py`** — one question at a time, read-only (every statement is
  checked for a read verb and refused if it chains; there is no write path). As a CLI:

  ```
  python -m src.knowledge.pack_probe sources                  # reachable, in which language,
                                                              # and which DECLARED ones are not
  python -m src.knowledge.pack_probe leaves      <src> <tbl>            # what a ROW carries
  python -m src.knowledge.pack_probe population  <src> <tbl> <col>      # total/not-null/not-blank/distinct
  python -m src.knowledge.pack_probe values      <src> <tbl> <col>      # the stored forms
  python -m src.knowledge.pack_probe spread      <src> <tbl> <col>      # the stored lengths
  python -m src.knowledge.pack_probe selectivity <src> <tbl> "<pred>"   # the base rate
  python -m src.knowledge.pack_probe pair        <src> <tbl> "<a>" "<b>"# all four cells
  python -m src.knowledge.pack_probe cost        <src> <tbl>            # elapsed vs the budget
  python -m src.knowledge.pack_probe sql         <src> "<statement>"    # anything else
  ```

  As a library when the question needs a loop: `probe = await Probe.open()`, then the same
  methods; each returns a `Measurement` (`ok` beside `rows`, so a failure is not an empty
  result) or a `Comparison` (`verdict()` refuses to call a zero-against-a-zero-control a
  finding). Read `sources` FIRST: a declared source that built no retriever cannot answer,
  and that is not a fact about the data.
* **`scripts/generate_source_schemas.py --pack <dir>`** — the complete field inventory per
  source into `knowledge/<pack>/schemas/`, arrays descended, `--measure` for per-leaf
  population. Re-running preserves hand-written `description:` text. This is the *build* half:
  run it once per source, annotate, then probe the specific claims you are about to declare.

Write a one-off script only for a question neither expresses — and then take the four rules
below from these two, because that is what they encode.

## The rule that makes a probe worth running

**A probe that failed and a probe that came back empty look identical from the outside.**
A helper that returns an empty list for both a clean "no matches" and a rejected request will
clear every candidate you feed it, and it will do so on the one step that could have
disqualified them. So:

* **Always include a control** — the same request with the interesting predicate removed. If
  the control is also empty, the probe established nothing about the predicate; it established
  that the window, the index or the credentials are wrong.
* **Ask through the engine's own retriever**, not a hand-rolled request. A hand-rolled call
  gets the headers, the encoding or the query language subtly wrong, fails for reasons that
  have nothing to do with the question, and reports "unsettled".
* **Separate the failure channel from the empty channel.** Let the exception surface, or
  return a distinguishable sentinel. Never collapse both into `[]`.
* **State in the docstring what each outcome would mean** *before* running it. A probe whose
  result you can interpret either way afterwards has measured nothing.

## What to probe, and the trap each one closes

### Does the column exist in the rows? (`leaves`, then `population`)

A generated schema is *sampled or projected*, so a leaf on the schema is not necessarily a
leaf in the row — and a leaf that is present is not necessarily populated. `population`
returns four numbers because they disagree: a column that is 100% non-null and 100% blank
reads as fully populated to any single check, and a column holding ONE value is populated and
useless (a constant OR-ed into an evidence group satisfies the group by itself). Two further
traps:

* **An index or table family holds a set of shapes, not one shape.** A flat sample can read
  only the oldest member of a rolling family, so a column that exists there and nowhere else
  becomes a filter that silently excludes every current row. Sample *per member*, and check
  the newest and oldest separately.
* **A confirmed path can sit outside the projection.** A leaf that exists in the store may be
  absent from the rows a given query returns, because the projection selects a subset and
  nested structures are flattened differently per route. Confirm the path in *the rows the run
  will actually see*.
* **A whole table may be scoped to one sub-population** — one organisational entity, one
  time box. Probe the spread of the scoping attribute before binding the source at all; a
  table that only ever holds one slice cannot answer a question about the estate.

### Does the value match what the column stores? (`values`, `spread`)

This is where the expensive failures live, because each one is a *valid predicate matching
nothing* — well-formed, plausible, silently empty. Probe each separately:

* **Namespace.** The same subject may have an identifier in one system and a login in
  another, both labelled with the same entity type. Querying one as the other matches nothing.
  Probe the identifier's namespace and the scope *as two separate questions* — a wrong
  namespace and a wrong scope both return zero, and fixing the wrong one leaves you back where
  you started.
* **Precision.** A long form of an identifier is a valid predicate against a column that
  stores only its core. Which precision a column stores is declared nowhere; measure it.
* **Position.** A short code may be a *segment* of a longer stored value rather than the whole
  of it. Establish where in the value it sits before choosing between equality and a
  positional pattern.
* **Numeric representation.** A money amount read from prose cannot be used as an equality
  predicate: the store holds a float whose decimal expansion differs from the printed figure.
  Probe the stored value; then match on a range, a rounded projection, or not at all. This
  one stays latent for as long as the predicate sits in an optional clause, and appears the
  day it is correctly promoted to a required one.
* **Masks and wildcards.** A value that already contains the source's own wildcard characters
  is not a literal, and a "masked" value and an unbounded pattern can spell the same thing.
  Decompose the shape and count what each form actually matches. If nearly every stored value
  ends in the same pattern, the pattern is not the discriminator — the anchor is.
* **Abbreviations and codes: probe them, never read them.** A short code's meaning is a
  measurement, not an inference from its letters. Codes that look like one taxonomy routinely
  turn out to be another, and one that looks specific may lump two distinct populations
  together — which makes any check built on it one-directional.

### Is it a discriminator? (`selectivity`, `pair`)

**Measure over the population, never against the record that raised the alert.** An attribute
that matches the alert *and* matches half of everything else discriminates nothing; used as a
condition it produces a confident finding on every case. The measurement is a base rate: how
many of the population carry this value, not whether this one does.

Two consequences:

* A **cohort** question ("does this subject appear elsewhere") is answered by one query scoped
  to the shared attribute, not by a per-subject query joined afterwards. The scoped query is
  cheaper and it is the only form that can see the cohort at all.
* A **composite key** must be conjunctive. Two key parts OR-ed together return every row that
  matches *either*, which for a widely-reused part is the whole population. If a key has parts,
  the predicate has an `AND`.

### What does it cost? (`cost`)

An unbounded scan is not a performance note; it is a future timeout, and a timeout on a
decisive source turns a verdict into INSUFFICIENT DATA. Probe:

* the **partition layout** as the backend reports it, and how many partitions the intended
  window touches;
* the **elapsed time** of the real query shape, and compare it against the budget the source
  will run under. Record the *margin*, not just the number: an accepted baseline finishing
  just under the cap means the next slowdown converts every condition to `unknown`.
* the **row count against the cap.** A result at the cap is truncated, and a truncated count
  is a floor, not a total. Probing a narrow window and a wide one shows whether truncation is
  in play — the wide window returning fewer distinct subjects than the narrow one is the
  signature.

## Recording a probe's result

The probe is disposable; its number is not. Put the number where the declaration is, as a
comment, with the date and the population it was measured over. A threshold with no recorded
provenance is a threshold nobody dares change and nobody re-measures — and a prose example of
a computed value (a window boundary, an epoch literal) goes stale silently and costs a source
every row it should have returned.
