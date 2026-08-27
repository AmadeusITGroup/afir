# The knowledge-pack template — copy this to start a pack

Every file a knowledge pack may contain, present, **live, and exhaustively commented**. It
loads cleanly, reaches no verdict, and names no domain.

```bash
cp -r docs/knowledge-pack-template knowledge/mydomain
# then set knowledge.pack_dir: "mydomain" in config/main_config.yaml
```

**Why this sits under `docs/` and not under `knowledge/`.** It is live YAML that a test really
loads, so it looks like a pack — but it is documentation. Anything inside `knowledge/` reads as a
pack you may *select*, and a `pack_dir:` aimed at a template that loads, retrieves nothing and
decides nothing is the worst kind of valid configuration: the run succeeds, every condition is
`unknown`, and the report is indistinguishable from an investigation that found nothing. Keeping
it out of `knowledge/` removes that choice instead of documenting against it.

`pytest tests/test_pack_template.py` passes as shipped, so your copy has a green baseline to
diff against the moment an edit breaks something.

## Why it is live rather than commented out

A pack of commented-out YAML cannot be loaded, so it cannot be *wrong* in a way anybody
notices. Every file here parses and every declaration is real, which buys three things:

- **A working baseline.** Your first edit either keeps the pack loading or it doesn't, and you
  find out in under a second.
- **The documented key names are provably real.** `tests/test_pack_template.py` scrapes every
  key this template documents and asserts the engine reads it. That check exists because of a
  measured defect: `trigger:` was declared by three shipped rulesets, carried a comment saying
  it made the report state the real trigger, and *nothing in `src/` read it*. The block was
  inert, so the report defect it was written to fix was still live — while the pack looked
  correct. Documentation drifts from an interface silently, and the reader is exactly the
  person who cannot tell.
- **No verdict, on purpose.** The ruleset ships one `stub` condition, which evaluates to
  `unknown` with a stated reason and touches no data. The file is provably live without the
  template asserting an outcome it cannot know.

## The order to fill it in

Each step leaves the pack loadable, so you can run after every one.

| # | Do this | File | Why this order |
|---|---|---|---|
| 1 | Name your entities | `entity_glossary.yaml` | Everything downstream references entity **types**. Get `value_forms` right here if one entity has several non-interchangeable surface forms — no later prose can restore a distinction the data model discarded. |
| 2 | Describe one source | `source_catalog.yaml` | Retrieval works with one source. Declare the *enforced* guarantees (`never_filter`, `require_all_entities`, `identity_keys`, `partition_columns`, `epoch_time_columns`) now — they are checked after generation, not prompted, so they are the only pack statements that cannot be talked out of. |
| 3 | Write the playbook | `use_cases/<name>/playbooks/*.md` | Prose first. Writing what to retrieve and how to join it is how you discover what the ruleset needs, and the `correlation:` frontmatter is a functional input, not decoration. |
| 4 | One condition, end to end | `use_cases/<name>/rules.yaml` | Prove the plumbing before writing twelve conditions. Replace the `stub`. |
| 5 | The scope gate, if any | same | It is answered *first* and a FAIL exits the case, so it changes what every other condition means. |
| 6 | Weighting | same | `decisive`, `polarity`, `exclusion_kind` — the part that needs real care. See below. |
| 7 | Reporting + notification | `reporting.yaml`, `notification:` | The artifact surface. Every slot is optional; the engine's default is a complete, procedure-free sentence. |
| 8 | Promote what is shared | `shared/checks/`, `shared/concepts/` | Once a *second* use case would ask the same question of the same data. Not before — this is a de-duplication mechanism, not a style rule. |
| 9 | List your own vocabulary | `domain_vocabulary.yaml` | Now, not at the start: you cannot list your domain's words before writing the pack, and by this point every one of them has been typed at least once. It is what makes the neutrality scan enforce the engine/pack boundary *for your domain*. |
| 10 | Your own tests | `tests/` | The generic tests prove the engine. Only you can prove your pack. |

## What files exist and what each is for

**Shared (pack root) — knowledge about the DATA**

| File | Holds | Notes |
|---|---|---|
| `entity_glossary.yaml` | what to recognise in incident text | Also the **closed** `abbreviations:` map, which forbids the LLM guessing an expansion. |
| `source_catalog.yaml` | where data is, how to query it *safely* | The five enforced guarantees live here, plus `not_answered_by` / `selection_guidance` (a `description` answers neither question the planner has). |
| `schemas/<source>.yaml` | the **complete** field inventory | Arrays descended, `explode_path` per leaf. RAG-only — half generated (structure), half hand-written (`description:`), and that split is the point. |
| `data/<name>.yaml` | reference lookup tables | Keyed by **file stem**, flattened globally. Stems must be unique across the whole pack. |
| `shared/concepts/*.md` | prose about the data | Loaded **untagged**, which is the mechanism: every use case can retrieve it. |
| `shared/checks/*.yaml` | importable check **mechanics** | File stem is the namespace. Mechanics only, never weighting. |
| `reporting.yaml` | domain-wide report vocabulary | Merged **per slot** with the scoped file below. |
| `domain_vocabulary.yaml` | **your words, so the engine can be proven not to speak them** | Read by no *pipeline* code — read by `validate_pack`, which scans `src/` and `config/templates/` for these words and reports a hit as a `neutrality-collision` error. Filling it in is how you install that guarantee; the hard part is deliberately *omitting* the ordinary English the engine needs for its own job, with the reason recorded. |
| `RETRIEVAL.md`, `BACKENDS.md` *(optional)* | pack-wide notes for the humans | Not ingested. Worth adding once you have enough sources that their query shapes and partition layouts need explaining in one place, and once the backend keys need listing by env-var name. |

**Specific (`use_cases/<name>/`) — knowledge about the PROCEDURE**

| File | Holds | Notes |
|---|---|---|
| `rules.yaml` | the verdict ruleset | The biggest file. All 14 condition kinds documented inline. |
| `playbooks/*.md` | how to investigate | Deduped by `playbook_id`; the `correlation:` block is layer-1 join resolution. |
| `concepts/*.md` | prose about the procedure | Tagged, so scoped to this use case. |
| `cases/*.md` | resolved investigations | **Gitignore these in a real pack** — add the entry *before* the first real case. |
| `data/*.yaml` | this procedure's lookups | Same global flattening; prefix the stem. |
| `reporting.yaml` | this procedure's wording | Overrides the root **per slot**; a clause number belongs here, never at the root. |
| `VERDICT.md` *(optional)* | what THIS procedure decided, and why | Not ingested — documentation beside the ruleset it explains. The place for your clause-number wiring, which exclusions you made categorical, and the incident history behind each. **Cite `shared/concepts/` with `[[concept_id]]`, never restate them:** a measurement copied into two files is one that will be corrected in only one of them. |

**Prose obeys the same split as YAML, and the test is one question: would this still be true if a
different procedure asked it?** A fact about the data is shared (`shared/concepts/`); a judgement a
procedure made about that fact belongs with the procedure. Getting this wrong is *not* caught by a
test, and it decays in a specific direction: while you have only one use case, "the pack's verdict
notes" and "this procedure's verdict notes" name the same set, so the root looks like a fine home
right up until a second ruleset has to choose between duplicating that file and breaking the
convention.

One budget to know before you "share" prose by promoting it to `concepts/`: a concept named in a
ruleset's `case_builder.concepts` costs a **flat 500 chars** of the brief injected into the report
LLM. Appending to a concept that already owns the subject is free; a new declared file is not.

## The three things newcomers get wrong

**1. A `label` is the REQUIREMENT, not the finding.** For an exclusion a FAIL *negates* the
label, so a report printing the label states the opposite of what was found. A live report read
`DECISIVE CONDITION: <actor> is not automated (<flag path>=True)` under a verdict resting on the
actor being an automation — the label and the evidence beside it contradicting each other, with
no way for the reader to tell which half to trust. Write `fail_detail`. (A fraud *indicator*'s
label already states its finding, so a FAIL affirms it and it prints unchanged; the asymmetry is
the polarity, not an inconsistency.)

**2. `polarity` decides which way a FAIL argues; `decisive` only decides whether it settles.** A
decisive *indicator* FAIL reaches the positive label; a decisive *exclusion* FAIL reaches the
negative one. Confusing them inverts the verdict while every row in the condition table still
reads correctly. And `decisive_on: [fail]` — which most real checks want — makes decisiveness
asymmetric: a check whose FAIL is conclusive but whose absence of evidence is *ordinary* would
otherwise force INSUFFICIENT DATA on every ordinary subject.

**3. Zero rows is not automatically a data problem.** A source can answer by being *empty* — no
row on an automation list means the actor is human, which is a finding. Declare
`zero_rows: {health_weight, meaning}` on the source, or the health scorer counts your answer as
a defect.

## Maintaining a pack: re-pointing a source

Packs rot in one specific place. A backend moves — a feed migrates to a new index, a table is
replaced by a view — and the natural edit is to change `indices:`/`tables:` and stop. **Do not
stop there: re-measure every `entity_bindings` field against the new target.**

The reason is that a stale field name **fails at no layer**. It parses as YAML, the generator
turns it into a predicate, the predicate matches nothing, and the query still returns rows — so
the stage reports success while the filter it was supposed to apply is absent. In one measured
case a repoint that kept its bindings produced a query whose only predicates were a type
discriminator and a date range: every alert that day, the subject's own somewhere in the page, and
a report built on it. Three sources over one index had the same defect, and the one shared by
*every* use case had it longest — no single use case owned it.

Two habits contain it:

- **Measure existence over ALL rows/documents, not a sample**, and regenerate the source's schema
  inventory afterwards. A generated inventory is the only artifact that *shows* a field's
  disappearance; a catalog cannot check itself.
- **Prefer a second `status: legacy` entry over one entry listing both targets.** A migrated feed
  and its predecessor rarely share a field shape, so two entries keep the history reachable
  without either binding list being wrong about the other's target.

And when judging whether a source is still current: **a row count cannot tell you.** A table that
still *holds* data is not a table still *receiving* it, and the two are indistinguishable by
`COUNT(*)`. Compare `MAX(<time column>)` and a rolling recent window. One legacy index in a real
pack held 700 documents and had received nothing in five months, which is exactly the profile of a
source that looks healthy right up to the moment an investigation depends on it.

Finally, when the new target *is* shared with another source (one index, several detectors' rows),
remember that a `populated_pct` measured over the whole target is not a fact about your slice.
Measure per slice, or a field that is 0% for you reads as populated.

## The failure mode this pack is built to prevent

Almost every hard-won comment in these files traces to one shape: **a check that cannot fire is
indistinguishable, in the report, from a check that had nothing to find.** Zero rows → decisive
conditions `unknown` → INSUFFICIENT DATA, and the retrieval stage still reports success.

It has arrived by every route: an unbounded partition scan (slow, then empty, then
"insufficient"); an epoch literal in a prose hint that was off by exactly one year; a query
whose filter deleted its own evidence; an absence check naming an array's *leaf* instead of its
container, so the very case it existed to catch reported `unknown` on every row; a boolean read
with a counting check; a timeout that fired on the system of record and thereby *decided* the
verdict rather than degrading it.

When a comment in these files sounds emphatic, that is why. Read the ones next to the key you
are about to fill in.

## Where the detail lives

- [`../../knowledge/mock_domain/README.md`](../../knowledge/mock_domain/README.md) — the small
  **working pack** that ships, in an invented domain. Read it when you want to see a real
  declaration end to end rather than a template. It is also what the generic engine tests assert
  against, so every key here is exercised there.
- [`../architecture/knowledge-pack-authoring.md`](../architecture/knowledge-pack-authoring.md) —
  the full authoring guide: file surface, RAG wiring, the tests.
- [`../architecture/knowledge.md`](../architecture/knowledge.md) — the loader and the RAG layer.
- `../architecture/verdict-engine.md` — every condition kind's exact semantics and parameters.
- [`../architecture/retrieval.md`](../architecture/retrieval.md) — how the enforced query
  guarantees actually work.
