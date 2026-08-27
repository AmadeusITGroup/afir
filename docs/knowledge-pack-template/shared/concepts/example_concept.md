---
concept_id: example_concept
title: Example shared concept — what a concept doc is for
---

# Example shared concept

> **TEMPLATE.** Replace the whole file. What matters is the *shape*: a shared
> concept doc explains something about the **data** — what a field means, where it lives,
> what it does *not* say, and what has been measured about it. Delete this paragraph.

## Why this file is in `shared/`, not under a use case

`shared/concepts/` is loaded **without a `use_case` tag**, and that is the entire mechanism.
An untagged concept is:

- retrievable by **every** use case's RAG (`concepts_for` unions the untagged shared set with
  the requesting use case's own), and
- nameable by **any** ruleset's `case_builder.concepts` list, exactly like a local id.

Tag it (by putting it under `use_cases/<name>/concepts/`) and you have made a claim of
ownership that the loader then enforces by hiding it from everybody else. So the placement
rule follows from what the document *says*:

| The doc says… | Put it in | Because |
|---|---|---|
| what a field/element/record **is**, where it lives, its traps | `shared/concepts/` | true for every procedure reading it |
| why **this** procedure treats that fact as decisive | `use_cases/<name>/concepts/` | a judgement, not a property of the data |

A concept that starts life inside one use case and turns out to be about the data should be
**promoted** to `shared/` — that is the normal direction of travel, and the reason the first
use case wrote it does not make it that use case's property.

## What to write

Four things, in roughly this order. The first three are what an investigator asks; the fourth
is what stops the next author repeating a mistake.

1. **The definition, in the domain's own words.** Not a paraphrase of the field name.
2. **Where the fact lives** — the concrete paths, including the shape (a boolean? a counter? an
   `array<struct>` that must be descended?). Name the source.
3. **What it does NOT say.** The most valuable sentence in most concept docs. A field that
   looks like an authorisation but is really an access log; a flag that is unset by default so
   its absence means nothing; a timestamp that is a *deadline* rather than an event time.
4. **What has been measured.** Row counts, coverage percentages, a query that returned zero
   rows and why. A measurement outranks a plausible reading, and writing it down is what makes
   it outrank one for the *next* reader.

## Cross-links

Link related concepts with `[[their_concept_id]]` — for example [[example_concept]] (itself,
here, only because the template ships one). The RAG layer follows these when assembling
context, and an id that does not exist yet is not an error: it marks something worth writing.

## What NOT to write here

- **Weighting.** "This is decisive" is a ruleset's `decisive: true`, not prose. Prose cannot
  be enforced, and a doc that asserts a verdict-level fact the ruleset does not encode is a
  disagreement nobody will notice.
- **Anything that must be *enforced*.** If a rule has to hold on every run, it belongs in the
  source catalog (`never_filter`, `require_all_entities`, `identity_keys`, …) or in the
  ruleset. A prompt instruction cannot be relied on to beat another prompt instruction.
- **Secrets, real identifiers, real case data.** Concept docs are committed. Resolved
  investigations belong in `use_cases/<name>/cases/`, which real packs gitignore.
