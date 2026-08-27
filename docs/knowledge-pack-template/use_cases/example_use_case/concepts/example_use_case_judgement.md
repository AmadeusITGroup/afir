---
concept_id: example_use_case_judgement
title: Example use-case-scoped concept — a judgement, not a property of the data
---

# Example use-case-scoped concept

> **TEMPLATE.** Replace the whole file. Delete this paragraph.

## Why this one is here and not in `shared/concepts/`

Compare with [[example_concept]], which sits in `shared/`. The difference is not importance and
not length — it is **what the document is a statement about**:

- A **shared** concept says what a field or element **is**: where it lives, what shape it has,
  what it does not say. That is a property of the data, so it is true for every procedure that
  reads it. Loaded untagged, retrievable by every use case.
- A **use-case** concept says why **this** procedure treats that fact a particular way. That is
  a judgement, and another procedure reading the same field may reach a different one. Loaded
  tagged with the use-case name, so `concepts_for` scopes it to this use case.

The practical test: if a second use case would want to read this document unchanged, it belongs
in `shared/`. If a second use case might legitimately **disagree** with it, it belongs here.

This split mirrors the one in `shared/checks/` exactly — mechanics are shared, weighting is the
importer's — and for the same reason. A concept that turns out to be about the data should be
**promoted** to `shared/`; the fact that this use case wrote it first does not make it this use
case's property.

## What to write

The reasoning a reviewer needs in order to accept the verdict:

1. **Which condition this justifies**, by its ruleset `id`. The link back is what makes the
   document checkable — a rationale nobody can tie to a rule is a rationale nobody can falsify.
2. **Why the fact is weighted the way it is.** Why this exclusion is *categorical* rather than
   heuristic (it rests on an attributed fact — who acted, and when — rather than on an inference
   about the subject's shape); why this indicator is not decisive alone.
3. **What the procedure requires**, cited to the procedure. Note the artifact only carries this
   when it *is* in the pack: prose in `src/` mis-cites every other use case.
4. **What would change the judgement.** A new data source, a measured false-positive rate, a
   revised procedure version.

Short YAML comments in the ruleset are the right place for a one-line "why this weight"; this
file is for the reasoning that needs a paragraph and that a report should be able to narrate
from.

## What NOT to write here

- **Facts about the data.** Those go in `shared/` (or in a `schemas/` `description:`) so every
  use case gets them. Writing them here is how two use cases end up with two copies of a field
  path that drift the first time a column moves.
- **Anything that must be enforced.** Enforcement lives in the catalog's declared guarantees or
  in the ruleset. A prompt instruction cannot be relied on to beat another prompt instruction.
