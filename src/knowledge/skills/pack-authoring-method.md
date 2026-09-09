---
name: pack-authoring-method
title: "The spine: measure, declare, prove"
when: "Always. Every other skill is a specialisation of this one."
always: true
triggers: []
---

# The spine of every pack change

## The one failure mode everything here defends against

A declaration that is wrong does not raise. It returns **zero rows**. Zero rows makes the
conditions that read that source resolve to `unknown`, `unknown` conditions produce
**INSUFFICIENT DATA**, and INSUFFICIENT DATA is *the same output* the engine produces when
the source genuinely had nothing to say. Every stage reports success. The health score is
green. The report is fluent, well-formed and empty.

So the question to ask of every line you are about to write is not "is this correct?" but:

> **If this line were wrong, what would I see?**

If the answer is "zero rows", "a condition that never fires", "a value that never matches" or
"a number nobody re-measured", the line is not finished until something *outside the pack*
has confirmed it. That confirmation is a probe, a measurement or a test — never a document,
never a reading of a column name, and never the model's own reasoning about what a field
probably holds.

Second-order form of the same trap: an *empty answer* and a **non-answer** are different
facts. A source that timed out, was skipped for missing credentials, or was asked with an
unfilled placeholder did not answer. The engine distinguishes these and says so; a pack
author reading a run's output must too, or a fix gets aimed at the wrong stage.

## The order to build in

Each step exists because the next one cannot be done honestly without it.

1. **Read the procedure** the pack is meant to automate, and write down what a human decides
   at each step. That is the ruleset's skeleton: the decisions, not the data.
2. **Declare the entity types** — what incidents are described *with*. Recognition hints and
   surface forms come from real incident text, not from an idealised example.
3. **Bind the sources** — for each source, what it holds, what it *cannot* answer, when it is
   worth asking, and which column each entity type lands on. **Every binding is measured.**
4. **Generate the schemas** so the field paths a condition names can be checked mechanically.
5. **Write the playbook** — the procedure in prose, and the discriminating vocabulary in its
   title (that title is what selects the ruleset; see `selecting-a-procedure` inside
   `writing-conditions`).
6. **Write the ruleset** — one condition per human decision, weighting stated where the
   procedure made the judgement.
7. **Write the report vocabulary** so the finding reads in the procedure's own words.
8. **Promote to `shared/`** only once a *second* use case needs the same mechanic.
9. **Validate, then prove** — the checker, then the pack's own tests, then a replay of a real
   incident.

Steps 1–3 are where the cost is. A ruleset written over unmeasured bindings is not a draft of
the right thing; it is a set of conditions that will all report `unknown` and look like a data
problem.

## Measure before you declare — the four questions

Before a source, column or number enters the pack, four things must have been established
*against the live system*, because each has produced a silent zero-row run:

* **Does the column exist, in the rows?** A leaf on a generated schema is not a leaf in the
  row: schemas are sampled or projected, and a projection may alias a column the table never
  had. An index or table family may hold *several shapes*, and a naive sample reads only one
  of them — filtering on a column that exists in the oldest shape only returns nothing from
  the newest.
* **Does the value the incident carries match what the column stores?** Same identity, but a
  different surface form, a different precision, a different namespace or a different segment
  of a longer value. Each of these is a *valid predicate that matches nothing* — the most
  expensive shape of wrong, because there is nothing malformed to notice.
* **What is the cardinality, over the population?** A constant column cannot discriminate.
  A "distinguishing" attribute that is on half the population distinguishes nothing. A
  discriminator is measured across the whole population, never confirmed against the one
  record that raised the alert.
* **What does the scan cost?** A query whose partitions are unbounded is not slow, it is a
  future timeout, and a timeout on a decisive source *decides the verdict* rather than
  degrading it.

**All four have shipped tooling — use it rather than writing a scratch script**, because a
hand-rolled request measures a path the run does not take, and the usual `except: return []`
makes a failed probe indistinguishable from a clean empty one:

| Question | How to measure it |
|---|---|
| Does the column exist, in the rows? | `python -m src.knowledge.pack_probe leaves <source> <table>` |
| Is it populated, and does it vary? | `… pack_probe population <source> <table> <column>` |
| What does the column store? | `… pack_probe values` / `… spread` (forms, then lengths) |
| Is it a discriminator? | `… pack_probe selectivity <source> <table> "<predicate>"` |
| What does the scan cost? | `… pack_probe cost <source> <table>` (elapsed vs the budget) |
| Every leaf of every source | `python scripts/generate_source_schemas.py --pack <dir> --measure` |

Both are pack-agnostic and go through the retrievers a real run builds. `pack_probe sources`
first, always: a *declared* source that built no retriever cannot answer anything, and that is
not a fact about the data. Details and the four probe rules in `probing-live-data`.

**If you are running as the pack assistant you can take these measurements yourself**: the
`probe` tool is the same library, one `op` per row of that table, read-only at the seam. The
budget is a handful of measurements for the whole session, so spend it on the value you are
about to write down rather than on orientation — `read_file` and `search` are free. Three
answers are *not* measurements and must never become one: a refusal, a failure and a timeout.
Each of those leaves you exactly where you were without the tool, which is a `questions` entry
naming the measurement to run. The one thing that must not happen is a number you did not
measure appearing in an approved diff, where it is indistinguishable from one you did.

## Prefer the smallest edit

Pack files are heavily commented and use YAML anchors, and both are load-bearing: the comment
is usually where the *measurement behind a number* is recorded, and an anchor's loss makes the
whole document unparseable. A parse failure does not raise either — the loader hands the
engine an **empty document**, so a broken catalogue presents as a pack with no sources and
every check resolves to `unknown`.

Therefore:

* Patch a line range; never rewrite a file whole to change three lines.
* If a comment above your change would become wrong, patch it in the same edit. A comment
  that contradicts the declaration below it is worse than no comment.
* When you add a number, add the measurement that produced it as a comment beside it, with
  the date. A number with no provenance cannot be re-checked, so it is never re-checked.

## Say what you could not determine

The three things a pack author must *not* infer, because being wrong is indistinguishable
from being right until a live run:

* whether a column really holds what its name suggests;
* whether a source is authoritative for a question, or merely mentions it;
* what the procedure intends when its prose is ambiguous.

For each of these the honest output is a question, not a plausible declaration. A guess
inside an approved change is indistinguishable from a finding: the reviewer sees a diff, and
a diff does not say which lines were measured.

## Two rules about the engine

* **The engine is generic, and stays generic.** Domain nouns, values, field paths, thresholds
  and procedure wording live in the pack. If a change seems to need engine code, first check
  whether an existing declaration already does it — the mechanism is often present and simply
  not declared, and a comment claiming "the engine has no way to do this" has been wrong
  before.
* **A fix must generalise.** Matching one expert's conclusion on one incident is not the
  goal; the goal is a declaration that is right on the next incident nobody has seen. A fix
  that reproduces the expected verdict by encoding the answer has removed the measurement
  that would have caught the next case.
