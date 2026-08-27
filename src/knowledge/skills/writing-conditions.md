---
name: writing-conditions
title: "Writing conditions: polarity, decisiveness, and the checks that cannot fire"
when: "Editing a ruleset or a shared check — adding a condition, changing weighting, importing a mechanic."
always: false
triggers:
  - condition
  - rule
  - rules
  - ruleset
  - verdict
  - check
  - shared check
  - polarity
  - decisive
  - exclusion
  - indicator
  - label
  - threshold
  - kind
  - use case
  - playbook
  - gate
  - rollup
  - scope_discovery
  - as_of
---

# Writing conditions

One condition per decision a human makes in the procedure. Not one per available column: a
condition that no step of the procedure depends on still contributes `unknown` to the rollup,
and `unknown` is not free.

## The three ways a condition silently never fires

Each of these produces a condition that reads exactly like a source that returned nothing.

1. **An unrecognised `kind`.** Only the kinds the engine dispatches on are evaluated; any other
   value falls through and the condition is *never evaluated at all*. Ask the pack for the list
   rather than assuming a plausible name exists.
2. **An unknown key.** The model layer **silently drops** a key it does not know. A declaration
   the engine does not read does nothing, and nothing reports it. If a key is not in the pack's
   own vocabulary, writing it is the same as writing a comment.
3. **A field path that exists on no schema** — or exists on the schema but outside the
   projection. The source is retrieved, the rows are real, and the leaf is not there. The
   checker reports this as a *warning* (a schema inventory is sampled, so it cannot be an
   error), which means it is on you to read the warning.

A fourth, subtler one: **the kind must match the data's shape.** A declaration that names paths
holding a *count* or a *collection* concludes presence by counting leaves underneath them. A
**boolean** holds neither, contributes no countable leaf, and so reads as a path that never
arrived — the check returns `unknown` on every row, *including* the rows where the flag is
plainly true. A boolean needs the flag kind.

## Every number the procedure chose must be declared

The engine supplies no default bound. A count needs its maximum as a whole number; a time gap
needs a window in the engine's own grammar (`90m`, `4h`, `2d` — not a bare number, not a prose
duration, not an ISO period); a ruleset with corroborating non-decisive indicators needs its
indicator threshold. Omit one and the check reports `unknown` and the checker errors.

This is not pedantry about defaults. When these had fallbacks, the invented figure was printed
as *the procedure's own finding* — a bound nothing in the pack states, quoted in a report as
though the procedure had chosen it — and a voting rule nobody wrote reached the fraud label.

`max: 0` is a perfectly good declared bound ("nobody other than the subject") and is never read
as unset. An indicator threshold of `0` is rejected: zero indicators already satisfy it, so
every subject would be accused.

## Polarity, decisiveness, and the label

These three are independent, and confusing any two of them inverts a verdict while every row of
the condition table still reads correctly.

* **`polarity` decides which way a FAIL argues.** A decisive *indicator* FAIL reaches the
  positive label; a decisive *exclusion* FAIL reaches the negative one. Without any indicator
  polarity at all, a ruleset can only ever reach the false-positive label or a textbook
  fingerprint — a non-textbook fraud is *unreachable*, which was a real defect found only when
  a live case the engine cleared was called fraud by a human.
* **`decisive` decides only whether a result settles the matter.** Prefer **`decisive_on:
  [fail]`**: `decisive: true` alone also lets an `unknown` force INSUFFICIENT DATA, which is
  wrong for a check whose FAIL is conclusive but whose *absence of evidence is ordinary*.
  Without that asymmetry every ordinary subject reads as insufficient.
* **A `label` is the REQUIREMENT, not the finding.** For an exclusion a FAIL *negates* the
  label, so printing it states the opposite of what was found. The engine assembles the note
  from the fail-side wording, never from the label. A fraud indicator's label is expected to
  state its finding, so a FAIL affirms it and it prints unchanged — the asymmetry is the
  polarity, not an inconsistency.

**That expectation is an obligation on the pack, and nothing enforces it**, because the engine
cannot negate prose. A live report once read `positive fraud indicator(s) — <X> was NOT changed
on this record` with the whole suite green: an exclusion-phrased label imported under indicator
polarity.

### Categorical beats heuristic

`exclusion_kind: categorical` rests on an **attributed fact** — who acted, and when — and cannot
be outvoted by indicators: once such a fact is on the record the indicators lose their
*meaning*, not merely their weight. A heuristic exclusion (an inference about the subject's
shape) can rightly be outvoted. Measured: an automated actor beats three behavioural indicators
at once, because identity evidence is prior to behavioural evidence.

### Name every rollup label, including the out-of-scope one

A `gate:` condition is asked first and a FAIL exits *without adjudicating*. If the out-of-scope
label is missing, that exit prints as the false-positive label — claiming the case was examined
and cleared when the procedure in fact declined to look.

## Importing a shared check

A shared check is a library of **mechanics**; the **weighting** belongs to whichever ruleset
imports it. The split is not stylistic — it is what lets two procedures use one mechanic with
opposite meanings.

* **Library holds:** the kind, the field paths, and a number the *data* dictates.
* **Importer holds:** decisive / decisive-on, polarity, exclusion kind, report grouping, order,
  gate, and any number the *procedure* chose.
* **`label` is the library's only while the importer keeps the polarity the library assumed.**
  Importing across a polarity means **overriding `label` so that the failing side reads true**.
* **The override is a ONE-LEVEL merge, and a list REPLACES.** A deep merge would union the
  library's paths with the importer's and select a leaf that does not exist, failing the whole
  query.
* **Check the kind actually reads the key you are overriding.** Some finding-wording keys are
  honoured by only a few evaluators; declared on a kind that ignores them, the override is a
  silent no-op and the wording must ride on a different key.
* **An unresolvable import RAISES at load** — deliberately the one fatal path in pack loading,
  because a dropped condition is indistinguishable from a source that returned no rows.
* **Promote to `shared/` only when a second use case needs it.** A fact about the *data* is
  shared; a judgement about the *procedure* stays with the use case that made it.

## Which procedure adjudicates is decided by prose

There is no ruleset-selection mechanism of its own. A playbook is matched, and the matched
playbook's use case selects the ruleset. So **a playbook title is not decoration — it is the
discriminating vocabulary of a procedure**, and adding a competing one can silently move an
existing procedure's incidents onto it. The engine cannot detect that: the losing procedure's
conditions still resolve against real rows and return a *confident wrong verdict* — one live run
returned INSUFFICIENT DATA with eighteen `unknown` conditions and a decisive line naming an
attribute the incident never had, with every stage green.

Therefore: when you add or rename a use case, **measure selection over a corpus of real
summaries** before committing the title. Score the candidate title without editing the pack,
check whether any existing procedure's incidents flip, and only then commit.

## Reading a subject over time

An append-per-change source keeps growing *after* the alert, so read whole it adjudicates the
responder's containment as the subject's conduct. The ruleset declares which sources are version
logs, and a row is excluded only on the **intersection**: later than the incident **and** written
by an identity the incident did not name. Not time alone — an exculpatory change written minutes
late *by the alerted actor* is that actor's conduct continuing, and cutting on time alone deletes
it. Either half missing and the rule abstains. The narrowing applies to the **conditions only**;
the chronology and the scope sweep read the chain whole.

## Two traps in absence and set checks

* **An absence check ORs across array siblings.** If the path resolves to an array *per row*, a
  check that "no member has value X" is satisfied by any sibling that does not, so a decisive
  exclusion can clear a case the array plainly contradicts. Establish whether the path is
  per-row or per-member before writing an absence check on it.
* **An alert's own facts are not evidence against each other.** Where a report declares a set of
  values, comparing each member against its confirmed siblings reports every member as a
  mismatch with itself. A set-valued declaration is arithmetic over the *declared list*, bounded
  explicitly — not a pairwise comparison.

## When emptiness means "not this subject" and when it means "we do not know"

A drop list has two possible meanings and the pack must say which: *these rows are not this
subject* (so the survivors decide) or *the meaning of these rows is not established* (so the
clear is withheld). The same list of patterns under the two readings produces opposite verdicts.
