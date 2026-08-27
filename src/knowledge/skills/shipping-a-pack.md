---
name: shipping-a-pack
title: "Shipping a pack change: the checker, the guards, and when the engine is the problem"
when: "Finishing a change — validating, adding a new use case or a new pack, or considering an engine edit."
always: false
triggers:
  - validate
  - validation
  - checker
  - pack_validate
  - ship
  - new pack
  - new use case
  - scaffold
  - vocabulary
  - neutral
  - coverage
  - engine
  - restart
  - checklist
---

# Shipping a pack change

## Run the checker, and read its warnings as warnings

An **error** means the pack does not work, or works while lying. A **warning** means something
the engine cannot decide mechanically — most importantly a field path that no schema records,
which is a warning precisely because a schema inventory is *sampled* and a projection may alias a
column no table ever had. Warnings are therefore the interesting output, not noise: they are the
class of defect that cannot be automated away, handed to the one reader who can settle it.

Two properties of the checker worth knowing when you read its output:

* It reports **how much it was able to check**. A pack that documents no schemas produces few
  findings, and without that count a pack nobody checked reads like a pack that passed.
* It does not use the pack loader to read the pack, because the loader is part of what is being
  checked: a loader that turns a parse error into an empty document would describe a *broken*
  pack as an *empty* one.

## The guards a pack carries, and the one it contributes

* **Behaviour on an unseen domain.** The engine is proved generic by running it against a
  deliberately foreign pack. That is a behavioural check and it catches hardcoded values fast.
* **Neutrality of the template** an author copies.
* **The pack's own hierarchy test** — asserted from the *outside*: no inlined mechanics in a
  ruleset, no import restating a field path, no library entry carrying weighting, shared concepts
  reachable from another use case. A test that reads the YAML back only proves somebody typed the
  block.
* **`domain_vocabulary.yaml` — the pack's own vocabulary boundary.** This is the pack's
  contribution to the engine's neutrality: a scan reads every installed pack's list and fails on
  a hit anywhere in the engine, so *installing a pack installs its neutrality guarantee*.

Three rules about that vocabulary file, all learned the hard way:

* **Curate it; do not derive it from the glossary.** Derivation is wrong in both directions. It
  flags words the engine legitimately owns (the generic actor, an abbreviation that collides with
  a product name or a status code), and it *misses* what actually leaks — because a leak is prose,
  and prose uses nouns no glossary key spells.
* **Record the omissions and why.** A word deliberately left out, with its measured hit count and
  the reason, is a decision a reader can see. Rediscovering it is how these scans get weakened,
  and the first response to a noisy check is always to weaken it.
* **A word that can never fire is worse than an omission**, because it reads as covered. Check
  that the matching rules can actually reach the spelling you listed.

## Coverage: a declared key with no shared test is a place to look

The measure of "generic" is not an assertion, it is a matrix: which pack-facing keys have a test
that exercises them independently of any one domain. The rule that matrix earned is worth
applying to anything you add: **a pack key with no shared test is a place to look** — both keys
that once had zero coverage held a latent defect. If you add a declarable key, add the test that
would fail if the engine stopped reading it.

## When the engine is the problem — and it usually is not

The engine must work for every pack unchanged. Use-case specificity lives in the pack. So before
concluding that a change needs engine code:

1. **Check whether the declaration already exists.** The mechanism is often present and simply
   not declared. A comment claiming the engine "has no way to" do something has been wrong
   before — it hid a working key while a real pivot reported "no data" over a hundred-odd rows.
2. **State the defect in domain-neutral terms.** If it cannot be stated without naming your
   domain, it is a pack change.
3. **If it is a real, generic defect, fix it generically and add the shared test.** A pack field
   that no-ops on some backends is exactly the bug it was meant to prevent.

## Adding a use case

Beyond the ruleset itself, two things are easy to forget and both are silent:

* **Measure procedure selection over a corpus of real incident summaries** before committing the
  playbook title. A new title can move an existing procedure's incidents onto it, and the loser's
  conditions still resolve against real rows — a confident wrong verdict rather than no verdict.
* **Name the report vocabulary for the new use case.** Resolution merges a mapping per sub-key
  but a **list replaces**, so a scoped wording block must inherit the slots it does not restate.
  Every slot falls back to a generic sentence, so the loss is invisible: it reads as a pack that
  never declared the wording.

## Scaffolding a new pack

Start from the template, but the vocabulary boundary is not copyable: the template's placeholder
words collide with its own example ruleset, so copying them makes the *template's* own test fail —
and a pack with no vocabulary file at all takes the whole suite down. Declare your own words.

Replace the template's single placeholder condition with **one real condition and run it** before
writing twelve. Twelve conditions over unmeasured bindings all report `unknown`, and the run tells
you nothing about which of the twelve is wrong.

## After the change

* **There is no hot reload.** Every pack mutation needs a restart to take effect; a mutating
  response says so.
* **Re-run the checker**, then the pack's own tests, then the acceptance replay.
* **Record every number you measured** beside the declaration that uses it, with its date and the
  population it was measured over.

## Checklist

- [ ] Every binding, threshold and filter guarantee is backed by a measurement, recorded as a
      comment beside it.
- [ ] Every condition's kind is one the engine dispatches on, and matches the data's shape.
- [ ] Every number the procedure chose is declared; nothing relies on a default.
- [ ] Every label reads true on the side that fails, for its polarity.
- [ ] Every rollup label is named, including the out-of-scope one.
- [ ] Filter guarantees use the enforced keys, not prose.
- [ ] Field paths resolve on a schema, and are inside the projection.
- [ ] The checker's errors are zero and its warnings are read.
- [ ] One fixture that clears and one that flags, asserting on the condition, not the verdict.
- [ ] Anything you could not determine is written down as a question rather than a declaration.
