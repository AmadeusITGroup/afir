---
# =============================================================================
# PACK TEMPLATE — use_cases/<name>/cases/*.md — A RESOLVED INVESTIGATION.
#
# WHAT THIS FOLDER IS FOR. A case is a CLOSED investigation with a KNOWN outcome,
# ingested into the RAG corpus so a later similar incident retrieves it. It is the
# only document type in the pack that carries a ground truth: everything else says
# what to look for, a case says what turned out to be true.
#
# Its value is precedent — "an incident that looked like this was cleared, and
# here is the fact that cleared it" — which is exactly what an engine cannot
# derive from rules. A case whose outcome DISAGREED with the engine is the most
# valuable file in the pack: it is a regression test written in prose, and it
# should be paired with a real one in `tests/`.
#
# ############ REAL CASES ARE NOT COMMITTED ############
# A resolved investigation contains real identifiers — real accounts, real people,
# real transactions. In a real pack `use_cases/*/cases/` is GITIGNORED, per domain,
# and the ignore is by CONTENT rather than by convenience. The template's cases DO
# commit, because this file is fictional and its whole purpose is to document the
# shape.
#
# BEFORE ADDING A DOMAIN'S FIRST REAL CASE: add the gitignore entry FIRST, then
# verify with `git status --ignored`. An ignore added afterwards does not remove
# what is already committed.
# ######################################################
#
# Frontmatter fields the loader retains (all optional except the id, which falls
# back to the file stem): case_id, verdict, subject, decisive_reasons, route,
# resolution, date, scope. Any OTHER key is silently dropped — the loader keeps a
# closed list, so a key it does not know reads as retained and is not.
# Bare YAML dates parse as `datetime.date`, which is not
# JSON-serialisable — the loader coerces them to ISO strings so the RAG dump does
# not break, but writing them quoted is clearer.
# =============================================================================

case_id: EXAMPLE-CASE-001
title: "Example resolved case — cleared by manual review"
# The FINAL, HUMAN-CONFIRMED outcome, in the ruleset's own label vocabulary. Not the engine's
# output: when the two differed, that difference is the reason to keep the file.
verdict: "EXAMPLE NEGATIVE"
# The subject entity value the verdict was rendered for. Fictional here.
subject: EX00000001
# WHY it resolved that way — the short list a later reader needs, and the field a similar
# incident's retrieval matches against.
decisive_reasons:
  - "A manual review was recorded against the record (categorical exclusion)."
  - "The servicing elements were present — the record was normally handled."
# The scope/route the case fell in, when the procedure has jurisdiction limits.
route: "EX001-EX002"
# What was actually DONE. Distinct from the verdict: a confirmed positive may still end in no
# action for a reason worth recording.
resolution: "Closed, no action. Claim paid."
date: "2026-01-15"
# The organisational unit the case sat in, whatever your domain calls one.
scope: EXAMPLE_UNIT_01
---

# Example resolved case — replace this whole body

> **TEMPLATE.** Fictional. Delete this quote and the whole body.

## What the alert said

The trigger verbatim, and what the alert asserted about the subject. Quoting it matters: the
alert is the one input the pipeline did not derive, so a later reader can check whether the
investigation actually reproduced what the alert claimed.

## What the evidence showed

The facts as retrieved, with the paths they came from. Include the ones that pointed the *other*
way — a case file that lists only the facts supporting the outcome cannot be used to calibrate
anything.

## Why it resolved this way

The decisive fact and why it was decisive. If the outcome rested on an **exclusion**, say
whether it was categorical (an attributed fact — who acted, and when) or heuristic (an inference
about the subject's shape), because only the first cannot be outvoted by indicators.

## Where the engine and the reviewer differed

**The most valuable section, when it applies.** If the engine reached a different verdict, say
what it read, what it missed, and what was changed as a result — a ruleset weight, a new
condition, a corrected field path, a source added to the catalog. Then say which test in
`tests/` pins the fix. Recorded only in prose, the same wrong verdict returns the first time
somebody refactors the thing that fixed it.

## What to reuse

The generalisable lesson, stated so it survives without this case's specifics. "A record with
servicing elements present is a handled record" generalises; "EX00000001 was fine" does not.
