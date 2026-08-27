---
# =============================================================================
# PACK TEMPLATE — use_cases/<name>/playbooks/*.md — THE INVESTIGATION PROCEDURE.
#
# A PLAYBOOK IS THE ONE DOCUMENT THAT SAYS *HOW TO INVESTIGATE*. The ruleset says
# how to ADJUDICATE once the evidence is in hand; the catalog says what each source
# holds; the playbook is the narrative a human would give another human: what the
# alert means, what to pull, in what order, how to join it, what would exonerate.
#
# It is both RAG-ingested prose AND — through the `correlation:` block below — a
# functional input to the correlation stage. That dual role is why the frontmatter
# matters as much as the body.
#
# DEDUPED BY `playbook_id` across the flat root and every use case, so two files
# claiming the same id means one of them is silently discarded. Ids must be unique.
# =============================================================================

# Unique across the pack. The dedupe key.
playbook_id: PB-EXAMPLE-001
title: Example Use Case — What This Playbook Investigates
domain: fraud / example
# The detector name(s) whose alerts this playbook answers. Used for matching.
detectors: [EXAMPLE-DETECTOR]
# The REAL source names (as in source_catalog.yaml), not logical ruleset names.
data_sources: [example_source]
# Verbatim shape of the alert text, which is often how a playbook gets matched to an incident.
typical_alert: "[EXAMPLE] Unusual example activity detected in <example_location>"
related_playbooks: []

# ---- THE FUNCTIONAL JOIN SPEC ------------------------------------------------
# Consumed by the correlation stage as LAYER-1 (authoritative) key resolution. Omit it and the
# stage discovers keys from the data instead, which works, is slower, and occasionally picks a
# DECOY — a field that correlates by coincidence within one incident's rows. Declaring the spec
# is how a domain expert's knowledge outranks a statistical guess.
correlation:
  # The ENTITY TYPES to join on, in priority order.
  keys: [example_entity, example_actor]

  # `same_day` | `within:24h` | `within:15m` | ... — the tolerance for calling two events
  # related. Choose it from the domain's real latency, not from tidiness: too tight silently
  # drops genuine matches and reads as "the source had nothing".
  time_window: within:24h

  # Per source, WHICH column is the event time. Naming the wrong one is the quiet killer here:
  # a column that is a DEADLINE rather than an event instant can legitimately precede the
  # record's own creation, so a window built on it excludes the very rows it should include.
  #
  # DECLARE IT EVEN WITH NO `time_window:` — naming the event and gating the join are
  # independent questions, and this key is read either way (with no window it still decides the
  # chronology's ordering and which columns survive trimming). Left undeclared the engine picks
  # by MEASURED resolution, which can only see precision: a scheduled or deadline column beats a
  # day-granular one that names the real event, and a constant placeholder that carries a
  # time-of-day beats a date. A coarse column naming the right instant is the better answer, and
  # only a declaration can say so.
  time_fields:
    example_source: example_date

  # entity type -> the column that holds it, PER SOURCE. This is the map that makes a join
  # possible across sources that name the same thing differently.
  fields:
    example_source:
      example_entity: example_id
      example_actor: example_actor_id
    # A source that does NOT hold one of the keys must simply OMIT it here. Mapping a key a
    # source does not carry produces a join that always misses, and a join that always misses
    # is reported as "no matching records found" — a finding, not an error. Omission is
    # correct and visible; a wrong mapping is neither.

  # `strict` | `cardinality` | `both` — how aggressively discovered keys are filtered.
  # `cardinality` drops keys whose value is nearly constant across rows (a decoy's signature).
  key_filter: cardinality
---

# Example Use Case — replace this whole body

> **TEMPLATE.** The headings below are the ones that earn their place in a real
> playbook, in the order an investigator needs them. Delete this quote and write the domain's
> real procedure.

## What it means

The allegation in one paragraph, in the domain's own words. State what the pattern IS, and —
just as importantly — **what is not part of it**. "The detector fires on the burst; it is not
amount-based, so a large value alone is not a trigger and a small one is not exculpatory" is
the kind of sentence that stops a report presenting the wrong fact as the core of the case.

## What to retrieve

Numbered, in order, each with **why** and **what it costs**:

1. **The alert record** — first, and authoritative. It carries this incident's values already
   in it, so a fact the alert *states* is ground truth to be **reproduced**, not re-derived. A
   divergence between the alert and the logs is a finding (wrong record selected, stale index),
   never a reason to quietly prefer the logs.
2. **The system of record** — where the decisive facts live. If it is
   `retrieval_class: primary`, say so and say why: a timeout that fires here does not degrade
   the answer, it *decides* it (0 rows → every decisive condition `unknown` → INSUFFICIENT
   DATA, indistinguishable from a source that genuinely held nothing).
3. **Conditional sources** — the ones worth pulling only for a specific question. Name the
   question. These should be selected by the catalog's `selection_guidance`, *not* pinned in
   the ruleset's `sources:` map, or their full scan cost is paid on every unrelated incident.

## How to correlate

Which key first, and why that one is the subject the verdict is rendered for. Then the
secondary keys, calling out any entity that has **several non-interchangeable surface forms**
(the same actor named two different ways in two different columns) — that distinction has to be
declared in the glossary's `value_forms`, because prose cannot restore a distinction the data
model discarded.

Say explicitly what **bounds a cohort** rather than joining: "did this actor do the same
elsewhere" needs one cohort-scoped query, not a subject-keyed one.

## What would exonerate

The exculpatory facts, and how each is evidenced. This section is what keeps the procedure
honest — a playbook that only lists incriminating evidence produces an engine that can only
confirm.

## Known traps

Measurements, not intuitions. A path that looks like an authorisation but is an access log; a
window that a partition pad silently widened; an abbreviation that means something other than
the obvious. Each one here is a wrong verdict that will not be produced again.
