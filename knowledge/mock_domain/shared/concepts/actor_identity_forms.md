---
concept_id: actor_identity_forms
title: A courier has two identifier forms, and they are not interchangeable
---

# A courier has two identifier forms

**A concept about the DATA. Shared, so any use case that has to name a courier reads the
same rules rather than re-deriving them.**

One person, two identifiers, both labelled `courier`:

| Form | Shape | Where it lives | Example |
|---|---|---|---|
| `badge` | 4 digits + a letter | `handler.badge`, `pod.captured_by`, `device_sessions.badge` | `0192C` |
| `device_login` | 6 lowercase letters | `handler.device_login`, `device_sessions.login` | `jdunne` |

An alert routinely carries both. **Bound to the wrong column, each returns zero rows** — and
zero rows reads downstream as "this courier did nothing", not as "we asked the wrong
column".

## Why this is a data-model fact and not a prompt instruction

Prose cannot fix it. If both values are extracted as plain type `courier`, then by the time
any query is generated *the difference no longer exists in the data* — and no instruction can
restore a distinction the data model discarded. So the form is:

1. **declared** as a `value_forms` regex list on the glossary entity (most-specific-first),
2. **classified** deterministically in the engine and stamped on the extracted entity,
3. **bound per form** by each source's `entity_bindings`:

```yaml
courier:
  badge: [handler.badge]
  device_login: [handler.device_login]
```

A value whose form a source binds **nothing** for is **dropped** from that source's filter —
never unioned onto the sibling form's column. Dropping produces a broader query that still
answers the right question; unioning produces a narrow query that answers the wrong one.

## Identifying an actor needs a tuple, not an identifier

Neither form alone names a courier across the whole estate: a login is unique within a depot,
not globally, and badges are reissued. So the identity sources declare
**priority-ordered candidate keys**:

```yaml
identity_keys:
  - [depot, courier]   # preferred
  - [courier]          # fallback
```

The engine picks the **first candidate whose every member the incident actually carries**. A
partial candidate is **skipped, not half-applied** — a half-applied guard is worse than no
guard at all, because the pack declaration reads as a guarantee while the query has quietly
reverted to an OR.

The order is a per-source, **measured** fact: which column is reliably populated differs
between a scan log and a session log, and a key order hard-coded for one is wrong for the
other. That is exactly why it belongs in the pack per source and not in the engine.

## A reference lookup that comes back empty has answered

`device_sessions` is keyed by the (depot, courier) pair and its whole purpose is an
exclusion: **no row means no automated-terminal session, i.e. a person acted.** That is a
finding, so the source declares `zero_rows.health_weight: 0.0` and the health scorer keeps
only the arithmetic. Scoring it as a retrieval defect would invert the evidence.

Two things still make an empty result *not* an answer, and both are detected rather than
assumed:

- **Truncation.** If the source returned exactly its row cap, the pair being looked up may
  sit in the rows never returned. Absence is then no evidence at all and reads `unknown`.
- **An unresolvable clause.** If the incident supplied no depot, the pair cannot be looked
  up. The check reports `unknown` and says which value was missing, rather than letting the
  unscoped rows — which cover other identities — stand in for the actor.

See also [[shipment_elements]].
