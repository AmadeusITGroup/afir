---
playbook_id: PB-REFUND-001
title: Refund Fraud — Fabricated Shipment Refund Claims
domain: fraud / logistics
detectors: [REFUND-BURST]
data_sources: [refund_alerts, shipment_ledger, device_sessions]
related_playbooks: [PB-COURIER-002]
typical_alert: "[REFUND] Unusual refund activity detected in depot <depot_code>"
correlation:
  # The FUNCTIONAL join spec, consumed by the correlation stage as layer-1 (authoritative)
  # key resolution. Without it the stage discovers keys from the data, which works and is
  # slower and occasionally picks a decoy — a field that correlates by coincidence.
  keys: [shipment, courier, depot]
  time_window: within:24h
  time_fields:
    shipment_ledger: created_at
    refund_alerts: alert.raised_at
    device_sessions: session.started_at
  fields:
    refund_alerts:
      shipment: alert.claimed_shipments
      courier: alert.courier_badge
      depot: alert.depot_code
    shipment_ledger:
      shipment: tracking_code
      courier: handler.badge
      depot: depot_code
    # `device_sessions` is keyed by ACTING IDENTITY and carries no tracking code, so it is
    # deliberately joined on courier/depot only. Mapping a shipment key it does not hold
    # would produce a join that always misses and read as "no session found".
    device_sessions:
      courier: login
      depot: depot_code
  key_filter: cardinality
---

# Refund Fraud — fabricated shipment refund claims

## What it means

A refund claim was raised against a shipment that shows no evidence of ever having moved.
The pattern is a courier (or someone using a courier's identity) creating shipment records,
claiming they were lost or damaged, and collecting the refund — the parcel never existed.

The detector fires on the **burst**: at least 3 refund claims within 24 hours in one depot.
It is not amount-based. A single large refund is not a trigger and a small one is not
exculpatory, so a report that presents the amount as the suspicious core of the case has
stated the wrong allegation.

## What to retrieve

1. **The alert record** (`refund_alerts`) — first, and it is authoritative. It names the
   depot, the courier (as *both* a badge and a device login) and every claimed shipment.
   Reproduce those facts; do not re-derive them. Two refund alerts in one window are two
   different incidents, and the depot is what separates them.
2. **The shipment ledger** (`shipment_ledger`) — the system of record. Every decisive fact
   is here: the element counters, the scan chain, the POD block, the refund block. It is
   `retrieval_class: primary`, so it gets the long timeout: a cap that fires here does not
   degrade the answer, it decides it.
3. **Device sessions** (`device_sessions`) — only when the question is whether an automated
   depot terminal, rather than a person, performed the action. The catalog's
   `selection_guidance` states when it is worth the cost.

## How to correlate

Join on the **shipment** first — it is the subject the verdict is rendered for. Then on
**courier**, and here the two identifier forms matter: a badge and a device login name the
same person and live in different columns. See [[actor_identity_forms]].

The depot bounds the **cohort**, not the join: "did this courier claim refunds on other
shipments too" is answered from the depot-scoped rows already retrieved, not from a second
subject-keyed query.

## Severity

| Finding | Severity |
|---|---|
| Bare shipment, no review, single handler, burst | High — fabricated movement |
| Elements present but reason code generic + burst | Medium — inflated but real claims |
| Manually reviewed and approved by a named person | Informational — the claim was adjudicated |

## Known patterns

- **The bare shipment.** No insurance, no signature requirement, no scan chain, refunded
  within hours of creation. The textbook shape. See [[bare_shipment_is_suspicious]].
- **The complicit reviewer.** Elements present, review recorded — but the reviewer approves
  every claim from one courier. This playbook cannot see it; PB-COURIER-002 can.
- **The terminal excuse.** The handler/POD divergence is real but an automated depot
  terminal captured the POD, so no second person is implied. An exclusion, and the reason
  `device_sessions` exists.

## Cross-links

- **PB-COURIER-002** — reads the same ledger for a different pattern (a handler/POD
  divergence with no refund at all). Both import the same shared checks and weigh them
  differently.
