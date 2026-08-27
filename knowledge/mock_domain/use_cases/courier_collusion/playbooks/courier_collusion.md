---
playbook_id: PB-COURIER-002
title: Courier Collusion — Handler Versus POD Signer Divergence
domain: fraud / logistics
detectors: [POD-DIVERGENCE]
data_sources: [shipment_ledger, device_sessions]
related_playbooks: [PB-REFUND-001]
typical_alert: "[POD] Repeated proof-of-delivery divergence in depot <depot_code>"
correlation:
  keys: [shipment, courier, depot]
  time_window: within:7d
  time_fields:
    shipment_ledger: created_at
    device_sessions: session.started_at
  fields:
    shipment_ledger:
      shipment: tracking_code
      courier: handler.badge
      depot: depot_code
    device_sessions:
      courier: login
      depot: depot_code
  key_filter: cardinality
---

# Courier Collusion — handler and POD signer divergence

## What it means

One courier is recorded as handling the shipment; a **different** courier captured the
proof-of-delivery. Done repeatedly by the same pair, that is two people covering for each
other: one is not doing the round, the other is signing as though they were.

No refund is required for this pattern, which is what makes it a separate playbook rather
than a section of PB-REFUND-001. A collusion pair may run for months with every claim
adjudicated and every amount ordinary.

## What to retrieve

The **shipment ledger** and **device sessions**. That is all — there is no alert record
declared for this ruleset, because the detector emits a depot-level trend rather than a
per-shipment document. The case builder handles that: an undeclared `alert_record:` block
simply produces no alert-facts section, rather than an empty one implying the alert was
silent.

## How to correlate

The comparison is **within one row** — `handler.badge` against `pod.captured_by` — so no
cross-source join carries it. The join that matters is to `device_sessions`, on the acting
identity, to answer the one question that explains a divergence innocently: was the POD
captured by an automated depot terminal rather than a person?

## Severity

| Finding | Severity |
|---|---|
| Handler ≠ POD signer, no automated session, repeated | High — collusion pair |
| Handler ≠ POD signer, automated terminal session found | Informational — explained |
| Handler = POD signer throughout | None — the alert is a false positive |

## Known patterns

- **The reciprocal pair.** A and B alternate: each signs for the other's rounds. Visible as
  two badges recurring across each other's shipments.
- **The complicit reviewer.** Divergent PODs whose refunds were all *approved on review*.
  This is why `shipment_elements/manually_reviewed` is imported here as an **indicator**
  rather than as the exclusion PB-REFUND-001 treats it as: for refund fraud a review clears
  the claim, but a colluding pair can route a claim past a reviewer who is part of it. Same
  check, same field, opposite reading — the reason the mechanics are shared and the weighting
  is not.

## Cross-links

- **PB-REFUND-001** — the refund-shaped version of the same abuse.
