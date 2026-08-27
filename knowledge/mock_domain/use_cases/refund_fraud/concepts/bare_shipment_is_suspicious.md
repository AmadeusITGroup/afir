---
concept_id: bare_shipment_is_suspicious
title: Why a bare shipment condemns a refund claim
---

# Why a bare shipment condemns a refund claim

**This is a concept about the PROCEDURE, so it stays under `use_cases/refund_fraud/`.** It
is the other half of [[shipment_elements]], which describes the same fields without saying
what they mean for any one investigation. Keep the halves separate: a second use case needs
the data half and must not inherit this reading of it — `courier_collusion` imports the very
same check and treats a bare shipment as a weak indicator, because a bare shipment is not
what collusion looks like.

## The reading

A parcel that genuinely moved accumulates evidence of moving. Somebody declared its value,
somebody required a signature, somebody scanned it at each handover. A shipment carrying
**none** of that, which is then refunded, is more likely never to have existed than to have
been handled unusually badly.

So `shipment_elements/no_servicing_elements` is imported here as a **decisive exclusion**: a
FAIL — elements *are* present — argues the claim is explained, because a normally-handled
parcel that went wrong is what the ordinary refund process is for.

## But it is a HEURISTIC exclusion, and that word does work

The ruleset declares `exclusion_kind` on two checks and gives them different values, which is
the difference between an inference and an attributed fact:

| Check | Kind | Why |
|---|---|---|
| `manually_reviewed` | `categorical` | A named person approved it on a stated date. That is a fact *about who acted*, checkable against the record. |
| `no_servicing_elements` | `heuristic` (the default) | An inference about the shipment's *shape* from element counts. |

The rollup treats them differently on purpose. Positive indicators may rightly outweigh a
heuristic exclusion — a fraud that is not textbook-shaped should still be catchable. They
must **not** outweigh a categorical one: once a named reviewer is on the record, the
behavioural indicators lose their *meaning* rather than their weight, because "generic reason
code plus a burst" is simply what a busy depot's ordinary claims look like.

Getting this the wrong way round produces a specific, recognisable failure: the engine has an
exculpatory fact in hand and lets a behavioural vote bury it.

## What a FAIL prints

The check's `label` states its **requirement** — "Shipment carries no servicing elements". A
FAIL *negates* that, so printing the label under a FAIL states the opposite of what was
found. The engine assembles the note from `fail_detail` instead, which is why the library
entry supplies neutral, finding-phrased wording for both outcomes.

An indicator is the mirror case: its label already states its finding, so a FAIL affirms it
and the label prints unchanged. The asymmetry is the condition's polarity, not an
inconsistency.
