---
concept_id: shipment_elements
title: Shipment servicing elements and the element counters
---

# Shipment servicing elements

**This is a concept about the DATA, so it lives in `shared/concepts/` and every use case can
read it.** It carries no procedure and no verdict. What a bare shipment *means* — whether it
clears a case or condemns it — belongs to whichever use case is asking; see
`use_cases/refund_fraud/concepts/bare_shipment_is_suspicious.md` for one procedure's reading
of the same facts.

A shipment accumulates **servicing elements** as it moves: an insurance declaration, a
signature requirement, a scan at each handover. The ledger records them two ways, and both
have to be read, because they answer different questions:

| Where | Path | What it holds |
|---|---|---|
| Counters | `element_counters.<TYPE>` | How many elements of that type the shipment carries. A struct of integers. |
| The elements | `scans[]`, `pod{}`, `refund{}` | The elements themselves, with their own fields. |

## The counters are not a complete vocabulary

`element_counters` has leaves for the element types the ledger *counts*, and that is not
every element type that exists. In the reference data there are exactly two:

- `element_counters.INS` — insurance elements
- `element_counters.SIG` — signature-required elements

There is **no `element_counters.SCAN` leaf**. Scans are counted by descending the `scans[]`
array, not by reading a counter — which is why `shipment_elements/no_servicing_elements`
declares `counters: [element_counters.INS, element_counters.SIG]` and separately
`arrays: [scans.scan]`.

**Selecting a struct leaf that does not exist fails the WHOLE query, not just that leaf.**
So an author adding `element_counters.SCAN` on the reasonable assumption that it must exist
does not get a check with one blank field — they get a source that returns nothing, and a
source that returns nothing is indistinguishable in the report from a shipment that carries
no elements. Confirm a leaf in `schemas/shipment_ledger.yaml` before you declare it.

## An absence check and a presence check disagree about a gap

They are not inverses of each other, and the engine treats them differently on purpose:

- **`element_absence`** — a path that was not retrieved makes the result `unknown`. "We did
  not look" is not "it is not there", and on an exclusion check a PASS is the
  fraud-consistent answer, so a partial-coverage PASS would silently convert a retrieval gap
  into evidence of fraud.
- **`element_presence`** — the moment ONE required path resolves with an element, the check
  PASSES and the others become irrelevant. A gap can only ever be `unknown`, never a FAIL.

## The refund block carries its own exculpatory flag

`refund.manually_reviewed` is declared `never_filter` on `shipment_ledger`, and the reason
generalises past this pack. The flag reads like a noise reducer — "drop the reviewed ones" —
and a query generator carrying a general "filter out noise" instruction will do exactly
that. But a manual review is the *exculpatory fact*: `AND manually_reviewed = false` deletes
precisely the rows that decide the verdict, and the check then reports "flag absent" and
goes inert. The guarantee is enforced after generation, on every backend, because a prompt
instruction cannot be relied on to beat another prompt instruction.

See also [[actor_identity_forms]] for why the same discipline applies to the courier's two
identifier forms.
