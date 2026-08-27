# `mock_domain/` — the shipped pack, and it actually runs

An **invented** domain: parcel-courier refund fraud. Three sources, two use cases, thirteen
conditions, six shared checks, synthetic rows. No real company, system, table or field name
appears anywhere in it — the domain does not exist, which is the point: this is the pack a
generic engine can ship with.

Read it when you want to see what a real declaration looks like end to end, rather than a
template telling you what one may contain. Read
[`../../docs/knowledge-pack-template/`](../../docs/knowledge-pack-template/README.md) when you
want the exhaustive list of keys — it documents every one, reaches no verdict, and lives under
`docs/` rather than here precisely so `pack_dir:` pointing at it can never be a valid choice.

```bash
pytest tests/test_mock_domain_pack.py     # 41 tests, and they are ENGINE tests
```

## What this pack is *for*

It is the generic engine's test subject. `tests/test_mock_domain_pack.py` runs the real
`evaluate_verdict`, the real `UseCaseAnalyzer` and the real pack loader against it and asserts
two things a domain pack's own tests structurally cannot:

1. **Every stage works on a domain the engine has never seen.**
2. **No sentence in the output names another domain** — including the engine's own fallback
   notes, which are the sneakiest leak because they only appear when something is missing.

That is not a theoretical benefit. Writing this pack surfaced **five real defects** that a green
suite and every domain regression had missed, all the same species — an engine detail that
happened to be true of one domain:

| Defect | What it looked like |
|---|---|
| `field_flag` returned `unknown` on any **nested** boolean (`resolve_path` drops bools by design, so a boolean is never mistaken for a join key) | a flag that **was** set read as "flag absent" |
| `record_absence` PASSED on every row for a **boolean** forbidden vocabulary | the check could not fire, and reported "no forbidden value" over rows where the flag was set on all of them |
| `route_membership` hard-coded a three-letter-code regex fitted to one domain's scope-point format | a domain whose scope point is a depot code got `unknown` on its scope **gate** — answered first, so the jurisdiction question was suppressed for the whole procedure |
| four evaluators wrote **domain literals** into their notes | one domain's vocabulary on every domain's report |
| the case builder's `projection_guard` probed with a collector that skips bools | the leaf arrives, the guard reports it missing, and a fully-evidenced verdict is marked `degraded` |

Every one is invisible in the shipped domain, and every one produces the failure mode this
codebase spends the most words on: **a check that cannot fire is indistinguishable, in the
report, from one that had nothing to find.** A second domain is the only test that catches them.

So if you add a condition kind, an evaluator, a rollup rule or a report slot: **exercise it
here**. A kind that only ever runs against one pack has not been shown to be generic.

## The domain, in one paragraph

A parcel (`shipment`, a 10-character tracking code) is moved by a `courier` assigned to one
`depot`. Servicing events accumulate against the shipment — scans, a proof-of-delivery
signature, manual review notes — and a `refund_claim` may be raised against it. The detector
fires on a burst of refund claims from one depot inside a window. Two procedures then ask
different questions of the same data: **`refund_fraud`** asks whether *this claim* is
fraudulent; **`courier_collusion`** asks whether *the courier* is complicit.

## The layout, and what each file demonstrates

```
mock_domain/
├── entity_glossary.yaml               5 entities; a CLOSED abbreviations map; a
│                                      multi-form entity (courier ↔ badge)
├── source_catalog.yaml                3 sources across 2 backend kinds; all five
│                                      ENFORCED query guarantees; not_answered_by;
│                                      selection_guidance; zero_rows; retrieval_class
├── schemas/
│   └── shipment_ledger.yaml           21 leaves, arrays descended, explode_path,
│                                      partition_columns — the RAG-only inventory
├── data/
│   └── refund_reason_map.yaml         a reference lookup, read by a pattern condition
├── shared/                            ─── knowledge about the DATA ───
│   ├── concepts/
│   │   ├── shipment_elements.md       what the servicing elements ARE
│   │   └── actor_identity_forms.md    why one entity has two non-interchangeable forms
│   └── checks/
│       ├── shipment_elements.yaml     4 importable checks — MECHANICS only
│       └── actor_identity.yaml        2 more
└── use_cases/                         ─── knowledge about the PROCEDURE ───
    ├── refund_fraud/
    │   ├── rules.yaml                 8 conditions, a scope gate, a categorical
    │   │                              exclusion, 4 indicators, trigger, case_builder,
    │   │                              notification
    │   ├── playbooks/refund_fraud.md  PB-REFUND-001, with a correlation: join spec
    │   ├── concepts/
    │   │   └── bare_shipment_is_suspicious.md   the PROCEDURE's reading of the
    │   │                                        shared data concept
    │   └── reporting.yaml             9 scoped phrase slots (no root reporting.yaml,
    │                                  on purpose — see below)
    └── courier_collusion/
        ├── rules.yaml                 5 conditions, 4 of them IMPORTED, weighted
        │                              differently; declares NO trigger
        └── playbooks/courier_collusion.md       PB-COURIER-002
```

## The four things this pack exists to show

**1. The same check, imported twice, weighted differently.**
`shipment_elements/manually_reviewed` is declared once — one `field_flag`, one field path — and
imported by both rulesets, which then disagree about it completely:

| | `refund_fraud` | `courier_collusion` |
|---|---|---|
| `report_group` | `validation` | `fraud_indicators` |
| `polarity` | *(exclusion — the default)* | `fraud_indicator` |
| `decisive` | `true`, `decisive_on: [fail]` | `false` |
| `exclusion_kind` | `categorical` | *(n/a on an indicator)* |
| `label` | the library's, verbatim | overridden |

The same FAIL therefore **clears** a refund claim (a named person adjudicated it, so the
behavioural indicators lose their meaning) and **incriminates** a courier (a review approved the
claim despite a handler/POD divergence, so the reviewer either missed it or is in on it). Neither
ruleset restates the field path. This is the whole point of the shared layer:

> mechanics are a fact about the DATA and are declared once; weighting is a judgement made by a
> PROCEDURE and is declared by each use case.

The library file therefore carries no `decisive`, `polarity`, `exclusion_kind`, `report_group` or
`gate`. `tests/test_mock_domain_pack.py` asserts both halves — identical mechanics, divergent
weighting — because a library that leaked one procedure's opinion would quietly become the place
weighting decisions get made.

**2. The two halves of a concept, split.** `shared/concepts/shipment_elements.md` says what the
servicing elements *are*. `use_cases/refund_fraud/concepts/bare_shipment_is_suspicious.md` says
what a bare shipment *means for a refund claim* — and `courier_collusion` reads the same data and
draws a weaker conclusion. The test for whether prose is shared or scoped is exactly that: **if a
second use case might legitimately disagree, it belongs in the use case.**

**3. Every verdict label is reachable.** `fraud` (both by a decisive indicator and by
corroboration), `false_positive` (by a categorical exclusion), `insufficient` (a decisive check
that cannot be evaluated) and `out_of_scope` (the gate) each have a test. `out_of_scope` is the
one most often forgotten in a real pack, and omitting it makes a gate FAIL print as the
false-positive label — which claims the case was examined and cleared when the procedure in fact
declined to adjudicate.

**4. Reporting resolution, deliberately lopsided.** `refund_fraud` ships a `reporting.yaml` and
this pack ships **no root one**. That asymmetry is a test: it proves a use case can supply the
domain's whole report vocabulary on its own, and `courier_collusion` — declaring none — proves
every slot still falls back to a complete, procedure-free engine sentence. (The *other* direction,
a scoped file overriding a root one **per slot**, is covered by
`tests/test_knowledge_pack.py`. A well-authored pack shipping no root file is exactly why a
whole-key shadowing bug in that merge went unnoticed for so long.)

## Things deliberately absent

- **`cases/`.** No resolved-investigation write-ups: there is nothing real to resolve. In a
  domain pack this folder holds live incident detail and must be gitignored *before* the first
  file lands in it.
- **A root `reporting.yaml`.** See above — the omission is the assertion.
- **A `trigger:` on `courier_collusion`.** Also an assertion: a ruleset that declares no trigger
  must make the report say **nothing** about one rather than inventing a plausible-looking
  sentence, which is precisely the defect wiring `trigger:` was meant to fix.
- **Real fixtures.** `tests/fixtures/*.json.gz` are real production rows for the domain pack,
  deliberately committed, because a fixture that isn't committed cannot pin behaviour. Here the
  opposite is right: these rows exist to exercise the engine's *dispatch*, so they are hand-built
  in the test file to this pack's declared shapes and hold nothing real.

## Running it as a pack, not just as a test

The pack loads with no credentials, so you can inspect everything it hands the LLM:

```python
from src.knowledge.pack import load_knowledge_pack
pack = load_knowledge_pack("knowledge/mock_domain")
print(pack.glossary_prompt())     # what the understanding stage is seeded with
print(pack.catalog_prompt())      # what the planner is told about the sources
print(pack.ruleset_spec("refund_fraud")["conditions"][0])   # a resolved condition
```

Actually *retrieving* would need real backends behind the `endpoints` in
`source_catalog.yaml`; the credentials for those come from `config/main_config.yaml`'s
`log_sources.backends`, by env-var name. The pack never holds a secret.
