# `knowledge/` — the domain brain

Everything AFIR knows about a *domain* lives here, as data. The engine in `src/` is
generic: it dispatches on declared condition kinds, enforces declared query guarantees and
narrates declared vocabulary. It contains no entity name, no source name, no field path and
no fraud rule. Re-targeting AFIR at a new domain is an authoring job in this folder, not a
code change.

One folder per pack. Point `knowledge.pack_dir` in `config/main_config.yaml` at the one you
want (or set `AFIR_KNOWLEDGE_DIR` to relocate this parent for a deployment).

| | What it is | Use it to |
|---|---|---|
| [`mock_domain/`](mock_domain/README.md) | The pack that **ships**: a small but complete pack for an *invented* domain (parcel-courier refund fraud) — 3 sources, 2 use cases, 13 conditions, 6 shared checks, synthetic rows. It loads with no credentials and reaches every verdict label. | Read something that runs, end to end. It is also what the generic engine tests assert against, which is why it is a real pack and not a fixture. |
| [`../docs/knowledge-pack-template/`](../docs/knowledge-pack-template/README.md) | A **fully-commented template** of every file a pack may contain, with every key documented and every declaration live YAML. Its one condition is a `stub`, so it provably loads without asserting an outcome it cannot know. | Start a new pack: copy it here as `knowledge/<your-domain>/` and fill it in. |

The template deliberately lives under `docs/` and **not** in this folder. Anything under
`knowledge/` reads as a pack you may select, and a `pack_dir:` pointing at a template that
loads, retrieves nothing and decides nothing is the worst kind of valid configuration — it
looks like a working investigation that simply found nothing.

A real domain's pack is one more folder here (`knowledge/<domain>/`, with its own README and
its own tests). It is authored, not generated; nothing about it exists in `src/` or in
`config/templates/`, hardcoded or in prose. That is mechanically enforced rather than promised:
the pack declares its own words in `domain_vocabulary.yaml`, and `validate_pack` scans both trees
for them and reports a hit as an error.

Read [`docs/architecture/knowledge-pack-authoring.md`](../docs/architecture/knowledge-pack-authoring.md)
before writing a pack. It walks the whole job: the file surface, wiring the RAG, and the
tests that keep a pack honest.

## The one rule that organises a pack

**Pack root = shared. `use_cases/<name>/` = specific.**

```text
knowledge/<domain>/
├── entity_glossary.yaml      ← shared: what to recognise in incident text
├── source_catalog.yaml       ← shared: where the data is, how to query it safely
├── schemas/                  ← shared: full field inventories (RAG-only)
├── data/                     ← shared: reference lookup tables
├── shared/
│   ├── concepts/             ← shared: prose about the DATA
│   └── checks/               ← shared: importable check MECHANICS
└── use_cases/<name>/
    ├── rules.yaml            ← specific: the verdict ruleset (the procedure)
    ├── playbooks/            ← specific: narrative investigation guidance
    ├── concepts/             ← specific: prose about the PROCEDURE
    ├── cases/                ← specific: resolved investigations (often gitignored)
    ├── data/                 ← specific: lookup tables only this procedure uses
    └── reporting.yaml        ← specific: the report's vocabulary
```

The split follows a seam that is not a matter of taste. Knowledge about the **data** —
what an element is, which column holds it, the trap somebody measured on it — is true for
every procedure that reads that element, so a second use case must not have to re-discover
it or copy a field path. Knowledge about the **procedure** — whether that element being
present clears the case or condemns it — is one team's judgement and belongs to their use
case.

So a check's *mechanics* are declared once in `shared/checks/` and a ruleset imports them:

```yaml
# knowledge/mock_domain/use_cases/refund_fraud/rules.yaml
conditions:
  - use: shipment_elements/manually_reviewed   # WHERE the fact lives — declared once
    decisive: true                             # WHAT IT MEANS here — this procedure's call
    decisive_on: [fail]
    exclusion_kind: categorical
    order: 20
```

A second use case imports the same check and weighs it differently — in `mock_domain/`,
`courier_collusion` imports that exact entry as a non-decisive *fraud indicator*, so the same
FAIL clears one case and incriminates the other. Neither restates a field path. A misspelled `use:` is **fatal at load** — a silently dropped condition is a
check that was never evaluated, and in a report that is indistinguishable from a check
whose source returned no rows.

## What a pack cannot do

- **Hold secrets.** Endpoints here are coordinates (index name, catalog/schema/table,
  service nickname). URLs, users and tokens live in `config/main_config.yaml` under
  `log_sources.backends`, referenced by env-var name.
- **Hold code.** A pack is YAML and Markdown. If your domain needs a new *kind* of check,
  that is an engine change (a new generic `kind` in `src/correlation.py`), and it must stay
  generic enough that a second domain could use it.
- **Gate execution.** An empty or missing pack loads cleanly; the pipeline then infers
  everything from the data, as it did before packs existed. A pack only adds precision.
