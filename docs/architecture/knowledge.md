# Knowledge pack, use-case case-builder, and the RAG layer

## Knowledge pack

`src/knowledge/pack.py` + `knowledge/<domain>/`: a *swappable* domain brain loaded by `load_knowledge_pack(knowledge_pack_dir(name))`, separate from the FAISS KB. Holds **targeting knowledge** injected deterministically (not by similarity):

- `entity_glossary.yaml` — entity types + candidate field aliases, plus optional `value_pattern`, `cardinality`, `examples`, `used_by`. Drives understanding extraction and field-mapping priors; `glossary_prompt()` appends `e.g. <examples>` (capped at 5) as few-shot hints. An optional top-level **`abbreviations:`** map (term → expansion) is emitted *ahead of* the entities as a **closed list**: use only these expansions, and for an unlisted acronym write it as-is rather than guessing. This is a guard, not a convenience — an LLM will expand an unfamiliar acronym rather than leave it alone, and a plausible wrong expansion (one live run had a three-letter code the pack uses for a *region* expanded into an invented alert-type name) is indistinguishable from a right one once it is in the report's prose. Absent key → nothing emitted.
- `source_catalog.yaml` — per-source description, filterable entities, per-source `entity_bindings`, backend `endpoints`, `used_by`. Drives `ApiCallGenerator` source selection **and** retriever wiring (see `retrieval.md`).
- `playbooks/*.md` — ingested into the FAISS KB as narrative docs.

Re-target the whole system to a new domain by editing the two YAMLs and pointing `knowledge.pack_dir` at the folder — no code changes. A missing/empty pack loads cleanly (pipeline infers everything from data). A production pack gets large (the deployed one measures 53 entities / 29 sources / 10 playbooks / 23 schema docs / 1 verdict ruleset). **Keep domain literals out of `src/` — they live in the pack.**

Two things ship alongside it and are the ones to read first — note they are deliberately not two packs. `knowledge/mock_domain/` is a small but **complete, WORKING** pack in an invented domain no deployment uses (parcel-courier refund fraud), and it is what the generic engine tests assert against: it is the only test that catches an engine detail which merely happens to be true of one domain. `docs/knowledge-pack-template/` is a fully-commented, LIVE template of every file a pack may contain — copy it to `knowledge/<domain>/` and fill it in. It sits under `docs/` rather than under `knowledge/` on purpose: anything in `knowledge/` reads as a selectable pack, and a `pack_dir:` aimed at a template that loads, retrieves nothing and decides nothing looks exactly like an investigation that ran and found nothing. Each has its own README; the authoring walkthrough is [`knowledge-pack-authoring.md`](knowledge-pack-authoring.md).

### The shared / specific split

**Pack root = shared. `use_cases/<name>/` = specific.** The seam: knowledge about the **data** (what an element is, which column holds it, the trap measured on it) is true for every procedure that reads that element; knowledge about the **procedure** (whether that element being present clears the case or condemns it) is one team's judgement. Operative test — *if a second use case might legitimately disagree, it belongs in the use case.*

- **`shared/concepts/*.md`** loads **UNTAGGED**, and that is the whole sharing mechanism: untagged, a concept is RAG-retrievable by every use case and nameable by any ruleset. Tagging it makes an ownership claim the loader then enforces by hiding it from everybody else. `concepts_for(use_case)` returns the use case's own docs FIRST, then the untagged shared ones.
- **`shared/checks/*.yaml`** holds importable check **MECHANICS** — `kind`, logical source, field paths, a neutral `label` and the `pass_detail`/`unknown_detail` wording. The **file stem is the namespace**, so `shared/checks/foo.yaml` → `use: foo/<id>`. A ruleset imports by id and supplies only its own weighting (`decisive`, `decisive_on`, `polarity`, `exclusion_kind`, `report_group`, `gate`, `order`), so the same check can be a categorical exclusion in one procedure and a fraud indicator in another with neither restating a field path. The library declaring a weighting key would hand every importer one procedure's opinion and quietly become the place weighting decisions get made.
- The override is a **one-level merge**: an import's keys replace the library's, and a **LIST replaces rather than concatenating** — a deep merge unioning a `counters` list would point the check at a leaf that does not exist on the source, and selecting a non-existent struct leaf fails the WHOLE query, not just that leaf.
- An unresolvable `use:` is **fatal at load**, checked eagerly over every ruleset. A silently dropped condition is a check that was never evaluated, and in a report that is indistinguishable from a check whose source returned no rows.
- **`data/`** from the root and from every use case flattens into ONE global `pack_data` dict keyed by **file stem** — so stems must be unique pack-wide.

A mature pack can still have an EMPTY `shared/checks/` with every concept tagged to the one use case that wrote it — promotion is a judgement nobody has to make until a second use case reads the same data, so it lags. That is a pack's own backlog and belongs in its README, not here.

### Per-use-case KB subtree

`knowledge/<domain>/use_cases/<name>/`: each use case encapsulates

- `rules.yaml` — verdict ruleset (MOVED here from the flat root `rulesets.yaml`, formerly a procedure-named `<name>_rules.yaml`; the loader still reads a flat root file for back-compat and **use_cases wins on key collision**)
- `playbooks/*.md` — deduped against the flat root by `playbook_id`
- `concepts/*.md` — concept docs (what a data shape IS, what an element means, which identities are automated) — RAG-embedded AND surfaced into the brief as `ConceptRef`s
- `cases/*.md` — **past-investigations / precedent library**: resolved incidents with `case_id`/`verdict`/`subject`/`decisive_reasons`/`route`/`resolution`/`date` frontmatter; RAG-embedded AND matched by the analyzer into the brief as `CasePrecedent`s
- `reporting.yaml` — **report presentation vocabulary** (see below); not RAG-embedded, read deterministically by `ReportGenerationModule`

`load_knowledge_pack` recurses `use_cases/` via `_read_use_cases` (guarded by `is_dir()` so flat-only packs are unchanged); `_read_markdown_docs(dir, doc_type)` generalizes the old `_read_playbooks`; bare YAML dates in case frontmatter are coerced to strings (`_jsonable`) so the RAG `documents.json` dump doesn't fail. Accessors: `concepts_for(use_case, ids=)` / `cases_for(use_case, verdict=)`; `KnowledgePack.concept_documents` / `case_documents` are ingested by `PlaybookSource` alongside `playbook_documents`.

### Playbook `correlation:` block

A playbook's YAML frontmatter may carry an optional structured `correlation:` block (`keys`, per-source `fields`/`time_fields`, `time_window`, `key_filter`) — the *functional* join spec for that fraud pattern. `_read_playbooks` retains it in playbook metadata; `KnowledgePack.correlation_specs()` returns all of them for correlation's layer-1 key resolution. A playbook whose pattern spans several sources carries a worked example; the rest degrade to understanding/discovery.

`SourceDef` also carries optional `query_hints` (free-text injected verbatim into a source's SQL-gen prompt).

### Telling the planner what a source cannot answer, and when to ask it at all

A `description` says what a source **holds**, and that turns out to answer neither of the two questions the planner is actually facing. Both gaps were measured on job 047a603c, and both are stated as their **own labelled lines** under the source in `catalog_prompt` rather than as another clause of a paragraph — a negative or conditional fact buried in prose reads as a list of features.

- **`not_answered_by: [{question, ask_instead}]`** — questions this source cannot answer, each naming the source that can. The planner asked an **access-audit** trail (who looked at what, retained for a privacy obligation) to "check for any reversal, reissue or split events on these documents"; those live in the record projection and its version history. The retrieved rows then cannot contain the answer, and the absence reads as "no reversal found" rather than "asked the wrong source" — the same invisible failure as a 0-row query. Emitted as `DOES NOT ANSWER: <question>. Ask <source> instead.`
- **`selection_guidance: {choose_when: [...], skip_when: [...]}`** — **when this source is worth retrieving**, for a source whose relevance is genuinely incident-dependent. Note there are two ways to get a source retrieved and they are not interchangeable: listing it in a use-case ruleset's `sources:` map makes it a hard **data dependency** — right for a source some condition READS, since a condition whose source is missing cannot be evaluated. That does **not** mean the engine adds it: a declared source the planner declined is *reported* (and gateable, and addable by an operator), never injected, because the fix belongs in the guidance keys below — see `retrieval.md`, "An unmet dependency is REPORTED, never injected". Describing it here leaves the decision with the planner, which is right for a source that answers a question the incident may or may not be asking. The motivating case was a raw-access log: decisive when the question is whether an identity *displayed* records beyond the ones it acted on (reconnaissance / account-takeover shapes) or whether an access was **refused**, irrelevant to a burst fully visible in the actor's own documents — and a 33 TB scan either way. Mandating it would spend that on every run; leaving it to a one-line description meant it was never picked at all. Emitted as `CHOOSE WHEN:` / `SKIP WHEN:` lines.

Both are **advisory by construction** — they steer the planner's own NL text and its source choice, neither of which is enforceable, and for `selection_guidance` that is the point: a guarantee here would just be the hard dependency wearing the wrong hat. Absent keys emit nothing.

### `retrieval_class: primary` — what a source's ABSENCE costs

A third question the `description` cannot answer, and this one has an answer the engine *does* enforce: **what happens to the investigation if this source returns nothing.** `retrieval_class: primary` declares that it cannot be concluded — and the consequence is purely operational, a far larger retrieval budget (`log_sources.primary_source_timeout_seconds`, default **7200s / 2h**, overridable per source with `primary_retrieval_timeout_seconds`).

The reason it belongs in the pack and not in `main_config.yaml` is that a timeout is configured per **backend**, and a backend's sources are not equally important. One pack's multi-TB record projection (MEASURED 717s standalone with the partition prune) shares a warehouse with several second-long reference lookups; one shared cap either starves the projection or hands the lookups a useless hour. Which of them matters is a *domain* judgement — exactly the kind of thing the pack owns.

And it matters because **a cap that fires on the system of record decides the verdict rather than degrading it**. Job 047a603c: the projection hit the backend's shared 1800s cap, returned 0 rows, every decisive condition went `unknown`, and the verdict fell back to whatever corroboration happened to be cheap — while the retrieval stage still reported success. So for a primary source the cap must fire only on a **genuinely stuck** statement, which for this projection means close to an hour. Nothing is lost by waiting: loss of contact with the warehouse is caught separately and far faster by `max_consecutive_poll_errors` (5 consecutive poll failures, ~25s), so elapsed time no longer has to double as a liveness signal — a statement sitting in `RUNNING` is working, not stuck.

`LogRetrievalEngine._apply_primary_budget` resolves it at config-build time, writing all three coupled budgets from one number: the engine's per-source `asyncio.wait_for` cap (`retrieval_timeout_seconds`), the retriever's own statement budget (`statement_timeout_seconds`), and the legacy `max_poll_attempts` floor. They must move together because the lowest one wins *silently* — computing the number in two places is how they drift apart. Against an operator-set value it only ever **raises**; against a value *it* previously wrote (marked with `_primary_budget_applied`) it rewrites in either direction, which is what makes the setting editable in the running process rather than only at startup — see [retrieval.md](retrieval.md) for `refresh_primary_budgets()` and the Configuration-tab wiring. And it is **strictly a budget signal**: it never changes *whether* a source is selected (`selection_guidance` / a ruleset's `sources:` map decide that) nor how a zero-row result is scored (`zero_rows`).

The default is **2 hours**, not the 3000s it started at, because the margin is the real budget: the accepted baseline finished the primary source in 2990.6s under a 3000s cap (27s, 0.9%), and a re-run of the *same* incident timed out during a warehouse-wide slowdown that also took a 2-row reference lookup from 11.9s to 386.6s — 15 conditions `unknown` against the baseline's 2. A cap fitted to one measurement is fitted to that measurement's load.

### `reporting.yaml` — the report's words belong to the domain

`use_cases/<name>/reporting.yaml` holds the wording `ReportGenerationModule` emits. It exists
because the engine had accumulated ten `§`-numbered citations from ONE team's written
procedure, a phase label naming that procedure's central record type, and a prompt line naming
four of its concepts — every one a fact about one procedure, none of them true of the next
investigation running through the same code. A clause number is the worst kind of leak: it reads
as authoritative, it cannot be checked from `src/`, and it silently mis-cites every other use
case's report.

Two keys, both optional:

- **`phases:`** — an ordered `[{keywords: [...], label: "..."}]` list classifying one
  chronology event into an incident phase, for the Incident Reconstruction's phase headers.
  Matched against `"<source> <action>"` lower-cased, **first hit wins**, so the specific rule
  goes first — an event from a source whose *name* contains the procedure's own acronym must be
  classified by what the event IS, not swept into "Alert and detection" by the coincidence.
- **`phrases:`** — `{slot: text}` for the named slots the engine renders
  (`containment_target`, `containment_action`, `containment_action_unknown`,
  `selling_platform`, `indicator_driven_fraud`, `categorical_exclusion`,
  `notification_draft_heading`, `notification_closure_heading`, `wider_evidence_framing`,
  `narration_concepts`, `reconstruction_shape`).

Accessors: `_reporting_for(use_case, key)` resolves **most-specific-first**
(`use_cases.<name>.<key>` → flat root `<key>`), `phase_rules(use_case)`, and
`report_phrase(slot, default, use_case)`.

**A MAPPING MERGES PER SUB-KEY; A LIST REPLACES.** `phrases` is a bag of independent named
slots, so a use case re-wording ONE of them must inherit the rest. Resolved whole-key (as
`_reporting_for` originally did), declaring a single scoped phrase silently discarded the
entire domain-wide map — and the loss is invisible, because every slot falls back to the
engine's own generic sentence, so the result reads as a pack that never declared wording
rather than as one whose wording was dropped. `phases` is an ORDERED decision procedure
(first keyword hit wins, so the order IS the semantics), and merging two authored orderings
would produce a precedence nobody wrote — for a list, replacement is the only safe reading.
Same rule, and for the same reason, as a `use:` check import.

A well-authored pack HIDES that defect: a single-use-case pack ships no root `reporting.yaml`
at all, which is exactly why whole-key shadowing never surfaced on the deployed one. Pinned by
`tests/test_knowledge_pack.py` (`test_a_scoped_phrase_overrides_one_slot_and_INHERITS_the_rest`,
`test_scoped_phases_REPLACE_the_root_list_rather_than_merging`) and by the pack template,
which ships both scopes so the behaviour is visible in a diff.

Three properties the engine relies on, and which a new slot must preserve:

- **Every key no-ops when absent.** No pack, no `reporting.yaml`, or no such slot all fall
  through to the engine's own default, which is written as a complete, procedure-free
  sentence. A pack-less domain gets a correct report — just without the citations.
- **Substitution is `str.replace`, never `.format`.** Procedure prose contains braces, on
  which `.format` raises. Placeholders are named in a comment above each slot.
- **Scoping is per use case for a reason.** Hoisting a clause number to the flat root would
  cite it in every use case's report; the flat root is for wording that is genuinely
  domain-wide.

`narration_concepts` and `reconstruction_shape` are the two slots injected into the *prompt*
rather than the artifact: the first tells the LLM to cite procedure concepts by their KB
names (paired with `entity_glossary.yaml`'s closed `abbreviations:` list, which forbids
guessing), the second describes the chronology shape a reviewer of that procedure reads.

**Caveats:** Pydantic **silently drops** unknown YAML keys — a new pack key does nothing until added to `EntityDef`/`SourceDef` (or, for the correlation block, retained explicitly in `_read_playbooks`). `source_catalog.yaml` uses YAML anchors (`&`/`*`); a `*ref` without its `&anchor` raises a ComposerError that makes the *whole* pack load empty.

Threaded into `IncidentUnderstandingModule`, `ApiCallGenerator`, `LogRetrievalEngine`→retrievers, and `CorrelationModule` (via RAG) using a `knowledge_pack=` kwarg.

## Editing a pack

Four modules under `src/knowledge/`, an HTTP surface (`docs/API.md` §11) and the UI's
Knowledge tab. Authoring used to be file-only; the two facts that shaped every design
decision here are the ones above — **a pack fails silently, and nothing validated it.** The
path an edit takes is store → lint → **dry run** → preview → apply: the lint says the engine
will read the declaration, the dry run says a real run would answer it, and neither is
implied by the other.

### `pack_store.py` — file CRUD whose write path is the whole point

`pack._read_yaml` swallows every parse error and returns `{}`, so a broken
`source_catalog.yaml` does not raise: the pack loads with **zero sources**, every condition
goes `unknown`, and the report reads INSUFFICIENT DATA — indistinguishable from *"the
sources had nothing"*. An editor that can reach that state silently is worse than no
editor, so `_atomic_write_verified` (modelled on `job_store.py`, not
`config_store._atomic_write`) makes **three** checks in order, and the third is why the
module exists:

1. the temp file **re-parses from disk** — a dangling `*alias` surfaces here as
   `ComposerError` rather than as an empty pack on the next run;
2. a `.md` file's frontmatter block parses;
3. **the candidate does not parse to empty when the pre-image did not.**

Any failure → unlink the temp file, raise, and the target is byte-identical. `PID`-suffixed
temp file, `flush` + `fsync`, `shutil.copy2` for the backup (not a rename, so the target is
never absent), then `os.replace`.

**Edits are anchored line-range replacements, never a re-serialisation.** `config_store`'s
dotted-path patchers do not transfer: measured, `_find_scalar_line(catalog, "sources")` →
34 but `"sources.record_lake"` → `None`, because both big pack files are *lists of
mappings* and a dotted path is meaningless past the first segment. And a
`safe_load` → `dump` round trip would expand every anchor and delete every comment — a
use case's `rules.yaml` is ~60% comments, and those comments are where the measurement
behind a threshold is recorded. A one-line patch through this path changes exactly one
line; the test asserts the output *bytes*, including all four `&same_as_*` anchors.

**Paths are rejected, never sanitised** (the precedent is report ids in
`report_delivery._safe_id`): per-segment `[A-Za-z0-9._\-]{1,128}`, depth ≤ 6, no absolute
path, `".."` refused **as a segment** (a filename may legitimately contain a dot pair), and
`.history` not addressable. A sanitised `../../etc/passwd` that silently becomes some other
real readable file is worse than a 400.

**History is content-addressed and gitignored** — `knowledge/<pack>/.history/index.json`
plus `blobs/<sha[:2]>/<sha256>`. No `git` and no `subprocess` (the repo has zero
`subprocess` usage and this was not the place to add the first). Content addressing makes
edit-then-revert cost one blob, which is what makes snapshotting **every** write cheap
enough to be unconditional — and unconditional is the property the operator was promised.
A restore is itself snapshotted, and `delete_file` snapshots *first*, refusing the unlink
if the snapshot could not be stored.

**`scaffold_pack` must not copy the template's `domain_vocabulary.yaml`.** Measured hazard:
the template declares `example_platform`, and its own `rules.yaml` contains
`example_platform_a:`; `_vocabulary_pattern` treats `_` as a boundary and
`tests/test_pack_template.py` scans the template against *all installed* vocabularies — so
a plain `cp -r` of the template makes **the template's own test fail**. The scaffold writes
a fresh vocabulary from the caller's word list and **refuses an empty one**, because a pack
directory with no vocabulary fails `test_every_installed_pack_declares_its_vocabulary` and
takes the whole suite down with it.

### `pack_validate.py` — the lint that did not exist

`validate_pack(pack_dir)` → `{pack, ok, errors, warnings, infos, diagnostics[], counts}`,
each diagnostic `{severity, code, path, line, message, detail, hint}`. Takes a
**directory**, not a pack name, so the checked-in template — which lives outside the packs
root — is validatable too; otherwise the one pack every author starts from is the one pack
nobody ever lints. It reuses `_resolve_check_imports`, `_read_shared_checks`, `EntityDef`
and `SourceDef` from `pack.py` but **deliberately not `pack._read_yaml`**, which is the
thing being checked.

The severity line is drawn on consequence, and the full code list lives in `docs/API.md`
§11:

- **error** — the pack does not work, or works while lying. `unknown-condition-kind` is the
  archetype: a kind the evaluator does not dispatch falls through `_eval_condition` and is
  never evaluated. The dispatched-kind list is derived with
  `inspect.getsource(correlation._eval_condition)`, not grepped — a naive
  `grep 'if kind =='` over `src/` returns 21 because seven are RAG *source* kinds.
- **warning** — it works, but a declaration is inert or a human is misled.
  `unread-pack-key` fired on a real `do_not_consider:` that was declared and read nowhere,
  the same defect class as the `trigger:` key that shipped in three rulesets doing nothing.
  Both are wired now, so it reports zero on every installed pack — and the test that used to
  pin the one hit now asserts **zero plus a byte-level mutation** that restores exactly one,
  because "zero warnings" is what a lint that stopped working also reports. `label-polarity-unaffirmed` is the defect that shipped with 1083 tests green and
  **cannot** be an error: the engine cannot read prose.
- **info** — `playbook-only-use-case`, deliberately not a warning. Nine of the deployed
  pack's ten use-case directories are playbooks-only, and nine permanent false alarms is
  how a check gets weakened and then ignored.

The neutrality pattern is **duplicated** from the test suite rather than imported (`src/`
must not import from `tests/`, and moving the scanner into `src/` would put it inside the
code being scanned). `test_the_neutrality_pattern_matches_the_test_suites` asserts both
sides produce the same pattern string and the same `_ENGINE_OWNS` set, so the copy cannot
drift into being the weaker check.

### `pack_dry_run.py` — the question the lint cannot ask

`pack_validate` answers *will the engine read this declaration*, and the field-path check
answers *is this leaf on the source*. Neither answers the one that costs a live
investigation: **a condition that is valid YAML, names a real leaf, and reads `unknown` on
every row that has ever come back.** It never votes, the verdict is unchanged, and the only
trace is one line in a report nobody reads as a defect. So `dry_run(pack_dir)` replays the
candidate ruleset over the evidence of runs already stored (`stored_runs` over the job
documents `job_store` holds) and counts the three outcomes per condition, plus two findings
the counts cannot express:

- **`always unknown`** — asked on every subject of every replayed run and answered nothing.
  The fix is a field path, a source, a row selector or a bound. `kind: stub` is excluded: a
  pack ships one to declare that a question was considered and cannot be asked yet.
- **`never evaluated`** — no check line at all. `evaluate_verdict` emits one line per
  condition per *subject* with no branch that skips a condition, so this is really a
  statement about the **ruleset** (it resolved no subject) arriving per condition. Neither is
  folded into `unknown`. A composite's **children** are marked `folded` rather than counted,
  because the parent owns the one report line and a child that produces none is by design —
  the first version of this reported every composite child as a defect.

**Two facts the verdict engine is given by its caller are absent from a stored run** — the
per-source `row_caps` and `keyed_sources` — so a replay is not the reading the run took, and
**each bound names its direction of error**: reporting "row_caps unavailable" without which
way it leans lets an author take a clean dry run for a guarantee. Both are accepted as
optional injected parameters for a caller that has them.

Pure, read-only, no LLM and no network, and it **never raises**: every failure becomes a
`problems` entry, because a dry run that aborts on one unreadable job document tells the
author less than one reporting 11 of 12. Two budgets, both because this runs inside a request
an operator is waiting on — 12 runs / 25s for the plan preview, 6 / 12s when the assistant
calls it as a tool inside an exploration loop. `render()` is the tool-facing text and
`as_dict()` the preview payload; `python -m src.knowledge.pack_dry_run <dir>` is the CLI.

`plan_checks` computes it **on the candidate tree, after the edit is applied in memory**, so
the operator sees what the pack would do *after* the change and not before it — and `/apply`
now refuses a plan that **introduces** a validation error (not one that inherits a
pre-existing one, or a pack already failing could never be repaired through this surface).

**`plan_checks` is four readings over that one candidate tree**, and the reason there are four
is that each one is blind to the next: `pack_validate` (will the engine read the declaration),
`pack_dry_run` (does the ruleset decide anything over rows that really came back),
`pack_selection_delta` (which procedure would adjudicate) and `pack_verdict_delta` (what the
edit does to the findings already on record). Only the first gates: the other three are
warning-severity by construction, because a moved selection or a moved finding is usually the
*point* of the edit and "is this move correct" is a judgement no arithmetic settles. `apply_plan`
runs the validation alone (`dry_run=False`, `deltas=False`) — spending a replay budget and two
corpus reads on every write to produce numbers no branch consults is a cost the preview already
paid, and `tests/test_pack_assistant.py` pins each of the three as skipped rather than counting
the two flags.

### `pack_selection_delta.py` — which procedure would adjudicate, before and after

`pack_validate` and `pack_dry_run` both read the **candidate pack in isolation**, and neither can
see the one side effect an authoring edit has on every procedure at once: which ruleset adjudicates
is a keyword score over the playbook titles and join keys, so adding a use case — or rewording one
title — silently re-scores every incident the pack has ever seen. That failure is invisible by
construction, because the losing procedure's conditions still resolve against real rows and every
stage reports success.

So this is a **difference and it needs both sides**: the same corpus scored under the base pack's
specs and under the candidate's, and the runs whose selected procedure moved are named. Three
properties, each because the alternative fails quietly:

* **One scorer, not two.** Every score comes from `correlation.select_correlation_spec_explained`
  through a duck-typed pack and analysis. A re-implementation here would drift, and the tie-break
  on the secondary hypotheses is exactly the detail a copy loses.
* **An unchanged vocabulary cannot move a selection**, so an edit leaving every title and key alone
  short-circuits *before* reading the corpus. Most pack edits are that edit, and a check costing
  seconds on every save is a check that gets turned off.
* **No corpus means silence, not a clean result.** `compared` is the field to read first; an empty
  `flips` with `compared=False` is a silence, and reporting "0 flipped" on a deployment with no
  stored runs would read as a guarantee.

`scripts/measure_playbook_selection.py` is now a thin CLI over the same functions, so the number in
the preview and the number an author measures offline come from one implementation.

### `pack_verdict_delta.py` — what the edit does to the findings already on record

The three checks above can all pass while a reworded field path, a threshold moved by one, or a
check imported under the other polarity changes what a report **concludes about a named person's
conduct**. This one re-adjudicates the same stored evidence under both packs and diffs the
per-subject, per-condition lines — the comparison one makes between two runs of one incident,
made here between two packs over one run. Four properties:

* **Base replay against candidate replay, never candidate against the RECORDED verdict.** A stored
  run's evidence sidecar is flattened and carries neither `row_caps` nor `keyed_sources`, so a
  replay legitimately reaches a weaker reading than the run did. Diffed against the recording that
  is a regression on every line; diffed against the base pack's replay of the same rows it cancels
  exactly, because both sides suffer it identically. The one caveat that does **not** cancel is
  stated in `limits`: a condition reading `unknown` for the replay's own reasons reads `unknown`
  under both packs, so an edit meant to fix such a check shows no change.
* **An identical replay surface cannot move a verdict.** The surface is exactly three things — the
  resolved ruleset specs, the `data/` files, and the entity→column bindings read through
  `field_priors_for` — so an edit leaving all three alone short-circuits with zero IO. And
  `replay_surface` **raises** on a pack it cannot read rather than returning an empty surface, which
  would compare equal to the other side's empty one and short-circuit as "nothing moved"; the caller
  degrades to `surface_changed=("unknown",)` and compares anyway.
* **The comparison is proved able to disagree with itself first.** The base pack replays the first
  run twice, and unless the two agree the delta is **withheld**. Without that control, a condition
  reading the clock reports the author's edit as the cause of a change it did not make. One extra
  replay for the whole check, not one per run.
* **A `decided_flip` is reported apart from a transition through `unknown`** — pass↔fail is a
  changed finding, a check that stopped answering is a broken one, and the remedies differ.

Same posture as its sibling: pure, read-only, never raises, `compared` first, `python -m
src.knowledge.pack_verdict_delta <base> <candidate>` as the CLI, and `scripts/replay_saved_
verdicts.py` unchanged as the whole-corpus instrument. Tests: `tests/test_pack_verdict_delta.py`
for the module, `tests/test_pack_assistant.py` for the seam — where the short-circuit is asserted by
making the corpus read **raise**, because a store that is merely unused and one that is unreachable
look identical from a passing test otherwise, and that is what keeps the editor's suite off the
deployment's job history.

### `pack_assistant.py` — propose, never write

`LLMClient.tool_call` is **one round and executes nothing**, so the agent loop is ours;
there was no example in the repo to copy, and the `messages` annotation actively misled
(widened `List[Dict[str, str]]` → `List[Dict[str, Any]]`, since a tool round-trip carries a
`tool_calls` *list* value).

Eight read-only tools — `list_files`, `read_file`, `search`, `pack_summary`, `validate`,
`dry_run`, `probe` (below) and `read_skill` (below) — and **no write tool at all**, which is what
makes "nothing touches
disk until the operator approves" structural rather than procedural. That list is pinned as
a closed set in `test_pack_skills.py` rather than screened for names containing *write* or
*patch*: a shadow run watched the model call **`edit_file`**, and a substring screen only
ever covers the names somebody thought of. `dispatch_tool` answers an invented name by
listing the real ones, which is what let that session recover and still propose a plan.

**`probe` is the eighth and the only one that leaves the machine**, and it is admitted on exactly
the terms the other seven are: it delegates to `pack_probe`, whose read-only posture is
*structural* — the first token checked against a closed verb tuple and a `;`-chain refused rather
than split — checked at the lane's seam before a connection is opened and again inside `Probe.ask`,
never as a sentence in a tool description, because a prompt cannot beat another prompt. It exists
because until it did, the method skills said *probe before you declare* and the model's only
recourse was to write the measurement into `questions` for a human to take, so the loop never
closed. Three structural bounds: a per-plan probe count, a row cap and a timeout, all three
config-readable (`knowledge.assistant_probes` / `_probe_row_cap` / `_probe_timeout_seconds`, where
`0` probes disables the lane); every probe recorded on the session snapshot, so the preview shows
*what was measured to justify this line*; and the engine opened **lazily**, on the first probe that
gets past its checks — a session that never probes builds no retriever and reaches no backend. When
no backend is reachable the probe is refused and the plan degrades to today's `questions` behaviour,
**never to a guessed number**.

`pack_summary` is the anti-hallucination tool:
it returns the entity types, source names, ruleset keys, shared-check ids, the 20 kinds the
evaluator actually dispatches and the 3 that honour `expected_label` — which is what stops
the model inventing a `kind` that `pack_validate` would then flag as an error. It reads the
YAML directly for the same reason the checker does: built on the loader it would describe a
*broken* pack as an *empty* one, and an assistant told a pack has no sources will happily
propose adding the ones already there.

Two hard budgets, because both failure modes are real: **8 turns** (a model that keeps
reading never proposes) and **200 KB of tool output** (three ~456 KB schema files blow any
context). Exhausting either appends an explicit *"your exploration budget is spent, propose
from what you have"* turn and emits an `assist_note`, rather than stopping silently — the
plan the operator sees must have been produced knowingly.

`EditPlan`/`EditOp` are Pydantic models, so they are also the LLM's output schema.
`plan_preview` renders the diffs **server-side** and `apply_plan` is all-or-nothing:
every op is prepared and verified in memory (path, suffix, size, exists/not-exists, line
bounds, pre-image anchors, final text parses and is not empty-when-it-wasn't) before the
first byte lands; an I/O failure mid-write restores from this transaction's snapshots and
reports `rolled_back`. An **unknown `op` is skipped and reported, never raised** — a model
inventing `rename` must not sink a proposal whose other three ops are fine. Sessions are
in-memory and capped at 20: a plan is computed against a snapshot of the files, so it is
worthless after a restart and persisting one would only invite applying a stale plan to
moved files.

**`pack_attachments.py` handles a document/image asymmetry that is not cosmetic.** A
document is converted to text *server-side* (via `src/rag/document_ingester.py`, spooled
under `exports_dir()/uploads` — **not `/tmp`**, which a Databricks App may not have), so
every endpoint can read it. An image stays bytes, and whether it works is a property of the
deployed model — so it is **probed with its own call** before the exploration loop. Letting
the first exploration turn double as the probe would put an image refusal inside the turn-0
handler, which reads any turn-0 failure as *"this endpoint cannot call tools"*: one
message, the wrong diagnosis, and the diagram silently dropped. On refusal each image
becomes a placeholder naming the file and saying it was **not** seen, and `image_mode`
reports `text_only`. Every cap is stated in both `note` and the model's own context,
because a truncated document that does not say so is read as the whole file. No Pillow, so
a byte cap and a count cap rather than resizing — adding an image codec to a fraud
pipeline's install path is the wrong trade.

Unlike the pack importer, `prepare()` is **not** all-or-nothing and returns
`(prepared, errors)`: an attachment only adds context to a question, so refusing four good
diagrams because a fifth was an unreadable scan would make the operator re-upload all five.
Every rejection is rendered beside the accepted files — the failure being avoided is an
assistant proposing from three attachments while the operator believes it read four.

### `pack_skills.py` — the method knowledge the assistant is given

`skills/*.md` is what this project learned about authoring a pack: probe before you declare,
which declarations are enforced and which are advisory, the ways a condition silently never
fires, why a passing fixture can be worthless, what to measure before a threshold is a
number. It is *method*, so it lives under `src/` — where `validate_pack`'s neutrality check
scans every `.md` against the pack's declared vocabulary, which is the **mechanical**
guarantee that a skill teaches technique and never one domain's nouns. `test_pack_skills.py`
asserts `SKILLS_DIR` is inside `_NEUTRALITY_ROOTS` **by path**, importing the list from
`pack_validate` rather than re-spelling it: moving the directory would otherwise end the
guarantee with nothing failing.

**Selection is deterministic, and that is the whole design decision.** The obvious shape is a
read tool plus a line in the system prompt suggesting it — which makes the library opt-in, and
a plan built on none of it is indistinguishable from one built on all of it, since both come
back as confident YAML. Standing rule: a prompt instruction cannot be relied on to beat
another prompt instruction. So `select()` matches the operator's question, the focus paths and
any rejection guidance against each skill's `triggers` and injects the winners into the
opening turn; `read_skill` is the *widening* path for what selection did not reach. Three
things the seam holds, none of which fails loudly on its own:

* **Always-on is a TIER, not a high score.** Ranked among the trigger counts, a question
  naming two of another skill's triggers demotes the spine and `MAX_INJECTED_SKILLS` drops it
  — the failure `INJECTED_CHARS_BUDGET`'s exemption exists to prevent, arriving through the
  count. Found by a test, before the library shipped.
* **A skill that will not parse is skipped and REPORTED** (`problems()` → an `assist_note`),
  like `tool_mode` / `image_mode`: the visible symptom of a silent drop is a proposal that
  quietly stopped knowing something.
* **Word-edge matching treats only letters and digits as word characters**, so `filter` finds
  `never_filter` while `cap` does not find `capture` — permissive at a separator, because an
  extra skill spends context and a missing one spends a live run.

**Measured in shadow mode, not asserted** (`scripts/shadow_pack_assist.py`): four generic
authoring questions against one installed pack, each asked twice — library on, and library
redirected to an empty directory so the control arm runs the same code path a deployment with
no library takes. Nothing is applied: `apply_plan` and every `pack_store` mutator are replaced
by tripwires armed *by name off the module*, and every file in the pack tree is hashed before
and after (16 files, byte-identical, no tripwire fired). One arm cannot be read — every plan
looks reasonable — so the control is the measurement.

Over the four questions: **3 ops proposed with the skills against 5 without, 0 blocked by
`plan_preview` against 2, and 22 of 28 questions naming a measurement against 16 of 23.**
The direction is the point rather than the magnitude: the skilled arm proposes *less* and asks
*more*, and both blocked ops belong to the control. The sharpest case is the ask for an
exclusion over a field the pack does not have — the control wrote the op anyway, on an
admitted guess at the field path, and `plan_preview` blocked it; the skilled arm proposed
**nothing** and said why, naming the failure it was declining to ship (a path on no schema is
a *warning* in `pack_validate`, because a schema doc is sampled, so the condition would read
`unknown` on every row and print as INSUFFICIENT DATA under a fully green run). What is
deliberately **not** measured is whether an edit is correct: there is no ground truth for that
here, and inventing a score would be the `fixes-must-generalize` mistake with a number
attached. The script prints every plan in full for that judgement.

### `pack_probe.py` — the measurement the skills tell you to take

A skill saying *measure it* is worth nothing if taking the measurement means writing a script,
because the script that gets written is the one that goes wrong. Every ad-hoc probe under
`scripts/` pinned its own backend URL, its own CA bundle and its own credential env names and
reimplemented the connection — so it measured **a path the pipeline does not take**, and the
declaration was then written from a number that was never true of the retrieval. `pack_probe`
is the shipped, pack-agnostic replacement: a library plus a CLI
(`python -m src.knowledge.pack_probe`, like `pack_validate`), built on `LogRetrievalEngine`, so
coordinates, credentials, TLS, row caps and per-source timeouts are whatever the config and the
pack actually say. It lives beside `pack_skills.py` for the same reason the skills do — under
`src/`, inside the neutrality scan, so the helper cannot learn a domain either.

Four properties, each of them a defect this repo has already paid for:

* **A failure is not an empty result.** `except Exception: return []` is the natural shape of a
  hand-rolled probe and it is fatal — a rejected request and a clean no-match come back
  identical, so every candidate is cleared by the one outcome that should have disqualified it.
  Nothing here returns a bare list: `Measurement` carries `ok` beside `rows` and keeps the
  exception **type**, and `unavailable()` separates a third case — a *declared* source that
  built no retriever, which cannot answer anything and is not a fact about the data.
* **A measurement without a control is not a measurement.** `selectivity` and `pair` run the
  control themselves, and `Comparison.verdict()` has four branches: a zero against a zero
  control reports **NOT TESTED**, never "empty". `within=` rides on both sides, since dropping
  the population bound from the control inflates the rate.
* **Read-only, structurally.** Every statement passes `_read_only_reason` — a first-token check
  against a closed verb tuple *and* a `;`-chain check, because `SELECT 1; DROP TABLE t` is what
  a predicate built from an unvalidated string looks like and a leading-verb test accepts it.
  Refused, never sanitised, and refused **before** the seam is called (asserted on the fake).
  Same posture as the assistant's missing write tool.
* **Three dialects, no approximations.** `sql` / `esql` / Query DSL builders, dispatched off the
  source's own kind onto the retrievers' raw seams (`_execute_sql`, `_execute_esql`, `_search` —
  the ES one was extracted for this, since inline it was the one backend an outside caller could
  reach only by duplicating both the client and the response flattening). A question a dialect
  cannot express faithfully raises `Unsupported`; an approximation that answers a *different*
  question is worse than no number, because it gets written into a declaration.

The seven questions are the four the spine skill demands, plus the two that only exist because
one number lies: `leaves` (what a **row** carries, not what a schema doc claims — flattened,
arrays through one `[]` segment, with the sampled count in the answer), `population` (total /
not-null / **not-blank** / distinct — a column that is 100% non-null and 100% blank reads as
fully populated to any single check, and a cardinality-1 column is populated and useless),
`values`, `spread` (lengths — the precision and position traps), `selectivity` (the base rate),
`pair` (all four cells, which is what licenses or refuses a composite key), `cost` (elapsed
against the budget the source really runs under, so the **margin** is the number recorded).

Its *build* counterpart is `scripts/generate_source_schemas.py`, now `--pack`-driven
(defaulting to the configured pack) rather than pinned to one directory — the complete field
inventory per source into `knowledge/<pack>/schemas/`, arrays descended, `--measure` for
per-leaf population, hand-written `description:` text preserved across regeneration. Generate,
annotate, then probe the specific claims a declaration is about to make.

Tests: `tests/test_pack_probe.py` — fake retrievers over the real builders and the real guard,
no backend and no LLM. Two of its assertions are about the module rather than its behaviour: the
CLI must expose every measurement (a helper nobody can invoke from a shell is one that gets
re-hand-rolled), and the file must contain **no host, no `.pem`, no env-var read** — if one
appears, the library has become the thing it replaced.

### What is not here

**No hot reload.** Every write answers `restart_required: true`, exactly as the
Configuration tab reports a non-live field, and `GET /api/v1/knowledge` says which pack is
loaded so that advice is actionable. Reloading a pack mid-run would swap the rules under an
in-flight investigation.

Tests: `tests/test_pack_store.py`, `tests/test_pack_validate.py`,
`tests/test_pack_assistant.py`, `tests/test_pack_attachments.py`,
`tests/test_knowledge_api.py`, `tests/test_pack_skills.py`, and the Knowledge tab's share of
`tests/test_webui.py`. The skills' *wiring* is asserted in `test_pack_assistant.py` against a
synthetic library plus one test on the shipped one — without that last, emptying `skills/`
leaves every other test green.

## Use-case case-builder → verdict-grounded reports

`src/usecases/`. The problem it solves: the verdict was correct but the LLM never *saw* it — the report narration ran with `rag=None` off an EvidencePack with no verdict field, so it could write "suspend the actor's identifier" and HIGH-confidence anomalies while the appended verdict said FALSE POSITIVE.

The fix: a deterministic **`UseCaseAnalyzer`** distills the verdict + raw rows + KB concepts + matching past-investigation precedents into a compact **`InvestigationBrief`** (`pydantic_models.py`: `InvestigationBrief`, `AssetTimelineEntry`, `ConceptRef`, `CasePrecedent`, `CaseAssessment` — all generic, no procedure-named fields), and the report/anomaly LLM calls narrate/score **from that brief** so they cannot contradict the verdict.

Package (flat imports, sibling of `correlation.py`):

- `base.py` — the ONE `UseCaseAnalyzer` (concrete, not an ABC) + its shared deterministic tools: `build_asset_timeline` walks the record + transaction rows into a subject→asset→reversal chronology via `resolve_path`/`_parse_ts`; `derive_action_backbone` keys next-steps off the verdict label (VALID FRAUD→lock the acting identity + raise a follow-up case, FALSE POSITIVE→close the case naming the exclusion, INSUFFICIENT→pull the full window) and appends a scope step reflecting what the sweep actually found; `build_scope_discovery` (below); `collect_concept_refs`; `collect_precedents` scores cases by same-verdict/route/decisive-reason overlap; `join_status_from_transforms` → explicit "ran, N matches" vs "not evaluated" so the report never confuses a join that found nothing with one never attempted (the run-to-run-inconsistency fix).

**One analyzer, no per-use-case Python — the ruleset's `case_builder:` block.** There used to be a per-use-case module in `src/usecases/` named after one team's procedure, holding that procedure's own `Analyzer` subclass; an engine with a procedure-named file in it cannot claim to be domain-agnostic. Measured before removing it: of ~290 lines, the domain-specific part was **five facts** — a concept-id list, a join-name list, a projection probe path, two scope-note templates. Everything else was orchestration every use case needs, so the next use case would have copied it to add its own five. The flow now lives once in `UseCaseAnalyzer.analyze`; the five are pack declarations:

```yaml
case_builder:
  concepts: [<concept id>, <concept id>, …]      # KB concept ids to ground the brief
  expected_joins: [join_subject, join_actor]     # "ran, N matches" vs "not evaluated"
  projection_guard:                # did the leaves the decisive checks read actually ARRIVE?
    source: record                 # a LOGICAL name from this ruleset's sources: map
    probe_paths:
      - {path: element_counters.<leaf>, note: "…UNKNOWN for a data reason, not because absent."}
      - {path: security.<flag>,        note: "…the access-grant exclusion is UNKNOWN…"}
    empty_note: "No rows returned for the record source ({source}); … are UNKNOWN."
  scope_notes:                     # scope honesty (see scope_status, below)
    unverified: "Impact scope UNVERIFIED: the scope sweep did not run ({status})…"
    wider:      "The scope sweep found {count} impacted subject(s) the alert did NOT name…"
  action_templates:                # next-step WORDING (the engine picks the branch)
    subject_label: "<record> {subject}"
    contain: "{subject_label}: {verb} Target: {target} — <clause>, the record CREATOR…"
    scope_not_run: "{subject_label}: the scope sweep did NOT run ({status})…"
```

Every key is optional and no-ops when absent, so a pack shipping only `conditions:` still gets a verdict-only brief (`test_a_use_case_with_no_case_builder_block_still_gets_a_verdict_brief`). Placeholders are substituted with `str.replace`, never `.format` — these are YAML-authored prose and a stray brace must not raise.

**`action_templates` is the second leak, and it was subtler than the per-use-case module.** `derive_action_backbone` f-stringed one domain's record noun, two `§` clause numbers, and a follow-up instruction naming a specific internal team — one written procedure compiled into the engine, which the next use case would have rendered verbatim and been told to reverse a document it does not have. The **branching** stays in Python (which branch fires follows from the verdict; that is an engine decision); only the prose moved. The defaults in `_ACTION_TEMPLATES` are deliberately domain-neutral (subject / asset / actor / document) and `test_no_domain_procedure_wording_is_hard_coded_in_the_engine` greps them for domain terms — the leak survived months of tests because a test asserting "§4.1.1 appears" cannot tell whether the engine or the pack produced it.

**Scope discovery — the ruleset's `scope_discovery:` block.** A per-subject retrieval cannot answer "is the impact wider than the alert said?", because it is scoped to the subjects we already know; on one live incident that let a whole additional subject, carrying a still-active fraudulent document, go unreported. The pack declares which logical source holds the actor-scoped sweep and which of its fields carry the asset facts:

```yaml
scope_discovery:
  source: scope_sweep            # -> sources.scope_sweep -> a real source name
  asset_id_separator: "-"
  fields:
    subject:  <subject id path>
    asset_id: <asset id path>            # the id may be stored SPLIT across two columns
    asset_id_suffix: <asset id remainder> # ...so it's joined by asset_id_separator
    amount:   <asset amount path>
    status:   <asset state path>
    actor:    <acting identity path>
    timestamp: <row time path>
    version:  [<flat alias>, <struct path>]      # WHICH version this row is
    event_date: [<flat alias>, <struct path>]    # the ASSET's own event date
  # event_window_days: 1         # OPTIONAL pad on the DERIVED window; normally omitted
```

Every path is a *pack* declaration and every one of them is per-source knowledge — a pack's own
retrieval/verdict notes are where the measured values belong (`knowledge/<domain>/`), not here.

`build_scope_discovery(logs, spec, known_subjects, event_window=…, row_caps=…)` → `(assets, additional_subjects, status)`, filling `brief.impacted_assets` (generic `ImpactedAsset`) / `additional_subjects` / `scope_status`. Five shapes matter, all learned from live runs:

- **Each field is a LIST of candidate paths, first non-empty wins.** The LLM-generated sweep SQL *flattens* the exploded document into its own per-column aliases (one column holding the already-concatenated id, one for the amount, one for the state), so a mapping naming only the nested struct paths resolved every asset leaf to EMPTY — the sweep found the extra subjects but reported them with no id and no amount. The pack lists both the flat aliases and the struct paths.
- **One entry per `(subject, asset)`, not per row.** A versioned backend returns the same asset once per version that touched it, so keying on state made one asset three "impacted assets" and any amount total a 3× overcount; the merge also recovers `amount`/`currency` from whichever row carried them (a versioned store nulls them on superseded versions). Rows with **no** asset still count as subjects (the sweep uses an OUTER explode on purpose — a subject created but never acted on is in scope) and are reported separately as `N row(s) with no asset`.
- **`version` resolves "which state is CURRENT".** When the pack names a `version` field, the highest version's state is the asset's current one and the rest become history: `"T (current; was I -> V)"`. The engine stays generic — it knows which version is current, never what a state *means*. When no `version` is declared the states are unordered observations, so it renders `"V / T (order unknown)"` and refuses to claim currency. See **the version model** in `retrieval.md`: an asset reversed in one version and REISSUED in a later one would otherwise read as reversed — a live fraudulent document reported as already contained.

- **`event_date` separates incident scope from the actor's adjacent business, on a boundary DERIVED from the evidence.** The sweep *has* to span a window wider than the incident (an impacted subject can predate the alert), so it necessarily also returns that identity's ordinary work. The pack names the asset's **own** event date — the date the asset itself was issued, not the date its parent record was created — and the analyzer passes `analysis.event_time` in; assets outside the incident's boundary get `in_window=False`. They are still listed in `impacted_assets` (an operator reviewing an identity wants to see them, labelled) but are excluded from `additional_subjects` — the containment work-list — and counted apart in `scope_status`.

  **`_derive_event_window` computes that boundary per incident rather than reading a configured tolerance.** The evidence is the event dates of the documents on the subjects the alert NAMED — those *are* the incident — unioned with the alert's own window; the boundary is the span they occupy. A single-day alert scopes to that day; a fraud worked across four days scopes to four; no pack edit either way. `scope_status` states the result and its provenance (`incident window 2026-07-27 → 2026-07-27, 1 day(s), derived from the alert window + the 114 dated document(s) on the alerted subject(s)`) — a derived window that is never shown is indistinguishable from a hardcoded one to the operator. `event_window_days` survives as an optional **pad** on the derived span, an escape hatch for a domain whose episode provably spills past its own documents; the deployed ruleset omits it.

  Three deliberate choices inside it, each the safe side of a real trade-off. The span is the **outer** bound, not the contiguous run within it: requiring contiguity would stop a subject that also holds a document a fortnight earlier from stretching the window, but it writes off the far side of a fraud that paused for a day — erring wide costs a review, erring narrow leaves a document live. There is **no continuation heuristic** past the named subjects ("the gap is only one day, so probably the same episode" reads as smart and, on one live incident, pulls in exactly the next-day document the expert excluded as that identity's ordinary work). And comparison is per **calendar day** via `_day_number` (`toordinal`, deliberately *not* `_epoch`): an issue date is typically a bare date while the window is a timestamp, so an instant comparison would read a document issued at 18:15 as preceding a 16:45 alert, and a naive bare date would floor onto the previous UTC day on a host east of UTC.

  Absent the field, or with no dates anywhere to derive from, everything stays `in_window` — the analyzer never narrows scope on a boundary it cannot compute, since that is the direction that drops a live fraudulent document. On one live incident this is the difference between the expert's 2 extra subjects and AFIR's 11. **Any** in-window version keeps the asset in scope, and `report_generation` labels the rest in *both* the exported artifact and the narration prompt (with an explicit "do not count it in the exposure").

- **A CAP IS NOT A TOTAL, and this is the step where that costs the most.** `row_caps` ({real source: the cap that applied}) comes in from the caller — the sweep is the one step that can widen scope past the alert, and truncated at its backend's shared `max_results` it produced the identical wording an exhaustive sweep does. Measured live: **500 rows returned, 3509 available**, reported as `ran, N asset(s)`, with only the report's evidence-limits section calling it a lower bound. Truncation is detected exactly as `evaluate_verdict` detects it (`len(rows) >= cap`; `>=` because a backend may overshoot by a row), so one source cannot read truncated in the verdict and exhaustive in the brief. What a cap licenses is deliberately narrow: **the assets it did return are real and are reported unchanged** — it is the counts, and the "not named in the alert" set, that become floors (`at least N`). The status appends `_SCOPE_TRUNCATED` plus the cap and the row count; that name is a module constant *because its reader is somewhere else* — `_scope_step` selects a fourth action-backbone line (`scope_truncated`) by matching it, so an operator's next-steps list says "it named no subject beyond the alert only as far as it could see — the wider impact is UNKNOWN, not clean. Re-run with a higher cap" instead of `scope_clean`'s "the impact scope is bounded as reported". Those two are one word apart in tone and opposite in meaning, which is why the marker is not a phrase either side spells for itself.

**`scope_status` is three-way and never lies by omission:** `""` = no sweep declared (this use case has no widen-the-scope step, so there is no gap to report), `"not attempted (…)"` = a declared sweep that returned nothing → `brief.degraded` + an explicit "impact scope UNVERIFIED" note, `"ran, N asset(s) …"` = a real answer (windowed: `"ran, N asset(s) in the incident window across M subject(s); K NOT named in the alert; also X asset(s) … OUTSIDE the incident window"`). Same invariant as `join_status_from_transforms`: *ran and found nothing* ≠ *never evaluated*.
- `registry.py` — `get_analyzer(playbook_id, ruleset_key, pack)` → the one `UseCaseAnalyzer`, scoped to the matched **ruleset key** as its `use_case` (falling back to the playbook id, then `""`); lazy import avoids the cycle. There is no per-use-case class to select any more.

Wired into `CorrelationModule.analyze` **best-effort** after the verdict block → optional trailing `CorrelationResult.brief` (backfills `verdict` if the analyzer produced one and the verdict stage didn't); never fails the stage.

`ReportGenerationModule.__init__` takes `rag=` (main.py passes it — this was the only output stage with `rag=None`); `_render_brief_for_prompt(brief)` renders a compact factual block injected as a **second system message** ("AUTHORITATIVE INVESTIGATION BRIEF … may NOT be contradicted"), the `system_prompt` gains a verdict-authority rule (Exec Summary + Next Steps must match the verdict), and `_incident_reconstruction` appends `_asset_impact_lines(brief)` — which renders **both** the asset timeline **and** the sweep's per-asset current state (`<asset id> on <subject> — <amount> <currency> — state: I (current; was T -> V)`, with `<-- NOT named in the alert` on unalerted subjects). That per-asset block is deterministic on purpose: it previously rode only in the narration prompt, and on one rerun the LLM dropped it from the exported artifact entirely — "is this document still live?" is the single fact containment turns on, so it may not depend on the model choosing to mention it.

**The brief is a fixed-width window onto the pack, and both of its widths were wrong in the same direction.** Two numbers govern what pack prose actually reaches a stage, and neither announces itself:

- **`DEFAULT_CONCEPT_SNIPPET_CHARS` (220, `src/usecases/base.py`, overridable per ruleset as `case_builder.concept_snippet_chars`) is how much of each declared concept doc is injected — flat, from the top of the file.** There used to be *two* independent cuts: `collect_concept_refs` kept 500 and `brief_prompt` re-cut the same string to 220, so 56% of every snippet reached no prompt and both rulesets' comments documented the number that didn't apply. The consequence is not a thinner brief but an unreachable one: a concept doc exists to correct a specific misreading, and a prohibition placed past the cut is prose no stage can read. Measured on the doc written to stop a live `PAX`/`CHD` misreading — the distinction sat at char **1,303** of 4,122 and the prohibition at **1,528**, six times past the real cut, so the edit could not have taken effect at any brief budget. Hence the authoring rule: **one doc carries one load-bearing claim, in its first sentence.** Everything below that line is for a human reading the pack, and for RAG if it retrieves the file.
- **`DEFAULT_BRIEF_CHAR_BUDGET` (6000, `src/brief_prompt.py`) is the whole brief, and the render used to truncate it as one string** — so overflow deleted whole *trailing* blocks, and the tail is concepts → precedents → notes. The block that renders last is not the block that matters least. It now fits the concept snippets to the room left (max-min fair, shortest first) with the **title as a floor**: a narrator that knows a doc exists can ask for it, one that never saw the heading cannot. Measured on one live pack at four budgets: a 1,854-char trigger paragraph plus its concepts block reached **0 of 6** concept titles at 3,000, 6/6 titles and 364 of 1,320 snippet chars at 4,000, and the whole block at 6,000.
- **And the two narrating stages must read the same amount.** `report_generation` passed no budget at all (the bare 3,000 default) while `anomaly_detection` deliberately used 4,000 — so the *acceptance artifact* narrated from the weaker reading of the same ground truth. Both now take `DEFAULT_BRIEF_CHAR_BUDGET`, `report_generation` overridably as `report_generation.brief_char_budget`.

**A FAIL is a finding whether or not it is decisive, and `explanatory_fails` is the channel for the half that isn't.** See `docs/architecture/verdict-engine.md` — the short form is that `decisive_fails` cannot carry a non-decisive check, because that list *is* the verdict's reason set and `collect_precedents` matches cases on its ids.

`AnomalyDetectionModule.detect` injects a verdict framing system message and **deterministically clamps** any `confidence_score > 0.5` to 0.5 when a subject is FALSE POSITIVE (a hard guard — don't trust the model to comply). That clamp touches only *scores*, and only on FALSE POSITIVE, so a second hard guard covers indicator-driven fraud: when `brief.containment_gated` is set, `_fallback_recommendations` re-labels each anomaly-supplied action as `PROPOSED … REQUIRES EXPERT CONFIRMATION` instead of `REMEDIATE` — otherwise the anomaly's verbatim "immediately reverse the documents, suspend the identity" lands directly beneath the backbone's own "do NOT auto-void/lock/freeze". `pipeline_runner._summ_correlation` emits a compact `brief` view (use_case, verdict_summary, decisive_fails ids, join_status, precedent case_ids, degraded). All brief-consuming code is guarded against a MagicMock brief (list fields must be real `list`s).

**Precedents are pack content, not engine content.** A resolved investigation is seeded as one file under `use_cases/<name>/cases/`, and it is the only part of a pack that holds live incident detail — subject ids, identities, amounts — so in a domain pack that folder must be gitignored *before* the first file lands in it. `collect_precedents` reads whatever is there; the engine never names a case.

## RAG / unified knowledge layer

`src/rag/`: RAG is the *single, consolidated* knowledge base every LLM call draws on.

- **`KnowledgeRetriever` Protocol** (`protocol.py`, `@runtime_checkable`): the only surface the pipeline consumes — `async retrieve(query, filter_source=None, filter_type=None)` + `format_context(docs)`. `EnhancedRAG` and `PlaybookFallback` both satisfy it, so they're interchangeable with no module changes. **In tests, `isinstance(x, KnowledgeRetriever)`, never `isinstance(x, EnhancedRAG)`** (dual-import breaks the concrete check).
- **`KnowledgeSource` ABC** (`sources/base.py`): pluggable source emitting the homogeneous doc schema `{title, content, type, metadata}`. Implementations: `PlaybookSource` (the pack's playbooks), `DocumentSource` (local folders / UC Volume mounts, wraps `DocumentIngester`), `ConfluenceSource` (wraps `ConfluenceIngester`), `DatabricksKnowledgeSource` (rows of a UC table → docs via the SQL Statement Execution API, config-driven `title_col`/`content_col`/`type_col`/`where_clause`/`extra_cols`). **Add a new backend = one new `KnowledgeSource` subclass + one `rag.sources[]` entry + a branch in `main._build_knowledge_sources`.**
- **`KnowledgeOrchestrator`** (`orchestrator.py`): `build()` consolidates every enabled source into ONE `KnowledgeBaseManager` (one `documents.json` + one FAISS index, anchored via `knowledge_base_dir()`) — idempotent per source (`remove_documents_by_source` → `add_documents`), a failing source is logged+skipped, FAISS rebuilt once (`force_rebuild=True`). Then it constructs `EnhancedRAG`; **the embedding model loads lazily** (a `@property` in `enhanced_rag.py`, not in `__init__`), so a missing/undownloadable model surfaces at the orchestrator's first `update_index()` — caught to return a **`PlaybookFallback`** (deterministic keyword retriever over the playbook docs, no embeddings/FAISS/network). `build()` returns `EnhancedRAG` | `PlaybookFallback` | `None`; **playbook context is never lost** even offline.
- `EnhancedRAG` itself: persistent disk-backed KB, cosine-similarity `IndexFlatIP`, async retrieval, filtering, reranking. The resulting retriever is passed to `LLMClient` calls (understanding/anomaly — prepended as a system message) and to `CorrelationModule` (playbook matching for transform planning).
- Sources are declared in `main_config.rag.sources[]` (discriminated by `type`: `playbook`/`document`/`confluence`/`databricks_table`); if that list is absent, `main()` falls back to the legacy `document_paths` + `confluence` keys. `main._build_knowledge_sources()` is the factory.
