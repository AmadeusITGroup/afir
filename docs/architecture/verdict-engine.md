# The pack-driven verdict engine

`src/correlation.py`, `evaluate_verdict`. Given a ruleset declared by the knowledge pack, the
engine produces a **per-subject verdict** (the pack's own `labels`, conventionally a positive
finding / a negative one / `INSUFFICIENT DATA`), the identity a containment action would target,
an optional platform/mode determination, and a notification draft.

**Every domain specific lives in the pack; the engine is generic.** It dispatches on declared
condition `kind`s, ranks by declared polarity, and narrates declared vocabulary. It contains no
entity name, no source name, no field path and no rule. The domain-side counterpart of this
document is the pack's own README (a real pack's measurements, clause numbers and incident
history belong there, not here).

## Ruleset

`knowledge/<domain>/use_cases/<name>/rules.yaml` → `verdicts.<key>`. A flat-root `rulesets.yaml`
still loads for back-compat; `use_cases/` wins on collision. A ruleset holds `labels`,
`subject_entity`, a `trigger:` block, a logical→real `sources` map, a `conditions` list where each
entry is a generic `kind` plus a `decisive` flag, an in-scope point set, `lock_target`,
`platform_mode` prefixes, and the `notification` template.

Loaded via `KnowledgePack.rulesets` + `ruleset_spec(key="")` — **empty means the pack's FIRST
declared ruleset**, because a hardcoded default key in the engine returned `None` for every other
domain's pack. `ruleset_keys()` lists them in declaration order. An empty pack returns `None` and
the verdict stage no-ops.

### WHICH ruleset — the selection chain, and why every link is load-bearing

`_playbook_correlation_spec` (the keyword match) → the matched playbook's `use_case` →
`ruleset_key_for(use_case)` → `ruleset_spec(key)`. There is no ruleset-selection mechanism *of its
own*: the procedure an incident is judged under is decided entirely by which **playbook** the
correlation stage matched, and that is a keyword score over the LLM's own prose. Resolved ONCE and
shared by the verdict and the case builder, so narration and conditions cannot cite two different
procedures.

**The failure mode is a confident WRONG verdict, never a missing one.** Two procedures in one domain
read overlapping sources, so the losing procedure's conditions still resolve against real rows. A
measured live run picked the sibling procedure and produced `INSUFFICIENT DATA: 1`, 18 conditions
`unknown`, zero subjects, and a decisive line naming an attribute the incident never had — under
that procedure's labels, with every stage reporting success. Three defects in one scorer, each
independently sufficient, each fixed and pinned in `test_correlation.py`:

1. **A prose title went through the field-name tokenizer.** `_name_tokens` splits on `.` and `_`
   only — correct for a column, and on a multi-word title it returns the whole title as ONE token
   that no text can contain. Every title scored 0 and the match collapsed onto the join keys, which
   are exactly the generic ones (the acting identity, its org unit, its session) that every
   procedure in a domain shares. `_prose_tokens` is the separate splitter; keeping them distinct is
   the invariant, in **both** directions (prose-splitting a dotted path destroys the leaf).
2. **`initial_hypotheses` / `key_investigation_areas` are ANTI-signal for this question.** Naming
   the rival patterns is what a good hypothesis list does, so scoring them votes for every procedure
   at once. Measured over 27 labelled live runs: incident summary alone **27/27**, summary +
   investigation areas **17/27**, all three fields **10/27**. They are kept only as a tie-break,
   where mentioning a procedure at all is the last evidence available.
3. **Raw hit-counting let the longest title win.** A token every spec declares cannot discriminate
   between them, so tokens are weighted by inverse spec frequency `(n_specs - n_declaring) /
   n_specs`. Needs ≥3 specs to change any outcome (with two, a shared token adds the same constant
   to both), which is why the pinning test ships three.

A tie is still broken by declaration order — unchanged, but now the last resort instead of the norm.
And "no keyword match at all, with more than one spec" still returns `None` rather than guessing.

**The weighting has a corollary the scorer cannot fix, so the validator reports it.** Presence is
tested with `t in text` — a SUBSTRING test, which is what lets a title token match an inflected word
— and full weight goes to a token only ONE spec declares. Put together, a *function word* or a
two-character fragment owned by a single title is simultaneously the strongest evidence for that
procedure and evidence of nothing: `on` matches inside "session", `and` inside "outstanding", `is`
inside almost anything. `pack_validate._check_correlation_specs` warns per spec
(`spec-title-weak-discriminator`), gated on there being **more than one** spec — with one the score is
moot, the engine returns it regardless — and only where the frequency is 1, because a stopword every
title shares already scores zero and reporting it would be a permanent false alarm. A **warning**, not
an error: whether a short token discriminates is a claim about prose, and in some domains a two-letter
code is exactly the right one. It found a live instance in the fixture pack on the day it was written
(a title reading `Handler and POD Signer Divergence`, `and` owned by it alone), which is the argument
for the check rather than a note in an authoring guide.

### And the verdict is EVALUATED once, for the same reason it is selected once

`evaluate_verdict` was called **twice per run**: by `CorrelationModule.analyze` with `row_caps=` and
`keyed_sources=`, and again inside `UseCaseAnalyzer.analyze`, which is handed neither. Those two
arguments carry the only facts the rows themselves cannot state — whether a result stopped at the
backend's cap, and whether the query that returned nothing had actually constrained the acting
identity — so the second evaluation is *strictly weaker* by construction, and it is the one that
lands on the brief. **The brief is what the report and anomaly stages narrate FROM**, so the weaker
reading is the one the reader sees. Measured live: the verdict read a reference-list lookup PASS
("the identity is absent from the register", which is the answer), the brief carried the same
condition `unknown` / "source did not return", and the report then denied the finding in three
separate passages and closed with a next step asking for a check that had already answered — with
the appended verdict on the same page saying otherwise. `analyze(..., verdict=)` now takes the
computed one verbatim; evaluating locally is the fallback for standalone use (the analyzer's own
tests, a dry-run script), which is precisely why the pipeline path must not rely on it. Two tests
pin it, one per direction.

### A FAIL is a finding whether or not it is decisive, and only one side was gated on that

`UseCaseAnalyzer`'s triage sorted a fail into `brief.decisive_fails` only `if c.decisive`, while a
fraud-INDICATOR fail was appended to `brief.decisive_indicators` **regardless** of decisiveness. So
the one side that is *invertible* — an exclusion, whose FAIL argues the opposite of what "FAIL"
reads like — was also the only side a non-decisive check could fall out of entirely: no brief field
carried it, and the "READ THE DIRECTION / do not re-describe this as evidence OF fraud" prohibition
that exists for exactly this inversion never fired. Merging it into `decisive_fails` is not the fix:
that list **is** the set of verdict reasons, and `collect_precedents(decisive_reason_ids=…)` matches
comparable cases on its ids, so a non-decisive check placed there would be reported as one the
verdict class rests on. Hence a channel of its own, `brief.explanatory_fails`, rendered by
`render_brief_for_prompt` under its own heading with the same direction statement plus one more
clause: *"not decisive" is a fact about the verdict CLASS only* — it does not make the finding
doubtful, unestablished, or contradicted. Measured live (job `ed265d70`): a non-decisive exclusion
FAIL whose evidence is a set-valued attribute read across sibling rows reached the scorer with no
statement of direction, which reported the check's own columns as "contradictory evidence" at 0.68 —
into the narrative and a recommended action — while the report's own condition list two sections
earlier printed the FAIL and what it established. A non-decisive **UNKNOWN** is deliberately in
neither list: unlike a FAIL it established nothing, so there is no direction to state.

**A condition's MECHANICS belong in `shared/checks/*.yaml`,** and the ruleset imports them: a
condition entry is a `use: <file stem>/<check id>` line carrying only what this procedure decided
(`decisive`, `decisive_on`, `polarity`, `exclusion_kind`, `report_group`, `order`, `gate`, the
detail wording). `_resolve_check_imports` expands them at load and **an unresolvable `use:`
raises**, because a dropped condition is indistinguishable, in the report, from a source that
returned no rows. The split is what lets a second use case read the same fact: whether a record
carries a given element is a fact about the DATA, not about one procedure. Mechanics belong to the
library, weighting to the importer — and the boundary runs through *numbers* too: a number the
DATA dictates (`max: 1` for "one acting identity") is the library's; a number the PROCEDURE chose
(`max: "1h"`) is the ruleset's. See `knowledge.md`.

**`trigger:`** states WHAT THE DETECTOR FIRES ON — `description` verbatim plus
`min_assets`/`window`/`scope` as a parenthetical (`_trigger_sentence`, `src/usecases/base.py`),
carried on `AlertFacts.trigger`, printed by the report as `WHAT THE DETECTOR FIRES ON:` and
injected into the narration prompt as a **prohibition** ("do NOT state or imply any other
trigger"). It exists because the narration otherwise infers the trigger from whichever retrieved
fact looks most suspicious — a detector rule such as "a burst of ≥4 issued assets in 24h within
one org unit, explicitly *not* payment-based" is unreconstructible from the numbers. A pack
declaring no `trigger:` emits nothing: silence is correct, an invented trigger is not.

## Condition kinds

`evaluate_verdict(spec, logs, analysis, entity_map)` is **pure and deterministic (no LLM, no
IO)** and dispatches on condition `kind`:

Exclusion-style (the tripwires a procedure uses to clear a case):

- `element_absence` — a required-absent element, read through per-element counters and array
  leaves.
- `element_presence` — the mirror, for a REQUIRED element. **Not** an inverted `element_absence`:
  the two disagree about what a coverage gap means. One resolving path carrying the element PASSES
  immediately; a gap is only ever `unknown`, and a FAIL needs FULL coverage with everything empty.
- `field_equality` — two identity fields that should agree, with `normalize: identifier` giving
  prefix-matching so a duty-coded suffix still matches its base identifier while an unrelated
  login or display name does not.
- `time_gap` — an elapsed-time bound between two timestamps. Either side may carry a `where:`
  clause (`field` + `any_of` + `match: exact|contains`, clauses AND-ed) that **selects which rows
  of the source that side means** — see "Timestamps need a row" below.
- `record_absence` — no reversal/cancellation record of a named kind.
- `field_flag` — a boolean flag on the subject or actor.
- `distinct_count` — at most N distinct values (e.g. a single acting identity), over the primary
  `(source, field)` plus ordered `fallbacks` (the first that yields any value answers, and its
  origin is reported). `exclude_subject: true` subtracts the subject's own identity values —
  including every alias `_merge_co_identified_subjects` left on the survivor — because "did anybody
  ELSE use this?" cannot be spelled as a bound: `max: 1` is only right when the subject is always
  among the counted values, and `max: 0` cannot return the good news when it is. **Truncation is
  asymmetric in the opposite direction to `cohort_membership`**: a count OVER the bound stands (the
  values already in hand break it and more cannot mend it), while a count WITHIN the bound on a
  source that hit its row cap is a **FLOOR, not a finding** → `unknown` + `truncated_detail`, since
  the values that would exceed the bound may be in the rows that never came back. The cap is read
  from `row_caps` in `evaluate_verdict` and stamped per source, so a condition whose sources all
  answered in full is byte-identical to before the rule existed. Two other emptinesses are counts
  rather than gaps, and both need a stamp only the caller can make: every collected value was the
  subject's own (`0 distinct, the subject's own value(s) excluded`) and every candidate source
  returned **no rows at all** under a query that carried the declared scope
  (`_count_scope_resolved_empty`). Rows-but-no-values stays `unknown` — a returned record missing
  the counted field is not a source that held nothing.
- `cohort_membership` — "does the subject's own key appear on ANOTHER record in this cohort?" One
  source supplies both sides, so **no join and no second retrieval**; `value_mismatch` cannot
  express it, because the question is whether the overlap survives removing the subject's own
  record. `discriminator`/`discriminators` name what makes a row OTHER — a row is other only when
  it differs on EVERY one, since with a subject locator alone the fraud's own sibling records from
  the same burst become the evidence that clears it. **Truncation is asymmetric**: finding the key
  elsewhere survives a row cap, not finding it under one is no evidence at all → `unknown`. At the
  wider window the truncated case is the ordinary one (measured: 167 rows over 30 days, 490
  against a 500-row cap over 90).
- `route_membership` — the subject's scope points against the ruleset's declared in-scope set. The
  whole-row fallback recognises scope points by SHAPE, and a default is all it may be, so
  **`point_pattern`** lets a ruleset state its own. Undeclared → unchanged; a pattern that fails to
  compile falls back with a warning rather than sinking the verdict. `unknown` on a scope GATE is
  the expensive kind of wrong, since the gate is answered first.

Positive fraud-indicator kinds:

- `value_mismatch` — two value-sets that should overlap, with an optional `lookup` resolving one
  side through pack `data/*.yaml` via `from_entity`/`from`. JSON-array-string values like `["XY"]`
  are expanded before comparison.
- `value_matches_pattern` — a field's values against a regex allow-list or forbidden-list,
  including a `data_map` mode that scans a nested list of `{key,value}` pairs (the shape a
  key-value attribute bag arrives in).
- `delimited_field_mismatch` — TWO POSITIONS OF ONE DELIMITED STRING that must agree, compared
  **within** each value. `value_mismatch` is the wrong question here and dangerously so: across a
  5-party record with country set {A, B} on the left and {B, A} on the right the sets overlap
  perfectly while every party could be a mismatch — the pairing IS the finding. `separator` +
  `left_index`/`right_index` name the positions and the engine reads no meaning into either. A
  value is tested only when BOTH positions are non-empty: most live lines of this shape are the
  document-less form (the delimited string present but with both compared positions blank), where
  empty-vs-empty would silently clear a
  record with nothing on file and populated-vs-empty would allege a mismatch on one.
- `velocity_count` — distinct subject keys per actor within a window.

Compositional kinds — polarity-neutral, because what a FAIL argues for is the importing ruleset's
declaration and not the kind's. Each is detailed under "The compositional vocabulary" below:

- `all_of` / `any_of` / `none_of` — one condition whose finding is a combination of its
  `children`, under three-valued logic.
- `numeric_compare` — aggregate a field, then compare the number, with the truncation direction
  decided per aggregate. `baseline:` makes the bound a multiplier of a second computed aggregate.
- `event_order` — an ordering claim between two timestamp sides. **`time_gap` is not one**: it
  compares magnitudes and accepts either order.
- `value_equivalence` — how many values are *the same thing* under a **pack-declared** equivalence
  form, either as a collision (the largest class) or as a targeting question (equivalent to a
  declared anchor).
- `stub` — a documented placeholder for a check the data cannot yet answer; always `unknown`, with
  a stated reason and no data access. Better than omitting it: the report then shows the
  requirement and says it is unanswered, instead of quietly not testing it.

  **But a stub is not a gap in THIS incident's evidence, and every reader that COUNTS unknowns has
  to say which it is.** Both land on `result="unknown"`; the remedies are opposite — a retrieval gap
  is a re-run or a query fix, a stub is authoring work no re-run can reach. `STUB_OBSERVED`
  (`models/pydantic_models.py`, beside the model whose field it describes, because `correlation` and
  `report_generation` reach that module by the two different import styles and a `str` is the only
  thing that compares equal under both identities) is the marker, and three readers use it:
  the evidence floor's own sentence excludes stubs from its denominator and names them separately
  (`N of 8`, not `N of 10`, plus "a further 2 declare no data path at all"); the report's gaps
  section splits the heading count and marks the stub's own row `[NO DATA PATH …]`; and
  `pack_validate`'s `unreachable-evidence-floor` bounds `min_evaluated_to_clear` against what can
  ever resolve — measured against the raw list, a floor of 10 on eleven conditions with two stubs
  passes validation and then clears nobody on any run, which is the state that check exists to name.
  Marked, never dropped: the procedure does require the check. Silent where no stub is declared.
  **The verdict is never at stake** — a stub is non-decisive, so it reaches neither
  `has_decisive_unknown` nor the floor's numerator — and the split must NOT be widened to
  `result == "unknown"`: `_unevaluated_labels` documents that mistake, which is why the marker looks
  narrower than it should.

**Every kind's finding wording is pack-declarable** — `pass_detail` / `fail_detail` /
`unknown_detail`, each falling back to a generic engine sentence naming only what the engine
resolved. Those fallbacks once carried one domain's vocabulary (a `field_flag` `unknown` naming
that domain's automation class, a `record_absence` naming its reversal record), which is the same
leak as a clause number in the containment block: a domain literal asserted on every other
domain's report.

**Reading a BOOLEAN is not reading a scalar.** `resolve_path`'s `_scalar` excludes bools by design
so a flag is never mistaken for a join key — right for key discovery, wrong for reading a flag.
`_flag_leaves` / `_collect_any` / `_first_present_flag` keep bools (APPENDED, never substituted, so
no string-valued check sees a different value set). Without them a check whose vocabulary includes
`true`/`false` is not merely missing a value: the predicate is unmatchable, so it PASSES on every
row and reports "no forbidden value" over a source where the flag is set on all of them — a check
that *cannot* fire, indistinguishable in the report from one that had nothing to find.

**An allow-list PASS is the one branch that CAN quote its evidence, so it must.**
`value_matches_pattern`'s two modes clear a subject for opposite reasons, and only one of them
has anything to show. A `forbidden` PASS is an *absence* — the finding is that no value matched,
so `observed` states the size of what it read (`none in N value(s) from …`) because there is
nothing to name. An `allowed` PASS is the exact inverse: every value it read is present and each
one is a REASON, so "all allowed" states the conclusion and throws away the evidence for it. The
FAIL side always named the offending values, which left a report that could say which value
convicted a subject and never which one cleared it — and on a reference source that is the entire
content of the finding. So the allowed branch renders the values it saw, sorted, capped at four
like every sibling render with the total beside it so the cap is visible rather than silent. Live:
an office-type check cleared its subject reading `all allowed` over a two-row profile whose single
value was the answer, and the scope gate deciding whether the procedure applied at all said the
same of a document number it had in hand. The *detail* stays the pack's `pass_detail` in both
modes — this is the `observed` field, which is the engine's own account of what it resolved.

**An absent FIELD is not an absent VALUE** (`platform_mode`): reading "no value came back" as a
determination turns a projection gap into a finding, and platform determinations select
non-interchangeable action sets, so the gap picks the wrong containment. Measured on a live run —
the projection carried no such leaf at all, so every declared path was never retrieved and the
code reported the negative determination with full confidence. Same rule as `element_absence`'s
coverage guard: absence may only be claimed for a path actually read. A blank leaf is likewise
not a value.

**And an UNDECLARED dimension is not an undetermined one.** `_detect_platform_mode` answers
`("unknown", "unknown")` for a ruleset that declares no `platform_mode` block, which is correct —
it has nothing to go on — but the two consumers of that answer used to fire on it regardless, so
the `platform_mode=` note (hence the report's `Platform: unknown` line) and
`lock_target["platform"]` (hence the engine's undetermined-platform sentence, in the one section
that names a real account for action) were published by every procedure whether or not it has a
platform question at all. Measured 2026-08-16 on the installed domain pack: **five of its six
rulesets declare no `platform_mode`**, and two had recorded the observation in their own
`reporting.yaml` — one of them describing the engine as it ought to work, which is how the defect
surfaced. Both are now emitted only behind the block that produces them, and the report's no-verb
branch tests the **missing verb** rather than an unknown platform, so a declared platform resolving
to a class the `actions:` map omits no longer prints a containment target with no action and no
explanation. The rule generalises past this one key: a claim the engine can only make from a pack
declaration must not be published where the pack declared nothing, because `unknown` reads to
every consumer as a gap in the evidence.

**And an UNDECLARED NUMBER is not a small one.** The sibling defect, in the direction that
matters more: where the rule above published an engine-invented *gap*, this one published an
engine-invented *finding*. Four declarations whose entire content is a number the procedure
chose were read with a literal fallback — `velocity_count`'s `max` defaulting to **3**,
`distinct_count`'s to **1** (and to **0** on one of its four branches, so the same undeclared
condition could PASS or FAIL over the same rows depending on which path it fell down),
`time_gap`'s window to **one hour**, and `indicator_threshold` to **2**. Each then printed the
invented figure as the procedure's own measured finding: `4 distinct values, more than the 1
allowed`, over a bound nothing in the pack states. `time_gap` was the subtlest, because its
grammar is narrow (`<N>h` / `<N>d` / `<N>m`): a ruleset declaring `1 week` or `PT48H` — neither
unreasonable, both unparseable — got the hour silently, so every interval over an hour FAILed.
And `indicator_threshold` was the most consequential, being the one omission that could
*manufacture* an accusation rather than withhold one: a ruleset declaring indicators and
forgetting the key reached its FRAUD label on any two of them, narrated as corroboration.

All four now report `unknown` (`_declared_bound`, `_NO_BOUND_EXPECTED`), the rollup refuses to
weigh indicators it has no rule for and says so in a note — `indicator_threshold=undeclared, N
corroborating indicator(s) fired and were not weighed`, because "no indicator fired" and "three
fired and nothing counts them" otherwise render as the same unremarkable clear — and
`pack_validate` errors at load, where the author can still supply the number
(`condition-bound-undeclared`, `indicator-threshold-undeclared`, plus
`indicator-threshold-vacuous` for the `0` that accuses every subject because `len([]) >= 0`, and
an `-unreachable` warning for a threshold no indicator count can reach). A
declared-but-unreadable value lands in the same place as an absent one: `int("two")` used to
propagate out of the evaluator and take the whole verdict with it, which a typo in a YAML scalar
must not be able to do. Measured 2026-08-17: all 20 counting conditions, both intervals and all
8 rulesets across both installed packs declare their own, so **no shipped verdict moves** — every
one of the four was purely latent, waiting for the next pack to leave a key out.

## The compositional vocabulary

**A new fraud pattern should be new YAML, not new Python.** The premise that got measured first was
that the kind list is on a treadmill, and it is not: 10 of the original 14 kinds landed in the single
commit that created this engine, 2 more within 24h, and only 2 in the months after — while the packs
grew to 12 rulesets and ~114 conditions authored as `use:` imports plus weighting overrides with zero
Python. The vocabulary was saturating. What it was **not** was compositional: there was no
`all_of`/`any_of`/`none_of`, no generic aggregate-then-compare, no ordering primitive, and no way for
a pack to say when two textual values are *the same thing*. Genericity comes from a closed operator
set that composes — the way SQL is domain-agnostic — not from being extensible at runtime.

Runtime LLM generation of the checklist was considered and **rejected on grounds independent of
determinism: an LLM can only emit kinds the engine dispatches.** For a pattern outside the vocabulary
it either picks the nearest wrong kind (a confident wrong verdict) or emits an unrecognised one, which
returns `unknown` — INSUFFICIENT DATA, from a typo. It would also forfeit `pack_validate`'s whole
error set and `git log` on the pack as the audit trail for why a case was decided. The LLM's role
stays at **authoring time** (`src/knowledge/pack_assistant.py`, `docs/architecture/knowledge.md`),
where the new kinds reach it for free through `pack_summary`'s `condition_kinds`.

Two rules run through everything below, and both are silent-wrong-verdict rules rather than crashes:
**`unknown` never reads as `pass`**, and **an undeclared parameter is not a small one** — every new
key follows `_declared_bound`'s precedent and reads `unknown` rather than falling back.

### `all_of` / `any_of` / `none_of` — one condition, many children

`children` is a list of full condition dicts, each of which may itself be a `use:` import or a
nested composite (`_resolve_check_imports` recurses, so an unresolved child keeps `kind: ""` and
falls through to the unrecognised-kind branch). Three-valued logic:

| kind | `pass` | `fail` | `unknown` |
|---|---|---|---|
| `all_of` | every child passes | any child fails | no fail, ≥1 unknown |
| `any_of` | any child passes | every child fails | no pass, ≥1 unknown |
| `none_of` | every child fails | any child passes | no pass, ≥1 unknown |

A decided result may coexist with unknown children **when those children cannot overturn it** — the
same principle as `_quantify_over_members`. `observed` is assembled from the children that settled it
(`_PIVOTAL_CHILD`), because the reader needs the deciding ones and not the ones that came along.

**The parent produces exactly one `ConditionCheck`**, so children are not report lines and
`_conditions_section` needed no change. The parent therefore owns **every** weighting key
(`decisive`, `polarity`, `report_group`, `label`, `fail_detail`, …): `mk()` reads them off the outer
dict, so a child declaring one is silently inert — hence `composite-child-weighting` is a validator
**error** rather than a warning. Fewer than two children is `composite-children`; nesting past
`_MAX_COMPOSITE_DEPTH` (5) is `composite-too-deep`, and at runtime the depth bound returns `unknown`
rather than recursing until the interpreter, not the pack, ends the verdict.

### The parent→child rule: run-level inherits, source-derived is computed per child

`evaluate_verdict` stamps underscore-prefixed keys onto each condition before evaluation, and getting
this wrong for a child is a **silent wrong verdict**. The two categories must not be conflated:

- **Run-level context INHERITS** — `_subject_identity_values`, `_routes`, `_lookup_data`,
  `_lookup_entities`. These are per-subject or per-pack facts. A `distinct_count` child with
  `exclude_subject: true` and no `_subject_identity_values` fails to subtract the subject and
  inflates its count by one; a `route_membership` child with no `_routes` evaluates against an empty
  set and always fails.
- **Source-derived facts are COMPUTED PER CHILD, from that child's own sources, never inherited** —
  `_scope_note`, `_scope_resolved_empty`, `_count_scope_resolved_empty`, `_count_truncated`,
  `_cohort_truncated`, `_subject_anchor_unresolved`. Inheriting a truncation flag from a *sibling's*
  source turns a sound check into `unknown`: a **false INSUFFICIENT DATA**, which is the failure mode
  this system is least able to see.

`_condition_sources` walks `children` recursively so the scope-note logic fires for a source only a
child reads. A `baseline:` population is **deliberately absent** from that walk (see below).

### `numeric_compare` — aggregate, then compare

One kind subsuming the counting family for new use cases, in the standing mechanics/weighting split:
`source`, `field`, `aggregate`, `where`, `group_by`, `fallbacks` belong in `shared/checks/*.yaml`;
`operator` and `bound` belong to the importing ruleset. **`bound`, not `max`**, because `max` states
the wrong thing under `operator: ">="`. Aggregates: `count`, `distinct`, `sum`, `min`, `max`, `avg`,
`median`, `ratio`, `mode`, `mode_share`. Bounds are read through `_declared_number`, a float-safe
sibling of `_declared_bound` — `bound: 0.25` coerced with `int()` becomes `0` silently, which is why
the incumbent kinds keep the integer coercion (changing it would move existing verdicts) and
`pack_validate` errors on a non-integer bound for them instead.

`group_by` aggregates per group and compares the **largest**, mirroring `velocity_count`'s
busiest-actor shape: the maximum answers both readings of an upper bound (`>`/`>=` asks whether ANY
group exceeds it, `<`/`<=` whether EVERY group is within it). Under `==` no group answers, so it
declines with `numeric-compare-group-equality`. `ratio`'s denominator is the row set **before**
`where`, so the clauses selecting the numerator use the same vocabulary as every other kind; an empty
denominator falls through to `unknown` and never raises.

**Truncation is asymmetric per aggregate AND per direction.** A truncated read is a *bound* on the
real value, so the conclusion stands only where more rows cannot flip the predicate (`_holds_when`):

| aggregate | more rows move it | conclusion survives truncation |
|---|---|---|
| `count`, `distinct`, `max`, `mode` | up only | `>` / `>=` true, or `<` / `<=` false |
| `sum` | up only **while nothing is negative** | as above — a column carrying credits is non-monotone |
| `min` | down only | `<` / `<=` true, or `>` / `>=` false |
| `avg`, `median`, `ratio`, `mode_share` | either way | never |
| any | — | `==` never survives: any movement breaks equality |

Under `group_by` only an `up` direction carries through, because the largest of several minima can
move either way as rows arrive. **Zero rows split two ways**, on flags that already existed: a keyed
source that returned nothing (`_count_scope_resolved_empty`) is a real zero for `count`/`distinct`/
`sum` and compares normally, while `min`/`max`/`avg`/`median` and both modal aggregates read
`unknown` — nothing has no minimum, and no most frequent value either. Rows present but the field
absent is a **projection gap** and stays `unknown`: a column that never arrived is not a zero.

`exclude_subject` is **not** available here and is a validator error
(`numeric-compare-exclude-subject`). It is not a filter but a three-way reading of what an emptiness
means, pinned by `distinct_count`'s own tests; silently accepting the key on a kind that ignores it
is the inert-declaration shape this engine keeps paying for.

### `mode` / `mode_share` — the aggregate whose answer is a VALUE

`count` and `distinct` already read text, but `sum`/`min`/`max`/`avg`/`median` need numbers, so a
concentration question could only ever be asked about a value the pack **named up front** (`where` +
`ratio`). "Whichever value is most frequent, and what share it holds" was not expressible — and that
is the question a targeting pattern is. `mode` is the frequency of the most frequent value;
`mode_share` is that frequency over the row count. Both compare numerically, so nothing else in the
kind changes.

**The winner's identity is the finding, so it reaches `observed`** (`_modal_note`): a report reading
`mode = 34 (>= 20)` states a concentration and names nothing, which is the same defect as a label
printing a requirement instead of a finding. **Ties resolve deterministically or the verdict is not
reproducible** — highest frequency, then the lowest key lexicographically; a dict-iteration winner
would name a different value on two evaluations of one row set. Under truncation the top frequency
only rises so `mode` stands upward, but **the identity is non-monotone** — rows that never came back
can carry a different value entirely — so it is reported as provisional rather than compared, and
`mode_share` is non-monotone in both terms and is always `unknown`, exactly like `ratio`.

### `baseline:` — when the bound is itself a computed aggregate

`baseline: {aggregate, source, field, where, group_by, per}` plus `bound` as a **multiplier** — the
3×-cohort-median shape. It rides on `bound` rather than a nested key of its own because the
one-level `use:` merge replaces a mapping wholesale, so a nested multiplier could not be re-weighted
without restating the mechanics.

**The truncation contract is hard, and it is the parent→child rule one level down.** The baseline
population's completeness is a fact about the *baseline's own* source: `_preprocess` stamps
`_baseline_rows` / `_baseline_gap` from that source, and the population is deliberately excluded from
`_condition_sources` — a subject-scoped truncation says nothing about the cohort, and a population the
subject has been filtered out of is not a baseline. If the baseline source was truncated or
scope-resolved-empty the condition returns `unknown` with its own sentence; there is never a
comparison against a partial baseline. `median` and the percentile family need the full population by
definition, so they are only ever available under a complete read. Seven validator errors bound the
declaration (`baseline-shape`, `-source`, `-field`, `-aggregate`, `-per-aggregate`,
`-ratio-unfiltered`, `-unread-kind`), the last because a `baseline` on a kind that does not read it is
a threshold the author believes is relative and the engine treats as absolute.

### `event_order` — an ordering claim, which `time_gap` is not

`time_gap` compares **magnitudes** and accepts either order (`timedelta(minutes=-5) <= gap <=
max_delta`, over earliest-of-each-side), so a check labelled *"the reversal **followed** the
booking"* passes on a reversal timestamped up to five minutes **before** it — the ordering word in
that label was never verified. `knowledge/<domain>/shared/checks/refund_void.yaml` and
`ticketing.yaml` both carried that shape, and the pack's adoption of this kind is what makes those
labels true. Both sides keep `time_gap`'s shape (`start`/`end`, each with an optional `where:` row
selector), so a check converts **by kind**.

`relation` is one of `after` / `not_before` / `before` / `not_after` — the two inclusive forms exist
because simultaneity is a real reading and a day-granular column produces it. `quantifier` is `every`
or `any`, and **neither key has a default**: the relation IS the check, and `every` and `any`
disagree over the same rows. The quantifier picks the **deciding pair** rather than a per-side
reduction the pack would have to declare twice — a universal is settled by the hardest pair, an
existential by the easiest, and which timestamp that is follows from the relation. An absent
`tolerance` is **exact**, never fabricated slack: `time_gap` forgives five minutes of clock skew from
a literal in `src/correlation.py`, which is a deployment's judgement about its own sources taken in
Python where no pack can restate it. An unparseable `tolerance` declines rather than comparing at
zero slack, which would answer a stricter question than the one declared.

### `value_equivalence` and pack-declared equivalence forms

**The problem.** A pattern can be "several textual values that are not equal but are *the same
thing* for the purpose being adjudicated". The engine cannot own that relation: what counts as the
same thing is a deployment's own rule, it differs between deployments of one domain, and it is
exactly the domain judgement that belongs in the pack.

**Measured: there were FOUR normalisation seams and they disagreed with each other.**
`_rows_matching`'s `normalize: identifier|exact`, `field_equality`'s inline read of the same key, and
`_values_match`'s `identifier|id_suffix|exact` were fixed engine opinions — one hardcoding a minimum
length of **6** — and their *defaults* differed: `_values_match` accepted containment where the other
two demanded equality, so **one pack key meant two relations depending on which kind read it**. The
fourth was `equivalent_values` on `delimited_field_mismatch`, a per-condition substitution table with
its own private `_norm` — the `map` operation below, already in production for a code written alpha-2
in one field and alpha-3 in another, and reachable from exactly one kind. No seam was reachable from
any counting kind, so no aggregate could be taken over an equivalence class.

**The governing rule: the engine supplies text OPERATIONS and owns no relation of its own. Whether
"the same thing" means exact equality, a substring, a shared prefix of N characters, or a fuzzy match
within N edits — and what N is — is declared by the pack.** The engine's contribution is that every
one of those is *available*, composable, and reported the same way.

Forms live in the pack at `shared/equivalence_forms.yaml`, loaded like the check library — **not**
`value_forms`, which already names the entity glossary's surface-form concept. A form is a named
pipeline over two operation families, because "the same thing" gets asked in two shapes:

- **Canonical projection** (value → key; two values are equivalent when their keys are equal):
  `case`, `keep`, `strip`, `prefix`, `suffix`, `tokens` (a mapping declaring `split`, optionally
  `order` and `take`), `collapse_repeats`, `map`. Transitive by construction, O(n).
- **Pairwise comparison** (two keys → bool): `exact`, `contains`, `shared_prefix`, `shared_suffix`,
  `edit_distance` (Levenshtein), `min_overlap`. Not transitive, O(n²).

A form may compose both — project first, then compare — so "fold case, keep letters, then accept
within one edit" is one declaration. The three hardcoded modes are all reproducible as YAML;
`id_suffix` is `project: [{case: upper}, {keep: alnum}]` + `compare: {contains: true}` +
`min_length: 6` + `linkage: single`, which is the point: its magic 6 and its mutual-containment rule
were a pack's decision taken in Python, where no pack could restate it.

**The engine defaults nothing: not the relation, not a threshold, not a minimum length.** An
undeclared parameter yields `unknown` (a `FormError` carrying the authoring sentence), never a
fallback. A silent fallback to `exact` is the one wrong answer that still looks like a working
check — the condition returns a well-formed count, the report prints it, and the relation it was
counted under is not the one any pack declared. That includes the form NAME: `_resolve_form` refuses
a name no pack file declares rather than reaching for the incumbent key, because answering a
different question under the authority of a declaration nobody wrote is worse than declining.

**A pairwise relation is not transitive, so the pack declares the shape of the question** — the one
place the engine would otherwise have to pick a policy. Two shapes, no default:

- `anchor: {source, field}` on the condition — compare every value to a declared anchor (commonly the
  subject's own). Deterministic, nothing to cluster, so no linkage is needed; an anchor carrying no
  readable value is `unknown`, since there is nothing to be equivalent TO.
- `linkage: single | complete` on the form — grouping into classes. **Required** when there is no
  anchor, because single and complete linkage produce different classes over identical rows and
  therefore different verdicts. Single linkage is connected components (union-find); complete is a
  greedy clique cover in sorted key order — deterministic, which is what the verdict needs, not
  minimal, which nothing here claims.

**The trap the form validator exists for: a form that reduces too far collapses unrelated values,
which fabricates a finding.** `prefix: 3` against a two-character value yields the value itself;
`keep: alpha` against a numeric value yields the empty string, and every such value then matches
every other. So a form may declare `min_length: N`, and a value the form cannot read is
**unresolvable — excluded, counted, and reported**, never canonicalised to `""` and never passed
through unchanged. Same rule as `encoded_fields`' absent part being omitted rather than `""`.
`_canonical_key` returns `None` for that case and every caller tests `is None`; the empty-projection
path matters on its own, because `min_length` is only a warning and a floor-less shortening form is a
pack the validator lets through.

**The kind.** `value_equivalence` takes `source`, `field`, `form`, plus `operator` and `bound` in the
usual split. One question, two framings: without an anchor, how many values fall into the largest
equivalence class (a *collision*); with one, how many are equivalent to it (a *targeting* question).
`observed` names the class key and its members, per `mode` — the identity is the finding — and
the unresolvable count rides beside it, because **a class of 4 drawn from 60 values of which 55 the
form could not read is not the finding it reads as**. A class can only grow, so the conclusion stands
upward only. **The pairwise cost is reported rather than capped**: a cap that truncates a class is
the fabricated-finding shape one layer down.

**`normalize: <form>` is also available wherever a counting kind reads a field**, which is what makes
the pattern expressible compositionally as well as directly: `distinct_count`, and
`numeric_compare`'s `distinct` / `mode` / `mode_share`. Those are the aggregates a form actually
changes — `count` reads text but counts rows, so a form leaves its answer untouched and the note is
suppressed rather than attributing the number to a projection that never ran
(`equivalence-form-inert-aggregate`, a warning). **`normalize` is one key over two vocabularies**:
`field_equality` and `_rows_matching` read it as a fixed mode, so a form name there is an **error**
(`equivalence-form-on-mode-seam`) — it would silently mean nothing. A form declaring `compare` is
**refused** rather than half-applied on a `normalize:` seam (`_projection_form`), since those seams
canonicalise and have nowhere to put a pairwise pass.

**Validation is where a bad form has to be caught**, because every failure here silently changes the
relation instead of erroring. Errors: an unrecognised projection or comparison op, two ops in one
step, a missing required parameter, an unknown token order, a `compare` that is not a mapping, an
anchorless pairwise form declaring no `linkage`, a `linkage` the engine does not implement, an
unreadable `min_length`, a pipeline that can only ever produce the empty string, a `form` no pack
file declares, and a pairwise form named on a kind with no pairwise reading of its own
(`equivalence-form-not-transitive`). Warnings: an unknown form key, a `linkage` on a transitive form
(inert), a form named on a kind that does not read it, an inert aggregate, and **`min_length`
absent** — naming the collapse risk, because a form over a fixed-width code does not need one. The
diagnostics carry the codes listed in `docs/architecture/knowledge-pack-authoring.md` §Equivalence
forms; each has a test of its own in `tests/test_pack_validate.py`, and each of those was
mutation-verified against the code it reports.

## Polarity and rollup

**Condition POLARITY** (`ConditionCheck.polarity`, default `exclusion`): an *exclusion* FAIL
argues for the negative label; a *fraud_indicator* FAIL is positive evidence OF the finding.

**EXCLUSION KIND** (`ConditionCheck.exclusion_kind`, default `heuristic`) — *the* discriminator,
and the one most easily got wrong:

| | Test | Rollup |
|---|---|---|
| **`categorical`** | An **ATTRIBUTED FACT**: names an actor and a date, so a reviewer can look it up and see the same thing | Ranked **ahead of** the indicator vote |
| **`heuristic`** (default) | An **INFERENCE** about the record's shape (element counts, a clock difference, a location class) | Indicators can outweigh it |

Once an attributed fact is on the record the behavioural indicators lose their *meaning*, not
merely their weight: "an unusual contact address + an unusual payment form + a burst of records" is
what a legitimate organisation's own automated integration looks like when it operates.

**Classify on the evidence CLASS, never on how many indicators happen to be firing.** A count of
behavioural signals cannot promote itself past an attributed fact, and an attributed fact needs no
corroboration to be checkable. Getting this wrong costs verdicts in *both* directions —
under-classifying buries the exculpatory fact, and over-classifying an inference would make the
engine structurally incapable of ever finding a non-textbook case. A pack's regression tests should
pin both directions plus a grep-level guard on the classification itself, because either test alone
is satisfiable by a wrong rule.

**Rollup with polarity + indicator tiering:**

0. a **decisive categorical-exclusion FAIL** → the negative label, ahead of everything below;
1. a **decisive fraud_indicator FAIL** OR **≥ `indicator_threshold` (default 2) non-decisive
   indicator FAILs together** → the positive label. This OVERRIDES the *heuristic* exclusions, and
   it is the fix for an exclusion-only engine that structurally misses any case which is not
   textbook. **Measured caveat, worth knowing before tuning anything here:** across one pack's
   whole stored incident history, the number of positive verdicts reached with NO exclusion FAIL at
   all — the tie-breaker case this tier was documented for — is **zero**. Every positive verdict it
   ever produced came from indicators overriding an exclusion FAIL, so step 0's classification is
   what actually decides, and a mis-`heuristic` exclusion is not a near-miss: it is the whole
   verdict;
2. else a decisive **exclusion FAIL** → the negative label;
3. else a decisive UNKNOWN → `INSUFFICIENT DATA`;
4. else all decisive PASS → the positive label.

Which indicators are standalone-*decisive*-eligible is the pack's call, and restraint here
prevents the mirror-image over-fire: with every corroborator non-decisive and `indicator_threshold`
at 2, a single unusual-but-legitimate signal alongside an exclusion FAIL stays on the negative
label.

`evaluate_verdict(..., pack_data=None)` takes the pack's `data/*.yaml` lookup tables (loaded into
`KnowledgePack.pack_data` from `use_cases/*/data/` + flat-root `data/`; `use_cases` wins on
collision, guarded by `is_dir()`).

An indicator-driven positive verdict recommends **EXPERT CONFIRMATION before containment** (no
automatic reversal or lockout) plus a scope sweep — surfaced by `derive_action_backbone`, the
`_verdict_section` indicator callout, the brief's `decisive_indicators` (+
`_render_brief_for_prompt`), `pipeline_runner._summ_correlation`, and the pack's own
indicator-driven notification variant.

### Timestamps need a row (the `where:` clause on a condition side)

An elapsed-time condition assumes both its timestamps exist as columns. Often the second one does
not: a document's own date column may be day-granular, and a nominally adjacent time column may be
a *deadline* that can PRECEDE the record's creation. What a versioned store does have is versions —
in a version-numbered source, **the moment a thing came into existence is the write time of the
first version whose payload shows it existing**. So the condition becomes
`start = record.creation_time` → `end = record.modification_time WHERE <status field> IN (<issued
value>)`. Measured on a live record: versions 0–5 carry no document, version 6 is the first at the
issued status, written at `16:11:34` against creation `16:09:00` — a **2m34s** gap, so the
condition fires with no help from any access log.

The generic engine addition this needed is one clause, not a per-domain branch: **a condition side
names a source AND, optionally, a predicate selecting which of its rows that side means**
(`_side_rows` in `correlation.py`, applied to both sides of `field_equality`, `value_mismatch` and
`time_gap`). Without it a side means "all rows of the source", which silently averages a versioned
history. `match` defaults to **`exact`**, because status vocabularies are single letters — under
substring matching `T` would also match any value merely containing it, and single-letter codes are
common enough that permissive-by-default is the wrong default here.

### Scope discovery — `scope_discovery:`

Recommending a wider sweep is not the same as *performing* one. Every subject-keyed source is hard
bound to the subjects the alert names (its query guarantees mandate `IN (<the alert's subjects>)`),
so a fraudulent actor's OTHER records are structurally unreachable — a containment miss, not a
scoring error, and the failure mode is a still-active asset left unactioned because the report
said the scope was bounded.

The ruleset's `scope_discovery:` block maps a **sweep source** — the same underlying table filtered
on the *actor* (`<creator field> LIKE '<identity>%'` + org unit) over a multi-day window instead of
on the alert's subjects — onto generic asset fields. `UseCaseAnalyzer.build_scope_discovery` diffs
the result against the subjects the alert named and fills `brief.impacted_assets` /
`additional_subjects` / `scope_status`. **`scope_status` is three-way** — *found extra subjects* /
*ran clean* / *did not run* — mirroring `join_status_from_transforms`: a sweep that never ran sets
`brief.degraded` and says the scope is UNVERIFIED, because presenting an unverified scope as
bounded is the exact failure it exists to prevent.

Four corrections to that block, all from live runs and all the same shape — *the sweep succeeded
but its facts were read wrong*:

- **Every field is a candidate list.** The generated SQL flattens an exploded document to plain
  aliases, so struct-only paths resolve EMPTY. The sweep's hint pins those aliases.
- **A `version:` field is required on a versioned source.** The same asset returns once per
  version, so without one the analyzer cannot say which state is current and a
  reversed-then-reissued asset reads as reversed. With it an asset reports `T (current; was I ->
  V)`; without any version field the analyzer says `(order unknown)` rather than guessing.
- **The scope BOUNDARY is the asset's own event date, not the record's creation date.**
  `event_date: [<alias>, <struct path>]` narrows the sweep's deliberately-wide *scan* window down
  to the incident's own days, because the procedure's step is "review this actor's activity on the
  alert day". Without it, the same actor's ordinary business on neighbouring days was reported as
  extra fraud scope — 11 "additional impacted subjects" against the expert's 2.
- **A row with NO event date is a third state, `unknown` — not in-window.** A versioned store
  returns a record's pre-issuance versions with a null document date, and one asset issued nine
  days earlier rode 15 such versions of an alert-day record straight back into scope. Undated
  evidence only decides when there is no dated evidence.

The window itself is **derived per incident, not configured** — the span of the alert's window plus
the event dates of the documents on the subjects the alert named (`_derive_event_window`, mechanics
in `knowledge.md`). A single-day alert scopes to that day; an actor who worked across four days
scopes to four, with no pack edit. A pack should therefore carry **no fixed window length**: pinned
at 0 it writes off the later days of a genuinely multi-day case as ordinary business, and widened
to cover them it drags every neighbouring day of normal work into every single-day case.
`scope_status` states the derived boundary and its provenance so the operator can see which it was.

A last counting subtlety: a subject is out-of-window only when **none** of its assets is in scope,
and the unalerted subjects are split into *holding an in-window document* vs *record-only* — a
record the actor created but never acted on is worth reviewing (the OUTER explode keeps it) but has
nothing to reverse, and folding the two into one count inflates the impacted-subject total.

### `subject_discovery:` — WHO is adjudicated, and WHAT ELSE was adjudicated with them

One declaration answering two questions, read on two different occasions, because both are the same
walk over the same rows:

```yaml
subject_discovery:
  source: <logical source name from the ruleset's `sources:` map>
  path: <dotted path to the repeated node>    # optional; omitted = the row itself
  subject: [<path relative to `path`>, ...]   # candidate list, first that resolves
  entities: {<entity type>: [<relative path>, ...]}   # optional, co-located
  max_subjects: <int>                         # optional bound
```

Every value in it is the pack's own vocabulary — a path, a source name, entity-type names. Nothing
in `src/` knows any of them.

**Half one, `subject:` — consulted only when extraction produced NO subject.** The subject values a
verdict adjudicates come from the understanding stage, which runs *before* retrieval, so an alert
naming only the records it concerns ("2 affected records, ids X and Y") yields no subject at all and
the engine falls back to one degenerate subject over the whole retrieved set. Every condition that
must know *who* acted then has nothing to resolve. The acting identities were in the rows the whole
time.

**Half two, `entities:` — read either way, and it is a GAP-FILL.** The subject being named settles
who is adjudicated; it says nothing about the other values a condition needs. A decisive exclusion
asking *what kind of identity acted* has its `row_match` clauses filled `from_entity`, and an alert
routinely names the record and the account without naming the login that touched them. Types the
incident ITSELF named are skipped: an extracted value is the alert's own statement about what
happened, and reading over it with a column would let retrieved rows correct the referral.

**UNANIMITY IS THE LICENCE ON HALF TWO, and it is the only thing separating the two halves.** On the
discovery path the subject is read *out of* the element, so the element's companions are that
identity's own record. For a NAMED subject the rows are everything it touched over the source's whole
retention — many actors, one subject — and `row_match` requires every clause on the SAME reference
row. So six signs unioned with five units ask "is any of the thirty pairs registered", where one
listed member clears the other twenty-nine through the strongest exit this engine has (measured at
**31.7% of 600 alerts** on the sibling failure). A type therefore fills only where the subject's rows
carry exactly ONE distinct value; a type they disagree on is left for the extracted entities and the
condition reads `unresolved` → `unknown`, **with the disagreement stated** on the subject
(`subject_companions=`), because "the rows name four different actors" is a finding and an empty
value set is silence. Measured on one estate: 44.0% of multi-row subjects are unanimous on the
column that names the acting identity (1,219,306 of 2,773,235), and the two live referrals that
motivated the fill are both in the disagreeing majority — so the visible improvement there is an
abstention that explains itself, not a resolved check.

**Both halves group PER ELEMENT, never as two flat lists.** Handed the union, every subject would be
matched against every sibling's unit and a reference row belonging to one identity would answer
another's question — the same 31.7% failure by a different route. A path resolving to nothing is
absent from the map, never an empty-string value, so a condition sees "no value for this type"
rather than a value that matches nothing. Aliases ride along, because a co-identified subject's rows
may carry only the folded-away form.

**`max_subjects` has no engine default — absent is unbounded — and the truncation is never silent:**
`subject_cap=<n> of <N>` is stamped on every subject, since a bounded subject list adjudicates part
of an incident and must say so or the report's own count reads as the incident's size. Note which
limit bites first: the **row cap** truncates the sample the subjects are read *from*, so an uncapped
discovery over a capped read enumerates whatever fitted (measured: 256 distinct subjects inside a
500-row cap) and reports it as the population.

### Containment is gated when the finding is indicator-driven

**`brief.containment_gated`** is set whenever the positive verdict is indicator-driven. The report
merges anomaly `recommended_actions` verbatim and the anomaly clamp only fires on the negative
label, so an LLM-authored "immediately reverse the assets, suspend the actor" was landing directly
beneath the backbone's own "an EXPERT MUST CONFIRM — do NOT act automatically".
`_fallback_recommendations` re-labels those as `PROPOSED … REQUIRES EXPERT CONFIRMATION`, and
`_render_brief_for_prompt` emits a `CONTAINMENT IS GATED` line — a hard guard, not a prompt
request.

### A condition's `label` is its REQUIREMENT, not its finding

`label` is the pack's statement of what the check *demands* — right for the condition table, where
the row's own PASS/FAIL mark supplies the polarity, and exactly wrong for the sentence listing the
conditions a verdict rests on, because those are the ones that FAILED. A live report printed
`DECISIVE CONDITION(S): <actor> is not a known automated account (AUTOMATED); <actor> is not
automated (<flag path>=True)` — prose stating the actor was human under a verdict resting on it being
an automation, with the evidence in the parentheses stating the truth. The reader cannot tell which
half to trust.

No LLM is involved. `_finding` (`src/correlation.py`) assembles the sentence from the condition's
`fail_detail`, falling back to the evaluator's own finding-phrased note and only then to the label
prefixed `FAILED:` — a marked negation is readable, an unmarked one is a false statement. A fraud
*indicator*'s label is expected to state its finding, so a FAIL affirms it and it prints unchanged:
the asymmetry is the condition's polarity, not an inconsistency.

**That expectation is an OBLIGATION ON THE PACK, and nothing enforces it.** The engine cannot
rephrase a label — it has no way to negate prose — so a requirement-phrased label imported under
`polarity: fraud_indicator` prints as the finding and states its opposite. Live on IR 37264678: a
shared mechanic declared for exclusion use as `Payment card was not changed on this record` was
imported as an indicator, and the report read `DECISIVE CONDITION(S): positive fraud indicator(s) —
… Payment card was NOT changed on this record (2 distinct: [<two element ids>])` — the heading
announcing an indicator, the label denying what was found, only the parenthesised evidence true.
The same defect this section exists to document, arriving from the other direction: for an
exclusion the engine can route around the label, for an indicator it cannot.

So when a shared check is imported under a polarity its library did not assume, the importing
ruleset must override `label` so that a FAIL affirms it. Two things to know when doing so: the
ruleset's `label` wins over the library's (verified through `load_knowledge_pack` →
`ruleset_spec`), and `expected_label` is **not** a general escape hatch — only three evaluators read
it, and `distinct_count` is not one of them (it renders `expected` as the bound itself, `<= 1`), so
declaring it there is a silent no-op and the PASS side must be carried by `pass_detail`. No test
catches any of this: 1083 passed with the contradictory label, because nothing asserts that a label
agrees with its polarity.

### The containment block's STRUCTURE is the engine's, its WORDS are the pack's

`_containment_block` composes the paragraph naming a real account for action. Its labels were once
engine literals — a clause citation, a scope noun, an identity noun, "the CREATOR, not the acting
agent" — putting one procedure's clause number and one domain's identity model into every other
domain's report, in the single most consequential paragraph of it. **A clause citation is the worst
kind of leak: it reads as authoritative and cannot be checked from the engine.**

So `lock_target` declares `heading` / `scope_label` / `identity_label` / `provenance_label` /
`class_label` / `none_nominated`, and the engine's fallbacks name only the two generic roles it
resolved ("Scope" / "Identity"). The block is *composed* rather than laid out in the notification
template because a verdict whose containment the gate withheld has NO target, and a template
spelling out the field labels then renders a stack of empty ones — which reads as missing data on a
case that deliberately declined to name anybody. One field, present or absent. The labels are also
stamped onto the target itself, so the actions section and the brief's action backbone render the
same words without a private copy.

## Two classes of exclusion, and why the distinction was needed

The `categorical` / `heuristic` split above was not designed up front; it was forced, in two steps,
by live incidents where the engine had the exculpatory evidence in hand and buried it. Both steps
are worth reading before adding a condition, because the reasoning is what a future author has to
apply to a condition this codebase has never seen.

**Step one — an identity fact must be able to decide.** An automated account acting is an
*exclusion*: automation, not a human fraudster. One incident printed `[FAIL] acting identity is on
the automation list` and then could not act on it, returning `INSUFFICIENT DATA`, while the human
expert closed the case on exactly that evidence. Three independent defects, each fixed generically:

1. **`decisive_on`** (generic) — plain `decisive: true` is *symmetric*: a FAIL forces the verdict
   **and** an `unknown` forces `INSUFFICIENT DATA`. Wrong for a check whose FAIL is conclusive but
   whose absence of evidence is the ordinary case. `decisive_on: [fail]` makes decisiveness
   asymmetric — `ConditionCheck.decisive` is computed as `decisive and result in decisive_on`. The
   default (`[fail, unknown]`) preserves every existing condition's behaviour.
2. **`row_match`** (generic) — `subject_scope: false` let a condition read **all** rows its source
   returned, but the retriever OR-s the incident's entities, so those rows cover *other*
   identities. `row_match` re-scopes them to the acting identity as an **AND of per-field value
   sets** (values pulled from extracted entities via `from_entity`, so no literals enter the
   ruleset). Matching the whole composite key is the only sound reading of a list keyed by it: the
   old check hit on one component OR the other and so "found" the alerted identity in a list it was
   absent from. A `row_match` that runs over real rows and finds nothing is a genuine **pass**
   (`_scope_resolved_empty`) — distinct from "the source returned nothing" (`unknown`); an identity
   the incident never supplied is `unknown` with a note, never a silent fallback to the unscoped
   rows.
3. **Truncation guard** — `evaluate_verdict(..., row_caps=)` (from `LogRetrievalEngine.row_caps()`,
   the same source as the evidence pack's `[TRUNCATED …]` marker). A composite match finding
   nothing over a source cut off at its row cap is **no evidence**, so it reads `unknown`, not
   `pass`.

**Why one component of a composite key proves nothing** (measured): one actor identifier appeared
on **89,937 rows across 89,937 distinct org units** — exactly one per unit, because it is the
standard automated-integration identifier nearly everywhere. The alerted *pair* is on the list (1
row); a known-human control returns 0. So neither column is selective on its own: the OR-ed query
asked for ~90k rows, got the 500-row cap, and returned 500 *other* units' automated accounts —
which the report then narrated as a compromised-automation theory. Hence
`SourceDef.require_all_entities`, injected into query generation *and* enforced deterministically
after it, because the generation prompt's own emphatic "OR the weak entities" rule otherwise wins.

**…and it must have data to decide with.** Those three fixes removed the *false* conviction but
left both identity conditions `unknown`, so the verdict came out positive — still wrong. Two
further defects, upstream of the verdict engine, were starving the checks:

4. **A ruleset's `sources:` map is a data dependency, not a suggestion.** Source selection is an
   LLM call (`ApiCallGenerator`), and on a re-run it simply omitted the automation-list source — so
   the check read `no rows` → `unknown`. Every condition reads one of the ruleset's declared
   sources, so an unretrieved one silently mutes its checks. The fix at the time appended a query
   for each declared source the planner skipped; that force-add has since been **removed** — a
   source that should have been chosen and was not is a defect in the catalog's own guidance, so
   the unmet dependency is now reported three ways, scored by the health gate, and addable by an
   operator from the run controls (`retrieval.md`, "An unmet dependency is REPORTED, never
   injected"). What the fix established stands: a `sources:` map is a data dependency, and a run
   that cannot meet it must say so rather than mute the check.
5. **A field can be evidence and look like noise.** The other identity check read `flag absent`
   because the generated SQL ended `AND <flag path> = false` — a predicate that deletes exactly the
   rows answering "was the actor an automation?" — and aliased the leaf to a name `flag_fields` did
   not list. Two fixes: `SourceDef.never_filter` declares fields the query may **RETURN but never
   FILTER on**, enforced for **every** backend by `src/retrievers/query_guards.py` (see
   `retrieval.md`) by removing the conjoined predicate post-generation (a shape too complex to
   rewrite safely — inside an OR-group — is left alone and warned about, since corrupting a boolean
   tree is worse than an over-filtered read); and the source's `projection` pins the flag plus the
   identity leaves `row_match` needs, so the alias is predictable. Same lesson as
   `require_all_entities`: **a prompt instruction cannot be relied on to beat another prompt
   instruction** — the same prompt pushes hard toward filtering noise out, and it wins.

**…and the rollup must rank identity evidence above behaviour.** With 4 and 5 fixed, the identity
FAIL fired on real data — and the verdict was **still** positive, because three non-decisive
indicators reached `indicator_threshold` and step 1 overrides the exclusions.

6. **Exclusions are of two kinds, and the rollup must rank them differently.** Step 1 exists for a
   reason (a real case that is not textbook is catchable only on positive indicators), so the fix
   is **not** to reorder exclusions against indicators generally. Instead `exclusion_kind:
   categorical` marks a condition that settles **who acted**, and only those outrank the
   indicators. **Identity evidence is prior to behavioural evidence.** The default stays
   `heuristic`, so every other condition is untouched — worth a dedicated test
   (`test_heuristic_exclusion_still_loses_to_indicators`).

   Transparency matters as much as the ranking: a categorical negative verdict must not read as
   though the engine ignored evidence it actually weighed. The engine emits a
   `categorical_exclusion=` note **naming the overridden indicators** ("which did fail … but
   describe the normal traffic shape of the identified automation"), and three consumers key off
   `ConditionCheck.exclusion_kind` — the same field the rollup ranks on — to stay consistent:
   `_verdict_section` suppresses its "this verdict rests on positive evidence" callout and renders
   the categorical marker instead; the use-case analyzer leaves `containment_gated` False and treats
   the verdict as terminal for its projection guard. That the signal is a *typed field* rather than
   the note's wording is deliberate — three files agreeing on a string prefix is not a contract.

**Step two — `categorical` is about attributed FACTS, not only identity.** Scoping `categorical` to
"who acted" turned out too narrow. A later incident was closed by the expert on two facts the engine
already had: a dated, written access grant from the owning org unit to the acting one, and the
record having been created by one identity and acted on by another — which the procedure itself
lists as an exclusion, the normal shape of one unit having another act for it. Neither is an
identity check, so no amount of identity-scoped classification reaches them. But both are
*attributed facts* in exactly the sense that matters — an operator wrote them, they name an actor
and a date, and a reviewer opening the record sees the same thing. So the definition broadened to
**ATTRIBUTED FACT vs INFERENCE** (the table above) and the affected conditions were reclassified
**in the pack**. No engine tier was added: the rollup code is byte-for-byte unchanged.

**This is also the answer to "why does each fix make the next incident worse?"** The measurement in
step 1 of the rollup is the diagnosis: the indicator-override tier had never once decided a tie, so
every incident-driven adjustment was really re-tuning *which exclusion gets buried*, one incident at
a time. A reclassification generalises where a threshold nudge cannot — it states a rule a future
condition author can apply to a condition nobody has written yet, which is why a pack's `rules.yaml`
should carry that rule as a comment block above `conditions:`, where they will be reading. A sweep
over all stored incidents confirmed it discriminates rather than capitulates: `changed: 2 |
unchanged: 19`, and the non-textbook case stayed positive because the only exclusion it fails is an
inference.

Two adjacent defects surfaced by the same incident, both fixed **in the pack**, not the engine:

- **The LLM invented an expansion for an acronym** in the summary that leads the report. A
  plausible wrong expansion is indistinguishable from a right one in prose, so detection after the
  fact is hopeless: `entity_glossary.yaml` carries an `abbreviations:` map, injected by
  `glossary_prompt()` as a **closed list** that both supplies the expansion and forbids guessing an
  unlisted one.
- **Two surface forms of one identity were conflated.** An alert naming a person and an identifier
  is one actor in two forms that are *not* interchangeable when queried (authentication sources key
  on one, business sources on the other, sometimes stored with a duty-coded suffix). Substituting
  one for the other returns zero rows silently. The glossary teaches both forms, the regex that
  recognises each (a value matching it is that form *whatever the label says*), and which source
  type keys on which — see `value_forms` in `knowledge.md`.

Also from that incident: the LLM narration called the differing identities "a decisive fraud
indicator … hallmark of a split-actor fraud pattern" — the *same observation* the expert cited as
the reason it was not fraud. The deterministic verdict outranks the narration by design; this is the
measurement showing why that ordering is not a formality.

## `as_of:` — the response is not the conduct

A source that appends a row per change **keeps accruing rows after the alert**, and the rows the
responder writes are among them: on `record_lake_4`, voiding a ticket writes an envelope like any
other change, so the containment lands in the same column as the conduct. Adjudicating the chain
whole therefore weighs somebody else's remediation as the subject's behaviour — confidently, and
invisibly, because every field is populated and every value real.

Measured on IR10000001 (alert 2026-07-27T16:45Z, 53 envelopes over two records), reading all 53
changed three conditions on **both** subjects: `single_actor` FAILED on two signs, the second
being the responder; `no_void_refund` FAILED on the status that responder set at 18:15:51;
`sparse` counted the remark the responder wrote naming the incident. All three are exclusions,
so they argue NOT-fraud and the indicators still carried the verdict — which is exactly why it
survived three live runs. One indicator fewer and the containment performed on a case clears it.

**And the obvious fix is its own defect.** On `job_83e94dd6_yob4or` the whole NOT-A-FRAUD verdict
rests on a split element written at envelope 10, 16:48:03 — six minutes after the 16:42 alert, by
the **alerted sign itself**, which had gone on working the record. A cut on time deletes signed,
dated, exculpatory evidence and turns a correct dismissal into a fraud finding. So the test is the
**intersection**: later than the incident **AND** written by an identity the incident did not name.
The named actor's own later versions are that actor's conduct, continuing, and are adjudicated
whichever way they cut; a third party's are somebody else's, in practice the response.

```yaml
as_of:
  <logical source>:
    timestamp_fields: [modification_date_time, creation_date_time]  # sub-day FIRST
    actor_fields: [last_updator.sign.red]     # who wrote THIS version
    actor_entities: [user]                    # which entity types are acting identities
```

Rules, all in `_as_of_rows` (`src/correlation.py`):

- **All three keys or nothing.** Either half alone is a known-wrong rule, so a partial declaration
  makes the filter abstain — safe, and therefore invisible. `pack_validate` is where it becomes
  loud (`as-of-no-timestamp`, `as-of-no-actor-field`, `as-of-no-actor-entity`, all errors).
- **Narrow only on positive evidence.** An unparseable write time is not evidence of lateness; a
  row with no writer is not evidence of a third party. Both unknowns KEEP.
- **Sub-day column first.** The boundary is only as precise as the column: a day-granular write
  time keeps every same-day version, including the ones written 90 minutes later.
- **The last updator, not the creator** — every envelope of a record shares one creator, so a
  creator-keyed rule keeps everything. List only **projected** leaves; an unprojected path reads
  as breadth and behaves as a no-op.
- **Padded identifiers match for free** via `_identifiers_match`, so an extracted `0303CD` matches
  a stored `0303CDSU`.
- **Never empty a source.** Zero rows → every condition `unknown` → INSUFFICIENT DATA, the same
  invisible failure inverted. If the boundary excludes everything the boundary is wrong about that
  source, so the unfiltered rows stand and an `as_of_fallback=` note says so.
- **Scoped to the CONDITIONS.** `build_asset_timeline`, `build_scope_discovery`, alert
  reconciliation and the containment assessment read `logs` untouched — a reviewer needs to see
  the response. The narrowing is stated on **every** subject (`as_of=` note, rendered as
  `Evidence as of the incident:`), naming who wrote what was excluded, because a condition that
  passed on 34 of 53 versions is a different finding from one that passed on the whole record.
- **A snapshot source under-excludes, and that is the right direction.** An envelope is a full
  snapshot, not a delta, so an envelope the named sign wrote after the alert is kept (correctly)
  and carries forward a third party's earlier edit. Row-level exclusion can never over-exclude;
  per-field attribution would need a per-element writer the schema does not carry.

Tested in `test_correlation.py` (one test per guard, each verified to die when its guard is
removed), `test_pack_validate.py` (one per diagnostic), and the pack's own
`test_<domain>_pack_integrity.py` (the declaration exists, is projected, and the sweep/alert/
reference sources are deliberately NOT declared).

## Wiring

Wired into `CorrelationModule.analyze` **best-effort** (try/except, never fails the stage) → an
optional trailing `CorrelationResult.verdict` (a `ValidationVerdict`;
`ConditionCheck`/`SubjectVerdict`/`ValidationVerdict` are generic and carry no procedure-named
fields, so existing `CorrelationResult(...)` construction stays valid).
`ReportGenerationModule._verdict_section(correlation)` renders it (condition table + containment
callout + platform mode + notification draft) appended before Evidence Artifacts, with no signature
change; `pipeline_runner._summ_correlation` emits a compact `verdict` block for the UI and
`?verbose=1`.

**The retrieval fix that makes this possible:** `SourceDef.projection` (pack) is threaded via
`_merge_endpoint` into `DatabricksRetriever._generate_sql` as a **REQUIRED PROJECTION** block that
OVERRIDES the retriever's default scalar-only bias — the root cause of the rich nested fields a
ruleset depends on being dropped. A source carrying such conditions needs a `projection` plus
array-explode `query_hints`.

Guard both `_verdict_section` and `_summ_correlation` against a MagicMock correlation (`subjects`
must be a real `list`). Verdict paths are tested in `test_correlation.py`; `ruleset_spec` and
projection in `test_knowledge_pack.py`; the projection prompt in `test_retrievers.py`; the section
in `test_report_generation.py`. The end-to-end proof that all of it is generic is
`tests/test_mock_domain_pack.py`, which runs the real `evaluate_verdict` against a pack in an
invented domain and asserts no other domain's vocabulary reaches the output.
