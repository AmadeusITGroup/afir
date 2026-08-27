# Human in the loop

Five places a person touches a run, in increasing order of how much they change it:

| Surface | When | Effect |
|---|---|---|
| **Run control** (`docs/architecture/run-modes.md`) | during a run | pause / step / cancel / retry — changes *whether* a stage runs |
| **Approval gates** (§3) | during a run | the run stops *by itself* and waits for approve / reject-with-guidance / override |
| **Retrieval-plan editor** (§2b) | during a run | adds / removes whole queries by naming a source; the server builds each one |
| **Stage-output override** (§2) | during a run | replaces *what* a stage produced; downstream runs on the human's data |
| **Feedback** (§1) | after a run | changes *how* future runs reason |

Run control is documented with the job machinery in `run-modes.md`. This doc covers
the other four, plus the durability (§4) that lets a gate outlive the process. §3 also
covers what a gate does when nobody answers, how an external system is *told* one
opened (webhooks), and the UI that makes both answerable without curl.

Two properties hold across all five, and are the reason the code looks the way it does:
**a human decision is deterministic** — never an LLM judging its own work — and **a
human contribution is subordinated in the prompt**, because an analyst's note is
evidence about what to look at, not a finding.

---

## 1. Feedback — a review that changes the next run

`src/feedback_loop.py`. Three responsibilities: **collect durably → distil →
apply**. The third one is what used to be missing: insights were written to a file
nothing ever read, so the "feedback loop" had no return path.

### Collect (durably)

`POST /api/v1/feedback` → `FeedbackLoop.collect_feedback(incident_id,
investigation_result, human_feedback, **structured)` builds a `FeedbackEntry`
(`src/models/pydantic_models.py`) and persists it **on arrival**, before anything
else can fail:

- `feedback_log.jsonl` — the permanent review history. Never truncated.
- `feedback_pending.jsonl` — the un-distilled batch. Truncated after a
  distillation, and **reloaded by `__init__`** so a restart mid-batch resumes
  instead of losing the reviews.

Two more files live alongside them: `feedback_insights.json` (every distillation)
and `feedback_thresholds.json` (every numeric adjustment). All four live under
`data_dir()` (repo root, or `$AFIR_DATA_DIR`) and are gitignored.
Reviews are read back through `history()`, `stats()`, and the pending list.

The typed contract (`FeedbackEntry`) carries `agrees_with_verdict`,
`analyst_verdict`, `missed_anomalies`, `false_positives`, `notes`, `analyst`,
`job_id` — while free-form `human_feedback` still works, so the original one-field
POST is unchanged. The endpoint requires `incident_id` plus *either* free text or
at least one structured field; a body that says nothing about the investigation is
a 400.

### Distil

At `feedback.batch_size` reviews (default 10) — or on demand via
`POST /api/v1/feedback/process`, which exists because a batch that never fills is
never distilled and so never reaches a prompt, and answers 409 when nothing is
pending — `process_feedback()` makes **one**
`structured_output` call to `FeedbackInsights`. The system prompt tells the model
its output is injected verbatim into future investigation prompts, so each item
must be a short general instruction rather than a restatement of one incident, and
must describe *what to look for*, never *what to conclude*. Only after
`apply_insights` succeeds is the in-memory batch cleared and the pending file
truncated.

`apply_insights` appends a timestamped record to `feedback_insights.json`.
`merged_insights()` folds every record newest-first, de-duped per category, and is
cached (`_insights_cache`, invalidated on write).

### Apply — `guidance_prompt(stage)`

This is the return path. `IncidentUnderstandingModule` and
`AnomalyDetectionModule` each call it and, if non-empty, inject the text as a
system message.

It is deliberately **weak**, because feedback is the least trustworthy input in the
system — it is one analyst's recollection, distilled by an LLM, applied to a
different incident:

- **Per-stage category allowlist** (`GUIDANCE_CATEGORIES`). `understanding` sees
  `new_fraud_patterns` / `common_success_patterns` / `process_improvements`;
  `anomaly_detection` sees `frequently_missed_anomalies` / `accuracy_improvements`
  / `confidence_threshold_recommendations`. Any other stage gets `""` — nothing
  leaks by default.
- **Capped**: 5 items per category (`_MAX_ITEMS_PER_CATEGORY`), ~1800 chars
  (`_MAX_GUIDANCE_CHARS`). A long feedback history cannot crowd the incident out of
  its own prompt.
- **Subordinated by its preamble**: the guidance is declared ADVISORY prior
  experience, "NOT evidence about the incident in front of you", ranking below the
  knowledge pack, the official procedure and any deterministic verdict — and
  explicitly must not be used to assert a finding the retrieved data doesn't
  support.
- **Ordered to lose ties.** In `anomaly_detection` the guidance is appended
  *before* the verdict framing, so the authoritative verdict is the last thing the
  model reads. The deterministic `_clamp_by_verdict` guard is untouched.
- **Never fatal.** Both call sites wrap the render in try/except; a broken insights
  file degrades the stage to no-guidance, it does not fail the run.

`feedback.apply_to_prompts: false` collects reviews without letting them steer
anything. `GET /api/v1/feedback/guidance` returns the exact per-stage text that
will be injected — the honest answer to "what has this system learned".

### Apply, part 2 — the numeric channel (`tune_threshold`)

Prose guidance can only widen what the model *looks at*. The one numeric knob
feedback can legitimately turn is `anomaly_detection.threshold`, the confidence
cut-off in `filter_anomalies` — and it is turned deterministically, not by an LLM.

**Why not use `confidence_threshold_recommendations`.** That field is free text.
Turning "consider raising confidence for disposable emails" into a float means an
LLM guessing a number that silently changes what the entire system reports, with no
way to audit why. It is the same failure shape as trusting a prompt to enforce a
pack guarantee: *a prompt cannot be relied on to beat another prompt*. So the prose
still reaches the anomaly-detection prompt exactly as before — it is simply not the
source of the number.

**The signal is counted, and directional.** Only structured review fields vote:

| review contains | means | direction |
|---|---|---|
| `false_positives` only | noise got through the cut-off | **raise** |
| `missed_anomalies` only | real findings were filtered out | **lower** |
| both | the *ranking* is wrong, not the cut-off | abstains |
| free text only | no auditable direction | abstains |

`notes` / `human_feedback` are deliberately **not** keyword-scanned.

**Five guards**, because this changes what the system reports:

1. **Opt-in.** `feedback.auto_tune_threshold` defaults to **false**. When off the
   recommendation is still computed and exposed at
   `GET /api/v1/feedback/threshold` with `applied: false` — collect and advise,
   don't act. This is the same posture as `apply_to_prompts`.
2. **Minimum evidence.** `min_reviews_for_tuning` (default 5) reviews must have
   arrived *since the last adjustment*. One analyst's single review cannot retune
   detection sensitivity.
3. **A clear direction.** The net imbalance must reach `_MIN_NET_SIGNAL` (2).
   Balanced complaints mean the threshold is not the problem.
4. **One small step, bounded.** At most `threshold_step` (0.05) per adjustment,
   never outside `[threshold_min, threshold_max]`, and never further than
   `max_threshold_drift` (0.15) from the operator's configured baseline. Feedback
   nudges the declared value; it cannot wander away from it.
5. **A watermark.** Each adjustment records `reviews_at_adjustment`, and the next
   window starts there. Without it the same complaints would be re-counted every
   window and walk the threshold to a bound — classic integral windup. This is the
   guard that is easiest to omit and hardest to notice missing.

**Config stays the source of truth.** The tuned value is appended to
`feedback_thresholds.json` beside the evidence that justified it, and is *never*
written back into the operator's YAML. Three consequences that matter:

- `anomaly_detection.threshold` remains the baseline every bound is computed
  against, so drift cannot compound (if the tuned value became the new baseline,
  `max_threshold_drift` would mean nothing).
- `effective_threshold()` **re-clamps on every read**, so tightening the bounds or
  moving the baseline in config immediately reins in a value tuned under the old
  settings.
- Flipping `auto_tune_threshold` back to false reverts instantly *without* deleting
  history; `POST /api/v1/feedback/threshold/reset` discards it for good.

**The consumer never depends on any of this working.**
`AnomalyDetectionModule.effective_threshold()` falls back to the configured value
when there is no feedback loop, when the store is unreadable, and on any exception.
Detection sensitivity must not depend on a JSON file being parseable.

`tune_threshold()` runs at the same batch boundary as the prose distillation (at the
end of `process_feedback`), so both halves of "apply" advance together.

---

## 2. Stage-output override — continuing on the human's data

`POST /api/v1/jobs/{job_id}/outputs/{stage}` →
`JobManager.set_stage_output(job_id, stage_name, value, actor)`.

The problem it solves: before this, a stage that produced *wrong* (not failed)
output left only two options — accept the whole downstream conclusion, or
`retry_all` and hope the LLM does better. Neither lets an analyst who already knows
the right answer supply it.

**Typing is the crux.** Downstream stages consume Pydantic objects, not dicts, so
the analyst's JSON is decoded through `decode_output_value` — the same
`_OUTPUT_CODECS` table the job export/import path uses. One codec table, so a shape
the exporter can write is a shape the override can read. A mismatch raises
`ValueError` → the endpoint returns 400. The job is never poisoned with a
half-typed value.

**It refuses while the stage is RUNNING**, because the stage would overwrite the
override the moment it returned.

On success the stage is marked `COMPLETED`, its `summary` is recomputed via
`summarize_stage`, and if it was the *failed* stage, `_failed_index` and `error`
are cleared — the job is no longer in a failed state.

### Continuing: `skip_stage`, not `retry_stage`

`retry_stage` re-runs the stage, which would discard the override. So the
continuation path is a new control action:

`skip_stage` (optionally `{"stage": "<name>"}`, defaulting to the failed stage)
resumes execution from `index + 1`. It marks the target `SKIPPED` **only if it is
not already COMPLETED** — so an overridden stage keeps its honest `completed`
status, while a genuinely skipped one reads as skipped. Skipping the last stage
completes the job.

### Audit trail

`Job.record_intervention(action, stage, detail, actor)` appends to
`Job.interventions`, which rides:

- `snapshot()` (so the UI and every API read see it),
- the SSE stream — a new `intervention` event, plus the `stage_output` event
  carrying `"overridden": true`,
- `export_job` / `import_job`.

A report built on hand-edited data must be traceable as such. That is the whole
point of the list.

### Overridable stages

`understanding`, `query_generation`, `log_retrieval`, `correlation`,
`anomaly_detection`, `report_generation`. Excluded: `plugins`, `export`, `output` —
their outputs are side effects, not data the pipeline reads.

---

## 2b. The retrieval-plan editor — naming a source, not writing a query

`GET` / `POST /api/v1/jobs/{job_id}/queries` (`src/incident_input.py`), and the
`#qpPanel` block inside the run-controls popover (`src/ui/script_core.py`). §2 can
already replace the whole `query_generation` output, so this exists for a reason:

**Why it is not just §2.** The edit an analyst actually makes is *this source should
have been queried and wasn't*. Through §2 that means hand-writing a `RetrievalQuery`
list — and a hand-written query is either unscoped (a bare date-window scan over a
partitioned table) or scoped by columns the source does not have, because **which
entities a source can bind is pack knowledge**: `_entities_for` filters the incident's
entities to the types this source declares it can filter on, and `render_filters`
routes each value to its own form's binding. So the client sends `source` +
an optional `question`, and the server builds the query through
`build_manual_query` → the same `_enrich_queries` a planned query goes through. A
second copy of that logic in JS would be a second answer to it.

**Why it exists at all.** Nothing is force-added any more (see
`docs/architecture/retrieval.md` — a deterministic injection was removed because
source selection belongs to the incident and the understanding). The planning stage
reports its unmet dependencies as a three-way finding instead, and the health scorer
gates on one of the three. This endpoint is the seam through which a human *acts* on
that finding, in one click, on the run in flight — while the catalog fix that stops it
recurring is made separately.

**What the GET answers**, and none of it is derivable client-side:

| Field | Why the server must say it |
|---|---|
| `queries[].scoped_by` | the entity types the query really narrows on (`time_window` excluded) — the entities were filtered to what the source binds |
| `unselected[]` | the addable menu: `declared` (a hard dependency of the adjudicating procedure), `deferred` (a follow-up pass will fetch it later), `scopable` (some incident entity is a type it can filter on), `purpose` (first line of the catalog description) |
| `dependencies` | `undeliverable` / `not_queried` / `unscopable` — the same object the health scorer reads |
| `row_counts` | what each source has returned *so far on this run*: a source may already have answered, and the query about to be removed may be the one holding the rows |

The menu is ordered with the procedure's own unmet dependencies first — that is what
the removed mechanism used to act on, and what the operator is most likely looking
for. A `deferred` target is **offered but flagged**: a pass-1 query against it merges
its rows into the pass-2 result under one source name, so a condition counting
distinct rows would read both scopes at once.

**Three rules the POST holds:**

- **All-or-nothing.** A source with no retriever, a non-integer index, an addition
  that is not an object — each refuses the whole request, so the plan is never left
  in a state neither the operator nor the run intended.
- **Indices resolve before anything is appended.** Removals are taken against the
  list the client read; otherwise a removal could target a row the same request just
  added, and the handles the client was given would not mean what they meant.
- **It is an intervention like any other.** Applied through `set_stage_output(...,
  detail="retrieval plan edited by analyst: added …; removed …")`, then
  `flush_persistence`. Continue with `skip_stage`, not `retry_stage` (§2) — a retry
  re-plans and discards the edit.

Refusals name the reason that applies, because "no understanding yet", "no such job"
and "jobs/generator not wired" are three different things to do next (409 / 404 /
503), and a `query_generation` stage currently `running` is a 409 too — it would
overwrite the edit the moment planning returned, which reads as the edit having
vanished.

**The panel collapses with the job.** `qpPlan` / `qpDrop` / `qpAdds` are indices into
ONE job's query list, so `ctlEnabled(false)` clears them: carrying staged edits across
an attach would apply them to another run.

---

## 3. Approval gates — the three run modes

`src/pipeline_runner.py` + `src/stage_health.py`. Run control (pause/step/cancel) lets
a human interrupt a run; a **gate** makes the run stop and wait *on its own*, at a
point where a decision is actually due.

`JobRunMode` — set per job at creation, not globally:

| Mode | Behaviour |
|---|---|
| `auto` | Every stage back-to-back. No gate ever opens. The pre-existing behaviour, and still the default. |
| `semi_auto` | Gates **only** where a stage's deterministic health score falls below its threshold. High-confidence stages pass through untouched. |
| `supervised` | Gates after every gateable stage. Nothing reaches a report unreviewed. |
| `step` | A distinct, older mechanism kept as-is: pauses **before** each stage. |

**`step` and gates are not the same thing and are deliberately separate.** `step` asks
permission for a stage to *start*; a gate opens **after** `stage_output` /
`stage_completed` and asks about what the stage *produced*. One is a throttle, the
other is a review.

### The score is deterministic, never the LLM's self-assessment

`score_stage(stage, output, ctx, config)` starts at 1.0 and subtracts a weight per
fired **signal** — each signal a countable fact about the output (`no_entities`,
`empty_sources`, `source_truncated`, `unresolved_join_keys`, `verdict_degraded`,
`report_fallback_used`, …). Asking the model "how confident are you?" would gate on the
same faculty that produced the output, and a model that hallucinated an entity is not
the thing to ask whether it hallucinated an entity.

- **`FATAL` (weight 1.0)** for "produced nothing usable" — score 0.0 regardless of how
  many healthy signals also hold. No amount of partial success offsets zero rows.
- **Pro-rata** where the magnitude matters: `empty_sources` scales by the *fraction* of
  sources that came back empty, so 1-of-19 and 18-of-19 don't cost the same.
- **Split where the degradations are different in KIND rather than degree.** The evidence
  pack's six-rung budget ladder is one signal only if every rung costs the same thing, and
  it does not: rungs 1–5 *aggregate* (fewer examples, top-3 distributions, daily buckets,
  capped chronology) and each rung still renders in full, while rung 6 gives up on fitting and
  hands the text itself to the renderer to thin — per-source detail (distributions, example
  rows) and the chronology/attribution lines drop out of what the downstream LLM stages read.
  So `evidence_degraded` (0.1) and `evidence_clipped` (0.3) are separate and
  mutually exclusive codes, and the clip's detail names the remedy —
  `correlation.evidence_char_budget`. **That detail has to name the loss it causes NOW, not the
  one it used to.** It read "sources at the tail are missing", which was true of the flat
  positional clip the renderer had before `_thin_source_lines` sub-budgeted the head against
  `[SOURCES]` and kept EVERY source's count line. A reason that overstates its own consequence
  sends the operator to the wrong place: told a source is missing they go looking for a
  retrieval failure, when the run consulted every source and the narrator simply read less
  about each. The weight stays 0.3 rather than sliding to the milder reading — reaching this
  rung means the corpus survived five degradations, and the renderer's own fallback can still
  drop count lines when those alone overbudget, in which case the render says how many went. Measured over one deployment's 21 distinct incidents:
  9 never degraded, 2 stopped at an aggregation rung, 4 reached rung 5 and **6 reached the
  clip**. One flat 0.3 scored all twelve identically, so the rung with a real evidence-loss
  incident behind it (job 4da14f65: 5 of 9 sources gone from the prompt, the system of record
  among them) read exactly like dropping a handful of example rows — while a live run landed
  on *exactly* its own gate threshold because aggregation alone cost it 0.3. Aggregation
  costs detail the deterministic verdict never reads (it reads the rows, not this pack); the
  clip costs the narration's input everything below each source's count line, which is the
  same class of defect as a truncated result — "lossier" and "incomplete" are different
  findings. The note the scorer matches on is `evidence.CLIP_NOTE`, imported rather
  than re-spelled, and pinned by a test — spelled twice, a reworded note would silently
  downgrade every clipped run to the aggregation weight and nothing else would change. **Its
  own sentence had to move once for the same reason this detail did**: it claimed a hard clip
  while the renderer's per-half thinning now usually fits the text without one (measured on
  job 562a60ad: the note fired, no character was clipped, and the real loss — 27 head lines —
  was reported two lines below it), and the LLM reads it too, so it says "the render is
  trimmed to fit".
- **Three rules about the ladder itself, all of which failed silently before they were rules.**
  (1) **A rung is charged to the half it shrinks.** `render_for_prompt` sub-budgets the
  chronology/attribution head at `_HEAD_BUDGET_SHARE` and gives `[SOURCES]` the rest, so a
  ladder gating on the TOTAL is *unsatisfiable* whenever one half alone exceeds the whole
  budget — every rung then runs whether or not it touches the half that is over. `_half_lens`
  gates rungs 1/2/5 on the source half and 3/4 on the head half; the total stays the early
  exit, and per-half gating is strictly stronger than it (both halves inside their caps
  implies the total is), so termination is unchanged. (2) **The values a CONDITION was decided
  on outlive every rung.** `SourceEvidence.adjudicated_values` holds the columns a verdict
  condition or the ruleset's `alert_record` block named, with their values, and renders as a
  mandatory `    * ` line — because rung 1 empties `examples` and rung 5 empties
  `distributions` while `kept_columns` keeps naming the column, and a narrator handed a column
  name with no value against it truthfully reports the value ABSENT, contradicting the
  deterministic condition that read it (measured on the same job: a `[FAIL] passenger-type`
  line beside a narration saying those values "are not present in the aggregated evidence" and
  a recommendation to go and read them). It is a subset, not a blanket: no ruleset means no
  promise, the same source's example rows and tallies are still degradable, a column whose
  whole population is blank promises nothing, and a spilled list counts what it dropped.
  (3) **`_rendered_len` measures the UNCLIPPED pack** — gating on the clipped length makes the
  whole ladder a no-op at the only budget anyone runs.
- **Every weight and threshold is config-overridable** (`stage_gates.weights`,
  `stage_gates.threshold`, `stage_gates.stages.<name>.threshold`) because the default
  0.6 is a starting point to calibrate against observed runs, not a measured constant.
  A signal can be disabled by setting its weight to `0.0` — no code change.
- **`StageHealth.reasons` carries the *why***, aggregated one entry per code with a
  count, capped at 12. A gate that says only "0.45" is not actionable.
- **`score_stage` never raises.** A scorer bug degrades the stage to `scored=False`; it
  does not fail the investigation.

**Unscored is not healthy.** `plugins` / `export` / `output` have no signals, so they
report `score=1.0, scored=False` — explicitly *unmeasured* rather than suspiciously
perfect. `_gate_applies` requires `health.scored` in `semi_auto`: treating a
placeholder 1.0 as a measurement would wave through a stage nobody scored.

**A signal must measure the pipeline, not the model's prose.** `source_without_query`
compares `log_sources_to_review` against each query's `target_log_source` — but the
first is free text the LLM wrote ("authentication logs — login session for identifier
<value>", "<vendor> fraud-management alerts") and the second is a canonical pack id
(`auth_events`, `fraud_alerts`). Comparing them literally fired on *every* entry of
*every* real run, pinning `query_generation` at 0.00 and recommending a gate for it
unconditionally, while the suite stayed green because its fixtures use tidy ids like
`src_a`. So an entry is resolved to the id it **names** (alphanumeric-only containment,
ids of ≥4 chars, longest match wins) and judged against `log_retrieval`'s live
retriever catalog: named-and-untargeted counts, named-but-unservable doesn't (an
unprovisioned source is not this stage's failure), and naming nothing recognisable is
**silence** — an unresolvable sentence is not evidence of under-retrieval. The
regression fixture in `tests/test_stage_health.py` is the verbatim prose from two live
runs, because inventing tidier strings is exactly how this survived a 740-test suite.

**An empty source is not always a defect — and only the pack knows which.** `empty_sources`
counted every zero-row source equally, which on job `ae050dbf` read *"4 of 9 source(s)
returned zero rows"* — naming a session log, an admin trail, a third-party alert feed and a
reference list of automated identities — at −0.18. Exactly one of those was a real gap (a stale epoch literal,
see `docs/architecture/retrieval.md`). The other three are the scorer misreading a correct
outcome, in three distinct ways:

- **The source answers BY being empty.** The reference list is an exclusion check; no row for
  the composite identity key means the actor is **not** a registered automation — i.e. a
  HUMAN — which is the finding the fraud path needs to be true. Penalising it penalises a
  check that *succeeded*.
- **The question was conditional and didn't apply.** The third-party feed only fires on one
  form of payment; on an incident using another, its emptiness is the wrong question,
  correctly answered.
- **The source is secondary for the playbook in flight.** The admin trail is primary for two
  other playbooks in the same pack and only a conditional cross-check for this one.

So the source declares `zero_rows: {health_weight, meaning}` in the pack (`SourceDef`), and
`src/stage_health.py` keeps only the arithmetic: the pro-rata numerator becomes the *sum of
weights* over the empty sources instead of their count, `0.0` means an empty result is a
valid ANSWER and fires nothing at all, and an undeclared pack scores exactly as before.
Two properties matter beyond the number. The domain judgement stays out of `src/` — the
engine never names a source. And **the discount is visible**: the reason string appends
`— discounted as expected/valid: automated_users (no row means … a HUMAN actor); …`,
because a silently smaller penalty is one the operator cannot check, and the whole point of
a deterministic score is that it can be checked. A malformed `health_weight` is logged and
ignored (counts in full) rather than allowed to break a scorer that runs on every stage.

**And the score is not where the reader looks.** The declaration was honoured by the scorer
and by nothing else, so the same emptiness reached the report as a bare `0 record(s)` —
which is indistinguishable from a source that failed to return. On the live session-anomaly run
`automated_users` came back empty (the answer: a human acted), the verdict read it as a
PASS, and the narration then said in **three separate passages** that the check *"did not
return"* and the actor's kind was UNKNOWN — while the condition line beside it read PASS.
The report's own gaps section had told it so, listing the source under *"returned NO rows …
unevaluated, not clear"*.

`KnowledgePack.zero_row_meanings()` closes that: the `meaning` rides onto
`SourceEvidence.zero_rows_meaning` (set only when the source is **actually** empty — on a
source that returned rows it describes a result that did not happen, and rendering it there
invites the narrator to read it as the finding), `render_for_prompt` prints it as *"ZERO ROWS
IS THE ANSWER HERE, not a gap"* in the line the LLM reads, and
`_not_retrieved_section` moves those sources to their own sentence instead of the gap list —
**reported, not omitted**, because "empty and that settles it" and "never consulted" both
look like silence to a reader.

Keyed on **`health_weight == 0`, not on the presence of prose.** The two are different
claims and a pack legitimately makes both: `0.0` is the value that says empty is a valid
answer, while a *discounted* source (0.1–0.9) is empty for a reason worth recording and
still a gap. One measured `meaning` on that shape reads *"the office has no OFP attribute
rows, so its commercial type could not be established"* — precisely the `unknown` the reader
must keep. Reading every declaration as an answer would convert those into fabricated
certainty: this seam's own failure mode, inverted. Of the eleven declarations in the live
pack, two qualify.

### Which stages can gate

The six LLM stages (`GATEABLE_STAGES`): understanding, query_generation,
log_retrieval, correlation, anomaly_detection, report_generation. `plugins`, `export`
and `output` **can never gate** — their results are side effects (a file written, a
mail sent), not data a later stage reasons over, so there is nothing to approve that
approving `report_generation` didn't already cover. Any of the six can be switched off
via `stage_gates.stages.<name>.enabled: false`.

**Two of the six can run more than once.** Where a use case declares a follow-up
retrieval pass, `query_generation` and `log_retrieval` repeat (see
[run-modes.md](run-modes.md#re-entry-a-stage-list-that-is-not-a-straight-line)), so a
gate on either opens **once per pass** and the gate record carries `pass`. The config
key is per stage and not per pass — an operator who wants to review a retrieval plan
wants to review each of them — and `_gate_applies` is unchanged in logic: it still
reads the stage name, because `pass_key` returns the bare name at pass 1 and
`stage_gates.stages.log_retrieval` must go on meaning what it meant. A pass-2 gate
scores on pass 2's own output, so a clean first plan does not vouch for a second one.

### Resolving a gate

`POST /api/v1/jobs/{id}/gate` → `resolve_gate(...)`. Three actions:

| Action | Effect |
|---|---|
| `approve` | Continue at `index + 1`. |
| `reject` | **Requires `guidance`.** Appends it to `ctx.stage_guidance[target]`, resets that stage and everything after it to `PENDING`, and re-runs from there — then **re-gates**. |
| `override` | Delegates to `set_stage_output` (§2), so the value passes the same codec, audit trail and re-scoring, then continues. |

**`reject` requires a correction** because a reject with none re-runs an identical
prompt, gets an identical answer, and reads to the operator as the rejection having
been ignored. `restart_from` may name an *earlier* stage (the wrong entity was
extracted in `understanding`, discovered at `correlation`); naming a *later* one is a
400, since a stage that runs after the gated one cannot be what produced its output.

The injected correction is rendered by `src/human_guidance.py` — **one** renderer for
all six stages, so an analyst note is capped and subordinated the same way everywhere
rather than each stage inventing its own framing. Same posture as feedback guidance
(§1): it widens what the model checks; it does not assert a finding.

`resolve_gate` raises `ValueError` when no gate is open → **409**, not 200. Answering
200 would let a UI display an approval that never happened.

### What a client sees

`_gate_record` puts everything needed to decide in **one** object — stage, reason,
full `health` (score, threshold, every reason), the stage `summary`, the available
actions, `opened_at` — carried by both the `gate_opened` SSE event *and* the job
snapshot. An integrating app renders a decision screen without a second call, and a
client that missed the event still finds the gate by polling. `GET /api/v1/gates` is
the cross-job inbox. Every resolution appends to `job.gate_history` with actor,
`reason_code` and both timestamps.

`JobStatus.AWAITING_APPROVAL` is **non-terminal** and excluded from
`_TERMINAL_STATUSES`, hence from TTL pruning — a job holding an unanswered question
must not be garbage-collected out from under the person answering it. Gates hold
**indefinitely** by default, which is what makes §4 necessary.

### When nobody answers — `timeout_seconds` / `on_timeout`

`stage_gate_timeout` / `stage_gate_on_timeout` (`src/stage_health.py`), applied in
`_gate_wait_timeout` / `_on_gate_timeout`.

**The default is no timeout at all.** A gate exists because a human decision is due,
and a timeout that quietly expires is a decision made by a clock — the behaviour the
operator opted *out* of by choosing `semi_auto` or `supervised`. Consequences of that
posture, all deliberate:

- `timeout_seconds: 0` and negatives read as **"no timeout"**, not "expire
  immediately". An instantly-expiring gate is never what someone means by writing `0`,
  and reading it literally would silently un-gate a supervised run.
- A non-numeric value is logged and ignored; an unknown `on_timeout` falls back to
  `hold`. Every failure mode lands on *more* human involvement, never less.

| `on_timeout` | Effect |
|---|---|
| `hold` (default) | Emit `gate_timeout` once, keep waiting. **Nothing is decided.** |
| `proceed` | Continue as if approved — recorded as *not reviewed*. |
| `abort` | Cancel the job. |

`hold` notifies **exactly once**: `_gate_wait_timeout` returns `None` once the deadline
has fired, so the wait reverts to indefinite. A repeating alarm on a gate nobody has
answered is how a recipient learns to mute the channel that carries the alerts that
matter.

`proceed` and `abort` are decisions taken without a human, so each writes to
`interventions` **and** `gate_history` with `actor: "timeout"` — a report must never
look reviewed when the reviewing was done by a clock. `hold` writes neither, because
nothing happened. `abort` sets the same `_cancel_all_flag` / `_cancel_event` an operator
cancel sets, so everything downstream that asks *"was this cancelled?"* — including the
outer `CancelledError` handler's shutdown-vs-cancel test — gets one answer.

The gate record advertises `timeout_seconds` and `on_timeout`, so an inbox can render
"expires in 1h → then: proceed" instead of implying every gate waits forever.

### Webhooks — the push half of "a gate is open"

`WebhookDispatcher` (`src/notifications.py`), the third sink on `EventEmitter.emit`.

SSE requires a client connected *and staying* connected. A gate that holds for two days
outlives any browser tab, so an integrating app needs a push it can receive while
nothing of its own is running. Subscribable events are restricted to the six an external
system acts on — `gate_opened`, `gate_resolved`, `gate_timeout`, `job_completed`,
`job_cancelled`, `stage_failed`. The per-stage stream is not subscribable: firing on all
of it is a self-inflicted DoS on the receiver.

`webhook_event_name` maps the internal event to its webhook name rather than adding
duplicate emits at every terminal site. Completion rides `job_status` with
`status: "completed"` because that is one stream a browser reads sequentially; a
subscriber wants the opposite — a *specific* event to register for. One wire format for
SSE, stable names for integrators, and no second call site that can forget to fire.

**Never blocks and never fails a run.** Delivery is a background task with bounded
linear backoff; 4xx is not retried (the same rejected payload would be rejected again);
every error is swallowed with a log line. `notify` is called *before* the JobManager
lookup and is not gated on it — an external subscriber's interest in "a gate opened"
does not depend on this process also holding the job in memory. `close()` awaits
in-flight deliveries for at most 5s, because the alternative to a dropped notification
is a container that will not stop.

**Not a delivery guarantee, by design.** No durable outbox: a POST in flight when the
process dies is gone. Acceptable *because* the state it announces is durable — the gate
is still open, still in `GET /api/v1/gates`, still in the job store. A webhook is a
latency optimisation over polling. Anything treating it as the system of record will
drop a gate at the first restart.

A URL is a secret (a Slack webhook URL *is* the credential), so config holds `${VAR}`
and never a literal — same rule as every other credential here. An unresolved `${VAR}`
is an **ERROR** and drops that target: sending nothing silently is how a missed approval
goes unnoticed.

### The UI surfaces

`src/webui.py`. The API is the contract and the UI is one client of it, but the UI is
the standalone product, so both halves of §3 have to be reachable without curl:

- **A mode radio per `JobRunMode`**, each with the copy that names what it will do.
  `MODE_STATUS` matters more than it looks: semi-auto and supervised **stop on their
  own**, so the status line has to say so — a run that stopped for approval must never
  be mistaken for a run that hung.
- **The gate panel** — above the stage cards, warn-coloured, pulsing on open. It renders
  from the `gate_opened` payload alone (no extra fetch): the stage, why it stopped, the
  score bar *and every reason behind it*, and the three action buttons. A gate showing
  only "0.45" is what trains people to rubber-stamp.
- **`reject` is checked client-side too**, so the analyst is told a correction is
  required immediately rather than after a round trip — the server still 400s, this is
  not the enforcement.
- **A `restart_from` select limited to the gated stage and earlier**, mirroring the
  server's rule rather than offering a choice that would 400.
- **A per-card health badge on every stage**, not only the ones that gated. In
  `semi_auto` that is the difference between "this stage was fine" and "nobody measured
  it".
- **The approvals inbox** (`GET /api/v1/gates`, polled every 5s). Polled, not pushed:
  SSE is per-job, and the whole point is gates on jobs this page never launched —
  including ones restored from before a restart. Clicking a row attaches the page to that
  job, rebuilding the cards and health badges from its snapshot so the decision is made
  with the full breakdown rather than a bare Approve button.
- **An unarmed `pending_gate` renders a message, never a button** — no runner is waiting
  on it yet, so answering would resolve nothing.
- **A `hold` timeout keeps the panel up.** Only `proceed`/`abort` hide it, and both say
  *who* decided ("expired after 1h — proceed without review"). Hiding the panel on every
  timeout would make an unanswered gate look answered.

`tests/test_webui.py` is the type-checker this inline HTML/JS otherwise lacks: it
asserts every referenced DOM id exists, every called function is defined, every fetch
URL matches a registered aiohttp route, and the JS vocabularies (mode values, gate
actions, overridable stages) equal the Python constants they mirror.

---

## 4. Durability — a pending approval outlives the process

`src/job_store.py`. An approval gate holds **indefinitely** by default (a wrong
auto-decision is worse than a late one), which makes a supervised job a multi-day
object while `JobManager._jobs` is a plain dict. Without persistence a deploy or an
App restart silently discards every pending approval: the analyst's queue empties and
nothing says why.

**One JSON file per job**, under `<AFIR_DATA_DIR>/jobs/<job_id>.json`, written via
`JobManager._persist` at every point where state a human cares about changes: each
stage completion, a stage failure, **a gate opening**, a gate resolution (*before*
the runner is woken, so a crash mid-resolution cannot re-ask a question already
answered), a stage-output override, and the terminal statuses.

**It reuses `export_job` / `import_job`, not a second format.** Same reason the
override endpoint rides `_OUTPUT_CODECS`: a model change must not leave persistence
decoding into a stale shape. There is one serialization of a job in this codebase.

### What a restart is allowed to claim

`import_job` will not restore a status that implies something live:

| Persisted | Restored as | Why |
|---|---|---|
| `RUNNING` | `PAUSED` | No task exists. Auto-resuming would re-run a stage whose side effects (a query issued, a mail sent) are not known to have completed. |
| `AWAITING_APPROVAL` | `PAUSED` + `pending_gate` | No runner is waiting to read a decision yet. |
| `PENDING` | `PENDING` | A job that never started really is pending. |
| stage `RUNNING` | stage `PENDING` | Nothing recorded a result, so it is unfinished work. |

**A shutdown under a gate is not a cancel.** The outer `CancelledError` handler
distinguishes a process stopping (`AWAITING_APPROVAL` preserved) from an operator
cancel (`_cancel_all_flag` / `_cancel_stage_flag` set → genuinely `CANCELLED`).
Conflating them persisted the pending approval as a dead job.

### Re-arming: `rearm_gates()`

Called from `main()` **after** `restore()`, because it parks real waiters and needs a
running loop. For each carried-over `pending_gate` it spawns `_rearm_one`, which
re-opens the gate on the **existing** output and waits through the same
`_wait_for_gate_decision` as a first-time gate.

Deliberately *not* re-running the gated stage: the stage already produced the output
under review, so a re-run would discard it, spend the LLM call again, and — health
being recomputed — could reach a different verdict than the one being reviewed.
`_rearm_one` also lifts the import-time pause, or the approval would be followed
immediately by a silent stop at the pause barrier.

Re-armed gates carry `reopened_after_restart: true` — whoever answers should be able
to see the gate predates a restart. `pending_gate` is reported separately from
`open_gate` in the snapshot so a client never renders a decision button that would
resolve nothing, and it exports **as** `open_gate` so a second restart before
re-arming still cannot lose the decision.

### Storage failure is degradation, not failure

Every method is best-effort and returns a bool. `AFIR_DATA_DIR` normally points at a
Unity Catalog Volume; a `mkdir` succeeding does not prove writability (a Volume can be
mounted read-only), so the store write-probes and falls back to local disk, logging at
**ERROR** — local disk does not survive a container restart, so a quiet downgrade is
how a pending approval disappears without explanation. Losing durability must not lose
the investigation.

### Evidence sidecar

Retrieved rows (`logs`) go in `<job_id>.evidence.json`, rewritten only when the
evidence actually changed — most saves are status transitions. Past
`jobs.max_evidence_mb` the evidence is **dropped and the omission recorded in the doc**
(`evidence_omitted`), because a job that resumed on silently-empty logs would present
a report built on no data as though it were built on data.

Writes are atomic: temp file → **parsed back** → `os.replace`, with the previous good
version kept as `.prev` (and read as a fallback). A truncated write that happens to be
valid JSON is indistinguishable from a real job doc at load time.

Job **files** outlive the in-memory TTL (`jobs.retention_days`, default 14 vs. 1 hour):
evicting a finished job from memory must not destroy a record an auditor wants next
week. Config: `jobs.{persist,dir,max_evidence_mb,retention_days}`.

---

## Where the code is

| File | Role |
|---|---|
| `src/feedback_loop.py` | collect / distil / apply; `guidance_prompt` (prose) + `threshold_recommendation` / `tune_threshold` / `effective_threshold` (numeric) |
| `src/models/pydantic_models.py` | `FeedbackEntry`, `FeedbackInsights` |
| `src/incident_understanding.py` | `_feedback_message()` injection |
| `src/anomaly_detection.py` | `_feedback_guidance()` injection; `effective_threshold()` in `filter_anomalies` |
| `src/stage_health.py` | deterministic per-stage health scoring (`score_stage`, `GATEABLE_STAGES`); gate config accessors incl. `stage_gate_timeout` / `stage_gate_on_timeout` |
| `src/human_guidance.py` | the single renderer subordinating an analyst correction in all six stage prompts |
| `src/job_store.py` | durable job docs + evidence sidecar; atomic verified writes; fallback + retention |
| `src/pipeline_runner.py` | `set_stage_output`, `decode_output_value`, `skip_stage`, `Job.interventions`; gates (`_gate_applies`, `_gate_record`, `_await_gate`, `_wait_for_gate_decision`, `_on_gate_timeout`, `resolve_gate`); `restore` / `rearm_gates` |
| `src/notifications.py` | `WebhookDispatcher` + `webhook_event_name` (the third `emit` sink); structured `extra=` twins on the event log line |
| `src/utils/logging_setup.py` | console vs. JSON formatter selection; `extra=` promotion, frame protection, noisy-logger quieting |
| `src/api_call_generator.py` | `dependency_report` / `unselected_sources` / `build_manual_query` — the plan editor's whole vocabulary, all stateless (one generator serves every job) |
| `src/incident_input.py` | the six feedback routes + the override route + the two plan routes (`_plan_context`, `get_job_queries`, `edit_job_queries`) + the gate routes (`/jobs/{id}/gate`, `/gates`) |
| `src/webui.py` | the mode selector, gate decision panel, per-stage health badges, approvals inbox, override controls, the retrieval-plan panel, analyst-review panel |
| `tests/test_hitl.py` | feedback, override, `skip_stage`, interventions, HTTP level included |
| `tests/test_gates.py` | the three run modes, guidance reaching the prompt, re-gating, cancel during a gate |
| `tests/test_gate_timeouts.py` | timeout config resolution, all three `on_timeout` outcomes, human-beats-timeout, webhook naming + delivery |
| `tests/test_job_store.py` | restart survival, re-arming, honest restored state, evidence cap, write integrity |
| `tests/test_logging_setup.py` | one-line JSON, `extra=` promotion end-to-end, fallback on bad config |
| `tests/test_webui.py` | the missing type-checker: DOM ids, undefined functions, fetch URLs vs. routes, JS/Python vocabularies |

API reference: `docs/API.md` §5, §5b, §5c (the plan editor), §7 (gates + timeouts),
§8 (webhooks), and Feedback.
