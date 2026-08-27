# Report generation

The report is the pipeline's acceptance artifact, so `ReportGenerationModule.generate`
must always produce one. It has two paths: the LLM **narrates** the sections, or a
**deterministic builder** (`_fallback_sections`) assembles them from data already in hand
(understanding, correlation, verdict/brief, anomalies, per-source row counts).

Both paths are legitimate, but they are not equivalent, and the difference is the whole
point of the stage. The deterministic path emits accurate bullet lists. The narrated path
does the *synthesis* — a chronological reconstruction that walks the responsible actor's
activity in time order, explains why each verdict condition passed or failed in plain
language, and cites the source each step came from. A human reading a bullet dump of 873
correlated records has been handed the raw material, not an investigation.

So: **the fallback is a safety net, not an acceptable steady state.** When a run lands on
it, or when `_ensure_complete_sections` backfills most of the report, that is a defect to
diagnose rather than a graceful degradation to accept.

## The token budget is the load-bearing setting

Measured live against the Databricks-served model on incident `83e94dd6` — a real
verdicted FALSE POSITIVE, **0 anomalies**, 22k-char prompt (18.6k of it evidence):

| `max_tokens` | `finish_reason` | completion tokens | content | outcome |
|---|---|---|---|---|
| 4096 | **`length`** | 4096 | `{}` | 6 of 7 sections backfilled |
| 8000 | `stop` | 6,239–6,740 | ~19–21k chars | **all 7 sections narrated** |

Three things follow, each of which had to be fixed:

**1. `max_tokens` is a cap, not a reservation — so never scale it down.** The output
budget used to be `report_min_tokens + 250 × anomaly_count`. Anomaly count is a poor
proxy for how much there is to *write*, and at zero it is actively inverted: a verdicted
incident can score no anomalies while carrying a full chronology, an asset timeline and
16 condition results to explain across seven required sections. Those sections are a
**fixed cost the anomaly term never covered**, so the incident with the most to explain
got the smallest budget. An unused allowance costs nothing (the model stops when it is
finished), so there is no saving in scaling down — only the risk of truncation.
`_report_max_tokens` now returns the configured ceiling, floored by `report_min_tokens`.

**2. A truncation must report itself as a truncation.** `finish_reason == "length"` is the
endpoint stating plainly that the response is incomplete, and nothing checked it. The
fragment flowed into `_validate_lenient`, which helpfully wrapped the bare `{}` into the
model's lone list field, and pydantic then complained about the first missing key:

```
Native structured output for InvestigationReport failed validation
(2 validation errors for InvestigationReport
sections.0.section_title
  Field required [type=missing, input_value={}, input_type=dict])
```

That reads as a model ignoring its schema. It sends you into `$defs`, `strict: true` and
`additionalProperties` — all of which were verified live to work fine — when the remedy
was one number. `_raise_if_truncated` now raises `LLMTruncatedError` **before** parsing,
naming the budget and the config keys that control it.

**3. Retrying a truncation at the same budget is pure waste.** `LLMTruncatedError`
subclasses `NonRetryableError` and `structured_output` re-raises it instead of falling
through to the JSON-prompting path, because the fallback gets the same `max_tokens` over
the same prompt and truncates at the same place. Unguarded, one doomed ~75-second call
became nine (3 inner × 3 outer) on the largest prompts in the pipeline.

`_narrate_sections` does retry **once**, at `min(tokens × 2, _TRUNCATION_RETRY_CEILING)` —
this is the one failure mode with a known remedy, and the synthesis is worth another call.
A second overrun means the prompt, not the budget, is the problem, so it stops and lets
the deterministic fallback run. A recovered truncation sets `last_truncation_retried`,
which scores the non-gating `narration_truncated_retried` signal: the report *is* complete,
the config is merely undersized for this data volume.

## The same defect hid upstream

`AnomalyDetectionModule.detect` passed no `max_tokens`, so it inherited the client's 4096.
Each `AnomalyItem` carries six prose fields, so a thorough incident overruns that easily —
and the log for the same run shows it did, with `anomalies.0.description Field required`.
Because `detect` is best-effort by contract, that surfaced as:

```
INFO - Detected 0 anomalies for incident 83e94dd6-...
```

A clean, plausible, entirely wrong line — the exact conflation between "broken stage" and
"clean incident" that module's degradation channel exists to prevent. Hence
`detect_max_tokens` (default 8000). The truncation now also trips `last_degraded`, so
stage health can tell the two apart.

## Companion defect: the clamp that deleted its own evidence

Worth reading alongside the above, because it starved the same report from the other end.
`_clamp_by_verdict` demotes anomaly confidence to `_FALSE_POSITIVE_CEILING` (0.5) when the
verdict dismisses a subject, and `filter_anomalies` then drops anything below
`anomaly_detection.threshold` (shipped: **0.8**). Clamp-then-filter therefore deleted
*every* anomaly on *any* FALSE POSITIVE — arithmetically, not analytically — which is
precisely the verdict class under test. "Detailed Findings" had nothing to narrate.

The clamp DEMOTES; it must not DELETE. `filter_anomalies(..., clamped=True)` lowers the
cut-off to the ceiling when the clamp fired: a reviewed-and-dismissed finding is still
evidence a human must read, and the verdict already carries the dismissal.

## Section structure

`_REQUIRED_SECTIONS` lists the six titles the narration must return. `_ensure_complete_
sections` keyword-matches what came back and backfills only the **gaps** from the
deterministic builder (LLM prose is richer where it exists), then `generate` orders
everything canonically. Each backfill scores `section_backfilled`, so a mostly-deterministic
report cannot pass a gate while looking narrated.

### The order is an argument, not a table of contents

`_SECTION_ORDER` interleaves the 6 narrated sections, the 7 deterministic ones and the
pack-titled verdict into one sequence, arranged **conclusion first**:

| Band | Sections | Why here |
|---|---|---|
| The answer | Executive Summary, **Verdict** | A reader who stops after one screen must still have read the verdict. It used to sit sixth, below three condition tables. |
| What was alleged | Alert facts (verbatim), Reconciliation | The claim, before any of AFIR's own reasoning — so a mis-transcribed alert field is visible as such. |
| How it was investigated | Investigation Scope and Method, Evidence Coverage and Gaps | What was asked of which source, and what did not come back. A gap read after the findings reads as an excuse. |
| What the evidence shows | Incident Reconstruction (+ its time-ordered subsection), Condition Assessment, Analysis and Findings, wider sweep | The chronology precedes the per-condition verdicts it explains. |
| What may be done | Impact and Exposure, Authorised Actions, Recommended Next Steps | Actions after evidence, never before. |
| Appendix | Evidence Artifacts | A section after the file listing reads as more file listing. |

Matching is by keyword, **longest key first**, and an unrecognised title lands at
`last_idx - 0.5` — just before Evidence Artifacts — so a section the LLM invents cannot
displace the appendix or land above the verdict.

Three changes are worth the note because each removed a specific defect measured on a live
report:

- **"Incident Overview" was deleted** (7 → 6 narrated sections). It restated the verbatim
  alert facts the deterministic section already prints, which is strictly worse: prose *about*
  an alert competes with the alert. Of 18 sections on the measured report, 7 restated
  another's content.
- **Three condition sections merged into one** `Condition Assessment (per subject, in
  procedure order)`. Scope gate, validation steps and hints were three top-level sections, so
  one subject appeared three times and a reader had to reassemble it. Now each subject has one
  heading, its groups ordered `gate → validation → hint`, each tagged with what its
  PASS/FAIL *means* — a gate's PASS is "the procedure applies", a validation step's PASS is
  "the condition is satisfied", and `_ROLE_MARKS` prints the gate's as `IN SCOPE` /
  `OUT OF SCOPE` so the two cannot be read as the same claim.
- **The deterministic chronology now reaches the narrated artifact.**
  `_incident_reconstruction` was reachable only from `_fallback_sections` and
  `_det_section_for`, i.e. **only when narration failed** — so a *successful* run shipped
  prose with none of the source-cited records it was written from, and the asset rollup (each
  document's current state, the fact containment turns on) never reached the artifact at all.
  `_attach_timeline` nests it as a **subsection** of Incident Reconstruction rather than a
  second top-level section, which would re-create the redundancy just removed. It no-ops when
  that section was itself backfilled (the deterministic text is already there).

## Narration is synthesis, not investigation

The prompt (`_build_system_prompt`) is built from `_REQUIRED_SECTIONS` itself, so a prompt
title and a backfill title cannot drift apart. It carries three absolute prohibitions,
each traceable to an observed failure:

- **Do not INVENT** — no fact that is not in the supplied stage output.
- **Do not REVERSE** — on IR10000004 the narration called a categorical *exclusion* "the
  hallmark of a split-actor fraud pattern": the very fact the expert cited as the reason the
  alert was **not** fraud, reported as the reason it was. The verdict is computed
  deterministically upstream; narration may explain it and may not contradict it.
- **Do not RE-RANK** — the indicator tiering and the decisive conditions are the verdict
  engine's, not the narrator's.

Plus one structural rule: **each fact belongs in exactly one section; overlap is a defect,
not thoroughness.** The sections the deterministic builders own are described to the LLM *by
content*, with an instruction not to reuse their titles — naming a title invites the model to
write the section.

## Report vocabulary is pack-scoped, not hard-coded

`src/report_generation.py` had accumulated ten `§`-numbered clause citations from ONE team's
written procedure, a phase label naming that procedure's central record type, and a prompt
line naming four of its concepts. A clause number is the worst kind of domain leak: it reads as
authoritative, it cannot be checked from `src/`, and it silently mis-cites every other use
case's report.

They now live in `knowledge/<domain>/use_cases/<case>/reporting.yaml` (`phases:` and
`phrases:`), resolved **most-specific-first** — `use_cases.<name>` → flat root → the
engine's own default. Two properties hold at every call site:

- **The default is a complete sentence.** `_phrase(phrases, slot, default, **subs)` returns
  the engine's own wording when there is no pack, no `reporting.yaml`, or no such slot, so a
  pack-less domain still gets a correct report — just without the citations.
- **Substitution is `str.replace`, never `.format`.** Procedure prose contains braces, on
  which `.format` raises.

`_GENERIC_PHASE_RULES` keeps three vocabulary-free phase labels (authentication, data
access, alert) as the fallback, so a pack-less run still gets a segmented chronology rather
than one flat list.
