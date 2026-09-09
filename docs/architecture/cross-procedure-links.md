# Cross-procedure links

One run adjudicates **one** procedure. A real campaign crosses procedures in both directions —
an ATO that *caused* the ticketing fraud in hand, a ticketing fraud that *led to* a loyalty
cash-out — and the reported incident is usually the tip of that iceberg. This is the mechanism
that names the rest of it **without paying another procedure's investigation cost to find out
that most candidates are false positives**.

Everything here lives in **one module** (`src/links.py`, pure and deterministic: no LLM, no IO,
no time) called **once** at the end of `CorrelationModule.analyze`, and in **one advisory field**
(`CorrelationResult.links` / `InvestigationBrief.links`, both defaulting to `[]`). There is no
new pipeline stage — a "reassess" stage would have cost 13 stage-descriptor edit sites and made
the link decision a prompt's judgement, which is the thing every query guard exists to prevent.

## The two lanes

The parent's `verdict`, its condition lines, its severity and its stage-health score are
**byte-identical whether links fired or not**. That is not a convention:
`tests/test_links_never_change_the_verdict.py` runs correlation with `links.enabled` false and
true over every ruleset and asserts the serialised verdict, the brief minus `.links`, and the
health score are equal. A `LinkFinding` is never readable by a parent condition and never feeds
severity — a router adds no evidence, so it must not add confidence (`ROUTER.md` R3).

What links *do* get is their own lane, addressed to a human, labelled as router-added:

```
Verdict          FRAUD CONFIRMED — severity HIGH        (procedure: record_misuse, 19 conditions)
Correlation      advisory HIGH — router-added            (2 links, act via Refer)
                 → ato        antecedent  probed_negative  actor <sign>: no auth anomaly in -30d
                 → mass_access       consequent  not_probed       actor <sign> on 3 declined auths
                 → loyalty    unreachable                  no pivot binds this subject entity
```

## Inbound, not outbound

A ruleset declares **how its own fraud shows up in somebody else's evidence** (`entry_signals:`),
not which siblings to go look at. Procedure #11 is then *one new file*, discoverable from all ten
existing runs, with zero edits to any sibling; the outbound shape is N² and every new procedure
would edit ten shared files. The authored `related_playbooks:` lists are **kept** and become the
reconciliation oracle, not a second answer (`entry-signal-not-related`,
`related-playbook-needs-probe`).

`opens_with.entity` **must equal the declaring ruleset's `subject_entity`** — a leg is opened by a
subject VALUE, never by a striking finding in prose (`ROUTER.md` R2, enforced as the
`entry-signal-subject-mismatch` **error**). Authoring keys and their failure modes are in
`knowledge-pack-authoring.md` §2.5.1.

## The cost ladder

Each rung is 10–100× the previous, and the decision to climb is **arithmetic over declared
facts**. Rungs 0–2 cost no query and no LLM call, so they always run for every candidate.

| Rung | Question | Mechanism | Cost |
|---|---|---|---|
| 0 | Is a value of B's `subject_entity` in hand? | `analysis.extracted_entities`, `verdict.subjects`, `brief.additional_subjects` (typed as A's `subject_entity`), `impacted_assets`, `lock_targets`, harvested pass entities, plus catalog-computed conversion | free |
| 1 | Does B's own applicability test survive A's rows? | the existing pure `evaluate_verdict` over B's spec pruned to its `scope_gate` | free |
| 2 | Did any of B's declared `entry_signals` fire on rows A already retrieved? | the existing row selectors | free |
| 3 | Confirm with one source B owns that A did not retrieve | one isolated probe query | rung-1 PASS + `auto_probe` + the probe budget |
| 4 | Adjudicate B properly | a child job pinned to B's ruleset | rung-1 PASS + a confirmed state + every cap below |

## The escalation precondition

**Rungs 3 and 4 are licensed by the pack's declarations and by THIS run's evidence, and by nothing
else.** The single bar is `gate_permits(gate_outcome)`: rung 1 = `pass`, i.e. B's own `gate: scope`
conditions re-evaluated over the rows A retrieved, through the existing pure `evaluate_verdict`.

- Deliberately **not** `!= "fail"` — a ruleset declaring no gate (`no_gate`) has not said this
  incident is in scope, it has said nothing, and reading silence as consent auto-escalates every
  link a pack ever declared. `unreachable`, `unknown`, `no_sources` and `no_subject` never
  auto-escalate; they surface as candidates and stated limitations.
- Deliberately **not** "a rung-2 signal fired" — the gate answers the prior question, and a row
  match under an unresolved gate is a referral for a human. Such a candidate settles
  `probed_positive` and still refuses at rung 4, on `gate_not_pass`.
- Deliberately **not** a measured per-pair base rate. A historical statistic cannot say whether
  *this* incident is in B's scope, and gating on one made `auto` a setting that shipped and did
  nothing on every pair (see the measured table below: 57 pairs, every one `STUB`).

Rung 4 therefore has **three independent barriers**, in firing order — `not_escalating` (the mode
clamp), `not_confirmed` (the state is not a spawn state), `gate_not_pass` — because there are three
different ways the licence can be absent and a backstop that only fires when another one would is a
backstop nobody has tested.

**The rung-1 trap.** A spec pruned to its `scope_gate` group falls through the rollup to
`spec.get("no_exclusion_fired") or "fraud"`, which would stamp **B's FRAUD label** on A's run. So
`_gate_outcome` reads the **per-condition results and the `None` return only**, never the rolled-up
label, and returns one of `pass | fail | unknown | no_gate | no_sources | no_subject`. A ruleset
with no `scope_gate` at all is `no_gate` and skips to rung 2 — not pruned to an empty condition
list, which would read as agreement.

**Never asked is not empty, at rung 2 too.** `_fired_signals` excludes a source this run never
retrieved from the `evaluated` count, and stamps `finding.signal_id` **before** `_settle`, so a
fired signal stays visible even when a gate FAIL outranks it.

## The four states never collapse

Same discipline as `unanswered_out` vs `zero_rows`: "we did not look" is not "we looked and found
nothing", and the second is a **finding**.

| state | meaning | reported as |
|---|---|---|
| `unreachable` | no pivot in this pack can open B's subject — a binding gap | a stated limitation, naming the missing binding |
| `not_probed` | reachable, but nothing decided it at rungs 0–2 | a candidate with a one-click referral |
| `probed_negative` | we looked and B does not apply | **a positive finding** — checked and ruled out |
| `probed_positive` | B's shape is present in A's evidence | a referral recommendation with an advisory severity |

`probed_negative` is the one that is easy to lose: rendered as silence it is indistinguishable
from a candidate nobody considered, so it renders as a line in the report and a card in the UI.

## Measured: the per-pair table

The measurement replays the local job corpus and reports **per (source → target) pair, never in
aggregate**. It is domain-neutral (the pack is read at runtime) and deterministic (two runs diff
identically).

**Its output gates nothing.** It is an authoring instrument: it tells an author which declarations
discriminate and which have become labels, and its numbers reach a run only as the additive
`signal_discriminates` term in the `semi_auto` confidence score. The table below is kept because it
is *why* the base-rate unlock was removed — a design in which the bars decide is a design in which
nothing ever escalates.

**The bars are pre-committed constants in the script**, so moving one is a visible diff:

| bar | value | why |
|---|---|---|
| `MIN_CORPUS` | 10 distinct incidents | a bar cleared on three runs is not cleared |
| `MAX_FIRE_RATE` | ≤ 50% of the source procedure's runs | a signal that fires on most of the population is a label, not a discriminator (`tender=CC` was on 340 of 586 alerts) |
| `MIN_AGREEMENT` | ≥ 50% | of the runs where it fired *and* B's gate was evaluable, the fraction where the gate did not FAIL — **B's own authored `scope_gate` is the label**, which is what makes precision measurable at zero cost with no hand-labelling |
| gate must FAIL ≥ once | over the pair's corpus | a gate that PASSES on every run of the corpus cannot disagree, so agreement against it is arithmetic and not evidence |

**What a replay cannot reproduce, and which way it biases.** A job export carries the rows
(`<job>.evidence.json`, the verbatim `logs`) and the decoded outputs, but **not** `row_caps`,
`keyed_sources` or `unanswered_sources`. Their absence makes a truncated source read as complete
and an unanswered one read as empty, so a gate the live run read as `unknown` can read here as
decided: the **agreement column is biased upward**, and reachability is unaffected. The script
prints this on every run, because a limitation that lives only in a source file is a number
somebody will quote.

### Result — 2026-08-20, 19 distinct incidents

The corpus is 135 files in `jobs/` → 73 job docs → 61 with a correlation output → **19 distinct
incidents** after deduping on the alert text and keeping the most complete run of each (completed
first, then most outputs, then most recent). A re-run count is not a population.

Runs per adjudicating procedure: `abusive_access_fraud=1, ato=3, record_misuse=6, session_anomaly=2, loyalty=3,
abnormal_amount=4`.

**57 pairs, 174 pair-run cells, and every pair reports `STUB` — no pair clears any bar.** Read as a
licensing gate that was the whole feature switched off: nothing measurable would be measured for
months, so `semi_auto`/`auto` were settings no pack could legally declare. **That measurement is
what retired the gate**, and the bar was not lowered to rescue it — it stopped being a bar. What
escalates today is decided per incident by rung 1, and these numbers say only which declarations an
author should re-word.

The pairs where anything at all happened (full table from the script; `gate P/F` counts the
target gate's PASS/FAIL over the pair's corpus):

| source → target | dir | corpus | reach | fire | agree | gate P/F | verdict |
|---|---|---|---|---|---|---|---|
| record_misuse → refund_void | consequent | 6 | 6 | 3 | 3/3 | 6/0 | STUB: corpus 6 < 10 |
| abnormal_amount → ato | antecedent | 4 | 4 | 2 | 2/2 | 2/0 | STUB: corpus 4 < 10 |
| record_misuse → ato | antecedent | 6 | 6 | 1 | 1/1 | 6/0 | STUB: corpus 6 < 10 |
| record_misuse → mass_access | antecedent | 6 | 6 | 1 | 1/1 | 1/0 | STUB: corpus 6 < 10 |
| abnormal_amount → refund_void | consequent | 4 | 4 | 1 | 1/1 | 4/0 | STUB: corpus 4 < 10 |
| loyalty → ato | antecedent | 3 | 3 | 1 | 1/1 | 1/0 | STUB: corpus 3 < 10 |
| session_anomaly → abusive_access_fraud | antecedent | 2 | 2 | 1 | – | 0/0 | STUB: corpus 2 < 10 |
| session_anomaly → ato | antecedent | 2 | 2 | 1 | 1/1 | 2/0 | STUB: corpus 2 < 10 |
| abusive_access_fraud → ato | antecedent | 1 | 1 | 1 | 1/1 | 1/0 | STUB: corpus 1 < 10 |
| abnormal_amount → record_misuse | consequent | 4 | 4 | 0 | – | **0/4** | STUB: corpus 4 < 10 |
| *(48 further pairs)* | | 1–6 | = corpus (except the 4 label rows) | 0 | – | 0/0 | STUB |

Four readings worth keeping:

- **The ladder prunes for free, on the real corpus and not only in a fixture.** The only gate
  FAILs anywhere are `abnormal_amount → record_misuse`, 4 of 4 → four `probed_negative` cells. A
  candidate ruled out at rung 1 costs one pure function call.
- **State totals across the 174 cells:** `not_probed` 132, `probed_positive` 28, `unreachable` 10,
  `probed_negative` 4. Gate outcomes: `unknown` 129, `pass` 28, `no_spec` 10, `fail` 4,
  `no_sources` 3. A declared signal fired in 12 cells.
- **All 10 `unreachable` cells are unresolvable *labels*, not binding gaps** — `record_misuse → ALL above`
  (4) and `→ cross_silo` (3 + 2 + 1, a router that is deliberately never selectable). Not one
  candidate was unreachable for want of a pivot, which corroborates the separate finding that the
  loyalty leg is a rung-3 probe rather than the binding gap the design expected: across 43 legs, 0
  are truly unreachable and 4 are reachable only outside the declaring procedure's own sources
  (`related-playbook-needs-probe`).
- **`agree` can be vacuous, hence the fourth bar.** `record_misuse → refund_void` reads 3/3 agreement, but
  `refund_void`'s gate PASSES on all 6 record-misuse runs — it never disagreed with anything, so 3/3 is
  arithmetic. It changes no verdict today (everything is STUB) but it stops a future stub
  retirement resting on a label that cannot say no.

**Corpus drift, reported rather than blurred.** 4 of the 19 runs recorded
`PB-CROSS-SILO-010 → record_misuse`: a playbook that has since been rewritten as the non-adjudicating
router, so their candidate list is today's declaration and not the one that ran. The script prints
this caveat line and flags the runs. It is the historical `default_ruleset` fallback that
`ROUTER.md`'s R1 refusal now prevents, not a current defect.

### The check: a declaration this corpus contradicts

Nothing in the engine reads `jobs/`, and no number here reaches a run except through the confidence
score. What the script still owns is the **reverse** direction — a declaration that has since become
a population — because a rate is transcribed by hand once and then nothing re-reads the corpus:

```
a replay of the job corpus  →  entry_signals[].base_rate in the pack  →  base_rate_measured()
   →  link_score()'s signal_discriminates term  (run time)
   →  entry-signal-unmeasured / -broad-selector  (authoring time)
```

- **The corpus bar is imported, not restated** — `MIN_CORPUS = link_escalation.MIN_BASE_RATE_CORPUS`.
  It is asked in three places that never see each other's inputs — a pack at authoring time, a
  confidence score at run time, and a corpus here — and all three must agree on what *measured*
  means, or a pack is warned about a rate its own score is already crediting. The validator's copy
  is pinned equal by `test_the_base_rate_READING_has_ONE_home` and the script's is the import.
- **So the report's second half is per declaration** rather than per pair, and `--check` exits
  non-zero on one state only:

| state | meaning | `--check` |
|---|---|---|
| `ELIGIBLE` | stub today, and every bar cleared — printed as the `base_rate:` block to paste | 0 |
| `AGREES` | declared and measured, both under the fire-rate bar | 0 |
| `CONTRADICTED` | declared as a discriminator, and this corpus says it fires over the bar | **1** |
| `INERT` | declared, but under the corpus bar — the score credits it nothing | 0 |
| `UNCHECKABLE` | never asked on this corpus, so nothing here bears on it | 0 |
| `THIN` | asked on too few incidents here to contradict a declaration | 0 |
| `STUB` | the honest placeholder, with what it is short of | 0 |

Only a contradiction fails, and the asymmetry is the usual one: reporting one costs an author a
re-measurement, missing one leaves a signal contributing confidence it has not earned. But a local
job history is not the only corpus a number may have been counted on, so *"I cannot check this
here"* must never render as *"this is wrong"* — hence `UNCHECKABLE` and `THIN` beside
`CONTRADICTED`. And **nothing is written to the pack from here**: a declaration is an authored act,
and an automatic edit would make the pack the output of the measurement instead of the claim under
measurement.

**A declaration's denominator is not its pair's corpus.** A pair's corpus is every run where the
target was a candidate; a signal's is every run where that signal was **asked** — its source
present in the rows already retrieved. The two are different numbers on purpose (a signal on a
source nobody retrieved was not silent, it was unasked), so the per-signal counts come from asking
`_fired_signals` one signal at a time rather than re-deriving the test.

**Result — 2026-08-20, the same 19 incidents: 9 declarations, all `STUB`, nothing eligible.**

| declaration | asked on | short of |
|---|---|---|
| `record_misuse/record_misuse_detector_alerted_on_record` | 3 | corpus < 10 |
| `session_anomaly/session_anomaly_detector_alerted_on_actor` | 5 | corpus < 10 |
| `mass_access/mass_access_detector_alerted_on_actor` | 4 | corpus < 10 |
| `abnormal_amount/abnormal_amount_alerted_on_actor` | 2 | corpus < 10 |
| `abusive_access_fraud/impersonation_recorded` | — | no pair carrying it clears every bar |
| `ato/rejected_authentication_burst` | — | no pair carrying it clears every bar |
| `refund_void/reversal_on_record` | — | no pair carrying it clears every bar |
| `inventory/seat_spinning_predicted` | 0 | never asked on this corpus |
| `loyalty/enrolment_agent_type_marker` | 0 | never asked on this corpus |

Under the design these numbers replaced, that table was the licence and every pair therefore shipped
`planned`. It now says only that no declaration here contributes a `signal_discriminates` term yet.
The two `never asked` rows are still the ones to watch: their sources are ones no run in the corpus
retrieved, which is a *reachability* fact about the catalog and not a rate, and no number of
additional runs of today's procedures will move them.

## Escalation modes

Per link, mirroring the job stage-gate modes: **`planned`** (compose only, a human executes),
**`semi_auto`** (escalate automatically only at or above a configured link-score threshold,
otherwise propose — **the default**), **`auto`** (escalate automatically, still bounded by every
cap). Settable in `correlation.links` config, overridable per pair in the pack, per job through the
API, and shown as a control on each link card; every automatic escalation is recorded in
`Job.interventions` and survives export/restore.

**`semi_auto` is the default and `auto` is not**, because the two differ in who bears an error. Both
require rung-1 PASS, so neither can escalate a link this run did not establish; what `semi_auto` adds
is that a *thin* PASS — a gate holding with no signal fired and no measured declaration behind it —
composes a referral rather than launching one. `planned` is not the default either: it is a mode that
never spends, and shipping it as the default is what made the previous design's whole paid half
unreachable.

### The link score (`src/link_escalation.py`)

The one number `semi_auto` is gated on. Deterministic from the free rungs only — a rung-0 pivot in
hand, the rung-1 sibling gate outcome, a rung-2 entry signal firing, and that signal's declared
`base_rate` — never from prose and never from an LLM. It reuses `src/stage_health.py`'s
*mechanism*: named signal codes, config-overridable weights merged over defaults, the reasons
shipped beside the number, one comparable value clamped to `[0, 1]`.

**Its polarity is INVERTED from stage health, deliberately.** A stage starts at `1.0` and each
fired signal SUBTRACTS. A link starts at `0.0` and each held rung ADDS, because a candidate about
which nothing is known is not a confident one — subtract-from-1.0 would score an unassessable link
at `1.0` and escalate it. The weights partition 1.0, so the threshold reads as *the fraction of
the free evidence a deployment insists on*:

| signal | weight | why |
|---|---|---|
| `sibling_gate_holds` | 0.35 | the only term authored by the TARGET procedure, about its own rows, for exactly this question |
| `entry_signal_fired` | 0.30 | a declared signal, scaled by its own `strength` where one is declared |
| `signal_discriminates` | 0.20 | `weight × (1 − fires_on/of)` — the base rate contributes DISCRIMINATION, not presence |
| `pivot_in_hand` | 0.15 | rung 0, and the cheapest thing to have |

`DEFAULT_MIN_ESCALATION_SCORE = 0.6` mirrors the stage gates' own default and is reachable by more
than one combination of the four. `correlation.links.min_escalation_score` is `live` and clamped to
`[0.0, 1.0]`: at `0.0` `semi_auto` *is* `auto` under a more cautious-sounding name, which is why
that end of the interval is a documented setting rather than a bug.

**A weighted sum cannot express an impossibility, so two facts VETO to `0.0`** rather than merely
withholding their term: the sibling gate FAILING (`probed_negative` — the target procedure saying
it does not apply, which no quantity of other evidence outranks) and no pivot value in hand (a
referral is scoped BY the pivot, so there is nothing to escalate). Both are asked before any
weight is read.

**Discrimination is the one term a historical statistic reaches, and it is purely ADDITIVE.** A
`base_rate` counted over too few runs, or a `kind: stub`, contributes nothing and says so — it never
subtracts and never withholds a licence, so a pair nobody has counted escalates on exactly the terms
a counted one does, one term lighter. An absent `strength` is **no discount, not a zero** — the same
reading `_advise` already gives it.

`score is None` means the gate does not fire, which keeps every caller that resolves a mode for a
pair with no candidate yet (`set_job_link_modes`) byte-identical.

**`mode_licensed` is a permission and `link_score` is a quantity, and the two must not merge.** A
licence is a yes/no the pack and this run's evidence decide; a score is a magnitude that only ever
adjusts a threshold comparison. Folding the score into the licence would let a low number withhold a
licence the declarations granted — the additive rule above, inverted — and folding the licence into
the score would give a refusal a magnitude it cannot have. They are computed apart and read apart in
`src/stage_health.py`.

**`clamp` and `score` are two different `MODE_SOURCES` with opposite remedies, and collapsing them
is the expensive mistake.** `clamp` is the rung-1 refusal and wants *evidence* — a source retrieved,
a pivot bound, or the honest answer that the target procedure does not apply here. `score` wants
nothing: the link is licensed, and this candidate's free evidence did not reach the threshold, which
is `semi_auto` behaving as defined and not a refusal. Each non-permitting rung-1 outcome is worded
separately (`_GATE_REFUSALS`, keyed by outcome) because the reader's next action differs per row —
chase a source, chase a pivot, or nothing at all. `resolve_link_mode` asks rung 1 FIRST, then the
score gate, then the budget note, and only `semi_auto` is score-gated.

**And a fourth reading of the same `planned`, which the seam used not to explain because it is the
one outcome the seam does not IMPOSE: a narrower layer's declaration overruling a wider layer's
escalating ask.** Precedence is narrowest-wins, so a pack that declares `mode: planned` for a pair
beats a deployment configured to `auto` — correctly, and silently, because the three notes were
written for the three refusals this function performs itself. Measured live: a run with the
deployment set to `auto` produced one candidate at mode=`planned`, source=`pack`, rung 1 = **PASS**,
`note` **empty**, beside nine clamped candidates each carrying a full sentence — i.e. the silence
landed on the one row whose gate held, where an operator reads the default meaning ("nobody set this
pair up") and concludes their setting was lost. It is now noted at the seam, printed in the
assessment log line, and rendered per candidate plus once as a section rollup, with the *layer*
named because editing that declaration is the only remedy that applies — unlike `clamp` (retrieve
more) and `score` (do nothing). Two things keep it from becoming boilerplate: the note is a claim
about an **explicit** ask and never about `DEFAULT_LINK_MODE` (itself escalating, so comparing
against the effective mode-so-far would print it on every `planned` declaration of every default
deployment), and the report's predicate lives in one place (`_mode_held_by_declaration`) read by
both the rollup and the candidate line. A declared hold is also reported as a hold **even where rung
1 would have refused too**: nothing was taken away, so a clamp's wording would send a reader to
retrieve rows for a link that stays held after they arrive.

## Bounds (rungs 3–4)

**These are the anti-drain protections, and they are the whole of them** — the base-rate bar used to
sit in front of them as a second tripwire, and removing it changed none of these. Every one is
asserted independently in `tests/test_link_probe.py` / `tests/test_link_children.py`.

- **The cheapest bound is the rung not taken, and on the flagship pair it is the one that fires.**
  `probe_candidates` subtracts what this run already retrieved from what the target's declarations
  nominated — whatever that source answered, **zero rows included**, because the emptiness is an
  answer rung 2 has already read and re-asking would spend a scan to reproduce it. Measured on job
  `49e3cf7d` (record-misuse → `abusive_access_fraud`): both nominations were in hand (`auth_events`, the
  `auto_probe` one, and `audit_raw_access`, a gate-scope one), so `probe_spent` is **false** and rung 4
  still spawned two children off rung 1 alone. **A `probe_spent: false` beside a nomination is
  therefore not a defect to chase** — it is a fact about the *declaration*, which is why
  `probe_nominations_already_answered` reports the difference rather than leaving it inside a
  filtered list where "nothing nominated" and "everything nominated is in hand" read identically.
  It holds on every incident whose planner reaches the same source, and the remedy — if there is
  one — is a nomination the source procedure cannot reach, never a second ask.
- Probe rows live in `ctx.link_logs`, **never** in `logs`. Namespaced keys inside `logs` are
  unsound: 16 enumeration sites read it, the worst being `_merge_co_identified_subjects`,
  `aggregate` (a FATAL `no_records`), `src/stage_health.py` and `src/follow_up.py`.
- `max_probes_per_run` plus a timeout slice, so a fan-out of weak links can never reach
  `primary_source_timeout_seconds` (7200s).
- **Both paid rungs ship armed and narrow — 2 probes and 1 child run — and each is disarmed on its
  own.** They shipped at `0` for one release, which made every bound above and below them
  unreachable in a shipped configuration: not one of the nine refusal codes had ever fired outside a
  test, so the rung nobody could reach was also the rung nobody exercised. Two probes cost less than
  one primary source's budget and one child is the smallest spend that is still a spend. Only a
  number at or below `0` disarms a rung (an absent or unreadable value takes the default, so an
  intended off-switch cannot be a typo), it disarms **that** rung only, and a disarmed rung keeps the
  referral. Hence the per-rung `probes_budgeted` / `children_budgeted` flags on
  `GET /api/v1/jobs/{id}/links` and in the run-controls panel: one number cannot say which of two
  rungs is off, and the lane-wide `escalation_budgeted` answers a different question — whether the
  lane can spend at all.
- A child job is pinned to its ruleset through the **single** `select_correlation_spec` /
  `ruleset_key_for` seam — never a second resolver, because prose-driven handoff is the largest
  risk in the manual fan-out `ROUTER.md` §3 documents.
- A depth cap, a per-run child budget, a cycle guard on `(procedure, pivot_value)` so A→B→C→A
  terminates, and `max_total_children` per parent checked at spawn time **independently** of both,
  so a failure of either cannot run unbounded. Hitting a cap stops spawning and records a
  `Job.interventions` entry; it is not an error.
- **A bound that fires must land on the CANDIDATE, and four of the nine reasons are underivable from
  the row.** `unpinnable`, `depth_cap`, `cycle` and `run_budget` are facts about the *run* or the
  *pack*, not about anything the persisted `LinkFinding` carries — the other five restate `mode`,
  `state`, `gate_outcome` or `pivot_values`, or already leave an intervention. So
  `plan_child_spawns` carries the finding on each refusal row (it never mutates it — planning stays
  pure) and `_spawn_link_children` writes the reason onto `child_note`, the same field a launch
  writes into. Measured on job `a577862d` — itself a referral, at chain depth 1 — whose one candidate
  refused on `depth_cap` while reading `state=probed_positive`, `gate_outcome=pass`, `mode=auto`,
  score 0.5, severity HIGH: **every barrier the row exposes said SPAWN**, the row showed no child and
  no reason, and the only trace was one `logger.info` on a server whose log had rotated away. Three
  couplings make that note reachable, and each was its own defect: `scope` is per **call site** and
  not per code, because `total_budget` is refused both run-wide and per candidate, and a run-scoped
  refusal rides on `findings[0]` merely to have somewhere to sit — stamping it would tell one
  arbitrary row that a run-wide fact is about it; `_spawn_link_children` returns the number of rows it
  **stamped** and not the children it launched, or a run that refused every candidate skips the
  caller's stage-summary rebuild and every note stays on an in-memory object nothing re-reads; and the
  UI badge keys on `child_job_id` and not on the note's presence, exactly as `linkProbeLine` keys on
  `probe_spent`, or the card prints *child run launched* above the sentence naming the cap that
  stopped it.
- Children start **after** the parent's retrieval, against the shared
  `Semaphore(max_concurrency=4)`.

## The sibling lane: this procedure's own open questions

Everything above answers *does **another** procedure apply to these rows*. The other question a
reader of a finished report actually asks is *what could **this** procedure not settle* — and the two
are the same shape on different axes, so the open-question lane (`src/inquiry.py`, the pure planner;
`src/inquiry_probe.py`, the one rung that spends a query) is built out of this one's parts rather than
beside them. A ruleset declares `open_questions:` (`knowledge-pack-authoring.md` §2.5.3); a pack
declaring none produces no finding, no probe and no field content, byte-identically.

What it inherits verbatim: the purity discipline (no IO, no clock, no `async` in the planner,
AST-asserted), rows arriving as an **argument** to the settlement and never merged into `logs`, the
query built through `build_manual_query` so every `_attach_query_guards` guarantee applies for free,
coded refusals instead of silences, and the invariance gate —
`tests/test_inquiries_never_change_the_verdict.py`, the same `_sacred()` comparison with `inquiries`
added to the exclusion set. `InquiryFinding` sits on `CorrelationResult.inquiries` /
`InvestigationBrief.inquiries`, exactly where `links` sits.

What is genuinely its own, and each of the four is a place the naive version collapses two facts:

- **Five states, not four, and the order is the sort key**: `answered`, `not_asked`, `empty`,
  `unanswered`, `unreachable` — actionable first, so a reader who stops halfway has seen every
  question with something to say. `empty` and `unanswered` are the pair the whole lane exists for: a
  source that answered with no rows is a finding whose meaning the **pack** wrote, a source that never
  answered is a credential or catalog gap, and the remedies are opposite.
- **The free rung is most of the value.** Where the run already retrieved the source a question names,
  the question is settled by reading those rows again — no probe, `probe_spent` stays `False`, and it
  is a separate named entry point (`settle_from_rows_in_hand`) rather than a flag, so a call site
  cannot mistake one for the other. The budget bounds only the questions whose source was *not*
  retrieved.
- **One shared clamped ceiling with this lane, subtracted rather than added.** `inquiry_budget` gives
  the count its own key (`max_inquiry_probes_per_run`, shipping at **1** — narrow but armed, for the
  reason the rungs above ship armed) and then caps `count × timeout` at what
  `links.max_probes_per_run × links.probe_timeout_seconds` leaves of `PROBE_BUDGET_CEILING_SECONDS`.
  So raising a link budget **narrows** this lane instead of widening the run, and the subtraction uses
  the link lane's *budgeted* worst case: a bound whose value depends on how slow the other lane
  happened to be is not a bound anybody can state in advance. With both lanes at their defaults the
  link lane budgets 240s and this one gets 120s of the remaining 660s.
- **A non-answer does not consume the probe COUNT; the wall clock is the hard bound.** Exact parity
  with `run_link_probes` — `spent += 1` sits only under `answer is not None` — because a timeout that
  burned the budget would let one unreachable source silence every question behind it, while a
  `deadline_seconds` check before each probe stops a slow lane regardless. A slice is reported through
  the shared `slice_text`, which both lanes now use: `int(seconds)` printed a nearly-exhausted deadline
  as an `0s` slice, i.e. in the words a lane configured to spend nothing produces, and those two facts
  have opposite remedies.

Two smaller ones worth knowing before editing the settlement: `row_cap_hit` is measured on the rows
that came **back**, before the declaration's own `where` selector runs (the cap bounds the retrieval,
so a selector keeping two of a capped hundred still owes the reader *and there were more*), and a
selector that raises yields `unanswered` with nothing claimed rather than a reading of every row
against a question nobody asked.

## The honest limitation

This mechanism covers **actor- and record-keyed** links well — `user`, `record`, `office`, `ticket`,
`ip_address` — because those are what the pack binds. It covers **campaign- and IOC-keyed** links
not at all: `ioc`, `campaign_id`, `asn` and `country` bind only on three sources with no schema,
and `device_fingerprint`, `user_agent`, `card_bin`, `debit_number` bind on **zero**. That inverts the
cross-silo procedure's own emphasis, and the response is to *report* the gap per candidate
(`unreachable`, with the missing binding named) rather than to hide it.
