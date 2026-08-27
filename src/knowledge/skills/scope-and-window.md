---
name: scope-and-window
title: "Scope and time: the window is the hard boundary of every answer"
when: "Choosing a lookup depth, a boundary column, a sweep, a row cap or a second retrieval pass."
always: false
triggers:
  - window
  - date
  - time
  - lookup
  - depth
  - default_lookup_days
  - scope
  - sweep
  - scope_discovery
  - cap
  - max_results
  - truncat
  - partition
  - pad
  - blast radius
  - follow_up
  - pass
  - velocity
---

# Scope and time

## The window is the hard scope of every query

Every retrieval is bounded by one date window. Nothing outside it is ever seen, by any
condition, in any report. So a wrong window is not a narrower answer — it is a *different
question*, asked identically and answered confidently.

Three rules, each earned:

* **A window the incident never stated must not be invented.** An undated report has no window;
  a fabricated one (the start of the run's own day, say) becomes the scope of every retrieval in
  the run. So an invented window is cleared.
* **What replaces it is DECLARED, in the pack.** How far back a store must be read to find the
  conduct behind an undated report is a fact about *your* domain — its retention, its alerting
  lag, how long the behaviour typically runs. The engine holds no depth of its own, and with the
  invented window cleared but no depth declared, the generator picked *eight different windows
  across twenty-nine queries*: sources that had to be compared to each other were read over
  different spans.
* **The window is anchored on the incident's ingestion timestamp**, so re-running the same
  incident later reads the same span.

## Which column is the boundary

The boundary column is the one the **procedure's clause** names, not whichever timestamp is most
convenient or most populated. These are routinely different columns with different meanings: when
a record was created, when it was last changed, when the act the procedure cares about took
effect. Pick the wrong one and the scope is populated with real rows about the wrong period.

Two specific traps:

* **A row whose boundary value is absent is `unknown`, not in-window.** Treating it as in-window
  inflates the scope with rows nobody established belong there; treating it as out-of-window
  hides them. It is a third state and must be reported as one.
* **A partition pad can silently become the window.** A pad declared for partition pruning on one
  column will override a cohort window expressed on another — one measured case turned a
  thirty-day cohort of 167 rows into 15. A pad is a *scan* optimisation; it must never be the
  thing that decides which rows exist.

## Do not let a depth restate the data's own age

When you measure a lead time to choose a depth, remember the measurement is **censored by the
data's age**: the maximum observed lead time tracks the age of the partition you sampled (a
96-day partition yields a 97-day maximum, a 369-day one yields 369). That is not a finding about
the domain; it is the sample's boundary.

And check the *direction*. If some fraction of cases need the window to extend **forward** of the
alert — the act came after the report — then a backward-only pad cannot be justified by the same
measurement, and the honest declaration covers both sides.

## A row cap is not a timeout, and both are not each other

Three different limits, and conflating them wastes the fix:

* **The row cap** bounds rows.
* **Every timeout** bounds seconds.
* **Extended retrieval** raises time only — it does not raise the cap.

A truncated result must *say* so, everywhere the count is read: `N rows` and `N rows, and there
were more` are different findings. Where a count is truncated, **counts become floors** and any
status derived from them carries a marker its reader matches on. This matters most on a scope
sweep — the one step that widens scope past the alert. Cut off at the cap it reported "ran, N
assets" in the exact wording an exhaustive sweep uses, and the operator's scope line has to read
"unknown, not clean" instead.

## Widening scope past the alert

The alert names what was noticed, not what happened. A source keyed to the subject the alert
named cannot widen scope — by construction its rows are about that subject. Establishing the
blast radius needs a query scoped to the **actor** or the **shared attribute** instead, and its
result is a three-way status: contained, wider, or *unknown*.

Two traps on that sweep:

* **The actor may sit on either side of the row.** Filtering only the side that raised the alert
  can return zero rows for a perfectly active actor. OR both arms — measured, that was also
  *faster* than either arm alone, and the new limit became the row cap rather than time.
* **The sweep's own cap decides whether its answer means anything.** See above: at the cap, its
  count is a floor and its status is not "clean".

## When one retrieval is not enough

A question can be scoped by values that only appear *inside* a retrieved payload — decoded after
retrieval, where nothing can go and fetch more. Retrieval may therefore repeat, and **the ruleset
declares whether that is worth its scan**: what to harvest, from where, into which source. A pack
declaring nothing runs single-pass, byte-identically.

Four rules if you declare one:

* **A pass ADDS.** Logs merge, queries accumulate, truncation stays per pass. Nothing from pass 1
  is replaced.
* **A follow-up target is deferred out of the earlier passes**, by both the ruleset's route and
  the planner's own pick.
* **The pass-N query IS the narrowing.** Harvested values ride on the query and do *not* become
  the incident's extracted entities — so a condition that resolves its clause values from the
  incident's entities will find none of them.
* **A pass costs a full scan.** Declare one when a question is genuinely unanswerable otherwise,
  not to enrich a result that is already sufficient.
