---
name: fixtures-and-verification
title: "Fixtures and verification: proving a change did what you think"
when: "Writing a fixture or test for a pack change, or claiming a change fixed something."
always: false
triggers:
  - test
  - tests
  - fixture
  - fixtures
  - regression
  - mock
  - verify
  - verified
  - prove
  - replay
  - rerun
  - re-run
  - corpus
  - baseline
  - acceptance
  - green
---

# Fixtures and verification

A pack change is a claim about behaviour on data nobody has seen yet. The fixture is what turns
that claim into something checkable, and almost every way a fixture can be useless is a way it
still passes.

## Prove the test can fail

A new assertion is worth nothing until it has been seen to fail. Break the thing deliberately,
watch the test go red, put it back. Two ways that proof itself lies:

* **A mutation that changes no behaviour proves nothing.** Editing a comment, a log message or a
  key nothing reads leaves the test green and tells you only that the test is insensitive to
  irrelevancies. Mutate the value the assertion is *about*.
* **Cached bytecode can outlive the revert.** A same-length constant change reverted in place can
  be served from stale compiled bytecode, so a restored file keeps failing and you chase a
  phantom. Clear the caches, and never undo a mutation by discarding version-control state — the
  edit you are reverting may not be the only edit in the tree.

## A fixture where everything passes cannot detect a check that never runs

This is the central fixture trap. A clean fixture exercises the *happy* path of every condition,
and a condition that is silently never evaluated looks exactly like one that passed. A signal
that has been dead for months stays green through every run.

So a fixture set needs both polarities:

* a case the procedure should **clear**, and
* a case it should **flag**, on the specific condition you are adding.

And the assertion must name the condition, not just the verdict. A verdict is reachable by many
routes; asserting only on the verdict lets the right answer arrive for the wrong reason.

## A mock must vary with its input

A stub whose return value is a fixed shape satisfies any call. One measured case: an encoder
mocked to return a fixed number of vectors passed for *every* input count, which kept alive a
test that could only have passed on an empty result. If the thing being mocked has a
relationship between input and output — one vector per document, one row per key — the mock must
honour it, or the assertion is a constant dressed as a check.

The same applies to a probe helper that returns an empty list both for "no matches" and for
"the request was rejected": every candidate is cleared, on the one step that could have
disqualified them. `pack_probe` keeps the two channels apart (`ok` beside `rows`) for this
reason — use it rather than a scratch helper, and shape a fixture row from
`pack_probe leaves <source> <table>` output rather than from a schema document, because the
leaves a ROW carries are what a condition reads.

## Verify a guard by ITS note, not by an outcome that has other causes

If a change is meant to make a specific mechanism fire, confirm *that mechanism fired*. An
outcome — a degraded status, a withheld clear, a different row count — usually has several
possible causes, most of them pre-existing. Two rules that follow:

* **Read the mechanism's own log line or note.** That is the only evidence that this change,
  rather than something else, produced the result.
* **Diff the executed query before attributing a row-count delta.** Two runs differ in more than
  your edit; the query text is the thing your edit was supposed to change.

And the degenerate case: **a re-run that returns zero rows verifies nothing at all.** Every
downstream difference it produces comes from the empty-result paths, which were already there.

## Replay offline rather than re-running live

A recorded run holds what the pipeline actually did, so most verification can be done against it
with no credentials and no scan cost — replay the recorded stage inputs through the code path you
changed. This is strictly better than editing the pack to force a live route: one such edit
closed the very route it was meant to test, and the live run then proved nothing.

One caveat that has cost a whole afternoon: **an exported shape is not the run shape.** An
evidence export may be *flattened* relative to the rows the run held, so a replay built on it
derives empty intermediate structures and any pack-bound mechanism reads as inert — a false
negative that looks like a broken feature. Check the shape you are replaying against the shape
the stage consumes.

## A corpus is a population, not a pile of runs

A directory of recorded runs is *runs*, not *incidents*: the same incident re-run ten times is
one data point, and a cancelled run beside its completed retry counts twice. Deduplicate on the
incident's own text, prefer the completed run, and say how many distinct incidents the number is
over. A base rate computed over re-runs is a measurement of your own debugging habits.

## Assert on what the consumer actually reads

A fact is only load-bearing where it is *reached*. Prose that reaches a prompt as its first few
hundred characters is, for that consumer, only those characters — so put the load-bearing claim
in the first sentence, and assert on the **snippet the consumer receives**, not on the file
containing it. A test that greps the whole document passes while the model never sees the line.

## Keep the tree quiet while the suite runs

Pack tests copy the installed pack tree, so an edit landing mid-run fails tests that pass on a
quiet tree. Before believing a failure, re-run it against an unmodified tree. Inventing a defect
that does not exist costs as much as missing one.

## The acceptance run

The strongest available check on a pack change is the one that holds the *engine* fixed and swaps
the *pack*: replay a known incident and compare the resulting condition lines against a recorded
baseline, line by line. Anything that differs is either your change or a defect, and the
difference has to be explainable in one sentence. "Most of it matched" is not a result; the count
of lines that matched verbatim and a named explanation for each that did not is.
