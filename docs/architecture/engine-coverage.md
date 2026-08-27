# Engine coverage — what "generic" is measured against

The engine is domain-free and mechanically kept that way: `validate_pack` scans `src/` and
`config/templates/` for every word the pack declares in `domain_vocabulary.yaml` — comments and
docstrings included — and reports a hit as a `neutrality-collision` error. That check answers one
question: does `src/` NAME a domain? It cannot answer the other one, which is the one every recent
defect has come through:

> **Generic is not the same as exercised.** A key can be read by `src/`, declared by a pack, and
> have no test anywhere that runs the *combination*. What survives there is not a domain leak — it
> is an engine literal, a missing branch, or a silent bound that no pack has happened to reach yet.

So coverage over the pack surface is **measured**, not asserted, and the measurement is repeatable.
This file records the method, the numbers, and what the numbers found.

## The method

Three read-only passes, no writes, nothing installed:

1. **Enumerate what packs declare.** For every installed pack, for every ruleset, walk the spec and
   count ruleset-level keys, condition `kind`s, and condition-level keys.
   **Go through `p.ruleset_spec(key)`, never the raw `p.rulesets["verdicts"][key]`** — `use:` check
   imports are resolved by that accessor, so iterating the raw mapping yields conditions with an
   empty `kind` and under-counts every shared check in the pack.
2. **Enumerate what shared tests reference.** Every file under `tests/` except a pack's own tests
   — those exercising one specific domain pack rather than the engine; on a branch carrying no real
   domain pack the set is empty — searched for each key/kind as a quoted literal.
3. **Cross them.** A key with a high pack count and a zero test count is the interesting cell.

Both scripts are short enough to keep here rather than in `scripts/`, because the audit is a
periodic measurement and not a gate — it produces a matrix a human reads, not a pass/fail:

```python
# pass 1 — what the installed packs declare
import collections, json
from pathlib import Path
from src.knowledge.pack import load_knowledge_pack

out = {}
for d in sorted(Path("knowledge").iterdir()):
    if not (d / "rulesets.yaml").exists() and not (d / "use_cases").exists():
        continue
    p = load_knowledge_pack(d)
    spec_keys, kinds, cond_keys = (collections.Counter() for _ in range(3))
    names = list((p.rulesets or {}).get("verdicts") or {})
    for k in names:
        spec = p.ruleset_spec(k) or {}          # NOT p.rulesets["verdicts"][k]
        for x in spec:
            spec_keys[x] += 1
        for c in spec.get("conditions") or []:
            if isinstance(c, dict):
                kinds[str(c.get("kind", ""))] += 1
                for kk in c:
                    cond_keys[kk] += 1
    out[d.name] = dict(rulesets=sorted(names), spec_keys=dict(spec_keys),
                       kinds=dict(kinds), cond_keys=dict(cond_keys))
json.dump(out, open("/tmp/axis_audit.json", "w"), indent=1, sort_keys=True)
```

```python
# pass 2+3 — cross it against the SHARED tests (a pack's own tests prove nothing generic)
import collections, json, re
from pathlib import Path

PACK_ONLY = {...}   # tests exercising one domain pack, not the engine; empty on main
shared = {p: p.read_text() for p in sorted(Path("tests").rglob("*.py"))
          if p.name not in PACK_ONLY}
audit = json.load(open("/tmp/axis_audit.json"))
for axis in ("kinds", "spec_keys"):
    keys = sorted({k for pack in audit.values() for k in pack[axis]})
    for k in keys:
        hits = {p.name: len(re.findall(r'["\']' + re.escape(k) + r'["\']', t))
                for p, t in shared.items()}
        print(axis, k, {n: c for n, c in hits.items() if c})
```

`validate_pack` is the other half of the same picture and is called **with a directory**, not a pack
name — the checked-in template lives outside the packs root and has to be lintable too, or the one
pack every author starts from is the one pack nobody ever lints. Either from a CLI:

```bash
python -m src.knowledge.pack_validate knowledge/<domain>        # 0 on findings-but-no-errors, 1 on an ERROR, 2 on a bad path
python -m src.knowledge.pack_validate knowledge/*              # several dirs; the worst outcome wins
```

or from Python, which is what a caller gating a write reads:

```python
from src.knowledge.pack_validate import validate_pack
r = validate_pack("knowledge/<domain>")
r["errors"], r["warnings"], r["infos"]   # 0, 20, 10 — integer COUNTS, not lists
r["counts"]["field_path_lists"]          # 211 — how many lists it could check at all
r["counts"]["projected_path_lists"]      # 115 — and how many against a projection
r["diagnostics"][0]["severity"]          # "severity", never "level"
```

Those five numbers are a **reading of one pack on one day** (`knowledge/<domain>`, re-measured
2026-08-21) and they move whenever the pack does — three of them were stale here for several commits
(204/108 against a measured 211/115), which is the ordinary fate of a count transcribed into prose.
Read them as the shape of the output, not as a threshold: the assertion that a pack yields **0
errors** lives in `tests/test_pack_validate.py`, parametrised over every installed pack, and that is
the only one of the five worth failing a build on.

**The CLI is newer than this document and the gap is the point.** It was named in four places and had
no `__main__` block, so it printed nothing and exited **0** — a stale sentence here (*"not a CLI"*)
recording the defect instead of the decision, which is the shape where a comment forbids its own fix.
It exits non-zero only on an **error**, mirroring `ok`: several diagnostics are claims about prose that
no mechanical check can settle, and a lint that fails on those is a lint somebody switches off. A
directory that cannot be read is its own exit code, because a mistyped path must not read like a pack
with no findings.

Those two `counts` are not decoration. Both path checks are **silent** where a source has no
generated inventory or declares no `projection`, so a pack that documents nothing reads exactly like a
pack that passed — the count is the only thing that tells them apart.

## The measurement (2026-08-17)

| | <domain> | mock_domain |
|---|---|---|
| rulesets | 6 (`abusive_access_fraud`, `ato`, `record_misuse`, `session_anomaly`, `loyalty`, `abnormal_amount`) | 2 (`courier_collusion`, `refund_fraud`) |
| conditions | 101 | 13 |
| distinct condition kinds | 14 | 9 |
| distinct ruleset-level keys | 28 | 14 |

**The honest headline is that shared-test coverage of the pack surface is nearly complete.** Of the
28 ruleset-level keys the two packs use between them, 26 were referenced by at least one shared test
before this round — synthetic specs in `test_correlation.py`, `test_usecases.py`,
`test_pack_validate.py` and `test_api_call_generator.py` cover the surface far better than a reading
of the pack alone suggests. That is the useful part of the measurement: it says where to look, and
the list is short.

**Exactly two keys read by `src/` had zero shared-test references, and both held a latent defect:**

| Key | Declared by | Shared tests (before) | What was wrong |
|---|---|---|---|
| `condition_groups` | 6 <domain> rulesets, 1 mock | 0 | A check whose `report_group` matched no declared group was **dropped from the report** while the verdict went on counting it. |
| `subject_links` | 1 <domain> ruleset | 0 | The derivation list was cut at a hard-coded 40 **with an early return**, and the list is a correction, not a display. |

Two for two is a small sample and not a law. What it does support is the ordering rule: **when a
generic mechanism has no shared test, write the test before trusting the mechanism** — the cost of
looking was two afternoons and the yield was two defects that no pack edit could have fixed.

## What the two defects were

### 1. A check filed under no declared group vanished from the acceptance artifact

`_condition_subsections` (`src/report_generation.py`) built one bucket per **declared** group and
nothing else, so a check whose `report_group` matched no declared id — a typo, a group deleted from
the block, or the blank the key defaults to — landed in no bucket and was **absent** from the
report, while the rollup above it kept counting it.

That is the one thing that section may never do. A decisive FAIL then reads as `1 exclusion fired`
in the summary and a table the reader cannot find it in, and the two remedies a reader would reach
for (fix the verdict / fix the pack's heading) are two different conclusions.

The fix is a synthetic `_ORPHAN_GROUP` printed last, rendered by **exactly the same row builder** as
every other check — a separate branch would be a second answer to "how is a check printed" and would
drift. Its title names the defect (`Checks this ruleset filed under no declared group`) rather than
claiming a procedure heading, and its `role` is deliberately none of the three known ones, because
the engine does not know what those checks mean — only that the ruleset did not say.

**Latent, not observed:** the census over all 8 installed rulesets / 114 conditions found **zero**
orphans. The two authoring-time halves are what make it stay latent — `unknown-report-group` (a name
matching nothing) already existed, and `missing-report-group` (a **blank** `report_group` in a
ruleset that declares groups, which is what the key defaults to) was added in the same round.
Reported as two codes because the remedies have nothing in common: one is a typo or a deleted group,
the other is a condition nobody filed, usually one appended after the block was written.

### 2. An engine literal that manufactured a scope finding

`build_subject_links` (`src/usecases/base.py`) stopped at 40 links and **returned early**. That cap
did not shorten a table: `_reclassify_derived` reads the list to move derived subjects *out of*
"scope the alert failed to name", so the 41st link left a record the alerted actor had created itself
standing in the under-reported-scope work-list.

The cap therefore **manufactured the finding it was cutting** — an engine literal producing a scope
claim, silently, in the direction that alleges more uncontained impact than exists. It is reachable:
one recorded actor-scoped sweep returned 593 subjects against this 40.

The real bound was already there and is not a number: deduplication on the relation (one entry per
distinct subject/related/role/element, whatever the row count). What a **reader** sees is bounded
where bounds belong — in the renderer, which now states what it did not print.

### 3. Every display cut in the report states its remainder

Found by grepping for the sibling of #2 rather than by a live run, and fixed in the same round
because of the existing invariant: *a truncated result must say so, and it must say so everywhere
the count is READ*.

`_cut_note` (`src/report_generation.py`) is the one wording, and the seven cuts that use it are the
one place each bound is named:

| Cut | Bound | Why it matters that it speaks |
|---|---|---|
| derivation link quotes | `_DERIVED_QUOTE_CAP` 10 | the evidence FOR a reclassification |
| un-derived new-scope subjects | `_NEW_SCOPE_CAP` 20 | a work-list somebody has to look at |
| the sweep's in-window asset table | `_WIDER_ASSET_CAP` 40 | read as the exposure total |
| the impact section's asset table | `_ASSET_TABLE_CAP` 60 | same, and reached on the fallback path |
| the chronology | `_CHRONO_LINE_CAP` 40 | **the slice keeps the first N, so a cut drops the tail** — the response, the containment, the reissue |
| the asset-event order | `_CHRONO_LINE_CAP` 40 | as above |
| other incidents' records | `_UNRELATED_QUOTE_CAP` 10 | a **disclaimer**: an id that falls off is a foreign record the report stopped excluding, while the chronology still prints its rows |
| a source's unreconciled values | `_MISSING_VALUE_QUOTE_CAP` 4 | nested in another line, so the compact form |

Two rules the shape of the fix encodes. **One function owns the wording**, because a reader who has
learned one sentence reads a differently-worded list as complete, and a grep for the callers is how
the next person finds every cut there is. And **silence is load-bearing**: `_cut_note` returns `""`
when nothing was cut, so an uncut list says nothing — three "and 0 further" sentences per report
would make the sentence noise, and therefore unread on the report where it matters.

## Where the coverage still is thin, measured

Not defects — a work-list, and the reason the next round of `knowledge/mock_domain/` growth is worth
more than another shared test:

- **5 of 14 condition kinds** have ≤1 shared-test reference: `stub` (21 uses in <domain>, and a stub
  is a *declaration of an absent number*, so it is the kind most likely to be mis-weighted),
  `cohort_membership`, `element_presence`, `value_mismatch`, `delimited_field_mismatch`.
- **`mock_domain` declares none of** `as_of`, `asset_timeline`, `containment_labels`,
  `do_not_consider`, `follow_up_passes`, `identity_classes`, `lock_target`,
  `min_evaluated_to_clear`, `no_exclusion_fired`, `platform_mode`, `reject_when`,
  `scope_discovery`, `subject_links`, `_co_identity` — so every test of those runs against a
  synthetic dict, never a loaded pack.
- Also absent from the fixture pack: `subject_entity: user` (every mock ruleset is record-scoped),
  a `gate` condition that is `decisive: True`, a default `decisive_on`, and a multi-entry `fields:`
  first-present list.

The rule that follows from #1 and #2 above: **grow the fixture pack toward the shapes with no
coverage, and treat a zero cell as a place to look rather than a place to assert.**
