"""Tests for `src/knowledge/pack_validate.py`.

Roughly half these tests assert silences: a false positive on correct authoring
teaches the operator to scroll past the list, hiding the real defects. Each
section's positive control is load-bearing for this reason.

The condition-kind list and the three `expected_label`-honouring kinds are derived
from `correlation._eval_condition` at runtime, so adding a new kind needs no edit
here. The neutrality pattern is checked against both copies (this module's and the
engine's) because a drift lets one enforce a weaker standard than the other.

Per-pack counts (how many unread keys, how many mislabelled indicators) are pinned
in each pack's own test file. What is asserted here is the checker's behaviour over
a fixture that ships on every branch.

Mutation tests run on writable copies of the checked-in template under `tmp_path`.
Nothing here writes into `knowledge/` or `docs/`.
"""

import shutil
import sys

import pytest
import yaml

from src import link_escalation
from src.knowledge import pack_validate as pv
from src.utils.paths import REPO_ROOT
from tests.installed_packs import installed_packs

TEMPLATE = REPO_ROOT / "docs" / "knowledge-pack-template"
PACKS = REPO_ROOT / "knowledge"
RULES_REL = "use_cases/example_use_case/rules.yaml"


# --------------------------------------------------------------------------- helpers


@pytest.fixture
def pack(tmp_path):
    """A writable copy of the checked-in pack template."""
    root = tmp_path / "scratch_pack"
    shutil.copytree(TEMPLATE, root)
    return root


def codes(result, severity=""):
    return [
        d["code"]
        for d in result["diagnostics"]
        if not severity or d["severity"] == severity
    ]


def find(result, code):
    return [d for d in result["diagnostics"] if d["code"] == code]


def edit_rules(pack, mutate):
    """Load the template ruleset, hand the ruleset body to `mutate`, write it back.

    A `safe_dump` round trip is fine HERE and nowhere near the store: this is a throwaway
    fixture whose comments and anchors nobody reads. `pack_store` may never do this, which
    is what `test_pack_store.py` asserts on output bytes.
    """
    path = pack / RULES_REL
    doc = yaml.safe_load(path.read_text())
    mutate(doc["verdicts"]["example_use_case"])
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def one_condition(spec, **overrides):
    """Replace the template's placeholder condition with one shaped for a single check."""
    cond = {
        "id": "probe",
        "kind": "field_flag",
        "source": "record",
        "label": "Probe",
        "report_group": "validation",
    }
    cond.update(overrides)
    spec["conditions"] = [cond]


# ------------------------------------------- the shipped packs, and the shipped template


@pytest.mark.parametrize("name", installed_packs())
def test_every_installed_pack_has_zero_errors(name):
    """Parametrised over whatever is installed, not over a list of names.

    The branches differ in which packs they ship and this file must be byte-identical
    on every branch.
    """
    root = PACKS / name
    result = pv.validate_pack(root)
    assert result["ok"], [
        (d["code"], d["path"], d["line"], d["message"])
        for d in result["diagnostics"]
        if d["severity"] == "error"
    ]
    assert result["errors"] == 0


def test_the_checked_in_template_has_zero_errors():
    """The template lives outside the packs root; an error here makes a fresh pack unsavable."""
    result = pv.validate_pack(TEMPLATE)
    assert result["ok"], [
        (d["code"], d["path"], d["line"]) for d in result["diagnostics"]
    ]


def test_counts_describe_the_pack_rather_than_the_diagnostics():
    result = pv.validate_pack(TEMPLATE)
    counts = result["counts"]
    for key in ("files", "yaml_files", "entities", "sources", "rulesets", "conditions"):
        assert key in counts, key
    assert counts["entities"] >= 1
    assert counts["sources"] >= 1
    assert counts["conditions"] >= 1
    assert counts["vocabulary"] >= 1


def test_a_missing_pack_directory_is_an_error_not_an_exception(tmp_path):
    """An HTTP handler validating an unknown pack must render, not 500."""
    result = pv.validate_pack(tmp_path / "nope")
    assert result["ok"] is False
    assert codes(result) == ["pack-missing"]


def test_diagnostics_are_ordered_errors_first(pack):
    (pack / "source_catalog.yaml").write_text("sources: [ *dangling ]\n")
    edit_rules(pack, lambda spec: one_condition(spec, kind="not_a_kind"))
    result = pv.validate_pack(pack)
    severities = [d["severity"] for d in result["diagnostics"]]
    assert severities == sorted(
        severities, key={"error": 0, "warning": 1, "info": 2}.get
    )
    assert result["errors"] >= 2


# ------------------------------------------------- two defect classes, fixture-asserted
#
# Per-pack counts are pinned in each pack's own test file. What belongs here is that
# the checker reports each class with a usable cursor over the fixture pack.


def test_an_unread_ruleset_key_is_reported_with_a_cursor(pack):
    """A ruleset key nothing in `src/` reads is a sentence the author believed took effect.

    The fixture key is deliberately unspellable: `_is_read_by_engine` greps `src/` for
    the literal, so a plausible key name here makes the test a countdown to its own repair
    when that key is wired. The positive control is the other half: without it, a grep that
    started matching everything would satisfy the silence above by reporting nothing.
    """
    edit_rules(
        pack, lambda spec: spec.update({"zzz_no_engine_reads_this": ["something"]})
    )
    hits = find(pv.validate_pack(pack), "unread-pack-key")
    assert [h["severity"] for h in hits] == ["warning"]
    assert "zzz_no_engine_reads_this" in hits[0]["message"]
    assert hits[0]["line"] > 0, "the operator needs a cursor, not just a filename"

    # `edit_rules` re-loads what it last wrote, so the unspellable key has to come back OUT
    # or the control asserts nothing about the key it names.
    def _swap(spec):
        spec.pop("zzz_no_engine_reads_this", None)
        spec["do_not_consider"] = ["something"]

    edit_rules(pack, _swap)
    assert find(pv.validate_pack(pack), "unread-pack-key") == [], (
        "`do_not_consider` IS read by the engine (the report states the carve-outs it "
        "cannot apply), so warning on it would be a false positive on a wired key"
    )


def test_a_requirement_phrased_indicator_label_is_reported_with_a_cursor(pack):
    """An indicator's label is printed verbatim as the finding.

    A requirement-phrased label under `polarity: fraud_indicator` makes the report state
    the opposite of what was found. The engine cannot negate prose, so authoring time is
    the only place this can be caught.
    """
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            polarity="fraud_indicator",
            label="Issuer matches the stated country",
            expected="issuer matches the stated country",
        ),
    )
    hits = find(pv.validate_pack(pack), "label-polarity-unaffirmed")
    assert len(hits) == 1, [h["detail"] for h in hits]
    assert hits[0]["severity"] == "warning"
    assert hits[0]["line"] > 0


def test_the_label_heuristic_is_quiet_on_correctly_worded_indicators():
    """The reason the heuristic is two signals and not one.

    A pure "imported under indicator polarity without overriding `label`" rule has a high
    false-positive rate. The discriminator is that a correct finding either uses different
    words from its requirement or differs by a negation; these cases pin that boundary.
    """
    # Same subject, opposite sense: high word overlap, one extra negation.
    assert not pv._label_echoes_requirement(
        "Cash form of payment used", "non-cash form of payment"
    )
    # Different wording entirely — no overlap to judge.
    assert not pv._label_echoes_requirement(
        "Booking created and ticketed within minutes", "a plausible interval elapsed"
    )
    # Negative phrasing that is nonetheless the finding.
    assert not pv._label_echoes_requirement(
        "Non-agency email address on the record", "an agency address"
    )
    # The defect: the requirement, restated, with the same negation count.
    assert pv._label_echoes_requirement(
        "Passport issuer matches the stated country",
        "passport issuing state matches the stated country",
    )


def test_a_condition_with_no_expected_is_not_judged_on_wording(pack):
    """The heuristic is one-directional: without `expected` there is nothing to compare.

    A linter guessing here would be guessing about prose, which is why the obligation is
    documented for pack authors and this check is only ever a warning.
    """
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            polarity="fraud_indicator",
            label="Payment method was not changed on this record",
        ),
    )
    assert "label-polarity-unaffirmed" not in codes(pv.validate_pack(pack))


def test_an_imported_indicator_label_says_it_was_inherited(pack):
    """The hint differs by mechanism, because the fix does.

    Inheriting a library label across a polarity is fixed by overriding `label` in the
    importer; a hand-written one is fixed by rewording it. A single generic message would
    send half the readers to the wrong file.
    """
    path = pack / "shared" / "checks" / "example_checks.yaml"
    library = yaml.safe_load(path.read_text())
    library["borrowed"] = {
        "kind": "field_flag",
        "source": "record",
        "label": "The account holder confirmed the change",
        "expected": "account holder confirmed the change",
    }
    path.write_text(yaml.safe_dump(library, sort_keys=False))

    def mutate(spec):
        spec["conditions"] = [
            {
                "use": "example_checks/borrowed",
                "polarity": "fraud_indicator",
                "report_group": "validation",
            }
        ]

    edit_rules(pack, mutate)
    hits = find(pv.validate_pack(pack), "label-polarity-unaffirmed")
    assert len(hits) == 1
    assert "inherited unchanged" in hits[0]["detail"]
    assert "override `label`" in hits[0]["hint"]


# ------------------------------------------------------------------ the silent-empty class


def test_a_dangling_alias_is_its_own_code(pack):
    """The blast radius is the FILE, not the line, so the message must not point at a typo.

    `source_catalog.yaml` is anchor-heavy, `_read_yaml` turns a `ComposerError` into `{}`
    for the whole document, and zero sources means every condition resolves `unknown`. The
    separate code exists because the reported line is often far from the edit that broke it.
    """
    (pack / "source_catalog.yaml").write_text(
        "sources:\n  - name: a\n    <<: *never_declared\n"
    )
    result = pv.validate_pack(pack)
    hit = find(result, "yaml-anchor-unresolved")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "anchor" in hit[0]["hint"]
    assert "yaml-parse-failed" not in codes(result)


def test_a_redirect_to_a_source_that_does_not_exist_is_reported(pack):
    """`ask_instead` reaches the planner verbatim, so a dead name steers it nowhere.

    And it fails in the quietest way available: the planner is handed a catalog that does not
    contain the name it was told to ask, so it silently falls back to its own judgement — the
    behaviour of a source that declared no redirect at all. Caught by hand while writing one:
    the value named a plausible source that the catalog did not declare.
    """
    (pack / "source_catalog.yaml").write_text(
        "sources:\n"
        "  - name: a\n"
        "    not_answered_by:\n"
        "      - question: what was transacted\n"
        "        ask_instead: no_such_source\n"
    )
    result = pv.validate_pack(pack)
    hit = find(result, "redirect-to-unknown-source")
    assert len(hit) == 1
    # A WARNING, not an error: the detection is a heuristic over prose (see below), and the
    # pack still works — it merely says something untrue to the planner.
    assert hit[0]["severity"] == "warning"
    assert "no_such_source" in hit[0]["message"] and "'a'" in hit[0]["message"]


def test_a_redirect_naming_a_declared_source_is_silent(pack):
    """The other half, or the test above would pass against a check that fires on everything."""
    (pack / "source_catalog.yaml").write_text(
        "sources:\n"
        "  - name: a\n"
        "    not_answered_by:\n"
        "      - question: what was transacted\n"
        "        ask_instead: b\n"
        "  - name: b\n"
    )
    # Note `b` is declared AFTER the redirect that names it: the check runs once the whole
    # catalog is indexed, or authoring order would decide whether a valid redirect is flagged.
    assert "redirect-to-unknown-source" not in codes(pv.validate_pack(pack))


def test_prose_saying_no_source_can_answer_is_not_read_as_a_source_name(pack):
    """The reason this is a heuristic and not a membership test.

    The honest answer is sometimes that nothing in the pack answers the question and a human
    has to go outside the estate. Measured over the installed packs, 10 of 52 `ask_instead`
    values are sentences of that kind — a membership test would flag every one of them, and a
    check that fires on correct authoring gets switched off.
    """
    (pack / "source_catalog.yaml").write_text(
        "sources:\n"
        "  - name: a\n"
        "    not_answered_by:\n"
        "      - question: what was transacted\n"
        "        ask_instead: >\n"
        "          no source in this pack holds it; a human must check outside the estate\n"
    )
    assert "redirect-to-unknown-source" not in codes(pv.validate_pack(pack))


def test_ordinary_bad_syntax_reports_a_line(pack):
    (pack / "reporting.yaml").write_text("phrases:\n  a: [unclosed\n")
    hit = find(pv.validate_pack(pack), "yaml-parse-failed")
    assert len(hit) == 1
    assert hit[0]["line"] > 0


def test_content_that_parses_to_nothing_is_an_error(pack):
    """Real uncommented content, and the loader still gets nothing.

    The realistic cause is a top-level key renamed to one the loader does not read — the
    silent-empty trap with no ambiguity about intent, so it blocks the save.
    """
    # An explicit null document: uncommented content, parses to None. A file whose top-level
    # key is merely misspelled would NOT do here — it parses to a real mapping, so it is
    # invisible to this check and is caught by `unread-pack-key` instead.
    (pack / "reporting.yaml").write_text("--- !!null\n# a note after it\n")
    assert yaml.safe_load((pack / "reporting.yaml").read_text()) is None
    hits = find(pv.validate_pack(pack), "yaml-empty-but-nonblank")
    assert [h["severity"] for h in hits] == ["error"]
    assert hits[0]["path"] == "reporting.yaml"


def test_a_file_holding_only_comments_is_a_warning_not_an_error(pack):
    """WHY THE SEVERITY SPLITS HERE.

    An all-comment file is genuinely ambiguous: it is equally a body commented out during
    an edit and never restored, and a stub somebody just created to fill in. As an error the
    editor could not save a new file until its first real key existed. Nothing is broken —
    the loader treats it as absent — so the operator is told and not stopped.
    """
    (pack / "entity_glossary.yaml").write_text("# entities:\n#   - type: subject\n")
    result = pv.validate_pack(pack)
    hits = find(result, "yaml-empty-but-nonblank")
    assert [h["severity"] for h in hits] == ["warning"]
    assert result["ok"] is True, "a stub must never block a save"


def test_a_genuinely_blank_file_is_not_flagged(pack):
    """An empty optional file is a pack that declined to declare something, not a defect.

    Blank means *nothing at all* — whitespace and document markers only. A file holding a
    comment is a different case and reports the warning above, because a comment is somebody
    having written something.
    """
    (pack / "notes.yaml").write_text("\n\n---\n")
    assert "yaml-empty-but-nonblank" not in codes(pv.validate_pack(pack))


def test_broken_frontmatter_is_an_error_but_absent_frontmatter_is_not(pack):
    """A concept's frontmatter is functional: it carries the id every reference resolves to.

    `_parse_frontmatter` degrades to `{}`, so a malformed block silently unnames the
    document and every reference to it becomes an orphan.
    """
    concepts = pack / "shared" / "concepts"
    (concepts / "broken.md").write_text("---\nconcept_id: [oops\n---\n\nBody.\n")
    (concepts / "plain.md").write_text("# Just a heading\n\nNo frontmatter at all.\n")
    hits = find(pv.validate_pack(pack), "frontmatter-parse-failed")
    assert [h["path"] for h in hits] == ["shared/concepts/broken.md"]


# ----------------------------------------------------------------- derived engine facts


def test_the_derived_kinds_match_the_evaluator():
    """Derived from `_eval_condition`, and checked against the real module.

    Two failure modes are pinned at once. A hand-kept list would report `stub` — the kind
    the template's own placeholder condition uses — as unknown the day somebody added it.
    A naive `kind ==` grep over all of `src/` returns 21, because seven are RAG *source*
    kinds from an unrelated vocabulary, so the region has to be bounded to this one
    function. The import happens here, in a test, where the flat-import path is set up;
    `src/` reads the file instead.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        import correlation
    finally:
        sys.path.pop(0)
    import inspect
    import re

    body = inspect.getsource(correlation._eval_condition)
    expected = set(re.findall(r'kind == "([a-z_]+)"', body))

    derived = pv.condition_kinds()
    assert derived == expected
    assert len(derived) == 20
    assert "stub" in derived
    assert (
        "databricks_table" not in derived
    ), "that is a SOURCE kind, not a condition kind"


def test_only_three_kinds_honour_expected_label():
    """`expected_label` is read at exactly three sites, and `distinct_count` is not one.

    It is the one that bites, because it looks like it should: it renders `expected` as its
    bound (`<= 1`), so an author reasonably expects to be able to reword the PASS side.
    """
    assert pv.expected_label_kinds() == {
        "element_absence",
        "record_absence",
        "cohort_membership",
    }
    assert pv.expected_label_kinds() < pv.condition_kinds()


def test_expected_label_is_flagged_on_distinct_count_and_not_on_the_three(pack):
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec, kind="distinct_count", expected_label="never printed"
        ),
    )
    hit = find(pv.validate_pack(pack), "expected-label-noop")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"
    assert "pass_detail" in hit[0]["hint"]

    edit_rules(
        pack,
        lambda spec: one_condition(
            spec, kind="record_absence", expected_label="printed on PASS"
        ),
    )
    assert "expected-label-noop" not in codes(pv.validate_pack(pack))


def test_blocking_the_pass_with_nothing_to_drop_is_flagged(pack):
    """`inconclusive_blocks_pass` without `inconclusive_patterns` withholds nothing.

    The key says "if a value was dropped as unreadable, do not clear on the survivors" — so with
    no drop list there is no value to drop and the declaration is inert. That degrades to the
    reading the author was rejecting, which is the same shape as `expected-label-noop` beside it:
    a declaration whose whole content is a JUDGEMENT, silently discarded, and reachable only at
    authoring time because both readings produce a well-formed verdict.
    """
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="value_matches_pattern",
            fields=["code"],
            match_mode="forbidden",
            patterns=["^ENT$"],
            inconclusive_blocks_pass=True,
        ),
    )
    hit = find(pv.validate_pack(pack), "inconclusive-blocks-pass-noop")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"
    assert "inconclusive_patterns" in hit[0]["message"]
    # The hint offers both ways out, because dropping the key is a legitimate answer: the
    # default reading is right wherever the dropped value is somebody else's.
    assert "drop the key" in hit[0]["hint"]


def test_a_drop_list_beside_the_key_says_nothing(pack):
    """The positive control — the declaration as it is meant to be written.

    Without this, a check that fired on every `value_matches_pattern` condition, or on every
    condition at all, would satisfy the test above and warn the shipped packs into noise.
    """
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="value_matches_pattern",
            fields=["code"],
            match_mode="forbidden",
            patterns=["^ENT$"],
            inconclusive_patterns=["^ZZZ$"],
            inconclusive_blocks_pass=True,
        ),
    )
    assert "inconclusive-blocks-pass-noop" not in codes(pv.validate_pack(pack))

    # And the drop list ALONE is silent too — that is the default reading, declared by every
    # condition that had one before this key existed.
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="value_matches_pattern",
            fields=["code"],
            match_mode="forbidden",
            patterns=["^ENT$"],
            inconclusive_patterns=["^ZZZ$"],
        ),
    )
    assert "inconclusive-blocks-pass-noop" not in codes(pv.validate_pack(pack))


def test_the_derivation_behind_the_noop_checks_is_the_evaluator_and_not_a_list(pack):
    """`kinds_reading(key)` reads the evaluator's own source, so a key cannot go stale here.

    Two no-op diagnostics now rest on it, and both would fire on every condition — or on none —
    if the derivation were a hand-written set that drifted from the code. The property asserted
    is the one that makes it safe: a key mentioned in exactly one `kind ==` block is attributed
    to exactly that kind, and a key the preamble mentions is attributed to no kind at all, which
    turns the check OFF rather than firing everywhere.
    """
    assert pv.kinds_reading("ordinary_patterns") == {"value_matches_pattern"}
    assert pv.kinds_reading("unclassified_detail") == {"value_matches_pattern"}
    assert pv.kinds_reading("inconclusive_blocks_pass") == {"value_matches_pattern"}
    assert pv.kinds_reading("ordinary_patterns") < pv.condition_kinds()
    # `expected_label_kinds` is now this same derivation, so the two cannot disagree.
    assert pv.expected_label_kinds() == pv.kinds_reading("expected_label")
    # A key no evaluator mentions yields the empty set — silence, not a warning on every kind.
    assert pv.kinds_reading("no_such_key_anywhere") == set()


def test_closing_the_vocabulary_on_a_kind_that_cannot_read_it_is_flagged(pack):
    """`ordinary_patterns` on any other kind leaves the vocabulary OPEN, silently.

    The key's whole content is a judgement — these values decide nothing, so anything outside
    every list is a code the pack has not classified and must be withheld rather than counted as
    an ordinary non-match. Declared where no evaluator reads it, the check keeps returning
    exactly the answer the author declared this key to stop it returning, and both readings
    produce a well-formed verdict. Same shape as `expected-label-noop` above.
    """
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="distinct_count",
            fields=["code"],
            ordinary_patterns=["^ORD$"],
        ),
    )
    hit = find(pv.validate_pack(pack), "ordinary-patterns-noop")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"
    assert "value_matches_pattern" in hit[0]["detail"]
    assert "ordinary non-match" in hit[0]["message"]

    # The positive control: on the kind that reads it, nothing is said. Without this, a check
    # firing on every condition would satisfy the assertion above and warn the packs into noise.
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="value_matches_pattern",
            fields=["code"],
            match_mode="forbidden",
            patterns=["^ENT$"],
            ordinary_patterns=["^ORD$"],
        ),
    )
    assert "ordinary-patterns-noop" not in codes(pv.validate_pack(pack))


def test_one_pattern_in_two_classifications_is_flagged(pack):
    """The pack classifying the same value two ways, where precedence is invisible.

    Which list wins is a property of the evaluator's filter order — the enumerated drop runs
    first, then `patterns` beats `ordinary_patterns` — and nothing the author wrote says so. The
    report is a clear, an `unknown` or a finding depending on that order, so the contradiction has
    to be caught at authoring time. Reported on the STRING and not on regex equivalence: two
    patterns can overlap for values neither author foresaw and no validator can decide that,
    while an identical string is one decision made twice in opposite directions.
    """
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="value_matches_pattern",
            fields=["code"],
            match_mode="forbidden",
            patterns=["^ENT$", "^ZZZ$"],
            inconclusive_patterns=["^ZZZ$"],
        ),
    )
    hit = find(pv.validate_pack(pack), "pattern-in-two-classifications")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"
    assert "^ZZZ$" in hit[0]["message"]
    assert "patterns" in hit[0]["message"] and "inconclusive_patterns" in hit[0]["message"]

    # Three disjoint lists say nothing — the shape the fix exists to allow.
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="value_matches_pattern",
            fields=["code"],
            match_mode="forbidden",
            patterns=["^ENT$"],
            inconclusive_patterns=["^ZZZ$"],
            ordinary_patterns=["^ORD$"],
        ),
    )
    assert "pattern-in-two-classifications" not in codes(pv.validate_pack(pack))

    # And a pattern repeated WITHIN one list is not a contradiction: it is redundant, which
    # changes no answer. A check that reported it would fire on a merge artefact.
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="value_matches_pattern",
            fields=["code"],
            match_mode="forbidden",
            patterns=["^ENT$", "^ENT$"],
        ),
    )
    assert "pattern-in-two-classifications" not in codes(pv.validate_pack(pack))


def test_an_undispatched_kind_is_an_error_naming_the_alternatives(pack):
    """A kind no evaluator dispatches falls through to `unknown` — never evaluated.

    Which in a report is indistinguishable from a check whose source returned no rows, so
    the message says that rather than leaving the reader to infer it.
    """
    edit_rules(pack, lambda spec: one_condition(spec, kind="feild_flag"))
    hit = find(pv.validate_pack(pack), "unknown-condition-kind")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "field_flag" in hit[0]["detail"]
    assert hit[0]["line"] > 0


# ------------------------------------------- two declarations that fail by staying quiet
#
# `pair_by` and `subject_scope: element` both narrow what a condition is asked. Getting
# either wrong leaves the wider reading in place; nothing differs at evaluation time and
# nothing differs in the report. Both vocabularies are derived from the engine and each
# derivation has an empty-set failure mode that turns its check off silently.


def test_the_pair_by_vocabulary_is_derived_from_the_evaluator():
    """Kinds and values both, and an empty set for either disables the check.

    `pair_by` is honoured inside one gate in `_eval_condition`, so the kinds are read off that
    gate rather than listed: a kind added to it later must not need an edit here, and a kind
    the gate never names must not be silently accepted. Only a two-sided kind has records to
    pair its sides WITHIN, so the set is a strict subset of the fourteen.
    """
    kinds, values = pv.pair_by_kinds(), pv.pair_by_values()
    assert kinds, "an empty set turns the pair-by-noop check off entirely"
    assert kinds < pv.condition_kinds()
    assert "field_flag" not in kinds, "a one-sided kind has no sides to pair"
    assert values, "an empty set turns the value check off entirely"
    assert values == {"record"}


def test_the_subject_scope_vocabulary_is_derived_from_the_rollup_not_the_evaluator():
    """A second region, because the two keys are read by two different functions.

    A condition's `kind` is dispatched by `_eval_condition`; how its rows are SCOPED to the
    subject is decided by the rollup in `evaluate_verdict`. Scanning the evaluator for these
    values derives the empty set, and the empty set is how the check turns itself off — so the
    region has to be located, and non-emptiness is the assertion that proves it was.
    """
    assert "def evaluate_verdict(" in pv._verdict_source()
    values = pv.subject_scope_values()
    assert values, "an empty set turns every subject-scope check off"
    assert "element" in values
    assert "false" not in values, "`false` is the boolean, and needs no deriving"


def test_pair_by_on_a_one_sided_kind_is_an_error_naming_the_kinds_that_honour_it(pack):
    edit_rules(pack, lambda spec: one_condition(spec, kind="field_flag", pair_by="record"))
    hit = find(pv.validate_pack(pack), "pair-by-noop")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "field_flag" in hit[0]["message"]
    for kind in pv.pair_by_kinds():
        assert kind in hit[0]["detail"]
    assert hit[0]["line"] > 0


def test_a_pair_by_value_the_evaluator_never_compares_is_an_error(pack):
    """`pair_by: row` reads as though it said something. The evaluator compares one string."""
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="field_equality",
            pair_by="row",
            left={"source": "record", "field": "a"},
            right={"source": "record", "field": "b"},
        ),
    )
    hit = find(pv.validate_pack(pack), "pair-by-noop")
    assert len(hit) == 1
    assert "record" in hit[0]["detail"]


def test_sides_that_cannot_share_a_record_are_refused_before_the_run_not_during_it(pack):
    """Three shapes, one code — and the third is the one a real ruleset ships.

    The evaluator refuses to pair sides that deliberately select DIFFERENT records (an
    interval whose two ends are two versions of one entity, so its end side carries a
    `where:`) and appends the refusal to the check's detail. That refusal is correct at
    evaluation time and useless to the author, who has already run the query.
    """
    for over, phrase in (
        ({"right": {"source": "other", "field": "b"}}, "different sources"),
        ({"right": {"source": "record", "field": "b", "records": "x.y"}}, "`records:`"),
        (
            {
                "right": {
                    "source": "record",
                    "field": "b",
                    "where": [{"field": "state", "equals": "latest"}],
                }
            },
            "`where:`",
        ),
    ):
        edit_rules(
            pack,
            lambda spec, over=over: one_condition(
                spec,
                kind="field_equality",
                pair_by="record",
                left={"source": "record", "field": "a"},
                **over,
            ),
        )
        hit = find(pv.validate_pack(pack), "pair-by-unpairable")
        assert len(hit) == 1, over
        assert hit[0]["severity"] == "error"
        assert phrase in hit[0]["message"], over

    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="field_equality",
            pair_by="record",
            left={"source": "record", "field": "a", "records": "x.y"},
            right={"source": "record", "field": "b", "records": "x.y"},
        ),
    )
    assert "pair-by-unpairable" not in codes(pv.validate_pack(pack))


def test_a_subject_scope_string_the_engine_never_acts_on_is_an_error(pack):
    """And the BOOLEAN stays quiet, because `false` is the one value that always applies.

    The two live in one key on purpose — `false` widens to every row, the strings narrow — so
    the check has to tell a boolean from a misspelt string rather than validating the key.
    """
    edit_rules(pack, lambda spec: one_condition(spec, subject_scope="elements"))
    hit = find(pv.validate_pack(pack), "subject-scope-noop")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "element" in hit[0]["detail"]
    assert "false" in hit[0]["detail"]

    edit_rules(pack, lambda spec: one_condition(spec, subject_scope=False))
    assert "subject-scope-noop" not in codes(pv.validate_pack(pack))


def test_an_element_scope_with_no_discovery_to_narrow_by_names_what_is_missing(pack):
    """The narrowing REUSES the discovery walk, so it needs all three of its declarations.

    Without them there is no way to tell one identity's entries from another's, and the
    condition silently keeps reading every entry on the record — including the entries that
    belong to the other identities the same record names.
    """
    edit_rules(pack, lambda spec: one_condition(spec, subject_scope="element"))
    hit = find(pv.validate_pack(pack), "element-scope-undeclared")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    for missing in ("path", "subject", "source"):
        assert missing in hit[0]["message"]

    def _partial(spec):
        one_condition(spec, subject_scope="element")
        spec["subject_discovery"] = {"source": "record", "path": "a.b"}

    edit_rules(pack, _partial)
    hit = find(pv.validate_pack(pack), "element-scope-undeclared")
    assert len(hit) == 1
    assert "subject" in hit[0]["message"]
    assert "path" not in hit[0]["message"]


def test_an_element_scope_on_a_source_carrying_no_elements_is_an_error(pack):
    """A condition on another source has no per-identity element to narrow to — at all.

    This is the discrimination the two halves needed: one live condition's sides sit under a
    single repeated array on one record and needs the element narrowing; another's sit on a
    different source entirely and can only be paired per record. Declaring the first key on
    the second condition would leave it exactly as pooled as before.
    """

    def _elsewhere(spec):
        one_condition(spec, subject_scope="element", source="other")
        spec["sources"]["other"] = spec["sources"]["record"]
        spec["subject_discovery"] = {
            "source": "record",
            "path": "a.b",
            "subject": ["who"],
        }

    edit_rules(pack, _elsewhere)
    hit = find(pv.validate_pack(pack), "element-scope-unreachable")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "other" in hit[0]["message"]
    assert "record" in hit[0]["message"]
    assert "pair_by" in hit[0]["hint"]

    def _reachable(spec):
        one_condition(spec, subject_scope="element", source="record")
        spec["subject_discovery"] = {
            "source": "record",
            "path": "a.b",
            "subject": ["who"],
        }

    edit_rules(pack, _reachable)
    assert "element-scope-unreachable" not in codes(pv.validate_pack(pack))


# ------------------------------------------ declarations that fabricate a clear answer
#
# A cohort check needs both halves of its declaration: the widening that admits other rows,
# and the selector that identifies the subject's own. Missing the selector compares nothing
# and reports a clear. Missing the widening removes other rows before the comparison. Either
# way the fabricated answer is a clear.


def test_the_cohort_vocabulary_is_derived_from_the_evaluator_by_its_KEY(pack):
    """Named after `subject_rows` and not after the kind, because the kind may be renamed.

    An empty set turns both checks below off — the standing failure mode of every vocabulary
    derived from the engine — so non-emptiness is asserted first and the contents second.
    """
    kinds = pv.subject_rows_kinds()
    assert kinds, "an empty set turns both cohort checks off entirely"
    assert kinds < pv.condition_kinds()
    assert "field_flag" not in kinds


def test_a_cohort_check_with_no_subject_selector_is_an_error(pack):
    """Three spellings of "no selector", one code — and a declared one stays quiet.

    The absent key is the shape that reaches production: nothing about it looks wrong in the
    YAML, and the check answers `pass` with the pack's own wording.
    """
    kind = sorted(pv.subject_rows_kinds())[0]
    for sel in (None, {"where": []}, {"where": [{"any_of": ["X"]}]}):
        over = {"subject_rows": sel} if sel is not None else {}
        edit_rules(
            pack,
            lambda spec, over=over: one_condition(
                spec, kind=kind, subject_scope=False, **over
            ),
        )
        hit = find(pv.validate_pack(pack), "cohort-subject-rows-missing")
        assert len(hit) == 1, sel
        assert hit[0]["severity"] == "error", sel
        assert kind in hit[0]["message"], sel
        assert hit[0]["line"] > 0, sel

    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind=kind,
            subject_scope=False,
            subject_rows={"where": [{"field": "record", "from_entity": "record"}]},
        ),
    )
    assert "cohort-subject-rows-missing" not in codes(pv.validate_pack(pack))


def test_a_cohort_check_without_the_widening_is_an_error(pack):
    """The other half, and the reason the pair is checked together.

    A live import declared NEITHER, and the safe `unknown` it returned came from an unrelated
    accident: the default narrowing had already emptied the source, so the reading nobody wanted
    never got to happen. Fixing only the widening would have converted that into a false PASS.
    """
    kind = sorted(pv.subject_rows_kinds())[0]
    both = {
        "kind": kind,
        "subject_rows": {"where": [{"field": "record", "from_entity": "record"}]},
    }
    edit_rules(pack, lambda spec: one_condition(spec, **both))
    hit = find(pv.validate_pack(pack), "cohort-scope-narrowed")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"

    # The string values narrow too, so only the boolean clears it.
    edit_rules(pack, lambda spec: one_condition(spec, subject_scope="element", **both))
    assert "cohort-scope-narrowed" in codes(pv.validate_pack(pack))

    edit_rules(pack, lambda spec: one_condition(spec, subject_scope=False, **both))
    assert "cohort-scope-narrowed" not in codes(pv.validate_pack(pack))


def test_a_row_match_without_the_widening_is_an_error(pack):
    """The same coupling one key over: `row_match` is only ever applied to a widened condition.

    Left inert, the check reads every row the retriever's OR-ed entity filter returned — other
    units' and other identities' rows — as this incident's evidence, which is how a presence
    test once found an identity in a list it was absent from.
    """
    clauses = [{"from_entity": "user", "fields": ["sign"]}]
    assert pv._row_match_needs_widening(), "an unmatched probe turns this check off"

    edit_rules(pack, lambda spec: one_condition(spec, row_match=clauses))
    hit = find(pv.validate_pack(pack), "row-match-inert")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "subject_scope: false" in hit[0]["message"]

    edit_rules(
        pack, lambda spec: one_condition(spec, row_match=clauses, subject_scope=False)
    )
    assert "row-match-inert" not in codes(pv.validate_pack(pack))


# --------------------------------------------------------------------- inert declarations


def test_an_unknown_entity_key_is_reported_because_pydantic_drops_it(pack):
    """`EntityDef` has the default `extra='ignore'`: the pack loads clean, the key does nothing.

    Zero of these exist across the installed packs, so this is a pure regression guard —
    the check that makes the next invented key visible on the save that introduces it.
    """
    path = pack / "entity_glossary.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["entities"][0]["descriptoin"] = "typo"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    hit = find(pv.validate_pack(pack), "unknown-model-key")
    assert len(hit) == 1
    assert "descriptoin" in hit[0]["message"]
    assert "description" in hit[0]["detail"]


def test_an_unknown_source_key_is_reported(pack):
    path = pack / "source_catalog.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["sources"][0]["retreival_class"] = "primary"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    hit = find(pv.validate_pack(pack), "unknown-model-key")
    assert len(hit) == 1
    assert "retrieval_class" in hit[0]["detail"]


def test_a_value_form_stem_that_can_name_nothing_is_an_ERROR(pack):
    """An ERROR, not a warning: mechanically decidable and provably inert.

    `value_stem` needs exactly one capture group — the part a source may store INSTEAD of the
    whole value. Without a group, or with an unparseable pattern, it returns `None` for every
    value, so the widening guard is never handed a stem and the pack reads as one that never
    opted in. What is then published is the predicate the key exists to repair: the long form
    alone against a column storing the core, matching no row and reporting 0 rows as a success.
    """
    path = pack / "entity_glossary.yaml"
    doc = yaml.safe_load(path.read_text())
    ent = doc["entities"][0]

    def _with(stem):
        ent["value_forms"] = [{"name": "coded", "pattern": "^[A-Z]+$", "stem": stem}]
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        return find(pv.validate_pack(pack), "unusable-value-form-stem")

    for stem, expect in (("^([A-Z]{2})", 0), ("^[A-Z]{2}", 1), ("^([A-Z])([A-Z])", 1),
                         ("^([A-Z]", 1)):
        hits = _with(stem)
        assert len(hits) == expect, (stem, hits)
        for hit in hits:
            assert hit["severity"] == "error"
            assert hit["line"] > 0
            assert "coded" in hit["message"]
    # A form declaring no stem is every form in every pack that has not opted in: silent.
    ent["value_forms"] = [{"name": "coded", "pattern": "^[A-Z]+$"}]
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    assert find(pv.validate_pack(pack), "unusable-value-form-stem") == []


def test_no_installed_pack_declares_an_unusable_stem():
    for name in installed_packs():
        hits = find(pv.validate_pack(PACKS / name), "unusable-value-form-stem")
        assert hits == [], [(h["path"], h["message"]) for h in hits]


def test_no_installed_source_or_entity_carries_an_unknown_key():
    """The other half of the regression guard: measured zero today, asserted to stay zero."""
    for name in installed_packs():
        root = PACKS / name
        hits = find(pv.validate_pack(root), "unknown-model-key")
        assert hits == [], [(h["path"], h["message"]) for h in hits]


# ------------------------------------------------- an actor key that resolves to nothing
#
# A key that cannot resolve leaves the query scoped to the generator's default OR, so
# other parties' rows come back. One of the three run-time causes is decidable from the
# catalog text alone (a declaration no incident can satisfy).
#
# Severities: a shape the resolver can never read is an error; a member the source
# declares no binding for is a warning, because `entity_bindings` is a prior and
# `map_entities` can still bind the type from the discovered schema.


def _catalog_source(pack, **overrides):
    """The template's one source, with `overrides` merged in. Returns the diagnostics."""
    path = pack / "source_catalog.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["sources"][0].update(overrides)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return pv.validate_pack(pack)


def test_a_well_formed_actor_key_is_accepted_and_says_nothing(pack):
    """Both arities: a pair, and a one-column key — neither is a defect.

    The one-column spelling is the case that matters here. It used to be inert in the engine,
    and the fix was to resolve it rather than to refuse it, so this file must not turn around
    and report the shape the engine now honours.
    """
    for keys in ([["example_entity", "example_container"]], [["example_entity"]]):
        result = _catalog_source(
            pack,
            identity_keys=keys,
            entity_bindings={
                "example_entity": ["some.leaf"],
                "example_container": ["some.other_leaf"],
            },
        )
        assert not [
            d for d in result["diagnostics"] if d["code"].startswith("identity-key")
        ], keys
        assert find(result, "actor-key-member-not-bound") == []


def test_an_identity_keys_that_is_not_a_list_of_candidates_is_an_ERROR(pack):
    """`identity_keys: <type>` reads as no candidate at all — the source is unkeyed."""
    result = _catalog_source(pack, identity_keys="example_entity")
    hits = find(result, "identity-keys-not-a-list")
    assert len(hits) == 1
    assert hits[0]["severity"] == "error"
    assert hits[0]["line"] > 0
    assert "str" in hits[0]["message"]


@pytest.mark.parametrize("candidate", ["example_entity", 7, {"entity": "x"}])
def test_a_candidate_that_is_not_a_list_is_an_ERROR(pack, candidate):
    """A bare string is the near miss: `- example_entity` instead of `- [example_entity]`.

    The resolver reads a candidate by iterating it, so a string would yield its CHARACTERS —
    types that map to nothing, which reads downstream as "this incident carried none of the
    key's members". The incident gets blamed for a catalog defect, so the catalog is where it
    has to be reported.
    """
    result = _catalog_source(pack, identity_keys=[candidate])
    hits = find(result, "identity-key-not-a-list")
    assert len(hits) == 1
    assert hits[0]["severity"] == "error"
    assert "candidate 1" in hits[0]["message"]


def test_a_candidate_member_that_cannot_NAME_a_type_is_an_ERROR(pack):
    """One bracket too many is a TypeError inside a guard, so the engine drops the member.

    Which leaves the candidate resolving a key the pack did not declare (or none) — an error
    because the declaration cannot be honoured as written, whatever the incident carries.
    """
    result = _catalog_source(
        pack, identity_keys=[[["example_entity", "example_container"]], ["", None]]
    )
    hits = find(result, "identity-key-member-not-a-name")
    assert [h["severity"] for h in hits] == ["error"] * len(hits)
    assert len(hits) == 3, [h["message"] for h in hits]
    assert "candidate 1" in hits[0]["message"]
    assert "candidate 2" in hits[1]["message"]


def test_a_candidate_repeating_ONE_type_is_a_warning_and_not_an_error(pack):
    """It still resolves — as the one column it names, which is not what two names ask for.

    Not an error, because the source is keyed (weakly) and the pack may have meant exactly
    that; a warning, because two spellings of one type look like a conjunction and no
    conjunction can be written from one column.
    """
    result = _catalog_source(
        pack,
        identity_keys=[["example_entity", "example_entity"]],
        entity_bindings={"example_entity": ["some.leaf"]},
    )
    hits = find(result, "identity-key-repeats-a-type")
    assert len(hits) == 1
    assert hits[0]["severity"] == "warning"
    assert not codes(result, "error")


def test_a_key_member_the_source_does_not_BIND_is_a_warning(pack):
    """The candidate is skipped on every incident, and nothing says so at run time.

    A warning and not an error for one reason: `entity_bindings` is a declaration, not the
    only route — the mapper binds off the DISCOVERED schema too, so a member named nowhere
    here can still resolve live. Both declarations are checked, because both resolve members
    through the same field map.
    """
    result = _catalog_source(
        pack,
        identity_keys=[["example_entity", "example_container"]],
        require_all_entities=["example_entity", "example_other"],
        entity_bindings={"example_entity": ["some.leaf"]},
    )
    hits = find(result, "actor-key-member-not-bound")
    assert len(hits) == 2, [h["message"] for h in hits]
    assert {h["severity"] for h in hits} == {"warning"}
    assert "example_container" in hits[0]["message"]
    assert "example_other" in hits[1]["message"]
    assert not codes(result, "error")


def test_a_source_binding_NOTHING_is_not_reported_for_every_member(pack):
    """A source declaring no bindings relies on discovery for every type.

    The check would then fire on all of them while knowing nothing — the false-positive
    shape this file's docstring is about.
    """
    result = _catalog_source(
        pack,
        identity_keys=[["example_entity", "example_container"]],
        entity_bindings={},
    )
    assert find(result, "actor-key-member-not-bound") == []


def test_a_require_all_entities_of_ONE_type_ANDs_nothing_and_is_a_warning(pack):
    """The key promises a conjunction by its name and cannot write one from one column.

    It is not inert — it resolves one field and licenses reading an empty result — so this is
    a warning that names the weaker meaning, and the hint offers the spelling that says it.
    """
    result = _catalog_source(
        pack,
        require_all_entities=["example_entity"],
        entity_bindings={"example_entity": ["some.leaf"]},
    )
    hits = find(result, "require-all-entities-not-a-conjunction")
    assert len(hits) == 1
    assert hits[0]["severity"] == "warning"
    assert "one-column key" in hits[0]["message"]
    assert "identity_keys" in hits[0]["hint"]
    assert not codes(result, "error")
    # Two types is the shape the key is for: silent.
    result = _catalog_source(
        pack,
        require_all_entities=["example_entity", "example_container"],
        entity_bindings={
            "example_entity": ["some.leaf"],
            "example_container": ["some.other_leaf"],
        },
    )
    assert find(result, "require-all-entities-not-a-conjunction") == []


def test_no_installed_pack_declares_an_unresolvable_actor_key():
    """Measured zero across every installed pack and the template, asserted to stay zero."""
    for root in [PACKS / name for name in installed_packs()] + [TEMPLATE]:
        hits = [
            d
            for d in pv.validate_pack(root)["diagnostics"]
            if d["code"].startswith("identity-key")
            or d["code"]
            in {"actor-key-member-not-bound", "require-all-entities-not-a-conjunction"}
        ]
        assert hits == [], [(root.name, h["code"], h["message"]) for h in hits]


# ------------------------------------------------------------------------- sources


def test_a_deleted_catalog_and_an_empty_one_report_the_same_code(pack):
    """One code, because the runtime consequence is one thing: zero sources.

    Both spellings load as a pack in which every condition resolves to `unknown` and the
    report reads INSUFFICIENT DATA. A caller groups on the code, so splitting them would
    make the *same* finding look like two, while an early return on absence — which is what
    this originally did — made a DELETED catalog the one edit producing no diagnostic at
    all. Now that the editor can delete a file in one request, that gap is reachable.
    """
    empty = find(
        pv.validate_pack(pack), "no-sources-declared"
    )  # the template declares sources, so start from an empty list
    assert empty == []
    (pack / "source_catalog.yaml").write_text("sources: []\n")
    on_empty = find(pv.validate_pack(pack), "no-sources-declared")
    assert len(on_empty) == 1
    assert on_empty[0]["severity"] == "warning"

    (pack / "source_catalog.yaml").unlink()
    on_missing = find(pv.validate_pack(pack), "no-sources-declared")
    assert len(on_missing) == 1
    assert on_missing[0]["code"] == on_empty[0]["code"]
    assert (
        "does not exist" in on_missing[0]["message"]
    ), "the wording must still say which of the two it is — only the fix differs"


def test_a_deleted_glossary_is_reported_too(pack):
    (pack / "entity_glossary.yaml").unlink()
    hit = find(pv.validate_pack(pack), "no-entities-declared")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"
    assert pv.validate_pack(pack)["counts"]["entities"] == 0


def test_a_logical_source_pointing_at_no_catalog_entry_is_an_error(pack):
    edit_rules(pack, lambda spec: spec["sources"].update({"record": "not_in_catalog"}))
    hit = find(pv.validate_pack(pack), "unknown-physical-source")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


def test_reading_an_undeclared_source_is_an_error(pack):
    """A condition reading a name the ruleset never declares can only ever be `unknown`.

    The scan runs over IMPORT-RESOLVED conditions, which is load-bearing — see the next
    test for what a raw-YAML scan gets wrong.
    """
    edit_rules(pack, lambda spec: one_condition(spec, source="ghost"))
    hit = find(pv.validate_pack(pack), "unknown-ruleset-source")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "ghost" in hit[0]["message"]


def test_a_source_named_only_by_an_imported_check_is_not_an_orphan(pack):
    """WHY THE SCAN RESOLVES `use:` FIRST.

    A shared check owns the MECHANICS, which includes the logical source name, so a
    condition that imports one may never mention `source:` itself. Scanning the raw YAML
    reported four of one installed pack's sources as unused and would have reported this
    ruleset's only source as both unused and undeclared — two errors and a warning, all
    three false.
    """
    path = pack / "shared" / "checks" / "example_checks.yaml"
    library = yaml.safe_load(path.read_text())
    library["reads_record"] = {
        "kind": "field_flag",
        "source": "record",
        "field": "record.flag",
        "label": "Flag is set",
    }
    path.write_text(yaml.safe_dump(library, sort_keys=False))

    def mutate(spec):
        spec["conditions"] = [
            {"use": "example_checks/reads_record", "report_group": "validation"}
        ]

    edit_rules(pack, mutate)
    result = pv.validate_pack(pack)
    assert "unknown-ruleset-source" not in codes(result)
    assert "orphan-logical-source" not in codes(result)


def test_a_declared_but_unread_source_is_only_a_warning():
    """AND THE CHECKED-IN TEMPLATE IS WHY.

    The plan had this as an error. Measured, the template declares a `sources:` entry whose
    only `source:` references are inside commented-out examples — so an error would make the
    starting pack unsavable. The fact is still worth stating: a `sources:` map is a hard
    data dependency retrieved on every run, so an unread entry spends its full scan cost for
    nothing. It just does not stop a save.
    """
    hits = find(pv.validate_pack(TEMPLATE), "orphan-logical-source")
    assert len(hits) == 1
    assert hits[0]["severity"] == "warning"
    assert "selection_guidance" in hits[0]["hint"], (
        "the two retrieval mechanisms are not interchangeable; the hint has to name the "
        "other one"
    )


def test_both_source_directions_are_clean_on_the_installed_packs():
    for name in installed_packs():
        root = PACKS / name
        result = pv.validate_pack(root)
        assert "unknown-ruleset-source" not in codes(result)
        assert "orphan-logical-source" not in codes(result), name


# ------------------------------------------------------------------ as-of declarations
#
# Every defect here is silent: the entry looks complete, the version log is adjudicated
# whole, and a later change written by the responder counts as the subject's conduct.
# All three keys are required together; the engine abstains on a partial declaration,
# and abstaining is the safe but invisible failure.

_AS_OF_OK = {
    "timestamp_fields": ["written_at", "created_at"],
    "actor_fields": ["last_updator"],
    "actor_entities": ["user"],
}


def _as_of_on_first_source(**overrides):
    """Put an `as_of` entry on whatever the fixture pack's first logical source is."""

    def mutate(spec):
        logical = next(iter(spec.get("sources") or {"record": "x"}))
        spec["as_of"] = {logical: {**_AS_OF_OK, **overrides}}

    return mutate


def test_an_as_of_entry_naming_no_declared_source_is_an_error(pack):
    edit_rules(pack, lambda s: s.update({"as_of": {"nosuchsource": dict(_AS_OF_OK)}}))
    hit = find(pv.validate_pack(pack), "as-of-unknown-source")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "sources" in hit[0]["hint"]


def test_an_as_of_entry_with_no_timestamp_field_is_an_error(pack):
    """It declares nothing the engine can order versions by, so it does nothing at all."""
    edit_rules(pack, _as_of_on_first_source(timestamp_fields=[]))
    hit = find(pv.validate_pack(pack), "as-of-no-timestamp")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


def test_an_as_of_entry_with_no_actor_field_is_an_error(pack):
    """The omission that looks most armed: a source and a timestamp are declared, so the YAML
    reads as complete — but with no writer column the engine cannot tell the responder's later
    changes from the subject's own, and cutting on time alone would discard the subject's
    later conduct including anything exculpatory."""
    edit_rules(pack, _as_of_on_first_source(actor_fields=[]))
    hit = find(pv.validate_pack(pack), "as-of-no-actor-field")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "last-updator" in hit[0]["hint"]


def test_an_as_of_entry_with_no_actor_entity_is_an_error(pack):
    """A writer column with nothing from the incident to compare it against."""
    edit_rules(pack, _as_of_on_first_source(actor_entities=[]))
    hit = find(pv.validate_pack(pack), "as-of-no-actor-entity")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


def test_a_well_formed_as_of_declaration_is_accepted(pack):
    edit_rules(pack, _as_of_on_first_source())
    assert not [c for c in codes(pv.validate_pack(pack)) if c.startswith("as-of")]


# --------------------------------------------------- follow-up passes: pass 2 and its scope
#
# Two families of defect, silent in opposite ways.
#
# A `default_ruleset` that matches nothing falls back to directory order, so the pack
# reads as having taken the decision while it is still alphabetical.
#
# A harvest's `where` clauses scope the follow-up question. `apply_where` skips an
# incomplete clause deliberately; the cost is that the harvest reads every row, and the pass
# produces a set of ordinary neighbours indistinguishable from a real finding.


def _follow_up(where=None, **entry):
    """Put one follow-up pass on the template ruleset, harvesting from its own source."""
    item = {"entity": "example_entity", "source": "record", "fields": ["some.leaf"]}
    if where is not None:
        item["where"] = where

    def mutate(spec):
        spec["follow_up_passes"] = [
            {"pass": 2, "source": "record", "harvest": [item], **entry}
        ]

    return mutate


def test_a_well_formed_follow_up_pass_and_row_selector_are_accepted(pack):
    """The positive control, and it is not decoration: every code below is reachable by a
    check that fires on correct authoring too, and a lint an author learns to scroll past
    reports nothing at all."""
    edit_rules(
        pack,
        _follow_up(where=[{"field": "some.status", "any_of": ["X"], "match": "exact"}]),
    )
    result = pv.validate_pack(pack)
    assert not [
        c
        for c in codes(result)
        if c.startswith("follow-up") or c.startswith("harvest-")
    ]


def _shared_pass(**entry):
    """One follow-up pass targeting TWO logical sources off one harvest.

    Both logical names point at the template's only catalog source, deliberately: what these
    checks read is the ruleset's `sources:` map, and a second physical source would add an
    unread-source warning that has nothing to do with the pass.
    """
    base = _follow_up(**entry)

    def mutate(spec):
        spec.setdefault("sources", {})["trail"] = "example_source"
        base(spec)

    return mutate


def test_a_follow_up_pass_targeting_two_declared_sources_is_accepted(pack):
    """The positive control for the list form, and the reason the list exists.

    A pass number is a scarce slot, so a procedure with more deferred questions than slots
    cannot ask one of them at all. Sharing one entry between targets that harvest identically
    is the fix, so the lint must be silent on it — a warning on correct authoring is what
    teaches an author to scroll past the whole list.
    """
    edit_rules(pack, _shared_pass(source=["record", "trail"]))
    result = pv.validate_pack(pack)
    assert not [
        c
        for c in codes(result)
        if c.startswith("follow-up") or c.startswith("harvest-")
    ]


def test_a_stale_target_in_a_shared_pass_is_reported_ON_ITS_OWN(pack):
    """One unresolvable name beside working ones is the case that reads as fine.

    The pass runs, the siblings are queried, and only that source's conditions go `unknown`
    while every stage reports success — so the target is named individually rather than the
    entry being reported as broken.
    """
    edit_rules(pack, _shared_pass(source=["record", "nosuchsource", "trail"]))
    hit = find(pv.validate_pack(pack), "follow-up-unknown-source")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "nosuchsource" in hit[0]["message"]
    # The siblings are not implicated, and the entry is not also reported as sourceless.
    assert "follow-up-no-source" not in codes(pv.validate_pack(pack))


def test_a_target_named_twice_in_one_pass_is_a_warning(pack):
    """Deduped at load, so it is queried once — a warning, because nothing is lost.

    It still reads as two questions to whoever counts the pass's targets, which is the only
    harm and the reason it is reported at all.
    """
    edit_rules(pack, _shared_pass(source=["record", "record"]))
    hit = find(pv.validate_pack(pack), "follow-up-duplicate-target")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"


def test_a_per_source_purpose_is_checked_against_the_pass_TARGETS(pack):
    """The purpose is the text the backend query is written from, so a mis-keyed one is inert.

    Two directions, and both leave the same artifact — a query described by the source's own
    catalog paragraph instead of by the question this pass asks, which on a reference table
    keyed differently from an event log is a query that can only answer empty. Neither is an
    error: the pass runs and the fallback text is a real one.
    """
    edit_rules(
        pack,
        _shared_pass(
            source=["record", "trail"],
            purpose={"record": "what it did", "elsewhere": "unused"},
        ),
    )
    result = pv.validate_pack(pack)
    unknown = find(result, "follow-up-purpose-unknown-target")
    assert len(unknown) == 1 and unknown[0]["severity"] == "warning"
    assert "elsewhere" in unknown[0]["message"]
    missing = find(result, "follow-up-purpose-missing-target")
    assert len(missing) == 1 and missing[0]["severity"] == "warning"
    assert "trail" in missing[0]["message"]

    # A mapping covering every target exactly is silent — as is a plain string, which still
    # means "every target" and is every declaration written before a mapping was possible.
    for purpose in ({"record": "did", "trail": "kind"}, "one question for both"):
        edit_rules(pack, _shared_pass(source=["record", "trail"], purpose=purpose))
        assert not [
            c for c in codes(pv.validate_pack(pack)) if c.startswith("follow-up-purpose")
        ]


def test_a_harvest_where_that_is_not_a_list_is_an_error(pack):
    edit_rules(pack, _follow_up(where={"field": "some.status", "any_of": ["X"]}))
    hit = find(pv.validate_pack(pack), "harvest-where-not-a-list")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "EVERY row" in hit[0]["message"]


def test_a_harvest_where_item_that_is_not_a_mapping_is_an_error(pack):
    edit_rules(pack, _follow_up(where=["some.status"]))
    hit = find(pv.validate_pack(pack), "harvest-where-not-a-mapping")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


@pytest.mark.parametrize(
    "clause, missing",
    [
        ({"any_of": ["X"]}, "`field`"),
        ({"field": "some.status"}, "`any_of`"),
        ({"field": "some.status", "any_of": []}, "`any_of`"),
        ({}, "`field` and `any_of`"),
    ],
)
def test_an_incomplete_harvest_where_clause_is_an_error_naming_what_is_missing(
    pack, clause, missing
):
    """The clause the engine SKIPS, which is the one that reads as a scope that was applied.

    Both halves are parametrised because a clause with a field and no values looks far more
    complete than an empty one, and it is skipped just as silently.
    """
    edit_rules(pack, _follow_up(where=[clause]))
    hit = find(pv.validate_pack(pack), "harvest-where-incomplete")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert missing in hit[0]["message"]


def test_an_unrecognised_harvest_where_match_is_an_error(pack):
    """Anything but `exact` is read as `substring`, so a typo WIDENS the scope silently.

    On a short vocabulary that matches far more rows than the author asked for, which is the
    same defect as no selector at all but harder to see, because a selector is present.
    """
    edit_rules(
        pack,
        _follow_up(
            where=[{"field": "some.status", "any_of": ["X"], "match": "contains"}]
        ),
    )
    hit = find(pv.validate_pack(pack), "harvest-where-bad-match")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "substring" in hit[0]["hint"]


def test_the_recognised_matches_are_accepted(pack):
    """`substring` is a real declaration, not merely the fallback a typo lands on."""
    for match in ("exact", "substring", "EXACT"):
        edit_rules(
            pack,
            _follow_up(
                where=[{"field": "some.status", "any_of": ["X"], "match": match}]
            ),
        )
        assert "harvest-where-bad-match" not in codes(pv.validate_pack(pack))


# A discarded `together:` widens the follow-up query rather than emptying it: the
# components fall back to per-type lists AND-ed together, so the query asks every
# pairing the cross product can spell. Only authoring time can tell the two apart.


def _together(components, **item_extra):
    """One follow-up pass whose only harvest declares a co-occurrence and nothing else.

    Deliberately no item-level `entity`/`fields`: that is how a real combination is spelled
    (the components carry them), and it is also what makes a discarded `together` visible as
    a dropped pass rather than as a silently widened one.
    """

    def mutate(spec):
        spec["follow_up_passes"] = [
            {
                "pass": 2,
                "source": "record",
                "harvest": [{"source": "record", "together": components, **item_extra}],
            }
        ]

    return mutate


_PAIR = [
    {"entity": "example_entity", "fields": ["some.leaf"]},
    {"entity": "example_container", "fields": ["some.other_leaf"]},
]


def test_a_well_formed_together_harvest_is_accepted(pack):
    """The positive control, and it carries a second claim the loader had to be fixed for.

    An item declaring its entities per COMPONENT declares none of its own, so a check reading
    the item level sees an empty harvest — which is this section's own headline error telling
    an author the pass is dropped while it runs. `follow-up-nothing-harvested` must be silent
    here for the same reason the pass must not be dropped at load.
    """
    edit_rules(pack, _together(_PAIR))
    result = pv.validate_pack(pack)
    assert not [
        c
        for c in codes(result)
        if c.startswith("harvest-together") or c == "follow-up-nothing-harvested"
    ]


def test_an_ordinary_harvest_declares_no_combination_and_is_not_asked_to(pack):
    """Every pack written before combinations could be declared has no `together` at all.

    Absent is not empty: the checks return before any of them can fire, so the per-type
    reading stays exactly as silent as it was.
    """
    edit_rules(pack, _follow_up())
    assert not [
        c for c in codes(pv.validate_pack(pack)) if c.startswith("harvest-together")
    ]


@pytest.mark.parametrize("declared", ["some.leaf", [], {}, 3])
def test_a_together_that_is_not_a_non_empty_list_is_an_error(pack, declared):
    """The whole co-occurrence is discarded, so the query asks the cross product.

    `[]` is parametrised beside a string because an empty list reads as "no components yet"
    to an author and as "this item harvests nothing" to the engine — the item is then left
    with neither its own `entity`/`fields` nor a component's, so the pass is ALSO dropped,
    and both diagnostics are asserted together: the second says the pass does not run, only
    the first says which declaration stopped it.
    """
    edit_rules(pack, _together(declared))
    result = pv.validate_pack(pack)
    hit = find(result, "harvest-together-not-a-list")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "CROSS PRODUCT" in hit[0]["message"]
    assert "follow-up-nothing-harvested" in codes(result)


def test_a_together_component_that_is_not_a_mapping_is_an_error(pack):
    """The worst of the three, because the surviving component still works.

    A skipped component is enforced out of the combination rather than with it, so the pass
    runs, narrows on one column, and reads as a co-occurrence that was applied.
    """
    edit_rules(pack, _together([_PAIR[0], "some.other_leaf"]))
    result = pv.validate_pack(pack)
    hit = find(result, "harvest-together-not-a-mapping")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    # The pass is NOT dropped — one readable component is still a harvest, which is exactly
    # why the widening has no artifact of its own.
    assert "follow-up-nothing-harvested" not in codes(result)


@pytest.mark.parametrize(
    "component, missing",
    [
        ({"fields": ["some.other_leaf"]}, "`entity`"),
        ({"entity": "example_container"}, "`fields`"),
        ({"entity": "example_container", "fields": []}, "`fields`"),
        ({"entity": "", "fields": [""]}, "`entity` and `fields`"),
    ],
)
def test_an_incomplete_together_component_is_an_error_naming_what_is_missing(
    pack, component, missing
):
    """Same parametrisation as the `where` clause above, and the same reason.

    A component with an entity and no fields looks far more complete than an empty one and is
    dropped just as silently — and a component naming a type the engine never harvests values
    for is the half that cannot be inferred from the remaining components.
    """
    edit_rules(pack, _together([_PAIR[0], component]))
    hit = find(pv.validate_pack(pack), "harvest-together-incomplete-component")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert missing in hit[0]["message"]


def test_a_per_component_source_is_an_error_because_the_engine_ignores_it(pack):
    """Honouring it would invent the pairings the mechanism exists to remove.

    Co-occurrence is a fact about ONE row: two values read from two sources never occurred
    together anywhere, so a component-level `source` can only be read as a licence to pair
    everything with everything. The engine ignores it; the author has to be told, or the
    declaration reads as a two-source combination that was enforced.
    """
    edit_rules(
        pack,
        _together(
            [_PAIR[0], {**_PAIR[1], "source": "trail"}],
        ),
    )
    hit = find(pv.validate_pack(pack), "harvest-together-component-source")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "example_container" in hit[0]["message"]
    assert "ONE source" in hit[0]["hint"]


def test_two_components_of_the_same_entity_type_collapse_and_are_an_error(pack):
    """Both values route to the one column the target binds for that type.

    A combination over a single column is what the per-type list already asks, so the guard
    has nothing to narrow and the values go back to being a flat list — reported once per
    duplicated type rather than once per component, because the type is the fault.
    """
    edit_rules(
        pack,
        _together(
            [
                {"entity": "example_entity", "fields": ["some.leaf"]},
                {"entity": "example_entity", "fields": ["some.other_leaf"]},
            ]
        ),
    )
    hit = find(pv.validate_pack(pack), "harvest-together-duplicate-entity")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "example_entity" in hit[0]["message"]
    assert "`fields`" in hit[0]["hint"]


def test_a_single_component_together_is_a_warning_and_not_an_error(pack):
    """It harvests correctly and narrows nothing, so the pass is right and the key is inert.

    A warning because there is no wrong answer to report: one column is already fully
    constrained by its own per-type list. It is reported at all because an author who
    declared the co-occurrence and lost a component to a bad indent gets no other signal.
    """
    edit_rules(pack, _together([_PAIR[0]]))
    result = pv.validate_pack(pack)
    hit = find(result, "harvest-together-single-component")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"
    assert not codes(result, "error")


def test_a_default_ruleset_naming_no_ruleset_is_an_error(pack):
    """The fallback reverts to use-case directory order, so the default MOVES when a
    procedure is added under an earlier name — while the pack reads as having declared one.
    """
    (pack / "rulesets.yaml").write_text("default_ruleset: nosuchprocedure\n")
    hit = find(pv.validate_pack(pack), "default-ruleset-unknown")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "example_use_case" in hit[0]["detail"]  # what it could have meant
    assert "directory order" in hit[0]["hint"]


def test_a_default_ruleset_naming_a_real_ruleset_is_accepted(pack):
    (pack / "rulesets.yaml").write_text("default_ruleset: example_use_case\n")
    assert not [
        c for c in codes(pv.validate_pack(pack)) if c.startswith("default-ruleset")
    ]


def test_a_default_ruleset_inside_a_use_case_is_reported_as_INERT(pack):
    """The loader merges a use case's `verdicts` and nothing else, so the key never arrives.

    A separate code from the unknown-name one because the fix is different — move it, not
    rename it — and because the declaration here may be perfectly correct and still do
    nothing, which no other diagnostic would mention.
    """
    path = pack / RULES_REL
    path.write_text("default_ruleset: example_use_case\n" + path.read_text())
    hit = find(pv.validate_pack(pack), "default-ruleset-not-at-pack-root")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert hit[0]["path"] == RULES_REL
    assert "rulesets.yaml" in hit[0]["hint"]
    # And it is NOT also reported as unknown: the name is real, the LOCATION is the defect.
    assert "default-ruleset-unknown" not in codes(pv.validate_pack(pack))


def test_a_pack_declaring_no_default_ruleset_is_never_mentioned(pack):
    """Every pack authored before the declaration existed. Silence is the whole point."""
    assert not [
        c for c in codes(pv.validate_pack(pack)) if c.startswith("default-ruleset")
    ]


# --- the exit taken when nothing decisive fired -----------------------------------------
# `no_exclusion_fired` names which label the rollup's last branch exits to. Both ways of
# getting it wrong are silent AND both land on the same value — the historical `fraud` exit,
# which is precisely what a ruleset declaring the key is trying to leave.


def test_an_unknown_no_exclusion_fired_exit_is_an_error(pack):
    """The engine falls back to `fraud` on an unreadable value, so a typo here re-accuses
    every clean subject while the declaration reads as though it took effect."""
    edit_rules(pack, lambda spec: spec.update({"no_exclusion_fired": "clean"}))
    hit = find(pv.validate_pack(pack), "unknown-no-exclusion-fired-exit")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    # It must name the exit it would actually take, not just the allowed set: the fix depends
    # on knowing that the fallback is the reading the author was trying to avoid.
    assert "fraud" in hit[0]["message"]
    assert "false_positive" in (hit[0]["hint"] or "")


def test_an_exit_whose_label_the_ruleset_never_declared_is_a_warning(pack):
    """A recognised class with no label of its own prints the ENGINE's generic wording in
    the middle of a procedure whose other exits are the pack's prose — and it prints it on
    the commonest verdict of all, the subject with nothing against it."""

    def mutate(spec):
        spec["no_exclusion_fired"] = "false_positive"
        spec["labels"] = {"fraud": "SOMETHING BAD"}

    edit_rules(pack, mutate)
    hit = find(pv.validate_pack(pack), "no-exclusion-fired-label-undeclared")
    assert len(hit) == 1
    assert (
        hit[0]["severity"] == "warning"
    ), "a generic label still adjudicates; not an error"
    # And NOT also reported as unknown — the class is real, the LABEL is what is missing.
    assert "unknown-no-exclusion-fired-exit" not in codes(pv.validate_pack(pack))


def test_a_declared_exit_with_its_label_is_accepted(pack):
    def mutate(spec):
        spec["no_exclusion_fired"] = "false_positive"
        spec.setdefault("labels", {})["false_positive"] = "EXAMINED AND CLEARED"

    edit_rules(pack, mutate)
    assert not [c for c in codes(pv.validate_pack(pack)) if "no-exclusion-fired" in c]


def test_a_ruleset_declaring_no_exit_is_never_mentioned(pack):
    """Every ruleset authored before the key existed keeps the historical exit in silence."""
    assert not [c for c in codes(pv.validate_pack(pack)) if "no-exclusion-fired" in c]


# --- the evidence floor guarding that exit ----------------------------------------------
# `min_evaluated_to_clear` is read ONLY on a `false_positive` exit, so every way of getting it
# wrong leaves a declaration that reads as a protection and enforces nothing.


def test_a_floor_on_a_ruleset_that_never_clears_is_reported_as_inert(pack):
    """The inert-key defect: the floor guards the CLEAR exit, so on a fraud-exiting ruleset it
    is a no-op — and a no-op that looks like a safeguard is worse than an absent one."""

    def mutate(spec):
        spec.pop("no_exclusion_fired", None)
        spec["min_evaluated_to_clear"] = 3

    edit_rules(pack, mutate)
    hit = find(pv.validate_pack(pack), "inert-evidence-floor")
    assert len(hit) == 1
    assert (
        hit[0]["severity"] == "warning"
    ), "the pack still works; it just does not do this"


def test_an_unreadable_floor_is_an_error_because_the_engine_silently_takes_one(pack):
    """The number written is then not the number enforced, and the direction of the error is
    a subject cleared on less evidence than the author demanded."""

    def mutate(spec):
        spec["no_exclusion_fired"] = "false_positive"
        spec.setdefault("labels", {})["false_positive"] = "CLEARED"
        spec["min_evaluated_to_clear"] = "three"

    edit_rules(pack, mutate)
    hit = find(pv.validate_pack(pack), "unreadable-evidence-floor")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


def test_a_floor_above_the_condition_count_can_never_be_met(pack):
    """Not a typo the engine can absorb: no subject could ever be cleared on the merits, so
    the procedure would answer INSUFFICIENT DATA for every clean account it ever examines.
    """

    def mutate(spec):
        spec["no_exclusion_fired"] = "false_positive"
        spec.setdefault("labels", {})["false_positive"] = "CLEARED"
        spec["min_evaluated_to_clear"] = len(spec.get("conditions") or []) + 1

    edit_rules(pack, mutate)
    hit = find(pv.validate_pack(pack), "unreachable-evidence-floor")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"


def _two_stubs_and_three_real(spec):
    """Five conditions: two the pack declares no data path for, three that can resolve.

    Built from the template's own single condition so no field name is invented here — the
    fixture ruleset ships exactly one, and it is already the stub, so a floor between three
    and five cannot be expressed without adding to it.
    """
    base = dict((spec.get("conditions") or [{}])[0])
    out = []
    for i in range(2):
        stub = dict(base)
        stub["id"] = f"stub_{i}"
        stub["kind"] = "stub"
        out.append(stub)
    for i in range(3):
        real = dict(base)
        real["id"] = f"real_{i}"
        real["kind"] = "field_flag"
        out.append(real)
    return out


def test_a_floor_above_what_can_ever_be_evaluated_is_caught_even_under_the_count(pack):
    """The bound is what could RESOLVE, not what is declared.

    A `stub` is `unknown` by construction — the engine returns it without reading a row — so
    it can never join the `pass`/`fail` count the floor is compared against. Measured against
    the raw list, a floor equal to the condition count minus one passes the check above and
    then cannot clear a subject on any run, which is exactly the state that check exists to
    name. The message must carry BOTH numbers: which of the two is the mistake — the floor,
    or a stub the author meant to wire — is not the validator's call.
    """

    def mutate(spec):
        spec["no_exclusion_fired"] = "false_positive"
        spec.setdefault("labels", {})["false_positive"] = "CLEARED"
        spec["conditions"] = _two_stubs_and_three_real(spec)
        # Under the declared FIVE, so the pre-existing bound is silent; above the THREE that
        # can ever resolve, so the corrected one is not.
        spec["min_evaluated_to_clear"] = 4

    edit_rules(pack, mutate)
    hit = find(pv.validate_pack(pack), "unreachable-evidence-floor")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"
    assert "2 declare no data path" in hit[0]["message"]


def test_a_floor_within_the_evaluable_conditions_is_not_reported(pack):
    """The other direction, and the reason the check can stay on: two stubs beside a floor
    that the remaining conditions can still satisfy is correct authoring, and a check that
    fires on correct authoring gets switched off."""

    def mutate(spec):
        spec["no_exclusion_fired"] = "false_positive"
        spec.setdefault("labels", {})["false_positive"] = "CLEARED"
        spec["conditions"] = _two_stubs_and_three_real(spec)
        spec["min_evaluated_to_clear"] = 3

    edit_rules(pack, mutate)
    assert not find(pv.validate_pack(pack), "unreachable-evidence-floor")


def test_a_ruleset_declaring_no_floor_is_never_mentioned(pack):
    """Silence for every pack authored before the key existed — the back-compat claim."""
    assert not [c for c in codes(pv.validate_pack(pack)) if "evidence-floor" in c]


def test_a_duplicated_source_name_is_an_error(pack):
    path = pack / "source_catalog.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["sources"].append(dict(doc["sources"][0]))
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    hit = find(pv.validate_pack(pack), "duplicate-source-name")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


# ---------------------------------------------------------------------- reference data


def test_two_use_cases_sharing_a_data_stem_is_an_error(pack):
    """`pack_data` is ONE FLAT DICT keyed by file stem, so the second file wins silently.

    An error rather than a warning because there is no declared precedence between two use
    cases: the loader iterates sorted directories, so whichever sorts later takes over a
    lookup table the other one is still pointing at.
    """
    other = pack / "use_cases" / "second_case" / "data"
    other.mkdir(parents=True)
    src = pack / "use_cases" / "example_use_case" / "data"
    stem = sorted(src.glob("*.yaml"))[0]
    (other / stem.name).write_text(stem.read_text())
    hit = find(pv.validate_pack(pack), "duplicate-data-stem")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


def test_a_use_case_shadowing_a_root_data_file_is_only_a_warning(pack):
    """Root-versus-use-case IS declared precedence, so it is reported and not refused."""
    root_data = pack / "data"
    stem = sorted(root_data.glob("*.yaml"))[0]
    uc = pack / "use_cases" / "example_use_case" / "data"
    (uc / stem.name).write_text(stem.read_text())
    hit = find(pv.validate_pack(pack), "duplicate-data-stem")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"
    assert "wins" in hit[0]["message"]


def test_no_installed_pack_has_a_duplicate_data_stem():
    for name in installed_packs():
        root = PACKS / name
        assert "duplicate-data-stem" not in codes(pv.validate_pack(root)), name


def test_a_lookup_naming_no_data_file_is_an_error(pack):
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec, kind="cohort_membership", lookup={"data": "no_such_table"}
        ),
    )
    hit = find(pv.validate_pack(pack), "unknown-lookup-data")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


def test_a_lookup_naming_a_real_data_file_is_accepted(pack):
    stem = sorted((pack / "data").glob("*.yaml"))[0].stem
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec, kind="cohort_membership", lookup={"data": stem}
        ),
    )
    assert "unknown-lookup-data" not in codes(pv.validate_pack(pack))


# -------------------------------------------------------------------- ids and grouping


def test_a_duplicated_condition_id_is_an_error(pack):
    def mutate(spec):
        one_condition(spec)
        spec["conditions"].append(dict(spec["conditions"][0]))

    edit_rules(pack, mutate)
    hit = find(pv.validate_pack(pack), "duplicate-condition-id")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


def test_a_condition_with_no_id_is_a_warning(pack):
    def mutate(spec):
        one_condition(spec)
        spec["conditions"][0].pop("id")

    edit_rules(pack, mutate)
    assert "condition-missing-id" in codes(pv.validate_pack(pack), "warning")


def test_a_report_group_no_ruleset_declares_is_a_warning(pack):
    edit_rules(pack, lambda spec: one_condition(spec, report_group="nowhere"))
    hit = find(pv.validate_pack(pack), "unknown-report-group")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"


def test_a_condition_naming_no_group_at_all_is_its_own_warning(pack):
    """The other way to match no declared group, and the one the key falls into by DEFAULT.

    Reported under its own code because the remedies share nothing: `unknown-report-group` is
    a name that has gone stale, this is a condition nobody filed — and it is the shape a
    condition appended after the `condition_groups:` block was written arrives in. Both used
    to be dropped from the report outright (see `test_report_generation.py`), so a pack could
    ship a decisive check the verdict counted and no reader could find.

    Asserted as EXCLUSIVE of the sibling code, because one condition cannot be both, and a
    check written as two independent `if`s would emit both on a blank.
    """
    edit_rules(pack, lambda spec: one_condition(spec, report_group=""))
    res = pv.validate_pack(pack)
    hit = find(res, "missing-report-group")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"
    assert "unknown-report-group" not in codes(res)


def test_a_ruleset_declaring_no_groups_is_not_asked_which_group(pack):
    """Silence where there is nothing to file under — the renderer prints such a pack FLAT.

    Every condition in a groupless ruleset names no declared group, there being none, so a
    check that only asked "does this name a declared group" would warn on every condition of
    every pack that never adopted grouping. Both codes must be silent, not just the new one.
    """

    def mutate(spec):
        one_condition(spec, report_group="")
        spec.pop("condition_groups", None)

    edit_rules(pack, mutate)
    present = codes(pv.validate_pack(pack))
    assert "missing-report-group" not in present
    assert "unknown-report-group" not in present


def test_installed_rulesets_have_no_grouping_or_id_defects():
    for name in installed_packs():
        root = PACKS / name
        present = codes(pv.validate_pack(root))
        for code in (
            "unknown-report-group",
            "missing-report-group",
            "duplicate-condition-id",
            "condition-missing-id",
        ):
            assert code not in present, (name, code)


def test_a_ruleset_with_no_conditions_is_a_warning(pack):
    edit_rules(pack, lambda spec: spec.update({"conditions": []}))
    assert "ruleset-no-conditions" in codes(pv.validate_pack(pack), "warning")


# ------------------------------------------------------------------------- scope label


def test_the_out_of_scope_label_is_required_only_when_a_scope_gate_exists(pack):
    """Conditional, because unconditional would be wrong advice.

    Out of scope means the procedure does not apply, which is not the same as a subject
    examined and cleared — so a ruleset that can reach a scope gate needs its own wording.
    A ruleset with no scope gate can never render that label, and telling its author to
    write one is telling them to author dead prose.
    """

    def add_gate(spec):
        one_condition(spec, gate="scope")
        spec["labels"].pop("out_of_scope", None)

    edit_rules(pack, add_gate)
    hit = find(pv.validate_pack(pack), "missing-out-of-scope-label")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"

    edit_rules(
        pack, lambda spec: spec["labels"].update({"out_of_scope": "OUT OF SCOPE"})
    )
    assert "missing-out-of-scope-label" not in codes(pv.validate_pack(pack))


def test_no_scope_gate_means_no_advice_about_the_label(pack):
    edit_rules(
        pack,
        lambda spec: (one_condition(spec), spec["labels"].pop("out_of_scope", None)),
    )
    assert "missing-out-of-scope-label" not in codes(pv.validate_pack(pack))


# --------------------------------------------------------------------------- concepts


def test_a_case_brief_asking_for_an_unknown_concept_is_a_warning(pack):
    """The brief silently omits what it cannot find, so nothing else would ever say so."""
    edit_rules(
        pack,
        lambda spec: spec.setdefault("case_builder", {}).update(
            {"concepts": ["not_a_concept"]}
        ),
    )
    hit = find(pv.validate_pack(pack), "orphan-concept-id")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"


def test_a_shared_concept_satisfies_a_use_case_brief(pack):
    """Shared concepts load UNTAGGED, so a use case may name one it does not own.

    A check that only looked inside the use case's own `concepts/` would flag every shared
    reference — which is the mechanism, not a defect.
    """
    shared = sorted((pack / "shared" / "concepts").glob("*.md"))[0]
    concept_id = shared.read_text().split("concept_id:")[1].split("\n")[0].strip()
    edit_rules(
        pack,
        lambda spec: spec.setdefault("case_builder", {}).update(
            {"concepts": [concept_id]}
        ),
    )
    assert "orphan-concept-id" not in codes(pv.validate_pack(pack))


def test_no_installed_pack_has_an_orphan_concept_id():
    for name in installed_packs():
        root = PACKS / name
        assert "orphan-concept-id" not in codes(pv.validate_pack(root)), name


# ------------------------------------------------------------------ imports and vocabulary


def test_an_unresolvable_check_import_is_an_error_listing_what_exists(pack):
    """The one already-fatal path in pack loading, surfaced before the pack is loaded.

    `_resolve_check_imports` raises rather than dropping the condition, precisely because a
    dropped condition reads exactly like a source that returned no rows. Here that raise
    becomes a diagnostic instead of a stack trace, and the detail carries the available ids
    — a typo in a namespace is otherwise a hunt through the library.
    """

    def mutate(spec):
        spec["conditions"] = [
            {"use": "example_checks/no_such_check", "report_group": "validation"}
        ]

    edit_rules(pack, mutate)
    hit = find(pv.validate_pack(pack), "unresolvable-check-import")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "example_checks/" in hit[0]["detail"]


def test_a_pack_with_no_vocabulary_is_an_error(pack):
    """A pack dir with no vocabulary takes a suite-wide test down with it.

    Nothing then keeps this domain's nouns out of the engine, which is the invariant the
    whole pack boundary rests on.
    """
    (pack / pv.VOCABULARY_FILE).write_text("domain_vocabulary: []\n")
    hit = find(pv.validate_pack(pack), "missing-domain-vocabulary")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"

    (pack / pv.VOCABULARY_FILE).unlink()
    assert "missing-domain-vocabulary" in codes(pv.validate_pack(pack), "error")


def test_claiming_a_word_the_engine_already_uses_is_an_error(pack):
    """The one diagnostic about a file OUTSIDE the pack, and it fires RETROACTIVELY.

    The neutrality scan reads every installed pack's vocabulary, so installing a pack that
    claims a word already in `src/` turns a green tree red with nothing in the engine
    having changed. The diagnostic is anchored on the pack's word list because that is the
    fixable end, and it cites the offending engine line so the operator can see which of the
    two is wrong.
    """
    doc = yaml.safe_load((pack / pv.VOCABULARY_FILE).read_text())
    doc["domain_vocabulary"] = list(doc["domain_vocabulary"]) + ["correlation"]
    (pack / pv.VOCABULARY_FILE).write_text(yaml.safe_dump(doc, sort_keys=False))
    hit = find(pv.validate_pack(pack), "neutrality-collision")
    assert hit and hit[0]["severity"] == "error"
    assert "correlation" in hit[0]["message"]
    assert (
        ".py:" in hit[0]["detail"]
    ), "the operator needs the engine line, not just the word"
    assert hit[0]["line"] > 0, "and the line in their own file to fix"


def test_a_word_the_engine_legitimately_owns_is_exempt(pack):
    """`_ENGINE_OWNS` is a real exemption, not a comment: a backend kind names its product."""
    doc = yaml.safe_load((pack / pv.VOCABULARY_FILE).read_text())
    doc["domain_vocabulary"] = list(doc["domain_vocabulary"]) + sorted(pv._ENGINE_OWNS)
    (pack / pv.VOCABULARY_FILE).write_text(yaml.safe_dump(doc, sort_keys=False))
    assert "neutrality-collision" not in codes(pv.validate_pack(pack))


def test_the_neutrality_boundary_is_not_a_word_boundary():
    """THE ONE BEHAVIOUR THAT MAKES THE SCAN WORTH RUNNING, asserted rather than commented.

    `\\b` versus this boundary is the whole design: Python's `\\b` treats `_` as a word
    character, so `\\b` finds `alpha` in `alpha.beta` but NOT in `some_alpha_column` — and a
    domain noun buried in a snake_case identifier is the commonest shape a leak takes. Four
    live leaks were measured hiding in exactly that position.

    So the assertions run in both directions. Matching inside a separator is the requirement;
    NOT matching an adjacent letter is what keeps the check believable, since a scan whose hits
    are mostly false gets weakened and then deleted.
    """
    pattern = pv.vocabulary_pattern(["alpha"])
    assert pattern.search("some_alpha_column"), "must match inside snake_case"
    assert pattern.search("alpha.beta")
    assert not pattern.search(
        "alphas"
    ), "an inflection is listed explicitly, not inferred"
    assert not pattern.search("realpha")


def test_an_empty_vocabulary_list_yields_no_pattern():
    assert pv.vocabulary_pattern([]) is None
    assert pv.vocabulary_pattern(["", None]) is None


# ------------------------------------------------------------- field paths vs the inventory
#
# The source is retrieved, every stage reports success, and the condition reads `unknown`
# because the path names no column the data has. In the report that is indistinguishable
# from a source with nothing to say.
#
# The check is a warning: a sampled inventory cannot prove absence, a SQL query may
# project an alias the pack declares in `query_hints` and no table has, and a pack
# documents only the sources it chose to. A false alarm on correct authoring is how a
# check earns a scroll-past.


def _template_path_condition(spec, paths, **overrides):
    """A condition on the template's one documented source, naming `paths`.

    The template's own condition is `kind: stub` with no field paths at all, so a mutation
    test has to introduce one — there is nothing to make dead otherwise.
    """
    one_condition(spec, kind="field_flag", fields=paths, **overrides)


def test_a_condition_whose_every_path_is_absent_from_the_schema_is_a_warning(pack):
    edit_rules(pack, lambda spec: _template_path_condition(spec, ["no.such.leaf"]))
    hits = find(pv.validate_pack(pack), "field-path-not-in-schema")
    assert len(hits) == 1
    assert hits[0]["severity"] == "warning", "a sampled inventory cannot prove absence"
    assert hits[0]["line"] > 0, "the operator needs a cursor"
    assert "no.such.leaf" in hits[0]["detail"]
    assert (
        "example_source" in hits[0]["message"]
    ), "name the source it was checked against"
    assert "unknown" in hits[0]["message"], "say what the check will actually read"


def test_a_real_leaf_of_the_template_inventory_is_accepted(pack):
    """The positive control. Without it the check could be passing by never resolving."""
    edit_rules(
        pack,
        lambda spec: _template_path_condition(spec, ["example_elements.element_code"]),
    )
    assert "field-path-not-in-schema" not in codes(pv.validate_pack(pack))


def test_one_live_spelling_among_dead_ones_is_not_reported(pack):
    """A candidate LIST is OR-ed and first-present, so a stale sibling changes no outcome.

    This is what makes the check reportable at all: per-path it would fire on every pack that
    names two spellings of one field across two backends, which is correct authoring.
    """
    edit_rules(
        pack,
        lambda spec: _template_path_condition(
            spec, ["gone.old.name", "example_elements.element_code"]
        ),
    )
    assert "field-path-not-in-schema" not in codes(pv.validate_pack(pack))


def test_a_source_with_no_schema_doc_is_never_reported(pack):
    """Absence of an inventory is not evidence about a path.

    A pack documents the sources it chose to; checking against a schema that does not exist
    would make writing the first condition on a new source a warning.
    """
    (pack / "schemas" / "example_table.yaml").unlink()
    edit_rules(
        pack, lambda spec: _template_path_condition(spec, ["no.such.leaf.at.all"])
    )
    result = pv.validate_pack(pack)
    assert "field-path-not-in-schema" not in codes(result)
    assert (
        result["counts"]["field_path_lists"] == 0
    ), "a pack that documents nothing must not read as a pack that passed"


def test_a_single_segment_name_is_never_treated_as_a_path(pack):
    """A bare word in a ruleset is an enum value (`decisive_on: fail`, `match: exact`), a
    projection alias, or prose.

    Distinguishing it from a row path requires an engine-owned map of which key holds
    which; that map is not maintained. Admitting them reports 60 / 45 / 12 dead names
    across the installed packs and every one is correct, so the cost is stated rather than
    hidden: a dead single-segment path is not caught, and this test records that.
    """
    edit_rules(
        pack, lambda spec: _template_path_condition(spec, ["definitely_not_a_column"])
    )
    assert "field-path-not-in-schema" not in codes(pv.validate_pack(pack))


# ------------------------------- the one single-segment name whose deadness is decidable
#
# A bare word that is the underscore-flattened prefix of a recorded struct container is
# provably a container. An enum member cannot become one by accident. The defect: the path
# resolves (by bare membership) to a struct, whose terminal leaves the evaluator discards.


def _nested_struct(pack, table="example_table", column="outer"):
    """Record `column.inner.leaf` on the template's inventory, container leaves included.

    A real inventory records a struct's own path beside the leaves under it, and that is
    load-bearing here rather than incidental: it is what makes `outer` resolve by bare
    membership while still being a container, which is the half of the rule the ordering
    exists for.
    """
    schema = pack / "schemas" / f"{table}.yaml"
    doc = yaml.safe_load(schema.read_text())
    doc["tables"][table]["columns"][column] = {
        "type": "struct",
        "leaves": [
            {"path": column, "kind": "struct"},
            {"path": f"{column}.inner", "kind": "struct"},
            {"path": f"{column}.inner.leaf", "kind": "scalar"},
        ],
    }
    schema.write_text(yaml.safe_dump(doc, sort_keys=False))


def test_a_flat_name_that_flattens_a_recorded_struct_is_reported(pack):
    """`outer_inner` is how the leaf `outer.inner.leaf` looks after an aliasing generator.

    This is the shape a source rebound from a FLAT table onto a NESTED one leaves behind: the
    declaration was right about the old table and reads exactly as right about the new one.
    """
    _nested_struct(pack)
    edit_rules(pack, lambda spec: _template_path_condition(spec, ["outer_inner"]))
    hits = find(pv.validate_pack(pack), "field-name-flattens-a-struct")
    assert len(hits) == 1
    assert hits[0]["severity"] == "warning", "an inventory is not an authority on absence"
    assert hits[0]["line"] > 0, "the operator needs a cursor"
    assert "outer_inner" in hits[0]["message"]
    assert (
        "outer.inner.leaf" in hits[0]["message"]
    ), "name a witness — the fix is choosing a leaf, not correcting a spelling"
    assert "unknown" in hits[0]["message"], "say what the check will actually read"


def test_a_name_the_inventory_RECORDS_is_reported_when_it_is_a_container(pack):
    """THE ORDERING, and the half of the rule that pays.

    An inventory records a struct's own name beside its leaves, so a name that IS the
    container resolves by `_path_in_schema`'s first rule — and resolves to the thing the
    evaluator throws away. Membership answers "does this name exist", which is not the
    question; the witness answers "does anything exist underneath it", which decides it. With
    membership winning, one of the two sites a live rebind left behind was invisible.
    """
    _nested_struct(pack)
    edit_rules(pack, lambda spec: _template_path_condition(spec, ["outer"]))
    hits = find(pv.validate_pack(pack), "field-name-flattens-a-struct")
    assert len(hits) == 1
    assert "outer" in hits[0]["message"]


def test_a_flat_name_with_nothing_recorded_under_it_is_not_reported(pack):
    """The precision control, and the reason the rule can exist beside the silence above.

    `example_id` is a recorded scalar: nothing is recorded underneath it, so it has no witness
    and lands on a leaf. A bare word that is not a container must read exactly as it did
    before this rule — which is the accepted cost the sibling test records, left intact.
    """
    edit_rules(pack, lambda spec: _template_path_condition(spec, ["example_id"]))
    result = pv.validate_pack(pack)
    assert "field-name-flattens-a-struct" not in codes(result)
    edit_rules(
        pack, lambda spec: _template_path_condition(spec, ["definitely_not_a_column"])
    )
    assert "field-name-flattens-a-struct" not in codes(pv.validate_pack(pack))


def test_a_member_that_lands_on_a_leaf_silences_the_container_report(pack):
    """Same first-present rule the dotted check follows, for the same reason.

    The engine ORs across a candidate list and takes the first that resolves, so a container
    spelling beside a working one changes no outcome — and reporting it would be reporting a
    pack that works, in the check whose whole value is that it is quiet.
    """
    _nested_struct(pack)
    edit_rules(
        pack, lambda spec: _template_path_condition(spec, ["outer_inner", "example_id"])
    )
    assert "field-name-flattens-a-struct" not in codes(pv.validate_pack(pack))


def test_a_key_whose_value_is_supposed_to_name_a_container_is_not_reported(pack):
    """`arrays:` is COUNTED by descending it, which is the opposite of reading a leaf.

    The one exemption the measurement asked for, and it is not a special case of the rule but
    a different question: naming the container is correct authoring here — the template's own
    inventory says so, because an empty array resolves for the container and for nothing
    underneath, so an absence check that named a sub-key would report `unknown` on every row.
    Kept out of `_NON_PATH_KEYS`, though: a misspelled container is as dead as a misspelled
    leaf, so the EXISTENCE check must still read it.
    """
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec, kind="element_absence", arrays=["example_elements"]
        ),
    )
    result = pv.validate_pack(pack)
    assert "field-name-flattens-a-struct" not in codes(result)
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec, kind="element_absence", arrays=["no.such.container"]
        ),
    )
    assert "field-path-not-in-schema" in codes(
        pv.validate_pack(pack)
    ), "exempt from the container rule, NOT from the existence check"


def test_a_container_name_does_not_count_toward_the_checked_lists(pack):
    """A flat name is not a candidate path, so it may not inflate the coverage count.

    `field_path_lists` is the number that stops a pack which documents nothing from reading
    like a pack that passed. Counting a name the existence check cannot evaluate would make
    that number claim coverage it does not have — in the direction that hides the gap.
    """
    _nested_struct(pack)
    edit_rules(pack, lambda spec: _template_path_condition(spec, ["outer.inner.leaf"]))
    dotted = pv.validate_pack(pack)["counts"]["field_path_lists"]
    assert dotted >= 1, "the positive control: a dotted list IS counted"
    edit_rules(pack, lambda spec: _template_path_condition(spec, ["outer_inner"]))
    reported = pv.validate_pack(pack)
    assert "field-name-flattens-a-struct" in codes(reported)
    assert reported["counts"]["field_path_lists"] == dotted - 1


def test_an_underscore_flattened_alias_resolves(pack):
    """`a.b.c` also arrives as `a_b_c` — the SQL generator aliases struct leaves that way.

    `resolve_path` matches both, so a check that did not would report a working condition.
    """
    schema = pack / "schemas" / "example_table.yaml"
    doc = yaml.safe_load(schema.read_text())
    # Recorded in the FLAT shape and under the alias only, so no other rule can reach it:
    # `nested_struct` is not itself a recorded path, so the opaque-prefix rule has no prefix
    # to accept and the suffix rule has no dotted tail to match.
    doc["tables"]["example_table"]["fields"] = {
        "nested_struct_inner_leaf": {"present_in_sampled_pct": 100.0}
    }
    schema.write_text(yaml.safe_dump(doc, sort_keys=False))
    edit_rules(
        pack, lambda spec: _template_path_condition(spec, ["nested_struct.inner.leaf"])
    )
    assert "field-path-not-in-schema" not in codes(pv.validate_pack(pack))


def test_a_path_relative_to_a_named_struct_resolves_by_its_suffix(pack):
    """A path written relative to an enclosing struct carries FEWER segments than the leaf.

    MEASURED on the installed packs: twelve real candidate paths resolve on this rule and no
    other — the ones a shared check writes relative to the array it declares, and the ones a
    `views` block writes relative to its own prefix. Without the rule every one of them is
    reported, which is a warning on correct authoring in the pack's most-reused checks.

    The recorded leaf here is TWO segments deeper than the declaration, so neither the alias
    nor the opaque-prefix rule can reach it.
    """
    schema = pack / "schemas" / "example_table.yaml"
    doc = yaml.safe_load(schema.read_text())
    doc["tables"]["example_table"]["columns"]["outer"] = {
        "type": "struct",
        "leaves": [
            {"path": "outer", "kind": "struct"},
            {"path": "outer.middle.inner.leaf", "kind": "scalar"},
        ],
    }
    schema.write_text(yaml.safe_dump(doc, sort_keys=False))
    edit_rules(pack, lambda spec: _template_path_condition(spec, ["inner.leaf"]))
    assert "field-path-not-in-schema" not in codes(pv.validate_pack(pack))


def test_a_tail_under_an_opaque_column_resolves(pack):
    """An inventory that stops at a blob says nothing about what is inside it.

    A JSON/VARIANT column is recorded as one leaf; the pack legitimately reads a path
    through it, and what that path is cannot be known from here — so a prefix present with
    no children recorded under it is accepted rather than guessed at.
    """
    schema = pack / "schemas" / "example_table.yaml"
    doc = yaml.safe_load(schema.read_text())
    doc["tables"]["example_table"]["columns"]["blob"] = {
        "type": "string",
        "leaves": [{"path": "blob", "kind": "scalar"}],
    }
    schema.write_text(yaml.safe_dump(doc, sort_keys=False))
    edit_rules(
        pack, lambda spec: _template_path_condition(spec, ["blob.whatever.is.in.it"])
    )
    assert "field-path-not-in-schema" not in codes(pv.validate_pack(pack))


def _aliased_group(pack, *entries):
    """Record a nested struct on the template's inventory and project `entries` beside it.

    The struct is what makes these tests about TRANSLATION rather than about silence: an
    entry whose only source path is a bare column resolves to nothing pinnable (the template
    projects two single-segment scalars), so an alias over it would be accepted by the
    unknowable branch and the assertions below would pass without the rename ever resolving.
    `outer.inner.leaf` gives the alias a recorded leaf to be translated back to.
    """
    schema = pack / "schemas" / "example_table.yaml"
    doc = yaml.safe_load(schema.read_text())
    doc["tables"]["example_table"]["columns"]["outer"] = {
        "type": "struct",
        "leaves": [
            {"path": "outer", "kind": "struct"},
            {"path": "outer.inner", "kind": "struct"},
            {"path": "outer.inner.leaf", "kind": "scalar"},
        ],
    }
    schema.write_text(yaml.safe_dump(doc, sort_keys=False))
    catalog = pack / "source_catalog.yaml"
    cat = yaml.safe_load(catalog.read_text())
    source = cat["sources"][0]
    source["projection"] = [*(source.get("projection") or []), *entries]
    catalog.write_text(yaml.safe_dump(cat, sort_keys=False))


def test_a_path_read_under_a_declared_projection_alias_is_checked_to_its_leaf(pack):
    """The second reason this check is a warning, stated as a resolution rule.

    A `projection` entry may narrow a repeated group inside the backend and carry its own
    ``AS`` — and then the row arrives under the ALIAS while every inventory of the target
    describes the path underneath. The two ways of not translating that are the two ways of
    being wrong: report every read under an alias (a false alarm on correct authoring, in the
    check whose whole value is that it is quiet) or accept the whole subtree (giving up the
    misspelled leaf the check exists for). So the alias is put back to the one path it renames
    and the comparison happens at the leaf.
    """
    _aliased_group(pack, "filter(outer.inner, x -> x.leaf IS NOT NULL) AS picked")
    edit_rules(pack, lambda spec: _template_path_condition(spec, ["picked.leaf"]))
    result = pv.validate_pack(pack)
    assert "field-path-not-in-schema" not in codes(result)
    assert result["counts"]["field_path_lists"] >= 1, "checked, not skipped"


def test_a_misspelled_leaf_under_a_projection_alias_is_still_reported(pack):
    """The half that makes the rule above a translation rather than an exemption.

    Same alias, same entry, one segment wrong — and it must still fire, or the fix is
    "accept anything written under an alias", which is the reading that hands a permanently
    `unknown` condition the check's own blessing.
    """
    _aliased_group(pack, "filter(outer.inner, x -> x.leaf IS NOT NULL) AS picked")
    edit_rules(pack, lambda spec: _template_path_condition(spec, ["picked.leef"]))
    hits = find(pv.validate_pack(pack), "field-path-not-in-schema")
    assert len(hits) == 1
    assert "picked.leef" in hits[0]["detail"]


def test_an_alias_the_projection_cannot_pin_to_one_path_is_silent(pack):
    """Unknowable is not absent.

    An entry combining TWO paths renames neither of them, so the remainder of the candidate
    cannot be attributed to any recorded leaf. There is nothing to compare, and a warning here
    would be the check reporting its own inability to translate — the same reasoning as the
    silence where a source has no inventory at all, one declaration further in.
    """
    _aliased_group(
        pack, "concat(outer.inner.leaf, example_elements.element_code) AS both"
    )
    edit_rules(pack, lambda spec: _template_path_condition(spec, ["both.whatever"]))
    assert "field-path-not-in-schema" not in codes(pv.validate_pack(pack))


def test_a_decode_target_is_exempt_because_no_backend_can_report_it(pack):
    """A pack-declared decode writes its table at a path no discovered schema can carry.

    `encoded_fields` is the declaration and `src/encoded_fields.decoded_tables` derives the
    target (including the default `into`), so the exemption is asked of the module that owns
    the shape instead of restated here. Without it this is the loudest false positive in the
    check — the decode is exactly the mechanism that makes those columns readable.
    """
    catalog = pack / "source_catalog.yaml"
    doc = yaml.safe_load(catalog.read_text())
    doc["sources"][0]["encoded_fields"] = [
        {"field": "payload", "encoding": "base64", "into": "payload_rows"}
    ]
    catalog.write_text(yaml.safe_dump(doc, sort_keys=False))
    edit_rules(
        pack, lambda spec: _template_path_condition(spec, ["payload_rows.column"])
    )
    result = pv.validate_pack(pack)
    assert "field-path-not-in-schema" not in codes(result)
    assert (
        result["counts"]["field_path_lists"] >= 1
    ), "the list was checked, not skipped"


def test_confirm_fields_are_resolved_against_the_confirm_in_sources(pack):
    """WHERE THE EVALUATOR LOOKS IS WHERE THE CHECK MUST LOOK.

    `confirm_fields` resolve against the sibling `confirm_in:` sources, not the condition's
    own — so against the condition's source every one of them would read as dead.
    """

    def mutate(spec):
        one_condition(
            spec,
            kind="field_flag",
            source="undocumented",
            confirm_in=["record"],
            confirm_fields=["example_elements.element_code"],
        )
        spec["sources"]["undocumented"] = "example_source"

    edit_rules(pack, mutate)
    assert "field-path-not-in-schema" not in codes(pv.validate_pack(pack))


def test_an_entity_type_name_in_declares_is_not_a_path(pack):
    """`alert_record.declares[].field` holds ENTITY TYPE names, not row paths."""

    def mutate(spec):
        spec["alert_record"] = {
            "source": "record",
            "declares": [{"field": "example.entity.type", "from": "x"}],
        }

    edit_rules(pack, mutate)
    assert "field-path-not-in-schema" not in codes(pv.validate_pack(pack))


def _es_fields(pack, fields):
    """Record `fields` in the generated ELASTICSEARCH shape on the template's one table.

    The template documents a SQL source (`columns` + `leaves`), so the null-only case has to
    be introduced in the other recorded shape — which is the shape it occurs in, because a
    null-only leaf is what a document walk produces from an explicit JSON `null`.
    """
    schema = pack / "schemas" / "example_table.yaml"
    doc = yaml.safe_load(schema.read_text())
    doc["tables"]["example_table"]["fields"] = fields
    schema.write_text(yaml.safe_dump(doc, sort_keys=False))


def test_a_leaf_that_only_ever_held_null_does_not_satisfy_a_path(pack):
    """A KEY PRESENT WITH A NULL VALUE IS NOT A FIELD A CONDITION CAN READ.

    MEASURED on the installed pack: `access_alerts.alertId` is recorded with
    `json_types: ['null']` at `present_in_sampled_pct: 0.09` — one sampled document carrying
    an explicit null. The generated inventory records a leaf whenever the WALK REACHED ITS
    NAME, which is not the same question as whether any document has a value there, and
    elasticsearch `exists` does not match a null. So counting it made the validator CONFIRM
    a path that can only ever read `unknown` — the single failure this whole check exists to
    report, arriving with the check's own blessing. Two of this exercise's own probes
    disagreed about that source before the cause was found here.
    """
    _es_fields(
        pack,
        {"meta.alert_id": {"json_types": ["null"], "present_in_sampled_pct": 0.09}},
    )
    edit_rules(pack, lambda spec: _template_path_condition(spec, ["meta.alert_id"]))
    hits = find(pv.validate_pack(pack), "field-path-not-in-schema")
    assert len(hits) == 1
    assert "meta.alert_id" in hits[0]["detail"]


def test_a_leaf_that_is_null_OR_a_value_still_satisfies_a_path(pack):
    """The other direction, and the reason the filter reads `all` rather than `any`.

    `['null', 'string']` is an ordinary OPTIONAL field, which is most of a real inventory —
    a filter that dropped those would report the pack's working conditions instead. And a
    leaf recording no types at all stays too: silence about a type is not evidence, the same
    rule that makes this diagnostic a warning in the first place.
    """
    for spec in (
        {"json_types": ["null", "string"], "present_in_sampled_pct": 71.0},
        {"json_types": [], "present_in_sampled_pct": 71.0},
        {"present_in_sampled_pct": 71.0},
    ):
        _es_fields(pack, {"meta.alert_id": spec})
        edit_rules(pack, lambda s: _template_path_condition(s, ["meta.alert_id"]))
        assert "field-path-not-in-schema" not in codes(pv.validate_pack(pack)), spec


def test_every_installed_pack_and_the_template_stay_at_or_below_the_measured_count():
    """The false-positive budget, pinned as a number. A jump is a resolver regression."""
    for root in [PACKS / n for n in installed_packs()] + [TEMPLATE]:
        result = pv.validate_pack(root)
        hits = find(result, "field-path-not-in-schema")
        assert len(hits) <= 1, [(h["path"], h["line"], h["detail"]) for h in hits]
        assert all(h["severity"] == "warning" for h in hits)
        assert result["ok"], "this check may never block a save"


# --------------------------------------------- field paths vs the declared projection
#
# The second question about the same paths; the answers are independent. The check above
# asks whether the leaf exists; this asks whether it will be in the row. A source
# declaring `projection:` makes that list the required SELECT, so a condition naming a
# documented leaf outside it reads `unknown` on every row with the schema check silent.
#
# Silent on a source with no inventory, no declared projection, and every backend that
# ignores the key.


def _project(pack, columns):
    """Declare `columns` as `example_source`'s projection (`None` removes the key)."""
    path = pack / "source_catalog.yaml"
    doc = yaml.safe_load(path.read_text())
    if columns is None:
        doc["sources"][0].pop("projection", None)
    else:
        doc["sources"][0]["projection"] = columns
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def test_a_documented_leaf_outside_the_declared_projection_is_a_warning(pack):
    """The finding. The template projects two scalars and documents an array leaf beside them.

    Both halves are asserted, because the whole value of this check is that it fires where the
    schema check CANNOT: the path exists, so `field-path-not-in-schema` is correctly silent,
    and the row will not carry it.
    """
    edit_rules(
        pack,
        lambda spec: _template_path_condition(spec, ["example_elements.element_code"]),
    )
    result = pv.validate_pack(pack)
    hits = find(result, "field-path-outside-projection")
    assert len(hits) == 1
    assert hits[0]["severity"] == "warning", "the prompt permits extra columns"
    assert hits[0]["line"] > 0, "the operator needs a cursor"
    assert "example_elements.element_code" in hits[0]["detail"]
    assert "example_source" in hits[0]["message"], "name the source it was checked against"
    assert "unknown" in hits[0]["message"], "say what the check will actually read"
    assert (
        "field-path-not-in-schema" not in codes(result)
    ), "the leaf IS documented — the two questions must be answered independently"
    assert result["ok"], "this check may never block a save"


def test_a_projected_leaf_is_accepted(pack):
    """The positive control, and it also pins the SILENCE as a check rather than a skip."""
    _project(pack, ["example_elements.element_code"])
    edit_rules(
        pack,
        lambda spec: _template_path_condition(spec, ["example_elements.element_code"]),
    )
    result = pv.validate_pack(pack)
    assert "field-path-outside-projection" not in codes(result)
    assert result["counts"]["projected_path_lists"] >= 1, "the list was checked"


def test_a_leaf_under_an_already_projected_struct_needs_no_entry_of_its_own(pack):
    """A projected struct brings its own leaves with it, which is why the test is a SUB-INVENTORY.

    Projecting `example_elements` selects the array; every recorded leaf under it arrives with
    it. Reporting those would be a warning on the ordinary way a nested source is projected —
    one installed pack projects six whole structs.
    """
    _project(pack, ["example_elements"])
    edit_rules(
        pack,
        lambda spec: _template_path_condition(spec, ["example_elements.element_code"]),
    )
    assert "field-path-outside-projection" not in codes(pv.validate_pack(pack))


def test_a_path_relative_to_a_projected_struct_is_not_reported(pack):
    """MEASURED: three of the first five findings this rule produced were this shape.

    A shared check declares the array it reads and writes its paths RELATIVE to it, so the
    path carries fewer segments than the recorded leaf. The naive rule — is the path a prefix
    match on a projection entry — reports every one of them, which is a warning on the pack's
    most-reused checks. Feeding the sub-inventory through `_path_in_schema` reuses the same
    suffix rule the resolver has, so it resolves instead.

    The second half is what stops the fixture being vacuous: with the enclosing struct NOT
    projected, the identical path fires. The rule is the projection, not the shape of the path.
    """
    schema = pack / "schemas" / "example_table.yaml"
    doc = yaml.safe_load(schema.read_text())
    doc["tables"]["example_table"]["columns"]["outer"] = {
        "type": "struct",
        "leaves": [
            {"path": "outer", "kind": "struct"},
            {"path": "outer.middle.inner.leaf", "kind": "scalar"},
        ],
    }
    schema.write_text(yaml.safe_dump(doc, sort_keys=False))
    edit_rules(pack, lambda spec: _template_path_condition(spec, ["inner.leaf"]))

    _project(pack, ["outer"])
    assert "field-path-outside-projection" not in codes(pv.validate_pack(pack))

    _project(pack, ["example_id"])
    assert "field-path-outside-projection" in codes(pv.validate_pack(pack))


def test_a_leaf_inside_an_aliased_projection_entry_is_not_reported(pack):
    """The sub-inventory grows from the path an entry reads, not from the alias it lands under.

    An entry with an AS alias: the alias is not a schema path, so growing the candidate set
    from it finds nothing under it and the subtree collapses to one name. Every condition
    reading a leaf inside a correctly-aliased struct is then reported.

    The second half stops the fixture being vacuous: the alias does not become a licence,
    so a leaf the projection genuinely does not cover still fires beside it.
    """
    _aliased_group(pack, "outer.inner AS outer_inner")

    edit_rules(pack, lambda spec: _template_path_condition(spec, ["outer.inner.leaf"]))
    result = pv.validate_pack(pack)
    assert "field-path-outside-projection" not in codes(result)
    assert result["counts"]["projected_path_lists"] >= 1, "checked, not skipped"

    edit_rules(
        pack,
        lambda spec: _template_path_condition(spec, ["example_elements.element_code"]),
    )
    assert "field-path-outside-projection" in codes(pv.validate_pack(pack))


def test_a_leaf_under_an_alias_the_entry_cannot_pin_is_accepted_as_opaque(pack):
    """Unknowable is not uncovered, and the accepting rule is the resolver's own fourth one.

    An entry combining TWO paths renames neither, so it contributes no root and only its own
    name — which is the honest answer, since what an arbitrary expression returns underneath
    itself is not knowable from the declaration. That name is then a prefix with nothing
    recorded under it, i.e. exactly the opaque-prefix case `_path_in_schema` already accepts,
    so the silence here is the resolver's rule and not a special case for aliases.
    """
    _aliased_group(pack, "concat(outer.inner.leaf, example_id) AS both")
    edit_rules(pack, lambda spec: _template_path_condition(spec, ["both.whatever"]))
    assert "field-path-outside-projection" not in codes(pv.validate_pack(pack))


def test_a_source_with_no_schema_doc_is_never_reported_against_its_projection(pack):
    """Without an inventory the sub-inventory IS the projection list, and that is not enough.

    A relative path can no longer be recognised as relative, an opaque blob's tail cannot be
    recognised as a tail, and the check would report correct authoring on every source a pack
    chose not to document. Same silence as the schema check, arrived at from the other side.
    """
    (pack / "schemas" / "example_table.yaml").unlink()
    edit_rules(
        pack,
        lambda spec: _template_path_condition(spec, ["example_elements.element_code"]),
    )
    result = pv.validate_pack(pack)
    assert "field-path-outside-projection" not in codes(result)
    assert (
        result["counts"]["projected_path_lists"] == 0
    ), "a pack nothing could be checked against must not read as a pack that passed"


def test_a_source_declaring_no_projection_is_never_reported(pack):
    """`projection:` is optional — omitting it means `SELECT *`, so nothing is outside it."""
    _project(pack, None)
    edit_rules(
        pack,
        lambda spec: _template_path_condition(spec, ["example_elements.element_code"]),
    )
    result = pv.validate_pack(pack)
    assert "field-path-outside-projection" not in codes(result)
    assert result["counts"]["projected_path_lists"] == 0


def test_a_backend_that_ignores_the_projection_is_never_reported(pack):
    """The key is INERT on every kind but one, and reporting an inert declaration is noise.

    Only the Databricks retriever reads `config.get("projection")`; on the other kinds the
    list is carried and never used, so a path "outside" it is outside nothing. The condition
    and the projection are the ones that DO fire above — only the endpoint kind changes.
    """
    path = pack / "source_catalog.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["sources"][0]["endpoints"] = {
        "kind": "elasticsearch",
        "cluster": "example_cluster",
        "indices": ["example-*"],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    edit_rules(
        pack,
        lambda spec: _template_path_condition(spec, ["example_elements.element_code"]),
    )
    result = pv.validate_pack(pack)
    assert "field-path-outside-projection" not in codes(result)
    assert result["counts"]["projected_path_lists"] == 0


def test_the_projection_kinds_are_derived_from_the_retrievers_that_read_the_key(pack):
    """The constant is a claim about `src/retrievers/`, checked against it.

    `_PROJECTION_KINDS` uses pack-vocabulary names (`databricks_uc`) while the read lives in
    a module reached through the internal retriever type. A second backend supporting the key
    without appearing here fails silently: the declaration would be enforced and the check
    would never ask about it.

    The last assertion catches a `_merge_endpoint` translation: `kibana` is no pack kind,
    so a read in that module cannot be expressed by this constant.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        import log_retrieval
    finally:
        sys.path.pop(0)
    import inspect

    read = 'config.get("projection")'
    readers = {
        name
        for name, cls in log_retrieval._RETRIEVER_TYPES.items()
        if read in inspect.getsource(sys.modules[cls.__module__])
    }
    assert readers, "the read moved or was reworded — this pin cannot see it any more"
    expected = {
        kind for kind, typ in log_retrieval._KIND_TO_TYPE.items() if typ in readers
    }
    assert pv._PROJECTION_KINDS == expected
    assert readers <= set(log_retrieval._KIND_TO_TYPE.values()), (
        "a retriever reads `projection` for a type no pack `kind` maps to, so no value of "
        "this constant can express it"
    )


def test_every_installed_pack_stays_within_the_projection_finding_budget():
    """The false-positive budget for this check, pinned as a number. A jump is a resolver drift.

    The two it allows are real, both on one pack's `platform_mode` block: the ruleset's own
    comment recorded that the projection widening had been decided and the catalog never got
    the leaves.
    """
    for root in [PACKS / n for n in installed_packs()] + [TEMPLATE]:
        result = pv.validate_pack(root)
        hits = find(result, "field-path-outside-projection")
        assert len(hits) <= 2, [(h["path"], h["line"], h["detail"]) for h in hits]
        assert all(h["severity"] == "warning" for h in hits)
        assert result["ok"], "this check may never block a save"


# ------------------------------------------- two projection entries, one name in the row
#
# The inverse of the check above, and the worse half. That one finds a leaf the row will
# not carry. This one finds a leaf the row carries another entry's value for: a row is
# built by zipping column names against values, so a name carried twice keeps the last.
#
# The un-aliased entry is at risk because a backend names a projected nested leaf after
# its last segment only. The generator aliases every dotted path to its flattened form,
# but a prompt instruction is not a mechanism.
#
# Two severities by whether anything can rescue it: colliding entries both pinning the same
# alias are an error (no generator makes a row carry both); a collision reached by falling
# back to the last segment is a warning (one AS removes it).


def test_two_projection_entries_ending_in_the_same_segment_are_reported(pack):
    """The finding, on the shape that produced it live: two leaves of one relation.

    Both nested paths are correct, distinct, and end in the same word — which is the whole
    defect, because the segment is the name.
    """
    _project(pack, ["outer.example_id", "inner.example_id"])
    result = pv.validate_pack(pack)
    hits = find(result, "projection-name-collision")
    assert len(hits) == 1, "one finding per colliding NAME, not per entry"
    assert (
        hits[0]["severity"] == "warning"
    ), "the flattening convention may still name them apart"
    assert hits[0]["line"] > 0, "the operator needs a cursor"
    assert "example_source" in hits[0]["message"], "name the source"
    assert "'example_id'" in hits[0]["message"], "name the row key they land on"
    assert "outer.example_id" in hits[0]["detail"]
    assert "inner.example_id" in hits[0]["detail"]
    assert result["ok"], "a conditional collision may not block a save"


def test_aliasing_the_entries_apart_silences_it_and_the_check_still_ran(pack):
    """The positive control, and it pins the SILENCE as a check rather than a skip.

    The remedy is the convention stated in the declaration instead of hoped for in a prompt,
    so aliasing to the flattened path changes nothing where the generator already obeys.
    """
    _project(
        pack,
        [
            "outer.example_id AS outer_example_id",
            "inner.example_id AS inner_example_id",
        ],
    )
    result = pv.validate_pack(pack)
    assert "projection-name-collision" not in codes(result)
    assert result["counts"]["projections_checked"] >= 1, "the projection was checked"


def test_two_entries_pinning_the_SAME_alias_are_an_error(pack):
    """The unconditional half: nothing downstream can make a row carry both.

    A warning here would be the wrong reading — there is no generator behaviour left to hope
    for, so the pack works while lying, which is this module's own bar for an error.
    """
    _project(pack, ["outer.example_id AS example_id", "inner.example_id AS example_id"])
    hits = find(pv.validate_pack(pack), "projection-name-collision")
    assert len(hits) == 1
    assert hits[0]["severity"] == "error"
    assert "cannot carry both" in hits[0]["message"]


def test_one_pinned_alias_beside_an_unaliased_sibling_is_still_only_a_warning(pack):
    """The mixed case, which is the boundary between the two severities.

    The bare entry may yet be aliased apart by the generator, so the collision is conditional
    even though its partner's name is settled — and a fix that read "any alias present" as
    proof would report a hard failure on a pack that has none.
    """
    _project(pack, ["outer.example_id AS example_id", "inner.example_id"])
    hits = find(pv.validate_pack(pack), "projection-name-collision")
    assert len(hits) == 1
    assert hits[0]["severity"] == "warning"


def test_a_repeated_identical_entry_is_not_a_collision(pack):
    """A duplicated line is a redundancy, not a lost leaf — the same value under one name.

    Worth a test because the arithmetic is a count of names and the naive version reports it,
    which would put a finding on the one shape here that costs nothing.
    """
    _project(pack, ["outer.example_id", "outer.example_id"])
    assert "projection-name-collision" not in codes(pv.validate_pack(pack))


def test_an_unaliased_expression_cannot_collide_with_anything(pack):
    """An entry that names nothing collides with nothing, because nothing can read it.

    The shape is the discriminating one rather than any expression: an expression reached
    THROUGH a member accessor ends in the same word as the bare path beside it, so a reading
    that fell back to an entry's last segment would report the two as one column. It must not,
    and not because the collision would be harmless — the expression is unreadable under ANY
    name, which is the other reader's finding (`projection_names` contributes nothing for it,
    so a condition naming that leaf is reported as outside the projection). Reporting it here
    too would put a second code on it whose remedy — alias the entries apart — is the remedy
    for a different defect.
    """
    _project(
        pack,
        [
            "filter(example_elements, e -> e.example_code).example_id",
            "inner.example_id",
        ],
    )
    result = pv.validate_pack(pack)
    assert "projection-name-collision" not in codes(result)
    assert result["counts"]["projections_checked"] >= 1, "the projection was checked"


def test_a_backend_that_ignores_the_projection_is_never_reported_for_a_collision(pack):
    """The key is INERT on every kind but one, so a collision there is a collision in nothing.

    Same projection that fires above; only the endpoint kind changes. The count is asserted
    too, because a silence that came from not looking must not read as a silence that looked.
    """
    path = pack / "source_catalog.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["sources"][0]["endpoints"] = {
        "kind": "elasticsearch",
        "cluster": "example_cluster",
        "indices": ["example-*"],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    _project(pack, ["outer.example_id", "inner.example_id"])
    result = pv.validate_pack(pack)
    assert "projection-name-collision" not in codes(result)
    assert result["counts"]["projections_checked"] == 0


def test_no_installed_pack_pins_one_alias_twice():
    """The unconditional half over the real packs, and the check's own floor.

    The warnings are NOT pinned to a number: at the time this landed one pack had six of them,
    every one real and each one a decision about a source whose queries are validated — so a
    budget here would be a record of that backlog rather than a check. What is asserted is the
    half no reading can excuse, plus that the check reached a projection at all, since zero
    findings is also what a check that stopped running reports.
    """
    checked = 0
    for root in [PACKS / n for n in installed_packs()] + [TEMPLATE]:
        result = pv.validate_pack(root)
        hard = [
            h
            for h in find(result, "projection-name-collision")
            if h["severity"] == "error"
        ]
        assert not hard, [(h["path"], h["line"], h["detail"]) for h in hard]
        checked += result["counts"].get("projections_checked", 0)
    assert checked >= 1, "no projection was checked anywhere — the check is not running"


# ------------------------------------------------ entity bindings vs the same inventory
#
# The same defect one stage earlier: a stale binding leaves the query unscoped for that
# entity type; other parties' rows come back.
#
# Granularity is the binding, not the path. A binding is a candidate list deliberately
# spelled several ways across backends, so per-path reporting would flag correct authoring:
# over the installed pack, 20 absent paths sit in six bindings and the other fourteen have a
# live sibling. All six were confirmed dead against the cluster's own `_field_caps`.


def _bind(pack, bindings):
    """Rewrite the template source's `entity_bindings` (whole-key, as an author would)."""
    path = pack / "source_catalog.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["sources"][0]["entity_bindings"] = bindings
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def test_a_binding_whose_every_path_is_absent_from_the_schema_is_a_warning(pack):
    _bind(pack, {"example_entity": ["no_such_column", "gone.old.name"]})
    hits = find(pv.validate_pack(pack), "binding-paths-not-in-schema")
    assert len(hits) == 1
    assert hits[0]["severity"] == "warning", "a sampled inventory cannot prove absence"
    assert hits[0]["line"] > 0, "the operator needs a cursor"
    assert "example_source" in hits[0]["message"], "name the source"
    assert "example_entity" in hits[0]["message"], "name the binding, not just the source"
    assert "no_such_column" in hits[0]["detail"] and "gone.old.name" in hits[0]["detail"]
    assert (
        "unscoped" in hits[0]["message"]
    ), "state the consequence: the query is not narrowed, it is not an `unknown`"


def test_the_templates_own_binding_is_accepted(pack):
    """The positive control. Without it the check could pass by never resolving anything."""
    result = pv.validate_pack(pack)
    assert "binding-paths-not-in-schema" not in codes(result)
    assert result["counts"]["entity_bindings_checked"] == 1


def test_one_live_spelling_among_dead_ones_is_not_a_binding_finding(pack):
    """A binding is a candidate LIST, deliberately spelled several ways across backends.

    This is the difference between six findings and twenty on the installed pack.
    """
    _bind(pack, {"example_entity": ["gone.old.name", "example_id", "also_gone"]})
    assert "binding-paths-not-in-schema" not in codes(pv.validate_pack(pack))


def test_a_per_form_binding_is_checked_per_form(pack):
    """`value_forms` binds as a MAP of form -> paths, and each form is its own channel.

    A form routed to a dead column has its values DROPPED rather than unioned onto the
    sibling form's column, so a live `login` cannot vouch for a dead `badge`.
    """
    _bind(pack, {"example_entity": {"live": ["example_id"], "dead": ["no_such_column"]}})
    hits = find(pv.validate_pack(pack), "binding-paths-not-in-schema")
    assert len(hits) == 1
    assert "example_entity.dead" in hits[0]["message"]


def test_a_source_with_no_schema_doc_has_no_binding_reported(pack):
    """Absence of an inventory is not evidence about a column — the same silence as a path.

    A pack documents the sources it chose to, so binding the first entity on a new source
    must not be a warning.
    """
    (pack / "schemas" / "example_table.yaml").unlink()
    _bind(pack, {"example_entity": ["no_such_column"]})
    result = pv.validate_pack(pack)
    assert "binding-paths-not-in-schema" not in codes(result)
    assert (
        result["counts"]["entity_bindings_checked"] == 0
    ), "a pack that documents nothing must not read as a pack that passed"


def test_a_binding_onto_a_decode_target_is_exempt(pack):
    """A decoded table's paths are absent from every discovered inventory BY CONSTRUCTION.

    `encoded_fields` writes its expansion at a pack-declared `into:`, so the loudest false
    positive here is the pack's own decode target — the same exemption the path check makes.
    """
    path = pack / "source_catalog.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["sources"][0]["entity_bindings"] = {"example_entity": ["parts.left"]}
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    # The control: with nothing declaring the decode, that path IS dead — so the silence
    # below is the exemption doing work and not the resolver accepting `parts.left` anyway.
    assert "binding-paths-not-in-schema" in codes(pv.validate_pack(pack))
    doc["sources"][0]["encoded_fields"] = [
        {"field": "example_id", "into": "parts", "keys": ["left", "right"]}
    ]
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    assert "binding-paths-not-in-schema" not in codes(pv.validate_pack(pack))


def test_two_dead_bindings_on_different_sources_get_different_lines(pack):
    """A per-source finding that always cites the FIRST `entity_bindings:` in the file reads
    as one finding repeated. The catalog declares the same key on every source, so the line
    has to be resolved from the entry that owns it."""
    path = pack / "source_catalog.yaml"
    doc = yaml.safe_load(path.read_text())
    first = doc["sources"][0]
    second = dict(first, name="second_source")
    first["entity_bindings"] = {"example_entity": ["no_such_column"]}
    second["entity_bindings"] = {"example_entity": ["also_no_such_column"]}
    doc["sources"] = [first, second]
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    # Both sources read the same table, so both are documented by the one inventory.
    schema = pack / "schemas" / "example_table.yaml"
    sdoc = yaml.safe_load(schema.read_text())
    (pack / "schemas" / "second_source.yaml").write_text(
        yaml.safe_dump(dict(sdoc, source="second_source"), sort_keys=False)
    )
    hits = find(pv.validate_pack(pack), "binding-paths-not-in-schema")
    assert len(hits) == 2
    assert len({h["line"] for h in hits}) == 2, [h["line"] for h in hits]
    assert all(h["line"] > 0 for h in hits)


def test_every_installed_pack_and_the_template_stay_at_or_below_the_binding_budget():
    """Zero budget: a finding here means an author bound a column their inventory does not record."""
    for root in [PACKS / n for n in installed_packs()] + [TEMPLATE]:
        result = pv.validate_pack(root)
        hits = find(result, "binding-paths-not-in-schema")
        assert not hits, [(h["path"], h["line"], h["detail"]) for h in hits]
        assert all(h["severity"] == "warning" for h in hits)
        assert result["ok"], "this check may never block a save"


# ------------------------------------------------------- one entity, two files, one agreement
#
# A value form is DECLARED in the glossary and HONOURED in the catalog, so a disagreement
# between the two is invisible in either file read alone — which is what this check is for and
# why it is the one that reads both. The two shapes are both legitimate: a MAP of form to
# fields routes each value to its own form's column and drops a value whose form the map does
# not name, a flat LIST filters every form's value onto every column in it. What is not
# legitimate is a pack that does both for one type, because then the routing is real on some
# sources and absent on others, and on the others one form's value lands on the other form's
# column — a valid predicate matching nothing, reported as a source that had nothing to say.


def _two_sources(pack, first_bindings, second_bindings):
    """The template's source plus a copy, each binding `example_entity` its own way."""
    path = pack / "source_catalog.yaml"
    doc = yaml.safe_load(path.read_text())
    first = dict(doc["sources"][0], entity_bindings=first_bindings)
    second = dict(first, name="second_source", entity_bindings=second_bindings)
    doc["sources"] = [first, second]
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _declare_forms(pack, *names):
    path = pack / "entity_glossary.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["entities"][0]["value_forms"] = [
        {"name": n, "pattern": f"^{n.upper()}[0-9]+$"} for n in names
    ]
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def test_a_pack_declaring_no_value_forms_is_untouched_and_says_so_in_the_counts(pack):
    """The template declares none, which is every pack that has not opted in.

    The counts are asserted as a PAIR because a check that reports nothing must not read like
    a check that found nothing: zero form-declaring entities is the reason for the silence, and
    without the number in the result there is no way to tell that from a check that ran and
    passed over a pack full of forms.
    """
    result = pv.validate_pack(pack)
    assert "value-form-binding-mixed" not in codes(result)
    assert "unknown-value-form-binding" not in codes(result)
    assert result["counts"]["value_form_entities"] == 0
    assert result["counts"]["value_form_bindings_checked"] == 0


def test_sources_that_all_agree_say_nothing_whichever_shape_they_agree_on(pack):
    """Both silences, and the ALL-FLAT one is the load-bearing half.

    A type may declare forms and be bound flat everywhere: forms still drive classification,
    `stem` widening and `co_identity`, none of which reads a binding. Reporting that would put
    a permanent warning on every pack that uses forms for what they were first added for.
    """
    _declare_forms(pack, "badge", "login")
    for shape in (
        ({"example_entity": ["example_id"]}, {"example_entity": ["example_id"]}),
        (
            {"example_entity": {"badge": ["example_id"]}},
            {"example_entity": {"login": ["example_id"]}},
        ),
    ):
        _two_sources(pack, *shape)
        result = pv.validate_pack(pack)
        assert "value-form-binding-mixed" not in codes(result), shape
        assert result["counts"]["value_form_bindings_checked"] == 2


def test_a_flat_binding_beside_a_per_form_one_is_a_warning_naming_the_flat_source(pack):
    """The finding is the FLAT source, not the per-form one, and a warning rather than an error.

    Flat can be the honest answer — one column really does hold both forms, which is the shape
    of an alert source that stores the whole incident text in a single field — and whether it
    does is a claim about the DATA that only a per-source measurement settles. So this reports
    the list of sources somebody still has to measure and never blocks a save.
    """
    _declare_forms(pack, "badge", "login")
    _two_sources(
        pack,
        {"example_entity": {"badge": ["example_id"], "login": ["example_id"]}},
        {"example_entity": ["example_id"]},
    )
    result = pv.validate_pack(pack)
    hits = find(result, "value-form-binding-mixed")
    assert len(hits) == 1
    assert hits[0]["severity"] == "warning"
    assert "second_source" in hits[0]["message"]
    assert hits[0]["line"] > 0
    assert "badge, login" in hits[0]["detail"] or "badge" in hits[0]["detail"]
    assert result["ok"]


def test_each_flat_source_is_its_own_finding_at_its_own_line(pack):
    """Per-source, because the fix is per-source: each one needs its own column measured.

    And the line has to come from the entry that owns the binding — the catalog declares
    `entity_bindings` on every source, so a finding that always cites the file's first hit
    reads as one finding repeated rather than two distinct ones.
    """
    _declare_forms(pack, "badge", "login")
    path = pack / "source_catalog.yaml"
    doc = yaml.safe_load(path.read_text())
    base = doc["sources"][0]
    doc["sources"] = [
        dict(base, entity_bindings={"example_entity": {"badge": ["example_id"]}}),
        dict(base, name="flat_one", entity_bindings={"example_entity": ["example_id"]}),
        dict(base, name="flat_two", entity_bindings={"example_entity": ["example_id"]}),
    ]
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    hits = find(pv.validate_pack(pack), "value-form-binding-mixed")
    assert len(hits) == 2
    assert len({h["line"] for h in hits}) == 2, [h["line"] for h in hits]
    assert all(h["line"] > 0 for h in hits)


def test_a_binding_under_an_undeclared_form_name_is_an_ERROR(pack):
    """An ERROR by this module's rule: mechanically decidable and provably inert.

    `classify_value_form` can only ever return a name the glossary declares, and
    `_form_scoped_fields` looks the classified name up in the binding map — so fields under a
    name that is not declared are never filtered on, and every value of the form they were
    meant to catch is dropped instead. It is exactly the typo that converting a flat binding
    into a per-form one invites, and its symptom is a source that quietly answers about nobody.
    """
    _declare_forms(pack, "badge", "login")
    _two_sources(
        pack,
        {"example_entity": {"badge": ["example_id"], "lognin": ["example_id"]}},
        {"example_entity": {"badge": ["example_id"]}},
    )
    result = pv.validate_pack(pack)
    hits = find(result, "unknown-value-form-binding")
    assert len(hits) == 1
    assert hits[0]["severity"] == "error"
    assert "lognin" in hits[0]["message"]
    assert "badge" in hits[0]["detail"] and "login" in hits[0]["detail"]
    assert result["ok"] is False
    # And it is not reported twice: the source binds per form, so it is not ALSO mixed.
    assert "value-form-binding-mixed" not in codes(result)


def test_no_installed_pack_binds_an_undeclared_form():
    """The error half is a budget of zero; the warning half is a per-pack number, so it is
    pinned in that pack's own test file where the measurement behind each entry lives."""
    for root in [PACKS / n for n in installed_packs()] + [TEMPLATE]:
        hits = find(pv.validate_pack(root), "unknown-value-form-binding")
        assert not hits, [(h["path"], h["line"], h["message"]) for h in hits]


# -------------------------------------------------------------------------- info, not warning


def test_a_playbook_only_use_case_is_info_and_never_a_warning(pack):
    """A playbook-only use case is the ordinary shape for a documented-but-not-automated one.

    As a warning it would be a permanent false alarm on most use-case directories.
    Reported at INFO; the severity is what this test pins.
    """
    for name in ("documented_only", "also_documented_only"):
        book = pack / "use_cases" / name / "playbooks"
        book.mkdir(parents=True)
        (book / f"{name}.md").write_text(f"# {name}\n\nProse, no ruleset.\n")
    result = pv.validate_pack(pack)
    hits = find(result, "playbook-only-use-case")
    assert len(hits) == 2, [h["path"] for h in hits]
    assert {h["severity"] for h in hits} == {"info"}
    assert result["warnings"] < len(hits)
    assert result["ok"], "info must never affect the verdict"


# --------------------------------------------- which procedure adjudicates, scored on titles


def _second_spec(pack, title):
    """A second playbook carrying a `correlation:` block, so the selector has a field."""
    book = pack / "use_cases" / "example_use_case" / "playbooks"
    (book / "rival_playbook.md").write_text(
        "---\n"
        "playbook_id: PB-RIVAL-001\n"
        f"title: {title}\n"
        "correlation:\n"
        "  keys: [example_entity]\n"
        "  time_window: within:24h\n"
        "---\n\nProse.\n"
    )


def test_a_function_word_owned_by_ONE_spec_title_is_reported(pack):
    """The scorer weights by inverse spec frequency and matches with `in` — a SUBSTRING test.

    Together those make a stopword no other title uses the strongest possible evidence for
    the procedure that owns it, while it fires inside unrelated words. The template's own
    title carries `this`, so the second spec is all this test has to add.
    """
    _second_spec(pack, "Rival Procedure — Unrelated Fraud Pattern")
    hits = find(pv.validate_pack(pack), "spec-title-weak-discriminator")
    assert len(hits) == 1, [(h["path"], h["detail"]) for h in hits]
    assert hits[0]["severity"] == "warning"
    assert "this" in hits[0]["detail"].split(", ")
    assert hits[0]["path"].endswith("example_playbook.md")


def test_a_function_word_SHARED_by_BOTH_titles_is_not_reported(pack):
    """The discriminating half: frequency, not the word.

    A stopword every spec declares already scores ZERO in the engine — it is not evidence
    for any of them — so reporting it would be noise on the one shape that is harmless. This
    is what separates the check from a stopword lint.
    """
    # The rival title shares `this` and introduces no weak token of its own — worth stating,
    # because the first draft ended `... Pattern Is` and that trailing `is` is itself a hit.
    _second_spec(pack, "Rival Procedure — What This Other Pattern Involves")
    hits = find(pv.validate_pack(pack), "spec-title-weak-discriminator")
    assert not hits, [(h["path"], h["detail"]) for h in hits]


def test_a_LONE_correlation_spec_reports_nothing_and_still_counts_itself(pack):
    """With one spec the score is moot — the engine returns the only candidate regardless.

    So a single-procedure pack must be silent however its title is worded. The count is
    asserted beside the silence for the reason every count in this module exists: a pack that
    stopped being checked must not read like a pack that passed.
    """
    result = pv.validate_pack(pack)
    assert result["counts"]["correlation_specs"] == 1
    assert not find(result, "spec-title-weak-discriminator")


def test_a_playbook_with_no_correlation_block_is_not_a_spec(pack):
    """`correlation:` is what makes a playbook a candidate — a prose-only one is not scored."""
    book = pack / "use_cases" / "example_use_case" / "playbooks"
    (book / "prose_only.md").write_text(
        "---\nplaybook_id: PB-PROSE-001\ntitle: A Note About This\n---\n\nProse.\n"
    )
    result = pv.validate_pack(pack)
    assert result["counts"]["correlation_specs"] == 1
    assert not find(result, "spec-title-weak-discriminator")


def test_the_prose_token_separator_is_derived_from_the_engine():
    """Derived, never restated — a second answer to "what is a token" drifts silently.

    Asserted as a non-empty pattern that really splits prose rather than by equality with a
    literal, which would be the same restatement one layer up. An unlocatable region yields
    `""` and turns the check off, which is this module's standing rule for every derived
    vocabulary and the reason the emptiness is worth pinning against.
    """
    pattern = pv.prose_split_pattern()
    assert pattern
    assert pv._spec_tokens("Two Words — Here", ["a_key"]) == {
        "two",
        "words",
        "here",
        "a",
        "key",
    }


@pytest.mark.parametrize("name", installed_packs())
def test_no_installed_pack_owns_a_weak_discriminator(name):
    """Zero findings, plus the count to confirm the check ran."""
    result = pv.validate_pack(PACKS / name)
    assert result["counts"]["correlation_specs"] >= 1
    hits = find(result, "spec-title-weak-discriminator")
    assert not hits, [(h["path"], h["detail"]) for h in hits]


def test_ok_tracks_errors_only(pack):
    """Warnings are what the operator is TOLD; they never block a save."""
    edit_rules(pack, lambda spec: spec.update({"conditions": []}))
    result = pv.validate_pack(pack)
    assert result["warnings"] >= 1
    assert result["ok"] is True
    assert result["errors"] == 0


def test_an_unrecognised_follow_up_window_is_an_error(pack):
    """An unreadable window mode is not an error at runtime — it is `inherit`.

    `_follow_up_window` resolves anything it does not recognise to the inherited window,
    deliberately, because the conservative direction is the narrow scan. The cost is that a
    typo'd `onwrads` produces a pass that runs, succeeds, returns rows and answers a question
    about the incident's own window instead of the one the pack declared. There is no artifact
    in which those two differ, so the lint is the only place it can be caught.
    """
    edit_rules(pack, _follow_up(window="onwrads"))
    hit = find(pv.validate_pack(pack), "follow-up-unknown-window")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "onwrads" in hit[0]["message"]


def test_a_lookback_window_with_no_depth_is_an_error(pack):
    """A depth is REQUIRED, and its absence degrades to the narrowest reading.

    `lookback` alone parses as far as the mode and no further, so the pass falls back to
    `inherit` — the antecedent question asked over the episode's window, which is the exact
    0-row reading the mode was added to fix. Reported separately from an unknown mode because
    the fix is different: one is a misspelling, this one is an omission.
    """
    edit_rules(pack, _follow_up(window="lookback"))
    hit = find(pv.validate_pack(pack), "follow-up-lookback-no-depth")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    # Not ALSO reported as an unrecognised mode: the author spelled a real mode.
    assert "follow-up-unknown-window" not in codes(pv.validate_pack(pack))


def _voting_indicators(spec, n=3, **spec_extra):
    """`n` non-decisive fraud indicators on the template ruleset, and nothing else."""
    spec["conditions"] = [
        {
            "id": f"ind{i}",
            "kind": "field_flag",
            "source": "record",
            "label": f"Indicator {i} fired",
            "polarity": "fraud_indicator",
            "decisive": False,
            "report_group": "validation",
        }
        for i in range(n)
    ]
    spec.pop("indicator_threshold", None)
    spec.update(spec_extra)


def test_a_ruleset_with_voting_indicators_and_no_threshold_is_an_error(pack):
    """How many indicators amount to an accusation is the procedure's call.

    The engine defaulted it to 2, so a ruleset that declared indicators and forgot the key
    reached its fraud label on any two of them under a rule nothing in the pack states. The
    engine now refuses to weigh them at all, which is safe and silently weaker — so the
    omission has to be caught here, where the author can still supply the number.
    """
    edit_rules(pack, lambda spec: _voting_indicators(spec, 3))
    hit = find(pv.validate_pack(pack), "indicator-threshold-undeclared")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "3 corroborating fraud indicator(s)" in hit[0]["message"]


def test_a_threshold_of_zero_is_reported_as_its_own_mistake(pack):
    """`len([]) >= 0` is true, so zero accuses every subject with no indicator firing at all.

    Reported under its own code because the remedy differs from an omission: the author here
    supplied a number and it is the one number that inverts the check.
    """
    edit_rules(pack, lambda spec: _voting_indicators(spec, 3, indicator_threshold=0))
    result = pv.validate_pack(pack)
    hit = find(result, "indicator-threshold-vacuous")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    # Not ALSO reported as undeclared: the key is there.
    assert "indicator-threshold-undeclared" not in codes(result)


def test_a_threshold_no_indicator_count_can_reach_is_a_warning(pack):
    """Three indicators and a threshold of four: the corroborated path is dead code.

    A WARNING and not an error, because a ruleset may legitimately reach its verdict by a
    decisive condition and keep the indicators as reported context — but it is never what an
    author meant to write, and it renders exactly like a procedure whose corroboration simply
    never fired.
    """
    edit_rules(pack, lambda spec: _voting_indicators(spec, 3, indicator_threshold=4))
    hit = find(pv.validate_pack(pack), "indicator-threshold-unreachable")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"
    assert "needs 4" in hit[0]["message"] and "only 3" in hit[0]["message"]


def test_a_ruleset_with_no_voting_indicators_is_not_asked_for_a_threshold(pack):
    """The silence that keeps the check honest, in both of its two forms.

    A ruleset with no indicators is not governed by the number, and a DECISIVE indicator needs
    no threshold — its own FAIL is the finding. Demanding one in either case would be a lint
    nobody could satisfy truthfully, which is how a check earns an exemption list.
    """
    for mutate in (
        lambda spec: one_condition(spec, polarity="exclusion"),
        lambda spec: one_condition(
            spec, polarity="fraud_indicator", decisive=True
        ),
    ):
        edit_rules(pack, mutate)
        assert not [
            c for c in codes(pv.validate_pack(pack)) if c.startswith("indicator-threshold-")
        ]


def test_no_installed_pack_leaves_its_voting_rule_to_the_engine():
    """All rulesets across the installed packs declare their threshold explicitly."""
    for name in installed_packs():
        result = pv.validate_pack(PACKS / name)
        assert not [
            d
            for d in result["diagnostics"]
            if d["code"].startswith("indicator-threshold-")
        ], name


@pytest.mark.parametrize(
    "kind,extra",
    [
        ("distinct_count", {"field": "handler"}),
        ("velocity_count", {"subject_field": "locator", "actor_field": "handler"}),
        (
            "time_gap",
            {
                "start": {"source": "record", "field": "opened"},
                "end": {"source": "record", "field": "closed"},
            },
        ),
    ],
)
def test_a_counting_condition_with_no_bound_is_an_error(pack, kind, extra):
    """The bound IS the finding for these kinds, so its absence is not a detail.

    The engine used to substitute one (3, 1, 0 or an hour, by kind and by branch) and print
    the invented figure as the procedure's own finding. Those defaults are gone and the
    condition now reports `unknown` — which is honest and also inert, and an inert decisive
    check is an INSUFFICIENT DATA verdict nobody declared. There is no artifact in which a
    silently-defaulted bound and a declared one differ, so the lint is where it gets caught.
    """
    edit_rules(pack, lambda spec: one_condition(spec, kind=kind, **extra))
    hit = find(pv.validate_pack(pack), "condition-bound-undeclared")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert kind in hit[0]["message"]
    assert "no `max` key" in hit[0]["detail"]


@pytest.mark.parametrize(
    "kind,value,ok",
    [
        ("distinct_count", 0, True),  # the commonest declared value; never "unset"
        ("distinct_count", 1, True),
        ("distinct_count", "2", True),  # a YAML scalar the engine coerces
        ("distinct_count", True, False),  # a bool is not a count
        ("distinct_count", "many", False),
        ("time_gap", "90m", True),
        ("time_gap", "4h", True),
        ("time_gap", "2d", True),
        ("time_gap", "1 week", False),  # parses to nothing → used to mean one hour
        ("time_gap", "PT48H", False),
        ("time_gap", 24, False),  # an interval is not a bare number
    ],
)
def test_the_accepted_bound_grammar_is_pinned(pack, kind, value, ok):
    """The positive control, without which the check above is satisfied by rejecting everything.

    It also pins which spellings an author may use, because the two grammars are not the same
    and neither is guessable: a count takes a whole number, an interval takes the engine's own
    `<N>h`/`<N>d`/`<N>m` window — and `24` alone, the obvious thing to write, is not a window.
    """
    extra = (
        {"field": "handler"}
        if kind == "distinct_count"
        else {
            "start": {"source": "record", "field": "opened"},
            "end": {"source": "record", "field": "closed"},
        }
    )
    edit_rules(pack, lambda spec: one_condition(spec, kind=kind, max=value, **extra))
    hit = find(pv.validate_pack(pack), "condition-bound-undeclared")
    assert bool(hit) is (not ok), f"{kind} max={value!r}"


def test_the_bounded_kind_list_is_the_engine_s_own_and_cannot_drift():
    """The map is hand-kept because a GRAMMAR cannot be derived — the membership can.

    A key→kinds map that nothing checks is how a declaration comes to be reported as fine on a
    kind that ignores it, or a new kind ships with the invented default nobody noticed was
    still there. Every bounded kind's refusal branch reports `_NO_BOUND_EXPECTED`, so that
    constant is the engine's own statement of which kinds need a bound, and the two must agree
    exactly. What the map adds beyond membership is which grammar each kind takes, and that is a
    real authoring fact no derivation could supply.

    Which is why a kind refusing for another reason gets its own constant: `event_order` needs a
    relation, not a magnitude, and borrowing this one would both enrol it in the map above and
    print `no bound declared` at a reader whose pack declares every bound it has.
    """
    assert set(pv._BOUNDED_KINDS) == pv.kinds_reading("_NO_BOUND_EXPECTED")
    assert set(pv._BOUNDED_KINDS.values()) == {"integer", "window", "numeric"}
    assert pv.kinds_reading("_NO_ORDER_EXPECTED") == {"event_order"}


def test_no_installed_pack_leaves_a_counting_bound_undeclared():
    """The shipped packs are the check's control: it must fire on none of them."""
    for name in installed_packs():
        result = pv.validate_pack(PACKS / name)
        assert not find(result, "condition-bound-undeclared"), name


def test_the_three_real_window_modes_are_accepted(pack):
    """The positive control both checks need, and it pins the accepted vocabulary.

    Without it the two errors above are satisfied by a lint that rejects every window, which
    would make `follow_up_passes` undeclarable and be discovered only by an author.
    """
    for mode in ("inherit", "onwards", "lookback:365d", "lookback=30d"):
        edit_rules(pack, _follow_up(window=mode))
        assert not [
            c for c in codes(pv.validate_pack(pack)) if c.startswith("follow-up-")
        ], mode


# ------------------------------------------------- inbound cross-procedure signals


def _signal(**over):
    """A WELL-FORMED `entry_signals` declaration on the template ruleset, one key changed.

    The baseline is what makes the rest of this block meaningful: every test below asserts one
    code fires, and a lint that rejected every declaration would satisfy all of them at once
    while making the key undeclarable. `test_a_well_formed_entry_signal_is_silent` is the
    control, and each mutation starts from the same bytes.

    `when.where` is part of well-formedness and not decoration, which is the redesign showing up
    in a fixture: what licenses an automatic escalation is now a declaration plus rung-1 PASS on
    this run's rows rather than a counted base rate, so the SELECTOR is the safety surface and a
    signal with none fires on the source having been retrieved. The baseline therefore names the
    rows it means — `test_a_signal_with_no_row_selector_is_a_WARNING` is the other direction.
    """
    entry = {
        "id": "example_inbound_signal",
        "direction": "antecedent",
        "opens_with": {"entity": "example_entity"},
        "when": {
            "source": "record",
            "min_rows": 3,
            "where": [{"field": "example_field", "any_of": ["EXAMPLE"]}],
        },
        "window": "lookback:30d",
        "strength": 0.6,
        "base_rate": {"fires_on": 3, "of": 27, "measured": "2026-08-19"},
    }
    drop = over.pop("_drop", ())
    entry.update(over)
    for key in drop:
        entry.pop(key, None)

    def mutate(spec):
        spec["entry_signals"] = [entry]

    return mutate


def signal_codes(pack):
    return [c for c in codes(pv.validate_pack(pack)) if c.startswith("entry-signal-")]


def test_a_well_formed_entry_signal_is_silent(pack):
    edit_rules(pack, _signal())
    assert signal_codes(pack) == []
    assert pv.validate_pack(pack)["counts"]["entry_signals"] == 1


def test_a_pack_declaring_no_entry_signals_is_untouched_and_says_so_in_the_counts(pack):
    """The whole mechanism must be inert for the pack every author starts from.

    Same guarantee `follow_up_passes` gives a single-pass pack, and the count is what tells a
    reader "this pack declares none" apart from "the check did not run".
    """
    result = pv.validate_pack(pack)
    assert [c for c in codes(result) if c.startswith("entry-signal-")] == []
    assert result["counts"]["entry_signals"] == 0


def test_an_entry_signal_opening_on_another_subject_is_an_ERROR(pack):
    """R2 made mechanical: a leg is opened by a SUBJECT value and by nothing else.

    A ruleset iterates its `subject_entity`, and every condition resolves against it — so a
    signal promising to open this procedure on some other type declares a referral the
    procedure cannot act on. Nothing fails at run time: the pivot is reported absent while the
    signal fires, which reads as evidence that was checked.
    """
    edit_rules(pack, _signal(opens_with={"entity": "other_entity"}))
    hit = find(pv.validate_pack(pack), "entry-signal-subject-mismatch")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "other_entity" in hit[0]["message"]
    assert "example_entity" in hit[0]["message"]


def test_an_entry_signal_with_no_pivot_is_an_ERROR(pack):
    edit_rules(pack, _signal(_drop=("opens_with",)))
    hit = find(pv.validate_pack(pack), "entry-signal-no-subject")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


def test_an_entry_signal_with_no_id_is_an_ERROR(pack):
    """The loader DROPS it, and a report has nothing to cite."""
    edit_rules(pack, _signal(_drop=("id",)))
    hit = find(pv.validate_pack(pack), "entry-signal-no-id")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


def test_an_entry_signal_reading_an_unknown_source_is_an_ERROR(pack):
    """A signal asked of a name no run produces can never fire, and never says so."""
    edit_rules(pack, _signal(when={"source": "no_such_source", "min_rows": 3}))
    hit = find(pv.validate_pack(pack), "entry-signal-unknown-source")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "no_such_source" in hit[0]["message"]


def test_an_entry_signal_may_name_a_PHYSICAL_source(pack):
    """The one thing that separates this from a hard dependency.

    An inbound signal is read on ANOTHER procedure's rows, so it may name a catalog source
    this ruleset's own `sources:` map never mentions — and it must NOT thereby become a
    dependency retrieved on every run. Both spellings resolve; neither is a finding.
    """
    edit_rules(
        pack,
        _signal(
            when={
                "source": "example_source",
                "min_rows": 3,
                # The baseline's selector, carried over: `when` is replaced whole here, and a
                # `where`-less one would report the broad-selector warning — a code about the
                # SELECTOR standing in for the claim about the source NAME.
                "where": [{"field": "example_field", "any_of": ["EXAMPLE"]}],
            }
        ),
    )
    assert signal_codes(pack) == []


def test_an_entry_signal_source_is_not_a_hard_dependency(pack):
    """The `_walk_source_refs` trap, asserted from the outside.

    The dependency walk reports a logical name a ruleset READS but does not declare. Left
    unexcluded, it converts every inbound signal source into a source retrieved on every
    incident of this procedure — the exact defect `required_source_not_queried` exists to
    report, manufactured by the lint.
    """
    edit_rules(pack, _signal(when={"source": "example_source", "min_rows": 3}))
    result = pv.validate_pack(pack)
    assert not [
        d
        for d in result["diagnostics"]
        if "example_source" in d["message"] and d["severity"] == "error"
    ], [d["code"] for d in result["diagnostics"]]


@pytest.mark.parametrize("direction", ["antecedent", "consequent"])
def test_both_causal_directions_are_accepted(direction, pack):
    edit_rules(pack, _signal(direction=direction))
    assert signal_codes(pack) == []


def test_an_unrecognised_direction_is_an_ERROR(pack):
    """Without a direction there is no causal reading and a referral gets no window.

    "This may have CAUSED the incident" and "the incident may have caused this" are different
    referrals with opposite windows, and the engine cannot infer which from prose.
    """
    edit_rules(pack, _signal(direction="folows"))
    hit = find(pv.validate_pack(pack), "entry-signal-bad-direction")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


def test_a_missing_direction_is_an_ERROR(pack):
    edit_rules(pack, _signal(_drop=("direction",)))
    assert find(pv.validate_pack(pack), "entry-signal-bad-direction")


def test_the_direction_vocabulary_is_the_engine_s_own_and_cannot_drift():
    """Derived from `src/links.py`, not listed here — the same rule as the condition kinds.

    A literal copy in the lint passes every test until somebody adds a third direction, at
    which point the lint rejects a declaration the engine honours.
    """
    from src.links import LINK_DIRECTIONS

    assert pv.link_directions() == tuple(LINK_DIRECTIONS)
    assert pv.link_directions()


_WINDOWS = ["inherit", "onwards", "lookback:365d", "lookback=30d"]


@pytest.mark.parametrize("mode", _WINDOWS)
def test_the_entry_signal_window_vocabulary_is_the_follow_up_one(mode, pack):
    """Same three modes, deliberately: the axis is the same axis."""
    edit_rules(pack, _signal(window=mode))
    assert signal_codes(pack) == []


def test_an_unreadable_entry_signal_window_is_an_ERROR(pack):
    edit_rules(pack, _signal(window="lookbak:30d"))
    hit = find(pv.validate_pack(pack), "entry-signal-bad-window")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


def test_an_absent_window_inherits_and_is_silent(pack):
    edit_rules(pack, _signal(_drop=("window",)))
    assert signal_codes(pack) == []


def test_an_unmeasured_base_rate_is_a_WARNING_not_an_error(pack):
    """The declaration still works; what a reader loses is the ability to discount a firing.

    A warning rather than an error because the alternative is that no pair can be authored
    before its corpus exists, and an author who cannot satisfy a check learns to ignore it.
    """
    edit_rules(pack, _signal(_drop=("base_rate",)))
    hit = find(pv.validate_pack(pack), "entry-signal-unmeasured")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"


def test_an_explicit_stub_base_rate_is_still_reported_as_unmeasured(pack):
    """`kind: stub` is HONEST and it is not a measurement.

    Reading it as measured would promote "nobody has counted this" to "this is rare" — the one
    reading the base rate exists to prevent.
    """
    edit_rules(pack, _signal(base_rate={"kind": "stub"}))
    hit = find(pv.validate_pack(pack), "entry-signal-unmeasured")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"


def test_a_base_rate_measured_over_a_tiny_corpus_is_unmeasured(pack):
    """A rate is a claim about a population, and 1 of 3 is not one.

    The floor is on the CORPUS, never the fire count: 1-of-3 and 33-of-100 declare the same
    rate and only the second is evidence.
    """
    edit_rules(pack, _signal(base_rate={"fires_on": 1, "of": 3}))
    hit = find(pv.validate_pack(pack), "entry-signal-unmeasured")
    assert len(hit) == 1
    assert "3" in hit[0]["message"]


def test_the_three_unmeasured_states_share_a_code_and_differ_in_the_SENTENCE(pack):
    """One code and one severity, three sentences — because the remedies differ in TIMING.

    All three are unmeasured and none may be trusted, so splitting the code would let a pack
    silence one class of them. But an omission is an authoring slip fixable now, a declared
    stub is a scheduled measurement waiting on a corpus that does not exist yet, and a thin
    corpus is a number that was counted and is not yet worth reading. The old wording said
    "declares no measured `base_rate`" about all three, which is factually WRONG about a
    declaration that says `kind: stub` — and a reader who catches a warning being wrong about
    the pack stops reading the warnings.
    """
    seen = {}
    for state, mutate in (
        ("omitted", _signal(_drop=("base_rate",))),
        ("stub", _signal(base_rate={"kind": "stub"})),
        ("thin", _signal(base_rate={"fires_on": 1, "of": 3})),
    ):
        edit_rules(pack, mutate)
        hit = find(pv.validate_pack(pack), "entry-signal-unmeasured")
        assert len(hit) == 1, state
        assert hit[0]["severity"] == "warning", state
        seen[state] = (hit[0]["message"], hit[0]["hint"])

    # Each names its OWN state, so the sentence is actionable without opening the file.
    assert "no `base_rate` at all" in seen["omitted"][0]
    assert "stub" in seen["stub"][0]
    assert "stub" not in seen["omitted"][0]
    assert "3" in seen["thin"][0] and "stub" not in seen["thin"][0]

    # And three distinct remedies: declare one / replace the stub / count more runs. Three
    # identical hints would make the differentiated message decoration.
    assert len({h for _, h in seen.values()}) == 3
    assert len({m for m, _ in seen.values()}) == 3


def test_auto_probe_without_a_base_rate_is_NOT_an_error(pack):
    """The rule this check used to enforce now contradicts the design, so it is gone.

    What licenses an escalation is a well-formed declaration plus **rung-1 PASS** — the target
    procedure's own scope gate, re-evaluated against THIS run's retrieved rows. A historical
    per-pair firing rate is an optional additive term in the `semi_auto` confidence score and
    gates nothing, so refusing `auto_probe: true` for want of one refuses a decision the engine
    now takes per incident, on evidence, for a statistic it does not read.

    The warning stays (`entry-signal-unmeasured`, asserted above): "we have not counted this"
    is still worth saying, and demoting a measurement to optional is not the same as deleting it.
    An ERROR is the claim that the pack does not work, and this one does.
    """
    edit_rules(pack, _signal(auto_probe=True, _drop=("base_rate",)))
    result = pv.validate_pack(pack)
    assert find(result, "entry-signal-auto-probe-unmeasured") == []
    errors = [
        d
        for d in result["diagnostics"]
        if d["code"].startswith("entry-signal-") and d["severity"] == "error"
    ]
    assert errors == [], (
        "an unmeasured pair is refused by some OTHER error, so the demotion is undone by a "
        f"neighbouring rule: {[d['code'] for d in errors]}"
    )
    # And the demotion is not a deletion: the same declaration is still REPORTED, one severity
    # down. Silence here would be the opposite defect — an uncounted pair reading as a counted one.
    assert [d["severity"] for d in find(result, "entry-signal-unmeasured")] == [
        "warning"
    ]


def test_the_strict_entry_signal_errors_are_UNCHANGED_by_that_demotion(pack):
    """The two claims that are still mechanically decidable, on the same relaxed signal.

    Asserted together with `auto_probe: true` set, because the demotion above is a change to when
    an ERROR fires and the risk is that it took the neighbouring errors with it. Both of these are
    about a declaration that CANNOT work: a signal opening on another procedure's subject names a
    pivot the ruleset never iterates, and a signal reading a source no run produces can never fire
    and never says so. Neither is a judgement about how often anything happens, which is why
    neither moved.
    """
    edit_rules(
        pack,
        _signal(
            auto_probe=True,
            _drop=("base_rate",),
            opens_with={"entity": "other_entity"},
        ),
    )
    hit = find(pv.validate_pack(pack), "entry-signal-subject-mismatch")
    assert len(hit) == 1 and hit[0]["severity"] == "error"

    edit_rules(
        pack,
        _signal(
            auto_probe=True,
            _drop=("base_rate",),
            when={"source": "no_such_source", "min_rows": 3},
        ),
    )
    hit = find(pv.validate_pack(pack), "entry-signal-unknown-source")
    assert len(hit) == 1 and hit[0]["severity"] == "error"


def test_a_signal_with_no_row_selector_is_a_WARNING(pack):
    """The check that REPLACES the base-rate bar, and the reason it is a warning.

    With eligibility resting on the declaration, authoring the selector is the safety surface — and
    the broadest possible selector is the one shape that switches it off: no `when.where` means any
    row of the source, so the signal fires on the source having been RETRIEVED rather than on
    anything in it. Not an error, because it is legitimate (a source that exists only when the
    shape does is a real declaration) and the engine cannot tell the two apart from the outside.

    Loudest where it costs something: the message names `auto_probe` when the same signal also
    spends a query, or a reader has to hold two declarations in mind to see the consequence.
    """
    edit_rules(pack, _signal(when={"source": "record", "min_rows": 3}))
    hit = find(pv.validate_pack(pack), "entry-signal-broad-selector")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"
    assert "auto_probe" not in hit[0]["message"]

    edit_rules(pack, _signal(auto_probe=True, when={"source": "record", "min_rows": 3}))
    louder = find(pv.validate_pack(pack), "entry-signal-broad-selector")
    assert len(louder) == 1
    assert "auto_probe" in louder[0]["message"], (
        "the same sentence is printed whether or not the signal spends a query, so the reader "
        "cannot tell an advisory over-broad signal from one that pays for its breadth"
    )


def test_a_min_rows_of_zero_is_a_WARNING(pack):
    """The engine floors it at 1, so the declaration reads as "one row of an ordinary source"."""
    edit_rules(
        pack,
        _signal(
            when={
                "source": "record",
                "min_rows": 0,
                "where": [{"field": "example_field", "any_of": ["EXAMPLE"]}],
            }
        ),
    )
    hit = find(pv.validate_pack(pack), "entry-signal-bad-min-rows")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"


def test_a_strength_outside_the_unit_range_is_a_WARNING(pack):
    edit_rules(pack, _signal(strength=6))
    hit = find(pv.validate_pack(pack), "entry-signal-bad-strength")
    assert len(hit) == 1
    assert hit[0]["severity"] == "warning"


def test_entry_signals_declared_as_a_mapping_is_an_ERROR(pack):
    """Not a list means every inbound signal is ignored, silently and wholesale."""

    def mutate(spec):
        spec["entry_signals"] = {"id": "oops"}

    edit_rules(pack, mutate)
    hit = find(pv.validate_pack(pack), "entry-signal-not-a-list")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


def test_no_installed_pack_declares_an_unusable_entry_signal():
    """Whatever the shipped packs declare, none of it may be inert."""
    for name in installed_packs():
        result = pv.validate_pack(PACKS / name)
        assert not [
            d
            for d in result["diagnostics"]
            if d["code"].startswith("entry-signal-") and d["severity"] == "error"
        ], name


def test_the_link_graph_reconciliation_is_info_only_on_every_installed_pack():
    """A prose edge the data cannot execute is a JUDGEMENT, so it may never gate a save.

    Which of the two graphs is wrong is not mechanically decidable: a newly authored signal may
    discover a real edge the prose never claimed, and a prose edge may be a real analytic
    relationship this pack's data cannot run. Reporting either as an error would make the
    disagreement unsavable and force the author to delete the true half.
    """
    for name in installed_packs():
        result = pv.validate_pack(PACKS / name)
        for d in result["diagnostics"]:
            if d["code"].startswith("related-playbook-"):
                assert d["severity"] == "info", (name, d["code"])


def test_a_related_playbook_naming_nothing_is_reported_apart_from_a_binding_gap():
    """Two facts, two codes: a leg nothing can bind is not a leg naming nothing.

    The shipped router declares `related_playbooks: [ALL above]` — prose inside a list of ids,
    which reaches the engine as one unresolvable playbook id. Its fix is to enumerate the ids;
    a binding gap's fix is to bind a column or accept the limitation. One code for both sends
    an author to the wrong file.
    """
    hits = [
        d
        for name in installed_packs()
        for d in pv.validate_pack(PACKS / name)["diagnostics"]
        if d["code"] == "related-playbook-unknown"
    ]
    for d in hits:
        assert "names no playbook" in d["message"]


# ------------------------------------------------- per-pair escalation mode


def _escalation(_gated=True, **block):
    """A `link_escalation` block on the template ruleset, with its `entry_signals` intact.

    The signal comes along because the two declarations are only meaningful together: the mode says
    whether this procedure may be entered automatically, and the signal is what a sibling run reads
    to notice the pair at all. A test that set the mode alone would be asserting against a pair
    nothing could ever propose.

    `_gated` adds a `gate: scope` condition, and it defaults to on for the same reason the signal
    comes along: rung 1 — this ruleset's own applicability test, re-evaluated against the sibling
    run's rows — is what licenses an escalation, so an escalating mode on a gateless ruleset is
    inert on every incident forever. Turn it off to assert exactly that
    (`test_an_escalating_mode_with_NO_SCOPE_GATE_is_an_ERROR`).
    """

    def mutate(spec):
        _signal()(spec)
        spec["link_escalation"] = block
        conditions = spec.get("conditions")
        if not _gated or not isinstance(conditions, list) or not conditions:
            return
        if not any(
            isinstance(c, dict) and str(c.get("gate", "") or "").lower() == "scope"
            for c in conditions
        ):
            conditions[0] = {**conditions[0], "gate": "scope"}

    return mutate


def escalation_codes(pack):
    return [
        c for c in codes(pv.validate_pack(pack)) if c.startswith("link-escalation-")
    ]


@pytest.mark.parametrize("mode", link_escalation.LINK_MODES)
def test_every_mode_this_engine_KNOWS_is_declarable_on_a_measured_pair(mode, pack):
    """The positive control, and it is parametrised for a reason.

    Each error below asserts one code fires, and a lint that rejected every mode would satisfy
    all of them at once while making the key undeclarable — the same trap the window vocabulary
    has. Parametrising over the engine's own tuple means a mode added to `LINK_MODES` without a
    validator that accepts it fails here rather than in a pack.
    """
    edit_rules(pack, _escalation(mode=mode))
    assert escalation_codes(pack) == []


def test_a_pack_declaring_no_escalation_mode_is_untouched(pack):
    """Inert for the pack every author starts from, like every other additive key."""
    edit_rules(pack, _signal())
    assert escalation_codes(pack) == []


def test_an_unspellable_mode_is_an_ERROR(pack):
    """It is DROPPED at run time, so the declaration reads exactly like declaring nothing.

    Mechanically decidable against a closed vocabulary and provably inert, which is what makes
    it an error rather than a warning.
    """
    edit_rules(pack, _escalation(mode="Auto-Escalate"))
    hit = find(pv.validate_pack(pack), "link-escalation-bad-mode")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "Auto-Escalate" in hit[0]["message"]
    # The hint has to carry the vocabulary; the message names what was written, and an author
    # who knew the right spelling would not have written the wrong one.
    for mode in link_escalation.LINK_MODES:
        assert mode in hit[0]["hint"]


def test_an_escalating_mode_with_NO_SCOPE_GATE_is_an_ERROR(pack):
    """The bar, at authoring time — and the successor to the base-rate rule, on the same defect.

    Rung-1 PASS is the licence: this ruleset's OWN applicability gate, re-evaluated against the
    sibling run's rows. A ruleset carrying no `gate: scope` condition has no rung 1 to pass, so
    the mode is not "refused this time" — it is unreachable on every incident forever. Same
    failure as the old unmeasured-pair error and the same reason it is an error rather than a
    warning: what ships is a setting that silently does nothing, and an operator who reads the
    mode off the card, sees no escalation and concludes the mechanism is broken.
    """
    edit_rules(pack, _escalation(_gated=False, mode="auto"))
    hit = find(pv.validate_pack(pack), "link-escalation-no-scope-gate")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "gate: scope" in hit[0]["message"]
    # And the hint has to offer both ways out, because declaring the mode `planned` is a
    # legitimate answer — the gate is what an escalation needs, not what a link needs.
    assert link_escalation.MANUAL_LINK_MODE in hit[0]["hint"]


def test_planned_on_a_GATELESS_ruleset_is_SILENT(pack):
    """The other half of the bar, and the half a first draft gets wrong.

    `planned` is not a refusal — it is the composed referral a human executes with one click, and
    it costs nothing, so it needs no applicability gate to be reachable. Reporting it here would
    fire on every gateless ruleset in every pack, which is the "37 findings against 2" shape that
    gets a diagnostic switched off.
    """
    edit_rules(pack, _escalation(_gated=False, mode="planned"))
    assert escalation_codes(pack) == []


def test_an_escalating_mode_on_an_UNMEASURED_pair_is_SILENT(pack):
    """The redesign's own assertion, and the one that would fail if the old rule crept back.

    A base rate is a HISTORICAL statistic, and this lane no longer spends anything on one: it may
    only ADD to the semi_auto confidence score. So an escalating mode over a stubbed pair is a
    complete, working declaration — every escalation it licenses is decided by rung 1 on this
    run's own rows — and the validator says nothing about it. The `entry-signal-unmeasured`
    warning still fires on the signal, which is the part that stayed: not counting a pair is
    worth reporting, it just no longer forbids anything.
    """

    def mutate(spec):
        _escalation(mode="auto")(spec)
        spec["entry_signals"][0]["base_rate"] = {"kind": "stub"}

    edit_rules(pack, mutate)
    result = pv.validate_pack(pack)
    assert escalation_codes(pack) == []
    assert [d["severity"] for d in find(result, "entry-signal-unmeasured")] == [
        "warning"
    ]


def test_a_per_source_override_names_a_ruleset_or_it_is_an_ERROR(pack):
    """The likeliest of the three, because it is the one string the author cannot self-check.

    A `from:` key is ANOTHER procedure's name, so a rename elsewhere in the pack leaves this one
    matching nothing — and an override that cannot match is not neutral: the pair silently takes
    the general mode, which is the opposite of what a per-source override is written to say.
    """
    edit_rules(pack, _escalation(**{"from": {"no_such_ruleset": "auto"}}))
    hit = find(pv.validate_pack(pack), "link-escalation-unknown-source")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "no_such_ruleset" in hit[0]["message"]


def test_a_per_source_override_may_name_a_ruleset_declared_ANYWHERE_in_the_pack(pack):
    """Which is why the check runs after the ruleset sweep and not inside it.

    Read per ruleset as it loads, every forward reference — a sibling further down the same file,
    or in a `use_cases/` file read later — would be reported as unknown. That failure mode is
    worse than the one the check exists for: it makes the correct declaration unsavable.
    """
    edit_rules(pack, _escalation(**{"from": {"example_use_case": "auto"}}))
    assert escalation_codes(pack) == []


def test_a_per_source_override_is_HELD_TO_THE_GATE_like_the_general_mode(pack):
    """The licence is about the RULESET, so no layer of the declaration escapes it.

    An override reached by a different key is still an escalating mode, and rung 1 is the same
    rung whichever key asked for it — a check that only read `mode:` would leave the narrower,
    more specific declaration as the one way around the gate. Asserted on a block declaring no
    general `mode:` at all, so the finding can only have come from the `from:` entry.
    """
    edit_rules(
        pack, _escalation(_gated=False, **{"from": {"example_use_case": "auto"}})
    )
    hit = find(pv.validate_pack(pack), "link-escalation-no-scope-gate")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


def test_an_escalation_block_of_the_wrong_SHAPE_is_an_ERROR(pack):
    """Both shapes, and separately: each is ignored wholesale in a different scope."""
    edit_rules(pack, _escalation())

    def not_a_mapping(spec):
        spec["link_escalation"] = "auto"

    edit_rules(pack, not_a_mapping)
    hit = find(pv.validate_pack(pack), "link-escalation-not-a-mapping")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"

    def bad_from(spec):
        spec["link_escalation"] = {"mode": "planned", "from": ["example_use_case"]}

    edit_rules(pack, bad_from)
    hit = find(pv.validate_pack(pack), "link-escalation-bad-from")
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


def test_the_base_rate_READING_has_ONE_home(pack):
    """ "Measured" must mean one thing, even now that it licenses nothing.

    The two readers are the validator (which REPORTS an uncounted pair) and the confidence score
    (which may ADD to it for a counted one), asked at different times about different inputs, so
    nothing brings them together except reading the same function. Two copies could disagree
    about which pairs count as measured — and a pack would then be warned about a rate its own
    confidence score was already crediting, or credited for one it was warned about.
    """
    assert pv._MIN_BASE_RATE_CORPUS == link_escalation.MIN_BASE_RATE_CORPUS
    for rate in (
        None,
        {},
        {"kind": "stub"},
        {"fires_on": 1, "of": 3},
        {"fires_on": 3, "of": 27},
        {"of": 27},
    ):
        assert pv._base_rate_state(rate or {}) == link_escalation.base_rate_measured(
            rate
        )


def test_no_installed_pack_declares_an_unusable_escalation_mode():
    """Whatever the shipped packs declare, none of it may be inert or unlicensed."""
    for name in installed_packs():
        result = pv.validate_pack(PACKS / name)
        assert not [
            d
            for d in result["diagnostics"]
            if d["code"].startswith("link-escalation-") and d["severity"] == "error"
        ], name


# ------------------------------------------------------------ the documented CLI


def test_the_documented_module_invocation_actually_RUNS(capsys):
    """`python -m src.knowledge.pack_validate <dir>` is documented in four places and had no
    `__main__` block, so it printed nothing and exited 0 — which reads exactly like a pack
    that passed. The output has to name the pack and state the three tallies, because
    "no findings" and "the checker did not run" are the same sentence otherwise.
    """
    rc = pv.main([str(TEMPLATE)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "0 error(s)" in out
    assert "checked: " in out and "conditions=" in out


def test_the_CLI_exits_NON_zero_on_an_ERROR_and_zero_on_a_WARNING(tmp_path, capsys):
    """Mirrors `ok`, and the warning half is the load-bearing one: several diagnostics are
    claims about prose that no mechanical check can settle, and a lint that fails on those
    is a lint somebody switches off.
    """
    pack = tmp_path / "cli_pack"
    shutil.copytree(TEMPLATE, pack)
    clean = pv.main([str(pack)])
    assert clean == 0

    rules = list(pack.rglob("rules*.yaml")) + list(pack.rglob("rulesets.yaml"))
    assert rules, "the template must ship a ruleset for this test to mean anything"
    doc = yaml.safe_load(rules[0].read_text())
    key = next(iter(doc))
    doc[key].setdefault("conditions", []).append(
        {"id": "cli_broken", "kind": "no_such_condition_kind_exists", "label": "x"}
    )
    rules[0].write_text(yaml.safe_dump(doc, sort_keys=False))

    result = pv.validate_pack(pack)
    assert result["errors"] >= 1
    assert pv.main([str(pack)]) == 1
    assert "FAILS" in capsys.readouterr().out


def test_the_CLI_reports_a_BAD_PATH_rather_than_reading_as_a_CLEAN_pack(capsys):
    """A mistyped directory is the failure most likely to be mistaken for a pass, so it is
    its own exit code and its own line on stderr — never a silent 0 with no diagnostics.
    """
    assert pv.main([]) == 2
    assert "usage:" in capsys.readouterr().err
    assert pv.main([str(REPO_ROOT / "no_such_pack_dir_anywhere")]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_the_CLI_takes_SEVERAL_packs_and_the_worst_outcome_wins(tmp_path, capsys):
    """Both packs are always rendered — a first bad pack must not hide a second one's
    findings — and the exit code is the worst of them, or a CI step passes on the tail.
    """
    good = tmp_path / "good"
    shutil.copytree(TEMPLATE, good)
    assert pv.main([str(good), str(REPO_ROOT / "still_no_such_dir")]) == 2
    captured = capsys.readouterr()
    assert "0 error(s)" in captured.out
    assert "not a directory" in captured.err


# ------------------------------------------ numeric_compare and the three composites


def test_the_derived_vocabularies_match_the_engine():
    """Every vocabulary here is read off `correlation.py`, so a widened operator set or a
    renamed constant must not become a spurious error on a pack that is right."""
    kinds = pv.condition_kinds()
    for kind in ("numeric_compare", "all_of", "any_of", "none_of", "event_order"):
        assert kind in kinds, kind
    # A bounded kind the evaluator does not dispatch would demand a bound nothing reads.
    assert set(pv._BOUNDED_KINDS) <= kinds
    assert set(pv._BOUND_KEYS) <= set(pv._BOUNDED_KINDS)
    assert set(pv.composite_kinds()) <= kinds
    assert pv.compare_operators() and "==" in pv.compare_operators()
    assert pv.max_composite_depth() >= 1
    # `_ORDER_RELATIONS` is a dict, which `_engine_literals`' tuple regex cannot see: read
    # through the wrong deriver it comes back empty and turns its own check off in silence.
    assert "after" in pv.order_relations()
    assert set(pv.order_quantifiers()) == {"every", "any"}


def test_a_numeric_compare_bound_may_be_fractional(pack):
    """The grammar exists because a ratio bound of 0.25 coerced to an integer becomes 0 — a
    bound every value satisfies. The integer kinds keep their stricter grammar."""
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="numeric_compare",
            aggregate="ratio",
            operator="<=",
            bound=0.25,
        ),
    )
    assert "condition-bound-undeclared" not in codes(pv.validate_pack(pack))

    edit_rules(
        pack,
        lambda spec: one_condition(
            spec, kind="numeric_compare", aggregate="count", operator=">", bound="not a number"
        ),
    )
    found = find(pv.validate_pack(pack), "condition-bound-undeclared")
    assert found and found[0]["severity"] == "error"
    assert "bound:" in found[0]["detail"]


def test_a_numeric_compare_declaration_the_engine_cannot_read_is_an_error(pack):
    """All three are silent at run time: the condition reports `unknown`, and a decisive one
    reads as INSUFFICIENT DATA with nothing naming the declaration that caused it."""
    for over, code in (
        ({"aggregate": "p95"}, "numeric-compare-aggregate"),
        ({"operator": "!="}, "numeric-compare-operator"),
    ):
        base = {
            "kind": "numeric_compare",
            "aggregate": "count",
            "operator": ">",
            "bound": 1,
        }
        base.update(over)
        edit_rules(pack, lambda spec, b=base: one_condition(spec, **b))
        found = find(pv.validate_pack(pack), code)
        assert found and found[0]["severity"] == "error", over
        assert "available:" in found[0]["detail"], over


def test_exclude_subject_on_a_numeric_compare_is_an_error(pack):
    """The engine deliberately does not read it there, and a silently-ignored key leaves the
    subject's own rows in the aggregate while the declaration says they were removed."""
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="numeric_compare",
            aggregate="distinct",
            field="actor",
            operator="<=",
            bound=1,
            exclude_subject=True,
        ),
    )
    found = find(pv.validate_pack(pack), "numeric-compare-exclude-subject")
    assert found and found[0]["severity"] == "error"
    assert "distinct_count" in found[0]["hint"]


def test_group_by_under_an_equality_is_reported(pack):
    """The evaluator compares the LARGEST group, which answers an upper bound and not an
    equality, so the check can only report `unknown`."""
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="numeric_compare",
            aggregate="count",
            group_by="actor",
            operator="==",
            bound=2,
        ),
    )
    found = find(pv.validate_pack(pack), "numeric-compare-group-equality")
    assert found and found[0]["severity"] == "warning"


def test_a_modal_aggregate_with_no_field_is_an_error(pack):
    """`mode` asks which VALUE is most frequent, so with no field the evaluator has nothing to
    rank and reports `unknown` — the same silence a source returning nothing produces. `count`
    is the one aggregate that reads the rows themselves, so it must not be reported."""
    for agg in ("mode", "mode_share"):
        edit_rules(
            pack,
            lambda spec, a=agg: one_condition(
                spec, kind="numeric_compare", aggregate=a, operator=">=", bound=2
            ),
        )
        found = find(pv.validate_pack(pack), "numeric-compare-modal-field")
        assert found and found[0]["severity"] == "error", agg
        assert "field" in found[0]["hint"], agg

    # The two silences: a field is a field wherever it is declared, and `count` needs none.
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="numeric_compare",
            aggregate="mode",
            operator=">=",
            bound=2,
            fallbacks=[{"source": "other", "field": "actor"}],
        ),
    )
    assert "numeric-compare-modal-field" not in codes(pv.validate_pack(pack))

    edit_rules(
        pack,
        lambda spec: one_condition(
            spec, kind="numeric_compare", aggregate="count", operator=">=", bound=2
        ),
    )
    assert "numeric-compare-modal-field" not in codes(pv.validate_pack(pack))


def _baseline_cond(baseline, **over):
    """A `numeric_compare` whose bound is a multiple of a computed population."""
    cond = {
        "kind": "numeric_compare",
        "aggregate": "count",
        "operator": ">=",
        "bound": 3,
        "baseline": baseline,
    }
    cond.update(over)
    return cond


def test_a_baseline_the_engine_cannot_compute_is_an_error(pack):
    """Every one of these leaves the condition `unknown`, so a decisive check reads as
    INSUFFICIENT DATA with nothing naming the declaration that emptied it."""
    for baseline, code in (
        ({"aggregate": "median", "field": "amount"}, "baseline-source"),
        ({"source": "peers", "aggregate": "p95", "field": "amount"}, "baseline-aggregate"),
        ({"source": "peers", "aggregate": "median"}, "baseline-field"),
        ({"source": "peers", "aggregate": "ratio"}, "baseline-ratio-unfiltered"),
        (
            {"source": "peers", "aggregate": "median", "field": "amount", "per": "actor"},
            "baseline-per-aggregate",
        ),
    ):
        edit_rules(
            pack,
            lambda spec, b=baseline: one_condition(spec, **_baseline_cond(b)),
        )
        found = find(pv.validate_pack(pack), code)
        assert found and found[0]["severity"] == "error", baseline

    # The positive control: a complete declaration reports nothing. Without it every
    # assertion above is satisfied by a check that reports on any baseline at all.
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            **_baseline_cond(
                {
                    "source": "peers",
                    "aggregate": "median",
                    "field": "amount",
                    "per": "actor",
                    "per_aggregate": "count",
                }
            ),
        ),
    )
    assert not [c for c in codes(pv.validate_pack(pack)) if c.startswith("baseline-")]


def test_a_baseline_the_engine_never_reads_is_an_error(pack):
    """The other direction, and the worse one: the declaration is not unreadable but
    unreached, so `bound` is compared as an ABSOLUTE threshold — a `3` meant as three times
    the cohort tested as "at least three" — and the condition decides, wrongly."""
    good = {"source": "peers", "aggregate": "median", "field": "amount"}
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="distinct_count",
            source="record",
            field="actor",
            max=1,
            baseline=good,
        ),
    )
    found = find(pv.validate_pack(pack), "baseline-unread-kind")
    assert found and found[0]["severity"] == "error"
    assert "numeric_compare" in found[0]["detail"]

    # A scalar or a list is ignored by the `isinstance` guard the evaluator reads it through,
    # which lands on the same absolute-threshold comparison.
    edit_rules(pack, lambda spec: one_condition(spec, **_baseline_cond(["peers"])))
    found = find(pv.validate_pack(pack), "baseline-shape")
    assert found and found[0]["severity"] == "error"
    assert "absolute" in found[0]["message"]


def _order_cond(**over):
    """An `event_order` asserting that the closing event followed the opening one."""
    cond = {
        "kind": "event_order",
        "relation": "after",
        "quantifier": "every",
        "start": {"source": "record", "field": "opened_at"},
        "end": {"source": "record", "field": "closed_at"},
    }
    cond.update(over)
    return cond


def test_an_event_order_the_engine_cannot_read_is_an_error(pack):
    """The declaration IS the check here: there is no relation to fall back to and no default
    quantifier, so each of these returns `unknown` and a decisive one reads as INSUFFICIENT
    DATA — an ordering claim nobody verified, reported as missing data."""
    for over, code in (
        ({"relation": None}, "event-order-relation"),
        ({"relation": "follows"}, "event-order-relation"),
        ({"quantifier": None}, "event-order-quantifier"),
        ({"quantifier": "most"}, "event-order-quantifier"),
        ({"tolerance": "soon"}, "event-order-tolerance"),
        ({"start": {"field": "opened_at"}}, "event-order-side"),
        ({"end": {"source": "record"}}, "event-order-side"),
        ({"end": "closed_at"}, "event-order-side"),
    ):
        edit_rules(
            pack, lambda spec, o=over: one_condition(spec, **_order_cond(**o))
        )
        found = find(pv.validate_pack(pack), code)
        assert found and found[0]["severity"] == "error", over

    # The positive control, with the one optional key declared: without it every assertion
    # above is satisfied by a check that reports on any `event_order` at all.
    edit_rules(pack, lambda spec: one_condition(spec, **_order_cond(tolerance="90m")))
    assert not [c for c in codes(pv.validate_pack(pack)) if c.startswith("event-order-")]


def test_a_time_gap_bound_left_on_an_event_order_is_an_error(pack):
    """The opposite failure, and the one a conversion produces: the condition runs, and
    correctly, while the magnitude bound the pack still declares is read by nobody — so an
    interval check reads as intact after becoming an ordering check."""
    edit_rules(pack, lambda spec: one_condition(spec, **_order_cond(max="4h")))
    found = find(pv.validate_pack(pack), "event-order-max")
    assert found and found[0]["severity"] == "error"
    assert "time_gap" in found[0]["hint"]


def _composite_cond(children, **over):
    cond = {
        "id": "combo",
        "kind": "all_of",
        "label": "The combination holds",
        "report_group": "validation",
        "fail_detail": "a leg did not hold",
        "children": children,
    }
    cond.update(over)
    return cond


def _leaf(cid="leg", **over):
    leaf = {"id": cid, "kind": "field_flag", "source": "record", "field": "flag"}
    leaf.update(over)
    return leaf


def test_a_well_formed_composite_is_silent(pack):
    """The positive control. A false positive here teaches the operator to scroll past."""
    edit_rules(
        pack,
        lambda spec: spec.__setitem__(
            "conditions", [_composite_cond([_leaf("a"), _leaf("b")])]
        ),
    )
    result = pv.validate_pack(pack)
    assert result["errors"] == 0, [
        (d["code"], d["message"]) for d in result["diagnostics"] if d["severity"] == "error"
    ]
    for code in (
        "composite-children",
        "composite-child-weighting",
        "composite-too-deep",
        "unknown-condition-kind",
    ):
        assert code not in codes(result), code


def test_a_composite_needs_two_children(pack):
    """One child is not a combination and reads as a typo for that child's own kind; none can
    only ever report `unknown`."""
    for children in ([], [_leaf("only")]):
        edit_rules(
            pack,
            lambda spec, c=children: spec.__setitem__(
                "conditions", [_composite_cond(c)]
            ),
        )
        found = find(pv.validate_pack(pack), "composite-children")
        assert found and found[0]["severity"] == "error", children


def test_a_composites_child_may_not_carry_the_parents_weight(pack):
    """`mk()` reads every weighting key off the outer dict only, so a child declaring one is
    inert — and an inert `decisive` is the difference between a conclusive verdict and an
    advisory one."""
    for key, value in (
        ("decisive", True),
        ("polarity", "fraud_indicator"),
        ("report_group", "validation"),
        ("subject_scope", "element"),
        ("row_match", [{"from_entity": "actor", "field": "actor"}]),
    ):
        edit_rules(
            pack,
            lambda spec, k=key, v=value: spec.__setitem__(
                "conditions", [_composite_cond([_leaf("a", **{k: v}), _leaf("b")])]
            ),
        )
        found = find(pv.validate_pack(pack), "composite-child-weighting")
        assert found and found[0]["severity"] == "error", key
        assert key in found[0]["message"], key


def test_a_composites_child_is_checked_by_the_same_rules_as_a_top_level_condition(pack):
    """Children are conditions, so an unrecognised kind or an unreadable bound inside one is
    the same defect it would be at the top level — and just as invisible in the report."""
    edit_rules(
        pack,
        lambda spec: spec.__setitem__(
            "conditions",
            [
                _composite_cond(
                    [
                        _leaf("typo", kind="distinct_counts"),
                        _leaf(
                            "unbounded",
                            kind="numeric_compare",
                            aggregate="count",
                            operator=">",
                        ),
                    ]
                )
            ],
        ),
    )
    result = pv.validate_pack(pack)
    assert "unknown-condition-kind" in codes(result, "error")
    assert "condition-bound-undeclared" in codes(result, "error")


def test_a_composite_nested_past_the_engines_bound_is_an_error(pack):
    """The engine stops and reports `unknown` there, so the pack must hear about it at
    authoring time rather than as a decisive check that never fired."""
    deep = _composite_cond([_leaf("a"), _leaf("b")], id="innermost")
    for i in range(pv.max_composite_depth() + 1):
        deep = _composite_cond([deep, _leaf(f"sib{i}")], id=f"n{i}")
    edit_rules(pack, lambda spec: spec.__setitem__("conditions", [deep]))
    found = find(pv.validate_pack(pack), "composite-too-deep")
    assert found and found[0]["severity"] == "error"


# ------------------------------------------------------ pack-declared equivalence forms


#: A well-formed vocabulary: one transitive projection and one pairwise relation. `folded`
#: shortens (`keep`) so it declares a floor; `near` does not, which is also the control for
#: the `min_length` warning being conditional.
_GOOD_FORMS = {
    "folded": {
        "description": "case and punctuation are not part of the identity",
        "project": [{"case": "upper"}, {"keep": "alnum"}],
        "min_length": 4,
    },
    "near": {
        "project": [{"case": "upper"}],
        "compare": {"edit_distance": 1},
        "linkage": "single",
    },
}


def write_forms(pack, doc):
    """Declare `shared/equivalence_forms.yaml` on the fixture pack.

    Takes any document, not just a mapping: the two ways a form disappears at load are a file
    that is not a map and an entry that is not one, and both have to be expressible here.
    """
    path = pack / "shared" / "equivalence_forms.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _eq_cond(**over):
    """A `value_equivalence` asking whether any two values collide under a declared form."""
    cond = {
        "kind": "value_equivalence",
        "source": "record",
        "field": "actor",
        "form": "folded",
        "operator": "<=",
        "bound": 1,
    }
    cond.update(over)
    return cond


def test_a_form_the_loader_drops_is_reported_by_the_name_that_was_lost(pack):
    """The loader skips a malformed form with a log line, so the only trace at run time is a
    condition refused for naming a form no pack declares — the reason in a log nobody reads."""
    write_forms(pack, ["folded"])
    found = find(pv.validate_pack(pack), "equivalence-forms-not-a-mapping")
    assert found and found[0]["severity"] == "error"
    assert "list" in found[0]["message"]

    write_forms(pack, {"folded": "upper", "near": _GOOD_FORMS["near"]})
    found = find(pv.validate_pack(pack), "equivalence-form-not-a-mapping")
    assert found and found[0]["severity"] == "error"
    assert "folded" in found[0]["message"]
    assert found[0]["line"] >= 1


def test_a_form_key_the_engine_does_not_read_is_a_warning(pack):
    """A misspelled `compare` leaves a projection-only form, which is a DIFFERENT relation
    that still evaluates and still reports the form's own name."""
    write_forms(pack, {"folded": dict(_GOOD_FORMS["folded"], compair={"exact": True})})
    found = find(pv.validate_pack(pack), "equivalence-form-unknown-key")
    assert found and found[0]["severity"] == "warning"
    assert "compair" in found[0]["message"]
    assert "compare" in found[0]["detail"]

    # `description` is read by nobody either and must stay silent: a form documents itself.
    write_forms(pack, _GOOD_FORMS)
    assert "equivalence-form-unknown-key" not in codes(pv.validate_pack(pack))


def test_a_pipeline_step_the_engine_cannot_apply_is_an_error(pack):
    """Every one of these is a relation OTHER than the one declared, and the condition still
    evaluates, still names the form, and reports a number arrived at some other way."""
    for over, code in (
        ({"project": {"case": "upper"}}, "equivalence-form-project-shape"),
        ({"project": [{"case": "upper", "prefix": 2}]}, "equivalence-form-step-shape"),
        ({"project": ["upper"]}, "equivalence-form-step-shape"),
        ({"project": [{"nope": 1}]}, "equivalence-form-unknown-op"),
        ({"project": [{"prefix": True}]}, "equivalence-form-op-parameter"),
        ({"project": [{"case": "title"}]}, "equivalence-form-op-parameter"),
        ({"project": [{"map": {}}]}, "equivalence-form-op-parameter"),
        ({"project": [{"tokens": {"order": "sorted"}}]}, "equivalence-form-op-parameter"),
        (
            {"project": [{"tokens": {"split": " ", "order": "backwards"}}]},
            "equivalence-form-token-order",
        ),
        ({"compare": {"exact": True, "contains": True}}, "equivalence-form-compare-shape"),
        ({"compare": {"sounds_like": True}}, "equivalence-form-unknown-op"),
        ({"compare": {"edit_distance": True}}, "equivalence-form-op-parameter"),
    ):
        write_forms(pack, {"probe": dict(_GOOD_FORMS["folded"], **over)})
        found = find(pv.validate_pack(pack), code)
        assert found and found[0]["severity"] == "error", over
        assert "probe" in found[0]["message"], over


def test_a_form_that_can_only_yield_the_empty_string_is_an_error(pack):
    """Every value is then unresolvable, so no two are ever equivalent and the count is a zero
    that reads exactly like a clean population."""
    for project in (
        [{"prefix": 0}],
        [{"suffix": -1}],
        [{"case": "upper"}, {"tokens": {"split": " ", "take": 0}}],
    ):
        write_forms(pack, {"probe": {"project": project, "min_length": 2}})
        found = find(pv.validate_pack(pack), "equivalence-form-always-empty")
        assert found and found[0]["severity"] == "error", project

    write_forms(pack, {"probe": {"project": [{"prefix": 2}], "min_length": 2}})
    assert "equivalence-form-always-empty" not in codes(pv.validate_pack(pack))


def test_a_linkage_the_engine_does_not_implement_is_an_error(pack):
    """Single and complete linkage produce different classes over identical rows, so a third
    name is not a near miss — an unanchored comparison under the form is refused outright."""
    write_forms(pack, {"probe": dict(_GOOD_FORMS["near"], linkage="average")})
    found = find(pv.validate_pack(pack), "equivalence-form-linkage")
    assert found and found[0]["severity"] == "error"
    assert "single" in found[0]["detail"] and "complete" in found[0]["detail"]


def test_a_linkage_on_a_transitive_form_changes_nothing_and_says_so(pack):
    """A projection's classes are the same under either linkage, so the key is inert — and an
    inert key reads as a grouping decision the pack took and the engine honoured."""
    write_forms(pack, {"probe": dict(_GOOD_FORMS["folded"], linkage="single")})
    found = find(pv.validate_pack(pack), "equivalence-form-linkage-noop")
    assert found and found[0]["severity"] == "warning"

    write_forms(pack, _GOOD_FORMS)
    assert "equivalence-form-linkage-noop" not in codes(pv.validate_pack(pack))


def test_min_length_is_an_error_unread_and_a_warning_absent(pack):
    """The floor is what stops a shortening pipeline collapsing unrelated values onto one key,
    which fabricates a finding rather than losing one. A fixed-width code needs none, so its
    absence is a warning and its presence in an unreadable form is an error."""
    write_forms(pack, {"probe": dict(_GOOD_FORMS["folded"], min_length="six")})
    found = find(pv.validate_pack(pack), "equivalence-form-min-length")
    assert found and found[0]["severity"] == "error"

    write_forms(pack, {"probe": {"project": [{"keep": "alpha"}]}})
    found = find(pv.validate_pack(pack), "equivalence-form-no-min-length")
    assert found and found[0]["severity"] == "warning"

    # The conditional half: `case` cannot shorten anything, so there is nothing to collapse.
    write_forms(pack, {"probe": {"project": [{"case": "upper"}]}})
    assert "equivalence-form-no-min-length" not in codes(pv.validate_pack(pack))


def test_a_value_equivalence_missing_its_relation_is_an_error(pack):
    """The engine owns no relation of its own, so an absent `form` is not a fallback to
    equality — the check can only report `unknown`, and a decisive one then reads as
    INSUFFICIENT DATA with nothing naming the declaration that caused it."""
    write_forms(pack, _GOOD_FORMS)
    for over, code in (
        ({"form": None}, "value-equivalence-form-undeclared"),
        ({"operator": None}, "value-equivalence-operator"),
        ({"operator": "~="}, "value-equivalence-operator"),
        ({"anchor": {"field": "actor"}}, "value-equivalence-anchor-shape"),
        ({"anchor": {"source": "record"}}, "value-equivalence-anchor-shape"),
        ({"anchor": "record.actor"}, "value-equivalence-anchor-shape"),
    ):
        edit_rules(pack, lambda spec, o=over: one_condition(spec, **_eq_cond(**o)))
        found = find(pv.validate_pack(pack), code)
        assert found and found[0]["severity"] == "error", over

    # The positive control, with the one optional key declared: without it every assertion
    # above is satisfied by a check that reports on any `value_equivalence` at all.
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec, **_eq_cond(anchor={"source": "record", "field": "actor"})
        ),
    )
    assert not [
        c for c in codes(pv.validate_pack(pack)) if c.startswith("value-equivalence-")
    ]


def test_a_form_named_on_a_kind_that_does_not_read_it_is_a_warning(pack):
    """The declared relation is then applied nowhere and the check compares on the engine's
    own incumbent reading, under the pack's form name."""
    write_forms(pack, _GOOD_FORMS)
    for key in ("form", "normalize"):
        edit_rules(
            pack,
            lambda spec, k=key: one_condition(spec, **{k: "folded", "field": "actor"}),
        )
        found = find(pv.validate_pack(pack), "equivalence-form-noop-kind")
        assert found and found[0]["severity"] == "warning", key
        assert key in found[0]["message"], key
        assert "read only by:" in found[0]["detail"], key


def test_a_form_name_on_a_fixed_mode_seam_is_an_error(pack):
    """`normalize` is one key over two vocabularies. On a mode seam a form name is not applied
    and not rejected, so the comparison runs on the engine's reading under the form's name."""
    write_forms(pack, _GOOD_FORMS)
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="field_equality",
            left={"source": "record", "field": "actor"},
            right={"source": "record", "field": "owner"},
            normalize="folded",
        ),
    )
    found = find(pv.validate_pack(pack), "equivalence-form-on-mode-seam")
    assert found and found[0]["severity"] == "error"
    assert "distinct_count" in found[0]["hint"]

    # The mode this seam does read is silent, and is not reported as an undeclared form.
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="field_equality",
            left={"source": "record", "field": "actor"},
            right={"source": "record", "field": "owner"},
            normalize="identifier",
        ),
    )
    assert not [
        c for c in codes(pv.validate_pack(pack)) if "equivalence-form" in c
    ]


def test_a_form_no_pack_declares_is_an_error(pack):
    """The engine refuses it rather than falling back, so the check reports `unknown` — and a
    silent fallback to equality is the one wrong answer that still looks like a working
    check."""
    write_forms(pack, _GOOD_FORMS)
    edit_rules(pack, lambda spec: one_condition(spec, **_eq_cond(form="never_declared")))
    found = find(pv.validate_pack(pack), "unknown-equivalence-form")
    assert found and found[0]["severity"] == "error"
    assert "folded" in found[0]["detail"]


def test_a_pairwise_form_has_no_distinct_count_of_its_own(pack):
    """A projection is transitive so "how many distinct" has one answer; a `compare` relation
    is not, and the same question becomes "how many classes under which linkage"."""
    write_forms(pack, _GOOD_FORMS)
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="distinct_count",
            field="actor",
            normalize="near",
            max=1,
        ),
    )
    found = find(pv.validate_pack(pack), "equivalence-form-not-transitive")
    assert found and found[0]["severity"] == "error"
    assert "value_equivalence" in found[0]["hint"]

    edit_rules(
        pack,
        lambda spec: one_condition(
            spec, kind="distinct_count", field="actor", normalize="folded", max=1
        ),
    )
    assert "equivalence-form-not-transitive" not in codes(pv.validate_pack(pack))


def test_grouping_under_a_pairwise_form_needs_a_linkage_or_an_anchor(pack):
    """The engine refuses to pick one, because single and complete linkage produce different
    classes over identical rows and therefore different verdicts."""
    anchorless = {"probe": {"compare": {"edit_distance": 1}}}
    write_forms(pack, anchorless)
    edit_rules(pack, lambda spec: one_condition(spec, **_eq_cond(form="probe")))
    found = find(pv.validate_pack(pack), "equivalence-form-linkage-required")
    assert found and found[0]["severity"] == "error"
    assert "anchor" in found[0]["hint"]

    # Both ways out: an anchor compares against a fixed side instead of clustering...
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            **_eq_cond(form="probe", anchor={"source": "record", "field": "actor"}),
        ),
    )
    assert "equivalence-form-linkage-required" not in codes(pv.validate_pack(pack))

    # ...and a declared linkage answers the question the engine would otherwise be taking.
    write_forms(pack, {"probe": dict(anchorless["probe"], linkage="complete")})
    edit_rules(pack, lambda spec: one_condition(spec, **_eq_cond(form="probe")))
    assert "equivalence-form-linkage-required" not in codes(pv.validate_pack(pack))


def test_a_form_on_an_aggregate_that_reads_no_text_is_a_warning(pack):
    """A form projects text, and `sum` does not read its field as text — so the declaration is
    inert while reading as a dedup rule the pack chose."""
    write_forms(pack, _GOOD_FORMS)
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec,
            kind="numeric_compare",
            aggregate="sum",
            field="amount",
            normalize="folded",
            operator="<=",
            bound=10,
        ),
    )
    found = find(pv.validate_pack(pack), "equivalence-form-inert-aggregate")
    assert found and found[0]["severity"] == "warning"
    assert "distinct" in found[0]["detail"]

    for agg in ("distinct", "mode", "mode_share"):
        edit_rules(
            pack,
            lambda spec, a=agg: one_condition(
                spec,
                kind="numeric_compare",
                aggregate=a,
                field="actor",
                normalize="folded",
                operator="<=",
                bound=1,
            ),
        )
        assert "equivalence-form-inert-aggregate" not in codes(
            pv.validate_pack(pack)
        ), agg


def test_a_well_formed_vocabulary_is_silent(pack):
    """The positive control for the whole section. A false positive here teaches the operator
    to scroll past the list, which is where the real defects are."""
    write_forms(pack, _GOOD_FORMS)
    edit_rules(
        pack,
        lambda spec: one_condition(
            spec, **_eq_cond(form="near", operator=">=", bound=2)
        ),
    )
    result = pv.validate_pack(pack)
    assert not [c for c in codes(result) if "equivalence" in c], codes(result)
    assert result["counts"]["equivalence_forms"] == len(_GOOD_FORMS)


@pytest.mark.parametrize("name", installed_packs())
def test_the_forms_check_reads_whatever_a_pack_declares(name):
    """Silence must mean "nothing declared", not "nothing checked" — an unchecked pack reads
    exactly like a clean one, so the count is the discriminator and it is emitted from the
    file's existence rather than from what parsed out of it."""
    root = PACKS / name
    result = pv.validate_pack(root)
    declared = (root / "shared" / "equivalence_forms.yaml").is_file()
    assert ("equivalence_forms" in result["counts"]) is declared
    assert not [
        d
        for d in result["diagnostics"]
        if d["severity"] == "error" and "equivalence" in d["code"]
    ]
