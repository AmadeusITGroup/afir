"""Tests for `docs/knowledge-pack-template/`, the template copied to start a new domain.

Two claims are verified: (1) the template loads as valid YAML, and (2) every key it documents
is actually read by `src/`. A key the template documents but the engine ignores propagates
silently into every derived pack. The template must also name no domain vocabulary.
"""

import re

import pytest
import yaml

from src.knowledge.pack import load_knowledge_pack
from src.knowledge.pack_validate import (
    VOCABULARY_FILE,
    _ENGINE_OWNS,
    vocabulary_pattern,
)
from src.utils.paths import REPO_ROOT

TEMPLATE_DIR = REPO_ROOT / "docs" / "knowledge-pack-template"


def _installed_vocabularies():
    """Every word every installed pack declares as its own.

    `pack_validate` reads one pack's list, because it validates one pack; the template is held
    against all of them at once — it is copied to start ANY pack, so a word belonging to a pack
    already on the branch is a word the copy would inherit as a collision.

    Every pack under `knowledge/` is read, not just the one `main_config.yaml` selects: a pack
    present on the branch is a pack whose words the template must not use, whether or not this
    deployment happens to load it, and reading only the configured pack would make the check's
    strictness depend on a gitignored config file.
    """
    words = set()
    knowledge = REPO_ROOT / "knowledge"
    if not knowledge.is_dir():
        return words
    for pack_dir in sorted(p for p in knowledge.iterdir() if p.is_dir()):
        declared = pack_dir / VOCABULARY_FILE
        if not declared.is_file():
            continue
        data = yaml.safe_load(declared.read_text(encoding="utf-8")) or {}
        for word in data.get("domain_vocabulary") or []:
            if str(word).strip():
                words.add(str(word).strip().lower())
    return words


@pytest.fixture(scope="module")
def pack():
    return load_knowledge_pack(TEMPLATE_DIR)


@pytest.fixture(scope="module")
def spec(pack):
    return pack.ruleset_spec("example_use_case")


# --- 1. the template loads ---------------------------------------------------


def test_the_template_loads_as_shipped(pack):
    """Every subsystem gets a real entry, so a new author has a working baseline to diff.

    A template that only PARSES is not enough: an author who copies it and sees zero sources
    cannot tell whether their edit broke the load or the template never populated that field.
    Each assertion below corresponds to one folder of the pack.
    """
    assert pack.name == "knowledge-pack-template"
    assert [e.type for e in pack.entities] == ["example_entity"]
    assert [s.name for s in pack.sources] == ["example_source"]
    assert len(pack.playbook_documents) == 1
    assert len(pack.schema_documents) == 1
    assert len(pack.case_documents) == 1
    assert sorted(pack.pack_data) == ["example_lookup", "example_use_case_lookup"]
    assert sorted(pack.shared_checks) == ["example_checks/example_shared_check"]
    assert sorted(pack.rulesets.get("verdicts") or {}) == ["example_use_case"]


def test_the_shared_layer_is_shared_and_the_scoped_layer_is_scoped(pack):
    """The pack's ONE structural rule: pack root = shared, `use_cases/<name>/` = specific.

    A shared concept loads UNTAGGED, and that is the whole mechanism — untagged, it is
    retrievable by every use case and nameable by any ruleset. Tag it and you have made an
    ownership claim the loader then enforces by hiding it from everybody else. The template
    ships one of each so the difference is visible in a diff rather than only described.
    """
    by_id = {
        d["metadata"]["concept_id"]: d["metadata"].get("use_case")
        for d in pack.concept_documents
    }
    assert by_id["example_concept"] is None, "a shared concept must load untagged"
    assert by_id["example_use_case_judgement"] == "example_use_case"
    # And the union a use case sees contains BOTH — specific first.
    assert [
        d["metadata"]["concept_id"] for d in pack.concepts_for("example_use_case")
    ] == ["example_use_case_judgement", "example_concept"]


def test_the_scoped_reporting_file_overrides_the_root_per_slot(pack):
    """Per-slot resolution, not all-or-nothing: a use case re-words one phrase, inherits the
    rest. Merged flat instead, one procedure's clause number would be cited in every other
    procedure's report — which is the defect that moved these strings out of `src/`.
    """
    scoped = pack.report_phrase("containment_target", "", use_case="example_use_case")
    root = pack.report_phrase("containment_target", "")
    assert scoped != root and "§" in scoped
    # A slot the use case does NOT re-declare falls through to the root file.
    assert pack.report_phrase(
        "containment_action", "", use_case="example_use_case"
    ) == pack.report_phrase("containment_action", "")
    # And an undeclared slot returns the ENGINE's default, never "".
    assert (
        pack.report_phrase("no_such_slot", "engine default", use_case="example_use_case")
        == "engine default"
    )


def test_the_template_ruleset_resolves_but_reaches_no_verdict(pack, spec):
    """Deliberately inert: one `stub` condition, so the file is provably live without the
    template asserting a domain outcome it cannot possibly know. `stub` is also the honest
    encoding of a procedure step whose data does not exist yet — it reports `unknown` with a
    stated reason instead of being omitted, which is what keeps the requirement visible.
    """
    assert spec is not None
    assert [(c["id"], c["kind"]) for c in spec["conditions"]] == [
        ("example_stub", "stub")
    ]
    assert spec["subject_entity"] == "example_entity"
    # The four rollup states are all named — including `out_of_scope`, the one most often
    # omitted. Without it a gate FAIL prints as the false-positive label, which claims the case
    # was examined and cleared when the procedure in fact declined to adjudicate.
    assert set(spec["labels"]) == {
        "fraud",
        "false_positive",
        "insufficient",
        "out_of_scope",
    }


def test_the_shared_check_library_is_importable_as_documented(pack):
    """The file STEM is the namespace, so the address in the docs must be the real address.

    An unresolvable `use:` is fatal at LOAD (eagerly, over every ruleset), so a wrong key here
    would already fail the load test above — but only if some ruleset imported it. The template
    keeps the import commented so an author sees an inert example, which means the key itself
    needs its own assertion.
    """
    check = pack.shared_checks["example_checks/example_shared_check"]
    assert check["kind"] == "field_flag"
    # MECHANICS ONLY. The library declaring a weighting key would hand every importer one
    # procedure's opinion and quietly make this the place weighting decisions get made.
    assert not (
        {"decisive", "decisive_on", "polarity", "exclusion_kind", "report_group", "gate"}
        & set(check)
    )


# --- 2. the documentation is true -------------------------------------------


def _commented_keys(path):
    """Every `# some_key:` the file documents — i.e. the keys it tells an author to write.

    A commented key looks like a YAML line that happens to be commented, so it must be followed
    by a VALUE or end the line (`# scope: org_unit`, `# lock_target:`). Requiring that is what
    separates it from wrapped explanatory prose whose continuation line happens to begin with a
    word and a colon — "# authors: a boolean holds neither a count nor..." is a sentence, and
    counting it as a key made this test demand that `src/` contain the word "authors".

    Deliberately still coarse in the other direction: `# scope: org_unit` inside a prose sentence
    would pass as a key. Over-reporting a key the engine does read costs nothing; the failure
    this test exists to catch is a key nothing reads.
    """
    return {
        m.group(1)
        for m in re.finditer(
            r"^\s*#\s{1,6}([a-z_][a-z0-9_]*):(?:$|\s+(?:[\[\"'>|{]|[-\w.]+\s*$))",
            path.read_text(),
            re.M,
        )
    }


def test_documented_keys_are_keys_the_engine_actually_reads():
    """Every commented-out key in the template must appear somewhere in `src/`.

    Coarse on purpose: a name anywhere in `src/`, docstrings included, passes. A pack key
    reaches the engine as a quoted `.get("k")`, as a bare Pydantic field, or via `**`
    expansion; both spellings are matched. An intentional exception belongs in `ALLOWED`
    with its reason.
    """
    src = "".join(p.read_text() for p in (REPO_ROOT / "src").rglob("*.py"))
    # Keys the pack author names freely within open maps (`entity_bindings` per-form sub-keys,
    # `lock_target.actions` platform keys, `case_builder.action_templates` step keys).
    # The engine does not recognise them by construction.
    ALLOWED = {
        "badge",  # a FORM name under entity_bindings — the glossary's value_forms names it
        "example_platform_a",  # a key in lock_target.actions — the pack names platforms
        "example_platform_b",  # ditto
        "fraud_step",  # a key in case_builder.action_templates (see _ACTION_TEMPLATES)
    }
    missing = set()
    for path in TEMPLATE_DIR.rglob("*.yaml"):
        for key in _commented_keys(path) - ALLOWED:
            quoted = f'"{key}"' in src or f"'{key}'" in src
            # A Pydantic field declaration or an attribute read: `value_forms:` /
            # `.value_forms`. Anchored so a substring of an unrelated identifier cannot pass.
            wired = re.search(rf"(?:^\s+{key}\s*:|\.{key}\b)", src, re.M)
            if not quoted and not wired:
                missing.add(f"{path.relative_to(TEMPLATE_DIR)}: {key}")
    assert not missing, (
        "the template documents keys the engine never reads (either wire them or delete "
        f"them from the template): {sorted(missing)}"
    )


#: Built with the engine's own pattern builder and its own exemption set, not a copy of either:
#: a duplicated ban-list drifts word by word, and the template and the engine are held to one
#: standard. Only the "which packs" question is answered here — see `_installed_vocabularies`.
_BANNED_VOCABULARY = vocabulary_pattern(
    {w for w in _installed_vocabularies() if w not in _ENGINE_OWNS}
)


def test_the_templates_yaml_names_no_domain_in_values_OR_COMMENTS():
    """A domain word in the template's YAML is inherited by every pack copied from it.

    Comments are scanned: the template is ~90% comments by line count and they are copied
    wholesale. A cited measurement may keep its structure while domain identifiers become
    placeholders.
    """
    offenders = []
    for path in TEMPLATE_DIR.rglob("*.yaml"):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            for m in _BANNED_VOCABULARY.finditer(line):
                offenders.append(f"{path.relative_to(TEMPLATE_DIR)}:{i}: {m.group(0)}")
    assert not offenders, (
        "the template's YAML must name no domain — every pack copied from it inherits both its "
        f"values and its documentation: {offenders[:20]}"
    )


def test_the_templates_markdown_guidance_names_no_domain():
    """The prose an author reads for guidance must not assume a domain either.

    Scanned in FULL, unlike the YAML above: a markdown TEMPLATE has no comment/value split, so
    every line of it is content the author is meant to keep and fill in. The concept, playbook
    and case templates are the three documents where a newcomer is most likely to mistake an
    illustration for a requirement, and the whole file is copyable — including a backticked
    `concept_id: some_value`, which is why inline code gets no exemption here.

    `README.md` is excluded and scanned separately below. It is the one `.md` in the pack that
    an author READS rather than fills in, and the distinction is the same one the YAML test
    draws: what leaks is what gets COPIED.
    """
    offenders = []
    for path in TEMPLATE_DIR.rglob("*.md"):
        if path.name == "README.md":
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            for m in _BANNED_VOCABULARY.finditer(line):
                offenders.append(f"{path.relative_to(TEMPLATE_DIR)}:{i}: {m.group(0)}")
    assert not offenders, f"template guidance names a domain: {offenders[:20]}"


def test_the_templates_own_prose_names_no_domain_outside_a_quotation():
    """`README.md`, scanned with backtick spans stripped.

    Backtick spans delimit quoted material (verbatim report lines, literals, filenames) from
    prose. The prose outside them is held to the full domain-vocabulary rule.
    """
    text = (TEMPLATE_DIR / "README.md").read_text()
    offenders = []
    for i, line in enumerate(text.splitlines(), 1):
        prose = re.sub(r"`[^`]*`", "", line)
        for m in _BANNED_VOCABULARY.finditer(prose):
            offenders.append(f"README.md:{i}: {m.group(0)}")
    assert not offenders, (
        "the template's README argues from a domain rather than quoting one: "
        f"{offenders[:20]}"
    )
