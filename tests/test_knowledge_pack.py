"""Tests for the swappable knowledge pack loader."""

import pytest

from src.knowledge.pack import KnowledgePack, load_knowledge_pack

_GLOSSARY = """
entities:
  - type: org_unit
    description: Point of sale.
    value_pattern: "^[A-Z0-9]{6,9}$"
    cardinality: many-per-org
    examples: [ORGUNIT2301, KKK1L12WX]
    used_by: [PB-APP-SCHEME-001]
    field_aliases: [orgUnitId, org_unit_id]
  - type: document
    field_aliases: [document_nbr]
"""

_CATALOG = """
sources:
  - name: transaction_logs
    description: Issuance transactions.
    entities: [org_unit, document]
    used_by: [PB-APP-SCHEME-001]
    endpoints:
      kind: databricks_uc
      catalog: main
      schema: gold
      tables: [settlement_report]
    projection: [creator.sign.red, route.air, element_counters.AUX]
    partition_columns:
      - {name: year,  role: year,  type: STRING}
      - {name: month, role: month, type: STRING, pad_days: 2}
    entity_bindings:
      org_unit: [pos_org_unit, org_unit_id]
      document: [document_number]
  - name: access_trail
    description: A source that declares no physical layout.
    entities: [document]
    endpoints:
      kind: databricks_uc
      catalog: main
      schema: gold
      tables: [audit_trail]
"""

_SCHEME_RULES = """
verdicts:
  scheme:
    label_scheme: scheme
    subject_entity: record
    labels:
      fraud: "VALID FRAUD"
      false_positive: "FALSE POSITIVE"
      insufficient: "INSUFFICIENT DATA"
    sources:
      record: record_lake
    conditions:
      - id: bare
        kind: element_absence
        source: record
        counters: [element_counters.AUX]
        decisive: true
    routes:
      - [LBV, CMN]
"""


def _write_pack(tmp_path, with_scheme=False):
    (tmp_path / "entity_glossary.yaml").write_text(_GLOSSARY)
    (tmp_path / "source_catalog.yaml").write_text(_CATALOG)
    if with_scheme:
        (tmp_path / "rulesets.yaml").write_text(_SCHEME_RULES)
    playbooks = tmp_path / "playbooks"
    playbooks.mkdir(exist_ok=True)
    (playbooks / "fraud.md").write_text("# Playbook\nWhat to correlate.")
    return tmp_path


def test_source_projection_loaded(tmp_path):
    pack = load_knowledge_pack(_write_pack(tmp_path))
    src = pack.source("transaction_logs")
    assert src.projection == [
        "creator.sign.red",
        "route.air",
        "element_counters.AUX",
    ]
    # A source without a projection defaults to an empty list (unchanged behavior).
    other = KnowledgePack().sources
    assert other == []


def test_source_partition_columns_loaded(tmp_path):
    """The pack's partition DECLARATION survives the load with its roles intact.

    This is the override for what backend metadata cannot report — a VIEW hides its
    underlying table's layout, and no catalog anywhere carries `role`/`pad_days`. If the
    loader dropped or flattened these, the retriever would silently fall back to an
    unbounded scan, which does not fail visibly: it just times out and reads as
    "the source had nothing".
    """
    pack = load_knowledge_pack(_write_pack(tmp_path))
    parts = pack.source("transaction_logs").partition_columns
    assert [p["name"] for p in parts] == [
        "year",
        "month",
    ]  # order = partition key order
    assert [p["role"] for p in parts] == ["year", "month"]
    assert parts[1]["pad_days"] == 2  # per-column, not global
    # A source that declares nothing gets an empty list, never None — the retriever
    # merges discovered specs on top of this and must be able to iterate it.
    assert pack.source("access_trail").partition_columns == []


def test_scheme_spec_present_and_shape(tmp_path):
    pack = load_knowledge_pack(_write_pack(tmp_path, with_scheme=True))
    spec = pack.ruleset_spec()
    assert spec is not None
    assert spec["subject_entity"] == "record"
    assert spec["labels"]["fraud"] == "VALID FRAUD"
    assert len(spec["conditions"]) == 1
    assert spec["routes"] == [["LBV", "CMN"]]


def test_scheme_spec_none_without_rules_file(tmp_path):
    # No rulesets.yaml written → ruleset_spec() is None (graceful; verdict stage no-ops).
    pack = load_knowledge_pack(_write_pack(tmp_path, with_scheme=False))
    assert pack.ruleset_spec() is None
    # An unknown ruleset key is also None.
    pack2 = load_knowledge_pack(_write_pack(tmp_path, with_scheme=True))
    assert pack2.ruleset_spec("does_not_exist") is None


def test_empty_pack_scheme_spec_none():
    assert KnowledgePack().ruleset_spec() is None


def test_loads_entities_sources_and_playbooks(tmp_path):
    pack = load_knowledge_pack(_write_pack(tmp_path))

    assert pack.entity_types() == ["org_unit", "document"]
    assert pack.aliases_for("org_unit") == ["orgUnitId", "org_unit_id"]
    assert [s.name for s in pack.sources] == ["transaction_logs", "access_trail"]
    assert len(pack.playbook_documents) == 1
    assert pack.playbook_documents[0]["type"] == "playbook"
    assert pack.playbook_documents[0]["title"] == "fraud"


def test_prompt_fragments(tmp_path):
    pack = load_knowledge_pack(_write_pack(tmp_path))

    glossary = pack.glossary_prompt()
    assert "org_unit" in glossary and "document" in glossary

    catalog = pack.catalog_prompt(["transaction_logs"])
    assert "transaction_logs" in catalog
    assert "Issuance transactions" in catalog

    hints = pack.alias_hints(["org_unit", "document"])
    assert "org_unit: orgUnitId, org_unit_id" in hints
    assert "document: document_nbr" in hints


def test_missing_pack_returns_empty(tmp_path):
    pack = load_knowledge_pack(tmp_path / "does_not_exist")
    assert isinstance(pack, KnowledgePack)
    assert pack.entities == []
    assert pack.sources == []
    assert pack.glossary_prompt() == ""
    assert pack.catalog_prompt() == ""


def test_source_lookup(tmp_path):
    pack = load_knowledge_pack(_write_pack(tmp_path))
    src = pack.source("transaction_logs")
    assert src is not None
    assert src.entities == ["org_unit", "document"]
    assert pack.source("nope") is None


def test_value_patterns_and_cardinality_loaded(tmp_path):
    pack = load_knowledge_pack(_write_pack(tmp_path))
    patterns = pack.value_patterns()
    assert patterns == {"org_unit": "^[A-Z0-9]{6,9}$"}  # document has none
    org_unit = next(e for e in pack.entities if e.type == "org_unit")
    assert org_unit.cardinality == "many-per-org"


def test_zero_row_meanings_reads_only_the_declarations_that_claim_an_answer(tmp_path):
    """The declaration reaching the READER, not just the health score — and only weight 0.

    `zero_rows.health_weight` discounts a score an operator may never look at; the same
    block's `meaning` is what tells the evidence render and the report's gaps section that
    an empty source ANSWERED. But the two claims differ and a pack makes both: only `0.0`
    says empty is a valid answer. A *discounted* source is empty for a recorded reason and
    still a gap — one measured `meaning` on that shape reads "its type could not be
    established", the `unknown` the reader must keep. Promoting it would fabricate certainty,
    which is this seam's own failure mode inverted.
    """
    pack = load_knowledge_pack(_write_pack(tmp_path))
    assert pack.zero_row_meanings() == {}  # the fixture declares none
    src = pack.sources[0]

    src.zero_rows = {"health_weight": 0.0}  # a weight with nothing to say
    assert pack.zero_row_meanings() == {}
    src.zero_rows = {"meaning": "empty means nothing was registered"}  # weight defaults 1.0
    assert pack.zero_row_meanings() == {}
    src.zero_rows = {"health_weight": 0.1, "meaning": "its type could not be established"}
    assert pack.zero_row_meanings() == {}
    src.zero_rows = {"health_weight": "not a number", "meaning": "the pair is not a robot"}
    assert pack.zero_row_meanings() == {}

    src.zero_rows = {"health_weight": 0.0, "meaning": "  the pair is not registered  "}
    assert pack.zero_row_meanings() == {src.name: "the pair is not registered"}


def test_field_priors_prefer_source_bindings(tmp_path):
    pack = load_knowledge_pack(_write_pack(tmp_path))

    # Per-source binding comes first, then global aliases (de-duplicated).
    priors = pack.field_priors_for("org_unit", "transaction_logs")
    assert priors[0] == "pos_org_unit"
    assert "org_unit_id" in priors and "orgUnitId" in priors
    assert priors.count("org_unit_id") == 1  # de-duped across binding + alias

    # No source / unknown source falls back to global aliases only.
    assert pack.field_priors_for("org_unit", None) == ["orgUnitId", "org_unit_id"]
    assert pack.field_priors_for("org_unit", "nope") == ["orgUnitId", "org_unit_id"]


def test_alias_hints_are_source_aware(tmp_path):
    pack = load_knowledge_pack(_write_pack(tmp_path))
    hint = pack.alias_hints(["org_unit", "document"], source_name="transaction_logs")
    assert "org_unit: pos_org_unit" in hint
    assert "document_number" in hint


def test_examples_used_by_and_endpoints_loaded(tmp_path):
    pack = load_knowledge_pack(_write_pack(tmp_path))

    org_unit = next(e for e in pack.entities if e.type == "org_unit")
    assert org_unit.examples == ["ORGUNIT2301", "KKK1L12WX"]
    assert org_unit.used_by == ["PB-APP-SCHEME-001"]

    src = pack.source("transaction_logs")
    assert src.used_by == ["PB-APP-SCHEME-001"]
    assert src.kind() == "databricks_uc"
    assert src.endpoints["catalog"] == "main"
    assert src.endpoints["tables"] == ["settlement_report"]


def test_glossary_prompt_includes_examples(tmp_path):
    pack = load_knowledge_pack(_write_pack(tmp_path))
    glossary = pack.glossary_prompt()
    assert "e.g. ORGUNIT2301" in glossary


def test_a_pack_with_no_abbreviations_says_nothing_about_them(tmp_path):
    """Every pack key must no-op when absent — a domain with no acronyms must not pay for
    the block, and the entities must still be described."""
    pack = load_knowledge_pack(_write_pack(tmp_path))
    assert pack.abbreviations == {}
    glossary = pack.glossary_prompt()
    assert "abbreviation" not in glossary.lower()
    assert glossary.startswith("Domain entities to extract")


def test_declared_abbreviations_are_injected_as_a_closed_list(tmp_path):
    """An LLM expands an unfamiliar acronym rather than leaving it alone, and a plausible
    wrong expansion reads exactly like a right one ("SCHEME (Contact Warning Alert)" — it is
    Central West Africa). So the prompt must both SUPPLY the expansion and FORBID guessing;
    supplying alone still leaves the undeclared terms open to invention."""
    pack_dir = _write_pack(tmp_path)
    glossary_file = pack_dir / "entity_glossary.yaml"
    glossary_file.write_text(
        "abbreviations:\n"
        '  SCHEME: "Central West Africa"\n'
        '  record: "Party Name Record"\n' + glossary_file.read_text()
    )
    pack = load_knowledge_pack(pack_dir)
    assert pack.abbreviations["SCHEME"] == "Central West Africa"
    glossary = pack.glossary_prompt()
    assert "SCHEME = Central West Africa" in glossary
    assert "do NOT guess" in glossary
    # Ahead of the entities, so the vocabulary is set before anything uses it.
    assert glossary.index("SCHEME =") < glossary.index("Domain entities to extract")


# --- use_cases/ subtree ------------------------------------------------------

_UC_RULES = """
verdicts:
  scheme:
    label_scheme: scheme
    subject_entity: record
    labels: {fraud: "VALID FRAUD", false_positive: "FALSE POSITIVE", insufficient: "INSUFFICIENT DATA"}
    sources: {record: record_lake, auth: auth_events}
    conditions:
      - id: bare
        kind: element_absence
        source: record
        counters: [element_counters.AUX]
        decisive: true
      - id: not_automated
        kind: field_flag
        source: auth
        flag_fields: [value.payload.userInfo.robot]
        decisive: false
"""

_UC_CONCEPT = """---
concept_id: bare_record
title: Bare record
use_case: scheme
---

A bare record carries only the minimum elements. AUX breaks bare.
"""

_UC_CASE = """---
case_id: case_a1b2c3
use_case: scheme
verdict: FALSE POSITIVE
subject: SUBJ03
route: DSS-CDG
decisive_reasons: ["Not bare: AUX present"]
resolution: Closed by GET-TOM.
date: 2026-07-24
---

# Case SUBJ03
Closed FALSE POSITIVE — AUX present, split with void+reissue, >1h gap.
"""


def _write_use_cases_pack(tmp_path):
    """A pack with a use_cases/scheme/ subtree (rules override + concept + case)."""
    _write_pack(tmp_path, with_scheme=False)
    uc = tmp_path / "use_cases" / "scheme"
    (uc / "playbooks").mkdir(parents=True)
    (uc / "concepts").mkdir(parents=True)
    (uc / "cases").mkdir(parents=True)
    (uc / "rules.yaml").write_text(_UC_RULES)
    # A playbook with the SAME id as the flat root one → deduped (flat root wins).
    (uc / "playbooks" / "issuance_fraud.md").write_text(
        "---\nplaybook_id: PB-APP-SCHEME-001\n---\n# SCHEME playbook\n"
    )
    (uc / "concepts" / "bare_record.md").write_text(_UC_CONCEPT)
    (uc / "cases" / "case_a1b2c3.md").write_text(_UC_CASE)
    return tmp_path


def test_use_cases_rules_merged(tmp_path):
    pack = load_knowledge_pack(_write_use_cases_pack(tmp_path))
    spec = pack.ruleset_spec()
    assert spec is not None
    # The use_cases rules.yaml supplied the verdict ruleset (flat root had none).
    assert spec["sources"]["auth"] == "auth_events"
    ids = [c["id"] for c in spec["conditions"]]
    assert "not_automated" in ids


def test_use_cases_wins_on_collision(tmp_path):
    # Flat root ships a scheme ruleset; the use_cases one must override it.
    _write_pack(
        tmp_path, with_scheme=True
    )  # flat rulesets.yaml (1 condition, no auth)
    uc = tmp_path / "use_cases" / "scheme"
    uc.mkdir(parents=True)
    (uc / "rules.yaml").write_text(_UC_RULES)
    pack = load_knowledge_pack(tmp_path)
    spec = pack.ruleset_spec()
    # use_cases version has auth + not_automated; the flat one did not.
    assert "auth" in spec["sources"]
    assert any(c["id"] == "not_automated" for c in spec["conditions"])


def test_concept_and_case_documents_loaded(tmp_path):
    pack = load_knowledge_pack(_write_use_cases_pack(tmp_path))
    assert len(pack.concept_documents) == 1
    cd = pack.concept_documents[0]
    assert cd["type"] == "concept"
    assert cd["metadata"]["concept_id"] == "bare_record"
    assert cd["metadata"]["use_case"] == "scheme"

    assert len(pack.case_documents) == 1
    case = pack.case_documents[0]
    assert case["type"] == "case"
    assert case["metadata"]["case_id"] == "case_a1b2c3"
    assert case["metadata"]["verdict"] == "FALSE POSITIVE"
    assert case["metadata"]["decisive_reasons"] == ["Not bare: AUX present"]


def test_pack_data_loads_from_use_case_data_dir(tmp_path):
    """data/*.yaml under use_cases/<name>/data/ loads into pack.pack_data by file stem."""
    _write_pack(tmp_path, with_scheme=False)
    data = tmp_path / "use_cases" / "scheme" / "data"
    data.mkdir(parents=True)
    (data / "issuer_prefix_map.yaml").write_text('map:\n  "400": XY\n')
    pack = load_knowledge_pack(tmp_path)
    assert "issuer_prefix_map" in pack.pack_data
    assert pack.pack_data["issuer_prefix_map"]["map"]["400"] == "XY"


def test_pack_data_use_case_wins_over_flat_root(tmp_path):
    """Flat-root data/ loads too; a use_cases data file of the same stem wins on collision."""
    _write_pack(tmp_path, with_scheme=False)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "issuer_prefix_map.yaml").write_text('map:\n  "400": XX\n')
    uc_data = tmp_path / "use_cases" / "scheme" / "data"
    uc_data.mkdir(parents=True)
    (uc_data / "issuer_prefix_map.yaml").write_text('map:\n  "400": XY\n')
    pack = load_knowledge_pack(tmp_path)
    assert pack.pack_data["issuer_prefix_map"]["map"]["400"] == "XY"  # use_cases wins


def test_pack_data_empty_without_data_dir(tmp_path):
    pack = load_knowledge_pack(_write_pack(tmp_path))
    assert pack.pack_data == {}


def test_playbook_dedup_across_use_cases(tmp_path):
    # The use_cases playbook shares PB-APP-SCHEME-001 with... nothing in the flat root here,
    # but a second identical id must not duplicate. Add a flat playbook with that id.
    _write_pack(tmp_path, with_scheme=False)
    (tmp_path / "playbooks" / "flat.md").write_text(
        "---\nplaybook_id: PB-APP-SCHEME-001\n---\n# flat scheme\n"
    )
    uc = tmp_path / "use_cases" / "scheme" / "playbooks"
    uc.mkdir(parents=True)
    (uc / "dup.md").write_text("---\nplaybook_id: PB-APP-SCHEME-001\n---\n# dup\n")
    pack = load_knowledge_pack(tmp_path)
    ids = [d["metadata"].get("playbook_id") for d in pack.playbook_documents]
    assert ids.count("PB-APP-SCHEME-001") == 1  # deduped


def test_concepts_for_and_cases_for_accessors(tmp_path):
    pack = load_knowledge_pack(_write_use_cases_pack(tmp_path))
    assert [d["metadata"]["concept_id"] for d in pack.concepts_for("scheme")] == [
        "bare_record"
    ]
    assert pack.concepts_for("scheme", ["bare_record"])[0]["metadata"]["concept_id"] == (
        "bare_record"
    )
    assert pack.concepts_for("nope") == []
    assert [d["metadata"]["case_id"] for d in pack.cases_for("scheme")] == ["case_a1b2c3"]
    assert pack.cases_for("scheme", "FALSE POSITIVE")[0]["metadata"]["case_id"] == (
        "case_a1b2c3"
    )
    assert pack.cases_for("scheme", "VALID FRAUD") == []


def _write_concept(tmp_path, body):
    """Drop one concept doc into a use_cases pack and return the loaded document."""
    _write_pack(tmp_path, with_scheme=False)
    concepts = tmp_path / "use_cases" / "scheme" / "concepts"
    concepts.mkdir(parents=True)
    (concepts / "note.md").write_text(body)
    docs = load_knowledge_pack(tmp_path).concept_documents
    assert len(docs) == 1
    return docs[0]


def test_an_authoring_note_never_reaches_a_prompt(tmp_path):
    """An HTML comment is stripped at load, so neither snippet cut can carry it.

    Unlike a YAML comment — dropped by `safe_load` before anything is built — a markdown
    comment survives `read_text`, and every consumer reads a PREFIX of that text (the RAG
    context builder the first 1000 chars, the brief's concept snippet ~220). Ten docs in one
    shipped pack put their note at chars 346-511, i.e. inside the wider cut, where a sentence
    addressed to the pack's *reader* is injected as domain knowledge.
    """
    doc = _write_concept(
        tmp_path,
        "---\nconcept_id: bare_record\ntitle: A record with no counters\n---\n"
        "# A record with no counters\n\n"
        "The counters are absent, which is the finding.\n\n"
        "<!-- The claim is first AND SHORT deliberately: this doc is injected FLAT at 220\n"
        "     characters. The rest is for a pack reader. -->\n\n"
        "## Why, measured\n\n"
        "Absent on 99.9% of 1105 sampled documents.\n",
    )

    content = doc["content"]
    assert "<!--" not in content and "-->" not in content
    assert "pack reader" not in content
    # The note's absence is what pulls the real prose into the cut every consumer reads.
    assert "Why, measured" in content[:1000]
    # And it leaves no hole behind it: the blank lines that surrounded the note would
    # otherwise spend the cut they just freed.
    assert "\n\n\n" not in content
    # Frontmatter is parsed from the RAW text, so stripping must not cost the metadata.
    assert doc["metadata"]["concept_id"] == "bare_record"
    assert doc["title"] == "A record with no counters"


def test_a_doc_with_no_authoring_note_is_byte_identical(tmp_path):
    """The strip is not a reformat: a doc carrying no comment must load unchanged."""
    body = (
        "---\nconcept_id: bare_record\n---\n# Heading\n\n"
        "One paragraph.\n\n\n\nA gap the author typed on purpose.\n"
    )
    assert _write_concept(tmp_path, body)["content"] == body


def test_an_unterminated_authoring_note_is_left_alone(tmp_path):
    """A missing `-->` must not delete the rest of the doc.

    The failure mode being refused is silent truncation: a greedy strip would take the note
    and every measurement after it, and the doc would still load, still embed, and still read
    as a complete concept.
    """
    body = "---\nconcept_id: bare_record\n---\n# Heading\n\n<!-- unterminated\n\nMeasured: 1105 rows.\n"
    content = _write_concept(tmp_path, body)["content"]
    assert content == body
    assert "Measured: 1105 rows." in content


def test_flat_only_pack_has_no_concepts_or_cases(tmp_path):
    # Back-compat: a pack with no use_cases/ dir loads with empty concept/case lists.
    pack = load_knowledge_pack(_write_pack(tmp_path, with_scheme=True))
    assert pack.concept_documents == []
    assert pack.case_documents == []
    assert pack.ruleset_spec() is not None  # flat rulesets.yaml still works


def test_case_frontmatter_dates_are_json_serializable(tmp_path):
    # A bare YAML date (2026-07-24) parses as datetime.date; the loader must coerce it to
    # a string so the RAG documents.json dump doesn't fail on it.
    import json

    pack = load_knowledge_pack(_write_use_cases_pack(tmp_path))
    json.dumps(pack.case_documents)  # must not raise
    assert isinstance(pack.case_documents[0]["metadata"]["date"], str)


# --- schemas/: the complete field inventory (RAG-only) ------------------------------

_SCHEMA_YAML = """\
source: record_lake
kind: databricks_uc
tables:
  record_table_4:
    fully_qualified_name: cat.sch.record_table_4
    partition_columns:
    - creation_date
    - modification_date
    description: Every record version version.
    columns:
      enrichment:
        type: struct
        description: Derived/enriched blocks.
        leaves:
        - path: enrichment.aux_docs.citizenship_iso3
          type: string
          kind: scalar
          explode_path: enrichment.aux_docs
          array_depth: 1
          populated_pct: 41.2
          description: ISO-3166 alpha-3 citizenship from the AUX DOCS element.
        - path: enrichment.aux_docs
          type: array<struct<...>>
          kind: array
          populated_pct: 41.2
          description: ''
    measured:
      rows_in_window: 10
  a_view:
    fully_qualified_name: cat.sch.a_view
    partition_columns: []
    columns: {}
    measured:
      skipped_reason: no partition column to bound and not demonstrably small
"""

_ES_SCHEMA_YAML = """\
source: siem_alerts_current
kind: elasticsearch
tables:
  raw.prd.siem-alerts*:
    index_pattern: raw.prd.siem-alerts*
    discovery: sampled from documents via the Kibana gateway
    sampled_docs: 150
    fields:
      ir.text:
        json_types:
        - string
        present_in_sampled_pct: 80.0
        searchable: false
        description: Free-text incident note.
"""


def _write_schemas(tmp_path):
    _write_pack(tmp_path)
    d = tmp_path / "schemas"
    d.mkdir()
    (d / "record_lake.yaml").write_text(_SCHEMA_YAML)
    (d / "siem_alerts_current.yaml").write_text(_ES_SCHEMA_YAML)
    return tmp_path


def test_schema_documents_are_one_per_table(tmp_path):
    """The retrieval unit is ONE TABLE, not one source.

    A per-source doc for a table with thousands of leaves is not selectively
    retrievable at all — which is the whole point of putting these in RAG.
    """
    pack = load_knowledge_pack(_write_schemas(tmp_path))
    titles = sorted(d["title"] for d in pack.schema_documents)
    assert titles == [
        "Field schema: record_lake / a_view",
        "Field schema: record_lake / record_table_4",
        "Field schema: siem_alerts_current / raw.prd.siem-alerts*",
    ]
    assert {d["type"] for d in pack.schema_documents} == {"schema"}


def test_schema_doc_states_the_explode_a_dotted_path_cannot_express(tmp_path):
    """An array leaf must carry its explode instruction, or the field is unusable.

    `enrichment.aux_docs.citizenship_iso3` is not a valid dotted SQL path — this is the
    exact leaf that was recorded as "PROBE-CONFIRMED absent" when it exists.
    """
    pack = load_knowledge_pack(_write_schemas(tmp_path))
    doc = next(d for d in pack.schema_documents if d["metadata"]["table"] == "record_table_4")
    body = doc["content"]
    assert "enrichment.aux_docs.citizenship_iso3" in body
    assert "explode(enrichment.aux_docs)" in body
    assert "ISO-3166 alpha-3 citizenship" in body
    # Partition columns are stated first-class: an unbounded partition column is the
    # difference between a 6-second answer and a full scan read as "nothing was there".
    assert "MUST be bounded" in body
    assert "creation_date" in body and "modification_date" in body


def test_schema_doc_distinguishes_unmeasured_from_absent(tmp_path):
    """`skipped_reason` must survive into the text: a missing measurement is a stated
    gap, not a zero."""
    pack = load_knowledge_pack(_write_schemas(tmp_path))
    doc = next(d for d in pack.schema_documents if d["metadata"]["table"] == "a_view")
    assert "Population NOT measured" in doc["content"]


def test_es_schema_doc_flags_a_non_filterable_field(tmp_path):
    """A field that is returnable but NOT searchable must say so.

    Filtering on one returns 0 rows, which then gets written down as "the source had
    nothing" — the most expensive kind of wrong negative.
    """
    pack = load_knowledge_pack(_write_schemas(tmp_path))
    doc = next(
        d for d in pack.schema_documents if d["metadata"]["backend_kind"] == "elasticsearch"
    )
    assert "NOT searchable" in doc["content"]
    assert "sampled from documents" in doc["content"]


def test_schema_documents_never_reach_the_catalog_prompt(tmp_path):
    """RAG-only by construction.

    The inventory is far too large for a prompt; pasting it would crowd out the pack's
    targeting guidance and shift query shapes on sources that work today.
    """
    pack = load_knowledge_pack(_write_schemas(tmp_path))
    assert pack.schema_documents  # the docs exist...
    prompt = pack.catalog_prompt()
    assert "citizenship_iso3" not in prompt  # ...and none of them is in the prompt
    assert "Field schema" not in prompt


def test_pack_without_schemas_dir_loads_clean(tmp_path):
    pack = load_knowledge_pack(_write_pack(tmp_path))
    assert pack.schema_documents == []
    assert KnowledgePack().schema_documents == []


def test_malformed_schema_file_does_not_break_the_pack(tmp_path):
    """A pack must never fail to load because one generated file is bad."""
    _write_pack(tmp_path)
    d = tmp_path / "schemas"
    d.mkdir()
    (d / "broken.yaml").write_text("tables: not-a-mapping\n")
    (d / "listish.yaml").write_text("- 1\n- 2\n")
    (d / "ok.yaml").write_text(_ES_SCHEMA_YAML)
    pack = load_knowledge_pack(tmp_path)
    assert len(pack.schema_documents) == 1


# THE SHARED LAYER: knowledge about the DATA, reachable by every use case.
#
# The pack root is shared (glossary, catalog, schemas, data) and `use_cases/<n>/` is
# specific. `shared/concepts/` and `shared/checks/` extend that rule to the two things that
# were previously trapped inside one use case: the prose describing an element, and the
# mechanics of the check that reads it. The tests below pin the seam, not any pack's
# content — a second use case checking the same element must be able to reuse both without
# copying a field path.

# The file STEM is the namespace and its top-level keys are the check ids, so this file
# defines `record_elements/bare` and `record_elements/split`. A bare id would collide the
# moment two subject areas both define `velocity`.
_SHARED_CHECKS = """
bare:
  kind: element_absence
  source: record
  counters: [element_counters.AUX, element_counters.INS]
  arrays: [remark.rm, remark.ry]
  label: "Bare record (no servicing elements)"
  expected_label: "all servicing elements absent"
split:
  kind: element_presence
  source: record
  arrays: [split.sp]
  label: "Record was split"
"""

_SHARED_CONCEPT = """---
concept_id: record_elements
title: Record elements
---

# Record elements

`element_counters` has no `RM` leaf; remarks live in the `remark.rm` array. Selecting a
non-existent struct leaf fails the WHOLE query, not just that leaf.
"""

_RULES_WITH_IMPORTS = """
verdicts:
  alpha:
    subject_entity: record
    sources:
      record: transaction_logs
    conditions:
      - use: record_elements/bare
        report_group: validation
        decisive: true
        decisive_on: [fail]
        order: 60
      - id: local_only
        kind: element_presence
        source: record
        arrays: [pricing.payment]
"""


def _write_shared(tmp_path, rules=_RULES_WITH_IMPORTS, checks=_SHARED_CHECKS):
    _write_pack(tmp_path)
    checks_dir = tmp_path / "shared" / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)
    (checks_dir / "record_elements.yaml").write_text(checks)
    concepts_dir = tmp_path / "shared" / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    (concepts_dir / "record_elements.md").write_text(_SHARED_CONCEPT)
    uc = tmp_path / "use_cases" / "alpha"
    uc.mkdir(parents=True, exist_ok=True)
    (uc / "rules.yaml").write_text(rules)
    return tmp_path


def test_a_ruleset_imports_shared_check_mechanics_and_keeps_its_own_weighting(tmp_path):
    """THE POINT OF THE LIBRARY: field paths written once, weighting stays per procedure.

    The mechanics (kind, source, counters, arrays) describe where a fact LIVES in the data
    and are identical for every use case reading that element. The weighting is what one
    procedure decided about it. So the import supplies the first and the ruleset the second.
    """
    pack = load_knowledge_pack(_write_shared(tmp_path))
    spec = pack.ruleset_spec("alpha")
    cond = next(c for c in spec["conditions"] if c["id"] == "bare")
    # mechanics came from the library...
    assert cond["kind"] == "element_absence"
    assert cond["counters"] == ["element_counters.AUX", "element_counters.INS"]
    assert cond["arrays"] == ["remark.rm", "remark.ry"]
    assert cond["label"] == "Bare record (no servicing elements)"
    # ...weighting from the importing ruleset.
    assert cond["decisive"] is True
    assert cond["decisive_on"] == ["fail"]
    assert cond["order"] == 60
    assert cond["report_group"] == "validation"
    # `use` is consumed, not left on the resolved condition.
    assert "use" not in cond


def test_the_imported_id_defaults_to_the_check_id_so_it_is_stable_across_use_cases(
    tmp_path,
):
    """One fact keeps ONE id: the report, the health scorer and the fixtures key on it."""
    pack = load_knowledge_pack(_write_shared(tmp_path))
    ids = [c["id"] for c in pack.ruleset_spec("alpha")["conditions"]]
    assert ids == ["bare", "local_only"]


def test_a_ruleset_may_override_a_mechanics_key_and_the_override_REPLACES(tmp_path):
    """A list override must replace, never concatenate.

    A deep merge that unioned `counters` would silently make a check read a leaf that does
    not exist on the source it was pointed at — and selecting a non-existent struct leaf
    fails the whole query, not just that leaf. Replacement is the only safe reading.
    """
    rules = _RULES_WITH_IMPORTS.replace(
        "        order: 60",
        "        order: 60\n        counters: [element_counters.AUX]",
    )
    pack = load_knowledge_pack(_write_shared(tmp_path, rules=rules))
    cond = next(
        c for c in pack.ruleset_spec("alpha")["conditions"] if c["id"] == "bare"
    )
    assert cond["counters"] == ["element_counters.AUX"]  # replaced, NOT unioned
    assert cond["arrays"] == ["remark.rm", "remark.ry"]  # untouched keys survive


def test_two_use_cases_import_the_same_check_with_DIFFERENT_weighting(tmp_path):
    """The scenario the split exists for: reuse the mechanics, disagree on the verdict.

    Bare is a decisive exclusion for one procedure and may be an ordinary indicator for
    another. Neither has to restate where the elements live.
    """
    tmp_path = _write_shared(tmp_path)
    beta = tmp_path / "use_cases" / "beta"
    beta.mkdir(parents=True, exist_ok=True)
    (beta / "rules.yaml").write_text(
        """
verdicts:
  beta:
    subject_entity: record
    sources:
      record: transaction_logs
    conditions:
      - use: record_elements/bare
        polarity: fraud_indicator
        decisive: false
"""
    )
    pack = load_knowledge_pack(tmp_path)
    a = next(c for c in pack.ruleset_spec("alpha")["conditions"] if c["id"] == "bare")
    b = next(c for c in pack.ruleset_spec("beta")["conditions"] if c["id"] == "bare")
    assert a["counters"] == b["counters"]  # same mechanics, declared once
    assert a["decisive"] is True and b["decisive"] is False  # different judgement
    assert b["polarity"] == "fraud_indicator"


def test_a_SECOND_ruleset_is_reachable_and_selects_its_own_conditions(tmp_path):
    """A pack's second procedure must be reachable, or it silently never adjudicates.

    ``ruleset_spec("")`` falls back to the FIRST declared ruleset and ``use_cases/`` merges
    alphabetically, so before the selection seam existed the second use case's procedure
    could not be chosen at all. That is not a graceful degrade to no-verdict: two procedures
    in one domain usually read the SAME sources, so the wrong one's conditions resolve
    against real rows and return a confident wrong verdict under the wrong labels — exactly
    what happens here, where both rulesets read ``transaction_logs`` and import the same
    check with OPPOSITE weighting.

    Asserted on the resolved OUTPUT (the condition's weighting, which is what changes a
    verdict), not on a mapping dict existing — a test that reads the index back proves only
    that somebody built an index.
    """
    tmp_path = _write_shared(tmp_path)
    beta = tmp_path / "use_cases" / "beta"
    beta.mkdir(parents=True, exist_ok=True)
    (beta / "rules.yaml").write_text(
        """
verdicts:
  beta:
    subject_entity: record
    sources:
      record: transaction_logs
    conditions:
      - use: record_elements/bare
        polarity: fraud_indicator
        decisive: false
"""
    )
    pack = load_knowledge_pack(tmp_path)
    # The pre-existing behaviour: an unkeyed lookup takes the first-declared ruleset.
    assert pack.ruleset_keys()[0] == "alpha"
    assert pack.ruleset_spec()["subject_entity"] == "record"

    # A use case's name selects its own ruleset...
    assert pack.ruleset_key_for("beta") == "beta"
    beta_cond = next(
        c
        for c in pack.ruleset_spec(pack.ruleset_key_for("beta"))["conditions"]
        if c["id"] == "bare"
    )
    # ...and the weighting proves it is BETA's procedure, not the first-declared one.
    assert beta_cond["decisive"] is False
    assert beta_cond["polarity"] == "fraud_indicator"

    # A use case that declares no ruleset of its own, or an unknown one, resolves to ""
    # — i.e. "caller decides", which keeps the first-declared fallback for every pack that
    # ships one ruleset or none. This is what makes the seam inert where it must be.
    assert pack.ruleset_key_for("narrative_only_use_case") == ""
    assert pack.ruleset_key_for("") == ""


# --- which procedure adjudicates when NOTHING selected one --------------------


_SECOND_RULES = """
verdicts:
  aardvark:
    subject_entity: record
    sources:
      record: transaction_logs
    conditions:
      - use: record_elements/bare
        polarity: fraud_indicator
        decisive: false
"""


def _write_two_use_cases(tmp_path, default=None):
    """A pack with TWO procedures, where the second sorts FIRST alphabetically.

    ``use_cases/`` merges in directory order, so `aardvark` lands ahead of `alpha` in
    declaration order — which is exactly the shape that moved a pack's unkeyed default when a
    third procedure was authored under an earlier name. ``default`` writes the flat-root
    declaration; ``None`` writes no ``rulesets.yaml`` at all, which is every pack that
    predates it.
    """
    tmp_path = _write_shared(tmp_path)
    second = tmp_path / "use_cases" / "aardvark"
    second.mkdir(parents=True, exist_ok=True)
    (second / "rules.yaml").write_text(_SECOND_RULES)
    if default is not None:
        (tmp_path / "rulesets.yaml").write_text(f"default_ruleset: {default}\n")
    return tmp_path


def test_the_unkeyed_default_is_DECLARED_and_not_alphabetical(tmp_path):
    """The fallback must not move when a procedure is added under an earlier name.

    Asserted on the resolved OUTPUT — the weighting of the condition both rulesets import
    with OPPOSITE judgement — and not on the key, because the key is a label while the
    weighting is what changes a verdict. `aardvark` is first in declaration order and
    `alpha` is the declared default; every unkeyed lookup must reach `alpha`.
    """
    pack = load_knowledge_pack(_write_two_use_cases(tmp_path, default="alpha"))
    # Declaration order is unchanged: this fixes WHICH one answers, not the ordering.
    assert pack.ruleset_keys() == ["aardvark", "alpha"]
    assert pack.default_ruleset_key() == "alpha"
    cond = next(c for c in pack.ruleset_spec()["conditions"] if c["id"] == "bare")
    assert cond["decisive"] is True  # alpha's judgement...
    assert cond.get("polarity") != "fraud_indicator"  # ...not aardvark's
    # And naming a key still selects that key — the default is only the empty-key answer.
    other = next(
        c for c in pack.ruleset_spec("aardvark")["conditions"] if c["id"] == "bare"
    )
    assert other["decisive"] is False


def test_a_pack_declaring_no_default_keeps_FIRST_DECLARED_exactly_as_before(tmp_path):
    """The no-regression pin: every pack authored before the declaration existed.

    Two of them, because a pack that ships ONE ruleset must be unaffected for a different
    reason than a pack that ships two — the second is where the old behaviour is a real
    choice, and it has to survive unchanged rather than start erroring or returning "".
    """
    pack = load_knowledge_pack(_write_two_use_cases(tmp_path))
    assert pack.default_ruleset_key() == "aardvark"  # first declared, as before
    cond = next(c for c in pack.ruleset_spec()["conditions"] if c["id"] == "bare")
    assert cond["decisive"] is False  # aardvark's judgement

    one = tmp_path / "one"
    one.mkdir()
    single = load_knowledge_pack(_write_pack(one, with_scheme=True))
    assert single.default_ruleset_key() == "scheme"
    assert single.ruleset_spec()["subject_entity"] == "record"
    # An empty pack names nothing rather than raising: the verdict stage degrades to
    # no-verdict, which is the pre-existing behaviour for a pack that ships no rules.
    assert KnowledgePack().default_ruleset_key() == ""


def test_a_default_naming_no_ruleset_falls_back_and_SAYS_SO(tmp_path, caplog):
    """A default nobody can honour reads as a decision that was taken.

    So the fallback is the old one — never an exception and never ``""``, either of which
    would cost the run its verdict over a typo — and it is LOGGED naming what the pack
    declared and what it has. `pack_validate` reports the same thing as an error, which is
    where an author sees it before a run.
    """
    tmp_path = _write_two_use_cases(tmp_path, default="alfa")
    pack = load_knowledge_pack(tmp_path)
    with caplog.at_level("WARNING"):
        assert pack.default_ruleset_key() == "aardvark"
    assert "alfa" in caplog.text
    assert "aardvark" in caplog.text and "alpha" in caplog.text


def test_default_ruleset_is_read_from_the_PACK_ROOT_only(tmp_path):
    """Declared inside a use case it is INERT, and that is why the lint reports it.

    ``_read_use_cases`` merges a use case's ``verdicts`` up and nothing else, so a
    ``default_ruleset:`` written beside a ruleset is silently discarded — the author's
    declaration exists and does nothing. Pinned here so the lint's error is a statement
    about real loader behaviour rather than about the checker's opinion.
    """
    tmp_path = _write_two_use_cases(tmp_path)
    rules = tmp_path / "use_cases" / "aardvark" / "rules.yaml"
    rules.write_text("default_ruleset: alpha\n" + _SECOND_RULES)
    pack = load_knowledge_pack(tmp_path)
    assert pack.default_ruleset_key() == "aardvark"  # NOT alpha: the key never arrived


# --- a follow-up harvest is SCOPED, and the scope reaches the harvester -------


_RULES_WITH_FOLLOW_UP = """
verdicts:
  alpha:
    subject_entity: record
    sources:
      record: transaction_logs
      trail: access_trail
    conditions:
      - use: record_elements/bare
    follow_up_passes:
      - pass: 2
        source: trail
        harvest:
          - entity: address
            source: record
            fields: [session.addr]
            where:
              - field: session.status
                any_of: [ANOMALOUS]
                match: exact
          - entity: counterparty
            source: record
            fields: [session.peer]
"""


def test_a_harvest_row_selector_rides_through_to_the_consumer(tmp_path):
    """`where` is the SCOPE OF THE QUESTION, so dropping it changes the answer.

    A harvest reads the named source's rows, and the subject's rows are not all evidence of
    the same thing: read whole, the harvest carries the subject's ORDINARY value beside the
    anomalous one, and the follow-up pass then asks its question about both. The reply — a
    set of ordinary neighbours — is indistinguishable from a real finding, which is why the
    clause has to survive the load rather than being tidied away as an optimisation.

    The logical source names are resolved on the way out, as they always were, and an item
    that declares no selector still reports one — an EMPTY list, so a consumer reads one
    shape and a declaration dropped for being malformed cannot pass for one never made.
    """
    pack = load_knowledge_pack(_write_shared(tmp_path, rules=_RULES_WITH_FOLLOW_UP))
    entry = pack.follow_up_pass("alpha", 2)
    assert entry is not None
    assert entry["source"] == "access_trail"  # logical -> physical, as before
    scoped, unscoped = entry["harvest"]
    assert scoped["source"] == "transaction_logs"
    assert scoped["where"] == [
        {"field": "session.status", "any_of": ["ANOMALOUS"], "match": "exact"}
    ]
    assert unscoped["where"] == []


def test_a_scoped_harvest_reads_ONLY_the_rows_the_pass_declared(tmp_path):
    """The measurement, end to end through the real harvester.

    Both values are on real rows of the same source under the same subject, and only one of
    them is what the incident is about. Read unscoped, the pass would carry the ordinary
    address as well and its blast-radius answer would name every legitimate neighbour of it.
    """
    from src.follow_up import harvest_follow_up

    pack = load_knowledge_pack(_write_shared(tmp_path, rules=_RULES_WITH_FOLLOW_UP))
    logs = {
        "transaction_logs": [
            {"session": {"status": "ORDINARY", "addr": "10.0.0.1", "peer": "P1"}},
            {"session": {"status": "ANOMALOUS", "addr": "203.0.113.9", "peer": "P2"}},
        ]
    }
    entry = pack.follow_up_pass("alpha", 2)
    entities, notes = harvest_follow_up(entry, logs, None, pack)
    carried = {(e["type"], e["value"]) for e in entities}
    assert ("address", "203.0.113.9") in carried
    assert ("address", "10.0.0.1") not in carried  # the ordinary one, excluded
    # The item that declared NO selector is unaffected: absent means every row, which is
    # how every declaration written before the selector existed behaves.
    assert {v for t, v in carried if t == "counterparty"} == {"P1", "P2"}
    assert notes == []


def test_a_selector_that_matches_nothing_SAYS_it_was_scoped(tmp_path):
    """Empty because the source said nothing, and empty because none of it was this KIND of
    row, are different findings that leave the same (absent) artifact.

    So the note distinguishes them. Without it a reader sees a pass that carried no value and
    has no way to tell a source with nothing to say from a scope that excluded everything.
    """
    from src.follow_up import harvest_follow_up

    pack = load_knowledge_pack(_write_shared(tmp_path, rules=_RULES_WITH_FOLLOW_UP))
    logs = {
        "transaction_logs": [
            {"session": {"status": "ORDINARY", "addr": "10.0.0.1", "peer": "P1"}}
        ]
    }
    entities, notes = harvest_follow_up(
        pack.follow_up_pass("alpha", 2), logs, None, pack
    )
    # the unscoped item survives
    assert [e["type"] for e in entities] == ["counterparty"]
    scoped_note = next(n for n in notes if "address" in n)
    assert "scope of the question" in scoped_note
    # ...and the unscoped item's own note, when it has one, must NOT claim a scope.
    assert not [n for n in notes if "counterparty" in n]


def test_a_value_the_incident_carries_is_still_harvested_for_a_target_that_has_not_answered(
    tmp_path,
):
    """The dedup drops a repeat of pass 1's question; for a DEFERRED target there is no repeat.

    Declaring a follow-up defers its target out of every earlier pass, so while that target
    holds no rows this pass is its ONLY retrieval — and dropping the harvest because the
    incident happens to carry the same value deletes the source's evidence rather than a
    duplicate of it. Measured live on two referrals of the same procedure: the one whose
    understanding stage ALSO extracted the issuer prefix skipped its cohort pass entirely,
    while its sibling, which extracted only the card, retrieved 500 rows for the same check.
    The better extraction retrieved less, and the condition then read "source returned no
    rows" about a query that never ran.
    """
    from src.follow_up import harvest_follow_up
    from src.models.pydantic_models import ExtractedEntity

    class _Named:
        def __init__(self):
            self.extracted_entities = [
                ExtractedEntity(type="address", value="203.0.113.9")
            ]

    pack = load_knowledge_pack(_write_shared(tmp_path, rules=_RULES_WITH_FOLLOW_UP))
    entry = pack.follow_up_pass("alpha", 2)
    rows = {
        "transaction_logs": [
            {"session": {"status": "ANOMALOUS", "addr": "203.0.113.9", "peer": "P2"}}
        ]
    }

    entities, notes = harvest_follow_up(entry, rows, _Named(), pack)
    assert ("address", "203.0.113.9") in {(e["type"], e["value"]) for e in entities}, (
        "the incident's own value was dropped from a pass whose target has returned "
        "nothing, which leaves that source unqueried on every run where the understanding "
        "stage extracted the value"
    )
    carried = next((n for n in notes if "carried anyway" in n), "")
    assert entry["source"] in carried and "returned nothing yet" in carried, notes

    # ...and once that target HAS answered, the dedup is exactly what it was: the question
    # was asked of this source, so re-asking it is the defect the drop exists to prevent.
    answered = dict(rows)
    answered[entry["source"]] = [{"seen": 1}]
    again, again_notes = harvest_follow_up(entry, answered, _Named(), pack)
    assert "address" not in {e["type"] for e in again}
    assert any("no new address" in n for n in again_notes), again_notes
    assert not [n for n in again_notes if "carried anyway" in n]


def test_an_empty_harvest_says_WHICH_of_four_states_produced_it(tmp_path):
    """One artifact, four causes, four different remedies — so the note has to name the cause.

    An empty follow-up is the same sentence whichever it was, and the four states are the same
    ladder every other reader of an empty result walks: a source that did not ANSWER is absent
    from ``logs``, one that answered with nothing is present with ``[]``, one that answered with
    rows carrying no such leaf is present with rows, and one whose rows are all outside the
    pass's declared scope is present with rows the selector rejects. Measured live (job
    f1287b43): the harvest source returned 0 rows because its query carried a fabricated
    predicate, and the note published state 3's reading — the rows were retrieved and carried
    no such value — for state 2, where the source was never seen at all. The deciding pass was
    skipped and the only trace said the evidence had nothing in it.

    Read as a SET: what each assertion pins is that the four phrases are pairwise different and
    that each carries the count its remedy turns on, because "0 rows" and "500 rows, none in
    scope" license opposite next steps and neither of them is "the field was empty".
    """
    from src.follow_up import harvest_follow_up

    pack = load_knowledge_pack(_write_shared(tmp_path, rules=_RULES_WITH_FOLLOW_UP))
    entry = pack.follow_up_pass("alpha", 2)
    src = entry["harvest"][0]["source"]  # the physical name, as the note prints it

    def _note(logs, etype="address"):
        _entities, notes = harvest_follow_up(entry, logs, None, pack)
        return next(n for n in notes if f"no new {etype} value" in n)

    # 1 — the source is not in `logs` at all: it was never queried, or it did not answer.
    absent = _note({})
    assert "is not among the retrieved sources" in absent, absent
    assert "it was not queried, or it did not answer" in absent, absent

    # 2 — present and empty: the source answered, and its answer was nothing.
    empty = _note({src: []})
    assert "answered with 0 row(s)" in empty, empty

    # 3 — rows came back and none of them carry the declared leaf. The scope is stated with
    # its own count, because a `where` that kept every row is not the reason this was empty.
    no_leaf = _note(
        {src: [{"session": {"status": "ANOMALOUS", "unrelated": "x"}}]}
    )
    assert "answered with 1 row(s) (1 in the pass's declared scope)" in no_leaf, no_leaf
    assert "none carried a value at those field(s)" in no_leaf, no_leaf

    # 4 — rows came back carrying the leaf, and the pass's own selector rejects all of them:
    # the question's SCOPE is what is empty, not the source and not the field.
    out_of_scope = _note({src: [{"session": {"status": "ORDINARY", "addr": "203.0.113.9"}}]})
    assert "answered with 1 row(s) and none of them are the rows the pass declares" in (
        out_of_scope
    ), out_of_scope
    assert "the question's own scope is what came back empty" in out_of_scope, out_of_scope

    # The four readings must be mutually exclusive: a fix that collapsed any two of them back
    # into one sentence is the defect this test exists for.
    assert len({absent, empty, no_leaf, out_of_scope}) == 4

    # ...and a harvest item with NO `where` never claims a scope, in any of the states, or the
    # note invents a restriction the pack did not declare.
    for logs in ({}, {src: []}, {src: [{"session": {"peer": None}}]}):
        unscoped = _note(logs, etype="counterparty")
        assert "scope" not in unscoped, unscoped


def test_a_COMBINATION_harvest_reads_its_OWN_row_selector_for_that_reason(tmp_path):
    """Same ladder from the co-occurrence branch, and its `where` is the ITEM's, not a sibling's.

    The two branches are separate loops over the same harvest list, and only the ordinary one
    binds `where` to a local. So the combination branch has to read `item['where']` — reading
    the local would be either undefined on the first item or, worse, the PREVIOUS item's scope,
    which reports one question's emptiness under another question's restriction.
    """
    from src.follow_up import harvest_follow_up

    rules = _RULES_WITH_TOGETHER.replace(
        "          - source: record\n",
        "          - source: record\n"
        "            where:\n"
        "              - field: alert.status\n"
        "                any_of: [ANOMALOUS]\n"
        "                match: exact\n",
    )
    pack, entry = _together_pack(tmp_path, rules=rules)
    assert entry["harvest"][0]["where"], "the fixture must actually declare a selector"
    src = entry["harvest"][0]["source"]
    rows = {
        src: [
            {
                "alert": {
                    "status": "ORDINARY",
                    "parties": [{"doc": "D1", "unit": "U1"}],
                }
            }
        ]
    }
    _entities, notes = harvest_follow_up(entry, rows, None, pack)
    combination = next(n for n in notes if "no new document value" in n)
    assert "as part of a complete combination" in combination, combination
    assert "none of them are the rows the pass declares" in combination, combination
    # The sibling ordinary item declares no selector, and its own note must not borrow this one.
    sibling = next(n for n in notes if "no new counterparty value" in n)
    assert "scope" not in sibling, sibling


_RULES_WITH_SHARED_PASS = """
verdicts:
  alpha:
    subject_entity: record
    sources:
      record: transaction_logs
      trail: access_trail
      reference: entity_reference
    conditions:
      - use: record_elements/bare
    follow_up_passes:
      - pass: 2
        source: [trail, reference, trail]
        harvest:
          - entity: counterparty
            source: record
            fields: [session.peer]
        purpose:
          trail: "what the counterparty DID"
          reference: "what KIND of counterparty it is"
"""


def test_one_follow_up_pass_may_target_SEVERAL_sources_off_ONE_harvest(tmp_path):
    """A pass number is a scarce slot, and two questions can share one harvest.

    ``jobs.max_retrieval_passes`` bounds how many passes a run gets, one entry occupied one
    pass, so a procedure with three deferred questions had one it could not ask at all —
    measured: the loser returned 0 rows against a ground truth of 2 and its condition read
    `unknown / field absent` where the answer was a PASS quoting a value. The passes it lost
    to harvested the IDENTICAL leaves from the IDENTICAL source, i.e. they were one pass
    wearing two numbers. So targets are a list, resolved like any logical name, deduped
    (a repeat would query the same source twice off one harvest) and order-preserving.
    """
    pack = load_knowledge_pack(_write_shared(tmp_path, rules=_RULES_WITH_SHARED_PASS))
    entry = pack.follow_up_pass("alpha", 2)
    assert entry["sources"] == ["access_trail", "entity_reference"]
    # `source` stays the FIRST target: a consumer written before this keeps working on a
    # one-target entry, and the two readings coincide exactly where the old shape was legal.
    assert entry["source"] == "access_trail"


def test_a_shared_pass_gives_each_TARGET_its_own_purpose(tmp_path):
    """The purpose seeds the request text a backend query is WRITTEN FROM, so it is per target.

    What the targets share is the harvest, not the question. One text over two sources asks
    the second one the first one's question — which on a reference table keyed differently
    from an event log is a query that can only answer zero rows. Keys resolve through the
    same `sources:` map as the targets, so a mapping is written in logical names.
    """
    pack = load_knowledge_pack(_write_shared(tmp_path, rules=_RULES_WITH_SHARED_PASS))
    entry = pack.follow_up_pass("alpha", 2)
    assert entry["purposes"] == {
        "access_trail": "what the counterparty DID",
        "entity_reference": "what KIND of counterparty it is",
    }
    # A plain string still means "every target" — every declaration written before a mapping
    # was possible — and the two keys never both carry text for the same source.
    assert entry["purpose"] == ""
    plain = load_knowledge_pack(
        _write_shared(tmp_path, rules=_RULES_WITH_FOLLOW_UP)
    ).follow_up_pass("alpha", 2)
    assert plain["purposes"] == {}  # always present, so the consumer reads one shape


def test_a_shared_pass_defers_until_EVERY_target_has_answered(tmp_path):
    """The dedup's premise is per TARGET, and reading it collectively deletes evidence.

    Dropping a value the incident already carries stops a pass re-asking pass 1's question,
    and declaring a follow-up defers its target out of every earlier pass — so while a target
    holds no rows, the value is new TO IT. With several targets that has to hold for each of
    them: the sibling that already replied is not evidence that the one still waiting was
    ever asked, and reading "answered" as "any of them answered" would drop the harvest and
    with it that source's only retrieval.
    """
    from src.follow_up import harvest_follow_up
    from src.models.pydantic_models import ExtractedEntity

    class _Named:
        def __init__(self):
            self.extracted_entities = [
                ExtractedEntity(type="counterparty", value="P2")
            ]

    pack = load_knowledge_pack(_write_shared(tmp_path, rules=_RULES_WITH_SHARED_PASS))
    entry = pack.follow_up_pass("alpha", 2)
    rows = {"transaction_logs": [{"session": {"peer": "P2"}}]}

    # One target answered, the other not: the value is still carried, for the one waiting.
    partial = dict(rows)
    partial["access_trail"] = [{"seen": 1}]
    entities, notes = harvest_follow_up(entry, partial, _Named(), pack)
    assert [e["value"] for e in entities] == ["P2"], (
        "a value the incident carries was dropped from a shared pass because ONE of its "
        "targets had replied, which leaves the other target unqueried"
    )
    carried = next((n for n in notes if "carried anyway" in n), "")
    assert "entity_reference" in carried and "access_trail" in carried, notes

    # Both answered: the dedup is exactly what it was, because both were asked.
    both = dict(partial)
    both["entity_reference"] = [{"seen": 1}]
    again, again_notes = harvest_follow_up(entry, both, _Named(), pack)
    assert again == []
    assert any("no new counterparty" in n for n in again_notes), again_notes


_RULES_WITH_TOGETHER = """
verdicts:
  alpha:
    subject_entity: record
    sources:
      record: transaction_logs
      trail: access_trail
    conditions:
      - use: record_elements/bare
    follow_up_passes:
      - pass: 2
        source: trail
        harvest:
          - source: record
            together:
              - entity: document
                fields: [alert.parties.doc]
              - entity: org_unit
                fields: [alert.parties.unit]
          - entity: counterparty
            source: record
            fields: [session.peer]
"""


def test_a_harvest_may_declare_that_its_VALUES_OCCURRED_TOGETHER(tmp_path):
    """The loader is the whole mechanism here, and it silently dropped the declaration.

    A co-occurrence item declares its entities PER COMPONENT and none of its own, so the
    loader's "an item needs `entity` and `fields`" test read a correct one as empty and
    dropped it — and with it the pass, whose only harvest it was. What that leaves is a run
    that skips a pass it declared, which is indistinguishable from a source with nothing to
    say. `together` is normalised here (each component's `capture` always present) so the
    consumer reads one shape, and it is present as an EMPTY list on an ordinary item: absent
    would make "these values are independent" — the reading of every pack written before
    combinations could be declared — look like a declaration that failed to parse.
    """
    pack = load_knowledge_pack(_write_shared(tmp_path, rules=_RULES_WITH_TOGETHER))
    entry = pack.follow_up_pass("alpha", 2)
    combination, ordinary = entry["harvest"]
    assert combination["together"] == [
        {"entity": "document", "fields": ["alert.parties.doc"], "capture": ""},
        {"entity": "org_unit", "fields": ["alert.parties.unit"], "capture": ""},
    ]
    assert combination["source"] == "transaction_logs"  # logical -> physical, as ever
    assert ordinary["together"] == []

    # A component the loader cannot read is dropped from the combination, not silently
    # kept as a half-declaration — and an item left with no readable component and no
    # `entity`/`fields` of its own is dropped whole, exactly as before this key existed.
    half = _RULES_WITH_TOGETHER.replace(
        "                fields: [alert.parties.unit]", ""
    )
    entry = load_knowledge_pack(_write_shared(tmp_path, rules=half)).follow_up_pass(
        "alpha", 2
    )
    assert [c["entity"] for c in entry["harvest"][0]["together"]] == ["document"]
    none_left = _RULES_WITH_TOGETHER.replace(
        "              - entity: document\n                fields: [alert.parties.doc]\n",
        "",
    ).replace("                fields: [alert.parties.unit]", "")
    entry = load_knowledge_pack(
        _write_shared(tmp_path, rules=none_left)
    ).follow_up_pass("alpha", 2)
    assert [h.get("entity") for h in entry["harvest"]] == ["counterparty"]


def _together_pack(tmp_path, rules=_RULES_WITH_TOGETHER):
    pack = load_knowledge_pack(_write_shared(tmp_path, rules=rules))
    return pack, pack.follow_up_pass("alpha", 2)


def test_the_harvested_COMBINATIONS_are_the_ones_the_rows_carried(tmp_path):
    """The measurement, end to end: the cross product is not the evidence.

    Read per type these rows say three documents and two units, and a follow-up query AND-ing
    those two lists asks for all six pairings — of which three never occurred. That failure is
    the inverse of every other one in retrieval: the query comes back FULL, so no check asking
    "did anything answer" fires, and the row cap truncates on other parties' rows before the
    subject's. The container is what bounds it — a combination is only ever read from inside
    the tightest path the declared fields share, here one element of `alert.parties`.
    """
    from src.follow_up import harvest_value_tuples

    pack, entry = _together_pack(tmp_path)
    logs = {
        "transaction_logs": [
            {
                "alert": {
                    "parties": [
                        {"doc": "D1", "unit": "U1"},
                        {"doc": "D2", "unit": "U2"},
                    ]
                }
            },
            {"alert": {"parties": [{"doc": "D3", "unit": "U1"}]}},
            # The same pairing again, from another row: one combination, not two.
            {"alert": {"parties": [{"doc": "D1", "unit": "U1"}]}},
        ]
    }
    tuples, notes = harvest_value_tuples(entry, logs, None, pack)
    assert [tuple(part["value"] for part in tup) for tup in tuples] == [
        ("D1", "U1"),
        ("D2", "U2"),
        ("D3", "U1"),
    ], "the combinations are not the ones the containers carried"
    # Order is the pack's declaration order, because that is what the target's columns are
    # resolved against — a set would make the two components interchangeable.
    assert [part["type"] for part in tuples[0]] == ["document", "org_unit"]
    assert notes == []
    # ...and the per-type lists a target binding only ONE component filters on are DERIVED
    # from these accepted tuples, so the two views cannot describe different evidence.
    from src.follow_up import harvest_follow_up

    entities, _ = harvest_follow_up(entry, logs, None, pack)
    by_type = {}
    for ent in entities:
        by_type.setdefault(ent["type"], set()).add(ent["value"])
    assert by_type["document"] == {"D1", "D2", "D3"}
    assert by_type["org_unit"] == {"U1", "U2"}


def test_a_combination_is_ARITY_AGNOSTIC_and_never_a_pair(tmp_path):
    """A key made of three fields is as ordinary as one made of two.

    Nothing in the engine knows the word "pair": the arity is whatever the pack declared, or
    the mechanism would have to be written a second time the first time a domain keys on
    three columns — and a generic engine cannot carry one domain's arity.
    """
    from src.follow_up import harvest_value_tuples

    rules = _RULES_WITH_TOGETHER.replace(
        "                fields: [alert.parties.unit]",
        "                fields: [alert.parties.unit]\n"
        "              - entity: address\n"
        "                fields: [alert.parties.addr]",
    )
    pack, entry = _together_pack(tmp_path, rules=rules)
    logs = {
        "transaction_logs": [
            {
                "alert": {
                    "parties": [
                        {"doc": "D1", "unit": "U1", "addr": "10.0.0.1"},
                        {"doc": "D2", "unit": "U1", "addr": "10.0.0.2"},
                    ]
                }
            }
        ]
    }
    tuples, _ = harvest_value_tuples(entry, logs, None, pack)
    assert [tuple(part["value"] for part in tup) for tup in tuples] == [
        ("D1", "U1", "10.0.0.1"),
        ("D2", "U1", "10.0.0.2"),
    ]
    assert [part["type"] for part in tuples[0]] == ["document", "org_unit", "address"]


def test_a_container_naming_only_SOME_components_carries_no_combination_and_SAYS_SO(
    tmp_path,
):
    """A partial combination is not a narrower one, it is a different question.

    Keeping the component that did resolve would put this container's value beside some other
    container's — the fabricated pairing in miniature, and the one shape where dropping
    evidence is the conservative direction. So the whole combination goes and the count is
    reported: a pass that carried two of five records' keys and one that found two records
    leave the same artifact otherwise.
    """
    from src.follow_up import harvest_value_tuples

    pack, entry = _together_pack(tmp_path)
    logs = {
        "transaction_logs": [
            {
                "alert": {
                    "parties": [
                        {"doc": "D1", "unit": "U1"},
                        {"doc": "D2"},  # no unit: nothing to pair D2 with
                        {"unit": "U9"},  # and no document for U9
                    ]
                }
            }
        ]
    }
    tuples, notes = harvest_value_tuples(entry, logs, None, pack)
    assert [tuple(part["value"] for part in tup) for tup in tuples] == [("D1", "U1")]
    note = next(n for n in notes if "only some of" in n)
    assert "2 of 3 record(s)" in note
    assert "document + org_unit" in note


def test_one_component_already_on_the_incident_does_not_drop_the_COMBINATION(tmp_path):
    """The dedup is per combination, because the combination is what is new.

    A pass must not re-ask pass 1's question, and for a single value "the incident carries it"
    settles that. For a combination it does not: the incident knowing the document says
    nothing about which unit it was used from, which is the whole question this pass defers.
    Reading the dedup per component would drop every combination whose subject the
    understanding stage extracted — i.e. exactly the ones about the subject.
    """
    from src.follow_up import harvest_value_tuples
    from src.models.pydantic_models import ExtractedEntity

    class _Named:
        def __init__(self, *values):
            self.extracted_entities = [
                ExtractedEntity(type="document", value=v) for v in values
            ]

    pack, entry = _together_pack(tmp_path)
    logs = {
        "transaction_logs": [
            {"alert": {"parties": [{"doc": "D1", "unit": "U1"}]}},
        ],
        # The pass's own target has answered, so the deferral no longer holds the dedup off.
        "access_trail": [{"seen": 1}],
    }
    tuples, _ = harvest_value_tuples(entry, logs, _Named("D1"), pack)
    assert [tuple(p["value"] for p in tup) for tup in tuples] == [("D1", "U1")]

    # Every component known IS the repeat the dedup exists to stop.
    both = _Named("D1")
    both.extracted_entities.append(ExtractedEntity(type="org_unit", value="U1"))
    tuples, notes = harvest_value_tuples(entry, logs, both, pack)
    assert tuples == []
    assert any("already on the incident" in n for n in notes), notes


def test_a_playbooks_correlation_spec_carries_the_use_case_that_selects_its_ruleset(
    tmp_path,
):
    """The wire between the two: the stage that matches a playbook must be able to say
    WHICH procedure the incident belongs to.

    The loader already tags every playbook document with its use case; ``correlation_specs``
    dropped it, so the only stage that knows the procedure could not tell the verdict stage.
    A flat-root playbook belongs to no use case and must report ``""`` rather than guessing.
    """
    tmp_path = _write_shared(tmp_path)
    pb_dir = tmp_path / "use_cases" / "alpha" / "playbooks"
    pb_dir.mkdir(parents=True, exist_ok=True)
    (pb_dir / "alpha_pb.md").write_text(
        """---
playbook_id: PB-ALPHA-001
title: Alpha procedure
correlation:
  keys: [record]
  time_window: within:24h
---

## What to do

Follow the alpha procedure.
"""
    )
    pack = load_knowledge_pack(tmp_path)
    spec = next(s for s in pack.correlation_specs() if s["playbook_id"] == "PB-ALPHA-001")
    assert spec["use_case"] == "alpha"
    # ...and that tag round-trips into a real ruleset selection.
    assert pack.ruleset_key_for(spec["use_case"]) == "alpha"


def test_an_unknown_import_RAISES_rather_than_dropping_the_condition(tmp_path):
    """DELIBERATELY FATAL, and the only fatal path in pack loading.

    A dropped condition is not an error at any later point — it is a check that is never
    evaluated, which reads in the report exactly like one whose source returned no rows.
    The pack spends thousands of words distinguishing those two, so a typo in a `use:` line
    must fail at LOAD, naming the ruleset and the available ids.
    """
    rules = _RULES_WITH_IMPORTS.replace(
        "record_elements/bare", "record_elements/barbone"
    )
    with pytest.raises(ValueError) as e:
        load_knowledge_pack(_write_shared(tmp_path, rules=rules))
    msg = str(e.value)
    assert "barbone" in msg  # the bad ref
    assert "alpha" in msg  # the ruleset that asked
    assert "record_elements/bare" in msg  # what it could have meant


def test_shared_concepts_are_visible_to_every_use_case(tmp_path):
    """A concept about the DATA is untagged, so no use case owns it or is denied it."""
    pack = load_knowledge_pack(_write_shared(tmp_path))
    ids = [
        (d.get("metadata") or {}).get("concept_id") for d in pack.concepts_for("alpha")
    ]
    assert "record_elements" in ids
    # ...and to a use case that does not exist in this pack at all: shared means shared.
    assert "record_elements" in [
        (d.get("metadata") or {}).get("concept_id")
        for d in pack.concepts_for("nonexistent")
    ]
    # It is RAG-ingestible like any other concept, and carries NO use_case tag.
    doc = next(
        d
        for d in pack.concept_documents
        if (d.get("metadata") or {}).get("concept_id") == "record_elements"
    )
    assert not (doc.get("metadata") or {}).get("use_case")
    assert doc["type"] == "concept"


def test_a_use_case_concept_of_the_same_id_wins_over_the_shared_one(tmp_path):
    """A use case may reinterpret an element; naming the id must get ITS version."""
    tmp_path = _write_shared(tmp_path)
    own = tmp_path / "use_cases" / "alpha" / "concepts"
    own.mkdir(parents=True, exist_ok=True)
    (own / "record_elements.md").write_text(
        "---\nconcept_id: record_elements\ntitle: Alpha's reading\n---\n\nSpecific.\n"
    )
    pack = load_knowledge_pack(tmp_path)
    got = pack.concepts_for("alpha", ids=["record_elements"])
    assert len(got) == 1
    assert got[0]["metadata"]["use_case"] == "alpha"


def test_a_pack_with_no_shared_folder_is_completely_unaffected(tmp_path):
    """Every pack predating the shared layer must load byte-identically."""
    pack = load_knowledge_pack(_write_pack(tmp_path, with_scheme=True))
    assert pack.shared_checks == {}
    assert KnowledgePack().shared_checks == {}
    # `scheme_spec` returns the ruleset untouched when nothing imports.
    spec = pack.ruleset_spec("scheme")
    assert spec is pack.rulesets["verdicts"]["scheme"]


def test_malformed_shared_check_file_does_not_break_the_pack(tmp_path):
    """Same rule as the schema files: one bad file must not empty the pack."""
    tmp_path = _write_shared(tmp_path)
    d = tmp_path / "shared" / "checks"
    (d / "listish.yaml").write_text("- 1\n- 2\n")
    (d / "not_mappings.yaml").write_text("some_check: not-a-mapping\n")
    pack = load_knowledge_pack(tmp_path)
    assert "record_elements/bare" in pack.shared_checks
    assert "not_mappings/some_check" not in pack.shared_checks


# --- report vocabulary: two scopes, resolved per SLOT -------------------------


def _write_reporting(tmp_path, root: str, scoped: str):
    """A pack with BOTH a domain-wide reporting.yaml and a use-case-scoped one."""
    _write_shared(tmp_path)
    (tmp_path / "reporting.yaml").write_text(root)
    (tmp_path / "use_cases" / "alpha" / "reporting.yaml").write_text(scoped)
    return tmp_path


def test_a_scoped_phrase_overrides_one_slot_and_INHERITS_the_rest(tmp_path):
    """`phrases` is a bag of independent named slots, so it merges per sub-key.

    Resolved whole-key, declaring a SINGLE scoped phrase silently discarded the entire
    domain-wide map — and the loss is invisible, because every slot falls back to the engine's
    own generic sentence. The report then reads as one from a pack that never declared any
    wording, rather than as one whose wording was dropped, so nobody looks for a bug. A pack
    that ships no root `reporting.yaml` never exercises the merge, which is why this survived.
    """
    pack = load_knowledge_pack(
        _write_reporting(
            tmp_path,
            root=(
                "phrases:\n"
                '  containment_target: "ROOT target"\n'
                '  containment_action: "ROOT action"\n'
            ),
            scoped='phrases:\n  containment_target: "SCOPED target (§9.2)"\n',
        )
    )
    assert pack.report_phrase("containment_target", "", use_case="alpha") == (
        "SCOPED target (§9.2)"
    )
    # THE ASSERTION THAT MATTERS: the slot the use case did NOT re-declare survives.
    assert pack.report_phrase("containment_action", "", use_case="alpha") == "ROOT action"
    # A use case with no reporting file of its own sees the root, unchanged.
    assert pack.report_phrase("containment_target", "", use_case="beta") == "ROOT target"
    # And an undeclared slot still returns the ENGINE's default, never "".
    assert (
        pack.report_phrase("no_such_slot", "engine default", use_case="alpha")
        == "engine default"
    )


def test_scoped_phases_REPLACE_the_root_list_rather_than_merging(tmp_path):
    """A list replaces, and here that is not a convention — it is the only safe reading.

    `phases` is an ORDERED decision procedure: checked in order, first keyword hit wins, so the
    order IS the semantics. Merging two authored orderings produces a precedence nobody wrote,
    and the symptom is a chronology event filed under the wrong phase — which reads as a
    presentation quirk, not as a merge bug. Same rule as a `use:` check import's list override.
    """
    pack = load_knowledge_pack(
        _write_reporting(
            tmp_path,
            root=(
                "phases:\n"
                "  - keywords: [alert]\n"
                '    label: "ROOT alert phase"\n'
                "  - keywords: [session]\n"
                '    label: "ROOT session phase"\n'
            ),
            scoped=("phases:\n  - keywords: [refund]\n    label: \"SCOPED refund phase\"\n"),
        )
    )
    assert [r["label"] for r in pack.phase_rules("alpha")] == ["SCOPED refund phase"]
    assert [r["label"] for r in pack.phase_rules("beta")] == [
        "ROOT alert phase",
        "ROOT session phase",
    ]


_FORMS_GLOSSARY = """
entities:
  - type: org_unit
    description: Point of sale, nine characters.
    value_pattern: "^[A-Z0-9]{2,9}$"
    value_forms:
      - name: full
        pattern: "^[A-Z]{3}[A-Z0-9]{2}[0-9][A-Z0-9]{3}$"
      - name: location_code
        pattern: "^[A-Z]{3}$"
        match: "{value}??????"
      - name: corporate_code
        pattern: "^[A-Z0-9]{2}$"
        match: "???{value}????"
      - name: no_match_declared
        pattern: "^[0-9]{4}$"
      - name: template_without_the_value
        pattern: "^[0-9]{5}$"
        match: "?????????"
"""


def _write_forms_pack(tmp_path, catalog=_CATALOG):
    (tmp_path / "entity_glossary.yaml").write_text(_FORMS_GLOSSARY)
    (tmp_path / "source_catalog.yaml").write_text(catalog)
    return tmp_path


def test_value_match_pattern_places_a_PARTIAL_value_inside_the_stored_one(tmp_path):
    """A form's `match` is a fixed-width WINDOW, and it is the stem read backwards.

    `stem` takes a shorter value OUT of a longer stored one; `match` puts a shorter value
    somewhere INSIDE it. Both exist because one entity type carries several surface forms and
    which precision a column stores is declared nowhere per column — but the failure they
    answer is the same and it is invisible: a real literal in a valid predicate that matches no
    row, reported as a source with nothing to say. Measured live on an incident naming two
    organisational units by their leading 3-character segment against columns storing the
    9-character identifier: 0 rows on every route, every stage green.

    The pattern is returned SUBSTITUTED and in the canonical `?`/`*` spelling, so no caller has
    to know the template and each backend translates only the wildcards.
    """
    pack = load_knowledge_pack(_write_forms_pack(tmp_path))
    # The three-letter location segment sits at the FRONT; the two-character corporate one is
    # in the middle. A pack states each position once, per form.
    assert pack.value_match_pattern("org_unit", "LBV") == "LBV??????"
    assert pack.value_match_pattern("org_unit", "2A") == "???2A????"
    # A COMPLETE value needs no widening: it classifies as the full form, which declares no
    # `match`, so the caller compares it exactly as written.
    assert pack.value_match_pattern("org_unit", "FFF1G07NP") is None
    # Three ways there is nothing to say, and all three are `None` rather than a guess:
    # a form declaring no `match`, a value matching no declared form, and an entity type with
    # no forms at all.
    assert pack.value_match_pattern("org_unit", "1234") is None
    assert pack.value_match_pattern("org_unit", "lbv-!") is None
    assert pack.value_match_pattern("document", "D1") is None
    assert pack.value_match_pattern("org_unit", "") is None
    assert pack.value_match_pattern("org_unit", None) is None
    # A template that does not name {value} is REFUSED, not rendered: it would match a whole
    # window of the column — every row of the population instead of this subject.
    assert pack.value_match_pattern("org_unit", "12345") is None


def test_default_lookup_days_is_a_DECLARATION_and_never_an_engine_constant(tmp_path):
    """How far back to read when the incident states no time — a domain fact, so a pack key.

    Without it the generator was simply asked to choose a window, and it chose per query
    (measured on one live undated incident: 29 queries carrying eight different windows,
    several of them the single day the run happened to start on). A window is the hard scope of
    every retrieval, so an invented one silently decides what the investigation can see.

    `None` where nothing is declared, and `None` again for anything that is not a positive
    whole number of days — the engine falls through to the config key and invents no depth of
    its own. Zero and a negative are the shapes that matter: a `0` read as a number would make
    every query a point in time.
    """
    catalog = "retrieval:\n  default_lookup_days: 30\n" + _CATALOG
    assert load_knowledge_pack(_write_forms_pack(tmp_path, catalog)).default_lookup_days() == 30
    # A pack declaring nothing — every pack until one opts in.
    assert load_knowledge_pack(_write_pack(tmp_path)).default_lookup_days() is None
    for bad in ("0", "-7", "'thirty'", "true", "1.5.2"):
        pack = load_knowledge_pack(
            _write_forms_pack(tmp_path, f"retrieval:\n  default_lookup_days: {bad}\n" + _CATALOG)
        )
        assert pack.default_lookup_days() is None, bad
    # A whole number written as a string is still a whole number of days.
    pack = load_knowledge_pack(
        _write_forms_pack(tmp_path, "retrieval:\n  default_lookup_days: '45'\n" + _CATALOG)
    )
    assert pack.default_lookup_days() == 45


_RULES_WITH_ENTRY_SIGNALS = """
verdicts:
  alpha:
    subject_entity: record
    sources:
      record: transaction_logs
      trail: access_trail
    conditions:
      - use: record_elements/bare
    entry_signals:
      - id: repeat_rejection
        direction: ANTECEDENT
        opens_with: {entity: record}
        when:
          source: trail
          where:
            - field: outcome
              any_of: [rejected]
              match: exact
          min_rows: 3
        window: LOOKBACK:30D
        strength: 0.6
        base_rate: {fires_on: 3, of: 27, measured: "2026-08-19"}
        note: One identity rejected repeatedly before the incident.
      - id: reads_a_physical_name
        direction: consequent
        opens_with: {entity: record}
        when:
          source: settlement_report
"""


def test_an_entry_signal_normalises_to_ONE_shape_the_consumer_can_read(tmp_path):
    """Every declared member arrives, and the absent ones arrive as a stated default.

    An inbound signal is read opportunistically off whatever a run happened to retrieve, so
    the consumer has no second chance to ask the pack what a missing key meant. `where` is
    therefore always a list (empty = every row of that source, exactly as a harvest's is),
    `window` always a mode (`inherit`, never ""), and `base_rate` always a mapping — a
    declaration dropped for being malformed must not be able to pass for one never made.

    `direction` and `window` fold case because they are vocabularies, not values; the source
    resolves through THIS ruleset's own `sources:` map, as a follow-up pass's target does.
    """
    pack = load_knowledge_pack(_write_shared(tmp_path, rules=_RULES_WITH_ENTRY_SIGNALS))
    first, second = pack.entry_signals("alpha")
    assert first == {
        "id": "repeat_rejection",
        "direction": "antecedent",
        "entity": "record",
        "source": "access_trail",  # logical -> physical
        "where": [{"field": "outcome", "any_of": ["rejected"], "match": "exact"}],
        "min_rows": 3,
        "window": "lookback:30d",
        "strength": 0.6,
        "base_rate": {"fires_on": 3, "of": 27, "measured": "2026-08-19"},
        "auto_probe": False,
        "note": "One identity rejected repeatedly before the incident.",
    }
    # Everything the second entry declines to declare, as a shape rather than a None.
    assert second["where"] == []
    assert second["min_rows"] == 1
    assert second["window"] == "inherit"
    assert second["strength"] == 0.0
    assert second["base_rate"] == {}
    assert second["auto_probe"] is False
    assert second["note"] == ""


def test_an_entry_signal_source_may_be_a_PHYSICAL_catalog_name(tmp_path):
    """A signal names sources INBOUND, so the declaring ruleset need not map every one.

    A logical name is resolved through the declaring ruleset's own map; anything else passes
    through unchanged, because the shape this mechanism exists for is "recognise my fraud in
    somebody else's rows" and those rows arrive under their catalog name. Resolving only
    logical names and dropping the rest would confine every signal to sources the declaring
    procedure already retrieves — which is the pair of procedures that needs no link.
    """
    pack = load_knowledge_pack(_write_shared(tmp_path, rules=_RULES_WITH_ENTRY_SIGNALS))
    _first, second = pack.entry_signals("alpha")
    assert second["source"] == "settlement_report"
    # And it is NOT a hard dependency: the ruleset's own `sources:` map is untouched by it.
    assert set(pack.ruleset_spec("alpha")["sources"].values()) == {
        "transaction_logs",
        "access_trail",
    }


def test_a_ruleset_declaring_no_entry_signals_gets_an_EMPTY_LIST(tmp_path):
    """Absence is a no-op — the guarantee every pack that exists today relies on.

    `entry_signals` is additive: a procedure declaring none contributes no candidate, and the
    accessor says so with the same empty list a ruleset that does not exist gets. There is no
    third state here on purpose, because a `None` would reach a consumer that iterates.
    """
    pack = load_knowledge_pack(_write_shared(tmp_path))
    assert pack.entry_signals("alpha") == []
    assert pack.entry_signals("no_such_ruleset") == []
    assert pack.entry_signals("") == []
    # Not a list is not a declaration either.
    rules = _RULES_WITH_ENTRY_SIGNALS.split("    entry_signals:")[0] + (
        "    entry_signals: {id: not_a_list}\n"
    )
    inert = load_knowledge_pack(_write_shared(tmp_path, rules=rules))
    assert inert.entry_signals("alpha") == []


def test_an_entry_signal_the_engine_cannot_READ_is_dropped_and_SAYS_which_part(
    tmp_path, caplog
):
    """Three members, three different impossibilities, and none of them can be defaulted.

    Without an `id` there is nothing for a report to cite; without a subject entity there is
    no leg to open (a leg is opened by a subject VALUE and by nothing else); without a source
    there are no rows to read. Each reads as an active declaration while being incapable of
    firing, which is the shape this whole mechanism exists to avoid — so the entry is dropped
    and the log names the three fields, rather than a silent skip that leaves an author
    reading a signal they believe is armed.

    A mismatched entity TYPE is a different matter and is NOT dropped here: `pack_validate`
    errors on it, because the loader must not silently delete a declaration an author can see
    in the file, and a value it can read is a value it can report.
    """
    rules = """
verdicts:
  alpha:
    subject_entity: record
    sources:
      record: transaction_logs
    conditions:
      - use: record_elements/bare
    entry_signals:
      - direction: antecedent
        opens_with: {entity: record}
        when: {source: record}
      - id: no_subject_to_open_a_leg_on
        direction: antecedent
        when: {source: record}
      - id: no_rows_to_read
        direction: antecedent
        opens_with: {entity: record}
      - id: opens_on_another_rulesets_subject
        direction: antecedent
        opens_with: {entity: document}
        when: {source: record}
"""
    with caplog.at_level("WARNING"):
        pack = load_knowledge_pack(_write_shared(tmp_path, rules=rules))
        signals = pack.entry_signals("alpha")
    assert [s["id"] for s in signals] == ["opens_on_another_rulesets_subject"]
    # Passed through for the validator to fail on, not deleted here.
    assert signals[0]["entity"] == "document"
    dropped = [r for r in caplog.records if "DROPPED" in r.getMessage()]
    assert len(dropped) == 3
    assert all("alpha" in r.getMessage() for r in dropped)


def test_a_signal_threshold_is_a_FLOOR_and_junk_never_becomes_a_zero(tmp_path):
    """`min_rows: 0` would mean "fires when the source returned nothing".

    A signal's threshold decides whether the shape is present, so a zero — or an unparseable
    value read as one — turns every reachable sibling into a firing candidate on an empty
    result, which is the `not_probed` state manufactured out of no evidence at all. The floor
    is 1. `strength` fails the other way, to 0.0, because it is what a threshold compares
    against: an unreadable weight must not clear a bar, and a pack that meant something by it
    is told by `pack_validate` rather than having a number guessed for it here.
    """
    rules = """
verdicts:
  alpha:
    subject_entity: record
    sources:
      record: transaction_logs
    conditions:
      - use: record_elements/bare
    entry_signals:
      - id: zero_rows_is_not_a_threshold
        opens_with: {entity: record}
        when: {source: record, min_rows: 0}
        strength: strong
      - id: junk_threshold
        opens_with: {entity: record}
        when: {source: record, min_rows: several}
        strength: 1.5
"""
    pack = load_knowledge_pack(_write_shared(tmp_path, rules=rules))
    floored, junk = pack.entry_signals("alpha")
    assert floored["min_rows"] == 1
    assert floored["strength"] == 0.0
    assert junk["min_rows"] == 1
    # Out of range is carried VERBATIM — the bound is the validator's to report, and
    # clamping it here would hide the declaration the author has to fix.
    assert junk["strength"] == 1.5
    # A direction nobody declared is empty, not a default one of the two: guessing a causal
    # direction reverses which window a referral would carry.
    assert floored["direction"] == ""


_RULES_WITH_LINK_ESCALATION = """
verdicts:
  alpha:
    subject_entity: record
    sources:
      record: transaction_logs
    conditions:
      - use: record_elements/bare
    link_escalation:
      mode: SEMI_AUTO
      from:
        beta: Auto
        gamma: '  '
  beta:
    subject_entity: record
    sources:
      record: transaction_logs
    conditions:
      - use: record_elements/bare
    link_escalation:
      from:
        alpha: planned
"""


def test_a_declared_escalation_mode_arrives_FOLDED_and_per_source(tmp_path):
    """The declaration a run reads, in the one shape the resolver takes.

    A mode is a vocabulary and not a value, so it folds case here for the same reason
    `direction` does — an author writing `Auto` means the word, and a run that read it
    literally would fall through to the default and spend nothing while the file says
    otherwise. A blank per-source entry is DROPPED rather than kept as `""`, because the
    resolver reads absence as "fall through to the next layer" and an empty string would be a
    third state nobody declared.
    """
    pack = load_knowledge_pack(
        _write_shared(tmp_path, rules=_RULES_WITH_LINK_ESCALATION)
    )
    assert pack.link_escalation("alpha") == {
        "mode": "semi_auto",
        "from": {"beta": "auto"},
    }
    # A per-pair override with no general mode beside it is a complete declaration: the
    # general case then falls through to the deployment's config, which is the layer under it.
    assert pack.link_escalation("beta") == {"from": {"alpha": "planned"}}


def test_an_unspellable_escalation_mode_is_CARRIED_and_not_swallowed(tmp_path):
    """The loader validates no mode word, and that is the decision rather than an omission.

    The vocabulary has ONE home (`src/link_escalation.py`), the consumer drops what it cannot
    read, and `pack_validate` is where an author sees the typo. A loader that deleted an
    unreadable word here would leave somebody reading a declaration in their own file that no
    run has ever honoured and no diagnostic has ever mentioned — which is the failure mode this
    whole mechanism is built against, arriving through the thing meant to prevent it.
    """
    rules = _RULES_WITH_LINK_ESCALATION.replace("mode: SEMI_AUTO", "mode: semi-auto")
    pack = load_knowledge_pack(_write_shared(tmp_path, rules=rules))
    assert pack.link_escalation("alpha")["mode"] == "semi-auto"


def test_a_ruleset_declaring_NO_escalation_gets_an_EMPTY_MAPPING(tmp_path):
    """Absence is a no-op, and it is the state every pack that exists today is in.

    One shape for every way of declaring nothing — absent, empty, the wrong type, a ruleset
    that does not exist — because the resolver reads this mapping with `.get()` and a `None`
    would reach a caller that subscripts it.
    """
    pack = load_knowledge_pack(_write_shared(tmp_path))
    assert pack.link_escalation("alpha") == {}
    assert pack.link_escalation("no_such_ruleset") == {}
    assert pack.link_escalation("") == {}
    for junk in ("link_escalation: {}", "link_escalation: auto", "link_escalation: []"):
        rules = _RULES_WITH_LINK_ESCALATION.split("    link_escalation:")[0] + (
            f"    {junk}\n"
        )
        inert = load_knowledge_pack(_write_shared(tmp_path, rules=rules))
        assert inert.link_escalation("alpha") == {}, junk


_ALPHA_PLAYBOOK = """---
playbook_id: PB-ALPHA-001
title: Alpha procedure
related_playbooks: [PB-BETA-002, PB-GHOST-999, PB-ALPHA-001]
---

# Alpha
"""

_BETA_PLAYBOOK = """---
playbook_id: PB-BETA-002
title: Beta procedure
---

# Beta
"""


def _write_related_playbooks(tmp_path):
    _write_pack(tmp_path)
    for uc, text in (("alpha", _ALPHA_PLAYBOOK), ("beta", _BETA_PLAYBOOK)):
        pbs = tmp_path / "use_cases" / uc / "playbooks"
        pbs.mkdir(parents=True, exist_ok=True)
        (pbs / f"{uc}.md").write_text(text)
    return tmp_path


def test_related_playbooks_resolves_each_id_to_the_use_case_that_ADJUDICATES_it(
    tmp_path,
):
    """The list is a RELATIONSHIP between procedures, so the useful answer is the procedure.

    A playbook id is what an author writes and a use case is what a run can be pinned to, so
    an accessor returning ids alone hands its caller a second lookup — and the caller that
    needs it is a reconciliation check comparing this graph against the operative one.
    Declaration order is kept, and a playbook naming ITSELF is not a relationship.
    """
    pack = load_knowledge_pack(_write_related_playbooks(tmp_path))
    assert pack.related_playbooks("PB-ALPHA-001") == [
        {"playbook_id": "PB-BETA-002", "use_case": "beta"},
        {"playbook_id": "PB-GHOST-999", "use_case": ""},
    ]
    # A playbook that names nobody, and an id no playbook carries.
    assert pack.related_playbooks("PB-BETA-002") == []
    assert pack.related_playbooks("PB-GHOST-999") == []
    assert pack.related_playbooks("") == []


def test_an_UNRESOLVABLE_related_playbook_is_returned_and_never_filtered(tmp_path):
    """A sibling this pack cannot reach is a FINDING ABOUT THE PACK, so it is returned.

    It is the whole output of the reconciliation check, and filtering it at the accessor
    would turn a stated relationship into a silence — the pack would read as declaring one
    related procedure where it declared two, and the leg that cannot be walked is the one an
    operator most needs named. So the id comes back with an empty `use_case`, which is a
    different value from a resolved one and not merely a shorter list.
    """
    pack = load_knowledge_pack(_write_related_playbooks(tmp_path))
    unresolved = [
        r for r in pack.related_playbooks("PB-ALPHA-001") if not r["use_case"]
    ]
    assert [r["playbook_id"] for r in unresolved] == ["PB-GHOST-999"]


_BINDING_GLOSSARY = """
entities:
  - type: org_unit
  - type: document
  - type: traveller
  - type: terminal
  - type: campaign_id
"""

_BINDING_CATALOG = """
sources:
  - name: transaction_logs
    description: Issuance transactions.
    entities: [org_unit, document]
    endpoints:
      kind: databricks_uc
      catalog: main
      schema: gold
      tables: [settlement_report]
    entity_bindings:
      org_unit: [pos_org_unit]
      document: [document_number]
  - name: access_trail
    description: Access audit trail.
    entities: [document, traveller]
    endpoints:
      kind: databricks_uc
      catalog: main
      schema: gold
      tables: [audit_trail]
    entity_bindings:
      document: [doc_nbr]
      traveller: [pax_name]
  - name: terminal_register
    description: One bound type and nothing beside it.
    entities: [terminal]
    endpoints:
      kind: databricks_uc
      catalog: main
      schema: gold
      tables: [terminals]
    entity_bindings:
      terminal: [terminal_id]
"""


def _write_binding_pack(tmp_path):
    (tmp_path / "entity_glossary.yaml").write_text(_BINDING_GLOSSARY)
    (tmp_path / "source_catalog.yaml").write_text(_BINDING_CATALOG)
    return tmp_path


def test_entity_binding_map_is_COMPUTED_from_what_co_occurs_on_a_row(tmp_path):
    """Reachable means a single row carries both — the only mechanical sense there is.

    A value of X can be put on a query that returns a value of Y only if some source binds
    them both, so the map is co-occurrence over `entity_bindings` and nothing else. It is
    computed rather than declared because a hand-counted table is a second answer that goes
    stale on the next binding added, and both answers then look equally authoritative.

    Unweighted on purpose: ranking by how many sources bind a type recommends the broadest
    type first, and the broadest type is typically a scoping one no procedure takes as its
    subject.
    """
    pack = load_knowledge_pack(_write_binding_pack(tmp_path))
    assert pack.entity_binding_map() == {
        "document": ["org_unit", "traveller"],
        "org_unit": ["document"],
        "terminal": [],
        "traveller": ["document"],
    }


def test_a_TERMINAL_leg_is_empty_and_a_BINDING_GAP_is_absent(tmp_path):
    """Two facts a caller must be able to tell apart, and one shape renders them the same.

    `terminal` is bound on a source and leads nowhere: a query can find it, and no row hands
    anything back. `campaign_id` is declared in the glossary and bound on NO source: nothing
    can find it at all. The first is a terminal leg, the second is a binding gap needing a
    catalog fix, and a map that answered `[]` for both would hide the gap behind a leg that
    is merely narrow.
    """
    pack = load_knowledge_pack(_write_binding_pack(tmp_path))
    reach = pack.entity_binding_map()
    assert reach["terminal"] == []  # bound, leads nowhere
    assert "campaign_id" not in reach  # declared, bound nowhere
    assert any(e.type == "campaign_id" for e in pack.entities)


def test_SCOPING_the_map_to_one_procedures_sources_is_a_strict_SUBSET(tmp_path):
    """Two questions that are easy to conflate, and only one of them a hand-off table asks.

    Unscoped: somewhere in this pack, could a query put a value of X on a row carrying Y —
    which is what a candidate link needs, since any source may be asked. Scoped: what does
    THIS procedure's own evidence hand back. Comparing a per-procedure table against the
    unscoped map reports a divergence on every pair that co-occurs anywhere in the catalog,
    i.e. fails for being right.
    """
    pack = load_knowledge_pack(_write_binding_pack(tmp_path))
    scoped = pack.entity_binding_map(source_names=["transaction_logs"])
    assert scoped == {"document": ["org_unit"], "org_unit": ["document"]}
    unscoped = pack.entity_binding_map()
    for etype, reach in scoped.items():
        assert set(reach) <= set(unscoped[etype])
    assert set(scoped) < set(unscoped)
    # A name no source carries narrows to nothing — never silently to everything.
    assert pack.entity_binding_map(source_names=["no_such_source"]) == {}
    assert pack.entity_binding_map(source_names=[]) == {}


_RULES_WITH_COMPOSITE = """
verdicts:
  alpha:
    subject_entity: record
    sources:
      record: transaction_logs
    conditions:
      - id: combo
        kind: any_of
        label: "Either servicing signal is present"
        report_group: validation
        decisive: true
        fail_detail: "neither signal was found"
        children:
          - use: record_elements/bare
          - use: record_elements/split
            id: was_split
"""


def test_a_composites_children_resolve_their_own_use_imports(tmp_path):
    """A child is a condition, so it imports by the same route.

    Left unresolved, the child keeps no `kind` and the evaluator can only report the parent
    `unknown` — which in the report is indistinguishable from a source that returned nothing.
    """
    pack = load_knowledge_pack(_write_shared(tmp_path, rules=_RULES_WITH_COMPOSITE))
    spec = pack.ruleset_spec("alpha")
    combo = spec["conditions"][0]
    assert combo["kind"] == "any_of" and combo["decisive"] is True
    kids = combo["children"]
    # Mechanics came from the library; the id defaults to the check id, or the child's own.
    assert [k["id"] for k in kids] == ["bare", "was_split"]
    assert kids[0]["kind"] == "element_absence"
    assert kids[0]["counters"] == ["element_counters.AUX", "element_counters.INS"]
    assert kids[1]["kind"] == "element_presence" and kids[1]["arrays"] == ["split.sp"]
    assert not any("use" in k for k in kids)
    # The raw ruleset is untouched: resolution copies rather than mutating the loaded doc.
    raw = pack.rulesets["verdicts"]["alpha"]["conditions"][0]["children"]
    assert [c.get("use") for c in raw] == ["record_elements/bare", "record_elements/split"]


def test_an_unresolvable_child_import_raises_like_any_other(tmp_path):
    """The only deliberately fatal path in pack loading, and a child is no exception."""
    bad = _RULES_WITH_COMPOSITE.replace("record_elements/split", "record_elements/nope")
    with pytest.raises(ValueError, match="unknown shared check"):
        load_knowledge_pack(_write_shared(tmp_path, rules=bad))
