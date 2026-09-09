"""
Swappable domain knowledge pack.

A knowledge pack is a directory (``knowledge/<domain>/``) holding targeting knowledge:
entity glossary, source catalog, playbooks, rulesets, and optional schemas, concepts,
cases, and shared checks.

Swap domains by pointing ``knowledge.pack_dir`` at a different directory. A missing or
empty pack loads cleanly and the pipeline infers everything from the data.

This module reads files only (sync). Playbook ingestion into the KB is left to the caller.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ValueForm(BaseModel):
    """One distinguishable surface form of a single entity type.

    When an entity type carries non-interchangeable identifier forms, a source must bind
    fields per form. The form is classified deterministically by ``pattern`` and carried on
    the extracted entity; a value whose form has a binding for this source is rendered only
    to that form's fields.
    """

    name: str
    # Regex matched in full against the entity value to recognise this form. First form to
    # match wins, so order the declarations most-specific-first.
    pattern: str = ""
    # Human-readable note, surfaced in the extraction prompt so the LLM labels the form it
    # extracted. Advisory only; the regex decides.
    description: str = ""
    # Identity core, where sources disagree about an optional trailing part. Exactly one
    # capture group; offered beside the value as an extra predicate, never as a replacement.
    stem: str = ""
    # Comparison template for a form that is a positional segment of the stored value.
    # ``{value}`` marks the value, ``?`` one character, ``*`` any run. Offered beside the
    # equality predicate, never instead. Absent means equality.
    match: str = ""


class CoIdentity(BaseModel):
    """When two declared forms of one entity type name the same real-world identity.

    The merge is licensed by evidence only: a retrieved row binding both forms and agreeing
    on every ``via`` type. No such row leaves two subjects. ``via`` is an AND.
    """

    #: Form names that co-identify, exactly as declared in ``value_forms``. Two or more.
    forms: List[str] = Field(default_factory=list)
    #: Entity types whose values must agree on the co-occurring row for the merge to be
    #: licensed. Empty means co-occurrence alone licences it.
    via: List[str] = Field(default_factory=list)
    #: Which form to keep as the surviving subject's value. Absent: first extracted wins.
    prefer: str = ""
    #: Human-readable note. Advisory.
    description: str = ""


class EntityDef(BaseModel):
    """One entity type the domain cares about."""

    type: str
    description: str = ""
    recognition_hints: str = ""
    # Candidate field/column names this entity tends to appear as, across sources.
    # A prior for mapping; never overrides what's actually discovered in a source.
    # Source-specific overrides live in SourceDef.entity_bindings.
    field_aliases: List[str] = Field(default_factory=list)
    # Optional regex the extracted *value* should match. Used to soft-validate
    # extraction (log a mismatch, never drop); vocabulary-level only, no query logic.
    value_pattern: str = ""
    # Distinguishable surface forms of this one type (see :class:`ValueForm`). Optional:
    # an entity with no declared forms behaves exactly as before.
    value_forms: List[ValueForm] = Field(default_factory=list)
    # Which of those forms share an identity, and what must agree for that to hold (see
    # :class:`CoIdentity`). Absent: every value becomes its own subject.
    co_identity: Optional[CoIdentity] = None
    # Free-text cardinality hint (e.g. "many-per-organization") to help the planner
    # reason about join keys. Descriptive only.
    cardinality: str = ""
    # Canonical surface forms, used as few-shot hints in the extraction prompt.
    # Vocabulary only; no query semantics. Loose-typed because YAML scalars vary
    # (e.g. a boolean-flag entity's examples are booleans); stringified when rendered.
    examples: List[Any] = Field(default_factory=list)
    # Playbook ids that consume this entity (descriptive; a soft hint for planning).
    used_by: List[str] = Field(default_factory=list)
    # ``"actor"`` (the identity events are attributed to), ``"scope"`` (a population or
    # incident metadata) or ``""``. Read through :meth:`actor_entity_types` /
    # :meth:`scope_entity_types`, which add the engine's generic fallback.
    role: str = ""


class SourceDef(BaseModel):
    """Catalog entry describing one configured log source."""

    name: str
    description: str = ""
    # Entity types this source can be filtered by (matches EntityDef.type values).
    entities: List[str] = Field(default_factory=list)
    # Per-source candidate field names: {entity_type: [field, ...]}. A prior only; the live
    # schema is authoritative. A type with value forms binds per form instead:
    #     user:
    #       login:      [user.userId]
    #       short_code: [alerts.loginArea.code]
    # ``form_bindings_for`` reads that shape; ``field_priors_for`` flattens it.
    entity_bindings: Dict[str, Union[List[str], Dict[str, List[str]]]] = Field(
        default_factory=dict
    )
    # Physical coordinates of the backend (loose dict; shape varies by `kind`):
    #   elasticsearch -> {kind, cluster, indices: [...]}
    #   databricks_uc -> {kind, catalog, schema, tables: [...]}
    #   rest/snowflake -> kind + service-specific keys (no retriever yet)
    # Credentials are not stored here; they live in main_config's log_sources.backends.
    endpoints: Dict[str, Any] = Field(default_factory=dict)
    # Free-text tuning hints injected verbatim into this source's backend-query prompt.
    # Optional.
    query_hints: str = ""
    # Required projection, overriding the retriever's default scalar-only bias, for a nested
    # source whose verdict-relevant fields would otherwise be dropped. Empty lets the LLM pick.
    projection: List[str] = Field(default_factory=list)
    # Freshness/role hint surfaced in the selection prompt. Conventionally "active",
    # "legacy", "fallback", or empty. Soft; does not hard-exclude a source. Optional.
    status: str = ""
    # {field: value} equality terms AND-ed on regardless of the generated query, pinning this
    # source to its slice of a shared index. Enforced by the retriever. Optional.
    default_filters: Dict[str, Any] = Field(default_factory=dict)
    # Entity types to AND rather than OR, for a lookup keyed by that tuple where no single
    # member is selective. Retrievers OR by default, which is right for an event log. Enforced
    # post-generation on every backend (src/retrievers/query_guards.py). Optional.
    require_all_entities: List[str] = Field(default_factory=list)
    # Priority-ordered candidate identity keys, most reliable first, as [[entity_type, ...], ...].
    # The first candidate the incident satisfies in full is enforced as a conjunction; an
    # unsatisfiable one leaves the query unchanged.
    identity_keys: List[List[str]] = Field(default_factory=list)
    # Identity shape for a source whose identity fields are not all the same kind of fact,
    # enforced as (synonym OR synonym) AND scope AND scope by ``enforce_identity_scope``.
    #   ``identity_synonyms``: alternative names for ONE value, OR-ed. A flat list is one
    #     family; a list of lists ORs within each and ANDs between.
    #   ``identity_scopes``: separate identity facts, AND-ed. Per-shape spellings of one fact
    #     belong in a family, not here: no document carries two, so the AND matches nothing.
    # Purely restrictive: a scope the query did not constrain is never added.
    identity_scopes: List[str] = Field(default_factory=list)
    identity_synonyms: Union[List[str], List[List[str]]] = Field(default_factory=list)
    # Fields that may be returned but must never appear as predicates, a filter on one being
    # able to delete the rows that decide the verdict. Stripped post-generation on every backend
    # (src/retrievers/query_guards.py). Dotted paths, matched on the last segment. Optional.
    never_filter: List[str] = Field(default_factory=list)
    # Partition columns every generated query must bound so the backend can prune. Normally
    # left empty: the retriever discovers them from backend metadata. Declare only what
    # discovery cannot see, since a VIEW hides its layout and no catalog carries
    # ``role``/``pad_days``. Entry fields, ``name`` required:
    #   name:      the column name.
    #   role:      ``date`` (default) or ``year``/``month``/``day``, one calendar part
    #              rendered as a list of integers.
    #   type:      column type for literal rendering (DATE/TIMESTAMP/STRING/INT).
    #   pad_days:  days to widen the window on both sides (default 1).
    # Merged with discovered entries by name, this winning on collision.
    partition_columns: List[Dict[str, Any]] = Field(default_factory=list)
    # Time columns stored as epoch integers. Bounds are computed from the incident window and
    # injected already converted. Never write an epoch literal into query_hints: a wrong unit
    # is a plausible-looking predicate that matches nothing. Entries: name (last segment
    # matched), unit (seconds / milliseconds (default) / microseconds / nanoseconds).
    epoch_time_columns: List[Dict[str, Any]] = Field(default_factory=list)
    # Fields whose value is an encoded payload, decoded by ``src/encoded_fields.py`` right after
    # retrieval and written back as real nested columns. Usually needs ``never_filter`` too:
    # that guards the value being returned, this one guards it being read. Keys and their
    # defaults are in docs/architecture/knowledge-pack-authoring.md §2.2; ``field`` is the only
    # required one, ``into`` must name a nested child, and a field that fails to decode is
    # logged and skipped with the encoded value left on the row.
    encoded_fields: List[Dict[str, Any]] = Field(default_factory=list)
    # What zero rows from this source means, where emptiness is an answer rather than a gap.
    # The health scorer otherwise counts every empty source as a defect. Keys, both optional:
    #   health_weight: 0.0-1.0 multiplier on this source's share of ``empty_sources``.
    #                  0.0 = empty is a valid answer; 1.0 (default) = counts in full.
    #   meaning:       one sentence stating what empty means here, surfaced in the
    #                  stage-health reason and the gaps list.
    # Affects scoring only, never retrieval or the verdict.
    zero_rows: Dict[str, Any] = Field(default_factory=dict)
    # Questions this source cannot answer, each naming the source that can, as
    # [{question: "...", ask_instead: "<source name>"}]. Emitted as its own labelled line,
    # since a negative fact inside a paragraph reads as a feature. Advisory.
    not_answered_by: List[Dict[str, Any]] = Field(default_factory=list)
    # When this source is worth choosing, as ``choose_when`` / ``skip_when`` lists of free-text
    # phrases. Prefer this over a ruleset ``sources:`` entry where relevance is
    # incident-dependent, since a hard dependency pays its scan cost on every run. Advisory;
    # the planner still decides.
    selection_guidance: Dict[str, Any] = Field(default_factory=dict)
    # Budget class, ``primary`` or ``""``. ``primary`` means the investigation cannot conclude
    # without this source, so it receives ``primary_source_timeout_seconds`` rather than the
    # backend's ordinary cap. Affects neither selection nor zero-row scoring.
    retrieval_class: str = ""
    # Playbook ids that consume this source (descriptive soft hint).
    used_by: List[str] = Field(default_factory=list)

    def kind(self) -> str:
        """Backend kind from the endpoints block (e.g. 'elasticsearch')."""
        return (self.endpoints or {}).get("kind", "")


class KnowledgePack(BaseModel):
    """The loaded targeting knowledge for one domain."""

    name: str = "default"
    entities: List[EntityDef] = Field(default_factory=list)
    sources: List[SourceDef] = Field(default_factory=list)
    # Domain abbreviations {term: expansion}, from entity_glossary.yaml ``abbreviations:``.
    # Injected into the extraction prompt as a closed list; an undeclared term is left as-is.
    # See ``glossary_prompt``.
    abbreviations: Dict[str, str] = Field(default_factory=dict)
    # ``{field-name token: entity type}`` from entity_glossary.yaml ``field_name_hints:``.
    # Read through :meth:`field_name_hints`, which validates types against declared entities.
    # Empty falls back to the engine's generic tokens.
    field_hints: Dict[str, str] = Field(default_factory=dict)
    # Narrative docs (playbooks) as ingestible {title, content, type} dicts.
    playbook_documents: List[Dict[str, Any]] = Field(default_factory=list)
    # Concept docs as ingestible {title, content, type='concept',
    # metadata:{use_case, concept_id}} dicts. RAG-embedded and surfaced into
    # InvestigationBrief. Shared concepts carry no use_case tag. Empty on a flat-only pack.
    concept_documents: List[Dict[str, Any]] = Field(default_factory=list)
    # Past-investigation records as ingestible {title, content, type='case',
    # metadata:{use_case, case_id, verdict, subject, decisive_reasons, route, resolution,
    # date}} dicts. RAG-embedded and matched into the brief as CasePrecedents.
    case_documents: List[Dict[str, Any]] = Field(default_factory=list)
    # Per-table field inventories from ``schemas/<source>.yaml`` as ingestible
    # {title, content, type='schema', metadata:{schema_source, table, backend_kind,
    # leaf_count, partition_columns}} dicts. RAG-only; never emitted by ``catalog_prompt``.
    schema_documents: List[Dict[str, Any]] = Field(default_factory=list)
    # Investigation procedures from ``rulesets.yaml``: {verdicts: {<key>: {labels,
    # subject_entity, sources, conditions:[...], routes:[...], lock_target, platform_mode,
    # notification}}}. Loose, because the verdict engine dispatches on generic condition
    # ``kind`` values and never on procedure-named fields. Empty means no verdict stage.
    rulesets: Dict[str, Any] = Field(default_factory=dict)
    # Report-presentation vocabulary from ``reporting.yaml``:
    #   phases:   [{keywords: [...], label: "..."}], classifying a chronology event.
    #   phrases:  {slot: text}, replacing the engine's generic wording for a named slot.
    # Scoped per use case under ``use_cases``, over a domain-wide flat-root base. A use case's
    # ``phrases`` merge per sub-key; its ``phases`` REPLACE, that list being an ordered
    # decision procedure. An absent key falls through to the engine's defaults.
    reporting: Dict[str, Any] = Field(default_factory=dict)
    # Reference data from ``data/*.yaml`` (flat-root and each ``use_cases/<name>/data/``),
    # keyed by file stem and read by a condition's ``lookup``. Empty with no ``data/`` folder.
    pack_data: Dict[str, Any] = Field(default_factory=dict)
    # Shared check library from ``shared/checks/*.yaml``, keyed ``<file stem>/<check id>``,
    # imported by a condition's ``use:`` and resolved in ``ruleset_spec``. Mechanics belong
    # here, weighting to the importing ruleset.
    shared_checks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # Named text-equivalence pipelines from ``shared/equivalence_forms.yaml``, attached to a
    # ruleset in ``ruleset_spec`` so a condition's ``form``/``normalize`` can name one. The
    # relation is the pack's; the engine supplies only the operations. Empty = the incumbent
    # single reading, which is what every pack declaring no forms keeps.
    equivalence_forms: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # Domain-wide defaults from ``source_catalog.yaml``'s top-level ``retrieval:`` key. Read
    # through the accessors below; an absent key falls through to the config.
    retrieval_defaults: Dict[str, Any] = Field(default_factory=dict)

    # -- accessors ---------------------------------------------------------

    def entity_types(self) -> List[str]:
        return [e.type for e in self.entities]

    def _types_with_role(self, role: str) -> List[str]:
        """Declared entity types carrying ``role``, in declaration order."""
        return [
            e.type
            for e in self.entities
            if str(getattr(e, "role", "") or "").strip().lower() == role and e.type
        ]

    def actor_entity_types(self) -> List[str]:
        """Entity types with ``role: actor``, in the pack's declaration order.

        Order is load-bearing: attribution takes the first type the source binds, so the type
        naming an individual must come before the broader one.
        """
        return self._types_with_role("actor")

    def scope_entity_types(self) -> List[str]:
        """Entity types that name a population or incident metadata, never a person.

        The engine's own fallback covers only words that are broad in every domain.
        """
        return self._types_with_role("scope")

    def field_name_hints(self) -> Dict[str, str]:
        """``{field-name token: entity type}`` for labelling a discovered join key.

        Layer-3 discovery identifies columns by value comparison rather than by name, so it
        has no label of its own. Tokens are lower-cased; a type naming no declared entity is
        dropped with a warning.
        """
        known = set(self.entity_types())
        out: Dict[str, str] = {}
        for token, etype in (self.field_hints or {}).items():
            tok, et = str(token).strip().lower(), str(etype).strip()
            if not tok or not et:
                continue
            if known and et not in known:
                logger.warning(
                    "field_name_hints: %r maps to %r, which is not a declared entity "
                    "type; ignoring.",
                    tok,
                    et,
                )
                continue
            out[tok] = et
        return out

    def aliases_for(self, entity_type: str) -> List[str]:
        for e in self.entities:
            if e.type == entity_type:
                return e.field_aliases
        return []

    def source(self, name: str) -> Optional[SourceDef]:
        return next((s for s in self.sources if s.name == name), None)

    def value_patterns(self) -> Dict[str, str]:
        """{entity_type: regex} for entities that declared a value_pattern."""
        return {e.type: e.value_pattern for e in self.entities if e.value_pattern}

    def default_lookup_days(self) -> Optional[int]:
        """How many days back to read when the incident states no time, or ``None``.

        Declared as ``retrieval.default_lookup_days`` in ``source_catalog.yaml``. A domain
        fact: retention, alerting lag and typical behaviour duration vary per domain.
        ``None`` (absent or non-positive) falls through to the config key; the engine
        invents no depth of its own.
        """
        raw = (self.retrieval_defaults or {}).get("default_lookup_days")
        if raw is None or isinstance(raw, bool):
            return None
        try:
            days = int(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Pack retrieval.default_lookup_days is %r, which is not a whole number of "
                "days — ignored, so the window falls through to the config key.",
                raw,
            )
            return None
        if days <= 0:
            logger.warning(
                "Pack retrieval.default_lookup_days is %d; a window must be at least one "
                "day, so it is ignored and the config key decides.",
                days,
            )
            return None
        return days

    def zero_row_meanings(self) -> Dict[str, str]:
        """``{source: what an empty result means}`` where empty is an answer, not a gap.

        Reads the ``zero_rows`` prose for sources whose ``health_weight == 0.0`` only. A
        discounted source (0.1-0.9) is still a gap and its meaning must not be promoted to
        a decisive finding.
        """
        out: Dict[str, str] = {}
        for src in self.sources:
            spec = getattr(src, "zero_rows", None)
            if not isinstance(spec, dict):
                continue
            meaning = str(spec.get("meaning") or "").strip()
            if not meaning:
                continue
            try:
                weight = float(spec.get("health_weight", 1.0))
            except (TypeError, ValueError):
                continue  # unreadable weight -> no claim; the scorer logs it
            if weight == 0.0:
                out[str(src.name)] = meaning
        return out

    def entity(self, entity_type: str) -> Optional[EntityDef]:
        return next((e for e in self.entities if e.type == entity_type), None)

    def value_forms_for(self, entity_type: str) -> List[ValueForm]:
        """The declared surface forms of one entity type (``[]`` when it has none)."""
        ent = self.entity(entity_type)
        return list(ent.value_forms) if ent is not None else []

    def co_identity_for(self, entity_type: str) -> Optional[CoIdentity]:
        """How this type's forms co-identify, or ``None`` (see :class:`CoIdentity`).

        A block naming fewer than two forms is treated as absent: it cannot license a merge.
        """
        ent = self.entity(entity_type)
        ci = getattr(ent, "co_identity", None) if ent is not None else None
        if ci is None or len({str(f) for f in (ci.forms or []) if str(f).strip()}) < 2:
            return None
        return ci

    def classify_value_form(self, entity_type: str, value: Any) -> Optional[str]:
        """Name the declared form ``value`` matches, or ``None``.

        Deterministic: first declared form whose ``pattern`` fully matches wins, so the pack
        orders forms most-specific-first. A bad regex is logged and skipped, never fatal.
        """
        text = "" if value is None else str(value).strip()
        if not text:
            return None
        for form in self.value_forms_for(entity_type):
            if not form.pattern:
                continue
            try:
                if re.fullmatch(form.pattern, text):
                    return form.name
            except re.error as e:
                logger.warning(
                    "Invalid value_forms pattern %r on entity %s: %s",
                    form.pattern,
                    entity_type,
                    e,
                )
        return None

    def value_stem(self, entity_type: str, value: Any) -> Optional[str]:
        """The declared identity core of ``value``, or ``None``.

        Returns the capture group of the matching form's ``stem`` regex. ``None`` when the
        type declares no forms, no matching form declares a stem, the regex does not match,
        or the stem equals the value. A bad regex is logged and skipped, never fatal.
        """
        text = "" if value is None else str(value).strip()
        if not text:
            return None
        form_name = self.classify_value_form(entity_type, text)
        if not form_name:
            return None
        for form in self.value_forms_for(entity_type):
            if form.name != form_name or not form.stem:
                continue
            try:
                m = re.match(form.stem, text)
            except re.error as e:
                logger.warning(
                    "Invalid value_forms stem %r on entity %s form %s: %s",
                    form.stem,
                    entity_type,
                    form_name,
                    e,
                )
                return None
            if not m or not m.groups():
                return None
            stem = (m.group(1) or "").strip()
            return stem if stem and stem != text else None
        return None

    def value_match_pattern(self, entity_type: str, value: Any) -> Optional[str]:
        """The declared comparison pattern for ``value``, with ``{value}`` substituted.

        Returns the matching form's ``match`` template with the value filled in. ``None``
        when the type declares no forms, no matching form declares a ``match``, or the
        template does not contain ``{value}`` (a template without it would match every row).
        """
        text = "" if value is None else str(value).strip()
        if not text:
            return None
        form_name = self.classify_value_form(entity_type, text)
        if not form_name:
            return None
        for form in self.value_forms_for(entity_type):
            if form.name != form_name or not form.match:
                continue
            template = str(form.match)
            if "{value}" not in template:
                logger.warning(
                    "value_forms match template %r on entity %s form %s does not name "
                    "{value} — ignored, since a pattern without the value matches every row "
                    "of the column instead of this one.",
                    template,
                    entity_type,
                    form_name,
                )
                return None
            return template.replace("{value}", text)
        return None

    def form_bindings_for(
        self, entity_type: str, source_name: Optional[str]
    ) -> Dict[str, List[str]]:
        """``{form_name: [field, ...]}`` for one entity on one source.

        Returns ``{}`` when the source binds this entity as a flat list (the ordinary case)
        or not at all. A pack declaring no forms is completely unaffected.
        """
        src = self.source(source_name) if source_name else None
        if src is None:
            return {}
        binding = src.entity_bindings.get(entity_type)
        if not isinstance(binding, dict):
            return {}
        out: Dict[str, List[str]] = {}
        for form, fields in binding.items():
            if isinstance(fields, str):
                out[str(form)] = [fields]
            elif isinstance(fields, list):
                out[str(form)] = [str(f) for f in fields]
        return out

    def ruleset_keys(self) -> List[str]:
        """The ruleset keys this pack declares, in declaration order. Empty with no rules."""
        verdicts = (self.rulesets or {}).get("verdicts") or {}
        return [str(k) for k in verdicts]

    def default_ruleset_key(self) -> str:
        """The ruleset that adjudicates when no use-case selection was made.

        Declared as ``default_ruleset`` in the flat-root rules file (sibling of ``verdicts``)::

            default_ruleset: <one of the keys under verdicts>

        Absent, falls back to the first key in declaration order (alphabetical). The
        declaration matters on a pack with two or more rulesets: without it the default shifts
        whenever a new use-case directory is added. A declared name that matches no ruleset
        logs a warning and falls back; ``pack_validate`` reports it as an error.
        """
        keys = self.ruleset_keys()
        declared = str((self.rulesets or {}).get("default_ruleset", "") or "").strip()
        if declared:
            if declared in keys:
                return declared
            logger.warning(
                "Pack %r declares default_ruleset=%r, which is not one of its rulesets "
                "(%s). Falling back to the first declared, which is DIRECTORY ORDER — so "
                "the procedure adjudicating an unselected incident is whichever name sorts "
                "first, not the one declared.",
                self.name,
                declared,
                ", ".join(keys) or "none",
            )
        return keys[0] if keys else ""

    def adjudication_policy(self) -> str:
        """What to do when no procedure was selected: ``default`` (adjudicate anyway) or ``abstain``.

        Declared beside ``default_ruleset`` in the flat-root rules file, because it decides
        what that declaration is *for*::

            adjudication_policy: abstain

        ``default`` — the shipped value, and what every pack did before this key existed —
        adjudicates an unselected incident under ``default_ruleset_key()`` and says so.
        ``abstain`` converts that verdict to the ruleset's own ``reject`` label instead, which
        a pack should only ask for once it has procedures for the incidents it expects: a
        reject is a routing decision back to the detector's owner, not a finding.

        An unrecognised value reads as ``default`` and logs, because the failure mode of
        guessing ``abstain`` is a pack that stops adjudicating for a typo; ``pack_validate``
        reports it as an error.
        """
        raw = str((self.rulesets or {}).get("adjudication_policy", "") or "").strip().lower()
        if not raw:
            return "default"
        if raw in ("default", "abstain"):
            return raw
        logger.warning(
            "Pack %r declares adjudication_policy=%r, which is neither 'default' nor "
            "'abstain'; adjudicating an unselected incident under the default ruleset, as a "
            "pack declaring nothing does.",
            self.name,
            raw,
        )
        return "default"

    def ruleset_key_for(self, use_case: str) -> str:
        """The ruleset key a use-case name selects, or ``""`` when it selects none.

        A ruleset key is a use-case directory name; the mapping needs no extra declaration.
        Returns ``""`` (caller keeps the pack default) when the use case is empty, unknown,
        or ships no ``rules.yaml``, leaving single-ruleset packs byte-identical.
        """
        name = str(use_case or "").strip()
        if not name:
            return ""
        verdicts = (self.rulesets or {}).get("verdicts") or {}
        return name if name in verdicts else ""

    def ruleset_spec(self, key: str = "") -> Optional[Dict[str, Any]]:
        """The named validation ruleset from ``rulesets.yaml``, or ``None`` if absent.

        Empty ``key`` selects the pack's default ruleset. Returns the raw spec dict
        (labels, subject_entity, sources, conditions, routes, lock_target, platform_mode,
        notification). Shared-check imports are resolved here so both consumers (verdict
        engine and case builder) receive a fully-expanded condition list. The subject
        type's co-identity is attached under ``_co_identity`` for the same reason.
        ``None`` on an empty pack or missing key; the verdict stage degrades to no-verdict.
        """
        verdicts = (self.rulesets or {}).get("verdicts") or {}
        if not key:
            key = self.default_ruleset_key()
        spec = verdicts.get(key)
        if not isinstance(spec, dict):
            return None
        spec = _resolve_check_imports(spec, self.shared_checks, key)
        # The subject type's co-identity is attached here because both consumers reach a
        # ruleset through this accessor; the verdict engine receives pack_data rather than
        # the pack and cannot ask the glossary directly. A ruleset declaring ``_co_identity``
        # would be overwritten: the glossary owns entity-form definitions, not a procedure.
        subject = str(spec.get("subject_entity", "") or "")
        ci = self.co_identity_for(subject) if subject else None
        if ci is not None:
            spec = dict(spec)
            spec["_co_identity"] = ci.model_dump()
        # Equivalence forms ride the same seam and for the same reason. Attached only when the
        # pack declares some, so a pack that declares none produces the condition dicts it
        # always did — the byte-identical checkpoint, held at the one place it can be.
        if self.equivalence_forms:
            spec = dict(spec)
            spec["_equivalence_forms"] = dict(self.equivalence_forms)
        return spec

    def follow_up_passes(self, key: str = "") -> List[Dict[str, Any]]:
        """Follow-up retrieval passes a ruleset declares, ordered by pass number.

        A follow-up pass asks questions whose scoping values only arrive from a prior
        retrieval result. Declaring none leaves retrieval single-pass. Entry shape and its
        keys: docs/architecture/retrieval.md.

        Returns entries with logical names resolved to physical source names; ``source`` may
        be a list, refused per name so one stale entry does not silence its siblings. An
        entry is dropped with a warning when pass < 2, no target, or nothing to harvest.
        """
        spec = self.ruleset_spec(key) or {}
        declared = spec.get("follow_up_passes")
        if not isinstance(declared, list):
            return []
        raw_map = spec.get("sources")
        logical = (
            {str(k): str(v) for k, v in raw_map.items()}
            if isinstance(raw_map, dict)
            else {}
        )
        out: List[Dict[str, Any]] = []
        for entry in declared:
            if not isinstance(entry, dict):
                continue
            try:
                number = int(entry.get("pass", 0))
            except (TypeError, ValueError):
                number = 0
            declared_targets = entry.get("source")
            if isinstance(declared_targets, (list, tuple, set)):
                targets = [str(t).strip() for t in declared_targets if str(t).strip()]
            else:
                targets = [t for t in [str(declared_targets or "").strip()] if t]
            target = targets[0] if targets else ""
            harvest: List[Dict[str, Any]] = []
            for item in entry.get("harvest") or []:
                if not isinstance(item, dict):
                    continue
                etype = str(item.get("entity", "") or "").strip()
                fields = [
                    str(f).strip() for f in (item.get("fields") or []) if str(f).strip()
                ]
                # A co-occurrence item declares entities per component; normalise here so
                # the consumer reads one shape regardless of which spelling was used.
                together: List[Dict[str, Any]] = []
                for comp in item.get("together") or []:
                    if not isinstance(comp, dict):
                        continue
                    centity = str(comp.get("entity", "") or "").strip()
                    cfields = [
                        str(f).strip()
                        for f in (comp.get("fields") or [])
                        if str(f).strip()
                    ]
                    if not centity or not cfields:
                        continue
                    together.append(
                        {
                            "entity": centity,
                            "fields": cfields,
                            "capture": str(comp.get("capture", "") or ""),
                        }
                    )
                if not together and (not etype or not fields):
                    continue
                from_source = str(item.get("source", "") or "").strip()
                where = [c for c in (item.get("where") or []) if isinstance(c, dict)]
                harvest.append(
                    {
                        "entity": etype,
                        # Empty means "read every source in the accumulated result", which
                        # is a legitimate declaration for a value that may arrive from a
                        # source the procedure did not predict.
                        "source": logical.get(from_source, from_source),
                        "fields": fields,
                        # Always present; empty means "every row": the consumer reads one
                        # shape, and a dropped malformed declaration cannot be mistaken for
                        # one that was never made.
                        "where": where,
                        # Always present; empty means "these values are independent": the
                        # per-type reading, applied for any pack that declared no combination.
                        "together": together,
                    }
                )
            if number < 2 or not target or not harvest:
                logger.warning(
                    "Ruleset %r declares a follow-up pass the engine cannot run and it is "
                    "DROPPED (pass=%r, source(s)=%r, %d harvest entries). A pass below 2, with "
                    "no target source, or with nothing to harvest would either re-run the "
                    "ordinary pass or query with no values at all — a window-only scan whose "
                    "rows answer nothing.",
                    key or "(first)",
                    entry.get("pass"),
                    targets,
                    len(harvest),
                )
                continue
            resolved = []
            for name in targets:
                physical = logical.get(name, name)
                if physical and physical not in resolved:
                    resolved.append(physical)
            # ``purpose`` may be a mapping of source to text when targets ask different
            # questions of the same harvest. A plain string applies to all targets.
            raw_purpose = entry.get("purpose")
            purposes: Dict[str, str] = {}
            purpose = ""
            if isinstance(raw_purpose, dict):
                for name, text in raw_purpose.items():
                    physical = logical.get(str(name).strip(), str(name).strip())
                    if physical:
                        purposes[physical] = str(text or "").strip()
            else:
                purpose = str(raw_purpose or "").strip()
            out.append(
                {
                    "pass": number,
                    # ``source`` stays the first target; ``sources`` carries all of them so
                    # a single-target consumer keeps working and both readings agree.
                    "source": resolved[0],
                    "sources": resolved,
                    "harvest": harvest,
                    "capture": str(entry.get("capture", "") or ""),
                    "window": str(entry.get("window", "") or "inherit").strip().lower(),
                    "skip_when_empty": bool(entry.get("skip_when_empty", True)),
                    "purpose": purpose,
                    # Always present; empty for a single shared purpose, keeping one shape
                    # for the consumer (as with a harvest's `where`).
                    "purposes": purposes,
                }
            )
        out.sort(key=lambda e: e["pass"])
        return out

    def follow_up_pass(self, key: str, number: int) -> Optional[Dict[str, Any]]:
        """The declared follow-up pass numbered ``number``, or ``None``.

        Two entries claiming the same number is an authoring error `pack_validate` reports;
        here the first declared wins, deterministically, so a run is reproducible either way.
        """
        for entry in self.follow_up_passes(key):
            if int(entry.get("pass", 0)) == int(number):
                return entry
        return None

    def entry_signals(self, key: str = "") -> List[Dict[str, Any]]:
        """Inbound recognition signals a ruleset declares, in declaration order.

        A signal says how this procedure's subject would show up in another procedure's
        evidence. Declared inbound (on the target) so adding a new procedure requires one
        file, not edits to every existing one. Entry shape and its keys:
        docs/architecture/knowledge-pack-authoring.md §2.5.1.

        ``when.source`` resolves through this ruleset's ``sources:`` map but is not a hard
        dependency. Returns ``[]`` on absence; an entry is dropped when it has no ``id``, no
        ``opens_with`` entity, or no source to read.
        """
        spec = self.ruleset_spec(key) or {}
        declared = spec.get("entry_signals")
        if not isinstance(declared, list):
            return []
        raw_map = spec.get("sources")
        logical = (
            {str(k): str(v) for k, v in raw_map.items()}
            if isinstance(raw_map, dict)
            else {}
        )
        out: List[Dict[str, Any]] = []
        for entry in declared:
            if not isinstance(entry, dict):
                continue
            signal_id = str(entry.get("id", "") or "").strip()
            opens = entry.get("opens_with")
            entity = (
                str((opens or {}).get("entity", "") or "").strip()
                if isinstance(opens, dict)
                else ""
            )
            when = entry.get("when") if isinstance(entry.get("when"), dict) else {}
            declared_source = str(when.get("source", "") or "").strip()
            source = logical.get(declared_source, declared_source)
            if not signal_id or not entity or not source:
                logger.warning(
                    "Ruleset %r declares an entry signal the engine cannot read and it is "
                    "DROPPED (id=%r, opens_with.entity=%r, when.source=%r). Without an id "
                    "there is nothing for a report to cite, without a subject entity there is "
                    "no leg to open, and without a source there are no rows to read.",
                    key or "(default)",
                    entry.get("id"),
                    entity,
                    declared_source,
                )
                continue
            try:
                min_rows = max(1, int(when.get("min_rows", 1)))
            except (TypeError, ValueError):
                min_rows = 1
            try:
                strength = float(entry.get("strength", 0.0) or 0.0)
            except (TypeError, ValueError):
                strength = 0.0
            base_rate = entry.get("base_rate")
            out.append(
                {
                    "id": signal_id,
                    # not validated against the engine's causal vocabulary here: that
                    # vocabulary has one home (`src/links.py`); the consumer drops what it
                    # cannot read, and `pack_validate` is where an author sees the typo.
                    "direction": str(entry.get("direction", "") or "").strip().lower(),
                    "entity": entity,
                    "source": source,
                    # Always present; empty means "every row of that source": one shape for
                    # the consumer, as with a harvest's `where`.
                    "where": [
                        c for c in (when.get("where") or []) if isinstance(c, dict)
                    ],
                    "min_rows": min_rows,
                    "window": str(entry.get("window", "") or "inherit").strip().lower(),
                    "strength": strength,
                    "base_rate": base_rate if isinstance(base_rate, dict) else {},
                    "auto_probe": bool(entry.get("auto_probe", False)),
                    "note": str(entry.get("note", "") or "").strip(),
                }
            )
        return out

    def link_escalation(self, key: str = "") -> Dict[str, Any]:
        """The escalation mode a ruleset declares for links pointing at it.

        Declared on the target procedure. Shape::

            link_escalation:
              mode: planned                 # planned | semi_auto | auto, for every source
              from:                         # optional per-source-procedure override
                <some use case>: auto

        ``from`` wins over ``mode``, which wins over the deployment config. Mode words are
        not validated here; ``pack_validate`` catches typos and ``src/link_escalation.py``
        is the vocabulary. Returns ``{}`` on absence.
        """
        spec = self.ruleset_spec(key) or {}
        declared = spec.get("link_escalation")
        if not isinstance(declared, dict) or not declared:
            return {}
        out: Dict[str, Any] = {}
        mode = str(declared.get("mode", "") or "").strip().lower()
        if mode:
            out["mode"] = mode
        raw = declared.get("from")
        if isinstance(raw, dict):
            per_source = {
                str(source_key): str(value or "").strip().lower()
                for source_key, value in raw.items()
                if str(value or "").strip()
            }
            if per_source:
                out["from"] = per_source
        return out

    def open_questions(self, key: str = "") -> List[Dict[str, Any]]:
        """Open questions a ruleset declares about its OWN evidence, in declaration order.

        The other axis from ``entry_signals``: that one says how a SIBLING procedure would
        show up here, this one says what THIS procedure could not settle and which one further
        source would say something about it. Entry shape and its keys:
        docs/architecture/knowledge-pack-authoring.md §2.5.3.

        ``ask.source`` resolves through this ruleset's ``sources:`` map and is NOT a hard
        dependency: an inquiry is asked with one bounded probe of its own, never by adding a
        source to every run of the procedure.

        Returns ``[]`` on absence. An entry is DROPPED when it has no ``id``, no trigger the
        engine can read (a condition id or a verdict class), no source to ask, or no meaning
        for every outcome — each of those makes the entry unaskable or unlabelled, and an
        unlabelled outcome is the failure class this repo is organised against.
        """
        spec = self.ruleset_spec(key) or {}
        declared = spec.get("open_questions")
        if not isinstance(declared, list):
            return []
        raw_map = spec.get("sources")
        logical = (
            {str(k): str(v) for k, v in raw_map.items()}
            if isinstance(raw_map, dict)
            else {}
        )
        out: List[Dict[str, Any]] = []
        for entry in declared:
            if not isinstance(entry, dict):
                continue
            question_id = str(entry.get("id", "") or "").strip()
            when = entry.get("when") if isinstance(entry.get("when"), dict) else {}
            ask = entry.get("ask") if isinstance(entry.get("ask"), dict) else {}
            raw_meaning = (
                entry.get("meaning") if isinstance(entry.get("meaning"), dict) else {}
            )
            condition = str(when.get("condition", "") or "").strip()
            verdict_class = str(when.get("verdict_class", "") or "").strip()
            declared_source = str(ask.get("source", "") or "").strip()
            source = logical.get(declared_source, declared_source)
            meaning = {
                outcome: str(raw_meaning.get(outcome, "") or "").strip()
                for outcome in ("rows", "empty", "unanswered")
            }
            missing = [outcome for outcome, text in meaning.items() if not text]
            if (
                not question_id
                or not (condition or verdict_class)
                or not source
                or missing
            ):
                logger.warning(
                    "Ruleset %r declares an open question the engine cannot read and it is "
                    "DROPPED (id=%r, when.condition=%r, when.verdict_class=%r, ask.source=%r, "
                    "meaning missing for %s). Without an id there is nothing for a report to "
                    "cite, without a trigger it can never be raised, without a source there is "
                    "nothing to ask, and an outcome with no declared meaning is a number nobody "
                    "can act on.",
                    key or "(default)",
                    entry.get("id"),
                    condition,
                    verdict_class,
                    declared_source,
                    ", ".join(missing) or "nothing",
                )
                continue
            out.append(
                {
                    "id": question_id,
                    "condition": condition,
                    # Default `unknown`, and deliberately: an open question is what a check
                    # that could not answer leaves behind. A pack wanting the other polarity
                    # says so. Not validated against the engine's result vocabulary here —
                    # `pack_validate` is where an author sees the typo.
                    "result": str(when.get("result", "") or "unknown").strip().lower(),
                    "verdict_class": verdict_class,
                    "source": source,
                    "question": str(ask.get("question", "") or "").strip(),
                    "scope_entity": str(ask.get("scope_entity", "") or "").strip(),
                    # Always present; empty means "every row the probe returned", one shape
                    # for the consumer as with an entry signal's `where`.
                    "where": [c for c in (ask.get("where") or []) if isinstance(c, dict)],
                    "meaning": meaning,
                    "note": str(entry.get("note", "") or "").strip(),
                }
            )
        return out

    def related_playbooks(self, playbook_id: str) -> List[Dict[str, str]]:
        """Playbooks a playbook names as related, each resolved to its use case.

        Returns ``[{"playbook_id": ..., "use_case": ...}]`` in declaration order. An
        unresolvable id is returned rather than filtered; it is a finding about the pack.
        This list states a relationship, not that a link exists in the current evidence.
        """
        want = str(playbook_id or "").strip()
        if not want:
            return []
        by_id: Dict[str, str] = {}
        for doc in self.playbook_documents:
            meta = doc.get("metadata", {}) or {}
            pid = str(meta.get("playbook_id", "") or "").strip()
            if pid and pid not in by_id:
                by_id[pid] = str(meta.get("use_case", "") or "")
        out: List[Dict[str, str]] = []
        seen = set()
        for doc in self.playbook_documents:
            meta = doc.get("metadata", {}) or {}
            if str(meta.get("playbook_id", "") or "").strip() != want:
                continue
            for related in meta.get("related_playbooks") or []:
                rid = str(related or "").strip()
                if not rid or rid == want or rid in seen:
                    continue
                seen.add(rid)
                out.append({"playbook_id": rid, "use_case": by_id.get(rid, "")})
        return out

    def entity_binding_map(
        self, source_names: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """``{entity type: [entity types reachable from it]}``, computed from the catalog.

        "Reachable" means some source that binds type X also binds type Y in the same row.
        Computed rather than declared so it stays correct when bindings change.

        Every type in any source's ``entity_bindings`` gets a key; a type that is bound but
        leads nowhere gets an empty list (different from a type bound on no source at all,
        which is absent). ``source_names`` narrows the computation to those sources when given;
        the unscoped map is a strict superset of any per-procedure scoped map.
        """
        wanted = {str(n) for n in source_names} if source_names is not None else None
        out: Dict[str, set] = {}
        for src in self.sources or []:
            if wanted is not None and str(src.name) not in wanted:
                continue
            bound = sorted(str(t) for t in (src.entity_bindings or {}))
            for etype in bound:
                out.setdefault(etype, set()).update(t for t in bound if t != etype)
        return {etype: sorted(reach) for etype, reach in sorted(out.items())}

    def _reporting_for(self, use_case: Optional[str], key: str) -> Any:
        """``reporting[key]``, use-case entry winning over the domain root.

        A mapping (``phrases``) is merged per sub-key so a use case can re-word one slot
        without discarding the rest. A list (``phases``) replaces entirely because its order
        is the semantics; merging two would produce a precedence nobody authored.
        """
        rep = self.reporting or {}
        if not isinstance(rep, dict):
            return None
        base = rep.get(key)
        if not use_case:
            return base
        scoped = (rep.get("use_cases") or {}).get(use_case)
        scoped_val = scoped.get(key) if isinstance(scoped, dict) else None
        if scoped_val is None:
            return base
        if isinstance(scoped_val, dict) and isinstance(base, dict):
            merged = dict(base)
            merged.update(scoped_val)
            return merged
        return scoped_val

    def phase_rules(self, use_case: Optional[str] = None) -> List[Dict[str, Any]]:
        """``reporting.phases`` as ``[{keywords: [...], label: str}]`` (``[]`` if absent).

        Only well-formed entries survive: a phase with no keywords or no label cannot
        match anything, so silently dropping it is preferable to a report that renders an
        empty ``[]`` header.
        """
        raw = self._reporting_for(use_case, "phases") or []
        out: List[Dict[str, Any]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            kws = [
                str(k).lower() for k in (entry.get("keywords") or []) if str(k).strip()
            ]
            label = str(entry.get("label", "") or "").strip()
            if kws and label:
                out.append({"keywords": kws, "label": label})
        return out

    def report_phrase(
        self, slot: str, default: str = "", use_case: Optional[str] = None
    ) -> str:
        """The pack's wording for a named report slot, else ``default``.

        This is the seam that keeps an official procedure's own language (its clause
        numbers, its term for a containment target) out of ``src/``. The engine writes the
        generic sentence as the ``default`` and asks here for the domain's version, so a
        pack that declares nothing still produces a complete report and a pack that declares
        a slot gets its exact wording. Placeholders in the pack's text are substituted by the
        caller with :meth:`str.replace`, never ``.format`` (procedure text contains braces).
        """
        phrases = self._reporting_for(use_case, "phrases") or {}
        value = phrases.get(slot) if isinstance(phrases, dict) else None
        text = str(value).strip() if value is not None else ""
        return text or default

    def correlation_specs(self) -> List[Dict[str, Any]]:
        """All playbooks' structured ``correlation`` blocks (functional join specs).

        Each entry is the parsed ``correlation:`` frontmatter dict, tagged with
        ``title``, ``playbook_id`` and ``use_case`` for matching. Empty list when no
        playbook declares one. Empty ``use_case`` on a flat-root playbook selects no
        procedure and the default ruleset stands.
        """
        specs: List[Dict[str, Any]] = []
        for doc in self.playbook_documents:
            meta = doc.get("metadata", {}) or {}
            corr = meta.get("correlation")
            if isinstance(corr, dict):
                specs.append(
                    {
                        **corr,
                        "title": doc.get("title", ""),
                        "playbook_id": meta.get("playbook_id", ""),
                        "use_case": meta.get("use_case", "") or "",
                    }
                )
        return specs

    def concepts_for(
        self, use_case: str, ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Concept docs available to a use case, optionally filtered to specific ids.

        Unions the use case's own concepts with the pack's shared concepts (no ``use_case``
        tag). The use case's own concepts come first; a re-declared shared id wins. When
        ``ids`` is given, only those ``concept_id``s are returned, in requested order.
        """
        matches = [
            d
            for d in self.concept_documents
            if (d.get("metadata", {}) or {}).get("use_case") == use_case
        ] + [
            d
            for d in self.concept_documents
            if not (d.get("metadata", {}) or {}).get("use_case")
        ]
        if ids is None:
            return matches
        # `setdefault`, not a dict comprehension: the specific concepts are first in `matches`
        # and a comprehension would let the shared entry with the same id overwrite them,
        # inverting the precedence the docstring promises.
        by_id: Dict[Any, Dict[str, Any]] = {}
        for d in matches:
            by_id.setdefault((d.get("metadata", {}) or {}).get("concept_id"), d)
        return [by_id[i] for i in ids if i in by_id]

    def cases_for(
        self, use_case: str, verdict: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Past-investigation case docs for a use case, optionally filtered by verdict.

        Returns the ingestible case dicts (``{title, content, metadata}``) whose
        ``metadata.use_case`` matches. When ``verdict`` is given, only cases whose
        ``metadata.verdict`` equals it (case-insensitive) are returned. Empty on a
        flat-only pack.
        """
        matches = [
            d
            for d in self.case_documents
            if (d.get("metadata", {}) or {}).get("use_case") == use_case
        ]
        if verdict is None:
            return matches
        want = str(verdict).strip().lower()
        return [
            d
            for d in matches
            if str((d.get("metadata", {}) or {}).get("verdict", "")).strip().lower()
            == want
        ]

    def field_priors_for(
        self,
        entity_type: str,
        source_name: Optional[str],
        value_form: Optional[str] = None,
    ) -> List[str]:
        """Candidate field names for an entity, source-specific binding first.

        Per-source ``entity_bindings`` take precedence over global ``field_aliases``.
        When the source binds per value form and ``value_form`` names one, only that form's
        fields are offered. Without ``value_form`` the form map is flattened, preserving
        behaviour for callers that do not know about forms.
        """
        ordered: List[str] = []
        forms = self.form_bindings_for(entity_type, source_name)
        if forms:
            if value_form is not None:
                # A form the source does not bind has no fields here. Returning the other
                # form's columns would introduce the wrong binding, so an unbound form
                # yields nothing.
                ordered.extend(forms.get(value_form, []))
            else:
                for fields in forms.values():
                    ordered.extend(fields)
        else:
            src = self.source(source_name) if source_name else None
            if src is not None:
                binding = src.entity_bindings.get(entity_type, [])
                if isinstance(binding, list):
                    ordered.extend(binding)
        # The global aliases span all forms, so they are only a safe fallback when the
        # source did not bind per form (otherwise they re-introduce the wrong column).
        if not (forms and value_form is not None):
            ordered.extend(self.aliases_for(entity_type))
        # De-dupe preserving order.
        seen = set()
        return [f for f in ordered if not (f in seen or seen.add(f))]

    # -- prompt fragments --------------------------------------------------

    def glossary_prompt(self) -> str:
        """Describe the domain's entities so the LLM knows what to extract."""
        if not self.entities:
            return ""
        lines = []
        if self.abbreviations:
            # Before the entities, as a closed list; sorted for prompt-cache stability.
            lines.append(
                "Domain abbreviations. Use ONLY these expansions. If a term is not "
                "listed, write the acronym as-is and do NOT guess what it stands for:"
            )
            for term in sorted(self.abbreviations):
                lines.append(f"- {term} = {self.abbreviations[term]}")
            lines.append("")
        lines.append(
            "Domain entities to extract from the incident (use these as the "
            "entity `type`):"
        )
        for e in self.entities:
            hint = f" — {e.description}" if e.description else ""
            rec = f" Recognition: {e.recognition_hints}" if e.recognition_hints else ""
            pat = f" Value pattern: {e.value_pattern}." if e.value_pattern else ""
            # Cap examples so a large glossary doesn't blow up the prompt.
            eg = (
                f" e.g. {', '.join(str(x) for x in e.examples[:5])}."
                if e.examples
                else ""
            )
            lines.append(f"- {e.type}{hint}.{rec}{pat}{eg}")
        return "\n".join(lines)

    def catalog_prompt(self, source_names: Optional[List[str]] = None) -> str:
        """Describe configured sources so the LLM picks them by capability."""
        sources = self.sources
        if source_names is not None:
            sources = [s for s in sources if s.name in source_names]
        if not sources:
            return ""
        lines = [
            "Available log sources (choose by what each contains; prefer sources "
            "marked [active] over [legacy]/[fallback] unless the incident is historical "
            "or the active source lacks the needed data):"
        ]
        for s in sources:
            ents = f" Filterable by: {', '.join(s.entities)}." if s.entities else ""
            desc = f" {s.description}" if s.description else ""
            st = f" [{s.status}]" if s.status else ""
            lines.append(f"- {s.name}{st}:{desc}{ents}")
            # Negative facts on their own lines, not buried in the description paragraph.
            for item in s.not_answered_by or []:
                if not isinstance(item, dict):
                    continue
                question = str(item.get("question") or "").strip()
                if not question:
                    continue
                instead = str(item.get("ask_instead") or "").strip()
                tail = f" Ask {instead} instead." if instead else ""
                lines.append(f"    DOES NOT ANSWER: {question}.{tail}")
            # Selection guidance on its own lines.
            guidance = s.selection_guidance or {}
            if isinstance(guidance, dict):
                for label, key in (
                    ("CHOOSE WHEN", "choose_when"),
                    ("SKIP WHEN", "skip_when"),
                ):
                    entries = guidance.get(key) or []
                    if isinstance(entries, str):
                        entries = [entries]
                    for entry in entries:
                        text = str(entry).strip()
                        if text:
                            lines.append(f"    {label}: {text}")
        return "\n".join(lines)

    def alias_hints(
        self, entity_types: List[str], source_name: Optional[str] = None
    ) -> str:
        """Compact 'entity -> candidate field names' hint string for mapping.

        When ``source_name`` is given, per-source ``entity_bindings`` are preferred
        over the global aliases (same entity, different column per source).
        """
        parts = []
        for et in entity_types:
            aliases = self.field_priors_for(et, source_name)
            if aliases:
                parts.append(f"{et}: {', '.join(aliases)}")
        return "; ".join(parts)


def _jsonable(value: Any) -> Any:
    """Coerce YAML scalars that aren't JSON-serializable (date/datetime) to strings.

    Recurses into lists/dicts. YAML parses a bare ``2026-07-24`` as ``datetime.date``;
    leaving it in doc metadata breaks the RAG ``documents.json`` dump. Everything else
    passes through unchanged.
    """
    import datetime as _dt

    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:  # a malformed pack must not crash startup
        logger.warning("Failed to read knowledge-pack file %s: %s", path, e)
        return {}


def _parse_frontmatter(content: str) -> Dict[str, Any]:
    """Parse a leading YAML frontmatter block (between --- fences); {} if absent/bad."""
    if not content.startswith("---"):
        return {}
    end = content.find("\n---", 3)
    if end == -1:
        return {}
    block = content[3:end]
    try:
        data = yaml.safe_load(block)
        return data if isinstance(data, dict) else {}
    except Exception as e:  # a malformed block must not break pack loading
        logger.warning("Failed to parse playbook frontmatter: %s", e)
        return {}


#: HTML comments in pack markdown docs are authoring notes. Stripped at load because
#: downstream consumers read only a prefix of the text, so a note near the start would
#: reach a prompt rather than the pack's domain knowledge.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
#: Removing a comment can leave excess blank lines; collapse runs to at most two.
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _strip_html_comments(content: str) -> str:
    """Drop ``<!-- ... -->`` spans from markdown text (unterminated ones are left alone)."""
    if "<!--" not in content:
        return content
    return _BLANK_RUN_RE.sub("\n\n", _HTML_COMMENT_RE.sub("", content))


def _read_markdown_docs(
    docs_dir: Path, doc_type: str, use_case: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Read ``*.md`` files from a dir into ingestible {title, content, type, metadata} dicts.

    Generic over ``doc_type`` ('playbook'/'concept'/'case'). Frontmatter is parsed and
    retained per type: playbooks keep ``playbook_id`` + the structured ``correlation``
    block; concepts get ``concept_id`` (frontmatter or file stem); cases get ``case_id``
    plus verdict/subject/decisive_reasons/route/resolution/date. ``use_case`` (when given)
    is stamped on every doc's metadata.
    """
    docs: List[Dict[str, Any]] = []
    if not docs_dir.is_dir():
        return docs
    for path in sorted(docs_dir.glob("*.md")):
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read %s doc %s: %s", doc_type, path, e)
            continue
        fm = _parse_frontmatter(content)
        metadata: Dict[str, Any] = {"file_path": str(path)}
        if use_case:
            metadata["use_case"] = use_case
        if doc_type == "playbook":
            if fm.get("playbook_id"):
                metadata["playbook_id"] = fm["playbook_id"]
            # Retain the optional structured correlation block (functional join spec).
            # Shape: {keys, fields, time_window, time_fields, key_filter}. Most playbooks
            # omit it.
            if isinstance(fm.get("correlation"), dict):
                metadata["correlation"] = fm["correlation"]
            # An unresolvable id is returned by ``related_playbooks`` as a finding; dropping
            # it at load would turn a stated relationship into a silence.
            if isinstance(fm.get("related_playbooks"), list):
                metadata["related_playbooks"] = [
                    str(p).strip() for p in fm["related_playbooks"] if str(p).strip()
                ]
        elif doc_type == "concept":
            metadata["concept_id"] = fm.get("concept_id") or path.stem
            if fm.get("title"):
                metadata["title"] = fm["title"]
        elif doc_type == "case":
            metadata["case_id"] = fm.get("case_id") or path.stem
            for k in (
                "verdict",
                "subject",
                "decisive_reasons",
                "route",
                "resolution",
                "date",
                "scope",
            ):
                if fm.get(k) is not None:
                    # YAML parses bare dates as datetime.date, which is not JSON-
                    # serializable and would break the RAG documents.json dump. Coerce
                    # date/datetime to ISO strings (metadata is display/text anyway).
                    metadata[k] = _jsonable(fm[k])
        docs.append(
            {
                "title": fm.get("title") or path.stem,
                # Frontmatter is parsed from the raw text above (a doc whose first
                # characters are not `---` has none), and the authoring notes come out
                # only here; see `_strip_html_comments`.
                "content": _strip_html_comments(content),
                "type": doc_type,
                "metadata": metadata,
            }
        )
    return docs


def _read_playbooks(playbooks_dir: Path) -> List[Dict[str, Any]]:
    """Back-compat wrapper: read a flat ``playbooks/`` dir of playbook docs."""
    return _read_markdown_docs(playbooks_dir, "playbook")


def _read_shared_checks(checks_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Read ``shared/checks/*.yaml`` into a library keyed ``<file stem>/<check id>``.

    Each file is a flat map of ``{check_id: {mechanics}}``. Mechanics (kind, field paths)
    belong here; weighting belongs to the importing ruleset and always wins. Missing dir
    returns ``{}``; a malformed file is skipped with a warning.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not checks_dir.is_dir():
        return out
    for f in sorted(checks_dir.glob("*.yaml")) + sorted(checks_dir.glob("*.yml")):
        try:
            doc = _read_yaml(f) or {}
        except Exception as e:  # one bad file must not break the whole pack
            logger.warning("Skipping shared check file %s: %s", f, e)
            continue
        if not isinstance(doc, dict):
            logger.warning("Shared check file %s is not a mapping; skipped.", f)
            continue
        for check_id, body in doc.items():
            if not isinstance(body, dict):
                logger.warning(
                    "Shared check %s/%s is not a mapping; skipped.", f.stem, check_id
                )
                continue
            out[f"{f.stem}/{check_id}"] = _jsonable(body)
    return out


def _read_equivalence_forms(forms_file: Path) -> Dict[str, Dict[str, Any]]:
    """Read ``shared/equivalence_forms.yaml`` into ``{form name: pipeline}``.

    A third pack-root sharing mechanism beside ``shared/concepts/`` and ``shared/checks/``,
    and the flattest of the three: one file, so a form name is the key directly with no
    ``<stem>/`` namespace to collide over. A form declares what "the same thing" means for
    this deployment — the relation is the pack's, never the engine's — and every condition
    that counts or compares text may name one.

    Missing file returns ``{}``, which is the single-relation behaviour every pack had before
    forms existed. A malformed file is skipped with a warning rather than emptying the pack:
    a condition naming a form nobody declares is refused at evaluation and reported by
    ``pack_validate``, which is a louder failure than a pack that would not load.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not forms_file.is_file():
        return out
    try:
        doc = _read_yaml(forms_file) or {}
    except Exception as e:
        logger.warning("Skipping equivalence forms file %s: %s", forms_file, e)
        return out
    if not isinstance(doc, dict):
        logger.warning(
            "Equivalence forms file %s is not a mapping; skipped.", forms_file
        )
        return out
    for name, body in doc.items():
        if not isinstance(body, dict):
            logger.warning("Equivalence form %s is not a mapping; skipped.", name)
            continue
        out[str(name)] = _jsonable(body)
    return out


def _resolve_check_imports(
    spec: Dict[str, Any], library: Dict[str, Dict[str, Any]], ruleset_key: str = ""
) -> Dict[str, Any]:
    """Expand each condition's ``use: <file>/<id>`` against the shared check library.

    The importing condition's keys are laid over the library entry (one-level merge). For
    lists, the importer's value replaces entirely; a deep merge would silently union field
    paths and select leaves that do not exist on the target source. The ``lookup`` key is
    the one exception: merged per sub-key since its fields are independent.

    Raises ``ValueError`` on an unresolvable import. A dropped condition would be
    indistinguishable from a source that returned nothing; the error must surface at load.

    A composite's ``children`` are conditions, so they import by the same route and are
    resolved by the same recursion — an unresolved child would leave the parent combining a
    member with no kind, which the evaluator can only report as ``unknown``.
    """

    def imports_anything(items: Any) -> bool:
        return isinstance(items, list) and any(
            isinstance(c, dict)
            and (c.get("use") or imports_anything(c.get("children")))
            for c in items
        )

    def resolve_one(cond: Any) -> Any:
        if not isinstance(cond, dict):
            return cond
        merged = cond
        if cond.get("use"):
            ref = str(cond["use"]).strip()
            base = library.get(ref)
            if base is None:
                available = ", ".join(sorted(library)) or (
                    "(none — is shared/checks/ present?)"
                )
                raise ValueError(
                    f"Ruleset '{ruleset_key or '?'}' condition imports an unknown shared "
                    f"check '{ref}'. Available: {available}"
                )
            merged = dict(base)
            for k, v in cond.items():
                if k == "use":
                    continue
                if (
                    k == "lookup"
                    and isinstance(v, dict)
                    and isinstance(merged.get(k), dict)
                ):
                    merged[k] = {**merged[k], **v}
                else:
                    merged[k] = v
            # Default id from the library key keeps the same id across use cases.
            merged.setdefault("id", ref.split("/")[-1])
        if imports_anything(merged.get("children")):
            merged = {
                **merged,
                "children": [resolve_one(k) for k in merged["children"]],
            }
        return merged

    conditions = spec.get("conditions")
    if not imports_anything(conditions):
        return spec  # nothing imports: the identical object, no copy, no behaviour change
    return {**spec, "conditions": [resolve_one(c) for c in conditions]}


def _render_schema_doc(source: str, kind: str, table: str, tbl: Dict[str, Any]) -> str:
    """Render one table's field inventory as retrievable prose."""
    lines: List[str] = [f"Source: {source} (backend: {kind})", f"Table: {table}"]
    for key, label in (
        ("fully_qualified_name", "Fully-qualified name"),
        ("index_pattern", "Index pattern"),
        ("discovery", "How this inventory was discovered"),
        ("sampled_docs", "Documents sampled"),
    ):
        if tbl.get(key) not in (None, "", []):
            lines.append(f"{label}: {tbl[key]}")
    if tbl.get("partition_columns"):
        lines.append(
            "Partition columns (MUST be bounded in every query): "
            + ", ".join(str(p) for p in tbl["partition_columns"])
        )
    if tbl.get("description"):
        lines.append(f"Purpose: {tbl['description']}")
    skipped = (tbl.get("measured") or {}).get("skipped_reason")
    if skipped:
        lines.append(f"Population NOT measured: {skipped}")

    lines.append("")
    lines.append("Fields:")
    # Two shapes: Databricks tables nest leaves under `columns`, ES indices are flat
    # under `fields`. Both render to one line per field so the text is uniform.
    for cname, col in (tbl.get("columns") or {}).items():
        col_desc = (col or {}).get("description") or ""
        if col_desc:
            lines.append(f"- {cname} ({(col or {}).get('type', '?')}): {col_desc}")
        for leaf in (col or {}).get("leaves") or []:
            bits = [f"  - {leaf.get('path')} : {leaf.get('type')}"]
            if leaf.get("explode_path"):
                # Without this a generator writes a dotted path that is not valid SQL.
                bits.append(
                    f"[inside array {leaf['explode_path']} — reachable only via "
                    f"explode({leaf['explode_path']})]"
                )
            if leaf.get("populated_pct") is not None:
                bits.append(f"[populated {leaf['populated_pct']}%]")
            if leaf.get("description"):
                bits.append(f"— {leaf['description']}")
            lines.append(" ".join(bits))
    for fname, fld in (tbl.get("fields") or {}).items():
        bits = [f"- {fname} : {','.join((fld or {}).get('json_types') or []) or '?'}"]
        if (fld or {}).get("searchable") is False:
            bits.append("[NOT searchable — returnable only, never a filter]")
        if (fld or {}).get("present_in_sampled_pct") is not None:
            bits.append(f"[present {fld['present_in_sampled_pct']}%]")
        if (fld or {}).get("description"):
            bits.append(f"— {fld['description']}")
        lines.append(" ".join(bits))
    return "\n".join(lines)


def _read_schemas(schemas_dir: Path) -> List[Dict[str, Any]]:
    """Read ``schemas/<source>.yaml`` field inventories into retrievable RAG documents.

    Retriever schema discovery stops at an ``ARRAY<STRUCT<...>>``, since a dotted path
    into an array is not valid SQL, so an array's leaves are invisible to the query
    generator. These files hold the full inventory with arrays descended and an
    ``explode_path`` on every leaf that needs one.

    One document per table, RAG-only. The whole inventory is too large for a prompt;
    completeness lives in the corpus and selectivity is applied at retrieval
    (``type='schema'``, one table at a time). Nothing here is added to ``catalog_prompt``.
    """
    docs: List[Dict[str, Any]] = []
    if not schemas_dir.is_dir():
        return docs
    for path in sorted(schemas_dir.glob("*.yaml")) + sorted(schemas_dir.glob("*.yml")):
        data = _read_yaml(path)
        if not isinstance(data, dict):
            continue
        source = str(data.get("source") or path.stem)
        kind = str(data.get("kind") or "unknown")
        tables = data.get("tables")
        if not isinstance(tables, dict):
            continue
        for table, tbl in tables.items():
            if not isinstance(tbl, dict):
                continue
            n_leaves = sum(
                len((c or {}).get("leaves") or [])
                for c in (tbl.get("columns") or {}).values()
            ) + len(tbl.get("fields") or {})
            docs.append(
                {
                    "title": f"Field schema: {source} / {table}",
                    "content": _render_schema_doc(source, kind, str(table), tbl),
                    "type": "schema",
                    "metadata": _jsonable(
                        {
                            "schema_source": source,
                            "table": str(table),
                            "backend_kind": kind,
                            "leaf_count": n_leaves,
                            "partition_columns": tbl.get("partition_columns") or [],
                        }
                    ),
                }
            )
    return docs


def _read_data_dir(data_dir: Path) -> Dict[str, Any]:
    """Read every ``*.yaml`` under ``data_dir`` into a dict keyed by file stem.

    Missing dir -> {} (back-compat). Each file's parsed contents become one entry so a
    condition can reference a named lookup table (e.g. ``issuer_prefix_map``). Values
    are ``_jsonable``-coerced so downstream JSON dumps never choke on bare YAML dates.
    """
    out: Dict[str, Any] = {}
    if not data_dir.is_dir():
        return out
    for f in sorted(data_dir.glob("*.yaml")) + sorted(data_dir.glob("*.yml")):
        try:
            out[f.stem] = _jsonable(_read_yaml(f))
        except Exception as e:  # a malformed data file must not break the whole pack
            logger.warning("Skipping pack data file %s: %s", f, e)
    return out


def _read_use_cases(
    use_cases_dir: Path,
    rulesets: Dict[str, Any],
    existing_playbook_ids: set,
) -> Dict[str, Any]:
    """Walk ``use_cases/<name>/`` merging rules + collecting playbooks/concepts/cases/data.

    For each sub-dir: merge ``rules.yaml`` ``verdicts`` into ``rulesets['verdicts']``
    (use_cases win on key collision), collect ``playbooks/`` (deduped vs the flat root by
    ``playbook_id``), ``concepts/``, ``cases/`` (all tagged with the use_case name), and
    ``data/*.yaml`` (keyed by file stem; use_cases win over flat-root data on collision).
    Mutates ``rulesets`` in place; returns {playbooks, concepts, cases, data}.
    """
    playbooks: List[Dict[str, Any]] = []
    concepts: List[Dict[str, Any]] = []
    cases: List[Dict[str, Any]] = []
    data: Dict[str, Any] = {}
    reporting: Dict[str, Any] = {}
    if not use_cases_dir.is_dir():
        return {
            "playbooks": playbooks,
            "concepts": concepts,
            "cases": cases,
            "data": data,
            "reporting": reporting,
        }
    for uc_dir in sorted(p for p in use_cases_dir.iterdir() if p.is_dir()):
        use_case = uc_dir.name
        # (1) merge the use case's rules.yaml verdicts (use_cases wins).
        uc_rules = _read_yaml(uc_dir / "rules.yaml")
        uc_verdicts = (uc_rules or {}).get("verdicts")
        if isinstance(uc_verdicts, dict):
            rulesets.setdefault("verdicts", {})
            rulesets["verdicts"].update(uc_verdicts)
        # (2) playbooks: dedupe vs flat root by playbook_id.
        for doc in _read_markdown_docs(uc_dir / "playbooks", "playbook", use_case):
            pid = (doc.get("metadata", {}) or {}).get("playbook_id")
            if pid and pid in existing_playbook_ids:
                continue
            if pid:
                existing_playbook_ids.add(pid)
            playbooks.append(doc)
        # (3) concepts + (4) cases + (5) data.
        concepts.extend(_read_markdown_docs(uc_dir / "concepts", "concept", use_case))
        cases.extend(_read_markdown_docs(uc_dir / "cases", "case", use_case))
        data.update(_read_data_dir(uc_dir / "data"))
        # (6) report vocabulary, scoped to this use case.
        uc_reporting = _read_yaml(uc_dir / "reporting.yaml")
        if uc_reporting:
            reporting[use_case] = _jsonable(uc_reporting)
    return {
        "playbooks": playbooks,
        "concepts": concepts,
        "cases": cases,
        "data": data,
        "reporting": reporting,
    }


def load_knowledge_pack(pack_dir) -> KnowledgePack:
    """Load a knowledge pack from a directory; returns an empty pack if absent."""
    pack_dir = Path(pack_dir)
    if not pack_dir.is_dir():
        logger.warning("Knowledge pack dir %s not found; using empty pack.", pack_dir)
        return KnowledgePack(name=pack_dir.name or "default")

    glossary = _read_yaml(pack_dir / "entity_glossary.yaml")
    catalog = _read_yaml(pack_dir / "source_catalog.yaml")
    # Flat-root rules (back-compat); use_cases/<name>/rules.yaml merges on top (wins).
    rulesets = _read_yaml(pack_dir / "rulesets.yaml")

    entities = [EntityDef(**e) for e in (glossary.get("entities") or [])]
    abbreviations = {
        str(k): str(v) for k, v in (glossary.get("abbreviations") or {}).items()
    }
    field_hints = {
        str(k): str(v) for k, v in (glossary.get("field_name_hints") or {}).items()
    }
    sources = [SourceDef(**s) for s in (catalog.get("sources") or [])]
    playbooks = _read_playbooks(pack_dir / "playbooks")

    # use_cases/ subtree: per-use-case rules + playbooks + concepts + cases. Flat-only
    # packs skip this cleanly (the dir simply doesn't exist).
    existing_ids = {
        (d.get("metadata", {}) or {}).get("playbook_id")
        for d in playbooks
        if (d.get("metadata", {}) or {}).get("playbook_id")
    }
    uc = _read_use_cases(pack_dir / "use_cases", rulesets, existing_ids)
    playbooks.extend(uc["playbooks"])
    # Shared concepts loaded without a use_case tag so every use case can retrieve them.
    # ``concepts_for`` unions these with the requesting use case's own.
    concepts = _read_markdown_docs(pack_dir / "shared" / "concepts", "concept")
    concepts.extend(uc["concepts"])
    cases = uc["cases"]
    # Reference/lookup data: flat-root data/ first, use_cases/*/data/ win on collision.
    pack_data = _read_data_dir(pack_dir / "data")
    pack_data.update(uc["data"])
    # Report vocabulary: the flat root is the domain-wide base; per-use-case files hang off
    # a `use_cases` key so a slot can be resolved most-specific-first (see `report_phrase`).
    reporting = _jsonable(_read_yaml(pack_dir / "reporting.yaml")) or {}
    if uc["reporting"]:
        reporting["use_cases"] = uc["reporting"]
    # Field inventories. RAG-only by construction; see `_read_schemas`.
    schemas = _read_schemas(pack_dir / "schemas")
    # Resolved lazily in ``ruleset_spec`` so both consumers see the same expanded list,
    # but validated eagerly here: a misspelled import must fail at startup, not silently
    # at the first run that exercises that ruleset.
    shared_checks = _read_shared_checks(pack_dir / "shared" / "checks")
    equivalence_forms = _read_equivalence_forms(
        pack_dir / "shared" / "equivalence_forms.yaml"
    )
    for _key in (rulesets.get("verdicts") or {}):
        _resolve_check_imports(
            (rulesets["verdicts"] or {}).get(_key) or {}, shared_checks, _key
        )

    logger.info(
        "Loaded knowledge pack '%s': %d entities, %d sources, %d playbooks, "
        "%d concepts, %d cases, %d schema docs%s%s%s.",
        pack_dir.name,
        len(entities),
        len(sources),
        len(playbooks),
        len(concepts),
        len(cases),
        len(schemas),
        f", {len(shared_checks)} shared check(s)" if shared_checks else "",
        f", {len(equivalence_forms)} equivalence form(s)" if equivalence_forms else "",
        (
            f", {len((rulesets.get('verdicts') or {}))} verdict ruleset(s)"
            if rulesets.get("verdicts")
            else ""
        ),
    )
    return KnowledgePack(
        name=pack_dir.name,
        entities=entities,
        abbreviations=abbreviations,
        field_hints=field_hints,
        sources=sources,
        playbook_documents=playbooks,
        concept_documents=concepts,
        case_documents=cases,
        schema_documents=schemas,
        rulesets=rulesets,
        pack_data=pack_data,
        reporting=reporting,
        shared_checks=shared_checks,
        equivalence_forms=equivalence_forms,
        retrieval_defaults=_jsonable(catalog.get("retrieval") or {}) or {},
    )
