"""Pydantic v2 data contracts for the AFIR pipeline.

Typed objects that flow between stages: incident → understanding → retrieval → logs →
anomalies → report. These models are also the JSON schemas handed to the LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, PrivateAttr

# --- Incident input ---------------------------------------------------------


class IncidentInput(BaseModel):
    """An incident entering the pipeline, from API, IR lookup, or free text."""

    id: str
    timestamp: str
    description: str
    source: str = "api"  # "api" | "ir_lookup" | "freetext"


# --- Understanding ----------------------------------------------------------


class ExtractedEntity(BaseModel):
    """A domain entity pulled out of the incident text.

    ``type`` is one of the entity types defined in the active knowledge pack's
    glossary (e.g. ``org_unit``, ``user``, ``record``, ``document``). ``value`` is the
    normalized identifier used for filtering; ``raw`` preserves the original text.
    """

    type: str = Field(description="Entity type from the knowledge-pack glossary.")
    value: str = Field(description="Normalized identifier used to filter logs.")
    raw: str = Field(default="", description="Original text the value came from.")
    # Set by the engine from value_forms regexes; empty when none declared; never set by the LLM.
    value_form: str = Field(
        default="",
        description=(
            "Leave empty. Assigned by the engine from the knowledge pack's value_forms."
        ),
    )
    # Set by the engine. str not List[str]: fails harmlessly when the LLM fills it. Readers
    # must .split() it.
    co_occurrence: str = Field(
        default="",
        description=(
            "Leave empty. Assigned by the engine from the incident text's own layout."
        ),
    )


class EventWindow(BaseModel):
    """The time range the incident actually occurred in, parsed from its text.

    Distinct from the ingestion timestamp: a free-text incident describing an event
    on 2024-08-26 must search 2024-08-26, not the moment it was submitted.
    """

    start: str = Field(description="Event start, ISO 8601 (date or datetime).")
    end: str = Field(description="Event end, ISO 8601 (date or datetime).")


class IncidentAnalysis(BaseModel):
    """Structured analysis produced by the understanding stage."""

    incident_summary: str
    severity: str = Field(
        default="",
        description=(
            "Severity as the incident states it, verbatim (e.g. '2', 'P2', 'HIGH'). Empty "
            "when the incident states none. NEVER a score you assign yourself."
        ),
    )
    severity_reasoning: str
    impact_assessment: str
    key_investigation_areas: List[str]
    log_sources_to_review: List[str]
    initial_hypotheses: List[str]
    recommended_actions: List[str]
    stakeholder_notification: List[str]
    extracted_entities: List[ExtractedEntity] = Field(
        default_factory=list,
        description="Domain entities found in the incident (org unit, user, record, ...).",
    )
    correlation_keys: List[str] = Field(
        default_factory=list,
        description=(
            "Entity types the retrieved sources should be correlated/joined on "
            "(e.g. 'user', 'record', 'org_unit'). Used by the correlation stage to line up "
            "the same entity across sources; falls back to extracted_entities when empty."
        ),
    )
    event_time: Optional[EventWindow] = Field(
        default=None,
        description="When the incident occurred, parsed from the text if present.",
    )
    # Separate from incident_summary: the summary selects the adjudicating procedure.
    requested_links: List[str] = Field(
        default_factory=list,
        description=(
            "Other fraud types the incident text EXPLICITLY asks to also be checked for, in "
            "its own words. Empty unless the text asks. Never a type you infer, and never "
            "repeated into any other field."
        ),
    )
    # Stamped by the engine from link_pin; the model's value is discarded.
    pinned_use_case: str = Field(
        default="",
        description=(
            "Leave empty. Set by the system when this incident is a referral from another run; "
            "never something to infer from the text."
        ),
    )


class UnderstandingResult(BaseModel):
    """Wraps the analysis with the originating incident id."""

    incident_id: str
    analysis: IncidentAnalysis
    # Ingestion timestamp; anchor for default window resolution. Set in code, never by the LLM.
    incident_timestamp: str = ""


# --- Retrieval --------------------------------------------------------------


class RetrievalQuery(BaseModel):
    """
    A request for logs. ``natural_language_query`` is the text a retriever turns
    into a backend query (``SQL`` / ``ES|QL`` today, or feeds a future Genie call verbatim).
    """

    target_log_source: str = Field(
        description="Name of the configured log source to query."
    )
    natural_language_query: str = Field(
        description="Plain-language description of exactly what logs to retrieve."
    )
    # Convenience scalars for sources with no entity mapping; `entities` is the typed path.
    scope_id: str = Field(
        default="*",
        description=(
            "Identifier of the organisational scope the incident sits in (the unit, "
            "tenant, or site owning the activity), or '*' if unknown."
        ),
    )
    actor_id: str = Field(
        default="*",
        description="Identifier of the suspected acting identity, or '*' if unknown.",
    )
    date_from: str = Field(description="Start date, format yyyy-MM-dd.")
    date_to: str = Field(description="End date, format yyyy-MM-dd.")
    entities: List[ExtractedEntity] = Field(
        default_factory=list,
        description="Incident entities to filter on; mapped to real fields per source.",
    )
    # Private: not exposed to the LLM. Full entity set (entities holds only bindable types)
    # so guards can detect type-column contradictions. Set by _enrich_queries.
    _incident_entities: List[ExtractedEntity] = PrivateAttr(default_factory=list)
    # Private: co-occurrence is a fact about retrieved rows, not something a planner knows.
    # Set by _follow_up_query; read by query_guards.enforce_value_tuples.
    _value_tuples: List[List[dict]] = PrivateAttr(default_factory=list)


# --- Backend query models (LLM-generated, executed by retrievers) -----------


class EsqlQuery(BaseModel):
    """An Elasticsearch ``ES|QL`` query string generated by the LLM."""

    query: str = Field(description="A complete ES|QL query string.")


class SqlQuery(BaseModel):
    """A Databricks SQL query string generated by the LLM."""

    query: str = Field(description="A complete ANSI SQL SELECT query.")


class EsQueryDsl(BaseModel):
    """An Elasticsearch Query DSL query object generated by the LLM.

    Used by the Kibana-gateway retriever, which reaches only ``_search`` (Query DSL);
    ``ES|QL`` and the raw REST API are blocked. ``query`` is the contents of the top-level
    ``query`` key (e.g. a ``bool``/``range``/``match`` object).
    """

    query: dict = Field(
        description=(
            "An Elasticsearch Query DSL object — the value of the top-level `query` "
            'key (e.g. {"bool": {"filter": [...]}}). Do NOT wrap it in another '
            "`query` key, and do not include `size`/`from`/`sort`."
        )
    )


class SchemaSelection(BaseModel):
    """The fraud-relevant tables the LLM picks from a catalog scan.

    Each entry is a fully-qualified ``catalog.schema.table`` name. Used once at
    startup to narrow a whole catalog down to the tables worth querying per incident.
    """

    tables: List[str] = Field(
        description="Fully-qualified catalog.schema.table names relevant to fraud investigation."
    )
    reasoning: str = Field(
        default="", description="Brief justification for the selection."
    )


# --- Field mapping ----------------------------------------------------------


class EntityFieldMap(BaseModel):
    """Binds one incident entity type to a real field/column of a chosen source."""

    entity_type: str = Field(description="Entity type, e.g. 'org_unit' or 'user'.")
    field: str = Field(description="Actual field/column name in this source.")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="How sure the mapping is correct."
    )


class FieldMapping(BaseModel):
    """Per-source entity-to-field map, computed against each source's discovered schema.

    The same entity maps to different field names across sources. Entities with no
    plausible field are omitted.
    """

    mappings: List[EntityFieldMap] = Field(default_factory=list)


# --- Anomalies --------------------------------------------------------------


class AnomalyItem(BaseModel):
    description: str
    supporting_data: str
    potential_implications: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    recommended_actions: str
    patterns: str
    below_threshold: bool = Field(
        default=False,
        description="Set by the ENGINE (any model-supplied value is overwritten): True "
        "when this item scored below the effective confidence threshold, so it is "
        "retained for the record but omitted from the narrative.",
    )


class AnomalyList(BaseModel):
    anomalies: List[AnomalyItem]


# --- Correlation ------------------------------------------------------------


class CorrelationFinding(BaseModel):
    """A narrated cross-entity / cross-source pattern (LLM-produced when feasible)."""

    title: str
    description: str
    supporting_entities: List[str] = Field(default_factory=list)


# --- Pattern-driven transforms ----------------------------------------------


class TransformStep(BaseModel):
    """One declarative transformation the LLM plans from the matched playbook.

    The LLM emits these; a deterministic Python executor runs them over the rows
    already retrieved into memory (no generated SQL/code is executed). Each ``op``
    reads only the fields it needs; the rest stay at their defaults. Fields must be
    chosen from the actual per-source columns provided to the planner, never invented.
    """

    op: Literal[
        "group_by", "cross_source_overlap", "time_bucket", "threshold", "distinct"
    ] = Field(description="Which transformation to apply.")
    label: str = Field(description="Short human-readable name for this step's result.")
    source: str = Field(
        default="", description="Source name this step reads (single-source ops)."
    )
    sources: List[str] = Field(
        default_factory=list, description="Source names (cross_source_overlap)."
    )
    keys: List[str] = Field(
        default_factory=list, description="Grouping columns (group_by)."
    )
    field: str = Field(
        default="",
        description="Column operated on (distinct/time_bucket; agg=distinct target).",
    )
    agg: Literal["count", "distinct"] = Field(
        default="count", description="Aggregation for group_by."
    )
    entity: str = Field(
        default="", description="Column whose values to overlap (cross_source_overlap)."
    )
    bucket: str = Field(
        default="1d", description="Time bucket size for time_bucket, e.g. '1h' or '1d'."
    )
    operator: Literal[">", ">=", "<", "<=", "=="] = Field(
        default=">", description="Comparison for threshold."
    )
    value: float = Field(default=0.0, description="Threshold value to compare against.")
    over: str = Field(
        default="",
        description="threshold: label of a prior group_by/time_bucket step to filter.",
    )
    time_window: str = Field(
        default="",
        description=(
            "cross_source_overlap: optional co-occurrence window, e.g. 'same_day' or "
            "'within:24h'. When set, a shared value counts only if the sources' events "
            "fall within this window of each other."
        ),
    )
    time_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="cross_source_overlap: {source: timestamp field} for time_window.",
    )


class TransformPlan(BaseModel):
    """The per-incident set of transforms the LLM derives from the fraud playbook."""

    steps: List[TransformStep] = Field(default_factory=list)
    reasoning: str = Field(
        default="", description="Why these transforms fit the incident's fraud pattern."
    )


class TransformResult(BaseModel):
    """The computed output of one TransformStep (produced by the Python executor)."""

    label: str
    op: str
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    note: str = Field(
        default="", description="Executor note, e.g. why a step was empty."
    )


class CorrelationKey(BaseModel):
    """One resolved correlation/join key: an entity type and the real field per source.

    Resolved in precedence order (playbook-declared, understanding-derived,
    data-discovered) to build deterministic cross-source joins. ``origin`` records
    which layer supplied it.
    """

    entity_hint: str = Field(
        description="Entity type this key represents (e.g. 'record')."
    )
    sources: Dict[str, str] = Field(
        default_factory=dict, description="{source_name: real field path} per source."
    )
    time_window: str = Field(
        default="",
        description="Optional co-occurrence window, e.g. 'same_day', 'within:24h'.",
    )
    time_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="{source_name: timestamp field} for time-window joins.",
    )
    origin: str = Field(
        default="", description="playbook | understanding | discovered."
    )
    overlap_score: float = Field(
        default=0.0, description="Value-overlap score (discovered keys only)."
    )


# --- Investigation evidence -------------------------------------------------


class ChronologyEvent(BaseModel):
    """One time-ordered event in the merged cross-source chronology.

    ``count`` > 1 and a populated ``examples`` list mean this is an aggregated
    group (collapsed by actor+action+time-bucket) rather than a single raw event.
    """

    timestamp: str = Field(
        default="", description="ISO-8601 normalized; '' if unparseable."
    )
    epoch: Optional[float] = Field(
        default=None, description="Sort key; None sorts last."
    )
    source: str = ""
    actor: str = ""
    action: str = ""
    entities: Dict[str, str] = Field(default_factory=dict)
    count: int = Field(
        default=1, description=">1 when this is an aggregated event group."
    )
    examples: List[str] = Field(default_factory=list)
    is_subject: bool = Field(
        default=False,
        description=(
            "True when this event involves an incident person-of-interest (actor or a "
            "touched entity matches an extracted value). Subject events are rendered "
            "first so the narrator focuses on the incident, not same-window background."
        ),
    )


class ActorRollup(BaseModel):
    """Per-actor attribution: what one identity did, when, and where."""

    actor: str
    event_count: int = 0
    action_counts: Dict[str, int] = Field(default_factory=dict)
    first_seen: str = ""
    last_seen: str = ""
    sources: List[str] = Field(default_factory=list)
    entities_touched: Dict[str, List[str]] = Field(default_factory=dict)
    is_subject: bool = Field(
        default=False,
        description=(
            "True when this actor is a person-of-interest from the incident itself "
            "(matches an extracted entity value), vs. background activity that merely "
            "shares the time window. Subjects are ranked first and drive containment."
        ),
    )


class SourceEvidence(BaseModel):
    """Trimmed, aggregated per-source view kept for the LLM (raw rows dropped)."""

    source: str
    record_count: int = 0
    kept_columns: List[str] = Field(default_factory=list)
    dropped_columns: List[str] = Field(default_factory=list)
    aggregated: bool = False
    row_limited: bool = Field(
        default=False,
        description=(
            "True when the source returned EXACTLY its configured row cap, so the row "
            "count is an artifact of the cap and the real total is unknown (>= this)."
        ),
    )
    zero_rows_meaning: str = Field(
        default="",
        description=(
            "What ZERO rows from this source means, when the pack declared it: for a "
            "source whose purpose is an exclusion lookup, empty is the ANSWER and not a "
            "gap. Empty string when undeclared or when the source returned rows."
        ),
    )
    distributions: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="column -> {value: count} top values."
    )
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    adjudicated_values: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "column -> its top distinct VALUES, for the columns a verdict condition (or "
            "the ruleset's alert_record block) named — the ones this run's findings rest "
            "on. Held separately from `distributions`/`examples` because those are "
            "budget-degradable colour and these are the readings the verdict already "
            "made: a report that receives the column NAME with no value against it "
            "reports the value absent, contradicting the condition that read it. Empty "
            "when no ruleset adjudicated (nothing to promise) or the column was dropped."
        ),
    )


class EvidencePack(BaseModel):
    """Deterministic, budget-sized investigation evidence built from raw logs.

    Feeds both anomaly detection and report generation so the LLM sees an
    evidence-dense chronology (what/when/who) + cross-source joins instead of a
    truncated raw-JSON dump. Domain- and incident-agnostic.
    """

    chronology: List[ChronologyEvent] = Field(default_factory=list)
    chronology_aggregated: bool = False
    actors: List[ActorRollup] = Field(default_factory=list)
    sources: List[SourceEvidence] = Field(default_factory=list)
    cross_source_joins: List[Dict[str, Any]] = Field(default_factory=list)
    total_records: int = 0
    degraded: bool = Field(
        default=False, description="True if aggregated harder to fit the char budget."
    )
    notes: List[str] = Field(default_factory=list)


# --- Validation verdict (pack-driven, domain-agnostic) ----------------------


STUB_OBSERVED = "NOT EVALUATED"
"""The observed string the stub condition kind writes.

Separates "no data path" (authoring work) from "data failed to answer" (retrieval gap).
Import this constant; a reworded copy collapses the distinction silently.
"""


class ConditionCheck(BaseModel):
    """One evaluated condition of a validation ruleset against one subject.

    Domain-agnostic: ``id``/``label`` come from the pack, ``result`` is the machine
    verdict for this single check, ``expected``/``observed`` capture what the rule
    wanted vs. what the data showed, and ``detail`` is a short human note. Nothing here
    is procedure-specific; every label/threshold lives in the knowledge pack.
    """

    id: str = Field(description="Stable condition id from the pack.")
    label: str = Field(default="", description="Human-readable condition name.")
    result: Literal["pass", "fail", "unknown"] = Field(
        description="pass = condition satisfied; fail = violated; unknown = no data."
    )
    expected: str = Field(default="", description="What the rule required.")
    observed: str = Field(default="", description="What the data actually showed.")
    detail: str = Field(default="", description="Short human explanation.")
    decisive: bool = Field(
        default=False,
        description="True if this check alone can force the subject verdict per the pack.",
    )
    polarity: Literal["exclusion", "fraud_indicator"] = Field(
        default="exclusion",
        description=(
            "exclusion = a FAIL argues the incident is a FALSE POSITIVE (e.g. the record is "
            "richer than the pattern requires, a reversal is present); fraud_indicator = a "
            "FAIL is positive evidence of fraud (e.g. an issuer mismatch). Generic — every "
            "procedure specific lives in the pack."
        ),
    )
    exclusion_kind: Literal["heuristic", "categorical"] = Field(
        default="heuristic",
        description=(
            "For an exclusion, WHAT KIND of evidence it carries. heuristic = an INFERENCE "
            "about the event's shape (element counts, a clock difference) — a tripwire "
            "positive indicators can outweigh. categorical = an ATTRIBUTED FACT: it names an "
            "actor and a date a reviewer could look up in the record (a written access "
            "grant, a lineage element, a duty-coded operator identity, a confirmed automation "
            "profile). Once such a fact is on the record the behavioural indicators lose "
            "their meaning, not merely their weight, so the engine ranks it ahead of the "
            "indicator vote. Classify on the evidence CLASS, never on how many indicators "
            "happen to be firing. Ignored for fraud_indicator polarity. Generic — the "
            "pack declares which of its conditions are categorical."
        ),
    )
    subject: str = Field(
        default="",
        description=(
            "WHICH subject this check was evaluated against. Redundant inside "
            "ValidationSubject.checks, where the parent states it — and load-bearing the "
            "moment a check is lifted OUT of that parent into one of the brief's "
            "cross-subject lists, because there nothing else states the attribution. "
            "Stamped at that lift only, so a standalone check is unchanged. Generic: the "
            "value is whatever the pack made the subject."
        ),
    )
    group: str = Field(
        default="",
        description=(
            "Pack-declared reporting group id (see ValidationVerdict.condition_groups). "
            "Purely presentational — it decides WHICH table a check is printed in, never "
            "how it is weighed. A procedure that separates its mandatory validation steps "
            "from its optional disambiguation hints reads as one flat list of 19 rows "
            "without it, which is how a hint that decided a case became indistinguishable "
            "from a step that merely ran."
        ),
    )


class ConditionGroup(BaseModel):
    """A reporting bucket for condition checks, declared by the pack.

    Generic: the engine copies each condition's ``report_group`` onto its check and
    carries the pack's ordered group list on the verdict so the report can print one table
    per group, in the procedure's own order, with the procedure's own wording.
    """

    id: str = Field(description="Group id, matched against ConditionCheck.group.")
    title: str = Field(default="", description="Heading to print for this group.")
    description: str = Field(
        default="", description="Optional one-line note under the heading."
    )
    role: Literal["gate", "validation", "hint"] = Field(
        default="validation",
        description=(
            "Which of the report's three condition sections this group prints in. "
            "gate = answered BEFORE the subject is examined (out of scope is not a "
            "clearance). validation = a MANDATORY step, printed in the validation table. "
            "hint = an optional disambiguating signal, printed separately with its result. "
            "The distinction is the procedure's, not the engine's: a hint that fired is not "
            "a failed validation step, and printing them in one table is how an exculpatory "
            "hint became invisible among nineteen rows."
        ),
    )


class SubjectVerdict(BaseModel):
    """The rolled-up verdict for one subject (e.g. one record) across all conditions."""

    subject_type: str = Field(description="Entity type of the subject (e.g. 'record').")
    subject_value: str = Field(description="The subject identifier.")
    verdict: str = Field(
        description="Rolled-up label from the pack (e.g. VALID FRAUD / FALSE POSITIVE / "
        "INSUFFICIENT DATA)."
    )
    # Stamped here (the rollup is over ruleset inputs no consumer holds); empty on old verdicts.
    verdict_class: str = Field(
        default="",
        description="Machine-readable class of `verdict`, stamped by the rollup: one of "
        "fraud / false_positive / insufficient / out_of_scope / reject. Empty when "
        "unknown (a verdict from an older store).",
    )
    checks: List[ConditionCheck] = Field(default_factory=list)
    lock_target: Dict[str, str] = Field(
        default_factory=dict,
        description="Who to lock/contain (e.g. {scope, identity, source}); empty if unknown.",
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Extra observations (multi-actor, channel, scope, ...).",
    )


class ValidationVerdict(BaseModel):
    """Per-subject verdicts + a drafted notification, produced by the verdict engine.

    Generic container: ``label_scheme`` records which pack ruleset produced it, the
    labels themselves come from the pack, and no field names any procedure directly.
    Optional trailing field on ``CorrelationResult`` so existing construction stays valid.
    """

    label_scheme: str = Field(
        default="",
        description="Which pack ruleset produced this — the pack's own name for its "
        "procedure. Empty when the pack declares none; never defaulted here, because a "
        "default would print one domain's procedure name over another domain's verdict.",
    )
    subjects: List[SubjectVerdict] = Field(default_factory=list)
    notification_draft: str = Field(
        default="",
        description="Ready-to-send notification text drafted from the pack template.",
    )
    summary: str = Field(default="", description="One-line rollup across subjects.")
    degraded: bool = Field(
        default=False,
        description="True when some conditions could not be evaluated (missing data).",
    )
    condition_groups: List[ConditionGroup] = Field(
        default_factory=list,
        description=(
            "The pack's ordered reporting groups for the condition checks. Empty when the "
            "ruleset declares none, in which case every check prints in one table exactly "
            "as before."
        ),
    )
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="The ruleset's own verdict vocabulary ({fraud, false_positive, "
        "insufficient, ...} -> label), so a consumer can recognise a verdict class "
        "without matching prose.",
    )


class AssetTimelineEntry(BaseModel):
    """One dated event in the impact chronology of an asset (a record / document / sale).

    Domain-agnostic: produced deterministically by the use-case analyzer from the
    retrieved rows so the report can narrate create->issue->reverse->refund impact along
    the timeline without the LLM re-deriving it.
    """

    timestamp: str = Field(
        default="", description="ISO/display timestamp of the event."
    )
    epoch: float = Field(
        default=0.0, description="Sort key (epoch seconds); 0 if unknown."
    )
    event_type: str = Field(
        default="",
        description="What happened (e.g. created / issued / reversed / refunded).",
    )
    entity_type: str = Field(
        default="", description="Asset type acted on (record / document / sale)."
    )
    entity_value: str = Field(
        default="", description="Asset identifier (a record or document id...)."
    )
    actor: str = Field(default="", description="Acting identity/scope if known.")
    source: str = Field(default="", description="Log source the event came from.")
    detail: str = Field(
        default="", description="Short human note (route, amount, ...)."
    )


class ImpactedAsset(BaseModel):
    """One asset found by the scope-discovery sweep.

    A scope sweep is any retrieval scoped by the actor rather than by the already-known
    subject, so it can return subjects the alert never named. ``known`` distinguishes an
    asset the alert already named from newly discovered scope.
    """

    subject: str = Field(default="", description="Owning subject (e.g. the parent record).")
    asset_id: str = Field(
        default="", description="Asset identifier (e.g. the document number)."
    )
    amount: str = Field(default="", description="Value of the asset, as returned.")
    currency: str = Field(default="", description="Currency of `amount`, as returned.")
    status: str = Field(default="", description="Asset state as returned (e.g. T/V/I).")
    actor: str = Field(default="", description="Acting identity/scope that produced it.")
    timestamp: str = Field(default="", description="When it was created/issued.")
    known: bool = Field(
        default=False, description="True when the alert already named this subject."
    )
    # `event_date` is the asset's own event date; `in_window` distinguishes incident scope
    # from the actor's adjacent ordinary activity.
    event_date: str = Field(
        default="", description="The asset's own event date, as returned."
    )
    in_window: bool = Field(
        default=True,
        description="False when the asset's event date falls outside the incident window.",
    )


class SubjectLink(BaseModel):
    """A typed derivation relation between two subjects, read off the evidence.

    A derived subject appears in the actor-scoped sweep alongside genuinely new scope; this
    model distinguishes them. ``role`` is from ``subject``'s perspective: ``parent`` means
    ``related_subject`` was derived from it; ``child`` is the reverse.
    """

    subject: str = Field(default="", description="The subject the link was found on.")
    related_subject: str = Field(
        default="", description="The subject at the other end of the link."
    )
    role: Literal["parent", "child", "unknown"] = Field(
        default="unknown",
        description=(
            "`subject`'s role: parent = `related_subject` was derived from it; "
            "child = `subject` was derived from `related_subject`."
        ),
    )
    kind: str = Field(
        default="", description="Pack-supplied relation name (e.g. 'split')."
    )
    actor: str = Field(
        default="", description="Acting id recorded on the link, if any."
    )
    scope: str = Field(
        default="",
        description="Organisational scope recorded on the link, if any (the pack names it).",
    )
    occurred: str = Field(default="", description="When the link was created.")
    element_id: str = Field(
        default="", description="Backend id of the link element itself (evidence)."
    )
    status: str = Field(default="", description="Link element status, as returned.")
    quote: str = Field(
        default="",
        description=(
            "The link rendered the way the domain's operators write it, so a report can "
            "quote the element rather than paraphrase it."
        ),
    )


class ConceptRef(BaseModel):
    """A knowledge-base concept snippet surfaced into the brief for grounding."""

    concept_id: str = Field(description="Concept file stem, as authored in the pack.")
    title: str = Field(default="", description="Concept title.")
    snippet: str = Field(
        default="", description="Short excerpt injected into the LLM prompt."
    )
    source_path: str = Field(default="", description="Where the concept doc lives.")


class CasePrecedent(BaseModel):
    """A resolved past investigation matched to the current case for grounding."""

    case_id: str = Field(
        description="Stable case identifier (frontmatter or file stem)."
    )
    verdict: str = Field(default="", description="How the precedent was resolved.")
    subject: str = Field(
        default="", description="Subject of the precedent (e.g. a record id)."
    )
    decisive_reasons: List[str] = Field(
        default_factory=list, description="Why the precedent got its verdict."
    )
    resolution: str = Field(default="", description="Action taken / how it was closed.")
    date: str = Field(default="", description="When it was closed.")
    snippet: str = Field(default="", description="Short narrative excerpt.")
    source_path: str = Field(default="", description="Where the case doc lives.")


class DeclaredFact(BaseModel):
    """One fact the originating alert states, reconciled against the retrieved data.

    An alert field is ground truth; any divergence is a finding, not a preference.
    ``status`` is four-valued: ``confirmed`` (data agrees), ``not_found`` (data silent,
    a retrieval gap), ``mismatch`` (data carries a different value, never overwrite the
    alert), ``stated`` (alert asserts it and no corroborating source was declared, so
    there is nothing to reconcile against and it must not read as a gap).
    """

    field: str = Field(
        description="Which fact this is (entity type or alert field name)."
    )
    declared: str = Field(default="", description="The value the alert states.")
    found: str = Field(
        default="", description="The corresponding value seen in the retrieved data."
    )
    status: Literal["confirmed", "not_found", "mismatch", "stated"] = Field(
        default="not_found",
        description=(
            "confirmed = the data agrees; not_found = the data never carried this fact "
            "(a gap); mismatch = the data carries a different value (a defect to report, "
            "never a licence to override the alert); stated = the alert asserts it and no "
            "corroborating source was declared, so there is nothing to reconcile against."
        ),
    )
    source: str = Field(
        default="", description="Log source the `found` value was read from, if any."
    )
    note: str = Field(
        default="",
        description=(
            "Why this reading is qualified — e.g. the corroborating source was "
            "truncated at its row cap, or its query did not constrain the subject, so "
            "the values it carries cannot contradict the alert."
        ),
    )


class AlertFacts(BaseModel):
    """The originating alert record and its stated facts, reconciled against the data.

    An alert is whatever record the detector emitted (a ``SIEM`` document, a monitoring
    email, a webhook payload). The pack declares how to find it and which fields are
    authoritative; this model carries the outcome.

    ``located`` distinguishes the two failure modes that a bare empty object conflates:
    the alert record was never retrieved versus it was retrieved and simply carries no
    value for some field.
    """

    located: bool = Field(
        default=False,
        description="True when the incident's own alert record was found in the data.",
    )
    locator: str = Field(
        default="",
        description="How it was located (which id matched which field), or why it wasn't.",
    )
    source: str = Field(
        default="", description="Log source the alert record came from."
    )
    record_id: str = Field(default="", description="The alert record's own identifier.")
    declared_facts: List[DeclaredFact] = Field(
        default_factory=list, description="Alert-stated facts, each reconciled."
    )
    unrelated_records: List[str] = Field(
        default_factory=list,
        description=(
            "Records from the same source that belong to a DIFFERENT incident, named so "
            "the report cannot narrate another alert's values as this incident's."
        ),
    )
    trigger: str = Field(
        default="",
        description="What the detector fires on, verbatim from the ruleset's trigger block.",
    )

    def mismatches(self) -> List["DeclaredFact"]:
        """Declared facts the data contradicts (each one is a reportable defect)."""
        return [f for f in self.declared_facts if f.status == "mismatch"]


class LinkFinding(BaseModel):
    """One candidate link between the procedure that ran and a sibling procedure.

    Advisory only: a link may not be read by a condition, feed a rollup, or move a health
    score. ``state`` is one of four values (see ``src.links.LINK_STATES``) that must not
    collapse: "not looked" is not "looked and found nothing".
    """

    target_use_case: str = Field(
        default="",
        description="The sibling procedure this link points at, as the pack's use_cases/ dir "
        "names it.",
    )
    target_playbook_id: str = Field(
        default="",
        description="The sibling's playbook id, where the pack declares one.",
    )
    direction: str = Field(
        default="",
        description="Causal direction of the candidate link (see src.links.LINK_DIRECTIONS).",
    )
    state: str = Field(
        default="",
        description="How far the ladder got and what it concluded (see src.links.LINK_STATES).",
    )
    rung: int = Field(
        default=0,
        description="Highest ladder rung reached (0 pivot, 1 sibling's own scope gate, "
        "2 declared entry signal). Recorded so a reader can tell a cheap conclusion from a "
        "thorough one.",
    )
    pivot_entity: str = Field(
        default="",
        description="Entity type that would open the sibling's leg — its own subject_entity.",
    )
    pivot_values: List[str] = Field(
        default_factory=list,
        description="Values of that type this run actually has in hand (empty when none, "
        "which is what makes the link unreachable).",
    )
    signal_id: str = Field(
        default="",
        description="Id of the sibling's `entry_signals` declaration that fired, or empty "
        "when the candidate rests on the pivot and the sibling's gate alone.",
    )
    evidence_note: str = Field(
        default="",
        description="What was found, in the sibling's own terms, with counts. Never a claim "
        "about the sibling's verdict — that needs the sibling's own run.",
    )
    advisory_severity: str = Field(
        default="",
        description="Triage default for a HUMAN, router-added. NEVER the verdict severity and "
        "never an input to it.",
    )
    advisory_note: str = Field(
        default="",
        description="Why the advisory severity is what it is, so the reader can discount it.",
    )
    window_hint: str = Field(
        default="",
        description="Which window a referral should ask over, derived from `direction` "
        "(antecedent looks back from the event, consequent looks forward).",
    )
    gap_reason: str = Field(
        default="",
        description="Why the ladder stopped: the missing binding for `unreachable`, the "
        "withheld budget or opt-in for `not_probed`. Empty on a probed state.",
    )
    base_rate: str = Field(
        default="",
        description="The declaration's measured base rate restated in words, or empty when "
        "the pack shipped the signal unmeasured (which is reported, not silently trusted).",
    )
    # Carried here so both adjudication sites read the same answer.
    gate_outcome: str = Field(
        default="",
        description="The target procedure's own scope gate, re-evaluated against this run's "
        "retrieved rows: pass, fail, unknown, no_gate, no_sources or no_subject. Only `pass` "
        "licenses an automatic escalation; empty means the gate was never reached.",
    )
    mode: str = Field(
        default="planned",
        description="Escalation mode for this pair (see src.link_escalation.LINK_MODES). "
        "`planned` composes a referral a human executes; `semi_auto` is the deployment default.",
    )
    mode_source: str = Field(
        default="default",
        description="Which layer the effective mode came from: default, config, pack, job — or "
        "`clamp`, meaning an escalating mode was asked for and rung 1 refused it, or `score`, "
        "meaning rung 1 held and the confidence did not reach the threshold.",
    )
    mode_note: str = Field(
        default="",
        description="Why the effective mode is not the one that was asked for, where those "
        "differ. Empty when nobody asked for anything else.",
    )
    proposed_action: str = Field(
        default="",
        description="What the effective mode would DO, in the words the report and the UI both "
        "print. Deterministic from `mode` and from nothing else.",
    )
    mode_licensed: bool = Field(
        default=False,
        description="May this link escalate on its own in THIS run — i.e. did the target "
        "procedure's own scope gate hold against the rows this run retrieved? False holds the "
        "link at `planned` whatever any layer asks for.",
    )
    mode_corpus: int = Field(
        default=0,
        description="How many incidents the pair's best-measured base rate was counted over. 0 "
        "means nothing was counted, which withholds confidence and never permission.",
    )
    # mode_licensed is a permission; link_score is a quantity. The two must stay separate.
    link_score: float = Field(
        default=0.0,
        description="Deterministic confidence in this candidate, 0.0-1.0, summed from the free "
        "rungs that held (pivot in hand, the target's own applicability test, a declared entry "
        "signal firing, and that signal's measured discrimination). `semi_auto` acts at or above "
        "the configured threshold and composes a referral below it.",
    )
    link_score_reasons: List[str] = Field(
        default_factory=list,
        description="One printable line per term that contributed to `link_score`, with its "
        "weight, so the number can be audited rather than taken on faith.",
    )
    probe_spent: bool = Field(
        default=False,
        description="Did this candidate cost a retrieval of its own? True only when a rung-3 "
        "probe query actually ran; the rows it returned are never added to the logs the verdict "
        "reads.",
    )
    probe_source: str = Field(
        default="",
        description="The one source a rung-3 probe asked, chosen from the TARGET procedure's own "
        "declarations and never retrieved by this run. Empty when no probe ran.",
    )
    probe_note: str = Field(
        default="",
        description="What the probe cost and what it settled — or, when nothing was spent, which "
        "bound declined it. Empty when no escalating mode applied to this candidate.",
    )
    child_job_id: str = Field(
        default="",
        description="The job id of the full child run this candidate was adjudicated by, when "
        "one was launched. Empty otherwise, including on every candidate a human refers by hand — "
        "composing a referral launches nothing.",
    )
    child_note: str = Field(
        default="",
        description="What rung 4 DID about this candidate, in the words the UI prints: what was "
        "launched and how it was scoped, or — when a bound held — which bound and what it means. "
        "One field for both, mirroring `probe_note`, because the four states above say what was "
        "ADJUDICATED and neither of them says what was SPENT; `child_job_id` is this rung's spend "
        "marker the way `probe_spent` is rung 3's, so a note without an id is a refusal and not a "
        "launch. Empty only when rung 4 never considered the candidate at all.",
    )


class InvestigationBrief(BaseModel):
    """Deterministic distillation of raw rows, verdict, and KB into a compact brief
    fed to the report and anomaly LLM calls.

    Generic across use cases; every procedure-specific detail lives in the pack.
    Rides as an optional trailing field on ``CorrelationResult`` so existing
    construction stays valid.
    """

    use_case: str = Field(
        default="", description="Use-case id, as the pack's use_cases/ dir names it."
    )
    playbook_id: str = Field(default="", description="Matched playbook id.")
    verdict: Optional["ValidationVerdict"] = Field(
        default=None,
        description="The authoritative verdict the narrative must not contradict.",
    )
    alert_facts: Optional["AlertFacts"] = Field(
        default=None,
        description="Alert-stated facts + declared-vs-found reconciliation; None when "
        "the ruleset declares no alert record.",
    )
    decisive_fails: List["ConditionCheck"] = Field(
        default_factory=list,
        description="Decisive checks that FAILED (exclusion triggers).",
    )
    decisive_unknowns: List["ConditionCheck"] = Field(
        default_factory=list,
        description="Decisive checks with no data (drive INSUFFICIENT).",
    )
    decisive_indicators: List["ConditionCheck"] = Field(
        default_factory=list,
        description="Positive fraud-indicator checks that FAILED (drive VALID FRAUD).",
    )
    explanatory_fails: List["ConditionCheck"] = Field(
        default_factory=list,
        description="Non-decisive exclusion checks that FAILED: an innocent explanation "
        "was found, but it did not change the verdict class.",
    )
    unanswered_attributions: List["ConditionCheck"] = Field(
        default_factory=list,
        description="Checks that settle WHO acted and could NOT be answered — the "
        "narrative may not assert what they would have established.",
    )
    asset_timeline: List[AssetTimelineEntry] = Field(default_factory=list)
    # scope_status ensures an unrun sweep cannot read as a clean one.
    impacted_assets: List[ImpactedAsset] = Field(
        default_factory=list,
        description="Assets found by the actor-scoped scope sweep.",
    )
    scope_status: str = Field(
        default="",
        description="Outcome of the scope sweep in words (never silently empty).",
    )
    additional_subjects: List[str] = Field(
        default_factory=list,
        description="Subjects the sweep found that the alert did NOT name (new scope).",
    )
    subject_links: List[SubjectLink] = Field(
        default_factory=list,
        description="Parent/child derivation links found on the alerted subjects.",
    )
    lock_targets: List[Dict[str, str]] = Field(
        default_factory=list, description="Who to lock/contain per the verdict."
    )
    action_backbone: List[str] = Field(
        default_factory=list,
        description="Deterministic next-steps skeleton the LLM narrates from.",
    )
    containment_gated: bool = Field(
        default=False,
        description=(
            "True when policy requires expert confirmation BEFORE any containment, so no "
            "downstream text may recommend an immediate irreversible action."
        ),
    )
    concept_refs: List[ConceptRef] = Field(default_factory=list)
    precedents: List[CasePrecedent] = Field(default_factory=list)
    join_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-expected-join outcome ('ran, N matches' / 'not evaluated').",
    )
    degraded: bool = Field(
        default=False, description="True when data gaps limited the analysis."
    )
    notes: List[str] = Field(default_factory=list)
    unenforced_carve_outs: List[str] = Field(
        default_factory=list,
        description="Procedure carve-outs the engine does not apply, stated where the "
        "check they would have narrowed returned a finding.",
    )
    links: List["LinkFinding"] = Field(
        default_factory=list,
        description="Candidate links to sibling procedures. ADVISORY: addressed to a human, "
        "never readable by a condition and never an input to this run's verdict.",
    )


class CaseAssessment(BaseModel):
    """Return of a UseCaseAnalyzer: the brief plus the verdict it produced (if any)."""

    brief: InvestigationBrief = Field(default_factory=InvestigationBrief)
    verdict: Optional["ValidationVerdict"] = Field(default=None)


class CorrelationResult(BaseModel):
    """Output of the correlation stage: programmatic aggregates + optional narrative.

    ``aggregations`` holds the deterministic, computed-in-python summaries (counts
    per entity, per-day volumes, cross-source overlaps). ``transforms`` holds the
    results of the per-incident, playbook-driven TransformPlan executed in Python.
    ``findings`` holds optional LLM narration, populated only when the summarized
    input is small enough. ``record_count`` is the total rows correlated.
    ``evidence`` holds the deterministic investigation evidence pack (chronology,
    actor attribution, trimmed aggregates) fed to the downstream LLM stages.
    """

    aggregations: Dict[str, Any] = Field(default_factory=dict)
    transforms: List[TransformResult] = Field(default_factory=list)
    findings: List[CorrelationFinding] = Field(default_factory=list)
    record_count: int = 0
    summary_text: str = Field(
        default="", description="Compact text fed downstream to anomaly detection."
    )
    evidence: Optional["EvidencePack"] = Field(
        default=None, description="Deterministic investigation evidence pack."
    )
    verdict: Optional["ValidationVerdict"] = Field(
        default=None,
        description="Per-subject validation verdict against a pack ruleset (the domain's "
        "own official procedure, encoded); None when the pack ships no matching ruleset.",
    )
    brief: Optional["InvestigationBrief"] = Field(
        default=None,
        description="Deterministic use-case grounding brief fed to the report/anomaly "
        "LLM calls; None when no use-case analyzer ran.",
    )
    links: List["LinkFinding"] = Field(
        default_factory=list,
        description="Candidate links to sibling procedures, computed deterministically after "
        "the verdict. Do not populate this field: it is written by the engine, not narrated.",
    )


# --- Report -----------------------------------------------------------------


class ReportSection(BaseModel):
    """
    A report section. ``content`` is a string, a list of strings, or nested
    subsections, mirroring the structure the PDF builder traverses.
    """

    section_title: str
    content: Union[str, List[str], List["ReportSection"]]


class InvestigationReport(BaseModel):
    sections: List[ReportSection]


# --- Feedback ---------------------------------------------------------------


class FeedbackEntry(BaseModel):
    """One analyst review of a completed investigation.

    ``human_feedback`` is the original free-form field (any JSON shape). The
    structured fields let the distiller reason over agreement, what the verdict
    should have been, and what the run missed or over-called. Every field except
    ``incident_id`` and ``received_at`` is optional.
    """

    incident_id: str
    received_at: str
    human_feedback: Optional[Any] = None
    agrees_with_verdict: Optional[bool] = None
    analyst_verdict: Optional[str] = None
    missed_anomalies: List[str] = Field(default_factory=list)
    false_positives: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    analyst: Optional[str] = None
    job_id: Optional[str] = None
    investigation_result: Optional[Any] = None


class FeedbackInsights(BaseModel):
    common_success_patterns: List[str]
    frequently_missed_anomalies: List[str]
    accuracy_improvements: List[str]
    confidence_threshold_recommendations: List[str]
    recommended_action_effectiveness: List[str]
    new_fraud_patterns: List[str]
    process_improvements: List[str]


ReportSection.model_rebuild()
