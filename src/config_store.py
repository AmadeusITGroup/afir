"""
Read/validate/write the app's YAML configuration; backing store for the Configuration
UI and the ``/api/v1/config`` endpoints.

Secrets are redacted on read; ``${ENV_VAR}`` references are returned verbatim.
Writes patch scalars in place so comments survive; anything ambiguous is rejected.
Every writing function reports ``durable``: whether the bytes survive a restart.
"""

import logging
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.link_children import (
    DEFAULT_MAX_CHILD_DEPTH,
    DEFAULT_MAX_CONCURRENT_CHILDREN,
    MAX_CHILD_DEPTH_CEILING,
    MAX_CHILDREN_PER_RUN_CEILING,
    MAX_CONCURRENT_CHILDREN_CEILING,
    MAX_TOTAL_CHILDREN_DEFAULT,
)
from src.link_escalation import (
    DEFAULT_LINK_MODE,
    DEFAULT_MIN_ESCALATION_SCORE,
    LINK_MODES,
    MAX_PROBES_CEILING,
    PROBE_ROW_CAP_DEFAULT,
    PROBE_TIMEOUT_DEFAULT,
    PROBE_TIMEOUT_MAX,
)
from src.utils.paths import config_dir

logger = logging.getLogger(__name__)

#: Set by ``MirrorSet.install()`` in App mode; ``None`` on local deployments.
_MIRROR = None


def set_mirror(mirror) -> None:
    """Install (or clear, with ``None``) the durable mirror for the config tree."""
    global _MIRROR
    _MIRROR = mirror


def mirror() -> object:
    """The installed mirror, or ``None``. For the health surface and the tests."""
    return _MIRROR


#: Placeholder returned in place of a literal secret, and accepted on write to mean
#: "leave the stored value alone". Deliberately not a plausible real value.
REDACTED = "__redacted__"

#: Matches a whole-value ``${VAR}`` env-var reference (what the templates use).
_ENV_REF = re.compile(r"^\s*\$\{[A-Za-z_][A-Za-z0-9_]*\}\s*$")

#: The four config files the UI can see. ``name`` is the URL segment.
CONFIG_FILES: Tuple[str, ...] = (
    "main_config.yaml",
    "llm_config.yaml",
    "plugin_config.yaml",
    "logging_config.yaml",
)

#: Words that mark a leaf as secret. Whole ``_``-delimited parts only — ``token`` is in
#: ``report_max_tokens``, so a substring match would redact an LLM length limit.
_SECRET_WORDS = frozenset(
    {"password", "passwd", "secret", "token", "credential", "credentials"}
)

#: Leaf names that are secrets but hold no word from ``_SECRET_WORDS`` (``key`` alone is
#: too broad). ``dsn`` embeds its credential inside the URL value; redacted whole.
_SECRET_LEAVES = frozenset({"api_key", "apikey", "private_key", "access_key", "dsn"})

#: Suffixes naming an env var rather than holding a secret; never redacted.
_ENV_NAME_SUFFIXES = ("_env", "_envvar", "_env_var")


def is_secret_key(path: str) -> bool:
    """True when the leaf of a dotted key path names a secret-bearing field.

    ``_env`` suffix is checked first so indirection keys (``api_key_env``) are never
    redacted.
    """
    leaf = path.rsplit(".", 1)[-1].lower()
    if leaf.endswith(_ENV_NAME_SUFFIXES):
        return False
    if leaf in _SECRET_LEAVES or leaf.endswith(tuple(_SECRET_LEAVES)):
        return True
    return bool(_SECRET_WORDS.intersection(re.split(r"[_\-]+", leaf)))


def is_env_reference(value: Any) -> bool:
    """True when a value is a whole-string ``${VAR}`` reference."""
    return isinstance(value, str) and bool(_ENV_REF.match(value))


# Field descriptors: what the UI renders, and what a write is checked against.


class Field:
    """One editable config key: its location, shape, and ``applies`` scope (``"live"`` or ``"restart"``)."""

    __slots__ = (
        "path",
        "file",
        "kind",
        "applies",
        "label",
        "help",
        "choices",
        "minimum",
        "maximum",
        "default",
    )

    def __init__(
        self,
        path,
        file,
        kind,
        applies="restart",
        label=None,
        help="",
        choices=None,
        minimum=None,
        maximum=None,
        default=None,
    ):
        self.path = path
        self.file = file
        self.kind = kind  # number | integer | boolean | string | choice | text
        self.applies = applies
        self.label = label or path.rsplit(".", 1)[-1].replace("_", " ")
        self.help = help
        self.choices = choices
        self.minimum = minimum
        self.maximum = maximum
        # Shown as "unset – default" in the UI; writing it inserts the key.
        self.default = default

    def to_dict(self) -> dict:
        out = {
            "path": self.path,
            "file": self.file,
            "kind": self.kind,
            "applies": self.applies,
            "label": self.label,
            "help": self.help,
        }
        if self.choices:
            out["choices"] = list(self.choices)
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        if self.default is not None:
            out["default"] = self.default
        return out


_MAIN = "main_config.yaml"
_LLM = "llm_config.yaml"

#: Every ``stage=`` label passed to ``LLMClient``, with the advised thinking budget.
#: ``tests/test_llm_client.py`` asserts it matches call sites. The budget is a floor
#: (``LLMClient._thinking_floor``): reasoning bills against the answer's budget.
THINKING_STAGES: Tuple[Tuple[str, int, str], ...] = (
    (
        "incident_understanding",
        12000,
        "Entity + event-window extraction. Measured 34.5s with thinking against 62.2s "
        "without, and a schema response cannot stop early — a starved one is reported as "
        "whichever field went missing.",
    ),
    (
        "api_call_generation",
        12000,
        "Source selection from the catalog: one tool call per source, judged against "
        "`selection_guidance`. A source not chosen here is evidence the run never sees.",
    ),
    (
        "log_retrieval",
        8000,
        "Shared by ALL retriever calls — schema curation, entity field mapping and query "
        "generation across the ~19-source fan-out, so this one setting multiplies by "
        "roughly 40-60 calls at max_concurrency 4. The most expensive place to enable it.",
    ),
    (
        "correlation",
        10000,
        "Transform planning and narration. The deterministic verdict is computed in code "
        "and thinking cannot move it; this is the prose around it.",
    ),
    (
        "anomaly_detection",
        16000,
        "Long output: every anomaly with its evidence. Thinking here shortened the list "
        "(10 items to 9) and dropped an exculpatory finding, so raise the budget if you "
        "enable it — 8000 already truncated once and retried at 16000.",
    ),
    (
        "report_generation",
        16000,
        "The acceptance artifact, and the longest output in the pipeline. With thinking on "
        "at an unchanged budget the report came out 61297 chars against 63764.",
    ),
    (
        "pack_assistant",
        16000,
        "The Knowledge tab's authoring loop: multi-turn tool calls that read the pack and "
        "then propose an edit plan. Reasoning helps most where a wrong answer is a wrong "
        "pack edit, and it is operator-driven rather than on the pipeline's critical path.",
    ),
    (
        "feedback_distillation",
        8000,
        "Turns collected reviews into guidance. Off the critical path — it runs on a batch "
        "boundary, not during an investigation.",
    ),
)


def _thinking_fields() -> Tuple[Field, ...]:
    """Three controls per stage: mode, effort, advised max_tokens. Generated to avoid copy-paste."""
    fields: List[Field] = []
    for stage, advised, why in THINKING_STAGES:
        label = stage.replace("_", " ")
        fields.append(
            Field(
                f"thinking_by_stage.{stage}.mode",
                _LLM,
                "choice",
                "live",
                label=f"{label} — thinking",
                choices=("", "unset", "disabled", "adaptive"),
                default="",
                help=why
                + "  •  Blank = follow the global setting above. `unset` sends no "
                "thinking parameter at all, so the endpoint's own default applies (the "
                "pre-pinning baseline). `disabled` pins it off. `adaptive` is real extended "
                "thinking — `enabled` is rejected by this model.",
            )
        )
        fields.append(
            Field(
                f"thinking_by_stage.{stage}.effort",
                _LLM,
                "choice",
                "live",
                label=f"{label} — effort",
                choices=("", "low", "medium", "high"),
                default="",
                help="Only read when this stage's mode is `adaptive`. Blank = fall back to "
                "the global effort, and if that is unset too the endpoint picks. Higher "
                "effort spends more of the budget below on reasoning.",
            )
        )
        fields.append(
            Field(
                f"thinking_by_stage.{stage}.max_tokens",
                _LLM,
                "integer",
                "live",
                label=f"{label} — max tokens (advised {advised})",
                # 0, not 256, because 0 is how the operator says "no advised floor, use
                # whatever the caller asked for". A minimum of 256 would make the only way
                # to undo a floor be hand-editing the YAML.
                minimum=0,
                maximum=200000,
                help=f"Advised {advised} for this stage with thinking ON; 0 or blank = no "
                "floor, take the caller's own budget. A FLOOR, never a "
                "cap: a reasoning block bills against the same budget as the answer, so "
                "enabling thinking without more room takes the room out of the deliverable. "
                "A caller asking for more (a truncation retry's doubled budget) still wins.",
            )
        )
    return tuple(fields)


def _gate_stage_fields() -> Tuple[Field, ...]:
    """Per-stage gate override: enabled flag and threshold. Generated from ``GATEABLE_STAGES``
    so a stage not in that list cannot silently accept a setting it can never act on.
    """
    from stage_health import GATEABLE_STAGES

    fields: List[Field] = []
    for stage in GATEABLE_STAGES:
        label = stage.replace("_", " ")
        fields.append(
            Field(
                f"stage_gates.stages.{stage}.enabled",
                _MAIN,
                "boolean",
                "live",
                label=f"{label} — may gate",
                default=True,
                help="Off = this stage never opens an approval gate, whatever it scores. "
                "The run does not stop here even in supervised mode. Its health is still "
                "computed and shown on the card.",
            )
        )
        fields.append(
            Field(
                f"stage_gates.stages.{stage}.threshold",
                _MAIN,
                "number",
                "live",
                label=f"{label} — gate below",
                minimum=0.0,
                maximum=1.0,
                help="Overrides the global threshold for THIS stage only. Blank = use the "
                "global value. Raise it for a stage whose output you always want to read.",
            )
        )
    return tuple(fields)


#: Sections shown in the Configuration UI, in display order. A key absent from this
#: table is visible in the raw editor but not offered as a form control.
SECTIONS: Tuple[Tuple[str, str, Tuple[Field, ...]], ...] = (
    (
        "llm",
        "LLM endpoint",
        (
            Field(
                "base_url",
                _LLM,
                "string",
                "restart",
                help="OpenAI-compatible endpoint. Databricks Model Serving, OpenAI, "
                "or any compatible URL.",
            ),
            Field("model", _LLM, "string", "restart", help="Served model name."),
            Field(
                "api_key_env",
                _LLM,
                "string",
                "restart",
                label="API key env var",
                help="NAME of the environment variable holding the token — never the "
                "token itself.",
            ),
            Field(
                "max_tokens", _LLM, "integer", "restart", minimum=256, maximum=200000
            ),
            Field(
                "structured_output_max_tokens",
                _LLM,
                "integer",
                "restart",
                minimum=512,
                maximum=200000,
                default=8000,
                label="structured output max tokens",
                help="Output cap for JSON-schema responses (understanding, queries, "
                "correlation, anomalies) unless a stage overrides it. A cut response is a "
                "`finish_reason=length` error, NOT a smaller answer — raise this if a stage "
                "reports a required field missing. It is a CAP: an unused allowance costs "
                "nothing, so never scale it to the size of the input.",
            ),
            Field("temperature", _LLM, "number", "restart", minimum=0.0, maximum=2.0),
            Field(
                "timeout",
                _LLM,
                "integer",
                "restart",
                minimum=10,
                maximum=1800,
                help="Per-request seconds. Set generously: a tight timeout turns a "
                "slow-but-successful call into a retry storm.",
            ),
            Field(
                "max_concurrency",
                _LLM,
                "integer",
                "restart",
                minimum=1,
                maximum=64,
                help="Cap on in-flight LLM requests. Lower this first if the endpoint "
                "returns 429s.",
            ),
            Field(
                "requests_per_minute",
                _LLM,
                "integer",
                "restart",
                minimum=0,
                maximum=10000,
                help="QPS smoothing; 0 disables the limit.",
            ),
        ),
    ),
    (
        "thinking",
        "Extended thinking (per stage)",
        (
            Field(
                "thinking",
                _LLM,
                "choice",
                "live",
                label="Global thinking",
                choices=("unset", "disabled", "adaptive"),
                default="disabled",
                help="The default for every stage that does not override it below — "
                "INCLUDING the untagged retriever fan-out. `unset` sends no thinking "
                "parameter, so the endpoint's own default applies; that default is not "
                "stable (the same endpoint answered one prompt with a plain string and the "
                "next with a reasoning block), which is why it is pinned to `disabled` "
                "here. Ignored by endpoints that do not support thinking: a refusal is "
                "learned once and the call retried without it, never failed.",
            ),
            Field(
                "thinking_effort",
                _LLM,
                "choice",
                "live",
                label="Global effort",
                choices=("", "low", "medium", "high"),
                default="",
                help="Only read where the mode is `adaptive`. Blank = let the endpoint "
                "choose. Measured on one judgement call: low spent 1839 completion tokens "
                "in 27.0s, the endpoint's own default 4676 in 65.9s.",
            ),
        )
        + _thinking_fields(),
    ),
    (
        "gates",
        "Approval gates",
        (
            Field(
                "stage_gates.threshold",
                _MAIN,
                "number",
                "live",
                minimum=0.0,
                maximum=1.0,
                default=0.6,
                help="Gate a stage whose deterministic health score is below this. "
                "Calibrate against your own runs — it is a starting point, not a "
                "measured constant.",
            ),
            Field(
                "stage_gates.timeout_seconds",
                _MAIN,
                "integer",
                "live",
                minimum=0,
                maximum=2592000,
                default=0,
                help="Unset or 0 = hold forever (the default, on purpose). A timeout "
                "that expires is a decision made by a clock.",
            ),
            Field(
                "stage_gates.on_timeout",
                _MAIN,
                "choice",
                "live",
                choices=("hold", "proceed", "abort"),
                default="hold",
                help="hold = notify once and keep waiting. proceed/abort are recorded "
                "with actor 'timeout' so a run never reads as reviewed.",
            ),
        )
        + _gate_stage_fields(),
    ),
    (
        "detection",
        "Detection & correlation",
        (
            Field("anomaly_detection.use_llm", _MAIN, "boolean", "live"),
            Field(
                "anomaly_detection.threshold",
                _MAIN,
                "number",
                "live",
                minimum=0.0,
                maximum=1.0,
                help="Confidence floor for reporting an anomaly. This is the ONE value "
                "feedback may auto-tune, bounded against whatever is set here.",
            ),
            Field(
                "anomaly_detection.max_anomalies",
                _MAIN,
                "integer",
                "live",
                minimum=1,
                maximum=1000,
            ),
            Field(
                "correlation.sample_rows",
                _MAIN,
                "integer",
                "live",
                minimum=1,
                maximum=500,
                help="Rows per source shown to the LLM when it narrates.",
            ),
            Field(
                "correlation.llm_max_records",
                _MAIN,
                "integer",
                "live",
                minimum=0,
                maximum=1000000,
                default=2000,
                help="Volume gate: above this the deterministic plan is used and the "
                "LLM is skipped.",
            ),
            Field(
                "correlation.llm_max_sources",
                _MAIN,
                "integer",
                "live",
                minimum=0,
                maximum=100,
                default=6,
            ),
            Field(
                "correlation.discovery_key_filter",
                _MAIN,
                "choice",
                "live",
                choices=("strict", "cardinality", "both"),
                default="strict",
                help="How aggressively join keys are discovered from data when no "
                "playbook overrides it.",
            ),
            Field(
                "correlation.evidence_char_budget",
                _MAIN,
                "integer",
                "live",
                minimum=1000,
                maximum=200000,
            ),
            Field(
                "correlation.links.escalation_mode",
                _MAIN,
                "choice",
                "live",
                choices=LINK_MODES,
                default=DEFAULT_LINK_MODE,
                help="May a cross-procedure link ACT on its own, or only propose? The global "
                "default; a ruleset overrides it per pair and an operator per job. A link whose "
                "target procedure's own scope gate did not PASS on this run's evidence is held at "
                "planned whatever this says.",
            ),
            Field(
                "correlation.links.min_escalation_score",
                _MAIN,
                "number",
                "live",
                minimum=0.0,
                maximum=1.0,
                default=DEFAULT_MIN_ESCALATION_SCORE,
                help="The deterministic link score at or above which semi_auto acts by itself; "
                "below it the link composes a referral instead. 0.0 makes semi_auto behave "
                "exactly like auto.",
            ),
            # Rung-3 budgets. Maxima imported from the engine so the field cannot accept
            # a value the engine silently clamps. All `live`: re-read per run.
            Field(
                "correlation.links.max_probes_per_run",
                _MAIN,
                "integer",
                "live",
                minimum=0,
                maximum=MAX_PROBES_CEILING,
                default=0,
                help="How many cross-procedure candidates one run may confirm with a query of "
                "their own. 0 disables the probe rung entirely, which is the default: the free "
                "rungs still run and cost nothing. A probe also needs an escalating mode, a "
                "PASS from the target procedure's own scope gate on this run's rows, and "
                "auto_probe on the declaration that fired.",
            ),
            Field(
                "correlation.links.probe_timeout_seconds",
                _MAIN,
                "integer",
                "live",
                minimum=1,
                maximum=PROBE_TIMEOUT_MAX,
                default=PROBE_TIMEOUT_DEFAULT,
                help="Seconds one probe may run. It multiplies with the count above: the engine "
                "shortens each probe rather than lengthening the run if the product would exceed "
                "its own collective ceiling, so the whole lane can never approach the "
                "primary-source budget.",
            ),
            Field(
                "correlation.links.probe_row_cap",
                _MAIN,
                "integer",
                "live",
                minimum=1,
                maximum=10000,
                default=PROBE_ROW_CAP_DEFAULT,
                help="Rows a probe may hand to the assessment — a different limit from every "
                "timeout. A probe asks whether a shape is present, so it is small on purpose; a "
                "probe returning exactly this number is reported as truncated rather than as an "
                "exhaustive read.",
            ),
            # Rung-4 budgets. All `live`: the spawner re-reads per correlation.
            Field(
                "correlation.links.max_children_per_run",
                _MAIN,
                "integer",
                "live",
                minimum=0,
                maximum=MAX_CHILDREN_PER_RUN_CEILING,
                default=0,
                help="How many full child runs of a sibling procedure one run may launch by "
                "itself. 0 disables the rung, which is the default: a child run is a whole "
                "investigation against a primary source estate. A child also needs an escalating "
                "mode on the pair, a confirmed candidate, a pivot value and a pinnable target.",
            ),
            Field(
                "correlation.links.max_child_depth",
                _MAIN,
                "integer",
                "live",
                minimum=0,
                maximum=MAX_CHILD_DEPTH_CEILING,
                default=DEFAULT_MAX_CHILD_DEPTH,
                help="How many referrals deep a lineage may go — A refers B refers C is depth 2. "
                "Counted from the chain the incident carries, so it survives a restart: a chain "
                "lost by one would reset the cap to 0 on exactly the runs it exists to bound.",
            ),
            Field(
                "correlation.links.max_concurrent_children",
                _MAIN,
                "integer",
                "live",
                minimum=1,
                maximum=MAX_CONCURRENT_CHILDREN_CEILING,
                default=DEFAULT_MAX_CONCURRENT_CHILDREN,
                help="Child runs in flight at once. Additionally floored against the LLM layer's "
                "own max_concurrency minus one, leaving the parent a slot: every job in this "
                "process contends on one semaphore and one backend estate.",
            ),
            Field(
                "correlation.links.max_total_children",
                _MAIN,
                "integer",
                "live",
                minimum=0,
                maximum=MAX_TOTAL_CHILDREN_DEFAULT,
                default=MAX_TOTAL_CHILDREN_DEFAULT,
                help="The runaway backstop: child runs this process may launch in total, checked "
                "independently of the depth cap and the cycle guard because a fan-out that is "
                "shallow, acyclic and wide satisfies both and is still unbounded. Hitting it "
                "stops the spawning and records why, rather than failing a run.",
            ),
        ),
    ),
    (
        "retrieval",
        "Retrieval budgets",
        (
            Field(
                "log_sources.per_source_timeout_seconds",
                _MAIN,
                "integer",
                "live",
                minimum=1,
                maximum=86400,
                default=20,
                label="per source timeout seconds",
                help="Cap for an ORDINARY source, so one slow backend fails fast rather "
                "than burning the run. Sources the pack marks primary ignore this and "
                "use the budget below.",
            ),
            Field(
                "log_sources.primary_source_timeout_seconds",
                _MAIN,
                "integer",
                "live",
                # A timeout here decides the verdict rather than degrading it; hence the
                # generous floor.
                minimum=60,
                maximum=86400,
                default=7200,
                label="primary source timeout seconds",
                help="Budget for a source the knowledge pack marks "
                "retrieval_class: primary — one the investigation cannot be concluded "
                "without. Default 7200 (2h). Set it ABOVE the worst cost you have "
                "measured, not just above the typical one: a cap that fires here decides "
                "the verdict rather than degrading it, and waiting costs only wall clock "
                "(a warehouse we have lost contact with is detected separately, in ~25s). "
                "An accepted baseline that finished 27s under a 3000s cap timed out on a "
                "re-run when the shared warehouse was busy, returning 15 unevaluated "
                "conditions. Takes effect on the next run; no restart.",
            ),
            Field(
                "log_sources.extended_retrieval",
                _MAIN,
                "boolean",
                "live",
                default=False,
                help="Global opt-in to 'let slow sources run longer' (also settable "
                "per-incident). Raises every source's TIME budget to the extended one "
                "below, or 4x its normal cap when that is unset. It does NOT raise the "
                "row cap — that is 'max results' below.",
            ),
            Field(
                "log_sources.extended_retrieval_timeout_seconds",
                _MAIN,
                "integer",
                "live",
                minimum=1,
                maximum=86400,
                help="The cap used when extended retrieval is on. Never lowers a "
                "source's normal cap — opting in can only grant more time.",
            ),
            Field(
                "log_sources.default_lookup_days",
                _MAIN,
                "integer",
                "live",
                minimum=1,
                maximum=3650,
                label="default lookup days",
                help="How many days back to read when the incident text states NO date, "
                "time or relative time expression at all — counted back from the incident's "
                "own ingestion timestamp, so a re-run reads the same span. The knowledge "
                "pack's retrieval.default_lookup_days WINS over this, because retention and "
                "alerting lag are facts about the domain's stores; this is the fallback for a "
                "deployment whose pack declares none. Leave it unset and each query keeps "
                "whatever window the generator invented for it — measured on one undated "
                "incident, 29 queries carried eight different windows, several of them the "
                "single day the run started on. A window is the hard scope of every "
                "retrieval, so an invented one decides what the investigation can see.",
            ),
            Field(
                "log_sources.max_results",
                _MAIN,
                "integer",
                # restart: the cap is baked into retrievers at build time.
                "restart",
                minimum=1,
                maximum=100000,
                default=500,
                label="max results",
                help="Rows returned per source. A DIFFERENT limit from every timeout "
                "above: extended retrieval buys a slow source more seconds, never more "
                "rows. A source returning exactly this number is truncated (the live "
                "retrieval line says so) and its real total is higher. NOTE this is the "
                "fallback only — an endpoint under log_sources.backends that states its "
                "own max_results keeps it, so if the shipped config pins 500 per "
                "endpoint, raise it there (or delete those lines to inherit this).",
            ),
            Field(
                "log_sources.cache.enabled",
                _MAIN,
                "boolean",
                "restart",
                default=False,
                label="cache enabled",
                help="Serve a source from the last answer to the SAME question (source + "
                "question + window + entities + row cap + analyst guidance) instead of "
                "re-querying. A hit costs no timeout slot and no query-generation LLM "
                "call, and says on the source's own line how old it is. Only an answer is "
                "cached: a timeout, a backend error, a cancel and an empty result from a "
                "query with an unfilled placeholder are all excluded, so a transient gap "
                "cannot become a persistent verdict. Per process and not durable — a "
                "restart empties it and one replica does not see another's.",
            ),
            Field(
                "log_sources.cache.ttl_seconds",
                _MAIN,
                "integer",
                "restart",
                minimum=1,
                maximum=604800,
                default=3600,
                label="cache ttl seconds",
                help="How long a NON-EMPTY answer stays usable. Long enough to cover a "
                "re-run, a gate rejection and a child run of the same incident; short "
                "enough that a source repaired mid-shift is asked again.",
            ),
            Field(
                "log_sources.cache.empty_ttl_seconds",
                _MAIN,
                "integer",
                "restart",
                minimum=0,
                maximum=604800,
                default=0,
                label="cache empty ttl seconds",
                help="How long an EMPTY answer stays usable. 0 means never stored, which "
                "is the default: zero rows from a keyed lookup IS a finding, but it is "
                "also the answer a re-ask most often changes, and a cached empty is "
                "indistinguishable from a source that genuinely has nothing.",
            ),
            Field(
                "log_sources.cache.max_entries",
                _MAIN,
                "integer",
                "restart",
                minimum=1,
                maximum=10000,
                default=64,
                label="cache max entries",
                help="Answers kept before the least recently used is evicted.",
            ),
            Field(
                "log_sources.cache.max_rows",
                _MAIN,
                "integer",
                "restart",
                minimum=1,
                maximum=10000000,
                default=200000,
                label="cache max rows",
                help="Rows kept across all answers. A second bound rather than a "
                "redundant one: a single truncated primary source can carry more rows "
                "than sixty small answers together.",
            ),
        ),
    ),
    (
        "report",
        "Report generation",
        (
            Field(
                "report_generation.output_format",
                _MAIN,
                "choice",
                "live",
                choices=("pdf", "txt"),
                help="The machine-facing return value. Markdown + PDF are ALWAYS "
                "written alongside it regardless of this setting.",
            ),
            Field(
                "report_generation.max_anomalies_in_prompt",
                _MAIN,
                "integer",
                "live",
                minimum=1,
                maximum=200,
                help="Top-N by confidence sent to the LLM. Every anomaly still reaches "
                "the exports and the tables.",
            ),
            Field(
                "report_generation.llm_input_char_budget",
                _MAIN,
                "integer",
                "live",
                minimum=1000,
                maximum=200000,
            ),
            Field(
                "report_generation.report_min_tokens",
                _MAIN,
                "integer",
                "live",
                minimum=500,
                maximum=100000,
                help="Floor so the ~6 fixed sections always fit; too low truncates the "
                "report mid-document.",
            ),
            Field(
                "report_generation.report_max_tokens",
                _MAIN,
                "integer",
                "live",
                minimum=500,
                maximum=200000,
            ),
        ),
    ),
    (
        "feedback",
        "Analyst feedback",
        (
            Field(
                "feedback.batch_size",
                _MAIN,
                "integer",
                "live",
                minimum=1,
                maximum=1000,
                default=10,
                help="Reviews collected before a distillation runs.",
            ),
            Field(
                "feedback.apply_to_prompts",
                _MAIN,
                "boolean",
                "live",
                default=True,
                help="Off = collect reviews without steering the LLM.",
            ),
            Field(
                "feedback.auto_tune_threshold",
                _MAIN,
                "boolean",
                "restart",
                default=False,
                help="Off (default) = the threshold recommendation is advisory only.",
            ),
            Field(
                "feedback.min_reviews_for_tuning",
                _MAIN,
                "integer",
                "restart",
                minimum=1,
                maximum=1000,
                default=5,
                label="min reviews before tuning",
                help="Reviews required before any threshold adjustment is proposed. Only "
                "COUNTED review fields feed it — never free text.",
            ),
            Field(
                "feedback.threshold_step",
                _MAIN,
                "number",
                "restart",
                minimum=0.01,
                maximum=0.5,
                default=0.05,
                label="max step per adjustment",
                help="How far one adjustment may move the anomaly threshold. Small on "
                "purpose: a big jump changes what the system reports before anyone has "
                "seen the effect of the last change.",
            ),
            Field(
                "feedback.max_threshold_drift",
                _MAIN,
                "number",
                "restart",
                minimum=0.01,
                maximum=0.6,
                default=0.15,
                label="max drift from baseline",
                help="Total distance the tuned threshold may sit from the value YOU "
                "configured. Measured against the configured baseline, never against the "
                "last tuned value — otherwise successive steps compound and this bound "
                "means nothing.",
            ),
            Field(
                "feedback.threshold_min",
                _MAIN,
                "number",
                "restart",
                minimum=0.0,
                maximum=1.0,
                default=0.3,
                label="threshold floor",
                help="Absolute floor for the tuned anomaly threshold, whatever the "
                "reviews say.",
            ),
            Field(
                "feedback.threshold_max",
                _MAIN,
                "number",
                "restart",
                minimum=0.0,
                maximum=1.0,
                default=0.95,
                label="threshold ceiling",
                help="Absolute ceiling for the tuned anomaly threshold.",
            ),
        ),
    ),
    (
        "jobs",
        "Jobs & persistence",
        (
            Field(
                "jobs.persist",
                _MAIN,
                "boolean",
                "restart",
                default=True,
                help="Off = a restart discards every pending approval.",
            ),
            Field(
                "jobs.max_evidence_mb",
                _MAIN,
                "integer",
                "restart",
                minimum=1,
                maximum=4096,
                default=25,
            ),
            Field(
                "jobs.retention_days",
                _MAIN,
                "integer",
                "restart",
                minimum=1,
                maximum=3650,
                default=14,
            ),
            # Exposed (not hardcoded) because gates display this summary. `live`: re-read per call.
            Field(
                "jobs.summary_max_items",
                _MAIN,
                "integer",
                "live",
                minimum=1,
                maximum=1000,
                default=30,
                label="items per summary list",
                help="How many entries each per-stage summary list keeps — extracted "
                "entities, generated queries, findings, anomalies, verdict subjects. This is "
                "what the Investigate cards and the APPROVAL GATES display, so a list cut "
                "here is a list the operator signs off on without seeing: the true total is "
                "always shown beside it and the card says 'Showing N of M'. Raise it for "
                "incidents naming many entities or sources; the full data is in the export "
                "regardless. Bounded because these summaries ride the live event stream.",
            ),
            Field(
                "jobs.max_retrieval_passes",
                _MAIN,
                "integer",
                "live",
                minimum=1,
                maximum=10,
                default=3,
                label="max retrieval passes",
                help="Ceiling on rounds of query generation + log retrieval in one run. A "
                "knowledge pack may declare that a procedure needs a further round driven by "
                "what the previous round returned — the identities it must ask about do not "
                "exist until an earlier answer comes back. A pack that declares none runs one "
                "pass. This is a runaway guard, not a tuning knob: set 1 to refuse follow-up "
                "passes entirely. Reaching the cap names the dropped pass in the log and on "
                "the job's intervention trail, because a pass that never ran and a pass that "
                "found nothing leave the same (absent) evidence.",
            ),
            # Both `live`: the queue re-reads these on every admission.
            Field(
                "jobs.max_concurrent_jobs",
                _MAIN,
                "integer",
                "live",
                minimum=1,
                maximum=8,
                default=2,
                label="runs at once",
                help="How many submitted runs execute stages simultaneously. Past this they "
                "WAIT in a FIFO backlog with a reported position rather than starting "
                "alongside. Raising it does not raise throughput past the LLM endpoint: one "
                "run fans out ~19 retrievers making two or three calls each, and they already "
                "share the endpoint's concurrency semaphore and rate limiter — past a few runs "
                "that semaphore becomes the real queue, with no depth, position or cancel "
                "anyone can see. Governs SUBMITTED runs only: a resumed run, a stage retry and "
                "a linked child run all start outside it, and a run parked on an approval gate "
                "hands its slot back while it waits.",
            ),
            Field(
                "jobs.max_queued_jobs",
                _MAIN,
                "integer",
                "live",
                minimum=1,
                maximum=4096,
                default=256,
                label="max waiting runs",
                help="How many submissions may wait for a slot. Past this the submit "
                "endpoints answer 429 naming the depth and this limit — a refusal, never a "
                "silent accept, because a job id for a run that may never start reads exactly "
                "like a run that is merely slow. Lower it to shed load early; raise it for an "
                "overnight backlog.",
            ),
        ),
    ),
    (
        "storage",
        "Durable state (local, volume, or database)",
        (
            # All `restart`: the backend object is built once and shared.
            Field(
                "storage.backend",
                _MAIN,
                "choice",
                "restart",
                choices=("local", "databricks", "sql"),
                default="local",
                label="state backend",
                help="WHERE job documents, evidence, exports and the feedback log live. "
                "local = a filesystem: disk under AFIR_DATA_DIR by default, or a mounted "
                "/ external volume by setting 'state directory' below. databricks = a "
                "Unity Catalog Volume over the Files API, MANAGED OR EXTERNAL (both live "
                "in /Volumes and need no different setting) — available on a laptop or VM "
                "too, not only inside an App. sql = a transactional database, the only "
                "option safe across several replicas.",
            ),
            Field(
                "storage.root",
                _MAIN,
                "string",
                "restart",
                label="state directory",
                help="Only for the local backend. Blank = AFIR_DATA_DIR, which is what "
                "every existing deployment gets. Set it to a MOUNTED or EXTERNAL volume "
                "to keep state off the machine's own disk — it is still a filesystem, so "
                "the atomic-replace write path is unchanged. Do NOT point this at "
                "/Volumes inside a Databricks App: it is not mounted there, so the write "
                "lands in /tmp and reports success while losing every pending approval "
                "on restart. Use the databricks backend for a UC Volume.",
            ),
            Field(
                "storage.mirror_config_and_pack",
                _MAIN,
                "choice",
                "restart",
                choices=("auto", "always", "never"),
                default="auto",
                label="mirror config + knowledge pack",
                help="The YAML configs and the pack are MIRRORED rather than moved: a "
                "write lands on the local working copy through the byte-level path that "
                "keeps its comments, then the bytes are pushed to the store. auto = "
                "mirror when the backend is remote. always = mirror even over local "
                "disk. never = treat config and pack as part of the immutable image.",
            ),
            Field(
                "storage.databricks.catalog",
                _MAIN,
                "string",
                "restart",
                label="volume catalog",
                help="Required for the databricks backend: the Volume is "
                "/Volumes/<catalog>/<schema>/<volume>. Empty means the backend cannot "
                "write and says so at boot rather than falling back silently.",
            ),
            Field(
                "storage.databricks.schema",
                _MAIN,
                "string",
                "restart",
                default="afir",
                label="volume schema",
            ),
            Field(
                "storage.databricks.volume",
                _MAIN,
                "string",
                "restart",
                default="state",
                label="volume name",
            ),
            Field(
                "storage.databricks.host",
                _MAIN,
                "string",
                "restart",
                label="workspace URL",
                help="Blank uses the Databricks SDK's own resolution, which is what an "
                "App wants (the platform injects OAuth credentials for the workspace "
                "holding the Volume). Set it for a local run against a PAT.",
            ),
            Field(
                "storage.databricks.token_env",
                _MAIN,
                "string",
                "restart",
                default="DATABRICKS_TOKEN",
                label="token env var",
                help="Name of the env var holding the PAT — a NAME, never a token. Used "
                "when the SDK resolves no ambient credential. A PAT is "
                "workspace-scoped: one issued for another workspace answers 403 with a "
                "token that looks perfectly valid.",
            ),
            Field(
                "storage.databricks.verify_ssl",
                _MAIN,
                "boolean",
                "restart",
                default=True,
                label="verify SSL",
                help="On by default here, unlike the log-source backends: those reach "
                "workspaces behind a self-signed corporate chain, and an insecure "
                "posture must not be inherited by a store that holds pending approvals.",
            ),
            # Only for `backend: sql`.
            Field(
                "storage.sql.dialect",
                _MAIN,
                "choice",
                "restart",
                choices=("sqlite", "postgresql"),
                default="sqlite",
                label="database dialect",
                help="sqlite = a single ACID file, which is a real answer for a mounted "
                "or external volume and needs no server. postgresql = a server, and the "
                "one configuration that is safe with more than one replica. A warehouse "
                "(Databricks SQL, Snowflake) is not offered: see src/storage/sql.py for "
                "the measured size ceiling that rules it out.",
            ),
            # Not `storage.sql.dsn`: a libpq URL embeds the password, which would
            # round-trip the redacted placeholder on save. A hand-written `dsn` still works.
            Field(
                "storage.sql.dsn_env",
                _MAIN,
                "string",
                "restart",
                default="AFIR_SQL_DSN",
                label="DSN env var",
                help="Name of the env var holding the connection string — a NAME, never "
                "the DSN itself. sqlite: a FILE PATH (its parent is created). postgresql: "
                "a libpq URL, which may embed a password, which is exactly why it is read "
                "from the environment and not stored here. Required for the sql backend: "
                "with neither this nor a hand-written `dsn` resolving to a value, the "
                "store refuses every write and says so at boot rather than accepting them "
                "and holding nothing.",
            ),
            Field(
                "storage.sql.table",
                _MAIN,
                "string",
                "restart",
                default="afir_blobs",
                label="table name",
                help="Created if absent. Validated as a SQL identifier rather than "
                "quoted, because it is the one value here that cannot be a bound "
                "parameter; optionally schema-qualified (myschema.afir_blobs).",
            ),
        ),
    ),
    (
        "webhooks",
        "Webhooks",
        (
            Field("webhooks.enabled", _MAIN, "boolean", "restart", default=False),
            Field(
                "webhooks.timeout_seconds",
                _MAIN,
                "integer",
                "restart",
                minimum=1,
                maximum=300,
                default=10,
            ),
            Field(
                "webhooks.max_attempts",
                _MAIN,
                "integer",
                "restart",
                minimum=1,
                maximum=10,
                default=3,
            ),
        ),
    ),
    (
        "logging",
        "Logging",
        (
            Field(
                "logging.format",
                _MAIN,
                "choice",
                "live",
                choices=("console", "json"),
                default="console",
                help="json makes stage/gate/health lines queryable where the console is "
                "the only telemetry that leaves the container. AFIR_LOG_FORMAT "
                "overrides this.",
            ),
            Field(
                "logging.level",
                _MAIN,
                "choice",
                "live",
                choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                default="INFO",
                help="AFIR_LOG_LEVEL overrides this.",
            ),
        ),
    ),
    (
        "knowledge",
        "Knowledge & RAG",
        (
            Field(
                "knowledge.pack_dir",
                _MAIN,
                "string",
                "restart",
                label="knowledge pack",
                help="Directory under knowledge/ — the domain pack supplying entities, "
                "sources and playbooks.",
            ),
            Field("rag.use_rag", _MAIN, "boolean", "restart"),
            # All `restart`: vectors are only comparable to an index built by the same model.
            Field(
                "rag.embedding_provider",
                _MAIN,
                "choice",
                "restart",
                label="embedding provider",
                choices=("sentence_transformers", "databricks"),
                default="sentence_transformers",
                help="sentence_transformers runs a local model (no network, ~420MB, needs "
                "torch); databricks calls a Model Serving endpoint (nothing to download, "
                "needs the workspace reachable). A provider that cannot produce vectors "
                "degrades to the deterministic keyword fallback, it does not fail the boot.",
            ),
            Field(
                "rag.embedding_model",
                _MAIN,
                "string",
                "restart",
                label="embedding model",
                help="For 'databricks', a serving-endpoint name (e.g. "
                "databricks-qwen3-embedding-0-6b). For 'sentence_transformers', a local "
                "directory or an HF model id. Blank falls back to the legacy "
                "rag.sentence_transformer_model.",
            ),
            Field(
                "rag.embedding_dim",
                _MAIN,
                "integer",
                "restart",
                minimum=1,
                maximum=8192,
                default=768,
                label="embedding dimension",
                help="Sizes the FAISS index before the first call. It is a CLAIM: the "
                "provider reports its real width on the first encode, warns if they "
                "disagree, and the index is built at the measured width regardless.",
            ),
            Field(
                "rag.embedding_host",
                _MAIN,
                "string",
                "restart",
                label="embedding host",
                help="Workspace URL for the 'databricks' provider. Blank uses the "
                "Databricks auth already resolved for the LLM — correct whenever the "
                "endpoint lives in the same workspace.",
            ),
            Field(
                "rag.embedding_token_env",
                _MAIN,
                "string",
                "restart",
                label="embedding token env var",
                help="Name of the env var holding a PAT for the embedding endpoint — a "
                "NAME, never the token. Blank uses the unified Databricks auth, which is "
                "what an App's OAuth identity needs.",
            ),
            Field(
                "rag.embedding_verify_ssl",
                _MAIN,
                "boolean",
                "restart",
                default=True,
                label="embedding verify SSL",
                help="Leave on. Off only for a self-signed corporate chain, matching the "
                "log-source backends' dev-only accommodation.",
            ),
            # A partial batch would index wrong documents under wrong ids, so the provider
            # refuses the whole batch. Both `restart`.
            Field(
                "rag.embedding_batch_size",
                _MAIN,
                "integer",
                "restart",
                minimum=1,
                maximum=1000,
                default=100,
                label="embedding batch size",
                help="Documents sent per embedding request. Both measured endpoints return "
                "a 400 at 151 inputs, so this is a platform limit rather than a tuning "
                "knob — lower it for an endpoint that refuses sooner. Local "
                "sentence-transformers providers batch in process and ignore it.",
            ),
            Field(
                "rag.embedding_max_chars",
                _MAIN,
                "integer",
                "restart",
                minimum=1000,
                maximum=1000000,
                default=96000,
                label="embedding max chars per doc",
                help="Per-document character bound, deliberately in characters rather than "
                "tokens: the providers disagree about tokenisation and this only has to be "
                "SAFE, not tight. ~4 chars/token against a 32k window leaves headroom, and "
                "it clears the 306k-char outlier that produced a 400.",
            ),
            Field(
                "rag.max_retrieved_documents",
                _MAIN,
                "integer",
                "restart",
                minimum=1,
                maximum=100,
            ),
            Field(
                "rag.similarity_threshold",
                _MAIN,
                "number",
                "restart",
                minimum=0.0,
                maximum=1.0,
            ),
        ),
    ),
    (
        "server",
        "Server & limits",
        (
            Field(
                "incident_input.blocking_endpoints",
                _MAIN,
                "choice",
                "live",
                label="blocking submit endpoints",
                choices=("auto", "on", "off"),
                default="auto",
                help="POST /api/v1/incidents, /incidents/freetext and /ir hold the "
                "connection open for the whole investigation. auto enables them "
                "everywhere except a Databricks App, whose ingress closes a request at "
                "~120s while a run takes 38min at the median — there they answer 501 "
                "naming POST /api/v1/jobs instead of hanging. The jobs API is unaffected.",
            ),
            Field("incident_input.host", _MAIN, "string", "restart"),
            Field(
                "incident_input.port",
                _MAIN,
                "integer",
                "restart",
                minimum=1,
                maximum=65535,
                help="Ignored inside a Databricks App, which injects "
                "DATABRICKS_APP_PORT.",
            ),
            Field(
                "incident_input.rate_limit.requests",
                _MAIN,
                "integer",
                "restart",
                minimum=1,
                maximum=1000000,
            ),
            Field(
                "incident_input.rate_limit.per_seconds",
                _MAIN,
                "integer",
                "restart",
                minimum=1,
                maximum=86400,
            ),
        ),
    ),
)

#: Flat path -> Field, for validation.
FIELDS: Dict[str, Field] = {
    field.path: field for _, _, fields in SECTIONS for field in fields
}


# Reading


def _walk(node, prefix=""):
    """Yield ``(dotted_path, value)`` for every scalar leaf in a nested structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(node, list):
        # Lists are surfaced whole; nothing in the editable field set is a list element.
        yield prefix, node
    else:
        yield prefix, node


def redact_tree(node, prefix=""):
    """Deep-copy ``node`` with literal secrets replaced by :data:`REDACTED`; ``${VAR}`` references survive."""
    if isinstance(node, dict):
        return {
            key: redact_tree(value, f"{prefix}.{key}" if prefix else str(key))
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [redact_tree(item, prefix) for item in node]
    if isinstance(node, str) and node and is_secret_key(prefix):
        return node if is_env_reference(node) else REDACTED
    return node


def get_path(tree, path: str, default=None):
    """Read a dotted path out of a nested dict."""
    node = tree
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def read_file(name: str) -> dict:
    """Parse one config file. Returns ``{}`` for a missing/empty file."""
    path = config_dir() / name
    if not path.exists():
        return {}
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def read_raw(name: str) -> str:
    """The file's exact text, for the raw editor. ``""`` when absent."""
    path = config_dir() / name
    if not path.exists():
        return ""
    with open(path, "r") as handle:
        return handle.read()


def redact_raw(text: str) -> str:
    """Mask literal secrets in raw YAML text line by line, preserving comments."""
    out = []
    for line in text.splitlines():
        match = re.match(r"^(\s*)([A-Za-z_][\w-]*)(\s*:\s*)(.*)$", line)
        if not match:
            out.append(line)
            continue
        indent, key, sep, rest = match.groups()
        value, comment = _split_comment(rest)
        stripped = value.strip()
        if stripped and is_secret_key(key) and not is_env_reference(_unquote(stripped)):
            spacer = " " if comment else ""
            out.append(f'{indent}{key}{sep}"{REDACTED}"{spacer}{comment}')
        else:
            out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def describe() -> dict:
    """Everything the Configuration UI needs to render itself, in one payload."""
    files = {}
    for name in CONFIG_FILES:
        tree = read_file(name)
        files[name] = {
            "exists": (config_dir() / name).exists(),
            "values": redact_tree(tree),
            "raw": redact_raw(read_raw(name)),
        }
    trees = {name: read_file(name) for name in CONFIG_FILES}
    sections = []
    for key, title, fields in SECTIONS:
        entries = []
        for field in fields:
            item = field.to_dict()
            sentinel = object()
            raw_value = get_path(trees.get(field.file) or {}, field.path, sentinel)
            item["set"] = raw_value is not sentinel
            if raw_value is sentinel:
                # Absent: report the code's fallback, flagged unset.
                raw_value = field.default
            if isinstance(raw_value, str) and is_secret_key(field.path):
                raw_value = raw_value if is_env_reference(raw_value) else REDACTED
            item["value"] = raw_value
            entries.append(item)
        sections.append({"key": key, "title": title, "fields": entries})
    return {
        "config_dir": str(config_dir()),
        "files": files,
        "sections": sections,
        "redacted_placeholder": REDACTED,
    }


# Validation


def _coerce(field: Field, value):
    """Coerce a JSON value to the field's declared type, or raise ValueError."""
    if field.kind == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in (
            "true",
            "false",
            "yes",
            "no",
            "1",
            "0",
            "on",
            "off",
        ):
            return value.strip().lower() in ("true", "yes", "1", "on")
        raise ValueError(f"{field.path}: expected true/false, got {value!r}")
    if field.kind == "integer":
        try:
            # bool is an int subclass; reject it explicitly so `true` cannot become 1.
            if isinstance(value, bool):
                raise ValueError
            coerced = int(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(f"{field.path}: expected a whole number, got {value!r}")
        return _check_range(field, coerced)
    if field.kind == "number":
        try:
            if isinstance(value, bool):
                raise ValueError
            coerced = float(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(f"{field.path}: expected a number, got {value!r}")
        return _check_range(field, coerced)
    if field.kind == "choice":
        text = str(value).strip()
        if text not in (field.choices or ()):
            raise ValueError(
                f"{field.path}: expected one of {', '.join(field.choices or ())}, "
                f"got {value!r}"
            )
        return text
    # string / text
    if value is None:
        return ""
    if not isinstance(value, (str, int, float)):
        raise ValueError(f"{field.path}: expected text, got {type(value).__name__}")
    return str(value)


def _check_range(field: Field, value):
    if field.minimum is not None and value < field.minimum:
        raise ValueError(f"{field.path}: must be >= {field.minimum} (got {value})")
    if field.maximum is not None and value > field.maximum:
        raise ValueError(f"{field.path}: must be <= {field.maximum} (got {value})")
    return value


def validate_updates(updates: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Coerce and range-check ``{path: value}``. Returns ``(accepted, errors)``.

    A redacted secret is silently dropped, so round-tripping the rendered form is safe.
    """
    accepted: Dict[str, Any] = {}
    errors: List[str] = []
    for path, value in (updates or {}).items():
        field = FIELDS.get(path)
        if field is None:
            errors.append(f"{path}: not an editable field")
            continue
        if isinstance(value, str) and value.strip() == REDACTED:
            continue  # unchanged secret; keep whatever is on disk
        try:
            accepted[path] = _coerce(field, value)
        except ValueError as exc:
            errors.append(str(exc))
    return accepted, errors


# Writing: surgical, comment-preserving scalar patches


def _split_comment(rest: str) -> Tuple[str, str]:
    """Split a YAML value from its trailing ``#`` comment, respecting quoted ``#`` chars."""
    quote = None
    for index, char in enumerate(rest):
        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#":
            return rest[:index].rstrip(), rest[index:]
    return rest.rstrip(), ""


def _unquote(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def _emit(value) -> str:
    """Render a Python scalar as YAML, quoting strings that need it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    text = str(value)
    if text == "":
        return '""'
    # Quote anything YAML could read as a non-string or that carries significant punctuation.
    if re.fullmatch(r"[A-Za-z0-9_./@\-]+", text) and not re.fullmatch(
        r"(true|false|null|yes|no|on|off|~|-?\d+(\.\d+)?)", text, re.I
    ):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _find_scalar_line(lines: List[str], path: str) -> Optional[int]:
    """Line index for ``path``, or ``None`` when not unambiguously present.

    Resolves direct children only; a same-named key nested deeper is not a match.
    Ambiguity returns ``None``: a write on the wrong key is worse than a refusal.
    """
    parts = path.split(".")
    depth = 0  # index into `parts` we are looking for
    parent_indent = -1  # indentation of the matched parent, -1 at top level
    start = 0

    while depth < len(parts):
        want = parts[depth]
        found = None
        # Learned from the first key, not assumed to be 2.
        child_indent = None
        for index in range(start, len(lines)):
            line = lines[index]
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if depth > 0 and indent <= parent_indent:
                break  # left the parent's block without finding the key
            match = re.match(r"^(\s*)([A-Za-z_][\w-]*)\s*:(.*)$", line)
            if not match:
                continue
            if depth == 0 and indent != 0:
                continue  # only top-level keys can match the first segment
            if depth > 0 and indent <= parent_indent:
                break
            if match.group(2) == want:
                # Direct child only; deeper same-named keys are skipped.
                if depth > 0 and child_indent is not None and indent != child_indent:
                    continue
                found = index
                break
            if depth > 0 and child_indent is None:
                # First key seen inside the parent block: this is its child indent.
                child_indent = indent
        if found is None:
            return None
        if depth == len(parts) - 1:
            return found
        parent_indent = len(lines[found]) - len(lines[found].lstrip())
        start = found + 1
        depth += 1
    return None


def _block_extent(lines: List[str], header: int) -> Tuple[int, Optional[int]]:
    """``(end_exclusive, child_indent)`` for the block at ``header``.

    ``end_exclusive`` is past the last non-blank line so an insert lands inside the
    section rather than after its trailing gap.
    """
    header_indent = len(lines[header]) - len(lines[header].lstrip())
    index = header + 1
    child_indent = None
    end = header + 1
    while index < len(lines):
        line = lines[index]
        if line.strip():
            indent = len(line) - len(line.lstrip())
            if indent <= header_indent:
                break
            if child_indent is None and not line.lstrip().startswith("#"):
                child_indent = indent
            end = index + 1
        index += 1
    return end, child_indent


def _holds_a_block(line: str) -> bool:
    """True when a ``key:`` line opens a nested block instead of holding a value."""
    if ":" not in line:
        return False
    return not _split_comment(line.split(":", 1)[1])[0].strip()


def plan_insert(lines: List[str], path: str) -> Optional[Tuple[int, List[str]]]:
    """Plan insertion of an absent key: ``(line_index, lines_to_insert)``.

    Walks as far down ``path`` as the file has, then synthesizes missing parents.
    Returns ``None`` when an existing path segment holds a scalar rather than a block
    (restructuring belongs in the raw editor, not a field write).
    """
    parts = path.split(".")
    # Deepest existing ancestor.
    header = None
    matched = 0
    for depth in range(len(parts) - 1, 0, -1):
        candidate = _find_scalar_line(lines, ".".join(parts[:depth]))
        if candidate is not None:
            header, matched = candidate, depth
            break

    if header is None:
        # Nothing on the path exists; append a fresh top-level section (or scalar).
        # No duplicate key can result since nothing exists yet.
        at = len(lines)
        while at > 0 and not lines[at - 1].strip():
            at -= 1  # before any trailing blank lines
        block = []
        for depth, part in enumerate(parts[:-1]):
            block.append(f"{'  ' * depth}{part}:")
        block.append(f"{'  ' * (len(parts) - 1)}{parts[-1]}: ")
        return at, ([""] + block if at > 0 else block)

    if not _holds_a_block(lines[header]):
        return None
    end, child_indent = _block_extent(lines, header)
    header_indent = len(lines[header]) - len(lines[header].lstrip())
    indent = child_indent if child_indent is not None else header_indent + 2
    block = []
    for depth, part in enumerate(parts[matched:-1]):
        block.append(f"{' ' * (indent + depth * 2)}{part}:")
    block.append(f"{' ' * (indent + (len(parts) - matched - 1) * 2)}{parts[-1]}: ")
    return end, block


def apply_updates(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Write validated ``{path: value}`` into their files in place; return a change report.

    Patches each line (or inserts into the right block); writes atomically with a ``.bak``.
    Never appends at the end of a file: a duplicate key's winner is parser-dependent.
    ``durable: False`` when a push to the mirror failed — the edit is live but not durable.
    """
    by_file: Dict[str, Dict[str, Any]] = {}
    for path, value in updates.items():
        field = FIELDS[path]
        by_file.setdefault(field.file, {})[path] = value

    changed: List[dict] = []
    skipped: List[dict] = []
    durable = True
    for name, items in by_file.items():
        target = config_dir() / name
        if not target.exists():
            for path in items:
                skipped.append({"path": path, "reason": f"{name} does not exist"})
            continue
        with open(target, "r") as handle:
            text = handle.read()
        lines = text.splitlines()
        dirty = False
        for path, value in items.items():
            index = _find_scalar_line(lines, path)
            if index is None:
                # Absent from the file: insert it, creating any missing parent sections.
                plan = plan_insert(lines, path)
                if plan is None:
                    skipped.append(
                        {
                            "path": path,
                            "reason": "a section on this path holds a value rather than "
                            "a block — restructure it in the raw editor",
                        }
                    )
                    continue
                at, block = plan
                block[-1] = block[-1] + _emit(value)
                lines[at:at] = block
                changed.append(
                    {
                        "path": path,
                        "from": None,
                        "to": str(value),
                        "applies": FIELDS[path].applies,
                        "inserted": True,
                    }
                )
                dirty = True
                continue
            match = re.match(r"^(\s*)([A-Za-z_][\w-]*)(\s*:\s*)(.*)$", lines[index])
            if match is None:
                skipped.append({"path": path, "reason": "unparsable line"})
                continue
            indent, key, sep, rest = match.groups()
            old_value, comment = _split_comment(rest)
            if not old_value.strip():
                # The key opens a nested block (``foo:`` with children). Replacing that
                # line with a scalar would orphan every child under it.
                skipped.append(
                    {"path": path, "reason": "key holds a block, not a value"}
                )
                continue
            new_rendered = _emit(value)
            if _unquote(old_value) == _unquote(new_rendered):
                continue  # no-op; don't churn the file or the .bak
            spacer = " " if comment else ""
            lines[index] = f"{indent}{key}{sep}{new_rendered}{spacer}{comment}"
            changed.append(
                {
                    "path": path,
                    "from": _unquote(old_value),
                    "to": str(value),
                    "applies": FIELDS[path].applies,
                }
            )
            dirty = True
        if dirty:
            new_text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
            _verify_parses(new_text, name)
            durable = _atomic_write(target, new_text) and durable
            logger.info(
                "Config updated: %s (%d key(s))",
                name,
                len([c for c in changed if FIELDS[c["path"]].file == name]),
            )
    return {"changed": changed, "skipped": skipped, "durable": durable}


def _verify_parses(text: str, name: str) -> None:
    """Parse the candidate text before it replaces a working file, so a patcher bug fails the request."""
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"the edit would make {name} unparsable: {exc}") from exc


def _atomic_write(target, text: str) -> bool:
    """Write via a temp file + rename; saves the overwritten copy as ``.bak``.

    Returns whether the bytes survive a restart. Mirror failures are reported, not raised.
    The push re-reads from disk: what must be durable is what ``read_file`` will parse.
    """
    if target.exists():
        shutil.copy2(target, str(target) + ".bak")
    tmp = str(target) + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(text)
    os.replace(tmp, target)
    if _MIRROR is None:
        return True
    return bool(_MIRROR.push_path(target))


def replace_file(name: str, text: str) -> dict:
    """Replace a whole config file (raw editor / import), validating first.

    Rejects text still carrying :data:`REDACTED`: writing it would silently overwrite a
    credential with the placeholder string.
    """
    if name not in CONFIG_FILES:
        raise ValueError(f"unknown config file '{name}'")
    if REDACTED in text:
        raise ValueError(
            f"the text still contains {REDACTED!r} — replace those placeholders with "
            "real values or an ${ENV_VAR} reference before saving"
        )
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
    if parsed is not None and not isinstance(parsed, dict):
        raise ValueError("a config file must be a YAML mapping at the top level")
    target = config_dir() / name
    durable = _atomic_write(target, text if text.endswith("\n") else text + "\n")
    logger.info("Config file replaced wholesale: %s (%d bytes)", name, len(text))
    return {
        "file": name,
        "bytes": len(text),
        "keys": sorted((parsed or {}).keys()),
        "durable": durable,
    }
