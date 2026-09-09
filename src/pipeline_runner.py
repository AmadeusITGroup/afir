"""Controllable pipeline runner: stages run one at a time under a JobManager (single-process asyncio only)."""

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from export_results import ResultExporter
from human_guidance import render_guidance
from identity import owner_of, owner_scoped
from job_queue import JobQueue, QueueFull
from job_store import EVIDENCE_KEYS
from link_children import child_budget, child_incident, plan_child_spawns
from link_escalation import ESCALATING_MODES
from links import compose_referral
from models.pydantic_models import (AnomalyItem, CorrelationResult,
                                    RetrievalQuery, UnderstandingResult)
from notifications import emitter
# Package-qualified deliberately: a flat import would be a second module object with its own
# installed store, so the runner would resolve nobody's personal credentials while main() and
# the HTTP layer resolved everybody's.
from src.user_secrets import (current_segment, reset_current_segment,
                              set_current_segment)
from stage_health import (score_stage, stage_gate_enabled,
                          stage_gate_on_timeout, stage_gate_timeout)
from utils.paths import exports_dir

logger = logging.getLogger(__name__)

_EVENT_HISTORY_CAP = 500  # replay buffer cap per job
_JOB_TTL_SECONDS = 3600   # terminal-job in-memory TTL, default for jobs.completed_ttl_seconds
#: How many pruned runs keep a compact row in memory. Bounds the list, not the store:
#: the documents outlive this at `jobs.retention_days` and rehydrate on demand.
_HISTORY_MAX_ITEMS = 2000
# Source's backend query generated; source is still `running`. Not a lifecycle status.
_QUERY_READY = "query_ready"

# The two stages a follow-up pass re-runs, in order.
_PASS_STAGES = ("query_generation", "log_retrieval")
#: Completion of this stage decides whether another pass is due.
_LAST_PASS_STAGE = _PASS_STAGES[-1]
#: Hard bound; prevents a mis-authored pack from looping on a self-qualifying harvest.
DEFAULT_MAX_RETRIEVAL_PASSES = 3


def _as_int(value, default: int) -> int:
    """``value`` as int, or ``default``; a typo must not silently become 0."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def max_retrieval_passes(config) -> int:
    """Configured ceiling on retrieval passes (jobs.max_retrieval_passes), floored at 1."""
    jobs = (config or {}).get("jobs") or {}
    try:
        value = int(jobs.get("max_retrieval_passes", DEFAULT_MAX_RETRIEVAL_PASSES))
    except (TypeError, ValueError):
        value = DEFAULT_MAX_RETRIEVAL_PASSES
    return max(1, value)


def pass_key(stage_name: str, pass_number: int = 1) -> str:
    """Record key for one stage in one pass.

    Pass 1 returns the bare stage name (load-bearing: config, UI labels, and persisted
    docs all address stages by name, so single-pass runs are byte-identical).
    """
    if pass_number <= 1 or stage_name not in _PASS_STAGES:
        return stage_name
    return f"{stage_name}#{pass_number}"


def split_pass_key(key: str) -> Tuple[str, int]:
    """Inverse of :func:`pass_key`: ``("log_retrieval#2")`` -> ``("log_retrieval", 2)``."""
    name, _, suffix = str(key).partition("#")
    if not suffix:
        return name, 1
    try:
        return name, max(1, int(suffix))
    except ValueError:
        return str(key), 1


def _pass_detail(key: str, detail: str) -> str:
    """detail with the pass number appended for pass > 1.

    Appended rather than replacing: Job.interventions is filtered by stage name.
    """
    number = split_pass_key(key)[1]
    return detail if number <= 1 else f"{detail} (pass {number})"


def _elapsed_ms(started_at: float) -> int:
    """Whole milliseconds elapsed since ``started_at`` (time.time() epoch)."""
    return int((time.time() - started_at) * 1000)


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    # `skipped`: human decided to proceed without this stage; `cancelled`: work stopped mid-flight.
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobStatus(str, Enum):
    PENDING = "pending"
    # Admitted to the backlog but not started (concurrency full). Distinct from
    # `pending` (created, not yet handed to a runner) and `paused` (started, then stopped).
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    STAGE_FAILED = "stage_failed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    # A gate is open. Not terminal. Distinct from `paused`: pause is operator→run; gate is run→operator.
    AWAITING_APPROVAL = "awaiting_approval"


class JobRunMode(str, Enum):
    """How much human approval a run requires.

    auto/semi_auto/supervised gate never/on-low-health/always after each stage;
    step pauses before each stage (permission to start, not review of output).
    """

    AUTO = "auto"  # run every stage back-to-back; no gates
    SEMI_AUTO = "semi_auto"  # gate only when a stage's health is below threshold
    SUPERVISED = "supervised"  # gate after every gateable stage
    STEP = "step"  # pause before each stage; advance only on a `step` action


def resolve_run_mode(mode) -> JobRunMode:
    """A client's run mode as a JobRunMode, falling back to AUTO with a warning (not a raise)."""
    if isinstance(mode, JobRunMode):
        return mode
    try:
        return JobRunMode(str(mode).lower())
    except ValueError:
        logger.warning("Unknown run mode %r; running in AUTO (no approval gates).", mode)
        return JobRunMode.AUTO


_GATING_MODES = {JobRunMode.SEMI_AUTO, JobRunMode.SUPERVISED}

# `awaiting_approval` is NOT terminal: a job holding a gate is mid-run.
_TERMINAL_STATUSES = {JobStatus.COMPLETED, JobStatus.CANCELLED}

# Rehydrated jobs in these statuses come back as `paused`. `pending` and `queued` are
# absent: nothing ran, so they keep their status and resume_queued() re-enqueues them.
_RESUMABLE_AS_PAUSED = {JobStatus.RUNNING, JobStatus.AWAITING_APPROVAL}

# Applied to a queued job, these tell the queue to stop tracking it.
_STARTING_ACTIONS = {"resume", "step", "retry_stage", "retry_all", "skip_stage"}

GATE_ACTIONS = ("approve", "reject", "override")

# Degrades the durability guarantee, not the gate, if this timeout fires before flush.
GATE_PERSIST_FLUSH_SECONDS = 8.0


@dataclass
class StageDescriptor:
    """One pipeline stage; is_synchronous runs in a worker thread, is_best_effort logs failures."""

    name: str
    run: Callable
    output_key: str = ""
    is_best_effort: bool = False
    is_synchronous: bool = False


@dataclass
class JobContext:
    """Per-job state shared with every stage callable."""

    job_id: str
    incident: dict
    modules: dict
    config: dict
    outputs: Dict[str, Any] = field(default_factory=dict)
    # Facts a stage records about how it ran (e.g. timeout vs empty). Stage health reads these.
    stage_facts: Dict[str, Any] = field(default_factory=dict)
    # Analyst direction per stage, accumulated on reject; injected when the stage re-runs.
    stage_guidance: Dict[str, List[str]] = field(default_factory=dict)
    # Per-link escalation modes for this run. On context (not outputs) so gate rejection
    # re-running correlation doesn't discard the setting.
    link_modes: Dict[str, str] = field(default_factory=dict)
    current_pass: int = 1  # which retrieval pass the repeatable stages are on (1-based)
    # Per-pass accumulation {pass: {"queries": [...], "logs": {...}, "notes": [...]}}. outputs
    # holds the merged accumulation; this holds per-pass splits for "found nothing" vs "not asked".
    pass_outputs: Dict[int, Dict[str, Any]] = field(default_factory=dict)


class Job:
    """A single investigation run driven through the stage list by a JobManager."""

    def __init__(self, job_id, incident, stage_names, run_mode, ctx):
        self.job_id = job_id
        self.incident = incident
        self.run_mode = run_mode
        self.context = ctx
        self.status = JobStatus.PENDING
        # Ordered stage names; stage_statuses is keyed by it.
        self.stage_names: List[str] = list(stage_names)
        # Identical to stage_names for a single-pass run. A follow-up pass inserts
        # its own stage#N keys after pass 1's retrieval (see register_pass).
        self.stage_keys: List[str] = list(stage_names)
        self.stage_statuses: Dict[str, StageStatus] = {
            name: StageStatus.PENDING for name in self.stage_keys
        }
        self.current_stage: Optional[str] = None
        self.error: Optional[str] = None
        self.stage_durations: Dict[str, int] = {}
        self.stage_summaries: Dict[str, Any] = {}
        # Epoch seconds when the stage's current attempt began; lets a mid-attach client
        # continue the elapsed-time counter.
        self.stage_started_at: Dict[str, float] = {}
        self.stage_health: Dict[str, Any] = {}
        now = datetime.now(timezone.utc).isoformat()
        self.created_at = now
        self.updated_at = now
        self.created_ts = time.time()  # numeric, for TTL pruning

        # --- control primitives ---
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # set = running; cleared = pause at next boundary
        self._step_event = asyncio.Event()  # advances one stage in step mode (starts cleared)
        self._cancel_stage_flag = False
        self._cancel_all_flag = False
        # Fires so the stage loop unblocks immediately on cancel, even if the library call
        # refuses to unwind promptly.
        self._cancel_event = asyncio.Event()
        self._current_task: Optional[asyncio.Future] = None
        self._failed_index: Optional[int] = None
        # Where the loop stopped without finishing (failure or cancel_stage); resume/step
        # restart from here; retry_stage defaults to it.
        self._stopped_index: Optional[int] = None
        # True while a _run_stages loop is driving this job. A restored or stage-cancelled
        # job has no loop, so resume must start one rather than just set an event.
        self._loop_live = False
        # Bumped by every action that starts a fresh loop; a stale loop stands down rather
        # than writing state over the newer loop's.
        self._run_gen = 0

        # --- approval gates (starts set; an auto job never waits) ---
        self._gate_event = asyncio.Event()
        self._gate_event.set()
        # Rides the snapshot so a polling client can render a decision screen.
        self.open_gate: Optional[dict] = None
        self._gate_decision: Optional[dict] = None  # {"action": ..., "restart_index": int|None}
        self.gate_history: List[dict] = []
        # Gate open when the process stopped; not answerable until rearm_gates() parks a waiter.
        self.pending_gate: Optional[dict] = None

        # --- event stream ---
        self._event_history: List[dict] = []
        self._subscriber_queues: List[asyncio.Queue] = []

        # Every data-changing intervention (override, skip). A report on hand-edited state
        # must be traceable.
        self.interventions: List[dict] = []

    def touch(self):
        self.updated_at = datetime.now(timezone.utc).isoformat()

    @property
    def current_pass(self) -> int:
        """Which retrieval pass the run is on (1-based)."""
        return int(getattr(self.context, "current_pass", 1) or 1)

    @property
    def total_passes(self) -> int:
        """How many retrieval passes this run has begun. 1 unless a follow-up was taken."""
        return max([1] + [split_pass_key(k)[1] for k in self.stage_keys])

    def register_pass(self, number: int) -> List[str]:
        """Add per-stage record keys for a follow-up pass. Returns the added keys.

        Inserted after the prior pass's keys (execution order). Idempotent: re-registering
        does not duplicate keys or reset statuses already recorded.
        """
        added: List[str] = []
        anchor = pass_key(_LAST_PASS_STAGE, number - 1)
        at = (
            self.stage_keys.index(anchor) + 1
            if anchor in self.stage_keys
            else len(self.stage_keys)
        )
        for stage_name in _PASS_STAGES:
            key = pass_key(stage_name, number)
            if key in self.stage_statuses:
                continue
            self.stage_keys.insert(at, key)
            at += 1
            self.stage_statuses[key] = StageStatus.PENDING
            added.append(key)
        return added

    def record_intervention(self, action, stage=None, detail="", actor=None):
        self.interventions.append(
            {
                "action": action,
                "stage": stage,
                "detail": detail,
                "actor": actor,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.touch()

    def snapshot(self) -> dict:
        """JSON-safe view for GET /jobs/{id}."""
        return {
            "job_id": self.job_id,
            "incident_id": self.incident.get("id"),
            "status": self.status.value,
            "run_mode": self.run_mode.value,
            "current_stage": self.current_stage,
            "description": self.incident.get("description"),
            "extended_retrieval": bool(self.incident.get("extended_retrieval")),
            # Who asked for this run, projected so a single-job read is attributable on its
            # own — the list endpoint is not the only place an admin looks.
            "owner": owner_of(self.incident),
            "owner_name": self.incident.get("owner_name"),
            # Queue position omitted: it changes as the backlog drains; list and batch
            # endpoints report the live value.
            "batch_id": self.incident.get("batch_id"),
            "stages": [
                {
                    "name": split_pass_key(key)[0],  # bare name; `pass` distinguishes multi-pass records
                    "pass": split_pass_key(key)[1],
                    "status": self.stage_statuses[key].value,
                    "duration_ms": self.stage_durations.get(key),
                    # Present only while running so a mid-attach client can continue the timer.
                    "started_at": (
                        self.stage_started_at.get(key)
                        if self.stage_statuses[key] == StageStatus.RUNNING
                        else None
                    ),
                    "summary": self.stage_summaries.get(key),
                    "health": self.stage_health.get(key),
                }
                for key in self.stage_keys
            ],
            "passes": {"total": self.total_passes, "current": self.current_pass},
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "interventions": list(self.interventions),
            "open_gate": dict(self.open_gate) if self.open_gate else None,
            # pending_gate is a held-over gate, not yet answerable; separate from open_gate
            # so a client never renders a button that resolves nothing.
            "pending_gate": dict(self.pending_gate) if self.pending_gate else None,
            "gate_history": list(self.gate_history),
        }


# Stage callables -- each mirrors one step of process_incident, reading from
# ctx.outputs so a stage can run standalone on retry.


def _guidance(ctx: JobContext, stage_name):
    """Analyst direction accumulated for this stage, or None."""
    items = (ctx.stage_guidance or {}).get(stage_name)
    return items or None


async def _run_understanding(ctx: JobContext):
    return await ctx.modules["understanding"].process(
        ctx.incident, guidance=_guidance(ctx, "understanding")
    )


def _pass_record(ctx: JobContext, pass_number: int) -> Dict[str, Any]:
    """This pass's own slice of ``ctx.pass_outputs``, created on first use."""
    return ctx.pass_outputs.setdefault(
        int(pass_number), {"queries": [], "logs": {}, "notes": []}
    )


def _engine_row_caps(ctx: JobContext) -> Dict[str, Any]:
    """{source: max_results} from the retrieval engine, or {}; an absent cap means unknown (not uncapped)."""
    try:
        engine = (ctx.modules or {}).get("log_retrieval")
        caps = engine.row_caps() if engine is not None else None
        return caps if isinstance(caps, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read row caps from the retrieval engine: %s", exc)
        return {}


def _accumulated_before(ctx: JobContext, pass_number: int, field: str):
    """field from every pass before pass_number in pass order; rebuilt from per-pass records so retry doesn't double-count."""
    numbers = sorted(n for n in ctx.pass_outputs if int(n) < int(pass_number))
    if field == "logs":
        merged: Dict[str, Any] = {}
        for number in numbers:
            for source, rows in (ctx.pass_outputs[number].get("logs") or {}).items():
                merged[source] = list(merged.get(source) or []) + list(rows or [])
        return merged
    out = []
    for number in numbers:
        out.extend(list(ctx.pass_outputs[number].get(field) or []))
    return out


def _adjudicating_ruleset_key(ctx: JobContext) -> Tuple[str, str]:
    """Ruleset key adjudicating this incident and how it resolved; ("", reason) when nothing resolves. Never raises."""
    generator = ctx.modules.get("api_call")
    pack = getattr(generator, "knowledge_pack", None)
    if pack is None:
        return "", "the run has no knowledge pack"
    understanding = ctx.outputs.get("understanding")
    analysis = getattr(understanding, "analysis", None)
    correlation = ctx.modules.get("correlation")
    unselected = False
    if analysis is not None and correlation is not None:
        try:
            spec, basis = correlation._playbook_correlation_spec_explained(analysis)
            unselected = bool(getattr(basis, "defaulted", False))
            key = pack.ruleset_key_for(str((spec or {}).get("use_case", "") or ""))
            if key:
                return key, "the matched playbook's use case"
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not resolve which procedure adjudicates this incident (%s); "
                "falling back to the pack's default for the follow-up decision.",
                exc,
            )
    try:
        # The reason distinguishes the two ways the default is reached, because an operator
        # reading "the pack's default ruleset" cannot tell a pack with one procedure from an
        # incident no procedure recognised.
        return pack.default_ruleset_key(), (
            "the pack's default ruleset, because NO procedure matched this incident"
            if unselected
            else "the pack's default ruleset"
        )
    except Exception:  # noqa: BLE001
        return "", "the pack could not name a default ruleset"


def pass_purpose(spec) -> str:
    """Declared purpose of one follow-up pass, falling back to per-target purposes joined."""
    if not isinstance(spec, dict):
        return ""
    one = str(spec.get("purpose", "") or "").strip()
    if one:
        return one
    per = spec.get("purposes") or {}
    if isinstance(per, dict):
        return "; ".join(
            f"{name}: {str(text or '').strip()}"
            for name, text in per.items()
            if str(text or "").strip()
        )
    return ""


def pass_targets(spec) -> str:
    """Sources a follow-up pass retrieves, comma-joined for operator-facing prose."""
    if not isinstance(spec, dict):
        return "?"
    names = [str(t).strip() for t in (spec.get("sources") or []) if str(t).strip()]
    if not names:
        names = [t for t in [str(spec.get("source", "") or "").strip()] if t]
    return ", ".join(names) or "?"


def follow_up_spec(ctx: JobContext, pass_number: int):
    """Follow-up spec for pass_number from the adjudicating ruleset; other rulesets' declarations are ignored. Unreadable → None."""
    if pass_number < 2:
        return None
    generator = ctx.modules.get("api_call")
    pack = getattr(generator, "knowledge_pack", None)
    if generator is None or pack is None:
        return None
    try:
        key, how = _adjudicating_ruleset_key(ctx)
        if not key:
            logger.info(
                "No procedure could be named for this incident (%s), so retrieval pass %s "
                "is not opened.",
                how,
                pass_number,
            )
            return None
        spec = pack.follow_up_pass(key, int(pass_number))
        if spec:
            logger.info(
                "Retrieval pass %s is declared by ruleset '%s' (%s), which is the procedure "
                "adjudicating this incident.",
                pass_number,
                key,
                how,
            )
            return spec
        # Nothing from the adjudicating ruleset. Log which ruleset(s) did declare it.
        others = [
            other
            for other in pack.ruleset_keys()
            if other != key and pack.follow_up_pass(other, int(pass_number))
        ]
        if others:
            logger.info(
                "Retrieval pass %s is declared by ruleset(s) %s but NOT by '%s', which is "
                "the procedure adjudicating this incident (%s), so the pass is not opened — "
                "its rows would answer none of the conditions this verdict is reached on.",
                pass_number,
                ", ".join(f"'{o}'" for o in others),
                key,
                how,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not determine whether a retrieval pass %s is declared (%s); "
            "continuing with the evidence the earlier pass(es) returned.",
            pass_number,
            exc,
        )
    return None


#: The query generator's dependency findings, in the order the scorer reads them.
_DEPENDENCY_FINDINGS = (
    "undeliverable_required",
    "declared_not_queried",
    "declared_unscopable",
    "selected_unparseable",
)


def _record_dependency_findings(ctx: JobContext, number: int, queries) -> None:
    """Write this run's dependency findings onto ctx; the generator is shared across jobs. Never raises."""
    try:
        generator = (ctx.modules or {}).get("api_call")
        if generator is None:
            return
        prior = ctx.stage_facts.get("query_generation") if number > 1 else None
        prior = prior if isinstance(prior, dict) else {}
        report = generator.dependency_report(
            list(queries or []), None, getattr(ctx.outputs.get("understanding"), "analysis", None)
        )
        facts = {
            "undeliverable_required": list(report.get("undeliverable") or []),
            "declared_not_queried": list(report.get("not_queried") or []),
            "declared_unscopable": list(report.get("unscopable") or []),
            "selected_unparseable": sorted(
                {
                    n
                    for n in list(prior.get("selected_unparseable") or [])
                    + list(getattr(generator, "selected_unparseable", None) or [])
                    if isinstance(n, str) and n
                }
            ),
        }
        ctx.stage_facts["query_generation"] = facts
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not record this run's plan dependency findings (%s: %s); stage health "
            "will fall back to the shared generator's attributes, which may hold another "
            "run's answer.",
            type(exc).__name__,
            exc,
        )


async def _run_query_generation(ctx: JobContext):
    understanding = ctx.outputs["understanding"]
    number = int(getattr(ctx, "current_pass", 1) or 1)
    guidance = _guidance(ctx, pass_key("query_generation", number))
    if number > 1:
        spec = follow_up_spec(ctx, number)
        if spec is None:
            # Only reachable on an out-of-band retry (pack edit, imported job).
            logger.warning(
                "Query generation was asked for pass %s, which the adjudicating ruleset "
                "does not declare; the accumulated queries are returned unchanged.",
                number,
            )
            return list(ctx.outputs.get("queries") or [])
        # Rebuild from per-pass records, not ctx.outputs: a re-run would double-count.
        prior = _accumulated_before(ctx, number, "queries")
        added, notes = await ctx.modules["api_call"].generate_follow_up(
            understanding,
            spec,
            ctx.outputs.get("logs") or {},
            prior_queries=prior,
            guidance=render_guidance(guidance) if guidance else "",
            row_caps=_engine_row_caps(ctx),  # a target can be skipped only if prior answer was complete
        )
        record = _pass_record(ctx, number)
        record["queries"] = list(added)
        record["notes"] = list(notes)
        # Accumulate: returning only the new queries would delete pass 1's plan from the document.
        accumulated = prior + list(added)
        _record_dependency_findings(ctx, number, accumulated)
        return accumulated
    planned = await ctx.modules["api_call"].generate(understanding, guidance=guidance)
    _record_dependency_findings(ctx, 1, planned)
    _pass_record(ctx, 1)["queries"] = list(planned or [])  # lets _accumulated_before rebuild
    return planned


async def _run_log_retrieval(ctx: JobContext):
    engine = ctx.modules["log_retrieval"]
    number = int(getattr(ctx, "current_pass", 1) or 1)
    # Rebuild from per-pass records; a re-run would otherwise double-count earlier rows.
    accumulated = _accumulated_before(ctx, number, "logs") if number > 1 else {}
    if number > 1:
        # Only this pass's queries; re-running earlier ones would double rows and cost.
        queries = list((ctx.pass_outputs.get(number) or {}).get("queries") or [])
        if not queries:
            logger.info(
                "Retrieval pass %s has no queries to run (the follow-up harvested "
                "nothing usable); the accumulated rows are returned unchanged.",
                number,
            )
            return accumulated
    else:
        queries = ctx.outputs["queries"]
    source_outcomes: Dict[str, Any] = {}  # lets the scorer tell "empty" from "timed out"
    # Which sources asked about the identity by full declared key. Lets the verdict tell
    # "not on the list" (finding) from "source never asked" (gap).
    keyed_sources: Dict[str, Any] = {}
    pass_queries: Dict[str, str] = {}  # backend query text, this pass; avoid shared-object state
    # Sources asked but unanswered: absent from `logs` (distinct from empty). Carried forward;
    # a source that later answers is popped.
    unanswered: Dict[str, str] = {}
    prior_facts = ctx.stage_facts.get("log_retrieval") if number > 1 else None
    if isinstance(prior_facts, dict):
        # Carry earlier passes' facts forward; consumers read one dict for the whole run.
        source_outcomes.update(dict(prior_facts.get("sources") or {}))
        keyed_sources.update(dict(prior_facts.get("keyed_sources") or {}))
        unanswered.update(dict(prior_facts.get("unanswered") or {}))
    ctx.stage_facts["log_retrieval"] = {
        "sources": source_outcomes,
        "keyed_sources": keyed_sources,
        "unanswered": unanswered,
    }
    # Per-pass facts only for follow-up passes: on pass 1 the key equals `log_retrieval`
    # and a second entry would overwrite the run-wide facts.
    pass_outcomes: Dict[str, Any] = {}
    if number > 1:
        ctx.stage_facts[pass_key("log_retrieval", number)] = {
            "sources": pass_outcomes,
            "pass": number,
        }

    def progress_cb(source, status, message):
        # query_ready annotates a still-running source and must not overwrite `running`.
        if status != _QUERY_READY:
            source_outcomes[source] = {"status": status, "message": message}
            pass_outcomes[source] = {"status": status, "message": message}
        retriever = getattr(engine, "retrievers", {}).get(source)
        data = {"source": source, "pass": number}  # pass lets the console key source#pass
        backend, target = _retriever_backend_target(retriever)
        if backend:
            data["backend"] = backend
        if target:
            data["target"] = target
        gen_q = pass_queries.get(source)  # from the dict the engine fills; never shared-object state
        if gen_q:
            data["generated_query"] = _clip(gen_q, _QUERY_STR)
        field_map = getattr(retriever, "last_field_map", None)
        if field_map:
            data["field_map"] = {str(k): str(v) for k, v in field_map.items()}
        emitter.emit(
            ctx.job_id,
            "source_progress",
            stage="log_retrieval",
            status=status,
            message=message,
            data=data,
        )

    extended = bool(
        ctx.incident.get("extended_retrieval")
        or engine.config.get("extended_retrieval")
    )
    if extended:
        logger.info("Log retrieval running in EXTENDED-timeout mode (opt-in).")
        # Record on the incident so snapshot() reports it correctly for global-config runs.
        ctx.incident["extended_retrieval"] = True
    guidance = render_guidance(_guidance(ctx, pass_key("log_retrieval", number)))
    extra = {"guidance": guidance} if guidance else {}
    if engine.config.get("use_ssh_tunnel"):
        fetched = await engine.retrieve_with_tunnel(
            queries,
            progress_cb=progress_cb,
            extended=extended,
            keyed_out=keyed_sources,
            queries_out=pass_queries,
            unanswered_out=unanswered,
            **extra,
        )
    else:
        fetched = await engine.retrieve(
            queries,
            progress_cb=progress_cb,
            extended=extended,
            keyed_out=keyed_sources,
            queries_out=pass_queries,
            unanswered_out=unanswered,
            **extra,
        )
    fetched = fetched if isinstance(fetched, dict) else {}
    _pass_record(ctx, number)["logs"] = {k: list(v or []) for k, v in fetched.items()}
    if number <= 1:
        return fetched
    # Merge, never replace: a re-queried source holds evidence from both passes.
    merged = accumulated
    for source, rows in fetched.items():
        existing = list(merged.get(source) or [])
        merged[source] = existing + list(rows or [])
    logger.info(
        "Retrieval pass %s added %d row(s) across %d source(s); the run now holds %d "
        "row(s) across %d source(s).",
        number,
        sum(len(v or []) for v in fetched.values()),
        len(fetched),
        sum(len(v or []) for v in merged.values()),
        len(merged),
    )
    return merged


async def _run_correlation(ctx: JobContext):
    if "correlation" not in ctx.modules:
        return None
    logs = ctx.outputs["logs"]
    understanding = ctx.outputs["understanding"]
    module = ctx.modules["correlation"]
    result = await module.analyze(
        logs,
        understanding,
        guidance=_guidance(ctx, "correlation"),
        keyed_sources=dict(
            (ctx.stage_facts.get("log_retrieval") or {}).get("keyed_sources") or {}
        ),
        unanswered_sources=dict(
            (ctx.stage_facts.get("log_retrieval") or {}).get("unanswered") or {}
        ),
        link_modes=dict(ctx.link_modes or {}),  # passed each call so gate-rejection re-run keeps it
    )
    _record_procedure_selection(ctx, module)
    return result


def _record_procedure_selection(ctx: JobContext, module) -> None:
    """Write how THIS run's procedure was chosen, and which narration path ran, onto ctx.

    Never raises. Same reason as `_record_dependency_findings`: the correlation module is
    shared across jobs, so its `last_selection` / `last_narration` attributes hold whichever
    run finished most recently.
    """
    try:
        facts = ctx.stage_facts.get("correlation")
        facts = dict(facts) if isinstance(facts, dict) else {}
        mode = getattr(module, "last_narration", None)
        if isinstance(mode, str) and mode:
            facts["narration"] = mode
        basis = getattr(module, "last_selection", None)
        if hasattr(basis, "to_dict"):
            facts["procedure_selection"] = basis.to_dict()
        ctx.stage_facts["correlation"] = facts
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not record how this run's procedure was selected (%s: %s); stage health "
            "falls back to the shared module's attribute, which may hold another run's answer.",
            type(exc).__name__,
            exc,
        )


async def _run_anomaly_detection(ctx: JobContext):
    logs = ctx.outputs["logs"]
    understanding = ctx.outputs["understanding"]
    correlation = ctx.outputs.get("correlation")
    return await ctx.modules["anomaly_detection"].detect(
        logs, understanding, correlation, guidance=_guidance(ctx, "anomaly_detection")
    )


async def _run_plugins(ctx: JobContext):
    plugins = ctx.modules["plugins"]
    understanding = ctx.outputs["understanding"]
    logs = ctx.outputs["logs"]
    anomalies = ctx.outputs["anomalies"]
    results = []
    for plugin in plugins.get_active_plugins():
        result = await plugins.execute_plugin(
            plugin, ctx.incident, understanding, logs, anomalies
        )
        results.append({"plugin": plugin, "result": result})
        logger.info(
            "Executed plugin %s for incident %s", plugin, ctx.incident.get("id")
        )
    return results


async def _run_report_generation(ctx: JobContext):
    understanding = ctx.outputs["understanding"]
    logs = ctx.outputs["logs"]
    anomalies = ctx.outputs["anomalies"]
    correlation = ctx.outputs.get("correlation")
    return await ctx.modules["report_generation"].generate(
        ctx.incident,
        understanding,
        logs,
        anomalies,
        correlation,
        guidance=_guidance(ctx, "report_generation"),
    )


def _run_export(ctx: JobContext):
    """Synchronous export next to the report (runs in a worker thread)."""
    incident = ctx.incident
    understanding = ctx.outputs["understanding"]
    anomalies = ctx.outputs["anomalies"]
    logs = ctx.outputs.get("logs") or {}
    correlation = ctx.outputs.get("correlation")
    exporter = ResultExporter(
        {
            "incident": incident,
            "understanding": {
                "analysis": understanding.analysis.model_dump_json(indent=2)
            },
            "anomalies": [a.model_dump() for a in anomalies],
        },
        storage=owner_scoped(ctx.modules.get("artifact_storage"), incident),
    )
    exports = exports_dir()
    json_path = str(exports / f"incident_{incident['id']}.json")
    csv_path = str(exports / f"incident_{incident['id']}.csv")
    raw_path = str(exports / f"evidence_raw_{incident['id']}.json")
    transformed_path = str(exports / f"evidence_transformed_{incident['id']}.json")
    exporter.export_json(json_path)
    exporter.export_csv(csv_path)
    # Separate evidence artifacts: full retrieved data and the transformed view.
    exporter.export_evidence_raw(raw_path, logs)
    exporter.export_evidence_transformed(transformed_path, correlation)
    return [json_path, csv_path, raw_path, transformed_path]


async def _run_output(ctx: JobContext):
    """Deliver the report through the configured channel (best-effort)."""
    if "output" not in ctx.modules:
        return None
    report = ctx.outputs.get("report")
    await ctx.modules["output"].send(report, ctx.incident["id"])
    return {"delivered": True}


def build_stage_descriptors() -> List[StageDescriptor]:
    """The nine pipeline stages, in canonical order (1:1 with process_incident)."""
    return [
        StageDescriptor("understanding", _run_understanding, "understanding"),
        StageDescriptor("query_generation", _run_query_generation, "queries"),
        StageDescriptor("log_retrieval", _run_log_retrieval, "logs"),
        StageDescriptor("correlation", _run_correlation, "correlation"),
        StageDescriptor("anomaly_detection", _run_anomaly_detection, "anomalies"),
        StageDescriptor("plugins", _run_plugins, "plugin_results"),
        StageDescriptor("report_generation", _run_report_generation, "report"),
        StageDescriptor("export", _run_export, "export_paths", is_synchronous=True),
        StageDescriptor("output", _run_output, "output", is_best_effort=True),
    ]


# Stage-output summaries, compact and JSON-safe; every summarizer is defensive and never raises.

# Per-collection item bound; _counted is the backstop that makes a cut say so.
_SUM_MAX_ITEMS_DEFAULT = 30  # list entries kept per collection, unless configured
_SUM_MAX_ITEMS = _SUM_MAX_ITEMS_DEFAULT  # effective bound; see configure_summaries
_SUM_MAX_SAMPLES = 3  # sample rows per log source
_SUM_STR = 300  # per-string cap for identifiers and labels
_SUM_ROW_STR = 500  # per-sample-row JSON cap

_PROSE_STR = 4000  # larger cap for human-read narrative; clipping mid-word reads as a defect
# Full-statement budget; partial clips hide filter and guard rewrites.
_QUERY_STR = 20000


def _clip(value, limit=_SUM_STR):
    """Coerce to a display string, clipped with an ellipsis marker."""
    try:
        s = value if isinstance(value, str) else str(value)
    except Exception:  # noqa: BLE001
        return ""
    return s if len(s) <= limit else s[:limit] + "…"


def _prose(value):
    """:func:`_clip` at the `PROSE_STR` cap; for text a human reads."""
    return _clip(value, _PROSE_STR)


def _counted(items, limit=None):
    """(bounded_list, true_total). limit reads the module global at call time so configure_summaries takes effect."""
    limit = _SUM_MAX_ITEMS if limit is None else limit
    seq = list(items or [])
    return seq[:limit], len(seq)


def configure_summaries(config):
    """Apply jobs.summary_max_items to the summary bound; out-of-range values fall back to default. Returns the effective value."""
    global _SUM_MAX_ITEMS
    raw = ((config or {}).get("jobs") or {}).get(
        "summary_max_items", _SUM_MAX_ITEMS_DEFAULT
    )
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = _SUM_MAX_ITEMS_DEFAULT
    if not 1 <= value <= 1000:
        value = _SUM_MAX_ITEMS_DEFAULT
    _SUM_MAX_ITEMS = value
    return _SUM_MAX_ITEMS


def _retriever_backend_target(retriever):
    """(backend_kind, concrete_target) for a retriever; backend_kind is the class family, target is the index or table(s). Never raises."""
    if retriever is None:
        return None, None
    cls = type(retriever).__name__
    backend = {
        "ElasticsearchRetriever": "elasticsearch",
        "KibanaRetriever": "kibana (elk)",
        "DatabricksRetriever": "databricks",
        "SnowflakeRetriever": "snowflake",
        "RestRetriever": "rest",
    }.get(cls, cls)
    target = None
    try:
        idx = getattr(retriever, "index", None)
        if idx:
            target = idx if isinstance(idx, str) else ", ".join(str(i) for i in idx)
        else:
            catalog = getattr(retriever, "catalog", None)
            tables = getattr(retriever, "tables", None) or []
            if catalog and tables:
                target = ", ".join(f"{catalog}…{t}" for t in tables)
            elif catalog:
                target = str(catalog)
    except Exception:  # noqa: BLE001
        target = None
    return backend, target


def _compact_row(row, limit=_SUM_ROW_STR):
    """A single retrieved row → a small JSON-safe dict (or clipped repr)."""
    if not isinstance(row, dict):
        return {"value": _clip(row, limit)}
    out = {}
    budget = limit
    for k, v in row.items():
        piece = (
            v if isinstance(v, (str, int, float, bool)) or v is None else _clip(v, 120)
        )
        out[str(k)] = piece if not isinstance(piece, str) else _clip(piece, 120)
        budget -= len(str(k)) + len(str(piece))
        if budget <= 0:
            out["…"] = "truncated"
            break
    return out


def _ent(e):
    """One extracted entity with its classified surface form, included so reviewers can see the routing."""
    form = getattr(e, "value_form", "") or ""
    out = {"type": getattr(e, "type", ""), "value": getattr(e, "value", "")}
    if form:
        out["value_form"] = form
    return out


def _summ_understanding(o):
    a = getattr(o, "analysis", None)
    if a is None:
        return {}
    # Totals ride alongside the bounded lists so the UI can say when it shows a subset.
    ents, ent_total = _counted(getattr(a, "extracted_entities", []))
    ent = [_ent(e) for e in ents]
    sources, source_total = _counted(getattr(a, "log_sources_to_review", []))
    ev = getattr(a, "event_time", None)
    return {
        "summary": _prose(getattr(a, "incident_summary", "")),
        "severity": getattr(a, "severity", "") or "",
        "severity_reasoning": _prose(getattr(a, "severity_reasoning", "")),
        "impact_assessment": _prose(getattr(a, "impact_assessment", "")),
        "entities": ent,
        "entity_count": ent_total,
        "correlation_keys": _counted(getattr(a, "correlation_keys", []))[0],
        "event_time": (
            {"start": getattr(ev, "start", ""), "end": getattr(ev, "end", "")}
            if ev is not None
            else None
        ),
        "log_sources_to_review": sources,
        "source_review_count": source_total,
        "initial_hypotheses": [
            _prose(h) for h in _counted(getattr(a, "initial_hypotheses", []))[0]
        ],
        "recommended_actions": [
            _prose(x) for x in _counted(getattr(a, "recommended_actions", []))[0]
        ],
    }


def _summ_queries(o):
    queries = []
    kept, total = _counted(o)
    for q in kept:
        queries.append(
            {
                "source": getattr(q, "target_log_source", ""),
                # _prose, not the label cap: the gate shows this text to the operator.
                "query": _prose(getattr(q, "natural_language_query", "")),
                "date_from": getattr(q, "date_from", ""),
                "date_to": getattr(q, "date_to", ""),
                "entities": [_ent(e) for e in _counted(getattr(q, "entities", []))[0]],
            }
        )
    # count is the number that will run; queries is what the gate displays.
    return {"count": total, "queries": queries}


def _summ_logs(o):
    if not isinstance(o, dict):
        return {"total_rows": 0, "sources": []}
    sources, total = [], 0
    for name, rows in o.items():
        rows = rows or []
        total += len(rows)
        sources.append(
            {
                "source": name,
                "rows": len(rows),
                "samples": [_compact_row(r) for r in rows[:_SUM_MAX_SAMPLES]],
            }
        )
    sources.sort(key=lambda s: s["rows"], reverse=True)
    return {"total_rows": total, "source_count": len(sources), "sources": sources}


def _summ_correlation(o):
    if o is None:
        return {"skipped": True}
    agg = getattr(o, "aggregations", {}) or {}
    resolved = agg.get("resolved_correlation_keys") or []
    discovered = agg.get("discovered_join_keys") or []

    # Keep only display-relevant fields from any key dicts.
    def _key_view(k):
        if not isinstance(k, dict):
            return {"value": _clip(k)}
        return {
            "entity_hint": k.get("entity_hint") or k.get("hint", ""),
            "sources": k.get("sources", {}),
            "time_window": k.get("time_window", ""),
            "origin": k.get("origin", ""),
            "overlap_score": k.get("overlap_score", 0.0),
        }

    transforms = [
        {
            "label": getattr(t, "label", ""),
            "op": getattr(t, "op", ""),
            "row_count": len(getattr(t, "rows", []) or []),
            "note": _clip(getattr(t, "note", "")),
        }
        for t in _counted(getattr(o, "transforms", []))[0]
    ]
    findings = [
        {
            "title": _clip(getattr(f, "title", "")),
            "description": _prose(getattr(f, "description", "")),
        }
        for f in _counted(getattr(o, "findings", []))[0]
    ]
    # A small subset of the raw aggregation map (skip the big/nested key lists).
    agg_highlights = {
        k: v
        for k, v in agg.items()
        if k not in ("resolved_correlation_keys", "discovered_join_keys")
    }
    # Investigation-evidence highlights (defensive; evidence may be absent).
    evidence = getattr(o, "evidence", None)
    evidence_view = None
    if evidence is not None:
        actors = getattr(evidence, "actors", []) or []
        evidence_view = {
            "chronology_events": len(getattr(evidence, "chronology", []) or []),
            "chronology_aggregated": bool(
                getattr(evidence, "chronology_aggregated", False)
            ),
            "actor_count": len(actors),
            "top_actors": [getattr(act, "actor", "") for act in actors[:3]],
            "cross_source_joins": len(
                getattr(evidence, "cross_source_joins", []) or []
            ),
            "degraded": bool(getattr(evidence, "degraded", False)),
        }
    # Pack-driven validation verdict: compact per-subject rollup for the UI
    # correlation card and ?verbose=1. Defensive; verdict may be absent.
    verdict = getattr(o, "verdict", None)
    verdict_view = None
    _vsubjects = getattr(verdict, "subjects", None) if verdict is not None else None
    if verdict is not None and isinstance(_vsubjects, list):
        subjects = _vsubjects

        def _subject_view(s):
            checks = getattr(s, "checks", []) or []
            lock = getattr(s, "lock_target", {}) or {}
            return {
                "subject": _clip(getattr(s, "subject_value", "")),
                "verdict": getattr(s, "verdict", ""),
                "n_pass": sum(1 for c in checks if getattr(c, "result", "") == "pass"),
                "n_fail": sum(1 for c in checks if getattr(c, "result", "") == "fail"),
                "n_unknown": sum(
                    1 for c in checks if getattr(c, "result", "") == "unknown"
                ),
                "lock_scope": lock.get("scope", ""),
                "lock_identity": lock.get("identity", ""),
            }

        verdict_view = {
            "label_scheme": getattr(verdict, "label_scheme", ""),
            "summary": _prose(getattr(verdict, "summary", "")),
            "degraded": bool(getattr(verdict, "degraded", False)),
            "has_notification": bool(getattr(verdict, "notification_draft", "")),
            "subject_count": len(subjects),
            "subjects": [_subject_view(s) for s in _counted(subjects)[0]],
        }
    # Investigation brief -- compact view for the UI. Defensive; brief may be absent.
    brief = getattr(o, "brief", None)
    brief_view = None
    if brief is not None:
        _fails = getattr(brief, "decisive_fails", None)
        _unknowns = getattr(brief, "decisive_unknowns", None)
        _indicators = getattr(brief, "decisive_indicators", None)
        _precedents = getattr(brief, "precedents", None)
        _join = getattr(brief, "join_status", None)
        _timeline = getattr(brief, "asset_timeline", None)
        _verdict = getattr(brief, "verdict", None)
        if (
            isinstance(_fails, list)
            or isinstance(_unknowns, list)
            or isinstance(getattr(brief, "action_backbone", None), list)
        ):
            brief_view = {
                "use_case": _clip(getattr(brief, "use_case", "")),
                "playbook_id": _clip(getattr(brief, "playbook_id", "")),
                "verdict_summary": _prose(
                    getattr(_verdict, "summary", "") if _verdict is not None else ""
                ),
                # These three are the verdict's reasoning; a dropped entry reads as absent.
                "decisive_fails": _counted(
                    [getattr(c, "id", "") for c in (_fails or [])]
                    if isinstance(_fails, list)
                    else []
                )[0],
                "decisive_unknowns": _counted(
                    [getattr(c, "id", "") for c in (_unknowns or [])]
                    if isinstance(_unknowns, list)
                    else []
                )[0],
                "decisive_indicators": _counted(
                    [getattr(c, "id", "") for c in (_indicators or [])]
                    if isinstance(_indicators, list)
                    else []
                )[0],
                "join_status": _join if isinstance(_join, dict) else {},
                "asset_timeline_events": (
                    len(_timeline) if isinstance(_timeline, list) else 0
                ),
                "precedents": _counted(
                    [getattr(p, "case_id", "") for p in (_precedents or [])]
                    if isinstance(_precedents, list)
                    else []
                )[0],
                "degraded": bool(getattr(brief, "degraded", False)),
            }

    # Read back off the notes the annotation wrote (correlation._annotate_procedure_selection)
    # rather than off `ctx.stage_facts`: `_summ_correlation` sees only the result, and deriving
    # it here means the UI states exactly what the report states, never a second computation of
    # the same fact that can disagree with it. One run-level line, not one per subject: the
    # remedy (`pinned_use_case`) is run-level too.
    def _unselected_note(res) -> str:
        holders = list(
            getattr(getattr(res, "verdict", None), "subjects", None) or []
        ) + ([getattr(res, "brief", None)] if getattr(res, "brief", None) else [])
        for holder in holders:
            for note in getattr(holder, "notes", None) or []:
                if isinstance(note, str) and note.startswith("procedure_unselected="):
                    return _prose(note.split("=", 1)[1])
        return ""

    # The advisory lane. Carried in the summary so an operator can act from here.
    # state rides verbatim: the four states must not collapse to a boolean anywhere.
    _links = getattr(o, "links", None)

    def _pivot_values(f):
        vals = getattr(f, "pivot_values", None)
        return [_clip(v, 120) for v in vals] if isinstance(vals, list) else []

    def _reasons(f):
        vals = getattr(f, "link_score_reasons", None)
        return [_prose(v) for v in vals[:6]] if isinstance(vals, list) else []

    def _as_score(f):
        # Coerce to float so a test double does not land on str(...).
        raw = getattr(f, "link_score", 0.0)
        try:
            return max(0.0, min(1.0, round(float(raw), 4)))
        except (TypeError, ValueError):
            return 0.0

    links_view = (
        [
            {
                "target_use_case": _clip(getattr(f, "target_use_case", "")),
                "target_playbook_id": _clip(getattr(f, "target_playbook_id", "")),
                "direction": _clip(getattr(f, "direction", ""), 40),
                "state": _clip(getattr(f, "state", ""), 40),
                "rung": (
                    getattr(f, "rung", 0)
                    if isinstance(getattr(f, "rung", None), int)
                    else 0
                ),
                "pivot_entity": _clip(getattr(f, "pivot_entity", ""), 80),
                "pivot_values": _counted(_pivot_values(f))[0],
                "pivot_value_count": len(_pivot_values(f)),
                "signal_id": _clip(getattr(f, "signal_id", ""), 120),
                "evidence_note": _prose(getattr(f, "evidence_note", "")),
                "advisory_severity": _clip(getattr(f, "advisory_severity", ""), 40),
                "advisory_note": _prose(getattr(f, "advisory_note", "")),
                "window_hint": _clip(getattr(f, "window_hint", ""), 120),
                "gap_reason": _prose(getattr(f, "gap_reason", "")),
                "base_rate": _clip(getattr(f, "base_rate", ""), 200),
                # Escalation setting: four fields because they answer different questions
                # (what the mode is, which layer set it, why, and what follows next).
                "mode": _clip(getattr(f, "mode", ""), 40),
                "mode_source": _clip(getattr(f, "mode_source", ""), 40),
                "mode_note": _prose(getattr(f, "mode_note", "")),
                "proposed_action": _prose(getattr(f, "proposed_action", "")),
                # Rung-1 outcome beside mode_licensed; "not this procedure" vs "no rows" license opposite next steps.
                "gate_outcome": _clip(getattr(f, "gate_outcome", ""), 40),
                # A mode the server silently refuses is indistinguishable from one that works until it is used.
                "mode_licensed": bool(getattr(f, "mode_licensed", False)),
                "mode_corpus": (
                    getattr(f, "mode_corpus", 0)
                    if isinstance(getattr(f, "mode_corpus", None), int)
                    else 0
                ),
                # Score and terms: semi_auto behaviour changes per incident, so a bare
                # number cannot be acted on.
                "link_score": _as_score(f),
                "link_score_reasons": _reasons(f),
                # probe_spent=True on a not_probed candidate must not re-authorise it.
                "probe_spent": bool(getattr(f, "probe_spent", False)),
                "probe_source": _clip(getattr(f, "probe_source", ""), 120),
                "probe_note": _prose(getattr(f, "probe_note", "")),
                # Not clipped to prose length: truncation would render a button that resolves nothing.
                "child_job_id": _clip(getattr(f, "child_job_id", ""), 64),
                "child_note": _prose(getattr(f, "child_note", "")),
            }
            for f in _counted(_links)[0]
        ]
        if isinstance(_links, list)
        else []
    )
    # The other advisory lane. Same posture, and the five states ride verbatim for the same
    # reason: an unasked question must never render as an answered one.
    _inquiries = getattr(o, "inquiries", None)

    def _scope_values(f):
        vals = getattr(f, "scope_values", None)
        return [_clip(v, 120) for v in vals] if isinstance(vals, list) else []

    inquiries_view = (
        [
            {
                "id": _clip(getattr(f, "id", ""), 120),
                "state": _clip(getattr(f, "state", ""), 40),
                "question": _prose(getattr(f, "question", "")),
                "source": _clip(getattr(f, "source", ""), 120),
                "trigger": _prose(getattr(f, "trigger", "")),
                "trigger_condition": _clip(getattr(f, "trigger_condition", ""), 120),
                "trigger_result": _clip(getattr(f, "trigger_result", ""), 40),
                "scope_entity": _clip(getattr(f, "scope_entity", ""), 80),
                "scope_values": _counted(_scope_values(f))[0],
                "scope_value_count": len(_scope_values(f)),
                # The pack's own sentence for the outcome. Without it the count below is a
                # number nobody can act on, which is the whole reason the lane declares it.
                "meaning": _prose(getattr(f, "meaning", "")),
                "rows_matched": (
                    getattr(f, "rows_matched", 0)
                    if isinstance(getattr(f, "rows_matched", None), int)
                    else 0
                ),
                # A count at its cap is a floor, not a total.
                "row_cap_hit": bool(getattr(f, "row_cap_hit", False)),
                # probe_spent=False on an answered question is the free rung, not a refusal.
                "probe_spent": bool(getattr(f, "probe_spent", False)),
                "probe_note": _prose(getattr(f, "probe_note", "")),
                "gap_reason": _prose(getattr(f, "gap_reason", "")),
                "advisory_note": _prose(getattr(f, "advisory_note", "")),
                "note": _prose(getattr(f, "note", "")),
            }
            for f in _counted(_inquiries)[0]
        ]
        if isinstance(_inquiries, list)
        else []
    )
    return {
        "record_count": getattr(o, "record_count", 0),
        "resolved_correlation_keys": [_key_view(k) for k in _counted(resolved)[0]],
        "resolved_key_count": len(resolved),
        "discovered_join_keys": [_key_view(k) for k in _counted(discovered)[0]],
        "discovered_key_count": len(discovered),
        "transforms": transforms,
        "transform_count": len(getattr(o, "transforms", []) or []),
        "findings": findings,
        "finding_count": len(getattr(o, "findings", []) or []),
        "summary_text": _prose(getattr(o, "summary_text", "")),
        "aggregation_keys": _counted(list(agg_highlights.keys()))[0],
        "evidence": evidence_view,
        "verdict": verdict_view,
        "brief": brief_view,
        "links": links_view,
        # True total beside the bounded list.
        "link_count": len(_links) if isinstance(_links, list) else 0,
        "inquiries": inquiries_view,
        "inquiry_count": len(_inquiries) if isinstance(_inquiries, list) else 0,
        # Empty on every run whose procedure was chosen, which is every run today.
        "procedure_unselected": _unselected_note(o),
    }


def _summ_anomalies(o):
    items = sorted(
        (o or []),
        key=lambda a: getattr(a, "confidence_score", 0.0),
        reverse=True,
    )
    top = [
        {
            "description": _prose(getattr(a, "description", "")),
            "confidence_score": getattr(a, "confidence_score", 0.0),
            "potential_implications": _prose(getattr(a, "potential_implications", "")),
            "recommended_actions": _prose(getattr(a, "recommended_actions", "")),
        }
        for a in _counted(items)[0]
    ]
    return {"count": len(o or []), "anomalies": top}


def _summ_plugins(o):
    return {
        "count": len(o or []),
        "plugins": [
            {"plugin": p.get("plugin", ""), "result": _clip(p.get("result", ""), 200)}
            for p in _counted(o)[0]
            if isinstance(p, dict)
        ],
    }


def _report_sections(report):
    """Best-effort outline (section titles) from a report of any supported shape."""
    data = report
    if hasattr(report, "model_dump"):
        data = report.model_dump()
    if isinstance(data, dict):
        secs = data.get("sections")
        if isinstance(secs, list):
            return _counted(
                [
                    _clip(s.get("section_title", ""), 120)
                    for s in secs
                    if isinstance(s, dict)
                ]
            )[0]
    return []


def _summ_report(o):
    if isinstance(o, (bytes, bytearray)):
        return {"format": "binary", "size_bytes": len(o)}
    return {"sections": _report_sections(o)}


def _summ_export(o):
    return {"paths": [_clip(p, 400) for p in _counted(o)[0]]}


def _summ_output(o):
    if not isinstance(o, dict):
        return {"delivered": bool(o)}
    return {"delivered": bool(o.get("delivered"))}


# stage name -> summarizer over that stage's return value.
_STAGE_SUMMARIZERS = {
    "understanding": _summ_understanding,
    "query_generation": _summ_queries,
    "log_retrieval": _summ_logs,
    "correlation": _summ_correlation,
    "anomaly_detection": _summ_anomalies,
    "plugins": _summ_plugins,
    "report_generation": _summ_report,
    "export": _summ_export,
    "output": _summ_output,
}


def summarize_stage(name, output):
    """Compact, JSON-safe summary of a stage's result. Never raises."""
    fn = _STAGE_SUMMARIZERS.get(name)
    if fn is None:
        return {}
    try:
        return fn(output)
    except Exception as exc:  # noqa: BLE001
        logger.debug("summarize_stage(%s) failed: %s", name, exc)
        return {}


# Export / import codecs: serialize each output to JSON-safe form and back.
# Keys not listed pass through untouched (logs, plugin_results, export_paths).


def _encode_report(report):
    if report is None:
        return None
    if isinstance(report, (bytes, bytearray)):
        return {"__bytes_b64__": base64.b64encode(bytes(report)).decode("ascii")}
    if hasattr(report, "model_dump"):
        return {"__model__": "report", "data": report.model_dump(mode="json")}
    return report


def _decode_report(doc):
    if isinstance(doc, dict) and "__bytes_b64__" in doc:
        return base64.b64decode(doc["__bytes_b64__"])
    if isinstance(doc, dict) and doc.get("__model__") == "report":
        return doc["data"]  # left as dict; report is terminal, no model needed
    return doc


_OUTPUT_CODECS = {
    "understanding": (
        lambda o: o.model_dump(mode="json"),
        lambda d: UnderstandingResult.model_validate(d),
    ),
    "queries": (
        lambda o: [q.model_dump(mode="json") for q in o],
        lambda d: [RetrievalQuery.model_validate(x) for x in d],
    ),
    "correlation": (
        lambda o: None if o is None else o.model_dump(mode="json"),
        lambda d: None if d is None else CorrelationResult.model_validate(d),
    ),
    "anomalies": (
        lambda o: [a.model_dump(mode="json") for a in o],
        lambda d: [AnomalyItem.model_validate(x) for x in d],
    ),
    "report": (_encode_report, _decode_report),
}


def _encode_outputs(outputs: dict) -> dict:
    """Encode ctx.outputs for export; None encodes as None."""
    encoded = {}
    for key, value in outputs.items():
        codec = _OUTPUT_CODECS.get(key)
        encoded[key] = codec[0](value) if (codec and value is not None) else value
    return encoded


def _decode_outputs(doc: dict) -> dict:
    """Inverse of ``_encode_outputs``; ``None`` round-trips as ``None`` (see above)."""
    decoded = {}
    for key, value in (doc or {}).items():
        codec = _OUTPUT_CODECS.get(key)
        decoded[key] = codec[1](value) if (codec and value is not None) else value
    return decoded


def decode_output_value(output_key: str, value):
    """Decode one JSON-supplied stage output through the import codec. Raises ValueError on a bad shape."""
    codec = _OUTPUT_CODECS.get(output_key)
    if codec is None:
        return value  # untyped passthrough (logs, plugin_results, export_paths)
    try:
        return codec[1](value)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Value does not fit the '{output_key}' contract: {e}") from e


class JobManager:
    """Creates, runs, controls, and streams jobs over the fixed stage list."""

    def __init__(
        self,
        stages,
        event_emitter,
        modules=None,
        config=None,
        ttl_seconds=None,
        store=None,
    ):
        self.stages = stages
        self.stage_names = [s.name for s in stages]
        self.emitter = event_emitter
        self.modules = modules or {}
        self.config = config or {}
        jobs_cfg = self.config.get("jobs") or {}
        # An explicit argument wins (0 and -1 are meaningful, so `is None` and not falsiness).
        self.ttl_seconds = (
            ttl_seconds
            if ttl_seconds is not None
            else _as_int(jobs_cfg.get("completed_ttl_seconds"), _JOB_TTL_SECONDS)
        )
        self.history_max_items = _as_int(
            jobs_cfg.get("history_max_items"), _HISTORY_MAX_ITEMS
        )
        # Optional durable store. None keeps the manager in-memory (classic blocking
        # endpoints, existing tests).
        self.store = store
        self._jobs: Dict[str, Job] = {}
        # Compact rows of runs evicted by _prune, so the TTL bounds MEMORY and not the
        # operator's history. Rehydrated from the store on demand, one read per click.
        self._history: Dict[str, dict] = {}
        # Reads its bounds from self.config on every call, so a width change applies to the next admission.
        self.queue = JobQueue(self.config)
        # Per-process child-run counter for the rung-4 backstop. Not persisted:
        # a restart is the way to get more children.
        self._link_children_total = 0
        # Built lazily on the running loop: a Semaphore bound to the wrong loop
        # refuses every acquire.
        self._child_semaphore: Optional[asyncio.Semaphore] = None
        self._child_semaphore_size = 0

    # -- persistence -------------------------------------------------------

    def _persist(self, job: Job, evidence_changed: bool = False):
        """Best-effort snapshot of one job to the durable store. Never raises.

        JobStore already logs its own failures.
        """
        if self.store is None:
            return
        try:
            self.store.save(
                self.export_job(job.job_id), evidence_changed=evidence_changed
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not persist job %s: %s", job.job_id, exc)

    def _flush_store(self, job: Job, timeout: float):
        """Wait (bounded) for the store's queued writes to land; a no-op for synchronous backends. Never raises."""
        if self.store is None:
            return
        try:
            if not self.store.flush(timeout=timeout):
                logger.error(
                    "Job %s was persisted but the write has not reached durable storage "
                    "after %.0fs. If this process dies now, that state is lost.",
                    job.job_id,
                    timeout,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not flush job %s to durable storage: %s", job.job_id, exc)

    async def _persist_durably(self, job: Job, evidence_changed: bool = False):
        """Persist and flush to durable storage without blocking the event loop."""
        self._persist(job, evidence_changed=evidence_changed)
        if self.store is None:
            return
        await asyncio.to_thread(self._flush_store, job, GATE_PERSIST_FLUSH_SECONDS)

    async def flush_persistence(self, job_id: str):
        """Flush this job's persisted state to durable storage; for callers that cannot await inside a mutation. Never raises."""
        job = self._jobs.get(job_id)
        if job is None or self.store is None:
            return
        await asyncio.to_thread(self._flush_store, job, GATE_PERSIST_FLUSH_SECONDS)

    def restore(self) -> int:
        """Reload persisted jobs into memory; mid-stage jobs come back as paused. Returns the count restored."""
        if self.store is None:
            return 0
        restored = 0
        try:
            docs = self.store.load_all()
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not list persisted jobs: %s", exc)
            return 0
        for doc in docs:
            try:
                job = self.import_job(doc)
            except (
                Exception
            ) as exc:  # noqa: BLE001
                logger.warning(
                    "Could not restore persisted job %s: %s", doc.get("job_id"), exc
                )
                continue
            restored += 1
            if job.gate_history or job.context.stage_guidance:
                logger.info(
                    "Restored job %s (status=%s) with %d prior gate decision(s).",
                    job.job_id,
                    job.status.value,
                    len(job.gate_history),
                )
        if restored:
            logger.info(
                "Restored %d persisted job(s). Jobs interrupted mid-stage are PAUSED "
                "and need an explicit resume/retry/cancel.",
                restored,
            )
        try:
            self.store.prune()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Job store prune failed: %s", exc)
        return restored

    def rearm_gates(self) -> int:
        """Park a fresh waiter on each restored-but-unresolved gate. Requires a running loop; called separately from restore(). Returns the count."""
        rearmed = 0
        for job in list(self._jobs.values()):
            gate = getattr(job, "pending_gate", None)
            if not gate:
                continue
            stage_name = gate.get("stage")
            if stage_name not in job.stage_statuses:
                logger.warning(
                    "Job %s had a pending gate on unknown stage '%s'; dropped.",
                    job.job_id,
                    stage_name,
                )
                job.pending_gate = None
                continue
            job.pending_gate = None
            record = dict(gate)
            record["reopened_after_restart"] = True
            asyncio.ensure_future(self._rearm_one(job, record))
            rearmed += 1
        if rearmed:
            logger.info(
                "Re-armed %d approval gate(s) held over from before the restart.",
                rearmed,
            )
        return rearmed

    async def _rearm_one(self, job: Job, record: dict):
        """Park a runner on a restored gate, then continue the pipeline from after it."""
        stage_name = record["stage"]
        # The pass the gate was opened on, as the record itself states it. Re-arming
        # as pass 1 would continue from the first fetch's position on approval.
        pass_number = int(record.get("pass") or 1)
        stage = self._stage_for_key(stage_name)
        job.open_gate = record
        job._gate_decision = None
        job._gate_event.clear()
        # Lift the import_job pause so approval doesn't immediately re-pause at the barrier.
        job._pause_event.set()
        job.status = JobStatus.AWAITING_APPROVAL
        job.touch()
        logger.info(
            "[gate] job=%s stage=%s RE-OPENED after restart — awaiting a decision",
            job.job_id,
            stage_name,
        )
        self._emit(
            job,
            "gate_opened",
            stage=stage_name,
            status=job.status.value,
            message=f"{stage_name} awaiting approval (held over from before a restart)",
            data=dict(record),
        )
        self._persist(job)
        # _loop_live=True so cancel takes the "loop running" path, not the direct-cancel path.
        job._loop_live = True
        generation = job._run_gen
        try:
            next_index = await self._wait_for_gate_decision(job, stage, pass_number)
        finally:
            if job._run_gen == generation:
                job._loop_live = False
        if next_index is None:
            return
        await self._run_stages(job, next_index)

    # -- lifecycle ---------------------------------------------------------

    def create_job(self, incident, run_mode=JobRunMode.AUTO) -> Job:
        self._prune()
        job_id = str(uuid.uuid4())
        ctx = JobContext(
            job_id=job_id,
            incident=incident,
            modules=self.modules,
            config=self.config,
        )
        job = Job(job_id, incident, self.stage_names, run_mode, ctx)
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id) -> Optional[Job]:
        return self._jobs.get(job_id)

    def hydrate(self, job_id) -> Optional[Job]:
        """The job, reloading it from the store if the TTL has evicted it. One read, on demand.

        Deliberately not folded into `get_job`, which is on the hot control paths: a run
        that is not in memory is not startable, and only a READ should pay for a fetch.
        """
        job = self._jobs.get(job_id)
        if job is not None or self.store is None:
            return job
        try:
            doc = self.store.load_one(str(job_id))
        except Exception as exc:  # noqa: BLE001 — a read must not fail the request
            logger.debug("Could not read persisted job %s: %s", job_id, exc)
            return None
        if not doc:
            return None
        try:
            job = self.import_job(doc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Persisted job %s could not be rehydrated: %s", job_id, exc)
            return None
        self._history.pop(str(job_id), None)
        logger.info(
            "Rehydrated job %s (status=%s) from the store on demand.",
            job.job_id,
            job.status.value,
        )
        return job

    # -- the run queue -----------------------------------------------------

    def submit_job(self, job: Job) -> bool:
        """Start job now if a slot is free, else queue it; raises QueueFull at the bound. True if it started."""
        if self.queue.admit(job.job_id):
            asyncio.ensure_future(self.run_job(job))
            return True
        job.status = JobStatus.QUEUED
        job.touch()
        position = self.queue.position(job.job_id)
        self._emit(
            job,
            "job_status",
            status=job.status.value,
            message=(
                f"Queued at position {position}; {self.queue.width()} run(s) in flight"
            ),
            data={"queue_position": position, "queue": self.queue.stats()},
        )
        self._persist(job)
        return False

    def _release_slot(self, job_id: str):
        """Give up a slot and start whatever the backlog admits; idempotent, safe for jobs that never held one. Never raises."""
        try:
            for next_id in self.queue.release(job_id):
                self._start_queued(next_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not drain the run queue after %s: %s", job_id, exc)

    def _release_to_a_human(self, job: Job, why: str):
        """Give up this run's queue slot while it waits on a gate or pause; the resume is not re-queued."""
        if job.job_id not in self.queue.running():
            return
        logger.info("Job %s released its run queue slot while %s.", job.job_id, why)
        self._release_slot(job.job_id)

    def _start_queued(self, job_id: str):
        """Start one job the backlog admitted, or pass its slot on if the job no longer needs it (admission is a proposal, not a start)."""
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.QUEUED:
            logger.info(
                "Queued job %s is no longer waiting to start (%s); its slot goes to the "
                "next submission.",
                job_id,
                job.status.value if job is not None else "no longer in memory",
            )
            self._release_slot(job_id)
            return
        logger.info(
            "Starting queued job %s from the backlog; %d still waiting.",
            job_id,
            self.queue.stats()["queued"],
        )
        self._emit(
            job,
            "job_status",
            status=JobStatus.RUNNING.value,
            message="Starting from the run queue",
            data={"queue": self.queue.stats()},
        )
        asyncio.ensure_future(self.run_job(job))

    def resume_queued(self) -> int:
        """Re-enqueue jobs restored as queued, oldest first; requires a running loop. Returns the count."""
        waiting = sorted(
            (j for j in self._jobs.values() if j.status == JobStatus.QUEUED),
            key=lambda j: j.created_at,
        )
        resumed = 0
        for job in waiting:
            try:
                self.submit_job(job)
            except QueueFull as exc:
                # Only reachable if the bound was lowered between runs; jobs left queued need manual resume or cancel.
                logger.error(
                    "Could not re-enqueue %d restored job(s) after a restart (%s). They "
                    "remain 'queued' and will not start on their own; resume or cancel "
                    "them, or raise jobs.max_queued_jobs.",
                    len(waiting) - resumed,
                    exc,
                )
                break
            resumed += 1
        if resumed:
            logger.info("Re-enqueued %d run(s) held in the backlog before a restart.", resumed)
        return resumed

    # -- batches -----------------------------------------------------------

    def submit_batch(self, incidents, run_mode="auto", batch_id=None) -> dict:
        """Create and submit one job per incident under a shared batch id; admission is per-job and partial by design."""
        batch = str(batch_id or uuid.uuid4())
        mode = resolve_run_mode(run_mode)
        accepted: List[dict] = []
        refused: List[dict] = []
        for incident in incidents:
            record = dict(incident)
            record["batch_id"] = batch
            job = self.create_job(record, run_mode=mode)
            try:
                started = self.submit_job(job)
            except QueueFull as exc:
                # Drop the job: a job id for a run nobody will start is worse than none.
                self._jobs.pop(job.job_id, None)
                refused.append(
                    {
                        "incident_id": record.get("id"),
                        "reason": str(exc),
                        "depth": exc.depth,
                        "limit": exc.limit,
                    }
                )
                continue
            accepted.append(
                {
                    "job_id": job.job_id,
                    "incident_id": record.get("id"),
                    "status": job.status.value,
                    "queue_position": 0 if started else self.queue.position(job.job_id),
                }
            )
        logger.info(
            "Batch %s: %d run(s) submitted, %d refused, %d already in flight.",
            batch,
            len(accepted),
            len(refused),
            self.queue.stats()["running"],
        )
        return {
            "batch_id": batch,
            "accepted": accepted,
            "refused": refused,
            "queue": self.queue.stats(),
        }

    def batch_jobs(self, batch_id: str) -> List[Job]:
        """Every live job labelled with this batch id, oldest first."""
        target = str(batch_id)
        jobs = [
            job
            for job in self._jobs.values()
            if str(job.incident.get("batch_id") or "") == target
        ]
        jobs.sort(key=lambda j: j.created_at)
        return jobs

    def batch_status(self, batch_id: str) -> dict:
        """Batch progress from its live jobs; done is a floor because _prune drops terminal jobs after TTL. Raises KeyError if none remain."""
        jobs = self.batch_jobs(batch_id)
        if not jobs:
            raise KeyError(batch_id)
        counts: Dict[str, int] = {}
        for job in jobs:
            counts[job.status.value] = counts.get(job.status.value, 0) + 1
        return {
            "batch_id": str(batch_id),
            "total": len(jobs),
            "counts": counts,
            # A batch is a label on its jobs, so its owner is theirs; taken from the first
            # rather than asserted, and empty when they disagree, which nothing creates.
            "owner": owner_of(jobs[0].incident),
            "done": sum(counts.get(s.value, 0) for s in _TERMINAL_STATUSES),
            "jobs": [
                {
                    "job_id": job.job_id,
                    "incident_id": job.incident.get("id"),
                    "status": job.status.value,
                    "current_stage": job.current_stage,
                    "queue_position": self.queue.position(job.job_id),
                    "error": job.error,
                }
                for job in jobs
            ],
        }

    def list_batches(self) -> List[dict]:
        """Every batch with at least one live job, newest first."""
        ids = {
            str(job.incident.get("batch_id"))
            for job in self._jobs.values()
            if job.incident.get("batch_id")
        }
        rows = [self.batch_status(batch_id) for batch_id in ids]
        rows.sort(key=lambda r: r["jobs"][0]["job_id"] if r["jobs"] else "")
        return rows

    def cancel_batch(self, batch_id: str) -> dict:
        """Cancel every non-terminal job in a batch; a job that ends mid-iteration counts as already done. Raises KeyError if none remain."""
        jobs = self.batch_jobs(batch_id)
        if not jobs:
            raise KeyError(batch_id)
        cancelled, already = [], []
        for job in jobs:
            if job.status in _TERMINAL_STATUSES:
                already.append(job.job_id)
                continue
            try:
                self.control(job.job_id, "cancel_all")
                cancelled.append(job.job_id)
            except (KeyError, ValueError) as exc:
                logger.info("Batch %s: %s was not cancelled (%s)", batch_id, job.job_id, exc)
                already.append(job.job_id)
        return {
            "batch_id": str(batch_id),
            "cancelled": cancelled,
            "already_finished": already,
        }

    def _job_row(self, job) -> dict:
        """One compact row for the list endpoints. Also the shape `_prune` keeps."""
        return {
            "job_id": job.job_id,
            "incident_id": job.incident.get("id"),
            "status": job.status.value,
            "run_mode": job.run_mode.value,
            "current_stage": job.current_stage,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            # Stage waiting on a human, so a list view can show "needs you".
            "awaiting_stage": (job.open_gate or {}).get("stage"),
            "batch_id": job.incident.get("batch_id"),
            # Who asked for this run. Carried in the row so the list endpoint can
            # narrow to one caller without re-reading every job.
            "owner": owner_of(job.incident),
            "owner_name": job.incident.get("owner_name"),
            # 1-based place in the backlog; 0 for anything not waiting. Read live here
            # rather than stored on the job, so it cannot go stale as the queue drains.
            "queue_position": self.queue.position(job.job_id),
            # Whether the run is still held in memory. A `false` row answers reads from the
            # store, so an operator can tell "finished a while ago" from "gone".
            "live": True,
        }

    def list_jobs(self) -> List[dict]:
        """Compact list of jobs (newest first) for GET /api/v1/jobs.

        Includes runs the TTL has evicted from memory, which are still in the store: a
        finished investigation vanishing from the list on someone else's submission reads
        as a run that was deleted.
        """
        rows = [self._job_row(job) for job in self._jobs.values()]
        live = {row["job_id"] for row in rows}
        rows.extend(
            row for job_id, row in self._history.items() if job_id not in live
        )
        rows.sort(key=lambda r: r["created_at"] or "", reverse=True)
        return rows

    def list_open_gates(self) -> List[dict]:
        """Every open gate awaiting a human, oldest-first; terminal jobs are excluded regardless of open_gate."""
        rows = [
            {
                "job_id": job.job_id,
                "incident_id": job.incident.get("id"),
                "run_mode": job.run_mode.value,
                # The job's own status, so a client can tell an answerable gate from one
                # on a run that has since ended.
                "status": job.status.value,
                "owner": owner_of(job.incident),
                **job.open_gate,
            }
            for job in self._jobs.values()
            if job.open_gate and job.status not in _TERMINAL_STATUSES
        ]
        rows.sort(key=lambda r: r.get("opened_at") or "")
        return rows

    def _prune(self):
        """Evict terminal in-memory jobs older than the TTL, keeping each one's compact row.

        The TTL bounds how much a long-lived process holds, not how far back the operator
        can see: the row stays listable and the document stays in the store under its own
        retention (`JobStore.prune`), so `_hydrate` can answer a read of either.
        """
        now = time.time()
        stale = [
            jid
            for jid, job in self._jobs.items()
            if job.status in _TERMINAL_STATUSES
            and (now - job.created_ts) > self.ttl_seconds
        ]
        for jid in stale:
            job = self._jobs.pop(jid, None)
            if job is None:
                continue
            row = self._job_row(job)
            row["live"] = False
            row["queue_position"] = 0
            self._history[jid] = row
        # Newest kept: an operator looking back looks back from now.
        while len(self._history) > max(0, self.history_max_items):
            oldest = min(
                self._history,
                key=lambda k: self._history[k].get("created_at") or "",
            )
            self._history.pop(oldest, None)

    # -- events ------------------------------------------------------------

    def _emit(
        self,
        job,
        event_type,
        stage=None,
        status=None,
        message="",
        data=None,
        pass_number=None,
    ):
        """Emit one job event; pass_number goes into data["pass"] so the bare stage name stays unchanged."""
        if pass_number is not None:
            data = dict(data or {})
            data["pass"] = int(pass_number)
        self.emitter.emit(
            job.job_id,
            event_type,
            stage=stage,
            status=status,
            message=message,
            data=data,
        )

    def _log_health(self, job, health):
        """Log score and reasons for a scored stage (debug if healthy, warning if below threshold)."""
        if not health.scored:
            return
        codes = ",".join(health.reason_codes) or "-"
        line = "[health] job=%s stage=%s score=%.2f threshold=%.2f reasons=[%s]"
        args = (job.job_id, health.stage, health.score, health.threshold, codes)
        extra = {
            "event": "stage_health",
            "job_id": job.job_id,
            "stage": health.stage,
            "score": round(health.score, 4),
            "threshold": health.threshold,
            "gate_recommended": health.gate_recommended,
            "reason_codes": health.reason_codes,
        }
        if health.gate_recommended:
            logger.warning(line + " BELOW THRESHOLD", *args, extra=extra)
            for reason in health.reasons:
                logger.warning(
                    "[health]   job=%s %s (-%.2f): %s",
                    job.job_id,
                    reason.code,
                    reason.weight,
                    reason.detail,
                    extra={
                        "event": "stage_health_reason",
                        "job_id": job.job_id,
                        "stage": health.stage,
                        "reason_code": reason.code,
                        "weight": reason.weight,
                    },
                )
        else:
            logger.debug(line, *args, extra=extra)

    async def subscribe(self, job_id):
        """Async generator: replay history then tail live events; queue registered before snapshot to avoid gaps."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        queue: asyncio.Queue = asyncio.Queue()
        job._subscriber_queues.append(queue)
        history_snapshot = list(job._event_history)
        try:
            for event in history_snapshot:
                yield event
            while True:
                event = await queue.get()
                yield event
        finally:
            if queue in job._subscriber_queues:
                job._subscriber_queues.remove(queue)

    # -- the stage-runner loop --------------------------------------------

    async def run_job(self, job: Job):
        await self._run_stages(job, 0)

    async def run_from_stage(self, job: Job, start_index: int):
        await self._run_stages(job, start_index)

    def _stage_for_key(self, key: str) -> StageDescriptor:
        """Stage descriptor for a record key; pass 2 uses the same callable as pass 1, the pass travels on ctx.current_pass. Raises KeyError if unknown."""
        name = split_pass_key(key)[0]
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(key)

    def _stage_record_key(self, job: Job, stage, pass_number=None) -> str:
        """stage_keys entry for a control action; a repeatable stage without pass_number defaults to the current pass."""
        if not stage:
            return ""
        name, embedded = split_pass_key(str(stage))
        if pass_number is None and embedded > 1:
            pass_number = embedded
        if pass_number is None:
            if name in _PASS_STAGES:
                candidate = pass_key(name, job.current_pass)
                if candidate in job.stage_statuses:
                    return candidate
            return name
        return pass_key(name, int(pass_number))

    def _record_pass_asked_nothing(self, job: Job, number: int) -> bool:
        """Record that pass number planned no query; the fetch is skipped and accumulated rows returned unchanged. Never raises."""
        try:
            record = (job.context.pass_outputs or {}).get(int(number)) or {}
            if record.get("queries"):
                return False
            notes = [str(n) for n in (record.get("notes") or []) if str(n).strip()]
            spec = follow_up_spec(job.context, int(number)) or {}
            targets = pass_targets(spec) or "no target"
            detail = (
                f"pass {number} on '{targets}' planned no query, so its fetch was skipped: "
                + ("; ".join(notes) if notes else "no reason was recorded")
            )
            logger.warning(
                "Retrieval pass %s planned no query (%s); the investigation proceeds on the "
                "evidence the earlier pass(es) returned.",
                number,
                "; ".join(notes) if notes else "no reason was recorded",
            )
            job.record_intervention(
                "retrieval_pass_asked_nothing",
                stage=_PASS_STAGES[0],
                detail=detail,
                actor="engine",
            )
            self._emit(
                job,
                "pass_skipped",
                stage=_PASS_STAGES[0],
                status=StageStatus.COMPLETED.value,
                message=(
                    f"Retrieval pass {number} on '{targets}' asked nothing — "
                    + (notes[0] if notes else "no reason was recorded")
                ),
                data={"notes": notes, "purpose": pass_purpose(spec)},
                pass_number=int(number),
            )
            job.touch()
            self._persist(job)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not record that retrieval pass %s planned no query (%s).",
                number,
                exc,
            )
            return False

    def _record_automatic_escalations(self, job: Job, result: Any) -> int:
        """Record one durable entry per automatically escalated link (trigger is probe attempt, not success). Never raises."""
        recorded = 0
        try:
            links = getattr(result, "links", None)
            if not isinstance(links, list):
                return 0
            for finding in links:
                mode = str(getattr(finding, "mode", "") or "")
                source = str(getattr(finding, "probe_source", "") or "")
                if mode not in ESCALATING_MODES or not source:
                    continue
                target = str(getattr(finding, "target_use_case", "") or "?")
                spent = bool(getattr(finding, "probe_spent", False))
                whose = str(getattr(finding, "mode_source", "") or "default")
                note = str(getattr(finding, "probe_note", "") or "no note was recorded")
                detail = (
                    f"escalation mode '{mode}' (set by {whose}) spent "
                    f"{'one probe' if spent else 'no probe'} of '{source}' on behalf of "
                    f"'{target}': {note} — this run's verdict, its severity and its stage "
                    "health were not read from it and did not move"
                )
                job.record_intervention(
                    "link_escalated",
                    stage="correlation",
                    detail=detail,
                    actor="engine",
                )
                recorded += 1
            if recorded:
                job.touch()
                self._persist(job)
                logger.info(
                    "Recorded %d automatic link escalation(s) for job %s; the advisory lane acted "
                    "and the verdict did not read any of it.",
                    recorded,
                    job.job_id,
                )
            return recorded
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not record the automatic link escalations for job %s (%s).",
                job.job_id,
                exc,
            )
            return recorded

    def _record_inquiry_probes(self, job: Job, result: Any) -> int:
        """Record one durable entry per open-question probe this run spent. Never raises.

        Same reason as the link escalations above: the engine spent a retrieval nobody clicked
        for, and a spend with no record is a spend nobody can audit. Only the questions that cost
        a probe are recorded — a refusal is on the finding and costs nothing.
        """
        recorded = 0
        try:
            inquiries = getattr(result, "inquiries", None)
            if not isinstance(inquiries, list):
                return 0
            for finding in inquiries:
                if not bool(getattr(finding, "probe_spent", False)):
                    continue
                note = str(getattr(finding, "probe_note", "") or "no note was recorded")
                detail = (
                    f"open question '{getattr(finding, 'id', '') or '?'}' spent one probe of "
                    f"'{getattr(finding, 'source', '') or '?'}' and settled as "
                    f"'{getattr(finding, 'state', '') or '?'}': {note} — this run's verdict, its "
                    "severity and its stage health were not read from it and did not move"
                )
                job.record_intervention(
                    "inquiry_probed",
                    stage="correlation",
                    detail=detail,
                    actor="engine",
                )
                recorded += 1
            if recorded:
                job.touch()
                self._persist(job)
                logger.info(
                    "Recorded %d open-question probe(s) for job %s; the advisory lane asked and "
                    "the verdict did not read any of it.",
                    recorded,
                    job.job_id,
                )
            return recorded
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not record the open-question probes for job %s (%s).",
                job.job_id,
                exc,
            )
            return recorded

    def _child_slot(self, size: int) -> asyncio.Semaphore:
        """Concurrency bound for child runs; rebuilt when size changes so a live config update takes effect."""
        if self._child_semaphore is None or self._child_semaphore_size != size:
            self._child_semaphore = asyncio.Semaphore(max(1, size))
            self._child_semaphore_size = size
        return self._child_semaphore

    def _pack(self) -> Any:
        """The loaded knowledge pack from whichever module holds it; None if none available."""
        for key in ("api_call", "correlation", "log_retrieval"):
            pack = getattr((self.modules or {}).get(key), "knowledge_pack", None)
            if pack is not None:
                return pack
        return None

    def _parent_pivots(self, result: Any) -> List[str]:
        """Subject values adjudicated by this run, read from the verdict (not the incident) so scope-widened subjects are included."""
        verdict = getattr(result, "verdict", None)
        subjects = getattr(verdict, "subjects", None)
        if not isinstance(subjects, list):
            return []
        values = []
        for subject in subjects:
            value = str(getattr(subject, "subject", "") or "").strip()
            if value and value not in values:
                values.append(value)
        return values

    async def _spawn_link_children(self, job: Job, result: Any) -> int:
        """Launch rung-4 child runs and stamp each candidate; returns stamped count (launches + refusals). Never raises."""
        launched = 0
        stamped = 0
        try:
            links = getattr(result, "links", None)
            if not isinstance(links, list) or not links:
                return 0
            correlation = (self.config or {}).get("correlation") or {}
            # The parent's procedure is read from the same resolution the verdict took, so
            # the cycle guard is keyed on the procedure the run actually adjudicated under.
            spawns, refusals = plan_child_spawns(
                links,
                incident=job.incident,
                config=correlation if isinstance(correlation, dict) else {},
                pack=self._pack(),
                parent_use_case=_adjudicating_ruleset_key(job.context)[0],
                parent_pivots=self._parent_pivots(result),
                parent_job=job.job_id,
                spawned_total=self._link_children_total,
                llm_concurrency=self._llm_concurrency(),
            )
            for refusal in refusals:
                # The total_budget refusal is run-scoped, not candidate-scoped; log it so
                # operators can distinguish "nothing to refer" from "budget exhausted".
                logger.info(
                    "Rung-4 child run refused for %r (%s): %s",
                    refusal.get("target_use_case"),
                    refusal.get("code"),
                    refusal.get("note"),
                )
                if refusal.get("code") == "total_budget":
                    job.record_intervention(
                        "link_child_refused",
                        stage="correlation",
                        detail=(
                            f"no further child run was spawned: {refusal.get('note')} — "
                            f"{self._link_children_total} child run(s) have been launched by "
                            "this process, and the candidates left keep their referrals"
                        ),
                        actor="engine",
                    )
                # Per-candidate codes are recoverable from no other field; an empty child_note reads as never considered.
                if refusal.get("scope") == "candidate":
                    candidate = refusal.get("finding")
                    try:
                        candidate.child_note = str(refusal.get("note") or "")
                        stamped += 1
                    except Exception:  # noqa: BLE001
                        pass
            if not spawns:
                if refusals:
                    job.touch()
                    self._persist(job)
                return stamped

            budget = child_budget(
                correlation if isinstance(correlation, dict) else {},
                self._llm_concurrency(),
            )
            slot = self._child_slot(budget["max_concurrent"])
            for spawn in spawns:
                finding = spawn.get("finding")
                composed = None
                try:
                    composed = compose_referral(
                        finding,
                        parent_job_id=job.job_id,
                        parent_incident_id=str(
                            (job.incident or {}).get("id", "") or ""
                        ),
                        window=self._referral_window(job, finding),
                        mode=job.run_mode.value,
                    )
                except (
                    Exception
                ) as exc:  # noqa: BLE001
                    logger.info(
                        "Could not compose the referral text for %r (%s); the child is launched "
                        "with its pin and its pivot, which are what decide its procedure and its "
                        "scope.",
                        spawn.get("target_use_case"),
                        type(exc).__name__,
                    )
                incident = child_incident(spawn, job.incident, composed)
                incident.setdefault("id", f"{job.job_id}-link-{launched + 1}")
                incident.setdefault("source", "link_referral")
                # The child inherits the parent's run mode so a supervised deployment keeps
                # gates on a run nobody explicitly asked for.
                child = self.create_job(incident, run_mode=job.run_mode)
                self._link_children_total += 1
                launched += 1
                asyncio.ensure_future(self._run_child(child, slot))
                job.record_intervention(
                    "link_child_spawned",
                    stage="correlation",
                    detail=(
                        f"a full run of '{spawn.get('target_use_case')}' was launched as job "
                        f"{child.job_id}, pinned to that procedure and scoped to "
                        f"{spawn.get('pivot_entity') or 'subject'} "
                        f"{spawn.get('pivot_value')} at chain depth {spawn.get('depth')} "
                        "— this run's verdict, its severity and its stage health were settled "
                        "before it started and are not read from it"
                    ),
                    actor="engine",
                )
                try:
                    finding.child_job_id = child.job_id
                    finding.child_note = (
                        f"a full run of this procedure was launched as job {child.job_id}, "
                        f"pinned and scoped to {spawn.get('pivot_value')}"
                    )
                    stamped += 1
                except Exception:  # noqa: BLE001
                    pass
            job.touch()
            self._persist(job)
            logger.info(
                "Rung-4: %d child run(s) launched from job %s (%d this process, bound %d "
                "concurrent); the parent's verdict was already settled.",
                launched,
                job.job_id,
                self._link_children_total,
                budget["max_concurrent"],
            )
            return stamped
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not spawn the rung-4 child runs for job %s (%s).",
                job.job_id,
                exc,
            )
            # Return stamped (not 0): the failure may have arrived after some rows were
            # already written, and the caller needs to know result may have changed.
            return stamped

    async def _run_child(self, child: Job, slot: asyncio.Semaphore) -> None:
        """Run one child under the concurrency bound, held for the whole run. Never raises."""
        async with slot:
            try:
                await self.run_job(child)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Referral child job %s ended in an error (%s).",
                    child.job_id,
                    type(exc).__name__,
                )

    def _llm_concurrency(self) -> Optional[int]:
        """The LLM client's max_concurrency, or None; read from the client rather than config so tests work without one."""
        client = (self.modules or {}).get("llm_client")
        value = getattr(client, "_max_concurrency", None)
        return value if isinstance(value, int) and value > 0 else None

    def _referral_window(self, job: Job, finding: Any) -> Tuple[str, str]:
        """Date window for the child run from the generator; ("", "") when anything is missing."""
        try:
            generator = (self.modules or {}).get("api_call")
            understanding = job.context.outputs.get("understanding")
            analysis = getattr(understanding, "analysis", None)
            if generator is None or analysis is None:
                return ("", "")
            queries = list(job.context.outputs.get("queries") or [])
            date_from, date_to, _resolved = generator.referral_window(
                analysis, queries, str(getattr(finding, "window_hint", "") or "")
            )
            return (str(date_from or ""), str(date_to or ""))
        except Exception:  # noqa: BLE001
            return ("", "")

    def _maybe_open_next_pass(self, job: Job, number: int) -> bool:
        """Register a follow-up pass if the pack declares one and the cap allows; hitting the cap is recorded as an intervention."""
        nxt = number + 1
        spec = follow_up_spec(job.context, nxt)
        if not spec:
            return False
        cap = max_retrieval_passes(self.config)
        if nxt > cap:
            logger.warning(
                "A retrieval pass %s is declared (source(s) '%s': %s) but jobs."
                "max_retrieval_passes is %s, so it was NOT run — the investigation "
                "proceeds on the evidence the first %s pass(es) returned.",
                nxt,
                pass_targets(spec),
                pass_purpose(spec) or "no purpose declared",
                cap,
                cap,
            )
            job.record_intervention(
                "retrieval_pass_capped",
                stage=_LAST_PASS_STAGE,
                detail=(
                    f"pass {nxt} on '{pass_targets(spec)}' was declared but not run "
                    f"(jobs.max_retrieval_passes={cap})"
                ),
                actor="config",
            )
            return False
        added = job.register_pass(nxt)
        logger.info(
            "Retrieval pass %s is due (source(s) '%s': %s); re-planning and re-fetching.",
            nxt,
            pass_targets(spec),
            pass_purpose(spec) or "no purpose declared",
        )
        self._emit(
            job,
            "pass_started",
            stage=_PASS_STAGES[0],
            status=StageStatus.PENDING.value,
            message=(
                f"Retrieval pass {nxt} on '{pass_targets(spec)}' "
                "— planning from what the earlier pass returned"
            ),
            data={"stages": list(added), "purpose": pass_purpose(spec)},
            pass_number=nxt,
        )
        job.touch()
        self._persist(job)
        return bool(added)

    async def _run_stages(self, job: Job, start_index: int):
        # Stale loops stand down when _run_gen advances past their generation.
        generation = job._run_gen
        job._loop_live = True
        job.status = JobStatus.RUNNING
        job.touch()
        self._emit(job, "job_status", status=job.status.value, message="Job running")
        # Whose personal credentials this run uses, taken from the run's OWNER and not from
        # whoever started this loop: a job submitted by one caller and resumed, retried or
        # restored by an administrator is still the submitter's run, and it must not silently
        # start using the administrator's token half way through.
        # An unowned run — every run of a deployment that resolves no identity — keeps whatever
        # the caller who started this loop was bound to, which there is the one operator.
        segment_token = set_current_segment(
            owner_of(job.incident) or current_segment()
        )
        try:
            index = start_index
            # while, not range: a follow-up pass registers its own keys into job.stage_keys
            # from inside this loop, so the bound must be re-read each iteration.
            while index < len(job.stage_keys):
                key = job.stage_keys[index]
                stage = self._stage_for_key(key)
                number = split_pass_key(key)[1]
                # Current pass for stage callables and the snapshot. Set before the
                # pause/step barriers so a parked job reports the pass it is about to run.
                job.context.current_pass = number
                index_after = index + 1
                if job._run_gen != generation:
                    return
                job.current_stage = stage.name

                # Pause boundary: blocks here while paused, resumes when set.
                if not job._pause_event.is_set():
                    job.status = JobStatus.PAUSED
                    job._stopped_index = index
                    job.touch()
                    # Named: the stopped position distinguishes a pause from a hung stage.
                    self._emit(
                        job,
                        "job_status",
                        stage=stage.name,
                        status=job.status.value,
                        message=f"Paused before {stage.name}",
                        pass_number=number,
                    )
                    self._release_to_a_human(job, "paused")
                await job._pause_event.wait()

                # Step mode: advance one stage per step action. Job reads `paused` while
                # waiting, since nothing is running.
                if job.run_mode == JobRunMode.STEP and not job._step_event.is_set():
                    job.status = JobStatus.PAUSED
                    job._stopped_index = index
                    job.touch()
                    self._emit(
                        job,
                        "job_status",
                        status=job.status.value,
                        message=f"Awaiting step into {stage.name}",
                    )
                    self._release_to_a_human(job, "awaiting a step")
                if job.run_mode == JobRunMode.STEP:
                    await job._step_event.wait()
                    job._step_event.clear()
                    # A cancel during the step-wait unblocks and terminates.

                if job._run_gen != generation:
                    return
                if job._cancel_all_flag or job._cancel_stage_flag:
                    self._mark_cancelled(job, stage, index, number)
                    return

                if job.status != JobStatus.RUNNING:
                    job.status = JobStatus.RUNNING
                    job.touch()
                    self._emit(
                        job,
                        "job_status",
                        status=job.status.value,
                        message="Job running",
                    )

                job.stage_statuses[key] = StageStatus.RUNNING
                job._stopped_index = index
                job.touch()
                stage_started_at = time.time()
                job.stage_started_at[key] = stage_started_at
                self._emit(
                    job,
                    "stage_started",
                    stage=stage.name,
                    status=StageStatus.RUNNING.value,
                    message=f"Stage {stage.name} started",
                    data={"started_at": stage_started_at},
                    pass_number=number,
                )

                try:
                    if stage.is_synchronous:
                        coro = asyncio.to_thread(stage.run, job.context)
                    else:
                        coro = stage.run(job.context)
                    job._cancel_event.clear()
                    job._current_task = asyncio.ensure_future(coro)
                    # Race the stage against a cancel signal. The cancel event lets
                    # the job terminate even if the stage task does not unwind promptly.
                    cancel_wait = asyncio.ensure_future(job._cancel_event.wait())
                    try:
                        done, _ = await asyncio.wait(
                            {job._current_task, cancel_wait},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        # An outer cancel abandons the waiter if not cleaned up here.
                        # The stage task is left alone (cancel-as-a-race).
                        if not cancel_wait.done():
                            cancel_wait.cancel()
                    if job._current_task in done:
                        result = job._current_task.result()
                    else:
                        # Cancel fired first: try to unwind the task, then terminate.
                        job._current_task.cancel()
                        self._mark_cancelled(job, stage, index, number)
                        return
                except asyncio.CancelledError:
                    # A job-level cancel marks the job cancelled; an outer cancel (shutdown)
                    # re-raises.
                    if job._cancel_all_flag or job._cancel_stage_flag:
                        self._mark_cancelled(job, stage, index, number)
                        return
                    raise
                except Exception as exc:  # noqa: BLE001
                    if stage.is_best_effort:
                        job.stage_statuses[key] = StageStatus.COMPLETED
                        job.touch()
                        logger.warning(
                            "Best-effort stage %s failed for job %s: %s",
                            stage.name,
                            job.job_id,
                            exc,
                        )
                        self._emit(
                            job,
                            "stage_completed",
                            stage=stage.name,
                            status=StageStatus.COMPLETED.value,
                            message=f"{stage.name} skipped (best-effort): {exc}",
                            pass_number=number,
                        )
                        index = index_after
                        continue
                    job.stage_statuses[key] = StageStatus.FAILED
                    job.status = JobStatus.STAGE_FAILED
                    job.error = str(exc)
                    job._failed_index = index
                    job.stage_durations[key] = _elapsed_ms(stage_started_at)
                    job.touch()
                    logger.error(
                        "Stage %s failed for job %s: %s", stage.name, job.job_id, exc
                    )
                    self._emit(
                        job,
                        "stage_failed",
                        stage=stage.name,
                        status=StageStatus.FAILED.value,
                        message=str(exc),
                        data={"duration_ms": _elapsed_ms(stage_started_at)},
                        pass_number=number,
                    )
                    self._emit(
                        job,
                        "job_status",
                        status=job.status.value,
                        message="Stage failed; awaiting retry or cancel",
                    )
                    self._persist(job)
                    return
                finally:
                    job._current_task = None

                if stage.output_key:
                    job.context.outputs[stage.output_key] = result
                job.stage_statuses[key] = StageStatus.COMPLETED
                # Mirror timing + summary onto the job for polling clients.
                duration_ms = _elapsed_ms(stage_started_at)
                summary = summarize_stage(stage.name, result)
                health = score_stage(stage.name, result, job.context, self.config)
                job.stage_durations[key] = duration_ms
                job.stage_summaries[key] = summary
                job.stage_health[key] = health.to_dict()
                job.touch()
                self._log_health(job, health)
                # Persist per stage; a crash costs at most one stage of work.
                self._persist(job, evidence_changed=stage.output_key in EVIDENCE_KEYS)
                # Emit stage output before the completion badge for the UI detail panel.
                self._emit(
                    job,
                    "stage_output",
                    stage=stage.name,
                    status=StageStatus.COMPLETED.value,
                    message=f"{stage.name} output",
                    data={"summary": summary, "health": health.to_dict()},
                    pass_number=number,
                )
                self._emit(
                    job,
                    "stage_completed",
                    stage=stage.name,
                    status=StageStatus.COMPLETED.value,
                    message=f"Stage {stage.name} completed",
                    data={"duration_ms": duration_ms},
                    pass_number=number,
                )

                # Check for a further pass after the fetch settles and before the gate,
                # so the reviewer sees what is queued and approval continues into it.
                if stage.name == _LAST_PASS_STAGE:
                    self._maybe_open_next_pass(job, number)

                # A follow-up pass that planned nothing: fetch is skipped and rows are
                # returned unchanged.
                if stage.name == _PASS_STAGES[0] and number > 1:
                    self._record_pass_asked_nothing(job, number)

                # Advisory lane record: before the gate so the reviewer sees what was spent.
                if stage.name == "correlation":
                    self._record_automatic_escalations(job, result)
                    self._record_inquiry_probes(job, result)

                # Approval gate: after the stage produced output, so the human reviews
                # the result (unlike the step wait, which grants permission to start).
                if self._gate_applies(job, stage, health):
                    next_index = await self._await_gate(
                        job, stage, health, summary, number
                    )
                    if next_index is None:
                        return
                    if next_index != index_after:
                        # A reject re-runs a stage; hand off to a fresh loop so the
                        # re-run gets normal stage treatment (timing, summary, health, gate).
                        await self._run_stages(job, next_index)
                        return

                # Rung 4 after the gate: a rejected gate re-runs correlation, so spawning first
                # launches a child off output the human is about to reject.
                if stage.name == "correlation":
                    if await self._spawn_link_children(job, result):
                        # Re-summarise: the summary was built before child_job_id was stamped, so
                        # links had no child to point at. Re-emit so a replay overwrites the
                        # earlier copy. Health is not re-scored, having settled before the gate.
                        summary = summarize_stage(stage.name, result)
                        job.stage_summaries[key] = summary
                        job.touch()
                        self._persist(job)
                        self._emit(
                            job,
                            "stage_output",
                            stage=stage.name,
                            status=StageStatus.COMPLETED.value,
                            message=f"{stage.name} output (child run(s) launched)",
                            data={"summary": summary, "health": health.to_dict()},
                            pass_number=number,
                        )
                index = index_after

            job.status = JobStatus.COMPLETED
            job.current_stage = None
            job._stopped_index = None
            job.touch()
            self._emit(
                job, "job_status", status=job.status.value, message="Job completed"
            )
            self._persist(job)
        except asyncio.CancelledError:
            # A shutdown at a gate preserves the pending approval; only a flagged cancel really cancels.
            shutting_down_at_a_gate = (
                job.status == JobStatus.AWAITING_APPROVAL
                and not job._cancel_all_flag
                and not job._cancel_stage_flag
            )
            if shutting_down_at_a_gate:
                logger.warning(
                    "Job %s was awaiting approval on '%s' when the process stopped; "
                    "preserved as pending so the decision is not lost.",
                    job.job_id,
                    (job.open_gate or {}).get("stage"),
                )
                job.touch()
                self._persist(job)
                raise
            job.status = JobStatus.CANCELLED
            job.touch()
            self._emit(
                job, "job_status", status=job.status.value, message="Job cancelled"
            )
            self._persist(job)
            raise
        finally:
            reset_current_segment(segment_token)
            # A stale loop must not hand away the slot the newer loop is running on.
            if job._run_gen == generation:
                job._loop_live = False
                self._release_slot(job.job_id)

    # -- approval gates ----------------------------------------------------

    def _gate_applies(self, job: Job, stage: StageDescriptor, health) -> bool:
        """Whether a human gate applies in this mode at this health; semi_auto skips unscored stages."""
        if job.run_mode not in _GATING_MODES:
            return False
        if not stage_gate_enabled(self.config, stage.name):
            return False
        if job.run_mode == JobRunMode.SUPERVISED:
            return True
        if not getattr(health, "scored", False):
            return False
        return bool(getattr(health, "gate_recommended", False))

    def _gate_record(
        self, job: Job, stage: StageDescriptor, health, summary, pass_number=1
    ) -> dict:
        """Build the gate payload; pass is a separate field so reviewers see the pass number."""
        health_dict = health.to_dict() if hasattr(health, "to_dict") else {}
        timeout = stage_gate_timeout(self.config, stage.name)
        return {
            "stage": stage.name,
            "pass": int(pass_number or 1),
            "reason": (
                "supervised mode"
                if job.run_mode == JobRunMode.SUPERVISED
                else "health below threshold"
            ),
            "health": health_dict,
            "summary": summary,
            "actions": list(GATE_ACTIONS),
            "opened_at": datetime.now(timezone.utc).isoformat(),
            # None means the gate waits indefinitely.
            "timeout_seconds": timeout,
            "on_timeout": (
                stage_gate_on_timeout(self.config, stage.name) if timeout else None
            ),
        }

    async def _await_gate(
        self, job: Job, stage: StageDescriptor, health, summary, pass_number=1
    ):
        """Block until a human resolves the gate; returns the next stage index, or None if cancelled."""
        job.open_gate = self._gate_record(job, stage, health, summary, pass_number)
        job._gate_decision = None
        job._gate_event.clear()
        job.status = JobStatus.AWAITING_APPROVAL
        job.touch()
        logger.info(
            "[gate] job=%s stage=%s OPEN (%s, score=%.2f) — awaiting approve/reject/override",
            job.job_id,
            stage.name,
            job.open_gate["reason"],
            float(job.open_gate["health"].get("score", 0.0) or 0.0),
            extra={
                "event": "gate_opened",
                "job_id": job.job_id,
                "stage": stage.name,
                "reason": job.open_gate["reason"],
                "run_mode": job.run_mode.value,
                "score": job.open_gate["health"].get("score"),
                "timeout_seconds": job.open_gate.get("timeout_seconds"),
            },
        )
        self._emit(
            job,
            "gate_opened",
            stage=stage.name,
            status=job.status.value,
            message=f"{stage.name} awaiting approval ({job.open_gate['reason']})",
            data=dict(job.open_gate),
        )
        # Durable save before waiting: the job may wait days for a human and a restart
        # must come back to this exact state.
        await self._persist_durably(job)
        self._release_to_a_human(job, "awaiting a gate decision")
        return await self._wait_for_gate_decision(job, stage, pass_number)

    def _gate_wait_timeout(self, job: Job, stage: StageDescriptor, already_fired: bool):
        """Timeout to pass to asyncio.wait; None after first fire so a hold outcome waits indefinitely."""
        if already_fired:
            return None
        return stage_gate_timeout(self.config, stage.name)

    def _on_gate_timeout(self, job: Job, stage: StageDescriptor) -> str:
        """Record and announce a gate timeout; returns the resolved action (hold/proceed/abort)."""
        action = stage_gate_on_timeout(self.config, stage.name)
        waited = stage_gate_timeout(self.config, stage.name)
        gate = job.open_gate or {}
        logger.warning(
            "[gate] job=%s stage=%s TIMED OUT after %ss; on_timeout=%s",
            job.job_id,
            stage.name,
            waited,
            action,
            extra={
                "event": "gate_timeout",
                "job_id": job.job_id,
                "stage": stage.name,
                "waited_seconds": waited,
                "on_timeout": action,
            },
        )
        record = {
            "stage": stage.name,
            "action": action,
            "actor": "timeout",
            "waited_seconds": waited,
            "opened_at": gate.get("opened_at"),
            "timed_out_at": datetime.now(timezone.utc).isoformat(),
        }
        self._emit(
            job,
            "gate_timeout",
            stage=stage.name,
            status=action,
            message=(
                f"{stage.name} gate unanswered after {waited}s; on_timeout={action}"
            ),
            data=record,
        )
        if action != "hold":
            job.record_intervention(
                f"gate_timeout_{action}",
                stage=stage.name,
                detail=f"no human decision within {waited}s; {action}",
                actor="timeout",
            )
            job.gate_history.append(record)
        job.touch()
        self._persist(job)
        return action

    async def _wait_for_gate_decision(
        self, job: Job, stage: StageDescriptor, pass_number=1
    ):
        """Wait on an open gate and return the next stage index; shared by _await_gate and _rearm_one. None means cancelled."""
        key = pass_key(stage.name, pass_number)
        index = (
            job.stage_keys.index(key)
            if key in job.stage_keys
            else self.stage_names.index(stage.name)
        )
        timed_out = False
        while True:
            gate_wait = asyncio.ensure_future(job._gate_event.wait())
            cancel_wait = asyncio.ensure_future(job._cancel_event.wait())
            try:
                done, _ = await asyncio.wait(
                    {gate_wait, cancel_wait},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=self._gate_wait_timeout(job, stage, timed_out),
                )
            finally:
                # Cancel inner futures on outer cancel to avoid "Task destroyed but pending".
                for fut in (gate_wait, cancel_wait):
                    if not fut.done():
                        fut.cancel()
            if done:
                break
            # Neither future completed => the configured timeout expired.
            timed_out = True
            outcome = self._on_gate_timeout(job, stage)
            if outcome == "hold":
                # hold notified; now wait indefinitely (see _gate_wait_timeout).
                continue
            if outcome == "abort":
                job.open_gate = None
                # Set the same flag an operator cancel sets for consistent cancel semantics.
                job._cancel_all_flag = True
                job._cancel_event.set()
                self._mark_cancelled(job, stage, pass_number=pass_number)
                return None
            # proceed: continue as if approved; the gate was not reviewed by a human.
            job.open_gate = None
            job._gate_decision = None
            job.status = JobStatus.RUNNING
            job.touch()
            self._emit(
                job, "job_status", status=job.status.value, message="Job running"
            )
            self._persist(job)
            return index + 1

        if gate_wait not in done:
            job.open_gate = None
            self._mark_cancelled(job, stage, pass_number=pass_number)
            return None

        decision = job._gate_decision or {"action": "approve"}
        job.open_gate = None
        job._gate_decision = None
        # A cancel can also arrive between the gate being resolved and this line.
        if job._cancel_all_flag or job._cancel_stage_flag:
            self._mark_cancelled(job, stage, pass_number=pass_number)
            return None
        if decision.get("action") == "abandon":
            # A control action restarted the pipeline while this gate was open; stand
            # down silently to avoid two loops racing over one JobContext.
            logger.info(
                "[gate] job=%s stage=%s gate abandoned; a control action took over",
                job.job_id,
                stage.name,
            )
            return None
        job.status = JobStatus.RUNNING
        job.touch()
        self._emit(job, "job_status", status=job.status.value, message="Job running")
        restart_index = decision.get("restart_index")
        return index + 1 if restart_index is None else restart_index

    def resolve_gate(
        self,
        job_id,
        action,
        guidance=None,
        reason_code=None,
        restart_from=None,
        value=None,
        actor=None,
        note=None,
        restart_pass=None,
    ) -> Job:
        """Resolve the open gate; restart_pass selects which pass to re-run, defaulting to the gate's pass. Raises KeyError or ValueError."""
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        gate = job.open_gate
        if not gate:
            raise ValueError("No gate is open for this job")
        if action not in GATE_ACTIONS:
            raise ValueError(
                f"Unknown gate action '{action}'; expected one of {', '.join(GATE_ACTIONS)}"
            )
        stage_name = gate["stage"]
        gate_pass = int(gate.get("pass") or 1)
        gate_key = self._stage_record_key(job, stage_name, gate_pass)
        restart_index = None
        restart_key = None

        if action == "reject":
            text = str(guidance or "").strip()
            if not text:
                # A reject with no guidance re-runs an identical prompt, which the operator
                # cannot distinguish from the rejection being ignored.
                raise ValueError("reject requires 'guidance' describing what to fix")
            target = str(restart_from or stage_name)
            wanted = restart_pass if restart_pass is not None else gate_pass
            restart_key = self._stage_record_key(job, target, wanted)
            if restart_key not in job.stage_statuses:
                # Fall back to the stage's first pass: the run may have only one record.
                restart_key = self._stage_record_key(job, target, 1)
            if restart_key not in job.stage_statuses:
                raise ValueError(f"Unknown stage '{target}' for restart_from")
            restart_index = job.stage_keys.index(restart_key)
            if restart_index > job.stage_keys.index(gate_key):
                raise ValueError(
                    f"restart_from '{target}' runs after the gated stage "
                    f"'{stage_name}'; it cannot be re-run to fix it"
                )
            # Guidance is keyed by pass_key so a correction for pass 2 does not re-steer pass 1.
            job.context.stage_guidance.setdefault(restart_key, []).append(text)
            # Everything from the restart point onward is about to be recomputed.
            for name in job.stage_keys[restart_index:]:
                job.stage_statuses[name] = StageStatus.PENDING
            job.error = None
            job._failed_index = None
        elif action == "override":
            # Delegates to set_stage_output for codec, audit trail, and re-scoring.
            self.set_stage_output(
                job_id, stage_name, value, actor=actor, pass_number=gate_pass
            )

        detail = note or guidance or ""
        if reason_code:
            detail = f"[{reason_code}] {detail}".strip()
        job.record_intervention(
            f"gate_{action}", stage=stage_name, detail=detail, actor=actor
        )
        resolution = {
            "stage": stage_name,
            "pass": gate_pass,
            "action": action,
            "actor": actor,
            "reason_code": reason_code,
            "guidance": str(guidance or "") or None,
            "restart_from": (
                split_pass_key(restart_key)[0] if restart_key is not None else None
            ),
            "restart_pass": (
                split_pass_key(restart_key)[1] if restart_key is not None else None
            ),
            "opened_at": gate.get("opened_at"),
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }
        job.gate_history.append(resolution)
        logger.info(
            "[gate] job=%s stage=%s RESOLVED action=%s actor=%s%s",
            job.job_id,
            stage_name,
            action,
            actor or "-",
            (
                f" restart_from={resolution['restart_from']}"
                if resolution["restart_from"]
                else ""
            ),
            extra={
                "event": "gate_resolved",
                "job_id": job.job_id,
                "stage": stage_name,
                "action": action,
                "actor": actor,
                "reason_code": reason_code,
                "restart_from": resolution["restart_from"],
            },
        )
        self._emit(
            job,
            "gate_resolved",
            stage=stage_name,
            status=action,
            message=f"{stage_name} gate resolved: {action}",
            data=resolution,
        )
        # Persist before waking the runner; if the process dies the restored job shows
        # the decision already taken.
        job.touch()
        self._persist(job, evidence_changed=action == "override")
        # Set the decision last: the runner reads it the moment the event fires.
        job._gate_decision = {"action": action, "restart_index": restart_index}
        job._gate_event.set()
        return job

    def _mark_cancelled(
        self, job: Job, stage: StageDescriptor, index=None, pass_number=None
    ):
        """Record a cancelled stage; cancel_stage leaves the job paused, cancel_all marks it terminal. index and pass_number are optional."""
        key = self._stage_record_key(job, stage.name, pass_number)
        if index is None:
            index = job.stage_keys.index(key) if key in job.stage_keys else None
        # A cancel at a gate arrives after the stage completed; overwriting `completed`
        # with `cancelled` would delete a real result and re-run it on next resume.
        already_done = job.stage_statuses.get(key) == StageStatus.COMPLETED
        if not already_done:
            job.stage_statuses[key] = StageStatus.CANCELLED
            job.stage_started_at.pop(key, None)
        job._stopped_index = index if not already_done else index + 1
        stage_only = job._cancel_stage_flag and not job._cancel_all_flag
        job.touch()
        if not already_done:
            self._emit(
                job,
                "stage_completed",
                stage=stage.name,
                status=StageStatus.CANCELLED.value,
                message="Stage cancelled",
                pass_number=split_pass_key(key)[1],
            )
        if stage_only:
            # Consume the flag here: leaving it set would cancel the next stage the
            # operator starts.
            job._cancel_stage_flag = False
            job._cancel_event.clear()
            job.status = JobStatus.PAUSED
            job._pause_event.clear()
            job.touch()
            self._emit(
                job,
                "job_status",
                status=job.status.value,
                message=f"Stage {stage.name} cancelled; run paused",
            )
            self._persist(job)
            return
        job.status = JobStatus.CANCELLED
        job.current_stage = None
        # A cancelled run has no gate to answer; leaving open_gate set keeps the job in
        # the approvals inbox after it has ended.
        job.open_gate = None
        job.pending_gate = None
        job.touch()
        self._emit(job, "job_status", status=job.status.value, message="Job cancelled")
        self._persist(job)

    # -- control -----------------------------------------------------------

    def _abandon_open_gate(self, job: Job, reason):
        """Signal the open gate to abandon so a restart action can own the pipeline alone."""
        if not job.open_gate:
            return
        stage_name = job.open_gate["stage"]
        job.record_intervention(
            "gate_abandoned",
            stage=stage_name,
            detail=f"gate discarded: {reason}",
        )
        self._emit(
            job,
            "gate_resolved",
            stage=stage_name,
            status="abandoned",
            message=f"{stage_name} gate discarded: {reason}",
            data={"stage": stage_name, "action": "abandoned", "reason": reason},
        )
        job.open_gate = None
        job._gate_decision = {"action": "abandon", "restart_index": None}
        job._gate_event.set()

    def _start_loop(self, job: Job, index: int):
        """Launch a fresh stage loop from index; bumps _run_gen so stale loops stand down."""
        job._run_gen += 1
        job._stopped_index = index
        asyncio.ensure_future(self.run_from_stage(job, index))

    def _reset_event_history(self, job: Job, reason):
        """Clear the replay buffer and emit run_reset so attached clients discard stale events."""
        job._event_history.clear()
        self._emit(
            job,
            "run_reset",
            status=job.status.value,
            message=f"Run state cleared ({reason})",
            data={"reason": reason},
        )

    def _resume_index(self, job: Job) -> int:
        """Index to restart from: the stopped stage if incomplete, else first uncompleted, else 0."""
        index = job._stopped_index
        if index is not None and 0 <= index < len(job.stage_keys):
            if job.stage_statuses.get(job.stage_keys[index]) != StageStatus.COMPLETED:
                return index
        for position, name in enumerate(job.stage_keys):
            if job.stage_statuses.get(name) != StageStatus.COMPLETED:
                return position
        return 0

    def _retry_index(self, job: Job, stage=None, pass_number=None) -> int:
        """Stage index to re-run; explicit stage name wins over failed, then stopped. Raises ValueError if indeterminate."""
        if stage:
            key = self._stage_record_key(job, stage, pass_number)
            if key not in job.stage_statuses:
                raise ValueError(f"Unknown stage '{stage}'")
            return job.stage_keys.index(key)
        if job._failed_index is not None:
            return job._failed_index
        if job._stopped_index is not None:
            return job._stopped_index
        raise ValueError("retry_stage needs a 'stage' (no stage is failed or stopped)")

    def _reset_stage_for_retry(self, job: Job, index: int):
        """Clear one stage's recorded outcome for a re-run; downstream stages keep their outputs."""
        name = job.stage_keys[index]
        job.stage_statuses[name] = StageStatus.PENDING
        job.stage_durations.pop(name, None)
        job.stage_summaries.pop(name, None)
        job.stage_health.pop(name, None)
        job.stage_started_at.pop(name, None)
        job.error = None
        job._failed_index = None

    async def _restart_after_cancel(self, job: Job, index: int, reset=True):
        """Wait (bounded) for the cancelled loop to unwind, then start a loop from index. reset=False preserves a prior override."""
        for _ in range(100):  # bounded wait for the cancelled loop to unwind
            if not job._loop_live:
                break
            await asyncio.sleep(0.05)
        job._cancel_stage_flag = False
        job._cancel_all_flag = False
        job._cancel_event.clear()
        job._pause_event.set()
        if reset:
            self._reset_stage_for_retry(job, index)
        self._start_loop(job, index)

    def _reset_run(self, job: Job):
        """Clear every recorded outcome for a whole-pipeline re-run. Interventions and
        gate_history survive: decisions already taken stand."""
        job.context.outputs.clear()
        job._failed_index = None
        job._stopped_index = None
        job._cancel_stage_flag = False
        job._cancel_all_flag = False
        job._cancel_event.clear()
        job._pause_event.set()
        job._step_event.clear()
        # Follow-up pass keys are derived from results this re-run discards; reset
        # to the fixed stage list so stale `pending` keys do not persist.
        job.stage_statuses = {name: StageStatus.PENDING for name in job.stage_names}
        job.stage_keys = list(job.stage_names)
        job.context.pass_outputs.clear()
        job.context.current_pass = 1
        job.stage_durations.clear()
        job.stage_summaries.clear()
        job.stage_health.clear()
        job.stage_started_at.clear()
        job.error = None
        job.current_stage = None
        self._reset_event_history(job, "retry_all")

    async def _reset_all_after_cancel(self, job: Job):
        """Wait (bounded) for the cancelled loop to unwind, then reset and re-run from 0."""
        for _ in range(100):
            if not job._loop_live:
                break
            await asyncio.sleep(0.05)
        self._reset_run(job)
        self._start_loop(job, 0)

    def _cancel_without_a_loop(self, job: Job):
        """Apply cancel to a job with no running loop (restored or stage-cancelled)."""
        # Discard any open gate so the approvals inbox does not list a cancelled job.
        self._abandon_open_gate(job, "the run was cancelled")
        # An unarmed gate (restored, no waiter yet) is not open_gate; clear it so
        # rearm_gates does not re-open it after the next restart.
        job.pending_gate = None
        key = self._stage_record_key(job, job.current_stage) or (
            job.stage_keys[job._stopped_index]
            if job._stopped_index is not None
            and 0 <= job._stopped_index < len(job.stage_keys)
            else ""
        )
        if key and job.stage_statuses.get(key) == StageStatus.RUNNING:
            # No task running; a `running` badge is stale.
            job.stage_statuses[key] = StageStatus.CANCELLED
            job.stage_started_at.pop(key, None)
            self._emit(
                job,
                "stage_completed",
                stage=split_pass_key(key)[0],
                status=StageStatus.CANCELLED.value,
                message="Stage cancelled",
                pass_number=split_pass_key(key)[1],
            )
        if job._cancel_stage_flag and not job._cancel_all_flag:
            job._cancel_stage_flag = False
            job._cancel_event.clear()
            job.status = JobStatus.PAUSED
            job._pause_event.clear()
            message = "Stage cancelled; run paused"
        else:
            job.status = JobStatus.CANCELLED
            job.current_stage = None
            message = "Job cancelled"
        job.touch()
        self._emit(job, "job_status", status=job.status.value, message=message)
        self._persist(job)
        # No loop exit to release the slot. Ordinarily this job holds none, but a loop
        # retired by an earlier action may still be unwinding under a stale generation.
        self._release_slot(job.job_id)

    def control(self, job_id, action, stage=None, pass_number=None) -> Job:
        """Apply a control action; pass_number targets one retrieval pass, defaulting to the current. Raises KeyError or ValueError."""
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)

        if job.status == JobStatus.QUEUED and (
            action in _STARTING_ACTIONS or action.startswith("cancel")
        ):
            # Dequeue so the drain doesn't start a run that was already cancelled or resumed.
            if self.queue.withdraw(job_id):
                job.record_intervention(
                    "dequeued",
                    detail=f"taken out of the run queue by '{action}'",
                )

        if action in ("retry_stage", "skip_stage", "retry_all") and job.open_gate:
            # These start a fresh loop; abandon the gate first so the old loop stands down.
            # Done before validation: a rejected action leaves the gate intact.
            self._abandon_open_gate(job, action)

        if action == "pause":
            if job.status in _TERMINAL_STATUSES:
                raise ValueError(
                    f"Job is {job.status.value}; there is nothing to pause"
                )
            job._pause_event.clear()
            if not job._loop_live:
                # Nothing is running to notice the cleared event; report the state directly.
                job.status = JobStatus.PAUSED
                job.touch()
                self._emit(job, "job_status", status=job.status.value, message="Paused")
            else:
                # A mid-stage pause takes effect when the current stage finishes.
                self._emit(
                    job,
                    "job_status",
                    stage=job.current_stage,
                    status=job.status.value,
                    message=(
                        "Pause requested — it takes effect when "
                        f"{job.current_stage or 'the running stage'} finishes "
                        "(Cancel stage stops it now)"
                    ),
                )
        elif action == "resume":
            if job.status in _TERMINAL_STATUSES:
                raise ValueError(
                    f"Job is {job.status.value}; use retry_all to run it again"
                )
            job._pause_event.set()
            if not job._loop_live:
                # Setting the event alone would leave the job reporting `running` with nothing running.
                self._start_loop(job, self._resume_index(job))
            elif job.status == JobStatus.PAUSED:
                job.status = JobStatus.RUNNING
                job.touch()
                self._emit(
                    job, "job_status", status=job.status.value, message="Job running"
                )
        elif action == "step":
            if job.status in _TERMINAL_STATUSES:
                raise ValueError(
                    f"Job is {job.status.value}; use retry_all to run it again"
                )
            # Unblock exactly one stage; also lift any pause so it can proceed.
            job._pause_event.set()
            job._step_event.set()
            if not job._loop_live:
                self._start_loop(job, self._resume_index(job))
        elif action in ("cancel_stage", "cancel_all"):
            if job.status in _TERMINAL_STATUSES:
                raise ValueError(f"Job is already {job.status.value}")
            # Record the cancel on the audit trail; pause/resume/step are not recorded
            # because they change no data.
            job.record_intervention(
                action,
                stage=job.current_stage,
                detail=(
                    "analyst cancelled the running stage"
                    if action == "cancel_stage"
                    else "analyst cancelled the run"
                ),
            )
            if action == "cancel_stage":
                job._cancel_stage_flag = True
            else:
                job._cancel_all_flag = True
            job._pause_event.set()
            job._step_event.set()
            job._cancel_event.set()
            if job._current_task is not None and not job._current_task.done():
                job._current_task.cancel()
            if not job._loop_live:
                # No loop to reach _mark_cancelled; apply it directly.
                self._cancel_without_a_loop(job)
        elif action == "release_gate":
            # Recorded as an intervention because a stage not reviewed must not look reviewed.
            if not job.open_gate:
                raise ValueError("No gate is open for this job")
            job.record_intervention(
                "gate_released",
                stage=job.open_gate["stage"],
                detail="gate abandoned without a decision; run continued",
            )
            job._gate_decision = {"action": "approve", "restart_index": None}
            job._gate_event.set()
        elif action == "retry_stage":
            index = self._retry_index(job, stage, pass_number)
            target = job.stage_keys[index]
            # A live loop anywhere in the pipeline has to be stopped, not just one sitting on
            # the target: _run_gen retires a stale loop at a stage BOUNDARY, so a second loop
            # started beside it shares job.context, job.current_stage and current_pass until
            # the first finishes its in-flight stage, and the two then interleave.
            if job._loop_live or job.stage_statuses.get(target) == StageStatus.RUNNING:
                # Cancel first, then re-run once the cancel lands.
                on_target = self._stage_record_key(job, job.current_stage) == target
                job._cancel_stage_flag = True
                job._cancel_event.set()
                job._pause_event.set()
                job._step_event.set()
                if job._current_task is not None and not job._current_task.done():
                    job._current_task.cancel()
                job.record_intervention(
                    "retry_stage",
                    stage=split_pass_key(target)[0],
                    detail=_pass_detail(
                        target,
                        "running stage cancelled and re-run"
                        if on_target
                        else "run stopped and rewound to stage",
                    ),
                )
                self._reset_stage_for_retry(job, index)
                asyncio.ensure_future(self._restart_after_cancel(job, index))
            else:
                job.record_intervention(
                    "retry_stage",
                    stage=split_pass_key(target)[0],
                    detail=_pass_detail(target, "stage re-run by analyst"),
                )
                self._reset_stage_for_retry(job, index)
                self._start_loop(job, index)
        elif action == "skip_stage":
            # Used after an analyst supplies a stage's output by hand; retry would discard the override.
            target = (
                self._stage_record_key(job, stage, pass_number)
                if stage
                else (
                    job.stage_keys[job._failed_index]
                    if job._failed_index is not None
                    else None
                )
            )
            if target is None:
                raise ValueError(
                    "skip_stage needs a 'stage' (or a failed stage to skip)"
                )
            if target not in job.stage_statuses:
                raise ValueError(f"Unknown stage '{stage or target}'")
            index = job.stage_keys.index(target)
            # A running stage must stop, or it finishes later and writes its output over a
            # pipeline that has moved past it — and that is true of a loop live on ANY stage,
            # not only on the one being skipped, since starting from index+1 beside it leaves
            # two loops on one Job until the first reaches a boundary.
            skipping_live = job._loop_live or (
                job.stage_statuses[target] == StageStatus.RUNNING
            )
            if skipping_live:
                job._cancel_stage_flag = True
                job._cancel_event.set()
                job._pause_event.set()
                job._step_event.set()
                if job._current_task is not None and not job._current_task.done():
                    job._current_task.cancel()
            if job.stage_statuses[target] != StageStatus.COMPLETED:
                job.stage_statuses[target] = StageStatus.SKIPPED
            job.stage_started_at.pop(target, None)
            job._failed_index = None
            job.error = None
            job.record_intervention(
                "skip_stage",
                stage=split_pass_key(target)[0],
                detail=_pass_detail(target, "analyst continued past stage"),
            )
            if index + 1 >= len(job.stage_keys):
                # Skipped stage was the last one.
                job._cancel_stage_flag = False
                job._cancel_all_flag = False
                job._cancel_event.clear()
                job._pause_event.set()
                job._run_gen += 1  # retire any loop still unwinding
                job.status = JobStatus.COMPLETED
                job.current_stage = None
                job._stopped_index = None
                self._emit(
                    job,
                    "job_status",
                    status=job.status.value,
                    message="Job completed (last stage skipped)",
                )
                # This branch retired the loop; its finally sees a stale generation and won't release.
                self._release_slot(job.job_id)
            elif skipping_live:
                asyncio.ensure_future(
                    self._restart_after_cancel(job, index + 1, reset=False)
                )
            else:
                job._cancel_stage_flag = False
                job._cancel_all_flag = False
                job._cancel_event.clear()
                job._pause_event.set()
                self._start_loop(job, index + 1)
        elif action == "retry_all":
            job.record_intervention(
                "retry_all", detail="analyst re-ran the whole pipeline"
            )
            if job._loop_live:
                # The reset below replaces job.stage_statuses and empties job.context; a loop
                # still inside a stage writes into both, so it would mark a stage of the FRESH
                # run completed off the old one's result. Stop it, then reset.
                job._cancel_stage_flag = True
                job._cancel_event.set()
                job._pause_event.set()
                job._step_event.set()
                if job._current_task is not None and not job._current_task.done():
                    job._current_task.cancel()
                asyncio.ensure_future(self._reset_all_after_cancel(job))
            else:
                self._reset_run(job)
                self._start_loop(job, 0)
        else:
            raise ValueError(f"Unknown control action: {action}")

        job.touch()
        return job

    # -- human-in-the-loop output override ---------------------------------

    def stage_output_key(self, stage_name) -> str:
        """The ``ctx.outputs`` key a stage writes to. Raises KeyError if unknown."""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage.output_key
        raise KeyError(stage_name)

    def set_stage_output(
        self, job_id, stage_name, value, actor=None, pass_number=None, detail=None
    ) -> Job:
        """Replace one stage's output; refused while the stage is running (pause first). Raises KeyError or ValueError."""
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        output_key = self.stage_output_key(stage_name)  # KeyError on unknown stage
        if not output_key:
            raise ValueError(f"Stage '{stage_name}' does not store an output")
        key = self._stage_record_key(job, stage_name, pass_number)
        if (
            job.current_stage == stage_name
            and job.stage_statuses.get(key) == StageStatus.RUNNING
        ):
            raise ValueError(
                f"Stage '{stage_name}' is running; pause or cancel it before overriding"
            )

        decoded = decode_output_value(output_key, value)  # ValueError on a bad shape
        job.context.outputs[output_key] = decoded
        job.stage_statuses[key] = StageStatus.COMPLETED
        summary = summarize_stage(stage_name, decoded)
        job.stage_summaries[key] = summary
        # Re-score against the human's value so the corrected run is not reported with
        # the old (bad) health score.
        health = score_stage(stage_name, decoded, job.context, self.config)
        job.stage_health[key] = health.to_dict()
        # Clear the failed marker so the job can be continued. skip_stage is the
        # continuation action; retry_stage would discard the override.
        if job._failed_index is not None and (
            0 <= job._failed_index < len(job.stage_keys)
            and job.stage_keys[job._failed_index] == key
        ):
            job._failed_index = None
            job.error = None
        job.record_intervention(
            "override_stage_output",
            stage=stage_name,
            # detail lets structured callers (the retrieval-plan editor) name what changed.
            detail=_pass_detail(
                key,
                str(detail or "").strip()
                or f"outputs['{output_key}'] replaced by analyst",
            ),
            actor=actor,
        )
        self._emit(
            job,
            "stage_output",
            stage=stage_name,
            status=StageStatus.COMPLETED.value,
            message=f"{stage_name} output overridden by analyst"
            + (f" ({actor})" if actor else ""),
            data={
                "summary": summary,
                "health": health.to_dict(),
                "overridden": True,
            },
            pass_number=split_pass_key(key)[1],
        )
        self._emit(
            job,
            "intervention",
            stage=stage_name,
            status="override",
            message=f"Analyst overrode {stage_name} output",
            data={"actor": actor} if actor else None,
            pass_number=split_pass_key(key)[1],
        )
        # Hand-edited data exists nowhere but here; persist immediately.
        self._persist(job, evidence_changed=True)
        return job

    # -- export / import ---------------------------------------------------

    def export_job(self, job_id) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return {
            "job_id": job.job_id,
            "incident": job.incident,
            "status": job.status.value,
            "run_mode": job.run_mode.value,
            "stage_statuses": {
                key: job.stage_statuses[key].value
                for key in job.stage_keys
                if key in job.stage_statuses
            },
            # The run's ordered plan, which may be longer than stage_names if a follow-up
            # pass registered. Exported because it cannot be re-derived without a pack.
            "stage_keys": list(job.stage_keys),
            "current_pass": job.current_pass,
            "current_stage": job.current_stage,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "stage_durations": dict(job.stage_durations),
            "stage_summaries": dict(job.stage_summaries),
            "stage_health": dict(job.stage_health),
            "event_history": list(job._event_history),
            # Where the run stopped without finishing. No failed_index export: a `failed`
            # stage status already says which one, and import_job recomputes the index.
            "stopped_index": job._stopped_index,
            # Exported so a report built on an overridden intermediate stays traceable.
            "interventions": list(job.interventions),
            # Gate decisions and stage corrections must survive a round-trip or a
            # rehydrated job re-runs a rejected stage with the original prompt.
            "gate_history": list(job.gate_history),
            # pending_gate counts as open for export: without it a pre-rearm restart
            # would drop the unanswered decision.
            "open_gate": (
                dict(job.open_gate)
                if job.open_gate
                else (dict(job.pending_gate) if job.pending_gate else None)
            ),
            "stage_guidance": {
                k: list(v) for k, v in (job.context.stage_guidance or {}).items()
            },
            # link_modes must survive a round-trip or a restored job re-runs correlation
            # with every link reverted to the engine default.
            "link_modes": {
                str(k): str(v) for k, v in (job.context.link_modes or {}).items()
            },
            "outputs": _encode_outputs(job.context.outputs),
        }

    def import_job(self, doc: dict) -> Job:
        """Rehydrate a job from an export doc so it can be studied / retried / stepped."""
        incident = doc.get("incident", {})
        run_mode = JobRunMode(doc.get("run_mode", JobRunMode.AUTO.value))
        job = self.create_job(incident, run_mode=run_mode)
        # Restore the original id so exported links and snapshots remain valid.
        original_id = doc.get("job_id")
        if original_id:
            self._jobs.pop(job.job_id, None)
            job.job_id = original_id
            job.context.job_id = original_id
            self._jobs[original_id] = job
        job.context.outputs = _decode_outputs(doc.get("outputs", {}))
        # Re-register follow-up passes before reading statuses: without this a pass-2
        # status has no key to land on and resume would re-plan it.
        for number in sorted(
            {
                split_pass_key(k)[1]
                for k in list(doc.get("stage_keys") or [])
                + list((doc.get("stage_statuses") or {}).keys())
                if split_pass_key(k)[1] > 1
            }
        ):
            job.register_pass(number)
        exported_keys = [
            str(k)
            for k in (doc.get("stage_keys") or [])
            if str(k) in job.stage_statuses
        ]
        if exported_keys and set(exported_keys) == set(job.stage_keys):
            # Restore the exported order: index-based control actions use this list.
            job.stage_keys = exported_keys
        for name, value in (doc.get("stage_statuses") or {}).items():
            if name in job.stage_statuses:
                job.stage_statuses[name] = StageStatus(value)
        try:
            job.context.current_pass = max(1, int(doc.get("current_pass") or 1))
        except (TypeError, ValueError):
            job.context.current_pass = 1
        try:
            job.status = JobStatus(doc.get("status", JobStatus.PENDING.value))
        except ValueError:
            job.status = JobStatus.PENDING
        if job.status in _RESUMABLE_AS_PAUSED:
            # No task is running on a rehydrated job; paused is the truthful status.
            job.status = JobStatus.PAUSED
            job._pause_event.clear()
        # A stage marked `running` did not finish; treat it as `pending`.
        for name, status in list(job.stage_statuses.items()):
            if status == StageStatus.RUNNING:
                job.stage_statuses[name] = StageStatus.PENDING
        job.current_stage = doc.get("current_stage")
        job.error = doc.get("error")
        # created_ts is NOT restored: it drives the in-memory TTL and the real age would evict the job.
        for stamp in ("created_at", "updated_at"):  # not `field`; dataclasses.field
            value = doc.get(stamp)
            if isinstance(value, str) and value:
                setattr(job, stamp, value)
        job.stage_durations = dict(doc.get("stage_durations", {}))
        job.stage_summaries = dict(doc.get("stage_summaries", {}))
        job.stage_health = dict(doc.get("stage_health", {}))
        job._event_history = list(doc.get("event_history", []))[-_EVENT_HISTORY_CAP:]
        stopped = doc.get("stopped_index")
        if isinstance(stopped, int) and 0 <= stopped < len(job.stage_keys):
            job._stopped_index = stopped
        job.interventions = [
            i for i in (doc.get("interventions") or []) if isinstance(i, dict)
        ]
        job.gate_history = [
            g for g in (doc.get("gate_history") or []) if isinstance(g, dict)
        ]
        job.context.stage_guidance = {
            str(k): [str(x) for x in v]
            for k, v in (doc.get("stage_guidance") or {}).items()
            if isinstance(v, list)
        }
        # Restored without vocabulary checks; an unrecognised value is treated as undeclared.
        job.context.link_modes = {
            str(k): str(v)
            for k, v in (doc.get("link_modes") or {}).items()
            if isinstance(v, str)
        }
        # A gate open at export time becomes pending_gate; rearm_gates() parks a runner on it.
        job.pending_gate = (
            dict(doc["open_gate"]) if isinstance(doc.get("open_gate"), dict) else None
        )
        if job.pending_gate:
            logger.info(
                "Imported job %s has an unresolved gate on '%s'; call rearm_gates() to "
                "make it answerable again.",
                job.job_id,
                job.pending_gate.get("stage"),
            )
        # Recompute _failed_index so retry_stage works on a reloaded job.
        for index, name in enumerate(job.stage_keys):
            if job.stage_statuses[name] == StageStatus.FAILED:
                job._failed_index = index
                break
        return job


async def process_incident_via_job(
    incident, modules, config, job_manager, return_job=False
) -> Any:
    """Blocking end-to-end run via the job machinery; return_job=True returns the Job instead of just the report."""
    job = job_manager.create_job(incident, run_mode=JobRunMode.AUTO)
    await job_manager.run_job(job)
    if job.status == JobStatus.COMPLETED:
        return job if return_job else job.context.outputs.get("report")
    raise RuntimeError(
        f"Incident {incident.get('id')} did not complete "
        f"(status={job.status.value}): {job.error or 'no error recorded'}"
    )
