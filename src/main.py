import asyncio
import logging
import os
import re

import yaml

from anomaly_detection import AnomalyDetectionModule
from api_call_generator import ApiCallGenerator
from correlation import CorrelationModule
from export_results import ResultExporter
from identity import owner_scoped
from feedback_loop import FeedbackLoop
from incident_input import IncidentInputInterface
from incident_understanding import IncidentUnderstandingModule
from inquiry_probe import build_inquiry_probe
from job_store import build_job_store
from knowledge.pack import load_knowledge_pack
from link_probe import build_link_probe
from log_retrieval import LogRetrievalEngine
from notifications import WebhookDispatcher, emitter, send_notification
from output_interface import OutputInterface
from pipeline_runner import (JobManager, build_stage_descriptors,
                             configure_summaries, process_incident_via_job,
                             resolve_run_mode)
from plugin_system import PluginManager
from rag.embeddings import build_embedding_provider
from rag.orchestrator import KnowledgeOrchestrator
from rag.sources.confluence_source import ConfluenceSource
from rag.sources.databricks_source import DatabricksKnowledgeSource
from rag.sources.document_source import DocumentSource
from rag.sources.pack_schema_source import PackSchemaSource
from rag.sources.playbook_source import PlaybookSource
from report_generation import ReportGenerationModule
# Package-qualified: a flat import creates a second module object that `incident_input`
# (which uses `from src import report_delivery`) never reads. Same for `src.storage`.
from src import report_delivery
# Package-qualified for the same reason, and because `incident_input` reaches it that way:
# two module objects would give the HTTP layer a different journal than main() started.
from src.audit_journal import build_audit_journal
from src.storage import PrefixedStorage, build_storage
# Package-qualified for the same reason: the HTTP layer reads `src.user_secrets`, and a flat
# import here would install the store on a second module object nothing else can see.
from src.user_secrets import (UserSecretStore, offered_names,
                              set_secret_store)
from src.storage.mirror import (build_mirrors, seed_working_copies,
                                working_copies_are_writable)
from utils.databricks_auth import try_build_auth
from utils.llm_client import LLMClient
from utils.logging_setup import configure_logging
from utils.paths import (config_path, exports_dir, knowledge_base_dir,
                         knowledge_pack_dir)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Matches ${VAR} and ${VAR:-default} env-var references in config values.
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(obj):
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` references from os.environ.

    An unset bare var expands to "", so an absent credential reads as missing rather than
    as a literal that only 401s at query time. Env-var-*name* keys
    (``api_key_env``/``password_env``/``token_env``) and strings without ``${}`` are
    untouched.

    ``${VAR:-default}`` yields the default when the var is unset or empty, matching the
    shell. Empty counts as unset because a platform can inject a declared-but-blank var,
    and a default collapsed to "" for ``storage.backend`` means container-local disk.
    """
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return _ENV_RE.sub(_expand_one, obj)
    return obj


def _expand_one(match: "re.Match") -> str:
    name, default = match.group(1), match.group(2)
    value = os.environ.get(name, "")
    if value or default is None:
        return value
    return default


def load_config(config_file):
    with open(config_file, "r") as f:
        return _expand_env(yaml.safe_load(f))


async def process_incident(incident, modules):
    """Run the full pipeline for one incident and return the investigation report.

    The ``(incident, modules)`` signature is part of the contract (tests/test_main.py
    mocks against it). Retry is per stage in the job runner, not here; this stays the
    canonical inline pipeline the job stages mirror 1:1.
    """
    try:
        send_notification(incident["id"], "processing", "Started processing incident")

        understanding = await modules["understanding"].process(incident)
        logger.info(f"Incident {incident['id']} understanding complete")

        queries = await modules["api_call"].generate(understanding)
        logger.info(
            f"Generated {len(queries)} retrieval queries for incident {incident['id']}"
        )

        # Per-run state: retrievers are shared across requests. `keyed_sources` decides
        # zero-row semantics; `unanswered_sources` distinguishes timed-out from unplanned.
        keyed_sources: dict = {}
        unanswered_sources: dict = {}
        if modules["log_retrieval"].config.get("use_ssh_tunnel"):
            logs = await modules["log_retrieval"].retrieve_with_tunnel(
                queries,
                keyed_out=keyed_sources,
                unanswered_out=unanswered_sources,
            )
        else:
            logs = await modules["log_retrieval"].retrieve(
                queries,
                keyed_out=keyed_sources,
                unanswered_out=unanswered_sources,
            )
        logger.info(
            f"Retrieved logs from {len(logs)} sources for incident {incident['id']}"
            + (
                f"; {len(unanswered_sources)} source(s) did not answer: "
                + ", ".join(sorted(unanswered_sources))
                if unanswered_sources
                else ""
            )
        )

        # Correlate/aggregate the retrieved rows before anomaly detection so the
        # detector reasons over structured signal, not just raw rows.
        correlation = None
        if "correlation" in modules:
            correlation = await modules["correlation"].analyze(
                logs,
                understanding,
                keyed_sources=keyed_sources,
                unanswered_sources=unanswered_sources,
            )
            logger.info(
                f"Correlated {correlation.record_count} records for incident {incident['id']}"
            )

        anomalies = await modules["anomaly_detection"].detect(
            logs, understanding, correlation
        )
        logger.info(
            f"Detected {len(anomalies)} anomalies for incident {incident['id']}"
        )

        for plugin in modules["plugins"].get_active_plugins():
            plugin_result = await modules["plugins"].execute_plugin(
                plugin, incident, understanding, logs, anomalies
            )
            logger.info(
                f"Executed plugin {plugin} for incident {incident['id']} with result: {plugin_result}"
            )

        report = await modules["report_generation"].generate(
            incident, understanding, logs, anomalies, correlation
        )
        logger.info(f"Generated investigation report for incident {incident['id']}")

        # Export results next to the report. The paths stay full paths because the exporter
        # keys on the basename; an absent backend writes locally to exports_dir().
        exporter = ResultExporter(
            {
                "incident": incident,
                "understanding": {
                    "analysis": understanding.analysis.model_dump_json(indent=2)
                },
                "anomalies": [a.model_dump() for a in anomalies],
            },
            storage=owner_scoped(modules.get("artifact_storage"), incident),
        )
        exports = exports_dir()
        exporter.export_json(str(exports / f"incident_{incident['id']}.json"))
        exporter.export_csv(str(exports / f"incident_{incident['id']}.csv"))
        # Evidence artifacts: the full retrieved data, and the aggregated/correlated view.
        exporter.export_evidence_raw(
            str(exports / f"evidence_raw_{incident['id']}.json"), logs
        )
        exporter.export_evidence_transformed(
            str(exports / f"evidence_transformed_{incident['id']}.json"), correlation
        )
        logger.info(f"Exported results for incident {incident['id']}")

        # Best-effort: a delivery failure must not fail the whole investigation.
        if "output" in modules:
            try:
                await modules["output"].send(report, incident["id"])
            except Exception as e:
                logger.error(
                    "Output delivery failed for incident %s: %s", incident["id"], e
                )

        send_notification(incident["id"], "completed", "Incident processing completed")
        return report
    except Exception as e:
        logger.error(f"Error processing incident {incident['id']}: {str(e)}")
        send_notification(
            incident["id"], "error", f"Error processing incident: {str(e)}"
        )
        raise


def _resolve_port(config) -> int:
    """Databricks Apps inject DATABRICKS_APP_PORT; locally fall back to config."""
    env_port = os.getenv("DATABRICKS_APP_PORT")
    if env_port:
        return int(env_port)
    return config["incident_input"].get("port", 5000)


def _build_knowledge_sources(rag_config, knowledge_pack, db_backend, auth):
    """Instantiate KnowledgeSources from the rag.sources registry.

    Falls back to the legacy keys (playbook pack + document_paths + confluence)
    when no `sources` list is configured.
    """
    src_cfgs = rag_config.get("sources")
    sources = []

    if not src_cfgs:
        sources.append(PlaybookSource(knowledge_pack))
        doc_paths = rag_config.get("document_paths", [])
        if doc_paths:
            sources.append(DocumentSource(doc_paths))
        if rag_config.get("confluence", {}).get("url"):
            sources.append(ConfluenceSource(rag_config["confluence"]))
        return sources

    for cfg in src_cfgs:
        if not cfg.get("enabled", True):
            continue
        kind = cfg.get("type")
        if kind == "playbook":
            sources.append(PlaybookSource(knowledge_pack))
        elif kind == "document":
            sources.append(
                DocumentSource(
                    cfg.get("paths", []),
                    source_name=cfg.get("name", "local_documents"),
                )
            )
        elif kind == "pack_schema":
            sources.append(PackSchemaSource(knowledge_pack))
        elif kind == "confluence":
            sources.append(ConfluenceSource(cfg))
        elif kind == "databricks_table":
            sources.append(
                DatabricksKnowledgeSource(
                    source_name=cfg.get("name", "databricks_kb"),
                    table=cfg.get("table", ""),
                    title_col=cfg.get("title_col", "title"),
                    content_col=cfg.get("content_col", "content"),
                    type_col=cfg.get("type_col"),
                    where_clause=cfg.get("where_clause", ""),
                    extra_cols=cfg.get("extra_cols", []),
                    auth=auth,
                    warehouse_id=cfg.get("warehouse_id")
                    or db_backend.get("warehouse_id", ""),
                    workspace_url=cfg.get("workspace_url")
                    or db_backend.get("workspace_url", ""),
                    api_key_env=cfg.get("api_key_env")
                    or db_backend.get("api_key_env", "DATABRICKS_TOKEN"),
                    max_results=cfg.get("max_results", 500),
                )
            )
        else:
            logger.warning("Unknown RAG source type '%s'; skipping.", kind)
    return sources


async def main():
    # No-op locally; in a container, seeds shipped defaults into the writable working copy.
    # Copies only what is missing, so a warm restart cannot overwrite an operator's edit.
    seed_working_copies()

    main_config = load_config(config_path("main_config.yaml"))
    llm_config = load_config(config_path("llm_config.yaml"))

    # Replaces the import-time basicConfig above, first so every later line is in the
    # configured format. In a container the console is the only telemetry that leaves, and
    # `json` is what makes it queryable.
    log_format = configure_logging(main_config)
    logger.info("Logging configured: format=%s", log_format)

    # Durable storage for job state, evidence, exports and the feedback log. Built once and
    # handed to every consumer, so no module decides for itself where its half of the state
    # lives. No `storage:` block means local disk under AFIR_DATA_DIR.
    storage = build_storage(main_config)
    artifacts = PrefixedStorage(storage, "exports")
    # One path for both backends: branching here would leave the local deployment never
    # exercising what the remote one depends on. Local resolves to exports_dir(), where the
    # artifacts already are.
    report_delivery.set_storage(artifacts)
    logger.info("Durable storage: %s", storage.kind)
    if storage.degradation:
        logger.error("Storage is DEGRADED: %s", storage.degradation)

    # Mirrored, not moved: writes land on the local working copy and are then pushed to the
    # durable store. None on a local deployment where the working copy is the durable one.
    mirrors = build_mirrors(main_config, storage)
    if mirrors is not None:
        writable = working_copies_are_writable()
        for label, ok in writable.items():
            if not ok:
                logger.error(
                    "The %s working copy is NOT writable. Every edit through the UI will "
                    "fail; point AFIR_%s_DIR at a writable directory.",
                    label,
                    label.upper(),
                )
        logger.info("Mirrored trees synced down: %s", mirrors.sync_down())
        mirrors.install()
        if mirrors.degradation:
            logger.error("Config/pack durability is DEGRADED: %s", mirrors.degradation)

        # The first read could only see bundle defaults (config determines where the store is).
        # `storage:` is bundle-only; everything else picks up the operator's values here.
        main_config = load_config(config_path("main_config.yaml"))
        llm_config = load_config(config_path("llm_config.yaml"))
        configure_logging(main_config)

    # Per-caller credential overrides. Installed here, after the config re-read, because the
    # set of names a caller may replace is read from the live config: a name no reader resolves
    # would be a secret somebody could save to no effect. Nothing is stored for anybody until a
    # caller asks, and with no offerable name the whole feature reports itself unavailable.
    offered = offered_names(main_config, llm_config)
    set_secret_store(UserSecretStore(storage, offered=offered))
    logger.info(
        "Personal credential overrides: %d name(s) a caller may replace for their own runs%s",
        len(offered),
        f" ({', '.join(sorted(offered))})" if offered else "",
    )

    # Works for a local PAT and an App's OAuth service principal. None when the SDK or its
    # credentials are absent, in which case the LLM client and retrievers fall back to the
    # static api_key_env token.
    auth = try_build_auth()

    # Blank base_url resolves to Model Serving via the SDK. An explicit non-Serving
    # base_url uses the static api_key_env token instead.
    configured_base_url = str(llm_config.get("base_url") or "").strip()
    use_auth_for_llm = auth is not None and (
        not configured_base_url or "serving-endpoints" in configured_base_url
    )
    llm_client = LLMClient(llm_config, auth=auth if use_auth_for_llm else None)

    # A missing credential is knowable here, before anything is accepted, and it fails every
    # LLM stage. Non-fatal, since an operator may start the server and export the token
    # before submitting anything.
    if not use_auth_for_llm and not llm_client.credential_available:
        env_name = llm_config.get("api_key_env") or "(none configured)"
        logger.error(
            "NO LLM CREDENTIAL: env var %s is empty, so EVERY LLM stage will fail "
            "with 401 (understanding, query generation, correlation, anomaly "
            "detection, report narration). Export it and restart — e.g. "
            "`source .afir_env && python app.py`.",
            env_name,
        )

    # The pack drives entity extraction, source selection, and field mapping. No fallback
    # name: a wrong pack adjudicates with the wrong rules, so unconfigured stops here.
    pack_name = str(
        (main_config.get("knowledge", {}) or {}).get("pack_dir", "") or ""
    ).strip()
    if not pack_name:
        raise ValueError(
            "knowledge.pack_dir is not set in main_config.yaml. Set it to the directory "
            "name of the domain knowledge pack to investigate with (a sub-directory of "
            "knowledge/). See knowledge/README.md for how to author one."
        )
    knowledge_pack = load_knowledge_pack(knowledge_pack_dir(pack_name))

    # Multiple source types consolidate into one knowledge base. Returns EnhancedRAG when
    # embedding loads, PlaybookFallback when it cannot; None when there is nothing to serve.
    rag = None
    rag_config = main_config.get("rag", {})
    if rag_config.get("use_rag"):
        try:
            db_backend = (
                main_config.get("log_sources", {})
                .get("backends", {})
                .get("databricks", {})
            )
            sources = _build_knowledge_sources(
                rag_config, knowledge_pack, db_backend, auth
            )
            # `auth` flows in so a remote provider inherits the App's OAuth identity. A
            # relative local model path is anchored to the repo root, not cwd.
            embedding_provider = build_embedding_provider(rag_config, auth)
            logger.info("Embedding provider: %s", embedding_provider.signature)
            orchestrator = KnowledgeOrchestrator(
                sources=sources,
                # Anchor the KB to the repo / AFIR_DATA_DIR (works from any cwd).
                knowledge_base_path=str(knowledge_base_dir()),
                embedding_provider=embedding_provider,
                embedding_dim=rag_config.get("embedding_dim", 768),
                max_retrieved_documents=rag_config.get("max_retrieved_documents", 10),
                similarity_threshold=rag_config.get("similarity_threshold", 0.5),
            )
            rag = await orchestrator.build()
            if rag is not None:
                logger.info(
                    "RAG layer ready: %s (%d total docs, %d playbooks).",
                    type(rag).__name__,
                    len(orchestrator.kb_manager.documents),
                    len(orchestrator.kb_manager.get_documents_by_source("playbook")),
                )
            else:
                logger.warning("RAG layer unavailable; continuing without context.")
        except Exception as e:
            logger.error("RAG setup failed; continuing without RAG: %s", e)
            rag = None

    # Built before the query generator so it can be told which sources actually have a
    # retriever; otherwise the LLM targets one skipped for absent credentials and retrieval
    # fails at query time.
    log_retrieval_engine = LogRetrievalEngine(
        main_config["log_sources"],
        llm_client,
        auth=auth,
        knowledge_pack=knowledge_pack,
    )
    available_sources = list(log_retrieval_engine.retrievers.keys())
    logger.info(
        "Retrieval engine built %d sources: %s",
        len(available_sources),
        ", ".join(sorted(available_sources)) or "(none)",
    )

    # Built before the pipeline modules; understanding and anomaly-detection read its
    # guidance. `anomaly_config` carries the declared threshold, the fallback when tuning
    # is off. Storage is the root, un-prefixed: feedback files live directly under data_dir.
    feedback_loop = FeedbackLoop(
        llm_client,
        main_config.get("feedback", {}),
        anomaly_config=main_config.get("anomaly_detection", {}),
        storage=storage,
    )

    # Two consumers need it: the query-generation stage and the link-lane probe.
    # Both build queries through `build_manual_query`, so it is hoisted out of the dict.
    api_call_generator = ApiCallGenerator(
        main_config["log_sources"],
        llm_client,
        knowledge_pack=knowledge_pack,
        available_sources=available_sources,
    )

    modules = {
        "understanding": IncidentUnderstandingModule(
            llm_client, rag, knowledge_pack=knowledge_pack, feedback=feedback_loop
        ),
        "api_call": api_call_generator,
        "log_retrieval": log_retrieval_engine,
        "correlation": CorrelationModule(
            main_config.get("correlation", {}),
            llm_client,
            rag,
            knowledge_pack=knowledge_pack,
            # Per-source row caps, so the evidence pack can flag a source that returned
            # exactly its cap instead of presenting the cap as a real count.
            row_caps=log_retrieval_engine.row_caps(),
            # The only thing that lets the advisory link lane reach a backend, injected
            # because correlation is deliberately IO-free. What it may spend is the config's
            # to bound; `max_probes_per_run: 0` withdraws the capability entirely.
            link_probe=build_link_probe(api_call_generator, log_retrieval_engine),
            # The same for the other advisory lane, through the same two seams: what an open
            # question may spend is bounded by `inquiries.max_inquiry_probes_per_run` and by the
            # ceiling it shares with the link lane, and `0` withdraws the capability entirely.
            inquiry_probe=build_inquiry_probe(api_call_generator, log_retrieval_engine),
        ),
        "anomaly_detection": AnomalyDetectionModule(
            main_config["anomaly_detection"], llm_client, rag, feedback=feedback_loop
        ),
        "report_generation": ReportGenerationModule(
            main_config["report_generation"],
            llm_client,
            rag,
            knowledge_pack=knowledge_pack,
            # The same `artifacts` view report_delivery reads through, so the .md/.pdf this
            # stage writes are the ones the Report tab serves.
            storage=artifacts,
        ),
        "output": OutputInterface(main_config["output_interface"]),
        "plugins": PluginManager(main_config["plugin_dir"]),
        "feedback": feedback_loop,
        # Not a pipeline stage: the export step of both run paths reads it to build a
        # ResultExporter. Carried in `modules` because `process_incident` takes nothing else;
        # an absent key means local disk.
        "artifact_storage": artifacts,
    }

    modules["plugins"].load_plugins()

    # Backs both controllable jobs and classic endpoints. The store is durable because
    # approval gates hold indefinitely; a pending decision must outlive a restart.
    job_store = build_job_store(main_config, storage=storage)
    # The other half of "usage": every durable record above is about a RUN, so a caller who
    # only browses, reads someone else's report, is refused at the door or edits the shared
    # configuration leaves no trace anywhere. Pruned before the server accepts requests
    # because that listing is the one blocking read it does.
    audit_journal = build_audit_journal(main_config, storage)
    if audit_journal.enabled:
        audit_journal.prune()
        logger.info(
            "Access journal enabled: %s storage, flushed every %.0fs, %s.",
            getattr(storage, "kind", "unknown"),
            audit_journal.flush_seconds,
            # `0` means keep everything, so printing the number would read as the opposite of
            # what it does — "kept 0 day(s)" is how a journal that discards its own output
            # would announce itself.
            f"kept {audit_journal.retention_days} day(s)"
            if audit_journal.retention_days
            else "kept indefinitely",
        )
    # Sets how many entries each per-stage summary list keeps. A module global, since the
    # summarizers are module-level functions; also re-set by the config-apply path, as
    # `jobs.summary_max_items` is `live`.
    configure_summaries(main_config)
    job_manager = JobManager(
        build_stage_descriptors(),
        emitter,
        modules=modules,
        config=main_config,
        store=job_store,
    )
    emitter.set_job_manager(job_manager)
    # Set after construction because the reference is circular. A process where this key
    # is absent (any test building `modules` by hand) behaves identically.
    modules["job_manager"] = job_manager

    # Optional third event sink. SSE needs a client to stay connected, and a gate holding for
    # days outlives any browser tab, so an integrating app needs a push it can receive while
    # nothing of its own is running. Disabled by default and never able to fail a run.
    webhooks = WebhookDispatcher(main_config)
    emitter.set_webhooks(webhooks)
    if webhooks.enabled and webhooks.targets:
        logger.info(
            "Webhooks enabled: %d target(s) — %s",
            len(webhooks.targets),
            ", ".join(t["name"] for t in webhooks.targets),
        )

    # Reload persisted jobs before the server accepts requests, so an analyst polling
    # /api/v1/gates right after a restart sees the real queue rather than an empty one.
    restored = job_manager.restore()
    if restored:
        unfinished = [
            r
            for r in job_manager.list_jobs()
            if r["status"] not in ("completed", "cancelled")
        ]
        logger.info(
            "Job history restored: %d job(s), %d unfinished and awaiting a decision.",
            restored,
            len(unfinished),
        )
    # Re-arm approval gates from before the restart. Separate from restore() because it
    # parks waiters on the event loop, which must already be running.
    job_manager.rearm_gates()
    # Same reason, one status over: a run that was still in the backlog when the process
    # stopped had done nothing, so it goes back into the backlog rather than waiting for
    # an operator to notice a pause it never chose.
    job_manager.resume_queued()

    # The classic endpoints run the pipeline end-to-end over the job machinery and return the
    # report. verbose=True returns the Job instead, so the endpoint can attach per-stage
    # timings and event history alongside it.
    async def process_fn(incident, verbose=False):
        return await process_incident_via_job(
            incident, modules, main_config, job_manager, return_job=verbose
        )

    # Launch a controllable job (non-blocking); progress streamed over SSE. `mode` is a
    # plain string: "auto", "semi_auto", "supervised", or "step".
    def launch_fn(incident, mode="auto"):
        job = job_manager.create_job(incident, run_mode=resolve_run_mode(mode))
        # Through the queue, not straight to a loop: a run costs ~38 minutes at the median
        # and fans out ~19 retrievers against one rate-limited endpoint, so submissions past
        # the configured width wait rather than making every run in flight slower. The
        # QueueFull refusal travels to the caller, which turns it into a 429.
        job_manager.submit_job(job)
        return job

    # POST /api/v1/feedback. **structured carries the review fields (agrees_with_verdict,
    # analyst_verdict, missed_anomalies, false_positives, notes, analyst, job_id).
    async def feedback_fn(
        incident_id, investigation_result, human_feedback, **structured
    ):
        return await modules["feedback"].collect_feedback(
            incident_id, investigation_result, human_feedback, **structured
        )

    port = _resolve_port(main_config)
    input_server = IncidentInputInterface(
        main_config["incident_input"],
        process_fn=process_fn,
        feedback_fn=feedback_fn,
        job_manager=job_manager,
        launch_fn=launch_fn,
        feedback_loop=feedback_loop,
        # The full config dict, so the Configuration UI can apply the fields marked `live`
        # (modules hold their slices by reference) instead of reporting every change as
        # restart-required.
        live_config=main_config,
        # For the one live field a module cannot pick up from the dict alone: each primary
        # source's retrieval budget is copied onto its retriever at build time, so it has to
        # be re-resolved for an edit to take effect on the next run.
        on_live_reload=log_retrieval_engine.refresh_primary_budgets,
        # The pack assistant shares the pipeline's client, and with it the concurrency
        # semaphore and rate limiter; a second client could push the endpoint past the limit
        # while an investigation is already fanning out.
        llm_client=llm_client,
        # Read-only, so `/health?deep=1` can answer where state lives and whether it is
        # usable: a store that accepts a backend switch and then refuses every write has to be
        # visible somewhere other than the boot log.
        storage=storage,
        # Read-only: which declared sources built no retriever. Without this the run
        # reports success while every decisive condition is `unknown`.
        retrieval_engine=log_retrieval_engine,
        # Read-only, and the loaded object rather than the name above: an absent pack
        # directory loads as an empty pack, which is the one degradation that leaves every
        # stage running and answers nothing.
        knowledge_pack=knowledge_pack,
        # Write-only from the interface's side: the HTTP layer is the only place that sees a
        # caller at all, so it is the only place that can record one.
        audit_journal=audit_journal,
    )

    try:
        await input_server.start_server(port=port)
        # After the server, because the periodic flush needs the running loop.
        audit_journal.start()
        logger.info(
            "Fraud Investigation System ready on port %s. Submit incidents to / or the API.",
            port,
        )
        # Single long-lived server: sleep until cancelled or shut down.
        await asyncio.Event().wait()
    finally:
        await modules["log_retrieval"].close()
        # A bounded moment for in-flight webhook POSTs to land, bounded because a shutdown
        # must not hang on an unresponsive receiver.
        await webhooks.close()
        # Before the storage drain, since draining the journal is what puts its last entries
        # INTO the storage queue.
        await audit_journal.stop()
        # Drain the storage queue last: a queueing backend holds writes in memory and a
        # shutdown that skips this discards the state the restart needs. Bounded, off-thread.
        await asyncio.to_thread(storage.close, 10.0)


if __name__ == "__main__":
    asyncio.run(main())
