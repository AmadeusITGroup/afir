import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

import aiohttp
from aiohttp import web

from src import (
    config_store,
    link_children,
    link_escalation,
    openapi_docs,
    report_delivery,
    retrieval_cache,
)
from src.knowledge import pack_assistant, pack_attachments, pack_store, pack_validate
from src.links import compose_referral
from src.utils.deployment import (
    is_recognised_mode,
    resolve_mode,
    running_as_databricks_app,
)
from src.utils.error_handling import async_retry_with_backoff
from src.utils.rate_limiter import AsyncRateLimiter
from src.webui import INDEX_HTML

logger = logging.getLogger(__name__)


def validate_ir_request(request):
    required_fields = ["id"]
    return all(field in request for field in required_fields)


def validate_incident(incident):
    # A description is the only hard requirement; id/timestamp are auto-filled.
    return bool(incident.get("description"))


def _truthy(value):
    """Coerce a JSON flag (bool / "true"/"1"/"yes" string) to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _pass_number(payload):
    """The optional ``pass`` on a job-control body: a positive int, else ``None``.

    ``None`` means the pass the run is on, not the first one. A malformed value is ``None``
    too, since the key only refines an action already specified without it.
    """
    if not isinstance(payload, dict):
        return None
    raw = payload.get("pass")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        number = int(raw)
    except (TypeError, ValueError):
        return None
    return number if number >= 1 else None


class _PlanUnavailable(Exception):
    """A refusal the query-plan read and write share, carrying the response to send."""

    def __init__(self, response):
        super().__init__("query plan unavailable")
        self.response = response


class IncidentInputInterface:
    def __init__(
        self,
        config,
        process_fn=None,
        feedback_fn=None,
        job_manager=None,
        launch_fn=None,
        feedback_loop=None,
        live_config=None,
        on_live_reload=None,
        llm_client=None,
        storage=None,
        retrieval_engine=None,
    ):
        self.config = config
        # Read side only, for `?deep=1`. None reports `null` rather than a failure.
        self.storage = storage
        self.retrieval_engine = retrieval_engine
        # Shared with the pipeline so the pack assistant does not compete with a run for the
        # same rate limit. None makes the assist endpoints answer 503.
        self.llm_client = llm_client
        # Called after a `live` field is written, for modules holding a value derived from it
        # at build time. Unwired, such fields are reported restart-required.
        self.on_live_reload = on_live_reload
        # The whole main_config dict, held by reference: the modules hold slices of it, so
        # writing back here is what makes a `live` field take effect without a restart.
        self.live_config = live_config if isinstance(live_config, dict) else None
        # The read side (history, stats, insights, guidance) plus force-distill; feedback_fn
        # stays the write path, so wiring only the function still works.
        self.feedback_loop = feedback_loop
        # process_fn(incident) -> report, for the inline endpoints. None rejects them.
        self.process_fn = process_fn
        # feedback_fn(incident_id, investigation_result, human_feedback) -> None.
        self.feedback_fn = feedback_fn
        # job_manager owns the jobs + SSE streams; launch_fn(incident, mode) creates and
        # starts one. Both None disables /api/v1/jobs; the classic endpoints still work.
        self.job_manager = job_manager
        self.launch_fn = launch_fn
        # A ceiling, not a policy: aiohttp defaults to 1 MB, which a generated schema file
        # exceeds, and each handler enforces its own smaller cap with a 413.
        self.app = web.Application(client_max_size=32 * 1024 * 1024)
        self.app.router.add_get("/", self.index)
        self.app.router.add_get("/health", self.health)
        # Fixed paths, not config-driven: a generator cannot discover a renamed one.
        self.app.router.add_get("/openapi.json", self.openapi_json)
        self.app.router.add_get("/docs", self.api_docs)
        self.app.router.add_post(
            config["post_incident_endpoint"], self.receive_incident
        )
        # aiohttp keys routes on (method, path), so the two coexist off one config value.
        self.app.router.add_get(
            config["post_incident_endpoint"], self.list_finished_incidents
        )
        self.app.router.add_post(config["post_ir_endpoint"], self.get_ir)
        freetext_endpoint = config.get(
            "post_freetext_endpoint", "/api/v1/incidents/freetext"
        )
        self.app.router.add_post(freetext_endpoint, self.receive_freetext_incident)
        feedback_endpoint = config.get("post_feedback_endpoint", "/api/v1/feedback")
        self.app.router.add_post(feedback_endpoint, self.receive_feedback)
        self.app.router.add_get(feedback_endpoint, self.get_feedback)
        self.app.router.add_get(
            feedback_endpoint + "/guidance", self.get_feedback_guidance
        )
        self.app.router.add_post(
            feedback_endpoint + "/process", self.process_feedback_now
        )
        # The numeric half of "apply": the feedback-derived threshold and its evidence.
        self.app.router.add_get(
            feedback_endpoint + "/threshold", self.get_feedback_threshold
        )
        self.app.router.add_post(
            feedback_endpoint + "/threshold/reset", self.reset_feedback_threshold
        )
        self.app.router.add_post("/api/v1/jobs", self.create_job)
        self.app.router.add_get("/api/v1/jobs", self.list_jobs)
        self.app.router.add_post("/api/v1/jobs/import", self.import_job)
        self.app.router.add_get("/api/v1/jobs/{job_id}", self.get_job)
        self.app.router.add_get("/api/v1/jobs/{job_id}/events", self.job_events)
        self.app.router.add_get("/api/v1/jobs/{job_id}/export", self.export_job)
        self.app.router.add_post("/api/v1/jobs/{job_id}/control", self.control_job)
        # A batch is a label on its jobs, so every job route above works on each member.
        self.app.router.add_post("/api/v1/batches", self.create_batch)
        self.app.router.add_get("/api/v1/batches", self.list_batches)
        self.app.router.add_get("/api/v1/batches/{batch_id}", self.get_batch)
        self.app.router.add_post(
            "/api/v1/batches/{batch_id}/cancel", self.cancel_batch
        )
        # GET reads the open gate, for a client that missed the SSE event; POST resolves it.
        self.app.router.add_get("/api/v1/jobs/{job_id}/gate", self.get_job_gate)
        self.app.router.add_post("/api/v1/jobs/{job_id}/gate", self.resolve_job_gate)
        # The approvals inbox, including jobs this client did not launch.
        self.app.router.add_get("/api/v1/gates", self.list_gates)
        self.app.router.add_post(
            "/api/v1/jobs/{job_id}/outputs/{stage}", self.override_stage_output
        )
        self.app.router.add_get("/api/v1/jobs/{job_id}/queries", self.get_job_queries)
        self.app.router.add_post("/api/v1/jobs/{job_id}/queries", self.edit_job_queries)
        self.app.router.add_post(
            "/api/v1/jobs/{job_id}/links/{index}/refer", self.refer_job_link
        )
        self.app.router.add_get(
            "/api/v1/jobs/{job_id}/links", self.get_job_links
        )
        # Keyed on the target procedure, not on an index: correlation re-runs from a gate
        # rejection and renumbers every link, so an index-keyed setting would migrate.
        self.app.router.add_post(
            "/api/v1/jobs/{job_id}/links/mode", self.set_job_link_modes
        )
        # --- artifacts. Keyed by incident id as well as job id, because they outlive the
        # in-memory job; both spellings resolve to the same files under exports_dir().
        self.app.router.add_get("/api/v1/jobs/{job_id}/report", self.get_job_report)
        self.app.router.add_get("/api/v1/jobs/{job_id}/evidence", self.get_job_evidence)
        self.app.router.add_get(
            "/api/v1/jobs/{job_id}/artifacts", self.get_job_artifacts
        )
        self.app.router.add_get(
            "/api/v1/incidents/{incident_id}/report", self.get_incident_report
        )
        self.app.router.add_get(
            "/api/v1/incidents/{incident_id}/evidence", self.get_incident_evidence
        )
        self.app.router.add_get(
            "/api/v1/incidents/{incident_id}/artifacts", self.get_incident_artifacts
        )
        # --- configuration: read (redacted), patch fields, replace/import whole files.
        self.app.router.add_get("/api/v1/config", self.get_config)
        self.app.router.add_put("/api/v1/config", self.patch_config)
        self.app.router.add_get("/api/v1/config/{name}", self.export_config_file)
        self.app.router.add_put("/api/v1/config/{name}", self.replace_config_file)
        self.app.router.add_post("/api/v1/config/import", self.import_config)
        # --- knowledge packs. Registered unconditionally, unlike the job routes: the pack
        # editor needs no pipeline, and `test_webui.py` scans this router without one.
        self.app.router.add_get("/api/v1/knowledge", self.list_knowledge_packs)
        # Literal before dynamic: aiohttp matches in registration order, so `/{pack}` first
        # would swallow `/scaffold` and route the create POST to the read handler.
        self.app.router.add_post(
            "/api/v1/knowledge/scaffold", self.scaffold_knowledge_pack
        )
        self.app.router.add_get("/api/v1/knowledge/{pack}", self.get_knowledge_pack)
        self.app.router.add_get(
            "/api/v1/knowledge/{pack}/tree", self.get_knowledge_tree
        )
        self.app.router.add_get(
            "/api/v1/knowledge/{pack}/file", self.get_knowledge_file
        )
        self.app.router.add_put(
            "/api/v1/knowledge/{pack}/file", self.save_knowledge_file
        )
        self.app.router.add_post(
            "/api/v1/knowledge/{pack}/file", self.create_knowledge_file
        )
        self.app.router.add_delete(
            "/api/v1/knowledge/{pack}/file", self.delete_knowledge_file
        )
        self.app.router.add_get(
            "/api/v1/knowledge/{pack}/validate", self.validate_knowledge_pack
        )
        self.app.router.add_get(
            "/api/v1/knowledge/{pack}/history", self.get_knowledge_history
        )
        self.app.router.add_post(
            "/api/v1/knowledge/{pack}/history/restore", self.restore_knowledge_file
        )
        self.app.router.add_post(
            "/api/v1/knowledge/{pack}/import", self.import_knowledge_files
        )
        # --- the pack assistant. Literal before dynamic here too.
        self.app.router.add_get(
            "/api/v1/knowledge/{pack}/assist", self.list_knowledge_assists
        )
        self.app.router.add_post(
            "/api/v1/knowledge/{pack}/assist", self.start_knowledge_assist
        )
        self.app.router.add_get(
            "/api/v1/knowledge/{pack}/assist/{session}", self.get_knowledge_assist
        )
        self.app.router.add_get(
            "/api/v1/knowledge/{pack}/assist/{session}/events",
            self.stream_knowledge_assist,
        )
        self.app.router.add_post(
            "/api/v1/knowledge/{pack}/assist/{session}/apply",
            self.apply_knowledge_assist,
        )
        self.app.router.add_post(
            "/api/v1/knowledge/{pack}/assist/{session}/reject",
            self.reject_knowledge_assist,
        )
        # The token-refill task needs a running loop, so it waits for start_server().
        self.rate_limiter = None

    async def start_server(self, port=None):
        self.rate_limiter = AsyncRateLimiter(
            self.config["rate_limit"]["requests"],
            self.config["rate_limit"]["per_seconds"],
        )
        runner = web.AppRunner(self.app)
        await runner.setup()
        # 0.0.0.0 so the same code serves locally and inside a Databricks App.
        host = self.config.get("host", "0.0.0.0")
        port = port or self.config["port"]
        site = web.TCPSite(runner, host, port)
        await site.start()
        logger.info(f"Incident input server started on {host}:{port}")

    # -- static routes -----------------------------------------------------

    async def index(self, request):
        return web.Response(text=INDEX_HTML, content_type="text/html")

    async def openapi_json(self, request):
        """The hand-authored description, as JSON for a client generator."""
        spec = openapi_docs.load_spec()
        if spec is None:
            return web.json_response(
                {"error": "specification unavailable",
                 "reason": openapi_docs.unavailable_reason()},
                status=503,
            )
        return web.json_response(spec)

    async def api_docs(self, request):
        """The same description rendered for a person, server-side: the App has no egress."""
        spec = openapi_docs.load_spec()
        if spec is None:
            return web.Response(
                text=openapi_docs.render_unavailable(openapi_docs.unavailable_reason()),
                content_type="text/html",
                status=503,
            )
        return web.Response(
            text=openapi_docs.render_html(spec), content_type="text/html"
        )

    async def health(self, request):
        # With no query parameter this is byte-identical to the plain 200: the platform
        # probe must not start depending on a collaborator being wired.
        if (request.query.get("deep") or "").lower() not in ("1", "true", "yes"):
            return web.json_response({"status": "ok"})
        # ?deep=1 reports the degradations that leave the server up. Each key distinguishes
        # "not wired" (None, legitimate in tests and in a pure-export deployment) from
        # "wired and unusable" (False).
        credential = None
        if self.llm_client is not None:
            credential = bool(getattr(self.llm_client, "credential_available", False))
        jobs = None
        run_queue = None
        if self.job_manager is not None:
            try:
                jobs = len(self.job_manager.list_jobs())
            except Exception:  # a broken lister must not fail the health answer
                jobs = None
            try:
                # The backlog beside the job count, because the two answer different
                # questions: a growing queue at a healthy width is a capacity fact, and
                # nothing else in the health answer would show it.
                run_queue = self.job_manager.queue.stats()
            except Exception:  # a broken reporter must not fail the health answer
                run_queue = None
        pack = None
        holder = getattr(self, "live_config", None)
        if isinstance(holder, dict):
            pack = bool((holder.get("knowledge") or {}).get("pack_dir"))
        # A remote store refusing every write reads as a working one until the restart that
        # finds nothing, so `storage_ok: False` carries the reason.
        storage_kind, storage_ok, storage_detail = None, None, None
        if self.storage is not None:
            storage_kind = str(getattr(self.storage, "kind", "") or "") or None
            try:
                storage_detail = self.storage.degradation
            except Exception:  # a broken reporter must not fail the health answer
                storage_detail = "the storage backend could not report its own state"
            storage_ok = storage_detail is None
        # Which declared sources built no retriever, with the reasons and not just a count:
        # they name the credential to go and set.
        sources_total, sources_unavailable = None, None
        if self.retrieval_engine is not None:
            try:
                unavailable = dict(
                    getattr(self.retrieval_engine, "unavailable_sources", None) or {}
                )
                built = len(getattr(self.retrieval_engine, "retrievers", None) or {})
                sources_unavailable = {str(k): str(v) for k, v in unavailable.items()}
                sources_total = built + len(sources_unavailable)
            except Exception:  # a broken engine must not fail the health answer
                sources_unavailable = {
                    "": "the retrieval engine could not report its own sources"
                }
        # The raw hit/miss counts ride beside the rate: a rate over three lookups is not a
        # measurement, and only the denominator says so.
        cache = None
        if self.retrieval_engine is not None:
            wired, stats = retrieval_cache.cache_stats(self.retrieval_engine)
            cache = stats if wired else None
        return web.json_response(
            {
                "status": "ok",
                "llm_credential": credential,
                "pack": pack,
                "jobs": jobs,
                "run_queue": run_queue,
                "pipeline": self.process_fn is not None,
                "storage": storage_kind,
                "storage_ok": storage_ok,
                "storage_detail": storage_detail,
                "sources_declared": sources_total,
                "sources_unavailable": sources_unavailable,
                "retrieval_cache": cache,
            }
        )

    async def list_finished_incidents(self, request):
        """Incidents with an artifact on disk, newest first: the Report tab's list."""
        try:
            limit = int(request.query.get("limit") or 200)
        except (TypeError, ValueError):
            limit = 200
        return web.json_response(
            {"incidents": report_delivery.list_incidents(limit=limit)}
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _normalize_incident(incident, source):
        """Fill in missing id/timestamp and tag the source."""
        incident.setdefault("id", str(uuid.uuid4()))
        incident.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        incident.setdefault("source", source)
        return incident

    @staticmethod
    def _wants_verbose(request):
        """True when the caller asked for per-stage timings + events (``?verbose=1``)."""
        val = (request.query.get("verbose") or "").lower()
        return val in ("1", "true", "yes")

    #: Sent by ``_run_pipeline``, the funnel all three blocking endpoints pass through.
    _BLOCKING_RETIRED_BODY = {
        "error": (
            "This endpoint holds the HTTP connection open for the whole "
            "investigation, and the platform's ingress closes a request long before one "
            "finishes (measured runs: 38 min median, 476 min longest). Submit the same "
            "incident to POST /api/v1/jobs, which returns a job id immediately; then "
            "poll GET /api/v1/jobs/{job_id} or stream "
            "GET /api/v1/jobs/{job_id}/events."
        ),
        "use_instead": "POST /api/v1/jobs",
    }

    def _blocking_endpoints_available(self) -> bool:
        """Whether the inline-until-done endpoints may run in this deployment.

        Off by default in a Databricks App, on everywhere else, and overridable in both
        directions (``incident_input.blocking_endpoints: auto|on|off``) because the ingress
        limit belongs to the platform rather than to us.
        """
        configured = (self.config or {}).get("blocking_endpoints")
        if not is_recognised_mode(configured):
            logger.error(
                "Unknown incident_input.blocking_endpoints %r; treating it as 'auto'. "
                "Valid values are 'auto', 'on' and 'off'.",
                configured,
            )
        return resolve_mode(
            configured, platform_default=not running_as_databricks_app()
        )

    async def _run_pipeline(self, incident, verbose=False):
        """Run the pipeline inline, returning ``{incident_id, report}``.

        ``verbose`` adds the job id, per-stage status and duration, and the event history:
        what the jobs API and the SSE stream surface, in one blocking call.
        """
        if not self._blocking_endpoints_available():
            logger.info(
                "Refusing a blocking submission for %s: this deployment cannot hold a "
                "request open for a full investigation. Directed to POST /api/v1/jobs.",
                incident.get("id"),
            )
            return web.json_response(dict(self._BLOCKING_RETIRED_BODY), status=501)
        if self.process_fn is None:
            return web.json_response(
                {"error": "Pipeline not available on this server instance."}, status=503
            )
        try:
            result = await self.process_fn(incident, verbose=verbose)
        except Exception as e:
            logger.error("Pipeline failed for incident %s: %s", incident["id"], e)
            return web.json_response(
                {"incident_id": incident["id"], "error": str(e)}, status=500
            )
        if not verbose:
            return web.json_response(
                {"incident_id": incident["id"], "report": _jsonable(result)}
            )
        job = result  # under verbose, process_fn returns the Job itself
        snap = job.snapshot()
        return web.json_response(
            {
                "incident_id": incident["id"],
                "job_id": job.job_id,
                "report": _jsonable(job.context.outputs.get("report")),
                "stages": snap["stages"],
                "events": list(job._event_history),
            }
        )

    # -- endpoints (inline) ------------------------------------------------

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def receive_incident(self, request):
        await self.rate_limiter.acquire()
        try:
            incident = await request.json()
            if not validate_incident(incident):
                return web.Response(status=400, text="Invalid incident data")

            incident = self._normalize_incident(incident, source="api")
            logger.info(f"Received incident: {incident['id']}")
            return await self._run_pipeline(
                incident, verbose=self._wants_verbose(request)
            )
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        except Exception as e:
            logger.error(f"Error receiving incident: {str(e)}")
            return web.Response(status=500, text="Internal server error")

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def receive_freetext_incident(self, request):
        """Accept a plain-text incident: only a description is required."""
        await self.rate_limiter.acquire()
        try:
            payload = await request.json()
            description = payload.get("description")
            if not description:
                return web.Response(
                    status=400, text="A non-empty 'description' is required"
                )

            base = {"description": description}
            if _truthy(payload.get("extended_retrieval")):
                base["extended_retrieval"] = True
            incident = self._normalize_incident(base, source="freetext")
            logger.info(f"Received free-text incident: {incident['id']}")
            return await self._run_pipeline(
                incident, verbose=self._wants_verbose(request)
            )
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        except Exception as e:
            logger.error(f"Error receiving free-text incident: {str(e)}")
            return web.Response(status=500, text="Internal server error")

    # Accepted alongside the free-form `human_feedback`; anything else is ignored.
    _FEEDBACK_FIELDS = (
        "agrees_with_verdict",
        "analyst_verdict",
        "missed_anomalies",
        "false_positives",
        "notes",
        "analyst",
        "job_id",
    )

    async def receive_feedback(self, request):
        """Accept human feedback on a completed investigation.

        Body: ``{"incident_id": str, ...}`` with ``human_feedback`` (any JSON) or at least
        one of ``_FEEDBACK_FIELDS``. Feeds the FeedbackLoop.
        """
        if self.feedback_fn is None:
            return web.json_response(
                {"error": "Feedback collection not available."}, status=503
            )
        await self.rate_limiter.acquire()
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                return web.Response(status=400, text="Body must be a JSON object")
            incident_id = payload.get("incident_id")
            human_feedback = payload.get("human_feedback")
            structured = {
                k: payload[k]
                for k in self._FEEDBACK_FIELDS
                if payload.get(k) is not None
            }
            if not incident_id:
                return web.Response(status=400, text="'incident_id' is required")
            if human_feedback is None and not structured:
                return web.Response(
                    status=400,
                    text=(
                        "provide 'human_feedback' or at least one structured review "
                        "field (" + ", ".join(self._FEEDBACK_FIELDS) + ")"
                    ),
                )
            # The list-valued fields accept a single string too.
            for key in ("missed_anomalies", "false_positives"):
                if isinstance(structured.get(key), str):
                    structured[key] = [structured[key]]
            await self.feedback_fn(
                incident_id,
                payload.get("investigation_result"),
                human_feedback,
                **structured,
            )
            logger.info("Collected feedback for incident %s", incident_id)
            body = {"incident_id": incident_id, "status": "received"}
            if self.feedback_loop is not None:
                body["stats"] = self.feedback_loop.stats()
            return web.json_response(body)
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        except Exception as e:
            logger.error("Error receiving feedback: %s", e)
            return web.Response(status=500, text="Internal server error")

    async def get_feedback(self, request):
        """Submitted reviews, the pending batch and distilled insights; ``?limit=N``."""
        if self.feedback_loop is None:
            return web.json_response(
                {"error": "Feedback inspection not available."}, status=503
            )
        try:
            limit = int(request.query.get("limit", 50))
        except ValueError:
            limit = 50
        loop = self.feedback_loop
        return web.json_response(
            {
                "stats": loop.stats(),
                "pending": list(loop.feedback_data),
                "history": loop.history(limit=limit),
                "insights": loop.merged_insights(refresh=True),
                "distillations": loop.load_insight_records(),
            }
        )

    async def get_feedback_guidance(self, request):
        """The exact guidance text injected into each consuming stage's prompt."""
        if self.feedback_loop is None:
            return web.json_response(
                {"error": "Feedback inspection not available."}, status=503
            )
        loop = self.feedback_loop
        stages = ("understanding", "anomaly_detection")
        return web.json_response(
            {
                "apply_to_prompts": loop.apply_to_prompts,
                "guidance": {
                    stage: loop.guidance_prompt(stage, refresh=True) for stage in stages
                },
            }
        )

    async def process_feedback_now(self, request):
        """Force-distill the pending batch without waiting for it to fill; 409 if empty.

        A partial batch is otherwise never distilled and never reaches a prompt.
        """
        if self.feedback_loop is None:
            return web.json_response(
                {"error": "Feedback processing not available."}, status=503
            )
        await self.rate_limiter.acquire()
        pending = len(self.feedback_loop.feedback_data)
        if pending == 0:
            return web.json_response(
                {"error": "No pending feedback to process."}, status=409
            )
        try:
            insights = await self.feedback_loop.process_feedback()
        except Exception as e:  # noqa: BLE001 — the distill LLM call can fail
            logger.error("Feedback distillation failed: %s", e)
            return web.json_response({"error": f"Distillation failed: {e}"}, status=502)
        return web.json_response(
            {
                "processed": pending,
                "insights": _jsonable(insights),
                "stats": self.feedback_loop.stats(),
                # Distillation re-evaluates the threshold, so report what that did.
                "threshold": self.feedback_loop.threshold_recommendation(refresh=True),
            }
        )

    async def get_feedback_threshold(self, request):
        """The feedback-derived anomaly confidence threshold, and the evidence for it.

        Readable with ``feedback.auto_tune_threshold`` off, where ``applied`` is false and
        ``recommended`` is advice. ``?configured=0.8`` overrides the baseline.
        """
        if self.feedback_loop is None:
            return web.json_response(
                {"error": "Feedback inspection not available."}, status=503
            )
        configured = request.query.get("configured")
        if configured is not None:
            try:
                configured = float(configured)
            except ValueError:
                return web.json_response(
                    {"error": "'configured' must be a number."}, status=400
                )
        loop = self.feedback_loop
        return web.json_response(
            {
                "threshold": loop.threshold_recommendation(configured, refresh=True),
                "effective": loop.effective_threshold(configured),
                "adjustments": loop.load_tuning_records(),
            }
        )

    async def reset_feedback_threshold(self, request):
        """Discard the tuning history so the configured baseline applies again."""
        if self.feedback_loop is None:
            return web.json_response(
                {"error": "Feedback tuning not available."}, status=503
            )
        ok = self.feedback_loop.reset_threshold()
        if not ok:
            return web.json_response(
                {"error": "Could not clear the tuning history."}, status=500
            )
        return web.json_response(
            {
                "status": "reset",
                "effective": self.feedback_loop.effective_threshold(),
                "stats": self.feedback_loop.stats(),
            }
        )

    # -- controllable jobs -------------------------------------------------

    _CONTROL_ACTIONS = {
        "pause",
        "resume",
        "step",
        "cancel_stage",
        "cancel_all",
        "retry_stage",
        "retry_all",
        # Continue past a stage instead of re-running it: the completion move after an
        # analyst overrides that stage's output, since retry would discard the override.
        "skip_stage",
        # Abandon an open approval gate and continue as if approved. Distinct from
        # `approve` on the gate endpoint: it records that nobody actually reviewed.
        "release_gate",
    }

    # Duplicated from pipeline_runner rather than imported: under the `src.`-qualified
    # style the import loads a second copy whose JobStatus no longer compares equal. The
    # manager re-validates, so this only buys a 400 instead of a 500.
    _GATE_ACTIONS = ("approve", "reject", "override")

    async def create_job(self, request):
        """Launch a controllable job and return its id immediately (non-blocking)."""
        if self.launch_fn is None:
            return web.json_response(
                {"error": "Job execution not available on this server instance."},
                status=503,
            )
        await self.rate_limiter.acquire()
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        description = payload.get("description")
        if not description:
            return web.Response(
                status=400, text="A non-empty 'description' is required"
            )
        mode = payload.get("mode", "auto")
        base = {"description": description}
        if _truthy(payload.get("extended_retrieval")):
            base["extended_retrieval"] = True
        # Not validated here: select_correlation_spec reads an unresolvable name as
        # "nothing was pinned", with a warning.
        pin = str(payload.get("link_pin", "") or "").strip()
        if pin:
            base["link_pin"] = pin
        incident = self._normalize_incident(base, source="freetext")
        try:
            job = self.launch_fn(incident, mode)
        except RuntimeError as exc:
            refusal = self._queue_refusal(exc)
            if refusal is None:
                raise
            return refusal
        logger.info("Launched job %s for incident %s", job.job_id, incident["id"])
        return web.json_response(
            {
                "job_id": job.job_id,
                "incident_id": incident["id"],
                # `queued` means accepted but not started; the run is real either way.
                "status": job.status.value,
            },
            status=201,
        )

    @staticmethod
    def _queue_refusal(exc):
        """A 429 for a full run queue, or None if ``exc`` is some other RuntimeError.

        Duck-typed on ``depth``/``limit`` rather than ``except QueueFull``, which is two
        classes under this repo's two import styles.
        """
        depth, limit = getattr(exc, "depth", None), getattr(exc, "limit", None)
        if not isinstance(depth, int) or not isinstance(limit, int):
            return None
        return web.json_response(
            {
                "error": str(exc),
                "queued": depth,
                "max_queued": limit,
                # Named so the caller retries rather than reading a lost incident.
                "retry": "poll GET /api/v1/jobs and resubmit once the backlog drains",
            },
            status=429,
        )

    async def list_jobs(self, request):
        """List all live jobs (newest first) so an API client can discover them."""
        if self.job_manager is None:
            return web.json_response({"error": "Jobs not available."}, status=503)
        return web.json_response(
            {
                "jobs": self.job_manager.list_jobs(),
                # Beside the rows, not on a route of its own: a caller reading `queued`
                # on a job needs the width and the depth in the same request.
                "queue": self.job_manager.queue.stats(),
            }
        )

    async def get_job(self, request):
        job = self._lookup_job(request)
        if job is None:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response(job.snapshot())

    #: Incidents one request may carry. A larger batch is refused whole, never truncated:
    #: a batch cut to its first N reads as a batch that ran.
    _MAX_BATCH_SIZE = 500

    async def create_batch(self, request):
        """``POST /api/v1/batches``: submit many incidents, each becoming a queued job.

        Body ``{"incidents": ["text", {"description": "..."}], "mode": "auto"}``. The batch
        is a label on its jobs, so every job endpoint works on each member unchanged.
        """
        if self.job_manager is None or self.launch_fn is None:
            return web.json_response({"error": "Jobs not available."}, status=503)
        await self.rate_limiter.acquire()
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        raw = payload.get("incidents")
        if not isinstance(raw, list) or not raw:
            return web.Response(
                status=400,
                text="A non-empty 'incidents' array is required",
            )
        if len(raw) > self._MAX_BATCH_SIZE:
            return web.json_response(
                {
                    "error": (
                        f"A batch carries at most {self._MAX_BATCH_SIZE} incidents; "
                        f"this request carried {len(raw)}. Split it."
                    )
                },
                status=413,
            )
        incidents, rejected = [], []
        for position, item in enumerate(raw):
            if isinstance(item, str):
                item = {"description": item}
            if not isinstance(item, dict):
                rejected.append({"index": position, "error": "not a string or object"})
                continue
            description = str(item.get("description") or "").strip()
            if not description:
                rejected.append({"index": position, "error": "empty 'description'"})
                continue
            base = {"description": description}
            if _truthy(item.get("extended_retrieval")):
                base["extended_retrieval"] = True
            pin = str(item.get("link_pin", "") or "").strip()
            if pin:
                base["link_pin"] = pin
            incidents.append(self._normalize_incident(base, source="batch"))
        if not incidents:
            return web.json_response(
                {"error": "No usable incident in the batch", "rejected": rejected},
                status=400,
            )
        result = self.job_manager.submit_batch(
            incidents,
            run_mode=payload.get("mode", "auto"),
            batch_id=str(payload.get("batch_id", "") or "").strip() or None,
        )
        # Beside what was accepted, never in place of it: a caller with 4 malformed out of
        # 300 needs both halves of the answer.
        result["rejected"] = rejected
        return web.json_response(result, status=201)

    async def list_batches(self, request):
        """Every batch with at least one live job, plus the queue's own counters."""
        if self.job_manager is None:
            return web.json_response({"error": "Jobs not available."}, status=503)
        return web.json_response(
            {
                "batches": self.job_manager.list_batches(),
                "queue": self.job_manager.queue.stats(),
            }
        )

    async def get_batch(self, request):
        """One batch's progress, derived from its jobs and nothing else."""
        if self.job_manager is None:
            return web.json_response({"error": "Jobs not available."}, status=503)
        try:
            return web.json_response(
                self.job_manager.batch_status(request.match_info["batch_id"])
            )
        except KeyError:
            # A batch whose jobs have aged out of memory reads like one that never existed;
            # saying so beats an empty batch that looks like nothing ran.
            return web.json_response(
                {"error": "No live job carries that batch id"}, status=404
            )

    async def cancel_batch(self, request):
        """Cancel every job in a batch that has not already ended."""
        if self.job_manager is None:
            return web.json_response({"error": "Jobs not available."}, status=503)
        try:
            return web.json_response(
                self.job_manager.cancel_batch(request.match_info["batch_id"])
            )
        except KeyError:
            return web.json_response(
                {"error": "No live job carries that batch id"}, status=404
            )

    async def job_events(self, request):
        """Stream a job's events as Server-Sent Events on the shared port."""
        if self.job_manager is None:
            return web.json_response({"error": "Jobs not available."}, status=503)
        job_id = request.match_info["job_id"]
        if self.job_manager.get_job(job_id) is None:
            return web.json_response({"error": "Job not found"}, status=404)
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # disable proxy buffering (nginx)
            },
        )
        await response.prepare(request)
        try:
            async for event in self.job_manager.subscribe(job_id):
                await response.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                if event.get("type") == "job_status" and event.get("status") in (
                    "completed",
                    "cancelled",
                ):
                    # Only if the job is *still* terminal: subscribers get the replay
                    # buffer first, so a stale terminal event would end a live stream.
                    job = self.job_manager.get_job(job_id)
                    if job is None or job.status.value in ("completed", "cancelled"):
                        break
        except (ConnectionResetError, asyncio.CancelledError):
            pass  # client disconnected; subscribe() cleans up its queue in finally
        except Exception as e:  # noqa: BLE001
            logger.warning("SSE stream error for job %s: %s", job_id, e)
        return response

    async def control_job(self, request):
        if self.job_manager is None:
            return web.json_response({"error": "Jobs not available."}, status=503)
        job_id = request.match_info["job_id"]
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        action = payload.get("action")
        if action not in self._CONTROL_ACTIONS:
            return web.json_response(
                {"error": f"Unknown action '{action}'"}, status=400
            )
        try:
            job = self.job_manager.control(
                job_id,
                action,
                payload.get("stage"),
                pass_number=_pass_number(payload),
            )
        except KeyError:
            return web.json_response({"error": "Job not found"}, status=404)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=409)
        return web.json_response(job.snapshot())

    async def get_job_gate(self, request):
        """The job's open gate, or ``{"open_gate": null}``, without replaying the stream."""
        job = self._lookup_job(request)
        if job is None:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response(
            {
                "job_id": job.job_id,
                "status": job.status.value,
                "open_gate": job.open_gate,
                "gate_history": job.gate_history,
            }
        )

    async def list_gates(self, request):
        """Every gate currently awaiting a human, across all live jobs (the inbox)."""
        if self.job_manager is None:
            return web.json_response({"error": "Jobs not available."}, status=503)
        return web.json_response({"gates": self.job_manager.list_open_gates()})

    async def resolve_job_gate(self, request):
        """Resolve an open approval gate.

        ``POST /api/v1/jobs/{job_id}/gate`` with one of:

        - ``approve``: continue. Optional ``actor``, ``note``.
        - ``reject``: re-run ``restart_from`` (default: the gated stage) with ``guidance``
          injected into its prompt, then gate again. ``guidance`` is required, since without
          it the retry re-runs an identical prompt. Optional ``reason_code``, ``pass``.
        - ``override``: replace the output with ``value`` and continue, via the same codec and
          audit path as ``POST /outputs/{stage}``.

        409 when no gate is open; 400 on a bad action or a value that does not fit the stage's
        contract.
        """
        if self.job_manager is None:
            return web.json_response({"error": "Jobs not available."}, status=503)
        job_id = request.match_info["job_id"]
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        if not isinstance(payload, dict):
            return web.Response(status=400, text="Body must be a JSON object")
        action = payload.get("action")
        if action not in self._GATE_ACTIONS:
            return web.json_response(
                {
                    "error": f"Unknown gate action '{action}'; expected one of "
                    + ", ".join(self._GATE_ACTIONS)
                },
                status=400,
            )
        if action == "override" and "value" not in payload:
            return web.Response(
                status=400, text="override requires 'value' (the corrected output)"
            )
        try:
            job = self.job_manager.resolve_gate(
                job_id,
                action,
                guidance=payload.get("guidance"),
                reason_code=payload.get("reason_code"),
                restart_from=payload.get("restart_from"),
                value=payload.get("value"),
                actor=payload.get("actor"),
                note=payload.get("note"),
                restart_pass=_pass_number(payload),
            )
        except KeyError:
            return web.json_response({"error": "Job not found"}, status=404)
        except ValueError as e:
            # "No gate is open" conflicts with the job's state; a malformed body does not.
            no_gate = "no gate is open" in str(e).lower()
            return web.json_response({"error": str(e)}, status=409 if no_gate else 400)
        # Durable before the analyst is told it was accepted: on a queueing backend a
        # restart in that window asks the same human the same question again.
        await self.job_manager.flush_persistence(job_id)
        logger.info(
            "Gate %s on job %s resolved: %s (actor=%s)",
            action,
            job_id,
            action,
            payload.get("actor"),
        )
        return web.json_response(job.snapshot())

    async def override_stage_output(self, request):
        """``POST /api/v1/jobs/{job_id}/outputs/{stage}``: replace a stage's output.

        Body ``{"value": <corrected output>, "actor": "<who>", "pass": N}``. The value is
        decoded through the same codecs as job import, so a value that does not fit the
        stage's contract is a 400 rather than a poisoned job. Continue with
        ``skip_stage``, not a retry, which would discard the override.
        """
        if self.job_manager is None:
            return web.json_response({"error": "Jobs not available."}, status=503)
        job_id = request.match_info["job_id"]
        stage = request.match_info["stage"]
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        if not isinstance(payload, dict) or "value" not in payload:
            return web.Response(
                status=400, text="Body must be a JSON object containing 'value'"
            )
        try:
            job = self.job_manager.set_stage_output(
                job_id,
                stage,
                payload["value"],
                actor=payload.get("actor"),
                pass_number=_pass_number(payload),
            )
        except KeyError:
            return web.json_response(
                {"error": f"Unknown job or stage '{stage}'"}, status=404
            )
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        logger.info(
            "Analyst overrode stage %s output for job %s (actor=%s)",
            stage,
            job_id,
            payload.get("actor"),
        )
        return web.json_response(job.snapshot())

    # -- the retrieval plan, edited by name --------------------------------

    def _plan_context(self, request):
        """``(job, generator, analysis, queries)`` for the query-plan endpoints.

        Raises ``_PlanUnavailable`` carrying the response to send, so both handlers refuse
        for the same reasons in the same words.
        """
        if self.job_manager is None:
            raise _PlanUnavailable(
                web.json_response({"error": "Jobs not available."}, status=503)
            )
        job = self._lookup_job(request)
        if job is None:
            raise _PlanUnavailable(
                web.json_response({"error": "Job not found"}, status=404)
            )
        generator = (getattr(self.job_manager, "modules", None) or {}).get("api_call")
        if generator is None or not hasattr(generator, "dependency_report"):
            raise _PlanUnavailable(
                web.json_response(
                    {"error": "Query planning is not available on this server instance."},
                    status=503,
                )
            )
        outputs = getattr(job.context, "outputs", None) or {}
        understanding = outputs.get("understanding")
        analysis = getattr(understanding, "analysis", None)
        if analysis is None:
            # 409 rather than an empty list, which would read as a planner that chose
            # nothing instead of a plan that does not exist yet.
            raise _PlanUnavailable(
                web.json_response(
                    {
                        "error": "This job has no understanding yet, so it has no "
                        "retrieval plan to edit."
                    },
                    status=409,
                )
            )
        queries = list(outputs.get("queries") or [])
        return job, generator, analysis, queries

    async def get_job_queries(self, request):
        """``GET /api/v1/jobs/{job_id}/queries``: the retrieval plan and what it skipped.

        ``{queries, unselected, dependencies, row_counts}``. Each query carries the ``index``
        POST removes by and the entity types it is scoped by, filtered to the ones its source
        can bind. ``unselected`` is the addable menu, unmet dependencies first.
        """
        try:
            job, generator, analysis, queries = self._plan_context(request)
        except _PlanUnavailable as stop:
            return stop.response
        logs = (getattr(job.context, "outputs", None) or {}).get("logs") or {}
        return web.json_response(
            {
                "job_id": job.job_id,
                "pass": job.current_pass,
                "queries": [
                    {
                        "index": i,
                        "source": q.target_log_source,
                        "question": q.natural_language_query,
                        "date_from": q.date_from,
                        "date_to": q.date_to,
                        "scoped_by": sorted(
                            {
                                e.type
                                for e in (q.entities or [])
                                if getattr(e, "type", "") and e.type != "time_window"
                            }
                        ),
                    }
                    for i, q in enumerate(queries)
                ],
                "unselected": generator.unselected_sources(queries, analysis),
                "dependencies": generator.dependency_report(queries, None, analysis),
                "row_counts": {
                    k: len(v or []) for k, v in logs.items() if isinstance(logs, dict)
                },
            }
        )

    async def edit_job_queries(self, request):
        """``POST /api/v1/jobs/{job_id}/queries``: add and remove whole queries.

        Body ``{"add": [{"source", "question"}], "remove": [<index>], "actor"}``, both keys
        optional. All-or-nothing, with removals resolved against the indices the GET returned
        before any addition is appended. Applied through ``set_stage_output``, so it lands in
        ``interventions``; continue with ``skip_stage``, since a retry re-plans.
        """
        try:
            job, generator, analysis, queries = self._plan_context(request)
        except _PlanUnavailable as stop:
            return stop.response
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        if not isinstance(payload, dict):
            return web.Response(status=400, text="Body must be a JSON object")
        additions = payload.get("add") or []
        removals = payload.get("remove") or []
        if not isinstance(additions, list) or not isinstance(removals, list):
            return web.Response(status=400, text="'add' and 'remove' must be arrays")
        if not additions and not removals:
            return web.json_response(
                {"error": "Nothing to do: supply 'add' and/or 'remove'"}, status=400
            )

        # Removals first, by index against the list the client read.
        drop = set()
        for raw in removals:
            try:
                index = int(raw)
            except (TypeError, ValueError):
                return web.json_response(
                    {"error": f"'remove' takes query indices; got {raw!r}"}, status=400
                )
            if not 0 <= index < len(queries):
                return web.json_response(
                    {
                        "error": f"No query at index {index}; the plan holds "
                        f"{len(queries)}. Re-read GET /queries — it may have changed."
                    },
                    status=409,
                )
            drop.add(index)
        kept = [q for i, q in enumerate(queries) if i not in drop]
        removed = [queries[i].target_log_source for i in sorted(drop)]

        # Then the additions, each built and scoped by the generator itself.
        window = next(
            ((q.date_from, q.date_to) for q in queries if q.date_from and q.date_to),
            None,
        )
        added = []
        for item in additions:
            if not isinstance(item, dict):
                return web.json_response(
                    {"error": "each 'add' entry must be an object with a 'source'"},
                    status=400,
                )
            try:
                query = generator.build_manual_query(
                    analysis,
                    item.get("source"),
                    question=item.get("question") or "",
                    window=window,
                )
            except ValueError as e:
                # The operator named a source that cannot be queried. Refused whole, so
                # nothing is applied.
                return web.json_response({"error": str(e)}, status=400)
            kept.append(query)
            added.append(query.target_log_source)

        value = [q.model_dump(mode="json") for q in kept]
        detail = "; ".join(
            part
            for part in (
                f"added {', '.join(added)}" if added else "",
                f"removed {', '.join(removed)}" if removed else "",
            )
            if part
        )
        try:
            job = self.job_manager.set_stage_output(
                job.job_id,
                "query_generation",
                value,
                actor=payload.get("actor"),
                pass_number=_pass_number(payload),
                detail=f"retrieval plan edited by analyst: {detail}",
            )
        except KeyError:
            return web.json_response({"error": "Job not found"}, status=404)
        except ValueError as e:
            # Includes "stage is running", where planning would overwrite the edit the moment
            # it returned and the edit would read as having vanished.
            return web.json_response({"error": str(e)}, status=409)
        await self.job_manager.flush_persistence(job.job_id)
        logger.info(
            "Analyst edited the retrieval plan for job %s: %s (actor=%s)",
            job.job_id,
            detail,
            payload.get("actor"),
        )
        return web.json_response(
            {
                "job_id": job.job_id,
                "added": added,
                "removed": removed,
                "queries": len(kept),
                "dependencies": generator.dependency_report(kept, None, analysis),
            }
        )

    # -- the advisory lane, acted on by naming an index ---------------------

    async def get_job_links(self, request):
        """``GET /api/v1/jobs/{job_id}/links``: what may be spent, the mode in force, on what.

        ``{escalation, job_modes, procedures, link_count, correlated}``. Answers before
        correlation has run, which is why the pack's procedure names are here: the POST below
        needs one to set a mode against.
        """
        if self.job_manager is None:
            return web.json_response({"error": "Jobs not available."}, status=503)
        job = self._lookup_job(request)
        if job is None:
            return web.json_response({"error": "Job not found"}, status=404)

        cfg = self._link_config()
        outputs = getattr(job.context, "outputs", None) or {}
        correlation = outputs.get("correlation")
        links = list(getattr(correlation, "links", None) or [])

        # Through the engine's own resolvers because each clamps: reporting the configured
        # number would promise a budget the run will not grant.
        probes = link_escalation.probe_budget(cfg)
        # The manager's own reader, never a second one: what matters is the width of the
        # semaphore the children contend on, and the manager is what holds it.
        llm_width = None
        if hasattr(self.job_manager, "_llm_concurrency"):
            try:
                llm_width = self.job_manager._llm_concurrency()
            except Exception:  # pragma: no cover - a read must not fail the endpoint
                llm_width = None
        children = link_children.child_budget(cfg, llm_concurrency=llm_width)

        pack = getattr(
            (getattr(self.job_manager, "modules", None) or {}).get("correlation"),
            "knowledge_pack",
            None,
        )
        procedures = []
        if pack is not None and hasattr(pack, "ruleset_keys"):
            try:
                procedures = [str(k) for k in (pack.ruleset_keys() or [])]
            except Exception:  # pragma: no cover - a read must not fail the endpoint
                procedures = []

        return web.json_response(
            {
                "job_id": job.job_id,
                "correlated": correlation is not None,
                "link_count": len(links),
                "job_modes": dict(job.context.link_modes or {}),
                "procedures": procedures,
                "escalation": {
                    "modes": list(link_escalation.LINK_MODES),
                    "engine_default": link_escalation.DEFAULT_LINK_MODE,
                    # "" where the deployment asked for nothing: a default in force and a
                    # chosen value are different answers.
                    "config_mode": link_escalation.normalise_mode(
                        cfg.get("escalation_mode")
                    ),
                    "min_escalation_score": link_escalation.min_escalation_score(cfg),
                    # Whether the lane can spend at all, derived from the caps rather than
                    # asserted. Which rung is the per-rung flags below.
                    "escalation_budgeted": link_escalation.escalation_budgeted(cfg),
                    "probes_budgeted": probes["max_probes"] > 0,
                    "max_probes_per_run": probes["max_probes"],
                    "probe_timeout_seconds": probes["timeout"],
                    "probe_deadline_seconds": probes["deadline_seconds"],
                    "probe_row_cap": link_escalation.probe_row_cap(cfg),
                    # Rung 4 has its own permit, so a run can reach a probe and stop there:
                    # one number cannot say which rung is disarmed.
                    "max_children_per_run": children["max_children"],
                    "max_child_depth": children["max_depth"],
                    "max_concurrent_children": children["max_concurrent"],
                    "max_total_children": children["max_total"],
                    "children_budgeted": children["max_children"] > 0
                    and children["max_depth"] > 0,
                },
            }
        )

    async def refer_job_link(self, request):
        """``POST /api/v1/jobs/{job_id}/links/{index}/refer``: compose a child run, launch none.

        Optional body ``{"mode", "actor", "launched_job_id"}`` → the request to POST to
        ``/api/v1/jobs``, its pivot, the direction-derived window and the advisory pin.
        Composed server-side because a link's scope is pack knowledge. ``launched_job_id``
        records a launch made elsewhere, refused if the registry does not know it.
        """
        if self.job_manager is None:
            return web.json_response({"error": "Jobs not available."}, status=503)
        job = self._lookup_job(request)
        if job is None:
            return web.json_response({"error": "Job not found"}, status=404)
        outputs = getattr(job.context, "outputs", None) or {}
        correlation = outputs.get("correlation")
        if correlation is None:
            return web.json_response(
                {
                    "error": "This job has not correlated yet, so it has no cross-procedure "
                    "links to refer."
                },
                status=409,
            )
        links = list(getattr(correlation, "links", None) or [])
        try:
            index = int(request.match_info["index"])
        except (TypeError, ValueError):
            return web.json_response(
                {"error": "The link handle is its index in the correlation output."},
                status=400,
            )
        if not 0 <= index < len(links):
            # 409 and not 404, as in the plan editor: a rejected gate re-runs correlation, so
            # an index read a minute ago can be stale rather than wrong.
            return web.json_response(
                {
                    "error": f"No link at index {index}; this run holds {len(links)}. "
                    "Re-read GET /api/v1/jobs/{job_id} — correlation may have re-run."
                },
                status=409,
            )
        finding = links[index]

        raw = await request.text()
        payload = {}
        if (raw or "").strip():
            try:
                payload = await request.json()
            except json.JSONDecodeError:
                return web.Response(status=400, text="Invalid JSON")
            if not isinstance(payload, dict):
                return web.Response(status=400, text="Body must be a JSON object")

        # The window is the direction, so it comes from the planner's own arithmetic rather
        # than a second reading of `lookback:<N>d` written here.
        try:
            _job, generator, analysis, queries = self._plan_context(request)
        except _PlanUnavailable as stop:
            return stop.response
        if not hasattr(generator, "referral_window"):
            return web.json_response(
                {
                    "error": "Referral windows are not available on this server instance, and "
                    "an unscoped window would make the referral a full-window scan."
                },
                status=503,
            )
        date_from, date_to, resolved = generator.referral_window(
            analysis, queries, getattr(finding, "window_hint", "")
        )
        try:
            composed = compose_referral(
                finding,
                parent_job_id=job.job_id,
                parent_incident_id=str(job.incident.get("id") or ""),
                window=(date_from, date_to),
                mode=str(payload.get("mode") or job.run_mode.value),
            )
        except ValueError as e:
            # 409 rather than 400: the request is well formed and the state is not.
            return web.json_response({"error": str(e)}, status=409)
        composed["scope"]["window_applied"] = resolved

        target = composed["pin"]["use_case"] or "an unnamed procedure"
        launched_id = str(payload.get("launched_job_id", "") or "").strip()
        if launched_id:
            child = self.job_manager.get_job(launched_id)
            if child is None:
                return web.json_response(
                    {
                        "error": f"No job {launched_id} to record a launch for. The trail is "
                        "read later to answer what a human did, so an id nothing resolves is "
                        "refused rather than written."
                    },
                    status=409,
                )
            # Beside the composition, not instead of it: two acts by possibly two people,
            # and composed-but-not-launched is most of them.
            job.record_intervention(
                "link_referral_launched",
                stage="correlation",
                detail=(
                    f"referral to {target} launched by hand as job {launched_id} — the child "
                    "has its own verdict, and this run's was settled before it started"
                ),
                actor=payload.get("actor"),
            )
        else:
            # "composed" and never "referred": a word implying a child run exists sends the
            # reader of this trail looking for one.
            job.record_intervention(
                "link_referral_composed",
                stage="correlation",
                detail=(
                    f"referral composed for {target} ({getattr(finding, 'state', '')}, "
                    f"{getattr(finding, 'direction', '') or 'no direction'}) via "
                    f"{composed['scope']['pivot_entity']} "
                    f"{', '.join(composed['scope']['pivot_values'])}, window {resolved} "
                    f"{date_from or 'unbounded'}..{date_to or 'unbounded'} — not launched"
                ),
                actor=payload.get("actor"),
            )
        await self.job_manager.flush_persistence(job.job_id)
        logger.info(
            "%s a link referral to %s from job %s (index %s, actor=%s)%s",
            "Recorded the hand-launch of" if launched_id else "Composed",
            target,
            job.job_id,
            index,
            payload.get("actor"),
            f" as job {launched_id}" if launched_id else "; not launched",
        )
        composed.update(
            {
                "job_id": job.job_id,
                "index": index,
                "link": {
                    "state": str(getattr(finding, "state", "") or ""),
                    "rung": getattr(finding, "rung", 0),
                    "advisory_severity": str(
                        getattr(finding, "advisory_severity", "") or ""
                    ),
                },
                # Whatever the body carried: this field claims what the endpoint did, and a
                # launch it merely recorded is `launch_recorded`, the weaker claim.
                "launched": False,
                "launch_recorded": launched_id,
            }
        )
        return web.json_response(composed)

    async def set_job_link_modes(self, request):
        """``POST /api/v1/jobs/{job_id}/links/mode``: set one or more links' escalation mode.

        Body ``{"mode": "planned|semi_auto|auto", "targets": ["<use case>"], "actor"}``;
        ``targets`` omitted means every link this run holds. All-or-nothing, resolved before
        anything is applied, and recorded on the job so a re-run resolves the same mode. The
        response carries the mode that took effect per target, which an unheld scope gate can
        hold at ``planned``, not the ask.
        """
        if self.job_manager is None:
            return web.json_response({"error": "Jobs not available."}, status=503)
        job = self._lookup_job(request)
        if job is None:
            return web.json_response({"error": "Job not found"}, status=404)

        raw = await request.text()
        if not (raw or "").strip():
            return web.Response(status=400, text="Body must name a mode")
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        if not isinstance(payload, dict):
            return web.Response(status=400, text="Body must be a JSON object")

        mode = link_escalation.normalise_mode(payload.get("mode"))
        if not mode:
            # Named rather than described: a closed vocabulary, and a typo is the likeliest
            # reason to be here.
            return web.json_response(
                {
                    "error": "`mode` must be one of: "
                    + ", ".join(link_escalation.LINK_MODES)
                },
                status=400,
            )

        outputs = getattr(job.context, "outputs", None) or {}
        correlation = outputs.get("correlation")
        links = list(getattr(correlation, "links", None) or [])
        by_target = {}
        for finding in links:
            by_target.setdefault(str(getattr(finding, "target_use_case", "") or ""), [])
            by_target[str(getattr(finding, "target_use_case", "") or "")].append(
                finding
            )

        asked = payload.get("targets")
        if asked is None:
            targets = [t for t in by_target if t]
            if not targets:
                # 409 and not 404: naming the targets explicitly is a legitimate way to set
                # the mode in advance, so this refuses the enumeration and says what works.
                return web.json_response(
                    {
                        "error": "This job holds no cross-procedure links to set a mode on yet. "
                        "Name the target procedures in `targets` to set the mode before "
                        "correlation runs."
                    },
                    status=409,
                )
        elif not isinstance(asked, list) or not all(
            isinstance(t, str) and t.strip() for t in asked
        ):
            return web.json_response(
                {"error": "`targets` must be a list of target procedure names."},
                status=400,
            )
        else:
            targets = [t.strip() for t in asked]

        # Against the pack, not the link list: a target with no link yet may still be a real
        # procedure. Skipped, and reported as skipped, where the pack is unreachable.
        pack = getattr(
            (getattr(self.job_manager, "modules", None) or {}).get("correlation"),
            "knowledge_pack",
            None,
        )
        checked_against_pack = pack is not None and hasattr(pack, "ruleset_spec")
        if checked_against_pack:
            unknown = [t for t in targets if not (pack.ruleset_spec(t) or {})]
            if unknown:
                return web.json_response(
                    {
                        "error": "No procedure named "
                        + ", ".join(sorted(unknown))
                        + " in the loaded knowledge pack."
                    },
                    status=400,
                )

        cfg = self._link_config()
        job.context.link_modes.update({t: mode for t in targets})
        applied = []
        for target in targets:
            findings = by_target.get(target, [])
            for finding in findings:
                effective = link_escalation.apply_link_mode(
                    finding,
                    config=cfg,
                    overrides={target: mode},
                )
            if not findings:
                # Set in advance. Still resolved through the same seam rather than echoed
                # back, so an `auto` with no candidate does not report as armed.
                effective = link_escalation.resolve_link_mode(
                    config_mode=cfg.get("escalation_mode"),
                    job_mode=mode,
                    gate_outcome=link_escalation.NO_CANDIDATE,
                    escalation_available=link_escalation.escalation_budgeted(cfg),
                    # None skips the score gate; 0.0 would refuse the setting outright.
                    score=None,
                )
            applied.append(
                {
                    "target_use_case": target,
                    "asked": mode,
                    "mode": effective["mode"],
                    "mode_source": effective["source"],
                    "mode_note": effective["note"],
                    "proposed_action": effective["action"],
                    "links": len(by_target.get(target, [])),
                }
            )

        held = [a["target_use_case"] for a in applied if a["mode"] != mode]
        job.record_intervention(
            "link_mode_set",
            stage="correlation",
            detail=(
                f"escalation mode '{mode}' set for {', '.join(targets)}"
                + (
                    # `MANUAL_LINK_MODE`, not `DEFAULT_LINK_MODE`: the clamp and the score
                    # gate land on `planned`, and the default is an escalating mode.
                    f" — held at '{link_escalation.MANUAL_LINK_MODE}' for "
                    f"{', '.join(held)}, each with its reason in its own mode_note"
                    if held
                    else ""
                )
            ),
            actor=payload.get("actor"),
        )
        await self.job_manager.flush_persistence(job.job_id)
        logger.info(
            "Link escalation mode '%s' set on job %s for %s (actor=%s); %s",
            mode,
            job.job_id,
            ", ".join(targets),
            payload.get("actor"),
            (
                f"held at '{link_escalation.MANUAL_LINK_MODE}' for {', '.join(held)}"
                if held
                else "licensed on every target"
            ),
        )
        return web.json_response(
            {
                "job_id": job.job_id,
                "mode": mode,
                "applied": applied,
                # Echoed from the job rather than the findings: this is what the next
                # correlation will read.
                "link_modes": dict(job.context.link_modes),
                "targets_checked_against_pack": checked_against_pack,
            }
        )

    async def export_job(self, request):
        if self.job_manager is None:
            return web.json_response({"error": "Jobs not available."}, status=503)
        job_id = request.match_info["job_id"]
        try:
            doc = self.job_manager.export_job(job_id)
        except KeyError:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response(doc)

    async def import_job(self, request):
        if self.job_manager is None:
            return web.json_response({"error": "Jobs not available."}, status=503)
        try:
            doc = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        try:
            job = self.job_manager.import_job(doc)
        except Exception as e:  # noqa: BLE001 — malformed export doc
            logger.error("Job import failed: %s", e)
            return web.json_response({"error": f"Import failed: {e}"}, status=400)
        return web.json_response({"job_id": job.job_id}, status=201)

    def _lookup_job(self, request):
        if self.job_manager is None:
            return None
        return self.job_manager.get_job(request.match_info["job_id"])

    def _link_config(self):
        """The ``correlation.links`` block, read from the whole config and not from this slice.

        ``self.config`` is ``main_config["incident_input"]`` and carries no ``correlation``
        key, so reading it there reports a budget of zero against a configured one.
        ``live_config`` is the full dict by reference, which is what makes a ``live`` edit
        visible; the fallback covers a caller handed a whole config as its slice.
        """
        for source in (self.live_config, self.config):
            if not isinstance(source, dict):
                continue
            block = (source.get("correlation") or {}).get("links")
            if isinstance(block, dict):
                return block
        return {}

    # -- artifacts: report + evidence --------------------------------------

    def _incident_id_for_job(self, request):
        """The incident id behind a job id, or None (a 404) when the job is unknown.

        Artifacts are named after the incident, so a job-scoped URL has to translate; a
        fallback to the job id would serve a mistyped one something.
        """
        job = self._lookup_job(request)
        if job is None:
            return None
        return (job.incident or {}).get("id")

    @staticmethod
    def _artifact_response(body, content_type, filename, download):
        """One response shape for every artifact; ``?download=1`` makes it an attachment."""
        headers = {}
        if download:
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        else:
            headers["Content-Disposition"] = f'inline; filename="{filename}"'
        return web.Response(body=body, content_type=content_type, headers=headers)

    async def _serve_report(self, incident_id, request, report=None):
        fmt = (request.query.get("format") or "html").strip().lower()
        try:
            body, content_type, filename = report_delivery.resolve_report(
                incident_id, fmt, report=report
            )
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        except FileNotFoundError as e:
            # Carries the reason so the UI can tell "not finished yet" from "the renderer
            # broke": the report write is best-effort.
            return web.json_response(
                {"error": str(e), "incident_id": incident_id, "format": fmt}, status=404
            )
        except OSError as e:
            logger.error("Report read failed for %s: %s", incident_id, e)
            return web.json_response(
                {"error": "Could not read the report."}, status=500
            )
        return self._artifact_response(
            body, content_type, filename, _truthy(request.query.get("download"))
        )

    async def _serve_evidence(self, incident_id, request):
        kind = (request.query.get("kind") or "raw").strip().lower()
        if kind not in report_delivery.EVIDENCE_KINDS:
            return web.json_response(
                {
                    "error": f"unknown evidence kind '{kind}'; expected one of "
                    + ", ".join(report_delivery.EVIDENCE_KINDS)
                },
                status=400,
            )
        # The bounded outline by default: raw evidence is every retrieved row and runs to
        # tens of MB. `?full=1` or `?download=1` serves the artifact itself.
        full = _truthy(request.query.get("full")) or _truthy(
            request.query.get("download")
        )
        try:
            if full:
                body, content_type, filename = report_delivery.resolve_evidence(
                    incident_id, kind
                )
                return self._artifact_response(
                    body,
                    content_type,
                    filename,
                    _truthy(request.query.get("download")),
                )
            outline = report_delivery.evidence_outline(incident_id, kind)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        except FileNotFoundError as e:
            return web.json_response(
                {"error": str(e), "incident_id": incident_id, "kind": kind}, status=404
            )
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Evidence read failed for %s (%s): %s", incident_id, kind, e)
            return web.json_response(
                {"error": "Could not read the evidence artifact."}, status=500
            )
        return web.json_response(outline)

    async def get_job_report(self, request):
        """``GET /api/v1/jobs/{job_id}/report?format=html|md|pdf|json[&download=1]``."""
        incident_id = self._incident_id_for_job(request)
        if incident_id is None:
            return web.json_response({"error": "Job not found"}, status=404)
        job = self._lookup_job(request)
        # The in-memory report where there is one, so a finished-but-not-yet-exported job can
        # answer; otherwise this falls through to the files on disk.
        ctx = getattr(job, "context", None)
        outputs = getattr(ctx, "outputs", None) or {}
        return await self._serve_report(
            incident_id, request, report=outputs.get("report")
        )

    async def get_job_evidence(self, request):
        """``GET /api/v1/jobs/{job_id}/evidence?kind=raw|transformed[&full=1]``."""
        incident_id = self._incident_id_for_job(request)
        if incident_id is None:
            return web.json_response({"error": "Job not found"}, status=404)
        return await self._serve_evidence(incident_id, request)

    async def get_job_artifacts(self, request):
        incident_id = self._incident_id_for_job(request)
        if incident_id is None:
            return web.json_response({"error": "Job not found"}, status=404)
        try:
            return web.json_response(report_delivery.artifact_inventory(incident_id))
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)

    async def get_incident_report(self, request):
        """Same as the job-scoped report, keyed by incident id (survives a restart)."""
        return await self._serve_report(request.match_info["incident_id"], request)

    async def get_incident_evidence(self, request):
        return await self._serve_evidence(request.match_info["incident_id"], request)

    async def get_incident_artifacts(self, request):
        try:
            return web.json_response(
                report_delivery.artifact_inventory(request.match_info["incident_id"])
            )
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)

    # -- configuration -----------------------------------------------------

    async def get_config(self, request):
        """The whole editable configuration, with literal secrets redacted.

        A literal value under a secret-shaped key comes back as the placeholder; an
        ``${ENV_VAR}`` reference comes back verbatim, being a name the operator needs. This
        API is unauthenticated, so ``config_store``'s redaction — which covers the raw text
        as well as this structured view — is all that stands between it and the credentials.
        """
        try:
            return web.json_response(config_store.describe())
        except Exception as e:  # noqa: BLE001 — an unreadable config must not 500 blind
            logger.error("Config read failed: %s", e)
            return web.json_response(
                {"error": f"Could not read the configuration: {e}"}, status=500
            )

    async def patch_config(self, request):
        """``PUT /api/v1/config`` with ``{"updates": {"<dotted.path>": value, ...}}``.

        All-or-nothing against the field descriptors: a half-landed write leaves the app in a
        state nobody chose. A value matching the redaction placeholder is dropped as
        "unchanged", so a UI that PUTs back what it read cannot overwrite a password with it.
        """
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        if not isinstance(payload, dict) or not isinstance(
            payload.get("updates"), dict
        ):
            return web.Response(
                status=400, text='Body must be {"updates": {"<path>": value, ...}}'
            )
        accepted, errors = config_store.validate_updates(payload["updates"])
        if errors:
            return web.json_response(
                {"error": "Validation failed", "errors": errors}, status=400
            )
        if not accepted:
            return web.json_response(
                {"changed": [], "skipped": [], "note": "nothing to change"}
            )
        try:
            result = config_store.apply_updates(accepted)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        except OSError as e:
            logger.error("Config write failed: %s", e)
            return web.json_response(
                {"error": f"Could not write the configuration: {e}"}, status=500
            )
        # Modules hold their slice by reference, so re-reading the file into the live dict
        # is what makes a `live` field take effect.
        result["reloaded"] = self._reload_live_config(result.get("changed") or [])
        return web.json_response(result)

    def _reload_live_config(self, changed):
        """Re-read main_config into the live dict for fields marked ``live``.

        Returns the paths that took effect; one the running process cannot pick up comes back
        as ``restart_required``. Best-effort — a reload failure leaves the file written.
        """
        live = [item["path"] for item in changed if item.get("applies") == "live"]
        restart = [item["path"] for item in changed if item.get("applies") != "live"]
        applied = []
        # A separate file with a separate consumer: the live dict below is main_config.
        applied.extend(self._reload_llm_thinking(live))
        if live and isinstance(self.live_config, dict):
            try:
                fresh = config_store.read_file("main_config.yaml")
                for path in live:
                    if config_store.FIELDS[path].file != "main_config.yaml":
                        continue
                    value = config_store.get_path(fresh, path)
                    if self._set_live(path, value):
                        applied.append(path)
                if any(p.startswith("logging.") for p in applied):
                    # It removes only its own handlers, so re-invoking swaps format and
                    # level in place rather than doubling every line.
                    from src.utils.logging_setup import configure_logging

                    configure_logging(self.live_config)
                # Copied onto each retriever at build time, so the module has to re-resolve
                # it, and it counts as applied only if that ran.
                if "log_sources.primary_source_timeout_seconds" in applied:
                    if not self._refresh_retrieval_budgets():
                        applied.remove("log_sources.primary_source_timeout_seconds")
                        restart.append("log_sources.primary_source_timeout_seconds")
                # Same rule, different owner: the bound is a module global the summarizers
                # read per call, so they never see the config dict.
                if "jobs.summary_max_items" in applied:
                    if not self._refresh_summary_bound():
                        applied.remove("jobs.summary_max_items")
                        restart.append("jobs.summary_max_items")
            except Exception as e:  # noqa: BLE001 — never fail the write on a reload
                logger.warning("Live config reload skipped: %s", e)
        return {"applied": applied, "restart_required": restart}

    def _reload_llm_thinking(self, live):
        """Push freshly written thinking settings onto the shared LLMClient.

        Returns the paths that took effect, so an unwired deployment reports them as
        restart-required rather than claiming an apply that did not happen. Only the thinking
        keys are live in ``llm_config.yaml``, being the only ones read per call rather than
        baked into the client or its throttles at construction.
        """
        paths = [
            p
            for p in live
            if config_store.FIELDS[p].file == "llm_config.yaml"
            and (p == "thinking" or p.startswith("thinking_"))
        ]
        if not paths:
            return []
        client = getattr(self, "llm_client", None)
        if client is None or not hasattr(client, "apply_thinking_config"):
            return []
        try:
            client.apply_thinking_config(config_store.read_file("llm_config.yaml"))
        except Exception as e:  # noqa: BLE001 — a reload must never fail the write
            logger.warning("Extended-thinking reload skipped: %s", e)
            return []
        return paths

    def _refresh_retrieval_budgets(self):
        """Ask the engine to re-resolve its primary-source budgets. True if it did.

        Through ``on_live_reload`` rather than a direct engine reference, which would let the
        config editor reach further into the pipeline than it needs.
        """
        hook = getattr(self, "on_live_reload", None)
        if hook is None:
            return False
        try:
            hook()
            return True
        except Exception as e:  # noqa: BLE001 — a refresh must never fail the write
            logger.warning("Retrieval budget refresh skipped: %s", e)
            return False

    def _refresh_summary_bound(self):
        """Push ``jobs.summary_max_items`` onto the summarizers' module global. True if set.

        Imported here rather than at module scope: ``pipeline_runner`` pulls the whole stage
        graph, and this server is constructed in tests that wire no job machinery.
        """
        try:
            from src.pipeline_runner import configure_summaries

            configure_summaries(self.live_config)
            return True
        except Exception as e:  # noqa: BLE001 — a refresh must never fail the write
            logger.warning("Summary bound refresh skipped: %s", e)
            return False

    def _set_live(self, path, value):
        """Write one dotted path into the live config dict, if every segment already exists.

        False for an unreachable path rather than creating a key nobody reads.
        """
        holder = getattr(self, "live_config", None)
        if not isinstance(holder, dict):
            return False
        parts = path.split(".")
        node = holder
        for part in parts[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
            if not isinstance(node, dict):
                return False
        if parts[-1] not in node:
            return False
        node[parts[-1]] = value
        return True

    async def export_config_file(self, request):
        """``GET /api/v1/config/{name}``: one file's raw text, secrets redacted.

        The redaction makes an export safe to share as a template, and the importer refuses
        text still carrying the placeholder rather than writing it over a real credential.
        """
        name = request.match_info["name"]
        if name not in config_store.CONFIG_FILES:
            return web.json_response(
                {"error": f"unknown config file '{name}'"}, status=404
            )
        try:
            text = config_store.redact_raw(config_store.read_raw(name))
        except OSError as e:
            return web.json_response({"error": str(e)}, status=500)
        if not text:
            return web.json_response(
                {"error": f"{name} does not exist in the config directory"}, status=404
            )
        return self._artifact_response(
            text.encode("utf-8"),
            "text/yaml",
            name,
            _truthy(request.query.get("download")),
        )

    async def replace_config_file(self, request):
        """``PUT /api/v1/config/{name}`` with ``{"text": "<yaml>"}``: whole-file save.

        Parses the candidate text before replacing anything, and keeps the previous version
        as ``<name>.yaml.bak``.
        """
        name = request.match_info["name"]
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            return web.Response(status=400, text='Body must be {"text": "<yaml>"}')
        try:
            result = config_store.replace_file(name, payload["text"])
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        except OSError as e:
            logger.error("Config replace failed for %s: %s", name, e)
            return web.json_response({"error": str(e)}, status=500)
        result["restart_required"] = True
        return web.json_response(result)

    async def import_config(self, request):
        """``POST /api/v1/config/import`` with ``{"files": {"<name>": "<yaml>"}}``.

        Every file is validated before any is written: a matched pair where only the first
        parses would leave the app running half of somebody's environment.
        """
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        files = (payload or {}).get("files") if isinstance(payload, dict) else None
        if not isinstance(files, dict) or not files:
            return web.Response(
                status=400,
                text='Body must be {"files": {"<config>.yaml": "<yaml text>"}}',
            )
        errors = []
        for name, text in files.items():
            if name not in config_store.CONFIG_FILES:
                errors.append(f"{name}: not a recognised config file")
            elif not isinstance(text, str):
                errors.append(f"{name}: contents must be a string")
            elif config_store.REDACTED in text:
                errors.append(
                    f"{name}: still contains {config_store.REDACTED} — replace those "
                    "placeholders with real values or ${ENV_VAR} references"
                )
            else:
                try:
                    parsed = config_store.yaml.safe_load(text)
                except Exception as e:  # noqa: BLE001 — YAMLError and friends
                    errors.append(f"{name}: invalid YAML — {e}")
                else:
                    # `replace_file`'s own rule, checked here too so a non-mapping file
                    # cannot fail mid-write and break the validate-all-first promise.
                    if parsed is not None and not isinstance(parsed, dict):
                        errors.append(
                            f"{name}: a config file must be a YAML mapping at the "
                            "top level"
                        )
        if errors:
            return web.json_response(
                {"error": "Import rejected; nothing was written", "errors": errors},
                status=400,
            )
        written = []
        for name, text in files.items():
            try:
                written.append(config_store.replace_file(name, text))
            except (ValueError, OSError) as e:
                # Pre-validated above, so this is an I/O fault; report what did land.
                logger.error("Config import failed on %s: %s", name, e)
                return web.json_response(
                    {"error": f"{name}: {e}", "written": written}, status=500
                )
        logger.info("Configuration imported: %s", ", ".join(files))
        return web.json_response({"written": written, "restart_required": True})

    # -- knowledge packs ---------------------------------------------------
    #
    # Each mutating handler is a thin shell over one `pack_store` function, which owns the
    # verifications, the snapshot-before-write and the anchored line replacement. A destructive
    # intent is never inferred from the request shape: a delete needs `confirm=1`, and create is
    # a POST while save is a PUT, so a mistyped path on a save 404s. Unauthenticated like the
    # rest of this API, so the bound is `pack_store`'s path guard and suffix allowlist.

    def _pack_error_response(self, exc):
        """One mapping from a store refusal to a status code, shared by every handler here.

        `PackTooLarge` is 413 and not 400: the request was well-formed, and the next move is
        the download the message offers.
        """
        if isinstance(exc, pack_store.PackTooLarge):
            return web.json_response({"error": str(exc)}, status=413)
        if isinstance(exc, pack_store.PackConflict):
            return web.json_response({"error": str(exc)}, status=409)
        if isinstance(exc, (pack_store.PackNotFound, pack_store.PackFileNotFound)):
            return web.json_response({"error": str(exc)}, status=404)
        if isinstance(exc, pack_store.PackFileExists):
            return web.json_response({"error": str(exc)}, status=409)
        if isinstance(exc, pack_store.PackWriteRejected):
            # The parser's own message and line: "does not parse" alone sends the operator
            # hunting for what PyYAML had already located exactly.
            return web.json_response(
                {"error": str(exc), "path": exc.path, "line": exc.line}, status=400
            )
        return web.json_response({"error": str(exc)}, status=400)

    async def list_knowledge_packs(self, request):
        """``GET /api/v1/knowledge``: every installed pack, and which one is loaded.

        Editing a pack does not hot-reload it, so ``loaded`` is how the editor can say whether
        this is the pack the running process uses.
        """
        try:
            packs = pack_store.list_packs()
        except OSError as e:
            logger.error("Pack listing failed: %s", e)
            return web.json_response(
                {"error": f"Could not read the knowledge directory: {e}"}, status=500
            )
        loaded = ""
        holder = getattr(self, "live_config", None)
        if isinstance(holder, dict):
            configured = ((holder.get("knowledge") or {}).get("pack_dir")) or ""
            # The config holds a path and this endpoint speaks in names, so compare on the
            # basename: "knowledge/x" and "/srv/knowledge/x" are one entry.
            loaded = str(configured).rstrip("/").rsplit("/", 1)[-1]
        for entry in packs:
            entry["loaded"] = entry["name"] == loaded
        return web.json_response(
            {"packs": packs, "loaded": loaded, "root": str(pack_store.packs_root())}
        )

    async def get_knowledge_pack(self, request):
        """``GET /api/v1/knowledge/{pack}``: the tree and the diagnostics in one payload.

        One payload rather than two round trips, which would show a clean tree in between —
        and a pack whose catalog does not parse looks normal in a file listing.
        """
        pack = request.match_info["pack"]
        try:
            tree = pack_store.pack_tree(pack)
        except pack_store.PackStoreError as e:
            return self._pack_error_response(e)
        except OSError as e:
            logger.error("Pack read failed for %s: %s", pack, e)
            return web.json_response({"error": str(e)}, status=500)
        payload = {
            "pack": tree["pack"],
            "nodes": tree["nodes"],
            "counts": tree["counts"],
            "limits": {
                "inline_edit_max_bytes": pack_store.INLINE_EDIT_MAX_BYTES,
                "read_max_bytes": pack_store.READ_MAX_BYTES,
                "write_max_bytes": pack_store.WRITE_MAX_BYTES,
                "editable_suffixes": sorted(pack_store.EDITABLE_SUFFIXES),
            },
        }
        payload["validate"] = self._validate_payload(pack)
        return web.json_response(payload)

    def _validate_payload(self, pack):
        """Diagnostics for one pack, or a diagnostic saying why there are none.

        Never raises: the editor has to stay usable on a pack that is already broken, so a
        failure inside the lint is reported in the shape of a finding.
        """
        try:
            return pack_validate.validate_pack(pack_store.pack_dir(pack))
        except Exception as e:  # noqa: BLE001 — never let the lint break the browser
            logger.error("Pack validation failed for %s: %s", pack, e)
            return {
                "pack": str(pack),
                "ok": False,
                "errors": 1,
                "warnings": 0,
                "infos": 0,
                "counts": {},
                "diagnostics": [
                    {
                        "severity": "error",
                        "code": "validator-failed",
                        "path": "",
                        "line": 0,
                        "message": f"the pack checker itself failed: {e}",
                        "detail": "",
                        "hint": "the pack's files are untouched; this is a checker fault",
                    }
                ],
            }

    async def get_knowledge_tree(self, request):
        """``GET /api/v1/knowledge/{pack}/tree``: the file list alone.

        The cheap refresh after a save, when the caller already has diagnostics from the
        write response and only the sizes and line counts have moved.
        """
        try:
            return web.json_response(pack_store.pack_tree(request.match_info["pack"]))
        except pack_store.PackStoreError as e:
            return self._pack_error_response(e)
        except OSError as e:
            return web.json_response({"error": str(e)}, status=500)

    async def get_knowledge_file(self, request):
        """``GET /api/v1/knowledge/{pack}/file?path=``: one file's text.

        ``?download=1`` serves it as an attachment, the answer for a file over the inline-edit
        limit — a generated schema runs to hundreds of kilobytes, and a textarea round trip is
        how one gets lost. ``sha256`` is the token the write half's 409 path checks.
        """
        pack = request.match_info["pack"]
        rel = request.query.get("path") or ""
        try:
            info = pack_store.read_file(pack, rel)
        except pack_store.PackStoreError as e:
            return self._pack_error_response(e)
        except OSError as e:
            return web.json_response({"error": str(e)}, status=500)
        if _truthy(request.query.get("download")):
            return self._artifact_response(
                info["text"].encode("utf-8"),
                "text/plain",
                info["path"].rsplit("/", 1)[-1],
                True,
            )
        return web.json_response(info)

    async def validate_knowledge_pack(self, request):
        """``GET /api/v1/knowledge/{pack}/validate``: diagnostics for the whole pack.

        200 even when the pack has errors: a report produced successfully is not an HTTP
        failure, and a client reading non-200 as "the request broke" would show nothing for
        the case this describes. ``?strict=1`` returns 422 with the identical body, for CI.
        """
        pack = request.match_info["pack"]
        try:
            pack_store.safe_pack_name(pack)
        except pack_store.PackStoreError as e:
            return self._pack_error_response(e)
        if not pack_store.pack_exists(pack):
            return web.json_response({"error": f"no pack named {pack!r}"}, status=404)
        result = self._validate_payload(pack)
        strict = _truthy(request.query.get("strict")) and not result.get("ok")
        return web.json_response(result, status=422 if strict else 200)

    async def get_knowledge_history(self, request):
        """``GET /api/v1/knowledge/{pack}/history[?path=][&snapshot=]``: the undo store.

        Three views of one record: the pack's snapshots, one file's, or with ``?snapshot=``
        an entry's stored text, which is what a revert preview and a diff need. Read-only.
        """
        pack = request.match_info["pack"]
        snapshot = (request.query.get("snapshot") or "").strip()
        rel = request.query.get("path") or ""
        try:
            if snapshot:
                return web.json_response(
                    {
                        "pack": pack_store.safe_pack_name(pack),
                        "snapshot": snapshot,
                        "text": pack_store.snapshot_text(pack, snapshot),
                    }
                )
            return web.json_response(
                {
                    "pack": pack_store.safe_pack_name(pack),
                    "path": rel,
                    "entries": pack_store.history(pack, rel),
                }
            )
        except pack_store.PackStoreError as e:
            return self._pack_error_response(e)
        except OSError as e:
            return web.json_response({"error": str(e)}, status=500)

    # -- knowledge packs: the write half -----------------------------------

    async def _pack_body(self, request):
        """``(payload, None)`` or ``(None, response)``: a body must be a JSON object.

        Settled once here, since a bare string reaching a handler that calls ``.get`` on it is
        a 500 for what is plainly a bad request.
        """
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return None, web.json_response({"error": "Invalid JSON"}, status=400)
        except ValueError as e:  # decoding / content-length faults
            return None, web.json_response({"error": str(e)}, status=400)
        if not isinstance(payload, dict):
            return None, web.json_response(
                {"error": "Body must be a JSON object"}, status=400
            )
        return payload, None

    def _pack_write_result(self, pack, result):
        """Every mutating response carries the pack's diagnostics and `restart_required`.

        A save is when a pack-level break appears — the store proved the file parses on its
        own, while the ruleset now imports a check that no longer exists — and a caller who
        needs a second request to learn that usually will not make it. `restart_required` is
        always true: editing a pack does not reload the one this process holds.
        """
        result["validate"] = self._validate_payload(pack)
        result["restart_required"] = True
        return web.json_response(result)

    async def save_knowledge_file(self, request):
        """``PUT /api/v1/knowledge/{pack}/file?path=``: save a file, whole or by lines.

        `start_line`/`end_line` route to `replace_lines`, the primitive that keeps a pack's
        comments and YAML anchors; absent, `text` replaces the file. Both snapshot first.

        The optional `expect_sha` from a read is what makes concurrent editing safe: a 409
        means the file moved under the editor, where a 200 would discard whoever saved first.
        `expect_first_line`/`expect_last_line` do the same for a range, where a stale line
        number still points at a line, just the wrong one.
        """
        pack = request.match_info["pack"]
        rel = request.query.get("path") or ""
        payload, bad = await self._pack_body(request)
        if bad is not None:
            return bad
        text = payload.get("text")
        if not isinstance(text, str):
            return web.json_response(
                {"error": 'Body must be {"text": "<file contents>"}'}, status=400
            )
        start, end = payload.get("start_line"), payload.get("end_line")
        ranged = start is not None or end is not None
        try:
            if ranged:
                result = pack_store.replace_lines(
                    pack,
                    rel,
                    int(start or 0),
                    int(end or 0),
                    text,
                    expect_first_line=str(payload.get("expect_first_line") or ""),
                    expect_last_line=str(payload.get("expect_last_line") or ""),
                    expect_sha=str(payload.get("expect_sha") or ""),
                    actor=str(payload.get("actor") or ""),
                )
            else:
                result = pack_store.write_file(
                    pack,
                    rel,
                    text,
                    expect_sha=str(payload.get("expect_sha") or ""),
                    actor=str(payload.get("actor") or ""),
                )
        except pack_store.PackStoreError as e:
            return self._pack_error_response(e)
        except (TypeError, ValueError) as e:
            # A non-numeric line number: a malformed request, not a store fault.
            return web.json_response({"error": str(e)}, status=400)
        except OSError as e:
            logger.error("Pack write failed for %s/%s: %s", pack, rel, e)
            return web.json_response({"error": str(e)}, status=500)
        logger.info("Knowledge pack %s: saved %s", pack, result.get("path"))
        return self._pack_write_result(pack, result)

    async def create_knowledge_file(self, request):
        """``POST /api/v1/knowledge/{pack}/file?path=``: add a file that does not exist.

        Separate from the PUT so a mistyped path on a save cannot create a file no loader
        reads, leaving the edit apparently vanished. 409 if it exists.
        """
        pack = request.match_info["pack"]
        rel = request.query.get("path") or ""
        payload, bad = await self._pack_body(request)
        if bad is not None:
            return bad
        text = payload.get("text", "")
        if not isinstance(text, str):
            return web.json_response({"error": '"text" must be a string'}, status=400)
        try:
            result = pack_store.create_file(
                pack, rel, text, actor=str(payload.get("actor") or "")
            )
        except pack_store.PackStoreError as e:
            return self._pack_error_response(e)
        except OSError as e:
            logger.error("Pack create failed for %s/%s: %s", pack, rel, e)
            return web.json_response({"error": str(e)}, status=500)
        logger.info("Knowledge pack %s: created %s", pack, result.get("path"))
        return self._pack_write_result(pack, result)

    async def delete_knowledge_file(self, request):
        """``DELETE /api/v1/knowledge/{pack}/file?path=&confirm=1``: remove a file.

        `confirm=1` rides in the request rather than in a dialog the UI happens to show, which
        would leave a script or a stray retry uncovered. The content survives in history:
        `delete_file` snapshots first and refuses the unlink if that could not be stored.
        """
        pack = request.match_info["pack"]
        rel = request.query.get("path") or ""
        if not _truthy(request.query.get("confirm")):
            return web.json_response(
                {
                    "error": (
                        f"refusing to delete {rel!r} without confirm=1 — repeat the "
                        "request with &confirm=1 if that is what you mean"
                    ),
                    "path": rel,
                },
                status=400,
            )
        try:
            result = pack_store.delete_file(
                pack, rel, actor=str(request.query.get("actor") or "")
            )
        except pack_store.PackStoreError as e:
            return self._pack_error_response(e)
        except OSError as e:
            logger.error("Pack delete failed for %s/%s: %s", pack, rel, e)
            return web.json_response({"error": str(e)}, status=500)
        logger.info(
            "Knowledge pack %s: deleted %s (snapshot %s)",
            pack,
            result.get("path"),
            result.get("snapshot"),
        )
        return self._pack_write_result(pack, result)

    async def restore_knowledge_file(self, request):
        """``POST /api/v1/knowledge/{pack}/history/restore`` with ``{"snapshot": "<id>"}``.

        Puts a snapshot's bytes back, snapshotting the current content first, so undo is undoable.
        The restored content need not verify — undo is most needed right after an edit went
        wrong — so `pack_store.restore` reports `parses`/`parse_error` instead of refusing.
        """
        pack = request.match_info["pack"]
        payload, bad = await self._pack_body(request)
        if bad is not None:
            return bad
        snapshot_id = str(payload.get("snapshot") or "").strip()
        if not snapshot_id:
            return web.json_response(
                {"error": 'Body must be {"snapshot": "<snapshot id>"}'}, status=400
            )
        try:
            result = pack_store.restore(
                pack,
                snapshot_id,
                rel=str(payload.get("path") or ""),
                actor=str(payload.get("actor") or ""),
            )
        except pack_store.PackStoreError as e:
            return self._pack_error_response(e)
        except OSError as e:
            logger.error("Pack restore failed for %s: %s", pack, e)
            return web.json_response({"error": str(e)}, status=500)
        logger.info(
            "Knowledge pack %s: restored %s from %s",
            pack,
            result.get("path"),
            snapshot_id,
        )
        return self._pack_write_result(pack, result)

    async def import_knowledge_files(self, request):
        """``POST /api/v1/knowledge/{pack}/import`` with ``{"files": {"<rel>": "<text>"}}``.

        Every file is validated before any is written, in `import_config`'s shape: a multi-file
        import is usually one coherent change, so landing half leaves a pack broken in a way
        neither file's author would recognise. Unlike the create route, an existing file is
        overwritten, its prior bytes snapshotted so the import is reversible file by file.
        """
        pack = request.match_info["pack"]
        payload, bad = await self._pack_body(request)
        if bad is not None:
            return bad
        files = payload.get("files")
        if not isinstance(files, dict) or not files:
            return web.json_response(
                {"error": 'Body must be {"files": {"<path>": "<text>"}}'}, status=400
            )
        try:
            pack_store.safe_pack_name(pack)
        except pack_store.PackStoreError as e:
            return self._pack_error_response(e)
        if not pack_store.pack_exists(pack):
            return web.json_response({"error": f"no pack named {pack!r}"}, status=404)
        errors = []
        prepared = []
        for rel, text in files.items():
            if not isinstance(text, str):
                errors.append(f"{rel}: contents must be a string")
                continue
            try:
                relp = pack_store.safe_rel_path(str(rel))
                pack_store.require_editable_suffix(relp)
                target = pack_store.pack_file(pack, str(relp))
                pre = (
                    target.read_text(encoding="utf-8", errors="replace")
                    if target.is_file()
                    else None
                )
                # The store's own verification, ahead of the batch: per-write, the first
                # invalid file would abort a transaction that already replaced its
                # predecessors.
                pack_store.verify_candidate(str(relp), text, pre_image=pre)
            except pack_store.PackStoreError as e:
                errors.append(str(e))
            else:
                prepared.append((str(relp), text, pre is None))
        if errors:
            return web.json_response(
                {"error": "Import rejected; nothing was written", "errors": errors},
                status=400,
            )
        written = []
        for relp, text, is_new in prepared:
            try:
                if is_new:
                    written.append(
                        pack_store.create_file(
                            pack, relp, text, actor=str(payload.get("actor") or "")
                        )
                    )
                else:
                    written.append(
                        pack_store.write_file(
                            pack,
                            relp,
                            text,
                            actor=str(payload.get("actor") or ""),
                            reason="import",
                        )
                    )
            except (pack_store.PackStoreError, OSError) as e:
                # Pre-verified above, so this is an I/O fault or a file that moved mid-import.
                # Report what landed: guessing the partial state wrong means importing twice.
                logger.error("Pack import failed on %s/%s: %s", pack, relp, e)
                return web.json_response(
                    {"error": f"{relp}: {e}", "written": written}, status=500
                )
        logger.info("Knowledge pack %s: imported %s", pack, ", ".join(files))
        return self._pack_write_result(pack, {"written": written})

    async def scaffold_knowledge_pack(self, request):
        """``POST /api/v1/knowledge/scaffold`` with ``{"name", "vocabulary": [...]}``.

        A new pack from the checked-in template, which loads, so the author diffs their first
        edit against a green baseline rather than an empty directory. `vocabulary` is required
        and written here — the template's own placeholder words collide with its example
        ruleset, so it ships without the file that proves the engine speaks no domain.
        """
        payload, bad = await self._pack_body(request)
        if bad is not None:
            return bad
        name = str(payload.get("name") or "").strip()
        if not name:
            return web.json_response(
                {"error": 'Body must be {"name": "<pack name>", "vocabulary": [...]}'},
                status=400,
            )
        try:
            result = pack_store.scaffold_pack(
                name,
                vocabulary=payload.get("vocabulary"),
                actor=str(payload.get("actor") or ""),
            )
        except pack_store.PackStoreError as e:
            return self._pack_error_response(e)
        except OSError as e:
            logger.error("Pack scaffold failed for %s: %s", name, e)
            return web.json_response({"error": str(e)}, status=500)
        logger.info("Knowledge pack scaffolded: %s", result.get("pack"))
        return self._pack_write_result(result["pack"], result)

    # -- the pack assistant ------------------------------------------------
    #
    # The split across four endpoints makes the approval step structural: `POST /assist` produces
    # a plan and nothing else, `/apply` is the only route that writes, and the assistant has no
    # write tool of its own. `/events` streams the exploration, which is how an operator judges
    # whether the proposal rests on the right files.

    def _assist_ready(self, pack):
        """``(assistant, None)``, or ``(None, 503)`` where no LLM client is wired.

        Unlike the pipeline stages, which fall back to a deterministic report, an assistant
        that cannot call a model has nothing to offer.
        """
        if self.llm_client is None:
            return None, web.json_response(
                {
                    "error": (
                        "The pack assistant is not available: no LLM endpoint is wired "
                        "into this server. The manual editor is unaffected."
                    )
                },
                status=503,
            )
        try:
            pack_store.safe_pack_name(pack)
        except pack_store.PackStoreError as e:
            return None, self._pack_error_response(e)
        if not pack_store.pack_exists(pack):
            return None, web.json_response(
                {"error": f"no pack named {pack!r}"}, status=404
            )
        return pack_assistant.PackAssistant(self.llm_client), None

    def _assist_session(self, request):
        """``(session, None)`` or ``(None, response)``. The pack has to match.

        Every op's path is relative to a pack root and the same relative path exists in most
        packs, so the id alone would let a stale tab apply a plan computed against another.
        """
        session = pack_assistant.get_session(request.match_info.get("session") or "")
        if session is None:
            return None, web.json_response(
                {
                    "error": (
                        "No such assist session. Sessions are held in memory and are "
                        "dropped on restart, because a plan is only valid against the "
                        "files it was computed from."
                    )
                },
                status=404,
            )
        if session.pack != pack_store.safe_pack_name(request.match_info["pack"]):
            return None, web.json_response(
                {"error": f"session {session.id} belongs to pack {session.pack!r}"},
                status=409,
            )
        return session, None

    async def list_knowledge_assists(self, request):
        """``GET /api/v1/knowledge/{pack}/assist``: this pack's live sessions."""
        try:
            pack = pack_store.safe_pack_name(request.match_info["pack"])
        except pack_store.PackStoreError as e:
            return self._pack_error_response(e)
        return web.json_response(
            {
                "sessions": pack_assistant.list_sessions(pack),
                "max_sessions": pack_assistant.MAX_SESSIONS,
            }
        )

    async def start_knowledge_assist(self, request):
        """``POST /api/v1/knowledge/{pack}/assist``: ask for a change, get a proposal.

        Body ``{question, focus: [paths], guidance, prior_plan, allow_delete}``. Answers 202
        with the session id and runs the loop in the background — exploration is several LLM
        round trips, and holding the request open makes a proxy timeout look identical to a
        model that produced nothing — so the caller watches ``/events`` or polls.

        Nothing here writes. `allow_delete` only lets the preview show a deletion as blocked
        rather than omitting it; the apply call carries its own flag.
        """
        assistant, bad = self._assist_ready(request.match_info["pack"])
        if bad is not None:
            return bad
        payload, bad = await self._pack_body(request)
        if bad is not None:
            return bad
        question = str(payload.get("question") or "").strip()
        if not question:
            return web.json_response(
                {"error": 'Body must be {"question": "<what to change>"}'}, status=400
            )
        if len(question) > 20_000:
            return web.json_response(
                {"error": "The question is over the 20,000-character limit"}, status=413
            )
        pack = pack_store.safe_pack_name(request.match_info["pack"])
        focus = [str(p) for p in (payload.get("focus") or []) if str(p).strip()]
        # Converted before the session starts, so an unreadable document lands in the 202
        # rather than three turns into an exploration. A rejection rides back in
        # `attachment_errors` without blocking the run.
        attachments, attach_errors = await pack_attachments.prepare(
            payload.get("attachments")
        )
        session = pack_assistant.new_session(pack, question)
        session.attachments = attachments
        # Parked on the session because asyncio holds only a weak reference, and a collected
        # task is a run cancelled mid-flight.
        session.task = asyncio.create_task(
            assistant.run(
                session,
                focus=focus,
                guidance=str(payload.get("guidance") or ""),
                prior_plan=(
                    payload.get("prior_plan")
                    if isinstance(payload.get("prior_plan"), dict)
                    else None
                ),
                allow_delete=_truthy(payload.get("allow_delete")),
            )
        )
        logger.info(
            "Pack assist %s started on %s (%d attachment(s), %d rejected)",
            session.id,
            pack,
            len(attachments),
            len(attach_errors),
        )
        body = session.snapshot()
        body["attachment_errors"] = attach_errors
        return web.json_response(body, status=202)

    async def get_knowledge_assist(self, request):
        """``GET /api/v1/knowledge/{pack}/assist/{session}``: status, trail, plan, diffs."""
        session, bad = self._assist_session(request)
        if bad is not None:
            return bad
        return web.json_response(session.snapshot())

    async def stream_knowledge_assist(self, request):
        """``GET .../assist/{session}/events``: SSE, one event per tool call and status.

        Replays the session's history before tailing, like the job stream, so a late
        subscriber still sees every file that was read.
        """
        session, bad = self._assist_session(request)
        if bad is not None:
            return bad
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)
        try:
            async for event in pack_assistant.subscribe(session.id):
                await response.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
        except (ConnectionResetError, asyncio.CancelledError):
            pass  # the client went away; subscribe() drops its queue in a finally
        except Exception as e:  # noqa: BLE001
            logger.warning("Assist SSE error for %s: %s", session.id, e)
        return response

    async def apply_knowledge_assist(self, request):
        """``POST .../assist/{session}/apply``: write the approved plan. All or nothing.

        Body ``{allow_delete, actor, plan}``. A `plan` here replaces the proposed one through
        the identical validation, so an operator can correct a proposal rather than choose
        between accepting it verbatim and rejecting it. A failing op is a 400 with nothing
        written and the per-op reasons, so the next attempt can be a correction.
        """
        session, bad = self._assist_session(request)
        if bad is not None:
            return bad
        payload, bad = await self._pack_body(request)
        if bad is not None:
            return bad
        plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else None
        edited = plan is not None
        if plan is None:
            plan = session.plan
        if not plan:
            return web.json_response(
                {"error": f"session {session.id} has no plan to apply"}, status=409
            )
        allow_delete = _truthy(payload.get("allow_delete"))
        actor = str(payload.get("actor") or "")
        try:
            # On a thread: building and validating a candidate tree is tens of seconds of
            # CPU, and inline it stalls every in-flight job's SSE stream. Computed once and
            # handed to the apply, so one click validates the pack once.
            #
            # `dry_run` only for a hand-edited plan: a proposal was already dry-run into the
            # preview being approved, while an edited one has never been measured.
            checks = await asyncio.to_thread(
                pack_assistant.plan_checks,
                session.pack,
                plan,
                allow_delete=allow_delete,
                dry_run=edited,
            )
            # Before the write, because a preview is computed against the pre-image: taken
            # after, every patch op fails its own anchors and the recorded plan reads as a
            # conflict with an empty diff.
            preview = pack_assistant.plan_preview(
                session.pack, plan, allow_delete=allow_delete, checks=checks
            )
            result = await asyncio.to_thread(
                pack_assistant.apply_plan,
                session.pack,
                plan,
                allow_delete=allow_delete,
                actor=actor,
                session=session.id,
                checks=checks,
            )
        except pack_store.PackStoreError as e:
            return self._pack_error_response(e)
        except OSError as e:
            logger.error("Assist apply failed on %s: %s", session.pack, e)
            return web.json_response({"error": str(e)}, status=500)
        if not result.get("applied"):
            status = 500 if result.get("rolled_back") is not None else 400
            return web.json_response(result, status=status)
        session.status = "applied"
        # The plan that was written, not the one proposed: a hand-edited apply reading back
        # as the model's proposal makes the intervention untraceable.
        if edited:
            session.plan = (
                pack_assistant.EditPlan(**plan) if isinstance(plan, dict) else plan
            )
            session.preview = preview
        pack_assistant.emit(
            session,
            "assist_status",
            status="applied",
            message=f"{len(result.get('written') or [])} file(s) written",
            edited=edited,
        )
        logger.info(
            "Pack assist %s applied to %s (%s file(s), edited=%s)",
            session.id,
            session.pack,
            len(result.get("written") or []),
            edited,
        )
        return self._pack_write_result(session.pack, result)

    async def reject_knowledge_assist(self, request):
        """``POST .../assist/{session}/reject``: send it back with a correction.

        Starts a new session carrying the rejected plan and the guidance, so the proposal that
        was turned down stays inspectable beside its replacement. Guidance is required: without
        it the retry sends the identical request.
        """
        session, bad = self._assist_session(request)
        if bad is not None:
            return bad
        assistant, bad = self._assist_ready(session.pack)
        if bad is not None:
            return bad
        payload, bad = await self._pack_body(request)
        if bad is not None:
            return bad
        guidance = str(payload.get("guidance") or "").strip()
        if not guidance:
            return web.json_response(
                {
                    "error": (
                        "Say what is wrong with the proposal. Without a correction the "
                        "retry sends the identical request and returns the same plan."
                    )
                },
                status=400,
            )
        session.status = "rejected"
        pack_assistant.emit(
            session, "assist_status", status="rejected", message=guidance[:200]
        )
        retry = pack_assistant.new_session(session.pack, session.question)
        # Inherited already converted, so a correction about one needs no re-upload. Shared
        # rather than copied: nothing mutates them after preparation.
        retry.attachments = list(session.attachments)
        retry.task = asyncio.create_task(
            assistant.run(
                retry,
                focus=list(session.focus),
                guidance=guidance,
                prior_plan=session.plan.model_dump() if session.plan else None,
                allow_delete=_truthy(payload.get("allow_delete")),
            )
        )
        logger.info(
            "Pack assist %s rejected; retrying as %s on %s",
            session.id,
            retry.id,
            session.pack,
        )
        return web.json_response(
            {"rejected": session.id, **retry.snapshot()}, status=202
        )

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def get_ir(self, request):
        await self.rate_limiter.acquire()
        # Checked here as well as in the funnel: this handler calls the external tracker
        # before it has an incident to run, and a refusal must not first spend somebody
        # else's work.
        if not self._blocking_endpoints_available():
            return web.json_response(dict(self._BLOCKING_RETIRED_BODY), status=501)
        try:
            incident = await request.json()
            if not validate_ir_request(incident):
                return web.Response(status=400, text="Invalid IR data")

            url = (
                self.config["win_url"]
                + "/api/v2/json/records/?rnid="
                + incident["id"]
                + "&with=full"
            )
            auth = aiohttp.BasicAuth(
                self.config["win_username"], self.config["win_password"]
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(url, auth=auth) as response:
                    if response.status != 200:
                        return web.Response(status=400, text="Could not get IR")
                    record = await response.json()

            fft = record["records"][0]["ffts"][0]
            ir = self._normalize_incident(
                {
                    "id": fft["rnid"],
                    "timestamp": fft["update_date"],
                    "description": fft["fft"],
                },
                source="ir_lookup",
            )
            logger.info(f"Received incident: {ir['id']}")
            return await self._run_pipeline(ir, verbose=self._wants_verbose(request))
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        except Exception as e:
            logger.error(f"Error receiving incident: {str(e)}")
            return web.Response(status=400, text="Invalid IR")


def _jsonable(report):
    """Coerce a report (Pydantic model, dict, or string) into JSON-serializable form."""
    if report is None:
        return None
    if hasattr(report, "model_dump"):
        return report.model_dump()
    return report
