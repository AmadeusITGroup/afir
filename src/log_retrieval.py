import asyncio
import logging

from sshtunnel import SSHTunnelForwarder

from src.encoded_fields import decode_logs
from src.models.pydantic_models import RetrievalQuery
from src.retrieval_cache import build_retrieval_cache, cache_key
from src.retrievers.base import unresolved_placeholders
from src.retrievers.databricks_retriever import DatabricksRetriever
from src.retrievers.elasticsearch_retriever import ElasticsearchRetriever
from src.retrievers.kibana_retriever import KibanaRetriever
from src.retrievers.rest_retriever import RestRetriever
from src.retrievers.snowflake_retriever import SnowflakeRetriever

logger = logging.getLogger(__name__)

_RETRIEVER_TYPES = {
    "elasticsearch": ElasticsearchRetriever,
    "kibana": KibanaRetriever,
    "databricks": DatabricksRetriever,
    "snowflake": SnowflakeRetriever,
    "rest": RestRetriever,
    # "databricks_genie": GenieRetriever,  # future drop-in, same interface
}

# Knowledge-pack endpoint `kind` -> internal retriever `type`. Kinds absent here
# have no retriever yet and are skipped with a log line.
_KIND_TO_TYPE = {
    "elasticsearch": "elasticsearch",
    "databricks_uc": "databricks",
    "snowflake": "snowflake",
    "rest": "rest",
}


def _is_placeholder(value) -> bool:
    """True when a config value is still an unfilled ``<...>`` template marker."""
    if not isinstance(value, str):
        return False
    v = value.strip()
    return "<" in v and ">" in v


# Rows per source when neither the endpoint nor the global setting declares one.
_DEFAULT_ROW_CAP = 500


class LogRetrievalEngine:
    """Dispatches each RetrievalQuery to the retriever for its configured source.

    Explicit ``log_sources.sources[]`` entries take precedence; pack ``endpoints``
    are merged with ``log_sources.backends`` credentials for everything else.
    """

    def __init__(self, config, llm_client, auth=None, knowledge_pack=None):
        self.config = config
        self.llm_client = llm_client
        self.auth = auth
        self.knowledge_pack = knowledge_pack
        self.cache = build_retrieval_cache(config)
        # One DatabricksAuth per workspace host; PATs are workspace-scoped.
        self._auth_by_host = {}
        if auth is not None:
            try:
                self._auth_by_host[auth.host.rstrip("/")] = auth
            except (
                Exception
            ):  # noqa: BLE001 — host lookup must never break construction
                pass
        self.retrievers = {}
        # {source: reason} for sources that built no retriever (never asked vs. answered nothing).
        self.unavailable_sources = {}
        self._skipped = {}
        for name, source_type, source_config in self._build_source_configs():
            retriever_cls = _RETRIEVER_TYPES.get(source_type)
            if retriever_cls is None:
                logger.warning(
                    "Unknown retriever type '%s' for source '%s'; skipping.",
                    source_type,
                    name,
                )
                self.unavailable_sources[name] = (
                    f"no retriever implements type '{source_type}'"
                )
                continue
            if source_type == "databricks":  # only Databricks uses unified auth
                self.retrievers[name] = retriever_cls(
                    source_config,
                    llm_client,
                    auth=self._auth_for_workspace(source_config.get("workspace_url")),
                    knowledge_pack=knowledge_pack,
                )
            else:
                self.retrievers[name] = retriever_cls(
                    source_config, llm_client, knowledge_pack=knowledge_pack
                )
        # An explicit entry may override a skipped pack source; built set wins.
        for name, reason in (self._skipped or {}).items():
            if name not in self.retrievers:
                self.unavailable_sources[name] = reason
        if self.unavailable_sources:
            logger.warning(
                "%d pack source(s) built no retriever and CANNOT be queried on this "
                "run: %s",
                len(self.unavailable_sources),
                ", ".join(sorted(self.unavailable_sources)),
            )

    def _build_source_configs(self):
        """Yield ``(name, type, config_dict)``; explicit main_config sources win over pack sources."""
        built = []
        seen = set()
        self._skipped = {}

        # 1. Explicit main_config sources (full connection info): highest priority.
        for source in self.config.get("sources", []):
            name = source.get("name")
            if not name:
                continue
            seen.add(name)
            # `setdefault` lets an explicit entry declare its own guards.
            pack_src = (
                self.knowledge_pack.source(name)
                if self.knowledge_pack is not None
                else None
            )
            if pack_src is not None:
                self._attach_query_guards(pack_src, source)
            built.append((name, source.get("type"), source))

        # 2. Pack sources, merged with the backends credential map.
        backends = self.config.get("backends", {}) or {}
        pack = self.knowledge_pack
        for src in getattr(pack, "sources", []) or []:
            if src.name in seen:
                continue  # explicit config already covers this name
            kind = src.kind()
            source_type = _KIND_TO_TYPE.get(kind)
            if source_type is None:
                logger.info(
                    "No retriever for endpoint kind '%s' (source '%s'); skipping.",
                    kind or "<none>",
                    src.name,
                )
                self._skipped[src.name] = (
                    f"endpoint kind '{kind or '<none>'}' has no retriever"
                )
                continue
            merged = self._merge_endpoint(src, source_type, backends)
            if merged is None:
                continue  # missing creds; recorded by _merge_endpoint, already logged
            # Guards are attached here so no backend branch can silently drop them.
            self._attach_query_guards(src, merged)
            self._apply_primary_budget(merged)
            # _merge_endpoint may refine the type (e.g. kibana gateway); trust merged type.
            built.append((src.name, merged.get("type", source_type), merged))

        return built

    def _row_cap_default(self) -> int:
        """Rows per source when the endpoint does not state its own ``max_results``."""
        try:
            configured = int(
                (getattr(self, "config", None) or {}).get("max_results") or 0
            )
        except (TypeError, ValueError):
            return _DEFAULT_ROW_CAP
        return configured if configured > 0 else _DEFAULT_ROW_CAP

    def _apply_primary_budget(self, merged: dict) -> None:
        """Raise the per-source cap and retriever statement budget for a primary source.

        Both caps are written from one value so the lower cannot win silently.
        Records the value under ``_primary_budget_applied`` so a live config change
        (via ``refresh_primary_budgets``) can also lower it.
        """
        if str(merged.get("retrieval_class", "") or "").lower() != "primary":
            return
        budget = int(
            merged.get("primary_retrieval_timeout_seconds")
            or self.config.get("primary_source_timeout_seconds")
            or self._PRIMARY_TIMEOUT_SECONDS
        )
        previous = merged.get("_primary_budget_applied")
        for key in ("retrieval_timeout_seconds", "statement_timeout_seconds"):
            current = int(merged.get(key) or 0)
            if budget > current or (previous is not None and current == int(previous)):
                merged[key] = budget
        interval = int(merged.get("poll_interval_seconds") or 5) or 5
        needed = budget // interval + 2
        current_attempts = int(merged.get("max_poll_attempts") or 0)
        if needed > current_attempts or (
            previous is not None and current_attempts == int(previous) // interval + 2
        ):
            merged["max_poll_attempts"] = needed
        merged["_primary_budget_applied"] = budget
        logger.info(
            "Source '%s' is PRIMARY: retrieval budget raised to %ss (a cap here decides the "
            "verdict rather than degrading it; genuine loss of contact is caught separately "
            "by consecutive poll errors).",
            merged.get("name", "?"),
            budget,
        )

    def refresh_primary_budgets(self) -> int:
        """Re-apply primary budgets from current config; returns sources touched.

        Needed after a live config change so the new budget takes effect without restart.
        """
        touched = 0
        for retriever in self.retrievers.values():
            cfg = getattr(retriever, "config", None)
            if (
                isinstance(cfg, dict)
                and str(cfg.get("retrieval_class", "") or "").lower() == "primary"
            ):
                self._apply_primary_budget(cfg)
                touched += 1
        return touched

    @staticmethod
    def _attach_query_guards(src, merged: dict) -> None:
        """Copy pack source query guarantees onto the retriever config.

        Applied here so every backend receives them; enforced in ``query_guards.py``.
        """
        merged.setdefault(
            "require_all_entities",
            list(getattr(src, "require_all_entities", []) or []),
        )
        merged.setdefault(
            "identity_keys",
            [list(c or []) for c in (getattr(src, "identity_keys", []) or [])],
        )
        merged.setdefault(
            "identity_scopes", list(getattr(src, "identity_scopes", []) or [])
        )
        merged.setdefault(
            "identity_synonyms", list(getattr(src, "identity_synonyms", []) or [])
        )
        merged.setdefault("never_filter", list(getattr(src, "never_filter", []) or []))
        merged.setdefault(
            "partition_columns", list(getattr(src, "partition_columns", []) or [])
        )
        merged.setdefault(
            "epoch_time_columns", list(getattr(src, "epoch_time_columns", []) or [])
        )
        merged.setdefault(
            "default_filters", dict(getattr(src, "default_filters", {}) or {})
        )
        merged.setdefault(
            "retrieval_class", str(getattr(src, "retrieval_class", "") or "")
        )

    def _note_skip(self, name, reason):
        """Record why a pack source built no retriever, for `unavailable_sources`."""
        if getattr(self, "_skipped", None) is None:
            self._skipped = {}
        self._skipped[name] = reason

    def _merge_endpoint(self, src, source_type, backends):
        """Merge pack endpoint coordinates with backend creds; returns ``None`` (logging) when creds absent."""
        endpoints = src.endpoints or {}
        if source_type == "elasticsearch":
            cluster = endpoints.get("cluster")
            creds = (backends.get("elasticsearch", {}) or {}).get(cluster)
            if not creds or not creds.get("url") or _is_placeholder(creds.get("url")):
                logger.info(
                    "No elasticsearch backend creds for cluster '%s' (source '%s'); "
                    "skipping.",
                    cluster,
                    src.name,
                )
                self._note_skip(
                    src.name,
                    f"no elasticsearch backend URL configured for cluster "
                    f"'{cluster}'",
                )
                return None
            if not creds.get("username") or not creds.get("password"):  # ${VAR} expands unset to ""
                logger.info(
                    "No elasticsearch credentials for cluster '%s' (source '%s'); "
                    "skipping.",
                    cluster,
                    src.name,
                )
                self._note_skip(
                    src.name,
                    f"elasticsearch cluster '{cluster}' has a URL but its "
                    f"username/password env vars are unset",
                )
                return None
            indices = endpoints.get("indices") or []
            index = ",".join(indices) if indices else endpoints.get("index", "")
            # `gateway: kibana` routes to KibanaRetriever (DSL) instead of ES|QL; same config shape.
            retriever_type = (
                "kibana" if creds.get("gateway") == "kibana" else "elasticsearch"
            )
            return {
                "name": src.name,
                "type": retriever_type,
                "url": creds["url"],
                "username": creds.get("username"),
                "password": creds.get("password"),
                "index": index,
                "timeout": creds.get("timeout", 30),
                "max_results": creds.get("max_results", self._row_cap_default()),
                "verify_ssl": creds.get("verify_ssl"),
                "ca_bundle": creds.get("ca_bundle"),
                "query_hints": getattr(src, "query_hints", "") or "",
            }
        if source_type == "databricks":
            db_backends = backends.get("databricks", {}) or {}
            # Two shapes: workspace-keyed map, or flat block when no `workspace` declared.
            workspace = endpoints.get("workspace")
            if workspace:
                creds = db_backends.get(workspace)
                if not creds or not creds.get("warehouse_id"):
                    logger.info(
                        "No databricks backend creds for workspace '%s' (source '%s'); "
                        "skipping.",
                        workspace,
                        src.name,
                    )
                    self._note_skip(
                        src.name,
                        f"no databricks warehouse configured for workspace "
                        f"'{workspace}'",
                    )
                    return None
            else:
                creds = db_backends
                if not creds.get("warehouse_id"):
                    logger.info(
                        "No databricks backend warehouse_id configured (source '%s'); "
                        "skipping.",
                        src.name,
                    )
                    self._note_skip(src.name, "no databricks warehouse_id configured")
                    return None
            return {
                "name": src.name,
                "type": "databricks",
                "workspace": workspace,
                "workspace_url": creds.get("workspace_url"),
                "warehouse_id": creds["warehouse_id"],
                "api_key_env": creds.get("api_key_env", "DATABRICKS_TOKEN"),
                "catalog": endpoints.get("catalog"),
                "schema": endpoints.get("schema"),
                "tables": endpoints.get("tables") or [],
                "max_results": creds.get("max_results", self._row_cap_default()),
                "poll_interval_seconds": creds.get("poll_interval_seconds", 5),
                "max_poll_attempts": creds.get("max_poll_attempts", 60),
                "retrieval_timeout_seconds": creds.get("retrieval_timeout_seconds"),
                # Defaults to retrieval_timeout_seconds in the retriever so the two caps stay in sync.
                "statement_timeout_seconds": creds.get("statement_timeout_seconds"),
                "max_consecutive_poll_errors": creds.get(
                    "max_consecutive_poll_errors", 5
                ),
                "http_connect_timeout_seconds": creds.get(
                    "http_connect_timeout_seconds", 15
                ),
                "http_total_timeout_seconds": creds.get(
                    "http_total_timeout_seconds", 70
                ),
                "warehouse_warmup": creds.get("warehouse_warmup", True),
                "field_schema": "",
                "query_hints": getattr(src, "query_hints", "") or "",
                "projection": list(getattr(src, "projection", []) or []),
                "verify_ssl": creds.get("verify_ssl"),
                "ca_bundle": creds.get("ca_bundle"),
            }
        if source_type == "snowflake":
            account = endpoints.get("account")
            creds = (backends.get("snowflake", {}) or {}).get(account)
            if (
                not creds
                or not creds.get("account")
                or _is_placeholder(creds.get("account"))
            ):
                logger.info(
                    "No snowflake backend creds for account '%s' (source '%s'); "
                    "skipping.",
                    account,
                    src.name,
                )
                self._note_skip(
                    src.name,
                    f"no snowflake backend configured for account '{account}'",
                )
                return None
            return {
                "name": src.name,
                "type": "snowflake",
                "account": creds["account"],
                "user": creds.get("user"),
                "password": creds.get("password"),
                "password_env": creds.get("password_env"),
                "private_key_env": creds.get("private_key_env"),
                "role": creds.get("role"),
                "warehouse": creds.get("warehouse"),
                "databases": endpoints.get("databases", []),
                "schemas": endpoints.get("schemas", []),
                "objects": endpoints.get("objects", []),
                "max_results": creds.get("max_results", self._row_cap_default()),
                "field_schema": "",
            }
        if source_type == "rest":
            service = endpoints.get("service")
            creds = (backends.get("rest", {}) or {}).get(service)
            if (
                not creds
                or not creds.get("base_url")
                or _is_placeholder(creds.get("base_url"))
            ):
                logger.info(
                    "No rest backend creds for service '%s' (source '%s'); skipping.",
                    service,
                    src.name,
                )
                self._note_skip(
                    src.name, f"no rest backend configured for service '{service}'"
                )
                return None
            return {
                "name": src.name,
                "type": "rest",
                "base_url": creds["base_url"],
                "username": creds.get("username"),
                "password": creds.get("password"),
                "token_env": creds.get("token_env"),
                "tables": endpoints.get("tables", []),
                "timeout": creds.get("timeout", 30),
                "max_results": creds.get("max_results", self._row_cap_default()),
            }
        return None

    def row_caps(self):
        """``{source: max_results}`` for every built retriever; needed to detect truncation."""
        caps = {}
        for name, retriever in self.retrievers.items():
            cap = getattr(retriever, "max_results", None)
            if cap is None:
                cap = (getattr(retriever, "config", None) or {}).get("max_results")
            try:
                caps[name] = int(cap)
            except (TypeError, ValueError):
                continue
        return caps

    def _keyed_source(self, source) -> bool:
        """True when the query just published for ``source`` constrained its declared key.

        Read at the instant rows land in ``_gather``; a shared retriever's
        ``last_key_enforced`` reflects whichever job published last otherwise.
        """
        retriever = self.retrievers.get(source)
        return getattr(retriever, "last_key_enforced", False) is True

    async def _gather(
        self,
        queries,
        progress_cb=None,
        extended=False,
        guidance="",
        keyed_out=None,
        queries_out=None,
        unanswered_out=None,
    ):
        """Retrieve from every source concurrently, each under its own timeout.

        ``extended=True`` raises caps and widens Databricks poll budgets for this call only.

        ``unanswered_out``: ``{source: why}`` for sources asked but not answered (timed-out,
        cancelled, placeholder-empty). Absent from ``logs`` like a source nobody asked —
        downstream must distinguish the two. Popped on success so follow-up passes retract it.

        Cache hits skip the task and the LLM call. Only ``logs`` entries are stored, so
        timeouts, backend errors, cancels, and placeholder-empties are never cached.
        """
        default_timeout = self.config.get("per_source_timeout_seconds", 20)
        # Per-run state; retrievers are shared across jobs, so these must not outlive _gather.
        keyed = {}
        placeholders = {}  # captured before send; `last_generated_query` outlives the run
        published = {}  # query text as published; also fed to the cache

        def _notify(source, status, message):
            if progress_cb is not None:
                try:
                    progress_cb(source, status, message)
                except Exception:  # noqa: BLE001 — progress must never break retrieval
                    logger.debug("progress_cb raised for %s", source, exc_info=True)

        def _announce(source):
            def _on_query(text):
                # Capture synchronously: `last_key_enforced`/`last_generated_query` reflect
                # whichever job published last.
                keyed[source] = self._keyed_source(source)
                if text:
                    published[source] = text
                if queries_out is not None and text:
                    queries_out[source] = text
                placeholders[source] = unresolved_placeholders(text)
                _notify(source, "query_ready", f"Query generated for {source}")

            return _on_query

        logs = {}
        # Cache lookup before any tasks exist; `extended=True` skips (bigger budget = new question).
        keys = {}
        for query in queries:
            source = query.target_log_source
            keys[source] = cache_key(query, self.row_caps().get(source), guidance)
        hits = {}
        if not extended:
            for source, key in keys.items():
                hit = self.cache.get(key)
                if hit is not None:
                    hits[source] = hit
        for source, hit in hits.items():
            logs[source] = hit.rows
            keyed[source] = hit.key_enforced  # replayed; no retriever ran
            published[source] = hit.query
            if queries_out is not None and hit.query:
                queries_out[source] = hit.query
            if unanswered_out is not None:
                unanswered_out.pop(source, None)
            _notify(source, "query_ready", f"Query for {source} answered from cache")
            _cap = self.row_caps().get(source)
            _capped = bool(_cap) and len(hit.rows) >= _cap
            _notify(
                source,
                "completed",
                f"Retrieved {len(hit.rows)} rows from {source} "
                f"({hit.describe()}, not re-queried)"
                + (
                    f" — TRUNCATED at the {_cap}-row cap (max_results); the real "
                    "total is higher"
                    if _capped
                    else ""
                ),
            )

        tasks = {
            query.target_log_source: asyncio.ensure_future(
                self._retrieve_one(
                    query, guidance, on_query=_announce(query.target_log_source)
                )
            )
            for query in queries
            if query.target_log_source not in hits
        }
        timeouts = {
            source: self._source_timeout(source, default_timeout, extended=extended)
            for source in tasks
        }
        poll_overrides = self._apply_extended_poll_budgets(timeouts) if extended else {}
        # Absolute deadlines: a relative timeout would count from the current iteration.
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadlines = {source: started + timeouts[source] for source in tasks}

        reported = set()  # prevents the cancel sweep from relabelling timed-out sources
        try:
            for source in tasks:
                _notify(source, "running", f"Querying {source}…")

            remaining_tasks = dict(tasks)
            while remaining_tasks:
                now = loop.time()
                next_deadline = min(deadlines[s] for s in remaining_tasks)
                # Wait on the tasks directly: asyncio.wait never cancels what it waits
                # on (no shield needed), and the task exception is consumed below.
                await asyncio.wait(
                    list(remaining_tasks.values()),
                    timeout=max(0.0, next_deadline - now),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                settled = [
                    s
                    for s, t in remaining_tasks.items()
                    if t.done() or loop.time() >= deadlines[s]
                ]
                if not settled:
                    continue
                for source in settled:
                    task = remaining_tasks.pop(source)
                    source_timeout = timeouts[source]
                    reported.add(source)
                    if not task.done():
                        logger.error(
                            "Timed out (%ss) retrieving logs for %s",
                            source_timeout,
                            source,
                        )
                        _notify(
                            source,
                            "timeout",
                            f"{source} timed out after {source_timeout}s",
                        )
                        if unanswered_out is not None:
                            unanswered_out[source] = (
                                f"did not answer within its {source_timeout}s budget"
                            )
                        continue
                    try:
                        logs[source] = task.result()
                        # Empty + unresolved placeholder = placeholder matched nothing, not the source.
                        if not logs[source] and placeholders.get(source):
                            _unfilled = ", ".join(placeholders[source])
                            logs.pop(source, None)
                            _notify(
                                source,
                                "failed",
                                f"{source} was asked with unresolved placeholder(s) "
                                f"{_unfilled} and returned 0 rows — that is the query "
                                "matching nothing, NOT the source having nothing",
                            )
                            if unanswered_out is not None:
                                unanswered_out[source] = (
                                    "was asked with unresolved placeholder(s) "
                                    f"{_unfilled}, so its empty result says nothing "
                                    "about the source"
                                )
                            continue
                        if unanswered_out is not None:
                            unanswered_out.pop(source, None)
                        _row_count = len(logs[source]) if logs[source] else 0
                        _cap = self.row_caps().get(source)
                        _capped = bool(_cap) and _row_count >= _cap
                        _notify(
                            source,
                            "completed",
                            f"Retrieved {_row_count} rows from {source}"
                            + (
                                f" — TRUNCATED at the {_cap}-row cap "
                                "(max_results); the real total is higher"
                                if _capped
                                else ""
                            ),
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as result:  # noqa: BLE001
                        logger.error("Error retrieving logs for %s: %s", source, result)
                        _notify(source, "failed", f"{source} failed: {result}")
                        if unanswered_out is not None:
                            unanswered_out[source] = (  # type, not message; full error is in the log
                                f"failed to answer ({type(result).__name__}); see the "
                                "source's own line for the backend error"
                            )
        finally:
            pending = {s: t for s, t in tasks.items() if not t.done()}
            for t in pending.values():
                t.cancel()
            # Notify before drain so the event is emitted even if a task refuses to unwind.
            for source in pending:
                if source not in reported:
                    _notify(source, "cancelled", f"{source} cancelled")
                    if unanswered_out is not None:
                        unanswered_out[source] = "was cancelled before it answered"
            if pending:
                await asyncio.gather(*pending.values(), return_exceptions=True)
            self._restore_poll_budgets(poll_overrides)
        self._decode_encoded_fields(logs, _notify)
        # Timed-out sources published a query but are absent from `logs`; key flag must not outlive rows.
        if keyed_out is not None:
            keyed_out.update({s: True for s in logs if keyed.get(s)})
        # Stored from `logs` after decode; non-answers never reach `logs` so they can't be cached.
        for source, rows in logs.items():
            if source in hits:
                continue
            self.cache.store(
                keys.get(source, ""),
                rows,
                query=published.get(source, ""),
                key_enforced=bool(keyed.get(source)),
            )
        return logs

    def _decode_encoded_fields(self, logs, notify=None) -> None:
        """Decode declared encoded fields in place; best-effort (failure keeps the encoded value)."""
        try:
            decoded = decode_logs(logs, self.knowledge_pack)
        except Exception as exc:  # noqa: BLE001 — never fail retrieval over a decode
            logger.warning("Encoded-field decoding failed: %s", exc)
            return
        if not decoded or notify is None:
            return
        for source, count in decoded.items():
            notify(
                source,
                "completed",
                f"Retrieved {len(logs.get(source) or [])} rows from {source} — "
                f"decoded {count} record(s) from its encoded field(s)",
            )

    def _apply_extended_poll_budgets(self, timeouts):
        """Widen Databricks poll budgets to cover the extended cap; returns originals for restore.

        Scoped to one ``_gather`` call; ``_restore_poll_budgets`` puts values back.
        """
        originals = {}
        for source, cap in timeouts.items():
            retriever = self.retrievers.get(source)
            # Duck-typed: only retrievers that poll (Databricks) carry these attrs.
            if retriever is None or not hasattr(retriever, "max_poll_attempts"):
                continue
            interval = getattr(retriever, "poll_interval", 5) or 5
            needed = int(cap / interval) + 2  # +2 margin so retriever polls past the cap
            if needed > retriever.max_poll_attempts:
                originals[source] = retriever.max_poll_attempts
                retriever.max_poll_attempts = needed
        return originals

    def _restore_poll_budgets(self, poll_overrides):
        for source, original in (poll_overrides or {}).items():
            retriever = self.retrievers.get(source)
            if retriever is not None and hasattr(retriever, "max_poll_attempts"):
                retriever.max_poll_attempts = original

    def _auth_for_workspace(self, workspace_url):
        """Return DatabricksAuth for ``workspace_url``, or ``None`` (→ ``api_key_env`` fallback).

        PATs are workspace-scoped; ``self.auth`` must not be reused for a different host.
        """
        if not workspace_url:
            return self.auth  # unknown host → preserve single-workspace behaviour
        host = workspace_url.rstrip("/")
        return self._auth_by_host.get(
            host
        )  # None -> retriever falls back to api_key_env

    _EXTENDED_MULTIPLIER = 4  # multiplier when no explicit extended cap is configured

    # Budget for primary sources; stops an abandoned statement (lost contact is caught
    # by the consecutive-poll-error counter). Overridable per source in the pack.
    _PRIMARY_TIMEOUT_SECONDS = 7200

    def _source_timeout(self, source, default_timeout, extended=False):
        """Per-source retrieval cap; primary budget is already on ``retrieval_timeout_seconds``.

        Extended: max of per-source, global, or normal × ``_EXTENDED_MULTIPLIER``.
        """
        retriever = self.retrievers.get(source)
        cfg = getattr(retriever, "config", None) or {}
        override = cfg.get("retrieval_timeout_seconds")
        normal = override if override else default_timeout
        if not extended:
            return normal
        source_ext = cfg.get("extended_retrieval_timeout_seconds")
        global_ext = self.config.get("extended_retrieval_timeout_seconds")
        extended_cap = source_ext or global_ext or (normal * self._EXTENDED_MULTIPLIER)
        return max(normal, extended_cap)

    async def _retrieve_one(
        self, query: RetrievalQuery, guidance: str = "", on_query=None
    ):
        retriever = self.retrievers.get(query.target_log_source)
        if retriever is None:
            raise ValueError(
                f"No retriever configured for log source: {query.target_log_source}"
            )
        kwargs = {}
        if guidance:
            kwargs["guidance"] = guidance  # must fail loudly if retriever ignores it
        if on_query is not None:  # observability; a retriever that rejects it should still run
            try:
                return await retriever.retrieve(query, on_query=on_query, **kwargs)
            except TypeError as exc:
                if "on_query" not in str(exc):
                    raise
                logger.debug(
                    "Retriever for '%s' does not accept on_query; its query will be "
                    "reported when the source settles.",
                    query.target_log_source,
                )
        return await retriever.retrieve(query, **kwargs)

    async def retrieve(
        self,
        queries,
        progress_cb=None,
        extended=False,
        guidance="",
        keyed_out=None,
        queries_out=None,
        unanswered_out=None,
    ):
        """Retrieve every query; ``guidance`` is analyst direction from a rejected gate.

        Passed per call: a shared engine instance must not carry per-run state.
        """
        return await self._gather(
            queries,
            progress_cb=progress_cb,
            extended=extended,
            guidance=guidance,
            keyed_out=keyed_out,
            queries_out=queries_out,
            unanswered_out=unanswered_out,
        )

    async def retrieve_with_tunnel(
        self,
        queries,
        progress_cb=None,
        extended=False,
        guidance="",
        keyed_out=None,
        queries_out=None,
        unanswered_out=None,
    ):
        tunnel = self.config["tunnel"]
        with SSHTunnelForwarder(
            tunnel["url"],
            ssh_username=tunnel["user"],
            ssh_password=tunnel["password"],
            remote_bind_address=(tunnel["remote_bind_url"], tunnel["remote_bind_port"]),
            local_bind_address=(tunnel["local_bind_url"], tunnel["local_bind_port"]),
        ):
            return await self._gather(
                queries,
                progress_cb=progress_cb,
                extended=extended,
                guidance=guidance,
                keyed_out=keyed_out,
                queries_out=queries_out,
                unanswered_out=unanswered_out,
            )

    async def close(self):
        for retriever in self.retrievers.values():
            try:
                await retriever.close()
            except Exception as e:
                logger.warning("Error closing retriever: %s", e)
