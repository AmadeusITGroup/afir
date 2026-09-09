"""Databricks retriever (SQL Statement Execution API).

LLM generates a SELECT; the retriever executes it via the Statement Execution REST API.
Schema is discovered from ``information_schema`` on first use and cached.
"""

import asyncio
import logging
import os
import re
import ssl
from typing import Dict, List, Optional

import aiohttp

from src.human_guidance import guidance_prompt_line
from src.models.pydantic_models import (RetrievalQuery, SchemaSelection,
                                        SqlQuery)
from src.retrievers.base import (DataRetriever, publish_query, raise_for_status_with_reason)
from src.retrievers.field_mapping import (declared_leaf_paths,
                                          event_time_column,
                                          form_split_bindings, incident_values,
                                          key_presence_values, map_entities,
                                          match_patterns, render_filters,
                                          render_identifiers,
                                          source_bindings, stem_literals,
                                          subject_anchor, value_tuple_columns)
from src.retrievers.query_guards import (conjunction_fields,
                                         enforce_conjunction,
                                         enforce_default_filters,
                                         enforce_conjunction_same_column,
                                         enforce_epoch_window,
                                         enforce_event_time_window,
                                         enforce_identity_scope,
                                         enforce_key_presence,
                                         enforce_partition_bounds,
                                         enforce_subject_anchor,
                                         enforce_value_tuples,
                                         epoch_prompt_line, guard_prompt_line,
                                         merge_partition_specs,
                                         partition_prompt_line,
                                         relax_form_conjunction,
                                         required_fields,
                                         resolve_identity_fields,
                                         same_column_conjunctions,
                                         strip_evidence_predicates,
                                         strip_fabricated_predicates,
                                         strip_vacuous_disjuncts,
                                         widen_match_patterns,
                                         widen_stem_literals)
from src.utils.error_handling import (NonRetryableError,
                                      async_retry_with_backoff)
from src.utils.projection import (expand_projection, projection_alias,
                                  projection_names, returned_names)

logger = logging.getLogger(__name__)

_TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELED", "CLOSED"}


class DatabricksRetriever(DataRetriever):
    def __init__(self, config: Dict, llm_client, auth=None, knowledge_pack=None):
        self.config = config
        self.llm_client = llm_client
        self.knowledge_pack = knowledge_pack
        # For UI/API visibility.
        self.last_generated_query = None
        self.last_field_map = None
        # DatabricksAuth preferred (fresh token + host); static api_key_env is the fallback.
        self.auth = auth
        workspace_url = config.get("workspace_url")
        if not workspace_url and auth is not None:
            workspace_url = auth.host
        if not workspace_url:
            raise ValueError(
                f"Databricks source '{config.get('name', '?')}' has no workspace_url "
                "and no auth host; set backends.databricks.workspace_url or provide auth."
            )
        self.workspace_url = workspace_url.rstrip("/")
        self.warehouse_id = config["warehouse_id"]
        self._static_token = (
            os.getenv(config["api_key_env"]) if config.get("api_key_env") else None
        )
        self.catalog = config.get("catalog")
        self.schema = config.get("schema")
        # Scopes schema discovery; prevents the LLM from inventing table names.
        self.tables = [t for t in (config.get("tables") or []) if t]
        self.query_hints = (config.get("query_hints") or "").strip()
        # Exact leaf columns the SQL must return; empty lets the LLM pick freely.
        self.projection = [c for c in (config.get("projection") or []) if c]
        # Enforced after generation (see query_guards); advisory hints cannot override them.
        self.require_all_entities = [
            e for e in (config.get("require_all_entities") or []) if e
        ]
        # Priority-ordered: first candidate this incident fully satisfies wins.
        self.identity_keys = [
            [t for t in (cand or []) if t]
            for cand in (config.get("identity_keys") or [])
        ]
        # Event-log identity shape: OR inside a synonym family, AND between scopes.
        self.identity_scopes = [f for f in (config.get("identity_scopes") or []) if f]
        self.identity_synonyms = [
            f for f in (config.get("identity_synonyms") or []) if f
        ]
        # Predicate stripped post-generation; unit is pack-declared.
        self.never_filter = [f for f in (config.get("never_filter") or []) if f]
        self.default_filters = config.get("default_filters") or {}
        # Discovered from metadata; pack's `partition_columns` merged on top for VIEWs.
        self.declared_partitions = [
            p for p in (config.get("partition_columns") or []) if p and p.get("name")
        ]
        self._discovered_partitions: List[Dict] = []
        # Time columns stored as an epoch integer; unit is pack-declared.
        self.epoch_time_columns = [
            c for c in (config.get("epoch_time_columns") or []) if c and c.get("name")
        ]
        # If unset, discovered from information_schema on first use.
        self._field_schema = config.get("field_schema") or None
        self._discovery_attempted = False
        self.max_results = config.get("max_results", 500)
        self.poll_interval = config.get("poll_interval_seconds", 5)
        self.max_poll_attempts = config.get("max_poll_attempts", 60)
        # Wall-clock budget; unset derives from the engine's per-source cap.
        self.statement_budget = config.get("statement_timeout_seconds") or (
            config.get("retrieval_timeout_seconds") or 0
        )
        # Exit on consecutive poll errors, not elapsed time: RUNNING is working, PENDING is queued.
        self.max_consecutive_poll_errors = config.get("max_consecutive_poll_errors", 5)
        # aiohttp default 300s; the poll loop carries the cold-start budget.
        self.http_connect_timeout = config.get("http_connect_timeout_seconds", 15)
        self.http_total_timeout = config.get("http_total_timeout_seconds", 70)
        self.warehouse_warmup = config.get("warehouse_warmup", True)
        self._warmed_up = False
        self._session = None

    def _bearer_token(self) -> Optional[str]:
        """Current token: this caller's own if they set one, then a fresh SDK token, else static."""
        own = self._personal_token()
        if own:
            return own
        if self.auth is not None:
            return self.auth.token()
        return self._static_token

    def _personal_token(self) -> Optional[str]:
        """The current run owner's own token for this backend's credential name, if any."""
        name = self.config.get("api_key_env")
        if not name:
            return None
        try:
            from src.user_secrets import personal_value

            return personal_value(name)
        except Exception as exc:  # noqa: BLE001 — an override may never fail a retrieval
            logger.debug("Personal credential lookup failed: %s", exc)
            return None

    def _auth_headers(self) -> Dict[str, str]:
        """Per-REQUEST authorization, because the session's is baked in at construction.

        One retriever is reused across runs and a session cannot be rebuilt while a statement
        is in flight, so the token is sent per call: that is what lets one caller's personal
        credential apply to their own run and nobody else's, and it makes the SDK refresh live
        on a session that outlives a token's lifetime.
        """
        return {"Authorization": f"Bearer {self._bearer_token()}"}

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # Deliberately no Authorization here: a session header is fixed at construction,
            # so it would be whichever caller's token happened to open the session, handed to
            # every request that forgot to override it. Every request site passes
            # `_auth_headers()`; an unauthenticated 401 is the honest failure of a new one
            # that does not.
            headers = {}
            ssl_ctx = _build_ssl_context(self.config)
            timeout = aiohttp.ClientTimeout(
                total=self.http_total_timeout,
                sock_connect=self.http_connect_timeout,
            )
            if ssl_ctx is None:
                self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
            else:
                connector = aiohttp.TCPConnector(ssl=ssl_ctx)
                self._session = aiohttp.ClientSession(
                    headers=headers, connector=connector, timeout=timeout
                )
        return self._session

    async def _execute_sql(self, sql: str) -> List[Dict]:
        """Run a SQL statement on the warehouse and return rows as dicts.

        Polls the same ``statement_id`` — never resubmits (a resubmit starts another warehouse).
        Best-effort cancel on budget exhaustion or outer cancellation.
        """
        session = self._get_session()
        payload = {
            "warehouse_id": self.warehouse_id,
            "statement": sql,
            # 50s is the API's maximum synchronous wait. A warm warehouse succeeds here in
            # one shot; a cold one comes back pending and is polled below.
            "wait_timeout": "50s",
            "row_limit": self.max_results,
        }
        if self.catalog:
            payload["catalog"] = self.catalog
        if self.schema:
            payload["schema"] = self.schema

        logger.info("Submitting Databricks SQL statement: %s", sql)
        headers = self._auth_headers()
        async with session.post(
            f"{self.workspace_url}/api/2.0/sql/statements",
            json=payload,
            headers=headers,
        ) as resp:
            await raise_for_status_with_reason(resp)
            data = await resp.json()

        statement_id = data["statement_id"]
        state = data["status"]["state"]

        try:
            budget = self._poll_budget_seconds()
            deadline = asyncio.get_running_loop().time() + budget
            poll_errors = 0
            attempts = 0
            while state not in _TERMINAL_STATES:
                if asyncio.get_running_loop().time() >= deadline:
                    # Not retryable: re-running the same slow statement from scratch can only
                    # blow the engine's per-source cap as well.
                    raise NonRetryableError(
                        f"Databricks statement {statement_id} still {state} after "
                        f"{budget:.0f}s (poll budget exhausted); giving up"
                    )
                await asyncio.sleep(self.poll_interval)
                attempts += 1
                try:
                    async with session.get(
                        f"{self.workspace_url}/api/2.0/sql/statements/{statement_id}",
                        # Re-derived per poll: a statement can outlive the token it was
                        # submitted with.
                        headers=self._auth_headers(),
                    ) as resp:
                        await raise_for_status_with_reason(resp)
                        data = await resp.json()
                    state = data["status"]["state"]
                    poll_errors = 0
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    poll_errors += 1
                    if poll_errors >= self.max_consecutive_poll_errors:
                        raise RuntimeError(
                            f"Lost contact with Databricks statement {statement_id} "
                            f"after {poll_errors} consecutive poll failures: {e}"
                        ) from e
                    logger.warning(
                        "Poll %d for statement %s failed (%s); retrying (%d/%d).",
                        attempts,
                        statement_id,
                        e,
                        poll_errors,
                        self.max_consecutive_poll_errors,
                    )

            if state != "SUCCEEDED":
                message = data.get("status", {}).get("error", {}).get("message", state)
                raise RuntimeError(
                    f"Databricks statement {statement_id} did not succeed: {message}"
                )

            return _rows_from_statement(data)
        except (asyncio.CancelledError, Exception):
            if state not in _TERMINAL_STATES:
                await self._cancel_statement(statement_id)
            raise

    def _poll_budget_seconds(self) -> float:
        """Wall-clock poll budget; explicit timeout wins over attempt-count product, larger of the two."""
        legacy = float(self.max_poll_attempts or 0) * float(self.poll_interval or 5)
        return max(float(self.statement_budget or 0), legacy) or 300.0

    async def _cancel_statement(self, statement_id: str) -> None:
        """Best-effort cancel of a running statement; never raises."""
        try:
            session = self._get_session()
            async with session.post(
                f"{self.workspace_url}/api/2.0/sql/statements/{statement_id}/cancel",
                headers=self._auth_headers(),
            ) as resp:
                # 200 on success; ignore anything else (already terminal, etc.).
                await resp.read()
        except Exception as e:  # noqa: BLE001 — cancel is best-effort cleanup
            logger.debug("Cancel of statement %s failed: %s", statement_id, e)

    async def _warmup(self) -> None:
        """Warm the warehouse once upfront; best-effort, never blocks retrieval."""
        if self._warmed_up or not self.warehouse_warmup:
            return
        self._warmed_up = True  # attempt once regardless of outcome
        try:
            await self._execute_sql("SELECT 1")
            logger.info(
                "Warehouse %s warm (source '%s').",
                self.warehouse_id,
                self.config.get("name", "?"),
            )
        except Exception as e:  # noqa: BLE001 — warm-up is best-effort
            logger.warning(
                "Warehouse warm-up failed for source '%s' (continuing): %s",
                self.config.get("name", "?"),
                e,
            )

    async def _get_field_schema(self) -> str:
        """Return configured or information_schema-discovered schema (cached).

        Priority: configured ``field_schema`` → one schema → full-catalog LLM curation.
        """
        if self._field_schema:
            return self._field_schema
        # A failed or empty discovery is cached so the full scan does not re-run.
        if self._discovery_attempted:
            return ""
        if not self.catalog:
            return ""  # cannot scope discovery without at least a catalog

        self._discovery_attempted = True
        await self._warmup()
        try:
            if self.schema:
                # Table names stay unqualified: the statement payload already scopes the SQL
                # to catalog.schema.
                rows = await self._discover_columns(self.schema, qualify=False)
            else:
                rows = await self._scan_and_curate_catalog()
        except Exception as e:
            logger.warning(
                "Schema discovery failed; the LLM will infer column names: %s", e
            )
            return ""

        self._discovered_partitions = self._partitions_from_columns(rows)
        if self._discovered_partitions:
            logger.info(
                "Source '%s': discovered partition column(s) %s — every generated query is "
                "bounded on each one whose meaning is known: a date/timestamp column, or a "
                "numeric one a `role:`/`epoch_time_columns` declaration explains. Discovery "
                "reports a name and a type and cannot say which, so a numeric column nothing "
                "declares is left unbounded (warned per query) rather than bounded by a date "
                "literal it would read as arithmetic.",
                self.config.get("name", "?"),
                ", ".join(
                    f"{p['name']} ({p.get('type') or '?'})"
                    for p in self._discovered_partitions
                ),
            )
        elif self.declared_partitions:
            logger.info(
                "Source '%s': the metadata reports no partition columns (a VIEW hides its "
                "underlying table's layout); using the pack's declared %s.",
                self.config.get("name", "?"),
                ", ".join(str(p.get("name")) for p in self.declared_partitions),
            )
        # Declared leaves render first so the per-table cap keeps them (checked by map_entities).
        self._field_schema = _render_schema(rows, self._declared_leaves())
        return self._field_schema

    def _declared_leaves(self) -> List[str]:
        """Field paths this source's own config names — `entity_bindings` + `projection`."""
        declared = declared_leaf_paths(
            self.knowledge_pack, self.config.get("name") or ""
        )
        # projection may arrive without a pack; expand_projection reads only column leaves.
        return list(declared) + [
            p for p in expand_projection(self.projection) if p not in declared
        ]

    async def _discover_columns(self, schema: str, qualify: bool = True) -> List[Dict]:
        """All columns of one schema from information_schema, else DESCRIBE; reconciles nested types."""
        table_filter = ""
        if self.tables:
            quoted = ", ".join(f"'{t}'" for t in self.tables)
            table_filter = f"AND table_name IN ({quoted}) "
        try:
            rows = await self._execute_sql(
                "SELECT table_name, column_name, data_type, full_data_type, partition_index "
                f"FROM {self.catalog}.information_schema.columns "
                f"WHERE table_schema = '{schema}' "
                f"{table_filter}"
                "ORDER BY table_name, ordinal_position"
            )
        except Exception as exc:  # noqa: BLE001 — a catalog without the view is not a failure
            logger.warning(
                "Source '%s': %s.information_schema is unreadable (%s); falling back to "
                "DESCRIBE TABLE. An empty schema is not a degraded run — it drops every "
                "declared binding as stale, so the query keeps its window bound and nothing "
                "else, and an arbitrary page reads exactly like the subject's own rows.",
                self.config.get("name", "?"),
                self.catalog,
                str(exc)[:200],
            )
            rows = []
        if not rows:
            rows = await self._describe_columns(schema)
        await self._reconcile_nested_types(rows, schema)
        if qualify:
            for row in rows:
                row["table_name"] = f"{schema}.{row['table_name']}"
        return rows

    async def _describe_columns(self, schema: str) -> List[Dict]:
        """Discovery rows built from ``DESCRIBE TABLE``, in the information_schema shape.

        The legacy Hive metastore exposes no ``information_schema``, so the only reading of
        such a catalog is per table. ``DESCRIBE TABLE`` also names the partition columns, in
        a repeated section below the column list — which is why they are parsed here rather
        than left to a pack declaration: a partition this path did not report is a partition
        no generated query bounds, over a table large enough to be partitioned.
        """
        tables = list(self.tables or [])
        if not tables:
            try:
                listed = await self._execute_sql(f"SHOW TABLES IN {self.catalog}.{schema}")
            except Exception as exc:  # noqa: BLE001 — degrade to no schema, as before
                logger.warning(
                    "Source '%s': cannot list tables in %s.%s (%s).",
                    self.config.get("name", "?"),
                    self.catalog,
                    schema,
                    str(exc)[:200],
                )
                return []
            tables = [
                str(r.get("tableName") or r.get("table_name") or "").strip()
                for r in listed
            ]
            tables = [t for t in tables if t][:_MAX_DESCRIBE_TABLES]
        rows: List[Dict] = []
        for table in tables[:_MAX_DESCRIBE_TABLES]:
            fq = f"{self.catalog}.{schema}.{table}"
            try:
                described = await self._execute_sql(f"DESCRIBE TABLE {fq}")
            except Exception as exc:  # noqa: BLE001 — one unreadable table is not the schema
                logger.warning("Could not DESCRIBE %s (%s).", fq, str(exc)[:200])
                continue
            rows.extend(_columns_from_describe(described, table))
        if rows:
            logger.info(
                "Source '%s': %d column(s) across %d table(s) read with DESCRIBE TABLE.",
                self.config.get("name", "?"),
                len(rows),
                len({r["table_name"] for r in rows}),
            )
        return rows

    async def _reconcile_nested_types(self, rows: List[Dict], schema: str) -> None:
        """Fix STRUCT types the catalog reports incompletely; leaves ``rows`` untouched on any failure.

        Uses ``DESCRIBE TABLE`` (+ ``DESCRIBE QUERY`` per partial column) to detect mismatches.
        Bounded by ``_MAX_RECONCILE_STATEMENTS``; unchecked tables are named in a warning.
        """
        by_table: Dict[str, List[Dict]] = {}
        for row in rows:
            declared = str(row.get("full_data_type") or row.get("data_type") or "")
            if declared.strip().upper().startswith("STRUCT<"):
                by_table.setdefault(str(row.get("table_name") or ""), []).append(row)
        if not by_table:
            return
        budget = _MAX_RECONCILE_STATEMENTS
        unchecked: List[str] = []
        for table, struct_rows in sorted(by_table.items()):
            if not table:
                continue
            if budget <= 0:
                unchecked.append(table)
                continue
            fq = f"{self.catalog}.{schema}.{table}"
            try:
                described = await self._execute_sql(f"DESCRIBE TABLE {fq}")
            except Exception as exc:  # noqa: BLE001 — degrade to the catalog's own answer
                logger.warning(
                    "Could not read the live schema of %s (%s); the catalog's nested types "
                    "are used unverified and may be missing recently added fields.",
                    fq,
                    exc,
                )
                continue
            budget -= 1
            live = _live_column_types(described)
            for row in struct_rows:
                column = str(row.get("column_name") or "")
                catalog_type = str(
                    row.get("full_data_type") or row.get("data_type") or ""
                )
                live_type = live.get(column)
                if not live_type:
                    continue
                if not type_is_partial(live_type):
                    if struct_child_names(live_type) != struct_child_names(catalog_type):
                        logger.warning(
                            "Column %s.%s: the catalog reports %d nested field(s), the live "
                            "schema %d. Using the live schema — a stale type would have the "
                            "generator query fields that hold nothing.",
                            fq,
                            column,
                            len(struct_child_names(catalog_type)),
                            len(struct_child_names(live_type)),
                        )
                        row["full_data_type"] = live_type
                    continue
                expected = len(struct_child_names(live_type)) + hidden_field_count(
                    live_type
                )
                known = len(struct_child_names(catalog_type))
                if known >= expected:
                    continue
                if budget <= 0:
                    unchecked.append(f"{table}.{column}")
                    continue
                budget -= 1
                expanded = await self._expand_struct_type(fq, column)
                if not expanded:
                    continue
                logger.warning(
                    "Column %s.%s: the catalog reports %d nested field(s) and the live schema "
                    "has %d. Expanded to %d from the live schema — the catalog's type parses "
                    "cleanly, so this would otherwise be invisible.",
                    fq,
                    column,
                    known,
                    expected,
                    len(struct_child_names(expanded)),
                )
                row["full_data_type"] = expanded
        if unchecked:
            logger.warning(
                "Nested-type reconciliation stopped after %d statements; these were NOT "
                "verified against the live schema and may be missing fields: %s",
                _MAX_RECONCILE_STATEMENTS,
                ", ".join(unchecked),
            )

    async def _expand_struct_type(
        self, fq_table: str, path: str, depth: int = 0
    ) -> Optional[str]:
        """Live STRUCT type from ``DESCRIBE QUERY SELECT <path>.* FROM <table>``, recursing on partial children.

        Returns ``None`` on any failure (caller keeps the catalog's value).
        """
        try:
            described = await self._execute_sql(
                f"DESCRIBE QUERY SELECT {path}.* FROM {fq_table}"
            )
        except Exception as exc:  # noqa: BLE001 — degrade, never raise into discovery
            logger.warning("Could not expand %s.%s: %s", fq_table, path, exc)
            return None
        children: List[tuple] = []
        for row in described:
            name = str(row.get("col_name") or "").strip()
            child_type = str(row.get("data_type") or "").strip()
            if not name or name.startswith("#"):
                continue
            if (
                type_is_partial(child_type)
                and child_type.upper().startswith("STRUCT<")
                and depth + 1 < _MAX_STRUCT_DEPTH
            ):
                deeper = await self._expand_struct_type(
                    fq_table, f"{path}.{name}", depth + 1
                )
                child_type = deeper or child_type
            children.append((name, child_type))
        return struct_type_from_children(children) or None

    @staticmethod
    def _partitions_from_columns(rows: List[Dict]) -> List[Dict]:
        """Partition specs from discovery rows, ordered by ``partition_index``, de-duplicated by name."""
        found: List[tuple] = []
        for row in rows or []:
            index = row.get("partition_index")
            if index is None or str(index).strip() == "":
                continue
            name = str(row.get("column_name") or "").strip()
            if not name:
                continue
            try:
                order = int(index)
            except (TypeError, ValueError):
                order = 0
            found.append((order, name, str(row.get("data_type") or "")))
        specs: List[Dict] = []
        seen = set()
        for _, name, col_type in sorted(found, key=lambda x: x[0]):
            if name in seen:
                continue
            seen.add(name)
            specs.append({"name": name, "type": col_type})
        return specs

    def partitions(self) -> List[Dict]:
        """Partition specs in force: discovered, with the pack's declarations layered on."""
        return merge_partition_specs(
            self._discovered_partitions, self.declared_partitions
        )

    async def _scan_and_curate_catalog(self) -> List[Dict]:
        """Scan all catalog schemas; LLM curates fraud-relevant tables. Runs once (cached)."""
        # List tables (no column data yet).
        table_rows = await self._execute_sql(
            "SELECT table_schema, table_name "
            f"FROM {self.catalog}.information_schema.tables "
            "WHERE table_schema <> 'information_schema' "
            "ORDER BY table_schema, table_name"
        )
        all_tables = [f"{r['table_schema']}.{r['table_name']}" for r in table_rows]
        if not all_tables:
            return []
        logger.info(
            "Scanned %s: %d tables across schemas.", self.catalog, len(all_tables)
        )

        # LLM curates the fraud-relevant subset.
        selected = await self._curate_tables(all_tables)
        if not selected:
            logger.warning(
                "LLM selected no tables; falling back to the full catalog scan."
            )
            selected = all_tables

        # Normalise to schema.table and fetch columns only for needed schemas.
        wanted = set()
        for t in selected:
            parts = t.split(".")
            wanted.add(".".join(parts[-2:]) if len(parts) >= 2 else t)

        schemas = {t.split(".")[0] for t in wanted if "." in t}
        rows: List[Dict] = []
        for schema in sorted(schemas):
            cols = await self._discover_columns(schema, qualify=True)
            rows.extend(r for r in cols if r["table_name"] in wanted)
        logger.info(
            "Curated %d fraud-relevant tables from %s.", len(wanted), self.catalog
        )
        return rows

    async def _curate_tables(self, all_tables: List[str]) -> List[str]:
        """Ask the LLM which scanned tables are relevant to fraud investigation."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are selecting database tables relevant to fraud investigation "
                    "(transactions, logins/auth, users/accounts, sessions, audit logs, "
                    "devices, payments). From the catalog's table list, return the "
                    "fraud-relevant schema.table names. Exclude staging/temp/backup and "
                    "obviously unrelated reference tables. Be selective but not empty."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Catalog: {self.catalog}\nTables (schema.table):\n"
                    + "\n".join(all_tables)
                ),
            },
        ]
        result = await self.llm_client.structured_output(
            messages, response_model=SchemaSelection, stage="log_retrieval"
        )
        return result.tables

    async def _generate_sql(
        self,
        query: RetrievalQuery,
        field_schema: str,
        filter_hints: str = "",
        guidance: str = "",
    ) -> str:
        target = ".".join(p for p in (self.catalog, self.schema) if p)
        # Pin exact table(s) so the LLM cannot invent a name when field_schema is empty.
        if self.tables:
            fq = [".".join(p for p in (target, t) if p) for t in self.tables]
            tables_line = (
                "You MUST query ONLY from "
                + (
                    f"this exact table: {fq[0]}"
                    if len(fq) == 1
                    else "these exact tables (choose the most relevant): "
                    + ", ".join(fq)
                )
                + ". Do NOT invent, guess, or alter the table name — use it verbatim.\n"
            )
        else:
            tables_line = ""
        filter_line = (
            "Entity filters (mapped incident value -> real column): "
            f"{filter_hints}.\n"
            "Treat these as EVIDENCE TO FIND, not all-mandatory. Pick the MOST "
            "SELECTIVE identifier(s) among them — one that names a single record, such "
            "as a record id or document number — and require it (AND) so the scan stays "
            "bounded. Combine the WEAKER, possibly-imprecise entities (an organisational "
            "unit, an acting account, a partner/product code) with OR, and OR that whole "
            "group onto the selective filter — do NOT AND every entity together. One wrong "
            "or mismatched value (the account that created a record often differs from the "
            "one that later acted on it, and a code reported in an alert is sometimes a "
            "different code system than the column holds) then yields ZERO rows even "
            "though the record exists. A hint of the "
            "form `col IN (a, b)` already means match ANY of those values. If only ONE "
            "selective identifier is available, filter on it plus the date range and "
            "omit the weak entities entirely rather than risk excluding the row.\n"
            if filter_hints
            else ""
        )
        hints_line = (
            f"Source-specific guidance (follow it): {self.query_hints}\n"
            if self.query_hints
            else ""
        )
        guard = guard_prompt_line(self.never_filter)
        hints_line += f"{guard}\n" if guard else ""
        prune = partition_prompt_line(
            self.partitions(), query.date_from, query.date_to, dialect="sql"
        )
        hints_line += f"{prune}\n" if prune else ""
        # Window pre-converted for epoch columns; the pack declares the unit.
        epoch = epoch_prompt_line(
            self.epoch_time_columns, query.date_from, query.date_to
        )
        hints_line += f"{epoch}\n" if epoch else ""
        if self.projection:
            aliased = [e for e in self.projection if projection_alias(e)]
            alias_line = (
                "MANDATORY ALIASES — these entries declare their own `AS <name>`. Copy each "
                "one VERBATIM, alias included, and do NOT rename it (the alias is the name "
                "the caller reads the value under; the underscore rule below does NOT apply "
                "to them):\n" + "\n".join(f"  {e}" for e in aliased) + "\n"
                if aliased
                else ""
            )
            proj_line = (
                "REQUIRED PROJECTION (overrides the scalar-preference guidance above): "
                "the SELECT MUST return AT LEAST these exact columns/leaf paths, using "
                "the names verbatim from the Schema — do NOT omit any, do NOT collapse a "
                "struct to fewer leaves:\n" + ", ".join(self.projection) + "\n"
                + alias_line
                + "CRITICAL — ALIAS every dotted/struct leaf THAT DOES NOT ALREADY CARRY "
                "ONE with a UNIQUE column name so "
                "no two leaves collide. Databricks names a projected struct leaf after "
                "its LAST segment only, so `record.id`, `creator.account.id` and "
                "`owner.account.id` would ALL become a column called `id` and overwrite "
                "each other. You MUST write an explicit alias built from the FULL path "
                "with underscores, e.g. `record.id AS record_id`, "
                "`creator.account.id AS creator_account_id`, "
                "`counters.NOTE AS counters_NOTE`. Keep the alias equal to "
                "the dotted path with '.' replaced by '_' so downstream code can find it.\n"
                "For any listed leaf that is an ARRAY (or a path INTO an array element) "
                "AND is not one of the MANDATORY ALIASES above (those already say how the "
                "group is to be reduced, and rewriting one discards a reduction that was "
                "measured), use LATERAL VIEW "
                "OUTER explode(...) (the OUTER is MANDATORY — a plain explode DROPS every "
                "row whose array is empty/null, and a record with NOTHING in that array is "
                "usually the exact case a check is looking for, so dropping it turns a "
                "finding into a missing row). Alias the exploded column too "
                "(e.g. `LATERAL VIEW OUTER explode(items.line) line_tbl AS items_line`). "
                "Scalar and struct-scalar leaves are selected directly (with their alias). "
                "You MAY add other relevant columns, but every leaf above must be present.\n"
            )
        else:
            proj_line = ""
        messages = [
            {
                "role": "system",
                "content": (
                    "You write Databricks ANSI SQL SELECT queries. Produce ONE read-only "
                    "SELECT statement.\n"
                    f"{'Tables live under: ' + target if target else ''}\n"
                    f"{tables_line}"
                    f"Schema: {field_schema or 'unknown — infer reasonable column names, but use the exact table name given above'}.\n"
                    f"{filter_line}"
                    "CRITICAL: Use table and column names EXACTLY as they appear in the "
                    "Schema — copy them verbatim. Do NOT abbreviate, pluralize, singularize "
                    "or otherwise alter a name (e.g. if the schema says `events_v4`, never "
                    "write `events`). Only select from columns present in the Schema.\n"
                    "The schema lists columns as `name type`. Names with an underscore "
                    "(e.g. creator_unit_id) are ONE flat column — reference them exactly, "
                    "never split on the underscore into a dotted path. A dotted name like "
                    "`items.line` is a STRUCT field: reference it verbatim as "
                    "items.line (struct field-access) — do NOT wrap the whole dotted "
                    "path in backticks (`items.line` is WRONG and unresolvable); "
                    "backtick only a single identifier segment if it needs quoting. "
                    "A column whose type starts with ARRAY<...>, MAP<...> or STRUCT<...> "
                    "canNOT be selected raw when it is large/complex — PREFER scalar "
                    "columns (string/int/date/timestamp/bigint/double) plus the date "
                    "column, and AVOID selecting big STRUCT/ARRAY columns whole. Use "
                    "explode() / LATERAL VIEW only if the request truly needs array "
                    "elements. When in doubt, SELECT a handful of scalar columns and the "
                    "date column — a simple, resolvable query beats a rich broken one.\n"
                    f"{proj_line}"
                    f"{hints_line}"
                    "PERFORMANCE: these tables can hold billions of rows. Do NOT add "
                    "ORDER BY unless the request explicitly needs sorted output — a sort "
                    "forces a full scan of the filtered set before any row limit applies "
                    "and is the most common cause of timeouts. Keep the WHERE clause "
                    "selective: ALWAYS include the date range, plus at least one "
                    "selective identifier (a record/document/entity id) when the incident provides "
                    "one — that identifier is what bounds the scan (see the entity "
                    "filter guidance above). If a date/"
                    "timestamp column is stored as a STRING, compare it as an ISO-8601 "
                    "string prefix (e.g. col >= '2026-07-17' AND col < '2026-07-18'), "
                    "which still prunes correctly.\n"
                    "Apply the date range and any entity filters. Do NOT add a LIMIT; "
                    "it is appended automatically. Never write INSERT/UPDATE/DELETE/DROP.\n"
                    # Last: operator guidance may not override mandatory rules.
                    f"{guidance_prompt_line(guidance)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Request: {query.natural_language_query}\n"
                    f"{render_identifiers(query)}"
                    f"date_from: {query.date_from}\ndate_to: {query.date_to}"
                ),
            },
        ]
        result = await self.llm_client.structured_output(
            messages, response_model=SqlQuery, stage="log_retrieval"
        )
        return _normalize_struct_paths(result.query)

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def retrieve(
        self, query: RetrievalQuery, guidance: str = "", on_query=None
    ) -> List[Dict]:
        field_schema = await self._get_field_schema()
        field_map = await map_entities(
            self.llm_client, query, field_schema, self.knowledge_pack
        )
        filter_hints = render_filters(
            field_map,
            query,
            require_all_entities=self.require_all_entities,
            knowledge_pack=self.knowledge_pack,
            identity_keys=self.identity_keys,
        )
        sql = await self._generate_sql(query, field_schema, filter_hints, guidance)
        sql = self._enforce_required_conjunction(sql, field_map, query)
        sql = self._strip_evidence_filters(sql)
        # Before the bounds: a fabricated predicate may be the only partition constraint.
        sql = self._strip_fabricated_filters(sql, query)
        # Before the identity rewrite: a vacuous arm is not a plain comparison.
        sql = strip_vacuous_disjuncts(sql, self.config.get("name", "?"))
        # After both strips; before the identity rewrite (narrower claim on that shape).
        sql = self._enforce_same_column_conjunction(sql, field_map, query)
        # After both strips (a stray undeclared clause costs the whole rewrite) and before bounds.
        sql = self._enforce_identity_scope(sql, field_map)
        # After both OR-group readers; before the three rewrites that reshape the body.
        sql = self._relax_form_conjunction(sql, query)
        # After the rewrite (which may lift a scope) and before the bounds.
        sql = self._enforce_subject_anchor(sql, field_map, query)
        # After the anchor (may share the same splice); before the widening.
        sql = self._enforce_key_presence(sql, field_map, query)
        # After both additive guards and the identity rewrite; before the widenings.
        sql = self._enforce_value_tuples(sql, field_map, query)
        # After the anchor and both strips (a fabricated predicate is dropped, not repaired).
        sql = self._widen_stem_literals(sql, field_map, query)
        # After stem widening (stem first: a shorter form may itself be a pattern candidate).
        sql = self._widen_match_patterns(sql, field_map, query)
        sql = self._enforce_partition_bounds(sql, query)
        # After the partition bound: narrows the pad-widened window.
        sql = self._enforce_event_time_window(sql, query, field_schema)
        sql = self._enforce_epoch_window(sql, query)
        # Last: strip THEN pin (reversed, the strip deletes the pinned conjunct).
        sql = enforce_default_filters(
            sql, self.default_filters, self.config.get("name", "?"), dialect="sql"
        )
        self._report_unreadable_projection(sql)
        self.last_field_map = field_map
        publish_query(self, sql, on_query)
        return await self._execute_sql(sql)

    def _report_unreadable_projection(self, sql: str) -> None:
        """Warn when a declared projection name will not be readable in the returned rows.

        Tests returned names (not text mentions) under both declared and underscore-flattened spellings.
        Silent when the returned set is not knowable from the text (``*``, CTE).
        """
        if not self.projection or not sql:
            return
        returned = returned_names(sql)
        if returned is None:
            return
        missing = [
            name
            for entry in self.projection
            for name in projection_names(entry)
            if not ({name, name.replace(".", "_")} & returned)
        ]
        if missing:
            logger.warning(
                "Source '%s': %d declared projection name(s) will not be readable in the "
                "returned rows — %s. A condition naming one reads `unknown` on every row "
                "while this stage reports success; check the generated SELECT's aliases.",
                self.config.get("name", "?"),
                len(missing),
                ", ".join(missing),
            )

    def _strip_evidence_filters(self, sql: str) -> str:
        """Deterministic backstop for the pack's ``never_filter`` (see query_guards)."""
        return strip_evidence_predicates(
            sql, self.never_filter, self.config.get("name", "?")
        )

    def _strip_fabricated_filters(self, sql: str, query: RetrievalQuery) -> str:
        """Drop predicates putting one entity type's value on another's column."""
        return strip_fabricated_predicates(
            sql,
            source_bindings(self.knowledge_pack, query.target_log_source),
            incident_values(query),
            self.config.get("name", "?"),
        )

    def _enforce_required_conjunction(
        self, sql: str, field_map: Dict[str, str], query: RetrievalQuery = None
    ) -> str:
        """Deterministic backstop for the source's actor key (see query_guards)."""
        return enforce_conjunction(
            sql,
            self._conjunction_fields(field_map, query)
            if query is not None
            else required_fields(field_map, self.require_all_entities),
            self.config.get("name", "?"),
        )

    def _enforce_same_column_conjunction(
        self, sql: str, field_map: Dict[str, str], query: RetrievalQuery = None
    ) -> str:
        """AND the key members this source binds to ONE column (see query_guards)."""
        if query is None:
            return sql
        return enforce_conjunction_same_column(
            sql,
            same_column_conjunctions(
                field_map,
                require_all_entities=self.require_all_entities,
                identity_keys=self.identity_keys,
                present_types=[
                    e.type for e in (query.entities or []) if e.value and e.value != "*"
                ],
                source_name=self.config.get("name", "?"),
            ),
            incident_values(query),
            self.config.get("name", "?"),
        )

    def _relax_form_conjunction(
        self, sql: str, query: RetrievalQuery = None
    ) -> str:
        """OR the forms of one entity type spread across several columns (mirror of _enforce_same_column_conjunction)."""
        if query is None:
            return sql
        return relax_form_conjunction(
            sql,
            form_split_bindings(self.knowledge_pack, query.target_log_source),
            incident_values(query),
            self.config.get("name", "?"),
        )

    def _enforce_identity_scope(self, sql: str, field_map: Dict[str, str]) -> str:
        """Deterministic backstop for an event log's identity shape (see query_guards)."""
        if not (self.identity_scopes or self.identity_synonyms):
            return sql
        name = self.config.get("name", "?")
        return enforce_identity_scope(
            sql,
            resolve_identity_fields(self.identity_scopes, field_map, name),
            resolve_identity_fields(self.identity_synonyms, field_map, name),
            name,
        )

    def _enforce_subject_anchor(
        self, sql: str, field_map: Dict[str, str], query: RetrievalQuery
    ) -> str:
        """ADD the incident's identity when the generated query constrains it nowhere (see query_guards)."""
        identity, scopes = subject_anchor(
            field_map,
            query,
            self.identity_scopes,
            self.identity_synonyms,
            self.knowledge_pack,
            self.config.get("name", "?"),
        )
        return enforce_subject_anchor(
            sql, identity, scopes, self.config.get("name", "?"), dialect="sql"
        )

    def _enforce_key_presence(
        self, sql: str, field_map: Dict[str, str], query: RetrievalQuery = None
    ) -> str:
        """ADD a declared key member the query never constrained (see query_guards)."""
        if query is None:
            return sql
        return enforce_key_presence(
            sql,
            key_presence_values(
                field_map,
                query,
                self._conjunction_fields(field_map, query),
                self.knowledge_pack,
                self.config.get("name", "?"),
            ),
            self.config.get("name", "?"),
            dialect="sql",
        )

    def _enforce_value_tuples(
        self, sql: str, field_map: Dict[str, str], query: RetrievalQuery
    ) -> str:
        """Restore value combinations harvested by an earlier retrieval pass (see query_guards)."""
        return enforce_value_tuples(
            sql,
            value_tuple_columns(field_map, query, self.knowledge_pack),
            self.config.get("name", "?"),
            dialect="sql",
        )

    def _widen_stem_literals(
        self, sql: str, field_map: Dict[str, str], query: RetrievalQuery
    ) -> str:
        """Offer a value's declared identity stem beside it (additive; see query_guards)."""
        return widen_stem_literals(
            sql,
            stem_literals(field_map, query, self.knowledge_pack),
            self.config.get("name", "?"),
            dialect="sql",
        )

    def _widen_match_patterns(
        self, sql: str, field_map: Dict[str, str], query: RetrievalQuery
    ) -> str:
        """Offer a value's declared positional pattern beside it (additive; see query_guards)."""
        return widen_match_patterns(
            sql,
            match_patterns(field_map, query, self.knowledge_pack),
            self.config.get("name", "?"),
            dialect="sql",
        )

    def _conjunction_fields(self, field_map, query):
        """Resolve and stash the key fields to AND; stashed so publish_query reads the same key."""
        fields = conjunction_fields(
            field_map,
            require_all_entities=self.require_all_entities,
            identity_keys=self.identity_keys,
            present_types=[
                e.type for e in (query.entities or []) if e.value and e.value != "*"
            ],
            source_name=self.config.get("name", "?"),
        )
        self.last_conjunction_fields = list(fields)
        return fields

    def _enforce_partition_bounds(self, sql: str, query: RetrievalQuery) -> str:
        """Deterministic backstop for the partition layout (see query_guards)."""
        return enforce_partition_bounds(
            sql,
            self.partitions(),
            query.date_from,
            query.date_to,
            self.config.get("name", "?"),
            dialect="sql",
        )

    def _enforce_event_time_window(
        self, sql: str, query: RetrievalQuery, field_schema: str
    ) -> str:
        """Bound the event-time column the partition pad otherwise stands in for (pack-declared, not LLM-resolved)."""
        resolved = event_time_column(
            field_schema,
            source_bindings(self.knowledge_pack, query.target_log_source),
            self.partitions(),
            self.epoch_time_columns,
            self.config.get("name", "?"),
        )
        if not resolved:
            return sql
        column, col_type = resolved
        return enforce_event_time_window(
            sql,
            column,
            col_type,
            query.date_from,
            query.date_to,
            self.config.get("name", "?"),
            dialect="sql",
        )

    def _enforce_epoch_window(self, sql: str, query: RetrievalQuery) -> str:
        """Deterministic backstop for epoch-integer time columns (see query_guards)."""
        return enforce_epoch_window(
            sql,
            self.epoch_time_columns,
            query.date_from,
            query.date_to,
            self.config.get("name", "?"),
            dialect="sql",
        )

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None


# Matches a backtick-quoted dotted path (e.g. `a.b`); `\w+` excludes the bare `.` between two quoted segments.
_BACKTICKED_DOTTED = re.compile(r"`(\w+(?:\.\w+)+)`")


def _normalize_struct_paths(sql: str) -> str:
    """Rewrite ``\\`a.b.c\\``` → ``\\`a\\`.\\`b\\`.\\`c\\``` so struct field-access resolves in Databricks SQL."""

    def _fix(match):
        segments = match.group(1).split(".")
        return ".".join(f"`{seg}`" for seg in segments)

    return _BACKTICKED_DOTTED.sub(_fix, sql)


def _build_ssl_context(config: Dict):
    """Resolve the aiohttp ssl= value from optional config keys.

    - ``verify_ssl: false``  -> ``False`` (disables TLS verification; prefer ``ca_bundle``).
    - ``ca_bundle: <path>``  -> an ``ssl.SSLContext`` trusting that CA bundle.
    - neither set            -> ``None`` (unchanged default: system trust store).
    """
    if config.get("verify_ssl") is False:
        return False
    ca_bundle = config.get("ca_bundle")
    if ca_bundle:
        return ssl.create_default_context(cafile=ca_bundle)
    return None


# Cap struct flattening depth and leaves per table.
_MAX_STRUCT_DEPTH = 3
_MAX_LEAVES_PER_TABLE = 60

# Cap metadata statements per discovery pass; unchecked tables are named in a warning.
_MAX_RECONCILE_STATEMENTS = 40

# Cap tables DESCRIBEd when a catalog has no information_schema (one statement each).
_MAX_DESCRIBE_TABLES = 20

# Partial struct type marker: "... N more fields" — the only machine-readable signal.
_TRUNCATED_TYPE_RE = re.compile(r"\.\.\.\s*(\d+)\s+more fields?", re.IGNORECASE)


def type_is_partial(type_str: str) -> bool:
    """True when the type string contains the ``... N more fields`` truncation marker."""
    return bool(_TRUNCATED_TYPE_RE.search(type_str or ""))


def hidden_field_count(type_str: str) -> int:
    """How many fields the marker says were omitted; 0 when there is no marker."""
    match = _TRUNCATED_TYPE_RE.search(type_str or "")
    return int(match.group(1)) if match else 0


def struct_child_names(type_str: str) -> List[str]:
    """Top-level field names of a ``STRUCT<...>`` type; returns ``[]`` for non-structs."""
    t = (type_str or "").strip()
    if not t.upper().startswith("STRUCT<") or not t.endswith(">"):
        return []
    names = []
    body = t[len("STRUCT<"):-1]
    for field in _split_top_level(body):
        if ":" not in field:
            continue  # the truncation marker, or an unparseable fragment
        names.append(field.split(":", 1)[0].strip())
    return names


def _live_column_types(described: List[Dict]) -> Dict[str, str]:
    """``{column: type}`` from ``DESCRIBE TABLE`` rows; stops at the first section header (``#``)."""
    types: Dict[str, str] = {}
    for row in described or []:
        name = str(row.get("col_name") or "").strip()
        if not name or name.startswith("#"):
            break
        types[name] = str(row.get("data_type") or "").strip()
    return types


def _columns_from_describe(described: List[Dict], table: str) -> List[Dict]:
    """``DESCRIBE TABLE`` rows as information_schema-shaped discovery rows.

    The output has three regions and only the first two are columns: the column list, then
    a repeated ``# Partition Information`` section naming the partition columns in order,
    then ``# Detailed Table Information``. A partition column appears TWICE, so it is
    matched back onto its own row rather than appended — appended, the generator is shown a
    duplicate leaf and the per-table leaf cap spends itself on it.
    """
    columns: List[Dict] = []
    partitions: List[str] = []
    region = "columns"
    for row in described or []:
        name = str(row.get("col_name") or "").strip()
        if not name:
            continue
        if name.startswith("#"):
            lowered = name.lower()
            if "partition information" in lowered:
                region = "partitions"
            elif "detailed table information" in lowered:
                break
            continue  # a section's own `# col_name` header row
        if region == "partitions":
            if name not in partitions:
                partitions.append(name)
            continue
        col_type = str(row.get("data_type") or "").strip()
        columns.append(
            {
                "table_name": table,
                "column_name": name,
                "data_type": col_type,
                "full_data_type": col_type,
                "partition_index": None,
            }
        )
    for order, name in enumerate(partitions):
        for col in columns:
            if col["column_name"] == name:
                col["partition_index"] = order
                break
    return columns


def struct_type_from_children(children: List[tuple]) -> str:
    """Rebuild ``STRUCT<name:type, ...>`` from ``[(name, type), ...]``."""
    if not children:
        return ""
    body = ", ".join(f"{name}:{ctype}" for name, ctype in children if name)
    return f"struct<{body}>" if body else ""


def _split_top_level(fields: str) -> List[str]:
    """Split a STRUCT/ARRAY field list on commas that are NOT inside angle brackets.

    ``a:INT, b:STRUCT<c:INT, d:STRING>`` -> ``['a:INT', 'b:STRUCT<c:INT, d:STRING>']``.
    """
    parts: List[str] = []
    depth = 0
    current = []
    for ch in fields:
        if ch == "<":
            depth += 1
            current.append(ch)
        elif ch == ">":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


_MAX_LEAF_TYPE_LEN = 120


def _compact_type(t: str) -> str:
    """Shorten a long nested type string so it can't bloat the prompt."""
    return t if len(t) <= _MAX_LEAF_TYPE_LEN else t[: _MAX_LEAF_TYPE_LEN - 3] + "..."


def _flatten_struct(col_name: str, type_str: str, depth: int = 0):
    """Flatten a column type to ``[(dotted_path, leaf_type), ...]``.

    Arrays and maps are terminal leaves — dot-access into an array is invalid SQL.
    Recursion capped at ``_MAX_STRUCT_DEPTH``; a deeper struct surfaces as one leaf.
    """
    t = (type_str or "").strip()
    upper = t.upper()

    if upper.startswith("ARRAY<") or upper.startswith("MAP<"):
        return [(col_name, _compact_type(t))]

    if upper.startswith("STRUCT<") and t.endswith(">"):
        if depth >= _MAX_STRUCT_DEPTH:
            return [(col_name, _compact_type(t))]
        body = t[len("STRUCT<") : -1]
        leaves = []
        for field in _split_top_level(body):
            if ":" not in field:
                continue
            fname, ftype = field.split(":", 1)
            leaves.extend(
                _flatten_struct(f"{col_name}.{fname.strip()}", ftype.strip(), depth + 1)
            )
        return leaves or [(col_name, _compact_type(t))]

    return [(col_name, t)]


def _is_declared_leaf(path: str, declared_lower: List[str]) -> bool:
    """True when ``path`` equals, is an ancestor of, or is a descendant of a declared name."""
    low = (path or "").lower()
    if not low:
        return False
    for dec in declared_lower:
        if low == dec or low.startswith(dec + ".") or dec.startswith(low + "."):
            return True
    return False


def _render_schema(rows: List[Dict], declared: Optional[List[str]] = None) -> str:
    """Render information_schema rows as ``table(col type, ...); ...``; declared leaves first, capped at ``_MAX_LEAVES_PER_TABLE``."""
    declared_lower = sorted(
        {d.lower() for d in (declared or []) if isinstance(d, str) and d},
        key=len,
        reverse=True,
    )
    tables: Dict[str, List[str]] = {}
    for row in rows:
        table = row["table_name"]
        type_str = row.get("full_data_type") or row.get("data_type") or ""
        # Collect uncapped first; declared leaves may be anywhere in the ordinal order.
        tables.setdefault(table, []).extend(
            (path, leaf_type)
            for path, leaf_type in _flatten_struct(row["column_name"], type_str)
        )
    rendered: Dict[str, List[str]] = {}
    for table, leaves in tables.items():
        if declared_lower and len(leaves) > _MAX_LEAVES_PER_TABLE:
            first = [lf for lf in leaves if _is_declared_leaf(lf[0], declared_lower)]
            rest = [lf for lf in leaves if not _is_declared_leaf(lf[0], declared_lower)]
        else:
            first, rest = [], list(leaves)
        ordered = first + rest
        kept = ordered[:_MAX_LEAVES_PER_TABLE]
        rendered[table] = [f"{path} {leaf_type}" for path, leaf_type in kept]
        if len(ordered) > _MAX_LEAVES_PER_TABLE:
            dropped_declared = max(0, len(first) - _MAX_LEAVES_PER_TABLE)
            logger.info(
                "Schema for %s capped at %d of %d leaf fields; %d declared leaf field(s) "
                "kept first%s.",
                table,
                _MAX_LEAVES_PER_TABLE,
                len(ordered),
                min(len(first), _MAX_LEAVES_PER_TABLE),
                (
                    f" — WARNING: {dropped_declared} DECLARED leaf field(s) still did not "
                    "fit, so a binding on them will read as stale; narrow the source's "
                    "entity_bindings/projection or raise the cap"
                    if dropped_declared
                    else ""
                ),
            )
    return "; ".join(f"{name}({', '.join(cols)})" for name, cols in rendered.items())


def _rows_from_statement(data: Dict) -> List[Dict]:
    """Convert a SQL Statement Execution result into a list of row dicts."""
    manifest = data.get("manifest", {})
    columns = [c["name"] for c in manifest.get("schema", {}).get("columns", [])]
    result = data.get("result", {}) or {}
    data_array = result.get("data_array", []) or []
    return [dict(zip(columns, row)) for row in data_array]
