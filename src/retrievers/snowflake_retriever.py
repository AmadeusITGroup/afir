"""
Snowflake retriever.

Mirrors ``DatabricksRetriever``: the LLM generates a read-only SQL SELECT from the
natural-language query plus a discovered field schema, and the retriever executes it
against Snowflake. The ``snowflake-connector-python`` driver is synchronous, so every
blocking call runs in a worker thread via ``asyncio.to_thread``.

If ``field_schema`` is not configured, the retriever discovers real columns from
``information_schema.columns`` across the configured databases/schemas on first use
and caches them, so the LLM always writes SQL against columns that actually exist.
"""

import asyncio
import logging
import os
import re
from typing import Dict, List

from src.human_guidance import guidance_prompt_line
from src.models.pydantic_models import RetrievalQuery, SqlQuery
from src.retrievers.base import DataRetriever, publish_query
from src.retrievers.field_mapping import (event_time_column,
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
                                         resolve_identity_fields,
                                         same_column_conjunctions,
                                         strip_evidence_predicates,
                                         strip_fabricated_predicates,
                                         strip_vacuous_disjuncts,
                                         widen_match_patterns,
                                         widen_stem_literals)
from src.utils.error_handling import async_retry_with_backoff

logger = logging.getLogger(__name__)


class SnowflakeRetriever(DataRetriever):
    def __init__(self, config: Dict, llm_client, knowledge_pack=None):
        self.config = config
        self.llm_client = llm_client
        self.knowledge_pack = knowledge_pack

        self.account = config.get("account")
        self.user = config.get("user")
        self.password = (
            os.getenv(config["password_env"]) if config.get("password_env") else None
        ) or config.get("password")
        self.private_key = (
            os.getenv(config["private_key_env"])
            if config.get("private_key_env")
            else None
        )
        self.role = config.get("role")
        self.warehouse = config.get("warehouse")
        # Logical scope (from the pack endpoints): databases/schemas/objects.
        self.databases = config.get("databases", []) or []
        self.schemas = config.get("schemas", []) or []
        self.objects = config.get("objects", []) or []
        self.database = config.get("database") or (
            self.databases[0] if self.databases else None
        )
        self.schema = config.get("schema") or (
            self.schemas[0] if self.schemas else None
        )
        self._field_schema = config.get("field_schema") or None
        self.max_results = config.get("max_results", 500)
        # Last generated backend query + entity mapping, for UI/API visibility.
        self.last_generated_query = None
        self.last_field_map = None
        # Pack-declared query guarantees, enforced deterministically after generation
        # (see src/retrievers/query_guards.py) because the generation prompt carries
        # general instructions that contradict them.
        self.require_all_entities = [
            e for e in (config.get("require_all_entities") or []) if e
        ]
        # PRIORITY-ORDERED candidate actor keys (pack `identity_keys`). The first candidate this
        # incident can fully satisfy wins, so a missing component falls through to the next
        # complete key rather than collapsing the conjunction to one field.
        self.identity_keys = [
            [t for t in (cand or []) if t]
            for cand in (config.get("identity_keys") or [])
        ]
        # The EVENT-LOG identity shape: families of interchangeable columns (OR inside, AND
        # between) plus separate scope facts. Wired on every backend, being a property of the
        # SOURCE's key shape rather than of the engine storing it.
        self.identity_scopes = [f for f in (config.get("identity_scopes") or []) if f]
        self.identity_synonyms = [f for f in (config.get("identity_synonyms") or []) if f]
        self.never_filter = [f for f in (config.get("never_filter") or []) if f]
        # The slice of a shared table this source IS: `{field: value}`, pinned rather than asked
        # for and AND-ed on after every guard. Read on every backend, because a declaration
        # honoured on one query family and dropped on another is a silent no-op.
        self.default_filters = config.get("default_filters") or {}
        # Columns every query must bound so the backend prunes. Snowflake's analogue is the
        # CLUSTERING KEY (micro-partitions being automatic), reported by
        # `information_schema.tables.clustering_key`, so it is DISCOVERED like the Databricks
        # partition_index with the pack's declaration layered on top.
        self.declared_partitions = [
            p for p in (config.get("partition_columns") or []) if p and p.get("name")
        ]
        self._discovered_partitions: List[Dict] = []
        # Time columns stored as an epoch INTEGER rather than a DATE/TIMESTAMP. Pack-declared
        # (no catalog reports the UNIT of an integer column) and enforced, because a wrong
        # epoch literal is a plausible-looking number that matches nothing.
        self.epoch_time_columns = [
            c for c in (config.get("epoch_time_columns") or []) if c and c.get("name")
        ]
        self._conn = None

    def partitions(self) -> List[Dict]:
        """Prune columns in force: discovered clustering keys + the pack's declarations."""
        return merge_partition_specs(
            self._discovered_partitions, self.declared_partitions
        )

    def _connect(self):
        """Open a Snowflake connection (sync; called inside a worker thread)."""
        import snowflake.connector

        kwargs = {
            "account": self.account,
            "user": self.user,
            "role": self.role,
            "warehouse": self.warehouse,
        }
        if self.database:
            kwargs["database"] = self.database
        if self.schema:
            kwargs["schema"] = self.schema
        if self.private_key:
            kwargs["private_key"] = self.private_key.encode()
        else:
            kwargs["password"] = self.password
        return snowflake.connector.connect(**kwargs)

    def _execute_sql_sync(self, sql: str) -> List[Dict]:
        if self._conn is None:
            self._conn = self._connect()
        cur = self._conn.cursor()
        try:
            cur.execute(sql)
            columns = [c[0] for c in cur.description] if cur.description else []
            rows = cur.fetchmany(self.max_results)
            return [dict(zip(columns, row)) for row in rows]
        finally:
            cur.close()

    async def _execute_sql(self, sql: str) -> List[Dict]:
        logger.info("Executing Snowflake SQL: %s", sql)
        return await asyncio.to_thread(self._execute_sql_sync, sql)

    async def _get_field_schema(self) -> str:
        """Discover real columns from information_schema, cached after first use."""
        if self._field_schema:
            return self._field_schema
        if not self.database:
            return ""
        try:
            rows = await self._discover_columns()
        except Exception as e:
            logger.warning(
                "Snowflake schema discovery failed; the LLM will infer columns: %s", e
            )
            # Cache the failure as empty-but-set so we don't rescan on every call.
            self._field_schema = ""
            return ""
        # Clustering keys are discovered in the same pass, best-effort: a failure here must
        # not cost the column schema (the query would then be built blind).
        try:
            self._discovered_partitions = await self._discover_clustering_keys()
            if self._discovered_partitions:
                logger.info(
                    "Source '%s': discovered clustering key column(s) %s — queries will be "
                    "bounded on them.",
                    self.config.get("name", "?"),
                    ", ".join(p["name"] for p in self._discovered_partitions),
                )
        except Exception as e:  # noqa: BLE001
            logger.info("Snowflake clustering-key discovery failed (continuing): %s", e)
        self._field_schema = _render_schema(rows)
        return self._field_schema

    async def _discover_clustering_keys(self) -> List[Dict]:
        """Clustering-key columns of the configured objects, from information_schema.

        ``clustering_key`` is reported as the DDL fragment ``LINEAR(a, b)`` (or with
        expressions such as ``to_date(ts)``). Only bare column references are usable as a
        bound — an expression is skipped rather than guessed at, and the pack can declare the
        underlying column if that case ever matters.
        """
        where = []
        if self.schemas:
            quoted = ", ".join(f"'{s}'" for s in self.schemas)
            where.append(f"table_schema IN ({quoted})")
        elif self.schema:
            where.append(f"table_schema = '{self.schema}'")
        if self.objects:
            quoted = ", ".join(f"'{o}'" for o in self.objects)
            where.append(f"table_name IN ({quoted})")
        where.append("clustering_key IS NOT NULL")
        rows = await self._execute_sql(
            "SELECT table_name, clustering_key "
            f"FROM {self.database}.information_schema.tables "
            "WHERE " + " AND ".join(where)
        )
        specs: List[Dict] = []
        seen = set()
        for row in rows or []:
            raw = str(row.get("clustering_key") or row.get("CLUSTERING_KEY") or "")
            inner = raw[raw.find("(") + 1 : raw.rfind(")")] if "(" in raw else raw
            for part in inner.split(","):
                name = part.strip().strip('"')
                # Bare identifier only — an expression key cannot be bounded verbatim.
                if not name or not re.fullmatch(r"\w+", name) or name in seen:
                    continue
                seen.add(name)
                specs.append({"name": name, "type": ""})
        return specs

    async def _discover_columns(self) -> List[Dict]:
        """Columns for the configured schemas/objects from information_schema.columns."""
        where = []
        if self.schemas:
            quoted = ", ".join(f"'{s}'" for s in self.schemas)
            where.append(f"table_schema IN ({quoted})")
        elif self.schema:
            where.append(f"table_schema = '{self.schema}'")
        if self.objects:
            quoted = ", ".join(f"'{o}'" for o in self.objects)
            where.append(f"table_name IN ({quoted})")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        return await self._execute_sql(
            "SELECT table_name, column_name, data_type "
            f"FROM {self.database}.information_schema.columns{clause} "
            "ORDER BY table_name, ordinal_position"
        )

    async def _generate_sql(
        self,
        query: RetrievalQuery,
        field_schema: str,
        filter_hints: str = "",
        guidance: str = "",
    ) -> str:
        target = ".".join(p for p in (self.database, self.schema) if p)
        filter_line = (
            f"Apply these entity filters using the mapped columns: {filter_hints}.\n"
            if filter_hints
            else ""
        )
        # Generated from the pack's `never_filter` (the post-generation strip is the
        # guarantee; stating it up front avoids needing the rewrite).
        guard = guard_prompt_line(self.never_filter)
        guard_line = f"{guard}\n" if guard else ""
        prune = partition_prompt_line(
            self.partitions(), query.date_from, query.date_to, dialect="sql"
        )
        guard_line += f"{prune}\n" if prune else ""
        # This incident's window, already converted for any epoch-integer time column — so no
        # source has to carry a hand-written epoch example that can (and did) go stale.
        epoch = epoch_prompt_line(
            self.epoch_time_columns, query.date_from, query.date_to
        )
        guard_line += f"{epoch}\n" if epoch else ""
        messages = [
            {
                "role": "system",
                "content": (
                    "You write Snowflake ANSI SQL SELECT queries. Produce ONE read-only "
                    "SELECT statement.\n"
                    f"{'Tables live under: ' + target if target else ''}\n"
                    f"Schema: {field_schema or 'unknown — infer reasonable table/column names'}.\n"
                    f"{filter_line}"
                    f"{guard_line}"
                    "Apply the date range and any entity filters given above. Do NOT add a "
                    "LIMIT; "
                    "it is bounded automatically. Never write INSERT/UPDATE/DELETE/DROP.\n"
                    # Last, so it is read after the mandatory rules it may not override.
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
        return result.query

    def _conjunction_fields(self, field_map, query):
        """The fields to AND for this source, resolved once per query.

        Same resolver `render_filters` used for the prompt hint, so the instruction the
        generator saw and the rewrite applied afterwards always name the same key.

        The resolved key is STASHED, because `publish_query` later has to answer "did the
        query that ran actually constrain this key" and must not re-resolve it: a second
        resolution could pick a different candidate and then verify the wrong tuple, which
        is the hint-vs-guard disagreement `conjunction_fields` exists to make impossible.
        """
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

    def _enforce_same_column_conjunction(
        self, sql: str, field_map: Dict[str, str], query: RetrievalQuery
    ) -> str:
        """AND the members of the key this source binds to ONE column (see query_guards).

        The same declaration `_conjunction_fields` resolves, on the source shape that resolves
        it to a single field and so silences the ordinary rewrite.
        """
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
        self, sql: str, query: RetrievalQuery
    ) -> str:
        """OR the value FORMS of one entity type this source stores in SEVERAL columns.

        The mirror of :meth:`_enforce_same_column_conjunction`, and the same one-sentence rule:
        AND between entity TYPES, OR within one type's values. That guard reaches it where a
        source collapses several types onto ONE column; this one where a source spreads ONE
        type's forms across several, and the AND there requires a single row to carry both
        spellings of the same subject (see query_guards for the measurement).
        """
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
        """ADD the incident's identity when the generated query constrains it nowhere."""
        name = self.config.get("name", "?")
        identity, scopes = subject_anchor(
            field_map,
            query,
            self.identity_scopes,
            self.identity_synonyms,
            self.knowledge_pack,
            name,
        )
        return enforce_subject_anchor(sql, identity, scopes, name, dialect="sql")

    def _enforce_key_presence(
        self, sql: str, field_map: Dict[str, str], query: RetrievalQuery = None
    ) -> str:
        """ADD a declared key member the generated query constrains nowhere.

        The additive counterpart to :meth:`_enforce_required_conjunction`, which can only turn
        an ``OR`` it can find into an ``AND`` — see :func:`enforce_key_presence`.
        """
        if query is None:
            return sql
        name = self.config.get("name", "?")
        return enforce_key_presence(
            sql,
            key_presence_values(
                field_map,
                query,
                self._conjunction_fields(field_map, query),
                self.knowledge_pack,
                name,
            ),
            name,
            dialect="sql",
        )

    def _enforce_value_tuples(
        self, sql: str, field_map: Dict[str, str], query: RetrievalQuery
    ) -> str:
        """Restore the value COMBINATIONS an earlier pass harvested (see query_guards)."""
        return enforce_value_tuples(
            sql,
            value_tuple_columns(field_map, query, self.knowledge_pack),
            self.config.get("name", "?"),
            dialect="sql",
        )

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
        # Pack-declared guarantees, enforced rather than requested (see query_guards).
        name = self.config.get("name", "?")
        sql = enforce_conjunction(
            sql, self._conjunction_fields(field_map, query), name
        )
        sql = strip_evidence_predicates(sql, self.never_filter, name)
        # Before the bounds, like the strip above: a fabricated predicate may be the only
        # thing constraining a prune column, and dropping it after would leave no bound.
        sql = strip_fabricated_predicates(
            sql,
            source_bindings(self.knowledge_pack, query.target_log_source),
            incident_values(query),
            name,
        )
        # BEFORE the identity rewrite: an arm reading `f LIKE 'X%' OR f IS NOT NULL` is not a
        # plain comparison, so that rewrite would decline on the whole group; dropping the
        # vacuous half first leaves a group it can act on.
        sql = strip_vacuous_disjuncts(sql, name)
        # AFTER both strips: this declines a whole OR-group unless EVERY arm is one of the
        # incident's values, so a fabricated or vacuous arm costs the rewrite. BEFORE the
        # identity rewrite, being the narrower claim on the same OR-groups: it acts only where
        # every arm sits on ONE declared key column.
        sql = self._enforce_same_column_conjunction(sql, field_map, query)
        # AFTER both strips and before the bounds: this declines an OR-group holding a predicate
        # on an undeclared field, having no family to place it in, so one stray clause left in
        # place would cost the whole rewrite.
        sql = self._enforce_identity_scope(sql, field_map)
        # AFTER both OR-group readers and BEFORE the three rewrites below. Run first, the
        # `(form_a OR form_b)` this makes is flattened back into an AND by
        # `_enforce_same_column_conjunction` or half-promoted out of the group by
        # `_enforce_identity_scope`; run later, the rewrites leave a conjunct it cannot parse.
        sql = self._relax_form_conjunction(sql, query)
        # AFTER the rewrite (which may lift a scope out of an OR-group, and this must see
        # the result) and before the bounds: the additive counterpart, for a query that
        # constrains the incident's identity on no column at all.
        sql = self._enforce_subject_anchor(sql, field_map, query)
        # AFTER the anchor — the same splice on a different declaration, so running first would
        # inject the same member twice — and BEFORE the widening, which offers a member's stem
        # beside the literal this may just have added.
        sql = self._enforce_key_presence(sql, field_map, query)
        # AFTER both additive guards and BEFORE the two widenings; every neighbour is
        # load-bearing. The additive pair splices `WHERE <clause> AND (<old body>)`, and that
        # clause is the per-TYPE value LISTS this guard narrows, so run first it reads components
        # still buried in an OR-group and declines. A widening run first turns a component's
        # equality into a `LIKE` group it cannot read; the identity rewrite must stay ahead of it,
        # or the OR-of-ANDs this produces is flattened into `unit = 'U1' AND unit = 'U2'`.
        # Measurements in docs/architecture/retrieval.md.
        sql = self._enforce_value_tuples(sql, field_map, query)
        # AFTER the anchor, which may be the very predicate that needs widening, and after both
        # strips: a fabricated predicate is DROPPED, not repaired.
        sql = widen_stem_literals(
            sql, stem_literals(field_map, query, self.knowledge_pack), name, dialect="sql"
        )
        # AFTER the stem widening and BEFORE the bounds. Both are the same additive repair — a
        # literal the column cannot match as written — and they compose in one order only: a stem
        # offers a shorter form of the value, itself a candidate for a positional pattern, while a
        # pattern rewritten first leaves a `LIKE` the stem guard does not read.
        sql = widen_match_patterns(
            sql,
            match_patterns(field_map, query, self.knowledge_pack),
            name,
            dialect="sql",
        )
        sql = enforce_partition_bounds(
            sql, self.partitions(), query.date_from, query.date_to, name, dialect="sql"
        )
        # AFTER the partition bound and alongside the epoch repair: this narrows the window that
        # bound's pad widened, so it must see it. Which column qualifies is the pack's
        # declaration, resolved through the one seam both query families share and never the
        # mapper's per-run LLM resolution, which is absent on exactly the runs that need it.
        resolved = event_time_column(
            field_schema,
            source_bindings(self.knowledge_pack, query.target_log_source),
            self.partitions(),
            self.epoch_time_columns,
            name,
        )
        if resolved:
            sql = enforce_event_time_window(
                sql,
                resolved[0],
                resolved[1],
                query.date_from,
                query.date_to,
                name,
                dialect="sql",
            )
        # Last, so it sees the final WHERE — the injections above can add range predicates.
        sql = enforce_epoch_window(
            sql,
            self.epoch_time_columns,
            query.date_from,
            query.date_to,
            name,
            dialect="sql",
        )
        # AND the pack's pinned slice on LAST, after every guard including the `never_filter`
        # strip that removes the same constant from wherever the generator misplaced it.
        # Reversed, the strip would delete the conjunct this writes.
        sql = enforce_default_filters(sql, self.default_filters, name, dialect="sql")
        self.last_field_map = field_map
        publish_query(self, sql, on_query)
        return await self._execute_sql(sql)

    async def close(self) -> None:
        if self._conn is not None:
            conn, self._conn = self._conn, None
            await asyncio.to_thread(conn.close)


def _render_schema(rows: List[Dict]) -> str:
    """Render information_schema column rows as ``table(col type, ...); table(...)``.

    Snowflake upper-cases unquoted identifiers, so column keys may be TABLE_NAME etc.
    """
    tables: Dict[str, List[str]] = {}
    for row in rows:
        name = row.get("TABLE_NAME") or row.get("table_name")
        col = row.get("COLUMN_NAME") or row.get("column_name")
        dtype = row.get("DATA_TYPE") or row.get("data_type")
        if name and col:
            tables.setdefault(name, []).append(f"{col} {dtype}")
    return "; ".join(f"{name}({', '.join(cols)})" for name, cols in tables.items())
