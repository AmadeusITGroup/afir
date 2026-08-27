"""Elasticsearch retriever: LLM generates ES|QL, this retriever injects the LIMIT."""

import logging
from typing import Dict, List

from elasticsearch import AsyncElasticsearch

from src.human_guidance import guidance_prompt_line
from src.models.pydantic_models import EsqlQuery, RetrievalQuery
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
                                         enforce_conjunction_same_column,
                                         enforce_epoch_window,
                                         enforce_event_time_window,
                                         enforce_identity_scope,
                                         enforce_key_presence,
                                         enforce_partition_bounds,
                                         enforce_subject_anchor,
                                         enforce_value_tuples,
                                         epoch_prompt_line, guard_prompt_line,
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


class ElasticsearchRetriever(DataRetriever):
    def __init__(self, config: Dict, llm_client, knowledge_pack=None):
        self.config = config
        self.llm_client = llm_client
        self.knowledge_pack = knowledge_pack
        # For UI/API visibility.
        self.last_generated_query = None
        self.last_field_map = None
        self.index = config["index"]
        self.max_results = config.get("max_results", 500)
        # Configured schema is a starting hint; real fields are discovered.
        self.field_schema = config.get("field_schema", "")
        self._discovered_schema = self.field_schema or None
        self.default_filters = config.get("default_filters") or {}
        self.query_hints = config.get("query_hints", "") or ""
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
        self.identity_synonyms = [f for f in (config.get("identity_synonyms") or []) if f]
        self.never_filter = [f for f in (config.get("never_filter") or []) if f]
        # Pack-declared; ES has no queryable partition metadata.
        self.declared_partitions = [
            p for p in (config.get("partition_columns") or []) if p and p.get("name")
        ]
        # Time columns stored as an epoch integer; unit is pack-declared.
        self.epoch_time_columns = [
            c for c in (config.get("epoch_time_columns") or []) if c and c.get("name")
        ]
        self._client = None

    def _get_client(self) -> AsyncElasticsearch:
        if self._client is None:
            kwargs = {
                "request_timeout": self.config.get("timeout", 30),
                # Never sniff: cluster-internal nodes are unreachable from here.
                "sniff_on_start": False,
                "sniff_on_node_failure": False,
                "sniff_before_requests": False,
            }
            # (None, None) sends a malformed Authorization header.
            username = self.config.get("username")
            password = self.config.get("password")
            if username and password:
                kwargs["basic_auth"] = (username, password)
            # verify_ssl:false disables verification; ca_bundle points at a CA file.
            if self.config.get("verify_ssl") is False:
                kwargs["verify_certs"] = False
                kwargs["ssl_show_warn"] = False
            elif self.config.get("ca_bundle"):
                kwargs["ca_certs"] = self.config["ca_bundle"]
            self._client = AsyncElasticsearch(self.config["url"], **kwargs)
        return self._client

    async def _get_field_schema(self) -> str:
        """Discover real index fields via the field-caps API, cached after first use.

        Falls back to the configured ``field_schema`` (or empty) if discovery fails,
        so a missing/unsupported field-caps endpoint never breaks retrieval.
        """
        if self._discovered_schema:
            return self._discovered_schema
        try:
            response = await self._get_client().field_caps(index=self.index, fields="*")
            self._discovered_schema = _render_field_caps(response)
        except Exception as e:
            logger.warning(
                "ES field discovery failed for %s; inferring field names: %s",
                self.index,
                e,
            )
            self._discovered_schema = self.field_schema or ""
        return self._discovered_schema

    async def _generate_esql(
        self,
        query: RetrievalQuery,
        field_schema: str,
        filter_hints: str = "",
        guidance: str = "",
    ) -> str:
        if filter_hints:
            filter_line = (
                f"Apply these entity filters using the mapped fields: {filter_hints}.\n"
            )
        else:
            # No incident entity maps to a filterable field on this source — restrict
            # to the time scope rather than inventing WHERE clauses that can't match.
            filter_line = (
                "No incident entity is filterable on this source. Filter ONLY on the "
                "date range — do NOT add WHERE clauses on other fields.\n"
            )
        hints_line = (
            f"Source-specific guidance: {self.query_hints}\n"
            if self.query_hints
            else ""
        )
        # Post-generation strip is the guarantee; asking first avoids the rewrite.
        guard = guard_prompt_line(self.never_filter)
        hints_line += f"{guard}\n" if guard else ""
        prune = partition_prompt_line(
            self.declared_partitions, query.date_from, query.date_to, dialect="esql"
        )
        hints_line += f"{prune}\n" if prune else ""
        # Already converted for this incident's window; no hand-written epoch example needed.
        epoch = epoch_prompt_line(
            self.epoch_time_columns, query.date_from, query.date_to
        )
        hints_line += f"{epoch}\n" if epoch else ""
        messages = [
            {
                "role": "system",
                "content": (
                    "You write Elasticsearch ES|QL queries. Produce ONE valid ES|QL query "
                    f"against the index `{self.index}`.\n"
                    f"Index fields: {field_schema or 'unknown — infer reasonable field names'}.\n"
                    f"{hints_line}"
                    f"{filter_line}"
                    "Apply the date range as the hard scope. Treat the entity filters "
                    "as EVIDENCE TO FIND, not all-mandatory: combine them with OR so a "
                    "row matching ANY of them (the organisational unit, OR the account, OR "
                    "an impacted record) is returned — do NOT AND every entity together, that "
                    "yields zero rows when one value differs. A hint of the form "
                    "`field IN (a, b)` means match ANY of those values. "
                    "Do NOT add a LIMIT clause; it is appended automatically.\n"
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
            messages, response_model=EsqlQuery, stage="log_retrieval"
        )
        return result.query

    def _conjunction_fields(self, field_map, query):
        """Fields to AND for this source: stashed so ``publish_query``'s key check reads the
        same tuple the prompt hint named, not a second resolution that could pick differently.
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
        self, esql: str, field_map: Dict[str, str], query: RetrievalQuery
    ) -> str:
        """AND the members of the key this source binds to one column (see query_guards)."""
        return enforce_conjunction_same_column(
            esql,
            same_column_conjunctions(
                field_map,
                require_all_entities=self.require_all_entities,
                identity_keys=self.identity_keys,
                present_types=[
                    e.type for e in (query.entities or []) if e.value and e.value != "*"
                ],
                source_name=self.config.get("name", self.index),
            ),
            incident_values(query),
            self.config.get("name", self.index),
        )

    def _relax_form_conjunction(
        self, esql: str, query: RetrievalQuery
    ) -> str:
        """OR the value forms of one entity type stored across several columns (see query_guards)."""
        return relax_form_conjunction(
            esql,
            form_split_bindings(self.knowledge_pack, query.target_log_source),
            incident_values(query),
            self.config.get("name", self.index),
        )

    def _enforce_identity_scope(self, esql: str, field_map: Dict[str, str]) -> str:
        """Deterministic backstop for an event log's identity shape (see query_guards)."""
        if not (self.identity_scopes or self.identity_synonyms):
            return esql
        name = self.config.get("name", self.index)
        return enforce_identity_scope(
            esql,
            resolve_identity_fields(self.identity_scopes, field_map, name),
            resolve_identity_fields(self.identity_synonyms, field_map, name),
            name,
        )

    def _enforce_subject_anchor(
        self, esql: str, field_map: Dict[str, str], query: RetrievalQuery
    ) -> str:
        """ADD the incident's identity when the generated query constrains it nowhere."""
        name = self.config.get("name", self.index)
        identity, scopes = subject_anchor(
            field_map,
            query,
            self.identity_scopes,
            self.identity_synonyms,
            self.knowledge_pack,
            name,
        )
        return enforce_subject_anchor(esql, identity, scopes, name, dialect="esql")

    def _enforce_key_presence(
        self, esql: str, field_map: Dict[str, str], query: RetrievalQuery = None
    ) -> str:
        """ADD a declared key member the generated query constrains nowhere.

        The additive counterpart to :meth:`_enforce_required_conjunction`, which can only turn
        an ``OR`` it can find into an ``AND`` — see :func:`enforce_key_presence`.
        """
        if query is None:
            return esql
        name = self.config.get("name", self.index)
        return enforce_key_presence(
            esql,
            key_presence_values(
                field_map,
                query,
                self._conjunction_fields(field_map, query),
                self.knowledge_pack,
                name,
            ),
            name,
            dialect="esql",
        )

    def _enforce_value_tuples(
        self, esql: str, field_map: Dict[str, str], query: RetrievalQuery
    ) -> str:
        """Restore the value COMBINATIONS an earlier pass harvested (see query_guards)."""
        return enforce_value_tuples(
            esql,
            value_tuple_columns(field_map, query, self.knowledge_pack),
            self.config.get("name", self.index),
            dialect="esql",
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
        esql = await self._generate_esql(query, field_schema, filter_hints, guidance)
        # Pack-declared guarantees enforced post-generation.
        esql = enforce_conjunction(
            esql,
            self._conjunction_fields(field_map, query),
            self.config.get("name", self.index),
        )
        esql = strip_evidence_predicates(
            esql,
            self.never_filter,
            self.config.get("name", self.index),
            pipe_stages=True,
        )
        # Before the bounds: a fabricated predicate may be the only partition constraint.
        esql = strip_fabricated_predicates(
            esql,
            source_bindings(self.knowledge_pack, query.target_log_source),
            incident_values(query),
            self.config.get("name", self.index),
            pipe_stages=True,
        )
        # Before the identity rewrite: a vacuous arm is not a plain comparison.
        esql = strip_vacuous_disjuncts(esql, self.config.get("name", self.index))
        # After both strips, before the identity rewrite.
        esql = self._enforce_same_column_conjunction(esql, field_map, query)
        # After both strips, before the three rewrites below.
        esql = self._enforce_identity_scope(esql, field_map)
        # After both OR-group readers, before the three rewrites below.
        esql = self._relax_form_conjunction(esql, query)
        # After the identity rewrite, before the bounds.
        esql = self._enforce_subject_anchor(esql, field_map, query)
        # After the anchor, before the stem widening.
        esql = self._enforce_key_presence(esql, field_map, query)
        # After both additive guards, before both widenings; see docs/architecture/retrieval.md.
        esql = self._enforce_value_tuples(esql, field_map, query)
        # After the anchor and both strips.
        esql = widen_stem_literals(
            esql,
            stem_literals(field_map, query, self.knowledge_pack),
            self.config.get("name", self.index),
            dialect="esql",
        )
        # After stem widening: composes in this order only (a stem is itself a pattern candidate).
        esql = widen_match_patterns(
            esql,
            match_patterns(field_map, query, self.knowledge_pack),
            self.config.get("name", self.index),
            dialect="esql",
        )
        # After the strip: a strip can remove the only clause that looked like a bound.
        esql = enforce_partition_bounds(
            esql,
            self.declared_partitions,
            query.date_from,
            query.date_to,
            self.config.get("name", self.index),
            dialect="esql",
        )
        # After the partition bound: narrows the window the partition pad widened.
        resolved = event_time_column(
            field_schema,
            source_bindings(self.knowledge_pack, query.target_log_source),
            self.declared_partitions,
            self.epoch_time_columns,
            self.config.get("name", self.index),
        )
        if resolved:
            esql = enforce_event_time_window(
                esql,
                resolved[0],
                resolved[1],
                query.date_from,
                query.date_to,
                self.config.get("name", self.index),
                dialect="esql",
            )
        # Last: guards above can add or remove range predicates.
        esql = enforce_epoch_window(
            esql,
            self.epoch_time_columns,
            query.date_from,
            query.date_to,
            self.config.get("name", self.index),
            dialect="esql",
        )
        esql = _apply_default_filters_esql(esql, self.default_filters)
        # The retriever owns the row cap, not the LLM.
        if "limit" not in esql.lower():
            esql = f"{esql.rstrip().rstrip(';')} | LIMIT {self.max_results}"

        self.last_field_map = field_map
        publish_query(self, esql, on_query)
        return await self._execute_esql(esql)

    async def _execute_esql(self, esql: str) -> List[Dict]:
        """Run one ES|QL statement and return rows as dicts.

        Named seam so callers outside the pipeline (e.g. ``pack_probe``) can reach this
        backend without duplicating client construction or response flattening.
        """
        logger.info("Executing ES|QL against %s: %s", self.index, esql)
        response = await self._get_client().esql.query(query=esql)
        return _rows_from_esql(response)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def _apply_default_filters_esql(esql: str, default_filters: Dict) -> str:
    """Append mandatory ``field == "value"`` equality filters as an ES|QL WHERE.

    A no-op when there are no default_filters. Values are string-quoted (a detector
    discriminator is a string enum). Appended as a separate ``| WHERE`` stage so it
    AND-combines with any WHERE the LLM already produced.
    """
    if not default_filters:
        return esql
    clauses = " AND ".join(
        f'{field} == "{value}"' for field, value in default_filters.items()
    )
    return f"{esql.rstrip().rstrip(';')} | WHERE {clauses}"


def _rows_from_esql(response) -> List[Dict]:
    """Convert an ES|QL columnar response into a list of row dicts."""
    body = response.body if hasattr(response, "body") else response
    columns = [c["name"] for c in body.get("columns", [])]
    values = body.get("values", [])
    return [dict(zip(columns, row)) for row in values]


def _render_field_caps(response) -> str:
    """Render a field-caps response as ``field: type, field: type, ...``.

    The ``fields`` map is ``{field_name: {type: {...}}}``; we take the first
    declared type per field. Metadata fields (leading ``_``) are skipped.
    """
    body = response.body if hasattr(response, "body") else response
    fields = body.get("fields", {}) or {}
    parts = []
    for name in sorted(fields):
        if name.startswith("_"):
            continue
        types = fields[name]
        type_name = next(iter(types), "") if isinstance(types, dict) else ""
        parts.append(f"{name}: {type_name}" if type_name else name)
    return ", ".join(parts)
