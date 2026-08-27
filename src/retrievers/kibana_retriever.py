"""Retriever for an ES cluster reachable only through a Kibana gateway.

Raw ES paths and ES|QL are blocked; ``POST /internal/search/es`` proxies Query DSL.
Fields are discovered by sampling ``_source`` keys (field-caps is 404 here).
The retriever, never the LLM, injects ``size``.
"""

import logging
import ssl
from pathlib import Path
from typing import Dict, List

import aiohttp

from src.human_guidance import guidance_prompt_line
from src.models.pydantic_models import EsQueryDsl, RetrievalQuery
from src.retrievers.base import DataRetriever, publish_query
from src.retrievers.field_mapping import (event_time_column,
                                          form_split_bindings, incident_values,
                                          key_presence_values, map_entities,
                                          match_patterns, render_filters,
                                          render_identifiers,
                                          source_bindings, stem_literals,
                                          subject_anchor, value_tuple_columns)
from src.retrievers.query_guards import (conjunction_fields,
                                         enforce_conjunction_dsl,
                                         enforce_conjunction_same_column_dsl,
                                         enforce_epoch_window_dsl,
                                         enforce_event_time_window_dsl,
                                         enforce_identity_scope_dsl,
                                         enforce_key_presence_dsl,
                                         enforce_partition_bounds_dsl,
                                         enforce_subject_anchor_dsl,
                                         enforce_value_tuples_dsl,
                                         drop_orphan_minimum_should_match,
                                         epoch_prompt_line, guard_prompt_line,
                                         partition_prompt_line,
                                         relax_form_conjunction_dsl,
                                         resolve_identity_fields,
                                         same_column_conjunctions,
                                         strip_evidence_filters_dsl,
                                         strip_fabricated_filters_dsl,
                                         strip_vacuous_should_clauses,
                                         widen_match_patterns_dsl,
                                         widen_stem_literals_dsl)
from src.utils.error_handling import async_retry_with_backoff
from src.utils.paths import REPO_ROOT

logger = logging.getLogger(__name__)

# Kibana rejects internal requests without these.
_KIBANA_HEADERS = {
    "kbn-xsrf": "true",
    "x-elastic-internal-origin": "Kibana",
    "Content-Type": "application/json",
}
# Per-index sample: coverage comes from the bucket count, not the per-index size.
_SCHEMA_DOCS_PER_INDEX = 3
_SCHEMA_INDEX_BUCKETS = 40
# Flat fallback when the aggregation returns no buckets.
_SCHEMA_SAMPLE_SIZE = 50
# Depth cap for field-name segments. Arrays add no segment (_collect_keys); set from the
# corpus maximum (5), not to it: too tight hides a leaf a pack binding depends on.
_SCHEMA_MAX_SEGMENTS = 6


class KibanaRetriever(DataRetriever):
    def __init__(self, config: Dict, llm_client, knowledge_pack=None):
        self.config = config
        self.llm_client = llm_client
        self.knowledge_pack = knowledge_pack
        # For UI/API visibility.
        self.last_generated_query = None
        self.last_field_map = None
        self.base_url = (config.get("url") or "").rstrip("/")
        self.index = config["index"]
        self.max_results = config.get("max_results", 500)
        self.timeout = config.get("timeout", 30)
        self._username = config.get("username")
        self._password = config.get("password")
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
        self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            auth = None
            if self._username and self._password:
                auth = aiohttp.BasicAuth(self._username, self._password)
            ssl_ctx = _build_ssl_context(self.config)
            connector = (
                aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx is not None else None
            )
            self._session = aiohttp.ClientSession(
                headers=_KIBANA_HEADERS,
                auth=auth,
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        return self._session

    async def _search(self, body: Dict) -> Dict:
        """POST a Query-DSL body through Kibana's Discover backend, return rawResponse."""
        url = f"{self.base_url}/internal/search/es"
        payload = {"params": {"index": self.index, "body": body}}
        async with self._get_session().post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
        return data.get("rawResponse", data)

    async def _get_field_schema(self) -> str:
        """Discover real fields by sampling docs per concrete index, cached after first use.

        Bucketing on ``_index`` (newest first) prevents a flat ``match_all`` from returning
        only the oldest index's shape, which would filter absent fields and return 0 rows.
        Falls back to the configured ``field_schema`` on failure.
        """
        if self._discovered_schema:
            return self._discovered_schema
        try:
            raw = await self._search(
                {
                    "size": 0,
                    "query": {"match_all": {}},
                    "aggs": {
                        "idx": {
                            "terms": {
                                "field": "_index",
                                "size": _SCHEMA_INDEX_BUCKETS,
                                # Date-suffix sorts reverse-chronologically: current shape survives cap.
                                "order": {"_key": "desc"},
                            },
                            "aggs": {
                                "docs": {"top_hits": {"size": _SCHEMA_DOCS_PER_INDEX}}
                            },
                        }
                    },
                }
            )
            hits = _sampled_hits(raw)
            if not hits:
                # `size: 0` leaves nothing to fall back on; re-ask flatly for basic coverage.
                logger.warning(
                    "Per-index field sampling returned no buckets for %s; "
                    "falling back to a flat sample (one document shape may be missed)",
                    self.index,
                )
                raw = await self._search(
                    {"size": _SCHEMA_SAMPLE_SIZE, "query": {"match_all": {}}}
                )
                hits = _sampled_hits(raw)
            fields = set()
            for hit in hits:
                _collect_keys(hit.get("_source", {}), "", fields)
            self._discovered_schema = ", ".join(sorted(fields))
        except Exception as e:
            logger.warning(
                "Kibana field discovery failed for %s; inferring field names: %s",
                self.index,
                e,
            )
            self._discovered_schema = self.field_schema or ""
        return self._discovered_schema

    async def _generate_dsl(
        self,
        query: RetrievalQuery,
        field_schema: str,
        filter_hints: str = "",
        guidance: str = "",
    ) -> Dict:
        if filter_hints:
            filter_line = (
                f"Apply these entity filters using the mapped fields: {filter_hints}.\n"
            )
        else:
            # No filterable entity: restrict to time scope to avoid invented clauses zeroing
            # the result under minimum_should_match:1.
            filter_line = (
                "No incident entity is filterable on this source. Return ONLY the "
                "date-range filter on the timestamp/date field — do NOT add any "
                "term/match/should clauses on other fields.\n"
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
            self.declared_partitions, query.date_from, query.date_to
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
                    "You write Elasticsearch Query DSL. Produce ONE valid query object "
                    f"for a `_search` against index `{self.index}`.\n"
                    "Available fields: "
                    f"{field_schema or 'unknown — infer reasonable field names'}.\n"
                    f"{hints_line}"
                    f"{filter_line}"
                    "Apply the date range as a range filter on the timestamp/date "
                    "field (this is the hard scope, in bool.filter). Treat the entity "
                    "filters as EVIDENCE TO FIND, not all-mandatory: put them in a "
                    "bool.should with minimum_should_match:1 so a record matching ANY "
                    "of them (e.g. the organisational unit, OR the account, OR an "
                    "impacted record) "
                    "is returned — do NOT AND every entity together, that yields zero "
                    "hits when one value differs. A hint of the form "
                    "`field IN (a, b)` means match ANY of those values on that field. "
                    "Use `term` for exact keyword fields; do not use `term` on a free "
                    "text field. Return only the query object (a bool/range/match), "
                    "NOT size/from/sort — the row cap is added automatically.\n"
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
            messages, response_model=EsQueryDsl, stage="log_retrieval"
        )
        return result.query or {"match_all": {}}

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
        self, dsl: Dict, field_map: Dict[str, str], query: RetrievalQuery
    ) -> Dict:
        """AND the members of the key this source binds to one column (see query_guards)."""
        return enforce_conjunction_same_column_dsl(
            dsl,
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
        self, dsl: Dict, query: RetrievalQuery
    ) -> Dict:
        """OR the value forms of one entity type stored across several columns (see query_guards)."""
        return relax_form_conjunction_dsl(
            dsl,
            form_split_bindings(self.knowledge_pack, query.target_log_source),
            incident_values(query),
            self.config.get("name", self.index),
        )

    def _enforce_identity_scope(self, dsl: Dict, field_map: Dict[str, str]) -> Dict:
        """Deterministic backstop for an event log's identity shape (see query_guards)."""
        if not (self.identity_scopes or self.identity_synonyms):
            return dsl
        name = self.config.get("name", self.index)
        return enforce_identity_scope_dsl(
            dsl,
            resolve_identity_fields(self.identity_scopes, field_map, name),
            resolve_identity_fields(self.identity_synonyms, field_map, name),
            name,
        )

    def _enforce_subject_anchor(
        self, dsl: Dict, field_map: Dict[str, str], query: RetrievalQuery
    ) -> Dict:
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
        return enforce_subject_anchor_dsl(dsl, identity, scopes, name)

    def _enforce_key_presence(
        self, dsl: Dict, field_map: Dict[str, str], query: RetrievalQuery
    ) -> Dict:
        """ADD a key member the generated query left unconstrained (see query_guards)."""
        name = self.config.get("name", self.index)
        return enforce_key_presence_dsl(
            dsl,
            key_presence_values(
                field_map,
                query,
                self._conjunction_fields(field_map, query),
                self.knowledge_pack,
                name,
            ),
            name,
        )

    def _enforce_value_tuples(
        self, dsl: Dict, field_map: Dict[str, str], query: RetrievalQuery
    ) -> Dict:
        """Restore the value COMBINATIONS an earlier pass harvested (see query_guards)."""
        return enforce_value_tuples_dsl(
            dsl,
            value_tuple_columns(field_map, query, self.knowledge_pack),
            self.config.get("name", self.index),
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
        dsl = await self._generate_dsl(query, field_schema, filter_hints, guidance)
        # Pack-declared guarantees enforced post-generation.
        dsl = enforce_conjunction_dsl(
            dsl,
            self._conjunction_fields(field_map, query),
            self.config.get("name", self.index),
            # Two routes because neither sees the other's case: declaration and literal lookup.
            form_bindings=form_split_bindings(
                self.knowledge_pack, query.target_log_source
            ),
            incident_values=incident_values(query),
        )
        dsl = strip_evidence_filters_dsl(
            dsl, self.never_filter, self.config.get("name", self.index)
        )
        # Before the bounds: a fabricated clause may be the only partition constraint.
        dsl = strip_fabricated_filters_dsl(
            dsl,
            source_bindings(self.knowledge_pack, query.target_log_source),
            incident_values(query),
            self.config.get("name", self.index),
        )
        # Before the identity rewrite: a vacuous arm is not a plain comparison.
        dsl = strip_vacuous_should_clauses(dsl, self.config.get("name", self.index))
        # After both strips, before the identity rewrite.
        dsl = self._enforce_same_column_conjunction(dsl, field_map, query)
        # After both strips, before the three rewrites below.
        dsl = self._enforce_identity_scope(dsl, field_map)
        # After both OR-group readers, before the three rewrites below.
        dsl = self._relax_form_conjunction(dsl, query)
        # After the identity rewrite, before the bounds.
        dsl = self._enforce_subject_anchor(dsl, field_map, query)
        # After the anchor, before the stem widening.
        dsl = self._enforce_key_presence(dsl, field_map, query)
        # After both additive guards, before both widenings.
        dsl = self._enforce_value_tuples(dsl, field_map, query)
        # After the anchor and both strips.
        dsl = widen_stem_literals_dsl(
            dsl,
            stem_literals(field_map, query, self.knowledge_pack),
            self.config.get("name", self.index),
        )
        # After stem widening: composes in this order only (a stem is itself a pattern candidate).
        dsl = widen_match_patterns_dsl(
            dsl,
            match_patterns(field_map, query, self.knowledge_pack),
            self.config.get("name", self.index),
        )
        # After the strip: a strip can remove the only clause that looked like a bound.
        dsl = enforce_partition_bounds_dsl(
            dsl,
            self.declared_partitions,
            query.date_from,
            query.date_to,
            self.config.get("name", self.index),
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
            dsl = enforce_event_time_window_dsl(
                dsl,
                resolved[0],
                resolved[1],
                query.date_from,
                query.date_to,
                self.config.get("name", self.index),
            )
        # Last: guards above can add or remove range clauses.
        dsl = enforce_epoch_window_dsl(
            dsl,
            self.epoch_time_columns,
            query.date_from,
            query.date_to,
            self.config.get("name", self.index),
        )
        dsl = _apply_default_filters(dsl, self.default_filters)
        # Last: repairs a stripped `should` that left an orphaned `minimum_should_match`.
        dsl = drop_orphan_minimum_should_match(dsl, self.config.get("name", self.index))

        body = {"size": self.max_results, "query": dsl}
        import json as _json

        self.last_field_map = field_map
        publish_query(self, _json.dumps(body), on_query)
        logger.info("Executing Kibana _search against %s: %s", self.index, body)
        raw = await self._search(body)
        return _rows_from_hits(raw)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None


def _apply_default_filters(dsl: Dict, default_filters: Dict) -> Dict:
    """AND mandatory ``{field: value}`` term clauses onto the generated query; no-op if empty."""
    if not default_filters:
        return dsl
    terms = [{"term": {field: value}} for field, value in default_filters.items()]
    inner = dsl or {"match_all": {}}
    # If the LLM already produced a bool, extend its filter rather than nest deeper.
    if isinstance(inner, dict) and "bool" in inner and isinstance(inner["bool"], dict):
        b = dict(inner["bool"])
        existing = b.get("filter", [])
        if isinstance(existing, dict):
            existing = [existing]
        b["filter"] = list(existing) + terms
        return {"bool": b}
    return {"bool": {"filter": terms, "must": [inner]}}


def _rows_from_hits(raw: Dict) -> List[Dict]:
    """Flatten an ES ``_search`` response into a list of ``_source`` row dicts."""
    hits = raw.get("hits", {}).get("hits", [])
    return [hit.get("_source", {}) for hit in hits]


def _sampled_hits(raw: Dict) -> List[Dict]:
    """Hits from the per-index agg, falling back to top-level hits if no buckets."""
    buckets = ((raw.get("aggregations") or {}).get("idx") or {}).get("buckets") or []
    hits: List[Dict] = []
    for bucket in buckets:
        inner = ((bucket.get("docs") or {}).get("hits") or {}).get("hits") or []
        hits.extend(h for h in inner if isinstance(h, dict))
    if hits:
        return hits
    return [h for h in (raw.get("hits", {}).get("hits") or []) if isinstance(h, dict)]


def _collect_keys(obj, prefix: str, out: set, depth: int = 0) -> None:
    """Union ``_source`` keys as dotted paths, depth-capped.

    Arrays are descended WITHOUT adding a segment: ``term: {a.b: v}`` matches whether ``a``
    holds one object or fifty, so an array must be transparent to discovery too.
    """
    if depth >= _SCHEMA_MAX_SEGMENTS:
        return
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                _collect_keys(item, prefix, out, depth)
        return
    if not isinstance(obj, dict):
        return
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        out.add(path)
        if isinstance(value, (dict, list)):
            _collect_keys(value, path, out, depth + 1)


def _build_ssl_context(config: Dict):
    """``False`` / ``SSLContext`` / ``None`` for ``verify_ssl:false`` / ``ca_bundle`` / neither."""
    if config.get("verify_ssl") is False:
        return False
    ca_bundle = config.get("ca_bundle")
    if ca_bundle:
        path = Path(ca_bundle)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return ssl.create_default_context(cafile=str(path))
    return None
