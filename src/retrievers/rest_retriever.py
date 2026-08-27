"""
REST retriever (ServiceNow Table API).

ServiceNow exposes records over ``GET /api/now/table/<table>`` with an encoded
``sysparm_query`` filter (``field=value^field2=value2``). The retriever maps
incident entities to ServiceNow field names (using the knowledge pack's per-source
``entity_bindings`` as the candidate field schema), builds the encoded query, and
paginates with ``sysparm_limit``/``sysparm_offset`` up to ``max_results``.

Auth is HTTP basic (username/password) or a Bearer token (``token_env``).
"""

import logging
import os
from typing import Dict, List

import aiohttp

from src.models.pydantic_models import RetrievalQuery
from src.retrievers.base import DataRetriever, publish_query
from src.retrievers.field_mapping import map_entities
from src.retrievers.query_guards import (epoch_clauses_encoded, leaf_of,
                                         partition_clauses_encoded)
from src.utils.error_handling import async_retry_with_backoff

logger = logging.getLogger(__name__)


class RestRetriever(DataRetriever):
    def __init__(self, config: Dict, llm_client, knowledge_pack=None):
        self.config = config
        self.llm_client = llm_client
        self.knowledge_pack = knowledge_pack

        self.base_url = (config.get("base_url") or "").rstrip("/")
        self.name = config.get("name")
        self.tables = config.get("tables", []) or []
        self.username = config.get("username")
        self.password = config.get("password")
        self.token = os.getenv(config["token_env"]) if config.get("token_env") else None
        self.timeout = config.get("timeout", 30)
        self.max_results = config.get("max_results", 500)
        self.page_size = min(config.get("page_size", 100), self.max_results)
        # Fields the pack marks as EVIDENCE: returnable, never filterable. The encoded
        # query is built in code here (so require_all_entities needs no rewrite — `^` is
        # already AND), but a never_filter field must still not become a clause.
        self.never_filter = [f for f in (config.get("never_filter") or []) if f]
        # Prune columns every query must bound. A REST/ServiceNow table has no partition
        # metadata to discover, so this is pack-declared only — but it is honoured here for
        # the same reason as everywhere else: the declaration belongs to the SOURCE, and a
        # field that silently no-ops on one backend is the bug the guards exist to prevent.
        self.declared_partitions = [
            p for p in (config.get("partition_columns") or []) if p and p.get("name")
        ]
        # Time columns stored as an epoch INTEGER. Nothing here is LLM-generated, so there is
        # nothing to repair — the bounds are simply appended. Wired for the same reason as
        # above: a pack field that no-ops on one backend is the bug the guards prevent.
        self.epoch_time_columns = [
            c for c in (config.get("epoch_time_columns") or []) if c and c.get("name")
        ]
        self._session = None
        # Last query published to the UI/API. Present on every retriever so the seam that
        # reports it does not have to know which backend it is talking to.
        self.last_generated_query = None
        self.last_field_map = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"Accept": "application/json"}
            auth = None
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            elif self.username is not None:
                auth = aiohttp.BasicAuth(self.username, self.password or "")
            self._session = aiohttp.ClientSession(
                headers=headers,
                auth=auth,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        return self._session

    def _field_schema(self) -> str:
        """Candidate ServiceNow fields from the pack's per-source entity_bindings."""
        pack = self.knowledge_pack
        if pack is None or not self.name:
            return ""
        src = pack.source(self.name)
        if src is None:
            return ""
        fields = []
        for binding in (src.entity_bindings or {}).values():
            # A binding is a list of columns OR a ``{value form: [column]}`` mapping, and
            # iterating the mapping yields the FORM NAMES. Flattened the same way
            # ``field_priors_for`` and ``_binding_paths`` do it: which form a column holds is
            # not this schema hint's question, only which columns exist — and offering a form
            # name as a candidate field would advertise a column the backend does not have,
            # while hiding the real one. The generator is told to infer field names, so it
            # would write the plausible-looking name and the predicate would match nothing.
            if isinstance(binding, dict):
                for per_form in binding.values():
                    if isinstance(per_form, str):
                        fields.append(per_form)
                    elif isinstance(per_form, list):
                        fields.extend(str(f) for f in per_form)
            elif isinstance(binding, str):
                fields.append(binding)
            elif isinstance(binding, list):
                fields.extend(str(f) for f in binding)
        # De-dupe preserving order.
        seen = set()
        return ", ".join(f for f in fields if not (f in seen or seen.add(f)))

    def _build_sysparm_query(
        self, field_map: Dict[str, str], query: RetrievalQuery
    ) -> str:
        """ServiceNow encoded query: field=value^field2=value2 (+ optional date range).

        Built in code rather than by the LLM, so the pack's ``require_all_entities`` needs
        no rewriting (``^`` is already AND). ``never_filter`` still applies: a field the
        pack marks as evidence must not become a clause here either.
        """
        by_type = {e.type: e.value for e in (query.entities or [])}
        never = {leaf_of(f).lower() for f in self.never_filter}
        clauses = []
        for entity_type, field in field_map.items():
            if leaf_of(field).lower() in never:
                logger.warning(
                    "Source '%s': dropped filter on evidence field '%s' — the pack marks "
                    "it never_filter (it must be returned, not filtered on).",
                    self.name or "?",
                    field,
                )
                continue
            value = by_type.get(entity_type)
            if value and value != "*":
                clauses.append(f"{field}={value}")
        # Add a created-on date range when present (ServiceNow uses >=/<= on sys_created_on).
        if query.date_from:
            clauses.append(f"sys_created_on>={query.date_from}")
        if query.date_to:
            clauses.append(f"sys_created_on<={query.date_to}")
        # Bound any declared prune column. `^` is already AND, and the clause list is built
        # here rather than generated, so this is an append with nothing to rewrite.
        clauses.extend(
            partition_clauses_encoded(
                self.declared_partitions,
                query.date_from,
                query.date_to,
                already=["sys_created_on"],
            )
        )
        clauses.extend(
            epoch_clauses_encoded(
                self.epoch_time_columns,
                query.date_from,
                query.date_to,
                already=["sys_created_on"],
            )
        )
        return "^".join(clauses)

    async def _query_table(self, table: str, sysparm_query: str) -> List[Dict]:
        session = self._get_session()
        url = f"{self.base_url}/api/now/table/{table}"
        rows: List[Dict] = []
        offset = 0
        while len(rows) < self.max_results:
            params = {
                "sysparm_limit": str(self.page_size),
                "sysparm_offset": str(offset),
                "sysparm_display_value": "true",
            }
            if sysparm_query:
                params["sysparm_query"] = sysparm_query
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            batch = data.get("result", []) or []
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < self.page_size:
                break
            offset += self.page_size
        return rows[: self.max_results]

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def retrieve(
        self, query: RetrievalQuery, guidance: str = "", on_query=None
    ) -> List[Dict]:
        # `guidance` is accepted for contract parity and ignored: this retriever builds
        # its sysparm_query deterministically from the entity field map, with no
        # generation prompt for analyst direction to steer. `on_query` is honoured
        # though — a deterministically-built query is exactly as worth reviewing.
        field_schema = self._field_schema()
        field_map = await map_entities(
            self.llm_client, query, field_schema, self.knowledge_pack
        )
        sysparm_query = self._build_sysparm_query(field_map, query)
        self.last_field_map = field_map
        publish_query(self, sysparm_query, on_query)
        tables = self.tables or ["incident"]
        results: List[Dict] = []
        for table in tables:
            logger.info(
                "Querying ServiceNow table %s with sysparm_query=%r",
                table,
                sysparm_query,
            )
            rows = await self._query_table(table, sysparm_query)
            for row in rows:
                row["_servicenow_table"] = table
            results.extend(rows)
            if len(results) >= self.max_results:
                break
        return results[: self.max_results]

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
