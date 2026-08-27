"""Knowledge source backed by a Databricks Unity Catalog table.

Reads rows from a UC table via the SQL Statement Execution API (the same auth +
async-poll pattern as ``DatabricksRetriever``) and maps each row to a homogeneous
knowledge document. The row->doc mapping is fully config-driven (which column is
the title, which is the content, which become metadata), so any table shape works
without code changes — this is the "wire any Databricks knowledge base easily" path.

For knowledge that lives as files in a UC Volume, point a ``DocumentSource`` at the
volume mount instead; this class is for tabular knowledge.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import aiohttp

from src.rag.sources.base import KnowledgeDoc, KnowledgeSource

logger = logging.getLogger(__name__)

_TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELED", "CLOSED"}


class DatabricksKnowledgeSource(KnowledgeSource):
    def __init__(
        self,
        source_name: str,
        table: str,
        title_col: str = "title",
        content_col: str = "content",
        type_col: Optional[str] = None,
        where_clause: str = "",
        extra_cols: Optional[List[str]] = None,
        auth=None,
        warehouse_id: str = "",
        workspace_url: str = "",
        api_key_env: str = "DATABRICKS_TOKEN",
        max_results: int = 500,
        poll_interval_seconds: int = 3,
        max_poll_attempts: int = 20,
    ):
        self._source_name = source_name
        self._table = table
        self._title_col = title_col
        self._content_col = content_col
        self._type_col = type_col
        self._where_clause = where_clause
        self._extra_cols = extra_cols or []
        self.auth = auth
        resolved_url = workspace_url or (auth.host if auth is not None else "")
        self._workspace_url = resolved_url.rstrip("/") if resolved_url else ""
        self._warehouse_id = warehouse_id
        self._static_token = os.getenv(api_key_env) if api_key_env else None
        self._max_results = max_results
        self._poll_interval = poll_interval_seconds
        self._max_poll_attempts = max_poll_attempts

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def enabled(self) -> bool:
        # Needs a table, a warehouse, and a reachable workspace URL to query.
        return bool(self._table and self._warehouse_id and self._workspace_url)

    def _bearer_token(self) -> Optional[str]:
        if self.auth is not None:
            return self.auth.token()
        return self._static_token

    async def fetch(self) -> List[KnowledgeDoc]:
        sql = f"SELECT * FROM {self._table}"
        if self._where_clause:
            sql += f" WHERE {self._where_clause}"
        rows = await self._execute_sql(sql)
        docs = [self._row_to_doc(r) for r in rows if self._content_col in r]
        logger.info(
            "DatabricksKnowledgeSource '%s' read %d rows -> %d docs from %s.",
            self._source_name,
            len(rows),
            len(docs),
            self._table,
        )
        return docs

    def _row_to_doc(self, row: Dict[str, Any]) -> KnowledgeDoc:
        doc_type = "databricks_row"
        if self._type_col and row.get(self._type_col):
            doc_type = str(row.get(self._type_col))
        metadata = {col: row.get(col) for col in self._extra_cols if col in row}
        metadata["table"] = self._table
        return {
            "title": str(row.get(self._title_col, "")),
            "content": str(row.get(self._content_col, "")),
            "type": doc_type,
            "metadata": metadata,
        }

    async def _execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """Run a read-only SQL statement and return rows as dicts (poll to terminal)."""
        async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self._bearer_token()}"}
        ) as session:
            payload = {
                "warehouse_id": self._warehouse_id,
                "statement": sql,
                "wait_timeout": "30s",
                "row_limit": self._max_results,
            }
            logger.info("DatabricksKnowledgeSource submitting SQL: %s", sql)
            async with session.post(
                f"{self._workspace_url}/api/2.0/sql/statements", json=payload
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            statement_id = data["statement_id"]
            state = data["status"]["state"]

            attempts = 0
            while state not in _TERMINAL_STATES and attempts < self._max_poll_attempts:
                await asyncio.sleep(self._poll_interval)
                attempts += 1
                async with session.get(
                    f"{self._workspace_url}/api/2.0/sql/statements/{statement_id}"
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                state = data["status"]["state"]

            if state != "SUCCEEDED":
                message = data.get("status", {}).get("error", {}).get("message", state)
                raise RuntimeError(
                    f"Databricks statement {statement_id} did not succeed: {message}"
                )

            return _rows_from_statement(data)


def _rows_from_statement(data: Dict) -> List[Dict]:
    """Convert a SQL Statement Execution result into a list of row dicts."""
    manifest = data.get("manifest", {})
    columns = [c["name"] for c in manifest.get("schema", {}).get("columns", [])]
    result = data.get("result", {}) or {}
    data_array = result.get("data_array", []) or []
    return [dict(zip(columns, row)) for row in data_array]
