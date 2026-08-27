"""Knowledge source backed by Confluence spaces / labels."""

import logging
from typing import Any, Dict, List

from src.rag.confluence_ingester import ConfluenceIngester
from src.rag.sources.base import KnowledgeDoc, KnowledgeSource

logger = logging.getLogger(__name__)


class ConfluenceSource(KnowledgeSource):
    """Ingests pages from configured Confluence spaces and/or labels.

    Wraps the existing ``ConfluenceIngester``. Config dict keys:
        url, username, password   — connection (url gates `enabled`)
        spaces: [space_key, ...]  — ingest every page of each space
        labels: [label, ...]      — ingest pages carrying each label
    """

    def __init__(self, confluence_config: Dict[str, Any]):
        self._config = confluence_config or {}
        self._spaces: List[str] = self._config.get("spaces", []) or []
        self._labels: List[str] = self._config.get("labels", []) or []
        self._ingester = None  # built lazily so a disabled source costs nothing

    @property
    def source_name(self) -> str:
        return "confluence"

    @property
    def enabled(self) -> bool:
        return bool(self._config.get("url"))

    def _get_ingester(self) -> ConfluenceIngester:
        if self._ingester is None:
            self._ingester = ConfluenceIngester(
                self._config["url"],
                self._config.get("username", ""),
                self._config.get("password", ""),
            )
        return self._ingester

    async def fetch(self) -> List[KnowledgeDoc]:
        ingester = self._get_ingester()
        docs: List[KnowledgeDoc] = []
        for space in self._spaces:
            docs.extend(await ingester.ingest_space(space))
        for label in self._labels:
            docs.extend(await ingester.ingest_pages_by_label(label))
        return docs
