"""Knowledge source backed by local document folders (pdf/docx/txt/csv/json)."""

import logging
from typing import List

from src.rag.document_ingester import DocumentIngester
from src.rag.sources.base import KnowledgeDoc, KnowledgeSource

logger = logging.getLogger(__name__)


class DocumentSource(KnowledgeSource):
    """Ingests every supported file under one or more local directories.

    Wraps the existing ``DocumentIngester`` (whose returned dicts already match
    the homogeneous schema). Point this at a Unity Catalog Volume mount path to
    ingest files that live in Databricks — no separate class needed.
    """

    def __init__(
        self, directory_paths: List[str], source_name: str = "local_documents"
    ):
        self._paths = directory_paths or []
        self._source_name = source_name
        self._ingester = DocumentIngester()

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def enabled(self) -> bool:
        return bool(self._paths)

    async def fetch(self) -> List[KnowledgeDoc]:
        docs: List[KnowledgeDoc] = []
        for path in self._paths:
            docs.extend(await self._ingester.ingest_directory(path))
        return docs
