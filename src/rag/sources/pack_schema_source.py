"""Knowledge source backed by the domain knowledge pack's field-schema inventories."""

import logging
from typing import List

from src.rag.sources.base import KnowledgeDoc, KnowledgeSource

logger = logging.getLogger(__name__)


class PackSchemaSource(KnowledgeSource):
    """Yields one document per (source, table) field inventory from ``schemas/*.yaml``.

    These are RAG documents rather than prompt text. Retriever schema discovery is built for
    generating SQL, so it stops at an ``ARRAY<STRUCT<...>>`` and caps depth and leaf count:
    right for query generation, wrong as a statement about the data, since a field that exists
    is recorded as absent and a "confirmed absent" note then stops anyone looking again. The
    pack's ``schemas/`` dir holds the complete inventory instead, arrays descended.

    The corpus is complete and retrieval is selective. Handing a prompt the whole inventory
    would crowd out the pack's targeting guidance, so nothing here is concatenated into one; it
    is pulled a table at a time via ``retrieve(..., filter_type="schema")``. That is also why
    the unit is one table: a per-source document for thousands of leaves is not selectively
    retrievable at all.

    A zero-I/O adapter: ``pack.schema_documents`` is already in the homogeneous schema.
    """

    def __init__(self, knowledge_pack):
        self._pack = knowledge_pack

    @property
    def source_name(self) -> str:
        return "pack_schema"

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._pack, "schema_documents", None))

    async def fetch(self) -> List[KnowledgeDoc]:
        return list(getattr(self._pack, "schema_documents", []) or [])
