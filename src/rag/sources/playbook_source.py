"""Knowledge source backed by the domain knowledge pack's playbooks."""

import logging
from typing import List

from src.rag.sources.base import KnowledgeDoc, KnowledgeSource

logger = logging.getLogger(__name__)


class PlaybookSource(KnowledgeSource):
    """Yields the knowledge pack's playbook, concept, and past-investigation documents.

    All three are already in the homogeneous schema (``title``/``content``/``type``/
    ``metadata``), so this source is a thin, zero-I/O adapter. Concepts (the pack's notes on
    the notions its procedure names) and cases (resolved precedents) are embedded + retrievable
    alongside the playbooks, so the report/anomaly LLM calls can cite them. It is the
    source the deterministic fallback relies on, so declare it first in the registry.
    """

    def __init__(self, knowledge_pack):
        self._pack = knowledge_pack

    @property
    def source_name(self) -> str:
        return "playbook"

    @property
    def enabled(self) -> bool:
        return bool(
            getattr(self._pack, "playbook_documents", None)
            or getattr(self._pack, "concept_documents", None)
            or getattr(self._pack, "case_documents", None)
        )

    async def fetch(self) -> List[KnowledgeDoc]:
        docs: List[KnowledgeDoc] = []
        docs.extend(getattr(self._pack, "playbook_documents", []) or [])
        docs.extend(getattr(self._pack, "concept_documents", []) or [])
        docs.extend(getattr(self._pack, "case_documents", []) or [])
        return docs
