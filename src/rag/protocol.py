"""
The retrieval seam shared by every knowledge backend.

Pipeline modules (``IncidentUnderstandingModule``, ``CorrelationModule``,
``AnomalyDetectionModule``) and ``LLMClient._augment`` only ever call two
methods on the ``rag`` object: ``retrieve`` and ``format_context``. Declaring
that surface as a ``Protocol`` lets the embedding-backed ``EnhancedRAG`` and the
no-embedding ``PlaybookFallback`` be used interchangeably, with no module changes.

Use ``isinstance(x, KnowledgeRetriever)`` (the protocol is ``runtime_checkable``)
rather than ``isinstance(x, EnhancedRAG)`` in tests — the dual-import convention
(``src.rag.*`` vs flat) means the concrete class can have two module identities,
which breaks concrete ``isinstance`` checks but not the protocol check.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class KnowledgeRetriever(Protocol):
    """The minimal retrieval surface the pipeline depends on."""

    async def retrieve(
        self,
        query: str,
        filter_source: Optional[str] = None,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return documents relevant to ``query`` (most relevant first)."""
        ...

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Render retrieved documents into a context string for the LLM."""
        ...
