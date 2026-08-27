"""
Deterministic, no-embedding playbook retriever.

When the embedding model or FAISS path is unavailable (offline, model not cached,
import failure), the pipeline must STILL get playbook context — losing it would
silently degrade every downstream LLM call. ``PlaybookFallback`` provides the same
retrieval surface as ``EnhancedRAG`` (the ``KnowledgeRetriever`` protocol) using
plain keyword/token-overlap matching over the playbook documents. No embeddings,
no FAISS, no network.

Relevance scores here are token-overlap ratios in [0, 1], NOT cosine similarities —
only their ordering is meaningful.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PlaybookFallback:
    """Keyword retriever over playbook docs; satisfies KnowledgeRetriever."""

    def __init__(
        self,
        playbook_docs: List[Dict[str, Any]],
        max_retrieved_documents: int = 5,
    ):
        self._docs = playbook_docs or []
        self._max = max_retrieved_documents

    async def retrieve(
        self,
        query: str,
        filter_source: Optional[str] = None,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        tokens = set(re.findall(r"\w+", (query or "").lower()))
        if not tokens:
            return []
        scored: List[Dict[str, Any]] = []
        for doc in self._docs:
            if filter_source and doc.get("source") != filter_source:
                continue
            if filter_type and doc.get("type") != filter_type:
                continue
            text = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
            hits = sum(1 for t in tokens if t in text)
            if hits > 0:
                entry = dict(doc)
                entry["relevance_score"] = hits / len(tokens)
                scored.append(entry)
        scored.sort(key=lambda d: d["relevance_score"], reverse=True)
        return scored[: self._max]

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Mirror EnhancedRAG.format_context so consumers see identical context."""
        if not retrieved_docs:
            return ""

        context_parts = ["Retrieved relevant information:\n"]
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"\n[Document {i}]")
            context_parts.append(f"Title: {doc.get('title', 'Untitled')}")
            context_parts.append(f"Source: {doc.get('source', 'playbook')}")
            context_parts.append(f"Type: {doc.get('type', 'playbook')}")
            context_parts.append(f"Relevance: {doc.get('relevance_score', 0):.2f}")
            context_parts.append(f"\nContent:\n{doc.get('content', '')[:1000]}...")
            context_parts.append("-" * 50)

        return "\n".join(context_parts)
