"""Knowledge consolidation and retriever construction.

``KnowledgeOrchestrator`` turns a list of ``KnowledgeSource``s into one ready retriever:

  1. Consolidate. Every enabled source is fetched into one ``KnowledgeBaseManager``, so the
     LLM sees a single homogeneous knowledge base regardless of origin. Ingest is idempotent
     per source (remove-then-add by ``source_name``), and a source that fails to fetch is
     logged and skipped while the others consolidate.
  2. Build ``EnhancedRAG`` and force an index rebuild, which is where the embedding model
     loads. Any failure here falls back to a deterministic ``PlaybookFallback`` over the
     ingested playbook docs, so the pipeline never loses playbook context.

``build()`` returns ``EnhancedRAG``, ``PlaybookFallback`` (embeddings unavailable but
playbooks present) or ``None``. All three are valid for every pipeline consumer.
"""

import logging
from typing import List, Optional

from src.rag.enhanced_rag import EnhancedRAG
from src.rag.fallback import PlaybookFallback
from src.rag.knowledge_base_manager import KnowledgeBaseManager
from src.rag.protocol import KnowledgeRetriever
from src.rag.sources.base import KnowledgeSource

logger = logging.getLogger(__name__)

_PLAYBOOK_SOURCE = "playbook"


class KnowledgeOrchestrator:
    def __init__(
        self,
        sources: List[KnowledgeSource],
        knowledge_base_path: str,
        model_name: str = "all-mpnet-base-v2",
        embedding_dim: int = 768,
        max_retrieved_documents: int = 10,
        similarity_threshold: float = 0.5,
        use_reranking: bool = True,
        embedding_provider=None,
    ):
        self._sources = sources or []
        self._kb_manager = KnowledgeBaseManager(knowledge_base_path)
        self._model_name = model_name
        self._embedding_dim = embedding_dim
        self._max_docs = max_retrieved_documents
        self._threshold = similarity_threshold
        self._use_reranking = use_reranking
        # None means "build one from model_name", which is what every caller that predates
        # the provider seam does — including every test.
        self._embedding_provider = embedding_provider

    @property
    def kb_manager(self) -> KnowledgeBaseManager:
        return self._kb_manager

    async def build(self) -> Optional[KnowledgeRetriever]:
        """Consolidate all sources, then return a ready retriever (or fallback/None)."""
        await self._ingest_all_sources()
        return await self._build_retriever()

    async def _ingest_all_sources(self) -> None:
        """Idempotent per-source ingest into the single knowledge base."""
        changed = False
        for source in self._sources:
            if not source.enabled:
                logger.info(
                    "Knowledge source '%s' disabled; skipping.", source.source_name
                )
                continue
            try:
                docs = await source.fetch()
            except Exception as e:  # one bad source must not sink the others
                logger.warning(
                    "Knowledge source '%s' failed to fetch; skipping: %s",
                    source.source_name,
                    e,
                )
                continue
            # Remove-then-add keeps re-ingestion idempotent across restarts.
            self._kb_manager.remove_documents_by_source(source.source_name)
            if docs:
                self._kb_manager.add_documents(docs, source.source_name)
                logger.info(
                    "Ingested %d docs from knowledge source '%s'.",
                    len(docs),
                    source.source_name,
                )
            else:
                logger.info(
                    "Knowledge source '%s' returned 0 documents.", source.source_name
                )
            changed = True
        if changed:
            self._kb_manager.save()

    async def _build_retriever(self) -> Optional[KnowledgeRetriever]:
        """Build EnhancedRAG; fall back to PlaybookFallback on any embedding failure."""
        if not self._kb_manager.documents:
            logger.warning("Knowledge base is empty; no retriever built.")
            return None
        try:
            rag = EnhancedRAG(
                self._kb_manager,
                model_name=self._model_name,
                embedding_dim=self._embedding_dim,
                max_retrieved_documents=self._max_docs,
                similarity_threshold=self._threshold,
                use_reranking=self._use_reranking,
                embedding_provider=self._embedding_provider,
            )
            # First encode happens here — this is where a missing model would raise.
            await rag.update_index(force_rebuild=True)
            logger.info(
                "EnhancedRAG ready with %d documents.", len(self._kb_manager.documents)
            )
            return rag
        except Exception as e:
            logger.error(
                "EnhancedRAG init failed (%s); activating deterministic PlaybookFallback.",
                e,
            )
            return self._make_fallback()

    def _make_fallback(self) -> Optional[KnowledgeRetriever]:
        playbook_docs = self._kb_manager.get_documents_by_source(_PLAYBOOK_SOURCE)
        if not playbook_docs:
            logger.warning(
                "PlaybookFallback: no playbook documents in the knowledge base; "
                "pipeline will run without retrieval context."
            )
            return None
        logger.info(
            "PlaybookFallback active with %d playbook documents.", len(playbook_docs)
        )
        return PlaybookFallback(playbook_docs, max_retrieved_documents=self._max_docs)
