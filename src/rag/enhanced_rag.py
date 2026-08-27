import logging
import os
import pickle
import ssl
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from src.rag.embeddings import SentenceTransformerProvider

# FlatIP not faiss: two OpenMP runtimes abort the process; see flat_index.py.
from src.rag.flat_index import FlatIP

ssl._create_default_https_context = ssl._create_unverified_context

from src.rag.knowledge_base_manager import KnowledgeBaseManager

logger = logging.getLogger(__name__)

# A pickle import triggers on class name alone, so the index reader refuses these.
_NATIVE_OMP_MODULES = ("faiss", "torch", "sklearn", "scipy")


class _NoNativeUnpickler(pickle.Unpickler):
    """Refuses classes from OpenMP-bundling modules; an unreadable cache triggers a rebuild.

    ``find_class`` is the only interception point — by ``pickle.load``'s return the module
    is already imported. Not a security boundary.
    """

    def find_class(self, module: str, name: str):
        root = module.split(".", 1)[0]
        if root in _NATIVE_OMP_MODULES:
            raise pickle.UnpicklingError(
                f"refusing to unpickle {module}.{name}: importing {root} would load a "
                "second OpenMP runtime and crash the process. The index will be rebuilt."
            )
        return super().find_class(module, name)


class EnhancedRAG:
    """Enhanced RAG system with better retrieval and ranking"""

    # Returned only when filter_type names them; schema docs are too large to crowd a general query.
    OPT_IN_TYPES = frozenset({"schema"})

    def __init__(
        self,
        knowledge_base_manager: KnowledgeBaseManager,
        model_name: str = "all-mpnet-base-v2",
        embedding_dim: int = 768,
        max_retrieved_documents: int = 10,
        similarity_threshold: float = 0.5,
        use_reranking: bool = True,
        embedding_provider=None,
    ):
        self.kb_manager = knowledge_base_manager
        # Provider loads lazily so a missing model surfaces at encode time, where the orchestrator can fall back.
        self._model_name = model_name
        self.embeddings = embedding_provider or SentenceTransformerProvider(
            model_name, expected_dim=embedding_dim
        )
        self.embedding_dim = embedding_dim
        self.max_retrieved_documents = max_retrieved_documents
        self.similarity_threshold = similarity_threshold
        self.use_reranking = use_reranking

        # Initialize or load index (disk only — safe without the model).
        self.index = None
        self.doc_embeddings = None
        self._index_signature = None
        self._initialize_index()

    @property
    def model(self):
        """The local sentence-transformers model; raises if the provider is not sentence-transformers."""
        st = getattr(self.embeddings, "st_model", None)
        if st is None:
            raise AttributeError(
                f"{type(self.embeddings).__name__} has no local model; use "
                "EnhancedRAG.embeddings to encode."
            )
        return st

    def _initialize_index(self):
        """Load the cached index if there is one, else start empty."""
        if os.path.exists(self.kb_manager.index_path):
            self._load_index()
        else:
            self.index = FlatIP(
                self.embedding_dim
            )  # inner product == cosine on unit norms
            self.doc_embeddings = []

    def _load_index(self):
        """Load the index from disk, discarding anything unreadable (triggers a rebuild).

        Legacy faiss indexes survive as a rebuild: ``_NoNativeUnpickler`` refuses the class
        and the except branch handles it.
        """
        try:
            with open(self.kb_manager.index_path, "rb") as f:
                data = _NoNativeUnpickler(f).load()
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            # None means pre-provider-seam index; treated as unknown → rebuild.
            self._index_signature = data.get("signature")
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a bad cache must not fail construction
            logger.warning(
                "Could not read the embedding index at %s (%s); starting empty and "
                "rebuilding on the next update_index().",
                self.kb_manager.index_path,
                exc,
            )
            self.index = FlatIP(self.embedding_dim)
            self.doc_embeddings = []
            self._index_signature = None
            return

        if embeddings.ndim != 2 or embeddings.size == 0:
            self.index = FlatIP(self.embedding_dim)
            self.doc_embeddings = []
            return
        # Width from stored vectors, not config: they diverge when a provider is swapped.
        self.embedding_dim = int(embeddings.shape[1])
        self.index = FlatIP(self.embedding_dim)
        self.index.add(embeddings)
        self.doc_embeddings = embeddings

    def save_index(self):
        """Pickle numpy arrays only — no index object — so reading back never imports a native runtime.

        The ``index`` key is written as ``None`` for backward compat: an old reader falls
        through to a rebuild rather than raising a KeyError.
        """
        with open(self.kb_manager.index_path, "wb") as f:
            pickle.dump(
                {
                    "index": None,
                    "embeddings": np.asarray(self.doc_embeddings, dtype=np.float32),
                    "signature": self.embeddings.signature,
                },
                f,
            )

    def _index_is_stale(self) -> bool:
        """Whether the loaded index came from a different model.

        A ``None`` signature (pre-check index) counts as stale. Width excluded from the
        signature (unknown before first call); width drift caught in ``retrieve``.
        """
        if self.doc_embeddings is None or len(self.doc_embeddings) == 0:
            return False
        current = self.embeddings.signature
        if self._index_signature == current:
            return False
        logger.info(
            "Embedding index was built by %s but the configured provider is %s; "
            "rebuilding. (An index from a different model loads without error and then "
            "answers every query in the wrong vector space.)",
            self._index_signature or "an unidentified model",
            current,
        )
        return True

    async def update_index(self, force_rebuild: bool = False):
        """Update the index with new documents"""
        documents = self.kb_manager.documents

        if (
            force_rebuild
            or len(documents) != len(self.doc_embeddings)
            or self._index_is_stale()
        ):
            logger.info("Rebuilding document index...")

            contents = []
            for doc in documents:
                # Combine title and content for better retrieval
                text = f"{doc.get('title', '')} {doc.get('content', '')}"
                contents.append(text)

            # Provider normalises; calling it again is harmless, assuming it was done is not.
            embeddings = await self.embeddings.encode(contents)

            # Width from provider, not config: they diverge when a provider is swapped.
            width = embeddings.shape[1] if embeddings.size else self.embedding_dim
            self.embedding_dim = int(width)
            self.index = FlatIP(self.embedding_dim)
            self.index.add(embeddings.astype("float32"))
            self.doc_embeddings = embeddings
            self._index_signature = self.embeddings.signature

            self.save_index()

            logger.info(
                "Index rebuilt with %d documents (%s, dim %d)",
                len(documents),
                self.embeddings.signature,
                self.embedding_dim,
            )

    async def retrieve(
        self,
        query: str,
        filter_source: Optional[str] = None,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.

        ``filter_type`` is also what makes an ``OPT_IN_TYPES`` document reachable at all —
        without it those types are skipped entirely (see the class attribute).
        """

        query_embedding = await self.embeddings.encode([query])

        # Width mismatch means the model changed behind the same endpoint name; rebuild rather than raise.
        if self.index is not None and query_embedding.shape[1] != self.index.d:
            logger.warning(
                "Index is %d-dimensional but %s now returns %d dimensions under the same "
                "name; rebuilding before answering this query.",
                self.index.d,
                self.embeddings.signature,
                query_embedding.shape[1],
            )
            await self.update_index(force_rebuild=True)

        # Widen pool when filtering: post-filter would lose results at a fixed 2x.
        pool = self.max_retrieved_documents * 2
        if filter_source or filter_type or self.OPT_IN_TYPES:
            pool = max(pool, self.max_retrieved_documents * 10)
        distances, indices = self.index.search(
            query_embedding.astype("float32"),
            min(pool, len(self.kb_manager.documents)),
        )

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.kb_manager.documents):
                continue

            doc = self.kb_manager.documents[idx].copy()
            doc["relevance_score"] = float(distance)

            if filter_source and doc.get("source") != filter_source:
                continue
            if filter_type and doc.get("type") != filter_type:
                continue
            # OPT_IN_TYPES: only reachable via an explicit filter_type.
            if not filter_type and doc.get("type") in self.OPT_IN_TYPES:
                continue

            if doc["relevance_score"] >= self.similarity_threshold:
                results.append(doc)

        # Rerank if enabled
        if self.use_reranking and results:
            results = await self._rerank_results(query, results)

        return results[: self.max_retrieved_documents]

    async def _rerank_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder or more sophisticated scoring"""

        # Simple reranking based on multiple factors
        for doc in results:
            score = doc["relevance_score"]

            if "last_updated" in doc.get("metadata", {}):
                try:
                    last_updated = datetime.fromisoformat(
                        doc["metadata"]["last_updated"]
                    )
                    days_old = (datetime.now() - last_updated).days
                    recency_boost = max(
                        0, 1 - (days_old / 365)
                    )  # Linear decay over a year
                    score += recency_boost * 0.1
                except:
                    pass

            source_weights = {
                "confluence": 1.2,
                "investigation_report": 1.3,
                "security_bulletin": 1.1,
            }
            source = doc.get("source", "")
            score *= source_weights.get(source, 1.0)

            doc["final_score"] = score

        results.sort(
            key=lambda x: x.get("final_score", x["relevance_score"]), reverse=True
        )

        return results

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string"""
        if not retrieved_docs:
            return ""

        context_parts = ["Retrieved relevant information:\n"]

        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"\n[Document {i}]")
            context_parts.append(f"Title: {doc.get('title', 'Untitled')}")
            context_parts.append(f"Source: {doc.get('source', 'Unknown')}")
            context_parts.append(f"Type: {doc.get('type', 'Unknown')}")
            context_parts.append(f"Relevance: {doc.get('relevance_score', 0):.2f}")

            if "url" in doc.get("metadata", {}):
                context_parts.append(f"URL: {doc['metadata']['url']}")

            context_parts.append(f"\nContent:\n{doc.get('content', '')[:1000]}...")
            context_parts.append("-" * 50)

        return "\n".join(context_parts)
