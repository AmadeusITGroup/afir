import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# This module stores documents and metadata as JSON and does no vector work; `enhanced_rag`
# owns the index. It used to import faiss, numpy, pickle, asyncio and SentenceTransformer at
# module top and use none of them, which is what made torch mandatory in a deployment whose
# default provider is a serving endpoint. If vector work lands here, import it where used.


class KnowledgeBaseManager:
    """Manages the knowledge base for RAG system"""

    def __init__(self, base_path: str = "knowledge_base"):
        self.base_path = base_path
        self.documents_path = os.path.join(base_path, "documents.json")
        self.index_path = os.path.join(base_path, "faiss_index.pkl")
        self.metadata_path = os.path.join(base_path, "metadata.json")

        os.makedirs(base_path, exist_ok=True)

        self.documents = self._load_documents()
        self.metadata = self._load_metadata()

    def _load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from storage"""
        if os.path.exists(self.documents_path):
            with open(self.documents_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata about the knowledge base"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {"last_updated": None, "total_documents": 0, "sources": {}}

    def save(self):
        """Save documents and metadata"""
        with open(self.documents_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)

        self.metadata["last_updated"] = datetime.now().isoformat()
        self.metadata["total_documents"] = len(self.documents)

        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def add_documents(self, documents: List[Dict[str, Any]], source: str):
        """Add new documents to the knowledge base"""
        for doc in documents:
            doc["source"] = source
            doc["added_at"] = datetime.now().isoformat()
            doc["id"] = (
                f"{source}_{len(self.documents)}_{doc.get('title', 'untitled')[:50]}"
            )
            self.documents.append(doc)

        if source not in self.metadata["sources"]:
            self.metadata["sources"][source] = 0
        self.metadata["sources"][source] += len(documents)

        logger.info(f"Added {len(documents)} documents from {source}")

    def get_documents_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific source"""
        return [doc for doc in self.documents if doc.get("source") == source]

    def remove_documents_by_source(self, source: str):
        """Remove all documents from a specific source"""
        original_count = len(self.documents)
        self.documents = [doc for doc in self.documents if doc.get("source") != source]
        removed_count = original_count - len(self.documents)

        if source in self.metadata["sources"]:
            del self.metadata["sources"][source]

        logger.info(f"Removed {removed_count} documents from {source}")
        return removed_count
