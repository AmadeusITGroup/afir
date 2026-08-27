"""
Test suite for the enhanced RAG system
"""

import os

os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.rag.confluence_ingester import ConfluenceIngester
from src.rag.document_ingester import DocumentIngester
from src.rag.enhanced_rag import EnhancedRAG
from src.rag.knowledge_base_manager import KnowledgeBaseManager


class TestKnowledgeBaseManager:
    """Test the knowledge base manager"""

    def setup_method(self):
        """Create a temporary directory for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.kb_manager = KnowledgeBaseManager(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test KB manager initialization"""
        assert self.kb_manager.documents == []
        assert self.kb_manager.metadata["total_documents"] == 0
        assert Path(self.temp_dir).exists()

    def test_add_documents(self):
        """Test adding documents"""
        test_docs = [
            {"title": "Test Doc 1", "content": "Content 1"},
            {"title": "Test Doc 2", "content": "Content 2"},
        ]

        self.kb_manager.add_documents(test_docs, "test_source")

        assert len(self.kb_manager.documents) == 2
        assert self.kb_manager.metadata["sources"]["test_source"] == 2
        assert all(doc["source"] == "test_source" for doc in self.kb_manager.documents)

    def test_save_and_load(self):
        """Test saving and loading documents"""
        test_doc = {"title": "Test", "content": "Test content"}
        self.kb_manager.add_documents([test_doc], "test")
        self.kb_manager.save()

        new_manager = KnowledgeBaseManager(self.temp_dir)

        assert len(new_manager.documents) == 1
        assert new_manager.documents[0]["title"] == "Test"
        assert new_manager.metadata["sources"]["test"] == 1

    def test_remove_documents_by_source(self):
        """Test removing documents by source"""
        self.kb_manager.add_documents([{"title": "Doc1", "content": "C1"}], "source1")
        self.kb_manager.add_documents([{"title": "Doc2", "content": "C2"}], "source2")

        removed = self.kb_manager.remove_documents_by_source("source1")

        assert removed == 1
        assert len(self.kb_manager.documents) == 1
        assert self.kb_manager.documents[0]["source"] == "source2"

    def test_playbook_ingest_is_idempotent_across_restarts(self):
        """remove+add (main.py's boot pattern) must not duplicate playbook docs.

        add_documents APPENDS, so re-ingesting on every boot without first removing
        the prior set would grow documents.json by N each restart. main() removes the
        'playbook' source before re-adding; simulate two boots and assert the count
        stays at N, not 2N.
        """
        playbooks = [
            {"title": f"pb{i}", "content": "x", "type": "playbook"} for i in range(10)
        ]

        # First "boot".
        self.kb_manager.remove_documents_by_source("playbook")
        self.kb_manager.add_documents(playbooks, "playbook")
        first = self.kb_manager.get_documents_by_source("playbook")

        # Second "boot" — same docs re-ingested.
        self.kb_manager.remove_documents_by_source("playbook")
        self.kb_manager.add_documents(playbooks, "playbook")
        second = self.kb_manager.get_documents_by_source("playbook")

        assert len(first) == 10
        assert len(second) == 10  # not 20
        assert self.kb_manager.metadata["sources"]["playbook"] == 10


class TestDocumentIngester:
    """Test the document ingester"""

    async def test_ingest_text_file(self, tmp_path):
        """Test ingesting a text file"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content")

        ingester = DocumentIngester()
        doc = await ingester.ingest_file(str(test_file))

        assert doc is not None
        assert doc["title"] == "test"
        assert doc["content"] == "This is test content"
        assert doc["type"] == "document_txt"

    async def test_ingest_json_file(self, tmp_path):
        """Test ingesting a JSON file"""
        test_data = {"key": "value", "number": 42}
        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(test_data))

        ingester = DocumentIngester()
        doc = await ingester.ingest_file(str(test_file))

        assert doc is not None
        assert doc["title"] == "test"
        assert json.loads(doc["content"]) == test_data

    async def test_ingest_directory(self, tmp_path):
        """Test ingesting a directory"""
        (tmp_path / "doc1.txt").write_text("Content 1")
        (tmp_path / "doc2.txt").write_text("Content 2")
        (tmp_path / "ignore.exe").write_text("Should be ignored")

        ingester = DocumentIngester()
        docs = await ingester.ingest_directory(str(tmp_path))

        assert len(docs) == 2
        assert all(doc["type"] == "document_txt" for doc in docs)
        assert set(doc["title"] for doc in docs) == {"doc1", "doc2"}


class TestEnhancedRAG:
    """Test the enhanced RAG system"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.kb_manager = KnowledgeBaseManager(self.temp_dir)

        test_docs = [
            {
                "title": "Fraud Pattern Alpha",
                "content": "This pattern involves multiple login attempts from different locations",
                "type": "fraud_pattern",
                "source": "test",
            },
            {
                "title": "Investigation Report Beta",
                "content": "Investigation of account takeover incident with credential stuffing",
                "type": "investigation_report",
                "source": "test",
            },
            {
                "title": "Security Playbook Gamma",
                "content": "Steps to handle suspicious authentication activity",
                "type": "playbook",
                "source": "test",
            },
        ]
        self.kb_manager.add_documents(test_docs, "test")
        self.kb_manager.save()

    def teardown_method(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    @patch("sentence_transformers.SentenceTransformer")
    async def test_retrieval(self, mock_transformer):
        """Test document retrieval"""
        # One row per input, not a fixed 3x768: the provider seam rejects a batch whose row
        # count disagrees with its input count, because 40 vectors for 53 documents would
        # index the wrong documents under the wrong ids. A fixed-shape double used to pass
        # only because FAISS treated the extra rows as extra *queries* and `retrieve` read
        # `indices[0]`.
        mock_model = Mock()
        mock_model.encode.side_effect = lambda texts, **kw: np.random.rand(
            len(texts), 768
        )
        mock_transformer.return_value = mock_model

        rag = EnhancedRAG(
            self.kb_manager, max_retrieved_documents=2, similarity_threshold=0.0
        )

        await rag.update_index()

        results = await rag.retrieve("login attempts fraud")

        assert len(results) > 0
        assert all("relevance_score" in doc for doc in results)
        mock_model.encode.assert_called()

    @patch("sentence_transformers.SentenceTransformer")
    async def test_filtered_retrieval(self, mock_transformer):
        """Test retrieval with filters.

        Deterministic embeddings, and `similarity_threshold=0.0` so every document is a
        candidate: with random vectors above the default 0.5 threshold this test could pass
        by returning *nothing*, which asserts nothing about the filter.
        """

        def _encode(texts, **kw):
            out = np.zeros((len(texts), 768), dtype="float32")
            for i, t in enumerate(texts):
                out[i][hash(str(t)[:40]) % 768] = 1.0
                out[i][0] = 0.5
            return out

        mock_model = Mock()
        mock_model.encode.side_effect = _encode
        mock_transformer.return_value = mock_model

        rag = EnhancedRAG(self.kb_manager, similarity_threshold=0.0)
        await rag.update_index()

        results = await rag.retrieve(
            "investigation", filter_type="investigation_report"
        )

        # Every RESULT is of the filtered type — the original read
        # `documents[0]["type"]`, which is "fraud_pattern", so it could only pass on an
        # empty result set.
        assert results
        assert all(doc["type"] == "investigation_report" for doc in results)

    @patch("sentence_transformers.SentenceTransformer")
    async def test_schema_docs_are_invisible_without_an_explicit_filter(
        self, mock_transformer
    ):
        """A `schema` doc must NEVER surface in a general query.

        This is what makes "complete corpus, selective retrieval" real. The correlation
        stage's playbook match calls `retrieve(query)` with no filter; the pack's field
        inventories are large and numerous, so if they competed for the top-N slots they
        would crowd out the targeting guidance that stage needs. Adding schema docs to the
        corpus must therefore leave every existing caller's result unchanged.
        """
        # Deterministic embeddings, one per input text: random ones desync the FAISS
        # index from the document list and make "did this doc drop out?" unanswerable,
        # which is the only thing this test is asking.
        def _encode(texts, **kw):
            out = np.zeros((len(texts), 768), dtype="float32")
            for i, t in enumerate(texts):
                out[i][hash(str(t)[:40]) % 768] = 1.0
                out[i][0] = 0.5
            return out

        mock_model = Mock()
        mock_model.encode.side_effect = _encode
        mock_transformer.return_value = mock_model

        rag = EnhancedRAG(self.kb_manager, similarity_threshold=-1.0)
        await rag.update_index()
        baseline = {d["title"] for d in await rag.retrieve("authentication fraud")}
        assert len(baseline) == 3  # every seeded doc, so a drop-out is detectable

        self.kb_manager.add_documents(
            [
                {
                    "title": "Field schema: record_lake / record_table_4",
                    "content": "Fields:\n  - enrichment.aux_docs.citizenship_iso3",
                    "type": "schema",
                }
            ],
            "pack_schema",
        )
        await rag.update_index(force_rebuild=True)

        after = await rag.retrieve("authentication fraud")
        assert all(d.get("type") != "schema" for d in after)
        # And the corpus addition must not SUBTRACT from what the caller already got:
        # FAISS ranks over the whole corpus and the filter is applied after, so a fixed
        # candidate pool would silently push previously-returned docs out.
        assert baseline <= {d["title"] for d in after}

        # Asked for by name, it is retrievable.
        scoped = await rag.retrieve("citizenship passport", filter_type="schema")
        assert [d["type"] for d in scoped] == ["schema"]

    def test_format_context(self):
        """Test context formatting"""
        rag = EnhancedRAG(self.kb_manager)

        test_docs = [
            {
                "title": "Test Doc",
                "content": "Test content",
                "source": "test",
                "type": "test_type",
                "relevance_score": 0.95,
            }
        ]

        context = rag.format_context(test_docs)

        assert "Test Doc" in context
        assert "test" in context
        assert "0.95" in context


class TestConfluenceIngester:
    """Test Confluence integration"""

    @patch("src.rag.confluence_ingester.Confluence")
    async def test_ingest_space(self, mock_confluence_class):
        """Test ingesting a Confluence space"""
        mock_confluence = Mock()
        mock_confluence_class.return_value = mock_confluence

        mock_confluence.get_all_pages_from_space.return_value = [
            {
                "id": "123",
                "title": "Test Page",
                "body": {"storage": {"value": "<p>Test content</p>"}},
                "version": {"number": 1, "when": "2024-01-01"},
            }
        ]

        ingester = ConfluenceIngester("url", "user", "pass")
        docs = await ingester.ingest_space("TEST")

        assert len(docs) == 1
        assert docs[0]["title"] == "Test Page"
        assert "Test content" in docs[0]["content"]
        assert docs[0]["type"] == "confluence_page"


@pytest.mark.asyncio
async def test_rag_performance():
    """Test RAG system performance"""
    import time

    with tempfile.TemporaryDirectory() as temp_dir:
        kb_manager = KnowledgeBaseManager(temp_dir)

        docs = []
        for i in range(100):
            docs.append(
                {
                    "title": f"Document {i}",
                    "content": f"Content for document {i} with various keywords",
                    "type": "test",
                    "source": "test",
                }
            )

        kb_manager.add_documents(docs, "test")

        # Test with minimal model for speed
        with patch("sentence_transformers.SentenceTransformer") as mock_transformer:
            mock_model = Mock()
            mock_model.encode.side_effect = lambda texts, **kw: np.random.rand(
                len(texts), 384
            )
            mock_transformer.return_value = mock_model

            rag = EnhancedRAG(kb_manager, embedding_dim=384)

            # Time index building
            start = time.time()
            await rag.update_index()
            index_time = time.time() - start

            # Time retrieval
            start = time.time()
            await rag.retrieve("test query")
            search_time = time.time() - start

            # Performance assertions
            assert index_time < 5.0  # Should build index in under 5 seconds
            assert search_time < 0.5  # Should search in under 500ms

            print(f"Index time: {index_time:.3f}s, Search time: {search_time:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
