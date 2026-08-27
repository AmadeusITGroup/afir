"""Tests for KnowledgeOrchestrator consolidation + fallback (no model/network)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.fallback import PlaybookFallback
from src.rag.orchestrator import KnowledgeOrchestrator
from src.rag.protocol import KnowledgeRetriever
from src.rag.sources.base import KnowledgeSource


class _FakeSource(KnowledgeSource):
    def __init__(self, name, docs, enabled=True, raises=False):
        self._name = name
        self._docs = docs
        self._enabled = enabled
        self._raises = raises

    @property
    def source_name(self):
        return self._name

    @property
    def enabled(self):
        return self._enabled

    async def fetch(self):
        if self._raises:
            raise RuntimeError("boom")
        return self._docs


def _doc(title):
    return {
        "title": title,
        "content": f"{title} body",
        "type": "playbook",
        "metadata": {},
    }


@pytest.mark.asyncio
async def test_consolidates_multiple_sources(tmp_path):
    sources = [
        _FakeSource("playbook", [_doc("ato")]),
        _FakeSource("confluence", [_doc("wiki1"), _doc("wiki2")]),
        _FakeSource("disabled", [_doc("x")], enabled=False),
    ]
    orch = KnowledgeOrchestrator(sources, knowledge_base_path=str(tmp_path))
    await orch._ingest_all_sources()
    assert len(orch.kb_manager.documents) == 3
    assert len(orch.kb_manager.get_documents_by_source("playbook")) == 1
    assert len(orch.kb_manager.get_documents_by_source("confluence")) == 2


@pytest.mark.asyncio
async def test_failing_source_is_skipped_not_fatal(tmp_path):
    sources = [
        _FakeSource("playbook", [_doc("ato")]),
        _FakeSource("bad", [], raises=True),
    ]
    orch = KnowledgeOrchestrator(sources, knowledge_base_path=str(tmp_path))
    await orch._ingest_all_sources()
    assert len(orch.kb_manager.get_documents_by_source("playbook")) == 1


@pytest.mark.asyncio
async def test_ingest_is_idempotent_across_rebuilds(tmp_path):
    src = _FakeSource("playbook", [_doc("ato"), _doc("refund")])
    orch = KnowledgeOrchestrator([src], knowledge_base_path=str(tmp_path))
    await orch._ingest_all_sources()
    await orch._ingest_all_sources()  # second "restart"
    assert len(orch.kb_manager.get_documents_by_source("playbook")) == 2


@pytest.mark.asyncio
async def test_build_falls_back_when_embedding_fails(tmp_path):
    sources = [_FakeSource("playbook", [_doc("ato")])]
    orch = KnowledgeOrchestrator(sources, knowledge_base_path=str(tmp_path))

    # Simulate the embedding model being unavailable: EnhancedRAG() raises.
    with patch(
        "src.rag.orchestrator.EnhancedRAG", side_effect=RuntimeError("no model")
    ):
        rag = await orch.build()

    assert isinstance(rag, PlaybookFallback)
    assert isinstance(rag, KnowledgeRetriever)
    results = await rag.retrieve("ato")
    assert results and results[0]["source"] == "playbook"


@pytest.mark.asyncio
async def test_build_returns_enhanced_rag_on_success(tmp_path):
    sources = [_FakeSource("playbook", [_doc("ato")])]
    orch = KnowledgeOrchestrator(sources, knowledge_base_path=str(tmp_path))

    fake_rag = MagicMock()
    fake_rag.update_index = AsyncMock()
    with patch("src.rag.orchestrator.EnhancedRAG", return_value=fake_rag):
        rag = await orch.build()

    assert rag is fake_rag
    fake_rag.update_index.assert_awaited_once()


@pytest.mark.asyncio
async def test_build_returns_none_when_no_docs(tmp_path):
    orch = KnowledgeOrchestrator(
        [_FakeSource("playbook", [])], knowledge_base_path=str(tmp_path)
    )
    assert await orch.build() is None


@pytest.mark.asyncio
async def test_fallback_none_when_embedding_fails_and_no_playbooks(tmp_path):
    # A non-playbook source exists, embedding fails -> no playbook docs to fall back on.
    sources = [_FakeSource("confluence", [_doc("wiki")])]
    orch = KnowledgeOrchestrator(sources, knowledge_base_path=str(tmp_path))
    with patch(
        "src.rag.orchestrator.EnhancedRAG", side_effect=RuntimeError("no model")
    ):
        rag = await orch.build()
    assert rag is None
