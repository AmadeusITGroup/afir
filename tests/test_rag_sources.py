"""Tests for the pluggable KnowledgeSource implementations (no network/LLM)."""

from unittest.mock import AsyncMock

import pytest

from src.rag.sources.databricks_source import DatabricksKnowledgeSource
from src.rag.sources.document_source import DocumentSource
from src.rag.sources.pack_schema_source import PackSchemaSource
from src.rag.sources.playbook_source import PlaybookSource

_DOC_KEYS = {"title", "content", "type", "metadata"}


class _FakePack:
    def __init__(self, docs):
        self.playbook_documents = docs


@pytest.mark.asyncio
async def test_playbook_source_returns_homogeneous_docs():
    docs = [{"title": "t", "content": "c", "type": "playbook", "metadata": {}}]
    src = PlaybookSource(_FakePack(docs))
    assert src.source_name == "playbook"
    assert src.enabled is True
    fetched = await src.fetch()
    assert _DOC_KEYS <= set(fetched[0])


@pytest.mark.asyncio
async def test_playbook_source_disabled_when_empty():
    src = PlaybookSource(_FakePack([]))
    assert src.enabled is False
    assert await src.fetch() == []


@pytest.mark.asyncio
async def test_playbook_source_includes_concepts_and_cases():
    pack = _FakePack(
        [{"title": "pb", "content": "c", "type": "playbook", "metadata": {}}]
    )
    pack.concept_documents = [
        {
            "title": "Bare",
            "content": "c",
            "type": "concept",
            "metadata": {"use_case": "scheme", "concept_id": "bare_record"},
        }
    ]
    pack.case_documents = [
        {
            "title": "case_a1b2c3",
            "content": "c",
            "type": "case",
            "metadata": {"use_case": "scheme", "case_id": "case_a1b2c3"},
        }
    ]
    src = PlaybookSource(pack)
    fetched = await src.fetch()
    types = {d["type"] for d in fetched}
    assert types == {"playbook", "concept", "case"}


@pytest.mark.asyncio
async def test_playbook_source_enabled_with_only_concepts():
    # A pack with no playbooks but concept docs is still enabled (concepts are embedded).
    pack = _FakePack([])
    pack.concept_documents = [
        {"title": "c", "content": "c", "type": "concept", "metadata": {}}
    ]
    assert PlaybookSource(pack).enabled is True


@pytest.mark.asyncio
async def test_document_source_disabled_without_paths():
    assert DocumentSource([]).enabled is False
    assert DocumentSource(["/x"]).enabled is True


def test_databricks_source_enabled_requires_table_warehouse_url():
    disabled = DatabricksKnowledgeSource(
        source_name="kb", table="", warehouse_id="", workspace_url=""
    )
    assert disabled.enabled is False
    enabled = DatabricksKnowledgeSource(
        source_name="kb",
        table="main.f.kb",
        warehouse_id="wh",
        workspace_url="https://w",
    )
    assert enabled.enabled is True


def test_databricks_source_row_to_doc_mapping():
    src = DatabricksKnowledgeSource(
        source_name="kb",
        table="main.f.kb",
        title_col="name",
        content_col="body",
        type_col="kind",
        extra_cols=["id"],
        warehouse_id="wh",
        workspace_url="https://w",
    )
    doc = src._row_to_doc({"name": "T", "body": "B", "kind": "proc", "id": 7})
    assert doc["title"] == "T"
    assert doc["content"] == "B"
    assert doc["type"] == "proc"
    assert doc["metadata"]["id"] == 7
    assert doc["metadata"]["table"] == "main.f.kb"


@pytest.mark.asyncio
async def test_databricks_source_fetch_maps_rows(monkeypatch):
    src = DatabricksKnowledgeSource(
        source_name="kb",
        table="main.f.kb",
        title_col="title",
        content_col="content",
        warehouse_id="wh",
        workspace_url="https://w",
    )
    src._execute_sql = AsyncMock(
        return_value=[
            {"title": "A", "content": "body-a"},
            {"title": "B"},  # missing content -> dropped
        ]
    )
    docs = await src.fetch()
    assert len(docs) == 1
    assert docs[0]["content"] == "body-a"


# --- PackSchemaSource: the pack's per-table field inventories -----------------------


class _SchemaPack:
    def __init__(self, docs):
        self.schema_documents = docs


@pytest.mark.asyncio
async def test_pack_schema_source_returns_homogeneous_docs():
    docs = [
        {
            "title": "Field schema: s / t",
            "content": "Fields:\n  - a : string",
            "type": "schema",
            "metadata": {"schema_source": "s", "table": "t"},
        }
    ]
    src = PackSchemaSource(_SchemaPack(docs))
    assert src.source_name == "pack_schema"
    assert src.enabled is True
    fetched = await src.fetch()
    assert _DOC_KEYS <= set(fetched[0])
    # `source`/`added_at`/`id` are stamped by KnowledgeBaseManager — a source must not
    # set them itself (see src/rag/sources/base.py).
    assert not {"source", "added_at", "id"} & set(fetched[0])


@pytest.mark.asyncio
async def test_pack_schema_source_disabled_without_schemas():
    src = PackSchemaSource(_SchemaPack([]))
    assert src.enabled is False
    assert await src.fetch() == []


@pytest.mark.asyncio
async def test_pack_schema_source_tolerates_pack_without_the_attribute():
    """A pack predating schemas/ (or an empty pack) must not break RAG construction."""

    class _Old:
        pass

    src = PackSchemaSource(_Old())
    assert src.enabled is False
    assert await src.fetch() == []
