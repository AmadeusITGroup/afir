"""Tests for the deterministic PlaybookFallback retriever."""

import pytest

from src.rag.fallback import PlaybookFallback
from src.rag.protocol import KnowledgeRetriever

_DOCS = [
    {
        "title": "Account takeover",
        "content": "detect suspicious login and credential stuffing",
        "type": "playbook",
        "source": "playbook",
    },
    {
        "title": "Refund fraud",
        "content": "voucher abuse and refund chargeback patterns",
        "type": "playbook",
        "source": "playbook",
    },
]


def test_fallback_satisfies_protocol():
    assert isinstance(PlaybookFallback(_DOCS), KnowledgeRetriever)


@pytest.mark.asyncio
async def test_fallback_ranks_by_keyword_overlap():
    fb = PlaybookFallback(_DOCS)
    results = await fb.retrieve("suspicious login credential stuffing")
    assert results
    # The account-takeover doc should rank first for these tokens.
    assert results[0]["title"] == "Account takeover"
    assert results[0]["relevance_score"] > 0


@pytest.mark.asyncio
async def test_fallback_empty_query_returns_nothing():
    fb = PlaybookFallback(_DOCS)
    assert await fb.retrieve("") == []


def test_fallback_format_context_shape():
    fb = PlaybookFallback(_DOCS)
    assert fb.format_context([]) == ""
    ctx = fb.format_context([dict(_DOCS[0], relevance_score=0.5)])
    assert "Retrieved relevant information" in ctx
    assert "Account takeover" in ctx
