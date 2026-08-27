"""
Tests for the config-driven TLS toggle used by the Databricks + ES retrievers.

Covers the module-level _build_ssl_context helper (Databricks) across its three
modes, and that the ES retriever threads the same keys into its client kwargs.
"""

import ssl
from unittest.mock import MagicMock, patch

from src.retrievers.databricks_retriever import _build_ssl_context
from src.retrievers.elasticsearch_retriever import ElasticsearchRetriever


def test_verify_ssl_false_disables_verification():
    assert _build_ssl_context({"verify_ssl": False}) is False


def test_ca_bundle_builds_context():
    # Patch create_default_context so we don't need a real CA file on disk; assert
    # the helper builds a verifying context from the configured bundle.
    sentinel = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with patch(
        "src.retrievers.databricks_retriever.ssl.create_default_context",
        return_value=sentinel,
    ) as mock_ctx:
        result = _build_ssl_context({"ca_bundle": "/path/ca.pem"})
    mock_ctx.assert_called_once_with(cafile="/path/ca.pem")
    assert result is sentinel


def test_default_is_none():
    assert _build_ssl_context({}) is None
    # verify_ssl:true is not "false", so also default behavior.
    assert _build_ssl_context({"verify_ssl": True}) is None


def test_elasticsearch_honors_verify_ssl_false():
    config = {
        "name": "src",
        "url": "https://es:9200",
        "index": "i",
        "verify_ssl": False,
        "field_schema": "x",
    }
    retriever = ElasticsearchRetriever(config, MagicMock())
    with patch("src.retrievers.elasticsearch_retriever.AsyncElasticsearch") as mock_es:
        retriever._get_client()
    _, kwargs = mock_es.call_args
    assert kwargs["verify_certs"] is False
    assert kwargs["ssl_show_warn"] is False


def test_elasticsearch_honors_ca_bundle():
    config = {
        "name": "src",
        "url": "https://es:9200",
        "index": "i",
        "ca_bundle": "/path/ca.pem",
        "field_schema": "x",
    }
    retriever = ElasticsearchRetriever(config, MagicMock())
    with patch("src.retrievers.elasticsearch_retriever.AsyncElasticsearch") as mock_es:
        retriever._get_client()
    _, kwargs = mock_es.call_args
    assert kwargs["ca_certs"] == "/path/ca.pem"


def test_elasticsearch_default_no_ssl_kwargs():
    config = {
        "name": "src",
        "url": "https://es:9200",
        "index": "i",
        "field_schema": "x",
    }
    retriever = ElasticsearchRetriever(config, MagicMock())
    with patch("src.retrievers.elasticsearch_retriever.AsyncElasticsearch") as mock_es:
        retriever._get_client()
    _, kwargs = mock_es.call_args
    assert "verify_certs" not in kwargs
    assert "ca_certs" not in kwargs
