"""Tests for building retrievers from the knowledge pack + backends credentials."""

from unittest.mock import MagicMock

from src.knowledge.pack import KnowledgePack, SourceDef
from src.log_retrieval import LogRetrievalEngine


def _pack():
    return KnowledgePack(
        sources=[
            SourceDef(
                name="scheme_alerts",
                endpoints={
                    "kind": "elasticsearch",
                    "cluster": "secondary-prd",
                    "indices": ["idx.a", "idx.b"],
                },
            ),
            SourceDef(
                name="scheme_alerts_delta",
                endpoints={
                    "kind": "databricks_uc",
                    "catalog": "cat",
                    "schema": "gold",
                    "tables": ["t1", "t2"],
                },
            ),
            SourceDef(
                name="ff_activity_snowflake",
                endpoints={"kind": "snowflake", "account": "acct"},
            ),
            SourceDef(
                name="servicenow_ir",
                endpoints={"kind": "rest", "service": "servicenow"},
            ),
        ]
    )


def _backends():
    return {
        "elasticsearch": {
            "secondary-prd": {
                "url": "https://elk:9200",
                "username": "u",
                "password": "p",
            }
        },
        "databricks": {"warehouse_id": "wh-1", "api_key_env": "DATABRICKS_TOKEN"},
    }


def _patch_retrievers(monkeypatch):
    """Replace the retriever classes in the dispatch table with capture stubs.

    The engine resolves classes via _RETRIEVER_TYPES (captured at import), so patch
    the dict, not the module-level names.
    """
    monkeypatch.setitem(
        __import__("src.log_retrieval", fromlist=["_RETRIEVER_TYPES"])._RETRIEVER_TYPES,
        "elasticsearch",
        lambda config, llm, knowledge_pack=None: ("es", config),
    )
    monkeypatch.setitem(
        __import__("src.log_retrieval", fromlist=["_RETRIEVER_TYPES"])._RETRIEVER_TYPES,
        "databricks",
        lambda config, llm, auth=None, knowledge_pack=None: ("dbx", config),
    )


def test_builds_es_and_databricks_skips_unsupported_kinds(monkeypatch):
    # Avoid constructing real clients; capture the config each retriever receives.
    _patch_retrievers(monkeypatch)

    config = {"backends": _backends()}  # no explicit sources
    engine = LogRetrievalEngine(config, MagicMock(), knowledge_pack=_pack())

    # snowflake + rest skipped; es + databricks built.
    assert set(engine.retrievers) == {"scheme_alerts", "scheme_alerts_delta"}

    _, es_config = engine.retrievers["scheme_alerts"]
    assert es_config["index"] == "idx.a,idx.b"  # comma-joined indices
    assert es_config["url"] == "https://elk:9200"

    _, dbx_config = engine.retrievers["scheme_alerts_delta"]
    assert dbx_config["warehouse_id"] == "wh-1"
    assert dbx_config["catalog"] == "cat"
    assert dbx_config["schema"] == "gold"


def test_missing_backend_creds_skips_source(monkeypatch):
    _patch_retrievers(monkeypatch)

    # No backends at all -> both real sources skipped, no crash.
    engine = LogRetrievalEngine({}, MagicMock(), knowledge_pack=_pack())
    assert engine.retrievers == {}


def test_explicit_config_source_takes_precedence(monkeypatch):
    _patch_retrievers(monkeypatch)

    # Explicit source with the same name as a pack source must win.
    config = {
        "backends": _backends(),
        "sources": [
            {
                "name": "scheme_alerts",
                "type": "elasticsearch",
                "url": "https://explicit:9200",
                "username": "x",
                "password": "y",
                "index": "explicit-index",
            }
        ],
    }
    engine = LogRetrievalEngine(config, MagicMock(), knowledge_pack=_pack())

    _, scheme_config = engine.retrievers["scheme_alerts"]
    assert scheme_config["url"] == "https://explicit:9200"
    assert scheme_config["index"] == "explicit-index"
