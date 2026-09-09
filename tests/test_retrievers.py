"""
Tests for the retriever abstraction. LLM query generation and the backend clients
are mocked, so these run without OpenAI, Elasticsearch, or Databricks access.
"""

import copy
import json
import logging
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.log_retrieval import LogRetrievalEngine, _is_placeholder
from src.models.pydantic_models import (EsqlQuery, EsQueryDsl, ExtractedEntity,
                                        RetrievalQuery, SchemaSelection,
                                        SqlQuery)
from src.retrievers.databricks_retriever import (DatabricksRetriever,
                                                 _columns_from_describe,
                                                 _flatten_struct,
                                                 _live_column_types,
                                                 _normalize_struct_paths,
                                                 _render_schema,
                                                 _rows_from_statement,
                                                 hidden_field_count,
                                                 struct_child_names,
                                                 struct_type_from_children,
                                                 type_is_partial)
from src.retrievers.elasticsearch_retriever import (ElasticsearchRetriever,
                                                    _rows_from_esql)
from src.retrievers.kibana_retriever import (KibanaRetriever, _collect_keys,
                                             _rows_from_hits, _sampled_hits)


def _query(source="application_logs"):
    return RetrievalQuery(
        target_log_source=source,
        natural_language_query="failed logins for user 42",
        scope_id="OFF1",
        actor_id="42",
        date_from="2024-01-01",
        date_to="2024-01-05",
    )


# --- helpers ---------------------------------------------------------------


def test_rows_from_esql_zips_columns():
    response = {"columns": [{"name": "a"}, {"name": "b"}], "values": [[1, 2], [3, 4]]}
    assert _rows_from_esql(response) == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


def test_rows_from_statement_zips_columns():
    data = {
        "manifest": {"schema": {"columns": [{"name": "x"}, {"name": "y"}]}},
        "result": {"data_array": [["1", "2"]]},
    }
    assert _rows_from_statement(data) == [{"x": "1", "y": "2"}]


# --- Elasticsearch ---------------------------------------------------------


@pytest.mark.asyncio
async def test_elasticsearch_retriever_uses_esql_and_basic_auth():
    config = {
        "name": "application_logs",
        "url": "http://es:9200",
        "username": "u",
        "password": "p",
        "index": "my-index",
        "max_results": 100,
        "field_schema": "orgUnitId keyword",  # set -> skips field-caps discovery
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=EsqlQuery(query='FROM my-index | WHERE user.userId == "42"')
    )

    retriever = ElasticsearchRetriever(config, llm)

    fake_client = MagicMock()
    fake_client.esql.query = AsyncMock(
        return_value={"columns": [{"name": "msg"}], "values": [["hit"]]}
    )
    with patch(
        "src.retrievers.elasticsearch_retriever.AsyncElasticsearch",
        return_value=fake_client,
    ) as es_cls:
        rows = await retriever.retrieve(_query())

    # basic_auth (not deprecated http_auth) is passed
    _, kwargs = es_cls.call_args
    assert kwargs["basic_auth"] == ("u", "p")
    # LIMIT injected by the retriever, not the LLM
    executed = fake_client.esql.query.call_args.kwargs["query"]
    assert "LIMIT 100" in executed
    assert rows == [{"msg": "hit"}]


@pytest.mark.asyncio
async def test_elasticsearch_discovers_fields_via_field_caps():
    config = {
        "name": "application_logs",
        "url": "http://es:9200",
        "username": "u",
        "password": "p",
        "index": "my-index",
        # no field_schema -> field-caps discovery runs
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=EsqlQuery(query="FROM my-index"))

    retriever = ElasticsearchRetriever(config, llm)
    fake_client = MagicMock()
    fake_client.field_caps = AsyncMock(
        return_value={
            "fields": {
                "org_unit_id": {"keyword": {}},
                "agent_sign": {"keyword": {}},
                "_id": {"_id": {}},  # metadata field, skipped
            }
        }
    )
    fake_client.esql.query = AsyncMock(
        return_value={"columns": [{"name": "msg"}], "values": [["hit"]]}
    )
    with patch(
        "src.retrievers.elasticsearch_retriever.AsyncElasticsearch",
        return_value=fake_client,
    ):
        await retriever.retrieve(_query())

    schema = await retriever._get_field_schema()
    fields = {part.split(":")[0].strip() for part in schema.split(",")}
    assert "org_unit_id" in fields
    assert "agent_sign" in fields
    assert "_id" not in fields  # metadata fields excluded
    fake_client.field_caps.assert_awaited_once()  # cached: only one discovery call


# --- Kibana gateway --------------------------------------------------------


def test_rows_from_hits_extracts_source():
    raw = {"hits": {"hits": [{"_source": {"a": 1}}, {"_source": {"b": 2}}]}}
    assert _rows_from_hits(raw) == [{"a": 1}, {"b": 2}]


def test_collect_keys_flattens_nested_source():
    out = set()
    _collect_keys({"a": 1, "b": {"c": 2, "d": {"e": 3}}}, "", out)
    assert {"a", "b", "b.c", "b.d", "b.d.e"} <= out


def test_collect_keys_depth_capped():
    """The cap counts dotted SEGMENTS — the unit a field name is measured in."""
    out = set()
    _collect_keys({"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}, "", out)
    assert "a.b.c.d.e.f" in out  # six segments, the bound itself
    assert "a.b.c.d.e.f.g" not in out


def test_collect_keys_descends_arrays_of_objects():
    """An ES array of objects is queried by the same dotted path as a single object.

    The live failure this pins: the path was absent from the discovered schema, so
    `_in_schema` rejected the pack's own measured binding, the retriever logged STALE
    BINDING, the predicate was dropped and the query became a window-only scan that
    returned 0 rows — which reads as a source with nothing to say.
    """
    out = set()
    _collect_keys(
        {"alerts": [{"record_ref": "AAA111", "loginArea": {"sign": "0505VW"}}]}, "", out
    )
    assert "alerts.record_ref" in out
    assert "alerts.loginArea.sign" in out


def test_collect_keys_array_adds_no_segment():
    """An array does not spend a segment of the cap, because the name has none."""
    shallow = {"a": {"b": {"c": {"d": {"e": {"f": "leaf"}}}}}}
    boxed = {"a": [{"b": [{"c": [{"d": [{"e": [{"f": "leaf"}]}]}]}]}]}
    flat, wrapped = set(), set()
    _collect_keys(shallow, "", flat)
    _collect_keys(boxed, "", wrapped)
    assert "a.b.c.d.e.f" in wrapped
    assert flat == wrapped


def test_collect_keys_ignores_scalar_arrays_and_non_dicts():
    """A list of scalars is a leaf: it contributes its own path and nothing under it."""
    out = set()
    _collect_keys({"tickets": ["001", "002"], "n": 3, "obj": None}, "", out)
    assert out == {"tickets", "n", "obj"}


@pytest.mark.asyncio
async def test_kibana_retriever_queries_internal_search_with_dsl():
    """The Kibana retriever discovers fields by sampling, then POSTs Query DSL."""
    config = {
        "name": "siem_alerts",
        "url": "https://kibana.example.net",
        "username": "u",
        "password": "p",
        "index": "raw.prd.siem-alerts*",
        "max_results": 25,
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=EsQueryDsl(
            query={"bool": {"filter": [{"range": {"timestamp": {}}}]}}
        )
    )
    retriever = KibanaRetriever(config, llm)

    # First _search = field-discovery sample; second = the real query.
    sample = _FakeResp(
        {
            "rawResponse": {
                "hits": {"hits": [{"_source": {"alertId": "x", "ir": {"id": "1"}}}]}
            }
        }
    )
    real = _FakeResp(
        {"rawResponse": {"hits": {"hits": [{"_source": {"alertId": "a1"}}]}}}
    )
    session = MagicMock()
    session.closed = False
    session.post = MagicMock(side_effect=[sample, real])
    retriever._session = session

    rows = await retriever.retrieve(_query("siem_alerts"))

    assert rows == [{"alertId": "a1"}]
    # Two POSTs: discovery sample + real query, both to /internal/search/es.
    assert session.post.call_count == 2
    for call in session.post.call_args_list:
        url = call.args[0]
        assert url.endswith("/internal/search/es")
    # The real query wraps the LLM's DSL under params.body.query with a size cap.
    real_body = session.post.call_args_list[1].kwargs["json"]
    assert real_body["params"]["index"] == "raw.prd.siem-alerts*"
    assert real_body["params"]["body"]["size"] == 25
    assert "bool" in real_body["params"]["body"]["query"]
    # Discovered fields (dotted) reached the field-mapping / DSL prompt.
    schema = await retriever._get_field_schema()
    assert "ir.id" in schema


@pytest.mark.asyncio
async def test_field_discovery_sees_every_index_shape_not_just_the_first():
    """An index PATTERN can hold several document shapes, and a flat sample misses them.

    Reproduces the live defect: a source spanning two patterns and 293K documents had all
    50 sampled docs come back from the single oldest monthly index, so the schema named a
    pre-migration column and none of the two current shapes'. The generated query then
    filtered a field no document in the window has, and the source reported 0 rows.
    """
    config = {
        "name": "svc_sessions",
        "url": "https://kibana.example.net",
        "username": "u",
        "password": "p",
        "index": "sessions*,raw*",
    }
    retriever = KibanaRetriever(config, MagicMock())

    # Three shapes with NO column in common, exactly as measured on the live source: the
    # sampler must reach all three, and a flat `hits` reading would only see whichever
    # index the shards happened to answer with.
    sample = _FakeResp(
        {
            "rawResponse": {
                "hits": {"hits": []},
                "aggregations": {
                    "idx": {
                        "buckets": [
                            {
                                "key": "sessions.2026.08",
                                "docs": {
                                    "hits": {
                                        "hits": [
                                            {
                                                "_source": {
                                                    "sign": "S",
                                                    "user": {"userId": "L"},
                                                }
                                            }
                                        ]
                                    }
                                },
                            },
                            {
                                "key": "raw.2026.08",
                                "docs": {
                                    "hits": {
                                        "hits": [
                                            {
                                                "_source": {
                                                    "payload": {"actor": {"id": "L"}}
                                                }
                                            }
                                        ]
                                    }
                                },
                            },
                            {
                                "key": "raw.2025.07",
                                "docs": {
                                    "hits": {
                                        "hits": [
                                            {"_source": {"payload": {"AdminSign": "S"}}}
                                        ]
                                    }
                                },
                            },
                        ]
                    }
                },
            }
        }
    )
    session = MagicMock()
    session.closed = False
    session.post = MagicMock(return_value=sample)
    retriever._session = session

    schema = await retriever._get_field_schema()

    # Every shape's identity column, not just the one an arbitrary shard answered with.
    assert "sign" in schema
    assert "user.userId" in schema
    assert "payload.actor.id" in schema
    assert "payload.AdminSign" in schema
    # And the sample is bucketed PER INDEX — a flat `size: N` request is what missed two
    # shapes, so asking for one is the defect, not an implementation detail.
    body = session.post.call_args_list[0].kwargs["json"]["params"]["body"]
    assert body["aggs"]["idx"]["terms"]["field"] == "_index"
    assert body["size"] == 0, "the docs come from the buckets, not the top-level hits"
    # Newest-first, so a truncated bucket list keeps the CURRENT shape rather than the
    # oldest one — which is precisely what the flat sample returned.
    assert body["aggs"]["idx"]["terms"]["order"] == {"_key": "desc"}


@pytest.mark.asyncio
async def test_field_discovery_degrades_to_a_flat_sample_not_an_empty_schema():
    """A cluster that buckets nothing must still discover fields.

    An empty schema does not fail loudly — the generator is told to infer field names, so
    it invents plausible ones and the source returns 0 rows. So the no-buckets response
    has to fall back to the flat sample it replaced, which needs a SECOND request: the
    aggregation is sent with `size: 0` and carries no documents to fall back to.
    """
    config = {
        "name": "s",
        "url": "https://kibana.example.net",
        "username": "u",
        "password": "p",
        "index": "i*",
    }
    retriever = KibanaRetriever(config, MagicMock())
    no_buckets = _FakeResp(
        {
            "rawResponse": {
                "hits": {"hits": []},
                "aggregations": {"idx": {"buckets": []}},
            }
        }
    )
    flat = _FakeResp(
        {"rawResponse": {"hits": {"hits": [{"_source": {"realField": 1}}]}}}
    )
    session = MagicMock()
    session.closed = False
    session.post = MagicMock(side_effect=[no_buckets, flat])
    retriever._session = session

    schema = await retriever._get_field_schema()

    assert schema == "realField"
    assert (
        session.post.call_count == 2
    ), "the fallback must re-ask; size:0 carried no docs"
    assert session.post.call_args_list[1].kwargs["json"]["params"]["body"]["size"] > 0


@pytest.mark.asyncio
async def test_a_binding_under_an_array_is_confirmed_by_the_discovered_schema():
    """Discovery and the stale-binding decision must agree, and one seam joins them.

    `_in_schema` is what decides whether a pack's declared binding is adopted or
    reported STALE, and its only input is this schema string — so an array-nested path
    missing from the string is a measured declaration reported as a vanished field.
    Measured live on `session.azure.ic*`: `svcAlerts.record_ref` is a populated `keyword`
    on all 12 monthly indices and on 200/200 sampled documents, the discovery could not
    see it, the predicate was dropped, and one day's window-only scan returned 0 rows.
    """
    from src.retrievers.field_mapping import _in_schema, _schema_field_names

    config = {
        "name": "ic_sessions",
        "url": "https://kibana.example.net",
        "username": "u",
        "password": "p",
        "index": "session.azure.ic*",
    }
    retriever = KibanaRetriever(config, MagicMock())
    sample = _FakeResp(
        {
            "rawResponse": {
                "hits": {"hits": []},
                "aggregations": {
                    "idx": {
                        "buckets": [
                            {
                                "key": "session.azure.ic.2026.08",
                                "docs": {
                                    "hits": {
                                        "hits": [
                                            {
                                                "_source": {
                                                    "unitId": "O1",
                                                    "svcAlerts": [
                                                        {
                                                            "record_ref": "AAA111",
                                                            "loginArea": {
                                                                "sign": "0505VW"
                                                            },
                                                        },
                                                        {"record_ref": "BBB222"},
                                                    ],
                                                }
                                            }
                                        ]
                                    }
                                },
                            }
                        ]
                    }
                },
            }
        }
    )
    session = MagicMock()
    session.closed = False
    session.post = MagicMock(return_value=sample)
    retriever._session = session

    schema = await retriever._get_field_schema()
    names = _schema_field_names(schema)

    assert _in_schema("svcAlerts.record_ref", names) == "svcAlerts.record_ref"
    sign = _in_schema("svcAlerts.loginArea.sign", names)
    assert sign == "svcAlerts.loginArea.sign"
    # And the flat sibling still resolves — the fix adds paths, it removes none.
    assert _in_schema("unitId", names) == "unitId"


def test_sampled_hits_reads_either_response_shape():
    agg = {
        "aggregations": {
            "idx": {"buckets": [{"docs": {"hits": {"hits": [{"_source": {"a": 1}}]}}}]}
        },
        "hits": {"hits": [{"_source": {"ignored": 1}}]},
    }
    assert _sampled_hits(agg) == [{"_source": {"a": 1}}]
    # No aggregation at all (an older cluster, or one that dropped it) -> the flat hits.
    assert _sampled_hits({"hits": {"hits": [{"_source": {"b": 2}}]}}) == [
        {"_source": {"b": 2}}
    ]
    assert _sampled_hits({}) == []


@pytest.mark.asyncio
async def test_kibana_retriever_field_discovery_falls_back_on_error():
    config = {
        "name": "siem_alerts",
        "url": "https://kibana.example.net",
        "username": "u",
        "password": "p",
        "index": "i*",
        "field_schema": "fallback_field",
    }
    retriever = KibanaRetriever(config, MagicMock())

    class _BoomResp:
        async def __aenter__(self):
            raise RuntimeError("network down")

        async def __aexit__(self, *a):
            return False

    session = MagicMock()
    session.closed = False
    session.post = MagicMock(return_value=_BoomResp())
    retriever._session = session

    schema = await retriever._get_field_schema()
    assert schema == "fallback_field"  # falls back, never raises


@pytest.mark.asyncio
async def test_node_direct_gateway_addresses_the_index_and_unwraps_nothing():
    """The same query language against a data node instead of the Kibana proxy.

    A network can reach 9200 and not Kibana's own port, and switching to the ES|QL
    retriever there would switch query LANGUAGES — invalidating every pack ``query_hints``
    written for DSL. So the route lives in this retriever, and the three things that differ
    are the path, the unwrapped body, and the absent ``rawResponse`` envelope.
    """
    config = {
        "name": "siem_alerts",
        "gateway": "node",
        "url": "https://es-node-1.example.net:9200",
        "username": "u",
        "password": "p",
        "index": "raw.prd.siem-alerts*",
        "max_results": 25,
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=EsQueryDsl(
            query={"bool": {"filter": [{"range": {"timestamp": {}}}]}}
        )
    )
    retriever = KibanaRetriever(config, llm)

    # A node answers with the response itself — no envelope to unwrap.
    sample = _FakeResp({"hits": {"hits": [{"_source": {"alertId": "x"}}]}})
    real = _FakeResp({"hits": {"hits": [{"_source": {"alertId": "a1"}}]}})
    session = MagicMock()
    session.closed = False
    session.post = MagicMock(side_effect=[sample, real])
    retriever._session = session

    rows = await retriever.retrieve(_query("siem_alerts"))

    assert rows == [{"alertId": "a1"}]
    for call in session.post.call_args_list:
        assert call.args[0] == (
            "https://es-node-1.example.net:9200/raw.prd.siem-alerts*/_search"
        )
    # The body is the search body itself, and the retriever still owns the row cap.
    real_body = session.post.call_args_list[1].kwargs["json"]
    assert "params" not in real_body
    assert real_body["size"] == 25
    assert "bool" in real_body["query"]


@pytest.mark.asyncio
async def test_the_gateway_route_is_unchanged_and_is_the_default():
    """A config naming no gateway is the gateway one — every caller that predates the
    node route meant that, and this retriever is constructed directly in several places."""
    config = {"name": "s", "url": "https://kibana.example.net", "index": "i*"}
    assert KibanaRetriever(config, MagicMock()).gateway == "kibana"
    assert KibanaRetriever(dict(config, gateway="node"), MagicMock()).gateway == "node"


def test_a_node_direct_session_does_not_claim_to_be_kibana():
    """``x-elastic-internal-origin`` is what ES reads to relax system-index restrictions.
    Sending it while talking to the node directly is claiming to be a component we are not.
    """
    from src.retrievers.kibana_retriever import _KIBANA_HEADERS, _NODE_HEADERS

    assert "x-elastic-internal-origin" in _KIBANA_HEADERS
    assert "kbn-xsrf" in _KIBANA_HEADERS
    assert set(_NODE_HEADERS) == {"Content-Type"}


def test_merge_endpoint_elasticsearch_routes_to_kibana_gateway():
    """A backend with gateway:kibana yields type 'kibana' (else 'elasticsearch')."""
    from types import SimpleNamespace

    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)
    backends = {
        "elasticsearch": {
            "secondary-prd": {
                "url": "https://kibana.example.net",
                "gateway": "kibana",
                "username": "u",
                "password": "p",
            },
            "direct-prd": {
                "url": "https://es.example.net:9200",
                "username": "u",
                "password": "p",
            },
        }
    }
    engine.config = {"backends": backends}
    gw_src = SimpleNamespace(
        name="siem_alerts",
        endpoints={
            "kind": "elasticsearch",
            "cluster": "secondary-prd",
            "indices": ["i*"],
        },
    )
    direct_src = SimpleNamespace(
        name="app_logs",
        endpoints={"kind": "elasticsearch", "cluster": "direct-prd", "indices": ["j*"]},
    )
    assert engine._merge_endpoint(gw_src, "elasticsearch", backends)["type"] == "kibana"
    assert (
        engine._merge_endpoint(direct_src, "elasticsearch", backends)["type"]
        == "elasticsearch"
    )


def test_merge_endpoint_routes_gateway_node_to_the_dsl_retriever_and_carries_it():
    """``gateway: node`` picks Query DSL over ES|QL *and* the addressing inside it.

    Two halves, and the wrong one alone is silent: routing without threading the value
    builds a retriever that POSTs ``/internal/search/es`` at a data node (404 on every
    query), and threading without routing leaves the value on an ES|QL config that reads it
    nowhere.
    """
    from types import SimpleNamespace

    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)
    backends = {
        "elasticsearch": {
            "node-prd": {
                "url": "https://es-node-1.example.net:9200",
                "gateway": "node",
                "username": "u",
                "password": "p",
            },
            "esql-prd": {
                "url": "https://es.example.net:9200",
                "username": "u",
                "password": "p",
            },
        }
    }
    engine.config = {"backends": backends}

    def src(cluster):
        return SimpleNamespace(
            name="app_logs",
            endpoints={
                "kind": "elasticsearch",
                "cluster": cluster,
                "indices": ["j*"],
            },
        )

    node = engine._merge_endpoint(src("node-prd"), "elasticsearch", backends)
    assert node["type"] == "kibana"
    assert node["gateway"] == "node"
    # And an undeclared gateway still means ES|QL, carrying no gateway to read.
    esql = engine._merge_endpoint(src("esql-prd"), "elasticsearch", backends)
    assert esql["type"] == "elasticsearch"
    assert esql["gateway"] is None


def test_merge_endpoint_threads_default_filters_and_query_hints():
    """A pack source's default_filters + query_hints reach the ES/Kibana config so
    the retriever can enforce e.g. type=scheme deterministically."""
    from types import SimpleNamespace

    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)
    backends = {
        "elasticsearch": {
            "secondary-prd": {
                "url": "https://kibana.example.net",
                "gateway": "kibana",
                "username": "u",
                "password": "p",
            }
        }
    }
    engine.config = {"backends": backends}
    src = SimpleNamespace(
        name="siem_alerts_current",
        endpoints={
            "kind": "elasticsearch",
            "cluster": "secondary-prd",
            "indices": ["raw.prd.siem-alerts*"],
        },
        default_filters={"type": "scheme"},
        query_hints="always type=scheme",
    )
    merged = engine._merge_endpoint(src, "elasticsearch", backends)
    # `query_hints` is per-branch (the prompt it feeds differs per dialect); `default_filters`
    # rides `_attach_query_guards`, because it lived in this branch alone and was therefore a
    # silent no-op on the other three — see the seam test further down.
    assert merged["query_hints"] == "always type=scheme"
    engine._attach_query_guards(src, merged)
    assert merged["default_filters"] == {"type": "scheme"}
    # A source without the keys still merges (empty defaults, no crash).
    bare = SimpleNamespace(
        name="x",
        endpoints={
            "kind": "elasticsearch",
            "cluster": "secondary-prd",
            "indices": ["i*"],
        },
    )
    merged_bare = engine._merge_endpoint(bare, "elasticsearch", backends)
    engine._attach_query_guards(bare, merged_bare)
    assert merged_bare["default_filters"] == {}
    assert merged_bare["query_hints"] == ""


# --- Databricks ------------------------------------------------------------


class _FakeResp:
    # `status` is modelled because the retrievers now read it: a fake without one cannot
    # tell a 200 from a 403, which is the whole thing the real code decides here.
    def __init__(self, payload, status=200, reason="OK"):
        self._payload = payload
        self.status = status
        self.reason = reason
        self.request_info = None
        self.history = ()
        self.headers = {}

    async def text(self):
        return json.dumps(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_databricks_retriever_polls_until_succeeded():
    config = {
        "name": "transaction_logs",
        "workspace_url": "https://workspace/",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "poll_interval_seconds": 0,
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=SqlQuery(query="SELECT * FROM t"))

    retriever = DatabricksRetriever(config, llm)

    post_resp = _FakeResp({"statement_id": "s1", "status": {"state": "RUNNING"}})
    get_running = _FakeResp({"statement_id": "s1", "status": {"state": "RUNNING"}})
    get_done = _FakeResp(
        {
            "statement_id": "s1",
            "status": {"state": "SUCCEEDED"},
            "manifest": {"schema": {"columns": [{"name": "amount"}]}},
            "result": {"data_array": [["100"]]},
        }
    )

    session = MagicMock()
    session.closed = False
    session.post = MagicMock(return_value=post_resp)
    session.get = MagicMock(side_effect=[get_running, get_done])
    retriever._session = session

    rows = await retriever.retrieve(_query("transaction_logs"))

    assert session.get.call_count == 2  # polled twice: RUNNING then SUCCEEDED
    assert rows == [{"amount": "100"}]


@pytest.mark.asyncio
async def test_databricks_retriever_raises_on_failure():
    config = {
        "name": "transaction_logs",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "poll_interval_seconds": 0,
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=SqlQuery(query="SELECT * FROM t"))

    retriever = DatabricksRetriever(config, llm)
    post_resp = _FakeResp(
        {
            "statement_id": "s1",
            "status": {"state": "FAILED", "error": {"message": "bad sql"}},
        }
    )
    session = MagicMock()
    session.closed = False
    session.post = MagicMock(return_value=post_resp)
    retriever._session = session

    with pytest.raises(RuntimeError, match="bad sql"):
        await retriever.retrieve(_query("transaction_logs"))


@pytest.mark.asyncio
async def test_databricks_discovers_field_schema_when_unset():
    config = {
        "name": "transaction_logs",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "catalog": "main",
        "schema": "fraud",
        # no field_schema -> should be discovered from information_schema
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=SqlQuery(query="SELECT 1"))

    retriever = DatabricksRetriever(config, llm)
    # First _execute_sql call (discovery) returns information_schema rows.
    retriever._execute_sql = AsyncMock(
        return_value=[
            {
                "table_name": "transactions",
                "column_name": "user_id",
                "data_type": "string",
            },
            {
                "table_name": "transactions",
                "column_name": "amount",
                "data_type": "double",
            },
        ]
    )

    schema = await retriever._get_field_schema()
    assert schema == "transactions(user_id string, amount double)"

    # Second call is cached: _execute_sql not invoked again.
    retriever._execute_sql.reset_mock()
    assert await retriever._get_field_schema() == schema
    retriever._execute_sql.assert_not_called()


@pytest.mark.asyncio
async def test_databricks_scans_all_schemas_and_curates():
    """catalog set but schema unset -> scan all schemas, LLM curates tables."""
    config = {
        "name": "transaction_logs",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "catalog": "main",
        # no schema -> triggers scan + curation
    }
    llm = MagicMock()
    # The curation call returns the fraud-relevant subset.
    llm.structured_output = AsyncMock(
        return_value=SchemaSelection(
            tables=["fraud.transactions"], reasoning="payments"
        )
    )

    retriever = DatabricksRetriever(config, llm)

    async def fake_execute(sql):
        if "information_schema.tables" in sql:
            return [
                {"table_schema": "fraud", "table_name": "transactions"},
                {"table_schema": "staging", "table_name": "tmp_load"},
            ]
        if "information_schema.columns" in sql:
            # Only queried for the 'fraud' schema (the curated pick).
            assert "table_schema = 'fraud'" in sql
            return [
                {
                    "table_name": "transactions",
                    "column_name": "amount",
                    "data_type": "double",
                },
            ]
        raise AssertionError(f"unexpected sql: {sql}")

    retriever._execute_sql = AsyncMock(side_effect=fake_execute)

    schema = await retriever._get_field_schema()
    assert schema == "fraud.transactions(amount double)"
    # Curation was invoked once with the full table list.
    llm.structured_output.assert_awaited_once()


@pytest.mark.asyncio
async def test_databricks_maps_entities_to_real_columns_in_sql():
    """Entities are mapped onto discovered columns and used to filter the SQL."""
    from src.knowledge.pack import EntityDef, KnowledgePack
    from src.models.pydantic_models import (EntityFieldMap, ExtractedEntity,
                                            FieldMapping)

    config = {
        "name": "transaction_logs",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "field_schema": "transactions(org_unit_id string, agent_sign string)",
    }
    pack = KnowledgePack(
        entities=[EntityDef(type="org_unit", field_aliases=["org_unit_id"])]
    )
    llm = MagicMock()

    # First structured_output call = entity mapping; second = SQL generation.
    llm.structured_output = AsyncMock(
        side_effect=[
            FieldMapping(
                mappings=[
                    EntityFieldMap(
                        entity_type="org_unit", field="org_unit_id", confidence=0.9
                    )
                ]
            ),
            SqlQuery(
                query="SELECT * FROM transactions WHERE org_unit_id = 'ORGUNIT2301'"
            ),
        ]
    )

    retriever = DatabricksRetriever(config, llm, knowledge_pack=pack)
    retriever._execute_sql = AsyncMock(return_value=[{"org_unit_id": "ORGUNIT2301"}])

    query = RetrievalQuery(
        target_log_source="transaction_logs",
        natural_language_query="issuance for org_unit",
        date_from="2024-08-25",
        date_to="2024-08-27",
        entities=[ExtractedEntity(type="org_unit", value="ORGUNIT2301")],
    )
    rows = await retriever.retrieve(query)

    assert rows == [{"org_unit_id": "ORGUNIT2301"}]
    # The SQL-generation prompt received the mapped column as a filter hint.
    sql_call_messages = llm.structured_output.call_args_list[1].args[0]
    system_msg = sql_call_messages[0]["content"]
    assert "org_unit_id = 'ORGUNIT2301'" in system_msg


@pytest.mark.asyncio
async def test_databricks_query_hints_reach_sql_prompt():
    """A source's query_hints are injected verbatim into the SQL-generation prompt,
    alongside the always-on performance guard (no gratuitous ORDER BY)."""
    from src.models.pydantic_models import FieldMapping

    config = {
        "name": "auth_events",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "field_schema": "auth_events_secondary(id string, dateTime string)",
        "query_hints": "Do NOT use ORDER BY; filter dateTime as an ISO string prefix.",
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        side_effect=[
            FieldMapping(mappings=[]),
            SqlQuery(query="SELECT id FROM auth_events_secondary"),
        ]
    )
    retriever = DatabricksRetriever(config, llm)
    retriever._execute_sql = AsyncMock(return_value=[])

    query = RetrievalQuery(
        target_log_source="auth_events",
        natural_language_query="auth events",
        date_from="2026-07-17",
        date_to="2026-07-18",
    )
    await retriever.retrieve(query)

    system_msg = llm.structured_output.call_args_list[1].args[0][0]["content"]
    assert "Do NOT use ORDER BY; filter dateTime as an ISO string prefix." in system_msg
    assert "PERFORMANCE:" in system_msg  # always-on guard present too


@pytest.mark.asyncio
async def test_databricks_sql_prompt_ors_weak_entities_not_all_and():
    """When entities map to columns, the SQL prompt must instruct the model to treat
    weak entities as OR-ed EVIDENCE around a selective identifier — not AND every
    entity together. ANDing org_unit+sign+provider onto the record is what returned 0 rows
    for record_lake even though the record existed."""
    from src.models.pydantic_models import (EntityFieldMap, ExtractedEntity,
                                            FieldMapping)

    config = {
        "name": "record_lake",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "field_schema": "record_table_4(locator string, creation_date date, creator string)",
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        side_effect=[
            FieldMapping(
                mappings=[
                    EntityFieldMap(
                        entity_type="record", field="locator", confidence=0.95
                    ),
                    EntityFieldMap(
                        entity_type="org_unit", field="creator", confidence=0.8
                    ),
                ]
            ),
            SqlQuery(
                query="SELECT locator FROM record_table_4 WHERE locator = 'SUBJ03'"
            ),
        ]
    )
    retriever = DatabricksRetriever(config, llm)
    retriever._execute_sql = AsyncMock(return_value=[])

    query = RetrievalQuery(
        target_log_source="record_lake",
        natural_language_query="record version",
        date_from="2026-07-24",
        date_to="2026-07-24",
        entities=[
            ExtractedEntity(type="record", value="SUBJ03"),
            ExtractedEntity(type="org_unit", value="ORG2428D4"),
        ],
    )
    await retriever.retrieve(query)

    system_msg = llm.structured_output.call_args_list[1].args[0][0]["content"]
    # The prompt names the selective-vs-weak / OR-not-AND strategy.
    assert "EVIDENCE TO FIND" in system_msg
    assert "SELECTIVE" in system_msg
    assert "do NOT AND every entity" in system_msg


@pytest.mark.asyncio
async def test_databricks_rewrites_or_to_and_for_composite_key_source():
    """`require_all_entities` is enforced DETERMINISTICALLY, not just requested.

    The prompt carries a strong general "OR the weak entities" rule (the test above), so on
    a composite-key lookup the model routinely ORs org_unit and sign anyway. There the OR is
    not a performance nit but a wrong answer: sign '6009JJ' is on 89,937 rows across as many
    org_units, so the query gets truncated at the row cap and returns other org_units' robots.
    IR10000002 lost a clean AUTOMATED exclusion exactly that way, so the OR is rewritten.
    """
    from src.models.pydantic_models import (EntityFieldMap, ExtractedEntity,
                                            FieldMapping)

    config = {
        "name": "automation_registry",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "field_schema": "automation_registry(orgUnitId string, sign string, profile string)",
        "require_all_entities": ["org_unit", "user"],
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        side_effect=[
            FieldMapping(
                mappings=[
                    EntityFieldMap(
                        entity_type="org_unit", field="orgUnitId", confidence=0.95
                    ),
                    EntityFieldMap(entity_type="user", field="sign", confidence=0.95),
                ]
            ),
            # The model follows the generic OR guidance — exactly the failure mode.
            SqlQuery(
                query=(
                    "SELECT orgUnitId, sign, profile FROM automation_registry "
                    "WHERE orgUnitId = 'QQQ1R17GH' OR sign = '6009JJ'"
                )
            ),
        ]
    )
    retriever = DatabricksRetriever(config, llm)
    retriever._execute_sql = AsyncMock(return_value=[])

    query = RetrievalQuery(
        target_log_source="automation_registry",
        natural_language_query="is this identity automated",
        date_from="2026-07-17",
        date_to="2026-07-17",
        entities=[
            ExtractedEntity(type="org_unit", value="QQQ1R17GH"),
            ExtractedEntity(type="user", value="6009JJ"),
        ],
    )
    await retriever.retrieve(query)

    executed = retriever._execute_sql.call_args.args[0]
    assert not re.search(r"\bOR\b", executed, re.IGNORECASE)
    assert "orgUnitId = 'QQQ1R17GH' AND sign = '6009JJ'" in executed
    # The AND requirement also reaches the prompt (belt and braces).
    system_msg = llm.structured_output.call_args_list[1].args[0][0]["content"]
    assert "MANDATORY" in system_msg


@pytest.mark.asyncio
async def test_databricks_leaves_or_alone_without_require_all_entities():
    """The rewrite must be opt-in: an ordinary event-log source keeps its OR-ed entities,
    which is what stops one mismatched value returning zero rows."""
    from src.models.pydantic_models import (EntityFieldMap, ExtractedEntity,
                                            FieldMapping)

    config = {
        "name": "auth_events",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "field_schema": "auth(org_unit string, sign string)",
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        side_effect=[
            FieldMapping(
                mappings=[
                    EntityFieldMap(
                        entity_type="org_unit", field="org_unit", confidence=0.9
                    ),
                    EntityFieldMap(entity_type="user", field="sign", confidence=0.9),
                ]
            ),
            SqlQuery(query="SELECT * FROM auth WHERE org_unit = 'A' OR sign = 'B'"),
        ]
    )
    retriever = DatabricksRetriever(config, llm)
    retriever._execute_sql = AsyncMock(return_value=[])
    await retriever.retrieve(
        RetrievalQuery(
            target_log_source="auth_events",
            natural_language_query="auth",
            date_from="2026-07-17",
            date_to="2026-07-17",
            entities=[
                ExtractedEntity(type="org_unit", value="A"),
                ExtractedEntity(type="user", value="B"),
            ],
        )
    )
    assert "OR sign = 'B'" in retriever._execute_sql.call_args.args[0]


def _never_filter_retriever(generated_sql):
    """A DatabricksRetriever whose SQL-gen returns `generated_sql`, with the robot flag
    declared as evidence (never_filter)."""
    from src.models.pydantic_models import FieldMapping

    config = {
        "name": "auth_events",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "field_schema": "auth(org_unit string, sign string, robot boolean)",
        "never_filter": ["value.payload.userInfo.robot"],
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        side_effect=[FieldMapping(mappings=[]), SqlQuery(query=generated_sql)]
    )
    retriever = DatabricksRetriever(config, llm)
    retriever._execute_sql = AsyncMock(return_value=[])
    return retriever


def _auth_query():
    return RetrievalQuery(
        target_log_source="auth_events",
        natural_language_query="auth events for the actor",
        date_from="2026-07-17",
        date_to="2026-07-17",
    )


@pytest.mark.asyncio
async def test_databricks_strips_filter_on_evidence_field():
    """`never_filter` fields must be RETURNED, never FILTERED — enforced, not requested.

    `robot = false` reads as "drop the service-account noise" and is actually the opposite:
    a automated actor is a SCHEME §3.2 fraud EXCLUSION, so that predicate deletes exactly the
    rows that decide the verdict. On IR10000002 the generated SQL carried it, the flag came
    back only as `false`, the `not_automated` condition read "flag absent" -> UNKNOWN, and the
    engine returned VALID FRAUD on the customer's own web-service application."""
    retriever = _never_filter_retriever(
        "SELECT value.payload.userInfo.robot AS robot_flag FROM auth "
        "WHERE date = DATE'2026-07-17' AND value.payload.userInfo.org_unit = 'QQQ1R17GH' "
        "AND value.payload.userInfo.robot = false"
    )
    await retriever.retrieve(_auth_query())

    executed = retriever._execute_sql.call_args.args[0]
    # The predicate is gone from the WHERE...
    where = executed.upper().split("WHERE", 1)[1]
    assert "ROBOT = FALSE" not in where.replace(" ", " ")
    # ...but the surviving filters and the PROJECTION of the flag are untouched.
    assert "date = DATE'2026-07-17'" in executed
    assert "org_unit = 'QQQ1R17GH'" in executed
    assert "AS robot_flag" in executed


@pytest.mark.asyncio
async def test_databricks_strips_leading_evidence_filter():
    """The predicate may be FIRST in the WHERE — then the trailing AND goes with it."""
    retriever = _never_filter_retriever(
        "SELECT * FROM auth WHERE robot = false AND date = DATE'2026-07-17'"
    )
    await retriever.retrieve(_auth_query())

    executed = retriever._execute_sql.call_args.args[0]
    assert "robot" not in executed.split("WHERE", 1)[1]
    assert "date = DATE'2026-07-17'" in executed
    # No dangling AND / empty WHERE.
    assert "WHERE AND" not in " ".join(executed.upper().split())


@pytest.mark.asyncio
async def test_databricks_leaves_sql_alone_without_never_filter():
    """Opt-in: a source that declares nothing keeps whatever the model generated."""
    from src.models.pydantic_models import FieldMapping

    config = {
        "name": "other",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "field_schema": "t(robot boolean)",
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        side_effect=[
            FieldMapping(mappings=[]),
            SqlQuery(query="SELECT * FROM t WHERE robot = false"),
        ]
    )
    retriever = DatabricksRetriever(config, llm)
    retriever._execute_sql = AsyncMock(return_value=[])
    await retriever.retrieve(_auth_query())
    assert "robot = false" in retriever._execute_sql.call_args.args[0]


@pytest.mark.asyncio
async def test_databricks_drops_an_evidence_filter_that_is_a_whole_or_arm():
    """An OR-ed evidence arm is DROPPED — the one shape where leaving it is the worse read.

    An AND-ed predicate on an evidence field deletes rows, so removing it WIDENS the query and
    is done carefully. OR-ed, the same predicate is true of every row carrying that value, so it
    satisfies the group ALONE, the arm scoping it beside it selects nothing, and the answer comes
    back FULL — which no "did anything come back" check can see. Dropping the disjunct NARROWS,
    which is the safe direction, and the rest of the tree is re-emitted verbatim: still valid
    SQL, no half-removed predicate.
    """
    retriever = _never_filter_retriever(
        "SELECT * FROM auth WHERE date = DATE'2026-07-17' "
        "AND (robot = false OR org_unit = 'QQQ1R17GH')"
    )
    await retriever.retrieve(_auth_query())

    executed = retriever._execute_sql.call_args.args[0]
    assert "robot" not in executed.split("WHERE", 1)[1]
    # The survivor is unchanged and still grouped, so nothing around it has to be re-read.
    assert "(org_unit = 'QQQ1R17GH')" in executed
    assert "date = DATE'2026-07-17'" in executed
    # No disjunction is left at all — a one-armed group, not a group with a hole in it.
    assert " OR " not in executed.upper()


@pytest.mark.asyncio
async def test_databricks_keeps_an_evidence_or_arm_when_dropping_it_would_empty_the_group():
    """Every arm on the evidence field → LEFT ALONE, because there is no narrowing to be had.

    A group with nothing surviving cannot be shortened, only broken. This is the refusal that
    keeps the guard from ever producing an empty `()` — and it is reachable: a generator handed
    one entity and one evidence field writes exactly this.
    """
    retriever = _never_filter_retriever(
        "SELECT * FROM auth WHERE date = DATE'2026-07-17' "
        "AND (robot = false OR robot = true)"
    )
    await retriever.retrieve(_auth_query())

    executed = retriever._execute_sql.call_args.args[0]
    assert "(robot = false OR robot = true)" in executed


@pytest.mark.asyncio
async def test_databricks_keeps_a_compound_evidence_or_arm():
    """Only a PLAIN comparison may be dropped — a compound arm names other columns too.

    An arm this module cannot fully account for may carry a real identity predicate beside the
    evidence one, so dropping it whole would delete a scoping this guard has no licence over.
    Left exactly as generated.
    """
    retriever = _never_filter_retriever(
        "SELECT * FROM auth WHERE date = DATE'2026-07-17' "
        "AND ((robot = false AND sign = '6007GG') OR org_unit = 'QQQ1R17GH')"
    )
    await retriever.retrieve(_auth_query())

    executed = retriever._execute_sql.call_args.args[0]
    assert "(robot = false AND sign = '6007GG')" in executed


@pytest.mark.asyncio
async def test_databricks_keeps_an_evidence_or_arm_under_a_negation():
    """A negation anywhere in the statement refuses the whole rewrite.

    ``NOT (a OR b)`` → ``NOT (a)`` EXCLUDES LESS, which is the one direction forbidden here, and
    the group walker offers a group without its surrounding context — so the refusal is coarse
    and covers the statement rather than being decided per group.
    """
    retriever = _never_filter_retriever(
        "SELECT * FROM auth WHERE date = DATE'2026-07-17' "
        "AND NOT (robot = false OR org_unit = 'QQQ1R17GH')"
    )
    await retriever.retrieve(_auth_query())

    executed = retriever._execute_sql.call_args.args[0]
    assert "NOT (robot = false OR org_unit = 'QQQ1R17GH')" in executed


@pytest.mark.asyncio
async def test_databricks_pins_exact_table_in_sql_prompt():
    """The configured table(s) are pinned in the SQL prompt so the LLM never invents
    a table name — critical when schema discovery timed out (field_schema empty)."""
    from src.models.pydantic_models import FieldMapping

    config = {
        "name": "record_lake",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "catalog": "wh_primary_prd",
        "schema": "src_tables",
        "tables": ["record_table_4"],
        "field_schema": "",  # discovery "failed" -> empty schema
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        side_effect=[
            FieldMapping(mappings=[]),
            SqlQuery(query="SELECT locator FROM record_table_4"),
        ]
    )
    retriever = DatabricksRetriever(config, llm)
    # Discovery attempted-and-empty so _get_field_schema returns "" without I/O.
    retriever._discovery_attempted = True
    retriever._execute_sql = AsyncMock(return_value=[])

    query = RetrievalQuery(
        target_log_source="record_lake",
        natural_language_query="record history",
        date_from="2026-07-17",
        date_to="2026-07-18",
    )
    await retriever.retrieve(query)

    system_msg = llm.structured_output.call_args_list[1].args[0][0]["content"]
    assert "wh_primary_prd.src_tables.record_table_4" in system_msg
    assert "Do NOT invent" in system_msg
    # And the generated query is captured for UI/API visibility.
    assert retriever.last_generated_query == "SELECT locator FROM record_table_4"


@pytest.mark.asyncio
async def test_databricks_warms_warehouse_before_discovery():
    """The warehouse is warmed once (SELECT 1) before schema discovery, so the
    cold-start cost is paid upfront rather than raced by the discovery query."""
    config = {
        "name": "record_lake",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "catalog": "main",
        "schema": "fraud",
        # no field_schema -> discovery runs -> warmup precedes it
    }
    llm = MagicMock()
    calls = []

    async def fake_execute(sql):
        calls.append(sql)
        if sql == "SELECT 1":
            return [{"1": 1}]
        return [{"table_name": "t", "column_name": "c", "data_type": "string"}]

    retriever = DatabricksRetriever(config, llm)
    retriever._execute_sql = AsyncMock(side_effect=fake_execute)

    await retriever._get_field_schema()
    # SELECT 1 warm-up ran first, then the information_schema discovery query.
    assert calls[0] == "SELECT 1"
    assert any("information_schema" in c for c in calls[1:])


@pytest.mark.asyncio
async def test_databricks_warmup_disabled_by_config():
    """warehouse_warmup: false skips the SELECT 1 warm-up entirely."""
    config = {
        "name": "record_lake",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "catalog": "main",
        "schema": "fraud",
        "warehouse_warmup": False,
    }
    llm = MagicMock()
    retriever = DatabricksRetriever(config, llm)
    retriever._execute_sql = AsyncMock(
        return_value=[{"table_name": "t", "column_name": "c", "data_type": "string"}]
    )
    await retriever._get_field_schema()
    # Only the discovery query ran; no SELECT 1.
    for call in retriever._execute_sql.call_args_list:
        assert call.args[0] != "SELECT 1"


@pytest.mark.asyncio
async def test_databricks_uses_auth_token_for_session():
    """When a DatabricksAuth is supplied, the session bearer comes from it."""
    config = {
        "name": "transaction_logs",
        "warehouse_id": "wh1",
        # no workspace_url, no api_key_env -> both come from auth
    }
    auth = MagicMock()
    auth.host = "https://from-auth"
    auth.token = MagicMock(return_value="oauth-token")

    retriever = DatabricksRetriever(config, MagicMock(), auth=auth)
    assert retriever.workspace_url == "https://from-auth"
    assert retriever._bearer_token() == "oauth-token"


# --- Dispatcher ------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_dispatches_to_correct_retriever():
    config = {
        "sources": [
            {
                "name": "application_logs",
                "type": "elasticsearch",
                "url": "x",
                "username": "u",
                "password": "p",
                "index": "i",
            },
            {
                "name": "transaction_logs",
                "type": "databricks",
                "workspace_url": "https://w",
                "warehouse_id": "wh",
                "api_key_env": "DATABRICKS_TOKEN",
            },
        ]
    }
    engine = LogRetrievalEngine(config, llm_client=MagicMock())

    assert set(engine.retrievers) == {"application_logs", "transaction_logs"}

    engine.retrievers["application_logs"].retrieve = AsyncMock(return_value=[{"a": 1}])
    engine.retrievers["transaction_logs"].retrieve = AsyncMock(return_value=[{"b": 2}])

    logs = await engine.retrieve(
        [_query("application_logs"), _query("transaction_logs")]
    )
    assert logs == {"application_logs": [{"a": 1}], "transaction_logs": [{"b": 2}]}


@pytest.mark.asyncio
async def test_engine_skips_unknown_source_type():
    config = {"sources": [{"name": "weird", "type": "mystery"}]}
    engine = LogRetrievalEngine(config, llm_client=MagicMock())
    assert engine.retrievers == {}


# --- Snowflake -------------------------------------------------------------


@pytest.mark.asyncio
async def test_snowflake_retriever_generates_sql_and_executes():
    from src.retrievers.snowflake_retriever import SnowflakeRetriever

    config = {
        "name": "ff_activity_snowflake",
        "account": "acct",
        "user": "u",
        "password": "p",
        "role": "r",
        "warehouse": "wh",
        "databases": ["FF_DB"],
        "schemas": ["CUSTOMER_DATA"],
        "field_schema": "ACTIVITY(user_id string, miles double)",  # skip discovery
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=SqlQuery(query="SELECT * FROM ACTIVITY")
    )
    retriever = SnowflakeRetriever(config, llm)
    # Mock the blocking driver path entirely.
    retriever._execute_sql_sync = MagicMock(return_value=[{"user_id": "42"}])

    rows = await retriever.retrieve(_query("ff_activity_snowflake"))
    assert rows == [{"user_id": "42"}]
    retriever._execute_sql_sync.assert_called_once()


@pytest.mark.asyncio
async def test_snowflake_discovers_schema_when_unset():
    from src.retrievers.snowflake_retriever import SnowflakeRetriever

    config = {
        "name": "ff_activity_snowflake",
        "account": "acct",
        "user": "u",
        "password": "p",
        "database": "FF_DB",
        "schemas": ["CUSTOMER_DATA"],
        "objects": ["ACTIVITY"],
    }
    llm = MagicMock()
    retriever = SnowflakeRetriever(config, llm)
    retriever._execute_sql = AsyncMock(
        return_value=[
            {"TABLE_NAME": "ACTIVITY", "COLUMN_NAME": "user_id", "DATA_TYPE": "TEXT"},
            {"TABLE_NAME": "ACTIVITY", "COLUMN_NAME": "miles", "DATA_TYPE": "NUMBER"},
        ]
    )
    schema = await retriever._get_field_schema()
    assert schema == "ACTIVITY(user_id TEXT, miles NUMBER)"
    # Cached: a second call does not re-query.
    retriever._execute_sql.reset_mock()
    assert await retriever._get_field_schema() == schema
    retriever._execute_sql.assert_not_called()


# --- REST (ServiceNow) -----------------------------------------------------


@pytest.mark.asyncio
async def test_rest_retriever_builds_sysparm_query_and_paginates():
    from src.knowledge.pack import EntityDef, KnowledgePack, SourceDef
    from src.models.pydantic_models import (EntityFieldMap, ExtractedEntity,
                                            FieldMapping)
    from src.retrievers.rest_retriever import RestRetriever

    config = {
        "name": "servicenow_ir",
        "base_url": "https://sn.example.com",
        "username": "u",
        "password": "p",
        "tables": ["incident"],
        "max_results": 100,
        "page_size": 100,
    }
    pack = KnowledgePack(
        entities=[EntityDef(type="incident_ir", field_aliases=["number"])],
        sources=[
            SourceDef(
                name="servicenow_ir",
                entity_bindings={"incident_ir": ["number", "sys_id"]},
            )
        ],
    )
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=FieldMapping(
            mappings=[
                EntityFieldMap(
                    entity_type="incident_ir", field="number", confidence=0.9
                )
            ]
        )
    )
    retriever = RestRetriever(config, llm, knowledge_pack=pack)

    captured = {}

    async def fake_query_table(table, sysparm_query):
        captured["table"] = table
        captured["q"] = sysparm_query
        return [{"number": "INC42"}]

    retriever._query_table = fake_query_table

    query = RetrievalQuery(
        target_log_source="servicenow_ir",
        natural_language_query="incident lookup",
        date_from="2024-01-01",
        date_to="2024-01-31",
        entities=[ExtractedEntity(type="incident_ir", value="INC42")],
    )
    rows = await retriever.retrieve(query)

    assert rows == [{"number": "INC42", "_servicenow_table": "incident"}]
    assert "number=INC42" in captured["q"]
    assert "sys_created_on>=2024-01-01" in captured["q"]


# --- merge_endpoint for snowflake / rest -----------------------------------


def test_merge_endpoint_snowflake_and_rest():
    from types import SimpleNamespace

    config = {
        "backends": {
            "snowflake": {
                "acct1": {
                    "account": "acct1",
                    "user": "u",
                    "password": "p",
                    "role": "r",
                    "warehouse": "wh",
                }
            },
            "rest": {"servicenow": {"base_url": "https://sn", "username": "u"}},
        }
    }
    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)
    engine.config = config

    sf_src = SimpleNamespace(
        name="ff_activity_snowflake",
        endpoints={"kind": "snowflake", "account": "acct1", "objects": ["A"]},
    )
    rest_src = SimpleNamespace(
        name="servicenow_ir",
        endpoints={"kind": "rest", "service": "servicenow", "tables": ["incident"]},
    )
    backends = config["backends"]

    sf = engine._merge_endpoint(sf_src, "snowflake", backends)
    assert sf["type"] == "snowflake" and sf["account"] == "acct1"
    assert sf["objects"] == ["A"]

    rest = engine._merge_endpoint(rest_src, "rest", backends)
    assert rest["type"] == "rest" and rest["base_url"] == "https://sn"
    assert rest["tables"] == ["incident"]


def test_merge_endpoint_skips_missing_creds():
    from types import SimpleNamespace

    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)
    engine.config = {"backends": {}}
    src = SimpleNamespace(
        name="snow", endpoints={"kind": "snowflake", "account": "nope"}
    )
    assert engine._merge_endpoint(src, "snowflake", {}) is None
    # The skip is RECORDED, not merely logged: a log line is not an output, and
    # downstream must be able to tell "this source had nothing" from "this source was
    # never asked" (job 4da14f65 dropped 16 of 30 sources this way, silently).
    assert "snow" in engine._skipped
    assert "nope" in engine._skipped["snow"]


def test_unavailable_sources_names_every_source_that_built_no_retriever():
    """The roster of sources that CANNOT be queried on this run, with the reason."""
    from types import SimpleNamespace

    pack = SimpleNamespace(
        sources=[
            SimpleNamespace(
                name="es_alerts",
                endpoints={"kind": "elasticsearch", "cluster": "prd"},
                kind=lambda: "elasticsearch",
            ),
            SimpleNamespace(
                name="txn_lake",
                endpoints={"kind": "databricks_uc", "catalog": "c", "schema": "s"},
                kind=lambda: "databricks_uc",
            ),
            SimpleNamespace(
                name="odd_kind",
                endpoints={"kind": "carrier_pigeon"},
                kind=lambda: "carrier_pigeon",
            ),
        ],
        source=lambda name: None,
    )
    config = {
        "backends": {
            # URL present but credentials unset (`${VAR}` expands to "") — the exact
            # shape that dropped every ELK source on job 4da14f65.
            "elasticsearch": {"prd": {"url": "https://es", "username": "", "password": ""}},
            "databricks": {"workspace_url": "https://ws", "warehouse_id": "wh1"},
        }
    }
    engine = LogRetrievalEngine(config, llm_client=None, knowledge_pack=pack)

    assert "txn_lake" in engine.retrievers
    assert set(engine.unavailable_sources) == {"es_alerts", "odd_kind"}
    assert "username/password" in engine.unavailable_sources["es_alerts"]
    assert "carrier_pigeon" in engine.unavailable_sources["odd_kind"]
    # A built source is never listed as unavailable.
    assert "txn_lake" not in engine.unavailable_sources


def test_merge_endpoint_databricks_threads_timeout_and_poll_budget():
    """Databricks per-source timeout + widened poll budget flow from the backend creds.

    Uses the FLAT (single-workspace) backend shape — a source with no `workspace`.
    """
    from types import SimpleNamespace

    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)
    backends = {
        "databricks": {
            "workspace_url": "https://ws",
            "warehouse_id": "wh1",
            "retrieval_timeout_seconds": 330,
            "poll_interval_seconds": 5,
            "max_poll_attempts": 60,
        }
    }
    engine.config = {"backends": backends}
    src = SimpleNamespace(
        name="txn", endpoints={"kind": "databricks_uc", "catalog": "c", "schema": "s"}
    )
    merged = engine._merge_endpoint(src, "databricks", backends)
    assert merged["retrieval_timeout_seconds"] == 330
    assert merged["poll_interval_seconds"] == 5
    assert merged["max_poll_attempts"] == 60
    assert merged["workspace"] is None  # flat block -> no workspace selector
    assert merged["query_hints"] == ""  # absent on source -> empty, not an error


def test_merge_endpoint_databricks_threads_query_hints():
    """A pack source's query_hints flow into the retriever config for SQL-gen."""
    from types import SimpleNamespace

    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)
    backends = {"databricks": {"workspace_url": "https://ws", "warehouse_id": "wh1"}}
    engine.config = {"backends": backends}
    src = SimpleNamespace(
        name="auth",
        endpoints={"kind": "databricks_uc", "catalog": "c", "schema": "s"},
        query_hints="Avoid ORDER BY; filter dateTime as a string prefix.",
    )
    merged = engine._merge_endpoint(src, "databricks", backends)
    assert "Avoid ORDER BY" in merged["query_hints"]


def test_merge_endpoint_databricks_workspace_keyed():
    """A source's endpoints.workspace selects one workspace from the keyed map."""
    from types import SimpleNamespace

    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)
    backends = {
        "databricks": {
            "primary_we": {
                "workspace_url": "https://primary_wh",
                "warehouse_id": "wh-primary_wh",
                "api_key_env": "DATABRICKS_TOKEN",
            },
            "secondary_we": {
                "workspace_url": "https://secondary_wh",
                "warehouse_id": "wh-fd",
                "api_key_env": "ANALYTICS_DATABRICKS_TOKEN",
            },
        }
    }
    engine.config = {"backends": backends}

    primary_src = SimpleNamespace(
        name="record",
        endpoints={"kind": "databricks_uc", "workspace": "primary_we", "catalog": "c"},
    )
    fd_src = SimpleNamespace(
        name="scheme",
        endpoints={
            "kind": "databricks_uc",
            "workspace": "secondary_we",
            "catalog": "c",
        },
    )

    primary_wh = engine._merge_endpoint(primary_src, "databricks", backends)
    assert primary_wh["workspace"] == "primary_we"
    assert primary_wh["warehouse_id"] == "wh-primary_wh"
    assert primary_wh["workspace_url"] == "https://primary_wh"
    assert primary_wh["api_key_env"] == "DATABRICKS_TOKEN"

    fd = engine._merge_endpoint(fd_src, "databricks", backends)
    assert fd["workspace"] == "secondary_we"
    assert fd["warehouse_id"] == "wh-fd"
    assert fd["workspace_url"] == "https://secondary_wh"
    assert fd["api_key_env"] == "ANALYTICS_DATABRICKS_TOKEN"


def test_merge_endpoint_databricks_skips_unknown_workspace():
    """A source selecting a workspace with no creds is skipped (graceful degrade)."""
    from types import SimpleNamespace

    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)
    backends = {"databricks": {"primary_we": {"warehouse_id": "wh1"}}}
    engine.config = {"backends": backends}
    src = SimpleNamespace(
        name="auth_svc", endpoints={"kind": "databricks_uc", "workspace": "mexohi_ne"}
    )
    assert engine._merge_endpoint(src, "databricks", backends) is None


def test_auth_for_workspace_only_reuses_matching_host():
    """Shared auth is reused only for its own host; other workspaces fall back to
    api_key_env (return None), because PATs are workspace-scoped."""
    from types import SimpleNamespace

    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)
    shared = SimpleNamespace(host="https://primary_wh")
    engine.auth = shared
    engine._auth_by_host = {"https://primary_wh": shared}

    # Exact host match -> reuse shared SDK/OAuth auth.
    assert engine._auth_for_workspace("https://primary_wh") is shared
    assert engine._auth_for_workspace("https://primary_wh/") is shared  # trailing slash
    # Different workspace -> None so the retriever uses its own api_key_env token.
    assert engine._auth_for_workspace("https://secondary_wh") is None
    # Unknown/blank host -> preserve prior single-workspace behavior (shared auth).
    assert engine._auth_for_workspace(None) is shared


# --- cold-start-tolerant polling / no-resubmit -----------------------------


class _CancelAwareResp:
    """Fake aiohttp response that also supports .read() for the cancel path."""

    def __init__(self, payload, status=200, reason="OK"):
        self._payload = payload
        self.status = status
        self.reason = reason
        self.request_info = None
        self.history = ()
        self.headers = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    async def text(self):
        return json.dumps(self._payload)

    async def json(self):
        return self._payload

    async def read(self):
        return b""


@pytest.mark.asyncio
async def test_databricks_polls_same_statement_no_resubmit():
    """A RUNNING statement is polled by the SAME id — never resubmitted (no new POST)."""
    config = {
        "name": "transaction_logs",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "poll_interval_seconds": 0,
        "max_poll_attempts": 10,
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=SqlQuery(query="SELECT * FROM t"))
    retriever = DatabricksRetriever(config, llm)

    post_resp = _FakeResp({"statement_id": "s1", "status": {"state": "PENDING"}})
    get_running = _FakeResp({"statement_id": "s1", "status": {"state": "RUNNING"}})
    get_done = _FakeResp(
        {
            "statement_id": "s1",
            "status": {"state": "SUCCEEDED"},
            "manifest": {"schema": {"columns": [{"name": "amount"}]}},
            "result": {"data_array": [["100"]]},
        }
    )
    session = MagicMock()
    session.closed = False
    session.post = MagicMock(return_value=post_resp)
    session.get = MagicMock(side_effect=[get_running, get_running, get_done])
    retriever._session = session

    rows = await retriever.retrieve(_query("transaction_logs"))

    assert rows == [{"amount": "100"}]
    # Exactly ONE POST (the initial submit) — the cold-start poll never resubmits.
    assert session.post.call_count == 1
    # All GET polls hit the same statement id.
    for call in session.get.call_args_list:
        assert "s1" in call.args[0]


@pytest.mark.asyncio
async def test_databricks_cancels_orphan_when_budget_exhausted():
    """If the budget is spent while still RUNNING, the statement is cancelled, not resubmitted.

    The budget is WALL-CLOCK (`statement_timeout_seconds`), not an attempt count, and
    exhausting it raises NonRetryableError: re-running the same slow statement from scratch
    would only blow the engine's per-source cap too, turning one slow query into three.
    """
    config = {
        "name": "transaction_logs",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "poll_interval_seconds": 0,
        "statement_timeout_seconds": 0.01,  # tiny budget -> exhausts while RUNNING
        "max_poll_attempts": 2,
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=SqlQuery(query="SELECT * FROM t"))
    retriever = DatabricksRetriever(config, llm)

    post_resp = _FakeResp({"statement_id": "s1", "status": {"state": "PENDING"}})
    get_running = _CancelAwareResp(
        {"statement_id": "s1", "status": {"state": "RUNNING"}}
    )
    cancel_resp = _CancelAwareResp({})

    posts = []

    def _post(url, **kwargs):
        posts.append(url)
        # Cancel is a POST to .../{id}/cancel; the submit is the plain statements URL.
        return cancel_resp if url.endswith("/cancel") else post_resp

    session = MagicMock()
    session.closed = False
    session.post = MagicMock(side_effect=_post)
    session.get = MagicMock(return_value=get_running)
    retriever._session = session

    from src.utils.error_handling import NonRetryableError

    with pytest.raises(NonRetryableError, match="poll budget exhausted"):
        await retriever.retrieve(_query("transaction_logs"))

    # A cancel POST was issued for the orphaned statement.
    assert any(u.endswith("/statements/s1/cancel") for u in posts)
    # ...and the query was submitted exactly ONCE — no retry of a too-slow statement.
    assert len([u for u in posts if not u.endswith("/cancel")]) == 1


@pytest.mark.asyncio
async def test_databricks_tolerates_transient_poll_errors():
    """A failed poll is not a failed statement — the statement keeps running.

    Losing contact briefly (a 5xx, a dropped connection) must not abandon a query that is
    still executing on the warehouse; only `max_consecutive_poll_errors` in a row means we
    have genuinely lost it. Without this, a single blip discarded a query that had already
    burned minutes of warehouse time."""
    config = {
        "name": "transaction_logs",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "poll_interval_seconds": 0,
        "statement_timeout_seconds": 60,
        "max_consecutive_poll_errors": 3,
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=SqlQuery(query="SELECT * FROM t"))
    retriever = DatabricksRetriever(config, llm)

    get_done = _FakeResp(
        {
            "statement_id": "s1",
            "status": {"state": "SUCCEEDED"},
            "manifest": {"schema": {"columns": [{"name": "amount"}]}},
            "result": {"data_array": [["100"]]},
        }
    )
    session = MagicMock()
    session.closed = False
    session.post = MagicMock(
        return_value=_FakeResp({"statement_id": "s1", "status": {"state": "PENDING"}})
    )
    # Two consecutive poll failures (below the threshold), then the result.
    session.get = MagicMock(
        side_effect=[
            ConnectionError("blip"),
            ConnectionError("blip"),
            get_done,
        ]
    )
    retriever._session = session

    assert await retriever.retrieve(_query("transaction_logs")) == [{"amount": "100"}]
    assert session.post.call_count == 1  # never resubmitted


@pytest.mark.asyncio
async def test_databricks_gives_up_after_consecutive_poll_errors():
    """Consecutive poll failures = lost contact = genuinely stuck; that is when we stop."""
    config = {
        "name": "transaction_logs",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "poll_interval_seconds": 0,
        "statement_timeout_seconds": 60,
        "max_consecutive_poll_errors": 3,
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=SqlQuery(query="SELECT * FROM t"))
    retriever = DatabricksRetriever(config, llm)
    session = MagicMock()
    session.closed = False
    session.post = MagicMock(
        return_value=_CancelAwareResp(
            {"statement_id": "s1", "status": {"state": "PENDING"}}
        )
    )
    session.get = MagicMock(side_effect=ConnectionError("gone"))
    retriever._session = session

    with pytest.raises(RuntimeError, match="Lost contact"):
        await retriever.retrieve(_query("transaction_logs"))


# ─────────────────────────────────────────────────────────────────────────────
# STRUCT flattening in Databricks schema discovery
# ─────────────────────────────────────────────────────────────────────────────
def test_flatten_struct_nested():
    """STRUCT<a:INT, b:STRUCT<c:STRING>> -> col.a, col.b.c."""
    leaves = _flatten_struct("x", "STRUCT<a:INT, b:STRUCT<c:STRING>>")
    assert leaves == [("x.a", "INT"), ("x.b.c", "STRING")]


def test_flatten_struct_array_is_terminal():
    """ARRAY<...> is a terminal leaf — NOT descended (dot-access into arrays is
    invalid SQL; elements need explode()/LATERAL VIEW)."""
    leaves = _flatten_struct("arr", "ARRAY<STRUCT<f:INT, g:STRING>>")
    assert len(leaves) == 1
    path, ltype = leaves[0]
    assert path == "arr"
    assert ltype.upper().startswith("ARRAY<")


def test_flatten_struct_nested_array_not_descended():
    """A struct containing an array keeps scalar fields but leaves the array whole."""
    leaves = _flatten_struct("x", "STRUCT<a:INT, items:ARRAY<STRUCT<c:STRING>>>")
    paths = {p for p, _ in leaves}
    assert "x.a" in paths  # scalar struct field flattened
    assert "x.items" in paths  # array field kept as one leaf
    assert "x.items.c" not in paths  # NOT descended into the array


def test_flatten_struct_non_struct_passthrough():
    """A plain scalar column passes through unchanged as a single leaf."""
    assert _flatten_struct("amount", "double") == [("amount", "double")]


def test_flatten_struct_depth_cap():
    """A struct nested past the depth cap is emitted as one leaf, not descended."""
    deep = "STRUCT<a:STRUCT<b:STRUCT<c:STRUCT<d:INT>>>>"
    leaves = _flatten_struct("x", deep)
    # depth cap is 3 -> we descend a, b, c and stop; the remaining STRUCT<d:INT>
    # surfaces as one leaf path.
    paths = [p for p, _ in leaves]
    assert paths == ["x.a.b.c"]
    assert leaves[0][1].upper().startswith("STRUCT<")


def test_render_schema_flattens_and_falls_back_to_data_type():
    """full_data_type is flattened; a row with only data_type behaves as before."""
    rows = [
        {
            "table_name": "record_table_4",
            "column_name": "locator",
            "data_type": "string",
            # no full_data_type -> falls back to data_type (legacy passthrough)
        },
        {
            "table_name": "record_table_4",
            "column_name": "financial",
            "data_type": "struct",
            "full_data_type": "STRUCT<total:DOUBLE, currency:STRING>",
        },
    ]
    rendered = _render_schema(rows)
    assert "locator string" in rendered
    assert "financial.total DOUBLE" in rendered
    assert "financial.currency STRING" in rendered


def test_normalize_struct_paths_rebackticks_segments():
    """`a.b.c` (one quoted token) -> `a`.`b`.`c` (per-segment struct access)."""
    sql = "SELECT `locator.red`, `auditedUseCase.outcome` FROM t"
    out = _normalize_struct_paths(sql)
    assert "`locator`.`red`" in out
    assert "`auditedUseCase`.`outcome`" in out
    assert "`locator.red`" not in out


def test_normalize_struct_paths_leaves_plain_and_simple_alone():
    """Unquoted dotted paths and simple backticked identifiers are untouched."""
    sql = "SELECT route.air, `locator`, amount FROM `src_tables`.`record_table_4`"
    out = _normalize_struct_paths(sql)
    assert "route.air" in out
    assert "`locator`" in out
    assert "amount" in out
    # A qualified table name split across two backticked segments is already correct
    # (each segment quoted separately) and must stay intact.
    assert "`src_tables`.`record_table_4`" in out


def test_render_schema_leaf_cap():
    """Per-table leaves are capped so a very wide struct can't blow up the prompt."""
    fields = ", ".join(f"f{i}:INT" for i in range(200))
    rows = [
        {
            "table_name": "wide",
            "column_name": "big",
            "full_data_type": f"STRUCT<{fields}>",
        }
    ]
    rendered = _render_schema(rows)
    # 60-leaf cap => at most 60 "fN INT" entries rendered.
    assert rendered.count(" INT") <= 60


# ─────────────────────────────────────────────────────────────────────────────
# The leaf cap must not decide which declarations survive
#
# The rendering the cap produces is also what `map_entities` validates `entity_bindings`
# against. Filling in ordinal order let a wide table decide which pack declarations were
# reported as STALE BINDING; declared columns are now sorted to the front.
# ─────────────────────────────────────────────────────────────────────────────


def _wide_rows(n: int = 200) -> list:
    """One table whose flattened width exceeds the cap, with a leaf at the far end."""
    fields = ", ".join(f"f{i}:INT" for i in range(n))
    return [
        {
            "table_name": "wide",
            "column_name": "big",
            "full_data_type": f"STRUCT<{fields}>",
        },
        {"table_name": "wide", "column_name": "subject", "full_data_type": "STRING"},
    ]


def _leaf_paths(rendered: str) -> list:
    body = rendered[rendered.index("(") + 1 : -1]
    return [entry.split(" ")[0] for entry in body.split(", ")]


def test_render_schema_cap_hides_an_undeclared_far_leaf():
    """The defect itself: past the cap, a real column is rendered nowhere."""
    assert "subject" not in _leaf_paths(_render_schema(_wide_rows()))


def test_render_schema_keeps_a_declared_leaf_the_cap_would_have_cut():
    """A declared leaf survives the cap however late it sits in ordinal order.

    This is the whole fix: `map_entities` checks the pack's binding against this string, so a
    declared column missing from it comes back as a stale binding and the source loses its
    deterministic filter.
    """
    paths = _leaf_paths(_render_schema(_wide_rows(), ["subject"]))
    assert "subject" in paths
    assert len(paths) <= 60, "the cap must still bound the prompt"


def test_render_schema_declaration_may_sit_below_the_rendered_leaf():
    """Flattening stops at an ARRAY<STRUCT<>>; a pack declares the field INSIDE it.

    Matching on equality alone would keep neither the array nor the declaration, which is
    exactly the live case — the subject column sat inside a terminal array.
    """
    rows = _wide_rows()
    rows.append(
        {
            "table_name": "wide",
            "column_name": "items",
            "full_data_type": "ARRAY<STRUCT<card:STRING>>",
        }
    )
    paths = _leaf_paths(_render_schema(rows, ["items.card"]))
    assert "items" in paths, "the array is the only renderable ancestor of the declaration"


def test_render_schema_declaration_may_sit_above_the_rendered_leaves():
    """A pack may name an enclosing struct while the scalars are what get rendered."""
    rows = _wide_rows()
    rows.append(
        {
            "table_name": "wide",
            "column_name": "actor",
            "full_data_type": "STRUCT<office:STRING, sign:STRING>",
        }
    )
    paths = _leaf_paths(_render_schema(rows, ["actor"]))
    assert "actor.office" in paths and "actor.sign" in paths


def test_render_schema_is_byte_identical_under_the_cap():
    """Under the cap every leaf is rendered either way, so the order must not change.

    Reordering there would alter the prompt for every source on this backend at once,
    including the ones nothing is wrong with. Confining the reorder to the truncating case is
    what makes the fix safe to land without re-measuring every working source.
    """
    rows = [
        {"table_name": "t", "column_name": "a", "full_data_type": "STRING"},
        {"table_name": "t", "column_name": "b", "full_data_type": "STRUCT<c:STRING,d:INT>"},
    ]
    assert _render_schema(rows) == _render_schema(rows, ["b.d"])


def test_render_schema_declaring_nothing_leaves_ordinal_order_untouched():
    """No declaration -> the previous behaviour exactly, cap and order alike."""
    rows = _wide_rows()
    assert _render_schema(rows) == _render_schema(rows, [])


def test_render_schema_never_invents_a_declared_path():
    """Ordering only. A declaration naming nothing real must not become a rendered leaf.

    Absence stays the renderer's answer to give — that is what makes the stale-binding
    warning downstream mean something.
    """
    rendered = _render_schema(_wide_rows(), ["nosuch.column"])
    assert "nosuch" not in rendered


def test_declared_leaf_paths_reads_bindings_and_projection(monkeypatch):
    """The pack-reading seam: both declaration sites, every value form, order preserved."""
    from src.retrievers.field_mapping import declared_leaf_paths

    class _Src:
        entity_bindings = {
            "card": ["items.card"],
            "user": {"sign": ["actor.sign"], "login": ["actor.login"]},
        }
        projection = ["items.card", "when"]

    class _Pack:
        def source(self, name):
            return _Src() if name == "s" else None

    paths = declared_leaf_paths(_Pack(), "s")
    # Every form's column is present: rendering is not the seam that routes a form, and
    # render_filters needs them all rendered to be able to route at all.
    assert set(paths) == {"items.card", "actor.sign", "actor.login", "when"}
    assert paths.count("items.card") == 1, "the two declaration sites must not duplicate"
    assert declared_leaf_paths(None, "s") == []
    assert declared_leaf_paths(_Pack(), "") == []
    assert declared_leaf_paths(_Pack(), "absent") == []


# ─────────────────────────────────────────────────────────────────────────────
# A projection entry may be an EXPRESSION, with two different names
#
# The schema renderer needs the leaf an expression READS; everything downstream needs the
# name the value ARRIVES under. Getting either wrong fails silently: unknown conditions.
# ─────────────────────────────────────────────────────────────────────────────


def test_projection_grammar_separates_what_an_entry_reads_from_what_it_is_called():
    """The two questions, over a bare path, an aliased path and an aliased expression."""
    from src.utils.projection import (expand_projection, projection_alias,
                                      projection_names, projection_sources,
                                      split_projection)

    # A bare path is its own name AND its own source: unchanged behaviour.
    assert split_projection("group.item") == ("group.item", "")
    assert projection_names("group.item") == ["group.item"]
    assert projection_sources("group.item") == ["group.item"]
    assert projection_alias("group.item") == ""

    # An aliased leaf arrives under the alias and under nothing else.
    assert projection_names("group.item.tag AS item_tag") == ["item_tag"]
    assert projection_sources("group.item.tag AS item_tag") == ["group.item.tag"]

    # An aliased expression: the lambda's own parameter is NOT a path on the table, so a
    # renderer must not try to keep it and a coverage check must not report it.
    entry = "filter(group.item, s -> s.tag LIKE 'A%').tag AS kept_tag"
    assert projection_names(entry) == ["kept_tag"]
    assert projection_sources(entry) == ["group.item"]

    # `AS` inside a call names a TYPE, not the entry — an alias must be the last thing.
    assert projection_alias("CAST(counters.n AS INT)") == ""
    assert projection_sources("CAST(counters.n AS INT)") == ["counters.n"]
    assert projection_alias("CAST(counters.n AS INT) AS counters_n") == "counters_n"

    # An UNALIASED expression names nothing, and says so. Returning its own text would
    # hand every consumer a name that matches no column while looking like an answer.
    assert projection_names("CAST(counters.n AS INT)") == []

    # An `AS` inside a string literal is not an alias either.
    assert projection_alias("concat(a.b, ' AS x') AS joined") == "joined"
    assert projection_sources("concat(a.b, ' AS x') AS joined") == ["a.b"]

    # The renderer's view: every entry's sources, flattened and de-duplicated.
    assert expand_projection(
        ["group.item.tag AS item_tag", entry, "when", None]
    ) == ["group.item.tag", "group.item", "when"]


def test_declared_leaves_protect_the_leaf_an_expression_reads_not_its_text(monkeypatch):
    """The renderer keeps `group.item`, which is the column the cap would otherwise cut."""
    from src.retrievers.field_mapping import declared_leaf_paths

    class _Src:
        entity_bindings = {}
        projection = ["filter(group.item, s -> s.tag LIKE 'A%').tag AS kept_tag"]

    class _Pack:
        def source(self, name):
            return _Src() if name == "s" else None

    paths = declared_leaf_paths(_Pack(), "s")
    assert paths == ["group.item"], "the expression's text is not a column"


def test_a_declared_projection_name_absent_from_the_sql_is_reported(caplog):
    """The one pack declaration this route cannot enforce must not be able to fail quietly.

    There is no rewrite that could add a column a SELECT omitted, so the engine's whole
    contribution is refusing to let the omission look like a source that answered with
    nothing. The test is "can anything downstream FIND it" — a leaf aliased to its last
    segment alone is IN the statement and still unreadable.
    """
    import logging

    def _mk(projection):
        return DatabricksRetriever(
            {
                "name": "transaction_logs",
                "workspace_url": "https://workspace/",
                "warehouse_id": "wh1",
                "api_key_env": "DATABRICKS_TOKEN",
                "poll_interval_seconds": 0,
                "projection": projection,
            },
            MagicMock(),
        )

    r = _mk(["group.item.tag AS item_tag", "event.stamp"])

    def _missing(sql):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            r._report_unreadable_projection(sql)
        return " ".join(caplog.messages)

    # Both readable: the alias verbatim, and the bare path in its underscore-flattened form
    # — the two spellings path resolution actually tries.
    assert not _missing(
        "SELECT group.item.tag AS item_tag, event.stamp AS event_stamp FROM t"
    )

    # The generator renamed the alias. It is IN the statement and findable by nobody.
    reported = _missing("SELECT group.item.tag AS tag, event.stamp AS event_stamp FROM t")
    assert "item_tag" in reported and "unknown" in reported
    assert "event.stamp" not in reported, "a readable name is not reported"

    # The bare path aliased to its LAST SEGMENT only: present, unreadable, reported.
    assert "event.stamp" in _missing(
        "SELECT group.item.tag AS item_tag, event.stamp AS stamp FROM t"
    )

    # No projection declared -> nothing to say, on any SQL.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _mk([])._report_unreadable_projection("SELECT 1")
    assert not caplog.messages


def test_an_alias_resolves_back_to_the_one_path_it_renames_or_to_nothing():
    """The THIRD question about the same entry, and the only one with two right answers.

    A reader holding both a condition's path and the target's own inventory has to translate
    between them, and the two ways of not translating are the two failures: report every read
    under an alias as a path that exists nowhere, or accept the whole subtree unchecked. So an
    entry pinning its alias to ONE path is resolved to its leaf, and an entry that pins it to
    no single path says so — `""`, which is unknowable and not absent.
    """
    from src.utils.projection import projection_renames

    m = projection_renames(
        [
            "filter(group.item, s -> s.tag LIKE 'A%').detail AS kept_detail",
            "group.item.tag AS item_tag",
            "CAST(counters.n AS INT) AS n",
            "element_at(group.item, 1) AS first_item",
            "concat(a.b, c.d) AS joined",
            "group.item",  # no alias: nothing to translate
            None,
        ]
    )
    # A call plus a member accessor: the accessor belongs to the path, so a read of
    # `kept_detail.x` is a read of `group.item.detail.x`.
    assert m["kept_detail"] == "group.item.detail"
    assert m["item_tag"] == "group.item.tag"
    assert m["n"] == "counters.n"
    assert m["first_item"] == "group.item"
    # Two paths in one entry: the alias names a value neither of them holds alone.
    assert m["joined"] == ""
    assert "group.item" not in m, "an unaliased entry renames nothing"


# ─────────────────────────────────────────────────────────────────────────────
# Nested-type reconciliation: the catalog's STRUCT type can be stale
#
# `information_schema.columns.full_data_type` is not guaranteed to be the CURRENT
# type. Measured on a live external table: it served 31 of 35 children in a valid,
# re-closed, marker-free type string, and the 4 it omitted were the only populated
# ones. A stale type parses, flattens and validates like a current one, so the
# disagreement must be sought out via the truncation marker.
# ─────────────────────────────────────────────────────────────────────────────
_PARTIAL = "struct<a:string,b:string,... 2 more fields>"


def test_type_is_partial_detects_the_truncation_marker():
    """The marker is the only machine-readable signal a type string is incomplete."""
    assert type_is_partial(_PARTIAL)
    assert hidden_field_count(_PARTIAL) == 2
    # A stale-but-complete type carries NO marker. That is the dangerous case, and
    # asserting it here records that this predicate cannot detect it.
    assert not type_is_partial("struct<a:string,b:string>")
    assert hidden_field_count("struct<a:string,b:string>") == 0
    assert not type_is_partial("")
    assert type_is_partial("struct<a:string,... 1 more field>")  # singular


def test_struct_child_names_skips_the_marker_and_ignores_non_structs():
    assert struct_child_names(_PARTIAL) == ["a", "b"]
    assert struct_child_names("STRUCT<x:INT, y:STRUCT<z:STRING>>") == ["x", "y"]
    # Arrays and maps are terminal leaves here, never dot-accessed.
    assert struct_child_names("ARRAY<STRUCT<f:INT>>") == []
    assert struct_child_names("string") == []
    assert struct_child_names("") == []


def test_struct_type_from_children_round_trips():
    rebuilt = struct_type_from_children([("a", "string"), ("b", "struct<c:int>")])
    assert struct_child_names(rebuilt) == ["a", "b"]
    assert _flatten_struct("v", rebuilt) == [("v.a", "string"), ("v.b.c", "int")]
    assert struct_type_from_children([]) == ""


def test_live_column_types_stops_at_the_partition_section():
    """DESCRIBE TABLE repeats the partition columns under a ``#`` header.

    Reading past it would take a table property's value for a column's type — and
    the repeated partition column would overwrite the real entry with whatever the
    second section spells.
    """
    described = [
        {"col_name": "value", "data_type": "struct<a:string>"},
        {"col_name": "date", "data_type": "date"},
        {"col_name": "", "data_type": ""},
        {"col_name": "# Partition Information", "data_type": ""},
        {"col_name": "# col_name", "data_type": "data_type"},
        {"col_name": "date", "data_type": "SHOULD NOT BE READ"},
    ]
    assert _live_column_types(described) == {
        "value": "struct<a:string>",
        "date": "date",
    }
    assert _live_column_types([]) == {}


def _reconciling_retriever(catalog_rows, live_rows, query_rows=None, llm=None):
    """A Databricks retriever whose three metadata statements are scripted.

    ``calls`` records every SQL string so a test can assert on the *cost* of
    reconciliation as well as its result — a bound that fires silently would read as
    "every type was verified".
    """
    config = {
        "name": "admin_events",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "catalog": "main",
        "schema": "audit",
    }
    retriever = DatabricksRetriever(config, llm or MagicMock())
    calls = []

    async def fake_execute(sql):
        calls.append(sql)
        if "information_schema.columns" in sql:
            return [dict(r) for r in catalog_rows]
        if sql.startswith("DESCRIBE TABLE"):
            table = sql.split()[-1]
            rows = live_rows.get(table)
            if rows is None:
                raise RuntimeError(f"no live schema for {table}")
            return [dict(r) for r in rows]
        if sql.startswith("DESCRIBE QUERY"):
            path = sql.split("SELECT ", 1)[1].split(".* FROM", 1)[0]
            return [dict(r) for r in (query_rows or {}).get(path, [])]
        return []

    retriever._execute_sql = AsyncMock(side_effect=fake_execute)
    return retriever, calls


@pytest.mark.asyncio
async def test_reconcile_adopts_the_live_type_when_the_catalog_is_stale():
    """A complete-looking catalog type that DISAGREES with the live one loses.

    This is the measured defect: the catalog's children are real columns that hold
    NULL on every row, so the generator writes queries against them, gets 0 rows,
    and every decisive condition reads ``unknown`` while retrieval reports success.
    """
    retriever, calls = _reconciling_retriever(
        catalog_rows=[
            {
                "table_name": "events",
                "column_name": "value",
                "data_type": "struct",
                # Stale: the producer added `actor` after the table was created.
                "full_data_type": "struct<legacyUserId:string>",
            }
        ],
        live_rows={
            "main.audit.events": [
                {
                    "col_name": "value",
                    "data_type": "struct<legacyUserId:string,actor:struct<id:string>>",
                }
            ]
        },
    )

    rows = await retriever._discover_columns("audit")

    assert "actor" in struct_child_names(rows[0]["full_data_type"])
    assert "value.actor.id" in _render_schema(rows)
    assert any(c.startswith("DESCRIBE TABLE") for c in calls)


@pytest.mark.asyncio
async def test_reconcile_expands_a_live_type_that_is_itself_truncated():
    """The two failure modes are complementary, so both have to be handled.

    ``DESCRIBE TABLE`` is CURRENT but renders a wide struct as the first fields plus
    ``... N more fields``. The catalog is complete-looking but may be stale. When the
    live type is visibly partial AND its arithmetic exceeds what the catalog holds,
    the truth is assembled field by field with ``DESCRIBE QUERY`` — which returns one
    row per child and so never truncates a field LIST.
    """
    retriever, calls = _reconciling_retriever(
        catalog_rows=[
            {
                "table_name": "events",
                "column_name": "value",
                "data_type": "struct",
                "full_data_type": "struct<legacyUserId:string,legacyOffice:string>",
            }
        ],
        live_rows={
            "main.audit.events": [
                {
                    "col_name": "value",
                    # 2 named + 2 hidden = 4 children exist; the catalog holds 2.
                    "data_type": (
                        "struct<legacyUserId:string,legacyOffice:string,"
                        "... 2 more fields>"
                    ),
                }
            ]
        },
        query_rows={
            "value": [
                {"col_name": "legacyUserId", "data_type": "string"},
                {"col_name": "legacyOffice", "data_type": "string"},
                # A child that is ITSELF partially rendered -> recurse into it.
                {"col_name": "actor", "data_type": "struct<id:string,... 1 more field>"},
                {"col_name": "ts", "data_type": "timestamp"},
            ],
            "value.actor": [
                {"col_name": "id", "data_type": "string"},
                {"col_name": "organization", "data_type": "string"},
            ],
        },
    )

    rows = await retriever._discover_columns("audit")

    expanded = rows[0]["full_data_type"]
    assert struct_child_names(expanded) == [
        "legacyUserId",
        "legacyOffice",
        "actor",
        "ts",
    ]
    # The recursion landed: `organization` is only reachable via DESCRIBE QUERY on
    # the child, which is the second statement the fix is allowed to spend.
    rendered = _render_schema(rows)
    assert "value.actor.id" in rendered
    assert "value.actor.organization" in rendered
    assert "DESCRIBE QUERY SELECT value.actor.* FROM main.audit.events" in calls
    # No marker survives into the schema handed to the generator.
    assert "more field" not in rendered


@pytest.mark.asyncio
async def test_reconcile_leaves_an_agreeing_catalog_type_untouched():
    """Agreement costs ONE statement per table with a nested column, and no rewrite.

    A reference table of scalars costs nothing at all — asserted by the existing
    discovery test, which scripts a single ``_execute_sql`` return value.
    """
    retriever, calls = _reconciling_retriever(
        catalog_rows=[
            {
                "table_name": "events",
                "column_name": "value",
                "data_type": "struct",
                "full_data_type": "struct<a:string,b:string>",
            },
            {
                "table_name": "events",
                "column_name": "date",
                "data_type": "date",
            },
        ],
        live_rows={
            "main.audit.events": [
                {"col_name": "value", "data_type": "struct<a:string,b:string>"},
                {"col_name": "date", "data_type": "date"},
            ]
        },
    )

    rows = await retriever._discover_columns("audit")

    assert rows[0]["full_data_type"] == "struct<a:string,b:string>"
    assert len([c for c in calls if c.startswith("DESCRIBE")]) == 1
    assert rows[0]["table_name"] == "audit.events"  # qualification still happens


@pytest.mark.asyncio
async def test_reconcile_degrades_to_the_catalog_when_describe_fails():
    """A reconciliation that cannot run degrades to today's behaviour, not to none."""
    retriever, _calls = _reconciling_retriever(
        catalog_rows=[
            {
                "table_name": "events",
                "column_name": "value",
                "data_type": "struct",
                "full_data_type": "struct<a:string>",
            }
        ],
        live_rows={},  # every DESCRIBE TABLE raises
    )

    rows = await retriever._discover_columns("audit")

    assert rows[0]["full_data_type"] == "struct<a:string>"
    assert "value.a" in _render_schema(rows)


@pytest.mark.asyncio
async def test_reconcile_bound_names_every_unverified_table(caplog):
    """When the statement bound bites, the skipped tables are NAMED in a warning.

    A silent cap here would read as "every type was verified", which is the same
    class of defect the reconciliation exists to remove.
    """
    catalog_rows = [
        {
            "table_name": f"t{i:03d}",
            "column_name": "value",
            "data_type": "struct",
            "full_data_type": "struct<a:string>",
        }
        for i in range(45)
    ]
    live_rows = {
        f"main.audit.t{i:03d}": [{"col_name": "value", "data_type": "struct<a:string>"}]
        for i in range(45)
    }
    retriever, calls = _reconciling_retriever(catalog_rows, live_rows)

    with caplog.at_level("WARNING"):
        await retriever._discover_columns("audit")

    assert len([c for c in calls if c.startswith("DESCRIBE TABLE")]) == 40
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "t044" in text and "t040" in text
    assert "t000" not in text


# ─────────────────────────────────────────────────────────────────────────────
# A catalog with no information_schema — the legacy Hive metastore
#
# Discovery was information_schema-only, and its failure returned an EMPTY schema. That is
# not a degraded run: `_in_schema` then rejects every path, so the pack's measured
# `entity_bindings` are all dropped as STALE, `render_filters` emits no predicate, and the
# generated query keeps its window bound alone — an arbitrary page of other identities'
# rows, which reads downstream exactly like the subject's own.
# ─────────────────────────────────────────────────────────────────────────────
_DESCRIBE_HIVE = [
    {"col_name": "timestamp", "data_type": "timestamp", "comment": None},
    {"col_name": "date", "data_type": "date", "comment": None},
    {"col_name": "login", "data_type": "string", "comment": None},
    {"col_name": "robot", "data_type": "boolean", "comment": None},
    {"col_name": "# Partition Information", "data_type": "", "comment": ""},
    {"col_name": "# col_name", "data_type": "data_type", "comment": "comment"},
    {"col_name": "date", "data_type": "date", "comment": None},
    {"col_name": "", "data_type": "", "comment": ""},
    {"col_name": "# Detailed Table Information", "data_type": "", "comment": ""},
    {"col_name": "Catalog", "data_type": "hive_metastore", "comment": ""},
    {"col_name": "Type", "data_type": "EXTERNAL", "comment": ""},
    # A value that would parse as a type if the section break were missed.
    {"col_name": "Provider", "data_type": "delta", "comment": ""},
]


def test_columns_from_describe_reads_the_three_regions():
    """A partition column is listed TWICE and must not become a second leaf."""
    rows = _columns_from_describe(_DESCRIBE_HIVE, "authhistorylog")

    assert [r["column_name"] for r in rows] == ["timestamp", "date", "login", "robot"]
    assert all(r["table_name"] == "authhistorylog" for r in rows)
    # The partition is marked on the column's own row, so `_partitions_from_columns` sees it.
    assert [r["partition_index"] for r in rows] == [None, 0, None, None]
    assert {"Catalog", "Type", "Provider"}.isdisjoint(
        {r["column_name"] for r in rows}
    )


@pytest.mark.asyncio
async def test_discovery_falls_back_to_describe_when_there_is_no_information_schema():
    config = {
        "name": "auth_history",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "catalog": "hive_metastore",
        "schema": "legacy_audit",
        "tables": ["authhistorylog"],
    }
    retriever = DatabricksRetriever(config, MagicMock())
    calls = []

    async def fake_execute(sql):
        calls.append(sql)
        if "information_schema" in sql:
            raise RuntimeError(
                "[TABLE_OR_VIEW_NOT_FOUND] The table or view "
                "`hive_metastore`.`information_schema`.`tables` cannot be found."
            )
        if sql.startswith("DESCRIBE TABLE"):
            return [dict(r) for r in _DESCRIBE_HIVE]
        return []

    retriever._execute_sql = AsyncMock(side_effect=fake_execute)

    rows = await retriever._discover_columns("legacy_audit")

    assert [r["column_name"] for r in rows] == ["timestamp", "date", "login", "robot"]
    assert any(c == "DESCRIBE TABLE hive_metastore.legacy_audit.authhistorylog" for c in calls)
    # The declared table list is honoured, so no table listing is needed.
    assert not any(c.startswith("SHOW TABLES") for c in calls)


@pytest.mark.asyncio
async def test_the_describe_fallback_still_yields_the_partition_bound():
    """An unbounded scan of a partitioned table is the same silent failure one step on:
    it times out, returns nothing, and every condition reading it goes `unknown`."""
    config = {
        "name": "auth_history",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "catalog": "hive_metastore",
        "schema": "legacy_audit",
        "tables": ["authhistorylog"],
        "warehouse_warmup": False,
    }
    retriever = DatabricksRetriever(config, MagicMock())

    async def fake_execute(sql):
        if "information_schema" in sql:
            raise RuntimeError("no information_schema in this catalog")
        if sql.startswith("DESCRIBE TABLE"):
            return [dict(r) for r in _DESCRIBE_HIVE]
        return []

    retriever._execute_sql = AsyncMock(side_effect=fake_execute)

    schema = await retriever._get_field_schema()

    assert "login" in schema
    assert [p["name"] for p in retriever._discovered_partitions] == ["date"]


@pytest.mark.asyncio
async def test_the_fallback_lists_tables_when_the_pack_declares_none():
    config = {
        "name": "legacy",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "catalog": "hive_metastore",
        "schema": "legacy_audit",
    }
    retriever = DatabricksRetriever(config, MagicMock())
    calls = []

    async def fake_execute(sql):
        calls.append(sql)
        if "information_schema" in sql:
            raise RuntimeError("no information_schema in this catalog")
        if sql.startswith("SHOW TABLES"):
            return [
                {"database": "legacy_audit", "tableName": "authhistorylog"},
                {"database": "legacy_audit", "tableName": "authhistorylog_lh"},
            ]
        if sql.startswith("DESCRIBE TABLE"):
            return [dict(r) for r in _DESCRIBE_HIVE]
        return []

    retriever._execute_sql = AsyncMock(side_effect=fake_execute)

    rows = await retriever._discover_columns("legacy_audit")

    # Schema-qualified, exactly as the information_schema path returns them: `_in_schema`
    # reads one shape and a listed table that arrives bare is a table it cannot match.
    assert {r["table_name"] for r in rows} == {
        "legacy_audit.authhistorylog",
        "legacy_audit.authhistorylog_lh",
    }
    assert sum(1 for c in calls if c.startswith("DESCRIBE TABLE")) == 2


@pytest.mark.asyncio
async def test_a_readable_information_schema_is_never_second_guessed():
    """The fallback is for a catalog that HAS none — it must not add a statement per table
    to every UC source, nor let a DESCRIBE override what the catalog already answered."""
    retriever, calls = _reconciling_retriever(
        catalog_rows=[
            {
                "table_name": "events",
                "column_name": "login",
                "data_type": "string",
                "full_data_type": "string",
            }
        ],
        live_rows={},
    )

    rows = await retriever._discover_columns("audit")

    assert [r["column_name"] for r in rows] == ["login"]
    assert not any(c.startswith("DESCRIBE TABLE") for c in calls)
    assert not any(c.startswith("SHOW TABLES") for c in calls)


# ─────────────────────────────────────────────────────────────────────────────
# Graceful skip of placeholder (template) creds
# ─────────────────────────────────────────────────────────────────────────────
def test_is_placeholder():
    assert _is_placeholder("https://<elk-host>:9200")
    assert _is_placeholder("<FILL: snowflake account>")
    assert not _is_placeholder("https://real-host:9200")
    assert not _is_placeholder("acct1.eu-west-1")
    assert not _is_placeholder(None)


def test_merge_endpoint_skips_placeholder_creds():
    """ES/Snowflake/REST branches skip when the key field is an unfilled placeholder."""
    from types import SimpleNamespace

    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)
    backends = {
        "elasticsearch": {"secondary-prd": {"url": "https://<elk-host>:9200"}},
        "snowflake": {"acct": {"account": "<FILL: account>"}},
        "rest": {"servicenow": {"base_url": "https://<sn-host>"}},
    }
    engine.config = {"backends": backends}

    es_src = SimpleNamespace(
        name="scheme_alerts",
        endpoints={
            "kind": "elasticsearch",
            "cluster": "secondary-prd",
            "indices": ["i"],
        },
    )
    sf_src = SimpleNamespace(
        name="ff", endpoints={"kind": "snowflake", "account": "acct"}
    )
    rest_src = SimpleNamespace(
        name="sn", endpoints={"kind": "rest", "service": "servicenow"}
    )

    assert engine._merge_endpoint(es_src, "elasticsearch", backends) is None
    assert engine._merge_endpoint(sf_src, "snowflake", backends) is None
    assert engine._merge_endpoint(rest_src, "rest", backends) is None


def test_merge_endpoint_elasticsearch_creds_present_builds():
    """A real (reachable-gateway) url + basic-auth creds yields a full ES config."""
    from types import SimpleNamespace

    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)
    backends = {
        "elasticsearch": {
            "secondary-prd": {
                "url": "https://kibana.example-gateway.internal",
                "username": "alice",
                "password": "s3cret",
                "max_results": 250,
            }
        }
    }
    engine.config = {"backends": backends}
    es_src = SimpleNamespace(
        name="scheme_alerts",
        endpoints={
            "kind": "elasticsearch",
            "cluster": "secondary-prd",
            "indices": ["session.cloud.correlatedalerts", "session.scheme.alerts"],
        },
    )

    merged = engine._merge_endpoint(es_src, "elasticsearch", backends)
    assert merged is not None
    assert merged["type"] == "elasticsearch"
    assert merged["url"] == "https://kibana.example-gateway.internal"
    assert merged["username"] == "alice"
    assert merged["password"] == "s3cret"
    # ES|QL FROM accepts a comma-separated index list.
    assert merged["index"] == ("session.cloud.correlatedalerts,session.scheme.alerts")
    assert merged["max_results"] == 250


def test_merge_endpoint_elasticsearch_missing_creds_skips():
    """Real url but empty username/password (unset ${VAR}) → skip-with-log (None)."""
    from types import SimpleNamespace

    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)
    backends = {
        "elasticsearch": {
            "secondary-prd": {
                "url": "https://kibana.example-gateway.internal",
                "username": "",  # unset ${ELK_USER} expands to ""
                "password": "",
            }
        }
    }
    engine.config = {"backends": backends}
    es_src = SimpleNamespace(
        name="siem_alerts",
        endpoints={
            "kind": "elasticsearch",
            "cluster": "secondary-prd",
            "indices": ["i"],
        },
    )
    assert engine._merge_endpoint(es_src, "elasticsearch", backends) is None


@pytest.mark.asyncio
async def test_databricks_projection_reaches_sql_prompt():
    """A source's `projection` is injected as a REQUIRED PROJECTION block in the SQL
    prompt, overriding the scalar-preference bias so rich SCHEME leaves are returned."""
    from src.models.pydantic_models import FieldMapping

    config = {
        "name": "record_lake",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "field_schema": "record_table_4(locator string)",
        "projection": [
            "locator.red",
            "creator.sign.red",
            "element_counters.AUX",
            "route.air",
        ],
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        side_effect=[
            FieldMapping(mappings=[]),
            SqlQuery(query="SELECT locator.red FROM record_table_4"),
        ]
    )
    retriever = DatabricksRetriever(config, llm)
    retriever._execute_sql = AsyncMock(return_value=[])

    query = RetrievalQuery(
        target_log_source="record_lake",
        natural_language_query="record version",
        date_from="2026-07-17",
        date_to="2026-07-18",
    )
    await retriever.retrieve(query)

    system_msg = llm.structured_output.call_args_list[1].args[0][0]["content"]
    assert "REQUIRED PROJECTION" in system_msg
    assert "locator.red" in system_msg and "element_counters.AUX" in system_msg
    # array leaf explode guidance present, and MUST require OUTER (empty arrays must not
    # drop a bare record) + unique per-leaf aliases (avoid the `.red` name collision).
    assert "explode" in system_msg.lower()
    assert "OUTER" in system_msg
    assert "ALIAS" in system_msg.upper() and "collide" in system_msg.lower()


@pytest.mark.asyncio
async def test_databricks_no_projection_omits_block():
    """A source WITHOUT a projection keeps the unchanged prompt (no REQUIRED PROJECTION)."""
    from src.models.pydantic_models import FieldMapping

    config = {
        "name": "auth_events",
        "workspace_url": "https://workspace",
        "warehouse_id": "wh1",
        "api_key_env": "DATABRICKS_TOKEN",
        "field_schema": "evt(id string)",
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        side_effect=[FieldMapping(mappings=[]), SqlQuery(query="SELECT id FROM evt")]
    )
    retriever = DatabricksRetriever(config, llm)
    retriever._execute_sql = AsyncMock(return_value=[])
    await retriever.retrieve(
        RetrievalQuery(
            target_log_source="auth_events",
            natural_language_query="events",
            date_from="2026-07-17",
            date_to="2026-07-18",
        )
    )
    system_msg = llm.structured_output.call_args_list[1].args[0][0]["content"]
    assert "REQUIRED PROJECTION" not in system_msg


def test_merge_endpoint_threads_projection():
    """_merge_endpoint copies a databricks pack source's `projection` into the config."""
    from src.log_retrieval import LogRetrievalEngine

    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)
    engine._auth_by_host = {}
    engine.auth = None

    class _Src:
        name = "record_lake"
        endpoints = {
            "kind": "databricks_uc",
            "workspace": "primary_we",
            "catalog": "cat",
            "schema": "sch",
            "tables": ["record_table_4"],
        }
        query_hints = ""
        default_filters = {}
        projection = ["locator.red", "creator.sign.red"]

    backends = {
        "databricks": {
            "primary_we": {
                "warehouse_id": "wh1",
                "workspace_url": "https://w",
                "api_key_env": "DATABRICKS_TOKEN",
            }
        }
    }
    merged = engine._merge_endpoint(_Src(), "databricks", backends)
    assert merged["projection"] == ["locator.red", "creator.sign.red"]


# --- pack query guarantees are backend-agnostic ----------------------------------------
#
# `never_filter` and `require_all_entities` are properties of the source, not the backend.
# These tests pin the guarantee on every backend that generates its own query.


def test_strip_evidence_predicates_removes_conjoined_predicate():
    from src.retrievers.query_guards import strip_evidence_predicates

    out = strip_evidence_predicates(
        "SELECT a FROM t WHERE d = '1' AND userInfo.robot = false",
        ["value.payload.userInfo.robot"],
    )
    assert "robot" not in out.split("WHERE", 1)[1]
    assert "d = '1'" in out


def test_strip_evidence_predicates_handles_esql_pipe_stage():
    """ES|QL puts each filter in its own `| WHERE` stage — dropping the predicate must
    drop the whole stage, not leave a dangling `| WHERE`."""
    from src.retrievers.query_guards import strip_evidence_predicates

    out = strip_evidence_predicates(
        'FROM idx | WHERE ts > "2026-01-01" | WHERE robot == false | LIMIT 10',
        ["userInfo.robot"],
        pipe_stages=True,
    )
    assert "robot" not in out
    assert "| LIMIT 10" in out
    assert "WHERE |" not in " ".join(out.split())


def test_enforce_conjunction_rewrites_esql_equality_or():
    """ES|QL uses `==`; the composite-key rewrite must recognise it too."""
    from src.retrievers.query_guards import enforce_conjunction

    out = enforce_conjunction(
        'FROM idx | WHERE orgUnitId == "A" OR sign == "B"', ["orgUnitId", "sign"]
    )
    assert 'orgUnitId == "A" AND sign == "B"' in out


# --- "was the key actually in the query that ran" --------------------------------------
#
# A keyed reference lookup answers by returning zero rows. `True` means "recognised in the
# published text", never "declared": a declared key that was not queried reads as a data gap.


def test_key_was_enforced_reads_the_conjunction_that_shipped():
    from src.retrievers.query_guards import key_was_enforced

    keyed = "SELECT * FROM reg WHERE org_unit = 'BBB1C03EF' AND sign = '0202YZ'"
    assert key_was_enforced(keyed, ["org_unit", "sign"]) is True
    # Qualified/quoted spellings of the same predicate are the same predicate.
    assert (
        key_was_enforced(
            "SELECT * FROM reg WHERE `r`.`org_unit` = 'D' AND r.sign IN ('0202YZ')",
            ["r.org_unit", "r.sign"],
        )
        is True
    )
    # ES|QL equality too, or the check is Databricks-only like the bug it mirrors.
    assert (
        key_was_enforced(
            'FROM reg | WHERE org_unit == "D" AND sign == "0202YZ"',
            ["org_unit", "sign"],
        )
        is True
    )


def test_key_was_enforced_says_NO_to_every_shape_it_cannot_vouch_for():
    """The False direction only leaves a question open; the True direction invents an
    answer. So each of these must be False, and none of them is a parse failure."""
    from src.retrievers.query_guards import key_was_enforced

    # A DISJUNCT is the whole point: `... org_unit = 'D' OR sign = '0202YZ'` returns other
    # units' rows, so an empty result says nothing about THIS pair.
    assert (
        key_was_enforced(
            "SELECT * FROM reg WHERE org_unit = 'D' OR sign = '0202YZ'",
            ["org_unit", "sign"],
        )
        is False
    )
    # Half the key: the guard declined to rewrite, so one member never constrained anything.
    assert (
        key_was_enforced("SELECT * FROM reg WHERE sign = '0202YZ'", ["org_unit", "sign"])
        is False
    )
    # A range is not an identity — a row outside it is excluded for the wrong reason.
    assert (
        key_was_enforced(
            "SELECT * FROM reg WHERE org_unit = 'D' AND sign > 'A'", ["org_unit", "sign"]
        )
        is False
    )
    # No declared key at all: an event log, or an incident that satisfied no candidate.
    assert key_was_enforced("SELECT * FROM reg WHERE org_unit = 'D'", []) is False
    assert key_was_enforced("", ["org_unit"]) is False


def test_key_was_enforced_dsl_answers_the_same_question_on_a_tree():
    """Both query families or it is not a shape — a reference lookup sits on either
    backend, and a seam wired for one makes the declaration a silent no-op on the other."""
    from src.retrievers.query_guards import key_was_enforced_dsl

    conjunctive = {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"org_unit": "BBB1C03EF"}},
                    {"term": {"sign": "0202YZ"}},
                ]
            }
        }
    }
    assert key_was_enforced_dsl(conjunctive, ["org_unit", "sign"]) is True
    # `should` IS the DSL's OR, wherever it sits, and one is enough to admit another
    # identity's rows.
    disjunctive = {
        "query": {
            "bool": {
                "should": [
                    {"term": {"org_unit": "BBB1C03EF"}},
                    {"term": {"sign": "0202YZ"}},
                ]
            }
        }
    }
    assert key_was_enforced_dsl(disjunctive, ["org_unit", "sign"]) is False
    # A `must` clause nested under a `should` is not conjunctive with the whole query.
    nested = {
        "query": {
            "bool": {
                "should": [
                    {
                        "bool": {
                            "must": [
                                {"term": {"org_unit": "D"}},
                                {"term": {"sign": "0202YZ"}},
                            ]
                        }
                    },
                    {"term": {"other": "x"}},
                ]
            }
        }
    }
    assert key_was_enforced_dsl(nested, ["org_unit", "sign"]) is False
    assert key_was_enforced_dsl({}, ["org_unit"]) is False


def test_a_disjunct_that_WIDENS_one_key_field_still_answers_the_absence_question():
    """Because this flag interprets an EMPTY result, a superset returning nothing is proof.

    `sign = 'X' OR sign LIKE 'X%'` finding no row proves `sign = 'X'` finds no row — the
    entailment runs the opposite way from "every row returned is about this identity", which
    is what the blanket OR-rejection was reading. Rejecting it made ONE incident answer
    differently on two runs: the automation register read PASS (the identity is not
    registered, which is the finding) and then UNKNOWN, solely because the generator widened
    the sign predicate the way that source's own `query_hints` instruct — a concatenated duty
    code has to match too. So the pack's advice and the engine's reading of the result were
    in direct contradiction, and which one won depended on the wording of the day.
    """
    from src.retrievers.query_guards import key_was_enforced

    fields = ["unitId", "sign"]
    assert (
        key_was_enforced(
            "SELECT * FROM reg WHERE unitId = 'BBB1C03EF' "
            "AND (sign = '0202YZ' OR sign LIKE '0202YZ%')",
            fields,
        )
        is True
    )
    # ES|QL spelling of the same widening.
    assert (
        key_was_enforced(
            'FROM reg | WHERE unitId == "D" AND (sign == "0202YZ" OR sign LIKE "0202YZ*")',
            fields,
        )
        is True
    )


def test_a_widening_disjunct_is_the_ONLY_admissible_one():
    """Every other OR shape still admits another identity's rows, so all of these are False.

    The exception is narrow on purpose: one key field, and one arm that is that field's own
    recognised equality, so the group provably CONTAINS the key predicate instead of
    replacing it. Each case below fails exactly one of those tests.
    """
    from src.retrievers.query_guards import key_was_enforced

    fields = ["unitId", "sign"]
    # Two DIFFERENT key fields OR-ed: the classic composite-key defect, returns other keys'
    # rows, so an empty result says nothing about this pair.
    assert (
        key_was_enforced("SELECT * FROM reg WHERE (unitId = 'D' OR sign = '0202YZ')", fields)
        is False
    )
    # An arm on a NON-key column widens the result set beyond the key — the rows returned
    # (or not) are no longer about this identity's key at all.
    assert (
        key_was_enforced(
            "SELECT * FROM reg WHERE unitId = 'D' "
            "AND (sign = '0202YZ' OR profile = 'ROBOT')",
            fields,
        )
        is False
    )
    # No equality arm: two patterns is a range-like search, not this identity's own lookup.
    assert (
        key_was_enforced(
            "SELECT * FROM reg WHERE unitId = 'D' "
            "AND (sign LIKE '0%' OR sign LIKE '1%')",
            fields,
        )
        is False
    )
    # A bare top-level OR is in no group at all, so nothing licenses it.
    assert (
        key_was_enforced(
            "SELECT * FROM reg WHERE unitId = 'D' AND sign = '0202YZ' OR 1 = 1", fields
        )
        is False
    )
    # An arm carrying its own parentheses is not read, and unread means refused — the
    # conservative direction, since a True here would invent an answer.
    assert (
        key_was_enforced(
            "SELECT * FROM reg WHERE unitId = 'D' "
            "AND ((sign = '0202YZ') OR sign LIKE '0202YZ%')",
            fields,
        )
        is False
    )
    # And the widening must not excuse a key field that was never constrained at all.
    assert (
        key_was_enforced(
            "SELECT * FROM reg WHERE (sign = '0202YZ' OR sign LIKE '0202YZ%')", fields
        )
        is False
    )
    # A group over a non-key field: every other check passes, but the group widens the result
    # on an axis the absence proof does not run along. Only the is-it-a-key-field test rejects
    # it, and the other assertions here cannot see this case.
    assert (
        key_was_enforced(
            "SELECT * FROM reg WHERE unitId = 'D' AND sign = '0202YZ' "
            "AND (profile = 'ROBOT' OR profile = 'BOT')",
            fields,
        )
        is False
    )


def test_the_dsl_admits_the_same_widening_and_nothing_more():
    """Both query families or the declaration is a silent no-op on half the backends.

    A `should` whose every clause constrains ONE key field is that field's widening; the
    same shape spanning two fields, or reaching a non-key field, still disqualifies.
    """
    from src.retrievers.query_guards import key_was_enforced_dsl

    fields = ["org_unit", "sign"]
    widened = {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"org_unit": "D"}},
                    {
                        "bool": {
                            "should": [
                                {"term": {"sign": "0202YZ"}},
                                {"prefix": {"sign": "0202YZ"}},
                            ]
                        }
                    },
                ]
            }
        }
    }
    assert key_was_enforced_dsl(widened, fields) is True
    # A non-key arm inside the `should` — the DSL spelling of the SQL case above.
    non_key = {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"org_unit": "D"}},
                    {
                        "bool": {
                            "should": [
                                {"term": {"sign": "0202YZ"}},
                                {"term": {"profile": "ROBOT"}},
                            ]
                        }
                    },
                ]
            }
        }
    }
    assert key_was_enforced_dsl(non_key, fields) is False
    # A widening of one field does not vouch for the OTHER key field.
    half = {
        "query": {
            "bool": {
                "filter": [
                    {
                        "bool": {
                            "should": [
                                {"term": {"sign": "0202YZ"}},
                                {"prefix": {"sign": "0202YZ"}},
                            ]
                        }
                    }
                ]
            }
        }
    }
    assert key_was_enforced_dsl(half, fields) is False


def test_publish_query_records_whether_the_key_survived_the_guards():
    """The flag is computed on the ONE seam every backend reaches with its final text, for
    the same reason the placeholder check lives there — and from the query, not the
    declaration, so a guard that dropped the predicate cannot report itself as keyed."""
    from src.retrievers.base import publish_query

    retriever = SimpleNamespace(config={"name": "reg"})
    retriever.last_conjunction_fields = ["org_unit", "sign"]
    publish_query(retriever, "SELECT * FROM reg WHERE org_unit = 'D' AND sign = 'S'")
    assert retriever.last_key_enforced is True

    # The declaration is identical; only the shipped text differs. This is the case that
    # must NOT report keyed — it is how a real UNKNOWN would become a fabricated PASS.
    retriever.last_conjunction_fields = ["org_unit", "sign"]
    publish_query(retriever, "SELECT * FROM reg WHERE org_unit = 'D' OR sign = 'S'")
    assert retriever.last_key_enforced is False

    # And the stash is CONSUMED, so a later publish that resolved no key cannot inherit
    # the previous query's answer — one retriever instance serves every job.
    publish_query(retriever, "SELECT * FROM reg WHERE org_unit = 'D' AND sign = 'S'")
    assert retriever.last_key_enforced is False


# --- an event log's identity shape: (synonym OR synonym) AND scope AND scope ----------
#
# Synonyms are two columns for one value (e.g. login/userId) and must stay OR-ed; scope
# fields are separate facts and must be AND-ed. A flat OR across all of them matches the
# whole organisation. The generator produced the flat OR regardless of prompt phrasing.


def test_identity_scope_ands_scopes_onto_the_synonym_group():
    from src.retrievers.query_guards import enforce_identity_scope

    sql = (
        "SELECT a FROM t\nWHERE date >= DATE'2026-07-03'\n  AND (\n"
        "    value.payload.userInfo.login = 'BSURNAME'\n"
        "    OR value.payload.userInfo.userId = 'BSURNAME'\n"
        "    OR value.payload.userInfo.sign = '6009JJ'\n"
        "    OR value.payload.userInfo.org_unit = 'HHH1J09ST'\n  )"
    )
    out = enforce_identity_scope(
        sql,
        ["value.payload.userInfo.org_unit", "value.payload.userInfo.sign"],
        ["value.payload.userInfo.login", "value.payload.userInfo.userId"],
        "auth_events",
    )
    flat = " ".join(out.split())
    # The synonyms stay OR-ed together...
    assert "login = 'BSURNAME' OR value.payload.userInfo.userId = 'BSURNAME'" in flat
    # ...and each scope is AND-ed on, never left as a disjunct.
    assert "AND value.payload.userInfo.sign = '6009JJ'" in flat
    assert "AND value.payload.userInfo.org_unit = 'HHH1J09ST'" in flat
    assert "OR value.payload.userInfo.org_unit" not in flat
    assert "OR value.payload.userInfo.sign" not in flat
    # The partition bound is untouched.
    assert "date >= DATE'2026-07-03'" in out


def test_identity_scope_leaves_a_pure_synonym_group_alone():
    """Two columns holding one value MUST stay OR-ed — that is not the bug."""
    from src.retrievers.query_guards import enforce_identity_scope

    sql = "WHERE d = 1 AND (a.login = 'X' OR a.userId = 'X')"
    assert (
        enforce_identity_scope(sql, ["a.org_unit", "a.sign"], ["a.login", "a.userId"])
        == sql
    )


def test_identity_scope_never_touches_an_unrelated_or():
    """A group whose disjuncts are not declared identity fields is left exactly alone."""
    from src.retrievers.query_guards import enforce_identity_scope

    sql = "WHERE (status = 'A' OR status = 'B')"
    assert (
        enforce_identity_scope(sql, ["a.org_unit"], ["a.login"]) == sql
    ), "restructured a non-identity OR-group"
    # Nor does it INVENT a scope the query never constrained.
    only = "WHERE a.login = 'X'"
    assert enforce_identity_scope(only, ["a.org_unit"], ["a.login"]) == only


def test_identity_scope_ands_an_all_scope_group():
    """org_unit OR sign with no synonym involved is still the everyone-in-the-org_unit bug."""
    from src.retrievers.query_guards import enforce_identity_scope

    out = enforce_identity_scope(
        "WHERE (a.sign = '6009JJ' OR a.org_unit = 'HHH1J09ST')",
        ["a.org_unit", "a.sign"],
        ["a.login"],
    )
    assert "AND" in out and "OR" not in out


# --- the same widening one nesting deep ------------------------------------------------
#
# A nested arm like `(login = 'X' OR userId = 'X') OR (office = 'Y' AND org = 'Z')` is
# classified by whether it constrains a synonym column. An arm that does not is dropped
# and its scope predicates are AND-ed on instead.


def test_identity_scope_reads_a_nested_widening_arm():
    """The live shape: an identity group OR-ed with a parenthesised scope conjunction."""
    from src.retrievers.query_guards import enforce_identity_scope

    sql = (
        "SELECT a FROM t\nWHERE date >= DATE'2025-04-13'\n  AND (\n"
        "    (a.login = 'X' OR a.userId = 'X')\n"
        "    OR (a.org_unit = 'Y' AND a.organization = 'Z')\n  )"
    )
    out = enforce_identity_scope(
        sql, ["a.org_unit", "a.sign"], ["a.login", "a.userId"], "auth_events"
    )
    flat = " ".join(out.split())
    assert "(a.login = 'X' OR a.userId = 'X')" in flat
    # The declared scope is lifted and AND-ed...
    assert "AND a.org_unit = 'Y'" in flat
    # ...while the whole-population column the pack deliberately did NOT declare is dropped
    # rather than lifted: lifting a column nobody measured would be this guard asserting a
    # filter of its own, which it must never do.
    assert "a.organization" not in flat
    # And the widening OR is gone, which is the entire point.
    assert "OR a.org_unit" not in flat and "OR (a.org_unit" not in flat
    assert "date >= DATE'2025-04-13'" in out


def test_identity_scope_reads_a_group_nested_TWO_deep():
    """The same shape one nesting deeper.

    When bare arms are OR-ed beside the correct conjunction, every disjunct after the first
    is a superset of it, making the conjunction decoration. The guard's group finder must be
    depth-aware: a regex tolerating only one level of parenthesis cannot match the outer
    group; the innermost synonym pair matches instead, has no scope beside it, and the rewrite
    correctly declines — indistinguishable from a guard with nothing to do. The assertion is
    on shape: no bare identity disjunct may survive beside the conjunction.
    """
    from src.retrievers.query_guards import enforce_identity_scope

    sql = (
        "SELECT a FROM t\nWHERE date >= DATE'2025-04-13'\n  AND (\n"
        "    (\n"
        "      (a.login = 'X' OR a.userId = 'X')\n"
        "      AND a.sign = 'S'\n"
        "      AND a.org_unit = 'O'\n"
        "    )\n"
        "    OR a.login = 'X'\n"
        "    OR a.userId = 'X'\n"
        "    OR a.sign = 'S'\n  )"
    )
    out = enforce_identity_scope(
        sql, ["a.org_unit", "a.sign"], ["a.login", "a.userId"], "auth_events"
    )
    flat = " ".join(out.split())
    assert out != sql, flat
    # Both scopes are AND-ed on, so the query is about ONE identity in ONE org_unit.
    assert "AND a.sign = 'S'" in flat, flat
    assert "AND a.org_unit = 'O'" in flat, flat
    # And no bare identity arm survives OR-ed beside them — that is what made the conjunction
    # vacuous. Every remaining OR is inside the synonym family.
    for arm in ("OR a.sign", "OR a.org_unit"):
        assert arm not in flat, flat
    # The mandatory partition bound is untouched, as always.
    assert "date >= DATE'2025-04-13'" in out
    # A superset check on the predicate itself: the rewritten group must not be satisfiable by
    # a row carrying only the sign, which is precisely what the live query returned 500 of.
    other = {"login": "someone-else", "userId": "someone-else", "sign": "S", "org_unit": "ELSEWHERE"}
    assert not _sql_group_matches(flat, other), flat
    subject = {"login": "X", "userId": "X", "sign": "S", "org_unit": "O"}
    assert _sql_group_matches(flat, subject), flat


def _sql_group_matches(flat_sql: str, row: dict) -> bool:
    """Evaluate the rewritten WHERE's identity part against one row, as Python booleans.

    Crude on purpose and used by ONE test: asserting on substrings says the text changed,
    which is not the property that matters. What matters is whether a row belonging to
    somebody else still satisfies the predicate — the defect was a semantic one (a group whose
    widest disjunct subsumed every other), and only evaluating it can show that it is gone.
    """
    import re as _re

    # The identity part is the outermost BALANCED group that mentions a synonym column. Sliced
    # on a substring instead, the expression comes out with an unmatched paren and the test
    # fails for a reason that has nothing to do with the guard.
    start = flat_sql.index("a.login")
    while flat_sql[start] != "(":
        start -= 1
    depth = 0
    for i in range(start, len(flat_sql)):
        depth += (flat_sql[i] == "(") - (flat_sql[i] == ")")
        if depth == 0:
            expr = flat_sql[start : i + 1]
            break
    py = _re.sub(
        r"a\.(\w+)\s*=\s*'([^']*)'",
        lambda m: repr(row.get(m.group(1)) == m.group(2)),
        expr,
    )
    py = _re.sub(r"\bAND\b", " and ", py)
    py = _re.sub(r"\bOR\b", " or ", py)
    if _re.search(r"[A-Za-z_]\w*\s*(?:=|<|>)", py):  # an unconverted comparison
        raise AssertionError(f"could not evaluate: {py}")
    return bool(eval(py))  # noqa: S307 — a test-local boolean expression, no input from data


def test_identity_scope_leaves_a_nested_arm_it_cannot_read():
    """Anything not fully explained by comparisons and boolean words is left alone."""
    from src.retrievers.query_guards import enforce_identity_scope

    scopes, syns = ["a.org_unit", "a.sign"], ["a.login", "a.userId"]
    # A function call: the arm may constrain something other than an identity value.
    fn = "WHERE ((a.login = 'X' OR a.userId = 'X') OR (lower(a.org_unit) = 'y' AND a.sign = 'S'))"
    assert enforce_identity_scope(fn, scopes, syns, "s") == fn
    # An arm naming no declared column at all — there is no evidence about what it is.
    unknown = "WHERE ((a.login = 'X' OR a.userId = 'X') OR (a.other = 'q' AND a.more = 'r'))"
    assert enforce_identity_scope(unknown, scopes, syns, "s") == unknown
    # An UNDECLARED column inside the ACTOR's own arm: it may be a further spelling of the
    # identifier that the pack never declared, and dropping it would lose the subject's rows.
    # (Asymmetric with the scope arm above, deliberately — see the guard's own comment.)
    mixed = "WHERE ((a.login = 'X' OR a.nickname = 'X') OR (a.org_unit = 'Y' AND a.sign = 'S'))"
    assert enforce_identity_scope(mixed, scopes, syns, "s") == mixed


def test_identity_scope_reads_an_arm_it_cannot_decompose_but_can_attribute():
    """A quantified arm on a declared synonym column is the subject's own identity arm.

    The two questions are not the same. DECOMPOSING an arm hands back every operand so the caller
    can re-use them, so an arm that cannot be fully accounted for is unattributable. But this
    guard re-emits arms VERBATIM, so all it needs is which declared column an arm is ABOUT — and
    an arm holding exactly one comparison is about that comparison's column whatever wraps it.

    The shape is one this package's own rules FORCE: a leaf inside a repeated field cannot be
    reached by a dotted path on a textual backend (schema discovery treats a repeated field as
    terminal), so the only way to constrain such a leaf is a quantified predicate — and that was
    the single form the guard was blind to, on exactly the arms naming the subject.
    """
    from src.retrievers.query_guards import enforce_identity_scope

    scopes, syns = ["a.org_unit"], [["a.login", "a.members.login"]]
    sql = (
        "SELECT * FROM t WHERE a.day = '1' AND (a.login = 'X' "
        "OR exists(a.members, m -> m.login = 'X') OR a.org_unit = 'Y')"
    )
    out = enforce_identity_scope(sql, scopes, syns, "s")
    # The scope is AND-ed onto the family instead of satisfying the group on its own...
    assert "((a.login = 'X' OR exists(a.members, m -> m.login = 'X')) AND a.org_unit = 'Y')" in out
    # ...and the arm this module cannot parse went back out character for character.
    assert "exists(a.members, m -> m.login = 'X')" in out


def test_identity_scope_refuses_an_unparsable_arm_that_negates():
    """A negation is not an identity arm, and filing one would report the family COMPLETE.

    A family is AND-ed against the scopes only when every member is present. An excluding arm
    constrains nobody, so counting it as its column's member lets the rewrite proceed on the
    strength of an arm that selects nothing — and an operator then reads the published group as
    the subject's own identity.
    """
    from src.retrievers.query_guards import enforce_identity_scope

    scopes, syns = ["a.org_unit"], [["a.login", "a.members.login"]]
    sql = (
        "SELECT * FROM t WHERE a.day = '1' AND (a.login = 'X' "
        "OR NOT exists(a.members, m -> m.login = 'X') OR a.org_unit = 'Y')"
    )
    assert enforce_identity_scope(sql, scopes, syns, "s") == sql
    # A value CONTAINING the token is fine — the test blanks quoted literals first, or a legitimate
    # arm would be refused for what somebody's data happens to spell.
    ok = (
        "SELECT * FROM t WHERE a.day = '1' AND (a.login = 'DO NOT USE' "
        "OR exists(a.members, m -> m.login = 'DO NOT USE') OR a.org_unit = 'Y')"
    )
    assert enforce_identity_scope(ok, scopes, syns, "s") != ok


def test_identity_scope_refuses_an_unparsable_arm_holding_two_comparisons():
    """Exactly ONE comparison in the arm, whatever it is on — two make the attribution a guess.

    The count is over comparisons and NOT over declared columns, which is the stricter of the two
    readings and deliberately so: filing a two-comparison arm under whichever matched first
    reports a family slot filled by an arm that is equally about something else, so the family can
    read COMPLETE off one arm and the scopes are AND-ed against a reading nobody chose. The whole
    group declines instead — the same answer the decomposing reader gives, reached one step later.

    Both directions are asserted because the cheap version of this check ("does it name a declared
    column?") passes the first and fails the second, and the second is the commoner arm: a
    quantified predicate carrying a housekeeping conjunct beside the identity.
    """
    from src.retrievers.query_guards import enforce_identity_scope

    scopes, syns = ["a.org_unit"], [["a.login", "a.members.login"]]
    # Two DECLARED columns in one arm.
    two_declared = (
        "SELECT * FROM t WHERE (a.login = 'X' "
        "OR exists(a.members, m -> m.login = 'X' AND m.org_unit = 'Z') OR a.org_unit = 'Y')"
    )
    assert enforce_identity_scope(two_declared, scopes, syns, "s") == two_declared
    # One declared column and one the pack never named — still two comparisons, still declined.
    one_declared = (
        "SELECT * FROM t WHERE (a.login = 'X' "
        "OR exists(a.members, m -> m.login = 'X' AND m.deleted = false) OR a.org_unit = 'Y')"
    )
    assert enforce_identity_scope(one_declared, scopes, syns, "s") == one_declared


def test_identity_scope_will_not_lift_a_scope_it_could_not_decompose():
    """Attribution is read on the SYNONYM side only — a scope reaches the AND by being re-emitted.

    The lift re-emits the PREDICATES it decomposed, so a scope arm has to be readable to be
    lifted. Hoisting an opaque wrapper as a mandatory conjunct is a filter no measurement backs,
    and it is the one direction that can delete the subject's own rows.
    """
    from src.retrievers.query_guards import enforce_identity_scope

    scopes, syns = ["a.org_unit"], [["a.login"]]
    sql = (
        "SELECT * FROM t WHERE (a.login = 'X' "
        "OR exists(a.units, u -> u.org_unit = 'Y'))"
    )
    assert enforce_identity_scope(sql, scopes, syns, "s") == sql


def test_identity_scope_never_widens_when_every_arm_is_dropped():
    """With no surviving identity arm, lifting the scopes would be a SUPERSET of the input."""
    from src.retrievers.query_guards import enforce_identity_scope

    # Two scope-only arms. Lifting `org_unit` out of both would leave `org_unit = 'Y'` — the
    # whole unit, i.e. MORE rows than were asked for. The one direction never permitted.
    sql = "WHERE ((a.ip = '1' AND a.org_unit = 'Y') OR (a.ip = '2' AND a.org_unit = 'Y'))"
    assert enforce_identity_scope(sql, ["a.org_unit"], ["a.login"], "s") == sql


def test_identity_scope_splits_or_at_paren_depth_zero():
    """A top-level split cannot be a regex, and getting it wrong INVERTS the classification.

    `\\s+OR\\s+(?![^(]*\\))` looks right and splits the FIRST `OR` of
    `(a OR b) OR (c AND d)` — the one inside the left arm — because from there the next
    parenthesis is a close. The reader then treats the actor's own arm as the stray branch and
    the widening arm as the identity.
    """
    from src.retrievers.query_guards import _split_top_level_or

    assert _split_top_level_or("(a OR b) OR (c AND d)") == ["(a OR b) ", " (c AND d)"]
    assert _split_top_level_or("a OR b") == ["a ", " b"]
    assert _split_top_level_or("(a OR b)") == ["(a OR b)"]
    # A parenthesis inside a literal must not shift the depth for the rest of the clause.
    assert _split_top_level_or("x = '(' OR y = 'z'") == ["x = '(' ", " y = 'z'"]
    # `OR` as part of a longer word is not an operator.
    assert _split_top_level_or("ORDER = 'A'") == ["ORDER = 'A'"]
    assert _split_top_level_or("a = 'ORB' OR b = 'c'") == ["a = 'ORB' ", " b = 'c'"]
    # Unbalanced input yields None, not a best-effort split: a half-parsed boolean is the
    # worst thing to rewrite.
    assert _split_top_level_or("(a OR b") is None
    assert _split_top_level_or("a) OR b") is None


def test_identity_scope_declaration_rides_to_every_backend():
    """`_attach_query_guards` is the one seam, so a new backend cannot drop the shape.

    The same defect this guard fixes would return silently if the declaration were wired
    per-retriever: that is exactly how `never_filter` was once databricks-only.
    """
    from src.log_retrieval import LogRetrievalEngine

    src = MagicMock()
    src.identity_scopes = ["a.org_unit"]
    src.identity_synonyms = ["a.login", "a.userId"]
    merged = {}
    LogRetrievalEngine._attach_query_guards(src, merged)
    assert merged["identity_scopes"] == ["a.org_unit"]
    assert merged["identity_synonyms"] == ["a.login", "a.userId"]


# --- a disjunct that constrains nothing ---------------------------------------------------
#
# `sign LIKE 'X%' OR sign IS NOT NULL` names the subject in the left arm but the right arm
# subsumes it: the cross-entity test has nothing to compare and the identity rewrite only
# runs for a source that declares an identity shape, which a scope sweep does not.


def test_vacuous_disjunct_is_dropped_when_a_real_arm_survives():
    """The live shape, verbatim in structure: an anchored LIKE OR-ed with IS NOT NULL."""
    from src.retrievers.query_guards import strip_vacuous_disjuncts

    sql = (
        "SELECT a FROM t\nWHERE d >= DATE'2025-04-12'\n  AND (\n"
        "    (a.office = 'O' AND (a.sign LIKE 'X%' OR a.sign IS NOT NULL))\n"
        "    OR (b.office = 'O' AND (b.sign LIKE 'X%' OR b.sign IS NOT NULL))\n  )"
    )
    out = strip_vacuous_disjuncts(sql, "sweep")
    flat = " ".join(out.split())
    # The vacuous half is gone from BOTH arms...
    assert "IS NOT NULL" not in flat
    # ...and the arm naming the subject is untouched, on both actor sides.
    assert "a.sign LIKE 'X%'" in flat and "b.sign LIKE 'X%'" in flat
    # The enclosing structure and the window survive: this guard drops an arm, it does not
    # restructure the boolean tree.
    assert "a.office = 'O'" in flat and "b.office = 'O'" in flat
    assert "d >= DATE'2025-04-12'" in out
    # The other spellings of the same claim.
    for rhs in ("<> ''", "!= ''", "LIKE '%'"):
        one = f"WHERE (a.sign LIKE 'X%' OR a.sign {rhs})"
        assert "a.sign LIKE 'X%'" in strip_vacuous_disjuncts(one, "s")
        assert rhs not in strip_vacuous_disjuncts(one, "s")


def test_vacuous_disjunct_alone_is_left_exactly_as_generated():
    """With no real arm on the column, dropping it would WIDEN the query."""
    from src.retrievers.query_guards import strip_vacuous_disjuncts

    # Every arm vacuous: removing them leaves the column unconstrained, which is more rows
    # than were asked for — the one direction no guard here may take.
    both = "WHERE (a.sign IS NOT NULL OR a.other IS NOT NULL)"
    assert strip_vacuous_disjuncts(both, "s") == both
    # A lone vacuous predicate is not in an OR-group at all, so there is nothing to subsume:
    # it may be the only thing keeping the scan bounded.
    lone = "WHERE a.sign IS NOT NULL AND d >= DATE'2025-04-12'"
    assert strip_vacuous_disjuncts(lone, "s") == lone
    # The surviving arm must constrain the SAME column, or the drop changes WHICH column is
    # bounded rather than only how tightly.
    other = "WHERE (a.office = 'O' OR a.sign IS NOT NULL)"
    assert strip_vacuous_disjuncts(other, "s") == other
    # An arm that says something real BESIDE the vacuous comparison is not an empty arm.
    mixed = "WHERE (a.sign LIKE 'X%' OR (a.sign IS NOT NULL AND a.office = 'O'))"
    assert strip_vacuous_disjuncts(mixed, "s") == mixed
    # And an ANCHORED pattern is a real filter, never touched.
    real = "WHERE (a.sign LIKE 'X%' OR a.sign LIKE 'Y%')"
    assert strip_vacuous_disjuncts(real, "s") == real


def test_vacuous_disjunct_leaves_a_null_tolerant_range_alone():
    """`col IS NULL OR (col BETWEEN ...)` is the opposite shape and must survive.

    Generated deliberately and often — "in the window, or undated" — and it is NARROWER than
    the range alone would be over a nullable column, not wider. `IS NULL` is not in the
    vacuous vocabulary for exactly this reason; only `IS NOT NULL` is.
    """
    from src.retrievers.query_guards import strip_vacuous_disjuncts

    sql = (
        "WHERE (td.date IS NULL OR (td.date >= DATE'2025-04-14' "
        "AND td.date <= DATE'2025-04-17'))"
    )
    assert strip_vacuous_disjuncts(sql, "s") == sql


def test_vacuous_should_clause_is_dropped_on_the_dsl_route_too():
    """Both query families or it is not a shape (the mistake the DSL twin was added for)."""
    from src.retrievers.query_guards import strip_vacuous_should_clauses

    dsl = {
        "query": {
            "bool": {
                "filter": [{"range": {"date": {"gte": "2025-04-12"}}}],
                "should": [
                    {"term": {"sign": "X"}},
                    {"exists": {"field": "sign"}},
                    {"wildcard": {"sign": "*"}},
                ],
                "minimum_should_match": 1,
            }
        }
    }
    out = strip_vacuous_should_clauses(dsl, "s")
    should = out["query"]["bool"]["should"]
    assert should == [{"term": {"sign": "X"}}]
    # The bound is untouched, and the input is not mutated.
    assert out["query"]["bool"]["filter"] == [{"range": {"date": {"gte": "2025-04-12"}}}]
    assert len(dsl["query"]["bool"]["should"]) == 3

    # An `exists` under `filter` is an AND-ed populated-ness requirement — it NARROWS, and is
    # very often exactly what the query means. Only a `should` is an OR.
    anded = {"query": {"bool": {"filter": [{"exists": {"field": "sign"}}]}}}
    assert strip_vacuous_should_clauses(anded, "s") == anded
    # Nothing real on the field: dropping the arm would unbound it.
    alone = {
        "query": {"bool": {"should": [{"exists": {"field": "sign"}}]}},
    }
    assert strip_vacuous_should_clauses(alone, "s") == alone
    # A real clause on a DIFFERENT field does not license the drop.
    elsewhere = {
        "query": {
            "bool": {
                "should": [{"term": {"office": "O"}}, {"exists": {"field": "sign"}}]
            }
        }
    }
    assert strip_vacuous_should_clauses(elsewhere, "s") == elsewhere
    # An anchored wildcard is a real filter.
    anchored = {
        "query": {
            "bool": {"should": [{"term": {"sign": "X"}}, {"wildcard": {"sign": "X*"}}]}
        }
    }
    assert strip_vacuous_should_clauses(anchored, "s") == anchored


def test_vacuous_guard_runs_on_every_backend_before_the_identity_rewrite():
    """Order is load-bearing, and a guard wired into one retriever is a silent no-op.

    An arm reading `sign LIKE 'X%' OR sign IS NOT NULL` is not a plain comparison, so
    `enforce_identity_scope` classifies it as unreadable and declines on the WHOLE group.
    Dropping the vacuous half first is what leaves a group it can act on — so this asserts the
    sequence in the source, not just that both functions exist.
    """
    import inspect

    from src.retrievers import (databricks_retriever, elasticsearch_retriever,
                                kibana_retriever, snowflake_retriever)

    textual = [
        (databricks_retriever, "strip_vacuous_disjuncts"),
        (elasticsearch_retriever, "strip_vacuous_disjuncts"),
        (snowflake_retriever, "strip_vacuous_disjuncts"),
        (kibana_retriever, "strip_vacuous_should_clauses"),
    ]
    for module, guard in textual:
        body = inspect.getsource(module)
        retrieve = body[body.index("async def retrieve") :]
        assert guard in retrieve, f"{module.__name__} does not run {guard}"
        assert retrieve.index(guard) < retrieve.index(
            "_enforce_identity_scope"
        ), f"{module.__name__} runs {guard} AFTER the identity rewrite"


# --- the same shape on the Query DSL route, and several families rather than one ---------
#
# The SQL guard was wired into one retriever; for a Query DSL source the pack declaration
# was a silent no-op. A flat `should` under `minimum_should_match: 1` matches the whole
# organisation even when one arm names the subject.


def test_identity_scope_dsl_splits_a_flat_should_into_families():
    from src.retrievers.query_guards import enforce_identity_scope_dsl

    dsl = {
        "query": {
            "bool": {
                "filter": [{"range": {"date": {"gte": "2026-07-08"}}}],
                "should": [
                    {"term": {"login": "X"}},
                    {"term": {"a.userId": "X"}},
                    {"term": {"episode": "E"}},
                    {"term": {"a.episodeId": "E"}},
                ],
                "minimum_should_match": 1,
            }
        }
    }
    out = enforce_identity_scope_dsl(
        dsl, [], [["login", "a.userId"], ["episode", "a.episodeId"]], "s"
    )
    body = out["query"]["bool"]
    # The two families become AND-ed conjuncts, each OR-ed inside...
    assert len(body["must"]) == 2
    groups = [sorted(next(iter(c["term"])) for c in m["bool"]["should"]) for m in body["must"]]
    assert sorted(groups) == [["a.episodeId", "episode"], ["a.userId", "login"]]
    assert all(m["bool"]["minimum_should_match"] == 1 for m in body["must"])
    # ...the flat group is gone, and with it the setting that described it: left behind,
    # `minimum_should_match` would apply to a clause list that no longer exists.
    assert "should" not in body and "minimum_should_match" not in body
    # The window is untouched.
    assert body["filter"] == [{"range": {"date": {"gte": "2026-07-08"}}}]
    # The input is not mutated.
    assert len(dsl["query"]["bool"]["should"]) == 4


def test_identity_scope_dsl_leaves_one_family_alone():
    """Columns holding ONE value must stay OR-ed — a single family is already correct."""
    from src.retrievers.query_guards import enforce_identity_scope_dsl

    dsl = {"bool": {"should": [{"term": {"login": "X"}}, {"term": {"userId": "X"}}]}}
    assert enforce_identity_scope_dsl(dsl, [], [["login", "userId"]], "s") == dsl


def test_identity_scope_dsl_declines_on_an_undeclared_clause():
    """One clause the pack does not declare and the whole group is left as generated.

    The guard cannot know which family such a clause belongs to, and narrowing around it
    could drop a constraint the caller did intend. This is also why the two strips must run
    BEFORE this guard in every retriever: measured on job 0184a3ce, two predicates putting
    the actor's own office on the target-office columns matched 0 rows of 500 and their only
    other effect was to block this rewrite entirely.
    """
    from src.retrievers.query_guards import enforce_identity_scope_dsl

    dsl = {
        "bool": {
            "should": [
                {"term": {"login": "X"}},
                {"term": {"episode": "E"}},
                {"term": {"somethingElse": "Z"}},
            ]
        }
    }
    assert enforce_identity_scope_dsl(dsl, [], [["login"], ["episode"]], "s") == dsl
    # Nor is a scope ever INVENTED: a family the query never constrained is not added.
    single = {"bool": {"should": [{"term": {"login": "X"}}, {"term": {"userId": "X"}}]}}
    out = enforce_identity_scope_dsl(single, ["org_unit"], [["login", "userId"]], "s")
    assert "org_unit" not in json.dumps(out)


def test_identity_scope_dsl_drops_an_INCOMPLETE_family():
    """A family generated on some of its columns is not the declared filter.

    MEASURED on job 0184a3ce: the office family names one column per document shape and the
    query carried 2 of the 3. Enforced as written it cut a correct 17-row result to 12 and
    deleted every row of the third shape — the five rows carrying the per-action detail the
    checks read. Dropping the clauses instead can only WIDEN an OR-group, never exclude a row
    that belonged in the answer.
    """
    from src.retrievers.query_guards import enforce_identity_scope_dsl

    dsl = {
        "bool": {
            "should": [
                {"term": {"login": "X"}},
                {"term": {"a.userId": "X"}},
                {"term": {"officeB": "O"}},
                {"term": {"officeC": "O"}},
            ],
            "minimum_should_match": 1,
        }
    }
    out = enforce_identity_scope_dsl(
        dsl, [], [["login", "a.userId"], ["officeA", "officeB", "officeC"]], "s"
    )
    text = json.dumps(out)
    assert "officeB" not in text and "officeC" not in text, "enforced a partial family"
    assert "login" in text and "a.userId" in text


def test_identity_scope_dsl_refuses_when_only_scopes_would_survive():
    """Scopes alone are the whole-population query this guard exists to prevent."""
    from src.retrievers.query_guards import enforce_identity_scope_dsl

    dsl = {
        "bool": {
            "should": [
                {"term": {"org_unit": "U"}},
                {"term": {"officeB": "O"}},
            ]
        }
    }
    assert (
        enforce_identity_scope_dsl(
            dsl, ["org_unit"], [["officeA", "officeB"]], "s"
        )
        == dsl
    )


def test_synonym_families_normalises_both_declared_forms():
    """One boundary reader, so the SQL route and the DSL route cannot disagree."""
    from src.retrievers.query_guards import synonym_families

    assert synonym_families(["a", "b"]) == [["a", "b"]]
    assert synonym_families([["a", "b"], ["c"]]) == [["a", "b"], ["c"]]
    assert synonym_families(None) == [] and synonym_families([]) == []


def test_resolve_identity_fields_keeps_a_flat_column_name():
    """A bare column is a column, not an unresolvable entity type.

    Telling the two apart by "does it contain a dot" reads a top-level column as a type and
    drops it — which empties the declaration only on sources that HAVE flat columns, so the
    mistake is invisible on a fully-nested backend. Measured on job 0184a3ce: `sign` and
    `channelId` were both discarded and the guard then found nothing to restructure.
    """
    from src.retrievers.query_guards import resolve_identity_fields

    out = resolve_identity_fields(["user", "sign", "a.b"], {"user": "mapped.login"}, "s")
    assert out == ["mapped.login", "sign", "a.b"]
    # Families keep their grouping through the resolver.
    assert resolve_identity_fields(
        [["user", "sign"], ["channelId"]], {"user": "mapped.login"}, "s"
    ) == [["mapped.login", "sign"], ["channelId"]]


def test_identity_scope_dsl_rides_to_the_dsl_retrievers():
    """A guard is only a guard if the retriever calls it, AFTER both strips.

    Same reason as the fabricated-literal wiring test: the funnel ORDER is load-bearing here,
    because the rewrite declines on any clause the pack does not declare and a strip is what
    removes those.
    """
    import inspect

    from src.retrievers import (databricks_retriever, elasticsearch_retriever,
                                kibana_retriever, snowflake_retriever)

    for module in (
        kibana_retriever,
        elasticsearch_retriever,
        snowflake_retriever,
        databricks_retriever,
    ):
        # ...its OWN retriever, not the imported abstract base (whose `retrieve` is a stub,
        # so matching on it would make every assertion below vacuous).
        cls = next(
            obj
            for _, obj in vars(module).items()
            if isinstance(obj, type)
            and obj.__name__.endswith("Retriever")
            and obj.__module__ == module.__name__
        )
        body = inspect.getsource(cls.retrieve)
        assert "_enforce_identity_scope" in body, module.__name__
        identity = body.index("_enforce_identity_scope")
        # Each retriever spells the strip its own way (a helper on one, the guard function
        # directly on the others), so match on whichever it uses rather than one literal.
        strip = min(
            body.index(name) for name in ("_strip_fabricated", "strip_fabricated") if name in body
        )
        assert (
            strip < identity
        ), f"{module.__name__}: identity guard must run after the fabricated strip"
        bounds = min(
            body.index(name)
            for name in ("enforce_partition_bounds", "_enforce_partition")
            if name in body
        )
        assert (
            identity < bounds
        ), f"{module.__name__}: identity guard must run before the bounds"


def test_required_fields_resolves_entity_types_in_order():
    from src.retrievers.query_guards import required_fields

    field_map = {"user": "sign", "org_unit": "orgUnitId", "record": "locator"}
    assert required_fields(field_map, ["org_unit", "user"]) == ["orgUnitId", "sign"]
    # An unmapped entity type is skipped rather than emitting a None field.
    assert required_fields(field_map, ["org_unit", "missing"]) == ["orgUnitId"]


def test_guard_prompt_line_is_empty_without_declaration():
    """The prompt line is derived from the pack field, so declaring `never_filter` is
    self-sufficient — no source has to hand-write the same prose into query_hints."""
    from src.retrievers.query_guards import guard_prompt_line

    assert guard_prompt_line([]) == ""
    line = guard_prompt_line(["a.b.robot"])
    assert "a.b.robot" in line and "NEVER" in line


@pytest.mark.asyncio
async def test_elasticsearch_strips_filter_on_evidence_field():
    """Same guarantee as the Databricks path, on ES|QL."""
    from src.models.pydantic_models import FieldMapping

    config = {
        "name": "auth_events",
        "url": "http://es:9200",
        "username": "u",
        "password": "p",
        "index": "auth",
        "max_results": 50,
        "field_schema": "robot boolean",
        "never_filter": ["userInfo.robot"],
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        side_effect=[
            FieldMapping(mappings=[]),
            EsqlQuery(query='FROM auth | WHERE ts > "2026-07-17" AND robot == false'),
        ]
    )
    retriever = ElasticsearchRetriever(config, llm)
    fake_client = MagicMock()
    fake_client.esql.query = AsyncMock(return_value={"columns": [], "values": []})
    with patch(
        "src.retrievers.elasticsearch_retriever.AsyncElasticsearch",
        return_value=fake_client,
    ):
        await retriever.retrieve(_query("auth_events"))

    executed = fake_client.esql.query.call_args.kwargs["query"]
    assert "robot" not in executed
    assert 'ts > "2026-07-17"' in executed


@pytest.mark.asyncio
async def test_elasticsearch_enforces_composite_key_conjunction():
    from src.models.pydantic_models import ExtractedEntity, FieldMapping
    from src.models.pydantic_models import RetrievalQuery as RQ

    config = {
        "name": "automation_registry",
        "url": "http://es:9200",
        "username": "u",
        "password": "p",
        "index": "automated",
        "field_schema": "orgUnitId keyword, sign keyword",
        "require_all_entities": ["org_unit", "user"],
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        side_effect=[
            FieldMapping(mappings=[]),
            EsqlQuery(query='FROM automated | WHERE orgUnitId == "A" OR sign == "B"'),
        ]
    )
    retriever = ElasticsearchRetriever(config, llm)
    retriever.last_field_map = None
    # map_entities is mocked out via FieldMapping(mappings=[]), so inject the map the
    # generator would have received.
    with patch(
        "src.retrievers.elasticsearch_retriever.map_entities",
        AsyncMock(return_value={"org_unit": "orgUnitId", "user": "sign"}),
    ):
        fake_client = MagicMock()
        fake_client.esql.query = AsyncMock(return_value={"columns": [], "values": []})
        with patch(
            "src.retrievers.elasticsearch_retriever.AsyncElasticsearch",
            return_value=fake_client,
        ):
            await retriever.retrieve(
                RQ(
                    target_log_source="automation_registry",
                    natural_language_query="is this identity automated",
                    date_from="2026-07-17",
                    date_to="2026-07-17",
                    entities=[
                        ExtractedEntity(type="org_unit", value="A"),
                        ExtractedEntity(type="user", value="B"),
                    ],
                )
            )

    executed = fake_client.esql.query.call_args.kwargs["query"]
    assert 'orgUnitId == "A" AND sign == "B"' in executed


def test_dsl_strip_removes_evidence_clause_from_bool_filter():
    from src.retrievers.query_guards import strip_evidence_filters_dsl

    dsl = {
        "bool": {
            "filter": [
                {"range": {"ts": {"gte": "now-1d"}}},
                {"term": {"userInfo.robot": False}},
            ]
        }
    }
    out = strip_evidence_filters_dsl(dsl, ["userInfo.robot"])
    assert out["bool"]["filter"] == [{"range": {"ts": {"gte": "now-1d"}}}]
    # Input not mutated.
    assert len(dsl["bool"]["filter"]) == 2


def test_dsl_strip_drops_emptied_occurrence_key():
    """`must: []` is not a reliably neutral filter, so an emptied list is removed and a
    bool left with no occurrence clauses degrades to match_all (still a valid query)."""
    from src.retrievers.query_guards import strip_evidence_filters_dsl

    out = strip_evidence_filters_dsl(
        {"bool": {"must": [{"term": {"robot": False}}]}}, ["robot"]
    )
    assert out == {"match_all": {}}


def test_dsl_strip_reaches_nested_bools():
    from src.retrievers.query_guards import strip_evidence_filters_dsl

    dsl = {
        "bool": {
            "must": [
                {
                    "bool": {
                        "should": [
                            {"term": {"org_unit": "A"}},
                            {"term": {"robot": True}},
                        ]
                    }
                }
            ]
        }
    }
    out = strip_evidence_filters_dsl(dsl, ["userInfo.robot"])
    assert out["bool"]["must"][0]["bool"]["should"] == [{"term": {"org_unit": "A"}}]


def test_dsl_enforce_conjunction_promotes_should_to_must():
    from src.retrievers.query_guards import enforce_conjunction_dsl

    dsl = {
        "bool": {
            "should": [{"term": {"orgUnitId": "A"}}, {"term": {"sign": "B"}}],
            "minimum_should_match": 1,
        }
    }
    out = enforce_conjunction_dsl(dsl, ["orgUnitId", "sign"])
    assert "should" not in out["bool"]
    assert "minimum_should_match" not in out["bool"]
    assert out["bool"]["must"] == [
        {"term": {"orgUnitId": "A"}},
        {"term": {"sign": "B"}},
    ]


def test_dsl_enforce_conjunction_keeps_an_unkeyed_arm_beside_the_conjunction():
    """An arm the key does not name is kept OR-ed BESIDE the conjunction, never dropped and
    never a reason to decline.

    This pinned the opposite behaviour (`out == dsl`) on the argument that narrowing a wider
    `should` would drop constraints the caller intended. Nothing is dropped — the `locator` arm
    below survives — and requiring the group to name the key and *nothing else* made the
    declaration unreachable in the normal case: a generator handed three entities ORs three
    arms, so a source declaring `identity_keys: [[office, user]]` shipped the full OR. That is
    the correction `enforce_conjunction_same_column_dsl` already carries.
    """
    from src.retrievers.query_guards import enforce_conjunction_dsl

    dsl = {
        "bool": {
            "should": [
                {"term": {"orgUnitId": "A"}},
                {"term": {"sign": "B"}},
                {"term": {"locator": "9CRPQG"}},
            ]
        }
    }
    out = enforce_conjunction_dsl(dsl, ["orgUnitId", "sign"])
    assert out != dsl
    assert out["bool"]["minimum_should_match"] == 1
    assert out["bool"]["should"] == [
        {"bool": {"must": [{"term": {"orgUnitId": "A"}}, {"term": {"sign": "B"}}]}},
        {"term": {"locator": "9CRPQG"}},
    ]


def test_dsl_enforce_conjunction_ors_two_values_of_one_key_column():
    """AND between entity TYPES, OR within one type's values — enforced per FIELD.

    One entity type can carry two non-interchangeable value forms, and a source binding both to
    one column yields two arms on that column. Promoting them wholesale produced
    `userId = <login> AND userId = <sign>`, which no document satisfies: the zero-row rewrite
    the OR-ing default exists to prevent, through the guard meant to prevent the opposite one.
    """
    from src.retrievers.query_guards import enforce_conjunction_dsl

    dsl = {
        "bool": {
            "should": [
                {"term": {"orgUnitId": "A"}},
                {"term": {"userId": "ALOGIN"}},
                {"term": {"userId": "0101XY"}},
            ],
            "minimum_should_match": 1,
        }
    }
    out = enforce_conjunction_dsl(dsl, ["orgUnitId", "userId"])
    assert out["bool"]["must"] == [
        {"term": {"orgUnitId": "A"}},
        {
            "bool": {
                "should": [
                    {"term": {"userId": "ALOGIN"}},
                    {"term": {"userId": "0101XY"}},
                ],
                # Explicit: the default is 1 only while the bool carries no `must`, and the
                # promotion just put one there — implicit would make both forms mandatory.
                "minimum_should_match": 1,
            }
        },
    ]
    assert "should" not in out["bool"]


def test_dsl_enforce_conjunction_preserves_an_existing_must_and_declines_a_partial_key():
    """The bool's own `must` (a window, a partition bound) is walked, never rebuilt — and a
    group that does not name every key member is left exactly as generated."""
    from src.retrievers.query_guards import enforce_conjunction_dsl

    dsl = {
        "bool": {
            "must": [{"range": {"ts": {"gte": "now-30d"}}}],
            "should": [
                {"term": {"orgUnitId": "A"}},
                {"term": {"sign": "B"}},
                {"term": {"locator": "9CRPQG"}},
            ],
            "minimum_should_match": 1,
        }
    }
    out = enforce_conjunction_dsl(dsl, ["orgUnitId", "sign"])
    assert out["bool"]["must"] == [{"range": {"ts": {"gte": "now-30d"}}}]
    assert out["bool"]["should"][0] == {
        "bool": {"must": [{"term": {"orgUnitId": "A"}}, {"term": {"sign": "B"}}]}
    }

    partial = {"bool": {"should": [{"term": {"orgUnitId": "A"}}, {"term": {"locator": "X"}}]}}
    assert enforce_conjunction_dsl(partial, ["orgUnitId", "sign"]) == partial

    # Two spellings of ONE leaf resolve to one column: there is no conjunction to write between
    # a field and itself, and promoting the group would AND two values of that column.
    assert enforce_conjunction_dsl(dsl, ["t.sign", "sign"]) == dsl


# The declaration below is `field_mapping.form_split_bindings`' shape verbatim: one entry per
# entity TYPE whose value FORMS are bound to more than one column on this source.
_SPLIT_USER = [("user", [("code", ["actorSign"]), ("login", ["actorUserId"])])]


def test_dsl_enforce_conjunction_groups_arms_per_entity_TYPE_and_not_per_column(caplog):
    """AND between entity types, OR within one type's values and forms.

    Per field, the sibling form column is a spare arm bounded by nothing. `form_bindings`
    lets the guard OR the form inside the member's own conjunct instead. The counterfactual
    (no `form_bindings`) is asserted to prove this is about the declaration: a pack declaring
    no split forms is byte-identical to before. The log line is the only trace of which
    reading ran: the guard reports the same headline either way.
    """
    import logging

    from src.retrievers.query_guards import enforce_conjunction_dsl

    dsl = {
        "bool": {
            "should": [
                {"term": {"unitId": "UNIT01"}},
                {"term": {"actorSign": "SIGN01"}},
                {"term": {"actorUserId": "LOGIN01"}},
            ],
            "minimum_should_match": 1,
        }
    }
    with caplog.at_level(logging.WARNING, logger="src.retrievers.query_guards"):
        out = enforce_conjunction_dsl(
            dsl, ["unitId", "actorSign"], "src", form_bindings=_SPLIT_USER
        )
    assert out["bool"]["must"] == [
        {"term": {"unitId": "UNIT01"}},
        {
            "bool": {
                "should": [
                    {"term": {"actorSign": "SIGN01"}},
                    {"term": {"actorUserId": "LOGIN01"}},
                ],
                "minimum_should_match": 1,
            }
        },
    ]
    logged = caplog.text
    assert "1 key member(s) constrained on several of their own entity TYPE's columns" in logged
    assert "the key does not name" not in logged
    # Nothing rode out beside the conjunction, and nothing was dropped.
    assert "should" not in out["bool"]
    assert "minimum_should_match" not in out["bool"]

    undeclared = enforce_conjunction_dsl(dsl, ["unitId", "actorSign"], "src")
    assert undeclared["bool"]["minimum_should_match"] == 1
    assert undeclared["bool"]["should"][-1] == {"term": {"actorUserId": "LOGIN01"}}


def test_dsl_enforce_conjunction_honours_a_key_member_on_its_SIBLING_column_alone():
    """A key member constrained ONLY on its other form's column is the key, not a partial one.

    Per field the resolved member never appeared, so `required <= seen` failed and the whole
    rewrite declined — shipping the full OR for a source that had declared the tuple. The
    canonicalisation makes the declaration reachable here too, and the promotion is the
    ordinary one: two conjuncts, no nested `should`, because only one form was asked for.
    """
    from src.retrievers.query_guards import enforce_conjunction_dsl

    dsl = {
        "bool": {
            "should": [
                {"term": {"unitId": "UNIT01"}},
                {"term": {"actorUserId": "LOGIN01"}},
            ],
            "minimum_should_match": 1,
        }
    }
    assert enforce_conjunction_dsl(dsl, ["unitId", "actorSign"], "src") == dsl
    out = enforce_conjunction_dsl(
        dsl, ["unitId", "actorSign"], "src", form_bindings=_SPLIT_USER
    )
    assert out["bool"]["must"] == [
        {"term": {"unitId": "UNIT01"}},
        {"term": {"actorUserId": "LOGIN01"}},
    ]
    assert "should" not in out["bool"]


def test_dsl_enforce_conjunction_folds_a_form_column_only_where_the_key_names_ONE_of_them():
    """Two bounds on the fold, and each is a different way it could delete a constraint.

    A split type the key does NOT name keeps the spare-arm reading — the fold is not a licence to
    treat any declared column as part of the key. And where the key itself names TWO of one type's
    columns, folding them together would delete a conjunction the pack asked for, so the per-field
    reading stands: they are AND-ed, exactly as declared.
    """
    from src.retrievers.query_guards import enforce_conjunction_dsl

    unnamed = {
        "bool": {
            "should": [
                {"term": {"unitId": "UNIT01"}},
                {"term": {"subjectRef": "SUBJ01"}},
                {"term": {"actorUserId": "LOGIN01"}},
            ]
        }
    }
    out = enforce_conjunction_dsl(
        unnamed, ["unitId", "subjectRef"], "src", form_bindings=_SPLIT_USER
    )
    assert out["bool"]["minimum_should_match"] == 1
    assert out["bool"]["should"] == [
        {
            "bool": {
                "must": [
                    {"term": {"unitId": "UNIT01"}},
                    {"term": {"subjectRef": "SUBJ01"}},
                ]
            }
        },
        {"term": {"actorUserId": "LOGIN01"}},
    ]

    # A THIRD form is what makes this half discriminating: with only two, the key naming both
    # leaves nothing to canonicalise and the assertion would hold whatever the guard did.
    three_forms = [
        (
            "user",
            [
                ("code", ["actorSign"]),
                ("login", ["actorUserId"]),
                ("alias", ["actorAlias"]),
            ],
        )
    ]
    both_named = {
        "bool": {
            "should": [
                {"term": {"actorSign": "SIGN01"}},
                {"term": {"actorUserId": "LOGIN01"}},
                {"term": {"actorAlias": "ALIAS01"}},
            ]
        }
    }
    keyed_on_both = enforce_conjunction_dsl(
        both_named,
        ["actorSign", "actorUserId"],
        "src",
        form_bindings=three_forms,
    )
    assert keyed_on_both["bool"]["should"] == [
        {
            "bool": {
                "must": [
                    {"term": {"actorSign": "SIGN01"}},
                    {"term": {"actorUserId": "LOGIN01"}},
                ]
            }
        },
        {"term": {"actorAlias": "ALIAS01"}},
    ]


def test_dsl_enforce_conjunction_folds_a_type_bound_FLAT_to_several_columns_by_its_VALUES(
    caplog,
):
    """The fold no declaration can supply: a type bound flat to several columns.

    `form_split_bindings` returns nothing for such a type; arms are attributed by the literals
    they carry through `_literal_value_types`. The counterfactual (no `incident_values`)
    produces the spare-arm reading, proving a caller that passes none is byte-identical to before.
    """
    import logging

    from src.retrievers.query_guards import enforce_conjunction_dsl

    dsl = {
        "bool": {
            "filter": [{"range": {"ts": {"gte": "a", "lte": "b"}}}],
            "should": [
                {"term": {"recordRef": "REF01"}},
                {"term": {"unitId": "UNIT01"}},
                {"term": {"altUnit": "UNIT01"}},
                {"terms": {"unitCode": ["UNIT01"]}},
            ],
            "minimum_should_match": 1,
        }
    }
    fields = ["recordRef", "unitId"]
    values = {"record": ["REF01"], "office": ["UNIT01"], "time_window": ["2026-07-18"]}

    with caplog.at_level(logging.WARNING, logger="src.retrievers.query_guards"):
        out = enforce_conjunction_dsl(dsl, fields, "src", incident_values=values)
    assert out["bool"]["must"] == [
        {"term": {"recordRef": "REF01"}},
        {
            "bool": {
                "should": [
                    {"term": {"unitId": "UNIT01"}},
                    {"term": {"altUnit": "UNIT01"}},
                    {"terms": {"unitCode": ["UNIT01"]}},
                ],
                "minimum_should_match": 1,
            }
        },
    ]
    assert "should" not in out["bool"]
    # The window rode in a `filter`, which this guard never rebuilds.
    assert out["bool"]["filter"] == dsl["bool"]["filter"]
    logged = caplog.text
    assert "1 key member(s) constrained on several of their own entity TYPE's columns" in logged
    assert "the key does not name" not in logged

    unattributed = enforce_conjunction_dsl(dsl, fields, "src")
    spare = unattributed["bool"]["should"][1:]
    assert unattributed["bool"]["minimum_should_match"] == 1
    assert spare == [
        {"term": {"altUnit": "UNIT01"}},
        {"terms": {"unitCode": ["UNIT01"]}},
    ]


def test_dsl_enforce_conjunction_refuses_to_fold_an_arm_it_cannot_attribute_to_ONE_type():
    """Four bounds on the value route, each a different way folding would NARROW the query.

    Folding an arm into a conjunct it does not belong to makes it mandatory, so every ambiguity is
    refused and the arm keeps the spare-arm reading it had before.
    """
    from src.retrievers.query_guards import enforce_conjunction_dsl

    fields = ["recordRef", "unitId"]
    values = {"record": ["REF01"], "office": ["UNIT01"], "user": ["SIGN01"]}

    # 1. A literal of no known type at all — an environment constant, a value the incident never
    #    carried. That arm is `never_filter`/`default_filters`' subject, not this guard's.
    constant = {
        "bool": {
            "should": [
                {"term": {"recordRef": "REF01"}},
                {"term": {"unitId": "UNIT01"}},
                {"term": {"phase": "PRD"}},
            ]
        }
    }
    out = enforce_conjunction_dsl(constant, fields, "src", incident_values=values)
    assert out["bool"]["should"][-1] == {"term": {"phase": "PRD"}}
    assert out["bool"]["minimum_should_match"] == 1

    # 2. One `terms` list mixing two types' values: the intersection across its literals is empty,
    #    so there is no ONE type it is about.
    mixed = {
        "bool": {
            "should": [
                {"term": {"recordRef": "REF01"}},
                {"term": {"unitId": "UNIT01"}},
                {"terms": {"blend": ["UNIT01", "SIGN01"]}},
            ]
        }
    }
    out = enforce_conjunction_dsl(mixed, fields, "src", incident_values=values)
    assert out["bool"]["should"][-1] == {"terms": {"blend": ["UNIT01", "SIGN01"]}}

    # 3. One LEAF two types both claim — a blob column several types collapse onto. That is
    #    `enforce_conjunction_same_column`'s shape, where the arms must be AND-ed; OR-ing them
    #    inside one member's conjunct is the opposite of the right answer, so the leaf is dropped
    #    from the grouping and both arms stay spare. Without the drop, the second claim overwrites
    #    the first and both fold into the actor's conjunct.
    blob = {
        "bool": {
            "should": [
                {"term": {"recordRef": "REF01"}},
                {"term": {"unitId": "UNIT01"}},
                {"term": {"irText": "UNIT01"}},
                {"term": {"irText": "SIGN01"}},
            ]
        }
    }
    out = enforce_conjunction_dsl(blob, fields, "src", incident_values=values)
    assert out["bool"]["should"] == [
        {
            "bool": {
                "must": [
                    {"term": {"recordRef": "REF01"}},
                    {"term": {"unitId": "UNIT01"}},
                ]
            }
        },
        {"term": {"irText": "UNIT01"}},
        {"term": {"irText": "SIGN01"}},
    ]

    # 4. One VALUE two types share, with nothing to break the tie. Picking either is a fold whose
    #    direction depends on set iteration order, i.e. a query that differs run to run.
    shared = {"record": ["REF01"], "office": ["UNIT01"], "user": ["UNIT01"]}
    ambiguous = {
        "bool": {
            "should": [
                {"term": {"recordRef": "REF01"}},
                {"term": {"unitId": "UNIT01"}},
                {"term": {"someCol": "UNIT01"}},
            ]
        }
    }
    out = enforce_conjunction_dsl(ambiguous, fields, "src", incident_values=shared)
    assert out["bool"]["should"][-1] == {"term": {"someCol": "UNIT01"}}
    assert out["bool"]["minimum_should_match"] == 1


def test_dsl_enforce_conjunction_reads_a_key_arm_the_generator_ALREADY_conjoined():
    """An arm already holding the conjunction reads as constraining no field.

    `_dsl_clause_fields` returns nothing for a nested `bool`; a conjunction over key members
    is narrower than the other arms (not wider) and must be recognised. The conjuncts are
    re-emitted verbatim, so the rewrite can only narrow: the sign arm folds into the login's
    own conjunct, and the record arm keeps the spare-arm reading it had.
    """
    from src.retrievers.query_guards import enforce_conjunction_dsl

    dsl = {
        "bool": {
            "should": [
                {
                    "bool": {
                        "must": [
                            {"term": {"unitId": "UNIT01"}},
                            {"term": {"actorUserId": "ALOGIN"}},
                        ]
                    }
                },
                {"term": {"actorSign": "0101XY"}},
                {"term": {"recordRef": "REF01"}},
            ]
        }
    }
    out = enforce_conjunction_dsl(
        dsl, ["unitId", "actorUserId"], "src", form_bindings=_SPLIT_USER
    )
    assert out["bool"]["minimum_should_match"] == 1
    assert out["bool"]["should"] == [
        {
            "bool": {
                "must": [
                    {"term": {"unitId": "UNIT01"}},
                    {
                        "bool": {
                            "should": [
                                {"term": {"actorUserId": "ALOGIN"}},
                                {"term": {"actorSign": "0101XY"}},
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                ]
            }
        },
        {"term": {"recordRef": "REF01"}},
    ]


def test_dsl_enforce_conjunction_flattens_ONE_key_arm_and_never_two():
    """Two conjoined arms are two records; flattening both asks for their cross product.

    `(unit_a AND actor_a) OR (unit_b AND actor_b)` per-field becomes
    `(unit_a OR unit_b) AND (actor_a OR actor_b)`, admitting pairings nobody asked for.
    `enforce_value_tuples` owns that shape; this guard leaves multiple records as generated.
    The bare `sign` arm makes this a real bound: two records alone would decline for that
    reason, so the single flatten target is the discriminating case.
    """
    from src.retrievers.query_guards import enforce_conjunction_dsl

    def _record(unit, actor):
        return {
            "bool": {
                "must": [
                    {"term": {"unitId": unit}},
                    {"term": {"actorUserId": actor}},
                ]
            }
        }

    two = {
        "bool": {
            "should": [
                _record("U1", "A1"),
                _record("U2", "A2"),
                {"term": {"actorSign": "0101XY"}},
            ]
        }
    }
    assert (
        enforce_conjunction_dsl(
            two, ["unitId", "actorUserId"], "src", form_bindings=_SPLIT_USER
        )
        == two
    )


def test_dsl_enforce_conjunction_leaves_a_conjoined_arm_it_has_nothing_to_FOLD_into():
    """Two silences, each a way flattening would change a query for no row.

    A group whose only other arms are ones the key does not name is already scoped correctly —
    the conjunction binds none of them, which IS the spare-arm reading — so flattening it would
    rewrite a correct query into an equivalent one and spend a log line saying so. And an arm
    carrying a conjunct on a column the key does not reach must not be decomposed at all: that
    conjunct would become a spare arm bounded by nothing, widening the group by exactly the arm
    it was conjoined to.
    """
    from src.retrievers.query_guards import enforce_conjunction_dsl

    keyed = {
        "bool": {
            "must": [
                {"term": {"unitId": "UNIT01"}},
                {"term": {"actorUserId": "ALOGIN"}},
            ]
        }
    }
    nothing_to_fold = {
        "bool": {"should": [keyed, {"term": {"recordRef": "REF01"}}]}
    }
    assert (
        enforce_conjunction_dsl(
            nothing_to_fold, ["unitId", "actorUserId"], "src", form_bindings=_SPLIT_USER
        )
        == nothing_to_fold
    )

    foreign = {
        "bool": {
            "should": [
                {
                    "bool": {
                        "must": [
                            {"term": {"unitId": "UNIT01"}},
                            {"term": {"docType": "T"}},
                        ]
                    }
                },
                {"term": {"actorSign": "0101XY"}},
            ]
        }
    }
    assert (
        enforce_conjunction_dsl(
            foreign, ["unitId", "actorUserId"], "src", form_bindings=_SPLIT_USER
        )
        == foreign
    )


def test_the_dsl_conjunction_guard_is_HANDED_both_attribution_routes_at_its_call_site():
    """The fold is an input the guard must be GIVEN, and a missing kwarg is a silent no-op.

    Nothing fails without either one: the guard keeps its per-field reading, the query is
    generated, rows come back — more of them than the key asked for. So the one call site that
    reaches this guard is pinned by inspection, the way the four `form_split_bindings` call sites
    already are. Both routes are asserted because neither covers the other's case: the declaration
    survives a literal no lookup recognises, and the literals are the only route to a type bound
    flat to several columns. `form_split_bindings` is also the resolver `_relax_form_conjunction`
    reads, so the two cannot come to disagree about which columns are forms of one type.
    """
    import inspect

    from src.retrievers import kibana_retriever

    src = inspect.getsource(kibana_retriever)
    call = src.index("enforce_conjunction_dsl(")
    end = src.index("\n\n", call)
    tail = src[call:end]
    assert "form_bindings=form_split_bindings(" in tail, tail
    assert "incident_values=incident_values(" in tail, tail


# --- a declared conjunction whose members share one column ------------------------------
#
# Where every key member maps to the same blob column, `required_fields` dedups to one field,
# `resolve_identity_key` falls through, and both standard enforcers bail on arity.
# `same_column_conjunctions` covers this shape with an AND-within-blob rewrite.


def test_same_column_conjunctions_reports_the_collapsed_key():
    """`required_fields` dedups the collapse away; this is the fact it discards."""
    from src.retrievers.query_guards import (required_fields,
                                             same_column_conjunctions)

    field_map = {"office": "ir.text", "user": "ir.text", "ts": "@timestamp"}
    # The premise: the ordinary resolver cannot express this key at all.
    assert required_fields(field_map, ["office", "user"]) == ["ir.text"]
    assert same_column_conjunctions(field_map, require_all_entities=["office", "user"]) == [
        ("ir.text", ["office", "user"])
    ]


def test_same_column_conjunctions_ignores_a_key_spread_over_two_columns():
    """Nothing to do where the ordinary conjunction guard already works."""
    from src.retrievers.query_guards import same_column_conjunctions

    field_map = {"office": "orgUnitId", "user": "sign"}
    assert same_column_conjunctions(field_map, require_all_entities=["office", "user"]) == []


def test_same_column_conjunctions_takes_the_first_satisfied_identity_key():
    """Candidates are priority-ordered ALTERNATIVES: a lower one is not an extra requirement."""
    from src.retrievers.query_guards import same_column_conjunctions

    field_map = {"office": "ir.text", "user": "ir.text", "organization": "ir.text"}
    groups = same_column_conjunctions(
        field_map,
        identity_keys=[["office", "user"], ["organization", "user"]],
        present_types=["office", "user", "organization"],
    )
    assert groups == [("ir.text", ["office", "user"])]


def test_same_column_conjunctions_skips_a_candidate_the_incident_cannot_satisfy():
    from src.retrievers.query_guards import same_column_conjunctions

    field_map = {"office": "ir.text", "user": "ir.text", "organization": "ir.text"}
    groups = same_column_conjunctions(
        field_map,
        identity_keys=[["office", "user"], ["organization", "user"]],
        present_types=["organization", "user"],
    )
    assert groups == [("ir.text", ["organization", "user"])]


def test_enforce_conjunction_same_column_ands_two_entity_types():
    from src.retrievers.query_guards import enforce_conjunction_same_column

    out = enforce_conjunction_same_column(
        'FROM siem-* | WHERE (ir.text LIKE "*AAA1B0980*" OR ir.text LIKE "*0606GK*") '
        "AND @timestamp >= NOW() - 30 days",
        [("ir.text", ["office", "user"])],
        {"office": ["AAA1B0980"], "user": ["0606GK"]},
    )
    assert 'ir.text LIKE "*AAA1B0980*" AND ir.text LIKE "*0606GK*"' in out
    # The window bound is outside the group and must survive untouched.
    assert "@timestamp >= NOW() - 30 days" in out


def test_enforce_conjunction_same_column_keeps_two_forms_of_one_type_ored():
    """Two surface FORMS of one entity type are two spellings of one actor, not two facts.

    AND-ing them asserts both appear in every document — a claim only a measurement on that
    source can make, and where it is false the result is zero rows, which is the failure this
    module exists to prevent. So the rewrite says what the pack declared and nothing more.
    """
    from src.retrievers.query_guards import enforce_conjunction_same_column

    out = enforce_conjunction_same_column(
        'FROM siem-* | WHERE (ir.text LIKE "*AAA1B0980*" OR ir.text LIKE "*0606GK*" '
        'OR ir.text LIKE "*ASURNAME*")',
        [("ir.text", ["office", "user"])],
        {"office": ["AAA1B0980"], "user": ["0606GK", "ASURNAME"]},
    )
    assert (
        'ir.text LIKE "*AAA1B0980*" AND (ir.text LIKE "*0606GK*" '
        'OR ir.text LIKE "*ASURNAME*")'
    ) in out


def test_enforce_conjunction_same_column_declines_a_single_type_group():
    """One required type among the arms is no conjunction — there is nothing to enforce."""
    from src.retrievers.query_guards import enforce_conjunction_same_column

    text = 'FROM siem-* | WHERE (ir.text LIKE "*0606GK*" OR ir.text LIKE "*ASURNAME*")'
    assert (
        enforce_conjunction_same_column(
            text,
            [("ir.text", ["office", "user"])],
            {"office": ["AAA1B0980"], "user": ["0606GK", "ASURNAME"]},
        )
        == text
    )


def test_enforce_conjunction_same_column_declines_an_unlabelled_arm():
    """An arm this cannot account for declines the WHOLE group.

    A partial promotion is the one way this guard could empty a source: under
    `minimum_should_match: 1` a stray arm is optional, and moving it to `must` would make it
    decide the result.
    """
    from src.retrievers.query_guards import enforce_conjunction_same_column

    text = (
        'FROM siem-* | WHERE (ir.text LIKE "*AAA1B0980*" OR ir.text LIKE "*0606GK*" '
        'OR ir.text LIKE "*something-else*")'
    )
    assert (
        enforce_conjunction_same_column(
            text,
            [("ir.text", ["office", "user"])],
            {"office": ["AAA1B0980"], "user": ["0606GK"]},
        )
        == text
    )


def test_enforce_conjunction_same_column_keeps_a_key_foreign_arm_ored_beside_it():
    """An arm carrying a value of a type the KEY does not name is kept, beside the conjunction.

    The pair with the test above is the whole reach of this guard, and the two arms differ in one
    way only: an unlabelled literal is something the generator wrote for reasons nothing here can
    inspect, while this one is a value the incident carried on a column the pack BOUND to its
    type. Declining on it would leave the full OR standing — `office OR sign OR record` — which is
    strictly WIDER than what this returns, so the conservative-looking reading was the losing one.

    It must not become a CONJUNCT: `office AND sign AND record` demands the subject and that record
    co-occur in one document, which is the zero-row rewrite this module exists to avoid.
    """
    from src.retrievers.query_guards import enforce_conjunction_same_column

    text = (
        'FROM siem-* | WHERE (ir.text LIKE "*AAA1B0980*" OR ir.text LIKE "*0606GK*" '
        'OR ir.text LIKE "*ABC123*")'
    )
    out = enforce_conjunction_same_column(
        text,
        [("ir.text", ["office", "user"])],
        {"office": ["AAA1B0980"], "user": ["0606GK"], "record": ["ABC123"]},
    )
    assert out == (
        "FROM siem-* | WHERE (("
        'ir.text LIKE "*AAA1B0980*" AND ir.text LIKE "*0606GK*") '
        'OR ir.text LIKE "*ABC123*")'
    )
    # The conjunction is one alternative, so the spare arm can still match on its own.
    assert ' AND ir.text LIKE "*ABC123*"' not in out


def test_enforce_conjunction_same_column_still_declines_when_a_spare_arm_sits_beside_junk():
    """A spare arm does not license an unlabelled one — the asymmetry has to survive together.

    Reading "some arm was unaccounted for" as "keep it beside the conjunction" would restore
    exactly the width the conjunction removes, on the one input this guard cannot judge.
    """
    from src.retrievers.query_guards import enforce_conjunction_same_column

    text = (
        'FROM siem-* | WHERE (ir.text LIKE "*AAA1B0980*" OR ir.text LIKE "*0606GK*" '
        'OR ir.text LIKE "*ABC123*" OR ir.text LIKE "*something-else*")'
    )
    assert (
        enforce_conjunction_same_column(
            text,
            [("ir.text", ["office", "user"])],
            {"office": ["AAA1B0980"], "user": ["0606GK"], "record": ["ABC123"]},
        )
        == text
    )


def test_dsl_same_column_conjunction_keeps_a_key_foreign_clause_beside_it():
    """The DSL twin of the two above — and here a spare clause changes the SHAPE, not the list.

    With nothing spare the `should` is promoted to `must`. With a spare clause it must STAY a
    `should`: promoting both would AND the conjunction and the spare clause together, which is
    the co-occurrence claim the textual side refuses in the same case. So the bool keeps its
    disjunction and only its arms change — and the `filter` it already carried is untouched,
    because a window or partition bound is not this guard's to move.
    """
    from src.retrievers.query_guards import enforce_conjunction_same_column_dsl

    dsl = {
        "bool": {
            "filter": [{"range": {"timestamp": {"gte": "now-30d"}}}],
            "should": [
                {"wildcard": {"ir.text": "*AAA1B0980*"}},
                {"wildcard": {"ir.text": "*0606GK*"}},
                {"wildcard": {"ir.text": "*ABC123*"}},
            ],
            "minimum_should_match": 1,
        }
    }
    out = enforce_conjunction_same_column_dsl(
        dsl,
        [("ir.text", ["office", "user"])],
        {"office": ["AAA1B0980"], "user": ["0606GK"], "record": ["ABC123"]},
    )
    assert out["bool"]["filter"] == [{"range": {"timestamp": {"gte": "now-30d"}}}]
    assert "must" not in out["bool"]
    assert out["bool"]["minimum_should_match"] == 1
    assert out["bool"]["should"] == [
        {
            "bool": {
                "must": [
                    {"wildcard": {"ir.text": "*AAA1B0980*"}},
                    {"wildcard": {"ir.text": "*0606GK*"}},
                ]
            }
        },
        {"wildcard": {"ir.text": "*ABC123*"}},
    ]


def test_enforce_conjunction_same_column_declines_an_arm_on_another_column():
    """Every arm must sit on the group's own column, or this is not the shape it reads."""
    from src.retrievers.query_guards import enforce_conjunction_same_column

    text = 'FROM siem-* | WHERE (ir.text LIKE "*AAA1B0980*" OR alertType == "0606GK")'
    assert (
        enforce_conjunction_same_column(
            text,
            [("ir.text", ["office", "user"])],
            {"office": ["AAA1B0980"], "user": ["0606GK"]},
        )
        == text
    )


def test_enforce_conjunction_same_column_declines_an_in_list():
    """An `IN` can hold values of two required types, and splitting it would re-quote
    literals this guard has promised to splice verbatim."""
    from src.retrievers.query_guards import enforce_conjunction_same_column

    text = "SELECT * FROM t WHERE (blob IN ('AAA1B0980', '0606GK') OR blob = 'AAA1B0980')"
    assert (
        enforce_conjunction_same_column(
            text,
            [("blob", ["office", "user"])],
            {"office": ["AAA1B0980"], "user": ["0606GK"]},
        )
        == text
    )


def test_enforce_conjunction_same_column_matches_a_declared_path_by_leaf():
    """The pack may declare `ir.text` where the query writes an alias, or the reverse."""
    from src.retrievers.query_guards import enforce_conjunction_same_column

    out = enforce_conjunction_same_column(
        "SELECT * FROM t WHERE (t.ir.text = 'AAA1B0980' OR t.ir.text = '0606GK')",
        [("ir.text", ["office", "user"])],
        {"office": ["AAA1B0980"], "user": ["0606GK"]},
    )
    assert "t.ir.text = 'AAA1B0980' AND t.ir.text = '0606GK'" in out


def test_dsl_same_column_conjunction_promotes_should_grouped_by_type():
    from src.retrievers.query_guards import \
        enforce_conjunction_same_column_dsl

    dsl = {
        "query": {
            "bool": {
                "should": [
                    {"wildcard": {"ir.text": "*AAA1B0980*"}},
                    {"wildcard": {"ir.text": "*0606GK*"}},
                    {"wildcard": {"ir.text": "*ASURNAME*"}},
                ],
                "minimum_should_match": 1,
                "filter": [{"range": {"@timestamp": {"gte": "now-30d"}}}],
            }
        }
    }
    out = enforce_conjunction_same_column_dsl(
        dsl,
        [("ir.text", ["office", "user"])],
        {"office": ["AAA1B0980"], "user": ["0606GK", "ASURNAME"]},
    )
    bool_body = out["query"]["bool"]
    assert "should" not in bool_body
    assert "minimum_should_match" not in bool_body
    # The window bound is a sibling occurrence list and must survive.
    assert bool_body["filter"] == [{"range": {"@timestamp": {"gte": "now-30d"}}}]
    assert bool_body["must"] == [
        {"wildcard": {"ir.text": "*AAA1B0980*"}},
        {
            "bool": {
                "should": [
                    {"wildcard": {"ir.text": "*0606GK*"}},
                    {"wildcard": {"ir.text": "*ASURNAME*"}},
                ],
                # EXPLICIT, and load-bearing: the default is 1 only while the bool carries no
                # `must`, and the promotion above puts one there. Left implicit, both forms of
                # one identity would have to co-occur in a document.
                "minimum_should_match": 1,
            }
        },
    ]


def test_dsl_same_column_conjunction_declines_an_unlabelled_clause():
    from src.retrievers.query_guards import \
        enforce_conjunction_same_column_dsl

    dsl = {
        "bool": {
            "should": [
                {"wildcard": {"ir.text": "*AAA1B0980*"}},
                {"wildcard": {"ir.text": "*0606GK*"}},
                {"wildcard": {"ir.text": "*unrelated*"}},
            ]
        }
    }
    assert (
        enforce_conjunction_same_column_dsl(
            dsl,
            [("ir.text", ["office", "user"])],
            {"office": ["AAA1B0980"], "user": ["0606GK"]},
        )
        == dsl
    )


def test_dsl_same_column_conjunction_declines_a_single_type_group():
    from src.retrievers.query_guards import \
        enforce_conjunction_same_column_dsl

    dsl = {
        "bool": {
            "should": [
                {"wildcard": {"ir.text": "*0606GK*"}},
                {"wildcard": {"ir.text": "*ASURNAME*"}},
            ]
        }
    }
    assert (
        enforce_conjunction_same_column_dsl(
            dsl,
            [("ir.text", ["office", "user"])],
            {"office": ["AAA1B0980"], "user": ["0606GK", "ASURNAME"]},
        )
        == dsl
    )


def test_same_column_guard_is_wired_on_every_retriever():
    """A guard wired on one backend is a silent no-op on the other — the directory's rule.

    Asserted structurally rather than through four mocked retrievals, because what breaks is
    a MISSING call: a new backend that never invokes it turns the pack declaration back into
    the no-op this whole section is about.
    """
    import inspect

    from src.retrievers import (databricks_retriever, elasticsearch_retriever,
                                kibana_retriever, snowflake_retriever)

    for module in (
        databricks_retriever,
        elasticsearch_retriever,
        snowflake_retriever,
        kibana_retriever,
    ):
        src = inspect.getsource(module)
        assert "same_column_conjunctions(" in src, module.__name__
        assert "_enforce_same_column_conjunction(" in src, module.__name__
        # AFTER both strips, so a fabricated or vacuous arm cannot decline the rewrite.
        # `= strip_vacuous` finds the CALL; a bare name would match the import list, which
        # sits above every call and would make this assertion vacuous.
        strip = src.index("= strip_vacuous")
        assert src.index("self._enforce_same_column_conjunction(") > strip, module.__name__
        # ...and BEFORE the identity rewrite, which reads the same OR-groups.
        assert src.index("self._enforce_same_column_conjunction(") < src.index(
            "self._enforce_identity_scope("
        ), module.__name__


# --- the same rule read the other way: one entity type spread across several columns ------
#
# AND-ing a type's value forms across columns requires one row to carry both spellings, which
# deletes every identity that stores theirs on only one. Forms must be OR-ed within the type,
# AND-ed across types.
#
# Measured over one day of one high-volume source: 17,863 of 414,595 identities lost 100% of
# their own rows, read downstream as an actor with no activity and licensed as decisive by
# `key_was_enforced`. The per-identity loss is bimodal — p50 0.00%, p90 98.55% — which is why
# every spot-check passed.

# Two forms of one type on two columns, one column bound to a different type beside them. The
# values are arbitrary strings — nothing here reads them but the rewriter under test.
_SPLIT_FORMS = [("actor", [("form_a", ["actor_sign"]), ("form_b", ["actor_login"])])]
_SPLIT_VALUES = {"actor": ["0202YZ1A", "ALOGIN"], "org_unit": ["UNIT00100"]}


def test_relax_form_conjunction_ors_two_forms_and_keeps_the_rest_anded():
    """The headline fix, and the bound that makes it safe in one assertion.

    The two forms of `actor` become an OR; `org_unit` — a DIFFERENT entity type and a real second
    member of the key — stays AND-ed beside them. A rewrite that widened the whole conjunction
    would answer about other people, which is the defect the section above exists for.
    """
    from src.retrievers.query_guards import relax_form_conjunction

    out = relax_form_conjunction(
        "SELECT * FROM t WHERE unit = 'UNIT00100' AND actor_sign LIKE '0202YZ1A%' "
        "AND actor_login = 'ALOGIN'",
        _SPLIT_FORMS,
        _SPLIT_VALUES,
        "s",
    )
    assert out == (
        "SELECT * FROM t WHERE unit = 'UNIT00100' "
        "AND (actor_sign LIKE '0202YZ1A%' OR actor_login = 'ALOGIN')"
    )


def test_relax_form_conjunction_reaches_a_parenthesised_and_chain():
    """The live shape: the identity conjunction sits one level down, inside a paren group.

    A generator that correctly ORs the subject against a record arm writes
    `(unit = U AND sign = S AND login = L) OR record = R`, so a walker that only looked at the
    top-level AND chain would find one conjunct and decline — the defect intact on the query that
    carries it.
    """
    from src.retrievers.query_guards import relax_form_conjunction

    out = relax_form_conjunction(
        "SELECT * FROM t WHERE (unit = 'UNIT00100' AND actor_login = 'ALOGIN' "
        "AND actor_sign LIKE '0202YZ1A%') OR rec = 'X'",
        _SPLIT_FORMS,
        _SPLIT_VALUES,
        "s",
    )
    assert out == (
        "SELECT * FROM t WHERE (unit = 'UNIT00100' "
        "AND (actor_login = 'ALOGIN' OR actor_sign LIKE '0202YZ1A%')) OR rec = 'X'"
    )


def test_relax_form_conjunction_refuses_across_a_union():
    """The one refusal this guard needs, and the hazard it walls off is REACHABLE.

    This is the only guard in the module that MOVES text: the merged arms leave their positions and
    reappear inside one group. Where the AND chain spans a `UNION ALL`, the branch boundary is glued
    to one conjunct — so most such queries decline by accident, that conjunct never parsing as a
    flat arm. Not this one: the sign arm and the login arm both parse, one per branch, and merging
    them would hoist the login into branch A and leave branch B scanning unbounded. Still valid
    SQL, which is what makes it worth a test rather than a comment.
    """
    from src.retrievers.query_guards import relax_form_conjunction

    q = (
        "SELECT a FROM t WHERE a = 1 AND actor_sign = '0202YZ1A' AND c = 3 "
        "UNION ALL SELECT a FROM u WHERE d = 4 AND actor_login = 'ALOGIN'"
    )
    assert relax_form_conjunction(q, _SPLIT_FORMS, _SPLIT_VALUES, "s") == q


def test_relax_form_conjunction_refuses_a_negated_group():
    """OR-ing two negations INVERTS them, and the reachable shape is a negated GROUP.

    `NOT a AND NOT b` is `NOT (a OR b)`, so a merge sweeping in an exclusion widens the result to
    nearly every row instead of to the subject's own. At the conjunct level that cannot happen —
    the arm reader's five operators are all positive, so every flat negated spelling below fails to
    parse and is skipped. Asserting only those would be asserting a redundancy: removing the check
    that once sat there killed nothing.

    The hazard is one level up, and it is the whole reason this test exists. `NOT (a AND b)` holds
    ONE conjunct at depth zero, so the query declines — and then the region walker offers the
    parenthesised body, whose two conjuncts parse cleanly, and the OR lands UNDER the negation.
    """
    from src.retrievers.query_guards import relax_form_conjunction

    negated_group = (
        "SELECT * FROM t WHERE NOT (actor_sign = '0202YZ1A' AND actor_login = 'ALOGIN')"
    )
    assert relax_form_conjunction(negated_group, _SPLIT_FORMS, _SPLIT_VALUES, "s") == (
        negated_group
    )
    # ...and the flat spellings, which the arm reader refuses on its own. Kept as a bound: an
    # operator set widened to `!=` or `NOT LIKE` would make the guard invert a query silently.
    for negated in ("actor_login NOT LIKE 'ALOGIN%'", "actor_login != 'ALOGIN'"):
        q = f"SELECT * FROM t WHERE actor_sign = '0202YZ1A' AND {negated}"
        assert relax_form_conjunction(q, _SPLIT_FORMS, _SPLIT_VALUES, "s") == q, negated


def test_relax_form_conjunction_requires_the_incidents_own_values():
    """A merge is licensed by the VALUE, not by the column.

    Two predicates on two form columns are only two spellings of one subject when each literal is
    one of the incident's values for that type. Without the test, a generator's invented literal
    would be OR-ed into the identity and made a legitimate way to match a row.
    """
    from src.retrievers.query_guards import relax_form_conjunction

    q = "SELECT * FROM t WHERE actor_sign = 'ZZZZZZ' AND actor_login = 'ALOGIN'"
    assert relax_form_conjunction(q, _SPLIT_FORMS, _SPLIT_VALUES, "s") == q


def test_relax_form_conjunction_declines_one_form_alone():
    """One form on its own column is already correct — several values of it OR inside an IN list.

    Nothing here is about how wide a query is; it is about two columns being read as two facts. One
    form is one fact.
    """
    from src.retrievers.query_guards import relax_form_conjunction

    q = "SELECT * FROM t WHERE unit = 'UNIT00100' AND actor_sign = '0202YZ1A'"
    assert relax_form_conjunction(q, _SPLIT_FORMS, _SPLIT_VALUES, "s") == q
    # And a source that declares no split forms at all is a byte-for-byte no-op.
    assert relax_form_conjunction(q, [], _SPLIT_VALUES, "s") == q


def test_relax_form_conjunction_leaves_an_unparseable_conjunct_in_place():
    """Failing to parse a conjunct must NOT decline the region — the one asymmetry with the OR side.

    Every real query's first conjunct carries the whole `SELECT ... FROM ... WHERE` prefix and
    parses as nothing at all, so a reader that declined on one unlabelled conjunct would decline on
    every query ever generated. The assertion is that the unreadable neighbour survives verbatim
    beside the merge — a rewrite that dropped or re-quoted it would be worse than no rewrite.
    """
    from src.retrievers.query_guards import relax_form_conjunction

    out = relax_form_conjunction(
        "SELECT * FROM t WHERE ts BETWEEN '2026-01-01' AND '2026-01-31' "
        "AND actor_sign = '0202YZ1A' AND actor_login = 'ALOGIN'",
        _SPLIT_FORMS,
        _SPLIT_VALUES,
        "s",
    )
    assert "ts BETWEEN '2026-01-01'" in out
    assert "(actor_sign = '0202YZ1A' OR actor_login = 'ALOGIN')" in out


def test_relax_form_conjunction_declines_a_column_two_forms_claim():
    """A column claimed by two forms is a pack defect, and the honest degradation is to do nothing.

    A column cannot store two non-interchangeable forms at once, so the declaration cannot be
    trusted to say the two arms are one subject — and merging on it could OR two genuinely
    different people. `form_split_bindings` filters this at the pack seam; the guard refuses it
    again, because the two halves are reached by different callers.
    """
    from src.retrievers.query_guards import relax_form_conjunction

    ambiguous = [("actor", [("form_a", ["blob"]), ("form_b", ["blob", "actor_login"])])]
    q = "SELECT * FROM t WHERE blob = '0202YZ1A' AND actor_login = 'ALOGIN'"
    assert relax_form_conjunction(q, ambiguous, _SPLIT_VALUES, "s") == q


def test_relax_form_conjunction_matches_a_declared_path_by_leaf():
    """The pack may declare `a.b` where the query writes `t.a.b`, or the reverse."""
    from src.retrievers.query_guards import relax_form_conjunction

    out = relax_form_conjunction(
        "SELECT * FROM t WHERE t.who.sign = '0202YZ1A' AND t.who.login = 'ALOGIN'",
        [("actor", [("form_a", ["who.sign"]), ("form_b", ["login"])])],
        _SPLIT_VALUES,
        "s",
    )
    assert out == (
        "SELECT * FROM t WHERE (t.who.sign = '0202YZ1A' OR t.who.login = 'ALOGIN')"
    )


def test_relax_form_conjunction_relaxes_every_split_type_not_the_first():
    """Two split types in one chain, and relaxing one is a fix that reads like a fix in the log.

    A source may bind two types across forms — a party and the unit that party belongs to is the
    ordinary pair — and each is an independent OR. The two groups must not merge with each other:
    that would put a unit value and an actor value in one disjunction, which answers about other
    people again.
    """
    from src.retrievers.query_guards import relax_form_conjunction

    out = relax_form_conjunction(
        "SELECT * FROM t WHERE actor_sign = '0202YZ1A' AND rec_a = 'R1' "
        "AND actor_login = 'ALOGIN' AND rec_b = 'R2'",
        _SPLIT_FORMS + [("record_id", [("f1", ["rec_a"]), ("f2", ["rec_b"])])],
        dict(_SPLIT_VALUES, record_id=["R1", "R2"]),
        "s",
    )
    assert out == (
        "SELECT * FROM t WHERE (actor_sign = '0202YZ1A' OR actor_login = 'ALOGIN') "
        "AND (rec_a = 'R1' OR rec_b = 'R2')"
    )


def test_relax_form_conjunction_dsl_moves_the_forms_into_a_should():
    """The same rule on the Query DSL route — a guarantee honoured on one transport is not one.

    A DSL `must` IS the AND, so the fix is a nested `should` with an EXPLICIT
    `minimum_should_match: 1`: the default is 1 only while the bool carries no `must`, and the
    surviving `org_unit` clause puts one there. Left implicit, both forms would have to co-occur
    again — the defect restored by a default. `filter` must survive untouched: that is where the
    mandatory window and the slice live.
    """
    from src.retrievers.query_guards import relax_form_conjunction_dsl

    dsl = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"actor_sign": "0202YZ1A"}},
                    {"term": {"actor_login": "ALOGIN"}},
                    {"term": {"unit": "UNIT00100"}},
                ],
                "filter": [{"range": {"ts": {"gte": "now-30d"}}}],
            }
        }
    }
    out = relax_form_conjunction_dsl(dsl, _SPLIT_FORMS, _SPLIT_VALUES, "s")
    body = out["query"]["bool"]
    assert body["filter"] == [{"range": {"ts": {"gte": "now-30d"}}}]
    assert body["must"] == [
        {"term": {"unit": "UNIT00100"}},
        {
            "bool": {
                "should": [
                    {"term": {"actor_sign": "0202YZ1A"}},
                    {"term": {"actor_login": "ALOGIN"}},
                ],
                "minimum_should_match": 1,
            }
        },
    ]
    # The input is not mutated — the caller's dict is still the query it generated.
    assert len(dsl["query"]["bool"]["must"]) == 3


def test_relax_form_conjunction_dsl_collapses_a_must_holding_nothing_else():
    """Nothing else mandatory, so the group becomes this bool's own `should` rather than nesting.

    A one-element `must` wrapping a `should` is the same query; it is not the shape a reviewer of
    the published DSL recognises, and the published query is what an operator reads.
    """
    from src.retrievers.query_guards import relax_form_conjunction_dsl

    out = relax_form_conjunction_dsl(
        {
            "bool": {
                "must": [
                    {"term": {"actor_sign": "0202YZ1A"}},
                    {"term": {"actor_login": "ALOGIN"}},
                ]
            }
        },
        _SPLIT_FORMS,
        _SPLIT_VALUES,
        "s",
    )
    assert out == {
        "bool": {
            "should": [
                {"term": {"actor_sign": "0202YZ1A"}},
                {"term": {"actor_login": "ALOGIN"}},
            ],
            "minimum_should_match": 1,
        }
    }


def test_relax_form_conjunction_dsl_reads_must_only_and_recurses_past_a_merge():
    """Two bounds that fail in opposite directions, which is why they share a test.

    `must_not` is an EXCLUSION and this guard's merge writes into `must`, so a walker that
    collected exclusions would make the excluded values MANDATORY — the widest inversion
    available here. The pure-exclusion bool inside `filter` is the direct statement of that: read,
    its two forms collapse into a `should` and "exclude both of these" becomes "match either".

    And a nested `bool` can carry the real defect one level down — an earlier draft returned as
    soon as the outer bool merged and left every sibling occurrence list unvisited, which is the
    half-wiring this section's other tests exist to catch, one level down instead of one backend
    over.
    """
    from src.retrievers.query_guards import relax_form_conjunction_dsl

    exclusion = {
        "bool": {
            "must_not": [
                {"term": {"actor_sign": "0202YZ1A"}},
                {"term": {"actor_login": "ALOGIN"}},
            ]
        }
    }
    out = relax_form_conjunction_dsl(
        {
            "bool": {
                "must": [
                    {"term": {"actor_sign": "0202YZ1A"}},
                    {"term": {"actor_login": "ALOGIN"}},
                ],
                "must_not": [
                    {"term": {"actor_sign": "0202YZ1A"}},
                    {"term": {"actor_login": "ALOGIN"}},
                ],
                "filter": [
                    {
                        "bool": {
                            "must": [
                                {"term": {"who.sign": "0202YZ1A"}},
                                {"term": {"who.login": "ALOGIN"}},
                            ]
                        }
                    },
                    exclusion,
                ],
            }
        },
        _SPLIT_FORMS + [("actor", [("form_a", ["who.sign"]), ("form_b", ["who.login"])])],
        _SPLIT_VALUES,
        "s",
    )
    # BOTH forms sit in `must_not` beside a `must` carrying the same pair. Asserting only that
    # `must_not` came back unchanged proves nothing — a walker that lifts an exclusion into the
    # AND leaves it there untouched — so the discriminating assertion is on the merged group:
    # exactly the two arms that were in `must`, and no third.
    assert out["bool"]["must_not"] == [
        {"term": {"actor_sign": "0202YZ1A"}},
        {"term": {"actor_login": "ALOGIN"}},
    ]
    assert "must" not in out["bool"]
    assert out["bool"]["should"] == [
        {"term": {"actor_sign": "0202YZ1A"}},
        {"term": {"actor_login": "ALOGIN"}},
    ]
    assert out["bool"]["minimum_should_match"] == 1
    assert out["bool"]["filter"][0]["bool"] == {
        "should": [
            {"term": {"who.sign": "0202YZ1A"}},
            {"term": {"who.login": "ALOGIN"}},
        ],
        "minimum_should_match": 1,
    }
    # A bool holding NOTHING but the two exclusions is byte-identical: there is no `must` to
    # collapse, so a walker reading exclusions turns this bool into `should: [both]` and the
    # exclusion becomes the match. Spelled out rather than compared to `exclusion`, which is the
    # object that went IN — an in-place rewrite would satisfy that comparison against itself.
    assert out["bool"]["filter"][1] == {
        "bool": {
            "must_not": [
                {"term": {"actor_sign": "0202YZ1A"}},
                {"term": {"actor_login": "ALOGIN"}},
            ]
        }
    }


def test_form_relax_guard_is_wired_on_every_retriever_after_the_or_readers():
    """A guard wired on one backend is a silent no-op on the other, and the ORDER is the fix.

    Both halves are load-bearing and neither is a comment. Run this BEFORE
    `_enforce_same_column_conjunction`, and that guard reads the `(form_a OR form_b)` this one just
    wrote as a same-column identity group and flattens it back into the AND — the defect restored
    by the fix's own neighbour. Run it AFTER the three below, and each has parenthesised or
    re-shaped the body into a conjunct this one cannot parse.
    """
    import inspect

    from src.retrievers import (databricks_retriever, elasticsearch_retriever,
                                kibana_retriever, snowflake_retriever)

    for module in (
        databricks_retriever,
        elasticsearch_retriever,
        snowflake_retriever,
        kibana_retriever,
    ):
        src = inspect.getsource(module)
        # `form_split_bindings(` finds the CALL; the bare name would match the import list.
        assert "form_split_bindings(" in src, module.__name__
        call = src.index("self._relax_form_conjunction(")
        for earlier in (
            "self._enforce_same_column_conjunction(",
            "self._enforce_identity_scope(",
        ):
            assert src.index(earlier) < call, f"{module.__name__} runs {earlier} too late"
        for later in ("self._enforce_value_tuples(", "self._enforce_subject_anchor("):
            assert call < src.index(later), f"{module.__name__} runs {later} too early"
        # The stem widening is a method on one retriever and a bare call on the other three, so
        # match the ASSIGNMENT either way rather than picking one spelling and passing vacuously on
        # the backends that use the other.
        widen = re.search(r"=\s*(?:self\._)?widen_stem_literals", src)
        assert widen, module.__name__
        assert call < widen.start(), f"{module.__name__} widens stems too early"


def test_form_split_bindings_returns_only_a_type_split_over_several_columns():
    """The pack seam, and its two filters are the whole selection rather than housekeeping.

    A type qualifies on TWO forms resolving to TWO distinct columns. One form has nothing to
    relax. Two forms on the SAME column are the other shape entirely — there the values already OR
    inside one `IN` list and any conjunction between them belongs to
    `enforce_conjunction_same_column`. A flat binding, the ordinary case, is not a form map at all,
    so a pack declaring no `value_forms` anywhere makes the guard a byte-for-byte no-op.
    """
    from src.retrievers.field_mapping import form_split_bindings

    pack = MagicMock()
    pack.source.return_value = SimpleNamespace(
        entity_bindings={
            "actor": {"form_a": ["actor_sign"], "form_b": ["actor_login"]},
            "one_form": {"form_a": ["only"]},
            "same_column": {"form_a": ["blob"], "form_b": "blob"},
            "org_unit": ["unit"],
        }
    )
    pack.form_bindings_for.side_effect = lambda etype, name: {
        k: (v if isinstance(v, list) else [v])
        for k, v in (pack.source.return_value.entity_bindings.get(etype) or {}).items()
    }
    assert form_split_bindings(pack, "records") == [
        ("actor", [("form_a", ["actor_sign"]), ("form_b", ["actor_login"])])
    ]
    # No pack, no source, and a mock without the accessor all degrade to nothing rather than
    # raising — a retriever built without a pack still has to run.
    assert form_split_bindings(None, "records") == []
    assert form_split_bindings(pack, "") == []
    missing = MagicMock()
    missing.source.return_value = None
    assert form_split_bindings(missing, "records") == []


@pytest.mark.asyncio
async def test_kibana_strips_evidence_filter_from_generated_dsl():
    config = {
        "name": "auth_events",
        "url": "https://kibana.example.net",
        "username": "u",
        "password": "p",
        "index": "auth*",
        "max_results": 10,
        "field_schema": "robot",
        "never_filter": ["userInfo.robot"],
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=EsQueryDsl(
            query={
                "bool": {
                    "filter": [
                        {"range": {"ts": {"gte": "now-1d"}}},
                        {"term": {"robot": False}},
                    ]
                }
            }
        )
    )
    retriever = KibanaRetriever(config, llm)
    retriever._search = AsyncMock(return_value={"hits": {"hits": []}})
    with patch(
        "src.retrievers.kibana_retriever.map_entities", AsyncMock(return_value={})
    ):
        await retriever.retrieve(_query("auth_events"))

    body = retriever._search.call_args.args[0]
    assert body["query"]["bool"]["filter"] == [{"range": {"ts": {"gte": "now-1d"}}}]


# --- the sibling key that describes a list an earlier guard deleted ----------------------
#
# `minimum_should_match` describes the `should` list a stripper just removed; it survives on
# any `bool` still carrying a `filter` and makes the query unsatisfiable. Measured on one index
# and window: `filter` alone 108 documents, the same `filter` + `minimum_should_match: 1` and no
# `should` 0, an explicit `should: []` beside it 0. The run reports that as a source that
# answered with nothing.


def test_an_orphaned_minimum_should_match_is_DROPPED_wherever_the_should_went():
    """The live shape: a `bool` keeping its `filter` after its last `should` arm was stripped.

    Asserted in both spellings of the same fact — no `should` key at all, and an explicit empty
    list — because a guard that deletes the key it emptied produces the first and a generator
    can write the second, and only one of them is reachable from inside this module.
    """
    from src.retrievers.query_guards import drop_orphan_minimum_should_match

    gone = {
        "bool": {
            "filter": [{"range": {"ts": {"gte": "2026-07-22"}}}, {"term": {"type": "session_anomaly"}}],
            "minimum_should_match": 1,
        }
    }
    out = drop_orphan_minimum_should_match(gone, "svc_alerts")
    assert "minimum_should_match" not in out["bool"]
    assert out["bool"]["filter"] == gone["bool"]["filter"], "the surviving filter moved"

    explicit = {
        "bool": {
            "filter": [{"term": {"type": "session_anomaly"}}],
            "should": [],
            "minimum_should_match": 1,
        }
    }
    out = drop_orphan_minimum_should_match(explicit, "svc_alerts")
    assert "minimum_should_match" not in out["bool"]
    # The empty list goes with it: it constrains nothing, and leaving it invites the next
    # reader to treat the bool as one that has a disjunction.
    assert "should" not in out["bool"]


def test_a_populated_should_keeps_its_minimum_should_match_untouched():
    """The no-op direction, which is most of every real query.

    This guard removes a requirement, so the one way it can be wrong is removing one that was
    doing work — a `should` of two identity arms under `minimum_should_match: 1` is the shape
    half the guards in this module WRITE, and dropping the key there would admit every document
    the filter alone admits. A single clause written directly rather than in a list is populated
    too, which is the spelling a hand-written body uses.
    """
    import copy

    from src.retrievers.query_guards import drop_orphan_minimum_should_match

    listed = {
        "bool": {
            "filter": [{"range": {"ts": {"gte": "2026-07-22"}}}],
            "should": [{"term": {"sign": "0202YZ"}}, {"term": {"login": "RSTUVW"}}],
            "minimum_should_match": 1,
        }
    }
    assert drop_orphan_minimum_should_match(copy.deepcopy(listed), "s") == listed

    bare = {
        "bool": {
            "should": {"term": {"sign": "0202YZ"}},
            "minimum_should_match": 1,
        }
    }
    assert drop_orphan_minimum_should_match(copy.deepcopy(bare), "s") == bare


def test_the_orphan_is_found_inside_a_NESTED_bool_and_not_only_at_the_root():
    """Every guard here writes nested groups, so the root is the one place it cannot only look.

    The outer bool is legitimate — a real `should` of two arms — and the orphan sits on an inner
    one whose own disjunction was emptied. A root-only fix passes the test above and leaves the
    live defect intact wherever a group is one level down, which is where these guards put them.
    """
    from src.retrievers.query_guards import drop_orphan_minimum_should_match

    dsl = {
        "bool": {
            "should": [
                {"term": {"office": "BBB1C03EF"}},
                {
                    "bool": {
                        "filter": [{"term": {"org": "1A"}}],
                        "minimum_should_match": 1,
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }
    out = drop_orphan_minimum_should_match(dsl, "s")
    assert out["bool"]["minimum_should_match"] == 1, "the outer requirement was real"
    inner = out["bool"]["should"][1]["bool"]
    assert "minimum_should_match" not in inner
    assert inner["filter"] == [{"term": {"org": "1A"}}]


@pytest.mark.asyncio
async def test_kibana_publishes_no_orphaned_minimum_should_match_end_to_end():
    """The WIRING, which is the half of this fix a comment cannot make true.

    The generator writes exactly the live shape — a `filter` plus one `should` arm on an
    evidence-only field — so the pack's `never_filter` declaration is what empties the `should`,
    and the orphan is left behind by a guard doing its job. Asserted on the body handed to
    `_search`, because that is the only artifact the backend sees, and reaching it needs the
    whole chain in the order the retriever runs it.
    """
    config = {
        "name": "svc_alerts",
        "url": "https://kibana.example.net",
        "username": "u",
        "password": "p",
        "index": "alerts*",
        "max_results": 10,
        "field_schema": "ir.userRemarks",
        "never_filter": ["ir.userRemarks"],
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=EsQueryDsl(
            query={
                "bool": {
                    "filter": [{"range": {"ts": {"gte": "now-30d"}}}],
                    "should": [{"term": {"ir.userRemarks": "session_anomaly"}}],
                    "minimum_should_match": 1,
                }
            }
        )
    )
    retriever = KibanaRetriever(config, llm)
    retriever._search = AsyncMock(return_value={"hits": {"hits": []}})
    with patch(
        "src.retrievers.kibana_retriever.map_entities", AsyncMock(return_value={})
    ):
        await retriever.retrieve(_query("svc_alerts"))

    published = retriever._search.call_args.args[0]["query"]
    assert "should" not in published["bool"], "the evidence filter survived the strip"
    assert "minimum_should_match" not in published["bool"], (
        "the key describing the deleted `should` reached the backend: it requires one match "
        "from an empty set, which matches NO document (measured 108 rows -> 0 live)"
    )
    assert published["bool"]["filter"], "the window bound was lost with it"


def test_merge_endpoint_attaches_query_guards_for_every_backend():
    """The guarantees ride on the SOURCE, so they must reach an Elasticsearch config too —
    not only the databricks branch that first implemented them."""
    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)

    class _Src:
        name = "auth_events"
        endpoints = {"kind": "elasticsearch", "cluster": "c1", "indices": ["auth*"]}
        query_hints = ""
        default_filters = {}
        require_all_entities = ["org_unit", "user"]
        never_filter = ["value.payload.userInfo.robot"]

    backends = {
        "elasticsearch": {
            "c1": {"url": "http://es:9200", "username": "u", "password": "p"}
        }
    }
    src = _Src()
    merged = engine._merge_endpoint(src, "elasticsearch", backends)
    engine._attach_query_guards(src, merged)
    assert merged["require_all_entities"] == ["org_unit", "user"]
    assert merged["never_filter"] == ["value.payload.userInfo.robot"]


def test_default_filters_reach_every_backend_and_not_only_elasticsearch():
    """The mandatory slice is a property of the source, not the backend.

    `default_filters` was in `_merge_endpoint`'s elasticsearch branch only; the Databricks
    and Snowflake retrievers called `enforce_default_filters` on an always-empty
    `config["default_filters"]`. This pins the guarantee on every backend by wiring
    `default_filters` through `_attach_query_guards` alongside the other source properties.

    Note the direction, which is the opposite of most failures on this page: a
    `databricks_uc` view declaring `application_phase: PRD` shipped a two-arm UNION with that
    conjunct on neither arm, so the result comes back FULL, carrying the non-production rows
    the slice exists to exclude, and the row cap then truncates the real ones to make room.
    """
    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)

    class _Src:
        name = "raw_access"
        query_hints = ""
        default_filters = {"application_phase": "PRD"}
        require_all_entities = []
        never_filter = ["application_phase"]

    class _Databricks(_Src):
        endpoints = {
            "kind": "databricks_uc",
            "workspace": "w1",
            "catalog": "cat",
            "schema": "sch",
            "tables": ["a_view", "b_view"],
        }

    class _Snowflake(_Src):
        endpoints = {"kind": "snowflake", "account": "acc1", "objects": ["T"]}

    backends = {
        "databricks": {"w1": {"warehouse_id": "wh", "workspace_url": "https://w1"}},
        "snowflake": {"acc1": {"account": "acc1", "user": "u", "password": "p"}},
    }
    for src, source_type in ((_Databricks(), "databricks"), (_Snowflake(), "snowflake")):
        merged = engine._merge_endpoint(src, source_type, backends)
        engine._attach_query_guards(src, merged)
        assert merged["default_filters"] == {"application_phase": "PRD"}, source_type
        # The pin and the strip are OPPOSITE keys over one column and both must arrive: the
        # strip lifts the constant out of the generator's OR-group (where it satisfies the
        # group on its own and the subject scoping becomes decoration), the pin AND-s it back
        # exactly once. Either alone leaves the query answering the wrong question.
        assert merged["never_filter"] == ["application_phase"], source_type
    # A source declaring nothing gets an empty map rather than a missing key, since every
    # retriever reads it unconditionally.
    bare = MagicMock()
    bare.default_filters = None
    merged = {}
    LogRetrievalEngine._attach_query_guards(bare, merged)
    assert merged["default_filters"] == {}


def test_rest_never_filter_drops_evidence_clause():
    """ServiceNow's encoded query is built in code (`^` is already AND, so no conjunction
    rewrite is needed), but an evidence field must still not become a filter clause."""
    from src.models.pydantic_models import ExtractedEntity
    from src.retrievers.rest_retriever import RestRetriever

    retriever = RestRetriever(
        {"name": "servicenow", "base_url": "https://sn", "never_filter": ["robot"]},
        MagicMock(),
    )
    query = RetrievalQuery(
        target_log_source="servicenow",
        natural_language_query="incidents",
        date_from="2026-07-17",
        date_to="2026-07-17",
        entities=[
            ExtractedEntity(type="org_unit", value="A"),
            ExtractedEntity(type="robot_flag", value="false"),
        ],
    )
    encoded = retriever._build_sysparm_query(
        {"org_unit": "org_unit_id", "robot_flag": "robot"}, query
    )
    assert "org_unit_id=A" in encoded
    assert "robot" not in encoded


def test_rest_field_schema_flattens_a_form_bound_binding():
    """A binding is a list of columns OR a `{value form: [column]}` mapping, and iterating
    the mapping yields the FORM NAMES.

    The schema hint tells the mapper which fields this table HAS; which form a column stores
    is not its question. Offering `servicenow`/`record` as candidate fields advertises columns
    the backend does not have while hiding the real ones — and the mapping prompt is told to
    infer field names, so it writes the plausible-looking one and the predicate matches
    nothing. Both shapes must flatten to columns, since one entity type being form-bound must
    not change how the flat bindings beside it are read.
    """
    from src.retrievers.rest_retriever import RestRetriever

    class _Src:
        entity_bindings = {
            "incident_ir": {"servicenow": ["number"], "record": ["ir.recordId"]},
            "org_unit": ["company", "u_office"],
            "user": "assigned_to",
        }

    pack = MagicMock()
    pack.source.return_value = _Src()
    retriever = RestRetriever(
        {"name": "servicenow", "base_url": "https://sn"}, MagicMock(), pack
    )
    fields = [f.strip() for f in retriever._field_schema().split(",")]

    assert fields == ["number", "ir.recordId", "company", "u_office", "assigned_to"]
    # The form names are not columns and must never be offered as candidates.
    assert "servicenow" not in fields
    assert "record" not in fields


# --- partition pruning is backend-agnostic ---------------------------------------------
#
# A timeout on an unbounded partition is indistinguishable downstream from "the source had
# nothing". Partition columns are discovered from metadata; the pack declares only what
# metadata cannot say (a VIEW hides its layout; no catalog carries `role`/`pad_days`).


def test_partition_bounds_pads_the_window_on_both_sides():
    """The partition column is rarely the column the incident is about.

    A record created before the alert and modified after it must stay in range, so the window
    is padded — a bound that stops at the event date silently drops the record's latest
    versions, i.e. exactly the current state that decides whether a document is still live.
    """
    from src.retrievers.query_guards import partition_bounds

    (bound,) = partition_bounds(
        [{"name": "modification_date", "type": "DATE"}], "2026-07-27", "2026-07-28"
    )
    assert bound["kind"] == "range"
    assert bound["low"] == "2026-07-26" and bound["high"] == "2026-07-29"
    # pad_days is declarable per source.
    (wide,) = partition_bounds(
        [{"name": "modification_date", "type": "DATE", "pad_days": 3}],
        "2026-07-27",
        "2026-07-27",
    )
    assert wide["low"] == "2026-07-24" and wide["high"] == "2026-07-30"


def test_partition_bounds_pads_the_append_side_asymmetrically_when_declared():
    """A record-version partition column needs a LONGER pad on the upper side.

    `modification_date` is when the ROW was written, not when the event happened, and those
    are different clocks: the event is fixed in the past while the record keeps accruing
    versions, the latest of which is the subject's CURRENT state.

    Measured on incident 83e94dd6 / record SUBJ04: event window 2026-07-09, symmetric pad of 1,
    39 of 51 versions returned. Version 39 (modification_date 07-11) carries a SECOND record
    split by a DIFFERENT agent, and version 50 (07-13) is the record's current version. Nothing
    errored -- 39 rows is a plausible number, so the verdict was rendered from a truncated
    record while reading as fully evidenced.
    """
    from src.retrievers.query_guards import partition_bounds

    (bound,) = partition_bounds(
        [
            {
                "name": "modification_date",
                "type": "DATE",
                "pad_days": 3,
                "pad_days_after": 21,
            }
        ],
        "2026-07-09",
        "2026-07-09",
    )
    assert bound["low"] == "2026-07-06", bound
    assert bound["high"] == "2026-07-30", bound
    # Absent, it stays symmetric -- no existing source changes behaviour.
    (sym,) = partition_bounds(
        [{"name": "modification_date", "type": "DATE", "pad_days": 3}],
        "2026-07-09",
        "2026-07-09",
    )
    assert sym["low"] == "2026-07-06" and sym["high"] == "2026-07-12"
    # A junk value falls back to the symmetric pad rather than dropping the bound: an
    # unbounded scan on a partitioned table is the same wrong answer by a slower route.
    (junk,) = partition_bounds(
        [
            {
                "name": "modification_date",
                "type": "DATE",
                "pad_days": 3,
                "pad_days_after": "soon",
            }
        ],
        "2026-07-09",
        "2026-07-09",
    )
    assert junk["low"] == "2026-07-06" and junk["high"] == "2026-07-12"


def test_partition_bounds_enumerates_calendar_part_layouts():
    """A `year=/month=/day=` layout is bounded by ENUMERATING the parts the window spans."""
    from src.retrievers.query_guards import partition_bounds

    specs = [
        {"name": "year", "role": "year", "type": "STRING"},
        {"name": "month", "role": "month", "type": "STRING"},
        {"name": "day", "role": "day", "type": "STRING"},
    ]
    # 2026-07-31 padded by 1 day spans 31 JUL -> 01 AUG, i.e. two months and two days.
    bounds = {b["name"]: b for b in partition_bounds(specs, "2026-07-31", "2026-07-31")}
    assert bounds["year"]["values"] == ["2026"]
    assert bounds["month"]["values"] == ["07", "08"]
    assert bounds["day"]["values"] == ["30", "31", "01"]
    # Zero-padded for STRING columns, bare for numeric ones (that is the stored value).
    numeric = partition_bounds(
        [{"name": "day", "role": "day", "type": "INT"}], "2026-07-05", "2026-07-05"
    )
    assert numeric[0]["values"] == ["4", "5", "6"]


def test_partition_bounds_skips_unusable_specs():
    from src.retrievers.query_guards import partition_bounds

    assert partition_bounds([{"name": "d"}], None, None) == []
    assert partition_bounds([{"type": "DATE"}], "2026-07-27", "2026-07-27") == []


def test_a_numeric_partition_column_with_no_declared_ROLE_is_left_unbounded(caplog):
    """A date window on a numeric column renders as arithmetic: `version >= 2026-07-17`
    becomes `>= 2002` — matching nothing. Discovery reports name and type but not meaning,
    so the default `date` role lands on every numeric partition. Skipping it is the
    asymmetric choice: unbounded costs storage scanned; contradictory costs every row.
    """
    from src.retrievers.query_guards import partition_bounds, partition_predicates

    for col_type in ("INT", "BIGINT", "int", "DECIMAL(10,0)", "NUMERIC"):
        assert (
            partition_bounds(
                [{"name": "version", "type": col_type}], "2026-07-18", "2026-08-17"
            )
            == []
        ), col_type
    # ...and an explicitly declared `date` role does not license it either: a numeric time
    # column is an EPOCH, which has its own declaration precisely because the unit and the era
    # cannot be read off the number.
    assert (
        partition_bounds(
            [{"name": "ts", "type": "BIGINT", "role": "date"}],
            "2026-07-18",
            "2026-08-17",
        )
        == []
    )
    # The refusal names both declarations that WOULD make the column bindable, because the
    # remedy differs and the operator cannot tell which applies from an empty result.
    with caplog.at_level(logging.WARNING):
        partition_predicates(
            [{"name": "version", "type": "INT"}], "2026-07-18", "2026-08-17"
        )
    warned = " ".join(r.getMessage() for r in caplog.records)
    assert "version" in warned and "epoch_time_columns" in warned
    assert "role: year|month|day" in warned

    # The two shapes that must be UNAFFECTED, because each is a real layout and skipping it
    # would trade this defect for the unbounded-scan one the bounds exist to prevent:
    # a numeric column whose role IS declared as a calendar part still enumerates...
    (part,) = partition_bounds(
        [{"name": "day", "role": "day", "type": "INT"}], "2026-07-05", "2026-07-05"
    )
    assert part["kind"] == "in" and part["values"] == ["4", "5", "6"]
    # ...and a date/timestamp/string column still gets its range, quoted for its own type.
    rendered = {
        p["name"]: p["predicate"]
        for p in partition_predicates(
            [
                {"name": "d", "type": "DATE"},
                {"name": "t", "type": "TIMESTAMP"},
                {"name": "s", "type": "STRING"},
                {"name": "u", "type": ""},
            ],
            "2026-07-27",
            "2026-07-27",
        )
    }
    assert rendered["d"].startswith("d >= DATE'2026-07-26'")
    assert rendered["t"].startswith("t >= TIMESTAMP'2026-07-26'")
    assert rendered["s"].startswith("s >= '2026-07-26'")
    # An UNTYPED spec is the pack's own declaration for a view metadata cannot describe, and
    # it must keep defaulting to DATE — reading "no type" as "not a date" would silently
    # disable every declared partition on every source that omits the key.
    assert rendered["u"].startswith("u >= DATE'2026-07-26'")


def test_partition_predicates_render_per_dialect():
    from src.retrievers.query_guards import partition_predicates

    specs = [
        {"name": "modification_date", "type": "DATE"},
        {"name": "day", "role": "day", "type": "STRING"},
    ]
    sql = {
        p["name"]: p["predicate"]
        for p in partition_predicates(specs, "2026-07-27", "2026-07-27")
    }
    assert sql["modification_date"] == (
        "modification_date >= DATE'2026-07-26' AND modification_date <= DATE'2026-07-28'"
    )
    assert sql["day"] == "day IN ('26', '27', '28')"
    # ES|QL has no DATE'...' literal and uses double quotes.
    esql = {
        p["name"]: p["predicate"]
        for p in partition_predicates(specs, "2026-07-27", "2026-07-27", dialect="esql")
    }
    assert 'modification_date >= "2026-07-26"' in esql["modification_date"]
    assert esql["day"] == 'day IN ("26", "27", "28")'


def test_enforce_partition_bounds_ands_onto_an_existing_where():
    from src.retrievers.query_guards import enforce_partition_bounds

    out = enforce_partition_bounds(
        "SELECT a FROM t WHERE locator = 'X' OR locator = 'Y' ORDER BY a",
        [{"name": "modification_date", "type": "DATE"}],
        "2026-07-27",
        "2026-07-27",
    )
    # The existing body is PARENTHESISED so its top-level OR cannot be re-associated.
    assert "AND (locator = 'X' OR locator = 'Y')" in out
    assert "modification_date >= DATE'2026-07-26'" in out
    assert out.rstrip().endswith("ORDER BY a")


def test_enforce_partition_bounds_adds_a_where_when_there_is_none():
    from src.retrievers.query_guards import enforce_partition_bounds

    out = enforce_partition_bounds(
        "SELECT a FROM t LIMIT 10",
        [{"name": "d", "type": "DATE"}],
        "2026-07-27",
        "2026-07-27",
    )
    assert (
        out
        == "SELECT a FROM t WHERE d >= DATE'2026-07-26' AND d <= DATE'2026-07-28' LIMIT 10"
    )


def test_enforce_partition_bounds_leaves_an_already_bounded_column_alone():
    """The generator may bound it more tightly than the padded window would."""
    from src.retrievers.query_guards import enforce_partition_bounds

    sql = "SELECT a FROM t WHERE modification_date = DATE'2026-07-27'"
    assert (
        enforce_partition_bounds(
            sql,
            [{"name": "modification_date", "type": "DATE"}],
            "2026-07-27",
            "2026-07-27",
        )
        == sql
    )


def test_a_partition_column_is_not_read_as_bounded_by_a_LONGER_column_name():
    """`_is_constrained` matched the column name as a suffix, so the bound was never injected.

    A false positive here means no bound is added and the query full-scans; the timeout reads
    downstream as INSUFFICIENT DATA. Two real shapes: column `d` was matched by `<anything>d`,
    and `date` by `creation_date`. The regex needs a left boundary; the right is pinned by the
    comparison operator.
    """
    from src.retrievers.query_guards import enforce_partition_bounds

    # `d` must NOT be considered bounded by a comparison on `record_id`.
    sql = "SELECT a FROM t WHERE record_id == 'X'"
    out = enforce_partition_bounds(
        sql, [{"name": "d", "type": "DATE"}], "2026-07-27", "2026-07-27"
    )
    assert out != sql and "d >=" in out.replace("`", "")

    # `date` must NOT be considered bounded by `creation_date`.
    sql2 = "SELECT a FROM t WHERE creation_date > DATE'2026-01-01'"
    out2 = enforce_partition_bounds(
        sql2, [{"name": "date", "type": "DATE"}], "2026-07-27", "2026-07-27"
    )
    assert out2 != sql2, "a partition column read as bounded by a longer column name"

    # And the genuine cases still short-circuit: an exact match, and a qualified one.
    for bounded in (
        "SELECT a FROM t WHERE d >= DATE'2026-07-01'",
        "SELECT a FROM t WHERE td.date >= DATE'2026-07-01'",
        "SELECT a FROM t WHERE `td`.`date` >= DATE'2026-07-01'",
    ):
        field = "d" if " d >=" in bounded else "date"
        assert (
            enforce_partition_bounds(
                bounded, [{"name": field, "type": "DATE"}], "2026-07-27", "2026-07-27"
            )
            == bounded
        ), f"re-bounded an already-bounded column: {bounded}"


def test_enforce_partition_bounds_refuses_to_guess_with_a_subquery():
    """Two WHEREs means picking the right one is guesswork — leave it alone (and log)."""
    from src.retrievers.query_guards import enforce_partition_bounds

    sql = "SELECT a FROM t WHERE id IN (SELECT id FROM u WHERE x = 1)"
    assert (
        enforce_partition_bounds(
            sql, [{"name": "d", "type": "DATE"}], "2026-07-27", "2026-07-27"
        )
        == sql
    )


def test_enforce_partition_bounds_esql_inserts_a_stage_after_the_source():
    from src.retrievers.query_guards import enforce_partition_bounds

    out = enforce_partition_bounds(
        'FROM idx | WHERE user == "A" | LIMIT 5',
        [{"name": "d", "type": "date"}],
        "2026-07-27",
        "2026-07-27",
        dialect="esql",
    )
    assert out.startswith('FROM idx | WHERE d >= "2026-07-26" AND d <= "2026-07-28"')
    assert '| WHERE user == "A"' in out and "| LIMIT 5" in out


def test_enforce_partition_bounds_dsl_nests_the_original_query():
    from src.retrievers.query_guards import enforce_partition_bounds_dsl

    original = {"term": {"record": "SUBJ01"}}
    out = enforce_partition_bounds_dsl(
        original, [{"name": "d", "type": "DATE"}], "2026-07-27", "2026-07-27"
    )
    clauses = out["bool"]["filter"]
    assert {"range": {"d": {"gte": "2026-07-26", "lte": "2026-07-28"}}} in clauses
    assert original in clauses


def test_partition_clauses_encoded_skips_already_bounded_fields():
    """ServiceNow's `^` is already AND and the clauses are built in code, so there is
    nothing to rewrite — only clauses to add."""
    from src.retrievers.query_guards import partition_clauses_encoded

    clauses = partition_clauses_encoded(
        [{"name": "sys_created_on", "type": "DATE"}, {"name": "d", "type": "DATE"}],
        "2026-07-27",
        "2026-07-27",
        already=["sys_created_on"],
    )
    assert all("sys_created_on" not in c for c in clauses)
    assert any(c.startswith("d>=") for c in clauses)


def test_merge_partition_specs_prefers_discovery_and_layers_the_declaration():
    """Discovery says WHICH columns partition the table (it cannot go stale); the pack
    supplies what no catalog carries (`role`, `pad_days`) and covers what discovery cannot
    see (a VIEW hides its underlying table's layout)."""
    from src.retrievers.query_guards import merge_partition_specs

    merged = {
        s["name"]: s
        for s in merge_partition_specs(
            [
                {"name": "day", "type": "STRING"},
                {"name": "modification_date", "type": "DATE"},
            ],
            [
                {"name": "day", "role": "day", "pad_days": 0},
                {"name": "year", "role": "year"},
            ],
        )
    }
    assert merged["day"]["type"] == "STRING"  # kept from discovery
    assert merged["day"]["role"] == "day"  # supplied by the pack
    assert merged["day"]["pad_days"] == 0
    assert "year" in merged  # declared-only column still bounds
    assert merged["modification_date"]["type"] == "DATE"


def test_partition_prompt_line_is_empty_without_a_partition():
    """Derived from the discovered/declared specs, so no source hand-writes the same prose
    into query_hints — where it would drift out of date with the physical layout."""
    from src.retrievers.query_guards import partition_prompt_line

    assert partition_prompt_line([]) == ""
    line = partition_prompt_line(
        [{"name": "modification_date", "type": "DATE"}], "2026-07-27", "2026-07-27"
    )
    assert "modification_date" in line and "PARTITION" in line.upper()
    assert "DATE'2026-07-26'" in line


def test_partitions_from_columns_reads_the_partition_index():
    """Databricks reports the layout in information_schema.columns.partition_index —
    ordered, and NULL for a non-partition column."""
    from src.retrievers.databricks_retriever import DatabricksRetriever

    specs = DatabricksRetriever._partitions_from_columns(
        [
            {"column_name": "locator", "data_type": "STRING", "partition_index": None},
            {"column_name": "day", "data_type": "STRING", "partition_index": 2},
            {"column_name": "year", "data_type": "STRING", "partition_index": 0},
            {"column_name": "month", "data_type": "STRING", "partition_index": 1},
        ]
    )
    assert [s["name"] for s in specs] == ["year", "month", "day"]
    assert all(s["type"] == "STRING" for s in specs)


def test_merge_endpoint_attaches_partition_columns_for_every_backend():
    """`partition_columns` is a property of the SOURCE's storage, so it must reach every
    retriever config — a pack field honoured on only some backends is the same class of
    silent no-op the guard fields exist to prevent."""
    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)

    class _Src:
        name = "settlement_report"
        endpoints = {
            "kind": "elasticsearch",
            "cluster": "c1",
            "indices": ["audit_trail*"],
        }
        query_hints = ""
        default_filters = {}
        require_all_entities = []
        never_filter = []
        partition_columns = [{"name": "year", "role": "year", "type": "STRING"}]

    backends = {
        "elasticsearch": {
            "c1": {"url": "http://es:9200", "username": "u", "password": "p"}
        }
    }
    src = _Src()
    merged = engine._merge_endpoint(src, "elasticsearch", backends)
    engine._attach_query_guards(src, merged)
    assert merged["partition_columns"] == [
        {"name": "year", "role": "year", "type": "STRING"}
    ]


@pytest.mark.asyncio
async def test_elasticsearch_bounds_a_declared_partition_column():
    """ES has no discoverable partition metadata, but the field belongs to the SOURCE — a
    source with a partition-like scoping column must not go unpruned because of where it
    happens to live."""
    from src.models.pydantic_models import FieldMapping

    config = {
        "name": "audit_like",
        "url": "http://es:9200",
        "username": "u",
        "password": "p",
        "index": "audit_trail",
        "field_schema": "d date, record keyword",
        "partition_columns": [{"name": "d", "type": "date"}],
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        side_effect=[
            FieldMapping(mappings=[]),
            EsqlQuery(query='FROM audit_trail | WHERE record == "SUBJ01"'),
        ]
    )
    retriever = ElasticsearchRetriever(config, llm)
    fake_client = MagicMock()
    fake_client.esql.query = AsyncMock(return_value={"columns": [], "values": []})
    with patch(
        "src.retrievers.elasticsearch_retriever.AsyncElasticsearch",
        return_value=fake_client,
    ):
        await retriever.retrieve(_query("audit_like"))

    executed = fake_client.esql.query.call_args.kwargs["query"]
    assert 'd >= "2023-12-31"' in executed and 'd <= "2024-01-06"' in executed
    assert 'record == "SUBJ01"' in executed


@pytest.mark.asyncio
async def test_databricks_bounds_the_discovered_partition_column():
    """The end-to-end guarantee on the backend that discovers its own layout: the generator
    bounds the column the REQUEST talks about (creation_date), and the retriever bounds the
    one the TABLE is laid out by (modification_date) on top. Prompting alone does not fix
    this — the two columns are different, and only one of them prunes."""
    retriever = DatabricksRetriever(
        {
            "name": "record_lake",
            "workspace_url": "https://workspace/",
            "warehouse_id": "wh1",
            "api_key_env": "DATABRICKS_TOKEN",
            "catalog": "cat",
            "schema": "src_tables",
        },
        MagicMock(),
    )
    # Stand in for the information_schema round-trip: `modification_date` is the partition.
    retriever._get_field_schema = AsyncMock(
        return_value="record_table_4(locator STRING)"
    )
    retriever._discovered_partitions = DatabricksRetriever._partitions_from_columns(
        [
            {"column_name": "locator", "data_type": "STRING", "partition_index": None},
            {
                "column_name": "modification_date",
                "data_type": "DATE",
                "partition_index": 0,
            },
        ]
    )
    retriever._execute_sql = AsyncMock(return_value=[])
    retriever.llm_client.structured_output = AsyncMock(
        return_value=SqlQuery(
            query="SELECT locator FROM record_table_4 WHERE creation_date >= DATE'2024-01-01'"
        )
    )
    with patch(
        "src.retrievers.databricks_retriever.map_entities", AsyncMock(return_value={})
    ):
        await retriever.retrieve(_query("record_lake"))

    executed = retriever._execute_sql.call_args.args[0]
    assert "modification_date >= DATE'2023-12-31'" in executed
    assert "modification_date <= DATE'2024-01-06'" in executed
    assert "creation_date >= DATE'2024-01-01'" in executed
    # The generator is also TOLD, so it can bound the column more tightly than the padded
    # window — enforcement is the floor, not a replacement for asking.
    prompt = retriever.llm_client.structured_output.call_args.args[0][0]["content"]
    assert "modification_date" in prompt


@pytest.mark.asyncio
async def test_databricks_falls_back_to_the_declared_partition_for_a_view():
    """A VIEW reports no partitions — it hides its underlying table's layout — so the pack's
    declaration is the only thing that can prune it. `role: day` means the layout stores
    calendar PARTS, which are bounded by enumeration rather than a range."""
    retriever = DatabricksRetriever(
        {
            "name": "settlement_report",
            "workspace_url": "https://workspace/",
            "warehouse_id": "wh1",
            "api_key_env": "DATABRICKS_TOKEN",
            "catalog": "cat",
            "schema": "gdpr_audit_trail",
            "partition_columns": [
                {"name": "year", "role": "year", "type": "STRING"},
                {"name": "month", "role": "month", "type": "STRING"},
                {"name": "day", "role": "day", "type": "STRING"},
            ],
        },
        MagicMock(),
    )
    retriever._get_field_schema = AsyncMock(
        return_value="audit_trail(relatedRecord STRING)"
    )
    retriever._discovered_partitions = []  # a VIEW: metadata reports nothing
    retriever._execute_sql = AsyncMock(return_value=[])
    retriever.llm_client.structured_output = AsyncMock(
        return_value=SqlQuery(
            query="SELECT relatedRecord FROM audit_trail WHERE relatedRecord = 'X'"
        )
    )
    with patch(
        "src.retrievers.databricks_retriever.map_entities", AsyncMock(return_value={})
    ):
        await retriever.retrieve(_query("settlement_report"))

    executed = retriever._execute_sql.call_args.args[0]
    # The padded window (2023-12-31 .. 2024-01-06) crosses year, month AND day boundaries,
    # so each part is enumerated over what the window actually spans — bounding only the
    # incident's own two dates would drop the partitions either side of midnight.
    assert "year IN ('2023', '2024')" in executed
    assert "month IN ('12', '01')" in executed
    assert "day IN ('31', '01', '02', '03', '04', '05', '06')" in executed
    assert "relatedRecord = 'X'" in executed


# --- epoch time windows are backend-agnostic too ----------------------------------------
#
# A wrong epoch literal is a valid predicate matching nothing; it reads downstream as "the
# source had no data". The pack declares `{name, unit}` only; the window is computed from
# the pipeline's `YYYY-MM-DD` dates, never stated as a literal that can go stale.


def test_epoch_window_covers_both_end_days_completely():
    from src.retrievers.query_guards import epoch_window

    low, high = epoch_window("2026-07-27", "2026-07-27")
    assert low == 1785110400000  # 2026-07-27T00:00:00Z
    assert high == 1785196799999  # 2026-07-27T23:59:59.999Z
    # The real document that returned nothing sits inside this window.
    assert low <= 1785170060721 <= high
    # Seconds and microseconds scale the same way; the unit is the only difference.
    assert epoch_window("2026-07-27", "2026-07-27", "seconds") == (
        1785110400,
        1785196799,
    )
    assert epoch_window("2026-07-27", "2026-07-27", "s")[0] == 1785110400
    # An unrecognised unit is refused rather than guessed at.
    assert epoch_window("2026-07-27", "2026-07-27", "fortnights") is None


def test_epoch_prompt_line_computes_this_incidents_window():
    """The prompt gets the numbers ALREADY CONVERTED, which is what makes a hand-written
    example in query_hints unnecessary — and the one that existed was wrong by a year.
    """
    from src.retrievers.query_guards import epoch_prompt_line

    line = epoch_prompt_line(
        [{"name": "timestamp", "unit": "milliseconds"}], "2026-07-27", "2026-07-27"
    )
    assert "1785110400000" in line and "1785196799999" in line
    assert "do NOT copy an epoch value from any example" in line
    # Nothing declared -> nothing said.
    assert epoch_prompt_line([], "2026-07-27", "2026-07-27") == ""


def test_enforce_epoch_window_repairs_a_bound_from_the_wrong_year():
    """The measured defect: the generator asked about a window 365 days before the data."""
    from src.retrievers.query_guards import enforce_epoch_window

    specs = [{"name": "timestamp", "unit": "milliseconds"}]
    out = enforce_epoch_window(
        "SELECT * FROM s WHERE timestamp >= 1753574400000 AND timestamp <= 1753660799999",
        specs,
        "2026-07-27",
        "2026-07-27",
        "app_session_events",
    )
    assert "timestamp >= 1785110400000" in out
    assert "timestamp <= 1785196799999" in out
    assert "1753574400000" not in out


def test_enforce_epoch_window_leaves_an_overlapping_bound_alone():
    """A generator that tightened to a few hours around the alert wrote a BETTER query than
    the padded window would be, so an overlapping interval is never widened."""
    from src.retrievers.query_guards import enforce_epoch_window

    specs = [{"name": "timestamp", "unit": "milliseconds"}]
    sql = "SELECT * FROM s WHERE timestamp >= 1785168000000 AND timestamp <= 1785171600000"
    assert enforce_epoch_window(sql, specs, "2026-07-27", "2026-07-27") == sql
    # ...and a source with nothing declared is never touched at all.
    assert enforce_epoch_window(sql, [], "2026-07-27", "2026-07-27") == sql


def test_enforce_epoch_window_converts_a_date_literal_on_an_epoch_column():
    """An ISO literal cannot compare against an integer column at all, so this shape is
    always wrong and always converted — the upper bound keeps the whole end day."""
    from src.retrievers.query_guards import enforce_epoch_window

    out = enforce_epoch_window(
        'FROM s | WHERE timestamp >= "2026-07-27" AND timestamp <= "2026-07-27"',
        [{"name": "timestamp", "unit": "milliseconds"}],
        "2026-07-27",
        "2026-07-27",
        "s",
        dialect="esql",
    )
    assert "timestamp >= 1785110400000" in out
    assert "timestamp <= 1785196799999" in out
    assert '"2026-07-27"' not in out


def test_enforce_epoch_window_matches_a_dotted_column_on_its_leaf():
    """A generated query may reference the field bare, dotted or through an array path."""
    from src.retrievers.query_guards import enforce_epoch_window

    out = enforce_epoch_window(
        "SELECT * FROM s WHERE appSchemeAlerts.date BETWEEN 1753574400000 AND 1753660799999",
        [{"name": "appSchemeAlerts.date", "unit": "milliseconds"}],
        "2026-07-27",
        "2026-07-27",
    )
    assert "BETWEEN 1785110400000 AND 1785196799999" in out


def test_dsl_epoch_repair_fixes_the_range_inside_bool_filter():
    """The shape the defect was measured in. The range sits in bool.filter, so it is AND-ed
    with everything — a wrong one zeroes the result no matter what the should clauses hit.
    """
    from src.retrievers.query_guards import enforce_epoch_window_dsl

    dsl = {
        "bool": {
            "filter": [
                {"range": {"timestamp": {"gte": 1753574400000, "lte": 1753660799999}}}
            ],
            "should": [{"term": {"orgUnitId": "ORGUNIT01"}}],
            "minimum_should_match": 1,
        }
    }
    out = enforce_epoch_window_dsl(
        dsl,
        [{"name": "timestamp", "unit": "milliseconds"}],
        "2026-07-27",
        "2026-07-27",
        "app_session_events",
    )
    assert out["bool"]["filter"][0]["range"]["timestamp"] == {
        "gte": 1785110400000,
        "lte": 1785196799999,
    }
    # Only the range is rewritten; the entity evidence is left exactly as generated.
    assert out["bool"]["should"] == [{"term": {"orgUnitId": "ORGUNIT01"}}]
    assert out["bool"]["minimum_should_match"] == 1
    # The input is not mutated.
    assert dsl["bool"]["filter"][0]["range"]["timestamp"]["gte"] == 1753574400000


def test_dsl_epoch_repair_converts_date_literals_and_drops_their_format_key():
    """`format`/`time_zone` describe a date literal that is no longer there once the bound is
    an integer, so leaving them behind would make the clause self-contradictory."""
    from src.retrievers.query_guards import enforce_epoch_window_dsl

    out = enforce_epoch_window_dsl(
        {
            "range": {
                "timestamp": {
                    "gte": "2026-07-27",
                    "lte": "2026-07-27",
                    "format": "yyyy-MM-dd",
                }
            }
        },
        [{"name": "timestamp", "unit": "milliseconds"}],
        "2026-07-27",
        "2026-07-27",
    )
    assert out["range"]["timestamp"] == {"gte": 1785110400000, "lte": 1785196799999}
    assert "format" not in out["range"]["timestamp"]


def test_dsl_epoch_repair_leaves_an_overlapping_range_and_other_columns_alone():
    from src.retrievers.query_guards import enforce_epoch_window_dsl

    dsl = {
        "bool": {
            "filter": [
                {"range": {"timestamp": {"gte": 1785168000000, "lte": 1785171600000}}},
                # A different column, undeclared: not this guard's business.
                {"range": {"ingestedAt": {"gte": 1, "lte": 2}}},
            ]
        }
    }
    assert (
        enforce_epoch_window_dsl(
            dsl,
            [{"name": "timestamp", "unit": "milliseconds"}],
            "2026-07-27",
            "2026-07-27",
        )
        == dsl
    )


@pytest.mark.asyncio
async def test_kibana_repairs_a_wrong_year_epoch_window_end_to_end():
    """Live reproduction: the generator emits the 2025 window the stale pack example taught
    it, and the executed query asks about 2026 anyway."""
    config = {
        "name": "app_session_events",
        "url": "https://kibana.example.net",
        "username": "u",
        "password": "p",
        "index": "session.azure.ic*",
        "epoch_time_columns": [{"name": "timestamp", "unit": "milliseconds"}],
    }
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=EsQueryDsl(
            query={
                "bool": {
                    "filter": [
                        {
                            "range": {
                                "timestamp": {
                                    "gte": 1753574400000,
                                    "lte": 1753660799999,
                                }
                            }
                        }
                    ],
                    "should": [{"term": {"user.userId": "JKLMNO"}}],
                    "minimum_should_match": 1,
                }
            }
        )
    )
    retriever = KibanaRetriever(config, llm)
    retriever._discovered_schema = "timestamp, user.userId"
    real = _FakeResp(
        {"rawResponse": {"hits": {"hits": [{"_source": {"alertId": "a"}}]}}}
    )
    session = MagicMock()
    session.closed = False
    session.post = MagicMock(side_effect=[real])
    retriever._session = session

    query = RetrievalQuery(
        target_log_source="app_session_events",
        natural_language_query="SCHEME session for JKLMNO",
        scope_id="ORGUNIT01",
        actor_id="JKLMNO",
        date_from="2026-07-27",
        date_to="2026-07-27",
    )
    with patch(
        "src.retrievers.kibana_retriever.map_entities", AsyncMock(return_value={})
    ):
        await retriever.retrieve(query)

    executed = session.post.call_args.kwargs["json"]["params"]["body"]["query"]
    assert executed["bool"]["filter"][0]["range"]["timestamp"] == {
        "gte": 1785110400000,
        "lte": 1785196799999,
    }
    # The generator was also TOLD the converted window — enforcement is the floor, not a
    # substitute for asking, exactly as with the partition bounds.
    prompt = llm.structured_output.call_args.args[0][0]["content"]
    assert "1785110400000" in prompt


def test_attach_query_guards_carries_epoch_columns_to_every_backend():
    """Which retriever runs for this source is decided purely by whether the cluster's creds
    carry `gateway: kibana`. A guarantee that holds on one route and not the other is not a
    guarantee — the same reason never_filter/require_all_entities are attached here."""
    from src.knowledge.pack import SourceDef

    engine = LogRetrievalEngine.__new__(LogRetrievalEngine)
    src = SourceDef(
        name="app_session_events",
        epoch_time_columns=[{"name": "timestamp", "unit": "milliseconds"}],
    )
    merged = {}
    engine._attach_query_guards(src, merged)
    assert merged["epoch_time_columns"] == [
        {"name": "timestamp", "unit": "milliseconds"}
    ]


# ── identity_keys must reach EVERY backend, or it is not a guarantee ──────────


def test_identity_keys_ride_the_guard_seam_to_every_retriever_config():
    """A key enforced on one backend route but not another is not a key.

    The same pack source is reachable via ES|QL or a Kibana DSL gateway depending only on
    which creds are configured, so `_attach_query_guards` is the one seam every retriever
    config passes and every source-level guarantee must ride it. This is the check that a
    new backend branch cannot silently turn the pack declaration into a no-op.
    """
    from src.knowledge.pack import SourceDef
    from src.log_retrieval import LogRetrievalEngine

    src = SourceDef(
        name="s",
        require_all_entities=["org_unit", "user"],
        identity_keys=[["org_unit", "user"], ["organization", "user"]],
    )
    merged = {}
    LogRetrievalEngine._attach_query_guards(src, merged)
    assert merged["identity_keys"] == [["org_unit", "user"], ["organization", "user"]]
    # All five source-level guarantees ride together.
    for key in (
        "require_all_entities",
        "identity_keys",
        "never_filter",
        "partition_columns",
        "epoch_time_columns",
    ):
        assert key in merged

    # A source declaring none still gets empty lists, never a missing key.
    bare = {}
    LogRetrievalEngine._attach_query_guards(SourceDef(name="b"), bare)
    assert bare["identity_keys"] == []


def test_every_retriever_reads_identity_keys_off_its_config():
    """Each backend must pick the key up, or the pack declaration dies at the boundary."""
    from src.retrievers.databricks_retriever import DatabricksRetriever
    from src.retrievers.elasticsearch_retriever import ElasticsearchRetriever
    from src.retrievers.kibana_retriever import KibanaRetriever
    from src.retrievers.snowflake_retriever import SnowflakeRetriever

    keys = [["org_unit", "user"], ["organization", "user"]]
    cfg = {
        "name": "s",
        "identity_keys": keys,
        # minimal per-backend connection settings
        "url": "https://es.invalid:9200",
        "index": "i",
        "workspace_url": "https://dbx.invalid",
        "warehouse_id": "abc",
        "api_key": "t",
        "catalog": "c",
        "schema": "s",
        "tables": ["t"],
        "account": "a",
        "user": "u",
        "password": "p",
        "database": "d",
        "warehouse": "w",
        "kibana_url": "https://kb.invalid",
    }
    for cls in (
        ElasticsearchRetriever,
        KibanaRetriever,
        DatabricksRetriever,
        SnowflakeRetriever,
    ):
        r = cls(dict(cfg), llm_client=MagicMock())
        assert r.identity_keys == keys, f"{cls.__name__} dropped identity_keys"


def test_unresolved_placeholders_flags_a_quoted_template_marker():
    """A quoted `'<...>'` in a FINISHED query is the silent-0-rows shape.

    Job c6c11f89 ran `WHERE OFFICE_NAME = '<UNKNOWN>' AND CODE IN ('UST','NDC')` against a
    473,085-office reference table. Syntactically valid, so the backend ran it happily and
    matched nothing — and "0 rows" is indistinguishable downstream from "the source had
    nothing to say": the stage reported SUCCESS, the conditions reading that source went
    `unknown`, and the office whose row was sitting there all along went unestablished.
    """
    from src.retrievers.base import unresolved_placeholders

    assert unresolved_placeholders(
        "SELECT OFFICE_NAME, CODE FROM ref.office_profile "
        "WHERE OFFICE_NAME = '<UNKNOWN>' AND CODE IN ('UST','NDC')"
    ) == ["'<UNKNOWN>'"]
    # The pack's own hint spelling, and a double-quoted variant.
    assert unresolved_placeholders("WHERE OFFICE_NAME = '<office>'") == ["'<office>'"]
    assert unresolved_placeholders('WHERE a = "<x>"') == ['"<x>"']


def test_unresolved_placeholders_sees_a_placeholder_inside_a_like_wildcard():
    """A prefix match writes its skeleton as `LIKE '<sign>%'`, and the wildcard used to
    put that literal outside a whole-literal match — so a source whose every predicate is
    a prefix match had NO detection at all.

    Measured on job ca4240c0, verbatim: the office half was caught and the sign half was
    not, and a hint spelling both sides with LIKE would have been caught on neither.
    """
    from src.retrievers.base import unresolved_placeholders

    assert unresolved_placeholders(
        "SELECT * FROM record WHERE creator.office_id = '<unitId>' "
        "AND creator.sign.red LIKE '<sign>%'"
    ) == ["'<unitId>'", "'<sign>%'"]
    # Wildcards on either side, and the single-character wildcard too.
    assert unresolved_placeholders("WHERE a LIKE '%<x>%'") == ["'%<x>%'"]
    assert unresolved_placeholders("WHERE a LIKE '<x>_'") == ["'<x>_'"]


def test_unresolved_placeholders_ignores_a_real_value_beside_a_wildcard():
    """ONLY wildcards may surround the placeholder, or the widening would swallow every
    prefix match ever generated. A literal carrying a real value is a real predicate.
    """
    from src.retrievers.base import unresolved_placeholders

    assert unresolved_placeholders("WHERE creator.sign.red LIKE '6006FF%'") == []
    # A real value beside a bracketed fragment is still not "nothing but a placeholder".
    assert unresolved_placeholders("WHERE a LIKE 'AB<x>%'") == []


def test_unresolved_placeholders_does_not_fire_on_comparison_operators():
    """Only the QUOTED form counts: `<` and `<>` are ordinary SQL, not placeholders."""
    from src.retrievers.base import unresolved_placeholders

    assert (
        unresolved_placeholders(
            "SELECT * FROM t WHERE OFFICE_NAME = 'NNN1P15CD' AND n < 5 AND a <> b "
            "AND d >= DATE'2026-08-01'"
        )
        == []
    )
    assert unresolved_placeholders("") == []
    assert unresolved_placeholders(None) == []


@pytest.mark.asyncio
async def test_publish_query_warns_on_an_unresolved_placeholder(caplog):
    """The check rides `publish_query` — the one seam every backend passes with its
    final text, after every guard — so it cannot be half-wired across five retrievers.
    """
    import logging

    retriever = _never_filter_retriever(
        "SELECT OFFICE_NAME FROM ref.office_profile WHERE OFFICE_NAME = '<UNKNOWN>'"
    )
    with caplog.at_level(logging.WARNING, logger="src.retrievers.base"):
        await retriever.retrieve(_auth_query())

    assert "unresolved placeholder" in caplog.text
    assert "'<UNKNOWN>'" in caplog.text
    # It WARNS, it does not raise: the operator can now see why the source came back
    # empty, and the query still ran.
    assert retriever._execute_sql.await_count == 1


# --- a fabricated literal: one entity type's value on another type's column ---------------
#
# The shape neither `unresolved_placeholders` nor `render_filters` can see: a real literal,
# in a valid predicate, on a column the source's `entity_bindings` bind to a different type.
# Every test below names which of the two declarations (`entity_bindings` vs extracted entity
# type) it is asserting, because only a contradiction between them is dropped.

# A locator-keyed source that binds the locator NESTED and binds no actor column at all —
# the narrow shape the guard most has to work on, since the fewer columns a source declares
# the more freely a generator invents against it.
_LOCATOR_BINDINGS = {
    "record": ["locator.primary"],
    "time_window": ["creation_date", "creation_date_time"],
}
# A richer source: the locator as a STRUCT, and party structs bound by two types each, so a
# sub-field inside one of them is legitimately a different type from its parent.
_PARTY_BINDINGS = {
    "record": ["locator"],
    "actor": {"sign": ["creator", "owner"]},
    "org_unit": ["creator", "owner", "customer_code"],
}


def test_fabricated_predicate_dropped_when_the_column_holds_another_type():
    """The measured defect: the actor's login on the record locator's column.

    Job 0184a3ce ran `WHERE locator.primary = '<the actor login>'` against a locator-keyed
    source. The incident had extracted NO entity of the locator's type at all, so no such
    value existed anywhere in the run — the literal was carried across from the actor. The
    query was valid, the backend ran it, it matched nothing, the stage reported success.
    """
    from src.retrievers.query_guards import strip_fabricated_predicates

    out = strip_fabricated_predicates(
        "SELECT * FROM records WHERE creation_date >= DATE'2026-08-01' "
        "AND locator.primary = 'RSTUVW'",
        _LOCATOR_BINDINGS,
        {"actor": ["RSTUVW", "0202YZ"]},
    )
    assert "RSTUVW" not in out
    # The date bound — the query's only real scope — survives untouched.
    assert "creation_date >= DATE'2026-08-01'" in out


def test_fabricated_predicate_dropped_on_a_field_inside_a_bound_struct():
    """The pack may name the STRUCT while the generator writes a field inside it.

    This is why the guard matches the whole path and not the leaf: `record: [locator]` says
    nothing about a segment called `primary`, so a leaf test asks about a name the pack never
    mentioned, finds no claim, and fails open on the very shape it exists for.
    """
    from src.retrievers.query_guards import strip_fabricated_predicates

    out = strip_fabricated_predicates(
        "SELECT * FROM records WHERE d = 1 AND locator.primary = 'RSTUVW'",
        _PARTY_BINDINGS,
        {"actor": ["RSTUVW"]},
    )
    assert "RSTUVW" not in out and "d = 1" in out


def test_a_bound_struct_does_not_claim_every_field_inside_it():
    """The converse, and the reason the enclosing reading is not subtree-wide.

    `actor: {sign: [owner]}` binds the party struct by the party it identifies; that struct
    still carries the party's OWN office, which the pack declares under a sibling parent. A
    predicate putting the incident's org unit on `owner.customer_code` is correct, and a guard
    reading "everything under an actor struct is an actor" would delete it.
    """
    from src.retrievers.query_guards import strip_fabricated_predicates

    sql = "SELECT * FROM records WHERE d = 1 AND owner.customer_code = 'BBB1C03EF'"
    assert (
        strip_fabricated_predicates(sql, _PARTY_BINDINGS, {"org_unit": ["BBB1C03EF"]})
        == sql
    )


def test_fabricated_guard_keeps_the_right_value_on_the_right_column():
    """A locator value on the locator's column is the predicate the source is FOR."""
    from src.retrievers.query_guards import strip_fabricated_predicates

    sql = "SELECT * FROM records WHERE d = 1 AND locator.primary = 'ABC123'"
    assert (
        strip_fabricated_predicates(sql, _LOCATOR_BINDINGS, {"record": ["ABC123"]})
        == sql
    )
    # ...and via a table alias the pack never wrote, which is the same column.
    aliased = "SELECT * FROM records t WHERE d = 1 AND t.locator.primary = 'ABC123'"
    assert (
        strip_fabricated_predicates(aliased, _LOCATOR_BINDINGS, {"record": ["ABC123"]})
        == aliased
    )


def test_fabricated_guard_keeps_a_value_carried_by_two_entity_types():
    """One value, two types: the predicate survives if EITHER is bound to the column.

    A domain whose types overlap (the same string being both an org unit and a party code)
    must not have correct predicates dropped on the strength of the other reading.
    """
    from src.retrievers.query_guards import strip_fabricated_predicates

    sql = "SELECT * FROM records WHERE d = 1 AND customer_code = 'X1'"
    assert (
        strip_fabricated_predicates(
            sql, _PARTY_BINDINGS, {"record": ["X1"], "org_unit": ["X1"]}
        )
        == sql
    )


def test_fabricated_guard_fails_open_on_an_undeclared_column():
    """No binding for the column means no claim to contradict — the predicate stays.

    The pack describes the columns somebody measured; a generated predicate on any other
    column is outside what this guard knows, and a guard that guessed there could turn a real
    finding into a missing one.
    """
    from src.retrievers.query_guards import strip_fabricated_predicates

    sql = "SELECT * FROM records WHERE d = 1 AND status = 'RSTUVW'"
    assert (
        strip_fabricated_predicates(sql, _LOCATOR_BINDINGS, {"actor": ["RSTUVW"]}) == sql
    )


def test_fabricated_guard_fails_open_on_a_literal_the_incident_never_carried():
    """A status/flag/code the generator read out of the schema is not the guard's business."""
    from src.retrievers.query_guards import strip_fabricated_predicates

    sql = "SELECT * FROM records WHERE locator.primary = 'CONFIRMED' AND d = 1"
    assert (
        strip_fabricated_predicates(sql, _LOCATOR_BINDINGS, {"actor": ["RSTUVW"]}) == sql
    )


def test_fabricated_guard_does_not_touch_the_incident_window():
    """The window is a BOUND, not a value, and is excluded from the incident's values.

    Deliberate: every source binds some column to it, so a guard treating it as a value would
    drop range predicates — including the only bound on a partition column, which is the
    slow-scan failure `enforce_partition_bounds` exists to prevent. Two guards pulling in
    opposite directions on one clause is worse than either defect.
    """
    from src.retrievers.field_mapping import incident_values
    from src.retrievers.query_guards import strip_fabricated_predicates

    query = RetrievalQuery(
        target_log_source="records",
        natural_language_query="records for the actor",
        date_from="2026-08-05",
        date_to="2026-08-05",
        entities=[
            ExtractedEntity(type="actor", value="RSTUVW", confidence=0.9),
            ExtractedEntity(type="time_window", value="2026-08-05T11:20:06", confidence=0.9),
        ],
    )
    values = incident_values(query)
    assert "time_window" not in values and values["actor"] == ["RSTUVW"]
    sql = (
        "SELECT * FROM records WHERE creation_date_time >= TIMESTAMP'2026-08-05T00:00:00' "
        "AND d = 1"
    )
    assert strip_fabricated_predicates(sql, _LOCATOR_BINDINGS, values) == sql


def test_fabricated_guard_drops_a_leading_where_predicate_and_an_esql_stage():
    """The three rewritable shapes: `... AND p`, `WHERE p AND ...`, and a whole `| WHERE p`."""
    from src.retrievers.query_guards import strip_fabricated_predicates

    leading = strip_fabricated_predicates(
        "SELECT * FROM records WHERE locator.primary = 'RSTUVW' AND d = 1",
        _LOCATOR_BINDINGS,
        {"actor": ["RSTUVW"]},
    )
    assert "RSTUVW" not in leading and "d = 1" in leading
    assert "WHERE AND" not in " ".join(leading.split())

    esql = strip_fabricated_predicates(
        'FROM idx | WHERE ts > "2026-08-01" | WHERE locator.primary == "RSTUVW" '
        "| LIMIT 10",
        _LOCATOR_BINDINGS,
        {"actor": ["RSTUVW"]},
        pipe_stages=True,
    )
    assert "RSTUVW" not in esql and "| LIMIT 10" in esql
    assert "WHERE |" not in " ".join(esql.split())


def test_fabricated_guard_leaves_an_or_group_alone_and_says_so(caplog):
    """A predicate inside an OR is not narrowing the result on its own.

    So it is left alone and LOGGED: rewriting someone's boolean tree to remove a wasted
    disjunct is the more expensive mistake, and the same reasoning the evidence strip uses.
    """
    import logging

    from src.retrievers.query_guards import strip_fabricated_predicates

    sql = (
        "SELECT * FROM records WHERE d = 1 AND (locator.primary = 'RSTUVW' "
        "OR creator.sign = 'RSTUVW')"
    )
    with caplog.at_level(logging.WARNING, logger="src.retrievers.query_guards"):
        out = strip_fabricated_predicates(
            sql, _PARTY_BINDINGS, {"actor": ["RSTUVW"]}, "records"
        )
    assert out == sql
    assert "too complex to rewrite safely" in caplog.text


def test_fabricated_guard_is_a_no_op_without_either_declaration():
    """Both inputs are required: no bindings, or no incident values, means no verdict."""
    from src.retrievers.query_guards import strip_fabricated_predicates

    sql = "SELECT * FROM records WHERE locator.primary = 'RSTUVW'"
    assert strip_fabricated_predicates(sql, {}, {"actor": ["RSTUVW"]}) == sql
    assert strip_fabricated_predicates(sql, _LOCATOR_BINDINGS, {}) == sql
    assert strip_fabricated_predicates("", _LOCATOR_BINDINGS, {"actor": ["R"]}) == ""


def test_fabricated_guard_logs_both_declarations_in_its_reason(caplog):
    """The log line must name what the column holds AND what the literal is.

    An operator reading "0 rows" needs to know which of the two declarations was contradicted
    — that is the difference between a pack binding to fix and a generator to re-prompt.
    """
    import logging

    from src.retrievers.query_guards import strip_fabricated_predicates

    with caplog.at_level(logging.WARNING, logger="src.retrievers.query_guards"):
        strip_fabricated_predicates(
            "SELECT * FROM records WHERE d = 1 AND locator.primary = 'RSTUVW'",
            _LOCATOR_BINDINGS,
            {"actor": ["RSTUVW"]},
            "records",
        )
    assert "records" in caplog.text
    assert "holds record" in caplog.text and "incident's actor" in caplog.text
    assert "0 rows" in caplog.text


def test_dsl_fabricated_filter_dropped_from_a_bool_filter():
    """The Query DSL half of the same guarantee."""
    from src.retrievers.query_guards import strip_fabricated_filters_dsl

    out = strip_fabricated_filters_dsl(
        {
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"creation_date": {"gte": "2026-08-01"}}},
                        {"term": {"locator.primary": "RSTUVW"}},
                    ]
                }
            }
        },
        _LOCATOR_BINDINGS,
        {"actor": ["RSTUVW"]},
    )
    assert out["query"]["bool"]["filter"] == [
        {"range": {"creation_date": {"gte": "2026-08-01"}}}
    ]


def test_dsl_fabricated_filter_removal_is_safe_in_every_occurrence():
    """A fabricated literal matches no document, so no position makes it load-bearing.

    Not the usual argument for a guard, and worth pinning: it contributes nothing to a
    `should`, nothing to a `must_not`, and removing it from a `must`/`filter` widens the
    result to what the query would have returned had the generator not invented it.
    """
    from src.retrievers.query_guards import strip_fabricated_filters_dsl

    for occurrence in ("must", "should", "must_not", "filter"):
        out = strip_fabricated_filters_dsl(
            {
                "query": {
                    "bool": {
                        occurrence: [{"term": {"locator.primary": "RSTUVW"}}],
                        "filter": [{"term": {"kind": "x"}}],
                    }
                }
            },
            _LOCATOR_BINDINGS,
            {"actor": ["RSTUVW"]},
        )
        bool_body = out["query"]["bool"]
        # The emptied occurrence key is DELETED, not left as `[]` (an empty `must` is not
        # the same query as no `must`), and the unrelated clause is untouched.
        if occurrence != "filter":
            assert occurrence not in bool_body
        assert bool_body["filter"] == [{"term": {"kind": "x"}}]


def test_dsl_fabricated_guard_ignores_a_range_clause():
    """A `range` is a bound, not an assertion that the column equals an incident value.

    `_dsl_clause_pairs` excludes it for the same reason the textual half never touches a time
    predicate: dropping one could remove the only bound on a partition column.
    """
    from src.retrievers.query_guards import strip_fabricated_filters_dsl

    dsl = {
        "query": {
            "bool": {
                "filter": [{"range": {"locator.primary": {"gte": "RSTUVW"}}}]
            }
        }
    }
    assert strip_fabricated_filters_dsl(
        dsl, _LOCATOR_BINDINGS, {"actor": ["RSTUVW"]}
    ) == dsl


def test_dsl_fabricated_guard_degrades_a_bare_clause_to_match_all():
    """A fabricated clause reached OUTSIDE any `bool` cannot just be deleted.

    Removing the only key under `query` would leave `{"query": {}}`, which Elasticsearch
    rejects — so it degrades to `match_all`, the same repair the evidence strip makes.
    """
    from src.retrievers.query_guards import strip_fabricated_filters_dsl

    out = strip_fabricated_filters_dsl(
        {"query": {"term": {"locator.primary": "RSTUVW"}}},
        _LOCATOR_BINDINGS,
        {"actor": ["RSTUVW"]},
    )
    assert out == {"query": {"match_all": {}}}


def test_dsl_fabricated_guard_judges_a_terms_list_per_value():
    """A `terms` mixing a real value with a fabricated one is judged on the fabricated one.

    Deliberate, and the conservative direction: the clause as a whole is a disjunction, so
    keeping it would keep a predicate the source's own binding contradicts, and the real value
    is still reachable through the clause the generator wrote for its own column.
    """
    from src.retrievers.query_guards import strip_fabricated_filters_dsl

    out = strip_fabricated_filters_dsl(
        {
            "query": {
                "bool": {
                    "filter": [
                        {"terms": {"locator.primary": ["ABC123", "RSTUVW"]}},
                        {"term": {"kind": "x"}},
                    ]
                }
            }
        },
        _LOCATOR_BINDINGS,
        {"actor": ["RSTUVW"], "record": ["ABC123"]},
    )
    assert out["query"]["bool"]["filter"] == [{"term": {"kind": "x"}}]


_BLOB_BINDINGS = {
    # A source whose identity lives in ONE text blob, bound to a single entity type. Real
    # shape: three alert sources on one live index bind a keyword notification body this way,
    # because every value is printed into it under a label.
    "actor": ["body.text"],
    "time_window": ["timestamp"],
}


def test_fabricated_guard_sees_a_value_wrapped_in_wildcards():
    """The literal-side twin of the leaf-vs-path failure, in BOTH query families.

    The test looked the literal up EXACTLY, so `*X*` / `%X%` was never one of the incident's
    values and the guard failed open on every `wildcard`, `LIKE` and `regexp` predicate — the
    only shapes a blob column can be filtered with at all. Measured 2026-08-15 on live job
    f906f127: a source binding `ir.text` to `user` was asked `wildcard ir.text *1A*` where
    `1A` was the incident's ORGANIZATION, and the clause came back untouched.
    """
    from src.retrievers.query_guards import (
        strip_fabricated_filters_dsl,
        strip_fabricated_predicates,
    )

    values = {"actor": ["AUSERNAME"], "org_unit": ["1A"]}

    out = strip_fabricated_filters_dsl(
        {
            "bool": {
                "should": [
                    {"wildcard": {"body.text": "*AUSERNAME*"}},
                    {"wildcard": {"body.text": "*1A*"}},
                ]
            }
        },
        _BLOB_BINDINGS,
        values,
        "blob",
    )
    assert out["bool"]["should"] == [{"wildcard": {"body.text": "*AUSERNAME*"}}]

    sql = (
        "SELECT * FROM t WHERE body.text LIKE '%1A%' AND body.text LIKE '%AUSERNAME%'"
    )
    kept = strip_fabricated_predicates(sql, _BLOB_BINDINGS, values, "blob")
    assert "'%1A%'" not in kept, kept
    assert "'%AUSERNAME%'" in kept, kept


def test_fabricated_guard_drops_a_substring_arm_that_is_true_of_every_row():
    """The inverse direction: a substring predicate on the wrong column may match every row.
    The guard's remit is the declaration, not syntactic vacuity: it removes a value on a
    column the source does not bind to that type, whether the clause matches nothing or
    everything. `strip_vacuous_should_clauses` tests syntactic vacuity and cannot reach this.
    """
    from src.retrievers.query_guards import (
        strip_fabricated_filters_dsl,
        strip_vacuous_should_clauses,
    )

    dsl = {
        "bool": {
            "filter": [{"term": {"type": "session_anomaly"}}],
            "should": [
                {"wildcard": {"body.text": "*AUSERNAME*"}},
                {"wildcard": {"body.text": "*1A*"}},
            ],
            "minimum_should_match": 1,
        }
    }
    # The syntactic guard is inert on it, which is why the declarative one has to fire.
    assert strip_vacuous_should_clauses(copy.deepcopy(dsl), "blob") == dsl

    out = strip_fabricated_filters_dsl(
        dsl, _BLOB_BINDINGS, {"actor": ["AUSERNAME"], "org_unit": ["1A"]}, "blob"
    )
    assert out["bool"]["should"] == [{"wildcard": {"body.text": "*AUSERNAME*"}}]
    assert out["bool"]["filter"] == [{"term": {"type": "session_anomaly"}}]


def test_fabricated_guard_keeps_a_pattern_the_value_is_not_the_whole_of():
    """The bound: only the ENDS are unwrapped, so an interior wildcard is left alone.

    `*NCE*09CO*` is not the office value with a wrapper, it is a pattern that could match
    other things, and this guard DELETES predicates — so it acts only where the value is
    unambiguously the whole of the pattern. Without this bound the unwrapping would become a
    fuzzy match on a code path whose failure mode is a silently narrower query.
    """
    from src.retrievers.query_guards import strip_fabricated_filters_dsl

    dsl = {"bool": {"should": [{"wildcard": {"body.text": "*NCE*09CO*"}}]}}
    assert (
        strip_fabricated_filters_dsl(
            dsl, _BLOB_BINDINGS, {"org_unit": ["AAA1B02CD"]}, "blob"
        )
        == dsl
    )
    # And a bare `*` stays a no-op rather than unwrapping to the empty string.
    star = {"bool": {"should": [{"wildcard": {"body.text": "*"}}]}}
    assert (
        strip_fabricated_filters_dsl(
            star, _BLOB_BINDINGS, {"org_unit": ["AAA1B02CD"]}, "blob"
        )
        == star
    )


def test_source_bindings_reads_the_declaration_unfiltered():
    """`source_bindings` must return EVERY type the source binds, not the incident's.

    The guard's question is "what does this column hold", asked of columns the incident's
    entities may have nothing to do with — the opposite of `_declared_fields`, which narrows
    to the incident and to one value form. A mock without the accessor degrades to `{}` rather
    than raising, so a retriever built without a pack still runs.
    """
    from src.retrievers.field_mapping import source_bindings

    pack = MagicMock()
    pack.source.return_value = SimpleNamespace(entity_bindings=_PARTY_BINDINGS)
    assert source_bindings(pack, "records") == _PARTY_BINDINGS
    assert source_bindings(None, "records") == {}
    assert source_bindings(pack, "") == {}
    missing = MagicMock()
    missing.source.return_value = None
    assert source_bindings(missing, "records") == {}


def test_incident_values_ignores_the_planners_guessed_scalars():
    """Read off `query.entities`, never off `scope_id`/`actor_id`.

    Those two labels are the planner's unverified guess at what a value is (the same guess
    `render_identifiers` refuses to print), and a guard fed a guessed type would drop
    predicates on the strength of it.
    """
    from src.retrievers.field_mapping import incident_values

    query = RetrievalQuery(
        target_log_source="records",
        natural_language_query="x",
        date_from="2026-08-05",
        date_to="2026-08-05",
        scope_id="BBB1C03EF",
        actor_id="RSTUVW",
        entities=[
            ExtractedEntity(type="actor", value="0202YZ", confidence=0.9),
            ExtractedEntity(type="org_unit", value="*", confidence=0.5),
        ],
    )
    assert incident_values(query) == {"actor": ["0202YZ"]}


def test_incident_values_reads_the_set_the_source_could_not_bind():
    """The guard's headline case is a type the source binds nothing for.

    `_enrich_queries` narrows `query.entities` to bindable types, deleting exactly the value
    this guard is looking for. The unfiltered set therefore rides separately on
    `_incident_entities`. The assertion covers both the mapping and the guard; each alone fails.
    """
    from src.retrievers.field_mapping import incident_values
    from src.retrievers.query_guards import strip_fabricated_predicates

    bindings = {"org_unit": ["OFFICE_NAME"], "time_window": ["LAST_UPDATE"]}
    sql = "SELECT * FROM office_profile WHERE d = 1 AND OFFICE_NAME = '8J6ONY'"

    def _query():
        return RetrievalQuery(
            target_log_source="office_profile",
            natural_language_query="the office's own profile",
            date_from="2026-08-05",
            date_to="2026-08-05",
            entities=[
                ExtractedEntity(
                    type="time_window", value="2026-08-05T11:20:06", confidence=0.9
                )
            ],
        )

    # As it shipped: the record locator is nowhere in the query, so there is nothing to
    # contradict and the predicate survives.
    blind = _query()
    assert incident_values(blind) == {}
    assert strip_fabricated_predicates(sql, bindings, incident_values(blind)) == sql

    seeing = _query()
    seeing._incident_entities = [
        ExtractedEntity(type="record", value="8J6ONY", confidence=0.9),
        ExtractedEntity(type="time_window", value="2026-08-05T11:20:06", confidence=0.9),
    ]
    values = incident_values(seeing)
    assert values == {"record": ["8J6ONY"]}
    out = strip_fabricated_predicates(sql, bindings, values, "office_profile")
    assert "8J6ONY" not in out and "d = 1" in out


def test_incident_values_unions_both_lists_so_a_harvested_literal_stays_visible():
    """Unioned, not preferred — a follow-up pass's values are on `entities` ALONE.

    Harvested values ride on `RetrievalQuery.entities` and are deliberately never added to the
    incident's `extracted_entities` (`src/follow_up.py`), so reading the incident set *instead*
    would make the guard blind on pass 2 to a literal it could see on pass 1. Both directions
    are asserted here because the union is what makes the change monotone: strictly more
    visible, never differently typed.
    """
    from src.retrievers.field_mapping import incident_values

    query = RetrievalQuery(
        target_log_source="office_profile",
        natural_language_query="the offices carried forward",
        date_from="2026-08-05",
        date_to="2026-08-05",
        entities=[
            ExtractedEntity(type="org_unit", value="JJJ1K10UV", confidence=0.9),
            ExtractedEntity(type="org_unit", value="PPP1Q16EF", confidence=0.9),
        ],
    )
    query._incident_entities = [
        ExtractedEntity(type="record", value="8J6ONY", confidence=0.9),
        ExtractedEntity(type="actor", value="6008HH", confidence=0.9),
    ]
    assert incident_values(query) == {
        "record": ["8J6ONY"],
        "actor": ["6008HH"],
        "org_unit": ["JJJ1K10UV", "PPP1Q16EF"],
    }
    # And a value carried on both lists is not duplicated — `_types_by_value` folds by value,
    # but the log line prints these values back and a repeated one reads as two facts.
    query._incident_entities = list(query._incident_entities) + [
        ExtractedEntity(type="org_unit", value="JJJ1K10UV", confidence=0.9)
    ]
    assert incident_values(query)["org_unit"] == ["JJJ1K10UV", "PPP1Q16EF"]


@pytest.mark.asyncio
async def test_databricks_strips_a_fabricated_predicate_end_to_end():
    """The guard rides the retriever, before the partition bounds.

    Order matters: a fabricated predicate may be the only thing constraining a prune column,
    so dropping it AFTER the bounds would leave the query unbounded with nothing left to
    inject a bound onto — and an unbounded scan is the slow-query failure that reads as an
    empty source one stage later.
    """
    from src.models.pydantic_models import FieldMapping

    pack = MagicMock()
    pack.source.return_value = SimpleNamespace(entity_bindings=_LOCATOR_BINDINGS)
    pack.field_priors_for.return_value = {}
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        side_effect=[
            FieldMapping(mappings=[]),
            SqlQuery(
                query="SELECT * FROM records WHERE creation_date >= DATE'2026-08-05' "
                "AND locator.primary = 'RSTUVW'"
            ),
        ]
    )
    retriever = DatabricksRetriever(
        {
            "name": "records",
            "workspace_url": "https://workspace",
            "warehouse_id": "wh1",
            "api_key_env": "DATABRICKS_TOKEN",
            "field_schema": "records(locator struct<primary:string>, creation_date date)",
        },
        llm,
        knowledge_pack=pack,
    )
    retriever._execute_sql = AsyncMock(return_value=[])
    await retriever.retrieve(
        RetrievalQuery(
            target_log_source="records",
            natural_language_query="records for the actor",
            date_from="2026-08-05",
            date_to="2026-08-05",
            entities=[ExtractedEntity(type="actor", value="RSTUVW", confidence=0.9)],
        )
    )
    executed = retriever._execute_sql.await_args[0][0]
    assert "RSTUVW" not in executed
    assert "creation_date >= DATE'2026-08-05'" in executed
    # Published AFTER the guard: what the operator is shown is what ran.
    assert "RSTUVW" not in (retriever.last_generated_query or "")


def test_fabricated_guard_rides_every_llm_generating_retriever():
    """One guarantee, four backends — the failure mode this whole section exists to prevent.

    `never_filter` was once enforced on the Databricks path only, which made declaring it on
    an Elasticsearch source a silent no-op. A guard wired into three of four retrievers is the
    same defect, so this asserts on the call sites rather than on behaviour four times over.
    The REST retriever is deliberately absent: it builds its clauses in code from `field_map`,
    so there is no generated query to repair.
    """
    import inspect

    from src.retrievers import (databricks_retriever, elasticsearch_retriever,
                                kibana_retriever, snowflake_retriever)
    from src.retrievers.snowflake_retriever import SnowflakeRetriever

    for module, cls, guard in (
        (databricks_retriever, DatabricksRetriever, "fabricated"),
        (elasticsearch_retriever, ElasticsearchRetriever, "fabricated"),
        (snowflake_retriever, SnowflakeRetriever, "fabricated"),
        (kibana_retriever, KibanaRetriever, "fabricated"),
    ):
        assert "fabricated" in inspect.getsource(module), (
            f"{module.__name__} never strips a fabricated predicate"
        )
        # Read off the FUNNEL, not the module: the guard may be called through a wrapper
        # method whose own definition sits after the funnel, and a whole-file index would
        # then compare the definition rather than the call.
        funnel = inspect.getsource(cls.retrieve)
        assert guard in funnel, f"{cls.__name__}.retrieve never strips a fabricated filter"
        # ...and it must run BEFORE the bounds are injected, for the reason above.
        assert funnel.index(guard) < funnel.index("enforce_partition_bounds")


# --- the fourth shape: a query that constrains the subject nowhere -----------------------
#
# `enforce_identity_scope` only restructures and never adds; its contract is "a scope the
# query never constrained is not invented here". `enforce_subject_anchor` covers the shape
# where the subject is not constrained conjunctively at all.


def _sql_predicate_matches(flat_sql: str, row: dict, opaque_true: bool = True) -> bool:
    """Evaluate a whole post-guard WHERE against one row, as Python booleans.

    Wider than `_sql_group_matches` (which slices out the identity group) because the property
    under test here is about the WHOLE predicate: an anchor is only worth adding if a stranger's
    row fails the query that ran. Anything this cannot evaluate — a membership subquery, a range
    on a column the row does not carry — is read as ``opaque_true``, i.e. the widest possible
    reading. That direction is the honest one: it grants the un-anchored parts of the query
    everything they could match, so a `False` here can only come from the anchor itself.
    """
    import re as _re

    from src.retrievers.query_guards import _tail_index

    body = flat_sql[_re.search(r"\bWHERE\b", flat_sql, _re.I).end() :]
    body = body[: _tail_index(body)]

    # A membership test is consumed WHOLE — column, keyword and balanced parenthesis group —
    # before anything else looks at the text. Rewriting its pieces one regex at a time leaves
    # the subquery's own predicates behind as bare identifiers, and the evaluator then fails
    # for a reason that has nothing to do with the guard.
    def _swallow_in(text: str) -> str:
        while True:
            m = _re.search(r"[\w.`]+\s+IN\s*\(", text, _re.I)
            if not m:
                return text
            depth, i = 0, m.end() - 1
            while i < len(text):
                depth += (text[i] == "(") - (text[i] == ")")
                if depth == 0:
                    break
                i += 1
            text = text[: m.start()] + repr(opaque_true) + text[i + 1 :]

    py = _swallow_in(body)
    py = _re.sub(
        r"[\w.`]*?(\w+)\s*=\s*'([^']*)'",
        lambda m: repr(row.get(m.group(1)) == m.group(2)),
        py,
    )
    # Whatever is left that still looks like a comparison — a window bound, a range on a
    # column the row does not carry — is opaque, i.e. granted everything it could match.
    py = _re.sub(
        r"[\w.`]+\s*(?:>=|<=|!=|<>|=|<|>)\s*(?:DATE|TIMESTAMP)?'[^']*'",
        repr(opaque_true),
        py,
    )
    py = _re.sub(r"\bAND\b", " and ", py)
    py = _re.sub(r"\bOR\b", " or ", py)
    if _re.search(r"[A-Za-z_`][\w.`]*\s*(?:=|<|>)|'", py):
        raise AssertionError(f"could not evaluate: {py}")
    return bool(eval(py))  # noqa: S307 — a test-local boolean expression, no input from data


def test_subject_anchor_binds_the_subject_in_all_three_generated_shapes():
    """One incident, three generated shapes; after the guard all three ask about the subject.

    The assertion that matters is not textual. For each shape the post-guard predicate is
    evaluated against the SUBJECT's row and against a STRANGER's row — a different login and
    sign in a different office, which is what the live run returned 500 of — and the stranger
    must fail. The un-anchored parts of the query are granted the widest possible reading
    (`opaque_true`), so a stranger failing can only be the anchor's doing.
    """
    from src.retrievers.query_guards import (enforce_identity_scope,
                                             enforce_subject_anchor)

    identity = {"a.login": ["AGENT"], "a.userId": ["AGENT"]}
    scopes = {"a.office": ["OFFICE1"]}
    subject = {"login": "AGENT", "userId": "AGENT", "sign": "SIGN1", "office": "OFFICE1"}
    stranger = {"login": "OTHER", "userId": "OTHER", "sign": "SIGN9", "office": "OFFICE9"}
    window = "date >= DATE'2025-04-13' AND date <= DATE'2025-04-18'"

    # Shape 1 — the generator got it right. Left EXACTLY as written, and it already excludes
    # the stranger, which is why leaving it alone is correct rather than merely conservative.
    right = (
        f"SELECT * FROM t WHERE {window} AND ((a.login = 'AGENT' OR a.userId = 'AGENT') "
        "AND a.sign = 'SIGN1' AND a.office = 'OFFICE1') LIMIT 500"
    )
    assert enforce_subject_anchor(right, identity, scopes, "auth_events") == right
    assert _sql_predicate_matches(right, subject)
    assert not _sql_predicate_matches(right, stranger)

    # Shape 2 — the bare arms OR-ed back beside the conjunction. The identity REWRITE owns this
    # one (it restructures a group the generator did write), and the anchor must then find it
    # already constrained and add nothing on top: two guards both narrowing the same fact would
    # publish the predicate twice, and an operator reads a duplicated conjunct as a second fact.
    ored = (
        f"SELECT * FROM t WHERE {window} AND ("
        "((a.login = 'AGENT' OR a.userId = 'AGENT') AND a.sign = 'SIGN1' AND a.office = 'OFFICE1')"
        " OR a.login = 'AGENT' OR a.userId = 'AGENT' OR a.sign = 'SIGN1') LIMIT 500"
    )
    rewritten = enforce_identity_scope(
        ored, ["a.office", "a.sign"], ["a.login", "a.userId"], "auth_events"
    )
    anchored = enforce_subject_anchor(rewritten, identity, scopes, "auth_events")
    assert anchored == rewritten, anchored
    flat = " ".join(anchored.split())
    assert flat.count("a.office = 'OFFICE1'") == 1, flat
    assert _sql_predicate_matches(flat, subject), flat
    assert not _sql_predicate_matches(flat, stranger), flat

    # Shape 3 — pass 2's question asked in pass 1: a membership subquery over the client
    # address, with no identity group for any rewrite to restructure. This is the shape that
    # returned the cap over 298 signs, and the one the anchor exists for. PRE-FIX this query is
    # published untouched and the stranger satisfies it.
    pivot = (
        f"SELECT * FROM t WHERE {window} AND (a.login = 'AGENT' OR a.userId = 'AGENT' "
        "OR a.ipPort IN (SELECT ipPort FROM t WHERE a.login = 'AGENT')) LIMIT 500"
    )
    assert _sql_predicate_matches(pivot, stranger), "premise: the stranger matches pre-guard"
    out = enforce_subject_anchor(pivot, identity, scopes, "auth_events")
    assert out != pivot
    flat = " ".join(out.split())
    assert _sql_predicate_matches(flat, subject), flat
    assert not _sql_predicate_matches(flat, stranger), flat
    # The window survives, and the row cap is untouched — the anchor narrows rows, nothing else.
    assert "date >= DATE'2025-04-13'" in flat and flat.endswith("LIMIT 500")
    # ...and the guard's own reason for firing on this shape, asserted separately because it is
    # the one decision that cannot be read off the output. The LOOSE presence test every other
    # guard here uses says the query already constrains the login — it does compare it — so
    # reusing that test would decline on the only shape this guard exists for, and the decline
    # is indistinguishable from a guard with nothing to do.
    from src.retrievers.query_guards import (_is_constrained,
                                             _is_constrained_conjunctively)

    assert _is_constrained(pivot, "a.login", "sql"), "premise: the loose test says anchored"
    assert not _is_constrained_conjunctively(pivot, identity, "sql")
    # And on shape 1, where the identity really does bind every returned row, both agree.
    assert _is_constrained_conjunctively(right, identity, "sql")


def test_subject_anchor_never_invents_an_identity_it_was_not_given():
    """Three ways there is nothing to anchor, and all three publish the query as generated."""
    from src.retrievers.query_guards import (enforce_subject_anchor,
                                             subject_anchor_values)

    sql = "SELECT * FROM t WHERE date >= DATE'2025-04-13' LIMIT 500"
    # 1. No identity column resolved — a window-scoped source by design.
    assert enforce_subject_anchor(sql, {}, {"a.office": ["O"]}, "s") == sql
    # 2. The incident carries a SCOPE value but no identity value. Anchoring the scope alone
    #    would ask about everybody sharing it — the whole-population query the identity guard
    #    refuses to write — so the resolver returns nothing at all.
    assert subject_anchor_values(
        {"a.office": ["O"]}, ["a.office"], ["a.login", "a.userId"], "s"
    ) == ({}, {})
    # 3. A source declaring neither list: two empty maps, so the guard cannot fire.
    assert subject_anchor_values({"a.login": ["X"]}, [], [], "s") == ({}, {})


def test_subject_anchor_places_a_value_on_EVERY_column_of_its_family():
    """The mapper returns one column per entity TYPE; a synonym family has several.

    MEASURED null rates on the live source: the first column of the family is null on 32.0% of
    rows and the second on 0.3%, because the producer populates whichever it has. Anchoring on
    the mapper's single choice would therefore delete about a third of the subject's own rows —
    the exact failure the OR exists to prevent, arriving as a smaller row count rather than as
    an error.
    """
    from src.retrievers.query_guards import subject_anchor_values

    identity, scopes = subject_anchor_values(
        {"a.login": ["AGENT"], "a.office": ["OFFICE1"], "a.org": ["ORG"]},
        ["a.office"],
        [["a.login", "a.userId"]],
        "s",
    )
    # The value found on one column of the family is placed on BOTH...
    assert identity == {"a.login": ["AGENT"], "a.userId": ["AGENT"]}
    # ...the declared scope is a separate fact, AND-ed...
    assert scopes == {"a.office": ["OFFICE1"]}
    # ...and the organisation column, which the pack declares NEITHER, is ignored: it is
    # strictly broader than the office and can only delete a row whose label differs from the
    # alert's while narrowing nothing the office does not already narrow.
    assert "a.org" not in identity and "a.org" not in scopes


def test_subject_anchor_does_not_narrow_a_follow_up_pass_query():
    """A pass-2 blast-radius query carries no subject identity, so there is nothing to add.

    `generate_follow_up` builds `entities` from the harvested values alone and never calls
    `_enrich_queries` — measured: the live pass-2 query carried only `ip_address`. So the
    resolver finds no identity value and the guard cannot re-narrow a question whose whole
    purpose is to widen past the subject. Asserted so it cannot silently regress.
    """
    from src.models.pydantic_models import ExtractedEntity, RetrievalQuery
    from src.retrievers.field_mapping import subject_anchor
    from src.retrievers.query_guards import enforce_subject_anchor

    query = RetrievalQuery(
        target_log_source="access_pivot",
        natural_language_query="which records were read from these addresses",
        date_from="2025-04-14",
        date_to="2025-04-17",
        entities=[ExtractedEntity(type="ip_address", value="10.0.0.1", confidence=1.0)],
    )
    identity, scopes = subject_anchor(
        {"user": "a.login", "ip_address": "a.ipPort", "office": "a.office"},
        query,
        ["a.office"],
        ["a.login", "a.userId"],
        None,
        "access_pivot",
    )
    assert (identity, scopes) == ({}, {})
    sql = "SELECT * FROM t WHERE a.ipPort = '10.0.0.1' LIMIT 500"
    assert enforce_subject_anchor(sql, identity, scopes, "access_pivot") == sql


def test_subject_anchor_runs_on_every_backend_after_the_identity_rewrite():
    """Wired into all four, in the one position that is correct — or it is not a guarantee.

    AFTER the rewrite, because that rewrite may lift a scope out of an OR-group and this must
    see the result before deciding the subject is unconstrained. BEFORE the bounds, for the
    reason the bounds themselves give. And on BOTH query families: `enforce_identity_scope` was
    textual-SQL-only for as long as it existed, so on a Query DSL source the declaration read
    as honoured and did nothing.
    """
    import inspect

    from src.retrievers import (databricks_retriever, elasticsearch_retriever,
                                kibana_retriever, snowflake_retriever)
    from src.retrievers.elasticsearch_retriever import ElasticsearchRetriever
    from src.retrievers.kibana_retriever import KibanaRetriever
    from src.retrievers.snowflake_retriever import SnowflakeRetriever

    for module, cls, guard in (
        (databricks_retriever, DatabricksRetriever, "enforce_subject_anchor"),
        (elasticsearch_retriever, ElasticsearchRetriever, "enforce_subject_anchor"),
        (snowflake_retriever, SnowflakeRetriever, "enforce_subject_anchor"),
        (kibana_retriever, KibanaRetriever, "enforce_subject_anchor_dsl"),
    ):
        assert guard in inspect.getsource(module), f"{module.__name__} never anchors"
        funnel = inspect.getsource(cls.retrieve)
        assert "_enforce_subject_anchor" in funnel, f"{cls.__name__}.retrieve never anchors"
        assert funnel.index("_enforce_identity_scope") < funnel.index(
            "_enforce_subject_anchor"
        ), f"{cls.__name__} anchors BEFORE the identity rewrite"
        bounds = "enforce_partition_bounds"
        assert funnel.index("_enforce_subject_anchor") < funnel.index(bounds), (
            f"{cls.__name__} anchors AFTER the bounds"
        )


def test_subject_anchor_dsl_nests_the_generated_query_beside_the_anchor():
    """The DSL twin: the original query keeps its own boolean structure exactly."""
    from src.retrievers.query_guards import enforce_subject_anchor_dsl

    dsl = {
        "query": {
            "bool": {
                "should": [{"term": {"a.ipPort": "10.0.0.1"}}],
                "minimum_should_match": 1,
            }
        }
    }
    out = enforce_subject_anchor_dsl(
        dsl, {"a.login": ["AGENT"], "a.userId": ["AGENT"]}, {"a.office": ["OFFICE1"]}, "s"
    )
    clauses = out["bool"]["filter"]
    # The two identity columns are ONE nested should (they may hold one value), the scope is a
    # sibling (a separate fact), and the generated query is the last conjunct, intact.
    assert clauses[0]["bool"]["minimum_should_match"] == 1
    assert sorted(next(iter(c["terms"])) for c in clauses[0]["bool"]["should"]) == [
        "a.login",
        "a.userId",
    ]
    assert clauses[1] == {"terms": {"a.office": ["OFFICE1"]}}
    assert clauses[2] == dsl
    # Not mutated, and an already-named field is left entirely alone.
    assert "a.login" not in json.dumps(dsl)
    named = {"bool": {"filter": [{"terms": {"a.login": ["AGENT"]}}]}}
    assert enforce_subject_anchor_dsl(named, {"a.login": ["AGENT"]}, {}, "s") == named


def test_the_event_time_bound_is_added_when_the_generated_query_omits_it():
    """The pad becomes the event window otherwise, and a row cap turns that into a verdict.

    Measured on one incident across three runs against a source binding a coarse `date`
    partition (pad 1) and the finer `value.timestamp` the event carries. The run that bounded
    the instant returned 150 rows and NONE on the padded day; the two that did not filled the
    500-row cap with 177 and then 430 pad-day rows, displacing the subject's own events —
    which reads downstream as INSUFFICIENT DATA and points nowhere near here.
    """
    from src.retrievers.field_mapping import event_time_column
    from src.retrievers.query_guards import enforce_event_time_window

    schema = "events(date date, value.timestamp timestamp, value.login string)"
    resolved = event_time_column(
        schema,
        {"time_window": ["date", "value.timestamp"], "user": ["value.login"]},
        [{"name": "date", "role": "date", "type": "DATE", "pad_days": 1}],
        [],
        "s",
    )
    # The partition column is skipped — bounding it again only duplicates the partition guard.
    assert resolved == ("value.timestamp", "timestamp")

    sql = (
        "SELECT value.login, value.timestamp FROM events "
        "WHERE date >= DATE'2025-04-13' AND date <= DATE'2025-04-18' "
        "AND value.login = 'AGENT' ORDER BY value.timestamp"
    )
    out = enforce_event_time_window(
        sql, resolved[0], resolved[1], "2025-04-14", "2025-04-17", "s", dialect="sql"
    )
    assert "value.timestamp >= TIMESTAMP'2025-04-14'" in out
    # Upper-EXCLUSIVE against the day after: `<= date_to` on an instant column cuts at
    # midnight and silently loses the last day's events.
    assert "value.timestamp < TIMESTAMP'2025-04-18'" in out
    # The trailing clause is preserved and the original body parenthesised, so a top-level OR
    # inside it cannot be re-associated by the AND.
    assert out.rstrip().endswith("ORDER BY value.timestamp")
    assert "AND (date >= DATE'2025-04-13'" in out
    # A pad-day row no longer satisfies the query; an in-window one still does.
    assert "2025-04-13" not in out.split("value.timestamp >=")[1].split("AND (")[0]

    # Already bounded (the run that worked) → returned byte-identical, tighter bound and all.
    tight = (
        "SELECT * FROM events WHERE date >= DATE'2025-04-13' "
        "AND value.timestamp >= TIMESTAMP'2025-04-15 08:00:00'"
    )
    assert (
        enforce_event_time_window(
            tight, resolved[0], resolved[1], "2025-04-14", "2025-04-17", "s"
        )
        == tight
    )


def test_the_event_time_guard_never_rebounds_the_partition_column():
    """Three declines, each of which would otherwise DELETE rows rather than add a bound."""
    from src.retrievers.field_mapping import event_time_column

    schema = "t(access_date date, access_timestamp timestamp, year string)"
    # 1. The only bound field IS the partition column → nothing finer to bound.
    assert (
        event_time_column(
            schema,
            {"time_window": ["access_date"]},
            [{"name": "access_date", "role": "date", "type": "DATE", "pad_days": 1}],
            [],
            "s",
        )
        is None
    )
    # 2. An ASYMMETRIC pad states the partition column runs on a DIFFERENT CLOCK from the
    #    event (it records when the row was last WRITTEN), so the finer column is not the same
    #    window narrowed — bounding it to the event window deletes the later versions that pad
    #    exists to keep.
    assert (
        event_time_column(
            schema,
            {"time_window": ["access_date", "access_timestamp"]},
            [
                {
                    "name": "access_date",
                    "role": "date",
                    "type": "DATE",
                    "pad_days": 3,
                    "pad_days_after": 21,
                }
            ],
            [],
            "s",
        )
        is None
    )
    # 3. A calendar-part layout is a different mapping from the window entirely.
    assert (
        event_time_column(
            schema,
            {"time_window": ["access_timestamp"]},
            [{"name": "year", "role": "year", "type": "STRING"}],
            [],
            "s",
        )
        is None
    )
    # 4. An epoch-integer column is `enforce_epoch_window`'s territory, not this guard's.
    assert (
        event_time_column(
            "t(d date, epoch bigint)",
            {"time_window": ["d", "epoch"]},
            [{"name": "d", "role": "date", "type": "DATE", "pad_days": 1}],
            [{"name": "epoch", "unit": "milliseconds"}],
            "s",
        )
        is None
    )


def test_an_undecidable_time_column_type_leaves_the_query_unchanged():
    """The decline is the point, and on one backend it is the NORMAL case.

    A rendered schema that carries only field NAMES — which is what the Kibana route's
    document sampling produces — cannot say whether a literal must be `TIMESTAMP'...'`,
    `DATE'...'` or a quoted string. An unbounded query returns background rows; a bound in the
    wrong form returns ZERO, which reads as "the source had nothing". So absence of a type
    declines, and so does a type no literal form here can be rendered for safely.
    """
    from src.retrievers.field_mapping import event_time_column
    from src.retrievers.query_guards import enforce_event_time_window

    parts = [{"name": "date", "role": "date", "type": "DATE", "pad_days": 1}]
    bindings = {"time_window": ["date", "value.timestamp"]}
    # Names only, no types: the whole Kibana route.
    assert event_time_column("date, value.timestamp, value.login", bindings, parts, [], "s") is None
    # Present in the schema but not temporal — a STRING column holding ISO text compares as a
    # prefix and a STRING column holding anything else returns zero rows; nothing here can
    # tell those apart, so it declines rather than defaulting.
    assert (
        event_time_column(
            "t(date date, value.timestamp string)", bindings, parts, [], "s"
        )
        is None
    )
    # Absent from the schema altogether (a stale binding).
    assert event_time_column("t(date date, other int)", bindings, parts, [], "s") is None
    # And a None column is a no-op at the guard, so the decline cannot become an exception.
    sql = "SELECT * FROM t WHERE date >= DATE'2025-04-13'"
    assert enforce_event_time_window(sql, None, "", "2025-04-14", "2025-04-17", "s") == sql


def test_event_time_guard_runs_on_every_backend():
    """Wired into all four, after the partition bound — or it is not a guarantee.

    AFTER, because this narrows the window that bound's pad widened and must see the bound it
    is narrowing. And on BOTH query families: a guard written for textual SQL alone is how a
    pack declaration comes to be honoured on one backend and silently ignored on the other.
    """
    import inspect

    from src.retrievers import (databricks_retriever, elasticsearch_retriever,
                                kibana_retriever, snowflake_retriever)
    from src.retrievers.elasticsearch_retriever import ElasticsearchRetriever
    from src.retrievers.kibana_retriever import KibanaRetriever
    from src.retrievers.snowflake_retriever import SnowflakeRetriever

    for module, cls, guard in (
        (databricks_retriever, DatabricksRetriever, "enforce_event_time_window"),
        (elasticsearch_retriever, ElasticsearchRetriever, "enforce_event_time_window"),
        (snowflake_retriever, SnowflakeRetriever, "enforce_event_time_window"),
        (kibana_retriever, KibanaRetriever, "enforce_event_time_window_dsl"),
    ):
        assert guard in inspect.getsource(module), f"{module.__name__} never bounds it"
        funnel = inspect.getsource(cls.retrieve)
        assert "event_time" in funnel, f"{cls.__name__}.retrieve never bounds it"
        bounds = "enforce_partition_bounds"
        assert funnel.index(bounds) < funnel.index("event_time"), (
            f"{cls.__name__} bounds the event time BEFORE the partition bound"
        )
        # Must be resolved from the pack's `source_bindings`, not the LLM-derived field_map.
        # Asserted by adjacency to `event_time_column(` in the class body.
        whole = inspect.getsource(cls)
        call = whole.index("event_time_column(")
        assert "source_bindings" in whole[call : call + 400], (
            f"{cls.__name__} resolves the event-time column off the field_map"
        )


def test_event_time_window_dsl_nests_the_generated_query_beside_the_bound():
    """The DSL twin: the original query keeps its own boolean structure exactly."""
    from src.retrievers.query_guards import enforce_event_time_window_dsl

    dsl = {"bool": {"should": [{"term": {"login": "AGENT"}}], "minimum_should_match": 1}}
    out = enforce_event_time_window_dsl(
        dsl, "value.timestamp", "timestamp", "2025-04-14", "2025-04-17", "s"
    )
    clauses = out["bool"]["filter"]
    assert clauses[0] == {
        "range": {"value.timestamp": {"gte": "2025-04-14", "lt": "2025-04-18"}}
    }
    assert clauses[1] == dsl
    assert "value.timestamp" not in json.dumps(dsl)  # not mutated
    # Already named anywhere in the query → left entirely alone.
    named = {"range": {"value.timestamp": {"gte": "2025-04-15"}}}
    assert (
        enforce_event_time_window_dsl(
            named, "value.timestamp", "timestamp", "2025-04-14", "2025-04-17", "s"
        )
        == named
    )


def test_a_lifted_scope_is_not_emitted_twice_when_the_group_also_names_it():
    """A scope lifted from a dropped arm, already present as a bare disjunct, was AND-ed twice.

    A scope predicate can enter ``scopes`` from two routes: the group dedup (over ``pieces``,
    one predicate per ``(column, literal)``) and the lift path (concatenated onto ``scopes``
    after the dedup has already run). The two routes cannot see each other, so a scope the
    generator wrote both inside a dropped arm and again as a bare disjunct is emitted once per
    route. The fix deduplicates ``scopes`` on ``(leaf, normalised predicate)`` rather than
    filtering the lifted list against the bare one.
    """
    from src.retrievers.query_guards import enforce_identity_scope

    # The live shape: a correct conjunction, the bare synonym arms OR-ed back beside it, and a
    # third arm naming the org_unit and NO synonym column — so that arm is dropped and its org_unit
    # predicate is lifted onto a group that already carries the very same predicate.
    sql = (
        "SELECT a FROM t\nWHERE date >= DATE'2025-04-13'\n  AND (\n"
        "    (\n"
        "      (a.login = 'X' OR a.userId = 'X')\n"
        "      AND a.sign = 'S'\n"
        "      AND a.org_unit = 'U'\n"
        "    )\n"
        "    OR (a.org_unit = 'U' AND a.robot = false)\n  )"
    )
    out = enforce_identity_scope(
        sql, ["a.org_unit", "a.sign"], ["a.login", "a.userId"], "auth_events"
    )
    flat = " ".join(out.split())
    # The scope is still enforced — the fix must not drop one, only stop repeating one.
    assert "a.org_unit = 'U'" in flat, flat
    assert "a.sign = 'S'" in flat, flat
    # ...exactly ONCE. This is the assertion that fails pre-fix.
    assert flat.count("a.org_unit = 'U'") == 1, flat
    assert flat.count("a.sign = 'S'") == 1, flat
    # And the query still means what it meant: the subject matches, a stranger sharing only the
    # scope does not. The literals must be the SQL's own, or a row that matches nothing passes
    # the negative assertion for the wrong reason and the positive one fails silently.
    subject = {"login": "X", "userId": "X", "sign": "S", "org_unit": "U"}
    other = {
        "login": "someone-else",
        "userId": "someone-else",
        "sign": "OTHER",
        "org_unit": "U",
    }
    assert _sql_group_matches(flat, subject), flat
    assert not _sql_group_matches(flat, other), flat
    # The mandatory partition bound is untouched, as always.
    assert "date >= DATE'2025-04-13'" in out


# --- a literal in the wrong precision: the only widening guard in the module --------------
#
# The long form of an identifier is a valid predicate matching nothing on a column storing
# its stem. Measured live: a reference table's key column holds the stem on 946,805 of 946,805
# non-empty rows and the long form on zero, so the lookup returned 0 rows with every stage
# reporting success — and because that source declares `zero_rows.health_weight: 0.0` (empty
# MEANS absent from the register), the ruleset's only decisive check did not degrade, it
# INVERTED.
#
# The column and literal both come from `field_mapping.stem_literals`; negated shapes are
# declined since a widened NOT-IN deletes the subject's rows and reads as a clean.


def test_stem_widening_offers_the_stem_inside_an_EXISTING_list():
    """The common shape, because `render_filters` emits `IN` for every multi-value column."""
    from src.retrievers.query_guards import widen_stem_literals

    stems = {"sign": {"7777ABCD": "7777AB"}}
    sql = (
        "SELECT profile FROM ref.robot_register\n"
        "WHERE unitId IN ('OFF1') AND sign IN ('7777ABCD', '6666WXYZ')"
    )
    out = widen_stem_literals(sql, stems, "robot_register")
    assert "sign IN ('7777ABCD', '6666WXYZ', '7777AB')" in out
    # Appending inside the parentheses cannot change the shape of anything else: the other
    # column, the projection and the AND structure are byte-identical.
    assert "unitId IN ('OFF1')" in out
    assert out.startswith("SELECT profile FROM ref.robot_register")
    # Already widened -> unchanged, so a re-run (or a generator that followed the hint) does
    # not publish the same fact twice.
    assert widen_stem_literals(out, stems, "r") == out
    # A table alias is the same column.
    aliased = "WHERE r.sign IN ('7777ABCD')"
    assert "r.sign IN ('7777ABCD', '7777AB')" in widen_stem_literals(aliased, stems, "r")
    # A column whose name merely ENDS with the declared one is a different column. Without
    # `(?<!\w)` the widening lands there instead.
    other = "WHERE design IN ('7777ABCD')"
    assert widen_stem_literals(other, stems, "r") == other


def test_stem_widening_promotes_an_equality_to_a_list():
    """`col = 'v'` has to become `IN`, and only for a literal the caller resolved."""
    from src.retrievers.query_guards import widen_stem_literals

    stems = {"sign": {"7777ABCD": "7777AB"}}
    out = widen_stem_literals("WHERE sign = '7777ABCD' AND d >= DATE'2026-08-01'", stems, "r")
    assert "sign IN ('7777ABCD', '7777AB')" in out
    # The window — every bound that is not the widened comparison — is untouched.
    assert "d >= DATE'2026-08-01'" in out
    # `==`, which one dialect here spells that way.
    assert "sign IN ('7777ABCD', '7777AB')" in widen_stem_literals(
        "WHERE sign == '7777ABCD'", stems, "r"
    )
    # A literal the caller did not resolve is not touched, even on the right column: the guard
    # cannot invent a stem, only publish the one the pack declared for a value the hint carried.
    unknown = "WHERE sign = 'SOMETHINGELSE'"
    assert widen_stem_literals(unknown, stems, "r") == unknown
    # An empty stem map is the state of every pack that has not opted in: byte-identical.
    assert widen_stem_literals(unknown, {}, "r") == unknown
    assert widen_stem_literals("", stems, "r") == ""
    # A pair whose stem equals its value contributes nothing.
    assert widen_stem_literals(unknown, {"sign": {"X": "X"}}, "r") == unknown


def test_stem_widening_REFUSES_every_negation_and_every_unparsed_shape():
    """Getting this backwards is worse than the defect it fixes.

    A widened `NOT IN` / `<>` NARROWS the result — on an exclusion check it would delete the
    subject's own rows and report the emptiness as a clean. `sign NOT IN (...)` also happens
    not to match the naive pattern, but an accident is not a guarantee and is not visible in
    the log, so the negation is matched on both sides of the column and declined explicitly.
    """
    from src.retrievers.query_guards import widen_stem_literals

    stems = {"sign": {"7777ABCD": "7777AB"}}
    for sql in (
        "WHERE sign NOT IN ('7777ABCD')",
        "WHERE NOT sign IN ('7777ABCD')",
        "WHERE NOT sign = '7777ABCD'",
        "WHERE sign <> '7777ABCD'",
        "WHERE sign != '7777ABCD'",
        # Not an equality against a literal: a pattern already has its own precision semantics,
        # and widening one would be a second guess about what the generator meant.
        "WHERE sign LIKE '7777ABCD%'",
        # Compared against an expression rather than a literal.
        "WHERE sign = other.sign",
        # A nested list this guard does not parse — left exactly as generated.
        "WHERE sign IN (SELECT s FROM t WHERE x IN ('7777ABCD'))",
    ):
        assert widen_stem_literals(sql, stems, "r") == sql, sql


def test_stem_widening_runs_on_every_backend_AFTER_the_anchor_and_both_strips():
    """A guard wired into one retriever is a pack declaration that silently does nothing.

    Order is load-bearing in both directions: the subject anchor may be the very predicate
    that needs widening (so this runs after it), and a fabricated predicate is DROPPED rather
    than repaired (so it must not be widened first).
    """
    import inspect

    from src.retrievers import (databricks_retriever, elasticsearch_retriever,
                                kibana_retriever, snowflake_retriever)

    wiring = [
        (databricks_retriever, "_widen_stem_literals", "_strip_fabricated_filters"),
        (elasticsearch_retriever, "widen_stem_literals", "strip_fabricated_predicates"),
        (snowflake_retriever, "widen_stem_literals", "strip_fabricated_predicates"),
        (kibana_retriever, "widen_stem_literals_dsl", "strip_fabricated_filters_dsl"),
    ]
    for module, guard, strip in wiring:
        body = inspect.getsource(module)
        retrieve = body[body.index("async def retrieve") :]
        assert guard in retrieve, f"{module.__name__} does not run {guard}"
        for earlier in (strip, "_enforce_subject_anchor"):
            assert retrieve.index(earlier) < retrieve.index(
                guard
            ), f"{module.__name__} runs {guard} BEFORE {earlier}"


def test_stem_widening_on_the_dsl_route_promotes_a_term_and_spares_must_not():
    """Both query families or it is not a guarantee (the mistake every twin here fixes)."""
    from src.retrievers.query_guards import widen_stem_literals_dsl

    stems = {"loginArea.sign": {"7777ABCD": "7777AB"}}
    dsl = {
        "query": {
            "bool": {
                "filter": [
                    {"range": {"date": {"gte": "2026-08-01"}}},
                    {"term": {"loginArea.sign": "7777ABCD"}},
                    {"terms": {"unitId": ["OFF1"]}},
                ],
                "must_not": [{"term": {"loginArea.sign": "7777ABCD"}}],
            }
        }
    }
    out = widen_stem_literals_dsl(dsl, stems, "r")
    filt = out["query"]["bool"]["filter"]
    # `term` -> `terms`, the JSON equivalent of `=` becoming `IN`.
    assert {"terms": {"loginArea.sign": ["7777ABCD", "7777AB"]}} in filt
    assert {"term": {"loginArea.sign": "7777ABCD"}} not in filt
    # Everything else is carried through untouched...
    assert {"range": {"date": {"gte": "2026-08-01"}}} in filt
    assert {"terms": {"unitId": ["OFF1"]}} in filt
    # ...and an exclusion is NOT widened, at any depth, for the reason the textual guard
    # declines a NOT IN.
    assert out["query"]["bool"]["must_not"] == [
        {"term": {"loginArea.sign": "7777ABCD"}}
    ]
    # The input is not mutated.
    assert dsl["query"]["bool"]["filter"][1] == {"term": {"loginArea.sign": "7777ABCD"}}

    # An existing `terms` list gains the stem, and a promoted field is merged beside a field
    # with no stem rather than overwriting it.
    mixed = {
        "bool": {
            "should": [
                {"terms": {"sign": ["7777ABCD"]}},
                {"term": {"sign": "7777ABCD", "unitId": "OFF1"}},
            ]
        }
    }
    got = widen_stem_literals_dsl(mixed, {"sign": {"7777ABCD": "7777AB"}}, "r")
    assert got["bool"]["should"][0] == {"terms": {"sign": ["7777ABCD", "7777AB"]}}
    assert got["bool"]["should"][1] == {
        "terms": {"sign": ["7777ABCD", "7777AB"]},
        "term": {"unitId": "OFF1"},
    }
    # Nothing declared -> the same object back, for every pack that has not opted in.
    assert widen_stem_literals_dsl(dsl, {}, "r") is dsl
    assert widen_stem_literals_dsl("not a dsl", stems, "r") == "not a dsl"


def test_the_reading_side_already_pairs_a_stem_with_its_long_form():
    """Why widening the PREDICATE is the whole fix, and no reading change is needed.

    `_identifiers_match` has always been prefix-tolerant and symmetric, so a register row
    holding the stem adjudicates against an acting identity carrying the trailing part. The
    asymmetry was only ever on the query side — which is what made it invisible: nothing
    downstream would have mismatched had the rows arrived.
    """
    from src.correlation import _identifiers_match, _matches_any_identifier, _norm_identifier

    assert _identifiers_match("7777AB", "7777ABCD")
    assert _identifiers_match("7777ABCD", "7777AB")
    assert not _identifiers_match("7777AB", "6666WX")
    # And the per-row form used by `_rows_matching`'s `normalize: identifier` clause.
    keys = {_norm_identifier("7777ABCD")}
    assert _matches_any_identifier("7777AB", keys)


# --- value combinations -------------------------------------------------------------------
#
# Input is retrieved rows, not a pack declaration. The failure is a full result: rows belong
# to other parties and every "did anything come back" check passes.


def test_value_tuples_replace_the_cross_product_of_the_and_ed_lists():
    """11 signs x 12 offices asked 132 combinations for 12 real ones — 90.9% nobody's."""
    from src.retrievers.query_guards import enforce_value_tuples

    tuples = [
        [("sign", ["AAA1"]), ("unitId", ["OFF1"])],
        [("sign", ["BBB2"]), ("unitId", ["OFF2"])],
    ]
    sql = (
        "SELECT a, b FROM t WHERE d >= '2026-08-01' AND sign IN ('AAA1','BBB2') "
        "AND unitId IN ('OFF1','OFF2') ORDER BY d"
    )
    out = enforce_value_tuples(sql, tuples, "r")
    # Components in the order the QUERY wrote them (`sign` before `unitId`), which is the same
    # promise the assertion below makes about every other conjunct, extended inside the group.
    assert "(sign = 'AAA1' AND unitId = 'OFF1')" in out
    assert "(sign = 'BBB2' AND unitId = 'OFF2')" in out
    # The two pairings that never occurred are gone, and with them the IN lists that asked
    # for them.
    assert "IN ('AAA1','BBB2')" not in out
    assert "OFF2' AND sign = 'AAA1" not in out
    assert "sign = 'AAA1' AND unitId = 'OFF2'" not in out
    # Every other conjunct keeps its text AND its position: a bound this guard reordered is a
    # bound the operator reading the published query has to re-derive.
    assert out.index("d >= '2026-08-01'") < out.index("unitId = 'OFF1'")
    assert out.strip().endswith("ORDER BY d")


def test_value_tuples_are_arity_agnostic_and_never_pair_specific():
    """A tuple is not a pair. Three components must work with no code that counts to two."""
    from src.retrievers.query_guards import enforce_value_tuples

    tuples = [
        [("sign", ["AAA1"]), ("unitId", ["OFF1"]), ("terminal", ["T1"])],
        [("sign", ["BBB2"]), ("unitId", ["OFF2"]), ("terminal", ["T2"])],
    ]
    sql = (
        "SELECT * FROM t WHERE sign IN ('AAA1','BBB2') AND unitId IN ('OFF1','OFF2') "
        "AND terminal IN ('T1','T2')"
    )
    out = enforce_value_tuples(sql, tuples, "r")
    assert "(sign = 'AAA1' AND unitId = 'OFF1' AND terminal = 'T1')" in out
    assert "(sign = 'BBB2' AND unitId = 'OFF2' AND terminal = 'T2')" in out
    # 2 of 8 — the arity is read off the declaration, not assumed.
    assert out.count(" OR ") == 1
    # And a single component is not a "combination": that is the per-type IN list already.
    assert enforce_value_tuples(sql, [[("sign", ["AAA1"])]], "r") == sql


def test_value_tuples_only_narrow_and_refuse_every_unproven_shape():
    """The four properties that make this strictly narrowing, one assertion each."""
    from src.retrievers.query_guards import enforce_value_tuples

    tuples = [
        [("sign", ["AAA1"]), ("unitId", ["OFF1"])],
        [("sign", ["BBB2"]), ("unitId", ["OFF2"])],
    ]
    # (1) A literal the pass did not harvest means the generator is constraining the column
    # for a reason nothing here can see — the whole rewrite is declined.
    unproven = "SELECT * FROM t WHERE sign IN ('AAA1','ZZZ9') AND unitId IN ('OFF1','OFF2')"
    assert enforce_value_tuples(unproven, tuples, "r") == unproven
    # (2) No harvested combination inside the query's own literals: a real finding about the
    # pass, not something to rewrite into a predicate matching nothing.
    crossed = "SELECT * FROM t WHERE sign = 'AAA1' AND unitId = 'OFF2'"
    assert enforce_value_tuples(crossed, tuples, "r") == crossed
    # (3) A component the query does not constrain contributes NO predicate — adding one
    # would be this guard filtering on something the generator never asked for.
    single = "SELECT * FROM t WHERE sign IN ('AAA1','BBB2')"
    assert enforce_value_tuples(single, tuples, "r") == single
    # (4) The lists already enumerating only real combinations leave the query untouched.
    exact = "SELECT * FROM t WHERE sign IN ('AAA1') AND unitId IN ('OFF1')"
    assert enforce_value_tuples(exact, tuples, "r") == exact
    # And anything it cannot read: a LIKE, a function, a subquery's WHERE, no WHERE at all.
    for text in (
        "SELECT * FROM t WHERE sign LIKE 'AAA1%' AND unitId IN ('OFF1','OFF2')",
        "SELECT * FROM t WHERE UPPER(sign) IN ('AAA1','BBB2') AND unitId IN ('OFF1','OFF2')",
        "SELECT * FROM t WHERE sign IN ('AAA1','BBB2') AND unitId IN ('OFF1','OFF2') "
        "AND id IN (SELECT id FROM u WHERE x = 1)",
        "SELECT * FROM t",
    ):
        assert enforce_value_tuples(text, tuples, "r") == text
    # Nothing harvested -> byte-identical, which is every single-pass run.
    sql = "SELECT * FROM t WHERE sign IN ('AAA1','BBB2') AND unitId IN ('OFF1','OFF2')"
    assert enforce_value_tuples(sql, [], "r") == sql


def test_value_tuples_splice_the_generators_own_literals_and_spellings():
    """It re-emits the text it read: a re-quoted literal is this guard choosing a syntax."""
    from src.retrievers.query_guards import enforce_value_tuples

    tuples = [
        [("loginArea.sign", ["AAA1"]), ("office.id", ["OFF1"])],
        [("loginArea.sign", ["BBB2"]), ("office.id", ["OFF2"])],
    ]
    # Qualified/backticked column spellings survive verbatim — one backend upper-cases every
    # identifier and another needs the struct path exactly as the schema gave it.
    sql = (
        "SELECT * FROM t WHERE `t`.`loginArea`.`sign` IN ('AAA1','BBB2') "
        "AND office.id IN ('OFF1','OFF2')"
    )
    out = enforce_value_tuples(sql, tuples, "r")
    assert "`t`.`loginArea`.`sign` = 'AAA1'" in out
    assert "office.id = 'OFF1'" in out
    # A value the pass harvested at two precisions (its declared stem) matches whichever one
    # the generator actually wrote, and the OTHER is not introduced.
    stemmed = [[("sign", ["7777ABCD", "7777AB"]), ("unitId", ["OFF1"])]]
    got = enforce_value_tuples(
        "SELECT * FROM t WHERE sign IN ('7777AB','OTHER') AND unitId IN ('OFF1','OFF2')",
        stemmed + [[("sign", ["OTHER"]), ("unitId", ["OFF2"])]],
        "r",
    )
    assert "sign = '7777AB'" in got and "7777ABCD" not in got


def test_value_tuples_run_on_the_esql_pipeline_per_where_stage():
    """A pipe dialect has no single WHERE, and the components may be split across stages."""
    from src.retrievers.query_guards import enforce_value_tuples

    tuples = [
        [("sign", ["AAA1"]), ("unitId", ["OFF1"])],
        [("sign", ["BBB2"]), ("unitId", ["OFF2"])],
    ]
    esql = (
        "FROM idx-* | WHERE @timestamp >= '2026-08-01' "
        "| WHERE sign IN ('AAA1','BBB2') AND unitId IN ('OFF1','OFF2') "
        "| KEEP sign, unitId | LIMIT 500"
    )
    out = enforce_value_tuples(esql, tuples, "r", dialect="esql")
    assert "(sign = 'AAA1' AND unitId = 'OFF1') OR (sign = 'BBB2' AND unitId = 'OFF2')" in out
    # The other stages are untouched and still in order.
    assert out.startswith("FROM idx-* | WHERE @timestamp >= '2026-08-01' |")
    assert out.rstrip().endswith("| KEEP sign, unitId | LIMIT 500")
    # A component split across two stages is still one conjunction to the backend, but not one
    # body to this reader — so it declines rather than rewriting half of it.
    split = "FROM idx | WHERE sign IN ('AAA1','BBB2') | WHERE unitId IN ('OFF1','OFF2')"
    assert enforce_value_tuples(split, tuples, "r", dialect="esql") == split


def test_value_tuples_on_the_dsl_route_nest_an_or_of_ands():
    """Both query families or it is not a guarantee (the mistake every twin here fixes)."""
    from src.retrievers.query_guards import enforce_value_tuples_dsl

    tuples = [
        [("sign", ["AAA1"]), ("unitId", ["OFF1"])],
        [("sign", ["BBB2"]), ("unitId", ["OFF2"])],
    ]
    dsl = {
        "query": {
            "bool": {
                "filter": [
                    {"range": {"date": {"gte": "2026-08-01"}}},
                    {"terms": {"sign": ["AAA1", "BBB2"]}},
                    {"terms": {"unitId": ["OFF1", "OFF2"]}},
                ]
            }
        },
        "size": 500,
    }
    out = enforce_value_tuples_dsl(dsl, tuples, "r")
    filt = out["query"]["bool"]["filter"]
    assert filt[0] == {"range": {"date": {"gte": "2026-08-01"}}}
    group = filt[1]["bool"]
    assert group["minimum_should_match"] == 1
    assert group["should"] == [
        {"bool": {"filter": [{"term": {"sign": "AAA1"}}, {"term": {"unitId": "OFF1"}}]}},
        {"bool": {"filter": [{"term": {"sign": "BBB2"}}, {"term": {"unitId": "OFF2"}}]}},
    ]
    assert len(filt) == 2  # the second value list was replaced, not left beside the group
    # The input is not mutated.
    assert dsl["query"]["bool"]["filter"][1] == {"terms": {"sign": ["AAA1", "BBB2"]}}

    # A `should` is already an OR: no cross product is being asked, and promoting one would
    # be inventing a constraint.
    should = {
        "query": {
            "bool": {
                "should": [
                    {"terms": {"sign": ["AAA1", "BBB2"]}},
                    {"terms": {"unitId": ["OFF1", "OFF2"]}},
                ],
                "minimum_should_match": 1,
            }
        }
    }
    assert enforce_value_tuples_dsl(should, tuples, "r") == should
    # Everything the textual guard refuses, refused here for the same reasons.
    for clause in (
        {"terms": {"sign": ["AAA1", "ZZZ9"]}},           # a value never harvested
        {"prefix": {"sign": "AAA1"}},                     # not a membership test
        {"term": {"sign": {"value": "AAA1", "boost": 2}}},  # says more than a value
    ):
        guarded = {"query": {"bool": {"filter": [clause, {"terms": {"unitId": ["OFF1", "OFF2"]}}]}}}
        assert enforce_value_tuples_dsl(guarded, tuples, "r") == guarded
    assert enforce_value_tuples_dsl(dsl, [], "r") is dsl
    assert enforce_value_tuples_dsl("not a dsl", tuples, "r") == "not a dsl"


def test_value_tuples_read_a_form_split_or_group_as_ONE_slot():
    """A component's conjunct may name SEVERAL columns, and the guard before this one writes it.

    `relax_form_conjunction` runs first and turns an AND across one entity type's two form
    columns into `(sign = ... OR login = ...)`. So the conjunction's third member is a GROUP, a
    leaf-only reader resolved two components out of three, and — with the group's own arms then
    counted per column — the arithmetic disagreed with the arms it was checking. The topology is
    read off the QUERY: columns OR-ed inside one conjunct are alternatives (one SLOT), and the
    combination is across slots. That is the shape a per-record request has, so this is the
    headline case and not an edge one: a unit AND either spelling of a subject, N times.
    """
    from src.retrievers.query_guards import enforce_value_tuples

    tuples = [
        [("unitId", ["OFF1"]), ("sign", ["AAA1"]), ("channelId", ["U1"])],
        [("unitId", ["OFF2"]), ("sign", ["BBB2"]), ("channelId", ["U2"])],
    ]
    sql = (
        "SELECT * FROM t WHERE record = 'P1' AND unitId IN ('OFF1','OFF2') "
        "AND (sign IN ('AAA1','BBB2') OR channelId IN ('U1','U2')) AND org = 'SV'"
    )
    out = enforce_value_tuples(sql, tuples, "r")
    # The OR inside the slot SURVIVES — it is the form disjunction, and AND-ing the two
    # spellings asserts both appear on one row, which is the zero-row failure that guard exists
    # to prevent. Reproducing the generator's topology is what keeps this narrowing.
    assert "(unitId = 'OFF1' AND (sign = 'AAA1' OR channelId = 'U1'))" in out
    assert "(unitId = 'OFF2' AND (sign = 'BBB2' OR channelId = 'U2'))" in out
    assert out.count(" OR (unitId") == 1  # two arms, one OR between them
    # The pairings that never occurred are gone, in both directions.
    assert "IN ('OFF1','OFF2')" not in out
    assert "'AAA1' OR channelId = 'U2'" not in out
    assert "OFF1' AND (sign = 'BBB2" not in out
    # The two conjuncts that are not components keep their text and their position.
    assert out.index("record = 'P1'") < out.index("unitId = 'OFF1'") < out.index("org = 'SV'")
    # And a slot is COVERED by a combination naming ONE of its columns — coverage is per slot
    # and not per column, because a slot's columns are alternatives. Per column, the record
    # carrying only the login (this source binds no column for its other spelling) covers
    # nothing on that slot and its arm is dropped, which is the one direction forbidden here.
    partial = enforce_value_tuples(
        "SELECT * FROM t WHERE unitId IN ('OFF1','OFF2') AND (sign = 'AAA1' OR channelId = 'U2')",
        [
            [("unitId", ["OFF1"]), ("sign", ["AAA1"])],
            [("unitId", ["OFF2"]), ("channelId", ["U2"])],
        ],
        "r",
    )
    assert "(unitId = 'OFF1' AND sign = 'AAA1')" in partial
    assert "(unitId = 'OFF2' AND channelId = 'U2')" in partial


def test_value_tuples_descend_into_a_parenthesised_and_chain():
    """A component the generator wrote INSIDE a nested AND group is still a slot.

    `AND` is associative, so `a AND (b AND c)` constrains exactly what `a AND b AND c` does — but
    a slot is a top-level conjunct, so a body whose two component lists sat inside one group
    beside the window bounds resolved ZERO component columns, declined for having fewer than two
    slots, and published the cross product. Measured on one live statement: 7 records' worth of
    components went out as their 49 pairings, 42 of them belonging to other parties — and the
    decline was logged as a fact about the pass rather than as a body the guard could not read.
    """
    from src.retrievers.query_guards import enforce_value_tuples

    tuples = [
        [("unitId", ["OFF1"]), ("sign", ["AAA1"])],
        [("unitId", ["OFF2"]), ("sign", ["BBB2"])],
    ]
    nested = (
        "SELECT * FROM t WHERE ts >= '2026-01-01' "
        "AND (d >= '2026-01-01' AND d <= '2026-01-02' "
        "AND sign IN ('AAA1','BBB2') AND unitId IN ('OFF1','OFF2'))"
    )
    out = enforce_value_tuples(nested, tuples, "r")
    assert "(sign = 'AAA1' AND unitId = 'OFF1')" in out
    assert "(sign = 'BBB2' AND unitId = 'OFF2')" in out
    assert "IN ('OFF1','OFF2')" not in out  # the cross product is gone
    assert "'AAA1' AND unitId = 'OFF2'" not in out  # and so is every pairing nobody observed
    # Only the group's parentheses go: every conjunct the descent hoisted keeps its own text.
    for kept in ("ts >= '2026-01-01'", "d >= '2026-01-01'", "d <= '2026-01-02'"):
        assert kept in out
    # Recursive, for the same reason it is sound once: a component two groups deep is a conjunct.
    deeper = (
        "SELECT * FROM t WHERE ts >= '2026-01-01' "
        "AND (d >= '2026-01-01' AND (sign IN ('AAA1','BBB2') AND unitId IN ('OFF1','OFF2')))"
    )
    assert "(sign = 'AAA1' AND unitId = 'OFF1')" in enforce_value_tuples(deeper, tuples, "r")


def test_the_and_chain_descent_refuses_every_shape_it_would_misread():
    """The descent DROPS the group's parentheses, so it is refused wherever that would lie.

    Read as a set: each refusal is a text that opens with a parenthesis and is not one AND chain,
    and hoisting any of them out of its own group changes the predicate rather than reassociating
    it. `_flatten_and_conjuncts` keeps `_split_top_level`'s contract exactly — `None` only for an
    unbalanced body — so a refusal leaves the conjunct in place and the caller reads it as it did
    before.
    """
    from src.retrievers.query_guards import (
        _flatten_and_conjuncts,
        _paren_and_chain,
        enforce_value_tuples,
    )

    # (1) An OR anywhere at the group's own depth: the parent's AND does not reach the arms.
    assert _paren_and_chain("(a AND b OR c)") is None
    assert _paren_and_chain("((a AND b) OR (c AND d))") is None
    # (2) A negated group opens with a token, so it is refused there — hoisting its conjuncts
    # would put each of them under no negation at all.
    assert _paren_and_chain("NOT (a AND b)") is None
    # (3) The opening parenthesis must close at the very END, or the text is not ONE group: a
    # function call over an AND chain, and a comparison BETWEEN two of them, both read as one
    # otherwise.
    assert _paren_and_chain("(a AND b) = (c AND d)") is None
    assert _paren_and_chain("(a AND b) IS NOT NULL") is None
    # (4) One conjunct is not a chain — there is nothing to reassociate and the parentheses may
    # be doing work (`(a)` beside a cast, an operator precedence the writer chose).
    assert _paren_and_chain("(a = 1)") is None
    # (5) The depth walk is quote-aware, so a literal carrying a parenthesis or an `AND` neither
    # unbalances the scan nor splits the chain.
    quoted = _paren_and_chain("(a = 'x)y' AND b = 'p AND q')")
    assert [part.strip() for part in quoted or []] == ["a = 'x)y'", "b = 'p AND q'"]
    # (6) An unbalanced text is refused BY THIS READER and not by its caller's contract — the
    # slice takes the first and last character off, and `(a AND b` would otherwise hoist `a` and
    # an empty conjunct out of half a predicate.
    assert _paren_and_chain("(a AND b") is None
    assert _paren_and_chain("a AND b)") is None
    # ...and so is a chain with no group at all, which is the same slice read one step further:
    # `a = 1 AND b = 2` loses its first and last character and still splits in two, so without
    # the opening test this reader would hand back two mangled halves of a real predicate.
    assert _paren_and_chain("a = 1 AND b = 2") is None
    # (7) `None` for an unbalanced body, exactly as the reader it replaces — never a best effort.
    assert _flatten_and_conjuncts("a AND (b AND c") is None
    assert [p.strip() for p in _flatten_and_conjuncts("a AND (b AND c)")] == ["a", "b", "c"]
    # And end to end: a negated group holding both components is left byte-identical, which is
    # the direction that matters — the guard may only narrow, and it cannot read that shape.
    tuples = [
        [("unitId", ["OFF1"]), ("sign", ["AAA1"])],
        [("unitId", ["OFF2"]), ("sign", ["BBB2"])],
    ]
    negated = (
        "SELECT * FROM t WHERE ts >= '2026-01-01' "
        "AND NOT (sign IN ('AAA1','BBB2') AND unitId IN ('OFF1','OFF2'))"
    )
    assert enforce_value_tuples(negated, tuples, "r") == negated


def test_value_tuples_refuse_every_slot_topology_they_cannot_prove_narrows():
    """One slot is not a combination, and an unreadable arm leaves its whole conjunct alone."""
    from src.retrievers.query_guards import enforce_value_tuples

    tuples = [
        [("unitId", ["OFF1"]), ("sign", ["AAA1"]), ("channelId", ["U1"])],
        [("unitId", ["OFF2"]), ("sign", ["BBB2"]), ("channelId", ["U2"])],
    ]
    # (1) Every component inside ONE group: the query is already asking for alternatives, so
    # there is no cross product and an OR-of-ANDs would be strictly WIDER on the third column.
    one_slot = (
        "SELECT * FROM t WHERE record = 'P1' AND (unitId IN ('OFF1','OFF2') "
        "OR sign IN ('AAA1','BBB2') OR channelId IN ('U1','U2'))"
    )
    assert enforce_value_tuples(one_slot, tuples, "r") == one_slot
    # (2) One arm this pass harvested nothing for declines the WHOLE conjunct — it stays
    # AND-ed as generated, which can only narrow — and the two components left are one slot.
    stranger = (
        "SELECT * FROM t WHERE unitId IN ('OFF1','OFF2') "
        "AND (sign IN ('AAA1','BBB2') OR role IN ('AGT'))"
    )
    assert enforce_value_tuples(stranger, tuples, "r") == stranger
    # (3) One column named twice inside a group: which arm is the component is unanswerable.
    twice = (
        "SELECT * FROM t WHERE unitId IN ('OFF1','OFF2') "
        "AND (sign = 'AAA1' OR sign = 'BBB2') AND channelId IN ('U1','U2')"
    )
    out = enforce_value_tuples(twice, tuples, "r")
    assert "sign = 'AAA1' OR sign = 'BBB2'" in out  # left exactly as written
    # ...and the two readable components beside it are two slots, so they still narrow.
    assert "(unitId = 'OFF1' AND channelId = 'U1')" in out
    # (4) An arm that is not a plain membership test declines the conjunct, not just the arm.
    like = (
        "SELECT * FROM t WHERE unitId IN ('OFF1','OFF2') "
        "AND (sign LIKE 'AAA%' OR channelId IN ('U1','U2'))"
    )
    assert enforce_value_tuples(like, tuples, "r") == like


def test_value_tuples_on_the_dsl_route_read_a_nested_should_as_one_slot():
    """The DSL twin of the slot, and the recursion that shape made unbounded.

    Each arm this guard writes is a `bool.filter` — a conjunction of one value per slot — so a
    top-down walker re-entered its own output and collapsed it again. It terminated only while
    every slot was a single column, where the observed count equals the asked product and the
    no-op check bails; a slot of two OR-ed forms makes the product larger and the recursion
    never ends. The walk is bottom-up and the rewrite's output is never walked.
    """
    from src.retrievers.query_guards import enforce_value_tuples_dsl

    tuples = [
        [("unitId", ["OFF1"]), ("sign", ["AAA1"]), ("channelId", ["U1"])],
        [("unitId", ["OFF2"]), ("sign", ["BBB2"]), ("channelId", ["U2"])],
    ]
    form_slot = {
        "bool": {
            "should": [
                {"terms": {"sign": ["AAA1", "BBB2"]}},
                {"terms": {"channelId": ["U1", "U2"]}},
            ],
            "minimum_should_match": 1,
        }
    }
    dsl = {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"record": "P1"}},
                    {"terms": {"unitId": ["OFF1", "OFF2"]}},
                    form_slot,
                ]
            }
        }
    }
    out = enforce_value_tuples_dsl(dsl, tuples, "r")  # must RETURN, not recurse
    filt = out["query"]["bool"]["filter"]
    assert filt[0] == {"term": {"record": "P1"}}
    assert len(filt) == 2
    group = filt[1]["bool"]
    assert group["minimum_should_match"] == 1
    assert group["should"] == [
        {
            "bool": {
                "filter": [
                    {"term": {"unitId": "OFF1"}},
                    {
                        "bool": {
                            "should": [
                                {"term": {"sign": "AAA1"}},
                                {"term": {"channelId": "U1"}},
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                ]
            }
        },
        {
            "bool": {
                "filter": [
                    {"term": {"unitId": "OFF2"}},
                    {
                        "bool": {
                            "should": [
                                {"term": {"sign": "BBB2"}},
                                {"term": {"channelId": "U2"}},
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                ]
            }
        },
    ]
    assert dsl["query"]["bool"]["filter"][2] is form_slot  # input not mutated
    # An `msm` of 2 is a conjunction wearing a disjunction's clothes: the clause stays as
    # generated, and the one component left beside it is not a combination.
    conjunctive = {
        "query": {
            "bool": {
                "filter": [
                    {"terms": {"unitId": ["OFF1", "OFF2"]}},
                    {
                        "bool": {
                            "should": [
                                {"terms": {"sign": ["AAA1", "BBB2"]}},
                                {"terms": {"channelId": ["U1", "U2"]}},
                            ],
                            "minimum_should_match": 2,
                        }
                    },
                ]
            }
        }
    }
    assert enforce_value_tuples_dsl(conjunctive, tuples, "r") == conjunctive
    # A `must_not` anywhere in the group is an exclusion: collapsing it would make the
    # excluded values MANDATORY inside the arm.
    excluding = {
        "query": {
            "bool": {
                "filter": [
                    {"terms": {"unitId": ["OFF1", "OFF2"]}},
                    {
                        "bool": {
                            "should": [{"terms": {"sign": ["AAA1", "BBB2"]}}],
                            "must_not": [{"term": {"channelId": "U9"}}],
                        }
                    },
                ]
            }
        }
    }
    assert enforce_value_tuples_dsl(excluding, tuples, "r") == excluding


def test_value_tuples_on_the_dsl_route_read_a_slot_that_asks_one_component_twice():
    """A slot constraining one component on several fields was read as unreadable and dropped.

    A generator may OR a second field for the same component; both reduce to the same value.
    The licence is the guard's own contract: a slot is one conjunct, its arms are alternatives,
    and each arm is re-emitted verbatim, so every term appears as generated and the result is
    a subset. AND-ing the fields instead claims one row carries the value on every field.
    """
    from src.retrievers.query_guards import enforce_value_tuples_dsl

    tuples = [
        [("grp", ["G1"]), ("a.k", ["V1"])],
        [("grp", ["G2"]), ("a.k", ["V2"])],
    ]
    two_fields = {
        "bool": {
            "should": [
                {"terms": {"a.k": ["V1", "V2"]}},
                {"terms": {"b.c.k": ["V1", "V2"]}},
            ],
            "minimum_should_match": 1,
        }
    }
    dsl = {
        "query": {
            "bool": {
                "filter": [{"terms": {"grp": ["G1", "G2"]}}, two_fields]
            }
        }
    }
    filt = enforce_value_tuples_dsl(dsl, tuples, "r")["query"]["bool"]["filter"]
    assert len(filt) == 1  # both slots replaced by the one OR-of-ANDs
    group = filt[0]["bool"]
    assert group["minimum_should_match"] == 1
    assert group["should"] == [
        {
            "bool": {
                "filter": [
                    {"term": {"grp": "G1"}},
                    {
                        "bool": {
                            "should": [
                                {"term": {"a.k": "V1"}},
                                {"term": {"b.c.k": "V1"}},
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                ]
            }
        },
        {
            "bool": {
                "filter": [
                    {"term": {"grp": "G2"}},
                    {
                        "bool": {
                            "should": [
                                {"term": {"a.k": "V2"}},
                                {"term": {"b.c.k": "V2"}},
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                ]
            }
        },
    ]
    # Every term of every arm was in the group as generated: the rewrite can only narrow.
    generated = {("a.k", "V1"), ("a.k", "V2"), ("b.c.k", "V1"), ("b.c.k", "V2")}
    for arm in group["should"]:
        for term in arm["bool"]["filter"][1]["bool"]["should"]:
            assert next(iter(term["term"].items())) in generated
    assert dsl["query"]["bool"]["filter"][1] is two_fields  # input not mutated
    # A component claimed by an EARLIER slot is still refused, and that clause is left exactly as
    # generated: two slots asking one component is not two components, and a rewrite that ANDed
    # them would assert one row carries the value on both.
    third = {"terms": {"z.k": ["V1", "V2"]}}
    spread = {
        "query": {
            "bool": {
                "filter": [
                    {"terms": {"grp": ["G1", "G2"]}},
                    {"terms": {"a.k": ["V1", "V2"]}},
                    third,
                ]
            }
        }
    }
    out = enforce_value_tuples_dsl(spread, tuples, "r")["query"]["bool"]["filter"]
    assert out[-1] == third
    assert len(out[0]["bool"]["should"]) == 2


def test_value_tuples_run_on_every_backend_after_the_two_additive_guards():
    """The one position that is correct, and every neighbour is the reason.

    After the identity rewrite: run before it, the OR-of-ANDs produced here reads back as a
    flat identity group and is flattened into `office = 'O1' AND office = 'O2'` (0 rows).
    After the anchor and key-presence guards too: they splice `WHERE <clause> AND (<body>)`,
    adding per-type value lists that are exactly what this guard narrows; run before them it
    declines, leaving the cross product they then publish unnarrowed — measured on one live
    2-arm statement, 800 pairings asked for the 25 identities the incident named.
    Before the two widenings, which turn a component equality into a wildcard group.
    """
    import inspect

    from src.retrievers import (databricks_retriever, elasticsearch_retriever,
                                kibana_retriever, snowflake_retriever)

    for module, guard in (
        (databricks_retriever, "enforce_value_tuples"),
        (elasticsearch_retriever, "enforce_value_tuples"),
        (snowflake_retriever, "enforce_value_tuples"),
        (kibana_retriever, "enforce_value_tuples_dsl"),
    ):
        body = inspect.getsource(module)
        assert guard in body, f"{module.__name__} never enforces value combinations"
        # ...and through the shared resolver, not a second answer to "which column is this".
        assert "value_tuple_columns" in body, f"{module.__name__} resolves its own columns"
        retrieve = body[body.index("async def retrieve") :]
        assert "_enforce_value_tuples" in retrieve, f"{module.__name__}.retrieve skips it"
        here = retrieve.index("_enforce_value_tuples")
        assert retrieve.index("_enforce_identity_scope") < here, (
            f"{module.__name__} narrows combinations BEFORE the identity rewrite, which "
            "flattens the OR-of-ANDs back into a 0-row conjunction"
        )
        for earlier in ("_enforce_subject_anchor", "_enforce_key_presence"):
            assert retrieve.index(earlier) < here, (
                f"{module.__name__} narrows combinations BEFORE {earlier}, whose per-TYPE "
                "value lists ARE the cross product this guard narrows — added afterwards "
                "they are published unnarrowable"
            )
        for later in ("widen_stem_literals", "enforce_partition_bounds"):
            assert here < retrieve.index(later), f"{module.__name__} runs {later} first"


def test_the_identity_rewrite_really_does_flatten_an_or_of_ands():
    """The measurement behind the ordering above, asserted rather than asserted-about.

    An ordering test that only reads source text proves the call sites are in a sequence; this
    one proves the sequence MATTERS, by running the neighbour on this guard's output and
    watching it produce the 0-row predicate. Without it, a later reordering looks harmless.
    """
    from src.retrievers.query_guards import (enforce_identity_scope,
                                             enforce_value_tuples)

    tuples = [
        [("sign", ["AAA1"]), ("unitId", ["OFF1"])],
        [("sign", ["BBB2"]), ("unitId", ["OFF2"])],
    ]
    sql = "SELECT * FROM t WHERE sign IN ('AAA1','BBB2') AND unitId IN ('OFF1','OFF2')"
    narrowed = enforce_value_tuples(sql, tuples, "r")
    flattened = enforce_identity_scope(narrowed, ["unitId"], ["sign"], "r")
    # Both offices AND-ed on one column: no document can satisfy it.
    assert "unitId = 'OFF1' AND unitId = 'OFF2'" in flattened
    # In the wired order the rewrite runs first, sees no OR-group, and leaves the conjunction
    # for this guard to narrow.
    assert enforce_identity_scope(sql, ["unitId"], ["sign"], "r") == sql
    assert enforce_value_tuples(sql, tuples, "r") == narrowed


# ── the tenth shape: a key member the query never mentions ──
#
# `enforce_conjunction` only turns an OR it can find; `enforce_identity_scope` is gated on
# `identity_scopes`/`identity_synonyms`, which a keyed lookup does not declare. A missing
# member makes `key_was_enforced` False, so an empty result reads as "source said nothing"
# rather than as a decisive negative.


def test_a_key_member_the_generated_query_never_mentions_is_ADDED():
    """The live query, and the two things it got wrong, asserted separately.

    The predicate assertion is behavioural, not textual: post-guard it is evaluated against
    the SUBJECT's row and against a STRANGER's — a different sign in the same unit, which is
    what the register holds hundreds of — and the stranger must fail. The second assertion is
    the one the report depended on: with the member in the predicate, an EMPTY result from
    this source is a decisive negative rather than a gap.
    """
    from src.retrievers.query_guards import (_is_constrained,
                                             _is_constrained_conjunctively,
                                             enforce_key_presence,
                                             key_was_enforced)

    key_values = {
        "org_unit": {"unitId": ["AAA1B02CD"]},
        "user": {"sign": ["0101XY"]},
    }
    sql = (
        "SELECT unitId, sign, profile, processedDate FROM reg "
        "WHERE unitId = 'AAA1B02CD' LIMIT 500"
    )
    subject = {"unitId": "AAA1B02CD", "sign": "0101XY"}
    stranger = {"unitId": "AAA1B02CD", "sign": "0404WX"}
    assert _sql_predicate_matches(sql, stranger), "premise: the stranger matches pre-guard"
    # The projection NAMES the column it fails to filter, and that must not read as
    # constrained — the loose test agrees here only because the filter region excludes the
    # SELECT list, which is exactly why the strong test is the one that decides.
    assert not _is_constrained(sql, "sign", "sql")
    assert not _is_constrained_conjunctively(sql, ["sign"], "sql")

    out = enforce_key_presence(sql, key_values, "robot_register")
    assert out != sql
    flat = " ".join(out.split())
    assert _sql_predicate_matches(flat, subject), flat
    assert not _sql_predicate_matches(flat, stranger), flat
    # It narrows rows and nothing else: the row cap and the projection survive.
    assert flat.startswith("SELECT unitId, sign, profile, processedDate FROM reg")
    assert flat.endswith("LIMIT 500")
    # ...and the unit predicate the generator DID write is not duplicated.
    assert flat.count("unitId = 'AAA1B02CD'") == 1, flat

    # The second half of the same defect: what ZERO rows means. Pre-guard the key was not
    # enforced, so an empty result is a gap; post-guard it is the finding.
    assert not key_was_enforced(sql, ["unitId", "sign"])
    assert key_was_enforced(flat, ["unitId", "sign"])


def test_key_presence_leaves_an_ALREADY_keyed_query_exactly_as_generated():
    """Four ways there is nothing to add, and all four publish the query as written."""
    from src.retrievers.query_guards import enforce_key_presence

    key_values = {
        "org_unit": {"unitId": ["AAA1B02CD"]},
        "user": {"sign": ["0101XY"]},
    }
    # 1. Both members constrained — the 52 of 57 real queries that were already right.
    keyed = (
        "SELECT * FROM reg WHERE unitId = 'AAA1B02CD' AND sign = '0101XY' LIMIT 500"
    )
    assert enforce_key_presence(keyed, key_values, "r") == keyed
    # 2. Constrained TIGHTER than this guard would have: a narrower member is left alone,
    #    because the guard's licence is that it can only narrow, and it has nothing to add.
    tighter = (
        "SELECT * FROM reg WHERE unitId = 'AAA1B02CD' AND sign = '0101XY' "
        "AND profile = 'ROBOT' LIMIT 500"
    )
    assert enforce_key_presence(tighter, key_values, "r") == tighter
    # 3. Nothing resolved — a source with no key, or an incident carrying no member. There is
    #    no literal to invent, and inventing one is the direction forbidden here.
    assert enforce_key_presence(keyed, {}, "r") == keyed
    assert enforce_key_presence(keyed, {"user": {}}, "r") == keyed
    assert enforce_key_presence(keyed, {"user": {"sign": []}}, "r") == keyed
    assert enforce_key_presence("", key_values, "r") == ""
    # 4. A member under an OR beside a WIDER arm is not constrained by it — that arm binds
    #    no row of its own — so the member IS injected, which is the direction the loose
    #    test gets wrong and the strong one gets right.
    ored = (
        "SELECT * FROM reg WHERE unitId = 'AAA1B02CD' "
        "AND (sign = '0101XY' OR profile = 'ROBOT')"
    )
    out = enforce_key_presence(ored, key_values, "r")
    assert out != ored, out
    assert not _sql_predicate_matches(
        " ".join(out.split()),
        {"unitId": "AAA1B02CD", "sign": "0404WX", "profile": "ROBOT"},
    )
    # The bound: the conjunct-region reader runs to end-of-text, so a trailing clause rides
    # inside the last conjunct and the group is unrecognisable; the loose fallback declines.
    # This is a property of the shared conjunctive helper that `enforce_subject_anchor` also
    # reads; changing it is a behaviour change to that family and needs its own measurement.
    assert enforce_key_presence(ored + " LIMIT 500", key_values, "r") == ored + " LIMIT 500"


def test_key_presence_asks_one_TYPES_columns_as_a_GROUP_and_ANDs_the_TYPES():
    """AND between entity TYPES, OR within one type's values and forms.

    The rule the eighth and ninth guards share, and this guard is additive, so getting it
    wrong writes the claim that both spellings of one identity appear on the same row — the
    zero-row failure arriving through the guard meant to prevent it.
    """
    from src.retrievers.query_guards import enforce_key_presence

    out = enforce_key_presence(
        "SELECT * FROM t WHERE eventDate = DATE'2026-08-16' LIMIT 500",
        {
            "org_unit": {"unitId": ["AAA1B02CD"]},
            "user": {"sign": ["0101XY"], "userId": ["AUSERNAME"]},
        },
        "auth",
    )
    flat = " ".join(out.split())
    assert "(sign = '0101XY' OR userId = 'AUSERNAME')" in flat, flat
    assert "sign = '0101XY' AND userId" not in flat, flat
    # The two TYPES are AND-ed, and a row carrying only one of them is not returned.
    assert "WHERE unitId = 'AAA1B02CD' AND (sign" in flat, flat
    assert _sql_predicate_matches(
        flat, {"unitId": "AAA1B02CD", "sign": "0101XY", "userId": "OTHER",
               "eventDate": "2026-08-16"}
    )
    assert not _sql_predicate_matches(
        flat, {"unitId": "OTHER", "sign": "0101XY", "userId": "AUSERNAME",
               "eventDate": "2026-08-16"}
    )
    # The generated body keeps its own bounds, parenthesised rather than re-associated.
    assert "eventDate = DATE'2026-08-16'" in flat and flat.endswith("LIMIT 500")


def test_key_presence_on_the_esql_route_splices_its_own_stage():
    """ES|QL is a pipeline, so the clause is a WHERE stage and the operator is `==`."""
    from src.retrievers.query_guards import enforce_key_presence

    out = enforce_key_presence(
        'FROM sessions-* | WHERE unitId == "AAA1B02CD" | LIMIT 500',
        {"org_unit": {"unitId": ["AAA1B02CD"]}, "user": {"sign": ["0101XY"]}},
        "sessions",
        dialect="esql",
    )
    assert 'WHERE sign == "0101XY"' in out, out
    assert out.startswith("FROM sessions-* |")
    assert out.rstrip().endswith("LIMIT 500")
    # Already keyed on that route too, and the same-shaped query is then untouched.
    keyed = 'FROM sessions-* | WHERE unitId == "X" AND sign == "0101XY" | LIMIT 500'
    assert (
        enforce_key_presence(
            keyed,
            {"org_unit": {"unitId": ["X"]}, "user": {"sign": ["0101XY"]}},
            "sessions",
            dialect="esql",
        )
        == keyed
    )


def test_key_presence_dsl_nests_the_generated_query_beside_the_members():
    """The DSL twin — or the guarantee holds only on the backend the defect was measured on."""
    from src.retrievers.query_guards import enforce_key_presence_dsl

    dsl = {
        "bool": {
            "should": [{"term": {"unitId": "AAA1B02CD"}}],
            "minimum_should_match": 1,
        }
    }
    out = enforce_key_presence_dsl(
        dsl,
        {
            "org_unit": {"unitId": ["AAA1B02CD"]},
            "user": {"sign": ["0101XY"], "userId": ["AUSERNAME"]},
        },
        "reg",
    )
    clauses = out["bool"]["filter"]
    # The unit is already named in the query, so only the actor is added — as ONE nested
    # should over its two columns, with an explicit minimum_should_match (the default is 1
    # only while the bool carries no `must`, and nesting the query under `filter` puts one there).
    assert len(clauses) == 2, clauses
    assert clauses[0]["bool"]["minimum_should_match"] == 1
    assert sorted(next(iter(c["terms"])) for c in clauses[0]["bool"]["should"]) == [
        "sign",
        "userId",
    ]
    assert clauses[1] == dsl
    # Not mutated, and a member the query already names anywhere is left entirely alone.
    assert "sign" not in json.dumps(dsl)
    named = {"bool": {"filter": [{"terms": {"sign": ["0101XY"]}}]}}
    assert enforce_key_presence_dsl(named, {"user": {"sign": ["0101XY"]}}, "r") == named
    # Nothing resolved, and a non-dict query: returned as given.
    assert enforce_key_presence_dsl(dsl, {}, "r") == dsl
    assert enforce_key_presence_dsl(None, {"user": {"sign": ["X"]}}, "r") is None
    # A single column needs no should wrapper at all.
    one = enforce_key_presence_dsl(
        {"match_all": {}}, {"user": {"sign": ["0101XY"]}}, "r"
    )
    assert one["bool"]["filter"][0] == {"terms": {"sign": ["0101XY"]}}


def test_key_presence_runs_on_every_backend_after_the_anchor():
    """Wired into all four, in the one position that is correct — or it is not a guarantee.

    AFTER the anchor because the two are the same addition on different declarations, so
    running first would inject the same member twice; BEFORE the widening because a member
    added here is exactly a predicate that may need its declared stem offered beside it.
    """
    import inspect

    from src.retrievers import (databricks_retriever, elasticsearch_retriever,
                                kibana_retriever, snowflake_retriever)
    from src.retrievers.elasticsearch_retriever import ElasticsearchRetriever
    from src.retrievers.kibana_retriever import KibanaRetriever
    from src.retrievers.snowflake_retriever import SnowflakeRetriever

    for module, cls, guard in (
        (databricks_retriever, DatabricksRetriever, "enforce_key_presence"),
        (elasticsearch_retriever, ElasticsearchRetriever, "enforce_key_presence"),
        (snowflake_retriever, SnowflakeRetriever, "enforce_key_presence"),
        (kibana_retriever, KibanaRetriever, "enforce_key_presence_dsl"),
    ):
        body = inspect.getsource(module)
        assert guard in body, f"{module.__name__} never enforces key presence"
        # ...and through the shared resolver, not a second reading of which columns hold the key.
        assert "key_presence_values" in body, f"{module.__name__} resolves its own key columns"
        retrieve = inspect.getsource(cls.retrieve)
        assert "_enforce_key_presence" in retrieve, f"{cls.__name__}.retrieve skips it"
        here = retrieve.index("_enforce_key_presence")
        assert retrieve.index("_enforce_subject_anchor") < here, (
            f"{cls.__name__} adds key members BEFORE the anchor, which adds the same member "
            "on a different declaration"
        )
        assert here < retrieve.index("widen_stem_literals"), (
            f"{cls.__name__} widens stems BEFORE adding the member that may need widening"
        )


# ── a set operation is not one statement, and four guards read it as one ──
#
# Every guard that reads "the filter region" or splices "the outer WHERE" silently answers
# about arm 1 alone; the arm it skips contributes exactly the rows it should exclude.


def test_a_set_operation_is_read_PER_ARM_by_the_conjunctive_reading():
    """The strongest check on the page returned a false ``True`` on a 2-arm UNION.

    ``_is_constrained_conjunctively`` asks whether EVERY row the query can return satisfies a
    comparison on one of the fields, so a union is that question per arm with every arm
    required. Read whole, ``_filter_region`` returns everything after the FIRST ``WHERE``, the
    depth-zero ``AND`` split cuts across the arm boundary, and the undecomposable fragment falls
    through to the LOOSE test — which says yes. Both additive guards are gated on this reading,
    so the false ``True`` disabled both of them in silence.
    """
    from src.retrievers.query_guards import _is_constrained_conjunctively

    # The measured shape: neither arm binds the actor conjunctively — arm 1's identity sits
    # inside an OR beside the locator, arm 2 has no identity predicate at all.
    live = (
        "SELECT a FROM t WHERE phase = 'PRD' AND "
        "(locator = 'QW34ER' OR (office IN ('O1','O2') AND sign LIKE '5678CD%'))\n"
        "UNION ALL\n"
        "SELECT a FROM v WHERE phase = 'PRD' AND locator = 'QW34ER' LIMIT 500"
    )
    assert _is_constrained_conjunctively(live, ["office"], "sql") is False
    assert _is_constrained_conjunctively(live, ["sign"], "sql") is False
    assert _is_constrained_conjunctively(live, ["locator"], "sql") is False
    # EVERY arm is required, and that is the direction the defect ran in: arm 1 alone is
    # conjunctively bound on the locator, so a reader that stopped at the first arm would say
    # `True` for the whole statement while arm 2 returns rows about nobody in particular.
    one_arm_only = (
        "SELECT a FROM t WHERE phase = 'PRD' AND locator = 'QW34ER'\n"
        "UNION ALL\nSELECT a FROM v WHERE phase = 'PRD'"
    )
    assert _is_constrained_conjunctively(one_arm_only, ["locator"], "sql") is False
    # ...and when both arms really do bind it, the answer is `True` — or the fix would be
    # "inject onto everything", which re-anchors queries the generator got right.
    both = one_arm_only.replace(
        "SELECT a FROM v WHERE phase = 'PRD'",
        "SELECT a FROM v WHERE phase = 'PRD' AND locator = 'QW34ER'",
    )
    assert _is_constrained_conjunctively(both, ["locator"], "sql") is True
    # A single statement is unaffected, which is what makes this a strict repair.
    single = "SELECT a FROM t WHERE phase = 'PRD' AND locator = 'QW34ER'"
    assert _is_constrained_conjunctively(single, ["locator"], "sql") is True
    assert _is_constrained_conjunctively(single, ["office"], "sql") is False


def test_the_two_ADDITIVE_guards_splice_onto_EVERY_arm_of_a_set_operation():
    """``_and_clause_onto`` pins arm 1; the identity has to hold of every returned row.

    The splice was written once for ``enforce_default_filters`` and reasoned about only there —
    the two guards that ADD an identity predicate went on calling the single-arm version, so
    half the statement asked about the key and half asked about the population, and the halves
    were concatenated into one result.
    """
    from src.retrievers.query_guards import (enforce_key_presence,
                                             enforce_subject_anchor)

    two = (
        "SELECT a FROM t WHERE phase = 'PRD'\n"
        "UNION ALL\n"
        "SELECT a FROM v WHERE phase = 'PRD' LIMIT 500"
    )
    keyed = enforce_key_presence(
        two, {"org_unit": {"office": ["O1"]}, "user": {"sign": ["5678CD"]}}, "audit", "sql"
    )
    assert keyed.count("office = 'O1'") == 2, keyed
    assert keyed.count("sign = '5678CD'") == 2, keyed
    # The operator is re-emitted and the statement-level tail stays at the end.
    assert "UNION ALL" in keyed and keyed.rstrip().endswith("LIMIT 500")
    anchored = enforce_subject_anchor(two, {"sign": ["5678CD"]}, {"office": ["O1"]}, "audit")
    assert anchored.count("sign = '5678CD'") == 2, anchored
    assert anchored.count("office = 'O1'") == 2, anchored
    # A PARENTHESISED arm has no depth-zero WHERE to extend, so the whole rewrite declines
    # rather than half-doing it — and each guard says what the refusal COSTS, because a decline
    # that logs nothing is indistinguishable from a guard with nothing to do.
    wrapped = "(SELECT a FROM t WHERE phase = 'PRD') UNION ALL (SELECT a FROM v WHERE x = 1)"
    assert enforce_key_presence(wrapped, {"user": {"sign": ["5678CD"]}}, "audit") == wrapped
    assert enforce_subject_anchor(wrapped, {"sign": ["5678CD"]}, {}, "audit") == wrapped
    # A single statement takes the ordinary splice, once.
    single = "SELECT a FROM t WHERE phase = 'PRD'"
    once = enforce_key_presence(single, {"user": {"sign": ["5678CD"]}}, "audit")
    assert once.count("sign = '5678CD'") == 1, once


def test_the_tuple_rewrite_narrows_EVERY_arm_AND_declines_a_subquery():
    """A second ``WHERE`` has two causes and they are OPPOSITE.

    ``_filter_bodies`` counted ``WHERE``s over the whole statement, so a subquery (where which
    body the components belong to is guesswork) and an ARM of a union (where the answer is *all
    of them*) both returned no body at all. Measured live: 27 harvested combinations against a
    key space of ~10,000 pairings, declined, with the only trace a warning saying the query
    offered no single body to rewrite.
    """
    from src.retrievers.query_guards import _filter_bodies, enforce_value_tuples

    tuples = [
        [("sign", ["AAA1"]), ("unitId", ["OFF1"])],
        [("sign", ["BBB2"]), ("unitId", ["OFF2"])],
    ]
    two = (
        "SELECT a FROM t WHERE sign IN ('AAA1','BBB2') AND unitId IN ('OFF1','OFF2')\n"
        "UNION ALL\n"
        "SELECT a FROM v WHERE sign IN ('AAA1','BBB2') AND unitId IN ('OFF1','OFF2') LIMIT 500"
    )
    assert len(_filter_bodies(two, "sql")) == 2
    out = enforce_value_tuples(two, tuples, "audit")
    # Both arms carry the OR-of-ANDs, and neither carries the cross product any more.
    assert out.count("sign = 'AAA1' AND unitId = 'OFF1'") == 2, out
    assert out.count("sign = 'BBB2' AND unitId = 'OFF2'") == 2, out
    assert "IN ('AAA1','BBB2')" not in out and "IN ('OFF1','OFF2')" not in out, out
    assert "UNION ALL" in out and out.rstrip().endswith("LIMIT 500")
    # An arm holding a SUBQUERY declines the WHOLE statement: a partial narrowing reads as a
    # complete one while the untouched arm contributes the very pairings this guard drops.
    with_sub = (
        "SELECT a FROM t WHERE sign IN ('AAA1','BBB2') AND unitId IN ('OFF1','OFF2') "
        "AND id IN (SELECT id FROM u WHERE x = 1)\n"
        "UNION ALL\n"
        "SELECT a FROM v WHERE sign IN ('AAA1','BBB2') AND unitId IN ('OFF1','OFF2')"
    )
    assert _filter_bodies(with_sub, "sql") == []
    assert enforce_value_tuples(with_sub, tuples, "audit") == with_sub


def test_the_tuple_rewrite_LOGS_the_two_refusals_that_used_to_be_SILENT(caplog):
    """A guard that declines is indistinguishable from a guard with nothing to do.

    The module's own rule, and this function broke it twice: a body offering no readable
    conjunction and a body whose components sit in fewer than two slots both returned ``None``
    with no line of their own — so a run could narrow nothing in silence, which is how the live
    statement's declined rewrite left only a warning about a *different* cause. Each refusal now
    names its own, and the caller adds one summary line whenever no body was rewritten at all.
    """
    import logging

    from src.retrievers.query_guards import enforce_value_tuples

    tuples = [
        [("sign", ["AAA1"]), ("unitId", ["OFF1"])],
        [("sign", ["BBB2"]), ("unitId", ["OFF2"])],
    ]
    # (1) One conjunct: no conjunction of value lists exists to narrow.
    one = "SELECT a FROM t WHERE sign IN ('AAA1','BBB2')"
    with caplog.at_level(logging.WARNING, logger="src.retrievers.query_guards"):
        assert enforce_value_tuples(one, tuples, "audit") == one
    assert "a single conjunct" in caplog.text, caplog.text
    # ...and the summary, which is what says the whole call narrowed nothing.
    assert "narrowed none of them" in caplog.text, caplog.text

    # (2) The components ARE in the query, but in one slot — a combination is a fact ACROSS
    # slots, so the refusal has to name the columns it found and how many slots they were in,
    # or the reader cannot tell this from case (1) or from a component missing entirely.
    caplog.clear()
    grouped = (
        "SELECT a FROM t WHERE phase = 'PRD' "
        "AND (sign IN ('AAA1','BBB2') OR unitId IN ('OFF1','OFF2'))"
    )
    with caplog.at_level(logging.WARNING, logger="src.retrievers.query_guards"):
        assert enforce_value_tuples(grouped, tuples, "audit") == grouped
    assert "1 slot(s)" in caplog.text, caplog.text
    assert "sign, unitid" in caplog.text, caplog.text
    assert "narrowed none of them" in caplog.text, caplog.text

    # (3) And a body it DID narrow logs no refusal and no summary — or every successful run
    # would carry the sentence that is supposed to mean something went unenforced.
    caplog.clear()
    both = "SELECT a FROM t WHERE sign IN ('AAA1','BBB2') AND unitId IN ('OFF1','OFF2')"
    with caplog.at_level(logging.WARNING, logger="src.retrievers.query_guards"):
        assert enforce_value_tuples(both, tuples, "audit") != both
    assert "narrowed none of them" not in caplog.text, caplog.text
    assert "rewrote" in caplog.text, caplog.text


def test_heterogeneous_combination_SHAPES_choose_a_family_and_keep_the_rest():
    """One run's combinations need not be of one SHAPE, and requiring every slot skips them all.

    Measured on the live incident: a report TABULATES its subjects (unit + sign + numeric id per
    line, 25 of them) and its own prose names the asset and the organisation in one sentence (2
    more). Requiring every slot matched no combination at all and declined the whole rewrite, so
    the guarantee was lost to the two prose combinations. A shape is chosen instead — the largest
    family — and every slot outside it keeps the predicate the generator wrote, which is the same
    narrowing this guard already makes about every conjunct it does not touch.
    """
    from src.retrievers.query_guards import enforce_value_tuples

    tuples = [
        [("sign", ["AAA1"]), ("unitId", ["OFF1"])],
        [("sign", ["BBB2"]), ("unitId", ["OFF2"])],
        # A different shape entirely: the asset and the organisation, named together in prose.
        [("locator", ["QW34ER"]), ("org", ["SV"])],
    ]
    sql = (
        "SELECT a FROM t WHERE locator = 'QW34ER' AND org = 'SV' "
        "AND sign IN ('AAA1','BBB2') AND unitId IN ('OFF1','OFF2')"
    )
    out = enforce_value_tuples(sql, tuples, "audit")
    # The two-member family narrows...
    assert "(sign = 'AAA1' AND unitId = 'OFF1')" in out, out
    assert "(sign = 'BBB2' AND unitId = 'OFF2')" in out, out
    # ...and the slots outside its shape keep their own conjuncts, exactly as generated.
    assert "locator = 'QW34ER'" in out and "org = 'SV'" in out, out


def test_heterogeneous_combination_shapes_choose_a_family_on_the_DSL_ROUTE_TOO(caplog):
    """The family choice was written on the textual side only, a silent no-op on the DSL route.

    A group mixing slots from two different shapes matched no combination and declined the
    whole rewrite. The decline reads as a finding about the pass ("none is fully within the
    values it asked for") rather than as a guard that cannot read its own input.
    """
    from src.retrievers.query_guards import enforce_value_tuples_dsl

    tuples = [
        [("unitId", ["UNIT01"]), ("actorSign", ["SIGN01"]), ("actorUserId", ["LOGIN01"])],
        [("unitId", ["UNIT02"]), ("actorSign", ["SIGN02"]), ("actorUserId", ["LOGIN02"])],
        # The other shape: the record and the organisation, named together in prose. Only its
        # record slot survives into the query, which is exactly why it covers ONE slot here.
        [("recordRef", ["REF01"]), ("orgId", ["ORG01"])],
    ]
    dsl = {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"recordRef": "REF01"}},
                    {"terms": {"unitId": ["UNIT01", "UNIT02"]}},
                    {
                        "bool": {
                            "should": [
                                {"terms": {"actorSign": ["SIGN01", "SIGN02"]}},
                                {"terms": {"actorUserId": ["LOGIN01", "LOGIN02"]}},
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                ]
            }
        }
    }
    with caplog.at_level(logging.WARNING, logger="src.retrievers.query_guards"):
        out = enforce_value_tuples_dsl(dsl, tuples, "alerts")
    assert "NONE of the" not in caplog.text, caplog.text
    filt = out["query"]["bool"]["filter"]
    # The slot outside the chosen shape keeps the predicate the generator wrote...
    assert filt[0] == {"term": {"recordRef": "REF01"}}
    # ...and the two it replaces become an OR of AND-groups, one per observed record, with the
    # identity's two forms still OR-ed INSIDE the group: they are one value spelled twice.
    assert len(filt) == 2
    assert filt[1] == {
        "bool": {
            "should": [
                {
                    "bool": {
                        "filter": [
                            {"term": {"unitId": "UNIT01"}},
                            {
                                "bool": {
                                    "should": [
                                        {"term": {"actorSign": "SIGN01"}},
                                        {"term": {"actorUserId": "LOGIN01"}},
                                    ],
                                    "minimum_should_match": 1,
                                }
                            },
                        ]
                    }
                },
                {
                    "bool": {
                        "filter": [
                            {"term": {"unitId": "UNIT02"}},
                            {
                                "bool": {
                                    "should": [
                                        {"term": {"actorSign": "SIGN02"}},
                                        {"term": {"actorUserId": "LOGIN02"}},
                                    ],
                                    "minimum_should_match": 1,
                                }
                            },
                        ]
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }
    # And the choice is by SIZE, not by which slot the generator wrote first: with the prose
    # shape the larger family and both of its columns constrained, it is the one that narrows
    # and the tabulated slots keep their generated lists.
    bigger = [
        [("unitId", ["UNIT01"]), ("actorSign", ["SIGN01"])],
        [("recordRef", ["REF01"]), ("orgId", ["ORG01"])],
        [("recordRef", ["REF02"]), ("orgId", ["ORG02"])],
    ]
    dsl2 = {
        "query": {
            "bool": {
                "filter": [
                    {"terms": {"unitId": ["UNIT01", "UNIT02"]}},
                    {"terms": {"actorSign": ["SIGN01", "SIGN02"]}},
                    {"terms": {"recordRef": ["REF01", "REF02"]}},
                    {"terms": {"orgId": ["ORG01", "ORG02"]}},
                ]
            }
        }
    }
    filt2 = enforce_value_tuples_dsl(dsl2, bigger, "alerts")["query"]["bool"]["filter"]
    assert filt2[0] == {"terms": {"unitId": ["UNIT01", "UNIT02"]}}
    assert filt2[1] == {"terms": {"actorSign": ["SIGN01", "SIGN02"]}}
    assert filt2[2] == {
        "bool": {
            "should": [
                {
                    "bool": {
                        "filter": [
                            {"term": {"recordRef": "REF01"}},
                            {"term": {"orgId": "ORG01"}},
                        ]
                    }
                },
                {
                    "bool": {
                        "filter": [
                            {"term": {"recordRef": "REF02"}},
                            {"term": {"orgId": "ORG02"}},
                        ]
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }
    assert len(filt2) == 3


def test_key_presence_THEN_the_tuple_rewrite_is_what_narrows_the_cross_product():
    """The ordering is the fix, and an ordering test that only reads source text cannot say so.

    The additive guards inject one per-TYPE value LIST per key member, AND-ed — which IS the
    cross product this guard narrows. Run the tuple rewrite FIRST and the members are constrained
    nowhere, so it declines, and the injection then publishes the cross product with nothing left
    to narrow it: measured on one live statement, 800 pairings for the 25 identities the incident
    named. Run it second and the same inputs produce the pinned shape.
    """
    from src.retrievers.query_guards import (enforce_key_presence,
                                             enforce_value_tuples)

    key = {
        "org_unit": {"retriever_office": ["SSS1T19LM", "LLL1M13YZ"]},
        "user": {
            "retriever_sign": ["5678CD", "1234AB"],
            "retriever_user_id": ["11005678", "11001234"],
        },
    }
    tuples = [
        [
            ("retriever_office", ["SSS1T19LM"]),
            ("retriever_sign", ["5678CD"]),
            ("retriever_user_id", ["11005678"]),
        ],
        [
            ("retriever_office", ["LLL1M13YZ"]),
            ("retriever_sign", ["1234AB"]),
            ("retriever_user_id", ["11001234"]),
        ],
    ]
    generated = "SELECT a FROM t WHERE phase = 'PRD' AND locator = 'QW34ER'"

    # THE WIRED ORDER.
    keyed = enforce_key_presence(generated, key, "audit", "sql")
    assert "retriever_office IN ('SSS1T19LM', 'LLL1M13YZ')" in keyed
    narrowed = enforce_value_tuples(keyed, tuples, "audit")
    # An identity is a TUPLE: office AND (sign OR numeric id), OR-ed across the pairs that were
    # really observed — never two flat IN lists AND-ed, which is 4 pairings for 2 identities.
    assert (
        "retriever_office = 'SSS1T19LM' AND "
        "(retriever_sign = '5678CD' OR retriever_user_id = '11005678')"
    ) in narrowed, narrowed
    assert (
        "retriever_office = 'LLL1M13YZ' AND "
        "(retriever_sign = '1234AB' OR retriever_user_id = '11001234')"
    ) in narrowed, narrowed
    assert "IN (" not in narrowed, narrowed
    # The generator's own bounds keep their positions.
    assert "phase = 'PRD'" in narrowed and "locator = 'QW34ER'" in narrowed

    # THE REVERSED ORDER, which is what shipped: the tuple guard sees a query constraining the
    # members on no column, declines, and the injected cross product is published as-is.
    assert enforce_value_tuples(generated, tuples, "audit") == generated
    assert enforce_key_presence(generated, key, "audit", "sql") == keyed
    assert "retriever_sign IN ('5678CD', '1234AB')" in keyed, keyed


def test_match_pattern_widening_offers_the_declared_window_beside_the_equality():
    """A fixed-width part of an identifier, compared as a pattern instead of as the whole.

    The measured shape: an incident named two organisational units by their leading
    3-character segment, every route rendered them verbatim against columns storing the
    9-character identifier, and each query returned 0 rows with every stage reporting success.
    The pack states where a form sits inside the stored value; this offers that pattern BESIDE
    the equality, so the rewrite can only widen — and a widened predicate that comes back empty
    proves the narrow one does, which is the superset rule `key_was_enforced` is read under.
    """
    from src.retrievers.query_guards import widen_match_patterns

    patterns = {"payload.unitId": {"DEL": "DEL??????", "DAC": "DAC??????"}}
    # The LIST shape, which is what `render_filters` emits for a multi-value column.
    listed = (
        "SELECT * FROM access WHERE unitId IN ('DEL', 'DAC') "
        "AND accessDate = DATE'2026-08-17' LIMIT 500"
    )
    out = widen_match_patterns(listed, patterns, "audit")
    assert "unitId IN ('DEL', 'DAC') OR unitId LIKE 'DEL______' " in out + " ", out
    assert "unitId LIKE 'DAC______'" in out, out
    # It narrows nothing and moves nothing: the bound, the projection and the cap survive, and
    # the original equality is still there to be satisfied by a column that stores the segment.
    assert out.startswith("SELECT * FROM access WHERE")
    assert "accessDate = DATE'2026-08-17'" in out and out.endswith("LIMIT 500")
    # The EQUALITY shape, and the wildcard is one character per `?` — not a prefix match, which
    # would also select a 12-character value on some other column.
    single = "SELECT * FROM access WHERE unitId = 'DEL'"
    out = widen_match_patterns(single, patterns, "audit")
    assert out == (
        "SELECT * FROM access WHERE (unitId = 'DEL' OR unitId LIKE 'DEL______')"
    ), out
    # ES|QL keeps the canonical spelling and its own string literal.
    esql = widen_match_patterns(
        'FROM access-* | WHERE unitId == "DEL" | LIMIT 500', patterns, "audit", dialect="esql"
    )
    assert 'unitId LIKE "DEL??????"' in esql, esql
    # Nothing declared, nothing carried, nothing to do — every pack that has not opted in.
    assert widen_match_patterns(single, {}, "audit") == single
    assert widen_match_patterns("", patterns, "audit") == ""
    # A value this query does not carry is not injected.
    assert widen_match_patterns(
        "SELECT * FROM access WHERE unitId = 'NCE'", patterns, "audit"
    ) == "SELECT * FROM access WHERE unitId = 'NCE'"


def test_match_pattern_widening_REFUSES_a_negation_and_a_wildcard_bearing_value():
    """Two refusals, and both are about the one direction this guard must never take.

    A widened `NOT IN` / `<>` NARROWS: on an exclusion check that deletes the subject's own
    rows and reports the emptiness as a clean. And a value carrying a character the backend
    reads as a wildcard is indistinguishable from the pattern once substituted, so translating
    it would turn one character of the subject's own identifier into "any character".
    """
    from src.retrievers.query_guards import widen_match_patterns

    patterns = {"unitId": {"DEL": "DEL??????"}}
    for negated in (
        "SELECT * FROM t WHERE unitId NOT IN ('DEL')",
        "SELECT * FROM t WHERE NOT unitId IN ('DEL')",
        "SELECT * FROM t WHERE NOT unitId = 'DEL'",
        "SELECT * FROM t WHERE unitId <> 'DEL'",
        "SELECT * FROM t WHERE unitId != 'DEL'",
    ):
        assert widen_match_patterns(negated, patterns, "audit") == negated, negated
    # A value holding this dialect's own wildcards: declined, and the equality publishes as
    # generated rather than as a pattern nobody declared.
    sql = "SELECT * FROM t WHERE unitId = 'DE_'"
    assert widen_match_patterns(sql, {"unitId": {"DE_": "DE_??????"}}, "audit") == sql
    # The same value is fine on a route whose wildcards are spelled differently, and the
    # forbidden set follows the DIALECT rather than being one global list.
    esql = 'FROM t | WHERE unitId == "DE_"'
    assert widen_match_patterns(
        esql, {"unitId": {"DE_": "DE_??????"}}, "audit", dialect="esql"
    ) != esql
    # ...and there it is `?`/`*` that are refused.
    starred = 'FROM t | WHERE unitId == "DE*"'
    assert widen_match_patterns(
        starred, {"unitId": {"DE*": "DE*??????"}}, "audit", dialect="esql"
    ) == starred


def test_match_pattern_widening_dsl_wraps_the_term_in_a_should_of_its_own():
    """The DSL twin, because the measured failure was on THIS route.

    `{"term": {<column>: "<segment>"}}`, published as generated, 0 rows. A guarantee honoured
    on one query family and not the other is not a guarantee.
    """
    from src.retrievers.query_guards import widen_match_patterns_dsl

    # Both columns the mapper resolved for the type, because which of them stores which
    # precision is declared nowhere per column — the same reason the stem widening offers its
    # shorter form BESIDE the value rather than instead of it.
    patterns = {
        "payload.actor.officeDetails.unitId": {"DEL": "DEL??????"},
        "payload.AdminOffice": {"DEL": "DEL??????"},
    }
    dsl = {
        "bool": {
            "filter": [{"range": {"date": {"gte": "2026-08-15"}}}],
            "should": [
                {"term": {"payload.actor.officeDetails.unitId": "DEL"}},
                {"terms": {"payload.AdminOffice": ["DEL", "NCE"]}},
            ],
            "minimum_should_match": 1,
        }
    }
    out = widen_match_patterns_dsl(dsl, patterns, "svc_sessions")
    widened = out["bool"]["should"][0]["bool"]
    # The original clause is kept beside the wildcard, with an explicit minimum_should_match:
    # the default is 1 only while the bool carries no `must`, and this clause may be spliced
    # under one later.
    assert widened["minimum_should_match"] == 1
    assert widened["should"][0] == {"term": {"payload.actor.officeDetails.unitId": "DEL"}}
    assert widened["should"][1] == {
        "wildcard": {"payload.actor.officeDetails.unitId": {"value": "DEL??????"}}
    }
    # A `terms` list is widened too, and only for the values that HAVE a declared pattern: the
    # complete sibling value rides in the original clause and gets no wildcard of its own.
    listed = out["bool"]["should"][1]["bool"]
    assert listed["should"][0] == {"terms": {"payload.AdminOffice": ["DEL", "NCE"]}}
    assert listed["should"][1:] == [
        {"wildcard": {"payload.AdminOffice": {"value": "DEL??????"}}}
    ], listed
    # A column the pack declared nothing for is left exactly as generated — the bindings are
    # matched per column and never widened onto a sibling nobody measured.
    only_one = widen_match_patterns_dsl(
        dsl, {"payload.actor.officeDetails.unitId": {"DEL": "DEL??????"}}, "svc_sessions"
    )
    assert only_one["bool"]["should"][1] == {"terms": {"payload.AdminOffice": ["DEL", "NCE"]}}
    # The window is untouched and the input is not mutated.
    assert out["bool"]["filter"] == [{"range": {"date": {"gte": "2026-08-15"}}}]
    assert "wildcard" not in json.dumps(dsl)
    # Under `must_not` nothing is widened at any depth, for the reason the textual guard
    # refuses a negation.
    negated = {"bool": {"must_not": [{"term": {"payload.AdminOffice": "DEL"}}]}}
    assert widen_match_patterns_dsl(negated, patterns, "s") == negated
    # Nothing declared, or not a query at all.
    assert widen_match_patterns_dsl(dsl, {}, "s") == dsl
    assert widen_match_patterns_dsl(None, patterns, "s") is None


def test_match_pattern_widening_runs_on_every_backend_after_the_stem_widening():
    """Wired into all four, and AFTER the stem widening — the two compose in one order only.

    A stem offers a SHORTER form of the value beside it, itself a candidate for a positional
    pattern; a pattern rewritten first leaves a `LIKE` the stem guard does not read. And both
    resolve their columns through the shared `field_mapping.match_patterns`, so the filter hint
    and the widening cannot name different columns.
    """
    import inspect

    from src.retrievers import (databricks_retriever, elasticsearch_retriever,
                                kibana_retriever, snowflake_retriever)
    from src.retrievers.elasticsearch_retriever import ElasticsearchRetriever
    from src.retrievers.kibana_retriever import KibanaRetriever
    from src.retrievers.snowflake_retriever import SnowflakeRetriever

    for module, cls, guard in (
        (databricks_retriever, DatabricksRetriever, "widen_match_patterns"),
        (elasticsearch_retriever, ElasticsearchRetriever, "widen_match_patterns"),
        (snowflake_retriever, SnowflakeRetriever, "widen_match_patterns"),
        (kibana_retriever, KibanaRetriever, "widen_match_patterns_dsl"),
    ):
        body = inspect.getsource(module)
        assert guard in body, f"{module.__name__} never widens to a declared match pattern"
        assert "match_patterns(" in body, f"{module.__name__} resolves its own pattern columns"
        retrieve = inspect.getsource(cls.retrieve)
        assert "widen_match_patterns" in retrieve, f"{cls.__name__}.retrieve skips it"
        assert retrieve.index("widen_stem_literals") < retrieve.index(
            "widen_match_patterns"
        ), f"{cls.__name__} widens a pattern before the stem it may contain"


def test_default_filters_are_PINNED_on_the_textual_SQL_routes_too():
    """The pack's pinned slice, AND-ed onto whatever the generator produced.

    `default_filters` is documented as enforced deterministically by the retriever, and that
    claim was true on the two Elasticsearch routes only: the Databricks and Snowflake ones read
    the declaration into an attribute they never used. A pack pinning a shared table to one
    slice therefore got the slice on an ES source and the whole table on a SQL one — a FULL
    result of other slices' rows, which passes every "did anything come back" check.
    """
    from src.retrievers.query_guards import enforce_default_filters

    sql = "SELECT * FROM audit WHERE recordLocator = 'AB12CD' LIMIT 500"
    out = enforce_default_filters(sql, {"application_phase": "PRD"}, "audit_raw_access")
    assert out == (
        "SELECT * FROM audit WHERE application_phase = 'PRD' AND "
        "(recordLocator = 'AB12CD') LIMIT 500"
    ), out
    # A top-level OR in the generated body is parenthesised, never re-associated — the pinned
    # conjunct must hold of every returned row.
    ored = "SELECT * FROM audit WHERE a = 1 OR b = 2"
    assert enforce_default_filters(ored, {"phase": "PRD"}, "s") == (
        "SELECT * FROM audit WHERE phase = 'PRD' AND (a = 1 OR b = 2)"
    )
    # A query with no WHERE at all gets one.
    assert enforce_default_filters("SELECT * FROM audit LIMIT 5", {"phase": "PRD"}, "s") == (
        "SELECT * FROM audit WHERE phase = 'PRD' LIMIT 5"
    )
    # Several entries are AND-ed — one fact per key, matching the map's semantics everywhere
    # else — and a genuinely numeric value is not quoted.
    out = enforce_default_filters(
        "SELECT * FROM t WHERE x = 1", {"phase": "PRD", "version": 3, "live": True}, "s"
    )
    assert "phase = 'PRD' AND version = 3 AND live = TRUE AND (x = 1)" in out, out
    # A quote inside a declared value is escaped rather than ending the literal.
    assert "'O''Hare'" in enforce_default_filters(
        "SELECT * FROM t WHERE x = 1", {"city": "O'Hare"}, "s"
    )
    # Nothing declared — every pack that pins nothing — and the ES|QL route gets a stage.
    assert enforce_default_filters(sql, {}, "s") == sql
    assert enforce_default_filters("", {"phase": "PRD"}, "s") == ""
    assert enforce_default_filters(
        "FROM idx-* | WHERE a == 1 | LIMIT 5", {"phase": "PRD"}, "s", dialect="esql"
    ).startswith("FROM idx-* | WHERE phase = 'PRD' |")


def test_default_filters_reach_EVERY_arm_of_a_set_operation():
    """A UNION is not a wall for a conjunct that must hold of every returned row.

    The splice alone pins the FIRST arm and leaves the rest as generated — valid SQL and half a
    guarantee, which on the shape this key exists for is the whole defect back: the source that
    needed it generates one SELECT per view of a shared relation and unions them, so the
    environment slice would hold on one view while the others contributed their test rows to
    the same result.
    """
    from src.retrievers.query_guards import enforce_default_filters

    two = (
        "SELECT a FROM recordretrieve_view WHERE recordLocator = 'AB12CD'\n"
        "UNION ALL\n"
        "SELECT a FROM recordsearch_view WHERE recordLocator = 'AB12CD' ORDER BY a LIMIT 500"
    )
    out = enforce_default_filters(two, {"application_phase": "PRD"}, "audit_raw_access")
    assert out.count("application_phase = 'PRD'") == 2, out
    # The operator is re-emitted verbatim, and the statement-level tail stays at the end.
    assert "\nUNION ALL\n" in out and out.rstrip().endswith("ORDER BY a LIMIT 500")
    # Three arms and a mix of operators, each arm pinned once.
    three = (
        "SELECT a FROM t1 WHERE x = 1 UNION ALL SELECT a FROM t2 WHERE y = 2 "
        "UNION SELECT a FROM t3 WHERE z = 3 LIMIT 10"
    )
    out = enforce_default_filters(three, {"phase": "PRD"}, "s")
    assert out.count("phase = 'PRD'") == 3, out
    assert "UNION ALL" in out and "\nUNION\n" in out
    # A PARENTHESISED arm declines the whole rewrite rather than half-doing it: there is no
    # depth-zero WHERE inside it to extend, and appending one after the bracket is not valid
    # SQL. A refusal costs the slice; a broken statement costs the source.
    parenthesised = "(SELECT a FROM t1 WHERE x = 1) UNION ALL (SELECT a FROM t2 WHERE y = 2)"
    assert enforce_default_filters(parenthesised, {"phase": "PRD"}, "s") == parenthesised
    # A UNION inside a SUBQUERY is not an arm boundary — depth decides, so the whole statement
    # keeps one pinned conjunct and the subquery is untouched.
    nested = (
        "SELECT a FROM t WHERE id IN "
        "(SELECT id FROM u WHERE p = 1 UNION ALL SELECT id FROM v WHERE q = 2) AND x = 1"
    )
    out = enforce_default_filters(nested, {"phase": "PRD"}, "s")
    assert out.count("phase = 'PRD'") == 1, out
    assert "(SELECT id FROM u WHERE p = 1 UNION ALL SELECT id FROM v WHERE q = 2)" in out
    # ...and neither is a UNION inside a STRING LITERAL.
    quoted = "SELECT a FROM t WHERE note = 'UNION ALL trick' AND x = 1"
    out = enforce_default_filters(quoted, {"phase": "PRD"}, "s")
    assert out.count("phase = 'PRD'") == 1, out
    assert "note = 'UNION ALL trick'" in out, out


def test_the_tail_reader_skips_a_TAIL_KEYWORD_INSIDE_A_STRING_LITERAL():
    """The one scanner on that page that could damage a query instead of declining on it.

    `_tail_index` tracked parentheses and not quotes, so a tail keyword inside a string literal
    read as the end of the filter region and the splice cut the literal in half — a different
    statement, and usually not a statement at all. Every additive guard shares this reader, and
    free-form producer text is exactly what a search or command column holds, so the literal is
    the generator's rather than an invented case.
    """
    from src.retrievers.query_guards import _tail_index, enforce_key_presence

    text = "SELECT * FROM t WHERE cmd = 'ORDER BY MAIL' AND x = 1 LIMIT 20"
    # The region ends at the REAL trailing clause, not at the one inside the literal.
    assert text[_tail_index(text) :] == "LIMIT 20"
    # An escaped quote inside a literal does not end it either.
    escaped = "SELECT * FROM t WHERE note = 'it''s LIMIT 5' AND x = 1 LIMIT 20"
    assert escaped[_tail_index(escaped) :] == "LIMIT 20"
    # A statement with no trailing clause reads to the end, unchanged behaviour.
    assert _tail_index("SELECT * FROM t WHERE x = 1") == len("SELECT * FROM t WHERE x = 1")
    # And the additive guard that reads it now splices around the literal instead of through it.
    out = enforce_key_presence(text, {"user": {"sign": ["0101XY"]}}, "t")
    assert "cmd = 'ORDER BY MAIL'" in out, out
    assert out.rstrip().endswith("LIMIT 20"), out


def test_default_filters_run_LAST_on_every_route_after_the_never_filter_strip():
    """Order is the guarantee: the strip LIFTS the constant, this PINS it exactly once.

    Reversed, the strip would delete the conjunct this writes and the declaration would be gone
    — and the two keys stay two because neither means the other: `never_filter` alone drops the
    environment scope, this alone leaves the misplaced OR arm satisfying its group.
    """
    import inspect

    from src.retrievers import (databricks_retriever, elasticsearch_retriever,
                                kibana_retriever, snowflake_retriever)
    from src.retrievers.elasticsearch_retriever import ElasticsearchRetriever
    from src.retrievers.kibana_retriever import KibanaRetriever
    from src.retrievers.snowflake_retriever import SnowflakeRetriever

    for module, cls, marker in (
        (databricks_retriever, DatabricksRetriever, "enforce_default_filters"),
        (snowflake_retriever, SnowflakeRetriever, "enforce_default_filters"),
        (kibana_retriever, KibanaRetriever, "_apply_default_filters"),
        (elasticsearch_retriever, ElasticsearchRetriever, "_apply_default_filters"),
    ):
        body = inspect.getsource(module)
        assert "default_filters" in body, f"{module.__name__} never reads the declaration"
        retrieve = inspect.getsource(cls.retrieve)
        assert marker in retrieve, f"{cls.__name__}.retrieve never pins the declared slice"
        here = retrieve.index(marker)
        # AFTER the strip that removes the same constant from wherever the generator put it.
        strip = next(
            (
                name
                for name in ("_strip_evidence_filters", "never_filter", "_strip_evidence")
                if name in retrieve
            ),
            None,
        )
        assert strip, f"{cls.__name__}.retrieve never strips a never_filter predicate"
        assert retrieve.index(strip) < here, (
            f"{cls.__name__} pins the slice BEFORE stripping the misplaced copy, so the strip "
            "deletes the conjunct the pin just wrote"
        )
        # ...and it is the LAST rewrite before the query is published, so no later guard can
        # re-shape the conjunct out of the body.
        assert here < retrieve.index("publish_query"), f"{cls.__name__} pins after publishing"


def test_the_two_SQL_retrievers_READ_the_declaration_off_the_source_config():
    """Riding the declaration to the config is only half of it — the other half is reading it.

    `_attach_query_guards` has put `default_filters` on every retriever config for as long as
    the key existed; the SQL routes stored it and never applied it. The attribute is asserted
    here so a future refactor cannot quietly go back to that state.
    """
    from src.retrievers.snowflake_retriever import SnowflakeRetriever

    config = {
        "name": "audit_raw_access",
        "warehouse_id": "0" * 16,
        "workspace_url": "https://example.cloud.databricks.com",
        "host": "example.cloud.databricks.com",
        "token": "t",
        "catalog": "c",
        "schema": "s",
        "tables": ["v"],
        "default_filters": {"application_phase": "PRD"},
    }
    assert DatabricksRetriever(config, MagicMock()).default_filters == {
        "application_phase": "PRD"
    }
    snow = SnowflakeRetriever(
        {**config, "account": "a", "user": "u", "password": "p", "database": "d"}, MagicMock()
    )
    assert snow.default_filters == {"application_phase": "PRD"}
    # A source declaring none gets an empty map, not None: the guard is then a no-op.
    assert DatabricksRetriever({**config, "default_filters": None}, MagicMock()).default_filters == {}


@pytest.mark.asyncio
async def test_a_4xx_carries_the_backend_reason_and_keeps_its_exception_type():
    """A 403 must say WHICH 403 it was, without changing what catches it.

    Measured on the deployed App: eleven sources failed as `403, message='Forbidden'`,
    which is the same string for an expired token, a missing grant and a workspace
    refusing the request's network — three different owners, one indistinguishable log.
    The type is asserted alongside the message because `_gather` records the exception
    TYPE in `unanswered_out` and the retry policy classifies on it.
    """
    from src.retrievers.base import raise_for_status_with_reason

    resp = SimpleNamespace(
        status=403,
        reason="Forbidden",
        request_info=None,
        history=(),
        headers={},
        text=AsyncMock(return_value='{"error_code":403,"message":"Invalid access token."}'),
    )
    with pytest.raises(aiohttp.ClientResponseError) as excinfo:
        await raise_for_status_with_reason(resp)
    assert excinfo.value.status == 403
    assert "Invalid access token" in excinfo.value.message
    assert "Forbidden" in excinfo.value.message


@pytest.mark.asyncio
async def test_the_reason_never_carries_a_credential():
    """An endpoint that echoes the request must not turn its error body into a token leak."""
    from src.retrievers.base import raise_for_status_with_reason

    resp = SimpleNamespace(
        status=401,
        reason="Unauthorized",
        request_info=None,
        history=(),
        headers={},
        text=AsyncMock(return_value="rejected header Authorization: Bearer dapi-secret-value"),
    )
    with pytest.raises(aiohttp.ClientResponseError) as excinfo:
        await raise_for_status_with_reason(resp)
    assert "dapi-secret-value" not in excinfo.value.message
    assert "<redacted>" in excinfo.value.message


@pytest.mark.asyncio
async def test_a_2xx_is_not_disturbed():
    """The helper replaces `raise_for_status`, so the success path must stay a no-op."""
    from src.retrievers.base import raise_for_status_with_reason

    text = AsyncMock(return_value="{}")
    resp = SimpleNamespace(status=200, reason="OK", request_info=None, history=(),
                           headers={}, text=text)
    assert await raise_for_status_with_reason(resp) is None
    # The body is left for the caller to read.
    text.assert_not_awaited()


def test_no_retriever_discards_a_4xx_body():
    """Every HTTP backend routes through the one seam, or its 4xx goes back to being opaque."""
    import pathlib

    offenders = []
    for path in sorted(pathlib.Path("src/retrievers").glob("*_retriever.py")):
        body = path.read_text()
        if "resp.raise_for_status()" in body:
            offenders.append(path.name)
    assert offenders == []
