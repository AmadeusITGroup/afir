import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.log_retrieval import LogRetrievalEngine


@pytest.fixture
def log_retrieval_engine():
    log_sources = [
        {
            "name": "application_logs",
            "type": "elasticsearch",
            "url": "http://elasticsearch:9200",
            "username": "test_user",
            "password": "test_pass",
            "timeout": 30
        },
        {
            "name": "security_logs",
            "type": "splunk",
            "url": "https://splunk.example.com:8089",
            "token": "test_token",
            "timeout": 30
        }
    ]
    return LogRetrievalEngine(log_sources)


@pytest.mark.asyncio
async def test_retrieve_logs(log_retrieval_engine, monkeypatch):
    # Mock API calls
    api_calls = [
        {"target_log_source": "application_logs", "query_parameters": {}},
        {"target_log_source": "security_logs", "query_parameters": {}}
    ]

    # Mock Elasticsearch and Splunk responses
    monkeypatch.setattr('src.log_retrieval.LogRetrievalEngine.get_elasticsearch_logs',
                        AsyncMock(return_value=[{'log': 'data1'}]))
    monkeypatch.setattr('src.log_retrieval.LogRetrievalEngine.get_splunk_logs',
                        AsyncMock(return_value=[{'log': 'data2'}]))

    # Call the function
    result = await log_retrieval_engine.retrieve(api_calls)

    # Assertions
    assert 'application_logs' in result
    assert 'security_logs' in result
    assert result['application_logs'] == [{'log': 'data1'}]
    assert result['security_logs'] == [{'log': 'data2'}]


@pytest.mark.asyncio
async def test_get_elasticsearch_logs(log_retrieval_engine):
    with patch('src.log_retrieval.AsyncElasticsearch') as mock_es:
        mock_es.return_value.search.return_value = {
            'hits': {
                'hits': [
                    {'_source': {'message': 'Test log 1'}},
                    {'_source': {'message': 'Test log 2'}}
                ]
            }
        }

        api_call = {
            "query_parameters": {
                "start_time": "2023-07-10T00:00:00Z",
                "end_time": "2023-07-10T23:59:59Z",
                "search_terms": "test"
            }
        }

        result = await log_retrieval_engine.get_elasticsearch_logs(api_call)

        assert len(result) == 2
        assert result[0]['message'] == 'Test log 1'
        assert result[1]['message'] == 'Test log 2'


@pytest.mark.asyncio
async def test_get_splunk_logs(log_retrieval_engine):
    with patch('aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value.json.return_value = {
            'sid': 'test_sid'}
        mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value.json.side_effect = [
            {'isDone': False},
            {'isDone': True},
            {'results': [{'event': 'Test log 1'}, {'event': 'Test log 2'}]}
        ]

        api_call = {
            "query_parameters": {
                "start_time": "2023-07-10T00:00:00Z",
                "end_time": "2023-07-10T23:59:59Z",
                "search_terms": "test"
            }
        }

        result = await log_retrieval_engine.get_splunk_logs(api_call)

        assert len(result) == 2
        assert result[0]['event'] == 'Test log 1'
        assert result[1]['event'] == 'Test log 2'


def test_build_elasticsearch_query(log_retrieval_engine):
    api_call = {
        "query_parameters": {
            "start_time": "2023-07-10T00:00:00Z",
            "end_time": "2023-07-10T23:59:59Z",
            "search_terms": "test"
        }
    }

    query = log_retrieval_engine.build_elasticsearch_query(api_call)

    assert 'query' in query
    assert 'bool' in query['query']
    assert 'must' in query['query']['bool']
    assert len(query['query']['bool']['must']) == 2


def test_build_splunk_query(log_retrieval_engine):
    api_call = {
        "query_parameters": {
            "start_time": "2023-07-10T00:00:00Z",
            "end_time": "2023-07-10T23:59:59Z",
            "search_terms": "test"
        }
    }

    query = log_retrieval_engine.build_splunk_query(api_call)

    assert 'search index=*' in query
    assert 'earliest=2023-07-10T00:00:00Z' in query
    assert 'latest=2023-07-10T23:59:59Z' in query
    assert 'test' in query
