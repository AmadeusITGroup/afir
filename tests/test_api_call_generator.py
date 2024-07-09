import pytest
from unittest.mock import AsyncMock, MagicMock
from src.api_call_generator import ApiCallGenerator


@pytest.fixture
def api_call_generator():
    llm_config = {'provider': 'test_provider'}
    rag = MagicMock()
    return ApiCallGenerator(llm_config, rag)


@pytest.mark.asyncio
async def test_generate_api_calls(api_call_generator, monkeypatch):
    # Mock understanding
    understanding = {
        'incident_id': 'INC-001',
        'analysis': 'Test analysis'
    }

    # Mock LLM response
    mock_llm_response = '[{"target_log_source": "application_logs", "query_parameters": {"start_time": "2023-07-10T00:00:00Z", "end_time": "2023-07-10T23:59:59Z", "search_terms": "login AND failure"}, "context": "Investigate failed logins"}]'
    monkeypatch.setattr('src.api_call_generator.get_llm_response', AsyncMock(return_value=mock_llm_response))

    # Call the function
    result = await api_call_generator.generate(understanding)

    # Assertions
    assert len(result) == 1
    assert result[0]['target_log_source'] == 'application_logs'
    assert 'query_parameters' in result[0]
    assert 'context' in result[0]


@pytest.mark.asyncio
async def test_generate_api_calls_error(api_call_generator, monkeypatch):
    # Mock understanding
    understanding = {
        'incident_id': 'INC-002',
        'analysis': 'Another test analysis'
    }

    # Mock LLM response with an error
    monkeypatch.setattr('src.api_call_generator.get_llm_response', AsyncMock(side_effect=Exception("Test error")))

    # Call the function and expect an exception
    with pytest.raises(Exception):
        await api_call_generator.generate(understanding)


def test_parse_api_calls_valid_json(api_call_generator):
    valid_json = '[{"target_log_source": "security_logs", "query_parameters": {}, "context": "Test"}]'
    result = api_call_generator.parse_api_calls(valid_json)
    assert len(result) == 1
    assert result[0]['target_log_source'] == 'security_logs'


def test_parse_api_calls_invalid_json(api_call_generator):
    invalid_json = 'This is not JSON'
    result = api_call_generator.parse_api_calls(invalid_json)
    assert result == []


def test_validate_api_call_valid(api_call_generator):
    valid_call = {
        'target_log_source': 'application_logs',
        'query_parameters': {},
        'context': 'Test context'
    }
    result = api_call_generator.validate_api_call(valid_call)
    assert result == valid_call


def test_validate_api_call_invalid(api_call_generator):
    invalid_call = {
        'target_log_source': 'application_logs',
        # missing 'query_parameters' and 'context'
    }
    result = api_call_generator.validate_api_call(invalid_call)
    assert 'query_parameters' in result
    assert 'context' in result
    assert result['query_parameters'] == 'Not provided'
    assert result['context'] == 'Not provided'
