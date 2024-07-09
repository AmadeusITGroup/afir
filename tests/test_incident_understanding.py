import pytest
from unittest.mock import AsyncMock, MagicMock
from src.incident_understanding import IncidentUnderstandingModule


@pytest.fixture
def incident_understanding_module():
    llm_config = {'provider': 'test_provider'}
    rag = MagicMock()
    return IncidentUnderstandingModule(llm_config, rag)


@pytest.mark.asyncio
async def test_process_incident(incident_understanding_module, monkeypatch):
    # Mock incident
    incident = {
        'id': 'INC-001',
        'timestamp': '2023-07-10T12:00:00Z',
        'description': 'Suspicious login attempt'
    }

    # Mock LLM response
    mock_llm_response = '{"summary": "Test summary", "severity": 7}'
    monkeypatch.setattr('src.incident_understanding.get_llm_response', AsyncMock(return_value=mock_llm_response))

    # Call the function
    result = await incident_understanding_module.process(incident)

    # Assertions
    assert result['incident_id'] == 'INC-001'
    assert 'analysis' in result
    assert result['analysis']['summary'] == 'Test summary'
    assert result['analysis']['severity'] == 7


@pytest.mark.asyncio
async def test_process_incident_error(incident_understanding_module, monkeypatch):
    # Mock incident
    incident = {
        'id': 'INC-002',
        'timestamp': '2023-07-10T13:00:00Z',
        'description': 'Another suspicious activity'
    }

    # Mock LLM response with an error
    monkeypatch.setattr('src.incident_understanding.get_llm_response', AsyncMock(side_effect=Exception("Test error")))

    # Call the function and expect an exception
    with pytest.raises(Exception):
        await incident_understanding_module.process(incident)


def test_structure_understanding_valid_json(incident_understanding_module):
    valid_json = '{"key": "value"}'
    result = incident_understanding_module.structure_understanding(valid_json)
    assert result == {"key": "value"}


def test_structure_understanding_invalid_json(incident_understanding_module):
    invalid_json = 'This is not JSON'
    result = incident_understanding_module.structure_understanding(invalid_json)
    assert 'raw_output' in result
    assert result['raw_output'] == 'This is not JSON'
