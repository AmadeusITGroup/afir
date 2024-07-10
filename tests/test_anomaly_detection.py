import pytest
from unittest.mock import AsyncMock, MagicMock
from src.anomaly_detection import AnomalyDetectionModule

@pytest.fixture
def anomaly_detection_module():
    config = {'threshold': 0.7}
    llm_config = {'provider': 'test_provider'}
    rag = MagicMock()
    return AnomalyDetectionModule(config, llm_config, rag)

@pytest.mark.asyncio
async def test_detect_anomalies(anomaly_detection_module, monkeypatch):
    # Mock logs and understanding
    logs = {'source1': ['log1', 'log2'], 'source2': ['log3']}
    understanding = {'incident_id': 'INC-001', 'analysis': 'Test analysis'}

    # Mock LLM response
    mock_llm_response = '[{"description": "Anomaly 1", "confidence_score": 0.8}, {"description": "Anomaly 2", "confidence_score": 0.6}]'
    monkeypatch.setattr('src.anomaly_detection.get_llm_response', AsyncMock(return_value=mock_llm_response))

    # Call the function
    result = await anomaly_detection_module.detect(logs, understanding)

    # Assertions
    assert len(result) == 1
    assert result[0]['description'] == 'Anomaly 1'
    assert result[0]['confidence_score'] == 0.8

@pytest.mark.asyncio
async def test_detect_anomalies_error(anomaly_detection_module, monkeypatch):
    # Mock logs and understanding
    logs = {'source1': ['log1']}
    understanding = {'incident_id': 'INC-002', 'analysis': 'Test analysis'}

    # Mock LLM response with an error
    monkeypatch.setattr('src.anomaly_detection.get_llm_response', AsyncMock(side_effect=Exception("Test error")))

    # Call the function and expect an exception
    with pytest.raises(Exception):
        await anomaly_detection_module.detect(logs, understanding)

def test_preprocess_logs(anomaly_detection_module):
    logs = {
        'source1': [{'timestamp': '2023-07-10T12:00:00Z', 'message': 'Log 1'}],
        'source2': [{'timestamp': '2023-07-10T12:01:00Z', 'message': 'Log 2'}]
    }
    result = anomaly_detection_module.preprocess_logs(logs)
    assert '[source1]' in result
    assert '[source2]' in result
    assert 'Log 1' in result
    assert 'Log 2' in result

def test_parse_llm_response_valid_json(anomaly_detection_module):
    valid_json = '[{"description": "Anomaly", "confidence_score": 0.9}]'
    result = anomaly_detection_module.parse_llm_response(valid_json)
    assert len(result) == 1
    assert result[0]['description'] == 'Anomaly'
    assert result[0]['confidence_score'] == 0.9

def test_parse_llm_response_invalid_json(anomaly_detection_module):
    invalid_json = 'This is not JSON'
    result = anomaly_detection_module.parse_llm_response(invalid_json)
    assert result == []

def test_filter_anomalies(anomaly_detection_module):
    anomalies = [
        {'description': 'Anomaly 1', 'confidence_score': 0.8},
        {'description': 'Anomaly 2', 'confidence_score': 0.6},
        {'description': 'Anomaly 3', 'confidence_score': 0.7}
    ]
    result = anomaly_detection_module.filter_anomalies(anomalies)
    assert len(result) == 2
    assert result[0]['description'] == 'Anomaly 1'
    assert result[1]['description'] == 'Anomaly 3'