import pytest
from unittest.mock import AsyncMock, MagicMock
from src.main import process_incident, main


@pytest.mark.asyncio
async def test_process_incident():
    # Mock incident and modules
    incident = {'id': 'INC-001', 'description': 'Test incident', 'timestamp': "2024-03-04T21:34"}
    modules = {
        'understanding': AsyncMock(),
        'api_call': AsyncMock(),
        'log_retrieval': AsyncMock(),
        'anomaly_detection': AsyncMock(),
        'plugins': MagicMock(),
        'report_generation': AsyncMock(),
        # 'output': AsyncMock(),
        # 'feedback': AsyncMock(),
    }

    # Set return values for mocks
    modules['understanding'].process.return_value = {'understanding': 'Test analysis'}
    modules['api_call'].generate.return_value = ['API call 1']
    modules['log_retrieval'].retrieve_with_tunnel.return_value = {'log1': 'Log data'}
    modules['anomaly_detection'].detect.return_value = [
        {
            'description': 'Anomaly description',
            'confidence_score': 0.00,
            'potential_implications': 'Potential implications',
            'recommended_actions': 'Recommended actions'
        }
    ]
    modules['plugins'].get_active_plugins.return_value = []
    modules['report_generation'].generate.return_value = 'Test report'

    # Call the function
    result = await process_incident(incident, modules)

    # Assertions
    assert result == 'Test report'
    modules['understanding'].process.assert_called_once_with(incident)
    modules['api_call'].generate.assert_called_once()
    modules['log_retrieval'].retrieve_with_tunnel.assert_called_once()
    modules['anomaly_detection'].detect.assert_called_once()
    modules['report_generation'].generate.assert_called_once()
    # modules['output'].send.assert_called_once()
    # modules['feedback'].collect_feedback.assert_called_once()
