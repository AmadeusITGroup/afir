import pytest
from unittest.mock import AsyncMock, MagicMock
from src.main import process_incident, main


@pytest.mark.asyncio
async def test_process_incident():
    # Mock incident and modules
    incident = {'id': 'INC-001', 'description': 'Test incident'}
    modules = {
        'understanding': AsyncMock(),
        'api_call': AsyncMock(),
        'log_retrieval': AsyncMock(),
        'anomaly_detection': AsyncMock(),
        'plugins': MagicMock(),
        'report_generation': AsyncMock(),
        'output': AsyncMock(),
        'feedback': AsyncMock(),
    }

    # Set return values for mocks
    modules['understanding'].process.return_value = {'analysis': 'Test analysis'}
    modules['api_call'].generate.return_value = ['API call 1']
    modules['log_retrieval'].retrieve.return_value = {'log1': 'Log data'}
    modules['anomaly_detection'].detect.return_value = ['Anomaly 1']
    modules['plugins'].get_active_plugins.return_value = []
    modules['report_generation'].generate.return_value = 'Test report'

    # Call the function
    result = await process_incident(incident, modules)

    # Assertions
    assert result == 'Test report'
    modules['understanding'].process.assert_called_once_with(incident)
    modules['api_call'].generate.assert_called_once()
    modules['log_retrieval'].retrieve.assert_called_once()
    modules['anomaly_detection'].detect.assert_called_once()
    modules['report_generation'].generate.assert_called_once()
    modules['output'].send.assert_called_once()
    modules['feedback'].collect_feedback.assert_called_once()


@pytest.mark.asyncio
async def test_main(monkeypatch):
    # Mock configuration loading
    monkeypatch.setattr('src.main.load_config', MagicMock(return_value={}))

    # Mock module initializations
    mock_modules = {
        'input': MagicMock(),
        'understanding': MagicMock(),
        'api_call': MagicMock(),
        'log_retrieval': MagicMock(),
        'anomaly_detection': MagicMock(),
        'report_generation': MagicMock(),
        'output': MagicMock(),
        'plugins': MagicMock(),
        'feedback': MagicMock(),
    }
    monkeypatch.setattr('src.main.IncidentInputInterface', lambda _: mock_modules['input'])
    monkeypatch.setattr('src.main.IncidentUnderstandingModule', lambda _, __: mock_modules['understanding'])
    monkeypatch.setattr('src.main.ApiCallGenerator', lambda _, __: mock_modules['api_call'])
    monkeypatch.setattr('src.main.LogRetrievalEngine', lambda _: mock_modules['log_retrieval'])
    monkeypatch.setattr('src.main.AnomalyDetectionModule', lambda _, __, ___: mock_modules['anomaly_detection'])
    monkeypatch.setattr('src.main.ReportGenerationModule', lambda _, __, ___: mock_modules['report_generation'])
    monkeypatch.setattr('src.main.OutputInterface', lambda _: mock_modules['output'])
    monkeypatch.setattr('src.main.PluginManager', lambda _: mock_modules['plugins'])
    monkeypatch.setattr('src.main.FeedbackLoop', lambda _, __: mock_modules['feedback'])

    # Mock RAG initialization
    monkeypatch.setattr('src.main.RAG', MagicMock())

    # Mock NotificationSystem
    monkeypatch.setattr('src.main.NotificationSystem', AsyncMock())

    # Mock process_incident
    monkeypatch.setattr('src.main.process_incident', AsyncMock())

    # Run main function
    await main()

    # Assertions
    mock_modules['input'].start_server.assert_called_once()
    mock_modules['input'].get_incidents.assert_called()
