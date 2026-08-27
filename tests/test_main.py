from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.main import process_incident
from src.models.pydantic_models import (AnomalyItem, CorrelationResult,
                                        IncidentAnalysis, UnderstandingResult)


@pytest.mark.asyncio
@patch("src.main.ResultExporter")
async def test_process_incident(mock_exporter):
    incident = {
        "id": "INC-001",
        "description": "Test incident",
        "timestamp": "2024-03-04T21:34",
    }
    modules = {
        "understanding": AsyncMock(),
        "api_call": AsyncMock(),
        "log_retrieval": AsyncMock(),
        "correlation": AsyncMock(),
        "anomaly_detection": AsyncMock(),
        "plugins": MagicMock(),
        "report_generation": AsyncMock(),
        "output": AsyncMock(),
    }

    modules["understanding"].process.return_value = UnderstandingResult(
        incident_id="INC-001",
        analysis=IncidentAnalysis(
            incident_summary="Test analysis",
            severity="5",
            severity_reasoning="Test reasoning",
            impact_assessment="Test impact",
            key_investigation_areas=[],
            log_sources_to_review=[],
            initial_hypotheses=[],
            recommended_actions=[],
            stakeholder_notification=[],
        ),
    )
    modules["api_call"].generate.return_value = ["API call 1"]
    modules["log_retrieval"].config = {"use_ssh_tunnel": True}
    modules["log_retrieval"].retrieve_with_tunnel.return_value = {"log1": "Log data"}
    modules["correlation"].analyze.return_value = CorrelationResult(
        record_count=1, summary_text="correlated"
    )
    modules["anomaly_detection"].detect.return_value = [
        AnomalyItem(
            description="Anomaly description",
            supporting_data="Supporting data",
            confidence_score=0.00,
            potential_implications="Potential implications",
            recommended_actions="Recommended actions",
            patterns="Patterns",
        )
    ]
    modules["plugins"].get_active_plugins.return_value = []
    modules["report_generation"].generate.return_value = "Test report"

    result = await process_incident(incident, modules)

    # Assertions
    assert result == "Test report"
    modules["understanding"].process.assert_called_once_with(incident)
    modules["api_call"].generate.assert_called_once()
    modules["log_retrieval"].retrieve_with_tunnel.assert_called_once()
    modules["correlation"].analyze.assert_called_once()
    modules["anomaly_detection"].detect.assert_called_once()
    modules["report_generation"].generate.assert_called_once()
    # The report is now delivered through the configured output channel.
    modules["output"].send.assert_called_once_with("Test report", "INC-001")
    # The separate evidence artifacts (raw + transformed) are written alongside.
    exporter_instance = mock_exporter.return_value
    exporter_instance.export_evidence_raw.assert_called_once()
    exporter_instance.export_evidence_transformed.assert_called_once()
