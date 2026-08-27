"""Tests for FeedbackLoop batching + insight persistence (LLM mocked)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.feedback_loop import FeedbackLoop
from src.models.pydantic_models import FeedbackInsights


def _insights():
    return FeedbackInsights(
        common_success_patterns=["p1"],
        frequently_missed_anomalies=[],
        accuracy_improvements=["lift recall"],
        confidence_threshold_recommendations=[],
        recommended_action_effectiveness=[],
        new_fraud_patterns=[],
        process_improvements=[],
    )


@pytest.mark.asyncio
async def test_collect_batches_until_ten(tmp_path, monkeypatch):
    # Entries persist on arrival, so the store must be isolated per test — otherwise a
    # pending batch left by an earlier run is reloaded and the count starts above zero.
    monkeypatch.setattr("src.feedback_loop.data_dir", lambda: tmp_path)

    loop = FeedbackLoop(MagicMock())
    loop.process_feedback = AsyncMock()
    for i in range(9):
        await loop.collect_feedback(f"INC{i}", "report", "good")
    loop.process_feedback.assert_not_awaited()
    await loop.collect_feedback("INC9", "report", "good")
    loop.process_feedback.assert_awaited_once()


@pytest.mark.asyncio
async def test_apply_insights_persists_file(tmp_path, monkeypatch):
    # Redirect the data dir to the tmp path.
    monkeypatch.setattr("src.feedback_loop.data_dir", lambda: tmp_path)

    loop = FeedbackLoop(MagicMock())
    await loop.apply_insights(_insights())

    out = tmp_path / "feedback_insights.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["insights"]["common_success_patterns"] == ["p1"]

    # A second call appends rather than overwrites.
    await loop.apply_insights(_insights())
    assert len(json.loads(out.read_text())) == 2


@pytest.mark.asyncio
async def test_process_feedback_distills_and_clears(tmp_path, monkeypatch):
    monkeypatch.setattr("src.feedback_loop.data_dir", lambda: tmp_path)
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=_insights())
    loop = FeedbackLoop(llm)
    loop.feedback_data = [
        {"incident_id": "X", "investigation_result": "r", "human_feedback": "f"}
    ]

    result = await loop.process_feedback()
    assert result is not None
    assert loop.feedback_data == []  # cleared after processing
    llm.structured_output.assert_awaited_once()
