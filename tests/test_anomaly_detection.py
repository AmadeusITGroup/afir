"""Tests for anomaly detection resilience.

Anomaly detection is the single LLM call over the largest payload (all logs +
understanding + correlation), so it is the most likely to hit a slow/rate-limited
serving endpoint. It is best-effort BY CONTRACT: it must never block the
investigation report. On any LLM failure it logs and returns [] so the deterministic
correlation result + full logs still flow to report generation.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.anomaly_detection import AnomalyDetectionModule
from src.models.pydantic_models import (AnomalyItem, AnomalyList,
                                        IncidentAnalysis, UnderstandingResult)


def _understanding():
    return UnderstandingResult(
        incident_id="INC-1",
        analysis=IncidentAnalysis(
            incident_summary="s",
            severity="5",
            severity_reasoning="r",
            impact_assessment="i",
            key_investigation_areas=[],
            log_sources_to_review=[],
            initial_hypotheses=[],
            recommended_actions=[],
            stakeholder_notification=[],
        ),
    )


def _anomaly(desc, conf):
    return AnomalyItem(
        description=desc,
        supporting_data="d",
        confidence_score=conf,
        potential_implications="p",
        recommended_actions="a",
        patterns="pat",
    )


@pytest.mark.asyncio
async def test_detect_returns_empty_on_llm_failure_instead_of_raising():
    """A persistently failing LLM must NOT propagate — the report must still run."""
    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=TimeoutError("Request timed out."))
    module = AnomalyDetectionModule({"threshold": 0.5}, llm)

    result = await module.detect(
        {"src": [{"a": 1}]}, _understanding(), correlation=None
    )

    assert result == []  # best-effort: swallowed, not raised


@pytest.mark.asyncio
async def test_detect_marks_sub_threshold_anomalies_instead_of_dropping_them():
    """The threshold decides what is NARRATED, not what is kept.

    It used to delete them, and the deletion was upstream of the exports, so a
    sub-threshold finding reached neither the report, the exports, the PDF table nor the
    evidence artifacts. Against the shipped ``threshold: 0.8`` that discards a 0.79
    finding outright: an incident nobody scored confidently and one scored just under the
    line produce byte-identical artifacts. So every item comes back, carrying its mark.
    """
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=AnomalyList(
            anomalies=[_anomaly("high", 0.9), _anomaly("low", 0.2)]
        )
    )
    module = AnomalyDetectionModule({"threshold": 0.5}, llm)

    result = await module.detect(
        {"src": [{"a": 1}]}, _understanding(), correlation=None
    )

    assert [a.description for a in result] == ["high", "low"]
    assert [a.below_threshold for a in result] == [False, True]
    # And the module records what it held back, because the returned list no longer
    # distinguishes them by absence and the health scorer must not re-derive the cut-off.
    assert module.last_filtered_out == 1
    assert module.last_filtered_max == pytest.approx(0.2)
    assert module.last_threshold_used == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_the_model_does_not_get_to_declare_itself_above_threshold():
    """``below_threshold`` is in the response schema, so the model can emit it. Whether a
    score clears a cut-off is arithmetic the engine owns — same reason gate health is
    never self-assessment."""
    lying = _anomaly("low but confident", 0.2)
    lying.below_threshold = False
    honest = _anomaly("high", 0.9)
    honest.below_threshold = True
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=AnomalyList(anomalies=[lying, honest])
    )
    module = AnomalyDetectionModule({"threshold": 0.5}, llm)

    result = await module.detect(
        {"src": [{"a": 1}]}, _understanding(), correlation=None
    )

    assert [a.below_threshold for a in result] == [True, False]


@pytest.mark.asyncio
async def test_detect_uses_evidence_pack_when_present():
    """When correlation carries an evidence pack, its rendered text (chronology/actors)
    is embedded in the prompt — NOT a raw [source] {...} JSON dump."""
    from src.evidence import build_evidence

    logs = {
        "app": [{"user": "USERNAMEX", "org_unit": "LBV", "ts": 1721853060000}],
    }
    evidence = build_evidence(
        logs, {"total_records": 1}, [], {"app": {"user": "user"}}, ["USERNAMEX"]
    )
    correlation = MagicMock()
    correlation.evidence = evidence
    correlation.summary_text = "corr summary"

    captured = {}

    async def _capture(messages, **kwargs):
        captured["content"] = messages[1]["content"]
        return AnomalyList(anomalies=[])

    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=_capture)
    module = AnomalyDetectionModule({"threshold": 0.5}, llm)

    await module.detect(logs, _understanding(), correlation=correlation)

    content = captured["content"]
    assert "[CHRONOLOGY]" in content and "USERNAMEX" in content
    assert '[app] {"' not in content  # not the raw preprocess_logs dump


@pytest.mark.asyncio
async def test_detect_falls_back_to_preprocess_when_no_evidence():
    """No evidence pack -> the raw preprocess_logs dump is used (back-compat)."""
    captured = {}

    async def _capture(messages, **kwargs):
        captured["content"] = messages[1]["content"]
        return AnomalyList(anomalies=[])

    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=_capture)
    module = AnomalyDetectionModule({"threshold": 0.5}, llm)

    await module.detect({"src": [{"a": 1}]}, _understanding(), correlation=None)

    assert '[src] {"a": 1}' in captured["content"]


# --- verdict-aware anomaly scoring (framing + deterministic clamp) ----------

from src.models.pydantic_models import (ConditionCheck, InvestigationBrief,
                                        SubjectVerdict, ValidationVerdict)


# The dismissal word deliberately is NOT the literal "FALSE POSITIVE". Every shipped
# ruleset names its own verdict vocabulary and none of them uses that phrase, so a fixture
# that did would test a string no real run produces — which is exactly how the clamp below
# came to be dead code on every live incident while these tests passed.
_LABELS = {
    "fraud": "CONFIRMED SCHEME",
    "false_positive": "NOT A SCHEME",
    "insufficient": "INSUFFICIENT DATA",
}


def _brief_correlation(verdict_label, labels=None, polarity="exclusion"):
    v = ValidationVerdict(
        label_scheme="scheme",
        summary=f"{verdict_label}: 1",
        labels=_LABELS if labels is None else labels,
        subjects=[
            SubjectVerdict(
                subject_type="record",
                subject_value="SUBJ03",
                verdict=verdict_label,
                checks=[
                    ConditionCheck(
                        id="bare",
                        label="Bare record",
                        result="fail",
                        decisive=True,
                        polarity=polarity,
                        detail="record carries no counterparty reference",
                    ),
                ],
            )
        ],
    )
    corr = MagicMock()
    corr.evidence = None
    corr.summary_text = "corr summary"
    corr.brief = InvestigationBrief(use_case="scheme", verdict=v)
    return corr


@pytest.mark.asyncio
async def test_a_dismissed_verdict_injects_framing_and_clamps_confidence():
    captured = {}

    async def _capture(messages, **kwargs):
        captured["messages"] = messages
        return AnomalyList(anomalies=[_anomaly("looks bad", 0.95)])

    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=_capture)
    module = AnomalyDetectionModule({"threshold": 0.0}, llm)

    result = await module.detect(
        {"record_lake": [{"a": 1}]},
        _understanding(),
        correlation=_brief_correlation("NOT A SCHEME"),
    )
    # Framing system message injected, naming the verdict in the PACK's words.
    system_texts = "\n".join(
        m["content"] for m in captured["messages"] if m["role"] == "system"
    )
    assert "NOT A SCHEME" in system_texts
    # THE FINDING, NOT THE REQUIREMENT. This check FAILED, and under exclusion polarity a
    # FAIL negates its label — so framing the model with "Bare record" would assert the
    # opposite of what was found. `detail` is the evaluator's finding-phrased note.
    assert "record carries no counterparty reference" in system_texts
    assert "Bare record" not in system_texts
    # Deterministic clamp: 0.95 -> 0.5 regardless of what the model returned.
    assert result[0].confidence_score == 0.5


@pytest.mark.asyncio
async def test_a_fraud_verdict_does_not_clamp():
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=AnomalyList(anomalies=[_anomaly("confirmed", 0.95)])
    )
    module = AnomalyDetectionModule({"threshold": 0.0}, llm)
    result = await module.detect(
        {"record_lake": [{"a": 1}]},
        _understanding(),
        # A decisive fraud_indicator FAIL: under that polarity a FAIL AFFIRMS the label,
        # so this subject is adjudicated fraud and nothing may be demoted.
        correlation=_brief_correlation("CONFIRMED SCHEME", polarity="fraud_indicator"),
    )
    assert result[0].confidence_score == 0.95  # unchanged


@pytest.mark.asyncio
async def test_the_clamp_compares_against_the_packs_word_not_a_hardcoded_one():
    """The regression this exists for: the test was a substring match on "false positive",
    which no shipped ruleset emits, so the clamp was dead on every live incident — a
    report carried a 0.93-confidence anomaly under a verdict that dismissed it. The
    comparison is now against `verdict.labels`, whatever the pack calls it."""
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=AnomalyList(anomalies=[_anomaly("looks bad", 0.93)])
    )
    module = AnomalyDetectionModule({"threshold": 0.0}, llm)
    result = await module.detect(
        {"record_lake": [{"a": 1}]},
        _understanding(),
        correlation=_brief_correlation(
            "AUCUNE FRAUDE", labels={"false_positive": "AUCUNE FRAUDE"}
        ),
    )
    assert result[0].confidence_score == 0.5


@pytest.mark.asyncio
async def test_the_stamped_class_decides_and_outranks_the_label():
    """`verdict_class` is the rollup's own answer, so it is read first.

    The label is a pack word that a consumer can only compare; the class is what the
    six-way rollup actually decided. Where they could disagree the class wins — asserted
    with a subject whose LABEL matches the dismissal word while its class says fraud.
    """
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=AnomalyList(anomalies=[_anomaly("confirmed", 0.95)])
    )
    module = AnomalyDetectionModule({"threshold": 0.0}, llm)
    corr = _brief_correlation("NOT A SCHEME")
    corr.brief.verdict.subjects[0].verdict_class = "fraud"

    result = await module.detect(
        {"record_lake": [{"a": 1}]}, _understanding(), correlation=corr
    )

    assert result[0].confidence_score == 0.95  # not clamped: the class says fraud


@pytest.mark.asyncio
async def test_a_verdict_with_neither_class_nor_labels_clamps_nothing():
    """A verdict from an older store carries no `verdict_class` AND no `labels`.

    It does NOT fall back to re-deriving the rollup from the checks. That version shipped
    and was measured wrong on real stored jobs: "a decisive exclusion FAIL with no decisive
    indicator FAIL" read two VALID FRAUD subjects and one FRAUD subject as dismissed,
    because the rollup also reaches fraud through corroborated NON-decisive indicators and
    a categorical exclusion outranks indicators entirely. The clamp DEMOTES, so with no
    trustworthy signal the safe answer is to leave the scores alone.
    """
    llm = MagicMock()
    llm.structured_output = AsyncMock(
        return_value=AnomalyList(anomalies=[_anomaly("looks bad", 0.95)])
    )
    module = AnomalyDetectionModule({"threshold": 0.0}, llm)
    result = await module.detect(
        {"record_lake": [{"a": 1}]},
        _understanding(),
        correlation=_brief_correlation("NOT A SCHEME", labels={}),
    )
    assert result[0].confidence_score == 0.95


@pytest.mark.asyncio
async def test_no_brief_no_framing():
    captured = {}

    async def _capture(messages, **kwargs):
        captured["messages"] = messages
        return AnomalyList(anomalies=[_anomaly("x", 0.9)])

    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=_capture)
    module = AnomalyDetectionModule({"threshold": 0.0}, llm)
    result = await module.detect(
        {"src": [{"a": 1}]}, _understanding(), correlation=None
    )
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user"]  # no injected framing
    assert result[0].confidence_score == 0.9  # not clamped


# --- degradation channel (read by src/stage_health.py) ---------------------


@pytest.mark.asyncio
async def test_llm_failure_is_recorded_so_health_can_tell_it_from_a_clean_incident():
    """`detect` returns [] both ways; only this flag separates broken from quiet."""
    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=TimeoutError("Request timed out."))
    module = AnomalyDetectionModule({"threshold": 0.5}, llm)

    assert module.last_degraded is False  # not degraded before it has run

    result = await module.detect(
        {"src": [{"a": 1}]}, _understanding(), correlation=None
    )

    assert result == []  # contract unchanged
    assert module.last_degraded is True
    assert "timed out" in module.last_error


@pytest.mark.asyncio
async def test_a_genuinely_empty_result_is_not_marked_degraded():
    llm = MagicMock()
    llm.structured_output = AsyncMock(return_value=AnomalyList(anomalies=[]))
    module = AnomalyDetectionModule({"threshold": 0.5}, llm)

    assert await module.detect({"src": [{"a": 1}]}, _understanding()) == []
    assert module.last_degraded is False


@pytest.mark.asyncio
async def test_a_later_success_clears_an_earlier_failure():
    """Stale state would keep re-reporting a fatal signal after a successful retry."""
    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=TimeoutError("boom"))
    module = AnomalyDetectionModule({"threshold": 0.5}, llm)
    await module.detect({"src": [{"a": 1}]}, _understanding())
    assert module.last_degraded is True

    llm.structured_output = AsyncMock(
        return_value=AnomalyList(anomalies=[_anomaly("high", 0.9)])
    )
    result = await module.detect({"src": [{"a": 1}]}, _understanding())

    assert len(result) == 1
    assert module.last_degraded is False
    assert module.last_error == ""


@pytest.mark.asyncio
async def test_a_truncated_anomaly_list_retries_once_at_a_wider_budget():
    """A truncation is the one failure with a known remedy, and the alternative outcome
    is the worst this module produces: the best-effort contract turns an incomplete
    response into a clean-looking "0 anomalies" that scores as FATAL. So it must widen
    the budget and re-ask itself, not wait for a human to edit detect_max_tokens."""
    from src.utils.llm_client import LLMTruncatedError

    budgets = []

    async def fake_structured_output(
        messages, response_model, max_tokens=None, rag=None, stage=None
    ):
        budgets.append(max_tokens)
        if len(budgets) == 1:
            raise LLMTruncatedError("hit the 8000-token limit")
        return AnomalyList(anomalies=[_anomaly("late finding", 0.9)])

    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=fake_structured_output)
    module = AnomalyDetectionModule({"threshold": 0.5}, llm)

    result = await module.detect({"src": [{"a": 1}]}, _understanding())

    assert len(budgets) == 2
    assert budgets[1] > budgets[0]  # the same budget would truncate identically
    assert budgets[1] <= AnomalyDetectionModule._TRUNCATION_RETRY_CEILING
    assert len(result) == 1
    # The retry SUCCEEDED, so this is not a degradation — but the config observation
    # (the budget is undersized for this data volume) is still reported.
    assert module.last_degraded is False
    assert module.last_truncation_retried is True


@pytest.mark.asyncio
async def test_a_second_truncation_gives_up_on_the_best_effort_contract():
    """Two overruns mean the prompt, not the budget. Still [], still not raised."""
    from src.utils.llm_client import LLMTruncatedError

    llm = MagicMock()
    llm.structured_output = AsyncMock(
        side_effect=LLMTruncatedError("hit the token limit")
    )
    module = AnomalyDetectionModule({"threshold": 0.5}, llm)

    assert await module.detect({"src": [{"a": 1}]}, _understanding()) == []
    assert llm.structured_output.await_count == 2
    assert module.last_degraded is True


@pytest.mark.asyncio
async def test_no_retry_when_the_configured_budget_is_already_at_the_ceiling():
    """Doubling past the ceiling clamps to the same number, and re-asking at the same
    budget just spends a second slow call to fail identically."""
    from src.utils.llm_client import LLMTruncatedError

    llm = MagicMock()
    llm.structured_output = AsyncMock(
        side_effect=LLMTruncatedError("hit the token limit")
    )
    module = AnomalyDetectionModule(
        {
            "threshold": 0.5,
            "detect_max_tokens": AnomalyDetectionModule._TRUNCATION_RETRY_CEILING,
        },
        llm,
    )

    assert await module.detect({"src": [{"a": 1}]}, _understanding()) == []
    assert llm.structured_output.await_count == 1
    assert module.last_truncation_retried is False


@pytest.mark.asyncio
async def test_truncation_is_caught_across_both_module_identities():
    """The retry must fire on the exception a REAL client raises.

    `src/utils/llm_client.py` is importable as both `utils.llm_client` (main.py's flat
    style) and `src.utils.llm_client`, which are two module objects with two distinct
    `LLMTruncatedError` classes. A stage spelling `except LLMTruncatedError` therefore
    catches only the identity it happened to import, and the exception raised by a
    client built through the other path sails straight past it into the best-effort
    handler — 0 anomalies, FATAL health, no retry. That is exactly how report
    generation's retry was dead in production with its unit test green, so pin BOTH
    identities here.
    """
    import src.utils.llm_client as qualified
    import utils.llm_client as flat

    assert (
        flat.LLMTruncatedError is not qualified.LLMTruncatedError
    ), "the two identities collapsed into one — this test no longer proves anything"

    for module_under_test in (flat, qualified):
        budgets = []

        async def fake_structured_output(
            messages, response_model, max_tokens=None, rag=None, stage=None, _m=module_under_test
        ):
            budgets.append(max_tokens)
            if len(budgets) == 1:
                raise _m.LLMTruncatedError("hit the 8000-token limit")
            return AnomalyList(anomalies=[_anomaly("late finding", 0.9)])

        llm = MagicMock()
        llm.structured_output = AsyncMock(side_effect=fake_structured_output)
        module = AnomalyDetectionModule({"threshold": 0.5}, llm)

        assert len(await module.detect({"src": [{"a": 1}]}, _understanding())) == 1
        assert len(budgets) == 2, f"no retry for {module_under_test.__name__}"
        assert module.last_truncation_retried is True


@pytest.mark.asyncio
async def test_a_non_truncation_failure_is_not_retried():
    """The marker check must not turn every exception into a second slow call."""
    llm = MagicMock()
    llm.structured_output = AsyncMock(side_effect=TimeoutError("boom"))
    module = AnomalyDetectionModule({"threshold": 0.5}, llm)

    assert await module.detect({"src": [{"a": 1}]}, _understanding()) == []
    assert llm.structured_output.await_count == 1
    assert module.last_truncation_retried is False
    assert module.last_degraded is True
