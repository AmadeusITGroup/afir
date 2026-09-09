"""
Tests for adaptive report sizing.

report_generation bounds what the LLM *narrates* (top-N anomalies by confidence +
truncated correlation) to keep the single report call under the endpoint's per-request
limit, but ALL anomalies still land in the exported artifact (txt/JSON here). These
tests assert both halves of that contract.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.pydantic_models import (AnomalyItem, IncidentAnalysis,
                                        InvestigationReport,
                                        UnderstandingResult)
from src.report_generation import ReportGenerationModule


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
async def test_report_prompt_capped_but_exports_all(tmp_path):
    """Only top-N anomalies reach the LLM prompt; all N reach the exported artifact."""
    config = {
        "output_format": "txt",
        "output_path": str(tmp_path),
        "max_anomalies_in_prompt": 3,
        "llm_input_char_budget": 15000,
    }
    llm = MagicMock()
    llm.max_tokens = 4096
    captured = {}

    async def fake_structured_output(
        messages, response_model, max_tokens=None, rag=None, stage=None
    ):
        captured["prompt"] = messages[1]["content"]
        captured["max_tokens"] = max_tokens
        return InvestigationReport(sections=[])

    llm.structured_output = AsyncMock(side_effect=fake_structured_output)

    module = ReportGenerationModule(config, llm)

    # 10 anomalies with ascending confidence; the top 3 are conf 0.9, 0.8, 0.7.
    anomalies = [_anomaly(f"anomaly-{i}", i / 10.0) for i in range(10)]

    text = await module.generate(
        {"id": "INC-1"}, _understanding(), {"src": []}, anomalies, correlation=None
    )

    prompt = captured["prompt"]
    # Highest-confidence descriptions ARE in the prompt.
    assert "anomaly-9" in prompt and "anomaly-8" in prompt and "anomaly-7" in prompt
    # A lower-confidence one is NOT (capped at 3).
    assert "anomaly-0" not in prompt
    # The omission is disclosed to the model.
    assert "additional lower-confidence anomalies omitted" in prompt

    # The exported artifact (txt = JSON sections) is produced; sections came back empty
    # here, but the exported file exists (all-anomalies detail lives in PDF table path).
    assert isinstance(text, str)

    # max_tokens is the configured ceiling (here the client's 4096, since no
    # report_max_tokens is set) and never below the floor.
    assert captured["max_tokens"] <= 4096
    assert captured["max_tokens"] >= config.get("report_min_tokens", 1500)


@pytest.mark.asyncio
async def test_report_truncates_correlation_to_budget(tmp_path):
    """A large correlation blob is truncated to the char budget before the LLM call."""
    config = {
        "output_format": "txt",
        "output_path": str(tmp_path),
        "max_anomalies_in_prompt": 15,
        "llm_input_char_budget": 200,
    }
    llm = MagicMock()
    llm.max_tokens = 4096
    captured = {}

    async def fake_structured_output(
        messages, response_model, max_tokens=None, rag=None, stage=None
    ):
        captured["prompt"] = messages[1]["content"]
        return InvestigationReport(sections=[])

    llm.structured_output = AsyncMock(side_effect=fake_structured_output)
    module = ReportGenerationModule(config, llm)

    # A correlation object whose JSON far exceeds the 200-char budget.
    correlation = MagicMock()
    correlation.evidence = None  # force the raw-correlation truncation path
    correlation.summary_text = ""  # real scalar (not a MagicMock) for backfill safety
    correlation.record_count = 0
    correlation.model_dump_json = MagicMock(return_value="X" * 5000)

    await module.generate(
        {"id": "INC-1"},
        _understanding(),
        {"src": []},
        [_anomaly("a", 0.5)],
        correlation=correlation,
    )

    assert "... [truncated]" in captured["prompt"]
    # The full 5000-char blob is not present.
    assert "X" * 5000 not in captured["prompt"]


@pytest.mark.asyncio
async def test_report_falls_back_deterministically_when_llm_fails(tmp_path):
    """The report is the deliverable: if the narration LLM call fails (slow / rate-
    limited endpoint), generate() must still return a report assembled from the data
    in hand — never raise and fail the whole investigation."""
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    # Every narration attempt raises (mimics a persistently timing-out endpoint).
    llm.structured_output = AsyncMock(side_effect=TimeoutError("Request timed out."))

    module = ReportGenerationModule(config, llm)
    anomalies = [_anomaly("burst of documents", 0.9)]
    logs = {"siem_alerts_current": [{"a": 1}], "app_session_events": [{"b": 2}]}

    text = await module.generate(
        {"id": "INC-1"}, _understanding(), logs, anomalies, correlation=None
    )

    # A report IS produced (string of JSON sections), not an exception.
    assert isinstance(text, str) and text
    # It is the deterministic fallback: discloses the model was unavailable and
    # includes the retrieved evidence + the anomaly.
    assert "deterministic" in text.lower()
    assert "siem_alerts_current" in text
    assert "burst of documents" in text


@pytest.mark.asyncio
async def test_report_uses_llm_sections_when_available(tmp_path):
    """When narration succeeds, the LLM's sections are used (no fallback marker)."""
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(
        return_value=InvestigationReport(
            sections=[{"section_title": "Executive Summary", "content": "All clear."}]
        )
    )
    module = ReportGenerationModule(config, llm)

    text = await module.generate(
        {"id": "INC-1"},
        _understanding(),
        {"src": []},
        [_anomaly("a", 0.5)],
        correlation=None,
    )
    assert "All clear." in text
    # The LLM's Executive Summary is kept — the deterministic-fallback NOTE (only emitted
    # when narration fails entirely) must NOT appear. (The word "deterministically" can
    # legitimately appear in the Evidence Artifacts description, so match the note text.)
    assert "narration model was unavailable" not in text.lower()


def _understanding_with_actions(actions):
    u = _understanding()
    u.analysis.recommended_actions = actions
    return u


def _evidence_correlation():
    """A correlation object carrying a real evidence pack (LH-style incident)."""
    from src.evidence import build_evidence

    logs = {
        "app": [
            {
                "user": "USERNAMEX",
                "org_unit": "LBV",
                "record": "SUBJ03",
                "ts": 1721853060000,
                "action": "ISSUE",
            }
        ],
        "record_lake": [{"locator": "SUBJ03", "off": "LBV", "cdate": "2026-07-24"}],
    }
    emap = {
        "app": {"user": "user", "org_unit": "org_unit", "record": "record"},
        "record_lake": {"record": "locator", "org_unit": "off"},
    }

    class _K:
        entity_hint = "record"
        sources = {"app": "record", "record_lake": "locator"}
        time_fields = {"app": "ts"}

    evidence = build_evidence(
        logs,
        {
            "total_records": 2,
            "cross_source_overlap": {"SUBJ03": {"app": 1, "record_lake": 1}},
        },
        [_K()],
        emap,
        ["USERNAMEX", "SUBJ03", "LBV"],
    )
    correlation = MagicMock()
    correlation.evidence = evidence
    correlation.summary_text = "correlated"
    correlation.record_count = 2
    return correlation


@pytest.mark.asyncio
async def test_fallback_recommendations_derived_from_evidence_not_understanding(
    tmp_path,
):
    """When the LLM fails, 'Recommended Next Steps' come from the evidence (responsible
    actor, cross-source joins, anomalies) — NOT the pre-retrieval understanding actions.
    """
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(side_effect=TimeoutError("down"))
    module = ReportGenerationModule(config, llm)

    generic = [
        "IMMEDIATE: Retrieve the full SCHEME alert payload to obtain session id."
    ]
    understanding = _understanding_with_actions(generic)

    text = await module.generate(
        {"id": "INC-1"},
        understanding,
        {"app": [{}]},
        [_anomaly("burst of documents issued", 0.9)],
        correlation=_evidence_correlation(),
    )

    # The generic pre-retrieval data-gathering action must NOT be the recommendation.
    assert "Retrieve the full SCHEME alert payload" not in text
    # Investigation-derived steps reference the actual actor / entity / anomaly.
    assert "USERNAMEX" in text
    assert "CONTAIN" in text or "ESCALATE" in text
    assert "burst of documents issued" in text


def test_fallback_recommendations_target_only_subjects():
    """CONTAINment must target the incident's subject actors, never higher-volume
    background identities that only share the time window (the run-7 regression where
    the report recommended locking unrelated agents)."""
    from src.models.pydantic_models import ActorRollup

    evidence = MagicMock()
    # Two subjects (low volume) + a high-volume background actor.
    evidence.actors = [
        ActorRollup(
            actor="0201GP", event_count=17, sources=["raw_access"], is_subject=True
        ),
        ActorRollup(
            actor="USERNAMEX", event_count=9, sources=["auth_svc"], is_subject=True
        ),
        ActorRollup(
            actor="0007VP", event_count=67, sources=["raw_access"], is_subject=False
        ),
    ]
    evidence.cross_source_joins = []
    steps = ReportGenerationModule._fallback_recommendations([], evidence, None)
    contain = [s for s in steps if s.startswith("CONTAIN")]
    joined = " ".join(contain)
    assert "0201GP" in joined and "USERNAMEX" in joined
    # The high-volume background actor is NOT a containment target.
    assert "0007VP" not in joined


def test_readable_correlation_summary_is_not_json_dump():
    """The Correlation Analysis section must be readable prose, never the raw JSON
    summary_text blob (the run-7 regression)."""
    from src.models.pydantic_models import CorrelationResult, TransformResult

    corr = CorrelationResult(
        record_count=1053,
        summary_text='{"record_counts": {"a": 1}}',  # JSON blob — must NOT be printed
        aggregations={"resolved_correlation_keys": [{"entity_hint": "org_unit"}]},
        transforms=[
            TransformResult(
                label="join_org_unit",
                op="cross_source_overlap",
                rows=[{"value": "ORG2428D4", "sources": ["auth_svc", "raw_access"]}],
            )
        ],
    )
    lines = ReportGenerationModule._readable_correlation_summary(corr)
    text = " ".join(lines)
    assert "{" not in text and "record_counts" not in text
    assert "ORG2428D4" in text and "join_org_unit" in text


@pytest.mark.asyncio
async def test_fallback_findings_use_chronology_when_no_anomalies(tmp_path):
    """LLM fails and no anomalies were scored, but evidence exists -> the findings section
    is a chronology/actor reconstruction, not 'No anomalies were scored'."""
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(side_effect=TimeoutError("down"))
    module = ReportGenerationModule(config, llm)

    text = await module.generate(
        {"id": "INC-1"},
        _understanding(),
        {"app": [{}]},
        [],  # no anomalies
        correlation=_evidence_correlation(),
    )

    assert "USERNAMEX" in text  # actor attribution surfaced
    assert "deterministic reconstruction" in text.lower()


@pytest.mark.asyncio
async def test_narrate_embeds_evidence_text(tmp_path):
    """When evidence exists, the narration prompt embeds the rendered evidence
    (chronology labels), not a raw correlation JSON dump."""
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    captured = {}

    async def fake_structured_output(
        messages, response_model, max_tokens=None, rag=None, stage=None
    ):
        captured["prompt"] = messages[1]["content"]
        return InvestigationReport(sections=[])

    llm.structured_output = AsyncMock(side_effect=fake_structured_output)
    module = ReportGenerationModule(config, llm)

    await module.generate(
        {"id": "INC-1"},
        _understanding(),
        {"app": [{}]},
        [_anomaly("a", 0.5)],
        correlation=_evidence_correlation(),
    )
    assert "[CHRONOLOGY]" in captured["prompt"]
    assert "USERNAMEX" in captured["prompt"]


@pytest.mark.asyncio
async def test_report_names_evidence_artifacts_when_narration_succeeds(tmp_path):
    """The report references the separate raw/transformed evidence files (deterministic
    filenames from the incident id) even when narration succeeds."""
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(
        return_value=InvestigationReport(
            sections=[{"section_title": "Executive Summary", "content": "All clear."}]
        )
    )
    module = ReportGenerationModule(config, llm)

    text = await module.generate(
        {"id": "INC-1"},
        _understanding(),
        {"src": []},
        [_anomaly("a", 0.5)],
        correlation=None,
    )
    assert "Evidence Artifacts" in text
    assert "evidence_raw_INC-1.json" in text
    assert "evidence_transformed_INC-1.json" in text


@pytest.mark.asyncio
async def test_report_names_evidence_artifacts_in_fallback(tmp_path):
    """The Evidence Artifacts reference is present in the deterministic fallback too."""
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(side_effect=TimeoutError("down"))
    module = ReportGenerationModule(config, llm)

    text = await module.generate(
        {"id": "INC-1"},
        _understanding(),
        {"app": [{"a": 1}]},
        [_anomaly("a", 0.5)],
        correlation=None,
    )
    assert "Evidence Artifacts" in text
    assert "evidence_raw_INC-1.json" in text
    assert "evidence_transformed_INC-1.json" in text
    # The report stays lean: no raw per-row JSON dump in the body.
    assert '{"a": 1}' not in text


def test_render_markdown_produces_readable_document():
    """The Markdown renderer turns the section list into a real document: # title, ##
    sections, ### nested subsections, and bulleted string lists — and NO redundant
    anomalies table (findings/implications/actions live in their own sections now)."""
    sections = [
        {"section_title": "Executive Summary", "content": "Fraud at ORG2428D4."},
        {
            "section_title": "Recommended Next Steps",
            "content": ["IMMEDIATE — Lock org_unit.", "URGENT — Audit signs."],
        },
        {
            "section_title": "Incident Overview",
            "content": {"section_title": "Details", "content": ["OrgUnit: ORG2428D4"]},
        },
    ]
    md = ReportGenerationModule.render_markdown(sections, {"id": "INC-9"})
    assert md.startswith("# Fraud Investigation Report — Incident INC-9")
    assert "## Executive Summary" in md
    assert "Fraud at ORG2428D4." in md
    # String lists become bullets.
    assert "- IMMEDIATE — Lock org_unit." in md
    # Nested subsection is a level deeper.
    assert "### Details" in md
    # No redundant anomalies table.
    assert "| Confidence | Description |" not in md
    # It is NOT the JSON dump.
    assert '"section_title"' not in md


@pytest.mark.asyncio
async def test_narration_missing_sections_are_backfilled(tmp_path):
    """When the LLM narrates only an Executive Summary (observed on the fallback path),
    the report must still gain the missing required sections deterministically —
    Analysis and Findings and Recommended Next Steps in particular."""
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    # Narration returns a single collapsed section.
    llm.structured_output = AsyncMock(
        return_value=InvestigationReport(
            sections=[{"section_title": "Executive Summary", "content": "Fraud found."}]
        )
    )
    module = ReportGenerationModule(config, llm)

    text = await module.generate(
        {"id": "INC-1"},
        _understanding(),
        {"raw_access": [{"x": 1}]},
        [_anomaly("dup issuance", 0.93)],
        correlation=None,
    )
    sections = json.loads(text)
    titles = " ".join(s.get("section_title", "") for s in sections).lower()
    # The LLM's own section is kept, and the key missing ones are backfilled.
    assert "executive summary" in titles
    assert "analysis and findings" in titles
    assert "recommended next steps" in titles
    # More than the single collapsed section + evidence artifacts.
    assert len(sections) >= 5


# --- the report must be narrated, not truncated into a bullet dump -----------
#
# Measured on a 0-anomaly verdicted incident: at max_tokens=4096 the endpoint returned
# finish_reason='length' with content '{}'; at 8000, all seven sections. The old sizing
# (report_min_tokens + 250/anomaly) gave that incident the 4000 floor, so the one with the
# most to explain got the smallest budget.


def test_zero_anomalies_still_gets_the_full_token_budget():
    """A verdicted incident can score 0 anomalies and still need a full chronology."""
    llm = MagicMock()
    llm.max_tokens = 4096
    config = {"report_min_tokens": 4000, "report_max_tokens": 8000}
    module = ReportGenerationModule(config, llm)

    # The regression: 0 anomalies must NOT collapse to the floor.
    assert module._report_max_tokens([]) == 8000
    # And a busy incident gets the same ceiling, not more.
    assert (
        module._report_max_tokens([_anomaly(f"a{i}", 0.9) for i in range(15)]) == 8000
    )


def test_report_min_tokens_still_floors_a_small_ceiling():
    """A deployment configuring a ceiling below the floor gets the floor."""
    llm = MagicMock()
    llm.max_tokens = 4096
    module = ReportGenerationModule(
        {"report_min_tokens": 4000, "report_max_tokens": 1000}, llm
    )
    assert module._report_max_tokens([]) == 4000


@pytest.mark.asyncio
async def test_truncated_narration_retries_once_at_a_wider_budget(tmp_path):
    """A truncation has a known remedy, so spend one more call rather than fall back to
    deterministic bullets — the synthesis IS this stage's deliverable."""
    from src.utils.llm_client import LLMTruncatedError

    config = {
        "output_format": "txt",
        "output_path": str(tmp_path),
        "report_min_tokens": 4000,
        "report_max_tokens": 8000,
    }
    llm = MagicMock()
    llm.max_tokens = 4096
    budgets = []

    async def fake_structured_output(
        messages, response_model, max_tokens=None, rag=None, stage=None
    ):
        budgets.append(max_tokens)
        if len(budgets) == 1:
            raise LLMTruncatedError("hit the 8000-token limit")
        return InvestigationReport(
            sections=[
                {"section_title": t, "content": "narrated prose"}
                for t in ReportGenerationModule._REQUIRED_SECTIONS
            ]
        )

    llm.structured_output = AsyncMock(side_effect=fake_structured_output)
    module = ReportGenerationModule(config, llm)

    text = await module.generate(
        {"id": "INC-1"},
        _understanding(),
        {"raw_access": [{"x": 1}]},
        [],
        correlation=None,
    )
    sections = json.loads(text)

    # Retried, and at a WIDER budget (the same budget would truncate identically).
    assert len(budgets) == 2
    assert budgets[1] > budgets[0]
    assert budgets[1] <= ReportGenerationModule._TRUNCATION_RETRY_CEILING
    # The retry's narration is used, so nothing was backfilled and no fallback ran.
    assert module.last_fallback_used is False
    assert module.last_backfilled_sections == []
    assert module.last_truncation_retried is True
    # Every required section is the LLM's prose, not a deterministic bullet list.
    for s in sections:
        if s.get("section_title") in ReportGenerationModule._REQUIRED_SECTIONS:
            assert s["content"] == "narrated prose"


@pytest.mark.asyncio
async def test_the_truncation_retry_fires_across_both_module_identities(tmp_path):
    """This test exists because the one above passed while the retry was DEAD in
    production. `src/utils/llm_client.py` is importable as both `utils.llm_client`
    (main.py's flat style, which builds the live client) and `src.utils.llm_client`
    (what this module imports), so there are two `LLMTruncatedError` classes. The old
    `except LLMTruncatedError` caught only the identity this file imported — the one the
    test raised — and every real run's truncation went to the fallback report instead.
    The stage now matches on the `is_truncation` marker, so pin BOTH identities.
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
            return InvestigationReport(
                sections=[
                    {"section_title": t, "content": "narrated prose"}
                    for t in ReportGenerationModule._REQUIRED_SECTIONS
                ]
            )

        llm = MagicMock()
        llm.max_tokens = 4096
        llm.structured_output = AsyncMock(side_effect=fake_structured_output)
        module = ReportGenerationModule(
            {
                "output_format": "txt",
                "output_path": str(tmp_path),
                "report_min_tokens": 4000,
                "report_max_tokens": 8000,
            },
            llm,
        )

        await module.generate(
            {"id": "INC-1"},
            _understanding(),
            {"raw_access": [{"x": 1}]},
            [],
            correlation=None,
        )

        assert len(budgets) == 2, f"no retry for {module_under_test.__name__}"
        assert module.last_fallback_used is False
        assert module.last_truncation_retried is True


@pytest.mark.asyncio
async def test_a_second_truncation_falls_back_rather_than_escalating(tmp_path):
    """Two overruns means the prompt is the problem; the report must still be produced."""
    from src.utils.llm_client import LLMTruncatedError

    config = {
        "output_format": "txt",
        "output_path": str(tmp_path),
        "report_min_tokens": 4000,
        "report_max_tokens": 8000,
    }
    llm = MagicMock()
    llm.max_tokens = 4096
    calls = {"n": 0}

    async def always_truncates(
        messages, response_model, max_tokens=None, rag=None, stage=None
    ):
        calls["n"] += 1
        raise LLMTruncatedError("truncated again")

    llm.structured_output = AsyncMock(side_effect=always_truncates)
    module = ReportGenerationModule(config, llm)

    text = await module.generate(
        {"id": "INC-1"},
        _understanding(),
        {"raw_access": [{"x": 1}]},
        [_anomaly("dup issuance", 0.93)],
        correlation=None,
    )

    assert calls["n"] == 2  # one retry, then stop escalating
    assert module.last_fallback_used is True
    # The acceptance artifact still exists and still carries the findings.
    sections = json.loads(text)
    titles = " ".join(s.get("section_title", "") for s in sections).lower()
    assert "analysis and findings" in titles


def test_anomaly_detection_budget_exceeds_the_client_default():
    """AnomalyList carries six prose fields per item; 4096 tokens truncates it."""
    import sys

    sys.path.insert(0, "src")
    from src.anomaly_detection import AnomalyDetectionModule

    llm = MagicMock()
    llm.max_tokens = 4096
    module = AnomalyDetectionModule({"threshold": 0.8}, llm)
    assert module._detect_max_tokens() > llm.max_tokens
    # Deployment-overridable.
    module = AnomalyDetectionModule({"threshold": 0.8, "detect_max_tokens": 12000}, llm)
    assert module._detect_max_tokens() == 12000


@pytest.mark.asyncio
async def test_generate_writes_markdown_and_pdf_artifacts(tmp_path, monkeypatch):
    """generate() writes fraud_report_<id>.md and .pdf into the exports dir alongside
    the machine-facing JSON, best-effort, keyed by incident id."""
    import src.report_generation as rg

    monkeypatch.setattr(rg, "exports_dir", lambda: tmp_path)
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(
        return_value=InvestigationReport(
            sections=[{"section_title": "Executive Summary", "content": "All clear."}]
        )
    )
    module = ReportGenerationModule(config, llm)

    await module.generate(
        {"id": "INC-2"},
        _understanding(),
        {"src": []},
        [_anomaly("a", 0.5)],
        correlation=None,
    )
    md_path = tmp_path / "fraud_report_INC-2.md"
    pdf_path = tmp_path / "fraud_report_INC-2.pdf"
    assert md_path.exists()
    assert "## Executive Summary" in md_path.read_text()
    assert "All clear." in md_path.read_text()
    assert pdf_path.exists() and pdf_path.stat().st_size > 0


# --- SCHEME verdict section ----------------------------------------------------

from src.models.pydantic_models import (STUB_OBSERVED, ConditionCheck,
                                        CorrelationResult, SourceEvidence,
                                        SubjectVerdict, ValidationVerdict)


def _verdict_correlation(verdict_label="VALID FRAUD"):
    verdict = ValidationVerdict(
        label_scheme="scheme",
        summary=f"{verdict_label}: 1",
        degraded=True,
        subjects=[
            SubjectVerdict(
                subject_type="record",
                subject_value="SUBJ03",
                verdict=verdict_label,
                checks=[
                    ConditionCheck(
                        id="bare",
                        label="Bare record",
                        result="pass",
                        observed="0",
                        decisive=True,
                    ),
                    ConditionCheck(
                        id="not_automated",
                        label="Not automated",
                        result="unknown",
                        detail="flag absent",
                    ),
                ],
                lock_target={
                    "scope": "ORG2428D4",
                    "identity": "0201GPSU",
                    "source": "record_lake",
                    "scope_field": "creator.org_unit_id",
                    "identity_field": "creator.sign.red",
                    # The role words as the verdict engine stamps them, from the pack's
                    # `lock_target.scope_label`/`identity_label`. A domain declaring nothing gets
                    # the generic "scope"/"identity", asserted separately.
                    "scope_label": "OrgUnit",
                    "identity_label": "Sign",
                    "platform": "classic",
                    "action": "LOCK the Sell Classic agent account.",
                    "prerequisites": "BLOCKING PREREQUISITE — contact the customer first.",
                },
                notes=[
                    "platform_mode=Sell Classic (ATID 58ABC)",
                    "asset_count=1",
                    # The pack's own noun for the asset, emitted BESIDE the stable count key
                    # rather than as the key itself — so the report prints "Documents impacted"
                    # without the engine knowing the word.
                    "asset_label=document",
                    "route=LBV-CMN",
                ],
            )
        ],
        notification_draft="Dear SMC,\nOutcome for record SUBJ03: VALID FRAUD.",
    )
    return CorrelationResult(verdict=verdict)


def test_verdict_section_leads_with_the_outcome_and_its_ground():
    """A reader's first two questions are "what" and "on what evidence".

    The verdict section answers both in its first two lines and then stops — the per-
    condition table lives in its own section now, because a nineteen-row table printed
    under the verdict buried the one row that decided the case.
    """
    sec = ReportGenerationModule._verdict_section(_verdict_correlation())
    assert sec is not None
    assert sec["section_title"] == "SCHEME Verdict"
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-1"})
    assert "VERDICT: VALID FRAUD" in md
    # No exclusion fired here, so the ground is the fingerprint: every decisive check passed.
    assert "DECISIVE CONDITION(S)" in md
    assert "Bare record" in md
    # Contextual facts about the incident stay; the ACTION does not — it is gated on the
    # verdict and lives in the Actions section, so it cannot be read out of context here.
    assert "Documents impacted: 1" in md and "In-scope route: LBV-CMN" in md
    assert "CONTAINMENT TARGET" not in md and "ACTION (§4.1.2)" not in md
    assert "NOT SENT by AFIR" not in md


def test_verdict_section_names_the_decisive_exclusion_from_the_engines_own_note():
    """The named ground is READ OFF the engine's record, never re-derived here.

    `decisive_exclusion=` is the verdict engine's own note, already ordered by the pack's
    evidential ranking. Re-scanning the checks in the report would be a second, drifting
    copy of that ranking — and the verdict block and the validation table could then
    disagree about what settled the case.
    """
    corr = _verdict_correlation()
    s = corr.verdict.subjects[0]
    s.verdict = "NOT A FRAUD"
    s.notes = s.notes + ["decisive_exclusion=No record split (SP) (xref.sp×11)"]
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._verdict_section(corr)], {"id": "INC-1"}
    )
    assert "VERDICT: NOT A FRAUD" in md
    assert "DECISIVE CONDITION(S): No record split (SP) (xref.sp×11)" in md


def test_verdict_section_names_unevaluated_conditions_when_nothing_could_be_checked():
    """A verdict with no stated ground cannot be audited.

    On INSUFFICIENT DATA there is no exclusion to name, so the section must say which
    decisive condition could not be answered rather than leaving the ground blank.
    """
    corr = _verdict_correlation()
    s = corr.verdict.subjects[0]
    s.verdict = "INSUFFICIENT DATA"
    s.checks[0].result = "unknown"
    s.checks[1].decisive = True
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._verdict_section(corr)], {"id": "INC-1"}
    )
    assert "none could be evaluated" in md
    assert "Bare record" in md and "Not automated" in md


def test_actions_section_renders_the_containment_target_and_its_resolved_action():
    """With NO pack, the section is complete and generic — no clause numbers, no domain."""
    sec = ReportGenerationModule._actions_section(_verdict_correlation())
    assert sec is not None
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-1"})
    # The heading names the TARGET, not a verb: the verb is platform-specific (Connect is
    # FROZEN, not locked; an automated sign is escalated instead), so the engine resolves it
    # and this section only renders it.
    assert "CONTAINMENT TARGET" in md
    assert "orgunit=ORG2428D4" in md and "0201GPSU" in md  # the CREATOR
    # ...with the fields it was read from, so the instruction can be checked against the
    # record it names.
    assert "creator.org_unit_id" in md and "creator.sign.red" in md
    # The blocking prerequisite is rendered BEFORE the action it blocks — a precondition
    # printed after an instruction is one a reader acts past.
    assert "BLOCKING PREREQUISITE" in md
    assert md.index("BLOCKING PREREQUISITE") < md.index("ACTION: LOCK")
    assert "ACTION: LOCK the Sell Classic" in md
    # The selling platform belongs with the action set it selects, not with the verdict.
    assert "Platform:" in md and "Sell Classic" in md
    # The draft rides verbatim, under a heading that says AFIR did not send it: a section
    # headed only "Notification draft" reads as correspondence someone is expected to send.
    assert "NOT SENT by AFIR" in md
    assert "Outcome for record SUBJ03" in md
    # This subject DOES carry a lock target, so it is an escalation draft, not a closure one.
    assert "case-closure" not in md
    # And with no pack, the domain's own vocabulary is nowhere in it: a clause citation
    # emitted by the ENGINE would be asserted on every use case running through it.
    assert "§" not in md


def test_the_no_verb_sentence_states_the_reason_that_applies_and_no_other():
    """Three shapes reach "no verb was resolved", and the engine may only assert the true one.

    * A ruleset that declares `platform_mode` and could not determine it — the historical
      sentence, unchanged: the action sets are not interchangeable and a human closes the gap.
    * A ruleset that declares no platform dimension — `lock_target` carries no `platform` key at
      all, so the platform sentence would be an engine-invented unknown about a dimension the
      procedure has not got, and no `Platform:` line may print either.
    * A ruleset whose declared platform RESOLVED to a class its `actions:` map does not cover —
      this printed a containment target with no verb and no sentence at all, because the branch
      tested the platform rather than the missing verb.

    One slot in all three: the pack overrides "no verb was resolved", not "no platform". Every
    ruleset in this repo's domain pack that declares `lock_target` overrides it, which is the
    measurement behind keeping it one slot.
    """
    corr = _verdict_correlation()
    lt = corr.verdict.subjects[0].lock_target

    # 1. Declared and undetermined — the gap is real and the wording is the historical one.
    lt["platform"] = "unknown"
    lt.pop("action")
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._actions_section(corr)], {"id": "INC-1"}
    )
    assert "ACTION: NOT DETERMINED — the platform this identity sells on" in md
    assert "CONTAINMENT TARGET" in md

    # 2. No dimension declared — the reason is the procedure, not the evidence, and the
    # platform is not mentioned in either direction.
    lt.pop("platform")
    corr.verdict.subjects[0].notes = [
        n for n in corr.verdict.subjects[0].notes if not n.startswith("platform_mode=")
    ]
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._actions_section(corr)], {"id": "INC-1"}
    )
    assert "ACTION: NOT DETERMINED — this procedure resolves no containment action" in md
    assert "platform" not in md and "Platform" not in md

    # 3. Declared, resolved, and unmapped — a target with no verb still says why.
    lt["platform"] = "some_class_the_map_omits"
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._actions_section(corr)], {"id": "INC-1"}
    )
    assert "ACTION: NOT DETERMINED" in md


def test_the_packs_procedure_wording_replaces_the_engines_generic_sentence():
    """A clause number is a fact about ONE procedure, so the pack supplies it, not `src/`.

    The engine writes a complete generic sentence for each slot and asks the pack whether
    the domain has a better one. Both halves are load-bearing and asserted here: the pack's
    exact wording (with its clause citations) reaches the artifact, and the placeholder is
    filled with the value the engine resolved. Substitution is `str.replace`, never
    `.format`, because procedure prose contains braces.
    """
    from src.knowledge.pack import KnowledgePack

    pack = KnowledgePack(
        reporting={
            "use_cases": {
                "scheme": {
                    "phrases": {
                        "containment_target": "CONTAINMENT TARGET (§4.1.1) — the record "
                        "CREATOR, not the issuance agent: {target}",
                        "containment_action": "ACTION (§4.1.2): {action}",
                        "selling_platform": "Selling platform (§4.1.2): {platform}",
                    }
                }
            }
        }
    )
    module = ReportGenerationModule({}, MagicMock(), knowledge_pack=pack)
    corr = _verdict_correlation()
    corr.brief = InvestigationBrief(use_case="scheme")
    phrases = module._report_phrases(module._use_case_of(corr))
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._actions_section(corr, phrases)], {"id": "INC-1"}
    )
    assert (
        "CONTAINMENT TARGET (§4.1.1) — the record CREATOR, not the issuance agent" in md
    )
    assert "orgunit=ORG2428D4" in md  # the placeholder was substituted, not printed
    assert "{target}" not in md
    assert "ACTION (§4.1.2): LOCK the Sell Classic" in md
    assert "Selling platform (§4.1.2)" in md
    # A slot the pack does NOT declare still renders — the engine's default stands alone.
    assert "NOT SENT by AFIR" in md


def test_a_pack_scopes_its_procedure_wording_to_one_use_case():
    """Hoisting a clause number to the domain root would cite it in every use case's report.

    `reporting.yaml` resolves most-specific-first: the use case's own entry wins per slot,
    the flat root is the domain-wide base, and a use case the pack knows nothing about gets
    the root (or, failing that, the engine's generic sentence).
    """
    from src.knowledge.pack import KnowledgePack

    pack = KnowledgePack(
        reporting={
            "phrases": {"containment_target": "Domain default — {target}"},
            "use_cases": {
                "scheme": {
                    "phrases": {"containment_target": "SCHEME §4.1.1 — {target}"}
                }
            },
        }
    )
    module = ReportGenerationModule({}, MagicMock(), knowledge_pack=pack)
    assert "SCHEME §4.1.1" in module._report_phrases("scheme")["containment_target"]
    # A different use case must NOT inherit SCHEME's clause number.
    assert module._report_phrases("rewards")["containment_target"].startswith(
        "Domain default"
    )
    # No pack at all: no phrases, and every call site falls back to its own default.
    assert ReportGenerationModule({}, MagicMock())._report_phrases("scheme") == {}


def test_the_chronologys_phase_labels_come_from_the_pack():
    """ "record activity" is a travel-distribution term, not a generic phase name.

    It used to be hard-coded in `_PHASE_RULES` alongside "Issuance", so an ATO or rewards
    investigation running through the same engine got labels that describe neither. The pack
    declares its phases; the engine keeps three vocabulary-free ones as the fallback so a
    pack-less run still gets a segmented chronology rather than one flat list.
    """
    from src.knowledge.pack import KnowledgePack

    pack = KnowledgePack(
        reporting={
            "use_cases": {
                "scheme": {
                    "phases": [
                        {"keywords": ["app", "document"], "label": "Document issuance"},
                        {
                            "keywords": ["record"],
                            "label": "record creation and amendment",
                        },
                    ]
                }
            }
        }
    )
    module = ReportGenerationModule({}, MagicMock(), knowledge_pack=pack)
    rules = module._phase_rules("scheme")
    assert ReportGenerationModule._phase_for("app_session_events", "issue", rules) == (
        "Document issuance"
    )
    assert ReportGenerationModule._phase_for("record_lake", "create", rules) == (
        "record creation and amendment"
    )
    # No pack: generic labels only, and no domain vocabulary anywhere in them.
    generic = ReportGenerationModule({}, MagicMock())._phase_rules("scheme")
    labels = [label for _, label in generic]
    assert "Authentication and session activity" in labels
    assert not any("record" in lbl or "Document" in lbl for lbl in labels)


def test_a_verdict_that_nominated_nobody_states_that_no_action_is_authorised():
    """A cleared verdict's Actions section must exist and must say "nothing".

    The engine withholds `lock_target` on any verdict outside the pack's
    `containment_labels`, so an empty target IS the record of that decision — the report
    reads it off there rather than re-deriving the policy from label strings. The section
    is still emitted: an absent Actions section reads as an omission, while one that says
    no action is authorised cannot be misread.
    """
    corr = _verdict_correlation()
    corr.verdict.subjects[0].verdict = "NOT A FRAUD"
    corr.verdict.subjects[0].lock_target = {}
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._actions_section(corr)], {"id": "INC-1"}
    )
    assert "NO ACTION IS AUTHORISED" in md
    assert "no containment" in md.lower() and "no external" in md.lower()
    assert "CONTAINMENT TARGET" not in md and "LOCK the Sell Classic" not in md
    assert "case-closure note" in md
    assert "no containment or escalation is recommended" in md.lower()


def _alert_facts_brief(**over):
    """A brief carrying located alert facts + a scope sweep, as the analyzer builds them."""
    from src.models.pydantic_models import (AlertFacts, DeclaredFact,
                                            ImpactedAsset, InvestigationBrief,
                                            SubjectLink)

    facts = AlertFacts(
        located=True,
        locator="matched on alert_id=scheme-alert-46127cea, org_unit=ORG262206",
        source="siem_alerts_current",
        record_id="0c08ca2f / 10000003",
        declared_facts=[
            DeclaredFact(
                field="org_unit",
                declared="ORG262206",
                found="ORG262206",
                status="confirmed",
                source="record_lake",
            ),
            DeclaredFact(
                field="user",
                declared="0303GH",
                found="0303GHSU",
                status="confirmed",
                source="record_lake",
            ),
            DeclaredFact(
                field="document", declared="100-2000002004", status="not_found"
            ),
            DeclaredFact(
                field="document", declared="100-2000002005", status="not_found"
            ),
            DeclaredFact(
                field="record",
                declared="SUBJ04",
                found="SUBJ07",
                status="mismatch",
                source="record_lake",
            ),
            DeclaredFact(
                field="alert_id", declared="scheme-alert-46127cea", status="stated"
            ),
        ],
        unrelated_records=["07b5ba6e / 38870257"],
        # WHAT THE DETECTOR FIRES ON, as the analyzer renders the ruleset's `trigger:` block.
        # Verbatim from the SCHEME pack, negative included: "NOT payment-based" is the half that
        # stops a reader treating the absence of a payment element as the allegation.
        trigger=(
            "A SCHEME alert fires on a BURST OF ISSUANCE: at least 4 suspicious documents "
            "issued within 24 hours in a single org_unit. It is NOT payment-based. "
            "(threshold: at least 4 subject(s), within 24h, in one org_unit)"
        ),
    )
    return InvestigationBrief(
        use_case="scheme",
        alert_facts=facts,
        scope_status="ran, 9 asset(s) across 4 subject(s); 1 of those (SUBJ07) is NOT new "
        "scope — derived from an alerted subject by a split recorded in the data",
        additional_subjects=["SUBJ07", "YAB1CD"],
        subject_links=[
            SubjectLink(
                subject="SUBJ04",
                related_subject="SUBJ07",
                role="parent",
                kind="split",
                quote="SP 09JUL/GGSU/ORG262206-SUBJ07",
                element_id="0-record-SP-64",
            )
        ],
        impacted_assets=[
            ImpactedAsset(
                subject="SUBJ04",
                asset_id="100-2000002004",
                amount="674200",
                currency="XAF",
                status="T",
                event_date="2026-07-09",
                actor="0303GH @ ORG262206",
                in_window=True,
            ),
            # No currency returned: printing the bare number would read as the reader's own.
            ImpactedAsset(
                subject="SUBJ07",
                asset_id="500-2000002008",
                amount="589300",
                status="V",
                event_date="2026-07-09",
                in_window=True,
            ),
            ImpactedAsset(
                subject="YAB1CD",
                asset_id="100-2000002003",
                amount="1",
                currency="XAF",
                event_date="2026-06-02",
                in_window=False,
            ),
        ],
        **over,
    )


def test_alert_facts_section_reproduces_the_payload_and_names_the_record_it_read():
    """The alert is the one input the pipeline did not derive, so it is quoted, not summarised.

    Two things this must always say, because their absence is what let earlier reports go
    wrong: WHICH record was read (the source returned two incidents' records) and which
    records belong to a different incident. Multi-valued facts stay one fact with N values —
    the alert listed four documents and a report narrated one "primary document".
    """
    sec = ReportGenerationModule._alert_facts_section(_grouped_with_facts())
    assert sec is not None
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-A"})
    assert "Alert record READ: 0c08ca2f / 10000003" in md
    assert "siem_alerts_current" in md
    assert "document: 100-2000002004, 100-2000002005" in md
    assert "DIFFERENT incident" in md and "07b5ba6e / 38870257" in md


def test_the_unrelated_records_sentence_promises_only_what_is_enforced():
    """It used to promise more: "no value of theirs appears anywhere in this report".

    Nothing enforced that, and on job ca4240c0 the same report went on to name both foreign
    record locators, their tickets and their signs — in the chronology, in the evidence, and
    (before `_incident_alert_rows`) in two full verdicts with authorised actions. The verdict
    scope is now enforced; the chronology deliberately still reads every retrieved row,
    because an unrelated alert in the same window is legitimate background. So the sentence
    promises the thing that is kept — no finding — and not the thing that is not.
    """
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._alert_facts_section(_grouped_with_facts())], {"id": "INC-A"}
    )
    assert "no value of theirs appears anywhere in this report" not in md
    assert "no verdict, finding or action below concerns them" in md


def test_the_report_states_the_real_trigger_and_the_prompt_forbids_substituting_one():
    """Both surfaces, because the wrong trigger was stated on the NARRATED one.

    On 83e94dd6 the narration said the alert fired on the ABSENCE of a payment element and
    built the case around it — for a detector that never looks at payment, on a record that did
    carry one. Nothing in the brief contradicted it because nothing in the brief said what the
    trigger WAS, so the model filled the gap with the most suspicious retrieved fact, which is
    the reasonable thing to do given that prompt.

    So the prompt states it as ground truth AND forbids substitution: the failure is not the
    model missing a fact, it is the model filling a gap. And the section prints it BEFORE the
    declared facts, since everything below is only meaningful against the allegation.
    """
    corr = _grouped_with_facts()
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._alert_facts_section(corr)], {"id": "INC-A"}
    )
    assert "WHAT THE DETECTOR FIRES ON:" in md
    assert "NOT payment-based" in md
    # Ordered: the trigger frames the facts, so it cannot come after them.
    assert md.index("WHAT THE DETECTOR FIRES ON") < md.index("org_unit: ORG262206")

    prompt = ReportGenerationModule._render_brief_for_prompt(corr.brief)
    assert "NOT payment-based" in prompt
    assert "Do NOT state or imply any other trigger" in prompt


def test_a_brief_with_no_declared_trigger_says_nothing_about_one():
    """An engine sentence about triggers in general would read as one having been established.

    So there is deliberately no fallback: an undeclared trigger removes the line from both the
    section and the prompt rather than replacing it with a generic hedge.
    """
    corr = _grouped_with_facts()
    corr.brief.alert_facts.trigger = ""
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._alert_facts_section(corr)], {"id": "INC-A"}
    )
    assert "DETECTOR FIRES ON" not in md
    assert (
        "trigger"
        not in ReportGenerationModule._render_brief_for_prompt(corr.brief).lower()
    )


def test_alert_facts_section_says_so_when_the_alert_record_was_never_read():
    """ "The alert says nothing" and "we never read the alert" are different statements."""
    corr = _grouped_with_facts()
    corr.brief.alert_facts.located = False
    corr.brief.alert_facts.locator = "NOT located: none of the 2 record(s) carry them"
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._alert_facts_section(corr)], {"id": "INC-A"}
    )
    assert "ALERT RECORD NOT READ" in md
    assert "none of it was confirmed at source" in md


def test_a_ruleset_that_declares_no_alert_record_reports_no_retrieval_failure():
    """"We did not look" is not "we looked and it was not there", and only one is a defect.

    A ruleset may declare a `trigger:` and no `alert_record:`; `build_alert_facts` then returns
    the allegation alone — `located=False` with an EMPTY locator, because no path attempted a
    location and so none wrote one. The section used to read the falsy `located` as a failure
    and print "ALERT RECORD NOT READ — the incident's own alert record was not retrieved" over
    a run whose alert source had returned its one row and whose every condition was evaluated
    against it (job a25ad11c). The locator is the discriminator: attempted-and-failed always
    sets one, which is why the test above still passes unchanged.

    The facts preamble goes with it. "These are the facts the ORIGINATING ALERT states … they
    are reproduced here verbatim" above zero facts describes some other report.
    """
    corr = _grouped_with_facts()
    corr.brief.alert_facts.located = False
    corr.brief.alert_facts.locator = ""
    corr.brief.alert_facts.record_id = ""
    corr.brief.alert_facts.source = ""
    corr.brief.alert_facts.declared_facts = []
    corr.brief.alert_facts.unrelated_records = []
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._alert_facts_section(corr)], {"id": "INC-A"}
    )
    assert "ALERT RECORD NOT READ" not in md
    assert "reproduced here" not in md
    # The allegation survives: it is a property of the DETECTOR, not of this run's retrieval.
    assert "WHAT THE DETECTOR FIRES ON" in md and "BURST OF ISSUANCE" in md


def test_the_alert_facts_section_is_absent_rather_than_a_bare_heading():
    """With no facts, no trigger and no location attempted there is nothing to say.

    Distinct from the case above, which still has the trigger. A section object here renders
    as a mandated heading with no content under it, which reads as content gone missing.
    """
    corr = _grouped_with_facts()
    corr.brief.alert_facts.located = False
    corr.brief.alert_facts.locator = ""
    corr.brief.alert_facts.trigger = ""
    corr.brief.alert_facts.declared_facts = []
    corr.brief.alert_facts.unrelated_records = []
    assert ReportGenerationModule._alert_facts_section(corr) is None


def test_reconciliation_keeps_all_four_outcomes_distinct():
    """A mismatch is a retrieval defect; a gap is silence; `stated` is not corroborable.

    Conflating the last two reports a retrieval gap on a fact nothing was ever going to
    confirm (the alert's own id), which sends the reader hunting for a source that was never
    declared. And a mismatch must never read as a correction TO the alert.
    """
    sec = ReportGenerationModule._reconciliation_section(_grouped_with_facts())
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-A"})
    assert "[CONFIRMED] user: declared '0303GH' — found '0303GHSU'" in md
    assert "[MISMATCH] record" in md and "carries 'SUBJ07'" in md
    assert "[NOT FOUND IN DATA] document" in md
    assert "no corroborating source declared" in md and "alert_id" in md
    # The delta is called out as a defect to investigate, not a licence to overwrite.
    assert "CONTRADICTED by the data" in md
    assert "The alert is authoritative" in md


def test_a_qualified_gap_prints_its_sample_AND_the_reason_it_settles_nothing():
    """A NOT FOUND that saw other values must read as neither a mismatch nor pure silence.

    The analyzer downgrades a disagreement to a gap when the source was truncated at its row
    cap or its query never named this subject (see `test_usecases.py`). Printed as a bare NOT
    FOUND it loses the sample the reader needs to recognise a stale or wrong-record read;
    printed with the sample and no reason it is a MISMATCH mislabelled, and the delta
    paragraph — the one that calls a mismatch a defect to investigate — must NOT claim it.
    """
    corr = _grouped_with_facts()
    from src.models.pydantic_models import DeclaredFact

    corr.brief.alert_facts.declared_facts = [
        DeclaredFact(
            field="org_unit",
            declared="ORG262206",
            found="ORG100001, ORG100002",
            status="not_found",
            source="event_log",
            note=(
                "event_log was TRUNCATED at its 500-row cap, so the values it carries are "
                "a sample and not its answer"
            ),
        )
    ]
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._reconciliation_section(corr)], {"id": "INC-A"}
    )
    assert "[NOT FOUND IN DATA] org_unit: declared 'ORG262206'" in md
    assert "event_log carries 'ORG100001, ORG100002', which does not settle it" in md
    assert "TRUNCATED at its 500-row cap" in md
    assert "CONTRADICTED by the data" not in md


def test_a_source_missing_the_alerts_own_subject_is_named_in_the_gaps_section():
    """The one gap no row count can show, printed where the reader acts on gaps.

    A source that answered rows none of which carry the alert's declared subject is either a
    stale read, the wrong record or an unscoped query — and every finding derived from it is
    suspect. The reconciliation table already computed it, but that table sits two headings
    from the conditions section that read the same source as authoritative. Keyed per SOURCE,
    because the remedy (re-retrieve, re-scope) is per source, and a source with no name
    attached cannot be actioned so it must not appear.
    """
    corr = _grouped_with_facts()
    corr.evidence = MagicMock()
    corr.evidence.sources = [SourceEvidence(source="record_lake", record_count=51)]
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._not_retrieved_section(corr)], {"id": "INC-A"}
    )
    line = next(ln for ln in md.split("\n") if "do NOT carry a value the ALERT" in ln)
    assert "record_lake (missing record 'SUBJ04')" in line
    assert "possibly not about the alerted subject" in line
    # The two `document` facts are unreconciled too, but no source was ever named for them —
    # they are already reported as retrieval gaps and there is nothing here to re-retrieve.
    assert "100-2000002004" not in line


def test_a_source_that_carries_one_of_a_declared_set_is_not_named_as_missing_it():
    """The gaps line claims the source carries NO declared value — so it must be true.

    Where the alert declares a set of one kind (several documents, one actor under two
    forms) and the source carries some of them, that source demonstrably IS about the
    alerted subject. Naming it here prints the opposite of the evidence and tells the
    reader to distrust every finding drawn from it. Selected per (source, FIELD), which
    is what the sentence asserts: the same source's unreconciled `record` fact still
    appears, because for that field it really does carry another value — and per source
    alone would have dropped the `6959d7b1` case this block exists for, where the record
    lake confirmed the locator and carried neither declared document.
    """
    from src.models.pydantic_models import DeclaredFact

    corr = _grouped_with_facts()
    corr.brief.alert_facts.declared_facts = list(
        corr.brief.alert_facts.declared_facts
    ) + [
        DeclaredFact(
            field="document",
            declared="100-2000002006",
            found="100-2000002006",
            status="confirmed",
            source="record_lake",
        ),
        DeclaredFact(
            field="document",
            declared="100-2000002007",
            found="100-2000002006",
            status="not_found",
            source="record_lake",
            note="every value record_lake carries for this fact is one the alert ALSO "
            "declares, so it corroborates those and is silent about this one — not a "
            "contradiction",
        ),
    ]
    corr.evidence = MagicMock()
    corr.evidence.sources = [SourceEvidence(source="record_lake", record_count=51)]
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._not_retrieved_section(corr)], {"id": "INC-A"}
    )
    line = next(ln for ln in md.split("\n") if "do NOT carry a value the ALERT" in ln)
    assert "100-2000002007" not in line
    assert "record_lake (missing record 'SUBJ04')" in line


def test_the_cut_note_states_a_remainder_and_stays_silent_when_nothing_was_cut():
    """The one wording every display cut in this report states itself with.

    Silence is the half worth pinning: it is returned on every uncut list in every report,
    so a note that fires at ``total == shown`` would append "and 0 further" to lists that
    are complete — and a reader who has seen that once stops believing the sentence on the
    list that really was cut. The non-numeric guard is not hypothetical either: these lists
    are built from a brief that tests (and a degraded run) hand through as a MagicMock.
    """
    note = ReportGenerationModule._cut_note(45, 40, "later event(s)")
    assert "and 5 further later event(s)" in note
    assert "there are 45 in all" in note
    # The claim the sentence has to make: the cut is the PRINTING, not the reasoning.
    assert "included in the counts and determinations above" in note
    assert ReportGenerationModule._cut_note(40, 40, "x") == ""
    assert ReportGenerationModule._cut_note(3, 40, "x") == ""
    # Nested inside another line's parenthesis: same claim, no second sentence.
    assert ReportGenerationModule._cut_note(7, 4, "", compact=True) == (
        "and 3 more not listed"
    )
    assert ReportGenerationModule._cut_note(None, 4, "x") == ""
    assert ReportGenerationModule._cut_note("many", 4, "x") == ""


def test_every_cut_list_in_the_wider_evidence_section_states_its_remainder():
    """A list that stops is read as a list that ended, and this section holds three.

    All three used to trail off at their bound. The derivation quotes are the EVIDENCE for
    the reclassification above them, the un-derived subjects are a work-list somebody has to
    go and look at, and the asset table is what a reader takes for the exposure — so a silent
    `[:N]` under-states, in that last case, the impact. The determinations are unaffected in
    all three (every link is reclassified, every subject is counted, every asset is exported),
    which is exactly why the sentence has to say so.
    """
    from src.models.pydantic_models import ImpactedAsset, SubjectLink

    corr = _grouped_with_facts()
    brief = corr.brief
    brief.subject_links = [
        SubjectLink(
            subject="SUBJ04",
            related_subject=f"DERV{i:03d}",
            role="parent",
            kind="split",
            quote=f"SP 09JUL/GGSU/ORG262206-DERV{i:03d}",
        )
        for i in range(14)
    ]
    brief.additional_subjects = [f"NEWS{i:03d}" for i in range(26)]
    brief.impacted_assets = [
        ImpactedAsset(
            subject="SUBJ04",
            asset_id=f"100-2000003{i:03d}",
            amount="1000",
            currency="XAF",
            status="T",
            event_date="2026-07-09",
            in_window=True,
        )
        for i in range(47)
    ]
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._wider_evidence_section(corr)], {"id": "INC-A"}
    )
    assert "and 4 further derivation link(s) of the same kind" in md
    assert "and 6 further such subject(s)" in md
    assert "and 7 further asset(s) inside the window" in md
    # Each states the FULL total, not only the remainder: a reader auditing scope needs the
    # number the counts above were computed on.
    for total in (14, 26, 47):
        assert f"there are {total} in all" in md
    # And the cut really is at the declared bound, in both directions.
    assert "DERV009" in md and "DERV010" not in md
    assert "NEWS019" in md and "NEWS020" not in md
    assert "100-2000003039" in md and "100-2000003040" not in md


def test_an_uncut_wider_evidence_section_claims_no_remainder():
    """The silence direction, on the fixture every other test in this group uses.

    Three cut notes in one section is three chances to print "and 0 further" on a complete
    list — which is worse than printing nothing, because it makes the sentence noise on
    every report and therefore unread on the one where it matters.
    """
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._wider_evidence_section(_grouped_with_facts())],
        {"id": "INC-A"},
    )
    assert "not printed here" not in md
    assert "in all" not in md


def test_a_chronology_cut_at_its_bound_says_its_TAIL_is_missing():
    """The slice keeps the FIRST N, so what a cut chronology drops is the response.

    That is the half a reader is most likely to conclude never happened — the containment,
    the reversal, the later actor. Asserted on the count (the note names how many were not
    printed and the total) AND on which events survived, because a note that said the right
    number while the cut happened somewhere else would pass on the number alone.
    """
    from src.models.pydantic_models import ChronologyEvent

    module = ReportGenerationModule({"output_format": "txt"}, MagicMock())
    corr = CorrelationResult()
    corr.evidence = MagicMock()
    corr.evidence.chronology = [
        ChronologyEvent(
            timestamp=f"2026-07-09T{i // 60:02d}:{i % 60:02d}:00Z",
            epoch=float(1_752_000_000 + i * 60),
            source="record_lake",
            actor="0303GH",
            action=f"step_{i:03d}",
            is_subject=True,
        )
        for i in range(46)
    ]
    corr.evidence.cross_source_joins = []
    lines = module._incident_reconstruction(corr, "INC-A")
    joined = "\n".join(lines)
    assert "and 6 further later event(s) in the same chronology" in joined
    assert "there are 46 in all" in joined
    assert "step_039" in joined and "step_040" not in joined
    # An uncut chronology says nothing about a remainder.
    corr.evidence.chronology = corr.evidence.chronology[:5]
    assert "not printed here" not in "\n".join(
        module._incident_reconstruction(corr, "INC-A")
    )


def test_the_asset_impact_tables_state_their_own_cuts():
    """Two lists in one block, cut independently, and the first is an exposure list.

    `_asset_impact_lines` is reached from the reconstruction and from the fallback report, so
    a cut here lands in the artifact on every path — including the one taken when narration
    failed, i.e. exactly when nothing else is going to mention what is missing.
    """
    from src.models.pydantic_models import AssetTimelineEntry, ImpactedAsset

    corr = _brief_correlation()
    corr.brief.impacted_assets = [
        ImpactedAsset(
            subject="SUBJ03",
            asset_id=f"057-50338{i:05d}",
            status="T",
            in_window=True,
            known=True,
        )
        for i in range(64)
    ]
    corr.brief.asset_timeline = [
        AssetTimelineEntry(
            timestamp=f"2026-07-24T20:{i % 60:02d}Z",
            event_type="issued",
            entity_type="document",
            entity_value=f"057-50338{i:05d}",
            actor="0201GP",
        )
        for i in range(43)
    ]
    joined = "\n".join(ReportGenerationModule._asset_impact_lines(corr))
    assert "and 4 further asset(s) touched by the same actor" in joined
    assert "there are 64 in all" in joined
    assert "and 3 further later asset event(s) in the same order" in joined
    assert "there are 43 in all" in joined
    # The bounds are independent: a cut in one list must not be reported for the other.
    corr.brief.asset_timeline = corr.brief.asset_timeline[:2]
    one = "\n".join(ReportGenerationModule._asset_impact_lines(corr))
    assert "there are 64 in all" in one and "in the same order" not in one


def test_a_cut_list_of_other_incidents_records_still_excludes_them_all():
    """The one cut that INVERTS: this list is a disclaimer, not a finding.

    An id that falls off the end is a foreign record the report has quietly stopped
    excluding — and the chronology below will still print its rows as background activity in
    the same window, which is exactly how a reader ends up reading the eleventh as this
    incident's. So the remainder sentence has to repeat the exclusion, not merely count.
    """
    corr = _grouped_with_facts()
    corr.brief.alert_facts.unrelated_records = [f"07b5ba6e / 3887{i:04d}" for i in range(13)]
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._alert_facts_section(corr)], {"id": "INC-A"}
    )
    assert "and 3 further record(s) of other incidents" in md
    assert "equally excluded from every finding" in md
    assert "there are 13 in all" in md
    assert "07b5ba6e / 38870009" in md and "07b5ba6e / 38870010" not in md


def test_a_source_missing_more_declared_values_than_are_printed_says_so_compactly():
    """A cut nested inside another line's parenthesis, where the full sentence would bury it.

    Same claim, shorter: "missing a, b, c, d" otherwise reads as the whole of what the source
    failed to carry, and this line's whole purpose is to tell the reader how far to distrust
    every finding drawn from that source.
    """
    from src.models.pydantic_models import DeclaredFact

    corr = _grouped_with_facts()
    corr.brief.alert_facts.declared_facts = [
        DeclaredFact(
            field="document",
            declared=f"100-2000003{i:03d}",
            status="not_found",
            source="record_lake",
        )
        for i in range(9)
    ]
    corr.evidence = MagicMock()
    corr.evidence.sources = [SourceEvidence(source="record_lake", record_count=51)]
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._not_retrieved_section(corr)], {"id": "INC-A"}
    )
    line = next(ln for ln in md.split("\n") if "do NOT carry a value the ALERT" in ln)
    assert "and 5 more not listed)" in line
    # The compact form is compact: no second sentence inside somebody else's parenthesis.
    assert "The cut is for length only" not in line


def test_wider_evidence_frames_the_sweep_as_evidence_and_not_as_a_finding():
    """§3.4 widens the evidence base; more documents by the same agent are not an anomaly.

    A DERIVED subject (the split child) must attach to its parent rather than read as scope
    the alert missed, and an amount with no currency must not print as a bare number.
    """
    sec = ReportGenerationModule._wider_evidence_section(_grouped_with_facts())
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-A"})
    assert "EVIDENCE GATHERING, not a finding" in md
    assert "SUBJ04 → SUBJ07 (split, the child)" in md
    assert "SP 09JUL/GGSU/ORG262206-SUBJ07" in md
    # SUBJ07 is derived, so only YAB1CD is named as un-derived extra scope. Asserted on
    # that one sentence, not the rest of the section — SUBJ07 legitimately reappears in the
    # asset table below it.
    extra = md.split("not derived from an alerted subject:")[1].split(".")[0]
    assert "YAB1CD" in extra and "SUBJ07" not in extra
    assert "674200 XAF" in md
    assert "589300 (CURRENCY NOT RETURNED)" in md
    # Out-of-window activity is the agent's adjacent business, not this incident's exposure.
    assert "OUTSIDE the incident window" in md and "NOT counted" in md


def test_not_retrieved_section_lists_gaps_and_labels_a_truncated_count_as_a_lower_bound():
    """Unevaluated conditions must be auditable, and a capped count is never a total."""
    corr = _grouped_with_facts()
    corr.evidence = MagicMock()
    corr.evidence.sources = [
        SourceEvidence(source="payment_alerts", record_count=0),
        SourceEvidence(
            source="record_document_scope_sweep", record_count=500, row_limited=True
        ),
    ]
    sec = ReportGenerationModule._not_retrieved_section(corr)
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-A"})
    assert "condition(s) NOT EVALUATED" in md and "Not automated" in md
    assert "Sources that returned NO rows: payment_alerts" in md
    assert "LOWER BOUND" in md and ">= 500 rows" in md


def test_a_condition_with_no_data_path_is_a_different_gap_from_one_with_no_data():
    """The section's whole premise is that gaps needing different remedies are listed apart.

    A `stub` is a condition the procedure requires and the pack declares no data path for.
    It reads identically here — same `unknown`, same heading, same count — so an operator
    chasing coverage spends the re-run on the one row a re-run cannot move. Marked rather than
    dropped, because the procedure does require the check and nobody has wired it; counted
    apart, so the heading stops overstating what this incident lost.
    """
    corr = _grouped_with_facts()
    s = corr.verdict.subjects[0]
    real_gaps = len([c for c in s.checks if c.result == "unknown"])
    assert real_gaps, "the premise: this fixture already has a genuine retrieval gap"
    s.checks.append(
        ConditionCheck(
            id="fare_code_legitimacy",
            label="Fare basis has no legitimate explanation",
            result="unknown",
            observed=STUB_OBSERVED,
            detail="no measurable pattern set exists for this check yet",
        )
    )
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._not_retrieved_section(corr)], {"id": "INC-A"}
    )
    assert f"{real_gaps + 1} condition(s) NOT EVALUATED" in md
    assert f"({real_gaps} for missing data, 1 declared with no data path)" in md
    # Still listed, and the marker sits on the stub's own row rather than the heading.
    assert "Fare basis has no legitimate explanation [NO DATA PATH" in md
    assert "a re-run cannot answer it" in md
    # And the real gap is NOT marked — the direction that would tell an operator not to bother.
    for line in md.splitlines():
        if "NO DATA PATH" in line:
            assert "Fare basis" in line, line


def test_a_verdict_with_no_stub_reads_exactly_as_it_did_before_the_split():
    """The back-compat half: no pack declares a stub by default, and for those the heading
    carries no parenthetical and no row carries a marker."""
    corr = _grouped_with_facts()
    assert any(c.result == "unknown" for c in corr.verdict.subjects[0].checks)
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._not_retrieved_section(corr)], {"id": "INC-A"}
    )
    assert "condition(s) NOT EVALUATED:" in md
    assert "no data path" not in md and "NO DATA PATH" not in md


def test_an_empty_source_whose_emptiness_is_the_answer_is_not_listed_as_a_gap():
    """The gaps section must not contradict the conditions section above it.

    Both read the same emptiness. A source whose purpose is an exclusion lookup ANSWERS by
    being empty — the verdict engine reads that as a PASS — so calling it a gap here puts
    the two sections in direct opposition, and on a live run the narration followed the gap
    line and denied what the condition had found. It is still REPORTED, because "empty and
    that settles it" and "never consulted" both look like silence to a reader.
    """
    corr = _grouped_with_facts()
    corr.evidence = MagicMock()
    corr.evidence.sources = [
        SourceEvidence(source="payment_alerts", record_count=0),
        SourceEvidence(
            source="robot_register",
            record_count=0,
            zero_rows_meaning="no row means the pair is not a registered robot",
        ),
    ]
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._not_retrieved_section(corr)], {"id": "INC-A"}
    )
    gap_line = next(ln for ln in md.split("\n") if "returned NO rows:" in ln)
    assert "payment_alerts" in gap_line and "robot_register" not in gap_line
    assert "EMPTY IS THE ANSWER" in md
    assert "not a registered robot" in md
    assert "answered, not unevaluated" in md


def test_a_carve_out_the_engine_cannot_apply_is_a_gap_no_re_run_can_close():
    """The fifth kind of gap, and the second one that is engineering rather than coverage.

    A procedure names things that must NOT be read as evidence; where the engine cannot apply
    one, the check it touches reads more broadly here than the procedure reads. That is a
    standing limitation of the adjudication, and a live pack declared three of them under a
    comment saying "declared here so the report states them" while nothing read the key — so
    the only place a reader could find them was the YAML. It goes in THIS section, beside the
    `NO DATA PATH` marker, because both need engineering and neither is answerable by a
    better query; and it says so, or an operator spends a re-run on it.

    Render-side only: the analyzer decides WHICH carve-outs are stated (a note fires only where
    the check it names FAILED), so the brief is built by hand here — this test survives a
    mutation of that gate and dies only if the section stops printing what it was handed, which
    is the split its builder-side twin in `test_usecases.py` completes.
    """
    corr = _grouped_with_facts()
    corr.brief.unenforced_carve_outs = [
        "The procedure exempts a split performed by the operating airline's own office and "
        "this engine cannot identify those offices. (applies to: no_split)"
    ]
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._not_retrieved_section(corr)], {"id": "INC-A"}
    )
    assert "Procedure rules this engine does NOT apply" in md
    assert "cannot identify those offices" in md and "applies to: no_split" in md
    # Named as the kind of gap it is: not retrieval, so no re-run closes it.
    assert "no re-run can close one" in md
    # ...and a brief carrying none prints no heading at all.
    plain = _grouped_with_facts()
    assert "does NOT apply" not in ReportGenerationModule.render_markdown(
        [ReportGenerationModule._not_retrieved_section(plain)], {"id": "INC-A"}
    )


def test_deterministic_sections_are_absent_rather_than_empty_without_their_input():
    """A non-SCHEME incident's report is exactly as it was: no empty mandated headings."""
    plain = CorrelationResult()
    for builder in (
        ReportGenerationModule._alert_facts_section,
        ReportGenerationModule._reconciliation_section,
        ReportGenerationModule._conditions_section,
        ReportGenerationModule._wider_evidence_section,
        ReportGenerationModule._not_retrieved_section,
        ReportGenerationModule._actions_section,
    ):
        assert builder(plain) is None, builder.__name__
        assert builder(None) is None, builder.__name__


def _grouped_with_facts():
    corr = _grouped_correlation()
    corr.brief = _alert_facts_brief()
    return corr


def _grouped_correlation():
    """A verdict whose ruleset declares the procedure's own reporting groups."""
    from src.models.pydantic_models import ConditionGroup

    corr = _verdict_correlation()
    corr.verdict.condition_groups = [
        ConditionGroup(
            id="scope_gate", title="Scope gate — does this apply?", role="gate"
        ),
        ConditionGroup(
            id="validation", title="Validation steps, in order", role="validation"
        ),
        ConditionGroup(id="hints", title="Disambiguation hints", role="hint"),
    ]
    s = corr.verdict.subjects[0]
    s.checks[0].group = "validation"
    s.checks[1].group = "hints"
    s.checks.append(
        ConditionCheck(
            id="route_in_scope",
            label="Route O&D is in scope",
            result="pass",
            observed="DSS-CDG",
            group="scope_gate",
        )
    )
    s.notes = s.notes + ["scope_gate=the procedure applies to this order"]
    return corr


def test_the_gate_the_steps_and_the_hints_are_grouped_under_one_subject_heading():
    """The procedure's grouping survives, but a subject is presented ONCE.

    Two things must hold at the same time and used to require three sections to get one of
    them. (a) The groups stay DISTINCT: printed flat, an exculpatory hint that decided a
    case is indistinguishable from a validation step that merely ran (what happened on
    83e94dd6), and a gate answers IN SCOPE / OUT OF SCOPE — never PASS, because out of scope
    is not a order examined and cleared. (b) The subject appears once: three top-level
    sections each opened their own `### RECORD SUBJ03`, so answering "how did this subject do?"
    meant reassembling three partial tables pages apart.

    Group ORDER is by role — gate, then validation, then hint — which is also the order the
    procedure states them in, regardless of how the pack happens to list them.
    """
    corr = _grouped_correlation()
    sec = ReportGenerationModule._conditions_section(corr)
    assert sec is not None
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-G"})

    # One subject heading, not three.
    assert md.count("### RECORD SUBJ03") == 1
    # The gate's own vocabulary, and the engine's note about what it means.
    assert "[IN SCOPE] Route O&D is in scope" in md and "DSS-CDG" in md
    assert "the procedure applies to this order" in md
    # Every group keeps its pack-declared heading, tagged with what its role means.
    assert "Scope gate — does this apply?" in md
    assert "Validation steps, in order" in md and "Bare record" in md
    assert "Disambiguation hints" in md and "Not automated" in md
    assert "VALIDATION — mandatory steps" in md
    assert "SIGNALS — optional" in md
    # ...and they stay in role order: gate before validation before hint.
    assert (
        md.index("Scope gate")
        < md.index("Validation steps")
        < md.index("Disambiguation hints")
    )


def test_a_group_id_cannot_change_a_verdict():
    """`report_group` is presentational: mis-grouping a condition must not move the rollup.

    This is what makes it safe to encode the procedure's sectioning in the pack — weighing
    stays entirely in decisive/polarity/exclusion_kind/order.
    """
    corr = _grouped_correlation()
    before = corr.verdict.subjects[0].verdict
    for c in corr.verdict.subjects[0].checks:
        c.group = "hints"  # everything mis-grouped as an optional hint
    sec = ReportGenerationModule._verdict_section(corr)
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-G"})
    assert corr.verdict.subjects[0].verdict == before
    assert f"VERDICT: {before}" in md


def test_a_check_filed_under_no_declared_group_is_still_printed():
    """A check the verdict counted may not be ABSENT from the report.

    The buckets were one per declared group and nothing else, so a `report_group` matching no
    declared id vanished from the conditions table while the rollup above went on counting it.
    Both ways to match nothing are exercised here because they arrive differently and the
    pack's remedy differs — a stale or mistyped name, and the blank the key defaults to when a
    condition is appended after the `condition_groups:` block was written — and neither is
    hypothetical: `pack_validate` warns on both precisely because both are authored by hand.

    The consequence is asserted rather than the mechanism: a DECISIVE FAIL is the worst case,
    since the summary says an exclusion fired and the reader cannot find which one, so both
    orphans here are decisive fails and the assertion is that their labels reach the markdown.
    The declared group must also be unaffected — a fallback that swept every check into it
    would pass a weaker version of this test.
    """
    corr = _grouped_correlation()
    s = corr.verdict.subjects[0]
    s.checks.append(
        ConditionCheck(
            id="stale_name",
            label="Filed under a group since deleted",
            result="fail",
            observed="AGENT01",
            decisive=True,
            group="group_that_no_longer_exists",
        )
    )
    s.checks.append(
        ConditionCheck(
            id="never_filed",
            label="Filed under nothing at all",
            result="fail",
            observed="AGENT02",
            decisive=True,
        )
    )
    sec = ReportGenerationModule._conditions_section(corr)
    assert sec is not None
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-G"})

    assert "Filed under a group since deleted" in md
    assert "Filed under nothing at all" in md
    assert "AGENT01" in md and "AGENT02" in md
    # Under a heading that names the defect, and marked decisive like any other check.
    assert "Checks this ruleset filed under no declared group" in md
    assert md.count("(decisive)") >= 2
    # The declared groups still hold exactly what they held before.
    assert "Validation steps, in order" in md and "Bare record" in md
    assert md.index("Validation steps") < md.index(
        "Checks this ruleset filed under no declared group"
    )
    # One subject heading still, not one per bucket.
    assert md.count("### RECORD SUBJ03") == 1
    # And every check exactly once. "Collect everything" also prints the orphans and passes the
    # assertions above, while duplicating each grouped check under a heading calling the ruleset
    # defective — a second reading of the same evidence, with no way to tell which is meant.
    for check in s.checks:
        assert md.count(check.label) == 1, check.label


def test_an_ungrouped_ruleset_gains_no_orphan_heading():
    """The fallback for a pack that declares no groups is the FLAT table, not the orphan one.

    Every check in such a ruleset matches no declared group — there are none — so a fix
    written as "collect what matched nothing" would file the entire table under a heading
    telling the reader the ruleset is defective, on every pack that never adopted grouping.
    """
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._conditions_section(_verdict_correlation())],
        {"id": "INC-F"},
    )
    assert "filed under no declared group" not in md
    assert "Bare record" in md


def test_condition_assessment_marks_an_unevaluated_condition_as_not_evaluated():
    """`unknown` is NOT a pass, and the row must carry the actual evidence value.

    An unretrieved condition reported as anything other than NOT EVALUATED is how a report
    claims a step ran that never did.
    """
    sec = ReportGenerationModule._conditions_section(_verdict_correlation())
    assert sec is not None
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-1"})
    assert "[PASS] Bare record (decisive)" in md
    assert "evidence: 0" in md
    assert "[NOT EVALUATED] Not automated" in md
    assert "flag absent" in md
    # A ruleset declaring no groups keeps its single flat table rather than losing rows.
    assert "RECORD SUBJ03" in md


def test_verdict_section_none_without_verdict():
    assert ReportGenerationModule._verdict_section(CorrelationResult()) is None
    assert ReportGenerationModule._verdict_section(None) is None


def test_containment_withheld_only_claims_withholding_when_it_can_see_subjects():
    """Absence of a verdict is not evidence that containment was withheld.

    The predicate gates report text that says "no containment target was nominated". A
    brief with no verdict, or a MagicMock whose attributes are all truthy mocks, must not
    be read as a cleared verdict — otherwise a run with no verdict engine at all starts
    telling the reader its recommendations are NOT APPLICABLE.
    """
    from src.report_generation import _containment_withheld

    assert _containment_withheld(None) is False
    assert _containment_withheld(MagicMock()) is False  # subjects is not a real list
    assert _containment_withheld(_verdict_correlation().brief or MagicMock()) is False

    brief = MagicMock()
    brief.verdict = MagicMock()
    brief.verdict.subjects = []  # a real list, but empty → nothing to conclude
    assert _containment_withheld(brief) is False

    # one subject still carrying a target is enough to keep containment in play
    nominated, cleared = MagicMock(), MagicMock()
    nominated.lock_target = {"org_unit": "ORG2428D4", "sign": "0201GPSU"}
    cleared.lock_target = {}
    brief.verdict.subjects = [cleared, nominated]
    assert _containment_withheld(brief) is False
    brief.verdict.subjects = [cleared]
    assert _containment_withheld(brief) is True
    # a target holding only provenance nominates nobody
    cleared.lock_target = {"source": "record_lake"}
    assert _containment_withheld(brief) is True


def _indicator_verdict_correlation():
    """A VALID FRAUD verdict driven by positive fraud indicators (polarity)."""
    verdict = ValidationVerdict(
        label_scheme="scheme",
        summary="VALID FRAUD: 1",
        degraded=False,
        subjects=[
            SubjectVerdict(
                subject_type="record",
                subject_value="SUBJ01",
                verdict="VALID FRAUD",
                checks=[
                    ConditionCheck(
                        id="bare",
                        label="Bare record",
                        result="fail",
                        observed="AUX=6",
                        decisive=True,
                        polarity="exclusion",
                    ),
                    ConditionCheck(
                        id="non_agency_email",
                        label="Non-agency email",
                        result="fail",
                        observed="ORG_UNIT@EXAMPLECO.RS",
                        polarity="fraud_indicator",
                    ),
                    ConditionCheck(
                        id="cash_tender",
                        label="Cash Tender",
                        result="fail",
                        observed="CA",
                        polarity="fraud_indicator",
                    ),
                ],
                lock_target={
                    "org_unit": "ORGUNIT01",
                    "sign": "0303CDSU",
                    "source": "record_lake",
                },
                notes=[
                    "fraud_indicators=Non-agency email (ORG_UNIT@EXAMPLECO.RS), Cash Tender (CA)"
                ],
            )
        ],
        notification_draft="Outcome for record SUBJ01: VALID FRAUD.",
    )
    return CorrelationResult(verdict=verdict)


def test_verdict_section_renders_indicator_callout():
    sec = ReportGenerationModule._verdict_section(_indicator_verdict_correlation())
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-IND"})
    assert "POSITIVE FRAUD INDICATOR" in md
    assert "EXPERT MUST CONFIRM" in md
    assert "auto-void/lock/freeze" in md
    assert "Non-agency email" in md and "Cash Tender" in md


def test_verdict_section_suppresses_indicator_callout_on_categorical_exclusion():
    """ "This verdict rests on positive evidence of fraud" cannot appear under a FALSE
    POSITIVE heading in the same section.

    Detection reads `ConditionCheck.exclusion_kind` — the same field the verdict engine
    ranked on — rather than string-matching the engine's prose note, so a reworded note
    can't silently re-enable the contradiction."""
    verdict = ValidationVerdict(
        label_scheme="scheme",
        summary="FALSE POSITIVE: 1",
        degraded=False,
        subjects=[
            SubjectVerdict(
                subject_type="record",
                subject_value="9CRPQG",
                verdict="FALSE POSITIVE",
                checks=[
                    ConditionCheck(
                        id="not_automated_ref",
                        label="Issuance identity is not a known AUTOMATED account",
                        result="fail",
                        observed="AUTOMATED",
                        decisive=True,
                        polarity="exclusion",
                        exclusion_kind="categorical",
                    ),
                    ConditionCheck(
                        id="cash_tender",
                        label="Cash Tender",
                        result="fail",
                        observed="CA",
                        polarity="fraud_indicator",
                    ),
                ],
                notes=[
                    "categorical_exclusion=Issuance identity is not a known AUTOMATED "
                    "account (AUTOMATED). This settles WHO acted, so it outranks the "
                    "behavioural fraud indicators."
                ],
            )
        ],
    )
    md = ReportGenerationModule.render_markdown(
        [ReportGenerationModule._verdict_section(CorrelationResult(verdict=verdict))],
        {"id": "IR10000002"},
    )
    assert "POSITIVE FRAUD INDICATOR" not in md
    # The indicator is still disclosed, and so is why it does not make this fraud.
    assert "CATEGORICAL" in md
    assert "outranks" in md


def test_render_brief_for_prompt_includes_indicators():
    from src.models.pydantic_models import InvestigationBrief

    brief = InvestigationBrief(
        use_case="scheme",
        decisive_indicators=[
            ConditionCheck(
                id="cash_tender",
                label="Cash Tender",
                result="fail",
                observed="CA",
                polarity="fraud_indicator",
            ),
        ],
        action_backbone=["record SUBJ01: EXPERT MUST CONFIRM before containment."],
    )
    text = ReportGenerationModule._render_brief_for_prompt(brief)
    assert "POSITIVE fraud indicator" in text
    assert "EXPERT CONFIRMATION" in text or "expert" in text.lower()


def test_the_brief_states_which_WAY_a_failed_exclusion_points():
    """A failed exclusion argues AGAINST fraud, and the brief must say so.

    "FAIL" reads as bad news. On IR10000004 the narration took the differing order and
    issuance signs — an exclusion, i.e. the normal shape of one org_unit issuing for
    another — and called it "a decisive fraud indicator … hallmark of a split-actor fraud
    pattern": the same observation the expert cited as the reason it was NOT fraud. The
    brief was already ground truth, but it named the check without stating its direction,
    so the inversion contradicted nothing. A categorical exclusion is additionally marked
    as the attributed fact the verdict rests on."""
    from src.models.pydantic_models import InvestigationBrief

    brief = InvestigationBrief(
        use_case="scheme",
        decisive_fails=[
            ConditionCheck(
                id="same_agent",
                label="Same agent booked and issued",
                result="fail",
                observed="['0707LMSU'] vs ['0606GK']",
                polarity="exclusion",
                exclusion_kind="categorical",
            ),
            ConditionCheck(
                id="bare",
                label="Bare record",
                result="fail",
                observed="AUX=9",
                polarity="exclusion",
            ),
        ],
        action_backbone=["record SUBJ05: no containment."],
    )
    text = ReportGenerationModule._render_brief_for_prompt(brief)
    assert "AGAINST fraud" in text
    assert "evidence" in text and "OF fraud" in text  # the explicit prohibition
    # The categorical one is singled out; the heuristic one is not.
    fact_lines = [ln for ln in text.splitlines() if "ATTRIBUTED FACT" in ln]
    assert len(fact_lines) == 1 and "0606GK" in fact_lines[0]


def _concept_brief(n_concepts=6, snippet_chars=220, trigger_chars=1854):
    """A brief shaped like the live one that exposed the tail guillotine: a long trigger
    paragraph, several declared concept docs, and nothing else worth cutting."""
    from src.models.pydantic_models import AlertFacts, ConceptRef, InvestigationBrief

    return InvestigationBrief(
        use_case="scheme",
        alert_facts=AlertFacts(trigger="T" * trigger_chars),
        concept_refs=[
            ConceptRef(
                concept_id=f"c{i}",
                title=f"CONCEPT_TITLE_{i}",
                snippet=f"S{i}" * (snippet_chars // 2),
            )
            for i in range(n_concepts)
        ],
        action_backbone=["record SUBJ07: no containment."],
        notes=["DATA_QUALITY_NOTE"],
    )


def test_an_over_budget_brief_shortens_its_snippets_instead_of_deleting_its_tail():
    """Overflow used to guillotine the joined string, so it deleted WHOLE BLOCKS off the end.

    The block that renders last is not the block that matters least: the tail is the pack's
    declared grounding concepts, then the precedents, then the data-quality notes. Measured on
    a live pack with a MINIMAL brief — one condition, no assets, no timeline, no precedents —
    against a 1,854-char trigger paragraph and six declared concepts: at the report stage's
    old 3,000-char budget the render was truncated and NONE of the six concepts appeared at
    all. So the one part whose length the pack tunes is fitted to the room actually left, and
    the floor is the title alone: a narrator that knows a doc exists can ask for it."""
    brief = _concept_brief()
    tight = ReportGenerationModule._render_brief_for_prompt(brief, char_budget=2600)
    # Every title survives, and so does the block AFTER the concepts.
    for i in range(6):
        assert f"CONCEPT_TITLE_{i}" in tight, (i, tight[-400:])
    assert "DATA_QUALITY_NOTE" in tight
    assert "Relevant KB concepts:" in tight
    # It fit rather than being cut off mid-word.
    assert "[truncated]" not in tight and len(tight) <= 2600
    # Ample budget: nothing is shortened at all.
    ample = ReportGenerationModule._render_brief_for_prompt(brief, char_budget=100000)
    assert "[truncated]" not in ample
    for i in range(6):
        assert f"S{i}" * 110 in ample, i
    # And the tightening is monotone in the budget, not all-or-nothing.
    assert len(tight) < len(ample)


def test_both_narrating_stages_read_the_same_amount_of_the_brief():
    """One brief, two stages, and the smaller budget belonged to the acceptance artifact.

    Report generation passed NO budget, taking the render function's bare 3,000 default, while
    anomaly detection deliberately used 4,000 — so the report, which is the artifact the case
    is accepted on, narrated from less of the verdict than the scorer did. The key name and
    the default are shared now; a stage may still override its own."""
    from src.anomaly_detection import AnomalyDetectionModule
    from src.brief_prompt import DEFAULT_BRIEF_CHAR_BUDGET

    rpt = ReportGenerationModule({}, llm_client=MagicMock())
    det = AnomalyDetectionModule({}, llm_client=MagicMock())
    assert rpt._brief_char_budget() == det._brief_char_budget()
    assert rpt._brief_char_budget() == DEFAULT_BRIEF_CHAR_BUDGET
    # Overridable per stage, by the same key.
    assert (
        ReportGenerationModule(
            {"brief_char_budget": 1234}, llm_client=MagicMock()
        )._brief_char_budget()
        == 1234
    )
    # And the shared default is the measured fit, not the old smaller of the two.
    assert DEFAULT_BRIEF_CHAR_BUDGET >= 6000


def test_a_failed_exclusion_that_moved_no_class_still_states_its_direction():
    """The SAME protection, for a FAIL that is not decisive — and it had none.

    FOUND BY A LIVE RUN (`ed265d70`, 2026-08-15): a non-decisive exclusion FAILED, so the
    innocent explanation WAS found, and the check reads several same-named columns of which
    only one carries the finding. `decisive_fails` could not carry it — that list is the
    verdict's reasons and `collect_precedents` matches on its ids — so the brief said
    nothing, and the scorer re-derived a direction from the rows: "contradictory evidence …
    blocks the exclusion instead of satisfying it" at 0.68, which reached a paragraph of
    narrative and a recommended action, in a report whose own condition list two sections
    earlier printed the FAIL and what it established. Note which side was already covered:
    a fraud-INDICATOR fail reaches the brief regardless of decisiveness. Only the exclusion
    side was gated, and the exclusion side is the invertible one."""
    from src.models.pydantic_models import InvestigationBrief

    brief = InvestigationBrief(
        use_case="scheme",
        explanatory_fails=[
            ConditionCheck(
                id="holder_entitlement",
                label="No holder entitlement explains the reduction",
                result="fail",
                observed="CHD",
                detail="an entitlement is recorded on this record",
                polarity="exclusion",
            ),
        ],
        action_backbone=["record SUBJ07: no containment."],
    )
    text = ReportGenerationModule._render_brief_for_prompt(brief)
    assert "AGAINST fraud" in text and "OF fraud" in text
    # The three re-descriptions the live run actually produced, each named.
    assert "contradiction" in text and "inconsistency" in text
    assert "disagreeing" in text
    # ...and the qualifier, or "not decisive" is read as "not established".
    assert "verdict CLASS only" in text
    assert "CHD" in text and "holder entitlement" in text.lower()


def test_a_cross_subject_check_list_names_the_subject_each_value_belongs_to():
    """Two subjects, one condition, two different values — and nothing said which was which.

    FOUND BY A LIVE RUN (`44475777`, 2026-08-15). Every brief list that carries checks is FLAT
    ACROSS SUBJECTS, so the parent that identified the subject is gone by the time the prompt
    is rendered, and the same condition on two records produces two bullets differing only in
    a number. On that run `remark.rm×80` belonged to one record and `remark.rm×88` to the
    other; the narration swapped them. It is NOT a narration slip to prompt against — the
    attribution was not in the input, so the narrator was guessing between two bullets it had
    no way to tell apart. The RENDER half is asserted here; the builder half (does the stamp
    arrive at all?) is in `test_usecases.py`, and neither implies the other: this test
    constructs the checks by hand, so it survives removing the stamp in the builder and dies
    only when the prompt stops printing it."""
    from src.models.pydantic_models import InvestigationBrief

    def _bare(subject, observed):
        return ConditionCheck(
            id="sparse",
            label="Record is bare",
            result="fail",
            observed=observed,
            detail="disallowed elements present",
            polarity="exclusion",
            decisive=True,
            subject=subject,
        )

    brief = InvestigationBrief(
        use_case="scheme",
        decisive_fails=[_bare("SUBJ01", "rm×80"), _bare("SUBJ02", "rm×88")],
        action_backbone=["no containment."],
    )
    text = ReportGenerationModule._render_brief_for_prompt(brief)
    # Each value travels with its own subject, on its own line.
    lines = [ln for ln in text.splitlines() if "rm×" in ln]
    assert len(lines) == 2
    assert any("[SUBJ01]" in ln and "rm×80" in ln for ln in lines)
    assert any("[SUBJ02]" in ln and "rm×88" in ln for ln in lines)
    # No line may carry one subject's tag beside the other's value.
    assert not any("[SUBJ01]" in ln and "rm×88" in ln for ln in lines)
    assert not any("[SUBJ02]" in ln and "rm×80" in ln for ln in lines)
    # And the prohibition is stated, because a prefix a reader may merge is not a fix.
    assert "SUBJECT it was measured on" in text

    # A single-subject case that never set it renders exactly as before — no empty brackets.
    plain = InvestigationBrief(
        use_case="scheme",
        decisive_fails=[
            ConditionCheck(id="sparse", label="Record is bare", result="fail",
                           observed="rm×80", polarity="exclusion", decisive=True)
        ],
        action_backbone=["no containment."],
    )
    assert "[] " not in ReportGenerationModule._render_brief_for_prompt(plain)


@pytest.mark.asyncio
async def test_generate_appends_verdict_section(tmp_path):
    """generate() appends the verdict section when correlation carries a verdict, even
    on the deterministic-fallback path (LLM down)."""
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(side_effect=TimeoutError("down"))
    module = ReportGenerationModule(config, llm)

    out = await module.generate(
        {"id": "INC-9"},
        _understanding(),
        {"record_lake": [{}]},
        [_anomaly("a", 0.5)],
        correlation=_verdict_correlation(),
    )
    # output_format=txt → returns the JSON-serialized sections; the verdict section and
    # the verdict-gated Actions section (with the containment target + its resolved
    # action) are present alongside the standard sections.
    assert "SCHEME Verdict" in out
    assert "CONTAINMENT TARGET" in out
    assert "LOCK the Sell Classic" in out


@pytest.mark.asyncio
async def test_generate_orders_the_mandated_sections_in_the_procedures_order(tmp_path):
    """The mandated sections appear, once each, in the report's own order.

    Ordering is the only mechanism that keeps them in place — the builders append in any
    order and `_order_sections` places them — so it is asserted end-to-end rather than per
    builder. The narrated sections interleave, but the mandated ones must not swap.
    """
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(side_effect=TimeoutError("down"))
    module = ReportGenerationModule(config, llm)

    corr = _grouped_correlation()
    corr.brief = _alert_facts_brief()
    out = await module.generate(
        {"id": "INC-ORDER"},
        _understanding(),
        {"record_lake": [{}]},
        [_anomaly("a", 0.5)],
        correlation=corr,
    )
    # Conclusion first: the verdict sits right after the Executive Summary, so a reader who stops
    # after one screen has still read the answer. Then what was alleged, how it was investigated,
    # what the evidence shows, what may be done, and the appendix.
    expected = [
        "SCHEME Verdict",
        module._SEC_ALERT_FACTS,
        module._SEC_RECONCILIATION,
        module._SEC_NOT_RETRIEVED,
        module._SEC_CONDITIONS,
        module._SEC_ACTIONS,
        module._SEC_ARTIFACTS,
    ]
    positions = [out.index(t) for t in expected]  # raises if any is missing
    assert positions == sorted(positions), dict(zip(expected, positions))
    # Evidence Artifacts is truly last: a section after it reads as more file listing.
    assert positions[-1] == max(positions)


# --- verdict-grounded narration (brief injection + anomaly clamp) -----------

from src.models.pydantic_models import (AssetTimelineEntry, ConceptRef,
                                        InvestigationBrief)


def _brief_correlation(verdict_label="FALSE POSITIVE"):
    """A correlation result carrying both a verdict and an InvestigationBrief."""
    corr = _verdict_correlation(verdict_label)
    fail = ConditionCheck(
        id="bare",
        label="Bare record",
        result="fail",
        observed="AUX=3",
        detail="not bare",
        decisive=True,
    )
    corr.brief = InvestigationBrief(
        use_case="scheme",
        playbook_id="PB-APP-SCHEME-001",
        verdict=corr.verdict,
        decisive_fails=[fail] if verdict_label == "FALSE POSITIVE" else [],
        asset_timeline=[
            AssetTimelineEntry(
                timestamp="2026-07-24T20:19Z",
                event_type="issued",
                entity_type="document",
                entity_value="0575033837736",
                actor="0201GP",
            )
        ],
        action_backbone=[
            (
                "record SUBJ03: CLOSE the IR as FALSE POSITIVE (decisive exclusion: Bare record)."
                if verdict_label == "FALSE POSITIVE"
                else "record SUBJ03: LOCK the order agent 0201GPSU @ ORG2428D4."
            )
        ],
        join_status={"join_record": "ran, 2 cross-source match(es)"},
        concept_refs=[
            ConceptRef(
                concept_id="bare_record",
                title="Bare record",
                snippet="A bare record carries only NM.",
            )
        ],
    )
    return corr


@pytest.mark.asyncio
async def test_brief_injected_as_system_message(tmp_path):
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    captured = {}

    async def fake_structured_output(
        messages, response_model, max_tokens=None, rag=None, stage=None
    ):
        captured["messages"] = messages
        return InvestigationReport(sections=[])

    llm.structured_output = AsyncMock(side_effect=fake_structured_output)
    module = ReportGenerationModule(config, llm)
    await module.generate(
        {"id": "INC-1"},
        _understanding(),
        {"record_lake": [{}]},
        [_anomaly("a", 0.9)],
        correlation=_brief_correlation("FALSE POSITIVE"),
    )
    # The brief rides as a second system message before the user message.
    system_texts = "\n".join(
        m["content"] for m in captured["messages"] if m["role"] == "system"
    )
    assert "AUTHORITATIVE INVESTIGATION BRIEF" in system_texts
    assert "FALSE POSITIVE" in system_texts
    assert "CLOSE the IR" in system_texts
    assert "join_record" in system_texts


@pytest.mark.asyncio
async def test_no_brief_leaves_prompt_unchanged(tmp_path):
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    captured = {}

    async def fake_structured_output(
        messages, response_model, max_tokens=None, rag=None, stage=None
    ):
        captured["messages"] = messages
        return InvestigationReport(sections=[])

    llm.structured_output = AsyncMock(side_effect=fake_structured_output)
    module = ReportGenerationModule(config, llm)
    await module.generate(
        {"id": "INC-1"},
        _understanding(),
        {"src": []},
        [_anomaly("a", 0.9)],
        correlation=None,
    )
    # Only the base system prompt + user message: no injected brief system message.
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user"]


def test_render_brief_for_prompt_guards_mock_and_none():
    from unittest.mock import MagicMock as MM

    assert ReportGenerationModule._render_brief_for_prompt(None) == ""
    assert ReportGenerationModule._render_brief_for_prompt(MM()) == ""


def test_asset_impact_lines_from_brief():
    lines = ReportGenerationModule._asset_impact_lines(_brief_correlation())
    joined = " ".join(lines)
    assert "Asset impact" in joined
    assert "0575033837736" in joined and "issued" in joined
    # No brief -> no lines.
    assert ReportGenerationModule._asset_impact_lines(CorrelationResult()) == []


def test_asset_impact_lines_carry_the_current_state_into_the_artifact():
    """The sweep's per-asset CURRENT state (highest version) must be in the EXPORTED
    report, not only in the narration prompt. "Is this document still live?" is the fact
    containment turns on; leaving it to the LLM meant a live document could be dropped
    from the artifact entirely (observed on the IR10000001 rerun)."""
    from src.models.pydantic_models import ImpactedAsset

    corr = _brief_correlation("VALID FRAUD")
    corr.brief.scope_status = (
        "ran, 2 asset(s) across 2 subject(s); 1 NOT named in the alert"
    )
    corr.brief.impacted_assets = [
        ImpactedAsset(
            subject="SUBJ01",
            asset_id="400-2000000006",
            amount="89211",
            currency="RSD",
            status="I (current; was T -> V)",
            actor="0303CDSU @ ORGUNIT01",
            known=True,
        ),
        ImpactedAsset(
            subject="SUBJ09",
            asset_id="200-2000000004",
            amount="",
            currency="",
            status="T",
            actor="0303CDSU @ ORGUNIT01",
            known=False,
        ),
    ]
    joined = "\n".join(ReportGenerationModule._asset_impact_lines(corr))
    assert "I (current; was T -> V)" in joined  # the version rollup, verbatim
    assert "200-2000000004" in joined
    assert "NOT named in the alert" in joined  # the unalerted subject is flagged
    assert "89211 RSD" in joined
    assert "1 NOT named in the alert" in joined  # scope_status rides along
    # An asset list with no timeline still produces the section.
    corr.brief.asset_timeline = []
    assert "200-2000000004" in "\n".join(
        ReportGenerationModule._asset_impact_lines(corr)
    )


def test_out_of_window_assets_are_labelled_in_both_the_artifact_and_the_prompt():
    """The scope sweep is actor-scoped over a window WIDER than the incident, so it also
    returns that agent's ordinary business. Both renderings must say so: the artifact so the
    operator does not read the total as the exposure, and the prompt so the LLM cannot
    narrate adjacent activity as fraud scope (IR10000001: 11 'extra' records vs the expert's 2,
    because the boundary is the document's ISSUE date, not the sweep's range)."""
    from src.models.pydantic_models import ImpactedAsset

    corr = _brief_correlation("VALID FRAUD")
    corr.brief.impacted_assets = [
        ImpactedAsset(
            subject="SUBJ09",
            asset_id="200-2000000004",
            status="T",
            known=False,
            in_window=True,
            event_date="2026-07-27",
        ),
        ImpactedAsset(
            subject="XYZABC",
            asset_id="200-2000000002",
            amount="40100",
            currency="RSD",
            status="T",
            known=False,
            in_window=False,
            event_date="2026-07-25",
        ),
    ]
    artifact = "\n".join(ReportGenerationModule._asset_impact_lines(corr))
    prompt = ReportGenerationModule._render_brief_for_prompt(corr.brief)

    for text in (artifact, prompt):
        assert "OUTSIDE the incident window" in text
        assert "2026-07-25" in text and "2026-07-27" in text
    # The in-window unalerted subject keeps the containment flag, not the window flag.
    in_line = [ln for ln in artifact.splitlines() if "200-2000000004" in ln][0]
    assert "NOT named in the alert" in in_line and "OUTSIDE" not in in_line
    # The prompt tells the model explicitly not to count it — a bare date label would leave
    # the inference to chance.
    assert "do not count it in the exposure" in prompt


def test_implications_scope_line_counts_only_in_window_documents():
    """'Potential Implications' derived its scope line from `evidence.actors`, which counts
    every entity the sign touched in the retrieved rows and knows nothing about the incident
    window — so it listed the actor's 16JUL and 25JUL documents as scope while the asset table
    right above it correctly flagged them as adjacent business. Prefer the brief's assets.
    """
    from src.models.pydantic_models import (ActorRollup, EvidencePack,
                                            ImpactedAsset)

    corr = _brief_correlation("VALID FRAUD")
    corr.brief.impacted_assets = [
        ImpactedAsset(subject="SUBJ01", asset_id="400-2000000006", in_window=True),
        ImpactedAsset(subject="SUBJ09", asset_id="200-2000000004", in_window=True),
        ImpactedAsset(subject="SUBJ12", asset_id="200-2000000003", in_window=False),
        ImpactedAsset(subject="SUBJ09", asset_id="200-2000000002", in_window=False),
    ]
    evidence = EvidencePack(
        actors=[
            ActorRollup(
                actor="0303CD",
                is_subject=True,
                entities_touched={
                    "document": ["400-2000000006", "200-2000000002", "200-2000000003"]
                },
            )
        ]
    )
    lines = ReportGenerationModule._fallback_implications([], evidence, None, corr)
    scope = [ln for ln in lines if ln.startswith("- Scope:")][0]
    assert "2 document(s) fall inside the incident window" in scope
    assert "400-2000000006" in scope and "200-2000000004" in scope
    assert "200-2000000002" not in scope and "200-2000000003" not in scope
    assert "A further 2 document(s)" in scope and "not incident scope" in scope

    # With no brief assets at all, the evidence-rollup line is still produced (verdict-less
    # incidents keep their existing behaviour).
    corr.brief.impacted_assets = []
    scope = [
        ln
        for ln in ReportGenerationModule._fallback_implications(
            [], evidence, None, corr
        )
        if ln.startswith("- Scope:")
    ][0]
    # The entity TYPE is whatever the extraction produced — the line used to be filtered to a
    # hard-coded pair of one domain's record types, so any other domain read as "no wider
    # scope" no matter how many records its subject touched.
    assert "touched 3 distinct documents" in scope

    evidence.actors[0].entities_touched = {
        "consignment": ["C1", "C2"],
        "claim": [
            "K9"
        ],  # a single value is not a WIDER scope, so it must not be listed
    }
    scope = [
        ln
        for ln in ReportGenerationModule._fallback_implications(
            [], evidence, None, corr
        )
        if ln.startswith("- Scope:")
    ][0]
    assert "2 distinct consignments" in scope
    assert "claim" not in scope


@pytest.mark.asyncio
async def test_deterministic_fallback_next_steps_follow_verdict_not_actor_lock(
    tmp_path,
):
    """When the LLM omits Next Steps (backfill) or is down (fallback), the deterministic
    builder must use the verdict-keyed action backbone — NOT 'lock the actor' — so it
    can't contradict a FALSE POSITIVE / INSUFFICIENT verdict."""
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(
        side_effect=TimeoutError("down")
    )  # force fallback
    module = ReportGenerationModule(config, llm)

    out = await module.generate(
        {"id": "INC-FP"},
        _understanding(),
        {"record_lake": [{}]},
        [_anomaly("looks bad", 0.9)],
        correlation=_brief_correlation("FALSE POSITIVE"),
    )
    # The backbone's CLOSE step is present; no 'lock the responsible identity' actor line.
    assert "CLOSE the IR" in out
    assert "lock the responsible identity" not in out


def test_containment_gated_prefixes_anomaly_actions_as_proposed():
    """An indicator-driven fraud is a CANDIDATE: the anomaly's own 'immediately void the
    documents' must not be emitted as a REMEDIATE step directly under the backbone's own
    'an EXPERT MUST CONFIRM — do NOT auto-void'. The clamp upstream only touches
    confidence SCORES on a FALSE POSITIVE, so this gate is what stops the contradiction.
    """
    corr = _brief_correlation("VALID FRAUD")
    corr.brief.containment_gated = True
    corr.brief.action_backbone = [
        "record SUBJ01: an EXPERT MUST CONFIRM before containment — do NOT auto-void/lock/freeze."
    ]
    anomaly = _anomaly("Cash payment on a same-day document", 0.85)
    anomaly.recommended_actions = (
        "Immediately void the documents and suspend sign 0303CD"
    )

    steps = ReportGenerationModule._fallback_recommendations(
        [anomaly], None, None, corr
    )
    joined = " ".join(steps)
    assert "Immediately void the documents" in joined  # the action is still surfaced...
    assert "REQUIRES EXPERT CONFIRMATION" in joined  # ...but explicitly gated
    assert not any(s.startswith("REMEDIATE") for s in steps)

    # Ungated (no confirmed indicator) -> the ordinary REMEDIATE phrasing.
    corr.brief.containment_gated = False
    steps = ReportGenerationModule._fallback_recommendations(
        [anomaly], None, None, corr
    )
    assert any(s.startswith("REMEDIATE") for s in steps)
    assert "REQUIRES EXPERT CONFIRMATION" not in " ".join(steps)


def test_render_brief_for_prompt_carries_scope_and_gating():
    """The prompt must name the records the sweep found beyond the alert, flag them as NEW,
    and state that containment is gated — the report can't scope what it never sees."""
    from src.models.pydantic_models import ImpactedAsset

    brief = InvestigationBrief(
        use_case="scheme",
        containment_gated=True,
        scope_status="ran, 3 asset(s) across 3 subject(s); 2 NOT named in the alert",
        additional_subjects=["YEFGHJ", "SUBJ09"],
        impacted_assets=[
            ImpactedAsset(
                subject="SUBJ01",
                asset_id="400-2000000006",
                amount="89211",
                currency="RSD",
                status="T",
                known=True,
            ),
            ImpactedAsset(
                subject="SUBJ09",
                asset_id="200-2000000004",
                amount="11755",
                currency="RSD",
                status="T",
                known=False,
            ),
        ],
    )
    text = ReportGenerationModule._render_brief_for_prompt(brief)
    assert "CONTAINMENT IS GATED" in text
    assert "SUBJ09" in text and "YEFGHJ" in text
    assert "200-2000000004" in text
    assert "NEW, not in the alert" in text


def test_fallback_recommendations_prefers_backbone():
    steps = ReportGenerationModule._fallback_recommendations(
        [], None, None, _brief_correlation("FALSE POSITIVE")
    )
    assert any("CLOSE the IR" in s for s in steps)
    # No brief -> heuristic path (manual-review line, since no evidence/anomaly).
    steps2 = ReportGenerationModule._fallback_recommendations([], None, None, None)
    assert steps2 and "CLOSE the IR" not in " ".join(steps2)


# --- degradation channel (read by src/stage_health.py) ---------------------


@pytest.mark.asyncio
async def test_a_fallback_report_records_that_it_degraded(tmp_path):
    """A fallback report looks like a report; only this flag says it wasn't narrated."""
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(side_effect=TimeoutError("Request timed out."))
    module = ReportGenerationModule(config, llm)

    assert module.last_fallback_used is False

    await module.generate(
        {"id": "INC-1"}, _understanding(), {"src": [{"a": 1}]}, [_anomaly("x", 0.9)]
    )

    assert module.last_fallback_used is True
    assert "timed out" in module.last_error


@pytest.mark.asyncio
async def test_backfilled_sections_are_recorded_as_partial_degradation(tmp_path):
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(
        return_value=InvestigationReport(
            sections=[{"section_title": "Executive Summary", "content": "Fraud found."}]
        )
    )
    module = ReportGenerationModule(config, llm)

    await module.generate(
        {"id": "INC-1"},
        _understanding(),
        {"raw_access": [{"x": 1}]},
        [_anomaly("d", 0.9)],
    )

    assert module.last_fallback_used is False  # narration DID succeed
    assert "Analysis and Findings" in module.last_backfilled_sections
    assert "Recommended Next Steps" in module.last_backfilled_sections


@pytest.mark.asyncio
async def test_a_complete_narration_records_no_degradation(tmp_path):
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(
        return_value=InvestigationReport(
            sections=[
                {"section_title": t, "content": "c"}
                for t in ReportGenerationModule._REQUIRED_SECTIONS
            ]
        )
    )
    module = ReportGenerationModule(config, llm)

    await module.generate(
        {"id": "INC-1"}, _understanding(), {"src": [{"a": 1}]}, [_anomaly("x", 0.9)]
    )

    assert module.last_fallback_used is False
    assert module.last_backfilled_sections == []


@pytest.mark.asyncio
async def test_a_later_success_clears_an_earlier_fallback(tmp_path):
    """Stale state would keep reporting a fatal signal after a successful retry."""
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(side_effect=TimeoutError("boom"))
    module = ReportGenerationModule(config, llm)
    await module.generate({"id": "INC-1"}, _understanding(), {"s": []}, [])
    assert module.last_fallback_used is True

    llm.structured_output = AsyncMock(
        return_value=InvestigationReport(
            sections=[
                {"section_title": t, "content": "c"}
                for t in ReportGenerationModule._REQUIRED_SECTIONS
            ]
        )
    )
    await module.generate({"id": "INC-1"}, _understanding(), {"s": []}, [])

    assert module.last_fallback_used is False
    assert module.last_error == ""


def test_a_declared_source_that_never_answered_is_named_in_the_verdict_block():
    """A source that was ASKED and did not answer leaves NO trace anywhere else.

    It is absent from the retrieved rows, so the evidence list, the per-source counts and
    the gaps list have nothing to render for it — measured live, two timed-out sources on
    the system of record produced a report that never mentioned them while the retrieval
    stage reported `completed`. The verdict note is the only carrier, so the renderer must
    print it, and under its own heading: "Data coverage" says the verdict stands without
    that corroboration, which is the opposite of what this note says.
    """
    corr = _verdict_correlation()
    corr.verdict.subjects[0].notes.append(
        "source_unanswered=record_lake: this procedure declares it and it did not answer "
        "within its 1800s budget — its rows are MISSING, not empty."
    )
    sec = ReportGenerationModule._verdict_section(corr)
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-1"})
    assert "SOURCE DID NOT ANSWER — record_lake" in md, md
    assert "MISSING, not empty" in md
    # Not folded into the coverage sentence.
    assert "Data coverage: this procedure declares it" not in md


@pytest.mark.parametrize(
    "key", ["evidence_floor", "no_verdict_reason", "no_verdict_partial"]
)
def test_why_there_is_no_verdict_reaches_the_report(key):
    """The one question the INSUFFICIENT headline cannot answer, and the one this renderer
    never asked. `evidence_floor=` had been emitted by the rollup since the floor shipped and
    was rendered by NO section — the sentence explaining the outcome existed on the object and
    nowhere a reader looks — and the two `no_verdict_*` keys answer the identical question for
    the other roads to the same label. Parametrised because they render through ONE branch on
    purpose: a reader shown only one of them reads the other's absence as "the question did
    not arise", which is what put a census of PASSes under an unexplained headline. The split
    between the last two is a DOWNSTREAM distinction — only one licenses "do not re-run", and
    the action backbone is where that is read — so a second heading here would be a second
    answer to a question the reader asked once.
    """
    corr = _verdict_correlation(verdict_label="INSUFFICIENT DATA")
    corr.verdict.subjects[0].notes.append(
        f"{key}=2 of 9 condition(s) were evaluated and the procedure does not clear on that"
    )
    sec = ReportGenerationModule._verdict_section(corr)
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-1"})
    assert "No verdict on the merits — 2 of 9 condition(s) were evaluated" in md, md
    # Under its own heading rather than merged into the coverage sentence, which says the
    # opposite: that the verdict stands without the missing corroboration.
    assert "Data coverage: 2 of 9" not in md


# --- The ADVISORY lane: cross-procedure correlation ------------------------------------
#
# A section that must never be read as part of the verdict, and four states that must never
# collapse into two. The one easy to lose is `probed_negative`: a sibling procedure CHECKED and
# ruled out, rendered as silence, reads like one nobody asked about, and only one of those still
# needs doing. So each state's reachability is asserted rather than assumed.


def _link(state, **kw):
    """One assessed candidate, with only the fields that state would really carry."""
    from src.models.pydantic_models import LinkFinding

    base = {"target_use_case": f"sibling_{state}", "state": state}
    base.update(kw)
    return LinkFinding(**base)


def _linked_correlation(*links):
    corr = CorrelationResult()
    corr.links = list(links)
    return corr


def test_all_four_link_states_render_and_the_ruled_out_one_is_a_FINDING():
    """Each state gets its own heading, and being ruled out is stated rather than omitted.

    The order is asserted too, and it is not cosmetic: actionable first, and the ruled-out
    candidate BEFORE the unreachable one, so a reader who stops halfway has still seen every
    candidate the engine examined rather than only the ones it could act on.
    """
    sec = ReportGenerationModule._links_section(
        _linked_correlation(
            _link(
                "unreachable",
                pivot_entity="loyalty_id",
                gap_reason="no source in this pack binds it",
            ),
            _link(
                "probed_negative",
                pivot_entity="user",
                pivot_values=["U1"],
                evidence_note="no rejected authentication in the window",
            ),
            _link("not_probed", pivot_entity="user", pivot_values=["U1"]),
            _link(
                "probed_positive",
                pivot_entity="user",
                pivot_values=["U1"],
                advisory_severity="HIGH",
            ),
        )
    )
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-L"})
    for heading in (
        "PRESENT IN THIS EVIDENCE (1)",
        "REACHABLE BUT NOT SETTLED (1)",
        "CONSIDERED AND RULED OUT (1)",
        "NOT REACHABLE FROM THIS EVIDENCE (1)",
    ):
        assert heading in md, (heading, md)
    positions = [
        md.index(h)
        for h in (
            "PRESENT IN THIS EVIDENCE",
            "REACHABLE BUT NOT SETTLED",
            "CONSIDERED AND RULED OUT",
            "NOT REACHABLE FROM THIS EVIDENCE",
        )
    ]
    assert positions == sorted(positions), md
    # The ruled-out candidate is a finding: it is NAMED, with what the rows showed, and the
    # heading says it was checked. Silence here is the defect.
    assert "sibling_probed_negative" in md
    assert "no rejected authentication in the window" in md
    # And the unreachable one names the binding it would have needed — the deliverable of the
    # state, since that is a fact about the pack rather than about this incident.
    assert "would need a loyalty_id value, and this run holds none" in md
    assert "no source in this pack binds it" in md


def test_the_section_says_in_its_OWN_words_that_it_is_not_the_verdict():
    """An advisory line printed beside measured findings is read as one of them.

    Asserted on the label text and not merely on the element, because the whole separation is
    carried by wording here: nothing structural stops a reader quoting an advisory HIGH in a
    handover as "the severity".
    """
    sec = ReportGenerationModule._links_section(
        _linked_correlation(
            _link(
                "probed_positive",
                advisory_severity="HIGH",
                advisory_note="a confirmed cross-procedure shape",
            )
        )
    )
    assert "advisory" in sec["section_title"].lower()
    assert "not part of the verdict" in sec["section_title"].lower()
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-L"})
    assert "ADVISORY, and addressed to a human" in md
    assert "was read by any condition" in md
    assert "stage health" in md
    # The per-candidate severity restates its own provenance, because a reader meets that line
    # without the preamble in front of them.
    assert (
        "advisory severity HIGH — router-added, addressed to a human, and not this run's "
        "severity: a confirmed cross-procedure shape" in md
    )
    # And a confirmed candidate is not a second verdict.
    assert "only a run of that procedure can adjudicate it" in md


def test_no_link_is_the_normal_answer_and_renders_NOTHING():
    """A pack that declares no link surface produces the report it produced before.

    Every branch that could put an empty heading in the artifact: no links at all, an empty
    list, a non-list, no correlation, and a MagicMock — which is what the rest of this file
    hands these builders, and whose every attribute is truthy.
    """
    assert ReportGenerationModule._links_section(None) is None
    assert ReportGenerationModule._links_section(CorrelationResult()) is None
    assert ReportGenerationModule._links_section(_linked_correlation()) is None
    mock = MagicMock()
    assert ReportGenerationModule._links_section(mock) is None
    assert ReportGenerationModule._links_of(mock) == []
    # A list field that is not a list must not raise on the way to reporting nothing. The
    # downstream per-item filter survives a string; only the `isinstance` guard survives a value
    # that cannot be iterated at all.
    for junk in ("not a list", 7, object()):
        loose = MagicMock()
        loose.links = junk
        loose.brief = None
        assert ReportGenerationModule._links_of(loose) == []
        assert ReportGenerationModule._links_section(loose) is None


def test_the_links_are_read_from_the_BRIEF_when_the_result_carries_none():
    """The same list rides on both; a degraded or imported result may carry only one."""
    corr = CorrelationResult()
    corr.links = []
    corr.brief = MagicMock()
    corr.brief.links = [_link("not_probed", target_use_case="sibling_b")]
    assert [f.target_use_case for f in ReportGenerationModule._links_of(corr)] == [
        "sibling_b"
    ]
    sec = ReportGenerationModule._links_section(corr)
    assert "sibling_b" in ReportGenerationModule.render_markdown([sec], {"id": "INC-L"})


def test_the_state_vocabulary_has_ONE_home_and_an_unknown_state_still_RENDERS():
    """The renderer's prose table must cover `src.links.LINK_STATES` exactly.

    Two claims, and the second is why the first is not enough on its own: a state added
    upstream with no entry here would render as silence, which is the one failure the four
    blocks exist to prevent — so it is printed under its own raw name as well, and this test
    fails loudly rather than the artifact failing quietly.
    """
    from src.links import LINK_STATES

    rendered = tuple(state for state, _, _ in ReportGenerationModule._LINK_STATE_BLOCKS)
    assert rendered == LINK_STATES, (rendered, LINK_STATES)
    sec = ReportGenerationModule._links_section(
        _linked_correlation(_link("some_future_state"))
    )
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-L"})
    assert "SOME FUTURE STATE (1)" in md
    assert "sibling_some_future_state" in md


def test_a_long_pivot_list_is_CUT_through_the_reports_one_cut_wording():
    """A list that stops is read as a list that ended, and this report has one wording for it."""
    values = [f"U{i}" for i in range(20)]
    sec = ReportGenerationModule._links_section(
        _linked_correlation(
            _link("not_probed", pivot_entity="user", pivot_values=values)
        )
    )
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-L"})
    assert "U0" in md and "U19" not in md
    assert (
        ReportGenerationModule._cut_note(
            20, ReportGenerationModule._LINK_MAX_PIVOT_VALUES, "value", compact=True
        )
        in md
    )


def test_an_UNMEASURED_entry_signal_says_so_and_a_pivot_only_candidate_does_not():
    """The caveat belongs to a DECLARED signal, and only where the pack shipped no base rate.

    Printed on every candidate it would be a caveat on most of the section, which teaches the
    reader to skip the line that matters — and a candidate resting on a named sibling and a
    pivot never had a base rate to ship in the first place.
    """
    md = ReportGenerationModule.render_markdown(
        [
            ReportGenerationModule._links_section(
                _linked_correlation(
                    _link("probed_positive", signal_id="declined_burst", rung=2)
                )
            )
        ],
        {"id": "INC-L"},
    )
    assert "shipped this signal UNMEASURED" in md
    assert "basis: declared entry signal 'declined_burst', assessed to rung 2" in md

    measured = ReportGenerationModule.render_markdown(
        [
            ReportGenerationModule._links_section(
                _linked_correlation(
                    _link(
                        "probed_positive",
                        signal_id="declined_burst",
                        base_rate="fired on 3 of 27 runs (2026-08-19)",
                    )
                )
            )
        ],
        {"id": "INC-L"},
    )
    assert "UNMEASURED" not in measured
    assert "fired on 3 of 27 runs" in measured

    pivot_only = ReportGenerationModule.render_markdown(
        [
            ReportGenerationModule._links_section(
                _linked_correlation(_link("unreachable", pivot_entity="loyalty_id"))
            )
        ],
        {"id": "INC-L"},
    )
    assert "UNMEASURED" not in pivot_only
    assert "basis: no entry signal" in pivot_only


def test_the_section_states_ONCE_whether_anything_could_act_on_its_own():
    """The escalation posture is a fact about the RUN, so it is stated even when it is "none".

    A section whose whole claim is that it cost the investigation nothing has to say so: a
    reader who cannot tell a run that spent nothing automatically from one that spent two scans
    has to assume the second. And it is stated once — the per-candidate line below exists only
    for candidates that DEPART from the default, because a sentence repeated on every line is
    one a reader learns to skip.
    """
    from src import link_escalation

    default_md = ReportGenerationModule.render_markdown(
        [
            ReportGenerationModule._links_section(
                _linked_correlation(
                    _link("not_probed", pivot_entity="user", pivot_values=["U1"]),
                    _link("probed_negative", pivot_entity="user", pivot_values=["U1"]),
                )
            )
        ],
        {"id": "INC-L"},
    )
    assert (
        "ESCALATION — none of these 2 candidate(s) acted on its own" in default_md
    ), default_md
    assert "was retrieved or run without being asked for" in default_md
    # Once, and nowhere else: nothing departs from the default, so no candidate carries a line
    # of its own and the default sentence is not repeated per candidate.
    assert default_md.count("escalation:") == 0, default_md
    assert default_md.count("ESCALATION —") == 1, default_md

    live_md = ReportGenerationModule.render_markdown(
        [
            ReportGenerationModule._links_section(
                _linked_correlation(
                    _link("not_probed", pivot_entity="user", pivot_values=["U1"]),
                    _link(
                        "probed_positive",
                        target_use_case="sibling_live",
                        mode="auto",
                        mode_source="pack",
                        proposed_action=link_escalation.proposed_action("auto"),
                    ),
                )
            )
        ],
        {"id": "INC-L"},
    )
    assert (
        "ESCALATION — 1 of these 2 candidate(s) are set to act without being asked "
        "(sibling_live)" in live_md
    ), live_md
    # The standing commitment is named on the candidate itself, with what it would do — a pair
    # set to act on its own spends on every future incident of this shape, not once.
    assert "escalation: this pair is set to 'auto', from the pack setting" in live_md
    assert link_escalation.proposed_action("auto") in live_md
    # And exactly the one candidate that departs from the default carries such a line.
    assert live_md.count("escalation: this pair is set to") == 1, live_md


def test_a_REFUSED_escalation_is_a_FINDING_and_the_default_is_not():
    """ "Nobody asked" and "asked and refused" read identically off the mode, and must not.

    They license opposite next steps — the first needs a decision, the second needs the target
    procedure's own applicability test to resolve — so the clamp is named at both levels, while a
    candidate sitting at the default because nothing asked otherwise adds nothing to the section it
    did not already say once.

    The section-level sentence has to name the REASON and not just the refusal, and the reason is a
    fact about this run rather than about the pair: a reader told the pack had not counted enough
    incidents goes off to measure a corpus, when what happened is that the rows deciding whether
    that procedure applies at all were never retrieved here.
    """
    from src import link_escalation

    md = ReportGenerationModule.render_markdown(
        [
            ReportGenerationModule._links_section(
                _linked_correlation(
                    _link("not_probed", target_use_case="sibling_quiet"),
                    _link(
                        "probed_positive",
                        target_use_case="sibling_asked",
                        mode="planned",
                        mode_source="clamp",
                        mode_note=(
                            "'auto' was asked for by the pack layer and this run does not "
                            "license it: " + link_escalation._GATE_REFUSALS["unknown"]
                        ),
                    ),
                )
            )
        ],
        {"id": "INC-L"},
    )
    assert "An escalating setting was asked for on 1 of them and REFUSED" in md, md
    assert (
        "the target procedure's own applicability test did not hold against the rows this "
        "run retrieved" in md
    )
    # And NOT the retired reason, which sent the reader to count a corpus.
    assert "base rate over enough incidents" not in md
    assert (
        "escalation: an automatic setting was asked for here and REFUSED, so this "
        "candidate only proposes" in md
    )
    assert "why it was refused: 'auto' was asked for by the pack" in md
    # The clamped candidate is still held at a composed referral, so the section's own opening
    # line must not claim anything acted by itself.
    assert "none of these 2 candidate(s) acted on its own" in md
    # And the quiet candidate carries no escalation line at all.
    assert md.count("escalation:") == 1, md


def test_the_link_SCORE_is_printed_on_every_candidate_with_the_terms_that_earned_it():
    """The one number in this section, and it is printed whether or not it gated anything.

    A number that appears only where it happened to refuse something is a number a reader cannot
    calibrate — 0.55 means nothing without having seen what a 0.20 looks like on the candidate
    above it. The terms are what make it auditable rather than a figure to take on faith, so they
    are printed beside it and cut through the same `_cut_note` every other list in this report
    states its remainder with.
    """
    md = ReportGenerationModule.render_markdown(
        [
            ReportGenerationModule._links_section(
                _linked_correlation(
                    _link(
                        "probed_positive",
                        target_use_case="sibling_scored",
                        link_score=0.98,
                        link_score_reasons=[
                            "a value of the target's subject entity is in hand (+0.15)",
                            "the target's own applicability test holds (+0.35)",
                            "a declared entry signal fired (+0.30)",
                            "the fired signal discriminates: 3 of 27 (+0.18)",
                            "a fifth term nobody has invented yet (+0.00)",
                        ],
                    ),
                    _link(
                        "unreachable",
                        target_use_case="sibling_vetoed",
                        pivot_entity="loyalty_id",
                        gap_reason="no source in this pack binds it",
                        link_score=0.0,
                        link_score_reasons=[
                            "no value of the target procedure's subject entity is in hand"
                        ],
                    ),
                )
            )
        ],
        {"id": "INC-L"},
    )
    assert (
        "link score: 0.98 of 1.00 (deterministic, from the free rungs only)" in md
    ), md
    assert "the target's own applicability test holds (+0.35)" in md
    # Cut through the shared wording, not a sentence of its own.
    assert ReportGenerationModule._cut_note(5, 4, "term", compact=True) in md
    # The vetoed candidate still carries the number — but not its single term, which restates
    # the `gap_reason` printed two lines above it.
    assert "link score: 0.00 of 1.00" in md, md
    assert "no value of the target procedure's subject entity is in hand" not in md
    assert md.count("link score:") == 2, md


def test_a_score_WITHHELD_escalation_is_the_setting_WORKING_and_not_a_refusal():
    """The third reading of a candidate held at a composed referral, and it needs its own words.

    `planned` because nobody asked, `clamp` because the pair is unmeasured, and this one — the
    pair IS measured, the setting IS `semi_auto`, and this incident's free evidence did not reach
    the threshold. Read as the default it would say nobody set the pair up; read as a clamp it
    would send somebody to go and measure a pair that is already measured. Nothing is missing
    here, which is why it is worded as the setting behaving as defined.
    """
    md = ReportGenerationModule.render_markdown(
        [
            ReportGenerationModule._links_section(
                _linked_correlation(
                    _link("not_probed", target_use_case="sibling_quiet"),
                    _link(
                        "probed_positive",
                        target_use_case="sibling_scored",
                        mode="planned",
                        mode_source="score",
                        link_score=0.45,
                        mode_note=(
                            "'semi_auto' is licensed for this pair, and this link's score of "
                            "0.45 is below the configured 0.60 required to act without a human"
                        ),
                    ),
                )
            )
        ],
        {"id": "INC-L"},
    )
    # Stated once for the section, as a posture and not as a shortfall.
    assert "the setting is 'semi_auto' and this incident's score fell below" in md, md
    assert "that is the setting behaving as defined, not a refusal" in md
    # And once on the candidate, with the arithmetic that produced it.
    assert (
        "escalation: this pair is set to act on its own above a score, and this "
        "candidate's evidence did not reach it" in md
    )
    assert "the arithmetic: 'semi_auto' is licensed for this pair" in md
    # Neither of the other two readings is printed: nothing was refused, and the section's
    # opening line must not claim anything acted by itself.
    assert "REFUSED" not in md, md
    assert "none of these 2 candidate(s) acted on its own" in md
    # Only the departing candidate carries a line — the quiet one adds nothing.
    assert md.count("escalation:") == 1, md


def test_a_DECLARED_hold_over_a_wider_ask_is_the_FOURTH_reading_and_prints_as_a_DECISION():
    """`planned` reaches this renderer four ways and the fourth had no words of its own.

    Nobody asked (silent, correctly), rung 1 refused (a retrieval gap to go and close), the score
    fell short (nothing to do), and — this one — an escalating setting WAS asked for and a narrower
    layer declares otherwise. Measured live at mode=`planned`, source=`pack`, rung 1 = PASS under a
    deployment set to `auto`: the candidate rendered no escalation line at all, so the report's own
    default reading applied and said nobody had set the pair up. The remedy is the one thing that
    separates it from the other three — edit a declaration — and it is unreachable unless the layer
    holding it is named.
    """
    md = ReportGenerationModule.render_markdown(
        [
            ReportGenerationModule._links_section(
                _linked_correlation(
                    _link("not_probed", target_use_case="sibling_quiet"),
                    _link(
                        "probed_positive",
                        target_use_case="sibling_held",
                        mode="planned",
                        mode_source="pack",
                        link_score=0.5,
                        mode_note=(
                            "'auto' was asked for by the config layer, and the narrower pack "
                            "layer declares 'planned' for this link, so the mode is held there"
                        ),
                    ),
                )
            )
        ],
        {"id": "INC-L"},
    )
    # Once for the section, as a hold and not as a failure.
    assert (
        "On 1 of them an escalating setting was asked for and a narrower declaration holds "
        "them at a composed referral anyway" in md
    ), md
    assert "a deliberate hold rather than a refusal or a score" in md
    # And once on the candidate, naming the layer an operator would have to edit.
    assert "escalation: this pair is held at 'planned' by the pack setting" in md, md
    assert "why it is held: 'auto' was asked for by the config layer" in md
    # None of the other three readings: nothing was refused, no score was consulted, and the
    # section's opening line must still not claim anything acted by itself.
    assert "REFUSED" not in md, md
    assert "did not reach it" not in md, md
    assert "none of these 2 candidate(s) acted on its own" in md
    # The quiet candidate stays quiet — a hold nobody asked against is still the silent default.
    assert md.count("escalation:") == 1, md


def test_a_hold_with_NO_reason_is_the_silent_default_and_prints_nothing():
    """The predicate is a conjunction, and the note is the half that carries the ASK.

    A pack declaring `planned` where the deployment never asked for more is the ordinary state of
    almost every candidate of almost every run. Keying the fourth reading on `mode_source` alone
    would print the hold sentence on all of them and the rollup on every section — the same
    boilerplate the per-candidate line is deliberately absent for above.
    """
    md = ReportGenerationModule.render_markdown(
        [
            ReportGenerationModule._links_section(
                _linked_correlation(
                    _link(
                        "probed_positive",
                        target_use_case="sibling_declared",
                        mode="planned",
                        mode_source="pack",
                    ),
                )
            )
        ],
        {"id": "INC-L"},
    )
    assert md.count("escalation:") == 0, md
    assert "narrower declaration" not in md, md
    assert "none of these 1 candidate(s) acted on its own" in md


def test_the_words_a_MODE_prints_have_ONE_home():
    """The report restates no mode vocabulary of its own — same rule as the four states.

    `proposed_action` is read off the finding, where `resolve_link_mode` already wrote it, so
    the report cannot become a second opinion about what a setting would do. Asserted per
    escalating mode, because that is where the sentence is printed and where a divergence would
    tell an operator their run does something it does not.
    """
    from src import link_escalation

    for mode in link_escalation.ESCALATING_MODES:
        md = ReportGenerationModule.render_markdown(
            [
                ReportGenerationModule._links_section(
                    _linked_correlation(
                        _link(
                            "probed_positive",
                            mode=mode,
                            mode_source="config",
                            proposed_action=link_escalation.proposed_action(mode),
                        )
                    )
                )
            ],
            {"id": "INC-L"},
        )
        assert f"this pair is set to '{mode}'" in md, (mode, md)
        assert link_escalation.proposed_action(mode) in md, (mode, md)


@pytest.mark.asyncio
async def test_the_link_section_survives_a_DOWN_narration_model_exactly_once(tmp_path):
    """The section must not depend on the narration call, and must not double on the fallback.

    It is built by the deterministic loop, which runs on both branches — so the requirement
    "including in the fallback path" is met by placement. Adding it to `_fallback_sections` as
    well would emit it twice on exactly the degraded path it exists to survive, so the count
    is what is asserted here and not merely the presence.
    """
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(side_effect=TimeoutError("down"))
    module = ReportGenerationModule(config, llm)

    corr = _verdict_correlation()
    corr.links = [
        _link("probed_negative", evidence_note="checked, and excluded"),
        _link("unreachable", pivot_entity="loyalty_id"),
    ]
    out = await module.generate(
        {"id": "INC-LINK"},
        _understanding(),
        {"record_lake": [{}]},
        [_anomaly("a", 0.5)],
        correlation=corr,
    )
    assert module.last_fallback_used is True, "the premise: narration was down"
    sections = json.loads(out)
    titles = [s.get("section_title") for s in sections]
    assert titles.count(ReportGenerationModule._SEC_LINKS) == 1, titles
    assert "checked, and excluded" in out
    # And the verdict is still the verdict: the advisory section sits with "what may be done",
    # after the authorised actions and before the narrated next steps.
    assert titles.index(ReportGenerationModule._SEC_LINKS) > titles.index(
        ReportGenerationModule._SEC_ACTIONS
    ), titles


@pytest.mark.asyncio
async def test_the_link_section_is_there_exactly_once_on_the_NARRATED_path_too(
    tmp_path,
):
    """The other branch of the same requirement, and it is not the same test twice.

    The deterministic loop runs on both branches, so this pair is what makes "including in the
    fallback" a *measurement* rather than a reading of the code: the section must be present
    whether narration succeeded or failed, and present ONCE either way. A narrated run is the
    branch where duplication is most likely, because `_ensure_complete_sections` keyword-matches
    titles into the backfill — a section title close enough to one of the six narrated ones
    would arrive twice here and never in the fallback test above.
    """
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(
        return_value=InvestigationReport(
            sections=[
                {"section_title": t, "content": "c"}
                for t in ReportGenerationModule._REQUIRED_SECTIONS
            ]
        )
    )
    module = ReportGenerationModule(config, llm)

    corr = _verdict_correlation()
    corr.links = [
        _link("probed_positive", pivot_entity="user", pivot_values=["U1"]),
        _link("probed_negative", evidence_note="checked, and excluded"),
    ]
    out = await module.generate(
        {"id": "INC-LINK-OK"},
        _understanding(),
        {"record_lake": [{}]},
        [_anomaly("a", 0.5)],
        correlation=corr,
    )
    assert module.last_fallback_used is False, "the premise: narration succeeded"
    titles = [s.get("section_title") for s in json.loads(out)]
    assert titles.count(ReportGenerationModule._SEC_LINKS) == 1, titles
    # And the ruled-out candidate is a finding on this path as well: a narrated report is the
    # one an operator actually reads, so silence here would be the defect in its usual place.
    assert "checked, and excluded" in out


# --- The ADVISORY lane, other axis: what THIS procedure could not settle ----------------
#
# A link asks whether another procedure applies. An open question asks what the adjudicating
# procedure left unanswered about its own subject — which makes it the more tempting of the two
# to read as evidence the verdict has merely not folded in yet. Five states, and the pair easiest
# to lose is `empty` against `unanswered`: an empty answer is a reading against a meaning the
# pack declared in advance, while a non-answer is a credential or catalog gap. Rendered as one
# silence they license opposite next steps, so each state's reachability is asserted.


def _inquiry(state, **kw):
    """One assessed open question, with only the fields that state would really carry."""
    from src.models.pydantic_models import InquiryFinding

    base = {"id": f"q_{state}", "state": state, "question": f"Was it {state}?"}
    base.update(kw)
    return InquiryFinding(**base)


def _inquiry_correlation(*inquiries):
    corr = CorrelationResult()
    corr.inquiries = list(inquiries)
    return corr


def test_all_five_open_question_states_render_and_the_two_SILENCES_are_told_apart():
    """Each state gets its own heading, and the two blank answers are different findings.

    The order is asserted and it is not cosmetic: actionable first, so a reader who stops halfway
    has seen the questions somebody can still act on. What the block prose has to carry is the
    distinction the states exist for — `empty` is an answer the procedure declared a meaning for,
    `unanswered` is an environment fault where re-reading the same rows supplies nothing.
    """
    sec = ReportGenerationModule._inquiries_section(
        _inquiry_correlation(
            _inquiry(
                "unreachable",
                scope_entity="refund_claim",
                gap_reason="this run holds no value of it",
            ),
            _inquiry(
                "unanswered",
                source="handover_register",
                meaning="still unknown, and the remedy is a credential",
            ),
            _inquiry(
                "empty",
                source="handover_register",
                rows_matched=0,
                meaning="nothing recorded a handover for this subject anywhere",
            ),
            _inquiry("not_asked", scope_entity="shipment", scope_values=["RT48192043"]),
            _inquiry(
                "answered",
                source="sessions",
                rows_matched=4,
                meaning="the handover was recorded elsewhere",
            ),
        )
    )
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-Q"})
    headings = [
        "ASKED AND ANSWERED (1)",
        "STILL OPEN (1)",
        "ASKED AND THE SOURCE HAD NOTHING (1)",
        "ASKED AND THE SOURCE DID NOT ANSWER (1)",
        "NOT ASKABLE FROM THIS EVIDENCE (1)",
    ]
    for heading in headings:
        assert heading in md, (heading, md)
    positions = [md.index(h) for h in headings]
    assert positions == sorted(positions), md
    # The two silences, each under its own heading and each naming what would resolve it.
    assert "an empty answer and a question nobody asked are otherwise the same silence" in md
    assert "the remedy is the environment" in md
    # And the unreachable one names the value it would have needed, which is a fact about the
    # declaration rather than about this incident.
    assert "would be asked about a refund_claim value, and this run holds none" in md


def test_an_answer_is_reported_WITH_the_meaning_the_procedure_declared():
    """A count with no meaning invites the reader to supply their own, which is the whole defect.

    Both directions are asserted from one section, because the pack declares a meaning per
    outcome: four rows and zero rows are different readings of the same question, and each has to
    arrive beside its own sentence rather than beside a shared one.
    """
    sec = ReportGenerationModule._inquiries_section(
        _inquiry_correlation(
            _inquiry(
                "answered",
                rows_matched=4,
                meaning="the handover was recorded elsewhere, so the gap has an explanation",
                trigger="condition example_stub read unknown",
                source="sessions",
            ),
            _inquiry(
                "empty",
                rows_matched=0,
                meaning="nothing recorded a handover for this subject anywhere",
            ),
        )
    )
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-Q"})
    assert "rows matching it: 4" in md
    assert (
        "what the procedure says that means: the handover was recorded elsewhere" in md
    )
    assert "rows matching it: 0" in md
    assert "nothing recorded a handover for this subject anywhere" in md
    # Why it was raised, because a question with no trigger reads as one somebody typed.
    assert "why it was raised: condition example_stub read unknown" in md
    assert "the source that would answer it: sessions" in md


def test_a_CAPPED_answer_says_its_count_is_a_floor_TWICE():
    """`4 rows` and `4 rows, and there were more` license different next steps.

    Stated on the question itself and again in the lane's cost summary, and that is deliberate
    rather than duplication: the summary is what a reader skimming the headings sees, the line is
    what the reader who stopped at one question sees, and the truncation is a property of the
    query in both places.
    """
    sec = ReportGenerationModule._inquiries_section(
        _inquiry_correlation(
            _inquiry("answered", rows_matched=500, row_cap_hit=True, meaning="recorded")
        )
    )
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-Q"})
    assert "rows matching it: 500 (a floor — the rows read were capped)" in md
    assert "the count below is a floor and not a total" in md
    # And an uncapped answer claims nothing of the kind.
    clean = ReportGenerationModule.render_markdown(
        [
            ReportGenerationModule._inquiries_section(
                _inquiry_correlation(_inquiry("answered", rows_matched=500))
            )
        ],
        {"id": "INC-Q"},
    )
    assert "a floor" not in clean, clean


def test_what_the_LANE_COST_is_stated_once_even_when_it_spent_nothing():
    """The free rung is half this lane and reads as a spend unless the report says otherwise.

    An operator deciding whether to authorise more looking needs the number, and "no query was
    spent" is the answer they are least likely to assume — so it is printed rather than implied
    by the absence of a cost line, which is the silence this whole lane is organised against.
    """
    free = ReportGenerationModule.render_markdown(
        [
            ReportGenerationModule._inquiries_section(
                _inquiry_correlation(
                    _inquiry("answered", rows_matched=1, meaning="recorded"),
                    _inquiry("empty", rows_matched=0, meaning="not recorded"),
                )
            )
        ],
        {"id": "INC-Q"},
    )
    assert "COST — none of these 2 question(s) cost a query" in free
    assert "2 of them were settled at no retrieval cost" in free

    spent = ReportGenerationModule.render_markdown(
        [
            ReportGenerationModule._inquiries_section(
                _inquiry_correlation(
                    _inquiry(
                        "answered",
                        rows_matched=1,
                        meaning="recorded",
                        probe_spent=True,
                        probe_note="one query spent on handover_register",
                    ),
                    _inquiry("not_asked", scope_entity="shipment"),
                )
            )
        ],
        {"id": "INC-Q"},
    )
    assert "COST — 1 of these 2 question(s) cost one bounded query each" in spent
    # After the verdict and outside the conditions' own sources: the two facts that make the
    # spend safe, and neither is derivable from the count.
    assert "made after the verdict was already decided" in spent
    assert "outside every source the conditions read" in spent
    assert "what it cost: one query spent on handover_register" in spent
    # A refusal is a coded row and never a silence, so the reason rides on the same field.
    refused = ReportGenerationModule.render_markdown(
        [
            ReportGenerationModule._inquiries_section(
                _inquiry_correlation(
                    _inquiry(
                        "not_asked",
                        scope_entity="shipment",
                        scope_values=["RT48192043"],
                        gap_reason="budget: no inquiry probe was available on this run",
                    )
                )
            )
        ],
        {"id": "INC-Q"},
    )
    assert "why it is not settled: budget: no inquiry probe was available" in refused


def test_the_open_question_section_says_in_its_OWN_words_that_it_is_not_the_verdict():
    """Two independent signals, because either alone fails a real reader.

    The title, for the reader who skims a table of contents, and the preamble, for the one who
    starts reading at the heading. The preamble names the three things an advisory finding would
    move if it were wired in — verdict, severity, stage health — and says the reading is against
    a meaning declared in advance, because a question left open is the row most easily taken for
    a negative finding.
    """
    sec = ReportGenerationModule._inquiries_section(
        _inquiry_correlation(_inquiry("not_asked", scope_entity="shipment"))
    )
    assert "advisory" in sec["section_title"].lower()
    assert "not part of the verdict" in sec["section_title"].lower()
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-Q"})
    assert "ADVISORY, and addressed to a human" in md
    assert "was read by any condition" in md
    assert "stage health" in md
    assert "a meaning the procedure wrote down in advance" in md
    assert "a question that stayed open is not a negative finding" in md


def test_no_open_question_is_the_normal_answer_and_renders_NOTHING():
    """A pack that declares no `open_questions:` produces the report it produced before.

    Every branch that could put an empty heading in the artifact: none at all, an empty list, a
    non-list, no correlation, and a MagicMock — which is what the rest of this file hands these
    builders, and whose every attribute is truthy.
    """
    assert ReportGenerationModule._inquiries_section(None) is None
    assert ReportGenerationModule._inquiries_section(CorrelationResult()) is None
    assert ReportGenerationModule._inquiries_section(_inquiry_correlation()) is None
    mock = MagicMock()
    assert ReportGenerationModule._inquiries_section(mock) is None
    assert ReportGenerationModule._inquiries_of(mock) == []
    for junk in ("not a list", 7, object()):
        loose = MagicMock()
        loose.inquiries = junk
        loose.brief = None
        assert ReportGenerationModule._inquiries_of(loose) == []
        assert ReportGenerationModule._inquiries_section(loose) is None


def test_the_open_questions_are_read_from_the_BRIEF_when_the_result_carries_none():
    """The same list rides on both; a degraded or imported result may carry only one."""
    corr = CorrelationResult()
    corr.inquiries = []
    corr.brief = MagicMock()
    corr.brief.inquiries = [_inquiry("not_asked", id="q_from_brief")]
    assert [f.id for f in ReportGenerationModule._inquiries_of(corr)] == ["q_from_brief"]
    sec = ReportGenerationModule._inquiries_section(corr)
    assert "q_from_brief" in ReportGenerationModule.render_markdown([sec], {"id": "INC-Q"})


def test_the_open_question_vocabulary_has_ONE_home_and_an_unknown_state_still_RENDERS():
    """The renderer's prose table must cover `src.inquiry.INQUIRY_STATES` exactly.

    Two claims, and the second is why the first is not enough: a state added upstream with no
    entry here would render as silence, which is the one failure the five blocks exist to
    prevent — so it prints under its own raw name and this test fails loudly instead.
    """
    from src.inquiry import INQUIRY_STATES

    rendered = tuple(
        state for state, _, _ in ReportGenerationModule._INQUIRY_STATE_BLOCKS
    )
    assert rendered == INQUIRY_STATES, (rendered, INQUIRY_STATES)
    sec = ReportGenerationModule._inquiries_section(
        _inquiry_correlation(_inquiry("some_future_state"))
    )
    md = ReportGenerationModule.render_markdown([sec], {"id": "INC-Q"})
    assert "SOME FUTURE STATE (1)" in md
    assert "Was it some_future_state?" in md


@pytest.mark.asyncio
async def test_the_open_question_section_survives_a_DOWN_narration_model_exactly_once(
    tmp_path,
):
    """Built by the deterministic loop, so it runs on both branches — and must not double.

    The count is what is asserted rather than the presence, because adding it to
    `_fallback_sections` as well would emit it twice on exactly the degraded path it exists to
    survive. And its place in the order is part of the claim: with "what may be done", after the
    authorised actions and after the other advisory lane, so both lanes sit below everything the
    verdict stands behind.
    """
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(side_effect=TimeoutError("down"))
    module = ReportGenerationModule(config, llm)

    corr = _verdict_correlation()
    corr.inquiries = [
        _inquiry("empty", rows_matched=0, meaning="nothing recorded it anywhere"),
        _inquiry("unreachable", scope_entity="refund_claim"),
    ]
    out = await module.generate(
        {"id": "INC-Q-DOWN"},
        _understanding(),
        {"record_lake": [{}]},
        [_anomaly("a", 0.5)],
        correlation=corr,
    )
    assert module.last_fallback_used is True, "the premise: narration was down"
    titles = [s.get("section_title") for s in json.loads(out)]
    assert titles.count(ReportGenerationModule._SEC_INQUIRIES) == 1, titles
    assert "nothing recorded it anywhere" in out
    assert titles.index(ReportGenerationModule._SEC_INQUIRIES) > titles.index(
        ReportGenerationModule._SEC_ACTIONS
    ), titles


@pytest.mark.asyncio
async def test_the_open_question_section_is_there_exactly_once_on_the_NARRATED_path_too(
    tmp_path,
):
    """The other branch of the same requirement, and not the same test twice.

    A narrated run is where duplication is most likely, because `_ensure_complete_sections`
    keyword-matches titles into the backfill — a title close enough to one of the narrated ones
    would arrive twice here and never in the fallback test above. It is also the report an
    operator actually reads, so a declared meaning going missing here is the defect in its usual
    place.
    """
    config = {"output_format": "txt", "output_path": str(tmp_path)}
    llm = MagicMock()
    llm.max_tokens = 4096
    llm.structured_output = AsyncMock(
        return_value=InvestigationReport(
            sections=[
                {"section_title": t, "content": "c"}
                for t in ReportGenerationModule._REQUIRED_SECTIONS
            ]
        )
    )
    module = ReportGenerationModule(config, llm)

    corr = _verdict_correlation()
    corr.inquiries = [
        _inquiry("answered", rows_matched=2, meaning="recorded elsewhere after all")
    ]
    # Both advisory lanes at once, because they are built by adjacent calls in one loop and a
    # section that swallowed its sibling would be invisible in a report that carried only one.
    corr.links = [_link("probed_negative", evidence_note="checked, and excluded")]
    out = await module.generate(
        {"id": "INC-Q-OK"},
        _understanding(),
        {"record_lake": [{}]},
        [_anomaly("a", 0.5)],
        correlation=corr,
    )
    assert module.last_fallback_used is False, "the premise: narration succeeded"
    titles = [s.get("section_title") for s in json.loads(out)]
    assert titles.count(ReportGenerationModule._SEC_INQUIRIES) == 1, titles
    assert "recorded elsewhere after all" in out
    assert titles.count(ReportGenerationModule._SEC_LINKS) == 1, titles
    assert "checked, and excluded" in out
    # In that order: the other procedure first, then what this one left open, which is the order
    # a reader needs — the second lane is about the verdict they have just read.
    assert titles.index(ReportGenerationModule._SEC_INQUIRIES) > titles.index(
        ReportGenerationModule._SEC_LINKS
    ), titles
