"""Tests for the separate evidence artifacts written by ResultExporter.

``export_evidence_raw`` persists all retrieved rows per source verbatim;
``export_evidence_transformed`` persists the CorrelationResult's aggregations,
transforms and the EvidencePack. Both are pure serialization (no LLM/backends).
"""

import json

from src.evidence import build_evidence
from src.export_results import ResultExporter
from src.models.pydantic_models import CorrelationResult, TransformResult


def _exporter():
    # The evidence methods don't touch self.result, so a minimal dict is fine.
    return ResultExporter({"incident": {"id": "INC-1"}})


def test_export_evidence_raw_writes_all_rows_verbatim(tmp_path):
    logs = {
        "app": [{"record": "P1", "user": "U1"}, {"record": "P2", "user": "U2"}],
        "record_lake": [{"locator": "P1", "status": "HK"}],
    }
    path = tmp_path / "evidence_raw_INC-1.json"
    _exporter().export_evidence_raw(str(path), logs)

    loaded = json.loads(path.read_text())
    assert set(loaded) == {"app", "record_lake"}
    assert len(loaded["app"]) == 2
    assert loaded["app"][0] == {"record": "P1", "user": "U1"}
    assert loaded["record_lake"][0]["locator"] == "P1"


def test_export_evidence_raw_handles_empty_and_none(tmp_path):
    path = tmp_path / "raw.json"
    _exporter().export_evidence_raw(str(path), None)
    assert json.loads(path.read_text()) == {}


def test_export_evidence_transformed_includes_aggregations_and_evidence(tmp_path):
    logs = {
        "app": [
            {
                "user": "USERNAMEX",
                "record": "SUBJ03",
                "ts": 1721853060000,
                "action": "ISSUE",
            }
        ],
        "record_lake": [{"locator": "SUBJ03", "off": "LBV"}],
    }
    emap = {
        "app": {"user": "user", "record": "record"},
        "record_lake": {"record": "locator"},
    }

    class _K:
        entity_hint = "record"
        sources = {"app": "record", "record_lake": "locator"}
        time_fields = {"app": "ts"}

    evidence = build_evidence(
        logs, {"total_records": 2}, [_K()], emap, ["USERNAMEX", "SUBJ03"]
    )
    correlation = CorrelationResult(
        record_count=2,
        aggregations={"record_counts": {"app": 1, "record_lake": 1}},
        transforms=[
            TransformResult(
                label="overlap", op="cross_source_overlap", rows=[], note="n"
            )
        ],
        summary_text="correlated",
    )
    # build_evidence returns an EvidencePack under the flat-import module identity
    # (models.*), which the src.-qualified CorrelationResult won't validate at
    # construction — the documented dual-import caveat. The real pipeline builds both
    # from the same module, so assign the attribute directly here.
    correlation.evidence = evidence
    path = tmp_path / "evidence_transformed_INC-1.json"
    _exporter().export_evidence_transformed(str(path), correlation)

    loaded = json.loads(path.read_text())
    assert loaded["record_count"] == 2
    assert loaded["summary_text"] == "correlated"
    assert loaded["aggregations"]["record_counts"]["app"] == 1
    assert loaded["transforms"][0]["op"] == "cross_source_overlap"
    assert loaded["evidence"] is not None
    assert any(a["actor"] == "USERNAMEX" for a in loaded["evidence"]["actors"])


def test_export_evidence_transformed_handles_none(tmp_path):
    path = tmp_path / "transformed.json"
    _exporter().export_evidence_transformed(str(path), None)
    loaded = json.loads(path.read_text())
    assert "note" in loaded


# --- the storage seam ---------------------------------------------------------
#
# Two behaviours, and the second is the one that was nearly lost: an INJECTED backend
# owns the location (the caller's directory is irrelevant, which is what lets the
# artifacts live on a Volume), while with NO backend the caller's directory is still
# honoured — because `_run_export` returns those paths as its answer for where the
# artifacts are, and a path that names a file written elsewhere is a false record.


def test_an_injected_backend_receives_the_artifacts_by_name(tmp_path):
    from src.storage import LocalStorage

    backend = LocalStorage(root=tmp_path / "remote")
    exporter = ResultExporter({"incident": {"id": "INC-1"}}, storage=backend)

    # A caller-supplied directory that does NOT exist: the backend must win, and the
    # write must not touch the path at all.
    exporter.export_evidence_raw(str(tmp_path / "ignored" / "raw.json"), {"app": [{}]})

    assert not (tmp_path / "ignored").exists()
    assert json.loads(backend.get_text("raw.json")) == {"app": [{}]}


def test_without_a_backend_the_callers_directory_is_honoured(tmp_path):
    nested = tmp_path / "run-1"
    nested.mkdir()
    path = nested / "incident_INC-1.json"
    ResultExporter(
        {
            "incident": {
                "id": "INC-1",
                "timestamp": "2024-01-01T00:00",
                "description": "d",
            },
            "understanding": {"analysis": "a"},
            "anomalies": [],
        }
    ).export_json(str(path))

    # Not exports_dir(): the artifact is where the caller said it would be.
    assert json.loads(path.read_text())["incident"]["id"] == "INC-1"
