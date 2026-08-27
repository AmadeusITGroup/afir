"""Export one investigation to the analyst-facing formats.

The four methods on the pipeline path (``export_json``, ``export_csv``,
``export_evidence_raw``, ``export_evidence_transformed``) serialise to text and hand over
one named artifact, so where the bytes land is the storage backend's business. It defaults
to local disk under ``exports_dir()``. A Databricks App's filesystem does not survive a
restart, and the evidence artifacts are what makes a finished report checkable.

``export_xml`` and ``export_excel`` still take a real filesystem path, because
``ElementTree.write`` and ``openpyxl``'s ``save`` write to one themselves. Neither is on
the pipeline path; they serve a caller that already has a local directory.
"""

import csv
import io
import json
import logging
import os
import xml.etree.ElementTree as ET
from typing import Optional

from openpyxl import Workbook

from src.storage import LocalStorage, StorageBackend
from src.utils.paths import exports_dir

logger = logging.getLogger(__name__)


class ResultExporter:
    def __init__(self, investigation_result, storage: Optional[StorageBackend] = None):
        self.result = investigation_result
        self._storage = storage

    def _backend(self, filename) -> StorageBackend:
        """The backend to write ``filename`` through.

        With an injected backend the caller's directory is irrelevant: the artifact is
        addressed by name and the backend owns the location. Without one, the directory the
        caller passed is still honoured, because these methods return the path they were
        given as where the artifact is. Defaulting to ``exports_dir()`` regardless would make
        that returned path a claim about a file somewhere else, for any caller whose
        directory is not ``exports_dir()``.
        """
        if self._storage is not None:
            return self._storage
        parent = os.path.dirname(str(filename))
        return LocalStorage(root=parent or exports_dir())

    @staticmethod
    def _key(filename) -> str:
        """The artifact name from whatever the caller passed.

        Callers hand these methods a full path (the existing signature, asserted on by
        ``test_main.py``), so the basename is the key. Splitting here rather than changing
        eight call sites keeps the seam invisible to them, and ``safe_key`` in the backend
        still rejects anything that is not a plain name.
        """
        return os.path.basename(str(filename))

    def _write(self, filename, text: str) -> None:
        if not self._backend(filename).put_text(self._key(filename), text):
            logger.error("Could not write the export artifact %s", self._key(filename))

    def export_json(self, filename):
        self._write(filename, json.dumps(self.result, indent=2, default=str))

    def export_csv(self, filename):
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["Incident ID", "Timestamp", "Description", "Severity"])
        writer.writerow(
            [
                self.result["incident"]["id"],
                self.result["incident"]["timestamp"],
                self.result["incident"]["description"],
                self.result["incident"].get("severity", "N/A"),
            ]
        )
        writer.writerow(
            [
                "Anomaly Description",
                "Confidence Score",
                "Potential Implications",
                "Recommended Actions",
            ]
        )
        for anomaly in self.result["anomalies"]:
            writer.writerow(
                [
                    anomaly["description"],
                    anomaly["confidence_score"],
                    anomaly["potential_implications"],
                    anomaly["recommended_actions"],
                ]
            )
        self._write(filename, buffer.getvalue())

    def export_xml(self, filename):
        root = ET.Element("investigation_result")
        incident = ET.SubElement(root, "incident")
        ET.SubElement(incident, "id").text = self.result["incident"]["id"]
        ET.SubElement(incident, "timestamp").text = self.result["incident"]["timestamp"]
        ET.SubElement(incident, "description").text = self.result["incident"][
            "description"
        ]
        ET.SubElement(incident, "severity").text = str(
            self.result["incident"].get("severity", "N/A")
        )

        understanding = ET.SubElement(root, "understanding")
        ET.SubElement(understanding, "analysis").text = self.result["understanding"][
            "analysis"
        ]

        anomalies = ET.SubElement(root, "anomalies")
        for anomaly in self.result["anomalies"]:
            anomaly_elem = ET.SubElement(anomalies, "anomaly")
            ET.SubElement(anomaly_elem, "description").text = anomaly["description"]
            ET.SubElement(anomaly_elem, "confidence_score").text = str(
                anomaly["confidence_score"]
            )
            ET.SubElement(anomaly_elem, "potential_implications").text = anomaly[
                "potential_implications"
            ]
            ET.SubElement(anomaly_elem, "recommended_actions").text = anomaly[
                "recommended_actions"
            ]

        tree = ET.ElementTree(root)
        tree.write(filename)

    def export_evidence_raw(self, filename, logs):
        """Persist the full retrieved dataset, per source, exactly as extracted.

        ``logs`` is the ``{source: [rows...]}`` dict the retrieval stage returned, written
        verbatim so the analyst has raw evidence to go back to (``default=str`` handles a
        non-JSON cell such as a datetime). Size is unconstrained: this artifact only ever
        goes to storage, never into a prompt.
        """
        payload = {source: (rows or []) for source, rows in (logs or {}).items()}
        self._write(filename, json.dumps(payload, indent=2, default=str))

    def export_evidence_transformed(self, filename, correlation):
        """Persist the transformed (aggregated + correlated) view of a run.

        Serializes the ``CorrelationResult``: aggregations (record counts, entity
        occurrences, cross-source overlap, resolved/discovered correlation keys),
        transforms, the summary text, and the full ``EvidencePack`` (chronology,
        actor attribution, per-source evidence, cross-source joins). Handles a missing
        correlation result gracefully (writes a small note object).
        """
        if correlation is None:
            payload = {"note": "No correlation result was produced for this incident."}
            self._write(filename, json.dumps(payload, indent=2, default=str))
            return

        def _dump(obj):
            # CorrelationResult and its members are Pydantic v2 models.
            md = getattr(obj, "model_dump", None)
            return md(mode="json") if md else obj

        transforms = getattr(correlation, "transforms", None) or []
        evidence = getattr(correlation, "evidence", None)
        payload = {
            "record_count": getattr(correlation, "record_count", None),
            "summary_text": getattr(correlation, "summary_text", None),
            "aggregations": getattr(correlation, "aggregations", None),
            "findings": getattr(correlation, "findings", None),
            "transforms": [_dump(t) for t in transforms],
            "evidence": _dump(evidence) if evidence is not None else None,
        }
        self._write(filename, json.dumps(payload, indent=2, default=str))

    def export_excel(self, filename):
        wb = Workbook()
        ws = wb.active
        ws.title = "Investigation Result"

        ws["A1"] = "Incident ID"
        ws["B1"] = "Timestamp"
        ws["C1"] = "Description"
        ws["D1"] = "Severity"

        ws["A2"] = self.result["incident"]["id"]
        ws["B2"] = self.result["incident"]["timestamp"]
        ws["C2"] = self.result["incident"]["description"]
        ws["D2"] = self.result["incident"].get("severity", "N/A")

        ws["A4"] = "Incident Understanding"
        ws["B4"] = self.result["understanding"]["analysis"]

        ws["A6"] = "Anomaly Description"
        ws["B6"] = "Confidence Score"
        ws["C6"] = "Potential Implications"
        ws["D6"] = "Recommended Actions"

        for i, anomaly in enumerate(self.result["anomalies"], start=7):
            ws[f"A{i}"] = anomaly["description"]
            ws[f"B{i}"] = anomaly["confidence_score"]
            ws[f"C{i}"] = anomaly["potential_implications"]
            ws[f"D{i}"] = anomaly["recommended_actions"]

        wb.save(filename)


# Example usage
if __name__ == "__main__":
    investigation_result = {
        "incident": {
            "id": "INC-12345",
            "timestamp": "2023-07-06T10:30:00",
            "description": "Unusual login activity detected",
            "severity": 8,
        },
        "understanding": {
            "analysis": "Potential unauthorized access detected with multiple failed login attempts followed by a successful login from an unrecognized IP address."
        },
        "anomalies": [
            {
                "description": "Multiple failed login attempts from various IP addresses",
                "confidence_score": 0.95,
                "potential_implications": "Possible brute force attack attempt",
                "recommended_actions": "Implement IP-based rate limiting and notify the account owner",
            },
            {
                "description": "Successful login from an unrecognized IP address after failed attempts",
                "confidence_score": 0.85,
                "potential_implications": "Potential account compromise",
                "recommended_actions": "Force password reset and enable two-factor authentication",
            },
        ],
    }

    exporter = ResultExporter(investigation_result)
    exporter.export_json("result.json")
    exporter.export_csv("result.csv")
    exporter.export_xml("result.xml")
    exporter.export_excel("result.xlsx")

    print("Export completed.")
