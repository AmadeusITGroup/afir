import json
import logging

from brief_prompt import DEFAULT_BRIEF_CHAR_BUDGET, render_brief_for_prompt
from human_guidance import guidance_message
from models.pydantic_models import AnomalyList
from utils.llm_client import is_truncation_error

logger = logging.getLogger(__name__)


def preprocess_logs(logs):
    combined_logs = []
    for source, entries in logs.items():
        for entry in entries:
            combined_logs.append(f"[{source}] {json.dumps(entry)}")

    max_chars = 15000  # bounded so the raw dump cannot exceed the prompt budget
    combined_logs_str = "\n".join(combined_logs)
    if len(combined_logs_str) > max_chars:
        combined_logs_str = combined_logs_str[:max_chars] + "... [truncated]"

    return combined_logs_str


class AnomalyDetectionModule:
    def __init__(self, config, llm_client, rag=None, feedback=None):
        self.config = config
        self.llm_client = llm_client
        self.rag = rag
        # Advisory guidance from past analyst reviews; None disables.
        self.feedback = feedback
        # Degradation signals read by stage_health; reset on every call.
        self.last_degraded = False
        self.last_error = ""
        # A retried truncation is not a degradation, but signals the budget is undersized.
        self.last_truncation_retried = False
        self.last_filtered_out = 0
        self.last_filtered_max = 0.0
        self.last_threshold_used = 0.0
        # Read by filter_anomalies to apply the FP ceiling instead of the normal threshold.
        self._clamp_applied = False
        self.system_prompt = (
            "You are a fraud-detection analyst. Inspect the incident understanding and the "
            "log data and report every anomaly, suspicious pattern, or fraud indicator. "
            "Consider unusual access/login patterns, suspicious transactions, abnormal user "
            "or system behavior, possible data breaches, unusual network traffic, and account "
            "inconsistencies. For each anomaly provide a description, the supporting log data, "
            "potential implications, a confidence score between 0 and 1, recommended actions, "
            "and any cross-entry patterns. Be cautious and thorough."
        )
        self.user_template = (
            "Incident Understanding:\n{incident_understanding}\n\n"
            "Correlation Summary:\n{correlation}\n\nLog Data:\n{log_data}"
        )

    async def detect(self, logs, understanding, correlation=None, guidance=None):
        """Return the LLM-scored anomalies, or ``[]`` on any LLM failure.

        Best-effort: the report generates from correlation + logs regardless. No outer
        retry — ``structured_output`` already retries transient errors.
        """
        incident_id = understanding.incident_id
        # Reset degradation channel for this call.
        self.last_degraded = False
        self.last_error = ""
        self.last_truncation_retried = False
        self.last_filtered_out = 0
        self.last_filtered_max = 0.0
        self.last_threshold_used = 0.0
        self._clamp_applied = False
        try:
            # Evidence pack is evidence-dense and budget-sized; raw dump hard-truncates.
            evidence = getattr(correlation, "evidence", None) if correlation else None
            if evidence is not None:
                from evidence import render_for_prompt

                budget = int(self.config.get("llm_input_char_budget", 15000))
                combined_logs = render_for_prompt(evidence, budget)
            else:
                combined_logs = preprocess_logs(logs)
            correlation_text = (
                correlation.summary_text
                if correlation is not None and correlation.summary_text
                else "No correlation summary available."
            )
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]
            # Message order: distilled guidance → analyst direction → verdict framing (authoritative).
            learned = self._feedback_guidance()
            if learned:
                messages.append({"role": "system", "content": learned})
            human_msg = guidance_message(guidance)
            if human_msg is not None:
                messages.append(human_msg)
            brief = getattr(correlation, "brief", None) if correlation else None
            # Added for every verdict class: same renderer as the narration call, so both
            # stages read the same ground truth.
            brief_text = render_brief_for_prompt(
                brief, char_budget=self._brief_char_budget()
            )
            if brief_text:
                messages.append(
                    {
                        "role": "system",
                        "content": "AUTHORITATIVE INVESTIGATION BRIEF (ground truth — your "
                        "anomaly scoring may NOT contradict it, and you must not re-derive "
                        "or re-interpret anything it settles):\n" + brief_text,
                    }
                )
            # Last: narrows scoring when a subject is dismissed.
            framing = self._verdict_framing(brief)
            if framing:
                messages.append({"role": "system", "content": framing})
            messages.append(
                {
                    "role": "user",
                    "content": self.user_template.format(
                        incident_understanding=understanding.analysis.model_dump_json(
                            indent=2
                        ),
                        correlation=correlation_text,
                        log_data=combined_logs,
                    ),
                }
            )

            tokens = self._detect_max_tokens()
            try:
                result = await self.llm_client.structured_output(
                    messages,
                    response_model=AnomalyList,
                    max_tokens=tokens,
                    rag=self.rag,
                    stage="anomaly_detection",
                )
            except Exception as e:
                # Matched on the marker, not the class: dual import paths → different objects.
                if not is_truncation_error(e):
                    raise
                # Truncation → retry once at wider budget; a second overrun means the prompt.
                retry_tokens = min(tokens * 2, self._TRUNCATION_RETRY_CEILING)
                if retry_tokens <= tokens:
                    raise
                logger.warning(
                    "Anomaly detection truncated at %d tokens (%s); retrying once "
                    "at %d.",
                    tokens,
                    e,
                    retry_tokens,
                )
                self.last_truncation_retried = True
                result = await self.llm_client.structured_output(
                    messages,
                    response_model=AnomalyList,
                    max_tokens=retry_tokens,
                    rag=self.rag,
                    stage="anomaly_detection",
                )

            # Deterministic guard; the model may ignore the framing.
            anomalies = self._clamp_by_verdict(result.anomalies, brief)
            # After a clamp, filter against the ceiling, not the normal threshold: a clamp
            # demotes scores to ≤ 0.5 and must not then delete them arithmetically.
            filtered = self.filter_anomalies(anomalies, clamped=self._clamp_applied)
            narrated = len(filtered) - self.last_filtered_out
            logger.info(
                "Detected %d anomalies for incident %s (%d at/above the %.2f threshold, "
                "%d below it and marked, none dropped)",
                len(filtered),
                incident_id,
                narrated,
                self.last_threshold_used,
                self.last_filtered_out,
            )
            return filtered
        except Exception as e:
            # Record why the list is empty so stage health distinguishes a broken stage
            # from a clean incident.
            self.last_degraded = True
            self.last_error = str(e)
            logger.error(
                "Anomaly detection LLM call failed for incident %s (%s); continuing "
                "with 0 anomalies so the report still generates from correlation + logs.",
                incident_id,
                e,
            )
            return []

    # AnomalyItem has several prose fields; a dozen items run past the 4096 client default.
    _DEFAULT_DETECT_MAX_TOKENS = 8000

    # Ceiling for the truncation retry; matches ReportGenerationModule._TRUNCATION_RETRY_CEILING.
    _TRUNCATION_RETRY_CEILING = 16000

    def _detect_max_tokens(self):
        """The response budget for the single detection call."""
        return int(
            self.config.get("detect_max_tokens", self._DEFAULT_DETECT_MAX_TOKENS)
        )

    # Separate budget so a data-heavy incident cannot truncate the verdict away; shared with
    # report generation so both stages read the same amount of it.
    _DEFAULT_BRIEF_CHAR_BUDGET = DEFAULT_BRIEF_CHAR_BUDGET

    def _brief_char_budget(self):
        return int(
            self.config.get("brief_char_budget", self._DEFAULT_BRIEF_CHAR_BUDGET)
        )

    def _feedback_guidance(self):
        """Learned analyst guidance for this stage, or "". Never raises."""
        if self.feedback is None:
            return ""
        try:
            return self.feedback.guidance_prompt("anomaly_detection") or ""
        except Exception as e:  # noqa: BLE001 — guidance is advisory, never fatal
            logger.warning("Could not render feedback guidance: %s", e)
            return ""

    def filter_anomalies(self, anomalies, clamped=False):
        """Mark anomalies below the confidence cut-off; return the full list.

        Items are marked, not dropped: the export path writes the full list. ``clamped``
        lowers the effective threshold to the FP ceiling so demoted items are not then
        filtered out as well.
        """
        threshold = self.effective_threshold()
        if clamped:
            threshold = min(threshold, self._FALSE_POSITIVE_CEILING)
        below = []
        for a in anomalies:
            # Always overwrite: the model may emit its own value, which is not the arithmetic.
            a.below_threshold = a.confidence_score < threshold
            if a.below_threshold:
                below.append(a)
        self.last_filtered_out = len(below)
        self.last_filtered_max = max((a.confidence_score for a in below), default=0.0)
        self.last_threshold_used = threshold
        return list(anomalies)

    def effective_threshold(self):
        """Confidence cut-off: feedback-tuned if available, else the configured value."""
        configured = self.config["threshold"]
        if self.feedback is None:
            return configured
        try:
            tuned = self.feedback.effective_threshold(configured)
            if tuned is None:
                return configured
            tuned = float(tuned)
            if tuned != configured:
                logger.info(
                    "Using feedback-tuned confidence threshold %.2f (configured %.2f).",
                    tuned,
                    configured,
                )
            return tuned
        except Exception as e:  # noqa: BLE001 — tuning must never break filtering
            logger.warning(
                "Could not read tuned threshold (%s); using configured %s.",
                e,
                configured,
            )
            return configured

    # Confidence ceiling for a dismissed verdict; prevents a cleared alert carrying
    # high-confidence "fraud confirmed" findings.
    _FALSE_POSITIVE_CEILING = 0.5

    # Fallback label when no pack declares `false_positive:`; matches `correlation.py`.
    _DEFAULT_FP_LABEL = "FALSE POSITIVE"

    @classmethod
    def _dismissed_subjects(cls, brief):
        """Return subjects in the dismissed verdict class, or None if not applicable.

        Uses ``verdict_class`` (stamped by the rollup), not prose matching or a re-derived
        rollup. Falls back to ``verdict.labels["false_positive"]``; under-fires in preference
        to clamping confirmed fraud. ``subjects`` must be a real list (guard against mock).
        """
        if brief is None:
            return None
        verdict = getattr(brief, "verdict", None)
        subjects = getattr(verdict, "subjects", None) if verdict is not None else None
        if not isinstance(subjects, list) or not subjects:
            return None
        # Checked across the whole list: mixing channels would judge siblings by different rules.
        if any(str(getattr(sv, "verdict_class", "") or "").strip() for sv in subjects):
            return [
                sv
                for sv in subjects
                if str(getattr(sv, "verdict_class", "") or "").strip()
                == "false_positive"
            ]
        labels = getattr(verdict, "labels", None) if verdict is not None else None
        fp_label = ""
        if isinstance(labels, dict):
            fp_label = str(labels.get("false_positive", "") or "").strip()
            if not fp_label and labels:
                # Pack declared other labels but not this one; use the engine default.
                fp_label = cls._DEFAULT_FP_LABEL
        if not fp_label:
            return []
        return [
            sv for sv in subjects if str(getattr(sv, "verdict", "")).strip() == fp_label
        ]

    @classmethod
    def _dismissal_label(cls, brief):
        """The pack's word for a dismissed verdict, for use in prose."""
        verdict = getattr(brief, "verdict", None) if brief is not None else None
        labels = getattr(verdict, "labels", None) if verdict is not None else None
        if isinstance(labels, dict):
            word = str(labels.get("false_positive", "") or "").strip()
            if word:
                return word
        return cls._DEFAULT_FP_LABEL

    def _verdict_framing(self, brief):
        """Framing system message for the verdict; returns "" when inapplicable."""
        fp = self._dismissed_subjects(brief)
        if fp is None:
            return ""
        verdict = getattr(brief, "verdict", None)
        summary = getattr(verdict, "summary", "") if verdict is not None else ""
        lines = [
            "CORRELATION VERDICT CONTEXT (authoritative — your anomaly scoring must not "
            f"contradict it): {summary}.",
        ]
        if fp:
            reasons = []
            for sv in fp:
                for c in getattr(sv, "checks", []) or []:
                    if (
                        getattr(c, "decisive", False)
                        and getattr(c, "result", "") == "fail"
                    ):
                        # Use `detail` (finding-phrased), not `label` (the requirement):
                        # for an exclusion a FAIL negates the label.
                        note = (getattr(c, "detail", "") or "").strip()
                        if not note:
                            req = getattr(c, "label", "") or getattr(c, "id", "")
                            note = f"violated requirement: {req}" if req else ""
                        if note:
                            reasons.append(note)
            reason_txt = (
                f" Decisive exclusion finding(s): {'; '.join(sorted(set(reasons)))}."
                if reasons
                else ""
            )
            lines.append(
                f"One or more subjects are assessed {self._dismissal_label(brief)} per the "
                "official procedure." + reason_txt + " For those subjects, do NOT report "
                "HIGH-confidence 'fraud confirmed' anomalies; keep confidence at or below "
                f"{self._FALSE_POSITIVE_CEILING} and describe findings as reviewed-and-"
                "dismissed, naming the exclusion reason."
            )
        return "\n".join(lines)

    def _clamp_by_verdict(self, anomalies, brief):
        """Clamp confidence to the dismissal ceiling for any dismissed subject. Never raises."""
        self._clamp_applied = False
        try:
            fp = self._dismissed_subjects(brief)
            if not fp:
                return anomalies
            self._clamp_applied = True
            for a in anomalies:
                if a.confidence_score > self._FALSE_POSITIVE_CEILING:
                    a.confidence_score = self._FALSE_POSITIVE_CEILING
            return anomalies
        except Exception as e:  # never fail the stage on a clamp error
            logger.warning("Anomaly verdict clamp failed (%s); using raw scores.", e)
            return anomalies
