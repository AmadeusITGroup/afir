"""
Human-in-the-loop feedback: collect, distill, and apply analyst reviews.

1. Collect. Reviews append to ``feedback_log.jsonl`` (audit trail) and
   ``feedback_pending.jsonl`` (batch), reloaded at construction.
2. Distill. At ``batch_size`` reviews, one LLM call produces ``FeedbackInsights``.
3. Apply. ``guidance_prompt()`` injects insights, ranked below pack and any verdict.
   ``tune_threshold()`` adjusts from counted fields; five guards: opt-in, min reviews
   (``min_reviews_for_tuning``), net signal (``_MIN_NET_SIGNAL``), one step
   (``threshold_step``) bounded by drift cap, per-window watermark.

Tuned value in ``feedback_thresholds.json``, not YAML (YAML is the baseline anchor).
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from src.models.pydantic_models import FeedbackEntry, FeedbackInsights
from src.storage import LocalStorage, json_verifier
from src.utils.paths import data_dir

logger = logging.getLogger(__name__)

# Filenames under the data dir.
_LOG_FILE = "feedback_log.jsonl"
_PENDING_FILE = "feedback_pending.jsonl"
_INSIGHTS_FILE = "feedback_insights.json"
# Tuned values and adjustment evidence. Not the operator's YAML, which stays the
# declared baseline.
_TUNING_FILE = "feedback_thresholds.json"

# Bounds on the injected guidance so a long feedback history cannot crowd out the
# incident itself in a prompt.
_MAX_ITEMS_PER_CATEGORY = 5
_MAX_GUIDANCE_CHARS = 1800

# Threshold auto-tuning bounds. Defaults are conservative: a small step, a hard
# floor/ceiling, and a cap on total drift from the operator's configured value.
_DEFAULT_STEP = 0.05
_DEFAULT_MIN_REVIEWS = 5
_DEFAULT_MAX_DRIFT = 0.15
_DEFAULT_FLOOR = 0.3
_DEFAULT_CEILING = 0.95
# Net (false_positive-leaning minus missed-leaning) reviews needed before the
# direction counts as real rather than noise.
_MIN_NET_SIGNAL = 2

# Insight categories each stage may see. Understanding gets pattern/process guidance;
# anomaly detection gets recall/threshold guidance.
GUIDANCE_CATEGORIES = {
    "understanding": (
        "new_fraud_patterns",
        "common_success_patterns",
        "process_improvements",
    ),
    "anomaly_detection": (
        "frequently_missed_anomalies",
        "accuracy_improvements",
        "confidence_threshold_recommendations",
    ),
}

_CATEGORY_LABELS = {
    "common_success_patterns": "Approaches that worked before",
    "frequently_missed_anomalies": "Things past runs MISSED (look for these)",
    "accuracy_improvements": "Accuracy notes",
    "confidence_threshold_recommendations": "Confidence-calibration notes",
    "recommended_action_effectiveness": "Action effectiveness",
    "new_fraud_patterns": "Fraud patterns analysts have confirmed",
    "process_improvements": "Process notes",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_jsonl(blob):
    """Parse ``JSONL`` into a list of dicts; a corrupt line is skipped, not fatal."""
    rows = []
    for line in (blob or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed feedback line")
    return rows


def _parse_records(blob, name):
    """Parse a JSON array of records; anything else yields an empty list."""
    if not blob:
        return []
    try:
        loaded = json.loads(blob)
    except Exception as e:  # noqa: BLE001 — a bad file must not break a run
        logger.warning("Could not parse %s: %s", name, e)
        return []
    if not isinstance(loaded, list):
        return []
    return [r for r in loaded if isinstance(r, dict)]


class FeedbackLoop:
    def __init__(self, llm_client, config=None, anomaly_config=None, storage=None):
        self.llm_client = llm_client
        # None is resolved per call in _backend(), keeping AFIR_DATA_DIR working and
        # the test seam (monkeypatch data_dir) available. Reviews are persisted on arrival,
        # so a test without the redirect writes into the repo root.
        self._storage = storage
        self.config = config or {}
        self.batch_size = int(self.config.get("batch_size", 10))
        self.apply_to_prompts = bool(self.config.get("apply_to_prompts", True))

        # --- numeric tuning -------------------------------------------------
        # Off by default; recommendation is still computed when off.
        self.auto_tune_threshold = bool(self.config.get("auto_tune_threshold", False))
        self.threshold_step = abs(
            float(self.config.get("threshold_step", _DEFAULT_STEP))
        )
        self.min_reviews_for_tuning = max(
            1, int(self.config.get("min_reviews_for_tuning", _DEFAULT_MIN_REVIEWS))
        )
        self.max_threshold_drift = abs(
            float(self.config.get("max_threshold_drift", _DEFAULT_MAX_DRIFT))
        )
        self.threshold_min = float(self.config.get("threshold_min", _DEFAULT_FLOOR))
        self.threshold_max = float(self.config.get("threshold_max", _DEFAULT_CEILING))
        # The operator's declared threshold: anchor for drift bounds, never overwritten.
        self.baseline_threshold = None
        if anomaly_config:
            try:
                self.baseline_threshold = float(anomaly_config.get("threshold"))
            except (TypeError, ValueError):
                self.baseline_threshold = None

        self.feedback_data = _parse_jsonl(self._backend().get_text(_PENDING_FILE))
        if self.feedback_data:
            logger.info(
                "Reloaded %d pending feedback entries from %s",
                len(self.feedback_data),
                _PENDING_FILE,
            )
        self._insights_cache = None
        self._tuning_cache = None

    # -- storage -----------------------------------------------------------

    def _backend(self):
        """The backend to read and write through. Local disk unless one was injected."""
        return self._storage if self._storage is not None else LocalStorage(data_dir())

    @staticmethod
    def _path(name):
        """On-disk path for ``name``; kept for log lines and layout tests (not the read/write path)."""
        return data_dir() / name

    # -- collect -----------------------------------------------------------

    async def collect_feedback(
        self,
        incident_id,
        investigation_result=None,
        human_feedback=None,
        **structured,
    ):
        """Record one review, persisting before distilling so the entry survives a crash.

        Extra keyword fields are validated via ``FeedbackEntry``; returns the entry as a dict.
        """
        entry = FeedbackEntry(
            incident_id=str(incident_id),
            received_at=_now(),
            human_feedback=human_feedback,
            investigation_result=investigation_result,
            **{k: v for k, v in structured.items() if v is not None},
        )
        record = entry.model_dump()
        self.feedback_data.append(record)
        await asyncio.to_thread(self._persist_entry, record)

        if len(self.feedback_data) >= self.batch_size:
            await self.process_feedback()
        return record

    def _persist_entry(self, record):
        """Append to both the permanent log and the pending batch buffer."""
        try:
            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            backend = self._backend()
            # append_text: concurrent submissions must not overwrite each other's entries.
            backend.append_text(_LOG_FILE, line)
            backend.append_text(_PENDING_FILE, line)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to persist feedback entry: %s", e)

    # -- distill -----------------------------------------------------------

    async def process_feedback(self):
        """Distill the pending batch into insights. Returns None when nothing is pending."""
        if not self.feedback_data:
            return None

        batch = list(self.feedback_data)
        messages = [
            {
                "role": "system",
                "content": (
                    "You analyze fraud-investigation feedback and recommend improvements: "
                    "common success patterns, frequently missed/misclassified anomalies, "
                    "accuracy improvements, confidence-threshold recommendations, recommended-"
                    "action effectiveness, new fraud patterns to incorporate, and overall "
                    "process improvements.\n"
                    "Your output is injected verbatim into future investigation prompts, so "
                    "each item must be a short, general, actionable instruction — not a "
                    "restatement of one incident. Never name a specific verdict as "
                    "predetermined; describe what to LOOK for, not what to conclude."
                ),
            },
            {"role": "user", "content": json.dumps(batch, indent=2, default=str)},
        ]

        insights = await self.llm_client.structured_output(
            messages, response_model=FeedbackInsights, stage="feedback_distillation"
        )
        await self.apply_insights(insights)
        # Only clear once the distillation is safely persisted.
        self.feedback_data = []
        await asyncio.to_thread(self._truncate_pending)
        await self.tune_threshold()
        return insights

    def _truncate_pending(self):
        try:
            self._backend().delete(_PENDING_FILE)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to clear pending feedback buffer: %s", e)

    async def apply_insights(self, insights: FeedbackInsights):
        """Log insights, persist them, and invalidate the guidance cache.

        ``guidance_prompt`` reads from the persisted file, so an appended insight
        changes the next run's behaviour.
        """
        logger.info("Feedback insights generated:")
        for category, recommendations in insights.model_dump().items():
            logger.info("%s:", category)
            for recommendation in recommendations or []:
                logger.info("  - %s", recommendation)

        await asyncio.to_thread(self._persist_insights, insights)
        self._insights_cache = None  # force re-read

    def _persist_insights(self, insights: FeedbackInsights):
        """Append the insights (timestamped) to feedback_insights.json under the data dir."""
        record = {
            "generated_at": _now(),
            "insights": insights.model_dump(),
        }
        try:
            backend = self._backend()
            existing = _parse_records(backend.get_text(_INSIGHTS_FILE), _INSIGHTS_FILE)
            existing.append(record)
            blob = json.dumps(existing, indent=2, ensure_ascii=False)
            if backend.put_text(_INSIGHTS_FILE, blob, verify=json_verifier):
                logger.info("Persisted feedback insights to %s", _INSIGHTS_FILE)
            else:
                logger.error(
                    "Failed to persist feedback insights to %s", _INSIGHTS_FILE
                )
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to persist feedback insights: %s", e)

    # -- read back ---------------------------------------------------------

    def load_insight_records(self, refresh=False):
        """Every persisted insight record (newest last). Cached; never raises."""
        if self._insights_cache is not None and not refresh:
            return self._insights_cache
        records = _parse_records(
            self._backend().get_text(_INSIGHTS_FILE), _INSIGHTS_FILE
        )
        self._insights_cache = records
        return records

    def merged_insights(self, refresh=False):
        """All insight records merged per category, de-duplicated, newest first.

        Later distillations reflect more recent analyst experience, so they lead.
        """
        merged = {}
        for record in reversed(self.load_insight_records(refresh=refresh)):
            data = record.get("insights")
            if not isinstance(data, dict):
                continue
            for category, items in data.items():
                if not isinstance(items, list):
                    continue
                bucket = merged.setdefault(category, [])
                for item in items:
                    text = str(item).strip()
                    if text and text not in bucket:
                        bucket.append(text)
        return merged

    def history(self, limit=50):
        """The most recent persisted reviews (newest first), for GET /api/v1/feedback."""
        rows = _parse_jsonl(self._backend().get_text(_LOG_FILE))
        rows.reverse()
        return rows[: max(0, int(limit))]

    def stats(self):
        """Counters an analyst can read at a glance."""
        rows = _parse_jsonl(self._backend().get_text(_LOG_FILE))
        agreed = sum(1 for r in rows if r.get("agrees_with_verdict") is True)
        disagreed = sum(1 for r in rows if r.get("agrees_with_verdict") is False)
        noisy, missed = self._direction_counts(rows)
        return {
            "total_reviews": len(rows),
            "pending_in_batch": len(self.feedback_data),
            "batch_size": self.batch_size,
            "distillations": len(self.load_insight_records()),
            "agreed_with_verdict": agreed,
            "disagreed_with_verdict": disagreed,
            "apply_to_prompts": self.apply_to_prompts,
            "false_positive_reports": noisy,
            "missed_anomaly_reports": missed,
            "auto_tune_threshold": self.auto_tune_threshold,
            "threshold_adjustments": len(self.load_tuning_records()),
            "effective_threshold": self.effective_threshold(),
        }

    # -- apply (the loop-closing step) -------------------------------------

    def guidance_prompt(self, stage, refresh=False):
        """Guidance for ``stage`` as a system-message body, or ``""`` when nothing applies.

        ``stage`` selects allowed categories; unknown stage yields ``""`` rather than leaking
        every category. Safe to append unconditionally.
        """
        if not self.apply_to_prompts:
            return ""
        categories = GUIDANCE_CATEGORIES.get(stage)
        if not categories:
            return ""
        merged = self.merged_insights(refresh=refresh)
        blocks = []
        for category in categories:
            items = merged.get(category) or []
            if not items:
                continue
            label = _CATEGORY_LABELS.get(category, category.replace("_", " "))
            lines = [f"{label}:"]
            lines += [f"  - {i}" for i in items[:_MAX_ITEMS_PER_CATEGORY]]
            blocks.append("\n".join(lines))
        if not blocks:
            return ""

        body = "\n".join(blocks)
        if len(body) > _MAX_GUIDANCE_CHARS:
            body = body[:_MAX_GUIDANCE_CHARS].rsplit("\n", 1)[0] + "\n  - …"
        return (
            "ANALYST FEEDBACK GUIDANCE (learned from past reviews of this system's "
            "own investigations). This is ADVISORY prior experience, NOT evidence "
            "about the incident in front of you. It ranks BELOW the knowledge pack, "
            "the official procedure, and any deterministic verdict — where it "
            "conflicts with those, they win. Use it to widen what you check for; "
            "never use it to assert a finding the retrieved data does not support.\n"
            + body
        )

    # -- apply (the numeric step) ------------------------------------------

    def load_tuning_records(self, refresh=False):
        """Every persisted threshold adjustment (oldest first). Cached; never raises."""
        if self._tuning_cache is not None and not refresh:
            return self._tuning_cache
        records = _parse_records(self._backend().get_text(_TUNING_FILE), _TUNING_FILE)
        self._tuning_cache = records
        return records

    def _baseline(self, configured=None):
        """The operator's declared threshold: explicit arg > ctor value > floor."""
        for candidate in (configured, self.baseline_threshold):
            if candidate is None:
                continue
            try:
                return float(candidate)
            except (TypeError, ValueError):
                continue
        return self.threshold_min

    def effective_threshold(self, configured=None, refresh=False):
        """The threshold in effect: tuned value if auto-tuning is on, else the configured baseline. Never raises."""
        baseline = self._baseline(configured)
        if not self.auto_tune_threshold:
            return baseline
        try:
            records = self.load_tuning_records(refresh=refresh)
            if not records:
                return baseline
            value = records[-1].get("threshold")
            if value is None:
                return baseline
            # Re-clamp on every read so an operator config change takes effect immediately.
            return self._clamp(float(value), baseline)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not read tuned threshold (%s); using configured %.2f.",
                e,
                baseline,
            )
            return baseline

    def _clamp(self, value, baseline):
        """Confine a threshold to [floor, ceiling] and within max_drift of baseline."""
        low = max(self.threshold_min, baseline - self.max_threshold_drift)
        high = min(self.threshold_max, baseline + self.max_threshold_drift)
        if low > high:  # a pathological config; the baseline always wins
            return baseline
        return round(min(high, max(low, value)), 4)

    @staticmethod
    def _direction_counts(rows):
        """Count FP-heavy vs missed-heavy reviews; only structured list fields counted (never keyword-scanned)."""
        noisy = missed = 0
        for r in rows:
            fps = r.get("false_positives")
            miss = r.get("missed_anomalies")
            has_fp = isinstance(fps, list) and len(fps) > 0
            has_miss = isinstance(miss, list) and len(miss) > 0
            # Both signals means the ranking is wrong, not the cut-off; cancel out.
            if has_fp and not has_miss:
                noisy += 1
            elif has_miss and not has_fp:
                missed += 1
        return noisy, missed

    def threshold_recommendation(self, configured=None, refresh=False):
        """Compute the recommended threshold without applying it. Always returns a dict.

        Keys: ``baseline``, ``current``, ``recommended``, ``change``, ``direction``,
        ``reason``, ``applied``, ``reviews_considered``, ``false_positive_reports``,
        ``missed_anomaly_reports``, ``adjustments``.
        """
        baseline = self._baseline(configured)
        current = self.effective_threshold(configured, refresh=refresh)
        records = self.load_tuning_records()
        # Count only reviews since the last adjustment; the same reviews must not
        # push the threshold repeatedly.
        watermark = 0
        if records:
            try:
                watermark = int(records[-1].get("reviews_at_adjustment", 0))
            except (TypeError, ValueError):
                watermark = 0
        all_rows = _parse_jsonl(self._backend().get_text(_LOG_FILE))
        window = all_rows[watermark:]
        noisy, missed = self._direction_counts(window)

        out = {
            "baseline": baseline,
            "current": current,
            "recommended": current,
            "change": 0.0,
            "direction": "hold",
            "applied": self.auto_tune_threshold,
            "reviews_considered": len(window),
            "false_positive_reports": noisy,
            "missed_anomaly_reports": missed,
            "adjustments": len(records),
            "step": self.threshold_step,
            "bounds": {
                "min": max(self.threshold_min, baseline - self.max_threshold_drift),
                "max": min(self.threshold_max, baseline + self.max_threshold_drift),
            },
            "reason": "",
        }

        if len(window) < self.min_reviews_for_tuning:
            out["reason"] = (
                f"Not enough new reviews: {len(window)} of "
                f"{self.min_reviews_for_tuning} required since the last adjustment."
            )
            return out

        net = noisy - missed
        if abs(net) < _MIN_NET_SIGNAL:
            out["reason"] = (
                f"No clear direction ({noisy} noise-leaning vs {missed} "
                "missed-leaning reviews); the threshold is probably not the problem."
            )
            return out

        direction = "raise" if net > 0 else "lower"
        proposed = current + (self.threshold_step if net > 0 else -self.threshold_step)
        clamped = self._clamp(proposed, baseline)
        out["direction"] = direction
        out["recommended"] = clamped
        out["change"] = round(clamped - current, 4)
        if clamped == current:
            out["direction"] = "hold"
            out["reason"] = (
                f"Would {direction} the threshold, but it is already at the bound "
                f"({clamped:.2f}) permitted by the configured baseline "
                f"{baseline:.2f} +/- {self.max_threshold_drift:.2f}."
            )
        else:
            out["reason"] = (
                f"{noisy} review(s) reported false positives and {missed} reported "
                f"missed anomalies over {len(window)} new review(s); {direction} the "
                f"confidence threshold from {current:.2f} to {clamped:.2f}."
            )
        return out

    async def tune_threshold(self, configured=None):
        """Apply the recommendation if auto-tuning is on and the change is real.

        Returns the dict either way; ``persisted`` says whether anything was written.
        Never raises.
        """
        try:
            rec = self.threshold_recommendation(configured, refresh=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("Threshold recommendation failed: %s", e)
            return {"persisted": False, "reason": f"recommendation failed: {e}"}

        rec["persisted"] = False
        if not self.auto_tune_threshold:
            rec["reason"] = (
                "Auto-tuning is disabled (feedback.auto_tune_threshold=false); "
                "recommendation computed but not applied. " + rec["reason"]
            )
            return rec
        if rec["direction"] == "hold" or rec["recommended"] == rec["current"]:
            return rec

        total_reviews = len(_parse_jsonl(self._backend().get_text(_LOG_FILE)))
        record = {
            "adjusted_at": _now(),
            "threshold": rec["recommended"],
            "previous": rec["current"],
            "baseline": rec["baseline"],
            "direction": rec["direction"],
            "reason": rec["reason"],
            "reviews_at_adjustment": total_reviews,
            "false_positive_reports": rec["false_positive_reports"],
            "missed_anomaly_reports": rec["missed_anomaly_reports"],
        }
        try:
            await asyncio.to_thread(self._persist_tuning, record)
        except Exception as e:  # noqa: BLE001
            logger.error("Could not persist threshold adjustment: %s", e)
            return rec
        rec["persisted"] = True
        logger.info(
            "Anomaly confidence threshold auto-tuned %.2f -> %.2f (%s). %s",
            rec["current"],
            rec["recommended"],
            rec["direction"],
            rec["reason"],
        )
        return rec

    def _persist_tuning(self, record):
        """Append one adjustment to feedback_thresholds.json (full history kept)."""
        backend = self._backend()
        existing = _parse_records(backend.get_text(_TUNING_FILE), _TUNING_FILE)
        existing.append(record)
        blob = json.dumps(existing, indent=2, ensure_ascii=False)
        if not backend.put_text(_TUNING_FILE, blob, verify=json_verifier):
            # Don't update the cache on a failed write; a stale cache silently vanishes at restart.
            logger.error("Failed to persist the tuned threshold to %s", _TUNING_FILE)
            return
        self._tuning_cache = existing

    def reset_threshold(self):
        """Discard the tuning history so the configured baseline applies again."""
        try:
            self._backend().delete(_TUNING_FILE)
        except Exception as e:  # noqa: BLE001
            logger.error("Could not reset tuned threshold: %s", e)
            return False
        self._tuning_cache = None
        logger.info(
            "Threshold tuning history cleared; configured baseline is in effect."
        )
        return True
