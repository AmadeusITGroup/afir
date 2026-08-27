import logging
import re

from human_guidance import guidance_message
from models.pydantic_models import IncidentAnalysis, UnderstandingResult
from utils.error_handling import async_retry_with_backoff

logger = logging.getLogger(__name__)

# Time expressions that could plausibly have produced a parsed event window. Inclusive
# by design: clearing a stated window is worse than keeping an invented one.
_TIME_EXPRESSION_RE = re.compile(
    r"""(
        \d{4}-\d{1,2}-\d{1,2}                    # 2026-08-17
      | \d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}          # 17/08/2026, 8.17.26, 17-08-2026
      | \d{1,2}:\d{2}                            # 14:05
      | \b\d{10,13}\b                            # an epoch instant
      | \b(19|20)\d{2}\b                         # a bare year
      # Full month names stand alone; abbreviations and "may" require an adjacent digit
      # (``(jan|...)[a-z]*`` matched ordinary English verbs on every real incident).
      | \b(january|february|march|april|june|july|august|september|october|november|december)\b
      | \b\d{1,2}(st|nd|rd|th)?\s+
        (jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)\.?\b
      | \b(jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)\.?\s+\d{1,4}\b
      | \b(mon|tue|wed|thu|fri|sat|sun)[a-z]*day\b
      | \b(today|yesterday|tomorrow|tonight|overnight|weekend)\b
      | \b(this|last|past|previous|recent|coming)\s+
        (\d+\s+)?(hour|day|night|week|weekend|month|quarter|year|morning|afternoon|evening)s?\b
      | \b\d+\s+(second|minute|hour|day|week|month|year)s?\s+ago\b
      | \bsince\b
      | \bbetween\s+\d
    )""",
    re.IGNORECASE | re.VERBOSE,
)


# Request-phrasing words exempted from the presence test. Missing words make the check
# stricter, which is the safe direction.
_ASK_FILLER = frozenset(
    """
    also check verify confirm whether there this that these those been being with without
    from into onto need needs needed want wants please kindly ensure make sure look looking
    review reviewed investigate investigated case incident report reported alert alerted
    fraud fraudulent abuse abusive suspicious activity related possible possibly potential
    """.split()
)

_ASK_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _is_traceable_ask(ask: str, text: str) -> bool:
    """True when every substantive token of ``ask`` appears in ``text`` (at least one required).
    ``all`` not ``any``: a fabricated ask must fabricate nothing. ``\\b`` guards ``_``-ids."""
    substantive = [t for t in _ASK_TOKEN_RE.findall(ask.lower()) if t not in _ASK_FILLER]
    if not substantive:
        return False
    return all(re.search(rf"\b{re.escape(token)}\b", text) for token in substantive)


class IncidentUnderstandingModule:
    def __init__(self, llm_client, rag=None, knowledge_pack=None, feedback=None):
        self.llm_client = llm_client
        self.rag = rag
        self.knowledge_pack = knowledge_pack
        self.feedback = feedback
        self.system_prompt = (
            "You are a fraud investigation analyst. Analyze the incident and produce a "
            "structured analysis. Set `severity` to the severity the INCIDENT ITSELF "
            "states, copied verbatim (e.g. '2', 'P2', 'HIGH'), and leave it EMPTY if the "
            "incident states none — do NOT invent a severity or a numeric scale of your "
            "own; use `severity_reasoning` for how serious the described activity appears "
            "and why. Identify the concrete log sources and "
            "systems worth reviewing, plausible hypotheses (including known fraud "
            "patterns), immediate recommended actions, and which stakeholders to notify. "
            "Relate the incident to similar past incidents when context is available.\n"
            "Extract every domain entity present in the incident into extracted_entities "
            "(set `type` to the entity type, `value` to the normalized identifier, `raw` "
            "to the original text). Parse the actual time the incident occurred into "
            "event_time (start/end, ISO 8601) from the text itself — do NOT use the "
            "ingestion timestamp; if a single instant is given, set start and end around "
            "it. Leave event_time null only if the text gives no time.\n"
            "Identify the correlation_keys: the entity types the different log sources "
            "should be joined/correlated on to investigate this incident. Use ONLY entity "
            "types from the domain glossary supplied to you. These are the identifiers "
            "expected to appear across multiple sources and link their records together. "
            "Only include an entity type as a correlation_key if the SAME value is expected "
            "to appear in more than one log source. Do NOT use an alert / monitoring "
            "document id (an id that names one alert record, such as a monitoring `_id` or "
            "`recordId`) as a correlation_key: it identifies a document in the alerting "
            "store and does not appear in the transactional systems of record. The "
            "event_time you parsed also bounds the time range the correlated data should "
            "fall within. List the most investigation-relevant entity types first.\n"
            "If — and ONLY if — the incident text explicitly asks for another kind of fraud "
            "to also be checked ('also confirm whether ...', 'please verify there was no "
            "...'), record that ask in requested_links, in the text's own words, one entry "
            "per ask. Leave it empty otherwise: an ask you infer is not an ask. Do NOT "
            "repeat it into incident_summary, initial_hypotheses, key_investigation_areas "
            "or any other field — the summary is read to decide which single procedure "
            "adjudicates THIS incident, and a second kind of fraud named there costs the "
            "run its own."
        )
        self.user_template = (
            "Incident ID: {incident_id}\n"
            "Ingestion timestamp (NOT the event time): {timestamp}\n"
            "Description: {description}"
        )

    def _glossary_message(self):
        """A system message listing the domain entities to extract, if a pack is set."""
        if self.knowledge_pack is None:
            return None
        glossary = self.knowledge_pack.glossary_prompt()
        if not glossary:
            return None
        return {"role": "system", "content": glossary}

    def _feedback_message(self):
        """A system message carrying learned analyst guidance, if any. Never raises."""
        if self.feedback is None:
            return None
        try:
            guidance = self.feedback.guidance_prompt("understanding")
        except Exception as e:  # noqa: BLE001 — guidance is advisory, never fatal
            logger.warning("Could not render feedback guidance: %s", e)
            return None
        if not guidance:
            return None
        return {"role": "system", "content": guidance}

    def _validate_entities(self, analysis, incident_id):
        """Log mismatches against glossary value_patterns; never drop an entity."""
        if self.knowledge_pack is None:
            return
        patterns = self.knowledge_pack.value_patterns()
        if not patterns:
            return
        for ent in getattr(analysis, "extracted_entities", []) or []:
            pattern = patterns.get(ent.type)
            if not pattern or not ent.value:
                continue
            try:
                if not re.fullmatch(pattern, ent.value):
                    logger.warning(
                        "Incident %s: entity %s value %r does not match pattern %r",
                        incident_id,
                        ent.type,
                        ent.value,
                        pattern,
                    )
            except re.error as e:
                logger.warning(
                    "Invalid value_pattern %r for entity %s: %s",
                    pattern,
                    ent.type,
                    e,
                )

    def _classify_value_forms(self, analysis, incident_id):
        """Stamp each entity with the declared surface form, overwriting the LLM value."""
        if self.knowledge_pack is None:
            return
        counts = {}
        for ent in getattr(analysis, "extracted_entities", []) or []:
            try:
                form = self.knowledge_pack.classify_value_form(ent.type, ent.value)
            except Exception as e:  # noqa: BLE001 — classification is never fatal
                logger.warning("Value-form classification failed for %s: %s", ent.type, e)
                continue
            ent.value_form = form or ""
            if form:
                counts[f"{ent.type}.{form}"] = counts.get(f"{ent.type}.{form}", 0) + 1
            elif self.knowledge_pack.value_forms_for(ent.type):
                # Declared forms but no match: the value won't be form-scoped to a field.
                logger.warning(
                    "Incident %s: entity %s value %r matches none of the declared "
                    "value_forms; it will not be form-scoped to a field.",
                    incident_id,
                    ent.type,
                    ent.value,
                )
        if counts:
            logger.info("Incident %s: entity value forms %s", incident_id, counts)

    def _group_co_occurring_entities(self, analysis, incident):
        """Stamp co_occurrence from the incident text's per-record layout (reads text as a table).

        Tries spans first, falls back to lines. Three refusals: a repeated (type, form) means
        prose; fewer than two types is not a combination; a shape seen once is a sentence.
        Stamp is space-joined labels (a column may recur). Never fatal.
        """
        entities = [
            e
            for e in (getattr(analysis, "extracted_entities", []) or [])
            if getattr(e, "type", "") and getattr(e, "value", "")
        ]
        if len(entities) < 2:
            return
        text = "\n".join(
            str(incident.get(key) or "")
            for key in ("description", "title", "summary", "raw_text")
        )
        if not text.strip():
            return
        candidates = []
        for number, line in enumerate(text.splitlines(), start=1):
            tokens = {t for t in re.split(r"[^0-9A-Za-z_]+", line) if t}
            if not tokens:
                continue
            on_line = [e for e in entities if e.value in tokens]
            # Span reading wins when it finds two or more records on this line.
            spans = self._raw_span_records(on_line, number)
            if len(spans) >= 2:
                candidates.extend(spans)
                continue
            by_field = {}
            for ent in on_line:
                by_field.setdefault(
                    (ent.type, getattr(ent, "value_form", "") or ""), []
                ).append(ent)
            # A record names each field once, so a repeated field means this line is prose.
            if any(len(v) > 1 for v in by_field.values()):
                continue
            if len({t for t, _form in by_field}) < 2:
                continue
            candidates.append((f"L{number:06d}", tuple(sorted(by_field)), on_line))
        shapes = {}
        for _label, shape, _members in candidates:
            shapes[shape] = shapes.get(shape, 0) + 1
        # A value may name more than one record, so the stamp is a set of labels.
        stamped = {}
        groups = 0
        for label, shape, members in candidates:
            if shapes[shape] < 2:
                continue  # a shape of one is a sentence, not a row of a table
            groups += 1
            for ent in members:
                # Zero-padded for natural sort order; appended in incident order.
                stamped.setdefault(id(ent), []).append(label)
        for ent in entities:
            labels = stamped.get(id(ent))
            if not labels:
                continue
            try:
                ent.co_occurrence = " ".join(labels)
            except (AttributeError, ValueError):  # noqa: BLE001 — a mock, or not our model
                return
        if groups:
            logger.info(
                "Incident %s: %d per-record entity combination(s) recovered from the "
                "incident text's layout; queries will ask for those combinations rather "
                "than the cross product of the per-type value lists.",
                incident.get("id"),
                groups,
            )

    @staticmethod
    def _raw_span_records(on_line, number):
        """Records on one line keyed by each entity's raw span.

        Same two per-span refusals as the line reading (repeated field → prose; <2 types).
        The third refusal (shape once) is applied by the caller. Preferred when ≥2 spans found.
        """
        by_span = {}
        for ent in on_line:
            raw = str(getattr(ent, "raw", "") or "").strip()
            if not raw:
                continue
            # Whole token match: an identifier is routinely a substring of others.
            tokens = {t for t in re.split(r"[^0-9A-Za-z_]+", raw) if t}
            if ent.value not in tokens:
                continue
            by_span.setdefault(raw, []).append(ent)
        out = []
        for index, (_raw, members) in enumerate(by_span.items(), start=1):
            by_field = {}
            for ent in members:
                by_field.setdefault(
                    (ent.type, getattr(ent, "value_form", "") or ""), []
                ).append(ent)
            if any(len(v) > 1 for v in by_field.values()):
                continue
            if len({t for t, _form in by_field}) < 2:
                continue
            out.append((f"L{number:06d}S{index:03d}", tuple(sorted(by_field)), members))
        return out

    def _drop_invented_event_window(self, analysis, incident):
        """Clear event_time when the incident text states no time; the cleared window is replaced
        by the declared default lookup depth. Conservative: any plausible time expression keeps it."""
        window = getattr(analysis, "event_time", None)
        if window is None:
            return
        text = " ".join(
            str(incident.get(key) or "")
            for key in ("description", "title", "summary", "raw_text")
        )
        if _TIME_EXPRESSION_RE.search(text):
            return
        logger.warning(
            "Incident %s: event_time %s → %s was parsed from a description that contains no "
            "date, time or relative time expression, so it cannot have come from the incident "
            "— discarding it. The retrieval window will come from the declared default lookup "
            "depth instead (pack retrieval.default_lookup_days, else "
            "log_sources.default_lookup_days).",
            incident.get("id"),
            getattr(window, "start", "?"),
            getattr(window, "end", "?"),
        )
        try:
            analysis.event_time = None
        except (AttributeError, ValueError):  # noqa: BLE001 — a mock, or not our model
            logger.warning("Could not clear the invented event_time; leaving it in place.")

    def _confine_requested_links(self, analysis, incident):
        """Drop requested links not traceable to the incident text; requested_links bypasses
        signal thresholds, so a fabricated one would state a referral nobody requested. Warns
        when a kept ask echoes in incident_summary (which selects the adjudicating procedure)."""
        asks = getattr(analysis, "requested_links", None)
        if not isinstance(asks, list) or not asks:
            return
        text = " ".join(
            str(incident.get(key) or "")
            for key in ("description", "title", "summary", "raw_text")
        ).lower()
        kept, dropped = [], []
        for raw in asks:
            ask = str(raw or "").strip()
            if ask and _is_traceable_ask(ask, text):
                kept.append(ask)
            elif ask:
                dropped.append(ask)
        if dropped:
            logger.warning(
                "Incident %s: %d requested link(s) name nothing the incident text says and "
                "are DISCARDED (%s). An explicit request is honoured without any declared "
                "signal behind it, so one the text does not make would state a referral "
                "nobody asked for.",
                incident.get("id"),
                len(dropped),
                "; ".join(repr(d) for d in dropped),
            )
        if kept != asks:
            try:
                analysis.requested_links = kept
            except (AttributeError, ValueError):  # noqa: BLE001 — a mock, or not our model
                logger.warning("Could not rewrite requested_links; leaving it in place.")
                return
        summary = str(getattr(analysis, "incident_summary", "") or "").lower()
        echoed = [a for a in kept if len(a) >= 12 and a.lower() in summary]
        if echoed:
            logger.warning(
                "Incident %s: the requested link(s) %s are repeated VERBATIM in "
                "incident_summary. That summary is the only text scored to decide which "
                "single procedure adjudicates this incident, so a second kind of fraud named "
                "there can move the selection — read the chosen procedure before trusting "
                "the verdict. The request itself is unaffected: it rides in its own field.",
                incident.get("id"),
                "; ".join(repr(e) for e in echoed),
            )
        if kept:
            logger.info(
                "Incident %s: %d link(s) explicitly requested by the reporter (%s). Each is "
                "assessed after correlation whatever its declared strength, and one this pack "
                "cannot serve is reported as such rather than silently producing nothing.",
                incident.get("id"),
                len(kept),
                "; ".join(kept),
            )

    def _pin_requested_use_case(self, analysis, incident):
        """Overwrite pinned_use_case with the incident's link_pin, enforcing the LLM instruction
        to leave it empty. An unknown value is treated as no pin by select_correlation_spec."""
        pin = str((incident or {}).get("link_pin", "") or "").strip()
        claimed = str(getattr(analysis, "pinned_use_case", "") or "").strip()
        try:
            analysis.pinned_use_case = pin
        except (AttributeError, ValueError):  # noqa: BLE001 — a mock, or not our model
            logger.warning(
                "Could not stamp the referral pin; leaving the field as it came."
            )
            return
        if claimed and claimed != pin:
            logger.warning(
                "Incident %s: the understanding stage returned a procedure pin (%r) that no "
                "caller asked for; it is DISCARDED. Which procedure adjudicates is decided by "
                "the incident and, for a referral, by the referring run — never by the stage "
                "that reads the text.",
                (incident or {}).get("id"),
                claimed,
            )
        if pin:
            logger.info(
                "Incident %s is a referral pinned to procedure '%s'; its ruleset, its join keys "
                "and its required sources all come from that pin rather than from scoring its "
                "description.",
                (incident or {}).get("id"),
                pin,
            )

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def process(self, incident, guidance=None):
        """Analyse the incident. ``guidance`` is analyst direction from a rejected gate."""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_template.format(
                        incident_id=incident["id"],
                        timestamp=incident["timestamp"],
                        description=incident["description"],
                    ),
                },
            ]
            glossary_msg = self._glossary_message()
            if glossary_msg is not None:
                messages.insert(1, glossary_msg)
            # After glossary, before user content.
            feedback_msg = self._feedback_message()
            if feedback_msg is not None:
                messages.insert(len(messages) - 1, feedback_msg)
            # After feedback, before incident text.
            human_msg = guidance_message(guidance)
            if human_msg is not None:
                messages.insert(len(messages) - 1, human_msg)

            analysis = await self.llm_client.structured_output(
                messages,
                response_model=IncidentAnalysis,
                rag=self.rag,
                stage="incident_understanding",
            )

            self._validate_entities(analysis, incident["id"])
            self._classify_value_forms(analysis, incident["id"])
            # After form classification: groups carry forms.
            self._group_co_occurring_entities(analysis, incident)
            self._drop_invented_event_window(analysis, incident)
            self._confine_requested_links(analysis, incident)
            self._pin_requested_use_case(analysis, incident)

            logger.info(f"Processed understanding for incident {incident['id']}")
            return UnderstandingResult(
                incident_id=incident["id"],
                analysis=analysis,
                incident_timestamp=str(incident.get("timestamp") or ""),
            )
        except Exception as e:
            logger.error(f"Error processing incident {incident['id']}: {str(e)}")
            raise
